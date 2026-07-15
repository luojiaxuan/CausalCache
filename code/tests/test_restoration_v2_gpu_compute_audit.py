from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import scripts.audit_restoration_v2_gpu_compute as audit_module
from causalcache.restoration_v2_gpu_kl import (
    cpu_full_vocab_mean_kl_oracle_for_tests,
)
from scripts.audit_restoration_v2_gpu_compute import (
    OUTCOME,
    _audit_planner,
    _equivalence_record,
    _gpu_runtime_identity,
    _invalid_numeric_distance_record,
    _nvidia_smi_identity,
    _require_cuda_runtime,
    _source_file_inventory,
    _synthetic_inputs,
    _validate_cli_identity,
    _validate_primitive_audits,
    _validate_repository_state,
    _write_exclusive,
    main,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _valid_cli_args(output: Path) -> list[str]:
    return [
        "--device",
        "cuda:0",
        "--run-git-commit",
        "1" * 40,
        "--container-image-digest",
        f"sha256:{'2' * 64}",
        "--host-alias",
        "hyper00",
        "--host-hostname",
        "node-radixark-16-0000",
        "--container-id",
        "3" * 64,
        "--output-summary",
        str(output),
    ]


class RestorationV2GPUComputeAuditIdentityTest(unittest.TestCase):
    def _namespace(self, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "device": "cuda:0",
            "run_git_commit": "1" * 40,
            "container_image_digest": f"sha256:{'2' * 64}",
            "host_alias": "hyper00",
            "host_hostname": "node-radixark-16-0000",
            "container_id": "3" * 64,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_cli_identity_requires_explicit_cuda_and_full_lowercase_hashes(self) -> None:
        self.assertEqual(_validate_cli_identity(self._namespace()), 0)
        self.assertEqual(
            _validate_cli_identity(self._namespace(device="cuda:17")),
            17,
        )
        invalid = (
            ({"device": "cuda"}, "explicit"),
            ({"device": "cpu"}, "explicit"),
            ({"device": "cuda:-1"}, "explicit"),
            ({"run_git_commit": "1" * 39}, "40-hex"),
            ({"run_git_commit": "A" * 40}, "40-hex"),
            ({"container_image_digest": "2" * 64}, "sha256"),
            (
                {"container_image_digest": f"sha256:{'A' * 64}"},
                "lowercase",
            ),
            ({"host_alias": "  "}, "host-alias"),
            ({"host_hostname": "node name"}, "host-hostname"),
            ({"container_id": "3" * 63}, "container-id"),
            ({"container_id": "A" * 64}, "container-id"),
        )
        for override, pattern in invalid:
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, pattern):
                    _validate_cli_identity(self._namespace(**override))

    def test_clean_repository_must_match_exact_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _git(root, "init", "--quiet")
            fixture = root / "fixture.txt"
            fixture.write_text("pinned\n", encoding="utf-8")
            _git(root, "add", "fixture.txt")
            _git(
                root,
                "-c",
                "user.name=CausalCache Test",
                "-c",
                "user.email=causalcache-test@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            )
            head = _git(root, "rev-parse", "HEAD")
            result = _validate_repository_state(root, run_git_commit=head)
            self.assertEqual(result["run_git_commit"], head)
            self.assertIs(result["worktree_clean"], True)

            with self.assertRaisesRegex(ValueError, "exactly equal"):
                _validate_repository_state(root, run_git_commit="0" * 40)
            fixture.write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clean Git worktree"):
                _validate_repository_state(root, run_git_commit=head)

    def test_source_inventory_hashes_checked_out_primitives_and_audit_cli(self) -> None:
        inventory = _source_file_inventory(REPOSITORY_ROOT)
        self.assertEqual(
            set(inventory),
            {
                "code/causalcache/restoration_v2_gpu_kl.py",
                "code/causalcache/restoration_v2_batching.py",
                "code/scripts/audit_restoration_v2_gpu_compute.py",
            },
        )
        for record in inventory.values():
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(record["size_bytes"], 0)
        with mock.patch.object(
            audit_module,
            "__file__",
            "/tmp/not-the-checked-out-audit.py",
        ):
            with self.assertRaisesRegex(ValueError, "audit CLI"):
                _source_file_inventory(REPOSITORY_ROOT)


class RestorationV2GPUComputeAuditProtocolTest(unittest.TestCase):
    @staticmethod
    def _fake_result(
        *,
        batch_stride: int,
        zero_copy: bool,
        reference_compute_batch_size: int = 1,
        scalar_host_reads: int = 0,
        full_tensor_host_transfers: int = 0,
        numeric_validation: str = "gpu_resident_per_example_predicates",
        invalid_numeric_output: str = "nan_final_distance",
    ) -> SimpleNamespace:
        audit_record = {
            "reference_batch_stride": batch_stride,
            "reference_zero_copy_batch_expansion": zero_copy,
            "reference_compute_batch_size": reference_compute_batch_size,
            "validation_scalar_host_reads": scalar_host_reads,
            "full_tensor_host_transfers": full_tensor_host_transfers,
            "numeric_validation": numeric_validation,
            "invalid_numeric_output": invalid_numeric_output,
        }
        audit = SimpleNamespace(
            **audit_record,
            to_dict=lambda: dict(audit_record),
        )
        return SimpleNamespace(audit=audit)

    def test_synthetic_fixture_is_rng_free_and_cpu_oracle_compatible(self) -> None:
        reference, candidate_a, candidate_b = _synthetic_inputs()
        self.assertEqual((len(reference), len(reference[0])), (4, 5))
        for row in reference:
            self.assertAlmostEqual(sum(math.exp(value) for value in row), 1.0)
        values = cpu_full_vocab_mean_kl_oracle_for_tests(
            [reference, reference],
            [candidate_a, candidate_b],
            candidate_representation="logits",
        )
        self.assertEqual(len(values), 2)
        self.assertTrue(all(math.isfinite(value) and value > 0 for value in values))

    def test_equivalence_record_uses_frozen_absolute_and_relative_tolerance(self) -> None:
        passing = _equivalence_record(
            actual=1.000005,
            expected=1.0,
            comparison="synthetic-pass",
        )
        self.assertIs(passing["passed"], True)
        self.assertEqual(passing["atol"], 1e-6)
        self.assertEqual(passing["rtol"], 1e-5)
        with self.assertRaisesRegex(ValueError, "equivalence failed"):
            _equivalence_record(
                actual=1.001,
                expected=1.0,
                comparison="synthetic-fail",
            )

    def test_planner_audit_freezes_batch_two_and_forbids_oom_fallback(self) -> None:
        result = _audit_planner()
        self.assertEqual(result["status"], "passed")
        self.assertIs(result["automatic_oom_fallback"], False)
        self.assertEqual(result["planning_scope"], "single_decision_state")
        planner = result["planner_audit"]
        self.assertEqual(planner["microbatch_size"], 2)
        self.assertIs(planner["automatic_oom_fallback"], False)
        self.assertEqual(result["observed_microbatches"][0]["size"], 2)

    def test_cuda_requirement_fails_closed_without_importing_torch(self) -> None:
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )
        with self.assertRaisesRegex(RuntimeError, "requires available CUDA"):
            _require_cuda_runtime(fake_torch, device="cuda:0", device_index=0)

    def test_invalid_numeric_audit_serializes_nan_as_a_string(self) -> None:
        record = _invalid_numeric_distance_record(math.nan)
        self.assertIs(record["final_distance_is_nan"], True)
        self.assertEqual(record["serialized_final_distance"], "nan")
        self.assertEqual(record["host_read_scope"], "final_distance_scalar_only")
        json.dumps(record, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "did not become NaN"):
            _invalid_numeric_distance_record(0.0)

    def test_nvidia_smi_and_runtime_identity_record_driver_and_platform(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout=(
                "0, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, 570.172.08\n"
                "1, GPU-ffffffff-1111-2222-3333-444444444444, 570.172.08\n"
            ),
            stderr="",
        )
        with mock.patch(
            "scripts.audit_restoration_v2_gpu_compute.subprocess.run",
            return_value=completed,
        ):
            observed = _nvidia_smi_identity(1)
        self.assertEqual(
            observed,
            {
                "gpu_uuid": "GPU-ffffffff-1111-2222-3333-444444444444",
                "driver_version": "570.172.08",
            },
        )
        with mock.patch(
            "scripts.audit_restoration_v2_gpu_compute.subprocess.run",
            return_value=completed,
        ):
            self.assertEqual(
                _nvidia_smi_identity(
                    0,
                    preferred_gpu_uuid=(
                        "GPU-ffffffff-1111-2222-3333-444444444444"
                    ),
                )["gpu_uuid"],
                "GPU-ffffffff-1111-2222-3333-444444444444",
            )
        remapped = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="7, GPU-remapped, 570.172.08\n",
            stderr="",
        )
        with mock.patch(
            "scripts.audit_restoration_v2_gpu_compute.subprocess.run",
            return_value=remapped,
        ):
            self.assertEqual(
                _nvidia_smi_identity(0),
                {
                    "gpu_uuid": "GPU-remapped",
                    "driver_version": "570.172.08",
                },
            )

        properties = SimpleNamespace(
            uuid=None,
            name="NVIDIA H200",
            total_memory=141_000_000_000,
            major=9,
            minor=0,
            multi_processor_count=132,
        )
        fake_torch = SimpleNamespace(
            __version__="2.11.0+cu130",
            version=SimpleNamespace(cuda="13.0"),
            backends=SimpleNamespace(
                cudnn=SimpleNamespace(version=lambda: 91500),
            ),
            cuda=SimpleNamespace(
                get_device_properties=lambda device: properties,
                device_count=lambda: 1,
            ),
        )
        with mock.patch(
            "scripts.audit_restoration_v2_gpu_compute._nvidia_smi_identity",
            return_value=observed,
        ):
            identity = _gpu_runtime_identity(
                fake_torch,
                device="cuda:0",
                device_index=0,
            )
        self.assertEqual(identity["nvidia_driver_version"], "570.172.08")
        self.assertEqual(identity["gpu_uuid"], observed["gpu_uuid"])
        self.assertTrue(identity["python_version"])
        self.assertTrue(identity["platform_machine"])
        with mock.patch(
            "scripts.audit_restoration_v2_gpu_compute._nvidia_smi_identity",
            return_value={"gpu_uuid": None, "driver_version": None},
        ):
            with self.assertRaisesRegex(RuntimeError, "driver version"):
                _gpu_runtime_identity(
                    fake_torch,
                    device="cuda:0",
                    device_index=0,
                )

    def test_primitive_audit_forbids_kernel_host_reads_and_full_transfers(self) -> None:
        first = self._fake_result(batch_stride=20, zero_copy=False)
        second = self._fake_result(batch_stride=20, zero_copy=False)
        batch = self._fake_result(batch_stride=0, zero_copy=True)
        validated = _validate_primitive_audits(first, second, batch)
        self.assertEqual(validated["reference_compute_batch_size"], 1)
        self.assertEqual(validated["validation_scalar_host_reads"], 0)
        self.assertEqual(validated["full_tensor_host_transfers"], 0)

        invalid = (
            (
                self._fake_result(
                    batch_stride=20,
                    zero_copy=False,
                    scalar_host_reads=1,
                ),
                second,
                batch,
                "scalar host read",
            ),
            (
                first,
                second,
                self._fake_result(
                    batch_stride=0,
                    zero_copy=True,
                    full_tensor_host_transfers=1,
                ),
                "full-tensor host transfer",
            ),
            (
                first,
                second,
                self._fake_result(
                    batch_stride=0,
                    zero_copy=True,
                    reference_compute_batch_size=2,
                ),
                "reuse one reference",
            ),
            (
                self._fake_result(
                    batch_stride=20,
                    zero_copy=False,
                    numeric_validation="cpu_scalar_reads",
                ),
                second,
                batch,
                "numeric validation contract",
            ),
            (
                first,
                second,
                self._fake_result(
                    batch_stride=0,
                    zero_copy=True,
                    invalid_numeric_output="exception_after_host_read",
                ),
                "invalid-numeric output contract",
            ),
        )
        for result_a, result_b, batch_result, pattern in invalid:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    _validate_primitive_audits(result_a, result_b, batch_result)


class RestorationV2GPUComputeAuditOutputTest(unittest.TestCase):
    def test_exclusive_write_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            _write_exclusive(path, {"outcome": OUTCOME})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                _write_exclusive(path, {"outcome": "DRIFTED"})
            self.assertEqual(path.read_bytes(), original)
            invalid_path = Path(directory) / "invalid.json"
            with self.assertRaises(ValueError):
                _write_exclusive(invalid_path, {"distance": math.nan})
            self.assertFalse(invalid_path.exists())

    def test_existing_output_fails_before_audit_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            output.write_text("sealed\n", encoding="utf-8")
            with mock.patch(
                "scripts.audit_restoration_v2_gpu_compute._execute_audit"
            ) as execute:
                with self.assertRaises(FileExistsError):
                    main(_valid_cli_args(output))
            execute.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "sealed\n")

    def test_audit_failure_leaves_no_partial_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            with mock.patch(
                "scripts.audit_restoration_v2_gpu_compute._execute_audit",
                side_effect=RuntimeError("synthetic CUDA failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic CUDA failure"):
                    main(_valid_cli_args(output))
            self.assertFalse(output.exists())

    def test_success_writes_exact_programmatic_argv_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            tokens = _valid_cli_args(output)

            def execute(
                args: argparse.Namespace,
                *,
                exact_argv: list[str],
            ) -> dict[str, object]:
                self.assertEqual(exact_argv[1:], tokens)
                self.assertEqual(args.output_summary, output)
                return {
                    "outcome": OUTCOME,
                    "argv": exact_argv,
                    "policy_loaded": False,
                    "policy_output_generated": False,
                    "restoration_output_generated": False,
                }

            with mock.patch(
                "scripts.audit_restoration_v2_gpu_compute._execute_audit",
                side_effect=execute,
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(tokens), 0)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["outcome"], OUTCOME)
            self.assertEqual(summary["argv"][1:], tokens)
            self.assertIs(summary["policy_loaded"], False)


if __name__ == "__main__":
    unittest.main()
