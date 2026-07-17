from __future__ import annotations

import copy
import contextlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import causalcache.restoration_v2_2_expansion_labels_scientific_repair_runner as runner
from causalcache.restoration_v2_2_expansion_labels_invalid_forensic import (
    InvalidForensicEvidence,
    canonical_json_bytes,
    sha256_bytes,
)
from causalcache.restoration_v2_2_expansion_labels_scientific_repair import (
    FORMAL_EXTERNAL_REPLAY_TIER,
    PASS_STATUS as CORE_PASS_STATUS,
)
from causalcache.restoration_v2_2_expansion_labels_scientific_repair_runner import (
    AUDIT_STATUS,
    FROZEN_CONFIG_SHA256,
    IMPORT_GUARD_MARKER,
    MANIFEST_STATUS,
    TEST_ONLY_STATUS,
    BoundStateSnapshot,
    OriginalInvalidSnapshot,
    ScientificRepairRunnerContract,
    SourceIdentity,
    atomic_publish_repaired_archive,
    build_completion_record,
    build_test_only_repaired_archive,
    exclusive_or_identical_state,
    inspect_repair_state,
    load_frozen_runner_contract,
    read_repaired_archive,
    repaired_archive_from_files,
    require_original_snapshot_unchanged,
    validate_import_guard,
    validate_no_gpu_launch_receipt,
)
from scripts.run_restoration_v2_2_expansion_labels_scientific_repair import (
    BLOCKED_IMPORT_ROOTS,
    NoModelFrameworkImportGuard,
    _read_hf_token,
    install_import_guard,
    remove_import_guard,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "code/configs/causalcache_restoration_v2_2_"
    "expansion_labels_scientific_repair_runner_v1.json"
)


class _SyntheticGuard:
    causalcache_guard_marker = IMPORT_GUARD_MARKER
    blocked_roots = BLOCKED_IMPORT_ROOTS

    def find_spec(self, fullname, path=None, target=None):
        del fullname, path, target
        return None


def _test_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return (
        [{"state_index": index, "kind": "raw"} for index in range(192)],
        [{"state_index": index, "kind": "derived"} for index in range(192)],
    )


def _test_contract_at(
    base: ScientificRepairRunnerContract,
    root: Path,
) -> ScientificRepairRunnerContract:
    data = copy.deepcopy(dict(base.data))
    data["output_contract"]["canonical_archive_path"] = str(root / "output.tar")
    data["output_contract"]["claim_path"] = str(root / "claim.json")
    data["output_contract"]["completion_path"] = str(root / "completion.json")
    data["no_gpu_runtime"]["launch_receipt_path"] = str(root / "launch.json")
    return ScientificRepairRunnerContract(
        data=data,
        sha256=base.sha256,
        repository_root=base.repository_root,
        source_path=base.source_path,
    )


def _formal_fixture_archive(
    contract: ScientificRepairRunnerContract,
) -> runner.RepairedArchive:
    raw_rows, derived_rows = _test_rows()
    return build_test_only_repaired_archive(
        contract,
        core_report={"status": "TEST_ONLY_CORE"},
        raw_states=raw_rows,
        derived_states=derived_rows,
    )


def _source() -> SourceIdentity:
    modules = (
        {
            "module": "causalcache",
            "path": "code/causalcache/__init__.py",
            "sha256": "2" * 64,
            "size_bytes": 1,
        },
    )
    return SourceIdentity(
        head="1" * 40,
        origin_main="1" * 40,
        remote_main="1" * 40,
        branch="main",
        origin_url="https://github.com/luojiaxuan/CausalCache.git",
        inventory=(),
        inventory_sha256=sha256_bytes(canonical_json_bytes(())),
        loaded_module_inventory=modules,
        loaded_module_inventory_sha256=sha256_bytes(
            canonical_json_bytes(modules)
        ),
    )


class ScientificRepairRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_frozen_runner_contract(
            CONFIG_PATH,
            repository_root=REPOSITORY_ROOT,
        )

    def _execute_mocked_formal(
        self,
        contract: ScientificRepairRunnerContract,
        *,
        revalidate_callback,
        events: list[str],
    ) -> dict[str, object]:
        source = _source()
        evidence = InvalidForensicEvidence(
            files={},
            manifest={"status": "INVALID"},
            inventory=(),
            tree_inventory_sha256="a" * 64,
        )
        original = OriginalInvalidSnapshot(
            source_evidence=evidence,
            local_archive_evidence=evidence,
            local_archive_bytes=b"invalid",
            bound_state=BoundStateSnapshot(b"p1", b"tag-claim", b"tag-completion"),
        )
        fresh = runner.FreshForensicDownload(
            evidence=evidence,
            archive_bytes=b"invalid",
            sidecar_bytes=b"sidecar",
            attestation={"formal_hf_download": True},
            formal_hf_download=True,
        )
        archive = _formal_fixture_archive(contract)
        receipt_runtime = {"receipt_sha256": "a" * 64}
        claim_preexisted = Path(contract.output["claim_path"]).exists()

        def download(*args, **kwargs):
            del args, kwargs
            if not claim_preexisted:
                self.assertFalse(Path(contract.output["claim_path"]).exists())
            events.append("fresh_hf")
            return fresh

        def run_core(*args, **kwargs):
            del args, kwargs
            self.assertTrue(Path(contract.output["claim_path"]).exists())
            events.append("core")
            return object()

        def stable(path, **kwargs):
            del kwargs
            candidate = Path(path)
            self.assertTrue(candidate.exists())
            events.append("stable_output")
            descriptor = os.open(candidate, os.O_RDONLY)
            metadata = os.fstat(descriptor)
            return runner.StableArchiveHandle(
                descriptor=descriptor,
                path=candidate,
                stat_fingerprint=runner._stat_fingerprint(metadata),
                payload=candidate.read_bytes(),
                archive=archive,
            )

        real_completion_publish = runner.atomic_publish_completion_last

        def publish_completion(*args, **kwargs):
            self.assertTrue(
                Path(contract.output["canonical_archive_path"]).exists()
            )
            events.append("completion_publish")
            return real_completion_publish(*args, **kwargs)

        patches = (
            mock.patch.object(runner, "_load_parent_contracts", return_value=(object(), object())),
            mock.patch.object(runner, "validate_external_replay_inputs", return_value={}),
            mock.patch.object(runner, "validate_clean_pushed_source", return_value=source),
            mock.patch.object(runner, "validate_import_guard", return_value={}),
            mock.patch.object(
                runner,
                "validate_no_gpu_launch_receipt",
                return_value=receipt_runtime,
            ),
            mock.patch.object(
                runner, "snapshot_original_invalid_attempt", return_value=original
            ),
            mock.patch.object(runner, "download_fresh_immutable_forensic", side_effect=download),
            mock.patch.object(runner, "_require_fresh_matches_original", return_value=None),
            mock.patch.object(
                runner,
                "_revalidate_immutable_preconditions",
                side_effect=revalidate_callback,
            ),
            mock.patch.object(
                runner,
                "build_deterministic_claim",
                return_value={"schema_version": "1.0.0", "status": "claim"},
            ),
            mock.patch.object(runner, "_run_core", side_effect=run_core),
            mock.patch.object(
                runner, "build_formal_repaired_archive", return_value=archive
            ),
            mock.patch.object(runner, "read_repaired_archive", return_value=archive),
            mock.patch.object(runner, "open_stable_repaired_archive", side_effect=stable),
            mock.patch.object(
                runner,
                "atomic_publish_completion_last",
                side_effect=publish_completion,
            ),
        )
        with contextlib.ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            return dict(
                runner.execute_formal_scientific_repair(
                    contract,
                    mode="run",
                    expected_source_git_commit=source.head,
                    receipt_path=contract.runtime["launch_receipt_path"],
                    hf_token="token",
                    fresh_download_parent=Path(contract.output["claim_path"]).parent,
                    argv=["runner.py", "run"],
                )
            )

    def test_frozen_contract_identity(self) -> None:
        self.assertEqual(self.contract.sha256, FROZEN_CONFIG_SHA256)
        self.assertEqual(
            self.contract.output["exact_members"],
            [
                "audit.json",
                "derived_labels.jsonl",
                "manifest.json",
                "raw_states.jsonl",
            ],
        )
        self.assertTrue(
            self.contract.output["completion_created_after_strict_output_readback"]
        )
        self.assertFalse(
            self.contract.data["planned_hf_publication"][
                "publication_authorized_by_this_runner"
            ]
        )

    def test_tampered_contract_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / runner.CANONICAL_CONFIG_PATH
            target.parent.mkdir(parents=True)
            payload = CONFIG_PATH.read_bytes().replace(
                b'"gate_training_unlocked": false',
                b'"gate_training_unlocked": true',
                1,
            )
            target.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "SHA256 drifted"):
                load_frozen_runner_contract(target, repository_root=root)

    def test_test_only_payload_is_deterministic_but_formal_ineligible(self) -> None:
        raw, derived = _test_rows()
        first = build_test_only_repaired_archive(
            self.contract,
            core_report={"status": "TEST_ONLY_CORE"},
            raw_states=raw,
            derived_states=derived,
        )
        second = build_test_only_repaired_archive(
            self.contract,
            core_report={"status": "TEST_ONLY_CORE"},
            raw_states=raw,
            derived_states=derived,
        )
        self.assertEqual(first.archive_bytes, second.archive_bytes)
        audit = json.loads(first.files["audit.json"])
        self.assertEqual(audit["status"], TEST_ONLY_STATUS)
        with self.assertRaisesRegex(ValueError, "test-only"):
            repaired_archive_from_files(
                first.files,
                contract=self.contract,
                require_formal=True,
            )

    def test_formal_reader_rejects_fabricated_raw_and_derived_rows(self) -> None:
        raw_rows = [
            {"run_contract_sha256": "a" * 64, "state": {}, "kind": "fake"}
            for _ in range(192)
        ]
        derived_rows = [{"kind": "fake"} for _ in range(192)]
        raw = b"".join(canonical_json_bytes(row) + b"\n" for row in raw_rows)
        derived = b"".join(
            canonical_json_bytes(row) + b"\n" for row in derived_rows
        )
        guard = {
            "guard_marker": IMPORT_GUARD_MARKER,
            "blocked_roots": list(BLOCKED_IMPORT_ROOTS),
            "forbidden_modules_imported": [],
            "pil_allowed": True,
        }
        report = {
            key: {}
            for key in (
                "original_attempt",
                "append_only_execution_log",
                "legacy_validator_negative_control",
                "monitor_validation",
                "raw_distance_reduction",
                "independent_stdlib_math_audit",
                "external_input_replay",
                "forensic_evidence",
                "execution_scope",
            )
        }
        report.update(
            {
                "schema_version": "1.0.0",
                "protocol_id": runner.CORE_PROTOCOL_ID,
                "status": CORE_PASS_STATUS,
                "formal_scientific_repair_pass": True,
            }
        )
        audit = {
            "schema_version": "1.0.0",
            "protocol_id": runner.PROTOCOL_ID,
            "status": AUDIT_STATUS,
            "runner_config_sha256": self.contract.sha256,
            "formal_scientific_repair_pass": True,
            "source": dict(runner._source_record(_source())),
            "runtime": {
                "receipt_sha256": "a" * 64,
                "receipt_size_bytes": 1,
                "container_id": "a" * 64,
                "container_name": "sglang-omni-jaxan-07171234",
                "container_hostname": "a" * 12,
                "container_image_digest": self.contract.runtime[
                    "required_container_image_digest"
                ],
                "docker_runtime": "runc",
                "device_requests_raw": None,
                "normalized_device_requests": [],
                "nvidia_device_nodes": [],
                "nvidia_visible_devices": "void",
                "cuda_visible_devices": "",
                "python_version": "3",
                "platform": "test",
                "device": "cpu",
                "import_guard": guard,
            },
            "import_guard": guard,
            "original_invalid_attempt": {"producer_reclassified": False},
            "fresh_immutable_forensic": {"formal_hf_download": True},
            "external_replay_inputs": {},
            "claim_sha256": "b" * 64,
            "repair_runtime_operation_counts": dict(runner._operation_counts()),
            "archived_producer_operation_counts": {},
            "scientific_repair_core_report": report,
            "scientific_summary": {},
            "formal_consumption": dict(self.contract.data["formal_consumption"]),
        }
        files = {
            "raw_states.jsonl": raw,
            "derived_labels.jsonl": derived,
            "audit.json": runner._pretty_json_bytes(audit),
        }
        inventory = runner._inventory(files)
        files["manifest.json"] = runner._pretty_json_bytes(
            {
                "schema_version": "1.0.0",
                "protocol_id": runner.PROTOCOL_ID,
                "status": MANIFEST_STATUS,
                "runner_config_sha256": self.contract.sha256,
                "source_git_commit": "1" * 40,
                "raw_state_count": 192,
                "derived_state_count": 192,
                "payload_inventory": list(inventory),
                "scientific_payload_sha256": sha256_bytes(
                    canonical_json_bytes(
                        {
                            "raw_states_sha256": sha256_bytes(raw),
                            "derived_labels_sha256": sha256_bytes(derived),
                            "core_report_sha256": sha256_bytes(
                                canonical_json_bytes(report)
                            ),
                        }
                    )
                ),
                "original_attempt_remains_invalid": True,
                "producer_reclassified": False,
                "formal_consumption": dict(
                    self.contract.data["formal_consumption"]
                ),
            }
        )
        with self.assertRaises((TypeError, ValueError)):
            repaired_archive_from_files(
                files, contract=self.contract, require_formal=True
            )

    def test_formal_archive_exact_members_and_ustar_metadata(self) -> None:
        archive = _formal_fixture_archive(self.contract)
        with tarfile.open(fileobj=io.BytesIO(archive.archive_bytes), mode="r:") as tar:
            members = tar.getmembers()
        prefix = self.contract.output["archive_member_prefix"]
        self.assertEqual(
            [member.name for member in members],
            [f"{prefix}/{name}" for name in self.contract.output["exact_members"]],
        )
        for member in members:
            self.assertEqual(member.mode, 0o644)
            self.assertEqual(member.uid, 0)
            self.assertEqual(member.gid, 0)
            self.assertEqual(member.mtime, 0)
            self.assertEqual(member.pax_headers, {})

    def test_formal_archive_strict_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.tar"
            archive = _formal_fixture_archive(self.contract)
            path.write_bytes(archive.archive_bytes)
            reread = read_repaired_archive(
                path, contract=self.contract, require_formal=False
            )
            self.assertEqual(reread.archive_bytes, archive.archive_bytes)
            self.assertEqual(reread.tree_inventory_sha256, archive.tree_inventory_sha256)

    def test_archive_missing_member_is_rejected(self) -> None:
        archive = _formal_fixture_archive(self.contract)
        files = dict(archive.files)
        files.pop("audit.json")
        with self.assertRaisesRegex(ValueError, "member inventory"):
            runner.deterministic_repaired_ustar_bytes(files, contract=self.contract)

    def test_noncanonical_jsonl_is_rejected(self) -> None:
        archive = _formal_fixture_archive(self.contract)
        files = dict(archive.files)
        files["raw_states.jsonl"] = files["raw_states.jsonl"].replace(
            b'{"kind":"raw","state_index":0}',
            b'{"state_index":0,"kind":"raw"}',
            1,
        )
        with self.assertRaisesRegex(ValueError, "not canonical JSON"):
            repaired_archive_from_files(
                files, contract=self.contract, require_formal=False
            )

    def test_noncanonical_tar_metadata_is_rejected(self) -> None:
        archive = _formal_fixture_archive(self.contract)
        destination = io.BytesIO()
        prefix = self.contract.output["archive_member_prefix"]
        with tarfile.open(
            fileobj=destination, mode="w", format=tarfile.USTAR_FORMAT
        ) as tar:
            for name in self.contract.output["exact_members"]:
                payload = archive.files[name]
                info = tarfile.TarInfo(f"{prefix}/{name}")
                info.size = len(payload)
                info.mode = 0o600 if name == "audit.json" else 0o644
                info.mtime = 0
                tar.addfile(info, io.BytesIO(payload))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.tar"
            path.write_bytes(destination.getvalue())
            with self.assertRaisesRegex(ValueError, "metadata drifted"):
                read_repaired_archive(
                    path, contract=self.contract, require_formal=False
                )

    def test_atomic_publish_is_no_replace_and_strictly_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = _test_contract_at(self.contract, root)
            archive = _formal_fixture_archive(contract)
            checks = []
            with mock.patch.object(
                runner, "read_repaired_archive", return_value=archive
            ):
                atomic_publish_repaired_archive(
                    root / "output.tar",
                    archive,
                    contract=contract,
                    pre_publish_check=lambda: checks.append("checked"),
                )
            self.assertEqual(checks, ["checked"])
            self.assertEqual(
                read_repaired_archive(
                    root / "output.tar", contract=contract, require_formal=False
                ).archive_bytes,
                archive.archive_bytes,
            )
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                atomic_publish_repaired_archive(
                    root / "output.tar",
                    archive,
                    contract=contract,
                    pre_publish_check=lambda: None,
                )

    def test_completion_without_output_is_permanently_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _test_contract_at(self.contract, Path(temporary))
            Path(contract.output["completion_path"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "completion-without-output"):
                inspect_repair_state(contract)

    def test_output_without_claim_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _test_contract_at(self.contract, Path(temporary))
            Path(contract.output["canonical_archive_path"]).write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "orphan repaired output"):
                inspect_repair_state(contract)

    def test_output_without_completion_is_recoverable_only_with_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = _test_contract_at(self.contract, root)
            Path(contract.output["claim_path"]).write_bytes(b"claim")
            Path(contract.output["canonical_archive_path"]).write_bytes(b"output")
            self.assertEqual(
                inspect_repair_state(contract),
                {
                    "claim_exists": True,
                    "output_exists": True,
                    "completion_exists": False,
                },
            )

    def test_exclusive_state_allows_only_identical_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "claim.json"
            first = exclusive_or_identical_state(path, {"value": 1}, mode=0o600)
            second = exclusive_or_identical_state(path, {"value": 1}, mode=0o600)
            self.assertTrue(first[2])
            self.assertFalse(second[2])
            self.assertEqual(first[:2], second[:2])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(ValueError, "not byte-identical"):
                exclusive_or_identical_state(path, {"value": 2}, mode=0o600)

    def test_completion_binds_archive_after_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _test_contract_at(self.contract, Path(temporary))
            archive = _formal_fixture_archive(contract)
            completion = build_completion_record(
                contract,
                claim_sha256="a" * 64,
                archive=archive,
                source=_source(),
            )
            self.assertEqual(completion["archive_sha256"], archive.archive_sha256)
            self.assertEqual(completion["member_count"], 4)
            self.assertTrue(
                completion["completion_created_after_strict_output_readback"]
            )

    def test_orchestrator_orders_fresh_claim_core_output_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _test_contract_at(self.contract, Path(temporary))
            events: list[str] = []

            def revalidate(*args, **kwargs):
                del args, kwargs
                events.append("revalidate")
                return _source()

            result = self._execute_mocked_formal(
                contract,
                revalidate_callback=revalidate,
                events=events,
            )
            self.assertLess(events.index("fresh_hf"), events.index("core"))
            self.assertLess(events.index("core"), events.index("stable_output"))
            self.assertLess(
                events.index("stable_output"), events.index("completion_publish")
            )
            self.assertTrue(Path(contract.output["claim_path"]).exists())
            self.assertTrue(Path(contract.output["canonical_archive_path"]).exists())
            self.assertTrue(Path(contract.output["completion_path"]).exists())
            self.assertTrue(result["completion_created"])

    def test_orchestrator_recovers_claim_and_output_only_after_full_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _test_contract_at(self.contract, Path(temporary))
            calls = 0

            def fail_before_completion(*args, **kwargs):
                nonlocal calls
                del args, kwargs
                calls += 1
                if calls == 4:
                    raise ValueError("injected immediately before completion")
                return _source()

            with self.assertRaisesRegex(ValueError, "injected immediately"):
                self._execute_mocked_formal(
                    contract,
                    revalidate_callback=fail_before_completion,
                    events=[],
                )
            self.assertTrue(Path(contract.output["claim_path"]).exists())
            self.assertTrue(Path(contract.output["canonical_archive_path"]).exists())
            self.assertFalse(Path(contract.output["completion_path"]).exists())

            replay_events: list[str] = []
            result = self._execute_mocked_formal(
                contract,
                revalidate_callback=lambda *args, **kwargs: _source(),
                events=replay_events,
            )
            self.assertIn("fresh_hf", replay_events)
            self.assertIn("core", replay_events)
            self.assertFalse(result["claim_created"])
            self.assertFalse(result["output_created"])
            self.assertTrue(result["completion_created"])

    def test_orchestrator_detects_output_mutation_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _test_contract_at(self.contract, Path(temporary))
            calls = 0

            def mutate_before_completion(*args, **kwargs):
                nonlocal calls
                del args, kwargs
                calls += 1
                if calls == 4:
                    Path(contract.output["canonical_archive_path"]).write_bytes(
                        b"mutated"
                    )
                return _source()

            with self.assertRaisesRegex(ValueError, "changed before completion"):
                self._execute_mocked_formal(
                    contract,
                    revalidate_callback=mutate_before_completion,
                    events=[],
                )
            self.assertFalse(Path(contract.output["completion_path"]).exists())

    def test_import_guard_blocks_framework_and_allows_pil(self) -> None:
        guard = install_import_guard()
        try:
            with self.assertRaisesRegex(ImportError, "forbidden"):
                guard.find_spec("torch.cuda")
            self.assertIsNone(guard.find_spec("PIL.Image"))
        finally:
            remove_import_guard(guard)

    def test_runner_import_guard_requires_bootstrap_marker(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_import_guard(self.contract)
        guard = install_import_guard()
        try:
            result = validate_import_guard(self.contract)
            self.assertTrue(result["pil_allowed"])
        finally:
            remove_import_guard(guard)

    def test_runner_import_guard_rejects_marker_lookalike(self) -> None:
        guard = _SyntheticGuard()
        sys.meta_path.insert(0, guard)
        try:
            with self.assertRaisesRegex(ValueError, "missing or drifted"):
                validate_import_guard(self.contract)
            with self.assertRaisesRegex(RuntimeError, "pre-existing"):
                install_import_guard()
        finally:
            sys.meta_path.remove(guard)

    def test_runner_import_guard_rejects_loaded_framework(self) -> None:
        guard = install_import_guard()
        previous = sys.modules.get("torch")
        sys.modules["torch"] = types.ModuleType("torch")
        try:
            with self.assertRaisesRegex(ValueError, "forbidden"):
                validate_import_guard(self.contract)
        finally:
            if previous is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous
            remove_import_guard(guard)

    def test_runner_import_guard_does_not_reject_loaded_pil(self) -> None:
        guard = install_import_guard()
        previous = sys.modules.get("PIL")
        sys.modules["PIL"] = types.ModuleType("PIL")
        try:
            self.assertEqual(
                validate_import_guard(self.contract)["forbidden_modules_imported"],
                [],
            )
        finally:
            if previous is None:
                sys.modules.pop("PIL", None)
            else:
                sys.modules["PIL"] = previous
            remove_import_guard(guard)

    def test_no_gpu_launch_receipt_accepts_null_device_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = _test_contract_at(self.contract, root)
            receipt = {
                "schema_version": "1.0.0",
                "status": contract.runtime["launch_receipt_status"],
                "host_alias": "hyper00",
                "host_hostname": "node-radixark-16-0000",
                "container_id": "a" * 64,
                "container_name": "sglang-omni-jaxan-07171234",
                "container_hostname": "a" * 12,
                "container_image_digest": contract.runtime[
                    "required_container_image_digest"
                ],
                "runtime": "runc",
                "privileged": False,
                "device_requests_raw": None,
                "normalized_device_requests": [],
                "explicit_devices": [],
                "nvidia_visible_devices": "void",
                "cuda_visible_devices": "",
                "source_git_commit": "1" * 40,
            }
            path = Path(contract.runtime["launch_receipt_path"])
            path.write_bytes(runner._pretty_json_bytes(receipt))
            path.chmod(0o600)
            with mock.patch.object(runner.socket, "gethostname", return_value="a" * 12), mock.patch.object(
                runner, "_nvidia_device_nodes", return_value=[]
            ), mock.patch.dict(
                os.environ,
                {"NVIDIA_VISIBLE_DEVICES": "void", "CUDA_VISIBLE_DEVICES": ""},
            ):
                result = validate_no_gpu_launch_receipt(
                    contract,
                    source=_source(),
                    receipt_path=path,
                )
            self.assertEqual(result["normalized_device_requests"], [])
            self.assertIsNone(result["device_requests_raw"])

    def test_no_gpu_launch_receipt_rejects_gpu_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _test_contract_at(self.contract, Path(temporary))
            receipt = {
                "schema_version": "1.0.0",
                "status": contract.runtime["launch_receipt_status"],
                "host_alias": "hyper00",
                "host_hostname": "node-radixark-16-0000",
                "container_id": "a" * 64,
                "container_name": "sglang-omni-jaxan-07171234",
                "container_hostname": "a" * 12,
                "container_image_digest": contract.runtime[
                    "required_container_image_digest"
                ],
                "runtime": "runc",
                "privileged": False,
                "device_requests_raw": [{"DeviceIDs": ["0"]}],
                "normalized_device_requests": [{"DeviceIDs": ["0"]}],
                "explicit_devices": [],
                "nvidia_visible_devices": "void",
                "cuda_visible_devices": "",
                "source_git_commit": "1" * 40,
            }
            path = Path(contract.runtime["launch_receipt_path"])
            path.write_bytes(runner._pretty_json_bytes(receipt))
            path.chmod(0o600)
            with mock.patch.object(runner.socket, "gethostname", return_value="a" * 12), mock.patch.object(
                runner, "_nvidia_device_nodes", return_value=[]
            ), mock.patch.dict(
                os.environ,
                {"NVIDIA_VISIBLE_DEVICES": "void", "CUDA_VISIBLE_DEVICES": ""},
            ), self.assertRaisesRegex(ValueError, "no-GPU runtime"):
                validate_no_gpu_launch_receipt(
                    contract,
                    source=_source(),
                    receipt_path=path,
                )

    def test_revalidation_exactly_binds_receipt_and_allows_only_module_growth(self) -> None:
        source = _source()
        added = {
            "module": "causalcache.lazy",
            "path": "code/causalcache/lazy.py",
            "sha256": "3" * 64,
            "size_bytes": 2,
        }
        grown_inventory = (*source.loaded_module_inventory, added)
        grown = replace(
            source,
            loaded_module_inventory=grown_inventory,
            loaded_module_inventory_sha256=sha256_bytes(
                canonical_json_bytes(grown_inventory)
            ),
        )
        evidence = InvalidForensicEvidence(
            files={}, manifest={}, inventory=(), tree_inventory_sha256="a" * 64
        )
        original = OriginalInvalidSnapshot(
            source_evidence=evidence,
            local_archive_evidence=evidence,
            local_archive_bytes=b"x",
            bound_state=BoundStateSnapshot(b"p1", b"claim", b"completion"),
        )
        patches = (
            mock.patch.object(runner, "validate_clean_pushed_source", return_value=grown),
            mock.patch.object(runner, "validate_import_guard", return_value={}),
            mock.patch.object(
                runner,
                "validate_no_gpu_launch_receipt",
                return_value={"receipt_sha256": "a" * 64},
            ),
            mock.patch.object(runner, "validate_external_replay_inputs", return_value={}),
            mock.patch.object(
                runner, "snapshot_original_invalid_attempt", return_value=original
            ),
        )
        with contextlib.ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            with self.assertRaisesRegex(ValueError, "receipt changed"):
                runner._revalidate_immutable_preconditions(
                    self.contract,
                    p0_contract=object(),
                    tag_contract=object(),
                    source=source,
                    expected_head=source.head,
                    receipt_path="/receipt",
                    runtime_before={"receipt_sha256": "b" * 64},
                    original_before=original,
                    external_inputs_before={},
                    allow_loaded_module_growth=True,
                )
            self.assertEqual(
                runner._revalidate_immutable_preconditions(
                    self.contract,
                    p0_contract=object(),
                    tag_contract=object(),
                    source=source,
                    expected_head=source.head,
                    receipt_path="/receipt",
                    runtime_before={"receipt_sha256": "a" * 64},
                    original_before=original,
                    external_inputs_before={},
                    allow_loaded_module_growth=True,
                ),
                grown,
            )
            with self.assertRaisesRegex(ValueError, "source changed"):
                runner._revalidate_immutable_preconditions(
                    self.contract,
                    p0_contract=object(),
                    tag_contract=object(),
                    source=source,
                    expected_head=source.head,
                    receipt_path="/receipt",
                    runtime_before={"receipt_sha256": "a" * 64},
                    original_before=original,
                    external_inputs_before={},
                    allow_loaded_module_growth=False,
                )

    def test_original_invalid_snapshot_exact_comparison(self) -> None:
        evidence = InvalidForensicEvidence(
            files={"x": b"x"},
            manifest={"status": "INVALID"},
            inventory=(),
            tree_inventory_sha256="a" * 64,
        )
        snapshot = OriginalInvalidSnapshot(
            source_evidence=evidence,
            local_archive_evidence=evidence,
            local_archive_bytes=b"archive",
            bound_state=BoundStateSnapshot(b"p1", b"claim", b"completion"),
        )
        require_original_snapshot_unchanged(snapshot, copy.deepcopy(snapshot))
        changed = OriginalInvalidSnapshot(
            source_evidence=evidence,
            local_archive_evidence=evidence,
            local_archive_bytes=b"changed",
            bound_state=snapshot.bound_state,
        )
        with self.assertRaisesRegex(ValueError, "changed"):
            require_original_snapshot_unchanged(snapshot, changed)

    def test_hf_token_reader_never_accepts_unsafe_modes_or_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "token"
            path.write_text("secret\n", encoding="utf-8")
            path.chmod(0o600)
            self.assertEqual(_read_hf_token(path), "secret")
            path.write_text("bad token\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one token"):
                _read_hf_token(path)
            path.write_text("secret", encoding="utf-8")
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "0400 or 0600"):
                _read_hf_token(path)

    def test_validate_contract_cli_is_network_free(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    REPOSITORY_ROOT
                    / "code/scripts/"
                    "run_restoration_v2_2_expansion_labels_scientific_repair.py"
                ),
                "validate-contract",
                "--repository-root",
                str(REPOSITORY_ROOT),
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["network_access_count"], 0)
        self.assertEqual(result["gpu_or_model_operation_count"], 0)
        self.assertFalse(result["formal_run_performed"])


if __name__ == "__main__":
    unittest.main()
