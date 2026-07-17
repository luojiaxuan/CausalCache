from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import causalcache.restoration_v2_2_expansion_substrate_artifact as artifact
from causalcache.restoration_v2_2_expansion_substrate_artifact import (
    FROZEN_BASE_CONFIG_SHA256,
    INVALID_OUTCOME,
    PASS_OUTCOME,
    PLANNED_COUNTS,
    _runner_interface_contract,
    _validate_monitor_gpu_uuid_binding,
    _validate_processor_identity,
    _validate_worker_file_window,
    _validate_worker_ledger,
    canonical_json_bytes,
    deterministic_ustar_bytes,
    expected_worker_specs,
    pretty_json_bytes,
    reduce_expansion_substrate_gate,
    sha256_bytes,
    sha256_file,
    validate_runner_freeze_data,
)
from scripts.manage_restoration_v2_2_expansion_substrate_artifact import (
    _build_parser,
    _exclusive_write,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = ROOT / artifact.CANONICAL_CONFIG_PATH


def _load_base_config() -> dict[str, object]:
    return json.loads(BASE_CONFIG_PATH.read_text())


def _identity(path: Path, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _freeze_fixture() -> tuple[dict[str, object], dict[str, object]]:
    base = _load_base_config()
    parent = [
        dict(record)
        for record in base["scientific_source_lock"]["source_files"]
    ]
    paths = {record["path"] for record in parent}
    paths.add(artifact.CANONICAL_CONFIG_PATH)
    paths.update(base["scientific_source_lock"]["reserved_execution_source_paths"])
    by_path = {record["path"]: record for record in parent}
    inventory = []
    for relative in sorted(paths):
        inventory.append(
            dict(by_path[relative])
            if relative in by_path
            else _identity(ROOT / relative, relative)
        )
    completion_path = ROOT / artifact.CANONICAL_COMPLETION_PATH
    freeze = {
        "schema_version": artifact.RUNNER_FREEZE_SCHEMA_VERSION,
        "protocol_id": artifact.RUNNER_FREEZE_PROTOCOL_ID,
        "status": "FROZEN_COMMITTED_PUSHED_EXPANSION_SUBSTRATE_RUNNER_SOURCE",
        "runner_source_git_commit": "a" * 40,
        "base_config": _identity(
            BASE_CONFIG_PATH,
            artifact.CANONICAL_CONFIG_PATH,
        ),
        "derived_completion": _identity(
            completion_path,
            artifact.CANONICAL_COMPLETION_PATH,
        ),
        "source_inventory": inventory,
        "source_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "interfaces": _runner_interface_contract(),
        "authorization": {
            "runner_source_commit_must_be_committed": True,
            "runner_source_commit_must_be_ancestor_of_execution_head": True,
            "execution_head_and_origin_main_must_match": True,
            "runner_freeze_file_must_be_committed_at_execution_head": True,
            "source_blobs_must_equal_runner_source_commit": True,
            "source_only_freeze_authorizes_gpu_execution": False,
            "formal_execution_requires_runtime_authorization_validation": True,
        },
    }
    return base, freeze


def _processor_identity(step: int, marker: str) -> dict[str, object]:
    value: dict[str, object] = {
        "processor_path": "GUIOwlV22EagerRuntime._encode_exact_batch",
        "rendered_prompt_sha256": sha256_bytes(f"render:{marker}".encode()),
        "input_ids_sha256": sha256_bytes(f"ids:{marker}".encode()),
        "attention_mask_sha256": sha256_bytes(b"mask"),
        "pixel_values_sha256": sha256_bytes(b"pixels"),
        "image_grid_thw_sha256": sha256_bytes(b"grid"),
        "input_ids_shape": [1, 32],
        "input_ids_dtype": "torch.int64",
        "attention_mask_shape": [1, 32],
        "attention_mask_dtype": "torch.int64",
        "pixel_values_shape": [step - 1, 1176, 1176],
        "pixel_values_dtype": "torch.bfloat16",
        "image_grid_thw_shape": [step - 1, 3],
        "image_grid_thw_dtype": "torch.int64",
        "image_count": step - 1,
        "policy_forward_executed": False,
    }
    value["processor_request_sha256"] = sha256_bytes(canonical_json_bytes(value))
    return value


class ExpansionSubstrateArtifactTest(unittest.TestCase):
    def test_monitor_gpu_uuid_binding_rejects_another_device_pair(self) -> None:
        runtimes = {
            "even": {"gpu_uuid": "GPU-worker-even"},
            "odd": {"gpu_uuid": "GPU-worker-odd"},
        }
        _validate_monitor_gpu_uuid_binding(
            {"visible_gpu_uuids": ["GPU-worker-even", "GPU-worker-odd"]},
            runtimes=runtimes,
        )
        with self.assertRaisesRegex(ValueError, "worker runtime GPUs"):
            _validate_monitor_gpu_uuid_binding(
                {"visible_gpu_uuids": ["GPU-other-even", "GPU-other-odd"]},
                runtimes=runtimes,
            )

    def test_runner_freeze_binds_exact_parent63_base1_reserved3(self) -> None:
        base, freeze = _freeze_fixture()
        validation = validate_runner_freeze_data(freeze, base_config=base)
        self.assertEqual(len(validation["source_inventory"]), 67)
        self.assertEqual(validation["base_config_sha256"], FROZEN_BASE_CONFIG_SHA256)

        tampered = copy.deepcopy(freeze)
        tampered["source_inventory"][0]["sha256"] = "0" * 64
        tampered["source_inventory_sha256"] = sha256_bytes(
            canonical_json_bytes(tampered["source_inventory"])
        )
        with self.assertRaisesRegex(ValueError, "parent63"):
            validate_runner_freeze_data(tampered, base_config=base)

    def test_wrong_base_config_hash_fails_closed(self) -> None:
        base, freeze = _freeze_fixture()
        wrong = copy.deepcopy(base)
        wrong["protocol_id"] = "tampered"
        with self.assertRaisesRegex(ValueError, "base config identity"):
            validate_runner_freeze_data(freeze, base_config=wrong)

    def test_processor_identity_tamper_is_rejected(self) -> None:
        identity = _processor_identity(4, "real")
        validated = _validate_processor_identity(
            identity,
            name="processor identity",
            decision_step_id=4,
        )
        self.assertEqual(validated["image_count"], 3)
        tampered = copy.deepcopy(identity)
        tampered["pixel_values_shape"] = [4, 1176, 1176]
        with self.assertRaisesRegex(ValueError, "image count|aggregate hash"):
            _validate_processor_identity(
                tampered,
                name="processor identity",
                decision_step_id=4,
            )

    def test_invalid_partial_union_accepts_only_bounded_write_windows(self) -> None:
        _validate_worker_file_window(
            outcome=INVALID_OUTCOME,
            attempted=[0, 2, 4],
            completed=[0],
            observed_markers=[0, 2],
            observed_states=[0, 2],
        )
        with self.assertRaisesRegex(ValueError, "bounded"):
            _validate_worker_file_window(
                outcome=INVALID_OUTCOME,
                attempted=[0, 2, 4],
                completed=[0],
                observed_markers=[0],
                observed_states=[0],
            )
        with self.assertRaisesRegex(ValueError, "terminal"):
            _validate_worker_file_window(
                outcome=PASS_OUTCOME,
                attempted=[0, 2],
                completed=[0, 2],
                observed_markers=[0],
                observed_states=[0, 2],
            )

    def test_retry_and_top_up_ledgers_are_rejected(self) -> None:
        spec = expected_worker_specs()[0]
        base = {
            "schema_version": artifact.SCHEMA_VERSION,
            "protocol_id": artifact.PROTOCOL_ID,
            "status": "WORKER_SHARD_RUNNING_NO_RETRY",
            "run_contract_sha256": "1" * 64,
            "worker": spec.to_dict(),
            "claimed_at_utc": "2026-07-17T00:00:00Z",
            "attempted_state_indices": [0],
            "completed_state_indices": [],
            "retry_count": 0,
            "top_up_count": 0,
        }
        for key in ("retry_count", "top_up_count"):
            tampered = dict(base)
            tampered[key] = 1
            with self.assertRaisesRegex(ValueError, "retry/top-up"):
                _validate_worker_ledger(
                    tampered,
                    spec=spec,
                    run_contract_sha256="1" * 64,
                )

    def test_failure_stays_in_fixed_192_state_gate_denominator(self) -> None:
        records = []
        for index in range(192):
            success = index != 17
            records.append(
                {
                    "state_id": f"state-{index:03d}",
                    "parse_success": success,
                    "finite_logit_distances": success,
                    "repeat_canonical_action_agreement": success,
                    "repeat_reference_kl": 1e-6 if success else None,
                    "summary_reference_kl": 1e-2 if success else None,
                }
            )
        gate = reduce_expansion_substrate_gate(
            records,
            actual_counts={
                "generation_call_count": 384,
                "teacher_forward_count": 573,
                "kl_measurement_count": 382,
            },
            frozen_config=_load_base_config(),
        )
        self.assertEqual(gate["metrics"]["fixed_state_denominator"], 192)
        self.assertEqual(gate["metrics"]["parse_success_count"], 191)
        self.assertFalse(gate["gate_passed"])
        self.assertEqual(gate["derived_outcome"], artifact.NO_GO_OUTCOME)

    def test_deterministic_ustar_and_exclusive_manifest_write(self) -> None:
        files = {"b.json": b"b\n", "a.json": b"a\n"}
        first = deterministic_ustar_bytes(files)
        second = deterministic_ustar_bytes(dict(reversed(list(files.items()))))
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            _exclusive_write(output, pretty_json_bytes({"ok": True}))
            with self.assertRaises(FileExistsError):
                _exclusive_write(output, pretty_json_bytes({"ok": False}))

    def test_manager_exposes_freeze_package_validate_and_fresh_hf_interfaces(self) -> None:
        help_text = _build_parser().format_help()
        for command in (
            "materialize-runner-freeze",
            "validate-runner-freeze",
            "package",
            "validate",
            "validate-fresh-hf",
            "create-manifest",
        ):
            self.assertIn(command, help_text)


if __name__ == "__main__":
    unittest.main()
