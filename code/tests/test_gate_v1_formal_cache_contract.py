from __future__ import annotations

import copy
import shutil
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from causalcache.gate_v1_formal_cache_contract import (
    CANONICAL_CONFIG_PATH,
    FEATURE_MANIFEST_STATUS,
    FINAL_COMPLETION_STATUS,
    FROZEN_CONFIG_SHA256,
    LABEL_MANIFEST_STATUS,
    RUNNER_FREEZE_B_PATH,
    VALIDATION_STATUS,
    load_frozen_formal_cache_contract,
    load_strict_json_object,
    sha256_bytes,
    validate_formal_cache_config,
    validate_source_only_contract,
)
from scripts.validate_gate_v1_formal_cache_contract import _parser


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


class GateV1FormalCacheContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_strict_json_object(CONFIG)

    def test_canonical_contract_replays_formal_geometry_and_stays_locked(self) -> None:
        result = validate_formal_cache_config(self.config, repository_root=ROOT)
        self.assertEqual(result["status"], VALIDATION_STATUS)
        self.assertEqual(result["formal_train_trajectory_count"], 58)
        self.assertEqual(result["formal_train_state_count"], 174)
        self.assertEqual(result["formal_train_candidate_feature_count"], 522)
        self.assertEqual(result["formal_train_distance_value_count"], 1624)
        self.assertEqual(result["formal_train_conditional_edge_count"], 1682)
        self.assertEqual(result["input_artifact_count"], 4)
        self.assertEqual(result["exact_hf_input_file_count"], 9)
        self.assertFalse(result["execution_authorized"])
        self.assertFalse(result["formal_label_access_authorized"])
        self.assertFalse(result["matched_nll_authorized"])
        self.assertFalse(result["closed_loop_authorized"])
        self.assertEqual(result["development_semantic_decode_count"], 0)
        self.assertEqual(result["confirm_semantic_decode_count"], 0)

    def test_frozen_loader_binds_exact_config_and_exposes_runner_adapter(self) -> None:
        contract = load_frozen_formal_cache_contract(repository_root=ROOT)
        self.assertEqual(contract.sha256, FROZEN_CONFIG_SHA256)
        self.assertEqual(sha256_bytes(CONFIG.read_bytes()), FROZEN_CONFIG_SHA256)
        self.assertEqual(contract.parent_gate["sha256"], self.config["parent_gate_contract"]["sha256"])
        self.assertEqual(set(contract.inputs), set(self.config["input_artifacts"]))
        self.assertEqual(contract.formats["feature"]["manifest_status"], FEATURE_MANIFEST_STATUS)
        self.assertEqual(contract.formats["label"]["manifest_status"], LABEL_MANIFEST_STATUS)
        self.assertEqual(contract.destination["exact_three_targets"], [
            "formal58/v1/feature-cache-v1.tar",
            "formal58/v1/label-cache-v1.tar",
            "formal58/v1/cache-bundle-manifest-v1.json",
        ])
        self.assertEqual(contract.runtime["device"], "cpu")
        self.assertTrue(all(value == 0 for value in contract.operations.values()))
        self.assertEqual(contract.execution_operations["trajectory_semantic_decode_count"], 58)
        self.assertEqual(contract.execution_operations["ocr_semantic_decode_count"], 290)

    def test_source_only_validation_has_no_network_or_write_side_effect(self) -> None:
        write_methods = (
            "write_bytes",
            "write_text",
            "touch",
            "mkdir",
            "rename",
            "replace",
            "unlink",
        )
        patches = [
            mock.patch.object(Path, name, side_effect=AssertionError(f"write: {name}"))
            for name in write_methods
        ]
        patches.extend(
            (
                mock.patch.object(socket, "socket", side_effect=AssertionError("network")),
                mock.patch("urllib.request.urlopen", side_effect=AssertionError("network")),
            )
        )
        for patch in patches:
            patch.start()
        try:
            result = validate_source_only_contract(repository_root=ROOT)
        finally:
            for patch in reversed(patches):
                patch.stop()
        self.assertEqual(result["status"], VALIDATION_STATUS)
        self.assertFalse(result["execution_authorized"])
        self.assertTrue(result["pending_runner_freeze"])
        self.assertFalse(result["runner_freeze_b_present"])
        self.assertEqual(result["runner_freeze_b_path"], RUNNER_FREEZE_B_PATH)
        self.assertTrue(
            all(value == 0 for value in result["source_only_operation_counts"].values())
        )

    def test_source_only_cli_has_no_token_data_or_output_surface(self) -> None:
        parser = _parser()
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertEqual(options, {"-h", "--help", "--contract", "--repository-root"})
        for forbidden in ("--hf-token-file", "--data", "--output", "--labels", "--confirm"):
            self.assertNotIn(forbidden, options)

    def test_exact_hf_inputs_and_publication_completion_are_bound(self) -> None:
        inputs = self.config["input_artifacts"]
        self.assertEqual(set(inputs), {
            "legacy_derived_features",
            "expansion_derived_features",
            "legacy_restoration_labels",
            "repaired_expansion_restoration_labels",
        })
        self.assertEqual(sum(len(value["files"]) for value in inputs.values()), 9)
        self.assertEqual(
            inputs["repaired_expansion_restoration_labels"]["immutable_revision"],
            "7a6c254b8cec0dd3d8111dfc9c080de357e5cef3",
        )
        completion = self.config["publication_completion_binding"]
        self.assertEqual(
            completion["completion_seal"]["sha256"],
            "995b3ee25b643550de3df680b52382ef8b6b1373a3aad4c00bdc9caa34cfa2ff",
        )
        self.assertTrue(completion["formal_label_loader_eligible"])
        self.assertFalse(completion["gate_trained"])
        self.assertFalse(completion["confirm_access_authorized"])

    def test_two_physical_cache_schemas_and_local_first_order_are_exact(self) -> None:
        formats = self.config["cache_formats"]
        self.assertNotEqual(formats["feature"]["archive_path"], formats["label"]["archive_path"])
        self.assertEqual(formats["common_ustar"]["float_encoding"], "ieee754_binary64_big_endian_lowercase_hex")
        self.assertEqual(formats["common_ustar"]["float_hex_pattern"], "[0-9a-f]{16}")
        self.assertFalse(formats["physical_separation"]["combined_cache_allowed"])
        states = self.config["local_first_state_machine"]["ordered_states"]
        self.assertEqual([state["name"] for state in states[:3]], [
            "global_claim",
            "feature_completion",
            "label_access_claim",
        ])
        self.assertEqual(
            [state["formal_label_semantic_access_allowed_after"] for state in states[:3]],
            [False, False, True],
        )
        self.assertEqual(states[-1]["status"], FINAL_COMPLETION_STATUS)
        self.assertTrue(self.config["local_first_state_machine"]["local_first_before_any_hf_output_mutation"])

    def test_source_a_and_execution_b_boundary_is_explicit(self) -> None:
        freeze = self.config["source_freeze"]
        source_a_remote = freeze["source_a_remote_validation"]
        self.assertEqual(source_a_remote["network_call_count"], 0)
        self.assertEqual(
            source_a_remote["remote_tracking_ref"], "refs/remotes/origin/main"
        )
        self.assertEqual(
            source_a_remote["required_equalities"], ["HEAD", "origin/main"]
        )
        execution_b_remote = freeze["execution_b_live_remote_validation"]
        self.assertTrue(execution_b_remote["required"])
        self.assertEqual(execution_b_remote["network_call_count"], 1)
        self.assertEqual(
            execution_b_remote["argv"],
            [
                "git",
                "ls-remote",
                "--exit-code",
                "origin",
                "refs/heads/main",
            ],
        )
        self.assertEqual(
            execution_b_remote["required_equalities"],
            ["HEAD", "origin/main", "live origin refs/heads/main"],
        )
        required = freeze["required_source_a_paths"]
        for path in (
            "code/causalcache/gate_v1_formal_cache.py",
            "code/causalcache/gate_v1_data.py",
            "code/tests/test_gate_v1_formal_cache.py",
            "code/tests/test_gate_v1_pipeline.py",
            "code/causalcache/gate_v1_formal_cache_runner.py",
            "code/scripts/manage_gate_v1_formal_cache.py",
            "code/tests/test_gate_v1_formal_cache_runner.py",
        ):
            self.assertIn(path, required)
        pending = freeze["execution_b_runner_freeze"]
        self.assertEqual(pending["path"], RUNNER_FREEZE_B_PATH)
        self.assertTrue(pending["must_be_absent_during_source_only_validation"])
        self.assertTrue(pending["bind_required_source_a_paths"])
        self.assertTrue(pending["bind_git_prerequisites"])
        self.assertTrue(pending["bind_source_a_inventory_sha256"])
        self.assertFalse(
            self.config["authorization"][
                "execution_allowed_without_machine_generated_runner_freeze_b"
            ]
        )
        self.assertTrue(
            all(value is False for value in self.config["authorization"].values())
        )
        gate_data = next(
            record
            for record in freeze["git_prerequisites"]
            if record["path"] == "code/causalcache/gate_v1_data.py"
        )
        self.assertEqual(gate_data["size_bytes"], 26772)
        self.assertEqual(
            gate_data["sha256"],
            "36764937c1e9e8ed2c6e771aeecc215c7cd409d0c8e19fa5af3147412784f8e4",
        )

    def test_execution_operation_counts_are_exact_and_non_scientific_work_is_zero(self) -> None:
        counts = self.config["execution_expected_operation_contract"]
        expected_positive = {
            "trajectory_semantic_decode_count": 58,
            "feature_state_count": 174,
            "ocr_semantic_decode_count": 290,
            "label_state_semantic_decode_count": 174,
            "distance_value_decode_count": 1624,
            "join_validation_count": 174,
            "candidate_feature_count": 522,
            "conditional_edge_count": 1682,
            "independent_target_count": 522,
        }
        for key, value in expected_positive.items():
            self.assertEqual(counts[key], value)
        for key in set(counts) - set(expected_positive):
            self.assertEqual(counts[key], 0)

    def test_mutating_any_security_boundary_fails_closed(self) -> None:
        mutations = []
        changed = copy.deepcopy(self.config)
        changed["parent_gate_contract"]["sha256"] = "0" * 64
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["input_artifacts"]["legacy_restoration_labels"]["immutable_revision"] = "0" * 40
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["formal_geometry"]["distance_value_count"] = 1623
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["access_firewall"]["generic_full_artifact_readers_allowed"] = True
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["cache_formats"]["label"]["archive_path"] = changed["cache_formats"]["feature"]["archive_path"]
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["local_first_state_machine"]["ordered_states"][1:3] = reversed(
            changed["local_first_state_machine"]["ordered_states"][1:3]
        )
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["destination"]["private"] = False
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["runtime_contract"]["gpu_required"] = True
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["source_only_operation_contract"]["network_call_count"] = 1
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["source_freeze"]["execution_b_live_remote_validation"]["required"] = False
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["execution_expected_operation_contract"]["trajectory_semantic_decode_count"] = 59
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["authorization"]["execution_authorized"] = True
        mutations.append(changed)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_formal_cache_config(mutation, repository_root=ROOT)

    def test_git_prerequisite_byte_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prerequisites = self.config["source_freeze"]["git_prerequisites"]
            for record in prerequisites:
                destination = root / record["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / record["path"], destination)
            target = root / "code/configs/causalcache_gate_v1_preregistration.json"
            target.write_bytes(target.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "size|SHA256"):
                validate_formal_cache_config(self.config, repository_root=root)


if __name__ == "__main__":
    unittest.main()
