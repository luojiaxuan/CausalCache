from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

from causalcache import gate_v1_formal_train_contract as contract_module
from scripts.validate_gate_v1_formal_train_contract import _parser


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / contract_module.CANONICAL_CONFIG_PATH


class FormalTrainContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_frozen_contract_identity_and_public_api(self) -> None:
        contract = contract_module.load_frozen_formal_train_contract(
            CONFIG,
            repository_root=ROOT,
        )
        self.assertEqual(
            hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            contract_module.FROZEN_CONFIG_SHA256,
        )
        self.assertEqual(contract.sha256, contract_module.FROZEN_CONFIG_SHA256)
        self.assertEqual(contract.data["protocol_id"], contract_module.PROTOCOL_ID)
        self.assertEqual(contract.cache_input, contract.data["formal_cache_input"])
        self.assertEqual(contract.geometry, contract.data["formal_geometry"])
        self.assertEqual(contract.firewall, contract.data["access_firewall"])
        self.assertEqual(contract.training, contract.data["training_contract"])
        self.assertEqual(contract.output, contract.data["output_contract"])
        self.assertEqual(contract.destination, contract.data["destination"])
        self.assertEqual(contract.local_state, contract.data["local_first_state_machine"])
        self.assertEqual(contract.runtime, contract.data["runtime_contract"])

    def test_exact_cache_completion_and_preregistration_are_bound(self) -> None:
        lineage = self.config["lineage"]
        cache = self.config["formal_cache_input"]
        self.assertEqual(
            lineage["gate_preregistration"]["sha256"],
            contract_module.GATE_PREREGISTRATION_SHA256,
        )
        self.assertEqual(
            lineage["formal_cache_completion_result"]["sha256"],
            contract_module.CACHE_RESULT_SUMMARY_SHA256,
        )
        self.assertEqual(
            cache["immutable_revision"], contract_module.FORMAL_CACHE_REVISION
        )
        self.assertEqual(
            cache["join_only_audit_sha256"],
            contract_module.FORMAL_CACHE_JOIN_AUDIT_SHA256,
        )
        self.assertEqual(
            tuple(cache["exact_three_files"]),
            contract_module.EXPECTED_CACHE_FILES,
        )
        self.assertEqual(
            self.config["formal_geometry"]["source_ids_sha256"],
            contract_module.FORMAL_SOURCE_IDS_SHA256,
        )

    def test_complete_grid_two_commit_and_private_model_destination_are_exact(self) -> None:
        training = self.config["training_contract"]
        output = self.config["output_contract"]
        destination = self.config["destination"]
        self.assertEqual(training["oof_trial_count_total"], 20)
        self.assertEqual(training["fold_training_track_count_total"], 100)
        self.assertEqual(training["model_initialization_count_total"], 110)
        self.assertEqual(training["persistent_checkpoint_count"], 10)
        self.assertEqual(
            tuple(output["payload_commit"]["exact_twelve_targets"]),
            contract_module.PAYLOAD_TARGETS,
        )
        self.assertEqual(
            tuple(output["manifest_commit"]["exact_four_new_targets"]),
            contract_module.MANIFEST_TARGETS,
        )
        self.assertTrue(
            output["manifest_commit"][
                "ensemble_checkpoint_bindings_must_use_payload_commit"
            ]
        )
        self.assertTrue(
            output["manifest_commit"]["manifest_files_must_not_embed_manifest_commit"]
        )
        self.assertEqual(destination["repo_type"], "model")
        self.assertTrue(destination["private"])
        self.assertEqual(
            destination["repo"],
            "gavinlaw/causalcache-gate-v1-formal58-selector-mobile",
        )
        self.assertEqual(destination["tag"], "gate-v1-formal58-train-v1")
        self.assertTrue(destination["tag_must_resolve_to_manifest_commit"])

    def test_source_and_execution_firewalls_keep_all_future_splits_sealed(self) -> None:
        firewall = self.config["access_firewall"]
        for key in (
            "source_a_fresh16_semantic_decode_count",
            "source_a_legacy_dev5_semantic_decode_count",
            "source_a_confirm20_access_count",
            "source_a_matched_nll_evaluation_count",
            "source_a_closed_loop_episode_count",
            "execution_b_fresh16_semantic_decode_count",
            "execution_b_legacy_dev5_semantic_decode_count",
            "execution_b_confirm20_access_count",
            "execution_b_matched_nll_evaluation_count",
            "execution_b_closed_loop_episode_count",
        ):
            self.assertEqual(firewall[key], 0)
        self.assertEqual(firewall["execution_b_allowed_semantic_roster"], "formal_train")
        self.assertTrue(firewall["formal_checkpoint_completion_before_any_development_access"])
        self.assertTrue(all(value is False for value in self.config["authorization"].values()))

    def test_runtime_is_pinned_cpu_fp32_no_gpu(self) -> None:
        runtime = self.config["runtime_contract"]
        self.assertEqual(runtime["device"], "cpu")
        self.assertEqual(runtime["dtype"], "float32")
        self.assertFalse(runtime["gpu_required"])
        self.assertEqual(runtime["normalized_device_requests"], [])
        self.assertFalse(runtime["container_privileged"])
        self.assertEqual(runtime["nvidia_visible_devices"], "void")
        self.assertEqual(runtime["cuda_visible_devices"], "")
        self.assertFalse(runtime["torch_cuda_available"])
        self.assertEqual(runtime["torch_cuda_device_count"], 0)
        self.assertEqual(runtime["python_version"], "3.12.3")
        self.assertEqual(runtime["torch_version"], "2.11.0+cu130")
        self.assertEqual(runtime["safetensors_version"], "0.7.0")
        self.assertEqual(runtime["huggingface_hub_version"], "1.16.1")
        self.assertEqual(runtime["torch_num_threads"], 1)
        self.assertEqual(runtime["torch_num_interop_threads"], 1)
        receipt = runtime["docker_inspect_receipt"]
        self.assertEqual(receipt["normalized_device_requests"], [])
        self.assertFalse(receipt["privileged"])
        self.assertEqual(receipt["runtime"], "runc")
        self.assertEqual(receipt["mode"], 0o600)

    def test_mutating_security_science_or_authorization_fails_closed(self) -> None:
        mutations = (
            ("formal_cache_input", "immutable_revision", "0" * 40),
            ("formal_geometry", "trajectory_count", 57),
            ("access_firewall", "execution_b_fresh16_semantic_decode_count", 1),
            ("training_contract", "maximum_epochs", 499),
            ("destination", "private", False),
            ("runtime_contract", "torch_num_threads", 2),
            ("authorization", "execution_authorized", True),
        )
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(self.config)
                changed[section][key] = value
                with self.assertRaisesRegex(ValueError, "canonical SHA256"):
                    contract_module.validate_formal_train_contract_data(changed)

    def test_source_only_validator_reports_no_execution_or_access(self) -> None:
        fake_inventory = tuple(
            {
                "path": path,
                "sha256": hashlib.sha256(path.encode()).hexdigest(),
                "size_bytes": len(path),
            }
            for path in contract_module.REQUIRED_SOURCE_A_PATHS
        )
        before = frozenset(sys.modules)
        with mock.patch.object(
            contract_module,
            "_source_inventory",
            return_value=fake_inventory,
        ):
            result = contract_module.validate_formal_train_source_only_contract(
                CONFIG,
                repository_root=ROOT,
            )
        self.assertEqual(result["status"], contract_module.VALIDATION_STATUS)
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["execution_authorized"])
        self.assertFalse(result["runner_freeze_b_present"])
        for key in (
            "formal58_cache_access_authorized",
            "formal58_training_authorized",
            "checkpoint_publication_authorized",
            "gate_trained",
            "fresh16_access_authorized",
            "legacy_dev5_access_authorized",
            "confirm20_access_authorized",
            "matched_nll_authorized",
            "closed_loop_authorized",
        ):
            self.assertFalse(result[key])
        for key in (
            "network_call_count",
            "hf_api_call_count",
            "file_write_count",
            "torch_import_count",
            "formal58_semantic_decode_count",
            "fresh16_semantic_decode_count",
            "legacy_dev5_semantic_decode_count",
            "confirm20_access_count",
            "matched_nll_evaluation_count",
            "closed_loop_episode_count",
        ):
            self.assertEqual(result[key], 0)
        newly_loaded = frozenset(sys.modules) - before
        self.assertFalse(any(name == "torch" or name.startswith("torch.") for name in newly_loaded))
        self.assertFalse(
            any(
                name == "huggingface_hub" or name.startswith("huggingface_hub.")
                for name in newly_loaded
            )
        )

    def test_source_validator_rejects_runner_freeze_b_in_source_a(self) -> None:
        original_lexists = contract_module.os.path.lexists

        def fake_lexists(path: str | Path) -> bool:
            if str(path).endswith(contract_module.RUNNER_FREEZE_B_PATH):
                return True
            return original_lexists(path)

        with (
            mock.patch.object(contract_module.os.path, "lexists", side_effect=fake_lexists),
            self.assertRaisesRegex(ValueError, "must remain absent"),
        ):
            contract_module.validate_formal_train_source_only_contract(
                CONFIG,
                repository_root=ROOT,
            )

    def test_source_path_inventory_anticipates_core_runner_and_tests_but_not_b(self) -> None:
        paths = tuple(self.config["source_freeze"]["required_source_a_paths"])
        self.assertEqual(paths, contract_module.REQUIRED_SOURCE_A_PATHS)
        self.assertIn("code/causalcache/gate_v1_formal_train.py", paths)
        self.assertIn("code/causalcache/gate_v1_formal_train_runner.py", paths)
        self.assertIn("code/scripts/manage_gate_v1_formal_train.py", paths)
        self.assertIn("code/tests/test_gate_v1_formal_train_runner.py", paths)
        self.assertNotIn(contract_module.RUNNER_FREEZE_B_PATH, paths)
        self.assertFalse((ROOT / contract_module.RUNNER_FREEZE_B_PATH).exists())

    def test_source_only_cli_has_no_execution_surface(self) -> None:
        parser = _parser()
        args = parser.parse_args([])
        self.assertEqual(args.contract, Path(contract_module.CANONICAL_CONFIG_PATH))
        destinations = {action.dest for action in parser._actions}
        self.assertNotIn("hf_token_file", destinations)
        self.assertNotIn("data_root", destinations)
        self.assertNotIn("output", destinations)
        self.assertNotIn("execution_b_git_commit", destinations)


if __name__ == "__main__":
    unittest.main()
