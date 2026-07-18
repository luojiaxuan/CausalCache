from __future__ import annotations

import copy
import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from causalcache.gate_v1_fresh16_claim_serialization_repair_contract import (
    CANONICAL_CONFIG_PATH,
    FROZEN_CONFIG_SHA256,
    NEW_ARTIFACT_DIRECTORY,
    NEW_DESTINATION_REPO,
    NEW_EXECUTION_NAMESPACE,
    NEW_RUNTIME_RECEIPT_PATH,
    PARENT_CONFIG_SHA256,
    RUNNER_FREEZE_B_PATH,
    load_frozen_fresh16_claim_serialization_repair_contract,
    validate_fresh16_claim_serialization_repair_contract_data,
    validate_fresh16_claim_serialization_repair_source_only_contract,
)
from causalcache.gate_v1_fresh16_evaluation_contract import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


class Fresh16ClaimSerializationRepairContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_frozen_fresh16_claim_serialization_repair_contract(
            repository_root=ROOT
        )
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_frozen_overlay_binds_parent_failure_and_new_identities(self) -> None:
        self.assertEqual(
            hashlib.sha256(CONFIG.read_bytes()).hexdigest(), FROZEN_CONFIG_SHA256
        )
        self.assertEqual(self.contract.parent.sha256, PARENT_CONFIG_SHA256)
        self.assertEqual(self.contract.sha256, FROZEN_CONFIG_SHA256)
        self.assertEqual(
            self.contract.data["local_first_state_machine"]["execution_namespace"],
            NEW_EXECUTION_NAMESPACE,
        )
        self.assertEqual(
            self.contract.data["local_first_state_machine"]["artifact_directory"],
            NEW_ARTIFACT_DIRECTORY,
        )
        self.assertEqual(
            self.contract.data["runtime_contract"]["docker_inspect_receipt"]["path"],
            NEW_RUNTIME_RECEIPT_PATH,
        )
        self.assertEqual(self.contract.destination["repo"], NEW_DESTINATION_REPO)
        self.assertEqual(len(self.contract.repair["parent_failure"]["ordered_receipts"]), 8)

    def test_immediate_parent_delta_is_exactly_four_operational_surfaces(self) -> None:
        parent = self.contract.parent.data
        effective = self.contract.data
        changed = {
            key
            for key in parent
            if canonical_json_bytes(parent[key]) != canonical_json_bytes(effective[key])
        }
        self.assertEqual(
            changed,
            {
                "source_freeze",
                "local_first_state_machine",
                "runtime_contract",
                "destination",
            },
        )
        for key in (
            "derived_input",
            "formal_model_input",
            "label_input",
            "fresh_geometry",
            "evaluation_contract",
            "output_contract",
        ):
            self.assertEqual(
                canonical_json_bytes(effective[key]), canonical_json_bytes(parent[key]), key
            )

    def test_claim_fix_is_only_json_safe_deep_snapshot(self) -> None:
        scope = self.contract.repair["scope"]
        self.assertEqual(
            scope["fix_contract"],
            "return a canonical-JSON deep snapshot from LabelAccessClaim.claim",
        )
        self.assertFalse(scope["canonical_claim_bytes_changed"])
        self.assertFalse(scope["scientific_protocol_changed"])
        self.assertFalse(scope["old_run_continuation_allowed"])

    def test_machine_runner_freeze_remains_a_single_mechanical_b_diff(self) -> None:
        runner = self.contract.source["execution_b_runner_freeze"]
        self.assertEqual(runner["path"], RUNNER_FREEZE_B_PATH)
        for key in (
            "must_be_absent_during_source_only_validation",
            "only_allowed_execution_b_source_tree_diff",
            "direct_single_parent_child_of_source_a",
            "bind_required_source_a_paths",
            "bind_git_prerequisites",
            "bind_source_a_inventory_sha256",
            "bind_loaded_module_inventory_sha256",
        ):
            self.assertTrue(runner[key], key)
        prerequisites = self.contract.source["git_prerequisites"]
        self.assertEqual(
            prerequisites[-3]["path"],
            "code/configs/causalcache_gate_v1_fresh16_inventory_repair_runner_v1.json",
        )

    def test_config_mutations_fail_closed(self) -> None:
        mutations = (
            lambda value: value["parent_contract"].__setitem__("sha256", "0" * 64),
            lambda value: value["source_lineage"]["failure_summary"].__setitem__(
                "sha256", "0" * 64
            ),
            lambda value: value["claim_serialization_repair"]["scope"].__setitem__(
                "scientific_protocol_changed", True
            ),
            lambda value: value["claim_serialization_repair"]["parent_failure"][
                "ordered_receipts"
            ][3].__setitem__("size_bytes", 1),
            lambda value: value["claim_serialization_repair"]["parent_failure"][
                "artifact_tree"
            ].__setitem__("canonical_inventory_sha256", "0" * 64),
            lambda value: value["effective_overrides"]["destination"].__setitem__(
                "repo", "gavinlaw/wrong"
            ),
            lambda value: value["source_only_contract"].__setitem__(
                "network_call_count", 1
            ),
        )
        for mutate in mutations:
            value = copy.deepcopy(self.config)
            mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_fresh16_claim_serialization_repair_contract_data(
                    value,
                    parent=self.contract.parent,
                )

    def test_source_only_validator_has_no_network_write_torch_or_authority(self) -> None:
        self.assertFalse((ROOT / RUNNER_FREEZE_B_PATH).exists())
        with (
            mock.patch("socket.socket", side_effect=AssertionError("network")),
            mock.patch.object(Path, "write_bytes", side_effect=AssertionError("write")),
            mock.patch.object(Path, "write_text", side_effect=AssertionError("write")),
        ):
            result = validate_fresh16_claim_serialization_repair_source_only_contract(
                repository_root=ROOT
            )
        self.assertEqual(result["config_sha256"], FROZEN_CONFIG_SHA256)
        self.assertEqual(result["derived_remote_path_count"], 15)
        self.assertFalse(result["runner_freeze_b_present"])
        for key in (
            "evaluation_executed",
            "execution_authorized",
            "fresh16_access_authorized",
            "label_access_authorized",
            "legacy_dev5_access_authorized",
            "confirm20_access_authorized",
            "matched_nll_authorized",
            "closed_loop_authorized",
        ):
            self.assertFalse(result[key], key)
        for key, value in result.items():
            if key.endswith("_count") and key != "derived_remote_path_count":
                self.assertEqual(value, 0, key)

    def test_source_only_rejects_present_execution_b(self) -> None:
        original = os.path.lexists

        def fake(path: object) -> bool:
            if Path(path) == ROOT / RUNNER_FREEZE_B_PATH:
                return True
            return original(path)

        with mock.patch(
            "causalcache.gate_v1_fresh16_claim_serialization_repair_contract.os.path.lexists",
            side_effect=fake,
        ), self.assertRaisesRegex(ValueError, "must be absent"):
            validate_fresh16_claim_serialization_repair_source_only_contract(
                repository_root=ROOT
            )


if __name__ == "__main__":
    unittest.main()
