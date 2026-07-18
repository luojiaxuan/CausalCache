from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest import mock

from causalcache.gate_v1_fresh16_inventory_repair_contract import (
    CANONICAL_CONFIG_PATH,
    DERIVED_BASE_PATHS,
    DERIVED_FULL_INVENTORY_SHA256,
    DERIVED_HISTORICAL_AUXILIARY_PATHS,
    FROZEN_CONFIG_SHA256,
    NEW_ARTIFACT_DIRECTORY,
    NEW_DESTINATION_REPO,
    NEW_DESTINATION_TAG,
    NEW_EXECUTION_NAMESPACE,
    NEW_RUNTIME_RECEIPT_PATH,
    PARENT_CONFIG_SHA256,
    RUNNER_FREEZE_B_PATH,
    load_frozen_fresh16_inventory_repair_contract,
    validate_fresh16_inventory_repair_contract_data,
    validate_fresh16_inventory_repair_source_only_contract,
)
from causalcache.gate_v1_fresh16_evaluation_contract import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


class Fresh16InventoryRepairContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_frozen_fresh16_inventory_repair_contract(
            CANONICAL_CONFIG_PATH,
            repository_root=ROOT,
        )
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_frozen_overlay_fully_loads_parent_and_exposes_runner_interface(self) -> None:
        self.assertEqual(
            hashlib.sha256(CONFIG.read_bytes()).hexdigest(), FROZEN_CONFIG_SHA256
        )
        self.assertEqual(self.contract.parent.sha256, PARENT_CONFIG_SHA256)
        self.assertEqual(self.contract.sha256, FROZEN_CONFIG_SHA256)
        self.assertEqual(self.contract.model, self.contract.parent.model)
        self.assertEqual(self.contract.labels, self.contract.parent.labels)
        self.assertEqual(self.contract.geometry, self.contract.parent.geometry)
        self.assertEqual(self.contract.output, self.contract.parent.output)

    def test_full_remote_inventory_is_exact_disjoint_union(self) -> None:
        tree = self.contract.repair["derived_remote_tree"]
        consumed = tuple(item["path"] for item in tree["consumed_files"])
        base = tuple(tree["base_paths"])
        auxiliary = tuple(tree["historical_auxiliary_paths"])
        full = tuple(tree["full_inventory_paths"])
        self.assertEqual(base, DERIVED_BASE_PATHS)
        self.assertEqual(auxiliary, DERIVED_HISTORICAL_AUXILIARY_PATHS)
        self.assertTrue(set(consumed).isdisjoint(base))
        self.assertTrue(set(consumed).isdisjoint(auxiliary))
        self.assertTrue(set(base).isdisjoint(auxiliary))
        self.assertEqual(full, tuple(sorted((*consumed, *base, *auxiliary))))
        self.assertEqual(len(full), 15)
        self.assertEqual(
            hashlib.sha256(canonical_json_bytes(list(full))).hexdigest(),
            DERIVED_FULL_INVENTORY_SHA256,
        )
        self.assertEqual(
            self.contract.derived["remote_tree_full_inventory_paths"], list(full)
        )
        self.assertEqual(
            self.contract.derived["remote_tree_full_inventory_sha256"],
            DERIVED_FULL_INVENTORY_SHA256,
        )

    def test_effective_delta_is_only_declared_repair_surface(self) -> None:
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
                "derived_input",
                "local_first_state_machine",
                "runtime_contract",
                "destination",
            },
        )
        local = effective["local_first_state_machine"]
        self.assertEqual(local["execution_namespace"], NEW_EXECUTION_NAMESPACE)
        self.assertEqual(local["artifact_directory"], NEW_ARTIFACT_DIRECTORY)
        self.assertEqual(
            effective["runtime_contract"]["docker_inspect_receipt"]["path"],
            NEW_RUNTIME_RECEIPT_PATH,
        )
        self.assertEqual(effective["destination"]["repo"], NEW_DESTINATION_REPO)
        self.assertEqual(effective["destination"]["tag"], NEW_DESTINATION_TAG)
        self.assertEqual(effective["output_contract"], parent["output_contract"])
        self.assertEqual(
            effective["evaluation_contract"], parent["evaluation_contract"]
        )

    def test_parent_failure_binds_exact_old_receipts_log_and_absence(self) -> None:
        failure = self.contract.repair["parent_failure"]
        self.assertEqual(
            [item["state_name"] for item in failure["ordered_receipts"]],
            ["runtime_receipts", "global_claim"],
        )
        self.assertEqual(
            [item["size_bytes"] for item in failure["ordered_receipts"]],
            [2535, 858],
        )
        self.assertEqual(failure["run_evidence"]["log"]["size_bytes"], 1386)
        self.assertEqual(failure["run_evidence"]["started"]["size_bytes"], 21)
        self.assertEqual(failure["run_evidence"]["exit"]["size_bytes"], 2)
        self.assertEqual(failure["old_runtime_receipt"]["size_bytes"], 18117)
        self.assertTrue(failure["old_artifact_directory_expected_empty"])
        self.assertTrue(failure["old_destination"]["expected_absent"])
        self.assertEqual(failure["old_destination"]["remote_mutation_count"], 0)
        self.assertEqual(len(failure["expected_absent_successor_state_names"]), 14)

    def test_overlay_mutations_fail_closed(self) -> None:
        mutations = (
            (
                lambda value: value["parent_contract"].__setitem__(
                    "sha256", "0" * 64
                ),
                "parent contract",
            ),
            (
                lambda value: value["source_lineage"].__setitem__(
                    "parent_execution_b_git_commit", "0" * 40
                ),
                "parent_execution_b_git_commit",
            ),
            (
                lambda value: value["inventory_repair"]["scope"].__setitem__(
                    "scientific_protocol_changed", True
                ),
                "scope",
            ),
            (
                lambda value: value["inventory_repair"]["parent_failure"][
                    "ordered_receipts"
                ][0].__setitem__("sha256", "0" * 64),
                "receipt",
            ),
            (
                lambda value: value["inventory_repair"]["derived_remote_tree"][
                    "historical_auxiliary_paths"
                ].pop(),
                "historical auxiliary|full-15",
            ),
            (
                lambda value: value["inventory_repair"]["derived_remote_tree"][
                    "full_inventory_paths"
                ].append("unexpected/path"),
                "full-15",
            ),
            (
                lambda value: value["effective_overrides"]["destination"].__setitem__(
                    "repo", "gavinlaw/wrong"
                ),
                "effective overrides",
            ),
            (
                lambda value: value["source_only_contract"].__setitem__(
                    "network_call_count", 1
                ),
                "source-only contract",
            ),
        )
        for mutate, message in mutations:
            value = copy.deepcopy(self.config)
            mutate(value)
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                validate_fresh16_inventory_repair_contract_data(
                    value,
                    parent=self.contract.parent,
                )

    def test_source_only_validator_has_zero_side_effect_contract(self) -> None:
        self.assertFalse((ROOT / RUNNER_FREEZE_B_PATH).exists())
        with (
            mock.patch("socket.socket", side_effect=AssertionError("network")),
            mock.patch.object(
                Path,
                "write_bytes",
                side_effect=AssertionError("write"),
            ),
            mock.patch.object(
                Path,
                "write_text",
                side_effect=AssertionError("write"),
            ),
        ):
            result = validate_fresh16_inventory_repair_source_only_contract(
                CANONICAL_CONFIG_PATH,
                repository_root=ROOT,
            )
        self.assertEqual(result["config_sha256"], FROZEN_CONFIG_SHA256)
        self.assertEqual(result["derived_remote_path_count"], 15)
        self.assertFalse(result["runner_freeze_b_present"])
        for key, value in result.items():
            if key.endswith("_count") and key != "derived_remote_path_count":
                self.assertEqual(value, 0, key)
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

    def test_source_only_validator_rejects_present_new_execution_b(self) -> None:
        original = __import__("os").path.lexists

        def fake_lexists(path: object) -> bool:
            if Path(path) == ROOT / RUNNER_FREEZE_B_PATH:
                return True
            return original(path)

        with mock.patch(
            "causalcache.gate_v1_fresh16_inventory_repair_contract.os.path.lexists",
            side_effect=fake_lexists,
        ), self.assertRaisesRegex(ValueError, "must be absent"):
            validate_fresh16_inventory_repair_source_only_contract(
                CANONICAL_CONFIG_PATH,
                repository_root=ROOT,
            )


if __name__ == "__main__":
    unittest.main()
