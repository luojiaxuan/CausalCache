from __future__ import annotations

import copy
import itertools
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_label_table import validate_complete_distance_table
from causalcache.restoration_v2_2_selector_geometry import SelectorGeometryState
from causalcache.restoration_v2_2_selector_geometry_contract_v2 import (
    CANONICAL_CONFIG_PATH,
    EXPECTED_CELL_COUNT,
    EXPECTED_RECORD_COUNT,
    EXPECTED_SCIENCE_STATUS,
    FROZEN_CONFIG_SHA256,
    METHODS,
    PARENT_CONFIG_PATH,
    PARENT_CONFIG_SHA256,
    PROTOCOL_ID,
    RestorationV22SelectorGeometryRepairContract,
    sha256_file,
    validate_contract,
    validate_repair_preserves_legacy_payload,
    validate_repaired_scientific_payload,
)
from causalcache.restoration_v2_2_selector_geometry_result import (
    build_selector_geometry_scientific_payload as build_legacy_payload,
)
from causalcache.restoration_v2_2_selector_geometry_result_v2 import (
    build_selector_geometry_scientific_payload as build_repaired_payload,
)
from scripts import run_restoration_v2_2_selector_geometry_v2 as runner


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH
PARENT_CONFIG = ROOT / PARENT_CONFIG_PATH


def _state(
    *,
    index: int,
    role: str,
    trajectory_id: str,
    event_count: int,
    trajectory_index: int,
) -> SelectorGeometryState:
    event_ids = tuple(range(1, event_count + 1))
    offset = float(trajectory_index + 1) / 100.0
    weights = {
        event_id: float(event_count + 1 - event_id) + offset
        for event_id in event_ids
    }
    distances = {}
    for size in range(event_count + 1):
        for coalition in itertools.combinations(event_ids, size):
            utility = sum(weights[event_id] for event_id in coalition)
            distances[coalition] = 100.0 - utility
    decision_step = event_count + 2
    return SelectorGeometryState(
        member_name=f"workers/even/states/{index:03d}.json",
        index=index,
        role=role,
        trajectory_id=trajectory_id,
        state_id=f"{trajectory_id}:decision_step:{decision_step:03d}",
        decision_step_id=decision_step,
        candidate_event_step_ids=event_ids,
        table=validate_complete_distance_table(event_ids, distances),
    )


def _formal_45_state_fixture() -> tuple[SelectorGeometryState, ...]:
    states = []
    index = 0
    specifications = (
        ("v2_label_train", "train", 10),
        ("v2_development", "dev", 5),
    )
    for role, prefix, trajectory_count in specifications:
        for trajectory_index in range(trajectory_count):
            trajectory_id = f"{prefix}-{trajectory_index:02d}"
            for event_count in (2, 3, 4):
                states.append(
                    _state(
                        index=index,
                        role=role,
                        trajectory_id=trajectory_id,
                        event_count=event_count,
                        trajectory_index=trajectory_index,
                    )
                )
                index += 1
    return tuple(states)


class RestorationV22SelectorGeometryContractV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = RestorationV22SelectorGeometryRepairContract.load(
            CONFIG,
            repository_root=ROOT,
        )
        cls.states = _formal_45_state_fixture()
        cls.repaired = build_repaired_payload(
            cls.states,
            bootstrap_resamples=1,
            bootstrap_seed=7,
        )
        cls.legacy = build_legacy_payload(
            cls.states,
            bootstrap_resamples=1,
            bootstrap_seed=7,
        )

    def test_canonical_repair_and_parent_contract_hashes_load(self) -> None:
        self.assertEqual(sha256_file(CONFIG), FROZEN_CONFIG_SHA256)
        self.assertEqual(sha256_file(PARENT_CONFIG), PARENT_CONFIG_SHA256)
        self.assertEqual(self.contract.path, CONFIG.resolve())
        self.assertEqual(self.contract.sha256, FROZEN_CONFIG_SHA256)
        self.assertEqual(self.contract.parent.path, PARENT_CONFIG.resolve())
        self.assertEqual(self.contract.parent.sha256, PARENT_CONFIG_SHA256)
        self.assertEqual(self.contract.data["protocol_id"], PROTOCOL_ID)
        self.assertEqual(
            self.contract.data["parent_contract"]["sha256"],
            PARENT_CONFIG_SHA256,
        )
        source_validation = validate_contract(CONFIG, repository_root=ROOT)
        self.assertEqual(
            source_validation["status"],
            "VALID_RESTORATION_V2_2_SELECTOR_GEOMETRY_V2_REPAIR_CONTRACT",
        )
        self.assertEqual(
            source_validation["expected_interaction_factor_cell_count"],
            144,
        )

    def test_formal_payload_validates_and_preserves_legacy_values(self) -> None:
        self.assertEqual(len(self.states), 45)
        self.assertEqual(
            len(self.repaired["state_budget_records"]),
            EXPECTED_RECORD_COUNT,
        )
        validate_repaired_scientific_payload(self.repaired)
        validate_repair_preserves_legacy_payload(self.repaired, self.legacy)

    def test_missing_cell_or_cell_method_fails_closed(self) -> None:
        without_cell = copy.deepcopy(self.repaired)
        without_cell["interaction_selector_reports"]["cells"].pop()
        with self.assertRaises(ValueError):
            validate_repaired_scientific_payload(without_cell)

        without_method = copy.deepcopy(self.repaired)
        first_cell = without_method["interaction_selector_reports"]["cells"][0]
        first_cell["methods"].pop(METHODS[0])
        with self.assertRaises(ValueError):
            validate_repaired_scientific_payload(without_method)

    def test_random_cardinality_drift_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.repaired)
        random_method = mutated["state_budget_records"][0]["methods"][
            "analytic_exact_cardinality_random"
        ]
        random_method["exact_cardinality"] += 1
        with self.assertRaises(ValueError):
            validate_repaired_scientific_payload(mutated)

    def test_development_assignment_and_train_cutpoint_drift_fail_closed(self) -> None:
        assignment_drift = copy.deepcopy(self.repaired)
        assignment = next(
            value
            for value in assignment_drift["interaction_state_assignments"]
            if value["role"] == "v2_development"
        )
        assignment["interaction_strength_stratum"] = (
            "high"
            if assignment["interaction_strength_stratum"] != "high"
            else "low"
        )
        with self.assertRaises(ValueError):
            validate_repaired_scientific_payload(assignment_drift)

        cutpoint_drift = copy.deepcopy(self.repaired)
        cutpoint_drift["train_interaction_tertile_cutpoints_by_n"]["2"][
            "lower_tertile"
        ] += 0.5
        with self.assertRaises(ValueError):
            validate_repaired_scientific_payload(cutpoint_drift)

    def test_selector_utility_drift_breaks_legacy_preservation(self) -> None:
        mutated = copy.deepcopy(self.repaired)
        mutated["state_budget_records"][0]["methods"]["exact_subset"][
            "utility"
        ] += 1.0
        with self.assertRaises(ValueError):
            validate_repair_preserves_legacy_payload(mutated, self.legacy)

    def test_runner_identity_status_output_and_source_paths_are_frozen(self) -> None:
        self.assertEqual(runner.PROTOCOL_ID, PROTOCOL_ID)
        self.assertEqual(
            runner.RUN_STATUS,
            "COMPLETED_RESTORATION_V2_2_SELECTOR_GEOMETRY_V2_REPAIR",
        )
        self.assertEqual(
            runner.VALID_STATUS,
            "VALID_RESTORATION_V2_2_SELECTOR_GEOMETRY_V2_REPAIR",
        )
        self.assertEqual(
            self.repaired["status"],
            EXPECTED_SCIENCE_STATUS,
        )
        self.assertEqual(
            self.contract.data["output_contract"]["canonical_result_directory"],
            "data/results/restoration_v2_2_selector_geometry_v2_repair",
        )
        self.assertEqual(
            self.contract.data["output_contract"]["superseded_result_directory"],
            "data/results/restoration_v2_2_selector_geometry_v1",
        )
        self.assertEqual(
            len(set(runner.FORMAL_SOURCE_PATHS)),
            len(runner.FORMAL_SOURCE_PATHS),
        )
        self.assertIn(CANONICAL_CONFIG_PATH, runner.FORMAL_SOURCE_PATHS)
        self.assertIn(
            "code/causalcache/restoration_v2_2_selector_geometry_contract_v2.py",
            runner.FORMAL_SOURCE_PATHS,
        )
        self.assertIn(
            "code/causalcache/restoration_v2_2_selector_geometry_result_v2.py",
            runner.FORMAL_SOURCE_PATHS,
        )
        self.assertIn(
            "code/scripts/run_restoration_v2_2_selector_geometry_v2.py",
            runner.FORMAL_SOURCE_PATHS,
        )
        self.assertIn(
            "code/scripts/validate_restoration_v2_2_selector_geometry_v2_contract.py",
            runner.FORMAL_SOURCE_PATHS,
        )
        self.assertEqual(
            len(self.repaired["interaction_selector_reports"]["cells"]),
            EXPECTED_CELL_COUNT,
        )
        for path in runner.FORMAL_SOURCE_PATHS:
            self.assertTrue((ROOT / path).is_file(), msg=path)


if __name__ == "__main__":
    unittest.main()
