from __future__ import annotations

import itertools
import json
import unittest

from causalcache.restoration_v2_2_label_table import validate_complete_distance_table
from causalcache.restoration_v2_2_selector_geometry import SelectorGeometryState
from causalcache.restoration_v2_2_selector_geometry_result import (
    build_selector_geometry_scientific_payload,
    build_state_records,
)


def _state(
    *,
    index: int,
    role: str,
    trajectory_id: str,
    event_count: int,
) -> SelectorGeometryState:
    event_ids = tuple(range(1, event_count + 1))
    weights = {event_id: float(event_count + 1 - event_id) for event_id in event_ids}
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


def _two_trajectory_fixture() -> tuple[SelectorGeometryState, ...]:
    result = []
    index = 0
    for role, trajectory_id in (
        ("v2_label_train", "train-a"),
        ("v2_development", "dev-a"),
    ):
        for event_count in (2, 3, 4):
            result.append(
                _state(
                    index=index,
                    role=role,
                    trajectory_id=trajectory_id,
                    event_count=event_count,
                )
            )
            index += 1
    return tuple(result)


class RestorationV22SelectorGeometryResultTest(unittest.TestCase):
    def test_state_grid_includes_zero_through_n_and_forced_fill_methods(self) -> None:
        states = _two_trajectory_fixture()
        records = build_state_records(
            states,
            enforce_formal_denominator=False,
        )
        self.assertEqual(len(records), 24)
        first_state = [
            record
            for record in records
            if record["state"]["state_id"] == states[0].state_id
        ]
        self.assertEqual(
            [record["budget_event_capacity"] for record in first_state],
            [0, 1, 2],
        )
        zero = first_state[0]
        self.assertEqual(zero["methods"]["exact_subset"]["normalized_recovery"], 0.0)
        self.assertIsNone(
            zero["methods"]["exact_subset"]["utility_ratio_to_exact_subset"]
        )
        self.assertIn("exact_cardinality_oracle", zero["methods"])
        self.assertIn("forced_fill_true_conditional_greedy", zero["methods"])
        self.assertIn(
            "forced_fill_budget_conditioned_independent",
            zero["methods"],
        )

    def test_payload_is_json_safe_and_trajectory_first_with_frozen_slices(self) -> None:
        payload = build_selector_geometry_scientific_payload(
            _two_trajectory_fixture(),
            bootstrap_resamples=20,
            bootstrap_seed=7,
            enforce_formal_denominator=False,
        )
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(payload["denominator"]["state_count"], 6)
        self.assertEqual(payload["denominator"]["trajectory_count"], 2)
        self.assertEqual(payload["denominator"]["state_budget_record_count"], 24)
        self.assertEqual(payload["bootstrap"]["resamples"], 20)
        self.assertEqual(payload["bootstrap"]["seed"], 7)
        self.assertEqual(
            sorted(payload["slices"]),
            [
                "hard_n_ge3_b2",
                "overall_b2_secondary",
                "primary_n4_b2",
                "secondary_n3_b2",
            ],
        )
        self.assertEqual(len(payload["budget_curve"]), 12)
        primary = payload["slices"]["primary_n4_b2"]
        exact_dev = primary["methods"]["exact_subset"]["v2_development"]
        self.assertEqual(exact_dev["trajectory_count"], 1)
        self.assertEqual(exact_dev["state_count"], 1)
        self.assertAlmostEqual(exact_dev["mean_normalized_recovery"], 0.07)
        projection = primary["paired_bootstrap"]["objective_projection_gap"]
        self.assertEqual(projection["development"]["mean_difference"], 0.0)
        self.assertEqual(
            payload["internal_method_shaping"]["conditioning_decision"],
            "prefer_independent_gate",
        )
        self.assertEqual(
            payload["internal_method_shaping"]["search_decision"],
            "online_greedy_sufficient",
        )
        self.assertEqual(
            payload["visual_baselines"]["ocr_and_rgb_similarity"],
            "pending_feature_stage",
        )
        self.assertTrue(
            all(value == 0 for value in payload["operation_counts"].values())
        )

    def test_formal_denominator_rejects_tiny_fixture(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 45"):
            build_state_records(_two_trajectory_fixture())


if __name__ == "__main__":
    unittest.main()
