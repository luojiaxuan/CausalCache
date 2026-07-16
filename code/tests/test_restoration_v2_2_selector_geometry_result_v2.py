from __future__ import annotations

import itertools
import json
import unittest
from collections import Counter

from causalcache.restoration_v2_2_label_table import validate_complete_distance_table
from causalcache.restoration_v2_2_selector_geometry import SelectorGeometryState
from causalcache.restoration_v2_2_selector_geometry_result import (
    METHODS,
    build_state_records as build_legacy_state_records,
)
from causalcache.restoration_v2_2_selector_geometry_result_v2 import (
    EXPECTED_INTERACTION_FACTOR_CELL_COUNT,
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


def _independent_jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    left_set = frozenset(left)
    right_set = frozenset(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


class RestorationV22SelectorGeometryResultV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.states = _two_trajectory_fixture()
        cls.legacy_records = build_legacy_state_records(
            cls.states,
            enforce_formal_denominator=False,
        )
        cls.repaired_records = build_state_records(
            cls.states,
            enforce_formal_denominator=False,
        )
        cls.payload = build_selector_geometry_scientific_payload(
            cls.states,
            bootstrap_resamples=20,
            bootstrap_seed=7,
            enforce_formal_denominator=False,
        )

    def test_repair_preserves_legacy_selector_values(self) -> None:
        self.assertEqual(len(self.repaired_records), len(self.legacy_records))
        preserved_method_fields = (
            "selected_coalition",
            "utility",
            "normalized_recovery",
            "absolute_regret_to_exact_subset",
            "normalized_recovery_regret_to_exact_subset",
            "utility_ratio_to_exact_subset",
        )
        for legacy_record, repaired_record in zip(
            self.legacy_records,
            self.repaired_records,
            strict=True,
        ):
            self.assertEqual(repaired_record["state"], legacy_record["state"])
            self.assertEqual(
                repaired_record["budget_event_capacity"],
                legacy_record["budget_event_capacity"],
            )
            self.assertEqual(repaired_record["gaps"], legacy_record["gaps"])
            self.assertEqual(set(repaired_record["methods"]), set(METHODS))
            for method_name in METHODS:
                legacy_method = legacy_record["methods"][method_name]
                repaired_method = repaired_record["methods"][method_name]
                for field in preserved_method_fields:
                    self.assertEqual(
                        repaired_method[field],
                        legacy_method[field],
                        msg=f"{method_name}.{field}",
                    )

    def test_analytic_random_cardinality_match_and_jaccard(self) -> None:
        zero = next(
            record
            for record in self.repaired_records
            if record["state"]["candidate_event_count"] == 4
            and record["budget_event_capacity"] == 0
        )
        zero_random = zero["methods"]["analytic_exact_cardinality_random"]
        self.assertEqual(zero_random["selected_cardinality"], 0)
        self.assertEqual(zero_random["exact_cardinality"], 0)
        self.assertEqual(zero_random["expected_exact_coalition_match"], 1.0)
        self.assertEqual(zero_random["expected_jaccard_to_exact_coalition"], 1.0)
        self.assertEqual(zero_random["exact_coalition_match"], 1.0)
        self.assertEqual(zero_random["jaccard_to_exact_coalition"], 1.0)

        n4_b2 = next(
            record
            for record in self.repaired_records
            if record["state"]["role"] == "v2_label_train"
            and record["state"]["candidate_event_count"] == 4
            and record["budget_event_capacity"] == 2
        )
        random_method = n4_b2["methods"]["analytic_exact_cardinality_random"]
        exact = tuple(n4_b2["methods"]["exact_subset"]["selected_coalition"])
        event_ids = tuple(n4_b2["state"]["candidate_event_step_ids"])
        coalitions = tuple(itertools.combinations(event_ids, 2))
        expected_match = sum(coalition == exact for coalition in coalitions) / len(
            coalitions
        )
        expected_jaccard = sum(
            _independent_jaccard(coalition, exact) for coalition in coalitions
        ) / len(coalitions)
        self.assertEqual(random_method["selected_coalition"], None)
        self.assertEqual(random_method["selected_cardinality"], 2)
        self.assertEqual(random_method["exact_cardinality"], 2)
        self.assertEqual(random_method["coalition_count"], 6)
        self.assertAlmostEqual(
            random_method["expected_exact_coalition_match"],
            expected_match,
        )
        self.assertAlmostEqual(
            random_method["expected_jaccard_to_exact_coalition"],
            expected_jaccard,
        )
        self.assertAlmostEqual(expected_match, 1.0 / 6.0)
        self.assertAlmostEqual(expected_jaccard, 7.0 / 18.0)

    def test_interaction_factor_cross_product_and_empty_schema(self) -> None:
        report = self.payload["interaction_selector_reports"]
        cells = report["cells"]
        self.assertEqual(
            EXPECTED_INTERACTION_FACTOR_CELL_COUNT,
            144,
        )
        self.assertEqual(report["expected_cell_count"], 144)
        self.assertEqual(report["observed_cell_count"], 144)
        self.assertEqual(len(cells), 144)
        self.assertEqual(len({cell["cell_id"] for cell in cells}), 144)

        expected_summary_keys = {
            "status",
            "state_count",
            "trajectory_count",
            "mean_actual_utility",
            "mean_normalized_recovery",
            "recovery_ratio_of_means_to_exact_subset",
            "mean_selected_cardinality",
            "exact_coalition_match_rate",
            "mean_jaccard_to_exact_coalition",
            "selected_cardinality_histogram",
            "selection_metric_semantics",
        }
        for cell in cells:
            self.assertEqual(set(cell["methods"]), set(METHODS))
            for summary in cell["methods"].values():
                self.assertEqual(set(summary), expected_summary_keys)
                self.assertEqual(summary["state_count"], cell["state_count"])
                if cell["state_count"] == 0:
                    self.assertEqual(summary["status"], "EMPTY_STRATUM")
                    self.assertEqual(summary["trajectory_count"], 0)
                    self.assertIsNone(summary["mean_actual_utility"])
                    self.assertIsNone(summary["mean_normalized_recovery"])
                    self.assertEqual(summary["selected_cardinality_histogram"], {})
                else:
                    self.assertEqual(summary["status"], "MEASURED")

        grouped_counts = Counter()
        grouped_cells = Counter()
        for cell in cells:
            key = (
                cell["role"],
                cell["candidate_event_count"],
                cell["budget_event_capacity"],
            )
            grouped_counts[key] += cell["state_count"]
            grouped_cells[key] += 1
        for role in ("v2_label_train", "v2_development"):
            for event_count in (2, 3, 4):
                for budget in range(event_count + 1):
                    key = (role, event_count, budget)
                    self.assertEqual(grouped_cells[key], 6)
                    self.assertEqual(grouped_counts[key], 1)

    def test_payload_is_json_serializable(self) -> None:
        self.assertEqual(json.loads(json.dumps(self.payload)), self.payload)

    def test_development_sign_counts_match_trajectory_deltas(self) -> None:
        summaries = tuple(self.payload["slices"].values()) + tuple(
            self.payload["budget_curve"].values()
        )
        for summary in summaries:
            for comparison, deltas in summary[
                "development_trajectory_deltas"
            ].items():
                counts = summary["development_sign_counts"][comparison]
                observed = Counter(delta["sign"] for delta in deltas)
                self.assertEqual(counts["wins"], observed["win"])
                self.assertEqual(counts["ties"], observed["tie"])
                self.assertEqual(counts["losses"], observed["loss"])
                self.assertEqual(counts["count"], len(deltas))
                self.assertEqual(
                    counts["count"],
                    counts["wins"] + counts["ties"] + counts["losses"],
                )


if __name__ == "__main__":
    unittest.main()
