from __future__ import annotations

import itertools
import unittest
from pathlib import Path
from unittest.mock import patch

from causalcache.restoration_v2_2_eager_artifact import pretty_json_bytes
from causalcache.restoration_v2_2_label_artifact import LabelEvidence
from causalcache.restoration_v2_2_label_contract import V1_ATTEMPT_PROFILE
from causalcache.restoration_v2_2_label_table import validate_complete_distance_table
from causalcache.restoration_v2_2_selector_geometry import (
    SelectorGeometryState,
    analytic_exact_cardinality_random,
    budget_conditioned_independent_selector,
    dynamic_recent_selector,
    evaluate_all_feasible_budgets,
    evaluate_selector_geometry_state,
    exact_cardinality_oracle,
    forced_fill_budget_conditioned_independent_selector,
    forced_fill_full_shapley_independent_selector,
    forced_fill_true_conditional_greedy_selector,
    full_shapley_independent_selector,
    load_selector_geometry_states,
    selector_geometry_states,
    state_interaction_metrics,
    true_conditional_greedy_selector,
)


def _distance_rows(
    event_ids: tuple[int, ...],
    utilities: dict[tuple[int, ...], float],
    *,
    baseline: float = 100.0,
) -> list[dict[str, object]]:
    expected = {
        subset
        for size in range(len(event_ids) + 1)
        for subset in itertools.combinations(event_ids, size)
    }
    if set(utilities) != expected:
        raise AssertionError("test utility table must cover the complete power set")
    return [
        {
            "coalition_event_step_ids": list(coalition),
            "distance_kl": baseline - utilities[coalition],
        }
        for coalition in sorted(utilities, key=lambda item: (len(item), item))
    ]


def _state_bytes(
    *,
    index: int,
    role: str,
    trajectory_id: str,
    decision_step_id: int,
    event_ids: tuple[int, ...],
    utilities: dict[tuple[int, ...], float],
) -> bytes:
    return pretty_json_bytes(
        {
            "state": {
                "index": index,
                "role": role,
                "trajectory_id": trajectory_id,
                "state_id": (
                    f"{trajectory_id}:decision_step:{decision_step_id:03d}"
                ),
                "decision_step_id": decision_step_id,
                "candidate_event_step_ids": list(event_ids),
            },
            "distance_rows": _distance_rows(event_ids, utilities),
        }
    )


def _evidence(files: dict[str, bytes]) -> LabelEvidence:
    return LabelEvidence(
        files=files,
        source_git_commit="1" * 40,
        run_contract_sha256="2" * 64,
        outcome=V1_ATTEMPT_PROFILE.pass_outcome,
        aggregate={},
        inventory=(),
        tree_inventory_sha256="3" * 64,
        profile=V1_ATTEMPT_PROFILE,
    )


def _geometry_state(
    event_ids: tuple[int, ...],
    utilities: dict[tuple[int, ...], float],
    *,
    baseline: float = 100.0,
) -> SelectorGeometryState:
    distances = {
        coalition: baseline - utility for coalition, utility in utilities.items()
    }
    return SelectorGeometryState(
        member_name="workers/even/states/000.json",
        index=0,
        role="v2_label_train",
        trajectory_id="trajectory-a",
        state_id="trajectory-a:decision_step:006",
        decision_step_id=6,
        candidate_event_step_ids=event_ids,
        table=validate_complete_distance_table(event_ids, distances),
    )


class RestorationV22SelectorGeometryLoaderTest(unittest.TestCase):
    def test_validated_archive_loader_preserves_metadata_and_sorts_by_index(
        self,
    ) -> None:
        two_event_utility = {(): 0.0, (1,): 1.0, (2,): 2.0, (1, 2): 3.0}
        files = {
            "workers/odd/states/001.json": _state_bytes(
                index=1,
                role="v2_development",
                trajectory_id="trajectory-b",
                decision_step_id=4,
                event_ids=(1, 2),
                utilities=two_event_utility,
            ),
            "workers/even/states/000.json": _state_bytes(
                index=0,
                role="v2_label_train",
                trajectory_id="trajectory-a",
                decision_step_id=4,
                event_ids=(1, 2),
                utilities=two_event_utility,
            ),
        }
        evidence = _evidence(files)
        with patch(
            "causalcache.restoration_v2_2_selector_geometry."
            "read_label_evidence_archive",
            return_value=evidence,
        ) as reader:
            states = load_selector_geometry_states(Path("validated.raw.tar"))
        reader.assert_called_once_with(Path("validated.raw.tar"))
        self.assertEqual([state.index for state in states], [0, 1])
        self.assertEqual(states[0].role, "v2_label_train")
        self.assertEqual(states[1].trajectory_id, "trajectory-b")
        self.assertEqual(states[1].state_id, "trajectory-b:decision_step:004")
        self.assertEqual(states[1].decision_step_id, 4)
        self.assertEqual(states[1].candidate_event_step_ids, (1, 2))
        self.assertEqual(states[1].table.utility((1, 2)), 3.0)

    def test_projection_rejects_duplicate_or_noncontiguous_indices(self) -> None:
        utility = {(): 0.0, (1,): 1.0, (2,): 2.0, (1, 2): 3.0}
        duplicate = _evidence(
            {
                "workers/even/states/000.json": _state_bytes(
                    index=0,
                    role="v2_label_train",
                    trajectory_id="a",
                    decision_step_id=4,
                    event_ids=(1, 2),
                    utilities=utility,
                ),
                "workers/odd/states/001.json": _state_bytes(
                    index=0,
                    role="v2_development",
                    trajectory_id="b",
                    decision_step_id=4,
                    event_ids=(1, 2),
                    utilities=utility,
                ),
            }
        )
        with self.assertRaisesRegex(ValueError, "unique indices"):
            selector_geometry_states(duplicate)


class RestorationV22SelectorGeometryMethodTest(unittest.TestCase):
    def test_budget_conditioned_and_full_shapley_are_distinct(self) -> None:
        event_ids = (1, 2, 3)
        utilities = {
            (): 0.0,
            (1,): 5.0,
            (2,): 4.0,
            (3,): 0.0,
            (1, 2): 50.0,
            (1, 3): 5.0,
            (2, 3): 50.0,
            (1, 2, 3): 50.0,
        }
        state = _geometry_state(event_ids, utilities)
        budget_selected, budget_scores = budget_conditioned_independent_selector(
            state,
            budget=1,
        )
        shapley_selected, shapley_scores = full_shapley_independent_selector(
            state,
            budget=1,
        )
        self.assertEqual(budget_scores, {1: 5.0, 2: 4.0, 3: 0.0})
        self.assertEqual(budget_selected, (1,))
        self.assertEqual(shapley_selected, (2,))
        self.assertGreater(shapley_scores[2], shapley_scores[1])

        record = evaluate_selector_geometry_state(state, budget=1)
        self.assertEqual(
            record["methods"]["exact_subset"]["selected_coalition"],
            [1],
        )
        self.assertEqual(
            record["methods"]["true_conditional_greedy"]["selected_coalition"],
            [1],
        )
        self.assertEqual(
            record["methods"]["full_shapley_independent"]["utility"],
            4.0,
        )
        self.assertEqual(
            record["gaps"]["full_shapley_projection_gap"]["raw_utility"],
            1.0,
        )
        self.assertAlmostEqual(
            record["gaps"]["full_shapley_projection_gap"][
                "normalized_recovery"
            ],
            0.01,
        )

    def test_true_greedy_uses_zero_epsilon_and_stops_on_nonpositive_gain(self) -> None:
        tiny = 1e-13
        event_ids = (1, 2)
        utilities = {
            (): 0.0,
            (1,): tiny,
            (2,): 0.0,
            (1, 2): 0.0,
        }
        state = _geometry_state(event_ids, utilities, baseline=1.0)
        greedy = true_conditional_greedy_selector(state, budget=2)
        self.assertEqual(greedy.coalition, frozenset({1}))
        self.assertEqual(len(greedy.trace), 1)
        self.assertAlmostEqual(greedy.trace[0].marginal_gain, tiny, places=16)

    def test_recent_random_normalization_regret_and_budget_curve(self) -> None:
        event_ids = (1, 2, 3)
        utilities = {
            (): 0.0,
            (1,): 2.0,
            (2,): 4.0,
            (3,): 6.0,
            (1, 2): 5.0,
            (1, 3): 8.0,
            (2, 3): 7.0,
            (1, 2, 3): 9.0,
        }
        state = _geometry_state(event_ids, utilities, baseline=10.0)
        self.assertEqual(dynamic_recent_selector(state, budget=2), (2, 3))
        random = analytic_exact_cardinality_random(state, budget=2)
        self.assertEqual(random["coalition_count"], 3)
        self.assertEqual(random["mean_utility"], 20.0 / 3.0)

        record = evaluate_selector_geometry_state(state, budget=2)
        exact = record["methods"]["exact_subset"]
        recent = record["methods"]["dynamic_recent"]
        random_method = record["methods"]["analytic_exact_cardinality_random"]
        self.assertEqual(exact["selected_coalition"], [1, 3])
        self.assertEqual(exact["normalized_recovery"], 0.8)
        self.assertEqual(recent["utility"], 7.0)
        self.assertEqual(recent["absolute_regret_to_exact_subset"], 1.0)
        self.assertAlmostEqual(random_method["normalized_recovery"], 2.0 / 3.0)
        self.assertEqual(
            [item["budget_event_capacity"] for item in evaluate_all_feasible_budgets(state)],
            [0, 1, 2, 3],
        )

    def test_forced_fill_variants_and_exact_cardinality_preserve_harm(self) -> None:
        event_ids = (1, 2)
        utilities = {(): 0.0, (1,): 2.0, (2,): 1.0, (1, 2): 0.5}
        state = _geometry_state(event_ids, utilities, baseline=10.0)
        exact_cardinality = exact_cardinality_oracle(state, budget=2)
        forced_greedy = forced_fill_true_conditional_greedy_selector(
            state,
            budget=2,
        )
        forced_budget, _ = forced_fill_budget_conditioned_independent_selector(
            state,
            budget=2,
        )
        forced_shapley, _ = forced_fill_full_shapley_independent_selector(
            state,
            budget=2,
        )
        self.assertEqual(exact_cardinality.coalition, frozenset({1, 2}))
        self.assertEqual(forced_greedy.coalition, frozenset({1, 2}))
        self.assertLess(forced_greedy.trace[-1].marginal_gain, 0.0)
        self.assertEqual(forced_budget, (1, 2))
        self.assertEqual(forced_shapley, (1, 2))

        record = evaluate_selector_geometry_state(state, budget=2)
        self.assertEqual(record["methods"]["exact_subset"]["utility"], 2.0)
        self.assertEqual(
            record["methods"]["exact_cardinality_oracle"]["utility"],
            0.5,
        )
        forced_gap = record["gaps"]["exact_cardinality_forced_fill_gap"]
        self.assertEqual(forced_gap["raw_utility"], 1.5)
        self.assertAlmostEqual(
            forced_gap["normalized_recovery"],
            0.15,
        )

    def test_interaction_metrics_separate_strict_and_one_e_minus_twelve_signs(
        self,
    ) -> None:
        event_ids = (1, 2)
        utilities = {
            (): 0.0,
            (1,): 1.0,
            (2,): 0.0,
            (1, 2): 1.0 - 5e-13,
        }
        state = _geometry_state(event_ids, utilities, baseline=10.0)
        metrics = state_interaction_metrics(state)
        self.assertEqual(metrics["interaction_count"], 1)
        self.assertTrue(metrics["has_strict_negative_marginal"])
        self.assertEqual(metrics["strict_negative_marginal_count"], 1)
        self.assertEqual(metrics["strict_negative_marginal_rate"], 0.25)
        self.assertFalse(metrics["has_epsilon_1e12_negative_marginal"])
        self.assertEqual(metrics["epsilon_1e12_negative_marginal_count"], 0)
        self.assertAlmostEqual(metrics["redundancy_share"], 1.0)
        self.assertAlmostEqual(
            metrics["normalized_mean_absolute_interaction"],
            5e-14,
            places=15,
        )

    def test_budget_zero_is_explicit_empty_sanity_record(self) -> None:
        event_ids = (1, 2)
        utilities = {(): 0.0, (1,): 1.0, (2,): 2.0, (1, 2): 3.0}
        state = _geometry_state(event_ids, utilities, baseline=10.0)
        record = evaluate_selector_geometry_state(state, budget=0)
        self.assertEqual(record["budget_event_capacity"], 0)
        self.assertEqual(
            record["methods"]["analytic_exact_cardinality_random"][
                "coalition_count"
            ],
            1,
        )
        for method in record["methods"].values():
            self.assertEqual(method["utility"], 0.0)
            if method["selection_type"] == "deterministic_subset":
                self.assertEqual(method["selected_coalition"], [])
        for gap in record["gaps"].values():
            self.assertEqual(gap, {"raw_utility": 0.0, "normalized_recovery": 0.0})

    def test_zero_exact_utility_has_null_ratio_and_zero_baseline_is_unnormalized(
        self,
    ) -> None:
        event_ids = (1, 2)
        utilities = {(): 0.0, (1,): 0.0, (2,): -1.0, (1, 2): -2.0}
        state = _geometry_state(event_ids, utilities, baseline=0.0)
        record = evaluate_selector_geometry_state(state, budget=1)
        for method in record["methods"].values():
            self.assertIsNone(method["normalized_recovery"])
            self.assertIsNone(method["utility_ratio_to_exact_subset"])

    def test_budget_above_candidate_count_fails_instead_of_clamping(self) -> None:
        event_ids = (1, 2)
        utilities = {(): 0.0, (1,): 1.0, (2,): 2.0, (1, 2): 3.0}
        state = _geometry_state(event_ids, utilities)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            evaluate_selector_geometry_state(state, budget=3)


if __name__ == "__main__":
    unittest.main()
