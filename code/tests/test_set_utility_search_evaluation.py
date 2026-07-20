from __future__ import annotations

import importlib.util
import inspect
import itertools
import math
import unittest

from causalcache.set_utility_data import SetUtilityState, SubsetUtilityTarget
from causalcache.set_utility_evaluation import (
    evaluate_joint_selectors,
    evaluate_prediction_table,
    paired_trajectory_win_tie_loss,
    trajectory_equal_mean,
)
from causalcache.set_utility_models import SetUtilityDimensions
from causalcache.set_utility_search import (
    conditional_greedy_at_most_budget_search,
    enumerate_at_most_budget_subsets,
    exact_utility_oracle,
    joint_at_most_budget_search,
    learned_joint_at_most_budget_search,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _state(
    trajectory_id: str,
    state_id: str,
    singleton_utilities: tuple[float, float],
    *,
    pair_utility: float,
    normalization_scale: float = 10.0,
) -> SetUtilityState:
    return SetUtilityState(
        trajectory_id=trajectory_id,
        state_id=state_id,
        event_ids=(1, 2),
        maximum_labeled_cardinality=2,
        query_features=(0.1, 0.2, 0.3),
        context_features=(0.4, 0.5),
        event_features=((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 1.0, 0.0)),
        targets=(
            SubsetUtilityTarget((), 0.0),
            SubsetUtilityTarget((1,), singleton_utilities[0]),
            SubsetUtilityTarget((2,), singleton_utilities[1]),
            SubsetUtilityTarget((1, 2), pair_utility),
        ),
        normalization_scale=normalization_scale,
    )


class JointSearchTest(unittest.TestCase):
    def test_at_most_budget_inventory_includes_empty(self) -> None:
        subsets = enumerate_at_most_budget_subsets((1, 2, 3, 4), budget=2)
        self.assertEqual(len(subsets), 11)
        self.assertEqual(subsets[0], ())
        self.assertEqual(subsets[1:5], ((1,), (2,), (3,), (4,)))
        self.assertTrue(all(len(subset) <= 2 for subset in subsets))
        self.assertEqual(
            enumerate_at_most_budget_subsets((1, 2), budget=0),
            ((),),
        )

    def test_empty_set_wins_when_every_nonempty_prediction_is_negative(self) -> None:
        result = joint_at_most_budget_search(
            (1, 2, 3),
            budget=2,
            score=lambda subset: 0.0 if not subset else -float(len(subset)),
        )
        self.assertEqual(result.selected_subset, ())
        self.assertEqual(result.selected_predicted_utility, 0.0)

    def test_ties_use_smaller_cardinality_then_lexicographic_event_tuple(self) -> None:
        smaller = joint_at_most_budget_search(
            (2, 4, 6),
            budget=2,
            score=lambda subset: 1.0 if subset else 0.0,
        )
        self.assertEqual(smaller.selected_subset, (2,))
        lexical = joint_at_most_budget_search(
            (2, 4, 6),
            budget=2,
            score=lambda subset: 2.0 if len(subset) == 2 else 0.0,
        )
        self.assertEqual(lexical.selected_subset, (2, 4))

    def test_budget_is_only_a_search_argument(self) -> None:
        self.assertIn(
            "budget", inspect.signature(joint_at_most_budget_search).parameters
        )
        self.assertIn(
            "budget",
            inspect.signature(learned_joint_at_most_budget_search).parameters,
        )
        self.assertNotIn("budget", SetUtilityState.__dataclass_fields__)
        self.assertEqual(
            set(inspect.signature(lambda subset: 0.0).parameters), {"subset"}
        )

    def test_exact_oracle_obeys_at_most_budget_and_true_ties(self) -> None:
        state = _state("trajectory", "state", (2.0, 2.0), pair_utility=2.0)
        result = exact_utility_oracle(state, budget=2)
        self.assertEqual(result.selected_subset, (1,))
        self.assertEqual(result.selected_predicted_utility, 2.0)
        with self.assertRaisesRegex(ValueError, "exactly every subset"):
            SetUtilityState(
                **{
                    **state.__dict__,
                    "state_id": "incomplete-state",
                    "targets": state.targets[:3],
                }
            )
        with self.assertRaisesRegex(ValueError, "label-cardinality cap"):
            exact_utility_oracle(state, budget=3)

    def test_search_rejects_invalid_inputs_and_nonfinite_scores(self) -> None:
        with self.assertRaises(ValueError):
            enumerate_at_most_budget_subsets((2, 1), budget=1)
        with self.assertRaises(ValueError):
            enumerate_at_most_budget_subsets((1, 2), budget=-1)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            joint_at_most_budget_search(
                (1,), budget=1, score=lambda subset: math.nan
            )

    def test_conditional_greedy_batches_variable_n_candidates_and_stops(self) -> None:
        calls = []
        weights = {1: 3.0, 2: -1.0, 3: -2.0}

        def score_batch(subsets):
            calls.append(subsets)
            return [sum(weights[event_id] for event_id in subset) for subset in subsets]

        result = conditional_greedy_at_most_budget_search(
            (1, 2, 3), budget=4, score_batch=score_batch
        )
        self.assertEqual(result.selected_subset, (1,))
        self.assertEqual(result.selected_predicted_utility, 3.0)
        self.assertEqual(
            calls,
            [
                ((),),
                ((1,), (2,), (3,)),
                ((1, 2), (1, 3)),
            ],
        )

    def test_conditional_greedy_supports_budgets_one_through_four_and_ties(self) -> None:
        weights = {1: 1.0, 2: 1.0, 3: 0.5, 4: 0.25, 5: 0.125}

        def score_batch(subsets):
            return [sum(weights[event_id] for event_id in subset) for subset in subsets]

        expected = {
            1: (1,),
            2: (1, 2),
            3: (1, 2, 3),
            4: (1, 2, 3, 4),
        }
        for budget, subset in expected.items():
            with self.subTest(budget=budget):
                result = conditional_greedy_at_most_budget_search(
                    (1, 2, 3, 4, 5),
                    budget=budget,
                    score_batch=score_batch,
                )
                self.assertEqual(result.selected_subset, subset)

    def test_conditional_greedy_keeps_empty_when_no_addition_is_positive(self) -> None:
        result = conditional_greedy_at_most_budget_search(
            (1, 2),
            budget=2,
            score_batch=lambda subsets: [0.0 if not subset else -1.0 for subset in subsets],
        )
        self.assertEqual(result.selected_subset, ())
        self.assertEqual(result.selected_predicted_utility, 0.0)


class SetUtilityEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.states = (
            _state("trajectory-a", "state-a1", (2.0, 1.0), pair_utility=2.5),
            _state("trajectory-a", "state-a2", (4.0, 3.0), pair_utility=4.5),
            _state("trajectory-b", "state-b1", (10.0, 8.0), pair_utility=10.5),
        )

    def test_trajectory_equal_aggregation_is_not_state_equal(self) -> None:
        values = {"state-a1": 1.0, "state-a2": 3.0, "state-b1": 8.0}
        self.assertEqual(trajectory_equal_mean(self.states, values), 5.0)
        self.assertNotEqual(trajectory_equal_mean(self.states, values), 4.0)

    def test_selector_is_scored_by_true_utility_and_reports_exact_ratio(self) -> None:
        selections = {
            "learned": {
                "state-a1": (2,),
                "state-a2": (2,),
                "state-b1": (2,),
            },
            "heuristic": {
                "state-a1": (),
                "state-a2": (1,),
                "state-b1": (1,),
            },
        }
        report = evaluate_joint_selectors(
            self.states,
            budget=1,
            selections_by_method=selections,
        )
        self.assertTrue(report["all_selected_sets_scored_by_true_utility"])
        learned = report["methods"]["learned"]
        self.assertEqual(learned["trajectory_equal_mean_raw_utility"], 5.0)
        self.assertEqual(learned["trajectory_equal_mean_exact_utility"], 6.5)
        self.assertAlmostEqual(
            learned["trajectory_equal_mean_exact_recovery_ratio"],
            ((1.0 / 2.0 + 3.0 / 4.0) / 2.0 + 8.0 / 10.0) / 2.0,
        )
        self.assertAlmostEqual(
            learned["ratio_of_trajectory_equal_mean_raw_utility_to_exact"],
            5.0 / 6.5,
        )
        self.assertEqual(
            report["records"]["state-a1"]["methods"]["learned"]["true_utility"],
            1.0,
        )
        self.assertEqual(
            learned["selected_cardinality_distribution"]["counts"],
            {"0": 0, "1": 3},
        )
        exact = report["methods"]["exact_subset_oracle"]
        self.assertEqual(
            exact["trajectory_equal_mean_exact_recovery_ratio"], 1.0
        )

    def test_prediction_can_select_a_true_bad_set_without_changing_evaluator(self) -> None:
        bad = _state("trajectory-c", "state-c", (1.0, -2.0), pair_utility=-3.0)
        predicted = joint_at_most_budget_search(
            bad.event_ids,
            budget=1,
            score=lambda subset: 100.0 if subset == (2,) else 0.0,
        )
        report = evaluate_joint_selectors(
            (bad,),
            budget=1,
            selections_by_method={
                "learned": {bad.state_id: predicted.selected_subset}
            },
        )
        learned_record = report["records"][bad.state_id]["methods"]["learned"]
        self.assertEqual(learned_record["selected_subset"], [2])
        self.assertEqual(learned_record["true_utility"], -2.0)
        self.assertEqual(
            report["records"][bad.state_id]["methods"]["exact_subset_oracle"][
                "true_utility"
            ],
            1.0,
        )

    def test_zero_exact_states_are_excluded_only_from_recovery_ratio(self) -> None:
        zero = _state(
            "trajectory-zero",
            "state-zero",
            (-1.0, -2.0),
            pair_utility=-3.0,
        )
        report = evaluate_joint_selectors(
            (zero,),
            budget=2,
            selections_by_method={"learned": {zero.state_id: ()}},
        )
        learned = report["methods"]["learned"]
        self.assertEqual(learned["trajectory_equal_mean_raw_utility"], 0.0)
        self.assertIsNone(learned["trajectory_equal_mean_exact_recovery_ratio"])
        self.assertEqual(learned["exact_recovery_excluded_state_count"], 1)
        self.assertEqual(learned["exact_recovery_excluded_trajectory_count"], 1)

    def test_paired_wins_ties_losses_are_trajectory_level(self) -> None:
        left = {"state-a1": 1.0, "state-a2": 3.0, "state-b1": 8.0}
        right = {"state-a1": 0.0, "state-a2": 4.0, "state-b1": 10.0}
        result = paired_trajectory_win_tie_loss(self.states, left, right)
        self.assertEqual(
            (result["wins"], result["ties"], result["losses"]),
            (0, 1, 1),
        )
        self.assertEqual(result["trajectory_equal_mean_delta"], -1.0)

    def test_prediction_diagnostics_are_balanced_and_exact_predictions_are_perfect(self) -> None:
        predictions = {
            state.state_id: {
                target.subset: target.raw_utility for target in state.targets
            }
            for state in self.states
        }
        report = evaluate_prediction_table(self.states, predictions)
        self.assertEqual(
            report["weighting"],
            "state_equal_then_cardinality_equal_within_state",
        )
        self.assertEqual(report["raw_regression"], {"mae": 0.0, "rmse": 0.0})
        self.assertEqual(
            report["normalized_regression"], {"mae": 0.0, "rmse": 0.0}
        )
        ranking = report["within_state_subset_ranking"]
        self.assertGreater(ranking["pair_count"], 0)
        self.assertAlmostEqual(
            ranking["balanced_accuracy_with_half_credit_for_prediction_ties"],
            1.0,
        )

    def test_evaluator_fails_closed_on_missing_or_over_budget_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "inventory"):
            evaluate_joint_selectors(
                self.states,
                budget=1,
                selections_by_method={"learned": {"state-a1": ()}},
            )
        with self.assertRaisesRegex(ValueError, "feasible"):
            evaluate_joint_selectors(
                self.states,
                budget=1,
                selections_by_method={
                    "learned": {
                        state.state_id: (1, 2) for state in self.states
                    }
                },
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional locally")
class LearnedJointSearchTorchTest(unittest.TestCase):
    def test_learned_search_calls_budget_free_model_once_and_applies_ties(self) -> None:
        import torch

        class TiedNonemptyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.tensor(0.0))
                self.call_count = 0

            def score_subsets(
                self,
                query_features,
                context_features,
                event_features,
                subset_masks,
                event_mask=None,
            ):
                self.call_count += 1
                del query_features, context_features, event_features, event_mask
                return subset_masks.any(dim=-1).to(self.anchor.dtype) + self.anchor

            forward = score_subsets

        state = _state("trajectory", "state", (1.0, 2.0), pair_utility=3.0)
        model = TiedNonemptyModel()
        result = learned_joint_at_most_budget_search(model, state, budget=2)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(result.selected_subset, (1,))
        self.assertNotIn("budget", inspect.signature(model.forward).parameters)
        self.assertEqual(len(result.scored_subsets), 4)


if __name__ == "__main__":
    unittest.main()
