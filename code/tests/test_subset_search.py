import itertools
import math
import unittest
from pathlib import Path

from causalcache.attribution import exact_permutation_restoration
from causalcache.selection import restoration_utility, select_positive_value_knapsack
from causalcache.subset_search import (
    MemoizedSetUtility,
    beam_subset_search,
    conditional_marginal_greedy,
    exact_subset_search,
    greedy_with_bounded_exchange,
    second_order_interaction,
)
from causalcache.synthetic import SyntheticBehaviorModel


ROOT = Path(__file__).resolve().parents[1]


def table_utility(values: dict[tuple[int, ...], float]):
    return lambda coalition: values[tuple(sorted(coalition))]


class SubsetSearchTest(unittest.TestCase):
    def test_exact_includes_empty_and_uses_deterministic_ties(self) -> None:
        all_negative = table_utility({(): 0.0, (10,): -1.0, (20,): -2.0, (10, 20): -0.5})
        result = exact_subset_search(all_negative, {10: 1, 20: 1}, budget=2)
        self.assertEqual(result.coalition, frozenset())
        self.assertEqual(result.unique_set_evaluations, 4)

        ties = table_utility({(): 0.0, (10,): 2.0, (20,): 2.0, (10, 20): 2.0})
        tied = exact_subset_search(ties, {10: 1, 20: 2}, budget=2)
        self.assertEqual(tied.coalition, frozenset({10}))

    def test_exact_limit_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            exact_subset_search(lambda coalition: float(len(coalition)), {0: 1, 1: 1}, budget=2, max_feasible_coalitions=3)

    def test_greedy_recomputes_conditional_marginals(self) -> None:
        values = {
            (): 0.0,
            (0,): 5.0,
            (1,): 4.0,
            (2,): 3.0,
            (0, 1): 5.1,
            (0, 2): 8.0,
            (1, 2): 7.0,
        }
        result = conditional_marginal_greedy(table_utility(values), {0: 1, 1: 1, 2: 1}, budget=2)
        self.assertEqual(result.coalition, frozenset({0, 2}))
        self.assertEqual([step.selected_event for step in result.trace], [0, 2])
        self.assertEqual(result.candidate_score_evaluations, 5)
        self.assertEqual(result.unique_set_evaluations, 6)

    def test_greedy_stops_before_nonpositive_event(self) -> None:
        values = {(): 0.0, (0,): 2.0, (1,): -1.0, (0, 1): 1.0}
        result = conditional_marginal_greedy(table_utility(values), {0: 1, 1: 1}, budget=2)
        self.assertEqual(result.coalition, frozenset({0}))

    def test_raw_and_density_diverge_with_variable_costs(self) -> None:
        values = {
            (): 0.0,
            (0,): 5.0,
            (1,): 3.0,
            (2,): 3.0,
            (1, 2): 6.0,
        }

        def utility(coalition: frozenset[int]) -> float:
            return values[tuple(sorted(coalition))]

        raw = conditional_marginal_greedy(utility, {0: 2, 1: 1, 2: 1}, budget=2, score_rule="raw_gain")
        density = conditional_marginal_greedy(
            utility,
            {0: 2, 1: 1, 2: 1},
            budget=2,
            score_rule="gain_per_cost",
        )
        self.assertEqual(raw.coalition, frozenset({0}))
        self.assertEqual(density.coalition, frozenset({1, 2}))

    def test_pure_complementarity_separates_greedy_beam_and_exact(self) -> None:
        values = {
            (): 0.0,
            (0,): 0.0,
            (1,): 0.0,
            (2,): 1.0,
            (0, 1): 10.0,
            (0, 2): 1.0,
            (1, 2): 1.0,
        }
        utility = table_utility(values)
        costs = {0: 1, 1: 1, 2: 1}
        greedy = conditional_marginal_greedy(utility, costs, budget=2)
        exact = exact_subset_search(utility, costs, budget=2)
        pruned_beam = beam_subset_search(
            utility,
            costs,
            budget=2,
            width=2,
            allow_nonpositive_prefixes=False,
        )
        permissive_beam = beam_subset_search(
            utility,
            costs,
            budget=2,
            width=3,
            allow_nonpositive_prefixes=True,
        )
        self.assertEqual(greedy.coalition, frozenset({2}))
        self.assertEqual(exact.coalition, frozenset({0, 1}))
        self.assertEqual(pruned_beam.coalition, frozenset({2}))
        self.assertEqual(permissive_beam.coalition, frozenset({0, 1}))

    def test_two_by_two_exchange_repairs_a_full_budget_trap(self) -> None:
        events = range(4)
        values = {
            tuple(subset): float(len(subset))
            for size in range(3)
            for subset in itertools.combinations(events, size)
        }
        values[(0,)] = 4.0
        values[(1,)] = 3.0
        values[(2,)] = 0.0
        values[(3,)] = 0.0
        values[(0, 1)] = 5.0
        values[(2, 3)] = 10.0
        utility = table_utility(values)
        costs = {event_id: 1 for event_id in events}
        greedy = conditional_marginal_greedy(utility, costs, budget=2)
        exchange = greedy_with_bounded_exchange(
            utility,
            costs,
            budget=2,
            max_remove=2,
            max_add=2,
            max_passes=4,
        )
        self.assertEqual(greedy.coalition, frozenset({0, 1}))
        self.assertEqual(exchange.coalition, frozenset({2, 3}))
        self.assertGreater(exchange.unique_set_evaluations, greedy.unique_set_evaluations)

    def test_memoized_utility_counts_each_set_once_and_validates(self) -> None:
        calls = 0

        def utility(coalition: frozenset[int]) -> float:
            nonlocal calls
            calls += 1
            return float(len(coalition))

        oracle = MemoizedSetUtility(utility, {5: 1}, budget=1)
        self.assertEqual(oracle(frozenset({5})), 1.0)
        self.assertEqual(oracle(frozenset({5})), 1.0)
        self.assertEqual(calls, 1)
        self.assertEqual(oracle.unique_evaluation_count, 1)
        with self.assertRaisesRegex(ValueError, "unknown"):
            oracle(frozenset({6}))
        with self.assertRaisesRegex(ValueError, "finite"):
            MemoizedSetUtility(lambda coalition: math.nan, {0: 1}, budget=1)(frozenset())

    def test_interaction_sign_and_symmetry(self) -> None:
        complement = table_utility({(): 0.0, (0,): 1.0, (1,): 1.0, (0, 1): 5.0})
        redundant = table_utility({(): 0.0, (0,): 4.0, (1,): 4.0, (0, 1): 5.0})
        self.assertEqual(second_order_interaction(complement, frozenset(), 0, 1), 3.0)
        self.assertEqual(
            second_order_interaction(complement, frozenset(), 0, 1),
            second_order_interaction(complement, frozenset(), 1, 0),
        )
        self.assertEqual(second_order_interaction(redundant, frozenset(), 0, 1), -3.0)

    def test_phase0_projection_gap_is_not_a_greedy_search_gap(self) -> None:
        model = SyntheticBehaviorModel.load(ROOT / "configs" / "synthetic_phase0.json")
        scores = exact_permutation_restoration(model.distance, model.event_costs, budget=model.budget)
        static = select_positive_value_knapsack(scores, model.event_costs, budget=model.budget)
        static_utility = restoration_utility(model.distance, static)

        base = model.distance(frozenset())
        utility = lambda coalition: base - model.distance(coalition)
        exact = exact_subset_search(utility, model.event_costs, budget=model.budget)
        greedy = conditional_marginal_greedy(utility, model.event_costs, budget=model.budget)
        self.assertAlmostEqual(static_utility / exact.utility, 0.858594, places=6)
        self.assertEqual(greedy.coalition, exact.coalition)
        self.assertAlmostEqual(greedy.utility / exact.utility, 1.0)


if __name__ == "__main__":
    unittest.main()
