import unittest

from causalcache.attribution import (
    estimate_budget_conditioned_restoration,
    exact_permutation_restoration,
    is_maximal_feasible_coalition,
    near_budget_coalition,
)
from causalcache.selection import select_positive_value_knapsack


class AttributionTest(unittest.TestCase):
    def test_additive_distance_recovers_exact_values(self) -> None:
        weights = {0: 3.0, 1: 2.0, 2: 1.0, 3: -0.5}
        costs = {event_id: 1 for event_id in weights}

        def distance(coalition: frozenset[int]) -> float:
            base = 10.0
            return base - sum(weights[event_id] for event_id in coalition)

        estimates = estimate_budget_conditioned_restoration(
            distance,
            costs,
            budget=2,
            sample_count=8,
            seed=7,
        )
        self.assertEqual({event_id: value.mean for event_id, value in estimates.items()}, weights)
        self.assertTrue(all(value.standard_error == 0.0 for value in estimates.values()))

    def test_exact_permutation_matches_additive_values(self) -> None:
        weights = {0: 1.5, 1: 0.5, 2: 2.0}
        costs = {event_id: 1 for event_id in weights}

        def distance(coalition: frozenset[int]) -> float:
            return 5.0 - sum(weights[event_id] for event_id in coalition)

        self.assertEqual(exact_permutation_restoration(distance, costs, budget=2), weights)

    def test_near_budget_coalition_is_maximal(self) -> None:
        costs = {0: 3, 1: 2, 2: 2, 3: 1}
        coalition = near_budget_coalition(
            (0, 1, 2, 3),
            excluded_event=0,
            event_costs=costs,
            capacity=4,
        )
        self.assertEqual(coalition, frozenset({1, 2}))
        self.assertTrue(
            is_maximal_feasible_coalition(
                coalition,
                excluded_event=0,
                event_costs=costs,
                capacity=4,
            )
        )

    def test_negative_scores_are_not_forced_into_selection(self) -> None:
        selected = select_positive_value_knapsack(
            {0: 2.0, 1: -1.0, 2: 0.0},
            {0: 1, 1: 1, 2: 1},
            budget=3,
        )
        self.assertEqual(selected, frozenset({0}))

    def test_estimator_is_seed_reproducible(self) -> None:
        costs = {0: 1, 1: 1, 2: 1}

        def distance(coalition: frozenset[int]) -> float:
            return float((3 - len(coalition)) ** 2)

        left = estimate_budget_conditioned_restoration(distance, costs, budget=2, sample_count=5, seed=11)
        right = estimate_budget_conditioned_restoration(distance, costs, budget=2, sample_count=5, seed=11)
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
