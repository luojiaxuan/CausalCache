from __future__ import annotations

import itertools
import math
import unittest

from causalcache.restoration_v2_2_label_table import (
    ATTRIBUTION_ROW_COUNTS_BY_EVENT_COUNT,
    DEPLOYMENT_EDGE_COUNTS_BY_EVENT_COUNT,
    EXACT_TIE_EPSILON,
    EXPECTED_FULL_45_ATTRIBUTION_ROW_COUNT,
    EXPECTED_FULL_45_DEPLOYMENT_EDGE_COUNT,
    EXPECTED_FULL_45_DISTANCE_ROW_COUNT,
    EXPECTED_FULL_45_FULL_EDGE_COUNT,
    EXPECTED_FULL_45_PAIR_INTERACTION_COUNT,
    FULL_DISTANCE_ROWS_BY_EVENT_COUNT,
    FULL_EDGE_COUNTS_BY_EVENT_COUNT,
    PAIR_INTERACTION_COUNTS_BY_EVENT_COUNT,
    STATES_PER_EVENT_COUNT,
    deployment_conditional_edges,
    exact_permutation_average_attribution,
    full_conditional_edges,
    pair_interactions,
    primary_exact_subset_oracle,
    restoration_utility,
    validate_complete_distance_table,
)


def complete_table(
    event_ids: tuple[int, ...],
    utility,
    *,
    empty_distance: float = 20.0,
) -> dict[frozenset[int], float]:
    return {
        frozenset(subset): empty_distance - float(utility(frozenset(subset)))
        for size in range(len(event_ids) + 1)
        for subset in itertools.combinations(event_ids, size)
    }


class RestorationV22LabelTableTest(unittest.TestCase):
    def test_frozen_counts_for_every_state_size_and_full_45(self) -> None:
        distance_total = 0
        deployment_edge_total = 0
        full_edge_total = 0
        pair_interaction_total = 0
        attribution_total = 0
        for event_count in (2, 3, 4):
            event_ids = tuple(range(1, event_count + 1))
            table = validate_complete_distance_table(
                event_ids,
                complete_table(event_ids, lambda coalition: len(coalition)),
            )
            self.assertEqual(
                len(table.rows),
                FULL_DISTANCE_ROWS_BY_EVENT_COUNT[event_count],
            )
            self.assertEqual(
                len(deployment_conditional_edges(table)),
                DEPLOYMENT_EDGE_COUNTS_BY_EVENT_COUNT[event_count],
            )
            self.assertEqual(
                len(full_conditional_edges(table)),
                FULL_EDGE_COUNTS_BY_EVENT_COUNT[event_count],
            )
            self.assertEqual(
                len(pair_interactions(table)),
                PAIR_INTERACTION_COUNTS_BY_EVENT_COUNT[event_count],
            )
            self.assertEqual(
                len(exact_permutation_average_attribution(table)),
                ATTRIBUTION_ROW_COUNTS_BY_EVENT_COUNT[event_count],
            )
            distance_total += len(table.rows) * STATES_PER_EVENT_COUNT
            deployment_edge_total += (
                len(deployment_conditional_edges(table)) * STATES_PER_EVENT_COUNT
            )
            full_edge_total += len(full_conditional_edges(table)) * STATES_PER_EVENT_COUNT
            pair_interaction_total += (
                len(pair_interactions(table)) * STATES_PER_EVENT_COUNT
            )
            attribution_total += (
                len(exact_permutation_average_attribution(table))
                * STATES_PER_EVENT_COUNT
            )
        self.assertEqual(distance_total, EXPECTED_FULL_45_DISTANCE_ROW_COUNT)
        self.assertEqual(
            deployment_edge_total,
            EXPECTED_FULL_45_DEPLOYMENT_EDGE_COUNT,
        )
        self.assertEqual(full_edge_total, EXPECTED_FULL_45_FULL_EDGE_COUNT)
        self.assertEqual(
            pair_interaction_total,
            EXPECTED_FULL_45_PAIR_INTERACTION_COUNT,
        )
        self.assertEqual(
            attribution_total,
            EXPECTED_FULL_45_ATTRIBUTION_ROW_COUNT,
        )
        self.assertEqual((distance_total, deployment_edge_total, full_edge_total), (420, 435, 720))
        self.assertEqual((pair_interaction_total, attribution_total), (465, 135))

    def test_utility_and_exact_oracle_allow_empty_and_use_exact_ties(self) -> None:
        event_ids = (1, 2)
        empty_best = validate_complete_distance_table(
            event_ids,
            {
                frozenset(): 0.0,
                frozenset({1}): 1.0,
                frozenset({2}): 1.0,
                frozenset({1, 2}): 2.0,
            },
        )
        self.assertEqual(primary_exact_subset_oracle(empty_best).coalition, ())
        self.assertEqual(restoration_utility(empty_best, (1,)), -1.0)

        cardinality_and_lexicographic_tie = validate_complete_distance_table(
            event_ids,
            {
                frozenset(): 5.0,
                frozenset({1}): 1.0,
                frozenset({2}): 1.0,
                frozenset({1, 2}): 1.0,
            },
        )
        oracle = primary_exact_subset_oracle(cardinality_and_lexicographic_tie)
        self.assertEqual(EXACT_TIE_EPSILON, 0.0)
        self.assertEqual(oracle.tie_epsilon, 0.0)
        self.assertEqual(oracle.coalition, (1,))
        self.assertEqual(oracle.evaluated_coalition_count, 4)

        exact_not_isclose = validate_complete_distance_table(
            event_ids,
            {
                frozenset(): 5.0,
                frozenset({1}): 1.0,
                frozenset({2}): 1.0 - 1e-12,
                frozenset({1, 2}): 2.0,
            },
        )
        self.assertEqual(primary_exact_subset_oracle(exact_not_isclose).coalition, (2,))

    def test_complementarity_and_redundancy_interaction_signs(self) -> None:
        event_ids = (1, 2)
        complement = validate_complete_distance_table(
            event_ids,
            complete_table(
                event_ids,
                lambda coalition: (
                    5.0 if coalition == frozenset({1, 2}) else float(len(coalition))
                ),
            ),
        )
        redundant = validate_complete_distance_table(
            event_ids,
            complete_table(
                event_ids,
                lambda coalition: (
                    0.0
                    if not coalition
                    else 4.0
                    if len(coalition) == 1
                    else 5.0
                ),
            ),
        )
        self.assertEqual(pair_interactions(complement)[0].interaction, 3.0)
        self.assertEqual(pair_interactions(redundant)[0].interaction, -3.0)
        self.assertEqual(
            [
                item.mean_marginal_gain
                for item in exact_permutation_average_attribution(complement)
            ],
            [2.5, 2.5],
        )
        complement_edges = deployment_conditional_edges(complement)
        self.assertEqual(
            next(
                edge.marginal_gain
                for edge in complement_edges
                if edge.base_coalition == (1,) and edge.event_id == 2
            ),
            4.0,
        )

    def test_negative_marginal_and_nonmonotone_oracle_are_preserved(self) -> None:
        event_ids = (1, 2)
        utility = {
            frozenset(): 0.0,
            frozenset({1}): 2.0,
            frozenset({2}): 1.0,
            frozenset({1, 2}): 0.5,
        }
        table = validate_complete_distance_table(
            event_ids,
            complete_table(event_ids, utility.__getitem__),
        )
        edges = deployment_conditional_edges(table)
        harmful = next(
            edge
            for edge in edges
            if edge.base_coalition == (1,) and edge.event_id == 2
        )
        self.assertEqual(harmful.marginal_gain, -1.5)
        self.assertEqual(primary_exact_subset_oracle(table).coalition, (1,))

    def test_full_edges_pair_interactions_and_permutation_average(self) -> None:
        event_ids = (1, 2, 3)
        weights = {1: 1.0, 2: 2.0, 3: 3.0}
        table = validate_complete_distance_table(
            event_ids,
            complete_table(
                event_ids,
                lambda coalition: sum(weights[event_id] for event_id in coalition),
            ),
        )
        self.assertEqual(len(full_conditional_edges(table)), 12)
        self.assertEqual(len(pair_interactions(table)), 6)
        self.assertTrue(
            all(interaction.interaction == 0.0 for interaction in pair_interactions(table))
        )
        attribution = exact_permutation_average_attribution(table)
        self.assertEqual(
            {item.event_id: item.mean_marginal_gain for item in attribution},
            weights,
        )
        self.assertTrue(all(item.permutation_count == math.factorial(3) for item in attribution))
        self.assertEqual(
            sum(item.mean_marginal_gain for item in attribution),
            restoration_utility(table, event_ids),
        )

    def test_missing_extra_duplicate_and_invalid_distances_fail_closed(self) -> None:
        valid = {
            frozenset(): 4.0,
            frozenset({1}): 3.0,
            frozenset({2}): 2.0,
            frozenset({1, 2}): 1.0,
        }
        missing = dict(valid)
        del missing[frozenset({1, 2})]
        with self.assertRaisesRegex(ValueError, "complete power set"):
            validate_complete_distance_table((1, 2), missing)

        with self.assertRaisesRegex(ValueError, "unknown events"):
            validate_complete_distance_table((1, 2), {**valid, (3,): 1.0})

        with self.assertRaisesRegex(ValueError, "duplicate coalitions"):
            validate_complete_distance_table(
                (1, 2),
                {**valid, (1, 2): 1.0},
            )

        for invalid in (-1.0, math.nan, math.inf):
            broken = dict(valid)
            broken[frozenset({1})] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError,
                "finite and non-negative",
            ):
                validate_complete_distance_table((1, 2), broken)

        broken_type = dict(valid)
        broken_type[frozenset({1})] = True
        with self.assertRaisesRegex(TypeError, "real number"):
            validate_complete_distance_table((1, 2), broken_type)

    def test_candidate_and_coalition_identity_are_strict(self) -> None:
        valid = {
            frozenset(): 4.0,
            frozenset({1}): 3.0,
            frozenset({2}): 2.0,
            frozenset({1, 2}): 1.0,
        }
        for event_ids in ((2, 1), (1, 1), (0, 1), (1,), (1, 2, 3, 4, 5)):
            with self.subTest(event_ids=event_ids), self.assertRaises((TypeError, ValueError)):
                validate_complete_distance_table(event_ids, valid)
        with self.assertRaisesRegex(TypeError, "integer event ids"):
            validate_complete_distance_table(
                (1, 2),
                {**valid, (1, "2"): 1.0},
            )


if __name__ == "__main__":
    unittest.main()
