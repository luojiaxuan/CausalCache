from __future__ import annotations

import itertools
import math
import unittest

from causalcache.gate_v1_data import (
    CandidateFeatures,
    FeatureState,
    GateState,
    independent_input,
)
from causalcache.restoration_v2_2_label_table import validate_complete_distance_table
from causalcache.set_conditioned_v3_data import (
    TARGET_SCALE_EPSILON,
    build_training_batch,
    feasible_coalitions,
    fit_target_scale,
    v3_independent_input,
)


def _state(source: str, event_count: int, *, multiplier: float = 1.0) -> GateState:
    event_ids = tuple(range(1, event_count + 1))
    utilities: dict[tuple[int, ...], float] = {}
    for size in range(event_count + 1):
        for coalition in itertools.combinations(event_ids, size):
            singleton = math.fsum(multiplier * event for event in coalition)
            interaction = math.fsum(
                multiplier * left * right / 10.0
                for left, right in itertools.combinations(coalition, 2)
            )
            utilities[coalition] = singleton + interaction
    table = validate_complete_distance_table(
        event_ids,
        {coalition: 100.0 - utility for coalition, utility in utilities.items()},
    )
    candidates = tuple(
        CandidateFeatures(
            event_step_id=event_id,
            h64=tuple(
                multiplier if index == event_id - 1 else 0.0 for index in range(64)
            ),
            g8=(0.1 * event_id,) * 8,
        )
        for event_id in event_ids
    )
    return GateState(
        source_id=source,
        state_id=f"{source}:n{event_count}:{multiplier}",
        decision_step_id=event_count + 2,
        candidate_event_step_ids=event_ids,
        q64=tuple(0.125 for _ in range(64)),
        candidates=candidates,
        table=table,
    )


def _feature_state(state: GateState) -> FeatureState:
    return FeatureState(
        source_id=state.source_id,
        state_id=state.state_id,
        decision_step_id=state.decision_step_id,
        candidate_event_step_ids=state.candidate_event_step_ids,
        q64=state.q64,
        candidates=state.candidates,
    )


class SetConditionedV3DataTest(unittest.TestCase):
    def test_exact_b2_geometry_is_at_most_eleven(self) -> None:
        self.assertEqual(len(feasible_coalitions(_state("s2", 2))), 4)
        self.assertEqual(len(feasible_coalitions(_state("s3", 3))), 7)
        self.assertEqual(len(feasible_coalitions(_state("s4", 4))), 11)
        self.assertEqual(
            feasible_coalitions(_state("s2", 2)),
            ((), (1,), (2,), (1, 2)),
        )

    def test_label_blind_input_exactly_replays_v1(self) -> None:
        state = _state("source", 4)
        feature_state = _feature_state(state)
        self.assertEqual(feasible_coalitions(feature_state), feasible_coalitions(state))
        for event_id in state.candidate_event_step_ids:
            self.assertEqual(
                v3_independent_input(feature_state, event_id),
                independent_input(state, event_id),
            )

    def test_targets_preserve_single_pair_and_interaction_identity(self) -> None:
        state = _state("source", 3)
        batch = build_training_batch((state,))
        singleton = {
            row.event_step_id: row.raw_utility for row in batch.singleton_targets
        }
        for row in batch.pair_targets:
            expected = (
                row.raw_utility
                - singleton[row.coalition[0]]
                - singleton[row.coalition[1]]
            )
            self.assertAlmostEqual(row.raw_residual, expected, places=12)
            self.assertAlmostEqual(
                row.normalized_residual,
                row.raw_residual / batch.target_scale.rms,
                places=12,
            )

    def test_trajectory_state_and_group_weights_are_equal(self) -> None:
        states = (_state("a", 2), _state("a", 3), _state("b", 4))
        batch = build_training_batch(states)
        self.assertAlmostEqual(batch.singleton_weight_sum, 1.0, places=12)
        self.assertAlmostEqual(batch.pair_weight_sum, 1.0, places=12)
        self.assertAlmostEqual(batch.residual_weight_sum, 1.0, places=12)
        singleton_by_state: dict[str, float] = {}
        pair_by_state: dict[str, float] = {}
        for row in batch.singleton_targets:
            singleton_by_state[row.state_id] = singleton_by_state.get(row.state_id, 0.0) + row.weight
        for row in batch.pair_targets:
            pair_by_state[row.state_id] = pair_by_state.get(row.state_id, 0.0) + row.utility_weight
        self.assertAlmostEqual(singleton_by_state[states[0].state_id], 0.25, places=12)
        self.assertAlmostEqual(singleton_by_state[states[1].state_id], 0.25, places=12)
        self.assertAlmostEqual(singleton_by_state[states[2].state_id], 0.50, places=12)
        for state_id, weight in singleton_by_state.items():
            self.assertAlmostEqual(weight, pair_by_state[state_id], places=12)

    def test_one_train_fold_rms_is_shared_across_all_three_groups(self) -> None:
        states = (_state("a", 2), _state("b", 4, multiplier=2.0))
        batch = build_training_batch(states)
        expected_square = (
            math.fsum(row.weight * row.raw_utility**2 for row in batch.singleton_targets)
            + math.fsum(
                row.utility_weight * row.raw_utility**2 for row in batch.pair_targets
            )
            + math.fsum(
                row.residual_weight * row.raw_residual**2 for row in batch.pair_targets
            )
        ) / 3.0
        self.assertAlmostEqual(batch.target_scale.rms, math.sqrt(expected_square), places=12)
        for row in batch.singleton_targets:
            self.assertAlmostEqual(
                batch.target_scale.to_raw(row.normalized_utility), row.raw_utility
            )
        small_fold = fit_target_scale((states[0],))
        self.assertNotEqual(small_fold.rms, batch.target_scale.rms)

    def test_all_feasible_set_ranking_pairs_are_built(self) -> None:
        state = _state("source", 2)
        batch = build_training_batch((state,))
        self.assertEqual(len(batch.ranking_pairs), math.comb(4, 2))
        self.assertAlmostEqual(batch.ranking_weight_sum, 1.0, places=12)
        observed = {
            (row.left_coalition, row.right_coalition)
            for row in batch.ranking_pairs
        }
        self.assertIn(((), (1, 2)), observed)

    def test_zero_signal_scale_is_clamped_and_ranking_is_empty(self) -> None:
        state = _state("zero", 2, multiplier=0.0)
        scale = fit_target_scale((state,))
        self.assertTrue(scale.clamped)
        self.assertEqual(scale.rms, TARGET_SCALE_EPSILON)
        batch = build_training_batch((state,), target_scale=scale)
        self.assertEqual(batch.ranking_pairs, ())


if __name__ == "__main__":
    unittest.main()
