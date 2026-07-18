from __future__ import annotations

import importlib.util
import itertools
import math
import unittest
from unittest import mock

from causalcache.gate_v1_data import CandidateFeatures, FeatureState, GateState
from causalcache.restoration_v2_2_label_table import validate_complete_distance_table
from causalcache.set_conditioned_v3_data import build_training_batch
from causalcache.set_conditioned_v3_training import (
    EXPECTED_PARAMETER_COUNT,
    V3StatePrediction,
    build_model,
    build_optimizer,
    gate_loss,
    optimization_step,
    predict_state,
    select_additive,
    select_ensemble_safe,
    select_from_utilities,
    select_unguarded,
    train_fixed_epochs,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None


def _state(event_count: int = 3) -> GateState:
    event_ids = tuple(range(1, event_count + 1))
    utilities = {
        coalition: math.fsum(coalition)
        + math.fsum(
            left * right / 10.0
            for left, right in itertools.combinations(coalition, 2)
        )
        for size in range(event_count + 1)
        for coalition in itertools.combinations(event_ids, size)
    }
    table = validate_complete_distance_table(
        event_ids,
        {coalition: 100.0 - utility for coalition, utility in utilities.items()},
    )
    candidates = tuple(
        CandidateFeatures(
            event_step_id=event_id,
            h64=tuple(
                1.0 if index == event_id - 1 else 0.0 for index in range(64)
            ),
            g8=(event_id / 10.0,) * 8,
        )
        for event_id in event_ids
    )
    return GateState(
        source_id="source",
        state_id=f"source:n{event_count}",
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


def _component_prediction(
    singleton_scores: tuple[float, ...],
    pair_residual_scores: tuple[float, ...],
) -> V3StatePrediction:
    event_ids = tuple(range(1, len(singleton_scores) + 1))
    pair_ids = tuple(itertools.combinations(event_ids, 2))
    singleton = tuple(zip(event_ids, singleton_scores, strict=True))
    residual = tuple(zip(pair_ids, pair_residual_scores, strict=True))
    singleton_by_event = dict(singleton)
    residual_by_pair = dict(residual)
    additive = (((), 0.0),) + tuple(
        ((event_id,), value) for event_id, value in singleton
    ) + tuple(
        (pair, singleton_by_event[pair[0]] + singleton_by_event[pair[1]])
        for pair in pair_ids
    )
    unguarded = tuple(
        (
            coalition,
            value if len(coalition) < 2 else value + residual_by_pair[coalition],
        )
        for coalition, value in additive
    )
    return V3StatePrediction(
        singleton_scores=singleton,
        pair_residual_scores=residual,
        additive_set_utilities=additive,
        unguarded_set_utilities=unguarded,
    )


class SetConditionedV3SelectionTest(unittest.TestCase):
    def test_ties_choose_smaller_cardinality_then_lexicographic(self) -> None:
        self.assertEqual(
            select_from_utilities(
                (((2,), 1.0), ((1,), 1.0), ((1, 2), 1.0), ((), 1.0))
            ),
            (),
        )
        self.assertEqual(
            select_from_utilities((((2,), 2.0), ((1,), 2.0), ((), 0.0))),
            (1,),
        )

    def test_safe_switches_only_when_pair_candidate_passes_both_votes(self) -> None:
        models = tuple(object() for _ in range(5))
        predictions = {
            model: _component_prediction((1.0, 0.0), (residual,))
            for model, residual in zip(
                models, (1.0, 1.0, 1.0, 1.0, -2.0), strict=True
            )
        }
        with mock.patch(
            "causalcache.set_conditioned_v3_training.predict_state",
            side_effect=lambda model, _state: predictions[model],
        ):
            result = select_ensemble_safe(models, _feature_state(_state(2)))
        self.assertEqual(result.coalition, (1, 2))
        self.assertFalse(result.used_fallback)
        self.assertEqual(result.pair_candidate, (1, 2))
        self.assertEqual(result.fallback_coalition, (1,))
        self.assertEqual(result.agreeing_seed_count, 4)
        self.assertEqual(result.positive_margin_seed_count, 4)
        self.assertEqual(result.seed_margins, (1.0, 1.0, 1.0, 1.0, -2.0))

    def test_pair_candidate_is_ensemble_mean_argmax_not_majority_mode(self) -> None:
        models = tuple(object() for _ in range(5))
        predictions = {
            model: _component_prediction((1.0, 0.0), (residual,))
            for model, residual in zip(
                models, (-0.1, -0.1, -0.1, 1.0, 1.0), strict=True
            )
        }
        with mock.patch(
            "causalcache.set_conditioned_v3_training.predict_state",
            side_effect=lambda model, _state: predictions[model],
        ):
            result = select_ensemble_safe(models, _feature_state(_state(2)))
        self.assertEqual(result.pair_candidate, (1, 2))
        self.assertEqual(result.agreeing_seed_count, 2)
        self.assertEqual(result.positive_margin_seed_count, 2)
        self.assertEqual(result.coalition, (1,))
        self.assertTrue(result.used_fallback)

    def test_positive_margin_vote_is_strict_even_when_argmax_agrees(self) -> None:
        models = tuple(object() for _ in range(5))
        predictions = {
            model: _component_prediction(
                (0.9, 1.0, 1.1),
                (residual_12, 0.0, 0.0),
            )
            for model, residual_12 in zip(
                models, (0.2, 0.2, 0.2, 0.2, 1.0), strict=True
            )
        }
        with mock.patch(
            "causalcache.set_conditioned_v3_training.predict_state",
            side_effect=lambda model, _state: predictions[model],
        ):
            result = select_ensemble_safe(models, _feature_state(_state(3)))
        self.assertEqual(result.pair_candidate, (1, 2))
        self.assertEqual(result.fallback_coalition, (2, 3))
        self.assertEqual(result.agreeing_seed_count, 5)
        self.assertEqual(result.positive_margin_seed_count, 1)
        self.assertEqual(result.coalition, (2, 3))
        self.assertTrue(result.used_fallback)

    def test_identical_pair_and_base_candidate_returns_directly(self) -> None:
        models = tuple(object() for _ in range(5))
        prediction = _component_prediction((1.0, 0.0), (-1.0,))
        with mock.patch(
            "causalcache.set_conditioned_v3_training.predict_state",
            return_value=prediction,
        ):
            result = select_ensemble_safe(models, _feature_state(_state(2)))
        self.assertEqual(result.pair_candidate, (1,))
        self.assertEqual(result.fallback_coalition, (1,))
        self.assertEqual(result.coalition, (1,))
        self.assertFalse(result.used_fallback)
        self.assertEqual(result.positive_margin_seed_count, 0)
        self.assertEqual(result.seed_margins, (0.0,) * 5)


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in the local docs environment")
class SetConditionedV3TorchTest(unittest.TestCase):
    def test_architecture_parameter_count_and_seed_determinism(self) -> None:
        first = build_model(seed=7)
        second = build_model(seed=7)
        self.assertEqual(sum(parameter.numel() for parameter in first.parameters()), EXPECTED_PARAMETER_COUNT)
        for left, right in zip(first.parameters(), second.parameters(), strict=True):
            self.assertTrue(left.equal(right))
        self.assertEqual(first.event_encoder[0].in_features, 200)
        self.assertEqual(first.event_encoder[0].out_features, 64)
        self.assertEqual(first.pair_residual_head[0].in_features, 320)
        self.assertEqual(first.pair_residual_head[0].out_features, 64)

    def test_pair_head_is_chronological_not_forced_symmetric(self) -> None:
        import torch

        model = build_model(seed=0)
        with torch.no_grad():
            for parameter in model.pair_residual_head.parameters():
                parameter.zero_()
            model.pair_residual_head[0].weight[0, 0] = 1.0
            model.pair_residual_head[2].weight[0, 0] = 1.0
        left = torch.zeros((1, 64), dtype=torch.float32)
        right = torch.zeros((1, 64), dtype=torch.float32)
        left[0, 0] = 2.0
        right[0, 0] = 1.0
        q64 = torch.zeros((1, 64), dtype=torch.float32)
        forward = model.pair_residual_scores(left, right, q64)
        reverse = model.pair_residual_scores(right, left, q64)
        self.assertNotEqual(float(forward[0]), float(reverse[0]))

    def test_zero_model_matches_frozen_loss_formula(self) -> None:
        import torch

        batch = build_training_batch((_state(2), _state(3)))
        model = build_model(seed=0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        total, scalars = gate_loss(model, batch)

        def smooth(value: float) -> float:
            absolute = abs(value)
            return 0.5 * absolute**2 if absolute < 1.0 else absolute - 0.5

        single = math.fsum(
            row.weight * smooth(row.normalized_utility)
            for row in batch.singleton_targets
        )
        pair = math.fsum(
            row.utility_weight * smooth(row.normalized_utility)
            for row in batch.pair_targets
        )
        residual = math.fsum(
            row.residual_weight * smooth(row.normalized_residual)
            for row in batch.pair_targets
        )
        ranking = math.log(2.0)
        expected = (single + pair + residual) / 3.0 + 0.25 * ranking
        self.assertAlmostEqual(float(total), expected, places=6)
        self.assertAlmostEqual(scalars["ranking"], ranking, places=6)

    def test_optimization_prediction_and_label_blind_selection(self) -> None:
        import torch

        state = _state(3)
        batch = build_training_batch((state,))
        model = build_model(seed=1)
        optimizer = build_optimizer(model, learning_rate=0.001)
        before = tuple(parameter.detach().clone() for parameter in model.parameters())
        scalars = optimization_step(model, optimizer, batch)
        self.assertTrue(math.isfinite(scalars["total"]))
        self.assertTrue(
            any(not left.equal(right) for left, right in zip(before, model.parameters(), strict=True))
        )
        gate_prediction = predict_state(model, state)
        feature_prediction = predict_state(model, _feature_state(state))
        self.assertEqual(gate_prediction, feature_prediction)
        raw = gate_prediction.to_raw(batch.target_scale)
        self.assertAlmostEqual(
            raw.singleton_scores[0][1],
            gate_prediction.singleton_scores[0][1] * batch.target_scale.rms,
        )
        self.assertIn(select_additive(model, _feature_state(state)), dict(gate_prediction.additive_set_utilities))
        self.assertIn(select_unguarded(model, _feature_state(state)), dict(gate_prediction.unguarded_set_utilities))

    def test_fixed_epoch_helper_runs_requested_number_of_steps(self) -> None:
        fit = train_fixed_epochs(
            (_state(2),), seed=3, epochs=2, learning_rate=0.001
        )
        self.assertEqual(tuple(row.epoch for row in fit.history), (1, 2))


if __name__ == "__main__":
    unittest.main()
