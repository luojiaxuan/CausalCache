from __future__ import annotations

import importlib.util
import inspect
import itertools
import math
import unittest
from collections import defaultdict

from causalcache.set_utility_data import (
    PaddedSetUtilityBatch,
    SetUtilityState,
    SubsetUtilityTarget,
    TensorSetUtilityBatch,
    build_padded_set_utility_batch,
    set_utility_state_from_feature_and_table,
    tensorize_set_utility_batch,
)
from causalcache.set_utility_features import SetUtilityFeatureState
from causalcache.set_utility_label_table import (
    validate_cardinality_capped_distance_table,
)
from causalcache.set_utility_models import (
    DeepSetsUtilityPredictor,
    SetUtilityDimensions,
)
from causalcache.set_utility_training import (
    UtilityLossWeights,
    utility_optimization_step,
    utility_predictor_loss,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _state(
    trajectory_id: str,
    state_id: str,
    event_count: int,
    *,
    maximum_cardinality: int = 2,
    normalization_scale: float | None = 20.0,
    pair_dimension: int | None = None,
) -> SetUtilityState:
    event_ids = tuple(range(1, event_count + 1))
    targets = tuple(
        SubsetUtilityTarget(
            subset,
            0.0 if not subset else float(sum(event * event for event in subset)),
        )
        for cardinality in range(maximum_cardinality + 1)
        for subset in itertools.combinations(event_ids, cardinality)
    )
    pair_features = (
        None
        if pair_dimension is None
        else tuple(
            tuple(
                tuple(float(left + right + offset) for offset in range(pair_dimension))
                for right in event_ids
            )
            for left in event_ids
        )
    )
    return SetUtilityState(
        trajectory_id=trajectory_id,
        state_id=state_id,
        event_ids=event_ids,
        maximum_labeled_cardinality=maximum_cardinality,
        query_features=(0.1, 0.2, 0.3),
        context_features=(0.4, 0.5),
        event_features=tuple(
            (float(event), float(event) / event_count, 1.0, -1.0)
            for event in event_ids
        ),
        targets=targets,
        normalization_scale=normalization_scale,
        pair_features=pair_features,
    )


class SetUtilityDataTest(unittest.TestCase):
    def test_padding_masks_subsets_and_padded_events(self) -> None:
        short = _state("trajectory-a", "state-short", 2)
        long = _state("trajectory-b", "state-long", 4)
        batch = build_padded_set_utility_batch((short, long))

        self.assertIsInstance(batch, PaddedSetUtilityBatch)
        self.assertEqual(batch.maximum_event_count, 4)
        self.assertEqual(batch.maximum_subset_count, 11)
        self.assertEqual(batch.event_ids[0], (1, 2, None, None))
        self.assertEqual(batch.event_mask[0], (True, True, False, False))
        self.assertEqual(sum(batch.subset_row_mask[0]), 4)
        self.assertFalse(any(row[2] or row[3] for row in batch.subset_masks[0]))
        for state_index, state in enumerate((short, long)):
            for row_index, target in enumerate(state.targets):
                selected = tuple(
                    event_id
                    for event_id, included in zip(
                        batch.event_ids[state_index],
                        batch.subset_masks[state_index][row_index],
                        strict=True,
                    )
                    if included
                )
                self.assertEqual(selected, target.subset)

    def test_state_and_cardinality_balancing_prevents_n16_row_domination(self) -> None:
        small = _state("trajectory-a", "state-n2", 2)
        large = _state("trajectory-b", "state-n16", 16)
        batch = build_padded_set_utility_batch((small, large))
        self.assertEqual(len(small.targets), 4)
        self.assertEqual(len(large.targets), 137)

        for state_index, state in enumerate((small, large)):
            weights = batch.raw_regression_weights[state_index][: len(state.targets)]
            self.assertAlmostEqual(math.fsum(weights), 0.5)
            by_cardinality: dict[int, float] = defaultdict(float)
            for target, weight in zip(state.targets, weights, strict=True):
                by_cardinality[len(target.subset)] += weight
            self.assertEqual(set(by_cardinality), {0, 1, 2})
            for total in by_cardinality.values():
                self.assertAlmostEqual(total, 1.0 / 6.0)

        by_ranking_state: dict[int, float] = defaultdict(float)
        by_state_bucket: dict[tuple[int, tuple[int, int]], float] = defaultdict(float)
        for row in batch.ranking_rows:
            by_ranking_state[row.batch_index] += row.weight
            by_state_bucket[(row.batch_index, row.cardinality_bucket)] += row.weight
        self.assertAlmostEqual(by_ranking_state[0], 0.5)
        self.assertAlmostEqual(by_ranking_state[1], 0.5)
        for state_index in (0, 1):
            bucket_totals = [
                total
                for (index, _), total in by_state_bucket.items()
                if index == state_index
            ]
            self.assertTrue(bucket_totals)
            self.assertTrue(
                all(
                    math.isclose(total, bucket_totals[0], abs_tol=1e-12)
                    for total in bucket_totals
                )
            )

    def test_normalized_weights_only_use_eligible_states(self) -> None:
        normalized = _state("trajectory-a", "state-normalized", 3)
        raw_only = _state(
            "trajectory-b",
            "state-raw-only",
            3,
            normalization_scale=None,
        )
        batch = build_padded_set_utility_batch((normalized, raw_only))
        self.assertEqual(batch.normalized_state_mask, (True, False))
        self.assertAlmostEqual(math.fsum(batch.normalized_regression_weights[0]), 1.0)
        self.assertEqual(math.fsum(batch.normalized_regression_weights[1]), 0.0)
        self.assertAlmostEqual(
            batch.normalized_targets[0][1],
            normalized.targets[1].raw_utility / normalized.normalization_scale,
        )

    def test_pair_features_pad_without_leaking_into_masked_events(self) -> None:
        short = _state("trajectory-a", "state-short", 2, pair_dimension=2)
        long = _state("trajectory-b", "state-long", 4, pair_dimension=2)
        batch = build_padded_set_utility_batch((short, long))
        self.assertIsNotNone(batch.pair_features)
        assert batch.pair_features is not None
        self.assertEqual(batch.pair_features[0][0][0], short.pair_features[0][0])
        self.assertEqual(batch.pair_features[0][3][3], (0.0, 0.0))

    def test_canonical_feature_and_label_join_drops_feature_padding(self) -> None:
        feature_state = SetUtilityFeatureState(
            source_id="trajectory-source",
            state_id="joined-state",
            decision_step_id=7,
            event_step_ids=(1, 3),
            query_features=(1.0, 2.0),
            context_features=(3.0,),
            event_features=((1.0,), (2.0,), (0.0,), (0.0,)),
            pair_features=(
                ((1.0,), (2.0,), (0.0,), (0.0,)),
                ((2.0,), (3.0,), (0.0,), (0.0,)),
                ((0.0,), (0.0,), (0.0,), (0.0,)),
                ((0.0,), (0.0,), (0.0,), (0.0,)),
            ),
            event_mask=(True, True, False, False),
        )
        table = validate_cardinality_capped_distance_table(
            split="train",
            state_id="joined-state",
            candidate_event_step_ids=(1, 3),
            maximum_labeled_cardinality=2,
            distances={(): 5.0, (1,): 4.0, (3,): 3.0, (1, 3): 1.0},
        )
        joined = set_utility_state_from_feature_and_table(feature_state, table)
        self.assertEqual(joined.trajectory_id, "trajectory-source")
        self.assertEqual(joined.maximum_labeled_cardinality, 2)
        self.assertEqual(joined.normalization_scale, 5.0)
        self.assertEqual(joined.event_features, ((1.0,), (2.0,)))
        self.assertEqual(joined.utility((1, 3)), 4.0)

    def test_batch_and_state_features_have_no_budget_field(self) -> None:
        for dataclass_type in (
            SetUtilityState,
            PaddedSetUtilityBatch,
            TensorSetUtilityBatch,
        ):
            self.assertNotIn("budget", dataclass_type.__dataclass_fields__)
        parameters = inspect.signature(build_padded_set_utility_batch).parameters
        self.assertNotIn("budget", parameters)

    def test_state_validation_fails_closed(self) -> None:
        valid = _state("trajectory-a", "state-a", 2)
        with self.assertRaisesRegex(ValueError, "empty-set utility"):
            SetUtilityState(
                **{
                    **valid.__dict__,
                    "targets": (
                        SubsetUtilityTarget((), 1.0),
                        *valid.targets[1:],
                    ),
                }
            )
        with self.assertRaisesRegex(ValueError, "sorted unique"):
            SetUtilityState(
                **{**valid.__dict__, "event_ids": (2, 1)}
            )
        with self.assertRaisesRegex(ValueError, "exactly every subset"):
            SetUtilityState(
                **{**valid.__dict__, "targets": valid.targets[:-1]}
            )
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            build_padded_set_utility_batch(
                (
                    valid,
                    SetUtilityState(
                        **{
                            **valid.__dict__,
                            "state_id": "state-other",
                            "query_features": (1.0,),
                        }
                    ),
                )
            )

    def test_tensorization_reports_optional_torch_boundary(self) -> None:
        batch = build_padded_set_utility_batch((_state("t", "s", 2),))
        if TORCH_AVAILABLE:
            self.assertIsInstance(tensorize_set_utility_batch(batch), TensorSetUtilityBatch)
        else:
            with self.assertRaisesRegex(RuntimeError, "requires PyTorch"):
                tensorize_set_utility_batch(batch)


class UtilityLossWeightsTest(unittest.TestCase):
    def test_weights_are_explicit_and_validated(self) -> None:
        weights = UtilityLossWeights(1.0, 0.5, 0.25, 1.0)
        self.assertEqual(weights.raw_regression, 1.0)
        with self.assertRaises(ValueError):
            UtilityLossWeights(0.0, 0.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            UtilityLossWeights(1.0, -1.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            UtilityLossWeights(1.0, 0.0, 0.0, 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional locally")
class UtilityTrainingTorchTest(unittest.TestCase):
    def test_combined_loss_and_optimizer_step_are_finite(self) -> None:
        import torch

        torch.manual_seed(13)
        states = (_state("trajectory-a", "state-a", 2), _state("trajectory-b", "state-b", 4))
        padded = build_padded_set_utility_batch(states)
        batch = tensorize_set_utility_batch(padded)
        model = DeepSetsUtilityPredictor(
            SetUtilityDimensions(query=3, context=2, event=4, hidden=12)
        )
        weights = UtilityLossWeights(1.0, 1.0, 0.25, 1.0)
        loss, metrics = utility_predictor_loss(model, batch, weights=weights)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(
            set(metrics),
            {
                "raw_regression",
                "normalized_regression",
                "within_state_ranking",
                "total",
            },
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        step = utility_optimization_step(
            model,
            optimizer,
            batch,
            weights=weights,
            maximum_gradient_norm=1.0,
        )
        self.assertTrue(all(math.isfinite(value) for value in step.values()))


if __name__ == "__main__":
    unittest.main()
