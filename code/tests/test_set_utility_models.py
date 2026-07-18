from __future__ import annotations

import importlib.util
import inspect
import unittest

from causalcache.set_utility_models import (
    DeepSetsUtilityPredictor,
    PairwiseAdditiveUtilityPredictor,
    SetUtilityDimensions,
    SetTransformerUtilityPredictor,
    build_deepsets_utility_predictor,
    build_pairwise_additive_utility_predictor,
    build_set_transformer_utility_predictor,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class SetUtilityDimensionsTest(unittest.TestCase):
    def test_dimensions_are_positive_and_budget_free(self) -> None:
        dimensions = SetUtilityDimensions(
            query=3,
            context=5,
            event=7,
            hidden=11,
        )
        self.assertEqual((dimensions.query, dimensions.context), (3, 5))
        self.assertNotIn("budget", dimensions.__dataclass_fields__)
        for name in ("query", "context", "event", "hidden"):
            values = {
                "query": 3,
                "context": 5,
                "event": 7,
                "hidden": 11,
            }
            values[name] = 0
            with self.assertRaises(ValueError):
                SetUtilityDimensions(**values)
        with self.assertRaises(ValueError):
            PairwiseAdditiveUtilityPredictor(
                dimensions,
                pair_feature_dimension=-1,
            )

    def test_builders_report_the_optional_pytorch_boundary(self) -> None:
        dimensions = SetUtilityDimensions(query=3, context=5, event=7)
        builders = (
            build_pairwise_additive_utility_predictor,
            build_deepsets_utility_predictor,
            build_set_transformer_utility_predictor,
        )
        if TORCH_AVAILABLE:
            for builder in builders:
                self.assertIsNotNone(builder(dimensions))
        else:
            for builder in builders:
                with self.assertRaisesRegex(RuntimeError, "require PyTorch"):
                    builder(dimensions)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional locally")
class SetUtilityModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

    def setUp(self) -> None:
        self.torch.manual_seed(17)
        self.dimensions = SetUtilityDimensions(
            query=3,
            context=2,
            event=4,
            hidden=12,
        )
        self.query = self.torch.randn(2, 3)
        self.context = self.torch.randn(2, 2)
        self.events = self.torch.randn(2, 4, 4)
        self.event_mask = self.torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        )
        self.subsets = self.torch.tensor(
            [
                [
                    [False, False, False, False],
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, True],
                ],
                [
                    [False, False, False, False],
                    [True, False, False, False],
                    [False, True, False, False],
                    [True, True, False, False],
                ],
            ]
        )

    def _models(self):
        return (
            PairwiseAdditiveUtilityPredictor(self.dimensions),
            DeepSetsUtilityPredictor(self.dimensions),
            SetTransformerUtilityPredictor(self.dimensions),
        )

    def test_batched_and_single_subset_shapes_and_empty_utility(self) -> None:
        for model in self._models():
            batched = model(
                self.query,
                self.context,
                self.events,
                self.subsets,
                self.event_mask,
            )
            single = model(
                self.query,
                self.context,
                self.events,
                self.subsets[:, 1],
                self.event_mask,
            )
            self.assertEqual(tuple(batched.shape), (2, 4))
            self.assertEqual(tuple(single.shape), (2,))
            self.assertTrue(self.torch.equal(batched[:, 0], self.torch.zeros(2)))
            self.torch.testing.assert_close(single, batched[:, 1])

    def test_models_are_permutation_invariant(self) -> None:
        permutation = self.torch.tensor([2, 0, 3, 1])
        for model in self._models():
            expected = model(
                self.query,
                self.context,
                self.events,
                self.subsets,
                self.event_mask,
            )
            observed = model(
                self.query,
                self.context,
                self.events[:, permutation],
                self.subsets[:, :, permutation],
                self.event_mask[:, permutation],
            )
            self.torch.testing.assert_close(observed, expected)

    def test_padding_is_semantically_inert(self) -> None:
        query = self.query[:1]
        context = self.context[:1]
        events = self.events[:1, :2]
        subsets = self.subsets[:1, :3, :2]
        padded_events = self.torch.cat(
            (events, self.torch.full((1, 3, 4), 1234.0)), dim=1
        )
        padded_subsets = self.torch.cat(
            (subsets, self.torch.zeros((1, 3, 3), dtype=self.torch.bool)), dim=2
        )
        for model in self._models():
            expected = model(query, context, events, subsets)
            observed = model(
                query,
                context,
                padded_events,
                padded_subsets,
                self.torch.tensor([[True, True, False, False, False]]),
            )
            self.torch.testing.assert_close(observed, expected)

    def test_empty_candidate_universe_is_supported(self) -> None:
        empty_events = self.torch.empty((2, 0, 4))
        empty_subsets = self.torch.empty((2, 3, 0), dtype=self.torch.bool)
        for model in self._models():
            values = model(self.query, self.context, empty_events, empty_subsets)
            self.assertEqual(tuple(values.shape), (2, 3))
            self.assertTrue(self.torch.equal(values, self.torch.zeros_like(values)))

    def test_query_and_context_condition_nonempty_predictions(self) -> None:
        subsets = self.subsets[:, 1:3]
        for model in self._models():
            baseline = model(
                self.query,
                self.context,
                self.events,
                subsets,
                self.event_mask,
            )
            changed_query = model(
                self.query + 2.0,
                self.context,
                self.events,
                subsets,
                self.event_mask,
            )
            changed_context = model(
                self.query,
                self.context - 2.0,
                self.events,
                subsets,
                self.event_mask,
            )
            self.assertFalse(self.torch.allclose(baseline, changed_query))
            self.assertFalse(self.torch.allclose(baseline, changed_context))

    def test_unselected_candidate_context_conditions_subset_utility(self) -> None:
        subsets = self.subsets[:, 1:2]
        changed_events = self.events.clone()
        changed_events[:, 1] += 7.0
        for model in self._models():
            baseline = model(
                self.query,
                self.context,
                self.events,
                subsets,
                self.event_mask,
            )
            changed = model(
                self.query,
                self.context,
                changed_events,
                subsets,
                self.event_mask,
            )
            self.assertFalse(self.torch.allclose(baseline, changed))

    def test_arbitrary_cardinality_outputs_are_differentiable(self) -> None:
        for model in self._models():
            values = model(
                self.query,
                self.context,
                self.events,
                self.subsets,
                self.event_mask,
            )
            loss = values.square().sum()
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertTrue(
                all(bool(self.torch.isfinite(gradient).all()) for gradient in gradients)
            )

    def test_pairwise_model_consumes_explicit_symmetric_pair_features(self) -> None:
        dimensions = SetUtilityDimensions(
            query=3,
            context=2,
            event=4,
            hidden=12,
        )
        model = PairwiseAdditiveUtilityPredictor(
            dimensions,
            pair_feature_dimension=2,
        )
        pair_features = self.torch.randn(2, 4, 4, 2)
        pair_features = 0.5 * (
            pair_features + pair_features.transpose(1, 2)
        )
        pair_features.requires_grad_()
        observed = model(
            self.query,
            self.context,
            self.events,
            self.subsets,
            self.event_mask,
            pair_features=pair_features,
        )
        observed.sum().backward()
        self.assertEqual(tuple(observed.shape), (2, 4))
        self.assertIsNotNone(pair_features.grad)
        self.assertGreater(float(pair_features.grad.abs().sum()), 0.0)

        permutation = self.torch.tensor([2, 0, 3, 1])
        permuted = model(
            self.query,
            self.context,
            self.events[:, permutation],
            self.subsets[:, :, permutation],
            self.event_mask[:, permutation],
            pair_features=pair_features.detach()[:, permutation][
                :, :, permutation
            ],
        )
        self.torch.testing.assert_close(permuted, observed.detach())

        base_events = self.events[:1, :2]
        base_subsets = self.subsets[:1, :3, :2]
        base_pairs = pair_features.detach()[:1, :2, :2]
        padded_events = self.torch.cat(
            (base_events, self.torch.full((1, 3, 4), -987.0)), dim=1
        )
        padded_subsets = self.torch.cat(
            (
                base_subsets,
                self.torch.zeros((1, 3, 3), dtype=self.torch.bool),
            ),
            dim=2,
        )
        padded_pairs = self.torch.randn(1, 5, 5, 2)
        padded_pairs[:, :2, :2] = base_pairs
        expected = model(
            self.query[:1],
            self.context[:1],
            base_events,
            base_subsets,
            pair_features=base_pairs,
        )
        padded = model(
            self.query[:1],
            self.context[:1],
            padded_events,
            padded_subsets,
            self.torch.tensor([[True, True, False, False, False]]),
            pair_features=padded_pairs,
        )
        self.torch.testing.assert_close(padded, expected)

        empty = model(
            self.query,
            self.context,
            self.torch.empty((2, 0, 4)),
            self.torch.empty((2, 3, 0), dtype=self.torch.bool),
            pair_features=self.torch.empty((2, 0, 0, 2)),
        )
        self.assertTrue(self.torch.equal(empty, self.torch.zeros_like(empty)))
        with self.assertRaises(ValueError):
            model(
                self.query,
                self.context,
                self.events,
                self.subsets,
                self.event_mask,
            )

    def test_masks_fail_closed(self) -> None:
        invalid = self.subsets.clone()
        invalid[1, 0, 3] = True
        for model in self._models():
            with self.assertRaises(ValueError):
                model(
                    self.query,
                    self.context,
                    self.events,
                    invalid,
                    self.event_mask,
                )
            with self.assertRaises(TypeError):
                model(
                    self.query,
                    self.context,
                    self.events,
                    self.subsets.float(),
                    self.event_mask,
                )
            with self.assertRaises(ValueError):
                model(
                    self.query.double(),
                    self.context.double(),
                    self.events.double(),
                    self.subsets,
                    self.event_mask,
                )

    def test_model_api_contains_no_budget_input(self) -> None:
        for model_class in (
            PairwiseAdditiveUtilityPredictor,
            DeepSetsUtilityPredictor,
            SetTransformerUtilityPredictor,
        ):
            parameters = inspect.signature(model_class.forward).parameters
            self.assertNotIn("budget", parameters)
            self.assertNotIn("b", parameters)
            model = model_class(self.dimensions)
            with self.assertRaises(TypeError):
                model(
                    self.query,
                    self.context,
                    self.events,
                    self.subsets,
                    self.event_mask,
                    budget=2,
                )

    def test_set_transformer_hyperparameters_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            SetTransformerUtilityPredictor(self.dimensions, num_heads=5)
        with self.assertRaises(ValueError):
            SetTransformerUtilityPredictor(self.dimensions, num_layers=0)
        with self.assertRaises(ValueError):
            SetTransformerUtilityPredictor(self.dimensions, dropout=1.0)


if __name__ == "__main__":
    unittest.main()
