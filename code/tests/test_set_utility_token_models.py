from __future__ import annotations

import unittest

from causalcache.set_utility_token_models import (
    TokenSetUtilityPredictor,
    TokenUtilityModelConfig,
)


class TokenUtilityConfigTest(unittest.TestCase):
    def test_rejects_invalid_family_and_head_geometry(self) -> None:
        with self.assertRaises(ValueError):
            TokenUtilityModelConfig(family="pairwise")
        with self.assertRaises(ValueError):
            TokenUtilityModelConfig(
                family="set_transformer", hidden_size=30, num_heads=8
            )


class TokenUtilityTorchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ModuleNotFoundError as error:
            raise unittest.SkipTest("PyTorch is an optional GPU dependency") from error
        cls.torch = torch

    def _batch(self) -> dict[str, object]:
        torch = self.torch
        batch_size, event_count, source_hidden = 2, 4, 16
        subset_masks = torch.tensor(
            [
                [
                    [False, False, False, False],
                    [True, False, False, False],
                    [False, True, True, False],
                ]
            ]
            * batch_size,
            dtype=torch.bool,
        )
        return {
            "query_visual_tokens": torch.randn(batch_size, 7, source_hidden),
            "query_visual_mask": torch.ones(batch_size, 7, dtype=torch.bool),
            "query_text_tokens": torch.randn(batch_size, 3, source_hidden),
            "query_text_mask": torch.ones(batch_size, 3, dtype=torch.bool),
            "event_visual_tokens": torch.randn(
                batch_size, event_count, 7, source_hidden
            ),
            "event_visual_mask": torch.ones(
                batch_size, event_count, 7, dtype=torch.bool
            ),
            "event_text_tokens": torch.randn(
                batch_size, event_count, 3, source_hidden
            ),
            "event_text_mask": torch.ones(
                batch_size, event_count, 3, dtype=torch.bool
            ),
            "event_numeric_features": torch.randn(batch_size, event_count, 5),
            "event_mask": torch.ones(batch_size, event_count, dtype=torch.bool),
            "subset_masks": subset_masks,
        }

    def test_deepsets_and_set_transformer_share_rich_inputs_and_empty_anchor(self) -> None:
        torch = self.torch
        for family in ("deepsets", "set_transformer"):
            with self.subTest(family=family):
                model = TokenSetUtilityPredictor(
                    TokenUtilityModelConfig(
                        family=family,
                        source_hidden_size=16,
                        numeric_feature_size=5,
                        hidden_size=16,
                        latent_count=4,
                        resampler_layers=1,
                        set_layers=2,
                        num_heads=4,
                        dropout=0.0,
                    )
                )
                predictions = model(**self._batch())
                self.assertEqual(tuple(predictions.shape), (2, 3))
                self.assertTrue(torch.equal(predictions[:, 0], torch.zeros(2)))
                predictions.sum().backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(any(value is not None for value in gradients))

    def test_rejects_selection_of_padded_event(self) -> None:
        batch = self._batch()
        batch["event_mask"][0, 3] = False
        batch["subset_masks"][0, 1, :] = False
        batch["subset_masks"][0, 1, 3] = True
        model = TokenSetUtilityPredictor(
            TokenUtilityModelConfig(
                family="deepsets",
                source_hidden_size=16,
                numeric_feature_size=5,
                hidden_size=16,
                latent_count=4,
                resampler_layers=1,
                set_layers=1,
                num_heads=4,
                dropout=0.0,
            )
        )
        with self.assertRaisesRegex(ValueError, "padded event"):
            model(**batch)


if __name__ == "__main__":
    unittest.main()
