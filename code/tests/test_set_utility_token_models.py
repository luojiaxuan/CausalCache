from __future__ import annotations

import math
import unittest

from causalcache.set_utility_token_models import (
    TokenSetUtilityPredictor,
    TokenUtilityModelConfig,
)
from scripts.train_set_utility_token_predictor import (
    _collate,
    _loss,
    _seed_training_runtime,
    _targets,
    _trajectory_uniform_epoch,
)


class TokenUtilityConfigTest(unittest.TestCase):
    def test_rejects_invalid_family_and_head_geometry(self) -> None:
        with self.assertRaises(ValueError):
            TokenUtilityModelConfig(family="pairwise")
        with self.assertRaises(ValueError):
            TokenUtilityModelConfig(
                family="set_transformer", hidden_size=30, num_heads=8
            )

    def test_training_epoch_is_trajectory_uniform_and_covers_long_trajectory(self) -> None:
        states = tuple(
            {"state_id": f"long-{index}", "trajectory_id": "long"}
            for index in range(3)
        ) + ({"state_id": "short-0", "trajectory_id": "short"},)
        order = _trajectory_uniform_epoch(states, seed=7)
        self.assertEqual(len(order), 6)
        self.assertEqual(
            {name: sum(row["trajectory_id"] == name for row in order) for name in ("long", "short")},
            {"long": 3, "short": 3},
        )
        self.assertEqual(
            {row["state_id"] for row in order if row["trajectory_id"] == "long"},
            {"long-0", "long-1", "long-2"},
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

    def test_training_seed_precedes_reproducible_model_initialization(self) -> None:
        config = TokenUtilityModelConfig(
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
        _seed_training_runtime(self.torch, 17)
        left = TokenSetUtilityPredictor(config)
        _seed_training_runtime(self.torch, 17)
        right = TokenSetUtilityPredictor(config)
        self.assertTrue(
            all(
                self.torch.equal(left_value, right_value)
                for left_value, right_value in zip(
                    left.state_dict().values(),
                    right.state_dict().values(),
                    strict=True,
                )
            )
        )

    def test_normalization_floor_bounds_tiny_baseline_scale(self) -> None:
        state = {
            "candidate_event_step_ids": [1],
            "distance_rows": [
                {"coalition_event_step_ids": [], "distance": 1e-6},
                {"coalition_event_step_ids": [1], "distance": 0.0},
            ],
        }
        _, raw, normalized, (scale, valid) = _targets(
            state, self.torch, normalization_floor=0.01
        )
        self.assertTrue(valid)
        self.assertEqual(scale, 0.01)
        self.assertEqual(raw.tolist()[0], 0.0)
        self.assertAlmostEqual(raw.tolist()[1], 1e-6, places=10)
        self.assertAlmostEqual(normalized.tolist()[1], 1e-4, places=10)

    def test_collate_pads_variable_sets_and_labels_without_dropping_rows(self) -> None:
        torch = self.torch

        class Cache:
            def visual(self, key: str) -> object:
                length = int(key.rsplit("-", 1)[1])
                return torch.full((length, 16), length, dtype=torch.bfloat16)

            def text(self, key: str) -> object:
                length = int(key.rsplit("-", 1)[1])
                return torch.full((length, 16), length, dtype=torch.bfloat16)

        states = []
        for index, event_count in enumerate((5, 7)):
            event_ids = tuple(range(1, event_count + 1))
            subsets = (
                ((), (1,), (1, event_count))
                if event_count == 5
                else ((), (2,), (1, 7), (1, 3, 7))
            )
            states.append(
                {
                    "candidate_event_step_ids": list(event_ids),
                    "current_image_key": f"query-{5 + index}",
                    "distance_rows": [
                        {
                            "coalition_event_step_ids": list(subset),
                            "distance": 1.0 - 0.1 * len(subset),
                        }
                        for subset in subsets
                    ],
                    "event_image_keys": [
                        f"event-{3 + event_index}"
                        for event_index in range(event_count)
                    ],
                    "event_numeric_features": [[0.0] * 5 for _ in event_ids],
                    "event_text_keys": [
                        f"text-{1 + event_index}"
                        for event_index in range(event_count)
                    ],
                    "instruction_text_key": f"instruction-{2 + index}",
                    "state_id": f"state-{index}",
                    "trajectory_id": f"trajectory-{index}",
                }
            )
        batch = _collate(states, cache=Cache(), device="cpu", torch=torch)
        model_inputs = batch["model"]
        self.assertEqual(tuple(model_inputs["query_visual_tokens"].shape), (2, 6, 16))
        self.assertEqual(tuple(model_inputs["event_visual_tokens"].shape), (2, 7, 9, 16))
        self.assertEqual(model_inputs["query_visual_mask"].sum(dim=1).tolist(), [5, 6])
        self.assertEqual(
            model_inputs["event_visual_mask"].sum(dim=2).tolist(),
            [[3, 4, 5, 6, 7, 0, 0], [3, 4, 5, 6, 7, 8, 9]],
        )
        self.assertEqual(model_inputs["event_mask"].sum(dim=1).tolist(), [5, 7])
        self.assertEqual(batch["label_mask"].sum(dim=1).tolist(), [3, 4])
        self.assertFalse(bool(model_inputs["subset_masks"][0, :, 5:].any()))

        model = TokenSetUtilityPredictor(
            TokenUtilityModelConfig(
                family="set_transformer",
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
        for key in (
            "query_visual_tokens",
            "query_text_tokens",
            "event_visual_tokens",
            "event_text_tokens",
        ):
            model_inputs[key] = model_inputs[key].float()
        predictions = model(**model_inputs)
        loss, metrics = _loss(
            predictions,
            batch,
            loss_config={
                "raw_smooth_l1_beta": 0.1,
                "normalized_smooth_l1_beta": 0.1,
                "raw_regression": 1.0,
                "normalized_regression": 1.0,
                "within_state_ranking": 0.2,
            },
            trajectory_weights=torch.ones(2),
            torch=torch,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
