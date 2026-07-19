from __future__ import annotations

import types
import unittest
from dataclasses import replace
from pathlib import Path

import causalcache.policy.gui_owl_v2_vision as vision
from causalcache.policy.gui_owl_v2_vision import (
    L2_NORM_EPSILON,
    MODEL_CLASS_NAME,
    MODEL_FILE_COUNT,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_TOTAL_BYTES,
    SNAPSHOT_MANIFEST_SHA256,
    TRANSFORMERS_SOURCE_SHA256,
    TRANSFORMERS_VERSION,
    VISION_OUTPUT_SIZE,
    VerifiedVisionRuntimeIdentity,
    canonical_image_grid_thw,
    extract_normalized_spatial_merger_embeddings,
    extract_spatial_merger_token_sequences,
    frozen_policy_vision_similarity_from_batch,
    visual_token_geometry,
)


class GUIOwlV2VisionGeometryTest(unittest.TestCase):
    def test_geometry_uses_per_image_raw_and_spatial_merger_boundaries(self) -> None:
        geometry = visual_token_geometry([[1, 4, 6], [2, 2, 4]])
        self.assertEqual(geometry["raw_patch_counts"], (24, 16))
        self.assertEqual(geometry["raw_patch_boundaries"], (0, 24, 40))
        self.assertEqual(geometry["merged_token_counts"], (6, 4))
        self.assertEqual(geometry["merged_token_boundaries"], (0, 6, 10))

    def test_grid_rejects_noninteger_nonpositive_and_nondivisible_rows(self) -> None:
        for value in ([], [[1, 2]], [[1, 2, 3.0]], [[1, 0, 2]], [[1, 3, 3]]):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    canonical_image_grid_thw(value)


class GUIOwlV2VisionTorchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ModuleNotFoundError as error:
            raise unittest.SkipTest("PyTorch is an optional GPU runtime dependency") from error
        cls.torch = torch

    def _model(self, features: list[object]) -> object:
        torch = self.torch

        class VisionConfig:
            depth = 27
            hidden_size = 1152
            out_hidden_size = 4096
            patch_size = 16
            temporal_patch_size = 2
            spatial_merge_size = 2
            deepstack_visual_indexes = [8, 16, 24]

        class Config:
            architectures = [MODEL_CLASS_NAME]
            vision_config = VisionConfig()
            _name_or_path = "/tmp/frozen-gui-owl-test"

        def get_image_features(
            self: object,
            *,
            pixel_values: object,
            image_grid_thw: object,
            return_dict: bool,
        ) -> object:
            self.calls += 1
            self.return_dict_seen = return_dict
            return types.SimpleNamespace(
                pooler_output=tuple(features),
                last_hidden_state=torch.full((20, 1152), 999, dtype=torch.bfloat16),
                deepstack_features=[torch.full((10, 4096), -999, dtype=torch.bfloat16)],
            )

        model_type = type(
            MODEL_CLASS_NAME,
            (),
            {
                "config": Config(),
                "calls": 0,
                "return_dict_seen": False,
                "get_image_features": get_image_features,
            },
        )
        return model_type()

    def _runtime_identity(self) -> VerifiedVisionRuntimeIdentity:
        return VerifiedVisionRuntimeIdentity(
            model_dir=str(Path("/tmp/frozen-gui-owl-test").resolve()),
            model_repo=MODEL_REPO,
            model_revision=MODEL_REVISION,
            snapshot_manifest_sha256=SNAPSHOT_MANIFEST_SHA256,
            verified_model_file_count=MODEL_FILE_COUNT,
            verified_model_total_bytes=MODEL_TOTAL_BYTES,
            transformers_version=TRANSFORMERS_VERSION,
            transformers_source_sha256=tuple(
                sorted(TRANSFORMERS_SOURCE_SHA256.items())
            ),
            _verification_token=vision._VERIFIED_RUNTIME_TOKEN,
        )

    def test_extractor_uses_pooler_output_float32_and_excludes_deepstack(self) -> None:
        torch = self.torch
        grids = torch.tensor([[1, 2, 2]] * 5, dtype=torch.int64)
        pixels = torch.zeros((20, 1536), dtype=torch.float32)
        features = []
        for index in range(5):
            vector = torch.zeros((1, VISION_OUTPUT_SIZE), dtype=torch.bfloat16)
            vector[0, index] = 1
            features.append(vector)
        model = self._model(features)
        batch = extract_normalized_spatial_merger_embeddings(
            model=model,
            pixel_values=pixels,
            image_grid_thw=grids,
            runtime_identity=self._runtime_identity(),
        )
        self.assertEqual(model.calls, 1)
        self.assertTrue(model.return_dict_seen)
        self.assertEqual(batch.feature_field, "pooler_output")
        self.assertTrue(batch.deepstack_features_excluded)
        self.assertEqual(batch.output_dtype, "torch.bfloat16")
        self.assertEqual(batch.reduction_dtype, "torch.float32")
        self.assertEqual(tuple(batch.normalized_embeddings.shape), (5, 4096))
        result = frozen_policy_vision_similarity_from_batch(
            batch,
            event_image_indices={1: 0, 2: 1, 3: 2, 4: 3},
            current_image_index=4,
        )
        self.assertEqual(result.ranked_event_step_ids, (1, 2, 3, 4))
        self.assertEqual(result.selected_event_step_ids, (1, 2))

    def test_token_extractor_preserves_every_bfloat16_merger_row(self) -> None:
        torch = self.torch
        grids = torch.tensor([[1, 4, 4], [1, 2, 4]], dtype=torch.int64)
        pixels = torch.zeros((24, 1536), dtype=torch.float32)
        features = (
            torch.arange(4 * VISION_OUTPUT_SIZE, dtype=torch.float32)
            .reshape(4, VISION_OUTPUT_SIZE)
            .to(torch.bfloat16),
            torch.full((2, VISION_OUTPUT_SIZE), 7.0, dtype=torch.bfloat16),
        )
        batch = extract_spatial_merger_token_sequences(
            model=self._model(list(features)),
            pixel_values=pixels,
            image_grid_thw=grids,
            runtime_identity=self._runtime_identity(),
        )
        self.assertEqual(batch.merged_token_counts, (4, 2))
        self.assertEqual(batch.hidden_size, VISION_OUTPUT_SIZE)
        self.assertEqual(batch.output_dtype, "torch.bfloat16")
        self.assertEqual(batch.feature_field, "pooler_output")
        self.assertTrue(batch.deepstack_features_excluded)
        self.assertEqual(len(batch.token_sequences), 2)
        self.assertTrue(torch.equal(batch.token_sequences[0], features[0]))
        self.assertTrue(torch.equal(batch.token_sequences[1], features[1]))

    def test_extractor_rejects_wrong_pooler_shape_dtype_nonfinite_and_zero_norm(self) -> None:
        torch = self.torch
        grids = torch.tensor([[1, 2, 2]], dtype=torch.int64)
        pixels = torch.zeros((4, 1536), dtype=torch.float32)
        cases = (
            torch.zeros((2, VISION_OUTPUT_SIZE), dtype=torch.bfloat16),
            torch.zeros((1, VISION_OUTPUT_SIZE), dtype=torch.float32),
            torch.full((1, VISION_OUTPUT_SIZE), float("nan"), dtype=torch.bfloat16),
            torch.zeros((1, VISION_OUTPUT_SIZE), dtype=torch.bfloat16),
        )
        for feature in cases:
            with self.subTest(shape=tuple(feature.shape), dtype=str(feature.dtype)):
                with self.assertRaises((TypeError, ValueError)):
                    extract_normalized_spatial_merger_embeddings(
                        model=self._model([feature]),
                        pixel_values=pixels,
                        image_grid_thw=grids,
                        runtime_identity=self._runtime_identity(),
                    )
        self.assertEqual(L2_NORM_EPSILON, 1e-12)

    def test_extractor_requires_verified_snapshot_and_batch_metadata(self) -> None:
        torch = self.torch
        grids = torch.tensor([[1, 2, 2]] * 5, dtype=torch.int64)
        pixels = torch.zeros((20, 1536), dtype=torch.float32)
        features = []
        for index in range(5):
            value = torch.zeros((1, VISION_OUTPUT_SIZE), dtype=torch.bfloat16)
            value[0, index] = 1
            features.append(value)
        forged = replace(self._runtime_identity(), _verification_token=object())
        with self.assertRaisesRegex(ValueError, "verified frozen runtime"):
            extract_normalized_spatial_merger_embeddings(
                model=self._model(features),
                pixel_values=pixels,
                image_grid_thw=grids,
                runtime_identity=forged,
            )
        batch = extract_normalized_spatial_merger_embeddings(
            model=self._model(features),
            pixel_values=pixels,
            image_grid_thw=grids,
            runtime_identity=self._runtime_identity(),
        )
        with self.assertRaisesRegex(ValueError, "metadata drifted"):
            frozen_policy_vision_similarity_from_batch(
                replace(batch, feature_field="deepstack_features"),
                event_image_indices={1: 0, 2: 1, 3: 2, 4: 3},
                current_image_index=4,
            )


if __name__ == "__main__":
    unittest.main()
