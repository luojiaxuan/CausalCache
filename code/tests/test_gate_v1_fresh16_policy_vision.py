from __future__ import annotations

import contextlib
import json
import sys
import types
import unittest
from unittest import mock

import causalcache.gate_v1_fresh16_policy_vision as fresh_module
from causalcache.gate_v1_fresh16_policy_vision import (
    Fresh16GUIOwlV22VisionFeatureRuntime,
    _require_variable_rgb_pil_images,
    _validate_variable_cpu_image_processor_output,
    fresh16_policy_vision_similarity_from_batch,
)
from causalcache.policy.gui_owl_v2_2_vision_runtime import (
    GUI_OWL_V2_2_VISION_FORBIDDEN_OPERATION_KEYS,
    GUIOwlV22VisionFeatureRuntime,
)
from causalcache.policy.gui_owl_v2_vision import (
    OUTPUT_DTYPE,
    REDUCTION_DTYPE,
    VISION_OUTPUT_SIZE,
    VisionEmbeddingBatch,
)
from causalcache.restoration_v2_baselines import BaselineSelection


class _FakeDType:
    pass


class _FakeDevice:
    def __init__(self, value: str) -> None:
        self.value = value
        self.type = value.split(":", 1)[0]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeDevice) and self.value == other.value


class _FakeTensor:
    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: object,
        device: _FakeDevice | None = None,
        data: object | None = None,
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device or _FakeDevice("cpu")
        self.data = data
        self.contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self.contiguous

    def detach(self) -> _FakeTensor:
        return self

    def to(
        self,
        device: object,
        non_blocking: bool = False,
    ) -> _FakeTensor:
        del non_blocking
        target = _FakeDevice(device) if isinstance(device, str) else device
        return _FakeTensor(
            shape=self.shape,
            dtype=self.dtype,
            device=target,
            data=self.data,
            contiguous=self.contiguous,
        )

    def tolist(self) -> object:
        return self.data


class _FakeTorch:
    def __init__(self) -> None:
        self.Tensor = _FakeTensor
        self.float32 = _FakeDType()
        self.int64 = _FakeDType()


class _FakePILImage:
    def __init__(self, mode: str = "RGB") -> None:
        self.mode = mode


class _FakeImageProcessor:
    def __init__(self, output: dict[str, object]) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return self.output


class _BoolTensor:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> bool:
        return self.value


class _NormTensor:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _IndexedEvents:
    def __init__(self, source: _EmbeddingTensor, indices: list[int]) -> None:
        self.source = source
        self.indices = indices


class _IndexedCurrent:
    def __init__(self, source: _EmbeddingTensor, index: int) -> None:
        self.source = source
        self.index = index


class _ScoreTensor:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def detach(self) -> _ScoreTensor:
        return self

    def cpu(self) -> _ScoreTensor:
        return self

    def tolist(self) -> list[float]:
        return self.values


class _EmbeddingTensor(_FakeTensor):
    def __init__(self, *, image_count: int, dtype: object, scores: list[float]) -> None:
        super().__init__(
            shape=(image_count, VISION_OUTPUT_SIZE),
            dtype=dtype,
            device=_FakeDevice("cuda:0"),
        )
        self.scores = scores

    def __getitem__(self, index: object) -> object:
        if isinstance(index, list):
            return _IndexedEvents(self, index)
        if type(index) is int:
            return _IndexedCurrent(self, index)
        raise TypeError("unsupported fake embedding index")


class _SimilarityTorch(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.Tensor = _FakeTensor
        self.float32 = _FakeDType()
        self.linalg = types.SimpleNamespace(vector_norm=self._vector_norm)

    @staticmethod
    def _vector_norm(value: object, *, dim: int) -> _NormTensor:
        if not isinstance(value, _EmbeddingTensor) or dim != 1:
            raise AssertionError("unexpected vector_norm call")
        return _NormTensor([1.0] * int(value.shape[0]))

    @staticmethod
    def isfinite(_: object) -> _BoolTensor:
        return _BoolTensor(True)

    @staticmethod
    def ones_like(value: _NormTensor) -> _NormTensor:
        return _NormTensor([1.0] * len(value.values))

    @staticmethod
    def allclose(
        left: _NormTensor,
        right: _NormTensor,
        *,
        rtol: float,
        atol: float,
    ) -> bool:
        return left.values == right.values and rtol == 1e-5 and atol == 1e-6

    @staticmethod
    def inference_mode() -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    @staticmethod
    def matmul(events: _IndexedEvents, current: _IndexedCurrent) -> _ScoreTensor:
        if events.source is not current.source:
            raise AssertionError("event and current vectors must share a batch")
        if current.index != int(events.source.shape[0]) - 1:
            raise AssertionError("current image must be the final image")
        return _ScoreTensor([events.source.scores[index] for index in events.indices])


def _pil_modules() -> dict[str, types.ModuleType]:
    pil = types.ModuleType("PIL")
    image = types.ModuleType("PIL.Image")
    image.Image = _FakePILImage
    pil.Image = image
    return {"PIL": pil, "PIL.Image": image}


def _processor_tensors(
    torch: _FakeTorch,
    *,
    image_count: int,
) -> tuple[_FakeTensor, _FakeTensor]:
    grid_rows = [(1, 2, 2)] * image_count
    return (
        _FakeTensor(
            shape=(4 * image_count, 1536),
            dtype=torch.float32,
        ),
        _FakeTensor(
            shape=(image_count, 3),
            dtype=torch.int64,
            data=grid_rows,
        ),
    )


def _runtime(
    *,
    torch: _FakeTorch,
    image_processor: _FakeImageProcessor,
) -> Fresh16GUIOwlV22VisionFeatureRuntime:
    runtime = object.__new__(Fresh16GUIOwlV22VisionFeatureRuntime)
    runtime.torch = torch
    runtime.device = _FakeDevice("cuda:0")
    runtime.image_processor = image_processor
    runtime.model = object()
    runtime.runtime_identity = object()
    runtime.metadata = {"frozen": True, "feature_only": True}
    runtime._image_processor_batch_count = 0
    runtime._vision_feature_forward_count = 0
    runtime._forbidden_operation_counts = {
        key: 0 for key in GUI_OWL_V2_2_VISION_FORBIDDEN_OPERATION_KEYS
    }
    return runtime


def _embedding_batch(
    *,
    torch: _SimilarityTorch,
    image_count: int,
    scores: list[float],
) -> VisionEmbeddingBatch:
    return VisionEmbeddingBatch(
        normalized_embeddings=_EmbeddingTensor(
            image_count=image_count,
            dtype=torch.float32,
            scores=scores,
        ),
        image_grid_thw=((1, 2, 2),) * image_count,
        raw_patch_boundaries=tuple(range(image_count + 1)),
        merged_token_boundaries=tuple(range(image_count + 1)),
        merged_token_counts=(1,) * image_count,
        output_dtype=OUTPUT_DTYPE,
        reduction_dtype=REDUCTION_DTYPE,
        hidden_size=VISION_OUTPUT_SIZE,
        feature_field="pooler_output",
        deepstack_features_excluded=True,
    )


class Fresh16PolicyVisionTest(unittest.TestCase):
    def test_n4_runtime_and_selector_delegate_to_frozen_paths_exactly(self) -> None:
        runtime = object.__new__(Fresh16GUIOwlV22VisionFeatureRuntime)
        images = [object()] * 5
        frozen_result = {
            "scores_by_event_step": {"1": 0.4, "2": 0.3, "3": 0.2, "4": 0.1},
            "ranked_event_step_ids": [1, 2, 3, 4],
            "selected_event_step_ids": [1, 2],
        }
        with mock.patch.object(
            GUIOwlV22VisionFeatureRuntime,
            "score_five_images",
            autospec=True,
            return_value=frozen_result,
        ) as frozen_score:
            observed = runtime.score_images(
                images,
                event_step_ids=(1, 2, 3, 4),
                feature_repeats=2,
            )
        self.assertIs(observed, frozen_result)
        frozen_score.assert_called_once_with(runtime, images, feature_repeats=2)

        batch = object()
        selection = BaselineSelection(
            scores_by_event_step=((1, 0.4), (2, 0.3), (3, 0.2), (4, 0.1)),
            ranked_event_step_ids=(1, 2, 3, 4),
            selected_event_step_ids=(1, 2),
        )
        mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        with mock.patch.object(
            fresh_module,
            "frozen_policy_vision_similarity_from_batch",
            return_value=selection,
        ) as frozen_selector:
            observed_selection = fresh16_policy_vision_similarity_from_batch(
                batch,
                event_step_ids=(1, 2, 3, 4),
                event_image_indices=mapping,
                current_image_index=4,
            )
        self.assertIs(observed_selection, selection)
        frozen_selector.assert_called_once_with(
            batch,
            event_image_indices=mapping,
            current_image_index=4,
        )

    def test_requires_exact_natural_rgb_image_batches(self) -> None:
        with mock.patch.dict(sys.modules, _pil_modules()):
            for image_count in (3, 4, 5):
                images = [_FakePILImage()] * image_count
                self.assertEqual(
                    _require_variable_rgb_pil_images(
                        images,
                        image_count=image_count,
                    ),
                    tuple(images),
                )
            invalid = (
                ([_FakePILImage()] * 2, 3),
                ([_FakePILImage()] * 4, 3),
                ([_FakePILImage()] * 2 + [_FakePILImage("L")], 3),
                ([_FakePILImage()] * 2 + [object()], 3),
            )
            for images, image_count in invalid:
                with self.subTest(length=len(images)), self.assertRaises(
                    (TypeError, ValueError)
                ):
                    _require_variable_rgb_pil_images(
                        images,
                        image_count=image_count,
                    )
        for image_count in (2, 6, True):
            with self.subTest(image_count=image_count), self.assertRaises(
                (TypeError, ValueError)
            ):
                _require_variable_rgb_pil_images([], image_count=image_count)

    def test_variable_processor_geometry_accepts_only_three_four_or_five(self) -> None:
        torch = _FakeTorch()
        for image_count in (3, 4, 5):
            pixels, grid = _processor_tensors(torch, image_count=image_count)
            returned_pixels, returned_grid, geometry = (
                _validate_variable_cpu_image_processor_output(
                    {"pixel_values": pixels, "image_grid_thw": grid},
                    torch=torch,
                    image_count=image_count,
                )
            )
            self.assertIs(returned_pixels, pixels)
            self.assertIs(returned_grid, grid)
            self.assertEqual(len(geometry["image_grid_thw"]), image_count)
            self.assertEqual(geometry["raw_patch_counts"], (4,) * image_count)

        pixels, grid = _processor_tensors(torch, image_count=3)
        invalid = (
            ({"pixel_values": pixels, "image_grid_thw": grid}, 4),
            (
                {
                    "pixel_values": pixels,
                    "image_grid_thw": grid,
                    "input_ids": object(),
                },
                3,
            ),
            (
                {
                    "pixel_values": _FakeTensor(
                        shape=(11, 1536),
                        dtype=torch.float32,
                    ),
                    "image_grid_thw": grid,
                },
                3,
            ),
        )
        for encoded, image_count in invalid:
            with self.subTest(image_count=image_count), self.assertRaises(ValueError):
                _validate_variable_cpu_image_processor_output(
                    encoded,
                    torch=torch,
                    image_count=image_count,
                )

    def test_n2_n3_runtime_preserves_geometry_and_operation_counts(self) -> None:
        for candidates in ((1, 2), (1, 2, 3)):
            image_count = len(candidates) + 1
            torch = _FakeTorch()
            pixels, grid = _processor_tensors(torch, image_count=image_count)
            processor = _FakeImageProcessor(
                {"pixel_values": pixels, "image_grid_thw": grid}
            )
            runtime = _runtime(torch=torch, image_processor=processor)
            images = [_FakePILImage()] * image_count
            batches = [
                types.SimpleNamespace(normalized_embeddings=object()),
                types.SimpleNamespace(normalized_embeddings=object()),
            ]
            score_rows = tuple(
                (step_id, 1.0 / step_id) for step_id in candidates
            )
            selection = BaselineSelection(
                scores_by_event_step=score_rows,
                ranked_event_step_ids=candidates,
                selected_event_step_ids=(1, 2),
            )
            with (
                mock.patch.dict(sys.modules, _pil_modules()),
                mock.patch.object(
                    fresh_module,
                    "extract_normalized_spatial_merger_embeddings",
                    side_effect=batches,
                ) as extract,
                mock.patch.object(
                    fresh_module,
                    "fresh16_policy_vision_similarity_from_batch",
                    return_value=selection,
                ) as similarity,
                mock.patch.object(
                    fresh_module,
                    "_variable_normalized_norm_range",
                    side_effect=((0.99999, 1.00001), (0.99998, 1.00002)),
                ),
            ):
                result = runtime.score_images(
                    images,
                    event_step_ids=candidates,
                    feature_repeats=2,
                )

            self.assertEqual(
                processor.calls,
                [{"images": images, "return_tensors": "pt"}],
            )
            self.assertEqual(extract.call_count, 2)
            self.assertEqual(similarity.call_count, 2)
            for call in similarity.call_args_list:
                self.assertEqual(call.kwargs["event_step_ids"], candidates)
                self.assertEqual(
                    call.kwargs["event_image_indices"],
                    {step_id: index for index, step_id in enumerate(candidates)},
                )
                self.assertEqual(call.kwargs["current_image_index"], image_count - 1)
            self.assertEqual(len(result["image_grid_thw"]), image_count)
            self.assertEqual(result["raw_patch_counts"], [4] * image_count)
            self.assertEqual(result["selected_event_step_ids"], [1, 2])
            self.assertTrue(result["same_device_replay"]["performed"])
            self.assertEqual(
                result["operation_counts"],
                {
                    "image_processor_batch_count": 1,
                    "policy_vision_feature_forward_count": 2,
                    **{
                        key: 0
                        for key in GUI_OWL_V2_2_VISION_FORBIDDEN_OPERATION_KEYS
                    },
                },
            )
            self.assertEqual(runtime.operation_counts, result["operation_counts"])
            json.dumps(result, allow_nan=False, sort_keys=True)

    def test_n2_and_n3_similarity_use_frozen_ties_and_fail_closed(self) -> None:
        torch = _SimilarityTorch()
        with mock.patch.dict(sys.modules, {"torch": torch}):
            batch = _embedding_batch(
                torch=torch,
                image_count=4,
                scores=[0.5, 0.5000000001, 0.1],
            )
            selection = fresh16_policy_vision_similarity_from_batch(
                batch,
                event_step_ids=(1, 2, 3),
                event_image_indices={1: 0, 2: 1, 3: 2},
                current_image_index=3,
            )
            self.assertEqual(
                selection.scores_by_event_step,
                ((1, 0.5), (2, 0.5000000001), (3, 0.1)),
            )
            self.assertEqual(selection.ranked_event_step_ids, (1, 2, 3))
            self.assertEqual(selection.selected_event_step_ids, (1, 2))

            with self.assertRaisesRegex(ValueError, "distinct"):
                fresh16_policy_vision_similarity_from_batch(
                    batch,
                    event_step_ids=(1, 2, 3),
                    event_image_indices={1: 0, 2: 1, 3: 3},
                    current_image_index=3,
                )
            with self.assertRaisesRegex(ValueError, "metadata"):
                fresh16_policy_vision_similarity_from_batch(
                    _embedding_batch(
                        torch=torch,
                        image_count=3,
                        scores=[0.5, 0.4],
                    ),
                    event_step_ids=(1, 2, 3),
                    event_image_indices={1: 0, 2: 1, 3: 2},
                    current_image_index=3,
                )

        for candidates in ((1,), (1, 3), (1, 2, 3, 4, 5)):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                fresh16_policy_vision_similarity_from_batch(
                    object(),
                    event_step_ids=candidates,
                    event_image_indices={},
                    current_image_index=0,
                )


if __name__ == "__main__":
    unittest.main()
