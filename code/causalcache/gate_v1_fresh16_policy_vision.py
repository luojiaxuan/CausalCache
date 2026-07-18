"""Variable-history GUI-Owl policy-vision scoring for fresh-16 evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.gate_v1_fresh16_heuristics import (
    FRESH16_NATURAL_CANDIDATE_PREFIXES,
    policy_vision_similarity_selection,
    validate_natural_candidate_event_step_ids,
)
from causalcache.policy.gui_owl_v2_2_vision_runtime import (
    GUI_OWL_V2_2_VISION_FORBIDDEN_OPERATION_KEYS,
    GUI_OWL_V2_2_VISION_MAX_FEATURE_REPEATS,
    GUI_OWL_V2_2_VISION_PROCESSOR_KEYS,
    GUIOwlV22VisionFeatureRuntime,
    _json_copy,
    _move_exact_inputs_to_device,
)
from causalcache.policy.gui_owl_v2_vision import (
    OUTPUT_DTYPE,
    REDUCTION_DTYPE,
    VISION_OUTPUT_SIZE,
    VISION_PATCH_SIZE,
    VISION_SPATIAL_MERGE_SIZE,
    VISION_TEMPORAL_PATCH_SIZE,
    VisionEmbeddingBatch,
    extract_normalized_spatial_merger_embeddings,
    frozen_policy_vision_similarity_from_batch,
    visual_token_geometry,
)
from causalcache.restoration_v2_baselines import BaselineSelection


FRESH16_POLICY_VISION_RUNTIME_ID = (
    "causalcache_gate_v1_fresh16_variable_history_policy_vision_v1"
)
FRESH16_POLICY_VISION_IMAGE_COUNTS = (3, 4, 5)


def _validate_image_count(image_count: int) -> int:
    if type(image_count) is not int:
        raise TypeError("policy-vision image count must be an integer")
    if image_count not in FRESH16_POLICY_VISION_IMAGE_COUNTS:
        raise ValueError(
            "fresh-16 policy vision requires exactly three, four, or five images"
        )
    return image_count


def _require_variable_rgb_pil_images(
    images: Any,
    *,
    image_count: int,
) -> tuple[Any, ...]:
    """Materialize and validate one natural history plus its current screenshot."""
    expected_count = _validate_image_count(image_count)
    if isinstance(images, (str, bytes, bytearray, Mapping)):
        raise TypeError("policy-vision input must be a sequence of PIL images")
    try:
        materialized = tuple(images)
    except TypeError as error:
        raise TypeError(
            "policy-vision input must be a sequence of PIL images"
        ) from error
    if len(materialized) != expected_count:
        raise ValueError(
            f"policy-vision input must contain exactly {expected_count} images"
        )
    try:
        from PIL import Image as PILImage
    except ModuleNotFoundError as error:
        raise RuntimeError("policy-vision runtime requires Pillow") from error
    for index, image in enumerate(materialized):
        if not isinstance(image, PILImage.Image):
            raise TypeError(f"policy-vision image {index} must be a PIL.Image.Image")
        if image.mode != "RGB":
            raise ValueError(f"policy-vision image {index} must already be RGB")
    return materialized


def _validate_variable_cpu_image_processor_output(
    encoded: Any,
    *,
    torch: Any,
    image_count: int,
) -> tuple[Any, Any, dict[str, Any]]:
    """Validate the frozen processor schema for one of the natural batch sizes."""
    expected_count = _validate_image_count(image_count)
    if not isinstance(encoded, Mapping):
        raise TypeError("AutoImageProcessor output must be a mapping")
    if set(encoded) != GUI_OWL_V2_2_VISION_PROCESSOR_KEYS:
        raise ValueError(
            "AutoImageProcessor output must contain exactly pixel_values and "
            "image_grid_thw"
        )
    pixel_values = encoded["pixel_values"]
    image_grid_thw = encoded["image_grid_thw"]
    for name, value in (
        ("pixel_values", pixel_values),
        ("image_grid_thw", image_grid_thw),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if getattr(value.device, "type", None) != "cpu":
            raise ValueError(f"{name} must be validated on CPU before CUDA transfer")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous on CPU")
    if pixel_values.dtype is not torch.float32:
        raise ValueError("CPU pixel_values must be torch.float32")
    if image_grid_thw.dtype is not torch.int64:
        raise ValueError("CPU image_grid_thw must be torch.int64")
    if pixel_values.ndim != 2:
        raise ValueError("CPU pixel_values must be rank 2")
    if image_grid_thw.ndim != 2 or tuple(image_grid_thw.shape) != (
        expected_count,
        3,
    ):
        raise ValueError(
            f"CPU image_grid_thw must have shape ({expected_count}, 3)"
        )
    geometry = visual_token_geometry(image_grid_thw)
    raw_patch_counts = geometry["raw_patch_counts"]
    if len(raw_patch_counts) != expected_count:
        raise ValueError(
            f"image grid must describe exactly {expected_count} images"
        )
    packed_patch_width = (
        3 * VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE
    )
    if tuple(pixel_values.shape) != (sum(raw_patch_counts), packed_patch_width):
        raise ValueError("CPU pixel_values shape differs from frozen visual geometry")
    return pixel_values, image_grid_thw, geometry


def _variable_normalized_norm_range(
    embeddings: Any,
    *,
    torch: Any,
    image_count: int,
) -> tuple[float, float]:
    expected_count = _validate_image_count(image_count)
    with torch.inference_mode():
        norms = torch.linalg.vector_norm(embeddings, dim=1)
        values = norms.detach().to(device="cpu").tolist()
    if len(values) != expected_count:
        raise ValueError("normalized vision embedding count drifted")
    finite = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in finite):
        raise ValueError("normalized vision embedding norms are non-finite")
    return min(finite), max(finite)


def fresh16_policy_vision_similarity_from_batch(
    batch: VisionEmbeddingBatch,
    *,
    event_step_ids: Sequence[int],
    event_image_indices: Mapping[int, int],
    current_image_index: int,
) -> BaselineSelection:
    """Compute natural-history cosines with the frozen ranking and tie rules."""
    candidates = validate_natural_candidate_event_step_ids(event_step_ids)
    if candidates == FRESH16_NATURAL_CANDIDATE_PREFIXES[-1]:
        # note (luojiaxuan): The five-image branch deliberately calls the exact
        # historical scorer. This keeps the already validated n=4 numerical,
        # metadata, ordering, and tolerance semantics byte-for-byte unchanged.
        return frozen_policy_vision_similarity_from_batch(
            batch,
            event_image_indices=event_image_indices,
            current_image_index=current_image_index,
        )
    if not isinstance(event_image_indices, Mapping):
        raise TypeError("event_image_indices must be a mapping")
    if set(event_image_indices) != set(candidates):
        raise ValueError("event_image_indices must cover exactly the candidate events")
    if any(type(value) is not int for value in event_image_indices.values()):
        raise TypeError("event image indices must be integers")
    if type(current_image_index) is not int:
        raise TypeError("current image index must be an integer")
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("GUI-Owl vision similarity requires PyTorch") from error
    embeddings = batch.normalized_embeddings
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise TypeError("normalized_embeddings must be a rank-2 tensor")
    if (
        embeddings.dtype is not torch.float32
        or int(embeddings.shape[1]) != VISION_OUTPUT_SIZE
    ):
        raise ValueError("normalized embedding dtype or hidden size drifted")
    image_count = len(candidates) + 1
    if (
        batch.feature_field != "pooler_output"
        or not batch.deepstack_features_excluded
        or batch.output_dtype != OUTPUT_DTYPE
        or batch.reduction_dtype != REDUCTION_DTYPE
        or batch.hidden_size != VISION_OUTPUT_SIZE
        or len(batch.image_grid_thw) != image_count
        or int(embeddings.shape[0]) != image_count
    ):
        raise ValueError("vision embedding batch metadata drifted")
    norms = torch.linalg.vector_norm(embeddings, dim=1)
    if not torch.isfinite(norms).all() or not torch.allclose(
        norms,
        torch.ones_like(norms),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("vision embeddings must remain L2-normalized")
    indices = tuple(event_image_indices[step_id] for step_id in candidates)
    all_indices = (*indices, current_image_index)
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("candidate and current image indices must be distinct")
    if set(all_indices) != set(range(image_count)):
        raise ValueError("candidate and current image indices must cover the batch")
    with torch.inference_mode():
        event_vectors = embeddings[list(indices)]
        current = embeddings[current_image_index]
        scores_tensor = torch.matmul(event_vectors, current)
        if not torch.isfinite(scores_tensor).all():
            raise ValueError("vision baseline cosine contains non-finite values")
        scores = [float(value) for value in scores_tensor.detach().cpu().tolist()]
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("vision baseline scalar cosine contains non-finite values")
    return policy_vision_similarity_selection(
        dict(zip(candidates, scores, strict=True)),
        event_step_ids=candidates,
    )


class Fresh16GUIOwlV22VisionFeatureRuntime(GUIOwlV22VisionFeatureRuntime):
    """Pinned feature-only runtime extended to natural n=2/3/4 histories."""

    def score_images(
        self,
        images: Sequence[Any],
        *,
        event_step_ids: Sequence[int],
        feature_repeats: int = 1,
    ) -> dict[str, Any]:
        """Score every event-post image against the final current screenshot."""
        candidates = validate_natural_candidate_event_step_ids(event_step_ids)
        if candidates == FRESH16_NATURAL_CANDIDATE_PREFIXES[-1]:
            # note (luojiaxuan): Delegating the complete five-image call avoids a
            # second implementation of the historical n=4 execution path. The
            # fresh-16 extension only owns the previously unsupported n=2/3 cases.
            return super().score_five_images(
                images,
                feature_repeats=feature_repeats,
            )
        if type(feature_repeats) is not int or not (
            1 <= feature_repeats <= GUI_OWL_V2_2_VISION_MAX_FEATURE_REPEATS
        ):
            raise ValueError("feature_repeats must be one or two")
        self._assert_forbidden_operations_zero()
        image_count = len(candidates) + 1
        validated_images = _require_variable_rgb_pil_images(
            images,
            image_count=image_count,
        )
        self._image_processor_batch_count += 1
        encoded = self.image_processor(
            images=list(validated_images),
            return_tensors="pt",
        )
        pixel_values, image_grid_thw, cpu_geometry = (
            _validate_variable_cpu_image_processor_output(
                encoded,
                torch=self.torch,
                image_count=image_count,
            )
        )
        device_pixels, device_grid = _move_exact_inputs_to_device(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            device=self.device,
            torch=self.torch,
        )
        repeat_results = []
        for repeat_index in range(feature_repeats):
            self._vision_feature_forward_count += 1
            batch = extract_normalized_spatial_merger_embeddings(
                model=self.model,
                pixel_values=device_pixels,
                image_grid_thw=device_grid,
                runtime_identity=self.runtime_identity,
            )
            selection = fresh16_policy_vision_similarity_from_batch(
                batch,
                event_step_ids=candidates,
                event_image_indices={
                    step_id: index for index, step_id in enumerate(candidates)
                },
                current_image_index=image_count - 1,
            )
            norm_min, norm_max = _variable_normalized_norm_range(
                batch.normalized_embeddings,
                torch=self.torch,
                image_count=image_count,
            )
            repeat_results.append(
                {
                    "repeat_index": repeat_index,
                    "scores_by_event_step": {
                        str(step_id): float(score)
                        for step_id, score in selection.scores_by_event_step
                    },
                    "ranked_event_step_ids": list(
                        selection.ranked_event_step_ids
                    ),
                    "selected_event_step_ids": list(
                        selection.selected_event_step_ids
                    ),
                    "normalized_embedding_norm_range": {
                        "minimum": norm_min,
                        "maximum": norm_max,
                        "maximum_absolute_deviation_from_one": max(
                            abs(norm_min - 1.0),
                            abs(norm_max - 1.0),
                        ),
                    },
                }
            )
        self._assert_forbidden_operations_zero()
        canonical = repeat_results[0]
        replay = repeat_results[-1]
        score_differences = [
            abs(
                canonical["scores_by_event_step"][str(step_id)]
                - replay["scores_by_event_step"][str(step_id)]
            )
            for step_id in candidates
        ]
        per_call_counts = {
            "image_processor_batch_count": 1,
            "policy_vision_feature_forward_count": feature_repeats,
            **{
                key: 0
                for key in GUI_OWL_V2_2_VISION_FORBIDDEN_OPERATION_KEYS
            },
        }
        result = {
            "scores_by_event_step": canonical["scores_by_event_step"],
            "ranked_event_step_ids": canonical["ranked_event_step_ids"],
            "selected_event_step_ids": canonical["selected_event_step_ids"],
            "image_grid_thw": [
                list(row) for row in cpu_geometry["image_grid_thw"]
            ],
            "raw_patch_counts": list(cpu_geometry["raw_patch_counts"]),
            "merged_token_counts": list(cpu_geometry["merged_token_counts"]),
            "normalized_embedding_norm_range": canonical[
                "normalized_embedding_norm_range"
            ],
            "feature_repeats": feature_repeats,
            "repeat_results": repeat_results,
            "same_device_replay": {
                "performed": feature_repeats == 2,
                "ranking_equal": (
                    canonical["ranked_event_step_ids"]
                    == replay["ranked_event_step_ids"]
                ),
                "selection_equal": (
                    canonical["selected_event_step_ids"]
                    == replay["selected_event_step_ids"]
                ),
                "max_abs_score_difference": max(score_differences),
            },
            "operation_counts": per_call_counts,
            "runtime_metadata": self.metadata,
        }
        return _json_copy(result)
