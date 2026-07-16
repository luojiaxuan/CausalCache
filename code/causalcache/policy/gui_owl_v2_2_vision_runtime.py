"""Feature-only GUI-Owl vision runtime for the restoration v2.2 baseline."""

from __future__ import annotations

import importlib.metadata
import json
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2_runtime import (
    FROZEN_GUI_OWL_V2_DTYPE,
    FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    _runtime_identity_metadata,
)
from causalcache.policy.gui_owl_v2_2_eager_runtime import (
    GUI_OWL_V2_2_EAGER_ATTENTION_IMPLEMENTATION,
    GUI_OWL_V2_2_EAGER_EXPECTED_GPU_NAME,
    GUI_OWL_V2_2_EAGER_NUMERICAL_CONTROL_CLAIM,
    GUI_OWL_V2_2_EAGER_SEED,
    _configure_and_validate_numerical_controls,
    _validate_software_stack,
    audit_gui_owl_v2_2_scientific_environment,
    gui_owl_v2_2_observed_attention,
    validate_gui_owl_v2_2_eager_attention,
)
from causalcache.policy.gui_owl_v2_vision import (
    VISION_PATCH_SIZE,
    VISION_SPATIAL_MERGE_SIZE,
    VISION_TEMPORAL_PATCH_SIZE,
    _validate_model_identity,
    extract_normalized_spatial_merger_embeddings,
    frozen_policy_vision_similarity_from_batch,
    verify_frozen_vision_runtime,
    visual_token_geometry,
)
from causalcache.restoration_v2_baselines import CANDIDATE_EVENT_STEP_IDS


GUI_OWL_V2_2_VISION_RUNTIME_ID = (
    "causalcache_restoration_v2_2_policy_vision_feature_only_runtime"
)
GUI_OWL_V2_2_VISION_RUNTIME_UUID_TYPE_ONLY_V2_ID = (
    "causalcache_restoration_v2_2_policy_vision_feature_only_runtime_"
    "uuid_type_only_v2"
)
GPU_UUID_TYPE_PROFILE_V1 = "cuda_device_property_uuid_str_bytes_v1"
GPU_UUID_TYPE_PROFILE_V2 = "cuda_device_property_uuid_torch_c_cuuuid_v2"
GPU_UUID_TYPE_PROFILES = frozenset(
    {GPU_UUID_TYPE_PROFILE_V1, GPU_UUID_TYPE_PROFILE_V2}
)
GUI_OWL_V2_2_VISION_IMAGE_COUNT = 5
GUI_OWL_V2_2_VISION_MAX_FEATURE_REPEATS = 2
GUI_OWL_V2_2_VISION_IMAGE_PROCESSOR_CLASS = "Qwen2VLImageProcessor"
GUI_OWL_V2_2_VISION_PILLOW_VERSION = "12.2.0"
GUI_OWL_V2_2_VISION_PROCESSOR_KEYS = frozenset(
    {"pixel_values", "image_grid_thw"}
)
GUI_OWL_V2_2_VISION_FORBIDDEN_OPERATION_KEYS = (
    "top_model_forward_count",
    "language_model_forward_count",
    "lm_head_forward_count",
    "generation_count",
)


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reject non-finite or non-JSON runtime metadata and return a detached copy."""
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("GUI-Owl vision result must be a JSON object")
    return decoded


def _runtime_profile_id(gpu_uuid_type_profile: str) -> str:
    if gpu_uuid_type_profile == GPU_UUID_TYPE_PROFILE_V1:
        return GUI_OWL_V2_2_VISION_RUNTIME_ID
    if gpu_uuid_type_profile == GPU_UUID_TYPE_PROFILE_V2:
        return GUI_OWL_V2_2_VISION_RUNTIME_UUID_TYPE_ONLY_V2_ID
    raise ValueError("GPU UUID type profile is not frozen")


def _canonical_gpu_uuid(
    value: Any,
    *,
    gpu_uuid_type_profile: str = GPU_UUID_TYPE_PROFILE_V1,
    loaded_torch_c_cuuuid_type: type[Any] | None = None,
) -> str:
    _runtime_profile_id(gpu_uuid_type_profile)
    if gpu_uuid_type_profile == GPU_UUID_TYPE_PROFILE_V1:
        if isinstance(value, bytes):
            try:
                value = value.decode("ascii")
            except UnicodeDecodeError as error:
                raise ValueError("GPU UUID bytes must be ASCII") from error
        if not isinstance(value, str):
            raise ValueError("v1 GPU UUID must be str or bytes")
    else:
        if not isinstance(loaded_torch_c_cuuuid_type, type):
            raise ValueError("v2 GPU UUID requires the loaded torch._C._CUuuid type")
        value_type = type(value)
        if value_type is not loaded_torch_c_cuuuid_type:
            raise ValueError(
                "v2 GPU UUID must have the exact loaded torch._C._CUuuid type"
            )
        if (
            value_type.__module__ != "torch._C"
            or value_type.__name__ != "_CUuuid"
        ):
            raise ValueError("v2 GPU UUID must have the exact torch._C._CUuuid type")
        try:
            value = str(value)
        except BaseException as error:
            raise ValueError("torch GPU UUID could not be converted to string") from error
    if not value.strip():
        raise ValueError("GPU UUID must be a non-empty string")
    normalized = value.strip()
    if normalized.lower().startswith("gpu-"):
        normalized = normalized[4:]
    if re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]+)+", normalized) is None:
        raise ValueError("GPU UUID format drifted")
    return "GPU-" + normalized.lower()


def _nvidia_smi_gpu_identity(expected_gpu_uuid: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    matches = []
    for raw_line in completed.stdout.splitlines():
        fields = tuple(field.strip() for field in raw_line.split(","))
        if len(fields) != 3:
            raise RuntimeError("nvidia-smi GPU identity schema drifted")
        raw_index, raw_uuid, raw_pci_bus_id = fields
        if _canonical_gpu_uuid(raw_uuid) != expected_gpu_uuid:
            continue
        if re.fullmatch(r"[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]", raw_pci_bus_id) is None:
            raise RuntimeError("nvidia-smi PCI bus id format drifted")
        try:
            index = int(raw_index)
        except ValueError as error:
            raise RuntimeError("nvidia-smi GPU index is not an integer") from error
        if index < 0:
            raise RuntimeError("nvidia-smi GPU index must be non-negative")
        matches.append(
            {
                "gpu_uuid": expected_gpu_uuid,
                "gpu_pci_bus_id": raw_pci_bus_id.lower(),
                "nvidia_smi_index": index,
            }
        )
    if len(matches) != 1:
        raise RuntimeError("expected GPU UUID must match exactly one nvidia-smi row")
    return matches[0]


def _validated_gpu_identity(
    *,
    torch: Any,
    device: Any,
    expected_gpu_uuid: str,
    gpu_uuid_type_profile: str,
) -> dict[str, Any]:
    expected = _canonical_gpu_uuid(expected_gpu_uuid)
    properties = torch.cuda.get_device_properties(device)
    observed_value = getattr(properties, "uuid", None)
    exact_uuid_type = None
    if gpu_uuid_type_profile == GPU_UUID_TYPE_PROFILE_V2:
        torch_c = getattr(torch, "_C", None)
        exact_uuid_type = getattr(torch_c, "_CUuuid", None)
        if not isinstance(exact_uuid_type, type):
            raise RuntimeError("torch._C._CUuuid type is unavailable")
        if type(observed_value) is not exact_uuid_type:
            raise ValueError(
                "v2 GPU UUID must have the exact loaded torch._C._CUuuid type"
            )
    observed = _canonical_gpu_uuid(
        observed_value,
        gpu_uuid_type_profile=gpu_uuid_type_profile,
        loaded_torch_c_cuuuid_type=exact_uuid_type,
    )
    if observed != expected:
        raise RuntimeError("selected CUDA device UUID differs from expected_gpu_uuid")
    identity = _nvidia_smi_gpu_identity(expected)
    identity["logical_device_index"] = int(device.index)
    return identity


def _require_exact_five_rgb_pil_images(images: Any) -> tuple[Any, ...]:
    if isinstance(images, (str, bytes, bytearray, Mapping)):
        raise TypeError("policy-vision input must be a sequence of PIL images")
    try:
        materialized = tuple(images)
    except TypeError as error:
        raise TypeError("policy-vision input must be a sequence of PIL images") from error
    if len(materialized) != GUI_OWL_V2_2_VISION_IMAGE_COUNT:
        raise ValueError("policy-vision input must contain exactly five images")
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


def _validate_cpu_image_processor_output(
    encoded: Any,
    *,
    torch: Any,
) -> tuple[Any, Any, dict[str, Any]]:
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
        GUI_OWL_V2_2_VISION_IMAGE_COUNT,
        3,
    ):
        raise ValueError("CPU image_grid_thw must have shape (5, 3)")
    geometry = visual_token_geometry(image_grid_thw)
    raw_patch_counts = geometry["raw_patch_counts"]
    if len(raw_patch_counts) != GUI_OWL_V2_2_VISION_IMAGE_COUNT:
        raise ValueError("image grid must describe exactly five images")
    packed_patch_width = (
        3 * VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE
    )
    if tuple(pixel_values.shape) != (sum(raw_patch_counts), packed_patch_width):
        raise ValueError("CPU pixel_values shape differs from frozen visual geometry")
    return pixel_values, image_grid_thw, geometry


def _move_exact_inputs_to_device(
    *,
    pixel_values: Any,
    image_grid_thw: Any,
    device: Any,
    torch: Any,
) -> tuple[Any, Any]:
    device_pixels = pixel_values.to(device=device, non_blocking=False)
    device_grid = image_grid_thw.to(device=device, non_blocking=False)
    for name, value, expected_dtype in (
        ("pixel_values", device_pixels, torch.float32),
        ("image_grid_thw", device_grid, torch.int64),
    ):
        if value.device != device:
            raise RuntimeError(f"{name} did not move to the selected CUDA device")
        if value.dtype is not expected_dtype:
            raise RuntimeError(f"{name} dtype changed during CUDA transfer")
        if not value.is_contiguous():
            raise RuntimeError(f"{name} became non-contiguous during CUDA transfer")
    return device_pixels, device_grid


def _normalized_norm_range(embeddings: Any, *, torch: Any) -> tuple[float, float]:
    with torch.inference_mode():
        norms = torch.linalg.vector_norm(embeddings, dim=1)
        values = norms.detach().to(device="cpu").tolist()
    if len(values) != GUI_OWL_V2_2_VISION_IMAGE_COUNT:
        raise ValueError("normalized vision embedding count drifted")
    finite = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in finite):
        raise ValueError("normalized vision embedding norms are non-finite")
    return min(finite), max(finite)


class GUIOwlV22VisionFeatureRuntime:
    """Pinned image-only runtime that cannot enter text-policy execution paths."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        expected_snapshot_manifest: str | Path,
        device: str,
        expected_gpu_uuid: str,
        gpu_uuid_type_profile: str = GPU_UUID_TYPE_PROFILE_V1,
        target_effective_visual_tokens_per_image: int = (
            FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        ),
    ) -> None:
        runtime_profile_id = _runtime_profile_id(gpu_uuid_type_profile)
        if re.fullmatch(r"cuda:[0-9]+", device) is None:
            raise ValueError("policy-vision runtime requires one explicit CUDA device")
        if (
            target_effective_visual_tokens_per_image
            != FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        ):
            raise ValueError("policy-vision effective visual token target is frozen to 2560")
        scientific_environment_audit = audit_gui_owl_v2_2_scientific_environment()
        try:
            import torch
            import transformers
            from transformers import AutoImageProcessor, AutoModelForImageTextToText
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "policy-vision runtime requires PyTorch and Transformers"
            ) from error

        selected_device = torch.device(device)
        if (
            not torch.cuda.is_available()
            or selected_device.index is None
            or selected_device.index >= torch.cuda.device_count()
        ):
            raise RuntimeError("policy-vision runtime requires the selected CUDA device")
        software_stack = _validate_software_stack(torch, transformers)
        pillow_version = importlib.metadata.version("Pillow")
        if pillow_version != GUI_OWL_V2_2_VISION_PILLOW_VERSION:
            raise RuntimeError("policy-vision Pillow version differs from the pinned stack")
        gpu_name = torch.cuda.get_device_name(selected_device)
        if gpu_name != GUI_OWL_V2_2_EAGER_EXPECTED_GPU_NAME:
            raise RuntimeError("policy-vision runtime requires the recovered H200 GPU class")
        gpu_identity = _validated_gpu_identity(
            torch=torch,
            device=selected_device,
            expected_gpu_uuid=expected_gpu_uuid,
            gpu_uuid_type_profile=gpu_uuid_type_profile,
        )
        numerical_controls = _configure_and_validate_numerical_controls(torch)

        identity = verify_frozen_vision_runtime(
            model_dir=model_dir,
            expected_snapshot_manifest=expected_snapshot_manifest,
        )
        pixels_per_image = target_effective_visual_tokens_per_image * (
            VISION_PATCH_SIZE * VISION_SPATIAL_MERGE_SIZE
        ) ** 2
        image_processor = AutoImageProcessor.from_pretrained(
            identity.model_dir,
            min_pixels=pixels_per_image,
            max_pixels=pixels_per_image,
            local_files_only=True,
        )
        if image_processor.__class__.__name__ != GUI_OWL_V2_2_VISION_IMAGE_PROCESSOR_CLASS:
            raise RuntimeError("policy-vision image processor class drifted")
        if hasattr(image_processor, "tokenizer") or hasattr(
            image_processor, "apply_chat_template"
        ):
            raise RuntimeError("policy-vision runtime forbids tokenizer/chat interfaces")
        processor_size = getattr(image_processor, "size", None)
        if not isinstance(processor_size, Mapping) or dict(processor_size) != {
            "shortest_edge": pixels_per_image,
            "longest_edge": pixels_per_image,
        }:
            raise ValueError("policy-vision image processor size drifted")
        if int(image_processor.merge_size) != VISION_SPATIAL_MERGE_SIZE:
            raise ValueError("policy-vision image processor merge size drifted")
        model = AutoModelForImageTextToText.from_pretrained(
            identity.model_dir,
            dtype=torch.bfloat16,
            attn_implementation=GUI_OWL_V2_2_EAGER_ATTENTION_IMPLEMENTATION,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(selected_device)
        model.eval().requires_grad_(False)
        _validate_model_identity(model, identity)
        if not callable(getattr(model, "get_image_features", None)):
            raise RuntimeError("policy-vision model lacks get_image_features")
        if bool(getattr(model, "training", True)):
            raise RuntimeError("policy-vision model must remain in eval mode")
        for parameter in model.parameters():
            if parameter.device != selected_device:
                raise RuntimeError("policy-vision model spans multiple devices")
            if parameter.is_floating_point() and parameter.dtype is not torch.bfloat16:
                raise RuntimeError("policy-vision floating parameters must remain bfloat16")
            if parameter.requires_grad:
                raise RuntimeError("policy-vision model parameters must remain frozen")
        observed_attention = gui_owl_v2_2_observed_attention(model)
        validate_gui_owl_v2_2_eager_attention(observed_attention)

        self.torch = torch
        self.device = selected_device
        self.image_processor = image_processor
        self.model = model
        self.runtime_identity = identity
        self._forbidden_operation_counts = {
            key: 0 for key in GUI_OWL_V2_2_VISION_FORBIDDEN_OPERATION_KEYS
        }
        self._vision_feature_forward_count = 0
        self._image_processor_batch_count = 0
        self._guard_handles = self._install_forbidden_operation_guards()
        identity_metadata = _runtime_identity_metadata(identity)
        identity_metadata.pop("model_dir", None)
        self.metadata = _json_copy(
            {
                **identity_metadata,
                "runtime_profile_id": runtime_profile_id,
                "dtype": FROZEN_GUI_OWL_V2_DTYPE,
                "device": str(selected_device),
                "frozen": True,
                "single_device": True,
                "feature_only": True,
                "image_processor_class": image_processor.__class__.__name__,
                "model_class": model.__class__.__name__,
                **software_stack,
                "pillow_version": pillow_version,
                "gpu_name": gpu_name,
                **gpu_identity,
                "target_effective_visual_tokens_per_image": (
                    target_effective_visual_tokens_per_image
                ),
                "min_pixels": pixels_per_image,
                "max_pixels": pixels_per_image,
                "requested_attention_implementation": (
                    GUI_OWL_V2_2_EAGER_ATTENTION_IMPLEMENTATION
                ),
                "observed_attention_implementation": observed_attention,
                "seed": GUI_OWL_V2_2_EAGER_SEED,
                **numerical_controls,
                "strict_cuda_determinism_claimed": False,
                "numerical_control_claim": GUI_OWL_V2_2_EAGER_NUMERICAL_CONTROL_CLAIM,
                "scientific_environment_variables_set_by_runtime": False,
                "scientific_environment_audit": scientific_environment_audit,
                "processor_interface": "AutoImageProcessor.__call__",
                "processor_output_keys": sorted(
                    GUI_OWL_V2_2_VISION_PROCESSOR_KEYS
                ),
                "processor_size": dict(processor_size),
                "tokenizer_loaded": False,
                "chat_template_called": False,
                "forbidden_operation_guards_installed": list(
                    GUI_OWL_V2_2_VISION_FORBIDDEN_OPERATION_KEYS
                ),
            }
        )

    def _forbidden_hook(self, key: str) -> Any:
        def fail(*_: Any, **__: Any) -> None:
            self._forbidden_operation_counts[key] += 1
            raise RuntimeError(f"policy-vision runtime forbids {key}")

        return fail

    def _install_forbidden_operation_guards(self) -> tuple[Any, ...]:
        core_model = getattr(self.model, "model", None)
        language_model = getattr(core_model, "language_model", None)
        lm_head = getattr(self.model, "lm_head", None)
        guarded_modules = (
            ("top_model_forward_count", self.model),
            ("language_model_forward_count", language_model),
            ("lm_head_forward_count", lm_head),
        )
        handles = []
        for key, module in guarded_modules:
            register = getattr(module, "register_forward_pre_hook", None)
            if not callable(register) or not callable(getattr(module, "forward", None)):
                raise RuntimeError(f"policy-vision cannot guard {key}")
            guard = self._forbidden_hook(key)
            handles.append(register(guard))
            module.forward = guard
        if not callable(getattr(self.model, "generate", None)):
            raise RuntimeError("policy-vision cannot guard generation")
        # note (luojiaxuan): The full conditional-generation model is loaded only
        # to reuse its pinned get_image_features implementation. Poisoning generate
        # and the three forward entry points makes any policy-path drift fail before
        # producing a baseline result while leaving the vision submodule callable.
        self.model.generate = self._forbidden_hook("generation_count")
        return tuple(handles)

    def _assert_forbidden_operations_zero(self) -> None:
        nonzero = {
            key: value
            for key, value in self._forbidden_operation_counts.items()
            if value != 0
        }
        if nonzero:
            raise RuntimeError(
                "policy-vision runtime observed a forbidden operation: " + repr(nonzero)
            )

    @property
    def operation_counts(self) -> dict[str, int]:
        return {
            "image_processor_batch_count": self._image_processor_batch_count,
            "policy_vision_feature_forward_count": self._vision_feature_forward_count,
            **self._forbidden_operation_counts,
        }

    def score_five_images(
        self,
        images: Sequence[Any],
        *,
        feature_repeats: int = 1,
    ) -> dict[str, Any]:
        """Score four event-post images against one current image without text."""
        if type(feature_repeats) is not int or not (
            1 <= feature_repeats <= GUI_OWL_V2_2_VISION_MAX_FEATURE_REPEATS
        ):
            raise ValueError("feature_repeats must be one or two")
        self._assert_forbidden_operations_zero()
        validated_images = _require_exact_five_rgb_pil_images(images)
        self._image_processor_batch_count += 1
        encoded = self.image_processor(
            images=list(validated_images),
            return_tensors="pt",
        )
        pixel_values, image_grid_thw, cpu_geometry = (
            _validate_cpu_image_processor_output(encoded, torch=self.torch)
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
            selection = frozen_policy_vision_similarity_from_batch(
                batch,
                event_image_indices={
                    step_id: step_id - 1 for step_id in CANDIDATE_EVENT_STEP_IDS
                },
                current_image_index=GUI_OWL_V2_2_VISION_IMAGE_COUNT - 1,
            )
            norm_min, norm_max = _normalized_norm_range(
                batch.normalized_embeddings,
                torch=self.torch,
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
            for step_id in CANDIDATE_EVENT_STEP_IDS
        ]
        per_call_counts = {
            "image_processor_batch_count": 1,
            "policy_vision_feature_forward_count": feature_repeats,
            **{key: 0 for key in GUI_OWL_V2_2_VISION_FORBIDDEN_OPERATION_KEYS},
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
