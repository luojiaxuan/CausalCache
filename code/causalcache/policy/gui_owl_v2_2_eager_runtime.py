"""Frozen eager BF16 runtime for the restoration v2.2 substrate."""

from __future__ import annotations

import os
import platform
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2_1_runtime import (
    GUIOwlV21GenerationParseError,
    GUIOwlV21GenerationResult,
    GUIOwlV21OfficialToolsRuntime,
    validate_gui_owl_v2_1_chat_template,
    validate_gui_owl_v2_1_generation_tokens,
)
from causalcache.policy.gui_owl_v2_runtime import (
    FROZEN_GUI_OWL_V2_DTYPE,
    FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    _runtime_identity_metadata,
)
from causalcache.policy.gui_owl_v2_vision import (
    VISION_PATCH_SIZE,
    VISION_SPATIAL_MERGE_SIZE,
    _validate_model_identity,
    verify_frozen_vision_runtime,
)
GUI_OWL_V2_2_EAGER_RUNTIME_ID = "causalcache_restoration_v2_2_eager_runtime"
GUI_OWL_V2_2_EAGER_SEED = 0
GUI_OWL_V2_2_EAGER_ATTENTION_IMPLEMENTATION = "eager"
GUI_OWL_V2_2_EAGER_NUMERICAL_CONTROL_CLAIM = (
    "eager_fixed_seed_tf32_disabled_numerical_control_"
    "not_strict_cuda_determinism"
)
GUI_OWL_V2_2_EAGER_EXPECTED_SOFTWARE_STACK = {
    "python_version": "3.12.3",
    "torch_version": "2.11.0+cu130",
    "torch_cuda_version": "13.0",
    "cudnn_version": 91900,
    "transformers_version": "5.6.0",
}
GUI_OWL_V2_2_EAGER_EXPECTED_GPU_NAME = "NVIDIA H200"
AUDITED_SCIENTIFIC_ENVIRONMENT_VARIABLES = (
    "CUBLAS_WORKSPACE_CONFIG",
    "NVIDIA_TF32_OVERRIDE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "FLASH_ATTENTION_DETERMINISTIC",
    "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT",
    "TORCH_CUDNN_V8_API_DISABLED",
    "TORCH_CUDNN_V8_API_ENABLED",
)


def audit_gui_owl_v2_2_scientific_environment() -> dict[str, Any]:
    """Require all frozen behavior-changing environment inputs to be absent."""
    present = tuple(
        name for name in AUDITED_SCIENTIFIC_ENVIRONMENT_VARIABLES if name in os.environ
    )
    if present:
        raise RuntimeError(
            "GUI-Owl v2.2 scientific environment variables must be absent: "
            + ", ".join(present)
        )
    return {
        "audited_names": list(AUDITED_SCIENTIFIC_ENVIRONMENT_VARIABLES),
        "present_names": [],
        "all_absent": True,
    }


def gui_owl_v2_2_observed_attention(model: Any) -> dict[str, Any]:
    """Read the resolved top-level, text, and vision attention backends."""
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("GUI-Owl v2.2 model config is unavailable")
    text_config = getattr(config, "text_config", None)
    vision_config = getattr(config, "vision_config", None)
    return {
        "top": getattr(config, "_attn_implementation", None),
        "text": getattr(text_config, "_attn_implementation", None),
        "vision": getattr(vision_config, "_attn_implementation", None),
    }


def validate_gui_owl_v2_2_eager_attention(attention: Mapping[str, Any]) -> None:
    """Fail closed unless every relevant model config resolved to eager."""
    expected_keys = {"top", "text", "vision"}
    if set(attention) != expected_keys:
        raise RuntimeError("GUI-Owl v2.2 attention inventory drifted")
    non_eager = {
        name: attention[name]
        for name in sorted(expected_keys)
        if attention[name] != GUI_OWL_V2_2_EAGER_ATTENTION_IMPLEMENTATION
    }
    if non_eager:
        raise RuntimeError(
            "GUI-Owl v2.2 requires observed eager attention for top/text/vision: "
            + repr(non_eager)
        )


def _configure_and_validate_numerical_controls(torch: Any) -> dict[str, Any]:
    torch.use_deterministic_algorithms(False, warn_only=False)
    torch.manual_seed(GUI_OWL_V2_2_EAGER_SEED)
    torch.cuda.manual_seed_all(GUI_OWL_V2_2_EAGER_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    observed = {
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only_enabled": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    expected = {
        "deterministic_algorithms_enabled": False,
        "deterministic_warn_only_enabled": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    if observed != expected:
        raise RuntimeError(
            "GUI-Owl v2.2 numerical controls did not resolve exactly: "
            + repr(observed)
        )
    return observed


def _validate_software_stack(torch: Any, transformers: Any) -> dict[str, Any]:
    observed = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "transformers_version": transformers.__version__,
    }
    if observed != GUI_OWL_V2_2_EAGER_EXPECTED_SOFTWARE_STACK:
        raise RuntimeError(
            "GUI-Owl v2.2 software stack differs from the recovered spatial audit: "
            + repr(observed)
        )
    return observed


class GUIOwlV22EagerRuntime(GUIOwlV21OfficialToolsRuntime):
    """v2.1 official-tool behavior under the frozen v2.2 eager profile."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        expected_snapshot_manifest: str | Path,
        device: str,
        target_effective_visual_tokens_per_image: int = (
            FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        ),
    ) -> None:
        if re.fullmatch(r"cuda:[0-9]+", device) is None:
            raise ValueError("GUI-Owl v2.2 requires one explicit CUDA device")
        if (
            target_effective_visual_tokens_per_image
            != FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        ):
            raise ValueError(
                "GUI-Owl v2.2 effective visual token target is frozen to 2560"
            )
        scientific_environment_audit = (
            audit_gui_owl_v2_2_scientific_environment()
        )

        try:
            import torch
            import transformers
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "GUI-Owl v2.2 runtime requires PyTorch and Transformers"
            ) from error

        selected_device = torch.device(device)
        if (
            not torch.cuda.is_available()
            or selected_device.index is None
            or selected_device.index >= torch.cuda.device_count()
        ):
            raise RuntimeError("GUI-Owl v2.2 requires the selected CUDA device")
        software_stack = _validate_software_stack(torch, transformers)
        gpu_name = torch.cuda.get_device_name(selected_device)
        if gpu_name != GUI_OWL_V2_2_EAGER_EXPECTED_GPU_NAME:
            raise RuntimeError("GUI-Owl v2.2 requires the recovered H200 GPU class")
        numerical_controls = _configure_and_validate_numerical_controls(torch)

        identity = verify_frozen_vision_runtime(
            model_dir=model_dir,
            expected_snapshot_manifest=expected_snapshot_manifest,
        )
        pixels_per_image = target_effective_visual_tokens_per_image * (
            VISION_PATCH_SIZE * VISION_SPATIAL_MERGE_SIZE
        ) ** 2
        processor = AutoProcessor.from_pretrained(
            identity.model_dir,
            min_pixels=pixels_per_image,
            max_pixels=pixels_per_image,
            local_files_only=True,
        )
        if int(processor.image_processor.merge_size) != VISION_SPATIAL_MERGE_SIZE:
            raise ValueError("GUI-Owl v2.2 processor spatial merge size drifted")
        model = AutoModelForImageTextToText.from_pretrained(
            identity.model_dir,
            dtype=torch.bfloat16,
            attn_implementation=GUI_OWL_V2_2_EAGER_ATTENTION_IMPLEMENTATION,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(selected_device)
        model.eval().requires_grad_(False)
        _validate_model_identity(model, identity)
        for parameter in model.parameters():
            if parameter.device != selected_device:
                raise RuntimeError("GUI-Owl v2.2 model spans multiple devices")
            if parameter.is_floating_point() and parameter.dtype is not torch.bfloat16:
                raise RuntimeError(
                    "GUI-Owl v2.2 floating parameters must remain bfloat16"
                )
            if parameter.requires_grad:
                raise RuntimeError("GUI-Owl v2.2 model parameters must remain frozen")
        observed_attention = gui_owl_v2_2_observed_attention(model)
        validate_gui_owl_v2_2_eager_attention(observed_attention)

        self.torch = torch
        self.device = selected_device
        self.processor = processor
        self.model = model
        self.runtime_identity = identity
        self.chat_template_identity = validate_gui_owl_v2_1_chat_template(
            model_dir=identity.model_dir,
            tokenizer=processor.tokenizer,
        )
        self.generation_tokens = validate_gui_owl_v2_1_generation_tokens(
            processor.tokenizer,
            model.generation_config,
        )
        self.metadata = {
            **_runtime_identity_metadata(identity),
            **self._interface_metadata(),
            "runtime_profile_id": GUI_OWL_V2_2_EAGER_RUNTIME_ID,
            "dtype": FROZEN_GUI_OWL_V2_DTYPE,
            "device": str(selected_device),
            "frozen": True,
            "single_device": True,
            "processor_class": processor.__class__.__name__,
            "model_class": model.__class__.__name__,
            **software_stack,
            "gpu_name": gpu_name,
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
            "deterministic_algorithms_requested": False,
            **numerical_controls,
            "strict_cuda_determinism_claimed": False,
            "numerical_control_claim": (
                GUI_OWL_V2_2_EAGER_NUMERICAL_CONTROL_CLAIM
            ),
            "scientific_environment_variables_set_by_runtime": False,
            "scientific_environment_audit": scientific_environment_audit,
        }

    def _generation_metadata_with_device(
        self, metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        bound = dict(metadata)
        device = str(self.device)
        if "device" in bound and bound["device"] != device:
            raise RuntimeError("GUI-Owl v2.2 generation metadata device drifted")
        bound["device"] = device
        return bound

    def generate_native_action(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> GUIOwlV21GenerationResult:
        """Preserve v2.1 generation while binding its worker CUDA device."""
        try:
            result = super().generate_native_action(messages)
        except GUIOwlV21GenerationParseError as error:
            error.metadata = self._generation_metadata_with_device(error.metadata)
            raise
        return GUIOwlV21GenerationResult(
            output_text=result.output_text,
            parsed_output=result.parsed_output,
            metadata=self._generation_metadata_with_device(result.metadata),
        )
