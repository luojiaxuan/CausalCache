"""Strict-determinism SDPA runtime for the focused D2 diagnostic."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


# note (luojiaxuan): cuBLAS reads this variable when its CUDA-side workspace is
# initialized.  Keep this assignment above every CausalCache policy import so
# the fresh D2 process cannot import PyTorch transitively before the setting is
# fixed.  A conflicting operator value is rejected rather than overwritten.
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_existing_cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _existing_cublas_workspace_config not in (None, _CUBLAS_WORKSPACE_CONFIG):
    raise RuntimeError("D2 found a conflicting CUBLAS_WORKSPACE_CONFIG")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG

from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v1 import (  # noqa: E402
    GUIOwlV21AutoActionStabilityRuntimeV1,
    validate_auto_attention_v1,
)
from causalcache.policy.gui_owl_v2_2_eager_runtime import (  # noqa: E402
    AUDITED_SCIENTIFIC_ENVIRONMENT_VARIABLES,
    GUI_OWL_V2_2_EAGER_EXPECTED_GPU_NAME,
    GUI_OWL_V2_2_EAGER_SEED,
    _validate_software_stack,
    gui_owl_v2_2_observed_attention,
)


GUI_OWL_V2_1_ACTION_STABILITY_STRICT_RUNTIME_V3_ID = (
    "causalcache_action_stability_strict_determinism_sdpa_v3"
)
GUI_OWL_V2_1_ACTION_STABILITY_STRICT_CLAIM_V3 = (
    "strict_pytorch_deterministic_algorithms_cublas_workspace_4096_8_"
    "tf32_disabled_sdpa"
)


def _audit_and_configure_strict_controls_before_cuda_v3() -> dict[str, Any]:
    """Configure and verify all D2 controls before CUDA initialization."""
    present = {
        name: os.environ[name]
        for name in AUDITED_SCIENTIFIC_ENVIRONMENT_VARIABLES
        if name in os.environ
    }
    if present != {"CUBLAS_WORKSPACE_CONFIG": _CUBLAS_WORKSPACE_CONFIG}:
        raise RuntimeError("D2 scientific environment differs from the strict profile")

    try:
        import torch
        import transformers
    except ModuleNotFoundError as error:
        raise RuntimeError("GUI-Owl D2 runtime requires PyTorch and Transformers") from error

    if torch.cuda.is_initialized():
        raise RuntimeError("GUI-Owl D2 controls require a fresh pre-CUDA process")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(GUI_OWL_V2_2_EAGER_SEED)
    torch.cuda.manual_seed_all(GUI_OWL_V2_2_EAGER_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_initialized():
        raise RuntimeError("GUI-Owl D2 control bootstrap initialized CUDA")

    observed = {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only_enabled": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    expected = {
        "cublas_workspace_config": _CUBLAS_WORKSPACE_CONFIG,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms_enabled": True,
        "deterministic_warn_only_enabled": False,
        "float32_matmul_precision": "highest",
    }
    if observed != expected:
        raise RuntimeError("GUI-Owl D2 strict controls did not resolve exactly")
    return {
        "numerical_controls": observed,
        "scientific_environment_audit": {
            "audited_names": list(AUDITED_SCIENTIFIC_ENVIRONMENT_VARIABLES),
            "present_names": ["CUBLAS_WORKSPACE_CONFIG"],
            "required_value": _CUBLAS_WORKSPACE_CONFIG,
        },
        "software_stack": _validate_software_stack(torch, transformers),
        "torch": torch,
    }


def validate_strict_sdpa_attention_v3(
    attention: Mapping[str, Any],
) -> dict[str, str]:
    """Fail closed unless top, text, and vision all remain on SDPA."""
    return validate_auto_attention_v1(attention)


class GUIOwlV21StrictDeterminismActionStabilityRuntimeV3(
    GUIOwlV21AutoActionStabilityRuntimeV1
):
    """Parent SDPA runtime under PyTorch's strict deterministic mode."""

    def __init__(self, **kwargs: Any) -> None:
        bootstrap = _audit_and_configure_strict_controls_before_cuda_v3()
        super().__init__(**kwargs)
        if self.torch is not bootstrap["torch"]:
            raise RuntimeError("GUI-Owl D2 torch module identity drifted")
        attention = validate_strict_sdpa_attention_v3(
            gui_owl_v2_2_observed_attention(self.model)
        )
        gpu_name = self.torch.cuda.get_device_name(self.device)
        if gpu_name != GUI_OWL_V2_2_EAGER_EXPECTED_GPU_NAME:
            raise RuntimeError("GUI-Owl D2 requires the frozen H200 GPU class")
        self.metadata = {
            **self.metadata,
            **bootstrap["software_stack"],
            **bootstrap["numerical_controls"],
            "fresh_process_per_state_required": True,
            "gpu_name": gpu_name,
            "memory_efficient_sdpa_required": True,
            "numerical_control_claim": (
                GUI_OWL_V2_1_ACTION_STABILITY_STRICT_CLAIM_V3
            ),
            "numerical_controls_configured_before_cuda_initialization": True,
            "observed_attention_implementation": attention,
            "requested_attention_implementation": "auto_parent_default",
            "runtime_profile_id": (
                GUI_OWL_V2_1_ACTION_STABILITY_STRICT_RUNTIME_V3_ID
            ),
            "scientific_environment_audit": bootstrap[
                "scientific_environment_audit"
            ],
            "scientific_environment_variables_set_by_runtime": True,
            "seed": GUI_OWL_V2_2_EAGER_SEED,
            "strict_cuda_determinism_claimed": True,
        }


__all__ = [
    "GUI_OWL_V2_1_ACTION_STABILITY_STRICT_CLAIM_V3",
    "GUI_OWL_V2_1_ACTION_STABILITY_STRICT_RUNTIME_V3_ID",
    "GUIOwlV21StrictDeterminismActionStabilityRuntimeV3",
    "validate_strict_sdpa_attention_v3",
]
