"""Memory-safe SDPA numerical-control runtime for diagnostic D1b."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v1 import (
    EXPECTED_AUTO_ATTENTION,
    GUIOwlV21AutoActionStabilityRuntimeV1,
    validate_auto_attention_v1,
)
from causalcache.policy.gui_owl_v2_2_eager_runtime import (
    GUI_OWL_V2_2_EAGER_EXPECTED_GPU_NAME,
    GUI_OWL_V2_2_EAGER_SEED,
    _configure_and_validate_numerical_controls,
    _validate_software_stack,
    audit_gui_owl_v2_2_scientific_environment,
    gui_owl_v2_2_observed_attention,
)


GUI_OWL_V2_1_ACTION_STABILITY_SDPA_NUMERICAL_CONTROL_RUNTIME_V2_ID = (
    "causalcache_action_stability_sdpa_numerical_control_v2"
)
GUI_OWL_V2_1_ACTION_STABILITY_SDPA_NUMERICAL_CONTROL_CLAIM_V2 = (
    "sdpa_fixed_seed_tf32_disabled_numerical_control_"
    "not_strict_cuda_determinism"
)


def _bootstrap_numerical_controls_before_cuda_v2() -> dict[str, Any]:
    """Freeze the recovered controls before the parent runtime touches CUDA."""
    scientific_environment_audit = audit_gui_owl_v2_2_scientific_environment()
    try:
        import torch
        import transformers
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "GUI-Owl D1b runtime requires PyTorch and Transformers"
        ) from error

    # note (luojiaxuan): D1b is launched in one fresh OS process per state.  This
    # guard makes that execution requirement observable instead of silently
    # accepting controls that were applied after a CUDA context already existed.
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "GUI-Owl D1b numerical controls require a fresh pre-CUDA process"
        )
    numerical_controls = _configure_and_validate_numerical_controls(torch)
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "GUI-Owl D1b numerical-control bootstrap unexpectedly initialized CUDA"
        )
    software_stack = _validate_software_stack(torch, transformers)
    return {
        "numerical_controls": numerical_controls,
        "scientific_environment_audit": scientific_environment_audit,
        "software_stack": software_stack,
        "torch": torch,
    }


def validate_sdpa_numerical_control_attention_v2(
    attention: Mapping[str, Any],
) -> dict[str, str]:
    """Fail closed unless top, text, and vision all resolved to SDPA."""
    if not isinstance(attention, Mapping):
        raise RuntimeError("GUI-Owl D1b attention inventory is invalid")
    return validate_auto_attention_v1(attention)


class GUIOwlV21SDPANumericalControlActionStabilityRuntimeV2(
    GUIOwlV21AutoActionStabilityRuntimeV1
):
    """Parent SDPA runtime with the recovered non-strict numerical controls."""

    def __init__(self, **kwargs: Any) -> None:
        bootstrap = _bootstrap_numerical_controls_before_cuda_v2()
        super().__init__(**kwargs)
        if self.torch is not bootstrap["torch"]:
            raise RuntimeError("GUI-Owl D1b torch module identity drifted")
        attention = validate_sdpa_numerical_control_attention_v2(
            gui_owl_v2_2_observed_attention(self.model)
        )
        gpu_name = self.torch.cuda.get_device_name(self.device)
        if gpu_name != GUI_OWL_V2_2_EAGER_EXPECTED_GPU_NAME:
            raise RuntimeError("GUI-Owl D1b requires the recovered H200 GPU class")
        self.metadata = {
            **self.metadata,
            **bootstrap["software_stack"],
            "runtime_profile_id": (
                GUI_OWL_V2_1_ACTION_STABILITY_SDPA_NUMERICAL_CONTROL_RUNTIME_V2_ID
            ),
            "requested_attention_implementation": "auto_parent_default",
            "observed_attention_implementation": attention,
            "memory_efficient_sdpa_required": True,
            "low_cpu_mem_usage": True,
            "seed": GUI_OWL_V2_2_EAGER_SEED,
            "deterministic_algorithms_requested": False,
            **bootstrap["numerical_controls"],
            "strict_cuda_determinism_claimed": False,
            "numerical_control_claim": (
                GUI_OWL_V2_1_ACTION_STABILITY_SDPA_NUMERICAL_CONTROL_CLAIM_V2
            ),
            "numerical_controls_configured_before_cuda_initialization": True,
            "fresh_process_per_state_required": True,
            "scientific_environment_variables_set_by_runtime": False,
            "scientific_environment_audit": bootstrap[
                "scientific_environment_audit"
            ],
            "gpu_name": gpu_name,
        }


__all__ = [
    "EXPECTED_AUTO_ATTENTION",
    "GUI_OWL_V2_1_ACTION_STABILITY_SDPA_NUMERICAL_CONTROL_CLAIM_V2",
    "GUI_OWL_V2_1_ACTION_STABILITY_SDPA_NUMERICAL_CONTROL_RUNTIME_V2_ID",
    "GUIOwlV21SDPANumericalControlActionStabilityRuntimeV2",
    "validate_sdpa_numerical_control_attention_v2",
]
