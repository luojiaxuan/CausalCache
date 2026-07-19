"""Deterministic GUI-Owl runtime for variable-history restoration labels."""

from __future__ import annotations

from typing import Any

from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v3 import (
    GUIOwlV21StrictDeterminismActionStabilityRuntimeV3,
)


VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE = 480
VARIABLE_HISTORY_MAXIMUM_MICROBATCH_SIZE = 16
VARIABLE_HISTORY_RUNTIME_PROFILE_ID = (
    "causalcache_variable_history_480_sdpa_state_repeat_filter_v2"
)


class GUIOwlVariableHistoryRuntime(
    GUIOwlV21StrictDeterminismActionStabilityRuntimeV3
):
    """Reuse one deterministic H200 runtime and filter unstable states."""

    allowed_effective_visual_tokens_per_image = frozenset(
        {VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE}
    )
    maximum_microbatch_size = VARIABLE_HISTORY_MAXIMUM_MICROBATCH_SIZE

    def __init__(self, **kwargs: Any) -> None:
        if (
            kwargs.get("target_effective_visual_tokens_per_image")
            != VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        ):
            raise ValueError("variable-history runtime requires 480 visual tokens per image")
        super().__init__(**kwargs)
        self.metadata = {
            **self.metadata,
            "fresh_process_per_state_required": False,
            "maximum_microbatch_size": self.maximum_microbatch_size,
            "runtime_profile_id": VARIABLE_HISTORY_RUNTIME_PROFILE_ID,
            "state_level_action_repeat_filter": True,
        }


__all__ = [
    "GUIOwlVariableHistoryRuntime",
    "VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE",
    "VARIABLE_HISTORY_MAXIMUM_MICROBATCH_SIZE",
    "VARIABLE_HISTORY_RUNTIME_PROFILE_ID",
]
