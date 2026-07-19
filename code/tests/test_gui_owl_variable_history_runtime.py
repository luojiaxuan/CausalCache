from causalcache.policy.gui_owl_v2_runtime import (
    FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    FROZEN_GUI_OWL_V2_MAX_MICROBATCH_SIZE,
    GUIOwlV2Runtime,
)
from causalcache.policy.gui_owl_variable_history_runtime import (
    GUIOwlVariableHistoryRuntime,
    VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    VARIABLE_HISTORY_MAXIMUM_MICROBATCH_SIZE,
)


def test_variable_history_profile_does_not_mutate_historical_runtime_limits() -> None:
    assert GUIOwlV2Runtime.allowed_effective_visual_tokens_per_image == {
        FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
    }
    assert GUIOwlV2Runtime.maximum_microbatch_size == FROZEN_GUI_OWL_V2_MAX_MICROBATCH_SIZE
    assert GUIOwlVariableHistoryRuntime.allowed_effective_visual_tokens_per_image == {
        VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
    }
    assert (
        GUIOwlVariableHistoryRuntime.maximum_microbatch_size
        == VARIABLE_HISTORY_MAXIMUM_MICROBATCH_SIZE
        == 16
    )
    assert VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE == 480
