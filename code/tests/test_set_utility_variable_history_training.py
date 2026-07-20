from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.set_utility_variable_history_training import (
    VARIABLE_HISTORY_NUMERIC_FEATURE_NAMES,
    normalized_ocr_tokens,
    variable_history_event_numeric_features,
)


def _low() -> LowFidelityEventV2:
    return LowFidelityEventV2(
        step_id=3,
        action_type="click",
        action_argument="coordinate_bin:x1_y2",
        foreground_app="app",
        screen_text_added=("Done",),
        screen_text_removed=("Start",),
        screen_change="medium",
        executor_result="accepted",
    )


def test_variable_history_numeric_features_are_fixed_width_and_bounded() -> None:
    values = variable_history_event_numeric_features(
        _low(),
        decision_step_id=10,
        event_ocr_tokens=("Done", "Button"),
        current_ocr_tokens=("done", "Next"),
        event_current_vlm_cosine=-0.25,
    )
    assert len(values) == len(VARIABLE_HISTORY_NUMERIC_FEATURE_NAMES) == 11
    assert all(0.0 <= value <= 1.0 for value in values)
    assert values[-2] == 1.0 / 3.0
    assert values[-1] == 0.375


def test_ocr_tokens_normalize_case_and_whitespace() -> None:
    assert normalized_ocr_tokens(("  Hello  World ", "HELLO WORLD")) == {
        "hello world"
    }
