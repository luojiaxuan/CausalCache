"""Label-blind training features for variable-history token predictors."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from causalcache.low_fidelity_v2 import LowFidelityEventV2, normalize_screen_text


VARIABLE_HISTORY_NUMERIC_FEATURE_NAMES = (
    "age_fraction",
    "step_fraction",
    "reciprocal_age",
    "action_argument_length_fraction",
    "screen_text_added_count_fraction",
    "screen_text_removed_count_fraction",
    "post_ocr_count_fraction",
    "screen_change_score",
    "executor_result_score",
    "current_ocr_jaccard",
    "current_vlm_mean_cosine_rescaled",
)

_SCREEN_CHANGE_SCORE = {
    "none": 0.0,
    "low": 1.0 / 3.0,
    "medium": 2.0 / 3.0,
    "high": 1.0,
}
_EXECUTOR_RESULT_SCORE = {"failed": 0.0, "unknown": 0.5, "accepted": 1.0}


def normalized_ocr_tokens(values: Iterable[str]) -> frozenset[str]:
    result = frozenset(
        normalized.casefold()
        for value in values
        if (normalized := normalize_screen_text(value))
    )
    return result


def variable_history_event_numeric_features(
    low: LowFidelityEventV2,
    *,
    decision_step_id: int,
    event_ocr_tokens: Sequence[str],
    current_ocr_tokens: Sequence[str],
    event_current_vlm_cosine: float,
) -> tuple[float, ...]:
    if not isinstance(low, LowFidelityEventV2):
        raise TypeError("low must be LowFidelityEventV2")
    if type(decision_step_id) is not int or decision_step_id <= low.step_id:
        raise ValueError("decision step must strictly follow the event")
    cosine = float(event_current_vlm_cosine)
    if not math.isfinite(cosine) or not -1.000001 <= cosine <= 1.000001:
        raise ValueError("event/current VLM cosine must be finite and in [-1,1]")
    cosine = min(1.0, max(-1.0, cosine))
    event_ocr = normalized_ocr_tokens(event_ocr_tokens)
    current_ocr = normalized_ocr_tokens(current_ocr_tokens)
    union = event_ocr | current_ocr
    jaccard = len(event_ocr & current_ocr) / len(union) if union else 1.0
    age = decision_step_id - low.step_id
    values = (
        age / decision_step_id,
        low.step_id / decision_step_id,
        1.0 / age,
        min(len(low.action_argument), 64) / 64.0,
        min(len(low.screen_text_added), 32) / 32.0,
        min(len(low.screen_text_removed), 32) / 32.0,
        min(len(event_ocr_tokens), 128) / 128.0,
        _SCREEN_CHANGE_SCORE[low.screen_change],
        _EXECUTOR_RESULT_SCORE[low.executor_result],
        jaccard,
        (cosine + 1.0) / 2.0,
    )
    if len(values) != len(VARIABLE_HISTORY_NUMERIC_FEATURE_NAMES) or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values
    ):
        raise RuntimeError("variable-history numeric feature construction drifted")
    return values


__all__ = [
    "VARIABLE_HISTORY_NUMERIC_FEATURE_NAMES",
    "normalized_ocr_tokens",
    "variable_history_event_numeric_features",
]
