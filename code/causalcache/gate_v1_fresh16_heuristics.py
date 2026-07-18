"""Label-blind variable-history heuristic selectors for fresh-16 gate evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from functools import cmp_to_key
from numbers import Real
from typing import Any

from causalcache.restoration_v2_baselines import (
    OCR_JACCARD_WEIGHT,
    RGB_HISTOGRAM_WEIGHT,
    SCORE_TIE_ABS_TOL,
    SCORE_TIE_REL_TOL,
    BaselineSelection,
    joint_rgb_histogram_cosine,
    normalized_ocr_token_set,
)


FRESH16_NATURAL_CANDIDATE_PREFIXES = (
    (1, 2),
    (1, 2, 3),
    (1, 2, 3, 4),
)
FRESH16_BUDGET_EVENT_CAPACITY = 2


def validate_natural_candidate_event_step_ids(
    event_step_ids: Iterable[int],
) -> tuple[int, ...]:
    """Require one of the preregistered natural n=2/3/4 history prefixes."""
    if isinstance(event_step_ids, (str, bytes, bytearray, Mapping)):
        raise TypeError("candidate event step ids must be an ordered iterable")
    try:
        result = tuple(event_step_ids)
    except TypeError as error:
        raise TypeError("candidate event step ids must be an ordered iterable") from error
    if any(type(step_id) is not int for step_id in result):
        raise TypeError("candidate event step ids must be integers")
    if result not in FRESH16_NATURAL_CANDIDATE_PREFIXES:
        raise ValueError(
            "candidate event step ids must be one of the natural prefixes "
            f"{FRESH16_NATURAL_CANDIDATE_PREFIXES}; got {result}"
        )
    return result


def _validate_budget(budget_event_capacity: int) -> int:
    if type(budget_event_capacity) is not int:
        raise TypeError("budget event capacity must be an integer")
    if budget_event_capacity != FRESH16_BUDGET_EVENT_CAPACITY:
        raise ValueError(
            "fresh-16 heuristic budget event capacity must equal "
            f"{FRESH16_BUDGET_EVENT_CAPACITY}"
        )
    return budget_event_capacity


def _candidate_mapping(
    value: Any,
    *,
    event_step_ids: tuple[int, ...],
    label: str,
) -> dict[int, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(type(step_id) is not int for step_id in value):
        raise TypeError(f"{label} keys must be integer event step ids")
    observed = set(value)
    expected = set(event_step_ids)
    if observed != expected:
        raise ValueError(
            f"{label} must cover exactly events {event_step_ids}; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return {step_id: value[step_id] for step_id in event_step_ids}


def _finite_score(value: Any, *, event_step_id: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"score for event {event_step_id} must be a real number")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"score for event {event_step_id} must be finite")
    return score


def recent_selection(
    event_step_ids: Iterable[int],
    *,
    budget_event_capacity: int = FRESH16_BUDGET_EVENT_CAPACITY,
) -> tuple[int, ...]:
    """Select the last ``min(B, n)`` events in chronological order."""
    candidates = validate_natural_candidate_event_step_ids(event_step_ids)
    budget = _validate_budget(budget_event_capacity)
    selected_count = min(budget, len(candidates))
    return candidates[-selected_count:]


def select_similarity_top_b(
    scores_by_event_step: Mapping[int, Real],
    *,
    event_step_ids: Iterable[int],
    budget_event_capacity: int = FRESH16_BUDGET_EVENT_CAPACITY,
) -> BaselineSelection:
    """Rank every candidate by the frozen score/tie rule and force-fill top-B."""
    candidates = validate_natural_candidate_event_step_ids(event_step_ids)
    budget = _validate_budget(budget_event_capacity)
    raw_scores = _candidate_mapping(
        scores_by_event_step,
        event_step_ids=candidates,
        label="scores_by_event_step",
    )
    scores = {
        step_id: _finite_score(raw_scores[step_id], event_step_id=step_id)
        for step_id in candidates
    }

    def compare(left: int, right: int) -> int:
        left_score = scores[left]
        right_score = scores[right]
        if math.isclose(
            left_score,
            right_score,
            rel_tol=SCORE_TIE_REL_TOL,
            abs_tol=SCORE_TIE_ABS_TOL,
        ):
            return -1 if left < right else 1
        return -1 if left_score > right_score else 1

    ranking = tuple(sorted(candidates, key=cmp_to_key(compare)))
    selected_count = min(budget, len(candidates))
    selected = tuple(sorted(ranking[:selected_count]))
    return BaselineSelection(
        scores_by_event_step=tuple((step_id, scores[step_id]) for step_id in candidates),
        ranked_event_step_ids=ranking,
        selected_event_step_ids=selected,
    )


def ocr_rgb_similarity_selection(
    *,
    event_step_ids: Iterable[int],
    event_ocr_tokens: Mapping[int, Iterable[str]],
    current_ocr_tokens: Iterable[str],
    event_resized_rgb_bytes: Mapping[int, bytes | bytearray | memoryview],
    current_resized_rgb_bytes: bytes | bytearray | memoryview,
    budget_event_capacity: int = FRESH16_BUDGET_EVENT_CAPACITY,
) -> BaselineSelection:
    """Apply the frozen equal-weight OCR-Jaccard/RGB-cosine similarity."""
    candidates = validate_natural_candidate_event_step_ids(event_step_ids)
    _validate_budget(budget_event_capacity)
    ocr_by_event = _candidate_mapping(
        event_ocr_tokens,
        event_step_ids=candidates,
        label="event_ocr_tokens",
    )
    rgb_by_event = _candidate_mapping(
        event_resized_rgb_bytes,
        event_step_ids=candidates,
        label="event_resized_rgb_bytes",
    )
    current_tokens = normalized_ocr_token_set(current_ocr_tokens)
    scores: dict[int, float] = {}
    for step_id in candidates:
        post_tokens = normalized_ocr_token_set(ocr_by_event[step_id])
        if not post_tokens and not current_tokens:
            text_score = 1.0
        else:
            text_score = len(post_tokens & current_tokens) / len(
                post_tokens | current_tokens
            )
        rgb_score = joint_rgb_histogram_cosine(
            rgb_by_event[step_id],
            current_resized_rgb_bytes,
        )
        scores[step_id] = (
            OCR_JACCARD_WEIGHT * text_score + RGB_HISTOGRAM_WEIGHT * rgb_score
        )
    return select_similarity_top_b(
        scores,
        event_step_ids=candidates,
        budget_event_capacity=budget_event_capacity,
    )


def policy_vision_similarity_selection(
    scores_by_event_step: Mapping[int, Real],
    *,
    event_step_ids: Iterable[int],
    budget_event_capacity: int = FRESH16_BUDGET_EVENT_CAPACITY,
) -> BaselineSelection:
    """Select from frozen-policy visual cosine scores without recomputing features."""
    return select_similarity_top_b(
        scores_by_event_step,
        event_step_ids=event_step_ids,
        budget_event_capacity=budget_event_capacity,
    )
