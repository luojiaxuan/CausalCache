"""Deterministic non-oracle baselines frozen by restoration-v2."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cmp_to_key
from itertools import combinations
from numbers import Real
from typing import Any

from causalcache.low_fidelity_v2 import normalize_screen_text


CANDIDATE_EVENT_STEP_IDS = (1, 2, 3, 4)
PRIMARY_BUDGET_EVENT_CAPACITY = 2
RECENT_EVENT_STEP_IDS = (3, 4)
RGB_HISTOGRAM_BINS_PER_CHANNEL = 16
RESIZED_RGB_WIDTH = 256
RESIZED_RGB_HEIGHT = 256
RESIZED_RGB_BYTE_COUNT = RESIZED_RGB_WIDTH * RESIZED_RGB_HEIGHT * 3
OCR_JACCARD_WEIGHT = 0.5
RGB_HISTOGRAM_WEIGHT = 0.5
SCORE_TIE_REL_TOL = 1e-9
SCORE_TIE_ABS_TOL = 1e-12


@dataclass(frozen=True)
class BaselineSelection:
    """Scores, deterministic ranking, and chronological top-two selection."""

    scores_by_event_step: tuple[tuple[int, float], ...]
    ranked_event_step_ids: tuple[int, ...]
    selected_event_step_ids: tuple[int, ...]

    def score(self, event_step_id: int) -> float:
        for candidate_step_id, value in self.scores_by_event_step:
            if candidate_step_id == event_step_id:
                return value
        raise KeyError(event_step_id)


@dataclass(frozen=True)
class UniformRandomExactExpectation:
    """Analytic random baseline over all six two-of-four coalitions."""

    subsets: tuple[tuple[int, int], ...]
    normalized_recovery_by_subset: tuple[tuple[tuple[int, int], float], ...]
    mean_normalized_recovery: float


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_candidate_event_step_ids(event_step_ids: Iterable[int]) -> tuple[int, ...]:
    if isinstance(event_step_ids, (str, bytes, bytearray, Mapping)):
        raise TypeError("candidate event step ids must be an ordered iterable")
    try:
        result = tuple(event_step_ids)
    except TypeError as error:
        raise TypeError("candidate event step ids must be an ordered iterable") from error
    if any(type(step_id) is not int for step_id in result):
        raise TypeError("candidate event step ids must be integers")
    if result != CANDIDATE_EVENT_STEP_IDS:
        raise ValueError(
            f"candidate event step ids must equal {CANDIDATE_EVENT_STEP_IDS}; got {result}"
        )
    return result


def _candidate_mapping(value: Any, name: str) -> dict[int, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(type(step_id) is not int for step_id in value):
        raise TypeError(f"{name} keys must be integer event step ids")
    observed = set(value)
    expected = set(CANDIDATE_EVENT_STEP_IDS)
    if observed != expected:
        raise ValueError(
            f"{name} must cover exactly events {CANDIDATE_EVENT_STEP_IDS}; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return {step_id: value[step_id] for step_id in CANDIDATE_EVENT_STEP_IDS}


def summary_only_selection(event_step_ids: Iterable[int]) -> tuple[()]:
    validate_candidate_event_step_ids(event_step_ids)
    return ()


def recent_selection(event_step_ids: Iterable[int]) -> tuple[int, int]:
    validate_candidate_event_step_ids(event_step_ids)
    return RECENT_EVENT_STEP_IDS


def uniform_random_subsets(
    event_step_ids: Iterable[int],
) -> tuple[tuple[int, int], ...]:
    candidates = validate_candidate_event_step_ids(event_step_ids)
    # note (luojiaxuan): The frozen random baseline is an analytic expectation;
    # it has no sampled selector, Monte Carlo variance, seed, or hash ordering.
    return tuple(combinations(candidates, PRIMARY_BUDGET_EVENT_CAPACITY))


def _normalized_two_event_subset(value: Any) -> tuple[int, int]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("random-baseline subset keys must be event-id iterables")
    try:
        raw = tuple(value)
    except TypeError as error:
        raise TypeError("random-baseline subset keys must be event-id iterables") from error
    if len(raw) != PRIMARY_BUDGET_EVENT_CAPACITY:
        raise ValueError("random-baseline subsets must contain exactly two events")
    if any(type(step_id) is not int for step_id in raw):
        raise TypeError("random-baseline subset event ids must be integers")
    if len(set(raw)) != len(raw):
        raise ValueError("random-baseline subsets cannot repeat an event")
    result = tuple(sorted(raw))
    if not set(result).issubset(CANDIDATE_EVENT_STEP_IDS):
        raise ValueError("random-baseline subset contains an unknown event")
    return result


def uniform_random_exact_expectation(
    normalized_recovery_by_subset: Mapping[Iterable[int], Real],
    *,
    event_step_ids: Iterable[int],
) -> UniformRandomExactExpectation:
    if not isinstance(normalized_recovery_by_subset, Mapping):
        raise TypeError("normalized_recovery_by_subset must be a mapping")
    expected_subsets = uniform_random_subsets(event_step_ids)
    normalized: dict[tuple[int, int], float] = {}
    for raw_subset, raw_recovery in normalized_recovery_by_subset.items():
        subset = _normalized_two_event_subset(raw_subset)
        if subset in normalized:
            raise ValueError("random baseline contains duplicate normalized subsets")
        normalized[subset] = _finite_float(
            raw_recovery,
            f"normalized recovery for subset {subset}",
        )
    observed = set(normalized)
    expected = set(expected_subsets)
    if observed != expected:
        raise ValueError(
            "random baseline must contain all and only the six two-of-four subsets; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    ordered = tuple((subset, normalized[subset]) for subset in expected_subsets)
    return UniformRandomExactExpectation(
        subsets=expected_subsets,
        normalized_recovery_by_subset=ordered,
        mean_normalized_recovery=math.fsum(value for _, value in ordered)
        / len(ordered),
    )


def normalized_ocr_token_set(tokens: Iterable[str]) -> frozenset[str]:
    if isinstance(tokens, (str, bytes, bytearray, Mapping)):
        raise TypeError("OCR tokens must be an iterable of token strings")
    try:
        values = tuple(tokens)
    except TypeError as error:
        raise TypeError("OCR tokens must be an iterable of token strings") from error
    result: set[str] = set()
    for token in values:
        if not isinstance(token, str):
            raise TypeError("OCR tokens must be strings")
        normalized = normalize_screen_text(token)
        if not normalized:
            raise ValueError("OCR tokens cannot normalize to empty text")
        result.update(normalized.split(" "))
    return frozenset(result)


def ocr_token_set_jaccard(
    post_tokens: Iterable[str],
    current_tokens: Iterable[str],
) -> float:
    post = normalized_ocr_token_set(post_tokens)
    current = normalized_ocr_token_set(current_tokens)
    if not post and not current:
        return 1.0
    return len(post & current) / len(post | current)


def joint_rgb_histogram(resized_rgb_bytes: bytes | bytearray | memoryview) -> tuple[int, ...]:
    if not isinstance(resized_rgb_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("resized RGB input must be bytes-like")
    pixels = bytes(resized_rgb_bytes)
    if len(pixels) != RESIZED_RGB_BYTE_COUNT:
        raise ValueError(
            f"resized RGB input must contain exactly {RESIZED_RGB_BYTE_COUNT} bytes"
        )
    histogram = [0] * (RGB_HISTOGRAM_BINS_PER_CHANNEL**3)
    for offset in range(0, len(pixels), 3):
        red_bin = pixels[offset] * RGB_HISTOGRAM_BINS_PER_CHANNEL // 256
        green_bin = pixels[offset + 1] * RGB_HISTOGRAM_BINS_PER_CHANNEL // 256
        blue_bin = pixels[offset + 2] * RGB_HISTOGRAM_BINS_PER_CHANNEL // 256
        index = (
            (red_bin * RGB_HISTOGRAM_BINS_PER_CHANNEL + green_bin)
            * RGB_HISTOGRAM_BINS_PER_CHANNEL
            + blue_bin
        )
        histogram[index] += 1
    return tuple(histogram)


def _histogram_cosine(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("RGB histograms must have the same non-zero dimension")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("RGB histogram norm must be positive")
    return min(1.0, max(0.0, dot / (left_norm * right_norm)))


def joint_rgb_histogram_cosine(
    post_resized_rgb_bytes: bytes | bytearray | memoryview,
    current_resized_rgb_bytes: bytes | bytearray | memoryview,
) -> float:
    return _histogram_cosine(
        joint_rgb_histogram(post_resized_rgb_bytes),
        joint_rgb_histogram(current_resized_rgb_bytes),
    )


def select_top_two(scores_by_event_step: Mapping[int, Real]) -> BaselineSelection:
    raw_scores = _candidate_mapping(scores_by_event_step, "scores_by_event_step")
    scores = {
        step_id: _finite_float(raw_scores[step_id], f"score for event {step_id}")
        for step_id in CANDIDATE_EVENT_STEP_IDS
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

    ranking = tuple(sorted(CANDIDATE_EVENT_STEP_IDS, key=cmp_to_key(compare)))
    selected = tuple(sorted(ranking[:PRIMARY_BUDGET_EVENT_CAPACITY]))
    return BaselineSelection(
        scores_by_event_step=tuple(
            (step_id, scores[step_id]) for step_id in CANDIDATE_EVENT_STEP_IDS
        ),
        ranked_event_step_ids=ranking,
        selected_event_step_ids=selected,
    )


def ocr_and_rgb_similarity_baseline(
    *,
    event_ocr_tokens: Mapping[int, Iterable[str]],
    current_ocr_tokens: Iterable[str],
    event_resized_rgb_bytes: Mapping[int, bytes | bytearray | memoryview],
    current_resized_rgb_bytes: bytes | bytearray | memoryview,
) -> BaselineSelection:
    ocr_by_event = _candidate_mapping(event_ocr_tokens, "event_ocr_tokens")
    rgb_by_event = _candidate_mapping(
        event_resized_rgb_bytes,
        "event_resized_rgb_bytes",
    )
    current_tokens = normalized_ocr_token_set(current_ocr_tokens)
    current_histogram = joint_rgb_histogram(current_resized_rgb_bytes)
    scores: dict[int, float] = {}
    for step_id in CANDIDATE_EVENT_STEP_IDS:
        post_tokens = normalized_ocr_token_set(ocr_by_event[step_id])
        if not post_tokens and not current_tokens:
            text_score = 1.0
        else:
            text_score = len(post_tokens & current_tokens) / len(
                post_tokens | current_tokens
            )
        rgb_score = _histogram_cosine(
            joint_rgb_histogram(rgb_by_event[step_id]),
            current_histogram,
        )
        scores[step_id] = (
            OCR_JACCARD_WEIGHT * text_score + RGB_HISTOGRAM_WEIGHT * rgb_score
        )
    return select_top_two(scores)
