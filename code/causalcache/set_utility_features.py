"""Label-blind, variable-cardinality features for set-utility prediction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.gate_v1_data import HASH_DIMENSION, normalize_text, signed_hash64
from causalcache.low_fidelity_v2 import LOW_FIDELITY_V2_KEYS, LowFidelityEventV2
from causalcache.restoration_v2_baselines import (
    RESIZED_RGB_BYTE_COUNT,
    joint_rgb_histogram,
    normalized_ocr_token_set,
)


MAX_CANDIDATE_EVENTS = 16
QUERY_FEATURE_DIMENSION = HASH_DIMENSION
CONTEXT_FEATURE_NAMES = (
    "candidate_count_fraction",
    "decision_step_fraction",
    "current_ocr_count_fraction",
    "candidate_time_span_fraction",
)
EVENT_NUMERIC_FEATURE_NAMES = (
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
    "current_rgb_histogram_cosine",
)
PAIR_FEATURE_NAMES = (
    "event_ocr_jaccard",
    "event_rgb_histogram_cosine",
    "absolute_step_gap_fraction",
    "reciprocal_step_gap",
    "semantic_hash_cosine",
    "same_action_type",
    "same_foreground_app",
    "same_executor_result",
)
CONTEXT_FEATURE_DIMENSION = len(CONTEXT_FEATURE_NAMES)
EVENT_FEATURE_DIMENSION = HASH_DIMENSION + len(EVENT_NUMERIC_FEATURE_NAMES)
PAIR_FEATURE_DIMENSION = len(PAIR_FEATURE_NAMES)

_SCREEN_CHANGE_SCORE = {
    "none": 0.0,
    "low": 1.0 / 3.0,
    "medium": 2.0 / 3.0,
    "high": 1.0,
}
_EXECUTOR_RESULT_SCORE = {"failed": 0.0, "unknown": 0.5, "accepted": 1.0}
_EVENT_INPUT_KEYS = {
    "low_fidelity_v2",
    "post_ocr_spatial_tokens",
    "post_resized_rgb_bytes",
}


def _ordered_ocr_tokens(value: Any, *, label: str) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes, bytearray, Mapping))
        or not isinstance(value, Sequence)
    ):
        raise ValueError(f"{label} must be an ordered OCR-token sequence")
    tokens = tuple(value)
    if any(not isinstance(token, str) or not normalize_text(token) for token in tokens):
        raise ValueError(f"{label} must contain non-empty text tokens")
    normalized_ocr_token_set(tokens)
    return tokens


def _resized_rgb_bytes(value: Any, *, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} must be bytes-like")
    result = bytes(value)
    if len(result) != RESIZED_RGB_BYTE_COUNT:
        raise ValueError(
            f"{label} must contain exactly {RESIZED_RGB_BYTE_COUNT} RGB bytes"
        )
    return result


@dataclass(frozen=True)
class SetUtilityEventInput:
    """One candidate's label-free low/high-fidelity feature substrate."""

    low_fidelity_v2: LowFidelityEventV2
    post_ocr_spatial_tokens: tuple[str, ...]
    post_resized_rgb_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.low_fidelity_v2, LowFidelityEventV2):
            raise TypeError("set-utility low-fidelity event must be canonical v2")
        tokens = _ordered_ocr_tokens(
            self.post_ocr_spatial_tokens,
            label="candidate post-state OCR",
        )
        pixels = _resized_rgb_bytes(
            self.post_resized_rgb_bytes,
            label="candidate post-state RGB",
        )
        object.__setattr__(self, "post_ocr_spatial_tokens", tokens)
        object.__setattr__(self, "post_resized_rgb_bytes", pixels)

    @property
    def event_step_id(self) -> int:
        return self.low_fidelity_v2.step_id

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SetUtilityEventInput":
        """Parse only the three label-blind fields and reject all extras."""
        if not isinstance(value, Mapping) or set(value) != _EVENT_INPUT_KEYS:
            raise ValueError(
                "set-utility event input must contain exactly the three label-blind fields"
            )
        low_fidelity = value["low_fidelity_v2"]
        if not isinstance(low_fidelity, Mapping):
            raise ValueError("set-utility low-fidelity input must be a mapping")
        return cls(
            low_fidelity_v2=LowFidelityEventV2.from_mapping(low_fidelity),
            post_ocr_spatial_tokens=_ordered_ocr_tokens(
                value["post_ocr_spatial_tokens"],
                label="candidate post-state OCR",
            ),
            post_resized_rgb_bytes=_resized_rgb_bytes(
                value["post_resized_rgb_bytes"],
                label="candidate post-state RGB",
            ),
        )


@dataclass(frozen=True)
class SetUtilityFeatureState:
    """Padded model features; identities and event ids are audit joins only."""

    source_id: str
    state_id: str
    decision_step_id: int
    event_step_ids: tuple[int, ...]
    query_features: tuple[float, ...]
    context_features: tuple[float, ...]
    event_features: tuple[tuple[float, ...], ...]
    pair_features: tuple[tuple[tuple[float, ...], ...], ...]
    event_mask: tuple[bool, ...]

    @property
    def event_count(self) -> int:
        return len(self.event_step_ids)

    @property
    def padded_event_count(self) -> int:
        return len(self.event_mask)

    def event_feature(self, event_step_id: int) -> tuple[float, ...]:
        try:
            index = self.event_step_ids.index(event_step_id)
        except ValueError as error:
            raise KeyError(event_step_id) from error
        return self.event_features[index]

    def pair_feature(
        self, left_event_step_id: int, right_event_step_id: int
    ) -> tuple[float, ...]:
        try:
            left = self.event_step_ids.index(left_event_step_id)
            right = self.event_step_ids.index(right_event_step_id)
        except ValueError as error:
            missing = (
                left_event_step_id
                if left_event_step_id not in self.event_step_ids
                else right_event_step_id
            )
            raise KeyError(missing) from error
        return self.pair_features[left][right]


def _semantic_hash(event: SetUtilityEventInput) -> tuple[float, ...]:
    low_fidelity = event.low_fidelity_v2.to_ordered_dict()
    fields: list[tuple[str, str | Sequence[Any]]] = [
        (name, low_fidelity[name]) for name in LOW_FIDELITY_V2_KEYS
    ]
    fields.append(
        ("candidate_post_ocr_spatial_token", event.post_ocr_spatial_tokens)
    )
    return signed_hash64(fields)


def _set_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _histogram_cosine(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("RGB histograms must have the same non-zero dimension")
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("RGB histogram norm must be positive")
    return min(1.0, max(0.0, dot / (left_norm * right_norm)))


def _hash_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return min(
        1.0,
        max(
            -1.0,
            math.fsum(a * b for a, b in zip(left, right, strict=True)),
        ),
    )


def _event_numeric_features(
    event: SetUtilityEventInput,
    *,
    decision_step_id: int,
    current_ocr: frozenset[str],
    current_histogram: tuple[int, ...],
    event_ocr: frozenset[str],
    event_histogram: tuple[int, ...],
) -> tuple[float, ...]:
    low = event.low_fidelity_v2
    age = decision_step_id - low.step_id
    values = (
        age / decision_step_id,
        low.step_id / decision_step_id,
        1.0 / age,
        min(len(low.action_argument), 64) / 64.0,
        len(low.screen_text_added) / 32.0,
        len(low.screen_text_removed) / 32.0,
        min(len(event.post_ocr_spatial_tokens), 128) / 128.0,
        _SCREEN_CHANGE_SCORE[low.screen_change],
        _EXECUTOR_RESULT_SCORE[low.executor_result],
        _set_jaccard(event_ocr, current_ocr),
        _histogram_cosine(event_histogram, current_histogram),
    )
    if len(values) != len(EVENT_NUMERIC_FEATURE_NAMES) or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values
    ):
        raise RuntimeError("set-utility event feature construction drifted")
    return values


def _pair_features(
    left: SetUtilityEventInput,
    right: SetUtilityEventInput,
    *,
    decision_step_id: int,
    left_hash: Sequence[float],
    right_hash: Sequence[float],
    left_ocr: frozenset[str],
    right_ocr: frozenset[str],
    left_histogram: tuple[int, ...],
    right_histogram: tuple[int, ...],
) -> tuple[float, ...]:
    gap = abs(left.event_step_id - right.event_step_id)
    values = (
        _set_jaccard(left_ocr, right_ocr),
        _histogram_cosine(left_histogram, right_histogram),
        gap / decision_step_id,
        1.0 / (1.0 + gap),
        _hash_cosine(left_hash, right_hash),
        float(left.low_fidelity_v2.action_type == right.low_fidelity_v2.action_type),
        float(
            left.low_fidelity_v2.foreground_app
            == right.low_fidelity_v2.foreground_app
        ),
        float(
            left.low_fidelity_v2.executor_result
            == right.low_fidelity_v2.executor_result
        ),
    )
    if len(values) != PAIR_FEATURE_DIMENSION or any(
        not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in values
    ):
        raise RuntimeError("set-utility pair feature construction drifted")
    return values


def build_set_utility_feature_state(
    *,
    source_id: str,
    state_id: str,
    decision_step_id: int,
    instruction: str,
    current_ocr_spatial_tokens: Sequence[str],
    current_resized_rgb_bytes: bytes | bytearray | memoryview,
    events: Sequence[SetUtilityEventInput],
    pad_to: int = MAX_CANDIDATE_EVENTS,
) -> SetUtilityFeatureState:
    """Build label-free features while preserving the caller's event order.

    Reordering ``events`` reorders the event axis and both pair axes, but never
    changes an individual feature value. This is the feature-side permutation
    equivariance required by invariant set predictors.
    """
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("set-utility source_id must be non-empty text")
    if not isinstance(state_id, str) or not state_id:
        raise ValueError("set-utility state_id must be non-empty text")
    if type(decision_step_id) is not int or decision_step_id <= 1:
        raise ValueError("set-utility decision step must exceed one")
    if not isinstance(instruction, str) or not normalize_text(instruction):
        raise ValueError("set-utility instruction must be non-empty text")
    if (
        isinstance(events, (str, bytes, bytearray, Mapping))
        or not isinstance(events, Sequence)
    ):
        raise TypeError("set-utility events must be an ordered sequence")
    parsed = tuple(events)
    if not parsed or any(not isinstance(event, SetUtilityEventInput) for event in parsed):
        raise ValueError("set-utility events must contain canonical event inputs")
    if len(parsed) > MAX_CANDIDATE_EVENTS:
        raise ValueError("set-utility candidate count exceeds the frozen maximum of 16")
    if type(pad_to) is not int or not len(parsed) <= pad_to <= MAX_CANDIDATE_EVENTS:
        raise ValueError("set-utility pad_to must cover all events and not exceed 16")

    event_ids = tuple(event.event_step_id for event in parsed)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("set-utility event step ids must be unique")
    if any(step_id >= decision_step_id for step_id in event_ids):
        raise ValueError("set-utility candidates must strictly precede the decision")

    current_tokens = _ordered_ocr_tokens(
        current_ocr_spatial_tokens,
        label="current-state OCR",
    )
    current_pixels = _resized_rgb_bytes(
        current_resized_rgb_bytes,
        label="current-state RGB",
    )
    current_ocr = normalized_ocr_token_set(current_tokens)
    current_histogram = joint_rgb_histogram(current_pixels)
    query_features = signed_hash64(
        (
            ("instruction", instruction),
            ("current_ocr_spatial_token", current_tokens),
        )
    )

    hashes = tuple(_semantic_hash(event) for event in parsed)
    ocr_sets = tuple(
        normalized_ocr_token_set(event.post_ocr_spatial_tokens) for event in parsed
    )
    histograms = tuple(
        joint_rgb_histogram(event.post_resized_rgb_bytes) for event in parsed
    )
    valid_event_features = tuple(
        (
            *hashes[index],
            *_event_numeric_features(
                event,
                decision_step_id=decision_step_id,
                current_ocr=current_ocr,
                current_histogram=current_histogram,
                event_ocr=ocr_sets[index],
                event_histogram=histograms[index],
            ),
        )
        for index, event in enumerate(parsed)
    )

    valid_pairs = tuple(
        tuple(
            _pair_features(
                left,
                right,
                decision_step_id=decision_step_id,
                left_hash=hashes[left_index],
                right_hash=hashes[right_index],
                left_ocr=ocr_sets[left_index],
                right_ocr=ocr_sets[right_index],
                left_histogram=histograms[left_index],
                right_histogram=histograms[right_index],
            )
            for right_index, right in enumerate(parsed)
        )
        for left_index, left in enumerate(parsed)
    )

    zero_event = (0.0,) * EVENT_FEATURE_DIMENSION
    zero_pair = (0.0,) * PAIR_FEATURE_DIMENSION
    padding = pad_to - len(parsed)
    event_features = valid_event_features + (zero_event,) * padding
    pair_features = tuple(
        row + (zero_pair,) * padding for row in valid_pairs
    ) + (tuple(zero_pair for _ in range(pad_to)),) * padding
    event_mask = (True,) * len(parsed) + (False,) * padding
    context_features = (
        len(parsed) / MAX_CANDIDATE_EVENTS,
        min(decision_step_id, 64) / 64.0,
        min(len(current_tokens), 128) / 128.0,
        (max(event_ids) - min(event_ids)) / decision_step_id,
    )

    if len(query_features) != QUERY_FEATURE_DIMENSION:
        raise RuntimeError("set-utility query feature dimension drifted")
    if len(context_features) != CONTEXT_FEATURE_DIMENSION:
        raise RuntimeError("set-utility context feature dimension drifted")
    if any(len(row) != EVENT_FEATURE_DIMENSION for row in event_features):
        raise RuntimeError("set-utility event feature dimension drifted")
    if len(pair_features) != pad_to or any(
        len(row) != pad_to
        or any(len(cell) != PAIR_FEATURE_DIMENSION for cell in row)
        for row in pair_features
    ):
        raise RuntimeError("set-utility pair feature dimensions drifted")
    all_scalars = (
        *query_features,
        *context_features,
        *(value for row in event_features for value in row),
        *(value for row in pair_features for cell in row for value in cell),
    )
    if any(not math.isfinite(value) for value in all_scalars):
        raise RuntimeError("set-utility features must be finite")

    return SetUtilityFeatureState(
        source_id=source_id,
        state_id=state_id,
        decision_step_id=decision_step_id,
        event_step_ids=event_ids,
        query_features=query_features,
        context_features=context_features,
        event_features=event_features,
        pair_features=pair_features,
        event_mask=event_mask,
    )


def event_ocr_rgb_score(event_features: Sequence[float]) -> float:
    """Recover the frozen equal-weight OCR/RGB heuristic from one event row."""
    if len(event_features) != EVENT_FEATURE_DIMENSION:
        raise ValueError("set-utility event feature dimension drifted")
    offset = HASH_DIMENSION
    ocr_index = EVENT_NUMERIC_FEATURE_NAMES.index("current_ocr_jaccard")
    rgb_index = EVENT_NUMERIC_FEATURE_NAMES.index("current_rgb_histogram_cosine")
    return 0.5 * (
        float(event_features[offset + ocr_index])
        + float(event_features[offset + rgb_index])
    )


def event_recency_score(event_features: Sequence[float]) -> float:
    """Return the explicit monotone recency scalar from one event row."""
    if len(event_features) != EVENT_FEATURE_DIMENSION:
        raise ValueError("set-utility event feature dimension drifted")
    index = EVENT_NUMERIC_FEATURE_NAMES.index("reciprocal_age")
    return float(event_features[HASH_DIMENSION + index])


__all__ = [
    "CONTEXT_FEATURE_DIMENSION",
    "CONTEXT_FEATURE_NAMES",
    "EVENT_FEATURE_DIMENSION",
    "EVENT_NUMERIC_FEATURE_NAMES",
    "MAX_CANDIDATE_EVENTS",
    "PAIR_FEATURE_DIMENSION",
    "PAIR_FEATURE_NAMES",
    "QUERY_FEATURE_DIMENSION",
    "SetUtilityEventInput",
    "SetUtilityFeatureState",
    "build_set_utility_feature_state",
    "event_ocr_rgb_score",
    "event_recency_score",
]
