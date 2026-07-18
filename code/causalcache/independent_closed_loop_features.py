"""Frozen live-history feature adapter for the independent closed-loop gate."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.gate_v1_data import (
    INDEPENDENT_INPUT_DIMENSION,
    CandidateFeatures,
    FeatureState,
    independent_input,
    normalize_text,
    signed_hash64,
)
from causalcache.low_fidelity_v2 import (
    LOW_FIDELITY_V2_KEYS,
    LowFidelityEventV2,
    ScreenTextNode,
    action_argument,
    screen_change_from_mean_absolute_rgb_difference,
    screen_text_delta,
)
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.restoration_v2_text_backend import (
    PreparedImage,
    build_ocr_record,
    canonical_json_bytes,
    mean_absolute_rgb_difference_from_prepared,
    prepare_image_bytes,
    validate_backend_config,
)


@dataclass(frozen=True)
class GateV1OCRBackendBinding:
    config_sha256: str
    manifest_sha256: str
    model_repo: str
    model_revision: str


FROZEN_GATE_V1_OCR_BACKEND = GateV1OCRBackendBinding(
    config_sha256=(
        "51e08a9565f804ecb1ee7f12887cc12742b84c04996fffbbde7c0f304e4bd036"
    ),
    manifest_sha256=(
        "107478672438b52e9c2ccc9ef8de5d13d4e16329ab1afbe95772e2d3df5bd2a9"
    ),
    model_repo="gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en",
    model_revision="0dbc766a73ee88d10d52285d434dbfec58617835",
)


class _FeatureStateGateView:
    def __init__(self, state: FeatureState) -> None:
        self.source_id = state.source_id
        self.state_id = state.state_id
        self.decision_step_id = state.decision_step_id
        self.candidate_event_step_ids = state.candidate_event_step_ids
        self.q64 = state.q64
        self.candidates = state.candidates

    def candidate(self, event_step_id: int) -> CandidateFeatures:
        for candidate in self.candidates:
            if candidate.event_step_id == event_step_id:
                return candidate
        raise KeyError(event_step_id)


def _candidate_h64(
    low_fidelity: Mapping[str, Any],
    post_ocr_tokens: Sequence[str],
) -> tuple[float, ...]:
    fields: list[tuple[str, str | Sequence[Any]]] = [
        (name, low_fidelity[name]) for name in LOW_FIDELITY_V2_KEYS
    ]
    fields.append(("candidate_post_ocr_spatial_token", post_ocr_tokens))
    return signed_hash64(fields)


def _candidate_g8(
    low_fidelity: Mapping[str, Any],
    *,
    history_length: int,
    event_age: int,
    post_ocr_tokens: Sequence[str],
) -> tuple[float, ...]:
    if type(history_length) is not int or history_length <= 0:
        raise ValueError("history length must be positive")
    if type(event_age) is not int or not 0 <= event_age < history_length:
        raise ValueError("event age must be in the live history horizon")
    argument = unicodedata.normalize("NFKC", str(low_fidelity["action_argument"]))
    added = low_fidelity["screen_text_added"]
    removed = low_fidelity["screen_text_removed"]
    if not isinstance(added, list) or not isinstance(removed, list):
        raise ValueError("screen-text deltas must be arrays")
    change_score = {
        "none": 0.0,
        "low": 1.0 / 3.0,
        "medium": 2.0 / 3.0,
        "high": 1.0,
    }
    result_score = {"failed": 0.0, "unknown": 0.5, "accepted": 1.0}
    try:
        values = (
            event_age / history_length,
            (history_length - event_age) / history_length,
            min(len(argument), 64) / 64.0,
            min(len(added), 32) / 32.0,
            min(len(removed), 32) / 32.0,
            min(len(post_ocr_tokens), 128) / 128.0,
            change_score[str(low_fidelity["screen_change"])],
            result_score[str(low_fidelity["executor_result"])],
        )
    except KeyError as error:
        raise ValueError("gate g8 categorical value drifted") from error
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("gate g8 values must be finite in [0, 1]")
    return values


def _spatial_tokens(value: Any, *, label: str) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes, bytearray, Mapping))
        or not isinstance(value, Sequence)
    ):
        raise ValueError(f"{label} must be an ordered OCR spatial-token sequence")
    tokens = tuple(value)
    if any(
        not isinstance(token, str)
        or not token
        or token != unicodedata.normalize("NFKC", token)
        or any(character.isspace() for character in token)
        for token in tokens
    ):
        raise ValueError(f"{label} must contain exact canonical OCR spatial tokens")
    return tokens


def _screen_text_nodes(record: Mapping[str, Any]) -> tuple[ScreenTextNode, ...]:
    nodes = record.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("OCR nodes must be a JSON array")
    result = []
    for node in nodes:
        if not isinstance(node, Mapping):
            raise ValueError("OCR nodes must be JSON objects")
        bbox = node.get("bbox_top_left_bottom_right")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("OCR node bbox must contain top,left,bottom,right")
        result.append(
            ScreenTextNode(
                text=str(node["normalized_text"]),
                top=bbox[0],
                left=bbox[1],
                bottom=bbox[2],
                right=bbox[3],
            )
        )
    return tuple(result)


def _validate_live_ocr_record(
    record: Mapping[str, Any],
    *,
    image_bytes: bytes,
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
) -> PreparedImage:
    validate_backend_config(backend_config)
    if not isinstance(record, Mapping):
        raise ValueError("live OCR record must be a canonical mapping")
    nodes = record.get("nodes")
    if not isinstance(nodes, list) or any(
        not isinstance(node, Mapping) for node in nodes
    ):
        raise ValueError("live OCR record nodes are malformed")
    try:
        rebuilt = build_ocr_record(
            image_member_path=record["image_member_path"],
            image_bytes=image_bytes,
            backend_config_sha256=backend_config_sha256,
            boxes=[node["polygon_xy"] for node in nodes],
            texts=[node["raw_text"] for node in nodes],
            scores=[float(node["confidence_decimal_string"]) for node in nodes],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("live OCR record cannot be canonically rebuilt") from error
    if canonical_json_bytes(dict(record)) != canonical_json_bytes(rebuilt):
        raise ValueError("live OCR record differs from exact canonical rebuild")
    return prepare_image_bytes(image_bytes)


def live_history_event_from_transition(
    *,
    step_id: int,
    action: GUIOwlV2Action,
    before_image_bytes: bytes,
    after_image_bytes: bytes,
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
    before_ocr_record: Mapping[str, Any],
    after_ocr_record: Mapping[str, Any],
    ocr_backend_binding: GateV1OCRBackendBinding,
) -> dict[str, Any]:
    """Build one exact gate event from a raw AndroidWorld transition."""
    if type(step_id) is not int or step_id <= 0:
        raise ValueError("live transition step_id must be a positive integer")
    if not isinstance(action, GUIOwlV2Action):
        raise TypeError("live transition action must be a canonical GUIOwlV2Action")
    if not isinstance(before_image_bytes, bytes) or not before_image_bytes:
        raise ValueError("before image must be non-empty bytes")
    if not isinstance(after_image_bytes, bytes) or not after_image_bytes:
        raise ValueError("after image must be non-empty bytes")
    if ocr_backend_binding != FROZEN_GATE_V1_OCR_BACKEND:
        raise ValueError(
            "live OCR backend identity drifted from the frozen gate-v1 binding"
        )
    if backend_config_sha256 != FROZEN_GATE_V1_OCR_BACKEND.config_sha256:
        raise ValueError("live OCR backend config SHA256 drifted")
    before_prepared = _validate_live_ocr_record(
        before_ocr_record,
        image_bytes=before_image_bytes,
        backend_config=backend_config,
        backend_config_sha256=backend_config_sha256,
    )
    after_prepared = _validate_live_ocr_record(
        after_ocr_record,
        image_bytes=after_image_bytes,
        backend_config=backend_config,
        backend_config_sha256=backend_config_sha256,
    )
    text_delta = screen_text_delta(
        _screen_text_nodes(before_ocr_record),
        _screen_text_nodes(after_ocr_record),
    )
    mean_difference = mean_absolute_rgb_difference_from_prepared(
        before_prepared,
        after_prepared,
    )
    low_fidelity = LowFidelityEventV2(
        step_id=step_id,
        action_type=action.action,
        action_argument=action_argument(action.arguments()),
        foreground_app="unknown",
        screen_text_added=text_delta.added,
        screen_text_removed=text_delta.removed,
        screen_change=screen_change_from_mean_absolute_rgb_difference(
            mean_difference
        ),
        executor_result="unknown",
    )
    return {
        "low_fidelity_v2": low_fidelity.to_ordered_dict(),
        "post_ocr_spatial_tokens": list(after_ocr_record["full_spatial_tokens"]),
    }


def feature_state_from_live_history(
    *,
    source_id: str,
    state_id: str,
    decision_step_id: int,
    instruction: str,
    current_ocr_spatial_tokens: Sequence[str],
    history_events: Sequence[Mapping[str, Any]],
    ocr_backend_binding: GateV1OCRBackendBinding,
) -> FeatureState:
    """Project an arbitrary-length live history into the frozen gate-v1 features.

    History must be oldest-to-newest. Age is frozen to ``N - 1 - index``; callers
    cannot provide it. The newest event therefore has age zero and is excluded by
    the frozen current-equivalent memory rule.
    """
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("live source_id must be non-empty text")
    if not isinstance(state_id, str) or not state_id:
        raise ValueError("live state_id must be non-empty text")
    if type(decision_step_id) is not int or decision_step_id <= 0:
        raise ValueError("live decision_step_id must be a positive integer")
    if not isinstance(instruction, str) or not normalize_text(instruction):
        raise ValueError("live instruction must be non-empty text")
    if ocr_backend_binding != FROZEN_GATE_V1_OCR_BACKEND:
        raise ValueError(
            "live OCR backend identity drifted from the frozen gate-v1 binding"
        )
    current_tokens = _spatial_tokens(
        current_ocr_spatial_tokens,
        label="current observation OCR",
    )
    if (
        isinstance(history_events, (str, bytes, bytearray, Mapping))
        or not isinstance(history_events, Sequence)
    ):
        raise ValueError("live history must be an ordered sequence")

    expected_keys = {"low_fidelity_v2", "post_ocr_spatial_tokens"}
    parsed: list[tuple[int, LowFidelityEventV2, tuple[str, ...]]] = []
    history_length = len(history_events)
    for index, raw_event in enumerate(history_events):
        if not isinstance(raw_event, Mapping) or set(raw_event) != expected_keys:
            raise ValueError("live history event field inventory drifted")
        age = history_length - 1 - index
        try:
            low_fidelity = LowFidelityEventV2.from_mapping(raw_event["low_fidelity_v2"])
        except (TypeError, ValueError) as error:
            raise ValueError("live history low_fidelity_v2 is not canonical") from error
        post_tokens = _spatial_tokens(
            raw_event["post_ocr_spatial_tokens"],
            label=f"history event {index} post-state OCR",
        )
        parsed.append((age, low_fidelity, post_tokens))

    event_ids = tuple(event.step_id for _, event, _ in parsed)
    if len(set(event_ids)) != history_length:
        raise ValueError("live history event step ids must be unique")
    if parsed and parsed[-1][2] != current_tokens:
        raise ValueError(
            "newest history post-state OCR must equal current observation OCR"
        )

    q64 = signed_hash64(
        (
            ("instruction", instruction),
            ("current_ocr_spatial_token", current_tokens),
        )
    )
    candidates = []
    for age, low_fidelity, post_tokens in parsed:
        if age == 0:
            continue
        low_mapping = low_fidelity.to_ordered_dict()
        candidates.append(
            CandidateFeatures(
                event_step_id=low_fidelity.step_id,
                h64=_candidate_h64(low_mapping, post_tokens),
                g8=_candidate_g8(
                    low_mapping,
                    history_length=history_length,
                    event_age=age,
                    post_ocr_tokens=post_tokens,
                ),
            )
        )
    candidates.sort(key=lambda candidate: candidate.event_step_id)
    candidate_ids = tuple(candidate.event_step_id for candidate in candidates)
    return FeatureState(
        source_id=source_id,
        state_id=state_id,
        decision_step_id=decision_step_id,
        candidate_event_step_ids=candidate_ids,
        q64=q64,
        candidates=tuple(candidates),
    )


def independent_input_from_feature_state(
    state: FeatureState,
    event_step_id: int,
) -> tuple[float, ...]:
    """Invoke the frozen gate-v1 200d function through a feature-only view."""
    if not isinstance(state, FeatureState):
        raise TypeError("independent live input requires one FeatureState")
    vector = independent_input(_FeatureStateGateView(state), event_step_id)
    if len(vector) != INDEPENDENT_INPUT_DIMENSION:
        raise RuntimeError("frozen independent input dimension drifted")
    return vector


__all__ = [
    "FROZEN_GATE_V1_OCR_BACKEND",
    "GateV1OCRBackendBinding",
    "feature_state_from_live_history",
    "independent_input_from_feature_state",
    "live_history_event_from_transition",
]
