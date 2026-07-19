"""Small exploratory data path for the first set-utility predictor result."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.restoration_v2_text_backend import prepare_image_bytes
from causalcache.set_utility_data import (
    SetUtilityState,
    set_utility_state_from_feature_and_table,
)
from causalcache.set_utility_features import (
    SetUtilityEventInput,
    SetUtilityFeatureState,
    build_set_utility_feature_state,
)
from causalcache.set_utility_label_inputs import JoinedUtilityQueryInput
from causalcache.set_utility_label_table import (
    validate_cardinality_capped_distance_table,
)
from causalcache.set_utility_processor_freeze import FrozenQueryCandidateRecord
from causalcache.set_utility_processor_substrate import SPLIT_ROLES


COMPLETED_STATE_STATUS = "COMPLETED_SET_UTILITY_MVP_STATE"
SKIPPED_STATE_STATUS = "SKIPPED_SET_UTILITY_MVP_STATE"


def validated_teacher_settings(config: Mapping[str, Any]) -> tuple[float, int]:
    teacher = config.get("teacher") if isinstance(config, Mapping) else None
    if not isinstance(teacher, Mapping):
        raise ValueError("MVP config is missing teacher settings")
    repeat_kl = teacher.get("maximum_reference_repeat_kl")
    microbatch_size = teacher.get("microbatch_size")
    if (
        isinstance(repeat_kl, bool)
        or not isinstance(repeat_kl, (int, float))
        or not math.isfinite(float(repeat_kl))
        or float(repeat_kl) < 0.0
    ):
        raise ValueError("maximum reference repeat KL must be finite and non-negative")
    if type(microbatch_size) is not int or microbatch_size not in (1, 2):
        raise ValueError("GUI-Owl MVP teacher microbatch size must be one or two")
    return float(repeat_kl), microbatch_size


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def read_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("JSONL rows must be objects")
            rows.append(value)
    return tuple(rows)


def select_worker_candidate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_count: int,
    query_kind: str,
    quota_by_role: Mapping[str, int],
    selection_salt: str,
) -> tuple[FrozenQueryCandidateRecord, ...]:
    """Select a small role-balanced roster without looking at policy outputs."""
    if type(candidate_count) is not int or candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if not isinstance(query_kind, str) or not query_kind:
        raise ValueError("query_kind must be non-empty")
    if not isinstance(selection_salt, str) or not selection_salt:
        raise ValueError("selection_salt must be non-empty")
    if set(quota_by_role) != set(SPLIT_ROLES) or any(
        type(value) is not int or value <= 0 for value in quota_by_role.values()
    ):
        raise ValueError("quota_by_role must contain positive train/tune/evaluation quotas")

    by_role: dict[str, list[FrozenQueryCandidateRecord]] = {
        role: [] for role in SPLIT_ROLES
    }
    for raw in records:
        candidate = FrozenQueryCandidateRecord.from_payload(raw)
        if (
            candidate.query_kind == query_kind
            and len(candidate.candidate_context.candidate_event_step_ids)
            == candidate_count
        ):
            by_role[candidate.role].append(candidate)

    selected: list[FrozenQueryCandidateRecord] = []
    for role in SPLIT_ROLES:
        ranked = sorted(
            by_role[role],
            key=lambda item: (
                hashlib.sha256(
                    f"{selection_salt}:{item.state_id}".encode("utf-8")
                ).hexdigest(),
                item.state_id,
            ),
        )
        quota = quota_by_role[role]
        if len(ranked) < quota:
            raise ValueError(f"worker has only {len(ranked)} eligible {role} states")
        selected.extend(ranked[:quota])
    return tuple(sorted(selected, key=lambda item: item.state_id))


def build_feature_state_from_joined(
    joined: JoinedUtilityQueryInput,
    *,
    ocr_records_by_path: Mapping[str, Mapping[str, Any]],
) -> SetUtilityFeatureState:
    if not isinstance(joined, JoinedUtilityQueryInput):
        raise TypeError("feature construction requires one joined query")
    query = joined.query
    source = joined.image_payloads
    history = {event.event_step_id: event for event in query.history_events}
    ocr = {
        path: tuple(record["full_spatial_tokens"])
        for path, record in validated_ocr_records(ocr_records_by_path).items()
    }
    prepared = {
        path: prepare_image_bytes(payload).resized_rgb_bytes
        for path, payload in source.items()
    }
    events = tuple(
        SetUtilityEventInput(
            low_fidelity_v2=history[step_id].low_fidelity_summary,
            post_ocr_spatial_tokens=ocr[
                history[step_id].high_fidelity_observation_ref
            ],
            post_resized_rgb_bytes=prepared[
                history[step_id].high_fidelity_observation_ref
            ],
        )
        for step_id in query.candidate_event_step_ids
    )
    return build_set_utility_feature_state(
        source_id=query.trajectory_id,
        state_id=query.state_id,
        decision_step_id=query.decision_step_id,
        instruction=query.task_instruction,
        current_ocr_spatial_tokens=ocr[query.current_observation_ref],
        current_resized_rgb_bytes=prepared[query.current_observation_ref],
        events=events,
        pad_to=len(events),
    )


def validated_ocr_records(
    records: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(records, Mapping):
        raise ValueError("MVP feature input is missing processor OCR records")
    for path, value in records.items():
        tokens = value.get("full_spatial_tokens") if isinstance(value, Mapping) else None
        if (
            not isinstance(path, str)
            or not isinstance(tokens, list)
            or any(not isinstance(token, str) for token in tokens)
        ):
            raise ValueError("processor OCR record schema drifted")
    return records


def feature_state_to_payload(state: SetUtilityFeatureState) -> dict[str, Any]:
    return {
        "context_features": list(state.context_features),
        "decision_step_id": state.decision_step_id,
        "event_features": [list(row) for row in state.event_features],
        "event_mask": list(state.event_mask),
        "event_step_ids": list(state.event_step_ids),
        "pair_features": [
            [[*cell] for cell in row] for row in state.pair_features
        ],
        "query_features": list(state.query_features),
        "source_id": state.source_id,
        "state_id": state.state_id,
    }


def feature_state_from_payload(value: Mapping[str, Any]) -> SetUtilityFeatureState:
    return SetUtilityFeatureState(
        source_id=str(value["source_id"]),
        state_id=str(value["state_id"]),
        decision_step_id=int(value["decision_step_id"]),
        event_step_ids=tuple(value["event_step_ids"]),
        query_features=tuple(value["query_features"]),
        context_features=tuple(value["context_features"]),
        event_features=tuple(tuple(row) for row in value["event_features"]),
        pair_features=tuple(
            tuple(tuple(cell) for cell in row) for row in value["pair_features"]
        ),
        event_mask=tuple(value["event_mask"]),
    )


def state_from_completed_record(value: Mapping[str, Any]) -> SetUtilityState:
    if value.get("status") != COMPLETED_STATE_STATUS:
        raise ValueError("MVP training accepts only completed state records")
    feature = feature_state_from_payload(value["feature"])
    distances = {
        tuple(row["coalition_event_step_ids"]): float(row["distance"])
        for row in value["distance_rows"]
    }
    table = validate_cardinality_capped_distance_table(
        split=str(value["role"]),
        state_id=str(value["state_id"]),
        candidate_event_step_ids=tuple(value["candidate_event_step_ids"]),
        maximum_labeled_cardinality=int(value["maximum_labeled_cardinality"]),
        distances=distances,
    )
    return set_utility_state_from_feature_and_table(feature, table)


__all__ = [
    "COMPLETED_STATE_STATUS",
    "SKIPPED_STATE_STATUS",
    "build_feature_state_from_joined",
    "canonical_json_bytes",
    "feature_state_from_payload",
    "feature_state_to_payload",
    "read_jsonl",
    "select_worker_candidate_records",
    "state_from_completed_record",
    "validated_teacher_settings",
    "validated_ocr_records",
]
