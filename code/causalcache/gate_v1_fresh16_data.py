"""Fail-closed selective semantic readers for the fresh-16 gate slice."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from causalcache.gate_v1_contract import canonical_json_bytes
from causalcache.gate_v1_data import LabelState, label_state_from_restoration_record


TRAJECTORY_TRANSPORT = "expansion_trajectories_jsonl"
RAW_STATE_TRANSPORT = "expansion_raw_states_jsonl"
FRESH_ROLE = "gate_development_expansion"
FRESH_TRAJECTORY_ROWS = tuple(range(48, 64))
FRESH_RAW_STATE_ROWS = tuple(range(144, 192))
FRESH_TRAJECTORY_COUNT = 16
FRESH_STATE_COUNT = 48
FRESH_SOURCE_IDS_SHA256 = (
    "1c37cbf6b67b0ddee61b3efe27d33b30c8fbfb471ced12f12e49624a10b82454"
)
FRESH_TRAJECTORY_TRANSPORT_SHA256 = (
    "fe93e9deeeb9a3c018227fb781b29f6efefefbb728db987584b650d9df353a6d"
)
FRESH_TRAJECTORY_TRANSPORT_SIZE_BYTES = 1_245_673
FRESH_RAW_STATE_TRANSPORT_SHA256 = (
    "850aa0f1a6c0c6dd312573e4963384257a724ba01a3fe78ac4496cfd0fd14267"
)
FRESH_RAW_STATE_TRANSPORT_SIZE_BYTES = 1_126_543
FRESH_LABEL_RUN_CONTRACT_SHA256 = (
    "63cddd712d9fa22bbd68110966161f36318d5fe1962b9b1477bae342da4180b8"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_VERIFICATION_SEAL = object()
_TRANSPORT_COUNTS = {
    TRAJECTORY_TRANSPORT: 64,
    RAW_STATE_TRANSPORT: 192,
}

_TRAJECTORY_KEYS = {
    "source_id",
    "role",
    "instruction",
    "instruction_sha256",
    "platform",
    "apps",
    "device_name",
    "resolution",
    "terminal_status",
    "source",
    "selection",
    "events",
    "decisions",
    "content_witness_sha256",
}
_EVENT_KEYS = {
    "step_id",
    "observation_before_path",
    "observation_before_sha256",
    "observation_after_path",
    "observation_after_sha256",
    "executed_action",
    "source_tool_call",
    "low_fidelity_v2",
    "low_fidelity_v2_serialized",
    "low_fidelity_v2_sha256",
    "low_fidelity_v2_metadata",
    "high_fidelity_v2",
    "ocr_record_refs",
}
_DECISION_KEYS = {
    "state_id",
    "decision_step_id",
    "history_event_step_ids",
    "candidate_event_step_ids",
    "current_equivalent_event_step_id",
    "current_observation_path",
    "current_observation_sha256",
    "candidate_event_post_states",
    "current_equivalence_witness",
    "content_witness_sha256",
    "current_expert_action_payload_included",
}
_RAW_STATE_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "run_contract_sha256",
    "worker",
    "state",
    "reference_teacher",
    "distance_rows",
    "operation_counts",
}
_STATE_KEYS = {
    "state_index",
    "role",
    "source_id",
    "decision_step_id",
    "state_id",
    "history_event_step_ids",
    "candidate_event_step_ids",
    "current_equivalent_event_step_id",
    "parent_member_name",
    "parent_member_sha256",
    "canonical_action_sha256",
    "teacher_target_sha256",
    "request_manifest_sha256",
    "slice_witness_sha256",
    "immutable_artifact_tree_sha256",
    "coalition_inputs",
}
_COALITION_INPUT_KEYS = {"coalition", "input_sha256"}
_DISTANCE_ROW_KEYS = {
    "coalition",
    "distance",
    "candidate_input_sha256",
    "teacher_forward_count",
    "kl_measurement_count",
    "scalar_host_transfer_count",
    "is_full_history_reference",
    "full_logit_tensor_host_transfer_count",
}


class VerifiedFresh16Transport:
    """Opaque byte-verified transport; creation does not decode JSON semantics."""

    __slots__ = ("_payload", "kind", "sha256", "size_bytes", "record_count", "_seal")

    def __init__(
        self,
        payload: bytes,
        *,
        kind: str,
        sha256: str,
        size_bytes: int,
        record_count: int,
        _seal: object,
    ) -> None:
        if _seal is not _VERIFICATION_SEAL:
            raise TypeError("fresh-16 transports must come from byte verification")
        self._payload = payload
        self.kind = kind
        self.sha256 = sha256
        self.size_bytes = size_bytes
        self.record_count = record_count
        self._seal = _seal


@dataclass(frozen=True)
class Fresh16TrajectorySlice:
    trajectories: tuple[Mapping[str, Any], ...]
    source_ids: tuple[str, ...]
    source_ids_sha256: str
    transport_sha256: str
    selected_row_indices: tuple[int, ...]
    semantic_decode_count: int
    candidate_occurrence_count: int


@dataclass(frozen=True)
class Fresh16LabelSlice:
    labels: tuple[LabelState, ...]
    source_ids: tuple[str, ...]
    source_ids_sha256: str
    transport_sha256: str
    selected_row_indices: tuple[int, ...]
    semantic_decode_count: int
    distance_value_decode_count: int
    run_contract_sha256: str


@dataclass(frozen=True)
class Fresh16SelectiveData:
    trajectories: Fresh16TrajectorySlice
    labels: Fresh16LabelSlice


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys drifted: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _jsonl_lines(payload: bytes, *, expected_count: int, label: str) -> tuple[bytes, ...]:
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be LF-terminated bytes")
    lines = tuple(payload[:-1].split(b"\n"))
    if len(lines) != expected_count or any(not line for line in lines):
        raise ValueError(f"{label} record count or blank-line inventory drifted")
    return lines


def _canonical_object(line: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            line.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=float,
            parse_int=int,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict canonical JSON") from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != line:
        raise ValueError(f"{label} is not strict canonical JSON")
    return value


def _roster(
    expected_source_ids: Sequence[str], expected_source_ids_sha256: str
) -> tuple[tuple[str, ...], str]:
    source_ids = tuple(expected_source_ids)
    digest = _require_sha256(expected_source_ids_sha256, "fresh-16 source digest")
    if (
        digest != FRESH_SOURCE_IDS_SHA256
        or len(source_ids) != FRESH_TRAJECTORY_COUNT
        or len(set(source_ids)) != FRESH_TRAJECTORY_COUNT
        or any(not isinstance(source_id, str) or not source_id for source_id in source_ids)
        or hashlib.sha256(canonical_json_bytes(list(source_ids))).hexdigest() != digest
    ):
        raise ValueError("fresh-16 source roster, order, or digest drifted")
    return source_ids, digest


def _verify_fresh16_jsonl_transport(
    payload: bytes,
    *,
    kind: Literal["expansion_trajectories_jsonl", "expansion_raw_states_jsonl"],
    expected_sha256: str,
    expected_size_bytes: int,
) -> VerifiedFresh16Transport:
    """Verify a frozen transport binding without decoding any JSON record."""
    if kind not in _TRANSPORT_COUNTS:
        raise ValueError("fresh-16 transport kind is not allowlisted")
    if not isinstance(payload, bytes):
        raise TypeError("fresh-16 transport payload must be bytes")
    digest = _require_sha256(expected_sha256, "fresh-16 transport SHA256")
    if type(expected_size_bytes) is not int or expected_size_bytes <= 0:
        raise ValueError("fresh-16 transport size must be a positive integer")
    if len(payload) != expected_size_bytes or hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("fresh-16 transport byte binding drifted")
    record_count = _TRANSPORT_COUNTS[kind]
    _jsonl_lines(payload, expected_count=record_count, label=kind)
    return VerifiedFresh16Transport(
        payload,
        kind=kind,
        sha256=digest,
        size_bytes=expected_size_bytes,
        record_count=record_count,
        _seal=_VERIFICATION_SEAL,
    )


def verify_fresh16_trajectory_transport(payload: bytes) -> VerifiedFresh16Transport:
    """Verify the one frozen derived trajectory transport used in production."""
    return _verify_fresh16_jsonl_transport(
        payload,
        kind=TRAJECTORY_TRANSPORT,
        expected_sha256=FRESH_TRAJECTORY_TRANSPORT_SHA256,
        expected_size_bytes=FRESH_TRAJECTORY_TRANSPORT_SIZE_BYTES,
    )


def verify_fresh16_raw_state_transport(payload: bytes) -> VerifiedFresh16Transport:
    """Verify the one frozen repaired raw-state member used in production."""
    return _verify_fresh16_jsonl_transport(
        payload,
        kind=RAW_STATE_TRANSPORT,
        expected_sha256=FRESH_RAW_STATE_TRANSPORT_SHA256,
        expected_size_bytes=FRESH_RAW_STATE_TRANSPORT_SIZE_BYTES,
    )


def _verified_lines(
    transport: VerifiedFresh16Transport,
    *,
    expected_kind: str,
) -> tuple[bytes, ...]:
    if (
        not isinstance(transport, VerifiedFresh16Transport)
        or transport._seal is not _VERIFICATION_SEAL
        or transport.kind != expected_kind
    ):
        raise TypeError("fresh-16 decoder requires its byte-verified transport kind")
    if (
        transport.record_count != _TRANSPORT_COUNTS[expected_kind]
        or len(transport._payload) != transport.size_bytes
        or hashlib.sha256(transport._payload).hexdigest() != transport.sha256
    ):
        raise ValueError("fresh-16 verified transport was mutated after verification")
    return _jsonl_lines(
        transport._payload,
        expected_count=_TRANSPORT_COUNTS[expected_kind],
        label=expected_kind,
    )


def _validate_trajectory(
    value: Mapping[str, Any], *, expected_source_id: str, row_index: int
) -> Mapping[str, Any]:
    _exact_keys(value, _TRAJECTORY_KEYS, f"fresh trajectory row {row_index}")
    if value.get("source_id") != expected_source_id or value.get("role") != FRESH_ROLE:
        raise ValueError("fresh trajectory identity, role, or source order drifted")
    instruction = value.get("instruction")
    if (
        not isinstance(instruction, str)
        or value.get("instruction_sha256")
        != hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    ):
        raise ValueError("fresh trajectory instruction witness drifted")
    events = value.get("events")
    decisions = value.get("decisions")
    if not isinstance(events, list) or not isinstance(decisions, list):
        raise ValueError("fresh trajectory events or decisions are missing")
    if [event.get("step_id") for event in events if isinstance(event, Mapping)] != [
        1,
        2,
        3,
        4,
        5,
    ]:
        raise ValueError("fresh trajectory event geometry drifted")
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("fresh trajectory event is not an object")
        _exact_keys(event, _EVENT_KEYS, "fresh trajectory event")
    decision_steps = [
        decision.get("decision_step_id")
        for decision in decisions
        if isinstance(decision, Mapping)
    ]
    if decision_steps != [
        4,
        5,
        6,
    ]:
        raise ValueError("fresh trajectory decision order drifted")
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("fresh trajectory decision is not an object")
        _exact_keys(decision, _DECISION_KEYS, "fresh trajectory decision")
        step = decision["decision_step_id"]
        candidate_ids = list(range(1, step - 1))
        if (
            decision.get("state_id")
            != f"{expected_source_id}:decision_step:{step:03d}"
            or decision.get("history_event_step_ids") != list(range(1, step))
            or decision.get("candidate_event_step_ids") != candidate_ids
            or len(candidate_ids) not in (2, 3, 4)
            or decision.get("current_equivalent_event_step_id") != step - 1
            or decision.get("current_observation_path")
            != events[step - 2].get("observation_after_path")
            or decision.get("current_expert_action_payload_included") is not False
        ):
            raise ValueError("fresh trajectory canonical 2/3/4 state geometry drifted")
    return value


def decode_fresh16_trajectories(
    transport: VerifiedFresh16Transport,
    *,
    expected_source_ids: Sequence[str],
    expected_source_ids_sha256: str,
) -> Fresh16TrajectorySlice:
    """Decode exactly expansion trajectory rows 48:64."""
    source_ids, source_digest = _roster(
        expected_source_ids, expected_source_ids_sha256
    )
    lines = _verified_lines(transport, expected_kind=TRAJECTORY_TRANSPORT)
    values = tuple(
        _validate_trajectory(
            _canonical_object(lines[row], label=f"fresh trajectory row {row}"),
            expected_source_id=source_ids[position],
            row_index=row,
        )
        for position, row in enumerate(FRESH_TRAJECTORY_ROWS)
    )
    state_ids = [
        decision["state_id"]
        for trajectory in values
        for decision in trajectory["decisions"]
    ]
    if len(state_ids) != FRESH_STATE_COUNT or len(set(state_ids)) != FRESH_STATE_COUNT:
        raise ValueError("fresh trajectory state inventory is not exactly 48 unique states")
    candidate_count = sum(
        len(decision["candidate_event_step_ids"])
        for trajectory in values
        for decision in trajectory["decisions"]
    )
    if candidate_count != 144:
        raise ValueError("fresh trajectory candidate denominator drifted")
    return Fresh16TrajectorySlice(
        trajectories=values,
        source_ids=source_ids,
        source_ids_sha256=source_digest,
        transport_sha256=transport.sha256,
        selected_row_indices=FRESH_TRAJECTORY_ROWS,
        semantic_decode_count=len(values),
        candidate_occurrence_count=candidate_count,
    )


def _coalitions(event_ids: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        coalition
        for size in range(len(event_ids) + 1)
        for coalition in itertools.combinations(event_ids, size)
    )


def _validate_raw_state(
    value: Mapping[str, Any],
    *,
    expected_source_id: str,
    row_index: int,
) -> tuple[LabelState, str, int]:
    _exact_keys(value, _RAW_STATE_KEYS, f"fresh raw-state row {row_index}")
    run_contract_sha256 = _require_sha256(
        value.get("run_contract_sha256"), "fresh raw-state run-contract SHA256"
    )
    state = value.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("fresh raw-state projection is missing")
    _exact_keys(state, _STATE_KEYS, f"fresh state projection {row_index}")
    step = 4 + (row_index - FRESH_RAW_STATE_ROWS[0]) % 3
    event_ids = tuple(range(1, step - 1))
    worker_id = "even" if row_index % 2 == 0 else "odd"
    if (
        state.get("state_index") != row_index
        or state.get("role") != FRESH_ROLE
        or state.get("source_id") != expected_source_id
        or state.get("decision_step_id") != step
        or state.get("state_id")
        != f"{expected_source_id}:decision_step:{step:03d}"
        or state.get("history_event_step_ids") != list(range(1, step))
        or state.get("candidate_event_step_ids") != list(event_ids)
        or len(event_ids) not in (2, 3, 4)
        or state.get("current_equivalent_event_step_id") != step - 1
        or state.get("parent_member_name")
        != f"workers/{worker_id}/states/{row_index:03d}.json"
    ):
        raise ValueError("fresh raw-state role, order, or canonical 2/3/4 geometry drifted")
    for key in (
        "parent_member_sha256",
        "canonical_action_sha256",
        "teacher_target_sha256",
        "request_manifest_sha256",
        "slice_witness_sha256",
        "immutable_artifact_tree_sha256",
    ):
        _require_sha256(state.get(key), f"fresh state {key}")
    coalition_inputs = state.get("coalition_inputs")
    expected_coalitions = _coalitions(event_ids)
    if not isinstance(coalition_inputs, list) or len(coalition_inputs) != len(
        expected_coalitions
    ):
        raise ValueError("fresh state coalition-input denominator drifted")
    input_witnesses: dict[tuple[int, ...], str] = {}
    for position, (raw, coalition) in enumerate(
        zip(coalition_inputs, expected_coalitions, strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("fresh state coalition-input witness is malformed")
        _exact_keys(raw, _COALITION_INPUT_KEYS, "fresh coalition-input witness")
        if raw.get("coalition") != list(coalition):
            raise ValueError("fresh coalition-input order drifted")
        input_witnesses[coalition] = _require_sha256(
            raw.get("input_sha256"), f"fresh coalition-input {position} SHA256"
        )
    rows = value.get("distance_rows")
    if not isinstance(rows, list) or len(rows) != len(expected_coalitions):
        raise ValueError("fresh distance-row denominator drifted")
    for position, (row, coalition) in enumerate(zip(rows, expected_coalitions, strict=True)):
        if not isinstance(row, Mapping):
            raise ValueError("fresh distance row is malformed")
        _exact_keys(row, _DISTANCE_ROW_KEYS, "fresh distance row")
        distance = row.get("distance")
        if (
            row.get("coalition") != list(coalition)
            or row.get("candidate_input_sha256") != input_witnesses[coalition]
            or not isinstance(distance, (int, float))
            or isinstance(distance, bool)
            or not math.isfinite(float(distance))
            or float(distance) < 0.0
        ):
            raise ValueError(f"fresh distance row {position} geometry or value drifted")
    if not isinstance(value.get("reference_teacher"), Mapping) or not isinstance(
        value.get("operation_counts"), Mapping
    ):
        raise ValueError("fresh raw-state witness envelopes are malformed")
    label = label_state_from_restoration_record(value, record_schema="expansion")
    full = label.table.distance(label.table.event_ids)
    if full != 0.0 or math.copysign(1.0, full) < 0.0:
        raise ValueError("fresh full-history distance must be canonical +0")
    return label, run_contract_sha256, len(rows)


def decode_fresh16_labels(
    transport: VerifiedFresh16Transport,
    *,
    expected_source_ids: Sequence[str],
    expected_source_ids_sha256: str,
) -> Fresh16LabelSlice:
    """Decode exactly repaired raw-state rows 144:192."""
    source_ids, source_digest = _roster(
        expected_source_ids, expected_source_ids_sha256
    )
    lines = _verified_lines(transport, expected_kind=RAW_STATE_TRANSPORT)
    labels: list[LabelState] = []
    run_hashes: set[str] = set()
    distance_count = 0
    for local_index, row in enumerate(FRESH_RAW_STATE_ROWS):
        label, run_hash, row_count = _validate_raw_state(
            _canonical_object(lines[row], label=f"fresh raw-state row {row}"),
            expected_source_id=source_ids[local_index // 3],
            row_index=row,
        )
        labels.append(label)
        run_hashes.add(run_hash)
        distance_count += row_count
    if (
        len(labels) != FRESH_STATE_COUNT
        or len({label.state_id for label in labels}) != FRESH_STATE_COUNT
    ):
        raise ValueError("fresh label inventory is not exactly 48 unique states")
    if run_hashes != {FRESH_LABEL_RUN_CONTRACT_SHA256} or distance_count != 448:
        raise ValueError("fresh raw-state run binding or distance denominator drifted")
    observed_sources = tuple(dict.fromkeys(label.source_id for label in labels))
    if observed_sources != source_ids:
        raise ValueError("fresh label source roster or order drifted")
    return Fresh16LabelSlice(
        labels=tuple(labels),
        source_ids=source_ids,
        source_ids_sha256=source_digest,
        transport_sha256=transport.sha256,
        selected_row_indices=FRESH_RAW_STATE_ROWS,
        semantic_decode_count=len(labels),
        distance_value_decode_count=distance_count,
        run_contract_sha256=next(iter(run_hashes)),
    )


def load_fresh16_selective_data(
    trajectory_transport: VerifiedFresh16Transport,
    raw_state_transport: VerifiedFresh16Transport,
    *,
    expected_source_ids: Sequence[str],
    expected_source_ids_sha256: str,
) -> Fresh16SelectiveData:
    """Load the two independently verified transports and enforce their join roster."""
    trajectories = decode_fresh16_trajectories(
        trajectory_transport,
        expected_source_ids=expected_source_ids,
        expected_source_ids_sha256=expected_source_ids_sha256,
    )
    labels = decode_fresh16_labels(
        raw_state_transport,
        expected_source_ids=expected_source_ids,
        expected_source_ids_sha256=expected_source_ids_sha256,
    )
    trajectory_keys = tuple(
        (trajectory["source_id"], decision["state_id"])
        for trajectory in trajectories.trajectories
        for decision in trajectory["decisions"]
    )
    label_keys = tuple((label.source_id, label.state_id) for label in labels.labels)
    if trajectory_keys != label_keys:
        raise ValueError("fresh trajectory/label state join order drifted")
    return Fresh16SelectiveData(trajectories=trajectories, labels=labels)


__all__ = [
    "FRESH_LABEL_RUN_CONTRACT_SHA256",
    "FRESH_RAW_STATE_ROWS",
    "FRESH_RAW_STATE_TRANSPORT_SHA256",
    "FRESH_RAW_STATE_TRANSPORT_SIZE_BYTES",
    "FRESH_ROLE",
    "FRESH_STATE_COUNT",
    "FRESH_SOURCE_IDS_SHA256",
    "FRESH_TRAJECTORY_COUNT",
    "FRESH_TRAJECTORY_ROWS",
    "FRESH_TRAJECTORY_TRANSPORT_SHA256",
    "FRESH_TRAJECTORY_TRANSPORT_SIZE_BYTES",
    "RAW_STATE_TRANSPORT",
    "TRAJECTORY_TRANSPORT",
    "Fresh16LabelSlice",
    "Fresh16SelectiveData",
    "Fresh16TrajectorySlice",
    "VerifiedFresh16Transport",
    "decode_fresh16_labels",
    "decode_fresh16_trajectories",
    "load_fresh16_selective_data",
    "verify_fresh16_raw_state_transport",
    "verify_fresh16_trajectory_transport",
]
