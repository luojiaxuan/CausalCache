"""Strict parent-state extraction for restoration-v2.2 label generation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.policy.gui_owl_v2 import (
    CANONICAL_GUI_OWL_V2_ACTIONS,
    GUI_OWL_V2_PARAMETERS_BY_ACTION,
    GUI_OWL_V2_SYSTEM_BUTTONS,
    GUIOwlV2Action,
)
from causalcache.restoration_v2_2_eager_artifact import (
    EXPECTED_STATE_COUNT,
    PASS_OUTCOME,
    STATE_OUTCOME_VALID,
    V22Evidence,
    strict_json_object_bytes,
)


_STATE_MEMBER = re.compile(r"workers/(?P<worker>even|odd)/states/(?P<index>[0-9]{3})\.json")
_PROJECTION_KEYS = {
    "index",
    "role",
    "trajectory_id",
    "decision_step_id",
    "state_id",
    "candidate_event_step_ids",
}
_VALID_MEASUREMENT_OUTCOME = "VALID_V2_1_FULL_45_SUBSTRATE_STATE"


@dataclass(frozen=True)
class V22LabelParentState:
    """Immutable teacher target and projection for one validated parent state."""

    index: int
    role: str
    trajectory_id: str
    decision_step_id: int
    state_id: str
    candidate_event_step_ids: tuple[int, ...]
    worker_id: str
    member_name: str
    member_sha256: str
    canonical_action: GUIOwlV2Action
    canonical_action_sha256: str


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _coordinate(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two JSON integers")
    if any(type(item) is not int for item in value):
        raise ValueError(f"{name} must contain exactly two JSON integers")
    if any(item < 0 or item > 999 for item in value):
        raise ValueError(f"{name} values must be in [0, 999]")
    return value[0], value[1]


def _strict_action(value: Any, name: str) -> GUIOwlV2Action:
    mapping = _mapping(value, name)
    action_name = mapping.get("action")
    if type(action_name) is not str or action_name not in CANONICAL_GUI_OWL_V2_ACTIONS:
        raise ValueError(f"{name} must use one canonical GUI-Owl v2 action")
    expected_keys = {"action", *GUI_OWL_V2_PARAMETERS_BY_ACTION[action_name]}
    if set(mapping) != expected_keys:
        raise ValueError(f"{name} arguments differ from the canonical action schema")

    coordinate = None
    coordinate2 = None
    text = None
    button = None
    status = None
    if "coordinate" in mapping:
        coordinate = _coordinate(mapping["coordinate"], f"{name}.coordinate")
    if "coordinate2" in mapping:
        coordinate2 = _coordinate(mapping["coordinate2"], f"{name}.coordinate2")
    if "text" in mapping:
        text_value = mapping["text"]
        if type(text_value) is not str or unicodedata.normalize("NFKC", text_value) != text_value:
            raise ValueError(f"{name}.text must be an NFKC JSON string")
        text = text_value
    if "button" in mapping:
        button_value = mapping["button"]
        if type(button_value) is not str or button_value not in GUI_OWL_V2_SYSTEM_BUTTONS:
            raise ValueError(f"{name}.button is not a canonical system button")
        button = button_value
    if "status" in mapping:
        if mapping["status"] != "success":
            raise ValueError(f"{name}.status must be success")
        status = "success"

    action = GUIOwlV2Action(
        action=action_name,
        coordinate=coordinate,
        coordinate2=coordinate2,
        text=text,
        button=button,
        status=status,
    )
    if action.arguments() != dict(mapping):
        raise ValueError(f"{name} is not a canonical GUI-Owl v2 action mapping")
    return action


def _canonical_action_sha256(action: GUIOwlV2Action) -> str:
    payload = json.dumps(
        action.arguments(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_projection(value: Any, *, index: int) -> Mapping[str, Any]:
    projection = _mapping(value, f"parent projection {index}")
    if set(projection) != _PROJECTION_KEYS:
        raise ValueError(f"parent projection {index} keys drifted")
    if type(projection.get("index")) is not int or projection["index"] != index:
        raise ValueError(f"parent projection {index} index drifted")
    expected_role = "v2_label_train" if index < 30 else "v2_development"
    if projection.get("role") != expected_role:
        raise ValueError(f"parent projection {index} role drifted")
    trajectory_id = projection.get("trajectory_id")
    decision_step_id = projection.get("decision_step_id")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError(f"parent projection {index} trajectory ID is invalid")
    if type(decision_step_id) is not int or decision_step_id != 4 + index % 3:
        raise ValueError(f"parent projection {index} decision step drifted")
    expected_state_id = f"{trajectory_id}:decision_step:{decision_step_id:03d}"
    if projection.get("state_id") != expected_state_id:
        raise ValueError(f"parent projection {index} state ID drifted")
    candidate_values = projection.get("candidate_event_step_ids")
    if not isinstance(candidate_values, list) or any(
        type(event_id) is not int for event_id in candidate_values
    ):
        raise ValueError(f"parent projection {index} candidate event IDs are invalid")
    expected_candidates = list(range(1, decision_step_id - 1))
    if candidate_values != expected_candidates:
        raise ValueError(f"parent projection {index} candidate event IDs drifted")
    return projection


def _state_members(files: Mapping[str, bytes]) -> dict[int, tuple[str, str, bytes]]:
    observed: dict[int, tuple[str, str, bytes]] = {}
    for member_name, payload in files.items():
        if not isinstance(member_name, str):
            raise ValueError("parent evidence member names must be strings")
        is_state_like = member_name.startswith("workers/") and "/states/" in member_name
        match = _STATE_MEMBER.fullmatch(member_name)
        if match is None:
            if is_state_like:
                raise ValueError("parent evidence has a malformed state member path")
            continue
        if not isinstance(payload, bytes):
            raise ValueError("parent state members must contain bytes")
        index = int(match.group("index"))
        if index in observed:
            raise ValueError(f"parent evidence has duplicate state index {index}")
        observed[index] = (match.group("worker"), member_name, payload)
    expected = set(range(EXPECTED_STATE_COUNT))
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise ValueError(
            f"parent evidence state inventory drifted; missing={missing}, extra={extra}"
        )
    return observed


def extract_v22_label_parent_states(
    evidence: V22Evidence,
) -> tuple[V22LabelParentState, ...]:
    """Extract all 45 immutable label-parent records from validated v2.2 evidence."""

    if not isinstance(evidence, V22Evidence):
        raise TypeError("evidence must be a validated V22Evidence")
    if (
        evidence.outcome != PASS_OUTCOME
        or evidence.completed_state_count != EXPECTED_STATE_COUNT
        or evidence.attempted_state_count != EXPECTED_STATE_COUNT
    ):
        raise ValueError("label parent must be the complete PASS v2.2 evidence")
    projections = _sequence(evidence.run_contract.get("states"), "parent projections")
    if len(projections) != EXPECTED_STATE_COUNT:
        raise ValueError("label parent must bind exactly 45 projections")
    members = _state_members(evidence.files)

    records: list[V22LabelParentState] = []
    observed_state_ids: set[str] = set()
    for index in range(EXPECTED_STATE_COUNT):
        projection = _validate_projection(projections[index], index=index)
        state_id = str(projection["state_id"])
        if state_id in observed_state_ids:
            raise ValueError(f"parent evidence has duplicate state ID {state_id}")
        observed_state_ids.add(state_id)

        worker_id, member_name, payload = members[index]
        expected_worker = "even" if index % 2 == 0 else "odd"
        expected_member = f"workers/{expected_worker}/states/{index:03d}.json"
        if worker_id != expected_worker or member_name != expected_member:
            raise ValueError(f"parent state {index} is in the wrong worker shard")
        envelope = strict_json_object_bytes(payload, label=f"parent state {index}")
        envelope_state = _mapping(envelope.get("state"), f"parent state {index} projection")
        if envelope_state != projection:
            raise ValueError(f"parent state {index} differs from its run-contract projection")
        worker = _mapping(envelope.get("worker"), f"parent state {index} worker")
        if worker.get("worker_id") != worker_id or envelope.get("outcome") != STATE_OUTCOME_VALID:
            raise ValueError(f"parent state {index} is not a valid v2.2 state")

        kernel = _mapping(envelope.get("measurement_kernel"), f"parent state {index} kernel")
        inner = _mapping(kernel.get("record"), f"parent state {index} kernel record")
        if (
            inner.get("state") != projection
            or inner.get("outcome") != _VALID_MEASUREMENT_OUTCOME
            or inner.get("repeat_canonical_action_agreement") is not True
        ):
            raise ValueError(f"parent state {index} measurement kernel is not valid")
        generations = _sequence(
            inner.get("native_generations"),
            f"parent state {index} native generations",
        )
        if len(generations) != 2:
            raise ValueError(f"parent state {index} must contain exactly two generations")
        generation_actions: list[GUIOwlV2Action] = []
        for repeat_index, generation_value in enumerate(generations, start=1):
            generation = _mapping(
                generation_value,
                f"parent state {index} generation {repeat_index}",
            )
            if generation.get("repeat_index") != repeat_index:
                raise ValueError(f"parent state {index} generation repeat order drifted")
            generation_actions.append(
                _strict_action(
                    generation.get("canonical_action"),
                    f"parent state {index} generation {repeat_index} action",
                )
            )
        if generation_actions[0] != generation_actions[1]:
            raise ValueError(f"parent state {index} generation canonical actions drifted")
        record_action = _strict_action(
            inner.get("canonical_action"),
            f"parent state {index} record action",
        )
        if record_action != generation_actions[0]:
            raise ValueError(f"parent state {index} record canonical action drifted")

        records.append(
            V22LabelParentState(
                index=index,
                role=str(projection["role"]),
                trajectory_id=str(projection["trajectory_id"]),
                decision_step_id=int(projection["decision_step_id"]),
                state_id=state_id,
                candidate_event_step_ids=tuple(projection["candidate_event_step_ids"]),
                worker_id=worker_id,
                member_name=member_name,
                member_sha256=hashlib.sha256(payload).hexdigest(),
                canonical_action=record_action,
                canonical_action_sha256=_canonical_action_sha256(record_action),
            )
        )
    return tuple(records)
