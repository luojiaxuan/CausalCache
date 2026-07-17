"""Strict parent extraction for restoration-v2.2 expansion exact labels."""

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
from causalcache.restoration_v2_2_expansion_substrate_artifact import (
    EXPECTED_STATE_COUNT,
    PASS_OUTCOME,
    STATE_OUTCOME_VALID,
    ExpansionSubstrateEvidence,
    strict_json_object_bytes,
)


EXPANSION_LABEL_ROLES = (
    "gate_train_expansion",
    "gate_development_expansion",
)
_STATE_MEMBER = re.compile(
    r"workers/(?P<worker>even|odd)/states/(?P<index>[0-9]{3})\.json"
)
_PROJECTION_KEYS = {
    "state_index",
    "role",
    "source_id",
    "state_id",
    "decision_step_id",
    "history_event_step_ids",
    "candidate_event_step_ids",
    "current_equivalent_event_step_id",
}
_VALID_MEASUREMENT_OUTCOME = "VALID_V2_1_FULL_45_SUBSTRATE_STATE"


@dataclass(frozen=True)
class V22ExpansionLabelParentState:
    """One immutable expansion state and its substrate-generated teacher action."""

    index: int
    role: str
    source_id: str
    decision_step_id: int
    state_id: str
    history_event_step_ids: tuple[int, ...]
    candidate_event_step_ids: tuple[int, ...]
    current_equivalent_event_step_id: int
    worker_id: str
    member_name: str
    member_sha256: str
    immutable_artifact_tree_sha256: str
    request_manifest_sha256: str
    slice_witness_sha256: str
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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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
        if (
            type(text_value) is not str
            or unicodedata.normalize("NFKC", text_value) != text_value
        ):
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


def _action_sha256(action: GUIOwlV2Action) -> str:
    return hashlib.sha256(_canonical_json_bytes(action.arguments())).hexdigest()


def _state_members(files: Mapping[str, bytes]) -> dict[int, tuple[str, str, bytes]]:
    observed: dict[int, tuple[str, str, bytes]] = {}
    for member_name, payload in files.items():
        if not isinstance(member_name, str):
            raise ValueError("expansion evidence member names must be strings")
        is_state_like = member_name.startswith("workers/") and "/states/" in member_name
        match = _STATE_MEMBER.fullmatch(member_name)
        if match is None:
            if is_state_like:
                raise ValueError("expansion evidence has a malformed state member path")
            continue
        if not isinstance(payload, bytes):
            raise ValueError("expansion state members must contain bytes")
        index = int(match.group("index"))
        if index in observed:
            raise ValueError(f"expansion evidence has duplicate state index {index}")
        observed[index] = (match.group("worker"), member_name, payload)
    expected = set(range(EXPECTED_STATE_COUNT))
    if set(observed) != expected:
        raise ValueError("expansion parent evidence state inventory is not complete 192")
    return observed


def _artifact_state_projection(artifact: Any, *, index: int) -> dict[str, Any]:
    states = getattr(artifact, "states", None)
    if not isinstance(states, tuple) or len(states) != EXPECTED_STATE_COUNT:
        raise ValueError("immutable expansion artifact must contain exactly 192 states")
    state = states[index]
    decision = _mapping(getattr(state, "decision", None), f"artifact state {index} decision")
    return {
        "state_index": getattr(state, "index", None),
        "role": getattr(state, "role", None),
        "source_id": getattr(state, "source_id", None),
        "state_id": getattr(state, "state_id", None),
        "decision_step_id": getattr(state, "decision_step_id", None),
        "history_event_step_ids": decision.get("history_event_step_ids"),
        "candidate_event_step_ids": decision.get("candidate_event_step_ids"),
        "current_equivalent_event_step_id": decision.get(
            "current_equivalent_event_step_id"
        ),
    }


def _validate_artifact(artifact: Any) -> str:
    # note (luojiaxuan): The validated loader is the immutable-manifest boundary;
    # requiring its public result type prevents a hand-written legacy 45-state
    # manifest from being accepted merely because it has similarly named fields.
    from scripts.run_restoration_v2_2_expansion_substrate import ExpansionArtifact

    if not isinstance(artifact, ExpansionArtifact):
        raise TypeError("artifact must be a validated immutable ExpansionArtifact")
    validation = _mapping(artifact.validation, "immutable artifact validation")
    if (
        validation.get("outcome")
        != "PASSED_GUIODYSSEY_RESTORATION_V2_EXPANSION_VALIDATION"
        or validation.get("counts", {}).get("state_count") != EXPECTED_STATE_COUNT
        or validation.get("per_decision_view_current_expert_action_payload_included")
        is not False
        or validation.get("consumer_must_slice_events_by_history_event_step_ids")
        is not True
    ):
        raise ValueError("immutable expansion artifact validation is not formal and safe")
    tree = _mapping(artifact.tree, "immutable artifact tree")
    tree_sha = tree.get("artifact_tree_sha256")
    if not isinstance(tree_sha, str) or re.fullmatch(r"[0-9a-f]{64}", tree_sha) is None:
        raise ValueError("immutable expansion artifact tree SHA256 is invalid")
    if len(artifact.states) != EXPECTED_STATE_COUNT:
        raise ValueError("immutable expansion artifact must contain exactly 192 states")
    return tree_sha


def _validate_projection(value: Any, *, index: int) -> dict[str, Any]:
    projection = dict(_mapping(value, f"expansion parent projection {index}"))
    if set(projection) != _PROJECTION_KEYS:
        raise ValueError(f"expansion parent projection {index} keys drifted")
    step = projection.get("decision_step_id")
    role = projection.get("role")
    source_id = projection.get("source_id")
    if (
        projection.get("state_index") != index
        or role not in EXPANSION_LABEL_ROLES
        or role
        != ("gate_train_expansion" if index < 144 else "gate_development_expansion")
        or type(step) is not int
        or step not in {4, 5, 6}
        or step != 4 + index % 3
        or not isinstance(source_id, str)
        or not source_id
        or projection.get("state_id") != f"{source_id}:decision_step:{step:03d}"
        or projection.get("history_event_step_ids") != list(range(1, step))
        or projection.get("candidate_event_step_ids") != list(range(1, step - 1))
        or projection.get("current_equivalent_event_step_id") != step - 1
    ):
        raise ValueError(f"expansion parent projection {index} identity/geometry drifted")
    return projection


def extract_expansion_label_parent_states(
    evidence: ExpansionSubstrateEvidence,
    artifact: Any,
) -> tuple[V22ExpansionLabelParentState, ...]:
    """Bind all 192 PASS substrate teacher actions to the immutable manifest."""

    if not isinstance(evidence, ExpansionSubstrateEvidence):
        raise TypeError("evidence must be validated ExpansionSubstrateEvidence")
    if (
        evidence.outcome != PASS_OUTCOME
        or evidence.fixed_state_denominator != EXPECTED_STATE_COUNT
        or evidence.attempted_state_count != EXPECTED_STATE_COUNT
        or evidence.completed_state_count != EXPECTED_STATE_COUNT
        or evidence.valid_state_count != EXPECTED_STATE_COUNT
        or evidence.failed_state_count != 0
        or not isinstance(evidence.gate_summary, Mapping)
        or evidence.gate_summary.get("gate_passed") is not True
    ):
        raise ValueError("expansion label parent must be the complete PASS/192 evidence")
    artifact_tree_sha = _validate_artifact(artifact)
    projections = _sequence(evidence.run_contract.get("states"), "parent projections")
    if len(projections) != EXPECTED_STATE_COUNT:
        raise ValueError("expansion label parent must bind exactly 192 projections")
    members = _state_members(evidence.files)

    records: list[V22ExpansionLabelParentState] = []
    seen_state_ids: set[str] = set()
    for index in range(EXPECTED_STATE_COUNT):
        projection = _validate_projection(projections[index], index=index)
        if _artifact_state_projection(artifact, index=index) != projection:
            raise ValueError(f"parent projection {index} differs from immutable manifest")
        state_id = str(projection["state_id"])
        if state_id in seen_state_ids:
            raise ValueError(f"expansion parent has duplicate state ID {state_id}")
        seen_state_ids.add(state_id)

        worker_id, member_name, payload = members[index]
        expected_worker = "even" if index % 2 == 0 else "odd"
        expected_member = f"workers/{expected_worker}/states/{index:03d}.json"
        if worker_id != expected_worker or member_name != expected_member:
            raise ValueError(f"expansion parent state {index} is in the wrong worker shard")
        envelope = strict_json_object_bytes(payload, label=f"expansion parent state {index}")
        if (
            envelope.get("state") != projection
            or envelope.get("outcome") != STATE_OUTCOME_VALID
            or envelope.get("worker", {}).get("worker_id") != worker_id
        ):
            raise ValueError(f"expansion parent state {index} is not a valid state")
        wrapper = _mapping(
            envelope.get("measurement_kernel"),
            f"expansion parent state {index} kernel wrapper",
        )
        inner = _mapping(wrapper.get("record"), f"expansion parent state {index} kernel")
        request_manifest = _mapping(
            inner.get("request_manifest"),
            f"expansion parent state {index} request manifest",
        )
        slice_witness = _mapping(
            inner.get("slice_witness"),
            f"expansion parent state {index} slice witness",
        )
        history = projection["history_event_step_ids"]
        request_manifest_sha256 = hashlib.sha256(
            _canonical_json_bytes(request_manifest)
        ).hexdigest()
        if (
            inner.get("state") != projection
            or inner.get("outcome") != _VALID_MEASUREMENT_OUTCOME
            or inner.get("parse_success") is not True
            or inner.get("finite_logit_distances") is not True
            or inner.get("repeat_canonical_action_agreement") is not True
            or inner.get("failure") is not None
            or request_manifest.get("state_id") != state_id
            or request_manifest.get("decision_step_id") != projection["decision_step_id"]
            or request_manifest.get("included_event_step_ids") != history
            or request_manifest.get("current_or_future_event_action_exposure_count") != 0
            or slice_witness.get("included_event_step_ids") != history
            or slice_witness.get("current_or_future_event_action_exposure_count") != 0
            or slice_witness
            != {**dict(request_manifest), "slice_witness_sha256": request_manifest_sha256}
        ):
            raise ValueError(f"expansion parent state {index} kernel/request identity drifted")

        generations = _sequence(
            inner.get("native_generations"),
            f"expansion parent state {index} generations",
        )
        if len(generations) != 2:
            raise ValueError(f"expansion parent state {index} needs two generations")
        actions: list[GUIOwlV2Action] = []
        for repeat_index, raw_generation in enumerate(generations, start=1):
            generation = _mapping(
                raw_generation,
                f"expansion parent state {index} generation {repeat_index}",
            )
            if generation.get("repeat_index") != repeat_index:
                raise ValueError(f"expansion parent state {index} repeat order drifted")
            actions.append(
                _strict_action(
                    generation.get("canonical_action"),
                    f"expansion parent state {index} generation {repeat_index} action",
                )
            )
        if actions[0] != actions[1]:
            raise ValueError(f"expansion parent state {index} generation actions drifted")
        record_action = _strict_action(
            inner.get("canonical_action"),
            f"expansion parent state {index} record action",
        )
        if record_action != actions[0]:
            raise ValueError(f"expansion parent state {index} record action drifted")

        records.append(
            V22ExpansionLabelParentState(
                index=index,
                role=str(projection["role"]),
                source_id=str(projection["source_id"]),
                decision_step_id=int(projection["decision_step_id"]),
                state_id=state_id,
                history_event_step_ids=tuple(history),
                candidate_event_step_ids=tuple(
                    projection["candidate_event_step_ids"]
                ),
                current_equivalent_event_step_id=int(
                    projection["current_equivalent_event_step_id"]
                ),
                worker_id=worker_id,
                member_name=member_name,
                member_sha256=hashlib.sha256(payload).hexdigest(),
                immutable_artifact_tree_sha256=artifact_tree_sha,
                request_manifest_sha256=request_manifest_sha256,
                slice_witness_sha256=request_manifest_sha256,
                canonical_action=record_action,
                canonical_action_sha256=_action_sha256(record_action),
            )
        )
    return tuple(records)


__all__ = [
    "EXPANSION_LABEL_ROLES",
    "V22ExpansionLabelParentState",
    "extract_expansion_label_parent_states",
]
