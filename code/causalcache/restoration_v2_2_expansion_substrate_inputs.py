"""Policy-blind decision-view input guards for the expansion substrate."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from causalcache.data.guiodyssey_restoration_v2_expansion import (
    canonical_json_bytes,
    decision_view_events,
    sha256_bytes,
)
from causalcache.low_fidelity_v2 import (
    LowFidelityEventV2,
    serialize_low_fidelity_v2,
)


ACTION_FIELDS = ("executed_action", "source_tool_call")


def build_decision_view_input(
    trajectory: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, Any]:
    source_id = trajectory.get("source_id")
    state_id = decision.get("state_id")
    step = decision.get("decision_step_id")
    events = trajectory.get("events")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("decision-view trajectory source_id is invalid")
    if not isinstance(state_id, str) or not state_id:
        raise ValueError("decision-view state_id is invalid")
    if type(step) is not int:
        raise ValueError("decision-view step is invalid")
    if state_id != f"{source_id}:decision_step:{step:03d}":
        raise ValueError("decision-view state_id does not bind source_id and step")
    if not isinstance(events, list):
        raise ValueError("decision-view trajectory events must be an array")
    view = decision_view_events(events, decision)
    included = [int(event["step_id"]) for event in view]
    expected = list(decision["history_event_step_ids"])
    if included != expected or any(event_step >= step for event_step in included):
        raise ValueError("decision-view slice exposed a current or future event")
    event_payload_sha256 = sha256_bytes(canonical_json_bytes(list(view)))
    witness = {
        "source_id": source_id,
        "state_id": state_id,
        "decision_step_id": step,
        "included_event_step_ids": included,
        "included_event_payload_sha256": event_payload_sha256,
        "current_or_future_event_action_exposure_count": 0,
    }
    return {
        "source_id": source_id,
        "state_id": state_id,
        "decision_step_id": step,
        "events": view,
        "request_manifest": witness,
        "slice_witness_sha256": sha256_bytes(canonical_json_bytes(witness)),
    }


def validate_request_manifest(
    request_manifest: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    included_events: Sequence[Mapping[str, Any]],
) -> None:
    expected_keys = {
        "source_id",
        "state_id",
        "decision_step_id",
        "included_event_step_ids",
        "included_event_payload_sha256",
        "current_or_future_event_action_exposure_count",
    }
    if set(request_manifest) != expected_keys:
        raise ValueError("raw request manifest schema drifted")
    expected = list(decision.get("history_event_step_ids", []))
    step = decision.get("decision_step_id")
    state_id = decision.get("state_id")
    source_id = request_manifest.get("source_id")
    if (
        not isinstance(source_id, str)
        or not isinstance(state_id, str)
        or type(step) is not int
        or state_id != f"{source_id}:decision_step:{step:03d}"
        or request_manifest.get("state_id") != state_id
        or request_manifest.get("decision_step_id") != step
    ):
        raise ValueError("raw request manifest state identity drifted")
    if isinstance(included_events, (str, bytes, bytearray, Mapping)) or not isinstance(
        included_events, Sequence
    ):
        raise TypeError("actual included events must be a sequence of objects")
    actual_events = list(included_events)
    if any(not isinstance(event, Mapping) for event in actual_events):
        raise TypeError("actual included events must be a sequence of objects")
    actual_steps = [event.get("step_id") for event in actual_events]
    if any(type(event_step) is not int for event_step in actual_steps):
        raise ValueError("actual included event steps must be integers")
    if actual_steps != expected or request_manifest.get(
        "included_event_step_ids"
    ) != actual_steps:
        raise ValueError("raw request manifest does not record the exact history slice")
    if any(event_step >= step for event_step in actual_steps):
        raise ValueError("raw request manifest includes current or future event steps")
    actual_payload_sha256 = sha256_bytes(canonical_json_bytes(actual_events))
    if request_manifest.get("included_event_payload_sha256") != actual_payload_sha256:
        raise ValueError("raw request manifest event payload SHA256 drifted")
    if request_manifest.get("current_or_future_event_action_exposure_count") != 0:
        raise ValueError("raw request manifest reports action leakage")


def _mutated_action_trajectory(
    trajectory: Mapping[str, Any], *, event_step: int, canary_id: str
) -> dict[str, Any]:
    mutated = copy.deepcopy(dict(trajectory))
    events = mutated.get("events")
    if not isinstance(events, list):
        raise ValueError("canary trajectory events must be an array")
    matches = [event for event in events if event.get("step_id") == event_step]
    if len(matches) != 1:
        raise ValueError("canary event step must identify exactly one event")
    event = matches[0]
    for field in ACTION_FIELDS:
        if field not in event:
            raise ValueError(f"canary event is missing action field: {field}")
        event[field] = {"causalcache_canary": canary_id, "field": field}
    if "low_fidelity_v2" in event:
        low_fidelity = dict(event["low_fidelity_v2"])
        low_fidelity["action_type"] = "answer"
        low_fidelity["action_argument"] = f"causalcache canary {canary_id}"
        validated = LowFidelityEventV2.from_mapping(low_fidelity)
        serialized = serialize_low_fidelity_v2(validated)
        event["low_fidelity_v2"] = validated.to_ordered_dict()
        event["low_fidelity_v2_serialized"] = serialized.decode("utf-8")
        event["low_fidelity_v2_sha256"] = sha256_bytes(serialized)
    return mutated


def validate_processor_byte_canary(
    trajectory: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    serialize_request: Callable[[Mapping[str, Any], Mapping[str, Any]], bytes],
) -> dict[str, Any]:
    base = serialize_request(trajectory, decision)
    if not isinstance(base, bytes):
        raise TypeError("processor canary serializer must return bytes")
    history = decision.get("history_event_step_ids")
    step = decision.get("decision_step_id")
    events = trajectory.get("events")
    if not isinstance(history, list) or not history or type(step) is not int:
        raise ValueError("processor canary decision geometry is invalid")
    if not isinstance(events, Sequence):
        raise ValueError("processor canary trajectory events are invalid")

    included_step = int(history[-1])
    included_mutation = _mutated_action_trajectory(
        trajectory,
        event_step=included_step,
        canary_id="included-history-action",
    )
    included_bytes = serialize_request(included_mutation, decision)
    if not isinstance(included_bytes, bytes):
        raise TypeError("processor canary serializer must return bytes")
    if included_bytes == base:
        raise ValueError("included history action canary did not change prompt bytes")

    excluded_steps = sorted(
        int(event["step_id"])
        for event in events
        if type(event.get("step_id")) is int and int(event["step_id"]) >= step
    )
    excluded_results: list[dict[str, Any]] = []
    for excluded_step in excluded_steps:
        excluded_mutation = _mutated_action_trajectory(
            trajectory,
            event_step=excluded_step,
            canary_id=f"excluded-current-or-future-action-step-{excluded_step}",
        )
        excluded_bytes = serialize_request(excluded_mutation, decision)
        if not isinstance(excluded_bytes, bytes):
            raise TypeError("processor canary serializer must return bytes")
        if excluded_bytes != base:
            raise ValueError(
                "excluded current or future action changed processor prompt bytes: "
                f"event step {excluded_step}"
            )
        excluded_results.append(
            {
                "event_step_id": excluded_step,
                "processor_prompt_sha256": sha256_bytes(excluded_bytes),
            }
        )

    return {
        "state_id": decision.get("state_id"),
        "decision_step_id": step,
        "included_history_action_canary_passed": True,
        "excluded_current_or_future_action_canary_applied": bool(excluded_steps),
        "excluded_current_or_future_action_canary_passed": True,
        "excluded_current_or_future_action_canary_mutation_count": len(
            excluded_results
        ),
        "excluded_current_or_future_action_canary_results": excluded_results,
        "base_processor_prompt_sha256": sha256_bytes(base),
    }


__all__ = [
    "ACTION_FIELDS",
    "build_decision_view_input",
    "validate_processor_byte_canary",
    "validate_request_manifest",
]
