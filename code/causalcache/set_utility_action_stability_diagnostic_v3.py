"""Focused strict-determinism action-stability diagnostic D2."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v1 import (
    EAGER_FROZEN_ENCODED_CONTROL,
)
from causalcache.set_utility_action_stability_diagnostic_v1 import (
    _condition_stable,
    _validate_condition_payload,
    _validate_metric_safe_tree,
)
from causalcache.set_utility_action_stability_diagnostic_v2 import (
    PROTOCOL_ID as D1B_PROTOCOL_ID,
    SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION,
    _validate_sdpa_condition_v2,
)
from causalcache.set_utility_gui_owl_v2_1_action_stability_adapter_v1 import (
    REPEAT_COUNT,
    run_frozen_encoded_condition_v1,
)
from causalcache.set_utility_label_producer import (
    UtilityQuerySpec,
    build_mixed_fidelity_prompt_plan,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_action_stability_diagnostic_v3"
STATUS = "IMPLEMENTED_STRICT_DETERMINISM_ACTION_STABILITY_D2_SOURCE_CORE"
VALIDATION_STATUS = "VALID_STRICT_DETERMINISM_ACTION_STABILITY_D2_SOURCE_CORE"

STRICT_DETERMINISM_FROZEN_ENCODED_CONDITION = (
    "strict_determinism_sdpa_frozen_encoded"
)
STRICT_DETERMINISM_PROFILE = (
    "pytorch_strict_determinism_cublas_workspace_4096_8_sdpa"
)
TARGET_STATE_ID = "0296753837938323:decision:006"
CONTROL_STATE_IDS = (
    "0336706763935531:decision:006",
    "0268406573756492:decision:010",
)
STATE_IDS = (TARGET_STATE_ID, *CONTROL_STATE_IDS)
STATE_ROLES = {
    TARGET_STATE_ID: "unstable_target",
    CONTROL_STATE_IDS[0]: "stable_control_same_decision_stratum",
    CONTROL_STATE_IDS[1]: "stable_control_longer_context_stratum",
}
STATE_WAVES = (STATE_IDS,)
EXPECTED_GENERATION_CALL_CEILING = len(STATE_IDS) * REPEAT_COUNT
EXPECTED_ENCODE_CALL_CEILING = len(STATE_IDS)

_SAFE_FAILURE_CLASS = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_CONDITION_KEYS = frozenset(
    {
        "canonical_action_equal",
        "condition_id",
        "decoded_output_equal",
        "encode_call_count",
        "encoded_input_unchanged_after",
        "encoded_input_unchanged_before",
        "encoded_input_unchanged_between",
        "exact_generated_sequence_equal",
        "failure_class",
        "generation_call_count",
        "generation_completed_count",
        "metric_safe",
        "repeat_count",
    }
)


def _validate_strict_condition_v3(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _CONDITION_KEYS:
        raise ValueError("D2 condition fields drifted")
    counts = (
        payload.get("encode_call_count"),
        payload.get("generation_call_count"),
        payload.get("generation_completed_count"),
    )
    if counts != (0, 0, 0):
        projected = dict(payload)
        projected["condition_id"] = SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION
        validated = _validate_condition_payload(
            projected,
            expected_condition_id=SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION,
        )
        validated["condition_id"] = STRICT_DETERMINISM_FROZEN_ENCODED_CONDITION
        return validated
    failure_class = payload.get("failure_class")
    nullable = _CONDITION_KEYS - {
        "condition_id",
        "encode_call_count",
        "failure_class",
        "generation_call_count",
        "generation_completed_count",
        "metric_safe",
        "repeat_count",
    }
    if (
        payload.get("condition_id") != STRICT_DETERMINISM_FROZEN_ENCODED_CONDITION
        or payload.get("metric_safe") is not True
        or payload.get("repeat_count") != REPEAT_COUNT
        or not isinstance(failure_class, str)
        or _SAFE_FAILURE_CLASS.fullmatch(failure_class) is None
        or any(payload.get(key) is not None for key in nullable)
    ):
        raise ValueError("D2 zero-call failure projection drifted")
    projected = dict(payload)
    _validate_metric_safe_tree(projected)
    return projected


def build_strict_execution_failure_partial_v3(
    state_id: str, *, failure_class: str
) -> dict[str, Any]:
    if state_id not in STATE_IDS:
        raise ValueError("D2 failure projection requires one frozen state")
    condition = _validate_strict_condition_v3(
        {
            "canonical_action_equal": None,
            "condition_id": STRICT_DETERMINISM_FROZEN_ENCODED_CONDITION,
            "decoded_output_equal": None,
            "encode_call_count": 0,
            "encoded_input_unchanged_after": None,
            "encoded_input_unchanged_before": None,
            "encoded_input_unchanged_between": None,
            "exact_generated_sequence_equal": None,
            "failure_class": failure_class,
            "generation_call_count": 0,
            "generation_completed_count": 0,
            "metric_safe": True,
            "repeat_count": REPEAT_COUNT,
        }
    )
    return {
        "conditions": [condition],
        "metric_safe": True,
        "profile": STRICT_DETERMINISM_PROFILE,
        "state_id": state_id,
    }


def run_strict_determinism_condition_v3(
    query: UtilityQuerySpec,
    *,
    reference_input_builder: Callable[[object], object],
    runtime: object,
) -> dict[str, Any]:
    """Encode once and run two generations from the same GPU tensor mapping."""
    if not isinstance(query, UtilityQuerySpec) or query.state_id not in STATE_IDS:
        raise ValueError("D2 run requires one frozen train state")
    if query.split != "train" or not callable(reference_input_builder):
        raise ValueError("D2 run requires a train-only input builder")
    plan = build_mixed_fidelity_prompt_plan(query, query.candidate_event_step_ids)
    reference_input = reference_input_builder(plan)
    legacy = run_frozen_encoded_condition_v1(
        runtime,
        reference_input,
        condition_id=EAGER_FROZEN_ENCODED_CONTROL,
    )
    if legacy.get("condition_id") != EAGER_FROZEN_ENCODED_CONTROL:
        raise RuntimeError("D2 frozen-input adapter identity drifted")
    condition = _validate_strict_condition_v3(
        {
            **legacy,
            "condition_id": STRICT_DETERMINISM_FROZEN_ENCODED_CONDITION,
        }
    )
    payload = {
        "conditions": [condition],
        "metric_safe": True,
        "profile": STRICT_DETERMINISM_PROFILE,
        "state_id": query.state_id,
    }
    _validate_metric_safe_tree(payload)
    return payload


def validate_strict_partial_v3(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "conditions",
        "metric_safe",
        "profile",
        "state_id",
    }:
        raise ValueError("D2 partial fields drifted")
    raw = payload.get("conditions")
    if (
        payload.get("state_id") not in STATE_IDS
        or payload.get("metric_safe") is not True
        or payload.get("profile") != STRICT_DETERMINISM_PROFILE
        or isinstance(raw, (str, bytes, bytearray, Mapping))
        or not isinstance(raw, Sequence)
        or len(raw) != 1
    ):
        raise ValueError("D2 partial identity drifted")
    projected = {
        "conditions": [_validate_strict_condition_v3(raw[0])],
        "metric_safe": True,
        "profile": STRICT_DETERMINISM_PROFILE,
        "state_id": payload["state_id"],
    }
    _validate_metric_safe_tree(projected)
    return projected


def _validate_d1b_parent_state_v3(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("state_id") not in STATE_IDS:
        raise ValueError("D2 parent state identity drifted")
    condition = _validate_sdpa_condition_v2(
        payload.get("sdpa_numerical_control_condition")
    )
    expected_stable = payload["state_id"] in CONTROL_STATE_IDS
    if (
        payload.get("metric_safe") is not True
        or _condition_stable(condition) is not expected_stable
    ):
        raise ValueError("D2 parent target/control stability drifted")
    projected = {
        "condition": condition,
        "metric_safe": True,
        "protocol_id": D1B_PROTOCOL_ID,
        "state_id": payload["state_id"],
    }
    _validate_metric_safe_tree(projected)
    return projected


def merge_action_stability_state_v3(
    parent_d1b_state: Mapping[str, Any], strict_partial: Mapping[str, Any]
) -> dict[str, Any]:
    parent = _validate_d1b_parent_state_v3(parent_d1b_state)
    partial = validate_strict_partial_v3(strict_partial)
    if parent["state_id"] != partial["state_id"]:
        raise ValueError("D2 parent and strict state identities differ")
    condition = partial["conditions"][0]
    if condition["failure_class"] is not None:
        diagnosis = "STRICT_DETERMINISM_RUNTIME_FAILURE"
    elif not _condition_stable(condition):
        diagnosis = (
            "STABLE_CONTROL_STRICT_INSTABILITY"
            if partial["state_id"] in CONTROL_STATE_IDS
            else "TARGET_STRICT_REPEAT_INSTABILITY"
        )
    else:
        diagnosis = (
            "STABLE_CONTROL_REPRODUCED_UNDER_STRICT_DETERMINISM"
            if partial["state_id"] in CONTROL_STATE_IDS
            else "TARGET_STABILIZED_UNDER_STRICT_DETERMINISM"
        )
    result = {
        "diagnosis": diagnosis,
        "d1b_context": parent,
        "metric_safe": True,
        "role": STATE_ROLES[partial["state_id"]],
        "state_id": partial["state_id"],
        "strict_condition": condition,
    }
    _validate_metric_safe_tree(result)
    return result


def aggregate_action_stability_diagnostic_v3(
    states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(states, (str, bytes, bytearray, Mapping)) or not isinstance(
        states, Sequence
    ):
        raise TypeError("D2 aggregate requires the frozen state sequence")
    if [state.get("state_id") for state in states] != list(STATE_IDS):
        raise ValueError("D2 aggregate state order drifted")
    validated = [dict(state) for state in states]
    for state in validated:
        if state.get("role") != STATE_ROLES[state["state_id"]]:
            raise ValueError("D2 aggregate role drifted")
        _validate_strict_condition_v3(state.get("strict_condition"))
    conditions = [state["strict_condition"] for state in validated]
    failures = [item for item in conditions if item["failure_class"] is not None]
    unstable_controls = [
        state
        for state in validated
        if state["state_id"] in CONTROL_STATE_IDS
        and state["strict_condition"]["failure_class"] is None
        and not _condition_stable(state["strict_condition"])
    ]
    target = validated[0]["strict_condition"]
    if failures:
        verdict = "INVALID_RUNTIME_FAILURE"
    elif unstable_controls:
        verdict = "INVALID_STABLE_CONTROL_INSTABILITY"
    elif not _condition_stable(target):
        verdict = "NO_GO_STRICT_DETERMINISM_REPEAT_INSTABILITY"
    else:
        verdict = "PASS_STRICT_DETERMINISM_REPEAT_STABILITY_DIAGNOSTIC"
    counts = {
        "encode_call_count": sum(item["encode_call_count"] for item in conditions),
        "generation_call_count": sum(
            item["generation_call_count"] for item in conditions
        ),
        "retry_count": 0,
        "state_process_count": len(STATE_IDS),
    }
    result = {
        "counts": counts,
        "metric_safe": True,
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "states": validated,
        "verdict": verdict,
    }
    _validate_metric_safe_tree(result)
    return result


__all__ = [
    "CONTROL_STATE_IDS",
    "EXPECTED_ENCODE_CALL_CEILING",
    "EXPECTED_GENERATION_CALL_CEILING",
    "PROTOCOL_ID",
    "STATE_IDS",
    "STATE_ROLES",
    "STATE_WAVES",
    "STRICT_DETERMINISM_FROZEN_ENCODED_CONDITION",
    "STRICT_DETERMINISM_PROFILE",
    "TARGET_STATE_ID",
    "aggregate_action_stability_diagnostic_v3",
    "build_strict_execution_failure_partial_v3",
    "merge_action_stability_state_v3",
    "run_strict_determinism_condition_v3",
    "validate_strict_partial_v3",
]
