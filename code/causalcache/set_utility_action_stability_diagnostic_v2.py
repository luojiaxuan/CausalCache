"""D1b memory-safe SDPA numerical-control repeat-stability diagnostic."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v1 import (
    AUTO_FRESH_ENCODE_CONDITION,
    AUTO_FROZEN_ENCODED_CONDITION,
    EAGER_FROZEN_ENCODED_CONTROL,
)
from causalcache.set_utility_action_stability_diagnostic_v1 import (
    CONTROL_STATE_IDS as D1_CONTROL_STATE_IDS,
    MISMATCH_STATE_IDS as D1_MISMATCH_STATE_IDS,
    PROTOCOL_ID as D1_PROTOCOL_ID,
    STATE_IDS as D1_STATE_IDS,
    _condition_stable,
    _validate_condition_payload,
    _validate_metric_safe_tree,
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
PROTOCOL_ID = "causalcache_set_utility_action_stability_diagnostic_v2"
STATUS = "IMPLEMENTED_SET_UTILITY_ACTION_STABILITY_DIAGNOSTIC_V2_SOURCE_CORE"
VALIDATION_STATUS = (
    "VALID_SET_UTILITY_ACTION_STABILITY_DIAGNOSTIC_V2_SOURCE_CORE"
)

SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION = (
    "sdpa_numerical_control_frozen_encoded"
)
SDPA_NUMERICAL_CONTROL_PROFILE = (
    "sdpa_numerical_control_not_strict_cuda_determinism"
)

STATE_IDS = tuple(D1_STATE_IDS)
MISMATCH_STATE_IDS = tuple(D1_MISMATCH_STATE_IDS)
CONTROL_STATE_IDS = tuple(D1_CONTROL_STATE_IDS)
STATE_WAVES = (
    (STATE_IDS[0], STATE_IDS[3], STATE_IDS[4], STATE_IDS[5]),
    (STATE_IDS[1], STATE_IDS[2]),
)
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


def state_ids_for_wave_v2(wave_index: int) -> tuple[str, ...]:
    if type(wave_index) is not int or not 0 <= wave_index < len(STATE_WAVES):
        raise ValueError("D1b wave index must be zero or one")
    return STATE_WAVES[wave_index]


def _validate_sdpa_condition_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _CONDITION_KEYS:
        raise ValueError("D1b condition fields drifted")
    count_keys = (
        "encode_call_count",
        "generation_call_count",
        "generation_completed_count",
    )
    is_outer_execution_failure = all(
        type(payload.get(key)) is int and payload.get(key) == 0
        for key in count_keys
    )
    if not is_outer_execution_failure:
        return _validate_condition_payload(
            payload,
            expected_condition_id=SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION,
        )
    failure_class = payload.get("failure_class")
    nullable_fields = (
        "canonical_action_equal",
        "decoded_output_equal",
        "encoded_input_unchanged_after",
        "encoded_input_unchanged_before",
        "encoded_input_unchanged_between",
        "exact_generated_sequence_equal",
    )
    if (
        payload.get("condition_id")
        != SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION
        or payload.get("metric_safe") is not True
        or payload.get("repeat_count") != REPEAT_COUNT
        or not isinstance(failure_class, str)
        or _SAFE_FAILURE_CLASS.fullmatch(failure_class) is None
        or any(payload.get(key) is not None for key in nullable_fields)
    ):
        raise ValueError("D1b zero-call execution-failure condition drifted")
    projected = dict(payload)
    _validate_metric_safe_tree(projected)
    return projected


def build_sdpa_execution_failure_partial_v2(
    state_id: str,
    *,
    failure_class: str,
) -> dict[str, Any]:
    """Project a pre-condition process failure without inventing policy calls."""
    if state_id not in STATE_IDS:
        raise ValueError("D1b execution failure requires one frozen state")
    condition = _validate_sdpa_condition_v2(
        {
            "canonical_action_equal": None,
            "condition_id": SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION,
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
        "profile": SDPA_NUMERICAL_CONTROL_PROFILE,
        "state_id": state_id,
    }


def _run_frozen_sdpa_condition_v2(
    runtime: object,
    reference_input: object,
) -> dict[str, Any]:
    # note (luojiaxuan): The v1 adapter's frozen-input mechanism is backend
    # agnostic, but its public identity gate is historical.  Run that exact
    # mechanism under its sentinel and immediately project the new v2 identity;
    # no eager result or eager runtime is consumed by D1b.
    legacy = run_frozen_encoded_condition_v1(
        runtime,
        reference_input,
        condition_id=EAGER_FROZEN_ENCODED_CONTROL,
    )
    if legacy.get("condition_id") != EAGER_FROZEN_ENCODED_CONTROL:
        raise RuntimeError("D1b frozen-input adapter identity drifted")
    projected = {
        **legacy,
        "condition_id": SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION,
    }
    return _validate_sdpa_condition_v2(projected)


def run_sdpa_numerical_control_condition_v2(
    query: UtilityQuerySpec,
    *,
    reference_input_builder: Callable[[object], object],
    runtime: object,
) -> dict[str, Any]:
    """Run D1b's only new condition for one frozen train state."""
    if not isinstance(query, UtilityQuerySpec) or query.state_id not in STATE_IDS:
        raise ValueError("D1b run requires one frozen D1 train state")
    if query.split != "train" or not callable(reference_input_builder):
        raise ValueError("D1b run requires a train-only input builder")
    plan = build_mixed_fidelity_prompt_plan(query, query.candidate_event_step_ids)
    reference_input = reference_input_builder(plan)
    condition = _run_frozen_sdpa_condition_v2(runtime, reference_input)
    payload = {
        "conditions": [condition],
        "metric_safe": True,
        "profile": SDPA_NUMERICAL_CONTROL_PROFILE,
        "state_id": query.state_id,
    }
    _validate_metric_safe_tree(payload)
    return payload


def _validate_sdpa_partial_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {"conditions", "metric_safe", "profile", "state_id"}
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError("D1b partial fields drifted")
    state_id = payload.get("state_id")
    raw_conditions = payload.get("conditions")
    if (
        state_id not in STATE_IDS
        or payload.get("metric_safe") is not True
        or payload.get("profile") != SDPA_NUMERICAL_CONTROL_PROFILE
        or isinstance(raw_conditions, (str, bytes, bytearray, Mapping))
        or not isinstance(raw_conditions, Sequence)
        or len(raw_conditions) != 1
    ):
        raise ValueError("D1b partial identity drifted")
    condition = _validate_sdpa_condition_v2(raw_conditions[0])
    projected = {
        "conditions": [condition],
        "metric_safe": True,
        "profile": SDPA_NUMERICAL_CONTROL_PROFILE,
        "state_id": state_id,
    }
    _validate_metric_safe_tree(projected)
    return projected


def _parent_d1_context_v2(parent_d1_state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(parent_d1_state, Mapping):
        raise TypeError("D1b parent state must be a mapping")
    state_id = parent_d1_state.get("state_id")
    expected_role = "mismatch" if state_id in MISMATCH_STATE_IDS else "stable_control"
    raw_conditions = parent_d1_state.get("conditions")
    if (
        state_id not in STATE_IDS
        or parent_d1_state.get("metric_safe") is not True
        or parent_d1_state.get("parent_role") != expected_role
        or isinstance(raw_conditions, (str, bytes, bytearray, Mapping))
        or not isinstance(raw_conditions, Sequence)
        or len(raw_conditions) < 2
    ):
        raise ValueError("D1b parent state auto-context identity drifted")
    expected_ids = (AUTO_FRESH_ENCODE_CONDITION, AUTO_FROZEN_ENCODED_CONDITION)
    auto_conditions = tuple(
        _validate_condition_payload(raw, expected_condition_id=condition_id)
        for raw, condition_id in zip(
            raw_conditions[:2],
            expected_ids,
            strict=True,
        )
    )
    if any(item["failure_class"] is not None for item in auto_conditions):
        raise ValueError("D1b requires complete D1 auto-condition context")
    return {
        "conditions": [dict(item) for item in auto_conditions],
        "metric_safe": True,
        "protocol_id": D1_PROTOCOL_ID,
    }


def _state_diagnosis_v2(
    *,
    parent_role: str,
    d1_context: Mapping[str, Any],
    sdpa_condition: Mapping[str, Any],
) -> str:
    if sdpa_condition.get("failure_class") is not None:
        return "INVALID_CONDITION_EXECUTION_FAILURE"
    fresh, frozen = d1_context["conditions"]
    if not _condition_stable(sdpa_condition):
        if parent_role == "stable_control":
            return "STABLE_CONTROL_SDPA_CONTROL_INSTABILITY"
        if not _condition_stable(frozen):
            return "D1_AUTO_FROZEN_INSTABILITY_UNRESOLVED_BY_SDPA_CONTROL"
        if not _condition_stable(fresh):
            return "MIXED_D1_FRESH_PATH_AND_SDPA_CONTROL_INSTABILITY"
        return "SDPA_CONTROL_ONLY_INSTABILITY"
    if not _condition_stable(frozen):
        return "AUTO_VS_CONTROLLED_SDPA_PROFILE_ASSOCIATION"
    if not _condition_stable(fresh):
        return "FRESH_VS_FROZEN_PATH_ASSOCIATION_WITH_SDPA_CONTROL_STABLE"
    if parent_role == "mismatch":
        return "PARENT_MISMATCH_NOT_REPRODUCED_UNDER_SDPA_CONTROL"
    return "STABLE_CONTROL_REPRODUCED"


def merge_action_stability_state_v2(
    parent_d1_state: Mapping[str, Any],
    sdpa_partial: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge frozen D1 auto context with D1b's sole new condition."""
    d1_context = _parent_d1_context_v2(parent_d1_state)
    partial = _validate_sdpa_partial_v2(sdpa_partial)
    state_id = parent_d1_state.get("state_id")
    if partial["state_id"] != state_id:
        raise ValueError("D1b parent and new-condition state identities differ")
    parent_role = "mismatch" if state_id in MISMATCH_STATE_IDS else "stable_control"
    condition = partial["conditions"][0]
    payload = {
        "d1_auto_context": d1_context,
        "diagnosis": _state_diagnosis_v2(
            parent_role=parent_role,
            d1_context=d1_context,
            sdpa_condition=condition,
        ),
        "metric_safe": True,
        "parent_role": parent_role,
        "sdpa_numerical_control_condition": condition,
        "state_id": state_id,
        "wave_index": 0 if state_id in STATE_WAVES[0] else 1,
    }
    _validate_metric_safe_tree(payload)
    return payload


def _validate_merged_state_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "d1_auto_context",
        "diagnosis",
        "metric_safe",
        "parent_role",
        "sdpa_numerical_control_condition",
        "state_id",
        "wave_index",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError("D1b merged-state fields drifted")
    state_id = payload.get("state_id")
    if state_id not in STATE_IDS or payload.get("metric_safe") is not True:
        raise ValueError("D1b merged-state identity drifted")
    expected_role = "mismatch" if state_id in MISMATCH_STATE_IDS else "stable_control"
    expected_wave = 0 if state_id in STATE_WAVES[0] else 1
    if (
        payload.get("parent_role") != expected_role
        or payload.get("wave_index") != expected_wave
    ):
        raise ValueError("D1b merged-state role or wave drifted")
    context = payload.get("d1_auto_context")
    if not isinstance(context, Mapping) or set(context) != {
        "conditions",
        "metric_safe",
        "protocol_id",
    }:
        raise ValueError("D1b merged-state context drifted")
    context_conditions = context.get("conditions")
    if (
        context.get("metric_safe") is not True
        or context.get("protocol_id") != D1_PROTOCOL_ID
        or isinstance(context_conditions, (str, bytes, bytearray, Mapping))
        or not isinstance(context_conditions, Sequence)
        or len(context_conditions) != 2
    ):
        raise ValueError("D1b merged-state context identity drifted")
    validated_context = {
        "conditions": [
            _validate_condition_payload(raw, expected_condition_id=condition_id)
            for raw, condition_id in zip(
                context_conditions,
                (AUTO_FRESH_ENCODE_CONDITION, AUTO_FROZEN_ENCODED_CONDITION),
                strict=True,
            )
        ],
        "metric_safe": True,
        "protocol_id": D1_PROTOCOL_ID,
    }
    if any(
        item["failure_class"] is not None
        for item in validated_context["conditions"]
    ):
        raise ValueError("D1b merged state lost complete parent auto context")
    condition = _validate_sdpa_condition_v2(
        payload.get("sdpa_numerical_control_condition")
    )
    diagnosis = _state_diagnosis_v2(
        parent_role=expected_role,
        d1_context=validated_context,
        sdpa_condition=condition,
    )
    if payload.get("diagnosis") != diagnosis:
        raise ValueError("D1b merged-state diagnosis drifted")
    projected = {
        "d1_auto_context": validated_context,
        "diagnosis": diagnosis,
        "metric_safe": True,
        "parent_role": expected_role,
        "sdpa_numerical_control_condition": condition,
        "state_id": state_id,
        "wave_index": expected_wave,
    }
    _validate_metric_safe_tree(projected)
    return projected


def aggregate_action_stability_diagnostic_v2(
    states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate exactly six fresh-process D1b condition results."""
    if (
        isinstance(states, (str, bytes, bytearray, Mapping))
        or not isinstance(states, Sequence)
        or len(states) != len(STATE_IDS)
    ):
        raise TypeError("D1b aggregate requires the exact six-state result sequence")
    validated = [_validate_merged_state_v2(state) for state in states]
    by_state = {state["state_id"]: state for state in validated}
    if len(by_state) != len(STATE_IDS) or set(by_state) != set(STATE_IDS):
        raise ValueError("D1b aggregate state roster drifted")
    ordered = [dict(by_state[state_id]) for state_id in STATE_IDS]
    conditions = [state["sdpa_numerical_control_condition"] for state in ordered]
    any_runtime_failure = any(
        condition["failure_class"] is not None for condition in conditions
    )
    stable_control_instability = any(
        state["state_id"] in CONTROL_STATE_IDS
        and not _condition_stable(state["sdpa_numerical_control_condition"])
        for state in ordered
    )
    if any_runtime_failure:
        verdict = "INVALID_RUNTIME_FAILURE"
    elif stable_control_instability:
        verdict = "INVALID_STABLE_CONTROL_INSTABILITY"
    elif all(_condition_stable(condition) for condition in conditions):
        verdict = "PASS_MEMORY_SAFE_SDPA_NUMERICAL_CONTROL_REPEAT_STABILITY"
    else:
        verdict = "NO_GO_SDPA_CONTROL_REPEAT_INSTABILITY"

    generation_calls = sum(int(item["generation_call_count"]) for item in conditions)
    encode_calls = sum(int(item["encode_call_count"]) for item in conditions)
    if (
        generation_calls > EXPECTED_GENERATION_CALL_CEILING
        or encode_calls > EXPECTED_ENCODE_CALL_CEILING
    ):
        raise ValueError("D1b aggregate exceeded its frozen operation ceiling")
    if not any_runtime_failure and (
        generation_calls != EXPECTED_GENERATION_CALL_CEILING
        or encode_calls != EXPECTED_ENCODE_CALL_CEILING
    ):
        raise ValueError("complete D1b aggregate operation counts drifted")
    payload = {
        "context": {
            "parent_calls_counted": False,
            "parent_condition_ids": [
                AUTO_FRESH_ENCODE_CONDITION,
                AUTO_FROZEN_ENCODED_CONDITION,
            ],
            "parent_protocol_id": D1_PROTOCOL_ID,
        },
        "counts": {
            "encode_call_ceiling": EXPECTED_ENCODE_CALL_CEILING,
            "encode_call_count": encode_calls,
            "generation_call_ceiling": EXPECTED_GENERATION_CALL_CEILING,
            "generation_call_count": generation_calls,
            "retry_count": 0,
            "state_count": len(ordered),
        },
        "metric_safe": True,
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "states": ordered,
        "verdict": verdict,
    }
    _validate_metric_safe_tree(payload)
    return payload


__all__ = [
    "CONTROL_STATE_IDS",
    "EXPECTED_ENCODE_CALL_CEILING",
    "EXPECTED_GENERATION_CALL_CEILING",
    "MISMATCH_STATE_IDS",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION",
    "SDPA_NUMERICAL_CONTROL_PROFILE",
    "STATE_IDS",
    "STATE_WAVES",
    "STATUS",
    "VALIDATION_STATUS",
    "aggregate_action_stability_diagnostic_v2",
    "build_sdpa_execution_failure_partial_v2",
    "merge_action_stability_state_v2",
    "run_sdpa_numerical_control_condition_v2",
    "state_ids_for_wave_v2",
]
