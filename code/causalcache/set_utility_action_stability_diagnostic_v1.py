"""Train-only D1 diagnostic for localizing GUI-Owl reference-action instability."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v1 import (
    AUTO_FRESH_ENCODE_CONDITION,
    AUTO_FROZEN_ENCODED_CONDITION,
    EAGER_FROZEN_ENCODED_CONTROL,
)
from causalcache.set_utility_gui_owl_v2_1_action_stability_adapter_v1 import (
    REPEAT_COUNT,
    run_fresh_encode_condition_v1,
    run_frozen_encoded_condition_v1,
)
from causalcache.set_utility_label_producer import (
    UtilityQuerySpec,
    build_mixed_fidelity_prompt_plan,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_action_stability_diagnostic_v1"
STATUS = "IMPLEMENTED_SET_UTILITY_ACTION_STABILITY_DIAGNOSTIC_V1_SOURCE_CORE"
VALIDATION_STATUS = "VALID_SET_UTILITY_ACTION_STABILITY_DIAGNOSTIC_V1_SOURCE_CORE"

PARENT_SOURCE_COMMIT = "d5e0cca5c5e05d4aeeef74a1bbfae5685a4254c9"
PARENT_ENVELOPE_COMMIT = "e5002d8820b3e4c8c89343b73ba4a5d13aa117c3"
PARENT_RESULT_COMMIT = "0779a64f95ba42333c51f224deac3790ee66a79e"
PARENT_AGGREGATE_SHA256 = (
    "35e0c250232efbd2b9bdccfb1b04a7c372fbed892635342cce4f9f81097ff33f"
)
PARENT_SUMMARY_SHA256 = (
    "832b78369ccc02aff11324fd622f0b789655524a282ec671c7ae314ca53e636d"
)

CONDITION_ORDER = (
    AUTO_FRESH_ENCODE_CONDITION,
    AUTO_FROZEN_ENCODED_CONDITION,
    EAGER_FROZEN_ENCODED_CONTROL,
)
MISMATCH_STATE_IDS = (
    "0296753837938323:decision:006",
    "0310939638496410:decision:006",
    "0271654003819383:decision:010",
    "0279447750102246:decision:010",
)
CONTROL_STATE_IDS = (
    "0336706763935531:decision:006",
    "0268406573756492:decision:010",
)
STATE_IDS = (
    "0296753837938323:decision:006",
    "0310939638496410:decision:006",
    "0336706763935531:decision:006",
    "0271654003819383:decision:010",
    "0279447750102246:decision:010",
    "0268406573756492:decision:010",
)
WORKER_INDEX_BY_STATE = {
    "0296753837938323:decision:006": 0,
    "0310939638496410:decision:006": 1,
    "0336706763935531:decision:006": 2,
    "0271654003819383:decision:010": 2,
    "0279447750102246:decision:010": 3,
    "0268406573756492:decision:010": 1,
}
EXPECTED_GENERATION_CALL_CEILING = len(STATE_IDS) * len(CONDITION_ORDER) * REPEAT_COUNT
EXPECTED_ENCODE_CALL_CEILING = len(STATE_IDS) * (REPEAT_COUNT + 1 + 1)

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
_SAFE_FAILURE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_FORBIDDEN_SERIALIZED_KEY_FRAGMENTS = (
    "coordinate",
    "decoded_output_sha",
    "generated_sequence_sha",
    "logit",
    "output_text",
    "token_id",
)


def state_ids_for_worker_v1(worker_index: int) -> tuple[str, ...]:
    if type(worker_index) is not int or not 0 <= worker_index < 4:
        raise ValueError("D1 worker index must be in [0, 3]")
    return tuple(
        state_id
        for state_id in STATE_IDS
        if WORKER_INDEX_BY_STATE[state_id] == worker_index
    )


def _validate_condition_payload(
    payload: Mapping[str, Any],
    *,
    expected_condition_id: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _CONDITION_KEYS:
        raise ValueError("D1 condition fields drifted")
    if (
        payload.get("condition_id") != expected_condition_id
        or payload.get("metric_safe") is not True
        or payload.get("repeat_count") != REPEAT_COUNT
    ):
        raise ValueError("D1 condition identity drifted")
    for key in ("encode_call_count", "generation_call_count", "generation_completed_count"):
        value = payload.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"D1 {key} is invalid")
    if payload["generation_call_count"] > REPEAT_COUNT:
        raise ValueError("D1 condition exceeded its generation budget")
    if payload["generation_completed_count"] > payload["generation_call_count"]:
        raise ValueError("D1 completed generations exceed attempts")
    failure = payload.get("failure_class")
    if failure is not None and (
        not isinstance(failure, str) or _SAFE_FAILURE.fullmatch(failure) is None
    ):
        raise ValueError("D1 failure class is unsafe")
    equal_keys = (
        "canonical_action_equal",
        "decoded_output_equal",
        "exact_generated_sequence_equal",
    )
    if failure is None:
        if payload["generation_completed_count"] != REPEAT_COUNT or any(
            type(payload[key]) is not bool for key in equal_keys
        ):
            raise ValueError("successful D1 condition lacks equality results")
    elif any(payload[key] is not None for key in equal_keys):
        raise ValueError("failed D1 condition exposed partial equality results")
    prepared_keys = (
        "encoded_input_unchanged_after",
        "encoded_input_unchanged_before",
        "encoded_input_unchanged_between",
    )
    if expected_condition_id == AUTO_FRESH_ENCODE_CONDITION:
        if payload["encode_call_count"] != payload["generation_call_count"] or any(
            payload[key] is not None for key in prepared_keys
        ):
            raise ValueError("fresh D1 condition prepared-input fields drifted")
    elif payload["encode_call_count"] != 1 or any(
        value is not None and type(value) is not bool
        for value in (payload[key] for key in prepared_keys)
    ):
        raise ValueError("frozen D1 condition input-stability fields drifted")
    projected = dict(payload)
    _validate_metric_safe_tree(projected)
    return projected


def _validate_metric_safe_tree(value: Any, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError("D1 serialized keys must be strings")
            lowered = key.lower()
            if any(fragment in lowered for fragment in _FORBIDDEN_SERIALIZED_KEY_FRAGMENTS):
                raise ValueError(f"D1 metric-safe output contains forbidden key: {key}")
            _validate_metric_safe_tree(nested, path=(*path, key))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _validate_metric_safe_tree(nested, path=path)
        return
    if value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError("D1 metric-safe output contains a non-JSON value")


def _condition_stable(condition: Mapping[str, Any]) -> bool:
    if condition.get("failure_class") is not None:
        return False
    if not all(
        condition.get(key) is True
        for key in (
            "canonical_action_equal",
            "decoded_output_equal",
            "exact_generated_sequence_equal",
        )
    ):
        return False
    if condition.get("condition_id") != AUTO_FRESH_ENCODE_CONDITION and not all(
        condition.get(key) is True
        for key in (
            "encoded_input_unchanged_after",
            "encoded_input_unchanged_before",
            "encoded_input_unchanged_between",
        )
    ):
        return False
    return True


def _state_diagnosis(conditions: Mapping[str, Mapping[str, Any]]) -> str:
    fresh = _condition_stable(conditions[AUTO_FRESH_ENCODE_CONDITION])
    frozen = _condition_stable(conditions[AUTO_FROZEN_ENCODED_CONDITION])
    eager = _condition_stable(conditions[EAGER_FROZEN_ENCODED_CONTROL])
    if not eager:
        return "PERSISTENT_GENERATION_INSTABILITY"
    if not fresh and frozen:
        return "ENCODING_OR_PREPARATION_PATH_IMPLICATED"
    if not frozen:
        return "AUTO_ATTENTION_OR_NUMERICAL_CONTROL_IMPLICATED"
    if fresh and frozen:
        return "PARENT_MISMATCH_NOT_REPRODUCED"
    return "PREPARED_PATH_OR_HIDDEN_STATE_IMPLICATED"


def run_default_conditions_v1(
    query: UtilityQuerySpec,
    *,
    reference_input_builder: Callable[[object], object],
    runtime: object,
) -> dict[str, Any]:
    if not isinstance(query, UtilityQuerySpec) or query.state_id not in STATE_IDS:
        raise ValueError("D1 default run requires one frozen train state")
    if query.split != "train" or not callable(reference_input_builder):
        raise ValueError("D1 default run requires a train-only input builder")
    plan = build_mixed_fidelity_prompt_plan(query, query.candidate_event_step_ids)
    reference_input = reference_input_builder(plan)
    fresh = run_fresh_encode_condition_v1(runtime, reference_input)
    frozen = run_frozen_encoded_condition_v1(
        runtime,
        reference_input,
        condition_id=AUTO_FROZEN_ENCODED_CONDITION,
    )
    return {
        "conditions": [fresh, frozen],
        "metric_safe": True,
        "profile": "auto_default",
        "state_id": query.state_id,
    }


def run_eager_condition_v1(
    query: UtilityQuerySpec,
    *,
    reference_input_builder: Callable[[object], object],
    runtime: object,
) -> dict[str, Any]:
    if not isinstance(query, UtilityQuerySpec) or query.state_id not in STATE_IDS:
        raise ValueError("D1 eager run requires one frozen train state")
    if query.split != "train" or not callable(reference_input_builder):
        raise ValueError("D1 eager run requires a train-only input builder")
    plan = build_mixed_fidelity_prompt_plan(query, query.candidate_event_step_ids)
    reference_input = reference_input_builder(plan)
    eager = run_frozen_encoded_condition_v1(
        runtime,
        reference_input,
        condition_id=EAGER_FROZEN_ENCODED_CONTROL,
    )
    return {
        "conditions": [eager],
        "metric_safe": True,
        "profile": "eager_numerical_control_not_strict_cuda_determinism",
        "state_id": query.state_id,
    }


def merge_action_stability_state_v1(
    default_partial: Mapping[str, Any],
    eager_partial: Mapping[str, Any],
) -> dict[str, Any]:
    state_id = default_partial.get("state_id")
    if (
        state_id not in STATE_IDS
        or eager_partial.get("state_id") != state_id
        or default_partial.get("profile") != "auto_default"
        or eager_partial.get("profile")
        != "eager_numerical_control_not_strict_cuda_determinism"
        or default_partial.get("metric_safe") is not True
        or eager_partial.get("metric_safe") is not True
    ):
        raise ValueError("D1 partial result identities drifted")
    raw_conditions = [
        *default_partial.get("conditions", ()),
        *eager_partial.get("conditions", ()),
    ]
    if len(raw_conditions) != len(CONDITION_ORDER):
        raise ValueError("D1 partial condition count drifted")
    conditions = {
        expected: _validate_condition_payload(raw, expected_condition_id=expected)
        for expected, raw in zip(CONDITION_ORDER, raw_conditions, strict=True)
    }
    payload = {
        "conditions": [conditions[condition] for condition in CONDITION_ORDER],
        "diagnosis": _state_diagnosis(conditions),
        "metric_safe": True,
        "parent_role": "mismatch" if state_id in MISMATCH_STATE_IDS else "stable_control",
        "state_id": state_id,
        "worker_index": WORKER_INDEX_BY_STATE[state_id],
    }
    _validate_metric_safe_tree(payload)
    return payload


def aggregate_action_stability_diagnostic_v1(
    states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(states, (str, bytes, bytearray, Mapping)):
        raise TypeError("D1 aggregate requires a sequence of state results")
    by_state = {str(state.get("state_id")): state for state in states}
    if len(by_state) != len(STATE_IDS) or set(by_state) != set(STATE_IDS):
        raise ValueError("D1 aggregate state roster drifted")
    ordered = [dict(by_state[state_id]) for state_id in STATE_IDS]
    for state in ordered:
        _validate_metric_safe_tree(state)
    controls_stable = all(
        state["diagnosis"] == "PARENT_MISMATCH_NOT_REPRODUCED"
        for state in ordered
        if state["state_id"] in CONTROL_STATE_IDS
    )
    diagnoses = {
        state["state_id"]: state["diagnosis"]
        for state in ordered
        if state["state_id"] in MISMATCH_STATE_IDS
    }
    if not controls_stable:
        verdict = "INVALID_STABLE_CONTROL_INSTABILITY"
    elif any(value == "PERSISTENT_GENERATION_INSTABILITY" for value in diagnoses.values()):
        verdict = "PERSISTENT_GENERATION_INSTABILITY"
    elif any(
        value == "AUTO_ATTENTION_OR_NUMERICAL_CONTROL_IMPLICATED"
        for value in diagnoses.values()
    ):
        verdict = "AUTO_ATTENTION_OR_NUMERICAL_CONTROL_IMPLICATED"
    elif any(
        value == "ENCODING_OR_PREPARATION_PATH_IMPLICATED"
        for value in diagnoses.values()
    ):
        verdict = "ENCODING_OR_PREPARATION_PATH_IMPLICATED"
    elif all(value == "PARENT_MISMATCH_NOT_REPRODUCED" for value in diagnoses.values()):
        verdict = "PARENT_MISMATCH_NOT_REPRODUCED"
    else:
        verdict = "MIXED_OR_HIDDEN_STATE_DIAGNOSIS"
    generation_calls = sum(
        int(condition["generation_call_count"])
        for state in ordered
        for condition in state["conditions"]
    )
    encode_calls = sum(
        int(condition["encode_call_count"])
        for state in ordered
        for condition in state["conditions"]
    )
    payload = {
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
    "CONDITION_ORDER",
    "CONTROL_STATE_IDS",
    "EXPECTED_ENCODE_CALL_CEILING",
    "EXPECTED_GENERATION_CALL_CEILING",
    "MISMATCH_STATE_IDS",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "STATE_IDS",
    "STATUS",
    "VALIDATION_STATUS",
    "WORKER_INDEX_BY_STATE",
    "aggregate_action_stability_diagnostic_v1",
    "merge_action_stability_state_v1",
    "run_default_conditions_v1",
    "run_eager_condition_v1",
    "state_ids_for_worker_v1",
]
