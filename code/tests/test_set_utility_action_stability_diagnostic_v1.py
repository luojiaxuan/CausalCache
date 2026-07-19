from __future__ import annotations

import copy

import pytest

from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v1 import (
    AUTO_FRESH_ENCODE_CONDITION,
    AUTO_FROZEN_ENCODED_CONDITION,
    EAGER_FROZEN_ENCODED_CONTROL,
)
from causalcache.set_utility_action_stability_diagnostic_v1 import (
    CONDITION_ORDER,
    CONTROL_STATE_IDS,
    EXPECTED_ENCODE_CALL_CEILING,
    EXPECTED_GENERATION_CALL_CEILING,
    MISMATCH_STATE_IDS,
    STATE_IDS,
    aggregate_action_stability_diagnostic_v1,
    merge_action_stability_state_v1,
    state_ids_for_worker_v1,
)


def _condition(condition_id: str, *, stable: bool = True):
    prepared = condition_id != AUTO_FRESH_ENCODE_CONDITION
    return {
        "canonical_action_equal": stable,
        "condition_id": condition_id,
        "decoded_output_equal": stable,
        "encode_call_count": 1 if prepared else 2,
        "encoded_input_unchanged_after": True if prepared else None,
        "encoded_input_unchanged_before": True if prepared else None,
        "encoded_input_unchanged_between": True if prepared else None,
        "exact_generated_sequence_equal": stable,
        "failure_class": None,
        "generation_call_count": 2,
        "generation_completed_count": 2,
        "metric_safe": True,
        "repeat_count": 2,
    }


def _state(state_id: str, *, fresh=True, frozen=True, eager=True):
    default = {
        "conditions": [
            _condition(AUTO_FRESH_ENCODE_CONDITION, stable=fresh),
            _condition(AUTO_FROZEN_ENCODED_CONDITION, stable=frozen),
        ],
        "metric_safe": True,
        "profile": "auto_default",
        "state_id": state_id,
    }
    eager_partial = {
        "conditions": [_condition(EAGER_FROZEN_ENCODED_CONTROL, stable=eager)],
        "metric_safe": True,
        "profile": "eager_numerical_control_not_strict_cuda_determinism",
        "state_id": state_id,
    }
    return merge_action_stability_state_v1(default, eager_partial)


def test_roster_and_worker_mapping_are_exact_and_balanced() -> None:
    assert len(STATE_IDS) == 6
    assert set(MISMATCH_STATE_IDS).isdisjoint(CONTROL_STATE_IDS)
    assert set(MISMATCH_STATE_IDS) | set(CONTROL_STATE_IDS) == set(STATE_IDS)
    assert [len(state_ids_for_worker_v1(index)) for index in range(4)] == [1, 2, 2, 1]
    assert EXPECTED_GENERATION_CALL_CEILING == 36
    assert EXPECTED_ENCODE_CALL_CEILING == 24


@pytest.mark.parametrize(
    ("fresh", "frozen", "eager", "diagnosis"),
    [
        (False, True, True, "FRESH_VS_FROZEN_PATH_ASSOCIATION"),
        (False, False, True, "AUTO_VS_EAGER_PROFILE_ASSOCIATION"),
        (False, False, False, "PERSISTENT_GENERATION_INSTABILITY"),
        (True, True, True, "PARENT_MISMATCH_NOT_REPRODUCED"),
    ],
)
def test_state_diagnosis_is_preregistered(fresh, frozen, eager, diagnosis) -> None:
    payload = _state(STATE_IDS[0], fresh=fresh, frozen=frozen, eager=eager)
    assert payload["diagnosis"] == diagnosis
    assert [item["condition_id"] for item in payload["conditions"]] == list(CONDITION_ORDER)


def test_aggregate_classifies_backend_recovery_and_preserves_call_budget() -> None:
    states = [_state(state_id) for state_id in STATE_IDS]
    states[0] = _state(STATE_IDS[0], fresh=False, frozen=False, eager=True)
    payload = aggregate_action_stability_diagnostic_v1(states)
    assert payload["verdict"] == "AUTO_VS_EAGER_PROFILE_ASSOCIATION"
    assert payload["counts"] == {
        "encode_call_ceiling": 24,
        "encode_call_count": 24,
        "generation_call_ceiling": 36,
        "generation_call_count": 36,
        "retry_count": 0,
        "state_count": 6,
    }


def test_unstable_control_invalidates_diagnostic() -> None:
    states = [_state(state_id) for state_id in STATE_IDS]
    control = CONTROL_STATE_IDS[0]
    states[STATE_IDS.index(control)] = _state(control, eager=False)
    assert (
        aggregate_action_stability_diagnostic_v1(states)["verdict"]
        == "INVALID_STABLE_CONTROL_INSTABILITY"
    )


def test_condition_schema_and_metric_firewall_fail_closed() -> None:
    default = {
        "conditions": [
            _condition(AUTO_FRESH_ENCODE_CONDITION),
            _condition(AUTO_FROZEN_ENCODED_CONDITION),
        ],
        "metric_safe": True,
        "profile": "auto_default",
        "state_id": STATE_IDS[0],
    }
    eager = {
        "conditions": [_condition(EAGER_FROZEN_ENCODED_CONTROL)],
        "metric_safe": True,
        "profile": "eager_numerical_control_not_strict_cuda_determinism",
        "state_id": STATE_IDS[0],
    }
    leaked = copy.deepcopy(default)
    leaked["conditions"][0]["token_ids"] = [1]
    with pytest.raises(ValueError, match="fields drifted"):
        merge_action_stability_state_v1(leaked, eager)
    reordered = copy.deepcopy(default)
    reordered["conditions"].reverse()
    with pytest.raises(ValueError, match="identity drifted"):
        merge_action_stability_state_v1(reordered, eager)


def test_aggregate_rejects_duplicates_extra_rows_and_forged_diagnosis() -> None:
    states = [_state(state_id) for state_id in STATE_IDS]
    with pytest.raises(TypeError, match="exact six-state"):
        aggregate_action_stability_diagnostic_v1([*states, states[0]])
    duplicated = [*states[:-1], states[0]]
    with pytest.raises(ValueError, match="roster drifted"):
        aggregate_action_stability_diagnostic_v1(duplicated)
    forged = copy.deepcopy(states)
    forged[0]["diagnosis"] = "PERSISTENT_GENERATION_INSTABILITY"
    with pytest.raises(ValueError, match="diagnosis drifted"):
        aggregate_action_stability_diagnostic_v1(forged)


def test_aggregate_revalidates_condition_counts_and_runtime_failure() -> None:
    states = [_state(state_id) for state_id in STATE_IDS]
    forged = copy.deepcopy(states)
    forged[0]["conditions"][0]["generation_call_count"] = 3
    with pytest.raises(ValueError, match="generation budget"):
        aggregate_action_stability_diagnostic_v1(forged)

    failed = copy.deepcopy(states)
    condition = failed[0]["conditions"][0]
    condition["canonical_action_equal"] = None
    condition["decoded_output_equal"] = None
    condition["exact_generated_sequence_equal"] = None
    condition["failure_class"] = "RuntimeError"
    condition["generation_call_count"] = 1
    condition["generation_completed_count"] = 0
    condition["encode_call_count"] = 1
    failed[0]["diagnosis"] = "INVALID_CONDITION_EXECUTION_FAILURE"
    assert (
        aggregate_action_stability_diagnostic_v1(failed)["verdict"]
        == "INVALID_RUNTIME_FAILURE"
    )


def test_successful_frozen_condition_requires_all_input_checks() -> None:
    states = [_state(state_id) for state_id in STATE_IDS]
    forged = copy.deepcopy(states)
    forged[0]["conditions"][1]["encoded_input_unchanged_between"] = None
    with pytest.raises(ValueError, match="requires unchanged inputs"):
        aggregate_action_stability_diagnostic_v1(forged)
