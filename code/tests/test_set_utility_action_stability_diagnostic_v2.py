from __future__ import annotations

import copy

import pytest

import causalcache.set_utility_action_stability_diagnostic_v2 as diagnostic_v2
from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v1 import (
    AUTO_FRESH_ENCODE_CONDITION,
    AUTO_FROZEN_ENCODED_CONDITION,
    EAGER_FROZEN_ENCODED_CONTROL,
)
from causalcache.set_utility_action_stability_diagnostic_v1 import (
    merge_action_stability_state_v1,
)
from causalcache.set_utility_action_stability_diagnostic_v2 import (
    CONTROL_STATE_IDS,
    EXPECTED_ENCODE_CALL_CEILING,
    EXPECTED_GENERATION_CALL_CEILING,
    SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION,
    SDPA_NUMERICAL_CONTROL_PROFILE,
    STATE_IDS,
    STATE_WAVES,
    aggregate_action_stability_diagnostic_v2,
    build_sdpa_execution_failure_partial_v2,
    merge_action_stability_state_v2,
    run_sdpa_numerical_control_condition_v2,
    state_ids_for_wave_v2,
)


def _condition(
    condition_id: str,
    *,
    stable: bool = True,
    failure_class: str | None = None,
) -> dict[str, object]:
    prepared = condition_id != AUTO_FRESH_ENCODE_CONDITION
    failed = failure_class is not None
    return {
        "canonical_action_equal": None if failed else stable,
        "condition_id": condition_id,
        "decoded_output_equal": None if failed else stable,
        "encode_call_count": 1 if prepared or failed else 2,
        "encoded_input_unchanged_after": True if prepared else None,
        "encoded_input_unchanged_before": True if prepared else None,
        "encoded_input_unchanged_between": None if failed and prepared else (True if prepared else None),
        "exact_generated_sequence_equal": None if failed else stable,
        "failure_class": failure_class,
        "generation_call_count": 1 if failed else 2,
        "generation_completed_count": 0 if failed else 2,
        "metric_safe": True,
        "repeat_count": 2,
    }


def _parent_state(
    state_id: str,
    *,
    fresh: bool = True,
    frozen: bool = True,
    eager_failure: bool = False,
) -> dict[str, object]:
    default = {
        "conditions": [
            _condition(AUTO_FRESH_ENCODE_CONDITION, stable=fresh),
            _condition(AUTO_FROZEN_ENCODED_CONDITION, stable=frozen),
        ],
        "metric_safe": True,
        "profile": "auto_default",
        "state_id": state_id,
    }
    eager = {
        "conditions": [
            _condition(
                EAGER_FROZEN_ENCODED_CONTROL,
                failure_class="OutOfMemoryError" if eager_failure else None,
            )
        ],
        "metric_safe": True,
        "profile": "eager_numerical_control_not_strict_cuda_determinism",
        "state_id": state_id,
    }
    return merge_action_stability_state_v1(default, eager)


def _partial(
    state_id: str,
    *,
    stable: bool = True,
    failure_class: str | None = None,
) -> dict[str, object]:
    return {
        "conditions": [
            _condition(
                SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION,
                stable=stable,
                failure_class=failure_class,
            )
        ],
        "metric_safe": True,
        "profile": SDPA_NUMERICAL_CONTROL_PROFILE,
        "state_id": state_id,
    }


def _merged_states() -> list[dict[str, object]]:
    return [
        merge_action_stability_state_v2(_parent_state(state_id), _partial(state_id))
        for state_id in STATE_IDS
    ]


def test_d1b_reuses_exact_roster_and_freezes_four_plus_two_waves() -> None:
    assert len(STATE_IDS) == 6
    assert STATE_WAVES == (
        (STATE_IDS[0], STATE_IDS[3], STATE_IDS[4], STATE_IDS[5]),
        (STATE_IDS[1], STATE_IDS[2]),
    )
    assert state_ids_for_wave_v2(0) == (
        STATE_IDS[0],
        STATE_IDS[3],
        STATE_IDS[4],
        STATE_IDS[5],
    )
    assert state_ids_for_wave_v2(1) == (STATE_IDS[1], STATE_IDS[2])
    assert EXPECTED_GENERATION_CALL_CEILING == 12
    assert EXPECTED_ENCODE_CALL_CEILING == 6
    with pytest.raises(ValueError, match="zero or one"):
        state_ids_for_wave_v2(2)


def test_merge_uses_only_parent_auto_context_even_when_d1_eager_oomed() -> None:
    state_id = STATE_IDS[3]
    merged = merge_action_stability_state_v2(
        _parent_state(state_id, eager_failure=True),
        _partial(state_id),
    )
    assert [
        item["condition_id"] for item in merged["d1_auto_context"]["conditions"]
    ] == [AUTO_FRESH_ENCODE_CONDITION, AUTO_FROZEN_ENCODED_CONDITION]
    assert EAGER_FROZEN_ENCODED_CONTROL not in repr(merged)
    assert merged["diagnosis"] == (
        "PARENT_MISMATCH_NOT_REPRODUCED_UNDER_SDPA_CONTROL"
    )


def test_parent_projection_ignores_mutated_eager_row_and_old_diagnosis() -> None:
    state_id = STATE_IDS[3]
    parent = _parent_state(state_id, eager_failure=True)
    expected = merge_action_stability_state_v2(parent, _partial(state_id))
    mutated = copy.deepcopy(parent)
    mutated["conditions"][2] = {
        "untrusted_eager_payload": object(),
        "output_text": "must not be inspected",
    }
    mutated["diagnosis"] = "UNTRUSTED_OLD_DIAGNOSIS"
    mutated["worker_index"] = -999
    assert merge_action_stability_state_v2(mutated, _partial(state_id)) == expected


def test_merge_preserves_parent_path_context_without_changing_new_verdict() -> None:
    fresh_path = merge_action_stability_state_v2(
        _parent_state(STATE_IDS[0], fresh=False, frozen=True),
        _partial(STATE_IDS[0]),
    )
    frozen_path = merge_action_stability_state_v2(
        _parent_state(STATE_IDS[1], fresh=True, frozen=False),
        _partial(STATE_IDS[1]),
    )
    assert fresh_path["diagnosis"] == (
        "FRESH_VS_FROZEN_PATH_ASSOCIATION_WITH_SDPA_CONTROL_STABLE"
    )
    assert frozen_path["diagnosis"] == (
        "AUTO_VS_CONTROLLED_SDPA_PROFILE_ASSOCIATION"
    )


@pytest.mark.parametrize(
    ("state_id", "fresh", "frozen", "sdpa_stable", "expected"),
    [
        (
            STATE_IDS[0],
            True,
            False,
            False,
            "D1_AUTO_FROZEN_INSTABILITY_UNRESOLVED_BY_SDPA_CONTROL",
        ),
        (
            STATE_IDS[0],
            False,
            True,
            False,
            "MIXED_D1_FRESH_PATH_AND_SDPA_CONTROL_INSTABILITY",
        ),
        (
            STATE_IDS[0],
            True,
            True,
            False,
            "SDPA_CONTROL_ONLY_INSTABILITY",
        ),
        (
            STATE_IDS[0],
            True,
            False,
            True,
            "AUTO_VS_CONTROLLED_SDPA_PROFILE_ASSOCIATION",
        ),
        (
            STATE_IDS[0],
            False,
            True,
            True,
            "FRESH_VS_FROZEN_PATH_ASSOCIATION_WITH_SDPA_CONTROL_STABLE",
        ),
        (
            STATE_IDS[0],
            True,
            True,
            True,
            "PARENT_MISMATCH_NOT_REPRODUCED_UNDER_SDPA_CONTROL",
        ),
        (
            CONTROL_STATE_IDS[0],
            True,
            True,
            True,
            "STABLE_CONTROL_REPRODUCED",
        ),
        (
            CONTROL_STATE_IDS[0],
            True,
            True,
            False,
            "STABLE_CONTROL_SDPA_CONTROL_INSTABILITY",
        ),
    ],
)
def test_state_diagnosis_matrix_is_frozen(
    state_id: str,
    fresh: bool,
    frozen: bool,
    sdpa_stable: bool,
    expected: str,
) -> None:
    merged = merge_action_stability_state_v2(
        _parent_state(state_id, fresh=fresh, frozen=frozen),
        _partial(state_id, stable=sdpa_stable),
    )
    assert merged["diagnosis"] == expected


def test_aggregate_passes_only_when_all_six_new_conditions_are_stable() -> None:
    payload = aggregate_action_stability_diagnostic_v2(_merged_states())
    assert payload["verdict"] == (
        "PASS_MEMORY_SAFE_SDPA_NUMERICAL_CONTROL_REPEAT_STABILITY"
    )
    assert payload["counts"] == {
        "encode_call_ceiling": 6,
        "encode_call_count": 6,
        "generation_call_ceiling": 12,
        "generation_call_count": 12,
        "retry_count": 0,
        "state_count": 6,
    }
    assert payload["context"]["parent_calls_counted"] is False


def test_mismatch_instability_and_stable_control_instability_have_distinct_verdicts() -> None:
    mismatch_states = _merged_states()
    mismatch_states[0] = merge_action_stability_state_v2(
        _parent_state(STATE_IDS[0]),
        _partial(STATE_IDS[0], stable=False),
    )
    assert aggregate_action_stability_diagnostic_v2(mismatch_states)["verdict"] == (
        "NO_GO_SDPA_CONTROL_REPEAT_INSTABILITY"
    )

    control_states = _merged_states()
    control_id = CONTROL_STATE_IDS[0]
    control_states[STATE_IDS.index(control_id)] = merge_action_stability_state_v2(
        _parent_state(control_id),
        _partial(control_id, stable=False),
    )
    aggregate = aggregate_action_stability_diagnostic_v2(control_states)
    assert aggregate["verdict"] == "INVALID_STABLE_CONTROL_INSTABILITY"
    assert aggregate["states"][STATE_IDS.index(control_id)]["diagnosis"] == (
        "STABLE_CONTROL_SDPA_CONTROL_INSTABILITY"
    )


def test_runtime_failure_is_invalid_and_may_end_below_call_ceiling() -> None:
    states = _merged_states()
    states[0] = merge_action_stability_state_v2(
        _parent_state(STATE_IDS[0]),
        _partial(STATE_IDS[0], failure_class="OutOfMemoryError"),
    )
    payload = aggregate_action_stability_diagnostic_v2(states)
    assert payload["verdict"] == "INVALID_RUNTIME_FAILURE"
    assert payload["counts"]["generation_call_count"] == 11
    assert payload["states"][0]["diagnosis"] == (
        "INVALID_CONDITION_EXECUTION_FAILURE"
    )


def test_zero_call_outer_execution_failure_is_aggregated_without_invented_calls() -> None:
    states = _merged_states()
    failure_partial = build_sdpa_execution_failure_partial_v2(
        STATE_IDS[0],
        failure_class="RuntimeError",
    )
    condition = failure_partial["conditions"][0]
    assert condition["encode_call_count"] == 0
    assert condition["generation_call_count"] == 0
    assert condition["generation_completed_count"] == 0
    states[0] = merge_action_stability_state_v2(
        _parent_state(STATE_IDS[0]),
        failure_partial,
    )
    payload = aggregate_action_stability_diagnostic_v2(states)
    assert payload["verdict"] == "INVALID_RUNTIME_FAILURE"
    assert payload["counts"]["encode_call_count"] == 5
    assert payload["counts"]["generation_call_count"] == 10
    assert payload["states"][0]["diagnosis"] == (
        "INVALID_CONDITION_EXECUTION_FAILURE"
    )


def test_zero_call_outer_execution_failure_rejects_unsafe_or_forged_rows() -> None:
    with pytest.raises(ValueError, match="zero-call execution-failure"):
        build_sdpa_execution_failure_partial_v2(
            STATE_IDS[0],
            failure_class="RuntimeError: secret",
        )
    partial = build_sdpa_execution_failure_partial_v2(
        STATE_IDS[0],
        failure_class="RuntimeError",
    )
    partial["conditions"][0]["canonical_action_equal"] = False
    with pytest.raises(ValueError, match="zero-call execution-failure"):
        merge_action_stability_state_v2(_parent_state(STATE_IDS[0]), partial)
    boolean_counts = build_sdpa_execution_failure_partial_v2(
        STATE_IDS[0],
        failure_class="RuntimeError",
    )
    boolean_counts["conditions"][0]["encode_call_count"] = False
    with pytest.raises(ValueError, match="encode_call_count"):
        merge_action_stability_state_v2(_parent_state(STATE_IDS[0]), boolean_counts)


def test_merge_rejects_parent_auto_failure_and_cross_state_pairing() -> None:
    parent = _parent_state(STATE_IDS[0])
    failed_parent = copy.deepcopy(parent)
    condition = failed_parent["conditions"][0]
    condition.update(_condition(AUTO_FRESH_ENCODE_CONDITION, failure_class="RuntimeError"))
    failed_parent["diagnosis"] = "INVALID_CONDITION_EXECUTION_FAILURE"
    with pytest.raises(ValueError, match="complete D1 auto-condition context"):
        merge_action_stability_state_v2(failed_parent, _partial(STATE_IDS[0]))
    with pytest.raises(ValueError, match="state identities differ"):
        merge_action_stability_state_v2(parent, _partial(STATE_IDS[1]))


def test_public_runner_builds_exact_input_once_and_runs_only_new_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQuery:
        split = "train"
        state_id = STATE_IDS[0]
        candidate_event_step_ids = (1, 2)

    query = FakeQuery()
    runtime = object()
    events: list[object] = []
    monkeypatch.setattr(diagnostic_v2, "UtilityQuerySpec", FakeQuery)
    monkeypatch.setattr(
        diagnostic_v2,
        "build_mixed_fidelity_prompt_plan",
        lambda observed, candidates: events.append((observed, candidates)) or "plan",
    )
    monkeypatch.setattr(
        diagnostic_v2,
        "_run_frozen_sdpa_condition_v2",
        lambda observed_runtime, reference_input: events.append(
            (observed_runtime, reference_input)
        )
        or _condition(SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION),
    )
    payload = run_sdpa_numerical_control_condition_v2(
        query,
        reference_input_builder=lambda plan: events.append(plan) or "reference-input",
        runtime=runtime,
    )
    assert events == [
        (query, (1, 2)),
        "plan",
        (runtime, "reference-input"),
    ]
    assert payload["profile"] == SDPA_NUMERICAL_CONTROL_PROFILE
    assert len(payload["conditions"]) == 1
    assert payload["conditions"][0]["condition_id"] == (
        SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION
    )
