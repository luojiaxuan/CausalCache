from __future__ import annotations

import json
from pathlib import Path

from causalcache.set_utility_action_stability_diagnostic_v3 import (
    CONTROL_STATE_IDS,
    STATE_IDS,
    STRICT_DETERMINISM_FROZEN_ENCODED_CONDITION,
    STRICT_DETERMINISM_PROFILE,
    aggregate_action_stability_diagnostic_v3,
    build_strict_execution_failure_partial_v3,
    merge_action_stability_state_v3,
)


ROOT = Path(__file__).resolve().parents[2]


def _parents() -> dict[str, dict[str, object]]:
    payload = json.loads(
        (ROOT / "data/results/set_utility_action_stability_diagnostic_v2/aggregate.json").read_text()
    )
    return {state["state_id"]: state for state in payload["diagnostic"]["states"]}


def _partial(state_id: str, *, stable: bool = True) -> dict[str, object]:
    return {
        "conditions": [
            {
                "canonical_action_equal": stable,
                "condition_id": STRICT_DETERMINISM_FROZEN_ENCODED_CONDITION,
                "decoded_output_equal": stable,
                "encode_call_count": 1,
                "encoded_input_unchanged_after": True,
                "encoded_input_unchanged_before": True,
                "encoded_input_unchanged_between": True,
                "exact_generated_sequence_equal": stable,
                "failure_class": None,
                "generation_call_count": 2,
                "generation_completed_count": 2,
                "metric_safe": True,
                "repeat_count": 2,
            }
        ],
        "metric_safe": True,
        "profile": STRICT_DETERMINISM_PROFILE,
        "state_id": state_id,
    }


def _merged(*, target_stable: bool = True, unstable_control: bool = False):
    parents = _parents()
    return [
        merge_action_stability_state_v3(
            parents[state_id],
            _partial(
                state_id,
                stable=(
                    target_stable
                    if index == 0
                    else not (unstable_control and index == 1)
                ),
            ),
        )
        for index, state_id in enumerate(STATE_IDS)
    ]


def test_d2_roster_is_one_unstable_target_plus_two_stable_controls() -> None:
    parents = _parents()
    assert len(STATE_IDS) == 3
    assert len(CONTROL_STATE_IDS) == 2
    assert parents[STATE_IDS[0]]["sdpa_numerical_control_condition"][
        "canonical_action_equal"
    ] is False
    assert all(
        parents[state_id]["sdpa_numerical_control_condition"][
            "canonical_action_equal"
        ]
        is True
        for state_id in CONTROL_STATE_IDS
    )


def test_d2_pass_does_not_require_or_emit_any_utility() -> None:
    payload = aggregate_action_stability_diagnostic_v3(_merged())
    assert payload["verdict"] == (
        "PASS_STRICT_DETERMINISM_REPEAT_STABILITY_DIAGNOSTIC"
    )
    assert payload["counts"] == {
        "encode_call_count": 3,
        "generation_call_count": 6,
        "retry_count": 0,
        "state_process_count": 3,
    }
    def keys(value):
        if isinstance(value, dict):
            return set(value) | {key for item in value.values() for key in keys(item)}
        if isinstance(value, list):
            return {key for item in value for key in keys(item)}
        return set()

    assert "utility" not in keys(payload)


def test_d2_verdict_precedence_is_runtime_then_control_then_target() -> None:
    parents = _parents()
    target_unstable = aggregate_action_stability_diagnostic_v3(
        _merged(target_stable=False)
    )
    assert target_unstable["verdict"] == (
        "NO_GO_STRICT_DETERMINISM_REPEAT_INSTABILITY"
    )
    control_unstable = aggregate_action_stability_diagnostic_v3(
        _merged(target_stable=False, unstable_control=True)
    )
    assert control_unstable["verdict"] == "INVALID_STABLE_CONTROL_INSTABILITY"
    failed = _merged(target_stable=False, unstable_control=True)
    failed[0] = merge_action_stability_state_v3(
        parents[STATE_IDS[0]],
        build_strict_execution_failure_partial_v3(
            STATE_IDS[0], failure_class="RuntimeError"
        ),
    )
    assert aggregate_action_stability_diagnostic_v3(failed)["verdict"] == (
        "INVALID_RUNTIME_FAILURE"
    )
