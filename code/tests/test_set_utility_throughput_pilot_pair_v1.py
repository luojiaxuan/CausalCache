from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.set_utility_label_producer import (
    MixedFidelityPromptPlan,
    UtilityHistoryEvent,
    UtilityQuerySpec,
    freeze_recent_candidates,
)
from causalcache.set_utility_throughput_pilot import (
    PilotReferenceGeneration,
    PilotReferenceTeacherForward,
    PilotRuntimeFailure,
    PilotRuntimePerformance,
)
from causalcache.set_utility_throughput_pilot_pair_v1 import (
    EXPECTED_SUCCESS_COUNTS_FOR_12_STATES,
    EXPECTED_SUCCESS_COUNTS_PER_STATE,
    EXPECTED_SUCCESS_NATIVE_CALLS_FOR_12_STATES,
    aggregate_train_only_set_utility_throughput_pilot_pairs_v1,
    run_train_only_set_utility_throughput_pilot_pair_v1,
)


def _history_event(step_id: int) -> UtilityHistoryEvent:
    return UtilityHistoryEvent(
        event_step_id=step_id,
        low_fidelity_summary=LowFidelityEventV2(
            step_id=step_id,
            action_type="wait",
            action_argument="wait",
            foreground_app="fixture.app",
            screen_text_added=(f"screen-{step_id}",),
            screen_text_removed=(),
            screen_change="low",
            executor_result="accepted",
        ),
        high_fidelity_observation_ref=f"fixture://post-{step_id:03d}.png",
    )


def _query(index: int = 0, *, split: str = "train") -> UtilityQuerySpec:
    context = freeze_recent_candidates(
        (1, 2, 3, 4),
        processor_only_length=lambda candidates: 20 + len(candidates),
        reserved_action_tokens=8,
        context_limit=32,
    )
    return UtilityQuerySpec(
        split=split,
        trajectory_id=f"trajectory-pair-{index:02d}",
        source_id=f"source-pair-{index:02d}",
        state_id=f"state-pair-{index:02d}",
        instruction_app_group_sha256="a" * 64,
        task_instruction="Open the fixture and wait.",
        decision_step_id=6,
        history_events=tuple(_history_event(step_id) for step_id in range(1, 6)),
        current_equivalent_event_step_id=5,
        candidate_context=context,
        maximum_labeled_cardinality=2,
        current_observation_ref="fixture://post-005.png",
        source_artifact_sha256="b" * 64,
        request_manifest_sha256="c" * 64,
        slice_witness_sha256="d" * 64,
    )


def _performance(call_index: int) -> PilotRuntimePerformance:
    return PilotRuntimePerformance(
        end_to_end_wall_seconds=call_index / 10,
        full_call_cuda_peak_allocated_bytes=call_index * 100,
        full_call_cuda_peak_reserved_bytes=call_index * 100 + 50,
    )


@dataclass(frozen=True)
class _OpaqueAction:
    equality_key: str
    secret: str


class _FakeRuntime:
    def __init__(
        self,
        generation_keys: tuple[str, ...] = ("same", "same", "same", "same"),
        *,
        projected_generation_failure_call: int | None = None,
        equality_error_call: int | None = None,
    ) -> None:
        self.generation_keys = generation_keys
        self.projected_generation_failure_call = projected_generation_failure_call
        self.equality_error_call = equality_error_call
        self.generation_calls = 0
        self.teacher_batch_sizes: list[int] = []
        self.equality_calls = 0

    def generate_reference_action(
        self,
        reference_input: object,
    ) -> PilotReferenceGeneration | PilotRuntimeFailure:
        assert reference_input == "native-reference-input"
        self.generation_calls += 1
        performance = _performance(self.generation_calls)
        if self.generation_calls == self.projected_generation_failure_call:
            return PilotRuntimeFailure(
                error_class_identifier="CudaOutOfMemoryError",
                performance=performance,
            )
        key = self.generation_keys[self.generation_calls - 1]
        return PilotReferenceGeneration(
            action_handle=_OpaqueAction(
                equality_key=key,
                secret=f"SECRET_ACTION_{key}_{self.generation_calls}",
            ),
            performance=performance,
        )

    def reference_actions_equal(self, left: object, right: object) -> bool:
        self.equality_calls += 1
        if self.equality_calls == self.equality_error_call:
            raise SensitiveEqualityError("SECRET_EXCEPTION_MESSAGE tokens=[1,2]")
        assert isinstance(left, _OpaqueAction)
        assert isinstance(right, _OpaqueAction)
        return left.equality_key == right.equality_key

    def teacher_force_reference(
        self,
        reference_inputs: tuple[object, ...],
        action_handles: tuple[object, ...],
    ) -> PilotReferenceTeacherForward:
        assert reference_inputs == ("native-reference-input",) * len(reference_inputs)
        assert len(action_handles) == len(reference_inputs)
        assert all(isinstance(action, _OpaqueAction) for action in action_handles)
        self.teacher_batch_sizes.append(len(reference_inputs))
        call_index = self.generation_calls + len(self.teacher_batch_sizes)
        return PilotReferenceTeacherForward(
            example_count=len(reference_inputs),
            performance=_performance(call_index),
        )


class SensitiveEqualityError(RuntimeError):
    pass


def _run(
    runtime: _FakeRuntime,
    *,
    query: UtilityQuerySpec | None = None,
) -> tuple[dict[str, object], list[MixedFidelityPromptPlan]]:
    plans: list[MixedFidelityPromptPlan] = []

    def build(plan: MixedFidelityPromptPlan) -> object:
        plans.append(plan)
        return "native-reference-input"

    result = run_train_only_set_utility_throughput_pilot_pair_v1(
        query or _query(),
        reference_input_builder=build,
        runtime=runtime,
    )
    return result, plans


def test_pair_runs_mb1_then_mb2_and_keeps_opaque_actions_out_of_output() -> None:
    runtime = _FakeRuntime()
    result, plans = _run(runtime)

    assert len(plans) == 2
    assert runtime.generation_calls == 4
    assert runtime.teacher_batch_sizes == [1, 1, 2]
    assert runtime.equality_calls == 3
    assert result["variant_order"] == [1, 2]
    assert result["cross_variant_reference_action_equal"] is True
    assert result["failure_class"] is None
    assert result["retry_count"] == 0
    assert result["counts"] == EXPECTED_SUCCESS_COUNTS_PER_STATE
    assert result["expected_success_counts_for_12_states"] == (
        EXPECTED_SUCCESS_COUNTS_FOR_12_STATES
    )
    assert result["variants"]["1"]["counts"][
        "reference_teacher_forward_call_count"
    ] == 2
    assert result["variants"]["2"]["counts"][
        "reference_teacher_forward_call_count"
    ] == 1
    serialized = json.dumps(result, allow_nan=False, sort_keys=True)
    assert "SECRET_ACTION" not in serialized
    assert "equality_key" not in serialized


def test_cross_variant_mismatch_stops_before_mb2_teacher_and_does_not_retry() -> None:
    runtime = _FakeRuntime(("left", "left", "right", "right"))
    result, _ = _run(runtime)

    assert runtime.generation_calls == 4
    assert runtime.teacher_batch_sizes == [1, 1]
    assert runtime.equality_calls == 3
    assert result["cross_variant_reference_action_equal"] is False
    assert result["failure_class"] == "CROSS_VARIANT_REFERENCE_ACTION_MISMATCH"
    assert result["retry_count"] == 0
    assert result["counts"]["native_call_count"] == 6
    assert result["counts"]["pair_completed_count"] == 0
    serialized = json.dumps(result, allow_nan=False, sort_keys=True)
    assert "SECRET_ACTION_left" not in serialized
    assert "SECRET_ACTION_right" not in serialized


def test_first_failure_does_not_start_second_variant_or_leak_failure_details() -> None:
    runtime = _FakeRuntime(projected_generation_failure_call=1)
    result, plans = _run(runtime)

    assert len(plans) == 1
    assert runtime.generation_calls == 1
    assert runtime.teacher_batch_sizes == []
    assert result["variants"]["2"] is None
    assert result["failure_class"] == (
        "MICROBATCH_1__REFERENCE_GENERATION__CudaOutOfMemoryError"
    )
    assert result["retry_count"] == 0
    assert result["counts"]["native_call_count"] == 1


def test_sensitive_cross_variant_comparison_error_serializes_class_only() -> None:
    runtime = _FakeRuntime(equality_error_call=3)
    result, _ = _run(runtime)

    assert runtime.generation_calls == 4
    assert runtime.teacher_batch_sizes == [1, 1]
    assert result["failure_class"] == (
        "MICROBATCH_2__REFERENCE_REPEAT_COMPARISON__SensitiveEqualityError"
    )
    serialized = json.dumps(result, allow_nan=False, sort_keys=True)
    assert "SECRET_EXCEPTION_MESSAGE" not in serialized
    assert "tokens" not in serialized


def test_exact_12_state_aggregate_closes_the_84_native_call_contract() -> None:
    pairs = [_run(_FakeRuntime(), query=_query(index))[0] for index in range(12)]

    aggregate = aggregate_train_only_set_utility_throughput_pilot_pairs_v1(pairs)

    assert aggregate["state_count"] == 12
    assert aggregate["all_pairs_successful"] is True
    assert aggregate["failure_class_counts"] == {}
    assert aggregate["counts"] == EXPECTED_SUCCESS_COUNTS_FOR_12_STATES
    assert aggregate["counts"]["native_call_count"] == 84
    assert aggregate["expected_success_native_call_count"] == (
        EXPECTED_SUCCESS_NATIVE_CALLS_FOR_12_STATES
    )
    serialized = json.dumps(aggregate, allow_nan=False, sort_keys=True)
    assert "SECRET_ACTION" not in serialized


def test_aggregate_rejects_wrong_denominator_and_duplicate_state() -> None:
    pairs = [_run(_FakeRuntime(), query=_query(index))[0] for index in range(12)]
    with pytest.raises(ValueError, match="exactly 12"):
        aggregate_train_only_set_utility_throughput_pilot_pairs_v1(pairs[:-1])
    pairs[-1] = pairs[0]
    with pytest.raises(ValueError, match="not unique"):
        aggregate_train_only_set_utility_throughput_pilot_pairs_v1(pairs)


def test_aggregate_rejects_unsafe_failure_text() -> None:
    pairs = [_run(_FakeRuntime(), query=_query(index))[0] for index in range(12)]
    pairs[0] = {**pairs[0], "failure_class": "SECRET message tokens=[1,2]"}
    with pytest.raises(ValueError, match="failure class"):
        aggregate_train_only_set_utility_throughput_pilot_pairs_v1(pairs)


def test_aggregate_rejects_extra_or_nested_sensitive_fields() -> None:
    pairs = [_run(_FakeRuntime(), query=_query(index))[0] for index in range(12)]
    pairs[0] = {**pairs[0], "action": "SECRET_ACTION"}
    with pytest.raises(ValueError, match="fields drifted"):
        aggregate_train_only_set_utility_throughput_pilot_pairs_v1(pairs)

    pairs = [_run(_FakeRuntime(), query=_query(index))[0] for index in range(12)]
    pairs[0]["variants"]["1"] = {
        **pairs[0]["variants"]["1"],
        "decoded_output": "SECRET_ACTION",
    }
    with pytest.raises(ValueError, match="core result fields drifted"):
        aggregate_train_only_set_utility_throughput_pilot_pairs_v1(pairs)


def test_aggregate_rejects_safe_but_false_failure_class() -> None:
    pairs = [_run(_FakeRuntime(), query=_query(index))[0] for index in range(12)]
    pairs[0] = {**pairs[0], "failure_class": "MICROBATCH_2__CudaError"}
    with pytest.raises(ValueError, match="failure differs"):
        aggregate_train_only_set_utility_throughput_pilot_pairs_v1(pairs)


def test_non_train_query_fails_before_any_runtime_call() -> None:
    runtime = _FakeRuntime()
    with pytest.raises(ValueError, match="train queries only"):
        _run(runtime, query=_query(split="tune"))
    assert runtime.generation_calls == 0
    assert runtime.teacher_batch_sizes == []
