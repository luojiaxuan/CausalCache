from __future__ import annotations

import dataclasses
import json
import types
from collections.abc import Callable

import pytest

from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1_throughput_runtime import (
    GUIOwlV21ThroughputRuntime,
)
from causalcache.set_utility_gui_owl_v2_1_throughput_adapter import (
    GUIOwlV21SetUtilityThroughputAdapter,
    GUIOwlV21ThroughputReferenceInput,
    build_gui_owl_v2_1_throughput_reference_input,
)
from causalcache.set_utility_label_inputs import JoinedUtilityQueryInput
from causalcache.set_utility_label_producer import (
    MixedFidelityPromptPlan,
    UtilityHistoryEvent,
    UtilityQuerySpec,
    build_mixed_fidelity_prompt_plan,
    freeze_recent_candidates,
)
from causalcache.set_utility_throughput_pilot import (
    PilotReferenceGeneration,
    PilotReferenceTeacherForward,
    PilotRuntimeFailure,
    run_train_only_set_utility_throughput_pilot,
)


def _low(step_id: int) -> LowFidelityEventV2:
    return LowFidelityEventV2(
        step_id=step_id,
        action_type="wait",
        action_argument="wait",
        foreground_app="fixture.app",
        screen_text_added=(f"screen-{step_id}",),
        screen_text_removed=(),
        screen_change="low",
        executor_result="accepted",
    )


def _query() -> UtilityQuerySpec:
    history = tuple(
        UtilityHistoryEvent(
            event_step_id=step_id,
            low_fidelity_summary=_low(step_id),
            high_fidelity_observation_ref=f"fixture://post-{step_id:03d}.png",
        )
        for step_id in range(1, 6)
    )
    context = freeze_recent_candidates(
        (1, 2, 3, 4),
        processor_only_length=lambda candidates: 10 + len(candidates),
        reserved_action_tokens=8,
        context_limit=64,
    )
    return UtilityQuerySpec(
        split="train",
        trajectory_id="trajectory-adapter",
        source_id="source-adapter",
        state_id="state-adapter",
        instruction_app_group_sha256="a" * 64,
        task_instruction="Open the fixture and wait.",
        decision_step_id=6,
        history_events=history,
        current_equivalent_event_step_id=5,
        candidate_context=context,
        maximum_labeled_cardinality=2,
        current_observation_ref="fixture://post-005.png",
        source_artifact_sha256="b" * 64,
        request_manifest_sha256="c" * 64,
        slice_witness_sha256="d" * 64,
    )


def _joined() -> JoinedUtilityQueryInput:
    query = _query()
    payloads = {
        event.high_fidelity_observation_ref: f"payload-{event.event_step_id}".encode()
        for event in query.history_events
    }
    return JoinedUtilityQueryInput(
        query=query,
        image_payloads=payloads,
        processor_worker_index=0,
        execution_worker_index=1,
        role_partition="train",
        processor_artifact_sha256=query.source_artifact_sha256,
        query_record_sha256=query.slice_witness_sha256,
    )


class _Clock:
    def __init__(self, events: list[str], values: tuple[float, ...]) -> None:
        self.events = events
        self.values = iter(values)

    def __call__(self) -> float:
        self.events.append("clock")
        return next(self.values)


class _FakeCuda:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def synchronize(self, device: object) -> None:
        assert device == "cuda:3"
        self.events.append("synchronize")

    def reset_peak_memory_stats(self, device: object) -> None:
        assert device == "cuda:3"
        self.events.append("reset_peak")

    def max_memory_allocated(self, device: object) -> int:
        assert device == "cuda:3"
        self.events.append("max_allocated")
        return 123_456

    def max_memory_reserved(self, device: object) -> int:
        assert device == "cuda:3"
        self.events.append("max_reserved")
        return 234_567


class _Disposable:
    def __init__(self, events: list[str], label: str) -> None:
        self.events = events
        self.label = label

    def __del__(self) -> None:
        self.events.append(f"dispose_{self.label}")


class SensitiveNativeError(RuntimeError):
    pass


class _FakeRuntime(GUIOwlV21ThroughputRuntime):
    def __init__(
        self,
        events: list[str],
        *,
        action: object = GUIOwlV2Action(action="wait"),
        generation_error: Exception | None = None,
        teacher_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.device = "cuda:3"
        self.torch = types.SimpleNamespace(cuda=_FakeCuda(events))
        self.action = action
        self.generation_error = generation_error
        self.teacher_error = teacher_error
        self.generation_owners: list[str] = []
        self.teacher_owners: list[str] = []

    def generate_native_action(
        self,
        messages: object,
        *,
        cuda_peak_measurement_owner: str = "runtime",
    ) -> object:
        assert isinstance(messages, tuple)
        self.generation_owners.append(cuda_peak_measurement_owner)
        for event in ("encode", "h2d", "generation_prep", "forward", "decode"):
            self.events.append(event)
        if self.generation_error is not None:
            raise self.generation_error
        return types.SimpleNamespace(
            output_text="SECRET_NATIVE_OUTPUT",
            parsed_output=types.SimpleNamespace(canonical_action=self.action),
            metadata={"tokens": [1, 2], "secret": "SECRET_METADATA"},
        )

    def teacher_forced_distance_logits(
        self,
        messages_batch: object,
        actions: object,
        *,
        cuda_peak_measurement_owner: str = "runtime",
    ) -> tuple[object, object]:
        assert isinstance(messages_batch, tuple)
        assert isinstance(actions, tuple)
        self.teacher_owners.append(cuda_peak_measurement_owner)
        for event in ("encode", "h2d", "teacher_prep", "forward"):
            self.events.append(event)
        if self.teacher_error is not None:
            raise self.teacher_error
        return (
            _Disposable(self.events, "logits"),
            _Disposable(self.events, "metadata"),
        )


def _reference_input() -> GUIOwlV21ThroughputReferenceInput:
    return GUIOwlV21ThroughputReferenceInput(
        messages=({"role": "user", "content": ()},),
    )


def _adapter(
    runtime: _FakeRuntime,
    events: list[str],
    *,
    clock_values: tuple[float, ...] = (10.0, 10.75),
) -> GUIOwlV21SetUtilityThroughputAdapter:
    return GUIOwlV21SetUtilityThroughputAdapter(
        runtime,
        clock=_Clock(events, clock_values),
    )


def test_builder_uses_only_exact_joined_payloads_in_frozen_plan_order() -> None:
    joined = _joined()
    plan = build_mixed_fidelity_prompt_plan(joined.query, (2, 4))
    decoded: list[bytes] = []

    reference_input = build_gui_owl_v2_1_throughput_reference_input(
        joined,
        plan,
        image_decoder=lambda payload: decoded.append(payload) or payload.decode(),
    )

    assert isinstance(reference_input, GUIOwlV21ThroughputReferenceInput)
    assert decoded == [
        joined.image_payloads["fixture://post-002.png"],
        joined.image_payloads["fixture://post-004.png"],
        joined.image_payloads["fixture://post-005.png"],
    ]
    images = [
        item["image"]
        for item in reference_input.messages[1]["content"]
        if item["type"] == "image"
    ]
    assert images == ["payload-2", "payload-4", "payload-5"]


def test_builder_rejects_plan_drift_and_join_rejects_payload_inventory_drift() -> None:
    joined = _joined()
    plan = build_mixed_fidelity_prompt_plan(joined.query, (2,))
    drifted = dataclasses.replace(plan, task_instruction="Different task")
    with pytest.raises(ValueError, match="differs"):
        build_gui_owl_v2_1_throughput_reference_input(
            joined,
            drifted,
            image_decoder=lambda payload: payload,
        )
    with pytest.raises(ValueError, match="inventory"):
        dataclasses.replace(
            joined,
            image_payloads={
                key: value
                for key, value in joined.image_payloads.items()
                if key != "fixture://post-001.png"
            },
        )


def test_generation_outer_measurement_covers_full_call_and_uses_caller_owner() -> None:
    events: list[str] = []
    runtime = _FakeRuntime(events)
    outcome = _adapter(runtime, events).generate_reference_action(_reference_input())

    assert isinstance(outcome, PilotReferenceGeneration)
    assert type(outcome.action_handle) is GUIOwlV2Action
    assert outcome.action_handle == GUIOwlV2Action(action="wait")
    assert runtime.generation_owners == ["caller"]
    assert events == [
        "synchronize",
        "reset_peak",
        "clock",
        "encode",
        "h2d",
        "generation_prep",
        "forward",
        "decode",
        "synchronize",
        "clock",
        "max_allocated",
        "max_reserved",
    ]
    assert outcome.performance.end_to_end_wall_seconds == 0.75
    assert outcome.performance.full_call_cuda_peak_allocated_bytes == 123_456
    assert outcome.performance.full_call_cuda_peak_reserved_bytes == 234_567


def test_teacher_discards_logits_and_metadata_before_outer_stop_and_projects_metrics() -> None:
    events: list[str] = []
    runtime = _FakeRuntime(events)
    adapter = _adapter(runtime, events)
    reference_input = _reference_input()
    action = GUIOwlV2Action(action="wait")

    outcome = adapter.teacher_force_reference(
        (reference_input, reference_input),
        (action, action),
    )

    assert isinstance(outcome, PilotReferenceTeacherForward)
    assert outcome.example_count == 2
    assert runtime.teacher_owners == ["caller"]
    assert events == [
        "synchronize",
        "reset_peak",
        "clock",
        "encode",
        "h2d",
        "teacher_prep",
        "forward",
        "dispose_logits",
        "dispose_metadata",
        "synchronize",
        "clock",
        "max_allocated",
        "max_reserved",
    ]
    assert set(dataclasses.asdict(outcome)) == {"example_count", "performance"}


def test_native_failure_becomes_class_only_metric_failure_without_output_leak() -> None:
    events: list[str] = []
    runtime = _FakeRuntime(
        events,
        generation_error=SensitiveNativeError(
            "output_text=SECRET logits=SECRET tokens=[1] utility=SECRET"
        ),
    )
    outcome = _adapter(runtime, events).generate_reference_action(_reference_input())

    assert isinstance(outcome, PilotRuntimeFailure)
    assert outcome.error_class_identifier == "SensitiveNativeError"
    assert outcome.performance.end_to_end_wall_seconds == 0.75
    serialized = json.dumps(dataclasses.asdict(outcome), sort_keys=True).lower()
    for forbidden in ("secret", "output_text", "logits", "tokens", "utility"):
        assert forbidden not in serialized
    assert events[-4:] == [
        "synchronize",
        "clock",
        "max_allocated",
        "max_reserved",
    ]


def test_only_canonical_action_handles_cross_the_adapter_boundary() -> None:
    events: list[str] = []
    runtime = _FakeRuntime(events, action="SECRET_NON_CANONICAL_ACTION")
    adapter = _adapter(runtime, events)

    generation = adapter.generate_reference_action(_reference_input())
    assert isinstance(generation, PilotRuntimeFailure)
    assert generation.error_class_identifier == "TypeError"
    with pytest.raises(TypeError, match="GUIOwlV2Action"):
        adapter.reference_actions_equal(object(), GUIOwlV2Action(action="wait"))

    events.clear()
    teacher = _adapter(runtime, events).teacher_force_reference(
        (_reference_input(),),
        ("SECRET_NON_CANONICAL_ACTION",),
    )
    assert isinstance(teacher, PilotRuntimeFailure)
    assert teacher.error_class_identifier == "TypeError"
    assert runtime.teacher_owners == []
    assert "SECRET_NON_CANONICAL_ACTION" not in json.dumps(
        dataclasses.asdict(teacher), sort_keys=True
    )


def test_adapter_requires_versioned_runtime_and_sanitizes_unsafe_exception_class() -> None:
    with pytest.raises(TypeError, match="GUIOwlV21ThroughputRuntime"):
        GUIOwlV21SetUtilityThroughputAdapter(object())

    unsafe_error_type = type("Bad:error=SECRET", (RuntimeError,), {})
    events: list[str] = []
    runtime = _FakeRuntime(
        events,
        teacher_error=unsafe_error_type("output=SECRET"),
    )
    outcome = _adapter(runtime, events).teacher_force_reference(
        (_reference_input(),),
        (GUIOwlV2Action(action="wait"),),
    )
    assert isinstance(outcome, PilotRuntimeFailure)
    assert outcome.error_class_identifier == "UnexpectedException"
    assert "secret" not in json.dumps(dataclasses.asdict(outcome)).lower()


def test_measurement_failure_discards_successful_native_result() -> None:
    events: list[str] = []
    runtime = _FakeRuntime(events)

    def decreasing_clock() -> Callable[[], float]:
        values = iter((10.0, 9.0))
        return lambda: next(values)

    adapter = GUIOwlV21SetUtilityThroughputAdapter(
        runtime,
        clock=decreasing_clock(),
    )
    outcome = adapter.generate_reference_action(_reference_input())

    assert isinstance(outcome, PilotRuntimeFailure)
    assert outcome.error_class_identifier == "ThroughputMeasurementError"
    assert outcome.performance.end_to_end_wall_seconds == 0.0


def test_reference_input_requires_exact_opaque_type_inside_measured_boundary() -> None:
    events: list[str] = []
    runtime = _FakeRuntime(events)
    outcome = _adapter(runtime, events).generate_reference_action(
        {"messages": "SECRET"}
    )
    assert isinstance(outcome, PilotRuntimeFailure)
    assert outcome.error_class_identifier == "TypeError"
    assert runtime.generation_owners == []


def test_adapter_runs_inside_metric_only_pilot_without_native_payload_escape() -> None:
    joined = _joined()
    events: list[str] = []
    runtime = _FakeRuntime(events)
    adapter = _adapter(
        runtime,
        events,
        clock_values=(1.0, 1.1, 2.0, 2.2, 3.0, 3.3),
    )

    result = run_train_only_set_utility_throughput_pilot(
        joined.query,
        reference_input_builder=lambda plan: (
            build_gui_owl_v2_1_throughput_reference_input(
                joined,
                plan,
                image_decoder=lambda payload: payload.decode(),
            )
        ),
        runtime=adapter,
        reference_teacher_microbatch_size=2,
    )

    assert result["failure_class"] is None
    assert result["counts"]["reference_generation_completed_count"] == 2
    assert result["counts"]["reference_teacher_forward_completed_call_count"] == 1
    assert runtime.generation_owners == ["caller", "caller"]
    assert runtime.teacher_owners == ["caller"]
    serialized = json.dumps(result, sort_keys=True).lower()
    for forbidden in (
        "secret",
        "output_text",
        "metadata",
        "token",
        "logit",
        "action",
        "utility",
    ):
        assert forbidden not in serialized
