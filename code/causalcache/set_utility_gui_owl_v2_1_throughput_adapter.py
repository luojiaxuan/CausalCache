"""Versioned GUI-Owl v2.1 adapter for the metric-only throughput pilot."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1_throughput_runtime import (
    GUIOwlV21ThroughputRuntime,
)
from causalcache.set_utility_label_inputs import JoinedUtilityQueryInput
from causalcache.set_utility_label_producer import (
    MixedFidelityPromptPlan,
    build_mixed_fidelity_prompt_plan,
)
from causalcache.set_utility_processor_prompt import (
    build_set_utility_gui_owl_v2_1_messages,
)
from causalcache.set_utility_throughput_pilot import (
    PilotReferenceGeneration,
    PilotReferenceTeacherForward,
    PilotRuntimeFailure,
    PilotRuntimePerformance,
)


_SAFE_ERROR_CLASS_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class GUIOwlV21ThroughputReferenceInput:
    """Opaque native messages retained only inside the pilot process."""

    messages: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple) or not self.messages:
            raise ValueError("GUI-Owl throughput messages must be a non-empty tuple")
        if any(not isinstance(message, Mapping) for message in self.messages):
            raise TypeError("GUI-Owl throughput messages contain a non-mapping")


@dataclass(frozen=True, slots=True)
class _MeasuredCall(Generic[_T]):
    value: _T | None
    failure_identifier: str | None
    performance: PilotRuntimePerformance

    def __post_init__(self) -> None:
        if (self.failure_identifier is None) == (self.value is None):
            raise RuntimeError("measured call must contain exactly one value or failure")


class ThroughputMeasurementError(RuntimeError):
    """Safe class-only identity for an invalid outer measurement."""


def _safe_error_class(error: Exception) -> str:
    identifier = error.__class__.__name__
    if _SAFE_ERROR_CLASS_IDENTIFIER.fullmatch(identifier) is None:
        return "UnexpectedException"
    return identifier


def _zero_performance() -> PilotRuntimePerformance:
    return PilotRuntimePerformance(
        end_to_end_wall_seconds=0.0,
        full_call_cuda_peak_allocated_bytes=0,
        full_call_cuda_peak_reserved_bytes=0,
    )


def build_gui_owl_v2_1_throughput_reference_input(
    joined: JoinedUtilityQueryInput,
    plan: MixedFidelityPromptPlan,
    *,
    image_decoder: Callable[[bytes], Any],
) -> GUIOwlV21ThroughputReferenceInput:
    """Render one exact joined query without copying or widening its image inventory."""
    if not isinstance(joined, JoinedUtilityQueryInput):
        raise TypeError("throughput input requires JoinedUtilityQueryInput")
    if not isinstance(plan, MixedFidelityPromptPlan):
        raise TypeError("throughput input requires MixedFidelityPromptPlan")
    if not callable(image_decoder):
        raise TypeError("throughput image decoder must be callable")
    expected_plan = build_mixed_fidelity_prompt_plan(
        joined.query,
        plan.restored_event_step_ids,
    )
    if plan != expected_plan:
        raise ValueError("throughput prompt plan differs from its joined utility query")

    loaded_references: list[str] = []

    def load_image(reference: str) -> bytes:
        if reference not in joined.image_payloads:
            raise ValueError("throughput prompt requested an image outside the joined input")
        payload = joined.image_payloads[reference]
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("throughput prompt image payload is invalid")
        loaded_references.append(reference)
        return payload

    messages = build_set_utility_gui_owl_v2_1_messages(
        plan,
        image_bytes_loader=load_image,
        image_decoder=image_decoder,
    )
    expected_loaded_references = (
        *(
            event.high_fidelity_observation_ref
            for event in plan.history_events
            if event.high_fidelity_observation_ref is not None
        ),
        plan.current_observation_ref,
    )
    if tuple(loaded_references) != expected_loaded_references:
        raise RuntimeError("throughput prompt image load order drifted")
    return GUIOwlV21ThroughputReferenceInput(messages=tuple(messages))


class GUIOwlV21SetUtilityThroughputAdapter:
    """Metric-only adapter around the versioned caller-owned-peak runtime."""

    def __init__(
        self,
        runtime: GUIOwlV21ThroughputRuntime,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not isinstance(runtime, GUIOwlV21ThroughputRuntime):
            raise TypeError("adapter requires GUIOwlV21ThroughputRuntime")
        if not callable(clock):
            raise TypeError("throughput adapter clock must be callable")
        self._runtime = runtime
        self._clock = clock

    def _measured_call(self, operation: Callable[[], _T]) -> _MeasuredCall[_T]:
        try:
            cuda = self._runtime.torch.cuda
            device = self._runtime.device
            cuda.synchronize(device)
            cuda.reset_peak_memory_stats(device)
            started = self._clock()
            if (
                isinstance(started, bool)
                or not isinstance(started, (int, float))
                or not math.isfinite(float(started))
            ):
                raise ThroughputMeasurementError
        except Exception as error:
            return _MeasuredCall(
                value=None,
                failure_identifier=_safe_error_class(error),
                performance=_zero_performance(),
            )

        value: _T | None = None
        failure_identifier: str | None = None
        try:
            value = operation()
        except Exception as error:
            failure_identifier = _safe_error_class(error)

        try:
            cuda.synchronize(device)
        except Exception as error:
            if failure_identifier is None:
                failure_identifier = _safe_error_class(error)

        try:
            finished = self._clock()
            if (
                isinstance(finished, bool)
                or not isinstance(finished, (int, float))
                or not math.isfinite(float(finished))
            ):
                raise ThroughputMeasurementError
            latency = float(finished) - float(started)
            if not math.isfinite(latency) or latency < 0.0:
                raise ThroughputMeasurementError
        except Exception as error:
            latency = 0.0
            if failure_identifier is None:
                failure_identifier = _safe_error_class(error)

        try:
            allocated = cuda.max_memory_allocated(device)
            reserved = cuda.max_memory_reserved(device)
            if (
                type(allocated) is not int
                or allocated < 0
                or type(reserved) is not int
                or reserved < allocated
            ):
                raise ThroughputMeasurementError
        except Exception as error:
            allocated = 0
            reserved = 0
            if failure_identifier is None:
                failure_identifier = _safe_error_class(error)

        performance = PilotRuntimePerformance(
            end_to_end_wall_seconds=latency,
            full_call_cuda_peak_allocated_bytes=allocated,
            full_call_cuda_peak_reserved_bytes=reserved,
        )
        if failure_identifier is not None:
            value = None
        return _MeasuredCall(
            value=value,
            failure_identifier=failure_identifier,
            performance=performance,
        )

    def generate_reference_action(
        self,
        reference_input: object,
    ) -> PilotReferenceGeneration | PilotRuntimeFailure:
        def generate() -> GUIOwlV2Action:
            if not isinstance(reference_input, GUIOwlV21ThroughputReferenceInput):
                raise TypeError("generation input has the wrong adapter type")
            native_result = self._runtime.generate_native_action(
                reference_input.messages,
                cuda_peak_measurement_owner="caller",
            )
            try:
                action = native_result.parsed_output.canonical_action
                if type(action) is not GUIOwlV2Action:
                    raise TypeError("generation did not return one canonical action")
                return action
            finally:
                del native_result

        measured = self._measured_call(generate)
        if measured.failure_identifier is not None:
            return PilotRuntimeFailure(
                error_class_identifier=measured.failure_identifier,
                performance=measured.performance,
            )
        action = measured.value
        if type(action) is not GUIOwlV2Action:
            raise RuntimeError("measured generation lost its canonical action")
        return PilotReferenceGeneration(
            action_handle=action,
            performance=measured.performance,
        )

    def reference_actions_equal(self, left: object, right: object) -> bool:
        if type(left) is not GUIOwlV2Action or type(right) is not GUIOwlV2Action:
            raise TypeError("reference action handles must be GUIOwlV2Action")
        return left == right

    def teacher_force_reference(
        self,
        reference_inputs: tuple[object, ...],
        action_handles: tuple[object, ...],
    ) -> PilotReferenceTeacherForward | PilotRuntimeFailure:
        def teacher() -> int:
            if (
                not isinstance(reference_inputs, tuple)
                or len(reference_inputs) not in (1, 2)
                or any(
                    not isinstance(value, GUIOwlV21ThroughputReferenceInput)
                    for value in reference_inputs
                )
            ):
                raise TypeError("teacher inputs must be one or two adapter inputs")
            if (
                not isinstance(action_handles, tuple)
                or len(action_handles) != len(reference_inputs)
                or any(type(action) is not GUIOwlV2Action for action in action_handles)
            ):
                raise TypeError("teacher action handles must be canonical actions")
            native_result = self._runtime.teacher_forced_distance_logits(
                tuple(value.messages for value in reference_inputs),
                action_handles,
                cuda_peak_measurement_owner="caller",
            )
            if type(native_result) is not tuple or len(native_result) != 2:
                del native_result
                raise TypeError("teacher runtime returned an invalid result")
            logits, metadata = native_result
            del native_result
            example_count = len(reference_inputs)
            del logits
            del metadata
            return example_count

        measured = self._measured_call(teacher)
        if measured.failure_identifier is not None:
            return PilotRuntimeFailure(
                error_class_identifier=measured.failure_identifier,
                performance=measured.performance,
            )
        example_count = measured.value
        if type(example_count) is not int or example_count not in (1, 2):
            raise RuntimeError("measured teacher call lost its example count")
        return PilotReferenceTeacherForward(
            example_count=example_count,
            performance=measured.performance,
        )


__all__ = [
    "GUIOwlV21SetUtilityThroughputAdapter",
    "GUIOwlV21ThroughputReferenceInput",
    "ThroughputMeasurementError",
    "build_gui_owl_v2_1_throughput_reference_input",
]
