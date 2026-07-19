"""Train-only source core for a metric-only set-utility throughput pilot."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from causalcache.set_utility_label_producer import (
    MixedFidelityPromptPlan,
    UtilityQuerySpec,
    build_mixed_fidelity_prompt_plan,
)


ALLOWED_REFERENCE_TEACHER_MICROBATCH_SIZES = (1, 2)
_SAFE_ERROR_CLASS_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")


@dataclass(frozen=True, slots=True)
class PilotRuntimePerformance:
    """Adapter-owned end-to-end wall time and full-call CUDA peaks.

    Measurement starts before encode, H2D, and teacher/generation preparation.
    It ends after native forward plus decode/parse or teacher-logit disposal.
    CUDA peak tracking covers that same complete adapter call.
    """

    end_to_end_wall_seconds: float
    full_call_cuda_peak_allocated_bytes: int
    full_call_cuda_peak_reserved_bytes: int

    def __post_init__(self) -> None:
        latency = self.end_to_end_wall_seconds
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or float(latency) < 0.0
        ):
            raise ValueError("pilot latency must be a finite non-negative number")
        for value, name in (
            (
                self.full_call_cuda_peak_allocated_bytes,
                "full-call allocated CUDA peak memory",
            ),
            (
                self.full_call_cuda_peak_reserved_bytes,
                "full-call reserved CUDA peak memory",
            ),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            self.full_call_cuda_peak_reserved_bytes
            < self.full_call_cuda_peak_allocated_bytes
        ):
            raise ValueError("reserved peak memory must cover allocated peak memory")
        object.__setattr__(self, "end_to_end_wall_seconds", float(latency))


@dataclass(frozen=True, slots=True)
class PilotReferenceGeneration:
    """Opaque in-memory action plus its safe performance projection."""

    action_handle: object
    performance: PilotRuntimePerformance

    def __post_init__(self) -> None:
        if self.action_handle is None:
            raise ValueError("reference generation action handle must not be None")
        if not isinstance(self.performance, PilotRuntimePerformance):
            raise TypeError("reference generation performance projection is invalid")


@dataclass(frozen=True, slots=True)
class PilotReferenceTeacherForward:
    """Safe projection after the adapter has discarded all teacher logits."""

    example_count: int
    performance: PilotRuntimePerformance

    def __post_init__(self) -> None:
        if type(self.example_count) is not int or self.example_count <= 0:
            raise ValueError("teacher example count must be a positive integer")
        if not isinstance(self.performance, PilotRuntimePerformance):
            raise TypeError("reference teacher performance projection is invalid")


@dataclass(frozen=True, slots=True)
class PilotRuntimeFailure:
    """Metric-only failed-call projection with no message or policy output."""

    error_class_identifier: str
    performance: PilotRuntimePerformance

    def __post_init__(self) -> None:
        if (
            type(self.error_class_identifier) is not str
            or _SAFE_ERROR_CLASS_IDENTIFIER.fullmatch(self.error_class_identifier) is None
        ):
            raise ValueError("pilot error class must be a safe ASCII identifier")
        if not isinstance(self.performance, PilotRuntimePerformance):
            raise TypeError("runtime failure performance projection is invalid")


class SetUtilityThroughputPilotRuntime(Protocol):
    """Adapter boundary that prevents policy outputs from reaching the core."""

    def generate_reference_action(
        self,
        reference_input: object,
    ) -> PilotReferenceGeneration | PilotRuntimeFailure: ...

    def reference_actions_equal(self, left: object, right: object) -> bool: ...

    def teacher_force_reference(
        self,
        reference_inputs: tuple[object, ...],
        action_handles: tuple[object, ...],
    ) -> PilotReferenceTeacherForward | PilotRuntimeFailure: ...


ReferenceInputBuilder = Callable[[MixedFidelityPromptPlan], object]


def _empty_counts(teacher_microbatch_size: int) -> dict[str, int]:
    return {
        "reference_teacher_microbatch_size": teacher_microbatch_size,
        "reference_plan_build_count": 0,
        "reference_input_build_call_count": 0,
        "reference_input_build_completed_count": 0,
        "reference_generation_call_count": 0,
        "reference_generation_completed_count": 0,
        "reference_teacher_forward_call_count": 0,
        "reference_teacher_forward_completed_call_count": 0,
        "reference_teacher_forward_example_count": 0,
        "reference_teacher_forward_completed_example_count": 0,
    }


def _payload(
    *,
    counts: dict[str, int],
    generation_wall_seconds: list[float],
    teacher_wall_seconds: list[float],
    allocated_peaks: list[int],
    reserved_peaks: list[int],
    failure_class: str | None,
) -> dict[str, object]:
    generation_wall = math.fsum(generation_wall_seconds)
    teacher_wall = math.fsum(teacher_wall_seconds)
    return {
        "latency_seconds": {
            "reference_generation_end_to_end_wall_total": generation_wall,
            "reference_teacher_forward_end_to_end_wall_total": teacher_wall,
            "native_call_end_to_end_wall_total": generation_wall + teacher_wall,
        },
        "peak_memory_bytes": {
            "full_call_cuda_allocated_max": max(allocated_peaks, default=0),
            "full_call_cuda_reserved_max": max(reserved_peaks, default=0),
        },
        "failure_class": failure_class,
        "counts": dict(counts),
    }


def _stage_failure_class(stage: str, error_class_identifier: str) -> str:
    safe_identifier = (
        error_class_identifier
        if _SAFE_ERROR_CLASS_IDENTIFIER.fullmatch(error_class_identifier) is not None
        else "UnexpectedException"
    )
    return f"{stage}:{safe_identifier}"


def _unexpected_failure_class(stage: str, error: Exception) -> str:
    return _stage_failure_class(stage, error.__class__.__name__)


def _record_performance(
    performance: PilotRuntimePerformance,
    *,
    wall_seconds: list[float],
    allocated_peaks: list[int],
    reserved_peaks: list[int],
) -> None:
    if not isinstance(performance, PilotRuntimePerformance):
        raise TypeError("runtime returned an invalid performance projection")
    validated = PilotRuntimePerformance(
        end_to_end_wall_seconds=performance.end_to_end_wall_seconds,
        full_call_cuda_peak_allocated_bytes=(
            performance.full_call_cuda_peak_allocated_bytes
        ),
        full_call_cuda_peak_reserved_bytes=(
            performance.full_call_cuda_peak_reserved_bytes
        ),
    )
    wall_seconds.append(validated.end_to_end_wall_seconds)
    allocated_peaks.append(validated.full_call_cuda_peak_allocated_bytes)
    reserved_peaks.append(validated.full_call_cuda_peak_reserved_bytes)


def run_train_only_set_utility_throughput_pilot(
    query: UtilityQuerySpec,
    *,
    reference_input_builder: ReferenceInputBuilder,
    runtime: SetUtilityThroughputPilotRuntime,
    reference_teacher_microbatch_size: int,
) -> dict[str, object]:
    """Run two reference generations and two logical reference teacher examples.

    The input builder may render the frozen plan into native messages, but it must
    not run the processor or touch CUDA.  Each runtime adapter owns the complete
    encode/H2D/preparation/native-call/decode-or-disposal measurement boundary.
    """
    if not isinstance(query, UtilityQuerySpec):
        raise TypeError("throughput pilot query must be UtilityQuerySpec")
    if query.split != "train":
        raise ValueError("throughput pilot accepts train queries only")
    if (
        type(reference_teacher_microbatch_size) is not int
        or reference_teacher_microbatch_size
        not in ALLOWED_REFERENCE_TEACHER_MICROBATCH_SIZES
    ):
        raise ValueError("reference teacher microbatch size must be 1 or 2")
    if not callable(reference_input_builder):
        raise TypeError("reference input builder must be callable")
    for method_name in (
        "generate_reference_action",
        "reference_actions_equal",
        "teacher_force_reference",
    ):
        if not callable(getattr(runtime, method_name, None)):
            raise TypeError(f"pilot runtime is missing {method_name}")

    counts = _empty_counts(reference_teacher_microbatch_size)
    generation_wall_seconds: list[float] = []
    teacher_wall_seconds: list[float] = []
    allocated_peaks: list[int] = []
    reserved_peaks: list[int] = []

    reference_plan = build_mixed_fidelity_prompt_plan(
        query,
        query.candidate_event_step_ids,
    )
    counts["reference_plan_build_count"] = 1
    counts["reference_input_build_call_count"] = 1
    try:
        reference_input = reference_input_builder(reference_plan)
        if reference_input is None:
            raise ValueError("reference input builder returned None")
    except Exception as error:
        return _payload(
            counts=counts,
            generation_wall_seconds=generation_wall_seconds,
            teacher_wall_seconds=teacher_wall_seconds,
            allocated_peaks=allocated_peaks,
            reserved_peaks=reserved_peaks,
            failure_class=_unexpected_failure_class("REFERENCE_INPUT_BUILD", error),
        )
    counts["reference_input_build_completed_count"] = 1

    generations: list[PilotReferenceGeneration] = []
    for _ in range(2):
        counts["reference_generation_call_count"] += 1
        try:
            generation_outcome = runtime.generate_reference_action(reference_input)
            if isinstance(generation_outcome, PilotRuntimeFailure):
                _record_performance(
                    generation_outcome.performance,
                    wall_seconds=generation_wall_seconds,
                    allocated_peaks=allocated_peaks,
                    reserved_peaks=reserved_peaks,
                )
                return _payload(
                    counts=counts,
                    generation_wall_seconds=generation_wall_seconds,
                    teacher_wall_seconds=teacher_wall_seconds,
                    allocated_peaks=allocated_peaks,
                    reserved_peaks=reserved_peaks,
                    failure_class=_stage_failure_class(
                        "REFERENCE_GENERATION",
                        generation_outcome.error_class_identifier,
                    ),
                )
            generation = generation_outcome
            if not isinstance(generation, PilotReferenceGeneration):
                raise TypeError("runtime returned an invalid generation projection")
            _record_performance(
                generation.performance,
                wall_seconds=generation_wall_seconds,
                allocated_peaks=allocated_peaks,
                reserved_peaks=reserved_peaks,
            )
        except Exception as error:
            return _payload(
                counts=counts,
                generation_wall_seconds=generation_wall_seconds,
                teacher_wall_seconds=teacher_wall_seconds,
                allocated_peaks=allocated_peaks,
                reserved_peaks=reserved_peaks,
                failure_class=_unexpected_failure_class(
                    "REFERENCE_GENERATION", error
                ),
            )
        generations.append(generation)
        counts["reference_generation_completed_count"] += 1

    try:
        actions_equal = runtime.reference_actions_equal(
            generations[0].action_handle,
            generations[1].action_handle,
        )
        if type(actions_equal) is not bool:
            raise TypeError("reference action comparison must return bool")
    except Exception as error:
        return _payload(
            counts=counts,
            generation_wall_seconds=generation_wall_seconds,
            teacher_wall_seconds=teacher_wall_seconds,
            allocated_peaks=allocated_peaks,
            reserved_peaks=reserved_peaks,
            failure_class=_unexpected_failure_class(
                "REFERENCE_REPEAT_COMPARISON", error
            ),
        )
    if not actions_equal:
        return _payload(
            counts=counts,
            generation_wall_seconds=generation_wall_seconds,
            teacher_wall_seconds=teacher_wall_seconds,
            allocated_peaks=allocated_peaks,
            reserved_peaks=reserved_peaks,
            failure_class="REFERENCE_ACTION_MISMATCH",
        )

    canonical_action_handle = generations[0].action_handle
    remaining_examples = 2
    while remaining_examples:
        batch_size = min(reference_teacher_microbatch_size, remaining_examples)
        reference_inputs = (reference_input,) * batch_size
        action_handles = (canonical_action_handle,) * batch_size
        counts["reference_teacher_forward_call_count"] += 1
        counts["reference_teacher_forward_example_count"] += batch_size
        try:
            teacher_outcome = runtime.teacher_force_reference(
                reference_inputs,
                action_handles,
            )
            if isinstance(teacher_outcome, PilotRuntimeFailure):
                _record_performance(
                    teacher_outcome.performance,
                    wall_seconds=teacher_wall_seconds,
                    allocated_peaks=allocated_peaks,
                    reserved_peaks=reserved_peaks,
                )
                return _payload(
                    counts=counts,
                    generation_wall_seconds=generation_wall_seconds,
                    teacher_wall_seconds=teacher_wall_seconds,
                    allocated_peaks=allocated_peaks,
                    reserved_peaks=reserved_peaks,
                    failure_class=_stage_failure_class(
                        "REFERENCE_TEACHER_FORWARD",
                        teacher_outcome.error_class_identifier,
                    ),
                )
            teacher = teacher_outcome
            if not isinstance(teacher, PilotReferenceTeacherForward):
                raise TypeError("runtime returned an invalid teacher projection")
            if teacher.example_count != batch_size:
                raise ValueError("teacher projection example count drifted")
            _record_performance(
                teacher.performance,
                wall_seconds=teacher_wall_seconds,
                allocated_peaks=allocated_peaks,
                reserved_peaks=reserved_peaks,
            )
        except Exception as error:
            return _payload(
                counts=counts,
                generation_wall_seconds=generation_wall_seconds,
                teacher_wall_seconds=teacher_wall_seconds,
                allocated_peaks=allocated_peaks,
                reserved_peaks=reserved_peaks,
                failure_class=_unexpected_failure_class(
                    "REFERENCE_TEACHER_FORWARD", error
                ),
            )
        counts["reference_teacher_forward_completed_call_count"] += 1
        counts["reference_teacher_forward_completed_example_count"] += batch_size
        remaining_examples -= batch_size

    return _payload(
        counts=counts,
        generation_wall_seconds=generation_wall_seconds,
        teacher_wall_seconds=teacher_wall_seconds,
        allocated_peaks=allocated_peaks,
        reserved_peaks=reserved_peaks,
        failure_class=None,
    )


__all__ = [
    "ALLOWED_REFERENCE_TEACHER_MICROBATCH_SIZES",
    "PilotReferenceGeneration",
    "PilotReferenceTeacherForward",
    "PilotRuntimeFailure",
    "PilotRuntimePerformance",
    "ReferenceInputBuilder",
    "SetUtilityThroughputPilotRuntime",
    "run_train_only_set_utility_throughput_pilot",
]
