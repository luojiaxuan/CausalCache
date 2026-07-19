"""Metric-only paired microbatch wrapper for the set-utility throughput pilot."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.set_utility_label_producer import UtilityQuerySpec
from causalcache.set_utility_throughput_pilot import (
    PilotReferenceGeneration,
    PilotRuntimeFailure,
    ReferenceInputBuilder,
    SetUtilityThroughputPilotRuntime,
    run_train_only_set_utility_throughput_pilot,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_throughput_pilot_pair_v1"
AGGREGATE_PROTOCOL_ID = (
    "causalcache_set_utility_throughput_pilot_pair_v1_exact_12_state_aggregate"
)
PILOT_STATE_COUNT = 12
VARIANT_ORDER = (1, 2)

_SAFE_FAILURE = re.compile(
    r"(?:[A-Z][A-Z0-9_]*)(?::[A-Za-z_][A-Za-z0-9_]{0,127})?"
)
_SAFE_PAIR_FAILURE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,511}")
_CORE_TOP_LEVEL_KEYS = frozenset(
    {"latency_seconds", "peak_memory_bytes", "failure_class", "counts"}
)
_LATENCY_KEYS = (
    "reference_generation_end_to_end_wall_total",
    "reference_teacher_forward_end_to_end_wall_total",
    "native_call_end_to_end_wall_total",
)
_PEAK_KEYS = (
    "full_call_cuda_allocated_max",
    "full_call_cuda_reserved_max",
)
_CORE_COUNT_KEYS = (
    "reference_teacher_microbatch_size",
    "reference_plan_build_count",
    "reference_input_build_call_count",
    "reference_input_build_completed_count",
    "reference_generation_call_count",
    "reference_generation_completed_count",
    "reference_teacher_forward_call_count",
    "reference_teacher_forward_completed_call_count",
    "reference_teacher_forward_example_count",
    "reference_teacher_forward_completed_example_count",
)
_PAIR_COUNT_KEYS = (
    "pair_attempt_count",
    "pair_completed_count",
    "variant_attempt_count",
    "variant_completed_count",
    "cross_variant_comparison_count",
    "cross_variant_equal_count",
    "reference_plan_build_count",
    "reference_input_build_call_count",
    "reference_input_build_completed_count",
    "reference_generation_call_count",
    "reference_generation_completed_count",
    "reference_teacher_forward_call_count",
    "reference_teacher_forward_completed_call_count",
    "reference_teacher_forward_example_count",
    "reference_teacher_forward_completed_example_count",
    "native_call_count",
    "native_completed_call_count",
)
_PAIR_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "state_id",
        "variant_order",
        "variants",
        "cross_variant_reference_action_equal",
        "failure_class",
        "retry_count",
        "metric_only",
        "latency_seconds",
        "peak_memory_bytes",
        "counts",
        "expected_success_counts_per_state",
        "expected_success_counts_for_12_states",
    }
)


def _success_counts(state_count: int) -> dict[str, int]:
    return {
        "pair_attempt_count": state_count,
        "pair_completed_count": state_count,
        "variant_attempt_count": 2 * state_count,
        "variant_completed_count": 2 * state_count,
        "cross_variant_comparison_count": state_count,
        "cross_variant_equal_count": state_count,
        "reference_plan_build_count": 2 * state_count,
        "reference_input_build_call_count": 2 * state_count,
        "reference_input_build_completed_count": 2 * state_count,
        "reference_generation_call_count": 4 * state_count,
        "reference_generation_completed_count": 4 * state_count,
        "reference_teacher_forward_call_count": 3 * state_count,
        "reference_teacher_forward_completed_call_count": 3 * state_count,
        "reference_teacher_forward_example_count": 4 * state_count,
        "reference_teacher_forward_completed_example_count": 4 * state_count,
        "native_call_count": 7 * state_count,
        "native_completed_call_count": 7 * state_count,
    }


EXPECTED_SUCCESS_COUNTS_PER_STATE = _success_counts(1)
EXPECTED_SUCCESS_COUNTS_FOR_12_STATES = _success_counts(PILOT_STATE_COUNT)
EXPECTED_SUCCESS_NATIVE_CALLS_FOR_12_STATES = 84


def _safe_nonnegative_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


def _safe_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _project_core_result(
    value: Mapping[str, object],
    *,
    expected_microbatch_size: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _CORE_TOP_LEVEL_KEYS:
        raise ValueError("throughput core result fields drifted")
    raw_latency = value["latency_seconds"]
    raw_peak = value["peak_memory_bytes"]
    raw_counts = value["counts"]
    if not isinstance(raw_latency, Mapping) or set(raw_latency) != set(_LATENCY_KEYS):
        raise ValueError("throughput core latency fields drifted")
    if not isinstance(raw_peak, Mapping) or set(raw_peak) != set(_PEAK_KEYS):
        raise ValueError("throughput core peak-memory fields drifted")
    if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(_CORE_COUNT_KEYS):
        raise ValueError("throughput core count fields drifted")

    latency = {
        key: _safe_nonnegative_float(raw_latency[key], label=key)
        for key in _LATENCY_KEYS
    }
    peak = {
        key: _safe_nonnegative_int(raw_peak[key], label=key)
        for key in _PEAK_KEYS
    }
    if peak["full_call_cuda_reserved_max"] < peak["full_call_cuda_allocated_max"]:
        raise ValueError("reserved peak memory must cover allocated peak memory")
    counts = {
        key: _safe_nonnegative_int(raw_counts[key], label=key)
        for key in _CORE_COUNT_KEYS
    }
    if counts["reference_teacher_microbatch_size"] != expected_microbatch_size:
        raise ValueError("throughput core microbatch identity drifted")
    failure_class = value["failure_class"]
    if failure_class is not None and (
        not isinstance(failure_class, str)
        or _SAFE_FAILURE.fullmatch(failure_class) is None
    ):
        raise ValueError("throughput core failure class is unsafe")
    return {
        "latency_seconds": latency,
        "peak_memory_bytes": peak,
        "failure_class": failure_class,
        "counts": counts,
    }


def _pair_failure(stage: str, failure_class: str) -> str:
    return f"{stage}__{failure_class.replace(':', '__')}"


def _expected_pair_failure(
    variants: Sequence[Mapping[str, object]],
    *,
    cross_variant_equal: bool | None,
) -> str | None:
    if not variants:
        return "MICROBATCH_1__INVALID_CORE_PROJECTION"
    last = variants[-1]
    variant_failure = last["failure_class"]
    if variant_failure is not None:
        if cross_variant_equal is False:
            return "CROSS_VARIANT_REFERENCE_ACTION_MISMATCH"
        return _pair_failure(
            f"MICROBATCH_{len(variants)}",
            str(variant_failure),
        )
    if len(variants) == 1:
        return "MICROBATCH_2__INVALID_CORE_PROJECTION"
    if cross_variant_equal is not True:
        raise ValueError("successful variants lack cross-variant equality")
    return None


@dataclass
class _PairedOpaqueRuntime:
    runtime: SetUtilityThroughputPilotRuntime
    active_microbatch_size: int | None = None
    first_variant_action_handle: object | None = None
    cross_variant_equal: bool | None = None
    cross_variant_comparison_count: int = 0

    def begin_variant(self, microbatch_size: int) -> None:
        expected = VARIANT_ORDER[0] if self.active_microbatch_size is None else 2
        if microbatch_size != expected:
            raise RuntimeError("paired throughput variant order drifted")
        if self.active_microbatch_size == 2:
            raise RuntimeError("paired throughput variants cannot be retried")
        self.active_microbatch_size = microbatch_size

    def generate_reference_action(
        self,
        reference_input: object,
    ) -> PilotReferenceGeneration | PilotRuntimeFailure:
        return self.runtime.generate_reference_action(reference_input)

    def reference_actions_equal(self, left: object, right: object) -> bool:
        within_variant = self.runtime.reference_actions_equal(left, right)
        if type(within_variant) is not bool:
            raise TypeError("opaque action equality must return bool")
        if not within_variant:
            return False
        if self.active_microbatch_size == 1:
            self.first_variant_action_handle = left
            return True
        if self.active_microbatch_size != 2 or self.first_variant_action_handle is None:
            raise RuntimeError("paired opaque action state is incomplete")
        cross_variant = self.runtime.reference_actions_equal(
            self.first_variant_action_handle,
            left,
        )
        if type(cross_variant) is not bool:
            raise TypeError("cross-variant opaque action equality must return bool")
        self.cross_variant_comparison_count += 1
        self.cross_variant_equal = cross_variant
        return cross_variant

    def teacher_force_reference(
        self,
        reference_inputs: tuple[object, ...],
        action_handles: tuple[object, ...],
    ) -> object:
        return self.runtime.teacher_force_reference(reference_inputs, action_handles)


def _pair_counts(
    variants: Sequence[Mapping[str, object]],
    *,
    cross_variant_comparison_count: int,
    cross_variant_equal: bool | None,
    pair_completed: bool,
) -> dict[str, int]:
    counts = {
        "pair_attempt_count": 1,
        "pair_completed_count": int(pair_completed),
        "variant_attempt_count": len(variants),
        "variant_completed_count": sum(
            variant["failure_class"] is None for variant in variants
        ),
        "cross_variant_comparison_count": cross_variant_comparison_count,
        "cross_variant_equal_count": int(cross_variant_equal is True),
    }
    for key in _CORE_COUNT_KEYS:
        if key == "reference_teacher_microbatch_size":
            continue
        counts[key] = sum(int(variant["counts"][key]) for variant in variants)
    counts["native_call_count"] = (
        counts["reference_generation_call_count"]
        + counts["reference_teacher_forward_call_count"]
    )
    counts["native_completed_call_count"] = (
        counts["reference_generation_completed_count"]
        + counts["reference_teacher_forward_completed_call_count"]
    )
    if set(counts) != set(_PAIR_COUNT_KEYS):
        raise RuntimeError("paired throughput count projection drifted")
    return counts


def _paired_payload(
    *,
    state_id: str,
    variants: Sequence[Mapping[str, object]],
    cross_variant_equal: bool | None,
    cross_variant_comparison_count: int,
    failure_class: str | None,
) -> dict[str, object]:
    pair_completed = (
        len(variants) == 2
        and failure_class is None
        and cross_variant_equal is True
        and all(variant["failure_class"] is None for variant in variants)
    )
    counts = _pair_counts(
        variants,
        cross_variant_comparison_count=cross_variant_comparison_count,
        cross_variant_equal=cross_variant_equal,
        pair_completed=pair_completed,
    )
    latency = {
        key: math.fsum(float(variant["latency_seconds"][key]) for variant in variants)
        for key in _LATENCY_KEYS
    }
    peak = {
        key: max((int(variant["peak_memory_bytes"][key]) for variant in variants), default=0)
        for key in _PEAK_KEYS
    }
    if pair_completed and counts != EXPECTED_SUCCESS_COUNTS_PER_STATE:
        raise RuntimeError("successful paired throughput operation count drifted")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "state_id": state_id,
        "variant_order": list(VARIANT_ORDER),
        "variants": {
            "1": variants[0] if len(variants) >= 1 else None,
            "2": variants[1] if len(variants) >= 2 else None,
        },
        "cross_variant_reference_action_equal": cross_variant_equal,
        "failure_class": failure_class,
        "retry_count": 0,
        "metric_only": True,
        "latency_seconds": latency,
        "peak_memory_bytes": peak,
        "counts": counts,
        "expected_success_counts_per_state": dict(
            EXPECTED_SUCCESS_COUNTS_PER_STATE
        ),
        "expected_success_counts_for_12_states": dict(
            EXPECTED_SUCCESS_COUNTS_FOR_12_STATES
        ),
    }


def run_train_only_set_utility_throughput_pilot_pair_v1(
    query: UtilityQuerySpec,
    *,
    reference_input_builder: ReferenceInputBuilder,
    runtime: SetUtilityThroughputPilotRuntime,
) -> dict[str, object]:
    """Run microbatch one then two and compare opaque actions before mb2 teacher."""
    if not isinstance(query, UtilityQuerySpec):
        raise TypeError("paired throughput pilot query must be UtilityQuerySpec")
    if query.split != "train":
        raise ValueError("paired throughput pilot accepts train queries only")
    if not callable(reference_input_builder):
        raise TypeError("paired throughput input builder must be callable")
    for method_name in (
        "generate_reference_action",
        "reference_actions_equal",
        "teacher_force_reference",
    ):
        if not callable(getattr(runtime, method_name, None)):
            raise TypeError(f"paired throughput runtime is missing {method_name}")

    proxy = _PairedOpaqueRuntime(runtime)
    variants: list[dict[str, object]] = []
    for microbatch_size in VARIANT_ORDER:
        proxy.begin_variant(microbatch_size)
        raw = run_train_only_set_utility_throughput_pilot(
            query,
            reference_input_builder=reference_input_builder,
            runtime=proxy,
            reference_teacher_microbatch_size=microbatch_size,
        )
        try:
            projected = _project_core_result(
                raw,
                expected_microbatch_size=microbatch_size,
            )
        except Exception:
            return _paired_payload(
                state_id=query.state_id,
                variants=variants,
                cross_variant_equal=proxy.cross_variant_equal,
                cross_variant_comparison_count=(
                    proxy.cross_variant_comparison_count
                ),
                failure_class=f"MICROBATCH_{microbatch_size}__INVALID_CORE_PROJECTION",
            )
        variants.append(projected)
        if projected["failure_class"] is not None:
            if proxy.cross_variant_equal is False:
                failure_class = "CROSS_VARIANT_REFERENCE_ACTION_MISMATCH"
            else:
                failure_class = _pair_failure(
                    f"MICROBATCH_{microbatch_size}",
                    str(projected["failure_class"]),
                )
            return _paired_payload(
                state_id=query.state_id,
                variants=variants,
                cross_variant_equal=proxy.cross_variant_equal,
                cross_variant_comparison_count=(
                    proxy.cross_variant_comparison_count
                ),
                failure_class=failure_class,
            )

    return _paired_payload(
        state_id=query.state_id,
        variants=variants,
        cross_variant_equal=proxy.cross_variant_equal,
        cross_variant_comparison_count=proxy.cross_variant_comparison_count,
        failure_class=None,
    )


def aggregate_train_only_set_utility_throughput_pilot_pairs_v1(
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate exactly twelve paired metric-only results and verify 84 calls."""
    if (
        isinstance(pairs, (str, bytes, bytearray, Mapping))
        or not isinstance(pairs, Sequence)
        or len(pairs) != PILOT_STATE_COUNT
    ):
        raise ValueError("paired throughput aggregate requires exactly 12 states")
    state_ids: list[str] = []
    failures: Counter[str] = Counter()
    aggregate_counts = {key: 0 for key in _PAIR_COUNT_KEYS}
    latency_parts = {key: [] for key in _LATENCY_KEYS}
    peak_values = {key: [] for key in _PEAK_KEYS}
    for pair in pairs:
        if not isinstance(pair, Mapping) or set(pair) != _PAIR_TOP_LEVEL_KEYS:
            raise ValueError("paired throughput result fields drifted")
        if (
            pair.get("schema_version") != SCHEMA_VERSION
            or pair.get("protocol_id") != PROTOCOL_ID
            or pair.get("variant_order") != list(VARIANT_ORDER)
            or pair.get("retry_count") != 0
            or pair.get("metric_only") is not True
        ):
            raise ValueError("paired throughput result identity drifted")
        state_id = pair.get("state_id")
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("paired throughput state id is invalid")
        state_ids.append(state_id)
        failure = pair.get("failure_class")
        if failure is not None:
            if (
                not isinstance(failure, str)
                or _SAFE_PAIR_FAILURE.fullmatch(failure) is None
            ):
                raise ValueError("paired throughput failure class is invalid")
            failures[failure] += 1
        raw_variants = pair.get("variants")
        if not isinstance(raw_variants, Mapping) or set(raw_variants) != {"1", "2"}:
            raise ValueError("paired throughput variant fields drifted")
        projected_variants: list[dict[str, object]] = []
        saw_missing_variant = False
        for microbatch_size in VARIANT_ORDER:
            raw_variant = raw_variants[str(microbatch_size)]
            if raw_variant is None:
                saw_missing_variant = True
                continue
            if saw_missing_variant:
                raise ValueError("paired throughput variants are not a strict prefix")
            projected_variants.append(
                _project_core_result(
                    raw_variant,
                    expected_microbatch_size=microbatch_size,
                )
            )
        raw_counts = pair.get("counts")
        raw_latency = pair.get("latency_seconds")
        raw_peak = pair.get("peak_memory_bytes")
        if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(
            _PAIR_COUNT_KEYS
        ):
            raise ValueError("paired throughput aggregate count fields drifted")
        if not isinstance(raw_latency, Mapping) or set(raw_latency) != set(
            _LATENCY_KEYS
        ):
            raise ValueError("paired throughput aggregate latency fields drifted")
        if not isinstance(raw_peak, Mapping) or set(raw_peak) != set(_PEAK_KEYS):
            raise ValueError("paired throughput aggregate peak fields drifted")
        cross_variant_equal = pair.get("cross_variant_reference_action_equal")
        if cross_variant_equal not in (None, True, False):
            raise ValueError("paired throughput cross-variant equality is invalid")
        if failure != _expected_pair_failure(
            projected_variants,
            cross_variant_equal=cross_variant_equal,
        ):
            raise ValueError("paired throughput failure differs from reconstruction")
        reconstructed = _paired_payload(
            state_id=state_id,
            variants=projected_variants,
            cross_variant_equal=cross_variant_equal,
            cross_variant_comparison_count=int(
                raw_counts.get("cross_variant_comparison_count", -1)
            ),
            failure_class=failure,
        )
        if dict(pair) != reconstructed:
            raise ValueError("paired throughput result differs from reconstruction")
        for key in _PAIR_COUNT_KEYS:
            aggregate_counts[key] += _safe_nonnegative_int(
                raw_counts[key], label=key
            )
        for key in _LATENCY_KEYS:
            latency_parts[key].append(
                _safe_nonnegative_float(raw_latency[key], label=key)
            )
        for key in _PEAK_KEYS:
            peak_values[key].append(_safe_nonnegative_int(raw_peak[key], label=key))
    if len(set(state_ids)) != PILOT_STATE_COUNT:
        raise ValueError("paired throughput aggregate state ids are not unique")
    all_pairs_successful = not failures
    if all_pairs_successful and aggregate_counts != EXPECTED_SUCCESS_COUNTS_FOR_12_STATES:
        raise ValueError("successful 12-state aggregate differs from the 84-call contract")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": AGGREGATE_PROTOCOL_ID,
        "state_count": PILOT_STATE_COUNT,
        "state_ids": sorted(state_ids),
        "variant_order": list(VARIANT_ORDER),
        "all_pairs_successful": all_pairs_successful,
        "failure_class_counts": dict(sorted(failures.items())),
        "retry_count": 0,
        "metric_only": True,
        "latency_seconds": {
            key: math.fsum(values) for key, values in latency_parts.items()
        },
        "peak_memory_bytes": {
            key: max(values, default=0) for key, values in peak_values.items()
        },
        "counts": aggregate_counts,
        "expected_success_counts": dict(EXPECTED_SUCCESS_COUNTS_FOR_12_STATES),
        "expected_success_native_call_count": (
            EXPECTED_SUCCESS_NATIVE_CALLS_FOR_12_STATES
        ),
    }


__all__ = [
    "AGGREGATE_PROTOCOL_ID",
    "EXPECTED_SUCCESS_COUNTS_FOR_12_STATES",
    "EXPECTED_SUCCESS_COUNTS_PER_STATE",
    "EXPECTED_SUCCESS_NATIVE_CALLS_FOR_12_STATES",
    "PILOT_STATE_COUNT",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "VARIANT_ORDER",
    "aggregate_train_only_set_utility_throughput_pilot_pairs_v1",
    "run_train_only_set_utility_throughput_pilot_pair_v1",
]
