"""Metric-safe adapter for the D1 GUI-Owl action-stability diagnostic."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v1 import (
    AUTO_FRESH_ENCODE_CONDITION,
    AUTO_FROZEN_ENCODED_CONDITION,
    EAGER_FROZEN_ENCODED_CONTROL,
    PreparedExactInputV1,
)
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21GenerationResult
from causalcache.set_utility_gui_owl_v2_1_throughput_adapter import (
    GUIOwlV21ThroughputReferenceInput,
)


REPEAT_COUNT = 2
_SAFE_ERROR_CLASS = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_CONDITION_IDS = (
    AUTO_FRESH_ENCODE_CONDITION,
    AUTO_FROZEN_ENCODED_CONDITION,
    EAGER_FROZEN_ENCODED_CONTROL,
)


class ActionStabilityRuntimeV1(Protocol):
    def generate_native_action(self, messages: object) -> GUIOwlV21GenerationResult: ...

    def prepare_exact_input_v1(self, messages: object) -> PreparedExactInputV1: ...

    def prepared_input_unchanged_v1(self, prepared: PreparedExactInputV1) -> bool: ...

    def generate_from_prepared_v1(
        self,
        prepared: PreparedExactInputV1,
    ) -> GUIOwlV21GenerationResult: ...


@dataclass(frozen=True, slots=True)
class _OpaqueGenerationObservationV1:
    canonical_action: GUIOwlV2Action
    generated_sequence_sha256: str
    decoded_output_sha256: str

    def __post_init__(self) -> None:
        if type(self.canonical_action) is not GUIOwlV2Action:
            raise TypeError("diagnostic observation requires one canonical action")
        for value in (self.generated_sequence_sha256, self.decoded_output_sha256):
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("diagnostic observation digest is invalid")


def _safe_error_class(error: BaseException) -> str:
    name = error.__class__.__name__
    return name if _SAFE_ERROR_CLASS.fullmatch(name) is not None else "UnexpectedException"


def _observation(result: GUIOwlV21GenerationResult) -> _OpaqueGenerationObservationV1:
    if not isinstance(result, GUIOwlV21GenerationResult):
        raise TypeError("diagnostic runtime returned the wrong generation type")
    action = result.parsed_output.canonical_action
    metadata = result.metadata
    if type(action) is not GUIOwlV2Action or not isinstance(metadata, Mapping):
        raise TypeError("diagnostic runtime lost its opaque generation fields")
    return _OpaqueGenerationObservationV1(
        canonical_action=action,
        generated_sequence_sha256=str(metadata["generated_token_ids_sha256"]),
        decoded_output_sha256=str(metadata["decoded_output_utf8_sha256"]),
    )


def _condition_payload(
    *,
    condition_id: str,
    encode_call_count: int,
    generation_call_count: int,
    observations: tuple[_OpaqueGenerationObservationV1, ...],
    unchanged_before: bool | None,
    unchanged_between: bool | None,
    unchanged_after: bool | None,
    failure_class: str | None,
) -> dict[str, Any]:
    if condition_id not in _CONDITION_IDS:
        raise ValueError("unknown action-stability condition")
    if type(encode_call_count) is not int or encode_call_count < 0:
        raise ValueError("diagnostic encode count is invalid")
    if type(generation_call_count) is not int or not 0 <= generation_call_count <= REPEAT_COUNT:
        raise ValueError("diagnostic generation count is invalid")
    complete = len(observations) == REPEAT_COUNT and failure_class is None
    left = observations[0] if complete else None
    right = observations[1] if complete else None
    payload = {
        "canonical_action_equal": (
            left.canonical_action == right.canonical_action if complete else None
        ),
        "condition_id": condition_id,
        "decoded_output_equal": (
            left.decoded_output_sha256 == right.decoded_output_sha256
            if complete
            else None
        ),
        "encode_call_count": encode_call_count,
        "encoded_input_unchanged_after": unchanged_after,
        "encoded_input_unchanged_before": unchanged_before,
        "encoded_input_unchanged_between": unchanged_between,
        "exact_generated_sequence_equal": (
            left.generated_sequence_sha256 == right.generated_sequence_sha256
            if complete
            else None
        ),
        "failure_class": failure_class,
        "generation_call_count": generation_call_count,
        "generation_completed_count": len(observations),
        "metric_safe": True,
        "repeat_count": REPEAT_COUNT,
    }
    return payload


def run_fresh_encode_condition_v1(
    runtime: ActionStabilityRuntimeV1,
    reference_input: GUIOwlV21ThroughputReferenceInput,
) -> dict[str, Any]:
    if not isinstance(reference_input, GUIOwlV21ThroughputReferenceInput):
        raise TypeError("fresh diagnostic input has the wrong adapter type")
    observations: list[_OpaqueGenerationObservationV1] = []
    failure_class: str | None = None
    generation_call_count = 0
    for _ in range(REPEAT_COUNT):
        try:
            generation_call_count += 1
            result = runtime.generate_native_action(reference_input.messages)
            observations.append(_observation(result))
            del result
        except Exception as error:
            failure_class = _safe_error_class(error)
            break
    return _condition_payload(
        condition_id=AUTO_FRESH_ENCODE_CONDITION,
        encode_call_count=generation_call_count,
        generation_call_count=generation_call_count,
        observations=tuple(observations),
        unchanged_before=None,
        unchanged_between=None,
        unchanged_after=None,
        failure_class=failure_class,
    )


def run_frozen_encoded_condition_v1(
    runtime: ActionStabilityRuntimeV1,
    reference_input: GUIOwlV21ThroughputReferenceInput,
    *,
    condition_id: str,
) -> dict[str, Any]:
    if condition_id not in (
        AUTO_FROZEN_ENCODED_CONDITION,
        EAGER_FROZEN_ENCODED_CONTROL,
    ):
        raise ValueError("frozen diagnostic condition identity drifted")
    if not isinstance(reference_input, GUIOwlV21ThroughputReferenceInput):
        raise TypeError("frozen diagnostic input has the wrong adapter type")
    observations: list[_OpaqueGenerationObservationV1] = []
    failure_class: str | None = None
    prepared: PreparedExactInputV1 | None = None
    unchanged_before: bool | None = None
    unchanged_between: bool | None = None
    unchanged_after: bool | None = None
    generation_call_count = 0
    try:
        prepared = runtime.prepare_exact_input_v1(reference_input.messages)
        unchanged_before = runtime.prepared_input_unchanged_v1(prepared)
        for repeat_index in range(REPEAT_COUNT):
            generation_call_count += 1
            result = runtime.generate_from_prepared_v1(prepared)
            observations.append(_observation(result))
            del result
            if repeat_index == 0:
                unchanged_between = runtime.prepared_input_unchanged_v1(prepared)
        unchanged_after = runtime.prepared_input_unchanged_v1(prepared)
    except Exception as error:
        failure_class = _safe_error_class(error)
        if prepared is not None:
            try:
                unchanged_after = runtime.prepared_input_unchanged_v1(prepared)
            except Exception:
                unchanged_after = False
    return _condition_payload(
        condition_id=condition_id,
        encode_call_count=1,
        generation_call_count=generation_call_count,
        observations=tuple(observations),
        unchanged_before=unchanged_before,
        unchanged_between=unchanged_between,
        unchanged_after=unchanged_after,
        failure_class=failure_class,
    )


__all__ = [
    "REPEAT_COUNT",
    "run_fresh_encode_condition_v1",
    "run_frozen_encoded_condition_v1",
]
