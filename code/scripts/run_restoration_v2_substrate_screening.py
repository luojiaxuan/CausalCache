"""Run the frozen 45-state restoration-v2 development substrate screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import platform
import re
import socket
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from causalcache.data.guiodyssey_restoration_v2 import (
    EXPECTED_FORMAL_COUNTS,
    PAYLOAD_PREFIX,
    load_json_object,
    sha256_file,
)
from causalcache.data.restoration_v2_screening import (
    EXPECTED_SCREENING_STATES,
    SCREENING_ROLES,
    ScreeningState,
    ValidatedScreeningArtifact,
    load_validated_screening_artifact,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_restoration_v2"
RUN_STATUS = "RESTORATION_V2_SUBSTRATE_SCREENING"
STATE_OUTCOME_VALID = "VALID_SUBSTRATE_STATE"
STATE_OUTCOME_FAILED = "FAILED_SUBSTRATE_STATE"
ATTEMPT_STATUS = "ATTEMPT_STARTED_NO_RETRY"
SCREENING_ALLOWED = "SCREENING_ALLOWED"
CONFIRM_LOCKED = "CONFIRM_LOCKED"
READINESS_DEPENDENCY_COUNT = 8
RUN_MANIFEST_FILENAME = "run_manifest.json"
AGGREGATE_FILENAME = "aggregate.json"
STATE_DIRECTORY = "states"
ATTEMPT_DIRECTORY = "attempts"
REPEAT_NOISE_FLOOR = 1e-4
REPEAT_NOISE_MULTIPLIER = 10.0
CANONICAL_SCIENTIFIC_CONFIG_PATH = "code/configs/causalcache_restoration_v2.json"
CANONICAL_SELECTION_MANIFEST_PATH = "data/manifests/restoration_v2_selection.json"
CANONICAL_OCR_BACKEND_CONFIG_PATH = "code/configs/restoration_v2_ocr_backend.json"
CANONICAL_OCR_BACKEND_MANIFEST_PATH = "data/manifests/restoration_v2_ocr_backend.json"
CUDA_DEVICE_PATTERN = re.compile(r"cuda:[0-9]+")
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
CONTAINER_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class SubstrateGateContract:
    minimum_screening_states: int
    minimum_parse_coverage: float
    minimum_finite_logit_coverage: float
    minimum_repeat_canonical_action_agreement: float
    minimum_memory_sensitive_states: int
    fail_outcome: str
    pass_outcome: str

    @classmethod
    def from_scientific_config(
        cls,
        scientific_config: Mapping[str, Any],
    ) -> "SubstrateGateContract":
        gate = scientific_config.get("substrate_gate")
        if not isinstance(gate, Mapping):
            raise ValueError("scientific config is missing substrate_gate")
        if gate.get("screening_roles") != list(SCREENING_ROLES):
            raise ValueError("substrate screening roles drifted")
        if gate.get("memory_sensitive_definition") != (
            "summary_reference_kl_greater_than_repeat_noise_epsilon"
        ):
            raise ValueError("substrate memory-sensitive definition drifted")

        integer_fields = (
            "minimum_screening_states",
            "minimum_memory_sensitive_states",
        )
        for field in integer_fields:
            if type(gate.get(field)) is not int or gate[field] <= 0:
                raise ValueError(f"substrate {field} must be a positive integer")
        rate_fields = (
            "minimum_parse_coverage",
            "minimum_finite_logit_coverage",
            "minimum_repeat_canonical_action_agreement",
        )
        for field in rate_fields:
            value = gate.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"substrate {field} must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"substrate {field} must be a finite rate")
        for field in ("fail_outcome", "pass_outcome"):
            if not isinstance(gate.get(field), str) or not gate[field]:
                raise ValueError(f"substrate {field} must be a non-empty string")
        return cls(
            minimum_screening_states=gate["minimum_screening_states"],
            minimum_parse_coverage=float(gate["minimum_parse_coverage"]),
            minimum_finite_logit_coverage=float(
                gate["minimum_finite_logit_coverage"]
            ),
            minimum_repeat_canonical_action_agreement=float(
                gate["minimum_repeat_canonical_action_agreement"]
            ),
            minimum_memory_sensitive_states=gate[
                "minimum_memory_sensitive_states"
            ],
            fail_outcome=gate["fail_outcome"],
            pass_outcome=gate["pass_outcome"],
        )


@dataclass(frozen=True)
class DistanceMeasurement:
    value: float
    audit: Mapping[str, Any]


class GPUFullVocabularyKLBackend:
    """GPU-only KL wrapper that transfers exactly one final scalar per call."""

    def __init__(self, *, torch_module: Any, kl_kernel: Callable[..., Any]) -> None:
        self.torch = torch_module
        self.kl_kernel = kl_kernel

    def prepare_reference(self, logits: Any) -> Any:
        if getattr(logits, "ndim", None) != 3 or int(logits.shape[0]) != 1:
            raise ValueError("reference logits must have shape [1,tokens,vocab]")
        if getattr(logits.device, "type", None) != "cuda":
            raise ValueError("reference logits must remain on CUDA")
        if str(logits.dtype) != "torch.bfloat16":
            raise TypeError("reference logits must remain BF16")
        if getattr(logits, "requires_grad", False):
            raise ValueError("reference logits must not require gradients")
        with self.torch.inference_mode():
            reference = self.torch.log_softmax(
                logits.to(dtype=self.torch.float32),
                dim=-1,
            )
        if reference.device != logits.device or str(reference.dtype) != "torch.float32":
            raise RuntimeError("reference log-probabilities left GPU FP32")
        return reference

    def measure(self, reference_log_probs: Any, candidate_logits: Any) -> DistanceMeasurement:
        result = self.kl_kernel(
            reference_log_probs,
            candidate_logits,
            candidate_representation="logits",
        )
        distances = result.per_example_mean_kl
        if getattr(distances, "ndim", None) != 1 or tuple(distances.shape) != (1,):
            raise RuntimeError("GPU KL must return exactly one final state distance")
        if getattr(distances.device, "type", None) != "cuda":
            raise RuntimeError("GPU KL distance left CUDA before final transfer")
        value = float(distances.detach().to(device="cpu").item())
        audit = result.audit.to_dict()
        if audit.get("full_tensor_host_transfers") != 0:
            raise RuntimeError("GPU KL audit reported a full-tensor host transfer")
        return DistanceMeasurement(value=value, audit=audit)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json_object(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            value[key] = item
        return value

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs_hook)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = _pretty_json_bytes(dict(value))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _state_projection(state: ScreeningState) -> dict[str, Any]:
    return {
        "index": state.index,
        "role": state.role,
        "trajectory_id": state.trajectory_id,
        "decision_step_id": state.decision_step_id,
        "state_id": state.state_id,
        "candidate_event_step_ids": list(state.candidate_event_step_ids),
    }


def validate_screening_authorization_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if result.get("state") != SCREENING_ALLOWED:
        raise PermissionError("restoration-v2 screening is not authorized")
    if result.get("confirm_state") != CONFIRM_LOCKED:
        raise PermissionError("restoration-v2 confirm role is not locked")
    if result.get("confirm_locked") is not True:
        raise PermissionError("restoration-v2 readiness did not lock confirm")
    if result.get("allowed_roles") != list(SCREENING_ROLES):
        raise PermissionError("restoration-v2 readiness allowed-role set drifted")
    if (
        result.get("passed_dependency_count") != READINESS_DEPENDENCY_COUNT
        or result.get("dependency_count") != READINESS_DEPENDENCY_COUNT
    ):
        raise PermissionError("restoration-v2 readiness dependencies are incomplete")
    return dict(result)


def authorize_screening(
    *,
    execution_config_path: Path,
    readiness_manifest_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Run the CPU readiness guard before any runtime module can be imported."""
    from scripts.validate_restoration_v2_readiness import (
        validate_screening_authorization,
    )

    result = validate_screening_authorization(
        execution_config_path,
        readiness_manifest_path,
        repository_root,
        require_role="screening",
    )
    if not isinstance(result, Mapping):
        raise TypeError("readiness validator must return a mapping")
    return validate_screening_authorization_result(result)


def preflight_screening_message_shapes(
    *,
    artifact: ValidatedScreeningArtifact,
    runtime: Any,
    image_decoder: Callable[[bytes], Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    """Validate all 90 prompts against the verified context before generation."""
    if len(artifact.states) != EXPECTED_SCREENING_STATES:
        raise ValueError("shape sweep requires the exact 45-state denominator")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("shape sweep max_new_tokens must be positive")
    maximum_context_tokens = runtime.maximum_context_tokens()
    if type(maximum_context_tokens) is not int or maximum_context_tokens <= 0:
        raise ValueError("runtime returned an invalid verified context limit")
    records: list[dict[str, Any]] = []
    for state in artifact.states:
        for fidelity, restored_ids, expected_image_count in (
            (
                "reference",
                state.candidate_event_step_ids,
                len(state.candidate_event_step_ids) + 1,
            ),
            ("summary_only", (), 1),
        ):
            messages = artifact.build_messages(
                state,
                restored_event_step_ids=restored_ids,
                image_decoder=image_decoder,
            )
            shape = runtime.prepare_native_message_shape(messages)
            if not isinstance(shape, Mapping):
                raise TypeError("runtime message-shape result must be a mapping")
            if shape.get("policy_forward_executed") is not False:
                raise RuntimeError("shape sweep unexpectedly executed a policy forward")
            image_count = shape.get("image_count")
            sequence_length = shape.get("sequence_length")
            if image_count != expected_image_count:
                raise ValueError("shape sweep image count differs from frozen fidelity")
            if type(sequence_length) is not int or sequence_length <= 0:
                raise ValueError("shape sweep sequence length is invalid")
            if sequence_length + max_new_tokens > maximum_context_tokens:
                raise ValueError(
                    "INVALID_BEFORE_POLICY_FORWARD: prompt plus reserved generation "
                    "tokens exceeds the verified context limit"
                )
            records.append(
                {
                    "state_index": state.index,
                    "state_id": state.state_id,
                    "fidelity": fidelity,
                    "image_count": image_count,
                    "sequence_length": sequence_length,
                }
            )
    if len(records) != 2 * EXPECTED_SCREENING_STATES:
        raise AssertionError("shape sweep did not cover all 90 frozen prompts")
    sequence_lengths = [record["sequence_length"] for record in records]
    return {
        "status": "PASSED_90_PROMPT_PROCESSOR_ONLY_SHAPE_SWEEP",
        "prompt_count": len(records),
        "state_count": len(artifact.states),
        "policy_forward_executed": False,
        "maximum_context_tokens": maximum_context_tokens,
        "reserved_generation_tokens": max_new_tokens,
        "minimum_sequence_length": min(sequence_lengths),
        "maximum_sequence_length": max(sequence_lengths),
        "shape_records_sha256": hashlib.sha256(
            _canonical_json_bytes(records)
        ).hexdigest(),
        "records": records,
    }


def _is_out_of_memory(error: BaseException) -> bool:
    name = error.__class__.__name__.casefold()
    message = str(error).casefold()
    return "outofmemory" in name or "out of memory" in message


def _exception_failure(
    *,
    stage: str,
    category: str,
    error: Exception,
) -> dict[str, str]:
    if _is_out_of_memory(error):
        raise error
    return {
        "stage": stage,
        "category": category,
        "exception_type": error.__class__.__name__,
        "message": str(error),
    }


def _raise_invalid_runtime(*, stage: str, error: Exception) -> None:
    """Keep contract/runtime failures separate from scientific gate failures."""
    if _is_out_of_memory(error):
        raise error
    raise RuntimeError(
        f"INVALID_V2_SUBSTRATE_RUNTIME at {stage}: "
        f"{error.__class__.__name__}: {error}"
    ) from error


def _action_arguments(action: Any) -> dict[str, Any]:
    arguments = action.arguments()
    if not isinstance(arguments, Mapping):
        raise TypeError("canonical action arguments must be a mapping")
    return dict(arguments)


def _base_state_record(
    *,
    state: ScreeningState,
    run_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_contract_sha256": run_contract_sha256,
        "state": _state_projection(state),
        "outcome": STATE_OUTCOME_FAILED,
        "failure": None,
        "parse_success": False,
        "parse_success_count": 0,
        "repeat_canonical_action_agreement": False,
        "finite_logit_distances": False,
        "canonical_action": None,
        "native_generations": [],
        "message_shapes": {},
        "teacher_forwards": {},
        "distances": {
            "repeat_reference_kl": None,
            "summary_reference_kl": None,
        },
        "distance_audits": {},
        "started_at_utc": _utc_now(),
        "ended_at_utc": None,
        "duration_seconds": None,
    }


def _finalize_state_record(
    record: dict[str, Any],
    *,
    started: float,
    outcome: str,
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    record["outcome"] = outcome
    record["failure"] = dict(failure) if failure is not None else None
    record["ended_at_utc"] = _utc_now()
    record["duration_seconds"] = time.perf_counter() - started
    return record


def run_screening_state(
    *,
    artifact: ValidatedScreeningArtifact,
    state: ScreeningState,
    runtime: Any,
    distance_backend: Any,
    image_decoder: Callable[[bytes], Any],
    run_contract_sha256: str,
) -> dict[str, Any]:
    """Evaluate one frozen state exactly once with two reference repeats."""
    started = time.perf_counter()
    record = _base_state_record(
        state=state,
        run_contract_sha256=run_contract_sha256,
    )
    try:
        reference_messages = artifact.build_messages(
            state,
            restored_event_step_ids=state.candidate_event_step_ids,
            image_decoder=image_decoder,
        )
        summary_messages = artifact.build_messages(
            state,
            restored_event_step_ids=(),
            image_decoder=image_decoder,
        )
    except Exception as error:
        _raise_invalid_runtime(stage="message_construction", error=error)

    for name, messages in (
        ("reference", reference_messages),
        ("summary_only", summary_messages),
    ):
        try:
            record["message_shapes"][name] = runtime.prepare_native_message_shape(
                messages
            )
        except Exception as error:
            _raise_invalid_runtime(stage=f"{name}_message_shape", error=error)

    generations = []
    for repeat_index in (1, 2):
        try:
            generation = runtime.generate_native_action(reference_messages)
        except ValueError as error:
            output_text = getattr(error, "output_text", None)
            metadata = getattr(error, "metadata", None)
            if isinstance(output_text, str) and isinstance(metadata, Mapping):
                record["native_generations"].append(
                    {
                        "repeat_index": repeat_index,
                        "output_text": output_text,
                        "canonical_action": None,
                        "metadata": dict(metadata),
                        "parse_error_type": getattr(
                            error,
                            "parse_error_type",
                            error.__class__.__name__,
                        ),
                        "parse_error_message": getattr(
                            error,
                            "parse_error_message",
                            str(error),
                        ),
                    }
                )
            failure = _exception_failure(
                stage=f"reference_generation_{repeat_index}",
                category="PARSE_FAILURE",
                error=error,
            )
            return _finalize_state_record(
                record,
                started=started,
                outcome=STATE_OUTCOME_FAILED,
                failure=failure,
            )
        except Exception as error:
            _raise_invalid_runtime(
                stage=f"reference_generation_{repeat_index}",
                error=error,
            )
        action = generation.parsed_output.canonical_action
        arguments = _action_arguments(action)
        generations.append((generation, action, arguments))
        record["parse_success_count"] = repeat_index
        record["native_generations"].append(
            {
                "repeat_index": repeat_index,
                "output_text": generation.output_text,
                "canonical_action": arguments,
                "metadata": dict(generation.metadata),
            }
        )
    record["parse_success"] = True
    first_generation, action, first_arguments = generations[0]
    del first_generation
    second_arguments = generations[1][2]
    if first_arguments != second_arguments:
        failure = {
            "stage": "reference_generation_repeat_comparison",
            "category": "CANONICAL_ACTION_MISMATCH",
            "exception_type": None,
            "message": "two deterministic full-history generations produced different actions",
        }
        return _finalize_state_record(
            record,
            started=started,
            outcome=STATE_OUTCOME_FAILED,
            failure=failure,
        )
    record["repeat_canonical_action_agreement"] = True
    record["canonical_action"] = first_arguments

    try:
        reference_logits, reference_metadata = runtime.teacher_forced_distance_logits(
            (reference_messages,),
            (action,),
        )
        reference_log_probs = distance_backend.prepare_reference(reference_logits)
        del reference_logits
        record["teacher_forwards"]["reference_1"] = dict(reference_metadata)
    except Exception as error:
        _raise_invalid_runtime(stage="reference_teacher_forward_1", error=error)

    try:
        repeat_logits, repeat_metadata = runtime.teacher_forced_distance_logits(
            (reference_messages,),
            (action,),
        )
        repeat_measurement = distance_backend.measure(
            reference_log_probs,
            repeat_logits,
        )
        del repeat_logits
        record["teacher_forwards"]["reference_2"] = dict(repeat_metadata)
        repeat_value = float(repeat_measurement.value)
        repeat_is_finite = math.isfinite(repeat_value)
        record["distances"]["repeat_reference_kl"] = (
            repeat_value if repeat_is_finite else None
        )
        record["distance_audits"]["repeat_reference_kl"] = dict(
            repeat_measurement.audit
        )
    except Exception as error:
        _raise_invalid_runtime(
            stage="reference_teacher_forward_2_or_repeat_kl",
            error=error,
        )

    try:
        summary_logits, summary_metadata = runtime.teacher_forced_distance_logits(
            (summary_messages,),
            (action,),
        )
        summary_measurement = distance_backend.measure(
            reference_log_probs,
            summary_logits,
        )
        del summary_logits
        del reference_log_probs
        record["teacher_forwards"]["summary_only"] = dict(summary_metadata)
        summary_value = float(summary_measurement.value)
        summary_is_finite = math.isfinite(summary_value)
        record["distances"]["summary_reference_kl"] = (
            summary_value if summary_is_finite else None
        )
        record["distance_audits"]["summary_reference_kl"] = dict(
            summary_measurement.audit
        )
    except Exception as error:
        _raise_invalid_runtime(
            stage="summary_teacher_forward_or_reference_kl",
            error=error,
        )

    if not repeat_is_finite or not summary_is_finite:
        failure = {
            "stage": "gpu_distance_validation",
            "category": "NONFINITE_DISTANCE",
            "exception_type": None,
            "message": "repeat or summary-reference KL is non-finite",
        }
        return _finalize_state_record(
            record,
            started=started,
            outcome=STATE_OUTCOME_FAILED,
            failure=failure,
        )
    record["finite_logit_distances"] = True
    return _finalize_state_record(
        record,
        started=started,
        outcome=STATE_OUTCOME_VALID,
        failure=None,
    )


def aggregate_substrate_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_states: Sequence[ScreeningState],
    gate: SubstrateGateContract,
    run_contract_sha256: str,
) -> dict[str, Any]:
    """Aggregate the immutable denominator and apply the preregistered gate."""
    if len(expected_states) != EXPECTED_SCREENING_STATES:
        raise ValueError("substrate gate requires exactly 45 expected states")
    if len(records) != EXPECTED_SCREENING_STATES:
        raise ValueError("substrate gate requires exactly 45 state records")
    expected_projections = [_state_projection(state) for state in expected_states]
    observed_projections = [record.get("state") for record in records]
    if observed_projections != expected_projections:
        raise ValueError("substrate state records differ from the frozen denominator")
    if any(
        record.get("run_contract_sha256") != run_contract_sha256
        for record in records
    ):
        raise ValueError("substrate state record run identity drifted")

    denominator = len(records)
    parse_count = sum(record.get("parse_success") is True for record in records)
    finite_count = sum(
        record.get("finite_logit_distances") is True for record in records
    )
    agreement_count = sum(
        record.get("repeat_canonical_action_agreement") is True
        for record in records
    )
    repeat_values = [
        float(record["distances"]["repeat_reference_kl"])
        for record in records
        if isinstance(record.get("distances"), Mapping)
        and isinstance(record["distances"].get("repeat_reference_kl"), (int, float))
        and not isinstance(record["distances"].get("repeat_reference_kl"), bool)
        and math.isfinite(float(record["distances"]["repeat_reference_kl"]))
    ]
    mean_repeat_kl = (
        math.fsum(repeat_values) / len(repeat_values) if repeat_values else None
    )
    repeat_noise_epsilon = (
        max(REPEAT_NOISE_FLOOR, REPEAT_NOISE_MULTIPLIER * mean_repeat_kl)
        if mean_repeat_kl is not None
        else None
    )
    memory_sensitive_state_ids = []
    if repeat_noise_epsilon is not None:
        for record in records:
            distances = record.get("distances")
            if not isinstance(distances, Mapping):
                continue
            summary = distances.get("summary_reference_kl")
            if (
                isinstance(summary, (int, float))
                and not isinstance(summary, bool)
                and math.isfinite(float(summary))
                and float(summary) > repeat_noise_epsilon
            ):
                memory_sensitive_state_ids.append(record["state"]["state_id"])

    metrics = {
        "fixed_state_denominator": denominator,
        "parse_success_count": parse_count,
        "parse_coverage": parse_count / denominator,
        "finite_logit_state_count": finite_count,
        "finite_logit_coverage": finite_count / denominator,
        "repeat_canonical_action_agreement_count": agreement_count,
        "repeat_canonical_action_agreement": agreement_count / denominator,
        "finite_repeat_kl_count": len(repeat_values),
        "mean_repeat_kl": mean_repeat_kl,
        "repeat_noise_epsilon": repeat_noise_epsilon,
        "memory_sensitive_state_count": len(memory_sensitive_state_ids),
        "memory_sensitive_state_ids": memory_sensitive_state_ids,
    }
    checks = {
        "minimum_screening_states": (
            denominator >= gate.minimum_screening_states
        ),
        "minimum_parse_coverage": (
            metrics["parse_coverage"] >= gate.minimum_parse_coverage
        ),
        "minimum_finite_logit_coverage": (
            metrics["finite_logit_coverage"]
            >= gate.minimum_finite_logit_coverage
        ),
        "minimum_repeat_canonical_action_agreement": (
            metrics["repeat_canonical_action_agreement"]
            >= gate.minimum_repeat_canonical_action_agreement
        ),
        "minimum_memory_sensitive_states": (
            metrics["memory_sensitive_state_count"]
            >= gate.minimum_memory_sensitive_states
        ),
    }
    failure_counts = Counter(
        str(record["failure"]["category"])
        for record in records
        if isinstance(record.get("failure"), Mapping)
    )
    passed = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_contract_sha256": run_contract_sha256,
        "status": "COMPLETED_FIXED_45_STATE_SUBSTRATE_SCREENING",
        "outcome": gate.pass_outcome if passed else gate.fail_outcome,
        "gate_passed": passed,
        "gate_contract": asdict(gate),
        "metrics": metrics,
        "checks": checks,
        "failure_category_counts": dict(sorted(failure_counts.items())),
        "sample_mutation_performed": False,
        "top_up_performed": False,
        "confirm_role_used": False,
    }


def _validated_existing_state_record(
    path: Path,
    *,
    state: ScreeningState,
    run_contract_sha256: str,
) -> dict[str, Any]:
    record = _strict_json_object(path)
    if record.get("state") != _state_projection(state):
        raise ValueError(f"resumed state identity drifted: {path}")
    if record.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError(f"resumed state run identity drifted: {path}")
    if record.get("outcome") not in {STATE_OUTCOME_VALID, STATE_OUTCOME_FAILED}:
        raise ValueError(f"resumed state record is not terminal: {path}")
    if record.get("ended_at_utc") is None:
        raise ValueError(f"resumed state record is incomplete: {path}")
    return record


def _attempt_marker(
    *,
    state: ScreeningState,
    run_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": ATTEMPT_STATUS,
        "run_contract_sha256": run_contract_sha256,
        "state": _state_projection(state),
        "created_at_utc": _utc_now(),
    }


def _validate_attempt_marker(
    path: Path,
    *,
    state: ScreeningState,
    run_contract_sha256: str,
) -> dict[str, Any]:
    marker = _strict_json_object(path)
    if (
        marker.get("status") != ATTEMPT_STATUS
        or marker.get("run_contract_sha256") != run_contract_sha256
        or marker.get("state") != _state_projection(state)
    ):
        raise ValueError(f"state attempt marker identity drifted: {path}")
    return marker


def run_substrate_screening(
    *,
    artifact: ValidatedScreeningArtifact,
    runtime: Any,
    distance_backend: Any,
    image_decoder: Callable[[bytes], Any],
    gate: SubstrateGateContract,
    run_contract: Mapping[str, Any],
    output_dir: str | Path,
    resume: bool,
) -> dict[str, Any]:
    """Run or resume all 45 states without retries, filtering, or top-up."""
    if type(resume) is not bool:
        raise TypeError("resume must be bool")
    if len(artifact.states) != EXPECTED_SCREENING_STATES:
        raise ValueError("screening artifact denominator drifted")
    if any(state.role not in SCREENING_ROLES for state in artifact.states):
        raise ValueError("screening artifact contains a forbidden confirm state")
    contract = dict(run_contract)
    contract_sha256 = hashlib.sha256(_canonical_json_bytes(contract)).hexdigest()
    root = Path(output_dir)
    run_manifest_path = root / RUN_MANIFEST_FILENAME
    state_root = root / STATE_DIRECTORY
    attempt_root = root / ATTEMPT_DIRECTORY
    if resume:
        if not root.is_dir() or not state_root.is_dir() or not attempt_root.is_dir():
            raise FileNotFoundError("resume requires an existing screening run directory")
        run_manifest = _strict_json_object(run_manifest_path)
        if (
            run_manifest.get("run_contract_sha256") != contract_sha256
            or run_manifest.get("run_contract") != contract
        ):
            raise ValueError("resume run contract differs from the existing run")
    else:
        root.mkdir(parents=True, exist_ok=False)
        state_root.mkdir()
        attempt_root.mkdir()
        _write_json_exclusive(
            run_manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": RUN_STATUS,
                "run_contract_sha256": contract_sha256,
                "run_contract": contract,
                "created_at_utc": _utc_now(),
            },
        )

    records = []
    for state in artifact.states:
        state_path = state_root / f"{state.index:03d}.json"
        attempt_path = attempt_root / f"{state.index:03d}.json"
        if state_path.exists():
            if not resume:
                raise FileExistsError(f"unexpected pre-existing state record: {state_path}")
            if not attempt_path.is_file():
                raise ValueError("terminal state record has no no-retry attempt marker")
            _validate_attempt_marker(
                attempt_path,
                state=state,
                run_contract_sha256=contract_sha256,
            )
            record = _validated_existing_state_record(
                state_path,
                state=state,
                run_contract_sha256=contract_sha256,
            )
        else:
            if attempt_path.exists():
                _validate_attempt_marker(
                    attempt_path,
                    state=state,
                    run_contract_sha256=contract_sha256,
                )
                raise RuntimeError(
                    "incomplete prior state attempt cannot be retried or topped up"
                )
            _write_json_exclusive(
                attempt_path,
                _attempt_marker(
                    state=state,
                    run_contract_sha256=contract_sha256,
                ),
            )
            record = run_screening_state(
                artifact=artifact,
                state=state,
                runtime=runtime,
                distance_backend=distance_backend,
                image_decoder=image_decoder,
                run_contract_sha256=contract_sha256,
            )
            _write_json_exclusive(state_path, record)
        records.append(record)

    aggregate = aggregate_substrate_gate(
        records,
        expected_states=artifact.states,
        gate=gate,
        run_contract_sha256=contract_sha256,
    )
    aggregate_path = root / AGGREGATE_FILENAME
    if aggregate_path.exists():
        if not resume:
            raise FileExistsError(f"unexpected pre-existing aggregate: {aggregate_path}")
        if _strict_json_object(aggregate_path) != aggregate:
            raise ValueError("resumed aggregate differs from recomputed fixed denominator")
    else:
        _write_json_exclusive(aggregate_path, aggregate)
    return aggregate


def _canonical_repository_file(
    value: str | Path,
    *,
    repository_root: Path,
    relative_path: str,
    name: str,
) -> Path:
    supplied = Path(value)
    resolved = (
        supplied.resolve()
        if supplied.is_absolute()
        else (repository_root / supplied).resolve()
    )
    expected = (repository_root / relative_path).resolve()
    if resolved != expected or not resolved.is_file():
        raise ValueError(f"{name} must resolve to canonical Git path {relative_path}")
    return resolved


def _dependency_evidence(
    execution_config: Mapping[str, Any],
    *,
    dependency_id: int,
    expected_path: str,
) -> Mapping[str, Any]:
    dependencies = execution_config.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("execution config dependencies are missing")
    matches = [
        record
        for record in dependencies
        if isinstance(record, Mapping) and record.get("id") == dependency_id
    ]
    if len(matches) != 1 or matches[0].get("status") != "passed":
        raise ValueError(f"execution dependency {dependency_id} is not uniquely passed")
    evidence = matches[0].get("evidence")
    if not isinstance(evidence, list):
        raise ValueError(f"execution dependency {dependency_id} evidence is missing")
    path_matches = [
        record
        for record in evidence
        if isinstance(record, Mapping) and record.get("path") == expected_path
    ]
    if len(path_matches) != 1:
        raise ValueError(
            f"execution dependency {dependency_id} does not bind {expected_path}"
        )
    return path_matches[0]


def _canonical_input_bindings(
    execution_config: Mapping[str, Any],
    *,
    repository_root: Path,
    scientific_config_path: str | Path,
    selection_manifest_path: str | Path,
    ocr_backend_config_path: str | Path,
) -> dict[str, Any]:
    scientific_path = _canonical_repository_file(
        scientific_config_path,
        repository_root=repository_root,
        relative_path=CANONICAL_SCIENTIFIC_CONFIG_PATH,
        name="scientific config",
    )
    selection_path = _canonical_repository_file(
        selection_manifest_path,
        repository_root=repository_root,
        relative_path=CANONICAL_SELECTION_MANIFEST_PATH,
        name="selection manifest",
    )
    ocr_path = _canonical_repository_file(
        ocr_backend_config_path,
        repository_root=repository_root,
        relative_path=CANONICAL_OCR_BACKEND_CONFIG_PATH,
        name="OCR backend config",
    )
    scientific_record = execution_config.get("scientific_contract")
    if scientific_record != {
        "path": CANONICAL_SCIENTIFIC_CONFIG_PATH,
        "sha256": sha256_file(scientific_path),
    }:
        raise ValueError("scientific config differs from the execution binding")

    selection_record = _dependency_evidence(
        execution_config,
        dependency_id=2,
        expected_path=CANONICAL_SELECTION_MANIFEST_PATH,
    )
    if selection_record.get("sha256") != sha256_file(selection_path):
        raise ValueError("selection manifest differs from dependency 2 binding")

    ocr_manifest_record = _dependency_evidence(
        execution_config,
        dependency_id=5,
        expected_path=CANONICAL_OCR_BACKEND_MANIFEST_PATH,
    )
    ocr_manifest_path = _canonical_repository_file(
        CANONICAL_OCR_BACKEND_MANIFEST_PATH,
        repository_root=repository_root,
        relative_path=CANONICAL_OCR_BACKEND_MANIFEST_PATH,
        name="OCR backend dependency manifest",
    )
    if ocr_manifest_record.get("sha256") != sha256_file(ocr_manifest_path):
        raise ValueError("OCR backend manifest differs from dependency 5 binding")
    _, ocr_manifest = load_json_object(ocr_manifest_path)
    source_files = ocr_manifest.get("source_files")
    if not isinstance(source_files, list):
        raise ValueError("OCR dependency manifest source files are missing")
    ocr_config_records = [
        record
        for record in source_files
        if isinstance(record, Mapping)
        and record.get("path") == CANONICAL_OCR_BACKEND_CONFIG_PATH
    ]
    if (
        len(ocr_config_records) != 1
        or ocr_config_records[0].get("sha256") != sha256_file(ocr_path)
    ):
        raise ValueError("OCR backend config differs from dependency 5 source binding")
    return {
        "scientific_config_path": scientific_path,
        "selection_manifest_path": selection_path,
        "ocr_backend_config_path": ocr_path,
        "records": {
            "scientific_config": dict(scientific_record),
            "selection_manifest": dict(selection_record),
            "ocr_backend_manifest": dict(ocr_manifest_record),
            "ocr_backend_config": dict(ocr_config_records[0]),
        },
    }


def _execution_bindings(
    execution_config: Mapping[str, Any],
    *,
    repository_root: Path,
    device: str,
    host_alias: str,
    host_hostname: str,
    container_id: str,
    container_image_digest: str,
    actual_container_hostname: str,
) -> dict[str, Any]:
    canonical_data = execution_config.get("canonical_data")
    if not isinstance(canonical_data, Mapping):
        raise ValueError("execution config is missing canonical_data")
    artifact_tree_sha256 = canonical_data.get("artifact_tree_sha256")
    if (
        not isinstance(artifact_tree_sha256, str)
        or len(artifact_tree_sha256) != 64
        or any(character not in "0123456789abcdef" for character in artifact_tree_sha256)
    ):
        raise ValueError("execution canonical_data artifact tree SHA256 is invalid")
    if canonical_data.get("payload_prefix") != PAYLOAD_PREFIX:
        raise ValueError("execution canonical_data payload prefix drifted")
    if canonical_data.get("counts") != EXPECTED_FORMAL_COUNTS:
        raise ValueError("execution canonical_data formal counts drifted")
    for field in ("repo", "immutable_revision"):
        if not isinstance(canonical_data.get(field), str) or not canonical_data[field]:
            raise ValueError(f"execution canonical_data {field} is invalid")

    canonical_policy = execution_config.get("canonical_policy")
    snapshot = (
        canonical_policy.get("snapshot_manifest")
        if isinstance(canonical_policy, Mapping)
        else None
    )
    if not isinstance(snapshot, Mapping):
        raise ValueError("execution config is missing canonical policy snapshot")
    snapshot_relative = snapshot.get("path")
    snapshot_sha256 = snapshot.get("sha256")
    if not isinstance(snapshot_relative, str) or not snapshot_relative:
        raise ValueError("execution snapshot-manifest path is invalid")
    snapshot_path = (repository_root / snapshot_relative).resolve()
    if not snapshot_path.is_file() or sha256_file(snapshot_path) != snapshot_sha256:
        raise ValueError("execution snapshot-manifest source identity drifted")

    execution_runtime = execution_config.get("execution_runtime")
    if (
        not isinstance(execution_runtime, Mapping)
        or CUDA_DEVICE_PATTERN.fullmatch(device) is None
        or execution_runtime.get("selected_device") != device
    ):
        raise ValueError("CLI CUDA device differs from the frozen execution config")
    for field, value in (
        ("host_alias", host_alias),
        ("host_hostname", host_hostname),
        ("container_id", container_id),
        ("container_image_digest", container_image_digest),
    ):
        if not isinstance(value, str) or execution_runtime.get(field) != value:
            raise ValueError(f"actual {field} differs from the frozen execution runtime")
    if CONTAINER_ID_PATTERN.fullmatch(container_id) is None:
        raise ValueError("--container-id must be a full lowercase Docker ID")
    if CONTAINER_DIGEST_PATTERN.fullmatch(container_image_digest) is None:
        raise ValueError("--container-image-digest must be sha256:<64 lowercase hex>")
    configured_container_hostname = execution_runtime.get("container_hostname")
    if (
        not isinstance(configured_container_hostname, str)
        or not configured_container_hostname
        or actual_container_hostname != configured_container_hostname
        or not container_id.startswith(configured_container_hostname)
    ):
        raise ValueError(
            "actual container hostname must equal config and prefix container ID"
        )
    actual_run_identity = {
        "host_alias": host_alias,
        "host_hostname": host_hostname,
        "selected_device": device,
        "container_id": container_id,
        "container_hostname": actual_container_hostname,
        "container_image": execution_runtime.get("container_image"),
        "container_image_digest": container_image_digest,
    }
    return {
        "canonical_data": dict(canonical_data),
        "snapshot_manifest_path": snapshot_path,
        "snapshot_manifest": dict(snapshot),
        "execution_runtime": dict(execution_runtime),
        "actual_run_identity": actual_run_identity,
    }


def _load_live_execution_runtime_identity(*, device: str) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("production screening requires the pinned PyTorch") from error
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("production screening requires available CUDA")
    match = CUDA_DEVICE_PATTERN.fullmatch(device)
    if match is None:
        raise ValueError("production screening device must be explicit CUDA")
    device_index = int(device.split(":", maxsplit=1)[1])
    visible_count = int(torch.cuda.device_count())
    if device_index >= visible_count:
        raise ValueError("production screening CUDA device is not visible")
    canonical_device = torch.device(device)
    torch.cuda.set_device(canonical_device)
    properties = torch.cuda.get_device_properties(canonical_device)
    property_uuid = getattr(properties, "uuid", None)
    if isinstance(property_uuid, bytes):
        gpu_uuid = property_uuid.decode("ascii")
    else:
        gpu_uuid = str(property_uuid) if property_uuid is not None else ""
    if gpu_uuid.startswith("GPU-"):
        gpu_uuid = gpu_uuid[4:]
    if not gpu_uuid:
        raise RuntimeError("PyTorch did not expose the selected GPU UUID")

    try:
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("nvidia-smi runtime identity query failed") from error
    records: list[dict[str, str]] = []
    for raw_line in nvidia_smi.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",", maxsplit=2)]
        if len(fields) == 3 and all(fields):
            records.append(
                {
                    "index": fields[0],
                    "uuid": fields[1],
                    "driver_version": fields[2],
                }
            )
    expected_nvidia_uuid = f"GPU-{gpu_uuid}"
    selected = next(
        (record for record in records if record["uuid"] == expected_nvidia_uuid),
        None,
    )
    if selected is None:
        raise RuntimeError("PyTorch and nvidia-smi GPU UUIDs differ")

    cudnn_version = (
        torch.backends.cudnn.version()
        if hasattr(torch.backends, "cudnn")
        else None
    )
    return {
        "platform_machine": platform.machine(),
        "gpu_name": str(properties.name),
        "gpu_uuid": gpu_uuid,
        "nvidia_smi_gpu_uuid": selected["uuid"],
        "nvidia_driver_version": selected["driver_version"],
        "gpu_compute_capability": [int(properties.major), int(properties.minor)],
        "gpu_multiprocessor_count": int(properties.multi_processor_count),
        "visible_cuda_device_count": visible_count,
        "selected_device": str(canonical_device),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_build_version": str(torch.version.cuda),
        "cudnn_version": cudnn_version,
        "transformers_version": importlib.metadata.version("transformers"),
    }


def _validate_live_execution_runtime_identity(
    configured: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    required_fields = (
        "platform_machine",
        "gpu_name",
        "gpu_uuid",
        "nvidia_smi_gpu_uuid",
        "nvidia_driver_version",
        "gpu_compute_capability",
        "gpu_multiprocessor_count",
        "visible_cuda_device_count",
        "selected_device",
        "python_version",
        "torch_version",
        "torch_cuda_build_version",
        "cudnn_version",
        "transformers_version",
    )
    for field in required_fields:
        if field not in observed or observed[field] != configured.get(field):
            raise ValueError(
                f"live execution runtime {field} differs from the frozen config"
            )
    return {field: observed[field] for field in required_fields}


def _load_runtime_class() -> type[Any]:
    from causalcache.policy.gui_owl_v2_runtime import GUIOwlV2Runtime

    return GUIOwlV2Runtime


def _load_gpu_kl_kernel() -> Callable[..., Any]:
    from causalcache.restoration_v2_gpu_kl import gpu_resident_full_vocab_mean_kl

    return gpu_resident_full_vocab_mean_kl


def _decode_rgb_image(payload: bytes) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def _git_head(repository_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_identity(source: object, *, repository_root: Path) -> dict[str, str]:
    source_path = inspect.getsourcefile(source)
    if source_path is None:
        raise ValueError("runtime source file cannot be identified")
    path = Path(source_path).resolve()
    try:
        relative = path.relative_to(repository_root).as_posix()
    except ValueError as error:
        raise ValueError("runtime source is outside the repository") from error
    return {"path": relative, "sha256": sha256_file(path)}


def execute_production_screening(
    args: argparse.Namespace,
    *,
    authorization_validator: Callable[..., Mapping[str, Any]] = authorize_screening,
    artifact_loader: Callable[..., ValidatedScreeningArtifact] = (
        load_validated_screening_artifact
    ),
    runtime_identity_loader: Callable[..., Mapping[str, Any]] = (
        _load_live_execution_runtime_identity
    ),
    runtime_class_loader: Callable[[], type[Any]] = _load_runtime_class,
    kl_kernel_loader: Callable[[], Callable[..., Any]] = _load_gpu_kl_kernel,
    run_executor: Callable[..., dict[str, Any]] = run_substrate_screening,
) -> dict[str, Any]:
    """Authorize, validate data, and only then load the frozen GPU runtime."""
    repository_root = Path(args.repository_root).resolve()
    authorization = authorization_validator(
        execution_config_path=Path(args.execution_config),
        readiness_manifest_path=Path(args.readiness_manifest),
        repository_root=repository_root,
    )
    authorization = validate_screening_authorization_result(authorization)
    _, execution_config = load_json_object(args.execution_config)
    input_bindings = _canonical_input_bindings(
        execution_config,
        repository_root=repository_root,
        scientific_config_path=args.scientific_config,
        selection_manifest_path=args.selection_manifest,
        ocr_backend_config_path=args.ocr_backend_config,
    )
    actual_container_hostname = socket.gethostname()
    bindings = _execution_bindings(
        execution_config,
        repository_root=repository_root,
        device=args.device,
        host_alias=args.host_alias,
        host_hostname=args.host_hostname,
        container_id=args.container_id,
        container_image_digest=args.container_image_digest,
        actual_container_hostname=actual_container_hostname,
    )
    live_runtime_identity = _validate_live_execution_runtime_identity(
        bindings["execution_runtime"],
        runtime_identity_loader(device=args.device),
    )
    bindings["actual_run_identity"].update(live_runtime_identity)
    artifact = artifact_loader(
        artifact_root=args.derived_artifact_root,
        backend_config_path=input_bindings["ocr_backend_config_path"],
        scientific_config_path=input_bindings["scientific_config_path"],
        selection_manifest_path=input_bindings["selection_manifest_path"],
        expected_artifact_tree_sha256=bindings["canonical_data"][
            "artifact_tree_sha256"
        ],
    )

    runtime_class = runtime_class_loader()
    _, scientific_config = load_json_object(input_bindings["scientific_config_path"])
    primary_policy = scientific_config.get("primary_policy")
    visual = (
        primary_policy.get("visual_preprocessing")
        if isinstance(primary_policy, Mapping)
        else None
    )
    target_tokens = (
        visual.get("target_effective_tokens_per_image")
        if isinstance(visual, Mapping)
        else None
    )
    if type(target_tokens) is not int or target_tokens <= 0:
        raise ValueError("scientific visual token target is invalid")
    decoding = (
        primary_policy.get("decoding")
        if isinstance(primary_policy, Mapping)
        else None
    )
    max_new_tokens = (
        decoding.get("max_new_tokens") if isinstance(decoding, Mapping) else None
    )
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("scientific generation token reservation is invalid")
    runtime = runtime_class(
        model_dir=args.model_dir,
        expected_snapshot_manifest=bindings["snapshot_manifest_path"],
        device=args.device,
        target_effective_visual_tokens_per_image=target_tokens,
    )
    shape_sweep = preflight_screening_message_shapes(
        artifact=artifact,
        runtime=runtime,
        image_decoder=_decode_rgb_image,
        max_new_tokens=max_new_tokens,
    )
    distance_backend = GPUFullVocabularyKLBackend(
        torch_module=runtime.torch,
        kl_kernel=kl_kernel_loader(),
    )
    gate = SubstrateGateContract.from_scientific_config(scientific_config)
    run_contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "authorization": authorization,
        "git_commit": _git_head(repository_root),
        "actual_run_identity": bindings["actual_run_identity"],
        "artifact": {
            "root": str(Path(args.derived_artifact_root).resolve()),
            "artifact_tree_sha256": artifact.artifact_tree_sha256,
            "artifact_manifest_sha256": artifact.artifact_manifest_sha256,
            "screening_manifest_sha256": artifact.screening_manifest_sha256,
            "canonical_data": bindings["canonical_data"],
        },
        "policy": {
            "model_dir": str(Path(args.model_dir).resolve()),
            "snapshot_manifest": bindings["snapshot_manifest"],
            "runtime_metadata": dict(runtime.metadata),
        },
        "execution_runtime": bindings["execution_runtime"],
        "canonical_inputs": input_bindings["records"],
        "processor_only_shape_sweep": shape_sweep,
        "gate": asdict(gate),
        "states": [_state_projection(state) for state in artifact.states],
        "sources": {
            "runner": _source_identity(
                execute_production_screening,
                repository_root=repository_root,
            ),
            "runtime": _source_identity(runtime_class, repository_root=repository_root),
        },
        "scientific_config_sha256": sha256_file(
            input_bindings["scientific_config_path"]
        ),
        "selection_manifest_sha256": sha256_file(
            input_bindings["selection_manifest_path"]
        ),
        "execution_config_sha256": sha256_file(args.execution_config),
        "readiness_manifest_sha256": sha256_file(args.readiness_manifest),
    }
    return run_executor(
        artifact=artifact,
        runtime=runtime,
        distance_backend=distance_backend,
        image_decoder=_decode_rgb_image,
        gate=gate,
        run_contract=run_contract,
        output_dir=args.output_dir,
        resume=args.resume,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--execution-config", required=True)
    parser.add_argument("--readiness-manifest", required=True)
    parser.add_argument("--derived-artifact-root", required=True)
    parser.add_argument("--scientific-config", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--ocr-backend-config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--host-hostname", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    aggregate = execute_production_screening(args)
    print(json.dumps(aggregate, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
