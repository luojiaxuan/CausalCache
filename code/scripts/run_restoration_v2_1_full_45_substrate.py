"""Run the frozen restoration-v2.1 full-45 substrate gate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from causalcache.data.guiodyssey_restoration_v2 import sha256_file
from causalcache.data.restoration_v2_1_processor_inputs import (
    V21ProcessorPromptSpec,
    build_v2_1_processor_messages,
)
from causalcache.data.restoration_v2_screening import (
    ScreeningState,
    ValidatedScreeningArtifact,
    load_validated_screening_artifact,
)
from causalcache.policy.gui_owl_v2 import gui_owl_v2_action_to_androidworld
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_PROTOCOL_ID,
    parse_gui_owl_v2_1_output,
)
from causalcache.restoration_v2_1_contract import RestorationV21PilotContract
from causalcache.restoration_v2_1_full_45_contract import (
    CANONICAL_ATTEMPT_ID,
    CANONICAL_CONFIG_PATH,
    CANONICAL_LEDGER_PATH,
    CANONICAL_OUTPUT_DIR,
    FULL_45_PROJECTION_SHA256,
    INVALID_OUTCOME,
    NO_GO_OUTCOME,
    PASS_OUTCOME,
    PROTOCOL_ID,
    RestorationV21Full45Contract,
)
from scripts.run_restoration_v2_1_interface_pilot import (
    CANONICAL_OCR_BACKEND_CONFIG_PATH,
    CANONICAL_PILOT_CONTAINER_ID,
    CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
    CANONICAL_PILOT_DEVICE,
    CANONICAL_PILOT_HOST_ALIAS,
    CANONICAL_PILOT_HOST_HOSTNAME,
    CANONICAL_PROCESSOR_PREFLIGHT_ARTIFACT_PATH,
    CANONICAL_SCIENTIFIC_CONFIG_PATH,
    CANONICAL_SELECTION_MANIFEST_PATH,
    GIT_SHA_PATTERN,
    RUNTIME_GENERATION_BINDING_KEYS,
    RUNTIME_METADATA_KEYS,
    _canonical_json_bytes,
    _canonical_repository_path,
    _decode_rgb_image,
    _duration_seconds,
    _load_live_runtime_identity,
    _screen_dimensions,
    _strict_json_object,
    _utc_now,
    _validate_generation_metadata,
    _validate_runtime_cli_identity,
    _write_json_exclusive,
    validate_clean_pushed_main,
    validate_committed_source_blobs,
)


SCHEMA_VERSION = "1.0.0"
RUN_STATUS = "RESTORATION_V2_1_FULL_45_SUBSTRATE"
ATTEMPT_STATUS = "ATTEMPT_STARTED_NO_RETRY_OR_TOP_UP"
GLOBAL_ATTEMPT_STATUS = "GLOBAL_ATTEMPT_STARTED_DELETION_FORBIDDEN"
STATE_OUTCOME_VALID = "VALID_V2_1_FULL_45_SUBSTRATE_STATE"
STATE_OUTCOME_FAILED = "FAILED_V2_1_FULL_45_SUBSTRATE_STATE"
PASS_OUTCOME = "PASS_V2_1_FULL_45_SUBSTRATE"
NO_GO_OUTCOME = "NO_GO_V2_1_FULL_45_SUBSTRATE"
INVALID_OUTCOME = "INVALID_V2_1_FULL_45_SUBSTRATE"
EXPECTED_STATE_COUNT = 45
EXPECTED_GENERATIONS_PER_STATE = 2
EXPECTED_TEACHER_FORWARDS_PER_ELIGIBLE_STATE = 3
EXPECTED_KL_MEASUREMENTS_PER_ELIGIBLE_STATE = 2
REPEAT_NOISE_FLOOR = 1e-4
REPEAT_NOISE_MULTIPLIER = 10.0
BFLOAT16_MIN_FINITE = -3.3895313892515355e38
CANONICAL_PILOT_ARTIFACT_PATH = (
    "data/results/restoration_v2_1_interface_pilot/artifact.json"
)
RUNNER_SOURCE_PATH = "code/scripts/run_restoration_v2_1_full_45_substrate.py"
RUN_MANIFEST_FILENAME = "run_manifest.json"
RUNTIME_IDENTITY_FILENAME = "runtime_identity.json"
AGGREGATE_FILENAME = "aggregate.json"
STATE_DIRECTORY = "states"
ATTEMPT_DIRECTORY = "attempts"

FORBIDDEN_OPERATION_COUNTS = {
    "retry_count": 0,
    "top_up_count": 0,
    "confirm_state_access_count": 0,
    "confirm_processor_prompt_count": 0,
    "confirm_decoder_input_count": 0,
    "confirm_generation_count": 0,
    "confirm_teacher_forward_count": 0,
    "expert_action_read_count": 0,
    "restoration_coalition_construction_count": 0,
    "restoration_candidate_prompt_count": 0,
    "restoration_label_count": 0,
    "baseline_selection_count": 0,
    "gate_training_example_count": 0,
}


def _replace_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    if not path.is_file():
        raise FileNotFoundError("durable full-45 ledger disappeared before update")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Full45Gate:
    minimum_screening_states: int
    minimum_parse_coverage: float
    minimum_finite_logit_coverage: float
    minimum_repeat_canonical_action_agreement: float
    minimum_memory_sensitive_states: int
    pass_outcome: str
    fail_outcome: str
    invalid_outcome: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Full45Gate":
        expected = {
            "fixed_state_denominator": EXPECTED_STATE_COUNT,
            "minimum_screening_states": 20,
            "minimum_parse_coverage": 0.99,
            "derived_required_parse_success_count": 45,
            "minimum_finite_logit_coverage": 1.0,
            "derived_required_finite_logit_state_count": 45,
            "minimum_repeat_canonical_action_agreement": 1.0,
            "derived_required_repeat_agreement_count": 45,
            "minimum_memory_sensitive_states": 8,
            "memory_sensitive_definition": (
                "summary_reference_kl_greater_than_repeat_noise_epsilon"
            ),
            "repeat_noise_epsilon": "max(1e-4,10*mean_repeat_kl)",
            "failed_states_remain_in_fixed_denominator": True,
            "pass_outcome": PASS_OUTCOME,
            "fail_outcome": NO_GO_OUTCOME,
            "invalid_outcome": INVALID_OUTCOME,
        }
        if dict(value) != expected:
            raise ValueError("v2.1 full-45 gate differs from the frozen contract")
        return cls(
            minimum_screening_states=20,
            minimum_parse_coverage=0.99,
            minimum_finite_logit_coverage=1.0,
            minimum_repeat_canonical_action_agreement=1.0,
            minimum_memory_sensitive_states=8,
            pass_outcome=PASS_OUTCOME,
            fail_outcome=NO_GO_OUTCOME,
            invalid_outcome=INVALID_OUTCOME,
        )


@dataclass(frozen=True)
class DistanceMeasurement:
    value: float
    audit: Mapping[str, Any]


class GPUFullVocabularyKLBackend:
    """Keep full-vocabulary KL on CUDA and transfer one scalar per call."""

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


@dataclass(frozen=True)
class RuntimeBindings:
    runtime_class: type[Any]
    parse_error_class: type[BaseException]
    kl_kernel: Callable[..., Any]


@dataclass(frozen=True)
class AuthorizedFull45:
    repository_root: Path
    contract: RestorationV21Full45Contract
    processor_audit: Mapping[str, Any]
    pilot_authorization: Mapping[str, Any]
    git_identity: Mapping[str, Any]
    source_inventory: Sequence[Mapping[str, str]]
    artifact: ValidatedScreeningArtifact
    states: tuple[ScreeningState, ...]
    snapshot_manifest_path: Path
    target_effective_visual_tokens_per_image: int
    policy_identity: Mapping[str, Any]
    runtime_identity: Mapping[str, Any]
    canonical_inputs: Mapping[str, Any]
    attempt_identity: Mapping[str, Any]
    external_integrity: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class Full45RunLayout:
    states: tuple[ScreeningState, ...]
    projections: tuple[dict[str, Any], ...]
    contract: dict[str, Any]
    contract_sha256: str
    root: Path
    manifest_path: Path
    runtime_identity_path: Path
    state_root: Path
    attempt_root: Path
    aggregate_path: Path
    ledger_path: Path
    attempt_identity: Mapping[str, Any]


def _full45_projection(state: ScreeningState, full45_index: int) -> dict[str, Any]:
    if state.index != full45_index:
        raise ValueError("full-45 state index differs from denominator position")
    return {
        "index": state.index,
        "role": state.role,
        "trajectory_id": state.trajectory_id,
        "decision_step_id": state.decision_step_id,
        "state_id": state.state_id,
        "candidate_event_step_ids": list(state.candidate_event_step_ids),
    }


def select_exact_full45_states(
    artifact: ValidatedScreeningArtifact,
    contract: Any,
) -> tuple[ScreeningState, ...]:
    states = tuple(artifact.states)
    if len(states) != EXPECTED_STATE_COUNT:
        raise ValueError("full-45 artifact must expose exactly 45 screening states")
    if [state.index for state in states] != list(range(EXPECTED_STATE_COUNT)):
        raise ValueError("full-45 source state order drifted")
    roles = [state.role for state in states]
    if roles != ["v2_label_train"] * 30 + ["v2_development"] * 15:
        raise PermissionError("full-45 role order must be 30 label-train then 15 development")
    projections = [
        _full45_projection(state, index) for index, state in enumerate(states)
    ]
    projection_sha256 = hashlib.sha256(_canonical_json_bytes(projections)).hexdigest()
    data = contract.data.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("full-45 contract lacks data binding")
    if (
        projection_sha256 != FULL_45_PROJECTION_SHA256
        or data.get("state_projection_sha256") != FULL_45_PROJECTION_SHA256
        or data.get("state_ids_in_exact_order") != [state.state_id for state in states]
    ):
        raise ValueError("full-45 state projection differs from the frozen contract")
    return states


def build_full45_messages(
    artifact: ValidatedScreeningArtifact,
    state: ScreeningState,
    fidelity: str,
    image_decoder: Callable[[bytes], Any],
) -> list[dict[str, Any]]:
    if state not in artifact.states or state.role not in {
        "v2_label_train",
        "v2_development",
    }:
        raise PermissionError("full-45 message construction received a forbidden state")
    if fidelity == "reference":
        restored = tuple(state.candidate_event_step_ids)
    elif fidelity == "summary_only":
        restored = ()
    else:
        raise ValueError("full-45 fidelity must be reference or summary_only")
    spec = V21ProcessorPromptSpec(
        prompt_index=2 * state.index + int(fidelity == "summary_only"),
        state=state,
        fidelity=fidelity,
        restored_event_step_ids=restored,
    )
    return build_v2_1_processor_messages(
        artifact,
        spec,
        image_decoder=image_decoder,
    )


def _action_arguments(action: Any) -> dict[str, Any]:
    arguments = action.arguments()
    if not isinstance(arguments, Mapping):
        raise TypeError("canonical action arguments must be a mapping")
    return dict(arguments)


def _failure(
    *,
    stage: str,
    category: str,
    message: str,
    exception_type: str | None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "category": category,
        "exception_type": exception_type,
        "message": message,
    }


def _base_state_record(
    *,
    projection: Mapping[str, Any],
    run_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_contract_sha256": run_contract_sha256,
        "state": dict(projection),
        "outcome": STATE_OUTCOME_FAILED,
        "failure": None,
        "parse_success": False,
        "parse_success_count": 0,
        "model_emitted_closer_count": 0,
        "androidworld_bridge_count": 0,
        "repeat_canonical_action_agreement": False,
        "finite_logit_distances": False,
        "canonical_action": None,
        "androidworld_bridge": None,
        "screen_dimensions": None,
        "native_generations": [],
        "teacher_forwards": {},
        "distances": {
            "repeat_reference_kl": None,
            "summary_reference_kl": None,
        },
        "distance_audits": {},
        "operation_counts": {
            "generation_call_count": 0,
            "teacher_forward_count": 0,
            "kl_measurement_count": 0,
            **FORBIDDEN_OPERATION_COUNTS,
        },
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


def run_full45_state_once(
    *,
    artifact: ValidatedScreeningArtifact,
    state: ScreeningState,
    full45_index: int,
    runtime: Any,
    parse_error_class: type[BaseException],
    distance_backend: Any,
    image_decoder: Callable[[bytes], Any],
    run_contract_sha256: str,
    message_builder: Callable[
        [ValidatedScreeningArtifact, ScreeningState, str, Callable[[bytes], Any]],
        list[dict[str, Any]],
    ] = build_full45_messages,
) -> dict[str, Any]:
    """Execute one immutable state without retry, filtering, or top-up."""
    started = time.perf_counter()
    record = _base_state_record(
        projection=_full45_projection(state, full45_index),
        run_contract_sha256=run_contract_sha256,
    )
    reference_messages = message_builder(
        artifact,
        state,
        "reference",
        image_decoder,
    )
    summary_messages = message_builder(
        artifact,
        state,
        "summary_only",
        image_decoder,
    )
    reference_dimensions = _screen_dimensions(reference_messages)
    summary_dimensions = _screen_dimensions(summary_messages)
    if reference_dimensions != summary_dimensions:
        raise RuntimeError("reference and summary current-observation dimensions differ")
    width, height = reference_dimensions
    record["screen_dimensions"] = {"width": width, "height": height}
    parsed_generations: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    parse_failures: list[dict[str, Any]] = []
    for repeat_index in (1, 2):
        record["operation_counts"]["generation_call_count"] += 1
        try:
            generation = runtime.generate_native_action(reference_messages)
        except parse_error_class as error:
            output_text = getattr(error, "output_text", None)
            metadata = getattr(error, "metadata", None)
            if not isinstance(output_text, str) or not isinstance(metadata, Mapping):
                raise RuntimeError("generation parse error lost native evidence") from error
            _validate_generation_metadata(metadata, raw_output=output_text)
            record["model_emitted_closer_count"] += int(
                metadata.get("model_emitted_tool_call_close") is True
            )
            generation_record = {
                "repeat_index": repeat_index,
                "output_text": output_text,
                "metadata": dict(metadata),
                "canonical_action": None,
                "androidworld_bridge": None,
                "parse_error_type": str(
                    getattr(error, "parse_error_type", error.__class__.__name__)
                ),
                "parse_error_message": str(
                    getattr(error, "parse_error_message", str(error))
                ),
            }
            record["native_generations"].append(generation_record)
            parse_failures.append(generation_record)
            continue
        output_text = getattr(generation, "output_text", None)
        metadata = getattr(generation, "metadata", None)
        if not isinstance(output_text, str) or not isinstance(metadata, Mapping):
            raise RuntimeError("runtime generation result lost native evidence")
        _validate_generation_metadata(metadata, raw_output=output_text)
        parsed = parse_gui_owl_v2_1_output(output_text)
        if (
            metadata.get("model_emitted_tool_call_close") is not True
            or metadata.get("generated_tool_call_close_token_count") != 1
            or metadata.get("final_generated_token_id") != 151658
        ):
            raise RuntimeError("parseable generation lacks the model-emitted closer")
        runtime_arguments = _action_arguments(
            generation.parsed_output.canonical_action
        )
        parsed_arguments = _action_arguments(parsed.canonical_action)
        if runtime_arguments != parsed_arguments:
            raise RuntimeError("runtime and independent parser actions differ")
        bridge = gui_owl_v2_action_to_androidworld(
            parsed.canonical_action,
            screen_width=width,
            screen_height=height,
        )
        record["parse_success_count"] += 1
        record["model_emitted_closer_count"] += int(
            metadata.get("model_emitted_tool_call_close") is True
        )
        record["androidworld_bridge_count"] += 1
        record["native_generations"].append(
            {
                "repeat_index": repeat_index,
                "output_text": output_text,
                "metadata": dict(metadata),
                "canonical_action": parsed_arguments,
                "androidworld_bridge": bridge,
                "parse_error_type": None,
                "parse_error_message": None,
            }
        )
        parsed_generations.append((parsed.canonical_action, parsed_arguments, bridge))

    if parse_failures:
        return _finalize_state_record(
            record,
            started=started,
            outcome=STATE_OUTCOME_FAILED,
            failure=_failure(
                stage="reference_generations",
                category="PARSE_FAILURE",
                exception_type=str(parse_failures[0]["parse_error_type"]),
                message=(
                    f"{len(parse_failures)} of 2 strict reference generations failed: "
                    f"{parse_failures[0]['parse_error_message']}"
                ),
            ),
        )
    record["parse_success"] = True
    first_action, first_arguments, first_bridge = parsed_generations[0]
    second_arguments = parsed_generations[1][1]
    if first_arguments != second_arguments:
        return _finalize_state_record(
            record,
            started=started,
            outcome=STATE_OUTCOME_FAILED,
            failure=_failure(
                stage="reference_generation_repeat_comparison",
                category="CANONICAL_ACTION_MISMATCH",
                exception_type=None,
                message="two deterministic reference generations produced different actions",
            ),
        )
    record["repeat_canonical_action_agreement"] = True
    record["canonical_action"] = first_arguments
    record["androidworld_bridge"] = first_bridge

    record["operation_counts"]["teacher_forward_count"] += 1
    reference_logits, reference_metadata = runtime.teacher_forced_distance_logits(
        (reference_messages,),
        (first_action,),
    )
    reference_log_probs = distance_backend.prepare_reference(reference_logits)
    del reference_logits
    record["teacher_forwards"]["reference_1"] = dict(reference_metadata)

    record["operation_counts"]["teacher_forward_count"] += 1
    repeat_logits, repeat_metadata = runtime.teacher_forced_distance_logits(
        (reference_messages,),
        (first_action,),
    )
    record["operation_counts"]["kl_measurement_count"] += 1
    repeat_measurement = distance_backend.measure(reference_log_probs, repeat_logits)
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

    record["operation_counts"]["teacher_forward_count"] += 1
    summary_logits, summary_metadata = runtime.teacher_forced_distance_logits(
        (summary_messages,),
        (first_action,),
    )
    record["operation_counts"]["kl_measurement_count"] += 1
    summary_measurement = distance_backend.measure(reference_log_probs, summary_logits)
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

    if not repeat_is_finite or not summary_is_finite:
        return _finalize_state_record(
            record,
            started=started,
            outcome=STATE_OUTCOME_FAILED,
            failure=_failure(
                stage="gpu_distance_validation",
                category="NONFINITE_DISTANCE",
                exception_type=None,
                message="repeat or summary-reference KL is non-finite",
            ),
        )
    record["finite_logit_distances"] = True
    return _finalize_state_record(
        record,
        started=started,
        outcome=STATE_OUTCOME_VALID,
        failure=None,
    )


def _state_record_keys() -> set[str]:
    return {
        "schema_version",
        "protocol_id",
        "run_contract_sha256",
        "state",
        "outcome",
        "failure",
        "parse_success",
        "parse_success_count",
        "model_emitted_closer_count",
        "androidworld_bridge_count",
        "repeat_canonical_action_agreement",
        "finite_logit_distances",
        "canonical_action",
        "androidworld_bridge",
        "screen_dimensions",
        "native_generations",
        "teacher_forwards",
        "distances",
        "distance_audits",
        "operation_counts",
        "started_at_utc",
        "ended_at_utc",
        "duration_seconds",
    }


def _validate_teacher_metadata(
    metadata: Mapping[str, Any],
    *,
    runtime_metadata: Mapping[str, Any] | None,
) -> None:
    expected = {
        "batch_size": 1,
        "device": CANONICAL_PILOT_DEVICE,
        "dtype": "torch.bfloat16",
        "distance_span": "official_tool_call_open_through_close_inclusive",
        "teacher_context": "official_tools_prompt_plus_assistant_prefix_direct",
        "teacher_carrier": None,
        "teacher_target_json_separators": [", ", ": "],
        "teacher_target_ends_with_model_generation_eos": True,
        "teacher_target_disjoint_from_suppressed_standard_eos": True,
        "teacher_standard_eos_suppressed_token_ids": [151645, 151643],
        "teacher_standard_eos_suppression_semantics": (
            "torch_finfo_bfloat16_min_finite_generation_alignment"
        ),
        "teacher_standard_eos_mask_application": (
            "same_mask_on_every_reference_and_candidate_action_path_position_"
            "before_float32_log_softmax"
        ),
        "teacher_raw_logits_mutated": False,
        "finite_logits_validation": (
            "deferred_to_gpu_kl_invalid_to_nan_final_distance"
        ),
        "full_logit_tensor_host_transfers": 0,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"teacher metadata {key} drifted")
    suppression = metadata.get("teacher_standard_eos_suppression_value")
    if (
        isinstance(suppression, bool)
        or not isinstance(suppression, (int, float))
        or float(suppression) != BFLOAT16_MIN_FINITE
    ):
        raise ValueError("teacher exact BF16 finite-min suppression value drifted")
    logits_to_keep = metadata.get("logits_to_keep")
    vocabulary_size = metadata.get("vocabulary_size")
    samples = metadata.get("samples")
    if (
        type(logits_to_keep) is not int
        or logits_to_keep <= 0
        or type(vocabulary_size) is not int
        or vocabulary_size <= 151658
        or not isinstance(samples, list)
        or len(samples) != 1
        or not isinstance(samples[0], Mapping)
        or samples[0].get("distance_action_tokens") != logits_to_keep
    ):
        raise ValueError("teacher distance-span shape metadata drifted")
    if runtime_metadata is not None:
        if any(
            metadata.get(key) != runtime_metadata.get(key)
            for key in RUNTIME_GENERATION_BINDING_KEYS
        ):
            raise ValueError("teacher metadata differs from runtime identity")
        if metadata.get("teacher_standard_eos_suppressed_token_ids") != runtime_metadata.get(
            "suppressed_standard_eos_token_ids"
        ):
            raise ValueError("teacher suppression IDs differ from generation runtime")


def _validate_terminal_state_record(
    record: Mapping[str, Any],
    *,
    projection: Mapping[str, Any],
    run_contract_sha256: str,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> None:
    if set(record) != _state_record_keys():
        raise ValueError("full-45 terminal state record keys drifted")
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("protocol_id") != PROTOCOL_ID
        or record.get("run_contract_sha256") != run_contract_sha256
        or record.get("state") != dict(projection)
        or record.get("outcome") not in {
            STATE_OUTCOME_VALID,
            STATE_OUTCOME_FAILED,
        }
    ):
        raise ValueError("full-45 terminal state identity drifted")
    dimensions = record.get("screen_dimensions")
    if not isinstance(dimensions, Mapping) or set(dimensions) != {"width", "height"}:
        raise ValueError("full-45 terminal state lacks screen dimensions")
    width = dimensions.get("width")
    height = dimensions.get("height")
    if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
        raise ValueError("full-45 screen dimensions are invalid")
    operations = record.get("operation_counts")
    expected_operation_keys = {
        "generation_call_count",
        "teacher_forward_count",
        "kl_measurement_count",
        *FORBIDDEN_OPERATION_COUNTS,
    }
    if not isinstance(operations, Mapping) or set(operations) != expected_operation_keys:
        raise ValueError("full-45 operation-count schema drifted")
    if (
        operations.get("generation_call_count") != 2
        or type(operations.get("teacher_forward_count")) is not int
        or operations["teacher_forward_count"] not in {0, 3}
        or type(operations.get("kl_measurement_count")) is not int
        or operations["kl_measurement_count"] not in {0, 2}
        or any(operations.get(key) != 0 for key in FORBIDDEN_OPERATION_COUNTS)
    ):
        raise ValueError("full-45 operation counts exceed the frozen schedule")
    native = record.get("native_generations")
    if not isinstance(native, list) or len(native) != 2:
        raise ValueError("full-45 state must preserve exactly two native generations")
    parsed_actions: list[dict[str, Any]] = []
    parsed_bridges: list[dict[str, Any]] = []
    closer_count = 0
    for repeat_index, generation in enumerate(native, start=1):
        if not isinstance(generation, Mapping) or set(generation) != {
            "repeat_index",
            "output_text",
            "metadata",
            "canonical_action",
            "androidworld_bridge",
            "parse_error_type",
            "parse_error_message",
        }:
            raise ValueError("full-45 native generation schema drifted")
        output = generation.get("output_text")
        metadata = generation.get("metadata")
        if (
            generation.get("repeat_index") != repeat_index
            or not isinstance(output, str)
            or not isinstance(metadata, Mapping)
        ):
            raise ValueError("full-45 native generation identity drifted")
        _validate_generation_metadata(metadata, raw_output=output)
        if runtime_metadata is not None and any(
            metadata.get(key) != runtime_metadata.get(key)
            for key in RUNTIME_GENERATION_BINDING_KEYS
        ):
            raise ValueError("generation metadata differs from runtime identity")
        closer_count += int(metadata.get("model_emitted_tool_call_close") is True)
        try:
            parsed = parse_gui_owl_v2_1_output(output)
        except (TypeError, ValueError):
            if (
                generation.get("canonical_action") is not None
                or generation.get("androidworld_bridge") is not None
                or not isinstance(generation.get("parse_error_type"), str)
                or not generation["parse_error_type"]
                or not isinstance(generation.get("parse_error_message"), str)
                or not generation["parse_error_message"]
            ):
                raise ValueError("parse failure generation evidence drifted")
            continue
        arguments = _action_arguments(parsed.canonical_action)
        bridge = gui_owl_v2_action_to_androidworld(
            parsed.canonical_action,
            screen_width=width,
            screen_height=height,
        )
        if (
            metadata.get("model_emitted_tool_call_close") is not True
            or metadata.get("generated_tool_call_close_token_count") != 1
            or generation.get("canonical_action") != arguments
            or generation.get("androidworld_bridge") != bridge
            or generation.get("parse_error_type") is not None
            or generation.get("parse_error_message") is not None
        ):
            raise ValueError("parseable generation evidence drifted")
        parsed_actions.append(arguments)
        parsed_bridges.append(bridge)

    parse_count = len(parsed_actions)
    parse_success = parse_count == 2
    agreement = parse_success and parsed_actions[0] == parsed_actions[1]
    if (
        record.get("parse_success_count") != parse_count
        or record.get("parse_success") is not parse_success
        or record.get("model_emitted_closer_count") != closer_count
        or record.get("androidworld_bridge_count") != parse_count
        or record.get("repeat_canonical_action_agreement") is not agreement
    ):
        raise ValueError("full-45 recomputed parse/agreement metrics drifted")
    teacher = record.get("teacher_forwards")
    distances = record.get("distances")
    audits = record.get("distance_audits")
    if (
        not isinstance(teacher, Mapping)
        or not isinstance(distances, Mapping)
        or set(distances) != {"repeat_reference_kl", "summary_reference_kl"}
        or not isinstance(audits, Mapping)
    ):
        raise ValueError("full-45 teacher/distance schema drifted")
    failure = record.get("failure")
    finite = record.get("finite_logit_distances") is True
    if not agreement:
        if (
            operations["teacher_forward_count"] != 0
            or operations["kl_measurement_count"] != 0
            or teacher
            or audits
            or any(value is not None for value in distances.values())
            or record.get("canonical_action") is not None
            or record.get("androidworld_bridge") is not None
            or finite
            or record.get("outcome") != STATE_OUTCOME_FAILED
            or not isinstance(failure, Mapping)
            or failure.get("category")
            not in {"PARSE_FAILURE", "CANONICAL_ACTION_MISMATCH"}
        ):
            raise ValueError("pre-teacher scientific failure evidence drifted")
    else:
        if (
            operations["teacher_forward_count"] != 3
            or operations["kl_measurement_count"] != 2
            or set(teacher) != {"reference_1", "reference_2", "summary_only"}
            or set(audits) != {"repeat_reference_kl", "summary_reference_kl"}
            or record.get("canonical_action") != parsed_actions[0]
            or record.get("androidworld_bridge") != parsed_bridges[0]
        ):
            raise ValueError("post-agreement teacher evidence drifted")
        if runtime_metadata is not None:
            for metadata in teacher.values():
                if not isinstance(metadata, Mapping):
                    raise ValueError("teacher metadata record is not a mapping")
                _validate_teacher_metadata(
                    metadata,
                    runtime_metadata=runtime_metadata,
                )
        else:
            for metadata in teacher.values():
                if not isinstance(metadata, Mapping):
                    raise ValueError("teacher metadata record is not a mapping")
                _validate_teacher_metadata(metadata, runtime_metadata=None)
        finite_values = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in distances.values()
        )
        if finite is not finite_values:
            raise ValueError("finite-distance flag differs from persisted distances")
        if finite:
            if record.get("outcome") != STATE_OUTCOME_VALID or failure is not None:
                raise ValueError("finite full-45 state is not valid")
        elif (
            record.get("outcome") != STATE_OUTCOME_FAILED
            or not isinstance(failure, Mapping)
            or failure.get("category") != "NONFINITE_DISTANCE"
            or all(value is not None for value in distances.values())
        ):
            raise ValueError("non-finite scientific failure evidence drifted")
    started_at = record.get("started_at_utc")
    ended_at = record.get("ended_at_utc")
    duration = record.get("duration_seconds")
    if (
        not isinstance(started_at, str)
        or not isinstance(ended_at, str)
        or not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or duration < 0
    ):
        raise ValueError("full-45 terminal state timing drifted")
    utc_duration = _duration_seconds(started_at, ended_at)
    if abs(float(duration) - utc_duration) > 1.0:
        raise ValueError("full-45 monotonic and UTC state durations differ")


def aggregate_full45_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_states: Sequence[ScreeningState],
    gate: Full45Gate,
    run_contract_sha256: str,
    started_at_utc: str,
    ended_at_utc: str,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the full fixed denominator and apply the frozen gate."""
    if len(records) != EXPECTED_STATE_COUNT or len(expected_states) != EXPECTED_STATE_COUNT:
        raise ValueError("full-45 gate requires exactly 45 states and records")
    for index, (record, state) in enumerate(zip(records, expected_states, strict=True)):
        _validate_terminal_state_record(
            record,
            projection=_full45_projection(state, index),
            run_contract_sha256=run_contract_sha256,
            runtime_metadata=runtime_metadata,
        )
    parse_count = sum(record["parse_success"] is True for record in records)
    finite_count = sum(record["finite_logit_distances"] is True for record in records)
    agreement_count = sum(
        record["repeat_canonical_action_agreement"] is True for record in records
    )
    repeat_values = [
        float(record["distances"]["repeat_reference_kl"])
        for record in records
        if isinstance(record["distances"]["repeat_reference_kl"], (int, float))
        and not isinstance(record["distances"]["repeat_reference_kl"], bool)
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
    memory_sensitive_state_ids: list[str] = []
    if repeat_noise_epsilon is not None:
        for record in records:
            value = record["distances"]["summary_reference_kl"]
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > repeat_noise_epsilon
            ):
                memory_sensitive_state_ids.append(str(record["state"]["state_id"]))
    generation_count = sum(
        int(record["operation_counts"]["generation_call_count"])
        for record in records
    )
    teacher_count = sum(
        int(record["operation_counts"]["teacher_forward_count"])
        for record in records
    )
    kl_count = sum(
        int(record["operation_counts"]["kl_measurement_count"])
        for record in records
    )
    forbidden = {
        key: sum(int(record["operation_counts"][key]) for record in records)
        for key in FORBIDDEN_OPERATION_COUNTS
    }
    metrics = {
        "fixed_state_denominator": EXPECTED_STATE_COUNT,
        "parse_success_count": parse_count,
        "parse_coverage": parse_count / EXPECTED_STATE_COUNT,
        "finite_logit_state_count": finite_count,
        "finite_logit_coverage": finite_count / EXPECTED_STATE_COUNT,
        "repeat_canonical_action_agreement_count": agreement_count,
        "repeat_canonical_action_agreement": agreement_count / EXPECTED_STATE_COUNT,
        "finite_repeat_kl_count": len(repeat_values),
        "mean_repeat_kl": mean_repeat_kl,
        "repeat_noise_epsilon": repeat_noise_epsilon,
        "memory_sensitive_state_count": len(memory_sensitive_state_ids),
        "memory_sensitive_state_ids": memory_sensitive_state_ids,
        "generation_call_count": generation_count,
        "teacher_forward_count": teacher_count,
        "kl_measurement_count": kl_count,
        "model_emitted_closer_count": sum(
            int(record["model_emitted_closer_count"]) for record in records
        ),
        "androidworld_bridge_count": sum(
            int(record["androidworld_bridge_count"]) for record in records
        ),
        **forbidden,
    }
    checks = {
        "minimum_screening_states": EXPECTED_STATE_COUNT >= gate.minimum_screening_states,
        "minimum_parse_coverage": metrics["parse_coverage"] >= gate.minimum_parse_coverage,
        "minimum_finite_logit_coverage": (
            metrics["finite_logit_coverage"] >= gate.minimum_finite_logit_coverage
        ),
        "minimum_repeat_canonical_action_agreement": (
            metrics["repeat_canonical_action_agreement"]
            >= gate.minimum_repeat_canonical_action_agreement
        ),
        "minimum_memory_sensitive_states": (
            metrics["memory_sensitive_state_count"]
            >= gate.minimum_memory_sensitive_states
        ),
        "maximum_generation_calls": generation_count <= 90,
        "maximum_teacher_forwards": teacher_count <= 135,
        "maximum_kl_measurements": kl_count <= 90,
        "prohibited_work_zero": all(value == 0 for value in forbidden.values()),
    }
    passed = all(checks.values())
    failure_counts = Counter(
        str(record["failure"]["category"])
        for record in records
        if isinstance(record.get("failure"), Mapping)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_contract_sha256": run_contract_sha256,
        "status": "COMPLETED_FIXED_45_STATE_V2_1_SUBSTRATE",
        "outcome": gate.pass_outcome if passed else gate.fail_outcome,
        "gate_passed": passed,
        "gate_contract": asdict(gate),
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_seconds": _duration_seconds(started_at_utc, ended_at_utc),
        "metrics": metrics,
        "checks": checks,
        "failure_category_counts": dict(sorted(failure_counts.items())),
        "sample_mutation_performed": False,
        "top_up_performed": False,
        "confirm_role_used": False,
        "expert_action_read": False,
        "restoration_work_performed": False,
    }


def _is_out_of_memory(error: BaseException) -> bool:
    return (
        "outofmemory" in error.__class__.__name__.casefold()
        or "out of memory" in str(error).casefold()
    )


def _invalid_aggregate(
    *,
    run_contract_sha256: str,
    stage: str,
    error: BaseException,
    completed_state_count: int,
    attempted_state_count: int,
    observed_inventory: Mapping[str, Any],
    started_at_utc: str,
) -> dict[str, Any]:
    ended_at_utc = _utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_contract_sha256": run_contract_sha256,
        "status": "TERMINATED_INVALID_V2_1_FULL_45_SUBSTRATE",
        "outcome": INVALID_OUTCOME,
        "gate_passed": False,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_seconds": _duration_seconds(started_at_utc, ended_at_utc),
        "invalid_failure": {
            "stage": stage,
            "category": "OUT_OF_MEMORY" if _is_out_of_memory(error) else "CONTRACT_OR_RUNTIME",
            "exception_type": error.__class__.__name__,
            "message": str(error) or repr(error),
        },
        "completed_state_count": completed_state_count,
        "attempted_state_count": attempted_state_count,
        "observed_inventory": dict(observed_inventory),
        "retry_count": 0,
        "top_up_count": 0,
        "prohibited_operation_counts": {
            key: 0 for key in FORBIDDEN_OPERATION_COUNTS if key not in {"retry_count", "top_up_count"}
        },
    }


def _prepare_layout(
    *,
    states: Sequence[ScreeningState],
    run_contract: Mapping[str, Any],
    output_dir: str | Path,
) -> Full45RunLayout:
    selected = tuple(states)
    if len(selected) != EXPECTED_STATE_COUNT:
        raise ValueError("full-45 runner requires exactly 45 states")
    projections = tuple(
        _full45_projection(state, index) for index, state in enumerate(selected)
    )
    contract = dict(run_contract)
    if contract.get("states") != list(projections):
        raise ValueError("full-45 run contract denominator drifted")
    if contract.get("prohibited_operation_counts") != FORBIDDEN_OPERATION_COUNTS:
        raise ValueError("full-45 run contract prohibited-work declaration drifted")
    attempt_identity = contract.get("attempt_identity")
    expected_identity = {
        "attempt_id": CANONICAL_ATTEMPT_ID,
        "canonical_persistent_output_dir": str(CANONICAL_OUTPUT_DIR),
        "canonical_global_attempt_ledger": str(CANONICAL_LEDGER_PATH),
        "canonical_host_alias": CANONICAL_PILOT_HOST_ALIAS,
        "canonical_host_hostname": CANONICAL_PILOT_HOST_HOSTNAME,
        "canonical_container_id": CANONICAL_PILOT_CONTAINER_ID,
        "canonical_container_image_digest": CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
        "canonical_device": CANONICAL_PILOT_DEVICE,
        "cross_host_attempt_allowed": False,
        "alternate_output_or_ledger_allowed": False,
        "output_or_ledger_deletion_after_first_attempt_allowed": False,
    }
    if not isinstance(attempt_identity, Mapping) or dict(attempt_identity) != expected_identity:
        raise ValueError("full-45 run contract attempt identity drifted")
    root = Path(output_dir).resolve()
    if root != CANONICAL_OUTPUT_DIR:
        raise ValueError("alternate full-45 output directory is forbidden")
    contract_sha256 = hashlib.sha256(_canonical_json_bytes(contract)).hexdigest()
    return Full45RunLayout(
        states=selected,
        projections=projections,
        contract=contract,
        contract_sha256=contract_sha256,
        root=root,
        manifest_path=root / RUN_MANIFEST_FILENAME,
        runtime_identity_path=root / RUNTIME_IDENTITY_FILENAME,
        state_root=root / STATE_DIRECTORY,
        attempt_root=root / ATTEMPT_DIRECTORY,
        aggregate_path=root / AGGREGATE_FILENAME,
        ledger_path=CANONICAL_LEDGER_PATH,
        attempt_identity=dict(attempt_identity),
    )


def _global_attempt_ledger(
    *,
    attempt_identity: Mapping[str, Any],
    run_contract_sha256: str,
    created_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": GLOBAL_ATTEMPT_STATUS,
        "attempt_identity": dict(attempt_identity),
        "run_contract_sha256": run_contract_sha256,
        "created_at_utc": created_at_utc,
        "attempted_state_count": 0,
        "completed_state_count": 0,
        "journal": [
            {
                "sequence": 0,
                "event": "GLOBAL_ATTEMPT_CLAIMED",
                "state_index": None,
                "attempted_state_count": 0,
                "completed_state_count": 0,
                "created_at_utc": created_at_utc,
            }
        ],
    }


def _validate_global_attempt_ledger(
    path: Path,
    *,
    attempt_identity: Mapping[str, Any],
    run_contract_sha256: str,
) -> dict[str, Any]:
    ledger = _strict_json_object(path)
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "attempt_identity",
        "run_contract_sha256",
        "created_at_utc",
        "attempted_state_count",
        "completed_state_count",
        "journal",
    }
    created = ledger.get("created_at_utc")
    if (
        set(ledger) != expected_keys
        or ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("protocol_id") != PROTOCOL_ID
        or ledger.get("status") != GLOBAL_ATTEMPT_STATUS
        or ledger.get("attempt_identity") != dict(attempt_identity)
        or ledger.get("run_contract_sha256") != run_contract_sha256
        or not isinstance(created, str)
    ):
        raise ValueError("full-45 global attempt ledger identity drifted")
    _duration_seconds(created, created)
    attempted = ledger.get("attempted_state_count")
    completed = ledger.get("completed_state_count")
    journal = ledger.get("journal")
    if (
        type(attempted) is not int
        or type(completed) is not int
        or not 0 <= completed <= attempted <= EXPECTED_STATE_COUNT
        or attempted - completed > 1
        or not isinstance(journal, list)
        or len(journal) != 1 + attempted + completed
    ):
        raise ValueError("full-45 durable ledger high-water drifted")
    previous_attempted = 0
    previous_completed = 0
    previous_timestamp = created
    for sequence, entry in enumerate(journal):
        if not isinstance(entry, Mapping) or set(entry) != {
            "sequence",
            "event",
            "state_index",
            "attempted_state_count",
            "completed_state_count",
            "created_at_utc",
        }:
            raise ValueError("full-45 durable ledger journal schema drifted")
        timestamp = entry.get("created_at_utc")
        if entry.get("sequence") != sequence or not isinstance(timestamp, str):
            raise ValueError("full-45 durable ledger journal identity drifted")
        _duration_seconds(previous_timestamp, timestamp)
        event = entry.get("event")
        entry_attempted = entry.get("attempted_state_count")
        entry_completed = entry.get("completed_state_count")
        state_index = entry.get("state_index")
        if sequence == 0:
            if dict(entry) != {
                "sequence": 0,
                "event": "GLOBAL_ATTEMPT_CLAIMED",
                "state_index": None,
                "attempted_state_count": 0,
                "completed_state_count": 0,
                "created_at_utc": created,
            }:
                raise ValueError("full-45 initial durable journal entry drifted")
        elif event == "STATE_ATTEMPT_CLAIMED":
            if (
                previous_attempted != previous_completed
                or state_index != previous_attempted
                or entry_attempted != previous_attempted + 1
                or entry_completed != previous_completed
            ):
                raise ValueError("full-45 attempted high-water transition drifted")
        elif event == "STATE_TERMINAL_PERSISTED":
            if (
                previous_attempted != previous_completed + 1
                or state_index != previous_completed
                or entry_attempted != previous_attempted
                or entry_completed != previous_completed + 1
            ):
                raise ValueError("full-45 completed high-water transition drifted")
        else:
            raise ValueError("full-45 durable journal event drifted")
        previous_attempted = int(entry_attempted)
        previous_completed = int(entry_completed)
        previous_timestamp = timestamp
    if previous_attempted != attempted or previous_completed != completed:
        raise ValueError("full-45 durable journal tail differs from high-water")
    return ledger


def _advance_ledger_high_water(
    layout: Full45RunLayout,
    *,
    event: str,
    state_index: int,
) -> dict[str, Any]:
    ledger = _validate_global_attempt_ledger(
        layout.ledger_path,
        attempt_identity=layout.attempt_identity,
        run_contract_sha256=layout.contract_sha256,
    )
    attempted = int(ledger["attempted_state_count"])
    completed = int(ledger["completed_state_count"])
    if event == "STATE_ATTEMPT_CLAIMED":
        if attempted != completed or state_index != attempted:
            raise ValueError("attempted high-water advance is out of order")
        attempted += 1
    elif event == "STATE_TERMINAL_PERSISTED":
        if attempted != completed + 1 or state_index != completed:
            raise ValueError("completed high-water advance is out of order")
        completed += 1
    else:
        raise ValueError("unknown full-45 durable journal event")
    timestamp = _utc_now()
    updated = dict(ledger)
    updated["attempted_state_count"] = attempted
    updated["completed_state_count"] = completed
    updated["journal"] = [
        *ledger["journal"],
        {
            "sequence": len(ledger["journal"]),
            "event": event,
            "state_index": state_index,
            "attempted_state_count": attempted,
            "completed_state_count": completed,
            "created_at_utc": timestamp,
        },
    ]
    _replace_json_durable(layout.ledger_path, updated)
    reread = _validate_global_attempt_ledger(
        layout.ledger_path,
        attempt_identity=layout.attempt_identity,
        run_contract_sha256=layout.contract_sha256,
    )
    if reread != updated:
        raise ValueError("durable ledger reread differs after atomic high-water advance")
    return reread


def _run_manifest_record(
    layout: Full45RunLayout,
    *,
    created_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": RUN_STATUS,
        "run_contract_sha256": layout.contract_sha256,
        "run_contract": layout.contract,
        "created_at_utc": created_at_utc,
    }


def _validate_run_manifest(
    record: Mapping[str, Any],
    *,
    layout: Full45RunLayout,
) -> str:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "run_contract_sha256",
        "run_contract",
        "created_at_utc",
    }
    created = record.get("created_at_utc")
    if (
        set(record) != expected_keys
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("protocol_id") != PROTOCOL_ID
        or record.get("status") != RUN_STATUS
        or record.get("run_contract_sha256") != layout.contract_sha256
        or record.get("run_contract") != layout.contract
        or not isinstance(created, str)
    ):
        raise ValueError("full-45 run manifest identity drifted")
    _duration_seconds(created, created)
    return created


def _runtime_identity_record(
    *,
    layout: Full45RunLayout,
    runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = dict(runtime_metadata)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "VALIDATED_POLICY_RUNTIME_AFTER_DURABLE_FULL_45_CLAIM",
        "run_contract_sha256": layout.contract_sha256,
        "runtime_metadata_sha256": hashlib.sha256(
            _canonical_json_bytes(metadata)
        ).hexdigest(),
        "runtime_metadata": metadata,
        "created_at_utc": _utc_now(),
    }


def _validate_runtime_identity_record(
    record: Mapping[str, Any],
    *,
    layout: Full45RunLayout,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "run_contract_sha256",
        "runtime_metadata_sha256",
        "runtime_metadata",
        "created_at_utc",
    }
    metadata = record.get("runtime_metadata")
    policy = layout.contract.get("policy")
    requirement = (
        policy.get("runtime_metadata_requirement")
        if isinstance(policy, Mapping)
        else None
    )
    if not isinstance(metadata, Mapping) or not isinstance(requirement, Mapping):
        raise ValueError("full-45 runtime metadata contract is missing")
    metadata_value = dict(metadata)
    metadata_sha256 = hashlib.sha256(_canonical_json_bytes(metadata_value)).hexdigest()
    created = record.get("created_at_utc")
    if (
        set(record) != expected_keys
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("protocol_id") != PROTOCOL_ID
        or record.get("status")
        != "VALIDATED_POLICY_RUNTIME_AFTER_DURABLE_FULL_45_CLAIM"
        or record.get("run_contract_sha256") != layout.contract_sha256
        or record.get("runtime_metadata_sha256") != metadata_sha256
        or set(metadata_value) != RUNTIME_METADATA_KEYS
        or requirement.get("required_metadata_keys") != sorted(RUNTIME_METADATA_KEYS)
        or metadata_value.get("protocol_id")
        != requirement.get("required_policy_protocol_id")
        or metadata_value.get("model_dir") != requirement.get("model_dir")
        or metadata_value.get("model_repo") != requirement.get("model_repo")
        or metadata_value.get("model_revision") != requirement.get("model_revision")
        or metadata_value.get("snapshot_manifest_sha256")
        != requirement.get("snapshot_manifest_sha256")
        or metadata_value.get("device") != requirement.get("device")
        or metadata_value.get("target_effective_visual_tokens_per_image")
        != requirement.get("target_effective_visual_tokens_per_image")
        or metadata_value.get("frozen") is not True
        or metadata_value.get("single_device") is not True
        or not isinstance(created, str)
    ):
        raise ValueError("full-45 runtime identity drifted")
    _duration_seconds(created, created)
    return metadata_value


def _attempt_marker(
    *,
    projection: Mapping[str, Any],
    run_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": ATTEMPT_STATUS,
        "run_contract_sha256": run_contract_sha256,
        "state": dict(projection),
        "attempt_ordinal": 1,
        "retry_count": 0,
        "top_up_count": 0,
        "created_at_utc": _utc_now(),
    }


def _validate_attempt_marker(
    path: Path,
    *,
    projection: Mapping[str, Any],
    run_contract_sha256: str,
) -> None:
    marker = _strict_json_object(path)
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "run_contract_sha256",
        "state",
        "attempt_ordinal",
        "retry_count",
        "top_up_count",
        "created_at_utc",
    }
    created = marker.get("created_at_utc")
    if (
        set(marker) != expected_keys
        or marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("protocol_id") != PROTOCOL_ID
        or marker.get("status") != ATTEMPT_STATUS
        or marker.get("run_contract_sha256") != run_contract_sha256
        or marker.get("state") != dict(projection)
        or marker.get("attempt_ordinal") != 1
        or marker.get("retry_count") != 0
        or marker.get("top_up_count") != 0
        or not isinstance(created, str)
    ):
        raise ValueError("full-45 state attempt marker identity drifted")
    _duration_seconds(created, created)


def _inventory_prefix(
    *,
    layout: Full45RunLayout,
) -> tuple[int, int]:
    snapshot = _inventory_snapshot(layout=layout)
    if (
        snapshot["all_entries_are_regular_files"] is not True
        or snapshot["contiguous_prefix"] is not True
    ):
        raise ValueError("full-45 state/attempt directory inventory drifted")
    return (
        int(snapshot["state_file_count"]),
        int(snapshot["attempt_marker_file_count"]),
    )


def _inventory_snapshot(*, layout: Full45RunLayout) -> dict[str, Any]:
    state_entries = tuple(layout.state_root.iterdir())
    attempt_entries = tuple(layout.attempt_root.iterdir())
    allowed = tuple(f"{index:03d}.json" for index in range(EXPECTED_STATE_COUNT))
    state_names = {path.name for path in state_entries}
    attempt_names = {path.name for path in attempt_entries}
    all_regular = all(
        path.is_file() and not path.is_symlink()
        for path in (*state_entries, *attempt_entries)
    )
    contiguous = (
        state_names != set(allowed[: len(state_names)])
        or attempt_names != set(allowed[: len(attempt_names)])
        or not state_names <= attempt_names
        or len(attempt_names) - len(state_names) > 1
    ) is False
    return {
        "state_entries": sorted(state_names),
        "attempt_entries": sorted(attempt_names),
        "state_file_count": len(state_names),
        "attempt_marker_file_count": len(attempt_names),
        "all_entries_are_regular_files": all_regular,
        "contiguous_prefix": contiguous,
    }


def _validate_invalid_aggregate(
    aggregate: Mapping[str, Any],
    *,
    layout: Full45RunLayout,
    started_at_utc: str,
) -> None:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "run_contract_sha256",
        "status",
        "outcome",
        "gate_passed",
        "started_at_utc",
        "ended_at_utc",
        "duration_seconds",
        "invalid_failure",
        "completed_state_count",
        "attempted_state_count",
        "observed_inventory",
        "retry_count",
        "top_up_count",
        "prohibited_operation_counts",
    }
    completed = aggregate.get("completed_state_count")
    attempted = aggregate.get("attempted_state_count")
    failure = aggregate.get("invalid_failure")
    prohibited = aggregate.get("prohibited_operation_counts")
    observed_inventory = aggregate.get("observed_inventory")
    ended = aggregate.get("ended_at_utc")
    ledger = _validate_global_attempt_ledger(
        layout.ledger_path,
        attempt_identity=layout.attempt_identity,
        run_contract_sha256=layout.contract_sha256,
    )
    try:
        expected_duration = _duration_seconds(str(started_at_utc), str(ended))
    except ValueError as error:
        raise ValueError("full-45 INVALID timing drifted") from error
    if (
        set(aggregate) != expected_keys
        or aggregate.get("schema_version") != SCHEMA_VERSION
        or aggregate.get("protocol_id") != PROTOCOL_ID
        or aggregate.get("run_contract_sha256") != layout.contract_sha256
        or aggregate.get("status") != "TERMINATED_INVALID_V2_1_FULL_45_SUBSTRATE"
        or aggregate.get("outcome") != INVALID_OUTCOME
        or aggregate.get("gate_passed") is not False
        or aggregate.get("started_at_utc") != started_at_utc
        or aggregate.get("duration_seconds") != expected_duration
        or type(completed) is not int
        or type(attempted) is not int
        or not 0 <= completed <= attempted <= EXPECTED_STATE_COUNT
        or attempted - completed > 1
        or completed != ledger["completed_state_count"]
        or attempted != ledger["attempted_state_count"]
        or not isinstance(observed_inventory, Mapping)
        or dict(observed_inventory) != _inventory_snapshot(layout=layout)
        or aggregate.get("retry_count") != 0
        or aggregate.get("top_up_count") != 0
        or prohibited
        != {
            key: 0
            for key in FORBIDDEN_OPERATION_COUNTS
            if key not in {"retry_count", "top_up_count"}
        }
        or not isinstance(failure, Mapping)
        or set(failure) != {"stage", "category", "exception_type", "message"}
        or failure.get("category") not in {"OUT_OF_MEMORY", "CONTRACT_OR_RUNTIME"}
        or any(
            not isinstance(failure.get(key), str) or not failure[key]
            for key in ("stage", "exception_type", "message")
        )
    ):
        raise ValueError("full-45 INVALID aggregate identity drifted")


def _ensure_claim_root(
    layout: Full45RunLayout,
    *,
    created_at_utc: str,
) -> str:
    if layout.root.exists() and not layout.root.is_dir():
        raise ValueError("canonical full-45 root is not a directory")
    layout.root.mkdir(exist_ok=True)
    allowed = {
        RUN_MANIFEST_FILENAME,
        RUNTIME_IDENTITY_FILENAME,
        STATE_DIRECTORY,
        ATTEMPT_DIRECTORY,
        AGGREGATE_FILENAME,
    }
    if any(path.name not in allowed for path in layout.root.iterdir()):
        raise ValueError("canonical full-45 root contains an unknown entry")
    for directory in (layout.state_root, layout.attempt_root):
        if directory.exists() and not directory.is_dir():
            raise ValueError("canonical full-45 state/attempt path is not a directory")
        directory.mkdir(exist_ok=True)
    if layout.manifest_path.exists():
        manifest = _strict_json_object(layout.manifest_path)
    else:
        manifest = _run_manifest_record(layout, created_at_utc=created_at_utc)
        _write_json_exclusive(layout.manifest_path, manifest)
    return _validate_run_manifest(manifest, layout=layout)


def _persist_invalid(
    layout: Full45RunLayout,
    *,
    stage: str,
    error: BaseException,
) -> dict[str, Any]:
    ledger = _validate_global_attempt_ledger(
        layout.ledger_path,
        attempt_identity=layout.attempt_identity,
        run_contract_sha256=layout.contract_sha256,
    )
    started = _ensure_claim_root(
        layout,
        created_at_utc=str(ledger["created_at_utc"]),
    )
    observed_inventory = _inventory_snapshot(layout=layout)
    if layout.aggregate_path.exists():
        raise FileExistsError("full-45 attempt already has terminal aggregate evidence")
    invalid = _invalid_aggregate(
        run_contract_sha256=layout.contract_sha256,
        stage=stage,
        error=error,
        completed_state_count=int(ledger["completed_state_count"]),
        attempted_state_count=int(ledger["attempted_state_count"]),
        observed_inventory=observed_inventory,
        started_at_utc=started,
    )
    _write_json_exclusive(layout.aggregate_path, invalid)
    _validate_invalid_aggregate(invalid, layout=layout, started_at_utc=started)
    return invalid


def _load_existing_records(
    layout: Full45RunLayout,
    *,
    completed: int,
    runtime_metadata: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(completed):
        projection = layout.projections[index]
        _validate_attempt_marker(
            layout.attempt_root / f"{index:03d}.json",
            projection=projection,
            run_contract_sha256=layout.contract_sha256,
        )
        record = _strict_json_object(layout.state_root / f"{index:03d}.json")
        _validate_terminal_state_record(
            record,
            projection=projection,
            run_contract_sha256=layout.contract_sha256,
            runtime_metadata=runtime_metadata,
        )
        records.append(record)
    return records


def _validate_completed_aggregate(
    layout: Full45RunLayout,
    *,
    gate: Full45Gate,
    aggregate: Mapping[str, Any],
    started_at_utc: str,
) -> dict[str, Any]:
    ledger = _validate_global_attempt_ledger(
        layout.ledger_path,
        attempt_identity=layout.attempt_identity,
        run_contract_sha256=layout.contract_sha256,
    )
    if (
        ledger["completed_state_count"] != EXPECTED_STATE_COUNT
        or ledger["attempted_state_count"] != EXPECTED_STATE_COUNT
    ):
        raise ValueError("completed aggregate differs from durable ledger high-water")
    if not layout.runtime_identity_path.is_file():
        raise FileNotFoundError("completed full-45 run lacks runtime identity")
    runtime_metadata = _validate_runtime_identity_record(
        _strict_json_object(layout.runtime_identity_path),
        layout=layout,
    )
    completed, attempted = _inventory_prefix(layout=layout)
    if completed != EXPECTED_STATE_COUNT or attempted != EXPECTED_STATE_COUNT:
        raise ValueError("completed full-45 inventory is not the exact denominator")
    records = _load_existing_records(
        layout,
        completed=completed,
        runtime_metadata=runtime_metadata,
    )
    ended = aggregate.get("ended_at_utc")
    if not isinstance(ended, str):
        raise ValueError("completed full-45 aggregate lacks end timestamp")
    recomputed = aggregate_full45_gate(
        records,
        expected_states=layout.states,
        gate=gate,
        run_contract_sha256=layout.contract_sha256,
        started_at_utc=started_at_utc,
        ended_at_utc=ended,
        runtime_metadata=runtime_metadata,
    )
    if dict(aggregate) != recomputed:
        raise ValueError("completed full-45 aggregate differs from raw denominator")
    return recomputed


def claim_full45_attempt(
    *,
    states: Sequence[ScreeningState],
    gate: Full45Gate,
    run_contract: Mapping[str, Any],
    output_dir: str | Path,
    resume: bool,
) -> dict[str, Any] | None:
    """Durably claim a new attempt or validate a resumable terminal prefix."""
    layout = _prepare_layout(
        states=states,
        run_contract=run_contract,
        output_dir=output_dir,
    )
    if not resume:
        if layout.root.exists() or layout.ledger_path.exists():
            raise FileExistsError("canonical full-45 attempt already exists")
        if not layout.root.parent.is_dir():
            raise FileNotFoundError("canonical full-45 output parent must already exist")
        started = _utc_now()
        _write_json_exclusive(
            layout.ledger_path,
            _global_attempt_ledger(
                attempt_identity=layout.attempt_identity,
                run_contract_sha256=layout.contract_sha256,
                created_at_utc=started,
            ),
        )
        layout.root.mkdir(exist_ok=False)
        layout.state_root.mkdir()
        layout.attempt_root.mkdir()
        manifest = _run_manifest_record(layout, created_at_utc=started)
        _write_json_exclusive(layout.manifest_path, manifest)
        _validate_run_manifest(manifest, layout=layout)
        return None

    if not layout.ledger_path.is_file():
        raise FileNotFoundError("resume requires the canonical full-45 ledger")
    ledger = _validate_global_attempt_ledger(
        layout.ledger_path,
        attempt_identity=layout.attempt_identity,
        run_contract_sha256=layout.contract_sha256,
    )
    root_was_missing = not layout.root.exists()
    manifest_was_missing = not layout.manifest_path.is_file()
    state_root_was_missing = not layout.state_root.is_dir()
    attempt_root_was_missing = not layout.attempt_root.is_dir()
    started = _ensure_claim_root(
        layout,
        created_at_utc=str(ledger["created_at_utc"]),
    )
    if (
        root_was_missing
        or manifest_was_missing
        or state_root_was_missing
        or attempt_root_was_missing
    ):
        missing = [
            name
            for name, missing_value in (
                ("root", root_was_missing),
                ("run_manifest", manifest_was_missing),
                ("states", state_root_was_missing),
                ("attempts", attempt_root_was_missing),
            )
            if missing_value
        ]
        return _persist_invalid(
            layout,
            stage="resume_detected_deleted_durable_claim_component",
            error=RuntimeError(
                "canonical durable claim component was deleted: " + ",".join(missing)
            ),
        )
    if layout.aggregate_path.is_file():
        aggregate = _strict_json_object(layout.aggregate_path)
        if aggregate.get("outcome") == INVALID_OUTCOME:
            _validate_invalid_aggregate(aggregate, layout=layout, started_at_utc=started)
            return dict(aggregate)
        if aggregate.get("outcome") in {PASS_OUTCOME, NO_GO_OUTCOME}:
            return _validate_completed_aggregate(
                layout,
                gate=gate,
                aggregate=aggregate,
                started_at_utc=started,
            )
        raise ValueError("full-45 aggregate has an unknown outcome")
    try:
        completed, attempted = _inventory_prefix(layout=layout)
    except Exception as error:
        return _persist_invalid(
            layout,
            stage="resume_detected_invalid_directory_inventory",
            error=error,
        )
    if (
        completed != ledger["completed_state_count"]
        or attempted != ledger["attempted_state_count"]
    ):
        return _persist_invalid(
            layout,
            stage="resume_detected_ledger_inventory_high_water_mismatch",
            error=RuntimeError(
                "directory inventory differs from durable attempted/completed high-water"
            ),
        )
    if completed == EXPECTED_STATE_COUNT and attempted == EXPECTED_STATE_COUNT:
        return _persist_invalid(
            layout,
            stage="resume_detected_deleted_terminal_aggregate",
            error=RuntimeError(
                "durable 45/45 high-water exists without terminal aggregate evidence"
            ),
        )
    if attempted > completed:
        _validate_attempt_marker(
            layout.attempt_root / f"{completed:03d}.json",
            projection=layout.projections[completed],
            run_contract_sha256=layout.contract_sha256,
        )
        return _persist_invalid(
            layout,
            stage=f"resume_detected_incomplete_state_{completed:03d}",
            error=RuntimeError("attempt marker exists without terminal state evidence"),
        )
    if completed and not layout.runtime_identity_path.is_file():
        return _persist_invalid(
            layout,
            stage="resume_detected_missing_runtime_identity",
            error=RuntimeError("terminal states exist without runtime identity evidence"),
        )
    runtime_metadata = (
        _validate_runtime_identity_record(
            _strict_json_object(layout.runtime_identity_path),
            layout=layout,
        )
        if layout.runtime_identity_path.is_file()
        else None
    )
    _load_existing_records(
        layout,
        completed=completed,
        runtime_metadata=runtime_metadata,
    )
    return None


def run_full45_substrate(
    *,
    artifact: ValidatedScreeningArtifact,
    states: Sequence[ScreeningState],
    runtime: Any,
    parse_error_class: type[BaseException],
    distance_backend: Any,
    image_decoder: Callable[[bytes], Any],
    gate: Full45Gate,
    run_contract: Mapping[str, Any],
    output_dir: str | Path,
    resume: bool,
    preclaimed: bool = False,
    integrity_guard: Callable[[], None],
    message_builder: Callable[
        [ValidatedScreeningArtifact, ScreeningState, str, Callable[[bytes], Any]],
        list[dict[str, Any]],
    ] = build_full45_messages,
) -> dict[str, Any]:
    """Run all unattempted states while preserving a terminal prefix on resume."""
    if type(resume) is not bool or type(preclaimed) is not bool:
        raise TypeError("resume and preclaimed must be bool")
    if not callable(integrity_guard):
        raise TypeError("integrity_guard must be callable")
    layout = _prepare_layout(
        states=states,
        run_contract=run_contract,
        output_dir=output_dir,
    )
    if not preclaimed:
        terminal = claim_full45_attempt(
            states=states,
            gate=gate,
            run_contract=run_contract,
            output_dir=output_dir,
            resume=resume,
        )
        if terminal is not None:
            return terminal
    ledger = _validate_global_attempt_ledger(
        layout.ledger_path,
        attempt_identity=layout.attempt_identity,
        run_contract_sha256=layout.contract_sha256,
    )
    _validate_run_manifest(_strict_json_object(layout.manifest_path), layout=layout)
    if not layout.runtime_identity_path.is_file():
        raise FileNotFoundError("claimed full-45 run lacks runtime identity")
    runtime_metadata = _validate_runtime_identity_record(
        _strict_json_object(layout.runtime_identity_path),
        layout=layout,
    )
    completed, attempted = _inventory_prefix(layout=layout)
    if (
        completed != ledger["completed_state_count"]
        or attempted != ledger["attempted_state_count"]
    ):
        return _persist_invalid(
            layout,
            stage="runner_detected_ledger_inventory_high_water_mismatch",
            error=RuntimeError(
                "directory inventory differs from durable attempted/completed high-water"
            ),
        )
    if attempted != completed:
        return _persist_invalid(
            layout,
            stage=f"runner_detected_incomplete_state_{completed:03d}",
            error=RuntimeError("attempt marker exists without terminal state evidence"),
        )
    if layout.aggregate_path.exists():
        raise ValueError("active full-45 runner encountered terminal aggregate")
    records = _load_existing_records(
        layout,
        completed=completed,
        runtime_metadata=runtime_metadata,
    )
    for index in range(completed, EXPECTED_STATE_COUNT):
        state = layout.states[index]
        projection = layout.projections[index]
        state_path = layout.state_root / f"{index:03d}.json"
        attempt_path = layout.attempt_root / f"{index:03d}.json"
        try:
            integrity_guard()
            _validate_runtime_identity_record(
                _strict_json_object(layout.runtime_identity_path),
                layout=layout,
            )
        except Exception as error:
            return _persist_invalid(
                layout,
                stage=f"pre_attempt_integrity_{index:03d}",
                error=error,
            )
        try:
            _advance_ledger_high_water(
                layout,
                event="STATE_ATTEMPT_CLAIMED",
                state_index=index,
            )
            _write_json_exclusive(
                attempt_path,
                _attempt_marker(
                    projection=projection,
                    run_contract_sha256=layout.contract_sha256,
                ),
            )
        except Exception as error:
            return _persist_invalid(
                layout,
                stage=f"durable_attempt_claim_{index:03d}",
                error=error,
            )
        try:
            integrity_guard()
            _validate_runtime_identity_record(
                _strict_json_object(layout.runtime_identity_path),
                layout=layout,
            )
        except Exception as error:
            return _persist_invalid(
                layout,
                stage=f"pre_generation_integrity_{index:03d}",
                error=error,
            )
        try:
            record = run_full45_state_once(
                artifact=artifact,
                state=state,
                full45_index=index,
                runtime=runtime,
                parse_error_class=parse_error_class,
                distance_backend=distance_backend,
                image_decoder=image_decoder,
                run_contract_sha256=layout.contract_sha256,
                message_builder=message_builder,
            )
            _validate_terminal_state_record(
                record,
                projection=projection,
                run_contract_sha256=layout.contract_sha256,
                runtime_metadata=runtime_metadata,
            )
            _write_json_exclusive(state_path, record)
            _advance_ledger_high_water(
                layout,
                event="STATE_TERMINAL_PERSISTED",
                state_index=index,
            )
        except parse_error_class as error:
            return _persist_invalid(
                layout,
                stage=f"uncaptured_parse_error_{index:03d}",
                error=error,
            )
        except Exception as error:
            return _persist_invalid(
                layout,
                stage=f"state_{index:03d}",
                error=error,
            )
        records.append(record)
    try:
        integrity_guard()
        _validate_runtime_identity_record(
            _strict_json_object(layout.runtime_identity_path),
            layout=layout,
        )
    except Exception as error:
        return _persist_invalid(
            layout,
            stage="post_state_integrity",
            error=error,
        )
    ended = _utc_now()
    try:
        aggregate = aggregate_full45_gate(
            records,
            expected_states=layout.states,
            gate=gate,
            run_contract_sha256=layout.contract_sha256,
            started_at_utc=str(ledger["created_at_utc"]),
            ended_at_utc=ended,
            runtime_metadata=runtime_metadata,
        )
    except Exception as error:
        return _persist_invalid(
            layout,
            stage="aggregate_recomputation",
            error=error,
        )
    if layout.aggregate_path.exists():
        raise ValueError("terminal aggregate appeared during full-45 execution")
    _write_json_exclusive(layout.aggregate_path, aggregate)
    return aggregate


def _scientific_execution_argv(execution_argv: Sequence[str]) -> list[str]:
    values = list(execution_argv)
    if values.count("--resume") > 1:
        raise ValueError("full-45 execution argv contains duplicate --resume flags")
    return [argument for argument in values if argument != "--resume"]


def build_production_run_contract(
    args: argparse.Namespace,
    authorized: AuthorizedFull45,
) -> dict[str, Any]:
    """Bind every source, parent authorization, input, and operation limit."""
    execution_argv = getattr(args, "execution_argv", None)
    if (
        isinstance(execution_argv, (str, bytes, bytearray, Mapping))
        or not isinstance(execution_argv, Sequence)
        or not execution_argv
        or any(not isinstance(value, str) or not value for value in execution_argv)
    ):
        raise ValueError("full-45 production run must preserve its complete argv")
    scientific_argv = _scientific_execution_argv(execution_argv)
    projections = [
        _full45_projection(state, index)
        for index, state in enumerate(authorized.states)
    ]
    policy = dict(authorized.policy_identity)
    required_policy_keys = {
        "repo",
        "revision",
        "snapshot_manifest_path",
        "snapshot_manifest_sha256",
        "policy_protocol_id",
    }
    if set(policy) != required_policy_keys:
        raise ValueError("authorized full-45 policy identity drifted")
    contract_data = authorized.contract.data
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_sha256": authorized.contract.source_sha256,
        "git_identity": dict(authorized.git_identity),
        "source_inventory": list(authorized.source_inventory),
        "parent_authorization": {
            "pilot_artifact_validation": dict(authorized.pilot_authorization),
            "processor_audit": dict(authorized.processor_audit),
        },
        "canonical_inputs": dict(authorized.canonical_inputs),
        "artifact": {
            "artifact_tree_sha256": authorized.artifact.artifact_tree_sha256,
            "artifact_manifest_sha256": authorized.artifact.artifact_manifest_sha256,
            "screening_manifest_sha256": authorized.artifact.screening_manifest_sha256,
            "derived_repo": dict(contract_data["data"]["derived_artifact"]),
        },
        "policy": {
            "repo": policy["repo"],
            "revision": policy["revision"],
            "model_dir": str(Path(args.model_dir).resolve()),
            "runtime_metadata_requirement": {
                "required_policy_protocol_id": policy["policy_protocol_id"],
                "validated_after_claim_before_generation": True,
                "native_generation_metadata_persisted_twice_per_state": True,
                "teacher_metadata_persisted_three_times_per_eligible_state": True,
                "required_metadata_keys": sorted(RUNTIME_METADATA_KEYS),
                "model_dir": str(Path(args.model_dir).resolve()),
                "model_repo": policy["repo"],
                "model_revision": policy["revision"],
                "snapshot_manifest_sha256": policy["snapshot_manifest_sha256"],
                "device": args.device,
                "target_effective_visual_tokens_per_image": (
                    authorized.target_effective_visual_tokens_per_image
                ),
            },
        },
        "runtime_identity": dict(authorized.runtime_identity),
        "execution_argv": scientific_argv,
        "operational_argv_policy": {
            "resume_flag_excluded_from_scientific_run_identity": True,
            "initial_invocation_is_recorded_without_resume": True,
            "resume_only_skips_terminal_prefix": True,
        },
        "seed_policy": {
            "decoding": "greedy_do_sample_false",
            "random_seed": None,
            "sampling_seed_not_applicable": True,
        },
        "output_dir": str(Path(args.output_dir).resolve()),
        "attempt_identity": dict(authorized.attempt_identity),
        "states": projections,
        "computation_schedule": dict(contract_data["computation_schedule"]),
        "prohibited_operation_counts": dict(FORBIDDEN_OPERATION_COUNTS),
        "confirm_state": "LOCKED_NO_PROMPT_IMAGE_DECODER_OR_POLICY_ACCESS",
        "promotion": dict(contract_data["promotion"]),
    }


def _validate_absolute_production_paths(args: argparse.Namespace) -> None:
    for field in (
        "repository_root",
        "contract",
        "pilot_evidence",
        "processor_preflight",
        "derived_artifact_root",
        "scientific_config",
        "selection_manifest",
        "ocr_backend_config",
        "model_dir",
        "output_dir",
    ):
        value = getattr(args, field, None)
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError(
                f"production --{field.replace('_', '-')} must be an explicit absolute path"
            )


def validate_canonical_attempt_identity(
    contract: RestorationV21Full45Contract,
    output_dir: str | Path,
) -> dict[str, Any]:
    execution = contract.data.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("full-45 contract lacks execution identity")
    expected = {
        "attempt_id": CANONICAL_ATTEMPT_ID,
        "canonical_persistent_output_dir": str(CANONICAL_OUTPUT_DIR),
        "canonical_global_attempt_ledger": str(CANONICAL_LEDGER_PATH),
        "canonical_host_alias": CANONICAL_PILOT_HOST_ALIAS,
        "canonical_host_hostname": CANONICAL_PILOT_HOST_HOSTNAME,
        "canonical_container_id": CANONICAL_PILOT_CONTAINER_ID,
        "canonical_container_image_digest": CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
        "canonical_device": CANONICAL_PILOT_DEVICE,
        "cross_host_attempt_allowed": False,
        "alternate_output_or_ledger_allowed": False,
        "output_or_ledger_deletion_after_first_attempt_allowed": False,
    }
    for key, value in expected.items():
        if execution.get(key) != value:
            raise ValueError(f"full-45 canonical execution {key} drifted")
    if (
        execution.get("state_attempt_marker_written_before_first_generation")
        is not True
        or execution.get("runtime_import_requires_fresh_parent_evidence_validation")
        is not True
        or Path(output_dir).resolve() != CANONICAL_OUTPUT_DIR
    ):
        raise ValueError("alternate full-45 output, ledger, or lifecycle is forbidden")
    return expected


def _load_pilot_artifact_validator() -> Callable[..., Mapping[str, Any]]:
    from causalcache.restoration_v2_1_pilot_artifact import (
        validate_committed_pilot_artifact,
    )

    return validate_committed_pilot_artifact


def _load_processor_audit_validator() -> Callable[..., Mapping[str, Any]]:
    from causalcache.restoration_v2_1_processor_audit import (
        validate_restoration_v2_1_processor_audit,
    )

    return validate_restoration_v2_1_processor_audit


def _load_runtime_bindings() -> RuntimeBindings:
    from causalcache.policy.gui_owl_v2_1_runtime import (
        GUIOwlV21GenerationParseError,
        GUIOwlV21OfficialToolsRuntime,
    )
    from causalcache.restoration_v2_gpu_kl import (
        gpu_resident_full_vocab_mean_kl,
    )

    return RuntimeBindings(
        runtime_class=GUIOwlV21OfficialToolsRuntime,
        parse_error_class=GUIOwlV21GenerationParseError,
        kl_kernel=gpu_resident_full_vocab_mean_kl,
    )


def _external_file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"external full-45 authorization input is missing: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def authorize_production_full45(
    args: argparse.Namespace,
    *,
    git_validator: Callable[[str | Path], Mapping[str, Any]] = validate_clean_pushed_main,
    source_validator: Callable[..., Sequence[Mapping[str, str]]] = (
        validate_committed_source_blobs
    ),
    pilot_validator_loader: Callable[[], Callable[..., Mapping[str, Any]]] = (
        _load_pilot_artifact_validator
    ),
    processor_validator_loader: Callable[[], Callable[..., Mapping[str, Any]]] = (
        _load_processor_audit_validator
    ),
    artifact_loader: Callable[..., ValidatedScreeningArtifact] = (
        load_validated_screening_artifact
    ),
    runtime_identity_loader: Callable[..., Mapping[str, Any]] = (
        _load_live_runtime_identity
    ),
) -> AuthorizedFull45:
    """Validate CPU/data/parent evidence before importing the policy runtime."""
    _validate_absolute_production_paths(args)
    root = Path(args.repository_root).resolve()
    contract = RestorationV21Full45Contract.load(
        args.contract,
        repository_root=root,
    )
    attempt_identity = validate_canonical_attempt_identity(contract, args.output_dir)
    if CANONICAL_OUTPUT_DIR == root or root in CANONICAL_OUTPUT_DIR.parents:
        raise ValueError("raw full-45 output must remain outside the Git worktree")
    git_identity = dict(git_validator(root))
    git_commit = git_identity.get("commit")
    if not isinstance(git_commit, str) or GIT_SHA_PATTERN.fullmatch(git_commit) is None:
        raise ValueError("full-45 Git authorization lacks a full commit SHA")

    pilot_path = Path(args.pilot_evidence).resolve()
    processor_path = Path(args.processor_preflight).resolve()
    if any(path == root or root in path.parents for path in (pilot_path, processor_path)):
        raise ValueError("fresh parent raw evidence must remain outside the Git worktree")
    pilot_external = _external_file_record(pilot_path)
    processor_external = _external_file_record(processor_path)
    parent = contract.data["parent_authorization"]
    pilot_binding = parent["fixed_15_pass_manifest"]
    processor_binding = parent["processor_pass_manifest"]
    if (
        pilot_external["sha256"] != pilot_binding["raw_archive_sha256"]
        or pilot_external["size_bytes"] != pilot_binding["raw_archive_size_bytes"]
        or processor_external["sha256"] != processor_binding["raw_evidence_sha256"]
        or processor_external["size_bytes"] != processor_binding["raw_evidence_size_bytes"]
    ):
        raise PermissionError("fresh parent evidence byte identity drifted")

    pilot_validation = pilot_validator_loader()(
        repository_root=root,
        current_git_commit=git_commit,
        evidence_path=pilot_path,
    )
    if not isinstance(pilot_validation, Mapping) or (
        pilot_validation.get("status")
        != "VALID_RESTORATION_V2_1_INTERFACE_PILOT_ARTIFACT"
        or pilot_validation.get("outcome") != "PASS_V2_1_INTERFACE_PILOT"
        or pilot_validation.get("archive_hash_verified") is not True
        or pilot_validation.get("current_git_commit") != git_commit
        or pilot_validation.get("evidence_source_kind") != "raw_archive"
    ):
        raise PermissionError("fresh fixed-15 raw evidence did not authorize full-45")
    processor_validation = processor_validator_loader()(
        _strict_json_object(processor_path),
        repository_root=root,
        current_git_commit=git_commit,
        mode="reuse",
        evidence_path=processor_path,
    )
    if not isinstance(processor_validation, Mapping) or (
        processor_validation.get("status")
        != processor_binding["required_status"]
        or processor_validation.get("prompt_count")
        != processor_binding["required_prompt_count"]
        or processor_validation.get("validation_mode") != "reuse"
        or processor_validation.get("current_git_commit") != git_commit
    ):
        raise PermissionError("fresh processor evidence did not authorize full-45")

    scientific_path = _canonical_repository_path(
        args.scientific_config,
        repository_root=root,
        relative_path=CANONICAL_SCIENTIFIC_CONFIG_PATH,
        name="parent scientific config",
    )
    selection_path = _canonical_repository_path(
        args.selection_manifest,
        repository_root=root,
        relative_path=CANONICAL_SELECTION_MANIFEST_PATH,
        name="selection manifest",
    )
    ocr_path = _canonical_repository_path(
        args.ocr_backend_config,
        repository_root=root,
        relative_path=CANONICAL_OCR_BACKEND_CONFIG_PATH,
        name="OCR backend config",
    )
    pilot_contract_record = parent["pilot_contract"]
    pilot_contract_path = _canonical_repository_path(
        root / pilot_contract_record["path"],
        repository_root=root,
        relative_path=pilot_contract_record["path"],
        name="pilot contract",
    )
    pilot_contract = RestorationV21PilotContract.load(
        pilot_contract_path,
        repository_root=root,
    )
    policy = pilot_contract.data["primary_policy"]
    snapshot_record = policy["snapshot_manifest"]
    snapshot_path = _canonical_repository_path(
        root / snapshot_record["path"],
        repository_root=root,
        relative_path=snapshot_record["path"],
        name="policy snapshot manifest",
    )
    if sha256_file(snapshot_path) != snapshot_record["sha256"]:
        raise ValueError("policy snapshot manifest SHA256 drifted")
    if sha256_file(scientific_path) != parent["parent_v2_scientific_contract"]["sha256"]:
        raise ValueError("parent scientific config SHA256 drifted")
    if sha256_file(selection_path) != contract.data["data"]["selection_manifest"]["sha256"]:
        raise ValueError("selection manifest SHA256 drifted")

    source_paths = tuple(
        str(value)
        for value in contract.data["source_lock"]["formal_run_source_inventory_paths"]
    )
    source_inventory = tuple(
        source_validator(
            repository_root=root,
            git_commit=git_commit,
            paths=source_paths,
        )
    )
    artifact = artifact_loader(
        artifact_root=args.derived_artifact_root,
        backend_config_path=ocr_path,
        scientific_config_path=scientific_path,
        selection_manifest_path=selection_path,
        expected_artifact_tree_sha256=contract.data["data"]["derived_artifact"][
            "artifact_tree_sha256"
        ],
    )
    states = select_exact_full45_states(artifact, contract)
    scientific = _strict_json_object(scientific_path)
    primary = scientific.get("primary_policy")
    visual = (
        primary.get("visual_preprocessing") if isinstance(primary, Mapping) else None
    )
    target_tokens = (
        visual.get("target_effective_tokens_per_image")
        if isinstance(visual, Mapping)
        else None
    )
    if type(target_tokens) is not int or target_tokens <= 0:
        raise ValueError("parent scientific visual token target is invalid")
    observed_runtime = runtime_identity_loader(device=args.device)
    runtime_identity = _validate_runtime_cli_identity(
        args,
        observed=observed_runtime,
        attempt_identity={
            **attempt_identity,
            "cross_host_attempt_allowed": False,
        },
    )
    pilot_manifest_path = root / CANONICAL_PILOT_ARTIFACT_PATH
    processor_manifest_path = root / CANONICAL_PROCESSOR_PREFLIGHT_ARTIFACT_PATH
    canonical_inputs = {
        "contract": {"path": CANONICAL_CONFIG_PATH, "sha256": contract.source_sha256},
        "pilot_contract": {
            "path": pilot_contract_record["path"],
            "sha256": pilot_contract_record["sha256"],
        },
        "pilot_evidence": {
            "external": pilot_external,
            "artifact_manifest_path": CANONICAL_PILOT_ARTIFACT_PATH,
            "artifact_manifest_sha256": sha256_file(pilot_manifest_path),
        },
        "processor_preflight": {
            "external": processor_external,
            "artifact_manifest_path": CANONICAL_PROCESSOR_PREFLIGHT_ARTIFACT_PATH,
            "artifact_manifest_sha256": sha256_file(processor_manifest_path),
        },
        "scientific_config": {
            "path": CANONICAL_SCIENTIFIC_CONFIG_PATH,
            "sha256": sha256_file(scientific_path),
        },
        "selection_manifest": dict(contract.data["data"]["selection_manifest"]),
        "ocr_backend_config": {
            "path": CANONICAL_OCR_BACKEND_CONFIG_PATH,
            "sha256": sha256_file(ocr_path),
        },
        "snapshot_manifest": dict(snapshot_record),
    }
    policy_identity = {
        "repo": policy["repo"],
        "revision": policy["revision"],
        "snapshot_manifest_path": snapshot_record["path"],
        "snapshot_manifest_sha256": snapshot_record["sha256"],
        "policy_protocol_id": GUI_OWL_V2_1_PROTOCOL_ID,
    }
    return AuthorizedFull45(
        repository_root=root,
        contract=contract,
        processor_audit=dict(processor_validation),
        pilot_authorization=dict(pilot_validation),
        git_identity=git_identity,
        source_inventory=source_inventory,
        artifact=artifact,
        states=states,
        snapshot_manifest_path=snapshot_path,
        target_effective_visual_tokens_per_image=target_tokens,
        policy_identity=policy_identity,
        runtime_identity=runtime_identity,
        canonical_inputs=canonical_inputs,
        attempt_identity=attempt_identity,
        external_integrity={
            "pilot_evidence": pilot_external,
            "processor_preflight": processor_external,
        },
    )


def revalidate_authorized_integrity(authorized: AuthorizedFull45) -> None:
    current_git = validate_clean_pushed_main(authorized.repository_root)
    if current_git != dict(authorized.git_identity):
        raise ValueError("authorized Git identity changed during full-45 attempt")
    paths = tuple(str(record["path"]) for record in authorized.source_inventory)
    current_sources = validate_committed_source_blobs(
        repository_root=authorized.repository_root,
        git_commit=str(current_git["commit"]),
        paths=paths,
    )
    if current_sources != list(authorized.source_inventory):
        raise ValueError("authorized source inventory changed during full-45 attempt")
    for name, expected in authorized.external_integrity.items():
        path = Path(str(expected["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != expected["size_bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise ValueError(f"authorized external {name} changed during full-45 attempt")


def execute_production_full45(
    args: argparse.Namespace,
    *,
    authorization_loader: Callable[[argparse.Namespace], AuthorizedFull45] = (
        authorize_production_full45
    ),
    integrity_revalidator: Callable[[AuthorizedFull45], None] = (
        revalidate_authorized_integrity
    ),
    runtime_bindings_loader: Callable[[], RuntimeBindings] = _load_runtime_bindings,
    run_executor: Callable[..., dict[str, Any]] = run_full45_substrate,
) -> dict[str, Any]:
    """Claim globally before policy import and model construction."""
    authorized = authorization_loader(args)
    integrity_revalidator(authorized)
    run_contract = build_production_run_contract(args, authorized)
    gate = Full45Gate.from_mapping(authorized.contract.data["substrate_gate"])
    try:
        terminal = claim_full45_attempt(
            states=authorized.states,
            gate=gate,
            run_contract=run_contract,
            output_dir=args.output_dir,
            resume=args.resume,
        )
    except Exception as error:
        layout = _prepare_layout(
            states=authorized.states,
            run_contract=run_contract,
            output_dir=args.output_dir,
        )
        if (
            layout.ledger_path.is_file()
            and not layout.aggregate_path.exists()
        ):
            return _persist_invalid(
                layout,
                stage="durable_claim_bootstrap_failure",
                error=error,
            )
        raise
    if terminal is not None:
        return terminal
    layout = _prepare_layout(
        states=authorized.states,
        run_contract=run_contract,
        output_dir=args.output_dir,
    )
    try:
        integrity_revalidator(authorized)
        bindings = runtime_bindings_loader()
        runtime = bindings.runtime_class(
            model_dir=args.model_dir,
            expected_snapshot_manifest=authorized.snapshot_manifest_path,
            device=args.device,
            target_effective_visual_tokens_per_image=(
                authorized.target_effective_visual_tokens_per_image
            ),
        )
        runtime_metadata = getattr(runtime, "metadata", None)
        if (
            not isinstance(runtime_metadata, Mapping)
            or runtime_metadata.get("protocol_id") != GUI_OWL_V2_1_PROTOCOL_ID
        ):
            raise RuntimeError("loaded policy runtime differs from official-tool v2.1")
        if layout.runtime_identity_path.exists():
            persisted_runtime_metadata = _validate_runtime_identity_record(
                _strict_json_object(layout.runtime_identity_path),
                layout=layout,
            )
            if dict(runtime_metadata) != persisted_runtime_metadata:
                raise ValueError("resumed live runtime differs from persisted identity")
        else:
            runtime_record = _runtime_identity_record(
                layout=layout,
                runtime_metadata=runtime_metadata,
            )
            _validate_runtime_identity_record(runtime_record, layout=layout)
            _write_json_exclusive(layout.runtime_identity_path, runtime_record)
        distance_backend = GPUFullVocabularyKLBackend(
            torch_module=runtime.torch,
            kl_kernel=bindings.kl_kernel,
        )
    except Exception as error:
        return _persist_invalid(
            layout,
            stage="policy_runtime_initialization_after_durable_claim",
            error=error,
        )
    try:
        result = run_executor(
            artifact=authorized.artifact,
            states=authorized.states,
            runtime=runtime,
            parse_error_class=bindings.parse_error_class,
            distance_backend=distance_backend,
            image_decoder=_decode_rgb_image,
            gate=gate,
            run_contract=run_contract,
            output_dir=args.output_dir,
            resume=args.resume,
            preclaimed=True,
            integrity_guard=lambda: integrity_revalidator(authorized),
        )
        if not isinstance(result, Mapping) or not layout.aggregate_path.is_file():
            raise RuntimeError("full-45 runner returned without terminal evidence")
        persisted = _strict_json_object(layout.aggregate_path)
        if dict(result) != persisted:
            raise ValueError("full-45 runner result differs from terminal evidence")
        return persisted
    except Exception as error:
        if layout.aggregate_path.exists():
            raise
        return _persist_invalid(
            layout,
            stage="full_45_runner_failure_after_durable_claim",
            error=error,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--pilot-evidence", required=True)
    parser.add_argument("--processor-preflight", required=True)
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
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(raw_argv)
    args.execution_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        *raw_argv,
    ]
    try:
        aggregate = execute_production_full45(args)
    except Exception as error:
        aggregate = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "INVALID_BEFORE_V2_1_FULL_45_RUN_DIRECTORY",
            "outcome": INVALID_OUTCOME,
            "invalid_failure": {
                "category": (
                    "OUT_OF_MEMORY" if _is_out_of_memory(error) else "CONTRACT_OR_RUNTIME"
                ),
                "exception_type": error.__class__.__name__,
                "message": str(error),
            },
        }
        print(json.dumps(aggregate, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps(aggregate, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if aggregate["outcome"] in {PASS_OUTCOME, NO_GO_OUTCOME} else 2


if __name__ == "__main__":
    raise SystemExit(main())
