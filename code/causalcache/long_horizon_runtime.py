"""Fail-closed GUI-Owl execution for the frozen long-horizon development study."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2_1 import (
    build_gui_owl_v2_1_mixed_fidelity_messages,
    parse_gui_owl_v2_1_output,
)
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21GenerationParseError
from causalcache.policy.gui_owl_v2_2_eager_runtime import GUIOwlV22EagerRuntime
from causalcache.policy.gui_owl_v2_runtime import FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS
from causalcache.restoration_v2_gpu_kl import gpu_resident_full_vocab_mean_kl


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_long_horizon_policy_runtime_v1"
RECEIPT_PROTOCOL_ID = "causalcache_long_horizon_terminal_receipt_v1"
LONG_HORIZON_ALLOWED_DECISION_STEPS = (10, 18)
N8_DECISION_STEP_ID = 10
N16_DECISION_STEP_ID = 18
N8_EVENT_IDS = tuple(range(1, 9))
N16_EVENT_IDS = tuple(range(1, 17))
N8_MAXIMUM_ENUMERATED_BUDGET = 4
N8_EXPECTED_ROW_COUNT = 163
DEFAULT_MAXIMUM_CONTEXT_TOKENS = 32_768
DEFAULT_COMPLETION_TOKEN_RESERVE = FROZEN_GUI_OWL_V2_MAX_NEW_TOKENS
SHA256_PATTERN_LENGTH = 64


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_PATTERN_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _event_subset(
    values: Sequence[int],
    *,
    candidates: tuple[int, ...],
    label: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{label} must be an event-id sequence")
    result = tuple(values)
    if any(type(event) is not int for event in result):
        raise TypeError(f"{label} must contain integer event ids")
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique and strictly increasing")
    if not set(result).issubset(candidates):
        raise ValueError(f"{label} contains an event outside the frozen candidate archive")
    return result


def enumerate_n8_coalitions() -> tuple[tuple[int, ...], ...]:
    """Return all and only n=8 coalitions with cardinality at most four."""
    result = tuple(
        coalition
        for size in range(N8_MAXIMUM_ENUMERATED_BUDGET + 1)
        for coalition in itertools.combinations(N8_EVENT_IDS, size)
    )
    if len(result) != N8_EXPECTED_ROW_COUNT:
        raise RuntimeError("n=8 at-most-four coalition denominator drifted")
    return result


N8_COALITIONS = enumerate_n8_coalitions()


def build_long_horizon_runtime(
    *,
    model_dir: str | Path,
    expected_snapshot_manifest: str | Path,
    device: str,
    runtime_class: type[Any] = GUIOwlV22EagerRuntime,
) -> Any:
    """Construct the frozen eager H200 runtime on one explicit worker device."""
    runtime = runtime_class(
        model_dir=model_dir,
        expected_snapshot_manifest=expected_snapshot_manifest,
        device=device,
    )
    required = (
        "prepare_native_message_shape",
        "generate_native_action",
        "teacher_forced_distance_logits",
    )
    if any(not callable(getattr(runtime, name, None)) for name in required):
        raise RuntimeError("long-horizon GUI-Owl runtime interface drifted")
    return runtime


def _decision_by_step(
    trajectory: Mapping[str, Any], decision_step_id: int
) -> Mapping[str, Any]:
    if trajectory.get("role") != "development":
        raise PermissionError("long-horizon runtime accepts development trajectories only")
    decisions = trajectory.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("long-horizon trajectory decisions must be a JSON array")
    matches = [
        decision
        for decision in decisions
        if isinstance(decision, Mapping)
        and decision.get("decision_step_id") == decision_step_id
    ]
    if len(matches) != 1:
        raise ValueError("long-horizon trajectory lacks one exact requested state")
    return matches[0]


def _validated_state(
    trajectory: Mapping[str, Any], *, decision_step_id: int
) -> tuple[Mapping[str, Any], tuple[int, ...]]:
    if not isinstance(trajectory, Mapping):
        raise TypeError("long-horizon trajectory must be a mapping")
    source_id = trajectory.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("long-horizon trajectory source_id must be non-empty text")
    decision = _decision_by_step(trajectory, decision_step_id)
    expected_candidates = (
        N8_EVENT_IDS if decision_step_id == N8_DECISION_STEP_ID else N16_EVENT_IDS
    )
    expected_history = tuple(range(1, decision_step_id))
    if (
        tuple(decision.get("candidate_event_step_ids", ())) != expected_candidates
        or tuple(decision.get("history_event_step_ids", ())) != expected_history
        or decision.get("current_equivalent_event_step_id") != decision_step_id - 1
    ):
        raise ValueError("long-horizon state geometry differs from the frozen contract")
    events = trajectory.get("events")
    if not isinstance(events, list):
        raise ValueError("long-horizon trajectory events must be a JSON array")
    event_ids = tuple(
        event.get("step_id") if isinstance(event, Mapping) else None for event in events
    )
    if event_ids != tuple(range(1, 18)):
        raise ValueError("long-horizon substrate must expose the exact 17-event prefix")
    if decision.get("current_expert_action_payload_included") is not False:
        raise PermissionError("current expert action payload must remain absent")
    return decision, expected_candidates


def _project_message_manifest(
    trajectory: Mapping[str, Any], *, decision_step_id: int
) -> dict[str, Any]:
    decision, _ = _validated_state(trajectory, decision_step_id=decision_step_id)
    history_ids = frozenset(decision["history_event_step_ids"])
    projected = dict(trajectory)
    projected["events"] = [
        dict(event) for event in trajectory["events"] if event["step_id"] in history_ids
    ]
    projected["decisions"] = [dict(decision)]
    if len(projected["events"]) != len(history_ids):
        raise ValueError("long-horizon history slice lost an archived event")
    return {"trajectories": [projected]}


def build_long_horizon_messages(
    trajectory: Mapping[str, Any],
    *,
    decision_step_id: int,
    restored_event_step_ids: Sequence[int],
    image_bytes_loader: Callable[[str], bytes],
    image_decoder: Callable[[bytes], Any],
) -> list[dict[str, Any]]:
    """Build one prompt after physically slicing shared storage to this state."""
    _, candidates = _validated_state(
        trajectory, decision_step_id=decision_step_id
    )
    restored = _event_subset(
        restored_event_step_ids,
        candidates=candidates,
        label="restored event subset",
    )
    manifest = _project_message_manifest(
        trajectory, decision_step_id=decision_step_id
    )
    return build_gui_owl_v2_1_mixed_fidelity_messages(
        manifest,
        trajectory_id=str(trajectory["source_id"]),
        decision_step_id=decision_step_id,
        restored_event_step_ids=restored,
        image_bytes_loader=image_bytes_loader,
        image_decoder=image_decoder,
        allowed_decision_steps=LONG_HORIZON_ALLOWED_DECISION_STEPS,
    )


@dataclass(frozen=True)
class DistanceMeasurement:
    value: float
    audit: Mapping[str, Any]


class GPUFullVocabularyKLBackend:
    """Use the canonical CUDA kernel and transfer only one final distance scalar."""

    def __init__(self, *, torch_module: Any) -> None:
        self.torch = torch_module

    def prepare_reference(self, logits: Any) -> Any:
        if getattr(logits, "ndim", None) != 3 or tuple(logits.shape[:1]) != (1,):
            raise ValueError("reference logits must have shape [1,tokens,vocab]")
        if getattr(getattr(logits, "device", None), "type", None) != "cuda":
            raise ValueError("reference logits must remain on CUDA")
        if str(getattr(logits, "dtype", None)) != "torch.bfloat16":
            raise TypeError("reference logits must remain BF16")
        if getattr(logits, "requires_grad", False):
            raise ValueError("reference logits must not require gradients")
        with self.torch.inference_mode():
            result = self.torch.log_softmax(logits.to(dtype=self.torch.float32), dim=-1)
        if result.device != logits.device or str(result.dtype) != "torch.float32":
            raise RuntimeError("reference log probabilities left CUDA FP32")
        return result

    def measure(self, reference_log_probs: Any, candidate_logits: Any) -> DistanceMeasurement:
        result = gpu_resident_full_vocab_mean_kl(
            reference_log_probs,
            candidate_logits,
            candidate_representation="logits",
        )
        values = result.per_example_mean_kl
        if getattr(values, "ndim", None) != 1 or tuple(values.shape) != (1,):
            raise RuntimeError("GPU KL must return one per-state distance")
        if getattr(getattr(values, "device", None), "type", None) != "cuda":
            raise RuntimeError("GPU KL distance left CUDA before final scalar transfer")
        value = float(values.detach().to(device="cpu").item())
        audit = result.audit.to_dict()
        if audit.get("full_tensor_host_transfers") != 0:
            raise RuntimeError("GPU KL audit reported a full-tensor host transfer")
        return DistanceMeasurement(value=value, audit=audit)


@dataclass(frozen=True)
class SealedComparatorPair:
    """Exactly one label-blind n=16 pair; it cannot represent a search table."""

    source_id: str
    state_id: str
    budget_event_capacity: int
    left_selector_name: str
    right_selector_name: str
    left_selected_event_step_ids: tuple[int, ...]
    right_selected_event_step_ids: tuple[int, ...]
    selection_seal_sha256: str

    def validate(self, trajectory: Mapping[str, Any]) -> None:
        decision, candidates = _validated_state(
            trajectory, decision_step_id=N16_DECISION_STEP_ID
        )
        if self.source_id != trajectory["source_id"] or self.state_id != decision["state_id"]:
            raise ValueError("sealed comparator pair state identity drifted")
        if self.budget_event_capacity not in {2, 4}:
            raise ValueError("n=16 pair budget must be 2 or 4")
        if (
            not self.left_selector_name
            or not self.right_selector_name
            or self.left_selector_name == self.right_selector_name
        ):
            raise ValueError("n=16 pair requires two distinct named comparators")
        left = _event_subset(
            self.left_selected_event_step_ids,
            candidates=candidates,
            label="left sealed selection",
        )
        right = _event_subset(
            self.right_selected_event_step_ids,
            candidates=candidates,
            label="right sealed selection",
        )
        reference = tuple(sorted(set(left) | set(right)))
        if reference == candidates:
            raise PermissionError("n=16 full-history reference is forbidden")
        if len(left) > self.budget_event_capacity or len(right) > self.budget_event_capacity:
            raise ValueError("sealed comparator selection exceeds its event budget")
        _sha256_text(self.selection_seal_sha256, "selection seal SHA256")
        if len(reference) > 2 * self.budget_event_capacity:
            raise ValueError("n=16 pair-union reference exceeds the frozen upper bound")

    @property
    def reference_event_step_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                set(self.left_selected_event_step_ids)
                | set(self.right_selected_event_step_ids)
            )
        )


def _operation_accounting(requested: Mapping[str, int]) -> dict[str, Any]:
    exact = {key: int(value) for key, value in requested.items()}
    if any(value < 0 for value in exact.values()):
        raise ValueError("requested operation counts must be non-negative")
    return {
        key: {"requested": value, "completed": 0, "missing": value}
        for key, value in exact.items()
    }


def _complete_operation(accounting: dict[str, Any], name: str) -> None:
    operation = accounting[name]
    operation["completed"] += 1
    if operation["completed"] > operation["requested"]:
        raise RuntimeError(f"completed operation count exceeds schedule: {name}")
    operation["missing"] = operation["requested"] - operation["completed"]


def _all_operations_complete(accounting: Mapping[str, Any]) -> bool:
    validate_operation_accounting(accounting)
    return all(
        operation == {
            "requested": operation["requested"],
            "completed": operation["requested"],
            "missing": 0,
        }
        for operation in accounting.values()
    )


def validate_operation_accounting(accounting: Mapping[str, Any]) -> None:
    if not isinstance(accounting, Mapping) or not accounting:
        raise ValueError("operation accounting must be a non-empty mapping")
    for name, operation in accounting.items():
        if not isinstance(name, str) or not name or not isinstance(operation, Mapping):
            raise ValueError("operation accounting names and records must be mappings")
        if set(operation) != {"requested", "completed", "missing"}:
            raise ValueError("operation accounting record field inventory drifted")
        requested = operation["requested"]
        completed = operation["completed"]
        missing = operation["missing"]
        if (
            type(requested) is not int
            or type(completed) is not int
            or type(missing) is not int
            or min(requested, completed, missing) < 0
            or completed + missing != requested
        ):
            raise ValueError("operation accounting does not conserve its request")


class PromptContextOverflow(ValueError):
    pass


class NonFiniteDistanceError(ValueError):
    pass


def _processor_preflight(
    runtime: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    maximum_context_tokens: int,
    completion_token_reserve: int,
) -> dict[str, Any]:
    if type(maximum_context_tokens) is not int or maximum_context_tokens <= 0:
        raise ValueError("maximum context tokens must be a positive integer")
    if type(completion_token_reserve) is not int or completion_token_reserve <= 0:
        raise ValueError("completion token reserve must be a positive integer")
    metadata = runtime.prepare_native_message_shape(messages)
    if not isinstance(metadata, Mapping):
        raise RuntimeError("processor preflight metadata must be a mapping")
    prompt_tokens = metadata.get("prompt_input_tokens")
    sequence_length = metadata.get("sequence_length")
    if (
        type(prompt_tokens) is not int
        or type(sequence_length) is not int
        or prompt_tokens != sequence_length
        or prompt_tokens <= 0
    ):
        raise RuntimeError("processor preflight lost its exact prompt token count")
    if metadata.get("policy_forward_executed") is not False:
        raise RuntimeError("processor preflight unexpectedly executed the policy")
    if prompt_tokens + completion_token_reserve > maximum_context_tokens:
        raise PromptContextOverflow(
            "processor prompt plus completion reserve exceeds context: "
            f"{prompt_tokens}+{completion_token_reserve}>{maximum_context_tokens}"
        )
    return {
        **dict(metadata),
        "completion_token_reserve": completion_token_reserve,
        "maximum_context_tokens": maximum_context_tokens,
    }


def _action_arguments(action: Any) -> dict[str, Any]:
    arguments = action.arguments()
    if not isinstance(arguments, Mapping):
        raise RuntimeError("canonical action arguments must be a mapping")
    return dict(arguments)


def _generation_evidence(
    runtime: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    parse_error_class: type[BaseException],
) -> tuple[dict[str, Any], Any | None]:
    try:
        generated = runtime.generate_native_action(messages)
    except parse_error_class as error:
        output = getattr(error, "output_text", None)
        metadata = getattr(error, "metadata", None)
        if not isinstance(output, str) or not isinstance(metadata, Mapping):
            raise RuntimeError("parse error lost native generation evidence") from error
        return (
            {
                "output_text": output,
                "metadata": dict(metadata),
                "canonical_action": None,
                "parse_error_type": str(
                    getattr(error, "parse_error_type", error.__class__.__name__)
                ),
                "parse_error_message": str(
                    getattr(error, "parse_error_message", str(error))
                ),
            },
            None,
        )
    output = getattr(generated, "output_text", None)
    metadata = getattr(generated, "metadata", None)
    if not isinstance(output, str) or not isinstance(metadata, Mapping):
        raise RuntimeError("generation result lost native output or metadata")
    try:
        independently_parsed = parse_gui_owl_v2_1_output(output)
    except (TypeError, ValueError) as error:
        return (
            {
                "output_text": output,
                "metadata": dict(metadata),
                "canonical_action": None,
                "parse_error_type": error.__class__.__name__,
                "parse_error_message": str(error),
            },
            None,
        )
    runtime_parsed = getattr(getattr(generated, "parsed_output", None), "canonical_action", None)
    if runtime_parsed != independently_parsed.canonical_action:
        raise RuntimeError("runtime parser and independent parser actions differ")
    arguments = _action_arguments(independently_parsed.canonical_action)
    return (
        {
            "output_text": output,
            "metadata": dict(metadata),
            "canonical_action": arguments,
            "parse_error_type": None,
            "parse_error_message": None,
        },
        independently_parsed.canonical_action,
    )


def _base_record(
    *,
    kind: str,
    trajectory: Mapping[str, Any],
    state_id: str,
    requested: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "kind": kind,
        "source_id": trajectory["source_id"],
        "state_id": state_id,
        "outcome": "INVALID_LONG_HORIZON_STATE",
        "valid": False,
        "failure": None,
        "native_generations": [],
        "canonical_action": None,
        "processor_preflights": [],
        "teacher_forwards": [],
        "distance_audits": [],
        "operation_counts": _operation_accounting(requested),
        "retry_count": 0,
        "top_up_count": 0,
        "reserve_access_count": 0,
    }


def _failure(record: dict[str, Any], *, category: str, stage: str, message: str) -> dict[str, Any]:
    record["failure"] = {"category": category, "stage": stage, "message": message}
    return record


def _preflight_prompts(
    *,
    record: dict[str, Any],
    trajectory: Mapping[str, Any],
    decision_step_id: int,
    prompt_subsets: Sequence[tuple[int, ...]],
    runtime: Any,
    image_bytes_loader: Callable[[str], bytes],
    image_decoder: Callable[[bytes], Any],
    maximum_context_tokens: int,
    completion_token_reserve: int,
) -> bool:
    for subset in prompt_subsets:
        try:
            messages = build_long_horizon_messages(
                trajectory,
                decision_step_id=decision_step_id,
                restored_event_step_ids=subset,
                image_bytes_loader=image_bytes_loader,
                image_decoder=image_decoder,
            )
            metadata = _processor_preflight(
                runtime,
                messages,
                maximum_context_tokens=maximum_context_tokens,
                completion_token_reserve=completion_token_reserve,
            )
        except PromptContextOverflow as error:
            _failure(
                record,
                category="CONTEXT_OVERFLOW",
                stage="processor_preflight",
                message=str(error),
            )
            return False
        record["processor_preflights"].append(
            {"restored_event_step_ids": list(subset), "metadata": metadata}
        )
        _complete_operation(record["operation_counts"], "processor_preflights")
    return True


def _run_reference_generations(
    *,
    record: dict[str, Any],
    runtime: Any,
    reference_messages: Sequence[Mapping[str, Any]],
    parse_error_class: type[BaseException],
) -> Any | None:
    parsed_actions = []
    for repeat_index in (1, 2):
        evidence, action = _generation_evidence(
            runtime,
            reference_messages,
            parse_error_class=parse_error_class,
        )
        evidence["repeat_index"] = repeat_index
        record["native_generations"].append(evidence)
        _complete_operation(
            record["operation_counts"], "canonical_action_parse_attempts"
        )
        if action is not None:
            parsed_actions.append(action)
    if len(parsed_actions) != 2:
        _failure(
            record,
            category="PARSE_FAILURE",
            stage="reference_generations",
            message="one or both strict reference generations failed to parse",
        )
        return None
    first_arguments = _action_arguments(parsed_actions[0])
    second_arguments = _action_arguments(parsed_actions[1])
    if first_arguments != second_arguments:
        _failure(
            record,
            category="CANONICAL_ACTION_MISMATCH",
            stage="reference_generation_repeat_comparison",
            message="two reference generations produced different canonical actions",
        )
        return None
    record["canonical_action"] = first_arguments
    return parsed_actions[0]


def _measure_distance(
    *,
    record: dict[str, Any],
    runtime: Any,
    distance_backend: Any,
    reference_log_probs: Any,
    messages: Sequence[Mapping[str, Any]],
    canonical_action: Any,
    restored_event_step_ids: tuple[int, ...],
) -> float:
    logits, metadata = runtime.teacher_forced_distance_logits(
        (messages,), (canonical_action,)
    )
    _complete_operation(record["operation_counts"], "candidate_teacher_forwards")
    measurement = distance_backend.measure(reference_log_probs, logits)
    del logits
    _complete_operation(record["operation_counts"], "distance_rows")
    value = float(measurement.value)
    if not math.isfinite(value) or value < 0.0:
        raise NonFiniteDistanceError(
            "GPU full-vocabulary KL returned a non-finite distance"
        )
    record["teacher_forwards"].append(
        {
            "role": "candidate",
            "restored_event_step_ids": list(restored_event_step_ids),
            "metadata": dict(metadata),
        }
    )
    record["distance_audits"].append(
        {
            "restored_event_step_ids": list(restored_event_step_ids),
            "audit": dict(measurement.audit),
        }
    )
    return value


def run_n8_state_once(
    *,
    trajectory: Mapping[str, Any],
    runtime: Any,
    image_bytes_loader: Callable[[str], bytes],
    image_decoder: Callable[[bytes], Any],
    distance_backend: Any | None = None,
    maximum_context_tokens: int = DEFAULT_MAXIMUM_CONTEXT_TOKENS,
    completion_token_reserve: int = DEFAULT_COMPLETION_TOKEN_RESERVE,
    parse_error_class: type[BaseException] = GUIOwlV21GenerationParseError,
) -> dict[str, Any]:
    """Run the complete n=8 exact-at-most-four distance table exactly once."""
    decision, candidates = _validated_state(
        trajectory, decision_step_id=N8_DECISION_STEP_ID
    )
    if candidates != N8_EVENT_IDS:
        raise RuntimeError("n=8 runtime candidate geometry drifted")
    requested = {
        "processor_preflights": 1 + len(N8_COALITIONS),
        "canonical_action_parse_attempts": 2,
        "reference_teacher_forwards": 1,
        "candidate_teacher_forwards": len(N8_COALITIONS),
        "distance_rows": len(N8_COALITIONS),
        "analytic_self_distance_rows": 1,
    }
    record = _base_record(
        kind="n8_exact_at_most_four",
        trajectory=trajectory,
        state_id=str(decision["state_id"]),
        requested=requested,
    )
    record.update(
        {
            "candidate_event_step_ids": list(candidates),
            "reference_event_step_ids": list(candidates),
            "maximum_enumerated_budget": N8_MAXIMUM_ENUMERATED_BUDGET,
            "distance_rows": [],
            "full_reference_distance": 0.0,
            "full_reference_distance_measured": False,
            "full_reference_candidate_forward_count": 0,
        }
    )
    prompt_subsets = (candidates, *N8_COALITIONS)
    if not _preflight_prompts(
        record=record,
        trajectory=trajectory,
        decision_step_id=N8_DECISION_STEP_ID,
        prompt_subsets=prompt_subsets,
        runtime=runtime,
        image_bytes_loader=image_bytes_loader,
        image_decoder=image_decoder,
        maximum_context_tokens=maximum_context_tokens,
        completion_token_reserve=completion_token_reserve,
    ):
        return record
    reference_messages = build_long_horizon_messages(
        trajectory,
        decision_step_id=N8_DECISION_STEP_ID,
        restored_event_step_ids=candidates,
        image_bytes_loader=image_bytes_loader,
        image_decoder=image_decoder,
    )
    action = _run_reference_generations(
        record=record,
        runtime=runtime,
        reference_messages=reference_messages,
        parse_error_class=parse_error_class,
    )
    if action is None:
        return record
    backend = distance_backend or GPUFullVocabularyKLBackend(
        torch_module=runtime.torch
    )
    reference_logits, reference_metadata = runtime.teacher_forced_distance_logits(
        (reference_messages,), (action,)
    )
    _complete_operation(record["operation_counts"], "reference_teacher_forwards")
    reference_log_probs = backend.prepare_reference(reference_logits)
    del reference_logits
    record["teacher_forwards"].append(
        {
            "role": "reference",
            "restored_event_step_ids": list(candidates),
            "metadata": dict(reference_metadata),
        }
    )
    for coalition in N8_COALITIONS:
        messages = build_long_horizon_messages(
            trajectory,
            decision_step_id=N8_DECISION_STEP_ID,
            restored_event_step_ids=coalition,
            image_bytes_loader=image_bytes_loader,
            image_decoder=image_decoder,
        )
        try:
            distance = _measure_distance(
                record=record,
                runtime=runtime,
                distance_backend=backend,
                reference_log_probs=reference_log_probs,
                messages=messages,
                canonical_action=action,
                restored_event_step_ids=coalition,
            )
        except NonFiniteDistanceError as error:
            del reference_log_probs
            return _failure(
                record,
                category="NONFINITE_DISTANCE",
                stage="gpu_full_vocabulary_kl",
                message=str(error),
            )
        record["distance_rows"].append(
            {"restored_event_step_ids": list(coalition), "distance": distance}
        )
    del reference_log_probs
    _complete_operation(record["operation_counts"], "analytic_self_distance_rows")
    if (
        len(record["distance_rows"]) != N8_EXPECTED_ROW_COUNT
        or not _all_operations_complete(record["operation_counts"])
    ):
        raise RuntimeError("n=8 completed operation accounting drifted")
    record["valid"] = True
    record["outcome"] = "VALID_LONG_HORIZON_N8_STATE"
    return record


def _unique_n16_candidate_subsets(
    pair: SealedComparatorPair,
) -> tuple[tuple[int, ...], ...]:
    reference = pair.reference_event_step_ids
    ordered = (
        (),
        pair.left_selected_event_step_ids,
        pair.right_selected_event_step_ids,
    )
    result: list[tuple[int, ...]] = []
    for subset in ordered:
        if subset != reference and subset not in result:
            result.append(subset)
    return tuple(result)


def run_n16_pair_once(
    *,
    trajectory: Mapping[str, Any],
    pair: SealedComparatorPair,
    runtime: Any,
    image_bytes_loader: Callable[[str], bytes],
    image_decoder: Callable[[bytes], Any],
    distance_backend: Any | None = None,
    maximum_context_tokens: int = DEFAULT_MAXIMUM_CONTEXT_TOKENS,
    completion_token_reserve: int = DEFAULT_COMPLETION_TOKEN_RESERVE,
    parse_error_class: type[BaseException] = GUIOwlV21GenerationParseError,
) -> dict[str, Any]:
    """Measure exactly one sealed comparator pair under its union reference."""
    pair.validate(trajectory)
    decision, candidates = _validated_state(
        trajectory, decision_step_id=N16_DECISION_STEP_ID
    )
    reference = pair.reference_event_step_ids
    candidate_subsets = _unique_n16_candidate_subsets(pair)
    requested = {
        "processor_preflights": 1 + len(candidate_subsets),
        "canonical_action_parse_attempts": 2,
        "reference_teacher_forwards": 1,
        "candidate_teacher_forwards": len(candidate_subsets),
        "distance_rows": len(candidate_subsets),
        "analytic_self_distance_rows": 1,
    }
    record = _base_record(
        kind="n16_single_comparator_pair_union",
        trajectory=trajectory,
        state_id=str(decision["state_id"]),
        requested=requested,
    )
    record.update(
        {
            "candidate_event_step_ids": list(candidates),
            "selection_seal_sha256": pair.selection_seal_sha256,
            "budget_event_capacity": pair.budget_event_capacity,
            "left_selector_name": pair.left_selector_name,
            "right_selector_name": pair.right_selector_name,
            "left_selected_event_step_ids": list(pair.left_selected_event_step_ids),
            "right_selected_event_step_ids": list(pair.right_selected_event_step_ids),
            "pair_union_reference_event_step_ids": list(reference),
            "distance_rows": [],
            "supports_exact_subset_oracle": False,
            "all_16_reference_forward_count": 0,
            "cross_pair_table_construction_count": 0,
            "enumerated_subset_count": 0,
        }
    )
    prompt_subsets = (reference, *candidate_subsets)
    if not _preflight_prompts(
        record=record,
        trajectory=trajectory,
        decision_step_id=N16_DECISION_STEP_ID,
        prompt_subsets=prompt_subsets,
        runtime=runtime,
        image_bytes_loader=image_bytes_loader,
        image_decoder=image_decoder,
        maximum_context_tokens=maximum_context_tokens,
        completion_token_reserve=completion_token_reserve,
    ):
        return record
    reference_messages = build_long_horizon_messages(
        trajectory,
        decision_step_id=N16_DECISION_STEP_ID,
        restored_event_step_ids=reference,
        image_bytes_loader=image_bytes_loader,
        image_decoder=image_decoder,
    )
    action = _run_reference_generations(
        record=record,
        runtime=runtime,
        reference_messages=reference_messages,
        parse_error_class=parse_error_class,
    )
    if action is None:
        return record
    backend = distance_backend or GPUFullVocabularyKLBackend(
        torch_module=runtime.torch
    )
    reference_logits, reference_metadata = runtime.teacher_forced_distance_logits(
        (reference_messages,), (action,)
    )
    _complete_operation(record["operation_counts"], "reference_teacher_forwards")
    reference_log_probs = backend.prepare_reference(reference_logits)
    del reference_logits
    record["teacher_forwards"].append(
        {
            "role": "reference",
            "restored_event_step_ids": list(reference),
            "metadata": dict(reference_metadata),
        }
    )
    measured: dict[tuple[int, ...], float] = {}
    for subset in candidate_subsets:
        messages = build_long_horizon_messages(
            trajectory,
            decision_step_id=N16_DECISION_STEP_ID,
            restored_event_step_ids=subset,
            image_bytes_loader=image_bytes_loader,
            image_decoder=image_decoder,
        )
        try:
            measured[subset] = _measure_distance(
                record=record,
                runtime=runtime,
                distance_backend=backend,
                reference_log_probs=reference_log_probs,
                messages=messages,
                canonical_action=action,
                restored_event_step_ids=subset,
            )
        except NonFiniteDistanceError as error:
            del reference_log_probs
            return _failure(
                record,
                category="NONFINITE_DISTANCE",
                stage="gpu_full_vocabulary_kl",
                message=str(error),
            )
    del reference_log_probs
    _complete_operation(record["operation_counts"], "analytic_self_distance_rows")
    measured[reference] = 0.0
    ordered_rows: list[dict[str, Any]] = []
    for subset in (
        (),
        pair.left_selected_event_step_ids,
        pair.right_selected_event_step_ids,
        reference,
    ):
        if any(row["restored_event_step_ids"] == list(subset) for row in ordered_rows):
            continue
        ordered_rows.append(
            {
                "restored_event_step_ids": list(subset),
                "distance": measured[subset],
                "analytic_self_distance": subset == reference,
            }
        )
    record["distance_rows"] = ordered_rows
    record["selector_distances"] = {
        pair.left_selector_name: measured[pair.left_selected_event_step_ids],
        pair.right_selector_name: measured[pair.right_selected_event_step_ids],
    }
    if not _all_operations_complete(record["operation_counts"]):
        raise RuntimeError("n=16 completed operation accounting drifted")
    record["valid"] = True
    record["outcome"] = "VALID_LONG_HORIZON_N16_PAIR"
    return record


def _work_item_identity(
    *,
    kind: str,
    trajectory: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    trajectory_witness = _sha256_text(
        trajectory.get("content_witness_sha256"), "trajectory content witness"
    )
    value = {
        "protocol_id": PROTOCOL_ID,
        "kind": kind,
        "source_id": trajectory.get("source_id"),
        "trajectory_content_witness_sha256": trajectory_witness,
        **dict(extra),
    }
    value["work_item_sha256"] = _sha256_bytes(_canonical_json_bytes(value))
    return value


def n8_work_item_identity(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    decision, _ = _validated_state(trajectory, decision_step_id=N8_DECISION_STEP_ID)
    return _work_item_identity(
        kind="n8_exact_at_most_four",
        trajectory=trajectory,
        extra={"state_id": decision["state_id"]},
    )


def n16_work_item_identity(
    trajectory: Mapping[str, Any], pair: SealedComparatorPair
) -> dict[str, Any]:
    pair.validate(trajectory)
    return _work_item_identity(
        kind="n16_single_comparator_pair_union",
        trajectory=trajectory,
        extra={
            "state_id": pair.state_id,
            "selection_seal_sha256": pair.selection_seal_sha256,
            "budget_event_capacity": pair.budget_event_capacity,
            "left_selector_name": pair.left_selector_name,
            "right_selector_name": pair.right_selector_name,
            "left_selected_event_step_ids": list(pair.left_selected_event_step_ids),
            "right_selected_event_step_ids": list(pair.right_selected_event_step_ids),
        },
    )


def _validate_terminal_record_for_work_item(
    record: Mapping[str, Any], work_item: Mapping[str, Any]
) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("terminal record must be a mapping")
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("protocol_id") != PROTOCOL_ID
        or record.get("kind") != work_item.get("kind")
        or record.get("source_id") != work_item.get("source_id")
        or record.get("state_id") != work_item.get("state_id")
        or type(record.get("valid")) is not bool
    ):
        raise ValueError("terminal record differs from its deterministic work item")
    if any(record.get(name) != 0 for name in ("retry_count", "top_up_count", "reserve_access_count")):
        raise ValueError("terminal record violates retry, top-up, or reserve firewalls")
    validate_operation_accounting(record.get("operation_counts"))
    if work_item.get("kind") == "n16_single_comparator_pair_union":
        for name in (
            "selection_seal_sha256",
            "budget_event_capacity",
            "left_selector_name",
            "right_selector_name",
            "left_selected_event_step_ids",
            "right_selected_event_step_ids",
        ):
            if record.get(name) != work_item.get(name):
                raise ValueError("n=16 terminal record differs from its sealed pair")


class DeterministicReceiptStore:
    """Map a frozen work item to one exclusive terminal JSON receipt."""

    def __init__(self, root: str | Path, *, run_identity_sha256: str) -> None:
        self.root = Path(root)
        self.run_identity_sha256 = _sha256_text(
            run_identity_sha256, "run identity SHA256"
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, work_item: Mapping[str, Any]) -> Path:
        digest = _sha256_text(work_item.get("work_item_sha256"), "work item SHA256")
        replay = dict(work_item)
        replay.pop("work_item_sha256")
        if _sha256_bytes(_canonical_json_bytes(replay)) != digest:
            raise ValueError("work item SHA256 does not replay")
        return self.root / f"{digest}.json"

    def load(self, work_item: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self.path_for(work_item)
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise ValueError("terminal receipt must be a regular file")
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("terminal receipt is not strict JSON") from error
        expected_keys = {
            "schema_version",
            "protocol_id",
            "run_identity_sha256",
            "work_item",
            "terminal_record",
            "terminal_record_sha256",
        }
        if not isinstance(envelope, dict) or set(envelope) != expected_keys:
            raise ValueError("terminal receipt envelope drifted")
        if (
            envelope["schema_version"] != SCHEMA_VERSION
            or envelope["protocol_id"] != RECEIPT_PROTOCOL_ID
            or envelope["run_identity_sha256"] != self.run_identity_sha256
            or envelope["work_item"] != dict(work_item)
            or _sha256_bytes(_canonical_json_bytes(envelope["terminal_record"]))
            != envelope["terminal_record_sha256"]
        ):
            raise ValueError("terminal receipt identity or payload drifted")
        record = dict(envelope["terminal_record"])
        _validate_terminal_record_for_work_item(record, work_item)
        return record

    def write(self, work_item: Mapping[str, Any], terminal_record: Mapping[str, Any]) -> Path:
        path = self.path_for(work_item)
        record = dict(terminal_record)
        _validate_terminal_record_for_work_item(record, work_item)
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": RECEIPT_PROTOCOL_ID,
            "run_identity_sha256": self.run_identity_sha256,
            "work_item": dict(work_item),
            "terminal_record": record,
            "terminal_record_sha256": _sha256_bytes(_canonical_json_bytes(record)),
        }
        payload = (
            json.dumps(
                envelope,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        temporary.unlink()
        directory = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return path


def run_n8_state_with_resume(
    *,
    receipt_store: DeterministicReceiptStore,
    trajectory: Mapping[str, Any],
    **runtime_kwargs: Any,
) -> tuple[dict[str, Any], bool]:
    work_item = n8_work_item_identity(trajectory)
    existing = receipt_store.load(work_item)
    if existing is not None:
        return existing, True
    record = run_n8_state_once(trajectory=trajectory, **runtime_kwargs)
    receipt_store.write(work_item, record)
    return record, False


def run_n16_pair_with_resume(
    *,
    receipt_store: DeterministicReceiptStore,
    trajectory: Mapping[str, Any],
    pair: SealedComparatorPair,
    **runtime_kwargs: Any,
) -> tuple[dict[str, Any], bool]:
    work_item = n16_work_item_identity(trajectory, pair)
    existing = receipt_store.load(work_item)
    if existing is not None:
        return existing, True
    record = run_n16_pair_once(
        trajectory=trajectory,
        pair=pair,
        **runtime_kwargs,
    )
    receipt_store.write(work_item, record)
    return record, False


__all__ = [
    "DEFAULT_MAXIMUM_CONTEXT_TOKENS",
    "DeterministicReceiptStore",
    "GPUFullVocabularyKLBackend",
    "LONG_HORIZON_ALLOWED_DECISION_STEPS",
    "N8_COALITIONS",
    "SealedComparatorPair",
    "build_long_horizon_runtime",
    "build_long_horizon_messages",
    "enumerate_n8_coalitions",
    "n8_work_item_identity",
    "n16_work_item_identity",
    "run_n8_state_once",
    "run_n8_state_with_resume",
    "run_n16_pair_once",
    "run_n16_pair_with_resume",
    "validate_operation_accounting",
]
