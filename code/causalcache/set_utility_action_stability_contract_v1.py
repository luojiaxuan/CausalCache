"""Fail-closed source contract for the train-only D1 action-stability diagnostic."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v1 import (
    AUTO_FRESH_ENCODE_CONDITION,
    AUTO_FROZEN_ENCODED_CONDITION,
    EAGER_FROZEN_ENCODED_CONTROL,
)
from causalcache.set_utility_action_stability_diagnostic_v1 import (
    CONDITION_ORDER,
    CONTROL_STATE_IDS,
    EXPECTED_ENCODE_CALL_CEILING,
    EXPECTED_GENERATION_CALL_CEILING,
    MISMATCH_STATE_IDS,
    STATE_IDS,
    WORKER_INDEX_BY_STATE,
)
from causalcache.set_utility_gui_owl_v2_1_action_stability_adapter_v1 import (
    REPEAT_COUNT,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_action_stability_diagnostic_source_v1"
STATUS = "SOURCE_ONLY_SET_UTILITY_ACTION_STABILITY_DIAGNOSTIC_V1_FROZEN"
VALIDATION_STATUS = "VALID_SET_UTILITY_ACTION_STABILITY_DIAGNOSTIC_SOURCE_V1"

CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_action_stability_diagnostic_v1.json"
)
CONTRACT_PATH = "code/causalcache/set_utility_action_stability_contract_v1.py"
CLI_PATH = "code/scripts/validate_set_utility_action_stability_contract_v1.py"

PARENT_SOURCE_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_train_only_throughput_pilot_"
    "v2_candidate_schedule_key_repair.json"
)
PARENT_EXECUTION_ENVELOPE_PATH = (
    "code/configs/causalcache_set_utility_train_only_throughput_pilot_"
    "v2_candidate_schedule_key_repair_execution.json"
)
PARENT_SUMMARY_PATH = (
    "data/results/set_utility_train_only_throughput_pilot_"
    "v2_candidate_schedule_key_repair/summary.json"
)
PARENT_AGGREGATE_PATH = (
    "data/results/set_utility_train_only_throughput_pilot_"
    "v2_candidate_schedule_key_repair/aggregate.json"
)

PARENT_SOURCE_COMMIT = "d5e0cca5c5e05d4aeeef74a1bbfae5685a4254c9"
PARENT_ENVELOPE_COMMIT = "e5002d8820b3e4c8c89343b73ba4a5d13aa117c3"
PARENT_RESULT_COMMIT = "0779a64f95ba42333c51f224deac3790ee66a79e"
PARENT_SOURCE_CONFIG_SHA256 = (
    "b2e224147a70dbec14c389098a8fdcdd80b789b4f7f7cbafca15857102eb41ae"
)
PARENT_EXECUTION_ENVELOPE_SHA256 = (
    "dd0e64fb40bd39f839a1df91240a1e9a1a528a0bbfe9131d49ba726e8ecf4e3a"
)
PARENT_SUMMARY_SHA256 = (
    "832b78369ccc02aff11324fd622f0b789655524a282ec671c7ae314ca53e636d"
)
PARENT_AGGREGATE_SHA256 = (
    "35e0c250232efbd2b9bdccfb1b04a7c372fbed892635342cce4f9f81097ff33f"
)
PARENT_SUMMARY_STATUS = (
    "VALID_COMPLETED_SET_UTILITY_TRAIN_ONLY_THROUGHPUT_PILOT_V2_"
    "SELECTION_NO_GO_REFERENCE_ACTION_INSTABILITY"
)
PARENT_AGGREGATE_STATUS = "COMPLETED_THROUGHPUT_PILOT_EXECUTION_AGGREGATE_V1"

WORKER_COUNT = 4
GENERATION_CALL_CEILING = 36
ENCODE_CALL_CEILING = 24

_ROSTER_METADATA = {
    "0296753837938323:decision:006": {
        "candidate_capacity_stratum": "decisions_6_9",
        "decision_step_id": 6,
        "initial_candidate_count": 4,
        "parent_observation": "CROSS_VARIANT_REFERENCE_ACTION_MISMATCH",
        "parent_role": "mismatch",
    },
    "0310939638496410:decision:006": {
        "candidate_capacity_stratum": "decisions_6_9",
        "decision_step_id": 6,
        "initial_candidate_count": 4,
        "parent_observation": "MICROBATCH_1__REFERENCE_ACTION_MISMATCH",
        "parent_role": "mismatch",
    },
    "0336706763935531:decision:006": {
        "candidate_capacity_stratum": "decisions_6_9",
        "decision_step_id": 6,
        "initial_candidate_count": 4,
        "parent_observation": "COMPLETED_STABLE_CONTROL",
        "parent_role": "stable_control",
    },
    "0271654003819383:decision:010": {
        "candidate_capacity_stratum": "decisions_10_17",
        "decision_step_id": 10,
        "initial_candidate_count": 8,
        "parent_observation": "MICROBATCH_1__REFERENCE_ACTION_MISMATCH",
        "parent_role": "mismatch",
    },
    "0279447750102246:decision:010": {
        "candidate_capacity_stratum": "decisions_10_17",
        "decision_step_id": 10,
        "initial_candidate_count": 8,
        "parent_observation": "MICROBATCH_1__REFERENCE_ACTION_MISMATCH",
        "parent_role": "mismatch",
    },
    "0268406573756492:decision:010": {
        "candidate_capacity_stratum": "decisions_10_17",
        "decision_step_id": 10,
        "initial_candidate_count": 8,
        "parent_observation": "COMPLETED_STABLE_CONTROL",
        "parent_role": "stable_control",
    },
}

# note (luojiaxuan): These reserved entrypoints make the inventory automatically
# absorb the formal envelope/runner implementation once those new files exist.
SOURCE_ENTRYPOINT_CANDIDATES = (
    CONTRACT_PATH,
    CLI_PATH,
    "code/causalcache/set_utility_action_stability_diagnostic_v1.py",
    "code/causalcache/set_utility_gui_owl_v2_1_action_stability_adapter_v1.py",
    "code/causalcache/policy/gui_owl_v2_1_action_stability_runtime_v1.py",
    "code/causalcache/set_utility_action_stability_envelope_v1.py",
    "code/causalcache/set_utility_action_stability_execution_v1.py",
    "code/scripts/materialize_set_utility_action_stability_envelope_v1.py",
    "code/scripts/run_set_utility_action_stability_worker_v1.py",
    "code/scripts/aggregate_set_utility_action_stability_v1.py",
    "code/scripts/validate_set_utility_action_stability_envelope_v1.py",
)


@dataclass(frozen=True, slots=True)
class ActionStabilitySourceContractV1:
    data: Mapping[str, Any]
    repository_root: Path
    config_sha256: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite value {value}")

    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    value = json.loads(
        decoded,
        object_pairs_hook=unique,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _repository_file(root: Path, relative: str, *, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"{label} path is invalid")
    path = root.joinpath(*pure.parts)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or not a regular file: {relative}")
    return path


def _file_binding(root: Path, relative: str) -> dict[str, Any]:
    payload = _repository_file(root, relative, label="bound file").read_bytes()
    return {
        "byte_count": len(payload),
        "path": relative,
        "sha256": sha256_bytes(payload),
    }


def _load_bound_json(
    root: Path,
    relative: str,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _repository_file(root, relative, label=label)
    payload = path.read_bytes()
    if sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"{label} SHA256 drifted")
    return _strict_json_object(payload, label=label), {
        "byte_count": len(payload),
        "path": relative,
        "sha256": expected_sha256,
    }


def _module_path(root: Path, module: str) -> str | None:
    base = module.replace(".", "/")
    for relative in (f"code/{base}.py", f"code/{base}/__init__.py"):
        path = root.joinpath(*PurePosixPath(relative).parts)
        if path.is_file() and not path.is_symlink():
            return relative
    return None


def _discover_source_paths(root: Path) -> tuple[str, ...]:
    pending = [
        path
        for path in SOURCE_ENTRYPOINT_CANDIDATES
        if root.joinpath(*PurePosixPath(path).parts).is_file()
    ]
    paths: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in paths:
            continue
        path = _repository_file(root, relative, label="D1 source")
        paths.add(relative)
        tree = ast.parse(path.read_bytes(), filename=relative)
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError("D1 source must use absolute project imports")
                if node.module is not None:
                    modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                if module.startswith("causalcache"):
                    dependency = _module_path(root, module)
                    if dependency is None:
                        raise ValueError(f"D1 project import does not resolve: {module}")
                    pending.append(dependency)
    for relative in (
        "code/causalcache/__init__.py",
        "code/causalcache/data/__init__.py",
        "code/causalcache/policy/__init__.py",
    ):
        if root.joinpath(*PurePosixPath(relative).parts).is_file():
            paths.add(relative)
    return tuple(sorted(paths))


def _source_projection(root: Path) -> dict[str, Any]:
    entrypoints = [
        path
        for path in SOURCE_ENTRYPOINT_CANDIDATES
        if root.joinpath(*PurePosixPath(path).parts).is_file()
    ]
    inventory = [_file_binding(root, path) for path in _discover_source_paths(root)]
    return {
        "entrypoint_paths": entrypoints,
        "file_count": len(inventory),
        "inventory": inventory,
        "inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "reserved_entrypoint_paths": list(SOURCE_ENTRYPOINT_CANDIDATES),
    }


def _validate_parent_result(
    summary: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> None:
    if summary.get("status") != PARENT_SUMMARY_STATUS:
        raise ValueError("parent throughput summary status drifted")
    selection = summary.get("selection")
    if selection != {
        "outcome": "NO_GO",
        "reason": "MICROBATCH_1_FAILURE",
        "selected_reference_teacher_microbatch_size": None,
    }:
        raise ValueError("parent throughput summary selection drifted")
    observed_failures = {
        (record.get("state_id"), record.get("failure_class"))
        for record in summary.get("failure_breakdown", ())
        if isinstance(record, Mapping)
    }
    expected_failures = {
        (state_id, metadata["parent_observation"])
        for state_id, metadata in _ROSTER_METADATA.items()
        if metadata["parent_role"] == "mismatch"
    }
    if observed_failures != expected_failures:
        raise ValueError("parent throughput mismatch roster drifted")
    if aggregate.get("status") != PARENT_AGGREGATE_STATUS:
        raise ValueError("parent throughput aggregate status drifted")
    if aggregate.get("selection") != selection or aggregate.get("metric_only") is not True:
        raise ValueError("parent throughput aggregate selection drifted")


def _roster() -> list[dict[str, Any]]:
    if set(_ROSTER_METADATA) != set(STATE_IDS):
        raise AssertionError("D1 roster metadata drifted from the source core")
    if set(MISMATCH_STATE_IDS) != {
        state_id
        for state_id, value in _ROSTER_METADATA.items()
        if value["parent_role"] == "mismatch"
    }:
        raise AssertionError("D1 mismatch roster drifted from the source core")
    if set(CONTROL_STATE_IDS) != {
        state_id
        for state_id, value in _ROSTER_METADATA.items()
        if value["parent_role"] == "stable_control"
    }:
        raise AssertionError("D1 control roster drifted from the source core")
    return [
        {
            **_ROSTER_METADATA[state_id],
            "source_id": state_id.split(":", 1)[0],
            "state_id": state_id,
            "worker_index": WORKER_INDEX_BY_STATE[state_id],
        }
        for state_id in STATE_IDS
    ]


def _conditions() -> list[dict[str, Any]]:
    expected = (
        AUTO_FRESH_ENCODE_CONDITION,
        AUTO_FROZEN_ENCODED_CONDITION,
        EAGER_FROZEN_ENCODED_CONTROL,
    )
    if tuple(CONDITION_ORDER) != expected or REPEAT_COUNT != 2:
        raise AssertionError("D1 condition core drifted")
    return [
        {
            "attention_runtime": "parent_v2_automatic_attention",
            "condition_id": AUTO_FRESH_ENCODE_CONDITION,
            "encode_calls_per_state": REPEAT_COUNT,
            "encoding_mode": "fresh_processor_encode_before_each_generation",
            "observed_attention_implementation": {
                "text": "sdpa",
                "top": "sdpa",
                "vision": "sdpa",
            },
            "process_stage": "auto",
            "generation_calls_per_state": REPEAT_COUNT,
            "repeat_count": REPEAT_COUNT,
        },
        {
            "attention_runtime": "parent_v2_automatic_attention",
            "condition_id": AUTO_FROZEN_ENCODED_CONDITION,
            "encode_calls_per_state": 1,
            "encoding_mode": "one_exact_gpu_tensor_mapping_reused_for_both_generations",
            "observed_attention_implementation": {
                "text": "sdpa",
                "top": "sdpa",
                "vision": "sdpa",
            },
            "process_stage": "auto",
            "generation_calls_per_state": REPEAT_COUNT,
            "repeat_count": REPEAT_COUNT,
        },
        {
            "attention_runtime": "v2_2_eager_numerical_control",
            "condition_id": EAGER_FROZEN_ENCODED_CONTROL,
            "encode_calls_per_state": 1,
            "encoding_mode": "one_exact_gpu_tensor_mapping_reused_for_both_generations",
            "observed_attention_implementation": {
                "text": "eager",
                "top": "eager",
                "vision": "eager",
            },
            "process_stage": "eager",
            "generation_calls_per_state": REPEAT_COUNT,
            "repeat_count": REPEAT_COUNT,
            "strict_cuda_determinism_claimed": False,
        },
    ]


def build_action_stability_source_v1_config_skeleton(
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Rebuild the D1 source-only contract from bound Git bytes."""
    root = Path(repository_root).resolve()
    _, parent_source = _load_bound_json(
        root,
        PARENT_SOURCE_CONFIG_PATH,
        PARENT_SOURCE_CONFIG_SHA256,
        label="parent throughput source config",
    )
    _, parent_envelope = _load_bound_json(
        root,
        PARENT_EXECUTION_ENVELOPE_PATH,
        PARENT_EXECUTION_ENVELOPE_SHA256,
        label="parent throughput execution envelope",
    )
    summary, summary_binding = _load_bound_json(
        root,
        PARENT_SUMMARY_PATH,
        PARENT_SUMMARY_SHA256,
        label="parent throughput result summary",
    )
    aggregate, aggregate_binding = _load_bound_json(
        root,
        PARENT_AGGREGATE_PATH,
        PARENT_AGGREGATE_SHA256,
        label="parent throughput aggregate",
    )
    _validate_parent_result(summary, aggregate)
    roster = _roster()
    conditions = _conditions()
    generation_ceiling = len(roster) * sum(
        condition["generation_calls_per_state"] for condition in conditions
    )
    encode_ceiling = len(roster) * sum(
        condition["encode_calls_per_state"] for condition in conditions
    )
    if (
        generation_ceiling != GENERATION_CALL_CEILING
        or generation_ceiling != EXPECTED_GENERATION_CALL_CEILING
        or encode_ceiling != ENCODE_CALL_CEILING
        or encode_ceiling != EXPECTED_ENCODE_CALL_CEILING
    ):
        raise AssertionError("D1 operation ceiling reconstruction drifted")
    worker_mapping = [
        {
            "state_ids": [
                state_id
                for state_id in STATE_IDS
                if WORKER_INDEX_BY_STATE[state_id] == worker_index
            ],
            "worker_index": worker_index,
        }
        for worker_index in range(WORKER_COUNT)
    ]
    return {
        "authorization": {
            "access_sealed_test": False,
            "generate_restoration_labels": False,
            "load_policy_or_vision_model": False,
            "mutate_hugging_face": False,
            "read_remote_processor_artifact": False,
            "read_repository_bound_metadata": True,
            "run_closed_loop": False,
            "run_gpu_or_cuda": False,
            "train_predictor": False,
            "validate_source_contract": True,
            "write_diagnostic_result": False,
        },
        "diagnostic": {
            "aggregation_order": list(range(WORKER_COUNT)),
            "condition_order": list(CONDITION_ORDER),
            "conditions": conditions,
            "encode_call_ceiling": encode_ceiling,
            "generation_call_ceiling": generation_ceiling,
            "no_retry": True,
            "no_top_up": True,
            "role": "train_only_failure_localization_only",
            "roster": roster,
            "same_state_all_conditions_same_worker_device": True,
            "state_count": len(roster),
            "state_ids_sha256": sha256_bytes(canonical_json_bytes(list(STATE_IDS))),
            "worker_count": WORKER_COUNT,
            "worker_mapping": worker_mapping,
        },
        "execution_boundary": {
            "cross_host_metric_merge_allowed": False,
            "execution_requires_separate_exact_envelope": True,
            "formal_run_allowed_from_source_commit": False,
            "new_attempt_identity_required": True,
            "parent_v2_retry_allowed": False,
            "required_gpu_class": "NVIDIA_H200",
            "required_host_class": "Hyper_H200",
            "same_host_container_and_runtime_stack_required": True,
            "worker_count": WORKER_COUNT,
        },
        "immutable_inputs": {
            "parent_aggregate": aggregate_binding,
            "parent_envelope_commit": PARENT_ENVELOPE_COMMIT,
            "parent_execution_envelope": parent_envelope,
            "parent_result_commit": PARENT_RESULT_COMMIT,
            "parent_source_commit": PARENT_SOURCE_COMMIT,
            "parent_source_config": parent_source,
            "parent_summary": summary_binding,
        },
        "metric_only_result_contract": {
            "action_handle_lifetime": "in_process_only_not_serialized",
            "allowed_observations": [
                "class_only_failure",
                "condition_and_state_identity",
                "encode_and_generation_counts",
                "exact_equality_booleans",
                "prepared_input_equality_booleans",
                "runtime_identity",
            ],
            "forbidden_serialized_fields": [
                "action",
                "action_text",
                "coordinate",
                "decoded_output",
                "decoded_output_digest",
                "generated_sequence_digest",
                "kl",
                "logits",
                "messages",
                "native_output",
                "text_argument",
                "token_ids",
                "tokens",
                "utility",
            ],
            "metric_safe_required": True,
            "retry_count": 0,
        },
        "negative_operations": {
            "closed_loop_episode_count": 0,
            "evaluation_state_access_count": 0,
            "gpu_or_cuda_call_count": 0,
            "hugging_face_mutation_count": 0,
            "matched_nll_count": 0,
            "policy_generation_count": 0,
            "predictor_training_count": 0,
            "processor_encode_count": 0,
            "remote_artifact_read_count": 0,
            "restoration_label_count": 0,
            "sealed_test_access_count": 0,
            "teacher_forward_count": 0,
            "tune_state_access_count": 0,
            "utility_or_kl_write_count": 0,
        },
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "scientific_locks": {
            "d1_does_not_select_teacher_microbatch": True,
            "d1_does_not_unlock_labels_or_training": True,
            "formal_labels_locked": True,
            "matched_nll_locked": True,
            "parent_v2_no_go_remains_immutable": True,
            "predictor_training_locked": True,
            "run_outcome_can_authorize_only_new_v3_throughput_source_freeze": True,
            "twelve_state_v3_throughput_required_before_labels": True,
        },
        "source": _source_projection(root),
        "status": STATUS,
    }


def validate_action_stability_source_v1_config(
    config: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("D1 source config must be one mapping")
    expected = build_action_stability_source_v1_config_skeleton(
        repository_root=repository_root
    )
    if dict(config) != expected:
        raise ValueError("D1 source config differs from the live skeleton")
    source = expected["source"]
    diagnostic = expected["diagnostic"]
    return {
        "config_sha256": sha256_bytes(canonical_pretty_json_bytes(expected)),
        "encode_call_ceiling": diagnostic["encode_call_ceiling"],
        "generation_call_ceiling": diagnostic["generation_call_ceiling"],
        "source_file_count": source["file_count"],
        "source_inventory_sha256": source["inventory_sha256"],
        "state_count": diagnostic["state_count"],
        "status": VALIDATION_STATUS,
        "worker_count": diagnostic["worker_count"],
    }


def load_action_stability_source_v1_contract(
    *,
    repository_root: str | Path,
    config_path: str | Path = CANONICAL_CONFIG_PATH,
) -> ActionStabilitySourceContractV1:
    root = Path(repository_root).resolve()
    supplied = Path(config_path)
    path = supplied if supplied.is_absolute() else root / supplied
    expected_path = (root / CANONICAL_CONFIG_PATH).resolve()
    if path.resolve() != expected_path:
        raise ValueError(f"D1 source config must be canonical: {expected_path}")
    payload = _repository_file(root, CANONICAL_CONFIG_PATH, label="D1 source config").read_bytes()
    config = _strict_json_object(payload, label="D1 source config")
    if payload != canonical_pretty_json_bytes(config):
        raise ValueError("D1 source config must be canonical pretty JSON")
    validate_action_stability_source_v1_config(config, repository_root=root)
    return ActionStabilitySourceContractV1(
        data=config,
        repository_root=root,
        config_sha256=sha256_bytes(payload),
    )


__all__ = [
    "ActionStabilitySourceContractV1",
    "CANONICAL_CONFIG_PATH",
    "CLI_PATH",
    "CONDITION_ORDER",
    "CONTRACT_PATH",
    "ENCODE_CALL_CEILING",
    "GENERATION_CALL_CEILING",
    "PARENT_AGGREGATE_PATH",
    "PARENT_AGGREGATE_SHA256",
    "PARENT_RESULT_COMMIT",
    "PARENT_SUMMARY_PATH",
    "PARENT_SUMMARY_SHA256",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "SOURCE_ENTRYPOINT_CANDIDATES",
    "STATE_IDS",
    "STATUS",
    "VALIDATION_STATUS",
    "WORKER_COUNT",
    "build_action_stability_source_v1_config_skeleton",
    "canonical_json_bytes",
    "canonical_pretty_json_bytes",
    "load_action_stability_source_v1_contract",
    "sha256_bytes",
    "validate_action_stability_source_v1_config",
]
