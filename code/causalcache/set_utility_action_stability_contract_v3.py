"""Source-only contract for the focused strict-determinism D2 diagnostic."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.set_utility_action_stability_contract_v1 import (
    canonical_json_bytes,
    canonical_pretty_json_bytes,
    sha256_bytes,
)
from causalcache.set_utility_action_stability_diagnostic_v1 import (
    _validate_metric_safe_tree,
)
from causalcache.set_utility_action_stability_diagnostic_v3 import (
    CONTROL_STATE_IDS,
    EXPECTED_ENCODE_CALL_CEILING,
    EXPECTED_GENERATION_CALL_CEILING,
    STATE_IDS,
    STATE_ROLES,
    STRICT_DETERMINISM_FROZEN_ENCODED_CONDITION,
    STRICT_DETERMINISM_PROFILE,
    TARGET_STATE_ID,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_action_stability_diagnostic_source_v3"
STATUS = "SOURCE_ONLY_STRICT_DETERMINISM_ACTION_STABILITY_D2_FROZEN"
VALIDATION_STATUS = "VALID_STRICT_DETERMINISM_ACTION_STABILITY_D2_SOURCE"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_action_stability_diagnostic_v3.json"
)
CANONICAL_EXECUTION_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_action_stability_diagnostic_v3_execution.json"
)
D1B_RESULT_COMMIT = "abdbf8853b6da9ce73cf7f110d44256763b6756e"
D1B_AGGREGATE_PATH = (
    "data/results/set_utility_action_stability_diagnostic_v2/aggregate.json"
)
D1B_AGGREGATE_SHA256 = (
    "b419f9640cb46276b7a52e292d6feabd81311f66efb22563642ce80c4561919c"
)
D1B_SUMMARY_PATH = (
    "data/results/set_utility_action_stability_diagnostic_v2/summary.json"
)
D1B_SUMMARY_SHA256 = (
    "78048c1bc65f036be7f15fac7435a640b24e1a1474eedd930d6c2f0e666cf367"
)
D1B_STATUS = "COMPLETED_EXACT_SIX_PROCESS_SDPA_CONTROL_AGGREGATE_V2"
D1B_VERDICT = "NO_GO_SDPA_CONTROL_REPEAT_INSTABILITY"

SOURCE_ENTRYPOINT_CANDIDATES = (
    "code/causalcache/policy/gui_owl_v2_1_action_stability_runtime_v3.py",
    "code/causalcache/set_utility_action_stability_contract_v3.py",
    "code/causalcache/set_utility_action_stability_diagnostic_v3.py",
    "code/causalcache/set_utility_action_stability_execution_v3.py",
    "code/scripts/aggregate_set_utility_action_stability_v3.py",
    "code/scripts/materialize_set_utility_action_stability_envelope_v3.py",
    "code/scripts/run_set_utility_action_stability_state_v3.py",
    "code/scripts/validate_set_utility_action_stability_contract_v3.py",
    "code/scripts/validate_set_utility_action_stability_envelope_v3.py",
)


@dataclass(frozen=True, slots=True)
class ActionStabilitySourceContractV3:
    repository_root: Path
    config_sha256: str
    data: Mapping[str, Any]


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"D2 {label} must be one regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"D2 {label} must be strict JSON") from error
    if not isinstance(value, dict) or canonical_pretty_json_bytes(value) != raw:
        raise ValueError(f"D2 {label} must be canonical pretty JSON")
    return value


def _file_binding(root: Path, relative: str) -> dict[str, Any]:
    path = root.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"D2 bound file is missing: {relative}")
    raw = path.read_bytes()
    return {"byte_count": len(raw), "path": relative, "sha256": sha256_bytes(raw)}


def _module_path(root: Path, module: str) -> str | None:
    if module == "causalcache" or not module.startswith("causalcache."):
        return None
    path = "code/" + module.replace(".", "/") + ".py"
    return path if root.joinpath(*PurePosixPath(path).parts).is_file() else None


def _discover_source_paths(root: Path) -> tuple[str, ...]:
    pending = list(SOURCE_ENTRYPOINT_CANDIDATES)
    seen: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.is_symlink():
            continue
        seen.add(relative)
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        for module in imports:
            candidate = _module_path(root, module)
            if candidate is not None and candidate not in seen:
                pending.append(candidate)
    return tuple(sorted(seen))


def _source_projection(root: Path) -> dict[str, Any]:
    paths = _discover_source_paths(root)
    files = [_file_binding(root, relative) for relative in paths]
    missing = [
        relative
        for relative in SOURCE_ENTRYPOINT_CANDIDATES
        if not root.joinpath(*PurePosixPath(relative).parts).is_file()
    ]
    return {
        "file_count": len(files),
        "files": files,
        "inventory_sha256": sha256_bytes(canonical_json_bytes(files)),
        "missing_reserved_entrypoint_paths": missing,
    }


def _d1b_context(root: Path) -> dict[str, Any]:
    aggregate_path = root.joinpath(*PurePosixPath(D1B_AGGREGATE_PATH).parts)
    summary_path = root.joinpath(*PurePosixPath(D1B_SUMMARY_PATH).parts)
    aggregate_binding = _file_binding(root, D1B_AGGREGATE_PATH)
    summary_binding = _file_binding(root, D1B_SUMMARY_PATH)
    if (
        aggregate_binding["sha256"] != D1B_AGGREGATE_SHA256
        or summary_binding["sha256"] != D1B_SUMMARY_SHA256
    ):
        raise ValueError("D2 D1b input bytes drifted")
    aggregate = _strict_json(aggregate_path, label="D1b aggregate")
    summary = _strict_json(summary_path, label="D1b summary")
    diagnostic = aggregate.get("diagnostic")
    states = diagnostic.get("states") if isinstance(diagnostic, Mapping) else None
    if (
        aggregate.get("status") != D1B_STATUS
        or aggregate.get("verdict") != D1B_VERDICT
        or summary.get("verdict") != D1B_VERDICT
        or not isinstance(states, list)
    ):
        raise ValueError("D2 D1b result identity drifted")
    by_state = {state.get("state_id"): state for state in states}
    if any(state_id not in by_state for state_id in STATE_IDS):
        raise ValueError("D2 D1b context is missing a frozen state")
    target_condition = by_state[TARGET_STATE_ID].get(
        "sdpa_numerical_control_condition"
    )
    control_conditions = [
        by_state[state_id].get("sdpa_numerical_control_condition")
        for state_id in CONTROL_STATE_IDS
    ]
    if (
        not isinstance(target_condition, Mapping)
        or target_condition.get("canonical_action_equal") is not False
        or any(
            not isinstance(item, Mapping)
            or item.get("canonical_action_equal") is not True
            for item in control_conditions
        )
    ):
        raise ValueError("D2 target/control parent roles drifted")
    projection = [
        {
            "parent_diagnosis": by_state[state_id].get("diagnosis"),
            "parent_repeat_stable": state_id in CONTROL_STATE_IDS,
            "role": STATE_ROLES[state_id],
            "state_id": state_id,
        }
        for state_id in STATE_IDS
    ]
    _validate_metric_safe_tree(projection)
    return {
        "aggregate_binding": aggregate_binding,
        "projection": projection,
        "projection_sha256": sha256_bytes(canonical_json_bytes(projection)),
        "summary_binding": summary_binding,
    }


def build_action_stability_source_v3_config_skeleton(
    *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("D2 repository root is invalid")
    context = _d1b_context(root)
    return {
        "authorization": {
            "access_sealed_test": False,
            "execution_authorized": False,
            "generate_restoration_labels": False,
            "load_policy_or_vision_model": False,
            "mutate_hugging_face": False,
            "read_git_bound_d1b_metric_context": True,
            "run_closed_loop": False,
            "run_gpu_or_cuda": False,
            "train_predictor": False,
            "validate_source_contract": True,
            "write_diagnostic_result": False,
        },
        "diagnostic": {
            "condition": {
                "condition_id": STRICT_DETERMINISM_FROZEN_ENCODED_CONDITION,
                "cublas_workspace_config": ":4096:8",
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "deterministic_algorithms": True,
                "deterministic_warn_only": False,
                "encode_calls_per_state": 1,
                "encoding_mode": (
                    "one_exact_gpu_tensor_mapping_reused_for_both_generations"
                ),
                "float32_matmul_precision": "highest",
                "fresh_os_process_per_state": True,
                "generation_calls_per_state": 2,
                "memory_efficient_sdpa_required": True,
                "observed_attention": {"text": "sdpa", "top": "sdpa", "vision": "sdpa"},
                "profile": STRICT_DETERMINISM_PROFILE,
                "tf32_allowed": False,
            },
            "encode_call_ceiling": EXPECTED_ENCODE_CALL_CEILING,
            "generation_call_ceiling": EXPECTED_GENERATION_CALL_CEILING,
            "no_retry": True,
            "no_top_up": True,
            "process_count": len(STATE_IDS),
            "roster": list(context["projection"]),
            "state_ids": list(STATE_IDS),
            "state_ids_sha256": sha256_bytes(canonical_json_bytes(list(STATE_IDS))),
        },
        "execution_boundary": {
            "execution_b_must_be_direct_child_of_source_a": True,
            "execution_b_only_changed_path": CANONICAL_EXECUTION_CONFIG_PATH,
            "fresh_five_second_fleet_preflight_required": True,
            "max_concurrent_state_processes": 3,
            "process_per_state": True,
            "required_gpu_class": "NVIDIA_H200",
            "required_host_class": "Hyper_H200",
            "same_host_container_runtime_required": True,
            "source_a_requires_clean_pushed_main": True,
        },
        "immutable_inputs": {
            "d1b_aggregate": context["aggregate_binding"],
            "d1b_projection_sha256": context["projection_sha256"],
            "d1b_result_commit": D1B_RESULT_COMMIT,
            "d1b_summary": context["summary_binding"],
            "d1b_verdict": D1B_VERDICT,
        },
        "metric_only_result_contract": {
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
        },
        "preregistered_interpretation": {
            "verdict_precedence": [
                "INVALID_RUNTIME_FAILURE",
                "INVALID_STABLE_CONTROL_INSTABILITY",
                "NO_GO_STRICT_DETERMINISM_REPEAT_INSTABILITY",
                "PASS_STRICT_DETERMINISM_REPEAT_STABILITY_DIAGNOSTIC",
            ],
            "pass_authorizes_only_new_full_roster_source_freeze": True,
            "pass_does_not_retroactively_change_d1b": True,
            "pass_does_not_unlock_labels_training_matched_nll_or_closed_loop": True,
        },
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "scientific_locks": {
            "d1b_is_not_retried_or_topped_up": True,
            "formal_labels_locked": True,
            "matched_nll_locked": True,
            "predictor_training_locked": True,
            "run_closed_loop_locked": True,
        },
        "source": _source_projection(root),
        "status": STATUS,
    }


def validate_action_stability_source_v3_config(
    config: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    expected = build_action_stability_source_v3_config_skeleton(
        repository_root=repository_root
    )
    if not isinstance(config, Mapping) or dict(config) != expected:
        raise ValueError("D2 source config differs from the live skeleton")
    if expected["source"]["missing_reserved_entrypoint_paths"]:
        raise ValueError("D2 source config has missing reserved entrypoints")
    return {
        "config_sha256": sha256_bytes(canonical_pretty_json_bytes(expected)),
        "encode_call_ceiling": EXPECTED_ENCODE_CALL_CEILING,
        "generation_call_ceiling": EXPECTED_GENERATION_CALL_CEILING,
        "source_file_count": expected["source"]["file_count"],
        "source_inventory_sha256": expected["source"]["inventory_sha256"],
        "state_count": len(STATE_IDS),
        "status": VALIDATION_STATUS,
    }


def load_action_stability_source_v3_contract(
    *, repository_root: str | Path, config_path: str = CANONICAL_CONFIG_PATH
) -> ActionStabilitySourceContractV3:
    root = Path(repository_root).resolve()
    if config_path != CANONICAL_CONFIG_PATH:
        raise ValueError("D2 source config path drifted")
    path = root.joinpath(*PurePosixPath(config_path).parts)
    data = _strict_json(path, label="source config")
    validation = validate_action_stability_source_v3_config(
        data, repository_root=root
    )
    return ActionStabilitySourceContractV3(
        repository_root=root,
        config_sha256=validation["config_sha256"],
        data=data,
    )


__all__ = [
    "CANONICAL_CONFIG_PATH",
    "CANONICAL_EXECUTION_CONFIG_PATH",
    "D1B_AGGREGATE_PATH",
    "D1B_AGGREGATE_SHA256",
    "D1B_RESULT_COMMIT",
    "D1B_STATUS",
    "D1B_VERDICT",
    "VALIDATION_STATUS",
    "build_action_stability_source_v3_config_skeleton",
    "load_action_stability_source_v3_contract",
    "validate_action_stability_source_v3_config",
]
