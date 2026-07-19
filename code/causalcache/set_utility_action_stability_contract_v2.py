"""Fail-closed source contract for the D1b controlled-SDPA diagnostic."""

from __future__ import annotations

import ast
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v1 import (
    AUTO_FRESH_ENCODE_CONDITION,
    AUTO_FROZEN_ENCODED_CONDITION,
)
from causalcache.set_utility_action_stability_contract_v1 import (
    canonical_json_bytes,
    canonical_pretty_json_bytes,
    sha256_bytes,
)
from causalcache.set_utility_action_stability_diagnostic_v1 import (
    PROTOCOL_ID as D1_DIAGNOSTIC_PROTOCOL_ID,
    _validate_condition_payload,
)
from causalcache.set_utility_action_stability_diagnostic_v2 import (
    CONTROL_STATE_IDS,
    EXPECTED_ENCODE_CALL_CEILING,
    EXPECTED_GENERATION_CALL_CEILING,
    MISMATCH_STATE_IDS,
    SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION,
    SDPA_NUMERICAL_CONTROL_PROFILE,
    STATE_IDS,
    STATE_WAVES,
)
from causalcache.set_utility_gui_owl_v2_1_action_stability_adapter_v1 import (
    REPEAT_COUNT,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_action_stability_diagnostic_source_v2"
STATUS = "SOURCE_ONLY_SET_UTILITY_ACTION_STABILITY_DIAGNOSTIC_V2_FROZEN"
VALIDATION_STATUS = "VALID_SET_UTILITY_ACTION_STABILITY_DIAGNOSTIC_SOURCE_V2"

CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_action_stability_diagnostic_v2.json"
)
CANONICAL_EXECUTION_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_action_stability_"
    "diagnostic_v2_execution.json"
)
CONTRACT_PATH = "code/causalcache/set_utility_action_stability_contract_v2.py"
CLI_PATH = "code/scripts/validate_set_utility_action_stability_contract_v2.py"

D1_RESULT_COMMIT = "a713555671d0c828b15d26f9652c7a889332cb57"
D1_AGGREGATE_PATH = (
    "data/results/set_utility_action_stability_diagnostic_v1/aggregate.json"
)
D1_AGGREGATE_SHA256 = (
    "2debde6de3f552e9551d0ee37d25b82fa2ca85dfb42746d389dde39eff577ba2"
)
D1_SUMMARY_PATH = (
    "data/results/set_utility_action_stability_diagnostic_v1/summary.json"
)
D1_SUMMARY_SHA256 = (
    "25221b05390d0aaafe42425ee6cb79cac31e6ce7e9ed9a90c73e44818ca4e4b9"
)
D1_RESULT_STATUS = "COMPLETED_EXACT_SIX_STATE_ACTION_STABILITY_AGGREGATE_V1"
D1_AGGREGATE_PROTOCOL_ID = (
    "causalcache_set_utility_action_stability_aggregate_v1"
)
D1_VERDICT = "INVALID_RUNTIME_FAILURE"

STATE_PROCESS_COUNT = 6
GPU_SLOT_COUNT = 4
GENERATION_CALL_CEILING = 12
ENCODE_CALL_CEILING = 6
WAVE_STATE_INDICES = ((0, 3, 4, 5), (1, 2))
EXPECTED_STATE_WAVES = tuple(
    tuple(STATE_IDS[index] for index in wave) for wave in WAVE_STATE_INDICES
)

STATE_DIAGNOSIS_PRECEDENCE = (
    {
        "diagnosis": "INVALID_CONDITION_EXECUTION_FAILURE",
        "d1_auto_fresh": "any",
        "d1_auto_frozen": "any",
        "parent_role": "any",
        "priority": 1,
        "sdpa_control": "execution_failure",
    },
    {
        "diagnosis": "STABLE_CONTROL_SDPA_CONTROL_INSTABILITY",
        "d1_auto_fresh": "any",
        "d1_auto_frozen": "any",
        "parent_role": "stable_control",
        "priority": 2,
        "sdpa_control": "repeat_unstable",
    },
    {
        "diagnosis": (
            "D1_AUTO_FROZEN_INSTABILITY_UNRESOLVED_BY_SDPA_CONTROL"
        ),
        "d1_auto_fresh": "any",
        "d1_auto_frozen": "repeat_unstable",
        "parent_role": "mismatch",
        "priority": 3,
        "sdpa_control": "repeat_unstable",
    },
    {
        "diagnosis": "MIXED_D1_FRESH_PATH_AND_SDPA_CONTROL_INSTABILITY",
        "d1_auto_fresh": "repeat_unstable",
        "d1_auto_frozen": "repeat_stable",
        "parent_role": "mismatch",
        "priority": 4,
        "sdpa_control": "repeat_unstable",
    },
    {
        "diagnosis": "SDPA_CONTROL_ONLY_INSTABILITY",
        "d1_auto_fresh": "repeat_stable",
        "d1_auto_frozen": "repeat_stable",
        "parent_role": "mismatch",
        "priority": 5,
        "sdpa_control": "repeat_unstable",
    },
    {
        "diagnosis": "AUTO_VS_CONTROLLED_SDPA_PROFILE_ASSOCIATION",
        "d1_auto_fresh": "any",
        "d1_auto_frozen": "repeat_unstable",
        "parent_role": "any",
        "priority": 6,
        "sdpa_control": "repeat_stable",
    },
    {
        "diagnosis": (
            "FRESH_VS_FROZEN_PATH_ASSOCIATION_WITH_SDPA_CONTROL_STABLE"
        ),
        "d1_auto_fresh": "repeat_unstable",
        "d1_auto_frozen": "repeat_stable",
        "parent_role": "any",
        "priority": 7,
        "sdpa_control": "repeat_stable",
    },
    {
        "diagnosis": (
            "PARENT_MISMATCH_NOT_REPRODUCED_UNDER_SDPA_CONTROL"
        ),
        "d1_auto_fresh": "repeat_stable",
        "d1_auto_frozen": "repeat_stable",
        "parent_role": "mismatch",
        "priority": 8,
        "sdpa_control": "repeat_stable",
    },
    {
        "diagnosis": "STABLE_CONTROL_REPRODUCED",
        "d1_auto_fresh": "repeat_stable",
        "d1_auto_frozen": "repeat_stable",
        "parent_role": "stable_control",
        "priority": 9,
        "sdpa_control": "repeat_stable",
    },
)

AGGREGATE_VERDICT_PRECEDENCE = (
    "INVALID_RUNTIME_FAILURE",
    "INVALID_STABLE_CONTROL_INSTABILITY",
    "NO_GO_SDPA_CONTROL_REPEAT_INSTABILITY",
    "PASS_MEMORY_SAFE_SDPA_NUMERICAL_CONTROL_REPEAT_STABILITY",
)

# note (luojiaxuan): These paths reserve the whole versioned D1b execution
# surface in source A.  Missing implementation files are omitted from the live
# inventory while development is in progress, but every path must exist before
# the canonical source config is frozen.
SOURCE_ENTRYPOINT_CANDIDATES = (
    CONTRACT_PATH,
    CLI_PATH,
    "code/causalcache/set_utility_action_stability_diagnostic_v2.py",
    "code/causalcache/policy/gui_owl_v2_1_action_stability_runtime_v2.py",
    "code/causalcache/set_utility_action_stability_envelope_v2.py",
    "code/causalcache/set_utility_action_stability_execution_v2.py",
    "code/scripts/materialize_set_utility_action_stability_envelope_v2.py",
    "code/scripts/validate_set_utility_action_stability_envelope_v2.py",
    "code/scripts/run_set_utility_action_stability_state_v2.py",
    "code/scripts/aggregate_set_utility_action_stability_v2.py",
)


@dataclass(frozen=True, slots=True)
class ActionStabilitySourceContractV2:
    data: Mapping[str, Any]
    repository_root: Path
    config_sha256: str


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, nested in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            value[key] = nested
        return value

    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    value = json.loads(
        decoded,
        object_pairs_hook=unique,
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError(f"{label} contains non-finite value {raw}")
        ),
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


def _load_historical_json(
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
    committed = subprocess.run(
        ["git", "show", f"{D1_RESULT_COMMIT}:{relative}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if committed != payload:
        raise ValueError(f"{label} differs from the frozen D1 result commit")
    value = _strict_json_object(payload, label=label)
    if payload != canonical_pretty_json_bytes(value):
        raise ValueError(f"{label} must be canonical pretty JSON")
    return value, {
        "byte_count": len(payload),
        "git_commit": D1_RESULT_COMMIT,
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
        relative
        for relative in SOURCE_ENTRYPOINT_CANDIDATES
        if root.joinpath(*PurePosixPath(relative).parts).is_file()
    ]
    paths: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in paths:
            continue
        path = _repository_file(root, relative, label="D1b source")
        paths.add(relative)
        tree = ast.parse(path.read_bytes(), filename=relative)
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError("D1b source must use absolute project imports")
                if node.module is not None:
                    modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                if module.startswith("causalcache"):
                    dependency = _module_path(root, module)
                    if dependency is None:
                        raise ValueError(
                            f"D1b project import does not resolve: {module}"
                        )
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
        relative
        for relative in SOURCE_ENTRYPOINT_CANDIDATES
        if root.joinpath(*PurePosixPath(relative).parts).is_file()
    ]
    missing = [
        relative
        for relative in SOURCE_ENTRYPOINT_CANDIDATES
        if not root.joinpath(*PurePosixPath(relative).parts).is_file()
    ]
    inventory = [_file_binding(root, relative) for relative in _discover_source_paths(root)]
    return {
        "entrypoint_paths": entrypoints,
        "file_count": len(inventory),
        "inventory": inventory,
        "inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "missing_reserved_entrypoint_paths": missing,
        "reserved_entrypoint_paths": list(SOURCE_ENTRYPOINT_CANDIDATES),
    }


def _repeat_stable(condition: Mapping[str, Any]) -> bool:
    if condition.get("failure_class") is not None:
        return False
    if not all(
        condition.get(key) is True
        for key in (
            "canonical_action_equal",
            "decoded_output_equal",
            "exact_generated_sequence_equal",
        )
    ):
        return False
    if condition.get("condition_id") == AUTO_FROZEN_ENCODED_CONDITION and not all(
        condition.get(key) is True
        for key in (
            "encoded_input_unchanged_after",
            "encoded_input_unchanged_before",
            "encoded_input_unchanged_between",
        )
    ):
        return False
    return True


def _parent_state_projection(
    parent_d1_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Project only the two frozen D1 auto rows used by D1b."""
    if not isinstance(parent_d1_state, Mapping):
        raise ValueError("D1b parent state must be one mapping")
    state_id = parent_d1_state.get("state_id")
    expected_role = (
        "mismatch"
        if state_id in MISMATCH_STATE_IDS
        else "stable_control" if state_id in CONTROL_STATE_IDS else None
    )
    conditions = parent_d1_state.get("conditions")
    if (
        expected_role is None
        or parent_d1_state.get("parent_role") != expected_role
        or parent_d1_state.get("metric_safe") is not True
        or isinstance(conditions, (str, bytes, bytearray, Mapping))
        or not isinstance(conditions, Sequence)
        or len(conditions) < 2
    ):
        raise ValueError("D1b parent state identity or auto context drifted")
    auto = [
        _validate_condition_payload(
            conditions[0], expected_condition_id=AUTO_FRESH_ENCODE_CONDITION
        ),
        _validate_condition_payload(
            conditions[1], expected_condition_id=AUTO_FROZEN_ENCODED_CONDITION
        ),
    ]
    if any(item["failure_class"] is not None for item in auto):
        raise ValueError("D1b requires complete frozen D1 auto context")
    return {
        "conditions": auto,
        "parent_role": expected_role,
        "state_id": state_id,
    }


def _d1_auto_context(
    aggregate: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        aggregate.get("protocol_id") != D1_AGGREGATE_PROTOCOL_ID
        or aggregate.get("status") != D1_RESULT_STATUS
        or aggregate.get("metric_safe") is not True
        or aggregate.get("diagnostic", {}).get("protocol_id")
        != D1_DIAGNOSTIC_PROTOCOL_ID
        or aggregate.get("diagnostic", {}).get("verdict") != D1_VERDICT
        or summary.get("status") != D1_RESULT_STATUS
        or summary.get("result_id")
        != "set_utility_action_stability_diagnostic_v1"
        or summary.get("aggregate")
        != {
            "bytes": 14672,
            "git_path": D1_AGGREGATE_PATH,
            "protocol_id": D1_AGGREGATE_PROTOCOL_ID,
            "sha256": D1_AGGREGATE_SHA256,
            "status": D1_RESULT_STATUS,
        }
        or summary.get("diagnostic", {}).get("verdict") != D1_VERDICT
    ):
        raise ValueError("frozen D1 result identity drifted")
    raw_states = aggregate.get("diagnostic", {}).get("states")
    if not isinstance(raw_states, list) or len(raw_states) != len(STATE_IDS):
        raise ValueError("frozen D1 state roster drifted")
    validated = [_parent_state_projection(raw) for raw in raw_states]
    by_state = {state["state_id"]: state for state in validated}
    if len(by_state) != len(STATE_IDS) or set(by_state) != set(STATE_IDS):
        raise ValueError("frozen D1 state identities drifted")
    projection_states = []
    historical_generation_calls = 0
    historical_encode_calls = 0
    stability: dict[str, tuple[bool, bool]] = {}
    for state_id in STATE_IDS:
        state = by_state[state_id]
        auto = state["conditions"][:2]
        if [item["condition_id"] for item in auto] != [
            AUTO_FRESH_ENCODE_CONDITION,
            AUTO_FROZEN_ENCODED_CONDITION,
        ] or any(item["failure_class"] is not None for item in auto):
            raise ValueError("D1b requires complete frozen D1 auto context")
        historical_generation_calls += sum(
            int(item["generation_call_count"]) for item in auto
        )
        historical_encode_calls += sum(int(item["encode_call_count"]) for item in auto)
        stability[state_id] = (_repeat_stable(auto[0]), _repeat_stable(auto[1]))
        projection_states.append(
            {
                "conditions": [dict(item) for item in auto],
                "parent_role": state["parent_role"],
                "state_id": state_id,
            }
        )
    if historical_generation_calls != 24 or historical_encode_calls != 18:
        raise ValueError("frozen D1 auto-context call counts drifted")
    summary_auto = summary.get("counts", {}).get("auto")
    if summary_auto != {
        "condition_count": 12,
        "encode_call_count": historical_encode_calls,
        "generation_call_count": historical_generation_calls,
        "state_count": 6,
        "worker_count": 4,
    }:
        raise ValueError("frozen D1 summary auto counts drifted")
    projection = {
        "condition_ids": [
            AUTO_FRESH_ENCODE_CONDITION,
            AUTO_FROZEN_ENCODED_CONDITION,
        ],
        "states": projection_states,
    }
    return {
        "historical_encode_call_count": historical_encode_calls,
        "historical_generation_call_count": historical_generation_calls,
        "projection_sha256": sha256_bytes(canonical_json_bytes(projection)),
        "stability_by_state": stability,
    }


def _process_schedule() -> list[dict[str, Any]]:
    if tuple(STATE_WAVES) != EXPECTED_STATE_WAVES:
        raise AssertionError("D1b non-contiguous 4+2 wave schedule drifted")
    schedule = []
    for wave_index, state_indices in enumerate(WAVE_STATE_INDICES):
        for gpu_slot, state_index in enumerate(state_indices):
            schedule.append(
                {
                    "gpu_slot": gpu_slot,
                    "state_id": STATE_IDS[state_index],
                    "state_index": state_index,
                    "wave_index": wave_index,
                }
            )
    return schedule


def _roster(stability: Mapping[str, tuple[bool, bool]]) -> list[dict[str, Any]]:
    schedule = {record["state_id"]: record for record in _process_schedule()}
    if set(MISMATCH_STATE_IDS).isdisjoint(CONTROL_STATE_IDS) is False or (
        set(MISMATCH_STATE_IDS) | set(CONTROL_STATE_IDS)
    ) != set(STATE_IDS):
        raise AssertionError("D1b mismatch/control roster drifted")
    return [
        {
            "d1_auto_fresh_repeat_stable": stability[state_id][0],
            "d1_auto_frozen_repeat_stable": stability[state_id][1],
            "gpu_slot": schedule[state_id]["gpu_slot"],
            "parent_role": (
                "mismatch" if state_id in MISMATCH_STATE_IDS else "stable_control"
            ),
            "source_id": state_id.split(":", 1)[0],
            "state_id": state_id,
            "state_index": STATE_IDS.index(state_id),
            "wave_index": schedule[state_id]["wave_index"],
        }
        for state_id in STATE_IDS
    ]


def build_action_stability_source_v2_config_skeleton(
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Rebuild the D1b source-only contract from Git-bound metric-safe bytes."""
    root = Path(repository_root).resolve()
    aggregate, aggregate_binding = _load_historical_json(
        root,
        D1_AGGREGATE_PATH,
        D1_AGGREGATE_SHA256,
        label="D1 aggregate",
    )
    summary, summary_binding = _load_historical_json(
        root,
        D1_SUMMARY_PATH,
        D1_SUMMARY_SHA256,
        label="D1 summary",
    )
    context = _d1_auto_context(aggregate, summary)
    schedule = _process_schedule()
    roster = _roster(context["stability_by_state"])
    if (
        len(schedule) != STATE_PROCESS_COUNT
        or len(roster) != STATE_PROCESS_COUNT
        or REPEAT_COUNT != 2
        or EXPECTED_GENERATION_CALL_CEILING != GENERATION_CALL_CEILING
        or EXPECTED_ENCODE_CALL_CEILING != ENCODE_CALL_CEILING
    ):
        raise AssertionError("D1b operation reconstruction drifted")
    return {
        "authorization": {
            "access_sealed_test": False,
            "execution_authorized": False,
            "generate_restoration_labels": False,
            "load_policy_or_vision_model": False,
            "mutate_hugging_face": False,
            "read_git_bound_d1_metric_context": True,
            "run_closed_loop": False,
            "run_gpu_or_cuda": False,
            "train_predictor": False,
            "validate_source_contract": True,
            "write_diagnostic_result": False,
        },
        "diagnostic": {
            "aggregation_state_order": list(STATE_IDS),
            "condition": {
                "condition_id": SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION,
                "encode_calls_per_state": 1,
                "encoding_mode": (
                    "one_exact_gpu_tensor_mapping_reused_for_both_generations"
                ),
                "fresh_os_process_per_state": True,
                "generation_calls_per_state": REPEAT_COUNT,
                "memory_efficient_sdpa_required": True,
                "numerical_controls_configured_before_cuda_initialization": True,
                "observed_attention_implementation": {
                    "text": "sdpa",
                    "top": "sdpa",
                    "vision": "sdpa",
                },
                "profile": SDPA_NUMERICAL_CONTROL_PROFILE,
                "repeat_count": REPEAT_COUNT,
                "requested_attention_implementation": "auto_parent_default",
                "strict_cuda_determinism_claimed": False,
            },
            "condition_id": SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION,
            "encode_call_ceiling": ENCODE_CALL_CEILING,
            "generation_call_ceiling": GENERATION_CALL_CEILING,
            "gpu_slot_count": GPU_SLOT_COUNT,
            "no_retry": True,
            "no_top_up": True,
            "process_count": STATE_PROCESS_COUNT,
            "process_schedule": schedule,
            "repeat_count": REPEAT_COUNT,
            "role": "train_only_memory_safe_control_localization_only",
            "roster": roster,
            "state_count": len(STATE_IDS),
            "state_ids": list(STATE_IDS),
            "state_ids_sha256": sha256_bytes(canonical_json_bytes(list(STATE_IDS))),
            "state_waves": [list(wave) for wave in EXPECTED_STATE_WAVES],
            "wave_sizes": [len(wave) for wave in EXPECTED_STATE_WAVES],
        },
        "execution_boundary": {
            "cross_host_metric_merge_allowed": False,
            "execution_b_must_be_direct_child_of_source_a": True,
            "execution_b_only_changed_path": CANONICAL_EXECUTION_CONFIG_PATH,
            "execution_requires_separate_exact_envelope": True,
            "formal_run_allowed_from_source_commit": False,
            "new_attempt_identity_required": True,
            "max_concurrent_processes": GPU_SLOT_COUNT,
            "max_concurrent_state_processes": GPU_SLOT_COUNT,
            "no_retry": True,
            "no_top_up": True,
            "process_per_state": True,
            "required_gpu_class": "NVIDIA_H200",
            "required_host_class": "Hyper_H200",
            "same_host_container_and_runtime_stack_required": True,
            "source_a_requires_clean_pushed_main": True,
            "state_process_count": STATE_PROCESS_COUNT,
            "wave_0_terminal_barrier_before_wave_1_attempt": True,
            "waves": [list(wave) for wave in EXPECTED_STATE_WAVES],
        },
        "historical_d1_auto_context": {
            "condition_ids": [
                AUTO_FRESH_ENCODE_CONDITION,
                AUTO_FROZEN_ENCODED_CONDITION,
            ],
            "d1_eager_condition_consumed": False,
            "d1_eager_oom_backfilled": False,
            "historical_calls_counted_against_d1b_execution": False,
            "historical_encode_call_count": context[
                "historical_encode_call_count"
            ],
            "historical_generation_call_count": context[
                "historical_generation_call_count"
            ],
            "projection_sha256": context["projection_sha256"],
            "source_protocol_id": D1_DIAGNOSTIC_PROTOCOL_ID,
        },
        "immutable_inputs": {
            "d1_aggregate": aggregate_binding,
            "d1_result_commit": D1_RESULT_COMMIT,
            "d1_summary": summary_binding,
            "parent_d1_aggregate": {
                "path": D1_AGGREGATE_PATH,
                "sha256": D1_AGGREGATE_SHA256,
                "status": D1_RESULT_STATUS,
                "verdict": D1_VERDICT,
            },
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
                "wave_and_gpu_slot_identity",
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
            "restoration_label_count": 0,
            "sealed_test_access_count": 0,
            "teacher_forward_count": 0,
            "training_example_count": 0,
            "tune_state_access_count": 0,
            "utility_or_kl_write_count": 0,
        },
        "preregistered_interpretation": {
            "aggregate_verdict_precedence": list(AGGREGATE_VERDICT_PRECEDENCE),
            "forbidden_ambiguous_verdict_names": [
                "PERSISTENT_GENERATION_INSTABILITY"
            ],
            "operational_gate": {
                "go_verdict": (
                    "PASS_MEMORY_SAFE_SDPA_NUMERICAL_CONTROL_REPEAT_STABILITY"
                ),
                "go_authorizes_only_new_v3_throughput_source_freeze": True,
                "labels_or_training_unlocked": False,
            },
            "state_diagnosis_precedence": [
                dict(rule) for rule in STATE_DIAGNOSIS_PRECEDENCE
            ],
        },
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "scientific_locks": {
            "closed_loop_locked": True,
            "d1_is_not_retried_or_topped_up": True,
            "d1b_does_not_unlock_labels_or_training": True,
            "formal_labels_locked": True,
            "matched_nll_locked": True,
            "predictor_training_locked": True,
            "twelve_state_v3_throughput_required_before_labels": True,
        },
        "source": _source_projection(root),
        "status": STATUS,
    }


def validate_action_stability_source_v2_config(
    config: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("D1b source config must be one mapping")
    expected = build_action_stability_source_v2_config_skeleton(
        repository_root=repository_root
    )
    if dict(config) != expected:
        raise ValueError("D1b source config differs from the live skeleton")
    source = expected["source"]
    if source["missing_reserved_entrypoint_paths"]:
        raise ValueError("D1b source config cannot freeze with missing entrypoints")
    diagnostic = expected["diagnostic"]
    return {
        "config_sha256": sha256_bytes(canonical_pretty_json_bytes(expected)),
        "encode_call_ceiling": diagnostic["encode_call_ceiling"],
        "generation_call_ceiling": diagnostic["generation_call_ceiling"],
        "process_count": diagnostic["process_count"],
        "source_file_count": source["file_count"],
        "source_inventory_sha256": source["inventory_sha256"],
        "state_count": diagnostic["state_count"],
        "status": VALIDATION_STATUS,
        "wave_sizes": diagnostic["wave_sizes"],
    }


def load_action_stability_source_v2_contract(
    *,
    repository_root: str | Path,
    config_path: str | Path = CANONICAL_CONFIG_PATH,
) -> ActionStabilitySourceContractV2:
    root = Path(repository_root).resolve()
    supplied = Path(config_path)
    path = supplied if supplied.is_absolute() else root / supplied
    expected_path = (root / CANONICAL_CONFIG_PATH).resolve()
    if path.resolve() != expected_path:
        raise ValueError(f"D1b source config must be canonical: {expected_path}")
    payload = _repository_file(
        root,
        CANONICAL_CONFIG_PATH,
        label="D1b source config",
    ).read_bytes()
    config = _strict_json_object(payload, label="D1b source config")
    if payload != canonical_pretty_json_bytes(config):
        raise ValueError("D1b source config must be canonical pretty JSON")
    validate_action_stability_source_v2_config(config, repository_root=root)
    return ActionStabilitySourceContractV2(
        data=config,
        repository_root=root,
        config_sha256=sha256_bytes(payload),
    )


__all__ = [
    "AGGREGATE_VERDICT_PRECEDENCE",
    "ActionStabilitySourceContractV2",
    "CANONICAL_CONFIG_PATH",
    "CANONICAL_EXECUTION_CONFIG_PATH",
    "CLI_PATH",
    "CONTRACT_PATH",
    "D1_AGGREGATE_PATH",
    "D1_AGGREGATE_SHA256",
    "D1_RESULT_COMMIT",
    "D1_SUMMARY_PATH",
    "D1_SUMMARY_SHA256",
    "ENCODE_CALL_CEILING",
    "GENERATION_CALL_CEILING",
    "GPU_SLOT_COUNT",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "SOURCE_ENTRYPOINT_CANDIDATES",
    "STATE_DIAGNOSIS_PRECEDENCE",
    "STATE_IDS",
    "STATE_PROCESS_COUNT",
    "STATUS",
    "VALIDATION_STATUS",
    "WAVE_STATE_INDICES",
    "build_action_stability_source_v2_config_skeleton",
    "canonical_json_bytes",
    "canonical_pretty_json_bytes",
    "load_action_stability_source_v2_contract",
    "sha256_bytes",
    "validate_action_stability_source_v2_config",
]
