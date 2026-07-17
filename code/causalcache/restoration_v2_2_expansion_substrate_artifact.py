"""Validate and package label-expansion substrate raw evidence."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import subprocess
import tarfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.restoration_v2_2_expansion_substrate_contract import (
    CANONICAL_COMPLETION_PATH,
    CANONICAL_CONFIG_PATH,
    EXPECTED_STACK,
    EXPECTED_SUBSTRATE_COUNTS,
    PROTOCOL_ID,
)
from causalcache.restoration_v2_2_eager_artifact import (
    _validate_measurement_kernel_record as _validate_eager_measurement_kernel,
)


SCHEMA_VERSION = "1.0.0"
ARTIFACT_PROTOCOL_ID = (
    "causalcache_restoration_v2_2_label_expansion_substrate_artifact_v1"
)
ATTEMPT_ID = "restoration-v2-2-label-expansion-substrate-v1"
ARCHIVE_MEMBER_PREFIX = ATTEMPT_ID
ARCHIVE_FORMAT = "ustar"
CANONICAL_OUTPUT_DIR = Path(
    "/data/experiments/causalcache/restoration-v2-2-label-expansion-substrate-v1"
)
CANONICAL_LEDGER_PATH = Path(
    "/data/experiments/causalcache/"
    ".restoration-v2-2-label-expansion-substrate-v1.attempt.json"
)
CANONICAL_ARCHIVE_PATH = Path(
    "/data/experiments/causalcache/"
    "restoration-v2-2-label-expansion-substrate-v1.tar"
)
CANONICAL_HF_REPO = (
    "gavinlaw/causalcache-restoration-v2-2-label-expansion-substrate-mobile"
)
CANONICAL_HF_TAG = "v2.2-label-expansion-substrate-v1"
CANONICAL_HF_PATH = "raw/v2.2-label-expansion-substrate-v1.tar"
CANONICAL_GIT_ORIGIN_URL = "https://github.com/luojiaxuan/CausalCache.git"

GLOBAL_LEDGER_ARCHIVE_NAME = "global_attempt_ledger.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"
AGGREGATE_FILENAME = "aggregate.json"
EXECUTION_EVIDENCE_FILENAME = "execution_evidence.json"
RUNTIME_BARRIER_RELEASE_FILENAME = "runtime_barrier_release.json"
RUNTIME_BARRIER_ABORT_FILENAME = "runtime_barrier_abort.json"
CANARY_RELEASE_FILENAME = "processor_canary_release.json"
CANARY_ABORT_FILENAME = "processor_canary_abort.json"
WORKER_DIRECTORY = "workers"
WORKER_SIBLING_LEDGER_DIRECTORY = "worker_sibling_ledgers"
WORKER_LEDGER_FILENAME = "worker_attempt_ledger.json"
WORKER_RUNTIME_FILENAME = "runtime_identity.json"
WORKER_TERMINAL_FILENAME = "terminal.json"
WORKER_CANARY_FILENAME = "processor_canary.json"
STATE_DIRECTORY = "states"
ATTEMPT_DIRECTORY = "attempts"

EXPECTED_STATE_COUNT = 192
EXPECTED_TRAJECTORY_COUNT = 64
PLANNED_COUNTS = dict(EXPECTED_SUBSTRATE_COUNTS)
PER_STATE_PLANNED_COUNTS = {
    "generation_call_count": 2,
    "teacher_forward_count": 3,
    "kl_measurement_count": 2,
}
PASS_OUTCOME = "PASS_V2_2_LABEL_EXPANSION_SUBSTRATE_V1"
NO_GO_OUTCOME = "NO_GO_V2_2_LABEL_EXPANSION_SUBSTRATE_V1"
INVALID_OUTCOME = "INVALID_V2_2_LABEL_EXPANSION_SUBSTRATE_V1"
TERMINAL_OUTCOMES = {PASS_OUTCOME, NO_GO_OUTCOME, INVALID_OUTCOME}
STATE_OUTCOME_VALID = "VALID_V2_2_LABEL_EXPANSION_SUBSTRATE_STATE"
STATE_OUTCOME_FAILED = "FAILED_V2_2_LABEL_EXPANSION_SUBSTRATE_STATE"
WORKER_OUTCOME_COMPLETE = "COMPLETED_V2_2_LABEL_EXPANSION_WORKER_SHARD"
WORKER_OUTCOME_INVALID = "INVALID_V2_2_LABEL_EXPANSION_WORKER_SHARD"

SOURCE_PATHS_REQUIRED_IN_RUN_MANIFEST = {
    "code/causalcache/restoration_v2_2_expansion_substrate_artifact.py",
    "code/scripts/run_restoration_v2_2_expansion_substrate.py",
    "code/scripts/manage_restoration_v2_2_expansion_substrate_artifact.py",
}
LOG_PATHS = {
    "preflight": "logs/preflight.log",
    "execution": "logs/execution.log",
    "utilization_monitor": "logs/gpu_utilization_monitor.log",
    "utilization_monitor_ready": "logs/gpu_utilization_monitor.ready.json",
    "utilization_monitor_stop_request": (
        "logs/gpu_utilization_monitor.stop-request.json"
    ),
}
MONITOR_SUMMARY_PATH = "monitor_summary.json"
MONITOR_STOP_REQUEST_STATUS = "EXPANSION_SUBSTRATE_MONITOR_STOP_REQUEST"
MONITOR_SUMMARY_STATUS = "GPU_UTILIZATION_MONITOR_SUMMARY"
MONITOR_STOP_REQUEST_KEYS = {
    "schema_version",
    "status",
    "run_contract_sha256",
    "requested_at_utc",
}
MONITOR_SUMMARY_KEYS = {
    "schema_version",
    "status",
    "run_contract_sha256",
    "container_name_prefix",
    "minimum_gpu_utilization_percent",
    "started_before_first_generation",
    "started_at_utc",
    "stopped_at_utc",
    "sample_count",
    "low_utilization_incident_count",
    "visible_gpu_count",
    "visible_gpu_uuids",
    "stop_request_sha256",
}
PROCESSOR_IDENTITY_KEYS = {
    "processor_path",
    "rendered_prompt_sha256",
    "input_ids_sha256",
    "attention_mask_sha256",
    "pixel_values_sha256",
    "image_grid_thw_sha256",
    "input_ids_shape",
    "input_ids_dtype",
    "attention_mask_shape",
    "attention_mask_dtype",
    "pixel_values_shape",
    "pixel_values_dtype",
    "image_grid_thw_shape",
    "image_grid_thw_dtype",
    "image_count",
    "policy_forward_executed",
    "processor_request_sha256",
}
PROCESSOR_IDENTITY_FIELDS = PROCESSOR_IDENTITY_KEYS
RUNNER_FREEZE_SCHEMA_VERSION = "1.0.0"
RUNNER_FREEZE_PROTOCOL_ID = (
    "causalcache_restoration_v2_2_label_expansion_substrate_runner_freeze_v1"
)
CANONICAL_RUNNER_FREEZE_PATH = (
    "code/configs/causalcache_restoration_v2_2_expansion_substrate_runner_v1.json"
)
FROZEN_BASE_CONFIG_SHA256 = (
    "42144f33e2473c787b3648c0ace18b22902fa3376615732aef8fd3aa5780a6e5"
)
FORBIDDEN_COUNT_KEYS = {
    "retry_count",
    "top_up_count",
    "resume_count",
    "replacement_count",
    "confirm_state_access_count",
    "confirm_prompt_or_image_access_count",
    "expert_action_target_read_count",
    "expert_action_semantic_consumption_count",
    "expert_action_backfill_count",
    "prior_policy_output_import_count",
    "restoration_coalition_construction_count",
    "restoration_candidate_prompt_count",
    "restoration_label_count",
    "exact_subset_search_count",
    "gate_training_example_count",
    "gate_model_forward_count",
    "matched_nll_evaluation_count",
    "closed_loop_episode_count",
}

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FAILURE_CATEGORY_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{0,63}")


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    device: str
    index_parity: int
    state_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "device": self.device,
            "index_parity": self.index_parity,
            "state_indices": list(self.state_indices),
        }

    def envelope_identity(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "device": self.device,
            "index_parity": self.index_parity,
        }


@dataclass(frozen=True)
class ExpansionSubstrateEvidence:
    files: Mapping[str, bytes]
    run_contract: Mapping[str, Any]
    source_git_commit: str
    execution_git_commit: str
    run_contract_sha256: str
    config_sha256: str
    completion_manifest_sha256: str
    derived_immutable_revision: str
    outcome: str
    aggregate_sha256: str
    fixed_state_denominator: int
    attempted_state_count: int
    completed_state_count: int
    planned_counts: Mapping[str, int]
    actual_counts: Mapping[str, int]
    valid_state_count: int
    failed_state_count: int
    failure_category_counts: Mapping[str, int]
    gate_summary: Mapping[str, Any] | None
    inventory: tuple[Mapping[str, Any], ...]
    tree_inventory_sha256: str
    runtime_identity_sha256: str
    worker_identity_sha256: str
    log_inventory_sha256: str
    monitor_summary_sha256: str | None


def expected_worker_specs() -> tuple[WorkerSpec, WorkerSpec]:
    return (
        WorkerSpec("even", "cuda:0", 0, tuple(range(0, EXPECTED_STATE_COUNT, 2))),
        WorkerSpec("odd", "cuda:1", 1, tuple(range(1, EXPECTED_STATE_COUNT, 2))),
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def strict_json_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    value = json.loads(
        payload,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    if pretty_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical pretty JSON")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys drifted")


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _git_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or GIT_SHA_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a full lowercase Git SHA")
    return value


def _safe_relative_path(value: Any, name: str = "path") -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{name} must use canonical relative POSIX syntax")
    return value


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp") from error
    return parsed


def _file_witness(value: Any, name: str) -> Mapping[str, Any]:
    record = _mapping(value, name)
    _exact_keys(record, {"path", "sha256", "size_bytes"}, name)
    _safe_relative_path(record.get("path"), f"{name}.path")
    _sha256(record.get("sha256"), f"{name}.sha256")
    if type(record.get("size_bytes")) is not int or record["size_bytes"] <= 0:
        raise ValueError(f"{name}.size_bytes must be positive")
    return record


def _inventory(files: Mapping[str, bytes]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "path": name,
            "sha256": sha256_bytes(files[name]),
            "size_bytes": len(files[name]),
        }
        for name in sorted(files)
    )


def _validate_worker_topology(value: Any) -> tuple[WorkerSpec, WorkerSpec]:
    expected = expected_worker_specs()
    topology = _mapping(value, "worker topology")
    if topology != {
        "worker_count": 2,
        "gpu_model": "NVIDIA H200",
        "one_process_per_device": True,
        "workers": [spec.to_dict() for spec in expected],
        "cross_worker_state_stealing_allowed": False,
        "worker_failure_invalidates_entire_attempt": True,
        "worker_outputs_must_be_disjoint": True,
        "worker_union_must_equal_fixed_denominator": True,
    }:
        raise ValueError("worker topology drifted")
    return expected


def _validate_states(value: Any, *, expected_sha256: str) -> tuple[dict[str, Any], ...]:
    records = _sequence(value, "run-contract states")
    if len(records) != EXPECTED_STATE_COUNT:
        raise ValueError("run contract must contain exactly 192 states")
    expected_keys = {
        "state_index",
        "role",
        "source_id",
        "state_id",
        "decision_step_id",
        "history_event_step_ids",
        "candidate_event_step_ids",
        "current_equivalent_event_step_id",
    }
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(records):
        state = dict(_mapping(raw, f"state {index}"))
        _exact_keys(state, expected_keys, f"state {index}")
        step = state.get("decision_step_id")
        source_id = state.get("source_id")
        state_id = state.get("state_id")
        if (
            state.get("state_index") != index
            or step not in {4, 5, 6}
            or not isinstance(source_id, str)
            or not source_id
            or state_id != f"{source_id}:decision_step:{step:03d}"
            or state_id in seen_ids
            or state.get("history_event_step_ids") != list(range(1, step))
            or state.get("candidate_event_step_ids") != list(range(1, step - 1))
            or state.get("current_equivalent_event_step_id") != step - 1
            or state.get("role")
            not in {"gate_train_expansion", "gate_development_expansion"}
        ):
            raise ValueError("state projection identity or geometry drifted")
        seen_ids.add(state_id)
        normalized.append(state)
    if sha256_bytes(canonical_json_bytes(normalized)) != expected_sha256:
        raise ValueError("state projection SHA256 differs from frozen config")
    return tuple(normalized)


def _repository_file(root: Path, relative: str) -> Path:
    canonical = _safe_relative_path(relative, "repository path")
    path = root.joinpath(*PurePosixPath(canonical).parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required repository file is missing: {relative}")
    return path


def _repository_identity(root: Path, relative: str) -> dict[str, Any]:
    path = _repository_file(root, relative)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _git_blob(root: Path, revision: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(f"{relative} is not committed at runner source revision")
    return result.stdout


def _runner_interface_contract() -> dict[str, Any]:
    return {
        "runner_module": "scripts.run_restoration_v2_2_expansion_substrate",
        "manager_module": (
            "scripts.manage_restoration_v2_2_expansion_substrate_artifact"
        ),
        "required_cli_options": [
            "--repository-root",
            "--contract",
            "--runner-freeze",
            "--derived-artifact-root",
            "--derived-hf-repo",
            "--derived-hf-tag",
            "--derived-hf-revision",
            "--ocr-backend-config",
            "--snapshot-manifest",
            "--model-dir",
            "--host-alias",
            "--host-hostname",
            "--container-id",
            "--container-image-digest",
            "--output-dir",
            "--global-ledger",
            "--preflight-log",
            "--utilization-monitor-log",
            "--monitor-ready-file",
            "--monitor-summary",
        ],
        "monitor_sidecar": {
            "subcommand": "monitor-sidecar",
            "required_cli_options": [
                "--ready-file",
                "--log-file",
                "--summary-file",
            ],
            "stop_request_path_derivation": "<summary-file>.stop-request.json",
            "visible_gpu_count": 2,
            "sample_command": [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            "sample_interval_seconds": 1.0,
            "ready_before_first_generation_required": True,
            "observational_only": True,
            "scientific_gate_inputs": [],
        },
        "barriers": {
            "runtime_release_filename": RUNTIME_BARRIER_RELEASE_FILENAME,
            "runtime_abort_filename": RUNTIME_BARRIER_ABORT_FILENAME,
            "processor_canary_release_filename": CANARY_RELEASE_FILENAME,
            "processor_canary_abort_filename": CANARY_ABORT_FILENAME,
            "required_order": (
                "monitor_ready<=worker_runtime_created<=runtime_release"
                "<worker_canary_completed"
                "<=canary_release<first_state_attempt_marker"
            ),
            "runtime_release_fields": [
                "schema_version",
                "protocol_id",
                "status",
                "run_contract_sha256",
                "worker_runtime_sha256",
                "released_at_utc",
            ],
            "canary_release_fields": [
                "schema_version",
                "protocol_id",
                "status",
                "run_contract_sha256",
                "worker_report_sha256",
                "state_count",
                "included_history_mutation_count",
                "excluded_state_count",
                "excluded_event_mutation_count",
                "released_at_utc",
            ],
        },
        "measurement_kernel": {
            "base_protocol_id": "causalcache_restoration_v2_1_full_45_substrate",
            "base_schema_version": "1.0.0",
            "expansion_outer_fields": [
                "slice_witness",
                "request_manifest",
                "processor_canary",
            ],
            "per_state_planned_counts": PER_STATE_PLANNED_COUNTS,
            "deep_eager_validator_required": True,
        },
        "raw_artifact": {
            "schema_version": SCHEMA_VERSION,
            "artifact_protocol_id": ARTIFACT_PROTOCOL_ID,
            "archive_format": ARCHIVE_FORMAT,
            "archive_member_prefix": ARCHIVE_MEMBER_PREFIX,
            "hf_repo": CANONICAL_HF_REPO,
            "hf_tag": CANONICAL_HF_TAG,
            "hf_path": CANONICAL_HF_PATH,
        },
        "processor_identity_fields": sorted(PROCESSOR_IDENTITY_FIELDS),
    }


def validate_runner_freeze_data(
    value: Any,
    *,
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    freeze = _mapping(value, "runner freeze")
    _exact_keys(
        freeze,
        {
            "schema_version",
            "protocol_id",
            "status",
            "runner_source_git_commit",
            "base_config",
            "derived_completion",
            "source_inventory",
            "source_inventory_sha256",
            "interfaces",
            "authorization",
        },
        "runner freeze",
    )
    runner_commit = _git_sha(
        freeze.get("runner_source_git_commit"), "runner source commit"
    )
    base_witness = _mapping(freeze.get("base_config"), "runner-freeze base config")
    _exact_keys(
        base_witness,
        {"path", "sha256", "size_bytes"},
        "runner-freeze base config",
    )
    base_payload = pretty_json_bytes(base_config)
    if (
        freeze.get("schema_version") != RUNNER_FREEZE_SCHEMA_VERSION
        or freeze.get("protocol_id") != RUNNER_FREEZE_PROTOCOL_ID
        or freeze.get("status")
        != "FROZEN_COMMITTED_PUSHED_EXPANSION_SUBSTRATE_RUNNER_SOURCE"
        or base_witness.get("path") != CANONICAL_CONFIG_PATH
        or base_witness.get("sha256") != FROZEN_BASE_CONFIG_SHA256
        or base_witness.get("sha256") != sha256_bytes(base_payload)
        or base_witness.get("size_bytes") != len(base_payload)
    ):
        raise ValueError("runner-freeze base config identity drifted")
    completion = _mapping(
        freeze.get("derived_completion"), "runner-freeze derived completion"
    )
    _exact_keys(
        completion,
        {"path", "sha256", "size_bytes"},
        "runner-freeze derived completion",
    )
    frozen_completion = _mapping(
        _mapping(base_config.get("immutable_inputs"), "immutable inputs").get(
            "derived_completion"
        ),
        "frozen completion witness",
    )
    if (
        completion.get("path") != CANONICAL_COMPLETION_PATH
        or completion.get("path") != frozen_completion.get("path")
        or completion.get("sha256") != frozen_completion.get("sha256")
        or type(completion.get("size_bytes")) is not int
        or completion["size_bytes"] <= 0
    ):
        raise ValueError("runner-freeze completion identity drifted")
    parent_files = _sequence(
        _mapping(base_config.get("scientific_source_lock"), "source lock").get(
            "source_files"
        ),
        "parent source files",
    )
    if len(parent_files) != 63:
        raise ValueError("parent source inventory must contain exactly 63 files")
    parent_by_path: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(parent_files):
        record = dict(_file_witness(raw, f"parent source file {index}"))
        if record["path"] in parent_by_path:
            raise ValueError("parent source inventory contains duplicate paths")
        parent_by_path[record["path"]] = record
    reserved = list(
        _sequence(
            _mapping(base_config.get("scientific_source_lock"), "source lock").get(
                "reserved_execution_source_paths"
            ),
            "reserved execution paths",
        )
    )
    if reserved != [
        "code/causalcache/restoration_v2_2_expansion_substrate_artifact.py",
        "code/scripts/run_restoration_v2_2_expansion_substrate.py",
        "code/scripts/manage_restoration_v2_2_expansion_substrate_artifact.py",
    ]:
        raise ValueError("reserved execution path inventory drifted")
    inventory = _sequence(freeze.get("source_inventory"), "runner source inventory")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(inventory):
        normalized.append(dict(_file_witness(raw, f"runner source file {index}")))
    expected_inventory_paths = {*parent_by_path, CANONICAL_CONFIG_PATH, *reserved}
    if (
        len(normalized) != 67
        or [record["path"] for record in normalized]
        != sorted(expected_inventory_paths)
        or any(
            record != parent_by_path[record["path"]]
            for record in normalized
            if record["path"] in parent_by_path
        )
    ):
        raise ValueError("runner source inventory is not exact parent63+base1+reserved3")
    inventory_sha = sha256_bytes(canonical_json_bytes(normalized))
    if freeze.get("source_inventory_sha256") != inventory_sha:
        raise ValueError("runner source inventory SHA256 drifted")
    if next(
        record for record in normalized if record["path"] == CANONICAL_CONFIG_PATH
    ) != dict(base_witness):
        raise ValueError("runner inventory base config differs from separate witness")
    if freeze.get("interfaces") != _runner_interface_contract():
        raise ValueError("runner CLI/barrier/kernel/artifact interface drifted")
    if freeze.get("authorization") != {
        "runner_source_commit_must_be_committed": True,
        "runner_source_commit_must_be_ancestor_of_execution_head": True,
        "execution_head_and_origin_main_must_match": True,
        "runner_freeze_file_must_be_committed_at_execution_head": True,
        "source_blobs_must_equal_runner_source_commit": True,
        "source_only_freeze_authorizes_gpu_execution": False,
        "formal_execution_requires_runtime_authorization_validation": True,
    }:
        raise ValueError("runner-freeze authorization declarations drifted")
    return {
        "runner_git_commit": runner_commit,
        "base_config_sha256": str(base_witness["sha256"]),
        "source_inventory": normalized,
        "source_inventory_sha256": inventory_sha,
    }


def build_runner_freeze(
    *,
    repository_root: str | Path,
    runner_source_git_commit: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    _git_sha(runner_source_git_commit, "runner source commit")
    base_path = _repository_file(root, CANONICAL_CONFIG_PATH)
    base_payload = base_path.read_bytes()
    base_config = strict_json_object_bytes(base_payload, label="frozen base config")
    if sha256_bytes(base_payload) != FROZEN_BASE_CONFIG_SHA256:
        raise ValueError("canonical base config SHA256 drifted")
    source_lock = _mapping(base_config["scientific_source_lock"], "source lock")
    parent_records = [
        dict(_file_witness(raw, f"parent source file {index}"))
        for index, raw in enumerate(_sequence(source_lock["source_files"], "source files"))
    ]
    reserved = list(_sequence(source_lock["reserved_execution_source_paths"], "reserved paths"))
    inventory: list[dict[str, Any]] = []
    for parent in parent_records:
        current = _repository_identity(root, parent["path"])
        if current != parent:
            raise ValueError(f"parent scientific source drifted: {parent['path']}")
        inventory.append(current)
    inventory.append(_repository_identity(root, CANONICAL_CONFIG_PATH))
    for relative in reserved:
        inventory.append(_repository_identity(root, str(relative)))
    inventory.sort(key=lambda record: record["path"])
    if len(inventory) != 67 or len({record["path"] for record in inventory}) != 67:
        raise ValueError("runner source inventory must contain 67 distinct files")
    for record in inventory:
        if _git_blob(root, runner_source_git_commit, record["path"]) != _repository_file(
            root, record["path"]
        ).read_bytes():
            raise ValueError(f"runner source blob drifted: {record['path']}")
    completion = _repository_identity(root, CANONICAL_COMPLETION_PATH)
    frozen_completion = _mapping(
        _mapping(base_config["immutable_inputs"], "immutable inputs")[
            "derived_completion"
        ],
        "frozen completion",
    )
    if completion["sha256"] != frozen_completion.get("sha256"):
        raise ValueError("derived completion differs from frozen base config")
    result = {
        "schema_version": RUNNER_FREEZE_SCHEMA_VERSION,
        "protocol_id": RUNNER_FREEZE_PROTOCOL_ID,
        "status": "FROZEN_COMMITTED_PUSHED_EXPANSION_SUBSTRATE_RUNNER_SOURCE",
        "runner_source_git_commit": runner_source_git_commit,
        "base_config": _repository_identity(root, CANONICAL_CONFIG_PATH),
        "derived_completion": completion,
        "source_inventory": inventory,
        "source_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "interfaces": _runner_interface_contract(),
        "authorization": {
            "runner_source_commit_must_be_committed": True,
            "runner_source_commit_must_be_ancestor_of_execution_head": True,
            "execution_head_and_origin_main_must_match": True,
            "runner_freeze_file_must_be_committed_at_execution_head": True,
            "source_blobs_must_equal_runner_source_commit": True,
            "source_only_freeze_authorizes_gpu_execution": False,
            "formal_execution_requires_runtime_authorization_validation": True,
        },
    }
    validate_runner_freeze_data(result, base_config=base_config)
    return result


def materialize_runner_freeze(
    *,
    repository_root: str | Path,
    runner_source_git_commit: str,
    output_path: str | Path = CANONICAL_RUNNER_FREEZE_PATH,
    require_clean_pushed_source: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if require_clean_pushed_source:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        remote = subprocess.run(
            ["git", "ls-remote", "--exit-code", "origin", "refs/heads/main"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        origin_url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        advertised = remote.stdout.split()
        if (
            head != runner_source_git_commit
            or status
            or branch != "main"
            or origin_url != CANONICAL_GIT_ORIGIN_URL
            or remote.returncode != 0
            or not advertised
            or advertised[0] != runner_source_git_commit
        ):
            raise ValueError(
                "runner-freeze materialization requires clean pushed source HEAD"
            )
    output = Path(output_path)
    destination = output if output.is_absolute() else root / output
    if destination.resolve() != (root / CANONICAL_RUNNER_FREEZE_PATH).resolve():
        raise ValueError("runner freeze output path is not canonical")
    if destination.exists():
        raise FileExistsError("runner freeze already exists; overwrite is forbidden")
    value = build_runner_freeze(
        repository_root=root,
        runner_source_git_commit=runner_source_git_commit,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as target:
        target.write(pretty_json_bytes(value))
        target.flush()
        os.fsync(target.fileno())
    return {
        "status": "MATERIALIZED_EXPANSION_SUBSTRATE_RUNNER_FREEZE",
        "path": CANONICAL_RUNNER_FREEZE_PATH,
        "sha256": sha256_file(destination),
        "runner_git_commit": runner_source_git_commit,
        "source_inventory_count": 67,
        "policy_or_gpu_execution_authorized": False,
    }


def load_and_validate_runner_freeze(
    freeze_path: str | Path,
    *,
    repository_root: str | Path,
    require_committed_pushed_main: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    supplied = Path(freeze_path)
    candidate = supplied if supplied.is_absolute() else root / supplied
    if candidate.is_symlink():
        raise ValueError("runner freeze may not be a symlink")
    actual = candidate.resolve()
    canonical = (root / CANONICAL_RUNNER_FREEZE_PATH).resolve()
    if actual != canonical or actual.is_symlink() or not actual.is_file():
        raise ValueError(f"runner freeze must be {CANONICAL_RUNNER_FREEZE_PATH}")
    payload = actual.read_bytes()
    freeze = strict_json_object_bytes(payload, label="runner freeze")
    base_payload = _repository_file(root, CANONICAL_CONFIG_PATH).read_bytes()
    base_config = strict_json_object_bytes(base_payload, label="base config")
    validation = validate_runner_freeze_data(freeze, base_config=base_config)
    rebuilt = build_runner_freeze(
        repository_root=root,
        runner_source_git_commit=validation["runner_git_commit"],
    )
    if freeze != rebuilt:
        raise ValueError("runner freeze differs from deterministic rebuild")
    if require_committed_pushed_main:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", validation["runner_git_commit"], head],
            cwd=root,
            check=False,
        )
        if ancestor.returncode != 0:
            raise ValueError("runner source commit is not an ancestor of execution HEAD")
        if _git_blob(root, head, CANONICAL_RUNNER_FREEZE_PATH) != payload:
            raise ValueError("runner freeze is not committed at execution HEAD")
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise ValueError("formal runner authorization requires a clean checkout")
        remote = subprocess.run(
            ["git", "ls-remote", "--exit-code", "origin", "refs/heads/main"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        origin_url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        advertised = remote.stdout.split()
        if (
            remote.returncode != 0
            or not advertised
            or advertised[0] != head
            or branch != "main"
            or origin_url != CANONICAL_GIT_ORIGIN_URL
        ):
            raise ValueError("execution HEAD is not pushed canonical origin/main")
        validation["execution_head"] = head
        validation["execution_branch"] = branch
        validation["origin_url"] = origin_url
        validation["origin_main_equals_execution_head"] = True
    validation.update(
        {
            "runner_freeze_sha256": sha256_bytes(payload),
            "runner_freeze_path": CANONICAL_RUNNER_FREEZE_PATH,
            "runner_freeze_committed_pushed_validation_required": (
                require_committed_pushed_main
            ),
        }
    )
    return freeze, validation


def validate_execution_run_contract(
    value: Any,
    *,
    require_canonical_attempt_identity: bool,
) -> tuple[dict[str, Any], ...]:
    contract = _mapping(value, "execution run contract")
    _exact_keys(
        contract,
        {
            "source",
            "frozen_config",
            "runner_freeze",
            "derived_completion",
            "derived_artifact",
            "canonical_inputs",
            "runtime_requirements",
            "worker_topology",
            "states",
            "schedule",
            "attempt_identity",
            "execution_argv",
            "logs",
            "monitor",
        },
        "execution run contract",
    )
    source = _mapping(contract["source"], "execution source")
    _exact_keys(
        source,
        {
            "runner_source_git_commit",
            "execution_git_commit",
            "branch",
            "origin_url",
            "clean_checkout",
            "head_equals_origin_main",
            "inventory",
        },
        "execution source",
    )
    runner_source_commit = _git_sha(
        source.get("runner_source_git_commit"), "runner source commit"
    )
    execution_commit = _git_sha(
        source.get("execution_git_commit"), "execution Git commit"
    )
    if source.get("clean_checkout") is not True or source.get(
        "head_equals_origin_main"
    ) is not True or source.get("branch") != "main" or source.get(
        "origin_url"
    ) != CANONICAL_GIT_ORIGIN_URL:
        raise ValueError("execution source must be clean canonical main")
    if execution_commit == runner_source_commit:
        raise ValueError("execution commit must include the later runner-freeze commit")
    inventory = _sequence(source.get("inventory"), "execution source inventory")
    normalized_inventory: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(inventory):
        record = dict(_file_witness(raw, f"source inventory {index}"))
        if record["path"] in seen_paths:
            raise ValueError("execution source inventory contains duplicate paths")
        seen_paths.add(record["path"])
        normalized_inventory.append(record)
    if [record["path"] for record in normalized_inventory] != sorted(seen_paths):
        raise ValueError("execution source inventory must be sorted")
    if not SOURCE_PATHS_REQUIRED_IN_RUN_MANIFEST.issubset(seen_paths):
        raise ValueError("execution source inventory lacks a required source/config file")

    frozen = _mapping(contract["frozen_config"], "frozen config")
    _exact_keys(frozen, {"path", "sha256", "size_bytes", "data"}, "frozen config")
    if frozen.get("path") != CANONICAL_CONFIG_PATH:
        raise ValueError("frozen config path drifted")
    frozen_bytes = pretty_json_bytes(_mapping(frozen.get("data"), "frozen config data"))
    if (
        frozen.get("sha256") != sha256_bytes(frozen_bytes)
        or frozen.get("size_bytes") != len(frozen_bytes)
        or frozen.get("sha256") != FROZEN_BASE_CONFIG_SHA256
    ):
        raise ValueError("frozen config hash or size drifted")
    frozen_data = _mapping(frozen["data"], "frozen config data")
    if frozen_data.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("frozen config protocol drifted")
    if _mapping(frozen_data.get("data_projection"), "frozen data projection").get(
        "fixed_state_denominator"
    ) != EXPECTED_STATE_COUNT:
        raise ValueError("frozen state denominator drifted")
    frozen_counts = _mapping(
        _mapping(frozen_data.get("substrate_schedule"), "frozen schedule").get(
            "planned_and_maximum_counts"
        ),
        "frozen planned counts",
    )
    if dict(frozen_counts) != PLANNED_COUNTS:
        raise ValueError("frozen substrate counts drifted")

    runner_freeze = _mapping(contract["runner_freeze"], "runner freeze")
    _exact_keys(
        runner_freeze,
        {"path", "sha256", "size_bytes", "runner_git_commit", "data"},
        "runner freeze",
    )
    runner_freeze_data = _mapping(
        runner_freeze.get("data"), "runner freeze embedded data"
    )
    runner_freeze_payload = pretty_json_bytes(runner_freeze_data)
    runner_validation = validate_runner_freeze_data(
        runner_freeze_data,
        base_config=frozen_data,
    )
    if (
        runner_freeze.get("path") != CANONICAL_RUNNER_FREEZE_PATH
        or runner_freeze.get("runner_git_commit") != runner_source_commit
        or runner_freeze.get("runner_git_commit")
        != runner_validation["runner_git_commit"]
        or runner_freeze.get("sha256") != sha256_bytes(runner_freeze_payload)
        or runner_freeze.get("size_bytes") != len(runner_freeze_payload)
        or normalized_inventory != runner_validation["source_inventory"]
    ):
        raise ValueError("runner-freeze binding drifted")
    _sha256(runner_freeze.get("sha256"), "runner-freeze SHA256")

    completion = _mapping(contract["derived_completion"], "derived completion")
    _exact_keys(completion, {"path", "sha256", "size_bytes"}, "derived completion")
    if completion.get("path") != CANONICAL_COMPLETION_PATH:
        raise ValueError("derived completion path drifted")
    _sha256(completion.get("sha256"), "derived completion SHA256")
    if type(completion.get("size_bytes")) is not int or completion["size_bytes"] <= 0:
        raise ValueError("derived completion size must be positive")
    frozen_completion = _mapping(
        _mapping(frozen_data.get("immutable_inputs"), "immutable inputs").get(
            "derived_completion"
        ),
        "frozen completion",
    )
    if (
        completion.get("path") != frozen_completion.get("path")
        or completion.get("sha256") != frozen_completion.get("sha256")
    ):
        raise ValueError("derived completion differs from frozen base config")

    derived = _mapping(contract["derived_artifact"], "derived artifact")
    _exact_keys(
        derived,
        {
            "repo",
            "tag",
            "immutable_revision",
            "payload_prefix",
            "artifact_tree_sha256",
            "artifact_file_count",
            "artifact_total_bytes",
        },
        "derived artifact",
    )
    frozen_derived = _mapping(
        _mapping(frozen_data.get("immutable_inputs"), "immutable inputs").get(
            "derived_artifact"
        ),
        "frozen derived artifact",
    )
    for field in derived:
        if derived[field] != frozen_derived.get(field):
            raise ValueError(f"derived artifact binding drifted: {field}")
    _git_sha(derived.get("immutable_revision"), "derived immutable revision")
    _sha256(derived.get("artifact_tree_sha256"), "derived artifact tree")

    canonical_inputs = _mapping(contract["canonical_inputs"], "canonical inputs")
    _exact_keys(
        canonical_inputs,
        {
            "derived_artifact_root",
            "derived_artifact",
            "completion_manifest",
            "ocr_backend_config",
            "ocr_backend_manifest",
            "model_snapshot_manifest",
            "model_snapshot",
            "fresh_immutable_derived_validation",
        },
        "canonical inputs",
    )
    derived_root = canonical_inputs.get("derived_artifact_root")
    if not isinstance(derived_root, str) or not Path(derived_root).is_absolute():
        raise ValueError("derived artifact root must be an absolute runtime path")
    if canonical_inputs.get("derived_artifact") != dict(derived):
        raise ValueError("canonical derived-artifact identity drifted")
    if canonical_inputs.get("completion_manifest") != dict(completion):
        raise ValueError("canonical completion-manifest identity drifted")
    inventory_by_path = {
        record["path"]: record for record in runner_validation["source_inventory"]
    }
    for field, path in (
        ("ocr_backend_config", "code/configs/restoration_v2_ocr_backend.json"),
        ("ocr_backend_manifest", "data/manifests/restoration_v2_ocr_backend.json"),
    ):
        if canonical_inputs.get(field) != inventory_by_path.get(path):
            raise ValueError(f"canonical input identity drifted: {field}")
    snapshot = _mapping(
        canonical_inputs.get("model_snapshot_manifest"), "model snapshot manifest"
    )
    _exact_keys(
        snapshot,
        {"path", "sha256", "size_bytes", "data"},
        "model snapshot manifest",
    )
    snapshot_path = "code/configs/gui_owl_1_5_8b_snapshot.json"
    if (
        {key: snapshot.get(key) for key in ("path", "sha256", "size_bytes")}
        != inventory_by_path.get(snapshot_path)
        or snapshot.get("path") != snapshot_path
    ):
        raise ValueError("model snapshot manifest bytes drifted")
    snapshot_data = _mapping(snapshot["data"], "model snapshot manifest data")
    _exact_keys(
        snapshot_data,
        {"repo", "revision", "files"},
        "model snapshot manifest data",
    )
    if not isinstance(snapshot_data.get("repo"), str) or not snapshot_data["repo"]:
        raise ValueError("model snapshot repository identity drifted")
    _git_sha(snapshot_data.get("revision"), "model snapshot revision")
    snapshot_files = list(_sequence(snapshot_data.get("files"), "model snapshot files"))
    normalized_snapshot_files: list[dict[str, Any]] = []
    for raw_file in snapshot_files:
        record = _mapping(raw_file, "model snapshot file")
        _exact_keys(record, {"path", "size", "sha256"}, "model snapshot file")
        path = _safe_relative_path(record.get("path"), "model snapshot file path")
        if type(record.get("size")) is not int or record["size"] <= 0:
            raise ValueError("model snapshot file size drifted")
        _sha256(record.get("sha256"), "model snapshot file")
        normalized_snapshot_files.append(dict(record, path=path))
    if (
        len({record["path"] for record in normalized_snapshot_files})
        != len(normalized_snapshot_files)
        or normalized_snapshot_files != snapshot_files
        or not normalized_snapshot_files
    ):
        raise ValueError("model snapshot file inventory drifted")
    model = _mapping(canonical_inputs.get("model_snapshot"), "model snapshot")
    _exact_keys(
        model,
        {
            "local_path",
            "repo",
            "revision",
            "files",
            "file_inventory_sha256",
            "all_files_verified",
        },
        "model snapshot",
    )
    if (
        not isinstance(model.get("local_path"), str)
        or not Path(model["local_path"]).is_absolute()
        or model.get("repo") != snapshot_data.get("repo")
        or model.get("revision") != snapshot_data.get("revision")
        or model.get("files") != snapshot_files
        or model.get("file_inventory_sha256")
        != sha256_bytes(canonical_json_bytes(snapshot_files))
        or model.get("all_files_verified") is not True
    ):
        raise ValueError("full local model snapshot identity drifted")
    fresh = _mapping(
        canonical_inputs.get("fresh_immutable_derived_validation"),
        "fresh immutable derived validation",
    )
    if fresh != {
        "repo": derived["repo"],
        "tag": derived["tag"],
        "immutable_revision": derived["immutable_revision"],
        "artifact_tree_sha256": derived["artifact_tree_sha256"],
        "validation_passed": True,
        "validated_before_runtime_import": True,
    }:
        raise ValueError("fresh immutable derived validation drifted")

    if dict(_mapping(contract["runtime_requirements"], "runtime requirements")) != {
        **EXPECTED_STACK,
        "dtype": "bfloat16",
        "attention_implementation": "eager",
    }:
        raise ValueError("runtime requirements drifted")
    _validate_worker_topology(contract["worker_topology"])
    projection_sha = _mapping(
        frozen_data.get("data_projection"), "frozen data projection"
    ).get("state_projection_sha256")
    _sha256(projection_sha, "frozen state projection")
    states = _validate_states(contract["states"], expected_sha256=projection_sha)

    schedule = _mapping(contract["schedule"], "execution schedule")
    if dict(schedule) != {
        "fixed_state_denominator": EXPECTED_STATE_COUNT,
        "planned_counts": PLANNED_COUNTS,
        "per_state_planned_counts": PER_STATE_PLANNED_COUNTS,
        "automatic_retry_allowed": False,
        "resume_allowed": False,
        "top_up_allowed": False,
        "replacement_allowed": False,
    }:
        raise ValueError("execution schedule drifted")

    attempt = _mapping(contract["attempt_identity"], "attempt identity")
    _exact_keys(
        attempt,
        {
            "attempt_id",
            "output_dir",
            "global_ledger",
            "host_alias",
            "host_hostname",
            "container_id",
            "container_image_digest",
        },
        "attempt identity",
    )
    if attempt.get("attempt_id") != ATTEMPT_ID:
        raise ValueError("attempt ID drifted")
    if require_canonical_attempt_identity and (
        attempt.get("output_dir") != str(CANONICAL_OUTPUT_DIR)
        or attempt.get("global_ledger") != str(CANONICAL_LEDGER_PATH)
    ):
        raise ValueError("attempt root or global ledger is not canonical")
    if not require_canonical_attempt_identity and (
        not isinstance(attempt.get("output_dir"), str)
        or not Path(attempt["output_dir"]).is_absolute()
        or not isinstance(attempt.get("global_ledger"), str)
        or not Path(attempt["global_ledger"]).is_absolute()
    ):
        raise ValueError("attempt root and global ledger must be absolute")
    if (
        not all(
            isinstance(attempt.get(field), str) and attempt[field]
            for field in ("host_alias", "host_hostname", "container_id")
        )
        or attempt.get("container_image_digest")
        != EXPECTED_STACK["container_image_digest"]
    ):
        raise ValueError("attempt host/container identity drifted")

    argv = list(_sequence(contract["execution_argv"], "execution argv"))
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ValueError("execution argv must contain non-empty strings")
    if any(
        token in " ".join(argv).casefold()
        for token in ("hf_token", "hf_key", "authorization:", "bearer ")
    ):
        raise ValueError("execution argv contains secret-bearing content")
    expected_options = _runner_interface_contract()["required_cli_options"]
    repository_root_arg = Path(
        argv[argv.index("--repository-root") + 1]
    ) if "--repository-root" in argv and argv.index("--repository-root") + 1 < len(argv) else None
    expected_script = (
        repository_root_arg
        / "code/scripts/run_restoration_v2_2_expansion_substrate.py"
        if repository_root_arg is not None
        else None
    )
    if (
        len(argv) != 2 + 2 * len(expected_options)
        or argv[2::2] != expected_options
        or not Path(argv[0]).is_absolute()
        or repository_root_arg is None
        or not repository_root_arg.is_absolute()
        or expected_script is None
        or Path(argv[1]) != expected_script
    ):
        raise ValueError("execution argv is not the exact frozen runner invocation")
    option_positions: dict[str, int] = {}
    for option in expected_options:
        positions = [index for index, item in enumerate(argv) if item == option]
        if len(positions) != 1 or positions[0] + 1 >= len(argv):
            raise ValueError(f"execution argv must contain {option} exactly once")
        value_index = positions[0] + 1
        if argv[value_index].startswith("--"):
            raise ValueError(f"execution argv lacks a value for {option}")
        option_positions[option] = value_index
    option_values = {
        option: argv[index] for option, index in option_positions.items()
    }
    exact_values = {
        "--contract": CANONICAL_CONFIG_PATH,
        "--runner-freeze": CANONICAL_RUNNER_FREEZE_PATH,
        "--derived-artifact-root": str(derived_root),
        "--derived-hf-repo": str(derived["repo"]),
        "--derived-hf-tag": str(derived["tag"]),
        "--derived-hf-revision": str(derived["immutable_revision"]),
        "--ocr-backend-config": "code/configs/restoration_v2_ocr_backend.json",
        "--snapshot-manifest": "code/configs/gui_owl_1_5_8b_snapshot.json",
        "--model-dir": str(model["local_path"]),
        "--host-alias": str(attempt["host_alias"]),
        "--host-hostname": str(attempt["host_hostname"]),
        "--container-id": str(attempt["container_id"]),
        "--container-image-digest": str(attempt["container_image_digest"]),
        "--output-dir": str(attempt["output_dir"]),
        "--global-ledger": str(attempt["global_ledger"]),
    }
    for option, expected in exact_values.items():
        actual = option_values[option]
        if option in {
            "--contract",
            "--runner-freeze",
            "--ocr-backend-config",
            "--snapshot-manifest",
        }:
            if not (actual == expected or actual.endswith(f"/{expected}")):
                raise ValueError(f"execution argv canonical path drifted: {option}")
        elif actual != expected:
            raise ValueError(f"execution argv value drifted: {option}")
    for option in (
        "--repository-root",
        "--preflight-log",
        "--utilization-monitor-log",
        "--monitor-ready-file",
        "--monitor-summary",
    ):
        if not Path(option_values[option]).is_absolute():
            raise ValueError(f"execution argv path must be absolute: {option}")
    if dict(_mapping(contract["logs"], "log paths")) != LOG_PATHS:
        raise ValueError("canonical log paths drifted")
    monitor = _mapping(contract["monitor"], "monitor contract")
    if monitor != {
        "enabled": True,
        "minimum_gpu_utilization_percent": 90,
        "container_name_prefix": "sglang-omni-jaxan",
        "log_path": LOG_PATHS["utilization_monitor"],
        "summary_path": MONITOR_SUMMARY_PATH,
        "started_before_first_generation_required": True,
        "low_utilization_requires_immediate_inspection": True,
    }:
        raise ValueError("utilization-monitor contract drifted")
    return states


def _validate_worker_ledger(
    value: Any,
    *,
    spec: WorkerSpec,
    run_contract_sha256: str,
) -> tuple[str, list[int], list[int]]:
    ledger = _mapping(value, f"{spec.worker_id} worker ledger")
    _exact_keys(
        ledger,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "worker",
            "claimed_at_utc",
            "attempted_state_indices",
            "completed_state_indices",
            "retry_count",
            "top_up_count",
        },
        f"{spec.worker_id} worker ledger",
    )
    if (
        ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("protocol_id") != PROTOCOL_ID
        or ledger.get("run_contract_sha256") != run_contract_sha256
        or ledger.get("worker") != spec.to_dict()
        or ledger.get("retry_count") != 0
        or ledger.get("top_up_count") != 0
        or ledger.get("status")
        not in {
            "WORKER_SIBLING_PREBOUND_BEFORE_SPAWN",
            "WORKER_ATTEMPT_CLAIMED_BEFORE_RUNTIME_IMPORT",
            "WORKER_SHARD_RUNNING_NO_RETRY",
            "WORKER_SHARD_COMPLETED",
            "WORKER_SHARD_INVALID",
        }
    ):
        raise ValueError("worker ledger identity/retry/top-up drifted")
    _timestamp(ledger.get("claimed_at_utc"), "worker claim time")
    attempted = list(_sequence(ledger.get("attempted_state_indices"), "attempted indices"))
    completed = list(_sequence(ledger.get("completed_state_indices"), "completed indices"))
    if (
        attempted != list(spec.state_indices[: len(attempted)])
        or completed != list(spec.state_indices[: len(completed)])
        or len(completed) > len(attempted)
    ):
        raise ValueError("worker ledger is not a fixed parity-shard prefix")
    return str(ledger["status"]), attempted, completed


def _validate_global_ledger(
    value: Any,
    *,
    run_contract_sha256: str,
    expected_global_ledger: Path,
) -> tuple[str, int, int]:
    ledger = _mapping(value, "global attempt ledger")
    _exact_keys(
        ledger,
        {
            "schema_version",
            "protocol_id",
            "status",
            "attempt_id",
            "run_contract_sha256",
            "worker_topology",
            "claimed_at_utc",
            "spawn_started_at_utc",
            "completed_state_count",
            "attempted_state_count",
            "planned_counts",
            "actual_counts",
            "worker_high_water",
            "worker_sibling_ledgers",
            "retry_count",
            "top_up_count",
        },
        "global attempt ledger",
    )
    if (
        ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("protocol_id") != PROTOCOL_ID
        or ledger.get("attempt_id") != ATTEMPT_ID
        or ledger.get("run_contract_sha256") != run_contract_sha256
        or ledger.get("planned_counts") != PLANNED_COUNTS
        or ledger.get("retry_count") != 0
        or ledger.get("top_up_count") != 0
        or ledger.get("status")
        not in {
            "GLOBAL_ATTEMPT_CLAIMED",
            "GLOBAL_ATTEMPT_RUNNING",
            "GLOBAL_ATTEMPT_COMPLETED",
            "GLOBAL_ATTEMPT_INVALID",
        }
    ):
        raise ValueError("global ledger identity/planned/retry/top-up drifted")
    _validate_worker_topology(ledger.get("worker_topology"))
    _timestamp(ledger.get("claimed_at_utc"), "global claim time")
    if ledger.get("spawn_started_at_utc") is not None:
        _timestamp(ledger.get("spawn_started_at_utc"), "worker spawn time")
    attempted = ledger.get("attempted_state_count")
    completed = ledger.get("completed_state_count")
    if (
        type(attempted) is not int
        or type(completed) is not int
        or not 0 <= completed <= attempted <= EXPECTED_STATE_COUNT
    ):
        raise ValueError("global ledger state counts drifted")
    actual = _mapping(ledger.get("actual_counts"), "global actual counts")
    if set(actual) != set(PLANNED_COUNTS) or any(
        type(actual[key]) is not int or not 0 <= actual[key] <= PLANNED_COUNTS[key]
        for key in PLANNED_COUNTS
    ):
        raise ValueError("global actual counts drifted")
    high_water = _mapping(ledger.get("worker_high_water"), "worker high-water")
    siblings = _mapping(ledger.get("worker_sibling_ledgers"), "worker sibling paths")
    specs = expected_worker_specs()
    if set(high_water) != {spec.worker_id for spec in specs} or set(siblings) != {
        spec.worker_id for spec in specs
    }:
        raise ValueError("global worker inventory drifted")
    for spec in specs:
        record = _mapping(high_water[spec.worker_id], "worker high-water record")
        _exact_keys(
            record,
            {
                "attempted_state_indices",
                "completed_state_indices",
                "worker_ledger_sha256",
            },
            "worker high-water record",
        )
        for key in ("attempted_state_indices", "completed_state_indices"):
            indices = list(_sequence(record[key], f"worker high-water {key}"))
            if indices != list(spec.state_indices[: len(indices)]):
                raise ValueError("global high-water is not a parity-shard prefix")
        digest = record.get("worker_ledger_sha256")
        if digest is not None:
            _sha256(digest, "worker high-water ledger SHA256")
        expected_sibling = expected_global_ledger.with_name(
            f"{expected_global_ledger.name[:-5]}.{spec.worker_id}.json"
        )
        if siblings.get(spec.worker_id) != str(expected_sibling):
            raise ValueError("worker sibling ledger path drifted")
    return str(ledger.get("status")), completed, attempted


def _validate_runtime_metadata(value: Any, *, spec: WorkerSpec) -> Mapping[str, Any]:
    metadata = _mapping(value, f"{spec.worker_id} runtime metadata")
    required = {
        "device",
        "gpu_name",
        "gpu_uuid",
        "gpu_pci_bus_id",
        "logical_device_index",
        "nvidia_smi_index",
        "container_image_digest",
        "python_version",
        "torch_version",
        "torch_cuda_version",
        "cudnn_version",
        "transformers_version",
        "nvidia_driver_version",
        "dtype",
        "requested_attention_implementation",
        "observed_attention_implementation",
        "seed",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "float32_matmul_precision",
        "strict_cuda_determinism_claimed",
    }
    if not required.issubset(metadata):
        raise ValueError("worker runtime metadata is incomplete")
    if (
        metadata.get("device") != spec.device
        or metadata.get("gpu_name") != EXPECTED_STACK["gpu_name"]
        or metadata.get("logical_device_index") != spec.index_parity
        or type(metadata.get("nvidia_smi_index")) is not int
        or not isinstance(metadata.get("gpu_uuid"), str)
        or not metadata["gpu_uuid"]
        or not isinstance(metadata.get("gpu_pci_bus_id"), str)
        or not metadata["gpu_pci_bus_id"]
        or metadata.get("container_image_digest")
        != EXPECTED_STACK["container_image_digest"]
        or metadata.get("python_version") != EXPECTED_STACK["python_version"]
        or metadata.get("torch_version") != EXPECTED_STACK["torch_version"]
        or metadata.get("torch_cuda_version")
        != EXPECTED_STACK["torch_cuda_version"]
        or metadata.get("cudnn_version") != EXPECTED_STACK["cudnn_version"]
        or metadata.get("transformers_version")
        != EXPECTED_STACK["transformers_version"]
        or metadata.get("nvidia_driver_version")
        != EXPECTED_STACK["nvidia_driver_version"]
        or metadata.get("dtype") != "torch.bfloat16"
        or metadata.get("requested_attention_implementation") != "eager"
        or metadata.get("observed_attention_implementation")
        != {"top": "eager", "text": "eager", "vision": "eager"}
        or metadata.get("seed") != 0
        or metadata.get("cudnn_deterministic") is not True
        or metadata.get("cudnn_benchmark") is not False
        or metadata.get("cuda_matmul_allow_tf32") is not False
        or metadata.get("cudnn_allow_tf32") is not False
        or metadata.get("float32_matmul_precision") != "highest"
        or metadata.get("strict_cuda_determinism_claimed") is not False
    ):
        raise ValueError("worker runtime metadata drifted")
    return metadata


def _validate_runtime_pair(values: Mapping[str, Mapping[str, Any]]) -> None:
    first, second = (values[spec.worker_id] for spec in expected_worker_specs())
    allowed = {
        "device",
        "gpu_uuid",
        "gpu_pci_bus_id",
        "logical_device_index",
        "nvidia_smi_index",
    }
    if {key: value for key, value in first.items() if key not in allowed} != {
        key: value for key, value in second.items() if key not in allowed
    }:
        raise ValueError("worker scientific runtime metadata differs")
    if any(first.get(key) == second.get(key) for key in allowed):
        raise ValueError("workers did not bind distinct device identities")


def _validate_attempt_marker(
    value: Any,
    *,
    spec: WorkerSpec,
    state_index: int,
    run_contract_sha256: str,
) -> datetime:
    marker = _mapping(value, "state attempt marker")
    _exact_keys(
        marker,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "worker",
            "state_index",
            "claimed_at_utc",
            "retry_count",
            "top_up_count",
        },
        "state attempt marker",
    )
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("protocol_id") != PROTOCOL_ID
        or marker.get("status") != "STATE_ATTEMPT_CLAIMED_NO_RETRY_OR_TOP_UP"
        or marker.get("run_contract_sha256") != run_contract_sha256
        or marker.get("worker") != spec.envelope_identity()
        or marker.get("state_index") != state_index
        or marker.get("retry_count") != 0
        or marker.get("top_up_count") != 0
    ):
        raise ValueError("state attempt marker drifted")
    return _timestamp(marker.get("claimed_at_utc"), "state attempt time")


def _validate_worker_file_window(
    *,
    outcome: str,
    attempted: Sequence[int],
    completed: Sequence[int],
    observed_markers: Sequence[int],
    observed_states: Sequence[int],
) -> None:
    if outcome in {PASS_OUTCOME, NO_GO_OUTCOME}:
        if list(observed_markers) != list(attempted) or list(observed_states) != list(
            completed
        ):
            raise ValueError("terminal worker files differ from durable ledger sets")
        return
    if (
        list(observed_markers) != list(attempted[: len(observed_markers)])
        or len(attempted) - len(observed_markers) not in {0, 1}
        or list(completed) != list(observed_states[: len(completed)])
        or len(observed_states) - len(completed) not in {0, 1}
        or any(index not in observed_markers for index in observed_states)
    ):
        raise ValueError("INVALID worker files exceed a bounded durable-write window")


def _validate_count_mapping(
    value: Any,
    *,
    name: str,
    maxima: Mapping[str, int],
) -> dict[str, int]:
    counts = _mapping(value, name)
    if set(counts) != set(maxima):
        raise ValueError(f"{name} keys drifted")
    result: dict[str, int] = {}
    for key, maximum in maxima.items():
        observed = counts[key]
        if type(observed) is not int or not 0 <= observed <= maximum:
            raise ValueError(f"{name}.{key} exceeds the frozen maximum")
        result[key] = observed
    return result


def _failure_category(value: Any, name: str) -> str | None:
    if value is None:
        return None
    failure = _mapping(value, name)
    category = failure.get("category")
    if not isinstance(category, str) or FAILURE_CATEGORY_PATTERN.fullmatch(category) is None:
        raise ValueError(f"{name} has an invalid failure category")
    if not isinstance(failure.get("stage"), str) or not failure["stage"]:
        raise ValueError(f"{name} lacks a failure stage")
    return category


def _validate_processor_identity(
    value: Any,
    *,
    name: str,
    decision_step_id: int,
) -> dict[str, Any]:
    identity = dict(_mapping(value, name))
    _exact_keys(identity, PROCESSOR_IDENTITY_FIELDS, name)
    hash_keys = {
        "rendered_prompt_sha256",
        "input_ids_sha256",
        "attention_mask_sha256",
        "pixel_values_sha256",
        "image_grid_thw_sha256",
        "processor_request_sha256",
    }
    for key in hash_keys:
        _sha256(identity[key], f"{name}.{key}")
    for tensor in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        shape = identity.get(f"{tensor}_shape")
        dtype = identity.get(f"{tensor}_dtype")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            or not isinstance(dtype, str)
            or not dtype
        ):
            raise ValueError(f"{name} {tensor} shape/dtype is invalid")
    if (
        identity.get("image_count") != decision_step_id - 1
        or identity.get("processor_path") != "GUIOwlV22EagerRuntime._encode_exact_batch"
        or identity.get("policy_forward_executed") is not False
    ):
        raise ValueError(f"{name} processor path/image count/no-forward drifted")
    unhashed = dict(identity)
    request_sha = unhashed.pop("processor_request_sha256")
    if request_sha != sha256_bytes(canonical_json_bytes(unhashed)):
        raise ValueError(f"{name} processor-request aggregate hash drifted")
    return identity


def _validate_state_record(
    value: Any,
    *,
    spec: WorkerSpec,
    projection: Mapping[str, Any],
    run_contract_sha256: str,
    runtime_metadata: Mapping[str, Any] | None,
) -> tuple[dict[str, int], str, str | None, dict[str, Any]]:
    record = _mapping(value, "state envelope")
    _exact_keys(
        record,
        {
            "schema_version",
            "protocol_id",
            "run_contract_sha256",
            "worker",
            "state",
            "outcome",
            "planned_operation_counts",
            "actual_operation_counts",
            "measurement_kernel",
        },
        "state envelope",
    )
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("protocol_id") != PROTOCOL_ID
        or record.get("run_contract_sha256") != run_contract_sha256
        or record.get("worker") != spec.envelope_identity()
        or record.get("state") != dict(projection)
        or record.get("planned_operation_counts") != PER_STATE_PLANNED_COUNTS
        or record.get("outcome") not in {STATE_OUTCOME_VALID, STATE_OUTCOME_FAILED}
    ):
        raise ValueError("state envelope identity/planned counts drifted")
    actual = _validate_count_mapping(
        record.get("actual_operation_counts"),
        name="state actual counts",
        maxima=PER_STATE_PLANNED_COUNTS,
    )
    kernel = _mapping(record.get("measurement_kernel"), "measurement kernel")
    _exact_keys(
        kernel,
        {"protocol_id", "schema_version", "record"},
        "measurement kernel wrapper",
    )
    inner = _mapping(kernel.get("record"), "measurement kernel record")
    operations = _mapping(inner.get("operation_counts"), "kernel operation counts")
    for key in PER_STATE_PLANNED_COUNTS:
        if operations.get(key) != actual[key]:
            raise ValueError("state actual counts differ from measurement kernel")
    for key in FORBIDDEN_COUNT_KEYS:
        if operations.get(key, 0) != 0:
            raise ValueError(f"forbidden operation count is nonzero: {key}")
    for key, raw in operations.items():
        if type(raw) is not int or raw < 0:
            raise ValueError("operation counts must be non-negative integers")
    slice_witness = _mapping(inner.get("slice_witness"), "decision-view slice witness")
    request_manifest = _mapping(inner.get("request_manifest"), "request manifest")
    canary = _mapping(inner.get("processor_canary"), "processor canary witness")
    expected_history = projection["history_event_step_ids"]
    if (
        slice_witness.get("included_event_step_ids") != expected_history
        or request_manifest.get("included_event_step_ids") != expected_history
        or slice_witness.get("current_or_future_event_action_exposure_count") != 0
        or request_manifest.get("current_or_future_event_action_exposure_count") != 0
        or canary.get("passed") is not True
    ):
        raise ValueError("state decision-view/canary witness drifted")
    for witness, name in (
        (slice_witness, "slice witness"),
        (request_manifest, "request manifest"),
        (canary, "processor canary"),
    ):
        for key in (
            "included_event_payload_sha256",
            "processor_request_sha256",
            "input_ids_sha256",
        ):
            if key in witness:
                _sha256(witness[key], f"{name}.{key}")
    failure = _failure_category(inner.get("failure"), "state failure")
    if record["outcome"] == STATE_OUTCOME_VALID:
        if actual != PER_STATE_PLANNED_COUNTS or failure is not None:
            raise ValueError("valid state lacks the full operation schedule")
    elif failure is None:
        raise ValueError("failed state lacks a preserved failure record")

    expansion_fields = {"slice_witness", "request_manifest", "processor_canary"}
    base_inner = {key: value for key, value in inner.items() if key not in expansion_fields}
    _validate_eager_measurement_kernel(
        base_inner,
        spec=spec,
        projection=projection,
        run_contract_sha256=run_contract_sha256,
        runtime_metadata=runtime_metadata,
    )
    expected_outer = (
        STATE_OUTCOME_VALID
        if base_inner.get("outcome") == "VALID_V2_1_FULL_45_SUBSTRATE_STATE"
        else STATE_OUTCOME_FAILED
    )
    if record["outcome"] != expected_outer:
        raise ValueError("outer outcome differs from validated measurement kernel")
    distances = _mapping(base_inner.get("distances"), "validated distances")
    state_real_request = _validate_processor_identity(
        canary.get("real_request"),
        name="state processor canary real request",
        decision_step_id=int(projection["decision_step_id"]),
    )
    reduction = {
        "state_id": projection["state_id"],
        "parse_success": base_inner.get("parse_success") is True,
        "finite_logit_distances": base_inner.get("finite_logit_distances") is True,
        "repeat_canonical_action_agreement": (
            base_inner.get("repeat_canonical_action_agreement") is True
        ),
        "repeat_reference_kl": distances.get("repeat_reference_kl"),
        "summary_reference_kl": distances.get("summary_reference_kl"),
        "processor_identity": dict(state_real_request),
    }
    return actual, str(record["outcome"]), failure, reduction


def _validate_canary_report(
    value: Any,
    *,
    spec: WorkerSpec,
    states: Sequence[Mapping[str, Any]],
    run_contract_sha256: str,
) -> tuple[int, int, int, dict[int, dict[str, Any]], datetime]:
    report = _mapping(value, f"{spec.worker_id} canary report")
    _exact_keys(
        report,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "worker",
            "state_indices",
            "records",
            "included_history_mutation_count",
            "excluded_state_count",
            "excluded_event_mutation_count",
            "completed_at_utc",
        },
        f"{spec.worker_id} canary report",
    )
    indices = list(_sequence(report.get("state_indices"), "canary state indices"))
    records = list(_sequence(report.get("records"), "canary records"))
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("status") != "WORKER_PROCESSOR_CANARY_COMPLETE"
        or report.get("run_contract_sha256") != run_contract_sha256
        or report.get("worker") != spec.envelope_identity()
        or indices != list(spec.state_indices)
        or len(records) != len(indices)
    ):
        raise ValueError("worker canary identity or state inventory drifted")
    included_count = 0
    excluded_states = 0
    excluded_mutations = 0
    real_requests: dict[int, dict[str, Any]] = {}
    for expected_index, raw in zip(indices, records, strict=True):
        record = _mapping(raw, "processor canary record")
        _exact_keys(
            record,
            {
                "state_index",
                "real_request",
                "included_history_mutation",
                "excluded_event_mutations",
            },
            "processor canary record",
        )
        if record.get("state_index") != expected_index:
            raise ValueError("processor canary record order drifted")
        real = _mapping(record["real_request"], "real processor request")
        included = _mapping(
            record["included_history_mutation"], "included-history canary"
        )
        decision_step = int(states[expected_index]["decision_step_id"])
        for name, witness in (("real", real), ("included", included)):
            _exact_keys(
                witness,
                {"event_step_id", *PROCESSOR_IDENTITY_FIELDS}
                if name == "included"
                else PROCESSOR_IDENTITY_FIELDS,
                f"{name} canary request",
            )
            identity_payload = (
                {key: witness[key] for key in PROCESSOR_IDENTITY_FIELDS}
                if name == "included"
                else witness
            )
            _validate_processor_identity(
                identity_payload,
                name=f"{name} canary request",
                decision_step_id=decision_step,
            )
        real_requests[expected_index] = dict(real)
        history = states[expected_index]["history_event_step_ids"]
        if (
            included.get("event_step_id") != history[-1]
            or included["rendered_prompt_sha256"] == real["rendered_prompt_sha256"]
            or included["input_ids_sha256"] == real["input_ids_sha256"]
            or included["processor_request_sha256"] == real["processor_request_sha256"]
            or any(
                included[key] != real[key]
                for key in (
                    "pixel_values_sha256",
                    "image_grid_thw_sha256",
                    "pixel_values_shape",
                    "pixel_values_dtype",
                    "image_grid_thw_shape",
                    "image_grid_thw_dtype",
                    "image_count",
                    "processor_path",
                    "policy_forward_executed",
                )
            )
        ):
            raise ValueError("included-history mutation did not change processor identity")
        included_count += 1
        excluded = list(
            _sequence(record["excluded_event_mutations"], "excluded-event canaries")
        )
        step = int(states[expected_index]["decision_step_id"])
        expected_excluded = list(range(step, 6))
        if len(excluded) != len(expected_excluded):
            raise ValueError("excluded-event canary mutation inventory drifted")
        if excluded:
            excluded_states += 1
        for expected_step, raw_mutation in zip(expected_excluded, excluded, strict=True):
            mutation = _mapping(raw_mutation, "excluded-event canary")
            _exact_keys(
                mutation,
                {"event_step_id", *PROCESSOR_IDENTITY_FIELDS},
                "excluded-event canary",
            )
            if (
                mutation.get("event_step_id") != expected_step
                or any(
                    mutation.get(key) != real[key]
                    for key in PROCESSOR_IDENTITY_FIELDS
                )
            ):
                raise ValueError("excluded-event action changed processor identity")
            excluded_mutations += 1
    if (
        report.get("included_history_mutation_count") != included_count
        or report.get("excluded_state_count") != excluded_states
        or report.get("excluded_event_mutation_count") != excluded_mutations
    ):
        raise ValueError("worker canary aggregate counts drifted")
    completed_at = _timestamp(
        report.get("completed_at_utc"), "worker canary completion time"
    )
    return (
        included_count,
        excluded_states,
        excluded_mutations,
        real_requests,
        completed_at,
    )


def _validate_execution_evidence(
    value: Any,
    *,
    files: Mapping[str, bytes],
    run_contract: Mapping[str, Any],
    run_contract_sha256: str,
    runtimes: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, str, str | None, bool]:
    evidence = _mapping(value, "execution evidence")
    _exact_keys(
        evidence,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "runner_source_git_commit",
            "execution_git_commit",
            "config_sha256",
            "completion_manifest_sha256",
            "derived_immutable_revision",
            "runtime_identity_sha256",
            "worker_identity_sha256",
            "logs",
            "monitor_summary",
            "terminal_failure",
            "forbidden_operation_counts",
            "ended_at_utc",
        },
        "execution evidence",
    )
    source = _mapping(run_contract["source"], "run-contract source")
    frozen = _mapping(run_contract["frozen_config"], "run-contract config")
    completion = _mapping(run_contract["derived_completion"], "run-contract completion")
    derived = _mapping(run_contract["derived_artifact"], "run-contract derived")
    runtime_projection = {
        worker: dict(runtimes[worker]) for worker in sorted(runtimes)
    }
    worker_projection = {
        worker: {
            key: runtimes[worker][key]
            for key in (
                "device",
                "gpu_name",
                "gpu_uuid",
                "gpu_pci_bus_id",
                "logical_device_index",
                "nvidia_smi_index",
            )
        }
        for worker in sorted(runtimes)
    }
    terminal_failed = evidence.get("status") == "FAILED_TERMINAL_EXECUTION_EVIDENCE"
    terminal_failure = evidence.get("terminal_failure")
    if terminal_failed:
        _failure_category(terminal_failure, "terminal execution-evidence failure")
    elif terminal_failure is not None:
        raise ValueError("valid terminal execution evidence carries a failure")
    if (
        evidence.get("schema_version") != SCHEMA_VERSION
        or evidence.get("protocol_id") != PROTOCOL_ID
        or evidence.get("status")
        not in {
            "TERMINAL_EXECUTION_EVIDENCE",
            "FAILED_TERMINAL_EXECUTION_EVIDENCE",
        }
        or evidence.get("run_contract_sha256") != run_contract_sha256
        or evidence.get("runner_source_git_commit")
        != source["runner_source_git_commit"]
        or evidence.get("execution_git_commit") != source["execution_git_commit"]
        or evidence.get("config_sha256") != frozen["sha256"]
        or evidence.get("completion_manifest_sha256") != completion["sha256"]
        or evidence.get("derived_immutable_revision")
        != derived["immutable_revision"]
        or evidence.get("runtime_identity_sha256")
        != sha256_bytes(canonical_json_bytes(runtime_projection))
        or evidence.get("worker_identity_sha256")
        != sha256_bytes(canonical_json_bytes(worker_projection))
        or evidence.get("forbidden_operation_counts")
        != {key: 0 for key in sorted(FORBIDDEN_COUNT_KEYS)}
    ):
        raise ValueError("terminal execution provenance drifted")
    _timestamp(evidence.get("ended_at_utc"), "execution evidence end time")
    log_records = _mapping(evidence.get("logs"), "execution log witnesses")
    if set(log_records) != set(LOG_PATHS):
        raise ValueError("execution log inventory drifted")
    normalized_logs: list[dict[str, Any]] = []
    for name in sorted(LOG_PATHS):
        witness = dict(_file_witness(log_records[name], f"{name} log witness"))
        if witness["path"] != LOG_PATHS[name]:
            raise ValueError("execution log path drifted")
        if witness["path"] not in files:
            raise ValueError("execution log is missing from raw evidence")
        payload = files[witness["path"]]
        if witness["sha256"] != sha256_bytes(payload) or witness[
            "size_bytes"
        ] != len(payload):
            raise ValueError("execution log witness differs from archived bytes")
        normalized_logs.append(witness)
    witness: dict[str, Any] | None = None
    summary: Mapping[str, Any] | None = None
    if terminal_failed:
        if evidence.get("monitor_summary") is not None or MONITOR_SUMMARY_PATH in files:
            raise ValueError("failed terminal evidence must not attest a monitor summary")
    else:
        monitor = _mapping(evidence.get("monitor_summary"), "monitor summary witness")
        witness = dict(_file_witness(monitor, "monitor summary witness"))
        if witness["path"] != MONITOR_SUMMARY_PATH or witness["path"] not in files:
            raise ValueError("monitor summary path or presence drifted")
        monitor_payload = files[witness["path"]]
        if witness["sha256"] != sha256_bytes(monitor_payload) or witness[
            "size_bytes"
        ] != len(monitor_payload):
            raise ValueError("monitor summary witness differs from archived bytes")
        summary = strict_json_object_bytes(monitor_payload, label="monitor summary")
    ready_path = LOG_PATHS["utilization_monitor_ready"]
    ready = strict_json_object_bytes(files[ready_path], label="monitor ready witness")
    _exact_keys(
        ready,
        {
            "schema_version",
            "status",
            "container_name_prefix",
            "minimum_gpu_utilization_percent",
            "monitor_process_alive",
            "monitor_pid",
            "started_at_utc",
        },
        "monitor ready witness",
    )
    ready_started = _timestamp(ready.get("started_at_utc"), "monitor ready start")
    if (
        ready.get("schema_version") != SCHEMA_VERSION
        or ready.get("status") != "GPU_UTILIZATION_MONITOR_READY"
        or ready.get("container_name_prefix") != "sglang-omni-jaxan"
        or ready.get("minimum_gpu_utilization_percent") != 90
        or ready.get("monitor_process_alive") is not True
        or type(ready.get("monitor_pid")) is not int
        or ready["monitor_pid"] <= 0
    ):
        raise ValueError("monitor-ready witness drifted")
    stop_path = LOG_PATHS["utilization_monitor_stop_request"]
    stop_payload = files[stop_path]
    stop = strict_json_object_bytes(stop_payload, label="monitor stop request")
    _exact_keys(stop, MONITOR_STOP_REQUEST_KEYS, "monitor stop request")
    stop_requested = _timestamp(
        stop.get("requested_at_utc"), "monitor stop request time"
    )
    if (
        stop.get("schema_version") != SCHEMA_VERSION
        or stop.get("status") != MONITOR_STOP_REQUEST_STATUS
        or stop.get("run_contract_sha256") != run_contract_sha256
        or stop_requested < ready_started
    ):
        raise ValueError("monitor stop request drifted")
    if summary is not None and (
        set(summary) != MONITOR_SUMMARY_KEYS
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != MONITOR_SUMMARY_STATUS
        or summary.get("run_contract_sha256") != run_contract_sha256
        or summary.get("container_name_prefix") != "sglang-omni-jaxan"
        or summary.get("minimum_gpu_utilization_percent") != 90
        or summary.get("started_before_first_generation") is not True
        or type(summary.get("sample_count")) is not int
        or summary["sample_count"] <= 0
        or type(summary.get("low_utilization_incident_count")) is not int
        or summary["low_utilization_incident_count"] < 0
        or summary["low_utilization_incident_count"] > summary["sample_count"]
        or summary.get("started_at_utc") != ready["started_at_utc"]
        or summary.get("visible_gpu_count") != 2
        or not isinstance(summary.get("visible_gpu_uuids"), list)
        or len(summary["visible_gpu_uuids"]) != 2
        or len(set(summary["visible_gpu_uuids"])) != 2
        or any(
            not isinstance(value, str) or not value.startswith("GPU-")
            for value in summary["visible_gpu_uuids"]
        )
        or summary.get("stop_request_sha256") != sha256_bytes(stop_payload)
    ):
        raise ValueError("utilization monitor summary drifted")
    if summary is not None:
        _validate_monitor_gpu_uuid_binding(summary, runtimes=runtimes)
    if summary is not None and _timestamp(
        summary.get("stopped_at_utc"), "monitor summary stop time"
    ) < stop_requested:
        raise ValueError("monitor summary predates its stop request")
    return (
        str(evidence["runtime_identity_sha256"]),
        str(evidence["worker_identity_sha256"]),
        sha256_bytes(canonical_json_bytes(normalized_logs)),
        str(witness["sha256"]) if witness is not None else None,
        terminal_failed,
    )


def _validate_monitor_gpu_uuid_binding(
    summary: Mapping[str, Any],
    *,
    runtimes: Mapping[str, Mapping[str, Any]],
) -> None:
    runtime_uuids = [runtime.get("gpu_uuid") for runtime in runtimes.values()]
    visible_uuids = summary.get("visible_gpu_uuids")
    if (
        len(runtime_uuids) != 2
        or any(
            not isinstance(value, str) or not value.startswith("GPU-")
            for value in runtime_uuids
        )
        or not isinstance(visible_uuids, list)
        or set(visible_uuids) != set(runtime_uuids)
    ):
        raise ValueError("utilization monitor GPUs differ from worker runtime GPUs")


def reduce_expansion_substrate_gate(
    reductions: Sequence[Mapping[str, Any]],
    *,
    actual_counts: Mapping[str, int],
    frozen_config: Mapping[str, Any],
) -> dict[str, Any]:
    if len(reductions) != EXPECTED_STATE_COUNT:
        raise ValueError("gate reduction requires the fixed 192-state denominator")
    gate_contract = _mapping(frozen_config.get("substrate_gate"), "substrate gate")
    thresholds = {
        "fixed_state_denominator": EXPECTED_STATE_COUNT,
        "minimum_parse_coverage": 0.99,
        "derived_required_parse_success_count": 191,
        "minimum_finite_logit_coverage": 1.0,
        "derived_required_finite_logit_state_count": 192,
        "minimum_repeat_canonical_action_agreement": 1.0,
        "derived_required_repeat_agreement_count": 192,
        "minimum_memory_sensitive_states": 8,
        "failed_states_remain_in_fixed_denominator": True,
    }
    for key, expected in thresholds.items():
        if gate_contract.get(key) != expected:
            raise ValueError(f"frozen substrate gate drifted: {key}")
    parse_count = sum(record.get("parse_success") is True for record in reductions)
    finite_count = sum(
        record.get("finite_logit_distances") is True for record in reductions
    )
    agreement_count = sum(
        record.get("repeat_canonical_action_agreement") is True
        for record in reductions
    )
    repeat_values = [
        float(record["repeat_reference_kl"])
        for record in reductions
        if isinstance(record.get("repeat_reference_kl"), (int, float))
        and not isinstance(record.get("repeat_reference_kl"), bool)
        and math.isfinite(float(record["repeat_reference_kl"]))
    ]
    mean_repeat = math.fsum(repeat_values) / len(repeat_values) if repeat_values else None
    epsilon = max(1e-4, 10.0 * mean_repeat) if mean_repeat is not None else None
    memory_sensitive_ids: list[str] = []
    if epsilon is not None:
        for record in reductions:
            value = record.get("summary_reference_kl")
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > epsilon
            ):
                memory_sensitive_ids.append(str(record["state_id"]))
    metrics = {
        "fixed_state_denominator": EXPECTED_STATE_COUNT,
        "parse_success_count": parse_count,
        "parse_coverage": parse_count / EXPECTED_STATE_COUNT,
        "finite_logit_state_count": finite_count,
        "finite_logit_coverage": finite_count / EXPECTED_STATE_COUNT,
        "repeat_canonical_action_agreement_count": agreement_count,
        "repeat_canonical_action_agreement": agreement_count / EXPECTED_STATE_COUNT,
        "finite_repeat_kl_count": len(repeat_values),
        "mean_repeat_kl": mean_repeat,
        "repeat_noise_epsilon": epsilon,
        "memory_sensitive_state_count": len(memory_sensitive_ids),
        "memory_sensitive_state_fraction": (
            len(memory_sensitive_ids) / EXPECTED_STATE_COUNT
        ),
        "memory_sensitive_state_ids": memory_sensitive_ids,
        **dict(actual_counts),
    }
    checks = {
        "minimum_parse_coverage": parse_count >= 191,
        "minimum_finite_logit_coverage": finite_count == EXPECTED_STATE_COUNT,
        "minimum_repeat_canonical_action_agreement": (
            agreement_count == EXPECTED_STATE_COUNT
        ),
        "minimum_memory_sensitive_states": len(memory_sensitive_ids) >= 8,
        "maximum_generation_calls": (
            actual_counts.get("generation_call_count", -1)
            <= PLANNED_COUNTS["generation_call_count"]
        ),
        "maximum_teacher_forwards": (
            actual_counts.get("teacher_forward_count", -1)
            <= PLANNED_COUNTS["teacher_forward_count"]
        ),
        "maximum_kl_measurements": (
            actual_counts.get("kl_measurement_count", -1)
            <= PLANNED_COUNTS["kl_measurement_count"]
        ),
        "fixed_denominator_preserved": len(reductions) == EXPECTED_STATE_COUNT,
    }
    passed = all(checks.values())
    return {
        "gate_passed": passed,
        "gate_contract": {
            key: gate_contract[key] for key in thresholds
        },
        "metrics": metrics,
        "checks": checks,
        "derived_outcome": PASS_OUTCOME if passed else NO_GO_OUTCOME,
    }


def validate_expansion_substrate_evidence_files(
    files: Mapping[str, bytes],
    *,
    expected_source_git_commit: str | None = None,
    expected_config_sha256: str | None = None,
    require_canonical_attempt_identity: bool = True,
) -> ExpansionSubstrateEvidence:
    copied = dict(files)
    if any(
        not isinstance(name, str) or not isinstance(payload, bytes)
        for name, payload in copied.items()
    ):
        raise TypeError("raw evidence must map relative paths to bytes")
    for name in copied:
        _safe_relative_path(name)
    required = {
        GLOBAL_LEDGER_ARCHIVE_NAME,
        RUN_MANIFEST_FILENAME,
        AGGREGATE_FILENAME,
        EXECUTION_EVIDENCE_FILENAME,
    }
    if not required.issubset(copied):
        raise ValueError("raw evidence lacks ledger, manifest, aggregate, or provenance")
    manifest = strict_json_object_bytes(copied[RUN_MANIFEST_FILENAME], label="run manifest")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "run_contract",
            "created_at_utc",
        },
        "run manifest",
    )
    run_contract = _mapping(manifest.get("run_contract"), "run contract")
    states = validate_execution_run_contract(
        run_contract,
        require_canonical_attempt_identity=require_canonical_attempt_identity,
    )
    run_contract_sha = sha256_bytes(canonical_json_bytes(run_contract))
    run_source = _mapping(run_contract["source"], "source")
    source_commit = str(run_source["runner_source_git_commit"])
    execution_commit = str(run_source["execution_git_commit"])
    config_sha = str(_mapping(run_contract["frozen_config"], "config")["sha256"])
    completion_sha = str(
        _mapping(run_contract["derived_completion"], "completion")["sha256"]
    )
    derived_revision = str(
        _mapping(run_contract["derived_artifact"], "derived artifact")[
            "immutable_revision"
        ]
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "EXPANSION_SUBSTRATE_GLOBAL_ATTEMPT_CLAIMED"
        or manifest.get("run_contract_sha256") != run_contract_sha
        or expected_source_git_commit is not None
        and source_commit != expected_source_git_commit
        or expected_config_sha256 is not None
        and config_sha != expected_config_sha256
    ):
        raise ValueError("run manifest source/config identity drifted")
    _timestamp(manifest.get("created_at_utc"), "run manifest creation time")

    attempt = _mapping(run_contract["attempt_identity"], "attempt identity")
    ledger = strict_json_object_bytes(
        copied[GLOBAL_LEDGER_ARCHIVE_NAME], label="global attempt ledger"
    )
    ledger_status, ledger_completed, ledger_attempted = _validate_global_ledger(
        ledger,
        run_contract_sha256=run_contract_sha,
        expected_global_ledger=Path(str(attempt["global_ledger"])),
    )
    aggregate = strict_json_object_bytes(copied[AGGREGATE_FILENAME], label="aggregate")
    execution_record = strict_json_object_bytes(
        copied[EXECUTION_EVIDENCE_FILENAME], label="execution evidence"
    )
    terminal_evidence_failed_declared = (
        execution_record.get("status") == "FAILED_TERMINAL_EXECUTION_EVIDENCE"
    )
    ready_path = LOG_PATHS["utilization_monitor_ready"]
    if ready_path not in copied:
        raise ValueError("raw evidence lacks the monitor-ready witness")
    ready_record = strict_json_object_bytes(
        copied[ready_path], label="monitor ready witness"
    )
    monitor_ready_started = _timestamp(
        ready_record.get("started_at_utc"), "monitor ready start"
    )
    outcome = aggregate.get("outcome")
    if outcome not in TERMINAL_OUTCOMES:
        raise ValueError("aggregate outcome is not PASS, NO-GO, or INVALID")

    expected_observed_files = set(required)
    runtime_values: dict[str, Mapping[str, Any]] = {}
    observed_states: dict[
        int, tuple[dict[str, int], str, str | None, dict[str, Any]]
    ] = {}
    attempted_total = 0
    completed_total = 0
    first_attempt_time: datetime | None = None
    worker_terminal_outcomes: dict[str, str] = {}
    worker_ledger_statuses: dict[str, str] = {}
    canary_totals = [0, 0, 0]
    canary_reports_present = 0
    canary_real_requests: dict[int, dict[str, Any]] = {}
    canary_completed_times: list[datetime] = []
    runtime_created_times: list[datetime] = []
    specs = expected_worker_specs()
    high_water = _mapping(ledger["worker_high_water"], "worker high-water")
    for spec in specs:
        prefix = f"{WORKER_DIRECTORY}/{spec.worker_id}"
        sibling_name = f"{WORKER_SIBLING_LEDGER_DIRECTORY}/{spec.worker_id}.json"
        if sibling_name not in copied:
            raise ValueError("raw evidence lacks a prebound worker sibling ledger")
        expected_observed_files.add(sibling_name)
        sibling_payload = copied[sibling_name]
        worker_ledger = strict_json_object_bytes(
            sibling_payload, label=f"{spec.worker_id} sibling ledger"
        )
        worker_status, attempted, completed = _validate_worker_ledger(
            worker_ledger, spec=spec, run_contract_sha256=run_contract_sha
        )
        worker_ledger_statuses[spec.worker_id] = worker_status
        attempted_total += len(attempted)
        completed_total += len(completed)
        high = _mapping(high_water[spec.worker_id], "worker high-water record")
        if (
            high.get("attempted_state_indices") != attempted
            or high.get("completed_state_indices") != completed
            or high.get("worker_ledger_sha256") != sha256_bytes(sibling_payload)
        ):
            raise ValueError("global ledger does not bind worker high-water bytes")
        root_ledger_name = f"{prefix}/{WORKER_LEDGER_FILENAME}"
        if root_ledger_name in copied:
            expected_observed_files.add(root_ledger_name)
            if copied[root_ledger_name] != sibling_payload and outcome != INVALID_OUTCOME:
                raise ValueError("completed worker root/sibling ledgers differ")
            stale = strict_json_object_bytes(
                copied[root_ledger_name], label="worker root ledger"
            )
            _root_status, root_attempted, root_completed = _validate_worker_ledger(
                stale, spec=spec, run_contract_sha256=run_contract_sha
            )
            if (
                root_attempted != attempted[: len(root_attempted)]
                or root_completed != completed[: len(root_completed)]
                or len(attempted) - len(root_attempted) not in {0, 1}
                or len(completed) - len(root_completed) not in {0, 1}
            ):
                raise ValueError("stale worker ledger exceeds one durable transition")
        elif outcome != INVALID_OUTCOME:
            raise ValueError("terminal evidence lacks worker root ledger")

        runtime_name = f"{prefix}/{WORKER_RUNTIME_FILENAME}"
        if runtime_name in copied:
            expected_observed_files.add(runtime_name)
            envelope = strict_json_object_bytes(
                copied[runtime_name], label="worker runtime identity"
            )
            _exact_keys(
                envelope,
                {
                    "schema_version",
                    "protocol_id",
                    "run_contract_sha256",
                    "worker",
                    "runtime_metadata",
                    "created_at_utc",
                },
                "worker runtime identity",
            )
            if (
                envelope.get("schema_version") != SCHEMA_VERSION
                or envelope.get("protocol_id") != PROTOCOL_ID
                or envelope.get("run_contract_sha256") != run_contract_sha
                or envelope.get("worker") != spec.to_dict()
            ):
                raise ValueError("worker runtime envelope drifted")
            runtime_created_times.append(
                _timestamp(envelope.get("created_at_utc"), "worker runtime time")
            )
            runtime_values[spec.worker_id] = _validate_runtime_metadata(
                envelope.get("runtime_metadata"), spec=spec
            )

        canary_name = f"{prefix}/{WORKER_CANARY_FILENAME}"
        if canary_name in copied:
            expected_observed_files.add(canary_name)
            (
                included_total,
                excluded_state_total,
                excluded_mutation_total,
                reals,
                canary_completed_at,
            ) = _validate_canary_report(
                strict_json_object_bytes(copied[canary_name], label="canary report"),
                spec=spec,
                states=states,
                run_contract_sha256=run_contract_sha,
            )
            canary_totals = [
                canary_totals[0] + included_total,
                canary_totals[1] + excluded_state_total,
                canary_totals[2] + excluded_mutation_total,
            ]
            if set(canary_real_requests).intersection(reals):
                raise ValueError("worker canary reports overlap state indices")
            canary_real_requests.update(reals)
            canary_completed_times.append(canary_completed_at)
            canary_reports_present += 1

        observed_marker_indices: list[int] = []
        observed_state_indices: list[int] = []
        for index in spec.state_indices:
            marker_name = f"{prefix}/{ATTEMPT_DIRECTORY}/{index:03d}.json"
            state_name = f"{prefix}/{STATE_DIRECTORY}/{index:03d}.json"
            if marker_name in copied:
                expected_observed_files.add(marker_name)
                observed_marker_indices.append(index)
                marker_time = _validate_attempt_marker(
                    strict_json_object_bytes(copied[marker_name], label="state marker"),
                    spec=spec,
                    state_index=index,
                    run_contract_sha256=run_contract_sha,
                )
                first_attempt_time = (
                    marker_time
                    if first_attempt_time is None
                    else min(first_attempt_time, marker_time)
                )
            if state_name in copied:
                expected_observed_files.add(state_name)
                observed_state_indices.append(index)
                observed_states[index] = _validate_state_record(
                    strict_json_object_bytes(copied[state_name], label="state record"),
                    spec=spec,
                    projection=states[index],
                    run_contract_sha256=run_contract_sha,
                    runtime_metadata=runtime_values.get(spec.worker_id),
                )
        _validate_worker_file_window(
            outcome=str(outcome),
            attempted=attempted,
            completed=completed,
            observed_markers=observed_marker_indices,
            observed_states=observed_state_indices,
        )
        terminal_name = f"{prefix}/{WORKER_TERMINAL_FILENAME}"
        if terminal_name in copied:
            expected_observed_files.add(terminal_name)
            terminal = strict_json_object_bytes(
                copied[terminal_name], label="worker terminal"
            )
            _exact_keys(
                terminal,
                {
                    "schema_version",
                    "protocol_id",
                    "run_contract_sha256",
                    "worker",
                    "outcome",
                    "planned_counts",
                    "actual_counts",
                    "completed_state_indices",
                    "attempted_state_indices",
                    "failure",
                    "ended_at_utc",
                    "retry_count",
                    "top_up_count",
                },
                "worker terminal",
            )
            if (
                terminal.get("schema_version") != SCHEMA_VERSION
                or terminal.get("protocol_id") != PROTOCOL_ID
                or terminal.get("run_contract_sha256") != run_contract_sha
                or terminal.get("worker") != spec.to_dict()
                or terminal.get("planned_counts")
                != {key: value // 2 for key, value in PLANNED_COUNTS.items()}
                or terminal.get("completed_state_indices") != completed
                or terminal.get("attempted_state_indices") != attempted
                or terminal.get("outcome")
                not in {WORKER_OUTCOME_COMPLETE, WORKER_OUTCOME_INVALID}
                or terminal.get("retry_count") != 0
                or terminal.get("top_up_count") != 0
            ):
                raise ValueError("worker terminal identity/counts drifted")
            _validate_count_mapping(
                terminal.get("actual_counts"),
                name="worker terminal actual counts",
                maxima={key: value // 2 for key, value in PLANNED_COUNTS.items()},
            )
            expected_worker_actual = {
                key: sum(observed_states[index][0][key] for index in completed)
                for key in PER_STATE_PLANNED_COUNTS
            }
            if terminal.get("actual_counts") != expected_worker_actual:
                raise ValueError("worker terminal actual counts differ from raw states")
            _failure_category(terminal.get("failure"), "worker terminal failure")
            if terminal["outcome"] == WORKER_OUTCOME_COMPLETE:
                if (
                    worker_status != "WORKER_SHARD_COMPLETED"
                    or attempted != list(spec.state_indices)
                    or completed != list(spec.state_indices)
                    or terminal.get("failure") is not None
                ):
                    raise ValueError("COMPLETE worker terminal/ledger coupling drifted")
            elif (
                worker_status != "WORKER_SHARD_INVALID"
                or not isinstance(terminal.get("failure"), Mapping)
            ):
                raise ValueError("INVALID worker terminal/ledger coupling drifted")
            _timestamp(terminal.get("ended_at_utc"), "worker terminal time")
            worker_terminal_outcomes[spec.worker_id] = str(terminal["outcome"])

    if (ledger_completed, ledger_attempted) != (completed_total, attempted_total):
        raise ValueError("global and worker high-water counts differ")
    actual_counts = {
        key: sum(record[0][key] for record in observed_states.values())
        for key in PER_STATE_PLANNED_COUNTS
    }
    if ledger.get("actual_counts") != actual_counts:
        raise ValueError("global ledger actual counts differ from raw state records")
    valid_count = sum(
        record[1] == STATE_OUTCOME_VALID for record in observed_states.values()
    )
    failed_count = sum(
        record[1] == STATE_OUTCOME_FAILED for record in observed_states.values()
    )
    failure_categories = Counter(
        record[2] for record in observed_states.values() if record[2] is not None
    )
    for index, record in observed_states.items():
        if canary_real_requests.get(index) != record[3]["processor_identity"]:
            raise ValueError("state processor identity differs from canary report")
    computed_gate = (
        reduce_expansion_substrate_gate(
            [observed_states[index][3] for index in range(EXPECTED_STATE_COUNT)],
            actual_counts=actual_counts,
            frozen_config=_mapping(run_contract["frozen_config"], "frozen config")[
                "data"
            ],
        )
        if sorted(observed_states) == list(range(EXPECTED_STATE_COUNT))
        else None
    )

    canary_release_time: datetime | None = None
    if CANARY_RELEASE_FILENAME in copied:
        expected_observed_files.add(CANARY_RELEASE_FILENAME)
        release = strict_json_object_bytes(
            copied[CANARY_RELEASE_FILENAME], label="processor canary release"
        )
        _exact_keys(
            release,
            {
                "schema_version",
                "protocol_id",
                "status",
                "run_contract_sha256",
                "worker_report_sha256",
                "state_count",
                "included_history_mutation_count",
                "excluded_state_count",
                "excluded_event_mutation_count",
                "released_at_utc",
            },
            "processor canary release",
        )
        report_hashes = {
            spec.worker_id: sha256_bytes(
                copied[f"{WORKER_DIRECTORY}/{spec.worker_id}/{WORKER_CANARY_FILENAME}"]
            )
            for spec in specs
            if f"{WORKER_DIRECTORY}/{spec.worker_id}/{WORKER_CANARY_FILENAME}" in copied
        }
        if (
            release.get("schema_version") != SCHEMA_VERSION
            or release.get("protocol_id") != PROTOCOL_ID
            or release.get("status") != "COORDINATOR_RELEASED_PROCESSOR_CANARY_BARRIER"
            or release.get("run_contract_sha256") != run_contract_sha
            or release.get("worker_report_sha256") != report_hashes
            or release.get("state_count") != EXPECTED_STATE_COUNT
            or release.get("included_history_mutation_count") != 192
            or release.get("excluded_state_count") != 128
            or release.get("excluded_event_mutation_count") != 192
            or canary_reports_present != 2
            or canary_totals != [192, 128, 192]
        ):
            raise ValueError("global processor canary release drifted")
        canary_release_time = _timestamp(
            release.get("released_at_utc"), "processor canary release time"
        )
    if CANARY_ABORT_FILENAME in copied:
        expected_observed_files.add(CANARY_ABORT_FILENAME)
        abort = strict_json_object_bytes(
            copied[CANARY_ABORT_FILENAME], label="processor canary abort"
        )
        if (
            abort.get("protocol_id") != PROTOCOL_ID
            or abort.get("run_contract_sha256") != run_contract_sha
            or abort.get("status") != "COORDINATOR_ABORTED_PROCESSOR_CANARY_BARRIER"
            or not isinstance(abort.get("failure"), Mapping)
        ):
            raise ValueError("processor canary abort drifted")
    if CANARY_RELEASE_FILENAME in copied and CANARY_ABORT_FILENAME in copied:
        raise ValueError("processor canary barrier cannot release and abort")
    if first_attempt_time is not None and (
        canary_release_time is None or canary_release_time >= first_attempt_time
    ):
        raise ValueError("processor canary release was not before every attempt marker")

    runtime_release_time: datetime | None = None
    if (
        RUNTIME_BARRIER_RELEASE_FILENAME in copied
        and RUNTIME_BARRIER_ABORT_FILENAME in copied
    ):
        raise ValueError("runtime barrier cannot release and abort")
    if RUNTIME_BARRIER_RELEASE_FILENAME in copied:
        expected_observed_files.add(RUNTIME_BARRIER_RELEASE_FILENAME)
        runtime_release = strict_json_object_bytes(
            copied[RUNTIME_BARRIER_RELEASE_FILENAME], label="runtime barrier release"
        )
        _exact_keys(
            runtime_release,
            {
                "schema_version",
                "protocol_id",
                "status",
                "run_contract_sha256",
                "worker_runtime_sha256",
                "released_at_utc",
            },
            "runtime barrier release",
        )
        expected_runtime_hashes = {
            spec.worker_id: sha256_bytes(
                copied[f"{WORKER_DIRECTORY}/{spec.worker_id}/{WORKER_RUNTIME_FILENAME}"]
            )
            for spec in specs
            if f"{WORKER_DIRECTORY}/{spec.worker_id}/{WORKER_RUNTIME_FILENAME}" in copied
        }
        if (
            runtime_release.get("schema_version") != SCHEMA_VERSION
            or runtime_release.get("protocol_id") != PROTOCOL_ID
            or runtime_release.get("status")
            != "COORDINATOR_RELEASED_BOTH_WORKER_RUNTIMES"
            or runtime_release.get("run_contract_sha256") != run_contract_sha
            or runtime_release.get("worker_runtime_sha256") != expected_runtime_hashes
            or len(expected_runtime_hashes) != 2
        ):
            raise ValueError("runtime barrier release identity drifted")
        runtime_release_time = _timestamp(
            runtime_release.get("released_at_utc"), "runtime release time"
        )
    if RUNTIME_BARRIER_ABORT_FILENAME in copied:
        expected_observed_files.add(RUNTIME_BARRIER_ABORT_FILENAME)
        runtime_abort = strict_json_object_bytes(
            copied[RUNTIME_BARRIER_ABORT_FILENAME], label="runtime barrier abort"
        )
        _exact_keys(
            runtime_abort,
            {
                "schema_version",
                "protocol_id",
                "status",
                "run_contract_sha256",
                "failure",
                "aborted_at_utc",
            },
            "runtime barrier abort",
        )
        if (
            runtime_abort.get("schema_version") != SCHEMA_VERSION
            or runtime_abort.get("protocol_id") != PROTOCOL_ID
            or runtime_abort.get("status") != "COORDINATOR_ABORTED_RUNTIME_BARRIER"
            or runtime_abort.get("run_contract_sha256") != run_contract_sha
            or not isinstance(runtime_abort.get("failure"), Mapping)
        ):
            raise ValueError("runtime barrier abort identity drifted")
        _timestamp(runtime_abort.get("aborted_at_utc"), "runtime abort time")
    if canary_release_time is not None:
        if (
            runtime_release_time is None
            or not runtime_created_times
            or not canary_completed_times
            or monitor_ready_started > min(runtime_created_times)
            or max(runtime_created_times) > runtime_release_time
            or monitor_ready_started > runtime_release_time
            or runtime_release_time >= min(canary_completed_times)
            or max(canary_completed_times) > canary_release_time
        ):
            raise ValueError("runtime/canary barrier temporal order drifted")

    if outcome in {PASS_OUTCOME, NO_GO_OUTCOME}:
        if (
            ledger_status != "GLOBAL_ATTEMPT_COMPLETED"
            or attempted_total != EXPECTED_STATE_COUNT
            or completed_total != EXPECTED_STATE_COUNT
            or sorted(observed_states) != list(range(EXPECTED_STATE_COUNT))
            or set(runtime_values) != {spec.worker_id for spec in specs}
            or worker_terminal_outcomes
            != {spec.worker_id: WORKER_OUTCOME_COMPLETE for spec in specs}
            or worker_ledger_statuses
            != {spec.worker_id: "WORKER_SHARD_COMPLETED" for spec in specs}
            or canary_release_time is None
            or RUNTIME_BARRIER_RELEASE_FILENAME not in copied
            or terminal_evidence_failed_declared
        ):
            raise ValueError("terminal PASS/NO-GO evidence is partial")
        _validate_runtime_pair(runtime_values)
        if computed_gate is None or computed_gate["derived_outcome"] != outcome:
            raise ValueError("aggregate outcome differs from raw-state gate reduction")
    else:
        if ledger_status != "GLOBAL_ATTEMPT_INVALID":
            raise ValueError("INVALID aggregate lacks INVALID global ledger")
        if not isinstance(aggregate.get("invalid_failure"), Mapping):
            raise ValueError("INVALID aggregate lacks preserved failure evidence")
        if computed_gate is not None and not terminal_evidence_failed_declared:
            raise ValueError("complete scientific denominator must be PASS or NO-GO")
        if computed_gate is not None:
            invalid_failure = _mapping(
                aggregate.get("invalid_failure"), "terminal evidence invalid failure"
            )
            if invalid_failure.get("stage") != "terminal_execution_evidence_finalization":
                raise ValueError("complete INVALID denominator lacks terminal-evidence failure")

    expected_terminal_files = expected_observed_files | set(LOG_PATHS.values())
    if not terminal_evidence_failed_declared:
        expected_terminal_files.add(MONITOR_SUMMARY_PATH)
    if set(copied) != expected_terminal_files:
        raise ValueError("raw evidence contains unknown or unbound files")
    if set(runtime_values) != {spec.worker_id for spec in specs}:
        if outcome != INVALID_OUTCOME:
            raise ValueError("terminal outcome lacks both runtime identities")
        # note (luojiaxuan): INVALID evidence binds an empty runtime projection
        # when failure happens before runtime identity is available.
    runtime_sha, worker_sha, log_sha, monitor_sha, terminal_evidence_failed = _validate_execution_evidence(
        execution_record,
        files=copied,
        run_contract=run_contract,
        run_contract_sha256=run_contract_sha,
        runtimes=runtime_values,
    )
    if terminal_evidence_failed != terminal_evidence_failed_declared:
        raise ValueError("terminal execution-evidence status drifted")

    _exact_keys(
        aggregate,
        {
            "schema_version",
            "protocol_id",
            "status",
            "outcome",
            "run_contract_sha256",
            "fixed_state_denominator",
            "planned_counts",
            "actual_counts",
            "attempted_state_count",
            "completed_state_count",
            "valid_state_count",
            "failed_state_count",
            "failure_category_counts",
            "retry_performed",
            "top_up_performed",
            "started_at_utc",
            "ended_at_utc",
            "gate",
            "invalid_failure",
            "execution_evidence_sha256",
        },
        "aggregate",
    )
    if (
        aggregate.get("schema_version") != SCHEMA_VERSION
        or aggregate.get("protocol_id") != PROTOCOL_ID
        or aggregate.get("status") != "TERMINAL_EXPANSION_SUBSTRATE_ATTEMPT"
        or aggregate.get("run_contract_sha256") != run_contract_sha
        or aggregate.get("fixed_state_denominator") != EXPECTED_STATE_COUNT
        or aggregate.get("planned_counts") != PLANNED_COUNTS
        or aggregate.get("actual_counts") != actual_counts
        or aggregate.get("attempted_state_count") != attempted_total
        or aggregate.get("completed_state_count") != completed_total
        or aggregate.get("valid_state_count") != valid_count
        or aggregate.get("failed_state_count") != failed_count
        or aggregate.get("failure_category_counts") != dict(sorted(failure_categories.items()))
        or aggregate.get("retry_performed") is not False
        or aggregate.get("top_up_performed") is not False
        or aggregate.get("execution_evidence_sha256")
        != sha256_bytes(copied[EXECUTION_EVIDENCE_FILENAME])
        or aggregate.get("gate")
        != (None if outcome == INVALID_OUTCOME else computed_gate)
    ):
        raise ValueError("stored aggregate differs from raw fixed-denominator reduction")
    aggregate_started = _timestamp(
        aggregate.get("started_at_utc"), "aggregate start time"
    )
    aggregate_ended = _timestamp(aggregate.get("ended_at_utc"), "aggregate end time")
    if aggregate_started > aggregate_ended:
        raise ValueError("aggregate UTC bracket is reversed")
    if outcome in {PASS_OUTCOME, NO_GO_OUTCOME} and aggregate.get(
        "invalid_failure"
    ) is not None:
        raise ValueError("PASS/NO-GO aggregate may not carry invalid failure evidence")
    if outcome == INVALID_OUTCOME and aggregate.get("gate") is not None:
        raise ValueError("INVALID aggregate may not carry a scientific gate result")
    if outcome in {PASS_OUTCOME, NO_GO_OUTCOME} and attempted_total != EXPECTED_STATE_COUNT:
        raise ValueError("failed states may not be removed from the fixed denominator")

    inventory = _inventory(copied)
    return ExpansionSubstrateEvidence(
        files=copied,
        run_contract=dict(run_contract),
        source_git_commit=source_commit,
        execution_git_commit=execution_commit,
        run_contract_sha256=run_contract_sha,
        config_sha256=config_sha,
        completion_manifest_sha256=completion_sha,
        derived_immutable_revision=derived_revision,
        outcome=str(outcome),
        aggregate_sha256=sha256_bytes(copied[AGGREGATE_FILENAME]),
        fixed_state_denominator=EXPECTED_STATE_COUNT,
        attempted_state_count=attempted_total,
        completed_state_count=completed_total,
        planned_counts=PLANNED_COUNTS,
        actual_counts=actual_counts,
        valid_state_count=valid_count,
        failed_state_count=failed_count,
        failure_category_counts=dict(sorted(failure_categories.items())),
        gate_summary=computed_gate,
        inventory=inventory,
        tree_inventory_sha256=sha256_bytes(canonical_json_bytes(inventory)),
        runtime_identity_sha256=runtime_sha,
        worker_identity_sha256=worker_sha,
        log_inventory_sha256=log_sha,
        monitor_summary_sha256=monitor_sha,
    )


def collect_raw_expansion_substrate_evidence(
    raw_output_dir: str | Path,
    global_attempt_ledger: str | Path,
    *,
    expected_source_git_commit: str | None = None,
    expected_config_sha256: str | None = None,
    require_canonical_location: bool = True,
) -> ExpansionSubstrateEvidence:
    root = Path(raw_output_dir)
    ledger = Path(global_attempt_ledger)
    if require_canonical_location and (
        root.resolve() != CANONICAL_OUTPUT_DIR or ledger.resolve() != CANONICAL_LEDGER_PATH
    ):
        raise ValueError("raw root or global ledger is not canonical")
    if root.is_symlink() or ledger.is_symlink() or not root.is_dir() or not ledger.is_file():
        raise ValueError("raw evidence root or global ledger is missing")
    files = {GLOBAL_LEDGER_ARCHIVE_NAME: ledger.read_bytes()}
    global_record = strict_json_object_bytes(files[GLOBAL_LEDGER_ARCHIVE_NAME], label="global ledger")
    siblings = _mapping(global_record.get("worker_sibling_ledgers"), "worker siblings")
    for spec in expected_worker_specs():
        sibling = Path(str(siblings.get(spec.worker_id)))
        if sibling.is_symlink() or not sibling.is_file():
            raise ValueError("worker sibling ledger is missing")
        files[f"{WORKER_SIBLING_LEDGER_DIRECTORY}/{spec.worker_id}.json"] = sibling.read_bytes()
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ValueError("raw evidence contains a symlink")
        if item.is_file():
            relative = item.relative_to(root).as_posix()
            _safe_relative_path(relative)
            files[relative] = item.read_bytes()
        elif not item.is_dir():
            raise ValueError("raw evidence contains a non-regular filesystem entry")
    return validate_expansion_substrate_evidence_files(
        files,
        expected_source_git_commit=expected_source_git_commit,
        expected_config_sha256=expected_config_sha256,
        require_canonical_attempt_identity=require_canonical_location,
    )


def deterministic_ustar_bytes(files: Mapping[str, bytes]) -> bytes:
    destination = io.BytesIO()
    with tarfile.open(fileobj=destination, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for relative in sorted(files):
            _safe_relative_path(relative)
            payload = files[relative]
            info = tarfile.TarInfo(f"{ARCHIVE_MEMBER_PREFIX}/{relative}")
            info.size = len(payload)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return destination.getvalue()


def read_expansion_substrate_archive(
    archive_path: str | Path,
    *,
    expected_source_git_commit: str | None = None,
    expected_config_sha256: str | None = None,
    require_canonical_attempt_identity: bool = True,
) -> ExpansionSubstrateEvidence:
    path = Path(archive_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("expansion substrate archive is missing or symlinked")
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(path, mode="r:") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != sorted(
                member.name for member in members
            ):
                raise ValueError("archive member order drifted")
            prefix = f"{ARCHIVE_MEMBER_PREFIX}/"
            for member in members:
                if (
                    not member.isfile()
                    or not member.name.startswith(prefix)
                    or member.mode != 0o644
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or member.pax_headers
                ):
                    raise ValueError("archive has non-canonical USTAR metadata")
                relative = _safe_relative_path(member.name[len(prefix) :])
                if relative in files:
                    raise ValueError("archive has duplicate members")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("archive member is unreadable")
                files[relative] = extracted.read()
    except tarfile.TarError as error:
        raise ValueError("archive is not readable USTAR") from error
    if path.read_bytes() != deterministic_ustar_bytes(files):
        raise ValueError("archive is not canonical deterministic USTAR")
    return validate_expansion_substrate_evidence_files(
        files,
        expected_source_git_commit=expected_source_git_commit,
        expected_config_sha256=expected_config_sha256,
        require_canonical_attempt_identity=require_canonical_attempt_identity,
    )


def package_raw_expansion_substrate_evidence(
    *,
    raw_output_dir: str | Path,
    global_attempt_ledger: str | Path,
    output_archive: str | Path,
    source_git_commit: str,
    config_sha256: str,
    require_canonical_location: bool = True,
) -> dict[str, Any]:
    output = Path(output_archive)
    if require_canonical_location and output.resolve() != CANONICAL_ARCHIVE_PATH:
        raise ValueError("raw archive path is not canonical")
    if output.exists():
        raise FileExistsError("raw archive already exists; overwrite is forbidden")
    evidence = collect_raw_expansion_substrate_evidence(
        raw_output_dir,
        global_attempt_ledger,
        expected_source_git_commit=source_git_commit,
        expected_config_sha256=config_sha256,
        require_canonical_location=require_canonical_location,
    )
    payload = deterministic_ustar_bytes(evidence.files)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    reread = read_expansion_substrate_archive(
        output,
        expected_source_git_commit=source_git_commit,
        expected_config_sha256=config_sha256,
        require_canonical_attempt_identity=require_canonical_location,
    )
    if reread.inventory != evidence.inventory or reread.outcome != evidence.outcome:
        raise ValueError("written archive differs from raw evidence")
    return {
        "status": "PACKAGED_V2_2_LABEL_EXPANSION_SUBSTRATE_RAW_EVIDENCE",
        "source_git_commit": source_git_commit,
        "config_sha256": config_sha256,
        "outcome": evidence.outcome,
        "archive_path": str(output.resolve()),
        "archive_sha256": sha256_file(output),
        "archive_size_bytes": output.stat().st_size,
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
        "file_count": len(evidence.inventory),
        "fixed_state_denominator": evidence.fixed_state_denominator,
        "attempted_state_count": evidence.attempted_state_count,
        "completed_state_count": evidence.completed_state_count,
        "planned_counts": dict(evidence.planned_counts),
        "actual_counts": dict(evidence.actual_counts),
    }


def validate_fresh_hf_evidence(
    *,
    source_archive: str | Path,
    fresh_immutable_archive: str | Path,
    source_git_commit: str,
    config_sha256: str,
    hf_repo: str,
    hf_tag: str,
    hf_path: str,
    hf_immutable_revision: str,
) -> tuple[ExpansionSubstrateEvidence, dict[str, Any]]:
    if hf_repo != CANONICAL_HF_REPO or hf_tag != CANONICAL_HF_TAG or hf_path != CANONICAL_HF_PATH:
        raise ValueError("HF repo/tag/path differs from the frozen private destination")
    _git_sha(hf_immutable_revision, "HF immutable revision")
    source = Path(source_archive)
    fresh = Path(fresh_immutable_archive)
    if source.resolve() == fresh.resolve():
        raise ValueError("fresh HF validation requires a distinct download path")
    source_evidence = read_expansion_substrate_archive(
        source,
        expected_source_git_commit=source_git_commit,
        expected_config_sha256=config_sha256,
    )
    fresh_evidence = read_expansion_substrate_archive(
        fresh,
        expected_source_git_commit=source_git_commit,
        expected_config_sha256=config_sha256,
    )
    if source.read_bytes() != fresh.read_bytes():
        raise ValueError("fresh immutable HF archive differs byte-for-byte from source")
    if source_evidence.inventory != fresh_evidence.inventory:
        raise ValueError("fresh immutable HF evidence inventory differs")
    return source_evidence, {
        "repo": hf_repo,
        "repo_type": "dataset",
        "visibility": "private",
        "tag": hf_tag,
        "path": hf_path,
        "immutable_revision": hf_immutable_revision,
        "fresh_download_archive_sha256": sha256_file(fresh),
        "fresh_download_archive_size_bytes": fresh.stat().st_size,
        "fresh_download_byte_identical": True,
        "fresh_download_tree_inventory_sha256": fresh_evidence.tree_inventory_sha256,
        "fresh_download_file_count": len(fresh_evidence.inventory),
        "fresh_download_validated": True,
    }


def build_expansion_substrate_artifact_manifest(
    *,
    source_archive: str | Path,
    fresh_immutable_archive: str | Path,
    source_git_commit: str,
    config_sha256: str,
    hf_repo: str,
    hf_tag: str,
    hf_path: str,
    hf_immutable_revision: str,
) -> dict[str, Any]:
    evidence, hf = validate_fresh_hf_evidence(
        source_archive=source_archive,
        fresh_immutable_archive=fresh_immutable_archive,
        source_git_commit=source_git_commit,
        config_sha256=config_sha256,
        hf_repo=hf_repo,
        hf_tag=hf_tag,
        hf_path=hf_path,
        hf_immutable_revision=hf_immutable_revision,
    )
    source = Path(source_archive)
    runner_freeze = _mapping(evidence.run_contract["runner_freeze"], "runner freeze")
    derived_input = _mapping(
        evidence.run_contract["derived_artifact"], "derived artifact"
    )
    gate_summary: dict[str, Any] | None = None
    if evidence.gate_summary is not None:
        gate_metrics = dict(
            _mapping(evidence.gate_summary["metrics"], "gate metrics")
        )
        gate_metrics.pop("memory_sensitive_state_ids", None)
        gate_summary = {
            "gate_passed": evidence.gate_summary["gate_passed"],
            "derived_outcome": evidence.gate_summary["derived_outcome"],
            "metrics": gate_metrics,
            "checks": dict(evidence.gate_summary["checks"]),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": ARTIFACT_PROTOCOL_ID,
        "status": "VERIFIED_IMMUTABLE_LABEL_EXPANSION_SUBSTRATE_ARTIFACT",
        "source": {
            "runner_source_git_commit": evidence.source_git_commit,
            "execution_git_commit": evidence.execution_git_commit,
            "config_path": CANONICAL_CONFIG_PATH,
            "config_sha256": evidence.config_sha256,
            "runner_freeze_path": CANONICAL_RUNNER_FREEZE_PATH,
            "runner_freeze_sha256": runner_freeze["sha256"],
            "completion_manifest_path": CANONICAL_COMPLETION_PATH,
            "completion_manifest_sha256": evidence.completion_manifest_sha256,
            "derived_immutable_revision": evidence.derived_immutable_revision,
            "run_contract_sha256": evidence.run_contract_sha256,
        },
        "input_artifacts": {
            "derived_repo": derived_input["repo"],
            "derived_tag": derived_input["tag"],
            "derived_immutable_revision": derived_input["immutable_revision"],
            "derived_payload_prefix": derived_input["payload_prefix"],
            "derived_artifact_tree_sha256": derived_input[
                "artifact_tree_sha256"
            ],
        },
        "raw_archive": {
            "format": ARCHIVE_FORMAT,
            "member_prefix": ARCHIVE_MEMBER_PREFIX,
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
            "file_count": len(evidence.inventory),
            "tree_inventory_sha256": evidence.tree_inventory_sha256,
            "aggregate_sha256": evidence.aggregate_sha256,
        },
        "hf_artifact": hf,
        "result": {
            "outcome": evidence.outcome,
            "fixed_state_denominator": evidence.fixed_state_denominator,
            "attempted_state_count": evidence.attempted_state_count,
            "completed_state_count": evidence.completed_state_count,
            "planned_counts": dict(evidence.planned_counts),
            "actual_counts": dict(evidence.actual_counts),
            "valid_state_count": evidence.valid_state_count,
            "failed_state_count": evidence.failed_state_count,
            "failure_category_counts": dict(evidence.failure_category_counts),
            "gate_summary": gate_summary,
        },
        "provenance": {
            "runtime_identity_sha256": evidence.runtime_identity_sha256,
            "worker_identity_sha256": evidence.worker_identity_sha256,
            "log_inventory_sha256": evidence.log_inventory_sha256,
            "monitor_summary_sha256": evidence.monitor_summary_sha256,
        },
        "negative_declarations": {
            "native_generation_text_copied_into_git_manifest": False,
            "state_records_copied_into_git_manifest": False,
            "failure_messages_copied_into_git_manifest": False,
            "retry_performed": False,
            "top_up_performed": False,
            "confirm_accessed": False,
            "restoration_labels_generated": False,
            "gate_training_started": False,
            "matched_nll_started": False,
            "closed_loop_started": False,
        },
    }


__all__ = [
    "AGGREGATE_FILENAME",
    "ARCHIVE_MEMBER_PREFIX",
    "ARTIFACT_PROTOCOL_ID",
    "ATTEMPT_DIRECTORY",
    "ATTEMPT_ID",
    "CANARY_ABORT_FILENAME",
    "CANARY_RELEASE_FILENAME",
    "CANONICAL_ARCHIVE_PATH",
    "CANONICAL_HF_PATH",
    "CANONICAL_HF_REPO",
    "CANONICAL_HF_TAG",
    "CANONICAL_GIT_ORIGIN_URL",
    "CANONICAL_LEDGER_PATH",
    "CANONICAL_OUTPUT_DIR",
    "CANONICAL_RUNNER_FREEZE_PATH",
    "EXECUTION_EVIDENCE_FILENAME",
    "ExpansionSubstrateEvidence",
    "EXPECTED_STATE_COUNT",
    "FORBIDDEN_COUNT_KEYS",
    "GLOBAL_LEDGER_ARCHIVE_NAME",
    "INVALID_OUTCOME",
    "LOG_PATHS",
    "MONITOR_SUMMARY_PATH",
    "NO_GO_OUTCOME",
    "PASS_OUTCOME",
    "PER_STATE_PLANNED_COUNTS",
    "PLANNED_COUNTS",
    "PROCESSOR_IDENTITY_KEYS",
    "PROCESSOR_IDENTITY_FIELDS",
    "RUNNER_FREEZE_PROTOCOL_ID",
    "RUNNER_FREEZE_SCHEMA_VERSION",
    "RUNTIME_BARRIER_ABORT_FILENAME",
    "RUNTIME_BARRIER_RELEASE_FILENAME",
    "RUN_MANIFEST_FILENAME",
    "SCHEMA_VERSION",
    "STATE_DIRECTORY",
    "STATE_OUTCOME_FAILED",
    "STATE_OUTCOME_VALID",
    "WORKER_CANARY_FILENAME",
    "WORKER_DIRECTORY",
    "WORKER_LEDGER_FILENAME",
    "WORKER_OUTCOME_COMPLETE",
    "WORKER_OUTCOME_INVALID",
    "WORKER_RUNTIME_FILENAME",
    "WORKER_SIBLING_LEDGER_DIRECTORY",
    "WORKER_TERMINAL_FILENAME",
    "WorkerSpec",
    "build_expansion_substrate_artifact_manifest",
    "build_runner_freeze",
    "canonical_json_bytes",
    "collect_raw_expansion_substrate_evidence",
    "deterministic_ustar_bytes",
    "expected_worker_specs",
    "load_and_validate_runner_freeze",
    "materialize_runner_freeze",
    "package_raw_expansion_substrate_evidence",
    "pretty_json_bytes",
    "read_expansion_substrate_archive",
    "reduce_expansion_substrate_gate",
    "sha256_bytes",
    "sha256_file",
    "strict_json_object_bytes",
    "validate_execution_run_contract",
    "validate_expansion_substrate_evidence_files",
    "validate_fresh_hf_evidence",
    "validate_runner_freeze_data",
]
