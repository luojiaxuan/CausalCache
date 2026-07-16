"""Evidence validation and deterministic packaging for restoration-v2.2 eager."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tarfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.policy.gui_owl_v2 import gui_owl_v2_action_to_androidworld
from causalcache.policy.gui_owl_v2_1 import parse_gui_owl_v2_1_output
from causalcache.restoration_v2_2_eager_contract import (
    CANONICAL_ARCHIVE_PATH,
    CANONICAL_ATTEMPT_ID,
    CANONICAL_HF_PATH,
    CANONICAL_HF_REPO,
    CANONICAL_HF_TAG,
    CANONICAL_LEDGER_PATH,
    CANONICAL_OUTPUT_DIR,
    CANONICAL_CONFIG_PATH,
    FROZEN_RESTORATION_V2_2_EAGER_SHA256,
    INVALID_OUTCOME,
    NO_GO_OUTCOME,
    PASS_OUTCOME,
    PROTOCOL_ID,
)
from scripts.run_restoration_v2_1_full_45_substrate import (
    _action_arguments,
    _duration_seconds,
    _state_record_keys,
)
from scripts.run_restoration_v2_1_interface_pilot import (
    RUNTIME_GENERATION_BINDING_KEYS,
    _validate_generation_metadata,
)


SCHEMA_VERSION = "1.0.0"
MEASUREMENT_KERNEL_PROTOCOL_ID = "causalcache_restoration_v2_1_full_45_substrate"
MEASUREMENT_KERNEL_SCHEMA_VERSION = "1.0.0"
RUNTIME_PROFILE_ID = "causalcache_restoration_v2_2_eager_runtime"
ARCHIVE_FORMAT = "ustar"
ARCHIVE_MEMBER_PREFIX = CANONICAL_ATTEMPT_ID
ARCHIVE_LEDGER_NAME = "global_attempt_ledger.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"
AGGREGATE_FILENAME = "aggregate.json"
WORKER_DIRECTORY = "workers"
WORKER_SIBLING_LEDGER_DIRECTORY = "worker_sibling_ledgers"
WORKER_LEDGER_FILENAME = "worker_attempt_ledger.json"
WORKER_RUNTIME_FILENAME = "runtime_identity.json"
WORKER_TERMINAL_FILENAME = "terminal.json"
RUNTIME_BARRIER_RELEASE_FILENAME = "runtime_barrier_release.json"
RUNTIME_BARRIER_ABORT_FILENAME = "runtime_barrier_abort.json"
STATE_DIRECTORY = "states"
ATTEMPT_DIRECTORY = "attempts"
EXPECTED_STATE_COUNT = 45
REPEAT_NOISE_FLOOR = 1e-4
REPEAT_NOISE_MULTIPLIER = 10.0
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
BFLOAT16_MIN_FINITE = -3.3895313892515355e38


def _expected_standard_eos_token_ids() -> tuple[int, ...]:
    from causalcache.policy.gui_owl_v2_1_runtime import (
        GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS,
    )

    return GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS

STATE_OUTCOME_VALID = "VALID_V2_2_EAGER_FULL_45_SUBSTRATE_STATE"
STATE_OUTCOME_FAILED = "FAILED_V2_2_EAGER_FULL_45_SUBSTRATE_STATE"
WORKER_OUTCOME_COMPLETE = "COMPLETED_V2_2_EAGER_WORKER_SHARD"
WORKER_OUTCOME_INVALID = "INVALID_V2_2_EAGER_WORKER_SHARD"

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
    "gate_model_forward_count": 0,
    "gate_selection_count": 0,
}

EXPECTED_SOURCE_INVENTORY_PATHS = (
    "code/configs/causalcache_restoration_v2_2_eager.json",
    "code/causalcache/restoration_v2_2_eager_contract.py",
    "code/causalcache/policy/gui_owl_v2_2_eager_runtime.py",
    "code/causalcache/restoration_v2_2_eager_artifact.py",
    "code/scripts/run_restoration_v2_2_eager_substrate.py",
    "code/scripts/manage_restoration_v2_2_eager_artifact.py",
    "code/scripts/validate_restoration_v2_2_eager_contract.py",
    "code/configs/causalcache_restoration_v2_1_full_45.json",
    "code/causalcache/restoration_v2_1_full_45_contract.py",
    "code/causalcache/restoration_v2_1_full_45_artifact.py",
    "code/scripts/run_restoration_v2_1_full_45_substrate.py",
    "code/scripts/manage_restoration_v2_1_full_45_artifact.py",
    "code/scripts/validate_restoration_v2_1_full_45_contract.py",
    "code/scripts/run_restoration_v2_1_interface_pilot.py",
    "code/causalcache/restoration_v2_1_contract.py",
    "code/causalcache/restoration_v2_1_pilot_artifact.py",
    "code/scripts/manage_restoration_v2_1_pilot_artifact.py",
    "code/causalcache/restoration_v2_1_processor_audit.py",
    "code/scripts/audit_gui_owl_v2_1_processor.py",
    "code/causalcache/data/restoration_v2_1_processor_inputs.py",
    "code/causalcache/data/restoration_v2_screening.py",
    "code/causalcache/policy/gui_owl_v2_1.py",
    "code/causalcache/policy/gui_owl_v2_1_runtime.py",
    "code/causalcache/restoration_v2_gpu_kl.py",
    "code/causalcache/data/guiodyssey_restoration_v2.py",
    "code/causalcache/data/restoration_v2_selection.py",
    "code/causalcache/data/guiodyssey.py",
    "code/causalcache/data/guiodyssey_independent.py",
    "code/causalcache/low_fidelity_v2.py",
    "code/causalcache/restoration_v2_contract.py",
    "code/causalcache/restoration_v2_text_backend.py",
    "code/causalcache/schema.py",
    "code/causalcache/policy/gui_owl_v2.py",
    "code/causalcache/policy/gui_owl_v2_runtime.py",
    "code/causalcache/policy/gui_owl_v2_vision.py",
    "code/causalcache/restoration_v2_baselines.py",
    "code/configs/causalcache_restoration_v2_1_pilot.json",
    "code/configs/causalcache_restoration_v2.json",
    "code/configs/restoration_v2_ocr_backend.json",
    "code/configs/gui_owl_1_5_8b_snapshot.json",
    "data/manifests/restoration_v2_selection.json",
    "data/manifests/restoration_v2_ocr_backend.json",
    "data/manifests/restoration_v2_derived_artifact.json",
    "data/manifests/restoration_v2_real_screen_source.json",
    "data/results/restoration_v2_1_interface_pilot/artifact.json",
    "data/results/restoration_v2_1_processor_preflight/artifact.json",
    "data/results/restoration_v2_1_full_45_substrate/artifact.json",
    "data/results/restoration_v2_1_full_45_substrate/summary.json",
    "data/results/spatial_reference_audit_v1/artifact.json",
    "data/results/spatial_reference_audit_v1/summary.json",
)


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


@dataclass(frozen=True)
class V22Evidence:
    files: Mapping[str, bytes]
    run_contract: Mapping[str, Any]
    source_git_commit: str
    run_contract_sha256: str
    outcome: str
    aggregate_sha256: str
    completed_state_count: int
    attempted_state_count: int
    generation_call_count: int
    teacher_forward_count: int
    kl_measurement_count: int
    inventory: tuple[Mapping[str, Any], ...]
    tree_inventory_sha256: str


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def expected_worker_specs() -> tuple[WorkerSpec, WorkerSpec]:
    return (
        WorkerSpec("even", "cuda:0", 0, tuple(range(0, EXPECTED_STATE_COUNT, 2))),
        WorkerSpec("odd", "cuda:1", 1, tuple(range(1, EXPECTED_STATE_COUNT, 2))),
    )


def validate_worker_topology(value: Any) -> tuple[WorkerSpec, WorkerSpec]:
    if not isinstance(value, Mapping):
        raise ValueError("worker topology must be a JSON object")
    expected_keys = {
        "worker_count",
        "gpu_model",
        "one_process_per_device",
        "workers",
        "cross_worker_state_stealing_allowed",
        "worker_failure_invalidates_entire_attempt",
        "worker_outputs_must_be_disjoint",
        "worker_union_must_equal_fixed_denominator",
    }
    if set(value) != expected_keys:
        raise ValueError("worker topology keys drifted")
    workers = value.get("workers")
    specs = expected_worker_specs()
    if (
        value.get("worker_count") != 2
        or value.get("gpu_model") != "NVIDIA H200"
        or value.get("one_process_per_device") is not True
        or value.get("cross_worker_state_stealing_allowed") is not False
        or value.get("worker_failure_invalidates_entire_attempt") is not True
        or value.get("worker_outputs_must_be_disjoint") is not True
        or value.get("worker_union_must_equal_fixed_denominator") is not True
        or not isinstance(workers, list)
        or workers != [spec.to_dict() for spec in specs]
        or specs[0].device == specs[1].device
    ):
        raise ValueError("worker topology is not the exact two-device parity split")
    return specs


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError("evidence path is not canonical relative POSIX")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys drifted")


def _validate_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be one UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be one valid UTC timestamp") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{label} must be UTC")
    return value


def _validate_run_contract_identity(
    run_contract: Mapping[str, Any], *, require_canonical_attempt_identity: bool
) -> None:
    _exact_keys(
        run_contract,
        {
            "schema_version",
            "protocol_id",
            "contract_source",
            "git_identity",
            "source_inventory",
            "parent_authorization",
            "canonical_inputs",
            "runtime_requirements",
            "worker_topology",
            "states",
            "gate",
            "operation_policy",
            "attempt_identity",
            "execution_argv",
        },
        "v2.2 run contract",
    )
    if run_contract.get("schema_version") != SCHEMA_VERSION or run_contract.get(
        "protocol_id"
    ) != PROTOCOL_ID:
        raise ValueError("v2.2 run contract protocol drifted")
    if run_contract.get("contract_source") != {
        "path": CANONICAL_CONFIG_PATH,
        "sha256": FROZEN_RESTORATION_V2_2_EAGER_SHA256,
    }:
        raise ValueError("v2.2 run contract does not bind the frozen config")
    git = _mapping(run_contract.get("git_identity"), "git identity")
    _exact_keys(
        git,
        {"branch", "commit", "origin_main", "remote_main", "remote_url", "worktree"},
        "git identity",
    )
    commit = git.get("commit")
    if (
        not isinstance(commit, str)
        or GIT_SHA_PATTERN.fullmatch(commit) is None
        or git.get("branch") != "main"
        or git.get("origin_main") != commit
        or git.get("remote_main") != commit
        or git.get("remote_url") != "https://github.com/luojiaxuan/CausalCache.git"
        or git.get("worktree") != "clean_including_untracked"
    ):
        raise ValueError("v2.2 run contract Git identity drifted")
    inventory = _sequence(run_contract.get("source_inventory"), "source inventory")
    if not inventory:
        raise ValueError("v2.2 run contract lacks a source inventory")
    paths: list[str] = []
    for item in inventory:
        record = _mapping(item, "source inventory record")
        _exact_keys(record, {"path", "sha256", "git_commit"}, "source inventory record")
        path = record.get("path")
        digest = record.get("sha256")
        if (
            not isinstance(path, str)
            or _safe_relative_path(path) != path
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            or record.get("git_commit") != commit
            or path in paths
        ):
            raise ValueError("v2.2 source inventory record drifted")
        paths.append(path)
    if tuple(paths) != EXPECTED_SOURCE_INVENTORY_PATHS:
        raise ValueError("v2.2 run contract source inventory path/order drifted")
    parents = _mapping(run_contract.get("parent_authorization"), "parent authorization")
    expected_parent_keys = {
        "spatial_reference_evidence",
        "v2_1_full_45_evidence",
        "pilot_evidence",
        "processor_evidence",
    }
    if set(parents) != expected_parent_keys:
        raise ValueError("v2.2 fresh parent authorization inventory drifted")
    for name in sorted(expected_parent_keys):
        record = _mapping(parents[name], f"{name} authorization")
        _exact_keys(
            record,
            {
                "path",
                "sha256",
                "size_bytes",
                "validation_status",
                "validation_outcome",
            },
            f"{name} authorization",
        )
        if (
            not isinstance(record.get("path"), str)
            or not Path(record["path"]).is_absolute()
            or not isinstance(record.get("sha256"), str)
            or SHA256_PATTERN.fullmatch(record["sha256"]) is None
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] <= 0
            or not isinstance(record.get("validation_status"), str)
            or not record["validation_status"]
            or not isinstance(record.get("validation_outcome"), str)
            or not record["validation_outcome"]
        ):
            raise ValueError("v2.2 fresh parent authorization binding drifted")
    expected_parent_science = {
        "spatial_reference_evidence": {
            "sha256": "d62ad05f6fdef06a3551f2ebe9f83f28327068f0020ff3e61891da4f46ce5ecc",
            "size_bytes": 1873920,
            "validation_status": "VALIDATED_CANONICAL_SPATIAL_REFERENCE_RAW_ARCHIVE",
            "validation_outcome": "EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY",
        },
        "v2_1_full_45_evidence": {
            "sha256": "8cd53d6e56d5ad509da2af91d73d9e83b4db989ffc26aa18e1bca84e4c4f4fa4",
            "size_bytes": 962560,
            "validation_status": "VALID_RESTORATION_V2_1_FULL_45_ARTIFACT",
            "validation_outcome": "NO_GO_V2_1_FULL_45_SUBSTRATE",
        },
        "pilot_evidence": {
            "sha256": "f71d5fd575dde48ae8b3e50a19dd2fbecfa02d7d5ae6087f909a47dfd7032064",
            "size_bytes": 133120,
            "validation_status": "VALID_RESTORATION_V2_1_INTERFACE_PILOT_ARTIFACT",
            "validation_outcome": "PASS_V2_1_INTERFACE_PILOT",
        },
        "processor_evidence": {
            "sha256": "5349ffc6b91bf93ed26d25104fe6c907f31ccc497007a5c0ae6ed5ddf5c84991",
            "size_bytes": 7609803,
            "validation_status": "PASSED_RESTORATION_V2_1_90_PROMPT_PROCESSOR_PREFLIGHT",
            "validation_outcome": "PASSED_90_PROMPT_PROCESSOR_PREFLIGHT",
        },
    }
    for name, expected in expected_parent_science.items():
        if any(parents[name].get(key) != value for key, value in expected.items()):
            raise ValueError(f"v2.2 parent authorization {name} conclusion drifted")
    canonical_inputs = _mapping(run_contract.get("canonical_inputs"), "canonical inputs")
    required_inputs = {
        "scientific_config",
        "selection_manifest",
        "ocr_backend_config",
        "snapshot_manifest",
        "derived_artifact",
    }
    if set(canonical_inputs) != required_inputs:
        raise ValueError("v2.2 canonical input inventory drifted")
    for name in required_inputs - {"derived_artifact"}:
        record = _mapping(canonical_inputs[name], f"canonical input {name}")
        _exact_keys(record, {"path", "sha256"}, f"canonical input {name}")
        if (
            not isinstance(record.get("path"), str)
            or _safe_relative_path(record["path"]) != record["path"]
            or not isinstance(record.get("sha256"), str)
            or SHA256_PATTERN.fullmatch(record["sha256"]) is None
        ):
            raise ValueError("v2.2 canonical input binding drifted")
    derived = _mapping(canonical_inputs["derived_artifact"], "derived artifact")
    if derived != {
        "repo": "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
        "immutable_revision": "89f136abaff797e14fe758a198996e51032a10a6",
        "artifact_tree_sha256": (
            "475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e"
        ),
    }:
        raise ValueError("v2.2 derived artifact binding drifted")
    requirements = _mapping(run_contract.get("runtime_requirements"), "runtime requirements")
    if requirements != {
        "gpu_name": "NVIDIA H200",
        "container_image_digest": (
            "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
        ),
        "python_version": "3.12.3",
        "torch_version": "2.11.0+cu130",
        "torch_cuda_version": "13.0",
        "cudnn_version": 91900,
        "transformers_version": "5.6.0",
        "nvidia_driver_version": "570.172.08",
        "runtime_profile_id": RUNTIME_PROFILE_ID,
        "attention_implementation": "eager",
    }:
        raise ValueError("v2.2 runtime requirements drifted")
    operation_policy = _mapping(run_contract.get("operation_policy"), "operation policy")
    if operation_policy != {
        "retry_count": 0,
        "top_up_count": 0,
        "v2_1_raw_state_reuse_count": 0,
        "same_state_repeats_must_remain_on_one_worker_device": True,
        "merge_only_after_both_worker_terminals": True,
        "cpu_recompute_gate_after_index_order_merge": True,
        "resume_allowed": False,
        "interrupted_attempt_is_terminal_invalid": True,
        "resume_or_top_up_count": 0,
    }:
        raise ValueError("v2.2 operation policy drifted")
    attempt = _mapping(run_contract.get("attempt_identity"), "attempt identity")
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
    if (
        attempt.get("attempt_id") != CANONICAL_ATTEMPT_ID
        or require_canonical_attempt_identity
        and attempt.get("output_dir") != str(CANONICAL_OUTPUT_DIR)
        or require_canonical_attempt_identity
        and attempt.get("global_ledger") != str(CANONICAL_LEDGER_PATH)
        or not require_canonical_attempt_identity
        and (
            not isinstance(attempt.get("output_dir"), str)
            or not Path(attempt["output_dir"]).is_absolute()
            or not isinstance(attempt.get("global_ledger"), str)
            or not Path(attempt["global_ledger"]).is_absolute()
        )
        or attempt.get("container_image_digest")
        != "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
        or not all(
            isinstance(attempt.get(key), str) and attempt.get(key)
            for key in ("host_alias", "host_hostname", "container_id")
        )
    ):
        raise ValueError("v2.2 attempt identity drifted")
    argv = _sequence(run_contract.get("execution_argv"), "execution argv")
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ValueError("v2.2 execution argv is invalid")


def validate_v22_run_contract_identity(
    run_contract: Mapping[str, Any], *, require_canonical_attempt_identity: bool
) -> None:
    _validate_run_contract_identity(
        run_contract,
        require_canonical_attempt_identity=require_canonical_attempt_identity,
    )


def _contract_state_projections(run_contract: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    states = _sequence(run_contract.get("states"), "run contract states")
    if len(states) != EXPECTED_STATE_COUNT:
        raise ValueError("run contract does not contain exactly 45 states")
    projections: list[dict[str, Any]] = []
    for index, raw in enumerate(states):
        state = dict(_mapping(raw, f"run contract state {index}"))
        if state.get("index") != index or not isinstance(state.get("state_id"), str):
            raise ValueError("run contract state order or identity drifted")
        projections.append(state)
    return tuple(projections)


def _validate_runtime_metadata(metadata: Any, *, spec: WorkerSpec) -> Mapping[str, Any]:
    value = _mapping(metadata, f"{spec.worker_id} runtime metadata")
    if (
        value.get("protocol_id")
        != "causalcache_restoration_v2_1_official_tool_interface"
        or value.get("runtime_profile_id") != RUNTIME_PROFILE_ID
        or value.get("device") != spec.device
        or value.get("requested_attention_implementation") != "eager"
        or value.get("observed_attention_implementation")
        != {"top": "eager", "text": "eager", "vision": "eager"}
        or value.get("seed") != 0
        or value.get("cudnn_deterministic") is not True
        or value.get("cudnn_benchmark") is not False
        or value.get("cuda_matmul_allow_tf32") is not False
        or value.get("cudnn_allow_tf32") is not False
        or value.get("float32_matmul_precision") != "highest"
        or value.get("deterministic_algorithms_requested") is not False
        or value.get("deterministic_algorithms_enabled") is not False
        or value.get("strict_cuda_determinism_claimed") is not False
        or value.get("gpu_name") != "NVIDIA H200"
        or value.get("logical_device_index") != spec.index_parity
        or type(value.get("nvidia_smi_index")) is not int
        or value.get("nvidia_smi_index") < 0
        or not isinstance(value.get("gpu_uuid"), str)
        or not value.get("gpu_uuid")
        or not isinstance(value.get("gpu_pci_bus_id"), str)
        or not value.get("gpu_pci_bus_id")
        or value.get("container_image_digest")
        != "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
        or value.get("python_version") != "3.12.3"
        or value.get("torch_version") != "2.11.0+cu130"
        or value.get("torch_cuda_version") != "13.0"
        or value.get("cudnn_version") != 91900
        or value.get("transformers_version") != "5.6.0"
        or value.get("nvidia_driver_version") != "570.172.08"
    ):
        raise ValueError(f"{spec.worker_id} eager runtime metadata drifted")
    return value


def _validate_runtime_pair(
    runtimes: Mapping[str, Mapping[str, Any]],
) -> None:
    specs = expected_worker_specs()
    first = runtimes[specs[0].worker_id]
    second = runtimes[specs[1].worker_id]
    allowed = {
        "device",
        "gpu_uuid",
        "gpu_pci_bus_id",
        "logical_device_index",
        "nvidia_smi_index",
    }
    first_scientific = {key: value for key, value in first.items() if key not in allowed}
    second_scientific = {key: value for key, value in second.items() if key not in allowed}
    if first_scientific != second_scientific:
        raise ValueError("worker scientific runtime metadata differs")
    if (
        first.get("device") == second.get("device")
        or first.get("gpu_uuid") == second.get("gpu_uuid")
        or first.get("gpu_pci_bus_id") == second.get("gpu_pci_bus_id")
        or first.get("logical_device_index") == second.get("logical_device_index")
        or first.get("nvidia_smi_index") == second.get("nvidia_smi_index")
    ):
        raise ValueError("workers did not bind two distinct physical devices")


def validate_worker_runtime_metadata(
    metadata: Mapping[str, Any], *, spec: WorkerSpec
) -> Mapping[str, Any]:
    return _validate_runtime_metadata(metadata, spec=spec)


def validate_worker_runtime_pair(
    runtimes: Mapping[str, Mapping[str, Any]],
) -> None:
    specs = expected_worker_specs()
    if set(runtimes) != {spec.worker_id for spec in specs}:
        raise ValueError("runtime pair worker inventory drifted")
    for spec in specs:
        _validate_runtime_metadata(runtimes[spec.worker_id], spec=spec)
    _validate_runtime_pair(runtimes)


def _validate_inner_device(inner: Mapping[str, Any], *, spec: WorkerSpec) -> None:
    generations = _sequence(inner.get("native_generations"), "native generations")
    if len(generations) != 2:
        raise ValueError("measurement kernel must preserve exactly two generations")
    for generation in generations:
        metadata = _mapping(
            _mapping(generation, "generation record").get("metadata"),
            "generation metadata",
        )
        if metadata.get("device") != spec.device:
            raise ValueError("state generation ran on the wrong worker device")
    teachers = _mapping(inner.get("teacher_forwards"), "teacher forwards")
    for metadata in teachers.values():
        if _mapping(metadata, "teacher metadata").get("device") != spec.device:
            raise ValueError("state teacher forward ran on the wrong worker device")


def _validate_teacher_metadata_v22(
    metadata: Any,
    *,
    spec: WorkerSpec,
    runtime_metadata: Mapping[str, Any] | None,
) -> None:
    teacher = _mapping(metadata, "teacher metadata")
    required = {
        "batch_size",
        "device",
        "dtype",
        "vocabulary_size",
        "distance_span",
        "teacher_context",
        "teacher_carrier",
        "teacher_target_json_separators",
        "teacher_target_ends_with_model_generation_eos",
        "teacher_target_disjoint_from_suppressed_standard_eos",
        "teacher_standard_eos_suppressed_token_ids",
        "teacher_standard_eos_suppression_value",
        "teacher_standard_eos_suppression_semantics",
        "teacher_standard_eos_mask_application",
        "teacher_raw_logits_mutated",
        "finite_logits_validation",
        "logits_to_keep",
        "full_logit_tensor_host_transfers",
        "samples",
    }
    if not required.issubset(teacher):
        raise ValueError("teacher suppression metadata is incomplete")
    suppressed_ids = list(_expected_standard_eos_token_ids())
    samples = teacher.get("samples")
    if (
        teacher.get("batch_size") != 1
        or teacher.get("device") != spec.device
        or teacher.get("dtype") != "torch.bfloat16"
        or type(teacher.get("vocabulary_size")) is not int
        or teacher["vocabulary_size"] <= max(suppressed_ids)
        or teacher.get("distance_span")
        != "official_tool_call_open_through_close_inclusive"
        or teacher.get("teacher_context")
        != "official_tools_prompt_plus_assistant_prefix_direct"
        or teacher.get("teacher_carrier") is not None
        or teacher.get("teacher_target_json_separators") != [", ", ": "]
        or teacher.get("teacher_target_ends_with_model_generation_eos") is not True
        or teacher.get("teacher_target_disjoint_from_suppressed_standard_eos")
        is not True
        or teacher.get("teacher_standard_eos_suppressed_token_ids") != suppressed_ids
        or teacher.get("teacher_standard_eos_suppression_value")
        != BFLOAT16_MIN_FINITE
        or teacher.get("teacher_standard_eos_suppression_semantics")
        != "torch_finfo_bfloat16_min_finite_generation_alignment"
        or teacher.get("teacher_standard_eos_mask_application")
        != (
            "same_mask_on_every_reference_and_candidate_action_path_position_"
            "before_float32_log_softmax"
        )
        or teacher.get("teacher_raw_logits_mutated") is not False
        or teacher.get("finite_logits_validation")
        != "deferred_to_gpu_kl_invalid_to_nan_final_distance"
        or type(teacher.get("logits_to_keep")) is not int
        or teacher["logits_to_keep"] <= 0
        or teacher.get("full_logit_tensor_host_transfers") != 0
        or not isinstance(samples, list)
        or len(samples) != 1
        or not isinstance(samples[0], Mapping)
        or samples[0].get("distance_action_tokens") != teacher["logits_to_keep"]
    ):
        raise ValueError("teacher metadata differs from the frozen measurement kernel")
    if runtime_metadata is not None:
        if any(
            teacher.get(key) != runtime_metadata.get(key)
            for key in RUNTIME_GENERATION_BINDING_KEYS
        ):
            raise ValueError("teacher metadata differs from persisted worker runtime")
        if runtime_metadata.get("suppressed_standard_eos_token_ids") != suppressed_ids:
            raise ValueError("teacher suppression IDs differ from worker runtime")


def _validate_distance_audit_v22(
    value: Any, *, teacher: Mapping[str, Any], spec: WorkerSpec
) -> None:
    audit = _mapping(value, "distance audit")
    expected_keys = {
        "operation",
        "candidate_representation",
        "batch_size",
        "distance_tokens",
        "vocabulary_size",
        "device",
        "reference_input_dtype",
        "candidate_input_dtype",
        "compute_dtype",
        "output_dtype",
        "reference_batch_stride",
        "reference_zero_copy_batch_expansion",
        "reference_compute_batch_size",
        "reduction",
        "log_normalization_atol",
        "negative_kl_atol",
        "numeric_validation",
        "invalid_numeric_output",
        "device_validation_category_count",
        "validation_scalar_host_reads",
        "full_tensor_host_transfers",
    }
    _exact_keys(audit, expected_keys, "distance audit")
    if (
        audit.get("operation")
        != "teacher_forced_full_vocabulary_mean_kl_on_distance_token_span"
        or audit.get("candidate_representation") != "logits"
        or audit.get("batch_size") != 1
        or audit.get("distance_tokens") != teacher.get("logits_to_keep")
        or audit.get("vocabulary_size") != teacher.get("vocabulary_size")
        or audit.get("device") != spec.device
        or audit.get("device") != teacher.get("device")
        or audit.get("reference_input_dtype") != "torch.float32"
        or audit.get("candidate_input_dtype") != "torch.bfloat16"
        or audit.get("compute_dtype") != "torch.float32"
        or audit.get("output_dtype") != "torch.float32"
        or type(audit.get("reference_batch_stride")) is not int
        or audit["reference_batch_stride"] <= 0
        or audit.get("reference_zero_copy_batch_expansion") is not False
        or audit.get("reference_compute_batch_size") != 1
        or audit.get("reduction")
        != "full_vocabulary_sum_then_distance_token_mean_per_example"
        or audit.get("log_normalization_atol") != 5e-4
        or audit.get("negative_kl_atol") != 1e-5
        or audit.get("numeric_validation") != "gpu_resident_per_example_predicates"
        or audit.get("invalid_numeric_output") != "nan_final_distance"
        or audit.get("device_validation_category_count") != 4
        or audit.get("validation_scalar_host_reads") != 0
        or audit.get("full_tensor_host_transfers") != 0
    ):
        raise ValueError("distance audit differs from the frozen GPU KL path")


def _validate_measurement_kernel_record(
    inner: Mapping[str, Any],
    *,
    spec: WorkerSpec,
    projection: Mapping[str, Any],
    run_contract_sha256: str,
    runtime_metadata: Mapping[str, Any] | None,
) -> None:
    if set(inner) != _state_record_keys():
        raise ValueError("measurement-kernel state keys drifted")
    if (
        inner.get("schema_version") != MEASUREMENT_KERNEL_SCHEMA_VERSION
        or inner.get("protocol_id") != MEASUREMENT_KERNEL_PROTOCOL_ID
        or inner.get("run_contract_sha256") != run_contract_sha256
        or inner.get("state") != dict(projection)
        or inner.get("outcome")
        not in {
            "VALID_V2_1_FULL_45_SUBSTRATE_STATE",
            "FAILED_V2_1_FULL_45_SUBSTRATE_STATE",
        }
    ):
        raise ValueError("measurement-kernel state identity drifted")
    dimensions = _mapping(inner.get("screen_dimensions"), "screen dimensions")
    if set(dimensions) != {"width", "height"}:
        raise ValueError("screen dimension schema drifted")
    width = dimensions.get("width")
    height = dimensions.get("height")
    if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
        raise ValueError("screen dimensions are invalid")
    operations = _mapping(inner.get("operation_counts"), "operation counts")
    old_forbidden = set(FORBIDDEN_OPERATION_COUNTS) - {
        "gate_model_forward_count",
        "gate_selection_count",
    }
    if set(operations) != {
        "generation_call_count",
        "teacher_forward_count",
        "kl_measurement_count",
        *old_forbidden,
    }:
        raise ValueError("measurement-kernel operation keys drifted")
    if (
        operations.get("generation_call_count") != 2
        or operations.get("teacher_forward_count") not in {0, 3}
        or operations.get("kl_measurement_count") not in {0, 2}
        or any(operations.get(key) != 0 for key in old_forbidden)
    ):
        raise ValueError("measurement-kernel operation schedule drifted")
    native = _sequence(inner.get("native_generations"), "native generations")
    if len(native) != 2:
        raise ValueError("measurement kernel must preserve two generations")
    parsed_actions: list[dict[str, Any]] = []
    parsed_bridges: list[dict[str, Any]] = []
    closer_count = 0
    for repeat_index, raw_generation in enumerate(native, start=1):
        generation = _mapping(raw_generation, "generation record")
        _exact_keys(
            generation,
            {
                "repeat_index",
                "output_text",
                "metadata",
                "canonical_action",
                "androidworld_bridge",
                "parse_error_type",
                "parse_error_message",
            },
            "generation record",
        )
        output = generation.get("output_text")
        metadata = _mapping(generation.get("metadata"), "generation metadata")
        if generation.get("repeat_index") != repeat_index or not isinstance(output, str):
            raise ValueError("generation repeat identity drifted")
        _validate_generation_metadata(metadata, raw_output=output)
        if metadata.get("device") != spec.device:
            raise ValueError("generation metadata device differs from worker")
        if runtime_metadata is not None and any(
            metadata.get(key) != runtime_metadata.get(key)
            for key in RUNTIME_GENERATION_BINDING_KEYS
        ):
            raise ValueError("generation metadata differs from persisted worker runtime")
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
            screen_width=int(width),
            screen_height=int(height),
        )
        if (
            metadata.get("model_emitted_tool_call_close") is not True
            or metadata.get("generated_tool_call_close_token_count") != 1
            or metadata.get("final_generated_token_id") != 151658
            or metadata.get("termination_reason")
            != "model_emitted_tool_call_close"
            or not output.endswith("</tool_call>")
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
        inner.get("parse_success_count") != parse_count
        or inner.get("parse_success") is not parse_success
        or inner.get("model_emitted_closer_count") != closer_count
        or inner.get("androidworld_bridge_count") != parse_count
        or inner.get("repeat_canonical_action_agreement") is not agreement
    ):
        raise ValueError("recomputed parse or repeat metrics drifted")
    teacher = _mapping(inner.get("teacher_forwards"), "teacher forwards")
    distances = _mapping(inner.get("distances"), "distances")
    audits = _mapping(inner.get("distance_audits"), "distance audits")
    if set(distances) != {"repeat_reference_kl", "summary_reference_kl"}:
        raise ValueError("distance schema drifted")
    failure = inner.get("failure")
    finite = inner.get("finite_logit_distances") is True
    if not agreement:
        if (
            operations["teacher_forward_count"] != 0
            or operations["kl_measurement_count"] != 0
            or teacher
            or audits
            or any(value is not None for value in distances.values())
            or inner.get("canonical_action") is not None
            or inner.get("androidworld_bridge") is not None
            or finite
            or inner.get("outcome") != "FAILED_V2_1_FULL_45_SUBSTRATE_STATE"
            or not isinstance(failure, Mapping)
            or failure.get("category")
            not in {"PARSE_FAILURE", "CANONICAL_ACTION_MISMATCH"}
        ):
            raise ValueError("pre-teacher failure evidence drifted")
    else:
        if (
            operations["teacher_forward_count"] != 3
            or operations["kl_measurement_count"] != 2
            or set(teacher) != {"reference_1", "reference_2", "summary_only"}
            or set(audits) != {"repeat_reference_kl", "summary_reference_kl"}
            or inner.get("canonical_action") != parsed_actions[0]
            or inner.get("androidworld_bridge") != parsed_bridges[0]
        ):
            raise ValueError("post-agreement teacher evidence drifted")
        for teacher_metadata in teacher.values():
            _validate_teacher_metadata_v22(
                teacher_metadata,
                spec=spec,
                runtime_metadata=runtime_metadata,
            )
        _validate_distance_audit_v22(
            audits["repeat_reference_kl"],
            teacher=_mapping(teacher["reference_2"], "repeat teacher"),
            spec=spec,
        )
        _validate_distance_audit_v22(
            audits["summary_reference_kl"],
            teacher=_mapping(teacher["summary_only"], "summary teacher"),
            spec=spec,
        )
        finite_values = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in distances.values()
        )
        if finite is not finite_values:
            raise ValueError("finite flag differs from persisted distances")
        if finite:
            if (
                inner.get("outcome") != "VALID_V2_1_FULL_45_SUBSTRATE_STATE"
                or failure is not None
            ):
                raise ValueError("finite measurement record is not valid")
        elif (
            inner.get("outcome") != "FAILED_V2_1_FULL_45_SUBSTRATE_STATE"
            or not isinstance(failure, Mapping)
            or failure.get("category") != "NONFINITE_DISTANCE"
            or all(value is not None for value in distances.values())
        ):
            raise ValueError("nonfinite measurement failure evidence drifted")
    started_at = inner.get("started_at_utc")
    ended_at = inner.get("ended_at_utc")
    duration = inner.get("duration_seconds")
    if (
        not isinstance(started_at, str)
        or not isinstance(ended_at, str)
        or not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or duration < 0
        or abs(float(duration) - _duration_seconds(started_at, ended_at)) > 1.0
    ):
        raise ValueError("measurement-kernel timing evidence drifted")


def validate_state_envelope(
    value: Any,
    *,
    spec: WorkerSpec,
    projection: Mapping[str, Any],
    run_contract_sha256: str,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    record = _mapping(value, "v2.2 state envelope")
    _exact_keys(
        record,
        {
            "schema_version",
            "protocol_id",
            "run_contract_sha256",
            "worker",
            "state",
            "outcome",
            "measurement_kernel",
        },
        "v2.2 state envelope",
    )
    worker = _mapping(record.get("worker"), "state worker")
    expected_worker = {
        "worker_id": spec.worker_id,
        "device": spec.device,
        "index_parity": spec.index_parity,
    }
    inner_wrapper = _mapping(record.get("measurement_kernel"), "measurement kernel")
    _exact_keys(
        inner_wrapper,
        {"protocol_id", "schema_version", "record"},
        "measurement kernel",
    )
    inner = _mapping(inner_wrapper.get("record"), "measurement kernel record")
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("protocol_id") != PROTOCOL_ID
        or record.get("run_contract_sha256") != run_contract_sha256
        or worker != expected_worker
        or record.get("state") != dict(projection)
        or int(projection["index"]) not in spec.state_indices
        or int(projection["index"]) % 2 != spec.index_parity
        or record.get("outcome") not in {STATE_OUTCOME_VALID, STATE_OUTCOME_FAILED}
        or inner_wrapper.get("protocol_id") != MEASUREMENT_KERNEL_PROTOCOL_ID
        or inner_wrapper.get("schema_version") != MEASUREMENT_KERNEL_SCHEMA_VERSION
        or inner.get("protocol_id") != MEASUREMENT_KERNEL_PROTOCOL_ID
        or inner.get("schema_version") != MEASUREMENT_KERNEL_SCHEMA_VERSION
        or inner.get("run_contract_sha256") != run_contract_sha256
        or inner.get("state") != dict(projection)
        or inner.get("outcome")
        not in {
            "VALID_V2_1_FULL_45_SUBSTRATE_STATE",
            "FAILED_V2_1_FULL_45_SUBSTRATE_STATE",
        }
    ):
        raise ValueError("v2.2 state or measurement-kernel identity drifted")
    expected_outcome = (
        STATE_OUTCOME_VALID
        if inner.get("outcome") == "VALID_V2_1_FULL_45_SUBSTRATE_STATE"
        else STATE_OUTCOME_FAILED
    )
    if record.get("outcome") != expected_outcome:
        raise ValueError("outer state outcome differs from measurement kernel")
    _validate_measurement_kernel_record(
        inner,
        spec=spec,
        projection=projection,
        run_contract_sha256=run_contract_sha256,
        runtime_metadata=runtime_metadata,
    )
    return inner


def _gate_contract(run_contract: Mapping[str, Any]) -> Mapping[str, Any]:
    gate = _mapping(run_contract.get("gate"), "run contract gate")
    required = {
        "minimum_screening_states": 20,
        "minimum_parse_coverage": 0.99,
        "minimum_finite_logit_coverage": 1.0,
        "minimum_repeat_canonical_action_agreement": 1.0,
        "minimum_memory_sensitive_states": 8,
    }
    if any(gate.get(key) != expected for key, expected in required.items()):
        raise ValueError("run contract gate thresholds drifted")
    return gate


def aggregate_v22_gate(
    envelopes: Sequence[Mapping[str, Any]],
    *,
    projections: Sequence[Mapping[str, Any]],
    run_contract: Mapping[str, Any],
    run_contract_sha256: str,
    started_at_utc: str,
    ended_at_utc: str,
) -> dict[str, Any]:
    if len(envelopes) != EXPECTED_STATE_COUNT or len(projections) != EXPECTED_STATE_COUNT:
        raise ValueError("v2.2 gate requires exactly 45 ordered states")
    specs = expected_worker_specs()
    inners: list[Mapping[str, Any]] = []
    for index, (envelope, projection) in enumerate(zip(envelopes, projections, strict=True)):
        spec = specs[index % 2]
        inners.append(
            validate_state_envelope(
                envelope,
                spec=spec,
                projection=projection,
                run_contract_sha256=run_contract_sha256,
            )
        )
    gate = _gate_contract(run_contract)
    parse_count = sum(inner.get("parse_success") is True for inner in inners)
    finite_count = sum(inner.get("finite_logit_distances") is True for inner in inners)
    agreement_count = sum(
        inner.get("repeat_canonical_action_agreement") is True for inner in inners
    )
    repeat_values = []
    for inner in inners:
        value = _mapping(inner.get("distances"), "distances").get(
            "repeat_reference_kl"
        )
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            repeat_values.append(float(value))
    mean_repeat_kl = math.fsum(repeat_values) / len(repeat_values) if repeat_values else None
    epsilon = (
        max(REPEAT_NOISE_FLOOR, REPEAT_NOISE_MULTIPLIER * mean_repeat_kl)
        if mean_repeat_kl is not None
        else None
    )
    memory_sensitive_ids: list[str] = []
    if epsilon is not None:
        for inner in inners:
            value = _mapping(inner.get("distances"), "distances").get(
                "summary_reference_kl"
            )
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > epsilon:
                memory_sensitive_ids.append(str(_mapping(inner.get("state"), "state")["state_id"]))
    operation_totals = {
        key: sum(int(_mapping(inner.get("operation_counts"), "operations").get(key, 0)) for inner in inners)
        for key in ("generation_call_count", "teacher_forward_count", "kl_measurement_count")
    }
    forbidden_totals = {
        key: sum(
            int(_mapping(inner.get("operation_counts"), "operations").get(key, 0))
            for inner in inners
        )
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
        "repeat_noise_epsilon": epsilon,
        "memory_sensitive_state_count": len(memory_sensitive_ids),
        "memory_sensitive_state_ids": memory_sensitive_ids,
        **operation_totals,
        **forbidden_totals,
    }
    checks = {
        "minimum_screening_states": EXPECTED_STATE_COUNT >= gate["minimum_screening_states"],
        "minimum_parse_coverage": metrics["parse_coverage"] >= gate["minimum_parse_coverage"],
        "minimum_finite_logit_coverage": metrics["finite_logit_coverage"] >= gate["minimum_finite_logit_coverage"],
        "minimum_repeat_canonical_action_agreement": metrics["repeat_canonical_action_agreement"] >= gate["minimum_repeat_canonical_action_agreement"],
        "minimum_memory_sensitive_states": metrics["memory_sensitive_state_count"] >= gate["minimum_memory_sensitive_states"],
        "maximum_generation_calls": operation_totals["generation_call_count"] <= 90,
        "maximum_teacher_forwards": operation_totals["teacher_forward_count"] <= 135,
        "maximum_kl_measurements": operation_totals["kl_measurement_count"] <= 90,
        "prohibited_work_zero": all(value == 0 for value in forbidden_totals.values()),
    }
    passed = all(checks.values())
    failures = Counter(
        str(_mapping(inner["failure"], "failure").get("category"))
        for inner in inners
        if isinstance(inner.get("failure"), Mapping)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_contract_sha256": run_contract_sha256,
        "status": "COMPLETED_FIXED_45_STATE_V2_2_EAGER_SUBSTRATE",
        "outcome": PASS_OUTCOME if passed else NO_GO_OUTCOME,
        "gate_passed": passed,
        "gate_contract": dict(gate),
        "measurement_kernel_protocol_id": MEASUREMENT_KERNEL_PROTOCOL_ID,
        "worker_topology": [spec.to_dict() for spec in specs],
        "merge_order": list(range(EXPECTED_STATE_COUNT)),
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "metrics": metrics,
        "checks": checks,
        "failure_category_counts": dict(sorted(failures.items())),
        "sample_mutation_performed": False,
        "top_up_performed": False,
        "retry_performed": False,
        "confirm_role_used": False,
        "expert_action_read": False,
        "restoration_work_performed": False,
    }


def _inventory(files: Mapping[str, bytes]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "path": relative,
            "size_bytes": len(files[relative]),
            "sha256": sha256_bytes(files[relative]),
        }
        for relative in sorted(files)
    )


def expected_complete_file_names() -> set[str]:
    names = {
        ARCHIVE_LEDGER_NAME,
        RUN_MANIFEST_FILENAME,
        AGGREGATE_FILENAME,
        RUNTIME_BARRIER_RELEASE_FILENAME,
    }
    for spec in expected_worker_specs():
        prefix = f"{WORKER_DIRECTORY}/{spec.worker_id}"
        names.update(
            {
                f"{WORKER_SIBLING_LEDGER_DIRECTORY}/{spec.worker_id}.json",
                f"{prefix}/{WORKER_LEDGER_FILENAME}",
                f"{prefix}/{WORKER_RUNTIME_FILENAME}",
                f"{prefix}/{WORKER_TERMINAL_FILENAME}",
            }
        )
        for index in spec.state_indices:
            names.add(f"{prefix}/{STATE_DIRECTORY}/{index:03d}.json")
            names.add(f"{prefix}/{ATTEMPT_DIRECTORY}/{index:03d}.json")
    return names


def _validate_global_ledger(
    value: Mapping[str, Any], *, run_contract_sha256: str, expected_global_ledger: Path
) -> tuple[str, int, int]:
    _exact_keys(
        value,
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
            "worker_high_water",
            "worker_sibling_ledgers",
            "retry_count",
            "top_up_count",
        },
        "global ledger",
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("attempt_id") != CANONICAL_ATTEMPT_ID
        or value.get("run_contract_sha256") != run_contract_sha256
        or value.get("retry_count") != 0
        or value.get("top_up_count") != 0
    ):
        raise ValueError("global ledger identity drifted")
    validate_worker_topology(value.get("worker_topology"))
    _validate_timestamp(value.get("claimed_at_utc"), "global claim time")
    if value.get("spawn_started_at_utc") is not None:
        _validate_timestamp(value.get("spawn_started_at_utc"), "spawn time")
    completed = value.get("completed_state_count")
    attempted = value.get("attempted_state_count")
    high_water = _mapping(value.get("worker_high_water"), "worker high water")
    if set(high_water) != {spec.worker_id for spec in expected_worker_specs()}:
        raise ValueError("global ledger worker high-water inventory drifted")
    for spec in expected_worker_specs():
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
            indices = list(_sequence(record.get(key), f"worker high-water {key}"))
            if indices != list(spec.state_indices[: len(indices)]):
                raise ValueError("global worker high-water is not a shard prefix")
        digest = record.get("worker_ledger_sha256")
        if digest is not None and (
            not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError("worker ledger high-water SHA256 is invalid")
    sibling_paths = _mapping(
        value.get("worker_sibling_ledgers"), "worker sibling ledgers"
    )
    if set(sibling_paths) != {spec.worker_id for spec in expected_worker_specs()}:
        raise ValueError("worker sibling ledger path inventory drifted")
    canonical_global = expected_global_ledger
    for spec in expected_worker_specs():
        expected_path = canonical_global.with_name(
            f"{canonical_global.name[:-5]}.{spec.worker_id}.json"
        )
        if sibling_paths.get(spec.worker_id) != str(expected_path):
            raise ValueError("worker sibling ledger canonical path drifted")
    if type(completed) is not int or type(attempted) is not int or not 0 <= completed <= attempted <= EXPECTED_STATE_COUNT:
        raise ValueError("global ledger high-water counts drifted")
    return str(value.get("status")), completed, attempted


def _validate_worker_ledger(
    value: Mapping[str, Any], *, spec: WorkerSpec, run_contract_sha256: str
) -> tuple[list[int], list[int]]:
    _exact_keys(
        value,
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
        "worker ledger",
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("run_contract_sha256") != run_contract_sha256
        or value.get("worker") != spec.to_dict()
        or value.get("status")
        not in {
            "WORKER_SIBLING_PREBOUND_BEFORE_SPAWN",
            "WORKER_ATTEMPT_CLAIMED_BEFORE_RUNTIME_IMPORT",
            "WORKER_SHARD_RUNNING_NO_RETRY",
            "WORKER_SHARD_COMPLETED",
            "WORKER_SHARD_INVALID",
        }
        or value.get("retry_count") != 0
        or value.get("top_up_count") != 0
    ):
        raise ValueError("worker ledger identity drifted")
    _validate_timestamp(value.get("claimed_at_utc"), "worker claim time")
    attempted = list(_sequence(value.get("attempted_state_indices"), "attempted indices"))
    completed = list(_sequence(value.get("completed_state_indices"), "completed indices"))
    if attempted != list(spec.state_indices[: len(attempted)]) or completed != list(spec.state_indices[: len(completed)]) or len(completed) > len(attempted):
        raise ValueError("worker ledger is not a parity-shard prefix")
    return attempted, completed


def _validate_attempt_marker(
    value: Mapping[str, Any], *, spec: WorkerSpec, index: int, run_contract_sha256: str
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "STATE_ATTEMPT_CLAIMED_NO_RETRY_OR_TOP_UP",
        "run_contract_sha256": run_contract_sha256,
        "worker": {
            "worker_id": spec.worker_id,
            "device": spec.device,
            "index_parity": spec.index_parity,
        },
        "state_index": index,
        "retry_count": 0,
        "top_up_count": 0,
    }
    claimed = value.get("claimed_at_utc")
    if not isinstance(claimed, str):
        raise ValueError("state marker lacks claim time")
    actual = dict(value)
    actual.pop("claimed_at_utc", None)
    if actual != expected:
        raise ValueError("state attempt marker drifted")
    _validate_timestamp(claimed, "state claim time")


def validate_v22_evidence_files(
    files: Mapping[str, bytes],
    *,
    expected_source_git_commit: str | None = None,
    require_canonical_attempt_identity: bool = True,
) -> V22Evidence:
    copied = dict(files)
    if any(not isinstance(name, str) or not isinstance(payload, bytes) for name, payload in copied.items()):
        raise TypeError("v2.2 evidence must map paths to bytes")
    for name in copied:
        _safe_relative_path(name)
    required = {ARCHIVE_LEDGER_NAME, RUN_MANIFEST_FILENAME, AGGREGATE_FILENAME}
    if not required.issubset(copied):
        raise ValueError("v2.2 evidence lacks ledger, manifest, or aggregate")
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
    _validate_run_contract_identity(
        run_contract,
        require_canonical_attempt_identity=require_canonical_attempt_identity,
    )
    run_contract_sha256 = sha256_bytes(canonical_json_bytes(run_contract))
    source_commit = _mapping(run_contract.get("git_identity"), "git identity").get("commit")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "V2_2_EAGER_GLOBAL_ATTEMPT_CLAIMED"
        or manifest.get("run_contract_sha256") != run_contract_sha256
        or not isinstance(source_commit, str)
        or GIT_SHA_PATTERN.fullmatch(source_commit) is None
        or expected_source_git_commit is not None
        and source_commit != expected_source_git_commit
    ):
        raise ValueError("run manifest or source identity drifted")
    created_at = _validate_timestamp(manifest.get("created_at_utc"), "manifest time")
    projections = _contract_state_projections(run_contract)
    topology = validate_worker_topology(run_contract.get("worker_topology"))
    ledger = strict_json_object_bytes(copied[ARCHIVE_LEDGER_NAME], label="global ledger")
    ledger_status, ledger_completed, ledger_attempted = _validate_global_ledger(
        ledger,
        run_contract_sha256=run_contract_sha256,
        expected_global_ledger=Path(
            str(_mapping(run_contract["attempt_identity"], "attempt identity")["global_ledger"])
        ),
    )
    aggregate = strict_json_object_bytes(copied[AGGREGATE_FILENAME], label="aggregate")
    outcome = aggregate.get("outcome")
    if outcome not in {PASS_OUTCOME, NO_GO_OUTCOME, INVALID_OUTCOME}:
        raise ValueError("aggregate outcome is not v2.2 PASS, NO_GO, or INVALID")

    envelopes_by_index: dict[int, Mapping[str, Any]] = {}
    attempted_total = 0
    completed_total = 0
    runtime_pair: dict[str, Mapping[str, Any]] = {}
    worker_terminal_outcomes: dict[str, str] = {}
    forensic_by_worker: dict[str, Mapping[str, Any]] = {}
    expected_observed_files = {
        ARCHIVE_LEDGER_NAME,
        RUN_MANIFEST_FILENAME,
        AGGREGATE_FILENAME,
    }
    if (
        RUNTIME_BARRIER_RELEASE_FILENAME in copied
        and RUNTIME_BARRIER_ABORT_FILENAME in copied
    ):
        raise ValueError("runtime barrier cannot be both released and aborted")
    if RUNTIME_BARRIER_RELEASE_FILENAME in copied:
        release = strict_json_object_bytes(
            copied[RUNTIME_BARRIER_RELEASE_FILENAME], label="runtime barrier release"
        )
        if release != {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "COORDINATOR_RELEASED_BOTH_WORKERS",
            "run_contract_sha256": run_contract_sha256,
        }:
            raise ValueError("runtime barrier release token drifted")
        expected_observed_files.add(RUNTIME_BARRIER_RELEASE_FILENAME)
    if RUNTIME_BARRIER_ABORT_FILENAME in copied:
        abort = strict_json_object_bytes(
            copied[RUNTIME_BARRIER_ABORT_FILENAME], label="runtime barrier abort"
        )
        _exact_keys(
            abort,
            {"schema_version", "protocol_id", "status", "run_contract_sha256", "message"},
            "runtime barrier abort",
        )
        if (
            abort.get("schema_version") != SCHEMA_VERSION
            or abort.get("protocol_id") != PROTOCOL_ID
            or abort.get("status") != "COORDINATOR_ABORTED_RUNTIME_BARRIER"
            or abort.get("run_contract_sha256") != run_contract_sha256
            or not isinstance(abort.get("message"), str)
        ):
            raise ValueError("runtime barrier abort token drifted")
        expected_observed_files.add(RUNTIME_BARRIER_ABORT_FILENAME)
    for spec in topology:
        prefix = f"{WORKER_DIRECTORY}/{spec.worker_id}"
        ledger_name = f"{prefix}/{WORKER_LEDGER_FILENAME}"
        sibling_name = f"{WORKER_SIBLING_LEDGER_DIRECTORY}/{spec.worker_id}.json"
        if sibling_name not in copied:
            raise ValueError("evidence lacks prebound worker sibling ledger")
        expected_observed_files.add(sibling_name)
        worker_ledger = strict_json_object_bytes(
            copied[sibling_name], label="worker sibling ledger"
        )
        root_ledger_matches = False
        root_attempted: list[int] | None = None
        root_completed: list[int] | None = None
        root_ledger_sha256: str | None = None
        if ledger_name in copied:
            expected_observed_files.add(ledger_name)
            root_ledger_sha256 = sha256_bytes(copied[ledger_name])
            root_ledger_matches = copied[ledger_name] == copied[sibling_name]
            if not root_ledger_matches:
                if outcome != INVALID_OUTCOME:
                    raise ValueError("worker root and sibling ledgers differ")
                stale_root = strict_json_object_bytes(
                    copied[ledger_name], label="stale worker root ledger"
                )
                stale_attempted, stale_completed = _validate_worker_ledger(
                    stale_root, spec=spec, run_contract_sha256=run_contract_sha256
                )
                root_attempted = stale_attempted
                root_completed = stale_completed
                sibling_attempted = list(worker_ledger["attempted_state_indices"])
                sibling_completed = list(worker_ledger["completed_state_indices"])
                if (
                    stale_attempted
                    != sibling_attempted[: len(stale_attempted)]
                    or stale_completed
                    != sibling_completed[: len(stale_completed)]
                ):
                    raise ValueError("stale root worker ledger is not a sibling prefix")
        elif outcome != INVALID_OUTCOME:
            raise ValueError("completed evidence lacks worker root ledger")
        attempted, completed = _validate_worker_ledger(
            worker_ledger, spec=spec, run_contract_sha256=run_contract_sha256
        )
        if root_ledger_matches:
            root_attempted = list(attempted)
            root_completed = list(completed)
        attempted_total += len(attempted)
        completed_total += len(completed)
        high_water = _mapping(ledger["worker_high_water"], "worker high water")[
            spec.worker_id
        ]
        if (
            high_water.get("attempted_state_indices") != attempted
            or high_water.get("completed_state_indices") != completed
            or high_water.get("worker_ledger_sha256")
            != sha256_bytes(copied[sibling_name])
        ):
            raise ValueError("global ledger does not bind worker ledger high-water bytes")
        runtime_name = f"{prefix}/{WORKER_RUNTIME_FILENAME}"
        if runtime_name in copied:
            expected_observed_files.add(runtime_name)
            runtime_record = strict_json_object_bytes(copied[runtime_name], label="runtime identity")
            _exact_keys(
                runtime_record,
                {
                    "schema_version",
                    "protocol_id",
                    "run_contract_sha256",
                    "worker",
                    "runtime_metadata",
                    "created_at_utc",
                },
                "runtime identity",
            )
            if (
                runtime_record.get("schema_version") != SCHEMA_VERSION
                or runtime_record.get("protocol_id") != PROTOCOL_ID
                or runtime_record.get("run_contract_sha256") != run_contract_sha256
                or runtime_record.get("worker") != spec.to_dict()
            ):
                raise ValueError("runtime identity envelope drifted")
            _validate_timestamp(runtime_record.get("created_at_utc"), "runtime time")
            runtime_pair[spec.worker_id] = _validate_runtime_metadata(
                runtime_record.get("runtime_metadata"), spec=spec
            )
        observed_markers = [
            index
            for index in spec.state_indices
            if f"{prefix}/{ATTEMPT_DIRECTORY}/{index:03d}.json" in copied
        ]
        observed_states = [
            index
            for index in spec.state_indices
            if f"{prefix}/{STATE_DIRECTORY}/{index:03d}.json" in copied
        ]
        forensic_by_worker[spec.worker_id] = {
            "sibling_attempted_state_indices": list(attempted),
            "sibling_completed_state_indices": list(completed),
            "observed_attempt_marker_indices": observed_markers,
            "observed_state_record_indices": observed_states,
            "missing_attempt_marker_indices": sorted(set(attempted) - set(observed_markers)),
            "orphan_attempt_marker_indices": sorted(set(observed_markers) - set(attempted)),
            "missing_state_record_indices": sorted(set(completed) - set(observed_states)),
            "orphan_state_record_indices": sorted(set(observed_states) - set(completed)),
            "root_ledger_status": (
                "exact_sibling_copy" if root_ledger_matches else "missing_or_stale"
            ),
            "root_ledger_attempted_state_indices": root_attempted,
            "root_ledger_completed_state_indices": root_completed,
            "root_ledger_sha256": root_ledger_sha256,
        }
        if outcome != INVALID_OUTCOME and (
            observed_markers != attempted or observed_states != completed
        ):
            raise ValueError("completed worker evidence differs from sibling high-water")
        for index in observed_markers:
            marker_name = f"{prefix}/{ATTEMPT_DIRECTORY}/{index:03d}.json"
            expected_observed_files.add(marker_name)
            _validate_attempt_marker(
                strict_json_object_bytes(copied[marker_name], label="state marker"),
                spec=spec,
                index=index,
                run_contract_sha256=run_contract_sha256,
            )
        for index in observed_states:
            state_name = f"{prefix}/{STATE_DIRECTORY}/{index:03d}.json"
            expected_observed_files.add(state_name)
            envelope = strict_json_object_bytes(copied[state_name], label="state envelope")
            validate_state_envelope(
                envelope,
                spec=spec,
                projection=projections[index],
                run_contract_sha256=run_contract_sha256,
                runtime_metadata=runtime_pair.get(spec.worker_id),
            )
            if index in envelopes_by_index:
                raise ValueError("state appears in both worker shards")
            envelopes_by_index[index] = envelope
        terminal_name = f"{prefix}/{WORKER_TERMINAL_FILENAME}"
        if terminal_name in copied:
            expected_observed_files.add(terminal_name)
            terminal = strict_json_object_bytes(copied[terminal_name], label="worker terminal")
            _exact_keys(
                terminal,
                {
                    "schema_version",
                    "protocol_id",
                    "run_contract_sha256",
                    "worker",
                    "outcome",
                    "completed_state_indices",
                    "attempted_state_indices",
                    "failure",
                    "ended_at_utc",
                },
                "worker terminal",
            )
            if (
                terminal.get("schema_version") != SCHEMA_VERSION
                or terminal.get("protocol_id") != PROTOCOL_ID
                or terminal.get("run_contract_sha256") != run_contract_sha256
                or terminal.get("worker") != spec.to_dict()
                or terminal.get("completed_state_indices") != completed
                or terminal.get("attempted_state_indices") != attempted
                or terminal.get("outcome") not in {WORKER_OUTCOME_COMPLETE, WORKER_OUTCOME_INVALID}
            ):
                raise ValueError("worker terminal drifted")
            _validate_timestamp(terminal.get("ended_at_utc"), "worker terminal time")
            worker_terminal_outcomes[spec.worker_id] = str(terminal["outcome"])

    if (ledger_completed, ledger_attempted) != (completed_total, attempted_total):
        raise ValueError("global and worker ledger high-water counts differ")
    generation_count = sum(
        int(_mapping(_mapping(envelope["measurement_kernel"], "kernel")["record"], "inner")["operation_counts"]["generation_call_count"])
        for envelope in envelopes_by_index.values()
    )
    teacher_count = sum(
        int(_mapping(_mapping(envelope["measurement_kernel"], "kernel")["record"], "inner")["operation_counts"]["teacher_forward_count"])
        for envelope in envelopes_by_index.values()
    )
    kl_count = sum(
        int(_mapping(_mapping(envelope["measurement_kernel"], "kernel")["record"], "inner")["operation_counts"]["kl_measurement_count"])
        for envelope in envelopes_by_index.values()
    )
    if outcome in {PASS_OUTCOME, NO_GO_OUTCOME}:
        if (
            ledger_status != "GLOBAL_ATTEMPT_COMPLETED"
            or
            set(copied) != expected_complete_file_names()
            or completed_total != EXPECTED_STATE_COUNT
            or attempted_total != EXPECTED_STATE_COUNT
            or sorted(envelopes_by_index) != list(range(EXPECTED_STATE_COUNT))
            or worker_terminal_outcomes
            != {spec.worker_id: WORKER_OUTCOME_COMPLETE for spec in topology}
            or set(runtime_pair) != {spec.worker_id for spec in topology}
        ):
            raise ValueError("completed v2.2 evidence is partial or has wrong inventory")
        _validate_runtime_pair(runtime_pair)
        ended = aggregate.get("ended_at_utc")
        _validate_timestamp(ended, "aggregate end time")
        recomputed = aggregate_v22_gate(
            [envelopes_by_index[index] for index in range(EXPECTED_STATE_COUNT)],
            projections=projections,
            run_contract=run_contract,
            run_contract_sha256=run_contract_sha256,
            started_at_utc=created_at,
            ended_at_utc=str(ended),
        )
        if aggregate != recomputed:
            raise ValueError("stored v2.2 aggregate differs from CPU raw reduction")
    else:
        if ledger_status != "GLOBAL_ATTEMPT_INVALID":
            raise ValueError("INVALID aggregate lacks INVALID global ledger status")
        if set(copied) != expected_observed_files:
            raise ValueError("INVALID evidence contains unknown or unbound files")
        _exact_keys(
            aggregate,
            {
                "schema_version",
                "protocol_id",
                "run_contract_sha256",
                "status",
                "outcome",
                "invalid_failure",
                "completed_state_count",
                "attempted_state_count",
                "started_at_utc",
                "ended_at_utc",
                "retry_performed",
                "top_up_performed",
                "forensic_inventory",
            },
            "INVALID aggregate",
        )
        if (
            aggregate.get("schema_version") != SCHEMA_VERSION
            or aggregate.get("protocol_id") != PROTOCOL_ID
            or aggregate.get("run_contract_sha256") != run_contract_sha256
            or aggregate.get("status") != "INVALID_TWO_WORKER_V2_2_EAGER_SUBSTRATE"
            or aggregate.get("completed_state_count") != completed_total
            or aggregate.get("attempted_state_count") != attempted_total
            or aggregate.get("retry_performed") is not False
            or aggregate.get("top_up_performed") is not False
            or aggregate.get("forensic_inventory") != forensic_by_worker
            or not isinstance(aggregate.get("invalid_failure"), Mapping)
        ):
            raise ValueError("INVALID aggregate evidence drifted")
        _validate_timestamp(aggregate.get("started_at_utc"), "invalid start time")
        _validate_timestamp(aggregate.get("ended_at_utc"), "invalid end time")
    inventory = _inventory(copied)
    return V22Evidence(
        files=copied,
        run_contract=dict(run_contract),
        source_git_commit=source_commit,
        run_contract_sha256=run_contract_sha256,
        outcome=str(outcome),
        aggregate_sha256=sha256_bytes(copied[AGGREGATE_FILENAME]),
        completed_state_count=completed_total,
        attempted_state_count=attempted_total,
        generation_call_count=generation_count,
        teacher_forward_count=teacher_count,
        kl_measurement_count=kl_count,
        inventory=inventory,
        tree_inventory_sha256=sha256_bytes(canonical_json_bytes(inventory)),
    )


def collect_raw_v22_evidence(
    raw_output_dir: str | Path,
    global_attempt_ledger: str | Path,
    *,
    expected_source_git_commit: str | None = None,
    require_canonical_location: bool = True,
) -> V22Evidence:
    root = Path(raw_output_dir)
    ledger = Path(global_attempt_ledger)
    if require_canonical_location and (
        root.resolve() != CANONICAL_OUTPUT_DIR or ledger.resolve() != CANONICAL_LEDGER_PATH
    ):
        raise ValueError("v2.2 raw root or global ledger is not canonical")
    if root.is_symlink() or ledger.is_symlink() or not root.is_dir() or not ledger.is_file():
        raise ValueError("v2.2 raw evidence root or ledger is missing")
    files = {ARCHIVE_LEDGER_NAME: ledger.read_bytes()}
    global_record = strict_json_object_bytes(
        files[ARCHIVE_LEDGER_NAME], label="global ledger"
    )
    sibling_paths = _mapping(
        global_record.get("worker_sibling_ledgers"), "worker sibling ledgers"
    )
    for spec in expected_worker_specs():
        sibling = Path(str(sibling_paths.get(spec.worker_id)))
        if sibling.is_symlink() or not sibling.is_file():
            raise ValueError("v2.2 worker sibling ledger is missing")
        files[
            f"{WORKER_SIBLING_LEDGER_DIRECTORY}/{spec.worker_id}.json"
        ] = sibling.read_bytes()
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ValueError("v2.2 raw evidence contains a symlink")
        if item.is_file():
            relative = item.relative_to(root).as_posix()
            _safe_relative_path(relative)
            files[relative] = item.read_bytes()
    return validate_v22_evidence_files(
        files,
        expected_source_git_commit=expected_source_git_commit,
        require_canonical_attempt_identity=require_canonical_location,
    )


def deterministic_tar_bytes(files: Mapping[str, bytes]) -> bytes:
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


def read_v22_evidence_archive(
    archive_path: str | Path,
    *,
    expected_source_git_commit: str | None = None,
    require_canonical_attempt_identity: bool = True,
) -> V22Evidence:
    path = Path(archive_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("v2.2 archive is missing or symlinked")
    files: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != sorted(member.name for member in members):
            raise ValueError("v2.2 archive member order drifted")
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
                raise ValueError("v2.2 archive has non-canonical metadata")
            relative = _safe_relative_path(member.name[len(prefix) :])
            if relative in files:
                raise ValueError("v2.2 archive has duplicate members")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("v2.2 archive member is unreadable")
            files[relative] = source.read()
    if path.read_bytes() != deterministic_tar_bytes(files):
        raise ValueError("v2.2 archive is not canonical deterministic USTAR")
    return validate_v22_evidence_files(
        files,
        expected_source_git_commit=expected_source_git_commit,
        require_canonical_attempt_identity=require_canonical_attempt_identity,
    )


def package_raw_v22_evidence(
    *,
    raw_output_dir: str | Path,
    global_attempt_ledger: str | Path,
    output_archive: str | Path,
    source_git_commit: str,
    require_canonical_location: bool = True,
) -> dict[str, Any]:
    output = Path(output_archive)
    if require_canonical_location and output.resolve() != CANONICAL_ARCHIVE_PATH:
        raise ValueError("v2.2 raw archive path is not canonical")
    if output.exists():
        raise FileExistsError("v2.2 archive already exists; overwrite is forbidden")
    evidence = collect_raw_v22_evidence(
        raw_output_dir,
        global_attempt_ledger,
        expected_source_git_commit=source_git_commit,
        require_canonical_location=require_canonical_location,
    )
    payload = deterministic_tar_bytes(evidence.files)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    reread = read_v22_evidence_archive(
        output, expected_source_git_commit=source_git_commit
    )
    if reread.inventory != evidence.inventory or reread.outcome != evidence.outcome:
        raise ValueError("written v2.2 archive differs from raw evidence")
    return {
        "status": "PACKAGED_RESTORATION_V2_2_EAGER_RAW_EVIDENCE",
        "source_git_commit": source_git_commit,
        "outcome": evidence.outcome,
        "archive_path": str(output.resolve()),
        "archive_sha256": sha256_file(output),
        "archive_size_bytes": output.stat().st_size,
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
        "file_count": len(evidence.inventory),
    }


def build_v22_artifact_manifest(
    *,
    evidence: V22Evidence,
    source_archive_path: str | Path,
    fresh_immutable_archive: str | Path,
    hf_repo: str,
    hf_immutable_revision: str,
    hf_path: str,
) -> dict[str, Any]:
    if GIT_SHA_PATTERN.fullmatch(hf_immutable_revision) is None:
        raise ValueError("HF immutable revision must be a full commit SHA")
    if hf_repo != CANONICAL_HF_REPO or hf_path != CANONICAL_HF_PATH:
        raise ValueError("HF repo or path differs from the frozen destination")
    source_archive = Path(source_archive_path).resolve()
    fresh_archive = Path(fresh_immutable_archive).resolve()
    if source_archive == fresh_archive:
        raise ValueError("fresh HF archive must be independent from the source archive")
    source = read_v22_evidence_archive(
        source_archive, expected_source_git_commit=evidence.source_git_commit
    )
    fresh = read_v22_evidence_archive(
        fresh_archive, expected_source_git_commit=evidence.source_git_commit
    )
    if (
        source.inventory != evidence.inventory
        or fresh.inventory != evidence.inventory
        or source_archive.read_bytes() != fresh_archive.read_bytes()
        or sha256_file(source_archive) != sha256_file(fresh_archive)
    ):
        raise ValueError("fresh HF archive differs from source evidence")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "VERIFIED_RESTORATION_V2_2_EAGER_ARTIFACT",
        "source_execution": {
            "source_git_commit": evidence.source_git_commit,
            "run_contract_sha256": evidence.run_contract_sha256,
        },
        "result": {
            "outcome": evidence.outcome,
            "completed_state_count": evidence.completed_state_count,
            "attempted_state_count": evidence.attempted_state_count,
        },
        "raw_archive": {
            "sha256": sha256_file(source_archive),
            "size_bytes": source_archive.stat().st_size,
            "tree_inventory_sha256": evidence.tree_inventory_sha256,
            "file_count": len(evidence.inventory),
            "format": ARCHIVE_FORMAT,
        },
        "hf_artifact": {
            "repo": hf_repo,
            "tag": CANONICAL_HF_TAG,
            "path": hf_path,
            "immutable_revision": hf_immutable_revision,
            "fresh_immutable_download_verified": True,
            "fresh_archive_sha256": sha256_file(fresh_archive),
            "fresh_archive_size_bytes": fresh_archive.stat().st_size,
        },
    }
