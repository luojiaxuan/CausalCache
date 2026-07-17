"""Validate, reduce, and package v2.2 expansion exact-label evidence.

The raw complete distance table is the only scientific source of truth.  Every
oracle, edge, interaction, and attribution exposed by this module is rebuilt
from that table during validation; producers are not trusted to serialize
derived labels.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import os
import re
import subprocess
import tarfile
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.restoration_v2_2_eager_artifact import (
    validate_worker_runtime_metadata as validate_eager_worker_runtime_metadata,
    validate_worker_runtime_pair as validate_eager_worker_runtime_pair,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_restoration_v2_2_expansion_exact_labels_v1"
ARTIFACT_PROTOCOL_ID = (
    "causalcache_restoration_v2_2_expansion_exact_labels_artifact_v1"
)
ATTEMPT_ID = "restoration-v2-2-expansion-exact-labels-v1"
ARCHIVE_MEMBER_PREFIX = ATTEMPT_ID
ARCHIVE_FORMAT = "ustar"

CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_expansion_labels_v1.json"
)
CANONICAL_RUNNER_FREEZE_PATH = (
    "code/configs/causalcache_restoration_v2_2_expansion_labels_runner_v1.json"
)
CANONICAL_OUTPUT_DIR = Path(
    "/data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1"
)
CANONICAL_LEDGER_PATH = Path(
    "/data/experiments/causalcache/"
    ".restoration-v2-2-expansion-exact-labels-v1.attempt.json"
)
CANONICAL_ARCHIVE_PATH = Path(
    "/data/experiments/causalcache/"
    "restoration-v2-2-expansion-exact-labels-v1.tar"
)
CANONICAL_HF_REPO = (
    "gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-mobile"
)
CANONICAL_HF_TAG = "v2.2-expansion-exact-labels-v1"
CANONICAL_HF_PATH = "raw/v2.2-expansion-exact-labels-v1.tar"
CANONICAL_GIT_ORIGIN_URL = "https://github.com/luojiaxuan/CausalCache.git"
SUBSTRATE_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_expansion_substrate_v1.json"
)

GLOBAL_LEDGER_ARCHIVE_NAME = "global_attempt_ledger.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"
AGGREGATE_FILENAME = "aggregate.json"
EXECUTION_EVIDENCE_FILENAME = "execution_evidence.json"
MONITOR_SUMMARY_FILENAME = "monitor_summary.json"
WORKER_DIRECTORY = "workers"
WORKER_SIBLING_LEDGER_DIRECTORY = "worker_sibling_ledgers"
WORKER_LEDGER_FILENAME = "worker_attempt_ledger.json"
WORKER_RUNTIME_FILENAME = "runtime_identity.json"
WORKER_TERMINAL_FILENAME = "terminal.json"
STATE_DIRECTORY = "states"
ATTEMPT_DIRECTORY = "attempts"
LOG_PATHS = {
    "preflight": "logs/preflight.log",
    "execution": "logs/execution.log",
    "utilization_monitor": "logs/gpu_utilization_monitor.log",
    "utilization_monitor_ready": "logs/gpu_utilization_monitor.ready.json",
    "utilization_monitor_stop_request": (
        "logs/gpu_utilization_monitor.stop-request.json"
    ),
}

EXPECTED_TRAJECTORY_COUNT = 64
EXPECTED_STATE_COUNT = 192
EXPECTED_DISTANCE_ROWS = 1_792
EXPECTED_DEPLOYMENT_EDGES = 1_856
EXPECTED_FULL_EDGES = 3_072
EXPECTED_PAIR_INTERACTIONS = 1_984
EXPECTED_ATTRIBUTION_ROWS = 576
EXPECTED_PRIMARY_ORACLES = 192
EXPECTED_TEACHER_FORWARDS = 1_984
EXPECTED_KL_MEASUREMENTS = 1_792
PRIMARY_BUDGET_EVENT_CAPACITY = 2
MAXIMUM_REPEAT_KL = 1e-4

EXPECTED_TOTALS = {
    "trajectory_count": EXPECTED_TRAJECTORY_COUNT,
    "state_count": EXPECTED_STATE_COUNT,
    "raw_distance_row_count": EXPECTED_DISTANCE_ROWS,
    "deployment_conditional_edge_count": EXPECTED_DEPLOYMENT_EDGES,
    "full_hypercube_edge_count": EXPECTED_FULL_EDGES,
    "pair_interaction_count": EXPECTED_PAIR_INTERACTIONS,
    "exact_permutation_attribution_count": EXPECTED_ATTRIBUTION_ROWS,
    "primary_exact_subset_oracle_count": EXPECTED_PRIMARY_ORACLES,
    "teacher_forward_count": EXPECTED_TEACHER_FORWARDS,
    "kl_measurement_count": EXPECTED_KL_MEASUREMENTS,
    "scalar_host_transfer_count": EXPECTED_KL_MEASUREMENTS,
    "full_logit_tensor_host_transfer_count": 0,
    "retry_count": 0,
    "top_up_count": 0,
}

PROHIBITED_OPERATION_COUNTS = {
    "generation_call_count": 0,
    "confirm_state_access_count": 0,
    "confirm_processor_prompt_count": 0,
    "confirm_decoder_input_count": 0,
    "confirm_generation_count": 0,
    "confirm_teacher_forward_count": 0,
    "expert_action_read_count": 0,
    "gate_training_example_count": 0,
    "gate_model_forward_count": 0,
    "gate_selection_count": 0,
    "matched_nll_evaluation_count": 0,
    "closed_loop_episode_count": 0,
    "full_logit_tensor_host_transfer_count": 0,
    "retry_count": 0,
    "top_up_count": 0,
}

PASS_OUTCOME = "PASS_V2_2_EXPANSION_EXACT_LABELS_V1"
STATE_STATUS = "COMPLETED_V2_2_EXPANSION_EXACT_LABEL_STATE"
WORKER_STATUS = "COMPLETED_V2_2_EXPANSION_EXACT_LABEL_WORKER"
RUNNER_FREEZE_SCHEMA_VERSION = "1.0.0"
RUNNER_FREEZE_PROTOCOL_ID = (
    "causalcache_restoration_v2_2_expansion_exact_labels_runner_freeze_v1"
)
SOURCE_PATHS_REQUIRED_IN_RUNNER_FREEZE = {
    "code/causalcache/restoration_v2_2_expansion_labels_artifact.py",
    "code/causalcache/restoration_v2_2_expansion_labels_contract.py",
    "code/causalcache/data/restoration_v2_2_expansion_label_parent.py",
    "code/causalcache/data/restoration_v2_2_expansion_label_inputs.py",
    "code/scripts/run_restoration_v2_2_expansion_labels.py",
    "code/scripts/manage_restoration_v2_2_expansion_labels_artifact.py",
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{8,64}")

CANONICAL_PARENT_SUBSTRATE = {
    "repo": "gavinlaw/causalcache-restoration-v2-2-label-expansion-substrate-mobile",
    "tag": "v2.2-label-expansion-substrate-v1",
    "path": "raw/v2.2-label-expansion-substrate-v1.tar",
    "immutable_revision": "25ac19cf6ef98adc243d421cd0039ac104ddb539",
    "archive_sha256": "4e77a38be34cb2f3c084a13abd47c0530e6729ff6cca72793977c62fa78ff47d",
    "archive_size_bytes": 7_475_200,
    "tree_inventory_sha256": "56f291053121ccf813698a48beb6269b1fa1d096b7e974f6eb9424f55bc45543",
    "outcome": "PASS_V2_2_LABEL_EXPANSION_SUBSTRATE_V1",
    "validated_state_count": EXPECTED_STATE_COUNT,
}

CANONICAL_DERIVED_ARTIFACT = {
    "repo": "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
    "tag": "restoration-v2-label-expansion-v1.0.0",
    "immutable_revision": "630363a6adb692d72774f16dd0653a50216313ff",
    "payload_prefix": "derived/restoration-v2-label-expansion-v1",
    "artifact_tree_sha256": (
        "9394b369e2b741e6aacf9ece4fc5dae3e6337b25a7e65402307e4e7862b94abc"
    ),
}
FROZEN_SUBSTRATE_STATE_PROJECTION_SHA256 = (
    "fc1c6bed6f069e4df28a12a835fa2a9264b532f206cc30c211e6e8f0e5f21be7"
)
ALGEBRA_RESIDUAL_TOLERANCE = 1e-12
MONITOR_SAMPLE_CADENCE_SECONDS = 1.0
MONITOR_SAMPLE_TOLERANCE_SECONDS = 2.0
MONITOR_MAX_SAMPLE_GAP_SECONDS = (
    MONITOR_SAMPLE_CADENCE_SECONDS + MONITOR_SAMPLE_TOLERANCE_SECONDS
)
CANONICAL_RUNTIME = {
    "container_image_digest": (
        "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
    ),
    "python_version": "3.12.3",
    "torch_version": "2.11.0+cu130",
    "transformers_version": "5.6.0",
    "torch_cuda_version": "13.0",
    "cudnn_version": 91900,
    "gpu_name": "NVIDIA H200",
    "nvidia_driver_version": "570.172.08",
    "dtype": "torch.bfloat16",
    "protocol_id": "causalcache_restoration_v2_1_official_tool_interface",
    "runtime_profile_id": "causalcache_restoration_v2_2_eager_runtime",
    "requested_attention_implementation": "eager",
    "observed_attention_implementation": {
        "top": "eager",
        "text": "eager",
        "vision": "eager",
    },
    "seed": 0,
    "strict_cuda_determinism_claimed": False,
    "deterministic_algorithms_requested": False,
    "deterministic_algorithms_enabled": False,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "float32_matmul_precision": "highest",
}


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
class ExpansionLabelEvidence:
    files: Mapping[str, bytes]
    run_contract: Mapping[str, Any]
    source_git_commit: str
    execution_git_commit: str
    run_contract_sha256: str
    config_sha256: str
    outcome: str
    aggregate: Mapping[str, Any]
    reduction: Mapping[str, Any]
    runtime_gpu_uuids: tuple[str, str]
    inventory: tuple[Mapping[str, Any], ...]
    tree_inventory_sha256: str


def expected_worker_specs() -> tuple[WorkerSpec, WorkerSpec]:
    return (
        WorkerSpec("even", "cuda:0", 0, tuple(range(0, EXPECTED_STATE_COUNT, 2))),
        WorkerSpec("odd", "cuda:1", 1, tuple(range(1, EXPECTED_STATE_COUNT, 2))),
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_exclusive_publish_bytes(
    path: str | Path,
    payload: bytes,
    *,
    publish_fn=os.link,
) -> None:
    """Publish complete bytes atomically without ever exposing a partial final."""
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {final}")
    temporary = final.with_name(f".{final.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        # note (luojiaxuan): Hard-link publication is the portable no-replace
        # equivalent of atomic renameat2(RENAME_NOREPLACE) for same-dir files.
        publish_fn(temporary, final)
        _fsync_directory(final.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def strict_json_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if pretty_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical pretty JSON")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} field inventory drifted; "
            f"missing={sorted(keys - set(value))}, extra={sorted(set(value) - keys)}"
        )


def _safe_relative_path(value: str, label: str = "path") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} is not canonical relative POSIX")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _git_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase Git SHA")
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return converted


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be a UTC timestamp")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be a UTC timestamp") from error


def _coalitions(event_ids: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        subset
        for size in range(len(event_ids) + 1)
        for subset in itertools.combinations(event_ids, size)
    )


def _state_operation_counts(event_count: int) -> dict[str, int]:
    power = 1 << event_count
    return {
        "reference_teacher_forward_count": 1,
        "reference_repeat_teacher_forward_count": 1,
        "non_reference_coalition_teacher_forward_count": power - 1,
        "teacher_forward_count": power + 1,
        "kl_measurement_count": power,
        "scalar_host_transfer_count": power,
        "raw_distance_row_count": power,
        "full_logit_tensor_host_transfer_count": 0,
        "retry_count": 0,
        "top_up_count": 0,
    }


def _state_projection(index: int, value: Any) -> dict[str, Any]:
    state = dict(_mapping(value, f"state {index} projection"))
    _exact_keys(
        state,
        {
            "state_index",
            "role",
            "source_id",
            "decision_step_id",
            "state_id",
            "history_event_step_ids",
            "candidate_event_step_ids",
            "current_equivalent_event_step_id",
            "parent_member_name",
            "parent_member_sha256",
            "canonical_action_sha256",
            "teacher_target_sha256",
            "request_manifest_sha256",
            "slice_witness_sha256",
            "immutable_artifact_tree_sha256",
            "coalition_inputs",
        },
        f"state {index} projection",
    )
    expected_step = 4 + index % 3
    expected_role = "gate_train_expansion" if index < 144 else "gate_development_expansion"
    event_ids = list(range(1, expected_step - 1))
    history_ids = list(range(1, expected_step))
    source_id = state.get("source_id")
    worker_id = "even" if index % 2 == 0 else "odd"
    expected_member = f"workers/{worker_id}/states/{index:03d}.json"
    if (
        state.get("state_index") != index
        or state.get("role") != expected_role
        or not isinstance(source_id, str)
        or not source_id
        or state.get("decision_step_id") != expected_step
        or state.get("state_id")
        != f"{source_id}:decision_step:{expected_step:03d}"
        or state.get("history_event_step_ids") != history_ids
        or state.get("candidate_event_step_ids") != event_ids
        or state.get("current_equivalent_event_step_id") != expected_step - 1
        or state.get("parent_member_name") != expected_member
        or state.get("immutable_artifact_tree_sha256")
        != CANONICAL_DERIVED_ARTIFACT["artifact_tree_sha256"]
    ):
        raise ValueError(f"state {index} dynamic projection or 2/3/4 geometry drifted")
    for key in (
        "parent_member_sha256",
        "canonical_action_sha256",
        "teacher_target_sha256",
        "request_manifest_sha256",
        "slice_witness_sha256",
    ):
        _sha256(state.get(key), f"state {index} {key}")
    coalition_inputs = _sequence(
        state.get("coalition_inputs"), f"state {index} coalition inputs"
    )
    expected_coalitions = _coalitions(tuple(event_ids))
    if len(coalition_inputs) != len(expected_coalitions):
        raise ValueError(f"state {index} coalition-input denominator drifted")
    normalized_inputs: list[dict[str, Any]] = []
    for position, (raw, coalition) in enumerate(
        zip(coalition_inputs, expected_coalitions, strict=True)
    ):
        witness = dict(_mapping(raw, f"state {index} coalition input {position}"))
        _exact_keys(
            witness,
            {"coalition", "input_sha256"},
            f"state {index} coalition input {position}",
        )
        if witness.get("coalition") != list(coalition):
            raise ValueError(
                f"state {index} coalition inputs are not cardinality-then-lexicographic"
            )
        _sha256(witness.get("input_sha256"), "coalition input SHA256")
        normalized_inputs.append(witness)
    state["coalition_inputs"] = normalized_inputs
    return state


def validate_state_projections(values: Any) -> tuple[dict[str, Any], ...]:
    rows = _sequence(values, "state projections")
    if len(rows) != EXPECTED_STATE_COUNT:
        raise ValueError("state projection denominator must be exactly 192")
    normalized = tuple(_state_projection(index, value) for index, value in enumerate(rows))
    if len({row["state_id"] for row in normalized}) != EXPECTED_STATE_COUNT:
        raise ValueError("state ids must be unique")
    for trajectory_offset in range(EXPECTED_TRAJECTORY_COUNT):
        group = normalized[trajectory_offset * 3 : trajectory_offset * 3 + 3]
        if len({row["source_id"] for row in group}) != 1:
            raise ValueError("each trajectory must own exactly adjacent decision steps 4/5/6")
    if len({row["source_id"] for row in normalized}) != EXPECTED_TRAJECTORY_COUNT:
        raise ValueError("trajectory denominator must be exactly 64")
    substrate_projection = [
        {
            "state_index": row["state_index"],
            "role": row["role"],
            "source_id": row["source_id"],
            "state_id": row["state_id"],
            "decision_step_id": row["decision_step_id"],
            "history_event_step_ids": row["history_event_step_ids"],
            "candidate_event_step_ids": row["candidate_event_step_ids"],
            "current_equivalent_event_step_id": row[
                "current_equivalent_event_step_id"
            ],
        }
        for row in normalized
    ]
    if (
        sha256_bytes(canonical_json_bytes(substrate_projection))
        != FROZEN_SUBSTRATE_STATE_PROJECTION_SHA256
    ):
        raise ValueError("192-state substrate projection SHA256 drifted")
    return normalized


def replay_external_input_projections(
    *,
    parent_substrate_archive: str | Path,
    derived_artifact_root: str | Path,
    ocr_backend_config: str | Path,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Rebuild every dynamic input witness without loading the policy runtime."""
    from causalcache.data.restoration_v2_2_expansion_label_inputs import (
        build_v2_2_expansion_label_messages,
        expansion_label_prompt_inventory,
        make_v2_2_expansion_label_prompt_spec,
    )
    from causalcache.data.restoration_v2_2_expansion_label_parent import (
        extract_expansion_label_parent_states,
    )
    from causalcache.policy.gui_owl_v2_1 import (
        serialize_gui_owl_v2_1_teacher_target,
    )
    from causalcache.restoration_v2_2_expansion_substrate_artifact import (
        read_expansion_substrate_archive,
    )
    from scripts.run_restoration_v2_2_expansion_substrate import (
        CANONICAL_OCR_BACKEND_CONFIG_PATH,
        load_immutable_expansion_artifact,
    )

    def decode_rgb_image(payload: bytes) -> Any:
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB").copy()

    supplied_parent = Path(parent_substrate_archive)
    if supplied_parent.is_symlink():
        raise ValueError("parent substrate archive cannot be a symlink")
    parent_path = supplied_parent.resolve()
    if (
        not parent_path.is_file()
        or sha256_file(parent_path) != CANONICAL_PARENT_SUBSTRATE["archive_sha256"]
        or parent_path.stat().st_size
        != CANONICAL_PARENT_SUBSTRATE["archive_size_bytes"]
    ):
        raise ValueError("parent substrate archive bytes drifted")
    parent_evidence = read_expansion_substrate_archive(parent_path)
    if (
        parent_evidence.outcome != CANONICAL_PARENT_SUBSTRATE["outcome"]
        or parent_evidence.completed_state_count
        != CANONICAL_PARENT_SUBSTRATE["validated_state_count"]
        or parent_evidence.tree_inventory_sha256
        != CANONICAL_PARENT_SUBSTRATE["tree_inventory_sha256"]
    ):
        raise ValueError("parent substrate scientific identity drifted")

    supplied_ocr = Path(ocr_backend_config)
    if supplied_ocr.is_symlink():
        raise ValueError("OCR backend config cannot be a symlink")
    ocr_path = supplied_ocr.resolve()
    suffix = Path(CANONICAL_OCR_BACKEND_CONFIG_PATH).parts
    if (
        not ocr_path.is_file()
        or tuple(ocr_path.parts[-len(suffix) :]) != suffix
    ):
        raise ValueError("OCR backend config is not at its canonical repository path")
    repository_root = ocr_path.parents[len(suffix) - 1]
    substrate_config_path = repository_root / SUBSTRATE_CONFIG_PATH
    if substrate_config_path.is_symlink() or not substrate_config_path.is_file():
        raise ValueError("frozen expansion substrate config is missing")
    substrate_config = strict_json_object_bytes(
        substrate_config_path.read_bytes(), label="frozen expansion substrate config"
    )
    artifact = load_immutable_expansion_artifact(
        artifact_root=derived_artifact_root,
        frozen_config=substrate_config,
        backend_config_path=ocr_path,
        derived_repo=CANONICAL_DERIVED_ARTIFACT["repo"],
        derived_revision=CANONICAL_DERIVED_ARTIFACT["immutable_revision"],
    )
    if (
        artifact.tree.get("artifact_tree_sha256")
        != CANONICAL_DERIVED_ARTIFACT["artifact_tree_sha256"]
    ):
        raise ValueError("derived artifact tree differs from the frozen input")
    parents = extract_expansion_label_parent_states(parent_evidence, artifact)
    if len(parents) != EXPECTED_STATE_COUNT:
        raise ValueError("external replay parent denominator drifted")

    projections: list[dict[str, Any]] = []
    coalition_witness_count = 0
    for parent in parents:
        coalition_inputs: list[dict[str, Any]] = []
        for coalition in _coalitions(parent.candidate_event_step_ids):
            spec = make_v2_2_expansion_label_prompt_spec(
                artifact,
                parent,
                restored_event_step_ids=coalition,
                budget_event_capacity=PRIMARY_BUDGET_EVENT_CAPACITY,
            )
            bundle = build_v2_2_expansion_label_messages(
                artifact,
                spec,
                image_decoder=decode_rgb_image,
            )
            prompt_inventory = expansion_label_prompt_inventory(spec, bundle)
            input_sha = sha256_bytes(
                canonical_json_bytes(
                    {
                        "prompt_inventory": prompt_inventory,
                        "immutable_artifact_tree_sha256": (
                            parent.immutable_artifact_tree_sha256
                        ),
                    }
                )
            )
            coalition_inputs.append(
                {"coalition": list(coalition), "input_sha256": input_sha}
            )
        coalition_witness_count += len(coalition_inputs)
        teacher_target = serialize_gui_owl_v2_1_teacher_target(
            parent.canonical_action
        )
        projections.append(
            {
                "state_index": parent.index,
                "role": parent.role,
                "source_id": parent.source_id,
                "decision_step_id": parent.decision_step_id,
                "state_id": parent.state_id,
                "history_event_step_ids": list(parent.history_event_step_ids),
                "candidate_event_step_ids": list(
                    parent.candidate_event_step_ids
                ),
                "current_equivalent_event_step_id": (
                    parent.current_equivalent_event_step_id
                ),
                "parent_member_name": parent.member_name,
                "parent_member_sha256": parent.member_sha256,
                "canonical_action_sha256": parent.canonical_action_sha256,
                "teacher_target_sha256": sha256_bytes(
                    teacher_target.encode("utf-8")
                ),
                "request_manifest_sha256": parent.request_manifest_sha256,
                "slice_witness_sha256": parent.slice_witness_sha256,
                "immutable_artifact_tree_sha256": (
                    parent.immutable_artifact_tree_sha256
                ),
                "coalition_inputs": coalition_inputs,
            }
        )
    normalized = validate_state_projections(projections)
    if coalition_witness_count != EXPECTED_DISTANCE_ROWS:
        raise ValueError("external replay coalition witness denominator drifted")
    replay_sha = sha256_bytes(canonical_json_bytes(normalized))
    tree_files = _sequence(artifact.tree.get("files"), "derived artifact files")
    return normalized, {
        "external_input_replay_verified": True,
        "policy_or_model_forward_executed": False,
        "parent_substrate_archive_sha256": sha256_file(parent_path),
        "parent_substrate_archive_size_bytes": parent_path.stat().st_size,
        "parent_substrate_tree_inventory_sha256": (
            parent_evidence.tree_inventory_sha256
        ),
        "derived_artifact_repo": CANONICAL_DERIVED_ARTIFACT["repo"],
        "derived_artifact_revision": CANONICAL_DERIVED_ARTIFACT[
            "immutable_revision"
        ],
        "derived_artifact_tree_sha256": artifact.tree[
            "artifact_tree_sha256"
        ],
        "derived_artifact_file_count": len(tree_files),
        "ocr_backend_config_sha256": sha256_file(ocr_path),
        "ocr_backend_config_size_bytes": ocr_path.stat().st_size,
        "state_count": len(normalized),
        "coalition_input_witness_count": coalition_witness_count,
        "external_state_projections_sha256": replay_sha,
    }


def validate_external_input_replay(
    run_contract: Mapping[str, Any],
    replayed_states: Sequence[Mapping[str, Any]],
    replay_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    embedded = validate_state_projections(run_contract.get("states"))
    replayed = validate_state_projections(replayed_states)
    embedded_bytes = canonical_json_bytes(embedded)
    replayed_bytes = canonical_json_bytes(replayed)
    replay_sha = sha256_bytes(replayed_bytes)
    if embedded_bytes != replayed_bytes:
        raise ValueError(
            "external input replay differs byte-for-byte from embedded run contract"
        )
    metadata = dict(_mapping(replay_metadata, "external input replay metadata"))
    if (
        metadata.get("external_input_replay_verified") is not True
        or metadata.get("policy_or_model_forward_executed") is not False
        or metadata.get("state_count") != EXPECTED_STATE_COUNT
        or metadata.get("coalition_input_witness_count") != EXPECTED_DISTANCE_ROWS
        or metadata.get("external_state_projections_sha256") != replay_sha
    ):
        raise ValueError("external input replay metadata drifted")
    return {
        **metadata,
        "embedded_state_projections_sha256": sha256_bytes(embedded_bytes),
        "byte_identical_to_embedded_run_contract": True,
    }


def _validate_reference_teacher(value: Any, *, state_index: int) -> dict[str, Any]:
    record = dict(_mapping(value, f"state {state_index} reference teacher"))
    _exact_keys(
        record,
        {
            "canonical_action_sha256",
            "teacher_target_sha256",
            "reference_input_sha256",
            "reference_repeat_kl",
            "repeat_scalar_host_transfer_count",
            "action_token_count",
            "vocabulary_size",
            "action_log_probs_shape",
            "action_log_probs_dtype",
            "action_log_probs_device_type",
            "full_logit_tensor_host_transfer_count",
        },
        f"state {state_index} reference teacher",
    )
    for key in (
        "canonical_action_sha256",
        "teacher_target_sha256",
        "reference_input_sha256",
    ):
        _sha256(record.get(key), f"state {state_index} {key}")
    repeat = _finite_nonnegative(
        record.get("reference_repeat_kl"), f"state {state_index} repeat KL"
    )
    if repeat > MAXIMUM_REPEAT_KL:
        raise ValueError(f"state {state_index} repeat KL exceeds the frozen ceiling")
    if record.get("repeat_scalar_host_transfer_count") != 1:
        raise ValueError("reference repeat KL must perform exactly one scalar transfer")
    for key in ("action_token_count", "vocabulary_size"):
        if type(record.get(key)) is not int or record[key] <= 0:
            raise ValueError(f"state {state_index} {key} must be positive")
    if (
        record.get("action_log_probs_shape")
        != [record["action_token_count"], record["vocabulary_size"]]
        or record.get("action_log_probs_dtype") != "torch.float32"
        or record.get("action_log_probs_device_type") != "cuda"
    ):
        raise ValueError("action log-prob shape, dtype, or device metadata drifted")
    if record.get("full_logit_tensor_host_transfer_count") != 0:
        raise ValueError("full action-logit tensors must remain GPU-only")
    record["reference_repeat_kl"] = repeat
    return record


def _validate_distance_rows(
    value: Any,
    *,
    state_index: int,
    event_ids: tuple[int, ...],
    reference: Mapping[str, Any],
    coalition_input_sha256: Mapping[tuple[int, ...], str],
) -> tuple[list[dict[str, Any]], dict[tuple[int, ...], float]]:
    raw_rows = _sequence(value, f"state {state_index} distance rows")
    expected_coalitions = _coalitions(event_ids)
    if len(raw_rows) != len(expected_coalitions):
        raise ValueError(f"state {state_index} distance row count drifted")
    normalized: list[dict[str, Any]] = []
    distances: dict[tuple[int, ...], float] = {}
    for position, (raw, expected_coalition) in enumerate(
        zip(raw_rows, expected_coalitions, strict=True)
    ):
        row = dict(_mapping(raw, f"state {state_index} distance row {position}"))
        _exact_keys(
            row,
            {
                "coalition",
                "distance",
                "candidate_input_sha256",
                "teacher_forward_count",
                "kl_measurement_count",
                "scalar_host_transfer_count",
                "is_full_history_reference",
                "full_logit_tensor_host_transfer_count",
            },
            f"state {state_index} distance row {position}",
        )
        coalition_raw = _sequence(row.get("coalition"), "coalition")
        coalition = tuple(coalition_raw)
        if (
            coalition != expected_coalition
            or any(type(item) is not int for item in coalition)
            or len(set(coalition)) != len(coalition)
            or not set(coalition).issubset(event_ids)
        ):
            raise ValueError(
                f"state {state_index} has missing, duplicate, or noncanonical coalition"
            )
        if coalition in distances:
            raise ValueError(f"state {state_index} contains duplicate coalition")
        distance = _finite_nonnegative(
            row.get("distance"), f"state {state_index} coalition distance"
        )
        _sha256(row.get("candidate_input_sha256"), "candidate input SHA256")
        if row.get("candidate_input_sha256") != coalition_input_sha256.get(coalition):
            raise ValueError(
                f"state {state_index} raw input SHA differs from run-contract witness"
            )
        is_reference = coalition == event_ids
        if (
            row.get("is_full_history_reference") is not is_reference
            or row.get("teacher_forward_count") != (0 if is_reference else 1)
            or row.get("kl_measurement_count") != (0 if is_reference else 1)
            or row.get("scalar_host_transfer_count") != (0 if is_reference else 1)
            or row.get("full_logit_tensor_host_transfer_count") != 0
        ):
            raise ValueError(f"state {state_index} coalition operation accounting drifted")
        if is_reference and (
            distance != 0.0
            or row.get("candidate_input_sha256")
            != reference["reference_input_sha256"]
        ):
            raise ValueError(f"state {state_index} full-history D(S) must be canonical zero")
        row["distance"] = distance
        normalized.append(row)
        distances[coalition] = distance
    if tuple(distances) != expected_coalitions:
        raise ValueError(f"state {state_index} complete power set drifted")
    return normalized, distances


def _edge_payloads(
    event_ids: tuple[int, ...],
    distances: Mapping[tuple[int, ...], float],
    *,
    maximum_restored_events: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for base in _coalitions(event_ids):
        if len(base) >= maximum_restored_events:
            continue
        for event_id in event_ids:
            if event_id in base:
                continue
            restored = tuple(sorted((*base, event_id)))
            if len(restored) > maximum_restored_events:
                continue
            result.append(
                {
                    "base_coalition": list(base),
                    "event_id": event_id,
                    "restored_coalition": list(restored),
                    "base_distance": distances[base],
                    "restored_distance": distances[restored],
                    "marginal_gain": distances[base] - distances[restored],
                }
            )
    return result


def _interaction_payloads(
    event_ids: tuple[int, ...],
    distances: Mapping[tuple[int, ...], float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for left, right in itertools.combinations(event_ids, 2):
        remaining = tuple(item for item in event_ids if item not in {left, right})
        for conditioning in _coalitions(remaining):
            with_left = tuple(sorted((*conditioning, left)))
            with_right = tuple(sorted((*conditioning, right)))
            with_pair = tuple(sorted((*conditioning, left, right)))
            result.append(
                {
                    "conditioning_coalition": list(conditioning),
                    "left_event_id": left,
                    "right_event_id": right,
                    "interaction": (
                        distances[with_left]
                        + distances[with_right]
                        - distances[conditioning]
                        - distances[with_pair]
                    ),
                }
            )
    return result


def _attribution_payloads(
    event_ids: tuple[int, ...],
    distances: Mapping[tuple[int, ...], float],
) -> list[dict[str, Any]]:
    samples = {event_id: [] for event_id in event_ids}
    permutations = tuple(itertools.permutations(event_ids))
    for permutation in permutations:
        prefix: tuple[int, ...] = ()
        for event_id in permutation:
            restored = tuple(sorted((*prefix, event_id)))
            samples[event_id].append(distances[prefix] - distances[restored])
            prefix = restored
    return [
        {
            "event_id": event_id,
            "mean_marginal_gain": math.fsum(samples[event_id]) / len(permutations),
            "permutation_count": len(permutations),
            "marginal_samples": samples[event_id],
        }
        for event_id in event_ids
    ]


def _oracle_payload(
    event_ids: tuple[int, ...],
    distances: Mapping[tuple[int, ...], float],
) -> dict[str, Any]:
    feasible = [
        coalition
        for coalition in _coalitions(event_ids)
        if len(coalition) <= PRIMARY_BUDGET_EVENT_CAPACITY
    ]
    selected = min(feasible, key=lambda item: (distances[item], len(item), item))
    return {
        "budget_event_capacity": PRIMARY_BUDGET_EVENT_CAPACITY,
        "tie_epsilon": 0.0,
        "coalition": list(selected),
        "distance": distances[selected],
        "utility": distances[()] - distances[selected],
        "evaluated_coalition_count": len(feasible),
    }


def _algebra_residuals(
    event_ids: tuple[int, ...],
    distances: Mapping[tuple[int, ...], float],
    attributions: Sequence[Mapping[str, Any]],
) -> tuple[float, float]:
    total = distances[()] - distances[event_ids]
    telescoping_residuals: list[float] = []
    for permutation in itertools.permutations(event_ids):
        prefix: tuple[int, ...] = ()
        gains: list[float] = []
        for event_id in permutation:
            restored = tuple(sorted((*prefix, event_id)))
            gains.append(distances[prefix] - distances[restored])
            prefix = restored
        telescoping_residuals.append(abs(math.fsum(gains) - total))
    shapley_residual = abs(
        math.fsum(float(row["mean_marginal_gain"]) for row in attributions)
        - total
    )
    return max(telescoping_residuals), shapley_residual


def validate_raw_state_record(
    value: Any,
    *,
    expected_state: Mapping[str, Any],
    expected_worker: WorkerSpec,
    run_contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    index = int(expected_state["state_index"])
    record = dict(_mapping(value, f"state record {index}"))
    _exact_keys(
        record,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "worker",
            "state",
            "reference_teacher",
            "distance_rows",
            "operation_counts",
        },
        f"state record {index}",
    )
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("protocol_id") != PROTOCOL_ID
        or record.get("status") != STATE_STATUS
        or record.get("run_contract_sha256") != run_contract_sha256
        or record.get("worker") != expected_worker.envelope_identity()
        or record.get("state") != expected_state
    ):
        raise ValueError(f"state record {index} envelope drifted")
    reference = _validate_reference_teacher(record.get("reference_teacher"), state_index=index)
    if reference["canonical_action_sha256"] != expected_state["canonical_action_sha256"]:
        raise ValueError(f"state {index} raw reference action differs from parent witness")
    if reference["teacher_target_sha256"] != expected_state["teacher_target_sha256"]:
        raise ValueError(f"state {index} raw teacher target differs from frozen serializer witness")
    event_ids = tuple(expected_state["candidate_event_step_ids"])
    coalition_input_sha256 = {
        tuple(item["coalition"]): str(item["input_sha256"])
        for item in expected_state["coalition_inputs"]
    }
    rows, distances = _validate_distance_rows(
        record.get("distance_rows"),
        state_index=index,
        event_ids=event_ids,
        reference=reference,
        coalition_input_sha256=coalition_input_sha256,
    )
    counts = _state_operation_counts(len(event_ids))
    if record.get("operation_counts") != counts:
        raise ValueError(f"state {index} worker schedule, retry, or top-up drifted")
    normalized = dict(record)
    normalized["reference_teacher"] = reference
    normalized["distance_rows"] = rows
    attribution = _attribution_payloads(event_ids, distances)
    telescoping_residual, shapley_residual = _algebra_residuals(
        event_ids, distances, attribution
    )
    if (
        telescoping_residual > ALGEBRA_RESIDUAL_TOLERANCE
        or shapley_residual > ALGEBRA_RESIDUAL_TOLERANCE
    ):
        raise ValueError(f"state {index} restoration algebra residual exceeds tolerance")
    derived = {
        "state": dict(expected_state),
        "baseline_summary_only_distance": distances[()],
        "full_history_distance": distances[event_ids],
        "deployment_conditional_edges": _edge_payloads(
            event_ids,
            distances,
            maximum_restored_events=PRIMARY_BUDGET_EVENT_CAPACITY,
        ),
        "full_hypercube_edges": _edge_payloads(
            event_ids,
            distances,
            maximum_restored_events=len(event_ids),
        ),
        "pair_interactions": _interaction_payloads(event_ids, distances),
        "exact_permutation_attribution": attribution,
        "primary_exact_subset_oracle": _oracle_payload(event_ids, distances),
        "telescoping_max_abs_residual": telescoping_residual,
        "shapley_efficiency_abs_residual": shapley_residual,
    }
    return normalized, derived


def reduce_raw_distance_states(
    state_records: Sequence[Mapping[str, Any]],
    *,
    expected_states: Sequence[Mapping[str, Any]],
    run_contract_sha256: str,
) -> dict[str, Any]:
    """Independently derive every scientific label from complete raw D(S)."""
    if len(state_records) != EXPECTED_STATE_COUNT or len(expected_states) != EXPECTED_STATE_COUNT:
        raise ValueError("raw reduction requires the exact fixed 192-state denominator")
    workers = expected_worker_specs()
    worker_by_parity = {spec.index_parity: spec for spec in workers}
    normalized_states: list[dict[str, Any]] = []
    derived_states: list[dict[str, Any]] = []
    repeat_values: list[float] = []
    for index, (raw, expected) in enumerate(zip(state_records, expected_states, strict=True)):
        normalized, derived = validate_raw_state_record(
            raw,
            expected_state=expected,
            expected_worker=worker_by_parity[index % 2],
            run_contract_sha256=run_contract_sha256,
        )
        normalized_states.append(normalized)
        derived_states.append(derived)
        repeat_values.append(float(normalized["reference_teacher"]["reference_repeat_kl"]))

    deployment = [edge for state in derived_states for edge in state["deployment_conditional_edges"]]
    full = [edge for state in derived_states for edge in state["full_hypercube_edges"]]
    interactions = [item for state in derived_states for item in state["pair_interactions"]]
    attributions = [item for state in derived_states for item in state["exact_permutation_attribution"]]
    counts = {
        "trajectory_count": len({state["state"]["source_id"] for state in derived_states}),
        "state_count": len(derived_states),
        "raw_distance_row_count": sum(len(record["distance_rows"]) for record in normalized_states),
        "deployment_conditional_edge_count": len(deployment),
        "full_hypercube_edge_count": len(full),
        "pair_interaction_count": len(interactions),
        "exact_permutation_attribution_count": len(attributions),
        "primary_exact_subset_oracle_count": len(derived_states),
        "teacher_forward_count": sum(record["operation_counts"]["teacher_forward_count"] for record in normalized_states),
        "kl_measurement_count": sum(record["operation_counts"]["kl_measurement_count"] for record in normalized_states),
        "scalar_host_transfer_count": sum(record["operation_counts"]["scalar_host_transfer_count"] for record in normalized_states),
        "full_logit_tensor_host_transfer_count": sum(record["operation_counts"]["full_logit_tensor_host_transfer_count"] for record in normalized_states),
        "retry_count": sum(record["operation_counts"]["retry_count"] for record in normalized_states),
        "top_up_count": sum(record["operation_counts"]["top_up_count"] for record in normalized_states),
    }
    if counts != EXPECTED_TOTALS:
        raise ValueError(f"raw reduction totals drifted: {counts}")
    marginal_values = [float(edge["marginal_gain"]) for edge in full]
    interaction_values = [float(item["interaction"]) for item in interactions]
    baselines = [float(state["baseline_summary_only_distance"]) for state in derived_states]
    oracle_utilities = [float(state["primary_exact_subset_oracle"]["utility"]) for state in derived_states]
    summary = {
        "role_state_counts": dict(sorted(Counter(state["state"]["role"] for state in derived_states).items())),
        "reference_repeat_kl": {
            "maximum": max(repeat_values),
            "mean": math.fsum(repeat_values) / len(repeat_values),
            "threshold": MAXIMUM_REPEAT_KL,
        },
        "marginal_sign_counts": {
            "negative": sum(value < 0 for value in marginal_values),
            "zero": sum(value == 0 for value in marginal_values),
            "positive": sum(value > 0 for value in marginal_values),
        },
        "interaction_sign_counts": {
            "negative": sum(value < 0 for value in interaction_values),
            "zero": sum(value == 0 for value in interaction_values),
            "positive": sum(value > 0 for value in interaction_values),
        },
        "nonmonotone_state_count": sum(
            any(float(edge["marginal_gain"]) < 0 for edge in state["full_hypercube_edges"])
            for state in derived_states
        ),
        "mean_summary_only_distance": math.fsum(baselines) / len(baselines),
        "mean_primary_exact_oracle_utility": math.fsum(oracle_utilities) / len(oracle_utilities),
        "algebra_residual_gate": {
            "tolerance": ALGEBRA_RESIDUAL_TOLERANCE,
            "maximum_telescoping_abs_residual": max(
                float(state["telescoping_max_abs_residual"])
                for state in derived_states
            ),
            "maximum_shapley_efficiency_abs_residual": max(
                float(state["shapley_efficiency_abs_residual"])
                for state in derived_states
            ),
            "passed": True,
        },
    }
    return {
        "counts": counts,
        "summary": summary,
        "states": derived_states,
        "derived_payload_sha256": sha256_bytes(canonical_json_bytes(derived_states)),
    }


def expected_file_names() -> frozenset[str]:
    names = {
        GLOBAL_LEDGER_ARCHIVE_NAME,
        RUN_MANIFEST_FILENAME,
        AGGREGATE_FILENAME,
        EXECUTION_EVIDENCE_FILENAME,
        MONITOR_SUMMARY_FILENAME,
        *LOG_PATHS.values(),
        f"{WORKER_SIBLING_LEDGER_DIRECTORY}/even.json",
        f"{WORKER_SIBLING_LEDGER_DIRECTORY}/odd.json",
    }
    for spec in expected_worker_specs():
        base = f"{WORKER_DIRECTORY}/{spec.worker_id}"
        names.update(
            {
                f"{base}/{WORKER_LEDGER_FILENAME}",
                f"{base}/{WORKER_RUNTIME_FILENAME}",
                f"{base}/{WORKER_TERMINAL_FILENAME}",
            }
        )
        for index in spec.state_indices:
            names.add(f"{base}/{ATTEMPT_DIRECTORY}/{index:03d}.json")
            names.add(f"{base}/{STATE_DIRECTORY}/{index:03d}.json")
    return frozenset(names)


def _json(files: Mapping[str, bytes], name: str) -> dict[str, Any]:
    try:
        payload = files[name]
    except KeyError as error:
        raise ValueError(f"required evidence member is missing: {name}") from error
    return strict_json_object_bytes(payload, label=name)


def _file_witness(value: Any, label: str) -> dict[str, Any]:
    record = dict(_mapping(value, label))
    _exact_keys(record, {"path", "sha256", "size_bytes"}, label)
    _safe_relative_path(record.get("path"), f"{label} path")
    _sha256(record.get("sha256"), f"{label} SHA256")
    if type(record.get("size_bytes")) is not int or record["size_bytes"] < 0:
        raise ValueError(f"{label} size must be non-negative")
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


def runner_interface_contract() -> dict[str, Any]:
    return {
        "runner_module": "scripts.run_restoration_v2_2_expansion_labels",
        "manager_module": (
            "scripts.manage_restoration_v2_2_expansion_labels_artifact"
        ),
        "required_cli_options": [
            "--repository-root",
            "--contract",
            "--runner-freeze",
            "--parent-substrate-archive",
            "--derived-artifact-root",
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
            "required_cli_options": ["--ready-file", "--log-file", "--summary-file"],
            "stop_request_path_derivation": "<summary-file>.stop-request.json",
            "visible_gpu_count": 2,
            "ready_before_first_teacher_forward_required": True,
            "sample_cadence_seconds": MONITOR_SAMPLE_CADENCE_SECONDS,
            "sample_tolerance_seconds": MONITOR_SAMPLE_TOLERANCE_SECONDS,
            "maximum_sample_gap_seconds": MONITOR_MAX_SAMPLE_GAP_SECONDS,
            "full_worker_execution_window_required": True,
            "scientific_gate_inputs": [],
        },
        "raw_state_schema": {
            "state_fields": [
                "schema_version",
                "protocol_id",
                "status",
                "run_contract_sha256",
                "worker",
                "state",
                "reference_teacher",
                "distance_rows",
                "operation_counts",
            ],
            "distance_row_fields": [
                "coalition",
                "distance",
                "candidate_input_sha256",
                "teacher_forward_count",
                "kl_measurement_count",
                "scalar_host_transfer_count",
                "is_full_history_reference",
                "full_logit_tensor_host_transfer_count",
            ],
            "producer_derived_label_fields": [],
            "raw_distance_is_only_canonical_scientific_truth": True,
        },
        "frozen_totals": EXPECTED_TOTALS,
        "prohibited_operation_counts": PROHIBITED_OPERATION_COUNTS,
        "dynamic_input_chain": {
            "derived_artifact": CANONICAL_DERIVED_ARTIFACT,
            "substrate_state_projection_sha256": (
                FROZEN_SUBSTRATE_STATE_PROJECTION_SHA256
            ),
            "coalition_order": "cardinality_then_lexicographic_event_step_ids",
            "raw_input_sha_must_equal_run_contract_witness": True,
        },
        "worker_topology": [spec.to_dict() for spec in expected_worker_specs()],
        "archive": {
            "format": ARCHIVE_FORMAT,
            "member_prefix": ARCHIVE_MEMBER_PREFIX,
            "hf_repo": CANONICAL_HF_REPO,
            "hf_tag": CANONICAL_HF_TAG,
            "hf_path": CANONICAL_HF_PATH,
        },
    }


def validate_runner_freeze_data(
    value: Any,
    *,
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    freeze = dict(_mapping(value, "runner freeze"))
    _exact_keys(
        freeze,
        {
            "schema_version",
            "protocol_id",
            "status",
            "runner_source_git_commit",
            "base_config",
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
    base_witness = _file_witness(freeze.get("base_config"), "runner base config")
    base_payload = pretty_json_bytes(base_config)
    if (
        freeze.get("schema_version") != RUNNER_FREEZE_SCHEMA_VERSION
        or freeze.get("protocol_id") != RUNNER_FREEZE_PROTOCOL_ID
        or freeze.get("status")
        != "FROZEN_COMMITTED_PUSHED_EXPANSION_EXACT_LABEL_RUNNER_SOURCE"
        or base_witness.get("path") != CANONICAL_CONFIG_PATH
        or base_witness.get("sha256") != sha256_bytes(base_payload)
        or base_witness.get("size_bytes") != len(base_payload)
    ):
        raise ValueError("runner-freeze base identity drifted")
    source_lock = _mapping(base_config.get("scientific_source_lock"), "source lock")
    locked = [
        _file_witness(raw, f"locked source {index}")
        for index, raw in enumerate(
            _sequence(source_lock.get("source_files"), "source files")
        )
    ]
    reserved = list(
        _sequence(
            source_lock.get("reserved_execution_source_paths"),
            "reserved execution source paths",
        )
    )
    if set(reserved) != SOURCE_PATHS_REQUIRED_IN_RUNNER_FREEZE or len(reserved) != 6:
        raise ValueError("reserved expansion-label execution sources drifted")
    inventory = [
        _file_witness(raw, f"runner source {index}")
        for index, raw in enumerate(
            _sequence(freeze.get("source_inventory"), "runner source inventory")
        )
    ]
    expected_paths = {record["path"] for record in locked} | {
        CANONICAL_CONFIG_PATH,
        *reserved,
    }
    if (
        [record["path"] for record in inventory] != sorted(expected_paths)
        or len(inventory) != len(expected_paths)
        or len(expected_paths) != len(locked) + 7
    ):
        raise ValueError("runner source inventory is not locked+base+reserved")
    by_path = {record["path"]: record for record in inventory}
    if any(by_path[record["path"]] != record for record in locked):
        raise ValueError("inherited scientific source witness drifted")
    if by_path[CANONICAL_CONFIG_PATH] != base_witness:
        raise ValueError("runner inventory base witness drifted")
    inventory_sha = sha256_bytes(canonical_json_bytes(inventory))
    if freeze.get("source_inventory_sha256") != inventory_sha:
        raise ValueError("runner source inventory SHA256 drifted")
    if freeze.get("interfaces") != runner_interface_contract():
        raise ValueError("runner interface contract drifted")
    if freeze.get("authorization") != {
        "runner_source_commit_must_be_committed": True,
        "runner_source_commit_must_be_ancestor_of_execution_head": True,
        "execution_head_and_origin_main_must_match": True,
        "runner_freeze_file_must_be_committed_at_execution_head": True,
        "source_blobs_must_equal_runner_source_commit": True,
        "source_only_freeze_authorizes_gpu_execution": False,
        "formal_execution_requires_runtime_authorization_validation": True,
    }:
        raise ValueError("runner authorization declarations drifted")
    return {
        "runner_source_git_commit": runner_commit,
        "base_config_sha256": base_witness["sha256"],
        "source_inventory": inventory,
        "source_inventory_sha256": inventory_sha,
    }


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


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_blob(root: Path, revision: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(f"{relative} is not committed at {revision}")
    return result.stdout


def build_runner_freeze(
    *,
    repository_root: str | Path,
    runner_source_git_commit: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    _git_sha(runner_source_git_commit, "runner source commit")
    base_path = _repository_file(root, CANONICAL_CONFIG_PATH)
    base_payload = base_path.read_bytes()
    base_config = strict_json_object_bytes(base_payload, label="base config")
    source_lock = _mapping(base_config.get("scientific_source_lock"), "source lock")
    locked = [
        _file_witness(raw, f"locked source {index}")
        for index, raw in enumerate(
            _sequence(source_lock.get("source_files"), "source files")
        )
    ]
    reserved = list(
        _sequence(
            source_lock.get("reserved_execution_source_paths"),
            "reserved execution source paths",
        )
    )
    if set(reserved) != SOURCE_PATHS_REQUIRED_IN_RUNNER_FREEZE or len(reserved) != 6:
        raise ValueError("reserved source inventory drifted")
    inventory = []
    for record in locked:
        observed = _repository_identity(root, record["path"])
        if observed != record:
            raise ValueError(f"locked source drifted: {record['path']}")
        inventory.append(observed)
    inventory.append(_repository_identity(root, CANONICAL_CONFIG_PATH))
    inventory.extend(_repository_identity(root, str(path)) for path in reserved)
    inventory.sort(key=lambda record: record["path"])
    if len({record["path"] for record in inventory}) != len(inventory):
        raise ValueError("runner source inventory contains duplicate paths")
    for record in inventory:
        if _git_blob(root, runner_source_git_commit, record["path"]) != _repository_file(
            root, record["path"]
        ).read_bytes():
            raise ValueError(f"runner source differs from commit A: {record['path']}")
    freeze = {
        "schema_version": RUNNER_FREEZE_SCHEMA_VERSION,
        "protocol_id": RUNNER_FREEZE_PROTOCOL_ID,
        "status": "FROZEN_COMMITTED_PUSHED_EXPANSION_EXACT_LABEL_RUNNER_SOURCE",
        "runner_source_git_commit": runner_source_git_commit,
        "base_config": _repository_identity(root, CANONICAL_CONFIG_PATH),
        "source_inventory": inventory,
        "source_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "interfaces": runner_interface_contract(),
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
    validate_runner_freeze_data(freeze, base_config=base_config)
    return freeze


def materialize_runner_freeze(
    *,
    repository_root: str | Path,
    runner_source_git_commit: str,
    output_path: str | Path = CANONICAL_RUNNER_FREEZE_PATH,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    expected = root / CANONICAL_RUNNER_FREEZE_PATH
    if output.resolve() != expected.resolve():
        raise ValueError("runner freeze output path is not canonical")
    head = _git_output(root, "rev-parse", "HEAD")
    origin = _git_output(root, "rev-parse", "origin/main")
    if head != runner_source_git_commit or origin != runner_source_git_commit:
        raise ValueError("commit A must equal clean HEAD and origin/main")
    if _git_output(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("runner freeze materialization requires a clean worktree")
    if output.exists():
        raise FileExistsError("runner freeze already exists")
    freeze = build_runner_freeze(
        repository_root=root,
        runner_source_git_commit=runner_source_git_commit,
    )
    atomic_exclusive_publish_bytes(output, pretty_json_bytes(freeze))
    return {
        "status": "MATERIALIZED_EXPANSION_EXACT_LABEL_RUNNER_FREEZE_PENDING_COMMIT_B",
        "output": str(output),
        "sha256": sha256_file(output),
        "runner_source_git_commit": runner_source_git_commit,
        "policy_or_gpu_execution_authorized": False,
    }


def load_and_validate_runner_freeze(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    freeze_path = Path(path)
    if not freeze_path.is_absolute():
        freeze_path = root / freeze_path
    if freeze_path.resolve() != (root / CANONICAL_RUNNER_FREEZE_PATH).resolve():
        raise ValueError("runner freeze path is not canonical")
    freeze = strict_json_object_bytes(freeze_path.read_bytes(), label="runner freeze")
    base_config = strict_json_object_bytes(
        _repository_file(root, CANONICAL_CONFIG_PATH).read_bytes(), label="base config"
    )
    validation = validate_runner_freeze_data(freeze, base_config=base_config)
    head = _git_output(root, "rev-parse", "HEAD")
    origin = _git_output(root, "rev-parse", "origin/main")
    branch = _git_output(root, "branch", "--show-current")
    remote_url = _git_output(root, "remote", "get-url", "origin")
    if (
        branch != "main"
        or head != origin
        or remote_url != CANONICAL_GIT_ORIGIN_URL
        or _git_output(root, "status", "--porcelain", "--untracked-files=all")
    ):
        raise ValueError("formal execution requires clean pushed canonical main")
    runner_commit = validation["runner_source_git_commit"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", runner_commit, head], cwd=root
    )
    if ancestor.returncode != 0:
        raise ValueError("runner source commit A is not an ancestor of execution B")
    if _git_blob(root, head, CANONICAL_RUNNER_FREEZE_PATH) != freeze_path.read_bytes():
        raise ValueError("runner freeze is not committed at execution B")
    for record in validation["source_inventory"]:
        if _git_blob(root, runner_commit, record["path"]) != _repository_file(
            root, record["path"]
        ).read_bytes():
            raise ValueError(f"execution source differs from commit A: {record['path']}")
    return freeze, {
        **validation,
        "execution_git_commit": head,
        "runner_freeze_sha256": sha256_file(freeze_path),
        "policy_or_gpu_execution_authorized": True,
    }


def validate_execution_run_contract(
    value: Any,
    *,
    expected_source_git_commit: str | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    contract = dict(_mapping(value, "execution run contract"))
    _exact_keys(
        contract,
        {
            "schema_version",
            "protocol_id",
            "status",
            "runner_source_git_commit",
            "execution_git_commit",
            "config",
            "runner_freeze",
            "parent_substrate",
            "derived_artifact",
            "states",
            "worker_topology",
            "planned_counts",
            "prohibited_operation_counts",
            "attempt_identity",
            "execution_argv",
        },
        "execution run contract",
    )
    source_commit = _git_sha(
        contract.get("runner_source_git_commit"), "runner source Git commit"
    )
    execution_commit = _git_sha(
        contract.get("execution_git_commit"), "execution Git commit"
    )
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("protocol_id") != PROTOCOL_ID
        or contract.get("status") != "AUTHORIZED_EXPANSION_EXACT_LABEL_EXECUTION"
        or (
            expected_source_git_commit is not None
            and source_commit != expected_source_git_commit
        )
    ):
        raise ValueError("execution run contract identity drifted")
    config = _file_witness(contract.get("config"), "run config")
    if config["path"] != CANONICAL_CONFIG_PATH or (
        expected_config_sha256 is not None
        and config["sha256"] != expected_config_sha256
    ):
        raise ValueError("execution config identity drifted")
    runner_freeze = dict(_mapping(contract.get("runner_freeze"), "runner freeze witness"))
    _exact_keys(
        runner_freeze,
        {"path", "sha256", "size_bytes", "runner_source_git_commit"},
        "runner freeze witness",
    )
    _safe_relative_path(runner_freeze.get("path"), "runner freeze path")
    _sha256(runner_freeze.get("sha256"), "runner freeze SHA256")
    if (
        runner_freeze.get("path") != CANONICAL_RUNNER_FREEZE_PATH
        or runner_freeze.get("runner_source_git_commit") != source_commit
        or type(runner_freeze.get("size_bytes")) is not int
        or runner_freeze["size_bytes"] <= 0
    ):
        raise ValueError("runner freeze execution binding drifted")
    if contract.get("parent_substrate") != CANONICAL_PARENT_SUBSTRATE:
        raise ValueError("parent substrate immutable artifact drifted")
    if contract.get("derived_artifact") != CANONICAL_DERIVED_ARTIFACT:
        raise ValueError("derived artifact immutable input drifted")
    states = validate_state_projections(contract.get("states"))
    if contract.get("worker_topology") != [
        spec.to_dict() for spec in expected_worker_specs()
    ]:
        raise ValueError("execution worker topology drifted")
    if contract.get("planned_counts") != EXPECTED_TOTALS:
        raise ValueError("execution planned totals drifted")
    if contract.get("prohibited_operation_counts") != PROHIBITED_OPERATION_COUNTS:
        raise ValueError("execution prohibited-operation counts must be exact zero")
    attempt = dict(_mapping(contract.get("attempt_identity"), "attempt identity"))
    _exact_keys(
        attempt,
        {"attempt_id", "output_dir", "global_ledger_path", "archive_path"},
        "attempt identity",
    )
    if attempt != {
        "attempt_id": ATTEMPT_ID,
        "output_dir": str(CANONICAL_OUTPUT_DIR),
        "global_ledger_path": str(CANONICAL_LEDGER_PATH),
        "archive_path": str(CANONICAL_ARCHIVE_PATH),
    }:
        raise ValueError("canonical attempt identity drifted")
    argv = _sequence(contract.get("execution_argv"), "execution argv")
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ValueError("execution argv must be a nonempty string array")
    observed_options = [item for item in argv if item.startswith("--")]
    if observed_options != runner_interface_contract()["required_cli_options"]:
        raise ValueError("execution argv option inventory or order drifted")
    return {
        "contract": contract,
        "runner_source_git_commit": source_commit,
        "execution_git_commit": execution_commit,
        "config_sha256": config["sha256"],
        "states": states,
    }


def _validate_global_ledger(
    value: Any,
    *,
    run_contract_sha256: str,
) -> None:
    ledger = dict(_mapping(value, "global attempt ledger"))
    _exact_keys(
        ledger,
        {
            "schema_version",
            "protocol_id",
            "status",
            "attempt_id",
            "run_contract_sha256",
            "output_dir",
            "worker_sibling_ledgers",
            "attempted_state_indices",
            "completed_state_indices",
            "retry_count",
            "top_up_count",
            "started_at_utc",
            "ended_at_utc",
        },
        "global attempt ledger",
    )
    expected_indices = list(range(EXPECTED_STATE_COUNT))
    expected_siblings = {
        spec.worker_id: str(
            CANONICAL_OUTPUT_DIR.parent
            / f".{ATTEMPT_ID}.{spec.worker_id}.attempt.json"
        )
        for spec in expected_worker_specs()
    }
    if (
        ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("protocol_id") != PROTOCOL_ID
        or ledger.get("status") != "COMPLETED_EXPANSION_EXACT_LABEL_ATTEMPT"
        or ledger.get("attempt_id") != ATTEMPT_ID
        or ledger.get("run_contract_sha256") != run_contract_sha256
        or ledger.get("output_dir") != str(CANONICAL_OUTPUT_DIR)
        or ledger.get("worker_sibling_ledgers") != expected_siblings
        or ledger.get("attempted_state_indices") != expected_indices
        or ledger.get("completed_state_indices") != expected_indices
        or ledger.get("retry_count") != 0
        or ledger.get("top_up_count") != 0
    ):
        raise ValueError("global worker schedule, retry, or top-up ledger drifted")
    if _timestamp(ledger.get("started_at_utc"), "global start") > _timestamp(
        ledger.get("ended_at_utc"), "global end"
    ):
        raise ValueError("global ledger timestamp bracket is reversed")


def _validate_runtime(
    value: Any,
    *,
    spec: WorkerSpec,
    run_contract_sha256: str,
) -> tuple[str, tuple[str, str], Mapping[str, Any]]:
    outer = dict(_mapping(value, f"{spec.worker_id} runtime identity"))
    _exact_keys(
        outer,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "worker",
            "runtime_metadata",
        },
        f"{spec.worker_id} runtime identity",
    )
    if (
        outer.get("schema_version") != SCHEMA_VERSION
        or outer.get("protocol_id") != PROTOCOL_ID
        or outer.get("status") != "VALIDATED_EXPANSION_EXACT_LABEL_WORKER_RUNTIME"
        or outer.get("run_contract_sha256") != run_contract_sha256
        or outer.get("worker") != spec.envelope_identity()
    ):
        raise ValueError(f"{spec.worker_id} runtime envelope drifted")
    runtime = dict(_mapping(outer.get("runtime_metadata"), "runtime metadata"))
    _exact_keys(
        runtime,
        {
            "container_id",
            "host_alias",
            "hostname",
            "visible_gpu_count",
            "visible_gpu_uuids",
            "worker_gpu_uuid",
            "worker_cuda_device",
            "bindings_metadata",
            "bindings_metadata_sha256",
        },
        f"{spec.worker_id} runtime metadata",
    )
    uuids = tuple(_sequence(runtime.get("visible_gpu_uuids"), "visible GPU UUIDs"))
    bindings = dict(_mapping(runtime.get("bindings_metadata"), "bindings metadata"))
    if runtime.get("bindings_metadata_sha256") != sha256_bytes(
        canonical_json_bytes(bindings)
    ):
        raise ValueError("full bindings metadata SHA256 drifted")
    validate_eager_worker_runtime_metadata(bindings, spec=spec)
    if (
        not isinstance(runtime.get("container_id"), str)
        or _CONTAINER_ID.fullmatch(runtime["container_id"]) is None
        or runtime.get("visible_gpu_count") != 2
        or len(uuids) != 2
        or len(set(uuids)) != 2
        or any(not isinstance(item, str) or _GPU_UUID.fullmatch(item) is None for item in uuids)
        or runtime.get("worker_gpu_uuid") != uuids[spec.index_parity]
        or runtime.get("worker_cuda_device") != spec.device
        or bindings.get("gpu_uuid") != runtime.get("worker_gpu_uuid")
        or bindings.get("device") != runtime.get("worker_cuda_device")
        or any(bindings.get(key) != value for key, value in CANONICAL_RUNTIME.items())
        or runtime.get("host_alias") not in {"hyper00", "hyper01"}
        or runtime.get("hostname")
        != {
            "hyper00": "node-radixark-16-0000",
            "hyper01": "node-radixark-16-0001",
        }.get(runtime.get("host_alias"))
    ):
        raise ValueError(f"{spec.worker_id} runtime GPU binding drifted")
    return (
        str(runtime["worker_gpu_uuid"]),
        (str(uuids[0]), str(uuids[1])),
        bindings,
    )


def _worker_operation_totals(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    keys = tuple(_state_operation_counts(2))
    return {
        key: sum(int(record["operation_counts"][key]) for record in records)
        for key in keys
    }


def _validate_worker_evidence(
    files: Mapping[str, bytes],
    *,
    spec: WorkerSpec,
    run_contract_sha256: str,
    states: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    str,
    tuple[str, str],
    Mapping[str, Any],
    datetime,
    datetime,
]:
    base = f"{WORKER_DIRECTORY}/{spec.worker_id}"
    expected_indices = list(spec.state_indices)
    ledger = _json(files, f"{base}/{WORKER_LEDGER_FILENAME}")
    sibling = _json(
        files, f"{WORKER_SIBLING_LEDGER_DIRECTORY}/{spec.worker_id}.json"
    )
    if sibling != ledger:
        raise ValueError(f"{spec.worker_id} root and sibling ledgers differ")
    _exact_keys(
        ledger,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "worker",
            "attempted_state_indices",
            "completed_state_indices",
            "retry_count",
            "top_up_count",
            "started_at_utc",
            "ended_at_utc",
        },
        f"{spec.worker_id} ledger",
    )
    if (
        ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("protocol_id") != PROTOCOL_ID
        or ledger.get("status") != WORKER_STATUS
        or ledger.get("run_contract_sha256") != run_contract_sha256
        or ledger.get("worker") != spec.to_dict()
        or ledger.get("attempted_state_indices") != expected_indices
        or ledger.get("completed_state_indices") != expected_indices
        or ledger.get("retry_count") != 0
        or ledger.get("top_up_count") != 0
    ):
        raise ValueError(f"{spec.worker_id} worker schedule, retry, or top-up drifted")
    if _timestamp(ledger.get("started_at_utc"), "worker start") > _timestamp(
        ledger.get("ended_at_utc"), "worker end"
    ):
        raise ValueError("worker ledger timestamp bracket is reversed")
    worker_uuid, visible_uuids, bindings_metadata = _validate_runtime(
        _json(files, f"{base}/{WORKER_RUNTIME_FILENAME}"),
        spec=spec,
        run_contract_sha256=run_contract_sha256,
    )
    raw_states: list[dict[str, Any]] = []
    claim_times: list[datetime] = []
    for index in spec.state_indices:
        marker = _json(files, f"{base}/{ATTEMPT_DIRECTORY}/{index:03d}.json")
        _exact_keys(
            marker,
            {
                "schema_version",
                "protocol_id",
                "status",
                "run_contract_sha256",
                "worker",
                "state_index",
                "retry_count",
                "top_up_count",
                "claimed_at_utc",
            },
            f"state {index} marker",
        )
        if (
            marker.get("schema_version") != SCHEMA_VERSION
            or marker.get("protocol_id") != PROTOCOL_ID
            or marker.get("status") != "CLAIMED_EXPANSION_EXACT_LABEL_STATE_NO_RETRY"
            or marker.get("run_contract_sha256") != run_contract_sha256
            or marker.get("worker") != spec.envelope_identity()
            or marker.get("state_index") != index
            or marker.get("retry_count") != 0
            or marker.get("top_up_count") != 0
        ):
            raise ValueError(f"state {index} attempt marker schedule drifted")
        claim_times.append(
            _timestamp(marker.get("claimed_at_utc"), f"state {index} claim time")
        )
        raw_states.append(_json(files, f"{base}/{STATE_DIRECTORY}/{index:03d}.json"))
    terminal = _json(files, f"{base}/{WORKER_TERMINAL_FILENAME}")
    _exact_keys(
        terminal,
        {
            "schema_version",
            "protocol_id",
            "status",
            "outcome",
            "run_contract_sha256",
            "worker",
            "attempted_state_indices",
            "completed_state_indices",
            "operation_counts",
            "failure",
            "ended_at_utc",
        },
        f"{spec.worker_id} terminal",
    )
    # note (luojiaxuan): State algebra is validated again by the global reducer;
    # this local total only closes the worker schedule before cross-worker merge.
    preliminary: list[dict[str, Any]] = []
    for raw, index in zip(raw_states, spec.state_indices, strict=True):
        normalized, _derived = validate_raw_state_record(
            raw,
            expected_state=states[index],
            expected_worker=spec,
            run_contract_sha256=run_contract_sha256,
        )
        preliminary.append(normalized)
    if (
        terminal.get("schema_version") != SCHEMA_VERSION
        or terminal.get("protocol_id") != PROTOCOL_ID
        or terminal.get("status") != "TERMINAL_EXPANSION_EXACT_LABEL_WORKER"
        or terminal.get("outcome") != WORKER_STATUS
        or terminal.get("run_contract_sha256") != run_contract_sha256
        or terminal.get("worker") != spec.to_dict()
        or terminal.get("attempted_state_indices") != expected_indices
        or terminal.get("completed_state_indices") != expected_indices
        or terminal.get("operation_counts") != _worker_operation_totals(preliminary)
        or terminal.get("failure") is not None
    ):
        raise ValueError(f"{spec.worker_id} terminal schedule drifted")
    terminal_time = _timestamp(
        terminal.get("ended_at_utc"), "worker terminal time"
    )
    if terminal_time < max(claim_times):
        raise ValueError("worker terminal predates its final state claim")
    return (
        preliminary,
        worker_uuid,
        visible_uuids,
        bindings_metadata,
        min(claim_times),
        terminal_time,
    )


def _validate_monitor_jsonl(
    payload: bytes,
    *,
    runtime_visible_uuids: tuple[str, str],
    runtime_physical_inventory: tuple[tuple[int, str], tuple[int, str]],
) -> tuple[list[datetime], int]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("utilization monitor JSONL must be nonempty and LF-terminated")
    lines = payload[:-1].split(b"\n")
    timestamps: list[datetime] = []
    low_incidents = 0
    frozen_physical_inventory: tuple[tuple[int, str], tuple[int, str]] | None = None
    for index, line in enumerate(lines):
        try:
            sample = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("utilization monitor JSONL is not strict JSON") from error
        if not isinstance(sample, dict) or canonical_json_bytes(sample) != line:
            raise ValueError("utilization monitor sample is not canonical JSON")
        _exact_keys(
            sample,
            {
                "schema_version",
                "status",
                "sample_index",
                "sampled_at_utc",
                "visible_gpu_count",
                "gpus",
            },
            f"monitor sample {index}",
        )
        gpus = _sequence(sample.get("gpus"), f"monitor sample {index} GPUs")
        if (
            sample.get("schema_version") != SCHEMA_VERSION
            or sample.get("status") != "GPU_UTILIZATION_SAMPLE"
            or sample.get("sample_index") != index
            or sample.get("visible_gpu_count") != 2
            or len(gpus) != 2
        ):
            raise ValueError("utilization monitor sample envelope drifted")
        observed_uuids: list[str] = []
        observed_physical: list[tuple[int, str]] = []
        low = False
        for gpu_index, raw in enumerate(gpus):
            gpu = dict(_mapping(raw, f"monitor sample {index} GPU {gpu_index}"))
            _exact_keys(
                gpu,
                {"nvidia_smi_index", "gpu_uuid", "utilization_percent"},
                f"monitor sample {index} GPU {gpu_index}",
            )
            utilization = gpu.get("utilization_percent")
            physical_index = gpu.get("nvidia_smi_index")
            if (
                type(physical_index) is not int
                or physical_index < 0
                or gpu.get("gpu_uuid") != runtime_visible_uuids[gpu_index]
                or type(utilization) is not int
                or utilization < 0
                or utilization > 100
            ):
                raise ValueError("utilization monitor GPU sample drifted")
            observed_uuids.append(str(gpu["gpu_uuid"]))
            observed_physical.append((physical_index, str(gpu["gpu_uuid"])))
            low = low or utilization < 90
        if tuple(observed_uuids) != runtime_visible_uuids:
            raise ValueError("utilization monitor sample UUID inventory drifted")
        physical_inventory = tuple(observed_physical)
        if len({item[0] for item in physical_inventory}) != 2:
            raise ValueError("utilization monitor physical GPU indices are not distinct")
        if frozen_physical_inventory is None:
            frozen_physical_inventory = physical_inventory
        elif physical_inventory != frozen_physical_inventory:
            raise ValueError("utilization monitor physical GPU inventory changed")
        timestamp = _timestamp(
            sample.get("sampled_at_utc"), f"monitor sample {index} timestamp"
        )
        if timestamps and timestamp < timestamps[-1]:
            raise ValueError("utilization monitor sample timestamps are not monotone")
        timestamps.append(timestamp)
        low_incidents += int(low)
    if frozen_physical_inventory != runtime_physical_inventory:
        raise ValueError("monitor physical GPU inventory differs from worker runtime")
    return timestamps, low_incidents


def _validate_monitor_and_execution_evidence(
    files: Mapping[str, bytes],
    *,
    run_contract_sha256: str,
    runtime_visible_uuids: tuple[str, str],
    runtime_physical_inventory: tuple[tuple[int, str], tuple[int, str]],
    first_state_claim_time: datetime,
    last_worker_terminal_time: datetime,
) -> None:
    for label in ("preflight", "execution", "utilization_monitor"):
        if not files[LOG_PATHS[label]]:
            raise ValueError(f"{label} log must be nonempty")
    ready = _json(files, LOG_PATHS["utilization_monitor_ready"])
    _exact_keys(
        ready,
        {
            "schema_version",
            "status",
            "run_contract_sha256",
            "monitor_pid",
            "visible_gpu_count",
            "visible_gpu_uuids",
            "started_at_utc",
        },
        "monitor ready record",
    )
    ready_uuids = tuple(_sequence(ready.get("visible_gpu_uuids"), "ready GPU UUIDs"))
    if (
        ready.get("schema_version") != SCHEMA_VERSION
        or ready.get("status") != "EXPANSION_EXACT_LABEL_MONITOR_READY"
        or ready.get("run_contract_sha256") != run_contract_sha256
        or type(ready.get("monitor_pid")) is not int
        or ready["monitor_pid"] <= 0
        or ready.get("visible_gpu_count") != 2
        or ready_uuids != runtime_visible_uuids
    ):
        raise ValueError("monitor ready GPU identity drifted")
    ready_time = _timestamp(ready.get("started_at_utc"), "monitor ready time")
    if ready_time > first_state_claim_time:
        raise ValueError("monitor was not ready before the first state claim")
    stop = _json(files, LOG_PATHS["utilization_monitor_stop_request"])
    _exact_keys(
        stop,
        {"schema_version", "status", "run_contract_sha256", "requested_at_utc"},
        "monitor stop request",
    )
    if (
        stop.get("schema_version") != SCHEMA_VERSION
        or stop.get("status") != "EXPANSION_EXACT_LABEL_MONITOR_STOP_REQUEST"
        or stop.get("run_contract_sha256") != run_contract_sha256
    ):
        raise ValueError("monitor stop request drifted")
    stop_time = _timestamp(stop.get("requested_at_utc"), "monitor stop time")
    summary = _json(files, MONITOR_SUMMARY_FILENAME)
    _exact_keys(
        summary,
        {
            "schema_version",
            "status",
            "run_contract_sha256",
            "container_name_prefix",
            "minimum_gpu_utilization_percent",
            "started_before_first_teacher_forward",
            "started_at_utc",
            "stopped_at_utc",
            "sample_count",
            "low_utilization_incident_count",
            "visible_gpu_count",
            "visible_gpu_uuids",
            "stop_request_sha256",
        },
        "monitor summary",
    )
    summary_uuids = tuple(
        _sequence(summary.get("visible_gpu_uuids"), "monitor summary GPU UUIDs")
    )
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != "GPU_UTILIZATION_MONITOR_SUMMARY"
        or summary.get("run_contract_sha256") != run_contract_sha256
        or summary.get("container_name_prefix") != "sglang-omni-jaxan"
        or summary.get("minimum_gpu_utilization_percent") != 90
        or summary.get("started_before_first_teacher_forward") is not True
        or type(summary.get("sample_count")) is not int
        or summary["sample_count"] <= 0
        or type(summary.get("low_utilization_incident_count")) is not int
        or summary["low_utilization_incident_count"] < 0
        or summary.get("visible_gpu_count") != 2
        or summary_uuids != runtime_visible_uuids
        or summary.get("stop_request_sha256")
        != sha256_bytes(files[LOG_PATHS["utilization_monitor_stop_request"]])
    ):
        raise ValueError("monitor summary or GPU UUID binding drifted")
    summary_start = _timestamp(summary.get("started_at_utc"), "monitor summary start")
    summary_stop = _timestamp(summary.get("stopped_at_utc"), "monitor summary stop")
    sample_times, reconstructed_low_count = _validate_monitor_jsonl(
        files[LOG_PATHS["utilization_monitor"]],
        runtime_visible_uuids=runtime_visible_uuids,
        runtime_physical_inventory=runtime_physical_inventory,
    )
    if (
        summary.get("sample_count") != len(sample_times)
        or summary.get("low_utilization_incident_count")
        != reconstructed_low_count
    ):
        raise ValueError("monitor summary counts differ from archived JSONL")
    coverage_points = [ready_time, *sample_times, summary_stop]
    maximum_gap = max(
        (right - left).total_seconds()
        for left, right in zip(coverage_points, coverage_points[1:])
    )
    if (
        sample_times[0] > first_state_claim_time
        or summary_stop < last_worker_terminal_time
        or stop_time < last_worker_terminal_time
        or (
            last_worker_terminal_time - sample_times[-1]
        ).total_seconds()
        > MONITOR_MAX_SAMPLE_GAP_SECONDS
        or maximum_gap > MONITOR_MAX_SAMPLE_GAP_SECONDS
    ):
        raise ValueError(
            "monitor did not cover the complete worker execution window at the "
            "frozen cadence"
        )
    if not (
        summary_start == ready_time
        and ready_time <= sample_times[0]
        and sample_times[-1] <= summary_stop
        and ready_time <= stop_time <= summary_stop
    ):
        raise ValueError("monitor lifecycle timestamps are inconsistent")

    execution = _json(files, EXECUTION_EVIDENCE_FILENAME)
    _exact_keys(
        execution,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "preflight_log_sha256",
            "execution_log_sha256",
            "utilization_monitor_log_sha256",
            "monitor_ready_sha256",
            "monitor_stop_request_sha256",
            "monitor_summary_sha256",
        },
        "execution evidence",
    )
    expected_hashes = {
        "preflight_log_sha256": sha256_bytes(files[LOG_PATHS["preflight"]]),
        "execution_log_sha256": sha256_bytes(files[LOG_PATHS["execution"]]),
        "utilization_monitor_log_sha256": sha256_bytes(
            files[LOG_PATHS["utilization_monitor"]]
        ),
        "monitor_ready_sha256": sha256_bytes(
            files[LOG_PATHS["utilization_monitor_ready"]]
        ),
        "monitor_stop_request_sha256": sha256_bytes(
            files[LOG_PATHS["utilization_monitor_stop_request"]]
        ),
        "monitor_summary_sha256": sha256_bytes(files[MONITOR_SUMMARY_FILENAME]),
    }
    if (
        execution.get("schema_version") != SCHEMA_VERSION
        or execution.get("protocol_id") != PROTOCOL_ID
        or execution.get("status") != "VALIDATED_EXPANSION_EXACT_LABEL_EXECUTION_EVIDENCE"
        or execution.get("run_contract_sha256") != run_contract_sha256
        or any(execution.get(key) != value for key, value in expected_hashes.items())
    ):
        raise ValueError("execution evidence hash binding drifted")


def aggregate_from_reduction(
    reduction: Mapping[str, Any],
    *,
    run_contract_sha256: str,
    execution_evidence_sha256: str,
    started_at_utc: str,
    ended_at_utc: str,
) -> dict[str, Any]:
    """Build the compact stored aggregate without serializing derived rows."""
    if reduction.get("counts") != EXPECTED_TOTALS:
        raise ValueError("aggregate reduction totals drifted")
    started = _timestamp(started_at_utc, "aggregate start")
    ended = _timestamp(ended_at_utc, "aggregate end")
    if started > ended:
        raise ValueError("aggregate timestamp bracket is reversed")
    _sha256(execution_evidence_sha256, "execution evidence SHA256")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "TERMINAL_EXPANSION_EXACT_LABEL_ATTEMPT",
        "outcome": PASS_OUTCOME,
        "run_contract_sha256": run_contract_sha256,
        "fixed_state_denominator": EXPECTED_STATE_COUNT,
        "planned_counts": EXPECTED_TOTALS,
        "actual_counts": reduction["counts"],
        "attempted_state_count": EXPECTED_STATE_COUNT,
        "completed_state_count": EXPECTED_STATE_COUNT,
        "retry_performed": False,
        "top_up_performed": False,
        "reduction": {
            "counts": reduction["counts"],
            "summary": reduction["summary"],
            "derived_payload_sha256": reduction["derived_payload_sha256"],
        },
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "execution_evidence_sha256": execution_evidence_sha256,
    }


def validate_expansion_label_evidence_files(
    files: Mapping[str, bytes],
    *,
    expected_source_git_commit: str | None = None,
    expected_config_sha256: str | None = None,
    require_canonical_attempt_identity: bool = True,
) -> ExpansionLabelEvidence:
    if not isinstance(files, Mapping):
        raise TypeError("expansion-label evidence files must be a mapping")
    normalized = {_safe_relative_path(name): payload for name, payload in files.items()}
    if any(not isinstance(payload, bytes) for payload in normalized.values()):
        raise TypeError("evidence payloads must be bytes")
    expected_names = expected_file_names()
    if set(normalized) != expected_names:
        raise ValueError(
            "expansion-label evidence file inventory drifted; "
            f"missing={sorted(expected_names - set(normalized))}, "
            f"extra={sorted(set(normalized) - expected_names)}"
        )
    manifest = _json(normalized, RUN_MANIFEST_FILENAME)
    _exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract",
            "run_contract_sha256",
        },
        "run manifest",
    )
    run_contract = _mapping(manifest.get("run_contract"), "run contract")
    run_contract_sha = sha256_bytes(canonical_json_bytes(run_contract))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "IMMUTABLE_EXPANSION_EXACT_LABEL_RUN_MANIFEST"
        or manifest.get("run_contract_sha256") != run_contract_sha
    ):
        raise ValueError("run manifest or run-contract hash drifted")
    validated_contract = validate_execution_run_contract(
        run_contract,
        expected_source_git_commit=expected_source_git_commit,
        expected_config_sha256=expected_config_sha256,
    )
    if require_canonical_attempt_identity:
        # note (luojiaxuan): Canonical paths are already exact in the embedded
        # run contract; this flag exists so synthetic filesystem tests can opt out.
        pass
    _validate_global_ledger(
        _json(normalized, GLOBAL_LEDGER_ARCHIVE_NAME),
        run_contract_sha256=run_contract_sha,
    )
    states = validated_contract["states"]
    by_index: dict[int, dict[str, Any]] = {}
    worker_gpu_uuids: list[str] = []
    bindings_by_worker: dict[str, Mapping[str, Any]] = {}
    first_claim_times: list[datetime] = []
    worker_terminal_times: list[datetime] = []
    runtime_visible: tuple[str, str] | None = None
    for spec in expected_worker_specs():
        (
            records,
            worker_uuid,
            visible,
            bindings_metadata,
            first_claim,
            terminal_time,
        ) = _validate_worker_evidence(
            normalized,
            spec=spec,
            run_contract_sha256=run_contract_sha,
            states=states,
        )
        worker_gpu_uuids.append(worker_uuid)
        bindings_by_worker[spec.worker_id] = bindings_metadata
        first_claim_times.append(first_claim)
        worker_terminal_times.append(terminal_time)
        if runtime_visible is None:
            runtime_visible = visible
        elif visible != runtime_visible:
            raise ValueError("worker visible GPU UUID sets drifted")
        for index, record in zip(spec.state_indices, records, strict=True):
            if index in by_index:
                raise ValueError("state was emitted by multiple workers")
            by_index[index] = record
    if (
        set(by_index) != set(range(EXPECTED_STATE_COUNT))
        or runtime_visible is None
        or tuple(worker_gpu_uuids) != runtime_visible
    ):
        raise ValueError("worker state schedule or GPU UUID assignment drifted")
    validate_eager_worker_runtime_pair(bindings_by_worker)
    _validate_monitor_and_execution_evidence(
        normalized,
        run_contract_sha256=run_contract_sha,
        runtime_visible_uuids=runtime_visible,
        runtime_physical_inventory=tuple(
            (
                int(bindings_by_worker[spec.worker_id]["nvidia_smi_index"]),
                str(bindings_by_worker[spec.worker_id]["gpu_uuid"]),
            )
            for spec in expected_worker_specs()
        ),
        first_state_claim_time=min(first_claim_times),
        last_worker_terminal_time=max(worker_terminal_times),
    )
    ordered_records = [by_index[index] for index in range(EXPECTED_STATE_COUNT)]
    reduction = reduce_raw_distance_states(
        ordered_records,
        expected_states=states,
        run_contract_sha256=run_contract_sha,
    )
    aggregate = _json(normalized, AGGREGATE_FILENAME)
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
            "retry_performed",
            "top_up_performed",
            "reduction",
            "started_at_utc",
            "ended_at_utc",
            "execution_evidence_sha256",
        },
        "aggregate",
    )
    rebuilt = aggregate_from_reduction(
        reduction,
        run_contract_sha256=run_contract_sha,
        execution_evidence_sha256=sha256_bytes(normalized[EXECUTION_EVIDENCE_FILENAME]),
        started_at_utc=str(aggregate.get("started_at_utc")),
        ended_at_utc=str(aggregate.get("ended_at_utc")),
    )
    if aggregate != rebuilt:
        raise ValueError("stored aggregate differs from independent raw D(S) reduction")
    inventory = _inventory(normalized)
    return ExpansionLabelEvidence(
        files=dict(normalized),
        run_contract=dict(run_contract),
        source_git_commit=validated_contract["runner_source_git_commit"],
        execution_git_commit=validated_contract["execution_git_commit"],
        run_contract_sha256=run_contract_sha,
        config_sha256=validated_contract["config_sha256"],
        outcome=PASS_OUTCOME,
        aggregate=aggregate,
        reduction=reduction,
        runtime_gpu_uuids=runtime_visible,
        inventory=inventory,
        tree_inventory_sha256=sha256_bytes(canonical_json_bytes(inventory)),
    )


def collect_raw_expansion_label_evidence(
    raw_output_dir: str | Path,
    global_attempt_ledger: str | Path,
    *,
    expected_source_git_commit: str | None = None,
    expected_config_sha256: str | None = None,
    require_canonical_location: bool = True,
) -> ExpansionLabelEvidence:
    root = Path(raw_output_dir)
    ledger_path = Path(global_attempt_ledger)
    if require_canonical_location and (
        root.resolve() != CANONICAL_OUTPUT_DIR
        or ledger_path.resolve() != CANONICAL_LEDGER_PATH
    ):
        raise ValueError("raw output or global ledger path is not canonical")
    if (
        root.is_symlink()
        or ledger_path.is_symlink()
        or not root.is_dir()
        or not ledger_path.is_file()
    ):
        raise ValueError("raw output or global ledger is missing")
    files: dict[str, bytes] = {
        GLOBAL_LEDGER_ARCHIVE_NAME: ledger_path.read_bytes()
    }
    ledger = strict_json_object_bytes(
        files[GLOBAL_LEDGER_ARCHIVE_NAME], label="global attempt ledger"
    )
    siblings = _mapping(ledger.get("worker_sibling_ledgers"), "sibling ledgers")
    for spec in expected_worker_specs():
        sibling = Path(str(siblings.get(spec.worker_id)))
        if sibling.is_symlink() or not sibling.is_file():
            raise ValueError("worker sibling ledger is missing")
        files[f"{WORKER_SIBLING_LEDGER_DIRECTORY}/{spec.worker_id}.json"] = (
            sibling.read_bytes()
        )
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("raw evidence contains a symlink")
        if path.is_file():
            relative = _safe_relative_path(path.relative_to(root).as_posix())
            if relative in files:
                raise ValueError("raw evidence path collides with external ledger")
            files[relative] = path.read_bytes()
        elif not path.is_dir():
            raise ValueError("raw evidence contains a non-regular entry")
    return validate_expansion_label_evidence_files(
        files,
        expected_source_git_commit=expected_source_git_commit,
        expected_config_sha256=expected_config_sha256,
        require_canonical_attempt_identity=require_canonical_location,
    )


def deterministic_ustar_bytes(files: Mapping[str, bytes]) -> bytes:
    destination = io.BytesIO()
    with tarfile.open(
        fileobj=destination, mode="w", format=tarfile.USTAR_FORMAT
    ) as archive:
        for relative in sorted(files):
            _safe_relative_path(relative)
            payload = files[relative]
            if not isinstance(payload, bytes):
                raise TypeError("archive payloads must be bytes")
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


def read_expansion_label_archive(
    archive_path: str | Path,
    *,
    expected_source_git_commit: str | None = None,
    expected_config_sha256: str | None = None,
    require_canonical_attempt_identity: bool = True,
) -> ExpansionLabelEvidence:
    path = Path(archive_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("expansion-label archive is missing or symlinked")
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
                    raise ValueError("archive contains duplicate members")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("archive member is unreadable")
                files[relative] = extracted.read()
    except tarfile.TarError as error:
        raise ValueError("archive is not readable USTAR") from error
    if path.read_bytes() != deterministic_ustar_bytes(files):
        raise ValueError("archive is not byte-canonical deterministic USTAR")
    return validate_expansion_label_evidence_files(
        files,
        expected_source_git_commit=expected_source_git_commit,
        expected_config_sha256=expected_config_sha256,
        require_canonical_attempt_identity=require_canonical_attempt_identity,
    )


def package_raw_expansion_label_evidence(
    *,
    raw_output_dir: str | Path,
    global_attempt_ledger: str | Path,
    output_archive: str | Path,
    source_git_commit: str,
    config_sha256: str,
    parent_substrate_archive: str | Path,
    derived_artifact_root: str | Path,
    ocr_backend_config: str | Path,
    require_canonical_location: bool = True,
) -> dict[str, Any]:
    output = Path(output_archive)
    if require_canonical_location and output.resolve() != CANONICAL_ARCHIVE_PATH:
        raise ValueError("raw archive path is not canonical")
    if output.exists():
        raise FileExistsError("raw archive overwrite is forbidden")
    evidence = collect_raw_expansion_label_evidence(
        raw_output_dir,
        global_attempt_ledger,
        expected_source_git_commit=source_git_commit,
        expected_config_sha256=config_sha256,
        require_canonical_location=require_canonical_location,
    )
    replayed_states, replay_metadata = replay_external_input_projections(
        parent_substrate_archive=parent_substrate_archive,
        derived_artifact_root=derived_artifact_root,
        ocr_backend_config=ocr_backend_config,
    )
    external_replay = validate_external_input_replay(
        evidence.run_contract,
        replayed_states,
        replay_metadata,
    )
    payload = deterministic_ustar_bytes(evidence.files)
    atomic_exclusive_publish_bytes(output, payload)
    reread = read_expansion_label_archive(
        output,
        expected_source_git_commit=source_git_commit,
        expected_config_sha256=config_sha256,
        require_canonical_attempt_identity=require_canonical_location,
    )
    if reread.inventory != evidence.inventory or reread.reduction != evidence.reduction:
        raise ValueError("written archive differs from raw evidence")
    return {
        "status": "PACKAGED_V2_2_EXPANSION_EXACT_LABEL_RAW_EVIDENCE",
        "source_git_commit": source_git_commit,
        "execution_git_commit": evidence.execution_git_commit,
        "config_sha256": config_sha256,
        "outcome": evidence.outcome,
        "archive_path": str(output.resolve()),
        "archive_sha256": sha256_file(output),
        "archive_size_bytes": output.stat().st_size,
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
        "file_count": len(evidence.inventory),
        "counts": dict(evidence.reduction["counts"]),
        "derived_payload_sha256": evidence.reduction["derived_payload_sha256"],
        "external_input_replay": external_replay,
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
) -> tuple[ExpansionLabelEvidence, dict[str, Any]]:
    if (
        hf_repo != CANONICAL_HF_REPO
        or hf_tag != CANONICAL_HF_TAG
        or hf_path != CANONICAL_HF_PATH
    ):
        raise ValueError("HF repo, tag, or path differs from frozen destination")
    _git_sha(hf_immutable_revision, "HF immutable revision")
    source = Path(source_archive)
    fresh = Path(fresh_immutable_archive)
    if source.resolve() == fresh.resolve():
        raise ValueError("archive byte replay requires a distinct path")
    source_evidence = read_expansion_label_archive(
        source,
        expected_source_git_commit=source_git_commit,
        expected_config_sha256=config_sha256,
    )
    fresh_evidence = read_expansion_label_archive(
        fresh,
        expected_source_git_commit=source_git_commit,
        expected_config_sha256=config_sha256,
    )
    if source.read_bytes() != fresh.read_bytes():
        raise ValueError("provided replay archive is not byte-identical")
    if source_evidence.inventory != fresh_evidence.inventory:
        raise ValueError("provided replay archive inventory differs")
    return source_evidence, {
        "repo": hf_repo,
        "repo_type": "dataset",
        "visibility": "UNVERIFIED_CALLER_SUPPLIED_CLAIM",
        "tag": hf_tag,
        "path": hf_path,
        "immutable_revision": hf_immutable_revision,
        "provided_replay_archive_sha256": sha256_file(fresh),
        "provided_replay_archive_size_bytes": fresh.stat().st_size,
        "provided_replay_byte_identical": True,
        "provided_replay_tree_inventory_sha256": fresh_evidence.tree_inventory_sha256,
        "provided_replay_file_count": len(fresh_evidence.inventory),
        "validation_scope": "CALLER_SUPPLIED_ARCHIVE_BYTE_REPLAY_ONLY",
        "hub_download_proven": False,
        "tag_resolution_verified": False,
    }


def validate_hub_download_attestation(
    value: Any,
    *,
    downloaded_archive: str | Path,
) -> dict[str, Any]:
    attestation = dict(_mapping(value, "HF Hub download attestation"))
    _exact_keys(
        attestation,
        {
            "repo",
            "repo_type",
            "repo_id",
            "repo_id_verified",
            "repo_private",
            "repo_private_verified",
            "tag",
            "path",
            "tag_query_revision",
            "download_requested_revision",
            "resolved_revision",
            "tag_resolved_revision",
            "pre_download_tag_resolved_revision",
            "pre_download_immutable_resolved_revision",
            "post_download_tag_resolved_revision",
            "post_download_immutable_resolved_revision",
            "tag_resolution_verified",
            "immutable_revision_resolution_verified",
            "tag_resolution_stable",
            "immutable_revision_resolution_stable",
            "force_download",
            "returned_path_exact",
            "no_symlink_components_verified",
            "downloaded_path",
            "downloaded_archive_sha256",
            "downloaded_archive_size_bytes",
        },
        "HF Hub download attestation",
    )
    downloaded = Path(downloaded_archive).resolve()
    resolved = _git_sha(attestation.get("resolved_revision"), "resolved HF revision")
    if (
        attestation.get("repo") != CANONICAL_HF_REPO
        or attestation.get("repo_type") != "dataset"
        or attestation.get("repo_id") != CANONICAL_HF_REPO
        or attestation.get("repo_id_verified") is not True
        or attestation.get("repo_private") is not True
        or attestation.get("repo_private_verified") is not True
        or attestation.get("tag") != CANONICAL_HF_TAG
        or attestation.get("path") != CANONICAL_HF_PATH
        or attestation.get("tag_query_revision") != CANONICAL_HF_TAG
        or attestation.get("download_requested_revision") != resolved
        or attestation.get("tag_resolved_revision") != resolved
        or attestation.get("pre_download_tag_resolved_revision") != resolved
        or attestation.get("pre_download_immutable_resolved_revision") != resolved
        or attestation.get("post_download_tag_resolved_revision") != resolved
        or attestation.get("post_download_immutable_resolved_revision") != resolved
        or attestation.get("tag_resolution_verified") is not True
        or attestation.get("immutable_revision_resolution_verified") is not True
        or attestation.get("tag_resolution_stable") is not True
        or attestation.get("immutable_revision_resolution_stable") is not True
        or attestation.get("force_download") is not True
        or attestation.get("returned_path_exact") is not True
        or attestation.get("no_symlink_components_verified") is not True
        or attestation.get("downloaded_path") != str(downloaded)
        or attestation.get("downloaded_archive_sha256") != sha256_file(downloaded)
        or attestation.get("downloaded_archive_size_bytes") != downloaded.stat().st_size
    ):
        raise ValueError("HF Hub download or tag-resolution attestation drifted")
    return attestation


def build_expansion_label_artifact_manifest(
    *,
    source_archive: str | Path,
    fresh_immutable_archive: str | Path,
    source_git_commit: str,
    config_sha256: str,
    hf_repo: str,
    hf_tag: str,
    hf_path: str,
    hf_immutable_revision: str,
    hub_download_attestation: Mapping[str, Any],
    parent_substrate_archive: str | Path,
    derived_artifact_root: str | Path,
    ocr_backend_config: str | Path,
) -> dict[str, Any]:
    evidence, replay = validate_fresh_hf_evidence(
        source_archive=source_archive,
        fresh_immutable_archive=fresh_immutable_archive,
        source_git_commit=source_git_commit,
        config_sha256=config_sha256,
        hf_repo=hf_repo,
        hf_tag=hf_tag,
        hf_path=hf_path,
        hf_immutable_revision=hf_immutable_revision,
    )
    attestation = validate_hub_download_attestation(
        hub_download_attestation,
        downloaded_archive=fresh_immutable_archive,
    )
    if attestation["resolved_revision"] != hf_immutable_revision:
        raise ValueError("downloaded HF revision differs from requested immutable revision")
    replayed_states, replay_metadata = replay_external_input_projections(
        parent_substrate_archive=parent_substrate_archive,
        derived_artifact_root=derived_artifact_root,
        ocr_backend_config=ocr_backend_config,
    )
    external_replay = validate_external_input_replay(
        evidence.run_contract,
        replayed_states,
        replay_metadata,
    )
    hf = {
        "repo": CANONICAL_HF_REPO,
        "repo_type": "dataset",
        "visibility": "private",
        "repo_id_verified": True,
        "repo_private_verified": True,
        "tag": CANONICAL_HF_TAG,
        "path": CANONICAL_HF_PATH,
        "immutable_revision": hf_immutable_revision,
        "tag_resolved_revision": attestation["tag_resolved_revision"],
        "tag_resolution_verified": True,
        "immutable_revision_resolution_verified": True,
        "pre_and_post_download_resolution_stable": True,
        "force_download": True,
        "fresh_download_archive_sha256": attestation[
            "downloaded_archive_sha256"
        ],
        "fresh_download_archive_size_bytes": attestation[
            "downloaded_archive_size_bytes"
        ],
        "fresh_download_byte_identical": replay["provided_replay_byte_identical"],
        "fresh_download_tree_inventory_sha256": replay[
            "provided_replay_tree_inventory_sha256"
        ],
        "fresh_download_file_count": replay["provided_replay_file_count"],
        "fresh_download_validated": True,
    }
    source = Path(source_archive)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": ARTIFACT_PROTOCOL_ID,
        "status": "VERIFIED_IMMUTABLE_V2_2_EXPANSION_EXACT_LABEL_ARTIFACT",
        "source": {
            "runner_source_git_commit": evidence.source_git_commit,
            "execution_git_commit": evidence.execution_git_commit,
            "config_path": CANONICAL_CONFIG_PATH,
            "config_sha256": evidence.config_sha256,
            "runner_freeze_path": CANONICAL_RUNNER_FREEZE_PATH,
            "run_contract_sha256": evidence.run_contract_sha256,
            "parent_substrate": CANONICAL_PARENT_SUBSTRATE,
            "derived_artifact": CANONICAL_DERIVED_ARTIFACT,
        },
        "raw_archive": {
            "format": ARCHIVE_FORMAT,
            "member_prefix": ARCHIVE_MEMBER_PREFIX,
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
            "file_count": len(evidence.inventory),
            "tree_inventory_sha256": evidence.tree_inventory_sha256,
            "aggregate_sha256": sha256_bytes(evidence.files[AGGREGATE_FILENAME]),
        },
        "hf_artifact": hf,
        "result": {
            "outcome": evidence.outcome,
            "counts": dict(evidence.reduction["counts"]),
            "summary": dict(evidence.reduction["summary"]),
            "derived_payload_sha256": evidence.reduction[
                "derived_payload_sha256"
            ],
        },
        "external_input_replay": external_replay,
        "validation_scope": {
            "source_locked_producer_runtime_accounting": True,
            "runtime_or_model_independently_revalidated": False,
            "raw_distance_to_derived_cpu_reduction_independently_recomputed": True,
            "external_dynamic_input_replay_independently_recomputed": True,
        },
        "negative_declarations": {
            "raw_distance_is_only_canonical_scientific_truth": True,
            "producer_supplied_derived_rows_trusted": False,
            "negative_marginals_clamped": False,
            "nonmonotone_states_removed": False,
            "full_logit_tensor_host_transfer_count": 0,
            "retry_performed": False,
            "top_up_performed": False,
            "confirm_accessed": False,
            "gate_training_started": False,
            "matched_nll_started": False,
            "closed_loop_started": False,
        },
    }


__all__ = [
    "AGGREGATE_FILENAME",
    "ALGEBRA_RESIDUAL_TOLERANCE",
    "ARCHIVE_FORMAT",
    "ARCHIVE_MEMBER_PREFIX",
    "ARTIFACT_PROTOCOL_ID",
    "ATTEMPT_DIRECTORY",
    "ATTEMPT_ID",
    "CANONICAL_ARCHIVE_PATH",
    "CANONICAL_CONFIG_PATH",
    "CANONICAL_DERIVED_ARTIFACT",
    "CANONICAL_HF_PATH",
    "CANONICAL_HF_REPO",
    "CANONICAL_HF_TAG",
    "CANONICAL_LEDGER_PATH",
    "CANONICAL_OUTPUT_DIR",
    "CANONICAL_PARENT_SUBSTRATE",
    "CANONICAL_RUNTIME",
    "CANONICAL_RUNNER_FREEZE_PATH",
    "EXECUTION_EVIDENCE_FILENAME",
    "EXPECTED_ATTRIBUTION_ROWS",
    "EXPECTED_DEPLOYMENT_EDGES",
    "EXPECTED_DISTANCE_ROWS",
    "EXPECTED_FULL_EDGES",
    "EXPECTED_KL_MEASUREMENTS",
    "EXPECTED_PAIR_INTERACTIONS",
    "EXPECTED_PRIMARY_ORACLES",
    "EXPECTED_STATE_COUNT",
    "EXPECTED_TEACHER_FORWARDS",
    "EXPECTED_TOTALS",
    "EXPECTED_TRAJECTORY_COUNT",
    "ExpansionLabelEvidence",
    "FROZEN_SUBSTRATE_STATE_PROJECTION_SHA256",
    "GLOBAL_LEDGER_ARCHIVE_NAME",
    "LOG_PATHS",
    "MONITOR_SUMMARY_FILENAME",
    "MONITOR_MAX_SAMPLE_GAP_SECONDS",
    "MONITOR_SAMPLE_CADENCE_SECONDS",
    "MONITOR_SAMPLE_TOLERANCE_SECONDS",
    "PASS_OUTCOME",
    "PROTOCOL_ID",
    "PROHIBITED_OPERATION_COUNTS",
    "RUN_MANIFEST_FILENAME",
    "RUNNER_FREEZE_PROTOCOL_ID",
    "RUNNER_FREEZE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "STATE_DIRECTORY",
    "STATE_STATUS",
    "WORKER_DIRECTORY",
    "WORKER_LEDGER_FILENAME",
    "WORKER_RUNTIME_FILENAME",
    "WORKER_SIBLING_LEDGER_DIRECTORY",
    "WORKER_STATUS",
    "WORKER_TERMINAL_FILENAME",
    "WorkerSpec",
    "aggregate_from_reduction",
    "atomic_exclusive_publish_bytes",
    "build_expansion_label_artifact_manifest",
    "build_runner_freeze",
    "canonical_json_bytes",
    "collect_raw_expansion_label_evidence",
    "deterministic_ustar_bytes",
    "expected_file_names",
    "expected_worker_specs",
    "load_and_validate_runner_freeze",
    "materialize_runner_freeze",
    "package_raw_expansion_label_evidence",
    "pretty_json_bytes",
    "read_expansion_label_archive",
    "reduce_raw_distance_states",
    "replay_external_input_projections",
    "runner_interface_contract",
    "sha256_bytes",
    "sha256_file",
    "strict_json_object_bytes",
    "validate_execution_run_contract",
    "validate_expansion_label_evidence_files",
    "validate_external_input_replay",
    "validate_fresh_hf_evidence",
    "validate_hub_download_attestation",
    "validate_raw_state_record",
    "validate_runner_freeze_data",
    "validate_state_projections",
]
