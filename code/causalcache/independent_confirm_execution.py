"""Formal execution orchestration for the frozen independent confirm-20 run."""

from __future__ import annotations

import hashlib
import io
import json
import math
import multiprocessing
import os
import queue as queue_module
import re
import stat
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol

from causalcache.gate_v1_data import FeatureState
from causalcache.gate_v1_training import FittedEnsemble
from causalcache.independent_confirm_artifact import (
    DESTINATION_PRIVATE,
    DESTINATION_REPO,
    DESTINATION_REPO_TYPE,
    DESTINATION_TAG,
    INDEPENDENT_DECISION_PATH,
    PAYLOAD_TARGETS,
    REPORT_TARGETS,
    FinalPublicationReceipt,
    LabelBlindPayloadSeal,
    PayloadPublicationReceipt,
    adopt_existing_payload_commit as adopt_artifact_existing_payload_commit,
    build_label_blind_payload,
    build_report_files,
    canonical_json_bytes,
    publish_payload_commit as publish_artifact_payload_commit,
    publish_report_commit as publish_artifact_report_commit,
    read_independent_decisions,
    reconcile_report_publication_state,
)
from causalcache.independent_confirm_contract import (
    PROTOCOL_ID as CONTRACT_PROTOCOL_ID,
    IndependentConfirmContract,
    pretty_json_bytes,
)
from causalcache.independent_confirm_data import (
    CONFIRM_CANDIDATE_EVENT_STEP_IDS,
    CONFIRM_SOURCE_IDS,
    CONFIRM_STATE_COUNT,
    Confirm20LabelBlindBundle,
    Confirm20PolicyWorkItem,
    ValidatedConfirm20Payloads,
    build_confirm20_label_blind_bundle,
)
from causalcache.independent_confirm_runner import (
    ConfirmWorkerAssignment,
    ConfirmWorkerResult,
    IndependentSelectionBundle,
    LoadedIndependentEnsemble,
    PayloadCommitReceipt,
    aggregate_independent_confirm_workers,
    bind_confirm_message_builder,
    confirm_worker_assignments,
    independent_selection_payload_bytes,
    load_independent_ensemble,
    persist_and_seal_label_blind_payload,
    publish_payload_commit as issue_runner_payload_receipt,
    run_confirm_state_once,
    score_and_select_independent,
)
from causalcache.policy.gui_owl_v2_1 import GUI_OWL_V2_1_PROTOCOL_ID
from causalcache.policy.gui_owl_v2_2_eager_runtime import (
    GUI_OWL_V2_2_EAGER_ATTENTION_IMPLEMENTATION,
    GUI_OWL_V2_2_EAGER_RUNTIME_ID,
)
from causalcache.policy.gui_owl_v2_2_vision_runtime import (
    GUI_OWL_V2_2_VISION_IMAGE_PROCESSOR_CLASS,
    GUI_OWL_V2_2_VISION_RUNTIME_UUID_SIZE_DICT_V3_ID,
)
from causalcache.policy.gui_owl_v2_runtime import FROZEN_GUI_OWL_V2_DTYPE
from causalcache.policy.gui_owl_v2_vision import (
    MODEL_CLASS_NAME,
    MODEL_FILE_COUNT,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_TOTAL_BYTES,
    SNAPSHOT_MANIFEST_SHA256,
    TRANSFORMERS_SOURCE_SHA256,
    TRANSFORMERS_VERSION,
)
from causalcache.restoration_v2_1_contract import (
    CANONICAL_TOOL_SCHEMA_SHA256,
    CHAT_TEMPLATE_FILE_SHA256,
    CHAT_TEMPLATE_TEXT_SHA256,
)
from causalcache.restoration_v2_baselines import select_top_two


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_independent_confirm20_execution_v1"
RUN_CONTRACT_PROTOCOL_ID = (
    "causalcache_independent_confirm20_execution_contract_v1"
)
COMPLETION_FILENAME = "completion.json"
FAILURE_FILENAME = "failure.json"
ATTEMPT_FILENAME = "attempt.json"
POLICY_VISION_FEATURE_REPEATS = 2
WORKER_COUNT = 4
STATES_PER_WORKER = 5
GPU_TOPOLOGY_SMOKE_SCHEMA_VERSION = "1"
GPU_TOPOLOGY_SMOKE_PROTOCOL_ID = (
    "causalcache_independent_confirm_gpu_topology_smoke_v1"
)
GPU_TOPOLOGY_SMOKE_PHASE_ORDER = (
    "policy_vision_spawn",
    "teacher_forced_fork",
)
GPU_TOPOLOGY_SMOKE_IMAGE_COUNT = 5
GPU_TOPOLOGY_SMOKE_FEATURE_REPEATS = 1
GPU_TOPOLOGY_SMOKE_FIXED_RGB_IMAGE_SET_SHA256 = (
    "ebdc0e7b23a740052d875d775c6f7dee911339cf0e5b5ff5b27760718fe0ae06"
)

_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CUDA_DEVICE = re.compile(r"cuda:[0-9]+")
_GPU_UUID = re.compile(
    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
)

_GPU_TOPOLOGY_SMOKE_SCOPE = {
    "confirm_data_accessed": False,
    "confirm_output_written": False,
    "huggingface_api_accessed": False,
    "network_accessed": False,
    "policy_action_generated": False,
    "artifact_file_written": False,
    "model_load_mode": "explicit_local_files_only",
}
_GPU_TOPOLOGY_VISION_COUNTS_PER_WORKER = {
    "image_processor_batch_count": 1,
    "policy_vision_feature_forward_count": 1,
    "top_model_forward_count": 0,
    "language_model_forward_count": 0,
    "lm_head_forward_count": 0,
    "generation_count": 0,
}
_GPU_TOPOLOGY_TEACHER_COUNTS_PER_WORKER = {
    "teacher_forced_distance_logits_call_count": 1,
    "teacher_forced_example_count": 1,
    "teacher_image_count": GPU_TOPOLOGY_SMOKE_IMAGE_COUNT,
    "finite_logits_validation_count": 1,
    "policy_generation_call_count": 0,
    "full_logit_tensor_host_transfer_count": 0,
}


@dataclass(frozen=True)
class PolicyVisionPhaseResult:
    records: tuple[Mapping[str, Any], ...]
    selections: Mapping[str, tuple[int, ...]]
    worker_metadata: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class RestorationPhaseResult:
    worker_results: tuple[ConfirmWorkerResult, ...]
    worker_metadata: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ConfirmExecutionResult:
    completion: Mapping[str, Any]
    report: Mapping[str, Any]
    payload_files: Mapping[str, bytes]
    report_files: Mapping[str, bytes]


class ConfirmPublisher(Protocol):
    def preflight_destination(self) -> Mapping[str, Any]: ...

    def bind_payload_seal(self, seal: LabelBlindPayloadSeal) -> None: ...

    def publish_payload(
        self,
        files: Mapping[str, bytes],
        inventory: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...

    def verify_payload(
        self,
        commit: str,
        inventory: Sequence[Mapping[str, Any]],
    ) -> bool: ...

    def publish_report(
        self,
        report_files: Mapping[str, bytes],
        *,
        payload_commit: str,
    ) -> Mapping[str, Any]: ...

    def report_publication_audit(self) -> Mapping[str, Any]: ...


PolicyVisionPhase = Callable[[Confirm20LabelBlindBundle], PolicyVisionPhaseResult]
RestorationPhase = Callable[
    [
        tuple[ConfirmWorkerAssignment, ...],
        PayloadCommitReceipt,
        str,
    ],
    RestorationPhaseResult,
]
IndependentScorer = Callable[
    [Sequence[FeatureState], FittedEnsemble], IndependentSelectionBundle
]


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _inventory(files: Mapping[str, bytes]) -> tuple[dict[str, Any], ...]:
    result = []
    for raw_path in sorted(files):
        path = _safe_relative(raw_path, label="formal artifact path")
        payload = files[raw_path]
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("formal artifact files must be non-empty bytes")
        result.append(
            {"path": path, "sha256": _sha256(payload), "size_bytes": len(payload)}
        )
    if not result:
        raise ValueError("formal artifact inventory cannot be empty")
    return tuple(result)


def _selection_score_maps(
    bundle: IndependentSelectionBundle,
) -> tuple[
    Mapping[str, Mapping[int, float]],
    tuple[Mapping[str, Mapping[int, float]], ...],
]:
    payload = independent_selection_payload_bytes(bundle)
    replay = read_independent_decisions(payload)
    ensemble = replay.get("ensemble_scores")
    seeds = replay.get("seed_scores")
    if (
        not isinstance(ensemble, Mapping)
        or set(ensemble)
        != {f"{source_id}:decision_step:006" for source_id in CONFIRM_SOURCE_IDS}
        or not isinstance(seeds, tuple)
        or len(seeds) != 5
        or any(not isinstance(item, Mapping) or set(item) != set(ensemble) for item in seeds)
    ):
        raise ValueError("canonical independent decision replay coverage drifted")
    return ensemble, seeds


def validate_policy_vision_phase(
    result: PolicyVisionPhaseResult,
    *,
    bundle: Confirm20LabelBlindBundle,
    expected_devices: Sequence[str] | None = None,
    expected_gpu_uuids: Sequence[str] | None = None,
) -> PolicyVisionPhaseResult:
    if not isinstance(result, PolicyVisionPhaseResult):
        raise TypeError("policy-vision phase must return PolicyVisionPhaseResult")
    records = tuple(result.records)
    if len(records) != CONFIRM_STATE_COUNT:
        raise ValueError("policy-vision phase must cover all 20 confirm states")
    assignment_by_ordinal = {
        ordinal: assignment.worker_id
        for assignment in confirm_worker_assignments(bundle.work_items)
        for ordinal in assignment.ordinals
    }
    normalized: list[Mapping[str, Any]] = []
    selections: dict[str, tuple[int, ...]] = {}
    worker_counts: Counter[str] = Counter()
    worker_devices: dict[str, tuple[str, str]] = {}
    for ordinal, (feature, raw) in enumerate(
        zip(bundle.feature_states, records, strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("policy-vision records must be mappings")
        state = raw.get("state")
        score_rows = raw.get("scores_by_event_step")
        worker_id = raw.get("worker_id")
        if (
            not isinstance(state, Mapping)
            or state.get("ordinal") != ordinal
            or state.get("source_id") != feature.source_id
            or state.get("state_id") != feature.state_id
            or worker_id != assignment_by_ordinal[ordinal]
            or raw.get("ordinal") != ordinal
            or raw.get("source_id") != feature.source_id
            or raw.get("state_id") != feature.state_id
            or raw.get("decision_step_id") != 6
            or raw.get("candidate_event_step_ids") != [1, 2, 3, 4]
            or not isinstance(score_rows, list)
            or len(score_rows) != 4
            or _CUDA_DEVICE.fullmatch(str(raw.get("device"))) is None
            or _GPU_UUID.fullmatch(str(raw.get("gpu_uuid"))) is None
        ):
            raise ValueError("policy-vision state, roster, or worker binding drifted")
        scores: dict[int, float] = {}
        for expected_step, row in zip(
            CONFIRM_CANDIDATE_EVENT_STEP_IDS, score_rows, strict=True
        ):
            if not isinstance(row, Mapping) or row.get("event_step_id") != expected_step:
                raise ValueError("policy-vision candidate score order drifted")
            score = float(row.get("score", math.nan))
            if not math.isfinite(score):
                raise ValueError("policy-vision score must be finite")
            scores[expected_step] = score
        frozen = select_top_two(scores)
        selected = tuple(frozen.selected_event_step_ids)
        if (
            raw.get("ranked_event_step_ids")
            != list(frozen.ranked_event_step_ids)
            or raw.get("selected_event_step_ids") != list(selected)
            or raw.get("feature_repeats") != POLICY_VISION_FEATURE_REPEATS
        ):
            raise ValueError("policy-vision ranking, selection, or replay count drifted")
        replay = raw.get("same_device_replay")
        if (
            not isinstance(replay, Mapping)
            or replay.get("performed") is not True
            or replay.get("ranking_equal") is not True
            or replay.get("selection_equal") is not True
            or float(replay.get("max_abs_score_difference", math.inf)) > 1e-6
        ):
            raise ValueError("policy-vision same-device replay failed")
        worker_counts[str(worker_id)] += 1
        observed_device = (str(raw["device"]), str(raw["gpu_uuid"]))
        previous_device = worker_devices.setdefault(str(worker_id), observed_device)
        if previous_device != observed_device:
            raise ValueError("policy-vision worker device identity changed within a shard")
        selections[feature.state_id] = selected
        normalized.append(MappingProxyType(json.loads(canonical_json_bytes(raw))))
    expected_workers = {f"worker-{index}" for index in range(WORKER_COUNT)}
    if worker_counts != Counter(
        {worker: STATES_PER_WORKER for worker in expected_workers}
    ):
        raise ValueError("policy-vision worker shards are not four-by-five")
    if dict(result.selections) != selections:
        raise ValueError("policy-vision selection map differs from score records")
    metadata = tuple(result.worker_metadata)
    if len(metadata) != WORKER_COUNT:
        raise ValueError("policy-vision worker metadata coverage drifted")
    expected_worker_order = tuple(f"worker-{index}" for index in range(WORKER_COUNT))
    observed_device_pairs = []
    for expected_worker, item in zip(expected_worker_order, metadata, strict=True):
        if (
            not isinstance(item, Mapping)
            or item.get("worker_id") != expected_worker
            or item.get("state_count") != STATES_PER_WORKER
            or _CUDA_DEVICE.fullmatch(str(item.get("device"))) is None
            or _GPU_UUID.fullmatch(str(item.get("gpu_uuid"))) is None
            or worker_devices.get(expected_worker)
            != (str(item["device"]), str(item["gpu_uuid"]))
        ):
            raise ValueError("policy-vision worker metadata/device binding drifted")
        observed_device_pairs.append((str(item["device"]), str(item["gpu_uuid"])))
    if (
        len({device for device, _ in observed_device_pairs}) != WORKER_COUNT
        or len({gpu_uuid for _, gpu_uuid in observed_device_pairs}) != WORKER_COUNT
    ):
        raise ValueError("policy-vision worker device identities must be unique")
    if expected_devices is not None or expected_gpu_uuids is not None:
        devices, gpu_uuids = _validate_device_allocation(
            () if expected_devices is None else expected_devices,
            () if expected_gpu_uuids is None else expected_gpu_uuids,
            label="policy-vision run contract",
        )
        if observed_device_pairs != list(zip(devices, gpu_uuids, strict=True)):
            raise ValueError("policy-vision metadata differs from run-contract devices")
    return PolicyVisionPhaseResult(
        records=tuple(normalized),
        selections=MappingProxyType(selections),
        worker_metadata=metadata,
    )


def build_canonical_label_blind_payload(
    *,
    bundle: Confirm20LabelBlindBundle,
    independent: IndependentSelectionBundle,
    policy_vision: PolicyVisionPhaseResult,
) -> LabelBlindPayloadSeal:
    ensemble_scores, seed_scores = _selection_score_maps(independent)
    seal = build_label_blind_payload(
        feature_states=bundle.feature_states,
        selections={
            "dynamic_recent": bundle.dynamic_recent,
            "ocr_rgb_v2": bundle.ocr_rgb_v2,
            "policy_vision_v3": policy_vision.selections,
        },
        score_records={
            "ocr_rgb_v2": bundle.ocr_rgb_score_records,
            "policy_vision_v3": policy_vision.records,
        },
        ensemble_scores=ensemble_scores,
        seed_scores=seed_scores,
    )
    runner_bytes = independent_selection_payload_bytes(independent)
    artifact_bytes = seal.files[INDEPENDENT_DECISION_PATH]
    if runner_bytes != artifact_bytes:
        raise RuntimeError(
            "runner and canonical artifact independent decisions are not byte-identical"
        )
    if set(seal.files) != set(PAYLOAD_TARGETS):
        raise RuntimeError("canonical label-blind payload is not the frozen eight files")
    return seal


class DurableConfirmOutput:
    """Exclusive local evidence store; an existing directory forbids a retry."""

    def __init__(self, root: str | Path) -> None:
        supplied = Path(root)
        if not supplied.is_absolute():
            raise ValueError("confirm output directory must be absolute")
        self.root = supplied
        self._attempt_started = False

    def start_attempt(self, value: Mapping[str, Any]) -> None:
        """Claim one output identity before external or semantic access begins."""
        if self.root.exists() or self.root.is_symlink():
            raise FileExistsError("confirm output already exists; retry is forbidden")
        self._write_files(
            {ATTEMPT_FILENAME: pretty_json_bytes(dict(value))},
            create_root=True,
        )
        self._attempt_started = True

    def _write_files(self, files: Mapping[str, bytes], *, create_root: bool) -> None:
        if create_root:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        elif not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("confirm output root is missing or unsafe")
        for relative, payload in files.items():
            safe = _safe_relative(relative, label="local evidence path")
            target = self.root.joinpath(*PurePosixPath(safe).parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open("xb") as destination:
                destination.write(payload)
                destination.flush()
                os.fsync(destination.fileno())
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def persist_label_blind(
        self,
        files: Mapping[str, bytes],
        inventory: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if (self.root.exists() or self.root.is_symlink()) and not self._attempt_started:
            raise FileExistsError("confirm output already exists; retry is forbidden")
        copied = {path: bytes(payload) for path, payload in files.items()}
        if _inventory(copied) != tuple(dict(item) for item in inventory):
            raise ValueError("label-blind local persistence inventory drifted")
        self._write_files(copied, create_root=not self._attempt_started)
        replay = {
            path: self.root.joinpath(*PurePosixPath(path).parts).read_bytes()
            for path in copied
        }
        if replay != copied:
            raise RuntimeError("label-blind local readback differs from persisted bytes")
        digest = _sha256(canonical_json_bytes(list(inventory)))
        return {
            "durable": True,
            "file_count": len(inventory),
            "inventory_sha256": digest,
            "persistence_id": f"local:{self.root}:{digest}",
        }

    def persist_report(self, files: Mapping[str, bytes]) -> None:
        self._write_files(files, create_root=False)

    def write_completion(self, value: Mapping[str, Any]) -> None:
        if self.has_failure():
            raise RuntimeError("confirm failure terminal already exists")
        self._write_files(
            {COMPLETION_FILENAME: pretty_json_bytes(dict(value))},
            create_root=False,
        )

    def write_failure(self, value: Mapping[str, Any]) -> None:
        if self.has_completion():
            raise RuntimeError("confirm completion terminal already exists")
        files = {FAILURE_FILENAME: pretty_json_bytes(dict(value))}
        self._write_files(files, create_root=not self.root.exists())

    def has_failure(self) -> bool:
        return (self.root / FAILURE_FILENAME).is_file()

    def has_completion(self) -> bool:
        return (self.root / COMPLETION_FILENAME).is_file()


def _validate_contract_for_execution(contract: IndependentConfirmContract) -> None:
    authorization = contract.data.get("authorization")
    geometry = contract.data.get("confirm_geometry")
    runtime = contract.data.get("runtime_contract")
    if (
        contract.data.get("protocol_id") != CONTRACT_PROTOCOL_ID
        or not isinstance(authorization, Mapping)
        or authorization.get("sealed_test_authorized") is not False
        or not isinstance(geometry, Mapping)
        or geometry.get("state_count") != CONFIRM_STATE_COUNT
        or geometry.get("source_ids") != list(CONFIRM_SOURCE_IDS)
        or not isinstance(runtime, Mapping)
        or runtime.get("worker_count") != WORKER_COUNT
        or runtime.get("states_per_worker") != STATES_PER_WORKER
    ):
        raise ValueError("confirm execution contract identity or geometry drifted")


def _same_canonical_json(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    try:
        return canonical_json_bytes(dict(left)) == canonical_json_bytes(dict(right))
    except (TypeError, ValueError):
        return False


def _canonical_absolute_path_text(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = Path(value)
    return path.is_absolute() and str(path) == value and ".." not in path.parts


def _gpu_topology_expected_runtime_identities(
    contract: IndependentConfirmContract,
) -> Mapping[str, Mapping[str, Any]]:
    restoration = contract.data.get("restoration_contract")
    runtime = contract.data.get("runtime_contract")
    if (
        not isinstance(restoration, Mapping)
        or not isinstance(runtime, Mapping)
        or restoration.get("policy_repo") != MODEL_REPO
        or restoration.get("policy_revision") != MODEL_REVISION
        or restoration.get("snapshot_manifest_sha256")
        != SNAPSHOT_MANIFEST_SHA256
        or restoration.get("dtype") != "bfloat16"
        or restoration.get("attention_implementation")
        != GUI_OWL_V2_2_EAGER_ATTENTION_IMPLEMENTATION
        or runtime.get("transformers_version") != TRANSFORMERS_VERSION
        or runtime.get("dtype") != "bfloat16"
    ):
        raise ValueError("topology smoke model/runtime contract identity drifted")
    common = {
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "snapshot_manifest_sha256": SNAPSHOT_MANIFEST_SHA256,
        "verified_model_file_count": MODEL_FILE_COUNT,
        "verified_model_total_bytes": MODEL_TOTAL_BYTES,
        "transformers_version": TRANSFORMERS_VERSION,
        "transformers_source_sha256": dict(TRANSFORMERS_SOURCE_SHA256),
        "dtype": FROZEN_GUI_OWL_V2_DTYPE,
        "model_class": MODEL_CLASS_NAME,
        "requested_attention_implementation": GUI_OWL_V2_2_EAGER_ATTENTION_IMPLEMENTATION,
        "observed_attention_implementation": {
            "top": GUI_OWL_V2_2_EAGER_ATTENTION_IMPLEMENTATION,
            "text": GUI_OWL_V2_2_EAGER_ATTENTION_IMPLEMENTATION,
            "vision": GUI_OWL_V2_2_EAGER_ATTENTION_IMPLEMENTATION,
        },
    }
    return MappingProxyType(
        {
            "policy_vision_spawn": MappingProxyType(
                {
                    **common,
                    "runtime_profile_id": (
                        GUI_OWL_V2_2_VISION_RUNTIME_UUID_SIZE_DICT_V3_ID
                    ),
                    "image_processor_class": (
                        GUI_OWL_V2_2_VISION_IMAGE_PROCESSOR_CLASS
                    ),
                }
            ),
            "teacher_forced_fork": MappingProxyType(
                {
                    **common,
                    "runtime_profile_id": GUI_OWL_V2_2_EAGER_RUNTIME_ID,
                    "processor_class": "Qwen3VLProcessor",
                    "protocol_id": GUI_OWL_V2_1_PROTOCOL_ID,
                    "generation_interface": (
                        "processor_apply_chat_template_official_tools_kwarg"
                    ),
                    "official_tool_schema_sha256": CANONICAL_TOOL_SCHEMA_SHA256,
                    "chat_template_file_sha256": CHAT_TEMPLATE_FILE_SHA256,
                    "chat_template_text_sha256": CHAT_TEMPLATE_TEXT_SHA256,
                }
            ),
        }
    )


def validate_gpu_topology_smoke_receipt(
    receipt: Mapping[str, Any],
    *,
    contract: IndependentConfirmContract,
    expected_devices: Sequence[str],
    expected_gpu_uuids: Sequence[str],
    expected_model_dir: str | Path | None = None,
    expected_snapshot_manifest: str | Path | None = None,
) -> Mapping[str, Any]:
    devices, gpu_uuids = _validate_device_allocation(
        expected_devices,
        expected_gpu_uuids,
        label="GPU topology smoke receipt",
    )
    expected_top_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "scope",
        "inputs",
        "controls",
        "allocation",
        "phase_order",
        "phases",
    }
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != expected_top_keys
        or receipt.get("schema_version") != GPU_TOPOLOGY_SMOKE_SCHEMA_VERSION
        or receipt.get("protocol_id") != GPU_TOPOLOGY_SMOKE_PROTOCOL_ID
        or receipt.get("status") != "PASS"
        or not isinstance(receipt.get("scope"), Mapping)
        or not _same_canonical_json(receipt["scope"], _GPU_TOPOLOGY_SMOKE_SCOPE)
        or receipt.get("phase_order") != list(GPU_TOPOLOGY_SMOKE_PHASE_ORDER)
    ):
        raise ValueError("GPU topology smoke receipt identity, status, or scope drifted")

    inputs = receipt.get("inputs")
    expected_input_keys = {
        "model_dir",
        "snapshot_manifest",
        "fixed_rgb_image_count",
        "fixed_rgb_image_set_sha256",
    }
    if (
        not isinstance(inputs, Mapping)
        or set(inputs) != expected_input_keys
        or inputs.get("fixed_rgb_image_count") != GPU_TOPOLOGY_SMOKE_IMAGE_COUNT
        or inputs.get("fixed_rgb_image_set_sha256")
        != GPU_TOPOLOGY_SMOKE_FIXED_RGB_IMAGE_SET_SHA256
        or not _canonical_absolute_path_text(inputs.get("model_dir"))
        or not _canonical_absolute_path_text(inputs.get("snapshot_manifest"))
    ):
        raise ValueError("GPU topology smoke receipt input identity drifted")
    if expected_model_dir is not None and inputs["model_dir"] != str(
        Path(expected_model_dir)
    ):
        raise ValueError("GPU topology smoke receipt model directory drifted")
    if expected_snapshot_manifest is not None and inputs["snapshot_manifest"] != str(
        Path(expected_snapshot_manifest)
    ):
        raise ValueError("GPU topology smoke receipt snapshot manifest drifted")

    controls = receipt.get("controls")
    scientific_runtime = contract.data.get("runtime_contract")
    if (
        not isinstance(controls, Mapping)
        or set(controls)
        != {"phase_timeout_seconds", "worker_termination_grace_seconds"}
        or not isinstance(scientific_runtime, Mapping)
        or type(controls.get("phase_timeout_seconds")) is not int
        or controls.get("phase_timeout_seconds")
        != scientific_runtime.get("gpu_topology_smoke_phase_timeout_seconds")
        or type(controls.get("worker_termination_grace_seconds")) is not int
        or controls.get("worker_termination_grace_seconds")
        != scientific_runtime.get("worker_termination_grace_seconds")
    ):
        raise ValueError("GPU topology smoke receipt runtime controls drifted")

    allocation = receipt.get("allocation")
    expected_allocation = [
        {
            "worker_id": f"worker-{index}",
            "device": device,
            "gpu_uuid": gpu_uuid,
        }
        for index, (device, gpu_uuid) in enumerate(
            zip(devices, gpu_uuids, strict=True)
        )
    ]
    if allocation != expected_allocation:
        raise ValueError("GPU topology smoke receipt allocation drifted")

    phases = receipt.get("phases")
    if not isinstance(phases, Mapping) or set(phases) != set(
        GPU_TOPOLOGY_SMOKE_PHASE_ORDER
    ):
        raise ValueError("GPU topology smoke receipt phase inventory drifted")
    expected_identities = _gpu_topology_expected_runtime_identities(contract)
    phase_contracts = {
        "policy_vision_spawn": {
            "keys": {
                "start_method",
                "worker_count",
                "image_count_per_worker",
                "feature_repeats",
                "operation_counts",
                "workers",
            },
            "start_method": "spawn",
            "extra_key": "feature_repeats",
            "extra_value": GPU_TOPOLOGY_SMOKE_FEATURE_REPEATS,
            "counts": _GPU_TOPOLOGY_VISION_COUNTS_PER_WORKER,
        },
        "teacher_forced_fork": {
            "keys": {
                "start_method",
                "worker_count",
                "image_count_per_worker",
                "canonical_teacher_action",
                "operation_counts",
                "workers",
            },
            "start_method": "fork",
            "extra_key": "canonical_teacher_action",
            "extra_value": "wait",
            "counts": _GPU_TOPOLOGY_TEACHER_COUNTS_PER_WORKER,
        },
    }
    common_identity_keys = {
        "model_repo",
        "model_revision",
        "snapshot_manifest_sha256",
        "verified_model_file_count",
        "verified_model_total_bytes",
        "transformers_version",
        "transformers_source_sha256",
        "dtype",
        "model_class",
        "requested_attention_implementation",
        "observed_attention_implementation",
    }
    cross_phase_common: Mapping[str, Any] | None = None
    for phase_name in GPU_TOPOLOGY_SMOKE_PHASE_ORDER:
        phase = phases[phase_name]
        phase_contract = phase_contracts[phase_name]
        per_worker_counts = phase_contract["counts"]
        expected_totals = {
            key: value * WORKER_COUNT for key, value in per_worker_counts.items()
        }
        workers = phase.get("workers") if isinstance(phase, Mapping) else None
        if (
            not isinstance(phase, Mapping)
            or set(phase) != phase_contract["keys"]
            or phase.get("start_method") != phase_contract["start_method"]
            or phase.get("worker_count") != WORKER_COUNT
            or phase.get("image_count_per_worker")
            != GPU_TOPOLOGY_SMOKE_IMAGE_COUNT
            or phase.get(phase_contract["extra_key"])
            != phase_contract["extra_value"]
            or not isinstance(phase.get("operation_counts"), Mapping)
            or not _same_canonical_json(
                phase["operation_counts"], expected_totals
            )
            or not isinstance(workers, list)
            or len(workers) != WORKER_COUNT
        ):
            raise ValueError(
                f"GPU topology smoke receipt {phase_name} order or counts drifted"
            )
        for index, (worker, allocation_item) in enumerate(
            zip(workers, expected_allocation, strict=True)
        ):
            if (
                not isinstance(worker, Mapping)
                or set(worker)
                != {
                    "worker_id",
                    "device",
                    "gpu_uuid",
                    "fixed_rgb_image_set_sha256",
                    "runtime_identity",
                    "operation_counts",
                }
                or any(
                    worker.get(key) != allocation_item[key]
                    for key in ("worker_id", "device", "gpu_uuid")
                )
                or worker.get("fixed_rgb_image_set_sha256")
                != inputs["fixed_rgb_image_set_sha256"]
                or not isinstance(worker.get("operation_counts"), Mapping)
                or not _same_canonical_json(
                    worker["operation_counts"], per_worker_counts
                )
                or not isinstance(worker.get("runtime_identity"), Mapping)
                or not _same_canonical_json(
                    worker["runtime_identity"], expected_identities[phase_name]
                )
            ):
                raise ValueError(
                    "GPU topology smoke receipt "
                    f"{phase_name} worker-{index} identity or counts drifted"
                )
            observed_common = {
                key: worker["runtime_identity"][key]
                for key in common_identity_keys
            }
            if cross_phase_common is None:
                cross_phase_common = observed_common
            elif not _same_canonical_json(observed_common, cross_phase_common):
                raise ValueError(
                    "GPU topology smoke receipt model/runtime identity differs "
                    "across phases"
                )
    try:
        normalized = json.loads(canonical_json_bytes(dict(receipt)))
    except (TypeError, ValueError) as error:
        raise ValueError("GPU topology smoke receipt is not canonical JSON") from error
    if not isinstance(normalized, dict):
        raise TypeError("GPU topology smoke receipt must be a JSON object")
    return MappingProxyType(normalized)


def _validate_run_contract(
    run_contract: Mapping[str, Any],
    *,
    contract: IndependentConfirmContract,
) -> Mapping[str, Any]:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "contract_sha256",
        "source_a_commit",
        "execution_b_commit",
        "runner_freeze",
        "parent_artifact",
        "independent_model",
        "policy_snapshot",
        "gpu_topology_smoke_receipt",
        "runtime",
        "publication",
        "prohibited_counts",
        "execution_argv_sha256",
    }
    if (
        not isinstance(run_contract, Mapping)
        or set(run_contract) != expected_keys
        or run_contract.get("schema_version") != SCHEMA_VERSION
        or run_contract.get("protocol_id") != RUN_CONTRACT_PROTOCOL_ID
        or run_contract.get("contract_sha256") != contract.sha256
        or _COMMIT.fullmatch(str(run_contract.get("source_a_commit"))) is None
        or _COMMIT.fullmatch(str(run_contract.get("execution_b_commit"))) is None
    ):
        raise ValueError("confirm run-contract identity or source binding drifted")
    runtime = run_contract.get("runtime")
    argv = runtime.get("execution_argv") if isinstance(runtime, Mapping) else None
    devices = runtime.get("devices") if isinstance(runtime, Mapping) else None
    gpu_uuids = runtime.get("gpu_uuids") if isinstance(runtime, Mapping) else None
    expected_runtime_keys = {
        "host_alias",
        "host_hostname",
        "container_id",
        "container_image_digest",
        "devices",
        "gpu_uuids",
        "worker_count",
        "states_per_worker",
        "policy_vision_feature_repeats",
        "restoration_teacher_forwards_per_valid_state",
        "policy_vision_phase_timeout_seconds",
        "restoration_phase_timeout_seconds",
        "worker_termination_grace_seconds",
        "execution_argv",
    }
    scientific_runtime = contract.data.get("runtime_contract")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != expected_runtime_keys
        or runtime.get("worker_count") != WORKER_COUNT
        or runtime.get("states_per_worker") != STATES_PER_WORKER
        or runtime.get("policy_vision_feature_repeats")
        != POLICY_VISION_FEATURE_REPEATS
        or runtime.get("restoration_teacher_forwards_per_valid_state") != 17
        or not isinstance(scientific_runtime, Mapping)
        or runtime.get("policy_vision_phase_timeout_seconds")
        != scientific_runtime.get("policy_vision_phase_timeout_seconds")
        or runtime.get("restoration_phase_timeout_seconds")
        != scientific_runtime.get("restoration_phase_timeout_seconds")
        or runtime.get("worker_termination_grace_seconds")
        != scientific_runtime.get("worker_termination_grace_seconds")
        or any(
            type(runtime.get(key)) is not int or runtime[key] <= 0
            for key in (
                "policy_vision_phase_timeout_seconds",
                "restoration_phase_timeout_seconds",
                "worker_termination_grace_seconds",
            )
        )
        or not isinstance(devices, list)
        or len(devices) != WORKER_COUNT
        or len(set(devices)) != WORKER_COUNT
        or any(_CUDA_DEVICE.fullmatch(str(item)) is None for item in devices)
        or not isinstance(gpu_uuids, list)
        or len(gpu_uuids) != WORKER_COUNT
        or len(set(gpu_uuids)) != WORKER_COUNT
        or any(_GPU_UUID.fullmatch(str(item)) is None for item in gpu_uuids)
        or not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
        or argv.count("--hf-token-file") != 1
        or argv.index("--hf-token-file") == len(argv) - 1
        or any(
            item.startswith("--hf-token") and item != "--hf-token-file"
            for item in argv
        )
        or argv.count("--gpu-topology-smoke-receipt") != 1
        or argv.index("--gpu-topology-smoke-receipt") == len(argv) - 1
        or argv.count("--gpu-topology-smoke-receipt-sha256") != 1
        or argv.index("--gpu-topology-smoke-receipt-sha256") == len(argv) - 1
        or run_contract.get("execution_argv_sha256")
        != _sha256(canonical_json_bytes(argv))
    ):
        raise ValueError("confirm run-contract four-by-five runtime binding drifted")
    prohibited = run_contract.get("prohibited_counts")
    expected_prohibited = {
        "retry_count",
        "top_up_count",
        "filter_count",
        "gate_training_count",
        "threshold_update_count",
        "sealed_androidworld_test_access_count",
    }
    if (
        not isinstance(prohibited, Mapping)
        or set(prohibited) != expected_prohibited
        or any(type(prohibited[key]) is not int or prohibited[key] != 0 for key in prohibited)
    ):
        raise ValueError("confirm run-contract prohibited operation count is nonzero")
    parent = run_contract.get("parent_artifact")
    model = run_contract.get("independent_model")
    policy_snapshot = run_contract.get("policy_snapshot")
    topology = run_contract.get("gpu_topology_smoke_receipt")
    runner_freeze = run_contract.get("runner_freeze")
    topology_path = topology.get("path") if isinstance(topology, Mapping) else None
    topology_sha256 = (
        topology.get("sha256") if isinstance(topology, Mapping) else None
    )
    topology_body = topology.get("body") if isinstance(topology, Mapping) else None
    if (
        not isinstance(topology, Mapping)
        or set(topology) != {"path", "sha256", "body"}
        or not _canonical_absolute_path_text(topology_path)
        or _SHA256.fullmatch(str(topology_sha256)) is None
        or not isinstance(topology_body, Mapping)
        or topology_sha256
        != _sha256(canonical_json_bytes(dict(topology_body)) + b"\n")
        or argv[argv.index("--gpu-topology-smoke-receipt") + 1] != topology_path
        or argv[argv.index("--gpu-topology-smoke-receipt-sha256") + 1]
        != topology_sha256
    ):
        raise ValueError("confirm run-contract GPU topology receipt binding drifted")
    validated_topology = validate_gpu_topology_smoke_receipt(
        topology_body,
        contract=contract,
        expected_devices=devices,
        expected_gpu_uuids=gpu_uuids,
    )
    topology_inputs = validated_topology["inputs"]
    expected_policy_snapshot_keys = {
        "model_dir",
        "model_repo",
        "model_revision",
        "snapshot_manifest_sha256",
        "verified_model_file_count",
        "verified_model_total_bytes",
    }
    if (
        not isinstance(runner_freeze, Mapping)
        or runner_freeze.get("source_a_commit")
        != run_contract.get("source_a_commit")
        or runner_freeze.get("contract_sha256") != contract.sha256
        or runner_freeze.get("confirm_access_authorized") is not True
        or runner_freeze.get("restoration_access_requires_payload_commit") is not True
        or runner_freeze.get("sealed_test_requires_development_go") is not True
        or not isinstance(parent, Mapping)
        or _SHA256.fullmatch(str(parent.get("artifact_tree_sha256"))) is None
        or parent.get("source_ids") != list(CONFIRM_SOURCE_IDS)
        or parent.get("state_count") != CONFIRM_STATE_COUNT
        or parent.get("policy_output_present") is not False
        or parent.get("restoration_output_present") is not False
        or not isinstance(model, Mapping)
        or model.get("repo") != contract.model.get("repo")
        or model.get("payload_commit") != contract.model.get("payload_commit")
        or model.get("manifest_commit") != contract.model.get("manifest_commit")
        or model.get("checkpoint_count") != 5
        or model.get("retraining_performed") is not False
        or not isinstance(policy_snapshot, Mapping)
        or set(policy_snapshot) != expected_policy_snapshot_keys
        or policy_snapshot.get("model_dir") != topology_inputs["model_dir"]
        or policy_snapshot.get("model_repo")
        != contract.data["restoration_contract"]["policy_repo"]
        or policy_snapshot.get("model_revision")
        != contract.data["restoration_contract"]["policy_revision"]
        or policy_snapshot.get("snapshot_manifest_sha256")
        != contract.data["restoration_contract"]["snapshot_manifest_sha256"]
        or policy_snapshot.get("verified_model_file_count") != MODEL_FILE_COUNT
        or policy_snapshot.get("verified_model_total_bytes") != MODEL_TOTAL_BYTES
        or dict(run_contract.get("publication", {})) != dict(contract.destination)
    ):
        raise ValueError("confirm run-contract label-blind substrate binding drifted")
    return MappingProxyType(json.loads(canonical_json_bytes(dict(run_contract))))


def _ordered_state_records(
    worker_results: Sequence[ConfirmWorkerResult],
    *,
    selection_sha256: str,
    run_contract_sha256: str,
    payload_commit: str,
) -> tuple[Mapping[str, Any], ...]:
    records = [
        record
        for result in worker_results
        for record in result.state_records
    ]
    records.sort(key=lambda item: int(item["state"]["ordinal"]))
    expected = [
        (
            ordinal,
            source_id,
            f"{source_id}:decision_step:006",
        )
        for ordinal, source_id in enumerate(CONFIRM_SOURCE_IDS)
    ]
    observed = [
        (
            record["state"]["ordinal"],
            record["state"]["source_id"],
            record["state"]["state_id"],
        )
        for record in records
    ]
    if observed != expected:
        raise ValueError("nested raw state record roster or order drifted")
    if any(record.get("selection_sha256") != selection_sha256 for record in records):
        raise ValueError("raw state record selection digest differs from sealed decisions")
    if any(
        record.get("run_contract_sha256") != run_contract_sha256
        for record in records
    ):
        raise ValueError("raw state record run-contract digest drifted")
    if any(record.get("payload_commit") != payload_commit for record in records):
        raise ValueError("raw state record payload commit drifted")
    return tuple(records)


def validate_restoration_phase(
    result: RestorationPhaseResult,
    *,
    assignments: Sequence[ConfirmWorkerAssignment],
    expected_devices: Sequence[str] | None = None,
    expected_gpu_uuids: Sequence[str] | None = None,
) -> RestorationPhaseResult:
    if not isinstance(result, RestorationPhaseResult):
        raise TypeError("restoration phase must return RestorationPhaseResult")
    frozen_assignments = tuple(assignments)
    worker_results = tuple(result.worker_results)
    metadata = tuple(result.worker_metadata)
    if (
        len(frozen_assignments) != WORKER_COUNT
        or len(worker_results) != WORKER_COUNT
        or len(metadata) != WORKER_COUNT
    ):
        raise ValueError("restoration phase must preserve the frozen four workers")
    device_pairs = []
    for assignment, worker_result, item in zip(
        frozen_assignments, worker_results, metadata, strict=True
    ):
        if (
            worker_result.worker_id != assignment.worker_id
            or len(worker_result.state_records) != STATES_PER_WORKER
            or not isinstance(item, Mapping)
            or item.get("worker_id") != assignment.worker_id
            or item.get("state_count") != STATES_PER_WORKER
            or _CUDA_DEVICE.fullmatch(str(item.get("device"))) is None
            or _GPU_UUID.fullmatch(str(item.get("gpu_uuid"))) is None
        ):
            raise ValueError("restoration worker shard/device metadata drifted")
        device_pairs.append((str(item["device"]), str(item["gpu_uuid"])))
    if (
        len({device for device, _ in device_pairs}) != WORKER_COUNT
        or len({gpu_uuid for _, gpu_uuid in device_pairs}) != WORKER_COUNT
    ):
        raise ValueError("restoration worker device identities must be unique")
    if expected_devices is not None or expected_gpu_uuids is not None:
        devices, gpu_uuids = _validate_device_allocation(
            () if expected_devices is None else expected_devices,
            () if expected_gpu_uuids is None else expected_gpu_uuids,
            label="restoration run contract",
        )
        if device_pairs != list(zip(devices, gpu_uuids, strict=True)):
            raise ValueError("restoration metadata differs from run-contract devices")
    return RestorationPhaseResult(
        worker_results=worker_results,
        worker_metadata=metadata,
    )


def execute_independent_confirm(
    *,
    contract: IndependentConfirmContract,
    bundle: Confirm20LabelBlindBundle,
    ensemble: FittedEnsemble,
    run_contract: Mapping[str, Any],
    output_dir: str | Path,
    publisher: ConfirmPublisher,
    policy_vision_phase: PolicyVisionPhase,
    restoration_phase: RestorationPhase,
    independent_scorer: IndependentScorer = score_and_select_independent,
) -> ConfirmExecutionResult:
    """Run the one-way payload barrier, restoration, evaluation, and publication."""
    _validate_contract_for_execution(contract)
    frozen_run_contract = _validate_run_contract(run_contract, contract=contract)
    if not isinstance(bundle, Confirm20LabelBlindBundle):
        raise TypeError("confirm execution requires a validated label-blind bundle")
    run_contract_bytes = canonical_json_bytes(dict(frozen_run_contract))
    run_contract_sha256 = _sha256(run_contract_bytes)
    output = DurableConfirmOutput(output_dir)
    payload_seal: LabelBlindPayloadSeal | None = None
    report_files: Mapping[str, bytes] = {}
    try:
        publisher.preflight_destination()
        independent = independent_scorer(bundle.feature_states, ensemble)
        policy_vision = validate_policy_vision_phase(
            policy_vision_phase(bundle),
            bundle=bundle,
            expected_devices=frozen_run_contract["runtime"]["devices"],
            expected_gpu_uuids=frozen_run_contract["runtime"]["gpu_uuids"],
        )
        payload_seal = build_canonical_label_blind_payload(
            bundle=bundle,
            independent=independent,
            policy_vision=policy_vision,
        )
        publisher.bind_payload_seal(payload_seal)
        local_seal = persist_and_seal_label_blind_payload(
            payload_seal.files,
            independent_selections=independent,
            independent_selection_path=INDEPENDENT_DECISION_PATH,
            persist_fn=output.persist_label_blind,
        )
        receipt = issue_runner_payload_receipt(
            local_seal,
            publish_fn=publisher.publish_payload,
            verify_fn=publisher.verify_payload,
        )
        assignments = confirm_worker_assignments(bundle.work_items)
        restoration = validate_restoration_phase(
            restoration_phase(assignments, receipt, run_contract_sha256),
            assignments=assignments,
            expected_devices=frozen_run_contract["runtime"]["devices"],
            expected_gpu_uuids=frozen_run_contract["runtime"]["gpu_uuids"],
        )
        heuristics = {
            "dynamic_recent": bundle.dynamic_recent,
            "ocr_rgb_v2": bundle.ocr_rgb_v2,
            "policy_vision_v3": policy_vision.selections,
        }
        aggregate = aggregate_independent_confirm_workers(
            bundle.feature_states,
            assignments=assignments,
            worker_results=restoration.worker_results,
            independent_selections=independent,
            heuristic_selections=heuristics,
        )
        execution = aggregate.get("execution")
        if not isinstance(execution, Mapping):
            raise ValueError("confirm aggregate lacks execution evidence")
        fixed_report = {
            key: value for key, value in aggregate.items() if key != "execution"
        }
        source_commit = frozen_run_contract.get("execution_b_commit")
        if not isinstance(source_commit, str) or _COMMIT.fullmatch(source_commit) is None:
            raise ValueError("confirm run contract lacks Execution-B source commit")
        raw_records = _ordered_state_records(
            restoration.worker_results,
            selection_sha256=_sha256(
                payload_seal.files[INDEPENDENT_DECISION_PATH]
            ),
            run_contract_sha256=run_contract_sha256,
            payload_commit=receipt.payload_commit,
        )
        report_files = build_report_files(
            state_records=raw_records,
            fixed_report=fixed_report,
            run_manifest={
                "source_git_commit": source_commit,
                "contract_sha256": contract.sha256,
                "runtime_metadata": {
                    "run_contract_sha256": run_contract_sha256,
                    "run_contract": dict(frozen_run_contract),
                    "policy_vision_workers": [
                        dict(item) for item in policy_vision.worker_metadata
                    ],
                    "restoration_workers": [
                        dict(item) for item in restoration.worker_metadata
                    ],
                },
                "operation_counts": dict(execution["operation_counts"]),
            },
            payload_seal=payload_seal,
            payload_commit=receipt.payload_commit,
        )
        if set(report_files) != set(REPORT_TARGETS):
            raise RuntimeError("canonical report is not the frozen four files")
        output.persist_report(report_files)
        final = publisher.publish_report(
            report_files,
            payload_commit=receipt.payload_commit,
        )
        if (
            final.get("payload_commit") != receipt.payload_commit
            or not isinstance(final.get("report_commit"), str)
            or _COMMIT.fullmatch(str(final["report_commit"])) is None
            or not isinstance(final.get("annotated_tag_object"), str)
            or _COMMIT.fullmatch(str(final["annotated_tag_object"])) is None
            or final.get("created_tag_count") != 1
            or final.get("byte_identical_fresh_replay") is not True
        ):
            raise ValueError("final report/tag/replay receipt is malformed")
        completion = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETED_AND_PUBLISHED_INDEPENDENT_CONFIRM20_V1",
            "confirm_outcome": fixed_report.get("status"),
            "confirm_go": fixed_report.get("go"),
            "base_commit": final.get("base_commit"),
            "payload_commit": receipt.payload_commit,
            "report_commit": final["report_commit"],
            "report_parent_commit": receipt.payload_commit,
            "annotated_tag_object": final["annotated_tag_object"],
            "created_tag_count": 1,
            "byte_identical_fresh_replay": True,
            "payload_file_count": len(PAYLOAD_TARGETS),
            "report_file_count": len(REPORT_TARGETS),
            "fixed_state_denominator": CONFIRM_STATE_COUNT,
            "sealed_androidworld_test_access_count": 0,
            "retry_count": 0,
            "top_up_count": 0,
            "filter_count": 0,
        }
        output.write_completion(completion)
        return ConfirmExecutionResult(
            completion=MappingProxyType(completion),
            report=MappingProxyType(fixed_report),
            payload_files=MappingProxyType(dict(payload_seal.files)),
            report_files=MappingProxyType(dict(report_files)),
        )
    except BaseException as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "INVALID_INDEPENDENT_CONFIRM20_EXECUTION_V1",
            "exception_type": error.__class__.__name__,
            "message": str(error),
            "payload_locally_sealed": payload_seal is not None,
            "report_locally_persisted": bool(report_files),
            "sealed_androidworld_test_access_count": 0,
            "retry_count": 0,
            "top_up_count": 0,
            "filter_count": 0,
        }
        try:
            output.write_failure(failure)
        except Exception as evidence_error:
            error.add_note(
                "failed to persist invalid confirm terminal: "
                f"{evidence_error.__class__.__name__}: {evidence_error}"
            )
        raise


def read_hf_token_file(path: str | Path) -> str:
    """Read one token from a no-follow 0400/0600 regular file."""
    descriptor: int | None = None
    try:
        descriptor = os.open(
            Path(path),
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) not in {0o400, 0o600}
        ):
            raise ValueError("HF token file must be regular mode 0400 or 0600")
        chunks = []
        while chunk := os.read(descriptor, 4096):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError("HF token file is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    fingerprint = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if fingerprint(before) != fingerprint(after) or len(payload) != after.st_size:
        raise ValueError("HF token file changed during its read")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("HF token file must contain UTF-8") from error
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0] or lines[0].strip() != lines[0]:
        raise ValueError("HF token file must contain exactly one nonempty token")
    return lines[0]


class HuggingFaceConfirmPublisher:
    """Thin credential adapter over the frozen artifact publication state machine."""

    __slots__ = (
        "_api",
        "_cache_dir",
        "_download_fn",
        "_operation_factory",
        "_payload_receipt",
        "_payload_seal",
        "_preflight_base_commit",
        "_report_publication_attempted",
        "_report_publication_audit",
        "_token",
    )

    def __init__(self, *, cache_dir: str | Path, token: str) -> None:
        cache = Path(cache_dir)
        if not cache.is_absolute() or not token:
            raise ValueError("HF cache must be absolute and token nonempty")
        cache.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
        except ModuleNotFoundError as error:
            raise RuntimeError("formal confirm publication requires huggingface_hub") from error
        self._cache_dir = cache
        self._token = token
        self._api = HfApi(token=token)
        self._download_fn = hf_hub_download
        self._operation_factory = CommitOperationAdd
        self._payload_seal: LabelBlindPayloadSeal | None = None
        self._payload_receipt: PayloadPublicationReceipt | None = None
        self._preflight_base_commit: str | None = None
        self._report_publication_attempted = False
        self._report_publication_audit: Mapping[str, Any] | None = None

    def __repr__(self) -> str:
        return "HuggingFaceConfirmPublisher(token=<redacted>)"

    def bind_payload_seal(self, seal: LabelBlindPayloadSeal) -> None:
        if self._payload_seal is not None:
            raise RuntimeError("payload seal may be bound only once")
        if set(seal.files) != set(PAYLOAD_TARGETS):
            raise ValueError("publisher requires the canonical eight-file payload")
        self._payload_seal = seal

    def adopt_existing_payload(
        self,
        *,
        expected_files: Mapping[str, bytes],
        expected_base_commit: str,
        expected_base_title: str,
        expected_payload_commit: str,
    ) -> Mapping[str, Any]:
        """Adopt the frozen payload stage without enabling payload publication."""
        if (
            self._payload_seal is not None
            or self._payload_receipt is not None
            or self._preflight_base_commit is not None
        ):
            raise RuntimeError(
                "existing payload may be adopted only by a fresh publisher"
            )
        fresh_parent = self._cache_dir / "adopted-payload-fresh-replay"
        fresh_parent.mkdir(mode=0o700, parents=True, exist_ok=True)

        def download(**kwargs: Any) -> str:
            return self._download_fn(**kwargs, token=self._token)

        adopted = adopt_artifact_existing_payload_commit(
            api=self._api,
            download_fn=download,
            expected_base_commit=expected_base_commit,
            expected_base_title=expected_base_title,
            expected_payload_commit=expected_payload_commit,
            expected_payload_files=expected_files,
            fresh_parent=fresh_parent,
        )
        self._payload_seal = adopted.payload_seal
        self._payload_receipt = adopted.payload_receipt
        return MappingProxyType(
            {
                "base_commit": adopted.payload_receipt.base_commit,
                "payload_commit": adopted.payload_receipt.payload_commit,
                "payload_inventory_sha256": (
                    adopted.payload_receipt.payload_inventory_sha256
                ),
                "payload_file_count": len(adopted.payload_seal.files),
                "byte_identical_fresh_replay": True,
                "remote_mutation_performed": False,
            }
        )

    def preflight_destination(self) -> Mapping[str, Any]:
        self.ensure_destination_base()
        info = self._api.repo_info(
            DESTINATION_REPO,
            repo_type=DESTINATION_REPO_TYPE,
            revision="main",
        )
        commit = getattr(info, "sha", None)
        files = frozenset(
            self._api.list_repo_files(
                DESTINATION_REPO,
                repo_type=DESTINATION_REPO_TYPE,
                revision="main",
            )
        )
        refs = self._api.list_repo_refs(
            DESTINATION_REPO, repo_type=DESTINATION_REPO_TYPE
        )
        tag_count = sum(
            getattr(item, "name", None) == DESTINATION_TAG for item in refs.tags
        )
        allowed_base_files = frozenset({"README.md", ".gitattributes"})
        if (
            getattr(info, "private", None) is not DESTINATION_PRIVATE
            or not isinstance(commit, str)
            or _COMMIT.fullmatch(commit) is None
            or files - allowed_base_files
            or set(files).intersection(PAYLOAD_TARGETS + REPORT_TARGETS)
            or tag_count != 0
        ):
            raise ValueError(
                "confirm destination is not a fresh private non-conflicting base"
            )
        if (
            self._preflight_base_commit is not None
            and commit != self._preflight_base_commit
        ):
            raise RuntimeError("confirm destination base commit moved after preflight")
        self._preflight_base_commit = commit
        return MappingProxyType(
            {
                "base_commit": commit,
                "base_files": sorted(files),
                "destination_private": True,
                "payload_target_count": 0,
                "report_target_count": 0,
                "tag_count": 0,
            }
        )

    def download_bytes(
        self,
        *,
        repo: str,
        repo_type: str,
        revision: str,
        path: str,
    ) -> bytes:
        relative = _safe_relative(path, label="HF download path")
        local = self._download_fn(
            repo_id=repo,
            repo_type=repo_type,
            revision=revision,
            filename=relative,
            cache_dir=str(self._cache_dir),
            token=self._token,
            force_download=True,
        )
        return Path(local).read_bytes()

    def ensure_destination_base(self) -> None:
        self._api.create_repo(
            DESTINATION_REPO,
            repo_type=DESTINATION_REPO_TYPE,
            private=DESTINATION_PRIVATE,
            exist_ok=True,
        )
        try:
            info = self._api.repo_info(
                DESTINATION_REPO, repo_type=DESTINATION_REPO_TYPE
            )
        except Exception:
            info = None
        if info is not None and getattr(info, "sha", None):
            return
        files = {
            ".gitattributes": b"*.json filter=lfs diff=lfs merge=lfs -text\n",
            "README.md": b"# CausalCache independent confirm-20\n",
        }
        operations = [
            self._operation_factory(
                path_in_repo=path,
                path_or_fileobj=io.BytesIO(files[path]),
            )
            for path in sorted(files)
        ]
        self._api.create_commit(
            DESTINATION_REPO,
            repo_type=DESTINATION_REPO_TYPE,
            commit_message="initialize independent confirm-20 private dataset",
            operations=operations,
        )

    def publish_payload(
        self,
        files: Mapping[str, bytes],
        inventory: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if (
            self._payload_seal is None
            or self._preflight_base_commit is None
            or dict(files) != dict(self._payload_seal.files)
        ):
            raise ValueError("runner payload bytes differ from the artifact seal")
        if _inventory(files) != tuple(dict(item) for item in inventory):
            raise ValueError("runner payload inventory differs from the artifact seal")
        self.ensure_destination_base()
        receipt = publish_artifact_payload_commit(
            api=self._api,
            operation_factory=self._operation_factory,
            payload_seal=self._payload_seal,
            expected_base_commit=self._preflight_base_commit,
        )
        if receipt.base_commit != self._preflight_base_commit:
            raise RuntimeError("payload receipt base differs from preflight base")
        self._payload_receipt = receipt
        return {
            "payload_commit": receipt.payload_commit,
            "inventory_sha256": _sha256(
                canonical_json_bytes([dict(item) for item in inventory])
            ),
        }

    def verify_payload(
        self,
        commit: str,
        inventory: Sequence[Mapping[str, Any]],
    ) -> bool:
        if (
            self._payload_seal is None
            or self._payload_receipt is None
            or commit != self._payload_receipt.payload_commit
            or _inventory(self._payload_seal.files)
            != tuple(dict(item) for item in inventory)
        ):
            return False
        with tempfile.TemporaryDirectory(
            prefix="independent-confirm20-payload-", dir=self._cache_dir
        ) as raw:
            replay_root = Path(raw).resolve()
            for path in PAYLOAD_TARGETS:
                returned = Path(
                    self._download_fn(
                        repo_id=DESTINATION_REPO,
                        repo_type=DESTINATION_REPO_TYPE,
                        filename=path,
                        revision=commit,
                        local_dir=replay_root,
                        token=self._token,
                        force_download=True,
                    )
                )
                if not returned.is_absolute() or returned != replay_root / path:
                    return False
                if returned.read_bytes() != self._payload_seal.files[path]:
                    return False
        return True

    def publish_report(
        self,
        report_files: Mapping[str, bytes],
        *,
        payload_commit: str,
    ) -> Mapping[str, Any]:
        if (
            self._payload_seal is None
            or self._payload_receipt is None
            or payload_commit != self._payload_receipt.payload_commit
        ):
            raise PermissionError("report publication lacks the payload-stage receipt")
        if self._report_publication_attempted:
            raise RuntimeError("report publication may be attempted only once")
        fresh_parent = self._cache_dir / "report-fresh-replay"
        fresh_parent.mkdir(mode=0o700, parents=True, exist_ok=True)

        def download(**kwargs: Any) -> str:
            return self._download_fn(**kwargs, token=self._token)

        self._report_publication_attempted = True
        try:
            final: FinalPublicationReceipt = publish_artifact_report_commit(
                api=self._api,
                operation_factory=self._operation_factory,
                download_fn=download,
                payload_seal=self._payload_seal,
                payload_receipt=self._payload_receipt,
                report_files=report_files,
                fresh_parent=fresh_parent,
            )
        except BaseException:
            self._report_publication_audit = reconcile_report_publication_state(
                api=self._api,
                payload_receipt=self._payload_receipt,
            )
            raise
        self._report_publication_audit = MappingProxyType(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "REPORT_COMMIT_AND_TAG_PRESENT",
                "payload_commit": final.payload_commit,
                "main_commit": final.report_commit,
                "main_tree_kind": "payload_plus_report",
                "payload_stage_intact": True,
                "report_commit": final.report_commit,
                "report_direct_child_verified": True,
                "annotated_tag_present": True,
                "annotated_tag_object": final.annotated_tag_object,
                "tag_resolved_commit": final.report_commit,
                "remote_mutation_count": 2,
                "minimum_remote_mutation_count": 2,
                "reconciliation_error": None,
            }
        )
        return {
            "base_commit": final.base_commit,
            "payload_commit": final.payload_commit,
            "report_commit": final.report_commit,
            "annotated_tag_object": final.annotated_tag_object,
            "created_tag_count": final.created_tag_count,
            "byte_identical_fresh_replay": True,
        }

    def report_publication_audit(self) -> Mapping[str, Any]:
        """Return the saved read-only reconciliation result without remote access."""
        if self._report_publication_audit is not None:
            return MappingProxyType(
                json.loads(canonical_json_bytes(dict(self._report_publication_audit)))
            )
        payload_commit = (
            None if self._payload_receipt is None else self._payload_receipt.payload_commit
        )
        return MappingProxyType(
            {
                "schema_version": SCHEMA_VERSION,
                "status": (
                    "REPORT_PUBLICATION_AUDIT_MISSING"
                    if self._report_publication_attempted
                    else "REPORT_PUBLICATION_NOT_ATTEMPTED"
                ),
                "payload_commit": payload_commit,
                "main_commit": payload_commit,
                "main_tree_kind": "payload" if payload_commit is not None else None,
                "payload_stage_intact": True if payload_commit is not None else None,
                "report_commit": None,
                "report_direct_child_verified": None,
                "annotated_tag_present": False,
                "annotated_tag_object": None,
                "tag_resolved_commit": None,
                "remote_mutation_count": (
                    None if self._report_publication_attempted else 0
                ),
                "minimum_remote_mutation_count": 0,
                "reconciliation_error": None,
            }
        )


def download_and_load_independent_ensemble(
    contract: IndependentConfirmContract,
    *,
    publisher: HuggingFaceConfirmPublisher,
) -> LoadedIndependentEnsemble:
    model = contract.model
    manifest = model["ensemble_manifest"]
    payloads = {
        manifest["path"]: publisher.download_bytes(
            repo=model["repo"],
            repo_type=model["repo_type"],
            revision=model["manifest_commit"],
            path=manifest["path"],
        )
    }
    for checkpoint in model["checkpoints"]:
        payloads[checkpoint["path"]] = publisher.download_bytes(
            repo=model["repo"],
            repo_type=model["repo_type"],
            revision=model["payload_commit"],
            path=checkpoint["path"],
        )
    return load_independent_ensemble(model, payloads)


def _decode_rgb_image(payload: bytes) -> Any:
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise RuntimeError("confirm image decoding requires Pillow") from error
    source = Image.open(io.BytesIO(payload))
    source.load()
    converted = source.convert("RGB")
    if converted is not source:
        source.close()
    return converted


def _collect_process_responses(
    *,
    processes: Sequence[Any],
    queue: Any,
    phase_label: str,
    phase_timeout_seconds: int,
    termination_grace_seconds: int,
) -> tuple[Mapping[str, Any], ...]:
    if (
        not phase_label
        or type(phase_timeout_seconds) is not int
        or phase_timeout_seconds <= 0
        or type(termination_grace_seconds) is not int
        or termination_grace_seconds <= 0
    ):
        raise ValueError("worker barrier requires explicit positive timeout controls")
    deadline = time.monotonic() + phase_timeout_seconds
    responses: list[Mapping[str, Any]] = []
    try:
        while len(responses) < len(processes):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{phase_label} exceeded {phase_timeout_seconds} seconds"
                )
            try:
                response = queue.get(timeout=min(1.0, remaining))
            except queue_module.Empty:
                if all(not process.is_alive() for process in processes):
                    break
                continue
            if not isinstance(response, Mapping):
                raise RuntimeError("confirm worker returned a malformed response")
            if response.get("ok") is not True:
                raise RuntimeError(f"{phase_label} worker failed: {dict(response)}")
            responses.append(response)
        for process in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0 and process.is_alive():
                raise TimeoutError(
                    f"{phase_label} exceeded {phase_timeout_seconds} seconds"
                )
            process.join(timeout=max(0.0, remaining))
        failures = [
            {"name": process.name, "exitcode": process.exitcode}
            for process in processes
            if process.exitcode != 0
        ]
        if failures or len(responses) != len(processes):
            raise RuntimeError(f"{phase_label} worker phase failed: {failures}")
        return tuple(responses)
    except BaseException as error:
        try:
            _terminate_and_join_workers(
                processes,
                termination_grace_seconds=termination_grace_seconds,
            )
        except BaseException as cleanup_error:
            error.add_note(
                "worker cleanup failed: "
                f"{cleanup_error.__class__.__name__}: {cleanup_error}"
            )
        raise


def _terminate_and_join_workers(
    processes: Sequence[Any],
    *,
    termination_grace_seconds: int,
) -> None:
    live = [process for process in processes if process.is_alive()]
    for process in live:
        process.terminate()
    deadline = time.monotonic() + termination_grace_seconds
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    survivors = [process for process in processes if process.is_alive()]
    for process in survivors:
        kill = getattr(process, "kill", None)
        if not callable(kill):
            raise RuntimeError(f"worker {process.name} survived terminate without kill")
        kill()
    for process in survivors:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    remaining = [process.name for process in processes if process.is_alive()]
    if remaining:
        raise RuntimeError(f"workers survived termination barrier: {remaining}")


def _validate_device_allocation(
    devices: Sequence[str],
    gpu_uuids: Sequence[str],
    *,
    label: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selected_devices = tuple(devices)
    selected_uuids = tuple(gpu_uuids)
    if (
        len(selected_devices) != WORKER_COUNT
        or len(set(selected_devices)) != WORKER_COUNT
        or any(_CUDA_DEVICE.fullmatch(item) is None for item in selected_devices)
        or len(selected_uuids) != WORKER_COUNT
        or len(set(selected_uuids)) != WORKER_COUNT
        or any(_GPU_UUID.fullmatch(item) is None for item in selected_uuids)
    ):
        raise ValueError(f"{label} requires four explicit CUDA devices and GPU UUIDs")
    return selected_devices, selected_uuids


class ForkParentCudaTripwire:
    """Block parent-side CUDA initialization until forked workers exist."""

    def __init__(self, torch_module: Any) -> None:
        self.torch_module = torch_module
        self.cuda = getattr(torch_module, "cuda", None)
        if self.cuda is None:
            raise RuntimeError("CUDA tripwire requires torch.cuda")
        names = ("is_available", "device_count", "_lazy_init")
        originals = {
            name: getattr(self.cuda, name, None)
            for name in names
            if callable(getattr(self.cuda, name, None))
        }
        if set(originals) != set(names):
            raise RuntimeError("CUDA tripwire entry-point inventory drifted")
        self._originals = originals
        self._blocked: dict[str, Callable[..., Any]] = {}
        self._active = False

    def install(self) -> None:
        if self._active:
            raise RuntimeError("CUDA tripwire is already active")

        def blocked(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(
                "CUDA initialization was attempted in the pre-fork parent"
            )

        self._blocked = {name: blocked for name in self._originals}
        for name in self._originals:
            setattr(self.cuda, name, blocked)
        self._active = True
        os.register_at_fork(after_in_child=self.restore_after_fork)

    def restore_after_fork(self) -> None:
        for name, function in self._originals.items():
            setattr(self.cuda, name, function)
        self._blocked = {}
        self._active = False

    def restore_parent(self) -> None:
        self.restore_after_fork()

    def assert_active(self) -> None:
        if not self._active or set(self._blocked) != set(self._originals) or any(
            getattr(self.cuda, name, None) is not self._blocked[name]
            for name in self._originals
        ):
            raise RuntimeError("CUDA parent tripwire is not active")


_ACTIVE_FORK_PARENT_CUDA_TRIPWIRE: ForkParentCudaTripwire | None = None


def install_fork_parent_cuda_tripwire(
    torch_module: Any | None = None,
    *,
    require_fresh_process: bool = False,
) -> ForkParentCudaTripwire:
    """Install a fork-parent tripwire before any formal preparation work."""
    global _ACTIVE_FORK_PARENT_CUDA_TRIPWIRE
    if _ACTIVE_FORK_PARENT_CUDA_TRIPWIRE is not None:
        raise RuntimeError("one CUDA parent tripwire is already installed")
    if torch_module is None:
        if require_fresh_process and "torch" in sys.modules:
            raise RuntimeError(
                "formal restoration coordinator must start before torch is imported"
            )
        try:
            import torch as torch_module
        except ModuleNotFoundError as error:
            raise RuntimeError("restoration fork tripwire requires PyTorch") from error
    assert_fork_parent_cuda_clean(torch_module)
    tripwire = ForkParentCudaTripwire(torch_module)
    tripwire.install()
    _ACTIVE_FORK_PARENT_CUDA_TRIPWIRE = tripwire
    return tripwire


def release_fork_parent_cuda_tripwire(tripwire: ForkParentCudaTripwire) -> None:
    global _ACTIVE_FORK_PARENT_CUDA_TRIPWIRE
    if tripwire is not _ACTIVE_FORK_PARENT_CUDA_TRIPWIRE:
        raise RuntimeError("CUDA parent tripwire identity drifted")
    tripwire.restore_parent()
    _ACTIVE_FORK_PARENT_CUDA_TRIPWIRE = None


def assert_fork_parent_cuda_clean(
    torch_module: Any | None = None,
    *,
    require_tripwire: bool = False,
) -> Mapping[str, bool]:
    """Fail closed unless the process about to fork has untouched CUDA state."""
    if torch_module is None:
        try:
            import torch as torch_module
        except ModuleNotFoundError as error:
            raise RuntimeError("restoration fork guard requires PyTorch") from error
    cuda = getattr(torch_module, "cuda", None)
    is_initialized = getattr(cuda, "is_initialized", None)
    if not callable(is_initialized):
        raise RuntimeError("restoration fork guard requires torch.cuda.is_initialized")
    initialized = is_initialized()
    if type(initialized) is not bool:
        raise RuntimeError("torch.cuda.is_initialized returned a non-boolean value")
    is_in_bad_fork = getattr(cuda, "_is_in_bad_fork", None)
    if is_in_bad_fork is None:
        bad_fork = False
        bad_fork_check_available = False
    else:
        if not callable(is_in_bad_fork):
            raise RuntimeError("torch.cuda._is_in_bad_fork is not callable")
        bad_fork = is_in_bad_fork()
        if type(bad_fork) is not bool:
            raise RuntimeError("torch.cuda._is_in_bad_fork returned a non-boolean value")
        bad_fork_check_available = True
    if initialized or bad_fork:
        raise RuntimeError(
            "restoration POSIX fork requires a CUDA-clean parent process"
        )
    tripwire = _ACTIVE_FORK_PARENT_CUDA_TRIPWIRE
    if require_tripwire:
        if tripwire is None or tripwire.torch_module is not torch_module:
            raise RuntimeError("formal restoration requires the active CUDA tripwire")
        tripwire.assert_active()
    return MappingProxyType(
        {
            "cuda_initialized": False,
            "cuda_bad_fork": False,
            "bad_fork_check_available": bad_fork_check_available,
        }
    )


def _policy_vision_worker_entry(
    queue: Any,
    *,
    worker_id: str,
    device: str,
    expected_gpu_uuid: str,
    work_items: tuple[Confirm20PolicyWorkItem, ...],
    image_payloads: Mapping[str, bytes],
    model_dir: str,
    snapshot_manifest: str,
) -> None:
    try:
        from causalcache.policy.gui_owl_v2_2_vision_runtime import (
            GPU_UUID_TYPE_PROFILE_V2,
            IMAGE_PROCESSOR_SIZE_PROFILE_V3,
            GUIOwlV22VisionFeatureRuntime,
        )

        runtime = GUIOwlV22VisionFeatureRuntime(
            model_dir=model_dir,
            expected_snapshot_manifest=snapshot_manifest,
            device=device,
            expected_gpu_uuid=expected_gpu_uuid,
            gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2,
            image_processor_size_profile=IMAGE_PROCESSOR_SIZE_PROFILE_V3,
        )
        records = []
        for item in work_items:
            requirements = (
                *(dict(item.event_images)[step] for step in CONFIRM_CANDIDATE_EVENT_STEP_IDS),
                item.current_image,
            )
            images = tuple(
                _decode_rgb_image(image_payloads[value.image_member_path])
                for value in requirements
            )
            try:
                result = runtime.score_five_images(
                    images, feature_repeats=POLICY_VISION_FEATURE_REPEATS
                )
            finally:
                for image in images:
                    image.close()
            records.append(
                {
                    "state": {
                        "ordinal": item.ordinal,
                        "source_id": item.source_id,
                        "state_id": item.state_id,
                    },
                    "worker_id": worker_id,
                    "device": device,
                    "gpu_uuid": runtime.metadata["gpu_uuid"],
                    "ordinal": item.ordinal,
                    "source_id": item.source_id,
                    "state_id": item.state_id,
                    "decision_step_id": item.decision_step_id,
                    "candidate_event_step_ids": list(item.candidate_event_step_ids),
                    "scores_by_event_step": [
                        {
                            "event_step_id": step,
                            "score": float(result["scores_by_event_step"][str(step)]),
                        }
                        for step in CONFIRM_CANDIDATE_EVENT_STEP_IDS
                    ],
                    "ranked_event_step_ids": result["ranked_event_step_ids"],
                    "selected_event_step_ids": result["selected_event_step_ids"],
                    "feature_repeats": result["feature_repeats"],
                    "same_device_replay": result["same_device_replay"],
                    "image_grid_thw": result["image_grid_thw"],
                    "merged_token_counts": result["merged_token_counts"],
                    "normalized_embedding_norm_range": result[
                        "normalized_embedding_norm_range"
                    ],
                    "operation_counts": result["operation_counts"],
                }
            )
        queue.put(
            {
                "ok": True,
                "worker_id": worker_id,
                "records": records,
                "metadata": {
                    "worker_id": worker_id,
                    "state_count": len(records),
                    "device": device,
                    "gpu_uuid": runtime.metadata["gpu_uuid"],
                    "runtime": runtime.metadata,
                    "operation_counts": runtime.operation_counts,
                },
            }
        )
    except BaseException as error:
        queue.put(
            {
                "ok": False,
                "worker_id": worker_id,
                "exception_type": error.__class__.__name__,
                "message": str(error),
            }
        )
        raise


def run_four_worker_policy_vision(
    bundle: Confirm20LabelBlindBundle,
    *,
    model_dir: str | Path,
    snapshot_manifest: str | Path,
    devices: Sequence[str],
    gpu_uuids: Sequence[str],
    phase_timeout_seconds: int,
    worker_termination_grace_seconds: int,
    context: Any | None = None,
) -> PolicyVisionPhaseResult:
    assignments = confirm_worker_assignments(bundle.work_items)
    selected_devices, selected_uuids = _validate_device_allocation(
        devices, gpu_uuids, label="policy-vision"
    )
    context = multiprocessing.get_context("spawn") if context is None else context
    result_queue = context.Queue()
    processes = []
    by_ordinal = {item.ordinal: item for item in bundle.work_items}
    for index, assignment in enumerate(assignments):
        items = tuple(by_ordinal[ordinal] for ordinal in assignment.ordinals)
        paths = {
            requirement.image_member_path
            for item in items
            for requirement in (
                *(value for _, value in item.event_images),
                item.current_image,
            )
        }
        process = context.Process(
            target=_policy_vision_worker_entry,
            kwargs={
                "queue": result_queue,
                "worker_id": assignment.worker_id,
                "device": selected_devices[index],
                "expected_gpu_uuid": selected_uuids[index],
                "work_items": items,
                "image_payloads": {
                    path: bundle.image_payloads_by_path[path] for path in paths
                },
                "model_dir": str(Path(model_dir).resolve()),
                "snapshot_manifest": str(Path(snapshot_manifest).resolve()),
            },
            name=f"causalcache-confirm-vision-{assignment.worker_id}",
        )
        processes.append(process)
    for process in processes:
        process.start()
    responses = _collect_process_responses(
        processes=processes,
        queue=result_queue,
        phase_label="policy-vision phase",
        phase_timeout_seconds=phase_timeout_seconds,
        termination_grace_seconds=worker_termination_grace_seconds,
    )
    by_worker = {response["worker_id"]: response for response in responses}
    if set(by_worker) != {assignment.worker_id for assignment in assignments}:
        raise RuntimeError("policy-vision worker response coverage drifted")
    for index, assignment in enumerate(assignments):
        response = by_worker[assignment.worker_id]
        metadata = response.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("worker_id") != assignment.worker_id
            or metadata.get("device") != selected_devices[index]
            or metadata.get("gpu_uuid") != selected_uuids[index]
        ):
            raise RuntimeError("policy-vision worker device response drifted")
    records = [record for response in responses for record in response["records"]]
    records.sort(key=lambda item: int(item["state"]["ordinal"]))
    return PolicyVisionPhaseResult(
        records=tuple(records),
        selections=MappingProxyType(
            {
                record["state"]["state_id"]: tuple(
                    record["selected_event_step_ids"]
                )
                for record in records
            }
        ),
        worker_metadata=tuple(
            by_worker[assignment.worker_id]["metadata"] for assignment in assignments
        ),
    )


def _restoration_worker_entry(
    queue: Any,
    *,
    assignment: ConfirmWorkerAssignment,
    work_items: tuple[Confirm20PolicyWorkItem, ...],
    payloads: ValidatedConfirm20Payloads,
    receipt: PayloadCommitReceipt,
    run_contract_sha256: str,
    model_dir: str,
    snapshot_manifest: str,
    device: str,
    expected_gpu_uuid: str,
) -> None:
    try:
        from causalcache.policy.gui_owl_v2_2_eager_runtime import GUIOwlV22EagerRuntime
        from causalcache.policy.gui_owl_v2_2_vision_runtime import (
            GPU_UUID_TYPE_PROFILE_V2,
            _validated_gpu_identity,
        )
        from causalcache.restoration_v2_gpu_kl import gpu_resident_full_vocab_mean_kl
        from scripts.run_restoration_v2_1_full_45_substrate import (
            GPUFullVocabularyKLBackend,
        )

        runtime = GUIOwlV22EagerRuntime(
            model_dir=model_dir,
            expected_snapshot_manifest=snapshot_manifest,
            device=device,
            target_effective_visual_tokens_per_image=2560,
        )
        gpu_identity = _validated_gpu_identity(
            torch=runtime.torch,
            device=runtime.device,
            expected_gpu_uuid=expected_gpu_uuid,
            gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2,
        )
        backend = GPUFullVocabularyKLBackend(
            torch_module=runtime.torch,
            kl_kernel=gpu_resident_full_vocab_mean_kl,
        )
        message_builder = bind_confirm_message_builder(
            payloads, image_decoder=_decode_rgb_image
        )
        records = tuple(
            run_confirm_state_once(
                work_item=item,
                message_builder=message_builder,
                runtime=runtime,
                distance_backend=backend,
                payload_commit_receipt=receipt,
                run_contract_sha256=run_contract_sha256,
            )
            for item in work_items
        )
        queue.put(
            {
                "ok": True,
                "worker_id": assignment.worker_id,
                "state_records": records,
                "metadata": {
                    "worker_id": assignment.worker_id,
                    "state_count": len(records),
                    "device": device,
                    "gpu_uuid": gpu_identity["gpu_uuid"],
                    "runtime": runtime.metadata,
                },
            }
        )
    except BaseException as error:
        queue.put(
            {
                "ok": False,
                "worker_id": assignment.worker_id,
                "exception_type": error.__class__.__name__,
                "message": str(error),
            }
        )
        raise


def run_four_worker_restoration(
    *,
    assignments: Sequence[ConfirmWorkerAssignment],
    receipt: PayloadCommitReceipt,
    run_contract_sha256: str,
    payloads: ValidatedConfirm20Payloads,
    work_items: Sequence[Confirm20PolicyWorkItem],
    model_dir: str | Path,
    snapshot_manifest: str | Path,
    devices: Sequence[str],
    gpu_uuids: Sequence[str],
    phase_timeout_seconds: int,
    worker_termination_grace_seconds: int,
    context: Any | None = None,
    require_parent_cuda_tripwire: bool = False,
) -> RestorationPhaseResult:
    frozen_assignments = tuple(assignments)
    selected_devices, selected_uuids = _validate_device_allocation(
        devices, gpu_uuids, label="restoration"
    )
    if len(frozen_assignments) != WORKER_COUNT:
        raise ValueError("restoration requires four frozen worker assignments")
    def guard_parent() -> None:
        if require_parent_cuda_tripwire:
            assert_fork_parent_cuda_clean(require_tripwire=True)
        else:
            assert_fork_parent_cuda_clean()

    if context is None:
        if os.name != "posix" or "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("typed payload receipt fan-out requires POSIX fork")
        guard_parent()
        context = multiprocessing.get_context("fork")
    else:
        get_start_method = getattr(context, "get_start_method", None)
        if not callable(get_start_method) or get_start_method() != "fork":
            raise RuntimeError("typed payload receipt fan-out requires a fork context")
        guard_parent()
    result_queue = context.Queue()
    item_by_ordinal = {item.ordinal: item for item in work_items}
    processes = []
    for index, assignment in enumerate(frozen_assignments):
        process = context.Process(
            target=_restoration_worker_entry,
            kwargs={
                "queue": result_queue,
                "assignment": assignment,
                "work_items": tuple(
                    item_by_ordinal[ordinal] for ordinal in assignment.ordinals
                ),
                "payloads": payloads,
                "receipt": receipt,
                "run_contract_sha256": run_contract_sha256,
                "model_dir": str(Path(model_dir).resolve()),
                "snapshot_manifest": str(Path(snapshot_manifest).resolve()),
                "device": selected_devices[index],
                "expected_gpu_uuid": selected_uuids[index],
            },
            name=f"causalcache-confirm-restoration-{assignment.worker_id}",
        )
        processes.append(process)
    for process in processes:
        process.start()
    responses = _collect_process_responses(
        processes=processes,
        queue=result_queue,
        phase_label="restoration phase",
        phase_timeout_seconds=phase_timeout_seconds,
        termination_grace_seconds=worker_termination_grace_seconds,
    )
    by_worker = {response["worker_id"]: response for response in responses}
    if set(by_worker) != {
        assignment.worker_id for assignment in frozen_assignments
    }:
        raise RuntimeError("restoration worker response coverage drifted")
    for index, assignment in enumerate(frozen_assignments):
        response = by_worker[assignment.worker_id]
        metadata = response.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("worker_id") != assignment.worker_id
            or metadata.get("device") != selected_devices[index]
            or metadata.get("gpu_uuid") != selected_uuids[index]
        ):
            raise RuntimeError("restoration worker device response drifted")
    return RestorationPhaseResult(
        worker_results=tuple(
            ConfirmWorkerResult(
                worker_id=assignment.worker_id,
                state_records=tuple(
                    by_worker[assignment.worker_id]["state_records"]
                ),
            )
            for assignment in frozen_assignments
        ),
        worker_metadata=tuple(
            by_worker[assignment.worker_id]["metadata"]
            for assignment in frozen_assignments
        ),
    )


def make_real_restoration_phase(
    *,
    payloads: ValidatedConfirm20Payloads,
    bundle: Confirm20LabelBlindBundle,
    model_dir: str | Path,
    snapshot_manifest: str | Path,
    devices: Sequence[str],
    gpu_uuids: Sequence[str],
    phase_timeout_seconds: int,
    worker_termination_grace_seconds: int,
    require_parent_cuda_tripwire: bool = False,
) -> RestorationPhase:
    def run(
        assignments: tuple[ConfirmWorkerAssignment, ...],
        receipt: PayloadCommitReceipt,
        run_contract_sha256: str,
    ) -> RestorationPhaseResult:
        return run_four_worker_restoration(
            assignments=assignments,
            receipt=receipt,
            run_contract_sha256=run_contract_sha256,
            payloads=payloads,
            work_items=bundle.work_items,
            model_dir=model_dir,
            snapshot_manifest=snapshot_manifest,
            devices=devices,
            gpu_uuids=gpu_uuids,
            phase_timeout_seconds=phase_timeout_seconds,
            worker_termination_grace_seconds=worker_termination_grace_seconds,
            require_parent_cuda_tripwire=require_parent_cuda_tripwire,
        )

    return run


def make_real_policy_vision_phase(
    *,
    model_dir: str | Path,
    snapshot_manifest: str | Path,
    devices: Sequence[str],
    gpu_uuids: Sequence[str],
    phase_timeout_seconds: int,
    worker_termination_grace_seconds: int,
) -> PolicyVisionPhase:
    return lambda bundle: run_four_worker_policy_vision(
        bundle,
        model_dir=model_dir,
        snapshot_manifest=snapshot_manifest,
        devices=devices,
        gpu_uuids=gpu_uuids,
        phase_timeout_seconds=phase_timeout_seconds,
        worker_termination_grace_seconds=worker_termination_grace_seconds,
    )


def bundle_from_validated_payloads(
    payloads: ValidatedConfirm20Payloads,
) -> Confirm20LabelBlindBundle:
    return build_confirm20_label_blind_bundle(payloads)


__all__ = [
    "ATTEMPT_FILENAME",
    "COMPLETION_FILENAME",
    "ConfirmExecutionResult",
    "ConfirmPublisher",
    "DurableConfirmOutput",
    "ForkParentCudaTripwire",
    "HuggingFaceConfirmPublisher",
    "PolicyVisionPhaseResult",
    "RestorationPhaseResult",
    "assert_fork_parent_cuda_clean",
    "build_canonical_label_blind_payload",
    "bundle_from_validated_payloads",
    "download_and_load_independent_ensemble",
    "execute_independent_confirm",
    "install_fork_parent_cuda_tripwire",
    "make_real_policy_vision_phase",
    "make_real_restoration_phase",
    "read_hf_token_file",
    "release_fork_parent_cuda_tripwire",
    "run_four_worker_policy_vision",
    "run_four_worker_restoration",
    "validate_policy_vision_phase",
    "validate_restoration_phase",
]
