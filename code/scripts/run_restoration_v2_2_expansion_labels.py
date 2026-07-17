"""Run the frozen two-H200 restoration-v2.2 expansion exact-label attempt."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causalcache.data.restoration_v2_2_expansion_label_inputs import (
    build_v2_2_expansion_label_messages,
    expansion_label_prompt_inventory,
    make_v2_2_expansion_label_prompt_spec,
)
from causalcache.data.restoration_v2_2_expansion_label_parent import (
    V22ExpansionLabelParentState,
    extract_expansion_label_parent_states,
)
from causalcache.policy.gui_owl_v2_1 import (
    serialize_gui_owl_v2_1_teacher_target,
)
from causalcache.policy.gui_owl_v2_vision import verify_frozen_vision_runtime
from causalcache.restoration_v2_2_expansion_labels_artifact import (
    AGGREGATE_FILENAME,
    ATTEMPT_DIRECTORY,
    ATTEMPT_ID,
    CANONICAL_ARCHIVE_PATH,
    CANONICAL_DERIVED_ARTIFACT,
    CANONICAL_LEDGER_PATH,
    CANONICAL_OUTPUT_DIR,
    CANONICAL_PARENT_SUBSTRATE,
    EXECUTION_EVIDENCE_FILENAME,
    EXPECTED_STATE_COUNT,
    EXPECTED_TOTALS,
    GLOBAL_LEDGER_ARCHIVE_NAME,
    LOG_PATHS,
    MAXIMUM_REPEAT_KL,
    MONITOR_SUMMARY_FILENAME,
    PASS_OUTCOME,
    PROHIBITED_OPERATION_COUNTS,
    PROTOCOL_ID,
    RUN_MANIFEST_FILENAME,
    SCHEMA_VERSION,
    STATE_DIRECTORY,
    STATE_STATUS,
    WORKER_DIRECTORY,
    WORKER_LEDGER_FILENAME,
    WORKER_RUNTIME_FILENAME,
    WORKER_SIBLING_LEDGER_DIRECTORY,
    WORKER_STATUS,
    WORKER_TERMINAL_FILENAME,
    WorkerSpec,
    aggregate_from_reduction,
    canonical_json_bytes,
    expected_worker_specs,
    load_and_validate_runner_freeze,
    pretty_json_bytes,
    reduce_raw_distance_states,
    runner_interface_contract,
    sha256_bytes,
    sha256_file,
    validate_execution_run_contract,
    validate_expansion_label_evidence_files,
    validate_raw_state_record,
)
from causalcache.restoration_v2_2_expansion_labels_contract import (
    CANONICAL_CONFIG_PATH,
    DERIVED_HF_REVISION,
    load_and_validate_contract,
)
from causalcache.restoration_v2_2_expansion_substrate_artifact import (
    read_expansion_substrate_archive,
)
from scripts.run_restoration_v2_1_full_45_substrate import _decode_rgb_image
from scripts.run_restoration_v2_2_expansion_substrate import (
    CANONICAL_IMAGE_DIGEST,
    CANONICAL_MODEL_DIR,
    CANONICAL_OCR_BACKEND_CONFIG_PATH,
    CANONICAL_SNAPSHOT_MANIFEST_PATH,
    ExpansionArtifact,
    load_immutable_expansion_artifact,
    load_worker_runtime,
    validate_worker_runtime_metadata as validate_eager_worker_runtime_metadata,
    validate_worker_runtime_pair as validate_eager_worker_runtime_pair,
)


RUNNER_SOURCE_PATH = "code/scripts/run_restoration_v2_2_expansion_labels.py"
SUBSTRATE_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_expansion_substrate_v1.json"
)
GLOBAL_CLAIM_STATUS = "CLAIMED_EXPANSION_EXACT_LABEL_ATTEMPT_BEFORE_RUNTIME_IMPORT"
GLOBAL_INVALID_STATUS = "INVALID_EXPANSION_EXACT_LABEL_ATTEMPT"
WORKER_CLAIM_STATUS = "CLAIMED_EXPANSION_EXACT_LABEL_WORKER"
WORKER_INVALID_STATUS = "INVALID_EXPANSION_EXACT_LABEL_WORKER"
RUNTIME_BARRIER_TIMEOUT_SECONDS = 900.0
MONITOR_SIDECAR_SUBCOMMAND = "monitor-sidecar"
MONITOR_SAMPLE_INTERVAL_SECONDS = 1.0
MONITOR_READY_STATUS = "EXPANSION_EXACT_LABEL_MONITOR_READY"
MONITOR_STOP_STATUS = "EXPANSION_EXACT_LABEL_MONITOR_STOP_REQUEST"
MONITOR_SUMMARY_STATUS = "GPU_UTILIZATION_MONITOR_SUMMARY"
MONITOR_PRECLAIM_ABORT_STOP_STATUS = (
    "EXPANSION_EXACT_LABEL_MONITOR_PRECLAIM_ABORT_STOP_REQUEST"
)
MONITOR_PRECLAIM_ABORT_SUMMARY_STATUS = (
    "GPU_UTILIZATION_MONITOR_PRECLAIM_ABORT_SUMMARY"
)
MONITOR_PRECLAIM_ABORT_REASON = "PRECLAIM_FAILED_BEFORE_FORMAL_RUN_CONTRACT"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

@dataclass(frozen=True)
class AuthorizedExpansionLabels:
    repository_root: Path
    frozen_config: Mapping[str, Any]
    contract_validation: Mapping[str, Any]
    runner_freeze: Mapping[str, Any]
    runner_validation: Mapping[str, Any]
    parent_evidence: Any
    artifact: ExpansionArtifact
    parents: tuple[V22ExpansionLabelParentState, ...]
    snapshot_manifest_path: Path
    state_projections: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ExpansionLabelLayout:
    root: Path
    global_ledger: Path
    publish_staging: Path
    run_contract: Mapping[str, Any]
    run_contract_sha256: str
    worker_sibling_ledgers: Mapping[str, Path]
    started_at_utc: str


@dataclass(frozen=True)
class WorkerRuntime:
    runtime: Any
    distance_backend: Any
    runtime_metadata: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _strict_json(path: str | Path) -> dict[str, Any]:
    payload = Path(path).read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict) or pretty_json_bytes(value) != payload:
        raise ValueError(f"{path} must contain canonical pretty JSON")
    return value


def _repo_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required repository file is missing: {relative}")
    return path


def _file_witness(path: Path, *, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json_exclusive(
    path: Path,
    value: Mapping[str, Any],
    *,
    staging: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / f"{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as destination:
            destination.write(pretty_json_bytes(dict(value)))
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_json_durable(
    path: Path,
    value: Mapping[str, Any],
    *,
    staging: Path,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"durable evidence disappeared: {path}")
    temporary = staging / f"{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as destination:
            destination.write(pretty_json_bytes(dict(value)))
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_exclusive(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise ValueError(f"external evidence is missing or empty: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())
    if destination.read_bytes() != source.read_bytes():
        raise RuntimeError("external evidence copy differs byte-for-byte")


def _append_execution_log(layout: ExpansionLabelLayout, message: str) -> None:
    path = layout.root / LOG_PATHS["execution"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as destination:
        destination.write(f"{_utc_now()} {message}\n".encode("utf-8"))
        destination.flush()
        os.fsync(destination.fileno())


def _state_projection(
    parent: V22ExpansionLabelParentState,
    *,
    artifact: ExpansionArtifact,
    image_decoder: Callable[[bytes], Any] = _decode_rgb_image,
) -> dict[str, Any]:
    coalition_inputs: list[dict[str, Any]] = []
    for coalition in _coalitions(parent.candidate_event_step_ids):
        spec = make_v2_2_expansion_label_prompt_spec(
            artifact,
            parent,
            restored_event_step_ids=coalition,
            budget_event_capacity=2,
        )
        bundle = build_v2_2_expansion_label_messages(
            artifact,
            spec,
            image_decoder=image_decoder,
        )
        coalition_inputs.append(
            {
                "coalition": list(coalition),
                "input_sha256": _input_sha256(spec, bundle),
            }
        )
    teacher_target = serialize_gui_owl_v2_1_teacher_target(parent.canonical_action)
    return {
        "state_index": parent.index,
        "role": parent.role,
        "source_id": parent.source_id,
        "decision_step_id": parent.decision_step_id,
        "state_id": parent.state_id,
        "history_event_step_ids": list(parent.history_event_step_ids),
        "candidate_event_step_ids": list(parent.candidate_event_step_ids),
        "current_equivalent_event_step_id": (
            parent.current_equivalent_event_step_id
        ),
        "parent_member_name": parent.member_name,
        "parent_member_sha256": parent.member_sha256,
        "canonical_action_sha256": parent.canonical_action_sha256,
        "teacher_target_sha256": sha256_bytes(teacher_target.encode("utf-8")),
        "request_manifest_sha256": parent.request_manifest_sha256,
        "slice_witness_sha256": parent.slice_witness_sha256,
        "immutable_artifact_tree_sha256": (
            parent.immutable_artifact_tree_sha256
        ),
        "coalition_inputs": coalition_inputs,
    }


def _coalitions(event_ids: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        coalition
        for size in range(len(event_ids) + 1)
        for coalition in itertools.combinations(event_ids, size)
    )


def state_operation_counts(event_count: int) -> dict[str, int]:
    if event_count not in {2, 3, 4}:
        raise ValueError("expansion exact labels require n=2/3/4 events")
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


def _input_sha256(spec: Any, bundle: Any) -> str:
    inventory = expansion_label_prompt_inventory(spec, bundle)
    return sha256_bytes(
        canonical_json_bytes(
            {
                "prompt_inventory": inventory,
                "immutable_artifact_tree_sha256": (
                    spec.parent.immutable_artifact_tree_sha256
                ),
            }
        )
    )


def _measurement_value(measurement: Any) -> float:
    audit = getattr(measurement, "audit", None)
    if (
        not isinstance(audit, Mapping)
        or audit.get("full_tensor_host_transfers") != 0
        or audit.get("validation_scalar_host_reads") != 0
    ):
        raise RuntimeError(
            "GPU KL measurement did not prove zero validation-scalar and "
            "full-tensor host transfers"
        )
    value = float(getattr(measurement, "value", math.nan))
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("GPU full-vocabulary KL is nonfinite or negative")
    return value


def run_expansion_label_state_once(
    *,
    artifact: ExpansionArtifact,
    parent: V22ExpansionLabelParentState,
    worker: WorkerSpec,
    runtime: Any,
    distance_backend: Any,
    run_contract_sha256: str,
    expected_state: Mapping[str, Any] | None = None,
    image_decoder: Callable[[bytes], Any] = _decode_rgb_image,
) -> dict[str, Any]:
    """Produce only the canonical raw D(S) record for one expansion state."""
    if worker.index_parity != parent.index % 2 or parent.index not in worker.state_indices:
        raise ValueError("state is owned by the wrong fixed-parity worker")
    if not isinstance(run_contract_sha256, str) or SHA256_PATTERN.fullmatch(
        run_contract_sha256
    ) is None:
        raise ValueError("run contract SHA256 is invalid")
    event_ids = parent.candidate_event_step_ids
    full = tuple(event_ids)
    state_projection = (
        _state_projection(parent, artifact=artifact, image_decoder=image_decoder)
        if expected_state is None
        else dict(expected_state)
    )
    teacher_target = serialize_gui_owl_v2_1_teacher_target(parent.canonical_action)
    teacher_target_sha = sha256_bytes(teacher_target.encode("utf-8"))
    parent_witness = {
        "state_index": parent.index,
        "role": parent.role,
        "source_id": parent.source_id,
        "decision_step_id": parent.decision_step_id,
        "state_id": parent.state_id,
        "history_event_step_ids": list(parent.history_event_step_ids),
        "candidate_event_step_ids": list(parent.candidate_event_step_ids),
        "current_equivalent_event_step_id": (
            parent.current_equivalent_event_step_id
        ),
        "parent_member_name": parent.member_name,
        "parent_member_sha256": parent.member_sha256,
        "canonical_action_sha256": parent.canonical_action_sha256,
        "teacher_target_sha256": teacher_target_sha,
        "request_manifest_sha256": parent.request_manifest_sha256,
        "slice_witness_sha256": parent.slice_witness_sha256,
        "immutable_artifact_tree_sha256": (
            parent.immutable_artifact_tree_sha256
        ),
    }
    if any(state_projection.get(key) != value for key, value in parent_witness.items()):
        raise ValueError("run-contract state witness differs from the parent")
    coalition_input_witnesses = {
        tuple(item["coalition"]): item["input_sha256"]
        for item in state_projection.get("coalition_inputs", ())
    }
    if tuple(coalition_input_witnesses) != _coalitions(event_ids):
        raise ValueError("run-contract coalition input witness inventory drifted")
    full_bundle = None
    reference_input_sha = None
    for coalition in _coalitions(event_ids):
        spec = make_v2_2_expansion_label_prompt_spec(
            artifact,
            parent,
            restored_event_step_ids=coalition,
            budget_event_capacity=2,
        )
        bundle = build_v2_2_expansion_label_messages(
            artifact,
            spec,
            image_decoder=image_decoder,
        )
        input_sha = _input_sha256(spec, bundle)
        if input_sha != coalition_input_witnesses[coalition]:
            raise ValueError("candidate input differs from the preclaim witness")
        if coalition == full:
            full_bundle = bundle
            reference_input_sha = input_sha
    if full_bundle is None or reference_input_sha is None:
        raise RuntimeError("full-reference prompt was not prepared")

    reference_logits, reference_metadata = runtime.teacher_forced_distance_logits(
        (full_bundle.messages,),
        (parent.canonical_action,),
    )
    reference_log_probs = distance_backend.prepare_reference(reference_logits)
    if (
        getattr(reference_log_probs, "ndim", None) != 3
        or int(reference_log_probs.shape[0]) != 1
        or str(reference_log_probs.dtype) != "torch.float32"
        or getattr(getattr(reference_log_probs, "device", None), "type", None)
        != "cuda"
    ):
        raise RuntimeError("reference action log-probability metadata drifted")
    reference_log_probs_shape = [
        int(reference_log_probs.shape[1]),
        int(reference_log_probs.shape[2]),
    ]
    reference_log_probs_dtype = str(reference_log_probs.dtype)
    reference_log_probs_device_type = str(reference_log_probs.device.type)
    del reference_logits

    repeat_logits, repeat_metadata = runtime.teacher_forced_distance_logits(
        (full_bundle.messages,),
        (parent.canonical_action,),
    )
    repeat_measurement = distance_backend.measure(reference_log_probs, repeat_logits)
    del repeat_logits
    repeat_kl = _measurement_value(repeat_measurement)
    if repeat_kl > MAXIMUM_REPEAT_KL:
        raise RuntimeError("reference repeat KL exceeds the frozen 1e-4 ceiling")

    sample_records = reference_metadata.get("samples")
    repeat_samples = repeat_metadata.get("samples")
    if (
        not isinstance(sample_records, list)
        or len(sample_records) != 1
        or not isinstance(repeat_samples, list)
        or len(repeat_samples) != 1
        or type(sample_records[0].get("distance_action_tokens")) is not int
        or sample_records[0]["distance_action_tokens"] <= 0
        or repeat_samples[0].get("distance_action_tokens")
        != sample_records[0]["distance_action_tokens"]
        or type(reference_metadata.get("vocabulary_size")) is not int
        or reference_metadata["vocabulary_size"] <= 0
        or repeat_metadata.get("vocabulary_size")
        != reference_metadata.get("vocabulary_size")
        or reference_metadata.get("full_logit_tensor_host_transfers") != 0
        or repeat_metadata.get("full_logit_tensor_host_transfers") != 0
    ):
        raise RuntimeError("teacher metadata action span or vocabulary drifted")

    rows: list[dict[str, Any]] = []
    for coalition in _coalitions(event_ids):
        if coalition == full:
            rows.append(
                {
                    "coalition": list(coalition),
                    "distance": 0.0,
                    "candidate_input_sha256": reference_input_sha,
                    "teacher_forward_count": 0,
                    "kl_measurement_count": 0,
                    "scalar_host_transfer_count": 0,
                    "is_full_history_reference": True,
                    "full_logit_tensor_host_transfer_count": 0,
                }
            )
            continue
        spec = make_v2_2_expansion_label_prompt_spec(
            artifact,
            parent,
            restored_event_step_ids=coalition,
            budget_event_capacity=2,
        )
        bundle = build_v2_2_expansion_label_messages(
            artifact,
            spec,
            image_decoder=image_decoder,
        )
        candidate_input_sha = _input_sha256(spec, bundle)
        if candidate_input_sha != coalition_input_witnesses[coalition]:
            raise RuntimeError("candidate input changed after preforward validation")
        candidate_logits, candidate_metadata = runtime.teacher_forced_distance_logits(
            (bundle.messages,),
            (parent.canonical_action,),
        )
        if (
            candidate_metadata.get("vocabulary_size")
            != reference_metadata["vocabulary_size"]
            or not isinstance(candidate_metadata.get("samples"), list)
            or len(candidate_metadata["samples"]) != 1
            or candidate_metadata["samples"][0].get("distance_action_tokens")
            != sample_records[0]["distance_action_tokens"]
            or candidate_metadata.get("full_logit_tensor_host_transfers") != 0
        ):
            raise RuntimeError("candidate teacher metadata drifted or copied logits")
        measurement = distance_backend.measure(reference_log_probs, candidate_logits)
        del candidate_logits
        rows.append(
            {
                "coalition": list(coalition),
                "distance": _measurement_value(measurement),
                "candidate_input_sha256": candidate_input_sha,
                "teacher_forward_count": 1,
                "kl_measurement_count": 1,
                "scalar_host_transfer_count": 1,
                "is_full_history_reference": False,
                "full_logit_tensor_host_transfer_count": 0,
            }
        )
    del reference_log_probs

    record = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": STATE_STATUS,
        "run_contract_sha256": run_contract_sha256,
        "worker": worker.envelope_identity(),
        "state": state_projection,
        "reference_teacher": {
            "canonical_action_sha256": parent.canonical_action_sha256,
            "teacher_target_sha256": teacher_target_sha,
            "reference_input_sha256": reference_input_sha,
            "reference_repeat_kl": repeat_kl,
            "repeat_scalar_host_transfer_count": 1,
            "action_token_count": sample_records[0]["distance_action_tokens"],
            "vocabulary_size": reference_metadata["vocabulary_size"],
            "action_log_probs_shape": reference_log_probs_shape,
            "action_log_probs_dtype": reference_log_probs_dtype,
            "action_log_probs_device_type": reference_log_probs_device_type,
            "full_logit_tensor_host_transfer_count": 0,
        },
        "distance_rows": rows,
        "operation_counts": state_operation_counts(len(event_ids)),
    }
    validate_raw_state_record(
        record,
        expected_state=record["state"],
        expected_worker=worker,
        run_contract_sha256=run_contract_sha256,
    )
    return record


def _canonical_execution_argv(args: argparse.Namespace) -> list[str]:
    values = {
        "--repository-root": args.repository_root,
        "--contract": args.contract,
        "--runner-freeze": args.runner_freeze,
        "--parent-substrate-archive": args.parent_substrate_archive,
        "--derived-artifact-root": args.derived_artifact_root,
        "--ocr-backend-config": args.ocr_backend_config,
        "--snapshot-manifest": args.snapshot_manifest,
        "--model-dir": args.model_dir,
        "--host-alias": args.host_alias,
        "--host-hostname": args.host_hostname,
        "--container-id": args.container_id,
        "--container-image-digest": args.container_image_digest,
        "--output-dir": args.output_dir,
        "--global-ledger": args.global_ledger,
        "--preflight-log": args.preflight_log,
        "--utilization-monitor-log": args.utilization_monitor_log,
        "--monitor-ready-file": args.monitor_ready_file,
        "--monitor-summary": args.monitor_summary,
    }
    result: list[str] = []
    for option in runner_interface_contract()["required_cli_options"]:
        result.extend((option, str(values[option])))
    return result


def _validate_formal_args(args: argparse.Namespace, root: Path) -> None:
    expected_hosts = {
        "hyper00": "node-radixark-16-0000",
        "hyper01": "node-radixark-16-0001",
    }
    expected_paths = {
        "contract": root / CANONICAL_CONFIG_PATH,
        "ocr_backend_config": root / CANONICAL_OCR_BACKEND_CONFIG_PATH,
        "snapshot_manifest": root / CANONICAL_SNAPSHOT_MANIFEST_PATH,
    }
    if any(
        Path(getattr(args, field)).resolve() != expected.resolve()
        for field, expected in expected_paths.items()
    ):
        raise ValueError("formal execution repository input path drifted")
    if (
        Path(args.output_dir).resolve() != CANONICAL_OUTPUT_DIR
        or Path(args.global_ledger).resolve() != CANONICAL_LEDGER_PATH
        or Path(args.model_dir).resolve() != CANONICAL_MODEL_DIR
        or args.container_image_digest != CANONICAL_IMAGE_DIGEST
        or args.host_alias not in expected_hosts
        or args.host_hostname != expected_hosts.get(args.host_alias)
        or re.fullmatch(r"[0-9a-f]{64}", args.container_id) is None
    ):
        raise ValueError("canonical attempt, model, Hyper host, or container drifted")
    supplied_argv = list(getattr(args, "execution_argv", _canonical_execution_argv(args)))
    if supplied_argv != _canonical_execution_argv(args):
        raise ValueError("formal execution argv order or value drifted")


def authorize_expansion_label_run(args: argparse.Namespace) -> AuthorizedExpansionLabels:
    """Validate every immutable input before the global attempt is claimed."""
    root = Path(args.repository_root).resolve()
    _validate_formal_args(args, root)
    frozen_config, contract_validation = load_and_validate_contract(
        args.contract,
        repository_root=root,
    )
    runner_freeze, runner_validation = load_and_validate_runner_freeze(
        args.runner_freeze,
        repository_root=root,
    )
    if runner_validation.get("policy_or_gpu_execution_authorized") is not True:
        raise PermissionError("committed runner freeze did not authorize execution")

    parent_path = Path(args.parent_substrate_archive).resolve()
    if (
        parent_path.is_symlink()
        or not parent_path.is_file()
        or sha256_file(parent_path) != CANONICAL_PARENT_SUBSTRATE["archive_sha256"]
        or parent_path.stat().st_size != CANONICAL_PARENT_SUBSTRATE["archive_size_bytes"]
    ):
        raise PermissionError("immutable parent substrate archive bytes drifted")
    parent_evidence = read_expansion_substrate_archive(parent_path)
    if (
        parent_evidence.outcome != CANONICAL_PARENT_SUBSTRATE["outcome"]
        or parent_evidence.completed_state_count
        != CANONICAL_PARENT_SUBSTRATE["validated_state_count"]
        or parent_evidence.tree_inventory_sha256
        != CANONICAL_PARENT_SUBSTRATE["tree_inventory_sha256"]
    ):
        raise PermissionError("parent substrate scientific identity drifted")

    substrate_config = _strict_json(_repo_file(root, SUBSTRATE_CONFIG_PATH))
    artifact = load_immutable_expansion_artifact(
        artifact_root=args.derived_artifact_root,
        frozen_config=substrate_config,
        backend_config_path=args.ocr_backend_config,
        derived_repo=frozen_config["immutable_inputs"]["derived_artifact"]["repo"],
        derived_revision=DERIVED_HF_REVISION,
    )
    parents = extract_expansion_label_parent_states(parent_evidence, artifact)
    if len(parents) != EXPECTED_STATE_COUNT:
        raise ValueError("parent/derived crosswalk is not the fixed 192 states")

    snapshot = _repo_file(root, CANONICAL_SNAPSHOT_MANIFEST_PATH)
    model_identity = verify_frozen_vision_runtime(
        model_dir=Path(args.model_dir).resolve(),
        expected_snapshot_manifest=snapshot,
    )
    if getattr(model_identity, "model_dir", None) != str(CANONICAL_MODEL_DIR):
        raise ValueError("full model snapshot preclaim identity drifted")
    projections = tuple(
        _state_projection(parent, artifact=artifact) for parent in parents
    )
    return AuthorizedExpansionLabels(
        repository_root=root,
        frozen_config=frozen_config,
        contract_validation=contract_validation,
        runner_freeze=runner_freeze,
        runner_validation=runner_validation,
        parent_evidence=parent_evidence,
        artifact=artifact,
        parents=parents,
        snapshot_manifest_path=snapshot,
        state_projections=projections,
    )


def build_run_contract(
    args: argparse.Namespace,
    authorized: AuthorizedExpansionLabels,
) -> dict[str, Any]:
    config_path = _repo_file(authorized.repository_root, CANONICAL_CONFIG_PATH)
    freeze_path = Path(args.runner_freeze).resolve()
    run_contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "AUTHORIZED_EXPANSION_EXACT_LABEL_EXECUTION",
        "runner_source_git_commit": authorized.runner_validation[
            "runner_source_git_commit"
        ],
        "execution_git_commit": authorized.runner_validation["execution_git_commit"],
        "config": _file_witness(
            config_path,
            relative=CANONICAL_CONFIG_PATH,
        ),
        "runner_freeze": {
            **_file_witness(
                freeze_path,
                relative=str(
                    Path(args.runner_freeze)
                    .resolve()
                    .relative_to(authorized.repository_root)
                ),
            ),
            "runner_source_git_commit": authorized.runner_validation[
                "runner_source_git_commit"
            ],
        },
        "parent_substrate": dict(CANONICAL_PARENT_SUBSTRATE),
        "derived_artifact": dict(CANONICAL_DERIVED_ARTIFACT),
        "states": [dict(state) for state in authorized.state_projections],
        "worker_topology": [spec.to_dict() for spec in expected_worker_specs()],
        "planned_counts": dict(EXPECTED_TOTALS),
        "prohibited_operation_counts": dict(PROHIBITED_OPERATION_COUNTS),
        "attempt_identity": {
            "attempt_id": ATTEMPT_ID,
            "output_dir": str(CANONICAL_OUTPUT_DIR),
            "global_ledger_path": str(CANONICAL_LEDGER_PATH),
            "archive_path": str(CANONICAL_ARCHIVE_PATH),
        },
        "execution_argv": _canonical_execution_argv(args),
    }
    validate_execution_run_contract(
        run_contract,
        expected_source_git_commit=authorized.runner_validation[
            "runner_source_git_commit"
        ],
        expected_config_sha256=authorized.contract_validation["config_sha256"],
    )
    return run_contract


def _worker_sibling_paths(root: Path) -> dict[str, Path]:
    return {
        spec.worker_id: root.parent / f".{ATTEMPT_ID}.{spec.worker_id}.attempt.json"
        for spec in expected_worker_specs()
    }


def _global_ledger_record(
    layout: ExpansionLabelLayout,
    *,
    status: str,
    attempted: Sequence[int],
    completed: Sequence[int],
    ended_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "attempt_id": ATTEMPT_ID,
        "run_contract_sha256": layout.run_contract_sha256,
        "output_dir": str(layout.root),
        "worker_sibling_ledgers": {
            key: str(value) for key, value in layout.worker_sibling_ledgers.items()
        },
        "attempted_state_indices": list(attempted),
        "completed_state_indices": list(completed),
        "retry_count": 0,
        "top_up_count": 0,
        "started_at_utc": layout.started_at_utc,
        "ended_at_utc": ended_at_utc,
    }


def _seal_claim_attempt_failure(
    layout: ExpansionLabelLayout,
    *,
    error: BaseException,
) -> dict[str, Any]:
    try:
        layout.publish_staging.mkdir(parents=True, exist_ok=True)
        tombstone_staging = layout.publish_staging
    except Exception:
        if layout.root.is_symlink() or not layout.root.is_dir():
            raise
        tombstone_staging = layout.root
    claim_ledger_was_present = (
        layout.global_ledger.is_file() and not layout.global_ledger.is_symlink()
    )
    claimed_sha = (
        sha256_file(layout.global_ledger) if claim_ledger_was_present else None
    )
    record = _global_ledger_record(
        layout,
        status=GLOBAL_INVALID_STATUS,
        attempted=(),
        completed=(),
        ended_at_utc=_utc_now(),
    )
    record.update(
        {
            "failure": {
                "stage": "claim_attempt",
                "exception_type": error.__class__.__name__,
                "message": str(error),
            },
            "claim_ledger_was_present": claim_ledger_was_present,
            "claimed_ledger_sha256": claimed_sha,
        }
    )
    if layout.global_ledger.exists() or layout.global_ledger.is_symlink():
        if not claim_ledger_was_present:
            raise RuntimeError("claim ledger path became non-regular during sealing")
        _replace_json_durable(
            layout.global_ledger,
            record,
            staging=tombstone_staging,
        )
    else:
        _write_json_exclusive(
            layout.global_ledger,
            record,
            staging=tombstone_staging,
        )
    archived = layout.root / GLOBAL_LEDGER_ARCHIVE_NAME
    if archived.exists() or archived.is_symlink():
        _replace_json_durable(archived, record, staging=tombstone_staging)
    else:
        _write_json_exclusive(archived, record, staging=tombstone_staging)
    return record


def claim_attempt(
    args: argparse.Namespace,
    run_contract: Mapping[str, Any],
    *,
    require_canonical: bool = True,
) -> ExpansionLabelLayout:
    root = Path(args.output_dir).resolve()
    global_ledger = Path(args.global_ledger).resolve()
    if require_canonical and (
        root != CANONICAL_OUTPUT_DIR or global_ledger != CANONICAL_LEDGER_PATH
    ):
        raise ValueError("formal claim requires the canonical attempt identity")
    siblings = _worker_sibling_paths(root)
    staging = root.parent / f".{ATTEMPT_ID}.publication-staging"
    forbidden = [root, global_ledger, staging, *siblings.values()]
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise FileExistsError("attempt, ledger, sibling, or staging already exists")
    run_sha = sha256_bytes(canonical_json_bytes(run_contract))
    started = _utc_now()
    layout = ExpansionLabelLayout(
        root=root,
        global_ledger=global_ledger,
        publish_staging=staging,
        run_contract=dict(run_contract),
        run_contract_sha256=run_sha,
        worker_sibling_ledgers=siblings,
        started_at_utc=started,
    )
    created_attempt_resource = False
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()
        created_attempt_resource = True
        staging.mkdir()
        _write_json_exclusive(
            root / RUN_MANIFEST_FILENAME,
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": "IMMUTABLE_EXPANSION_EXACT_LABEL_RUN_MANIFEST",
                "run_contract": dict(run_contract),
                "run_contract_sha256": run_sha,
            },
            staging=staging,
        )
        initial = _global_ledger_record(
            layout,
            status=GLOBAL_CLAIM_STATUS,
            attempted=(),
            completed=(),
            ended_at_utc=started,
        )
        _write_json_exclusive(global_ledger, initial, staging=staging)
        _copy_exclusive(
            Path(args.preflight_log).resolve(),
            root / LOG_PATHS["preflight"],
        )
        _append_execution_log(layout, "global attempt claimed before runtime import")
        return layout
    except Exception as error:
        created_attempt_resource = created_attempt_resource or any(
            path.exists() or path.is_symlink()
            for path in (root, staging, global_ledger)
        )
        if created_attempt_resource:
            try:
                _seal_claim_attempt_failure(layout, error=error)
            except Exception as sealing_error:
                error.add_note(f"claim tombstone sealing also failed: {sealing_error}")
        raise


def _worker_ledger_record(
    layout: ExpansionLabelLayout,
    spec: WorkerSpec,
    *,
    status: str,
    attempted: Sequence[int],
    completed: Sequence[int],
    started_at_utc: str,
    ended_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "run_contract_sha256": layout.run_contract_sha256,
        "worker": spec.to_dict(),
        "attempted_state_indices": list(attempted),
        "completed_state_indices": list(completed),
        "retry_count": 0,
        "top_up_count": 0,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
    }


def _worker_ledger_paths(
    layout: ExpansionLabelLayout,
    spec: WorkerSpec,
) -> tuple[Path, Path, Path]:
    return (
        layout.root
        / WORKER_DIRECTORY
        / spec.worker_id
        / WORKER_LEDGER_FILENAME,
        layout.root / WORKER_SIBLING_LEDGER_DIRECTORY / f"{spec.worker_id}.json",
        layout.worker_sibling_ledgers[spec.worker_id],
    )


def _publish_worker_ledgers(
    layout: ExpansionLabelLayout,
    spec: WorkerSpec,
    value: Mapping[str, Any],
    *,
    first: bool,
) -> None:
    paths = _worker_ledger_paths(layout, spec)
    for path in paths:
        if first:
            _write_json_exclusive(path, value, staging=layout.publish_staging)
        else:
            _replace_json_durable(path, value, staging=layout.publish_staging)
    payloads = [path.read_bytes() for path in paths]
    if len(set(payloads)) != 1:
        raise RuntimeError("worker root, archived sibling, and external ledgers differ")


def _visible_gpu_uuids(torch_module: Any) -> tuple[str, str]:
    if torch_module.cuda.device_count() != 2:
        raise RuntimeError("exact-label container must expose exactly two GPUs")
    values: list[str] = []
    for index in range(2):
        raw = getattr(torch_module.cuda.get_device_properties(index), "uuid", None)
        if isinstance(raw, bytes):
            value = raw.decode("ascii")
        else:
            value = str(raw) if raw is not None else ""
        if value and not value.startswith("GPU-"):
            value = f"GPU-{value}"
        if not value:
            raise RuntimeError("PyTorch did not expose one visible GPU UUID")
        values.append(value)
    if len(set(values)) != 2:
        raise RuntimeError("two logical devices resolve to the same physical GPU")
    return values[0], values[1]


def _artifact_runtime_metadata(
    *,
    bindings: Any,
    spec: WorkerSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = dict(bindings.metadata)
    visible = _visible_gpu_uuids(bindings.runtime.torch)
    if metadata.get("gpu_uuid") != visible[spec.index_parity]:
        raise RuntimeError("runtime worker UUID differs from logical CUDA assignment")
    return {
        "container_id": args.container_id,
        "host_alias": args.host_alias,
        "hostname": args.host_hostname,
        "visible_gpu_count": 2,
        "visible_gpu_uuids": list(visible),
        "worker_gpu_uuid": visible[spec.index_parity],
        "worker_cuda_device": spec.device,
        "bindings_metadata": metadata,
        "bindings_metadata_sha256": sha256_bytes(canonical_json_bytes(metadata)),
    }


def _load_runtime(
    spec: WorkerSpec,
    authorized: AuthorizedExpansionLabels,
    args: argparse.Namespace,
) -> WorkerRuntime:
    bindings = load_worker_runtime(
        spec=spec,
        model_dir=args.model_dir,
        snapshot_manifest=authorized.snapshot_manifest_path,
        container_image_digest=args.container_image_digest,
    )
    return WorkerRuntime(
        runtime=bindings.runtime,
        distance_backend=bindings.distance_backend,
        runtime_metadata=_artifact_runtime_metadata(
            bindings=bindings,
            spec=spec,
            args=args,
        ),
    )


def _wait_runtime_release(
    release_event: Any,
    abort_event: Any,
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not release_event.is_set():
        if abort_event.is_set():
            raise RuntimeError("global runtime barrier aborted")
        if time.monotonic() >= deadline:
            raise TimeoutError("global two-worker runtime barrier timed out")
        time.sleep(0.05)


def _sum_operation_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    keys = tuple(state_operation_counts(2))
    return {
        key: sum(int(record["operation_counts"][key]) for record in records)
        for key in keys
    }


def run_worker(
    *,
    layout: ExpansionLabelLayout,
    spec: WorkerSpec,
    authorized: AuthorizedExpansionLabels | None,
    args: argparse.Namespace,
    release_event: Any | None = None,
    abort_event: Any | None = None,
    authorization_loader: Callable[
        [ExpansionLabelLayout, argparse.Namespace], AuthorizedExpansionLabels
    ]
    | None = None,
    runtime_loader: Callable[
        [WorkerSpec, AuthorizedExpansionLabels, argparse.Namespace], WorkerRuntime
    ] = _load_runtime,
    state_runner: Callable[..., Mapping[str, Any]] = run_expansion_label_state_once,
) -> dict[str, Any]:
    worker_root = layout.root / WORKER_DIRECTORY / spec.worker_id
    started = _utc_now()
    attempted: list[int] = []
    completed: list[int] = []
    records: list[Mapping[str, Any]] = []
    initial = _worker_ledger_record(
        layout,
        spec,
        status=WORKER_CLAIM_STATUS,
        attempted=attempted,
        completed=completed,
        started_at_utc=started,
        ended_at_utc=started,
    )
    _publish_worker_ledgers(layout, spec, initial, first=True)
    failure_stage = "worker_authorization"
    try:
        if authorized is None:
            if authorization_loader is None:
                raise RuntimeError("spawned worker lacks fresh authorization loader")
            authorized = authorization_loader(layout, args)
        failure_stage = "runtime_barrier_or_fixed_shard_execution"
        bindings = runtime_loader(spec, authorized, args)
        _write_json_exclusive(
            worker_root / WORKER_RUNTIME_FILENAME,
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": "VALIDATED_EXPANSION_EXACT_LABEL_WORKER_RUNTIME",
                "run_contract_sha256": layout.run_contract_sha256,
                "worker": spec.envelope_identity(),
                "runtime_metadata": dict(bindings.runtime_metadata),
            },
            staging=layout.publish_staging,
        )
        if release_event is not None and abort_event is not None:
            _wait_runtime_release(
                release_event,
                abort_event,
                timeout_seconds=RUNTIME_BARRIER_TIMEOUT_SECONDS,
            )
        for index in spec.state_indices:
            if abort_event is not None and abort_event.is_set():
                raise RuntimeError("sibling worker aborted the fixed shard")
            attempted.append(index)
            running = _worker_ledger_record(
                layout,
                spec,
                status="RUNNING_EXPANSION_EXACT_LABEL_WORKER_NO_RETRY",
                attempted=attempted,
                completed=completed,
                started_at_utc=started,
                ended_at_utc=_utc_now(),
            )
            _publish_worker_ledgers(layout, spec, running, first=False)
            _write_json_exclusive(
                worker_root / ATTEMPT_DIRECTORY / f"{index:03d}.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol_id": PROTOCOL_ID,
                    "status": "CLAIMED_EXPANSION_EXACT_LABEL_STATE_NO_RETRY",
                    "run_contract_sha256": layout.run_contract_sha256,
                    "worker": spec.envelope_identity(),
                    "state_index": index,
                    "retry_count": 0,
                    "top_up_count": 0,
                    "claimed_at_utc": _utc_now(),
                },
                staging=layout.publish_staging,
            )
            if abort_event is not None and abort_event.is_set():
                raise RuntimeError("sibling worker aborted before teacher forward")
            record = state_runner(
                artifact=authorized.artifact,
                parent=authorized.parents[index],
                worker=spec,
                runtime=bindings.runtime,
                distance_backend=bindings.distance_backend,
                run_contract_sha256=layout.run_contract_sha256,
                expected_state=authorized.state_projections[index],
            )
            _write_json_exclusive(
                worker_root / STATE_DIRECTORY / f"{index:03d}.json",
                record,
                staging=layout.publish_staging,
            )
            records.append(record)
            completed.append(index)
            advanced = _worker_ledger_record(
                layout,
                spec,
                status="RUNNING_EXPANSION_EXACT_LABEL_WORKER_NO_RETRY",
                attempted=attempted,
                completed=completed,
                started_at_utc=started,
                ended_at_utc=_utc_now(),
            )
            _publish_worker_ledgers(layout, spec, advanced, first=False)
        outcome = WORKER_STATUS
        ledger_status = WORKER_STATUS
        failure = None
    except Exception as error:
        outcome = WORKER_INVALID_STATUS
        ledger_status = WORKER_INVALID_STATUS
        failure = {
            "stage": failure_stage,
            "exception_type": error.__class__.__name__,
            "message": str(error),
        }
        if abort_event is not None:
            abort_event.set()
    ended = _utc_now()
    final_ledger = _worker_ledger_record(
        layout,
        spec,
        status=ledger_status,
        attempted=attempted,
        completed=completed,
        started_at_utc=started,
        ended_at_utc=ended,
    )
    _publish_worker_ledgers(layout, spec, final_ledger, first=False)
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "TERMINAL_EXPANSION_EXACT_LABEL_WORKER",
        "outcome": outcome,
        "run_contract_sha256": layout.run_contract_sha256,
        "worker": spec.to_dict(),
        "attempted_state_indices": attempted,
        "completed_state_indices": completed,
        "operation_counts": _sum_operation_counts(records),
        "failure": failure,
        "ended_at_utc": ended,
    }
    _write_json_exclusive(
        worker_root / WORKER_TERMINAL_FILENAME,
        terminal,
        staging=layout.publish_staging,
    )
    return terminal


def _runtime_envelopes(layout: ExpansionLabelLayout) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for spec in expected_worker_specs():
        path = (
            layout.root
            / WORKER_DIRECTORY
            / spec.worker_id
            / WORKER_RUNTIME_FILENAME
        )
        result[spec.worker_id] = _strict_json(path)
    return result


def validate_runtime_pair_for_release(
    envelopes: Mapping[str, Mapping[str, Any]],
    *,
    run_contract_sha256: str,
) -> tuple[str, str]:
    specs = expected_worker_specs()
    if set(envelopes) != {spec.worker_id for spec in specs}:
        raise ValueError("runtime barrier lacks both fixed workers")
    visible: tuple[str, str] | None = None
    worker_uuids: list[str] = []
    shared: dict[str, Any] | None = None
    bindings_by_worker: dict[str, Mapping[str, Any]] = {}
    metadata_keys = {
        "container_id",
        "host_alias",
        "hostname",
        "visible_gpu_count",
        "visible_gpu_uuids",
        "worker_gpu_uuid",
        "worker_cuda_device",
        "bindings_metadata",
        "bindings_metadata_sha256",
    }
    for spec in specs:
        envelope = envelopes[spec.worker_id]
        if (
            set(envelope)
            != {
                "schema_version",
                "protocol_id",
                "status",
                "run_contract_sha256",
                "worker",
                "runtime_metadata",
            }
            or envelope.get("schema_version") != SCHEMA_VERSION
            or envelope.get("protocol_id") != PROTOCOL_ID
            or envelope.get("status")
            != "VALIDATED_EXPANSION_EXACT_LABEL_WORKER_RUNTIME"
            or envelope.get("run_contract_sha256") != run_contract_sha256
            or envelope.get("worker") != spec.envelope_identity()
        ):
            raise ValueError("runtime barrier envelope identity drifted")
        metadata = envelope.get("runtime_metadata")
        if not isinstance(metadata, Mapping) or set(metadata) != metadata_keys:
            raise ValueError("runtime barrier metadata schema drifted")
        bindings = metadata.get("bindings_metadata")
        if not isinstance(bindings, Mapping):
            raise ValueError("runtime barrier lacks full bindings metadata")
        bindings = dict(bindings)
        if metadata.get("bindings_metadata_sha256") != sha256_bytes(
            canonical_json_bytes(bindings)
        ):
            raise ValueError("runtime barrier bindings metadata SHA256 drifted")
        validate_eager_worker_runtime_metadata(bindings, spec=spec)
        observed = tuple(metadata.get("visible_gpu_uuids", ()))
        if (
            not isinstance(metadata.get("container_id"), str)
            or re.fullmatch(r"[0-9a-f]{64}", metadata["container_id"]) is None
            or metadata.get("host_alias") not in {"hyper00", "hyper01"}
            or metadata.get("hostname")
            != {
                "hyper00": "node-radixark-16-0000",
                "hyper01": "node-radixark-16-0001",
            }.get(metadata.get("host_alias"))
            or metadata.get("visible_gpu_count") != 2
            or len(observed) != 2
            or len(set(observed)) != 2
            or any(
                not isinstance(item, str)
                or re.fullmatch(r"GPU-[0-9a-fA-F-]{8,64}", item) is None
                for item in observed
            )
            or metadata.get("worker_gpu_uuid") != observed[spec.index_parity]
            or metadata.get("worker_cuda_device") != spec.device
            or bindings.get("gpu_uuid") != metadata.get("worker_gpu_uuid")
            or bindings.get("device") != metadata.get("worker_cuda_device")
        ):
            raise ValueError("runtime barrier GPU assignment drifted")
        if visible is None:
            visible = (str(observed[0]), str(observed[1]))
        elif observed != visible:
            raise ValueError("workers observed different visible GPU sets")
        worker_uuids.append(str(metadata["worker_gpu_uuid"]))
        bindings_by_worker[spec.worker_id] = bindings
        projection = {
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "worker_gpu_uuid",
                "worker_cuda_device",
                "bindings_metadata",
                "bindings_metadata_sha256",
            }
        }
        if shared is None:
            shared = projection
        elif projection != shared:
            raise ValueError("worker container/software runtime identities differ")
    if visible is None or tuple(worker_uuids) != visible:
        raise ValueError("even/odd workers do not own cuda:0/cuda:1 respectively")
    validate_eager_worker_runtime_pair(bindings_by_worker)
    return visible


def _validate_external_ready(
    path: Path,
    *,
    visible_gpu_uuids: tuple[str, str] | None = None,
) -> dict[str, Any]:
    ready = _strict_json(path)
    expected_keys = {
        "schema_version",
        "status",
        "monitor_pid",
        "visible_gpu_count",
        "visible_gpu_uuids",
        "started_at_utc",
    }
    observed = tuple(ready.get("visible_gpu_uuids", ()))
    timestamp = ready.get("started_at_utc")
    if (
        set(ready) != expected_keys
        or ready.get("schema_version") != SCHEMA_VERSION
        or ready.get("status") != MONITOR_READY_STATUS
        or type(ready.get("monitor_pid")) is not int
        or ready["monitor_pid"] <= 0
        or ready.get("visible_gpu_count") != 2
        or len(observed) != 2
        or len(set(observed)) != 2
        or any(
            not isinstance(item, str)
            or re.fullmatch(r"GPU-[0-9a-fA-F-]{8,64}", item) is None
            for item in observed
        )
        or not isinstance(timestamp, str)
        or not timestamp.endswith("Z")
        or (
            visible_gpu_uuids is not None
            and observed != visible_gpu_uuids
        )
    ):
        raise ValueError("external utilization monitor ready identity drifted")
    try:
        datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("external monitor ready timestamp is invalid") from error
    try:
        os.kill(int(ready["monitor_pid"]), 0)
    except OSError as error:
        raise RuntimeError("utilization monitor process is not alive") from error
    return ready


def bind_monitor_ready_before_release(
    layout: ExpansionLabelLayout,
    *,
    external_ready_file: str | Path,
    visible_gpu_uuids: tuple[str, str],
) -> dict[str, Any]:
    external = _validate_external_ready(
        Path(external_ready_file),
        visible_gpu_uuids=visible_gpu_uuids,
    )
    bound = {
        **external,
        "run_contract_sha256": layout.run_contract_sha256,
    }
    _write_json_exclusive(
        layout.root / LOG_PATHS["utilization_monitor_ready"],
        bound,
        staging=layout.publish_staging,
    )
    return bound


def rebuild_worker_authorization(
    layout: ExpansionLabelLayout,
    args: argparse.Namespace,
    *,
    authorization_loader: Callable[
        [argparse.Namespace], AuthorizedExpansionLabels
    ] = authorize_expansion_label_run,
    contract_builder: Callable[
        [argparse.Namespace, AuthorizedExpansionLabels], Mapping[str, Any]
    ] = build_run_contract,
) -> AuthorizedExpansionLabels:
    authorized = authorization_loader(args)
    rebuilt = contract_builder(args, authorized)
    if sha256_bytes(canonical_json_bytes(rebuilt)) != layout.run_contract_sha256:
        raise ValueError("worker rebuilt authorization differs from global claim")
    return authorized


def _worker_process_entry(
    layout: ExpansionLabelLayout,
    spec: WorkerSpec,
    args: argparse.Namespace,
    release_event: Any,
    abort_event: Any,
) -> None:
    terminal = run_worker(
        layout=layout,
        spec=spec,
        authorized=None,
        args=args,
        release_event=release_event,
        abort_event=abort_event,
        authorization_loader=rebuild_worker_authorization,
    )
    if terminal["outcome"] != WORKER_STATUS:
        raise RuntimeError(str(terminal["failure"]))


def _coordinate_runtime_barrier(
    layout: ExpansionLabelLayout,
    processes: Mapping[str, Any],
    release_event: Any,
    abort_event: Any,
    *,
    monitor_ready_file: str | Path,
    timeout_seconds: float = RUNTIME_BARRIER_TIMEOUT_SECONDS,
) -> tuple[str, str]:
    deadline = time.monotonic() + timeout_seconds
    paths = {
        spec.worker_id: (
            layout.root
            / WORKER_DIRECTORY
            / spec.worker_id
            / WORKER_RUNTIME_FILENAME
        )
        for spec in expected_worker_specs()
    }
    while not all(path.is_file() for path in paths.values()):
        failed = [
            worker_id
            for worker_id, process in processes.items()
            if process.exitcode is not None
        ]
        if failed:
            abort_event.set()
            raise RuntimeError(f"worker exited before runtime barrier: {failed}")
        if time.monotonic() >= deadline:
            abort_event.set()
            raise TimeoutError("parent runtime barrier timed out")
        time.sleep(0.05)
    visible = validate_runtime_pair_for_release(
        _runtime_envelopes(layout),
        run_contract_sha256=layout.run_contract_sha256,
    )
    bind_monitor_ready_before_release(
        layout,
        external_ready_file=monitor_ready_file,
        visible_gpu_uuids=visible,
    )
    release_event.set()
    return visible


def _supervise_released_workers(
    processes: Mapping[str, Any],
    abort_event: Any,
) -> None:
    while True:
        if any(
            process.exitcode is not None and process.exitcode != 0
            for process in processes.values()
        ):
            abort_event.set()
        if all(process.exitcode is not None for process in processes.values()):
            break
        time.sleep(0.05)
    for process in processes.values():
        process.join()


def monitor_stop_request_path(summary_file: str | Path) -> Path:
    summary = Path(summary_file)
    return summary.with_name(f"{summary.name}.stop-request.json")


def _write_external_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as destination:
            destination.write(pretty_json_bytes(dict(value)))
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def sample_visible_monitor_gpus() -> tuple[dict[str, Any], ...]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError("nvidia-smi monitor sample schema drifted")
        records.append(
            {
                "nvidia_smi_index": int(fields[0]),
                "gpu_uuid": fields[1],
                "utilization_percent": int(fields[2]),
            }
        )
    return _validated_monitor_sample(tuple(records))


def _validated_monitor_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_inventory: tuple[tuple[int, str], ...] | None = None,
) -> tuple[dict[str, Any], ...]:
    if len(records) != 2:
        raise RuntimeError("monitor sidecar must observe exactly two GPUs")
    normalized: list[dict[str, Any]] = []
    for record in records:
        value = dict(record)
        if set(value) != {
            "nvidia_smi_index",
            "gpu_uuid",
            "utilization_percent",
        }:
            raise ValueError("monitor GPU sample keys drifted")
        index = value["nvidia_smi_index"]
        gpu_uuid = value["gpu_uuid"]
        utilization = value["utilization_percent"]
        if (
            type(index) is not int
            or index < 0
            or not isinstance(gpu_uuid, str)
            or re.fullmatch(r"GPU-[0-9a-fA-F-]{8,64}", gpu_uuid) is None
            or type(utilization) is not int
            or utilization < 0
            or utilization > 100
        ):
            raise ValueError("monitor GPU identity or utilization drifted")
        normalized.append(value)
    inventory = tuple(
        (record["nvidia_smi_index"], record["gpu_uuid"]) for record in normalized
    )
    if (
        len(set(inventory)) != 2
        or len({item[0] for item in inventory}) != 2
        or len({item[1] for item in inventory}) != 2
    ):
        raise ValueError("monitor GPU inventory is not two distinct devices")
    if expected_inventory is not None and inventory != expected_inventory:
        raise RuntimeError("monitor visible GPU inventory changed during the attempt")
    return tuple(normalized)


def _preclaim_abort_identity(monitor_ready_sha256: str) -> str:
    if SHA256_PATTERN.fullmatch(monitor_ready_sha256) is None:
        raise ValueError("preclaim monitor identity requires a ready-file SHA256")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "attempt_id": ATTEMPT_ID,
                "monitor_ready_sha256": monitor_ready_sha256,
                "reason": MONITOR_PRECLAIM_ABORT_REASON,
            }
        )
    )


def run_monitor_sidecar(
    *,
    ready_file: str | Path,
    log_file: str | Path,
    summary_file: str | Path,
    sampler: Callable[[], tuple[dict[str, Any], ...]] = sample_visible_monitor_gpus,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    ready_path = Path(ready_file).resolve()
    log_path = Path(log_file).resolve()
    summary_path = Path(summary_file).resolve()
    stop_path = monitor_stop_request_path(summary_path)
    if any(
        path.exists() or path.is_symlink()
        for path in (ready_path, log_path, summary_path, stop_path)
    ):
        raise FileExistsError("monitor ready, log, summary, and stop paths must be fresh")
    first = _validated_monitor_sample(sampler())
    inventory = tuple(
        (record["nvidia_smi_index"], record["gpu_uuid"]) for record in first
    )
    visible = tuple(record["gpu_uuid"] for record in first)
    started = _utc_now()
    _write_external_json_exclusive(
        ready_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": MONITOR_READY_STATUS,
            "monitor_pid": os.getpid(),
            "visible_gpu_count": 2,
            "visible_gpu_uuids": list(visible),
            "started_at_utc": started,
        },
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    low_count = 0
    pending = first
    with log_path.open("xb") as destination:
        while True:
            sample = {
                "schema_version": SCHEMA_VERSION,
                "status": "GPU_UTILIZATION_SAMPLE",
                "sample_index": sample_count,
                "sampled_at_utc": _utc_now(),
                "visible_gpu_count": 2,
                "gpus": list(pending),
            }
            destination.write(canonical_json_bytes(sample) + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
            sample_count += 1
            low_count += int(
                any(int(record["utilization_percent"]) < 90 for record in pending)
            )
            if stop_path.is_file():
                break
            sleep_fn(MONITOR_SAMPLE_INTERVAL_SECONDS)
            if stop_path.is_file():
                break
            pending = _validated_monitor_sample(
                sampler(),
                expected_inventory=inventory,
            )
    stop = _strict_json(stop_path)
    if stop.get("status") == MONITOR_STOP_STATUS:
        if (
            set(stop)
            != {
                "schema_version",
                "status",
                "run_contract_sha256",
                "requested_at_utc",
            }
            or stop.get("schema_version") != SCHEMA_VERSION
            or not isinstance(stop.get("run_contract_sha256"), str)
            or SHA256_PATTERN.fullmatch(stop["run_contract_sha256"]) is None
        ):
            raise ValueError("monitor stop request drifted")
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": MONITOR_SUMMARY_STATUS,
            "run_contract_sha256": stop["run_contract_sha256"],
            "container_name_prefix": "sglang-omni-jaxan",
            "minimum_gpu_utilization_percent": 90,
            "started_before_first_teacher_forward": True,
            "started_at_utc": started,
            "stopped_at_utc": _utc_now(),
            "sample_count": sample_count,
            "low_utilization_incident_count": low_count,
            "visible_gpu_count": 2,
            "visible_gpu_uuids": list(visible),
            "stop_request_sha256": sha256_file(stop_path),
        }
    elif stop.get("status") == MONITOR_PRECLAIM_ABORT_STOP_STATUS:
        expected_ready_sha = sha256_file(ready_path)
        expected_identity = _preclaim_abort_identity(expected_ready_sha)
        if (
            set(stop)
            != {
                "schema_version",
                "status",
                "attempt_id",
                "monitor_ready_sha256",
                "preclaim_abort_identity_sha256",
                "reason",
                "requested_at_utc",
            }
            or stop.get("schema_version") != SCHEMA_VERSION
            or stop.get("attempt_id") != ATTEMPT_ID
            or stop.get("monitor_ready_sha256") != expected_ready_sha
            or stop.get("preclaim_abort_identity_sha256") != expected_identity
            or stop.get("reason") != MONITOR_PRECLAIM_ABORT_REASON
        ):
            raise ValueError("preclaim monitor abort identity drifted")
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": MONITOR_PRECLAIM_ABORT_SUMMARY_STATUS,
            "attempt_id": ATTEMPT_ID,
            "monitor_ready_sha256": expected_ready_sha,
            "preclaim_abort_identity_sha256": expected_identity,
            "reason": MONITOR_PRECLAIM_ABORT_REASON,
            "started_at_utc": started,
            "stopped_at_utc": _utc_now(),
            "sample_count": sample_count,
            "low_utilization_incident_count": low_count,
            "visible_gpu_count": 2,
            "visible_gpu_uuids": list(visible),
            "stop_request_sha256": sha256_file(stop_path),
        }
    else:
        raise ValueError("monitor stop request status drifted")
    _write_external_json_exclusive(summary_path, result)
    return result


def publish_monitor_stop_request(
    *,
    summary_file: str | Path,
    run_contract_sha256: str,
) -> dict[str, Any]:
    if SHA256_PATTERN.fullmatch(run_contract_sha256) is None:
        raise ValueError("monitor stop request run-contract SHA256 is invalid")
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": MONITOR_STOP_STATUS,
        "run_contract_sha256": run_contract_sha256,
        "requested_at_utc": _utc_now(),
    }
    _write_external_json_exclusive(monitor_stop_request_path(summary_file), record)
    return record


def publish_monitor_preclaim_abort_stop_request(
    *,
    summary_file: str | Path,
    ready_file: str | Path,
) -> dict[str, Any]:
    ready_path = Path(ready_file).resolve()
    ready = _strict_json(ready_path)
    if (
        ready.get("schema_version") != SCHEMA_VERSION
        or ready.get("status") != MONITOR_READY_STATUS
    ):
        raise ValueError("preclaim abort requires the live monitor ready record")
    ready_sha = sha256_file(ready_path)
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": MONITOR_PRECLAIM_ABORT_STOP_STATUS,
        "attempt_id": ATTEMPT_ID,
        "monitor_ready_sha256": ready_sha,
        "preclaim_abort_identity_sha256": _preclaim_abort_identity(ready_sha),
        "reason": MONITOR_PRECLAIM_ABORT_REASON,
        "requested_at_utc": _utc_now(),
    }
    _write_external_json_exclusive(monitor_stop_request_path(summary_file), record)
    return record


def _wait_for_monitor_summary(
    path: Path,
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("utilization monitor summary timed out")
        time.sleep(0.05)
    return _strict_json(path)


def terminate_preclaim_monitor(args: argparse.Namespace) -> dict[str, Any]:
    ready_path = Path(args.monitor_ready_file).resolve()
    summary_path = Path(args.monitor_summary).resolve()
    stop_path = monitor_stop_request_path(summary_path)
    expected_ready_sha = sha256_file(ready_path)
    expected_identity = _preclaim_abort_identity(expected_ready_sha)
    if stop_path.exists() or stop_path.is_symlink():
        stop = _strict_json(stop_path)
    else:
        stop = publish_monitor_preclaim_abort_stop_request(
            summary_file=summary_path,
            ready_file=ready_path,
        )
    if (
        stop.get("status") != MONITOR_PRECLAIM_ABORT_STOP_STATUS
        or stop.get("attempt_id") != ATTEMPT_ID
        or stop.get("monitor_ready_sha256") != expected_ready_sha
        or stop.get("preclaim_abort_identity_sha256") != expected_identity
        or stop.get("reason") != MONITOR_PRECLAIM_ABORT_REASON
    ):
        raise ValueError("preclaim monitor stop request identity drifted")
    summary = _wait_for_monitor_summary(summary_path, timeout_seconds=10.0)
    if (
        summary.get("status") != MONITOR_PRECLAIM_ABORT_SUMMARY_STATUS
        or summary.get("attempt_id") != ATTEMPT_ID
        or summary.get("monitor_ready_sha256") != expected_ready_sha
        or summary.get("preclaim_abort_identity_sha256") != expected_identity
        or summary.get("reason") != MONITOR_PRECLAIM_ABORT_REASON
        or summary.get("stop_request_sha256") != sha256_file(stop_path)
    ):
        raise ValueError("preclaim monitor abort summary identity drifted")
    return summary


def finalize_monitor_evidence(
    layout: ExpansionLabelLayout,
    args: argparse.Namespace,
    *,
    visible_gpu_uuids: tuple[str, str],
) -> dict[str, Any]:
    stop_path = monitor_stop_request_path(args.monitor_summary)
    stop = _strict_json(stop_path)
    summary_path = Path(args.monitor_summary).resolve()
    summary = _wait_for_monitor_summary(summary_path)
    if (
        stop.get("run_contract_sha256") != layout.run_contract_sha256
        or summary.get("run_contract_sha256") != layout.run_contract_sha256
        or tuple(summary.get("visible_gpu_uuids", ())) != visible_gpu_uuids
        or summary.get("started_before_first_teacher_forward") is not True
        or summary.get("stop_request_sha256") != sha256_file(stop_path)
    ):
        raise ValueError("terminal monitor summary binding drifted")
    _copy_exclusive(
        Path(args.utilization_monitor_log).resolve(),
        layout.root / LOG_PATHS["utilization_monitor"],
    )
    _copy_exclusive(
        stop_path,
        layout.root / LOG_PATHS["utilization_monitor_stop_request"],
    )
    _copy_exclusive(
        summary_path,
        layout.root / MONITOR_SUMMARY_FILENAME,
    )
    return summary


def _execution_evidence(layout: ExpansionLabelLayout) -> dict[str, Any]:
    files = {
        "preflight_log_sha256": layout.root / LOG_PATHS["preflight"],
        "execution_log_sha256": layout.root / LOG_PATHS["execution"],
        "utilization_monitor_log_sha256": (
            layout.root / LOG_PATHS["utilization_monitor"]
        ),
        "monitor_ready_sha256": (
            layout.root / LOG_PATHS["utilization_monitor_ready"]
        ),
        "monitor_stop_request_sha256": (
            layout.root / LOG_PATHS["utilization_monitor_stop_request"]
        ),
        "monitor_summary_sha256": layout.root / MONITOR_SUMMARY_FILENAME,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "VALIDATED_EXPANSION_EXACT_LABEL_EXECUTION_EVIDENCE",
        "run_contract_sha256": layout.run_contract_sha256,
        **{key: sha256_file(path) for key, path in files.items()},
    }


def _load_ordered_state_records(layout: ExpansionLabelLayout) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(EXPECTED_STATE_COUNT):
        spec = expected_worker_specs()[index % 2]
        result.append(
            _strict_json(
                layout.root
                / WORKER_DIRECTORY
                / spec.worker_id
                / STATE_DIRECTORY
                / f"{index:03d}.json"
            )
        )
    return result


def _terminal_records(layout: ExpansionLabelLayout) -> dict[str, dict[str, Any]]:
    return {
        spec.worker_id: _strict_json(
            layout.root
            / WORKER_DIRECTORY
            / spec.worker_id
            / WORKER_TERMINAL_FILENAME
        )
        for spec in expected_worker_specs()
    }


def _evidence_files(layout: ExpansionLabelLayout) -> dict[str, bytes]:
    return {
        path.relative_to(layout.root).as_posix(): path.read_bytes()
        for path in layout.root.rglob("*")
        if path.is_file()
    }


def finalize_successful_attempt(
    layout: ExpansionLabelLayout,
    authorized: AuthorizedExpansionLabels,
    args: argparse.Namespace,
    *,
    visible_gpu_uuids: tuple[str, str],
) -> dict[str, Any]:
    terminals = _terminal_records(layout)
    for spec in expected_worker_specs():
        terminal = terminals[spec.worker_id]
        if (
            terminal.get("outcome") != WORKER_STATUS
            or terminal.get("attempted_state_indices") != list(spec.state_indices)
            or terminal.get("completed_state_indices") != list(spec.state_indices)
            or terminal.get("failure") is not None
        ):
            raise RuntimeError("worker did not complete its exact 96-state shard")
    publish_monitor_stop_request(
        summary_file=args.monitor_summary,
        run_contract_sha256=layout.run_contract_sha256,
    )
    finalize_monitor_evidence(
        layout,
        args,
        visible_gpu_uuids=visible_gpu_uuids,
    )
    _append_execution_log(
        layout,
        "both 96-state shards and terminal monitor evidence completed",
    )
    execution = _execution_evidence(layout)
    _write_json_exclusive(
        layout.root / EXECUTION_EVIDENCE_FILENAME,
        execution,
        staging=layout.publish_staging,
    )
    raw_states = _load_ordered_state_records(layout)
    reduction = reduce_raw_distance_states(
        raw_states,
        expected_states=authorized.state_projections,
        run_contract_sha256=layout.run_contract_sha256,
    )
    ended = _utc_now()
    aggregate = aggregate_from_reduction(
        reduction,
        run_contract_sha256=layout.run_contract_sha256,
        execution_evidence_sha256=sha256_file(
            layout.root / EXECUTION_EVIDENCE_FILENAME
        ),
        started_at_utc=layout.started_at_utc,
        ended_at_utc=ended,
    )
    _write_json_exclusive(
        layout.root / AGGREGATE_FILENAME,
        aggregate,
        staging=layout.publish_staging,
    )
    final_ledger = _global_ledger_record(
        layout,
        status="COMPLETED_EXPANSION_EXACT_LABEL_ATTEMPT",
        attempted=tuple(range(EXPECTED_STATE_COUNT)),
        completed=tuple(range(EXPECTED_STATE_COUNT)),
        ended_at_utc=ended,
    )
    _replace_json_durable(
        layout.global_ledger,
        final_ledger,
        staging=layout.publish_staging,
    )
    _copy_exclusive(
        layout.global_ledger,
        layout.root / GLOBAL_LEDGER_ARCHIVE_NAME,
    )
    evidence = validate_expansion_label_evidence_files(
        _evidence_files(layout),
        expected_source_git_commit=authorized.runner_validation[
            "runner_source_git_commit"
        ],
        expected_config_sha256=authorized.contract_validation["config_sha256"],
    )
    return {
        "outcome": evidence.outcome,
        "run_contract_sha256": evidence.run_contract_sha256,
        "counts": dict(evidence.reduction["counts"]),
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
    }


def seal_invalid_attempt(
    layout: ExpansionLabelLayout,
    *,
    error: BaseException,
) -> dict[str, Any]:
    attempted: set[int] = set()
    completed: set[int] = set()
    for spec in expected_worker_specs():
        terminal_path = (
            layout.root
            / WORKER_DIRECTORY
            / spec.worker_id
            / WORKER_TERMINAL_FILENAME
        )
        if terminal_path.is_file():
            terminal = _strict_json(terminal_path)
            attempted.update(terminal.get("attempted_state_indices", ()))
            completed.update(terminal.get("completed_state_indices", ()))
    record = _global_ledger_record(
        layout,
        status=GLOBAL_INVALID_STATUS,
        attempted=sorted(attempted),
        completed=sorted(completed),
        ended_at_utc=_utc_now(),
    )
    record["failure"] = {
        "exception_type": error.__class__.__name__,
        "message": str(error),
    }
    # note (luojiaxuan): INVALID evidence intentionally does not satisfy the
    # successful artifact schema; it remains a durable one-shot tombstone and
    # can never be resumed, retried, topped up, or repackaged as PASS.
    current = _strict_json(layout.global_ledger)
    record["claimed_ledger_sha256"] = sha256_bytes(pretty_json_bytes(current))
    _replace_json_durable(
        layout.global_ledger,
        record,
        staging=layout.publish_staging,
    )
    try:
        _append_execution_log(layout, f"attempt invalidated: {error}")
    except OSError:
        pass
    return record


def execute_claimed_attempt(
    layout: ExpansionLabelLayout,
    authorized: AuthorizedExpansionLabels,
    args: argparse.Namespace,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    release_event = context.Event()
    abort_event = context.Event()
    processes = {
        spec.worker_id: context.Process(
            target=_worker_process_entry,
            args=(layout, spec, args, release_event, abort_event),
            name=f"expansion-label-{spec.worker_id}",
        )
        for spec in expected_worker_specs()
    }
    try:
        for process in processes.values():
            process.start()
        visible = _coordinate_runtime_barrier(
            layout,
            processes,
            release_event,
            abort_event,
            monitor_ready_file=args.monitor_ready_file,
        )
        _supervise_released_workers(processes, abort_event)
        failed = {
            worker_id: process.exitcode
            for worker_id, process in processes.items()
            if process.exitcode != 0
        }
        if failed:
            raise RuntimeError(f"fixed worker shard process failed: {failed}")
        return finalize_successful_attempt(
            layout,
            authorized,
            args,
            visible_gpu_uuids=visible,
        )
    except Exception as error:
        abort_event.set()
        for process in processes.values():
            if process.is_alive():
                process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        if not monitor_stop_request_path(args.monitor_summary).exists():
            try:
                publish_monitor_stop_request(
                    summary_file=args.monitor_summary,
                    run_contract_sha256=layout.run_contract_sha256,
                )
            except Exception:
                pass
        seal_invalid_attempt(layout, error=error)
        raise


def execute_attempt(args: argparse.Namespace) -> dict[str, Any]:
    _validate_external_ready(Path(args.monitor_ready_file).resolve())
    try:
        authorized = authorize_expansion_label_run(args)
        run_contract = build_run_contract(args, authorized)
        layout = claim_attempt(args, run_contract)
    except Exception as error:
        try:
            terminate_preclaim_monitor(args)
        except Exception as monitor_error:
            error.add_note(
                "preclaim monitor termination also failed: " + str(monitor_error)
            )
        raise
    return execute_claimed_attempt(layout, authorized, args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen restoration-v2.2 expansion exact labels",
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--runner-freeze", required=True)
    parser.add_argument("--parent-substrate-archive", required=True)
    parser.add_argument("--derived-artifact-root", required=True)
    parser.add_argument("--ocr-backend-config", required=True)
    parser.add_argument("--snapshot-manifest", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--host-hostname", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--global-ledger", required=True)
    parser.add_argument("--preflight-log", required=True)
    parser.add_argument("--utilization-monitor-log", required=True)
    parser.add_argument("--monitor-ready-file", required=True)
    parser.add_argument("--monitor-summary", required=True)
    return parser


def _build_monitor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor the exact-label runner's two visible GPUs",
    )
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--summary-file", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == MONITOR_SIDECAR_SUBCOMMAND:
        args = _build_monitor_parser().parse_args(values[1:])
        result = run_monitor_sidecar(
            ready_file=args.ready_file,
            log_file=args.log_file,
            summary_file=args.summary_file,
        )
    else:
        args = _build_parser().parse_args(values)
        args.execution_argv = values
        result = execute_attempt(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
