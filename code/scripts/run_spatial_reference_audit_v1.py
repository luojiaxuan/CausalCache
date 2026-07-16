"""Run one fail-closed numerical profile of spatial_reference_audit_v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.data.guiodyssey_restoration_v2 import (
    canonical_json_bytes,
    sha256_file,
)
from causalcache.data.restoration_v2_screening import load_validated_screening_artifact
from causalcache.policy.gui_owl_spatial_audit_runtime import (
    GUIOwlSpatialAuditRuntime,
    audit_absent_scientific_environment,
)
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21GenerationParseError
from causalcache.spatial_reference_audit_v1 import (
    PROTOCOL_ID,
    action_from_mapping,
    canonical_profile_payload_sha256,
    coordinate_delta,
    load_and_validate_config,
    load_parent_mismatches,
    profile_by_id,
    profile_operation_counts,
    retokenize_parent_mismatch,
    validate_repository_inputs,
)
from scripts.run_restoration_v2_1_full_45_substrate import (
    _action_arguments,
    _decode_rgb_image,
    _utc_now,
    build_full45_messages,
)
from scripts.run_restoration_v2_1_interface_pilot import validate_clean_pushed_main


SCHEMA_VERSION = "1.0.0"
ATTEMPT_ID = "spatial-reference-audit-v1"
CANONICAL_ATTEMPT_ROOT = Path(
    "/data/experiments/causalcache/spatial-reference-audit-v1"
)
CANONICAL_GLOBAL_LEDGER = Path(
    "/data/experiments/causalcache/.spatial-reference-audit-v1.attempt.json"
)
PROFILE_ORDER = ("bf16_auto", "bf16_eager_control", "fp32_eager_control")
OPERATION_COUNT_KEYS = (
    "generation_calls",
    "teacher_forwards",
    "confirm_state_accesses",
    "restoration_coalition_constructions",
    "gate_training_examples",
)
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
CONTAINER_HOSTNAME_PATTERN = re.compile(r"[0-9a-f]{12,64}")
IMAGE_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class AttemptLayout:
    root: Path
    global_ledger: Path
    profiles_root: Path
    attempt_start: Path


@dataclass(frozen=True)
class ProfileLayout:
    position: int
    root: Path
    states_root: Path
    claim: Path
    start: Path
    terminal: Path


@dataclass(frozen=True)
class StateLayout:
    ordinal: int
    root: Path
    start: Path
    terminal: Path


def _absolute(path: str, *, name: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise ValueError(f"--{name} must be an absolute path")
    return value.resolve()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_exclusive_durable(path: Path) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"directory parent does not exist: {path.parent}")
    path.mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(path.parent)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"JSON parent directory does not exist: {path.parent}")
    payload = _json_bytes(value)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"durable ledger disappeared: {path}")
    payload = _json_bytes(value)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid durable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"durable JSON root must be an object: {path}")
    return value


def _zero_operation_counts() -> dict[str, int]:
    return {key: 0 for key in OPERATION_COUNT_KEYS}


def _add_operation_counts(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for key in OPERATION_COUNT_KEYS:
        first = left.get(key)
        second = right.get(key)
        if type(first) is not int or first < 0 or type(second) is not int or second < 0:
            raise ValueError("operation counts must be non-negative integers")
        result[key] = first + second
    return result


def _exception_record(error: BaseException) -> dict[str, str]:
    return {
        "exception_type": error.__class__.__name__,
        "exception_message": str(error)[:4000],
    }


def _validated_attempt_identity(config: Mapping[str, Any]) -> AttemptLayout:
    raw = config.get("attempt_identity")
    if not isinstance(raw, Mapping):
        raise TypeError("spatial-reference config lacks attempt_identity")
    expected = {
        "attempt_id": ATTEMPT_ID,
        "canonical_persistent_output_dir": str(CANONICAL_ATTEMPT_ROOT),
        "canonical_global_attempt_ledger": str(CANONICAL_GLOBAL_LEDGER),
        "canonical_host_alias": "hyper01",
        "canonical_host_hostname": "node-radixark-16-0001",
        "canonical_device": "cuda:0",
        "profile_order": list(PROFILE_ORDER),
        "alternate_output_or_ledger_allowed": False,
        "incomplete_attempt_retry_allowed": False,
        "profile_retry_allowed": False,
        "state_retry_allowed": False,
        "output_or_ledger_deletion_after_claim_allowed": False,
    }
    if dict(raw) != expected:
        raise ValueError("spatial-reference canonical attempt identity drifted")
    root = Path(str(raw["canonical_persistent_output_dir"]))
    ledger = Path(str(raw["canonical_global_attempt_ledger"]))
    if root.resolve() != CANONICAL_ATTEMPT_ROOT or ledger.resolve() != CANONICAL_GLOBAL_LEDGER:
        raise ValueError("spatial-reference attempt paths are not canonical")
    if root.parent != ledger.parent or root == ledger:
        raise ValueError("spatial-reference global ledger must be a sibling of the root")
    return AttemptLayout(
        root=root,
        global_ledger=ledger,
        profiles_root=root / "profiles",
        attempt_start=root / "start.json",
    )


def _live_software_identity() -> dict[str, Any]:
    try:
        import torch
        import transformers
    except ModuleNotFoundError as error:
        raise RuntimeError("pinned audit runtime packages are unavailable") from error
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "transformers_version": str(transformers.__version__),
    }


def _live_execution_identity(
    args: argparse.Namespace,
    *,
    runtime_constraints: Mapping[str, Any],
) -> dict[str, Any]:
    container_id = str(args.container_id)
    image_digest = str(args.container_image_digest)
    if CONTAINER_ID_PATTERN.fullmatch(container_id) is None:
        raise ValueError("--container-id must be the full lowercase 64-hex container ID")
    if IMAGE_DIGEST_PATTERN.fullmatch(image_digest) is None:
        raise ValueError("--container-image-digest must be a sha256 digest")
    if image_digest != runtime_constraints.get("container_image_digest"):
        raise ValueError("CLI container image digest differs from the frozen runtime")
    expected_environment_names = runtime_constraints.get(
        "audited_scientific_environment_variables"
    )
    scientific_environment_audit = audit_absent_scientific_environment()
    if scientific_environment_audit.get("audited_names") != expected_environment_names:
        raise ValueError("scientific environment audit inventory drifted")
    software = _live_software_identity()
    for key in (
        "python_version",
        "torch_version",
        "torch_cuda_version",
        "cudnn_version",
        "transformers_version",
    ):
        if software.get(key) != runtime_constraints.get(key):
            raise RuntimeError(f"live {key} differs from the frozen runtime")
    live_hostname = socket.gethostname().strip().lower()
    if (
        CONTAINER_HOSTNAME_PATTERN.fullmatch(live_hostname) is None
        or not container_id.startswith(live_hostname)
    ):
        raise RuntimeError("live container hostname is not a prefix of the full container ID")
    if args.device != "cuda:0":
        raise ValueError("the one-visible-H200 audit requires --device cuda:0")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()[:1000]}")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("spatial-reference audit requires exactly one visible GPU")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 5:
        raise RuntimeError("nvidia-smi returned an unexpected GPU identity row")
    gpu_index, gpu_name, gpu_uuid, pci_bus_id, driver_version = fields
    try:
        physical_gpu_index = int(gpu_index)
    except ValueError as error:
        raise RuntimeError("nvidia-smi returned a non-integer GPU index") from error
    if physical_gpu_index < 0 or gpu_name != "NVIDIA H200":
        raise RuntimeError("visible cuda:0 must map to exactly one NVIDIA H200")
    if driver_version != runtime_constraints.get("nvidia_driver_version"):
        raise RuntimeError("live NVIDIA driver differs from the frozen runtime")
    return {
        "alias": str(args.host_alias),
        "hostname": str(args.host_hostname),
        "container_hostname": live_hostname,
        "container_id": container_id,
        "container_image_digest": image_digest,
        "device": str(args.device),
        "visible_gpu_count": 1,
        "cuda_visible_ordinal": 0,
        "nvidia_smi_gpu_index": physical_gpu_index,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "gpu_pci_bus_id": pci_bus_id,
        "nvidia_driver_version": driver_version,
        "software": software,
        "scientific_environment_audit": scientific_environment_audit,
        "nvidia_smi_query": (
            "nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version "
            "--format=csv,noheader,nounits"
        ),
    }


def _execution_inputs(
    *,
    repository_root: Path,
    config_path: Path,
    raw_archive: Path,
    derived_root: Path,
    model_dir: Path,
) -> dict[str, str]:
    return {
        "repository_root": str(repository_root),
        "config_path": str(config_path),
        "parent_raw_archive": str(raw_archive),
        "derived_artifact_root": str(derived_root),
        "model_dir": str(model_dir),
    }


def _initial_ledger(
    *,
    layout: AttemptLayout,
    source_git_commit: str,
    config_sha256: str,
    host: Mapping[str, Any],
    execution_inputs: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "attempt_id": ATTEMPT_ID,
        "status": "ACTIVE_SPATIAL_REFERENCE_AUDIT_ATTEMPT",
        "canonical_attempt_root": str(layout.root),
        "canonical_global_ledger": str(layout.global_ledger),
        "source_git_commit": source_git_commit,
        "config_sha256": config_sha256,
        "host": dict(host),
        "execution_inputs": dict(execution_inputs),
        "profile_order": list(PROFILE_ORDER),
        "next_profile_position": 0,
        "operation_counts": _zero_operation_counts(),
        "profiles": [
            {
                "position": position,
                "profile_id": profile_id,
                "status": "PENDING",
                "operation_counts": _zero_operation_counts(),
                "states": [],
            }
            for position, profile_id in enumerate(PROFILE_ORDER)
        ],
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "retry_allowed": False,
        "deletion_allowed": False,
    }


def _validate_existing_ledger(
    value: Mapping[str, Any],
    *,
    layout: AttemptLayout,
    source_git_commit: str,
    config_sha256: str,
    host: Mapping[str, Any],
    execution_inputs: Mapping[str, str],
) -> None:
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "attempt_id": ATTEMPT_ID,
        "canonical_attempt_root": str(layout.root),
        "canonical_global_ledger": str(layout.global_ledger),
        "source_git_commit": source_git_commit,
        "config_sha256": config_sha256,
        "host": dict(host),
        "execution_inputs": dict(execution_inputs),
        "profile_order": list(PROFILE_ORDER),
        "retry_allowed": False,
        "deletion_allowed": False,
    }
    if any(value.get(key) != expected for key, expected in expected_identity.items()):
        raise PermissionError("existing spatial-reference attempt identity differs")
    profiles = value.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != len(PROFILE_ORDER):
        raise ValueError("global attempt ledger profile inventory drifted")
    for position, (record, profile_id) in enumerate(zip(profiles, PROFILE_ORDER, strict=True)):
        if not isinstance(record, Mapping):
            raise TypeError("global attempt profile record must be a mapping")
        if record.get("position") != position or record.get("profile_id") != profile_id:
            raise ValueError("global attempt profile order drifted")
    counts = value.get("operation_counts")
    if not isinstance(counts, Mapping):
        raise TypeError("global attempt operation counts are missing")
    _add_operation_counts(counts, _zero_operation_counts())


def _ensure_attempt_layout(
    *,
    layout: AttemptLayout,
    profile_id: str,
    source_git_commit: str,
    config_sha256: str,
    host: Mapping[str, Any],
    execution_inputs: Mapping[str, str],
    invocation_argv: Sequence[str],
) -> dict[str, Any]:
    root_exists = layout.root.exists()
    ledger_exists = layout.global_ledger.exists()
    if root_exists != ledger_exists:
        raise PermissionError("incomplete canonical attempt exists; retry is forbidden")
    if not root_exists:
        if profile_id != PROFILE_ORDER[0]:
            raise PermissionError("the first canonical profile must be bf16_auto")
        layout.root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _fsync_directory(layout.root.parent)
        _mkdir_exclusive_durable(layout.root)
        _mkdir_exclusive_durable(layout.profiles_root)
        _write_json_exclusive(
            layout.attempt_start,
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "attempt_id": ATTEMPT_ID,
                "status": "STARTED_NO_RETRY_OR_DELETION",
                "source_git_commit": source_git_commit,
                "config_sha256": config_sha256,
                "host": dict(host),
                "execution_inputs": dict(execution_inputs),
                "profile_order": list(PROFILE_ORDER),
                "first_invocation_argv": list(invocation_argv),
                "started_at_utc": _utc_now(),
            },
        )
        ledger = _initial_ledger(
            layout=layout,
            source_git_commit=source_git_commit,
            config_sha256=config_sha256,
            host=host,
            execution_inputs=execution_inputs,
        )
        _write_json_exclusive(
            layout.profiles_root / f".{0:03d}-{profile_id}.claim.json",
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "attempt_id": ATTEMPT_ID,
                "status": "PROFILE_INVOCATION_CLAIMED_NO_RETRY",
                "position": 0,
                "profile_id": profile_id,
                "invocation_argv": list(invocation_argv),
                "claimed_at_utc": _utc_now(),
            },
        )
        _write_json_exclusive(layout.global_ledger, ledger)
        return ledger
    if not layout.root.is_dir() or not layout.attempt_start.is_file():
        raise PermissionError("canonical attempt layout is incomplete; retry is forbidden")
    ledger = _load_json_object(layout.global_ledger)
    _validate_existing_ledger(
        ledger,
        layout=layout,
        source_git_commit=source_git_commit,
        config_sha256=config_sha256,
        host=host,
        execution_inputs=execution_inputs,
    )
    position = ledger.get("next_profile_position")
    if type(position) is not int or position < 0 or position >= len(PROFILE_ORDER):
        raise PermissionError("canonical attempt is terminal; no profile retry is allowed")
    profiles = ledger["profiles"]
    if (
        ledger.get("status") != "ACTIVE_SPATIAL_REFERENCE_AUDIT_ATTEMPT"
        or profiles[position].get("status") != "PENDING"
        or PROFILE_ORDER[position] != profile_id
    ):
        raise PermissionError("profile order drifted or a profile retry was attempted")
    claim_path = layout.profiles_root / f".{position:03d}-{profile_id}.claim.json"
    if claim_path.exists():
        raise PermissionError("profile invocation was already claimed; retry is forbidden")
    _write_json_exclusive(
        claim_path,
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "attempt_id": ATTEMPT_ID,
            "status": "PROFILE_INVOCATION_CLAIMED_NO_RETRY",
            "position": position,
            "profile_id": profile_id,
            "invocation_argv": list(invocation_argv),
            "claimed_at_utc": _utc_now(),
        },
    )
    return ledger


def _mutate_ledger(
    layout: AttemptLayout,
    mutation: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    ledger = _load_json_object(layout.global_ledger)
    mutation(ledger)
    ledger["updated_at_utc"] = _utc_now()
    _replace_json_durable(layout.global_ledger, ledger)
    return ledger


def _begin_profile(
    *,
    layout: AttemptLayout,
    ledger: Mapping[str, Any],
    profile: Any,
    invocation_argv: Sequence[str],
) -> ProfileLayout:
    position = ledger.get("next_profile_position")
    if type(position) is not int or position < 0 or position >= len(PROFILE_ORDER):
        raise PermissionError("canonical attempt is terminal; no profile retry is allowed")
    if PROFILE_ORDER[position] != profile.profile_id:
        raise PermissionError("profiles must execute once in the frozen order")
    if ledger.get("status") != "ACTIVE_SPATIAL_REFERENCE_AUDIT_ATTEMPT":
        raise PermissionError("canonical attempt is not active")
    profiles = ledger.get("profiles")
    if not isinstance(profiles, list) or profiles[position].get("status") != "PENDING":
        raise PermissionError("profile already started; retry is forbidden")
    if position > 0:
        previous = profiles[position - 1]
        previous_terminal = Path(str(previous.get("terminal_path", "")))
        if previous.get("status") != "COMPLETED" or not previous_terminal.is_file():
            raise PermissionError("the preceding profile lacks a durable terminal record")
    profile_root = layout.profiles_root / f"{position:03d}-{profile.profile_id}"
    profile_layout = ProfileLayout(
        position=position,
        root=profile_root,
        states_root=profile_root / "states",
        claim=layout.profiles_root / f".{position:03d}-{profile.profile_id}.claim.json",
        start=profile_root / "start.json",
        terminal=profile_root / "terminal.json",
    )
    claim = _load_json_object(profile_layout.claim)
    if (
        claim.get("status") != "PROFILE_INVOCATION_CLAIMED_NO_RETRY"
        or claim.get("position") != position
        or claim.get("profile_id") != profile.profile_id
        or claim.get("invocation_argv") != list(invocation_argv)
    ):
        raise PermissionError("profile invocation claim differs from the live invocation")
    _mkdir_exclusive_durable(profile_layout.root)
    _mkdir_exclusive_durable(profile_layout.states_root)
    _write_json_exclusive(
        profile_layout.start,
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "attempt_id": ATTEMPT_ID,
            "status": "PROFILE_STARTED_NO_RETRY",
            "position": position,
            "profile_id": profile.profile_id,
            "state_indices": list(profile.state_indices),
            "invocation_argv": list(invocation_argv),
            "started_at_utc": _utc_now(),
        },
    )

    def mutation(value: dict[str, Any]) -> None:
        records = value["profiles"]
        record = records[position]
        if record["status"] != "PENDING":
            raise PermissionError("profile ledger was already claimed")
        record.update(
            {
                "status": "STARTED",
                "start_path": str(profile_layout.start),
                "terminal_path": str(profile_layout.terminal),
                "invocation_argv": list(invocation_argv),
                "state_indices": list(profile.state_indices),
                "started_at_utc": _utc_now(),
            }
        )

    _mutate_ledger(layout, mutation)
    return profile_layout


def _state_layout(
    profile_layout: ProfileLayout,
    *,
    ordinal: int,
    state_index: int,
) -> StateLayout:
    root = profile_layout.states_root / f"{ordinal:03d}-{state_index:03d}"
    return StateLayout(
        ordinal=ordinal,
        root=root,
        start=root / "start.json",
        terminal=root / "terminal.json",
    )


def _begin_state(
    *,
    attempt_layout: AttemptLayout,
    profile_layout: ProfileLayout,
    profile_id: str,
    state_layout: StateLayout,
    state: Any,
    expected_counts: Mapping[str, int],
) -> None:
    _mkdir_exclusive_durable(state_layout.root)
    _write_json_exclusive(
        state_layout.start,
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "attempt_id": ATTEMPT_ID,
            "status": "STATE_STARTED_NO_RETRY",
            "profile_id": profile_id,
            "ordinal": state_layout.ordinal,
            "index": state.index,
            "role": state.role,
            "state_id": state.state_id,
            "expected_operation_counts": dict(expected_counts),
            "started_at_utc": _utc_now(),
        },
    )

    def mutation(value: dict[str, Any]) -> None:
        profile_record = value["profiles"][profile_layout.position]
        if profile_record["status"] != "STARTED":
            raise PermissionError("profile is not active for state start")
        states = profile_record["states"]
        if len(states) != state_layout.ordinal:
            raise PermissionError("state order drifted or a retry was attempted")
        states.append(
            {
                "ordinal": state_layout.ordinal,
                "index": state.index,
                "state_id": state.state_id,
                "status": "STARTED",
                "start_path": str(state_layout.start),
                "terminal_path": str(state_layout.terminal),
                "operation_counts": _zero_operation_counts(),
            }
        )

    _mutate_ledger(attempt_layout, mutation)


def _finish_state(
    *,
    attempt_layout: AttemptLayout,
    profile_layout: ProfileLayout,
    state_layout: StateLayout,
    status: str,
    operation_counts: Mapping[str, int],
) -> None:
    if status not in {"COMPLETED", "FAILED"}:
        raise ValueError("state terminal status is invalid")

    def mutation(value: dict[str, Any]) -> None:
        profile_record = value["profiles"][profile_layout.position]
        record = profile_record["states"][state_layout.ordinal]
        if record["status"] != "STARTED":
            raise PermissionError("state ledger was already terminal")
        record["status"] = status
        record["operation_counts"] = dict(operation_counts)
        record["ended_at_utc"] = _utc_now()
        profile_record["operation_counts"] = _add_operation_counts(
            profile_record["operation_counts"],
            operation_counts,
        )
        value["operation_counts"] = _add_operation_counts(
            value["operation_counts"],
            operation_counts,
        )

    _mutate_ledger(attempt_layout, mutation)


def _actual_generated_ids(metadata: Mapping[str, Any]) -> list[int]:
    values = metadata.get("generated_token_ids")
    if not isinstance(values, list) or any(type(item) is not int or item < 0 for item in values):
        raise TypeError("runtime metadata must contain actual generated token ids")
    expected = metadata.get("generated_token_ids_sha256")
    observed = hashlib.sha256(canonical_json_bytes(values)).hexdigest()
    if expected != observed:
        raise ValueError("actual generated token ids differ from their runtime SHA256")
    return list(values)


def _run_state(
    *,
    runtime: GUIOwlSpatialAuditRuntime,
    artifact: Any,
    state: Any,
    mismatch: Any,
    profile: Any,
    attempt_layout: AttemptLayout,
    profile_layout: ProfileLayout,
    state_layout: StateLayout,
) -> dict[str, Any]:
    expected_counts = {
        "generation_calls": profile.generation_repeat_count,
        "teacher_forwards": (
            profile.shared_prefix_forward_repeat_count
            + profile.full_parent_action_forward_count
        ),
        "confirm_state_accesses": 0,
        "restoration_coalition_constructions": 0,
        "gate_training_examples": 0,
    }
    _begin_state(
        attempt_layout=attempt_layout,
        profile_layout=profile_layout,
        profile_id=profile.profile_id,
        state_layout=state_layout,
        state=state,
        expected_counts=expected_counts,
    )
    operation_counts = _zero_operation_counts()
    started_at = _utc_now()
    started = time.perf_counter()
    try:
        messages = build_full45_messages(
            artifact,
            state,
            "reference",
            _decode_rgb_image,
        )
        parent_tokens = retokenize_parent_mismatch(
            mismatch,
            runtime.processor.tokenizer,
        )
        generations = []
        for repeat_index in range(profile.generation_repeat_count):
            operation_counts["generation_calls"] += 1
            try:
                generated = runtime.generate_native_action(messages)
            except GUIOwlV21GenerationParseError as error:
                metadata = dict(error.metadata)
                generations.append(
                    {
                        "repeat_index": repeat_index + 1,
                        "output_text": error.output_text,
                        "metadata": metadata,
                        "generated_token_ids": _actual_generated_ids(metadata),
                        "canonical_action": None,
                        "parse_error_type": error.parse_error_type,
                        "parse_error_message": error.parse_error_message,
                    }
                )
                continue
            metadata = dict(generated.metadata)
            generations.append(
                {
                    "repeat_index": repeat_index + 1,
                    "output_text": generated.output_text,
                    "metadata": metadata,
                    "generated_token_ids": _actual_generated_ids(metadata),
                    "canonical_action": _action_arguments(
                        generated.parsed_output.canonical_action
                    ),
                    "parse_error_type": None,
                    "parse_error_message": None,
                }
            )
        parsed = [
            record for record in generations if record["canonical_action"] is not None
        ]
        exact_token_stable = (
            len(parsed) == profile.generation_repeat_count
            and len({tuple(record["generated_token_ids"]) for record in parsed}) == 1
        )
        exact_action_stable = (
            len(parsed) == profile.generation_repeat_count
            and len(
                {
                    json.dumps(record["canonical_action"], sort_keys=True)
                    for record in parsed
                }
            )
            == 1
        )
        parent_actions = tuple(
            action_from_mapping(action) for action in mismatch.actions
        )
        shared_prefix_repeats = []
        divergence_index: int | None = None
        for repeat_index in range(profile.shared_prefix_forward_repeat_count):
            operation_counts["teacher_forwards"] += 1
            diagnostic = dict(
                runtime.shared_prefix_parent_pair_diagnostics(
                    messages,
                    parent_actions,
                )
            )
            observed_divergence = diagnostic.get("first_divergent_token_index")
            if type(observed_divergence) is not int or observed_divergence < 0:
                raise ValueError("shared-prefix diagnostic lacks a valid divergence index")
            if (
                diagnostic.get("teacher_token_ids") != parent_tokens.get("token_ids")
                or observed_divergence
                != parent_tokens.get("first_divergent_token_index")
            ):
                raise RuntimeError(
                    "shared-prefix teacher path differs from retokenized parent outputs"
                )
            if divergence_index is None:
                divergence_index = observed_divergence
            elif observed_divergence != divergence_index:
                raise RuntimeError("shared-prefix divergence changed across repeats")
            shared_prefix_repeats.append(
                {"repeat_index": repeat_index + 1, **diagnostic}
            )
        if divergence_index is None:
            raise RuntimeError("shared-prefix diagnostics did not execute")
        if profile.full_parent_action_forward_count != len(parent_actions):
            raise ValueError("full-parent forward count must equal the two parent actions")
        full_action_diagnostics = []
        for action_index, action in enumerate(parent_actions):
            operation_counts["teacher_forwards"] += 1
            full_diagnostic = dict(
                runtime.full_parent_action_diagnostic(
                    messages,
                    action,
                    divergence_index=divergence_index,
                )
            )
            if full_diagnostic.get("divergence_index") != divergence_index:
                raise RuntimeError("full-parent diagnostic used the wrong divergence")
            full_action_diagnostics.append(
                {
                    "parent_action_index": action_index,
                    "parent_action": dict(mismatch.actions[action_index]),
                    **full_diagnostic,
                }
            )
        if operation_counts != expected_counts:
            raise RuntimeError("state operation counts differ from the frozen schedule")
        record = {
            "index": state.index,
            "role": state.role,
            "state_id": state.state_id,
            "candidate_event_step_ids": list(state.candidate_event_step_ids),
            "parent_state_member_sha256": mismatch.state_member_sha256,
            "parent_actions": [dict(action) for action in mismatch.actions],
            "parent_output_texts": list(mismatch.output_texts),
            "parent_generated_token_ids_sha256": list(
                mismatch.generated_token_ids_sha256
            ),
            "parent_retokenization": parent_tokens,
            "coordinate_delta": coordinate_delta(mismatch),
            "generations": generations,
            "exact_generated_token_sequence_stable": exact_token_stable,
            "exact_canonical_action_stable": exact_action_stable,
            "shared_prefix_parent_pair_repeats": shared_prefix_repeats,
            "full_parent_action_diagnostics": full_action_diagnostics,
            "operation_counts": dict(operation_counts),
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now(),
            "duration_seconds": time.perf_counter() - started,
        }
        terminal = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "attempt_id": ATTEMPT_ID,
            "status": "COMPLETED_SPATIAL_REFERENCE_AUDIT_STATE",
            "profile_id": profile.profile_id,
            "ordinal": state_layout.ordinal,
            "operation_counts": dict(operation_counts),
            "record": record,
        }
        _write_json_exclusive(state_layout.terminal, terminal)
        _finish_state(
            attempt_layout=attempt_layout,
            profile_layout=profile_layout,
            state_layout=state_layout,
            status="COMPLETED",
            operation_counts=operation_counts,
        )
        return record
    except BaseException as error:
        if not state_layout.terminal.exists():
            _write_json_exclusive(
                state_layout.terminal,
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol_id": PROTOCOL_ID,
                    "attempt_id": ATTEMPT_ID,
                    "status": "FAILED_SPATIAL_REFERENCE_AUDIT_STATE",
                    "profile_id": profile.profile_id,
                    "ordinal": state_layout.ordinal,
                    "index": state.index,
                    "state_id": state.state_id,
                    "operation_counts": dict(operation_counts),
                    "failure": _exception_record(error),
                    "started_at_utc": started_at,
                    "ended_at_utc": _utc_now(),
                    "duration_seconds": time.perf_counter() - started,
                },
            )
            _finish_state(
                attempt_layout=attempt_layout,
                profile_layout=profile_layout,
                state_layout=state_layout,
                status="FAILED",
                operation_counts=operation_counts,
            )
        raise


def _profile_failure_counts(profile_layout: ProfileLayout) -> dict[str, int]:
    counts = _zero_operation_counts()
    if not profile_layout.states_root.is_dir():
        return counts
    for state_root in sorted(profile_layout.states_root.iterdir()):
        terminal = state_root / "terminal.json"
        if not terminal.is_file():
            continue
        value = _load_json_object(terminal)
        observed = value.get("operation_counts")
        if not isinstance(observed, Mapping):
            raise ValueError("state terminal lacks operation counts")
        counts = _add_operation_counts(counts, observed)
    return counts


def _finish_profile_ledger(
    *,
    attempt_layout: AttemptLayout,
    profile_layout: ProfileLayout,
    status: str,
    operation_counts: Mapping[str, int],
) -> None:
    if status not in {"COMPLETED", "FAILED"}:
        raise ValueError("profile terminal status is invalid")

    def mutation(value: dict[str, Any]) -> None:
        record = value["profiles"][profile_layout.position]
        if record["status"] != "STARTED":
            raise PermissionError("profile ledger was already terminal")
        if record["operation_counts"] != dict(operation_counts):
            raise RuntimeError("profile ledger operation counts differ from state terminals")
        record["status"] = status
        record["ended_at_utc"] = _utc_now()
        if status == "FAILED":
            value["status"] = "FAILED_SPATIAL_REFERENCE_AUDIT_ATTEMPT"
            value["failed_profile_id"] = record["profile_id"]
        else:
            next_position = profile_layout.position + 1
            value["next_profile_position"] = next_position
            if next_position == len(PROFILE_ORDER):
                value["status"] = "COMPLETED_SPATIAL_REFERENCE_AUDIT_ATTEMPT"
                value["completed_at_utc"] = _utc_now()

    _mutate_ledger(attempt_layout, mutation)


def run_profile(
    args: argparse.Namespace,
    *,
    invocation_argv: Sequence[str],
) -> dict[str, Any]:
    root = _absolute(args.repository_root, name="repository-root")
    config_path = _absolute(args.config, name="config")
    raw_archive = _absolute(args.parent_raw_archive, name="parent-raw-archive")
    derived_root = _absolute(args.derived_artifact_root, name="derived-artifact-root")
    model_dir = _absolute(args.model_dir, name="model-dir")
    if config_path != root / "code/configs/spatial_reference_audit_v1.json":
        raise ValueError("spatial-reference audit config path is not canonical")
    git = dict(validate_clean_pushed_main(root))
    if git.get("commit") != args.source_git_commit:
        raise ValueError("audit source commit differs from clean pushed main")
    config = load_and_validate_config(config_path)
    validate_repository_inputs(root, config)
    attempt_layout = _validated_attempt_identity(config)
    attempt_identity = config["attempt_identity"]
    if (
        args.host_alias != attempt_identity["canonical_host_alias"]
        or args.host_hostname != attempt_identity["canonical_host_hostname"]
        or args.device != attempt_identity["canonical_device"]
    ):
        raise ValueError("CLI host/device differs from the frozen attempt identity")
    runtime_constraints = config["runtime_constraints"]
    host = _live_execution_identity(
        args,
        runtime_constraints=runtime_constraints,
    )
    profile = profile_by_id(config, args.profile_id)
    if profile.profile_id not in PROFILE_ORDER:
        raise ValueError("profile is outside the frozen canonical order")
    inputs = config["inputs"]
    mismatches = load_parent_mismatches(
        raw_archive=raw_archive,
        fixture_path=root / inputs["parent_mismatch_fixture"]["path"],
        config=config,
    )
    mismatch_by_index = {record.index: record for record in mismatches}
    artifact = load_validated_screening_artifact(
        artifact_root=derived_root,
        backend_config_path=root / inputs["ocr_backend_config"]["path"],
        scientific_config_path=root / inputs["scientific_config"]["path"],
        selection_manifest_path=root / inputs["selection_manifest"]["path"],
        expected_artifact_tree_sha256=inputs["derived_artifact"][
            "artifact_tree_sha256"
        ],
    )
    state_by_index = {state.index: state for state in artifact.states}
    if tuple(sorted(mismatch_by_index)) != tuple(
        config["mismatch_denominator"]["state_indices"]
    ):
        raise ValueError("parent mismatch denominator drifted after artifact load")
    execution_inputs = _execution_inputs(
        repository_root=root,
        config_path=config_path,
        raw_archive=raw_archive,
        derived_root=derived_root,
        model_dir=model_dir,
    )
    config_sha = sha256_file(config_path)
    ledger = _ensure_attempt_layout(
        layout=attempt_layout,
        profile_id=profile.profile_id,
        source_git_commit=args.source_git_commit,
        config_sha256=config_sha,
        host=host,
        execution_inputs=execution_inputs,
        invocation_argv=invocation_argv,
    )
    profile_layout = _begin_profile(
        layout=attempt_layout,
        ledger=ledger,
        profile=profile,
        invocation_argv=invocation_argv,
    )
    started_at = _utc_now()
    started = time.perf_counter()
    try:
        runtime = GUIOwlSpatialAuditRuntime(
            model_dir=model_dir,
            expected_snapshot_manifest=root
            / inputs["model_snapshot_manifest"]["path"],
            device=args.device,
            profile=profile,
        )
        runtime_software = {
            key: runtime.metadata.get(key)
            for key in (
                "python_version",
                "torch_version",
                "torch_cuda_version",
                "cudnn_version",
                "transformers_version",
            )
        }
        if (
            runtime_software != host["software"]
            or runtime.metadata.get("scientific_environment_audit")
            != host["scientific_environment_audit"]
        ):
            raise RuntimeError("loaded policy runtime differs from the pre-claim identity")
        records = []
        for ordinal, index in enumerate(profile.state_indices):
            records.append(
                _run_state(
                    runtime=runtime,
                    artifact=artifact,
                    state=state_by_index[index],
                    mismatch=mismatch_by_index[index],
                    profile=profile,
                    attempt_layout=attempt_layout,
                    profile_layout=profile_layout,
                    state_layout=_state_layout(
                        profile_layout,
                        ordinal=ordinal,
                        state_index=index,
                    ),
                )
            )
        expected_counts = profile_operation_counts(profile)
        observed_counts = _profile_failure_counts(profile_layout)
        if observed_counts != expected_counts:
            raise RuntimeError("profile operation counts differ from the frozen schedule")
        exact_token_count = sum(
            record["exact_generated_token_sequence_stable"] for record in records
        )
        exact_action_count = sum(
            record["exact_canonical_action_stable"] for record in records
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "attempt_id": ATTEMPT_ID,
            "status": "COMPLETED_SPATIAL_REFERENCE_AUDIT_PROFILE",
            "profile_position": profile_layout.position,
            "profile": {
                "profile_id": profile.profile_id,
                "dtype": profile.dtype,
                "attention_implementation": profile.attention_implementation,
                "deterministic_algorithms": profile.deterministic_algorithms,
                "eager_numerical_controls": profile.eager_numerical_controls,
                "seed": profile.seed,
                "state_indices": list(profile.state_indices),
                "generation_repeat_count": profile.generation_repeat_count,
                "shared_prefix_forward_repeat_count": (
                    profile.shared_prefix_forward_repeat_count
                ),
                "full_parent_action_forward_count": (
                    profile.full_parent_action_forward_count
                ),
            },
            "source_git_commit": args.source_git_commit,
            "git_identity": git,
            "invocation_argv": list(invocation_argv),
            "config_path": str(config_path),
            "config_sha256": config_sha,
            "parent_raw_archive_path": str(raw_archive),
            "parent_raw_archive_sha256": sha256_file(raw_archive),
            "derived_artifact_root": str(derived_root),
            "derived_artifact_tree_sha256": artifact.artifact_tree_sha256,
            "attempt_layout": {
                "root": str(attempt_layout.root),
                "global_ledger": str(attempt_layout.global_ledger),
                "profile_start": str(profile_layout.start),
                "profile_terminal": str(profile_layout.terminal),
            },
            "host": dict(host),
            "runtime_metadata": dict(runtime.metadata),
            "records": records,
            "metrics": {
                "state_count": len(records),
                "exact_generated_token_sequence_stable_count": exact_token_count,
                "exact_canonical_action_stable_count": exact_action_count,
                "exact_generated_token_sequence_stable_rate": exact_token_count
                / len(records),
                "exact_canonical_action_stable_rate": exact_action_count / len(records),
            },
            "operation_counts": observed_counts,
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now(),
            "duration_seconds": time.perf_counter() - started,
            "confirm_policy_output_accessed": False,
            "restoration_work_performed": False,
            "gate_training_performed": False,
        }
        result["scientific_payload_sha256"] = canonical_profile_payload_sha256(
            {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "started_at_utc",
                    "ended_at_utc",
                    "duration_seconds",
                    "scientific_payload_sha256",
                }
            }
        )
        _write_json_exclusive(profile_layout.terminal, result)
        _finish_profile_ledger(
            attempt_layout=attempt_layout,
            profile_layout=profile_layout,
            status="COMPLETED",
            operation_counts=observed_counts,
        )
        return result
    except BaseException as error:
        counts = _profile_failure_counts(profile_layout)
        if not profile_layout.terminal.exists():
            _write_json_exclusive(
                profile_layout.terminal,
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol_id": PROTOCOL_ID,
                    "attempt_id": ATTEMPT_ID,
                    "status": "FAILED_SPATIAL_REFERENCE_AUDIT_PROFILE",
                    "profile_position": profile_layout.position,
                    "profile_id": profile.profile_id,
                    "invocation_argv": list(invocation_argv),
                    "operation_counts": counts,
                    "failure": _exception_record(error),
                    "started_at_utc": started_at,
                    "ended_at_utc": _utc_now(),
                    "duration_seconds": time.perf_counter() - started,
                },
            )
            _finish_profile_ledger(
                attempt_layout=attempt_layout,
                profile_layout=profile_layout,
                status="FAILED",
                operation_counts=counts,
            )
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--parent-raw-archive", required=True)
    parser.add_argument("--derived-artifact-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--profile-id", choices=list(PROFILE_ORDER), required=True)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--host-hostname", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-image-digest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    invocation_argv = [str(Path(__file__).resolve()), *parsed_argv]
    args = _build_parser().parse_args(parsed_argv)
    result = run_profile(args, invocation_argv=invocation_argv)
    print(
        json.dumps(
            {
                "status": result["status"],
                "profile_id": result["profile"]["profile_id"],
                "profile_terminal": result["attempt_layout"]["profile_terminal"],
                "scientific_payload_sha256": result["scientific_payload_sha256"],
                "metrics": result["metrics"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
