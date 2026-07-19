"""Fresh four-H200 execution envelope for the D1 action-stability diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.set_utility_action_stability_contract_v1 import (
    CANONICAL_CONFIG_PATH,
    ENCODE_CALL_CEILING,
    GENERATION_CALL_CEILING,
    STATE_IDS,
    WORKER_COUNT,
    canonical_json_bytes,
    canonical_pretty_json_bytes,
    load_action_stability_source_v1_contract,
    sha256_bytes,
)
from causalcache.set_utility_action_stability_diagnostic_v1 import (
    CONDITION_ORDER,
    WORKER_INDEX_BY_STATE,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_action_stability_execution_envelope_v1"
AUTHORIZED_STATUS = "AUTHORIZED_EXACT_36_CALL_ACTION_STABILITY_EXECUTION_V1"
VALIDATION_STATUS = "VALID_SET_UTILITY_ACTION_STABILITY_EXECUTION_ENVELOPE_V1"
PREFLIGHT_PROTOCOL_ID = "causalcache_set_utility_action_stability_preflight_v1"
PREFLIGHT_STATUS = "PASSED_FRESH_TEN_SECOND_FOUR_H200_PREFLIGHT_V1"
CANONICAL_GIT_ENVELOPE_PATH = (
    "code/configs/causalcache_set_utility_action_stability_"
    "diagnostic_v1_execution.json"
)
PARENT_EXECUTION_ENVELOPE_SHA256 = (
    "dd0e64fb40bd39f839a1df91240a1e9a1a528a0bbfe9131d49ba726e8ecf4e3a"
)
SOURCE_BRANCH = "main"
SOURCE_REMOTE = "origin"
PROFILE_ORDER = ("auto", "eager")
GPU_NAME = "NVIDIA H200"
PREFLIGHT_MINIMUM_SECONDS = 10
PREFLIGHT_MAX_AGE_SECONDS = 15 * 60
PERSISTENT_DATA_ROOT = Path("/data")
WORKER_ENTRYPOINT = "code/scripts/run_set_utility_action_stability_worker_v1.py"
AGGREGATE_ENTRYPOINT = "code/scripts/aggregate_set_utility_action_stability_v1.py"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{16,}")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_DRIVER = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}")
_HOSTS = {
    "hyper00": "node-radixark-16-0000",
    "hyper01": "node-radixark-16-0001",
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one mapping")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be one sequence")
    return value


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(raw: str) -> None:
        raise ValueError(f"{label} contains non-finite value {raw}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be whole-second UTC RFC3339")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be whole-second UTC RFC3339") from error
    if parsed.tzinfo != timezone.utc or parsed.microsecond:
        raise ValueError(f"{label} must be whole-second UTC RFC3339")
    return parsed


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _absolute(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        raise ValueError(f"{label} must be an absolute normalized path")
    return path


def _regular_binding(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an absolute regular non-symlink file")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        digest = hashlib.sha256()
        size = 0
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        size != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError(f"{label} changed while being hashed")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": size}


def canonical_run_layout(run_root: str | Path) -> dict[str, Any]:
    root = _absolute(str(run_root), label="D1 run root")
    try:
        relative = root.relative_to(PERSISTENT_DATA_ROOT)
    except ValueError as error:
        raise ValueError("D1 run root must be under persistent /data") from error
    if not relative.parts:
        raise ValueError("D1 run root must be task-specific")
    workers = []
    for worker_index in range(WORKER_COUNT):
        worker_root = root / "workers" / f"worker-{worker_index:02d}"
        profiles = {}
        for profile in PROFILE_ORDER:
            profiles[profile] = {
                "attempt_path": str(worker_root / f"{profile}-attempt.json"),
                "stderr_log_path": str(
                    root / "logs" / f"{profile}-worker-{worker_index:02d}.stderr.log"
                ),
                "stdout_log_path": str(
                    root / "logs" / f"{profile}-worker-{worker_index:02d}.stdout.log"
                ),
                "terminal_path": str(worker_root / f"{profile}-terminal.json"),
            }
        workers.append(
            {"profiles": profiles, "worker_index": worker_index}
        )
    return {
        "aggregate_result_path": str(root / "aggregate.json"),
        "aggregate_stderr_log_path": str(root / "logs" / "aggregate.stderr.log"),
        "aggregate_stdout_log_path": str(root / "logs" / "aggregate.stdout.log"),
        "envelope_path": str(root / "execution-envelope.json"),
        "preflight_evidence_path": str(root / "preflight-evidence.json"),
        "run_root": str(root),
        "workers": workers,
    }


def _normalized_gpus(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(values) != WORKER_COUNT:
        raise ValueError("D1 preflight requires exactly four GPUs")
    normalized = []
    for visible_index, raw in enumerate(values):
        value = _mapping(raw, label=f"D1 GPU {visible_index}")
        expected = {
            "host_index",
            "name",
            "sample_window_seconds",
            "sampled_utilization_percent",
            "total_memory_bytes",
            "uuid",
            "visible_index",
        }
        if set(value) != expected:
            raise ValueError("D1 GPU preflight schema drifted")
        samples = _sequence(
            value["sampled_utilization_percent"], label="D1 GPU utilization samples"
        )
        if (
            value["visible_index"] != visible_index
            or type(value["host_index"]) is not int
            or value["host_index"] < 0
            or value["name"] != GPU_NAME
            or not isinstance(value["uuid"], str)
            or _GPU_UUID.fullmatch(value["uuid"]) is None
            or type(value["total_memory_bytes"]) is not int
            or value["total_memory_bytes"] <= 0
            or not isinstance(value["sample_window_seconds"], (int, float))
            or isinstance(value["sample_window_seconds"], bool)
            or float(value["sample_window_seconds"]) < PREFLIGHT_MINIMUM_SECONDS
            or len(samples) < 2
            or any(type(sample) is not int or sample != 0 for sample in samples)
        ):
            raise ValueError("D1 selected GPU did not pass the ten-second idle preflight")
        normalized.append(dict(value))
    if len({gpu["uuid"] for gpu in normalized}) != WORKER_COUNT or len(
        {gpu["host_index"] for gpu in normalized}
    ) != WORKER_COUNT:
        raise ValueError("D1 selected GPU identities must be unique")
    return normalized


def build_action_stability_preflight_evidence_v1(
    *,
    raw_cleanup_log_path: str | Path,
    started_at_utc: str,
    completed_at_utc: str,
    host_alias: str,
    hostname: str,
    container_id: str,
    driver_version: str,
    selected_gpus: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    started = _timestamp(started_at_utc, label="D1 preflight start")
    completed = _timestamp(completed_at_utc, label="D1 preflight completion")
    if (completed - started).total_seconds() < PREFLIGHT_MINIMUM_SECONDS:
        raise ValueError("D1 preflight must cover at least ten seconds")
    if _HOSTS.get(host_alias) != hostname:
        raise ValueError("D1 preflight host is not a registered Hyper H200 host")
    if _CONTAINER_ID.fullmatch(container_id) is None or _DRIVER.fullmatch(
        driver_version
    ) is None:
        raise ValueError("D1 preflight runtime identity is invalid")
    cleanup = _regular_binding(Path(raw_cleanup_log_path), label="D1 cleanup log")
    gpus = _normalized_gpus(selected_gpus)
    return {
        "cleanup": {**cleanup, "all_containers_allowed": True},
        "completed_at_utc": completed_at_utc,
        "container_id": container_id,
        "driver_version": driver_version,
        "gpus": gpus,
        "host_alias": host_alias,
        "hostname": hostname,
        "minimum_idle_window_seconds": PREFLIGHT_MINIMUM_SECONDS,
        "protocol_id": PREFLIGHT_PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": started_at_utc,
        "status": PREFLIGHT_STATUS,
    }


def _load_parent_envelope(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _regular_binding(path, label="parent throughput envelope")
    if binding["sha256"] != PARENT_EXECUTION_ENVELOPE_SHA256:
        raise ValueError("parent throughput envelope SHA256 drifted")
    payload = path.read_bytes()
    parent = _strict_json_object(payload, label="parent throughput envelope")
    if payload != canonical_pretty_json_bytes(parent):
        raise ValueError("parent throughput envelope is not canonical pretty JSON")
    parent_execution = _mapping(parent.get("execution"), label="parent execution")
    parent_layout = _mapping(
        parent_execution.get("output_layout"),
        label="parent output layout",
    )
    if parent_layout.get("envelope_path") != str(path):
        raise ValueError("parent throughput envelope must use its canonical run path")
    return parent, binding


def _parent_artifact_paths(parent: Mapping[str, Any]) -> tuple[str, str]:
    artifacts = _mapping(parent.get("artifacts"), label="parent artifacts")
    model = _mapping(artifacts.get("model"), label="parent model")
    processor = _mapping(artifacts.get("processor"), label="parent processor")
    return str(model.get("local_path")), str(processor.get("local_root"))


def _worker_state_ids(worker_index: int) -> list[str]:
    return [
        state_id
        for state_id in STATE_IDS
        if WORKER_INDEX_BY_STATE[state_id] == worker_index
    ]


def _canonical_argv(
    *,
    repository_root: Path,
    python_executable: str,
    envelope_path: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    worker_script = repository_root / WORKER_ENTRYPOINT
    aggregate_script = repository_root / AGGREGATE_ENTRYPOINT
    if not worker_script.is_file() or not aggregate_script.is_file():
        raise ValueError("D1 runner or aggregate entrypoint is missing")
    workers = []
    for profile in PROFILE_ORDER:
        for worker_index in range(WORKER_COUNT):
            workers.append(
                {
                    "argv": [
                        python_executable,
                        str(worker_script),
                        "--execution-envelope",
                        envelope_path,
                        "--profile",
                        profile,
                        "--worker-index",
                        str(worker_index),
                    ],
                    "profile": profile,
                    "worker_index": worker_index,
                }
            )
    aggregate = [
        python_executable,
        str(aggregate_script),
        "--execution-envelope",
        envelope_path,
    ]
    return workers, aggregate


def build_action_stability_envelope_v1(
    *,
    repository_root: str | Path,
    source_git_revision: str,
    run_root: str | Path,
    python_executable: str,
    preflight_evidence_path: str | Path,
    parent_envelope_path: str | Path,
    processor_root: str | Path,
    model_dir: str | Path,
    host_alias: str,
    hostname: str,
    container_id: str,
    container_image_reference: str,
    container_image_digest: str,
    driver_version: str,
    software_versions: Mapping[str, str],
    materialized_at_utc: str | None = None,
    source_contract_loader: Callable[..., Any] = (
        load_action_stability_source_v1_contract
    ),
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if not root.is_dir() or root.is_symlink() or _COMMIT.fullmatch(
        source_git_revision
    ) is None:
        raise ValueError("D1 repository/source revision is invalid")
    contract = source_contract_loader(repository_root=root)
    if contract.data["diagnostic"]["generation_call_ceiling"] != GENERATION_CALL_CEILING:
        raise ValueError("D1 source operation ceiling drifted")
    layout = canonical_run_layout(run_root)
    evidence_path = _absolute(str(preflight_evidence_path), label="D1 preflight evidence")
    if str(evidence_path) != layout["preflight_evidence_path"]:
        raise ValueError("D1 preflight evidence path is not canonical")
    evidence_binding = _regular_binding(evidence_path, label="D1 preflight evidence")
    evidence_payload = evidence_path.read_bytes()
    evidence = _strict_json_object(evidence_payload, label="D1 preflight evidence")
    if evidence_payload != canonical_pretty_json_bytes(evidence):
        raise ValueError("D1 preflight evidence is not canonical pretty JSON")
    normalized = build_action_stability_preflight_evidence_v1(
        raw_cleanup_log_path=evidence["cleanup"]["path"],
        started_at_utc=evidence["started_at_utc"],
        completed_at_utc=evidence["completed_at_utc"],
        host_alias=host_alias,
        hostname=hostname,
        container_id=container_id,
        driver_version=driver_version,
        selected_gpus=evidence["gpus"],
    )
    if evidence != normalized:
        raise ValueError("D1 preflight evidence reconstruction drifted")
    parent, parent_binding = _load_parent_envelope(
        _absolute(str(parent_envelope_path), label="parent throughput envelope")
    )
    embedded_model, embedded_processor = _parent_artifact_paths(parent)
    model = _absolute(str(model_dir), label="D1 model directory")
    processor = _absolute(str(processor_root), label="D1 processor root")
    if (
        str(model) != embedded_model
        or str(processor) != embedded_processor
        or not model.is_dir()
        or not processor.is_dir()
    ):
        raise ValueError("D1 local artifact paths differ from the parent envelope")
    if _HOSTS.get(host_alias) != hostname or evidence["host_alias"] != host_alias:
        raise ValueError("D1 host identity drifted")
    if evidence["container_id"] != container_id or _CONTAINER_ID.fullmatch(
        container_id
    ) is None:
        raise ValueError("D1 container identity drifted")
    if _IMAGE_DIGEST.fullmatch(container_image_digest) is None:
        raise ValueError("D1 image digest is invalid")
    required_versions = {
        "cuda_runtime_version",
        "python_version",
        "torch_version",
        "transformers_version",
    }
    if set(software_versions) != required_versions or any(
        not isinstance(value, str) or not value for value in software_versions.values()
    ):
        raise ValueError("D1 software-version inventory drifted")
    materialized_text = materialized_at_utc or utc_now()
    materialized = _timestamp(materialized_text, label="D1 materialization")
    completed = _timestamp(evidence["completed_at_utc"], label="D1 preflight completion")
    if materialized < completed or (
        materialized - completed
    ).total_seconds() > PREFLIGHT_MAX_AGE_SECONDS:
        raise ValueError("D1 preflight is stale at materialization")
    workers, aggregate_argv = _canonical_argv(
        repository_root=root,
        python_executable=python_executable,
        envelope_path=layout["envelope_path"],
    )
    gpu_identity = [
        {
            key: gpu[key]
            for key in ("host_index", "name", "total_memory_bytes", "uuid", "visible_index")
        }
        for gpu in evidence["gpus"]
    ]
    for worker in workers:
        worker_index = worker["worker_index"]
        gpu = gpu_identity[worker_index]
        worker.update(
            {
                "cuda_visible_devices": gpu["uuid"],
                "gpu_uuid": gpu["uuid"],
                "output": layout["workers"][worker_index]["profiles"][worker["profile"]],
                "python_path": str(root / "code"),
                "state_ids": _worker_state_ids(worker_index),
            }
        )
    source_binding = {
        "path": CANONICAL_CONFIG_PATH,
        "sha256": contract.config_sha256,
        "size_bytes": (root / CANONICAL_CONFIG_PATH).stat().st_size,
    }
    return {
        "artifacts": {
            "model_dir": str(model),
            "parent_execution_envelope": parent_binding,
            "processor_root": str(processor),
        },
        "authorization": {
            "generate_restoration_labels": False,
            "run_closed_loop": False,
            "run_exact_36_call_action_stability_diagnostic": True,
            "train_predictor": False,
        },
        "container": {
            "id": container_id,
            "image_digest": container_image_digest,
            "image_reference": container_image_reference,
        },
        "execution": {
            "aggregate": {
                "argv": aggregate_argv,
                "output": {
                    key: layout[key]
                    for key in (
                        "aggregate_result_path",
                        "aggregate_stderr_log_path",
                        "aggregate_stdout_log_path",
                    )
                },
            },
            "auto_all_four_terminals_required_before_any_eager_attempt": True,
            "condition_order": list(CONDITION_ORDER),
            "encode_call_ceiling": ENCODE_CALL_CEILING,
            "generation_call_ceiling": GENERATION_CALL_CEILING,
            "no_retry": True,
            "no_top_up": True,
            "output_layout": layout,
            "profiles": list(PROFILE_ORDER),
            "worker_count": WORKER_COUNT,
            "worker_mapping": workers,
        },
        "gpus": gpu_identity,
        "host": {"alias": host_alias, "hostname": hostname},
        "materialized_at_utc": materialized_text,
        "preflight": {
            "completed_at_utc": evidence["completed_at_utc"],
            "evidence": evidence_binding,
            "freshness_maximum_seconds": PREFLIGHT_MAX_AGE_SECONDS,
            "minimum_window_seconds": PREFLIGHT_MINIMUM_SECONDS,
            "parent_preflight_reused": False,
            "started_at_utc": evidence["started_at_utc"],
        },
        "protocol_id": PROTOCOL_ID,
        "runtime": {
            "attention_backend_by_profile": {
                "auto": {"text": "sdpa", "top": "sdpa", "vision": "sdpa"},
                "eager": {"text": "eager", "top": "eager", "vision": "eager"},
            },
            "python_executable": python_executable,
            "python_path": str(root / "code"),
            "software_versions": dict(software_versions),
        },
        "schema_version": SCHEMA_VERSION,
        "source": {
            "branch": SOURCE_BRANCH,
            "config": source_binding,
            "execution_config_path": CANONICAL_GIT_ENVELOPE_PATH,
            "git_revision": source_git_revision,
            "inventory_sha256": contract.data["source"]["inventory_sha256"],
            "remote": SOURCE_REMOTE,
        },
        "status": AUTHORIZED_STATUS,
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_clean_pushed_source(
    *, repository_root: str | Path, expected_git_revision: str
) -> dict[str, str]:
    root = Path(repository_root).resolve()
    head = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    tracking = _git(root, "rev-parse", f"refs/remotes/{SOURCE_REMOTE}/{SOURCE_BRANCH}")
    if (
        head != expected_git_revision
        or _COMMIT.fullmatch(head) is None
        or branch != SOURCE_BRANCH
        or status
        or tracking != head
    ):
        raise ValueError("D1 envelope materialization requires clean pushed source A")
    committed = subprocess.run(
        ["git", "show", f"{head}:{CANONICAL_CONFIG_PATH}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if committed != (root / CANONICAL_CONFIG_PATH).read_bytes():
        raise ValueError("D1 source config differs from source-A Git bytes")
    return {"branch": branch, "git_revision": head, "remote_revision": tracking}


def validate_committed_envelope_lifecycle(
    *,
    repository_root: str | Path,
    source_git_revision: str,
    data_envelope_path: str | Path,
    expected_envelope: Mapping[str, Any],
) -> dict[str, str]:
    root = Path(repository_root).resolve()
    head = _git(root, "rev-parse", "HEAD")
    parents = _git(root, "rev-list", "--parents", "-n", "1", head).split()
    changed = _git(
        root, "diff", "--name-only", "--no-renames", source_git_revision, head
    ).splitlines()
    if (
        _git(root, "branch", "--show-current") != SOURCE_BRANCH
        or _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        or _git(root, "rev-parse", f"refs/remotes/{SOURCE_REMOTE}/{SOURCE_BRANCH}")
        != head
        or parents != [head, source_git_revision]
        or changed != [CANONICAL_GIT_ENVELOPE_PATH]
    ):
        raise ValueError("D1 runtime requires pushed direct-child envelope B")
    payload = canonical_pretty_json_bytes(expected_envelope)
    git_path = root.joinpath(*PurePosixPath(CANONICAL_GIT_ENVELOPE_PATH).parts)
    data_path = _absolute(str(data_envelope_path), label="D1 data envelope")
    committed = subprocess.run(
        ["git", "show", f"{head}:{CANONICAL_GIT_ENVELOPE_PATH}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if (
        not git_path.is_file()
        or git_path.is_symlink()
        or not data_path.is_file()
        or data_path.is_symlink()
        or git_path.read_bytes() != payload
        or data_path.read_bytes() != payload
        or committed != payload
    ):
        raise ValueError("D1 Git-B and /data envelope bytes differ")
    return {"envelope_git_revision": head, "source_git_revision": source_git_revision}


def validate_action_stability_envelope_v1(
    envelope: Mapping[str, Any],
    *,
    repository_root: str | Path,
    envelope_path: str | Path,
    verify_repository: bool = True,
    require_fresh_preflight: bool = True,
    verify_current_environment: bool = False,
    now_utc: str | None = None,
    source_contract_loader: Callable[..., Any] = (
        load_action_stability_source_v1_contract
    ),
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if set(envelope) != {
        "artifacts",
        "authorization",
        "container",
        "execution",
        "gpus",
        "host",
        "materialized_at_utc",
        "preflight",
        "protocol_id",
        "runtime",
        "schema_version",
        "source",
        "status",
    }:
        raise ValueError("D1 envelope top-level schema drifted")
    if (
        envelope.get("protocol_id") != PROTOCOL_ID
        or envelope.get("schema_version") != SCHEMA_VERSION
        or envelope.get("status") != AUTHORIZED_STATUS
    ):
        raise ValueError("D1 envelope identity drifted")
    contract = source_contract_loader(repository_root=root)
    if envelope.get("authorization") != {
        "generate_restoration_labels": False,
        "run_closed_loop": False,
        "run_exact_36_call_action_stability_diagnostic": True,
        "train_predictor": False,
    }:
        raise ValueError("D1 envelope authorization drifted")
    source = _mapping(envelope["source"], label="D1 envelope source")
    if (
        set(source)
        != {
            "branch",
            "config",
            "execution_config_path",
            "git_revision",
            "inventory_sha256",
            "remote",
        }
        or source.get("branch") != SOURCE_BRANCH
        or source.get("remote") != SOURCE_REMOTE
        or not isinstance(source.get("git_revision"), str)
        or _COMMIT.fullmatch(str(source.get("git_revision"))) is None
        or source.get("config")
        != {
            "path": CANONICAL_CONFIG_PATH,
            "sha256": contract.config_sha256,
            "size_bytes": (root / CANONICAL_CONFIG_PATH).stat().st_size,
        }
        or source.get("inventory_sha256")
        != contract.data["source"]["inventory_sha256"]
        or source.get("execution_config_path") != CANONICAL_GIT_ENVELOPE_PATH
    ):
        raise ValueError("D1 envelope source binding drifted")
    runtime = _mapping(envelope["runtime"], label="D1 runtime")
    expected_python_path = str(root / "code")
    required_versions = {
        "cuda_runtime_version",
        "python_version",
        "torch_version",
        "transformers_version",
    }
    versions = _mapping(runtime.get("software_versions"), label="D1 software versions")
    if (
        set(runtime)
        != {
            "attention_backend_by_profile",
            "python_executable",
            "python_path",
            "software_versions",
        }
        or runtime.get("python_path") != expected_python_path
        or not isinstance(runtime.get("python_executable"), str)
        or not _absolute(runtime.get("python_executable"), label="D1 Python").is_file()
        or set(versions) != required_versions
        or any(not isinstance(value, str) or not value for value in versions.values())
    ):
        raise ValueError("D1 runtime identity drifted")
    execution = _mapping(envelope["execution"], label="D1 execution")
    if set(execution) != {
        "aggregate",
        "auto_all_four_terminals_required_before_any_eager_attempt",
        "condition_order",
        "encode_call_ceiling",
        "generation_call_ceiling",
        "no_retry",
        "no_top_up",
        "output_layout",
        "profiles",
        "worker_count",
        "worker_mapping",
    }:
        raise ValueError("D1 execution schema drifted")
    layout = _mapping(execution.get("output_layout"), label="D1 output layout")
    expected_layout = canonical_run_layout(layout.get("run_root"))
    if dict(layout) != expected_layout or str(envelope_path) != layout["envelope_path"]:
        raise ValueError("D1 canonical output layout drifted")
    if (
        execution.get("profiles") != list(PROFILE_ORDER)
        or execution.get("condition_order") != list(CONDITION_ORDER)
        or execution.get("worker_count") != WORKER_COUNT
        or execution.get("generation_call_ceiling") != GENERATION_CALL_CEILING
        or execution.get("encode_call_ceiling") != ENCODE_CALL_CEILING
        or execution.get("no_retry") is not True
        or execution.get("no_top_up") is not True
        or execution.get("auto_all_four_terminals_required_before_any_eager_attempt")
        is not True
    ):
        raise ValueError("D1 execution contract drifted")
    gpus = [dict(_mapping(gpu, label="D1 GPU")) for gpu in _sequence(
        envelope["gpus"], label="D1 GPUs"
    )]
    gpu_keys = {"host_index", "name", "total_memory_bytes", "uuid", "visible_index"}
    if (
        len(gpus) != WORKER_COUNT
        or any(set(gpu) != gpu_keys for gpu in gpus)
        or any(gpu["visible_index"] != index for index, gpu in enumerate(gpus))
        or any(
            type(gpu["host_index"]) is not int
            or gpu["host_index"] < 0
            or gpu["name"] != GPU_NAME
            or type(gpu["total_memory_bytes"]) is not int
            or gpu["total_memory_bytes"] <= 0
            or not isinstance(gpu["uuid"], str)
            or _GPU_UUID.fullmatch(gpu["uuid"]) is None
            for gpu in gpus
        )
        or len({gpu["host_index"] for gpu in gpus}) != WORKER_COUNT
        or len({gpu["uuid"] for gpu in gpus}) != WORKER_COUNT
    ):
        raise ValueError("D1 envelope is not bound to four H200 GPUs")
    workers, aggregate_argv = _canonical_argv(
        repository_root=root,
        python_executable=runtime["python_executable"],
        envelope_path=layout["envelope_path"],
    )
    observed_workers = _sequence(execution.get("worker_mapping"), label="D1 workers")
    if len(observed_workers) != 2 * WORKER_COUNT:
        raise ValueError("D1 worker schedule count drifted")
    for expected, observed in zip(workers, observed_workers, strict=True):
        record = _mapping(observed, label="D1 worker")
        worker_index = expected["worker_index"]
        gpu = gpus[worker_index]
        expected.update(
            {
                "cuda_visible_devices": gpu["uuid"],
                "gpu_uuid": gpu["uuid"],
                "output": layout["workers"][worker_index]["profiles"][expected["profile"]],
                "python_path": str(root / "code"),
                "state_ids": _worker_state_ids(worker_index),
            }
        )
        if dict(record) != expected:
            raise ValueError("D1 worker schedule drifted")
    aggregate = _mapping(execution["aggregate"], label="D1 aggregate execution")
    if dict(aggregate) != {
        "argv": aggregate_argv,
        "output": {
            key: layout[key]
            for key in (
                "aggregate_result_path",
                "aggregate_stderr_log_path",
                "aggregate_stdout_log_path",
            )
        },
    }:
        raise ValueError("D1 aggregate argv drifted")
    artifacts = _mapping(envelope["artifacts"], label="D1 artifacts")
    if set(artifacts) != {
        "model_dir",
        "parent_execution_envelope",
        "processor_root",
    }:
        raise ValueError("D1 artifact schema drifted")
    parent, parent_binding = _load_parent_envelope(
        _absolute(
            artifacts["parent_execution_envelope"]["path"],
            label="parent envelope",
        )
    )
    if (
        parent_binding != artifacts["parent_execution_envelope"]
        or parent_binding["sha256"] != PARENT_EXECUTION_ENVELOPE_SHA256
    ):
        raise ValueError("D1 parent throughput envelope binding drifted")
    embedded_model, embedded_processor = _parent_artifact_paths(parent)
    if (
        artifacts.get("model_dir") != embedded_model
        or artifacts.get("processor_root") != embedded_processor
        or not Path(embedded_model).is_dir()
        or not Path(embedded_processor).is_dir()
    ):
        raise ValueError("D1 artifact path binding drifted")
    host = _mapping(envelope["host"], label="D1 host")
    container = _mapping(envelope["container"], label="D1 container")
    if (
        dict(host) != {
            "alias": str(host.get("alias")),
            "hostname": str(host.get("hostname")),
        }
        or _HOSTS.get(host.get("alias")) != host.get("hostname")
        or set(container) != {"id", "image_digest", "image_reference"}
        or not isinstance(container.get("id"), str)
        or _CONTAINER_ID.fullmatch(str(container.get("id"))) is None
        or not isinstance(container.get("image_digest"), str)
        or _IMAGE_DIGEST.fullmatch(str(container.get("image_digest"))) is None
        or not isinstance(container.get("image_reference"), str)
        or not container.get("image_reference")
    ):
        raise ValueError("D1 host/container identity drifted")
    preflight = _mapping(envelope["preflight"], label="D1 preflight")
    if set(preflight) != {
        "completed_at_utc",
        "evidence",
        "freshness_maximum_seconds",
        "minimum_window_seconds",
        "parent_preflight_reused",
        "started_at_utc",
    }:
        raise ValueError("D1 preflight schema drifted")
    evidence_path = _absolute(preflight["evidence"]["path"], label="D1 evidence")
    evidence_binding = _regular_binding(evidence_path, label="D1 preflight evidence")
    if evidence_binding != preflight["evidence"]:
        raise ValueError("D1 preflight evidence bytes drifted")
    evidence_payload = evidence_path.read_bytes()
    evidence = _strict_json_object(evidence_payload, label="D1 evidence")
    if (
        evidence_payload != canonical_pretty_json_bytes(evidence)
        or evidence.get("status") != PREFLIGHT_STATUS
        or evidence.get("gpus") is None
    ):
        raise ValueError("D1 preflight evidence status drifted")
    rebuilt_evidence = build_action_stability_preflight_evidence_v1(
        raw_cleanup_log_path=evidence["cleanup"]["path"],
        started_at_utc=evidence["started_at_utc"],
        completed_at_utc=evidence["completed_at_utc"],
        host_alias=host["alias"],
        hostname=host["hostname"],
        container_id=container["id"],
        driver_version=evidence["driver_version"],
        selected_gpus=evidence["gpus"],
    )
    evidence_gpu_identity = [
        {
            key: gpu[key]
            for key in (
                "host_index",
                "name",
                "total_memory_bytes",
                "uuid",
                "visible_index",
            )
        }
        for gpu in evidence["gpus"]
    ]
    if evidence != rebuilt_evidence or evidence_gpu_identity != gpus:
        raise ValueError("D1 preflight evidence reconstruction drifted")
    started = _timestamp(preflight["started_at_utc"], label="D1 preflight start")
    completed = _timestamp(preflight["completed_at_utc"], label="D1 preflight completion")
    materialized = _timestamp(envelope["materialized_at_utc"], label="D1 materialized")
    if (
        (completed - started).total_seconds() < PREFLIGHT_MINIMUM_SECONDS
        or not started <= completed <= materialized
        or preflight.get("started_at_utc") != evidence.get("started_at_utc")
        or preflight.get("completed_at_utc") != evidence.get("completed_at_utc")
        or preflight.get("minimum_window_seconds") != PREFLIGHT_MINIMUM_SECONDS
        or preflight.get("freshness_maximum_seconds") != PREFLIGHT_MAX_AGE_SECONDS
        or preflight.get("parent_preflight_reused") is not False
    ):
        raise ValueError("D1 preflight temporal binding drifted")
    if require_fresh_preflight:
        now = _timestamp(now_utc or utc_now(), label="D1 validation time")
        if now < materialized or (
            now - completed
        ).total_seconds() > PREFLIGHT_MAX_AGE_SECONDS:
            raise ValueError("D1 preflight is no longer fresh")
    if runtime.get("attention_backend_by_profile") != {
        "auto": {"text": "sdpa", "top": "sdpa", "vision": "sdpa"},
        "eager": {"text": "eager", "top": "eager", "vision": "eager"},
    }:
        raise ValueError("D1 attention backend binding drifted")
    if verify_current_environment:
        if os.environ.get("PYTHONPATH") != expected_python_path:
            raise ValueError("D1 current PYTHONPATH differs from the source worktree")
        # note (luojiaxuan): Reuse the already exercised container/GPU observer
        # from the parent throughput envelope so Docker hostnames and isolated
        # CUDA UUIDs are checked without assuming socket.gethostname() is the host.
        from causalcache.set_utility_throughput_pilot_envelope_v1 import (
            _validate_current_execution_identity,
        )

        _validate_current_execution_identity(
            hostname=host["hostname"],
            container_id=container["id"],
            driver_version=evidence["driver_version"],
            gpus=gpus,
            python_executable=runtime["python_executable"],
            software_versions=versions,
            allow_single_gpu_isolation=True,
        )
    if verify_repository:
        validate_committed_envelope_lifecycle(
            repository_root=root,
            source_git_revision=source["git_revision"],
            data_envelope_path=envelope_path,
            expected_envelope=envelope,
        )
    payload = canonical_pretty_json_bytes(envelope)
    return {
        "envelope_sha256": sha256_bytes(payload),
        "freshness_validated": require_fresh_preflight,
        "parent_envelope_sha256": PARENT_EXECUTION_ENVELOPE_SHA256,
        "profiles": list(PROFILE_ORDER),
        "source_config_sha256": contract.config_sha256,
        "source_inventory_sha256": contract.data["source"]["inventory_sha256"],
        "status": VALIDATION_STATUS,
        "this_validated_envelope_authorizes_gpu_execution": True,
        "worker_count": WORKER_COUNT,
    }


def load_set_utility_action_stability_envelope_v1(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
    verify_repository: bool = True,
    require_fresh_preflight: bool = True,
    verify_current_environment: bool = True,
) -> dict[str, Any]:
    supplied = Path(path)
    if not supplied.is_absolute() or supplied.is_symlink() or not supplied.is_file():
        raise ValueError("D1 execution envelope must be an absolute regular file")
    payload = supplied.read_bytes()
    envelope = _strict_json_object(payload, label="D1 execution envelope")
    if payload != canonical_pretty_json_bytes(envelope):
        raise ValueError("D1 execution envelope must be canonical pretty JSON")
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    validation = validate_action_stability_envelope_v1(
        envelope,
        repository_root=root,
        envelope_path=supplied,
        verify_repository=verify_repository,
        require_fresh_preflight=require_fresh_preflight,
        verify_current_environment=verify_current_environment,
    )
    execution = envelope["execution"]
    gpus = envelope["gpus"]
    return {
        "device_by_worker": {str(index): "cuda:0" for index in range(WORKER_COUNT)},
        "envelope_path": execution["output_layout"]["envelope_path"],
        "model_dir": envelope["artifacts"]["model_dir"],
        "parent_envelope_path": envelope["artifacts"]["parent_execution_envelope"]["path"],
        "processor_root": envelope["artifacts"]["processor_root"],
        "profiles": list(PROFILE_ORDER),
        "repository_root": str(root),
        "run_root": execution["output_layout"]["run_root"],
        "source_config_path": str(root / CANONICAL_CONFIG_PATH),
        "validation": validation,
        "worker_gpu_uuid": {
            str(index): gpus[index]["uuid"] for index in range(WORKER_COUNT)
        },
    }


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"D1 output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError("D1 exclusive write made no progress")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_preflight_evidence_exclusive(
    *, run_root: str | Path, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(canonical_run_layout(run_root)["preflight_evidence_path"])
    payload = canonical_pretty_json_bytes(evidence)
    _write_exclusive(path, payload)
    return {"path": str(path), "sha256": sha256_bytes(payload), "size_bytes": len(payload)}


def write_canonical_envelope_pair_exclusive(
    *,
    repository_root: str | Path,
    data_path: str | Path,
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    data = Path(data_path)
    git_path = root.joinpath(*PurePosixPath(CANONICAL_GIT_ENVELOPE_PATH).parts)
    expected_data = envelope["execution"]["output_layout"]["envelope_path"]
    if str(data) != expected_data or os.path.lexists(data) or os.path.lexists(git_path):
        raise FileExistsError("D1 Git and /data envelope paths must both be fresh")
    payload = canonical_pretty_json_bytes(envelope)
    created: list[Path] = []
    try:
        _write_exclusive(git_path, payload)
        created.append(git_path)
        _write_exclusive(data, payload)
        created.append(data)
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return {
        "data_path": str(data),
        "git_path": CANONICAL_GIT_ENVELOPE_PATH,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "status": AUTHORIZED_STATUS,
    }


__all__ = [
    "AGGREGATE_ENTRYPOINT",
    "AUTHORIZED_STATUS",
    "CANONICAL_GIT_ENVELOPE_PATH",
    "GPU_NAME",
    "PREFLIGHT_MAX_AGE_SECONDS",
    "PREFLIGHT_MINIMUM_SECONDS",
    "PREFLIGHT_STATUS",
    "PROFILE_ORDER",
    "PROTOCOL_ID",
    "VALIDATION_STATUS",
    "WORKER_ENTRYPOINT",
    "build_action_stability_envelope_v1",
    "build_action_stability_preflight_evidence_v1",
    "canonical_run_layout",
    "load_set_utility_action_stability_envelope_v1",
    "utc_now",
    "validate_action_stability_envelope_v1",
    "validate_clean_pushed_source",
    "validate_committed_envelope_lifecycle",
    "write_canonical_envelope_pair_exclusive",
    "write_preflight_evidence_exclusive",
]
