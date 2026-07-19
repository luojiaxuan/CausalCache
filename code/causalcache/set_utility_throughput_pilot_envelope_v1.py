"""Exact runtime authorization envelope for the train-only throughput pilot."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import socket
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.set_utility_throughput_pilot_contract_v1 import (
    CANONICAL_CONFIG_PATH,
    NATIVE_CALL_CEILING,
    WORKER_COUNT,
    canonical_json_bytes,
    canonical_pretty_json_bytes,
    load_train_only_throughput_pilot_v1_contract,
    sha256_bytes,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_throughput_pilot_execution_envelope_v1"
AUTHORIZED_STATUS = "AUTHORIZED_EXACT_84_CALL_GPU_EXECUTION"
VALIDATION_STATUS = "VALID_EXACT_84_CALL_GPU_EXECUTION_ENVELOPE"
ARTIFACT_VALIDATION_STATUS = "FULL_SHA256_AND_STAT_IDENTITY_VALIDATED"
ARTIFACT_VALIDATION_METHOD = "full_sha256_inventory_with_stat_identity_receipt_v1"
PREFLIGHT_EVIDENCE_SCHEMA_VERSION = "1.0.0"
PREFLIGHT_EVIDENCE_PROTOCOL_ID = (
    "causalcache_set_utility_throughput_pilot_preflight_evidence_v1"
)
PREFLIGHT_EVIDENCE_STATUS = "PASSED_EXACT_FOUR_GPU_PREFLIGHT"
HOST_ALIAS = "hyper00"
HOSTNAME = "node-radixark-16-0000"
SOURCE_BRANCH = "main"
SOURCE_REMOTE = "origin"
CANONICAL_GIT_ENVELOPE_PATH = (
    "code/configs/causalcache_set_utility_train_only_throughput_pilot_v1_execution.json"
)
GPU_COUNT = 4
PREFLIGHT_MAX_AGE_SECONDS = 15 * 60
PERSISTENT_DATA_ROOT = Path("/data")
WORKER_ENTRYPOINT = "code/scripts/run_set_utility_throughput_pilot_worker_v1.py"
AGGREGATE_ENTRYPOINT = "code/scripts/aggregate_set_utility_throughput_pilot_v1.py"

_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{16,}")
_DRIVER_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}")
_CONTAINER_ID_FRAGMENT = re.compile(r"(?<![0-9a-f])[0-9a-f]{12,64}(?![0-9a-f])")
_FORBIDDEN_SERIALIZED_KEYS = frozenset(
    {
        "action",
        "action_text",
        "decoded_output",
        "kl",
        "logits",
        "messages",
        "native_output",
        "policy_output",
        "secret",
        "token",
        "tokens",
        "utility",
    }
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one JSON object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be one JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} schema drifted")


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def load_strict_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    supplied = Path(path)
    if not supplied.is_absolute() or supplied.is_symlink() or not supplied.is_file():
        raise ValueError(f"{label} must be an absolute regular non-symlink file")
    return _strict_json_object(supplied.read_bytes(), label=label)


def _timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be UTC RFC3339 with a Z suffix")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be UTC RFC3339") from error
    if parsed.tzinfo != timezone.utc or parsed.microsecond != 0:
        raise ValueError(f"{label} must have whole-second UTC precision")
    return parsed


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _absolute_normalized_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be one absolute path string")
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        raise ValueError(f"{label} must be an absolute normalized path")
    return path


def _path_under(root: Path, path: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must be under the canonical run root") from error


def canonical_run_layout(run_root: str | Path) -> dict[str, Any]:
    root = _absolute_normalized_path(str(run_root), label="run root")
    try:
        relative = root.relative_to(PERSISTENT_DATA_ROOT)
    except ValueError as error:
        raise ValueError(
            "run root must be a task-specific persistent /data path"
        ) from error
    if not relative.parts:
        raise ValueError("run root must be a task-specific persistent /data path")
    workers = []
    for worker_index in range(WORKER_COUNT):
        workers.append(
            {
                "attempt_path": str(
                    root / "workers" / f"worker-{worker_index:02d}" / "attempt.json"
                ),
                "claim_path": str(
                    root / "claims" / f"worker-{worker_index:02d}.claim.json"
                ),
                "stderr_log_path": str(
                    root / "logs" / f"worker-{worker_index:02d}.stderr.log"
                ),
                "stdout_log_path": str(
                    root / "logs" / f"worker-{worker_index:02d}.stdout.log"
                ),
                "terminal_path": str(
                    root / "workers" / f"worker-{worker_index:02d}" / "terminal.json"
                ),
                "worker_index": worker_index,
            }
        )
    return {
        "aggregate_claim_path": str(root / "claims" / "aggregate.claim.json"),
        "aggregate_result_path": str(root / "aggregate.json"),
        "aggregate_stderr_log_path": str(root / "logs" / "aggregate.stderr.log"),
        "aggregate_stdout_log_path": str(root / "logs" / "aggregate.stdout.log"),
        "envelope_path": str(root / "execution-envelope.json"),
        "run_root": str(root),
        "workers": workers,
    }


def canonical_preflight_evidence_path(run_root: str | Path) -> Path:
    layout = canonical_run_layout(run_root)
    return Path(layout["run_root"]) / "preflight-evidence.json"


def _inspect_regular(path: Path, *, allow_symlink: bool, label: str) -> dict[str, Any]:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if path.is_symlink() and not allow_symlink:
        raise ValueError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} is missing") from error
    if not resolved.is_file():
        raise ValueError(f"{label} must resolve to one regular file")
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must resolve to one regular file")
        digest = hashlib.sha256()
        byte_count = 0
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or byte_count != after.st_size:
        raise RuntimeError(f"{label} changed while being inspected")
    return {"sha256": digest.hexdigest(), "size_bytes": byte_count}


def _inventory_tree(
    root: Path,
    expected: Sequence[Mapping[str, Any]],
    *,
    allow_file_symlinks: bool,
    label: str,
) -> list[dict[str, Any]]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root must be an absolute real directory")
    expected_paths: list[str] = []
    normalized_expected: list[dict[str, Any]] = []
    for index, raw in enumerate(expected):
        record = _mapping(raw, label=f"{label} expected file {index}")
        if set(record) not in (
            {"path", "sha256", "size"},
            {"path", "sha256", "size_bytes"},
        ):
            raise ValueError(f"{label} expected inventory schema drifted")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"{label} expected path is invalid")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ValueError(f"{label} expected path is unsafe")
        size = record.get("size", record.get("size_bytes"))
        digest = record.get("sha256")
        if (
            not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or type(size) is not int
            or size < 0
        ):
            raise ValueError(f"{label} expected identity is invalid")
        expected_paths.append(relative)
        normalized_expected.append(
            {"path": relative, "sha256": digest, "size_bytes": size}
        )
    if expected_paths != sorted(expected_paths) or len(set(expected_paths)) != len(
        expected_paths
    ):
        raise ValueError(f"{label} expected inventory must be sorted and unique")

    observed_paths: list[str] = []
    for directory, directories, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directories:
            if (base / name).is_symlink():
                raise ValueError(f"{label} must not contain directory symlinks")
        for name in files:
            observed_paths.append((base / name).relative_to(root).as_posix())
    if sorted(observed_paths) != expected_paths:
        raise ValueError(f"{label} local file inventory differs from the freeze")

    observed: list[dict[str, Any]] = []
    for expected_record in normalized_expected:
        relative = expected_record["path"]
        inspected = _inspect_regular(
            root.joinpath(*PurePosixPath(relative).parts),
            allow_symlink=allow_file_symlinks,
            label=f"{label} file {relative}",
        )
        record = {"path": relative, **inspected}
        if record != expected_record:
            raise ValueError(f"{label} local file identity drifted: {relative}")
        observed.append(record)
    return observed


def _file_binding(path: Path, *, label: str) -> dict[str, Any]:
    inspected = _inspect_regular(path, allow_symlink=False, label=label)
    return {"path": str(path), **inspected}


def _observed_tree_paths(root: Path, *, label: str) -> list[str]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root must be an absolute real directory")
    observed: list[str] = []
    for directory, directories, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directories:
            if (base / name).is_symlink():
                raise ValueError(f"{label} must not contain directory symlinks")
        for name in files:
            observed.append((base / name).relative_to(root).as_posix())
    return sorted(observed)


def _stat_identity_inventory(
    root: Path,
    expected: Sequence[Mapping[str, Any]],
    *,
    allow_file_symlinks: bool,
    label: str,
) -> list[dict[str, Any]]:
    """Capture a cheap immutable-file identity receipt after the full hash pass."""
    expected_paths = [str(record["path"]) for record in expected]
    if expected_paths != sorted(expected_paths) or len(set(expected_paths)) != len(
        expected_paths
    ):
        raise ValueError(f"{label} expected inventory must be sorted and unique")
    if _observed_tree_paths(root, label=label) != expected_paths:
        raise ValueError(f"{label} local file inventory differs from the freeze")
    identities: list[dict[str, Any]] = []
    for expected_record in expected:
        relative = str(expected_record["path"])
        supplied = root.joinpath(*PurePosixPath(relative).parts)
        if supplied.is_symlink() and not allow_file_symlinks:
            raise ValueError(f"{label} file {relative} must not be a symlink")
        try:
            resolved = supplied.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"{label} file {relative} is missing") from error
        descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            observed = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError(f"{label} file {relative} must resolve to a regular file")
        if observed.st_size != expected_record["size_bytes"]:
            raise ValueError(f"{label} file {relative} size drifted")
        identities.append(
            {
                "device": observed.st_dev,
                "inode": observed.st_ino,
                "mode": stat.S_IMODE(observed.st_mode),
                "mtime_ns": observed.st_mtime_ns,
                "path": relative,
                "resolved_path": str(resolved),
                "size_bytes": observed.st_size,
            }
        )
    return identities


def _validate_stat_identity_receipt(
    root: Path,
    expected_inventory: Sequence[Mapping[str, Any]],
    receipt: Mapping[str, Any],
    *,
    allow_file_symlinks: bool,
    label: str,
) -> None:
    identities = _stat_identity_inventory(
        root,
        expected_inventory,
        allow_file_symlinks=allow_file_symlinks,
        label=label,
    )
    if (
        receipt.get("file_count") != len(expected_inventory)
        or receipt.get("file_inventory_sha256")
        != sha256_bytes(canonical_json_bytes(expected_inventory))
        or receipt.get("stat_identity_sha256")
        != sha256_bytes(canonical_json_bytes(identities))
        or receipt.get("stat_identities") != identities
    ):
        raise ValueError(f"{label} stat identity receipt drifted")


def _container_identity_fragments() -> set[str]:
    fragments: set[str] = set()
    for candidate in (
        socket.gethostname(),
        os.environ.get("HOSTNAME", ""),
    ):
        if re.fullmatch(r"[0-9a-f]{12,64}", candidate):
            fragments.add(candidate)
    for path in (Path("/proc/self/cgroup"), Path("/proc/1/cgroup"), Path("/proc/self/mountinfo")):
        try:
            content = path.read_text(encoding="utf-8", errors="strict")
        except OSError:
            continue
        fragments.update(_CONTAINER_ID_FRAGMENT.findall(content))
    return fragments


def _observe_current_execution_identity() -> dict[str, Any]:
    torch_module = importlib.import_module("torch")
    transformers_module = importlib.import_module("transformers")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("nvidia-smi current-execution identity query failed")
    gpus: list[dict[str, Any]] = []
    drivers: set[str] = set()
    for visible_index, line in enumerate(result.stdout.splitlines()):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ValueError("nvidia-smi current-execution identity row drifted")
        host_index_text, uuid, name, memory_mib_text, driver = fields
        try:
            host_index = int(host_index_text)
            memory_mib = int(memory_mib_text)
        except ValueError as error:
            raise ValueError("nvidia-smi numeric identity field is invalid") from error
        drivers.add(driver)
        gpus.append(
            {
                "name": name,
                "reported_index": host_index,
                "total_memory_bytes": memory_mib * 1024 * 1024,
                "uuid": uuid,
                "visible_index": visible_index,
            }
        )
    if len(drivers) != 1:
        raise ValueError("selected GPUs do not expose one driver version")
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    if not isinstance(cuda_version, str) or not cuda_version:
        raise ValueError("torch CUDA runtime version is unavailable")
    return {
        "container_id_fragments": sorted(_container_identity_fragments()),
        "container_hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpus,
        "driver_version": next(iter(drivers)),
        "python_executable": str(Path(sys.executable).resolve()),
        "software_versions": {
            "cuda_runtime_version": cuda_version,
            "python_version": platform.python_version(),
            "torch_version": str(getattr(torch_module, "__version__")),
            "transformers_version": str(getattr(transformers_module, "__version__")),
        },
    }


def _model_expected_inventory(repository_root: Path, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    binding = _mapping(
        _mapping(source.get("inputs"), label="source inputs").get(
            "model_snapshot_manifest"
        ),
        label="model snapshot manifest binding",
    )
    relative = binding.get("path")
    if not isinstance(relative, str):
        raise ValueError("model snapshot manifest path drifted")
    path = repository_root.joinpath(*PurePosixPath(relative).parts)
    payload = path.read_bytes()
    if (
        len(payload) != binding.get("byte_count")
        or sha256_bytes(payload) != binding.get("sha256")
    ):
        raise ValueError("model snapshot manifest binding drifted")
    manifest = _strict_json_object(payload, label="model snapshot manifest")
    files = _sequence(manifest.get("files"), label="model snapshot files")
    inventory = [
        {
            "path": record["path"],
            "sha256": record["sha256"],
            "size_bytes": record["size"],
        }
        for record in (
            _mapping(value, label="model snapshot file") for value in files
        )
    ]
    inventory.append(
        {
            "path": ".snapshot.json",
            "sha256": binding["sha256"],
            "size_bytes": binding["byte_count"],
        }
    )
    inventory.sort(key=lambda record: record["path"])
    if [record["path"] for record in inventory] != sorted(
        record["path"] for record in inventory
    ):
        raise ValueError("model snapshot manifest inventory must be sorted")
    return inventory


def _validate_source_authorization(source: Mapping[str, Any]) -> None:
    authorization = _mapping(source.get("authorization"), label="source authorization")
    for key in (
        "load_policy_or_vision_model",
        "run_gpu_or_cuda",
        "write_pilot_result",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"source config must keep {key}=false")
    rule = _mapping(
        source.get("deployment_decision_rule"), label="source decision rule"
    )
    if (
        rule.get("execution_requires_separate_exact_envelope") is not True
        or rule.get("no_retry") is not True
        or rule.get("no_top_up") is not True
    ):
        raise ValueError("source config execution firewall drifted")


def _normalize_gpus(gpus: Sequence[Mapping[str, Any]], *, driver_version: str) -> list[dict[str, Any]]:
    if len(gpus) != GPU_COUNT:
        raise ValueError("execution envelope requires exactly four GPUs")
    if _DRIVER_VERSION.fullmatch(driver_version) is None:
        raise ValueError("NVIDIA driver version is invalid")
    normalized: list[dict[str, Any]] = []
    for visible_index, raw in enumerate(gpus):
        gpu = _mapping(raw, label=f"GPU {visible_index}")
        _exact_keys(
            gpu,
            {"host_index", "name", "total_memory_bytes", "uuid", "visible_index"},
            label=f"GPU {visible_index}",
        )
        if (
            type(gpu.get("host_index")) is not int
            or gpu["host_index"] < 0
            or gpu.get("visible_index") != visible_index
            or not isinstance(gpu.get("uuid"), str)
            or _GPU_UUID.fullmatch(gpu["uuid"]) is None
            or not isinstance(gpu.get("name"), str)
            or "H200" not in gpu["name"]
            or type(gpu.get("total_memory_bytes")) is not int
            or gpu["total_memory_bytes"] <= 0
        ):
            raise ValueError(f"GPU {visible_index} identity is invalid")
        normalized.append(dict(gpu))
    if len({gpu["host_index"] for gpu in normalized}) != GPU_COUNT or len(
        {gpu["uuid"] for gpu in normalized}
    ) != GPU_COUNT:
        raise ValueError("GPU host ids and UUIDs must be unique")
    if len({gpu["name"] for gpu in normalized}) != 1 or len(
        {gpu["total_memory_bytes"] for gpu in normalized}
    ) != 1:
        raise ValueError("all selected GPUs must have the same H200 identity")
    return normalized


def _software_versions(value: Mapping[str, Any]) -> dict[str, str]:
    expected = {
        "cuda_runtime_version",
        "python_version",
        "torch_version",
        "transformers_version",
    }
    _exact_keys(value, expected, label="software versions")
    result: dict[str, str] = {}
    for key in sorted(expected):
        raw = value.get(key)
        if not isinstance(raw, str) or not raw or len(raw) > 128:
            raise ValueError(f"{key} must be one explicit version string")
        result[key] = raw
    return result


def _normalize_preflight_selected_devices(
    value: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(value) != GPU_COUNT:
        raise ValueError("preflight evidence requires exactly four selected GPUs")
    full_devices: list[dict[str, Any]] = []
    identity_devices: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        selected = _mapping(raw, label=f"preflight selected GPU {index}")
        _exact_keys(
            selected,
            {
                "host_index",
                "name",
                "post_cleanup_compute_app_count",
                "post_cleanup_memory_used_mib",
                "sampled_max_utilization_percent",
                "total_memory_bytes",
                "uuid",
                "visible_index",
            },
            label=f"preflight selected GPU {index}",
        )
        if (
            selected.get("sampled_max_utilization_percent") != 0
            or selected.get("post_cleanup_compute_app_count") != 0
            or type(selected.get("post_cleanup_memory_used_mib")) is not int
            or selected["post_cleanup_memory_used_mib"] < 0
            or selected["post_cleanup_memory_used_mib"] > 500
        ):
            raise ValueError("preflight selected GPU is not idle after cleanup")
        full_devices.append(dict(selected))
        identity_devices.append(
            {
                key: selected[key]
                for key in (
                    "host_index",
                    "name",
                    "total_memory_bytes",
                    "uuid",
                    "visible_index",
                )
            }
        )
    normalized = _normalize_gpus(
        identity_devices,
        driver_version="0.0",
    )
    if normalized != identity_devices:
        raise ValueError("preflight selected GPU identities are not canonical")
    return full_devices, identity_devices


def build_set_utility_throughput_pilot_preflight_evidence_v1(
    *,
    raw_cleanup_log_path: str | Path,
    preflight_started_at_utc: str,
    preflight_completed_at_utc: str,
    host_alias: str,
    hostname: str,
    container_id: str,
    container_image_reference: str,
    container_image_digest: str,
    driver_version: str,
    selected_gpus: Sequence[Mapping[str, Any]],
    python_executable: str,
    software_versions: Mapping[str, Any],
) -> dict[str, Any]:
    started = _timestamp(preflight_started_at_utc, label="preflight evidence start")
    completed = _timestamp(
        preflight_completed_at_utc, label="preflight evidence completion"
    )
    if (completed - started).total_seconds() < 10:
        raise ValueError("preflight evidence must cover at least ten seconds")
    if host_alias != HOST_ALIAS or hostname != HOSTNAME:
        raise ValueError("throughput pilot preflight is authorized only on hyper00")
    if _CONTAINER_ID.fullmatch(container_id) is None:
        raise ValueError("container id must be one full lowercase Docker id")
    if not isinstance(container_image_reference, str) or not container_image_reference:
        raise ValueError("container image reference must be explicit")
    if _IMAGE_DIGEST.fullmatch(container_image_digest) is None:
        raise ValueError("container image digest must be sha256:<64 lowercase hex>")
    if _DRIVER_VERSION.fullmatch(driver_version) is None:
        raise ValueError("NVIDIA driver version is invalid")
    full_devices, _ = _normalize_preflight_selected_devices(selected_gpus)
    raw_log_path = _absolute_normalized_path(
        str(raw_cleanup_log_path), label="raw cleanup log path"
    )
    return {
        "completed_at_utc": preflight_completed_at_utc,
        "container": {
            "id": container_id,
            "image_digest": container_image_digest,
            "image_reference": container_image_reference,
        },
        "gpus": {"devices": full_devices, "driver_version": driver_version},
        "host": {"alias": host_alias, "hostname": hostname},
        "protocol_id": PREFLIGHT_EVIDENCE_PROTOCOL_ID,
        "raw_cleanup_log": _file_binding(raw_log_path, label="raw cleanup log"),
        "runtime": {
            "python_executable": str(
                _absolute_normalized_path(
                    python_executable, label="Python executable"
                )
            ),
            "software_versions": _software_versions(software_versions),
        },
        "schema_version": PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
        "started_at_utc": preflight_started_at_utc,
        "status": PREFLIGHT_EVIDENCE_STATUS,
    }


def _validated_preflight_evidence(
    path: Path,
    *,
    host_alias: str,
    hostname: str,
    container_id: str,
    container_image_reference: str,
    container_image_digest: str,
    driver_version: str,
    gpus: Sequence[Mapping[str, Any]],
    python_executable: str,
    software_versions: Mapping[str, Any],
    preflight_started_at_utc: str,
    preflight_completed_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("preflight evidence must be an absolute regular non-symlink file")
    payload = path.read_bytes()
    evidence = _strict_json_object(payload, label="preflight evidence")
    if payload != canonical_pretty_json_bytes(evidence):
        raise ValueError("preflight evidence must be canonical pretty JSON")
    _exact_keys(
        evidence,
        {
            "completed_at_utc",
            "container",
            "gpus",
            "host",
            "protocol_id",
            "raw_cleanup_log",
            "runtime",
            "schema_version",
            "started_at_utc",
            "status",
        },
        label="preflight evidence",
    )
    expected_container = {
        "id": container_id,
        "image_digest": container_image_digest,
        "image_reference": container_image_reference,
    }
    expected_host = {"alias": host_alias, "hostname": hostname}
    evidence_gpus = _mapping(evidence.get("gpus"), label="preflight evidence GPUs")
    _exact_keys(
        evidence_gpus, {"devices", "driver_version"}, label="preflight evidence GPUs"
    )
    _, selected_devices = _normalize_preflight_selected_devices(
        [
            _mapping(value, label="preflight evidence GPU device")
            for value in _sequence(
                evidence_gpus.get("devices"), label="preflight evidence GPU devices"
            )
        ]
    )
    expected_gpus = [dict(value) for value in gpus]
    expected_runtime = {
        "python_executable": str(
            _absolute_normalized_path(python_executable, label="Python executable")
        ),
        "software_versions": _software_versions(software_versions),
    }
    if (
        evidence.get("schema_version") != PREFLIGHT_EVIDENCE_SCHEMA_VERSION
        or evidence.get("protocol_id") != PREFLIGHT_EVIDENCE_PROTOCOL_ID
        or evidence.get("status") != PREFLIGHT_EVIDENCE_STATUS
        or evidence.get("started_at_utc") != preflight_started_at_utc
        or evidence.get("completed_at_utc") != preflight_completed_at_utc
        or evidence.get("host") != expected_host
        or evidence.get("container") != expected_container
        or evidence_gpus.get("driver_version") != driver_version
        or selected_devices != expected_gpus
        or evidence.get("runtime") != expected_runtime
    ):
        raise ValueError("preflight evidence identity binding drifted")
    evidence_started = _timestamp(
        preflight_started_at_utc, label="preflight evidence start"
    )
    evidence_completed = _timestamp(
        preflight_completed_at_utc, label="preflight evidence completion"
    )
    if (evidence_completed - evidence_started).total_seconds() < 10:
        raise ValueError("preflight evidence must cover at least ten seconds")
    raw_log = _mapping(evidence.get("raw_cleanup_log"), label="raw cleanup log")
    _exact_keys(raw_log, {"path", "sha256", "size_bytes"}, label="raw cleanup log")
    raw_log_path = _absolute_normalized_path(
        raw_log.get("path"), label="raw cleanup log path"
    )
    if dict(raw_log) != _file_binding(raw_log_path, label="raw cleanup log"):
        raise ValueError("raw cleanup stdout log bytes drifted")
    return evidence, {
        "path": str(path),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _validate_current_execution_identity(
    *,
    hostname: str,
    container_id: str,
    driver_version: str,
    gpus: Sequence[Mapping[str, Any]],
    python_executable: str,
    software_versions: Mapping[str, Any],
    allow_single_gpu_isolation: bool = False,
) -> None:
    observed = _observe_current_execution_identity()
    fragments = observed.get("container_id_fragments")
    if (
        not isinstance(fragments, list)
        or not fragments
        or not any(container_id.startswith(fragment) for fragment in fragments)
    ):
        raise ValueError("caller-supplied full container id is not the current container")
    container_hostname = observed.get("container_hostname")
    if container_hostname != hostname and not (
        isinstance(container_hostname, str)
        and re.fullmatch(r"[0-9a-f]{12,64}", container_hostname)
        and container_id.startswith(container_hostname)
    ):
        raise ValueError("caller-supplied hostname is not current host/container evidence")
    if observed.get("driver_version") != driver_version:
        raise ValueError("caller-supplied NVIDIA driver differs from current runtime")
    observed_gpus = observed.get("gpus")
    expected_visible_gpus = [
        {
            "name": value["name"],
            "total_memory_bytes": value["total_memory_bytes"],
            "uuid": value["uuid"],
            "visible_index": value["visible_index"],
        }
        for value in gpus
    ]
    observed_visible_gpus = [
        {
            key: value[key]
            for key in ("name", "total_memory_bytes", "uuid", "visible_index")
        }
        for value in _sequence(observed_gpus, label="observed current GPUs")
    ]
    isolated_uuid = observed.get("cuda_visible_devices")
    if allow_single_gpu_isolation and isinstance(isolated_uuid, str) and isolated_uuid:
        if "," in isolated_uuid or isolated_uuid not in {
            value["uuid"] for value in expected_visible_gpus
        }:
            raise ValueError("worker CUDA_VISIBLE_DEVICES is not one authorized GPU UUID")
        if len(observed_visible_gpus) == 1:
            expected_isolated = next(
                value for value in expected_visible_gpus if value["uuid"] == isolated_uuid
            )
            observed_isolated = observed_visible_gpus[0]
            if (
                observed_isolated["uuid"] != expected_isolated["uuid"]
                or observed_isolated["name"] != expected_isolated["name"]
                or observed_isolated["total_memory_bytes"]
                != expected_isolated["total_memory_bytes"]
                or observed_isolated["visible_index"] != 0
            ):
                raise ValueError("isolated worker GPU identity differs from authorization")
        elif observed_visible_gpus != expected_visible_gpus:
            raise ValueError("container GPU inventory differs from four-card authorization")
    elif observed_visible_gpus != expected_visible_gpus:
        raise ValueError("caller-supplied four-GPU identity differs from current runtime")
    expected_python = str(
        _absolute_normalized_path(python_executable, label="Python executable").resolve()
    )
    if observed.get("python_executable") != expected_python:
        raise ValueError("caller-supplied Python executable is not current runtime")
    if observed.get("software_versions") != _software_versions(software_versions):
        raise ValueError("caller-supplied software versions differ from current runtime")


def _canonical_argv(
    *,
    repository_root: Path,
    python_executable: str,
    envelope_path: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    python_path = _absolute_normalized_path(
        python_executable, label="Python executable"
    )
    worker_script = repository_root.joinpath(*PurePosixPath(WORKER_ENTRYPOINT).parts)
    aggregate_script = repository_root.joinpath(
        *PurePosixPath(AGGREGATE_ENTRYPOINT).parts
    )
    for path, label in (
        (worker_script, "worker entrypoint"),
        (aggregate_script, "aggregate entrypoint"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} is missing")
    workers = [
        {
            "argv": [
                str(python_path),
                str(worker_script),
                "--execution-envelope",
                envelope_path,
                "--worker-index",
                str(worker_index),
            ],
            "worker_index": worker_index,
        }
        for worker_index in range(WORKER_COUNT)
    ]
    aggregate = [
        str(python_path),
        str(aggregate_script),
        "--execution-envelope",
        envelope_path,
    ]
    return workers, aggregate


def _assert_no_forbidden_serialized_keys(value: Any, *, path: str = "$env") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _FORBIDDEN_SERIALIZED_KEYS:
                raise ValueError(f"execution envelope contains forbidden field {path}.{key}")
            _assert_no_forbidden_serialized_keys(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _assert_no_forbidden_serialized_keys(child, path=f"{path}[{index}]")


def build_set_utility_throughput_pilot_envelope_v1(
    *,
    repository_root: str | Path,
    source_git_revision: str,
    run_root: str | Path,
    python_executable: str,
    preflight_started_at_utc: str,
    preflight_completed_at_utc: str,
    preflight_evidence_path: str | Path,
    host_alias: str,
    hostname: str,
    container_id: str,
    container_image_reference: str,
    container_image_digest: str,
    driver_version: str,
    gpus: Sequence[Mapping[str, Any]],
    model_local_path: str | Path,
    processor_local_root: str | Path,
    software_versions: Mapping[str, Any],
    materialized_at_utc: str | None = None,
    inspect_local_artifacts: bool = True,
    verify_current_environment: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("repository root must be an absolute real directory")
    if _COMMIT.fullmatch(source_git_revision) is None:
        raise ValueError("source Git revision must be one full lowercase commit")
    source_contract = load_train_only_throughput_pilot_v1_contract(
        repository_root=root
    )
    source = source_contract.data
    _validate_source_authorization(source)
    if host_alias != HOST_ALIAS or hostname != HOSTNAME:
        raise ValueError("throughput pilot execution is authorized only on hyper00")
    if _CONTAINER_ID.fullmatch(container_id) is None:
        raise ValueError("container id must be one full lowercase Docker id")
    if not isinstance(container_image_reference, str) or not container_image_reference:
        raise ValueError("container image reference must be explicit")
    if _IMAGE_DIGEST.fullmatch(container_image_digest) is None:
        raise ValueError("container image digest must be sha256:<64 lowercase hex>")

    if not inspect_local_artifacts:
        raise ValueError("an authorizing execution envelope requires full artifact hashing")
    started = _timestamp(preflight_started_at_utc, label="preflight start")
    completed = _timestamp(preflight_completed_at_utc, label="preflight completion")
    evidence_path = _absolute_normalized_path(
        str(preflight_evidence_path), label="preflight evidence path"
    )
    normalized_gpus = _normalize_gpus(gpus, driver_version=driver_version)
    normalized_versions = _software_versions(software_versions)
    _, evidence = _validated_preflight_evidence(
        evidence_path,
        host_alias=host_alias,
        hostname=hostname,
        container_id=container_id,
        container_image_reference=container_image_reference,
        container_image_digest=container_image_digest,
        driver_version=driver_version,
        gpus=normalized_gpus,
        python_executable=python_executable,
        software_versions=normalized_versions,
        preflight_started_at_utc=preflight_started_at_utc,
        preflight_completed_at_utc=preflight_completed_at_utc,
    )
    if verify_current_environment:
        _validate_current_execution_identity(
            hostname=hostname,
            container_id=container_id,
            driver_version=driver_version,
            gpus=normalized_gpus,
            python_executable=python_executable,
            software_versions=normalized_versions,
        )
    layout = canonical_run_layout(run_root)
    envelope_path = layout["envelope_path"]
    argv_workers, aggregate_argv = _canonical_argv(
        repository_root=root,
        python_executable=python_executable,
        envelope_path=envelope_path,
    )

    model_path = _absolute_normalized_path(str(model_local_path), label="model path")
    processor_root = _absolute_normalized_path(
        str(processor_local_root), label="processor root"
    )
    model_expected = _model_expected_inventory(root, source)
    processor_source = _mapping(
        _mapping(source.get("inputs"), label="source inputs").get(
            "processor_publication"
        ),
        label="processor publication",
    )
    processor_expected = [
        dict(_mapping(value, label="processor formal file"))
        for value in _sequence(
            processor_source.get("formal_file_inventory"),
            label="processor formal inventory",
        )
    ]
    model_inventory = _inventory_tree(
        model_path,
        model_expected,
        allow_file_symlinks=True,
        label="model snapshot",
    )
    processor_inventory = _inventory_tree(
        processor_root,
        processor_expected,
        allow_file_symlinks=False,
        label="processor artifact",
    )
    model_stat_identities = _stat_identity_inventory(
        model_path,
        model_inventory,
        allow_file_symlinks=True,
        label="model snapshot",
    )
    processor_stat_identities = _stat_identity_inventory(
        processor_root,
        processor_inventory,
        allow_file_symlinks=False,
        label="processor artifact",
    )
    materialized_text = materialized_at_utc or utc_now()
    materialized = _timestamp(materialized_text, label="materialization time")
    if not started <= completed <= materialized:
        raise ValueError("preflight/materialization timestamps are not monotonic")
    if (materialized - completed).total_seconds() > PREFLIGHT_MAX_AGE_SECONDS:
        raise ValueError("GPU preflight is stale after artifact validation")

    pilot = _mapping(source.get("pilot"), label="source pilot")
    source_worker_mapping = [
        dict(_mapping(value, label="source worker mapping"))
        for value in _sequence(pilot.get("worker_mapping"), label="worker mapping")
    ]
    worker_layout = {
        record["worker_index"]: record for record in layout["workers"]
    }
    argv_by_worker = {
        record["worker_index"]: record["argv"] for record in argv_workers
    }
    workers: list[dict[str, Any]] = []
    for worker_index in range(WORKER_COUNT):
        gpu = normalized_gpus[worker_index]
        source_worker = source_worker_mapping[worker_index]
        if source_worker.get("worker_index") != worker_index:
            raise ValueError("source worker mapping order drifted")
        workers.append(
            {
                "argv": argv_by_worker[worker_index],
                "cuda_visible_devices": gpu["uuid"],
                "gpu_host_index": gpu["host_index"],
                "gpu_uuid": gpu["uuid"],
                "gpu_visible_index_before_worker_isolation": gpu["visible_index"],
                "output": worker_layout[worker_index],
                "state_ids": list(source_worker["state_ids"]),
                "worker_index": worker_index,
            }
        )

    config_binding = {
        "path": CANONICAL_CONFIG_PATH,
        "sha256": source_contract.config_sha256,
        "size_bytes": (root / CANONICAL_CONFIG_PATH).stat().st_size,
    }
    envelope = {
        "artifact_validation": {
            "completed_at_utc": materialized_text,
            "method": ARTIFACT_VALIDATION_METHOD,
            "model": {
                "file_count": len(model_inventory),
                "file_inventory_sha256": sha256_bytes(
                    canonical_json_bytes(model_inventory)
                ),
                "stat_identities": model_stat_identities,
                "stat_identity_sha256": sha256_bytes(
                    canonical_json_bytes(model_stat_identities)
                ),
            },
            "processor": {
                "file_count": len(processor_inventory),
                "file_inventory_sha256": sha256_bytes(
                    canonical_json_bytes(processor_inventory)
                ),
                "stat_identities": processor_stat_identities,
                "stat_identity_sha256": sha256_bytes(
                    canonical_json_bytes(processor_stat_identities)
                ),
            },
            "status": ARTIFACT_VALIDATION_STATUS,
        },
        "artifacts": {
            "model": {
                "file_count": len(model_inventory),
                "file_inventory": model_inventory,
                "file_inventory_sha256": sha256_bytes(
                    canonical_json_bytes(model_inventory)
                ),
                "local_path": str(model_path),
                "repo": _mapping(
                    _mapping(source["inputs"], label="source inputs")[
                        "model_snapshot_manifest"
                    ],
                    label="source model",
                )["repo"],
                "revision": _mapping(
                    _mapping(source["inputs"], label="source inputs")[
                        "model_snapshot_manifest"
                    ],
                    label="source model",
                )["revision"],
                "snapshot_manifest": {
                    key: _mapping(
                        _mapping(source["inputs"], label="source inputs")[
                            "model_snapshot_manifest"
                        ],
                        label="source model",
                    )[key]
                    for key in ("byte_count", "path", "sha256")
                },
                "total_byte_count": sum(
                    record["size_bytes"] for record in model_inventory
                ),
            },
            "processor": {
                "file_count": len(processor_inventory),
                "file_inventory": processor_inventory,
                "file_inventory_sha256": sha256_bytes(
                    canonical_json_bytes(processor_inventory)
                ),
                "formal_inventory_sha256": processor_source[
                    "formal_file_inventory_sha256"
                ],
                "hf_prefix": processor_source["hf_prefix"],
                "hf_repo": processor_source["hf_repo"],
                "hf_revision": processor_source["hf_revision"],
                "hf_tag": processor_source["hf_tag"],
                "local_root": str(processor_root),
                "total_byte_count": sum(
                    record["size_bytes"] for record in processor_inventory
                ),
            },
        },
        "authorization": {
            "authorized_native_call_ceiling": NATIVE_CALL_CEILING,
            "exact_four_gpu_execution_only": True,
            "may_generate_restoration_labels": False,
            "may_run_closed_loop": False,
            "may_train_predictor": False,
            "metric_only_output_required": True,
            "source_config_authorizes_gpu_execution": False,
            "this_validated_envelope_authorizes_gpu_execution": True,
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
                        "aggregate_claim_path",
                        "aggregate_result_path",
                        "aggregate_stderr_log_path",
                        "aggregate_stdout_log_path",
                    )
                },
            },
            "allowed_microbatch_order": list(pilot["allowed_microbatch_order"]),
            "claim_before_model_load_required": True,
            "decision_rule": source["deployment_decision_rule"],
            "native_call_ceiling": pilot["native_call_ceiling"],
            "no_retry": True,
            "no_top_up": True,
            "output_layout": layout,
            "worker_count": WORKER_COUNT,
            "worker_mapping": workers,
        },
        "gpus": {
            "device_count": GPU_COUNT,
            "devices": normalized_gpus,
            "driver_version": driver_version,
            "homogeneous_h200_and_driver": True,
        },
        "host": {"alias": host_alias, "hostname": hostname},
        "materialized_at_utc": materialized_text,
        "preflight": {
            "completed_at_utc": preflight_completed_at_utc,
            "evidence": evidence,
            "freshness_maximum_seconds": PREFLIGHT_MAX_AGE_SECONDS,
            "started_at_utc": preflight_started_at_utc,
        },
        "protocol_id": PROTOCOL_ID,
        "runtime": {
            "python_executable": str(
                _absolute_normalized_path(
                    python_executable, label="Python executable"
                )
            ),
            "software_versions": normalized_versions,
        },
        "schema_version": SCHEMA_VERSION,
        "source": {
            "branch": SOURCE_BRANCH,
            "config": config_binding,
            "execution_config_path": CANONICAL_GIT_ENVELOPE_PATH,
            "git_revision": source_git_revision,
            "remote": SOURCE_REMOTE,
            "source_authorization": {
                "load_policy_or_vision_model": False,
                "run_gpu_or_cuda": False,
                "write_pilot_result": False,
            },
            "source_status": source["status"],
        },
        "status": AUTHORIZED_STATUS,
    }
    validate_set_utility_throughput_pilot_envelope_v1(
        envelope,
        repository_root=root,
        verify_repository=False,
        verify_local_artifacts=False,
        require_fresh_preflight=True,
        now_utc=materialized_text,
    )
    return envelope


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"Git inspection failed: {detail}")
    return result.stdout.strip()


def validate_clean_pushed_source(
    *, repository_root: str | Path, expected_git_revision: str
) -> dict[str, str]:
    root = Path(repository_root).resolve()
    if _COMMIT.fullmatch(expected_git_revision) is None:
        raise ValueError("expected Git revision is invalid")
    head = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    tracking = _git(root, "rev-parse", f"refs/remotes/{SOURCE_REMOTE}/{SOURCE_BRANCH}")
    remote_output = _git(
        root,
        "ls-remote",
        "--exit-code",
        SOURCE_REMOTE,
        f"refs/heads/{SOURCE_BRANCH}",
    )
    fields = remote_output.split()
    if (
        head != expected_git_revision
        or branch != SOURCE_BRANCH
        or status
        or tracking != expected_git_revision
        or len(fields) != 2
        or fields[0] != expected_git_revision
        or fields[1] != f"refs/heads/{SOURCE_BRANCH}"
    ):
        raise ValueError("execution envelope requires clean pushed canonical main")
    config_payload = (root / CANONICAL_CONFIG_PATH).read_bytes()
    committed_payload = subprocess.run(
        ["git", "show", f"{expected_git_revision}:{CANONICAL_CONFIG_PATH}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if config_payload != committed_payload:
        raise ValueError("source config bytes differ from the committed Git blob")
    return {
        "branch": branch,
        "git_revision": head,
        "remote": SOURCE_REMOTE,
        "remote_main_git_revision": fields[0],
    }


def validate_committed_execution_envelope_lifecycle(
    *,
    repository_root: str | Path,
    expected_source_git_revision: str,
    data_envelope_path: str | Path,
    expected_envelope: Mapping[str, Any],
) -> dict[str, str]:
    """Validate the source-A/envelope-B two-commit execution lifecycle."""
    root = Path(repository_root).resolve()
    if _COMMIT.fullmatch(expected_source_git_revision) is None:
        raise ValueError("source-A Git revision is invalid")
    head = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    tracking = _git(root, "rev-parse", f"refs/remotes/{SOURCE_REMOTE}/{SOURCE_BRANCH}")
    remote_output = _git(
        root,
        "ls-remote",
        "--exit-code",
        SOURCE_REMOTE,
        f"refs/heads/{SOURCE_BRANCH}",
    )
    fields = remote_output.split()
    parents = _git(root, "rev-list", "--parents", "-n", "1", head).split()
    changed = _git(
        root,
        "diff",
        "--name-only",
        "--no-renames",
        expected_source_git_revision,
        head,
    ).splitlines()
    if (
        _COMMIT.fullmatch(head) is None
        or head == expected_source_git_revision
        or branch != SOURCE_BRANCH
        or status
        or tracking != head
        or len(fields) != 2
        or fields[0] != head
        or fields[1] != f"refs/heads/{SOURCE_BRANCH}"
        or parents != [head, expected_source_git_revision]
        or changed != [CANONICAL_GIT_ENVELOPE_PATH]
    ):
        raise ValueError(
            "runtime requires clean pushed envelope commit B directly above source A"
        )

    payload = canonical_pretty_json_bytes(expected_envelope)
    git_path = root.joinpath(*PurePosixPath(CANONICAL_GIT_ENVELOPE_PATH).parts)
    if git_path.is_symlink() or not git_path.is_file() or git_path.read_bytes() != payload:
        raise ValueError("Git execution envelope bytes differ from validated envelope")
    committed = subprocess.run(
        ["git", "show", f"{head}:{CANONICAL_GIT_ENVELOPE_PATH}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    data_path = _absolute_normalized_path(
        str(data_envelope_path), label="data execution envelope path"
    )
    if data_path.is_symlink() or not data_path.is_file():
        raise ValueError("canonical /data execution envelope is missing")
    if committed != payload or data_path.read_bytes() != payload:
        raise ValueError("commit-B, Git, and /data envelope bytes must be identical")
    source_config_blob = subprocess.run(
        [
            "git",
            "show",
            f"{expected_source_git_revision}:{CANONICAL_CONFIG_PATH}",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    source_binding = _mapping(
        _mapping(expected_envelope.get("source"), label="envelope source").get(
            "config"
        ),
        label="source config binding",
    )
    if (
        len(source_config_blob) != source_binding.get("size_bytes")
        or sha256_bytes(source_config_blob) != source_binding.get("sha256")
    ):
        raise ValueError("source-A config blob differs from the envelope binding")
    return {
        "envelope_git_revision": head,
        "envelope_path": CANONICAL_GIT_ENVELOPE_PATH,
        "source_git_revision": expected_source_git_revision,
        "status": "VALID_CLEAN_PUSHED_ENVELOPE_COMMIT_B",
    }


def validate_set_utility_throughput_pilot_envelope_v1(
    envelope: Mapping[str, Any],
    *,
    repository_root: str | Path,
    verify_repository: bool = True,
    verify_local_artifacts: bool | str = "stat",
    require_fresh_preflight: bool = True,
    now_utc: str | None = None,
    verify_current_environment: bool = False,
    allow_single_gpu_isolation: bool = False,
) -> dict[str, Any]:
    if not isinstance(envelope, Mapping):
        raise TypeError("execution envelope must be one mapping")
    _exact_keys(
        envelope,
        {
            "artifact_validation",
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
        },
        label="execution envelope",
    )
    if (
        envelope.get("schema_version") != SCHEMA_VERSION
        or envelope.get("protocol_id") != PROTOCOL_ID
        or envelope.get("status") != AUTHORIZED_STATUS
    ):
        raise ValueError("execution envelope identity/status drifted")
    _assert_no_forbidden_serialized_keys(envelope)
    if verify_local_artifacts is True:
        local_verification_mode = "full"
    elif verify_local_artifacts is False:
        local_verification_mode = "none"
    elif verify_local_artifacts in {"full", "stat"}:
        local_verification_mode = verify_local_artifacts
    else:
        raise ValueError("local artifact verification mode must be full, stat, or none")

    root = Path(repository_root).resolve()
    contract = load_train_only_throughput_pilot_v1_contract(repository_root=root)
    source_data = contract.data
    _validate_source_authorization(source_data)
    source = _mapping(envelope.get("source"), label="envelope source")
    _exact_keys(
        source,
        {
            "branch",
            "config",
            "execution_config_path",
            "git_revision",
            "remote",
            "source_authorization",
            "source_status",
        },
        label="envelope source",
    )
    expected_config = {
        "path": CANONICAL_CONFIG_PATH,
        "sha256": contract.config_sha256,
        "size_bytes": (root / CANONICAL_CONFIG_PATH).stat().st_size,
    }
    if (
        source.get("branch") != SOURCE_BRANCH
        or source.get("remote") != SOURCE_REMOTE
        or source.get("execution_config_path") != CANONICAL_GIT_ENVELOPE_PATH
        or source.get("config") != expected_config
        or source.get("source_status") != source_data["status"]
        or source.get("source_authorization")
        != {
            "load_policy_or_vision_model": False,
            "run_gpu_or_cuda": False,
            "write_pilot_result": False,
        }
        or not isinstance(source.get("git_revision"), str)
        or _COMMIT.fullmatch(source["git_revision"]) is None
    ):
        raise ValueError("execution envelope source binding drifted")
    if verify_repository:
        validate_committed_execution_envelope_lifecycle(
            repository_root=root,
            expected_source_git_revision=source["git_revision"],
            data_envelope_path=_mapping(
                _mapping(envelope.get("execution"), label="execution").get(
                    "output_layout"
                ),
                label="output layout",
            ).get("envelope_path"),
            expected_envelope=envelope,
        )

    authorization = _mapping(
        envelope.get("authorization"), label="envelope authorization"
    )
    if dict(authorization) != {
        "authorized_native_call_ceiling": NATIVE_CALL_CEILING,
        "exact_four_gpu_execution_only": True,
        "may_generate_restoration_labels": False,
        "may_run_closed_loop": False,
        "may_train_predictor": False,
        "metric_only_output_required": True,
        "source_config_authorizes_gpu_execution": False,
        "this_validated_envelope_authorizes_gpu_execution": True,
    }:
        raise ValueError("execution envelope authorization scope drifted")

    host = _mapping(envelope.get("host"), label="host")
    if dict(host) != {"alias": HOST_ALIAS, "hostname": HOSTNAME}:
        raise ValueError("execution envelope host drifted")
    container = _mapping(envelope.get("container"), label="container")
    _exact_keys(
        container,
        {"id", "image_digest", "image_reference"},
        label="container",
    )
    if (
        not isinstance(container.get("id"), str)
        or _CONTAINER_ID.fullmatch(container["id"]) is None
        or not isinstance(container.get("image_digest"), str)
        or _IMAGE_DIGEST.fullmatch(container["image_digest"]) is None
        or not isinstance(container.get("image_reference"), str)
        or not container["image_reference"]
    ):
        raise ValueError("container identity is invalid")

    gpu_section = _mapping(envelope.get("gpus"), label="GPUs")
    _exact_keys(
        gpu_section,
        {"device_count", "devices", "driver_version", "homogeneous_h200_and_driver"},
        label="GPUs",
    )
    if (
        gpu_section.get("device_count") != GPU_COUNT
        or gpu_section.get("homogeneous_h200_and_driver") is not True
        or not isinstance(gpu_section.get("driver_version"), str)
    ):
        raise ValueError("GPU execution shape drifted")
    normalized_gpus = _normalize_gpus(
        [
            _mapping(value, label="GPU")
            for value in _sequence(gpu_section.get("devices"), label="GPU devices")
        ],
        driver_version=gpu_section["driver_version"],
    )
    if list(gpu_section["devices"]) != normalized_gpus:
        raise ValueError("GPU device records are not canonical")

    runtime = _mapping(envelope.get("runtime"), label="runtime")
    _exact_keys(
        runtime,
        {"python_executable", "software_versions"},
        label="runtime",
    )
    python_executable = str(
        _absolute_normalized_path(
            runtime.get("python_executable"), label="Python executable"
        )
    )
    if dict(
        _mapping(runtime.get("software_versions"), label="software versions")
    ) != _software_versions(
        _mapping(runtime.get("software_versions"), label="software versions")
    ):
        raise ValueError("software version record is not canonical")

    materialized = _timestamp(
        envelope.get("materialized_at_utc"), label="materialization time"
    )
    preflight = _mapping(envelope.get("preflight"), label="preflight")
    _exact_keys(
        preflight,
        {
            "completed_at_utc",
            "evidence",
            "freshness_maximum_seconds",
            "started_at_utc",
        },
        label="preflight",
    )
    started = _timestamp(preflight.get("started_at_utc"), label="preflight start")
    completed = _timestamp(
        preflight.get("completed_at_utc"), label="preflight completion"
    )
    if (
        preflight.get("freshness_maximum_seconds") != PREFLIGHT_MAX_AGE_SECONDS
        or not started <= completed <= materialized
        or (materialized - completed).total_seconds() > PREFLIGHT_MAX_AGE_SECONDS
    ):
        raise ValueError("preflight freshness binding drifted")
    if require_fresh_preflight:
        now = _timestamp(now_utc or utc_now(), label="validation time")
        if now < materialized or (now - completed).total_seconds() > PREFLIGHT_MAX_AGE_SECONDS:
            raise ValueError("execution envelope preflight is no longer fresh")
    evidence = _mapping(preflight.get("evidence"), label="preflight evidence")
    _exact_keys(
        evidence,
        {"path", "sha256", "size_bytes"},
        label="preflight evidence",
    )
    evidence_path = _absolute_normalized_path(
        evidence.get("path"), label="preflight evidence path"
    )
    if local_verification_mode != "none":
        _, observed_evidence_binding = _validated_preflight_evidence(
            evidence_path,
            host_alias=host["alias"],
            hostname=host["hostname"],
            container_id=container["id"],
            container_image_reference=container["image_reference"],
            container_image_digest=container["image_digest"],
            driver_version=gpu_section["driver_version"],
            gpus=normalized_gpus,
            python_executable=python_executable,
            software_versions=_mapping(
                runtime.get("software_versions"), label="software versions"
            ),
            preflight_started_at_utc=preflight["started_at_utc"],
            preflight_completed_at_utc=preflight["completed_at_utc"],
        )
        if dict(evidence) != observed_evidence_binding:
            raise ValueError("preflight evidence bytes drifted")
    if verify_current_environment:
        _validate_current_execution_identity(
            hostname=host["hostname"],
            container_id=container["id"],
            driver_version=gpu_section["driver_version"],
            gpus=normalized_gpus,
            python_executable=python_executable,
            software_versions=_mapping(
                runtime.get("software_versions"), label="software versions"
            ),
            allow_single_gpu_isolation=allow_single_gpu_isolation,
        )

    execution = _mapping(envelope.get("execution"), label="execution")
    _exact_keys(
        execution,
        {
            "aggregate",
            "allowed_microbatch_order",
            "claim_before_model_load_required",
            "decision_rule",
            "native_call_ceiling",
            "no_retry",
            "no_top_up",
            "output_layout",
            "worker_count",
            "worker_mapping",
        },
        label="execution",
    )
    pilot = _mapping(source_data.get("pilot"), label="source pilot")
    if (
        execution.get("allowed_microbatch_order")
        != pilot["allowed_microbatch_order"]
        or execution.get("claim_before_model_load_required") is not True
        or execution.get("decision_rule")
        != source_data["deployment_decision_rule"]
        or execution.get("native_call_ceiling") != NATIVE_CALL_CEILING
        or execution.get("no_retry") is not True
        or execution.get("no_top_up") is not True
        or execution.get("worker_count") != WORKER_COUNT
    ):
        raise ValueError("exact execution semantics drifted")
    layout = _mapping(execution.get("output_layout"), label="output layout")
    expected_layout = canonical_run_layout(layout.get("run_root", ""))
    if dict(layout) != expected_layout:
        raise ValueError("canonical output/claim/log layout drifted")
    if evidence_path != canonical_preflight_evidence_path(layout["run_root"]):
        raise ValueError("preflight evidence path is not canonical for the run root")
    for key, value in layout.items():
        if key == "workers" or key == "run_root":
            continue
        _path_under(Path(layout["run_root"]), Path(value), label=key)

    argv_workers, aggregate_argv = _canonical_argv(
        repository_root=root,
        python_executable=python_executable,
        envelope_path=layout["envelope_path"],
    )
    source_workers = pilot["worker_mapping"]
    workers = _sequence(execution.get("worker_mapping"), label="execution workers")
    if len(workers) != WORKER_COUNT:
        raise ValueError("execution worker count drifted")
    for worker_index, raw in enumerate(workers):
        worker = _mapping(raw, label=f"worker {worker_index}")
        _exact_keys(
            worker,
            {
                "argv",
                "cuda_visible_devices",
                "gpu_host_index",
                "gpu_uuid",
                "gpu_visible_index_before_worker_isolation",
                "output",
                "state_ids",
                "worker_index",
            },
            label=f"worker {worker_index}",
        )
        gpu = normalized_gpus[worker_index]
        if worker != {
            "argv": argv_workers[worker_index]["argv"],
            "cuda_visible_devices": gpu["uuid"],
            "gpu_host_index": gpu["host_index"],
            "gpu_uuid": gpu["uuid"],
            "gpu_visible_index_before_worker_isolation": gpu["visible_index"],
            "output": expected_layout["workers"][worker_index],
            "state_ids": source_workers[worker_index]["state_ids"],
            "worker_index": worker_index,
        }:
            raise ValueError(f"worker {worker_index} exact mapping/argv drifted")
    aggregate = _mapping(execution.get("aggregate"), label="aggregate execution")
    expected_aggregate = {
        "argv": aggregate_argv,
        "output": {
            key: expected_layout[key]
            for key in (
                "aggregate_claim_path",
                "aggregate_result_path",
                "aggregate_stderr_log_path",
                "aggregate_stdout_log_path",
            )
        },
    }
    if dict(aggregate) != expected_aggregate:
        raise ValueError("aggregate exact argv/output binding drifted")

    artifacts = _mapping(envelope.get("artifacts"), label="artifacts")
    _exact_keys(artifacts, {"model", "processor"}, label="artifacts")
    model = _mapping(artifacts.get("model"), label="model artifact")
    processor = _mapping(artifacts.get("processor"), label="processor artifact")
    _exact_keys(
        model,
        {
            "file_count",
            "file_inventory",
            "file_inventory_sha256",
            "local_path",
            "repo",
            "revision",
            "snapshot_manifest",
            "total_byte_count",
        },
        label="model artifact",
    )
    _exact_keys(
        processor,
        {
            "file_count",
            "file_inventory",
            "file_inventory_sha256",
            "formal_inventory_sha256",
            "hf_prefix",
            "hf_repo",
            "hf_revision",
            "hf_tag",
            "local_root",
            "total_byte_count",
        },
        label="processor artifact",
    )
    model_inventory = [
        dict(_mapping(value, label="model inventory file"))
        for value in _sequence(model.get("file_inventory"), label="model inventory")
    ]
    expected_model_inventory = _model_expected_inventory(root, source_data)
    source_model = source_data["inputs"]["model_snapshot_manifest"]
    expected_model_projection = {
        "file_count": len(expected_model_inventory),
        "file_inventory": expected_model_inventory,
        "file_inventory_sha256": sha256_bytes(
            canonical_json_bytes(expected_model_inventory)
        ),
        "local_path": model.get("local_path"),
        "repo": source_model["repo"],
        "revision": source_model["revision"],
        "snapshot_manifest": {
            key: source_model[key] for key in ("byte_count", "path", "sha256")
        },
        "total_byte_count": sum(
            record["size_bytes"] for record in expected_model_inventory
        ),
    }
    if dict(model) != expected_model_projection:
        raise ValueError("model snapshot identity drifted")

    source_processor = source_data["inputs"]["processor_publication"]
    processor_inventory = [
        dict(_mapping(value, label="processor inventory file"))
        for value in _sequence(
            processor.get("file_inventory"), label="processor inventory"
        )
    ]
    expected_processor_inventory = [
        dict(value) for value in source_processor["formal_file_inventory"]
    ]
    expected_processor_projection = {
        "file_count": source_processor["formal_file_count"],
        "file_inventory": expected_processor_inventory,
        "file_inventory_sha256": sha256_bytes(
            canonical_json_bytes(expected_processor_inventory)
        ),
        "formal_inventory_sha256": source_processor[
            "formal_file_inventory_sha256"
        ],
        "hf_prefix": source_processor["hf_prefix"],
        "hf_repo": source_processor["hf_repo"],
        "hf_revision": source_processor["hf_revision"],
        "hf_tag": source_processor["hf_tag"],
        "local_root": processor.get("local_root"),
        "total_byte_count": source_processor["formal_total_byte_count"],
    }
    if dict(processor) != expected_processor_projection:
        raise ValueError("processor local/HF identity drifted")

    artifact_validation = _mapping(
        envelope.get("artifact_validation"), label="artifact validation receipt"
    )
    _exact_keys(
        artifact_validation,
        {"completed_at_utc", "method", "model", "processor", "status"},
        label="artifact validation receipt",
    )
    artifact_completed = _timestamp(
        artifact_validation.get("completed_at_utc"),
        label="artifact validation completion",
    )
    if (
        artifact_validation.get("status") != ARTIFACT_VALIDATION_STATUS
        or artifact_validation.get("method") != ARTIFACT_VALIDATION_METHOD
        or artifact_completed != materialized
    ):
        raise ValueError("artifact validation receipt identity/timestamp drifted")
    model_receipt = _mapping(
        artifact_validation.get("model"), label="model validation receipt"
    )
    processor_receipt = _mapping(
        artifact_validation.get("processor"), label="processor validation receipt"
    )
    receipt_keys = {
        "file_count",
        "file_inventory_sha256",
        "stat_identities",
        "stat_identity_sha256",
    }
    _exact_keys(model_receipt, receipt_keys, label="model validation receipt")
    _exact_keys(
        processor_receipt, receipt_keys, label="processor validation receipt"
    )
    for receipt, inventory, label in (
        (model_receipt, model_inventory, "model validation receipt"),
        (processor_receipt, processor_inventory, "processor validation receipt"),
    ):
        stat_identities = [
            dict(_mapping(value, label=f"{label} stat identity"))
            for value in _sequence(
                receipt.get("stat_identities"), label=f"{label} stat identities"
            )
        ]
        if len(stat_identities) != len(inventory):
            raise ValueError(f"{label} stat identity count drifted")
        for identity, expected_file in zip(
            stat_identities, inventory, strict=True
        ):
            _exact_keys(
                identity,
                {
                    "device",
                    "inode",
                    "mode",
                    "mtime_ns",
                    "path",
                    "resolved_path",
                    "size_bytes",
                },
                label=f"{label} stat identity",
            )
            if (
                identity.get("path") != expected_file["path"]
                or identity.get("size_bytes") != expected_file["size_bytes"]
                or any(
                    type(identity.get(key)) is not int or identity[key] < 0
                    for key in ("device", "inode", "mode", "mtime_ns")
                )
                or not isinstance(identity.get("resolved_path"), str)
                or not Path(identity["resolved_path"]).is_absolute()
            ):
                raise ValueError(f"{label} stat identity is invalid")
        if (
            receipt.get("file_count") != len(inventory)
            or receipt.get("file_inventory_sha256")
            != sha256_bytes(canonical_json_bytes(inventory))
            or receipt.get("stat_identity_sha256")
            != sha256_bytes(canonical_json_bytes(stat_identities))
        ):
            raise ValueError(f"{label} binding drifted")

    model_path = _absolute_normalized_path(
        model.get("local_path"), label="model local path"
    )
    processor_root = _absolute_normalized_path(
        processor.get("local_root"), label="processor local root"
    )
    if local_verification_mode == "full":
        if _inventory_tree(
            model_path,
            model_inventory,
            allow_file_symlinks=True,
            label="model snapshot",
        ) != model_inventory:
            raise ValueError("model local snapshot validation drifted")
        if _inventory_tree(
            processor_root,
            processor_inventory,
            allow_file_symlinks=False,
            label="processor artifact",
        ) != processor_inventory:
            raise ValueError("processor local artifact validation drifted")
    if local_verification_mode in {"full", "stat"}:
        _validate_stat_identity_receipt(
            model_path,
            model_inventory,
            model_receipt,
            allow_file_symlinks=True,
            label="model snapshot",
        )
        _validate_stat_identity_receipt(
            processor_root,
            processor_inventory,
            processor_receipt,
            allow_file_symlinks=False,
            label="processor artifact",
        )

    return {
        "container_id": container["id"],
        "gpu_count": GPU_COUNT,
        "host": HOST_ALIAS,
        "local_artifact_verification_mode": local_verification_mode,
        "model_revision": model["revision"],
        "native_call_ceiling": NATIVE_CALL_CEILING,
        "processor_hf_revision": processor["hf_revision"],
        "source_config_sha256": contract.config_sha256,
        "source_git_revision": source["git_revision"],
        "status": VALIDATION_STATUS,
        "worker_count": WORKER_COUNT,
    }


def load_set_utility_throughput_pilot_envelope_v1(
    path: str | Path,
    *,
    repository_root: str | Path,
    verify_repository: bool = True,
    verify_local_artifacts: bool | str = "stat",
    require_fresh_preflight: bool = True,
    verify_current_environment: bool = False,
    allow_single_gpu_isolation: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = Path(path)
    envelope = load_strict_json_object(supplied, label="execution envelope")
    payload = supplied.read_bytes()
    if payload != canonical_pretty_json_bytes(envelope):
        raise ValueError("execution envelope must be canonical pretty JSON")
    execution = _mapping(envelope.get("execution"), label="execution")
    layout = _mapping(execution.get("output_layout"), label="output layout")
    if str(supplied) != layout.get("envelope_path"):
        raise ValueError("execution envelope must be loaded from its canonical path")
    validation = validate_set_utility_throughput_pilot_envelope_v1(
        envelope,
        repository_root=repository_root,
        verify_repository=verify_repository,
        verify_local_artifacts=verify_local_artifacts,
        require_fresh_preflight=require_fresh_preflight,
        verify_current_environment=verify_current_environment,
        allow_single_gpu_isolation=allow_single_gpu_isolation,
    )
    return envelope, validation


def load_set_utility_throughput_pilot_execution_envelope_v1(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
    verify_repository: bool = True,
    verify_local_artifacts: bool | str = "stat",
    require_fresh_preflight: bool = True,
    verify_current_environment: bool = True,
) -> dict[str, Any]:
    """Load, validate, and project the exact fields consumed by the runner."""
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    envelope, validation = load_set_utility_throughput_pilot_envelope_v1(
        path,
        repository_root=root,
        verify_repository=verify_repository,
        verify_local_artifacts=verify_local_artifacts,
        require_fresh_preflight=require_fresh_preflight,
        verify_current_environment=verify_current_environment,
        allow_single_gpu_isolation=True,
    )
    execution = envelope["execution"]
    workers = execution["worker_mapping"]
    return {
        "config_path": str(root / CANONICAL_CONFIG_PATH),
        "device_by_worker": {
            str(worker["worker_index"]): "cuda:0" for worker in workers
        },
        "envelope": envelope,
        "envelope_path": execution["output_layout"]["envelope_path"],
        "model_dir": envelope["artifacts"]["model"]["local_path"],
        "processor_root": envelope["artifacts"]["processor"]["local_root"],
        "repository_root": str(root),
        "run_root": execution["output_layout"]["run_root"],
        "validation": validation,
        "worker_gpu_uuid": {
            str(worker["worker_index"]): worker["gpu_uuid"] for worker in workers
        },
    }


def _write_exclusive_payload(destination: Path, payload: bytes) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(f"execution envelope already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_preflight_evidence_exclusive(
    *,
    run_root: str | Path,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    destination = canonical_preflight_evidence_path(run_root)
    payload = canonical_pretty_json_bytes(evidence)
    if _strict_json_object(payload, label="preflight evidence") != dict(evidence):
        raise ValueError("preflight evidence is not strict canonical JSON data")
    _write_exclusive_payload(destination, payload)
    return {
        "path": str(destination),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "status": PREFLIGHT_EVIDENCE_STATUS,
    }


def write_canonical_envelope_exclusive(path: str | Path, envelope: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    if not destination.is_absolute():
        raise ValueError("execution envelope output must be absolute")
    execution = _mapping(envelope.get("execution"), label="execution")
    layout = _mapping(execution.get("output_layout"), label="output layout")
    if str(destination) != layout.get("envelope_path"):
        raise ValueError("execution envelope output path is not canonical")
    payload = canonical_pretty_json_bytes(envelope)
    _write_exclusive_payload(destination, payload)
    return {
        "path": str(destination),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "status": AUTHORIZED_STATUS,
    }


def write_canonical_envelope_pair_exclusive(
    *,
    repository_root: str | Path,
    data_path: str | Path,
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    data_destination = Path(data_path)
    git_destination = root.joinpath(
        *PurePosixPath(CANONICAL_GIT_ENVELOPE_PATH).parts
    )
    layout = _mapping(
        _mapping(envelope.get("execution"), label="execution").get(
            "output_layout"
        ),
        label="output layout",
    )
    if str(data_destination) != layout.get("envelope_path"):
        raise ValueError("/data execution envelope output path is not canonical")
    if os.path.lexists(data_destination) or os.path.lexists(git_destination):
        raise FileExistsError("both canonical execution envelope paths must be fresh")
    payload = canonical_pretty_json_bytes(envelope)
    git_created = False
    data_created = False
    try:
        _write_exclusive_payload(git_destination, payload)
        git_created = True
        _write_exclusive_payload(data_destination, payload)
        data_created = True
    except BaseException:
        for created, owned in (
            (data_destination, data_created),
            (git_destination, git_created),
        ):
            try:
                if owned and created.is_file() and not created.is_symlink():
                    created.unlink()
            except OSError:
                pass
        raise
    return {
        "data_path": str(data_destination),
        "git_path": CANONICAL_GIT_ENVELOPE_PATH,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "status": AUTHORIZED_STATUS,
    }


__all__ = [
    "AGGREGATE_ENTRYPOINT",
    "ARTIFACT_VALIDATION_METHOD",
    "ARTIFACT_VALIDATION_STATUS",
    "AUTHORIZED_STATUS",
    "CANONICAL_GIT_ENVELOPE_PATH",
    "GPU_COUNT",
    "HOSTNAME",
    "HOST_ALIAS",
    "PREFLIGHT_MAX_AGE_SECONDS",
    "PREFLIGHT_EVIDENCE_PROTOCOL_ID",
    "PREFLIGHT_EVIDENCE_SCHEMA_VERSION",
    "PREFLIGHT_EVIDENCE_STATUS",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "VALIDATION_STATUS",
    "WORKER_ENTRYPOINT",
    "build_set_utility_throughput_pilot_preflight_evidence_v1",
    "build_set_utility_throughput_pilot_envelope_v1",
    "canonical_preflight_evidence_path",
    "canonical_run_layout",
    "load_set_utility_throughput_pilot_envelope_v1",
    "load_set_utility_throughput_pilot_execution_envelope_v1",
    "load_strict_json_object",
    "utc_now",
    "validate_clean_pushed_source",
    "validate_committed_execution_envelope_lifecycle",
    "validate_set_utility_throughput_pilot_envelope_v1",
    "write_canonical_envelope_exclusive",
    "write_canonical_envelope_pair_exclusive",
    "write_preflight_evidence_exclusive",
]
