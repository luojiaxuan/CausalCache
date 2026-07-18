"""Source-A/B boundary, local state, and HF publication for formal gate v1."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import socket
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.gate_v1_contract import derive_rosters, load_strict_json_object
from causalcache.gate_v1_formal_train import (
    FORMAL_JOIN_AUDIT_SHA256,
    FormalTrainingResult,
    _combined_operation_counts,
    _family_operation_counts,
    load_repaired_formal_training_inputs,
    load_safetensors_checkpoint,
    run_formal_training,
)
from causalcache.gate_v1_formal_train_contract import (
    CANONICAL_CONFIG_PATH,
    MANIFEST_TARGETS,
    PAYLOAD_TARGETS,
    PROTOCOL_ID,
    RUNNER_FREEZE_B_PATH,
    SCHEMA_VERSION,
    FormalTrainContract,
    canonical_json_bytes,
    pretty_json_bytes,
    sha256_bytes,
)
from causalcache.gate_v1_provenance import (
    ArtifactBinding,
    FrozenEnsembleProvenance,
    FrozenTrainingProvenance,
    SeedCheckpointProvenance,
    frozen_ensemble_provenance_from_manifest,
)
from causalcache.gate_v1_training import (
    FamilySelection,
    OOFTrial,
    family_selection_payload,
)


SOURCE_VALIDATION_STATUS = "VALID_GATE_V1_FORMAL58_TRAIN_SOURCE_A_V1"
RUNNER_FREEZE_STATUS = "FROZEN_GATE_V1_FORMAL58_TRAIN_EXECUTION_B_V1"
GLOBAL_CLAIM_STATUS = "CLAIMED_GATE_V1_FORMAL58_TRAIN_V1"
CACHE_INPUT_COMPLETION_STATUS = "COMPLETED_GATE_V1_FORMAL58_TRAIN_INPUT_V1"
TRAINING_ARTIFACT_COMPLETION_STATUS = (
    "COMPLETED_GATE_V1_FORMAL58_TRAIN_ARTIFACTS_V1"
)
REMOTE_BASE_STATUS = "CAPTURED_GATE_V1_FORMAL58_TRAIN_REMOTE_BASE_V1"
PAYLOAD_COMMIT_STATUS = "PUBLISHED_GATE_V1_FORMAL58_TRAIN_PAYLOAD_V1"
FINAL_COMPLETION_STATUS = "COMPLETED_GATE_V1_FORMAL58_TRAIN_PUBLICATION_V1"
RUN_STATUS = "VALID_GATE_V1_FORMAL58_TRAIN_PUBLICATION_V1"
VALIDATE_STATUS = "REVALIDATED_GATE_V1_FORMAL58_TRAIN_PUBLICATION_V1"
DOCKER_INSPECT_STATUS = "CAPTURED_GATE_V1_FORMAL58_TRAIN_DOCKER_INSPECT_V1"

_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ALLOWED_REPO_BASE_FILES = frozenset({".gitattributes", "README.md"})


@dataclass(frozen=True)
class SourceIdentity:
    head: str
    remote_main: str
    branch: str
    origin_url: str
    source_inventory: tuple[Mapping[str, Any], ...]
    loaded_module_inventory: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class RuntimeIdentity:
    python_implementation: str
    python_version: str
    machine: str
    torch_version: str
    safetensors_version: str
    huggingface_hub_version: str
    torch_num_threads: int
    torch_num_interop_threads: int
    thread_environment: Mapping[str, str]
    docker_inspect_receipt_sha256: str
    container_id: str
    container_name: str
    container_hostname: str
    image_id: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "machine": self.machine,
            "torch_version": self.torch_version,
            "safetensors_version": self.safetensors_version,
            "huggingface_hub_version": self.huggingface_hub_version,
            "torch_num_threads": self.torch_num_threads,
            "torch_num_interop_threads": self.torch_num_interop_threads,
            "thread_environment": dict(self.thread_environment),
            "docker_inspect_receipt_sha256": self.docker_inspect_receipt_sha256,
            "container_id": self.container_id,
            "container_name": self.container_name,
            "container_hostname": self.container_hostname,
            "image_id": self.image_id,
            "normalized_device_requests": [],
            "container_privileged": False,
            "container_runtime": "runc",
            "device": "cpu",
            "dtype": "float32",
            "torch_cuda_available": False,
            "torch_cuda_device_count": 0,
            "gpu_device_nodes": [],
            "nvidia_visible_devices": "void",
            "cuda_visible_devices": "",
        }


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value in {label}: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys drifted")


def _remote_target_partition(
    remote_files: Sequence[str],
) -> tuple[set[str], set[str]]:
    payload_present = set(PAYLOAD_TARGETS) & set(remote_files)
    manifest_present = set(MANIFEST_TARGETS) & set(remote_files)
    if payload_present not in (set(), set(PAYLOAD_TARGETS)) or (
        manifest_present and manifest_present != set(MANIFEST_TARGETS)
    ):
        raise ValueError("formal model remote is partial or conflicting")
    return payload_present, manifest_present


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} is not canonical relative POSIX")
    return value


def _regular_file_bytes(
    path: Path,
    *,
    label: str,
    mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if mode is not None and stat.S_IMODE(before.st_mode) != mode:
            raise ValueError(f"{label} mode drifted")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is missing or unsafe") from error
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
    if fingerprint(before) != fingerprint(after) or len(payload) != before.st_size:
        raise ValueError(f"{label} changed while being read")
    return payload, after


def _exclusive_or_identical(path: Path, payload: bytes, *, mode: int) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve(strict=True) != path.parent.absolute():
        raise ValueError(f"file parent traverses a symlink: {path.parent}")
    if path.exists() or path.is_symlink():
        existing, _ = _regular_file_bytes(path, label=str(path), mode=mode)
        if existing != payload:
            raise ValueError(f"pre-existing file differs: {path}")
        return False
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
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive write made no progress")
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    observed, _ = _regular_file_bytes(path, label=str(path), mode=mode)
    if observed != payload:
        raise ValueError(f"exclusive file readback differs: {path}")
    return True


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(f"git {' '.join(arguments)} failed")
    return result.stdout


def _source_paths(contract: FormalTrainContract) -> tuple[str, ...]:
    source = _mapping(contract.data["source_freeze"], "source freeze")
    raw = source.get("required_source_a_paths")
    if not isinstance(raw, list) or not raw:
        raise ValueError("required Source-A paths are missing")
    paths = tuple(_safe_relative(item, "Source-A path") for item in raw)
    if len(paths) != len(set(paths)) or RUNNER_FREEZE_B_PATH in paths:
        raise ValueError("Source-A path inventory is duplicated or includes B")
    return paths


def _source_inventory(
    root: Path, commit: str, paths: Sequence[str]
) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    for relative in paths:
        payload, _ = _regular_file_bytes(root / relative, label=relative)
        if payload != _git(root, "show", f"{commit}:{relative}"):
            raise ValueError(f"source blob differs from commit: {relative}")
        records.append(
            {"path": relative, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        )
    return tuple(records)


def _loaded_module_inventory(
    root: Path, commit: str
) -> tuple[Mapping[str, Any], ...]:
    code_root = root / "code"
    records: list[Mapping[str, Any]] = []
    for name, module in sorted(sys.modules.items()):
        if not (
            name == "causalcache"
            or name.startswith("causalcache.")
            or name == "scripts"
            or name.startswith("scripts.")
        ):
            continue
        source = getattr(module, "__file__", None)
        if not isinstance(source, str):
            raise ValueError(f"loaded project module lacks source: {name}")
        path = Path(source)
        if not path.is_absolute() or path.suffix != ".py":
            raise ValueError(f"loaded project module is not canonical source: {name}")
        try:
            relative = (Path("code") / path.relative_to(code_root)).as_posix()
        except ValueError as error:
            raise ValueError(f"loaded module escaped repository: {name}") from error
        payload, _ = _regular_file_bytes(path, label=name)
        if payload != _git(root, "show", f"{commit}:{relative}"):
            raise ValueError(f"loaded module differs from commit: {name}")
        records.append(
            {
                "module": name,
                "path": relative,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    if not records:
        raise ValueError("loaded module inventory is empty")
    return tuple(records)


def _live_remote_main(root: Path, remote: str) -> str:
    output = _git(
        root,
        "ls-remote",
        "--exit-code",
        remote,
        "refs/heads/main",
    ).decode("ascii")
    lines = output.splitlines()
    suffix = "\trefs/heads/main"
    if len(lines) != 1 or not lines[0].endswith(suffix):
        raise ValueError("canonical remote main response drifted")
    commit = lines[0][: -len(suffix)]
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("canonical remote main is not one commit")
    return commit


def validate_clean_pushed_source(
    contract: FormalTrainContract,
    *,
    expected_commit: str | None,
    require_live_remote: bool,
) -> SourceIdentity:
    root = contract.repository_root
    source = _mapping(contract.data["source_freeze"], "source freeze")
    branch = source["branch"]
    remote = source["origin_name"]
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    tracking = _git(root, "rev-parse", f"{remote}/{branch}").decode().strip()
    current_branch = _git(root, "branch", "--show-current").decode().strip()
    origin_url = _git(root, "remote", "get-url", remote).decode().strip()
    remote_main = _live_remote_main(root, remote) if require_live_remote else tracking
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if (
        _COMMIT.fullmatch(head) is None
        or (expected_commit is not None and head != expected_commit)
        or head != tracking
        or head != remote_main
        or current_branch != branch
        or origin_url != source["origin_url"]
        or dirty
    ):
        raise ValueError("formal train source must be clean pushed canonical main")
    return SourceIdentity(
        head=head,
        remote_main=remote_main,
        branch=current_branch,
        origin_url=origin_url,
        source_inventory=_source_inventory(root, head, _source_paths(contract)),
        loaded_module_inventory=_loaded_module_inventory(root, head),
    )


def _source_record(source: SourceIdentity) -> dict[str, Any]:
    return {
        "git_commit": source.head,
        "remote_main_git_commit": source.remote_main,
        "branch": source.branch,
        "origin_url": source.origin_url,
        "source_inventory": list(source.source_inventory),
        "source_inventory_sha256": sha256_bytes(
            canonical_json_bytes(source.source_inventory)
        ),
        "loaded_module_inventory": list(source.loaded_module_inventory),
        "loaded_module_inventory_sha256": sha256_bytes(
            canonical_json_bytes(source.loaded_module_inventory)
        ),
    }


def validate_source_a(
    contract: FormalTrainContract,
    *,
    expected_source_a_git_commit: str | None = None,
) -> Mapping[str, Any]:
    runner = contract.repository_root / RUNNER_FREEZE_B_PATH
    if runner.exists() or runner.is_symlink():
        raise ValueError("execution-B runner freeze must be absent from Source-A")
    source = validate_clean_pushed_source(
        contract,
        expected_commit=expected_source_a_git_commit,
        require_live_remote=False,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": SOURCE_VALIDATION_STATUS,
        "contract_sha256": contract.sha256,
        "source_a": _source_record(source),
        "training_executed": False,
        "execution_authorized": False,
    }


def _runner_freeze_payload(
    contract: FormalTrainContract, source: SourceIdentity
) -> Mapping[str, Any]:
    source_freeze = _mapping(contract.data["source_freeze"], "source freeze")
    prerequisites = list(source_freeze["git_prerequisites"])
    paths = list(_source_paths(contract))
    inventory_sha256 = sha256_bytes(canonical_json_bytes(source.source_inventory))
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": RUNNER_FREEZE_STATUS,
        "contract_sha256": contract.sha256,
        "source_a_git_commit": source.head,
        "execution_b_required_unique_diff": [RUNNER_FREEZE_B_PATH],
        "required_source_a_paths": paths,
        "required_source_a_paths_sha256": sha256_bytes(canonical_json_bytes(paths)),
        "git_prerequisites": prerequisites,
        "git_prerequisites_sha256": sha256_bytes(
            canonical_json_bytes(prerequisites)
        ),
        "source_blob_inventory": list(source.source_inventory),
        "source_blob_inventory_sha256": inventory_sha256,
        "source_a_inventory_sha256": inventory_sha256,
        "loaded_module_inventory": list(source.loaded_module_inventory),
        "loaded_module_inventory_sha256": sha256_bytes(
            canonical_json_bytes(source.loaded_module_inventory)
        ),
        "execution_authorized_after_clean_pushed_b_only": True,
    }


def materialize_runner_freeze(
    contract: FormalTrainContract,
    *,
    expected_source_a_git_commit: str | None = None,
) -> Mapping[str, Any]:
    validation = validate_source_a(
        contract, expected_source_a_git_commit=expected_source_a_git_commit
    )
    record = validation["source_a"]
    source = SourceIdentity(
        head=record["git_commit"],
        remote_main=record["remote_main_git_commit"],
        branch=record["branch"],
        origin_url=record["origin_url"],
        source_inventory=tuple(record["source_inventory"]),
        loaded_module_inventory=tuple(record["loaded_module_inventory"]),
    )
    payload = _runner_freeze_payload(contract, source)
    path = contract.repository_root / RUNNER_FREEZE_B_PATH
    _exclusive_or_identical(path, pretty_json_bytes(payload), mode=0o644)
    status_lines = tuple(
        line
        for line in _git(
            contract.repository_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        .decode("utf-8")
        .splitlines()
        if line
    )
    if status_lines != (f"?? {RUNNER_FREEZE_B_PATH}",):
        raise ValueError("runner freeze is not the unique Source-A worktree diff")
    return payload


def load_runner_freeze(contract: FormalTrainContract) -> Mapping[str, Any]:
    payload, _ = _regular_file_bytes(
        contract.repository_root / RUNNER_FREEZE_B_PATH,
        label="execution-B runner freeze",
        mode=0o644,
    )
    value = _strict_json(payload, label="execution-B runner freeze")
    if payload != pretty_json_bytes(value):
        raise ValueError("execution-B runner freeze is not canonical pretty JSON")
    _exact_keys(
        value,
        {
            "schema_version",
            "protocol_id",
            "status",
            "contract_sha256",
            "source_a_git_commit",
            "execution_b_required_unique_diff",
            "required_source_a_paths",
            "required_source_a_paths_sha256",
            "git_prerequisites",
            "git_prerequisites_sha256",
            "source_blob_inventory",
            "source_blob_inventory_sha256",
            "source_a_inventory_sha256",
            "loaded_module_inventory",
            "loaded_module_inventory_sha256",
            "execution_authorized_after_clean_pushed_b_only",
        },
        "execution-B runner freeze",
    )
    paths = list(_source_paths(contract))
    prerequisites = list(contract.data["source_freeze"]["git_prerequisites"])
    inventory = value.get("source_blob_inventory")
    modules = value.get("loaded_module_inventory")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != RUNNER_FREEZE_STATUS
        or value.get("contract_sha256") != contract.sha256
        or value.get("execution_b_required_unique_diff") != [RUNNER_FREEZE_B_PATH]
        or value.get("required_source_a_paths") != paths
        or value.get("required_source_a_paths_sha256")
        != sha256_bytes(canonical_json_bytes(paths))
        or value.get("git_prerequisites") != prerequisites
        or value.get("git_prerequisites_sha256")
        != sha256_bytes(canonical_json_bytes(prerequisites))
        or not isinstance(inventory, list)
        or value.get("source_blob_inventory_sha256")
        != sha256_bytes(canonical_json_bytes(inventory))
        or value.get("source_a_inventory_sha256")
        != sha256_bytes(canonical_json_bytes(inventory))
        or not isinstance(modules, list)
        or value.get("loaded_module_inventory_sha256")
        != sha256_bytes(canonical_json_bytes(modules))
        or value.get("execution_authorized_after_clean_pushed_b_only") is not True
    ):
        raise ValueError("execution-B runner freeze identity drifted")
    return value


def validate_execution_b_source(
    contract: FormalTrainContract,
    *,
    expected_execution_b_git_commit: str,
) -> SourceIdentity:
    freeze = load_runner_freeze(contract)
    source = validate_clean_pushed_source(
        contract,
        expected_commit=expected_execution_b_git_commit,
        require_live_remote=True,
    )
    source_a = freeze.get("source_a_git_commit")
    if not isinstance(source_a, str) or _COMMIT.fullmatch(source_a) is None:
        raise ValueError("runner freeze Source-A commit is invalid")
    parents = (
        _git(contract.repository_root, "rev-list", "--parents", "-n", "1", source.head)
        .decode("ascii")
        .strip()
        .split()
    )
    if parents != [source.head, source_a]:
        raise ValueError("Execution-B must be the direct single-parent child of Source-A")
    changed = tuple(
        line
        for line in _git(
            contract.repository_root,
            "diff",
            "--name-only",
            source_a,
            source.head,
        )
        .decode("utf-8")
        .splitlines()
        if line
    )
    if changed != (RUNNER_FREEZE_B_PATH,):
        raise ValueError("Execution-B differs from Source-A outside runner freeze")
    if source.source_inventory != tuple(freeze["source_blob_inventory"]):
        raise ValueError("Execution-B source blobs differ from Source-A freeze")
    frozen_modules = {item["module"]: item for item in freeze["loaded_module_inventory"]}
    current_modules = {item["module"]: item for item in source.loaded_module_inventory}
    if current_modules != frozen_modules:
        raise ValueError("Execution-B loaded modules differ from Source-A freeze")
    return source


def capture_docker_inspect_receipt(
    contract: FormalTrainContract,
    *,
    source: SourceIdentity,
    container_name: str,
    host_data_root: Path,
) -> Mapping[str, Any]:
    specification = _mapping(
        contract.runtime.get("docker_inspect_receipt"),
        "Docker inspect receipt specification",
    )
    prefix = specification["container_name_prefix"]
    if re.fullmatch(re.escape(prefix) + r"[0-9]{8}", container_name) is None:
        raise ValueError("formal container name does not match the frozen timestamp form")
    result = subprocess.run(
        ["docker", "inspect", "--type", "container", container_name],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError("Docker inspect failed for the formal container")

    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate Docker inspect key: {key}")
            value[key] = item
        return value

    try:
        decoded = json.loads(
            result.stdout.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite Docker inspect value: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Docker inspect did not return strict UTF-8 JSON") from error
    if not isinstance(decoded, list) or len(decoded) != 1:
        raise ValueError("Docker inspect must return exactly one container")
    inspect_record = _mapping(decoded[0], "Docker inspect record")
    host_config = _mapping(inspect_record.get("HostConfig"), "Docker HostConfig")
    config = _mapping(inspect_record.get("Config"), "Docker Config")
    state = _mapping(inspect_record.get("State"), "Docker State")
    container_id = inspect_record.get("Id")
    image_id = inspect_record.get("Image")
    hostname = config.get("Hostname")
    if (
        not isinstance(container_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or inspect_record.get("Name") != f"/{container_name}"
        or not isinstance(image_id, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
        or hostname != container_id[:12]
        or state.get("Running") is not True
        or host_config.get("DeviceRequests") not in (None, [])
        or host_config.get("Privileged") is not False
        or host_config.get("Runtime") != specification["runtime"]
    ):
        raise ValueError("Docker inspect runtime boundary drifted")
    mounts = inspect_record.get("Mounts")
    if not isinstance(mounts, list):
        raise ValueError("Docker inspect mounts are missing")
    data_mounts = [
        _mapping(item, "Docker data mount")
        for item in mounts
        if isinstance(item, Mapping)
        and item.get("Destination") == specification["data_mount_destination"]
    ]
    root = host_data_root.resolve(strict=True)
    if host_data_root.is_symlink() or not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("host data root must be a real directory")
    if len(data_mounts) != 1:
        raise ValueError("Docker inspect must contain one /data mount")
    data_mount = data_mounts[0]
    source_path = data_mount.get("Source")
    if (
        data_mount.get("Type") != "bind"
        or data_mount.get("RW") is not specification["data_mount_rw"]
        or not isinstance(source_path, str)
        or Path(source_path).resolve(strict=True) != root
    ):
        raise ValueError("Docker inspect /data mount drifted")
    receipt = {
        "schema_version": specification["schema_version"],
        "protocol_id": PROTOCOL_ID,
        "status": specification["status"],
        "contract_sha256": contract.sha256,
        "source": _source_record(source),
        "container": {
            "id": container_id,
            "name": container_name,
            "hostname": hostname,
            "image_id": image_id,
            "running": True,
        },
        "runtime": {
            "normalized_device_requests": [],
            "privileged": False,
            "runtime": host_config["Runtime"],
        },
        "data_mount": {
            "type": "bind",
            "source": str(root),
            "destination": specification["data_mount_destination"],
            "rw": True,
        },
        "training_executed": False,
        "execution_authorized": True,
    }
    receipt_path = _data_root_path(
        host_data_root,
        specification["path"],
        label="Docker inspect receipt",
    )
    _exclusive_or_identical(
        receipt_path,
        pretty_json_bytes(receipt),
        mode=int(specification["mode"]),
    )
    return {**receipt, "receipt_path": specification["path"]}


def validate_execution_runtime(
    contract: FormalTrainContract,
    *,
    source: SourceIdentity,
    data_root: Path,
) -> RuntimeIdentity:
    runtime = contract.runtime
    receipt_specification = _mapping(
        runtime.get("docker_inspect_receipt"),
        "Docker inspect receipt specification",
    )
    receipt_path = _data_root_path(
        data_root,
        receipt_specification["path"],
        label="Docker inspect receipt",
    )
    receipt_payload, _ = _regular_file_bytes(
        receipt_path,
        label="Docker inspect receipt",
        mode=int(receipt_specification["mode"]),
    )
    receipt = _strict_json(receipt_payload, label="Docker inspect receipt")
    container = _mapping(receipt.get("container"), "Docker receipt container")
    docker_runtime = _mapping(receipt.get("runtime"), "Docker receipt runtime")
    data_mount = _mapping(receipt.get("data_mount"), "Docker receipt data mount")
    _exact_keys(
        receipt,
        {
            "schema_version",
            "protocol_id",
            "status",
            "contract_sha256",
            "source",
            "container",
            "runtime",
            "data_mount",
            "training_executed",
            "execution_authorized",
        },
        "Docker inspect receipt",
    )
    _exact_keys(
        container,
        {"id", "name", "hostname", "image_id", "running"},
        "Docker receipt container",
    )
    _exact_keys(
        docker_runtime,
        {"normalized_device_requests", "privileged", "runtime"},
        "Docker receipt runtime",
    )
    _exact_keys(
        data_mount,
        {"type", "source", "destination", "rw"},
        "Docker receipt data mount",
    )
    if (
        receipt_payload != pretty_json_bytes(receipt)
        or receipt.get("schema_version") != receipt_specification["schema_version"]
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status") != receipt_specification["status"]
        or receipt.get("contract_sha256") != contract.sha256
        or receipt.get("source") != _source_record(source)
        or receipt.get("training_executed") is not False
        or receipt.get("execution_authorized") is not True
        or docker_runtime
        != {
            "normalized_device_requests": [],
            "privileged": False,
            "runtime": "runc",
        }
        or data_mount.get("destination") != "/data"
        or data_mount.get("rw") is not True
        or data_mount.get("type") != "bind"
        or container.get("running") is not True
        or container.get("hostname") != socket.gethostname()
    ):
        raise ValueError("Docker inspect receipt identity drifted")
    for key in ("id", "name", "hostname", "image_id"):
        if not isinstance(container.get(key), str) or not container[key]:
            raise ValueError("Docker inspect receipt container identity is malformed")
    if (
        re.fullmatch(r"[0-9a-f]{64}", container["id"]) is None
        or container["hostname"] != container["id"][:12]
        or re.fullmatch(r"sha256:[0-9a-f]{64}", container["image_id"]) is None
        or re.fullmatch(
            re.escape(receipt_specification["container_name_prefix"]) + r"[0-9]{8}",
            container["name"],
        )
        is None
    ):
        raise ValueError("Docker inspect receipt container identity drifted")
    nodes = sorted(
        str(path)
        for pattern in ("/dev/nvidia*", "/dev/dri/renderD*")
        for path in Path("/").glob(pattern.lstrip("/"))
        if path.exists() or path.is_symlink()
    )
    observed_environment = {
        key: os.environ.get(key) for key in runtime["thread_environment"]
    }
    if (
        nodes
        or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "void"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or observed_environment != dict(runtime["thread_environment"])
        or platform.python_implementation() != runtime["python_implementation"]
        or platform.python_version() != runtime["python_version"]
        or platform.machine() != runtime["machine"]
    ):
        raise ValueError("formal training host/runtime boundary drifted")
    import huggingface_hub
    import safetensors
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if (
        torch.__version__ != runtime["torch_version"]
        or safetensors.__version__ != runtime["safetensors_version"]
        or huggingface_hub.__version__ != runtime["huggingface_hub_version"]
        or torch.cuda.is_available()
        or torch.cuda.device_count() != 0
        or torch.get_num_threads() != 1
        or torch.get_num_interop_threads() != 1
        or not torch.are_deterministic_algorithms_enabled()
    ):
        raise ValueError("formal training framework/runtime boundary drifted")
    return RuntimeIdentity(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        machine=platform.machine(),
        torch_version=torch.__version__,
        safetensors_version=safetensors.__version__,
        huggingface_hub_version=huggingface_hub.__version__,
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        thread_environment={key: str(value) for key, value in observed_environment.items()},
        docker_inspect_receipt_sha256=sha256_bytes(receipt_payload),
        container_id=container["id"],
        container_name=container["name"],
        container_hostname=container["hostname"],
        image_id=container["image_id"],
    )


def _data_root_path(data_root: Path, configured: str, *, label: str) -> Path:
    root_meta = data_root.lstat()
    if data_root.is_symlink() or not stat.S_ISDIR(root_meta.st_mode):
        raise ValueError("data root must be a real directory")
    path = PurePosixPath(configured)
    if not path.is_absolute() or len(path.parts) < 3 or path.parts[1] != "data":
        raise ValueError(f"{label} is not rooted under canonical /data")
    result = data_root.resolve().joinpath(*path.parts[2:])
    try:
        result.relative_to(data_root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escaped data root") from error
    return result


def _state_paths(
    contract: FormalTrainContract, data_root: Path
) -> dict[str, Path]:
    local = contract.local_state
    records = local.get("ordered_states")
    if not isinstance(records, list):
        raise ValueError("local state inventory is missing")
    result: dict[str, Path] = {}
    for raw in records:
        record = _mapping(raw, "local state record")
        name = record.get("name")
        path = record.get("path")
        if not isinstance(name, str) or not isinstance(path, str) or name in result:
            raise ValueError("local state record is malformed or duplicated")
        result[name] = _data_root_path(data_root, path, label=f"state {name}")
    expected = {
        "global_claim",
        "cache_input_completion",
        "training_artifact_completion",
        "remote_base_receipt",
        "payload_commit_receipt",
        "completion_staging",
        "final_completion",
    }
    if set(result) != expected:
        raise ValueError("local state inventory drifted")
    return result


def _artifact_root(contract: FormalTrainContract, data_root: Path) -> Path:
    configured = contract.local_state.get("artifact_directory")
    if not isinstance(configured, str):
        raise ValueError("artifact directory is missing")
    return _data_root_path(data_root, configured, label="artifact directory")


def _state_bytes(
    contract: FormalTrainContract,
    *,
    status: str,
    source: SourceIdentity,
    payload: Mapping[str, Any],
) -> bytes:
    return pretty_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": status,
            "contract_sha256": contract.sha256,
            "source": _source_record(source),
            **dict(payload),
        }
    )


def _state_record(
    contract: FormalTrainContract,
    path: Path,
    *,
    expected_status: str,
    mode: int,
) -> Mapping[str, Any]:
    payload, _ = _regular_file_bytes(path, label=str(path), mode=mode)
    value = _strict_json(payload, label=str(path))
    if (
        payload != pretty_json_bytes(value)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != expected_status
        or value.get("contract_sha256") != contract.sha256
    ):
        raise ValueError(f"local state identity drifted: {path}")
    return value


def _require_state_lineage(
    record: Mapping[str, Any],
    source: SourceIdentity,
    *,
    claim_sha256: str | None = None,
) -> None:
    if record.get("source") != _source_record(source):
        raise ValueError("local state source identity drifted")
    if claim_sha256 is not None and record.get("claim_sha256") != claim_sha256:
        raise ValueError("local state claim identity drifted")


def _file_record(path: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _cache_file_records(contract: FormalTrainContract) -> tuple[Mapping[str, Any], ...]:
    records = contract.cache_input.get("exact_three_files")
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("formal cache exact-three binding is missing")
    return tuple(_mapping(record, "cache file") for record in records)


def _tag_snapshot(api: Any, *, repo: str, repo_type: str, tag: str) -> tuple[str, str] | None:
    refs = api.list_repo_refs(repo, repo_type=repo_type)
    matches = [item for item in refs.tags if item.name == tag]
    if len(matches) > 1:
        raise ValueError("HF tag is duplicated")
    if not matches:
        return None
    object_identity = getattr(matches[0], "target_commit", None)
    resolved = getattr(
        api.repo_info(repo, repo_type=repo_type, revision=tag), "sha", None
    )
    if (
        not isinstance(object_identity, str)
        or _COMMIT.fullmatch(object_identity) is None
        or not isinstance(resolved, str)
        or _COMMIT.fullmatch(resolved) is None
        or object_identity == resolved
    ):
        raise ValueError("HF tag is not an annotated immutable tag")
    return object_identity, resolved


def _fresh_download(
    *,
    download_fn: Callable[..., str],
    repo: str,
    repo_type: str,
    revision: str,
    expected: Mapping[str, bytes],
    fresh_parent: Path,
    prefix: str,
) -> None:
    parent_meta = fresh_parent.lstat()
    if fresh_parent.is_symlink() or not stat.S_ISDIR(parent_meta.st_mode):
        raise ValueError("fresh download parent must be a real directory")
    with tempfile.TemporaryDirectory(prefix=prefix, dir=fresh_parent) as raw:
        root = Path(raw).resolve()
        if root.is_symlink() or any(root.iterdir()):
            raise ValueError("fresh HF directory did not start empty")
        for relative in sorted(expected):
            returned = Path(
                download_fn(
                    repo_id=repo,
                    repo_type=repo_type,
                    filename=relative,
                    revision=revision,
                    local_dir=root,
                    force_download=True,
                )
            )
            target = root / relative
            if not returned.is_absolute() or returned != target:
                raise ValueError("HF download returned a noncanonical path")
            current = root
            for component in returned.relative_to(root).parts[:-1]:
                current /= component
                if not stat.S_ISDIR(current.lstat().st_mode):
                    raise ValueError("HF download traversed a symlink directory")
            payload, _ = _regular_file_bytes(returned, label=relative)
            if payload != expected[relative]:
                raise ValueError(f"fresh immutable HF bytes drifted: {relative}")


def _fresh_observed_files(
    *,
    download_fn: Callable[..., str],
    repo: str,
    repo_type: str,
    revision: str,
    paths: Sequence[str],
    fresh_parent: Path,
    prefix: str,
) -> Mapping[str, bytes]:
    parent_meta = fresh_parent.lstat()
    if fresh_parent.is_symlink() or not stat.S_ISDIR(parent_meta.st_mode):
        raise ValueError("fresh inventory parent must be a real directory")
    with tempfile.TemporaryDirectory(prefix=prefix, dir=fresh_parent) as raw:
        root = Path(raw).resolve()
        if root.is_symlink() or any(root.iterdir()):
            raise ValueError("fresh HF inventory directory did not start empty")
        result: dict[str, bytes] = {}
        for relative in sorted(paths):
            returned = Path(
                download_fn(
                    repo_id=repo,
                    repo_type=repo_type,
                    filename=relative,
                    revision=revision,
                    local_dir=root,
                    force_download=True,
                )
            )
            if not returned.is_absolute() or returned != root / relative:
                raise ValueError("HF inventory download returned a noncanonical path")
            current = root
            for component in returned.relative_to(root).parts[:-1]:
                current /= component
                if not stat.S_ISDIR(current.lstat().st_mode):
                    raise ValueError("HF inventory download traversed a symlink directory")
            result[relative], _ = _regular_file_bytes(returned, label=relative)
        return result


def download_formal_cache_inputs(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: FormalTrainContract,
    fresh_parent: Path,
) -> Mapping[str, bytes]:
    cache = contract.cache_input
    repo = cache["repo"]
    revision = cache["immutable_revision"]
    info = api.repo_info(repo, repo_type="dataset", revision=revision)
    if getattr(info, "private", None) is not True or getattr(info, "sha", None) != revision:
        raise ValueError("formal cache repo privacy/revision drifted")
    files = set(api.list_repo_files(repo, repo_type="dataset", revision=revision))
    expected_paths = {record["path"] for record in _cache_file_records(contract)}
    if expected_paths - files or files - expected_paths - _ALLOWED_REPO_BASE_FILES:
        raise ValueError("formal cache remote inventory drifted")
    before = _tag_snapshot(
        api,
        repo=repo,
        repo_type="dataset",
        tag=cache["tag"],
    )
    if before != (cache["annotated_tag_object"], revision):
        raise ValueError("formal cache tag binding drifted")
    expected: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="gate-v1-formal-input-", dir=fresh_parent) as raw:
        root = Path(raw).resolve()
        if root.is_symlink() or any(root.iterdir()):
            raise ValueError("formal cache input directory did not start empty")
        for record in _cache_file_records(contract):
            path = record["path"]
            returned = Path(
                download_fn(
                    repo_id=repo,
                    repo_type="dataset",
                    filename=path,
                    revision=revision,
                    local_dir=root,
                    force_download=True,
                )
            )
            if returned != root / path or not returned.is_absolute():
                raise ValueError("formal cache download path drifted")
            payload, _ = _regular_file_bytes(returned, label=path)
            if (
                len(payload) != record["size_bytes"]
                or sha256_bytes(payload) != record["sha256"]
            ):
                raise ValueError("formal cache downloaded bytes drifted")
            expected[path] = payload
    after = _tag_snapshot(
        api,
        repo=repo,
        repo_type="dataset",
        tag=cache["tag"],
    )
    if after != before:
        raise ValueError("formal cache tag moved during download")
    return expected


def _formal_source_ids(contract: FormalTrainContract) -> tuple[str, ...]:
    root = contract.repository_root
    gate = load_strict_json_object(root / "code/configs/causalcache_gate_v1_preregistration.json")
    lineage = _mapping(gate["lineage"], "gate lineage")
    legacy = load_strict_json_object(root / lineage["legacy_selection"]["path"])
    expansion = load_strict_json_object(root / lineage["label_expansion_split"]["path"])
    formal = tuple(derive_rosters(legacy, expansion)["formal_train"])
    if len(formal) != 58:
        raise ValueError("formal source roster count drifted")
    return formal


def _checkpoint_path(family: str, seed: int) -> str:
    path = f"formal58-train/v1/checkpoints/{family}-seed-{seed}.safetensors"
    if path not in PAYLOAD_TARGETS:
        raise ValueError("checkpoint output path drifted")
    return path


def _report_path(family: str) -> str:
    path = f"formal58-train/v1/reports/{family}-full-oof-report.json"
    if path not in PAYLOAD_TARGETS:
        raise ValueError("OOF report output path drifted")
    return path


def _payload_files(result: FormalTrainingResult) -> Mapping[str, bytes]:
    files: dict[str, bytes] = {}
    for checkpoint in result.checkpoints:
        files[_checkpoint_path(checkpoint.family, checkpoint.seed)] = checkpoint.payload
    for family in ("conditional", "independent"):
        files[_report_path(family)] = canonical_json_bytes(result.family_reports[family]) + b"\n"
    if tuple(sorted(files)) != tuple(sorted(PAYLOAD_TARGETS)):
        raise ValueError("formal training payload output inventory drifted")
    return files


def _publish_local_files(
    root: Path,
    files: Mapping[str, bytes],
    *,
    mode: int,
) -> int:
    created = 0
    for relative in sorted(files):
        created += int(
            _exclusive_or_identical(root / relative, files[relative], mode=mode)
        )
    return created


def _read_local_files(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    mode: int,
) -> Mapping[str, bytes]:
    files: dict[str, bytes] = {}
    for record in records:
        path = _safe_relative(record.get("path"), "artifact path")
        payload, _ = _regular_file_bytes(root / path, label=path, mode=mode)
        if (
            len(payload) != record.get("size_bytes")
            or sha256_bytes(payload) != record.get("sha256")
        ):
            raise ValueError(f"local artifact identity drifted: {path}")
        files[path] = payload
    return files


def _response_commit(value: Any) -> str:
    for name in ("oid", "commit_id"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, str) and _COMMIT.fullmatch(candidate):
            return candidate
    raise ValueError("HF commit response omitted immutable commit")


def _operation(factory: Callable[..., Any], path: str, payload: bytes) -> Any:
    return factory(path_in_repo=path, path_or_fileobj=io.BytesIO(payload))


def _repo_snapshot(api: Any, contract: FormalTrainContract) -> tuple[str, tuple[str, ...]]:
    destination = contract.destination
    info = api.repo_info(destination["repo"], repo_type="model", revision="main")
    if getattr(info, "private", None) is not True:
        raise ValueError("formal model repo must remain private")
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or _COMMIT.fullmatch(revision) is None:
        raise ValueError("formal model main revision is invalid")
    files = tuple(
        sorted(
            api.list_repo_files(
                destination["repo"], repo_type="model", revision=revision
            )
        )
    )
    if len(files) != len(set(files)):
        raise ValueError("formal model remote file inventory is duplicated")
    return revision, files


def _commit_history(
    api: Any,
    contract: FormalTrainContract,
    *,
    revision: str,
) -> tuple[tuple[str, str], ...]:
    commits = api.list_repo_commits(
        contract.destination["repo"],
        repo_type="model",
        revision=revision,
        formatted=False,
    )
    records: list[tuple[str, str]] = []
    for commit in commits:
        commit_id = getattr(commit, "commit_id", None)
        title = getattr(commit, "title", None)
        if (
            not isinstance(commit_id, str)
            or _COMMIT.fullmatch(commit_id) is None
            or not isinstance(title, str)
        ):
            raise ValueError("formal model commit history is malformed")
        records.append((commit_id, title))
    if not records or len({record[0] for record in records}) != len(records):
        raise ValueError("formal model commit history is empty or duplicated")
    return tuple(records)


def _validate_initial_model_base(
    api: Any,
    contract: FormalTrainContract,
    *,
    base_commit: str,
) -> None:
    history = _commit_history(api, contract, revision=base_commit)
    if len(history) != 1 or history[0][0] != base_commit:
        raise ValueError("formal model repo is not a one-commit initial base")


def _recover_payload_base(
    api: Any,
    contract: FormalTrainContract,
    *,
    payload_commit: str,
) -> str:
    history = _commit_history(api, contract, revision=payload_commit)
    if (
        len(history) != 2
        or history[0]
        != (payload_commit, contract.output["payload_commit"]["commit_title"])
    ):
        raise ValueError("formal payload commit is not the unique child of initial base")
    return history[1][0]


def _recover_manifest_chain(
    api: Any,
    contract: FormalTrainContract,
    *,
    manifest_commit: str,
) -> tuple[str, str]:
    history = _commit_history(api, contract, revision=manifest_commit)
    if (
        len(history) != 3
        or history[0]
        != (manifest_commit, contract.output["manifest_commit"]["commit_title"])
        or history[1][1] != contract.output["payload_commit"]["commit_title"]
    ):
        raise ValueError("formal manifest commit does not replay the exact train history")
    return history[2][0], history[1][0]


def _validate_two_commit_chain(
    api: Any,
    contract: FormalTrainContract,
    *,
    remote_base_commit: str,
    payload_commit: str,
    manifest_commit: str,
) -> None:
    if len({remote_base_commit, payload_commit, manifest_commit}) != 3:
        raise ValueError("formal model base/payload/manifest commits must be distinct")
    history = _commit_history(api, contract, revision=manifest_commit)
    expected = (
        (manifest_commit, contract.output["manifest_commit"]["commit_title"]),
        (payload_commit, contract.output["payload_commit"]["commit_title"]),
    )
    if len(history) != 3 or history[:2] != expected or history[2][0] != remote_base_commit:
        raise ValueError("formal model does not have the frozen exact two-commit chain")
    main_revision, files = _repo_snapshot(api, contract)
    targets = set(PAYLOAD_TARGETS) | set(MANIFEST_TARGETS)
    if main_revision != manifest_commit or set(files) - _ALLOWED_REPO_BASE_FILES != targets:
        raise ValueError("formal model final main tree or target inventory drifted")


def _ensure_model_repo(
    api: Any,
    contract: FormalTrainContract,
) -> tuple[str, tuple[str, ...], int]:
    destination = contract.destination
    mutations = 0
    try:
        revision, files = _repo_snapshot(api, contract)
    except Exception as error:
        if error.__class__.__name__ != "RepositoryNotFoundError":
            raise
        mutations += 1
        try:
            api.create_repo(
                destination["repo"],
                repo_type="model",
                private=True,
                exist_ok=False,
            )
        except Exception:
            revision, files = _repo_snapshot(api, contract)
        else:
            revision, files = _repo_snapshot(api, contract)
    if set(files) - _ALLOWED_REPO_BASE_FILES - set(PAYLOAD_TARGETS) - set(MANIFEST_TARGETS):
        raise ValueError("formal model repo contains unexpected files")
    return revision, files, mutations


def _commit_files(
    *,
    api: Any,
    operation_factory: Callable[..., Any],
    contract: FormalTrainContract,
    files: Mapping[str, bytes],
    parent: str,
    title: str,
) -> tuple[str, int]:
    destination = contract.destination
    operations = [
        _operation(operation_factory, path, files[path]) for path in sorted(files)
    ]
    mutations = 1
    try:
        response = api.create_commit(
            destination["repo"],
            repo_type="model",
            revision="main",
            parent_commit=parent,
            commit_message=title,
            operations=operations,
        )
        commit = _response_commit(response)
    except Exception:
        commit, remote_paths = _repo_snapshot(api, contract)
        if not set(files).issubset(remote_paths):
            raise
    return commit, mutations


def _download_and_compare_revision(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: FormalTrainContract,
    revision: str,
    expected: Mapping[str, bytes],
    fresh_parent: Path,
) -> Mapping[str, bytes]:
    destination = contract.destination
    info = api.repo_info(destination["repo"], repo_type="model", revision=revision)
    if getattr(info, "private", None) is not True or getattr(info, "sha", None) != revision:
        raise ValueError("formal model immutable revision drifted")
    files = set(
        api.list_repo_files(
            destination["repo"], repo_type="model", revision=revision
        )
    )
    target_files = files - _ALLOWED_REPO_BASE_FILES
    if target_files != set(expected):
        raise ValueError("formal model immutable revision target inventory drifted")
    _fresh_download(
        download_fn=download_fn,
        repo=destination["repo"],
        repo_type="model",
        revision=revision,
        expected=expected,
        fresh_parent=fresh_parent,
        prefix="gate-v1-formal-model-",
    )
    return _fresh_observed_files(
        download_fn=download_fn,
        repo=destination["repo"],
        repo_type="model",
        revision=revision,
        paths=tuple(sorted(files & _ALLOWED_REPO_BASE_FILES)),
        fresh_parent=fresh_parent,
        prefix="gate-v1-formal-base-",
    )


def _create_or_validate_tag(
    *,
    api: Any,
    contract: FormalTrainContract,
    manifest_commit: str,
) -> tuple[str, int]:
    destination = contract.destination
    existing = _tag_snapshot(
        api,
        repo=destination["repo"],
        repo_type="model",
        tag=destination["tag"],
    )
    if existing is not None:
        if existing[1] != manifest_commit:
            raise ValueError("formal model tag points to another commit")
        return existing[0], 0
    mutations = 1
    try:
        api.create_tag(
            destination["repo"],
            repo_type="model",
            tag=destination["tag"],
            tag_message=destination["tag_message"],
            revision=manifest_commit,
            exist_ok=False,
        )
    except Exception:
        observed = _tag_snapshot(
            api,
            repo=destination["repo"],
            repo_type="model",
            tag=destination["tag"],
        )
        if observed is None or observed[1] != manifest_commit:
            raise
        return observed[0], mutations
    observed = _tag_snapshot(
        api,
        repo=destination["repo"],
        repo_type="model",
        tag=destination["tag"],
    )
    if observed is None or observed[1] != manifest_commit:
        raise ValueError("formal model annotated tag did not bind manifest commit")
    return observed[0], mutations


def _trial_from_payload(value: Any, *, label: str) -> OOFTrial:
    record = _mapping(value, label)
    _exact_keys(
        record,
        {
            "family",
            "learning_rate",
            "seed",
            "selected_epoch",
            "best_raw_utility_ratio",
            "epochs_run",
            "metric_by_epoch",
        },
        label,
    )
    metrics = record.get("metric_by_epoch")
    if not isinstance(metrics, list):
        raise ValueError(f"{label} metric history is not an array")
    return OOFTrial(
        family=record["family"],
        learning_rate=record["learning_rate"],
        seed=record["seed"],
        selected_epoch=record["selected_epoch"],
        best_raw_utility_ratio=record["best_raw_utility_ratio"],
        epochs_run=record["epochs_run"],
        metric_by_epoch=tuple(metrics),
    )


def _selection_from_payload(value: Any, *, family: str) -> FamilySelection:
    selection = _mapping(value, f"{family} OOF selection")
    _exact_keys(
        selection,
        {
            "family",
            "learning_rate",
            "five_seed_mean_oof_ratio",
            "selected_trials",
            "grid_trials",
        },
        f"{family} OOF selection",
    )
    selected = selection.get("selected_trials")
    grid = selection.get("grid_trials")
    if not isinstance(selected, list) or not isinstance(grid, list):
        raise ValueError("OOF trial inventories are not arrays")
    result = FamilySelection(
        family=selection["family"],
        learning_rate=selection["learning_rate"],
        five_seed_mean_oof_ratio=selection["five_seed_mean_oof_ratio"],
        trials=tuple(
            _trial_from_payload(item, label=f"{family} selected trial {index}")
            for index, item in enumerate(selected)
        ),
        grid_trials=tuple(
            _trial_from_payload(item, label=f"{family} grid trial {index}")
            for index, item in enumerate(grid)
        ),
    )
    if result.family != family or family_selection_payload(result) != dict(selection):
        raise ValueError("OOF selection does not replay the frozen trainer rules")
    return result


def _expected_input_binding(contract: FormalTrainContract) -> Mapping[str, Any]:
    feature, label, bundle = _cache_file_records(contract)
    cache = contract.cache_input
    return {
        "gate_preregistration_sha256": contract.data["lineage"][
            "gate_preregistration"
        ]["sha256"],
        "formal_source_ids_sha256": contract.geometry["source_ids_sha256"],
        "cache": {
            "repository": cache["repo"],
            "revision": cache["immutable_revision"],
            "tag": cache["tag"],
            "feature": {"path": feature["path"], "sha256": feature["sha256"]},
            "label": {"path": label["path"], "sha256": label["sha256"]},
            "bundle_manifest": {
                "path": bundle["path"],
                "sha256": bundle["sha256"],
            },
            "join_audit_sha256": cache["join_only_audit_sha256"],
        },
        "trajectory_count": contract.geometry["trajectory_count"],
        "state_count": contract.geometry["state_count"],
        "fold_sizes": list(contract.geometry["oof_fold_sizes"]),
    }


def _validate_family_report(
    family: str,
    report: Mapping[str, Any],
    contract: FormalTrainContract,
) -> FamilySelection:
    _exact_keys(
        report,
        {
            "schema_version",
            "protocol_id",
            "status",
            "family",
            "input_binding",
            "oof",
            "final_fit",
            "operation_counts",
            "access_counts",
        },
        f"{family} OOF report",
    )
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("status") != "COMPLETED_GATE_V1_FORMAL_TRAIN_FAMILY_V1"
        or report.get("family") != family
        or report.get("input_binding") != _expected_input_binding(contract)
    ):
        raise ValueError("OOF report identity or input binding drifted")
    oof = _mapping(report.get("oof"), "OOF report selection")
    _exact_keys(oof, {"complete_grid", "selection_sha256"}, "OOF report selection")
    selection = _mapping(oof.get("complete_grid"), "OOF complete grid")
    parsed_selection = _selection_from_payload(selection, family=family)
    selection_sha = oof.get("selection_sha256")
    if (
        not isinstance(selection_sha, str)
        or _SHA256.fullmatch(selection_sha) is None
        or sha256_bytes(canonical_json_bytes(selection)) != selection_sha
    ):
        raise ValueError("OOF selection digest drifted")
    final_fit = _mapping(report.get("final_fit"), "OOF final fit")
    _exact_keys(
        final_fit,
        {"learning_rate", "seeds", "selected_epochs", "checkpoints"},
        "OOF final fit",
    )
    epochs_raw = final_fit.get("selected_epochs")
    checkpoints = final_fit.get("checkpoints")
    if (
        final_fit.get("learning_rate") != parsed_selection.learning_rate
        or final_fit.get("seeds") != [0, 1, 2, 3, 4]
        or not isinstance(epochs_raw, list)
        or epochs_raw
        != [trial.selected_epoch for trial in parsed_selection.trials]
        or not isinstance(checkpoints, list)
        or len(checkpoints) != 5
    ):
        raise ValueError("OOF report selected metadata is malformed")
    expected_family_counts = _family_operation_counts(parsed_selection)
    if report.get("operation_counts") != expected_family_counts:
        raise ValueError("OOF report dynamic operation counts do not replay")
    expected_access = {
        "formal58_trajectory_semantic_decode_count": 58,
        "formal58_feature_state_semantic_decode_count": 174,
        "formal58_label_state_semantic_decode_count": 174,
        "fresh16_semantic_decode_count": 0,
        "legacy_dev5_semantic_decode_count": 0,
        "confirm20_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
    }
    if report.get("access_counts") != expected_access:
        raise ValueError("OOF report access counts drifted")
    return parsed_selection


def _replay_dynamic_operation_counts(
    contract: FormalTrainContract,
    payload_files: Mapping[str, bytes],
) -> Mapping[str, int | str]:
    selections = []
    for family in ("conditional", "independent"):
        report = _strict_json(
            payload_files[_report_path(family)],
            label=f"{family} OOF report",
        )
        selections.append(_validate_family_report(family, report, contract))
    return _combined_operation_counts(tuple(selections))


def _report_selection_sha256(
    payload_files: Mapping[str, bytes],
) -> Mapping[str, str]:
    return {
        family: _strict_json(
            payload_files[_report_path(family)],
            label=f"{family} OOF report",
        )["oof"]["selection_sha256"]
        for family in ("conditional", "independent")
    }


def _ensemble_manifest(
    *,
    family: str,
    report: Mapping[str, Any],
    payload_commit: str,
    payload_files: Mapping[str, bytes],
    contract: FormalTrainContract,
) -> Mapping[str, Any]:
    selection = _validate_family_report(family, report, contract)
    learning_rate = selection.learning_rate
    epochs = tuple(trial.selected_epoch for trial in selection.trials)
    selection_sha = report["oof"]["selection_sha256"]
    checkpoints = report["final_fit"]["checkpoints"]
    cache = contract.cache_input
    feature_record, label_record, _ = _cache_file_records(contract)
    training = FrozenTrainingProvenance(
        family=family,
        gate_config_sha256=contract.data["lineage"]["gate_preregistration"]["sha256"],
        formal_train_source_ids_sha256=contract.geometry["source_ids_sha256"],
        learning_rate=learning_rate,
        selected_epochs=epochs,
        oof_selection_sha256=selection_sha,
        feature_artifact=ArtifactBinding(
            repository=cache["repo"],
            revision=cache["immutable_revision"],
            path=feature_record["path"],
            sha256=feature_record["sha256"],
        ),
        label_artifact=ArtifactBinding(
            repository=cache["repo"],
            revision=cache["immutable_revision"],
            path=label_record["path"],
            sha256=label_record["sha256"],
        ),
        training_report_artifact=ArtifactBinding(
            repository=contract.destination["repo"],
            revision=payload_commit,
            path=_report_path(family),
            sha256=sha256_bytes(payload_files[_report_path(family)]),
        ),
    )
    provenance_checkpoints: list[SeedCheckpointProvenance] = []
    for seed, (epoch, raw) in enumerate(zip(epochs, checkpoints, strict=True)):
        record = _mapping(raw, "checkpoint report")
        _exact_keys(
            record,
            {
                "family",
                "seed",
                "selected_epoch",
                "format",
                "model_state_sha256",
                "checkpoint_sha256",
                "size_bytes",
            },
            "checkpoint report",
        )
        path = _checkpoint_path(family, seed)
        if (
            record.get("family") != family
            or record.get("seed") != seed
            or record.get("selected_epoch") != epoch
            or record.get("format") != "safetensors"
            or not isinstance(record.get("model_state_sha256"), str)
            or _SHA256.fullmatch(record["model_state_sha256"]) is None
            or not isinstance(record.get("checkpoint_sha256"), str)
            or _SHA256.fullmatch(record["checkpoint_sha256"]) is None
            or record.get("checkpoint_sha256") != sha256_bytes(payload_files[path])
            or record.get("size_bytes") != len(payload_files[path])
        ):
            raise ValueError("checkpoint report differs from payload bytes")
        provenance_checkpoints.append(
            SeedCheckpointProvenance(
                seed=seed,
                selected_epoch=epoch,
                model_state_sha256=record["model_state_sha256"],
                checkpoint_artifact=ArtifactBinding(
                    repository=contract.destination["repo"],
                    revision=payload_commit,
                    path=path,
                    sha256=record["checkpoint_sha256"],
                ),
            )
        )
    provenance = FrozenEnsembleProvenance(
        training=training,
        checkpoints=tuple(provenance_checkpoints),
    )
    provenance.validate()
    return provenance.to_payload()


def _manifest_files(
    *,
    contract: FormalTrainContract,
    source: SourceIdentity,
    runtime: RuntimeIdentity,
    payload_commit: str,
    payload_files: Mapping[str, bytes],
    training_completion: Mapping[str, Any],
) -> Mapping[str, bytes]:
    reports = {
        family: _strict_json(
            payload_files[_report_path(family)], label=f"{family} OOF report"
        )
        for family in ("conditional", "independent")
    }
    ensembles = {
        family: _ensemble_manifest(
            family=family,
            report=reports[family],
            payload_commit=payload_commit,
            payload_files=payload_files,
            contract=contract,
        )
        for family in ("conditional", "independent")
    }
    ensemble_bytes = {
        f"formal58-train/v1/manifests/{family}-ensemble-manifest.json": (
            canonical_json_bytes(ensembles[family]) + b"\n"
        )
        for family in ("conditional", "independent")
    }
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "IMMUTABLE_GATE_V1_FORMAL58_TRAIN_RUN_MANIFEST_V1",
        "contract_sha256": contract.sha256,
        "source": _source_record(source),
        "runtime": runtime.to_payload(),
        "cache_input": dict(contract.cache_input),
        "cache_join_audit_sha256": FORMAL_JOIN_AUDIT_SHA256,
        "payload_commit": payload_commit,
        "payload_files": [_file_record(path, payload_files[path]) for path in sorted(payload_files)],
        "selection_sha256": dict(training_completion["selection_sha256"]),
        "operation_counts": dict(training_completion["operation_counts"]),
        "checkpoint_count": 10,
        "training_executed": True,
        "fresh16_semantic_decode_count": 0,
        "legacy_dev5_semantic_decode_count": 0,
        "confirm20_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
    }
    run_path = "formal58-train/v1/manifests/run-manifest.json"
    run_bytes = canonical_json_bytes(run_manifest) + b"\n"
    partial = {**ensemble_bytes, run_path: run_bytes}
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "IMMUTABLE_GATE_V1_FORMAL58_TRAIN_BUNDLE_MANIFEST_V1",
        "contract_sha256": contract.sha256,
        "repo": contract.destination["repo"],
        "tag": contract.destination["tag"],
        "payload_commit": payload_commit,
        "payload_files": [_file_record(path, payload_files[path]) for path in sorted(payload_files)],
        "manifest_files_excluding_self": [
            _file_record(path, partial[path]) for path in sorted(partial)
        ],
        "final_tree_target_count": 16,
        "gate_trained_after_final_completion_only": True,
        "fresh16_access_authorized": False,
        "legacy_dev5_access_authorized": False,
        "confirm20_access_authorized": False,
        "matched_nll_authorized": False,
        "closed_loop_authorized": False,
    }
    bundle_path = "formal58-train/v1/manifests/bundle-manifest.json"
    files = {**partial, bundle_path: canonical_json_bytes(bundle) + b"\n"}
    if tuple(sorted(files)) != tuple(sorted(MANIFEST_TARGETS)):
        raise ValueError("formal manifest output inventory drifted")
    return files


def _validate_checkpoints_from_manifests(
    payload_files: Mapping[str, bytes],
    manifest_files: Mapping[str, bytes],
) -> Mapping[str, Any]:
    digests: dict[str, Any] = {}
    all_state_digests: list[str] = []
    all_artifact_digests: list[str] = []
    for family in ("conditional", "independent"):
        path = f"formal58-train/v1/manifests/{family}-ensemble-manifest.json"
        manifest = _strict_json(manifest_files[path], label=path)
        provenance = frozen_ensemble_provenance_from_manifest(manifest)
        if provenance.training.family != family:
            raise ValueError("ensemble manifest family drifted")
        family_states = []
        family_artifacts = []
        for checkpoint in provenance.checkpoints:
            artifact = checkpoint.checkpoint_artifact
            payload = payload_files[artifact.path]
            load_safetensors_checkpoint(
                payload,
                family=family,
                seed=checkpoint.seed,
                expected_model_state_sha256=checkpoint.model_state_sha256,
                expected_checkpoint_sha256=artifact.sha256,
            )
            family_states.append(checkpoint.model_state_sha256)
            family_artifacts.append(artifact.sha256)
        all_state_digests.extend(family_states)
        all_artifact_digests.extend(family_artifacts)
        digests[family] = {
            "ensemble_manifest_sha256": sha256_bytes(manifest_files[path]),
            "ensemble_provenance_sha256": provenance.sha256,
            "selection_sha256": provenance.training.oof_selection_sha256,
            "model_state_sha256": family_states,
            "checkpoint_artifact_sha256": family_artifacts,
        }
    if len(set(all_state_digests)) != 10 or len(set(all_artifact_digests)) != 10:
        raise ValueError("formal ten-checkpoint digests are not globally distinct")
    return digests


def _repo_exists_snapshot(
    api: Any, contract: FormalTrainContract
) -> tuple[str, tuple[str, ...]] | None:
    try:
        return _repo_snapshot(api, contract)
    except Exception as error:
        if error.__class__.__name__ == "RepositoryNotFoundError":
            return None
        raise


def _expected_operation_counts(
    contract: FormalTrainContract,
    payload_files: Mapping[str, bytes],
) -> Mapping[str, int | str]:
    return {
        **dict(contract.execution_operations),
        **dict(_replay_dynamic_operation_counts(contract, payload_files)),
    }


def _validate_operation_counts(
    contract: FormalTrainContract,
    counts: Mapping[str, Any],
    payload_files: Mapping[str, bytes],
) -> None:
    expected = _expected_operation_counts(contract, payload_files)
    if dict(counts) != expected:
        raise ValueError("formal training operation counts do not replay exactly")


def _training_completion_payload(
    contract: FormalTrainContract,
    result: FormalTrainingResult,
    payload_files: Mapping[str, bytes],
) -> Mapping[str, Any]:
    _validate_checkpoint_result(result)
    selections = {
        "conditional": result.conditional_ensemble.selection_sha256,
        "independent": result.independent_ensemble.selection_sha256,
    }
    if selections != _report_selection_sha256(payload_files):
        raise ValueError("formal ensemble selections differ from OOF reports")
    replayed_dynamic = _replay_dynamic_operation_counts(contract, payload_files)
    if dict(result.operation_counts) != replayed_dynamic:
        raise ValueError("formal core operation counts differ from OOF report replay")
    operation_counts = {
        **dict(contract.execution_operations),
        **dict(replayed_dynamic),
    }
    return {
        "training_executed": True,
        "payload_files": [
            _file_record(path, payload_files[path]) for path in sorted(payload_files)
        ],
        "selection_sha256": selections,
        "operation_counts": operation_counts,
        "core_run_manifest_sha256": result.run_manifest_sha256,
        "checkpoint_count": 10,
        "full_oof_report_count": 2,
        "fresh16_semantic_decode_count": 0,
        "legacy_dev5_semantic_decode_count": 0,
        "confirm20_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
    }


def _validate_checkpoint_result(result: FormalTrainingResult) -> None:
    if len(result.checkpoints) != 10:
        raise ValueError("formal training did not produce ten checkpoints")
    identities = tuple(
        (item.family, item.seed, item.selected_epoch) for item in result.checkpoints
    )
    expected = tuple(
        (
            family,
            seed,
            (
                result.conditional_ensemble.selected_epochs[seed]
                if family == "conditional"
                else result.independent_ensemble.selected_epochs[seed]
            ),
        )
        for family in ("conditional", "independent")
        for seed in range(5)
    )
    if identities != expected:
        raise ValueError("formal checkpoint family/seed/epoch order drifted")
    if (
        len({item.model_state_sha256 for item in result.checkpoints}) != 10
        or len({item.checkpoint_sha256 for item in result.checkpoints}) != 10
    ):
        raise ValueError("formal checkpoint digests are not globally distinct")


def _completion_replay(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: FormalTrainContract,
    source: SourceIdentity,
    runtime: RuntimeIdentity,
    paths: Mapping[str, Path],
    artifact_root: Path,
    fresh_parent: Path,
    state_mode: int,
    artifact_mode: int,
) -> Mapping[str, Any]:
    claim = _state_record(
        contract,
        paths["global_claim"],
        expected_status=GLOBAL_CLAIM_STATUS,
        mode=state_mode,
    )
    _require_state_lineage(claim, source)
    claim_bytes, _ = _regular_file_bytes(
        paths["global_claim"], label="global claim", mode=state_mode
    )
    claim_sha256 = sha256_bytes(claim_bytes)
    cache_completion = _state_record(
        contract,
        paths["cache_input_completion"],
        expected_status=CACHE_INPUT_COMPLETION_STATUS,
        mode=state_mode,
    )
    training_completion = _state_record(
        contract,
        paths["training_artifact_completion"],
        expected_status=TRAINING_ARTIFACT_COMPLETION_STATUS,
        mode=state_mode,
    )
    remote_base = _state_record(
        contract,
        paths["remote_base_receipt"],
        expected_status=REMOTE_BASE_STATUS,
        mode=state_mode,
    )
    payload_receipt = _state_record(
        contract,
        paths["payload_commit_receipt"],
        expected_status=PAYLOAD_COMMIT_STATUS,
        mode=state_mode,
    )
    for record in (
        cache_completion,
        training_completion,
        remote_base,
        payload_receipt,
    ):
        _require_state_lineage(record, source, claim_sha256=claim_sha256)
    if remote_base.get("repo_initially_absent") is not True:
        raise ValueError("formal remote-base receipt drifted")
    final = _state_record(
        contract,
        paths["final_completion"],
        expected_status=FINAL_COMPLETION_STATUS,
        mode=state_mode,
    )
    staged = _state_record(
        contract,
        paths["completion_staging"],
        expected_status=FINAL_COMPLETION_STATUS,
        mode=state_mode,
    )
    _require_state_lineage(final, source, claim_sha256=claim_sha256)
    _require_state_lineage(staged, source, claim_sha256=claim_sha256)
    _, final_meta = _regular_file_bytes(
        paths["final_completion"], label="final completion", mode=state_mode
    )
    _, staged_meta = _regular_file_bytes(
        paths["completion_staging"], label="staged completion", mode=state_mode
    )
    if final != staged or (
        final_meta.st_dev,
        final_meta.st_ino,
    ) != (
        staged_meta.st_dev,
        staged_meta.st_ino,
    ):
        raise ValueError("formal completion is not the retained-stage hard link")
    payload_records = final.get("payload_files")
    manifest_records = final.get("manifest_files")
    if not isinstance(payload_records, list) or not isinstance(manifest_records, list):
        raise ValueError("formal completion artifact inventory is missing")
    payload_files = _read_local_files(
        artifact_root, payload_records, mode=artifact_mode
    )
    manifest_files = _read_local_files(
        artifact_root, manifest_records, mode=artifact_mode
    )
    _validate_operation_counts(
        contract,
        _mapping(training_completion.get("operation_counts"), "operation counts"),
        payload_files,
    )
    if training_completion.get("selection_sha256") != _report_selection_sha256(
        payload_files
    ):
        raise ValueError("formal completion selection digests drifted")
    payload_commit = final.get("payload_commit")
    manifest_commit = final.get("manifest_commit")
    tag_object = final.get("annotated_tag_object")
    remote_base_commit = final.get("remote_base_commit")
    if any(
        not isinstance(value, str) or _COMMIT.fullmatch(value) is None
        for value in (
            remote_base_commit,
            payload_commit,
            manifest_commit,
            tag_object,
        )
    ) or len({remote_base_commit, payload_commit, manifest_commit}) != 3:
        raise ValueError("formal completion commit/tag identities are malformed")
    if (
        payload_receipt.get("payload_commit") != payload_commit
        or payload_receipt.get("remote_base_commit") != remote_base_commit
        or payload_receipt.get("payload_files") != payload_records
        or training_completion.get("payload_files") != payload_records
        or training_completion.get("selection_sha256")
        != final.get("selection_sha256")
        or training_completion.get("operation_counts")
        != final.get("operation_counts")
    ):
        raise ValueError("formal completion local receipt chain drifted")
    expected_manifest_files = _manifest_files(
        contract=contract,
        source=source,
        runtime=runtime,
        payload_commit=payload_commit,
        payload_files=payload_files,
        training_completion=training_completion,
    )
    if manifest_files != expected_manifest_files:
        raise ValueError("formal completion manifest bytes do not replay from provenance")
    _validate_two_commit_chain(
        api,
        contract,
        remote_base_commit=remote_base_commit,
        payload_commit=payload_commit,
        manifest_commit=manifest_commit,
    )
    observed_tag = _tag_snapshot(
        api,
        repo=contract.destination["repo"],
        repo_type="model",
        tag=contract.destination["tag"],
    )
    if observed_tag != (tag_object, manifest_commit):
        raise ValueError("formal completion tag identity drifted")
    initial_base_files = _download_and_compare_revision(
        api=api,
        download_fn=download_fn,
        contract=contract,
        revision=remote_base_commit,
        expected={},
        fresh_parent=fresh_parent,
    )
    payload_base_files = _download_and_compare_revision(
        api=api,
        download_fn=download_fn,
        contract=contract,
        revision=payload_commit,
        expected=payload_files,
        fresh_parent=fresh_parent,
    )
    manifest_base_files = _download_and_compare_revision(
        api=api,
        download_fn=download_fn,
        contract=contract,
        revision=manifest_commit,
        expected={**payload_files, **manifest_files},
        fresh_parent=fresh_parent,
    )
    remote_base_records = [
        _file_record(path, initial_base_files[path])
        for path in sorted(initial_base_files)
    ]
    if (
        initial_base_files != payload_base_files
        or initial_base_files != manifest_base_files
        or payload_receipt.get("remote_base_files") != remote_base_records
        or final.get("remote_base_files") != remote_base_records
    ):
        raise ValueError("formal model base blobs changed across immutable revisions")
    checkpoint_replay = _validate_checkpoints_from_manifests(
        payload_files, manifest_files
    )
    if (
        _tag_snapshot(
            api,
            repo=contract.destination["repo"],
            repo_type="model",
            tag=contract.destination["tag"],
        )
        != observed_tag
    ):
        raise ValueError("formal model tag moved during immutable replay")
    return {
        "payload_commit": payload_commit,
        "remote_base_commit": remote_base_commit,
        "manifest_commit": manifest_commit,
        "annotated_tag_object": tag_object,
        "payload_files": payload_records,
        "manifest_files": manifest_records,
        "checkpoint_replay": checkpoint_replay,
        "remote_mutation_call_count": 0,
        "training_executed": False,
        "gate_trained": True,
        "fresh16_access_authorized": False,
        "legacy_dev5_access_authorized": False,
        "confirm20_access_authorized": False,
        "matched_nll_authorized": False,
        "closed_loop_authorized": False,
    }


def execute_formal_train(
    *,
    mode: str,
    contract: FormalTrainContract,
    api: Any,
    download_fn: Callable[..., str],
    operation_factory: Callable[..., Any],
    expected_execution_b_git_commit: str,
    data_root: Path,
    fresh_download_parent: Path,
) -> Mapping[str, Any]:
    if mode not in {"run", "validate"}:
        raise ValueError("formal train mode must be run or validate")
    source = validate_execution_b_source(
        contract,
        expected_execution_b_git_commit=expected_execution_b_git_commit,
    )
    runtime = validate_execution_runtime(
        contract,
        source=source,
        data_root=data_root,
    )
    paths = _state_paths(contract, data_root)
    artifact_root = _artifact_root(contract, data_root)
    state_mode = int(contract.local_state["state_file_mode"])
    artifact_mode = int(contract.local_state["artifact_file_mode"])

    if paths["final_completion"].exists() or paths["final_completion"].is_symlink():
        replay = _completion_replay(
            api=api,
            download_fn=download_fn,
            contract=contract,
            source=source,
            runtime=runtime,
            paths=paths,
            artifact_root=artifact_root,
            fresh_parent=fresh_download_parent,
            state_mode=state_mode,
            artifact_mode=artifact_mode,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": VALIDATE_STATUS,
            "contract_sha256": contract.sha256,
            "source": _source_record(source),
            "runtime": runtime.to_payload(),
            **replay,
        }
    if mode == "validate":
        raise ValueError("read-only validate requires final completion")

    claim = _state_bytes(
        contract,
        status=GLOBAL_CLAIM_STATUS,
        source=source,
        payload={
            "runtime": runtime.to_payload(),
            "cache_input": dict(contract.cache_input),
            "training_executed": False,
            "fresh16_access_authorized": False,
            "legacy_dev5_access_authorized": False,
            "confirm20_access_authorized": False,
            "matched_nll_authorized": False,
            "closed_loop_authorized": False,
        },
    )
    claim_created = _exclusive_or_identical(
        paths["global_claim"], claim, mode=state_mode
    )
    claim_sha = sha256_bytes(claim)

    cache_bytes = download_formal_cache_inputs(
        api=api,
        download_fn=download_fn,
        contract=contract,
        fresh_parent=fresh_download_parent,
    )
    cache_records = {
        record["kind"]: record for record in _cache_file_records(contract)
    }
    cache_completion_created = False
    training_completion_created = False
    training_completion_path = paths["training_artifact_completion"]

    if training_completion_path.exists() or training_completion_path.is_symlink():
        cache_completion = _state_record(
            contract,
            paths["cache_input_completion"],
            expected_status=CACHE_INPUT_COMPLETION_STATUS,
            mode=state_mode,
        )
        _require_state_lineage(
            cache_completion,
            source,
            claim_sha256=claim_sha,
        )
        training_completion = _state_record(
            contract,
            training_completion_path,
            expected_status=TRAINING_ARTIFACT_COMPLETION_STATUS,
            mode=state_mode,
        )
        _require_state_lineage(
            training_completion,
            source,
            claim_sha256=claim_sha,
        )
        payload_records = training_completion.get("payload_files")
        if not isinstance(payload_records, list):
            raise ValueError("training completion payload inventory is missing")
        payload_files = _read_local_files(
            artifact_root, payload_records, mode=artifact_mode
        )
        _validate_operation_counts(
            contract,
            _mapping(training_completion["operation_counts"], "operation counts"),
            payload_files,
        )
        if training_completion.get("selection_sha256") != _report_selection_sha256(
            payload_files
        ):
            raise ValueError("training completion selection digests drifted")
    else:
        formal_inputs = load_repaired_formal_training_inputs(
            cache_bytes[cache_records["feature_cache"]["path"]],
            cache_bytes[cache_records["label_cache"]["path"]],
            expected_source_ids=_formal_source_ids(contract),
            join_audit_sha256=contract.cache_input["join_only_audit_sha256"],
        )
        cache_completion_payload = _state_bytes(
            contract,
            status=CACHE_INPUT_COMPLETION_STATUS,
            source=source,
            payload={
                "claim_sha256": claim_sha,
                "cache_files": [dict(record) for record in _cache_file_records(contract)],
                "join_only_audit_sha256": FORMAL_JOIN_AUDIT_SHA256,
                "trajectory_count": len(formal_inputs.source_ids),
                "state_count": len(formal_inputs.states),
                "fold_sizes": [len(fold) for fold in formal_inputs.folds],
                "formal58_semantic_decode_allowed": True,
                "fresh16_semantic_decode_count": 0,
                "legacy_dev5_semantic_decode_count": 0,
                "confirm20_access_count": 0,
                "matched_nll_evaluation_count": 0,
                "closed_loop_episode_count": 0,
                "optimizer_step_count": 0,
            },
        )
        cache_completion_created = _exclusive_or_identical(
            paths["cache_input_completion"],
            cache_completion_payload,
            mode=state_mode,
        )
        result = run_formal_training(formal_inputs)
        payload_files = _payload_files(result)
        _publish_local_files(
            artifact_root, payload_files, mode=artifact_mode
        )
        training_payload = _training_completion_payload(
            contract,
            result,
            payload_files,
        )
        training_completion_bytes = _state_bytes(
            contract,
            status=TRAINING_ARTIFACT_COMPLETION_STATUS,
            source=source,
            payload={
                "claim_sha256": claim_sha,
                **training_payload,
            },
        )
        training_completion_created = _exclusive_or_identical(
            training_completion_path,
            training_completion_bytes,
            mode=state_mode,
        )
        training_completion = _strict_json(
            training_completion_bytes, label="training completion"
        )

    remote_snapshot = _repo_exists_snapshot(api, contract)
    remote_base_path = paths["remote_base_receipt"]
    if remote_base_path.exists() or remote_base_path.is_symlink():
        remote_base = _state_record(
            contract,
            remote_base_path,
            expected_status=REMOTE_BASE_STATUS,
            mode=state_mode,
        )
        _require_state_lineage(
            remote_base,
            source,
            claim_sha256=claim_sha,
        )
    else:
        if remote_snapshot is not None:
            raise ValueError("formal model repo pre-existed without remote-base receipt")
        remote_base_bytes = _state_bytes(
            contract,
            status=REMOTE_BASE_STATUS,
            source=source,
            payload={
                "claim_sha256": claim_sha,
                "repo": contract.destination["repo"],
                "repo_type": "model",
                "private": True,
                "repo_initially_absent": True,
                "target_paths_initially_absent": True,
                "tag_initially_absent": True,
            },
        )
        _exclusive_or_identical(remote_base_path, remote_base_bytes, mode=state_mode)
        remote_base = _strict_json(remote_base_bytes, label="remote base")
    if remote_base.get("repo_initially_absent") is not True:
        raise ValueError("formal remote-base receipt drifted")
    if remote_snapshot is None and any(
        paths[name].exists() or paths[name].is_symlink()
        for name in ("payload_commit_receipt", "completion_staging")
    ):
        raise ValueError("formal remote repo vanished after a durable downstream receipt")

    main_revision, remote_files, repo_mutations = _ensure_model_repo(api, contract)
    payload_present, manifest_present = _remote_target_partition(remote_files)
    early_tag = _tag_snapshot(
        api,
        repo=contract.destination["repo"],
        repo_type="model",
        tag=contract.destination["tag"],
    )
    if early_tag is not None and (
        manifest_present != set(MANIFEST_TARGETS)
        or early_tag[1] != main_revision
    ):
        raise ValueError("formal model tag exists before a matching complete manifest tree")

    payload_receipt_path = paths["payload_commit_receipt"]
    if payload_receipt_path.exists() or payload_receipt_path.is_symlink():
        payload_receipt = _state_record(
            contract,
            payload_receipt_path,
            expected_status=PAYLOAD_COMMIT_STATUS,
            mode=state_mode,
        )
        _require_state_lineage(
            payload_receipt,
            source,
            claim_sha256=claim_sha,
        )
        payload_commit = payload_receipt.get("payload_commit")
        remote_base_commit = payload_receipt.get("remote_base_commit")
        if not isinstance(payload_commit, str) or _COMMIT.fullmatch(payload_commit) is None:
            raise ValueError("payload commit receipt identity drifted")
        if (
            not isinstance(remote_base_commit, str)
            or _COMMIT.fullmatch(remote_base_commit) is None
        ):
            raise ValueError("payload commit receipt base identity drifted")
        expected_payload_records = [
            _file_record(path, payload_files[path]) for path in sorted(payload_files)
        ]
        if payload_receipt.get("payload_files") != expected_payload_records:
            raise ValueError("payload commit receipt artifact inventory drifted")
        payload_mutations = 0
    elif payload_present == set(PAYLOAD_TARGETS) and not manifest_present:
        payload_commit = main_revision
        remote_base_commit = _recover_payload_base(
            api,
            contract,
            payload_commit=payload_commit,
        )
        payload_mutations = 0
    elif (
        payload_present == set(PAYLOAD_TARGETS)
        and manifest_present == set(MANIFEST_TARGETS)
    ):
        remote_base_commit, payload_commit = _recover_manifest_chain(
            api,
            contract,
            manifest_commit=main_revision,
        )
        payload_mutations = 0
    elif not payload_present and not manifest_present:
        remote_base_commit = main_revision
        _validate_initial_model_base(
            api,
            contract,
            base_commit=remote_base_commit,
        )
        payload_commit, payload_mutations = _commit_files(
            api=api,
            operation_factory=operation_factory,
            contract=contract,
            files=payload_files,
            parent=main_revision,
            title=contract.output["payload_commit"]["commit_title"],
        )
    else:
        raise ValueError("manifest remote state exists without payload receipt")
    payload_base_files = _download_and_compare_revision(
        api=api,
        download_fn=download_fn,
        contract=contract,
        revision=payload_commit,
        expected=payload_files,
        fresh_parent=fresh_download_parent,
    )
    remote_base_records = [
        _file_record(path, payload_base_files[path])
        for path in sorted(payload_base_files)
    ]
    if payload_receipt_path.exists() or payload_receipt_path.is_symlink():
        if payload_receipt.get("remote_base_files") != remote_base_records:
            raise ValueError("payload receipt base blob inventory drifted")
    payload_receipt_bytes = _state_bytes(
        contract,
        status=PAYLOAD_COMMIT_STATUS,
        source=source,
        payload={
            "claim_sha256": claim_sha,
            "remote_base_commit": remote_base_commit,
            "remote_base_files": remote_base_records,
            "payload_commit": payload_commit,
            "payload_files": [
                _file_record(path, payload_files[path]) for path in sorted(payload_files)
            ],
        },
    )
    _exclusive_or_identical(payload_receipt_path, payload_receipt_bytes, mode=state_mode)

    manifest_files = _manifest_files(
        contract=contract,
        source=source,
        runtime=runtime,
        payload_commit=payload_commit,
        payload_files=payload_files,
        training_completion=training_completion,
    )
    _publish_local_files(artifact_root, manifest_files, mode=artifact_mode)
    current_main, current_files = _repo_snapshot(api, contract)
    present_manifests = set(MANIFEST_TARGETS) & set(current_files)
    if present_manifests == set(MANIFEST_TARGETS):
        manifest_commit = current_main
        manifest_mutations = 0
    elif not present_manifests and current_main == payload_commit:
        manifest_commit, manifest_mutations = _commit_files(
            api=api,
            operation_factory=operation_factory,
            contract=contract,
            files=manifest_files,
            parent=payload_commit,
            title=contract.output["manifest_commit"]["commit_title"],
        )
    else:
        raise ValueError("formal model manifest remote state is conflicting")
    if manifest_commit == payload_commit:
        raise ValueError("payload and manifest commits must be distinct")
    _validate_two_commit_chain(
        api,
        contract,
        remote_base_commit=remote_base_commit,
        payload_commit=payload_commit,
        manifest_commit=manifest_commit,
    )
    initial_base_files = _download_and_compare_revision(
        api=api,
        download_fn=download_fn,
        contract=contract,
        revision=remote_base_commit,
        expected={},
        fresh_parent=fresh_download_parent,
    )
    manifest_base_files = _download_and_compare_revision(
        api=api,
        download_fn=download_fn,
        contract=contract,
        revision=manifest_commit,
        expected={**payload_files, **manifest_files},
        fresh_parent=fresh_download_parent,
    )
    if initial_base_files != payload_base_files or initial_base_files != manifest_base_files:
        raise ValueError("formal model base blobs changed across immutable revisions")
    tag_object, tag_mutations = _create_or_validate_tag(
        api=api,
        contract=contract,
        manifest_commit=manifest_commit,
    )
    checkpoint_replay = _validate_checkpoints_from_manifests(
        payload_files, manifest_files
    )
    if (
        _tag_snapshot(
            api,
            repo=contract.destination["repo"],
            repo_type="model",
            tag=contract.destination["tag"],
        )
        != (tag_object, manifest_commit)
    ):
        raise ValueError("formal model tag moved during checkpoint replay")

    completion_payload = {
        "claim_sha256": claim_sha,
        "payload_commit": payload_commit,
        "remote_base_commit": remote_base_commit,
        "remote_base_files": remote_base_records,
        "manifest_commit": manifest_commit,
        "annotated_tag_object": tag_object,
        "tag": contract.destination["tag"],
        "repo": contract.destination["repo"],
        "payload_files": [
            _file_record(path, payload_files[path]) for path in sorted(payload_files)
        ],
        "manifest_files": [
            _file_record(path, manifest_files[path]) for path in sorted(manifest_files)
        ],
        "selection_sha256": dict(training_completion["selection_sha256"]),
        "operation_counts": dict(training_completion["operation_counts"]),
        "checkpoint_replay": checkpoint_replay,
        "gate_trained": True,
        "fresh16_access_authorized": False,
        "legacy_dev5_access_authorized": False,
        "confirm20_access_authorized": False,
        "matched_nll_authorized": False,
        "closed_loop_authorized": False,
    }
    completion_bytes = _state_bytes(
        contract,
        status=FINAL_COMPLETION_STATUS,
        source=source,
        payload=completion_payload,
    )
    staging_created = _exclusive_or_identical(
        paths["completion_staging"], completion_bytes, mode=state_mode
    )
    if paths["final_completion"].exists() or paths["final_completion"].is_symlink():
        raise ValueError("final completion unexpectedly appeared before terminal link")
    os.link(paths["completion_staging"], paths["final_completion"])
    _fsync_directory(paths["final_completion"].parent)
    final_bytes, final_meta = _regular_file_bytes(
        paths["final_completion"], label="final completion", mode=state_mode
    )
    staged_bytes, staged_meta = _regular_file_bytes(
        paths["completion_staging"], label="staged completion", mode=state_mode
    )
    if (
        final_bytes != completion_bytes
        or staged_bytes != completion_bytes
        or (final_meta.st_dev, final_meta.st_ino)
        != (staged_meta.st_dev, staged_meta.st_ino)
    ):
        raise ValueError("terminal completion hard-link publication drifted")
    remote_mutations = repo_mutations + payload_mutations + manifest_mutations + tag_mutations
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": RUN_STATUS,
        "contract_sha256": contract.sha256,
        "source": _source_record(source),
        "runtime": runtime.to_payload(),
        "claim_created": claim_created,
        "cache_input_completion_created": cache_completion_created,
        "training_artifact_completion_created": training_completion_created,
        "completion_staging_created": staging_created,
        "remote_base_commit": remote_base_commit,
        "payload_commit": payload_commit,
        "manifest_commit": manifest_commit,
        "annotated_tag_object": tag_object,
        "remote_mutation_call_count": remote_mutations,
        "selection_sha256": dict(training_completion["selection_sha256"]),
        "operation_counts": dict(training_completion["operation_counts"]),
        "checkpoint_replay": checkpoint_replay,
        "training_executed": training_completion_created,
        "gate_trained": True,
        "fresh16_access_authorized": False,
        "legacy_dev5_access_authorized": False,
        "confirm20_access_authorized": False,
        "matched_nll_authorized": False,
        "closed_loop_authorized": False,
    }


__all__ = [
    "CACHE_INPUT_COMPLETION_STATUS",
    "FINAL_COMPLETION_STATUS",
    "GLOBAL_CLAIM_STATUS",
    "PAYLOAD_COMMIT_STATUS",
    "REMOTE_BASE_STATUS",
    "RUNNER_FREEZE_STATUS",
    "RUN_STATUS",
    "SOURCE_VALIDATION_STATUS",
    "TRAINING_ARTIFACT_COMPLETION_STATUS",
    "VALIDATE_STATUS",
    "download_formal_cache_inputs",
    "execute_formal_train",
    "load_runner_freeze",
    "materialize_runner_freeze",
    "validate_clean_pushed_source",
    "validate_execution_b_source",
    "validate_execution_runtime",
    "validate_source_a",
]
