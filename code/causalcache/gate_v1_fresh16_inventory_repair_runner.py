"""Fail-closed execution wrapper for the fresh-16 inventory repair.

The scientific evaluation remains implemented by the frozen fresh-16 runner.
This module only proves that the invalid v1 attempt is still intact and that
its abandoned destination remains absent before the repaired execution starts.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache import gate_v1_fresh16_evaluation_runner as base_runner


validate_source_a = base_runner.validate_source_a
materialize_runner_freeze = base_runner.materialize_runner_freeze
validate_execution_b_source = base_runner.validate_execution_b_source
capture_docker_inspect_receipt = base_runner.capture_docker_inspect_receipt

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_STATE_NAME = re.compile(r"[a-z0-9_]+")
_RETAINED_STATE_NAMES = ("runtime_receipts", "global_claim")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _parent_failure(contract: Any) -> Mapping[str, Any]:
    repair_value = getattr(contract, "repair", None)
    if repair_value is None:
        overlay = getattr(contract, "overlay", None)
        if isinstance(overlay, Mapping):
            repair_value = overlay.get("inventory_repair")
    if repair_value is None:
        data = _mapping(getattr(contract, "data", None), "repair contract data")
        repair_value = data.get("inventory_repair")
    repair = _mapping(repair_value, "inventory repair")
    return _mapping(repair.get("parent_failure"), "parent failure")


def _real_data_root(data_root: Path) -> Path:
    supplied = Path(data_root)
    try:
        metadata = supplied.lstat()
    except OSError as error:
        raise ValueError("data root is missing or unsafe") from error
    if supplied.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("data root must be a real directory")
    resolved = supplied.resolve(strict=True)
    if resolved != supplied.absolute():
        raise ValueError("data root must be a canonical real directory")
    return resolved


def _under_data_root(data_root: Path, canonical: Any, *, label: str) -> Path:
    if not isinstance(canonical, str):
        raise ValueError(f"{label} path must be text")
    pure = PurePosixPath(canonical)
    if (
        not pure.is_absolute()
        or pure.parts[:2] != ("/", "data")
        or pure.as_posix() != canonical
        or any(part in {"", ".", ".."} for part in pure.parts[2:])
    ):
        raise ValueError(f"{label} must be canonically rooted under /data")
    root = _real_data_root(data_root)
    result = root.joinpath(*pure.parts[2:])
    current = root
    for part in pure.parts[2:-1]:
        current /= part
        if not os.path.lexists(current):
            break
        metadata = current.lstat()
        if current.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} parent is unsafe")
    return result


def _regular_file_identity(
    data_root: Path,
    raw_record: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    record = _mapping(raw_record, label)
    if set(record) - {"path", "mode", "size_bytes", "sha256", "state_name"}:
        raise ValueError(f"{label} contains unsupported identity fields")
    mode = record.get("mode")
    size = record.get("size_bytes")
    digest = record.get("sha256")
    if (
        type(mode) is not int
        or mode not in {0o400, 0o444, 0o600, 0o644}
        or type(size) is not int
        or size < 0
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    ):
        raise ValueError(f"{label} identity is malformed")
    path = _under_data_root(data_root, record.get("path"), label=label)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != mode
            or before.st_size != size
        ):
            raise ValueError(f"{label} type, mode, or size drifted")
        hasher = hashlib.sha256()
        observed_size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            observed_size += len(chunk)
            hasher.update(chunk)
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
    if (
        fingerprint(before) != fingerprint(after)
        or observed_size != size
        or hasher.hexdigest() != digest
    ):
        raise ValueError(f"{label} bytes changed or drifted")
    return {
        "path": record["path"],
        "mode": mode,
        "size_bytes": size,
        "sha256": digest,
    }


def _real_directory(data_root: Path, canonical: Any, *, label: str) -> Path:
    path = _under_data_root(data_root, canonical, label=label)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or unsafe") from error
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or path.resolve(strict=True) != path.absolute()
    ):
        raise ValueError(f"{label} must be a real canonical directory")
    return path


def _assert_absent(data_root: Path, canonical: str, *, label: str) -> None:
    path = _under_data_root(data_root, canonical, label=label)
    if os.path.lexists(path):
        raise ValueError(f"{label} must remain absent")


def validate_retained_v1_failure(
    contract: Any,
    data_root: Path,
) -> Mapping[str, Any]:
    """Verify every retained byte and every required v1 absence read-only."""
    failure = _parent_failure(contract)
    namespace = failure.get("old_state_namespace")
    state_root = _real_directory(data_root, namespace, label="old state namespace")
    if tuple(sorted(entry.name for entry in os.scandir(state_root))) != (
        "ordered-state-receipts",
    ):
        raise ValueError(
            "old state namespace must contain only ordered-state-receipts"
        )
    receipt_records = tuple(
        _mapping(item, "old ordered receipt")
        for item in _sequence(
            failure.get("ordered_receipts"), "old ordered receipts"
        )
    )
    names = tuple(record.get("state_name") for record in receipt_records)
    if names != _RETAINED_STATE_NAMES:
        raise ValueError("old ordered receipt prefix must be runtime/global only")
    receipt_root = _real_directory(
        data_root,
        f"{namespace}/ordered-state-receipts",
        label="old ordered receipt directory",
    )
    expected_filenames = tuple(
        Path(str(record.get("path"))).name for record in receipt_records
    )
    if expected_filenames != (
        "00-runtime_receipts.json",
        "01-global_claim.json",
    ):
        raise ValueError("old ordered receipt filenames drifted")
    for ordinal, (name, record) in enumerate(zip(names, receipt_records, strict=True)):
        if record.get("path") != (
            f"{namespace}/ordered-state-receipts/{ordinal:02d}-{name}.json"
        ):
            raise ValueError("old ordered receipt path escaped its state namespace")
    observed_filenames = tuple(sorted(entry.name for entry in os.scandir(receipt_root)))
    if observed_filenames != tuple(sorted(expected_filenames)):
        raise ValueError("old ordered receipt set must contain exactly two files")
    receipts = tuple(
        _regular_file_identity(
            data_root,
            record,
            label=f"old ordered receipt {record['state_name']}",
        )
        for record in receipt_records
    )

    successor_names = tuple(
        _sequence(
            failure.get("expected_absent_successor_state_names"),
            "old absent successor state names",
        )
    )
    if (
        len(successor_names) != 14
        or len(set(successor_names)) != 14
        or any(
            not isinstance(name, str) or _SAFE_STATE_NAME.fullmatch(name) is None
            for name in successor_names
        )
    ):
        raise ValueError("old absent successor state inventory is malformed")
    all_names = (*_RETAINED_STATE_NAMES, *successor_names)
    if len(set(all_names)) != 16:
        raise ValueError("old state inventory is duplicated")
    for ordinal, name in enumerate(successor_names, start=2):
        _assert_absent(
            data_root,
            f"{namespace}/ordered-state-receipts/{ordinal:02d}-{name}.json",
            label=f"old successor receipt {name}",
        )

    if failure.get("old_artifact_directory_expected_empty") is not True:
        raise ValueError("old artifact directory must be frozen as empty")
    artifact_root = _real_directory(
        data_root,
        failure.get("old_artifact_directory"),
        label="old artifact directory",
    )
    if any(True for _entry in os.scandir(artifact_root)):
        raise ValueError("old artifact directory is not empty")

    runtime = _regular_file_identity(
        data_root,
        failure.get("old_runtime_receipt"),
        label="old Docker runtime receipt",
    )
    run_evidence = _mapping(failure.get("run_evidence"), "old run evidence")
    if set(run_evidence) != {"log", "started", "exit"}:
        raise ValueError("old run evidence must contain log/start/exit only")
    run_files = {
        name: _regular_file_identity(
            data_root,
            run_evidence[name],
            label=f"old run {name} evidence",
        )
        for name in ("log", "started", "exit")
    }
    return {
        "status": "VALID_RETAINED_GATE_V1_FRESH16_V1_PRESEMANTIC_FAILURE",
        "old_state_namespace": str(namespace),
        "ordered_receipt_state_names": list(names),
        "ordered_receipt_count": len(receipts),
        "absent_successor_receipt_count": len(successor_names),
        "old_artifact_directory": str(failure["old_artifact_directory"]),
        "old_artifact_entry_count": 0,
        "old_runtime_receipt": dict(runtime),
        "run_evidence": {name: dict(record) for name, record in run_files.items()},
        "fresh_semantic_access_count": 0,
        "remote_mutation_count": 0,
    }


def _assert_hf_repo_absent(
    api: Any,
    destination: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    repo = destination.get("repo")
    repo_type = destination.get("repo_type")
    if (
        not isinstance(repo, str)
        or not repo
        or repo.count("/") != 1
        or repo_type not in {"dataset", "model"}
    ):
        raise ValueError(f"{label} identity is malformed")
    try:
        api.repo_info(repo, repo_type=repo_type)
    except Exception as error:
        if error.__class__.__name__ != "RepositoryNotFoundError":
            raise
    else:
        raise ValueError(f"{label} must remain absent")
    return {
        "repo": repo,
        "repo_type": repo_type,
        "repository_existed": False,
        "remote_read_count": 1,
        "remote_mutation_count": 0,
    }


def validate_abandoned_v1_destination_absent(
    contract: Any,
    api: Any,
) -> Mapping[str, Any]:
    failure = _parent_failure(contract)
    old = _mapping(failure.get("old_destination"), "old destination")
    if (
        old.get("expected_absent") is not True
        or old.get("remote_mutation_count") != 0
        or not isinstance(old.get("tag"), str)
        or not old["tag"]
    ):
        raise ValueError("old destination absence contract drifted")
    return _assert_hf_repo_absent(api, old, label="abandoned v1 destination")


def validate_repair_destination_absent(
    contract: Any,
    api: Any,
) -> Mapping[str, Any]:
    destination = _mapping(
        getattr(contract, "destination", None), "repair destination"
    )
    return _assert_hf_repo_absent(api, destination, label="repair destination")


def execute_fresh16_inventory_repair(
    *,
    mode: str,
    contract: Any,
    api: Any,
    download_fn: Callable[..., str],
    operation_factory: Callable[..., Any],
    expected_execution_b_git_commit: str,
    data_root: Path,
    fresh_download_parent: Path,
    model_dir: Path,
    snapshot_manifest: Path,
    docker_inspect_receipt: Path | None,
    devices: Sequence[str],
    gpu_uuids: Sequence[str],
    evidence_validator: Callable[[Any, Path], Mapping[str, Any]] | None = None,
    old_destination_validator: Callable[[Any, Any], Mapping[str, Any]] | None = None,
    repair_destination_validator: Callable[[Any, Any], Mapping[str, Any]] | None = None,
    execute_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Guard the base runner with immutable failure and destination checks."""
    if mode not in {"run", "validate"}:
        raise ValueError("fresh16 inventory repair mode must be run or validate")
    evidence_validator = evidence_validator or validate_retained_v1_failure
    old_destination_validator = (
        old_destination_validator or validate_abandoned_v1_destination_absent
    )
    repair_destination_validator = (
        repair_destination_validator or validate_repair_destination_absent
    )
    execute_fn = execute_fn or base_runner.execute_fresh16_evaluation

    retained = evidence_validator(contract, data_root)
    old_remote = old_destination_validator(contract, api)
    new_remote = (
        repair_destination_validator(contract, api) if mode == "run" else None
    )
    preflight = {
        "retained_parent_failure": dict(retained),
        "abandoned_v1_destination": dict(old_remote),
        "repair_destination_before_run": (
            dict(new_remote) if new_remote is not None else None
        ),
        "fresh_semantic_access_before_delegate": 0,
        "remote_mutation_before_delegate": 0,
    }
    result = execute_fn(
        mode=mode,
        contract=contract,
        api=api,
        download_fn=download_fn,
        operation_factory=operation_factory,
        expected_execution_b_git_commit=expected_execution_b_git_commit,
        data_root=data_root,
        fresh_download_parent=fresh_download_parent,
        model_dir=model_dir,
        snapshot_manifest=snapshot_manifest,
        docker_inspect_receipt=docker_inspect_receipt,
        devices=devices,
        gpu_uuids=gpu_uuids,
        execution_preflight=preflight,
    )
    return {
        **result,
        "inventory_repair_preflight": preflight,
    }


__all__ = [
    "capture_docker_inspect_receipt",
    "execute_fresh16_inventory_repair",
    "materialize_runner_freeze",
    "validate_abandoned_v1_destination_absent",
    "validate_execution_b_source",
    "validate_repair_destination_absent",
    "validate_retained_v1_failure",
    "validate_source_a",
]
