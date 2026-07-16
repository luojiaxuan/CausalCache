"""Deterministic raw-artifact packaging for spatial-reference audit v1."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tarfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.data.guiodyssey_restoration_v2 import canonical_json_bytes, sha256_file
from causalcache.spatial_reference_audit_v1 import (
    PROTOCOL_ID,
    load_and_validate_config,
    validate_repository_inputs,
)


ARCHIVE_MEMBER_PREFIX = "spatial-reference-audit-v1"
ARCHIVE_LEDGER_MEMBER = "global_attempt_ledger.json"
CANONICAL_AUDIT_ROOT = Path(
    "/data/experiments/causalcache/spatial-reference-audit-v1"
)
CANONICAL_GLOBAL_LEDGER = Path(
    "/data/experiments/causalcache/.spatial-reference-audit-v1.attempt.json"
)
CANONICAL_RAW_ARCHIVE = Path(
    "/data/experiments/causalcache/spatial-reference-audit-v1.tar"
)
TERMINAL_ATTEMPT_STATUSES = {
    "COMPLETED_SPATIAL_REFERENCE_AUDIT_ATTEMPT",
    "FAILED_SPATIAL_REFERENCE_AUDIT_ATTEMPT",
}
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("audit artifact member path is unsafe")
    return path.as_posix()


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path.read_bytes()


def _collect_tree_files(audit_root: Path) -> dict[str, bytes]:
    try:
        root_metadata = audit_root.lstat()
    except OSError as error:
        raise ValueError("audit root is missing") from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("audit root must be a non-symlink directory")
    files: dict[str, bytes] = {}
    for path in sorted(audit_root.rglob("*")):
        relative = _safe_relative_path(path.relative_to(audit_root).as_posix())
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("audit root contains a symlink or non-regular file")
        if relative == ARCHIVE_LEDGER_MEMBER:
            raise ValueError("audit root collides with the reserved ledger member")
        files[relative] = path.read_bytes()
    if not files:
        raise ValueError("audit root contains no regular evidence files")
    return files


def collect_spatial_reference_audit_files(
    audit_root: str | Path,
    global_attempt_ledger: str | Path,
) -> dict[str, bytes]:
    root = Path(audit_root)
    ledger_path = Path(global_attempt_ledger)
    files = _collect_tree_files(root)
    ledger_payload = _regular_file_bytes(
        ledger_path,
        label="global attempt ledger",
    )
    try:
        ledger = json.loads(ledger_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("global attempt ledger is invalid JSON") from error
    if not isinstance(ledger, dict):
        raise TypeError("global attempt ledger must be a JSON object")
    files[ARCHIVE_LEDGER_MEMBER] = ledger_payload
    return files


def _deterministic_ustar_bytes(files: Mapping[str, bytes]) -> bytes:
    destination = io.BytesIO()
    with tarfile.open(
        fileobj=destination,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for relative in sorted(files):
            safe_relative = _safe_relative_path(relative)
            payload = files[relative]
            if not isinstance(payload, bytes):
                raise TypeError("audit artifact payloads must be bytes")
            info = tarfile.TarInfo(f"{ARCHIVE_MEMBER_PREFIX}/{safe_relative}")
            info.type = tarfile.REGTYPE
            info.size = len(payload)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return destination.getvalue()


def _read_canonical_archive_bytes(payload: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if names != sorted(names) or len(names) != len(set(names)):
                raise ValueError("audit archive member ordering or uniqueness drifted")
            prefix = f"{ARCHIVE_MEMBER_PREFIX}/"
            for member in members:
                if (
                    not member.isfile()
                    or member.type != tarfile.REGTYPE
                    or not member.name.startswith(prefix)
                    or member.mode != 0o644
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or member.pax_headers
                ):
                    raise ValueError("audit archive contains a non-canonical member")
                relative = _safe_relative_path(member.name[len(prefix) :])
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("audit archive member cannot be read")
                member_payload = source.read()
                if len(member_payload) != member.size:
                    raise ValueError("audit archive member size drifted")
                files[relative] = member_payload
    except tarfile.TarError as error:
        raise ValueError("audit archive is not readable USTAR") from error
    if ARCHIVE_LEDGER_MEMBER not in files:
        raise ValueError("audit archive lacks the global attempt ledger")
    if payload != _deterministic_ustar_bytes(files):
        raise ValueError("audit archive bytes are not canonical deterministic USTAR")
    return files


def read_spatial_reference_audit_archive(path: str | Path) -> dict[str, bytes]:
    archive = Path(path)
    payload = _regular_file_bytes(archive, label="audit archive")
    return _read_canonical_archive_bytes(payload)


def _tree_inventory_sha256(files: Mapping[str, bytes]) -> str:
    inventory = [
        {
            "path": relative,
            "size_bytes": len(files[relative]),
            "sha256": hashlib.sha256(files[relative]).hexdigest(),
        }
        for relative in sorted(files)
    ]
    return hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()


def _validate_packaging_config(config: Mapping[str, Any]) -> None:
    packaging = config.get("artifact_packaging")
    if not isinstance(packaging, Mapping):
        raise TypeError("spatial-reference config lacks artifact_packaging")
    expected = {
        "format": "ustar",
        "archive_member_prefix": ARCHIVE_MEMBER_PREFIX,
        "global_attempt_ledger_member": ARCHIVE_LEDGER_MEMBER,
        "canonical_local_archive_path": str(CANONICAL_RAW_ARCHIVE),
        "normalized_file_mode": "0644",
        "normalized_uid": 0,
        "normalized_gid": 0,
        "normalized_mtime": 0,
        "new_output_only": True,
        "symlinks_or_nonregular_files_allowed": False,
        "byte_identity_rebuild_required": True,
    }
    if dict(packaging) != expected:
        raise ValueError("spatial-reference artifact packaging contract drifted")


def _require_path_identity(actual: Path, expected: Path, *, label: str) -> None:
    if actual.resolve() != expected:
        raise ValueError(f"{label} differs from the canonical path")


def package_spatial_reference_audit_evidence(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    source_git_commit: str,
    audit_root: str | Path,
    global_attempt_ledger: str | Path,
    output_archive: str | Path,
    require_canonical_location: bool = True,
) -> dict[str, Any]:
    if GIT_SHA_PATTERN.fullmatch(source_git_commit) is None:
        raise ValueError("source Git commit must be a full lowercase commit SHA")
    repository = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    if config_file != repository / "code/configs/spatial_reference_audit_v1.json":
        raise ValueError("spatial-reference packaging config path is not canonical")
    config = load_and_validate_config(config_file)
    validate_repository_inputs(repository, config)
    _validate_packaging_config(config)
    root = Path(audit_root)
    ledger_path = Path(global_attempt_ledger)
    archive = Path(output_archive)
    if require_canonical_location:
        _require_path_identity(root, CANONICAL_AUDIT_ROOT, label="audit root")
        _require_path_identity(
            ledger_path,
            CANONICAL_GLOBAL_LEDGER,
            label="global attempt ledger",
        )
        _require_path_identity(
            archive,
            CANONICAL_RAW_ARCHIVE,
            label="audit archive",
        )
    files = collect_spatial_reference_audit_files(root, ledger_path)
    ledger = json.loads(files[ARCHIVE_LEDGER_MEMBER])
    if (
        ledger.get("schema_version") != "1.0.0"
        or ledger.get("protocol_id") != PROTOCOL_ID
        or ledger.get("attempt_id") != "spatial-reference-audit-v1"
        or ledger.get("status") not in TERMINAL_ATTEMPT_STATUSES
        or ledger.get("source_git_commit") != source_git_commit
        or ledger.get("config_sha256") != sha256_file(config_file)
    ):
        raise ValueError("global attempt ledger identity is not terminal or frozen")
    payload = _deterministic_ustar_bytes(files)
    if _read_canonical_archive_bytes(payload) != files:
        raise ValueError("in-memory audit archive reconstruction drifted")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    directory = os.open(archive.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    reread = read_spatial_reference_audit_archive(archive)
    if reread != files:
        raise ValueError("written audit archive differs from the source tree")
    return {
        "status": "PACKAGED_SPATIAL_REFERENCE_AUDIT_V1_RAW_EVIDENCE",
        "source_git_commit": source_git_commit,
        "attempt_status": ledger["status"],
        "archive_path": str(archive.absolute()),
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "file_count": len(files),
        "tree_inventory_sha256": _tree_inventory_sha256(files),
    }
