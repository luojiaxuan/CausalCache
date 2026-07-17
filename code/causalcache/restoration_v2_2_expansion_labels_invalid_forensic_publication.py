"""Crash-recoverable publication for the P0 invalid-forensic archive.

The module is transport-only: publication preserves the P0 invalid-attempt
classification and never makes the archive eligible for formal labels.
"""

from __future__ import annotations

import io
import os
import re
import stat
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.restoration_v2_2_expansion_labels_invalid_forensic import (
    ARCHIVE_STATUS as P0_ARCHIVE_STATUS,
    ARTIFACT_CLASS,
    InvalidForensicContract,
    PROTOCOL_ID as P0_PROTOCOL_ID,
    canonical_json_bytes,
    pretty_json_bytes,
    read_invalid_forensic_archive,
    sha256_bytes,
    strict_pretty_json_object_bytes,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = (
    "causalcache_restoration_v2_2_expansion_exact_labels_"
    "invalid_forensic_publication_v1"
)
SOURCE_STATUS = "source_only_frozen_before_any_private_hf_publication"
SIDECAR_STATUS = "IMMUTABLE_INVALID_FORENSIC_ARCHIVE_PUBLICATION_MANIFEST_V1"
CLAIM_STATUS = "CLAIMED_INVALID_FORENSIC_ARCHIVE_PUBLICATION_V1"
COMPLETION_STATUS = "COMPLETED_INVALID_FORENSIC_ARCHIVE_PUBLICATION_V1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_"
    "expansion_labels_invalid_forensic_publication_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "affb54cf44a656394c983cebe5d004d3e1ff216537c43da942dc031fa6baaccb"
)
PAIR_COMMIT_TITLE = "Publish invalid expansion exact-label forensic archive"
REMOTE_EMPTY = "EMPTY"
REMOTE_BYTE_IDENTICAL_UNTAGGED = "BYTE_IDENTICAL_UNTAGGED"
REMOTE_TAGGED_BYTE_IDENTICAL = "TAGGED_BYTE_IDENTICAL"
ALLOWED_REMOTE_STATES = (
    REMOTE_EMPTY,
    REMOTE_BYTE_IDENTICAL_UNTAGGED,
    REMOTE_TAGGED_BYTE_IDENTICAL,
)
DESTINATION = {
    "repo": (
        "gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-"
        "invalid-attempts-mobile"
    ),
    "repo_type": "dataset",
    "private": True,
    "tag": "v2.2-expansion-exact-labels-v1-attempt-1-forensic-v1",
    "archive_path": (
        "attempts/restoration-v2-2-expansion-exact-labels-v1/attempt-1/"
        "raw-evidence-v1.tar"
    ),
    "sidecar_path": (
        "attempts/restoration-v2-2-expansion-exact-labels-v1/attempt-1/"
        "archive-manifest-v1.json"
    ),
    "archive_sidecar_single_commit_required": True,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40,64}")


@dataclass(frozen=True)
class InvalidForensicPublicationContract:
    data: Mapping[str, Any]
    sha256: str
    source_path: Path | None = None

    @property
    def destination(self) -> Mapping[str, Any]:
        return _mapping(self.data["destination"], "destination")

    @property
    def p0_source(self) -> Mapping[str, Any]:
        return _mapping(self.data["p0_forensic_source"], "p0 forensic source")

    @property
    def local_state(self) -> Mapping[str, Any]:
        return _mapping(self.data["local_state"], "local state")


@dataclass(frozen=True)
class PreparedInvalidForensicPublication:
    archive_path: Path
    archive_bytes: bytes
    archive_sha256: str
    sidecar: Mapping[str, Any]
    sidecar_bytes: bytes
    sidecar_sha256: str
    source_identity: Mapping[str, Any]


@dataclass(frozen=True)
class RemoteInspection:
    state: str
    main_revision: str
    immutable_revision: str | None


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} schema drifted")


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase immutable commit")
    return value


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be nonempty")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be canonical relative POSIX")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} must be absolute")
    return Path(value)


def _regular_file_bytes(path: str | Path, label: str) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            Path(path),
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is missing or is not safe to read") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise ValueError(f"{label} changed while being read")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise ValueError(f"{label} size changed while being read")
    return payload


def validate_publication_contract_data(value: Any) -> Mapping[str, Any]:
    data = dict(_mapping(value, "invalid-forensic publication contract"))
    _exact_keys(
        data,
        {
            "schema_version",
            "protocol_id",
            "status",
            "destination",
            "p0_forensic_source",
            "publication_contract",
            "local_state",
            "git_source_contract",
            "fresh_download_contract",
            "formal_consumption",
        },
        "invalid-forensic publication contract",
    )
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("protocol_id") != PROTOCOL_ID
        or data.get("status") != SOURCE_STATUS
    ):
        raise ValueError("invalid-forensic publication identity drifted")

    destination = dict(_mapping(data["destination"], "destination"))
    if destination != DESTINATION:
        raise ValueError("private HF destination drifted")
    _safe_relative_path(destination["archive_path"], "archive destination")
    _safe_relative_path(destination["sidecar_path"], "sidecar destination")

    p0_source = dict(_mapping(data["p0_forensic_source"], "p0 source"))
    _exact_keys(
        p0_source,
        {
            "protocol_id",
            "config_path",
            "config_sha256",
            "artifact_class",
            "required_archive_status",
        },
        "p0 source",
    )
    if (
        p0_source.get("protocol_id") != P0_PROTOCOL_ID
        or p0_source.get("artifact_class") != ARTIFACT_CLASS
        or p0_source.get("required_archive_status") != P0_ARCHIVE_STATUS
    ):
        raise ValueError("P0 invalid-forensic source identity drifted")
    _safe_relative_path(p0_source["config_path"], "P0 config path")
    _sha256(p0_source["config_sha256"], "P0 config SHA256")

    publication = dict(
        _mapping(data["publication_contract"], "publication contract")
    )
    if publication != {
        "allowed_remote_states": list(ALLOWED_REMOTE_STATES),
        "commit_title": PAIR_COMMIT_TITLE,
        "conflict_action": "FAIL_CLOSED",
        "no_overwrite": True,
        "sidecar_status": SIDECAR_STATUS,
        "tag_response_loss_requery_required": True,
    }:
        raise ValueError("publication recovery state machine drifted")

    local_state = dict(_mapping(data["local_state"], "local state"))
    _exact_keys(
        local_state,
        {
            "claim_path",
            "completion_seal_path",
            "file_mode",
            "o_excl_first_write",
            "same_bytes_reusable",
        },
        "local state",
    )
    _absolute_path(local_state["claim_path"], "claim path")
    _absolute_path(local_state["completion_seal_path"], "completion seal path")
    if local_state != {
        **local_state,
        "file_mode": 0o600,
        "o_excl_first_write": True,
        "same_bytes_reusable": True,
    }:
        raise ValueError("local publication state contract drifted")
    if local_state["claim_path"] == local_state["completion_seal_path"]:
        raise ValueError("claim and completion paths must differ")

    git_source = dict(_mapping(data["git_source_contract"], "git source"))
    _exact_keys(
        git_source,
        {
            "branch",
            "origin_name",
            "origin_url",
            "head_must_equal_origin_main",
            "worktree_must_be_clean",
            "required_source_paths",
        },
        "git source",
    )
    paths = git_source.get("required_source_paths")
    if not isinstance(paths, list) or len(paths) != 5 or len(set(paths)) != 5:
        raise ValueError("git source inventory must contain five unique paths")
    for source_path in paths:
        _safe_relative_path(source_path, "required source path")
    if (
        git_source.get("branch") != "main"
        or git_source.get("origin_name") != "origin"
        or git_source.get("origin_url")
        != "https://github.com/luojiaxuan/CausalCache.git"
        or git_source.get("head_must_equal_origin_main") is not True
        or git_source.get("worktree_must_be_clean") is not True
    ):
        raise ValueError("clean pushed Git source contract drifted")

    if dict(_mapping(data["fresh_download_contract"], "fresh download")) != {
        "download_directory_must_start_empty": True,
        "force_download": True,
        "immutable_revision_pre_post_stable": True,
        "p0_strict_archive_readback_required": True,
        "tag_revision_pre_post_stable": True,
    }:
        raise ValueError("fresh-download verification contract drifted")
    if dict(_mapping(data["formal_consumption"], "formal consumption")) != {
        "formal_label_loader_eligible": False,
        "gate_training_unlocked": False,
        "producer_attempt_reclassified": False,
    }:
        raise ValueError("publication cannot reclassify invalid evidence")
    return data


def publication_contract_from_data(
    value: Any,
) -> InvalidForensicPublicationContract:
    data = validate_publication_contract_data(value)
    payload = pretty_json_bytes(data)
    return InvalidForensicPublicationContract(
        data=data,
        sha256=sha256_bytes(payload),
    )


def load_frozen_publication_contract(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> InvalidForensicPublicationContract:
    supplied = Path(path)
    if not supplied.is_absolute():
        if repository_root is None:
            raise ValueError("relative config requires an explicit repository root")
        supplied = Path(repository_root) / supplied
    payload = _regular_file_bytes(supplied, "frozen publication config")
    if sha256_bytes(payload) != FROZEN_CONFIG_SHA256:
        raise ValueError("frozen publication config SHA256 drifted")
    data = strict_pretty_json_object_bytes(payload, label="publication config")
    validate_publication_contract_data(data)
    return InvalidForensicPublicationContract(
        data=data,
        sha256=FROZEN_CONFIG_SHA256,
        source_path=supplied.resolve(),
    )


def _git_bytes(repository_root: Path, arguments: Sequence[str]) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    return result.stdout


def validate_clean_pushed_source(
    *,
    repository_root: str | Path,
    contract: InvalidForensicPublicationContract,
) -> Mapping[str, Any]:
    supplied_root = Path(repository_root)
    if supplied_root.is_symlink():
        raise ValueError("repository root must not be a symlink")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise ValueError("repository root must be a non-symlink directory")
    top = Path(
        _git_bytes(root, ["rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    ).resolve()
    if top != root:
        raise ValueError("repository root is not the Git toplevel")
    source_contract = _mapping(contract.data["git_source_contract"], "git source")
    branch = _git_bytes(root, ["symbolic-ref", "--short", "HEAD"]).decode().strip()
    head = _git_bytes(root, ["rev-parse", "HEAD"]).decode().strip()
    origin_main = _git_bytes(
        root,
        [
            "rev-parse",
            (
                f"refs/remotes/{source_contract['origin_name']}/"
                f"{source_contract['branch']}"
            ),
        ],
    ).decode().strip()
    origin_url = _git_bytes(
        root, ["remote", "get-url", source_contract["origin_name"]]
    ).decode().strip()
    status = _git_bytes(
        root, ["status", "--porcelain=v1", "--untracked-files=all"]
    )
    if (
        branch != source_contract["branch"]
        or head != origin_main
        or origin_url != source_contract["origin_url"]
        or status
    ):
        raise ValueError("publication source must be clean main at pushed origin/main")
    _commit(head, "publication source Git commit")

    inventory: list[Mapping[str, Any]] = []
    for relative in source_contract["required_source_paths"]:
        local = _regular_file_bytes(root / relative, f"source file {relative}")
        committed = _git_bytes(root, ["show", f"{head}:{relative}"])
        if local != committed:
            raise ValueError(f"source file differs from pushed commit: {relative}")
        inventory.append(
            {
                "path": relative,
                "sha256": sha256_bytes(local),
                "size_bytes": len(local),
            }
        )
    return {
        "publication_source_git_commit": head,
        "origin_main_git_commit": origin_main,
        "branch": branch,
        "origin_url": origin_url,
        "source_path_count": len(inventory),
        "source_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
    }


def _validate_source_identity(value: Any) -> Mapping[str, Any]:
    source = dict(_mapping(value, "publication source identity"))
    _exact_keys(
        source,
        {
            "publication_source_git_commit",
            "origin_main_git_commit",
            "branch",
            "origin_url",
            "source_path_count",
            "source_inventory_sha256",
        },
        "publication source identity",
    )
    if (
        _commit(source["publication_source_git_commit"], "source commit")
        != _commit(source["origin_main_git_commit"], "origin/main commit")
        or source["branch"] != "main"
        or source["origin_url"] != "https://github.com/luojiaxuan/CausalCache.git"
        or type(source["source_path_count"]) is not int
        or source["source_path_count"] <= 0
    ):
        raise ValueError("publication source identity is not clean pushed main")
    _sha256(source["source_inventory_sha256"], "source inventory SHA256")
    return source


def prepare_invalid_forensic_publication(
    *,
    archive_path: str | Path,
    p0_contract: InvalidForensicContract,
    contract: InvalidForensicPublicationContract,
    source_identity: Mapping[str, Any],
) -> PreparedInvalidForensicPublication:
    source = _validate_source_identity(source_identity)
    if (
        p0_contract.sha256 != contract.p0_source["config_sha256"]
        or p0_contract.data["protocol_id"] != contract.p0_source["protocol_id"]
    ):
        raise ValueError("P0 contract differs from the frozen publication source")
    archive = Path(os.path.abspath(archive_path))
    before = _regular_file_bytes(archive, "P0 invalid-forensic archive")
    evidence = read_invalid_forensic_archive(archive, contract=p0_contract)
    after = _regular_file_bytes(archive, "P0 invalid-forensic archive")
    if before != after:
        raise ValueError("P0 archive changed during strict readback")
    if (
        evidence.manifest["status"] != P0_ARCHIVE_STATUS
        or evidence.manifest["artifact_class"] != ARTIFACT_CLASS
        or evidence.manifest["formal_consumption"]["formal_label_loader_eligible"]
        is not False
    ):
        raise ValueError("P0 archive is not permanently invalid forensic evidence")
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": SIDECAR_STATUS,
        "artifact_class": ARTIFACT_CLASS,
        "destination": dict(contract.destination),
        "source_archive": {
            "p0_protocol_id": P0_PROTOCOL_ID,
            "p0_archive_status": P0_ARCHIVE_STATUS,
            "p0_config_path": contract.p0_source["config_path"],
            "p0_config_sha256": p0_contract.sha256,
            "archive_sha256": sha256_bytes(before),
            "archive_size_bytes": len(before),
            "tree_inventory_sha256": evidence.tree_inventory_sha256,
            "member_count": len(evidence.files),
        },
        "publication_source": {
            "publication_config_sha256": contract.sha256,
            **source,
        },
        "formal_consumption": {
            "formal_label_loader_eligible": False,
            "gate_training_unlocked": False,
            "producer_attempt_reclassified": False,
        },
    }
    sidecar_bytes = pretty_json_bytes(sidecar)
    return PreparedInvalidForensicPublication(
        archive_path=archive,
        archive_bytes=before,
        archive_sha256=sha256_bytes(before),
        sidecar=sidecar,
        sidecar_bytes=sidecar_bytes,
        sidecar_sha256=sha256_bytes(sidecar_bytes),
        source_identity=source,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_or_identical_state(path: Path, payload: bytes, mode: int) -> bool:
    try:
        parent = path.parent.lstat()
    except OSError as error:
        raise ValueError(f"state parent does not exist: {path.parent}") from error
    if not stat.S_ISDIR(parent.st_mode):
        raise ValueError("state parent must be a non-symlink directory")
    if path.exists() or path.is_symlink():
        existing = _regular_file_bytes(path, "existing publication state")
        metadata = path.stat(follow_symlinks=False)
        if stat.S_IMODE(metadata.st_mode) != mode or existing != payload:
            raise ValueError("existing publication state is not byte-identical")
        return False
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(temporary, flags, mode)
    except FileExistsError as error:
        raise ValueError(
            "publication state staging path unexpectedly exists"
        ) from error
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("state write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        # note (luojiaxuan): The O_EXCL staging file is fully durable before
        # hard-link no-replace publication, so a crash cannot expose a partial
        # final claim or completion seal.
        os.link(temporary, path)
        _fsync_directory(path.parent)
        return True
    except FileExistsError:
        existing = _regular_file_bytes(path, "existing publication state")
        metadata = path.stat(follow_symlinks=False)
        if stat.S_IMODE(metadata.st_mode) != mode or existing != payload:
            raise ValueError("existing publication state is not byte-identical")
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _claim_payload(
    *,
    contract: InvalidForensicPublicationContract,
    prepared: PreparedInvalidForensicPublication,
) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": CLAIM_STATUS,
        "destination": dict(contract.destination),
        "archive_sha256": prepared.archive_sha256,
        "sidecar_sha256": prepared.sidecar_sha256,
        "publication_source": dict(prepared.sidecar["publication_source"]),
    }


def _tag_target(api: Any, contract: InvalidForensicPublicationContract) -> str | None:
    destination = contract.destination
    refs = api.list_repo_refs(
        destination["repo"], repo_type=destination["repo_type"]
    )
    tags = [ref for ref in refs.tags if ref.name == destination["tag"]]
    if len(tags) > 1:
        raise ValueError("remote tag is ambiguous")
    if not tags:
        return None
    return _commit(tags[0].target_commit, "remote tag target")


def _revision_sha(
    api: Any,
    contract: InvalidForensicPublicationContract,
    revision: str,
) -> str:
    destination = contract.destination
    info = api.dataset_info(destination["repo"], revision=revision)
    if info.private is not True:
        raise ValueError("destination dataset must remain private")
    return _commit(info.sha, f"remote revision {revision}")


def _download_pair(
    *,
    download_fn: Callable[..., str],
    contract: InvalidForensicPublicationContract,
    prepared: PreparedInvalidForensicPublication,
    p0_contract: InvalidForensicContract,
    revision: str,
    local_dir: Path,
) -> None:
    if not local_dir.is_dir() or local_dir.is_symlink() or any(local_dir.iterdir()):
        raise ValueError("fresh download directory must start empty")
    destination = contract.destination
    archive_path = Path(
        download_fn(
            repo_id=destination["repo"],
            filename=destination["archive_path"],
            repo_type=destination["repo_type"],
            revision=revision,
            local_dir=local_dir,
            force_download=True,
        )
    )
    _assert_download_path(
        local_dir,
        archive_path,
        destination["archive_path"],
        "downloaded archive",
    )
    sidecar_path = Path(
        download_fn(
            repo_id=destination["repo"],
            filename=destination["sidecar_path"],
            repo_type=destination["repo_type"],
            revision=revision,
            local_dir=local_dir,
            force_download=True,
        )
    )
    _assert_download_path(
        local_dir,
        archive_path,
        destination["archive_path"],
        "downloaded archive",
    )
    _assert_download_path(
        local_dir,
        sidecar_path,
        destination["sidecar_path"],
        "downloaded sidecar",
    )
    archive_bytes = _regular_file_bytes(archive_path, "downloaded archive")
    sidecar_bytes = _regular_file_bytes(sidecar_path, "downloaded sidecar")
    if (
        archive_bytes != prepared.archive_bytes
        or sidecar_bytes != prepared.sidecar_bytes
    ):
        raise ValueError("remote archive or sidecar differs from frozen local bytes")
    parsed = strict_pretty_json_object_bytes(
        sidecar_bytes, label="downloaded publication sidecar"
    )
    if parsed != prepared.sidecar:
        raise ValueError("downloaded publication sidecar semantic binding drifted")
    evidence = read_invalid_forensic_archive(archive_path, contract=p0_contract)
    if (
        evidence.tree_inventory_sha256
        != prepared.sidecar["source_archive"]["tree_inventory_sha256"]
        or len(evidence.files) != prepared.sidecar["source_archive"]["member_count"]
    ):
        raise ValueError("downloaded P0 strict archive readback drifted")


def _assert_download_path(
    local_dir: Path,
    returned_path: Path,
    relative_path: str,
    label: str,
) -> None:
    if not local_dir.is_absolute() or not returned_path.is_absolute():
        raise ValueError("HF download paths must be absolute")
    expected = local_dir / relative_path
    if returned_path != expected:
        raise ValueError("HF download returned a lexically unexpected path")
    try:
        relative = returned_path.relative_to(local_dir)
    except ValueError as error:
        raise ValueError(
            "HF download returned a path outside the fresh root"
        ) from error
    root_metadata = local_dir.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("fresh download root became a symlink or non-directory")
    current = local_dir
    for component in relative.parts[:-1]:
        current = current / component
        metadata = current.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} has a symlink or non-directory component")
    final = returned_path.lstat()
    if not stat.S_ISREG(final.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")


def _verify_single_pair_commit(
    *,
    api: Any,
    contract: InvalidForensicPublicationContract,
    immutable_revision: str,
) -> str:
    destination = contract.destination
    immutable = _commit(immutable_revision, "pair commit immutable revision")
    commits = api.list_repo_commits(
        destination["repo"],
        repo_type=destination["repo_type"],
        revision=immutable,
    )
    if len(commits) < 2:
        raise ValueError("pair commit lacks a direct predecessor")
    head = commits[0]
    predecessor = commits[1]
    head_commit = _commit(head.commit_id, "pair commit history head")
    predecessor_commit = _commit(
        predecessor.commit_id, "pair commit direct predecessor"
    )
    if head_commit != immutable or head.title != PAIR_COMMIT_TITLE:
        raise ValueError("immutable revision is not the frozen pair commit")
    expected = {destination["archive_path"], destination["sidecar_path"]}
    head_files = set(
        api.list_repo_files(
            destination["repo"],
            repo_type=destination["repo_type"],
            revision=head_commit,
        )
    )
    predecessor_files = set(
        api.list_repo_files(
            destination["repo"],
            repo_type=destination["repo_type"],
            revision=predecessor_commit,
        )
    )
    if not expected.issubset(head_files) or expected & predecessor_files:
        raise ValueError(
            "archive and sidecar did not first appear together in one commit"
        )
    return predecessor_commit


def _verify_remote_pair(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: InvalidForensicPublicationContract,
    prepared: PreparedInvalidForensicPublication,
    p0_contract: InvalidForensicContract,
    revision: str,
    fresh_download_parent: Path,
) -> None:
    destination = contract.destination
    _verify_single_pair_commit(
        api=api,
        contract=contract,
        immutable_revision=revision,
    )
    files = set(
        api.list_repo_files(
            destination["repo"],
            repo_type=destination["repo_type"],
            revision=revision,
        )
    )
    expected = {destination["archive_path"], destination["sidecar_path"]}
    if not expected.issubset(files):
        raise ValueError("remote archive and sidecar pair is incomplete")
    with tempfile.TemporaryDirectory(
        prefix="invalid-forensic-hf-readback-",
        dir=fresh_download_parent,
    ) as directory:
        _download_pair(
            download_fn=download_fn,
            contract=contract,
            prepared=prepared,
            p0_contract=p0_contract,
            revision=revision,
            local_dir=Path(directory),
        )


def _validated_fresh_download_parent(value: str | Path) -> Path:
    parent = Path(value)
    if (
        not parent.is_absolute()
        or not parent.is_dir()
        or parent.is_symlink()
    ):
        raise ValueError(
            "fresh-download parent must be an absolute non-symlink directory"
        )
    return parent


def inspect_remote_state(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: InvalidForensicPublicationContract,
    prepared: PreparedInvalidForensicPublication,
    p0_contract: InvalidForensicContract,
    fresh_download_parent: str | Path,
) -> RemoteInspection:
    parent = _validated_fresh_download_parent(fresh_download_parent)
    destination = contract.destination
    main_revision = _revision_sha(api, contract, "main")
    tag_target = _tag_target(api, contract)
    if tag_target is not None:
        resolved = _revision_sha(api, contract, destination["tag"])
        if resolved != tag_target:
            raise ValueError("tag target and resolved immutable revision disagree")
        _verify_remote_pair(
            api=api,
            download_fn=download_fn,
            contract=contract,
            prepared=prepared,
            p0_contract=p0_contract,
            revision=tag_target,
            fresh_download_parent=parent,
        )
        return RemoteInspection(
            state=REMOTE_TAGGED_BYTE_IDENTICAL,
            main_revision=main_revision,
            immutable_revision=tag_target,
        )

    files = set(
        api.list_repo_files(
            destination["repo"],
            repo_type=destination["repo_type"],
            revision=main_revision,
        )
    )
    archive_present = destination["archive_path"] in files
    sidecar_present = destination["sidecar_path"] in files
    if archive_present != sidecar_present:
        raise ValueError("remote archive/sidecar partial state is a conflict")
    if not archive_present:
        return RemoteInspection(
            state=REMOTE_EMPTY,
            main_revision=main_revision,
            immutable_revision=None,
        )
    _verify_remote_pair(
        api=api,
        download_fn=download_fn,
        contract=contract,
        prepared=prepared,
        p0_contract=p0_contract,
        revision=main_revision,
        fresh_download_parent=parent,
    )
    return RemoteInspection(
        state=REMOTE_BYTE_IDENTICAL_UNTAGGED,
        main_revision=main_revision,
        immutable_revision=main_revision,
    )


def _operation(
    operation_factory: Callable[..., Any], path: str, payload: bytes
) -> Any:
    return operation_factory(path_in_repo=path, path_or_fileobj=io.BytesIO(payload))


def _commit_oid(value: Any) -> str:
    for attribute in ("oid", "commit_id"):
        candidate = getattr(value, attribute, None)
        if candidate is not None:
            return _commit(candidate, "HF commit response")
    raise ValueError("HF commit response lacks an immutable commit id")


def _create_tag_recovering_response_loss(
    *,
    api: Any,
    contract: InvalidForensicPublicationContract,
    immutable_revision: str,
) -> None:
    destination = contract.destination
    try:
        api.create_tag(
            destination["repo"],
            tag=destination["tag"],
            tag_message="Freeze invalid expansion exact-label attempt forensic bytes",
            revision=immutable_revision,
            repo_type=destination["repo_type"],
            exist_ok=False,
        )
    except Exception:
        recovered = _tag_target(api, contract)
        if recovered == immutable_revision:
            return
        raise
    observed = _tag_target(api, contract)
    if observed != immutable_revision:
        raise ValueError("created tag does not resolve to the intended commit")


def _fresh_stability_attestation(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: InvalidForensicPublicationContract,
    prepared: PreparedInvalidForensicPublication,
    p0_contract: InvalidForensicContract,
    immutable_revision: str,
    fresh_download_parent: Path,
) -> Mapping[str, Any]:
    destination = contract.destination
    tag_pre = _tag_target(api, contract)
    immutable_pre = _revision_sha(api, contract, immutable_revision)
    tag_resolved_pre = _revision_sha(api, contract, destination["tag"])
    if {tag_pre, immutable_pre, tag_resolved_pre} != {immutable_revision}:
        raise ValueError("tag or immutable revision drifted before fresh download")
    with tempfile.TemporaryDirectory(
        prefix="invalid-forensic-hf-fresh-",
        dir=fresh_download_parent,
    ) as directory:
        _download_pair(
            download_fn=download_fn,
            contract=contract,
            prepared=prepared,
            p0_contract=p0_contract,
            revision=immutable_revision,
            local_dir=Path(directory),
        )
    tag_post = _tag_target(api, contract)
    immutable_post = _revision_sha(api, contract, immutable_revision)
    tag_resolved_post = _revision_sha(api, contract, destination["tag"])
    if {tag_post, immutable_post, tag_resolved_post} != {immutable_revision}:
        raise ValueError("tag or immutable revision drifted after fresh download")
    return {
        "immutable_revision": immutable_revision,
        "tag_revision_pre": tag_pre,
        "tag_revision_post": tag_post,
        "immutable_revision_pre": immutable_pre,
        "immutable_revision_post": immutable_post,
        "archive_sha256": prepared.archive_sha256,
        "sidecar_sha256": prepared.sidecar_sha256,
        "force_download": True,
        "download_directory_started_empty": True,
        "p0_strict_archive_readback_passed": True,
    }


def _completion_payload(
    *,
    contract: InvalidForensicPublicationContract,
    prepared: PreparedInvalidForensicPublication,
    claim_sha256: str,
    remote_state_before_publish: str,
    immutable_revision: str,
    attestation: Mapping[str, Any],
) -> Mapping[str, Any]:
    if remote_state_before_publish not in ALLOWED_REMOTE_STATES:
        raise ValueError("completion remote state is not allowed")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": COMPLETION_STATUS,
        "destination": dict(contract.destination),
        "claim_sha256": _sha256(claim_sha256, "claim SHA256"),
        "archive_sha256": prepared.archive_sha256,
        "sidecar_sha256": prepared.sidecar_sha256,
        "publication_source": dict(prepared.sidecar["publication_source"]),
        "remote_state_before_publish": remote_state_before_publish,
        "immutable_revision": _commit(immutable_revision, "immutable revision"),
        "fresh_download_attestation": dict(attestation),
        "formal_label_loader_eligible": False,
    }


def _load_existing_completion(
    *,
    path: Path,
    contract: InvalidForensicPublicationContract,
    prepared: PreparedInvalidForensicPublication,
    claim_sha256: str,
) -> Mapping[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    metadata = path.stat(follow_symlinks=False)
    if stat.S_IMODE(metadata.st_mode) != contract.local_state["file_mode"]:
        raise ValueError("completion seal mode drifted")
    completion = strict_pretty_json_object_bytes(
        _regular_file_bytes(path, "completion seal"), label="completion seal"
    )
    required = {
        "schema_version",
        "protocol_id",
        "status",
        "destination",
        "claim_sha256",
        "archive_sha256",
        "sidecar_sha256",
        "publication_source",
        "remote_state_before_publish",
        "immutable_revision",
        "fresh_download_attestation",
        "formal_label_loader_eligible",
    }
    _exact_keys(completion, required, "completion seal")
    if (
        completion["schema_version"] != SCHEMA_VERSION
        or completion["protocol_id"] != PROTOCOL_ID
        or completion["status"] != COMPLETION_STATUS
        or completion["destination"] != contract.destination
        or completion["claim_sha256"] != claim_sha256
        or completion["archive_sha256"] != prepared.archive_sha256
        or completion["sidecar_sha256"] != prepared.sidecar_sha256
        or completion["publication_source"]
        != prepared.sidecar["publication_source"]
        or completion["remote_state_before_publish"] not in ALLOWED_REMOTE_STATES
        or completion["formal_label_loader_eligible"] is not False
    ):
        raise ValueError("completion seal binding drifted")
    _commit(completion["immutable_revision"], "completion immutable revision")
    return completion


def publish_invalid_forensic_archive(
    *,
    api: Any,
    download_fn: Callable[..., str],
    operation_factory: Callable[..., Any],
    contract: InvalidForensicPublicationContract,
    p0_contract: InvalidForensicContract,
    prepared: PreparedInvalidForensicPublication,
    fresh_download_parent: str | Path,
) -> Mapping[str, Any]:
    parent = _validated_fresh_download_parent(fresh_download_parent)
    destination = contract.destination
    claim = _claim_payload(contract=contract, prepared=prepared)
    claim_bytes = pretty_json_bytes(claim)
    claim_sha256 = sha256_bytes(claim_bytes)
    claim_path = Path(contract.local_state["claim_path"])
    completion_path = Path(contract.local_state["completion_seal_path"])
    mode = contract.local_state["file_mode"]
    _exclusive_or_identical_state(claim_path, claim_bytes, mode)

    existing_completion = _load_existing_completion(
        path=completion_path,
        contract=contract,
        prepared=prepared,
        claim_sha256=claim_sha256,
    )
    if existing_completion is not None:
        immutable = existing_completion["immutable_revision"]
        inspection = inspect_remote_state(
            api=api,
            download_fn=download_fn,
            contract=contract,
            prepared=prepared,
            p0_contract=p0_contract,
            fresh_download_parent=parent,
        )
        if (
            inspection.state != REMOTE_TAGGED_BYTE_IDENTICAL
            or inspection.immutable_revision != immutable
        ):
            raise ValueError("completed publication no longer has exact tagged bytes")
        attestation = _fresh_stability_attestation(
            api=api,
            download_fn=download_fn,
            contract=contract,
            prepared=prepared,
            p0_contract=p0_contract,
            immutable_revision=immutable,
            fresh_download_parent=parent,
        )
        expected = _completion_payload(
            contract=contract,
            prepared=prepared,
            claim_sha256=claim_sha256,
            remote_state_before_publish=existing_completion[
                "remote_state_before_publish"
            ],
            immutable_revision=immutable,
            attestation=attestation,
        )
        if existing_completion != expected:
            raise ValueError("completion seal differs from fresh remote attestation")
        return existing_completion

    api.create_repo(
        destination["repo"],
        repo_type=destination["repo_type"],
        private=True,
        exist_ok=True,
    )
    _revision_sha(api, contract, "main")
    inspection = inspect_remote_state(
        api=api,
        download_fn=download_fn,
        contract=contract,
        prepared=prepared,
        p0_contract=p0_contract,
        fresh_download_parent=parent,
    )
    if inspection.state == REMOTE_TAGGED_BYTE_IDENTICAL:
        assert inspection.immutable_revision is not None
        immutable_revision = inspection.immutable_revision
    elif inspection.state == REMOTE_BYTE_IDENTICAL_UNTAGGED:
        assert inspection.immutable_revision is not None
        immutable_revision = inspection.immutable_revision
        _create_tag_recovering_response_loss(
            api=api,
            contract=contract,
            immutable_revision=immutable_revision,
        )
    elif inspection.state == REMOTE_EMPTY:
        operations = [
            _operation(
                operation_factory,
                destination["archive_path"],
                prepared.archive_bytes,
            ),
            _operation(
                operation_factory,
                destination["sidecar_path"],
                prepared.sidecar_bytes,
            ),
        ]
        response = api.create_commit(
            destination["repo"],
            operations=operations,
            commit_message=PAIR_COMMIT_TITLE,
            repo_type=destination["repo_type"],
            revision="main",
            parent_commit=inspection.main_revision,
        )
        immutable_revision = _commit_oid(response)
        _verify_remote_pair(
            api=api,
            download_fn=download_fn,
            contract=contract,
            prepared=prepared,
            p0_contract=p0_contract,
            revision=immutable_revision,
            fresh_download_parent=parent,
        )
        _create_tag_recovering_response_loss(
            api=api,
            contract=contract,
            immutable_revision=immutable_revision,
        )
    else:
        raise AssertionError(f"unknown remote publication state: {inspection.state}")

    attestation = _fresh_stability_attestation(
        api=api,
        download_fn=download_fn,
        contract=contract,
        prepared=prepared,
        p0_contract=p0_contract,
        immutable_revision=immutable_revision,
        fresh_download_parent=parent,
    )
    completion = _completion_payload(
        contract=contract,
        prepared=prepared,
        claim_sha256=claim_sha256,
        remote_state_before_publish=inspection.state,
        immutable_revision=immutable_revision,
        attestation=attestation,
    )
    completion_bytes = pretty_json_bytes(completion)
    _exclusive_or_identical_state(completion_path, completion_bytes, mode)
    return completion


__all__ = [
    "ALLOWED_REMOTE_STATES",
    "CANONICAL_CONFIG_PATH",
    "COMPLETION_STATUS",
    "DESTINATION",
    "FROZEN_CONFIG_SHA256",
    "InvalidForensicPublicationContract",
    "PAIR_COMMIT_TITLE",
    "PreparedInvalidForensicPublication",
    "PROTOCOL_ID",
    "REMOTE_BYTE_IDENTICAL_UNTAGGED",
    "REMOTE_EMPTY",
    "REMOTE_TAGGED_BYTE_IDENTICAL",
    "RemoteInspection",
    "SIDECAR_STATUS",
    "SOURCE_STATUS",
    "inspect_remote_state",
    "load_frozen_publication_contract",
    "prepare_invalid_forensic_publication",
    "publication_contract_from_data",
    "publish_invalid_forensic_archive",
    "validate_clean_pushed_source",
    "validate_publication_contract_data",
]
