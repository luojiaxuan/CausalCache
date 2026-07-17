"""Read-only repair for the P1 annotated-tag resolution interface failure.

The child validates an already published private archive/sidecar pair.  It
never mutates Hugging Face state and never changes the invalid producer label.
"""

from __future__ import annotations

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
    canonical_json_bytes,
    pretty_json_bytes,
    read_invalid_forensic_archive,
    sha256_bytes,
    strict_pretty_json_object_bytes,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = (
    "causalcache_restoration_v2_2_expansion_exact_labels_"
    "invalid_forensic_tag_resolution_v1"
)
SOURCE_STATUS = "source_only_frozen_before_read_only_tag_resolution_validation"
CLAIM_STATUS = "CLAIMED_READ_ONLY_INVALID_FORENSIC_TAG_RESOLUTION_V1"
COMPLETION_STATUS = "COMPLETED_READ_ONLY_INVALID_FORENSIC_TAG_RESOLUTION_V1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_"
    "expansion_labels_invalid_forensic_tag_resolution_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "f257dcdebb218533a080739e8a5067cc44fe35fd533ad57808db5ce789cfd956"
)
P1_PROTOCOL_ID = (
    "causalcache_restoration_v2_2_expansion_exact_labels_"
    "invalid_forensic_publication_v1"
)
P1_CLAIM_STATUS = "CLAIMED_INVALID_FORENSIC_ARCHIVE_PUBLICATION_V1"
P1_SIDECAR_STATUS = "IMMUTABLE_INVALID_FORENSIC_ARCHIVE_PUBLICATION_MANIFEST_V1"
P0_PROTOCOL_ID = (
    "causalcache_restoration_v2_2_expansion_exact_labels_invalid_forensic_v1"
)
PAIR_COMMIT_TITLE = "Publish invalid expansion exact-label forensic archive"
ALLOWED_REMOTE_API_METHODS = (
    "dataset_info",
    "list_repo_commits",
    "list_repo_files",
    "list_repo_refs",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class TagResolutionContract:
    data: Mapping[str, Any]
    sha256: str
    source_path: Path | None = None

    @property
    def remote(self) -> Mapping[str, Any]:
        return _mapping(self.data["remote_evidence"], "remote evidence")

    @property
    def parent(self) -> Mapping[str, Any]:
        return _mapping(
            self.data["parent_publication_attempt"], "parent publication"
        )

    @property
    def child_state(self) -> Mapping[str, Any]:
        return _mapping(self.data["child_local_state"], "child local state")

    @property
    def p0_source(self) -> Mapping[str, Any]:
        return _mapping(self.data["p0_forensic_source"], "P0 forensic source")


@dataclass(frozen=True)
class ParentPublicationState:
    claim: Mapping[str, Any]
    claim_bytes: bytes
    claim_sha256: str


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
        raise ValueError(f"{label} must be one lowercase 40-hex commit")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return Path(value)


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


def _validate_formal_consumption(value: Any) -> Mapping[str, Any]:
    formal = dict(_mapping(value, "formal consumption"))
    expected = {
        "formal_label_loader_eligible": False,
        "gate_training_unlocked": False,
        "producer_attempt_reclassified": False,
        "transport_validation_only": True,
    }
    if formal != expected:
        raise ValueError("formal-consumption boundary drifted")
    return formal


def validate_tag_resolution_contract_data(value: Any) -> Mapping[str, Any]:
    data = dict(_mapping(value, "tag-resolution contract"))
    _exact_keys(
        data,
        {
            "schema_version",
            "protocol_id",
            "status",
            "git_source_contract",
            "p0_forensic_source",
            "parent_publication_attempt",
            "remote_evidence",
            "remote_operation_contract",
            "fresh_download_contract",
            "child_local_state",
            "formal_consumption",
        },
        "tag-resolution contract",
    )
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("protocol_id") != PROTOCOL_ID
        or data.get("status") != SOURCE_STATUS
    ):
        raise ValueError("tag-resolution source identity drifted")

    git = dict(_mapping(data["git_source_contract"], "Git source contract"))
    _exact_keys(
        git,
        {
            "branch",
            "failure_evidence_commit",
            "failure_evidence_path",
            "failure_evidence_sha256",
            "head_must_equal_origin_main",
            "origin_name",
            "origin_url",
            "parent_publication_source_commit",
            "required_source_paths",
            "worktree_must_be_clean",
        },
        "Git source contract",
    )
    if (
        git["branch"] != "main"
        or git["origin_name"] != "origin"
        or git["origin_url"]
        != "https://github.com/luojiaxuan/CausalCache.git"
        or git["head_must_equal_origin_main"] is not True
        or git["worktree_must_be_clean"] is not True
    ):
        raise ValueError("Git source policy drifted")
    _commit(git["parent_publication_source_commit"], "parent source commit")
    _commit(git["failure_evidence_commit"], "failure evidence commit")
    _safe_relative_path(git["failure_evidence_path"], "failure evidence path")
    _sha256(git["failure_evidence_sha256"], "failure evidence SHA256")
    paths = git["required_source_paths"]
    if (
        not isinstance(paths, list)
        or not paths
        or len(paths) != len(set(paths))
    ):
        raise ValueError("required source paths must be one unique nonempty list")
    for path in paths:
        _safe_relative_path(path, "required source path")
    if CANONICAL_CONFIG_PATH not in paths:
        raise ValueError("tag-resolution config is absent from source inventory")

    p0 = dict(_mapping(data["p0_forensic_source"], "P0 forensic source"))
    _exact_keys(
        p0,
        {
            "protocol_id",
            "config_path",
            "config_sha256",
            "required_archive_status",
            "archive_sha256",
            "archive_size_bytes",
            "archive_member_count",
            "tree_inventory_sha256",
        },
        "P0 forensic source",
    )
    if (
        p0["protocol_id"] != P0_PROTOCOL_ID
        or p0["required_archive_status"] != P0_ARCHIVE_STATUS
    ):
        raise ValueError("P0 forensic identity drifted")
    _safe_relative_path(p0["config_path"], "P0 config path")
    _sha256(p0["config_sha256"], "P0 config SHA256")
    _sha256(p0["archive_sha256"], "P0 archive SHA256")
    _positive_int(p0["archive_size_bytes"], "P0 archive size")
    _positive_int(p0["archive_member_count"], "P0 archive member count")
    _sha256(p0["tree_inventory_sha256"], "P0 tree inventory SHA256")

    parent = dict(
        _mapping(data["parent_publication_attempt"], "parent publication")
    )
    _exact_keys(
        parent,
        {
            "protocol_id",
            "source_git_commit",
            "failure_evidence_commit",
            "config_path",
            "config_sha256",
            "parent_source_path_count",
            "parent_source_inventory_sha256",
            "claim",
            "completion_seal",
        },
        "parent publication",
    )
    if parent["protocol_id"] != P1_PROTOCOL_ID:
        raise ValueError("parent P1 protocol drifted")
    if (
        _commit(parent["source_git_commit"], "P1 source commit")
        != git["parent_publication_source_commit"]
        or _commit(parent["failure_evidence_commit"], "P1 failure commit")
        != git["failure_evidence_commit"]
    ):
        raise ValueError("parent P1 Git lineage drifted")
    _safe_relative_path(parent["config_path"], "P1 config path")
    _sha256(parent["config_sha256"], "P1 config SHA256")
    _positive_int(parent["parent_source_path_count"], "parent source path count")
    _sha256(
        parent["parent_source_inventory_sha256"],
        "parent source inventory SHA256",
    )
    claim = dict(_mapping(parent["claim"], "parent claim"))
    _exact_keys(
        claim,
        {"path", "mode", "sha256", "size_bytes", "status"},
        "parent claim",
    )
    _absolute_path(claim["path"], "parent claim path")
    if claim["mode"] != 0o600 or claim["status"] != P1_CLAIM_STATUS:
        raise ValueError("parent claim mode or status drifted")
    _sha256(claim["sha256"], "parent claim SHA256")
    _positive_int(claim["size_bytes"], "parent claim size")
    completion = dict(_mapping(parent["completion_seal"], "P1 completion"))
    _exact_keys(completion, {"path", "must_be_absent"}, "P1 completion")
    _absolute_path(completion["path"], "P1 completion path")
    if completion["must_be_absent"] is not True:
        raise ValueError("parent P1 completion must remain absent")
    if claim["path"] == completion["path"]:
        raise ValueError("parent P1 claim and completion paths must differ")

    remote = dict(_mapping(data["remote_evidence"], "remote evidence"))
    _exact_keys(
        remote,
        {
            "repo",
            "repo_type",
            "private",
            "tag",
            "tag_object_identity",
            "tag_resolved_commit",
            "main_resolved_commit",
            "archive_path",
            "archive_sha256",
            "sidecar_path",
            "sidecar_sha256",
            "pair_commit",
        },
        "remote evidence",
    )
    if (
        not isinstance(remote["repo"], str)
        or not remote["repo"]
        or remote["repo_type"] != "dataset"
        or remote["private"] is not True
        or not isinstance(remote["tag"], str)
        or not remote["tag"]
    ):
        raise ValueError("private dataset destination drifted")
    tag_object = _commit(remote["tag_object_identity"], "tag object identity")
    tag_resolved = _commit(remote["tag_resolved_commit"], "tag-resolved commit")
    main_resolved = _commit(remote["main_resolved_commit"], "main commit")
    if tag_object == tag_resolved:
        raise ValueError("annotated tag object and resolved commit must stay distinct")
    if tag_resolved != main_resolved:
        raise ValueError("frozen main and tag-resolved commits must agree")
    archive_path = _safe_relative_path(remote["archive_path"], "archive path")
    sidecar_path = _safe_relative_path(remote["sidecar_path"], "sidecar path")
    if archive_path == sidecar_path:
        raise ValueError("archive and sidecar paths must differ")
    if (
        _sha256(remote["archive_sha256"], "remote archive SHA256")
        != p0["archive_sha256"]
    ):
        raise ValueError("remote archive differs from P0 archive")
    _sha256(remote["sidecar_sha256"], "remote sidecar SHA256")
    pair = dict(_mapping(remote["pair_commit"], "pair commit"))
    _exact_keys(
        pair,
        {
            "revision",
            "title",
            "predecessor_commit",
            "predecessor_files",
            "files",
        },
        "pair commit",
    )
    if (
        _commit(pair["revision"], "pair revision") != tag_resolved
        or pair["title"] != PAIR_COMMIT_TITLE
    ):
        raise ValueError("pair commit identity drifted")
    _commit(pair["predecessor_commit"], "pair predecessor")
    predecessor_files = pair["predecessor_files"]
    pair_files = pair["files"]
    if (
        not isinstance(predecessor_files, list)
        or not isinstance(pair_files, list)
        or predecessor_files != sorted(set(predecessor_files))
        or pair_files != sorted(set(pair_files))
    ):
        raise ValueError("pair file inventories must be sorted unique lists")
    for path in [*predecessor_files, *pair_files]:
        _safe_relative_path(path, "pair inventory path")
    if set(pair_files) != set(predecessor_files) | {archive_path, sidecar_path}:
        raise ValueError("pair snapshot must add exactly archive and sidecar")
    if {archive_path, sidecar_path} & set(predecessor_files):
        raise ValueError("target paths must be absent from predecessor")

    operation = dict(
        _mapping(data["remote_operation_contract"], "remote operation contract")
    )
    if operation != {
        "allowed_api_methods": list(ALLOWED_REMOTE_API_METHODS),
        "remote_mutation_call_count": 0,
    }:
        raise ValueError("read-only remote operation boundary drifted")
    fresh = dict(
        _mapping(data["fresh_download_contract"], "fresh-download contract")
    )
    if fresh != {
        "download_directory_must_start_empty": True,
        "force_download": True,
        "fresh_parent_must_be_absolute_non_symlink": True,
        "p0_strict_archive_readback_required": True,
        "pre_post_remote_identity_stable": True,
    }:
        raise ValueError("fresh-download contract drifted")
    child = dict(_mapping(data["child_local_state"], "child local state"))
    _exact_keys(
        child,
        {
            "claim_path",
            "completion_seal_path",
            "file_mode",
            "o_excl_first_write",
            "same_bytes_reusable",
        },
        "child local state",
    )
    claim_path = _absolute_path(child["claim_path"], "child claim path")
    completion_path = _absolute_path(
        child["completion_seal_path"], "child completion path"
    )
    if (
        child["file_mode"] != 0o600
        or child["o_excl_first_write"] is not True
        or child["same_bytes_reusable"] is not True
        or claim_path == completion_path
    ):
        raise ValueError("child state contract drifted")
    if str(claim_path) in {claim["path"], completion["path"]} or str(
        completion_path
    ) in {claim["path"], completion["path"]}:
        raise ValueError("child state must not overlap parent P1 state")
    _validate_formal_consumption(data["formal_consumption"])
    return data


def tag_resolution_contract_from_data(value: Any) -> TagResolutionContract:
    data = validate_tag_resolution_contract_data(value)
    return TagResolutionContract(data=data, sha256=sha256_bytes(pretty_json_bytes(data)))


def load_frozen_tag_resolution_contract(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> TagResolutionContract:
    source = Path(path)
    if repository_root is not None and not source.is_absolute():
        source = Path(repository_root) / source
    payload = _regular_file_bytes(source, "tag-resolution config")
    data = strict_pretty_json_object_bytes(payload, label="tag-resolution config")
    contract = tag_resolution_contract_from_data(data)
    if sha256_bytes(payload) != FROZEN_CONFIG_SHA256:
        raise ValueError("tag-resolution config SHA256 drifted")
    return TagResolutionContract(
        data=contract.data,
        sha256=FROZEN_CONFIG_SHA256,
        source_path=source.resolve(),
    )


def _regular_file_bytes(path: str | Path, label: str) -> bytes:
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
            raise ValueError(f"{label} must be a regular non-symlink file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    payload = b"".join(chunks)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or before.st_mode != after.st_mode
        or len(payload) != after.st_size
    ):
        raise ValueError(f"{label} changed while being read")
    return payload


def _git_bytes(root: Path, arguments: Sequence[str]) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git source validation failed: {detail}")
    return process.stdout


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    process = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise ValueError(f"required Git ancestor is absent: {ancestor}")


def validate_clean_pushed_source(
    *, repository_root: str | Path, contract: TagResolutionContract
) -> Mapping[str, Any]:
    supplied = Path(repository_root)
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ValueError("repository root must be absolute and non-symlink")
    root = supplied.resolve()
    if not root.is_dir():
        raise ValueError("repository root must be a directory")
    top = Path(
        _git_bytes(root, ["rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    ).resolve()
    if top != root:
        raise ValueError("repository root is not the Git toplevel")
    source = _mapping(contract.data["git_source_contract"], "Git source")
    branch = _git_bytes(root, ["symbolic-ref", "--short", "HEAD"]).decode().strip()
    head = _git_bytes(root, ["rev-parse", "HEAD"]).decode().strip()
    origin_main = _git_bytes(
        root,
        [
            "rev-parse",
            f"refs/remotes/{source['origin_name']}/{source['branch']}",
        ],
    ).decode().strip()
    origin_url = _git_bytes(
        root, ["remote", "get-url", source["origin_name"]]
    ).decode().strip()
    status = _git_bytes(root, ["status", "--porcelain=v1", "--untracked-files=all"])
    if (
        branch != source["branch"]
        or head != origin_main
        or origin_url != source["origin_url"]
        or status
    ):
        raise ValueError("tag-resolution source must be clean pushed main")
    _commit(head, "tag-resolution source commit")
    parent_commit = source["parent_publication_source_commit"]
    failure_commit = source["failure_evidence_commit"]
    _require_ancestor(root, parent_commit, failure_commit)
    _require_ancestor(root, failure_commit, head)

    parent = contract.parent
    parent_config = _git_bytes(root, ["show", f"{parent_commit}:{parent['config_path']}"])
    if sha256_bytes(parent_config) != parent["config_sha256"]:
        raise ValueError("parent P1 config does not match frozen source commit")
    parent_config_data = strict_pretty_json_object_bytes(
        parent_config, label="parent P1 config"
    )
    parent_paths = _mapping(
        parent_config_data["git_source_contract"], "parent Git source"
    )["required_source_paths"]
    if len(parent_paths) != parent["parent_source_path_count"]:
        raise ValueError("parent P1 source path count drifted")
    parent_inventory: list[Mapping[str, Any]] = []
    for relative in parent_paths:
        payload = _git_bytes(root, ["show", f"{parent_commit}:{relative}"])
        parent_inventory.append(
            {
                "path": relative,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    parent_inventory_sha = sha256_bytes(canonical_json_bytes(parent_inventory))
    if parent_inventory_sha != parent["parent_source_inventory_sha256"]:
        raise ValueError("parent P1 source inventory drifted")
    failure_payload = _git_bytes(
        root, ["show", f"{failure_commit}:{source['failure_evidence_path']}"]
    )
    if sha256_bytes(failure_payload) != source["failure_evidence_sha256"]:
        raise ValueError("failure evidence commit bytes drifted")

    inventory: list[Mapping[str, Any]] = []
    for relative in source["required_source_paths"]:
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
        "tag_resolution_source_git_commit": head,
        "origin_main_git_commit": origin_main,
        "branch": branch,
        "origin_url": origin_url,
        "source_path_count": len(inventory),
        "source_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "parent_publication_source_git_commit": parent_commit,
        "parent_source_inventory_sha256": parent_inventory_sha,
        "failure_evidence_commit": failure_commit,
        "failure_evidence_sha256": sha256_bytes(failure_payload),
    }


def _assert_state_parent(path: Path, label: str) -> None:
    try:
        metadata = path.parent.lstat()
    except OSError as error:
        raise ValueError(f"{label} parent is missing") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} parent must be a non-symlink directory")


def _assert_absent(path: Path, label: str) -> None:
    _assert_state_parent(path, label)
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise ValueError(f"{label} must remain absent")


def _parent_destination(contract: TagResolutionContract) -> Mapping[str, Any]:
    remote = contract.remote
    return {
        "archive_path": remote["archive_path"],
        "archive_sidecar_single_commit_required": True,
        "private": True,
        "repo": remote["repo"],
        "repo_type": remote["repo_type"],
        "sidecar_path": remote["sidecar_path"],
        "tag": remote["tag"],
    }


def validate_parent_publication_state(
    contract: TagResolutionContract,
) -> ParentPublicationState:
    parent = contract.parent
    completion_path = Path(parent["completion_seal"]["path"])
    _assert_absent(completion_path, "parent P1 completion seal")
    claim_contract = parent["claim"]
    claim_path = Path(claim_contract["path"])
    _assert_state_parent(claim_path, "parent P1 claim")
    metadata = claim_path.stat(follow_symlinks=False)
    if stat.S_IMODE(metadata.st_mode) != claim_contract["mode"]:
        raise ValueError("parent P1 claim mode drifted")
    payload = _regular_file_bytes(claim_path, "parent P1 claim")
    if (
        len(payload) != claim_contract["size_bytes"]
        or sha256_bytes(payload) != claim_contract["sha256"]
    ):
        raise ValueError("parent P1 claim bytes drifted")
    claim = strict_pretty_json_object_bytes(payload, label="parent P1 claim")
    _exact_keys(
        claim,
        {
            "schema_version",
            "protocol_id",
            "status",
            "destination",
            "archive_sha256",
            "sidecar_sha256",
            "publication_source",
        },
        "parent P1 claim",
    )
    publication_source = dict(
        _mapping(claim["publication_source"], "parent publication source")
    )
    _exact_keys(
        publication_source,
        {
            "publication_config_sha256",
            "publication_source_git_commit",
            "origin_main_git_commit",
            "branch",
            "origin_url",
            "source_path_count",
            "source_inventory_sha256",
        },
        "parent publication source",
    )
    if (
        claim["schema_version"] != SCHEMA_VERSION
        or claim["protocol_id"] != parent["protocol_id"]
        or claim["status"] != claim_contract["status"]
        or claim["destination"] != _parent_destination(contract)
        or claim["archive_sha256"] != contract.remote["archive_sha256"]
        or claim["sidecar_sha256"] != contract.remote["sidecar_sha256"]
        or publication_source["publication_config_sha256"]
        != parent["config_sha256"]
        or publication_source["publication_source_git_commit"]
        != parent["source_git_commit"]
        or publication_source["origin_main_git_commit"]
        != parent["source_git_commit"]
        or publication_source["branch"] != "main"
        or publication_source["origin_url"]
        != "https://github.com/luojiaxuan/CausalCache.git"
        or publication_source["source_path_count"]
        != parent["parent_source_path_count"]
        or publication_source["source_inventory_sha256"]
        != parent["parent_source_inventory_sha256"]
    ):
        raise ValueError("parent P1 claim semantic binding drifted")
    return ParentPublicationState(
        claim=claim,
        claim_bytes=payload,
        claim_sha256=sha256_bytes(payload),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_or_identical_state(path: Path, payload: bytes, mode: int) -> bool:
    _assert_state_parent(path, "child state")
    if path.exists() or path.is_symlink():
        metadata = path.stat(follow_symlinks=False)
        existing = _regular_file_bytes(path, "existing child state")
        if stat.S_IMODE(metadata.st_mode) != mode or existing != payload:
            raise ValueError("existing child state is not byte-identical")
        return False
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
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
        raise ValueError("child state staging path unexpectedly exists") from error
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("child state write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        # note (luojiaxuan): The durable O_EXCL staging file is linked into the
        # final namespace without replacement, so partial final seals are never visible.
        os.link(temporary, path)
        _fsync_directory(path.parent)
        return True
    except FileExistsError:
        metadata = path.stat(follow_symlinks=False)
        existing = _regular_file_bytes(path, "existing child state")
        if stat.S_IMODE(metadata.st_mode) != mode or existing != payload:
            raise ValueError("existing child state is not byte-identical")
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _validate_source_identity(
    value: Any, *, contract: TagResolutionContract
) -> Mapping[str, Any]:
    source = dict(_mapping(value, "tag-resolution source identity"))
    _exact_keys(
        source,
        {
            "tag_resolution_source_git_commit",
            "origin_main_git_commit",
            "branch",
            "origin_url",
            "source_path_count",
            "source_inventory_sha256",
            "parent_publication_source_git_commit",
            "parent_source_inventory_sha256",
            "failure_evidence_commit",
            "failure_evidence_sha256",
        },
        "tag-resolution source identity",
    )
    frozen_source = contract.data["git_source_contract"]
    frozen_parent = contract.parent
    if (
        _commit(source["tag_resolution_source_git_commit"], "child source commit")
        != _commit(source["origin_main_git_commit"], "origin/main commit")
        or source["branch"] != "main"
        or source["origin_url"]
        != "https://github.com/luojiaxuan/CausalCache.git"
        or type(source["source_path_count"]) is not int
        or source["source_path_count"]
        != len(frozen_source["required_source_paths"])
        or source["parent_publication_source_git_commit"]
        != frozen_parent["source_git_commit"]
        or source["parent_source_inventory_sha256"]
        != frozen_parent["parent_source_inventory_sha256"]
        or source["failure_evidence_commit"]
        != frozen_source["failure_evidence_commit"]
        or source["failure_evidence_sha256"]
        != frozen_source["failure_evidence_sha256"]
    ):
        raise ValueError("tag-resolution source differs from frozen clean-main lineage")
    _commit(source["parent_publication_source_git_commit"], "parent source")
    _sha256(source["parent_source_inventory_sha256"], "parent inventory")
    _commit(source["failure_evidence_commit"], "failure evidence commit")
    _sha256(source["failure_evidence_sha256"], "failure evidence SHA256")
    _sha256(source["source_inventory_sha256"], "child source inventory")
    return source


def _child_claim_payload(
    *,
    contract: TagResolutionContract,
    source_identity: Mapping[str, Any],
    parent_state: ParentPublicationState,
) -> Mapping[str, Any]:
    source = _validate_source_identity(source_identity, contract=contract)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": CLAIM_STATUS,
        "tag_resolution_config_sha256": contract.sha256,
        "tag_resolution_source": dict(source),
        "parent_publication": {
            "protocol_id": contract.parent["protocol_id"],
            "source_git_commit": contract.parent["source_git_commit"],
            "failure_evidence_commit": contract.parent[
                "failure_evidence_commit"
            ],
            "config_sha256": contract.parent["config_sha256"],
            "claim_sha256": parent_state.claim_sha256,
            "completion_seal_observed_absent": True,
        },
        "remote_evidence": dict(contract.remote),
        "remote_mutation_call_count": 0,
        "formal_consumption": dict(contract.data["formal_consumption"]),
    }


def _validated_fresh_parent(value: str | Path) -> Path:
    parent = Path(value)
    if not parent.is_absolute() or not parent.is_dir() or parent.is_symlink():
        raise ValueError("fresh-download parent must be absolute non-symlink directory")
    return parent


def _dataset_revision(api: Any, contract: TagResolutionContract, revision: str) -> str:
    remote = contract.remote
    info = api.dataset_info(remote["repo"], revision=revision)
    if info.private is not True:
        raise ValueError("destination dataset must remain private")
    return _commit(info.sha, f"remote revision {revision}")


def _tag_object_identity(api: Any, contract: TagResolutionContract) -> str:
    remote = contract.remote
    refs = api.list_repo_refs(remote["repo"], repo_type=remote["repo_type"])
    tags = [reference for reference in refs.tags if reference.name == remote["tag"]]
    if len(tags) != 1:
        raise ValueError("frozen annotated tag must exist exactly once")
    return _commit(tags[0].target_commit, "annotated tag object identity")


def _remote_identity_snapshot(api: Any, contract: TagResolutionContract) -> Mapping[str, Any]:
    remote = contract.remote
    snapshot = {
        "main_resolved_commit": _dataset_revision(api, contract, "main"),
        "tag_object_identity": _tag_object_identity(api, contract),
        "tag_resolved_commit": _dataset_revision(api, contract, remote["tag"]),
        "immutable_resolved_commit": _dataset_revision(
            api, contract, remote["pair_commit"]["revision"]
        ),
    }
    expected = {
        "main_resolved_commit": remote["main_resolved_commit"],
        "tag_object_identity": remote["tag_object_identity"],
        "tag_resolved_commit": remote["tag_resolved_commit"],
        "immutable_resolved_commit": remote["pair_commit"]["revision"],
    }
    if snapshot != expected:
        raise ValueError("remote main/tag-object/tag-resolved/immutable identity drifted")
    return snapshot


def _verify_pair_provenance(api: Any, contract: TagResolutionContract) -> Mapping[str, Any]:
    remote = contract.remote
    pair = remote["pair_commit"]
    commits = api.list_repo_commits(
        remote["repo"],
        repo_type=remote["repo_type"],
        revision=pair["revision"],
    )
    if len(commits) < 2:
        raise ValueError("pair commit lacks its frozen direct predecessor")
    if (
        _commit(commits[0].commit_id, "pair history head") != pair["revision"]
        or commits[0].title != pair["title"]
        or _commit(commits[1].commit_id, "pair direct predecessor")
        != pair["predecessor_commit"]
    ):
        raise ValueError("pair commit title or direct history drifted")
    pair_files = sorted(
        api.list_repo_files(
            remote["repo"],
            repo_type=remote["repo_type"],
            revision=pair["revision"],
        )
    )
    predecessor_files = sorted(
        api.list_repo_files(
            remote["repo"],
            repo_type=remote["repo_type"],
            revision=pair["predecessor_commit"],
        )
    )
    if pair_files != pair["files"] or predecessor_files != pair["predecessor_files"]:
        raise ValueError("pair or predecessor file inventory drifted")
    return {
        "pair_revision": pair["revision"],
        "pair_title": pair["title"],
        "predecessor_commit": pair["predecessor_commit"],
        "pair_files": pair_files,
        "predecessor_files": predecessor_files,
    }


def _assert_download_path(
    root: Path, returned: Path, relative_path: str, label: str
) -> None:
    if not root.is_absolute() or not returned.is_absolute():
        raise ValueError("HF download paths must be absolute")
    expected = root / relative_path
    if returned != expected:
        raise ValueError("HF download returned a lexically unexpected path")
    root_metadata = root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("fresh download root became a symlink or non-directory")
    relative = returned.relative_to(root)
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        metadata = current.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} has a symlink or non-directory component")
    final = returned.lstat()
    if not stat.S_ISREG(final.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")


def _validate_sidecar(
    payload: bytes,
    *,
    contract: TagResolutionContract,
    parent_state: ParentPublicationState,
) -> Mapping[str, Any]:
    sidecar = strict_pretty_json_object_bytes(payload, label="remote P1 sidecar")
    _exact_keys(
        sidecar,
        {
            "schema_version",
            "protocol_id",
            "status",
            "artifact_class",
            "destination",
            "source_archive",
            "publication_source",
            "formal_consumption",
        },
        "remote P1 sidecar",
    )
    source_archive = dict(_mapping(sidecar["source_archive"], "sidecar source archive"))
    _exact_keys(
        source_archive,
        {
            "p0_protocol_id",
            "p0_archive_status",
            "p0_config_path",
            "p0_config_sha256",
            "archive_sha256",
            "archive_size_bytes",
            "tree_inventory_sha256",
            "member_count",
        },
        "sidecar source archive",
    )
    expected_source = contract.p0_source
    expected_formal = {
        "formal_label_loader_eligible": False,
        "gate_training_unlocked": False,
        "producer_attempt_reclassified": False,
    }
    if (
        sidecar["schema_version"] != SCHEMA_VERSION
        or sidecar["protocol_id"] != contract.parent["protocol_id"]
        or sidecar["status"] != P1_SIDECAR_STATUS
        or sidecar["artifact_class"] != ARTIFACT_CLASS
        or sidecar["destination"] != _parent_destination(contract)
        or sidecar["publication_source"]
        != parent_state.claim["publication_source"]
        or sidecar["formal_consumption"] != expected_formal
        or source_archive
        != {
            "p0_protocol_id": expected_source["protocol_id"],
            "p0_archive_status": expected_source["required_archive_status"],
            "p0_config_path": expected_source["config_path"],
            "p0_config_sha256": expected_source["config_sha256"],
            "archive_sha256": expected_source["archive_sha256"],
            "archive_size_bytes": expected_source["archive_size_bytes"],
            "tree_inventory_sha256": expected_source["tree_inventory_sha256"],
            "member_count": expected_source["archive_member_count"],
        }
    ):
        raise ValueError("remote P1 sidecar semantic binding drifted")
    return sidecar


def _download_and_validate_pair(
    *,
    download_fn: Callable[..., str],
    contract: TagResolutionContract,
    p0_contract: InvalidForensicContract,
    parent_state: ParentPublicationState,
    directory: Path,
) -> Mapping[str, Any]:
    if not directory.is_absolute() or directory.is_symlink() or any(directory.iterdir()):
        raise ValueError("fresh download directory must start empty and absolute")
    remote = contract.remote
    archive = Path(
        download_fn(
            repo_id=remote["repo"],
            filename=remote["archive_path"],
            repo_type=remote["repo_type"],
            revision=remote["pair_commit"]["revision"],
            local_dir=directory,
            force_download=True,
        )
    )
    _assert_download_path(directory, archive, remote["archive_path"], "archive")
    sidecar = Path(
        download_fn(
            repo_id=remote["repo"],
            filename=remote["sidecar_path"],
            repo_type=remote["repo_type"],
            revision=remote["pair_commit"]["revision"],
            local_dir=directory,
            force_download=True,
        )
    )
    _assert_download_path(directory, archive, remote["archive_path"], "archive")
    _assert_download_path(directory, sidecar, remote["sidecar_path"], "sidecar")
    archive_bytes = _regular_file_bytes(archive, "downloaded archive")
    sidecar_bytes = _regular_file_bytes(sidecar, "downloaded sidecar")
    if (
        len(archive_bytes) != contract.p0_source["archive_size_bytes"]
        or sha256_bytes(archive_bytes) != remote["archive_sha256"]
        or sha256_bytes(sidecar_bytes) != remote["sidecar_sha256"]
    ):
        raise ValueError("downloaded archive or sidecar bytes drifted")
    _validate_sidecar(sidecar_bytes, contract=contract, parent_state=parent_state)
    evidence = read_invalid_forensic_archive(archive, contract=p0_contract)
    if (
        evidence.manifest["status"] != P0_ARCHIVE_STATUS
        or evidence.tree_inventory_sha256
        != contract.p0_source["tree_inventory_sha256"]
        or len(evidence.files) != contract.p0_source["archive_member_count"]
    ):
        raise ValueError("downloaded archive failed frozen P0 strict readback")
    return {
        "archive_sha256": sha256_bytes(archive_bytes),
        "archive_size_bytes": len(archive_bytes),
        "sidecar_sha256": sha256_bytes(sidecar_bytes),
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
        "member_count": len(evidence.files),
        "force_download": True,
        "download_directory_started_empty": True,
        "p0_strict_archive_readback_passed": True,
    }


def _completion_payload(
    *,
    contract: TagResolutionContract,
    claim_sha256: str,
    source_identity: Mapping[str, Any],
    parent_state: ParentPublicationState,
    attestation: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": COMPLETION_STATUS,
        "tag_resolution_config_sha256": contract.sha256,
        "claim_sha256": _sha256(claim_sha256, "child claim SHA256"),
        "tag_resolution_source": dict(
            _validate_source_identity(source_identity, contract=contract)
        ),
        "parent_publication": {
            "claim_sha256": parent_state.claim_sha256,
            "completion_seal_observed_absent_pre_post": True,
        },
        "remote_attestation": dict(attestation),
        "remote_mutation_call_count": 0,
        "formal_consumption": dict(contract.data["formal_consumption"]),
    }


def validate_remote_tag_resolution(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: TagResolutionContract,
    p0_contract: InvalidForensicContract,
    source_identity: Mapping[str, Any],
    fresh_download_parent: str | Path,
) -> Mapping[str, Any]:
    if (
        p0_contract.sha256 != contract.p0_source["config_sha256"]
        or p0_contract.data["protocol_id"] != contract.p0_source["protocol_id"]
    ):
        raise ValueError("P0 contract differs from tag-resolution contract")
    parent = _validated_fresh_parent(fresh_download_parent)
    parent_state_pre = validate_parent_publication_state(contract)
    claim = _child_claim_payload(
        contract=contract,
        source_identity=source_identity,
        parent_state=parent_state_pre,
    )
    claim_bytes = pretty_json_bytes(claim)
    claim_sha = sha256_bytes(claim_bytes)
    child_claim_path = Path(contract.child_state["claim_path"])
    child_completion_path = Path(contract.child_state["completion_seal_path"])
    completion_present = child_completion_path.exists() or child_completion_path.is_symlink()
    claim_present = child_claim_path.exists() or child_claim_path.is_symlink()
    if completion_present and not claim_present:
        raise ValueError("orphan child completion exists without its claim")
    _exclusive_or_identical_state(
        child_claim_path, claim_bytes, contract.child_state["file_mode"]
    )

    identity_pre = _remote_identity_snapshot(api, contract)
    provenance_pre = _verify_pair_provenance(api, contract)
    with tempfile.TemporaryDirectory(
        prefix="invalid-forensic-tag-resolution-",
        dir=parent,
    ) as directory:
        download = _download_and_validate_pair(
            download_fn=download_fn,
            contract=contract,
            p0_contract=p0_contract,
            parent_state=parent_state_pre,
            directory=Path(directory),
        )
    identity_post = _remote_identity_snapshot(api, contract)
    provenance_post = _verify_pair_provenance(api, contract)
    if identity_pre != identity_post or provenance_pre != provenance_post:
        raise ValueError("remote identity or pair provenance drifted during replay")
    parent_state_post = validate_parent_publication_state(contract)
    if (
        parent_state_pre.claim_bytes != parent_state_post.claim_bytes
        or parent_state_pre.claim_sha256 != parent_state_post.claim_sha256
    ):
        raise ValueError("parent P1 claim changed during read-only replay")
    attestation = {
        "identity_pre": dict(identity_pre),
        "identity_post": dict(identity_post),
        "pair_provenance_pre": dict(provenance_pre),
        "pair_provenance_post": dict(provenance_post),
        "fresh_download": dict(download),
        "private_repo_verified": True,
        "parent_p1_completion_absent_pre_post": True,
    }
    completion = _completion_payload(
        contract=contract,
        claim_sha256=claim_sha,
        source_identity=source_identity,
        parent_state=parent_state_post,
        attestation=attestation,
    )
    completion_bytes = pretty_json_bytes(completion)
    _exclusive_or_identical_state(
        child_completion_path,
        completion_bytes,
        contract.child_state["file_mode"],
    )
    return completion


__all__ = [
    "ALLOWED_REMOTE_API_METHODS",
    "CANONICAL_CONFIG_PATH",
    "CLAIM_STATUS",
    "COMPLETION_STATUS",
    "FROZEN_CONFIG_SHA256",
    "PROTOCOL_ID",
    "SOURCE_STATUS",
    "ParentPublicationState",
    "TagResolutionContract",
    "load_frozen_tag_resolution_contract",
    "tag_resolution_contract_from_data",
    "validate_clean_pushed_source",
    "validate_parent_publication_state",
    "validate_remote_tag_resolution",
    "validate_tag_resolution_contract_data",
]
