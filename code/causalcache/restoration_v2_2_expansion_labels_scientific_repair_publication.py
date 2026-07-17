"""Crash-recoverable publication of the completed repaired label payload."""

from __future__ import annotations

import io
import json
import os
import re
import stat
import subprocess
import tarfile
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = (
    "causalcache_restoration_v2_2_expansion_exact_labels_"
    "scientific_repair_publication_v1"
)
SOURCE_STATUS = "source_only_frozen_before_repaired_private_hf_publication"
SIDECAR_STATUS = "IMMUTABLE_REPAIRED_EXPANSION_LABELS_PUBLICATION_MANIFEST_V1"
CLAIM_STATUS = "CLAIMED_REPAIRED_EXPANSION_LABELS_PUBLICATION_V1"
REMOTE_BASE_RECEIPT_STATUS = "SEALED_REPAIRED_LABEL_PUBLICATION_REMOTE_BASE_V1"
COMPLETION_STATUS = "COMPLETED_REPAIRED_EXPANSION_LABELS_PUBLICATION_V1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_"
    "expansion_labels_scientific_repair_publication_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "547b291022e2430fe6af5158d9350132f405c4b6019198488386dd7fb50be0aa"
)
PAIR_COMMIT_TITLE = "Publish repaired expansion exact-label scientific payload"
TAG_MESSAGE = "Freeze repaired expansion exact-label scientific payload"
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
        "repaired-mobile"
    ),
    "repo_type": "dataset",
    "private": True,
    "tag": "v2.2-expansion-exact-labels-scientific-repair-v1",
    "archive_path": (
        "repaired/v2.2-expansion-exact-labels-scientific-repair-v1/"
        "repaired-labels-v1.tar"
    ),
    "sidecar_path": (
        "repaired/v2.2-expansion-exact-labels-scientific-repair-v1/"
        "artifact-manifest-v1.json"
    ),
    "archive_sidecar_same_commit_required": True,
    "annotated_tag_required": True,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ScientificRepairPublicationContract:
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path | None = None
    source_path: Path | None = None

    @property
    def destination(self) -> Mapping[str, Any]:
        return _mapping(self.data["destination"], "destination")

    @property
    def source(self) -> Mapping[str, Any]:
        return _mapping(self.data["scientific_repair_source"], "repair source")

    @property
    def local_state(self) -> Mapping[str, Any]:
        return _mapping(self.data["local_state"], "local state")


@dataclass(frozen=True)
class PreparedScientificRepairPublication:
    archive_path: Path
    archive_bytes: bytes
    archive_sha256: str
    sidecar: Mapping[str, Any]
    sidecar_bytes: bytes
    sidecar_sha256: str
    source_identity: Mapping[str, Any]
    producer_claim_sha256: str
    producer_completion_sha256: str


@dataclass(frozen=True)
class TagSnapshot:
    object_identity: str
    resolved_commit: str


@dataclass(frozen=True)
class RemoteInspection:
    state: str
    main_revision: str
    immutable_revision: str | None
    tag_object_identity: str | None


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} schema drifted")


def _sha(value: Any, label: str) -> str:
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


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def strict_pretty_json_object_bytes(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    result = dict(_mapping(value, label))
    if pretty_json_bytes(result) != payload:
        raise ValueError(f"{label} is not canonical pretty JSON")
    return result


def _regular_file_bytes(
    path: str | Path,
    label: str,
    *,
    mode: int | None = None,
) -> bytes:
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
        if mode is not None and stat.S_IMODE(before.st_mode) != mode:
            raise ValueError(f"{label} mode drifted")
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


def validate_publication_contract_data(value: Any) -> Mapping[str, Any]:
    data = dict(_mapping(value, "scientific-repair publication contract"))
    _exact_keys(
        data,
        {
            "schema_version",
            "protocol_id",
            "status",
            "destination",
            "scientific_repair_source",
            "git_source_contract",
            "publication_contract",
            "local_state",
            "fresh_download_contract",
            "formal_consumption",
        },
        "scientific-repair publication contract",
    )
    if (
        data["schema_version"] != SCHEMA_VERSION
        or data["protocol_id"] != PROTOCOL_ID
        or data["status"] != SOURCE_STATUS
    ):
        raise ValueError("publication contract identity drifted")
    if dict(_mapping(data["destination"], "destination")) != DESTINATION:
        raise ValueError("repaired private-HF destination drifted")
    _safe_relative_path(DESTINATION["archive_path"], "archive destination")
    _safe_relative_path(DESTINATION["sidecar_path"], "sidecar destination")

    source = dict(_mapping(data["scientific_repair_source"], "repair source"))
    _exact_keys(
        source,
        {
            "protocol_id",
            "source_git_commit",
            "result_commit",
            "runner_config",
            "result_summary",
            "archive",
            "claim",
            "completion",
            "formal_audit_status",
            "manifest_status",
        },
        "repair source",
    )
    if source["protocol_id"] != (
        "causalcache_restoration_v2_2_expansion_exact_labels_"
        "scientific_repair_runner_v1"
    ):
        raise ValueError("repair producer protocol drifted")
    _commit(source["source_git_commit"], "repair source commit")
    _commit(source["result_commit"], "repair result commit")
    runner = dict(_mapping(source["runner_config"], "runner config"))
    _exact_keys(runner, {"path", "sha256"}, "runner config")
    _safe_relative_path(runner["path"], "runner config path")
    _sha(runner["sha256"], "runner config SHA256")
    summary = dict(_mapping(source["result_summary"], "result summary"))
    _exact_keys(summary, {"path", "sha256", "size_bytes"}, "result summary")
    _safe_relative_path(summary["path"], "result summary path")
    _sha(summary["sha256"], "result summary SHA256")
    _positive_int(summary["size_bytes"], "result summary size")

    archive = dict(_mapping(source["archive"], "repair archive"))
    _exact_keys(
        archive,
        {
            "path",
            "sha256",
            "size_bytes",
            "member_count",
            "member_inventory",
            "tree_inventory_sha256",
            "archive_member_prefix",
        },
        "repair archive",
    )
    _absolute_path(archive["path"], "repair archive path")
    _sha(archive["sha256"], "repair archive SHA256")
    _positive_int(archive["size_bytes"], "repair archive size")
    _sha(archive["tree_inventory_sha256"], "repair tree SHA256")
    _safe_relative_path(archive["archive_member_prefix"], "archive prefix")
    inventory = archive["member_inventory"]
    if (
        not isinstance(inventory, list)
        or len(inventory) != archive["member_count"]
        or len(inventory) != 4
    ):
        raise ValueError("repair archive must bind exactly four members")
    normalized_inventory: list[Mapping[str, Any]] = []
    for record in inventory:
        item = dict(_mapping(record, "archive member"))
        _exact_keys(item, {"path", "sha256", "size_bytes"}, "archive member")
        _safe_relative_path(item["path"], "archive member path")
        _sha(item["sha256"], "archive member SHA256")
        _positive_int(item["size_bytes"], "archive member size")
        normalized_inventory.append(item)
    if [item["path"] for item in normalized_inventory] != sorted(
        item["path"] for item in normalized_inventory
    ) or {item["path"] for item in normalized_inventory} != {
        "audit.json",
        "derived_labels.jsonl",
        "manifest.json",
        "raw_states.jsonl",
    }:
        raise ValueError("repair archive inventory drifted")
    if sha256_bytes(canonical_json_bytes(normalized_inventory)) != archive[
        "tree_inventory_sha256"
    ]:
        raise ValueError("repair archive tree inventory SHA256 drifted")

    for label in ("claim", "completion"):
        state = dict(_mapping(source[label], f"producer {label}"))
        _exact_keys(
            state,
            {"path", "sha256", "size_bytes", "mode", "status"},
            f"producer {label}",
        )
        _absolute_path(state["path"], f"producer {label} path")
        _sha(state["sha256"], f"producer {label} SHA256")
        _positive_int(state["size_bytes"], f"producer {label} size")
        if state["mode"] != 0o600 or not isinstance(state["status"], str):
            raise ValueError(f"producer {label} mode or status drifted")
    if source["claim"]["path"] == source["completion"]["path"]:
        raise ValueError("producer claim and completion paths must differ")
    if source["formal_audit_status"] != (
        "VALIDATED_EXPANSION_LABEL_SCIENTIFIC_REPAIR_AUDIT_V1"
    ) or source["manifest_status"] != (
        "IMMUTABLE_REPAIRED_EXPANSION_LABEL_PAYLOAD_MANIFEST_V1"
    ):
        raise ValueError("formal repair payload statuses drifted")

    git = dict(_mapping(data["git_source_contract"], "Git source contract"))
    _exact_keys(
        git,
        {
            "branch",
            "origin_name",
            "origin_url",
            "head_must_equal_origin_main",
            "worktree_must_be_clean",
            "real_ls_remote_main_required",
            "required_source_paths",
        },
        "Git source contract",
    )
    if (
        git["branch"] != "main"
        or git["origin_name"] != "origin"
        or git["origin_url"] != "https://github.com/luojiaxuan/CausalCache.git"
        or git["head_must_equal_origin_main"] is not True
        or git["worktree_must_be_clean"] is not True
        or git["real_ls_remote_main_required"] is not True
    ):
        raise ValueError("Git source policy drifted")
    paths = git["required_source_paths"]
    if not isinstance(paths, list) or len(paths) != 6 or len(set(paths)) != 6:
        raise ValueError("publication source inventory must contain six paths")
    for relative in paths:
        _safe_relative_path(relative, "publication source path")
    if CANONICAL_CONFIG_PATH not in paths or summary["path"] not in paths:
        raise ValueError("publication config or result summary missing from source")

    publication = dict(_mapping(data["publication_contract"], "publication"))
    if publication != {
        "allowed_remote_states": list(ALLOWED_REMOTE_STATES),
        "commit_title": PAIR_COMMIT_TITLE,
        "completion_is_last_local_mutation": True,
        "conflict_action": "FAIL_CLOSED",
        "exact_pair_diff_required": True,
        "no_overwrite": True,
        "remote_base_receipt_required_before_content_mutation": True,
        "remote_history_comparison_is_set_based": True,
        "remote_recursive_tree_inventory_required": True,
        "response_loss_requery_required": True,
        "sidecar_status": SIDECAR_STATUS,
    }:
        raise ValueError("publication recovery contract drifted")
    local = dict(_mapping(data["local_state"], "local state"))
    _exact_keys(
        local,
        {
            "claim_path",
            "remote_base_receipt_path",
            "completion_staging_path",
            "completion_seal_path",
            "file_mode",
            "o_excl_first_write",
            "retained_completion_staging_required",
            "same_bytes_reusable",
        },
        "local state",
    )
    claim_path = _absolute_path(local["claim_path"], "publication claim")
    base_receipt_path = _absolute_path(
        local["remote_base_receipt_path"], "remote base receipt"
    )
    completion_staging_path = _absolute_path(
        local["completion_staging_path"], "retained completion staging"
    )
    completion_path = _absolute_path(
        local["completion_seal_path"], "publication completion"
    )
    if (
        local["file_mode"] != 0o600
        or local["o_excl_first_write"] is not True
        or local["retained_completion_staging_required"] is not True
        or local["same_bytes_reusable"] is not True
        or len(
            {
                claim_path,
                base_receipt_path,
                completion_staging_path,
                completion_path,
            }
        )
        != 4
    ):
        raise ValueError("local publication state drifted")
    if dict(_mapping(data["fresh_download_contract"], "fresh download")) != {
        "download_directory_must_start_empty": True,
        "force_download": True,
        "immutable_revision_pre_post_stable": True,
        "main_revision_pre_post_stable": True,
        "strict_repaired_transport_readback_required": True,
        "tag_object_and_resolved_commit_pre_post_stable": True,
    }:
        raise ValueError("fresh-download contract drifted")
    if dict(_mapping(data["formal_consumption"], "formal consumption")) != {
        "gate_training_unlocked_before_publication_completion": False,
        "gate_training_unlocked_by_sidecar_alone": False,
        "original_invalid_forensic_formal_label_loader_eligible": False,
        "producer_attempt_reclassified": False,
        "repaired_payload_formal_label_loader_eligible_after_publication_completion": True,
    }:
        raise ValueError("formal-consumption boundary drifted")
    return data


def publication_contract_from_data(value: Any) -> ScientificRepairPublicationContract:
    data = validate_publication_contract_data(value)
    return ScientificRepairPublicationContract(
        data=data,
        sha256=sha256_bytes(pretty_json_bytes(data)),
    )


def load_frozen_publication_contract(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> ScientificRepairPublicationContract:
    root = Path(repository_root).resolve()
    supplied = Path(path)
    if not supplied.is_absolute():
        supplied = root / supplied
    payload = _regular_file_bytes(supplied, "publication config")
    if sha256_bytes(payload) != FROZEN_CONFIG_SHA256:
        raise ValueError("frozen publication config SHA256 drifted")
    data = strict_pretty_json_object_bytes(payload, label="publication config")
    validate_publication_contract_data(data)
    return ScientificRepairPublicationContract(
        data=data,
        sha256=FROZEN_CONFIG_SHA256,
        repository_root=root,
        source_path=supplied.resolve(),
    )


def _git_bytes(root: Path, arguments: Sequence[str]) -> bytes:
    process = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise ValueError(f"git {' '.join(arguments)} failed")
    return process.stdout


def _remote_main_head(root: Path, remote_name: str) -> str:
    process = subprocess.run(
        ["git", "ls-remote", "--exit-code", remote_name, "refs/heads/main"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise ValueError("canonical remote main could not be resolved")
    try:
        lines = process.stdout.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("canonical remote main response is not ASCII") from error
    suffix = "\trefs/heads/main"
    if len(lines) != 1 or not lines[0].endswith(suffix):
        raise ValueError("canonical remote main response schema drifted")
    return _commit(lines[0][: -len(suffix)], "canonical remote main")


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    process = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise ValueError(f"required Git ancestor is absent: {ancestor}")


def validate_clean_pushed_source(
    *,
    repository_root: str | Path,
    contract: ScientificRepairPublicationContract,
) -> Mapping[str, Any]:
    supplied = Path(repository_root)
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ValueError("repository root must be absolute and non-symlink")
    root = supplied.resolve()
    if not root.is_dir():
        raise ValueError("repository root must be a directory")
    top = Path(_git_bytes(root, ["rev-parse", "--show-toplevel"]).decode().strip())
    if top.resolve() != root:
        raise ValueError("repository root is not the Git toplevel")
    git = contract.data["git_source_contract"]
    head = _git_bytes(root, ["rev-parse", "HEAD"]).decode().strip()
    origin_main = _git_bytes(root, ["rev-parse", "origin/main"]).decode().strip()
    branch = _git_bytes(root, ["branch", "--show-current"]).decode().strip()
    origin_url = _git_bytes(root, ["remote", "get-url", "origin"]).decode().strip()
    status = _git_bytes(root, ["status", "--porcelain=v1", "--untracked-files=all"])
    remote_main = _remote_main_head(root, git["origin_name"])
    if (
        head != origin_main
        or head != remote_main
        or branch != git["branch"]
        or origin_url != git["origin_url"]
        or status
    ):
        raise ValueError("publication source must be clean pushed canonical main")
    _commit(head, "publication source commit")
    source = contract.source
    _require_ancestor(root, source["source_git_commit"], source["result_commit"])
    _require_ancestor(root, source["result_commit"], head)
    summary_relative = source["result_summary"]["path"]
    result_summary = _git_bytes(
        root, ["show", f"{source['result_commit']}:{summary_relative}"]
    )
    if (
        len(result_summary) != source["result_summary"]["size_bytes"]
        or sha256_bytes(result_summary) != source["result_summary"]["sha256"]
    ):
        raise ValueError("frozen result summary differs at result commit")
    runner_relative = source["runner_config"]["path"]
    runner_config = _git_bytes(
        root, ["show", f"{source['source_git_commit']}:{runner_relative}"]
    )
    if sha256_bytes(runner_config) != source["runner_config"]["sha256"]:
        raise ValueError("formal runner config differs at repair source commit")
    inventory: list[Mapping[str, Any]] = []
    for relative in git["required_source_paths"]:
        local = _regular_file_bytes(root / relative, f"source {relative}")
        committed = _git_bytes(root, ["show", f"{head}:{relative}"])
        if local != committed:
            raise ValueError(f"source file differs from pushed HEAD: {relative}")
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
        "externally_resolved_remote_main_git_commit": remote_main,
        "branch": branch,
        "origin_url": origin_url,
        "source_path_count": len(inventory),
        "source_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "repair_source_git_commit": source["source_git_commit"],
        "repair_result_git_commit": source["result_commit"],
        "repair_result_summary_sha256": source["result_summary"]["sha256"],
    }


def _validate_source_identity(value: Any) -> Mapping[str, Any]:
    source = dict(_mapping(value, "publication source identity"))
    _exact_keys(
        source,
        {
            "publication_source_git_commit",
            "origin_main_git_commit",
            "externally_resolved_remote_main_git_commit",
            "branch",
            "origin_url",
            "source_path_count",
            "source_inventory_sha256",
            "repair_source_git_commit",
            "repair_result_git_commit",
            "repair_result_summary_sha256",
        },
        "publication source identity",
    )
    commits = {
        _commit(source[key], key)
        for key in (
            "publication_source_git_commit",
            "origin_main_git_commit",
            "externally_resolved_remote_main_git_commit",
        )
    }
    if (
        len(commits) != 1
        or source["branch"] != "main"
        or source["origin_url"] != "https://github.com/luojiaxuan/CausalCache.git"
        or type(source["source_path_count"]) is not int
        or source["source_path_count"] <= 0
    ):
        raise ValueError("publication source identity is not clean pushed main")
    _sha(source["source_inventory_sha256"], "source inventory SHA256")
    _commit(source["repair_source_git_commit"], "repair source commit")
    _commit(source["repair_result_git_commit"], "repair result commit")
    _sha(source["repair_result_summary_sha256"], "repair result summary SHA256")
    return source


def _strict_transport_readback(
    payload: bytes,
    *,
    contract: ScientificRepairPublicationContract,
) -> Mapping[str, bytes]:
    archive_contract = contract.source["archive"]
    if (
        len(payload) != archive_contract["size_bytes"]
        or sha256_bytes(payload) != archive_contract["sha256"]
    ):
        raise ValueError("repaired archive bytes differ from frozen producer output")
    prefix = f"{archive_contract['archive_member_prefix']}/"
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != sorted(
                member.name for member in members
            ):
                raise ValueError("repaired USTAR member order drifted")
            for member in members:
                if (
                    not member.isfile()
                    or not member.name.startswith(prefix)
                    or member.mode != 0o644
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or member.pax_headers
                ):
                    raise ValueError("repaired USTAR metadata drifted")
                relative = _safe_relative_path(
                    member.name[len(prefix) :], "repaired USTAR member"
                )
                if relative in files:
                    raise ValueError("repaired USTAR contains duplicate members")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("repaired USTAR member is unreadable")
                content = extracted.read()
                if len(content) != member.size:
                    raise ValueError("repaired USTAR member size drifted")
                files[relative] = content
    except tarfile.TarError as error:
        raise ValueError("repaired archive is not readable USTAR") from error
    inventory = [
        {
            "path": name,
            "sha256": sha256_bytes(files[name]),
            "size_bytes": len(files[name]),
        }
        for name in sorted(files)
    ]
    if (
        inventory != archive_contract["member_inventory"]
        or sha256_bytes(canonical_json_bytes(inventory))
        != archive_contract["tree_inventory_sha256"]
    ):
        raise ValueError("repaired archive member inventory drifted")
    manifest = strict_pretty_json_object_bytes(
        files["manifest.json"], label="repaired manifest"
    )
    audit = strict_pretty_json_object_bytes(files["audit.json"], label="repair audit")
    formal = contract.data["formal_consumption"]
    if (
        manifest.get("status") != contract.source["manifest_status"]
        or manifest.get("runner_config_sha256")
        != contract.source["runner_config"]["sha256"]
        or manifest.get("source_git_commit") != contract.source["source_git_commit"]
        or manifest.get("original_attempt_remains_invalid") is not True
        or manifest.get("producer_reclassified") is not False
        or audit.get("status") != contract.source["formal_audit_status"]
        or audit.get("formal_scientific_repair_pass") is not True
        or audit.get("runner_config_sha256")
        != contract.source["runner_config"]["sha256"]
        or audit.get("source", {}).get("git_commit")
        != contract.source["source_git_commit"]
        or audit.get("original_invalid_attempt", {}).get("producer_reclassified")
        is not False
        or audit.get("formal_consumption", {}).get("gate_training_unlocked")
        is not False
        or formal["gate_training_unlocked_before_publication_completion"]
        is not False
    ):
        raise ValueError("repaired archive formal transport binding drifted")
    return files


def _validate_result_summary(
    payload: bytes,
    *,
    contract: ScientificRepairPublicationContract,
) -> Mapping[str, Any]:
    source = contract.source
    summary_contract = source["result_summary"]
    if (
        len(payload) != summary_contract["size_bytes"]
        or sha256_bytes(payload) != summary_contract["sha256"]
    ):
        raise ValueError("repair result summary bytes drifted")
    summary = strict_pretty_json_object_bytes(payload, label="repair result summary")
    archive = summary.get("archive", {})
    state = summary.get("state_files", {})
    if (
        summary.get("formal_scientific_repair_pass") is not True
        or summary.get("source", {}).get("git_commit") != source["source_git_commit"]
        or summary.get("source", {}).get("runner_config_sha256")
        != source["runner_config"]["sha256"]
        or archive.get("sha256") != source["archive"]["sha256"]
        or archive.get("size_bytes") != source["archive"]["size_bytes"]
        or archive.get("tree_inventory_sha256")
        != source["archive"]["tree_inventory_sha256"]
        or state.get("claim", {}).get("sha256") != source["claim"]["sha256"]
        or state.get("completion", {}).get("sha256")
        != source["completion"]["sha256"]
        or summary.get("original_invalid_attempt", {}).get("producer_reclassified")
        is not False
        or summary.get("formal_consumption", {}).get("gate_training_unlocked")
        is not False
    ):
        raise ValueError("repair result summary semantic binding drifted")
    return summary


def _validate_producer_state(
    *,
    contract: ScientificRepairPublicationContract,
    archive_sha256: str,
) -> tuple[bytes, bytes]:
    source = contract.source
    claim_contract = source["claim"]
    completion_contract = source["completion"]
    claim = _regular_file_bytes(
        claim_contract["path"], "producer claim", mode=claim_contract["mode"]
    )
    completion = _regular_file_bytes(
        completion_contract["path"],
        "producer completion",
        mode=completion_contract["mode"],
    )
    if (
        len(claim) != claim_contract["size_bytes"]
        or sha256_bytes(claim) != claim_contract["sha256"]
        or len(completion) != completion_contract["size_bytes"]
        or sha256_bytes(completion) != completion_contract["sha256"]
    ):
        raise ValueError("producer claim or completion bytes drifted")
    claim_json = strict_pretty_json_object_bytes(claim, label="producer claim")
    completion_json = strict_pretty_json_object_bytes(
        completion, label="producer completion"
    )
    if (
        claim_json.get("status") != claim_contract["status"]
        or completion_json.get("status") != completion_contract["status"]
        or completion_json.get("runner_config_sha256")
        != source["runner_config"]["sha256"]
        or completion_json.get("claim_sha256") != claim_contract["sha256"]
        or completion_json.get("source_git_commit") != source["source_git_commit"]
        or completion_json.get("archive_sha256") != archive_sha256
        or completion_json.get("archive_size_bytes") != source["archive"]["size_bytes"]
        or completion_json.get("member_count") != source["archive"]["member_count"]
        or completion_json.get("member_inventory")
        != source["archive"]["member_inventory"]
        or completion_json.get("tree_inventory_sha256")
        != source["archive"]["tree_inventory_sha256"]
        or completion_json.get("completion_created_after_strict_output_readback")
        is not True
        or completion_json.get("original_attempt_reclassified") is not False
        or completion_json.get("formal_consumption", {}).get(
            "gate_training_unlocked"
        )
        is not False
    ):
        raise ValueError("producer completion semantic binding drifted")
    return claim, completion


def prepare_scientific_repair_publication(
    *,
    contract: ScientificRepairPublicationContract,
    source_identity: Mapping[str, Any],
) -> PreparedScientificRepairPublication:
    source = _validate_source_identity(source_identity)
    if (
        source["repair_source_git_commit"] != contract.source["source_git_commit"]
        or source["repair_result_git_commit"] != contract.source["result_commit"]
        or source["repair_result_summary_sha256"]
        != contract.source["result_summary"]["sha256"]
    ):
        raise ValueError("publication source differs from frozen repair lineage")
    archive_path = Path(contract.source["archive"]["path"])
    archive = _regular_file_bytes(archive_path, "completed repaired archive")
    _strict_transport_readback(archive, contract=contract)
    claim, completion = _validate_producer_state(
        contract=contract, archive_sha256=sha256_bytes(archive)
    )
    if contract.repository_root is None:
        root = Path.cwd()
    else:
        root = contract.repository_root
    summary_bytes = _regular_file_bytes(
        root / contract.source["result_summary"]["path"], "repair result summary"
    )
    _validate_result_summary(summary_bytes, contract=contract)
    if (
        _regular_file_bytes(archive_path, "completed repaired archive") != archive
        or _regular_file_bytes(
            contract.source["claim"]["path"],
            "producer claim",
            mode=contract.source["claim"]["mode"],
        )
        != claim
        or _regular_file_bytes(
            contract.source["completion"]["path"],
            "producer completion",
            mode=contract.source["completion"]["mode"],
        )
        != completion
        or _regular_file_bytes(
            root / contract.source["result_summary"]["path"],
            "repair result summary",
        )
        != summary_bytes
    ):
        raise ValueError("completed repair source changed during preparation")
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": SIDECAR_STATUS,
        "artifact_class": "repaired_expansion_exact_labels",
        "destination": dict(contract.destination),
        "scientific_repair": {
            "producer_protocol_id": contract.source["protocol_id"],
            "producer_source_git_commit": contract.source["source_git_commit"],
            "producer_result_git_commit": contract.source["result_commit"],
            "runner_config_sha256": contract.source["runner_config"]["sha256"],
            "result_summary_sha256": contract.source["result_summary"]["sha256"],
            "producer_claim_sha256": contract.source["claim"]["sha256"],
            "producer_completion_sha256": contract.source["completion"]["sha256"],
            "archive_sha256": contract.source["archive"]["sha256"],
            "archive_size_bytes": contract.source["archive"]["size_bytes"],
            "member_count": contract.source["archive"]["member_count"],
            "member_inventory": list(contract.source["archive"]["member_inventory"]),
            "tree_inventory_sha256": contract.source["archive"][
                "tree_inventory_sha256"
            ],
            "formal_scientific_repair_pass": True,
            "original_attempt_remains_invalid": True,
            "producer_attempt_reclassified": False,
        },
        "publication_source": {
            "publication_config_sha256": contract.sha256,
            **source,
        },
        "formal_consumption": {
            "formal_label_loader_eligible_by_sidecar_alone": False,
            "gate_training_unlocked_by_sidecar_alone": False,
            "publication_completion_required": True,
            "producer_attempt_reclassified": False,
        },
    }
    sidecar_bytes = pretty_json_bytes(sidecar)
    return PreparedScientificRepairPublication(
        archive_path=archive_path,
        archive_bytes=archive,
        archive_sha256=sha256_bytes(archive),
        sidecar=sidecar,
        sidecar_bytes=sidecar_bytes,
        sidecar_sha256=sha256_bytes(sidecar_bytes),
        source_identity=source,
        producer_claim_sha256=sha256_bytes(claim),
        producer_completion_sha256=sha256_bytes(completion),
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
        existing = _regular_file_bytes(path, "existing publication state", mode=mode)
        if existing != payload:
            raise ValueError("existing publication state is not byte-identical")
        return False
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("publication state write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        # note (luojiaxuan): Durable staging plus a hard-link no-replace makes
        # each pre-completion state visible atomically without overwrite.
        os.link(temporary, path)
        _fsync_directory(path.parent)
        return True
    except FileExistsError:
        existing = _regular_file_bytes(path, "raced publication state", mode=mode)
        if existing != payload:
            raise ValueError("concurrent publication state is not byte-identical")
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _reachable_history(
    api: Any,
    contract: ScientificRepairPublicationContract,
    revision: str,
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    destination = contract.destination
    commits = list(
        api.list_repo_commits(
            destination["repo"],
            repo_type=destination["repo_type"],
            revision=revision,
        )
    )
    identifiers: list[str] = []
    titles: dict[str, str] = {}
    for record in commits:
        identifier = _commit(record.commit_id, "reachable history commit")
        title = getattr(record, "title", None)
        if not isinstance(title, str):
            raise ValueError("reachable history commit title is missing")
        if identifier in titles:
            raise ValueError("reachable history contains a duplicate commit")
        identifiers.append(identifier)
        titles[identifier] = title
    if revision not in identifiers or not identifiers:
        raise ValueError("reachable history does not contain its requested revision")
    return tuple(sorted(identifiers)), titles


def _lfs_identity(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        size = value.get("size")
        digest = value.get("sha256")
        pointer_size = value.get("pointer_size")
    else:
        size = getattr(value, "size", None)
        digest = getattr(value, "sha256", None)
        pointer_size = getattr(value, "pointer_size", None)
    if (
        type(size) is not int
        or size < 0
        or not isinstance(digest, str)
        or not digest
        or type(pointer_size) is not int
        or pointer_size < 0
    ):
        raise ValueError("remote LFS identity is incomplete")
    return {
        "size": size,
        "sha256": digest,
        "pointer_size": pointer_size,
    }


def _recursive_tree_inventory(
    api: Any,
    contract: ScientificRepairPublicationContract,
    revision: str,
) -> tuple[Mapping[str, Any], ...]:
    destination = contract.destination
    entries = list(
        api.list_repo_tree(
            destination["repo"],
            path_in_repo=None,
            recursive=True,
            expand=True,
            revision=revision,
            repo_type=destination["repo_type"],
        )
    )
    inventory: list[Mapping[str, Any]] = []
    observed_paths: set[str] = set()
    for entry in entries:
        path = _safe_relative_path(getattr(entry, "path", None), "remote tree path")
        if path in observed_paths:
            raise ValueError("remote recursive tree contains a duplicate path")
        observed_paths.add(path)
        blob_id = getattr(entry, "blob_id", None)
        tree_id = getattr(entry, "tree_id", None)
        if isinstance(blob_id, str) and blob_id:
            entry_type = "file"
            identity = blob_id
            size = getattr(entry, "size", None)
            if type(size) is not int or size < 0:
                raise ValueError("remote blob size is invalid")
            lfs = _lfs_identity(getattr(entry, "lfs", None))
            xet_hash = getattr(entry, "xet_hash", None)
            if xet_hash is not None and (
                not isinstance(xet_hash, str) or not xet_hash
            ):
                raise ValueError("remote Xet identity is invalid")
        elif isinstance(tree_id, str) and tree_id:
            entry_type = "directory"
            identity = tree_id
            size = None
            lfs = None
            xet_hash = None
        else:
            raise ValueError("remote tree entry lacks a blob-or-tree identity")
        inventory.append(
            {
                "path": path,
                "type": entry_type,
                "identity": identity,
                "size_bytes": size,
                "lfs_identity": lfs,
                "xet_hash": xet_hash,
            }
        )
    return tuple(sorted(inventory, key=lambda record: (record["path"], record["type"])))


def _blob_inventory(
    inventory: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(record for record in inventory if record["type"] == "file")


def _remote_base_receipt_payload(
    *,
    contract: ScientificRepairPublicationContract,
    claim_sha256: str,
    base_main_revision: str,
    history_commit_ids: Sequence[str],
    tree_inventory: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    history = list(history_commit_ids)
    tree = [dict(record) for record in tree_inventory]
    blobs = [dict(record) for record in _blob_inventory(tree_inventory)]
    target_paths = {
        contract.destination["archive_path"],
        contract.destination["sidecar_path"],
    }
    if target_paths & {record["path"] for record in blobs}:
        raise ValueError("publication targets already exist in the remote base")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": REMOTE_BASE_RECEIPT_STATUS,
        "destination": dict(contract.destination),
        "claim_sha256": _sha(claim_sha256, "publication claim SHA256"),
        "base_main_revision": _commit(base_main_revision, "remote base revision"),
        "base_reachable_history_commit_ids": history,
        "base_reachable_history_sha256": sha256_bytes(canonical_json_bytes(history)),
        "base_recursive_tree_inventory": tree,
        "base_recursive_tree_inventory_sha256": sha256_bytes(
            canonical_json_bytes(tree)
        ),
        "base_recursive_blob_inventory": blobs,
        "base_recursive_blob_inventory_sha256": sha256_bytes(
            canonical_json_bytes(blobs)
        ),
        "remote_state_at_receipt": REMOTE_EMPTY,
        "target_paths_absent": True,
        "formal_label_loader_eligible": False,
        "gate_training_unlocked": False,
    }


def _load_remote_base_receipt(
    *,
    path: Path,
    contract: ScientificRepairPublicationContract,
    claim_sha256: str,
) -> tuple[Mapping[str, Any], bytes] | None:
    if not path.exists() and not path.is_symlink():
        return None
    payload = _regular_file_bytes(
        path,
        "remote base receipt",
        mode=contract.local_state["file_mode"],
    )
    receipt = strict_pretty_json_object_bytes(payload, label="remote base receipt")
    _exact_keys(
        receipt,
        {
            "schema_version",
            "protocol_id",
            "status",
            "destination",
            "claim_sha256",
            "base_main_revision",
            "base_reachable_history_commit_ids",
            "base_reachable_history_sha256",
            "base_recursive_tree_inventory",
            "base_recursive_tree_inventory_sha256",
            "base_recursive_blob_inventory",
            "base_recursive_blob_inventory_sha256",
            "remote_state_at_receipt",
            "target_paths_absent",
            "formal_label_loader_eligible",
            "gate_training_unlocked",
        },
        "remote base receipt",
    )
    history = receipt["base_reachable_history_commit_ids"]
    tree = receipt["base_recursive_tree_inventory"]
    blobs = receipt["base_recursive_blob_inventory"]
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or receipt["protocol_id"] != PROTOCOL_ID
        or receipt["status"] != REMOTE_BASE_RECEIPT_STATUS
        or receipt["destination"] != contract.destination
        or receipt["claim_sha256"] != claim_sha256
        or not isinstance(history, list)
        or history != sorted(set(history))
        or not history
        or any(_COMMIT.fullmatch(str(identifier)) is None for identifier in history)
        or receipt["base_main_revision"] not in history
        or receipt["base_reachable_history_sha256"]
        != sha256_bytes(canonical_json_bytes(history))
        or not isinstance(tree, list)
        or not isinstance(blobs, list)
        or receipt["base_recursive_tree_inventory_sha256"]
        != sha256_bytes(canonical_json_bytes(tree))
        or receipt["base_recursive_blob_inventory_sha256"]
        != sha256_bytes(canonical_json_bytes(blobs))
        or blobs != [record for record in tree if record.get("type") == "file"]
        or receipt["remote_state_at_receipt"] != REMOTE_EMPTY
        or receipt["target_paths_absent"] is not True
        or receipt["formal_label_loader_eligible"] is not False
        or receipt["gate_training_unlocked"] is not False
    ):
        raise ValueError("remote base receipt binding drifted")
    _commit(receipt["base_main_revision"], "remote base receipt revision")
    return receipt, payload


def _capture_remote_base_receipt(
    *,
    api: Any,
    contract: ScientificRepairPublicationContract,
    claim_sha256: str,
) -> Mapping[str, Any]:
    if _tag_snapshot(api, contract) is not None:
        raise ValueError("remote base cannot be captured after a tag exists")
    main = _revision_sha(api, contract, "main")
    history, _titles = _reachable_history(api, contract, main)
    tree = _recursive_tree_inventory(api, contract, main)
    return _remote_base_receipt_payload(
        contract=contract,
        claim_sha256=claim_sha256,
        base_main_revision=main,
        history_commit_ids=history,
        tree_inventory=tree,
    )


def _revalidate_remote_base_receipt(
    *,
    api: Any,
    contract: ScientificRepairPublicationContract,
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    base = receipt["base_main_revision"]
    if _revision_sha(api, contract, base) != base:
        raise ValueError("remote base revision no longer resolves to itself")
    history, _titles = _reachable_history(api, contract, base)
    tree = _recursive_tree_inventory(api, contract, base)
    if (
        list(history) != receipt["base_reachable_history_commit_ids"]
        or [dict(record) for record in tree]
        != receipt["base_recursive_tree_inventory"]
    ):
        raise ValueError("remote base history or recursive tree drifted")
    return {
        "base_main_revision": base,
        "base_reachable_history_sha256": receipt[
            "base_reachable_history_sha256"
        ],
        "base_recursive_tree_inventory_sha256": receipt[
            "base_recursive_tree_inventory_sha256"
        ],
        "base_recursive_blob_inventory_sha256": receipt[
            "base_recursive_blob_inventory_sha256"
        ],
    }


def _claim_payload(
    contract: ScientificRepairPublicationContract,
    prepared: PreparedScientificRepairPublication,
) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": CLAIM_STATUS,
        "destination": dict(contract.destination),
        "archive_sha256": prepared.archive_sha256,
        "sidecar_sha256": prepared.sidecar_sha256,
        "producer_claim_sha256": prepared.producer_claim_sha256,
        "producer_completion_sha256": prepared.producer_completion_sha256,
        "publication_source": dict(prepared.sidecar["publication_source"]),
        "formal_label_loader_eligible": False,
        "gate_training_unlocked": False,
    }


def _revision_sha(
    api: Any,
    contract: ScientificRepairPublicationContract,
    revision: str,
) -> str:
    destination = contract.destination
    info = api.dataset_info(destination["repo"], revision=revision)
    if info.private is not True:
        raise ValueError("destination dataset must remain private")
    return _commit(info.sha, f"remote revision {revision}")


def _tag_snapshot(
    api: Any,
    contract: ScientificRepairPublicationContract,
) -> TagSnapshot | None:
    destination = contract.destination
    refs = api.list_repo_refs(
        destination["repo"], repo_type=destination["repo_type"]
    )
    tags = [reference for reference in refs.tags if reference.name == destination["tag"]]
    if len(tags) > 1:
        raise ValueError("remote tag is ambiguous")
    if not tags:
        return None
    object_identity = _commit(tags[0].target_commit, "annotated tag object")
    resolved = _revision_sha(api, contract, destination["tag"])
    if destination["annotated_tag_required"] and object_identity == resolved:
        raise ValueError("frozen tag must be annotated, not lightweight")
    return TagSnapshot(object_identity=object_identity, resolved_commit=resolved)


def _assert_download_path(
    root: Path,
    returned: Path,
    relative_path: str,
    label: str,
) -> None:
    if not root.is_absolute() or not returned.is_absolute():
        raise ValueError("HF download paths must be absolute")
    expected = root / relative_path
    if returned != expected:
        raise ValueError("HF download returned a lexically unexpected path")
    try:
        relative = returned.relative_to(root)
    except ValueError as error:
        raise ValueError("HF download escaped the fresh root") from error
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("fresh download root became a symlink or non-directory")
    current = root
    for component in relative.parts[:-1]:
        current /= component
        if not stat.S_ISDIR(current.lstat().st_mode):
            raise ValueError(f"{label} traverses a symlink or non-directory")
    if not stat.S_ISREG(returned.lstat().st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")


def _download_pair(
    *,
    download_fn: Callable[..., str],
    contract: ScientificRepairPublicationContract,
    prepared: PreparedScientificRepairPublication,
    revision: str,
    directory: Path,
) -> Mapping[str, Any]:
    if not directory.is_absolute() or directory.is_symlink() or any(directory.iterdir()):
        raise ValueError("fresh download directory must start empty and absolute")
    destination = contract.destination
    archive_path = Path(
        download_fn(
            repo_id=destination["repo"],
            filename=destination["archive_path"],
            repo_type=destination["repo_type"],
            revision=revision,
            local_dir=directory,
            force_download=True,
        )
    )
    _assert_download_path(directory, archive_path, destination["archive_path"], "archive")
    sidecar_path = Path(
        download_fn(
            repo_id=destination["repo"],
            filename=destination["sidecar_path"],
            repo_type=destination["repo_type"],
            revision=revision,
            local_dir=directory,
            force_download=True,
        )
    )
    _assert_download_path(directory, archive_path, destination["archive_path"], "archive")
    _assert_download_path(directory, sidecar_path, destination["sidecar_path"], "sidecar")
    archive = _regular_file_bytes(archive_path, "downloaded repaired archive")
    sidecar = _regular_file_bytes(sidecar_path, "downloaded repaired sidecar")
    if archive != prepared.archive_bytes or sidecar != prepared.sidecar_bytes:
        raise ValueError("remote repaired archive or sidecar differs from local bytes")
    _strict_transport_readback(archive, contract=contract)
    parsed = strict_pretty_json_object_bytes(sidecar, label="downloaded sidecar")
    if parsed != prepared.sidecar:
        raise ValueError("downloaded sidecar semantic binding drifted")
    return {
        "archive_sha256": prepared.archive_sha256,
        "archive_size_bytes": len(archive),
        "sidecar_sha256": prepared.sidecar_sha256,
        "force_download": True,
        "download_directory_started_empty": True,
        "strict_repaired_transport_readback_passed": True,
    }


def _verify_exact_pair_commit(
    *,
    api: Any,
    contract: ScientificRepairPublicationContract,
    immutable_revision: str,
    remote_base_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    destination = contract.destination
    immutable = _commit(immutable_revision, "pair commit")
    if _revision_sha(api, contract, immutable) != immutable:
        raise ValueError("pair commit does not resolve to itself")
    history, titles = _reachable_history(api, contract, immutable)
    base_history = set(remote_base_receipt["base_reachable_history_commit_ids"])
    if set(history) != base_history | {immutable}:
        raise ValueError("pair reachable history is not base plus exactly pair commit")
    if titles.get(immutable) != PAIR_COMMIT_TITLE:
        raise ValueError("pair commit title drifted")
    pair_tree = _recursive_tree_inventory(api, contract, immutable)
    pair_blobs = [dict(record) for record in _blob_inventory(pair_tree)]
    base_blobs = remote_base_receipt["base_recursive_blob_inventory"]
    expected = {destination["archive_path"], destination["sidecar_path"]}
    if expected & {record["path"] for record in base_blobs}:
        raise ValueError("pair target paths were already present in remote base")
    targets = [record for record in pair_blobs if record["path"] in expected]
    non_targets = [record for record in pair_blobs if record["path"] not in expected]
    target_sizes = {
        destination["archive_path"]: contract.source["archive"]["size_bytes"],
        destination["sidecar_path"]: None,
    }
    if (
        {record["path"] for record in targets} != expected
        or len(targets) != 2
        or any(record["type"] != "file" for record in targets)
        or targets[0].get("identity") is None
        or targets[1].get("identity") is None
        or non_targets != base_blobs
        or next(
            record for record in targets if record["path"] == destination["archive_path"]
        )["size_bytes"]
        != target_sizes[destination["archive_path"]]
    ):
        raise ValueError(
            "pair recursive blobs after removing targets differ from remote base"
        )
    return {
        "pair_revision": immutable,
        "pair_title": PAIR_COMMIT_TITLE,
        "base_main_revision": remote_base_receipt["base_main_revision"],
        "base_reachable_history_commit_ids": list(
            remote_base_receipt["base_reachable_history_commit_ids"]
        ),
        "pair_reachable_history_commit_ids": list(history),
        "pair_reachable_history_sha256": sha256_bytes(
            canonical_json_bytes(list(history))
        ),
        "base_recursive_blob_inventory_sha256": remote_base_receipt[
            "base_recursive_blob_inventory_sha256"
        ],
        "pair_non_target_blob_inventory_sha256": sha256_bytes(
            canonical_json_bytes(non_targets)
        ),
        "target_blob_inventory": targets,
    }


def _verify_remote_pair(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: ScientificRepairPublicationContract,
    prepared: PreparedScientificRepairPublication,
    revision: str,
    fresh_parent: Path,
    remote_base_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    provenance = _verify_exact_pair_commit(
        api=api,
        contract=contract,
        immutable_revision=revision,
        remote_base_receipt=remote_base_receipt,
    )
    with tempfile.TemporaryDirectory(
        prefix="repaired-label-publication-readback-", dir=fresh_parent
    ) as directory:
        download = _download_pair(
            download_fn=download_fn,
            contract=contract,
            prepared=prepared,
            revision=revision,
            directory=Path(directory),
        )
    return {"provenance": provenance, "download": download}


def _validated_fresh_parent(value: str | Path) -> Path:
    parent = Path(value)
    if not parent.is_absolute() or not parent.is_dir() or parent.is_symlink():
        raise ValueError("fresh-download parent must be absolute non-symlink directory")
    return parent


def inspect_remote_state(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: ScientificRepairPublicationContract,
    prepared: PreparedScientificRepairPublication,
    fresh_download_parent: str | Path,
    remote_base_receipt: Mapping[str, Any] | None,
) -> RemoteInspection:
    parent = _validated_fresh_parent(fresh_download_parent)
    destination = contract.destination
    main = _revision_sha(api, contract, "main")
    tag = _tag_snapshot(api, contract)
    if tag is not None:
        if remote_base_receipt is None:
            raise ValueError("non-empty remote has no sealed remote-base receipt")
        _verify_remote_pair(
            api=api,
            download_fn=download_fn,
            contract=contract,
            prepared=prepared,
            revision=tag.resolved_commit,
            fresh_parent=parent,
            remote_base_receipt=remote_base_receipt,
        )
        return RemoteInspection(
            state=REMOTE_TAGGED_BYTE_IDENTICAL,
            main_revision=main,
            immutable_revision=tag.resolved_commit,
            tag_object_identity=tag.object_identity,
        )
    tree = _recursive_tree_inventory(api, contract, main)
    blob_paths = {record["path"] for record in _blob_inventory(tree)}
    archive_present = destination["archive_path"] in blob_paths
    sidecar_present = destination["sidecar_path"] in blob_paths
    if archive_present != sidecar_present:
        raise ValueError("remote archive/sidecar partial state is a conflict")
    if not archive_present:
        if remote_base_receipt is not None:
            if main != remote_base_receipt["base_main_revision"]:
                raise ValueError("empty remote main differs from sealed remote base")
            _revalidate_remote_base_receipt(
                api=api,
                contract=contract,
                receipt=remote_base_receipt,
            )
        return RemoteInspection(
            state=REMOTE_EMPTY,
            main_revision=main,
            immutable_revision=None,
            tag_object_identity=None,
        )
    if remote_base_receipt is None:
        raise ValueError("non-empty remote has no sealed remote-base receipt")
    _verify_remote_pair(
        api=api,
        download_fn=download_fn,
        contract=contract,
        prepared=prepared,
        revision=main,
        fresh_parent=parent,
        remote_base_receipt=remote_base_receipt,
    )
    return RemoteInspection(
        state=REMOTE_BYTE_IDENTICAL_UNTAGGED,
        main_revision=main,
        immutable_revision=main,
        tag_object_identity=None,
    )


def _operation(factory: Callable[..., Any], path: str, payload: bytes) -> Any:
    return factory(path_in_repo=path, path_or_fileobj=io.BytesIO(payload))


def _commit_oid(value: Any) -> str:
    for attribute in ("oid", "commit_id"):
        candidate = getattr(value, attribute, None)
        if candidate is not None:
            return _commit(candidate, "HF commit response")
    raise ValueError("HF commit response lacks an immutable commit id")


def _create_tag_recovering_response_loss(
    *,
    api: Any,
    contract: ScientificRepairPublicationContract,
    immutable_revision: str,
) -> TagSnapshot:
    destination = contract.destination
    try:
        api.create_tag(
            destination["repo"],
            tag=destination["tag"],
            tag_message=TAG_MESSAGE,
            revision=immutable_revision,
            repo_type=destination["repo_type"],
            exist_ok=False,
        )
    except Exception:
        recovered = _tag_snapshot(api, contract)
        if recovered is not None and recovered.resolved_commit == immutable_revision:
            return recovered
        raise
    observed = _tag_snapshot(api, contract)
    if observed is None or observed.resolved_commit != immutable_revision:
        raise ValueError("created annotated tag does not resolve to intended commit")
    return observed


def _remote_identity_snapshot(
    api: Any,
    contract: ScientificRepairPublicationContract,
    immutable_revision: str,
) -> Mapping[str, Any]:
    tag = _tag_snapshot(api, contract)
    if tag is None:
        raise ValueError("annotated publication tag is absent")
    immutable = _revision_sha(api, contract, immutable_revision)
    main = _revision_sha(api, contract, "main")
    if tag.resolved_commit != immutable_revision or immutable != immutable_revision:
        raise ValueError("tag-resolved or immutable commit drifted")
    return {
        "main_resolved_commit": main,
        "tag_object_identity": tag.object_identity,
        "tag_resolved_commit": tag.resolved_commit,
        "immutable_resolved_commit": immutable,
    }


def _fresh_stability_attestation(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: ScientificRepairPublicationContract,
    prepared: PreparedScientificRepairPublication,
    immutable_revision: str,
    fresh_parent: Path,
    remote_base_receipt: Mapping[str, Any],
    remote_base_receipt_sha256: str,
) -> Mapping[str, Any]:
    base_pre = _revalidate_remote_base_receipt(
        api=api,
        contract=contract,
        receipt=remote_base_receipt,
    )
    identity_pre = _remote_identity_snapshot(api, contract, immutable_revision)
    provenance_pre = _verify_exact_pair_commit(
        api=api,
        contract=contract,
        immutable_revision=immutable_revision,
        remote_base_receipt=remote_base_receipt,
    )
    with tempfile.TemporaryDirectory(
        prefix="repaired-label-publication-fresh-", dir=fresh_parent
    ) as directory:
        download = _download_pair(
            download_fn=download_fn,
            contract=contract,
            prepared=prepared,
            revision=immutable_revision,
            directory=Path(directory),
        )
    identity_post = _remote_identity_snapshot(api, contract, immutable_revision)
    provenance_post = _verify_exact_pair_commit(
        api=api,
        contract=contract,
        immutable_revision=immutable_revision,
        remote_base_receipt=remote_base_receipt,
    )
    base_post = _revalidate_remote_base_receipt(
        api=api,
        contract=contract,
        receipt=remote_base_receipt,
    )
    if (
        identity_pre != identity_post
        or provenance_pre != provenance_post
        or base_pre != base_post
    ):
        raise ValueError("remote identity or exact pair provenance drifted during replay")
    if identity_pre["main_resolved_commit"] != immutable_revision:
        raise ValueError("main must remain at the frozen pair commit during completion")
    return {
        "remote_base_receipt_sha256": _sha(
            remote_base_receipt_sha256, "remote base receipt SHA256"
        ),
        "remote_base_pre": dict(base_pre),
        "remote_base_post": dict(base_post),
        "identity_pre": dict(identity_pre),
        "identity_post": dict(identity_post),
        "pair_provenance_pre": dict(provenance_pre),
        "pair_provenance_post": dict(provenance_post),
        "fresh_download": dict(download),
        "private_repo_verified": True,
        "annotated_tag_verified": True,
    }


def _completion_payload(
    *,
    contract: ScientificRepairPublicationContract,
    prepared: PreparedScientificRepairPublication,
    claim_sha256: str,
    remote_state_before_publish: str,
    immutable_revision: str,
    attestation: Mapping[str, Any],
) -> Mapping[str, Any]:
    if remote_state_before_publish not in ALLOWED_REMOTE_STATES:
        raise ValueError("completion remote state is not allowed")
    identity = attestation["identity_post"]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": COMPLETION_STATUS,
        "destination": dict(contract.destination),
        "claim_sha256": _sha(claim_sha256, "publication claim SHA256"),
        "archive_sha256": prepared.archive_sha256,
        "sidecar_sha256": prepared.sidecar_sha256,
        "producer_completion_sha256": prepared.producer_completion_sha256,
        "remote_base_receipt_sha256": _sha(
            attestation["remote_base_receipt_sha256"],
            "remote base receipt SHA256",
        ),
        "publication_source": dict(prepared.sidecar["publication_source"]),
        "remote_state_before_publish": remote_state_before_publish,
        "immutable_revision": _commit(immutable_revision, "immutable revision"),
        "annotated_tag_object_identity": _commit(
            identity["tag_object_identity"], "annotated tag object"
        ),
        "tag_resolved_commit": _commit(
            identity["tag_resolved_commit"], "tag-resolved commit"
        ),
        "fresh_download_attestation": dict(attestation),
        "original_invalid_forensic_formal_label_loader_eligible": False,
        "producer_attempt_reclassified": False,
        "formal_label_loader_eligible": True,
        "gate_training_unlocked": True,
    }


def _load_existing_completion(
    *,
    path: Path,
    staging_path: Path,
    contract: ScientificRepairPublicationContract,
    prepared: PreparedScientificRepairPublication,
    claim_sha256: str,
) -> Mapping[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if not staging_path.exists() and not staging_path.is_symlink():
        raise ValueError("publication completion exists without retained staging")
    mode = contract.local_state["file_mode"]
    final_bytes = _regular_file_bytes(path, "publication completion", mode=mode)
    staged_bytes = _regular_file_bytes(
        staging_path, "retained publication completion staging", mode=mode
    )
    final_stat = path.stat(follow_symlinks=False)
    staged_stat = staging_path.stat(follow_symlinks=False)
    if (
        final_bytes != staged_bytes
        or final_stat.st_dev != staged_stat.st_dev
        or final_stat.st_ino != staged_stat.st_ino
    ):
        raise ValueError("completion and retained staging are not the same inode")
    completion = strict_pretty_json_object_bytes(
        final_bytes,
        label="publication completion",
    )
    required = {
        "schema_version",
        "protocol_id",
        "status",
        "destination",
        "claim_sha256",
        "archive_sha256",
        "sidecar_sha256",
        "producer_completion_sha256",
        "remote_base_receipt_sha256",
        "publication_source",
        "remote_state_before_publish",
        "immutable_revision",
        "annotated_tag_object_identity",
        "tag_resolved_commit",
        "fresh_download_attestation",
        "original_invalid_forensic_formal_label_loader_eligible",
        "producer_attempt_reclassified",
        "formal_label_loader_eligible",
        "gate_training_unlocked",
    }
    _exact_keys(completion, required, "publication completion")
    if (
        completion["schema_version"] != SCHEMA_VERSION
        or completion["protocol_id"] != PROTOCOL_ID
        or completion["status"] != COMPLETION_STATUS
        or completion["destination"] != contract.destination
        or completion["claim_sha256"] != claim_sha256
        or completion["archive_sha256"] != prepared.archive_sha256
        or completion["sidecar_sha256"] != prepared.sidecar_sha256
        or completion["producer_completion_sha256"]
        != prepared.producer_completion_sha256
        or completion["remote_base_receipt_sha256"]
        != completion["fresh_download_attestation"].get(
            "remote_base_receipt_sha256"
        )
        or completion["publication_source"]
        != prepared.sidecar["publication_source"]
        or completion["remote_state_before_publish"] not in ALLOWED_REMOTE_STATES
        or completion["original_invalid_forensic_formal_label_loader_eligible"]
        is not False
        or completion["producer_attempt_reclassified"] is not False
        or completion["formal_label_loader_eligible"] is not True
        or completion["gate_training_unlocked"] is not True
    ):
        raise ValueError("publication completion binding drifted")
    _commit(completion["immutable_revision"], "completion immutable revision")
    _commit(completion["annotated_tag_object_identity"], "completion tag object")
    if completion["tag_resolved_commit"] != completion["immutable_revision"]:
        raise ValueError("completion tag-resolved commit differs from immutable")
    return completion


def _publish_completion_from_retained_stage_last(
    *,
    staging_path: Path,
    completion_path: Path,
    payload: bytes,
    mode: int,
) -> None:
    staged = _regular_file_bytes(
        staging_path,
        "retained publication completion staging",
        mode=mode,
    )
    if staged != payload:
        raise ValueError("retained completion staging bytes drifted")
    if completion_path.exists() or completion_path.is_symlink():
        final = _regular_file_bytes(
            completion_path,
            "publication completion",
            mode=mode,
        )
        staged_stat = staging_path.stat(follow_symlinks=False)
        final_stat = completion_path.stat(follow_symlinks=False)
        if (
            final != payload
            or staged_stat.st_dev != final_stat.st_dev
            or staged_stat.st_ino != final_stat.st_ino
        ):
            raise ValueError("existing completion is not the retained-stage inode")
        return
    # note (luojiaxuan): The retained stage is already durable and remains in
    # place. This hard-link plus directory fsync is the final namespace mutation.
    os.link(staging_path, completion_path)
    _fsync_directory(completion_path.parent)


def publish_scientific_repair_archive(
    *,
    api: Any,
    download_fn: Callable[..., str],
    operation_factory: Callable[..., Any],
    contract: ScientificRepairPublicationContract,
    prepared: PreparedScientificRepairPublication,
    fresh_download_parent: str | Path,
) -> Mapping[str, Any]:
    parent = _validated_fresh_parent(fresh_download_parent)
    claim = _claim_payload(contract, prepared)
    claim_bytes = pretty_json_bytes(claim)
    claim_sha = sha256_bytes(claim_bytes)
    claim_path = Path(contract.local_state["claim_path"])
    base_receipt_path = Path(contract.local_state["remote_base_receipt_path"])
    completion_staging_path = Path(contract.local_state["completion_staging_path"])
    completion_path = Path(contract.local_state["completion_seal_path"])
    mode = contract.local_state["file_mode"]
    claim_present = claim_path.exists() or claim_path.is_symlink()
    receipt_present = base_receipt_path.exists() or base_receipt_path.is_symlink()
    staging_present = (
        completion_staging_path.exists() or completion_staging_path.is_symlink()
    )
    completion_present = completion_path.exists() or completion_path.is_symlink()
    if completion_present and not (claim_present and receipt_present and staging_present):
        raise ValueError(
            "orphan publication completion lacks claim, base receipt, or staging"
        )
    if (receipt_present or staging_present) and not claim_present:
        raise ValueError("orphan publication state exists without its claim")
    if staging_present and not receipt_present:
        raise ValueError("completion staging exists without remote-base receipt")
    _exclusive_or_identical_state(claim_path, claim_bytes, mode)
    loaded_receipt = _load_remote_base_receipt(
        path=base_receipt_path,
        contract=contract,
        claim_sha256=claim_sha,
    )
    existing = _load_existing_completion(
        path=completion_path,
        staging_path=completion_staging_path,
        contract=contract,
        prepared=prepared,
        claim_sha256=claim_sha,
    )
    if existing is not None:
        if loaded_receipt is None:
            raise ValueError("completed publication has no remote-base receipt")
        remote_base_receipt, receipt_bytes = loaded_receipt
        receipt_sha = sha256_bytes(receipt_bytes)
        if existing["remote_base_receipt_sha256"] != receipt_sha:
            raise ValueError("completion remote-base receipt binding drifted")
        _revalidate_remote_base_receipt(
            api=api,
            contract=contract,
            receipt=remote_base_receipt,
        )
        immutable = existing["immutable_revision"]
        inspection = inspect_remote_state(
            api=api,
            download_fn=download_fn,
            contract=contract,
            prepared=prepared,
            fresh_download_parent=parent,
            remote_base_receipt=remote_base_receipt,
        )
        if (
            inspection.state != REMOTE_TAGGED_BYTE_IDENTICAL
            or inspection.immutable_revision != immutable
            or inspection.tag_object_identity
            != existing["annotated_tag_object_identity"]
        ):
            raise ValueError("completed publication no longer has exact tagged bytes")
        attestation = _fresh_stability_attestation(
            api=api,
            download_fn=download_fn,
            contract=contract,
            prepared=prepared,
            immutable_revision=immutable,
            fresh_parent=parent,
            remote_base_receipt=remote_base_receipt,
            remote_base_receipt_sha256=receipt_sha,
        )
        expected = _completion_payload(
            contract=contract,
            prepared=prepared,
            claim_sha256=claim_sha,
            remote_state_before_publish=remote_base_receipt[
                "remote_state_at_receipt"
            ],
            immutable_revision=immutable,
            attestation=attestation,
        )
        if existing != expected:
            raise ValueError("completion differs from fresh immutable replay")
        return existing

    destination = contract.destination
    if loaded_receipt is None:
        try:
            api.create_repo(
                destination["repo"],
                repo_type=destination["repo_type"],
                private=True,
                exist_ok=True,
            )
        except Exception:
            try:
                _revision_sha(api, contract, "main")
            except Exception:
                raise
        remote_base_receipt = _capture_remote_base_receipt(
            api=api,
            contract=contract,
            claim_sha256=claim_sha,
        )
        receipt_bytes = pretty_json_bytes(remote_base_receipt)
        _exclusive_or_identical_state(base_receipt_path, receipt_bytes, mode)
        loaded_receipt = _load_remote_base_receipt(
            path=base_receipt_path,
            contract=contract,
            claim_sha256=claim_sha,
        )
        if loaded_receipt is None or loaded_receipt[0] != remote_base_receipt:
            raise ValueError("durable remote-base receipt readback drifted")
        receipt_bytes = loaded_receipt[1]
    else:
        remote_base_receipt, receipt_bytes = loaded_receipt
        _revalidate_remote_base_receipt(
            api=api,
            contract=contract,
            receipt=remote_base_receipt,
        )
    receipt_sha = sha256_bytes(receipt_bytes)
    inspection = inspect_remote_state(
        api=api,
        download_fn=download_fn,
        contract=contract,
        prepared=prepared,
        fresh_download_parent=parent,
        remote_base_receipt=remote_base_receipt,
    )
    immutable: str
    if inspection.state == REMOTE_TAGGED_BYTE_IDENTICAL:
        assert inspection.immutable_revision is not None
        immutable = inspection.immutable_revision
    elif inspection.state == REMOTE_BYTE_IDENTICAL_UNTAGGED:
        assert inspection.immutable_revision is not None
        immutable = inspection.immutable_revision
        _create_tag_recovering_response_loss(
            api=api, contract=contract, immutable_revision=immutable
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
        try:
            response = api.create_commit(
                destination["repo"],
                operations=operations,
                commit_message=PAIR_COMMIT_TITLE,
                repo_type=destination["repo_type"],
                revision="main",
                parent_commit=inspection.main_revision,
            )
            immutable = _commit_oid(response)
        except Exception:
            recovered = inspect_remote_state(
                api=api,
                download_fn=download_fn,
                contract=contract,
                prepared=prepared,
                fresh_download_parent=parent,
                remote_base_receipt=remote_base_receipt,
            )
            if recovered.state not in {
                REMOTE_BYTE_IDENTICAL_UNTAGGED,
                REMOTE_TAGGED_BYTE_IDENTICAL,
            } or recovered.immutable_revision is None:
                raise
            immutable = recovered.immutable_revision
        _verify_remote_pair(
            api=api,
            download_fn=download_fn,
            contract=contract,
            prepared=prepared,
            revision=immutable,
            fresh_parent=parent,
            remote_base_receipt=remote_base_receipt,
        )
        tag = _tag_snapshot(api, contract)
        if tag is None:
            _create_tag_recovering_response_loss(
                api=api, contract=contract, immutable_revision=immutable
            )
        elif tag.resolved_commit != immutable:
            raise ValueError("existing tag resolves to a conflicting commit")
    else:
        raise AssertionError(f"unknown remote state: {inspection.state}")

    attestation = _fresh_stability_attestation(
        api=api,
        download_fn=download_fn,
        contract=contract,
        prepared=prepared,
        immutable_revision=immutable,
        fresh_parent=parent,
        remote_base_receipt=remote_base_receipt,
        remote_base_receipt_sha256=receipt_sha,
    )
    completion = _completion_payload(
        contract=contract,
        prepared=prepared,
        claim_sha256=claim_sha,
        remote_state_before_publish=remote_base_receipt["remote_state_at_receipt"],
        immutable_revision=immutable,
        attestation=attestation,
    )
    completion_bytes = pretty_json_bytes(completion)
    _exclusive_or_identical_state(
        completion_staging_path,
        completion_bytes,
        mode,
    )
    # note (luojiaxuan): The next helper performs only the final no-replace
    # hard-link and directory fsync. Returning immediately preserves last-mutation.
    _publish_completion_from_retained_stage_last(
        staging_path=completion_staging_path,
        completion_path=completion_path,
        payload=completion_bytes,
        mode=mode,
    )
    return completion


__all__ = [
    "ALLOWED_REMOTE_STATES",
    "CANONICAL_CONFIG_PATH",
    "CLAIM_STATUS",
    "COMPLETION_STATUS",
    "DESTINATION",
    "FROZEN_CONFIG_SHA256",
    "PAIR_COMMIT_TITLE",
    "PROTOCOL_ID",
    "PreparedScientificRepairPublication",
    "REMOTE_BYTE_IDENTICAL_UNTAGGED",
    "REMOTE_BASE_RECEIPT_STATUS",
    "REMOTE_EMPTY",
    "REMOTE_TAGGED_BYTE_IDENTICAL",
    "RemoteInspection",
    "SIDECAR_STATUS",
    "SOURCE_STATUS",
    "ScientificRepairPublicationContract",
    "TagSnapshot",
    "canonical_json_bytes",
    "inspect_remote_state",
    "load_frozen_publication_contract",
    "prepare_scientific_repair_publication",
    "pretty_json_bytes",
    "publication_contract_from_data",
    "publish_scientific_repair_archive",
    "sha256_bytes",
    "strict_pretty_json_object_bytes",
    "validate_clean_pushed_source",
    "validate_publication_contract_data",
]
