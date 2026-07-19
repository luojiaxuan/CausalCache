"""Strict immutable-HF publication for the processor-v2 formal artifact."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.set_utility_processor_artifacts import (
    ARTIFACT_FILENAME_TEMPLATE,
    CANDIDATE_SCHEDULE_FILENAME,
    canonical_json_bytes,
    canonical_pretty_json_bytes,
)
from causalcache.set_utility_processor_freeze import WORKER_COUNT
from causalcache.set_utility_processor_postflight_v2 import VALIDATION_STATUS
from causalcache.set_utility_processor_result_v2 import (
    EXPECTED_FORMAL_FILE_COUNT,
    PENDING_PUBLICATION_STATUS,
    PROTOCOL_ID as RESULT_PROTOCOL_ID,
    VALID_RESULT_STATUS,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_processor_freeze_v2_hf_publication"
PUBLICATION_STATUS = "PUBLISHED_PROCESSOR_V2_IMMUTABLE_HF_VERIFIED"
REPO_TYPE = "dataset"
COMMIT_MESSAGE = "Publish processor-v2 image-contract repair artifact"
TAG_MESSAGE = "Immutable processor-v2 image-contract repair artifact"
REMOTE_FORMAL_DIRECTORY = "formal"
REMOTE_GIT_SUMMARY = "git/summary.json"
REMOTE_GIT_CARD = "git/card.md"
EXPECTED_REMOTE_FILE_COUNT = EXPECTED_FORMAL_FILE_COUNT + 2

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_HF_REPO = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
)
_HF_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _read_regular(path: Path, *, label: str) -> bytes:
    if not path.is_absolute() or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError(f"{label} path must be absolute and support O_NOFOLLOW")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        blocks: list[bytes] = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            blocks.append(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    payload = b"".join(blocks)
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
    if identity_before != identity_after or len(payload) != after.st_size:
        raise RuntimeError(f"{label} changed while being read")
    return payload


def _inspect_regular(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_absolute() or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError(f"{label} path must be absolute and support O_NOFOLLOW")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        digest = hashlib.sha256()
        byte_count = 0
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
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


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _commit(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase 40-character commit")
    return value


def _safe_remote(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a safe relative POSIX path")
    return value


def _expected_formal_paths() -> tuple[str, ...]:
    paths = {
        "manifest.json",
        "run-identity.json",
        CANDIDATE_SCHEDULE_FILENAME,
    }
    paths.update(
        f"candidate-parts/worker-{worker:02d}.jsonl"
        for worker in range(WORKER_COUNT)
    )
    paths.update(
        f"logs/{phase}-worker-{worker:02d}.log"
        for phase in ("ocr", "processor")
        for worker in range(WORKER_COUNT)
    )
    paths.update(
        f"receipts/ocr-worker-{worker:02d}.json"
        for worker in range(WORKER_COUNT)
    )
    paths.update(
        "substrate/" + ARTIFACT_FILENAME_TEMPLATE.format(worker_index=worker)
        for worker in range(WORKER_COUNT)
    )
    if len(paths) != EXPECTED_FORMAL_FILE_COUNT:
        raise AssertionError("processor-v2 publication no longer binds 23 files")
    return tuple(sorted(paths))


def _snapshot_formal(root: Path) -> tuple[dict[str, Any], ...]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("formal root must be an absolute real directory")
    observed: list[str] = []
    for directory, directories, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directories:
            if (base / name).is_symlink():
                raise ValueError("formal root must not contain symlinks")
        for name in files:
            path = base / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError("formal root must not contain symlinks")
            observed.append(relative)
    expected = _expected_formal_paths()
    if tuple(sorted(observed)) != expected:
        raise ValueError("formal root must contain the exact 23-file inventory")
    inventory = []
    for relative in expected:
        inspected = _inspect_regular(
            root / relative, label=f"formal file {relative}"
        )
        inventory.append(
            {
                "path": relative,
                **inspected,
            }
        )
    return tuple(inventory)


def _validate_pending_summary(
    summary: Mapping[str, Any],
    *,
    formal_root: Path,
    inventory: Sequence[Mapping[str, Any]],
    repo: str,
    tag: str,
) -> None:
    required_top = {
        "artifact",
        "bindings",
        "execution",
        "failure",
        "negative_operations",
        "postflight",
        "postflight_accepted_for_formal_result",
        "protocol_id",
        "publication",
        "schema_version",
        "scientific_eligibility",
        "status",
    }
    if set(summary) != required_top:
        raise ValueError("Git result summary top-level fields drifted")
    if (
        summary.get("schema_version") != "1.0.0"
        or summary.get("protocol_id") != RESULT_PROTOCOL_ID
        or summary.get("status") != VALID_RESULT_STATUS
        or summary.get("scientific_eligibility") is not True
        or summary.get("postflight_accepted_for_formal_result") is not True
        or summary.get("failure") is not None
    ):
        raise ValueError("Git result summary is not one valid formal result")
    publication = summary.get("publication")
    if not isinstance(publication, Mapping) or dict(publication) != {
        "hf_mutation_count": 0,
        "intended_private_hf_repo": repo,
        "intended_tag": tag,
        "status": PENDING_PUBLICATION_STATUS,
    }:
        raise ValueError("Git result summary is not valid PENDING_HF_UPLOAD")
    artifact = summary.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("Git result artifact summary is missing")
    normalized = [dict(record) for record in inventory]
    if (
        artifact.get("file_count") != EXPECTED_FORMAL_FILE_COUNT
        or artifact.get("file_inventory") != normalized
        or artifact.get("file_inventory_sha256")
        != _sha256_bytes(canonical_json_bytes(normalized))
        or artifact.get("total_byte_count")
        != sum(record["size_bytes"] for record in normalized)
        or artifact.get("output_root") != str(formal_root)
        or artifact.get("contains_raw_candidate_or_ocr_copy") is not False
        or artifact.get("staging_root_present_after_success") is not False
    ):
        raise ValueError("Git result artifact differs from exact formal root")
    expected_staging = formal_root.parent / f".{formal_root.name}.incomplete"
    if (
        artifact.get("staging_root") != str(expected_staging)
        or expected_staging.exists()
        or expected_staging.is_symlink()
    ):
        raise ValueError("Git result staging-root absence contract drifted")
    postflight = summary.get("postflight")
    if not isinstance(postflight, Mapping) or postflight.get("status") != (
        VALIDATION_STATUS
    ):
        raise ValueError("Git result summary lacks valid processor-v2 postflight")
    negative = summary.get("negative_operations")
    if not isinstance(negative, Mapping) or any(
        value != 0 for value in negative.values()
    ):
        raise ValueError("Git result summary contains a forbidden negative operation")


def _destination(repo: str, tag: str, prefix: str) -> dict[str, str]:
    if _HF_REPO.fullmatch(repo) is None:
        raise ValueError("HF repo must use owner/name")
    if _HF_TAG.fullmatch(tag) is None:
        raise ValueError("HF tag is invalid")
    prefix = _safe_remote(prefix, label="HF prefix").rstrip("/")
    if prefix in {REMOTE_FORMAL_DIRECTORY, "git"}:
        raise ValueError("HF prefix must be an independent namespace")
    return {"repo": repo, "repo_type": REPO_TYPE, "tag": tag, "prefix": prefix}


def _source_descriptor(
    *,
    formal_root: Path,
    inventory: Sequence[Mapping[str, Any]],
    summary_path: Path,
    summary_sha: str,
    summary_size_bytes: int,
    card_path: Path,
    card_bytes: bytes,
) -> dict[str, Any]:
    normalized = [dict(record) for record in inventory]
    return {
        "formal_root": str(formal_root),
        "formal_file_count": EXPECTED_FORMAL_FILE_COUNT,
        "formal_file_inventory": normalized,
        "formal_file_inventory_sha256": _sha256_bytes(
            canonical_json_bytes(normalized)
        ),
        "formal_total_byte_count": sum(
            record["size_bytes"] for record in normalized
        ),
        "git_summary_path": str(summary_path),
        "git_summary_sha256": summary_sha,
        "git_summary_size_bytes": summary_size_bytes,
        "git_card_path": str(card_path),
        "git_card_sha256": _sha256_bytes(card_bytes),
        "git_card_size_bytes": len(card_bytes),
    }


def prepare_processor_v2_publication(
    *,
    formal_root: str | Path,
    git_summary: str | Path,
    git_card: str | Path,
    expected_result_summary_sha256: str,
    hf_repo: str,
    hf_tag: str,
    hf_prefix: str,
) -> dict[str, Any]:
    root = Path(formal_root)
    summary_path = Path(git_summary)
    card_path = Path(git_card)
    if (
        not root.is_absolute()
        or not summary_path.is_absolute()
        or not card_path.is_absolute()
    ):
        raise ValueError(
            "formal root, Git summary, and Git card paths must be absolute"
        )
    destination = _destination(hf_repo, hf_tag, hf_prefix)
    expected_sha = _sha(
        expected_result_summary_sha256,
        label="expected result summary SHA256",
    )
    inventory = _snapshot_formal(root)
    summary_bytes = _read_regular(summary_path, label="Git result summary")
    if _sha256_bytes(summary_bytes) != expected_sha:
        raise ValueError("Git result summary SHA256 differs from expected")
    summary = _strict_json_object(summary_bytes, label="Git result summary")
    if summary_bytes != canonical_pretty_json_bytes(summary):
        raise ValueError("Git result summary is not canonical pretty JSON")
    _validate_pending_summary(
        summary,
        formal_root=root,
        inventory=inventory,
        repo=hf_repo,
        tag=hf_tag,
    )
    card_bytes = _read_regular(card_path, label="Git result card")
    if not card_bytes or b"\x00" in card_bytes:
        raise ValueError("Git result card must be non-empty text")
    try:
        card_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Git result card must be UTF-8") from error
    return {
        "destination": destination,
        "source": _source_descriptor(
            formal_root=root,
            inventory=inventory,
            summary_path=summary_path,
            summary_sha=expected_sha,
            summary_size_bytes=len(summary_bytes),
            card_path=card_path,
            card_bytes=card_bytes,
        ),
    }


def _revision(api: Any, repo: str, revision: str) -> str:
    info = api.dataset_info(repo, revision=revision)
    if getattr(info, "private", None) is not True:
        raise ValueError("HF destination dataset must already exist and remain private")
    return _commit(getattr(info, "sha", None), label=f"HF revision {revision}")


def _tag_snapshot(api: Any, *, repo: str, tag: str) -> dict[str, str] | None:
    refs = api.list_repo_refs(repo, repo_type=REPO_TYPE)
    matches = [item for item in refs.tags if item.name == tag]
    if len(matches) > 1:
        raise ValueError("HF publication tag is ambiguous")
    if not matches:
        return None
    tag_object = _commit(matches[0].target_commit, label="annotated tag object")
    resolved = _revision(api, repo, tag)
    if tag_object == resolved:
        raise ValueError("HF publication tag is lightweight, not annotated")
    return {"object_identity": tag_object, "resolved_commit": resolved}


def _remote_paths(api: Any, *, repo: str, revision: str, prefix: str) -> set[str]:
    paths = api.list_repo_files(repo, repo_type=REPO_TYPE, revision=revision)
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise ValueError("HF remote file listing is invalid")
    return {
        path
        for path in paths
        if path == prefix or path.startswith(prefix + "/")
    }


def _remote_path_map(prepared: Mapping[str, Any]) -> dict[str, Path]:
    prefix = prepared["destination"]["prefix"]
    source = prepared["source"]
    root = Path(source["formal_root"])
    result = {
        f"{prefix}/{REMOTE_FORMAL_DIRECTORY}/{record['path']}": root
        / record["path"]
        for record in source["formal_file_inventory"]
    }
    result[f"{prefix}/{REMOTE_GIT_SUMMARY}"] = Path(source["git_summary_path"])
    result[f"{prefix}/{REMOTE_GIT_CARD}"] = Path(source["git_card_path"])
    if len(result) != EXPECTED_REMOTE_FILE_COUNT:
        raise AssertionError("processor-v2 publication must contain 25 remote files")
    return result


def _commit_oid(response: Any) -> str:
    for attribute in ("oid", "commit_id"):
        value = getattr(response, attribute, None)
        if value is not None:
            return _commit(value, label="HF commit response")
    raise ValueError("HF commit response lacks an immutable commit id")


def _validated_parent(path: str | Path, *, label: str) -> Path:
    parent = Path(path)
    if not parent.is_absolute() or parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"{label} must be an absolute real directory")
    return parent.resolve()


def _git_download_metadata(prepared: Mapping[str, Any]) -> dict[str, Any]:
    source = prepared["source"]
    prefix = prepared["destination"]["prefix"]
    return {
        "card": {
            "path": f"{prefix}/{REMOTE_GIT_CARD}",
            "sha256": source["git_card_sha256"],
            "size_bytes": source["git_card_size_bytes"],
        },
        "summary": {
            "path": f"{prefix}/{REMOTE_GIT_SUMMARY}",
            "sha256": source["git_summary_sha256"],
            "size_bytes": source["git_summary_size_bytes"],
        },
    }


def _download_publication(
    *,
    download_fn: Callable[..., str],
    prepared: Mapping[str, Any],
    revision: str,
    directory: Path,
) -> dict[str, Any]:
    directory.mkdir(mode=0o700, exist_ok=False)
    destination = prepared["destination"]
    source = prepared["source"]
    observed = []

    def download_one(
        remote_path: str,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
        label: str,
    ) -> None:
        returned = Path(
            download_fn(
                repo_id=destination["repo"],
                repo_type=REPO_TYPE,
                revision=revision,
                filename=remote_path,
                local_dir=str(directory),
                force_download=True,
            )
        )
        expected = directory / remote_path
        if returned != expected:
            raise ValueError("HF fresh download returned an unexpected path")
        inspected = _inspect_regular(returned, label=f"fresh download {label}")
        if inspected != {
            "size_bytes": expected_size_bytes,
            "sha256": expected_sha256,
        }:
            raise ValueError(
                f"HF fresh download differs from {label} source bytes"
            )

    for record in source["formal_file_inventory"]:
        remote_path = (
            f"{destination['prefix']}/{REMOTE_FORMAL_DIRECTORY}/{record['path']}"
        )
        download_one(
            remote_path,
            expected_sha256=record["sha256"],
            expected_size_bytes=record["size_bytes"],
            label=f"formal file {record['path']}",
        )
        observed.append(dict(record))
    git_metadata = _git_download_metadata(prepared)
    for name, metadata in sorted(git_metadata.items()):
        download_one(
            metadata["path"],
            expected_sha256=metadata["sha256"],
            expected_size_bytes=metadata["size_bytes"],
            label=f"Git {name}",
        )
    return {
        "all_remote_files_byte_identical": True,
        "byte_identical": True,
        "directory": str(directory),
        "formal_file_count": len(observed),
        "formal_file_inventory_sha256": _sha256_bytes(
            canonical_json_bytes(observed)
        ),
        "formal_total_byte_count": sum(item["size_bytes"] for item in observed),
        "git_metadata": git_metadata,
        "remote_file_count": len(observed) + len(git_metadata),
        "revision": revision,
    }


def _verify_existing_download(
    record: Mapping[str, Any],
    *,
    prepared: Mapping[str, Any],
) -> None:
    expected_keys = {
        "all_remote_files_byte_identical",
        "byte_identical",
        "directory",
        "formal_file_count",
        "formal_file_inventory_sha256",
        "formal_total_byte_count",
        "git_metadata",
        "remote_file_count",
        "revision",
    }
    if (
        set(record) != expected_keys
        or record.get("byte_identical") is not True
        or record.get("all_remote_files_byte_identical") is not True
    ):
        raise ValueError("publication receipt fresh-download fields drifted")
    root = _validated_parent(record["directory"], label="fresh-download directory")
    source = prepared["source"]
    destination = prepared["destination"]
    for item in source["formal_file_inventory"]:
        path = (
            root
            / destination["prefix"]
            / REMOTE_FORMAL_DIRECTORY
            / item["path"]
        )
        inspected = _inspect_regular(
            path, label=f"retained fresh file {item['path']}"
        )
        if inspected != {
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }:
            raise ValueError("retained fresh-download bytes drifted")
    expected_git_metadata = _git_download_metadata(prepared)
    if record["git_metadata"] != expected_git_metadata:
        raise ValueError("retained fresh-download Git metadata drifted")
    for name, metadata in expected_git_metadata.items():
        inspected = _inspect_regular(
            root / metadata["path"],
            label=f"retained fresh Git {name}",
        )
        if inspected != {
            "size_bytes": metadata["size_bytes"],
            "sha256": metadata["sha256"],
        }:
            raise ValueError("retained fresh-download Git bytes drifted")
    if (
        record["formal_file_count"] != EXPECTED_FORMAL_FILE_COUNT
        or record["formal_file_inventory_sha256"]
        != source["formal_file_inventory_sha256"]
        or record["formal_total_byte_count"] != source["formal_total_byte_count"]
        or record["remote_file_count"] != EXPECTED_REMOTE_FILE_COUNT
    ):
        raise ValueError("retained fresh-download inventory drifted")


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("receipt must have an absolute real existing parent")
    if path.exists() or path.is_symlink():
        raise FileExistsError("publication receipt already exists")
    payload = canonical_pretty_json_bytes(receipt)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("publication receipt write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if _read_regular(path, label="publication receipt") != payload:
        raise RuntimeError("publication receipt readback drifted")
    _validate_receipt_mode(path)


def _validate_receipt_mode(path: Path) -> None:
    if not path.is_absolute() or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("publication receipt path must be absolute and no-follow")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise ValueError("publication receipt is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ValueError("publication receipt mode must be exactly 0600")


def publish_processor_v2_artifact(
    *,
    api: Any,
    download_fn: Callable[..., str],
    operation_factory: Callable[..., Any],
    formal_root: str | Path,
    git_summary: str | Path,
    git_card: str | Path,
    expected_result_summary_sha256: str,
    hf_repo: str,
    hf_tag: str,
    hf_prefix: str,
    fresh_download_parent: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    prepared = prepare_processor_v2_publication(
        formal_root=formal_root,
        git_summary=git_summary,
        git_card=git_card,
        expected_result_summary_sha256=expected_result_summary_sha256,
        hf_repo=hf_repo,
        hf_tag=hf_tag,
        hf_prefix=hf_prefix,
    )
    parent = _validated_parent(fresh_download_parent, label="fresh-download parent")
    receipt = Path(receipt_path)
    formal = Path(prepared["source"]["formal_root"])
    if not receipt.is_absolute():
        raise ValueError("publication receipt path must be absolute")
    if receipt.is_relative_to(formal):
        raise ValueError("publication receipt must remain outside the formal root")
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError("publication receipt already exists")
    destination = prepared["destination"]
    repo = destination["repo"]
    tag = destination["tag"]
    prefix = destination["prefix"]
    main_parent = _revision(api, repo, "main")
    if _tag_snapshot(api, repo=repo, tag=tag) is not None:
        raise ValueError("HF publication tag already exists")
    if _remote_paths(api, repo=repo, revision="main", prefix=prefix):
        raise ValueError("HF publication prefix already exists")
    path_map = _remote_path_map(prepared)
    operations = [
        operation_factory(path_in_repo=remote, path_or_fileobj=str(local))
        for remote, local in sorted(path_map.items())
    ]
    response = api.create_commit(
        repo,
        repo_type=REPO_TYPE,
        revision="main",
        parent_commit=main_parent,
        operations=operations,
        commit_message=COMMIT_MESSAGE,
    )
    immutable = _commit_oid(response)
    if _revision(api, repo, "main") != immutable:
        raise ValueError("HF main does not resolve to the publication commit")
    expected_remote = set(path_map)
    if _remote_paths(api, repo=repo, revision=immutable, prefix=prefix) != (
        expected_remote
    ):
        raise ValueError("HF publication commit path inventory drifted")
    api.create_tag(
        repo,
        repo_type=REPO_TYPE,
        tag=tag,
        tag_message=TAG_MESSAGE,
        revision=immutable,
        exist_ok=False,
    )
    tag_snapshot = _tag_snapshot(api, repo=repo, tag=tag)
    if tag_snapshot is None or tag_snapshot["resolved_commit"] != immutable:
        raise ValueError("HF annotated tag does not resolve to publication commit")
    if _remote_paths(api, repo=repo, revision=tag, prefix=prefix) != expected_remote:
        raise ValueError("HF tagged publication path inventory drifted")
    stem = f"processor-v2-{prepared['source']['git_summary_sha256'][:12]}"
    commit_download = _download_publication(
        download_fn=download_fn,
        prepared=prepared,
        revision=immutable,
        directory=parent / f"{stem}-commit",
    )
    tag_download = _download_publication(
        download_fn=download_fn,
        prepared=prepared,
        revision=tag,
        directory=parent / f"{stem}-tag",
    )
    if _tag_snapshot(api, repo=repo, tag=tag) != tag_snapshot:
        raise ValueError("HF annotated tag drifted during fresh replay")
    if _snapshot_formal(formal) != tuple(
        prepared["source"]["formal_file_inventory"]
    ):
        raise RuntimeError("formal root changed during HF publication")
    if _sha256_bytes(
        _read_regular(
            Path(prepared["source"]["git_summary_path"]),
            label="Git result summary post-publication",
        )
    ) != prepared["source"]["git_summary_sha256"] or _sha256_bytes(
        _read_regular(
            Path(prepared["source"]["git_card_path"]),
            label="Git result card post-publication",
        )
    ) != prepared["source"]["git_card_sha256"]:
        raise RuntimeError("Git summary or card changed during HF publication")
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": PUBLICATION_STATUS,
        "source": prepared["source"],
        "destination": {
            **destination,
            "parent_main_revision": main_parent,
            "immutable_revision": immutable,
            "remote_file_count": EXPECTED_REMOTE_FILE_COUNT,
        },
        "remote": {
            "private_repo_verified": True,
            "single_commit_operation_count": EXPECTED_REMOTE_FILE_COUNT,
            "single_commit_verified": True,
            "annotated_tag_object_identity": tag_snapshot["object_identity"],
            "tag_resolved_commit": tag_snapshot["resolved_commit"],
            "no_overwrite_verified": True,
        },
        "fresh_downloads": {
            "commit": commit_download,
            "tag": tag_download,
            "formal_files_verified_twice": EXPECTED_FORMAL_FILE_COUNT,
            "remote_files_verified_twice": EXPECTED_REMOTE_FILE_COUNT,
        },
        "token_serialized": False,
    }
    _write_receipt(receipt, result)
    return result


def validate_processor_v2_publication(
    *,
    api: Any,
    formal_root: str | Path,
    git_summary: str | Path,
    git_card: str | Path,
    expected_result_summary_sha256: str,
    hf_repo: str,
    hf_tag: str,
    hf_prefix: str,
    fresh_download_parent: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    prepared = prepare_processor_v2_publication(
        formal_root=formal_root,
        git_summary=git_summary,
        git_card=git_card,
        expected_result_summary_sha256=expected_result_summary_sha256,
        hf_repo=hf_repo,
        hf_tag=hf_tag,
        hf_prefix=hf_prefix,
    )
    parent = _validated_parent(fresh_download_parent, label="fresh-download parent")
    receipt_file = Path(receipt_path)
    _validate_receipt_mode(receipt_file)
    receipt_payload = _read_regular(receipt_file, label="publication receipt")
    receipt = _strict_json_object(receipt_payload, label="publication receipt")
    if receipt_payload != canonical_pretty_json_bytes(receipt):
        raise ValueError("publication receipt is not canonical strict JSON")
    if set(receipt) != {
        "schema_version",
        "protocol_id",
        "status",
        "source",
        "destination",
        "remote",
        "fresh_downloads",
        "token_serialized",
    }:
        raise ValueError("publication receipt top-level fields drifted")
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or receipt["protocol_id"] != PROTOCOL_ID
        or receipt["status"] != PUBLICATION_STATUS
        or receipt["source"] != prepared["source"]
        or receipt["token_serialized"] is not False
    ):
        raise ValueError("publication receipt source or identity drifted")
    destination = receipt["destination"]
    if (
        not isinstance(destination, Mapping)
        or set(destination)
        != {
            "repo",
            "repo_type",
            "tag",
            "prefix",
            "parent_main_revision",
            "immutable_revision",
            "remote_file_count",
        }
        or any(
            destination.get(key) != value
            for key, value in prepared["destination"].items()
        )
        or destination.get("remote_file_count") != EXPECTED_REMOTE_FILE_COUNT
    ):
        raise ValueError("publication receipt destination drifted")
    _commit(
        destination.get("parent_main_revision"),
        label="receipt parent main revision",
    )
    immutable = _commit(
        destination.get("immutable_revision"), label="receipt immutable revision"
    )
    if _revision(api, hf_repo, immutable) != immutable:
        raise ValueError("remote immutable revision drifted")
    tag_snapshot = _tag_snapshot(api, repo=hf_repo, tag=hf_tag)
    remote = receipt["remote"]
    if (
        not isinstance(remote, Mapping)
        or set(remote)
        != {
            "private_repo_verified",
            "single_commit_operation_count",
            "single_commit_verified",
            "annotated_tag_object_identity",
            "tag_resolved_commit",
            "no_overwrite_verified",
        }
        or remote.get("private_repo_verified") is not True
        or remote.get("single_commit_operation_count")
        != EXPECTED_REMOTE_FILE_COUNT
        or remote.get("single_commit_verified") is not True
        or remote.get("no_overwrite_verified") is not True
        or tag_snapshot is None
        or tag_snapshot["resolved_commit"] != immutable
        or remote.get("annotated_tag_object_identity")
        != tag_snapshot["object_identity"]
        or remote.get("tag_resolved_commit") != immutable
    ):
        raise ValueError("remote annotated tag drifted from publication receipt")
    expected_paths = set(_remote_path_map(prepared))
    for revision in (immutable, hf_tag):
        if _remote_paths(api, repo=hf_repo, revision=revision, prefix=hf_prefix) != (
            expected_paths
        ):
            raise ValueError("remote publication path inventory drifted")
    fresh = receipt["fresh_downloads"]
    if not isinstance(fresh, Mapping) or set(fresh) != {
        "commit",
        "tag",
        "formal_files_verified_twice",
        "remote_files_verified_twice",
    }:
        raise ValueError("publication receipt fresh-download section drifted")
    if fresh["formal_files_verified_twice"] != EXPECTED_FORMAL_FILE_COUNT:
        raise ValueError("publication receipt fresh-download count drifted")
    if fresh["remote_files_verified_twice"] != EXPECTED_REMOTE_FILE_COUNT:
        raise ValueError("publication receipt remote replay count drifted")
    if Path(fresh["commit"]["directory"]).parent != parent or Path(
        fresh["tag"]["directory"]
    ).parent != parent:
        raise ValueError("publication receipt fresh-download parent drifted")
    if fresh["commit"]["revision"] != immutable or fresh["tag"]["revision"] != hf_tag:
        raise ValueError("publication receipt fresh-download revisions drifted")
    _verify_existing_download(fresh["commit"], prepared=prepared)
    _verify_existing_download(fresh["tag"], prepared=prepared)
    return receipt


__all__ = [
    "COMMIT_MESSAGE",
    "EXPECTED_REMOTE_FILE_COUNT",
    "PROTOCOL_ID",
    "PUBLICATION_STATUS",
    "TAG_MESSAGE",
    "prepare_processor_v2_publication",
    "publish_processor_v2_artifact",
    "validate_processor_v2_publication",
]
