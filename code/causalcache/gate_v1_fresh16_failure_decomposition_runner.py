"""Selective immutable replay and publication for fresh-16 failure analysis."""

from __future__ import annotations

import io
import json
import math
import os
import re
import stat
import struct
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_gate_v1_fresh16_failure_decomposition_v1"
RUN_STATUS = "COMPLETED_GATE_V1_FRESH16_FAILURE_DECOMPOSITION_V1"
VALIDATE_STATUS = "REVALIDATED_GATE_V1_FRESH16_FAILURE_DECOMPOSITION_V1"
BUNDLE_STATUS = "FROZEN_GATE_V1_FRESH16_FAILURE_DECOMPOSITION_BUNDLE_V1"

PARENT_REPO = (
    "gavinlaw/causalcache-gate-v1-fresh16-claim-serialization-repair-mobile"
)
PARENT_REVISION = "3541fe1ea2c46e555c29cc53483e6f3b809f8f81"
PARENT_TAG = "gate-v1-fresh16-claim-serialization-repair-v1"
PARENT_TAG_OBJECT = "34d5928db4e82532f46f7596702aec3f018c4339"
PARENT_BUNDLE_PATH = "fresh16-eval/v1/manifests/bundle-manifest-v1.json"
PARENT_LABEL_PATH = "fresh16-eval/v1/caches/label-states-v1.jsonl"
PARENT_STATE_RECORDS_PATH = (
    "fresh16-eval/v1/reports/primary-state-records-v1.jsonl"
)
PARENT_TARGETS = (
    PARENT_BUNDLE_PATH,
    PARENT_LABEL_PATH,
    PARENT_STATE_RECORDS_PATH,
)
PARENT_BINDINGS = {
    PARENT_BUNDLE_PATH: {
        "size_bytes": 2133,
        "sha256": "a11e9d693c2fa33f5f5f97b123649608bdf98583693caa818e166d8009382844",
    },
    PARENT_LABEL_PATH: {
        "size_bytes": 48134,
        "sha256": "832096b98011264a6b7fee74b2e78798876ea0d5ffc5c9c9baf7642be8e92382",
    },
    PARENT_STATE_RECORDS_PATH: {
        "size_bytes": 128914,
        "sha256": "9aaf8ffe217872986d9f87f88ed55e92c2742bebce944f8668de9d597dc0ac3f",
    },
}

CHILD_STATE_PATH = (
    "fresh16-failure-decomposition/v1/state-decomposition-v1.jsonl"
)
CHILD_REPORT_PATH = (
    "fresh16-failure-decomposition/v1/failure-decomposition-report-v1.json"
)
CHILD_BUNDLE_PATH = (
    "fresh16-failure-decomposition/v1/bundle-manifest-v1.json"
)
CHILD_TARGETS = (CHILD_STATE_PATH, CHILD_REPORT_PATH, CHILD_BUNDLE_PATH)
ORDERED_LOCAL_STATES = (
    "global_claim",
    "parent_readonly_attestation",
    "local_report_completion",
    "child_remote_base_receipt",
    "child_report_commit_receipt",
    "child_tag_receipt",
    "immutable_replay_completion",
    "completion_staging",
    "final_completion",
)

_ALLOWED_BASE_FILES = frozenset({".gitattributes", "README.md"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_F64_HEX = re.compile(r"[0-9a-f]{16}")


@dataclass(frozen=True)
class ParentExactThree:
    files: Mapping[str, bytes]
    manifest: Mapping[str, Any]
    tag_object_identity: str
    identity: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedFailureDecomposition:
    files: Mapping[str, bytes]
    state_decomposition: bytes
    report: bytes
    bundle_manifest: Mapping[str, Any]


@dataclass(frozen=True)
class ChildRemoteState:
    main_revision: str
    target_state: str
    tag_object_identity: str | None
    tag_resolved_commit: str | None


class _LocalStateMachine:
    def __init__(self, contract: Any, *, mode: str):
        local = _section(contract, ("local_first_state_machine", "local_state"))
        state_root = local.get("state_directory")
        if not isinstance(state_root, str) or not Path(state_root).is_absolute():
            raise ValueError("local state directory contract is malformed")
        if tuple(local.get("ordered_states", ())) != ORDERED_LOCAL_STATES:
            raise ValueError("local ordered-state contract drifted")
        if local.get("state_file_mode") != 0o600:
            raise ValueError("local state mode contract drifted")
        self.root = Path(state_root)
        self.mode = mode
        self.previous_sha256: str | None = None
        self.next_ordinal = 0
        if mode == "run":
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        elif not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("completed local state directory is missing or unsafe")
        self.root = self.root.resolve()

    def path(self, ordinal: int, name: str) -> Path:
        return self.root / f"{ordinal:02d}-{name.replace('_', '-')}.json"

    def write(self, name: str, value: Mapping[str, Any]) -> None:
        if self.mode != "run":
            return
        if self.next_ordinal >= 7 or ORDERED_LOCAL_STATES[self.next_ordinal] != name:
            raise ValueError("local state transition is out of frozen order")
        record = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "DURABLE_GATE_V1_FRESH16_FAILURE_DECOMPOSITION_STATE_V1",
            "ordinal": self.next_ordinal,
            "name": name,
            "previous_state_sha256": self.previous_sha256,
            "payload": dict(value),
        }
        payload = pretty_json_bytes(record)
        path = self.path(self.next_ordinal, name)
        _exclusive_or_identical(path, payload, mode=0o600)
        self.previous_sha256 = sha256_bytes(payload)
        self.next_ordinal += 1

    def complete(self, value: Mapping[str, Any]) -> None:
        if self.mode != "run":
            self.validate()
            return
        if self.next_ordinal != 7:
            raise ValueError("completion preceded a required local state")
        record = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": RUN_STATUS,
            "ordinal": 8,
            "name": "final_completion",
            "previous_state_sha256": self.previous_sha256,
            "payload": dict(value),
        }
        payload = pretty_json_bytes(record)
        staging = self.path(7, "completion_staging")
        final = self.path(8, "final_completion")
        _exclusive_or_identical(staging, payload, mode=0o600)
        if final.exists() or final.is_symlink():
            if (
                _regular_file_bytes(final, label="final completion") != payload
                or final.stat().st_ino != staging.stat().st_ino
                or final.stat().st_dev != staging.stat().st_dev
            ):
                raise ValueError("final completion is not the retained staging hard link")
        else:
            os.link(staging, final, follow_symlinks=False)
            _fsync_directory(self.root)
        self.next_ordinal = 9
        self.validate()

    def validate(self) -> None:
        expected = {
            self.path(index, name) for index, name in enumerate(ORDERED_LOCAL_STATES)
        }
        observed = set(self.root.iterdir())
        if observed != expected:
            raise ValueError("local ordered-state tree is not exact-nine")
        previous: str | None = None
        for index, name in enumerate(ORDERED_LOCAL_STATES[:7]):
            path = self.path(index, name)
            if stat.S_IMODE(path.stat().st_mode) != 0o600:
                raise ValueError("local ordered-state mode drifted")
            payload = _regular_file_bytes(path, label=name)
            record = _strict_json(payload, label=name)
            if (
                record.get("ordinal") != index
                or record.get("name") != name
                or record.get("previous_state_sha256") != previous
            ):
                raise ValueError("local ordered-state chain drifted")
            previous = sha256_bytes(payload)
        staging = self.path(7, "completion_staging")
        final = self.path(8, "final_completion")
        if (
            stat.S_IMODE(staging.stat().st_mode) != 0o600
            or stat.S_IMODE(final.stat().st_mode) != 0o600
            or staging.stat().st_ino != final.stat().st_ino
            or staging.stat().st_dev != final.stat().st_dev
            or _regular_file_bytes(staging, label="completion staging")
            != _regular_file_bytes(final, label="final completion")
        ):
            raise ValueError("final completion hard-link identity drifted")
        completion = _strict_json(
            _regular_file_bytes(final, label="final completion"),
            label="final completion",
        )
        if (
            completion.get("status") != RUN_STATUS
            or completion.get("ordinal") != 8
            or completion.get("name") != "final_completion"
            or completion.get("previous_state_sha256") != previous
        ):
            raise ValueError("final completion chain binding drifted")


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _contract_data(contract: Any) -> Mapping[str, Any]:
    return _mapping(getattr(contract, "data", contract), "failure decomposition contract")


def _section(
    contract: Any,
    names: Sequence[str],
    *,
    required: bool = True,
) -> Mapping[str, Any]:
    for name in names:
        candidate = getattr(contract, name, None)
        if candidate is not None:
            return _mapping(candidate, name)
    data = _contract_data(contract)
    for name in names:
        if name in data:
            return _mapping(data[name], name)
    if required:
        raise ValueError(f"contract is missing section {tuple(names)!r}")
    return {}


def _one_of(
    value: Mapping[str, Any],
    names: Sequence[str],
    *,
    label: str,
    default: Any = None,
) -> Any:
    present = [name for name in names if name in value]
    if len(present) > 1:
        observed = [value[name] for name in present]
        if any(item != observed[0] for item in observed[1:]):
            raise ValueError(f"{label} has conflicting aliases: {present}")
        return observed[0]
    if present:
        return value[present[0]]
    if default is not None:
        return default
    raise ValueError(f"{label} is missing one of {tuple(names)!r}")


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be nonempty text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} has duplicate key {key!r}")
            result[key] = item
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"{label} has non-finite constant {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    return _mapping(value, label)


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
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
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    payload = b"".join(chunks)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(payload) != after.st_size:
        raise ValueError(f"{label} changed while being read")
    return payload


def _fresh_parent(value: str | Path) -> Path:
    path = Path(value)
    try:
        current = path.lstat()
    except OSError as error:
        raise ValueError("fresh-download parent is missing") from error
    if not path.is_absolute() or not stat.S_ISDIR(current.st_mode) or path.is_symlink():
        raise ValueError("fresh-download parent must be an absolute non-symlink directory")
    return path.resolve()


def _parent_section(contract: Any) -> Mapping[str, Any]:
    data = _contract_data(contract)
    if "parent_primary" in data:
        parent = _mapping(data["parent_primary"], "parent primary")
        return _mapping(parent.get("hf"), "parent primary HF")
    return _section(contract, ("parent_input", "sealed_parent", "parent_artifact", "parent"))


def _destination(contract: Any) -> Mapping[str, Any]:
    destination = _section(contract, ("destination", "output_destination"))
    repo = _one_of(destination, ("repo", "repository"), label="destination repo")
    if not isinstance(repo, str) or "/" not in repo:
        raise ValueError("destination repo is malformed")
    if destination.get("repo_type", "dataset") != "dataset":
        raise ValueError("failure decomposition destination must be a dataset repo")
    if destination.get("private") is not True:
        raise ValueError("failure decomposition destination must be private")
    output = _section(contract, ("output_contract", "output"), required=False)
    target_source = output if output else destination
    targets = _one_of(
        target_source,
        ("exact_three_targets", "exact_targets", "targets"),
        label="destination targets",
        default=list(CHILD_TARGETS),
    )
    if not isinstance(targets, list) or tuple(targets) != CHILD_TARGETS:
        raise ValueError("destination must bind the frozen child exact-three targets")
    tag = _one_of(destination, ("tag", "immutable_tag"), label="destination tag")
    if not isinstance(tag, str) or not tag:
        raise ValueError("destination tag is malformed")
    return destination


def validate_parent_contract(contract: Any) -> Mapping[str, Any]:
    parent = _parent_section(contract)
    repo = _one_of(parent, ("repo", "repository"), label="parent repo")
    revision = _one_of(
        parent,
        ("immutable_revision", "revision", "report_commit", "tag_resolved_commit"),
        label="parent revision",
    )
    tag = _one_of(parent, ("tag", "immutable_tag"), label="parent tag")
    input_contract = _section(contract, ("input_contract", "input"), required=False)
    raw_targets = input_contract.get("exact_force_download_targets") if input_contract else None
    if raw_targets is None:
        targets = _one_of(
            parent,
            ("exact_three_targets", "exact_targets", "targets"),
            label="parent targets",
            default=list(PARENT_TARGETS),
        )
    else:
        if not isinstance(raw_targets, list) or any(
            not isinstance(item, Mapping) for item in raw_targets
        ):
            raise ValueError("parent selective target contract is malformed")
        targets = [item.get("path") for item in raw_targets]
    if (
        repo != PARENT_REPO
        or revision != PARENT_REVISION
        or tag != PARENT_TAG
        or parent.get("repo_type", "dataset") != "dataset"
        or tuple(targets) != PARENT_TARGETS
    ):
        raise ValueError("fresh-16 parent immutable identity drifted")
    if parent.get("annotated_tag_object", PARENT_TAG_OBJECT) != PARENT_TAG_OBJECT:
        raise ValueError("fresh-16 parent annotated-tag object drifted")
    configured_bindings = parent.get("target_bindings", parent.get("bindings"))
    if configured_bindings is not None:
        if isinstance(configured_bindings, list):
            configured_bindings = {
                item.get("path"): {
                    "size_bytes": item.get("size_bytes"),
                    "sha256": item.get("sha256"),
                }
                for item in configured_bindings
                if isinstance(item, Mapping)
            }
        if dict(_mapping(configured_bindings, "parent bindings")) != PARENT_BINDINGS:
            raise ValueError("fresh-16 parent exact-three bindings drifted")
    return parent


def _revision_info(api: Any, repo: str, revision: str) -> Any:
    info = api.dataset_info(repo, revision=revision)
    observed = getattr(info, "sha", None)
    if not isinstance(observed, str) or _COMMIT.fullmatch(observed) is None:
        raise ValueError(f"HF revision is not immutable: {revision}")
    if getattr(info, "private", True) is not True:
        raise ValueError("HF dataset unexpectedly became public")
    return info


def _revision_sha(api: Any, repo: str, revision: str) -> str:
    return getattr(_revision_info(api, repo, revision), "sha")


def _tag_snapshot(api: Any, *, repo: str, tag: str) -> tuple[str, str] | None:
    refs = api.list_repo_refs(repo, repo_type="dataset")
    matches = [item for item in refs.tags if getattr(item, "name", None) == tag]
    if len(matches) > 1:
        raise ValueError("HF tag is duplicated")
    if not matches:
        return None
    object_identity = getattr(matches[0], "target_commit", None)
    resolved = _revision_sha(api, repo, tag)
    if (
        not isinstance(object_identity, str)
        or _COMMIT.fullmatch(object_identity) is None
        or object_identity == resolved
    ):
        raise ValueError("failure decomposition requires an annotated tag")
    return object_identity, resolved


def _parent_identity_snapshot(api: Any, contract: Any) -> Mapping[str, Any]:
    parent = _parent_section(contract)
    repo = _one_of(parent, ("repo", "repository"), label="parent repo")
    revision = _one_of(
        parent,
        ("immutable_revision", "revision", "report_commit", "tag_resolved_commit"),
        label="parent revision",
    )
    tag = _one_of(parent, ("tag", "immutable_tag"), label="parent tag")
    resolved = _revision_sha(api, repo, revision)
    snapshot = _tag_snapshot(api, repo=repo, tag=tag)
    if (
        resolved != PARENT_REVISION
        or snapshot != (PARENT_TAG_OBJECT, PARENT_REVISION)
    ):
        raise ValueError("parent immutable revision/tag identity drifted")
    files = tuple(
        sorted(
            _safe_relative(item, "parent remote file")
            for item in api.list_repo_files(
                repo,
                repo_type="dataset",
                revision=revision,
            )
        )
    )
    commits = tuple(
        getattr(item, "commit_id", None)
        for item in api.list_repo_commits(
            repo,
            repo_type="dataset",
            revision=revision,
        )
    )
    if (
        len(files) != len(set(files))
        or not set(PARENT_TARGETS).issubset(files)
        or len(commits) < 2
        or commits[:2]
        != (
            PARENT_REVISION,
            "9f0c61b9437773ca5d3f7e0cabd6e2a987e3c908",
        )
        or any(not isinstance(item, str) or _COMMIT.fullmatch(item) is None for item in commits)
    ):
        raise ValueError("parent immutable file/history identity drifted")
    value = {
        "repo": repo,
        "revision": resolved,
        "tag": tag,
        "tag_object_identity": snapshot[0],
        "remote_files": list(files),
        "reachable_commits": list(commits),
    }
    return {**value, "identity_sha256": sha256_bytes(canonical_json_bytes(value))}


def _assert_download_path(root: Path, returned: str | Path, relative: str) -> Path:
    path = Path(returned)
    expected = root / relative
    if not path.is_absolute() or path != expected:
        raise ValueError("HF download returned a noncanonical path")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("HF download escaped its fresh root") from error
    current = root
    for component in PurePosixPath(relative).parts[:-1]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise ValueError("HF download parent is missing") from error
        if not stat.S_ISDIR(mode):
            raise ValueError("HF download traversed a symlink or non-directory")
    return path


def _download_exact_three(
    *,
    download_fn: Callable[..., str],
    repo: str,
    revision: str,
    targets: Sequence[str],
    fresh_parent: Path,
    prefix: str,
) -> Mapping[str, bytes]:
    if tuple(dict.fromkeys(targets)) != tuple(targets) or len(targets) != 3:
        raise ValueError("selective replay requires exactly three unique targets")
    with tempfile.TemporaryDirectory(prefix=prefix, dir=fresh_parent) as raw:
        root = Path(raw).resolve()
        if root.is_symlink() or any(root.iterdir()):
            raise ValueError("fresh replay root did not start empty")
        files: dict[str, bytes] = {}
        for target in targets:
            relative = _safe_relative(target, "HF target")
            returned = download_fn(
                repo_id=repo,
                repo_type="dataset",
                filename=relative,
                revision=revision,
                local_dir=root,
                force_download=True,
            )
            path = _assert_download_path(root, returned, relative)
            files[relative] = _regular_file_bytes(path, label=relative)
    return files


def _inventory(files: Mapping[str, bytes]) -> list[Mapping[str, Any]]:
    return [
        {
            "path": path,
            "size_bytes": len(files[path]),
            "sha256": sha256_bytes(files[path]),
        }
        for path in sorted(files)
    ]


def _validate_parent_bundle(files: Mapping[str, bytes]) -> Mapping[str, Any]:
    if tuple(files) != PARENT_TARGETS:
        raise ValueError("parent selective file order or denominator drifted")
    observed = {
        path: {"size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        for path, payload in files.items()
    }
    if observed != PARENT_BINDINGS:
        raise ValueError("parent exact-three bytes differ from the immutable binding")
    manifest = _strict_json(files[PARENT_BUNDLE_PATH], label="parent bundle manifest")
    if (
        manifest.get("schema_version") != "1.0.0"
        or manifest.get("protocol_id")
        != "causalcache_gate_v1_fresh16_evaluation_v1"
        or manifest.get("status") != "FROZEN_GATE_V1_FRESH16_BUNDLE_MANIFEST_V1"
        or manifest.get("final_target_count") != 13
        or manifest.get("report_commit_embedded") is not False
    ):
        raise ValueError("parent bundle manifest identity drifted")
    bound: dict[str, Mapping[str, Any]] = {}
    for key in ("payload_files", "report_files_excluding_bundle"):
        rows = manifest.get(key)
        if not isinstance(rows, list):
            raise ValueError("parent bundle manifest inventory is malformed")
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "path",
                "size_bytes",
                "sha256",
            }:
                raise ValueError("parent bundle manifest record is malformed")
            path = row["path"]
            if path in bound:
                raise ValueError("parent bundle manifest has duplicate paths")
            bound[path] = row
    for path in (PARENT_LABEL_PATH, PARENT_STATE_RECORDS_PATH):
        expected = {"path": path, **PARENT_BINDINGS[path]}
        if dict(bound.get(path, {})) != expected:
            raise ValueError("parent manifest does not bind a selectively read payload")
    return manifest


def read_parent_exact_three(
    *,
    contract: Any,
    api: Any,
    download_fn: Callable[..., str],
    fresh_download_parent: str | Path,
) -> ParentExactThree:
    validate_parent_contract(contract)
    parent = _parent_section(contract)
    repo = _one_of(parent, ("repo", "repository"), label="parent repo")
    tag = _one_of(parent, ("tag", "immutable_tag"), label="parent tag")
    revision = _one_of(
        parent,
        ("immutable_revision", "revision", "report_commit", "tag_resolved_commit"),
        label="parent revision",
    )
    before = _parent_identity_snapshot(api, contract)
    files = _download_exact_three(
        download_fn=download_fn,
        repo=repo,
        revision=revision,
        targets=PARENT_TARGETS,
        fresh_parent=_fresh_parent(fresh_download_parent),
        prefix="fresh16-decomposition-parent-",
    )
    manifest = _validate_parent_bundle(files)
    after = _parent_identity_snapshot(api, contract)
    if before != after:
        raise ValueError("parent immutable identity changed across selective replay")
    return ParentExactThree(
        files=files,
        manifest=manifest,
        tag_object_identity=PARENT_TAG_OBJECT,
        identity=before,
    )


def _read_label_states_jsonl(payload: bytes) -> tuple[Any, ...]:
    from causalcache.gate_v1_data import LabelState
    from causalcache.restoration_v2_2_label_table import (
        validate_complete_distance_table,
    )

    if not payload.endswith(b"\n"):
        raise ValueError("parent label-state JSONL must be LF-terminated")
    lines = payload[:-1].split(b"\n")
    if len(lines) != 48 or any(not line for line in lines):
        raise ValueError("parent label-state denominator must be exactly 48")
    states = []
    distance_count = 0
    for ordinal, line in enumerate(lines):
        record = _strict_json(line, label=f"parent label state {ordinal}")
        if (
            canonical_json_bytes(record) != line
            or record.get("schema_version") != "1.0.0"
            or record.get("protocol_id")
            != "causalcache_gate_v1_fresh16_primary_v1"
            or record.get("status") != "FROZEN_GATE_V1_FRESH16_LABEL_STATES_V1"
            or record.get("ordinal") != ordinal
            or not isinstance(record.get("candidate_event_step_ids"), list)
            or not isinstance(record.get("distance_rows"), list)
        ):
            raise ValueError("parent label-state identity or encoding drifted")
        distances: dict[tuple[int, ...], float] = {}
        for row in record["distance_rows"]:
            item = _mapping(row, "parent label distance row")
            if set(item) != {
                "coalition_event_step_ids",
                "distance_kl_f64_hex",
            }:
                raise ValueError("parent label distance-row schema drifted")
            coalition = item["coalition_event_step_ids"]
            encoded = item["distance_kl_f64_hex"]
            if (
                not isinstance(coalition, list)
                or not isinstance(encoded, str)
                or _F64_HEX.fullmatch(encoded) is None
            ):
                raise ValueError("parent label distance-row value drifted")
            key = tuple(coalition)
            if key in distances:
                raise ValueError("parent label table has a duplicate coalition")
            distance = struct.unpack(">d", bytes.fromhex(encoded))[0]
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError("parent label distance is not finite nonnegative")
            distances[key] = distance
        table = validate_complete_distance_table(
            record["candidate_event_step_ids"], distances
        )
        distance_count += len(table.rows)
        states.append(
            LabelState(
                source_id=record["source_id"],
                state_id=record["state_id"],
                decision_step_id=record["decision_step_id"],
                table=table,
            )
        )
    if distance_count != 448:
        raise ValueError("parent label distance denominator drifted")
    return tuple(states)


def _lazy_reducer() -> Callable[..., Any]:
    from causalcache.gate_v1_fresh16_failure_decomposition import (
        build_failure_decomposition,
    )

    def reduce(label_payload: bytes, state_payload: bytes, *, contract: Any) -> Any:
        labels = _read_label_states_jsonl(label_payload)
        if not state_payload.endswith(b"\n"):
            raise ValueError("parent state-record JSONL must be LF-terminated")
        lines = state_payload[:-1].split(b"\n")
        if len(lines) != 48 or any(not line for line in lines):
            raise ValueError("parent state-record denominator must be exactly 48")
        records = tuple(
            _strict_json(line, label=f"parent state record {index}")
            for index, line in enumerate(lines)
        )
        return build_failure_decomposition(labels, records, contract=contract)

    return reduce


def _as_payload(value: Any, *, label: str, json_object: bool) -> bytes:
    if isinstance(value, bytes):
        payload = value
    elif json_object and isinstance(value, Mapping):
        payload = pretty_json_bytes(value)
    else:
        raise TypeError(f"{label} must be bytes" + (" or a mapping" if json_object else ""))
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be nonempty and LF-terminated")
    if json_object:
        _strict_json(payload, label=label)
    else:
        lines = payload[:-1].split(b"\n")
        if len(lines) != 48 or any(not line for line in lines):
            raise ValueError("state decomposition must contain exactly 48 JSONL rows")
        for index, line in enumerate(lines):
            _strict_json(line, label=f"state decomposition row {index}")
    return payload


def _normalize_reducer_result(value: Any) -> tuple[bytes, bytes]:
    state: Any = None
    report: Any = None
    if isinstance(value, Mapping):
        if "state_rows" in value:
            rows = value["state_rows"]
            if not isinstance(rows, list) or len(rows) != 48:
                raise ValueError("reducer state_rows denominator must be exactly 48")
            state_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
            expected_sha = value.get("state_rows_sha256")
            if expected_sha is not None and expected_sha != sha256_bytes(state_payload):
                raise ValueError("reducer state_rows SHA256 differs from JSONL encoding")
            report_value = dict(value)
            report_value.pop("state_rows")
            report_value["state_rows_sha256"] = sha256_bytes(state_payload)
            report_value["state_rows_size_bytes"] = len(state_payload)
            report_value["state_rows_record_count"] = 48
            return state_payload, canonical_json_bytes(report_value) + b"\n"
        state = value.get(CHILD_STATE_PATH)
        report = value.get(CHILD_REPORT_PATH)
        if state is None:
            state = value.get("state_decomposition", value.get("state_decomposition_bytes"))
        if report is None:
            report = value.get("report", value.get("report_bytes"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) == 2:
            state, report = value
    else:
        for name in ("state_decomposition", "state_decomposition_bytes"):
            if hasattr(value, name):
                state = getattr(value, name)
                break
        for name in ("report", "report_bytes"):
            if hasattr(value, name):
                report = getattr(value, name)
                break
    return (
        _as_payload(state, label="state decomposition", json_object=False),
        _as_payload(report, label="failure decomposition report", json_object=True),
    )


def _validate_reducer_outputs(
    *,
    state_payload: bytes,
    report_payload: bytes,
    contract: Any,
) -> None:
    state_lines = state_payload[:-1].split(b"\n")
    state_ids: set[str] = set()
    source_ids: set[str] = set()
    for ordinal, line in enumerate(state_lines):
        record = _strict_json(line, label=f"state decomposition row {ordinal}")
        if (
            canonical_json_bytes(record) != line
            or record.get("schema_version") != SCHEMA_VERSION
            or record.get("protocol_id") != PROTOCOL_ID
            or record.get("status")
            != "FROZEN_GATE_V1_FRESH16_FAILURE_DECOMPOSITION_V1"
            or record.get("ordinal") != ordinal
            or not isinstance(record.get("source_id"), str)
            or not isinstance(record.get("state_id"), str)
        ):
            raise ValueError("state decomposition identity or encoding drifted")
        if record["state_id"] in state_ids:
            raise ValueError("state decomposition contains duplicate state ids")
        state_ids.add(record["state_id"])
        source_ids.add(record["source_id"])
    if len(state_ids) != 48 or len(source_ids) != 16:
        raise ValueError("state decomposition state/source denominator drifted")

    report = _strict_json(report_payload, label="failure decomposition report")
    expected_contract_sha = getattr(contract, "sha256", None)
    trajectory_rows = report.get("trajectory_rows")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("status")
        != "FROZEN_GATE_V1_FRESH16_FAILURE_DECOMPOSITION_V1"
        or report.get("contract_sha256") != expected_contract_sha
        or report.get("source_count") != 16
        or report.get("state_count") != 48
        or report.get("state_rows_record_count") != 48
        or report.get("state_rows_sha256") != sha256_bytes(state_payload)
        or report.get("state_rows_size_bytes") != len(state_payload)
        or not isinstance(trajectory_rows, list)
        or len(trajectory_rows) != 16
    ):
        raise ValueError("failure decomposition report identity or denominator drifted")
    trajectory_payload = b"".join(
        canonical_json_bytes(_mapping(row, "trajectory decomposition row")) + b"\n"
        for row in trajectory_rows
    )
    if (
        len({row.get("source_id") for row in trajectory_rows}) != 16
        or report.get("trajectory_rows_sha256") != sha256_bytes(trajectory_payload)
        or report.get("trajectory_rows_size_bytes") != len(trajectory_payload)
    ):
        raise ValueError("trajectory decomposition inventory drifted")
    decision = _mapping(report.get("decision_tree"), "failure routing decision")
    routing = _section(contract, ("routing_contract",))
    final = _mapping(routing.get("final_routing"), "final routing contract")
    if (
        decision.get("outcome")
        not in {final.get("pass_status"), final.get("fail_status")}
        or decision.get("v1_verdict_unchanged") is not True
        or decision.get("fresh16_is_consumed_diagnostic_only") is not True
        or decision.get("confirm_matched_nll_closed_loop_authorized") is not False
    ):
        raise ValueError("failure decomposition routing boundary drifted")
    operations = _mapping(report.get("operation_counts"), "reducer operations")
    if (
        operations.get("scope")
        != "pure_reducer_after_two_sealed_48_record_decodes"
        or operations.get("sealed_fresh16_label_state_decode_count") != 48
        or operations.get("sealed_parent_state_record_decode_count") != 48
        or any(
            operations.get(name) != 0
            for name in (
                "file_read",
                "file_write",
                "network",
                "torch_import",
                "gpu",
                "model_load",
                "model_forward",
                "policy_forward",
                "fresh16_raw_source_access_count",
                "upstream_raw_label_read_count",
                "legacy_dev5_semantic_decode_count",
                "confirm20_access_count",
                "matched_nll_evaluation_count",
                "closed_loop_episode_count",
                "gate_training_step_count",
                "hf_mutation",
            )
        )
    ):
        raise ValueError("failure decomposition reducer operation boundary drifted")


def _build_child_manifest(
    *,
    contract: Any,
    parent: ParentExactThree,
    state_payload: bytes,
    report_payload: bytes,
    source_execution: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    outputs = {
        CHILD_STATE_PATH: state_payload,
        CHILD_REPORT_PATH: report_payload,
    }
    execution = _section(
        contract,
        ("execution_fixed_operation_contract", "execution_operations"),
        required=False,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": BUNDLE_STATUS,
        "exact_target_count": 3,
        "report_commit_embedded": False,
        "contract_sha256": getattr(contract, "sha256", None),
        "source_execution": dict(source_execution or {}),
        "parent": {
            "repo": PARENT_REPO,
            "repo_type": "dataset",
            "immutable_revision": PARENT_REVISION,
            "tag": PARENT_TAG,
            "tag_object_identity": parent.tag_object_identity,
            "selective_target_count": 3,
            "selective_files": _inventory(parent.files),
            "identity_sha256": parent.identity["identity_sha256"],
        },
        "output_files_excluding_bundle": _inventory(outputs),
        "execution_fixed_operation_contract": dict(execution),
        "prohibited_access": {
            "raw_image_read_count": 0,
            "gui_owl_model_load_count": 0,
            "gate_checkpoint_load_count": 0,
            "upstream_raw_label_read_count": 0,
            "legacy_dev5_semantic_decode_count": 0,
            "confirm20_access_count": 0,
            "matched_nll_evaluation_count": 0,
            "closed_loop_episode_count": 0,
        },
    }


def prepare_failure_decomposition(
    *,
    contract: Any,
    parent: ParentExactThree,
    reducer: Callable[..., Any] | None = None,
    source_execution: Mapping[str, Any] | None = None,
) -> PreparedFailureDecomposition:
    reducer_fn = reducer if reducer is not None else _lazy_reducer()
    reduced = reducer_fn(
        parent.files[PARENT_LABEL_PATH],
        parent.files[PARENT_STATE_RECORDS_PATH],
        contract=contract,
    )
    state_payload, report_payload = _normalize_reducer_result(reduced)
    _validate_reducer_outputs(
        state_payload=state_payload,
        report_payload=report_payload,
        contract=contract,
    )
    manifest = _build_child_manifest(
        contract=contract,
        parent=parent,
        state_payload=state_payload,
        report_payload=report_payload,
        source_execution=source_execution,
    )
    files = {
        CHILD_STATE_PATH: state_payload,
        CHILD_REPORT_PATH: report_payload,
        CHILD_BUNDLE_PATH: pretty_json_bytes(manifest),
    }
    return PreparedFailureDecomposition(
        files=files,
        state_decomposition=state_payload,
        report=report_payload,
        bundle_manifest=manifest,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_or_identical(path: Path, payload: bytes, *, mode: int = 0o444) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    current = path.parent
    while current != current.parent:
        if current.exists() and (current.is_symlink() or not current.is_dir()):
            raise ValueError("local output traverses a symlink or non-directory")
        current = current.parent
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
    except FileExistsError:
        if _regular_file_bytes(path, label=str(path)) != payload:
            raise ValueError("retained local output differs from prepared bytes")
        return False
    except OSError as error:
        raise ValueError("local output cannot be created safely") from error
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, mode, follow_symlinks=False)
    _fsync_directory(path.parent)
    return True


def stage_exact_three(
    directory: str | Path,
    prepared: PreparedFailureDecomposition,
    *,
    allow_create: bool,
) -> Mapping[str, Any]:
    root = Path(directory)
    if not root.is_absolute():
        raise ValueError("local output directory must be absolute")
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ValueError("local output directory is unsafe")
    elif allow_create:
        root.mkdir(parents=True, mode=0o700)
    else:
        raise ValueError("completed local output directory is missing")
    root = root.resolve()
    writes = 0
    for relative in CHILD_TARGETS:
        target = root / relative
        if allow_create:
            writes += int(_exclusive_or_identical(target, prepared.files[relative]))
        elif _regular_file_bytes(target, label=relative) != prepared.files[relative]:
            raise ValueError("completed local exact-three output drifted")
        if stat.S_IMODE(target.stat().st_mode) != 0o444:
            raise ValueError("local exact-three output mode drifted")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed != set(CHILD_TARGETS):
        raise ValueError("local output directory is not exact-three")
    allowed_directories = {
        parent.as_posix()
        for target in CHILD_TARGETS
        for parent in PurePosixPath(target).parents
        if parent.as_posix() != "."
    }
    observed_directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if observed_directories != allowed_directories:
        raise ValueError("local output directory contains extra or missing directories")
    return {"local_write_count": writes, "local_files": _inventory(prepared.files)}


def _destination_values(contract: Any) -> tuple[str, str, str, str]:
    destination = _destination(contract)
    repo = _one_of(destination, ("repo", "repository"), label="destination repo")
    tag = _one_of(destination, ("tag", "immutable_tag"), label="destination tag")
    message = _one_of(
        destination,
        ("commit_message", "commit_title"),
        label="destination commit message",
        default="Publish CausalCache fresh-16 failure decomposition",
    )
    if not isinstance(message, str) or not message:
        raise ValueError("destination commit message is malformed")
    tag_message = _one_of(
        destination,
        ("tag_message",),
        label="destination tag message",
    )
    if not isinstance(tag_message, str) or not tag_message:
        raise ValueError("destination tag message is malformed")
    return repo, tag, message, tag_message


def _remote_target_state(
    api: Any,
    *,
    repo: str,
    revision: str,
) -> str:
    files = tuple(api.list_repo_files(repo, repo_type="dataset", revision=revision))
    normalized = {_safe_relative(item, "remote file") for item in files}
    unexpected = normalized - _ALLOWED_BASE_FILES - set(CHILD_TARGETS)
    if unexpected:
        raise ValueError(f"child repo has unexpected files: {sorted(unexpected)}")
    targets = normalized & set(CHILD_TARGETS)
    if not targets:
        return "EMPTY"
    if targets != set(CHILD_TARGETS):
        raise ValueError("child repo contains a partial exact-three publication")
    return "COMPLETE"


def _inspect_child(api: Any, *, repo: str, tag: str) -> ChildRemoteState:
    main = _revision_sha(api, repo, "main")
    target_state = _remote_target_state(api, repo=repo, revision=main)
    snapshot = _tag_snapshot(api, repo=repo, tag=tag)
    return ChildRemoteState(
        main_revision=main,
        target_state=target_state,
        tag_object_identity=None if snapshot is None else snapshot[0],
        tag_resolved_commit=None if snapshot is None else snapshot[1],
    )


def _is_repository_not_found(error: Exception) -> bool:
    if type(error).__name__ in {"RepositoryNotFoundError", "_NotFound"}:
        return True
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == 404 and "repo" in str(error).casefold()


def _validate_write_identity(api: Any) -> Mapping[str, Any]:
    identity = _mapping(api.whoami(), "HF write identity")
    auth = _mapping(identity.get("auth"), "HF auth identity")
    token = _mapping(auth.get("accessToken"), "HF access-token identity")
    if (
        identity.get("name") != "gavinlaw"
        or auth.get("type") != "access_token"
        or token.get("role") != "write"
    ):
        raise ValueError("HF credential is not gavinlaw write access")
    return {
        "name": identity["name"],
        "auth_type": auth["type"],
        "role": token["role"],
    }


def _operation(factory: Callable[..., Any], path: str, payload: bytes) -> Any:
    return factory(path_in_repo=path, path_or_fileobj=io.BytesIO(payload))


def _commit_oid(value: Any) -> str:
    oid = getattr(value, "oid", None)
    if not isinstance(oid, str) or _COMMIT.fullmatch(oid) is None:
        raise ValueError("HF create_commit did not return one immutable oid")
    return oid


def _validate_child_commit_parent(api: Any, *, repo: str, commit: str) -> str:
    history = tuple(
        getattr(item, "commit_id", None)
        for item in api.list_repo_commits(
            repo,
            repo_type="dataset",
            revision=commit,
        )
    )
    if (
        len(history) < 2
        or history[0] != commit
        or not isinstance(history[1], str)
        or _COMMIT.fullmatch(history[1]) is None
    ):
        raise ValueError("child exact-three commit history is malformed")
    return history[1]


def _compare_child_download(
    *,
    contract: Any,
    download_fn: Callable[..., str],
    revision: str,
    expected: Mapping[str, bytes],
    fresh_parent: Path,
) -> None:
    repo, _, _, _ = _destination_values(contract)
    observed = _download_exact_three(
        download_fn=download_fn,
        repo=repo,
        revision=revision,
        targets=CHILD_TARGETS,
        fresh_parent=fresh_parent,
        prefix="fresh16-decomposition-child-",
    )
    if dict(observed) != dict(expected):
        raise ValueError("child immutable exact-three readback differs")


def publish_failure_decomposition(
    *,
    mode: str,
    contract: Any,
    prepared: PreparedFailureDecomposition,
    api: Any,
    download_fn: Callable[..., str],
    operation_factory: Callable[..., Any],
    fresh_download_parent: str | Path,
    state_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    parent_identity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if mode not in {"run", "validate"}:
        raise ValueError("failure decomposition mode must be run or validate")
    repo, tag, commit_message, tag_message = _destination_values(contract)
    parent = _fresh_parent(fresh_download_parent)
    write_identity = _validate_write_identity(api)
    mutation_count = 0
    destination_absent = False
    try:
        state = _inspect_child(api, repo=repo, tag=tag)
    except Exception as error:
        if not _is_repository_not_found(error):
            raise
        if mode != "run":
            raise ValueError("child destination does not exist in validate mode")
        destination_absent = True
        state = ChildRemoteState(
            main_revision="",
            target_state="ABSENT",
            tag_object_identity=None,
            tag_resolved_commit=None,
        )
    if state_callback is not None:
        state_callback(
            "child_remote_base_receipt",
            {
                "repo": repo,
                "state": state.target_state,
                "main_revision": state.main_revision or None,
                "tag_object_identity": state.tag_object_identity,
                "tag_resolved_commit": state.tag_resolved_commit,
                "write_identity": write_identity,
            },
        )
    if destination_absent:
        api.create_repo(
            repo,
            repo_type="dataset",
            private=True,
            exist_ok=False,
        )
        mutation_count += 1
        state = _inspect_child(api, repo=repo, tag=tag)
    if state.target_state == "EMPTY":
        if mode != "run":
            raise ValueError("child exact-three publication is absent in validate mode")
        if state.tag_resolved_commit is not None:
            raise ValueError("child tag exists before its exact-three commit")
        operations = [
            _operation(operation_factory, path, prepared.files[path])
            for path in CHILD_TARGETS
        ]
        try:
            response = api.create_commit(
                repo,
                repo_type="dataset",
                revision="main",
                parent_commit=state.main_revision,
                commit_message=commit_message,
                operations=operations,
            )
            commit = _commit_oid(response)
            mutation_count += 1
        except Exception:
            recovered = _inspect_child(api, repo=repo, tag=tag)
            if recovered.target_state != "COMPLETE":
                raise
            commit = recovered.main_revision
            mutation_count += 1
        state = _inspect_child(api, repo=repo, tag=tag)
        if state.main_revision != commit or state.target_state != "COMPLETE":
            raise ValueError("child exact-three commit was interposed or incomplete")
    else:
        commit = state.main_revision
    child_base_commit = _validate_child_commit_parent(api, repo=repo, commit=commit)
    _compare_child_download(
        contract=contract,
        download_fn=download_fn,
        revision=commit,
        expected=prepared.files,
        fresh_parent=parent,
    )
    if state_callback is not None:
        state_callback(
            "child_report_commit_receipt",
            {
                "repo": repo,
                "base_commit": child_base_commit,
                "report_commit": commit,
                "files": _inventory(prepared.files),
            },
        )
    state = _inspect_child(api, repo=repo, tag=tag)
    if state.tag_resolved_commit is None:
        if mode != "run":
            raise ValueError("child immutable tag is absent in validate mode")
        try:
            api.create_tag(
                repo,
                repo_type="dataset",
                tag=tag,
                tag_message=tag_message,
                revision=commit,
                exist_ok=False,
            )
            mutation_count += 1
        except Exception:
            recovered = _tag_snapshot(api, repo=repo, tag=tag)
            if recovered is None or recovered[1] != commit:
                raise
            mutation_count += 1
        state = _inspect_child(api, repo=repo, tag=tag)
    if (
        state.main_revision != commit
        or state.target_state != "COMPLETE"
        or state.tag_resolved_commit != commit
        or state.tag_object_identity is None
    ):
        raise ValueError("child annotated tag does not resolve to its exact-three commit")
    if state_callback is not None:
        state_callback(
            "child_tag_receipt",
            {
                "repo": repo,
                "tag": tag,
                "tag_object_identity": state.tag_object_identity,
                "tag_resolved_commit": state.tag_resolved_commit,
            },
        )
    _compare_child_download(
        contract=contract,
        download_fn=download_fn,
        revision=tag,
        expected=prepared.files,
        fresh_parent=parent,
    )
    if parent_identity is not None:
        after_parent = _parent_identity_snapshot(api, contract)
        if dict(after_parent) != dict(parent_identity):
            raise ValueError("parent immutable identity changed during child publication")
    if state_callback is not None:
        state_callback(
            "immutable_replay_completion",
            {
                "child_target_count": 3,
                "child_commit": commit,
                "child_tag": tag,
                "parent_identity_sha256": None
                if parent_identity is None
                else parent_identity["identity_sha256"],
                "remote_mutation_call_count": mutation_count,
            },
        )
    return {
        "status": RUN_STATUS if mode == "run" else VALIDATE_STATUS,
        "repo": repo,
        "commit": commit,
        "tag": tag,
        "tag_object_identity": state.tag_object_identity,
        "target_count": 3,
        "remote_mutation_call_count": mutation_count,
        "files": _inventory(prepared.files),
    }


def _local_output_directory(contract: Any) -> Path | None:
    local = _section(
        contract,
        ("local_first_state_machine", "local_state", "local_output"),
        required=False,
    )
    if not local:
        return None
    value = None
    for name in ("artifact_directory", "output_directory", "artifact_root"):
        if name in local:
            if value is not None:
                raise ValueError("local output directory aliases are ambiguous")
            value = local[name]
    if value is None:
        return None
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("local output directory contract is malformed")
    return Path(value)


def execute_failure_decomposition(
    *,
    mode: str,
    contract: Any,
    api: Any,
    download_fn: Callable[..., str],
    operation_factory: Callable[..., Any],
    fresh_download_parent: str | Path,
    reducer: Callable[..., Any] | None = None,
    output_directory: str | Path | None = None,
    source_execution: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if mode not in {"run", "validate"}:
        raise ValueError("failure decomposition mode must be run or validate")
    _destination(contract)
    reducer_fn = reducer if reducer is not None else _lazy_reducer()
    state_machine = _LocalStateMachine(contract, mode=mode)
    state_machine.write(
        "global_claim",
        {
            "contract_sha256": getattr(contract, "sha256", None),
            "source_execution": dict(source_execution or {}),
            "parent_repo": PARENT_REPO,
            "parent_revision": PARENT_REVISION,
            "child_repo": _destination_values(contract)[0],
        },
    )
    parent = read_parent_exact_three(
        contract=contract,
        api=api,
        download_fn=download_fn,
        fresh_download_parent=fresh_download_parent,
    )
    state_machine.write(
        "parent_readonly_attestation",
        {
            "identity": dict(parent.identity),
            "selective_files": _inventory(parent.files),
            "parent_hf_mutation_count": 0,
        },
    )
    prepared = prepare_failure_decomposition(
        contract=contract,
        parent=parent,
        reducer=reducer_fn,
        source_execution=source_execution,
    )
    local_path = (
        Path(output_directory)
        if output_directory is not None
        else _local_output_directory(contract)
    )
    local_audit: Mapping[str, Any] = {"local_write_count": 0, "local_files": []}
    if local_path is not None:
        local_audit = stage_exact_three(
            local_path,
            prepared,
            allow_create=mode == "run",
        )
    state_machine.write(
        "local_report_completion",
        {
            "files": _inventory(prepared.files),
            "local_write_count": local_audit["local_write_count"],
        },
    )
    publication = publish_failure_decomposition(
        mode=mode,
        contract=contract,
        prepared=prepared,
        api=api,
        download_fn=download_fn,
        operation_factory=operation_factory,
        fresh_download_parent=fresh_download_parent,
        state_callback=state_machine.write,
        parent_identity=parent.identity,
    )
    state_machine.complete(
        {
            "child_repo": publication["repo"],
            "child_commit": publication["commit"],
            "child_tag": publication["tag"],
            "child_tag_object_identity": publication["tag_object_identity"],
            "parent_identity_sha256": parent.identity["identity_sha256"],
            "source_execution": dict(source_execution or {}),
            "artifact_files": _inventory(prepared.files),
        }
    )
    return {
        **publication,
        "parent_selective_download_count": 3,
        "child_immutable_download_count": 6,
        "reducer_invocation_count": 1,
        "local": dict(local_audit),
        "prohibited_access": dict(prepared.bundle_manifest["prohibited_access"]),
    }


__all__ = [
    "BUNDLE_STATUS",
    "CHILD_BUNDLE_PATH",
    "CHILD_REPORT_PATH",
    "CHILD_STATE_PATH",
    "CHILD_TARGETS",
    "PARENT_BINDINGS",
    "PARENT_BUNDLE_PATH",
    "PARENT_LABEL_PATH",
    "PARENT_REPO",
    "PARENT_REVISION",
    "PARENT_STATE_RECORDS_PATH",
    "PARENT_TAG",
    "PARENT_TARGETS",
    "PROTOCOL_ID",
    "ParentExactThree",
    "PreparedFailureDecomposition",
    "execute_failure_decomposition",
    "prepare_failure_decomposition",
    "publish_failure_decomposition",
    "read_parent_exact_three",
    "sha256_bytes",
    "stage_exact_three",
    "validate_parent_contract",
]
