"""CPU-only selective replay for independent confirm-20 failure decomposition."""

from __future__ import annotations

import io
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from causalcache.independent_confirm_artifact import (
    read_fixed_report,
    read_state_records_jsonl,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_independent_confirm20_failure_decomposition_v1"
DECOMPOSITION_STATUS = (
    "FROZEN_INDEPENDENT_CONFIRM20_ORACLE_INDEPENDENT_DECOMPOSITION_V1"
)
RUN_STATUS = "COMPLETED_INDEPENDENT_CONFIRM20_FAILURE_DECOMPOSITION_V1"
VALIDATE_STATUS = "REVALIDATED_INDEPENDENT_CONFIRM20_FAILURE_DECOMPOSITION_V1"
BUNDLE_STATUS = "FROZEN_INDEPENDENT_CONFIRM20_FAILURE_DECOMPOSITION_BUNDLE_V1"

PARENT_REPO = "gavinlaw/causalcache-independent-confirm20-mobile"
PARENT_REVISION = "a0b408e58d629299be334a74ecbd0ec2fa2ed1fc"
PARENT_TAG = "independent-confirm20-v1"
PARENT_TAG_OBJECT = "48921cafa00d7102d8b4c709d58061951f90e42c"
PARENT_PAYLOAD_COMMIT = "6d0cd95997186293e01c65276f3c082c11a9f52d"
PARENT_BASE_COMMIT = "c66a67c3451cee7e20ed737401dc8b074151574f"
PARENT_BUNDLE_PATH = "independent-confirm20/v1/report/bundle-manifest-v1.json"
PARENT_RAW_STATE_PATH = "independent-confirm20/v1/report/raw-state-records-v1.jsonl"
PARENT_FIXED_REPORT_PATH = "independent-confirm20/v1/report/fixed-report-v1.json"
PARENT_TARGETS = (
    PARENT_BUNDLE_PATH,
    PARENT_RAW_STATE_PATH,
    PARENT_FIXED_REPORT_PATH,
)
PARENT_BINDINGS: Mapping[str, Mapping[str, Any]] = {
    PARENT_BUNDLE_PATH: {
        "size_bytes": 2261,
        "sha256": "ccc996283b89f41db77c06f320e3b1ac88f3933bc19cfeea4c8b5b508ae4b351",
    },
    PARENT_RAW_STATE_PATH: {
        "size_bytes": 1_388_707,
        "sha256": "208fb36abb36b5e82eadcc8dac4ed20c5a030919902afe3e5b69cf6cad76a52a",
    },
    PARENT_FIXED_REPORT_PATH: {
        "size_bytes": 36_666,
        "sha256": "969729deb66e610846694b0fa1139c6b47f3927db7a58b50367f8ed93365198b",
    },
}

CHILD_REPO = "gavinlaw/causalcache-independent-confirm20-failure-decomposition-mobile"
CHILD_TAG = "independent-confirm20-failure-decomposition-v1"
CHILD_PREFIX = "independent-confirm20-failure-decomposition/v1"
CHILD_STATE_PATH = f"{CHILD_PREFIX}/state-decomposition-v1.jsonl"
CHILD_REPORT_PATH = f"{CHILD_PREFIX}/failure-decomposition-report-v1.json"
CHILD_BUNDLE_PATH = f"{CHILD_PREFIX}/bundle-manifest-v1.json"
CHILD_TARGETS = (CHILD_STATE_PATH, CHILD_REPORT_PATH, CHILD_BUNDLE_PATH)

_ALLOWED_BASE_FILES = frozenset({".gitattributes", "README.md"})
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ParentExactThree:
    files: Mapping[str, bytes]
    identity: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedFailureDecomposition:
    files: Mapping[str, bytes]
    report: Mapping[str, Any]
    bundle_manifest: Mapping[str, Any]


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
    data = _contract_data(contract)
    for name in names:
        candidate = getattr(contract, name, None)
        if candidate is not None:
            return _mapping(candidate, name)
        if name in data:
            return _mapping(data[name], name)
    if required:
        raise ValueError(f"contract lacks section {tuple(names)!r}")
    return {}


def _parent_hf(contract: Any) -> Mapping[str, Any]:
    parent = _section(
        contract,
        ("parent_input", "parent_report", "parent_artifact", "parent_confirm"),
    )
    return _mapping(parent["hf"], "parent HF") if "hf" in parent else parent


def _destination(contract: Any) -> Mapping[str, Any]:
    return _section(contract, ("destination", "output_destination"))


def _value(
    source: Mapping[str, Any],
    names: Sequence[str],
    *,
    label: str,
    default: Any = None,
) -> Any:
    values = [source[name] for name in names if name in source]
    if values:
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"{label} aliases disagree")
        return values[0]
    if default is not None:
        return default
    raise ValueError(f"{label} is missing")


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
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {item}")
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
    before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_id != after_id or len(payload) != after.st_size:
        raise ValueError(f"{label} changed while being read")
    return payload


def _fresh_parent(value: str | Path) -> Path:
    path = Path(value)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ValueError("fresh-download parent is missing") from error
    if not path.is_absolute() or not stat.S_ISDIR(mode) or path.is_symlink():
        raise ValueError("fresh-download parent must be an absolute non-symlink directory")
    return path.resolve()


def _inventory(files: Mapping[str, bytes]) -> list[Mapping[str, Any]]:
    return [
        {"path": path, "size_bytes": len(files[path]), "sha256": sha256_bytes(files[path])}
        for path in sorted(files)
    ]


def validate_contract_identity(contract: Any) -> Mapping[str, Any]:
    data = _contract_data(contract)
    if data.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("failure decomposition protocol identity drifted")
    parent = _parent_hf(contract)
    parent_identity = (
        _value(parent, ("repo", "repository"), label="parent repo"),
        _value(
            parent,
            ("report_commit", "immutable_revision", "revision", "tag_resolved_commit"),
            label="parent revision",
        ),
        _value(parent, ("tag", "immutable_tag"), label="parent tag"),
        _value(
            parent,
            ("annotated_tag_object", "tag_object"),
            label="parent tag object",
            default=PARENT_TAG_OBJECT,
        ),
        _value(
            parent,
            ("payload_commit", "report_parent_commit"),
            label="parent payload commit",
            default=PARENT_PAYLOAD_COMMIT,
        ),
    )
    if parent_identity != (
        PARENT_REPO,
        PARENT_REVISION,
        PARENT_TAG,
        PARENT_TAG_OBJECT,
        PARENT_PAYLOAD_COMMIT,
    ):
        raise ValueError("parent immutable identity drifted")
    input_contract = _section(contract, ("input_contract", "input"), required=False)
    targets = input_contract.get("exact_force_download_targets")
    if targets is None:
        targets = parent.get("exact_three_targets", list(PARENT_TARGETS))
    if isinstance(targets, list) and targets and isinstance(targets[0], Mapping):
        targets = [item.get("path") for item in targets]
    if tuple(targets) != PARENT_TARGETS:
        raise ValueError("parent selective target inventory drifted")

    destination = _destination(contract)
    output = _section(contract, ("output_contract", "output"), required=False)
    child_targets = (
        output.get("exact_targets")
        or output.get("exact_three_targets")
        or destination.get("exact_targets")
        or destination.get("exact_three_targets")
        or list(CHILD_TARGETS)
    )
    if (
        _value(destination, ("repo", "repository"), label="child repo") != CHILD_REPO
        or _value(destination, ("tag", "immutable_tag"), label="child tag") != CHILD_TAG
        or destination.get("repo_type", "dataset") != "dataset"
        or destination.get("private") is not True
        or tuple(child_targets) != CHILD_TARGETS
    ):
        raise ValueError("child private exact-three identity drifted")
    runtime = _section(contract, ("runtime_contract", "runtime"), required=False)
    if runtime and (
        runtime.get("cpu_only") is not True
        or runtime.get("gpu_count", 0) != 0
        or runtime.get("model_runtime_allowed", False) is not False
    ):
        raise ValueError("failure decomposition must remain CPU-only and model-free")
    return {"parent_revision": PARENT_REVISION, "child_repo": CHILD_REPO}


def _revision_sha(api: Any, repo: str, revision: str) -> str:
    info = api.dataset_info(repo, revision=revision)
    observed = getattr(info, "sha", None)
    if (
        not isinstance(observed, str)
        or _COMMIT.fullmatch(observed) is None
        or getattr(info, "private", True) is not True
    ):
        raise ValueError("HF dataset revision is not immutable and private")
    return observed


def _tag_snapshot(api: Any, *, repo: str, tag: str) -> tuple[str, str] | None:
    refs = api.list_repo_refs(repo, repo_type="dataset")
    matches = [item for item in refs.tags if getattr(item, "name", None) == tag]
    if len(matches) > 1:
        raise ValueError("HF tag is duplicated")
    if not matches:
        return None
    tag_object = getattr(matches[0], "target_commit", None)
    resolved = _revision_sha(api, repo, tag)
    if (
        not isinstance(tag_object, str)
        or _COMMIT.fullmatch(tag_object) is None
        or tag_object == resolved
    ):
        raise ValueError("an annotated HF tag is required")
    return tag_object, resolved


def _parent_identity_snapshot(api: Any) -> Mapping[str, Any]:
    files = tuple(
        sorted(
            api.list_repo_files(
                PARENT_REPO,
                repo_type="dataset",
                revision=PARENT_REVISION,
            )
        )
    )
    commits = tuple(
        getattr(item, "commit_id", None)
        for item in api.list_repo_commits(
            PARENT_REPO,
            repo_type="dataset",
            revision=PARENT_REVISION,
        )
    )
    if (
        _revision_sha(api, PARENT_REPO, PARENT_REVISION) != PARENT_REVISION
        or _tag_snapshot(api, repo=PARENT_REPO, tag=PARENT_TAG)
        != (PARENT_TAG_OBJECT, PARENT_REVISION)
        or not set(PARENT_TARGETS).issubset(files)
        or commits[:3] != (PARENT_REVISION, PARENT_PAYLOAD_COMMIT, PARENT_BASE_COMMIT)
    ):
        raise ValueError("parent report/tag/history identity drifted")
    value = {
        "repo": PARENT_REPO,
        "revision": PARENT_REVISION,
        "tag": PARENT_TAG,
        "annotated_tag_object": PARENT_TAG_OBJECT,
        "payload_commit": PARENT_PAYLOAD_COMMIT,
        "remote_file_count": len(files),
        "reachable_commits": list(commits),
    }
    return {**value, "identity_sha256": sha256_bytes(canonical_json_bytes(value))}


def _download_files(
    *,
    download_fn: Callable[..., str],
    repo: str,
    revision: str,
    targets: Sequence[str],
    fresh_parent: Path,
    prefix: str,
) -> Mapping[str, bytes]:
    with tempfile.TemporaryDirectory(prefix=prefix, dir=fresh_parent) as raw:
        root = Path(raw).resolve()
        if root.is_symlink() or any(root.iterdir()):
            raise ValueError("fresh replay root did not start empty")
        files: dict[str, bytes] = {}
        for target in targets:
            relative = _safe_relative(target, "HF target")
            returned = Path(
                download_fn(
                    repo_id=repo,
                    repo_type="dataset",
                    filename=relative,
                    revision=revision,
                    local_dir=root,
                    force_download=True,
                )
            )
            expected = root / relative
            if not returned.is_absolute() or returned != expected:
                raise ValueError("HF download returned a noncanonical path")
            current = root
            for part in PurePosixPath(relative).parts[:-1]:
                current /= part
                if current.is_symlink() or not current.is_dir():
                    raise ValueError("HF download traversed an unsafe parent")
            files[relative] = _regular_file_bytes(returned, label=relative)
    return MappingProxyType(files)


def _validate_parent_exact_three(files: Mapping[str, bytes]) -> None:
    if tuple(files) != PARENT_TARGETS:
        raise ValueError("parent exact-three order drifted")
    observed = {
        path: {"size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        for path, payload in files.items()
    }
    if observed != {path: dict(PARENT_BINDINGS[path]) for path in PARENT_TARGETS}:
        raise ValueError("parent exact-three bytes differ from frozen bindings")
    bundle = _strict_json(files[PARENT_BUNDLE_PATH], label="parent bundle")
    if (
        bundle.get("protocol_id") != "causalcache_independent_confirm_closed_loop_v1"
        or bundle.get("status") != "BUNDLED_INDEPENDENT_CONFIRM20_REPORT_V1"
        or bundle.get("payload_commit") != PARENT_PAYLOAD_COMMIT
    ):
        raise ValueError("parent bundle semantic identity drifted")
    report_inventory = bundle.get("report_inventory_without_bundle")
    if not isinstance(report_inventory, list):
        raise ValueError("parent bundle lacks report inventory")
    bound = {
        row.get("path"): row
        for row in report_inventory
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    for path in (PARENT_RAW_STATE_PATH, PARENT_FIXED_REPORT_PATH):
        if dict(bound.get(path, {})) != {"path": path, **dict(PARENT_BINDINGS[path])}:
            raise ValueError("parent bundle does not bind a selectively read file")
    raw = read_state_records_jsonl(files[PARENT_RAW_STATE_PATH])
    fixed = read_fixed_report(files[PARENT_FIXED_REPORT_PATH])
    if (
        len(raw) != 20
        or fixed.get("status") != "NO_GO_INDEPENDENT_CONFIRM"
        or fixed.get("evaluation_performed") is not True
        or fixed.get("fixed_state_denominator") != 20
        or fixed.get("go") is not False
        or not isinstance(fixed.get("records"), list)
        or len(fixed["records"]) != 20
    ):
        raise ValueError("parent confirm verdict or denominator drifted")


def read_parent_exact_three(
    *,
    contract: Any,
    api: Any,
    download_fn: Callable[..., str],
    fresh_download_parent: str | Path,
) -> ParentExactThree:
    validate_contract_identity(contract)
    before = _parent_identity_snapshot(api)
    files = _download_files(
        download_fn=download_fn,
        repo=PARENT_REPO,
        revision=PARENT_REVISION,
        targets=PARENT_TARGETS,
        fresh_parent=_fresh_parent(fresh_download_parent),
        prefix="independent-confirm20-parent-",
    )
    _validate_parent_exact_three(files)
    after = _parent_identity_snapshot(api)
    if before != after:
        raise ValueError("parent immutable identity changed during selective replay")
    return ParentExactThree(files=files, identity=MappingProxyType(dict(before)))


def _lazy_reducer() -> Callable[[bytes, bytes], Mapping[str, Any]]:
    from causalcache.independent_confirm_failure_decomposition import (
        build_independent_confirm_failure_decomposition,
    )

    return build_independent_confirm_failure_decomposition


def prepare_failure_decomposition(
    *,
    parent: ParentExactThree,
    contract: Any,
    reducer: Callable[[bytes, bytes], Mapping[str, Any]] | None = None,
    source_execution: Mapping[str, Any] | None = None,
) -> PreparedFailureDecomposition:
    reduced = _mapping(
        (reducer or _lazy_reducer())(
            parent.files[PARENT_RAW_STATE_PATH],
            parent.files[PARENT_FIXED_REPORT_PATH],
        ),
        "pure reducer result",
    )
    state_rows = reduced.get("state_rows")
    if not isinstance(state_rows, list) or len(state_rows) != 20:
        raise ValueError("pure reducer must return exactly 20 state rows")
    normalized_rows = []
    for ordinal, row in enumerate(state_rows):
        row = _mapping(row, f"state row {ordinal}")
        if (
            row.get("ordinal") != ordinal
            or row.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION
            or row.get("protocol_id", PROTOCOL_ID) != PROTOCOL_ID
            or row.get("status", DECOMPOSITION_STATUS) != DECOMPOSITION_STATUS
        ):
            raise ValueError("pure reducer state identity or order drifted")
        normalized_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": DECOMPOSITION_STATUS,
                **dict(row),
            }
        )
    state_payload = b"".join(
        canonical_json_bytes(row) + b"\n" for row in normalized_rows
    )
    report = {key: value for key, value in reduced.items() if key != "state_rows"}
    report.setdefault("state_count", 20)
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("status") != DECOMPOSITION_STATUS
        or report.get("state_count") != 20
    ):
        raise ValueError("pure reducer report identity or denominator drifted")
    report_payload = pretty_json_bytes(report)
    decision = _mapping(report.get("decision"), "failure decomposition decision")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": BUNDLE_STATUS,
        "contract_sha256": getattr(contract, "sha256", None),
        "source_execution": dict(source_execution or {}),
        "parent": {
            **dict(parent.identity),
            "exact_three_inventory": _inventory(parent.files),
        },
        "child_inventory_without_bundle": _inventory(
            {CHILD_STATE_PATH: state_payload, CHILD_REPORT_PATH: report_payload}
        ),
        "state_record_count": 20,
        "scientific_outcome": decision.get("status", decision.get("case")),
        "parent_verdict_unchanged": True,
        "closed_loop_matched_nll_sealed_test_authorized": False,
        "operation_counts": {
            "parent_selective_download_count": 3,
            "parent_state_decode_count": 20,
            "child_state_record_count": 20,
            "parent_hf_mutation_count": 0,
            "gpu_operation_count": 0,
            "model_load_count": 0,
            "model_forward_count": 0,
            "checkpoint_load_count": 0,
            "policy_forward_count": 0,
            "restoration_teacher_forward_count": 0,
            "gate_training_step_count": 0,
            "closed_loop_episode_count": 0,
            "matched_nll_evaluation_count": 0,
            "sealed_androidworld_test_access_count": 0,
        },
    }
    files = {
        CHILD_STATE_PATH: state_payload,
        CHILD_REPORT_PATH: report_payload,
        CHILD_BUNDLE_PATH: pretty_json_bytes(manifest),
    }
    return PreparedFailureDecomposition(
        files=MappingProxyType(files),
        report=MappingProxyType(report),
        bundle_manifest=MappingProxyType(manifest),
    )


def _exclusive_or_identical(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
    except FileExistsError:
        if _regular_file_bytes(path, label=str(path)) != payload:
            raise ValueError("retained local output differs from prepared bytes")
        return False
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444, follow_symlinks=False)
    return True


def _local_output_directory(contract: Any) -> Path | None:
    local = _section(
        contract,
        ("local_first_state_machine", "local_output"),
        required=False,
    )
    raw = local.get("artifact_directory") if local else None
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError("local artifact directory must be absolute")
    return path


def stage_exact_three(
    prepared: PreparedFailureDecomposition,
    *,
    directory: Path | None,
    allow_create: bool,
) -> Mapping[str, Any]:
    if directory is None:
        return {"local_write_count": 0, "local_output_directory": None}
    if directory.exists():
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("local artifact directory is unsafe")
    elif allow_create:
        directory.mkdir(parents=True, mode=0o700)
    else:
        raise ValueError("local artifact directory is missing during validation")
    writes = 0
    for path in CHILD_TARGETS:
        target = directory / path
        if allow_create:
            writes += int(_exclusive_or_identical(target, prepared.files[path]))
        elif _regular_file_bytes(target, label=path) != prepared.files[path]:
            raise ValueError("local exact-three differs from pure replay")
    observed = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed != set(CHILD_TARGETS):
        raise ValueError("local artifact directory is not exact-three")
    return {"local_write_count": writes, "local_output_directory": str(directory)}


def _is_not_found(error: Exception) -> bool:
    status = getattr(getattr(error, "response", None), "status_code", None)
    return status == 404 or "notfound" in error.__class__.__name__.lower()


def _child_snapshot(api: Any) -> Mapping[str, Any] | None:
    try:
        main = _revision_sha(api, CHILD_REPO, "main")
    except Exception as error:
        if _is_not_found(error):
            return None
        raise
    files = tuple(
        sorted(api.list_repo_files(CHILD_REPO, repo_type="dataset", revision=main))
    )
    if set(files) - _ALLOWED_BASE_FILES - set(CHILD_TARGETS):
        raise ValueError("child dataset contains unexpected files")
    return {
        "main": main,
        "files": files,
        "tag": _tag_snapshot(api, repo=CHILD_REPO, tag=CHILD_TAG),
    }


def _write_identity(api: Any) -> None:
    identity = api.whoami()
    name = identity.get("name") if isinstance(identity, Mapping) else None
    auth = identity.get("auth", {}) if isinstance(identity, Mapping) else {}
    token = auth.get("accessToken", {}) if isinstance(auth, Mapping) else {}
    if name != CHILD_REPO.split("/", 1)[0] or token.get("role") not in {"write", "admin"}:
        raise ValueError("HF token lacks the frozen child write identity")


def _commit_oid(value: Any) -> str:
    oid = getattr(value, "oid", None) or getattr(value, "commit_id", None)
    if not isinstance(oid, str) or _COMMIT.fullmatch(oid) is None:
        raise ValueError("HF commit response lacks an immutable oid")
    return oid


def _replay_child(
    *,
    api: Any,
    download_fn: Callable[..., str],
    prepared: PreparedFailureDecomposition,
    fresh_parent: Path,
    commit: str,
) -> int:
    count = 0
    for revision in (commit, CHILD_TAG):
        files = _download_files(
            download_fn=download_fn,
            repo=CHILD_REPO,
            revision=revision,
            targets=CHILD_TARGETS,
            fresh_parent=fresh_parent,
            prefix="independent-confirm20-child-",
        )
        count += len(files)
        if dict(files) != dict(prepared.files) or _revision_sha(api, CHILD_REPO, revision) != commit:
            raise ValueError("child immutable replay differs from prepared exact-three")
    return count


def publish_or_validate_child(
    *,
    mode: str,
    contract: Any,
    api: Any,
    download_fn: Callable[..., str],
    operation_factory: Callable[..., Any],
    prepared: PreparedFailureDecomposition,
    fresh_download_parent: str | Path,
) -> Mapping[str, Any]:
    if mode not in {"run", "validate"}:
        raise ValueError("mode must be run or validate")
    destination = _destination(contract)
    mutations = 0
    snapshot = _child_snapshot(api)
    if mode == "run":
        _write_identity(api)
        if snapshot is None:
            api.create_repo(CHILD_REPO, repo_type="dataset", private=True, exist_ok=False)
            mutations += 1
            snapshot = _child_snapshot(api)
        if snapshot is None or snapshot["tag"] is not None or set(snapshot["files"]) - _ALLOWED_BASE_FILES:
            raise ValueError("child destination is not an unused private base")
        response = api.create_commit(
            CHILD_REPO,
            repo_type="dataset",
            revision="main",
            parent_commit=snapshot["main"],
            commit_message=_value(
                destination,
                ("commit_message", "commit_title"),
                label="child commit message",
                default="Publish independent confirm-20 oracle decomposition",
            ),
            operations=[
                operation_factory(path_in_repo=path, path_or_fileobj=io.BytesIO(prepared.files[path]))
                for path in CHILD_TARGETS
            ],
        )
        commit = _commit_oid(response)
        mutations += 1
        api.create_tag(
            CHILD_REPO,
            repo_type="dataset",
            tag=CHILD_TAG,
            tag_message=_value(
                destination,
                ("tag_message",),
                label="child tag message",
                default="Freeze independent confirm-20 oracle decomposition",
            ),
            revision=commit,
            exist_ok=False,
        )
        mutations += 1
    else:
        if snapshot is None or snapshot["tag"] is None:
            raise ValueError("completed child dataset/tag is absent")
        commit = snapshot["main"]
    final = _child_snapshot(api)
    if (
        final is None
        or set(final["files"]) - _ALLOWED_BASE_FILES != set(CHILD_TARGETS)
        or final["main"] != commit
        or final["tag"] is None
        or final["tag"][1] != commit
    ):
        raise ValueError("child main/tag/tree identity drifted")
    downloads = _replay_child(
        api=api,
        download_fn=download_fn,
        prepared=prepared,
        fresh_parent=_fresh_parent(fresh_download_parent),
        commit=commit,
    )
    return {
        "repo": CHILD_REPO,
        "report_commit": commit,
        "tag": CHILD_TAG,
        "annotated_tag_object": final["tag"][0],
        "remote_mutation_count": mutations,
        "immutable_download_count": downloads,
        "byte_identical_fresh_replay": True,
    }


def execute_failure_decomposition(
    *,
    mode: str,
    contract: Any,
    api: Any,
    download_fn: Callable[..., str],
    operation_factory: Callable[..., Any],
    fresh_download_parent: str | Path,
    source_execution: Mapping[str, Any] | None = None,
    reducer: Callable[[bytes, bytes], Mapping[str, Any]] | None = None,
    output_directory: str | Path | None = None,
) -> Mapping[str, Any]:
    """Replay the frozen parent, reduce on CPU, then publish or validate."""
    parent = read_parent_exact_three(
        contract=contract,
        api=api,
        download_fn=download_fn,
        fresh_download_parent=fresh_download_parent,
    )
    prepared = prepare_failure_decomposition(
        parent=parent,
        contract=contract,
        reducer=reducer,
        source_execution=source_execution,
    )
    directory = Path(output_directory) if output_directory is not None else _local_output_directory(contract)
    local = stage_exact_three(prepared, directory=directory, allow_create=mode == "run")
    child = publish_or_validate_child(
        mode=mode,
        contract=contract,
        api=api,
        download_fn=download_fn,
        operation_factory=operation_factory,
        prepared=prepared,
        fresh_download_parent=fresh_download_parent,
    )
    decision = _mapping(prepared.report.get("decision"), "failure decomposition decision")
    return MappingProxyType(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": RUN_STATUS if mode == "run" else VALIDATE_STATUS,
            "scientific_outcome": decision.get("status", decision.get("case")),
            "parent_verdict_unchanged": True,
            "parent": dict(parent.identity),
            "parent_exact_three_inventory": _inventory(parent.files),
            "parent_selective_download_count": 3,
            "child": child,
            "child_exact_three_inventory": _inventory(prepared.files),
            "child_immutable_download_count": child["immutable_download_count"],
            "remote_mutation_count": child["remote_mutation_count"],
            "local": local,
            "operation_counts": dict(prepared.bundle_manifest["operation_counts"]),
            "gpu_model_checkpoint_policy_restoration_training_count": 0,
            "closed_loop_matched_nll_sealed_test_authorized": False,
        }
    )


__all__ = [
    "CHILD_REPO",
    "CHILD_TAG",
    "CHILD_TARGETS",
    "DECOMPOSITION_STATUS",
    "PARENT_BINDINGS",
    "PARENT_REPO",
    "PARENT_REVISION",
    "PARENT_TAG",
    "PARENT_TAG_OBJECT",
    "PARENT_TARGETS",
    "PROTOCOL_ID",
    "RUN_STATUS",
    "VALIDATE_STATUS",
    "execute_failure_decomposition",
    "prepare_failure_decomposition",
    "read_parent_exact_three",
    "sha256_bytes",
    "validate_contract_identity",
]
