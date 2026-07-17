"""Source-frozen, crash-recoverable formal-58 cache materialization."""

from __future__ import annotations

import io
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_gate_v1_formal_cache_runner_v1"
SOURCE_VALIDATION_STATUS = "VALID_SOURCE_A_GATE_V1_FORMAL_CACHE_RUNNER_V1"
RUNNER_FREEZE_STATUS = "FROZEN_GATE_V1_FORMAL_CACHE_EXECUTION_B_RUNNER_V1"
GLOBAL_CLAIM_STATUS = "CLAIMED_GATE_V1_FORMAL58_CACHE_V1"
FEATURE_COMPLETION_STATUS = "COMPLETED_GATE_V1_FORMAL58_FEATURE_CACHE_V1"
LABEL_ACCESS_CLAIM_STATUS = "CLAIMED_GATE_V1_FORMAL58_LABEL_ACCESS_V1"
LABEL_COMPLETION_STATUS = "COMPLETED_GATE_V1_FORMAL58_LABEL_CACHE_V1"
REMOTE_BASE_STATUS = "CAPTURED_GATE_V1_FORMAL58_CACHE_REMOTE_BASE_V1"
SIDECAR_STATUS = "IMMUTABLE_GATE_V1_FORMAL58_CACHE_BUNDLE_MANIFEST_V1"
FINAL_COMPLETION_STATUS = "COMPLETED_GATE_V1_FORMAL58_CACHE_PUBLICATION_V1"
RUN_STATUS = "VALID_GATE_V1_FORMAL58_CACHE_PUBLICATION_V1"
VALIDATE_STATUS = "REVALIDATED_GATE_V1_FORMAL58_CACHE_PUBLICATION_V1"
REMOTE_EMPTY = "EMPTY"
REMOTE_BYTE_IDENTICAL_UNTAGGED = "BYTE_IDENTICAL_UNTAGGED"
REMOTE_TAGGED_BYTE_IDENTICAL = "TAGGED_BYTE_IDENTICAL"
PAIR_COMMIT_TITLE = "Publish CausalCache gate v1 formal-58 caches"
TAG_MESSAGE = "Freeze CausalCache gate v1 formal-58 caches"
RUNNER_FREEZE_PATH = (
    "code/configs/causalcache_gate_v1_formal_cache_runner_v1.json"
)
RUNNER_SOURCE_PATHS = (
    "code/causalcache/gate_v1_formal_cache_runner.py",
    "code/scripts/manage_gate_v1_formal_cache.py",
    "code/tests/test_gate_v1_formal_cache_runner.py",
)
FORBIDDEN_IMPORT_ROOTS = (
    "accelerate",
    "bitsandbytes",
    "flash_attn",
    "jax",
    "sglang",
    "tensorflow",
    "torch",
    "torchvision",
    "transformers",
    "triton",
    "vllm",
    "xformers",
)
ZERO_OPERATION_KEYS = (
    "gpu_operation_count",
    "model_load_count",
    "model_forward_count",
    "training_example_count",
    "development_semantic_access_count",
    "confirm_state_access_count",
    "matched_nll_evaluation_count",
    "closed_loop_episode_count",
    "development_semantic_decode_count",
    "confirm_semantic_decode_count",
    "optimizer_step_count",
    "oracle_metric_count",
    "oof_metric_count",
    "checkpoint_count",
    "test_semantic_decode_count",
    "confirm_semantic_access_count",
    "test_semantic_access_count",
    "training_operation_count",
    "model_operation_count",
    "policy_operation_count",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class CacheArtifact:
    payload: bytes
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class FormalCacheHooks:
    build_feature: Callable[[Any, Path], CacheArtifact]
    read_feature: Callable[[Any, bytes], Mapping[str, Any]]
    build_label: Callable[[Any, Path], CacheArtifact]
    read_label: Callable[[Any, bytes], Mapping[str, Any]]
    join_only_audit: Callable[
        [Any, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
    ]


@dataclass(frozen=True)
class SourceIdentity:
    head: str
    remote_main: str
    branch: str
    origin_url: str
    source_inventory: tuple[Mapping[str, Any], ...]
    loaded_module_inventory: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class RemoteInspection:
    state: str
    main_revision: str
    main_files: tuple[str, ...]
    immutable_revision: str | None
    tag_object_identity: str | None
    tag_resolved_commit: str | None


@dataclass(frozen=True)
class PreparedBundle:
    feature: CacheArtifact
    label: CacheArtifact
    join_audit: Mapping[str, Any]
    manifest: Mapping[str, Any]
    manifest_bytes: bytes
    files: Mapping[str, bytes]


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


def _contract_data(contract: Any) -> Mapping[str, Any]:
    return _mapping(getattr(contract, "data", contract), "formal cache contract")


def _section(contract: Any, name: str) -> Mapping[str, Any]:
    candidate = getattr(contract, name, None)
    if candidate is not None:
        return _mapping(candidate, name)
    aliases = {
        "local_state": "local_first_state_machine",
        "formats": "cache_formats",
        "runtime": "runtime_contract",
        "operations": "source_only_operation_contract",
        "execution_operations": "execution_expected_operation_contract",
    }
    key = aliases.get(name, name)
    return _mapping(_contract_data(contract).get(key), name)


def _value(section: Mapping[str, Any], *names: str) -> Any:
    present = [name for name in names if name in section]
    if len(present) != 1:
        raise ValueError(f"expected exactly one field from {names!r}")
    return section[present[0]]


def _contract_sha(contract: Any) -> str:
    value = getattr(contract, "sha256", None)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("formal cache contract SHA256 is invalid")
    return value


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
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
    return _mapping(value, label)


def _regular_file_bytes(
    path: str | Path,
    *,
    label: str,
    mode: int | None = None,
) -> tuple[bytes, os.stat_result]:
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
    payload = b"".join(chunks)
    fingerprint = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if fingerprint(before) != fingerprint(after) or len(payload) != after.st_size:
        raise ValueError(f"{label} changed while being read")
    return payload, after


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_or_identical_bytes(
    path: str | Path,
    payload: bytes,
    *,
    mode: int,
) -> tuple[str, bool]:
    final = Path(path)
    if not final.parent.is_dir() or final.parent.is_symlink():
        raise ValueError("state parent must preexist as a real directory")
    if final.exists() or final.is_symlink():
        existing, _ = _regular_file_bytes(final, label="existing state", mode=mode)
        if existing != payload:
            raise ValueError("existing state differs from deterministic bytes")
        return sha256_bytes(existing), False
    temporary = final.with_name(
        f".{final.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
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
                raise OSError("state write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, final)
        _fsync_directory(final.parent)
        return sha256_bytes(payload), True
    except FileExistsError:
        existing, _ = _regular_file_bytes(final, label="raced state", mode=mode)
        if existing != payload:
            raise ValueError("concurrent state differs from deterministic bytes")
        return sha256_bytes(existing), False
    finally:
        temporary.unlink(missing_ok=True)


def _publish_artifact(path: Path, payload: bytes, *, mode: int) -> bool:
    digest, created = exclusive_or_identical_bytes(path, payload, mode=mode)
    if digest != sha256_bytes(payload):
        raise ValueError("artifact strict readback SHA256 drifted")
    return created


def _read_state(path: Path, expected: Mapping[str, Any], *, mode: int) -> bytes:
    payload, _ = _regular_file_bytes(path, label=path.name, mode=mode)
    observed = _strict_json(payload, label=path.name)
    if observed != expected or payload != pretty_json_bytes(expected):
        raise ValueError(f"state bytes drifted: {path}")
    return payload


def _state_path(contract: Any, *names: str) -> Path:
    state = _section(contract, "local_state")
    aliases = {
        "global_claim_path": "global_claim",
        "claim_path": "global_claim",
        "feature_completion_path": "feature_completion",
        "label_access_claim_path": "label_access_claim",
        "label_completion_path": "label_completion",
        "remote_base_receipt_path": "remote_base_receipt",
        "remote_base_path": "remote_base_receipt",
        "completion_staging_path": "completion_staging",
        "retained_completion_staging_path": "completion_staging",
        "completion_path": "final_completion",
        "final_completion_path": "final_completion",
    }
    requested = {aliases[name] for name in names}
    ordered = state.get("ordered_states")
    if not isinstance(ordered, list):
        value = _value(state, *names)
    else:
        matches = [
            item
            for item in ordered
            if isinstance(item, Mapping) and item.get("name") in requested
        ]
        if len(matches) != 1:
            raise ValueError(f"state path is missing or duplicated: {names!r}")
        value = matches[0].get("path")
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"state path is not absolute: {names!r}")
    return Path(value)


def _state_mode(contract: Any) -> int:
    value = _value(_section(contract, "local_state"), "file_mode", "state_file_mode")
    if value != 0o600:
        raise ValueError("formal cache state files must use mode 0600")
    return value


def _feature_path(contract: Any) -> Path:
    formats = _section(contract, "formats")
    feature = _mapping(formats.get("feature"), "feature cache format")
    value = _value(feature, "archive_path", "feature_cache_path")
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("feature cache path must be absolute")
    return Path(value)


def _label_path(contract: Any) -> Path:
    formats = _section(contract, "formats")
    label = _mapping(formats.get("label"), "label cache format")
    value = _value(label, "archive_path", "label_cache_path")
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("label cache path must be absolute")
    return Path(value)


def _runner_freeze_path(contract: Any) -> str:
    source = _mapping(_contract_data(contract).get("source_freeze"), "source freeze")
    execution = _mapping(
        source.get("execution_b_runner_freeze"), "execution-B runner freeze"
    )
    path = execution.get("path", RUNNER_FREEZE_PATH)
    if path != RUNNER_FREEZE_PATH:
        raise ValueError("runner-freeze path drifted")
    return path


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


def _remote_main(root: Path, remote: str, contract: Any) -> str:
    source = _mapping(_contract_data(contract).get("source_freeze"), "source freeze")
    live = _mapping(
        source.get("execution_b_live_remote_validation"),
        "execution-B live remote validation",
    )
    expected_argv = [
        "git",
        "ls-remote",
        "--exit-code",
        remote,
        "refs/heads/main",
    ]
    if (
        live.get("required") is not True
        or live.get("network_call_count") != 1
        or live.get("argv") != expected_argv
        or live.get("expected_stdout_record_count") != 1
        or live.get("expected_commit_pattern") != "[0-9a-f]{40}"
        or live.get("expected_ref") != "refs/heads/main"
        or live.get("required_equalities")
        != ["HEAD", f"{remote}/main", f"live {remote} refs/heads/main"]
    ):
        raise ValueError("execution-B live remote source contract drifted")
    output = _git(root, *expected_argv[1:])
    lines = output.decode("ascii").splitlines()
    suffix = "\trefs/heads/main"
    if len(lines) != 1 or not lines[0].endswith(suffix):
        raise ValueError("canonical remote main response drifted")
    commit = lines[0][: -len(suffix)]
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("canonical remote main is not one commit")
    return commit


def _source_paths(contract: Any) -> tuple[str, ...]:
    freeze = _mapping(_contract_data(contract).get("source_freeze"), "source freeze")
    raw = freeze.get("required_source_a_paths")
    if not isinstance(raw, list) or not raw:
        raw = list(RUNNER_SOURCE_PATHS)
    paths = tuple(_safe_relative(item, "source path") for item in raw)
    if len(paths) != len(set(paths)) or not set(RUNNER_SOURCE_PATHS).issubset(paths):
        raise ValueError("source-A path inventory is incomplete or duplicated")
    return paths


def _source_inventory(root: Path, commit: str, paths: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
    records = []
    for relative in paths:
        payload, _ = _regular_file_bytes(root / relative, label=relative)
        if payload != _git(root, "show", f"{commit}:{relative}"):
            raise ValueError(f"source blob differs from commit: {relative}")
        records.append(
            {"path": relative, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        )
    return tuple(records)


def _loaded_module_inventory(root: Path, commit: str) -> tuple[Mapping[str, Any], ...]:
    code_root = root / "code"
    records = []
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
            raise ValueError(f"loaded project module escaped repository: {name}") from error
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
        raise ValueError("loaded project module inventory is empty")
    return tuple(records)


def validate_clean_pushed_source(
    contract: Any,
    *,
    expected_commit: str | None,
    require_live_remote: bool = False,
) -> SourceIdentity:
    root = Path(getattr(contract, "repository_root", ".")).resolve()
    source = _mapping(_contract_data(contract).get("source_freeze"), "source freeze")
    branch = source.get("branch", "main")
    remote = source.get("origin_name", "origin")
    origin_url_expected = source.get(
        "origin_url", "https://github.com/luojiaxuan/CausalCache.git"
    )
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    remote_tracking = _git(root, "rev-parse", f"{remote}/{branch}").decode().strip()
    current_branch = _git(root, "branch", "--show-current").decode().strip()
    origin_url = _git(root, "remote", "get-url", remote).decode().strip()
    offline = _mapping(
        source.get("source_a_remote_validation"), "Source-A remote validation"
    )
    if (
        offline.get("network_call_count") != 0
        or offline.get("remote_tracking_ref") != f"refs/remotes/{remote}/{branch}"
        or offline.get("required_equalities") != ["HEAD", f"{remote}/{branch}"]
    ):
        raise ValueError("Source-A offline remote source contract drifted")
    # note (luojiaxuan): Source-A validation is offline by frozen contract and
    # uses its tracking ref. Execution-B must additionally bind live main.
    remote_main = (
        _remote_main(root, remote, contract)
        if require_live_remote
        else remote_tracking
    )
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if (
        _COMMIT.fullmatch(head) is None
        or (expected_commit is not None and head != expected_commit)
        or head != remote_tracking
        or head != remote_main
        or current_branch != branch
        or origin_url != origin_url_expected
        or status
    ):
        raise ValueError("formal cache source must be clean pushed canonical main")
    return SourceIdentity(
        head=head,
        remote_main=remote_main,
        branch=current_branch,
        origin_url=origin_url,
        source_inventory=_source_inventory(root, head, _source_paths(contract)),
        loaded_module_inventory=_loaded_module_inventory(root, head),
    )


def _source_record(source: SourceIdentity) -> Mapping[str, Any]:
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
    contract: Any,
    *,
    expected_source_a_git_commit: str | None = None,
) -> Mapping[str, Any]:
    root = Path(getattr(contract, "repository_root", ".")).resolve()
    freeze_path = root / _runner_freeze_path(contract)
    if freeze_path.exists() or freeze_path.is_symlink():
        raise ValueError("execution-B runner freeze must be absent from source-A")
    source = validate_clean_pushed_source(
        contract,
        expected_commit=expected_source_a_git_commit,
        require_live_remote=False,
    )
    validate_no_gpu_or_model_runtime(require_environment=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": SOURCE_VALIDATION_STATUS,
        "contract_sha256": _contract_sha(contract),
        "source_a": dict(_source_record(source)),
        "execution_authorized": False,
        "formal_cache_materialized": False,
    }


def _runner_freeze_payload(contract: Any, source: SourceIdentity) -> Mapping[str, Any]:
    source_freeze = _mapping(
        _contract_data(contract).get("source_freeze"), "source freeze"
    )
    paths = list(_source_paths(contract))
    prerequisites = list(
        _mapping(item, "Git prerequisite")
        for item in source_freeze.get("git_prerequisites", ())
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": RUNNER_FREEZE_STATUS,
        "contract_sha256": _contract_sha(contract),
        "source_a_git_commit": source.head,
        "execution_b_required_unique_diff": [_runner_freeze_path(contract)],
        "required_source_a_paths": paths,
        "required_source_a_paths_sha256": sha256_bytes(canonical_json_bytes(paths)),
        "git_prerequisites": prerequisites,
        "git_prerequisites_sha256": sha256_bytes(
            canonical_json_bytes(prerequisites)
        ),
        "source_blob_inventory": list(source.source_inventory),
        "source_blob_inventory_sha256": sha256_bytes(
            canonical_json_bytes(source.source_inventory)
        ),
        "source_a_inventory_sha256": sha256_bytes(
            canonical_json_bytes(source.source_inventory)
        ),
        "loaded_module_inventory": list(source.loaded_module_inventory),
        "loaded_module_inventory_sha256": sha256_bytes(
            canonical_json_bytes(source.loaded_module_inventory)
        ),
        "execution_authorized_after_clean_pushed_b_only": True,
    }


def materialize_runner_freeze(
    contract: Any,
    *,
    expected_source_a_git_commit: str | None = None,
) -> Mapping[str, Any]:
    validation = validate_source_a(
        contract, expected_source_a_git_commit=expected_source_a_git_commit
    )
    source_record = validation["source_a"]
    source = SourceIdentity(
        head=source_record["git_commit"],
        remote_main=source_record["remote_main_git_commit"],
        branch=source_record["branch"],
        origin_url=source_record["origin_url"],
        source_inventory=tuple(source_record["source_inventory"]),
        loaded_module_inventory=tuple(source_record["loaded_module_inventory"]),
    )
    payload = _runner_freeze_payload(contract, source)
    root = Path(getattr(contract, "repository_root", ".")).resolve()
    path = root / _runner_freeze_path(contract)
    path.parent.mkdir(parents=True, exist_ok=True)
    exclusive_or_identical_bytes(path, pretty_json_bytes(payload), mode=0o644)
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    lines = tuple(line.decode("utf-8") for line in status.splitlines())
    expected = (f"?? {_runner_freeze_path(contract)}",)
    if lines != expected:
        raise ValueError("runner freeze is not the unique source-A worktree diff")
    return payload


def load_runner_freeze(contract: Any) -> Mapping[str, Any]:
    root = Path(getattr(contract, "repository_root", ".")).resolve()
    path = root / _runner_freeze_path(contract)
    payload, _ = _regular_file_bytes(path, label="execution-B runner freeze")
    value = _strict_json(payload, label="execution-B runner freeze")
    if payload != pretty_json_bytes(value):
        raise ValueError("execution-B runner freeze is not canonical pretty JSON")
    source_freeze = _mapping(
        _contract_data(contract).get("source_freeze"), "source freeze"
    )
    execution_binding = _mapping(
        source_freeze.get("execution_b_runner_freeze"),
        "execution-B runner freeze binding",
    )
    paths = list(_source_paths(contract))
    prerequisites = list(source_freeze.get("git_prerequisites", ()))
    inventory = value.get("source_blob_inventory")
    modules = value.get("loaded_module_inventory")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != RUNNER_FREEZE_STATUS
        or value.get("contract_sha256") != _contract_sha(contract)
        or value.get("execution_b_required_unique_diff")
        != [_runner_freeze_path(contract)]
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
        or any(
            execution_binding.get(key) is not True
            for key in (
                "only_allowed_execution_b_source_tree_diff",
                "bind_required_source_a_paths",
                "bind_git_prerequisites",
                "bind_source_a_inventory_sha256",
            )
        )
    ):
        raise ValueError("execution-B runner freeze identity drifted")
    return value


def validate_execution_b_source(
    contract: Any,
    *,
    expected_execution_b_git_commit: str,
) -> SourceIdentity:
    freeze = load_runner_freeze(contract)
    source = validate_clean_pushed_source(
        contract,
        expected_commit=expected_execution_b_git_commit,
        require_live_remote=True,
    )
    root = Path(getattr(contract, "repository_root", ".")).resolve()
    source_a = freeze.get("source_a_git_commit")
    if not isinstance(source_a, str) or _COMMIT.fullmatch(source_a) is None:
        raise ValueError("source-A commit in runner freeze is invalid")
    parent_record = (
        _git(root, "rev-list", "--parents", "-n", "1", source.head)
        .decode("ascii")
        .strip()
        .split()
    )
    if parent_record != [source.head, source_a]:
        raise ValueError("execution-B must be the single-parent direct child of source-A")
    changed = tuple(
        line
        for line in _git(root, "diff", "--name-only", source_a, source.head)
        .decode("utf-8")
        .splitlines()
        if line
    )
    if changed != (_runner_freeze_path(contract),):
        raise ValueError("execution-B differs from source-A outside runner freeze")
    expected_inventory = tuple(freeze.get("source_blob_inventory", ()))
    if source.source_inventory != expected_inventory:
        raise ValueError("execution-B source blobs differ from source-A freeze")
    frozen_modules = {
        item["module"]: item for item in freeze.get("loaded_module_inventory", ())
    }
    loaded_modules = {item["module"]: item for item in source.loaded_module_inventory}
    if loaded_modules != frozen_modules:
        raise ValueError("execution-B loaded source modules differ from source-A")
    return source


def validate_no_gpu_or_model_runtime(
    *,
    require_environment: bool,
    contract: Any | None = None,
) -> Mapping[str, Any]:
    imported = sorted(
        name
        for name in sys.modules
        if any(name == root or name.startswith(root + ".") for root in FORBIDDEN_IMPORT_ROOTS)
    )
    nodes = sorted(
        str(path)
        for pattern in ("/dev/nvidia*", "/dev/dri/renderD*")
        for path in Path("/").glob(pattern.lstrip("/"))
        if path.exists() or path.is_symlink()
    )
    environment_ok = (
        os.environ.get("NVIDIA_VISIBLE_DEVICES") == "void"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    )
    if imported or nodes or (require_environment and not environment_ok):
        raise ValueError("formal cache runtime exposed GPU or model frameworks")
    thread_environment: Mapping[str, Any] = {}
    if require_environment:
        if contract is None:
            raise ValueError("execution runtime validation requires frozen contract")
        runtime = _section(contract, "runtime")
        thread_environment = _mapping(
            runtime.get("thread_environment"), "runtime thread environment"
        )
        observed_thread_environment = {
            key: os.environ.get(key) for key in thread_environment
        }
        if (
            runtime.get("device") != "cpu"
            or runtime.get("gpu_required") is not False
            or runtime.get("normalized_device_requests") != []
            or runtime.get("nvidia_visible_devices") != "void"
            or runtime.get("cuda_visible_devices") != ""
            or runtime.get("nvidia_device_nodes") != []
            or runtime.get("model_framework_import_allowed") is not False
            or runtime.get("python_implementation") != platform.python_implementation()
            or runtime.get("python_version") != platform.python_version()
            or runtime.get("machine") != platform.machine()
            or observed_thread_environment != dict(thread_environment)
        ):
            raise ValueError("formal cache execution runtime differs from frozen contract")
    return {
        "forbidden_modules_imported": imported,
        "gpu_device_nodes": nodes,
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": "cpu",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "thread_environment": dict(thread_environment),
    }


def _assert_zero_operations(value: Mapping[str, Any], *, label: str) -> None:
    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key in ZERO_OPERATION_KEYS and child != 0:
                    raise ValueError(f"{label} performed prohibited operation: {key}")
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)


def _execution_operation_contract(contract: Any) -> Mapping[str, int]:
    raw = _section(contract, "execution_operations")
    result: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("execution operation contract must contain integer counts")
        if value < 0:
            raise ValueError("execution operation contract contains a negative count")
        result[key] = value
    if not result:
        raise ValueError("execution operation contract is empty")
    return result


def _validate_execution_operation_audit(
    contract: Any,
    audit: Mapping[str, Any],
) -> Mapping[str, int]:
    expected = dict(_execution_operation_contract(contract))
    candidate = audit.get("execution_operation_counts", audit)
    observed = _mapping(candidate, "execution operation audit")
    selected = {key: observed.get(key) for key in expected}
    if selected != expected:
        raise ValueError("formal cache execution operation counts drifted")
    extra_count_keys = {
        key
        for key in observed
        if key.endswith("_count") and key not in expected
    }
    if extra_count_keys:
        raise ValueError("formal cache execution audit contains unknown count keys")
    return expected


def _zero_operations() -> Mapping[str, int]:
    return {key: 0 for key in ZERO_OPERATION_KEYS}


def _artifact_record(payload: bytes, audit: Mapping[str, Any]) -> Mapping[str, Any]:
    _assert_zero_operations(audit, label="formal cache audit")
    return {
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "audit": dict(audit),
        "audit_sha256": sha256_bytes(canonical_json_bytes(audit)),
    }


def _normalize_artifact(value: Any, label: str) -> CacheArtifact:
    if not isinstance(value, CacheArtifact) or not isinstance(value.payload, bytes):
        raise TypeError(f"{label} builder must return CacheArtifact")
    audit = dict(_mapping(value.audit, f"{label} audit"))
    _assert_zero_operations(audit, label=label)
    return CacheArtifact(payload=value.payload, audit=audit)


def _readback_artifact(
    path: Path,
    expected: CacheArtifact,
    reader: Callable[[Any, bytes], Mapping[str, Any]],
    contract: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    payload, _ = _regular_file_bytes(path, label=label, mode=0o444)
    if payload != expected.payload:
        raise ValueError(f"{label} bytes differ after publication")
    audit = dict(_mapping(reader(contract, payload), f"{label} readback"))
    _assert_zero_operations(audit, label=f"{label} readback")
    if audit != expected.audit:
        raise ValueError(f"{label} readback audit differs from builder")
    return audit


def _destination(contract: Any) -> Mapping[str, Any]:
    destination = _section(contract, "destination")
    if destination.get("private") is not True or destination.get("repo_type") != "dataset":
        raise ValueError("formal cache destination must be one private dataset")
    if (
        destination.get("sidecar_status") != SIDECAR_STATUS
        or destination.get("annotated_tag_required") is not True
        or destination.get("archive_and_sidecar_same_commit_required") is not True
    ):
        raise ValueError("formal cache destination security contract drifted")
    return destination


def _remote_paths(contract: Any) -> tuple[str, str, str]:
    destination = _destination(contract)
    exact = destination.get("exact_three_targets")
    if isinstance(exact, list):
        if len(exact) != 3:
            raise ValueError("formal cache destination must bind exactly three targets")
        paths = tuple(exact)
    else:
        paths = (
            _value(destination, "feature_cache_path", "feature_path"),
            _value(destination, "label_cache_path", "label_path"),
            _value(
                destination,
                "bundle_manifest_path",
                "manifest_path",
                "sidecar_path",
            ),
        )
    normalized = tuple(_safe_relative(path, "remote cache path") for path in paths)
    if len(set(normalized)) != 3:
        raise ValueError("formal cache remote exact-three paths collide")
    return normalized


def _revision_info(api: Any, contract: Any, revision: str) -> Any:
    destination = _destination(contract)
    return api.dataset_info(destination["repo"], revision=revision)


def _revision_sha(api: Any, contract: Any, revision: str) -> str:
    value = getattr(_revision_info(api, contract, revision), "sha", None)
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ValueError(f"HF revision is not immutable: {revision}")
    return value


def _repo_files(api: Any, contract: Any, revision: str) -> tuple[str, ...]:
    destination = _destination(contract)
    values = api.list_repo_files(
        destination["repo"], repo_type="dataset", revision=revision
    )
    result = tuple(sorted(_safe_relative(item, "remote file") for item in values))
    if len(result) != len(set(result)):
        raise ValueError("remote file inventory contains duplicates")
    return result


def _remote_history(
    api: Any,
    contract: Any,
    revision: str,
) -> tuple[Mapping[str, str], ...]:
    destination = _destination(contract)
    records = list(
        api.list_repo_commits(
            destination["repo"],
            repo_type="dataset",
            revision=revision,
        )
    )
    result: list[Mapping[str, str]] = []
    seen: set[str] = set()
    for record in records:
        identifier = getattr(record, "commit_id", None)
        title = getattr(record, "title", None)
        if (
            not isinstance(identifier, str)
            or _COMMIT.fullmatch(identifier) is None
            or identifier in seen
            or not isinstance(title, str)
        ):
            raise ValueError("HF reachable history record drifted")
        seen.add(identifier)
        result.append({"commit_id": identifier, "title": title})
    if not result or result[0]["commit_id"] != revision:
        raise ValueError("HF reachable history is not newest-first at requested revision")
    return tuple(result)


def _remote_blob_inventory(
    api: Any,
    contract: Any,
    revision: str,
) -> tuple[Mapping[str, Any], ...]:
    destination = _destination(contract)
    entries = list(
        api.list_repo_tree(
            destination["repo"],
            path_in_repo=None,
            recursive=True,
            expand=True,
            revision=revision,
            repo_type="dataset",
        )
    )
    result: list[Mapping[str, Any]] = []
    observed: set[str] = set()
    for entry in entries:
        blob_id = getattr(entry, "blob_id", None)
        if blob_id is None:
            continue
        path = _safe_relative(getattr(entry, "path", None), "remote blob path")
        size = getattr(entry, "size", None)
        if (
            path in observed
            or not isinstance(blob_id, str)
            or not blob_id
            or type(size) is not int
            or size < 0
        ):
            raise ValueError("remote blob inventory identity drifted")
        observed.add(path)
        lfs = getattr(entry, "lfs", None)
        if lfs is None:
            lfs_record = None
        else:
            getter = lfs.get if isinstance(lfs, Mapping) else lambda key: getattr(lfs, key, None)
            lfs_record = {
                "size": getter("size"),
                "sha256": getter("sha256"),
                "pointer_size": getter("pointer_size"),
            }
            if (
                type(lfs_record["size"]) is not int
                or lfs_record["size"] < 0
                or not isinstance(lfs_record["sha256"], str)
                or _SHA256.fullmatch(lfs_record["sha256"]) is None
                or type(lfs_record["pointer_size"]) is not int
                or lfs_record["pointer_size"] < 0
            ):
                raise ValueError("remote LFS identity drifted")
        xet_hash = getattr(entry, "xet_hash", None)
        if xet_hash is not None and (not isinstance(xet_hash, str) or not xet_hash):
            raise ValueError("remote Xet identity drifted")
        result.append(
            {
                "path": path,
                "blob_id": blob_id,
                "size_bytes": size,
                "lfs_identity": lfs_record,
                "xet_hash": xet_hash,
            }
        )
    return tuple(sorted(result, key=lambda item: item["path"]))


def _tag_snapshot(api: Any, contract: Any) -> tuple[str, str] | None:
    destination = _destination(contract)
    refs = api.list_repo_refs(destination["repo"], repo_type="dataset")
    matches = [item for item in refs.tags if item.name == destination["tag"]]
    if len(matches) > 1:
        raise ValueError("formal cache tag is duplicated")
    if not matches:
        return None
    object_identity = getattr(matches[0], "target_commit", None)
    resolved = _revision_sha(api, contract, destination["tag"])
    if (
        not isinstance(object_identity, str)
        or _COMMIT.fullmatch(object_identity) is None
        or object_identity == resolved
    ):
        raise ValueError("formal cache tag must be annotated and distinct from commit")
    return object_identity, resolved


def _download_bundle(
    *,
    download_fn: Callable[..., str],
    contract: Any,
    revision: str,
    expected: Mapping[str, bytes],
    fresh_parent: Path,
) -> None:
    destination = _destination(contract)
    with tempfile.TemporaryDirectory(prefix="gate-v1-formal-cache-", dir=fresh_parent) as raw:
        root = Path(raw).resolve()
        if any(root.iterdir()) or root.is_symlink():
            raise ValueError("fresh HF replay directory did not start empty")
        for remote_path in _remote_paths(contract):
            returned = Path(
                download_fn(
                    repo_id=destination["repo"],
                    filename=remote_path,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=root,
                    force_download=True,
                )
            )
            target = root / remote_path
            if returned != target or not returned.is_absolute():
                raise ValueError("HF download returned a noncanonical path")
            current = root
            for component in returned.relative_to(root).parts[:-1]:
                current /= component
                if not stat.S_ISDIR(current.lstat().st_mode):
                    raise ValueError("HF download traversed a symlink directory")
            payload, _ = _regular_file_bytes(returned, label=remote_path)
            if payload != expected[remote_path]:
                raise ValueError("fresh immutable HF cache bytes drifted")


def inspect_remote_state(
    *,
    api: Any,
    download_fn: Callable[..., str],
    contract: Any,
    expected: Mapping[str, bytes],
    fresh_parent: Path,
) -> RemoteInspection:
    destination = _destination(contract)
    info = _revision_info(api, contract, "main")
    if getattr(info, "private", None) is not True:
        raise ValueError("formal cache HF repository is not private")
    main = _revision_sha(api, contract, "main")
    files = _repo_files(api, contract, main)
    targets = set(_remote_paths(contract))
    present = targets & set(files)
    if present and present != targets:
        raise ValueError("formal cache remote is a partial exact-three conflict")
    tag = _tag_snapshot(api, contract)
    if tag is not None:
        object_identity, resolved = tag
        if main != resolved or present != targets:
            raise ValueError("tagged formal cache does not match main exact-three")
        _download_bundle(
            download_fn=download_fn,
            contract=contract,
            revision=resolved,
            expected=expected,
            fresh_parent=fresh_parent,
        )
        return RemoteInspection(
            REMOTE_TAGGED_BYTE_IDENTICAL,
            main,
            files,
            resolved,
            object_identity,
            resolved,
        )
    if not present:
        return RemoteInspection(REMOTE_EMPTY, main, files, None, None, None)
    _download_bundle(
        download_fn=download_fn,
        contract=contract,
        revision=main,
        expected=expected,
        fresh_parent=fresh_parent,
    )
    return RemoteInspection(
        REMOTE_BYTE_IDENTICAL_UNTAGGED,
        main,
        files,
        main,
        None,
        None,
    )


def _operation(factory: Callable[..., Any], path: str, payload: bytes) -> Any:
    return factory(path_in_repo=path, path_or_fileobj=io.BytesIO(payload))


def _response_commit(value: Any) -> str:
    for name in ("oid", "commit_id"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, str) and _COMMIT.fullmatch(candidate):
            return candidate
    raise ValueError("HF commit response omitted immutable commit id")


def _capture_remote_base(
    contract: Any,
    inspection: RemoteInspection,
    *,
    claim_sha256: str,
    history: Sequence[Mapping[str, str]],
    blobs: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if inspection.state != REMOTE_EMPTY:
        raise ValueError("remote-base receipt requires EMPTY state")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": REMOTE_BASE_STATUS,
        "contract_sha256": _contract_sha(contract),
        "claim_sha256": claim_sha256,
        "repo": _destination(contract)["repo"],
        "private": True,
        "base_main_revision": inspection.main_revision,
        "base_file_inventory": list(inspection.main_files),
        "base_reachable_history": [dict(item) for item in history],
        "base_reachable_history_sha256": sha256_bytes(
            canonical_json_bytes(history)
        ),
        "base_blob_inventory": [dict(item) for item in blobs],
        "base_blob_inventory_sha256": sha256_bytes(canonical_json_bytes(blobs)),
        "target_paths_absent": True,
        "tag_absent": True,
    }


def _validate_remote_base_receipt_record(
    contract: Any,
    receipt: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "contract_sha256",
        "claim_sha256",
        "repo",
        "private",
        "base_main_revision",
        "base_file_inventory",
        "base_reachable_history",
        "base_reachable_history_sha256",
        "base_blob_inventory",
        "base_blob_inventory_sha256",
        "target_paths_absent",
        "tag_absent",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status") != REMOTE_BASE_STATUS
        or receipt.get("contract_sha256") != _contract_sha(contract)
        or receipt.get("repo") != _destination(contract)["repo"]
        or receipt.get("private") is not True
        or receipt.get("target_paths_absent") is not True
        or receipt.get("tag_absent") is not True
    ):
        raise ValueError("remote-base receipt schema or frozen flags drifted")
    inventory = receipt.get("base_file_inventory")
    history = receipt.get("base_reachable_history")
    blobs = receipt.get("base_blob_inventory")
    if (
        not isinstance(inventory, list)
        or inventory != sorted(inventory)
        or len(inventory) != len(set(inventory))
        or any(_safe_relative(item, "remote-base file") != item for item in inventory)
        or not isinstance(history, list)
        or receipt.get("base_reachable_history_sha256")
        != sha256_bytes(canonical_json_bytes(history))
        or not isinstance(blobs, list)
        or receipt.get("base_blob_inventory_sha256")
        != sha256_bytes(canonical_json_bytes(blobs))
    ):
        raise ValueError("remote-base receipt inventory or history drifted")


def _validate_remote_base(
    contract: Any,
    receipt: Mapping[str, Any],
    inspection: RemoteInspection,
    api: Any,
) -> None:
    _validate_remote_base_receipt_record(contract, receipt)
    targets = set(_remote_paths(contract))
    non_targets = tuple(path for path in inspection.main_files if path not in targets)
    base = receipt.get("base_main_revision")
    if not isinstance(base, str) or _COMMIT.fullmatch(base) is None:
        raise ValueError("remote-base commit identity drifted")
    base_history = _remote_history(api, contract, base)
    base_blobs = _remote_blob_inventory(api, contract, base)
    if (
        receipt.get("status") != REMOTE_BASE_STATUS
        or receipt.get("contract_sha256") != _contract_sha(contract)
        or receipt.get("repo") != _destination(contract)["repo"]
        or receipt.get("private") is not True
        or tuple(receipt.get("base_file_inventory", ())) != non_targets
        or receipt.get("base_reachable_history")
        != [dict(item) for item in base_history]
        or receipt.get("base_reachable_history_sha256")
        != sha256_bytes(canonical_json_bytes(base_history))
        or receipt.get("base_blob_inventory")
        != [dict(item) for item in base_blobs]
        or receipt.get("base_blob_inventory_sha256")
        != sha256_bytes(canonical_json_bytes(base_blobs))
    ):
        raise ValueError("remote-base receipt or non-target inventory drifted")
    if inspection.state == REMOTE_EMPTY:
        if inspection.main_revision != base:
            raise ValueError("remote main interposed after base receipt")
        return
    immutable = inspection.immutable_revision
    if immutable is None:
        raise ValueError("published remote lacks immutable revision")
    pair_history = _remote_history(api, contract, immutable)
    pair_blobs = _remote_blob_inventory(api, contract, immutable)
    targets = set(_remote_paths(contract))
    target_blobs = [item for item in pair_blobs if item["path"] in targets]
    non_target_blobs = tuple(
        item for item in pair_blobs if item["path"] not in targets
    )
    if (
        len(pair_history) != len(base_history) + 1
        or pair_history[0]["commit_id"] != immutable
        or pair_history[0]["title"] != _destination(contract)["commit_title"]
        or pair_history[1:] != base_history
        or {item["path"] for item in target_blobs} != targets
        or len(target_blobs) != 3
        or non_target_blobs != base_blobs
    ):
        raise ValueError("exact-three commit is not the titled direct child of base")


def _create_tag_recovering(
    api: Any,
    contract: Any,
    revision: str,
) -> tuple[str, str]:
    destination = _destination(contract)
    try:
        api.create_tag(
            destination["repo"],
            tag=destination["tag"],
            tag_message=TAG_MESSAGE,
            revision=revision,
            repo_type="dataset",
            exist_ok=False,
        )
    except Exception:
        recovered = _tag_snapshot(api, contract)
        if recovered is None or recovered[1] != revision:
            raise
        return recovered
    observed = _tag_snapshot(api, contract)
    if observed is None or observed[1] != revision:
        raise ValueError("annotated HF tag did not resolve to intended commit")
    return observed


def _publish_remote(
    *,
    api: Any,
    download_fn: Callable[..., str],
    operation_factory: Callable[..., Any],
    contract: Any,
    expected: Mapping[str, bytes],
    fresh_parent: Path,
    remote_base: Mapping[str, Any],
) -> tuple[RemoteInspection, int]:
    inspection = inspect_remote_state(
        api=api,
        download_fn=download_fn,
        contract=contract,
        expected=expected,
        fresh_parent=fresh_parent,
    )
    _validate_remote_base(contract, remote_base, inspection, api)
    mutations = 0
    if inspection.state == REMOTE_EMPTY:
        if (
            inspection.main_revision != remote_base["base_main_revision"]
            or tuple(inspection.main_files)
            != tuple(remote_base["base_file_inventory"])
        ):
            raise ValueError("remote interposition occurred after base receipt")
        operations = [
            _operation(operation_factory, path, expected[path])
            for path in _remote_paths(contract)
        ]
        destination = _destination(contract)
        mutations += 1
        try:
            response = api.create_commit(
                destination["repo"],
                operations=operations,
                commit_message=destination["commit_title"],
                repo_type="dataset",
                revision="main",
                parent_commit=inspection.main_revision,
            )
            immutable = _response_commit(response)
        except Exception:
            recovered = inspect_remote_state(
                api=api,
                download_fn=download_fn,
                contract=contract,
                expected=expected,
                fresh_parent=fresh_parent,
            )
            _validate_remote_base(contract, remote_base, recovered, api)
            if recovered.state not in {
                REMOTE_BYTE_IDENTICAL_UNTAGGED,
                REMOTE_TAGGED_BYTE_IDENTICAL,
            }:
                raise
            immutable = recovered.immutable_revision
        if immutable is None:
            raise ValueError("HF exact-three commit has no immutable revision")
        inspection = inspect_remote_state(
            api=api,
            download_fn=download_fn,
            contract=contract,
            expected=expected,
            fresh_parent=fresh_parent,
        )
        _validate_remote_base(contract, remote_base, inspection, api)
        if inspection.immutable_revision != immutable:
            raise ValueError("HF commit response differs from observed exact-three")
    if inspection.state == REMOTE_BYTE_IDENTICAL_UNTAGGED:
        _validate_remote_base(contract, remote_base, inspection, api)
        assert inspection.immutable_revision is not None
        before = _tag_snapshot(api, contract)
        if before is not None:
            raise ValueError("tag interposition occurred before annotated tag creation")
        mutations += 1
        _create_tag_recovering(api, contract, inspection.immutable_revision)
    final = inspect_remote_state(
        api=api,
        download_fn=download_fn,
        contract=contract,
        expected=expected,
        fresh_parent=fresh_parent,
    )
    _validate_remote_base(contract, remote_base, final, api)
    if final.state != REMOTE_TAGGED_BYTE_IDENTICAL:
        raise ValueError("formal cache remote did not reach tagged exact-three")
    return final, mutations


def _bundle_manifest(
    contract: Any,
    *,
    source: SourceIdentity,
    feature: CacheArtifact,
    label: CacheArtifact,
    join_audit: Mapping[str, Any],
) -> Mapping[str, Any]:
    _assert_zero_operations(join_audit, label="join-only audit")
    execution_counts = _validate_execution_operation_audit(contract, join_audit)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": SIDECAR_STATUS,
        "contract_sha256": _contract_sha(contract),
        "source": dict(_source_record(source)),
        "feature_cache": dict(_artifact_record(feature.payload, feature.audit)),
        "label_cache": dict(_artifact_record(label.payload, label.audit)),
        "join_only_audit": dict(join_audit),
        "execution_operation_counts": dict(execution_counts),
        "formal_consumption": {
            "formal_cache_only": True,
            "formal58_cache_loader_eligible_by_sidecar_alone": False,
            "formal58_training_input_eligible_by_sidecar_alone": False,
            "gate_training_authorized_by_sidecar_alone": False,
            "fresh_immutable_replay_required": True,
            "final_completion_required": True,
            "gate_trained": False,
            "development_access_authorized": False,
            "confirm_access_authorized": False,
            "matched_nll_authorized": False,
            "closed_loop_authorized": False,
        },
    }


def _prepared_bundle(
    contract: Any,
    *,
    source: SourceIdentity,
    feature: CacheArtifact,
    label: CacheArtifact,
    join_audit: Mapping[str, Any],
) -> PreparedBundle:
    manifest = _bundle_manifest(
        contract,
        source=source,
        feature=feature,
        label=label,
        join_audit=join_audit,
    )
    manifest_bytes = pretty_json_bytes(manifest)
    paths = _remote_paths(contract)
    return PreparedBundle(
        feature=feature,
        label=label,
        join_audit=join_audit,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        files={paths[0]: feature.payload, paths[1]: label.payload, paths[2]: manifest_bytes},
    )


def _publish_or_read_json_state(
    path: Path,
    record: Mapping[str, Any],
    *,
    mode: int,
    allow_create: bool,
) -> tuple[bytes, bool]:
    payload = pretty_json_bytes(record)
    if allow_create:
        _, created = exclusive_or_identical_bytes(path, payload, mode=mode)
    else:
        created = False
        _read_state(path, record, mode=mode)
    return payload, created


def _assert_state_order(contract: Any) -> None:
    paths = (
        _state_path(contract, "global_claim_path", "claim_path"),
        _feature_path(contract),
        _state_path(contract, "feature_completion_path"),
        _state_path(contract, "label_access_claim_path"),
        _label_path(contract),
        _state_path(contract, "label_completion_path"),
        _state_path(contract, "remote_base_receipt_path", "remote_base_path"),
        _state_path(contract, "completion_staging_path", "retained_completion_staging_path"),
        _state_path(contract, "completion_path", "final_completion_path"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    observed_gap = False
    for exists in present:
        if not exists:
            observed_gap = True
        elif observed_gap:
            raise ValueError("formal cache state is orphaned or out of phase order")
    if present[-1]:
        staged, staged_meta = _regular_file_bytes(paths[-2], label="retained completion", mode=0o600)
        final, final_meta = _regular_file_bytes(paths[-1], label="final completion", mode=0o600)
        if (
            staged != final
            or staged_meta.st_dev != final_meta.st_dev
            or staged_meta.st_ino != final_meta.st_ino
        ):
            raise ValueError("final completion is not the retained-stage hard link")


def _publish_final_hardlink_last(
    staging: Path,
    final: Path,
    payload: bytes,
) -> None:
    staged, _ = _regular_file_bytes(staging, label="retained completion", mode=0o600)
    if staged != payload or final.exists() or final.is_symlink():
        raise ValueError("final completion precondition drifted")
    os.link(staging, final)
    _fsync_directory(final.parent)


def execute_formal_cache(
    *,
    mode: str,
    contract: Any,
    hooks: FormalCacheHooks,
    api: Any,
    download_fn: Callable[..., str],
    operation_factory: Callable[..., Any],
    expected_execution_b_git_commit: str,
    data_root: str | Path,
    fresh_download_parent: str | Path,
) -> Mapping[str, Any]:
    if mode not in {"run", "validate"}:
        raise ValueError("formal cache execution mode must be run or validate")
    data = Path(data_root)
    fresh_parent = Path(fresh_download_parent)
    if (
        not data.is_absolute()
        or not data.is_dir()
        or data.is_symlink()
        or not fresh_parent.is_absolute()
        or not fresh_parent.is_dir()
        or fresh_parent.is_symlink()
    ):
        raise ValueError("formal cache data and fresh-download roots must be real absolute directories")
    _assert_state_order(contract)
    run_mode = mode == "run"
    final_state_path = _state_path(
        contract, "completion_path", "final_completion_path"
    )
    completed_at_entry = final_state_path.exists() or final_state_path.is_symlink()
    write_allowed = run_mode and not completed_at_entry
    if mode == "validate":
        if not completed_at_entry:
            raise ValueError("read-only validate requires completed formal cache state")
    source = validate_execution_b_source(
        contract, expected_execution_b_git_commit=expected_execution_b_git_commit
    )
    runtime = validate_no_gpu_or_model_runtime(
        require_environment=True,
        contract=contract,
    )
    mode_bits = _state_mode(contract)
    claim_path = _state_path(contract, "global_claim_path", "claim_path")
    claim = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": GLOBAL_CLAIM_STATUS,
        "contract_sha256": _contract_sha(contract),
        "source": dict(_source_record(source)),
        "runtime": dict(runtime),
        "data_root": str(data),
        "feature_before_label_required": True,
        "operation_counts": dict(_zero_operations()),
    }
    claim_bytes, claim_created = _publish_or_read_json_state(
        claim_path, claim, mode=mode_bits, allow_create=write_allowed
    )
    claim_sha = sha256_bytes(claim_bytes)

    feature = _normalize_artifact(hooks.build_feature(contract, data), "feature cache")
    feature_path = _feature_path(contract)
    if write_allowed:
        feature_created = _publish_artifact(feature_path, feature.payload, mode=0o444)
    else:
        feature_created = False
    feature_readback = _readback_artifact(
        feature_path,
        feature,
        hooks.read_feature,
        contract,
        label="feature cache",
    )
    feature_completion = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": FEATURE_COMPLETION_STATUS,
        "contract_sha256": _contract_sha(contract),
        "claim_sha256": claim_sha,
        "feature_cache": dict(_artifact_record(feature.payload, feature_readback)),
        "label_semantic_access_count": 0,
        "operation_counts": dict(_zero_operations()),
    }
    feature_completion_path = _state_path(contract, "feature_completion_path")
    feature_completion_bytes, feature_completion_created = _publish_or_read_json_state(
        feature_completion_path,
        feature_completion,
        mode=mode_bits,
        allow_create=write_allowed,
    )
    label_access_claim = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": LABEL_ACCESS_CLAIM_STATUS,
        "contract_sha256": _contract_sha(contract),
        "global_claim_sha256": claim_sha,
        "feature_completion_sha256": sha256_bytes(feature_completion_bytes),
        "label_semantic_access_authorized": True,
        "development_semantic_access_count": 0,
        "confirm_state_access_count": 0,
    }
    label_access_path = _state_path(contract, "label_access_claim_path")
    label_access_bytes, label_access_created = _publish_or_read_json_state(
        label_access_path,
        label_access_claim,
        mode=mode_bits,
        allow_create=write_allowed,
    )

    label = _normalize_artifact(hooks.build_label(contract, data), "label cache")
    label_path = _label_path(contract)
    if write_allowed:
        label_created = _publish_artifact(label_path, label.payload, mode=0o444)
    else:
        label_created = False
    label_readback = _readback_artifact(
        label_path,
        label,
        hooks.read_label,
        contract,
        label="label cache",
    )
    label_completion = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": LABEL_COMPLETION_STATUS,
        "contract_sha256": _contract_sha(contract),
        "global_claim_sha256": claim_sha,
        "label_access_claim_sha256": sha256_bytes(label_access_bytes),
        "label_cache": dict(_artifact_record(label.payload, label_readback)),
        "operation_counts": dict(_zero_operations()),
    }
    label_completion_path = _state_path(contract, "label_completion_path")
    label_completion_bytes, label_completion_created = _publish_or_read_json_state(
        label_completion_path,
        label_completion,
        mode=mode_bits,
        allow_create=write_allowed,
    )
    join_audit = dict(
        _mapping(
            hooks.join_only_audit(contract, feature_readback, label_readback),
            "join-only audit",
        )
    )
    _assert_zero_operations(join_audit, label="join-only audit")
    execution_counts = _validate_execution_operation_audit(contract, join_audit)
    prepared = _prepared_bundle(
        contract,
        source=source,
        feature=feature,
        label=label,
        join_audit=join_audit,
    )

    remote_base_path = _state_path(
        contract, "remote_base_receipt_path", "remote_base_path"
    )
    if remote_base_path.exists() or remote_base_path.is_symlink():
        remote_base_bytes, _ = _regular_file_bytes(
            remote_base_path, label="remote-base receipt", mode=mode_bits
        )
        remote_base = _strict_json(remote_base_bytes, label="remote-base receipt")
        if remote_base_bytes != pretty_json_bytes(remote_base):
            raise ValueError("remote-base receipt is not canonical pretty JSON")
    else:
        if not write_allowed:
            raise ValueError("read-only validate cannot create remote-base receipt")
        destination = _destination(contract)
        try:
            api.create_repo(
                destination["repo"], repo_type="dataset", private=True, exist_ok=True
            )
        except Exception:
            _revision_sha(api, contract, "main")
        empty = inspect_remote_state(
            api=api,
            download_fn=download_fn,
            contract=contract,
            expected=prepared.files,
            fresh_parent=fresh_parent,
        )
        remote_base = _capture_remote_base(
            contract,
            empty,
            claim_sha256=claim_sha,
            history=_remote_history(api, contract, empty.main_revision),
            blobs=_remote_blob_inventory(api, contract, empty.main_revision),
        )
        remote_base_bytes, _ = _publish_or_read_json_state(
            remote_base_path,
            remote_base,
            mode=mode_bits,
            allow_create=write_allowed,
        )
    _validate_remote_base_receipt_record(contract, remote_base)
    if remote_base.get("claim_sha256") != claim_sha:
        raise ValueError("remote-base receipt differs from global claim")

    before_mutation_counts = {
        "create_commit": getattr(api, "create_commit_call_count", None),
        "create_tag": getattr(api, "create_tag_call_count", None),
    }
    remote_write_allowed = write_allowed
    if remote_write_allowed:
        remote, remote_mutations = _publish_remote(
            api=api,
            download_fn=download_fn,
            operation_factory=operation_factory,
            contract=contract,
            expected=prepared.files,
            fresh_parent=fresh_parent,
            remote_base=remote_base,
        )
    else:
        remote = inspect_remote_state(
            api=api,
            download_fn=download_fn,
            contract=contract,
            expected=prepared.files,
            fresh_parent=fresh_parent,
        )
        _validate_remote_base(contract, remote_base, remote, api)
        if remote.state != REMOTE_TAGGED_BYTE_IDENTICAL:
            raise ValueError("completed/read-only replay requires tagged exact-three remote")
        remote_mutations = 0
    after_mutation_counts = {
        "create_commit": getattr(api, "create_commit_call_count", None),
        "create_tag": getattr(api, "create_tag_call_count", None),
    }
    if (mode == "validate" or completed_at_entry) and (
        before_mutation_counts != after_mutation_counts
    ):
        raise ValueError("read-only validation mutated the remote")
    assert remote.immutable_revision is not None
    assert remote.tag_object_identity is not None
    _download_bundle(
        download_fn=download_fn,
        contract=contract,
        revision=remote.immutable_revision,
        expected=prepared.files,
        fresh_parent=fresh_parent,
    )
    final = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": FINAL_COMPLETION_STATUS,
        "contract_sha256": _contract_sha(contract),
        "global_claim_sha256": claim_sha,
        "feature_completion_sha256": sha256_bytes(feature_completion_bytes),
        "label_access_claim_sha256": sha256_bytes(label_access_bytes),
        "label_completion_sha256": sha256_bytes(label_completion_bytes),
        "remote_base_receipt_sha256": sha256_bytes(remote_base_bytes),
        "immutable_revision": remote.immutable_revision,
        "annotated_tag_object_identity": remote.tag_object_identity,
        "tag_resolved_commit": remote.tag_resolved_commit,
        "remote_exact_three_paths": list(_remote_paths(contract)),
        "remote_file_sha256": {
            path: sha256_bytes(payload) for path, payload in prepared.files.items()
        },
        "join_only_audit_sha256": sha256_bytes(canonical_json_bytes(join_audit)),
        "execution_operation_counts": dict(execution_counts),
        "formal_cache_only": True,
        "formal58_cache_loader_eligible": True,
        "formal58_training_input_eligible": True,
        "gate_training_authorized": True,
        "gate_trained": False,
        "development_access_authorized": False,
        "confirm_access_authorized": False,
        "matched_nll_authorized": False,
        "closed_loop_authorized": False,
    }
    final_bytes = pretty_json_bytes(final)
    staging_path = _state_path(
        contract, "completion_staging_path", "retained_completion_staging_path"
    )
    final_path = final_state_path
    _, staging_created = exclusive_or_identical_bytes(
        staging_path, final_bytes, mode=mode_bits
    ) if write_allowed else (sha256_bytes(_read_state(staging_path, final, mode=mode_bits)), False)
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": RUN_STATUS if mode == "run" else VALIDATE_STATUS,
        "contract_sha256": _contract_sha(contract),
        "source_git_commit": source.head,
        "claim_created": claim_created,
        "feature_cache_created": feature_created,
        "feature_completion_created": feature_completion_created,
        "label_access_claim_created": label_access_created,
        "label_cache_created": label_created,
        "label_completion_created": label_completion_created,
        "completion_staging_created": staging_created,
        "immutable_revision": remote.immutable_revision,
        "annotated_tag_object_identity": remote.tag_object_identity,
        "remote_mutation_call_count": remote_mutations,
        "formal_cache_only": True,
        "formal58_cache_loader_eligible": True,
        "formal58_training_input_eligible": True,
        "gate_training_authorized": True,
        "gate_trained": False,
    }
    if final_path.exists() or final_path.is_symlink():
        staged, staged_meta = _regular_file_bytes(
            staging_path, label="retained completion", mode=mode_bits
        )
        completed, completed_meta = _regular_file_bytes(
            final_path, label="final completion", mode=mode_bits
        )
        if (
            staged != final_bytes
            or completed != final_bytes
            or staged_meta.st_dev != completed_meta.st_dev
            or staged_meta.st_ino != completed_meta.st_ino
        ):
            raise ValueError("completed replay final bytes or inode drifted")
        result["remote_mutation_call_count"] = 0
        return result
    if not write_allowed:
        raise ValueError("read-only validate cannot create final completion")

    # note (luojiaxuan): This hard link is the terminal mutation. All cache,
    # source, remote, fresh-replay, and retained-stage checks occur above.
    _publish_final_hardlink_last(staging_path, final_path, final_bytes)
    return result


__all__ = [
    "CacheArtifact",
    "FINAL_COMPLETION_STATUS",
    "FormalCacheHooks",
    "GLOBAL_CLAIM_STATUS",
    "LABEL_ACCESS_CLAIM_STATUS",
    "LABEL_COMPLETION_STATUS",
    "PAIR_COMMIT_TITLE",
    "PROTOCOL_ID",
    "REMOTE_BYTE_IDENTICAL_UNTAGGED",
    "REMOTE_EMPTY",
    "REMOTE_TAGGED_BYTE_IDENTICAL",
    "RUNNER_FREEZE_PATH",
    "RUNNER_FREEZE_STATUS",
    "RUN_STATUS",
    "SIDECAR_STATUS",
    "SOURCE_VALIDATION_STATUS",
    "VALIDATE_STATUS",
    "execute_formal_cache",
    "exclusive_or_identical_bytes",
    "inspect_remote_state",
    "load_runner_freeze",
    "materialize_runner_freeze",
    "sha256_bytes",
    "validate_clean_pushed_source",
    "validate_execution_b_source",
    "validate_no_gpu_or_model_runtime",
    "validate_source_a",
]
