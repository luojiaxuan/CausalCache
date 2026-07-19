"""Git-safe formal result recording for the processor image-contract v2 run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import causalcache.set_utility_processor_freeze_contract_v2 as _contract_v2_module
import causalcache.set_utility_processor_postflight as _structural_postflight_module
import causalcache.set_utility_processor_postflight_v2 as _postflight_v2_module
from causalcache.set_utility_processor_artifacts import (
    ARTIFACT_FILENAME_TEMPLATE,
    CANDIDATE_SCHEDULE_FILENAME,
    canonical_json_bytes,
    canonical_pretty_json_bytes,
    sha256_bytes,
)
from causalcache.set_utility_processor_freeze import WORKER_COUNT
from causalcache.set_utility_processor_freeze_contract_v2 import (
    CANONICAL_EXECUTION_CONFIG_PATH,
    CONTRACT_PATH,
    OUTPUT_VALIDATOR_PATH,
    POSTFLIGHT_PATH,
    REQUIRED_IDENTITY_ARGUMENTS,
    REQUIRED_PATH_ARGUMENTS,
    REQUIRED_VERSION_ARGUMENTS_BY_PHASE,
    RUNNER_PATH,
    load_execution_contract,
)
from causalcache.set_utility_processor_postflight_v2 import (
    VALIDATION_STATUS,
    build_processor_freeze_postflight_context_v2,
    validate_completed_processor_freeze_root_v2,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = (
    "causalcache_set_utility_processor_freeze_v2_formal_result_record"
)
VALID_RESULT_STATUS = (
    "VALID_RECORDED_SET_UTILITY_PROCESSOR_FREEZE_V2_IMAGE_CONTRACT_REPAIR"
)
INVALID_RESULT_STATUS = (
    "INVALID_RECORDED_SET_UTILITY_PROCESSOR_FREEZE_V2_FORMAL_ATTEMPT"
)
PENDING_PUBLICATION_STATUS = "PENDING_HF_UPLOAD"
INVALID_PUBLICATION_STATUS = "NOT_ELIGIBLE_FOR_HF_PUBLICATION"
EXPECTED_FORMAL_FILE_COUNT = 23
EXPECTED_RESULT_FILES = frozenset({"README.md", "summary.json"})
RECORDER_MODULE_PATH = (
    "code/causalcache/set_utility_processor_result_v2.py"
)
RECORDER_CLI_PATH = (
    "code/scripts/record_set_utility_processor_freeze_v2_result.py"
)
STRUCTURAL_POSTFLIGHT_PATH = (
    "code/causalcache/set_utility_processor_postflight.py"
)
RECORDER_SOURCE_PATHS = (
    RECORDER_MODULE_PATH,
    RECORDER_CLI_PATH,
    POSTFLIGHT_PATH,
    CONTRACT_PATH,
    STRUCTURAL_POSTFLIGHT_PATH,
)
START_EVIDENCE_KEYS = frozenset(
    {
        "started_at_utc",
        "shell_pid",
        "git_revision",
        "host_alias",
        "host_hostname",
        "container_id",
        "worker_count",
        "output_root",
    }
)
END_EVIDENCE_KEYS = frozenset(
    {"ended_at_utc", "watcher_pid", "exit_code", "output_root"}
)

_GIT_REVISION = re.compile(r"[0-9a-f]{40}")
_HF_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_HF_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_FAILURE_ERROR_CODES = {
    "runtime_identity": "RUNTIME_IDENTITY_INVALID",
    "run_argv_evidence": "RUN_ARGV_EVIDENCE_INVALID",
    "start_evidence": "START_EVIDENCE_INVALID",
    "outer_log_evidence": "OUTER_LOG_EVIDENCE_INVALID",
    "exit_code_evidence": "EXIT_CODE_EVIDENCE_INVALID",
    "end_evidence": "END_EVIDENCE_INVALID",
    "postflight_argv_evidence": "POSTFLIGHT_ARGV_EVIDENCE_INVALID",
    "postflight_evidence": "POSTFLIGHT_EVIDENCE_INVALID",
    "staging_absence": "STAGING_ROOT_PRESENT",
    "formal_tree_before": "FORMAL_TREE_PREFLIGHT_INVALID",
    "run_identity_before": "RUN_IDENTITY_PREFLIGHT_MISMATCH",
    "fresh_postflight": "FRESH_POSTFLIGHT_INVALID",
    "formal_tree_after": "FORMAL_TREE_POSTFLIGHT_INVALID",
    "run_identity_after": "RUN_IDENTITY_POSTFLIGHT_MISMATCH",
    "postflight_evidence_match": "POSTFLIGHT_EVIDENCE_MISMATCH",
    "formal_exit": "FORMAL_EXIT_NONZERO",
}


def _strict_json(payload: bytes, *, label: str) -> Any:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    value = _strict_json(payload, label=label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_evidence(
    path: str | Path,
    *,
    label: str,
) -> tuple[bytes, dict[str, Any]]:
    source = Path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for evidence reads")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        path_before = os.stat(source, follow_symlinks=False)
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"{label} must be one regular non-symlink file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be one regular non-symlink file")
        blocks: list[bytes] = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            blocks.append(block)
        payload = b"".join(blocks)
        after = os.fstat(descriptor)
        path_after = os.stat(source, follow_symlinks=False)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    path_before_identity = (
        path_before.st_dev,
        path_before.st_ino,
        path_before.st_mode,
        path_before.st_size,
        path_before.st_mtime_ns,
    )
    path_after_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_mode,
        path_after.st_size,
        path_after.st_mtime_ns,
    )
    if (
        before_identity != after_identity
        or before_identity != path_before_identity
        or after_identity != path_after_identity
        or len(payload) != after.st_size
    ):
        raise RuntimeError(f"{label} changed while it was being recorded")
    return payload, {
        "byte_count": len(payload),
        "path": str(source.resolve()),
        "sha256": sha256_bytes(payload),
    }


def _evidence_descriptor(path: str | Path, *, label: str) -> dict[str, Any]:
    _, descriptor = _read_evidence(path, label=label)
    return descriptor


def _expected_formal_paths() -> frozenset[str]:
    result = {
        "manifest.json",
        "run-identity.json",
        CANDIDATE_SCHEDULE_FILENAME,
    }
    result.update(
        f"candidate-parts/worker-{worker:02d}.jsonl"
        for worker in range(WORKER_COUNT)
    )
    result.update(
        f"logs/{phase}-worker-{worker:02d}.log"
        for phase in ("ocr", "processor")
        for worker in range(WORKER_COUNT)
    )
    result.update(
        f"receipts/ocr-worker-{worker:02d}.json"
        for worker in range(WORKER_COUNT)
    )
    result.update(
        "substrate/"
        + ARTIFACT_FILENAME_TEMPLATE.format(worker_index=worker)
        for worker in range(WORKER_COUNT)
    )
    if len(result) != EXPECTED_FORMAL_FILE_COUNT:
        raise AssertionError("processor formal file contract no longer has 23 files")
    return frozenset(result)


def _formal_tree_snapshot(root: Path) -> tuple[
    tuple[dict[str, Any], ...], tuple[tuple[str, int, int, str | None], ...]
]:
    if not root.is_absolute():
        raise ValueError("formal output root must be absolute")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("formal output root must be one real directory")
    inventory: list[dict[str, Any]] = []
    snapshot: list[tuple[str, int, int, str | None]] = []

    def visit(path: Path, relative: str) -> None:
        metadata = path.lstat()
        mode = metadata.st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("formal output tree must not contain symlinks")
        if stat.S_ISREG(mode):
            digest = _sha256_file(path)
            snapshot.append((relative, mode, metadata.st_size, digest))
            inventory.append(
                {
                    "path": relative,
                    "sha256": digest,
                    "size_bytes": metadata.st_size,
                }
            )
            return
        if not stat.S_ISDIR(mode):
            raise ValueError("formal output tree contains a non-file entry")
        snapshot.append((relative, mode, metadata.st_size, None))
        for entry in sorted(os.scandir(path), key=lambda item: item.name):
            child_relative = (
                entry.name if relative == "." else f"{relative}/{entry.name}"
            )
            visit(path / entry.name, child_relative)

    visit(root, ".")
    observed_paths = frozenset(record["path"] for record in inventory)
    expected_paths = _expected_formal_paths()
    if observed_paths != expected_paths:
        raise ValueError(
            "formal output 23-file inventory drifted: "
            f"missing={sorted(expected_paths - observed_paths)}, "
            f"extra={sorted(observed_paths - expected_paths)}"
        )
    return tuple(inventory), tuple(snapshot)


def _formal_inventory_summary(
    inventory: Sequence[Mapping[str, Any]],
    snapshot: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    if len(inventory) != EXPECTED_FORMAL_FILE_COUNT:
        raise ValueError("formal output file count differs from 23")
    normalized_inventory = [dict(record) for record in inventory]
    normalized_snapshot = [list(record) for record in snapshot]
    return {
        "file_count": len(normalized_inventory),
        "file_inventory": normalized_inventory,
        "file_inventory_sha256": sha256_bytes(
            canonical_json_bytes(normalized_inventory)
        ),
        "output_tree_snapshot_sha256": sha256_bytes(
            canonical_json_bytes(normalized_snapshot)
        ),
        "total_byte_count": sum(
            int(record["size_bytes"]) for record in normalized_inventory
        ),
    }


def _runtime_cli(
    output_root: Path,
    *,
    producer_repository_root: Path,
    producer_execution_config: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    path = output_root / "run-identity.json"
    payload, descriptor = _read_evidence(path, label="formal run identity")
    identity = _strict_json_object(payload, label="formal run identity")
    runtime = identity.get("runtime_cli")
    if not isinstance(runtime, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in runtime.items()
    ):
        raise ValueError("formal run identity runtime_cli is invalid")
    normalized = dict(runtime)
    if (
        normalized.get("repository_root")
        != str(producer_repository_root.resolve())
        or normalized.get("execution_config")
        != str(producer_execution_config.resolve())
    ):
        raise ValueError(
            "formal run identity differs from the explicit producer checkout"
        )
    return normalized, descriptor


def _expected_run_argv(
    runtime: Mapping[str, str],
) -> list[str]:
    producer_root = Path(runtime["repository_root"])
    if not producer_root.is_absolute():
        raise ValueError("formal run identity repository_root must be absolute")
    result = [
        runtime["processor_python_executable"],
        str((producer_root / RUNNER_PATH).resolve()),
    ]
    for argument in REQUIRED_PATH_ARGUMENTS:
        key = argument.removeprefix("--").replace("-", "_")
        result.extend((argument, runtime[key]))
    for argument in REQUIRED_IDENTITY_ARGUMENTS:
        key = argument.removeprefix("--").replace("-", "_")
        result.extend((argument, runtime[key]))
    for arguments in REQUIRED_VERSION_ARGUMENTS_BY_PHASE.values():
        for argument in arguments:
            key = argument.removeprefix("--").replace("-", "_")
            result.extend((argument, runtime[key]))
    return result


def _validate_run_argv_evidence(
    path: str | Path,
    *,
    runtime: Mapping[str, str],
) -> tuple[list[str], dict[str, Any]]:
    payload, descriptor = _read_evidence(path, label="run argv evidence")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("run argv evidence must be UTF-8") from error
    if not text.endswith("\n") or len(text.splitlines()) != 1:
        raise ValueError("run argv evidence must be exactly one newline-terminated line")
    try:
        tokens = shlex.split(text[:-1], posix=True)
    except ValueError as error:
        raise ValueError("run argv evidence is not valid shell tokenization") from error
    launcher = ["env", "-u", "PYTHONPATH"]
    expected = _expected_run_argv(runtime)
    if tokens != [*launcher, *expected]:
        raise ValueError("run argv evidence differs from the formal run identity")
    return expected, {
        **descriptor,
        "launcher_prefix": launcher,
    }


def _utc_timestamp(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise ValueError(f"{label} must use YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(f"{label} is not a real UTC timestamp") from error
    return value


def _validate_start_evidence(
    path: str | Path,
    *,
    expected_git_revision: str,
    output_root: Path,
    runtime: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, descriptor = _read_evidence(path, label="start evidence")
    value = _strict_json_object(payload, label="start evidence")
    if set(value) != START_EVIDENCE_KEYS:
        raise ValueError("start evidence fields drifted")
    _utc_timestamp(value["started_at_utc"], label="start evidence timestamp")
    if type(value["shell_pid"]) is not int or value["shell_pid"] <= 0:
        raise ValueError("start evidence shell_pid must be positive")
    expected = {
        "git_revision": expected_git_revision,
        "host_alias": runtime["host_alias"],
        "host_hostname": runtime["host_hostname"],
        "container_id": runtime["container_id"],
        "worker_count": int(runtime["worker_count"]),
        "output_root": str(output_root.resolve()),
    }
    if any(value[key] != expected_value for key, expected_value in expected.items()):
        raise ValueError("start evidence differs from the formal run identity")
    return value, descriptor


def _validate_exit_code_evidence(
    path: str | Path,
) -> tuple[int, dict[str, Any]]:
    payload, descriptor = _read_evidence(path, label="exit-code evidence")
    if re.fullmatch(rb"(?:0|[1-9][0-9]*)\n", payload) is None:
        raise ValueError("exit-code evidence must be one non-negative integer line")
    return int(payload), descriptor


def _validate_end_evidence(
    path: str | Path,
    *,
    output_root: Path,
    expected_exit_code: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, descriptor = _read_evidence(path, label="end evidence")
    value = _strict_json_object(payload, label="end evidence")
    if set(value) != END_EVIDENCE_KEYS:
        raise ValueError("end evidence fields drifted")
    _utc_timestamp(value["ended_at_utc"], label="end evidence timestamp")
    if type(value["watcher_pid"]) is not int or value["watcher_pid"] <= 0:
        raise ValueError("end evidence watcher_pid must be positive")
    if (
        type(value["exit_code"]) is not int
        or value["exit_code"] != expected_exit_code
    ):
        raise ValueError("end evidence exit code differs from exit-code evidence")
    if value["output_root"] != str(output_root.resolve()):
        raise ValueError("end evidence output root differs from the formal root")
    return value, descriptor


def _elapsed_seconds(started_at_utc: str, ended_at_utc: str) -> int:
    started = datetime.strptime(started_at_utc, "%Y-%m-%dT%H:%M:%SZ")
    ended = datetime.strptime(ended_at_utc, "%Y-%m-%dT%H:%M:%SZ")
    elapsed = int((ended - started).total_seconds())
    if elapsed < 0:
        raise ValueError("end evidence timestamp precedes start evidence")
    return elapsed


def _expected_postflight_argv(
    repository_root: Path,
    execution_config: Path,
    output_root: Path,
    expected_git_revision: str,
    runtime: Mapping[str, str],
) -> list[str]:
    return [
        runtime["processor_python_executable"],
        str((repository_root / OUTPUT_VALIDATOR_PATH).resolve()),
        "--repository-root",
        str(repository_root.resolve()),
        "--execution-config",
        str(execution_config.resolve()),
        "--output-root",
        str(output_root.resolve()),
        "--expected-git-revision",
        expected_git_revision,
    ]


def _validate_postflight_argv_evidence(
    path: str | Path,
    *,
    expected_argv: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    payload, descriptor = _read_evidence(
        path, label="postflight argv evidence"
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("postflight argv evidence must be UTF-8") from error
    if not text.endswith("\n") or len(text.splitlines()) != 1:
        raise ValueError(
            "postflight argv evidence must be exactly one newline-terminated line"
        )
    try:
        tokens = shlex.split(text[:-1], posix=True)
    except ValueError as error:
        raise ValueError(
            "postflight argv evidence is not valid shell tokenization"
        ) from error
    launcher = ["env", "-u", "PYTHONPATH"]
    expected = list(expected_argv)
    if tokens != [*launcher, *expected]:
        raise ValueError(
            "postflight argv evidence differs from the producer-bound validator"
        )
    return expected, {**descriptor, "launcher_prefix": launcher}


def _validate_hf_destination(repository: str, tag: str) -> None:
    if _HF_REPO.fullmatch(repository) is None:
        raise ValueError("intended Hugging Face repository is invalid")
    if _HF_TAG.fullmatch(tag) is None:
        raise ValueError("intended Hugging Face tag is invalid")


def _verify_clean_checkout(
    repository_root: Path,
    *,
    expected_revision: str,
    label: str,
) -> None:
    if _GIT_REVISION.fullmatch(expected_revision) is None:
        raise ValueError(
            f"expected {label} Git revision must contain 40 lowercase hex digits"
        )
    try:
        observed = subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        status = subprocess.check_output(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"{label} root must be one readable Git checkout") from error
    if observed != expected_revision:
        raise ValueError(f"{label} checkout HEAD differs from its expected revision")
    if status:
        raise ValueError(f"{label} checkout must be completely clean")


def _verify_module_origins(repository_root: Path) -> None:
    expected = {
        "recorder module": repository_root / RECORDER_MODULE_PATH,
        "v2 contract module": repository_root / CONTRACT_PATH,
        "v2 postflight module": repository_root / POSTFLIGHT_PATH,
        "structural postflight module": (
            repository_root / STRUCTURAL_POSTFLIGHT_PATH
        ),
    }
    observed = {
        "recorder module": Path(__file__),
        "v2 contract module": Path(_contract_v2_module.__file__),
        "v2 postflight module": Path(_postflight_v2_module.__file__),
        "structural postflight module": Path(
            _structural_postflight_module.__file__
        ),
    }
    if any(
        observed[label].resolve() != path.resolve()
        for label, path in expected.items()
    ):
        raise ValueError("executed module origin differs from recorder checkout")


def _recorder_source_bindings(repository_root: Path) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    for relative in RECORDER_SOURCE_PATHS:
        path = repository_root.joinpath(*Path(relative).parts)
        payload, descriptor = _read_evidence(
            path, label="recorder source binding"
        )
        bindings[relative] = {
            "byte_count": len(payload),
            "path": relative,
            "sha256": descriptor["sha256"],
        }
    return bindings


def _validate_git_result_dir(repository_root: Path, result_dir: Path) -> Path:
    if not result_dir.is_absolute():
        raise ValueError("Git result directory must be absolute")
    repository_root = repository_root.resolve()
    expected_parent = (repository_root / "data" / "results").resolve()
    destination = result_dir.resolve(strict=False)
    if destination == expected_parent:
        raise ValueError("Git result directory must be a child of data/results")
    try:
        destination.relative_to(expected_parent)
    except ValueError as error:
        raise ValueError("Git result directory must stay under data/results") from error
    current = repository_root
    for part in destination.relative_to(repository_root).parts[:-1]:
        current /= part
        if current.is_symlink():
            raise ValueError("Git result directory must not traverse a symlink")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Git result directory already exists")
    staging = destination.parent / f".{destination.name}.incomplete"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError("Git result staging directory already exists")
    return destination


def _safe_failure(error: Exception, *, stage: str) -> dict[str, str]:
    if stage not in _FAILURE_ERROR_CODES:
        raise AssertionError("uncontrolled formal-result failure stage")
    return {
        "error_code": _FAILURE_ERROR_CODES[stage],
        "exception_type": type(error).__name__,
        "stage": stage,
    }


def _render_readme(summary: Mapping[str, Any]) -> str:
    valid = summary["scientific_eligibility"] is True
    artifact = summary["artifact"]
    postflight = summary.get("postflight") or {}
    if valid:
        conclusion = (
            "本次 processor image-contract v2 formal root 已通过 fresh committed "
            "postflight，并被记录为可发布的 processor-only candidate freeze。"
        )
        status_block = str(postflight["status"])
        publication = (
            "artifact 尚未上传；当前严格状态为 `PENDING_HF_UPLOAD`。在完成 private "
            "Hugging Face 上传与 fresh-download 校验前，不得写成已发布。"
        )
    else:
        conclusion = (
            "本目录只记录一次无效 formal attempt；它没有通过完整 evidence/postflight "
            "闭包，不能作为 valid candidate freeze 或下游训练输入。"
        )
        status_block = str(summary["status"])
        publication = "该 attempt 不具备 Hugging Face publication eligibility。"
    return f"""# Processor Freeze v2 Formal Result

## 正式结论

{conclusion}

```text
{status_block}
```

## Git-safe 证据

- formal root：`{artifact.get('output_root', 'unavailable')}`；
- exact file count：`{artifact.get('file_count', 'unavailable')}`；
- file inventory SHA256：`{artifact.get('file_inventory_sha256', 'unavailable')}`；
- tree snapshot SHA256：`{artifact.get('output_tree_snapshot_sha256', 'unavailable')}`；
- execution config SHA256：`{summary['bindings'].get('execution_config_sha256', 'unavailable')}`；
- run identity SHA256：`{summary['bindings'].get('run_identity_sha256', 'unavailable')}`。

完整 argv、runtime evidence hashes、postflight hashes、operation budget、format histogram 与
negative-operation counts 见 [`summary.json`](summary.json)。本目录不复制 candidate/OCR/image/raw log。

## Publication 与边界

{publication}

本步骤的 policy inference、restoration labels、training、matched-NLL、closed-loop 与 HF mutation 均为 0。
"""


def _write_result(directory: Path, summary: Mapping[str, Any]) -> None:
    staging = directory.parent / f".{directory.name}.incomplete"
    staging.mkdir(parents=False, exist_ok=False)
    summary_bytes = canonical_pretty_json_bytes(summary)
    readme_bytes = _render_readme(summary).encode("utf-8")
    (staging / "summary.json").write_bytes(summary_bytes)
    (staging / "README.md").write_bytes(readme_bytes)
    observed = frozenset(entry.name for entry in os.scandir(staging))
    if observed != EXPECTED_RESULT_FILES:
        raise RuntimeError("Git result staging inventory drifted")
    if (staging / "summary.json").read_bytes() != summary_bytes or (
        staging / "README.md"
    ).read_bytes() != readme_bytes:
        raise RuntimeError("Git result readback drifted")
    os.rename(staging, directory)


def _inventory_sha256(
    inventory: Sequence[Mapping[str, Any]],
    *,
    relative_path: str,
) -> str:
    matches = [
        record.get("sha256")
        for record in inventory
        if record.get("path") == relative_path
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise ValueError("formal inventory omitted one required bound file")
    return matches[0]


def record_processor_freeze_v2_result(
    *,
    repository_root: str | Path,
    producer_repository_root: str | Path,
    execution_config: str | Path,
    output_root: str | Path,
    expected_git_revision: str,
    expected_recorder_git_revision: str,
    result_dir: str | Path,
    run_argv_evidence: str | Path,
    start_evidence: str | Path,
    outer_log: str | Path,
    exit_code_evidence: str | Path,
    end_evidence: str | Path,
    postflight_argv_evidence: str | Path,
    postflight_evidence: str | Path,
    intended_hf_repo: str,
    intended_hf_tag: str,
    record_invalid: bool = False,
) -> dict[str, Any]:
    """Validate formal/evidence roots and atomically write two Git-safe files."""
    recorder_root = Path(repository_root).resolve()
    producer_root = Path(producer_repository_root).resolve()
    producer_config = Path(execution_config).resolve()
    output = Path(output_root)
    _verify_clean_checkout(
        recorder_root,
        expected_revision=expected_recorder_git_revision,
        label="recorder",
    )
    if _GIT_REVISION.fullmatch(expected_git_revision) is None:
        raise ValueError("expected Git revision must contain 40 lowercase hex digits")
    _verify_clean_checkout(
        producer_root,
        expected_revision=expected_git_revision,
        label="producer",
    )
    _verify_module_origins(recorder_root)
    destination = _validate_git_result_dir(recorder_root, Path(result_dir))
    expected_producer_config = (
        producer_root / CANONICAL_EXECUTION_CONFIG_PATH
    ).resolve()
    recorder_config = (
        recorder_root / CANONICAL_EXECUTION_CONFIG_PATH
    ).resolve()
    if producer_config != expected_producer_config:
        raise ValueError(
            "execution config must be the producer canonical committed v2 config"
        )
    _validate_hf_destination(intended_hf_repo, intended_hf_tag)
    producer_config_payload, producer_config_descriptor = _read_evidence(
        producer_config, label="producer execution config"
    )
    recorder_config_payload, recorder_config_descriptor = _read_evidence(
        recorder_config, label="recorder execution config"
    )
    if (
        producer_config_payload != recorder_config_payload
        or producer_config_descriptor["sha256"]
        != recorder_config_descriptor["sha256"]
    ):
        raise ValueError("producer and recorder execution config bytes differ")
    contract = load_execution_contract(
        repository_root=producer_root,
        execution_config_path=producer_config,
    )
    if contract.config_sha256 != producer_config_descriptor["sha256"]:
        raise ValueError("loaded v2 contract/config bytes drifted")
    recorder_sources = _recorder_source_bindings(recorder_root)

    evidence: dict[str, Any] = {}
    validation_error: Exception | None = None
    failure_stage = "runtime_identity"
    postflight: dict[str, Any] | None = None
    postflight_payload = b""
    inventory_summary: dict[str, Any] = {}
    runtime: dict[str, str] = {}
    run_argv: list[str] = []
    postflight_argv: list[str] = []
    start: dict[str, Any] = {}
    end: dict[str, Any] = {}
    exit_code: int | None = None
    elapsed_seconds: int | None = None
    run_identity_descriptor: dict[str, Any] = {}
    try:
        failure_stage = "runtime_identity"
        runtime, run_identity_descriptor = _runtime_cli(
            output,
            producer_repository_root=producer_root,
            producer_execution_config=producer_config,
        )
        failure_stage = "run_argv_evidence"
        run_argv, evidence["run_argv"] = _validate_run_argv_evidence(
            run_argv_evidence,
            runtime=runtime,
        )
        failure_stage = "start_evidence"
        start, evidence["start"] = _validate_start_evidence(
            start_evidence,
            expected_git_revision=expected_git_revision,
            output_root=output,
            runtime=runtime,
        )
        failure_stage = "outer_log_evidence"
        evidence["outer_log"] = _evidence_descriptor(
            outer_log, label="outer log evidence"
        )
        failure_stage = "exit_code_evidence"
        exit_code, evidence["exit_code"] = _validate_exit_code_evidence(
            exit_code_evidence
        )
        failure_stage = "end_evidence"
        end, evidence["end"] = _validate_end_evidence(
            end_evidence,
            output_root=output,
            expected_exit_code=exit_code,
        )
        elapsed_seconds = _elapsed_seconds(
            start["started_at_utc"], end["ended_at_utc"]
        )
        failure_stage = "postflight_argv_evidence"
        expected_postflight_argv = _expected_postflight_argv(
            producer_root,
            producer_config,
            output,
            expected_git_revision,
            runtime,
        )
        postflight_argv, evidence["postflight_argv"] = (
            _validate_postflight_argv_evidence(
                postflight_argv_evidence,
                expected_argv=expected_postflight_argv,
            )
        )
        failure_stage = "postflight_evidence"
        postflight_payload, evidence["postflight"] = _read_evidence(
            postflight_evidence, label="postflight evidence"
        )
        failure_stage = "staging_absence"
        staging_root = output.parent / f".{output.name}.incomplete"
        if staging_root.exists() or staging_root.is_symlink():
            raise ValueError("resumable staging root remains beside formal output")
        failure_stage = "formal_tree_before"
        before_inventory, before_snapshot = _formal_tree_snapshot(output)
        failure_stage = "run_identity_before"
        if _inventory_sha256(
            before_inventory, relative_path="run-identity.json"
        ) != run_identity_descriptor.get("sha256"):
            raise RuntimeError(
                "run identity changed between evidence read and formal inventory"
            )
        failure_stage = "fresh_postflight"
        context = build_processor_freeze_postflight_context_v2(
            contract,
            expected_git_revision=expected_git_revision,
        )
        postflight = validate_completed_processor_freeze_root_v2(
            output,
            context=context,
        )
        failure_stage = "formal_tree_after"
        after_inventory, after_snapshot = _formal_tree_snapshot(output)
        if (
            after_inventory != before_inventory
            or after_snapshot != before_snapshot
        ):
            raise RuntimeError("formal output tree changed during result recording")
        failure_stage = "run_identity_after"
        if _inventory_sha256(
            after_inventory, relative_path="run-identity.json"
        ) != run_identity_descriptor.get("sha256"):
            raise RuntimeError(
                "run identity changed between evidence read and postflight inventory"
            )
        inventory_summary = _formal_inventory_summary(
            before_inventory, before_snapshot
        )
        failure_stage = "postflight_evidence_match"
        observed_postflight = _strict_json_object(
            postflight_payload,
            label="postflight evidence",
        )
        if observed_postflight != postflight:
            raise ValueError(
                "postflight evidence differs from fresh committed validation"
            )
        if postflight.get("status") != VALIDATION_STATUS:
            raise ValueError("fresh postflight did not return the exact valid status")
        if (
            postflight.get("run_identity_sha256")
            != run_identity_descriptor.get("sha256")
        ):
            raise ValueError("fresh postflight run-identity SHA256 drifted")
        failure_stage = "formal_exit"
        if exit_code != 0:
            raise ValueError("formal runner exit code is nonzero")
    except Exception as error:
        validation_error = error

    valid = validation_error is None
    if not valid and not record_invalid:
        assert validation_error is not None
        raise validation_error

    bindings = {
        "execution_config_sha256": contract.config_sha256,
        "producer_git_revision": expected_git_revision,
        "producer_repository_root": str(producer_root),
        "recorder_execution_config_sha256": recorder_config_descriptor[
            "sha256"
        ],
        "recorder_git_revision": expected_recorder_git_revision,
        "recorder_repository_root": str(recorder_root),
        "recorder_sources": recorder_sources,
        "run_identity_sha256": (
            postflight.get("run_identity_sha256") if postflight else None
        ),
        "processor_freeze_summary_sha256": (
            postflight.get("processor_freeze_summary_sha256")
            if postflight
            else None
        ),
        "image_contract_sha256": (
            postflight.get("image_contract_sha256") if postflight else None
        ),
    }
    execution = {
        "argv": run_argv,
        "device": "cpu",
        "dtype": "not_applicable",
        "evidence": evidence,
        "elapsed_seconds": elapsed_seconds,
        "ended_at_utc": end.get("ended_at_utc"),
        "exit_code": exit_code,
        "gpu_count": 0,
        "postflight_argv": postflight_argv,
        "seed": "not_applicable",
        "started_at_utc": start.get("started_at_utc"),
    }
    summary: dict[str, Any] = {
        "artifact": {
            **inventory_summary,
            "contains_raw_candidate_or_ocr_copy": False,
            "output_root": str(output.resolve()),
            "staging_root": str(
                (output.parent / f".{output.name}.incomplete").resolve()
            ),
            "staging_root_present_after_success": False if valid else None,
        },
        "bindings": bindings,
        "execution": execution,
        "failure": (
            _safe_failure(validation_error, stage=failure_stage)
            if validation_error
            else None
        ),
        "negative_operations": {
            "closed_loop_episode_count": 0,
            "hugging_face_mutation_count": 0,
            "matched_nll_count": 0,
            "model_or_policy_inference_count": 0,
            "restoration_label_count": 0,
            "training_count": 0,
        },
        "postflight": postflight,
        "postflight_accepted_for_formal_result": valid,
        "protocol_id": PROTOCOL_ID,
        "publication": {
            "hf_mutation_count": 0,
            "intended_private_hf_repo": intended_hf_repo,
            "intended_tag": intended_hf_tag,
            "status": (
                PENDING_PUBLICATION_STATUS
                if valid
                else INVALID_PUBLICATION_STATUS
            ),
        },
        "schema_version": SCHEMA_VERSION,
        "scientific_eligibility": valid,
        "status": VALID_RESULT_STATUS if valid else INVALID_RESULT_STATUS,
    }
    _write_result(destination, summary)
    return summary


__all__ = [
    "EXPECTED_FORMAL_FILE_COUNT",
    "INVALID_RESULT_STATUS",
    "PENDING_PUBLICATION_STATUS",
    "PROTOCOL_ID",
    "VALID_RESULT_STATUS",
    "record_processor_freeze_v2_result",
]
