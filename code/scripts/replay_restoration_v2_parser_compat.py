"""Replay the format-only parser adapter over one immutable raw trace archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import causalcache.policy.gui_owl_v2 as gui_owl_v2_module
import causalcache.policy.gui_owl_v2_compat as gui_owl_v2_compat_module
import causalcache.restoration_v2_parser_replay as parser_replay_module
from causalcache.restoration_v2_parser_replay import (
    PARSER_REPLAY_PROTOCOL_ID,
    audit_restoration_v2_parser_compatibility_archive,
)


CANONICAL_REMOTE = "https://github.com/luojiaxuan/CausalCache.git"
SOURCE_ARTIFACT_MANIFEST_PATH = (
    "data/results/restoration_v2_substrate_screening/artifact_manifest.json"
)
SOURCE_ARTIFACT_MANIFEST_SHA256 = (
    "ce817c5047484d1f9cecb6b60423e318a6509f4480477c5a1504cff58bc456f2"
)
GOLDEN_CONTRACT_PATH = (
    "data/manifests/restoration_v2_parser_compatibility_golden.json"
)
SOURCE_HF_IMMUTABLE_REVISION = "c073e143b935a79befd8ab1fd7123796792efad8"

_EXECUTED_MODULES: tuple[tuple[ModuleType, str], ...] = (
    (gui_owl_v2_module, "code/causalcache/policy/gui_owl_v2.py"),
    (gui_owl_v2_compat_module, "code/causalcache/policy/gui_owl_v2_compat.py"),
    (parser_replay_module, "code/causalcache/restoration_v2_parser_replay.py"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(repository_root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository_root), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_blob(repository_root: Path, git_commit: str, relative_path: str) -> bytes:
    result = subprocess.run(
        (
            "git",
            "-C",
            str(repository_root),
            "show",
            f"{git_commit}:{relative_path}",
        ),
        check=True,
        capture_output=True,
    )
    return result.stdout


def _validate_clean_pushed_main(
    repository_root: Path,
    *,
    expected_git_commit: str,
) -> dict[str, str]:
    head = _git(repository_root, "rev-parse", "HEAD")
    origin_main = _git(repository_root, "rev-parse", "origin/main")
    branch = _git(repository_root, "rev-parse", "--abbrev-ref", "HEAD")
    remote = _git(repository_root, "remote", "get-url", "origin")
    status = _git(repository_root, "status", "--porcelain")
    remote_main_record = _git(
        repository_root,
        "ls-remote",
        "--exit-code",
        "origin",
        "refs/heads/main",
    )
    expected_remote_record = f"{expected_git_commit}\trefs/heads/main"
    if (
        head != expected_git_commit
        or origin_main != expected_git_commit
        or branch != "main"
        or remote != CANONICAL_REMOTE
        or status
        or remote_main_record != expected_remote_record
    ):
        raise ValueError("parser replay requires clean pushed canonical remote main")
    return {
        "git_commit": head,
        "origin_main_git_commit": origin_main,
        "remote_main_git_commit": expected_git_commit,
        "branch": branch,
        "remote": remote,
    }


def _source_identity(
    repository_root: Path,
    relative_path: str,
    *,
    git_commit: str,
) -> dict[str, Any]:
    path = (repository_root / relative_path).resolve()
    try:
        path.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError("source path escapes repository root") from error
    payload = path.read_bytes()
    committed_payload = _git_blob(repository_root, git_commit, relative_path)
    if payload != committed_payload:
        raise ValueError(f"working source differs from committed Git blob: {relative_path}")
    return {
        "path": relative_path,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validate_executed_source_inventory(
    repository_root: Path,
    *,
    git_commit: str,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for module, relative_path in _EXECUTED_MODULES:
        module_file = getattr(module, "__file__", None)
        expected_path = (repository_root / relative_path).resolve()
        if module_file is None or Path(module_file).resolve() != expected_path:
            raise ValueError(f"executed module path is not canonical: {relative_path}")
        sources.append(
            _source_identity(
                repository_root,
                relative_path,
                git_commit=git_commit,
            )
        )
    cli_relative_path = "code/scripts/replay_restoration_v2_parser_compat.py"
    if Path(__file__).resolve() != (repository_root / cli_relative_path).resolve():
        raise ValueError("executed CLI path is not canonical")
    sources.append(
        _source_identity(
            repository_root,
            cli_relative_path,
            git_commit=git_commit,
        )
    )
    return sources


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def _strict_json_object(payload: bytes, *, name: str) -> dict[str, Any]:
    value = json.loads(
        payload,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _load_frozen_source_contract(
    repository_root: Path,
    *,
    git_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    artifact_identity = _source_identity(
        repository_root,
        SOURCE_ARTIFACT_MANIFEST_PATH,
        git_commit=git_commit,
    )
    if artifact_identity["sha256"] != SOURCE_ARTIFACT_MANIFEST_SHA256:
        raise ValueError("source artifact manifest differs from frozen SHA256")
    artifact_payload = (repository_root / SOURCE_ARTIFACT_MANIFEST_PATH).read_bytes()
    artifact = _strict_json_object(
        artifact_payload,
        name=SOURCE_ARTIFACT_MANIFEST_PATH,
    )
    packaging = artifact.get("packaging")
    run = artifact.get("run")
    huggingface = artifact.get("huggingface")
    if (
        artifact.get("schema_version") != "1.0.0"
        or artifact.get("protocol_id") != "causalcache_restoration_v2"
        or artifact.get("artifact_kind")
        != "raw_restoration_v2_substrate_screening_trace_shard"
        or not isinstance(packaging, Mapping)
        or not isinstance(run, Mapping)
        or not isinstance(huggingface, Mapping)
        or run.get("outcome") != "NO_GO_V2_SUBSTRATE"
    ):
        raise ValueError("source artifact manifest identity drifted")

    golden_identity = _source_identity(
        repository_root,
        GOLDEN_CONTRACT_PATH,
        git_commit=git_commit,
    )
    golden_payload = (repository_root / GOLDEN_CONTRACT_PATH).read_bytes()
    golden = _strict_json_object(golden_payload, name=GOLDEN_CONTRACT_PATH)
    source_artifact = golden.get("source_artifact")
    expected_source_artifact = {
        "archive_filename": packaging.get("archive_filename"),
        "archive_sha256": packaging.get("archive_sha256"),
        "artifact_manifest_path": SOURCE_ARTIFACT_MANIFEST_PATH,
        "artifact_manifest_sha256": SOURCE_ARTIFACT_MANIFEST_SHA256,
        "hf_immutable_revision": SOURCE_HF_IMMUTABLE_REVISION,
        "hf_path_prefix": huggingface.get("path_prefix"),
        "hf_repo_id": huggingface.get("repo_id"),
        "hf_repo_type": huggingface.get("repo_type"),
        "hf_tag": huggingface.get("target_tag"),
        "run_contract_sha256": run.get("run_contract_sha256"),
        "run_prefix": str(packaging.get("archive_filename", "")).removesuffix(
            ".tar.gz"
        ),
        "source_run_git_commit": run.get("git_commit"),
    }
    if (
        golden.get("schema_version") != "1.0.0"
        or golden.get("protocol_id") != PARSER_REPLAY_PROTOCOL_ID
        or source_artifact != expected_source_artifact
        or not isinstance(golden.get("expected_metrics"), Mapping)
        or not isinstance(golden.get("expected_inventory"), Mapping)
        or not isinstance(golden.get("expected_rejections"), list)
        or golden.get("expected_outcome") != "NO_GO_ADAPTER_ONLY"
    ):
        raise ValueError("parser replay golden contract drifted")
    return expected_source_artifact, golden, [artifact_identity, golden_identity]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_golden_result(result: Mapping[str, Any], golden: Mapping[str, Any]) -> None:
    if (
        result.get("outcome") != golden.get("expected_outcome")
        or result.get("metrics") != golden.get("expected_metrics")
        or result.get("inventory") != golden.get("expected_inventory")
    ):
        raise ValueError("parser replay aggregate differs from frozen golden result")
    records = result.get("records")
    if not isinstance(records, list) or _canonical_sha256(records) != golden.get(
        "expected_records_canonical_sha256"
    ):
        raise ValueError("parser replay per-state classifications drifted")
    rejections = [
        {
            "state_index": record.get("state_index"),
            "state_id": record.get("state_id"),
            "raw_output_sha256": record.get("raw_output_sha256"),
            "rejection_code": record.get("rejection_code"),
        }
        for record in records
        if isinstance(record, Mapping) and record.get("rejection_code") is not None
    ]
    if rejections != golden.get("expected_rejections"):
        raise ValueError("parser replay rejection identities drifted")


def _write_json_exclusive(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--audit-git-commit", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(raw_argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    repository_root = Path(args.repository_root).resolve()
    archive = Path(args.archive).resolve()
    started_at_utc = _utc_now()

    pre_git_identity = _validate_clean_pushed_main(
        repository_root,
        expected_git_commit=args.audit_git_commit,
    )
    pre_executed_sources = _validate_executed_source_inventory(
        repository_root,
        git_commit=args.audit_git_commit,
    )
    source_artifact, golden, pre_contract_sources = _load_frozen_source_contract(
        repository_root,
        git_commit=args.audit_git_commit,
    )
    if archive.name != source_artifact["archive_filename"]:
        raise ValueError("archive filename differs from frozen source artifact")

    result = audit_restoration_v2_parser_compatibility_archive(
        archive_path=archive,
        expected_archive_sha256=source_artifact["archive_sha256"],
        run_prefix=source_artifact["run_prefix"],
        expected_run_contract_sha256=source_artifact["run_contract_sha256"],
        expected_source_run_git_commit=source_artifact["source_run_git_commit"],
    )
    _validate_golden_result(result, golden)

    post_git_identity = _validate_clean_pushed_main(
        repository_root,
        expected_git_commit=args.audit_git_commit,
    )
    post_executed_sources = _validate_executed_source_inventory(
        repository_root,
        git_commit=args.audit_git_commit,
    )
    post_source_artifact, post_golden, post_contract_sources = (
        _load_frozen_source_contract(
            repository_root,
            git_commit=args.audit_git_commit,
        )
    )
    if (
        post_git_identity != pre_git_identity
        or post_executed_sources != pre_executed_sources
        or post_contract_sources != pre_contract_sources
        or post_source_artifact != source_artifact
        or post_golden != golden
    ):
        raise ValueError("repository or audit contract mutated during parser replay")

    result["immutable_source"] = source_artifact
    result["golden_contract"] = {
        "path": GOLDEN_CONTRACT_PATH,
        "sha256": pre_contract_sources[1]["sha256"],
        "records_canonical_sha256": golden[
            "expected_records_canonical_sha256"
        ],
        "validation_passed": True,
    }
    result["audit_run"] = {
        "argv": ["python3", "-m", "scripts.replay_restoration_v2_parser_compat", *raw_argv],
        "started_at_utc": started_at_utc,
        "ended_at_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "git_pre_and_post_identity": pre_git_identity,
        "pre_and_post_validation_matched": True,
        "executed_sources": pre_executed_sources,
        "contract_sources": pre_contract_sources,
    }
    _write_json_exclusive(output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
