#!/usr/bin/env python3
"""Wait for one successful processor-v2 supervisor, then run its recorder."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


_RECORDER_RELATIVE_PATH = Path(
    "code/scripts/record_set_utility_processor_freeze_v2_result.py"
)
_SUPERVISOR_KEYS = frozenset(
    {
        "formal_exit_code",
        "postflight_exit_code",
        "supervisor_ended_at_utc",
        "supervisor_pid",
    }
)
_EVIDENCE_FILENAMES = {
    "run_argv_evidence": "argv.txt",
    "start_evidence": "start.json",
    "outer_log": "outer.log",
    "exit_code_evidence": "exit_code",
    "end_evidence": "end.json",
    "postflight_argv_evidence": "postflight-argv.txt",
    "postflight_evidence": "postflight.json",
    "postflight_exit_code_evidence": "postflight-exit-code",
    "supervisor_evidence": "supervisor.json",
}
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorder-python-executable", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--producer-repository-root", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-revision", required=True)
    parser.add_argument("--expected-recorder-git-revision", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--run-argv-evidence", type=Path, required=True)
    parser.add_argument("--start-evidence", type=Path, required=True)
    parser.add_argument("--outer-log", type=Path, required=True)
    parser.add_argument("--exit-code-evidence", type=Path, required=True)
    parser.add_argument("--end-evidence", type=Path, required=True)
    parser.add_argument("--postflight-argv-evidence", type=Path, required=True)
    parser.add_argument("--postflight-evidence", type=Path, required=True)
    parser.add_argument(
        "--postflight-exit-code-evidence", type=Path, required=True
    )
    parser.add_argument("--supervisor-evidence", type=Path, required=True)
    parser.add_argument("--intended-hf-repo", required=True)
    parser.add_argument("--intended-hf-tag", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    return parser.parse_args(argv)


def _read_regular_file(path: Path, *, label: str) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for watcher evidence reads")
    try:
        path_before = os.stat(path, follow_symlinks=False)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError(f"{label} must be one regular non-symlink file") from error
    try:
        before = os.fstat(descriptor)
        payload_blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            payload_blocks.append(block)
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    identities = {
        (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
        )
        for value in (path_before, before, after, path_after)
    }
    payload = b"".join(payload_blocks)
    if len(identities) != 1 or len(payload) != after.st_size:
        raise RuntimeError(f"{label} changed while the watcher read it")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be one regular non-symlink file")
    return payload


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        result = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(result, dict):
        raise ValueError(f"{label} must contain one JSON object")
    canonical = (
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if payload != canonical:
        raise ValueError(f"{label} is not canonical newline-terminated JSON")
    return result


def _validate_supervisor(payload: bytes) -> None:
    value = _strict_json_object(payload, label="supervisor evidence")
    if set(value) != _SUPERVISOR_KEYS:
        raise ValueError("supervisor evidence fields drifted")
    timestamp = value["supervisor_ended_at_utc"]
    if (
        not isinstance(timestamp, str)
        or _UTC_TIMESTAMP.fullmatch(timestamp) is None
    ):
        raise ValueError("supervisor evidence timestamp format drifted")
    try:
        datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("supervisor evidence timestamp is invalid") from error
    if type(value["supervisor_pid"]) is not int or value["supervisor_pid"] <= 0:
        raise ValueError("supervisor evidence pid must be positive")
    if type(value["formal_exit_code"]) is not int:
        raise ValueError("supervisor formal exit code must be an integer")
    if type(value["postflight_exit_code"]) is not int:
        raise ValueError("supervisor postflight exit code must be an integer")
    if value["formal_exit_code"] != 0 or value["postflight_exit_code"] != 0:
        raise RuntimeError(
            "formal run and committed postflight must both exit zero before recording"
        )


def _validate_evidence_layout(args: argparse.Namespace) -> Path:
    evidence_paths = {
        name: Path(getattr(args, name)) for name in _EVIDENCE_FILENAMES
    }
    if any(not path.is_absolute() for path in evidence_paths.values()):
        raise ValueError("all watcher evidence paths must be absolute")
    roots = {path.parent for path in evidence_paths.values()}
    if len(roots) != 1:
        raise ValueError("all watcher evidence files must share one evidence root")
    evidence_root = roots.pop()
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise ValueError("watcher evidence root must be one real directory")
    if evidence_root.resolve() != evidence_root.absolute():
        raise ValueError("watcher evidence root must not traverse symlinks")
    for name, expected_filename in _EVIDENCE_FILENAMES.items():
        if evidence_paths[name].name != expected_filename:
            raise ValueError(
                f"{name} must use the frozen filename {expected_filename!r}"
            )
    return evidence_root


def _wait_for_success(args: argparse.Namespace) -> None:
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    _validate_evidence_layout(args)
    destination = Path(args.result_dir)
    staging = destination.parent / f".{destination.name}.incomplete"
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Git result directory appeared before recording")
        if staging.exists() or staging.is_symlink():
            raise FileExistsError(
                "Git result staging directory appeared before recording"
            )
        try:
            supervisor_payload = _read_regular_file(
                Path(args.supervisor_evidence), label="supervisor evidence"
            )
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for supervisor evidence")
            time.sleep(
                min(args.poll_seconds, max(0.0, deadline - time.monotonic()))
            )
            continue
        _validate_supervisor(supervisor_payload)
        formal_exit = _read_regular_file(
            Path(args.exit_code_evidence), label="formal exit-code evidence"
        )
        postflight_exit = _read_regular_file(
            Path(args.postflight_exit_code_evidence),
            label="postflight exit-code evidence",
        )
        if formal_exit != b"0\n" or postflight_exit != b"0\n":
            raise RuntimeError(
                "terminal exit-code files must both contain exactly zero"
            )
        return


def _recorder_command(args: argparse.Namespace) -> list[str]:
    repository_root = Path(args.repository_root).resolve()
    return [
        str(Path(args.recorder_python_executable)),
        str((repository_root / _RECORDER_RELATIVE_PATH).resolve()),
        "--repository-root",
        str(repository_root),
        "--producer-repository-root",
        str(Path(args.producer_repository_root).resolve()),
        "--execution-config",
        str(Path(args.execution_config).resolve()),
        "--output-root",
        str(Path(args.output_root).resolve()),
        "--expected-git-revision",
        args.expected_git_revision,
        "--expected-recorder-git-revision",
        args.expected_recorder_git_revision,
        "--result-dir",
        str(Path(args.result_dir).resolve()),
        "--run-argv-evidence",
        str(Path(args.run_argv_evidence).resolve()),
        "--start-evidence",
        str(Path(args.start_evidence).resolve()),
        "--outer-log",
        str(Path(args.outer_log).resolve()),
        "--exit-code-evidence",
        str(Path(args.exit_code_evidence).resolve()),
        "--end-evidence",
        str(Path(args.end_evidence).resolve()),
        "--postflight-argv-evidence",
        str(Path(args.postflight_argv_evidence).resolve()),
        "--postflight-evidence",
        str(Path(args.postflight_evidence).resolve()),
        "--intended-hf-repo",
        args.intended_hf_repo,
        "--intended-hf-tag",
        args.intended_hf_tag,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _wait_for_success(args)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        _recorder_command(args),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    sys.stdout.buffer.write(completed.stdout)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(completed.stderr)
    sys.stderr.buffer.flush()
    return (
        completed.returncode
        if completed.returncode >= 0
        else 128 - completed.returncode
    )


if __name__ == "__main__":
    sys.exit(main())
