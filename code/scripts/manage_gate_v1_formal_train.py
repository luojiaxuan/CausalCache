#!/usr/bin/env python3
"""Validate, freeze, execute, or replay formal-58 gate training."""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.gate_v1_formal_train_contract import (
    CANONICAL_CONFIG_PATH,
    load_frozen_formal_train_contract,
    validate_formal_train_source_only_contract,
)
from causalcache.gate_v1_formal_train_runner import (
    capture_docker_inspect_receipt,
    execute_formal_train,
    materialize_runner_freeze,
    validate_execution_b_source,
    validate_source_a,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def source_parser(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--repository-root", type=Path, required=True)
        child.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
        child.add_argument("--source-a-git-commit")
        return child

    source_parser("validate-source")
    source_parser("materialize-runner-freeze")

    capture = subparsers.add_parser("capture-runtime")
    capture.add_argument("--repository-root", type=Path, required=True)
    capture.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
    capture.add_argument("--execution-b-git-commit", required=True)
    capture.add_argument("--container-name", required=True)
    capture.add_argument("--host-data-root", type=Path, required=True)

    for name in ("run", "validate"):
        child = subparsers.add_parser(name)
        child.add_argument("--repository-root", type=Path, required=True)
        child.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
        child.add_argument("--execution-b-git-commit", required=True)
        child.add_argument("--hf-token-file", type=Path, required=True)
        child.add_argument("--data-root", type=Path, required=True)
        child.add_argument("--fresh-download-parent", type=Path, required=True)
    return parser


def _secure_token(path: Path) -> str:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) not in {
            0o400,
            0o600,
        }:
            raise ValueError("HF token must be a mode-0400/0600 regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 4096):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError("HF token is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    fingerprint = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if fingerprint(before) != fingerprint(after):
        raise ValueError("HF token changed while being read")
    payload = b"".join(chunks)
    try:
        token = payload.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("HF token is not ASCII") from error
    if not token or any(character.isspace() for character in token):
        raise ValueError("HF token is empty or malformed")
    return token


def _run(args: argparse.Namespace) -> Mapping[str, Any]:
    contract = load_frozen_formal_train_contract(
        args.contract,
        repository_root=args.repository_root,
    )
    if args.command == "validate-source":
        validation = validate_formal_train_source_only_contract(
            args.contract,
            repository_root=args.repository_root,
        )
        return {
            **validation,
            "runner": validate_source_a(
                contract,
                expected_source_a_git_commit=args.source_a_git_commit,
            ),
        }
    if args.command == "materialize-runner-freeze":
        validate_formal_train_source_only_contract(
            args.contract,
            repository_root=args.repository_root,
        )
        return materialize_runner_freeze(
            contract,
            expected_source_a_git_commit=args.source_a_git_commit,
        )
    if args.command == "capture-runtime":
        source = validate_execution_b_source(
            contract,
            expected_execution_b_git_commit=args.execution_b_git_commit,
        )
        return capture_docker_inspect_receipt(
            contract,
            source=source,
            container_name=args.container_name,
            host_data_root=args.host_data_root,
        )

    token = _secure_token(args.hf_token_file)
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)

    def authenticated_download(**kwargs: Any) -> str:
        return hf_hub_download(**kwargs, token=token)

    return execute_formal_train(
        mode=args.command,
        contract=contract,
        api=api,
        download_fn=authenticated_download,
        operation_factory=CommitOperationAdd,
        expected_execution_b_git_commit=args.execution_b_git_commit,
        data_root=args.data_root,
        fresh_download_parent=args.fresh_download_parent,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
