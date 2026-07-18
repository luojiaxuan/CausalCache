#!/usr/bin/env python3
"""Validate, freeze, execute, or replay confirm-20 failure decomposition."""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.independent_confirm_failure_decomposition_contract import (
    CANONICAL_CONFIG_PATH,
    load_frozen_failure_decomposition_contract,
    materialize_runner_freeze,
    validate_execution_b_source,
    validate_source_a,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("validate-source")
    source.add_argument("--repository-root", type=Path, required=True)
    source.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
    source.add_argument("--source-a-git-commit")
    freeze = commands.add_parser("materialize-runner-freeze")
    freeze.add_argument("--repository-root", type=Path, required=True)
    freeze.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
    freeze.add_argument("--source-a-git-commit", required=True)
    for name in ("run", "validate"):
        child = commands.add_parser(name)
        child.add_argument("--repository-root", type=Path, required=True)
        child.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
        child.add_argument("--execution-b-git-commit", required=True)
        child.add_argument("--fresh-download-parent", type=Path, required=True)
        child.add_argument("--hf-token-file", type=Path, required=True)
    return parser


def _secure_token(path: Path) -> str:
    if not path.is_absolute():
        raise ValueError("HF token path must be absolute")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) not in {0o400, 0o600}
        ):
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
    payload = b"".join(chunks)
    if fingerprint(before) != fingerprint(after) or len(payload) != before.st_size:
        raise ValueError("HF token changed while being read")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("HF token must be ASCII") from error
    token = text[:-1] if text.endswith("\n") else text
    if (
        not token
        or text not in {token, token + "\n"}
        or any(character.isspace() for character in token)
    ):
        raise ValueError("HF token must contain exactly one nonempty token")
    return token


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return value


def _delegate(args: argparse.Namespace, contract: Any) -> Mapping[str, Any]:
    source = validate_execution_b_source(
        contract,
        expected_execution_b_git_commit=args.execution_b_git_commit,
    )
    token = _secure_token(args.hf_token_file)
    from causalcache.independent_confirm_failure_decomposition_runner import (
        execute_failure_decomposition,
    )
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)

    def authenticated_download(**kwargs: Any) -> str:
        return hf_hub_download(**kwargs, token=token)

    result = execute_failure_decomposition(
        mode=args.command,
        contract=contract,
        api=api,
        download_fn=authenticated_download,
        operation_factory=CommitOperationAdd,
        fresh_download_parent=args.fresh_download_parent,
        source_execution=source,
    )
    return {**dict(result), "execution_b_source": dict(source)}


def _run(args: argparse.Namespace) -> Mapping[str, Any]:
    contract = load_frozen_failure_decomposition_contract(
        args.contract,
        repository_root=args.repository_root,
    )
    if args.command == "validate-source":
        return validate_source_a(
            contract,
            expected_source_a_git_commit=args.source_a_git_commit,
        )
    if args.command == "materialize-runner-freeze":
        return materialize_runner_freeze(
            contract,
            expected_source_a_git_commit=args.source_a_git_commit,
        )
    if args.command in {"run", "validate"}:
        return _delegate(args, contract)
    raise AssertionError(f"unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    result = _run(_parser().parse_args(argv))
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
