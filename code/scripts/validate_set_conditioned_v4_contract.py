#!/usr/bin/env python3
"""Validate frozen-base residual v4 Source-A or Execution-B identity."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causalcache.set_conditioned_v4_contract import (
    CANONICAL_CONFIG_PATH,
    validate_execution_b,
    validate_source_a,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("source-a")
    source.add_argument("--repository-root", type=Path, default=Path("."))
    source.add_argument("--contract", type=Path, default=Path(CANONICAL_CONFIG_PATH))
    execution = subparsers.add_parser("execution-b")
    execution.add_argument("--repository-root", type=Path, default=Path("."))
    execution.add_argument("--contract", type=Path, default=Path(CANONICAL_CONFIG_PATH))
    execution.add_argument("--source-a-git-commit", required=True)
    execution.add_argument("--execution-b-git-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "source-a":
        result = validate_source_a(
            args.contract,
            repository_root=args.repository_root,
        )
    else:
        result = validate_execution_b(
            args.contract,
            repository_root=args.repository_root,
            source_a_git_commit=args.source_a_git_commit,
            execution_b_git_commit=args.execution_b_git_commit,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
