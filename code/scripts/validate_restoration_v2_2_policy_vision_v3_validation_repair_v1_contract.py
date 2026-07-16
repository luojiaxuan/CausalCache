"""Validate the source-only policy-vision v3 validation-repair contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract import (
    validate_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--validation-source-git-commit", required=True)
    parser.add_argument("--attempt-ledger", type=Path)
    parser.add_argument("--completion-seal", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = validate_contract(
        args.contract.resolve(),
        repository_root=args.repository_root.resolve(),
        validation_source_git_commit=args.validation_source_git_commit,
        require_clean_pushed_main=True,
        require_outputs_absent=True,
        ledger_path=(
            None if args.attempt_ledger is None else args.attempt_ledger.resolve()
        ),
        completion_seal_path=(
            None
            if args.completion_seal is None
            else args.completion_seal.resolve()
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
