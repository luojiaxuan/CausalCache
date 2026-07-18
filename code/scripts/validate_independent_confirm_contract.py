#!/usr/bin/env python3
"""Validate the frozen independent-confirm Source-A without data or model access."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from causalcache.independent_confirm_contract import (
    RUNNER_FREEZE_PATH,
    IndependentConfirmContract,
    build_source_validation,
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_clean_pushed_main(
    root: Path, contract: IndependentConfirmContract
) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    remote = _git(root, "rev-parse", "origin/main")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    origin = _git(root, "remote", "get-url", "origin")
    live = _git(root, "ls-remote", "--exit-code", "origin", "refs/heads/main")
    live_rows = [line.split() for line in live.splitlines() if line.strip()]
    parent_count = int(_git(root, "rev-list", "--parents", "-n", "1", "HEAD").count(" "))
    if (
        branch != "main"
        or head != remote
        or status
        or origin != contract.source["origin_url"]
        or len(live_rows) != 1
        or live_rows[0] != [head, "refs/heads/main"]
        or parent_count != 1
    ):
        raise ValueError("formal Source-A validation requires clean pushed main")
    return {
        "branch": branch,
        "head": head,
        "origin_main": remote,
        "origin_url": origin,
        "parent_count": parent_count,
        "clean": True,
    }


def validate(
    *,
    repository_root: str | Path,
    contract_path: str | Path,
    require_clean_pushed_main: bool,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    contract = IndependentConfirmContract.load(
        contract_path,
        repository_root=root,
        require_runner_absent=True,
    )
    result = dict(build_source_validation(contract))
    if require_clean_pushed_main:
        result["git_identity"] = validate_clean_pushed_main(root, contract)
        result["formal_source_a_validated"] = True
        result["network_call_count"] = 1
        result["network_scope"] = "git_ls_remote_origin_main_only"
    else:
        result["status"] = "DEVELOPMENT_ONLY_INDEPENDENT_CONFIRM_SOURCE_CHECK_V1"
        result["git_identity"] = None
        result["formal_source_a_validated"] = False
        result["network_scope"] = "none"
    result["runner_freeze_path"] = RUNNER_FREEZE_PATH
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--require-clean-pushed-main", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(
        json.dumps(
            validate(
                repository_root=args.repository_root,
                contract_path=args.contract,
                require_clean_pushed_main=args.require_clean_pushed_main,
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
