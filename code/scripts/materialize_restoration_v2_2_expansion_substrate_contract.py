"""Materialize the source-only 192-state expansion-substrate contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Sequence

from causalcache.restoration_v2_2_expansion_substrate_contract import (
    CANONICAL_COMPLETION_PATH,
    CANONICAL_CONFIG_PATH,
    build_contract,
    load_json_object,
    pretty_json_bytes,
    sha256_bytes,
    validate_contract_data,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument(
        "--completion-manifest",
        type=Path,
        default=Path(CANONICAL_COMPLETION_PATH),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(CANONICAL_CONFIG_PATH),
    )
    return parser.parse_args(argv)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True
    ).strip()


def verify_clean_pushed_main(root: Path, revision: str) -> None:
    if _git(root, "symbolic-ref", "--short", "HEAD") != "main":
        raise ValueError("contract materialization requires symbolic branch main")
    if _git(root, "rev-parse", "HEAD") != revision:
        raise ValueError("source Git commit does not equal HEAD")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("contract materialization requires a clean checkout")
    if _git(root, "rev-parse", "refs/remotes/origin/main") != revision:
        raise ValueError("source Git commit does not equal origin/main")
    advertised = _git(
        root, "ls-remote", "--exit-code", "origin", "refs/heads/main"
    ).split()
    if len(advertised) != 2 or advertised[0] != revision:
        raise ValueError("source Git commit is not pushed canonical main")


def _canonical_path(root: Path, supplied: Path, relative: str) -> Path:
    expected = (root / relative).resolve()
    actual = (
        supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    )
    if actual != expected:
        raise ValueError(f"path must be canonical: {relative}")
    return actual


def _exclusive_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink() or os.path.lexists(path):
        raise FileExistsError(f"contract output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    verify_clean_pushed_main(root, args.source_git_commit)
    completion_path = _canonical_path(
        root, args.completion_manifest, CANONICAL_COMPLETION_PATH
    )
    output_path = _canonical_path(root, args.output, CANONICAL_CONFIG_PATH)
    if completion_path.is_symlink() or not completion_path.is_file():
        raise ValueError("canonical derived completion manifest is missing")
    completion_payload, completion = load_json_object(completion_path)
    contract = build_contract(
        repository_root=root,
        source_git_commit=args.source_git_commit,
        completion_manifest=completion,
        completion_manifest_sha256=sha256_bytes(completion_payload),
    )
    validate_contract_data(
        contract,
        repository_root=root,
        completion_manifest=completion,
        completion_manifest_sha256=sha256_bytes(completion_payload),
        require_git_blobs=True,
    )
    payload = pretty_json_bytes(contract)
    _exclusive_write(output_path, payload)
    print(
        json.dumps(
            {
                "status": "MATERIALIZED_EXPANSION_SUBSTRATE_SOURCE_ONLY_CONTRACT",
                "config_path": CANONICAL_CONFIG_PATH,
                "config_sha256": sha256_bytes(payload),
                "trajectory_count": 64,
                "state_count": 192,
                "generation_call_count": 384,
                "teacher_forward_count": 576,
                "kl_measurement_count": 384,
                "policy_or_gpu_execution_authorized": False,
                "confirm_access_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
