#!/usr/bin/env python3
"""Build the structural-only long-horizon development/reserve selection."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

from causalcache.long_horizon_contract import (
    CANONICAL_CONFIG_PATH,
    LongHorizonContract,
    SELECTION_MANIFEST_FREEZE_PATH,
    load_strict_json,
    sha256_file,
)
from causalcache.long_horizon_selection import (
    build_long_horizon_selection_manifest,
    pretty_json_bytes,
    reconstruct_long_horizon_pool,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=Path(CANONICAL_CONFIG_PATH))
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _verify_clean_pushed_source_a(root: Path, revision: str) -> None:
    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True
        ).strip()

    branch = git("symbolic-ref", "--short", "HEAD")
    if git("rev-parse", "HEAD") != revision:
        raise ValueError("--git-revision differs from Source-A checkout HEAD")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("selection construction requires a clean Source-A checkout")
    if git("rev-parse", f"refs/remotes/origin/{branch}") != revision:
        raise ValueError("selection construction requires the pushed current branch")
    advertised = git(
        "ls-remote", "--exit-code", "origin", f"refs/heads/{branch}"
    ).split()
    if len(advertised) != 2 or advertised[0] != revision:
        raise ValueError("canonical remote does not advertise Source-A HEAD")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if re.fullmatch(r"[0-9a-f]{40}", args.git_revision) is None:
        raise ValueError("--git-revision must be a full lowercase Git commit")
    if args.output_manifest.exists():
        raise FileExistsError("selection output must not already exist")

    root = args.repository_root.resolve()
    expected_output = (root / SELECTION_MANIFEST_FREEZE_PATH).resolve()
    if args.output_manifest.resolve() != expected_output:
        raise ValueError(
            "selection output must use the canonical Git freeze path"
        )
    _verify_clean_pushed_source_a(root, args.git_revision)
    contract = LongHorizonContract.load(
        args.contract,
        repository_root=root,
        require_runner_absent=True,
    )
    parent_path = root / contract.source_pool["parent_protocol_config"]["path"]
    source_manifest_path = root / contract.source_pool["source_file_manifest"]["path"]
    parent = load_strict_json(parent_path, label="parent protocol config")
    source_manifest = load_strict_json(
        source_manifest_path,
        label="source file manifest",
    )
    pool = reconstruct_long_horizon_pool(
        source_root=args.source_root.resolve(),
        parent_config=parent,
        source_file_manifest=source_manifest,
        contract=contract,
    )
    module_path = root / "code/causalcache/long_horizon_selection.py"
    cli_path = Path(__file__).resolve()
    validator_path = root / "code/scripts/validate_long_horizon_contract.py"
    manifest = build_long_horizon_selection_manifest(
        pool=pool,
        contract=contract,
        parent_config_sha256=sha256_file(parent_path),
        source_file_manifest_sha256=sha256_file(source_manifest_path),
        generator={
            "git_revision": args.git_revision,
            "module_path": "code/causalcache/long_horizon_selection.py",
            "module_sha256": sha256_file(module_path),
            "cli_path": "code/scripts/build_long_horizon_selection.py",
            "cli_sha256": sha256_file(cli_path),
            "contract_validator_path": "code/scripts/validate_long_horizon_contract.py",
            "contract_validator_sha256": sha256_file(validator_path),
        },
    )
    payload = pretty_json_bytes(manifest)
    _exclusive_write(args.output_manifest, payload)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_manifest": str(args.output_manifest),
                "output_manifest_sha256": sha256_file(args.output_manifest),
                "eligible_trajectory_count": manifest["reconstruction"][
                    "eligible_trajectory_count"
                ],
                "development_trajectory_count": manifest["splits"]["development"][
                    "trajectory_count"
                ],
                "unopened_reserve_trajectory_count": manifest["splits"][
                    "unopened_reserve"
                ]["trajectory_count"],
                "development_state_count": manifest["splits"]["development"][
                    "state_count"
                ],
                "historical_intersection_count": manifest[
                    "historical_role_firewall"
                ]["intersection_count"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
