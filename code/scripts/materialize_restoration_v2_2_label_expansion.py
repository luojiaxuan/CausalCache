"""Materialize the frozen policy-blind restoration label-expansion split."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from causalcache.restoration_v2_2_label_expansion import (
    CONFIG_PATH,
    PARENT_PATH,
    build_expansion_manifest,
    load_json_object,
    pretty_json_bytes,
    sha256_bytes,
    summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-selection", type=Path, required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repo_root,
        text=True,
    ).strip()


def _verify_clean_pushed_main(repo_root: Path, expected_revision: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
        raise ValueError("--git-revision must be a full 40-character commit")
    if _git(repo_root, "symbolic-ref", "--short", "HEAD") != "main":
        raise ValueError("label-expansion materialization requires branch main")
    if _git(repo_root, "rev-parse", "HEAD") != expected_revision:
        raise ValueError("--git-revision does not match checkout HEAD")
    if _git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("label-expansion materialization requires a clean checkout")
    if _git(repo_root, "rev-parse", "refs/remotes/origin/main") != expected_revision:
        raise ValueError("label-expansion materialization requires HEAD == origin/main")
    advertised = _git(repo_root, "ls-remote", "--exit-code", "origin", "refs/heads/main")
    fields = advertised.split()
    if len(fields) != 2 or fields[0] != expected_revision:
        raise ValueError("label-expansion materialization requires pushed canonical main")


def _require_canonical_path(actual: Path, repo_root: Path, expected: str) -> None:
    if actual.resolve() != (repo_root / expected).resolve():
        raise ValueError(f"input must use canonical path {expected}")


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    _require_canonical_path(args.config, repo_root, CONFIG_PATH)
    _require_canonical_path(args.parent_selection, repo_root, PARENT_PATH)
    if args.output.exists():
        raise FileExistsError("label-expansion output already exists")
    _verify_clean_pushed_main(repo_root, args.git_revision)

    config_payload, config = load_json_object(args.config)
    parent_payload, parent = load_json_object(args.parent_selection)
    module_path = repo_root / "code/causalcache/restoration_v2_2_label_expansion.py"
    validator_path = repo_root / (
        "code/scripts/validate_restoration_v2_2_label_expansion.py"
    )
    generator = {
        "git_revision": args.git_revision,
        "module_path": "code/causalcache/restoration_v2_2_label_expansion.py",
        "module_sha256": sha256_bytes(module_path.read_bytes()),
        "cli_path": "code/scripts/materialize_restoration_v2_2_label_expansion.py",
        "cli_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "validator_path": "code/scripts/validate_restoration_v2_2_label_expansion.py",
        "validator_sha256": sha256_bytes(validator_path.read_bytes()),
    }
    manifest = build_expansion_manifest(
        config=config,
        config_sha256=sha256_bytes(config_payload),
        parent=parent,
        parent_sha256=sha256_bytes(parent_payload),
        generator=generator,
    )
    payload = pretty_json_bytes(manifest)
    _exclusive_write(args.output, payload)
    print(json.dumps(summary(manifest, manifest_sha256=sha256_bytes(payload)), indent=2))


if __name__ == "__main__":
    main()
