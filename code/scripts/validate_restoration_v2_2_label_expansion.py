"""Validate a frozen restoration-v2.2 label-expansion split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from causalcache.restoration_v2_2_label_expansion import (
    CONFIG_PATH,
    PARENT_PATH,
    load_json_object,
    sha256_bytes,
    summary,
    validate_expansion_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def _git_bytes(repo_root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=repo_root)


def _verify_generator(repo_root: Path, generator: dict) -> None:
    revision = str(generator["git_revision"])
    subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "origin/main"],
        cwd=repo_root,
        check=True,
    )
    source_paths = {
        "module": "code/causalcache/restoration_v2_2_label_expansion.py",
        "cli": "code/scripts/materialize_restoration_v2_2_label_expansion.py",
        "validator": "code/scripts/validate_restoration_v2_2_label_expansion.py",
    }
    for prefix, relative_path in source_paths.items():
        if generator[f"{prefix}_path"] != relative_path:
            raise ValueError(f"generator {prefix} path mismatch")
        current = (repo_root / relative_path).read_bytes()
        current_sha = hashlib.sha256(current).hexdigest()
        if generator[f"{prefix}_sha256"] != current_sha:
            raise ValueError(f"generator {prefix} SHA256 mismatch")
        committed = _git_bytes(repo_root, "show", f"{revision}:{relative_path}")
        if hashlib.sha256(committed).hexdigest() != current_sha:
            raise ValueError(f"generator {prefix} differs from its Git revision")


def _verify_committed_input(
    repo_root: Path,
    *,
    revision: str,
    relative_path: str,
    expected_sha256: str,
) -> None:
    current = (repo_root / relative_path).read_bytes()
    if sha256_bytes(current) != expected_sha256:
        raise ValueError(f"current input SHA256 mismatch: {relative_path}")
    committed = _git_bytes(repo_root, "show", f"{revision}:{relative_path}")
    if sha256_bytes(committed) != expected_sha256:
        raise ValueError(f"committed input SHA256 mismatch: {relative_path}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_payload, config = load_json_object(args.config)
    parent_payload, parent = load_json_object(args.parent_selection)
    manifest_payload, manifest = load_json_object(args.manifest)
    config_sha = sha256_bytes(config_payload)
    parent_sha = sha256_bytes(parent_payload)
    validate_expansion_manifest(
        manifest,
        config=config,
        config_sha256=config_sha,
        parent=parent,
        parent_sha256=parent_sha,
    )
    generator = manifest["generator"]
    _verify_generator(repo_root, generator)
    revision = str(generator["git_revision"])
    _verify_committed_input(
        repo_root,
        revision=revision,
        relative_path=CONFIG_PATH,
        expected_sha256=config_sha,
    )
    _verify_committed_input(
        repo_root,
        revision=revision,
        relative_path=PARENT_PATH,
        expected_sha256=parent_sha,
    )
    print(
        json.dumps(
            summary(manifest, manifest_sha256=sha256_bytes(manifest_payload)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
