"""Validate a frozen restoration-v2.2 label-expansion exposure ledger."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from causalcache.restoration_v2_2_label_expansion_exposure import (
    CONFIG_PATH,
    EXPANSION_SELECTION_PATH,
    GENERATOR_PATHS,
    PARENT_EXPOSURE_PATH,
    PARENT_PATH,
    PRIOR_OUTPUT_EXPOSURE_PATH,
    SOURCE_MANIFEST_PATH,
    load_json_object,
    sha256_bytes,
    summary,
    validate_exposure_ledger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-selection", type=Path, required=True)
    parser.add_argument("--source-file-manifest", type=Path, required=True)
    parser.add_argument("--parent-exposure", type=Path, required=True)
    parser.add_argument("--prior-output-exposure", type=Path, required=True)
    parser.add_argument("--expansion-selection", type=Path)
    parser.add_argument("--ledger", type=Path, required=True)
    return parser.parse_args()


def _git_bytes(repo_root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=repo_root)


def _require_canonical_path(actual: Path, repo_root: Path, expected: str) -> None:
    if actual.resolve() != (repo_root / expected).resolve():
        raise ValueError(f"input must use canonical path {expected}")


def _optional_expansion_path(
    requested: Path | None, *, repo_root: Path
) -> Path | None:
    canonical = repo_root / EXPANSION_SELECTION_PATH
    if requested is not None:
        _require_canonical_path(requested, repo_root, EXPANSION_SELECTION_PATH)
        if not requested.exists():
            raise FileNotFoundError(requested)
        return requested
    return canonical if canonical.exists() else None


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
    for prefix, relative_path in GENERATOR_PATHS.items():
        if generator[f"{prefix}_path"] != relative_path:
            raise ValueError(f"generator {prefix} path mismatch")
        current = (repo_root / relative_path).read_bytes()
        current_sha = sha256_bytes(current)
        if generator[f"{prefix}_sha256"] != current_sha:
            raise ValueError(f"generator {prefix} SHA256 mismatch")
        committed = _git_bytes(repo_root, "show", f"{revision}:{relative_path}")
        if sha256_bytes(committed) != current_sha:
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
    canonical_inputs = {
        args.config: CONFIG_PATH,
        args.parent_selection: PARENT_PATH,
        args.source_file_manifest: SOURCE_MANIFEST_PATH,
        args.parent_exposure: PARENT_EXPOSURE_PATH,
        args.prior_output_exposure: PRIOR_OUTPUT_EXPOSURE_PATH,
    }
    for path, expected in canonical_inputs.items():
        _require_canonical_path(path, repo_root, expected)
    expansion_path = _optional_expansion_path(
        args.expansion_selection, repo_root=repo_root
    )

    config_payload, config = load_json_object(args.config)
    parent_payload, parent = load_json_object(args.parent_selection)
    source_payload, source = load_json_object(args.source_file_manifest)
    parent_exposure_payload, parent_exposure = load_json_object(args.parent_exposure)
    prior_exposure_payload, prior_exposure = load_json_object(
        args.prior_output_exposure
    )
    expansion_payload: bytes | None = None
    expansion_selection: dict | None = None
    if expansion_path is not None:
        expansion_payload, expansion_selection = load_json_object(expansion_path)
    ledger_payload, ledger = load_json_object(args.ledger)
    validate_exposure_ledger(
        ledger,
        config=config,
        config_sha256=sha256_bytes(config_payload),
        parent=parent,
        parent_sha256=sha256_bytes(parent_payload),
        source_manifest=source,
        source_manifest_sha256=sha256_bytes(source_payload),
        parent_exposure=parent_exposure,
        parent_exposure_sha256=sha256_bytes(parent_exposure_payload),
        prior_output_exposure=prior_exposure,
        prior_output_exposure_sha256=sha256_bytes(prior_exposure_payload),
        expansion_selection=expansion_selection,
        expansion_selection_sha256=(
            sha256_bytes(expansion_payload) if expansion_payload is not None else None
        ),
    )
    generator = ledger["generator"]
    _verify_generator(repo_root, generator)
    revision = str(generator["git_revision"])
    identities = {
        CONFIG_PATH: sha256_bytes(config_payload),
        PARENT_PATH: sha256_bytes(parent_payload),
        SOURCE_MANIFEST_PATH: sha256_bytes(source_payload),
        PARENT_EXPOSURE_PATH: sha256_bytes(parent_exposure_payload),
        PRIOR_OUTPUT_EXPOSURE_PATH: sha256_bytes(prior_exposure_payload),
    }
    if expansion_payload is not None:
        identities[EXPANSION_SELECTION_PATH] = sha256_bytes(expansion_payload)
    for relative_path, expected_sha in identities.items():
        _verify_committed_input(
            repo_root,
            revision=revision,
            relative_path=relative_path,
            expected_sha256=expected_sha,
        )
    print(
        json.dumps(
            summary(ledger, ledger_sha256=sha256_bytes(ledger_payload)), indent=2
        )
    )


if __name__ == "__main__":
    main()
