"""Materialize the frozen restoration-v2.2 label-expansion exposure ledger."""

from __future__ import annotations

import argparse
import json
import re
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
    build_exposure_ledger,
    load_json_object,
    pretty_json_bytes,
    sha256_bytes,
    summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-selection", type=Path, required=True)
    parser.add_argument("--source-file-manifest", type=Path, required=True)
    parser.add_argument("--parent-exposure", type=Path, required=True)
    parser.add_argument("--prior-output-exposure", type=Path, required=True)
    parser.add_argument("--expansion-selection", type=Path)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repo_root, text=True
    ).strip()


def _verify_clean_pushed_main(repo_root: Path, expected_revision: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
        raise ValueError("--git-revision must be a full 40-character commit")
    if _git(repo_root, "symbolic-ref", "--short", "HEAD") != "main":
        raise ValueError("exposure-ledger materialization requires branch main")
    if _git(repo_root, "rev-parse", "HEAD") != expected_revision:
        raise ValueError("--git-revision does not match checkout HEAD")
    if _git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("exposure-ledger materialization requires a clean checkout")
    if _git(repo_root, "rev-parse", "refs/remotes/origin/main") != expected_revision:
        raise ValueError("exposure-ledger materialization requires HEAD == origin/main")
    advertised = _git(
        repo_root, "ls-remote", "--exit-code", "origin", "refs/heads/main"
    ).split()
    if len(advertised) != 2 or advertised[0] != expected_revision:
        raise ValueError("exposure-ledger materialization requires pushed canonical main")


def _require_canonical_path(actual: Path, repo_root: Path, expected: str) -> None:
    if actual.resolve() != (repo_root / expected).resolve():
        raise ValueError(f"input must use canonical path {expected}")


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


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
    if args.output.exists():
        raise FileExistsError("label-expansion exposure output already exists")
    _verify_clean_pushed_main(repo_root, args.git_revision)
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

    generator = {"git_revision": args.git_revision}
    for prefix, relative_path in GENERATOR_PATHS.items():
        path = repo_root / relative_path
        generator[f"{prefix}_path"] = relative_path
        generator[f"{prefix}_sha256"] = sha256_bytes(path.read_bytes())
    ledger = build_exposure_ledger(
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
        generator=generator,
    )
    payload = pretty_json_bytes(ledger)
    _exclusive_write(args.output, payload)
    print(json.dumps(summary(ledger, ledger_sha256=sha256_bytes(payload)), indent=2))


if __name__ == "__main__":
    main()
