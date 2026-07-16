"""Package and validate restoration-v2.2 eager raw evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causalcache.restoration_v2_2_eager_artifact import (
    build_v22_artifact_manifest,
    package_raw_v22_evidence,
    pretty_json_bytes,
    read_v22_evidence_archive,
)
from causalcache.restoration_v2_2_eager_contract import (
    CANONICAL_ARCHIVE_PATH,
    CANONICAL_HF_PATH,
    CANONICAL_HF_REPO,
    CANONICAL_LEDGER_PATH,
    CANONICAL_OUTPUT_DIR,
)
from scripts.run_restoration_v2_2_eager_substrate import seal_interrupted_attempt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    archive = commands.add_parser("archive")
    archive.add_argument("--raw-output-dir", type=Path, default=CANONICAL_OUTPUT_DIR)
    archive.add_argument(
        "--global-attempt-ledger", type=Path, default=CANONICAL_LEDGER_PATH
    )
    archive.add_argument("--output", type=Path, default=CANONICAL_ARCHIVE_PATH)
    archive.add_argument("--source-git-commit", required=True)

    manifest = commands.add_parser("create-manifest")
    manifest.add_argument(
        "--source-archive", type=Path, default=CANONICAL_ARCHIVE_PATH
    )
    manifest.add_argument("--fresh-immutable-archive", type=Path, required=True)
    manifest.add_argument("--source-git-commit", required=True)
    manifest.add_argument("--hf-immutable-revision", required=True)
    manifest.add_argument("--hf-repo", default=CANONICAL_HF_REPO)
    manifest.add_argument("--hf-path", default=CANONICAL_HF_PATH)
    manifest.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--evidence", type=Path, required=True)
    validate.add_argument("--source-git-commit", required=True)

    seal = commands.add_parser("seal-interrupted")
    seal.add_argument("--raw-output-dir", type=Path, default=CANONICAL_OUTPUT_DIR)
    seal.add_argument(
        "--global-attempt-ledger", type=Path, default=CANONICAL_LEDGER_PATH
    )
    return parser


def _archive(args: argparse.Namespace) -> dict[str, object]:
    return package_raw_v22_evidence(
        raw_output_dir=args.raw_output_dir,
        global_attempt_ledger=args.global_attempt_ledger,
        output_archive=args.output,
        source_git_commit=args.source_git_commit,
    )


def _create_manifest(args: argparse.Namespace) -> dict[str, object]:
    evidence = read_v22_evidence_archive(
        args.source_archive,
        expected_source_git_commit=args.source_git_commit,
    )
    manifest = build_v22_artifact_manifest(
        evidence=evidence,
        source_archive_path=args.source_archive,
        fresh_immutable_archive=args.fresh_immutable_archive,
        hf_repo=args.hf_repo,
        hf_immutable_revision=args.hf_immutable_revision,
        hf_path=args.hf_path,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        destination.write(pretty_json_bytes(manifest))
    return {
        "status": "CREATED_RESTORATION_V2_2_EAGER_ARTIFACT_MANIFEST",
        "output": str(output.resolve()),
        "outcome": evidence.outcome,
        "hf_artifact": manifest["hf_artifact"],
        "raw_archive": manifest["raw_archive"],
    }


def _validate(args: argparse.Namespace) -> dict[str, object]:
    evidence = read_v22_evidence_archive(
        args.evidence,
        expected_source_git_commit=args.source_git_commit,
    )
    return {
        "status": "VALID_RESTORATION_V2_2_EAGER_RAW_EVIDENCE",
        "source_git_commit": evidence.source_git_commit,
        "outcome": evidence.outcome,
        "completed_state_count": evidence.completed_state_count,
        "attempted_state_count": evidence.attempted_state_count,
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
        "file_count": len(evidence.inventory),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "archive":
        result = _archive(args)
    elif args.command == "create-manifest":
        result = _create_manifest(args)
    elif args.command == "validate":
        result = _validate(args)
    elif args.command == "seal-interrupted":
        result = seal_interrupted_attempt(
            output_dir=args.raw_output_dir,
            global_ledger=args.global_attempt_ledger,
        )
    else:
        raise AssertionError(f"unknown v2.2 artifact command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
