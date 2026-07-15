"""Package and validate raw evidence for the v2.1 full-45 substrate run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causalcache.restoration_v2_1_full_45_artifact import (
    CANONICAL_GIT_ARTIFACT_PATH,
    CANONICAL_GLOBAL_LEDGER_PATH,
    CANONICAL_HF_PATH,
    CANONICAL_HF_REPO,
    CANONICAL_RAW_ARCHIVE_PATH,
    CANONICAL_RAW_OUTPUT_DIR,
    package_raw_full_45_evidence,
    validate_clean_pushed_main,
    validate_committed_full_45_artifact,
    write_full_45_artifact_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic full-45 raw archive, bind its immutable "
            "private-HF copy to Git, or validate the committed binding."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    archive = subparsers.add_parser(
        "archive",
        help="validate canonical source-X raw output and write deterministic USTAR",
    )
    archive.add_argument("--repository-root", type=Path, required=True)
    archive.add_argument(
        "--raw-output-dir",
        type=Path,
        default=CANONICAL_RAW_OUTPUT_DIR,
    )
    archive.add_argument(
        "--global-attempt-ledger",
        type=Path,
        default=CANONICAL_GLOBAL_LEDGER_PATH,
    )
    archive.add_argument(
        "--output",
        type=Path,
        default=CANONICAL_RAW_ARCHIVE_PATH,
    )
    archive.add_argument("--source-git-commit", required=True)

    manifest = subparsers.add_parser(
        "create-manifest",
        help="bind the fresh immutable private-HF archive to the canonical Git path",
    )
    manifest.add_argument("--repository-root", type=Path, required=True)
    manifest.add_argument(
        "--fresh-immutable-archive",
        type=Path,
        required=True,
        help="fresh private-HF immutable-revision download, not the source archive",
    )
    manifest.add_argument("--source-git-commit", required=True)
    manifest.add_argument("--hf-repo", default=CANONICAL_HF_REPO)
    manifest.add_argument("--hf-immutable-revision", required=True)
    manifest.add_argument("--hf-path", default=CANONICAL_HF_PATH)
    manifest.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate",
        help="recompute archive or extracted evidence from clean descendant main Y",
    )
    validate.add_argument("--repository-root", type=Path, required=True)
    validate.add_argument("--evidence", type=Path, required=True)
    validate.add_argument("--current-git-commit", required=True)
    validate.add_argument(
        "--artifact",
        type=Path,
        default=Path(CANONICAL_GIT_ARTIFACT_PATH),
    )
    return parser


def _archive(args: argparse.Namespace) -> dict[str, object]:
    return package_raw_full_45_evidence(
        repository_root=args.repository_root,
        raw_output_dir=args.raw_output_dir,
        global_attempt_ledger=args.global_attempt_ledger,
        output_archive=args.output,
        source_git_commit=args.source_git_commit,
    )


def _create_manifest(args: argparse.Namespace) -> dict[str, object]:
    manifest = write_full_45_artifact_manifest(
        repository_root=args.repository_root,
        fresh_immutable_archive=args.fresh_immutable_archive,
        source_git_commit=args.source_git_commit,
        hf_repo=args.hf_repo,
        hf_immutable_revision=args.hf_immutable_revision,
        hf_path=args.hf_path,
        output=args.output,
    )
    return {
        "status": "CREATED_RESTORATION_V2_1_FULL_45_ARTIFACT",
        "output": str(Path(args.output).resolve()),
        "source_git_commit": manifest["source_execution"]["source_git_commit"],
        "outcome": manifest["result"]["outcome"],
        "hf_artifact": manifest["hf_artifact"],
        "raw_archive": {
            "sha256": manifest["raw_archive"]["sha256"],
            "size_bytes": manifest["raw_archive"]["size_bytes"],
            "tree_inventory_sha256": manifest["raw_archive"][
                "tree_inventory_sha256"
            ],
        },
    }


def _validate(args: argparse.Namespace) -> dict[str, object]:
    git = dict(validate_clean_pushed_main(args.repository_root))
    if git.get("commit") != args.current_git_commit:
        raise ValueError("--current-git-commit differs from clean pushed main")
    return validate_committed_full_45_artifact(
        repository_root=args.repository_root,
        current_git_commit=args.current_git_commit,
        evidence_path=args.evidence,
        artifact_path=args.artifact,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "archive":
        result = _archive(args)
    elif args.command == "create-manifest":
        result = _create_manifest(args)
    elif args.command == "validate":
        result = _validate(args)
    else:
        raise AssertionError(f"unknown full-45 artifact command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
