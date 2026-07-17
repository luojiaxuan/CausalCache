"""Manage expansion-substrate runner freeze and immutable raw evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causalcache.restoration_v2_2_expansion_substrate_artifact import (
    CANONICAL_ARCHIVE_PATH,
    CANONICAL_HF_PATH,
    CANONICAL_HF_REPO,
    CANONICAL_HF_TAG,
    CANONICAL_LEDGER_PATH,
    CANONICAL_OUTPUT_DIR,
    CANONICAL_RUNNER_FREEZE_PATH,
    build_expansion_substrate_artifact_manifest,
    load_and_validate_runner_freeze,
    materialize_runner_freeze,
    package_raw_expansion_substrate_evidence,
    pretty_json_bytes,
    read_expansion_substrate_archive,
    validate_fresh_hf_evidence,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("materialize-runner-freeze")
    freeze.add_argument("--repository-root", type=Path, required=True)
    freeze.add_argument("--runner-source-git-commit", required=True)
    freeze.add_argument(
        "--output",
        type=Path,
        default=Path(CANONICAL_RUNNER_FREEZE_PATH),
    )

    validate_freeze = commands.add_parser("validate-runner-freeze")
    validate_freeze.add_argument("--repository-root", type=Path, required=True)
    validate_freeze.add_argument(
        "--runner-freeze",
        type=Path,
        default=Path(CANONICAL_RUNNER_FREEZE_PATH),
    )

    package = commands.add_parser("package")
    package.add_argument("--raw-output-dir", type=Path, default=CANONICAL_OUTPUT_DIR)
    package.add_argument(
        "--global-attempt-ledger", type=Path, default=CANONICAL_LEDGER_PATH
    )
    package.add_argument("--output", type=Path, default=CANONICAL_ARCHIVE_PATH)
    package.add_argument("--source-git-commit", required=True)
    package.add_argument("--config-sha256", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--evidence", type=Path, required=True)
    validate.add_argument("--source-git-commit", required=True)
    validate.add_argument("--config-sha256", required=True)

    fresh = commands.add_parser("validate-fresh-hf")
    _add_fresh_arguments(fresh)

    manifest = commands.add_parser("create-manifest")
    _add_fresh_arguments(manifest)
    manifest.add_argument("--output", type=Path, required=True)
    return parser


def _add_fresh_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--fresh-immutable-archive", type=Path, required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--hf-repo", default=CANONICAL_HF_REPO)
    parser.add_argument("--hf-tag", default=CANONICAL_HF_TAG)
    parser.add_argument("--hf-path", default=CANONICAL_HF_PATH)
    parser.add_argument("--hf-immutable-revision", required=True)


def _fresh_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "source_archive": args.source_archive,
        "fresh_immutable_archive": args.fresh_immutable_archive,
        "source_git_commit": args.source_git_commit,
        "config_sha256": args.config_sha256,
        "hf_repo": args.hf_repo,
        "hf_tag": args.hf_tag,
        "hf_path": args.hf_path,
        "hf_immutable_revision": args.hf_immutable_revision,
    }


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(payload)


def _materialize_runner_freeze(args: argparse.Namespace) -> dict[str, object]:
    return materialize_runner_freeze(
        repository_root=args.repository_root,
        runner_source_git_commit=args.runner_source_git_commit,
        output_path=args.output,
    )


def _validate_runner_freeze(args: argparse.Namespace) -> dict[str, object]:
    _freeze, validation = load_and_validate_runner_freeze(
        args.runner_freeze,
        repository_root=args.repository_root,
    )
    return {
        "status": "VALID_COMMITTED_PUSHED_EXPANSION_SUBSTRATE_RUNNER_FREEZE",
        **validation,
        "policy_or_gpu_execution_authorized": True,
    }


def _package(args: argparse.Namespace) -> dict[str, object]:
    return package_raw_expansion_substrate_evidence(
        raw_output_dir=args.raw_output_dir,
        global_attempt_ledger=args.global_attempt_ledger,
        output_archive=args.output,
        source_git_commit=args.source_git_commit,
        config_sha256=args.config_sha256,
    )


def _validate(args: argparse.Namespace) -> dict[str, object]:
    evidence = read_expansion_substrate_archive(
        args.evidence,
        expected_source_git_commit=args.source_git_commit,
        expected_config_sha256=args.config_sha256,
    )
    return {
        "status": "VALID_V2_2_LABEL_EXPANSION_SUBSTRATE_RAW_EVIDENCE",
        "source_git_commit": evidence.source_git_commit,
        "execution_git_commit": evidence.execution_git_commit,
        "outcome": evidence.outcome,
        "fixed_state_denominator": evidence.fixed_state_denominator,
        "attempted_state_count": evidence.attempted_state_count,
        "completed_state_count": evidence.completed_state_count,
        "planned_counts": dict(evidence.planned_counts),
        "actual_counts": dict(evidence.actual_counts),
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
        "file_count": len(evidence.inventory),
    }


def _validate_fresh(args: argparse.Namespace) -> dict[str, object]:
    evidence, hf = validate_fresh_hf_evidence(**_fresh_kwargs(args))
    return {
        "status": "VALID_FRESH_IMMUTABLE_EXPANSION_SUBSTRATE_HF_EVIDENCE",
        "outcome": evidence.outcome,
        "raw_tree_inventory_sha256": evidence.tree_inventory_sha256,
        "hf_artifact": hf,
    }


def _create_manifest(args: argparse.Namespace) -> dict[str, object]:
    manifest = build_expansion_substrate_artifact_manifest(**_fresh_kwargs(args))
    output = Path(args.output)
    _exclusive_write(output, pretty_json_bytes(manifest))
    return {
        "status": "CREATED_EXPANSION_SUBSTRATE_ARTIFACT_MANIFEST",
        "output": str(output.resolve()),
        "outcome": manifest["result"]["outcome"],
        "raw_archive": manifest["raw_archive"],
        "hf_artifact": manifest["hf_artifact"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "materialize-runner-freeze":
        result = _materialize_runner_freeze(args)
    elif args.command == "validate-runner-freeze":
        result = _validate_runner_freeze(args)
    elif args.command == "package":
        result = _package(args)
    elif args.command == "validate":
        result = _validate(args)
    elif args.command == "validate-fresh-hf":
        result = _validate_fresh(args)
    elif args.command == "create-manifest":
        result = _create_manifest(args)
    else:
        raise AssertionError(f"unknown artifact manager command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
