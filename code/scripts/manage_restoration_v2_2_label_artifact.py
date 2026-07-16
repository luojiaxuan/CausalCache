"""Package and validate v2.2 eager restoration-label evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causalcache.restoration_v2_2_label_artifact import (
    build_artifact_manifest,
    package_label_evidence,
    pretty_json_bytes,
    read_label_evidence_archive,
)
from causalcache.restoration_v2_2_label_contract import (
    CANONICAL_ATTEMPT_ID,
    label_attempt_profile_for_id,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    archive = commands.add_parser("archive")
    archive.add_argument("--attempt-id", default=CANONICAL_ATTEMPT_ID)
    archive.add_argument("--raw-output-dir", type=Path)
    archive.add_argument("--global-ledger", type=Path)
    archive.add_argument("--output", type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("--evidence", type=Path, required=True)
    manifest = commands.add_parser("create-manifest")
    manifest.add_argument("--source-archive", type=Path, required=True)
    manifest.add_argument("--fresh-immutable-archive", type=Path, required=True)
    manifest.add_argument("--hf-revision", required=True)
    manifest.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "archive":
        profile = label_attempt_profile_for_id(args.attempt_id)
        result = package_label_evidence(
            raw_output_dir=args.raw_output_dir or profile.output_dir,
            global_ledger=args.global_ledger or profile.ledger_path,
            output_archive=args.output or profile.archive_path,
        )
    elif args.command == "validate":
        evidence = read_label_evidence_archive(args.evidence)
        result = {
            "status": "VALID_RESTORATION_V2_2_EAGER_LABEL_EVIDENCE",
            "outcome": evidence.outcome,
            "source_git_commit": evidence.source_git_commit,
            "run_contract_sha256": evidence.run_contract_sha256,
            "file_count": len(evidence.files),
            "tree_inventory_sha256": evidence.tree_inventory_sha256,
        }
    elif args.command == "create-manifest":
        result = build_artifact_manifest(
            source_archive=args.source_archive,
            fresh_immutable_archive=args.fresh_immutable_archive,
            hf_revision=args.hf_revision,
        )
        with args.output.open("xb") as destination:
            destination.write(pretty_json_bytes(result))
    else:
        raise AssertionError(f"unknown label artifact command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
