"""Validate or package the source-only invalid expansion-label forensic bytes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causalcache.restoration_v2_2_expansion_labels_invalid_forensic import (
    CANONICAL_CONFIG_PATH,
    load_frozen_forensic_contract,
    package_invalid_forensic_archive,
    read_invalid_forensic_archive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    validate_contract = commands.add_parser("validate-contract")
    _add_contract_arguments(validate_contract)

    package = commands.add_parser("package")
    _add_contract_arguments(package)
    package.add_argument("--raw-output-dir", type=Path, required=True)
    package.add_argument("--external-global-ledger", type=Path, required=True)
    package.add_argument("--external-even-ledger", type=Path, required=True)
    package.add_argument("--external-odd-ledger", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate")
    _add_contract_arguments(validate)
    validate.add_argument("--evidence", type=Path, required=True)
    return parser


def _add_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(CANONICAL_CONFIG_PATH),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    contract = load_frozen_forensic_contract(
        args.config,
        repository_root=args.repository_root,
    )
    if args.command == "validate-contract":
        result = {
            "status": "VALID_SOURCE_ONLY_INVALID_FORENSIC_CONTRACT",
            "protocol_id": contract.data["protocol_id"],
            "config_sha256": contract.sha256,
            "expected_member_count": contract.archive_contract[
                "expected_member_count"
            ],
            "formal_label_loader_eligible": contract.data["formal_consumption"][
                "formal_label_loader_eligible"
            ],
            "hf_publish_authorized": contract.data["network_contract"][
                "hf_publish_authorized"
            ],
        }
    elif args.command == "package":
        result = package_invalid_forensic_archive(
            contract=contract,
            raw_output_dir=args.raw_output_dir,
            external_global_ledger=args.external_global_ledger,
            external_worker_ledgers={
                "even": args.external_even_ledger,
                "odd": args.external_odd_ledger,
            },
            output_archive=args.output,
        )
    elif args.command == "validate":
        evidence = read_invalid_forensic_archive(args.evidence, contract=contract)
        result = {
            "status": "VALID_INVALID_EXPANSION_EXACT_LABEL_FORENSIC_ARCHIVE",
            "protocol_id": evidence.manifest["protocol_id"],
            "artifact_class": evidence.manifest["artifact_class"],
            "producer_attempt_status": evidence.manifest["source_attempt"][
                "original_attempt_status"
            ],
            "formal_label_loader_eligible": evidence.manifest[
                "formal_consumption"
            ]["formal_label_loader_eligible"],
            "member_count": len(evidence.files),
            "tree_inventory_sha256": evidence.tree_inventory_sha256,
        }
    else:
        raise AssertionError(f"unknown invalid-forensic command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
