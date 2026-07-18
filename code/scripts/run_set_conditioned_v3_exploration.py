#!/usr/bin/env python3
"""Run the sealed CPU-only set-conditioned v3 exploration state machine."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.set_conditioned_v3_exploration import (
    claim_fresh_label_access,
    evaluate_sealed_fresh16,
    train_and_seal,
    verify_fresh_label_access_claim,
    verify_label_blind_seal,
)
from causalcache.set_conditioned_v3_contract import (
    CANONICAL_CONFIG_PATH,
    load_frozen_set_conditioned_v3_contract,
    validate_source_a,
)


FORMAL_FEATURE_NAME = "formal-feature-cache.tar"
FORMAL_LABEL_NAME = "formal-label-cache.tar"
FRESH_FEATURE_NAME = "fresh16-feature-states.jsonl"
FRESH_LABEL_NAME = "fresh16-label-states.jsonl"
HISTORICAL_INDEPENDENT_NAME = "fresh16-historical-independent.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train-seal")
    train.add_argument("--input-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--repository-root", type=Path, required=True)
    train.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
    train.add_argument("--source-a-git-commit", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--input-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    return parser


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required local input is missing or unsafe: {path}")
    return path.read_bytes()


def _run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.command == "train-seal":
        contract = load_frozen_set_conditioned_v3_contract(
            args.contract,
            repository_root=args.repository_root,
        )
        validation = validate_source_a(
            args.contract,
            repository_root=args.repository_root,
        )
        if validation["source_a"]["head"] != args.source_a_git_commit:
            raise ValueError("requested Source-A commit differs from validated HEAD")
        return train_and_seal(
            formal_feature_archive=_read(args.input_dir / FORMAL_FEATURE_NAME),
            formal_label_archive=_read(args.input_dir / FORMAL_LABEL_NAME),
            fresh_feature_payload=_read(args.input_dir / FRESH_FEATURE_NAME),
            output_dir=args.output_dir,
            source_a_git_commit=args.source_a_git_commit,
            contract_sha256=contract.sha256,
        )
    if args.command == "evaluate":
        if (args.output_dir / "fresh16-development-report.json").exists():
            raise ValueError(
                "fresh16 development evaluation is already complete; use validate"
            )
        # note (luojiaxuan): The persisted claim is verified before this process
        # opens even one fresh-label byte, preserving the Source-A access order.
        claim_fresh_label_access(args.output_dir)
        verify_fresh_label_access_claim(args.output_dir)
        return evaluate_sealed_fresh16(
            output_dir=args.output_dir,
            feature_payload=_read(args.input_dir / FRESH_FEATURE_NAME),
            label_payload=_read(args.input_dir / FRESH_LABEL_NAME),
            historical_independent_payload=_read(
                args.input_dir / HISTORICAL_INDEPENDENT_NAME
            ),
        )
    seal = verify_label_blind_seal(args.output_dir)
    result: dict[str, Any] = {"seal": seal}
    claim_path = args.output_dir / "fresh-label-access-claim.json"
    if claim_path.exists():
        result["claim"] = verify_fresh_label_access_claim(args.output_dir)
    report_path = args.output_dir / "fresh16-development-report.json"
    if report_path.exists():
        report = json.loads(_read(report_path))
        if not isinstance(report, Mapping):
            raise ValueError("fresh16 development report must be a JSON object")
        result["report"] = report
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(_run(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
