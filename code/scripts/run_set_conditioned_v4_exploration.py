#!/usr/bin/env python3
"""Run the sealed CPU-only frozen-base residual v4 development study."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.set_conditioned_v4_contract import (
    CANONICAL_CONFIG_PATH,
    EXECUTION_B_RUNNER_FREEZE_PATH,
    validate_execution_b,
)
from causalcache.set_conditioned_v4_exploration import (
    BASE_CHECKPOINT_RECORDS,
    claim_fresh_label_access,
    evaluate_sealed_fresh16,
    load_fresh_features,
    train_and_seal,
    verify_evaluation_report,
    verify_fresh_label_access_claim,
    verify_label_blind_seal,
)


FORMAL_FEATURE_NAME = "formal-feature-cache.tar"
FORMAL_LABEL_NAME = "formal-label-cache.tar"
FRESH_FEATURE_NAME = "fresh16-feature-states.jsonl"
FRESH_LABEL_NAME = "fresh16-label-states.jsonl"
HISTORICAL_INDEPENDENT_NAME = "fresh16-historical-independent.json"
BASE_MANIFEST_NAME = "independent-ensemble-manifest.json"
DEVELOPMENT_REPORT_NAME = "fresh16-consumed-development-report.json"
LABEL_ACCESS_CLAIM_NAME = "fresh-label-access-claim.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train-seal")
    train.add_argument("--input-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--repository-root", type=Path, required=True)
    train.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
    train.add_argument("--source-a-git-commit", required=True)
    train.add_argument("--execution-b-git-commit", required=True)
    train.add_argument(
        "--execution-b-freeze",
        default=EXECUTION_B_RUNNER_FREEZE_PATH,
    )

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--input-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--repository-root", type=Path, required=True)
    evaluate.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
    evaluate.add_argument("--source-a-git-commit", required=True)
    evaluate.add_argument("--execution-b-git-commit", required=True)
    evaluate.add_argument(
        "--execution-b-freeze",
        default=EXECUTION_B_RUNNER_FREEZE_PATH,
    )

    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    return parser


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required local input is missing or unsafe: {path}")
    return path.read_bytes()


def _validate_execution(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.execution_b_freeze != EXECUTION_B_RUNNER_FREEZE_PATH:
        raise ValueError("Execution-B runner-freeze path differs from v4 contract")
    return validate_execution_b(
        args.contract,
        repository_root=args.repository_root,
        source_a_git_commit=args.source_a_git_commit,
        execution_b_git_commit=args.execution_b_git_commit,
    )


def _verify_seal_execution(
    seal: Mapping[str, Any],
    args: argparse.Namespace,
    validation: Mapping[str, Any],
) -> None:
    if (
        seal.get("source_a_git_commit") != args.source_a_git_commit
        or seal.get("execution_b_git_commit")
        != validation["execution_b_git_commit"]
        or seal.get("runner_freeze_sha256") != validation["runner_freeze_sha256"]
        or seal.get("contract_sha256") != validation["contract_sha256"]
    ):
        raise ValueError("v4 label-blind seal differs from current Execution-B")


def _base_checkpoint_inputs(input_dir: Path) -> Mapping[str, bytes]:
    return {
        str(record["path"]): _read(input_dir / Path(str(record["path"])).name)
        for record in BASE_CHECKPOINT_RECORDS
    }


def _run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.command == "train-seal":
        # note (luojiaxuan): Fail-closed lineage validation must finish before
        # opening formal, Fresh, manifest, or checkpoint input bytes.
        validation = _validate_execution(args)
        return train_and_seal(
            formal_feature_archive=_read(args.input_dir / FORMAL_FEATURE_NAME),
            formal_label_archive=_read(args.input_dir / FORMAL_LABEL_NAME),
            fresh_feature_payload=_read(args.input_dir / FRESH_FEATURE_NAME),
            base_manifest_payload=_read(args.input_dir / BASE_MANIFEST_NAME),
            base_checkpoint_payloads=_base_checkpoint_inputs(args.input_dir),
            output_dir=args.output_dir,
            source_a_git_commit=args.source_a_git_commit,
            execution_b_git_commit=validation["execution_b_git_commit"],
            runner_freeze_sha256=validation["runner_freeze_sha256"],
            contract_sha256=validation["contract_sha256"],
        )
    if args.command == "evaluate":
        # note (luojiaxuan): The pushed Execution-B and byte-identical seal are
        # verified before creating the one Fresh16 access claim or reading data.
        validation = _validate_execution(args)
        seal = verify_label_blind_seal(args.output_dir)
        _verify_seal_execution(seal, args, validation)
        if (args.output_dir / DEVELOPMENT_REPORT_NAME).exists():
            raise ValueError("v4 consumed-development evaluation is already complete")
        if (args.output_dir / LABEL_ACCESS_CLAIM_NAME).exists():
            raise ValueError(
                "v4 access was already claimed without a report; use a versioned repair"
            )
        feature_payload = _read(args.input_dir / FRESH_FEATURE_NAME)
        prevalidated_features = load_fresh_features(feature_payload)
        claim_fresh_label_access(args.output_dir)
        verify_fresh_label_access_claim(args.output_dir)
        label_payload = _read(args.input_dir / FRESH_LABEL_NAME)
        historical_payload = _read(args.input_dir / HISTORICAL_INDEPENDENT_NAME)
        return evaluate_sealed_fresh16(
            output_dir=args.output_dir,
            feature_payload=feature_payload,
            label_payload=label_payload,
            historical_independent_payload=historical_payload,
            prevalidated_features=prevalidated_features,
        )

    report_path = args.output_dir / DEVELOPMENT_REPORT_NAME
    if report_path.exists():
        return {"report": verify_evaluation_report(args.output_dir)}
    seal = verify_label_blind_seal(args.output_dir)
    result: dict[str, Any] = {"seal": seal}
    claim_path = args.output_dir / LABEL_ACCESS_CLAIM_NAME
    if claim_path.exists():
        result["claim"] = verify_fresh_label_access_claim(args.output_dir)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(_run(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
