#!/usr/bin/env python3
"""Run the v3 parser-repair v2 exact state-id join and label replay."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.set_conditioned_v3_parser_repair import (
    evaluate_parser_repair_v2,
    read_parser_repair_v2_report,
)
from causalcache.set_conditioned_v3_parser_repair_v2_contract import (
    CANONICAL_CONFIG_PATH,
    RUNNER_FREEZE_PATH,
    exact_v2_input_bytes,
    load_frozen_parser_repair_v2_contract,
    validate_parser_repair_v2_execution_b,
    validate_v2_parent_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--repository-root", type=Path, required=True)
    evaluate.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
    evaluate.add_argument("--source-a-git-commit", required=True)
    evaluate.add_argument("--execution-b-git-commit", required=True)
    evaluate.add_argument("--execution-b-freeze", default=RUNNER_FREEZE_PATH)
    evaluate.add_argument("--input-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--repository-root", type=Path, required=True)
    validate.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
    validate.add_argument("--source-a-git-commit", required=True)
    validate.add_argument("--execution-b-git-commit", required=True)
    validate.add_argument("--execution-b-freeze", default=RUNNER_FREEZE_PATH)
    validate.add_argument("--expected-report-sha256", required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    return parser


def _run(args: argparse.Namespace) -> Mapping[str, Any]:
    contract = load_frozen_parser_repair_v2_contract(
        args.contract,
        repository_root=args.repository_root,
    )
    if args.command == "evaluate":
        if args.execution_b_freeze != RUNNER_FREEZE_PATH:
            raise ValueError("parser-repair v2 runner-freeze path drifted")
        source = validate_parser_repair_v2_execution_b(
            args.contract,
            repository_root=args.repository_root,
            source_a_git_commit=args.source_a_git_commit,
            execution_b_git_commit=args.execution_b_git_commit,
        )
        parent = validate_v2_parent_artifacts(
            contract,
            output_dir=args.output_dir,
            require_v2_report_absent=True,
        )
        feature_payload = exact_v2_input_bytes(
            contract,
            input_dir=args.input_dir,
            name="fresh_feature",
        )
        historical_payload = exact_v2_input_bytes(
            contract,
            input_dir=args.input_dir,
            name="historical_independent",
        )
        label_payload = exact_v2_input_bytes(
            contract,
            input_dir=args.input_dir,
            name="fresh_label",
        )
        report = evaluate_parser_repair_v2(
            contract=contract,
            output_dir=args.output_dir,
            feature_payload=feature_payload,
            label_payload=label_payload,
            historical_independent_payload=historical_payload,
            repair_source_a_git_commit=args.source_a_git_commit,
            repair_execution_b_git_commit=args.execution_b_git_commit,
            repair_runner_freeze_sha256=source["runner_freeze_sha256"],
        )
        return {"source": source, "parent": parent, "report": report}
    if args.execution_b_freeze != RUNNER_FREEZE_PATH:
        raise ValueError("parser-repair v2 runner-freeze path drifted")
    source = validate_parser_repair_v2_execution_b(
        args.contract,
        repository_root=args.repository_root,
        source_a_git_commit=args.source_a_git_commit,
        execution_b_git_commit=args.execution_b_git_commit,
    )
    parent = validate_v2_parent_artifacts(
        contract,
        output_dir=args.output_dir,
        require_v2_report_absent=False,
    )
    return {
        "source": source,
        "parent": parent,
        "report": read_parser_repair_v2_report(
            args.output_dir,
            contract=contract,
            expected_report_sha256=args.expected_report_sha256,
            expected_source_a_git_commit=args.source_a_git_commit,
            expected_execution_b_git_commit=args.execution_b_git_commit,
            expected_runner_freeze_sha256=source["runner_freeze_sha256"],
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(_run(_parser().parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
