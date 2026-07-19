#!/usr/bin/env python3
"""Record one processor v2 formal result as Git-safe README and JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
while str(_CODE_ROOT) in sys.path:
    sys.path.remove(str(_CODE_ROOT))
sys.path.insert(0, str(_CODE_ROOT))
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

from causalcache.set_utility_processor_result_v2 import (
    record_processor_freeze_v2_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--producer-repository-root", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-revision", required=True)
    parser.add_argument("--expected-recorder-git-revision", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--run-argv-evidence", type=Path, required=True)
    parser.add_argument("--start-evidence", type=Path, required=True)
    parser.add_argument("--outer-log", type=Path, required=True)
    parser.add_argument("--exit-code-evidence", type=Path, required=True)
    parser.add_argument("--end-evidence", type=Path, required=True)
    parser.add_argument("--postflight-argv-evidence", type=Path, required=True)
    parser.add_argument("--postflight-evidence", type=Path, required=True)
    parser.add_argument("--intended-hf-repo", required=True)
    parser.add_argument("--intended-hf-tag", required=True)
    parser.add_argument("--record-invalid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repository_root.resolve() != _REPOSITORY_ROOT:
        raise ValueError(
            "--repository-root differs from the executing recorder checkout"
        )
    summary = record_processor_freeze_v2_result(
        repository_root=args.repository_root,
        producer_repository_root=args.producer_repository_root,
        execution_config=args.execution_config,
        output_root=args.output_root,
        expected_git_revision=args.expected_git_revision,
        expected_recorder_git_revision=args.expected_recorder_git_revision,
        result_dir=args.result_dir,
        run_argv_evidence=args.run_argv_evidence,
        start_evidence=args.start_evidence,
        outer_log=args.outer_log,
        exit_code_evidence=args.exit_code_evidence,
        end_evidence=args.end_evidence,
        postflight_argv_evidence=args.postflight_argv_evidence,
        postflight_evidence=args.postflight_evidence,
        intended_hf_repo=args.intended_hf_repo,
        intended_hf_tag=args.intended_hf_tag,
        record_invalid=args.record_invalid,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
