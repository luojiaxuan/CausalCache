#!/usr/bin/env python3
"""Validate a completed processor image-contract v2 output root read-only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from causalcache.set_utility_processor_freeze_contract_v2 import (
    load_execution_contract,
)
from causalcache.set_utility_processor_postflight_v2 import (
    build_processor_freeze_postflight_context_v2,
    validate_completed_processor_freeze_root_v2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_execution_contract(
        repository_root=args.repository_root,
        execution_config_path=args.execution_config,
    )
    context = build_processor_freeze_postflight_context_v2(
        contract,
        expected_git_revision=args.expected_git_revision,
    )
    summary = validate_completed_processor_freeze_root_v2(
        args.output_root,
        context=context,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
