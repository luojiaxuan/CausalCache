#!/usr/bin/env python3
"""Validate a completed processor-freeze v2 root with four semantic workers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from causalcache.set_utility_processor_postflight_parallel_contract_v1 import (
    load_parallel_postflight_contract_v1,
)
from causalcache.set_utility_processor_postflight_parallel_v1 import (
    validate_completed_processor_freeze_root_parallel_v1,
)
from causalcache.set_utility_processor_freeze_contract_v2 import (
    CANONICAL_EXECUTION_CONFIG_PATH,
)
from causalcache.set_utility_processor_postflight_v2 import (
    build_processor_freeze_postflight_context_v2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--parallel-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parallel_contract = load_parallel_postflight_contract_v1(
        repository_root=args.repository_root,
        contract_path=args.parallel_contract,
    )
    supplied_execution_config = args.execution_config.resolve()
    canonical_execution_config = (
        parallel_contract.repository_root
        / CANONICAL_EXECUTION_CONFIG_PATH
    ).resolve()
    if supplied_execution_config != canonical_execution_config:
        raise ValueError("--execution-config differs from the bound canonical config")
    context = build_processor_freeze_postflight_context_v2(
        parallel_contract.historical_execution_contract,
        expected_git_revision=args.expected_git_revision,
    )
    summary = validate_completed_processor_freeze_root_parallel_v1(
        args.output_root,
        context=context,
    )
    summary["parallel_contract_sha256"] = parallel_contract.config_sha256
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
