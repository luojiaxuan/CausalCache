#!/usr/bin/env python3
"""Validate one completed selected-image census v2 output root read-only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from causalcache.set_utility_selected_image_census_contract_v2 import (
    load_execution_contract,
)
from causalcache.set_utility_selected_image_census_postflight_v2 import (
    build_selected_image_census_v2_postflight_context,
    validate_completed_selected_image_census_v2_root,
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
    context = build_selected_image_census_v2_postflight_context(
        contract,
        expected_git_revision=args.expected_git_revision,
    )
    summary = validate_completed_selected_image_census_v2_root(
        args.output_root,
        context=context,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
