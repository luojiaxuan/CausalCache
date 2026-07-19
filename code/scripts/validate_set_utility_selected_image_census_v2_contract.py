#!/usr/bin/env python3
"""Validate the committed selected-image census v2 repair contract."""

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
    validate_cpu_only_source,
    validation_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    contract = load_execution_contract(
        repository_root=root,
        execution_config_path=args.execution_config,
    )
    summary = validation_summary(contract)
    summary["source_validation"] = validate_cpu_only_source(repository_root=root)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
