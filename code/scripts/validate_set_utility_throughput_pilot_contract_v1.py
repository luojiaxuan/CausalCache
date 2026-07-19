#!/usr/bin/env python3
"""Validate the 12-state train-only throughput pilot source contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_throughput_pilot_contract_v1 import (
    CANONICAL_CONFIG_PATH,
    load_train_only_throughput_pilot_v1_contract,
    validate_train_only_throughput_pilot_v1_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", default=CANONICAL_CONFIG_PATH)
    args = parser.parse_args()
    contract = load_train_only_throughput_pilot_v1_contract(
        repository_root=args.repository_root,
        config_path=args.config,
    )
    validation = validate_train_only_throughput_pilot_v1_config(
        contract.data,
        repository_root=contract.repository_root,
    )
    print(json.dumps(validation, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
