#!/usr/bin/env python3
"""Build or validate the train-only D1b action-stability source contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_action_stability_contract_v2 import (
    CANONICAL_CONFIG_PATH,
    build_action_stability_source_v2_config_skeleton,
    load_action_stability_source_v2_contract,
    validate_action_stability_source_v2_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", default=CANONICAL_CONFIG_PATH)
    parser.add_argument("--emit-skeleton", action="store_true")
    args = parser.parse_args()
    if args.emit_skeleton:
        payload = build_action_stability_source_v2_config_skeleton(
            repository_root=args.repository_root
        )
    else:
        contract = load_action_stability_source_v2_contract(
            repository_root=args.repository_root,
            config_path=args.config,
        )
        payload = validate_action_stability_source_v2_config(
            contract.data,
            repository_root=contract.repository_root,
        )
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
