#!/usr/bin/env python3
"""Validate the committed full-pool census execution contract without running it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_full_pool_contract import load_execution_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_execution_contract(
        repository_root=args.repository_root.resolve(),
        execution_config_path=args.execution_config,
    )
    print(
        json.dumps(
            {
                "status": "VALID_SET_UTILITY_FULL_POOL_CENSUS_V2_EXECUTION",
                "protocol_id": contract.data["protocol_id"],
                "p1_manifest_sha256": contract.binding_sha256(
                    "p1_inventory_manifest"
                ),
                "consumed_ledger_sha256": contract.binding_sha256(
                    "consumed_ledger"
                ),
                "source_file_access_authorized": True,
                "row_decode_authorized": True,
                "semantic_census_authorized": True,
                "output_write_authorized": True,
                "role_assignment_count": 0,
                "query_state_selection_count": 0,
                "model_load_count": 0,
                "restoration_label_count": 0,
                "training_count": 0,
                "gpu_count": 0,
                "closed_loop_count": 0,
                "sealed_test_access_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
