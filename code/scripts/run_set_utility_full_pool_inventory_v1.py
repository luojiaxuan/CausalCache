#!/usr/bin/env python3
"""Validate or materialize the metadata-only GUIOdyssey full-pool inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_full_pool_inventory_v1 import (
    CANONICAL_CONFIG_PATH,
    load_frozen_inventory_contract,
    materialize_remote_inventory,
    read_token_file,
    validate_inventory_source_only,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-source")
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--token-file", type=Path, required=True)
    inventory.add_argument("--output", type=Path)
    return parser.parse_args()


def _canonical_output_path(
    *, repository_root: Path, configured_path: str, supplied_path: Path | None
) -> Path:
    expected = repository_root / configured_path
    supplied = expected if supplied_path is None else supplied_path
    supplied = supplied if supplied.is_absolute() else repository_root / supplied
    if supplied.resolve() != expected.resolve() or expected.is_symlink():
        raise ValueError("full-pool inventory output path is not canonical")
    return expected


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve()
    if args.command == "validate-source":
        result = validate_inventory_source_only(
            repository_root=root,
            contract_path=args.contract,
        )
    else:
        contract = load_frozen_inventory_contract(
            repository_root=root,
            contract_path=args.contract,
        )
        output = _canonical_output_path(
            repository_root=root,
            configured_path=str(contract.output["manifest_path"]),
            supplied_path=args.output,
        )
        token = read_token_file(args.token_file)
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "metadata inventory requires huggingface_hub"
            ) from error
        api = HfApi(token=token)
        manifest, manifest_sha256 = materialize_remote_inventory(
            api=api,
            contract=contract,
            output_path=output,
        )
        result = {
            "status": manifest["status"],
            "protocol_id": manifest["protocol_id"],
            "revision": manifest["source"]["revision"],
            "file_count": manifest["inventory"]["file_count"],
            "total_size_bytes": manifest["inventory"]["total_size_bytes"],
            "manifest_sha256": manifest_sha256,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
