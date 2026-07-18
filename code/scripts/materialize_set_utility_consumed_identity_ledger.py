#!/usr/bin/env python3
"""Materialize the canonical set-utility consumed-identity ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from causalcache.set_utility_consumed_ledger import (
    CANONICAL_LEDGER_MANIFEST_PATH,
    build_canonical_consumed_identity_ledger,
    write_ledger_manifest_exclusive,
)


def materialize(
    *,
    repository_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    ledger = build_canonical_consumed_identity_ledger(
        repository_root=repository_root
    )
    file_sha256 = write_ledger_manifest_exclusive(ledger, output_path)
    return {
        "status": ledger.payload["status"],
        "protocol_id": ledger.payload["protocol_id"],
        "output_path": str(output_path),
        "file_sha256": file_sha256,
        "assignment_count": ledger.payload["assignment_count"],
        "assignment_inventory_sha256": ledger.payload[
            "assignment_inventory_sha256"
        ],
        "partition_counts": {
            name: record["source_id_count"]
            for name, record in ledger.payload["partitions"].items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    output = args.output
    if output is None:
        output = repository_root / CANONICAL_LEDGER_MANIFEST_PATH
    elif not output.is_absolute():
        output = repository_root / output
    summary = materialize(
        repository_root=repository_root,
        output_path=output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
