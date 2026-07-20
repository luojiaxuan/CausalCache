#!/usr/bin/env python3
"""Materialize label-blind schedules for the frozen held-out selectors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from causalcache.set_utility_evaluation_schedule import (
    materialize_evaluation_schedules,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-inventory", type=Path, required=True)
    parser.add_argument("--sealed-selections", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partition-index", type=int, required=True)
    parser.add_argument("--partition-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize_evaluation_schedules(
        config_path=args.config.resolve(),
        inventory_path=args.state_inventory.resolve(),
        selections_path=args.sealed_selections.resolve(),
        source_manifest_path=args.source_manifest.resolve(),
        output_root=args.output_root.resolve(),
        partition_index=args.partition_index,
        partition_count=args.partition_count,
        workers=args.workers,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
