#!/usr/bin/env python3
"""Materialize exact triples/quads for the small-history B4 oracle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from causalcache.set_utility_b4_oracle import materialize_b4_oracle_schedules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()
    result = materialize_b4_oracle_schedules(
        config_path=args.config.resolve(),
        selections_path=args.selections.resolve(),
        source_manifest_path=args.source_manifest.resolve(),
        output_root=args.output_root.resolve(),
        workers=args.workers,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
