#!/usr/bin/env python3
"""Materialize a train-only on-policy enriched contextual snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_contextual_enrichment import (
    materialize_contextual_enriched_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-input-root", type=Path, required=True)
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duplicate-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    manifest = materialize_contextual_enriched_inputs(
        base_input_root=args.base_input_root.resolve(),
        schedule_root=args.schedule_root.resolve(),
        label_roots=tuple(path.resolve() for path in args.label_root),
        output_root=args.output_root.resolve(),
        duplicate_tolerance=args.duplicate_tolerance,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
