#!/usr/bin/env python3
"""Merge disjoint host-local held-out feature snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_heldout_snapshot import (
    merge_heldout_feature_snapshots,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = merge_heldout_feature_snapshots(
        input_roots=args.input_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
