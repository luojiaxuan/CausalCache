#!/usr/bin/env python3
"""Freeze post-GO native-action replay schedules on the exact held-out track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_native_replay import materialize_native_replay_schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-config", type=Path, required=True)
    parser.add_argument("--state-inventory", type=Path, required=True)
    parser.add_argument("--sealed-selections", type=Path, required=True)
    parser.add_argument("--heldout-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_native_replay_schedule(
        config_path=args.heldout_config.resolve(),
        inventory_path=args.state_inventory.resolve(),
        selections_path=args.sealed_selections.resolve(),
        heldout_result_path=args.heldout_result.resolve(),
        source_manifest_path=args.source_manifest.resolve(),
        label_roots=tuple(path.resolve() for path in args.label_root),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
