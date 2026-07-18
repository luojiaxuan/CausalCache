#!/usr/bin/env python3
"""Validate the committed Freeze-B source config without materializing data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_freeze_b_v1 import (
    load_freeze_b_config,
    validate_freeze_b_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        default="code/configs/causalcache_set_utility_freeze_b_v1.json",
    )
    args = parser.parse_args()
    config = load_freeze_b_config(args.repository_root, args.config)
    result = validate_freeze_b_config(
        config,
        repository_root=args.repository_root,
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
