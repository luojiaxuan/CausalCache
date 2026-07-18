#!/usr/bin/env python3
"""Validate the long-pool discovery Source-A without touching source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_long_pool import (
    CANONICAL_CONFIG_PATH,
    validate_discovery_source_only,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_discovery_source_only(
        repository_root=args.repository_root,
        contract_path=args.contract,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
