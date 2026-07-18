#!/usr/bin/env python3
"""Validate the frozen source-only census contract without executing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_full_pool_contract import validate_source_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            validate_source_only(repository_root=args.repository_root.resolve()),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
