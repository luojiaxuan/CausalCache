#!/usr/bin/env python3
"""Validate the Freeze-B terminal-index repair source contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_freeze_b_v2 import (
    DEFAULT_CONFIG_PATH,
    load_freeze_b_v2_config,
    validate_freeze_b_v2_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    config = load_freeze_b_v2_config(args.repository_root, args.config)
    result = validate_freeze_b_v2_config(
        config,
        repository_root=args.repository_root,
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
