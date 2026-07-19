#!/usr/bin/env python3
"""Materialize the byte-bound selected-image census v2 repair contract once."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from causalcache.set_utility_selected_image_census_contract_v2 import (
    CANONICAL_CONFIG_PATH,
    build_execution_config_skeleton,
    canonical_pretty_json_bytes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output = root / CANONICAL_CONFIG_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("selected-image census v2 repair contract already exists")
    config = build_execution_config_skeleton(repository_root=root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_pretty_json_bytes(config))


if __name__ == "__main__":
    main()
