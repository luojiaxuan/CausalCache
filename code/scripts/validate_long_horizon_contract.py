#!/usr/bin/env python3
"""Validate the source-only long-horizon development contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causalcache.long_horizon_contract import (
    CANONICAL_CONFIG_PATH,
    validate_source_a,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=Path(CANONICAL_CONFIG_PATH))
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = validate_source_a(
        args.contract,
        repository_root=args.repository_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
