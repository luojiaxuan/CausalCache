"""Validate the expansion-substrate source freeze without loading the policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from causalcache.restoration_v2_2_expansion_substrate_contract import (
    CANONICAL_CONFIG_PATH,
    load_and_validate_contract,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--config", type=Path, default=Path(CANONICAL_CONFIG_PATH))
    return parser.parse_args(argv)


def validate_from_args(args: argparse.Namespace) -> dict[str, object]:
    root = (
        args.repository_root.resolve()
        if args.repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    _, validation = load_and_validate_contract(
        args.config,
        repository_root=root,
    )
    return validation


def main(argv: Sequence[str] | None = None) -> None:
    result = validate_from_args(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
