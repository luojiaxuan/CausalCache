#!/usr/bin/env python3
"""Validate the source-only fresh-16 claim serialization repair contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causalcache.gate_v1_fresh16_claim_serialization_repair_contract import (
    CANONICAL_CONFIG_PATH,
    validate_fresh16_claim_serialization_repair_source_only_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", default=CANONICAL_CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate_fresh16_claim_serialization_repair_source_only_contract(
        args.contract,
        repository_root=args.repository_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
