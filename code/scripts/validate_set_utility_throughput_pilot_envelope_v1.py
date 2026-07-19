#!/usr/bin/env python3
"""Validate one fresh exact-84-call execution envelope without loading torch."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causalcache.set_utility_throughput_pilot_envelope_v1 import (
    load_set_utility_throughput_pilot_envelope_v1,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-envelope", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    _, validation = load_set_utility_throughput_pilot_envelope_v1(
        args.execution_envelope,
        repository_root=repository_root,
        verify_repository=True,
        verify_local_artifacts="stat",
        require_fresh_preflight=True,
        verify_current_environment=True,
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
