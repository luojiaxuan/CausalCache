#!/usr/bin/env python3
"""Validate one committed D1b envelope without loading the policy."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causalcache.set_utility_action_stability_envelope_v2 import (
    load_set_utility_action_stability_envelope_v2,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-envelope", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    projection = load_set_utility_action_stability_envelope_v2(
        args.execution_envelope,
        repository_root=Path(__file__).resolve().parents[2],
        verify_repository=True,
    )
    print(
        json.dumps(
            projection["validation"], ensure_ascii=False, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
