#!/usr/bin/env python3
"""Validate the committed strict-determinism D2 execution envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_action_stability_execution_v3 import (
    load_action_stability_envelope_v3,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-envelope", type=Path, required=True)
    parser.add_argument("--allow-stale-preflight", action="store_true")
    parser.add_argument("--skip-current-container", action="store_true")
    args = parser.parse_args()
    projection = load_action_stability_envelope_v3(
        args.execution_envelope,
        require_fresh_preflight=not args.allow_stale_preflight,
        verify_current_container=not args.skip_current_container,
    )
    print(json.dumps(projection["validation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
