#!/usr/bin/env python3
"""Materialize contextual entity requirements from a label-blind snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_contextual_inputs import (
    EVALUATION_ROLE_SCOPE,
    TRAIN_TUNE_ROLE_SCOPE,
    materialize_contextual_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--role-scope",
        choices=(TRAIN_TUNE_ROLE_SCOPE, EVALUATION_ROLE_SCOPE),
        default=TRAIN_TUNE_ROLE_SCOPE,
    )
    args = parser.parse_args()
    result = materialize_contextual_inputs(
        input_root=args.input_root.resolve(),
        output_root=args.output_root.resolve(),
        role_scope=args.role_scope,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
