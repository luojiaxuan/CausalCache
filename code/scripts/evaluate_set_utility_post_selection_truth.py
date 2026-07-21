#!/usr/bin/env python3
"""Plan or reduce selected-subset truth for the unified heldout evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_post_selection_truth import (
    plan_post_selection_truth,
    reduce_post_selection_truth,
)


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selector-output", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument(
        "--truth-root",
        action="append",
        type=Path,
        required=True,
        help="Repeat for every sealed formal truth root to union.",
    )
    parser.add_argument("--output", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    plan = subparsers.add_parser("plan")
    _shared(plan)
    reduce = subparsers.add_parser("reduce")
    _shared(reduce)
    reduce.add_argument("--normalization-floor", type=float, default=0.01)
    reduce.add_argument("--bootstrap-resamples", type=int, default=10_000)
    reduce.add_argument("--bootstrap-seed", type=int, default=20260721)
    reduce.add_argument("--bootstrap-interval", type=float, default=0.95)
    args = parser.parse_args()
    shared = {
        "selector_output_path": args.selector_output.resolve(),
        "input_root": args.input_root.resolve(),
        "heldout_manifest_path": args.heldout_manifest.resolve(),
        "truth_roots": tuple(path.resolve() for path in args.truth_root),
        "output_path": args.output.resolve(),
    }
    if args.mode == "plan":
        result = plan_post_selection_truth(**shared)
    else:
        result = reduce_post_selection_truth(
            **shared,
            normalization_floor=args.normalization_floor,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_interval=args.bootstrap_interval,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
