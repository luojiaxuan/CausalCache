#!/usr/bin/env python3
"""Materialize or reduce budget-deferral selected-only evaluation truth."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from causalcache.set_utility_budget_deferral_truth_v1 import (
    materialize_budget_deferral_truth_schedule,
    reduce_budget_deferral_truth,
)


def _contract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-seal", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    schedule = subparsers.add_parser("schedule")
    _contract_args(schedule)
    schedule.add_argument("--output-root", type=Path, required=True)
    schedule.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))

    reduce = subparsers.add_parser("reduce")
    _contract_args(reduce)
    reduce.add_argument("--schedule-root", type=Path, required=True)
    reduce.add_argument("--source-manifest", type=Path, required=True)
    reduce.add_argument("--label-root", type=Path, action="append", required=True)
    reduce.add_argument("--output", type=Path, required=True)
    reduce.add_argument("--normalization-floor", type=float, default=0.01)
    reduce.add_argument("--bootstrap-resamples", type=int, default=10_000)
    reduce.add_argument("--bootstrap-seed", type=int, default=20260721)
    reduce.add_argument("--bootstrap-interval", type=float, default=0.95)
    args = parser.parse_args()
    shared = {
        "config_path": args.config.resolve(),
        "selection_path": args.selection.resolve(),
        "selection_seal_path": args.selection_seal.resolve(),
    }
    if args.mode == "schedule":
        result = materialize_budget_deferral_truth_schedule(
            **shared,
            output_root=args.output_root.resolve(),
            workers=args.workers,
        )
    else:
        result = reduce_budget_deferral_truth(
            **shared,
            schedule_root=args.schedule_root.resolve(),
            source_manifest_path=args.source_manifest.resolve(),
            label_roots=tuple(path.resolve() for path in args.label_root),
            output_path=args.output.resolve(),
            normalization_floor=args.normalization_floor,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_interval=args.bootstrap_interval,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
