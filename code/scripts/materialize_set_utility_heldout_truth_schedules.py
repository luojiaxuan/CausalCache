#!/usr/bin/env python3
"""Materialize one shared runner schedule from heldout model rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_heldout_truth_schedule import (
    materialize_heldout_truth_schedules,
    seal_formal_truth_root,
)


def _named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in result:
            raise ValueError("--truth-schedule requires unique NAME=PATH values")
        result[name] = Path(raw_path).resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--expected-input-content-sha256")
    parser.add_argument("--heldout-manifest", type=Path)
    parser.add_argument("--expected-heldout-content-sha256")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--expected-source-manifest-sha256")
    parser.add_argument(
        "--truth-schedule",
        action="append",
        help="Repeat NAME=PATH for each model-specific signed truth schedule.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--seal-truth-root",
        type=Path,
        help=(
            "Seal this completed label root using --output-root as the already "
            "materialized schedule root."
        ),
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if args.seal_truth_root is not None:
        if args.truth_schedule:
            raise ValueError("sealing does not accept --truth-schedule")
        result = seal_formal_truth_root(
            schedule_root=args.output_root.resolve(),
            truth_root=args.seal_truth_root.resolve(),
        )
        print(json.dumps(result, sort_keys=True))
        return
    required = {
        "--input-root": args.input_root,
        "--expected-input-content-sha256": args.expected_input_content_sha256,
        "--heldout-manifest": args.heldout_manifest,
        "--expected-heldout-content-sha256": args.expected_heldout_content_sha256,
        "--source-manifest": args.source_manifest,
        "--expected-source-manifest-sha256": args.expected_source_manifest_sha256,
        "--truth-schedule": args.truth_schedule,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("materialization requires " + ", ".join(missing))
    result = materialize_heldout_truth_schedules(
        input_root=args.input_root.resolve(),
        expected_input_content_sha256=args.expected_input_content_sha256,
        heldout_manifest_path=args.heldout_manifest.resolve(),
        expected_heldout_content_sha256=args.expected_heldout_content_sha256,
        source_manifest_path=args.source_manifest.resolve(),
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        truth_schedule_paths=_named_paths(args.truth_schedule),
        output_root=args.output_root.resolve(),
        workers=args.workers,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
