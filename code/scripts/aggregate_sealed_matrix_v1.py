#!/usr/bin/env python3
"""Aggregate the frozen 116-template AndroidWorld zero-shot matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from causalcache.sealed_matrix_v1 import (
    aggregate_matrix,
    load_attempts,
    load_roster,
    sha256_file,
)


def parse_input(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("--input must use LABEL=/absolute/path")
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("matrix input paths must be absolute")
    return label, path


def atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_status_markers(
    output_dir: Path,
    *,
    report: dict,
    source_commit: str,
) -> None:
    payload = {
        "completed_at": report["completed_at"],
        "formal_cell_count": report["audit"]["formal_cell_count"],
        "source_commit": source_commit,
        "status": report["status"],
    }
    atomic_write_json(output_dir / "STATUS.json", payload)
    done_path = output_dir / "DONE"
    done_path.unlink(missing_ok=True)
    if report["status"].startswith("COMPLETE_"):
        atomic_write_json(done_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--input", type=parse_input, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    roster = load_roster(args.roster)
    attempts, rejected = load_attempts(args.input, roster=roster)
    report = aggregate_matrix(
        attempts,
        roster=roster,
        rejected=rejected,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    report["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    report["inputs"] = {
        label: str(path) for label, path in args.input
    }
    report["roster"] = {
        "path": str(args.roster),
        "sha256": sha256_file(args.roster),
    }
    report["source_commit"] = args.source_commit

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "aggregate.json", report)
    missing_path = args.output_dir / "missing-cells.jsonl"
    temporary = missing_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in report["audit"]["missing_cells"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(missing_path)
    write_status_markers(
        args.output_dir,
        report=report,
        source_commit=args.source_commit,
    )
    print(
        json.dumps(
            {
                "formal_cell_count": report["audit"]["formal_cell_count"],
                "missing_cell_count": report["audit"]["missing_cell_count"],
                "status": report["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_complete and report["status"].startswith("INCOMPLETE_"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
