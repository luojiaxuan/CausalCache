"""Summarize immutable AndroidWorld trace shards without policy execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from causalcache.long_horizon_incidence import summarize_gzip_trace_shard


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-shard", type=Path, required=True)
    parser.add_argument("--artifact-label", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-record-count", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = args.trace_shard.read_bytes()
    summary = summarize_gzip_trace_shard(
        payload,
        artifact_label=args.artifact_label,
    )
    if summary["trace_shard_sha256"] != args.expected_sha256:
        raise ValueError("trace shard SHA256 differs from its immutable binding")
    if summary["record_count"] != args.expected_record_count:
        raise ValueError("trace shard record count differs from its immutable binding")
    serialized = json.dumps(
        summary,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
