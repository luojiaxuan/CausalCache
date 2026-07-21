#!/usr/bin/env python3
"""Merge disjoint tune-selector shards into one bound selection payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_inference import canonical_json_bytes


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("merged tune selection output already exists")
    if len(args.input) < 2:
        raise ValueError("at least two tune selection shards are required")

    shards = [_read(path) for path in args.input]
    identity_keys = (
        "cache_content_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "input_content_sha256",
        "role",
        "schema_version",
        "status",
        "variant",
    )
    identity = {key: shards[0].get(key) for key in identity_keys}
    if identity["status"] != "COMPLETED_SET_UTILITY_TUNE_SELECTIONS":
        raise ValueError("a tune selection shard is incomplete")

    records: dict[str, dict[str, Any]] = {}
    for shard in shards:
        if {key: shard.get(key) for key in identity_keys} != identity:
            raise ValueError("tune selection shard identity drifted")
        expected_sha = hashlib.sha256(
            canonical_json_bytes(
                {key: value for key, value in shard.items() if key != "content_sha256"}
            )
        ).hexdigest()
        if shard.get("content_sha256") != expected_sha:
            raise ValueError("tune selection shard content hash drifted")
        for record in shard.get("records", ()):
            state_id = record["state_id"]
            if state_id in records:
                raise ValueError(f"duplicate tune state across shards: {state_id}")
            records[state_id] = record
    if not records:
        raise ValueError("tune selection shards contain no records")

    result = {**identity, "records": [records[key] for key in sorted(records)]}
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result) + b"\n")
    print(
        json.dumps(
            {
                "content_sha256": result["content_sha256"],
                "state_count": len(records),
                "status": result["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
