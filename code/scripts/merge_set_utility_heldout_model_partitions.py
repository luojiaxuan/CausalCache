#!/usr/bin/env python3
"""Merge one model's disjoint host-local held-out selection partitions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from causalcache.set_utility_heldout_inference import (
    canonical_json_bytes,
    merge_model_selection_partitions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--full-input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("merged model selection output already exists")
    manifest = json.loads(
        (args.full_input_root / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "COMPLETED_SET_UTILITY_HELDOUT_FEATURE_SNAPSHOT":
        raise ValueError("full held-out input snapshot is incomplete")
    state_ids = tuple(
        json.loads(line)["state_id"]
        for line in (args.full_input_root / manifest["states_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    result = merge_model_selection_partitions(
        payloads,
        full_input_content_sha256=manifest["content_sha256"],
        expected_state_ids=state_ids,
    )
    payload = canonical_json_bytes(result) + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "content_sha256": result["content_sha256"],
                "model_name": result["model_name"],
                "record_count": len(result["records"]),
                "status": result["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
