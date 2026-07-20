#!/usr/bin/env python3
"""Merge model-specific held-out selections before restoration labels open."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from causalcache.set_utility_heldout_inference import (
    canonical_json_bytes,
    merge_model_selection_payloads,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("merged held-out selection output already exists")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    result = merge_model_selection_payloads(payloads)
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
                "record_count": len(result["records"]),
                "status": result["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
