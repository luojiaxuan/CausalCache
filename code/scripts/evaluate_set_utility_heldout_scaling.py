#!/usr/bin/env python3
"""Evaluate learning-curve checkpoints on existing exact held-out truth."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from causalcache.set_utility_heldout_evaluation import (
    canonical_json_bytes,
    load_label_terminals,
    sha256_file,
)
from causalcache.set_utility_heldout_scaling import evaluate_scaling_exact_track


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", action="append", required=True)
    parser.add_argument("--label-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selections = {}
    for raw in args.selection:
        name, separator, path = raw.partition("=")
        if not separator or not name or not path or name in selections:
            raise ValueError("selection must be a unique MODEL=PATH binding")
        selections[name] = json.loads(Path(path).read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = evaluate_scaling_exact_track(
        config=config,
        config_sha256=sha256_file(args.config),
        selections=selections,
        terminals=load_label_terminals(args.label_root),
    )
    result["bindings"] = {
        "config_sha256": sha256_file(args.config),
        "label_roots": [str(path.resolve()) for path in args.label_root],
        "selection_files": {
            name: sha256_file(Path(raw.partition("=")[2]))
            for name, raw in zip(selections, args.selection, strict=True)
        },
    }
    unsigned = dict(result)
    unsigned.pop("content_sha256")
    import hashlib

    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    payload = canonical_json_bytes(result, pretty=True)
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
                "output": str(args.output),
                "status": result["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
