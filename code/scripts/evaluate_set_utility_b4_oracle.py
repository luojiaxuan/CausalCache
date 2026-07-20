#!/usr/bin/env python3
"""Reduce self-contained exact-B4 labels against frozen selector choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.set_utility_b4_oracle_evaluation import evaluate_b4_oracle
from causalcache.set_utility_heldout_evaluation import (
    canonical_json_bytes,
    load_label_terminals,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, action="append", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selections = json.loads(args.selections.read_text(encoding="utf-8"))
    if sha256_file(args.selections) != config["inputs"]["sealed_selections"][
        "file_sha256"
    ]:
        raise ValueError("B4 selection file drifted")
    result = evaluate_b4_oracle(
        config=config,
        config_sha256=sha256_file(args.config),
        execution_config_sha256=sha256_file(args.execution_config),
        selections=selections,
        terminals=load_label_terminals(args.label_root),
        source_revision=args.source_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result, pretty=True))
    print(json.dumps({"content_sha256": result["content_sha256"], "status": result["status"]}))


if __name__ == "__main__":
    main()
