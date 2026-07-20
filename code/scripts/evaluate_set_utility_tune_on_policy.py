#!/usr/bin/env python3
"""Evaluate contextual tune selectors against targeted restoration truth."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_evaluation import canonical_json_bytes
from causalcache.set_utility_tune_on_policy_evaluation import evaluate_tune_on_policy


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or re.fullmatch(r"[a-z0-9_]+", name) is None or not raw_path:
        raise argparse.ArgumentTypeError("selection must be name=path")
    return name, Path(raw_path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_atomic(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value, pretty=True) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=_named_path, action="append", required=True)
    parser.add_argument("--label-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalization-floor", type=float, default=0.01)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    parser.add_argument("--bootstrap-interval", type=float, default=0.95)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("tune on-policy evaluation output already exists")
    selections = {name: _read_json(path) for name, path in args.selection}
    if len(selections) != len(args.selection):
        raise ValueError("selection names must be unique")
    terminals = {}
    for root in args.label_root:
        for path in sorted((root / "states").glob("*.json")):
            terminal = _read_json(path)
            state_id = terminal.get("state_id")
            if not isinstance(state_id, str) or state_id in terminals:
                raise ValueError("label terminal identity is missing or duplicated")
            terminals[state_id] = terminal
    result = evaluate_tune_on_policy(
        selections=selections,
        terminals=terminals,
        normalization_floor=args.normalization_floor,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_interval=args.bootstrap_interval,
    )
    _write_atomic(args.output, result)
    print(
        json.dumps(
            {
                "completed_state_count": result["coverage"]["completed_state_count"],
                "content_sha256": result["content_sha256"],
                "status": result["status"],
                "winner": result["winner"]["model"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
