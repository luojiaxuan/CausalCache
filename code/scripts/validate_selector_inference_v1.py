#!/usr/bin/env python3
"""Fail-closed validation for Stage-1/Stage-2 selector inference."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.train_hgkv_singleton_selector_v1 import (
    atomic_write_json,
    sha256_file,
)


def validate(
    path: Path,
    *,
    budgets: list[int],
    methods: list[str],
) -> dict[str, Any]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    states = set()
    candidates_by_state: dict[str, tuple[int, ...]] = {}
    selected_size_counts: dict[str, int] = defaultdict(int)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pair_group = str(row["pair_group"])
            method = str(row["method"])
            budget = int(row["budget"])
            key = (pair_group, method, budget)
            if key in rows:
                raise ValueError(f"duplicate selector inference row: {key}")
            if method not in methods or budget not in budgets:
                raise ValueError(f"unexpected selector inference row: {key}")
            candidates = tuple(int(value) for value in row["candidate_event_step_ids"])
            previous_candidates = candidates_by_state.setdefault(
                pair_group, candidates
            )
            if previous_candidates != candidates:
                raise ValueError(f"candidate inventory drift: {pair_group}")
            selected = tuple(
                int(value) for value in row["selected_event_step_ids"]
            )
            if len(selected) != len(set(selected)):
                raise ValueError(f"duplicate selected event: {key}")
            if len(selected) > budget or not set(selected).issubset(candidates):
                raise ValueError(f"invalid selected set: {key}")
            rows[key] = row
            states.add(pair_group)
            selected_size_counts[f"{method}:B{budget}:K{len(selected)}"] += 1

    expected = {
        (pair_group, method, budget)
        for pair_group in states
        for method in methods
        for budget in budgets
    }
    if set(rows) != expected:
        raise ValueError(
            f"selector inference coverage mismatch: observed={len(rows)} "
            f"expected={len(expected)}"
        )
    if 1 in budgets and {
        "hgkv_singleton",
        "hgkv_set_conditioned",
    }.issubset(methods):
        for pair_group in states:
            singleton = rows[
                (pair_group, "hgkv_singleton", 1)
            ]["selected_event_step_ids"]
            conditioned = rows[
                (pair_group, "hgkv_set_conditioned", 1)
            ]["selected_event_step_ids"]
            if singleton != conditioned:
                raise ValueError(f"B1 parity drift: {pair_group}")
    return {
        "b1_parity_states": len(states) if 1 in budgets else None,
        "budgets": budgets,
        "methods": methods,
        "row_count": len(rows),
        "selected_size_counts": dict(sorted(selected_size_counts.items())),
        "state_count": len(states),
        "status": "PASS_SELECTOR_INFERENCE_V1",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["hgkv_singleton", "hgkv_set_conditioned"],
    )
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    summary = validate(
        args.selections,
        budgets=args.budgets,
        methods=args.methods,
    )
    summary.update(
        {
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "selections_sha256": sha256_file(args.selections),
            "source_commit": args.source_commit,
        }
    )
    atomic_write_json(args.output, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

