#!/usr/bin/env python3
"""Plan the frozen n<=8 development exact-search validation coalitions."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

from causalcache.hgkv_selector_v2 import restored_set_key
from scripts.plan_hgkv_teacher_beam_v2 import (
    expand_paths,
    load_cache,
    load_states,
    sha256_file,
)


def build_exact_plan_rows(
    *,
    states: Mapping[str, Mapping[str, Any]],
    cache: Mapping[tuple[str, str], Mapping[str, Any]],
    max_candidates: int = 8,
    max_budget: int = 4,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_group, state in sorted(states.items()):
        candidates = tuple(
            int(value) for value in state["candidate_event_step_ids"]
        )
        if state.get("split") != "dev" or len(candidates) > max_candidates:
            continue
        for size in range(1, min(max_budget, len(candidates)) + 1):
            for coalition in combinations(candidates, size):
                set_key = restored_set_key(coalition)
                cached = cache.get((pair_group, set_key))
                rows.append(
                    {
                        "schema_version": (
                            "causalcache.hgkv_selector_v2."
                            "exact_search_plan.v1"
                        ),
                        "episode": str(state["episode"]),
                        "pair_group": pair_group,
                        "decision_step": int(state["decision_step"]),
                        "history_length": int(state["history_length"]),
                        "candidate_event_step_ids": list(candidates),
                        "selected_event_step_ids": list(coalition[:-1]),
                        "candidate_event_step_id": int(coalition[-1]),
                        "restored_event_step_ids": list(coalition),
                        "restored_set_key": set_key,
                        "depth": size - 1,
                        "edge": "exact_search",
                        "cache_hit": cached is not None,
                        "cached_u_act": (
                            None if cached is None else float(cached["u_act"])
                        ),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", action="append", required=True)
    parser.add_argument("--coalition-cache", action="append", required=True)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-budget", type=int, default=4)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.max_candidates != 8 or args.max_budget != 4:
        parser.error("the frozen V2 exact-search limits are n<=8 and B<=4")
    state_paths = expand_paths(args.states)
    cache_paths = expand_paths(args.coalition_cache)
    if not state_paths or not cache_paths:
        parser.error("state and coalition-cache patterns must match files")
    rows = build_exact_plan_rows(
        states=load_states(state_paths),
        cache=load_cache(cache_paths),
        max_candidates=args.max_candidates,
        max_budget=args.max_budget,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    unique_states = {str(row["pair_group"]) for row in rows}
    missing = sum(not bool(row["cache_hit"]) for row in rows)
    manifest = {
        "schema_version": (
            "causalcache.hgkv_selector_v2.exact_search_plan_manifest.v1"
        ),
        "split": "dev",
        "max_candidates": args.max_candidates,
        "max_budget": args.max_budget,
        "states": len(unique_states),
        "coalitions": len(rows),
        "cached_coalitions": len(rows) - missing,
        "missing_coalitions": missing,
        "source_commit": args.source_commit,
        "state_sha256": {
            str(path): sha256_file(path) for path in state_paths
        },
        "cache_sha256": {
            str(path): sha256_file(path) for path in cache_paths
        },
        "output_sha256": sha256_file(args.output),
        "status": "DONE",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
