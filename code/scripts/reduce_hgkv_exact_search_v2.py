#!/usr/bin/env python3
"""Reduce n<=8 exact-search scores and compare the true-U teacher beam."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

from causalcache.hgkv_selector_v2 import restored_set_key
from scripts.build_selector_gate_plan_v2 import teacher_selections
from scripts.plan_hgkv_teacher_beam_v2 import (
    expand_paths,
    load_cache,
    sha256_file,
)


BUDGETS = (1, 2, 4)


def load_exact_plan(
    paths: Iterable[Path],
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    observed: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != (
                "causalcache.hgkv_selector_v2.exact_search_plan.v1"
            ):
                raise ValueError(f"{path}:{line_no} invalid exact plan schema")
            pair_group = str(row["pair_group"])
            state = {
                "episode": str(row["episode"]),
                "pair_group": pair_group,
                "candidate_event_step_ids": [
                    int(value) for value in row["candidate_event_step_ids"]
                ],
            }
            previous = states.get(pair_group)
            if previous is not None and previous != state:
                raise ValueError(f"{path}:{line_no} conflicting state")
            states[pair_group] = state
            coalition = tuple(
                int(value) for value in row["restored_event_step_ids"]
            )
            if coalition in observed[pair_group]:
                raise ValueError(f"{path}:{line_no} duplicate coalition")
            observed[pair_group].add(coalition)
    for pair_group, state in states.items():
        candidates = tuple(state["candidate_event_step_ids"])
        expected = {
            coalition
            for size in range(1, min(4, len(candidates)) + 1)
            for coalition in combinations(candidates, size)
        }
        if observed[pair_group] != expected:
            raise ValueError(
                f"{pair_group} exact plan coverage differs by "
                f"{len(observed[pair_group] ^ expected)} coalitions"
            )
    return states


def _best_at_most(
    *,
    pair_group: str,
    candidates: tuple[int, ...],
    budget: int,
    cache: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[tuple[int, ...], float]:
    eligible = [
        coalition
        for size in range(0, min(budget, len(candidates)) + 1)
        for coalition in combinations(candidates, size)
    ]
    missing = [
        coalition
        for coalition in eligible
        if (pair_group, restored_set_key(coalition)) not in cache
    ]
    if missing:
        raise KeyError(
            f"{pair_group} B{budget} misses {len(missing)} exact utilities"
        )
    selected = min(
        eligible,
        key=lambda coalition: (
            -float(cache[(pair_group, restored_set_key(coalition))]["u_act"]),
            coalition,
        ),
    )
    return (
        selected,
        float(cache[(pair_group, restored_set_key(selected))]["u_act"]),
    )


def reduce_exact_search(
    *,
    states: Mapping[str, Mapping[str, Any]],
    cache: Mapping[tuple[str, str], Mapping[str, Any]],
    teacher: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pair_group, state in sorted(states.items()):
        candidates = tuple(
            int(value) for value in state["candidate_event_step_ids"]
        )
        for budget in BUDGETS:
            exact_set, exact_u = _best_at_most(
                pair_group=pair_group,
                candidates=candidates,
                budget=budget,
                cache=cache,
            )
            teacher_set = tuple(
                int(value)
                for value in teacher[
                    (pair_group, "teacher_beam4", budget)
                ]["selected_event_step_ids"]
            )
            teacher_u = float(
                cache[(pair_group, restored_set_key(teacher_set))]["u_act"]
            )
            union = set(exact_set) | set(teacher_set)
            jaccard = (
                1.0
                if not union
                else len(set(exact_set) & set(teacher_set)) / len(union)
            )
            diagnostic = {
                "pair_group": pair_group,
                "budget": budget,
                "recovered": teacher_set == exact_set,
                "regret": exact_u - teacher_u,
                "utility_gap_teacher_minus_exact": teacher_u - exact_u,
                "jaccard": jaccard,
            }
            if diagnostic["regret"] < -1e-8:
                raise ValueError("teacher utility cannot exceed the exact oracle")
            by_budget[budget].append(diagnostic)
            selections.append(
                {
                    "schema_version": (
                        "causalcache.hgkv_selector_v2."
                        "exact_oracle_selection.v1"
                    ),
                    "episode": str(state["episode"]),
                    "pair_group": pair_group,
                    "method": "exact_oracle",
                    "budget": budget,
                    "selected_event_step_ids": list(exact_set),
                    "true_u_act": exact_u,
                    "teacher_beam4_selected_event_step_ids": list(teacher_set),
                    "teacher_beam4_true_u_act": teacher_u,
                    **diagnostic,
                }
            )
    summary: dict[str, Any] = {
        "schema_version": (
            "causalcache.hgkv_selector_v2.exact_search_diagnostic.v1"
        ),
        "states": len(states),
        "budgets": {},
    }
    for budget, rows in sorted(by_budget.items()):
        summary["budgets"][f"B{budget}"] = {
            "states": len(rows),
            "teacher_beam4_oracle_recovery": (
                sum(bool(row["recovered"]) for row in rows) / len(rows)
            ),
            "teacher_beam4_mean_regret": math.fsum(
                float(row["regret"]) for row in rows
            )
            / len(rows),
            "teacher_beam4_mean_utility_gap": math.fsum(
                float(row["utility_gap_teacher_minus_exact"]) for row in rows
            )
            / len(rows),
            "teacher_beam4_mean_top_b_jaccard": math.fsum(
                float(row["jaccard"]) for row in rows
            )
            / len(rows),
        }
    return selections, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="append", required=True)
    parser.add_argument("--coalition-cache", action="append", required=True)
    parser.add_argument("--teacher-run-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--selections-output", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    plan_paths = expand_paths(args.plan)
    cache_paths = expand_paths(args.coalition_cache)
    if not plan_paths or not cache_paths:
        parser.error("plan and coalition-cache patterns must match files")
    states = load_exact_plan(plan_paths)
    cache = load_cache(cache_paths)
    teacher = teacher_selections(
        states=states,
        teacher_run_root=args.teacher_run_root,
        cache=cache,
    )
    selections, diagnostic = reduce_exact_search(
        states=states,
        cache=cache,
        teacher=teacher,
    )
    args.selections_output.parent.mkdir(parents=True, exist_ok=True)
    with args.selections_output.open("w", encoding="utf-8") as handle:
        for row in selections:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    args.diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostic_output.write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": (
            "causalcache.hgkv_selector_v2.exact_search_reduce_manifest.v1"
        ),
        "states": len(states),
        "selection_rows": len(selections),
        "source_commit": args.source_commit,
        "plan_sha256": {
            str(path): sha256_file(path) for path in plan_paths
        },
        "cache_sha256": {
            str(path): sha256_file(path) for path in cache_paths
        },
        "selections_sha256": sha256_file(args.selections_output),
        "diagnostic_sha256": sha256_file(args.diagnostic_output),
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
