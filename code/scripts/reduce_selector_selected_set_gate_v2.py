#!/usr/bin/env python3
"""Reduce the formal true selected-set utility gate for selector V2."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from causalcache.hgkv_selector_v2 import restored_set_key
from scripts.build_hgkv_selector_v2_training_data import sha256_file
from scripts.train_hgkv_singleton_selector_v1 import (
    atomic_write_json,
    load_b0,
)


PRIMARY_METHOD = "hgkv_v2_beam4"


def load_selected_scores(
    paths: Iterable[Path],
) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["variant"] != "selected_set":
                raise ValueError(f"{path}:{line_no} unexpected variant")
            key = (str(row["pair_group"]), str(row["restored_set_key"]))
            value = float(row["target_logprob_mean"])
            if key in scores:
                raise ValueError(f"{path}:{line_no} duplicate score {key}")
            if not np.isfinite(value):
                raise ValueError(f"{path}:{line_no} non-finite score")
            scores[key] = value
    return scores


def load_plan(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (
            str(row["pair_group"]),
            str(row["method"]),
            int(row["budget"]),
        )
        if key in seen:
            raise ValueError(f"{path}:{line_no} duplicate plan row {key}")
        seen.add(key)
        rows.append(row)
    return rows


def cluster_bootstrap_ci(
    episode_values: dict[str, float],
    *,
    seed: int,
    replicates: int,
) -> list[float]:
    values = np.asarray(
        [episode_values[key] for key in sorted(episode_values)],
        dtype=np.float64,
    )
    if not len(values):
        raise ValueError("bootstrap comparison has no episodes")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        size = min(1000, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start : start + size] = values[indices].mean(axis=1)
    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def episode_method_values(
    rows: Iterable[dict[str, Any]],
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values[str(row["episode"])].append(float(row["u_act"]))
    return {
        episode: float(np.mean(episode_rows))
        for episode, episode_rows in values.items()
    }


def paired_comparison(
    left: Iterable[dict[str, Any]],
    right: Iterable[dict[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    left_values = episode_method_values(left)
    right_values = episode_method_values(right)
    if set(left_values) != set(right_values):
        raise ValueError("paired comparison episode coverage differs")
    deltas = {
        episode: left_values[episode] - right_values[episode]
        for episode in left_values
    }
    return {
        "episode_macro_mean": float(np.mean(list(deltas.values()))),
        "ci95": cluster_bootstrap_ci(
            deltas, seed=seed, replicates=replicates
        ),
        "n_episodes": len(deltas),
    }


def reduce_gate(
    plan_rows: list[dict[str, Any]],
    *,
    selected_scores: dict[tuple[str, str], float],
    b0: dict[str, float],
    seed: int,
    bootstrap_replicates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_state: list[dict[str, Any]] = []
    for row in plan_rows:
        pair_group = str(row["pair_group"])
        if pair_group not in b0:
            raise ValueError(f"B0 exact join miss {pair_group}")
        selected = tuple(int(value) for value in row["selected_event_step_ids"])
        score_key = (pair_group, restored_set_key(selected))
        score = selected_scores.get(score_key)
        if score is None:
            raise ValueError(f"selected-set exact join miss {score_key}")
        per_state.append(
            {
                "episode": str(row["episode"]),
                "pair_group": pair_group,
                "method": str(row["method"]),
                "budget": int(row["budget"]),
                "selected_event_step_ids": list(selected),
                "selected_size": len(selected),
                "target_logprob_mean": score,
                "b0_target_logprob_mean": b0[pair_group],
                "u_act": score - b0[pair_group],
            }
        )
    by_budget_method: dict[
        tuple[int, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in per_state:
        by_budget_method[(row["budget"], row["method"])].append(row)
    budgets = sorted({row["budget"] for row in per_state})
    methods = sorted({row["method"] for row in per_state})
    aggregate: dict[str, Any] = {}
    for budget in budgets:
        aggregate[f"B{budget}"] = {}
        primary_states = {
            row["pair_group"]
            for row in by_budget_method[(budget, PRIMARY_METHOD)]
        }
        if not primary_states:
            raise ValueError(f"primary method missing at B{budget}")
        for method in methods:
            rows = by_budget_method[(budget, method)]
            if not rows:
                continue
            states = {row["pair_group"] for row in rows}
            subset = states != primary_states
            episode_values = episode_method_values(rows)
            aggregate[f"B{budget}"][method] = {
                "episode_macro_mean_u": float(
                    np.mean(list(episode_values.values()))
                ),
                "mean_selected_size": float(
                    np.mean([row["selected_size"] for row in rows])
                ),
                "stop_rate": float(
                    np.mean([row["selected_size"] < budget for row in rows])
                ),
                "n_episodes": len(episode_values),
                "n_states": len(rows),
                "subset_only": subset,
            }

    lookup = {
        (row["pair_group"], row["method"], row["budget"]): row
        for row in per_state
    }
    b1_mismatches = 0
    if "hgkv_v2_greedy" in methods:
        for pair_group in {
            row["pair_group"]
            for row in by_budget_method[(1, PRIMARY_METHOD)]
        }:
            left = lookup[(pair_group, PRIMARY_METHOD, 1)]
            right = lookup[(pair_group, "hgkv_v2_greedy", 1)]
            if (
                left["selected_event_step_ids"]
                != right["selected_event_step_ids"]
                or abs(left["u_act"] - right["u_act"]) > 1e-10
            ):
                b1_mismatches += 1
    if b1_mismatches:
        raise ValueError(f"V2 greedy/beam B1 parity failed {b1_mismatches}")

    comparisons: dict[str, Any] = {}
    formal_pass = True
    comparator_seed = 0
    for budget in budgets:
        primary_rows = by_budget_method[(budget, PRIMARY_METHOD)]
        for comparator in ("recent", "similarity", "random"):
            comparator_rows = by_budget_method[(budget, comparator)]
            comparison = paired_comparison(
                primary_rows,
                comparator_rows,
                seed=seed + comparator_seed,
                replicates=bootstrap_replicates,
            )
            comparator_seed += 1
            key = f"B{budget}:{PRIMARY_METHOD}-minus-{comparator}"
            comparisons[key] = comparison
            if budget == 1 and comparator == "recent":
                formal_pass = formal_pass and comparison["ci95"][1] >= 0.0
            if budget in (2, 4):
                formal_pass = formal_pass and comparison["ci95"][0] > 0.0

    for comparator in (
        "hgkv_v2_greedy",
        "hgkv_v1_singleton",
        "teacher_beam4",
        "exact_oracle",
    ):
        if comparator not in methods:
            continue
        for budget in budgets:
            comparator_rows = by_budget_method[(budget, comparator)]
            if not comparator_rows:
                continue
            primary_by_state = {
                row["pair_group"]: row
                for row in by_budget_method[(budget, PRIMARY_METHOD)]
            }
            shared_primary = [
                primary_by_state[row["pair_group"]]
                for row in comparator_rows
                if row["pair_group"] in primary_by_state
            ]
            shared_comparator = [
                row
                for row in comparator_rows
                if row["pair_group"] in primary_by_state
            ]
            comparisons[
                f"B{budget}:{PRIMARY_METHOD}-minus-{comparator}"
            ] = paired_comparison(
                shared_primary,
                shared_comparator,
                seed=seed + comparator_seed,
                replicates=bootstrap_replicates,
            )
            comparator_seed += 1

    average_deltas: dict[str, list[float]] = defaultdict(list)
    for row in per_state:
        if row["method"] != PRIMARY_METHOD:
            continue
        recent = lookup[(row["pair_group"], "recent", row["budget"])]
        average_deltas[row["episode"]].append(row["u_act"] - recent["u_act"])
    avg_episode = {
        episode: float(np.mean(values))
        for episode, values in average_deltas.items()
    }
    average_comparison = {
        "episode_macro_mean": float(np.mean(list(avg_episode.values()))),
        "ci95": cluster_bootstrap_ci(
            avg_episode,
            seed=seed + comparator_seed,
            replicates=bootstrap_replicates,
        ),
        "n_episodes": len(avg_episode),
    }
    comparisons["AvgB1B2B4:hgkv_v2_beam4-minus-recent"] = (
        average_comparison
    )
    formal_pass = formal_pass and average_comparison["ci95"][0] > 0.0
    summary = {
        "schema_version": (
            "causalcache.hgkv_selector_v2.selected_set_gate.v1"
        ),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "bootstrap_replicates": bootstrap_replicates,
        "b1_parity_mismatches": 0,
        "formal_gate": {
            "B1": "beam4-minus-recent CI upper >= 0",
            "B2": "beam4-minus-{recent,similarity,random} CI lower > 0",
            "B4": "beam4-minus-{recent,similarity,random} CI lower > 0",
            "average": "Avg(B1,B2,B4) beam4-minus-recent CI lower > 0",
        },
        "formal_utility_source": (
            "rerendered_hg_s100_complete_set_exact_join_frozen_b0"
        ),
        "methods": methods,
        "status": (
            "PASS_SELECTED_SET_GATE_V2"
            if formal_pass
            else "NO_GO_SELECTED_SET_GATE_V2"
        ),
    }
    return per_state, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--selected-scores", type=Path, nargs="+", required=True)
    parser.add_argument("--b0-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    per_state, summary = reduce_gate(
        load_plan(args.plan),
        selected_scores=load_selected_scores(args.selected_scores),
        b0=load_b0(args.b0_scores),
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_state_path = args.output_dir / "per-state.jsonl"
    with per_state_path.open("w", encoding="utf-8") as handle:
        for row in per_state:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary.update(
        {
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "per_state_rows": len(per_state),
            "plan_sha256": sha256_file(args.plan),
            "b0_scores_sha256": sha256_file(args.b0_scores),
            "selected_score_sha256": {
                str(path): sha256_file(path)
                for path in args.selected_scores
            },
        }
    )
    atomic_write_json(args.output_dir / "aggregate.json", summary)
    atomic_write_json(
        args.output_dir / "DONE",
        {
            "status": summary["status"],
            "source_commit": args.source_commit,
            "per_state_rows": len(per_state),
            "aggregate_sha256": sha256_file(
                args.output_dir / "aggregate.json"
            ),
        },
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
