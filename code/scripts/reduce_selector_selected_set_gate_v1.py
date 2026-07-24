#!/usr/bin/env python3
"""Reduce true selected-set utilities for the frozen selector gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_odyssey_sft_dataset import restored_set_key
from scripts.train_hgkv_singleton_selector_v1 import (
    atomic_write_json,
    load_b0,
    sha256_file,
)


def load_selected_scores(paths: list[Path]) -> dict[tuple[str, str], float]:
    scores = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["variant"] != "selected_set":
                    raise ValueError(f"unexpected selected-set variant in {path}")
                key = (str(row["pair_group"]), str(row["restored_set_key"]))
                value = float(row["target_logprob_mean"])
                if key in scores:
                    raise ValueError(f"duplicate selected-set score: {key}")
                if not np.isfinite(value):
                    raise ValueError(f"non-finite selected-set score: {key}")
                scores[key] = value
    return scores


def load_plan(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["pair_group"], row["method"], int(row["budget"]))
            if key in seen:
                raise ValueError(f"duplicate gate-plan row: {key}")
            seen.add(key)
            rows.append(row)
    return rows


def cluster_bootstrap_ci(
    deltas: dict[str, list[float]],
    *,
    seed: int,
    replicates: int,
) -> list[float]:
    episodes = sorted(deltas)
    episode_values = np.asarray(
        [np.mean(deltas[episode]) for episode in episodes], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    chunk = 1000
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        indices = rng.integers(
            0, len(episode_values), size=(size, len(episode_values))
        )
        means[start : start + size] = episode_values[indices].mean(axis=1)
    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def reduce_gate(
    plan_rows: list[dict[str, Any]],
    *,
    selected_scores: dict[tuple[str, str], float],
    b0: dict[str, float],
    seed: int,
    bootstrap_replicates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_state = []
    by_budget_method: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in plan_rows:
        pair_group = str(row["pair_group"])
        if pair_group not in b0:
            raise ValueError(f"B0 join miss: {pair_group}")
        selected = tuple(int(value) for value in row["selected_event_step_ids"])
        key = (pair_group, restored_set_key(selected))
        if key not in selected_scores:
            raise ValueError(f"selected-set score join miss: {key}")
        result = {
            "budget": int(row["budget"]),
            "episode": str(row["episode"]),
            "method": str(row["method"]),
            "pair_group": pair_group,
            "selected_event_step_ids": list(selected),
            "selected_size": len(selected),
            "target_logprob_mean": selected_scores[key],
            "u_act": selected_scores[key] - b0[pair_group],
        }
        per_state.append(result)
        by_budget_method[(result["budget"], result["method"])].append(result)

    budgets = sorted({row["budget"] for row in per_state})
    methods = sorted({row["method"] for row in per_state})
    aggregate: dict[str, Any] = {}
    for budget in budgets:
        aggregate[f"B{budget}"] = {}
        expected_states = None
        for method in methods:
            rows = by_budget_method[(budget, method)]
            if expected_states is None:
                expected_states = {row["pair_group"] for row in rows}
            elif {row["pair_group"] for row in rows} != expected_states:
                raise ValueError(f"state coverage mismatch at B{budget}")
            episode_means: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                episode_means[row["episode"]].append(row["u_act"])
            aggregate[f"B{budget}"][method] = {
                "episode_macro_mean_u": float(
                    np.mean(
                        [
                            np.mean(values)
                            for values in episode_means.values()
                        ]
                    )
                ),
                "mean_selected_size": float(
                    np.mean([row["selected_size"] for row in rows])
                ),
                "n_episodes": len(episode_means),
                "n_states": len(rows),
                "state_macro_mean_u": float(
                    np.mean([row["u_act"] for row in rows])
                ),
                "stop_rate": float(
                    np.mean([row["selected_size"] < budget for row in rows])
                ),
            }

    comparators = ("random", "recent", "similarity")
    comparisons = {}
    gate_pass = True
    lookup = {
        (row["pair_group"], row["method"], row["budget"]): row
        for row in per_state
    }
    for budget in budgets:
        for comparator_index, comparator in enumerate(comparators):
            deltas_by_episode: dict[str, list[float]] = defaultdict(list)
            for row in by_budget_method[(budget, "hgkv_set_conditioned")]:
                comparator_row = lookup[
                    (row["pair_group"], comparator, budget)
                ]
                deltas_by_episode[row["episode"]].append(
                    row["u_act"] - comparator_row["u_act"]
                )
            episode_deltas = [
                float(np.mean(values)) for values in deltas_by_episode.values()
            ]
            ci = cluster_bootstrap_ci(
                deltas_by_episode,
                seed=seed + 10 * budget + comparator_index,
                replicates=bootstrap_replicates,
            )
            key = f"B{budget}:hgkv_set_conditioned-minus-{comparator}"
            comparisons[key] = {
                "ci95": ci,
                "episode_macro_mean": float(np.mean(episode_deltas)),
                "n_episodes": len(episode_deltas),
            }
            gate_pass = gate_pass and ci[0] > 0
    summary = {
        "aggregate": aggregate,
        "bootstrap_replicates": bootstrap_replicates,
        "comparisons": comparisons,
        "gate_rule": (
            "For every B in {1,2,4}, the episode-cluster bootstrap 95% "
            "lower bound of hgkv_set_conditioned minus each of "
            "{Recent, Similarity, Random} must be > 0."
        ),
        "methods": methods,
        "status": (
            "PASS_SELECTED_SET_GATE_V1"
            if gate_pass
            else "NO_GO_SELECTED_SET_GATE_V1"
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

    plan_rows = load_plan(args.plan)
    selected_scores = load_selected_scores(args.selected_scores)
    b0 = load_b0(args.b0_scores)
    per_state, summary = reduce_gate(
        plan_rows,
        selected_scores=selected_scores,
        b0=b0,
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
            "b0_scores_sha256": sha256_file(args.b0_scores),
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "per_state_rows": len(per_state),
            "plan_sha256": sha256_file(args.plan),
            "selected_score_files": {
                str(path): sha256_file(path) for path in args.selected_scores
            },
            "source_commit": args.source_commit,
        }
    )
    atomic_write_json(args.output_dir / "aggregate.json", summary)
    atomic_write_json(
        args.output_dir / "DONE",
        {
            "completed_at": summary["completed_at"],
            "per_state_rows": len(per_state),
            "source_commit": args.source_commit,
            "status": summary["status"],
        },
    )


if __name__ == "__main__":
    main()
