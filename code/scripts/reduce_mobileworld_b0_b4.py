#!/usr/bin/env python3
"""Reduce paired MobileWorld B0/B4 trajectories on the frozen strict roster."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


SUCCESS_THRESHOLD = 0.99


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _read_scores(
    roots: list[Path],
    roster: list[str],
) -> dict[str, float]:
    known = set(roster)
    scores: dict[str, float] = {}
    for root in roots:
        for result_path in sorted(root.glob("*/result.txt")):
            task = result_path.parent.name
            if task not in known:
                raise ValueError(f"result for unknown MobileWorld task: {task}")
            for line in result_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("score:"):
                    scores[task] = float(line.split(":", 1)[1].strip())
                    break
            else:
                raise ValueError(f"MobileWorld result lacks a score: {result_path}")
    return scores


def _rate(successes: int, denominator: int) -> dict[str, Any]:
    return {
        "successes": successes,
        "denominator": denominator,
        "rate": successes / denominator if denominator else None,
    }


def _arm_summary(
    roster: list[str],
    records: dict[str, dict[str, Any]],
    scores: dict[str, float],
) -> dict[str, Any]:
    success = {task for task, score in scores.items() if score > SUCCESS_THRESHOLD}

    def summarize_tasks(tasks: list[str]) -> dict[str, Any]:
        observed = [task for task in tasks if task in scores]
        wins = sum(task in success for task in tasks)
        return {
            "observed": _rate(wins, len(observed)),
            "strict_missing_as_zero": _rate(wins, len(tasks)),
            "missing_tasks": [task for task in tasks if task not in scores],
        }

    return {
        "overall": summarize_tasks(roster),
        "by_split": {
            split: summarize_tasks(
                [
                    task
                    for task in roster
                    if records[task]["memory_split"] == split
                ]
            )
            for split in (
                "cross_app_memory_candidate",
                "single_app_control",
            )
        },
    }


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    tail = sum(
        math.comb(discordant, index)
        for index in range(lower + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def _paired_comparison(
    tasks: list[str],
    b4_scores: dict[str, float],
    b0_scores: dict[str, float],
    *,
    missing_as_zero: bool,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if not missing_as_zero:
        tasks = [
            task for task in tasks if task in b4_scores and task in b0_scores
        ]
    b4 = [
        b4_scores.get(task, 0.0) > SUCCESS_THRESHOLD
        for task in tasks
    ]
    b0 = [
        b0_scores.get(task, 0.0) > SUCCESS_THRESHOLD
        for task in tasks
    ]
    b0_only = sum(
        current and not reference
        for reference, current in zip(b4, b0, strict=True)
    )
    b4_only = sum(
        reference and not current
        for reference, current in zip(b4, b0, strict=True)
    )
    diffs = [
        int(current) - int(reference)
        for reference, current in zip(b4, b0, strict=True)
    ]
    rng = random.Random(seed)
    boots = sorted(
        sum(rng.choice(diffs) for _ in diffs) / len(diffs)
        for _ in range(iterations)
    ) if diffs else []
    return {
        "denominator": len(tasks),
        "b4_successes": sum(b4),
        "b0_successes": sum(b0),
        "b0_minus_b4": sum(diffs) / len(diffs) if diffs else None,
        "discordant": {
            "b0_only": b0_only,
            "b4_only": b4_only,
        },
        "ties": len(tasks) - b0_only - b4_only,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(b0_only, b4_only),
        "paired_bootstrap_95_ci": (
            [
                boots[int(0.025 * len(boots))],
                boots[int(0.975 * len(boots)) - 1],
            ]
            if boots
            else None
        ),
        "bootstrap_iterations": iterations,
    }


def reduce(
    *,
    repository_root: Path,
    config_path: Path,
    b4_roots: list[Path],
    b0_roots: list[Path],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    config = _load_json(config_path)
    plan_path = repository_root / config["upstream"]["memory_split"]
    plan = _load_json(plan_path)
    roster = list(plan["benchmark_profiles"]["frozen_gui_owl_gui_only"])
    if len(roster) != config["benchmark"]["expected_denominator"]:
        raise ValueError("MobileWorld frozen strict denominator drifted")
    records = {record["task_name"]: record for record in plan["records"]}
    if not set(roster).issubset(records):
        raise ValueError("MobileWorld frozen roster lacks task records")

    b4_scores = _read_scores(b4_roots, roster)
    b0_scores = _read_scores(b0_roots, roster)
    split_tasks = {
        split: [
            task
            for task in roster
            if records[task]["memory_split"] == split
        ]
        for split in (
            "cross_app_memory_candidate",
            "single_app_control",
        )
    }
    task_rows = [
        {
            "task": task,
            "memory_split": records[task]["memory_split"],
            "b4_score": b4_scores.get(task),
            "b0_score": b0_scores.get(task),
            "b4_success": b4_scores.get(task, 0.0) > SUCCESS_THRESHOLD,
            "b0_success": b0_scores.get(task, 0.0) > SUCCESS_THRESHOLD,
        }
        for task in roster
    ]
    return {
        "schema_version": "causalcache.mobileworld.b0_b4_paired_result.v1",
        "config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "memory_plan": {
            "path": str(plan_path),
            "sha256": _sha256(plan_path),
        },
        "roster_count": len(roster),
        "success_threshold": SUCCESS_THRESHOLD,
        "input_roots": {
            "b4": [str(root) for root in b4_roots],
            "b0": [str(root) for root in b0_roots],
        },
        "arms": {
            "b4": _arm_summary(roster, records, b4_scores),
            "b0": _arm_summary(roster, records, b0_scores),
        },
        "paired": {
            "strict_117_missing_as_zero": _paired_comparison(
                roster,
                b4_scores,
                b0_scores,
                missing_as_zero=True,
                iterations=iterations,
                seed=seed,
            ),
            "observed_intersection": _paired_comparison(
                roster,
                b4_scores,
                b0_scores,
                missing_as_zero=False,
                iterations=iterations,
                seed=seed,
            ),
            "strict_by_split": {
                split: _paired_comparison(
                    tasks,
                    b4_scores,
                    b0_scores,
                    missing_as_zero=True,
                    iterations=iterations,
                    seed=seed,
                )
                for split, tasks in split_tasks.items()
            },
        },
        "tasks": task_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--b4-trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--b0-trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_726)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    result = reduce(
        repository_root=args.repository_root.resolve(),
        config_path=args.config.resolve(),
        b4_roots=[path.resolve() for path in args.b4_trajectories],
        b0_roots=[path.resolve() for path in args.b0_trajectories],
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["paired"], sort_keys=True))


if __name__ == "__main__":
    main()
