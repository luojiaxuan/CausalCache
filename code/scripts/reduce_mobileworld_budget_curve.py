#!/usr/bin/env python3
"""Reduce the strict paired MobileWorld B0--B4 history-image budget curve."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from scripts.reduce_mobileworld_b0_b4 import _mcnemar_exact


SUCCESS_THRESHOLD = 0.99
ARM_NAMES = ("B0", "B1", "B2", "B3", "B4")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _wilson_interval(successes: int, denominator: int) -> list[float]:
    if denominator <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    z = 1.959963984540054
    rate = successes / denominator
    scale = 1 + z * z / denominator
    center = (rate + z * z / (2 * denominator)) / scale
    radius = (
        z
        * math.sqrt(
            rate * (1 - rate) / denominator
            + z * z / (4 * denominator * denominator)
        )
        / scale
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _load_task_scores(
    path: Path,
    expected_arm: str,
    roster: list[str],
) -> dict[str, float]:
    payload = _load_json(path)
    if payload.get("arm") != expected_arm:
        raise ValueError(f"{path} does not contain {expected_arm} scores")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"{path} rows must be a list")
    scores: dict[str, float] = {}
    for row in rows:
        task = row["task"]
        if task in scores:
            raise ValueError(f"{path} contains duplicate task {task}")
        scores[task] = float(row["score"])
    if set(scores) != set(roster):
        missing = sorted(set(roster) - set(scores))
        extra = sorted(set(scores) - set(roster))
        raise ValueError(f"{path} roster mismatch: missing={missing}, extra={extra}")
    return scores


def _paired_comparison(
    left_arm: str,
    right_arm: str,
    tasks: list[str],
    scores: dict[str, dict[str, float]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    left = [
        scores[left_arm][task] >= SUCCESS_THRESHOLD
        for task in tasks
    ]
    right = [
        scores[right_arm][task] >= SUCCESS_THRESHOLD
        for task in tasks
    ]
    left_only = sum(
        left_value and not right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    right_only = sum(
        right_value and not left_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    diffs = [
        int(right_value) - int(left_value)
        for left_value, right_value in zip(left, right, strict=True)
    ]
    rng = random.Random(seed)
    bootstrap = sorted(
        sum(rng.choice(diffs) for _ in diffs) / len(diffs)
        for _ in range(iterations)
    )
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "denominator": len(tasks),
        "left_successes": sum(left),
        "right_successes": sum(right),
        "right_minus_left": sum(diffs) / len(diffs),
        "paired_bootstrap_95_ci": [
            bootstrap[int(0.025 * len(bootstrap))],
            bootstrap[int(0.975 * len(bootstrap)) - 1],
        ],
        "discordant": {
            "left_only": left_only,
            "right_only": right_only,
        },
        "ties": len(tasks) - left_only - right_only,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(
            left_only,
            right_only,
        ),
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
    }


def _task_matrix_rows(
    base_rows: list[dict[str, Any]],
    scores: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_row in base_rows:
        task = base_row["task"]
        row: dict[str, Any] = {
            "task": task,
            "memory_split": base_row["memory_split"],
        }
        for arm in ARM_NAMES:
            arm_key = arm.lower()
            score = scores[arm][task]
            row[f"{arm_key}_score"] = score
            row[f"{arm_key}_success"] = score >= SUCCESS_THRESHOLD
        rows.append(row)
    return rows


def reduce(
    *,
    b0_b4_summary_path: Path,
    task_score_paths: dict[str, Path],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    base = _load_json(b0_b4_summary_path)
    base_rows = base.get("tasks")
    if not isinstance(base_rows, list):
        raise TypeError("B0/B4 summary tasks must be a list")
    roster = [row["task"] for row in base_rows]
    if len(roster) != 117 or len(set(roster)) != 117:
        raise ValueError("MobileWorld strict roster must contain 117 unique tasks")

    scores = {
        "B0": {row["task"]: float(row["b0_score"] or 0.0) for row in base_rows},
        "B4": {row["task"]: float(row["b4_score"] or 0.0) for row in base_rows},
    }
    for arm in ("B1", "B2", "B3"):
        scores[arm] = _load_task_scores(
            task_score_paths[arm],
            arm,
            roster,
        )

    task_rows = _task_matrix_rows(base_rows, scores)
    arms = {}
    for arm in ARM_NAMES:
        successes = sum(
            scores[arm][task] >= SUCCESS_THRESHOLD
            for task in roster
        )
        arms[arm] = {
            "successes": successes,
            "denominator": len(roster),
            "success_rate": successes / len(roster),
            "wilson_95_ci": _wilson_interval(successes, len(roster)),
        }

    relative_to_b0 = {
        arm: _paired_comparison(
            "B0",
            arm,
            roster,
            scores,
            iterations=iterations,
            seed=seed,
        )
        for arm in ("B1", "B2", "B3", "B4")
    }
    adjacent = {
        right: _paired_comparison(
            left,
            right,
            roster,
            scores,
            iterations=iterations,
            seed=seed,
        )
        for left, right in zip(ARM_NAMES[:-1], ARM_NAMES[1:], strict=True)
    }
    return {
        "schema_version": "causalcache.mobileworld.budget_curve_paired_result.v1",
        "success_threshold": SUCCESS_THRESHOLD,
        "roster_count": len(roster),
        "inputs": {
            "b0_b4_summary": {
                "path": str(b0_b4_summary_path),
                "sha256": _sha256(b0_b4_summary_path),
            },
            "task_scores": {
                arm: {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for arm, path in sorted(task_score_paths.items())
            },
        },
        "bootstrap": {
            "iterations": iterations,
            "seed": seed,
            "unit": "task",
        },
        "arms": arms,
        "paired": {
            "relative_to_b0": relative_to_b0,
            "adjacent": adjacent,
        },
        "tasks": task_rows,
    }


def _write_task_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-b4-summary", type=Path, required=True)
    parser.add_argument("--b1-task-scores", type=Path, required=True)
    parser.add_argument("--b2-task-scores", type=Path, required=True)
    parser.add_argument("--b3-task-scores", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_726)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-matrix-output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    result = reduce(
        b0_b4_summary_path=args.b0_b4_summary.resolve(),
        task_score_paths={
            "B1": args.b1_task_scores.resolve(),
            "B2": args.b2_task_scores.resolve(),
            "B3": args.b3_task_scores.resolve(),
        },
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_task_matrix(args.task_matrix_output, result["tasks"])
    print(json.dumps({"arms": result["arms"], "paired": result["paired"]}))


if __name__ == "__main__":
    main()
