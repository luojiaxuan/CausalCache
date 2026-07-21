#!/usr/bin/env python3
"""In-sample diagnostic: join selector predictions with long-oracle truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_evaluation import canonical_json_bytes
from causalcache.set_utility_long_oracle import (
    WAVES,
    greedy_selection_from_distances,
    recent_prefix,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        tail = index
        while (
            tail + 1 < len(order)
            and values[order[tail + 1]] == values[order[index]]
        ):
            tail += 1
        average = (index + tail) / 2.0 + 1.0
        for position in range(index, tail + 1):
            ranks[order[position]] = average
        index = tail + 1
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        raise ValueError("spearman requires at least three paired values")
    a, b = _ranks(left), _ranks(right)
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a == 0.0 or var_b == 0.0:
        return 0.0
    return cov / math.sqrt(var_a * var_b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-output", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("in-sample diagnostic output already exists")
    selector = _read_json(args.selector_output)

    truth: dict[str, dict[tuple[int, ...], float]] = defaultdict(dict)
    references: dict[str, set[str]] = defaultdict(set)
    for wave in WAVES:
        for host in ("terminals-hyper00", "terminals-hyper01"):
            states_dir = args.truth_root / f"wave-{wave}" / host / "states"
            if not states_dir.is_dir():
                continue
            for path in sorted(states_dir.glob("*.json")):
                terminal = _read_json(path)
                if terminal["status"] != "COMPLETED_VARIABLE_HISTORY_LABEL_STATE":
                    continue
                state_id = terminal["state_id"]
                references[state_id].add(
                    terminal["reference"]["serialized_action"]
                )
                for row in terminal["distance_rows"]:
                    subset = tuple(row["coalition_event_step_ids"])
                    value = float(row["distance"])
                    known = truth[state_id].get(subset)
                    if known is not None and abs(known - value) > 1e-6:
                        raise ValueError(
                            f"{state_id} duplicate distance drifted: {subset}"
                        )
                    truth[state_id][subset] = value

    wave1_singletons: dict[str, dict[int, float]] = {}
    per_state: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for record in selector["records"]:
        state_id = record["state_id"]
        candidates = tuple(record["candidate_event_ids"])
        distances = truth.get(state_id)
        if not distances or len(references[state_id]) != 1:
            skipped[state_id] = "truth_missing_or_reference_drift"
            continue
        empty = distances.get(())
        if empty is None or empty <= 0.0:
            skipped[state_id] = "empty_distance_not_positive"
            continue
        singles_true = {
            subset[0]: value
            for subset, value in distances.items()
            if len(subset) == 1
        }
        if set(singles_true) != set(candidates):
            skipped[state_id] = "incomplete_singleton_truth"
            continue

        ranked = [
            row
            for row in record["conditional_steps"][0]["ranked_candidates"]
            if len(row["subset"]) == 1
        ]
        predicted = {row["subset"][0]: float(row["predicted_utility"]) for row in ranked}
        if set(predicted) != set(candidates):
            skipped[state_id] = "incomplete_singleton_predictions"
            continue

        def recovery(subset: tuple[int, ...]) -> float | None:
            value = distances.get(subset)
            if value is None:
                return None
            return (empty - value) / empty

        true_utility = {
            event: (empty - value) / empty for event, value in singles_true.items()
        }
        true_best = min(singles_true, key=lambda event: (singles_true[event], event))
        model_top1 = max(predicted, key=lambda event: (predicted[event], -event))
        predicted_top4 = sorted(
            predicted, key=lambda event: (-predicted[event], event)
        )[:4]
        recent4 = set(recent_prefix(candidates, 4))

        prefix: tuple[int, ...] = ()
        oracle_prefixes = []
        for _ in range(4):
            rows = [
                {"coalition_event_step_ids": list(subset), "distance": value}
                for subset, value in distances.items()
            ]
            try:
                prefix = greedy_selection_from_distances(
                    candidate_event_ids=candidates,
                    distance_rows=rows,
                    previous_selected=prefix,
                )
            except ValueError:
                break
            oracle_prefixes.append(prefix)

        budgets = {}
        for budget in (1, 2, 3, 4):
            learned = tuple(sorted(record["learned"][str(budget)]))
            budgets[str(budget)] = {
                "learned_subset": list(learned),
                "learned_recovery": recovery(learned),
                "oracle_recovery": (
                    max(
                        value
                        for value in (
                            recovery(oracle_prefixes[index])
                            for index in range(min(budget, len(oracle_prefixes)))
                        )
                        if value is not None
                    )
                    if oracle_prefixes
                    else None
                ),
                "recent_recovery": recovery(recent_prefix(candidates, budget)),
            }

        per_state.append(
            {
                "candidate_count": len(candidates),
                "history_bin": "long" if len(candidates) <= 32 else "very_long",
                "budgets": budgets,
                "model_top1_in_recent4": model_top1 in recent4,
                "model_top1_is_true_best": model_top1 == true_best,
                "singleton_spearman": _spearman(
                    [predicted[event] for event in candidates],
                    [true_utility[event] for event in candidates],
                ),
                "state_id": state_id,
                "trajectory_id": record["trajectory_id"],
                "true_best_in_predicted_top4": true_best in predicted_top4,
                "true_best_outside_recent4": true_best not in recent4,
            }
        )
        wave1_singletons[state_id] = singles_true

    if not per_state:
        raise ValueError("no state joined selector predictions with truth")

    def trajectory_equal(rows: list[dict[str, Any]], value) -> float:
        by_trajectory: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            extracted = value(row)
            if extracted is not None:
                by_trajectory[row["trajectory_id"]].append(float(extracted))
        return sum(
            sum(values) / len(values) for values in by_trajectory.values()
        ) / len(by_trajectory)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        outside = [row for row in rows if row["true_best_outside_recent4"]]
        budget_summary = {}
        for budget in ("1", "2", "3", "4"):
            covered = [
                row
                for row in rows
                if row["budgets"][budget]["learned_recovery"] is not None
            ]
            budget_summary[f"B{budget}"] = {
                "learned_truth_coverage": len(covered) / len(rows),
                "learned_recovery_covered": (
                    trajectory_equal(
                        covered, lambda row: row["budgets"][budget]["learned_recovery"]
                    )
                    if covered
                    else None
                ),
                "oracle_recovery_covered": (
                    trajectory_equal(
                        covered, lambda row: row["budgets"][budget]["oracle_recovery"]
                    )
                    if covered
                    else None
                ),
                "recent_recovery_covered": (
                    trajectory_equal(
                        covered, lambda row: row["budgets"][budget]["recent_recovery"]
                    )
                    if covered
                    else None
                ),
            }
        return {
            "budgets": budget_summary,
            "model_top1_in_recent4_rate": trajectory_equal(
                rows, lambda row: row["model_top1_in_recent4"]
            ),
            "model_top1_is_true_best_rate": trajectory_equal(
                rows, lambda row: row["model_top1_is_true_best"]
            ),
            "singleton_spearman_mean": trajectory_equal(
                rows, lambda row: row["singleton_spearman"]
            ),
            "state_count": len(rows),
            "true_best_in_predicted_top4_rate": trajectory_equal(
                rows, lambda row: row["true_best_in_predicted_top4"]
            ),
            "true_best_outside_recent4_rate": len(outside) / len(rows),
            "true_best_outside_recent4_found_in_predicted_top4_rate": (
                trajectory_equal(
                    outside, lambda row: row["true_best_in_predicted_top4"]
                )
                if outside
                else None
            ),
        }

    summary = {
        "checkpoint_sha256": selector["checkpoint_sha256"],
        "input_content_sha256": selector["input_content_sha256"],
        "overall": summarize(per_state),
        "by_bin": {
            name: summarize(
                [row for row in per_state if row["history_bin"] == name]
            )
            for name in ("long", "very_long")
        },
        "per_state": {row["state_id"]: row for row in per_state},
        "purpose": "IN_SAMPLE_TRAIN_DIAGNOSTIC_ONLY_NOT_EVALUATION",
        "schema_version": "causalcache.set_utility_long_oracle_insample.v1",
        "selector_variant": selector["variant"],
        "skipped_states": dict(sorted(skipped.items())),
        "status": "COMPLETED_SET_UTILITY_LONG_ORACLE_INSAMPLE_DIAGNOSTIC",
    }
    summary["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(summary)
    ).hexdigest()
    payload = canonical_json_bytes(summary) + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(json.dumps({"overall": summary["overall"], "by_bin_state_counts": {k: v["state_count"] for k, v in summary["by_bin"].items()}}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
