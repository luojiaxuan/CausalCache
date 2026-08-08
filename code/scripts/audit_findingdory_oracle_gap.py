"""Audit a completed FindingDory exact-budget oracle-gap artifact."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from causalcache.findingdory_oracle_gap import (
    exact_budget_selections,
    frame_is_valid,
    parse_answer_groups,
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _forbidden_summary_keys(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if re.search(r"frame|time|index", str(key), flags=re.IGNORECASE):
                found.append(str(key))
            found.extend(_forbidden_summary_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_forbidden_summary_keys(item))
    return found


def audit(input_dir: Path, config_path: Path) -> dict[str, Any]:
    config = _json(config_path)
    tasks = _jsonl(input_dir / "tasks.jsonl")
    rows = [row for path in sorted(input_dir.glob("results-shard-*.jsonl")) for row in _jsonl(path)]
    budgets = tuple(int(value) for value in config["memory"]["budgets"])
    current_frame = int(config["benchmark"]["current_frame"])
    video_frames = int(config["benchmark"]["video_frames"])
    chunk_size = int(config["memory"]["summary_chunk_frames"])

    task_lookup = {(task["ep_id"], task["task_id"]): task for task in tasks}
    expected_keys = {
        (task["ep_id"], task["task_id"], budget, arm)
        for task in tasks
        for budget in budgets
        for arm in ("recent", "oracle")
    }
    actual_keys = [
        (row["episode_id"], row["task_id"], int(row["budget"]), row["arm"])
        for row in rows
    ]
    key_counts = Counter(actual_keys)

    exact_budget_violations = 0
    selection_rule_violations = 0
    oracle_missing_valid_evidence = 0
    success_recompute_violations = 0
    answer_group_violations = 0
    parse_failures: Counter[str] = Counter()
    out_of_domain_predictions: Counter[str] = Counter()
    for row in rows:
        budget = int(row["budget"])
        arm = str(row["arm"])
        selected = tuple(int(frame) for frame in row["selected_frames"])
        task = task_lookup[(row["episode_id"], row["task_id"])]
        task_groups = parse_answer_groups(task["answer"])
        row_groups = tuple(tuple(int(frame) for frame in group) for group in row["answer_groups"])
        if row_groups != task_groups:
            answer_group_violations += 1
        expected = exact_budget_selections(task_groups, budget=budget, current_frame=current_frame)[arm]
        if len(selected) != budget or any(frame < 0 or frame >= current_frame for frame in selected):
            exact_budget_violations += 1
        if selected != expected:
            selection_rule_violations += 1
        if arm == "oracle" and not any(frame_is_valid(frame, task_groups) for frame in selected):
            oracle_missing_valid_evidence += 1
        prediction = row["predicted_frame"]
        if prediction is None:
            parse_failures[f"B{budget}_{arm}"] += 1
        elif not isinstance(prediction, int) or not 0 <= prediction < video_frames:
            out_of_domain_predictions[f"B{budget}_{arm}"] += 1
        if bool(row["success"]) != frame_is_valid(prediction, task_groups):
            success_recompute_violations += 1

    expected_episodes = sorted({task["ep_id"] for task in tasks})
    summary_files = sorted((input_dir / "summaries").glob("*.json"))
    summary_episode_ids: list[str] = []
    summary_chunks = 0
    numeric_namespace_violations = 0
    forbidden_summary_key_violations = 0
    canonical_range_violations = 0
    for path in summary_files:
        summary = _json(path)
        summary_episode_ids.append(str(summary["episode_id"]))
        chunks = summary["chunks"]
        summary_chunks += len(chunks)
        if len(chunks) != video_frames // chunk_size:
            canonical_range_violations += 1
        for index, chunk in enumerate(chunks):
            expected_start = index * chunk_size
            expected_end = min(expected_start + chunk_size, video_frames) - 1
            if (int(chunk["start_frame"]), int(chunk["end_frame"])) != (expected_start, expected_end):
                canonical_range_violations += 1
            content = chunk["content_summary"]
            if re.search(r"\d", json.dumps(content, ensure_ascii=False)):
                numeric_namespace_violations += 1
            forbidden_summary_key_violations += len(_forbidden_summary_keys(content))

    critical = {
        "missing_result_keys": len(expected_keys - set(actual_keys)),
        "unexpected_result_keys": len(set(actual_keys) - expected_keys),
        "duplicate_result_keys": sum(count - 1 for count in key_counts.values() if count > 1),
        "answer_group_violations": answer_group_violations,
        "exact_budget_violations": exact_budget_violations,
        "selection_rule_violations": selection_rule_violations,
        "oracle_missing_valid_evidence": oracle_missing_valid_evidence,
        "success_recompute_violations": success_recompute_violations,
        "missing_or_unexpected_summary_episodes": len(set(expected_episodes) ^ set(summary_episode_ids)),
        "duplicate_summary_episodes": len(summary_episode_ids) - len(set(summary_episode_ids)),
        "numeric_namespace_violations": numeric_namespace_violations,
        "forbidden_summary_key_violations": forbidden_summary_key_violations,
        "canonical_range_violations": canonical_range_violations,
    }
    return {
        "schema_version": "1.0.0",
        "valid": all(value == 0 for value in critical.values()),
        "counts": {
            "tasks": len(tasks),
            "result_rows": len(rows),
            "expected_result_rows": len(expected_keys),
            "summary_files": len(summary_files),
            "summary_chunks": summary_chunks,
        },
        "critical_violations": critical,
        "diagnostics": {
            "parse_failures": dict(sorted(parse_failures.items())),
            "out_of_domain_predictions": dict(sorted(out_of_domain_predictions.items())),
        },
        "protocol": {
            "budgets": budgets,
            "current_frame": current_frame,
            "pixel_preprocessing": config["memory"]["pixel_preprocessing"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.input_dir, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
