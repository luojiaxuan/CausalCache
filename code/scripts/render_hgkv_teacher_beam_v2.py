#!/usr/bin/env python3
"""Render the unique cache-missing coalitions from one V2 teacher depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.build_odyssey_sft_dataset import render_trajectory
from scripts.plan_hgkv_teacher_beam_v2 import expand_paths


def load_missing_coalitions(
    paths: Iterable[Path],
) -> dict[str, dict[str, Any]]:
    coalitions: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["cache_hit"]:
                continue
            pair_group = str(row["pair_group"])
            set_key = str(row["restored_set_key"])
            key = f"{pair_group}/{set_key}"
            candidate = {
                "episode": str(row["episode"]),
                "pair_group": pair_group,
                "decision_step": int(row["decision_step"]),
                "candidate_event_step_ids": [
                    int(value) for value in row["candidate_event_step_ids"]
                ],
                "selected_event_step_ids": [
                    int(value) for value in row["restored_event_step_ids"]
                ],
                "restored_set_key": set_key,
                "depth": int(row["depth"]),
            }
            previous = coalitions.get(key)
            if previous is not None and previous != candidate:
                raise ValueError(
                    f"{path}:{line_no} conflicting missing coalition {key}"
                )
            coalitions[key] = candidate
    return coalitions


def selected_set_plan(
    coalitions: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    plan: dict[str, list[dict[str, Any]]] = {}
    for coalition in coalitions.values():
        pair_group = str(coalition["pair_group"])
        plan.setdefault(pair_group, []).append(
            {
                "candidate_event_step_ids": coalition[
                    "candidate_event_step_ids"
                ],
                "selected_event_step_ids": coalition[
                    "selected_event_step_ids"
                ],
                "budget": len(coalition["selected_event_step_ids"]),
                "method": "teacher_beam_v2",
            }
        )
    for values in plan.values():
        values.sort(key=lambda item: item["selected_event_step_ids"])
    return plan


def annotate_samples(
    samples: list[dict[str, Any]],
    *,
    expected_by_key: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        if sample.get("variant") != "selected_set":
            raise ValueError("teacher render produced a non-selected-set sample")
        key = (
            f"{sample['pair_group']}/"
            f"{sample['restored_set_key']}"
        )
        expected = expected_by_key.get(key)
        if expected is None:
            raise ValueError(f"teacher render produced unexpected coalition {key}")
        if key in seen:
            raise ValueError(f"teacher render duplicated coalition {key}")
        seen.add(key)
        result = dict(sample)
        result["schema_version"] = (
            "causalcache.hgkv_selector_v2.teacher_coalition_sample.v1"
        )
        result["teacher_depth"] = int(expected["depth"])
        annotated.append(result)
    if seen != set(expected_by_key):
        raise ValueError(
            f"teacher render missed {len(set(expected_by_key) - seen)} coalitions"
        )
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--plan", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must be in [0, shard count)")
    plan_paths = expand_paths(args.plan)
    if not plan_paths:
        parser.error("plan patterns must match files")
    missing = load_missing_coalitions(plan_paths)
    plan = selected_set_plan(missing)

    from pyarrow import parquet as pq

    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"samples-shard{args.shard_index:03d}.jsonl"
    expected_in_shard: dict[str, dict[str, Any]] = {}
    source_rows: dict[str, dict[str, Any]] = {}
    for shard_number, shard_path in enumerate(
        sorted(args.source_root.glob("shard-*.parquet"))
    ):
        if shard_number % args.shard_count != args.shard_index:
            continue
        for source_row in pq.read_table(shard_path).to_pylist():
            episode = str(source_row["source_id"])
            if episode in source_rows:
                raise ValueError(f"duplicate source trajectory {episode}")
            source_rows[episode] = source_row
    for key, coalition in missing.items():
        if coalition["episode"] in source_rows:
            expected_in_shard[key] = coalition
    expected_by_episode: dict[str, list[dict[str, Any]]] = {}
    for coalition in expected_in_shard.values():
        expected_by_episode.setdefault(str(coalition["episode"]), []).append(
            coalition
        )

    rendered: list[dict[str, Any]] = []
    for episode, source_row in sorted(source_rows.items()):
        state_rows = expected_by_episode.get(episode, [])
        if not state_rows:
            continue
        pair_group = str(state_rows[0]["pair_group"])
        samples = render_trajectory(
            source_row,
            annotations_root=args.annotations_root,
            decisions=[int(state_rows[0]["decision_step"])],
            output_root=args.output_root,
            shared_early_decisions=2,
            contrast_variants=False,
            donor_paths=None,
            selected_set_plan={pair_group: plan[pair_group]},
        )
        rendered.extend(samples)
    rendered = annotate_samples(
        rendered,
        expected_by_key=expected_in_shard,
    )
    with output.open("w", encoding="utf-8") as handle:
        for sample in rendered:
            handle.write(
                json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n"
            )
    print(
        json.dumps(
            {
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "missing_coalitions_total": len(missing),
                "rendered_coalitions": len(rendered),
                "status": "DONE",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
