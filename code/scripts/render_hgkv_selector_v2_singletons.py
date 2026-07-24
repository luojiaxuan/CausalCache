#!/usr/bin/env python3
"""Render full-history singleton samples for HGKV selector V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from causalcache.hgkv_selector_v2 import temporal_features
from scripts.build_odyssey_sft_dataset import render_trajectory


def load_states(path: Path) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        episode = str(row["episode"])
        if episode in states:
            raise ValueError(f"{path}:{line_no} duplicate episode {episode}")
        candidates = tuple(int(value) for value in row["candidate_event_step_ids"])
        if len(candidates) < 2 or len(candidates) != len(set(candidates)):
            raise ValueError(f"{path}:{line_no} invalid candidate inventory")
        if row.get("recent_candidate_cap") is not None:
            raise ValueError(f"{path}:{line_no} recent cap is forbidden")
        states[episode] = row
    return states


def annotate_v2_singleton_samples(
    samples: list[dict[str, Any]], state: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates = tuple(int(value) for value in state["candidate_event_step_ids"])
    singleton_ids = tuple(
        int(sample["singleton_event_step_id"])
        for sample in samples
        if sample.get("singleton_event_step_id") is not None
    )
    if singleton_ids != candidates:
        raise ValueError(
            f"{state['pair_group']} rendered candidates {singleton_ids} "
            f"!= full inventory {candidates}"
        )
    if len(samples) != len(candidates) + 1:
        raise ValueError(f"{state['pair_group']} singleton group is incomplete")
    annotated: list[dict[str, Any]] = []
    for sample in samples:
        result = dict(sample)
        result["schema_version"] = "causalcache.hgkv_selector_v2.singleton_sample.v1"
        result["candidate_event_step_ids"] = list(candidates)
        result["history_length"] = int(state["history_length"])
        result["v2_feature_schema"] = (
            "hgkv_readout_1280_plus_relative_age_normalized_age_"
            "normalized_position_is_most_recent_log_history_length"
        )
        singleton = sample.get("singleton_event_step_id")
        if singleton is not None:
            result["temporal_features"] = list(
                temporal_features(
                    event_step_id=int(singleton),
                    decision_step=int(state["decision_step"]),
                    history_length=int(state["history_length"]),
                    candidate_event_step_ids=candidates,
                )
            )
        annotated.append(result)
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must be in [0, shard count)")

    from pyarrow import parquet as pq

    states = load_states(args.states)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"samples-shard{args.shard_index:03d}.jsonl"
    groups = 0
    rows_written = 0
    seen_episodes: set[str] = set()
    with output.open("w", encoding="utf-8") as handle:
        for shard_number, shard_path in enumerate(
            sorted(args.source_root.glob("shard-*.parquet"))
        ):
            if shard_number % args.shard_count != args.shard_index:
                continue
            for source_row in pq.read_table(shard_path).to_pylist():
                episode = str(source_row["source_id"])
                state = states.get(episode)
                if state is None:
                    continue
                if episode in seen_episodes:
                    raise ValueError(f"duplicate source trajectory {episode}")
                seen_episodes.add(episode)
                samples = render_trajectory(
                    source_row,
                    annotations_root=args.annotations_root,
                    decisions=[int(state["decision_step"])],
                    output_root=args.output_root,
                    shared_early_decisions=2,
                    contrast_variants=False,
                    donor_paths=None,
                    selector_singletons=True,
                    singleton_max_candidates=len(
                        state["candidate_event_step_ids"]
                    ),
                )
                samples = annotate_v2_singleton_samples(samples, state)
                for sample in samples:
                    handle.write(
                        json.dumps(sample, ensure_ascii=False, sort_keys=True)
                        + "\n"
                    )
                groups += 1
                rows_written += len(samples)
    expected_episodes = {
        episode
        for index, episode in enumerate(sorted(states))
        if index % args.shard_count == args.shard_index
    }
    # note (luojiaxuan): parquet shard assignment and episode lexical order are
    # unrelated, so exact coverage is enforced by the global validator after all
    # renderer shards, not by this per-source-shard process.
    print(
        json.dumps(
            {
                "groups": groups,
                "rows": rows_written,
                "states_total": len(states),
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "lexical_expected_hint": len(expected_episodes),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
