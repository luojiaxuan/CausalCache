#!/usr/bin/env python3
"""Build the frozen heldout selected-set gate plan and non-oracle baselines."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from causalcache.restoration_v2_baselines import (
    joint_rgb_histogram_cosine,
    ocr_token_set_jaccard,
)
from causalcache.restoration_v2_text_backend import prepare_image_bytes
from scripts.train_hgkv_singleton_selector_v1 import (
    atomic_write_json,
    load_scores,
    sha256_file,
)


def deterministic_random_selection(
    pair_group: str,
    candidates: Iterable[int],
    *,
    budget: int,
    seed: int,
) -> tuple[int, ...]:
    ranked = sorted(
        candidates,
        key=lambda event_id: hashlib.sha256(
            f"{seed}:{pair_group}:B{budget}:{event_id}".encode()
        ).digest(),
    )
    return tuple(sorted(ranked[:budget]))


def ocr_tokens_for(
    ocr_records: list | dict, *, source_id: str, step_id: int
) -> list[str]:
    record = None
    if isinstance(ocr_records, list):
        record = ocr_records[step_id] if step_id < len(ocr_records) else None
    elif isinstance(ocr_records, dict):
        for key in (
            str(step_id),
            f"observation-{step_id:03d}",
            f"observation-{step_id:03d}.png",
            f"images/{source_id}/observation-{step_id:03d}.png",
        ):
            if key in ocr_records:
                record = ocr_records[key]
                break
    if isinstance(record, dict):
        return list(record.get("full_spatial_tokens", []))
    return []


def similarity_ranking(
    *,
    candidates: list[int],
    event_ocr_tokens: dict[int, list[str]],
    current_ocr_tokens: list[str],
    event_image_bytes: dict[int, bytes],
    current_image_bytes: bytes,
) -> tuple[list[int], dict[int, float]]:
    current_rgb = prepare_image_bytes(
        current_image_bytes
    ).resized_rgb_bytes
    scores = {}
    for event_id in candidates:
        event_rgb = prepare_image_bytes(
            event_image_bytes[event_id]
        ).resized_rgb_bytes
        ocr_score = ocr_token_set_jaccard(
            event_ocr_tokens[event_id], current_ocr_tokens
        )
        rgb_score = joint_rgb_histogram_cosine(event_rgb, current_rgb)
        scores[event_id] = 0.5 * ocr_score + 0.5 * rgb_score
    ranking = sorted(candidates, key=lambda event_id: (-scores[event_id], event_id))
    return ranking, scores


def load_learned_selections(
    path: Path,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (
                str(row["pair_group"]),
                str(row["method"]),
                int(row["budget"]),
            )
            if key in rows:
                raise ValueError(f"duplicate learned selection: {key}")
            rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--singleton-scores", type=Path, required=True)
    parser.add_argument("--learned-selections", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--random-seed", type=int, default=20260724)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    singleton_rows = load_scores(args.singleton_scores)
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in singleton_rows:
        by_state[row["pair_group"]].append(row)
    learned = load_learned_selections(args.learned_selections)
    learned_methods = ("hgkv_set_conditioned", "hgkv_singleton")
    expected_learned = {
        (pair_group, method, budget)
        for pair_group in by_state
        for method in learned_methods
        for budget in args.budgets
    }
    if set(learned) != expected_learned:
        raise ValueError(
            f"learned selection coverage mismatch: observed={len(learned)} "
            f"expected={len(expected_learned)}"
        )

    states_by_episode: dict[str, list[str]] = defaultdict(list)
    for pair_group, rows in by_state.items():
        states_by_episode[rows[0]["episode"]].append(pair_group)
    similarity_by_state: dict[str, tuple[list[int], dict[int, float]]] = {}
    from pyarrow import parquet as pq

    remaining_episodes = set(states_by_episode)
    for source_path in sorted(args.source_root.glob("shard-*.parquet")):
        for row in pq.read_table(source_path).to_pylist():
            source_id = str(row["source_id"])
            if source_id not in remaining_episodes:
                continue
            images = [
                bytes(item["bytes"]) if isinstance(item, dict) else bytes(item)
                for item in row["images"]
            ]
            ocr_records = json.loads(row["ocr_records_json"])
            for pair_group in states_by_episode[source_id]:
                candidates = sorted(
                    candidate["event_id"] for candidate in by_state[pair_group]
                )
                decision = int(pair_group.rsplit(":", 1)[1])
                current_step = decision - 1
                ranking, scores = similarity_ranking(
                    candidates=candidates,
                    event_ocr_tokens={
                        event_id: ocr_tokens_for(
                            ocr_records,
                            source_id=source_id,
                            step_id=event_id,
                        )
                        for event_id in candidates
                    },
                    current_ocr_tokens=ocr_tokens_for(
                        ocr_records,
                        source_id=source_id,
                        step_id=current_step,
                    ),
                    event_image_bytes={
                        event_id: images[event_id] for event_id in candidates
                    },
                    current_image_bytes=images[current_step],
                )
                similarity_by_state[pair_group] = (ranking, scores)
            remaining_episodes.remove(source_id)
        if not remaining_episodes:
            break
    if remaining_episodes:
        raise ValueError(
            f"source rows missing for {len(remaining_episodes)} episodes"
        )
    if set(similarity_by_state) != set(by_state):
        raise ValueError("similarity state coverage mismatch")

    output_rows = []
    methods = [
        "hgkv_set_conditioned",
        "hgkv_singleton",
        "random",
        "recent",
        "similarity",
    ]
    for pair_group in sorted(by_state):
        candidates = sorted(
            row["event_id"] for row in by_state[pair_group]
        )
        episode = by_state[pair_group][0]["episode"]
        ranking, similarity_scores = similarity_by_state[pair_group]
        for budget in args.budgets:
            if len(candidates) < budget:
                raise ValueError(f"{pair_group} has fewer than B{budget} candidates")
            selected_by_method = {
                "hgkv_set_conditioned": tuple(
                    sorted(
                        learned[
                            (pair_group, "hgkv_set_conditioned", budget)
                        ]["selected_event_step_ids"]
                    )
                ),
                "hgkv_singleton": tuple(
                    sorted(
                        learned[
                            (pair_group, "hgkv_singleton", budget)
                        ]["selected_event_step_ids"]
                    )
                ),
                "random": deterministic_random_selection(
                    pair_group,
                    candidates,
                    budget=budget,
                    seed=args.random_seed,
                ),
                "recent": tuple(sorted(candidates[-budget:])),
                "similarity": tuple(sorted(ranking[:budget])),
            }
            for method in methods:
                selected = selected_by_method[method]
                if len(selected) > budget or not set(selected).issubset(candidates):
                    raise ValueError(
                        f"invalid {method} selection for {pair_group} B{budget}"
                    )
                row = {
                    "budget": budget,
                    "candidate_event_step_ids": candidates,
                    "episode": episode,
                    "method": method,
                    "pair_group": pair_group,
                    "selected_event_step_ids": selected,
                }
                if method == "similarity":
                    row["ranking"] = ranking
                    row["scores_by_event_step"] = [
                        [event_id, similarity_scores[event_id]]
                        for event_id in candidates
                    ]
                if method == "random":
                    row["random_seed"] = args.random_seed
                output_rows.append(row)

    if any(
        not math.isfinite(score)
        for row in output_rows
        for _, score in row.get("scores_by_event_step", [])
    ):
        raise ValueError("non-finite similarity score")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(args.output)
    output_sha256 = sha256_file(args.output)
    atomic_write_json(
        args.output.parent / "DONE",
        {
            "budgets": args.budgets,
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "learned_selections_sha256": sha256_file(args.learned_selections),
            "methods": methods,
            "output": args.output.name,
            "output_rows": len(output_rows),
            "output_sha256": output_sha256,
            "random_seed": args.random_seed,
            "singleton_scores_sha256": sha256_file(args.singleton_scores),
            "source_commit": args.source_commit,
            "states": len(by_state),
            "status": "DONE",
        },
    )


if __name__ == "__main__":
    main()
