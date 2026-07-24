#!/usr/bin/env python3
"""Build the full-history formal selected-set gate plan for selector V2."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from causalcache.hgkv_selector_v2 import restored_set_key
from scripts.build_hgkv_selector_v2_training_data import (
    expand_paths,
    sha256_file,
)
from scripts.build_selector_gate_plan_v1 import (
    deterministic_random_selection,
    ocr_tokens_for,
    similarity_ranking,
)
from scripts.plan_hgkv_teacher_beam_v2 import load_cache
from scripts.train_hgkv_singleton_selector_v1 import atomic_write_json


BUDGETS = (1, 2, 4)
V2_METHODS = ("hgkv_v2_greedy", "hgkv_v2_beam4")


def load_states(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_group = str(row["pair_group"])
            if pair_group in states:
                raise ValueError(f"{path}:{line_no} duplicate state")
            states[pair_group] = row
    return states


def load_selections(
    paths: Iterable[Path],
    *,
    allowed_methods: set[str] | None = None,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            method = str(row["method"])
            if allowed_methods is not None and method not in allowed_methods:
                continue
            key = (str(row["pair_group"]), method, int(row["budget"]))
            if key in rows:
                raise ValueError(f"{path}:{line_no} duplicate selection {key}")
            rows[key] = row
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]


def teacher_selections(
    *,
    states: Mapping[str, Mapping[str, Any]],
    teacher_run_root: Path,
    cache: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    candidates_by_size: dict[str, dict[int, set[tuple[int, ...]]]] = {
        pair_group: defaultdict(set) for pair_group in states
    }
    for pair_group in states:
        candidates_by_size[pair_group][0].add(())
    for depth in range(3):
        beam_path = teacher_run_root / f"depth{depth}/beam.jsonl"
        if not beam_path.is_file():
            raise FileNotFoundError(f"teacher beam incomplete: {beam_path}")
        for row in _read_jsonl(beam_path):
            pair_group = str(row["pair_group"])
            for prefix in row["prefixes"]:
                candidates_by_size[pair_group][depth + 1].add(
                    tuple(int(value) for value in prefix)
                )
    depth3_plan = teacher_run_root / "depth3/plan.jsonl"
    if not depth3_plan.is_file():
        raise FileNotFoundError(f"teacher beam incomplete: {depth3_plan}")
    for row in _read_jsonl(depth3_plan):
        candidates_by_size[str(row["pair_group"])][4].add(
            tuple(int(value) for value in row["restored_event_step_ids"])
        )

    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for pair_group, state in states.items():
        for budget in BUDGETS:
            eligible = {
                coalition
                for size in range(budget + 1)
                for coalition in candidates_by_size[pair_group][size]
            }
            missing = [
                coalition
                for coalition in eligible
                if (pair_group, restored_set_key(coalition)) not in cache
            ]
            if missing:
                raise KeyError(
                    f"teacher selection lacks {len(missing)} utilities "
                    f"for {pair_group} B{budget}"
                )
            selected = min(
                eligible,
                key=lambda coalition: (
                    -float(
                        cache[(pair_group, restored_set_key(coalition))][
                            "u_act"
                        ]
                    ),
                    coalition,
                ),
            )
            key = (pair_group, "teacher_beam4", budget)
            output[key] = {
                "episode": str(state["episode"]),
                "pair_group": pair_group,
                "method": "teacher_beam4",
                "budget": budget,
                "selected_event_step_ids": list(selected),
            }
    return output


def similarity_by_state(
    *,
    states: Mapping[str, Mapping[str, Any]],
    source_root: Path,
) -> dict[str, tuple[list[int], dict[int, float]]]:
    from pyarrow import parquet as pq

    pair_groups_by_episode: dict[str, list[str]] = defaultdict(list)
    for pair_group, state in states.items():
        pair_groups_by_episode[str(state["episode"])].append(pair_group)
    remaining = set(pair_groups_by_episode)
    result: dict[str, tuple[list[int], dict[int, float]]] = {}
    for source_path in sorted(source_root.glob("shard-*.parquet")):
        for source_row in pq.read_table(source_path).to_pylist():
            episode = str(source_row["source_id"])
            if episode not in remaining:
                continue
            images = [
                bytes(item["bytes"]) if isinstance(item, dict) else bytes(item)
                for item in source_row["images"]
            ]
            ocr_records = json.loads(source_row["ocr_records_json"])
            for pair_group in pair_groups_by_episode[episode]:
                state = states[pair_group]
                candidates = [
                    int(value)
                    for value in state["candidate_event_step_ids"]
                ]
                current_step = int(state["decision_step"]) - 1
                result[pair_group] = similarity_ranking(
                    candidates=candidates,
                    event_ocr_tokens={
                        event_id: ocr_tokens_for(
                            ocr_records,
                            source_id=episode,
                            step_id=event_id,
                        )
                        for event_id in candidates
                    },
                    current_ocr_tokens=ocr_tokens_for(
                        ocr_records,
                        source_id=episode,
                        step_id=current_step,
                    ),
                    event_image_bytes={
                        event_id: images[event_id] for event_id in candidates
                    },
                    current_image_bytes=images[current_step],
                )
            remaining.remove(episode)
        if not remaining:
            break
    if remaining or set(result) != set(states):
        raise ValueError("source coverage is incomplete for similarity baseline")
    return result


def build_plan(
    *,
    states: Mapping[str, Mapping[str, Any]],
    v2: Mapping[tuple[str, str, int], Mapping[str, Any]],
    v1: Mapping[tuple[str, str, int], Mapping[str, Any]],
    teacher: Mapping[tuple[str, str, int], Mapping[str, Any]],
    exact: Mapping[tuple[str, str, int], Mapping[str, Any]],
    similarity: Mapping[str, tuple[list[int], dict[int, float]]],
    random_seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for pair_group, state in sorted(states.items()):
        candidates = [
            int(value) for value in state["candidate_event_step_ids"]
        ]
        ranking, similarity_scores = similarity[pair_group]
        for budget in BUDGETS:
            selected_by_method: dict[str, list[int]] = {
                method: [
                    int(value)
                    for value in v2[(pair_group, method, budget)][
                        "selected_event_step_ids"
                    ]
                ]
                for method in V2_METHODS
            }
            selected_by_method.update(
                {
                    "hgkv_v1_singleton": [
                        int(value)
                        for value in v1[
                            (pair_group, "hgkv_v1_singleton", budget)
                        ]["selected_event_step_ids"]
                    ],
                    "teacher_beam4": [
                        int(value)
                        for value in teacher[
                            (pair_group, "teacher_beam4", budget)
                        ]["selected_event_step_ids"]
                    ],
                    "random": list(
                        deterministic_random_selection(
                            pair_group,
                            candidates,
                            budget=budget,
                            seed=random_seed,
                        )
                    ),
                    "recent": sorted(candidates[-budget:]),
                    "similarity": sorted(ranking[:budget]),
                }
            )
            exact_row = exact.get((pair_group, "exact_oracle", budget))
            if exact_row is not None:
                selected_by_method["exact_oracle"] = [
                    int(value)
                    for value in exact_row["selected_event_step_ids"]
                ]
            for method, selected_values in selected_by_method.items():
                selected = sorted(selected_values)
                if (
                    len(selected) > budget
                    or len(selected) != len(set(selected))
                    or not set(selected).issubset(candidates)
                ):
                    raise ValueError(
                        f"invalid {method} selection for {pair_group} B{budget}"
                    )
                row = {
                    "schema_version": (
                        "causalcache.hgkv_selector_v2.gate_plan.v1"
                    ),
                    "episode": str(state["episode"]),
                    "pair_group": pair_group,
                    "method": method,
                    "budget": budget,
                    "candidate_event_step_ids": candidates,
                    "selected_event_step_ids": selected,
                }
                if method == "random":
                    row["random_seed"] = random_seed
                if method == "similarity":
                    row["ranking"] = ranking
                    row["scores_by_event_step"] = [
                        [event_id, similarity_scores[event_id]]
                        for event_id in candidates
                    ]
                output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", action="append", required=True)
    parser.add_argument("--v2-selections", action="append", required=True)
    parser.add_argument("--v1-selections", action="append", required=True)
    parser.add_argument("--teacher-run-root", type=Path, required=True)
    parser.add_argument("--coalition-cache", action="append", required=True)
    parser.add_argument("--exact-selections", action="append", default=[])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=20260724)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    state_paths = expand_paths(args.states)
    v2_paths = expand_paths(args.v2_selections)
    v1_paths = expand_paths(args.v1_selections)
    cache_paths = expand_paths(args.coalition_cache)
    exact_paths = expand_paths(args.exact_selections)
    if not state_paths or not v2_paths or not v1_paths or not cache_paths:
        parser.error("required state/selection/cache patterns must match")
    states = load_states(state_paths)
    v2 = load_selections(v2_paths, allowed_methods=set(V2_METHODS))
    expected_v2 = {
        (pair_group, method, budget)
        for pair_group in states
        for method in V2_METHODS
        for budget in BUDGETS
    }
    if set(v2) != expected_v2:
        raise ValueError("V2 learned selection coverage mismatch")
    v1_raw = load_selections(v1_paths)
    v1 = {}
    for (pair_group, method, budget), row in v1_raw.items():
        if method not in ("hgkv_v1_singleton", "hgkv_singleton"):
            continue
        normalized = dict(row)
        normalized["method"] = "hgkv_v1_singleton"
        v1[(pair_group, "hgkv_v1_singleton", budget)] = normalized
    expected_v1 = {
        (pair_group, "hgkv_v1_singleton", budget)
        for pair_group in states
        for budget in BUDGETS
    }
    if set(v1) != expected_v1:
        raise ValueError("V1 singleton selection coverage mismatch")
    cache = load_cache(cache_paths)
    teacher = teacher_selections(
        states=states,
        teacher_run_root=args.teacher_run_root,
        cache=cache,
    )
    exact = load_selections(
        exact_paths, allowed_methods={"exact_oracle"}
    )
    similarity = similarity_by_state(
        states=states,
        source_root=args.source_root,
    )
    rows = build_plan(
        states=states,
        v2=v2,
        v1=v1,
        teacher=teacher,
        exact=exact,
        similarity=similarity,
        random_seed=args.random_seed,
    )
    if any(
        not math.isfinite(score)
        for row in rows
        for _event, score in row.get("scores_by_event_step", [])
    ):
        raise ValueError("similarity baseline emitted non-finite score")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema_version": (
            "causalcache.hgkv_selector_v2.gate_plan_manifest.v1"
        ),
        "source_commit": args.source_commit,
        "states": len(states),
        "rows": len(rows),
        "budgets": list(BUDGETS),
        "methods": sorted({row["method"] for row in rows}),
        "random_seed": args.random_seed,
        "output_sha256": sha256_file(args.output),
        "state_sha256": {
            str(path): sha256_file(path) for path in state_paths
        },
        "selection_sha256": {
            str(path): sha256_file(path)
            for path in sorted(set(v2_paths + v1_paths + exact_paths))
        },
        "cache_sha256": {
            str(path): sha256_file(path) for path in cache_paths
        },
        "status": "DONE",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.manifest, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
