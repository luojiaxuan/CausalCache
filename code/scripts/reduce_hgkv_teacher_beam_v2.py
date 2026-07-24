#!/usr/bin/env python3
"""Reduce one V2 teacher depth into true marginals and the next true-U beam."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from causalcache.hgkv_selector_v2 import (
    restored_set_key,
    select_true_u_beam,
)
from scripts.plan_hgkv_teacher_beam_v2 import (
    expand_paths,
    load_cache,
    sha256_file,
)


def load_plan(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != (
                "causalcache.hgkv_selector_v2.teacher_plan_edge.v1"
            ):
                raise ValueError(f"{path}:{line_no} invalid plan schema")
            rows.append(row)
    return rows


def reduce_depth(
    *,
    plan_rows: list[Mapping[str, Any]],
    cache: Mapping[tuple[str, str], Mapping[str, Any]],
    depth: int,
    beam_width: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edge_groups: dict[
        tuple[str, tuple[int, ...]], list[Mapping[str, Any]]
    ] = defaultdict(list)
    state_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in plan_rows:
        if int(row["depth"]) != depth:
            raise ValueError("plan contains the wrong teacher depth")
        pair_group = str(row["pair_group"])
        prefix = tuple(int(value) for value in row["selected_event_step_ids"])
        edge_groups[(pair_group, prefix)].append(row)
        state_rows[pair_group].append(row)

    labels: list[dict[str, Any]] = []
    for (pair_group, prefix), rows in sorted(edge_groups.items()):
        candidates = tuple(
            sorted(int(row["candidate_event_step_id"]) for row in rows)
        )
        inventory = tuple(
            int(value) for value in rows[0]["candidate_event_step_ids"]
        )
        expected_candidates = tuple(
            value for value in inventory if value not in prefix
        )
        if candidates != expected_candidates:
            raise ValueError(
                f"{pair_group}/{prefix} does not cover every remaining candidate"
            )
        prefix_key = restored_set_key(prefix)
        prefix_row = cache.get((pair_group, prefix_key))
        if prefix_row is None:
            raise KeyError(f"missing prefix utility for {pair_group}/{prefix_key}")
        prefix_u = float(prefix_row["u_act"])
        child_utilities: list[float] = []
        marginal_targets: list[float] = []
        for row in sorted(rows, key=lambda item: item["candidate_event_step_id"]):
            child_key = str(row["restored_set_key"])
            child = cache.get((pair_group, child_key))
            if child is None:
                raise KeyError(
                    f"missing child utility for {pair_group}/{child_key}"
                )
            child_u = float(child["u_act"])
            marginal = child_u - prefix_u
            if not math.isfinite(marginal):
                raise ValueError("teacher marginal must be finite")
            child_utilities.append(child_u)
            marginal_targets.append(marginal)
        labels.append(
            {
                "schema_version": (
                    "causalcache.hgkv_selector_v2.teacher_prefix_group.v1"
                ),
                "episode": str(rows[0]["episode"]),
                "pair_group": pair_group,
                "candidate_event_step_ids": list(inventory),
                "selected_event_step_ids": list(prefix),
                "remaining_candidate_event_step_ids": list(candidates),
                "child_u_act": child_utilities,
                "marginal_targets": marginal_targets,
                "prefix_u_act": prefix_u,
                "stop_marginal": 0.0,
                "stop_is_optimal": max(marginal_targets) <= 0.0,
                "depth": depth,
                "edge": f"edge{depth}",
            }
        )

    next_beam: list[dict[str, Any]] = []
    if depth < 3:
        for pair_group, rows in sorted(state_rows.items()):
            utility_by_set_key: dict[str, float] = {}
            for row in rows:
                key = str(row["restored_set_key"])
                cached = cache.get((pair_group, key))
                if cached is None:
                    raise KeyError(
                        f"missing child utility for {pair_group}/{key}"
                    )
                utility_by_set_key[key] = float(cached["u_act"])
            prefixes = select_true_u_beam(
                rows,
                utility_by_set_key,
                beam_width=beam_width,
            )
            next_beam.append(
                {
                    "schema_version": (
                        "causalcache.hgkv_selector_v2.teacher_beam.v1"
                    ),
                    "episode": str(rows[0]["episode"]),
                    "pair_group": pair_group,
                    "beam_depth": depth + 1,
                    "beam_width": beam_width,
                    "prefixes": [list(prefix) for prefix in prefixes],
                    "true_u_act": [
                        utility_by_set_key[restored_set_key(prefix)]
                        for prefix in prefixes
                    ],
                }
            )
    return labels, next_beam


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="append", required=True)
    parser.add_argument("--coalition-cache", action="append", required=True)
    parser.add_argument("--depth", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--labels-output", type=Path, required=True)
    parser.add_argument("--beam-output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.beam_width != 4:
        parser.error("the frozen first V2 protocol requires beam width 4")
    if args.depth < 3 and args.beam_output is None:
        parser.error("depths 0-2 require --beam-output")
    if args.depth == 3 and args.beam_output is not None:
        parser.error("depth 3 does not produce another beam")
    plan_paths = expand_paths(args.plan)
    cache_paths = expand_paths(args.coalition_cache)
    if not plan_paths or not cache_paths:
        parser.error("plan and cache patterns must match files")
    labels, beam = reduce_depth(
        plan_rows=load_plan(plan_paths),
        cache=load_cache(cache_paths),
        depth=args.depth,
        beam_width=args.beam_width,
    )
    _write_jsonl(args.labels_output, labels)
    if args.beam_output is not None:
        _write_jsonl(args.beam_output, beam)
    manifest = {
        "schema_version": (
            "causalcache.hgkv_selector_v2.teacher_reduce_manifest.v1"
        ),
        "depth": args.depth,
        "edge": f"edge{args.depth}",
        "beam_width": args.beam_width,
        "prefix_groups": len(labels),
        "stop_optimal_groups": sum(row["stop_is_optimal"] for row in labels),
        "marginal_rows": sum(
            len(row["marginal_targets"]) for row in labels
        ),
        "next_beam_states": len(beam),
        "source_commit": args.source_commit,
        "plan_sha256": {
            str(path): sha256_file(path) for path in plan_paths
        },
        "cache_sha256": {
            str(path): sha256_file(path) for path in cache_paths
        },
        "labels_sha256": sha256_file(args.labels_output),
        "beam_sha256": (
            None
            if args.beam_output is None
            else sha256_file(args.beam_output)
        ),
        "status": "DONE",
    }
    packed = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest["manifest_payload_sha256"] = hashlib.sha256(packed).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
