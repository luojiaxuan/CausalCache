#!/usr/bin/env python3
"""Resume the CPU control plane for the four-depth V2 teacher beam."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.plan_hgkv_teacher_beam_v2 import (
    build_plan_rows,
    expand_paths,
    load_cache,
    load_prefixes,
    load_states,
    plan_manifest,
    sha256_file,
)
from scripts.reduce_hgkv_teacher_beam_v2 import reduce_depth


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    partial.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]


def current_missing(
    plan_rows: list[Mapping[str, Any]],
    cache: Mapping[tuple[str, str], Mapping[str, Any]],
) -> set[tuple[str, str]]:
    required = {
        (str(row["pair_group"]), str(row["restored_set_key"]))
        for row in plan_rows
    }
    return required - set(cache)


def run(args: argparse.Namespace) -> dict[str, Any]:
    state_paths = expand_paths(args.states)
    cache_paths = expand_paths(args.coalition_cache)
    if not state_paths or not cache_paths:
        raise ValueError("state and cache patterns must match files")
    states = load_states(state_paths)
    cache = load_cache(cache_paths)
    args.run_root.mkdir(parents=True, exist_ok=True)
    completed_depths: list[int] = []
    previous_beam: Path | None = None
    for depth in range(4):
        depth_root = args.run_root / f"depth{depth}"
        plan_path = depth_root / "plan.jsonl"
        plan_manifest_path = depth_root / "plan-manifest.json"
        prefix_paths = [] if previous_beam is None else [previous_beam]
        if not plan_path.is_file():
            prefixes = load_prefixes(prefix_paths, depth=depth)
            plan_rows = build_plan_rows(
                states=states,
                cache=cache,
                prefixes_by_state=prefixes,
                depth=depth,
            )
            write_jsonl(plan_path, plan_rows)
            manifest = plan_manifest(
                plan_rows,
                depth=depth,
                beam_width=args.beam_width,
                source_commit=args.source_commit,
                state_paths=state_paths,
                cache_paths=cache_paths,
                prefix_paths=prefix_paths,
            )
            manifest["output_sha256"] = sha256_file(plan_path)
            write_json(plan_manifest_path, manifest)
        else:
            if not plan_manifest_path.is_file():
                raise FileNotFoundError(
                    f"existing plan lacks manifest: {plan_manifest_path}"
                )
            existing_plan_manifest = json.loads(
                plan_manifest_path.read_text(encoding="utf-8")
            )
            if (
                existing_plan_manifest["source_commit"] != args.source_commit
                or int(existing_plan_manifest["depth"]) != depth
                or existing_plan_manifest["output_sha256"]
                != sha256_file(plan_path)
            ):
                raise ValueError(f"stale or corrupted teacher plan {plan_path}")
            plan_rows = load_jsonl(plan_path)
        missing = current_missing(plan_rows, cache)
        if missing:
            status = {
                "schema_version": (
                    "causalcache.hgkv_selector_v2.teacher_runner_status.v1"
                ),
                "status": "AWAITING_RENDER_SCORE_CACHE_REFRESH",
                "depth": depth,
                "edge": f"edge{depth}",
                "missing_child_coalitions": len(missing),
                "plan": str(plan_path),
                "plan_sha256": sha256_file(plan_path),
                "source_commit": args.source_commit,
                "completed_depths": completed_depths,
            }
            write_json(args.run_root / "STATUS.json", status)
            return status

        labels_path = depth_root / "labels.jsonl"
        beam_path = None if depth == 3 else depth_root / "beam.jsonl"
        reduce_manifest_path = depth_root / "reduce-manifest.json"
        if not reduce_manifest_path.is_file():
            labels, beam = reduce_depth(
                plan_rows=plan_rows,
                cache=cache,
                depth=depth,
                beam_width=args.beam_width,
            )
            write_jsonl(labels_path, labels)
            if beam_path is not None:
                write_jsonl(beam_path, beam)
            reduce_manifest = {
                "schema_version": (
                    "causalcache.hgkv_selector_v2.teacher_reduce_manifest.v1"
                ),
                "depth": depth,
                "edge": f"edge{depth}",
                "beam_width": args.beam_width,
                "prefix_groups": len(labels),
                "stop_optimal_groups": sum(
                    row["stop_is_optimal"] for row in labels
                ),
                "marginal_rows": sum(
                    len(row["marginal_targets"]) for row in labels
                ),
                "next_beam_states": len(beam),
                "source_commit": args.source_commit,
                "plan_sha256": sha256_file(plan_path),
                "cache_sha256": {
                    str(path): sha256_file(path) for path in cache_paths
                },
                "labels_sha256": sha256_file(labels_path),
                "beam_sha256": (
                    None if beam_path is None else sha256_file(beam_path)
                ),
                "status": "DONE",
            }
            write_json(reduce_manifest_path, reduce_manifest)
        else:
            existing_reduce = json.loads(
                reduce_manifest_path.read_text(encoding="utf-8")
            )
            if (
                existing_reduce["source_commit"] != args.source_commit
                or existing_reduce["labels_sha256"] != sha256_file(labels_path)
                or (
                    beam_path is not None
                    and existing_reduce["beam_sha256"]
                    != sha256_file(beam_path)
                )
            ):
                raise ValueError(
                    f"stale or corrupted teacher reduction {depth_root}"
                )
        completed_depths.append(depth)
        previous_beam = beam_path

    done = {
        "schema_version": (
            "causalcache.hgkv_selector_v2.teacher_runner_done.v1"
        ),
        "status": "DONE",
        "completed_depths": completed_depths,
        "beam_width": args.beam_width,
        "states": len(states),
        "source_commit": args.source_commit,
        "cache_sha256": {
            str(path): sha256_file(path) for path in cache_paths
        },
    }
    write_json(args.run_root / "DONE", done)
    write_json(args.run_root / "STATUS.json", done)
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", action="append", required=True)
    parser.add_argument("--coalition-cache", action="append", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.beam_width != 4:
        parser.error("the frozen first V2 protocol requires beam width 4")
    result = run(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
