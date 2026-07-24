#!/usr/bin/env python3
"""Plan one exact-cache-aware depth of the V2 true-utility teacher beam."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from causalcache.hgkv_selector_v2 import teacher_depth_expansions


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(Path(value) for value in glob.glob(pattern))
    return sorted(paths)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_states(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_group = str(row["pair_group"])
            if pair_group in states:
                raise ValueError(
                    f"{path}:{line_no} duplicate state {pair_group}"
                )
            if row.get("recent_candidate_cap") is not None:
                raise ValueError(f"{path}:{line_no} recent cap is forbidden")
            states[pair_group] = row
    return states


def load_cache(
    paths: Iterable[Path],
) -> dict[tuple[str, str], dict[str, Any]]:
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["pair_group"]), str(row["restored_set_key"]))
            previous = cache.get(key)
            if previous is not None and (
                previous["target_action_sha256"]
                != row["target_action_sha256"]
                or abs(float(previous["u_act"]) - float(row["u_act"])) > 1e-8
            ):
                raise ValueError(
                    f"{path}:{line_no} conflicting exact coalition {key}"
                )
            cache[key] = row
    return cache


def load_prefixes(
    paths: Iterable[Path], *, depth: int
) -> dict[str, list[list[int]]]:
    if depth == 0:
        if list(paths):
            raise ValueError("depth 0 must not receive a prefix file")
        return {}
    prefixes: dict[str, list[list[int]]] = {}
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_group = str(row["pair_group"])
            if pair_group in prefixes:
                raise ValueError(
                    f"{path}:{line_no} duplicate beam state {pair_group}"
                )
            if int(row["beam_depth"]) != depth:
                raise ValueError(
                    f"{path}:{line_no} beam depth does not equal {depth}"
                )
            prefixes[pair_group] = [
                [int(value) for value in prefix]
                for prefix in row["prefixes"]
            ]
    return prefixes


def build_plan_rows(
    *,
    states: Mapping[str, Mapping[str, Any]],
    cache: Mapping[tuple[str, str], Mapping[str, Any]],
    prefixes_by_state: Mapping[str, list[list[int]]],
    depth: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_group, state in sorted(states.items()):
        prefixes = (
            [[]] if depth == 0 else prefixes_by_state.get(pair_group)
        )
        if prefixes is None:
            raise KeyError(f"missing depth-{depth} beam for {pair_group}")
        expansions = teacher_depth_expansions(
            state["candidate_event_step_ids"],
            prefixes,
            depth=depth,
        )
        for expansion in expansions:
            key = (pair_group, expansion["restored_set_key"])
            cached = cache.get(key)
            rows.append(
                {
                    "schema_version": (
                        "causalcache.hgkv_selector_v2.teacher_plan_edge.v1"
                    ),
                    "episode": str(state["episode"]),
                    "pair_group": pair_group,
                    "decision_step": int(state["decision_step"]),
                    "history_length": int(state["history_length"]),
                    "candidate_event_step_ids": [
                        int(value)
                        for value in state["candidate_event_step_ids"]
                    ],
                    **expansion,
                    "cache_hit": cached is not None,
                    "cached_u_act": (
                        None if cached is None else float(cached["u_act"])
                    ),
                }
            )
    return rows


def plan_manifest(
    rows: list[Mapping[str, Any]],
    *,
    depth: int,
    beam_width: int,
    source_commit: str,
    state_paths: list[Path],
    cache_paths: list[Path],
    prefix_paths: list[Path],
) -> dict[str, Any]:
    unique = {
        (str(row["pair_group"]), str(row["restored_set_key"]))
        for row in rows
    }
    missing = {
        (str(row["pair_group"]), str(row["restored_set_key"]))
        for row in rows
        if not row["cache_hit"]
    }
    return {
        "schema_version": (
            "causalcache.hgkv_selector_v2.teacher_plan_manifest.v1"
        ),
        "depth": depth,
        "edge": f"edge{depth}",
        "beam_width": beam_width,
        "states": len({str(row["pair_group"]) for row in rows}),
        "edge_rows": len(rows),
        "unique_child_coalitions": len(unique),
        "cached_child_coalitions": len(unique - missing),
        "missing_child_coalitions": len(missing),
        "source_commit": source_commit,
        "state_sha256": {
            str(path): sha256_file(path) for path in state_paths
        },
        "cache_sha256": {
            str(path): sha256_file(path) for path in cache_paths
        },
        "prefix_sha256": {
            str(path): sha256_file(path) for path in prefix_paths
        },
        "status": "DONE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", action="append", required=True)
    parser.add_argument("--coalition-cache", action="append", required=True)
    parser.add_argument("--prefixes", action="append", default=[])
    parser.add_argument("--depth", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.beam_width != 4:
        parser.error("the frozen first V2 protocol requires beam width 4")
    state_paths = expand_paths(args.states)
    cache_paths = expand_paths(args.coalition_cache)
    prefix_paths = expand_paths(args.prefixes)
    if not state_paths or not cache_paths:
        parser.error("state and coalition-cache patterns must match files")
    prefixes = load_prefixes(prefix_paths, depth=args.depth)
    rows = build_plan_rows(
        states=load_states(state_paths),
        cache=load_cache(cache_paths),
        prefixes_by_state=prefixes,
        depth=args.depth,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = plan_manifest(
        rows,
        depth=args.depth,
        beam_width=args.beam_width,
        source_commit=args.source_commit,
        state_paths=state_paths,
        cache_paths=cache_paths,
        prefix_paths=prefix_paths,
    )
    manifest["output_sha256"] = sha256_file(args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
