#!/usr/bin/env python3
"""Build budget-replicated prefix groups for the unified V2 selector."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from causalcache.hgkv_selector_v2 import V2_FEATURE_DIM


BUDGET_REPLICATION = {
    0: (1, 2, 4),
    1: (1, 3),
    2: (2,),
    3: (1,),
}


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
                raise ValueError(f"{path}:{line_no} duplicate state")
            if row["split"] not in ("train", "dev"):
                raise ValueError(f"{path}:{line_no} invalid split")
            states[pair_group] = row
    return states


def load_feature_keys(
    paths: Iterable[Path],
) -> tuple[dict[tuple[str, int], int], str]:
    keys: set[tuple[str, int]] = set()
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row["pair_group"]),
                int(row["singleton_event_step_id"]),
            )
            if key in keys:
                raise ValueError(f"{path}:{line_no} duplicate feature key {key}")
            if (
                int(row.get("feature_dim", len(row["feature"])))
                != V2_FEATURE_DIM
                or len(row["feature"]) != V2_FEATURE_DIM
            ):
                raise ValueError(f"{path}:{line_no} feature dimension drifted")
            keys.add(key)
    ordered = sorted(keys)
    packed = "".join(f"{pair_group}\t{candidate}\n" for pair_group, candidate in ordered)
    return {key: index for index, key in enumerate(ordered)}, hashlib.sha256(
        packed.encode("utf-8")
    ).hexdigest()


def load_labels(paths: Iterable[Path]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row["pair_group"]),
                tuple(int(value) for value in row["selected_event_step_ids"]),
            )
            if key in seen:
                raise ValueError(f"{path}:{line_no} duplicate prefix group {key}")
            seen.add(key)
            labels.append(row)
    return labels


def build_training_rows(
    *,
    states: Mapping[str, Mapping[str, Any]],
    feature_index: Mapping[tuple[str, int], int],
    labels: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label in labels:
        pair_group = str(label["pair_group"])
        state = states.get(pair_group)
        if state is None:
            raise KeyError(f"teacher label has no V2 state {pair_group}")
        inventory = [int(value) for value in state["candidate_event_step_ids"]]
        if inventory != [
            int(value) for value in label["candidate_event_step_ids"]
        ]:
            raise ValueError(f"{pair_group} teacher inventory drifted")
        selected = [int(value) for value in label["selected_event_step_ids"]]
        remaining = [
            int(value)
            for value in label["remaining_candidate_event_step_ids"]
        ]
        expected_remaining = [
            value for value in inventory if value not in selected
        ]
        if remaining != expected_remaining:
            raise ValueError(f"{pair_group}/{selected} remaining set drifted")
        targets = [float(value) for value in label["marginal_targets"]]
        if len(targets) != len(remaining):
            raise ValueError(f"{pair_group}/{selected} target width drifted")
        depth = int(label["depth"])
        if depth != len(selected):
            raise ValueError(f"{pair_group}/{selected} depth drifted")
        selected_indices = [
            feature_index[(pair_group, candidate)] for candidate in selected
        ]
        candidate_indices = [
            feature_index[(pair_group, candidate)] for candidate in remaining
        ]
        for remaining_budget in BUDGET_REPLICATION[depth]:
            result.append(
                {
                    "schema_version": (
                        "causalcache.hgkv_selector_v2.training_group.v1"
                    ),
                    "episode": str(state["episode"]),
                    "pair_group": pair_group,
                    "split": str(state["split"]),
                    "inventory_event_step_ids": inventory,
                    "selected_event_step_ids": selected,
                    "candidate_event_step_ids": remaining,
                    "selected_feature_indices": selected_indices,
                    "candidate_feature_indices": candidate_indices,
                    "marginal_targets": targets,
                    "remaining_budget": remaining_budget,
                    "depth": depth,
                    "stop_is_optimal": bool(label["stop_is_optimal"]),
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", action="append", required=True)
    parser.add_argument("--features", action="append", required=True)
    parser.add_argument("--teacher-labels", action="append", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    state_paths = expand_paths(args.states)
    feature_paths = expand_paths(args.features)
    label_paths = expand_paths(args.teacher_labels)
    if not state_paths or not feature_paths or not label_paths:
        parser.error("state, feature, and teacher-label patterns must match")
    states = load_states(state_paths)
    feature_index, feature_index_sha256 = load_feature_keys(feature_paths)
    required_features = {
        (pair_group, int(candidate))
        for pair_group, state in states.items()
        for candidate in state["candidate_event_step_ids"]
    }
    if set(feature_index) != required_features:
        raise ValueError(
            "feature coverage mismatch: "
            f"missing={len(required_features - set(feature_index))} "
            f"unexpected={len(set(feature_index) - required_features)}"
        )
    rows = build_training_rows(
        states=states,
        feature_index=feature_index,
        labels=load_labels(label_paths),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema_version": (
            "causalcache.hgkv_selector_v2.training_data_manifest.v1"
        ),
        "source_commit": args.source_commit,
        "states": len(states),
        "features": len(feature_index),
        "feature_index_sha256": feature_index_sha256,
        "prefix_groups_before_budget_replication": len(
            {
                (
                    row["pair_group"],
                    tuple(row["selected_event_step_ids"]),
                )
                for row in rows
            }
        ),
        "training_groups": len(rows),
        "groups_by_split": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "dev")
        },
        "groups_by_depth": {
            str(depth): sum(row["depth"] == depth for row in rows)
            for depth in range(4)
        },
        "state_sha256": {
            str(path): sha256_file(path) for path in state_paths
        },
        "feature_sha256": {
            str(path): sha256_file(path) for path in feature_paths
        },
        "label_sha256": {
            str(path): sha256_file(path) for path in label_paths
        },
        "output_sha256": sha256_file(args.output),
        "status": "DONE",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
