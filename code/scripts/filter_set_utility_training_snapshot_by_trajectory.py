#!/usr/bin/env python3
"""Freeze deterministic train/tune trajectory fractions from one snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_STATUS = "COMPLETED_MERGED_VARIABLE_HISTORY_TRAINING_INPUT_SNAPSHOT"


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _select_trajectories(
    trajectory_ids: set[str], *, fraction: float, seed: str
) -> set[str]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("trajectory fractions must be in (0, 1]")
    count = max(1, math.floor(len(trajectory_ids) * fraction + 0.5))
    ranked = sorted(
        trajectory_ids,
        key=lambda trajectory_id: (
            hashlib.sha256(f"{seed}:{trajectory_id}".encode()).hexdigest(),
            trajectory_id,
        ),
    )
    return set(ranked[:count])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--selection-seed", required=True)
    parser.add_argument("--train-fraction", type=float, required=True)
    parser.add_argument("--tune-fraction", type=float, default=1.0)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("filtered training snapshot output already exists")

    input_manifest = _read_json(args.input_root / "manifest.json")
    if (
        input_manifest.get("status") != EXPECTED_STATUS
        or input_manifest.get("evaluation_labels_included") is not False
    ):
        raise ValueError("training snapshot status or split firewall drifted")
    states = _read_jsonl(args.input_root / input_manifest["states_jsonl"])
    texts = {
        row["key"]: row["text"]
        for row in _read_jsonl(args.input_root / input_manifest["texts_jsonl"])
    }
    roles_by_trajectory: dict[str, str] = {}
    for state in states:
        role = state["role"]
        if role not in {"train", "tune"}:
            raise ValueError("filtered snapshot received a non-train/tune role")
        prior = roles_by_trajectory.setdefault(state["trajectory_id"], role)
        if prior != role:
            raise ValueError("one trajectory appears in more than one split")

    trajectory_ids = {
        role: {
            trajectory_id
            for trajectory_id, observed_role in roles_by_trajectory.items()
            if observed_role == role
        }
        for role in ("train", "tune")
    }
    selected = {
        "train": _select_trajectories(
            trajectory_ids["train"],
            fraction=args.train_fraction,
            seed=f"{args.selection_seed}:train",
        ),
        "tune": _select_trajectories(
            trajectory_ids["tune"],
            fraction=args.tune_fraction,
            seed=f"{args.selection_seed}:tune",
        ),
    }
    selected_states = tuple(
        state
        for state in states
        if state["trajectory_id"] in selected[state["role"]]
    )
    if not selected_states or {state["role"] for state in selected_states} != {
        "train",
        "tune",
    }:
        raise ValueError("trajectory filtering removed an entire split")
    required_text_keys = {
        key
        for state in selected_states
        for key in (state["instruction_text_key"], *state["event_text_keys"])
    }
    if not required_text_keys.issubset(texts):
        raise ValueError("filtered states reference an absent text payload")

    state_payload = b"".join(
        _canonical_json(state)
        for state in sorted(selected_states, key=lambda row: row["state_id"])
    )
    text_payload = b"".join(
        _canonical_json({"key": key, "text": texts[key]})
        for key in sorted(required_text_keys)
    )
    _write_atomic(args.output_root / "states.jsonl", state_payload)
    _write_atomic(args.output_root / "texts.jsonl", text_payload)
    manifest = {
        "content_sha256": hashlib.sha256(state_payload + text_payload).hexdigest(),
        "evaluation_labels_included": False,
        "label_source_revisions": input_manifest["label_source_revisions"],
        "numeric_feature_names": input_manifest["numeric_feature_names"],
        "parent_content_sha256": input_manifest["content_sha256"],
        "parent_snapshot_id": input_manifest["snapshot_id"],
        "role_counts": dict(
            sorted(Counter(state["role"] for state in selected_states).items())
        ),
        "schema_version": "1.0.0",
        "selection": {
            "contract": "TUNING_ONLY_FROZEN_TRAJECTORY_FRACTION",
            "seed": args.selection_seed,
            "selected_trajectory_ids": {
                role: sorted(selected[role]) for role in ("train", "tune")
            },
            "source_trajectory_counts": {
                role: len(trajectory_ids[role]) for role in ("train", "tune")
            },
            "trajectory_fractions": {
                "train": args.train_fraction,
                "tune": args.tune_fraction,
            },
        },
        "snapshot_id": args.snapshot_id,
        "source_manifest_sha256": input_manifest["source_manifest_sha256"],
        "state_count": len(selected_states),
        "states_jsonl": "states.jsonl",
        "states_sha256": hashlib.sha256(state_payload).hexdigest(),
        "status": EXPECTED_STATUS,
        "text_count": len(required_text_keys),
        "texts_jsonl": "texts.jsonl",
        "texts_sha256": hashlib.sha256(text_payload).hexdigest(),
        "trajectory_count": sum(len(rows) for rows in selected.values()),
        "visual_logical_shards": sorted(
            {int(state["logical_shard"]) for state in selected_states}
        ),
        "visual_token_profile": input_manifest["visual_token_profile"],
    }
    _write_atomic(args.output_root / "manifest.json", _canonical_json(manifest))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
