#!/usr/bin/env python3
"""Materialize targeted train-only on-policy restoration schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file
from causalcache.set_utility_train_on_policy import (
    enrichment_coalitions,
    select_targeted_states,
    validate_train_selections,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("selection must be name=path")
    return name, Path(raw_path)


def _write_atomic(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=_named_path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("train on-policy schedule output already exists")
    config = _read_json(args.config)
    input_manifest_path = args.input_root / "manifest.json"
    input_manifest = _read_json(input_manifest_path)
    if input_manifest.get("evaluation_labels_included") is not False:
        raise ValueError("train enrichment crossed the evaluation firewall")
    states = {
        row["state_id"]: row
        for line in (args.input_root / input_manifest["states_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line
        for row in (json.loads(line),)
        if row["role"] == "train"
    }
    selections = {name: _read_json(path) for name, path in args.selection}
    if len(selections) != len(args.selection):
        raise ValueError("selection names must be unique")
    records_by_model = validate_train_selections(selections)
    if set(records_by_model["set_transformer"]) != set(states):
        raise ValueError("train selector inventory does not match input states")
    selected_ids, selection_summary = select_targeted_states(
        selections,
        target_fraction=float(config["target_fraction"]),
        history_bin_weights=config["history_bin_weights"],
        maximum_states_per_trajectory=int(config["maximum_states_per_trajectory"]),
    )
    by_shard = {index: [] for index in range(256)}
    source_counts = Counter()
    coalition_count = 0
    for state_id in selected_ids:
        state = states[state_id]
        coalitions = enrichment_coalitions(state_id, records_by_model)
        source_counts.update(row["source"] for row in coalitions)
        coalition_count += len(coalitions)
        by_shard[state["logical_shard"]].append(
            {
                "candidate_event_ids": state["candidate_event_step_ids"],
                "coalitions": list(coalitions),
                "exact": False,
                "logical_shard": state["logical_shard"],
                "role": "train",
                "state_id": state_id,
                "trajectory_id": state["trajectory_id"],
            }
        )
    selection_bindings = {
        name: {
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "content_sha256": payload["content_sha256"],
            "variant": payload["variant"],
        }
        for name, payload in sorted(selections.items())
    }
    config_sha = sha256_file(args.config)
    receipts = []
    for logical_shard in range(256):
        rows = sorted(by_shard[logical_shard], key=lambda row: row["state_id"])
        payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        relative = f"schedule-shards/shard-{logical_shard:03d}-of-256.jsonl"
        path = args.output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        receipt = {
            "coalition_count": sum(len(row["coalitions"]) for row in rows),
            "identity_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        "config_sha256": config_sha,
                        "input_manifest_sha256": sha256_file(input_manifest_path),
                        "logical_shard": logical_shard,
                        "selection_bindings": selection_bindings,
                    }
                )
            ).hexdigest(),
            "logical_shard": logical_shard,
            "role_state_counts": {"train": len(rows)} if rows else {},
            "schedule_byte_count": len(payload),
            "schedule_sha256": hashlib.sha256(payload).hexdigest(),
            "state_count": len(rows),
            "status": "COMPLETED_SET_UTILITY_TRAIN_ON_POLICY_SCHEDULE_SHARD",
        }
        _write_atomic(
            args.output_root / "receipts" / f"shard-{logical_shard:03d}-of-256.json",
            receipt,
        )
        receipts.append(receipt)
    manifest = {
        "coalition_count": coalition_count,
        "config_sha256": config_sha,
        "input_content_sha256": input_manifest["content_sha256"],
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "schema_version": "1.0.0",
        "selection_bindings": selection_bindings,
        "selection_summary": selection_summary,
        "source_counts": dict(sorted(source_counts.items())),
        "state_count": len(selected_ids),
        "status": "COMPLETED_SET_UTILITY_TRAIN_ON_POLICY_SCHEDULES",
    }
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    _write_atomic(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
