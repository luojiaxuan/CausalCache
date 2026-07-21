#!/usr/bin/env python3
"""Materialize missing train-only direct/recent/hybrid expansion schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from causalcache.set_utility_direct_on_policy import (
    candidate_complete_coalitions,
    collection_bases,
    complete_group_count,
    missing_candidate_complete_coalitions,
    runner_schedule_coalitions,
    validate_direct_train_selection_record,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file
from causalcache.set_utility_variable_history import history_bin


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_atomic(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _selection_records(
    paths: tuple[Path, ...],
    *,
    expected_collection_sha: str,
    expected_input_content_sha: str,
    expected_checkpoint_sha: str,
) -> tuple[dict[str, dict[str, Any]], tuple[dict[str, Any], ...]]:
    records: dict[str, dict[str, Any]] = {}
    receipts = []
    binding = None
    for path in paths:
        payload = _read_json(path)
        current_binding = (
            payload.get("cache_content_sha256"),
            payload.get("checkpoint_sha256"),
            payload.get("collection_config_sha256"),
            payload.get("config_sha256"),
            payload.get("input_content_sha256"),
            payload.get("role"),
            payload.get("status"),
            payload.get("variant"),
        )
        if (
            payload.get("schema_version")
            != "causalcache.direct_marginal_selections.v2"
            or payload.get("role") != "train"
            or payload.get("status") != "COMPLETED_SET_UTILITY_TRAIN_SELECTIONS"
            or payload.get("collection_config_sha256") != expected_collection_sha
            or payload.get("input_content_sha256") != expected_input_content_sha
            or payload.get("checkpoint_sha256") != expected_checkpoint_sha
        ):
            raise ValueError("direct train selection binding drifted")
        if binding is None:
            binding = current_binding
        elif current_binding != binding:
            raise ValueError("direct train selection shards have different bindings")
        for record in payload.get("records", ()):
            validate_direct_train_selection_record(record)
            state_id = record.get("state_id")
            if not isinstance(state_id, str) or state_id in records:
                raise ValueError("direct train selection state is invalid or duplicated")
            records[state_id] = record
        receipts.append(
            {
                "content_sha256": payload["content_sha256"],
                "file_sha256": sha256_file(path),
                "path": str(path),
                "state_count": len(payload.get("records", ())),
            }
        )
    return records, tuple(receipts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("direct on-policy schedule output already exists")
    config_path = args.config.resolve()
    config = _read_json(config_path)
    collection = config["on_policy_collection"]
    input_root = args.input_root.resolve()
    input_manifest_path = input_root / "manifest.json"
    input_manifest = _read_json(input_manifest_path)
    if (
        input_manifest.get("evaluation_labels_included") is not False
        or input_manifest.get("content_sha256")
        != config["input"]["training_input_content_sha256"]
    ):
        raise ValueError("direct on-policy input identity or firewall drifted")
    states = {
        row["state_id"]: row
        for line in (input_root / input_manifest["states_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line
        for row in (json.loads(line),)
        if row["role"] == "train"
    }
    if len(states) != int(collection["expected_train_state_count"]):
        raise ValueError("direct on-policy train state inventory drifted")
    records, selection_receipts = _selection_records(
        tuple(path.resolve() for path in args.selection),
        expected_collection_sha=sha256_file(config_path),
        expected_input_content_sha=input_manifest["content_sha256"],
        expected_checkpoint_sha=config["input"]["checkpoint_sha256"],
    )
    if set(records) != set(states):
        raise ValueError("direct train selection does not cover every train state")

    maximum_base_cardinality = int(collection["maximum_base_cardinality"])
    by_shard = {index: [] for index in range(256)}
    history_counts = Counter()
    scheduled_history_counts = Counter()
    source_counts = Counter()
    existing_desired_count = 0
    desired_coalition_count = 0
    desired_group_count = 0
    existing_complete_group_count = 0
    resulting_complete_group_count = 0
    scheduled_coalition_count = 0
    scheduled_full_anchor_count = 0
    scheduled_state_count = 0
    uncovered_state_count = 0
    for state_id, state in sorted(states.items()):
        record = records[state_id]
        candidates = tuple(state["candidate_event_step_ids"])
        if (
            tuple(record["candidate_event_ids"]) != candidates
            or record["trajectory_id"] != state["trajectory_id"]
            or record["logical_shard"] != state["logical_shard"]
        ):
            raise ValueError("direct selection and state identity differ")
        bin_name = history_bin(len(candidates))
        history_counts[bin_name] += 1
        existing = tuple(
            tuple(row["coalition_event_step_ids"])
            for row in state["distance_rows"]
        )
        desired = candidate_complete_coalitions(
            record, maximum_base_cardinality=maximum_base_cardinality
        )
        bases = collection_bases(
            record, maximum_base_cardinality=maximum_base_cardinality
        )
        missing = missing_candidate_complete_coalitions(
            record,
            existing,
            maximum_base_cardinality=maximum_base_cardinality,
        )
        desired_coalition_count += len(desired)
        desired_group_count += len(bases)
        existing_desired_count += len(set(desired).intersection(existing))
        state_existing_complete_groups = complete_group_count(
            candidates,
            existing,
            maximum_base_cardinality=maximum_base_cardinality,
        )
        existing_complete_group_count += state_existing_complete_groups
        if state_existing_complete_groups > 0:
            resulting_complete_group_count += state_existing_complete_groups
            continue
        uncovered_state_count += 1
        if not missing:
            continue
        resulting = tuple(set(existing).union(desired))
        resulting_complete_group_count += complete_group_count(
            candidates,
            resulting,
            maximum_base_cardinality=maximum_base_cardinality,
        )
        scheduled_state_count += 1
        scheduled_history_counts[bin_name] += 1
        scheduled_coalition_count += len(missing)
        source_counts.update(row["source"] for row in missing)
        runner_coalitions = runner_schedule_coalitions(candidates, missing)
        if len(runner_coalitions) == len(missing) + 1:
            scheduled_full_anchor_count += 1
        by_shard[state["logical_shard"]].append(
            {
                "candidate_event_ids": list(candidates),
                "coalitions": list(runner_coalitions),
                "exact": False,
                "logical_shard": state["logical_shard"],
                "role": "train",
                "state_id": state_id,
                "trajectory_id": state["trajectory_id"],
            }
        )

    if uncovered_state_count != int(collection["expected_uncovered_state_count"]):
        raise ValueError("direct on-policy uncovered state inventory drifted")

    config_sha = sha256_file(config_path)
    selection_inventory_sha = hashlib.sha256(
        canonical_json_bytes(selection_receipts)
    ).hexdigest()
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
                        "selection_inventory_sha256": selection_inventory_sha,
                    }
                )
            ).hexdigest(),
            "logical_shard": logical_shard,
            "role_state_counts": {"train": len(rows)} if rows else {},
            "schedule_byte_count": len(payload),
            "schedule_sha256": hashlib.sha256(payload).hexdigest(),
            "state_count": len(rows),
            "status": "COMPLETED_DIRECT_ON_POLICY_SCHEDULE_SHARD",
        }
        _write_atomic(
            args.output_root / "receipts" / f"shard-{logical_shard:03d}-of-256.json",
            receipt,
        )
        receipts.append(receipt)
    manifest = {
        "config_sha256": config_sha,
        "content_counts": {
            "desired_coalitions": desired_coalition_count,
            "desired_complete_groups": desired_group_count,
            "existing_complete_groups": existing_complete_group_count,
            "existing_desired_coalitions": existing_desired_count,
            "resulting_complete_groups": resulting_complete_group_count,
            "scheduled_missing_coalitions": scheduled_coalition_count,
            "scheduled_zero_cost_full_anchors": scheduled_full_anchor_count,
        },
        "history_bin_counts": dict(sorted(history_counts.items())),
        "input_content_sha256": input_manifest["content_sha256"],
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "schema_version": "1.0.0",
        "scheduled_history_bin_counts": dict(sorted(scheduled_history_counts.items())),
        "scheduled_state_count": scheduled_state_count,
        "selection_inventory_sha256": selection_inventory_sha,
        "selection_receipts": list(selection_receipts),
        "source_counts": dict(sorted(source_counts.items())),
        "state_count": len(states),
        "uncovered_state_count": uncovered_state_count,
        "status": "COMPLETED_SET_UTILITY_DIRECT_ON_POLICY_SCHEDULES",
    }
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    _write_atomic(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
