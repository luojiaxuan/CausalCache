"""Small-history exact-B4 schedule and oracle reduction helpers."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file


B4_SCHEDULE_STATUS = "COMPLETED_SET_UTILITY_B4_ORACLE_SCHEDULE"
B4_SHARD_STATUS = "COMPLETED_SET_UTILITY_B4_ORACLE_SCHEDULE_SHARD"


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _selection_content_is_valid(payload: Mapping[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    if not isinstance(claimed, str):
        return False
    unsigned = dict(payload)
    del unsigned["content_sha256"]
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() == claimed


def _schedule_row(
    record: Mapping[str, Any], *, coalition_cardinalities: Sequence[int]
) -> dict[str, Any]:
    candidates = tuple(record["candidate_event_ids"])
    coalitions = []
    for cardinality in coalition_cardinalities:
        for subset in itertools.combinations(candidates, cardinality):
            source = f"exact_b4:cardinality:{cardinality}"
            coalitions.append(
                {"event_ids": list(subset), "source": source, "sources": [source]}
            )
    coalitions.append(
        {
            "event_ids": list(candidates),
            "source": "anchor:full",
            "sources": ["anchor:full"],
        }
    )
    return {
        "candidate_event_ids": list(candidates),
        "coalitions": coalitions,
        "logical_shard": record["logical_shard"],
        "role": "evaluation",
        "state_id": record["state_id"],
        "tracks": ["b4_exact_small_history"],
        "trajectory_id": record["trajectory_id"],
    }


def materialize_b4_oracle_schedules(
    *,
    config_path: Path,
    selections_path: Path,
    source_manifest_path: Path,
    output_root: Path,
    workers: int,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("schedule worker count must be positive")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection_contract = config["inputs"]["sealed_selections"]
    if sha256_file(selections_path) != selection_contract["file_sha256"]:
        raise ValueError("B4 schedule selections file drifted")
    selections = json.loads(selections_path.read_text(encoding="utf-8"))
    if (
        selections.get("content_sha256") != selection_contract["content_sha256"]
        or not _selection_content_is_valid(selections)
    ):
        raise ValueError("B4 schedule sealed selection content drifted")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source.get("status") != "COMPLETED_VARIABLE_HISTORY_SOURCE":
        raise ValueError("B4 schedule source manifest is incomplete")
    shard_count = config["new_truth_schedule"]["logical_shard_count"]
    shards = {row["logical_shard"]: row for row in source["shards"]}
    if set(shards) != set(range(shard_count)):
        raise ValueError("B4 schedule source logical shards drifted")
    trajectory_shards = {}
    for logical_shard, shard in shards.items():
        for trajectory_id in shard["trajectory_ids"]:
            if trajectory_id in trajectory_shards:
                raise ValueError("B4 schedule source trajectory is duplicated")
            trajectory_shards[trajectory_id] = logical_shard

    state_contract = config["state_selection"]
    cardinalities = tuple(config["new_truth_schedule"]["coalition_cardinalities"])
    if (
        not cardinalities
        or tuple(sorted(set(cardinalities))) != cardinalities
        or cardinalities[0] < 0
        or cardinalities[-1] > state_contract["maximum_candidate_count"]
    ):
        raise ValueError("B4 schedule coalition cardinalities are invalid")
    minimum = state_contract["minimum_candidate_count"]
    maximum = state_contract["maximum_candidate_count"]
    selected = []
    for record in selections["records"]:
        candidate_count = len(record["candidate_event_ids"])
        if (
            state_contract["required_track"] in record["tracks"]
            and minimum <= candidate_count <= maximum
        ):
            if trajectory_shards.get(record["trajectory_id"]) != record["logical_shard"]:
                raise ValueError("B4 state/source logical shard drifted")
            selected.append(record)
    if len(selected) != state_contract["expected_state_count"]:
        raise ValueError("B4 selected state count drifted")
    if len({row["trajectory_id"] for row in selected}) != state_contract[
        "expected_trajectory_count"
    ]:
        raise ValueError("B4 selected trajectory count drifted")
    states_by_shard = {index: [] for index in range(shard_count)}
    for record in selected:
        states_by_shard[record["logical_shard"]].append(record)

    bindings = {
        "config_sha256": sha256_file(config_path),
        "selections_sha256": sha256_file(selections_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
    }

    def build(logical_shard: int) -> dict[str, Any]:
        rows = tuple(
            _schedule_row(record, coalition_cardinalities=cardinalities)
            for record in sorted(
                states_by_shard[logical_shard], key=lambda item: item["state_id"]
            )
        )
        payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        schedule_path = (
            output_root
            / "schedule-shards"
            / f"shard-{logical_shard:03d}-of-{shard_count}.jsonl"
        )
        receipt_path = (
            output_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-{shard_count}.json"
        )
        coalition_count = sum(len(row["coalitions"]) for row in rows)
        forward_count = sum(
            tuple(coalition["event_ids"]) != tuple(row["candidate_event_ids"])
            for row in rows
            for coalition in row["coalitions"]
        )
        receipt = {
            **bindings,
            "coalition_count": coalition_count,
            "forward_coalition_count": forward_count,
            "logical_shard": logical_shard,
            "schedule_byte_count": len(payload),
            "schedule_sha256": hashlib.sha256(payload).hexdigest(),
            "source_shard_sha256": shards[logical_shard]["sha256"],
            "state_count": len(rows),
            "status": B4_SHARD_STATUS,
        }
        expected_receipt = canonical_json_bytes(receipt, pretty=True)
        if schedule_path.exists() or receipt_path.exists():
            if (
                not schedule_path.is_file()
                or not receipt_path.is_file()
                or schedule_path.read_bytes() != payload
                or receipt_path.read_bytes() != expected_receipt
            ):
                raise ValueError("existing B4 schedule shard drifted")
            return receipt
        _write_atomic(schedule_path, payload)
        _write_atomic(receipt_path, expected_receipt)
        return receipt

    with ThreadPoolExecutor(max_workers=workers) as executor:
        receipts = list(executor.map(build, range(shard_count)))
    summary = {
        **bindings,
        "coalition_count": sum(row["coalition_count"] for row in receipts),
        "forward_coalition_count": sum(
            row["forward_coalition_count"] for row in receipts
        ),
        "logical_shard_count": shard_count,
        "state_count": sum(row["state_count"] for row in receipts),
        "status": B4_SCHEDULE_STATUS,
        "trajectory_count": len({row["trajectory_id"] for row in selected}),
    }
    expected_schedule = config["new_truth_schedule"]
    if (
        summary["coalition_count"]
        != expected_schedule["expected_total_schedule_coalition_count"]
        or summary["forward_coalition_count"]
        != expected_schedule["expected_forward_coalition_count"]
    ):
        raise ValueError("B4 schedule coalition count drifted")
    _write_atomic(output_root / "summary.json", canonical_json_bytes(summary, pretty=True))
    return summary


__all__ = [
    "B4_SCHEDULE_STATUS",
    "B4_SHARD_STATUS",
    "materialize_b4_oracle_schedules",
]
