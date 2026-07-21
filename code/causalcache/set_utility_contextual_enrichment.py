"""Merge train-only on-policy restoration labels into contextual inputs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from causalcache.set_utility_contextual_inputs import (
    CONTEXTUAL_INPUT_STATUS,
    CONTEXTUAL_REQUIREMENT_STATUS,
    contextual_input_lineage_sha256s,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file


SCHEDULE_STATUS = "COMPLETED_SET_UTILITY_TRAIN_ON_POLICY_SCHEDULES"
LABEL_STATUS = "COMPLETED_VARIABLE_HISTORY_LABEL_STATE"


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


def _coalition(row: dict[str, Any]) -> tuple[int, ...]:
    values = row.get("coalition_event_step_ids")
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise ValueError("distance-row coalition must be an integer list")
    coalition = tuple(values)
    if coalition != tuple(sorted(set(coalition))):
        raise ValueError("distance-row coalition must be sorted and unique")
    return coalition


def _distance_table(
    rows: list[dict[str, Any]], *, candidates: tuple[int, ...]
) -> dict[tuple[int, ...], float]:
    candidate_set = set(candidates)
    table: dict[tuple[int, ...], float] = {}
    for row in rows:
        coalition = _coalition(row)
        if not set(coalition).issubset(candidate_set):
            raise ValueError("distance-row coalition left the candidate inventory")
        distance = row.get("distance")
        if type(distance) not in {int, float} or not math.isfinite(float(distance)):
            raise ValueError("distance-row value must be finite")
        if coalition in table:
            raise ValueError("duplicate distance-row coalition")
        table[coalition] = float(distance)
    return table


def _load_schedules(
    schedule_root: Path, *, manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    schedules: dict[str, dict[str, Any]] = {}
    for path in sorted((schedule_root / "schedule-shards").glob("*.jsonl")):
        for row in _read_jsonl(path):
            state_id = row.get("state_id")
            if not isinstance(state_id, str) or state_id in schedules:
                raise ValueError("schedule state inventory is invalid or duplicated")
            schedules[state_id] = row
    if len(schedules) != manifest.get("state_count"):
        raise ValueError("schedule state count drifted")
    return schedules


def _load_labels(label_roots: tuple[Path, ...]) -> tuple[dict[str, dict[str, Any]], str]:
    labels: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    for root in label_roots:
        for path in sorted((root / "states").glob("*.json")):
            payload = path.read_bytes()
            row = json.loads(payload)
            state_id = row.get("state_id")
            if not isinstance(state_id, str) or state_id in labels:
                raise ValueError("label state inventory is invalid or duplicated")
            labels[state_id] = row
            digest.update(state_id.encode("utf-8"))
            digest.update(hashlib.sha256(payload).digest())
    return labels, digest.hexdigest()


def materialize_contextual_enriched_inputs(
    *,
    base_input_root: Path,
    schedule_root: Path,
    label_roots: tuple[Path, ...],
    output_root: Path,
    duplicate_tolerance: float = 1e-6,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("enriched contextual output already exists")
    if not label_roots:
        raise ValueError("at least one label root is required")
    if duplicate_tolerance < 0.0:
        raise ValueError("duplicate tolerance cannot be negative")

    base_manifest_path = base_input_root / "manifest.json"
    base_manifest = _read_json(base_manifest_path)
    if (
        base_manifest.get("status") != CONTEXTUAL_INPUT_STATUS
        or base_manifest.get("evaluation_labels_included") is not False
    ):
        raise ValueError("enrichment requires a train/tune-only contextual snapshot")
    states_path = base_input_root / base_manifest["states_jsonl"]
    if sha256_file(states_path) != base_manifest.get("states_sha256"):
        raise ValueError("base contextual states drifted")
    states = _read_jsonl(states_path)
    if len(states) != base_manifest.get("state_count"):
        raise ValueError("base contextual state count drifted")
    states_by_id = {row["state_id"]: row for row in states}
    if len(states_by_id) != len(states):
        raise ValueError("base contextual state ids are duplicated")

    schedule_manifest_path = schedule_root / "manifest.json"
    schedule_manifest = _read_json(schedule_manifest_path)
    if (
        schedule_manifest.get("status") != SCHEDULE_STATUS
        or schedule_manifest.get("input_content_sha256")
        != base_manifest.get("content_sha256")
    ):
        raise ValueError("train on-policy schedule binding drifted")
    schedules = _load_schedules(schedule_root, manifest=schedule_manifest)
    labels, labels_content_sha256 = _load_labels(label_roots)
    if set(labels) != set(schedules):
        raise ValueError("completed label inventory does not exactly cover the schedule")

    duplicate_count = 0
    duplicate_max_abs_delta = 0.0
    added_row_count = 0
    enriched = []
    source_revisions: set[str] = set()
    for state in states:
        state_id = state["state_id"]
        schedule = schedules.get(state_id)
        if schedule is None:
            enriched.append(state)
            continue
        if state.get("role") != "train" or schedule.get("role") != "train":
            raise ValueError("on-policy enrichment crossed the evaluation firewall")
        candidates = tuple(state["candidate_event_step_ids"])
        if (
            schedule.get("trajectory_id") != state.get("trajectory_id")
            or tuple(schedule.get("candidate_event_ids", ())) != candidates
        ):
            raise ValueError("schedule candidate or trajectory binding drifted")
        label = labels[state_id]
        if (
            label.get("status") != LABEL_STATUS
            or label.get("role") != "train"
            or label.get("trajectory_id") != state.get("trajectory_id")
            or tuple(label.get("candidate_event_step_ids", ())) != candidates
        ):
            raise ValueError("label terminal binding drifted")
        source_revision = label.get("source_revision")
        if not isinstance(source_revision, str) or len(source_revision) != 40:
            raise ValueError("label source revision is missing")
        source_revisions.add(source_revision)

        scheduled = {
            tuple(row["event_ids"])
            for row in schedule.get("coalitions", ())
            if isinstance(row, dict)
        }
        label_table = _distance_table(label["distance_rows"], candidates=candidates)
        expected_label_coalitions = scheduled | {candidates}
        if set(label_table) != expected_label_coalitions:
            raise ValueError("label terminal does not exactly cover scheduled coalitions")
        merged_table = _distance_table(state["distance_rows"], candidates=candidates)
        for coalition, distance in label_table.items():
            if coalition in merged_table:
                delta = abs(merged_table[coalition] - distance)
                duplicate_count += 1
                duplicate_max_abs_delta = max(duplicate_max_abs_delta, delta)
                if delta > duplicate_tolerance:
                    raise ValueError("duplicate restoration distance exceeded tolerance")
                continue
            merged_table[coalition] = distance
            added_row_count += 1
        merged_rows = [
            {"coalition_event_step_ids": list(coalition), "distance": distance}
            for coalition, distance in sorted(
                merged_table.items(), key=lambda item: (len(item[0]), item[0])
            )
        ]
        labeled_cardinalities = [
            len(coalition) for coalition in merged_table if coalition != candidates
        ]
        enriched.append(
            {
                **state,
                "distance_rows": merged_rows,
                "maximum_labeled_cardinality": max(labeled_cardinalities, default=0),
            }
        )

    if not set(schedules).issubset(states_by_id):
        raise ValueError("schedule references a state absent from the base snapshot")
    if duplicate_max_abs_delta > duplicate_tolerance:
        raise RuntimeError("duplicate restoration distance postcondition failed")

    tune_payload_before = b"".join(
        canonical_json_bytes(row) + b"\n"
        for row in sorted(
            (row for row in states if row["role"] == "tune"),
            key=lambda row: row["state_id"],
        )
    )
    tune_payload_after = b"".join(
        canonical_json_bytes(row) + b"\n"
        for row in sorted(
            (row for row in enriched if row["role"] == "tune"),
            key=lambda row: row["state_id"],
        )
    )
    if tune_payload_before != tune_payload_after:
        raise RuntimeError("tune states changed during train-only enrichment")

    state_payload = b"".join(
        canonical_json_bytes(row) + b"\n"
        for row in sorted(enriched, key=lambda row: row["state_id"])
    )
    enrichment = {
        "added_distance_row_count": added_row_count,
        "duplicate_distance_row_count": duplicate_count,
        "duplicate_max_abs_delta": duplicate_max_abs_delta,
        "duplicate_tolerance": duplicate_tolerance,
        "label_source_revisions": sorted(source_revisions),
        "label_terminal_content_sha256": labels_content_sha256,
        "schedule_content_sha256": schedule_manifest["content_sha256"],
        "schedule_manifest_sha256": sha256_file(schedule_manifest_path),
        "state_count": len(schedules),
        "tune_state_content_sha256": hashlib.sha256(tune_payload_after).hexdigest(),
    }
    manifest = {
        **base_manifest,
        "ancestor_content_sha256s": sorted(
            contextual_input_lineage_sha256s(base_manifest)
        ),
        "content_sha256": hashlib.sha256(
            state_payload
            + "".join(
                row["sha256"] for row in base_manifest["requirement_shards"]
            ).encode("ascii")
        ).hexdigest(),
        "enrichment": enrichment,
        "parent_content_sha256": base_manifest["content_sha256"],
        "states_sha256": hashlib.sha256(state_payload).hexdigest(),
    }

    temporary_root = output_root.with_name(f"{output_root.name}.{os.getpid()}.tmp")
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    temporary_root.mkdir(parents=True)
    try:
        (temporary_root / "states.jsonl").write_bytes(state_payload)
        requirement_root = temporary_root / "requirement-shards"
        requirement_root.mkdir()
        for receipt in base_manifest["requirement_shards"]:
            if receipt.get("status") != CONTEXTUAL_REQUIREMENT_STATUS:
                raise ValueError("base contextual requirement receipt drifted")
            source = base_input_root / receipt["path"]
            if sha256_file(source) != receipt["sha256"]:
                raise ValueError("base contextual requirement shard drifted")
            shutil.copy2(source, temporary_root / receipt["path"])
        (temporary_root / "manifest.json").write_bytes(
            canonical_json_bytes(manifest, pretty=True) + b"\n"
        )
        os.replace(temporary_root, output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return manifest


__all__ = ["materialize_contextual_enriched_inputs"]
