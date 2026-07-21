"""Merge several compatible train-only restoration sources into one snapshot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.set_utility_contextual_inputs import (
    CONTEXTUAL_INPUT_STATUS,
    CONTEXTUAL_REQUIREMENT_STATUS,
    contextual_input_lineage_sha256s,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file


LABEL_STATUS = "COMPLETED_VARIABLE_HISTORY_LABEL_STATE"
SCHEDULE_STATUSES = frozenset(
    (
        "COMPLETED_SET_UTILITY_DIRECT_ON_POLICY_SCHEDULES",
        "COMPLETED_SET_UTILITY_TRAIN_ON_POLICY_SCHEDULES",
        "COMPLETED_SET_UTILITY_LONG_ORACLE_SCHEDULES",
    )
)


@dataclass(frozen=True)
class ContextualEnrichmentSource:
    name: str
    schedule_root: Path
    label_roots: tuple[Path, ...]
    expected_schedule_content_sha256: str
    expected_schedule_config_sha256: str
    expected_label_scientific_config_sha256: str
    expected_label_execution_config_sha256: str


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


def _content_is_valid(payload: dict[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    if not isinstance(claimed, str):
        return False
    unsigned = dict(payload)
    del unsigned["content_sha256"]
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() == claimed


def _coalition(row: dict[str, Any]) -> tuple[int, ...]:
    values = row.get("coalition_event_step_ids")
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise ValueError("distance-row coalition must be an integer list")
    result = tuple(values)
    if result != tuple(sorted(set(result))):
        raise ValueError("distance-row coalition must be sorted and unique")
    return result


def _distance_table(
    rows: list[dict[str, Any]], *, candidates: tuple[int, ...]
) -> dict[tuple[int, ...], float]:
    candidate_set = set(candidates)
    result: dict[tuple[int, ...], float] = {}
    for row in rows:
        coalition = _coalition(row)
        distance = row.get("distance")
        if (
            not set(coalition).issubset(candidate_set)
            or type(distance) not in {int, float}
            or not math.isfinite(float(distance))
            or coalition in result
        ):
            raise ValueError("distance table is invalid")
        result[coalition] = float(distance)
    return result


def _load_schedules(
    source: ContextualEnrichmentSource, *, base_content_sha256: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
    manifest_path = source.schedule_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("status") not in SCHEDULE_STATUSES
        or not _content_is_valid(manifest)
        or manifest.get("content_sha256")
        != source.expected_schedule_content_sha256
        or manifest.get("config_sha256")
        != source.expected_schedule_config_sha256
    ):
        raise ValueError(f"schedule binding drifted: {source.name}")
    bound_input = manifest.get("input_content_sha256")
    if bound_input is not None and bound_input != base_content_sha256:
        raise ValueError(f"schedule input binding drifted: {source.name}")
    schedules = {}
    for path in sorted(
        source.schedule_root.glob("schedule-shards/shard-*-of-256.jsonl")
    ):
        for row in _read_jsonl(path):
            state_id = row.get("state_id")
            if not isinstance(state_id, str) or state_id in schedules:
                raise ValueError(f"schedule inventory is invalid: {source.name}")
            schedules[state_id] = row
    expected_count = manifest.get(
        "scheduled_state_count", manifest.get("state_count")
    )
    if len(schedules) != expected_count:
        raise ValueError(f"schedule state count drifted: {source.name}")
    return manifest, schedules, manifest_path


def _load_labels(
    source: ContextualEnrichmentSource,
) -> tuple[dict[str, dict[str, Any]], str]:
    labels = {}
    digest = hashlib.sha256()
    for root in source.label_roots:
        for path in sorted((root / "states").glob("*.json")):
            if path.name.startswith("._"):
                continue
            payload = path.read_bytes()
            row = json.loads(payload)
            state_id = row.get("state_id")
            if not isinstance(state_id, str) or state_id in labels:
                raise ValueError(f"label inventory is invalid: {source.name}")
            if (
                row.get("scientific_config_sha256")
                != source.expected_label_scientific_config_sha256
                or row.get("execution_config_sha256")
                != source.expected_label_execution_config_sha256
            ):
                raise ValueError(f"label runtime binding drifted: {source.name}")
            labels[state_id] = row
            digest.update(state_id.encode("utf-8"))
            digest.update(hashlib.sha256(payload).digest())
    return labels, digest.hexdigest()


def materialize_contextual_multisource_enriched_inputs(
    *,
    base_input_root: Path,
    sources: tuple[ContextualEnrichmentSource, ...],
    output_root: Path,
    config_sha256: str,
    duplicate_tolerance: float = 1e-6,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("multisource contextual output already exists")
    if not sources or len({source.name for source in sources}) != len(sources):
        raise ValueError("multisource enrichment names must be non-empty and unique")
    if duplicate_tolerance < 0.0:
        raise ValueError("duplicate tolerance cannot be negative")
    ordered_sources = tuple(sorted(sources, key=lambda source: source.name))
    base_manifest_path = base_input_root / "manifest.json"
    base_manifest = _read_json(base_manifest_path)
    if (
        base_manifest.get("status") != CONTEXTUAL_INPUT_STATUS
        or base_manifest.get("evaluation_labels_included") is not False
    ):
        raise ValueError("multisource enrichment requires train/tune-only inputs")
    states_path = base_input_root / base_manifest["states_jsonl"]
    if sha256_file(states_path) != base_manifest.get("states_sha256"):
        raise ValueError("base contextual states drifted")
    states = _read_jsonl(states_path)
    states_by_id = {row["state_id"]: row for row in states}
    if len(states_by_id) != len(states) or len(states) != base_manifest["state_count"]:
        raise ValueError("base contextual state inventory drifted")
    tables = {
        state_id: _distance_table(
            state["distance_rows"],
            candidates=tuple(state["candidate_event_step_ids"]),
        )
        for state_id, state in states_by_id.items()
    }
    source_bindings = {}
    source_revisions = set()
    touched_states = set()
    total_added = 0
    total_duplicates = 0
    duplicate_max_abs_delta = 0.0
    for source in ordered_sources:
        schedule_manifest, schedules, schedule_manifest_path = _load_schedules(
            source, base_content_sha256=base_manifest["content_sha256"]
        )
        labels, labels_content_sha256 = _load_labels(source)
        if set(labels) != set(schedules):
            raise ValueError(f"label inventory does not cover schedule: {source.name}")
        source_added = 0
        source_duplicates = 0
        source_max_delta = 0.0
        for state_id, schedule in schedules.items():
            state = states_by_id.get(state_id)
            if state is None:
                raise ValueError(f"schedule state is absent from inputs: {source.name}")
            candidates = tuple(state["candidate_event_step_ids"])
            label = labels[state_id]
            if (
                state.get("role") != "train"
                or schedule.get("role") != "train"
                or label.get("role") != "train"
                or schedule.get("trajectory_id") != state.get("trajectory_id")
                or label.get("trajectory_id") != state.get("trajectory_id")
                or tuple(schedule.get("candidate_event_ids", ())) != candidates
                or tuple(label.get("candidate_event_step_ids", ())) != candidates
                or label.get("status") != LABEL_STATUS
            ):
                raise ValueError(f"train label binding drifted: {source.name}")
            scheduled = {
                tuple(row["event_ids"])
                for row in schedule.get("coalitions", ())
                if isinstance(row, dict)
            }
            label_table = _distance_table(label["distance_rows"], candidates=candidates)
            if set(label_table) != scheduled | {candidates}:
                raise ValueError(f"label table does not cover schedule: {source.name}")
            revision = label.get("source_revision")
            if not isinstance(revision, str) or len(revision) != 40:
                raise ValueError(f"label source revision is missing: {source.name}")
            source_revisions.add(revision)
            touched_states.add(state_id)
            merged = tables[state_id]
            for coalition, distance in label_table.items():
                if coalition not in merged:
                    merged[coalition] = distance
                    source_added += 1
                    total_added += 1
                    continue
                delta = abs(merged[coalition] - distance)
                source_duplicates += 1
                total_duplicates += 1
                source_max_delta = max(source_max_delta, delta)
                duplicate_max_abs_delta = max(duplicate_max_abs_delta, delta)
                if delta > duplicate_tolerance:
                    raise ValueError("duplicate restoration distance exceeded tolerance")
        source_bindings[source.name] = {
            "added_distance_row_count": source_added,
            "duplicate_distance_row_count": source_duplicates,
            "duplicate_max_abs_delta": source_max_delta,
            "label_terminal_content_sha256": labels_content_sha256,
            "schedule_config_sha256": schedule_manifest["config_sha256"],
            "schedule_content_sha256": schedule_manifest["content_sha256"],
            "schedule_manifest_sha256": sha256_file(schedule_manifest_path),
            "schedule_status": schedule_manifest["status"],
            "state_count": len(schedules),
        }
    enriched = []
    for state in states:
        if state["state_id"] not in touched_states:
            enriched.append(state)
            continue
        candidates = tuple(state["candidate_event_step_ids"])
        table = tables[state["state_id"]]
        rows = [
            {"coalition_event_step_ids": list(coalition), "distance": distance}
            for coalition, distance in sorted(
                table.items(), key=lambda item: (len(item[0]), item[0])
            )
        ]
        cardinalities = [len(coalition) for coalition in table if coalition != candidates]
        enriched.append(
            {
                **state,
                "distance_rows": rows,
                "maximum_labeled_cardinality": max(cardinalities, default=0),
            }
        )
    tune_before = b"".join(
        canonical_json_bytes(row) + b"\n"
        for row in sorted(
            (row for row in states if row["role"] == "tune"),
            key=lambda row: row["state_id"],
        )
    )
    tune_after = b"".join(
        canonical_json_bytes(row) + b"\n"
        for row in sorted(
            (row for row in enriched if row["role"] == "tune"),
            key=lambda row: row["state_id"],
        )
    )
    if tune_before != tune_after:
        raise RuntimeError("tune states changed during multisource enrichment")
    state_payload = b"".join(
        canonical_json_bytes(row) + b"\n"
        for row in sorted(enriched, key=lambda row: row["state_id"])
    )
    enrichment = {
        "added_distance_row_count": total_added,
        "config_sha256": config_sha256,
        "duplicate_distance_row_count": total_duplicates,
        "duplicate_max_abs_delta": duplicate_max_abs_delta,
        "duplicate_tolerance": duplicate_tolerance,
        "label_source_revisions": sorted(source_revisions),
        "schema_version": "causalcache.contextual_multisource_enrichment.v1",
        "source_bindings": source_bindings,
        "state_count": len(touched_states),
        "tune_state_content_sha256": hashlib.sha256(tune_after).hexdigest(),
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
        (temporary_root / "requirement-shards").mkdir()
        for receipt in base_manifest["requirement_shards"]:
            if receipt.get("status") != CONTEXTUAL_REQUIREMENT_STATUS:
                raise ValueError("base contextual requirement receipt drifted")
            source_path = base_input_root / receipt["path"]
            if sha256_file(source_path) != receipt["sha256"]:
                raise ValueError("base contextual requirement shard drifted")
            shutil.copy2(source_path, temporary_root / receipt["path"])
        (temporary_root / "manifest.json").write_bytes(
            canonical_json_bytes(manifest, pretty=True) + b"\n"
        )
        os.replace(temporary_root, output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return manifest


__all__ = [
    "ContextualEnrichmentSource",
    "materialize_contextual_multisource_enriched_inputs",
]
