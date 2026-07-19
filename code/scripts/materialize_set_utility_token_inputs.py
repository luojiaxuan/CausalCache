#!/usr/bin/env python3
"""Join a train/tune label snapshot with exact dense images and event text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from causalcache import set_utility_processor_artifacts as processor_artifacts
from causalcache.gate_v1_data import HASH_DIMENSION
from causalcache.low_fidelity_v2 import serialize_low_fidelity_v2
from causalcache.set_utility_dense import derive_dense_states_from_query_pair
from causalcache.set_utility_mvp import canonical_json_bytes


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _backfill_by_trajectory(root: Path) -> dict[str, dict[str, bytes]]:
    manifest = _read_json(root / "manifest.json")
    if manifest.get("status") != "COMPLETED_DENSE_IMAGE_BACKFILL":
        raise ValueError("dense image backfill is incomplete")
    result: dict[str, dict[str, bytes]] = defaultdict(dict)
    for record in manifest["files"]:
        reference = str(record["reference"])
        payload = (root / reference).read_bytes()
        if (
            len(payload) != record["byte_count"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise ValueError("dense image backfill payload drifted")
        result[str(record["source_id"])][reference] = payload
    return result


def _snapshot_records(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = _read_json(root / "manifest.json")
    if (
        manifest.get("status") != "FROZEN_SET_UTILITY_TOKEN_PILOT_SNAPSHOT"
        or manifest.get("evaluation_labels_included") is not False
    ):
        raise ValueError("token input materialization requires a train/tune snapshot")
    records: dict[str, dict[str, Any]] = {}
    for entry in manifest["records"]:
        path = root / entry["file"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("snapshot label bytes drifted")
        record = json.loads(payload)
        if record["state_id"] != entry["state_id"] or record["role"] not in {
            "train",
            "tune",
        }:
            raise ValueError("snapshot record identity drifted")
        records[record["state_id"]] = record
    if len(records) != manifest["state_count"]:
        raise ValueError("snapshot state count drifted")
    return records, manifest


def _worker_manifest(path: Path) -> Any:
    with tarfile.open(path, mode="r:") as archive:
        return processor_artifacts._read_manifest(archive, expected_worker=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--processor-root", type=Path, required=True)
    parser.add_argument("--backfill-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    snapshot_root = args.snapshot_root.resolve()
    processor_root = args.processor_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"token input output already exists: {output_root}")
    image_root = output_root / "images"
    image_root.mkdir(parents=True)

    labels, snapshot_manifest = _snapshot_records(snapshot_root)
    target_by_trajectory: dict[str, set[str]] = defaultdict(set)
    for record in labels.values():
        target_by_trajectory[str(record["trajectory_id"])].add(record["state_id"])
    supplemental = _backfill_by_trajectory(args.backfill_root.resolve())
    found: dict[str, dict[str, Any]] = {}
    image_bytes_by_key: dict[str, bytes] = {}
    text_keys: set[str] = set()

    for worker_index in range(4):
        artifact = (
            processor_root
            / "substrate"
            / f"processor-substrate-worker-{worker_index:02d}.tar"
        )
        iterator = processor_artifacts.iter_processor_query_artifact_records(
            artifact,
            expected_worker=_worker_manifest(artifact),
        )
        while True:
            try:
                pair = (next(iterator), next(iterator))
            except StopIteration:
                break
            trajectory_id = pair[0].trajectory_id
            wanted = target_by_trajectory.get(trajectory_id)
            if not wanted:
                continue
            for dense in derive_dense_states_from_query_pair(
                pair,
                supplemental_image_payloads=supplemental.get(trajectory_id),
            ):
                state_id = dense.query.state_id
                if state_id not in wanted:
                    continue
                label = labels[state_id]
                candidates = dense.query.candidate_event_step_ids
                if tuple(label["candidate_event_step_ids"]) != candidates:
                    raise ValueError("dense candidate ids differ from the label")
                history = {event.event_step_id: event for event in dense.query.history_events}

                def image_key(reference: str) -> str:
                    payload = dense.image_payloads[reference]
                    key = hashlib.sha256(payload).hexdigest()
                    prior = image_bytes_by_key.setdefault(key, payload)
                    if prior != payload:
                        raise ValueError("image SHA256 collision")
                    destination = image_root / f"{key}.png"
                    if not destination.exists():
                        with destination.open("xb") as handle:
                            handle.write(payload)
                            handle.flush()
                            os.fsync(handle.fileno())
                    return key

                instruction = dense.query.task_instruction
                instruction_key = hashlib.sha256(
                    instruction.encode("utf-8")
                ).hexdigest()
                text_keys.add(instruction_key)
                event_texts = []
                event_text_keys = []
                for step in candidates:
                    text = serialize_low_fidelity_v2(
                        history[step].low_fidelity_summary
                    ).decode("utf-8").rstrip("\n")
                    event_texts.append(text)
                    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    event_text_keys.append(key)
                    text_keys.add(key)
                numeric = [
                    row[HASH_DIMENSION:]
                    for row in label["feature"]["event_features"]
                ]
                if len(numeric) != 4 or any(len(row) != 11 for row in numeric):
                    raise ValueError("event numeric feature geometry drifted")
                found[state_id] = {
                    "candidate_event_step_ids": list(candidates),
                    "distance_rows": label["distance_rows"],
                    "event_image_keys": [
                        image_key(history[step].high_fidelity_observation_ref)
                        for step in candidates
                    ],
                    "event_numeric_features": numeric,
                    "event_text_keys": event_text_keys,
                    "event_texts": event_texts,
                    "instruction": instruction,
                    "instruction_text_key": instruction_key,
                    "maximum_labeled_cardinality": 2,
                    "role": label["role"],
                    "state_id": state_id,
                    "trajectory_id": trajectory_id,
                    "current_image_key": image_key(
                        dense.query.current_observation_ref
                    ),
                }

    missing = sorted(set(labels) - set(found))
    if missing:
        raise RuntimeError(f"failed to materialize {len(missing)} snapshot states")
    state_path = output_root / "states.jsonl"
    with state_path.open("xb") as handle:
        for state_id in sorted(found):
            handle.write(canonical_json_bytes(found[state_id]) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    state_bytes = state_path.read_bytes()
    manifest = {
        "evaluation_labels_included": False,
        "image_count": len(image_bytes_by_key),
        "image_directory": "images",
        "role_counts": dict(
            sorted(Counter(row["role"] for row in found.values()).items())
        ),
        "schema_version": "1.0.0",
        "snapshot_content_sha256": snapshot_manifest["content_sha256"],
        "state_count": len(found),
        "states_jsonl": "states.jsonl",
        "states_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "status": "MATERIALIZED_SET_UTILITY_TOKEN_INPUTS",
        "text_count": len(text_keys),
        "trajectory_count": len(target_by_trajectory),
    }
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    _write_json(output_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
