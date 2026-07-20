#!/usr/bin/env python3
"""Freeze completed distributed labels into token-predictor input rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from causalcache.low_fidelity_v2 import LowFidelityEventV2, serialize_low_fidelity_v2
from causalcache.set_utility_variable_history_training import (
    VARIABLE_HISTORY_NUMERIC_FEATURE_NAMES,
    variable_history_event_numeric_features,
)


COMPLETED = "COMPLETED_VARIABLE_HISTORY_LABEL_STATE"


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _text_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _visual_key(trajectory_id: str, observation_step: int) -> str:
    return f"{trajectory_id}:observation:{observation_step:03d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("training snapshot output already exists")
    try:
        from pyarrow import parquet as pq
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise RuntimeError("training snapshot requires pyarrow and safetensors") from error

    terminal_paths = tuple(sorted((args.label_root / "states").glob("*.json")))
    if not terminal_paths:
        raise ValueError("training snapshot found no terminal label states")
    terminals = []
    terminal_hashes = []
    status_counts = Counter()
    for path in terminal_paths:
        terminal_hashes.append(_sha256_file(path))
        row = json.loads(path.read_text(encoding="utf-8"))
        status_counts[row["status"]] += 1
        if row["status"] == COMPLETED:
            terminals.append(row)
    if not terminals:
        raise ValueError("training snapshot found no completed label states")
    by_trajectory: dict[str, list[dict[str, Any]]] = {}
    for terminal in terminals:
        by_trajectory.setdefault(terminal["trajectory_id"], []).append(terminal)

    source_manifest_path = args.source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_shards = {
        trajectory_id: shard
        for shard in source_manifest["shards"]
        for trajectory_id in shard["trajectory_ids"]
        if trajectory_id in by_trajectory
    }
    if set(source_shards) != set(by_trajectory):
        raise ValueError("label trajectories are not covered by the source manifest")

    output_rows = []
    texts: dict[str, str] = {}
    observed_states = set()
    for logical_shard in sorted({row["logical_shard"] for row in source_shards.values()}):
        source_path = (
            args.source_root
            / "trajectory-shards"
            / f"shard-{logical_shard:03d}-of-256.parquet"
        )
        token_path = (
            args.token_root
            / "token-shards"
            / f"shard-{logical_shard:03d}-of-256.safetensors"
        )
        if not token_path.exists():
            raise FileNotFoundError("visual token shard required by snapshot is missing")
        means_tensors = load_file(str(token_path), device="cpu")
        source_rows = pq.read_table(
            source_path,
            columns=[
                "history_events_json",
                "ocr_records_json",
                "role",
                "source_id",
                "task_instruction",
            ],
        ).to_pylist()
        for source in source_rows:
            trajectory_id = source["source_id"]
            if trajectory_id not in by_trajectory:
                continue
            history = {
                int(event["event_step_id"]): event
                for event in json.loads(source["history_events_json"])
            }
            ocr = json.loads(source["ocr_records_json"])
            means = means_tensors[f"means__{trajectory_id}"]
            instruction = source["task_instruction"]
            instruction_key = _text_key(instruction)
            texts.setdefault(instruction_key, instruction)
            for terminal in sorted(
                by_trajectory[trajectory_id], key=lambda item: item["state_id"]
            ):
                candidates = tuple(terminal["candidate_event_step_ids"])
                if candidates != tuple(range(1, len(candidates) + 1)):
                    raise ValueError("training snapshot candidate universe is not full history")
                if terminal["role"] != source["role"]:
                    raise ValueError("training snapshot source/label role drifted")
                current_step = len(candidates)
                current_reference = (
                    f"images/{trajectory_id}/observation-{current_step:03d}.png"
                )
                current_ocr = ocr[current_reference]["full_spatial_tokens"]
                event_texts = []
                event_text_keys = []
                numeric = []
                for step in candidates:
                    event = history[step]
                    low = LowFidelityEventV2.from_mapping(event["low_fidelity_summary"])
                    text = serialize_low_fidelity_v2(low).decode("utf-8").rstrip("\n")
                    key = _text_key(text)
                    prior = texts.setdefault(key, text)
                    if prior != text:
                        raise ValueError("training snapshot text SHA256 collision")
                    event_texts.append(text)
                    event_text_keys.append(key)
                    event_reference = event["high_fidelity_observation_ref"]
                    cosine = float((means[step] * means[current_step]).sum().item())
                    numeric.append(
                        variable_history_event_numeric_features(
                            low,
                            decision_step_id=current_step + 1,
                            event_ocr_tokens=ocr[event_reference]["full_spatial_tokens"],
                            current_ocr_tokens=current_ocr,
                            event_current_vlm_cosine=cosine,
                        )
                    )
                row = {
                    "candidate_event_step_ids": list(candidates),
                    "current_image_key": _visual_key(trajectory_id, current_step),
                    "distance_rows": terminal["distance_rows"],
                    "event_image_keys": [
                        _visual_key(trajectory_id, step) for step in candidates
                    ],
                    "event_numeric_features": numeric,
                    "event_text_keys": event_text_keys,
                    "event_texts": event_texts,
                    "instruction": instruction,
                    "instruction_text_key": instruction_key,
                    "logical_shard": logical_shard,
                    "maximum_labeled_cardinality": max(
                        len(row["coalition_event_step_ids"])
                        for row in terminal["distance_rows"]
                        if tuple(row["coalition_event_step_ids"]) != candidates
                    ),
                    "role": terminal["role"],
                    "state_id": terminal["state_id"],
                    "trajectory_id": trajectory_id,
                }
                output_rows.append(row)
                observed_states.add(terminal["state_id"])
        del means_tensors
    expected_states = {terminal["state_id"] for terminal in terminals}
    if observed_states != expected_states:
        raise RuntimeError("training snapshot did not materialize every completed state")

    state_payload = b"".join(
        _canonical_json(row) for row in sorted(output_rows, key=lambda item: item["state_id"])
    )
    text_payload = b"".join(
        _canonical_json({"key": key, "text": value})
        for key, value in sorted(texts.items())
    )
    _write_atomic(args.output_root / "states.jsonl", state_payload)
    _write_atomic(args.output_root / "texts.jsonl", text_payload)
    manifest = {
        "content_sha256": hashlib.sha256(state_payload + text_payload).hexdigest(),
        "evaluation_labels_included": False,
        "label_source_revisions": sorted({row["source_revision"] for row in terminals}),
        "numeric_feature_names": list(VARIABLE_HISTORY_NUMERIC_FEATURE_NAMES),
        "role_counts": dict(sorted(Counter(row["role"] for row in output_rows).items())),
        "schema_version": "1.0.0",
        "snapshot_id": args.snapshot_id,
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "state_count": len(output_rows),
        "states_jsonl": "states.jsonl",
        "states_sha256": hashlib.sha256(state_payload).hexdigest(),
        "status": "COMPLETED_VARIABLE_HISTORY_TRAINING_INPUT_SNAPSHOT",
        "terminal_file_count": len(terminal_paths),
        "terminal_set_sha256": hashlib.sha256(
            "".join(sorted(terminal_hashes)).encode("ascii")
        ).hexdigest(),
        "terminal_status_counts": dict(sorted(status_counts.items())),
        "text_count": len(texts),
        "texts_jsonl": "texts.jsonl",
        "texts_sha256": hashlib.sha256(text_payload).hexdigest(),
        "trajectory_count": len(by_trajectory),
        "visual_token_profile": "full_480_target_actual_grid_bf16_4096",
    }
    _write_atomic(args.output_root / "manifest.json", _canonical_json(manifest))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
