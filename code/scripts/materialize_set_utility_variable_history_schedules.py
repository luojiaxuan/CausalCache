#!/usr/bin/env python3
"""Materialize label-blind variable-history coalition schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from causalcache.set_utility_variable_history import (
    combined_pair_similarity,
    load_variable_history_config,
    sample_broad_subsets,
)
from causalcache.set_utility_variable_history_inputs import (
    build_variable_history_queries_from_source_row,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validated_existing(
    schedule_path: Path,
    receipt_path: Path,
    *,
    identity: str,
) -> dict[str, Any] | None:
    if not schedule_path.exists() and not receipt_path.exists():
        return None
    if not schedule_path.exists() or not receipt_path.exists():
        raise ValueError("incomplete schedule/receipt pair cannot resume")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("identity_sha256") != identity
        or receipt.get("schedule_sha256") != _sha256_file(schedule_path)
        or receipt.get("schedule_byte_count") != schedule_path.stat().st_size
    ):
        raise ValueError("existing schedule receipt drifted")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partition-index", type=int, required=True)
    parser.add_argument("--partition-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()
    if not 0 <= args.partition_index < args.partition_count:
        raise ValueError("partition index is outside partition count")
    if args.workers <= 0:
        raise ValueError("worker count must be positive")
    try:
        from pyarrow import parquet as pq
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise RuntimeError("schedule materialization requires pyarrow and safetensors") from error

    config = load_variable_history_config(args.config)
    source_manifest_path = args.source_root / "manifest.json"
    source_manifest_sha = _sha256_file(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    selected = tuple(
        shard
        for shard in source_manifest["shards"]
        if shard["logical_shard"] % args.partition_count == args.partition_index
    )
    config_sha = _sha256_file(args.config)

    def build(source_shard: dict[str, Any]) -> dict[str, Any]:
        logical_shard = int(source_shard["logical_shard"])
        token_path = (
            args.token_root
            / "token-shards"
            / f"shard-{logical_shard:03d}-of-256.safetensors"
        )
        token_receipt_path = (
            args.token_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-256.json"
        )
        if not token_path.exists() or not token_receipt_path.exists():
            raise FileNotFoundError("schedule source token shard is incomplete")
        token_receipt_sha = _sha256_file(token_receipt_path)
        identity = hashlib.sha256(
            (
                config_sha
                + source_manifest_sha
                + source_shard["sha256"]
                + _sha256_file(token_path)
                + token_receipt_sha
            ).encode("ascii")
        ).hexdigest()
        schedule_path = (
            args.output_root
            / "schedule-shards"
            / f"shard-{logical_shard:03d}-of-256.jsonl"
        )
        receipt_path = (
            args.output_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-256.json"
        )
        existing = _validated_existing(
            schedule_path,
            receipt_path,
            identity=identity,
        )
        if existing is not None:
            return existing

        source_path = (
            args.source_root
            / "trajectory-shards"
            / f"shard-{logical_shard:03d}-of-256.parquet"
        )
        source_rows = pq.read_table(
            source_path,
            columns=[
                "decision_count",
                "history_events_json",
                "ocr_records_json",
                "role",
                "source_id",
                "task_instruction",
            ],
        ).to_pylist()
        tensors = load_file(str(token_path), device="cpu")
        output_rows = []
        source_counts = Counter()
        role_counts = Counter()
        coalition_count = 0
        forward_count = 0
        for source_row in source_rows:
            if source_row["role"] not in config["broad_training_labels"]["roles"]:
                continue
            trajectory_id = source_row["source_id"]
            means = tensors[f"means__{trajectory_id}"].tolist()
            ocr = json.loads(source_row["ocr_records_json"])
            for query in build_variable_history_queries_from_source_row(source_row):
                events = query.candidate_event_step_ids
                visual = {event: means[event] for event in events}
                tokens = {
                    event: {
                        str(token).casefold()
                        for token in ocr[
                            f"images/{trajectory_id}/observation-{event:03d}.png"
                        ]["full_spatial_tokens"]
                    }
                    for event in events
                }
                similarity = combined_pair_similarity(
                    events,
                    visual_embeddings=visual,
                    ocr_token_sets=tokens,
                )
                schedule = sample_broad_subsets(
                    state_id=query.state_id,
                    candidate_event_ids=events,
                    pair_similarity=similarity,
                    target_unique_count=config["broad_training_labels"][
                        "target_unique_subsets_per_state"
                    ],
                    small_history_exact_maximum_n=config[
                        "broad_training_labels"
                    ]["small_history_exact_maximum_n"],
                    seed=config["broad_training_labels"]["seed"],
                )
                output_rows.append(
                    {
                        **schedule.to_payload(),
                        "logical_shard": logical_shard,
                        "role": query.role,
                        "trajectory_id": trajectory_id,
                    }
                )
                role_counts[query.role] += 1
                coalition_count += len(schedule.coalitions)
                forward_count += sum(
                    coalition != schedule.candidate_event_ids
                    for coalition in schedule.coalitions
                )
                source_counts.update(schedule.sources)
        payload = b"".join(
            json.dumps(row, allow_nan=False, sort_keys=True).encode("utf-8") + b"\n"
            for row in sorted(output_rows, key=lambda item: item["state_id"])
        )
        _write_atomic(schedule_path, payload)
        receipt = {
            "coalition_count": coalition_count,
            "forward_coalition_count": forward_count,
            "identity_sha256": identity,
            "logical_shard": logical_shard,
            "role_state_counts": dict(sorted(role_counts.items())),
            "schedule_byte_count": len(payload),
            "schedule_sha256": hashlib.sha256(payload).hexdigest(),
            "source_component_counts": dict(sorted(source_counts.items())),
            "state_count": len(output_rows),
            "status": "COMPLETED_VARIABLE_HISTORY_TRAIN_TUNE_SCHEDULE_SHARD",
            "token_receipt_sha256": token_receipt_sha,
        }
        _write_atomic(receipt_path, _canonical_json(receipt))
        return receipt

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        receipts = list(executor.map(build, selected))
    summary = {
        "coalition_count": sum(row["coalition_count"] for row in receipts),
        "completed_during_or_before_invocation": len(receipts),
        "forward_coalition_count": sum(
            row["forward_coalition_count"] for row in receipts
        ),
        "partition_count": args.partition_count,
        "partition_index": args.partition_index,
        "state_count": sum(row["state_count"] for row in receipts),
        "status": "COMPLETED_VARIABLE_HISTORY_TRAIN_TUNE_SCHEDULE_PARTITION",
    }
    _write_atomic(
        args.output_root
        / "workers"
        / f"worker-{args.partition_index:03d}-of-{args.partition_count:03d}.json",
        _canonical_json(summary),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
