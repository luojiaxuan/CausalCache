#!/usr/bin/env python3
"""Run resumable post-GO GUI-Owl native-action replay on one GPU partition."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2_1 import parse_gui_owl_v2_1_output
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21GenerationParseError
from causalcache.set_utility_native_replay import (
    NATIVE_REPLAY_METHODS,
    NATIVE_REPLAY_SCHEMA_VERSION,
    NATIVE_REPLAY_SHARD_STATUS,
    NATIVE_REPLAY_STATUS,
    canonical_json_bytes,
    compare_action_components,
    sha256_file,
    signed_content_hash_is_valid,
)
from causalcache.set_utility_variable_history_inputs import (
    build_variable_history_messages,
    build_variable_history_queries,
    trajectory_from_source_row,
)


COMPLETED_REPLAY_STATE = "COMPLETED_SET_UTILITY_NATIVE_REPLAY_STATE"
COMPLETED_REPLAY_COALITION = "COMPLETED_SET_UTILITY_NATIVE_REPLAY_COALITION"
FAILED_REPLAY_PARSE = "FAILED_SET_UTILITY_NATIVE_REPLAY_PARSE"


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


def _write_atomic(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value, pretty=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _state_lane(state_id: str, lane_count: int) -> int:
    if lane_count <= 0:
        raise ValueError("state lane count must be positive")
    digest = hashlib.sha256(state_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % lane_count


def _runtime(repository_root: Path, model_dir: Path) -> tuple[Any, Any]:
    from PIL import Image

    from causalcache.policy.gui_owl_variable_history_runtime import (
        GUIOwlVariableHistoryRuntime,
        VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )

    runtime = GUIOwlVariableHistoryRuntime(
        model_dir=model_dir,
        expected_snapshot_manifest=(
            repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device="cuda:0",
        target_effective_visual_tokens_per_image=(
            VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        ),
    )

    def decode(payload: bytes) -> Any:
        source = Image.open(io.BytesIO(payload))
        try:
            return source.convert("RGB")
        finally:
            source.close()

    return runtime, decode


def _state_identity(
    *,
    source_revision: str,
    source_shard_sha256: str,
    schedule_manifest_sha256: str,
    schedule_receipt_sha256: str,
    schedule: dict[str, Any],
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schedule": schedule,
                "schedule_manifest_sha256": schedule_manifest_sha256,
                "schedule_receipt_sha256": schedule_receipt_sha256,
                "source_revision": source_revision,
                "source_shard_sha256": source_shard_sha256,
            }
        )
    ).hexdigest()


def _run_state(
    *,
    query: Any,
    trajectory: Any,
    schedule: dict[str, Any],
    output_root: Path,
    runtime: Any,
    base_decoder: Any,
    state_identity: str,
    source_revision: str,
    worker_index: int,
) -> dict[str, Any]:
    state_key = query.state_id.replace(":", "_")
    terminal_path = output_root / "states" / f"{state_key}.json"
    if terminal_path.exists():
        terminal = _read_json(terminal_path)
        if terminal.get("state_identity_sha256") != state_identity:
            raise ValueError("existing native replay state identity drifted")
        return terminal
    if (
        schedule.get("state_id") != query.state_id
        or schedule.get("trajectory_id") != query.trajectory_id
        or tuple(schedule.get("candidate_event_ids", ()))
        != query.candidate_event_step_ids
        or schedule.get("role") != "evaluation"
    ):
        raise ValueError("native replay schedule/query identity drifted")
    reference = schedule.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("native replay schedule omits its reference")
    reference_action = parse_gui_owl_v2_1_output(
        reference["serialized_action"]
    ).canonical_action
    if reference_action.arguments() != reference.get("canonical_action"):
        raise ValueError("native replay serialized/canonical reference drifted")

    decoded: dict[int, Any] = {}

    def decode(payload: bytes) -> Any:
        key = id(payload)
        if key not in decoded:
            decoded[key] = base_decoder(payload)
        return decoded[key]

    def close_decoded_images() -> None:
        for image in decoded.values():
            close = getattr(image, "close", None)
            if close is not None:
                close()
        decoded.clear()

    records = []
    native_generation_count = 0
    for ordinal, raw in enumerate(schedule.get("coalitions", ())):
        coalition = tuple(raw["event_ids"])
        coalition_identity = hashlib.sha256(
            canonical_json_bytes(
                {
                    "coalition": raw,
                    "ordinal": ordinal,
                    "state_identity_sha256": state_identity,
                }
            )
        ).hexdigest()
        progress_path = (
            output_root
            / "progress"
            / state_key
            / f"coalition-{ordinal:02d}-{coalition_identity[:12]}.json"
        )
        if progress_path.exists():
            record = _read_json(progress_path)
            if record.get("coalition_identity_sha256") != coalition_identity:
                raise ValueError("existing native replay coalition identity drifted")
            records.append(record)
            continue
        messages = build_variable_history_messages(
            query,
            coalition,
            image_bytes_loader=lambda image_ref: trajectory.image_payloads[image_ref],
            image_decoder=decode,
        )
        native_generation_count += 1
        try:
            generated = runtime.generate_native_action(messages)
            prediction = generated.parsed_output.canonical_action
            record = {
                "coalition_event_step_ids": list(coalition),
                "coalition_identity_sha256": coalition_identity,
                "comparison": compare_action_components(reference_action, prediction),
                "generation_metadata": dict(generated.metadata),
                "native_output": generated.output_text,
                "predicted_action": prediction.arguments(),
                "sources": list(raw["sources"]),
                "status": COMPLETED_REPLAY_COALITION,
            }
        except GUIOwlV21GenerationParseError as error:
            record = {
                "coalition_event_step_ids": list(coalition),
                "coalition_identity_sha256": coalition_identity,
                "failure": {
                    "message": error.parse_error_message,
                    "type": error.parse_error_type,
                },
                "generation_metadata": dict(error.metadata),
                "native_output": error.output_text,
                "sources": list(raw["sources"]),
                "status": FAILED_REPLAY_PARSE,
            }
        _write_atomic(progress_path, record)
        records.append(record)
    terminal = {
        "candidate_event_step_ids": list(query.candidate_event_step_ids),
        "operations_this_invocation": {
            "native_generation_count": native_generation_count,
        },
        "records": records,
        "reference": dict(reference),
        "role": query.role,
        "source_revision": source_revision,
        "state_id": query.state_id,
        "state_identity_sha256": state_identity,
        "status": COMPLETED_REPLAY_STATE,
        "trajectory_id": query.trajectory_id,
        "winner_model": schedule["winner_model"],
        "worker_index": worker_index,
    }
    _write_atomic(terminal_path, terminal)
    close_decoded_images()
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partition-index", type=int, required=True)
    parser.add_argument("--partition-count", type=int, required=True)
    parser.add_argument("--state-lane-index", type=int, default=0)
    parser.add_argument("--state-lane-count", type=int, default=1)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if not 0 <= args.partition_index < args.partition_count:
        raise ValueError("partition index is outside partition count")
    if not 0 <= args.state_lane_index < args.state_lane_count:
        raise ValueError("state lane index is outside lane count")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is None:
        raise ValueError("source revision must be a full Git SHA")

    source_manifest = _read_json(args.source_root / "manifest.json")
    schedule_manifest_path = args.schedule_root / "manifest.json"
    schedule_manifest = _read_json(schedule_manifest_path)
    if (
        schedule_manifest.get("status") != NATIVE_REPLAY_STATUS
        or not signed_content_hash_is_valid(schedule_manifest)
        or schedule_manifest.get("schema_version") != NATIVE_REPLAY_SCHEMA_VERSION
        or tuple(schedule_manifest.get("methods", ())) != NATIVE_REPLAY_METHODS
        or schedule_manifest.get("state_count") != 320
        or schedule_manifest.get("logical_shard_count") != 256
    ):
        raise ValueError("native replay schedule is not frozen")
    if schedule_manifest.get("source_manifest_sha256") != sha256_file(
        args.source_root / "manifest.json"
    ):
        raise ValueError("native replay/source manifest binding drifted")
    schedule_manifest_sha = sha256_file(schedule_manifest_path)
    manifest_shards = {
        row["logical_shard"]: row for row in schedule_manifest.get("shards", ())
    }
    if set(manifest_shards) != set(range(256)):
        raise ValueError("native replay manifest shard inventory drifted")
    selected_shards = tuple(
        shard
        for shard in source_manifest["shards"]
        if shard["logical_shard"] % args.partition_count == args.partition_index
    )
    runtime, decoder = _runtime(args.repository_root, args.model_dir)
    counts = Counter()
    started = time.time()
    for source_shard in selected_shards:
        logical_shard = source_shard["logical_shard"]
        source_path = (
            args.source_root
            / "trajectory-shards"
            / f"shard-{logical_shard:03d}-of-256.parquet"
        )
        schedule_path = (
            args.schedule_root
            / "schedule-shards"
            / f"shard-{logical_shard:03d}-of-256.jsonl"
        )
        receipt_path = (
            args.schedule_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-256.json"
        )
        receipt = _read_json(receipt_path)
        manifest_shard = manifest_shards.get(logical_shard)
        if (
            manifest_shard is None
            or receipt.get("status") != NATIVE_REPLAY_SHARD_STATUS
            or receipt.get("schedule_sha256") != sha256_file(schedule_path)
            or receipt.get("source_manifest_sha256")
            != schedule_manifest["source_manifest_sha256"]
            or manifest_shard.get("schedule_sha256")
            != receipt.get("schedule_sha256")
            or manifest_shard.get("receipt_sha256") != sha256_file(receipt_path)
        ):
            raise ValueError("native replay shard receipt drifted")
        schedules = tuple(
            row
            for row in _read_jsonl(schedule_path)
            if _state_lane(row["state_id"], args.state_lane_count)
            == args.state_lane_index
        )
        if not schedules:
            continue
        schedules_by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for schedule in schedules:
            schedules_by_trajectory[schedule["trajectory_id"]].append(schedule)
        try:
            from pyarrow import parquet as pq
        except ModuleNotFoundError as error:
            raise RuntimeError("native replay requires pyarrow") from error
        rows = pq.read_table(
            source_path,
            columns=[
                "decision_count",
                "history_events_json",
                "images",
                "ocr_records_json",
                "role",
                "source_id",
                "task_instruction",
            ],
        ).to_pylist()
        for row in rows:
            trajectory_id = str(row["source_id"])
            if trajectory_id not in schedules_by_trajectory:
                continue
            trajectory = trajectory_from_source_row(row)
            queries = {
                query.state_id: query
                for query in build_variable_history_queries(trajectory)
            }
            for schedule in sorted(
                schedules_by_trajectory[trajectory_id],
                key=lambda item: item["state_id"],
            ):
                identity = _state_identity(
                    source_revision=args.source_revision,
                    source_shard_sha256=source_shard["sha256"],
                    schedule_manifest_sha256=schedule_manifest_sha,
                    schedule_receipt_sha256=sha256_file(receipt_path),
                    schedule=schedule,
                )
                result = _run_state(
                    query=queries[schedule["state_id"]],
                    trajectory=trajectory,
                    schedule=schedule,
                    output_root=args.output_root,
                    runtime=runtime,
                    base_decoder=decoder,
                    state_identity=identity,
                    source_revision=args.source_revision,
                    worker_index=args.partition_index,
                )
                counts[result["status"]] += 1
    worker = {
        "counts": dict(sorted(counts.items())),
        "elapsed_seconds": time.time() - started,
        "partition_count": args.partition_count,
        "partition_index": args.partition_index,
        "runtime_metadata": runtime.metadata,
        "schedule_manifest_sha256": schedule_manifest_sha,
        "source_revision": args.source_revision,
        "state_lane_count": args.state_lane_count,
        "state_lane_index": args.state_lane_index,
        "status": "COMPLETED_SET_UTILITY_NATIVE_REPLAY_WORKER",
    }
    worker_name = f"worker-{args.partition_index:03d}-of-{args.partition_count:03d}"
    if args.state_lane_count > 1:
        worker_name += (
            f"-lane-{args.state_lane_index:02d}-of-{args.state_lane_count:02d}"
        )
    _write_atomic(args.output_root / "workers" / f"{worker_name}.json", worker)
    print(json.dumps(worker, sort_keys=True))


if __name__ == "__main__":
    main()
