#!/usr/bin/env python3
"""Run resumable variable-history restoration labels on one GPU partition."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2_1 import (
    parse_gui_owl_v2_1_output,
    serialize_gui_owl_v2_1_teacher_target,
)
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21GenerationParseError
from causalcache.set_utility_variable_history import load_variable_history_config
from causalcache.set_utility_variable_history_inputs import (
    build_variable_history_messages,
    build_variable_history_queries,
    trajectory_from_source_row,
)


COMPLETED = "COMPLETED_VARIABLE_HISTORY_LABEL_STATE"
SKIPPED = "SKIPPED_VARIABLE_HISTORY_LABEL_STATE"


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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    payload = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _runtime(repository_root: Path, model_dir: Path) -> tuple[Any, Any]:
    from PIL import Image

    from causalcache.policy.gui_owl_variable_history_runtime import (
        GUIOwlVariableHistoryRuntime,
        VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    from causalcache.restoration_v2_gpu_kl import gpu_resident_full_vocab_mean_kl

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

    return runtime, (decode, gpu_resident_full_vocab_mean_kl)


def _reference_log_probs(runtime: Any, logits: Any) -> Any:
    with runtime.torch.inference_mode():
        result = runtime.torch.log_softmax(logits.to(dtype=runtime.torch.float32), dim=-1)
    del logits
    return result


def _measure_kl(kernel: Any, reference_log_probs: Any, logits: Any) -> tuple[float, ...]:
    expanded = reference_log_probs.expand(int(logits.shape[0]), -1, -1)
    result = kernel(expanded, logits, candidate_representation="logits")
    values = tuple(float(item) for item in result.per_example_mean_kl.cpu().tolist())
    del logits
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise RuntimeError("variable-history KL measurement is invalid")
    return values


def _state_identity(
    *,
    source_revision: str,
    scientific_config_sha: str,
    execution_config_sha: str,
    source_shard_sha: str,
    schedule_receipt_sha: str,
    schedule: dict[str, Any],
) -> str:
    payload = {
        "candidate_event_ids": schedule["candidate_event_ids"],
        "coalitions": schedule["coalitions"],
        "execution_config_sha256": execution_config_sha,
        "scientific_config_sha256": scientific_config_sha,
        "schedule_receipt_sha256": schedule_receipt_sha,
        "source_revision": source_revision,
        "source_shard_sha256": source_shard_sha,
        "state_id": schedule["state_id"],
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _microbatches(
    schedule: dict[str, Any],
    *,
    microbatch_size: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    candidates = tuple(schedule["candidate_event_ids"])
    by_cardinality: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for row in schedule["coalitions"]:
        coalition = tuple(row["event_ids"])
        if coalition == candidates:
            continue
        by_cardinality[len(coalition)].append(coalition)
    result = []
    for cardinality in sorted(by_cardinality):
        values = by_cardinality[cardinality]
        result.extend(
            tuple(values[start : start + microbatch_size])
            for start in range(0, len(values), microbatch_size)
        )
    return tuple(result)


def _run_state(
    *,
    query: Any,
    trajectory: Any,
    schedule: dict[str, Any],
    output_root: Path,
    runtime: Any,
    base_decoder: Any,
    kl_kernel: Any,
    state_identity: str,
    source_revision: str,
    scientific_config_sha: str,
    execution_config_sha: str,
    worker_index: int,
    maximum_reference_repeat_kl: float,
    teacher_microbatch_size: int,
) -> dict[str, Any]:
    state_key = query.state_id.replace(":", "_")
    terminal_path = output_root / "states" / f"{state_key}.json"
    if terminal_path.exists():
        terminal = _read_json(terminal_path)
        if terminal.get("state_identity_sha256") != state_identity:
            raise ValueError("existing terminal state identity drifted")
        return terminal
    progress_root = output_root / "progress" / state_key
    reference_path = progress_root / "reference.json"
    decoded: dict[int, Any] = {}

    def decode(payload: bytes) -> Any:
        key = id(payload)
        if key not in decoded:
            decoded[key] = base_decoder(payload)
        return decoded[key]

    def messages(coalition: tuple[int, ...]) -> tuple[Any, ...]:
        return build_variable_history_messages(
            query,
            coalition,
            image_bytes_loader=lambda reference: trajectory.image_payloads[reference],
            image_decoder=decode,
        )

    def close_decoded_images() -> None:
        for image in decoded.values():
            image.close()
        decoded.clear()

    full_messages = messages(query.candidate_event_step_ids)
    if reference_path.exists():
        reference = _read_json(reference_path)
        if reference.get("state_identity_sha256") != state_identity:
            raise ValueError("persisted reference action identity drifted")
        action = parse_gui_owl_v2_1_output(reference["serialized_action"]).canonical_action
    else:
        try:
            first = runtime.generate_native_action(
                full_messages
            ).parsed_output.canonical_action
            second = runtime.generate_native_action(
                full_messages
            ).parsed_output.canonical_action
        except GUIOwlV21GenerationParseError as error:
            terminal = {
                "execution_config_sha256": execution_config_sha,
                "failure_class": type(error).__name__,
                "failure_message": str(error),
                "role": query.role,
                "scientific_config_sha256": scientific_config_sha,
                "source_revision": source_revision,
                "state_id": query.state_id,
                "state_identity_sha256": state_identity,
                "status": SKIPPED,
                "trajectory_id": query.trajectory_id,
                "worker_index": worker_index,
            }
            _write_atomic(terminal_path, terminal)
            close_decoded_images()
            return terminal
        if first != second:
            terminal = {
                "execution_config_sha256": execution_config_sha,
                "failure_class": "ReferenceActionMismatch",
                "role": query.role,
                "scientific_config_sha256": scientific_config_sha,
                "source_revision": source_revision,
                "state_id": query.state_id,
                "state_identity_sha256": state_identity,
                "status": SKIPPED,
                "trajectory_id": query.trajectory_id,
                "worker_index": worker_index,
            }
            _write_atomic(terminal_path, terminal)
            close_decoded_images()
            return terminal
        action = first
        reference = {
            "serialized_action": serialize_gui_owl_v2_1_teacher_target(action),
            "state_identity_sha256": state_identity,
            "status": "COMPLETED_VARIABLE_HISTORY_REFERENCE_ACTION",
        }
        _write_atomic(reference_path, reference)

    reference_logits, reference_metadata = runtime.teacher_forced_distance_logits(
        (full_messages,),
        (action,),
    )
    reference_log_probs = _reference_log_probs(runtime, reference_logits)
    repeat_logits, _ = runtime.teacher_forced_distance_logits(
        (full_messages,),
        (action,),
    )
    repeat_kl = _measure_kl(kl_kernel, reference_log_probs, repeat_logits)[0]
    if repeat_kl > maximum_reference_repeat_kl:
        terminal = {
            "execution_config_sha256": execution_config_sha,
            "failure_class": "ReferenceRepeatKLExceeded",
            "reference_repeat_kl": repeat_kl,
            "role": query.role,
            "scientific_config_sha256": scientific_config_sha,
            "source_revision": source_revision,
            "state_id": query.state_id,
            "state_identity_sha256": state_identity,
            "status": SKIPPED,
            "trajectory_id": query.trajectory_id,
            "worker_index": worker_index,
        }
        _write_atomic(terminal_path, terminal)
        del reference_log_probs
        close_decoded_images()
        return terminal

    distances = {tuple(query.candidate_event_step_ids): 0.0}
    candidate_teacher_call_count = 0
    for batch_index, batch in enumerate(
        _microbatches(schedule, microbatch_size=teacher_microbatch_size)
    ):
        batch_path = progress_root / "microbatches" / f"batch-{batch_index:03d}.json"
        batch_identity = hashlib.sha256(
            (
                state_identity
                + json.dumps(batch, separators=(",", ":"), sort_keys=True)
            ).encode("utf-8")
        ).hexdigest()
        if batch_path.exists():
            record = _read_json(batch_path)
            if record.get("microbatch_identity_sha256") != batch_identity:
                raise ValueError("persisted coalition microbatch identity drifted")
            measured = tuple(float(value) for value in record["distances"])
        else:
            logits, _ = runtime.teacher_forced_distance_logits(
                tuple(messages(coalition) for coalition in batch),
                tuple(action for _ in batch),
            )
            measured = _measure_kl(kl_kernel, reference_log_probs, logits)
            record = {
                "coalitions": [list(coalition) for coalition in batch],
                "distances": list(measured),
                "microbatch_identity_sha256": batch_identity,
                "state_identity_sha256": state_identity,
                "status": "COMPLETED_VARIABLE_HISTORY_COALITION_MICROBATCH",
            }
            _write_atomic(batch_path, record)
            candidate_teacher_call_count += 1
        distances.update(zip(batch, measured, strict=True))
    del reference_log_probs

    ordered = tuple(tuple(row["event_ids"]) for row in schedule["coalitions"])
    if set(distances) != set(ordered):
        raise RuntimeError("completed distances do not cover the frozen schedule")
    terminal = {
        "candidate_event_step_ids": list(query.candidate_event_step_ids),
        "distance_rows": [
            {
                "coalition_event_step_ids": list(coalition),
                "distance": distances[coalition],
            }
            for coalition in ordered
        ],
        "execution_config_sha256": execution_config_sha,
        "input": {
            "current_observation_step": query.decision_step_id - 1,
            "event_observation_steps": list(query.candidate_event_step_ids),
            "instruction": query.task_instruction,
        },
        "operations_this_invocation": {
            "candidate_teacher_call_count": candidate_teacher_call_count,
            "reference_teacher_call_count": 1,
            "reference_repeat_teacher_call_count": 1,
        },
        "reference": {
            "action_token_count": reference_metadata["samples"][0][
                "distance_action_tokens"
            ],
            "repeat_kl": repeat_kl,
            "serialized_action": serialize_gui_owl_v2_1_teacher_target(action),
        },
        "role": query.role,
        "scientific_config_sha256": scientific_config_sha,
        "source_revision": source_revision,
        "state_id": query.state_id,
        "state_identity_sha256": state_identity,
        "status": COMPLETED,
        "trajectory_id": query.trajectory_id,
        "worker_index": worker_index,
    }
    _write_atomic(terminal_path, terminal)
    close_decoded_images()
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--scientific-config", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partition-index", type=int, required=True)
    parser.add_argument("--partition-count", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if not 0 <= args.partition_index < args.partition_count:
        raise ValueError("partition index is outside partition count")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is None:
        raise ValueError("source revision must be a full Git SHA")

    scientific = load_variable_history_config(args.scientific_config)
    execution = _read_json(args.execution_config)
    scientific_sha = _sha256_file(args.scientific_config)
    execution_sha = _sha256_file(args.execution_config)
    if execution["scientific_config_sha256"] != scientific_sha:
        raise ValueError("execution config does not bind the scientific config")
    if execution["logical_shard_count"] != 256:
        raise ValueError("label execution requires 256 logical shards")
    if execution["reference_action_generation_repeats"] != 2:
        raise ValueError("label execution requires exactly two action generations")
    if scientific["reference"]["target_effective_visual_tokens_per_image"] != 480:
        raise ValueError("label execution requires the context-fit 480-token profile")
    source_manifest = _read_json(args.source_root / "manifest.json")
    selected_shards = tuple(
        shard
        for shard in source_manifest["shards"]
        if shard["logical_shard"] % args.partition_count == args.partition_index
    )
    runtime, helpers = _runtime(args.repository_root, args.model_dir)
    decoder, kl_kernel = helpers
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
        schedule_receipt_path = (
            args.schedule_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-256.json"
        )
        schedule_receipt_sha = _sha256_file(schedule_receipt_path)
        schedules = _read_jsonl(schedule_path)
        schedules_by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for schedule in schedules:
            schedules_by_trajectory[schedule["trajectory_id"]].append(schedule)
        try:
            from pyarrow import parquet as pq
        except ModuleNotFoundError as error:
            raise RuntimeError("variable-history labels require pyarrow") from error
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
            trajectory = trajectory_from_source_row(row)
            queries = {
                query.state_id: query
                for query in build_variable_history_queries(trajectory)
            }
            for schedule in sorted(
                schedules_by_trajectory.get(trajectory.trajectory_id, ()),
                key=lambda item: item["state_id"],
            ):
                identity = _state_identity(
                    source_revision=args.source_revision,
                    scientific_config_sha=scientific_sha,
                    execution_config_sha=execution_sha,
                    source_shard_sha=source_shard["sha256"],
                    schedule_receipt_sha=schedule_receipt_sha,
                    schedule=schedule,
                )
                result = _run_state(
                    query=queries[schedule["state_id"]],
                    trajectory=trajectory,
                    schedule=schedule,
                    output_root=args.output_root,
                    runtime=runtime,
                    base_decoder=decoder,
                    kl_kernel=kl_kernel,
                    state_identity=identity,
                    source_revision=args.source_revision,
                    scientific_config_sha=scientific_sha,
                    execution_config_sha=execution_sha,
                    worker_index=args.partition_index,
                    maximum_reference_repeat_kl=float(
                        execution["maximum_reference_repeat_kl"]
                    ),
                    teacher_microbatch_size=int(
                        execution["teacher_microbatch_size"]
                    ),
                )
                counts[result["status"]] += 1
    worker = {
        "counts": dict(sorted(counts.items())),
        "elapsed_seconds": time.time() - started,
        "execution_config_sha256": execution_sha,
        "partition_count": args.partition_count,
        "partition_index": args.partition_index,
        "runtime_metadata": runtime.metadata,
        "scientific_config_sha256": scientific_sha,
        "source_revision": args.source_revision,
        "status": "COMPLETED_VARIABLE_HISTORY_LABEL_WORKER",
    }
    _write_atomic(
        args.output_root
        / "workers"
        / f"worker-{args.partition_index:03d}-of-{args.partition_count:03d}.json",
        worker,
    )
    print(json.dumps(worker, sort_keys=True))


if __name__ == "__main__":
    main()
