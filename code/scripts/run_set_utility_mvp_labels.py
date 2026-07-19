#!/usr/bin/env python3
"""Generate resumable exploratory set-utility labels on four independent GPUs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from causalcache import set_utility_processor_artifacts as processor_artifacts
from causalcache.policy.gui_owl_v2_1 import serialize_gui_owl_v2_1_teacher_target
from causalcache.set_utility_dense import (
    build_dense_feature_state,
    build_dense_messages,
    build_dense_prompt_plan,
    derive_dense_states_from_query_pair,
)
from causalcache.set_utility_label_inputs import (
    LabelExecutionPartition,
    build_joined_utility_query_input,
    read_allowlisted_processor_queries,
)
from causalcache.set_utility_label_producer import (
    build_mixed_fidelity_prompt_plan,
)
from causalcache.set_utility_label_schedule import (
    enumerate_cardinality_capped_subsets,
)
from causalcache.set_utility_mvp import (
    COMPLETED_STATE_STATUS,
    SKIPPED_STATE_STATUS,
    build_feature_state_from_joined,
    canonical_json_bytes,
    feature_state_to_payload,
    read_jsonl,
    select_worker_candidate_records,
    validated_teacher_settings,
)
from causalcache.set_utility_processor_substrate import (
    SelectedTrajectoryAssignment,
)


SCHEMA_VERSION = "1.0.0"
WORKER_COUNT = 4


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _artifact_sha_by_worker(repository_root: Path) -> tuple[str, ...]:
    summary = _read_json(
        repository_root
        / "data/results/set_utility_processor_freeze_v2_image_contract_repair/summary.json"
    )
    inventory = summary["artifact"]["file_inventory"]
    by_path = {item["path"]: item["sha256"] for item in inventory}
    return tuple(
        by_path[f"substrate/processor-substrate-worker-{index:02d}.tar"]
        for index in range(WORKER_COUNT)
    )


def _load_expected_worker(path: Path) -> Any:
    with tarfile.open(path, mode="r:") as archive:
        return processor_artifacts._read_manifest(archive, expected_worker=None)


def _assignments(repository_root: Path) -> tuple[dict[str, Any], str]:
    path = (
        repository_root
        / "data/manifests/set_utility_freeze_b_v2_terminal_index_repair.json"
    )
    raw = path.read_bytes()
    payload = json.loads(raw)
    assignments = {
        item["source_id"]: SelectedTrajectoryAssignment.from_mapping(item)
        for item in payload["assignments"]
    }
    return assignments, hashlib.sha256(raw).hexdigest()


def _runtime_bundle(repository_root: Path, model_dir: Path) -> tuple[Any, Any, Any]:
    from PIL import Image

    from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v3 import (
        GUIOwlV21StrictDeterminismActionStabilityRuntimeV3,
    )
    from causalcache.policy.gui_owl_v2_runtime import (
        FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    from causalcache.restoration_v2_gpu_kl import (
        gpu_resident_full_vocab_mean_kl,
    )
    from causalcache.set_utility_gui_owl_v2_1_throughput_adapter import (
        build_gui_owl_v2_1_throughput_reference_input,
    )

    runtime = GUIOwlV21StrictDeterminismActionStabilityRuntimeV3(
        model_dir=model_dir,
        expected_snapshot_manifest=(
            repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device="cuda:0",
        target_effective_visual_tokens_per_image=(
            FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        ),
    )

    def decode_rgb(payload: bytes) -> Any:
        source = Image.open(io.BytesIO(payload))
        try:
            return source.convert("RGB")
        finally:
            source.close()

    return runtime, build_gui_owl_v2_1_throughput_reference_input, (
        decode_rgb,
        gpu_resident_full_vocab_mean_kl,
    )


def _reference_log_probs(runtime: Any, logits: Any) -> Any:
    torch = runtime.torch
    with torch.inference_mode():
        result = torch.log_softmax(logits.to(dtype=torch.float32), dim=-1)
    del logits
    return result


def _measure_kl(kernel: Any, reference_log_probs: Any, logits: Any) -> tuple[float, ...]:
    expanded = reference_log_probs.expand(int(logits.shape[0]), -1, -1)
    result = kernel(expanded, logits, candidate_representation="logits")
    values = tuple(float(item) for item in result.per_example_mean_kl.cpu().tolist())
    del logits
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise RuntimeError("MVP KL measurement is invalid")
    return values


def _messages(
    joined: Any,
    coalition: tuple[int, ...],
    *,
    input_builder: Any,
    image_decoder: Any,
    plan_builder: Any = build_mixed_fidelity_prompt_plan,
) -> tuple[Any, ...]:
    plan = plan_builder(joined.query, coalition)
    built = input_builder(joined, plan, image_decoder=image_decoder)
    return built.messages


def _run_state(
    joined: Any,
    selected: Any,
    *,
    runtime: Any,
    input_builder: Any,
    image_decoder: Any,
    kl_kernel: Any,
    maximum_reference_repeat_kl: float,
    teacher_microbatch_size: int,
    git_revision: str,
    config_sha256: str,
    worker_index: int,
    plan_builder: Any = build_mixed_fidelity_prompt_plan,
    feature_builder: Any = None,
) -> dict[str, Any]:
    query = joined.query
    candidates = query.candidate_event_step_ids
    full_messages = _messages(
        joined,
        candidates,
        input_builder=input_builder,
        image_decoder=image_decoder,
        plan_builder=plan_builder,
    )
    first = runtime.generate_native_action(full_messages).parsed_output.canonical_action
    second = runtime.generate_native_action(full_messages).parsed_output.canonical_action
    if first != second:
        raise RuntimeError("ReferenceActionMismatch")

    reference_logits, reference_metadata = runtime.teacher_forced_distance_logits(
        (full_messages,),
        (first,),
    )
    reference_log_probs = _reference_log_probs(runtime, reference_logits)
    repeat_logits, _ = runtime.teacher_forced_distance_logits(
        (full_messages,),
        (first,),
    )
    repeat_kl = _measure_kl(kl_kernel, reference_log_probs, repeat_logits)[0]
    if repeat_kl > maximum_reference_repeat_kl:
        raise RuntimeError("ReferenceRepeatKLExceeded")

    coalitions = enumerate_cardinality_capped_subsets(
        candidates,
        maximum_cardinality=query.maximum_labeled_cardinality,
    )
    by_cardinality: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for coalition in coalitions:
        by_cardinality[len(coalition)].append(coalition)
    distances: dict[tuple[int, ...], float] = {}
    teacher_call_count = 2
    for cardinality in sorted(by_cardinality):
        values = by_cardinality[cardinality]
        for offset in range(0, len(values), teacher_microbatch_size):
            batch = values[offset : offset + teacher_microbatch_size]
            messages_batch = tuple(
                _messages(
                    joined,
                    coalition,
                    input_builder=input_builder,
                    image_decoder=image_decoder,
                    plan_builder=plan_builder,
                )
                for coalition in batch
            )
            logits, _ = runtime.teacher_forced_distance_logits(
                messages_batch,
                tuple(first for _ in batch),
            )
            measured = _measure_kl(kl_kernel, reference_log_probs, logits)
            distances.update(zip(batch, measured, strict=True))
            teacher_call_count += 1
    del reference_log_probs

    if feature_builder is None:
        feature = build_feature_state_from_joined(
            joined,
            ocr_records_by_path=selected.query.ocr_records_by_path,
        )
    else:
        feature = feature_builder(joined)
    return {
        "candidate_event_step_ids": list(candidates),
        "config_sha256": config_sha256,
        "distance_rows": [
            {
                "coalition_event_step_ids": list(coalition),
                "distance": distances[coalition],
            }
            for coalition in coalitions
        ],
        "feature": feature_state_to_payload(feature),
        "git_revision": git_revision,
        "maximum_labeled_cardinality": query.maximum_labeled_cardinality,
        "operations": {
            "canonical_action_generation_count": 2,
            "reference_teacher_forward_count": 1,
            "reference_repeat_teacher_forward_count": 1,
            "candidate_teacher_call_count": teacher_call_count - 2,
            "candidate_example_count": len(coalitions),
        },
        "reference": {
            "action": serialize_gui_owl_v2_1_teacher_target(first),
            "action_repeat_equal": True,
            "action_token_count": reference_metadata["samples"][0][
                "distance_action_tokens"
            ],
            "repeat_kl": repeat_kl,
        },
        "role": query.split,
        "schema_version": SCHEMA_VERSION,
        "state_id": query.state_id,
        "status": COMPLETED_STATE_STATUS,
        "trajectory_id": query.trajectory_id,
        "worker_index": worker_index,
    }


def run_worker(args: argparse.Namespace) -> None:
    repository_root = args.repository_root.resolve()
    processor_root = args.processor_root.resolve()
    output_root = args.output_root.resolve()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    maximum_reference_repeat_kl, teacher_microbatch_size = (
        validated_teacher_settings(config)
    )
    config_sha = _sha256_file(config_path)
    worker_index = args.worker_index
    candidate_path = (
        processor_root / "candidate-parts" / f"worker-{worker_index:02d}.jsonl"
    )
    selected_candidates = select_worker_candidate_records(
        read_jsonl(candidate_path),
        candidate_count=config["data"]["candidate_count"],
        query_kind=config["data"]["query_kind"],
        quota_by_role=config["data"]["quota_per_processor_worker"],
        selection_salt=config["data"]["selection_salt"],
    )
    candidates = {item.state_id: item for item in selected_candidates}
    state_ids = tuple(candidates)
    artifact_path = (
        processor_root
        / "substrate"
        / f"processor-substrate-worker-{worker_index:02d}.tar"
    )
    expected_worker = _load_expected_worker(artifact_path)
    expected_sha = _artifact_sha_by_worker(repository_root)[worker_index]
    selected_queries = read_allowlisted_processor_queries(
        artifact_path,
        expected_worker=expected_worker,
        expected_artifact_sha256=expected_sha,
        state_ids=state_ids,
        allowed_roles=tuple(config["data"]["quota_per_processor_worker"]),
    )
    selected_by_state = {item.query.state_id: item for item in selected_queries}
    assignments, request_sha = _assignments(repository_root)
    joined = []
    for state_id in state_ids:
        selected = selected_by_state[state_id]
        joined.append(
            (
                build_joined_utility_query_input(
                    selected,
                    candidate=candidates[state_id],
                    assignment=assignments[candidates[state_id].source_id],
                    execution=LabelExecutionPartition(
                        state_id=state_id,
                        worker_index=worker_index,
                        role_partition=candidates[state_id].role,
                    ),
                    request_manifest_sha256=request_sha,
                ),
                selected,
            )
        )

    runtime, input_builder, runtime_helpers = _runtime_bundle(
        repository_root, args.model_dir.resolve()
    )
    image_decoder, kl_kernel = runtime_helpers
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state_root = output_root / "states"
    counts = Counter()
    started = time.time()
    for item, selected in joined:
        path = state_root / f"{item.query.state_id.replace(':', '_')}.json"
        if path.exists():
            existing = _read_json(path)
            counts[existing["status"]] += 1
            continue
        try:
            result = _run_state(
                item,
                selected,
                runtime=runtime,
                input_builder=input_builder,
                image_decoder=image_decoder,
                kl_kernel=kl_kernel,
                maximum_reference_repeat_kl=maximum_reference_repeat_kl,
                teacher_microbatch_size=teacher_microbatch_size,
                git_revision=git_revision,
                config_sha256=config_sha,
                worker_index=worker_index,
            )
        except Exception as error:
            result = {
                "config_sha256": config_sha,
                "failure_class": error.__class__.__name__,
                "failure_reason": str(error),
                "git_revision": git_revision,
                "role": item.query.split,
                "schema_version": SCHEMA_VERSION,
                "state_id": item.query.state_id,
                "status": SKIPPED_STATE_STATUS,
                "trajectory_id": item.query.trajectory_id,
                "worker_index": worker_index,
            }
        _write_json(path, result)
        counts[result["status"]] += 1
    _write_json(
        output_root / f"worker-{worker_index:02d}.json",
        {
            "config_sha256": config_sha,
            "counts": dict(counts),
            "elapsed_seconds": time.time() - started,
            "git_revision": git_revision,
            "runtime_metadata": dict(runtime.metadata),
            "selected_state_count": len(state_ids),
            "state_ids": list(state_ids),
            "status": "COMPLETED_SET_UTILITY_MVP_WORKER",
            "worker_index": worker_index,
        },
    )


def _backfill_by_trajectory(root: Path) -> dict[str, dict[str, bytes]]:
    manifest = _read_json(root / "manifest.json")
    if manifest.get("status") != "COMPLETED_DENSE_IMAGE_BACKFILL":
        raise ValueError("dense image backfill is incomplete")
    result: dict[str, dict[str, bytes]] = defaultdict(dict)
    for record in manifest["files"]:
        reference = record["reference"]
        payload = (root / reference).read_bytes()
        if (
            len(payload) != record["byte_count"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise ValueError("dense image backfill payload drifted")
        result[record["source_id"]][reference] = payload
    return result


def run_dense_worker(args: argparse.Namespace) -> None:
    repository_root = args.repository_root.resolve()
    processor_root = args.processor_root.resolve()
    output_root = args.output_root.resolve()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    maximum_reference_repeat_kl, teacher_microbatch_size = validated_teacher_settings(
        config
    )
    config_sha = _sha256_file(config_path)
    worker_index = args.worker_index
    artifact_path = (
        processor_root
        / "substrate"
        / f"processor-substrate-worker-{worker_index:02d}.tar"
    )
    expected_worker = _load_expected_worker(artifact_path)
    expected_count = sum(
        item.observation_count - 5 for item in expected_worker.trajectories
    )
    supplemental = _backfill_by_trajectory(args.backfill_root.resolve())
    runtime, _, runtime_helpers = _runtime_bundle(
        repository_root, args.model_dir.resolve()
    )
    image_decoder, kl_kernel = runtime_helpers
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    counts = Counter()
    observed_count = 0
    state_root = output_root / "states"
    started = time.time()
    iterator = processor_artifacts.iter_processor_query_artifact_records(
        artifact_path,
        expected_worker=expected_worker,
    )
    while True:
        try:
            pair = (next(iterator), next(iterator))
        except StopIteration:
            break
        trajectory_id = pair[0].trajectory_id
        states = derive_dense_states_from_query_pair(
            pair,
            supplemental_image_payloads=supplemental.get(trajectory_id),
        )
        trajectory_expected = len(pair[1].history_events) - 4
        if len(states) != trajectory_expected:
            raise RuntimeError(
                f"dense expansion is incomplete for {trajectory_id}: "
                f"{len(states)} != {trajectory_expected}"
            )
        for state in states:
            observed_count += 1
            path = state_root / f"{state.query.state_id.replace(':', '_')}.json"
            if path.exists():
                existing = _read_json(path)
                counts[existing["status"]] += 1
                continue
            try:
                result = _run_state(
                    state,
                    state,
                    runtime=runtime,
                    input_builder=build_dense_messages,
                    image_decoder=image_decoder,
                    kl_kernel=kl_kernel,
                    maximum_reference_repeat_kl=maximum_reference_repeat_kl,
                    teacher_microbatch_size=teacher_microbatch_size,
                    git_revision=git_revision,
                    config_sha256=config_sha,
                    worker_index=worker_index,
                    plan_builder=build_dense_prompt_plan,
                    feature_builder=build_dense_feature_state,
                )
            except Exception as error:
                result = {
                    "config_sha256": config_sha,
                    "failure_class": error.__class__.__name__,
                    "failure_reason": str(error),
                    "git_revision": git_revision,
                    "role": state.query.split,
                    "schema_version": SCHEMA_VERSION,
                    "state_id": state.query.state_id,
                    "status": SKIPPED_STATE_STATUS,
                    "trajectory_id": state.query.trajectory_id,
                    "worker_index": worker_index,
                }
            _write_json(path, result)
            counts[result["status"]] += 1
    if observed_count != expected_count:
        raise RuntimeError(
            f"dense worker state count drifted: {observed_count} != {expected_count}"
        )
    _write_json(
        output_root / f"worker-{worker_index:02d}.json",
        {
            "config_sha256": config_sha,
            "counts": dict(counts),
            "elapsed_seconds": time.time() - started,
            "git_revision": git_revision,
            "runtime_metadata": dict(runtime.metadata),
            "selected_state_count": observed_count,
            "status": "COMPLETED_SET_UTILITY_DENSE_WORKER",
            "worker_index": worker_index,
        },
    )


def run_all(args: argparse.Namespace, *, worker_command: str = "worker") -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    processes = []
    logs = []
    for worker_index in range(WORKER_COUNT):
        log_path = output_root / f"worker-{worker_index:02d}.log"
        log = log_path.open("ab")
        logs.append(log)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            worker_command,
            "--repository-root",
            str(args.repository_root),
            "--processor-root",
            str(args.processor_root),
            "--model-dir",
            str(args.model_dir),
            "--output-root",
            str(args.output_root),
            "--config",
            str(args.config),
            "--worker-index",
            str(worker_index),
        ]
        if worker_command == "dense-worker":
            command.extend(["--backfill-root", str(args.backfill_root)])
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(worker_index)
        processes.append(
            subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        )
    failures = []
    for worker_index, process in enumerate(processes):
        return_code = process.wait()
        if return_code:
            failures.append((worker_index, return_code))
    for log in logs:
        log.close()
    if failures:
        raise SystemExit(f"MVP label workers failed: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("worker", "all", "dense-worker", "dense-all"):
        child = subparsers.add_parser(command)
        child.add_argument("--repository-root", type=Path, required=True)
        child.add_argument("--processor-root", type=Path, required=True)
        child.add_argument("--model-dir", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--config", type=Path, required=True)
        if command in {"worker", "dense-worker"}:
            child.add_argument("--worker-index", type=int, choices=range(4), required=True)
        if command in {"dense-worker", "dense-all"}:
            child.add_argument("--backfill-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        run_worker(args)
    elif args.command == "dense-worker":
        run_dense_worker(args)
    elif args.command == "dense-all":
        run_all(args, worker_command="dense-worker")
    else:
        run_all(args)


if __name__ == "__main__":
    main()
