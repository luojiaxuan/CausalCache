#!/usr/bin/env python3
"""Run a data-blind four-GPU topology smoke before independent confirm."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import queue as queue_module
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


PROTOCOL_ID = "causalcache_independent_confirm_gpu_topology_smoke_v1"
SCHEMA_VERSION = "1"
WORKER_COUNT = 4
IMAGE_COUNT = 5
FEATURE_REPEATS = 1
FIXED_RGB_IMAGE_SET_SHA256 = (
    "ebdc0e7b23a740052d875d775c6f7dee911339cf0e5b5ff5b27760718fe0ae06"
)
_CUDA_DEVICE = re.compile(r"cuda:[0-9]+")
_GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]+)+")
_RUNTIME_IDENTITY_KEYS = (
    "model_repo",
    "model_revision",
    "snapshot_manifest_sha256",
    "verified_model_file_count",
    "verified_model_total_bytes",
    "transformers_version",
    "transformers_source_sha256",
    "runtime_profile_id",
    "dtype",
    "model_class",
    "requested_attention_implementation",
    "observed_attention_implementation",
)
_RUNTIME_IDENTITY_OPTIONAL_KEYS = (
    "processor_class",
    "image_processor_class",
    "protocol_id",
    "generation_interface",
    "official_tool_schema_sha256",
    "chat_template_file_sha256",
    "chat_template_text_sha256",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_allocation(
    devices: Sequence[str], gpu_uuids: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selected_devices = tuple(devices)
    selected_uuids = tuple(gpu_uuids)
    if (
        len(selected_devices) != WORKER_COUNT
        or len(set(selected_devices)) != WORKER_COUNT
        or any(_CUDA_DEVICE.fullmatch(value) is None for value in selected_devices)
        or len(selected_uuids) != WORKER_COUNT
        or len(set(selected_uuids)) != WORKER_COUNT
        or any(_GPU_UUID.fullmatch(value) is None for value in selected_uuids)
    ):
        raise ValueError("topology smoke requires four unique CUDA devices and GPU UUIDs")
    return selected_devices, selected_uuids


def _fixed_rgb_images() -> tuple[Any, ...]:
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise RuntimeError("topology smoke requires Pillow") from error
    images = []
    width = 64
    height = 64
    for image_index in range(IMAGE_COUNT):
        payload = bytes(
            channel
            for y in range(height)
            for x in range(width)
            for channel in (
                (17 * x + 29 * y + 37 * image_index) % 256,
                (43 * x + 11 * y + 53 * image_index) % 256,
                (7 * x + 61 * y + 97 * image_index) % 256,
            )
        )
        images.append(Image.frombytes("RGB", (width, height), payload))
    return tuple(images)


def _fixed_image_set_sha256(images: Sequence[Any]) -> str:
    if len(tuple(images)) != IMAGE_COUNT:
        raise ValueError("fixed image identity requires exactly five images")
    digest = hashlib.sha256()
    for index, image in enumerate(images):
        if getattr(image, "mode", None) != "RGB" or getattr(image, "size", None) != (
            64,
            64,
        ):
            raise ValueError("fixed topology image geometry drifted")
        digest.update(f"{index}:RGB:64x64\n".encode("ascii"))
        digest.update(image.tobytes())
    observed = digest.hexdigest()
    if observed != FIXED_RGB_IMAGE_SET_SHA256:
        raise RuntimeError("fixed topology RGB image set identity drifted")
    return observed


def _official_tools_messages(images: Sequence[Any]) -> list[dict[str, Any]]:
    from causalcache.policy.gui_owl_v2_1 import (
        GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
        GUI_OWL_V2_1_SYSTEM_PROMPT,
        validate_gui_owl_v2_1_native_messages,
    )

    materialized = tuple(images)
    if len(materialized) != IMAGE_COUNT:
        raise ValueError("official-tools smoke message requires exactly five images")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_V2_1_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "GPU topology smoke with four fixed event images and one "
                        "fixed current image."
                    ),
                },
                *({"type": "image", "image": image} for image in materialized),
                {
                    "type": "text",
                    "text": GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
                },
            ],
        },
    ]
    if validate_gui_owl_v2_1_native_messages(messages) != IMAGE_COUNT:
        raise RuntimeError("official-tools smoke message image count drifted")
    return messages


def _runtime_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in _RUNTIME_IDENTITY_KEYS if key not in metadata]
    if missing:
        raise RuntimeError(f"runtime identity is incomplete: {missing}")
    identity = {
        key: metadata[key]
        for key in (*_RUNTIME_IDENTITY_KEYS, *_RUNTIME_IDENTITY_OPTIONAL_KEYS)
        if key in metadata
    }
    normalized = json.loads(canonical_json_bytes(identity))
    if not isinstance(normalized, dict):
        raise TypeError("runtime identity must be a JSON object")
    return normalized


def _vision_worker_logic(
    *,
    worker_id: str,
    device: str,
    expected_gpu_uuid: str,
    model_dir: str,
    snapshot_manifest: str,
    runtime_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if runtime_factory is None:
        from causalcache.policy.gui_owl_v2_2_vision_runtime import (
            GPU_UUID_TYPE_PROFILE_V2,
            IMAGE_PROCESSOR_SIZE_PROFILE_V3,
            GUIOwlV22VisionFeatureRuntime,
        )

        runtime_factory = GUIOwlV22VisionFeatureRuntime
        runtime_kwargs = {
            "gpu_uuid_type_profile": GPU_UUID_TYPE_PROFILE_V2,
            "image_processor_size_profile": IMAGE_PROCESSOR_SIZE_PROFILE_V3,
        }
    else:
        runtime_kwargs = {}
    images = _fixed_rgb_images()
    image_identity = _fixed_image_set_sha256(images)
    try:
        runtime = runtime_factory(
            model_dir=model_dir,
            expected_snapshot_manifest=snapshot_manifest,
            device=device,
            expected_gpu_uuid=expected_gpu_uuid,
            **runtime_kwargs,
        )
        result = runtime.score_five_images(images, feature_repeats=FEATURE_REPEATS)
    finally:
        for image in images:
            image.close()
    metadata = runtime.metadata
    if (
        metadata.get("device") != device
        or metadata.get("gpu_uuid") != expected_gpu_uuid
        or result.get("feature_repeats") != FEATURE_REPEATS
    ):
        raise RuntimeError("policy-vision smoke worker identity drifted")
    operation_counts = dict(result.get("operation_counts", {}))
    expected_counts = {
        "image_processor_batch_count": 1,
        "policy_vision_feature_forward_count": 1,
        "top_model_forward_count": 0,
        "language_model_forward_count": 0,
        "lm_head_forward_count": 0,
        "generation_count": 0,
    }
    if operation_counts != expected_counts:
        raise RuntimeError("policy-vision topology operation counts drifted")
    return {
        "worker_id": worker_id,
        "device": device,
        "gpu_uuid": expected_gpu_uuid,
        "fixed_rgb_image_set_sha256": image_identity,
        "runtime_identity": _runtime_identity(metadata),
        "operation_counts": operation_counts,
    }


def _teacher_worker_logic(
    *,
    worker_id: str,
    device: str,
    expected_gpu_uuid: str,
    model_dir: str,
    snapshot_manifest: str,
    runtime_factory: Callable[..., Any] | None = None,
    gpu_identity_validator: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if runtime_factory is None:
        from causalcache.policy.gui_owl_v2_2_eager_runtime import GUIOwlV22EagerRuntime

        runtime_factory = GUIOwlV22EagerRuntime
    if gpu_identity_validator is None:
        from causalcache.policy.gui_owl_v2_2_vision_runtime import (
            GPU_UUID_TYPE_PROFILE_V2,
            _validated_gpu_identity,
        )

        def gpu_identity_validator(**kwargs: Any) -> Mapping[str, Any]:
            return _validated_gpu_identity(
                **kwargs, gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2
            )

    from causalcache.policy.gui_owl_v2 import GUIOwlV2Action

    images = _fixed_rgb_images()
    image_identity = _fixed_image_set_sha256(images)
    try:
        messages = _official_tools_messages(images)
        runtime = runtime_factory(
            model_dir=model_dir,
            expected_snapshot_manifest=snapshot_manifest,
            device=device,
            target_effective_visual_tokens_per_image=2560,
        )
        gpu_identity = gpu_identity_validator(
            torch=runtime.torch,
            device=runtime.device,
            expected_gpu_uuid=expected_gpu_uuid,
        )
        logits, teacher_metadata = runtime.teacher_forced_distance_logits(
            (messages,), (GUIOwlV2Action(action="wait"),)
        )
        finite = runtime.torch.isfinite(logits).all()
        finite_value = bool(finite.item())
        if not finite_value:
            raise RuntimeError("teacher-forced topology logits contain non-finite values")
        if (
            str(logits.device) != device
            or str(logits.dtype) != "torch.bfloat16"
            or getattr(logits, "requires_grad", False)
            or getattr(logits, "ndim", None) != 3
            or int(logits.shape[0]) != 1
        ):
            raise RuntimeError("teacher-forced topology logits left BF16 single-GPU form")
        runtime.torch.cuda.synchronize(runtime.device)
    finally:
        for image in images:
            image.close()
    if (
        runtime.metadata.get("device") != device
        or gpu_identity.get("gpu_uuid") != expected_gpu_uuid
        or teacher_metadata.get("device") != device
        or teacher_metadata.get("dtype") != "torch.bfloat16"
        or teacher_metadata.get("batch_size") != 1
        or [sample.get("image_count") for sample in teacher_metadata.get("samples", ())]
        != [IMAGE_COUNT]
    ):
        raise RuntimeError("teacher-forced smoke worker identity or message shape drifted")
    return {
        "worker_id": worker_id,
        "device": device,
        "gpu_uuid": expected_gpu_uuid,
        "fixed_rgb_image_set_sha256": image_identity,
        "runtime_identity": _runtime_identity(runtime.metadata),
        "operation_counts": {
            "teacher_forced_distance_logits_call_count": 1,
            "teacher_forced_example_count": 1,
            "teacher_image_count": IMAGE_COUNT,
            "finite_logits_validation_count": 1,
            "policy_generation_call_count": 0,
            "full_logit_tensor_host_transfer_count": 0,
        },
    }


def _vision_worker_entry(queue: Any, **kwargs: Any) -> None:
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        with contextlib.redirect_stdout(sys.stderr):
            record = _vision_worker_logic(**kwargs)
        queue.put({"ok": True, "record": record})
    except BaseException as error:
        queue.put(
            {
                "ok": False,
                "worker_id": kwargs.get("worker_id"),
                "exception_type": error.__class__.__name__,
                "message": str(error),
            }
        )
        raise


def _teacher_worker_entry(queue: Any, **kwargs: Any) -> None:
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        with contextlib.redirect_stdout(sys.stderr):
            record = _teacher_worker_logic(**kwargs)
        queue.put({"ok": True, "record": record})
    except BaseException as error:
        queue.put(
            {
                "ok": False,
                "worker_id": kwargs.get("worker_id"),
                "exception_type": error.__class__.__name__,
                "message": str(error),
            }
        )
        raise


def _terminate_and_join(
    processes: Sequence[Any],
    *,
    grace_seconds: int,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = clock() + grace_seconds
    for process in processes:
        process.join(timeout=max(0.0, deadline - clock()))
    survivors = [process for process in processes if process.is_alive()]
    for process in survivors:
        kill = getattr(process, "kill", None)
        if not callable(kill):
            raise RuntimeError(f"worker {process.name} survived terminate without kill")
        kill()
    for process in survivors:
        process.join(timeout=max(0.0, deadline - clock()))
    remaining = [process.name for process in processes if process.is_alive()]
    if remaining:
        raise RuntimeError(f"topology workers survived termination: {remaining}")


def _collect_phase(
    *,
    processes: Sequence[Any],
    queue: Any,
    phase_label: str,
    timeout_seconds: int,
    grace_seconds: int,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[Mapping[str, Any], ...]:
    if timeout_seconds <= 0 or grace_seconds <= 0:
        raise ValueError("phase timeout controls must be positive")
    deadline = clock() + timeout_seconds
    responses: list[Mapping[str, Any]] = []
    try:
        while len(responses) < len(processes):
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError(f"{phase_label} exceeded {timeout_seconds} seconds")
            try:
                response = queue.get(timeout=min(1.0, remaining))
            except queue_module.Empty:
                if all(not process.is_alive() for process in processes):
                    break
                continue
            if not isinstance(response, Mapping) or response.get("ok") is not True:
                raise RuntimeError(f"{phase_label} worker failed: {response!r}")
            record = response.get("record")
            if not isinstance(record, Mapping):
                raise RuntimeError(f"{phase_label} worker returned no record")
            responses.append(record)
        for process in processes:
            remaining = deadline - clock()
            if remaining <= 0 and process.is_alive():
                raise TimeoutError(f"{phase_label} exceeded {timeout_seconds} seconds")
            process.join(timeout=max(0.0, remaining))
        failures = [
            {"name": process.name, "exitcode": process.exitcode}
            for process in processes
            if process.exitcode != 0
        ]
        if failures or len(responses) != len(processes):
            raise RuntimeError(f"{phase_label} worker coverage failed: {failures}")
        return tuple(responses)
    except BaseException as error:
        try:
            _terminate_and_join(
                processes,
                grace_seconds=grace_seconds,
                clock=clock,
            )
        except BaseException as cleanup_error:
            error.add_note(
                "worker cleanup failed: "
                f"{cleanup_error.__class__.__name__}: {cleanup_error}"
            )
        raise


def _run_phase(
    *,
    context: Any,
    required_start_method: str,
    target: Callable[..., None],
    phase_label: str,
    model_dir: str,
    snapshot_manifest: str,
    devices: Sequence[str],
    gpu_uuids: Sequence[str],
    timeout_seconds: int,
    grace_seconds: int,
) -> tuple[Mapping[str, Any], ...]:
    get_start_method = getattr(context, "get_start_method", None)
    if callable(get_start_method) and get_start_method() != required_start_method:
        raise RuntimeError(f"{phase_label} requires {required_start_method} context")
    result_queue = context.Queue()
    processes = []
    for index, (device, gpu_uuid) in enumerate(
        zip(devices, gpu_uuids, strict=True)
    ):
        worker_id = f"worker-{index}"
        process = context.Process(
            target=target,
            kwargs={
                "queue": result_queue,
                "worker_id": worker_id,
                "device": device,
                "expected_gpu_uuid": gpu_uuid,
                "model_dir": model_dir,
                "snapshot_manifest": snapshot_manifest,
            },
            name=f"causalcache-topology-{required_start_method}-{worker_id}",
        )
        processes.append(process)
    for process in processes:
        process.start()
    records = _collect_phase(
        processes=processes,
        queue=result_queue,
        phase_label=phase_label,
        timeout_seconds=timeout_seconds,
        grace_seconds=grace_seconds,
    )
    by_worker = {str(record.get("worker_id")): record for record in records}
    if len(by_worker) != WORKER_COUNT:
        raise RuntimeError(f"{phase_label} returned duplicate worker identities")
    ordered = []
    for index, (device, gpu_uuid) in enumerate(
        zip(devices, gpu_uuids, strict=True)
    ):
        worker_id = f"worker-{index}"
        record = by_worker.get(worker_id)
        if (
            record is None
            or record.get("device") != device
            or record.get("gpu_uuid") != gpu_uuid
            or not isinstance(record.get("runtime_identity"), Mapping)
            or not isinstance(record.get("operation_counts"), Mapping)
        ):
            raise RuntimeError(f"{phase_label} four-GPU identity coverage drifted")
        ordered.append(record)
    image_digests = {
        str(record.get("fixed_rgb_image_set_sha256")) for record in ordered
    }
    if len(image_digests) != 1 or not next(iter(image_digests)):
        raise RuntimeError(f"{phase_label} fixed image identity drifted")
    runtime_identities = {
        canonical_json_bytes(record["runtime_identity"]) for record in ordered
    }
    if len(runtime_identities) != 1:
        raise RuntimeError(f"{phase_label} runtime identity differs across GPUs")
    return tuple(ordered)


def _sum_operation_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        counts = record.get("operation_counts")
        if not isinstance(counts, Mapping):
            raise TypeError("worker operation counts must be a mapping")
        for key, value in counts.items():
            if not isinstance(key, str) or type(value) is not int or value < 0:
                raise ValueError("worker operation count schema drifted")
            totals[key] = totals.get(key, 0) + value
    return dict(sorted(totals.items()))


def run_gpu_topology_smoke(
    *,
    model_dir: str | Path,
    snapshot_manifest: str | Path,
    devices: Sequence[str],
    gpu_uuids: Sequence[str],
    phase_timeout_seconds: int,
    worker_termination_grace_seconds: int,
    spawn_context: Any | None = None,
    fork_context: Any | None = None,
    vision_worker_target: Callable[..., None] = _vision_worker_entry,
    teacher_worker_target: Callable[..., None] = _teacher_worker_entry,
) -> dict[str, Any]:
    selected_devices, selected_uuids = _validate_allocation(devices, gpu_uuids)
    if (
        type(phase_timeout_seconds) is not int
        or phase_timeout_seconds <= 0
        or type(worker_termination_grace_seconds) is not int
        or worker_termination_grace_seconds <= 0
    ):
        raise ValueError("topology smoke timeout controls must be positive integers")
    model_root = Path(model_dir).resolve()
    manifest_path = Path(snapshot_manifest).resolve()
    if not model_root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError("explicit local model directory or snapshot manifest is absent")
    if spawn_context is None:
        spawn_context = multiprocessing.get_context("spawn")
    if fork_context is None:
        if os.name != "posix" or "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("teacher-forced topology phase requires POSIX fork")
        fork_context = multiprocessing.get_context("fork")

    vision_records = _run_phase(
        context=spawn_context,
        required_start_method="spawn",
        target=vision_worker_target,
        phase_label="policy-vision topology phase",
        model_dir=str(model_root),
        snapshot_manifest=str(manifest_path),
        devices=selected_devices,
        gpu_uuids=selected_uuids,
        timeout_seconds=phase_timeout_seconds,
        grace_seconds=worker_termination_grace_seconds,
    )
    # note (luojiaxuan): The fork phase is intentionally constructed only after
    # every spawned feature-only worker has joined. The parent never initializes
    # CUDA, so the second phase inherits neither a parent CUDA context nor live
    # feature-runtime state.
    teacher_records = _run_phase(
        context=fork_context,
        required_start_method="fork",
        target=teacher_worker_target,
        phase_label="teacher-forced topology phase",
        model_dir=str(model_root),
        snapshot_manifest=str(manifest_path),
        devices=selected_devices,
        gpu_uuids=selected_uuids,
        timeout_seconds=phase_timeout_seconds,
        grace_seconds=worker_termination_grace_seconds,
    )
    if (
        teacher_records[0]["fixed_rgb_image_set_sha256"]
        != vision_records[0]["fixed_rgb_image_set_sha256"]
    ):
        raise RuntimeError("feature and teacher topology phases used different fixed images")
    allocation = [
        {"worker_id": f"worker-{index}", "device": device, "gpu_uuid": gpu_uuid}
        for index, (device, gpu_uuid) in enumerate(
            zip(selected_devices, selected_uuids, strict=True)
        )
    ]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "scope": {
            "confirm_data_accessed": False,
            "confirm_output_written": False,
            "huggingface_api_accessed": False,
            "network_accessed": False,
            "policy_action_generated": False,
            "artifact_file_written": False,
            "model_load_mode": "explicit_local_files_only",
        },
        "inputs": {
            "model_dir": str(model_root),
            "snapshot_manifest": str(manifest_path),
            "fixed_rgb_image_count": IMAGE_COUNT,
            "fixed_rgb_image_set_sha256": vision_records[0][
                "fixed_rgb_image_set_sha256"
            ],
        },
        "controls": {
            "phase_timeout_seconds": phase_timeout_seconds,
            "worker_termination_grace_seconds": worker_termination_grace_seconds,
        },
        "allocation": allocation,
        "phase_order": ["policy_vision_spawn", "teacher_forced_fork"],
        "phases": {
            "policy_vision_spawn": {
                "start_method": "spawn",
                "worker_count": WORKER_COUNT,
                "image_count_per_worker": IMAGE_COUNT,
                "feature_repeats": FEATURE_REPEATS,
                "operation_counts": _sum_operation_counts(vision_records),
                "workers": list(vision_records),
            },
            "teacher_forced_fork": {
                "start_method": "fork",
                "worker_count": WORKER_COUNT,
                "image_count_per_worker": IMAGE_COUNT,
                "canonical_teacher_action": "wait",
                "operation_counts": _sum_operation_counts(teacher_records),
                "workers": list(teacher_records),
            },
        },
    }
    normalized = json.loads(canonical_json_bytes(receipt))
    if not isinstance(normalized, dict):
        raise TypeError("topology smoke receipt must be a JSON object")
    return normalized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--snapshot-manifest", required=True)
    parser.add_argument("--devices", required=True, nargs=WORKER_COUNT)
    parser.add_argument("--gpu-uuids", required=True, nargs=WORKER_COUNT)
    parser.add_argument("--phase-timeout-seconds", required=True, type=int)
    parser.add_argument(
        "--worker-termination-grace-seconds", required=True, type=int
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    receipt = run_gpu_topology_smoke(
        model_dir=args.model_dir,
        snapshot_manifest=args.snapshot_manifest,
        devices=args.devices,
        gpu_uuids=args.gpu_uuids,
        phase_timeout_seconds=args.phase_timeout_seconds,
        worker_termination_grace_seconds=args.worker_termination_grace_seconds,
    )
    sys.stdout.buffer.write(canonical_json_bytes(receipt) + b"\n")


if __name__ == "__main__":
    main()
