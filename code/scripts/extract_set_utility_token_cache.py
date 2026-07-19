#!/usr/bin/env python3
"""Cache full frozen GUI-Owl visual and text token sequences in shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2_2_vision_runtime import (
    GPU_UUID_TYPE_PROFILE_V2,
    IMAGE_PROCESSOR_SIZE_PROFILE_V3,
    GUIOwlV22VisionFeatureRuntime,
)
from causalcache.set_utility_mvp import canonical_json_bytes


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("token input rows must be JSON objects")
            rows.append(value)
    return tuple(rows)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _partition(key: str, count: int) -> int:
    return int(hashlib.sha256(key.encode("ascii")).hexdigest(), 16) % count


def _save_shard(
    *,
    tensors: dict[str, Any],
    destination: Path,
    safetensors_save_file: Any,
) -> dict[str, Any]:
    if not tensors:
        raise ValueError("cannot save an empty token shard")
    safetensors_save_file(tensors, str(destination))
    payload = destination.read_bytes()
    return {
        "byte_count": len(payload),
        "path": destination.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "tensor_count": len(tensors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--partition-index", type=int, required=True)
    parser.add_argument("--partition-count", type=int, required=True)
    parser.add_argument("--visual-shard-size", type=int, default=16)
    parser.add_argument("--text-shard-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--maximum-text-tokens", type=int, default=128)
    args = parser.parse_args()

    if not 0 <= args.partition_index < args.partition_count:
        raise ValueError("partition index is outside partition count")
    for name in (
        "visual_shard_size",
        "text_shard_size",
        "text_batch_size",
        "maximum_text_tokens",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")

    repository_root = args.repository_root.resolve()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    part_root = output_root / f"part-{args.partition_index:02d}"
    if part_root.exists():
        raise FileExistsError(f"token cache partition already exists: {part_root}")
    part_root.mkdir(parents=True)
    input_manifest = _read_json(input_root / "manifest.json")
    if (
        input_manifest.get("status") != "MATERIALIZED_SET_UTILITY_TOKEN_INPUTS"
        or input_manifest.get("evaluation_labels_included") is not False
    ):
        raise ValueError("token extraction requires materialized train/tune inputs")
    state_path = input_root / input_manifest["states_jsonl"]
    state_bytes = state_path.read_bytes()
    if hashlib.sha256(state_bytes).hexdigest() != input_manifest["states_sha256"]:
        raise ValueError("token input states bytes drifted")
    states = _read_jsonl(state_path)

    image_keys = sorted(
        {
            key
            for state in states
            for key in (
                state["current_image_key"],
                *state["event_image_keys"],
            )
            if _partition(key, args.partition_count) == args.partition_index
        }
    )
    texts: dict[str, str] = {}
    for state in states:
        candidates = {
            state["instruction_text_key"]: state["instruction"],
            **dict(zip(state["event_text_keys"], state["event_texts"], strict=True)),
        }
        for key, text in candidates.items():
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != key:
                raise ValueError("text cache key differs from its content")
            prior = texts.setdefault(key, text)
            if prior != text:
                raise ValueError("text cache SHA256 collision")
    text_keys = sorted(
        key
        for key in texts
        if _partition(key, args.partition_count) == args.partition_index
    )

    try:
        import torch
        from PIL import Image
        from safetensors.torch import save_file
        from transformers import AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "token extraction requires PyTorch, Pillow, safetensors and Transformers"
        ) from error

    runtime = GUIOwlV22VisionFeatureRuntime(
        model_dir=args.model_dir.resolve(),
        expected_snapshot_manifest=(
            repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device=args.device,
        expected_gpu_uuid=args.expected_gpu_uuid,
        gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2,
        image_processor_size_profile=IMAGE_PROCESSOR_SIZE_PROFILE_V3,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir.resolve(), local_files_only=True
    )
    embedding_layer = runtime.model.model.language_model.embed_tokens

    shard_inventory = []
    tensor_inventory: dict[str, dict[str, Any]] = {}
    pending: dict[str, Any] = {}
    visual_shard_index = 0

    def flush_visual() -> None:
        nonlocal pending, visual_shard_index
        if not pending:
            return
        filename = f"visual-{visual_shard_index:05d}.safetensors"
        record = _save_shard(
            tensors=pending,
            destination=part_root / filename,
            safetensors_save_file=save_file,
        )
        record["kind"] = "visual"
        shard_inventory.append(record)
        for tensor_name, tensor in pending.items():
            key = tensor_name[2:]
            tensor_inventory[f"visual:{key}"] = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "shard": filename,
                "tensor": tensor_name,
            }
        pending = {}
        visual_shard_index += 1

    for start in range(0, len(image_keys), 5):
        actual_keys = image_keys[start : start + 5]
        padded_keys = [*actual_keys]
        while len(padded_keys) < 5:
            padded_keys.append(padded_keys[-1])
        images = []
        for key in padded_keys:
            path = input_root / "images" / f"{key}.png"
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != key:
                raise ValueError("materialized image key differs from its bytes")
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        batch = runtime.encode_five_image_token_sequences(images)
        for key, tensor, token_count in zip(
            actual_keys,
            batch.token_sequences,
            batch.merged_token_counts,
            strict=True,
        ):
            cpu = tensor.detach().to(device="cpu").contiguous()
            if (
                cpu.dtype is not torch.bfloat16
                or cpu.ndim != 2
                or tuple(cpu.shape) != (token_count, 4096)
                or token_count <= 0
            ):
                raise ValueError("GUI-Owl visual token cache geometry drifted")
            pending[f"v_{key}"] = cpu
            if len(pending) >= args.visual_shard_size:
                flush_visual()
    flush_visual()

    pending = {}
    text_shard_index = 0

    def flush_text() -> None:
        nonlocal pending, text_shard_index
        if not pending:
            return
        filename = f"text-{text_shard_index:05d}.safetensors"
        record = _save_shard(
            tensors=pending,
            destination=part_root / filename,
            safetensors_save_file=save_file,
        )
        record["kind"] = "text"
        shard_inventory.append(record)
        for tensor_name, tensor in pending.items():
            key = tensor_name[2:]
            tensor_inventory[f"text:{key}"] = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "shard": filename,
                "tensor": tensor_name,
            }
        pending = {}
        text_shard_index += 1

    for start in range(0, len(text_keys), args.text_batch_size):
        keys = text_keys[start : start + args.text_batch_size]
        encoded = tokenizer(
            [texts[key] for key in keys],
            add_special_tokens=True,
            max_length=args.maximum_text_tokens,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(runtime.device)
        attention_mask = encoded["attention_mask"]
        with torch.inference_mode():
            embeddings = embedding_layer(input_ids)
        if embeddings.dtype is not torch.bfloat16 or embeddings.shape[-1] != 4096:
            raise ValueError("GUI-Owl text embedding geometry drifted")
        lengths = attention_mask.sum(dim=1).tolist()
        for row, (key, length) in enumerate(zip(keys, lengths, strict=True)):
            tensor = embeddings[row, : int(length)].detach().to("cpu").contiguous()
            pending[f"t_{key}"] = tensor
            if len(pending) >= args.text_shard_size:
                flush_text()
    flush_text()

    manifest = {
        "evaluation_labels_included": False,
        "input_content_sha256": input_manifest["content_sha256"],
        "maximum_text_tokens": args.maximum_text_tokens,
        "partition_count": args.partition_count,
        "partition_index": args.partition_index,
        "runtime_metadata": runtime.metadata,
        "schema_version": "1.0.0",
        "shards": shard_inventory,
        "status": "COMPLETED_SET_UTILITY_TOKEN_CACHE_PARTITION",
        "tensor_inventory": tensor_inventory,
        "text_count": len(text_keys),
        "tokenizer_class": tokenizer.__class__.__name__,
        "visual_count": len(image_keys),
    }
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    _write_json(part_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
