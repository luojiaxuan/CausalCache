#!/usr/bin/env python3
"""Extract resumable full GUI-Owl visual tokens for all source observations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2_vision import (
    extract_spatial_merger_token_sequences,
)
from causalcache.policy.gui_owl_variable_history_runtime import (
    GUIOwlVariableHistoryRuntime,
    VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
)


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


def _existing_receipt(
    output_root: Path,
    *,
    logical_shard: int,
    identity: str,
) -> dict[str, Any] | None:
    tensor_path = output_root / "token-shards" / f"shard-{logical_shard:03d}-of-256.safetensors"
    receipt_path = output_root / "receipts" / f"shard-{logical_shard:03d}-of-256.json"
    if not tensor_path.exists() and not receipt_path.exists():
        return None
    if not tensor_path.exists() or not receipt_path.exists():
        raise ValueError("incomplete token shard/receipt pair cannot resume")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("identity_sha256") != identity
        or receipt.get("sha256") != _sha256_file(tensor_path)
        or receipt.get("byte_count") != tensor_path.stat().st_size
    ):
        raise ValueError("existing token shard receipt drifted")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--partition-index", type=int, required=True)
    parser.add_argument("--partition-count", type=int, required=True)
    parser.add_argument("--image-batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 0 <= args.partition_index < args.partition_count:
        raise ValueError("partition index is outside partition count")
    if not 1 <= args.image_batch_size <= 64:
        raise ValueError("image batch size must be in [1,64]")

    try:
        import torch
        from PIL import Image
        from pyarrow import parquet as pq
        from safetensors.torch import save_file
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "variable-history token extraction requires torch, Pillow, pyarrow, and safetensors"
        ) from error

    source_manifest_path = args.source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "COMPLETED_VARIABLE_HISTORY_SOURCE":
        raise ValueError("variable-history source is not complete")
    selected_shards = tuple(
        shard
        for shard in source_manifest["shards"]
        if shard["logical_shard"] % args.partition_count == args.partition_index
    )
    identity_prefix = hashlib.sha256(
        (
            _sha256_file(args.config)
            + _sha256_file(source_manifest_path)
            + str(VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE)
        ).encode("utf-8")
    ).hexdigest()

    pending = []
    for shard in selected_shards:
        identity = hashlib.sha256(
            f"{identity_prefix}:{shard['logical_shard']}:{shard['sha256']}".encode(
                "utf-8"
            )
        ).hexdigest()
        if _existing_receipt(
            args.output_root,
            logical_shard=shard["logical_shard"],
            identity=identity,
        ) is None:
            pending.append((shard, identity))

    runtime = None
    completed = []
    for shard, identity in pending:
        if runtime is None:
            runtime = GUIOwlVariableHistoryRuntime(
                model_dir=args.model_dir,
                expected_snapshot_manifest=(
                    args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
                ),
                device=args.device,
                target_effective_visual_tokens_per_image=(
                    VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
                ),
            )
        logical_shard = shard["logical_shard"]
        source_path = (
            args.source_root
            / "trajectory-shards"
            / f"shard-{logical_shard:03d}-of-256.parquet"
        )
        table = pq.read_table(source_path, columns=["images", "source_id"])
        source_rows = table.to_pylist()
        image_records = [
            (row["source_id"], step, image["bytes"])
            for row in source_rows
            for step, image in enumerate(row["images"])
        ]
        tensors: dict[str, Any] = {}
        means_by_trajectory: dict[str, list[Any]] = {
            row["source_id"]: [] for row in source_rows
        }
        counts_by_trajectory: dict[str, list[int]] = {
            row["source_id"]: [] for row in source_rows
        }
        for start in range(0, len(image_records), args.image_batch_size):
            batch_records = image_records[start : start + args.image_batch_size]
            images = []
            for _, _, payload in batch_records:
                with Image.open(io.BytesIO(payload)) as image:
                    images.append(image.convert("RGB"))
            encoded = runtime.processor.image_processor(
                images=images,
                return_tensors="pt",
            )
            pixel_values = encoded["pixel_values"].to(
                device=runtime.device,
                dtype=torch.bfloat16,
            )
            image_grid_thw = encoded["image_grid_thw"].to(runtime.device)
            with torch.inference_mode():
                batch = extract_spatial_merger_token_sequences(
                    model=runtime.model,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    runtime_identity=runtime.runtime_identity,
                )
            if len(batch.token_sequences) != len(batch_records):
                raise RuntimeError("vision token batch size drifted")
            for (trajectory_id, step, _), tokens, token_count in zip(
                batch_records,
                batch.token_sequences,
                batch.merged_token_counts,
                strict=True,
            ):
                cpu = tokens.detach().to(device="cpu").contiguous()
                if cpu.dtype is not torch.bfloat16 or tuple(cpu.shape) != (
                    token_count,
                    4096,
                ):
                    raise ValueError("variable-history visual token geometry drifted")
                tensors[f"tokens__{trajectory_id}__{step:03d}"] = cpu
                mean = cpu.to(dtype=torch.float32).mean(dim=0)
                mean = mean / mean.norm(p=2).clamp_min(1e-12)
                means_by_trajectory[trajectory_id].append(mean)
                counts_by_trajectory[trajectory_id].append(int(token_count))
            del batch, encoded, image_grid_thw, pixel_values

        for row in source_rows:
            trajectory_id = row["source_id"]
            expected_count = len(row["images"])
            if (
                len(means_by_trajectory[trajectory_id]) != expected_count
                or len(counts_by_trajectory[trajectory_id]) != expected_count
            ):
                raise RuntimeError("trajectory visual cache is incomplete")
            tensors[f"means__{trajectory_id}"] = torch.stack(
                means_by_trajectory[trajectory_id]
            ).contiguous()
            tensors[f"counts__{trajectory_id}"] = torch.tensor(
                counts_by_trajectory[trajectory_id], dtype=torch.int32
            )
        tensor_path = (
            args.output_root
            / "token-shards"
            / f"shard-{logical_shard:03d}-of-256.safetensors"
        )
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = tensor_path.with_suffix(tensor_path.suffix + f".{os.getpid()}.tmp")
        save_file(tensors, str(temporary))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, tensor_path)
        receipt = {
            "byte_count": tensor_path.stat().st_size,
            "identity_sha256": identity,
            "logical_shard": logical_shard,
            "observation_count": len(image_records),
            "runtime_profile_id": runtime.metadata["runtime_profile_id"],
            "sha256": _sha256_file(tensor_path),
            "status": "COMPLETED_VARIABLE_HISTORY_TOKEN_SHARD",
            "target_effective_visual_tokens_per_image": (
                VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
            ),
            "trajectory_count": len(source_rows),
        }
        _write_atomic(
            args.output_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-256.json",
            _canonical_json(receipt),
        )
        completed.append(receipt)

    worker = {
        "completed_during_invocation": len(completed),
        "partition_count": args.partition_count,
        "partition_index": args.partition_index,
        "schema_version": "1.0.0",
        "selected_logical_shard_count": len(selected_shards),
        "status": "COMPLETED_VARIABLE_HISTORY_TOKEN_WORKER",
    }
    _write_atomic(
        args.output_root
        / "workers"
        / f"worker-{args.partition_index:03d}-of-{args.partition_count:03d}.json",
        _canonical_json(worker),
    )
    print(json.dumps(worker, sort_keys=True))


if __name__ == "__main__":
    main()
