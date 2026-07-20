#!/usr/bin/env python3
"""Extract resumable contextual GUI-Owl entity hidden states."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_variable_history_runtime import (
    GUIOwlVariableHistoryRuntime,
    VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
)
from causalcache.set_utility_contextual_hidden import (
    CONTEXTUAL_SOURCE_HIDDEN_SIZE,
    CONTEXTUAL_TEXT_TOKEN_LIMIT,
    CONTEXTUAL_VISUAL_TOKEN_COUNT,
    build_contextual_entity_messages,
    contextual_hidden_forward,
)
from causalcache.set_utility_contextual_inputs import (
    CONTEXTUAL_INPUT_STATUS,
    CONTEXTUAL_PROFILE_ID,
    CONTEXTUAL_REQUIREMENT_STATUS,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file


CHUNK_STATUS = "COMPLETED_SET_UTILITY_CONTEXTUAL_HIDDEN_CHUNK"
WORKER_STATUS = "COMPLETED_SET_UTILITY_CONTEXTUAL_HIDDEN_WORKER"


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _chunk_paths(
    output_root: Path, *, logical_shard: int, chunk_index: int
) -> tuple[Path, Path]:
    stem = f"chunk-{chunk_index:05d}"
    relative = Path(f"shard-{logical_shard:03d}-of-256")
    return (
        output_root / "context-shards" / relative / f"{stem}.safetensors",
        output_root / "receipts" / relative / f"{stem}.json",
    )


def _existing_chunk(
    output_root: Path,
    *,
    logical_shard: int,
    chunk_index: int,
    identity_sha256: str,
) -> dict[str, Any] | None:
    tensor_path, receipt_path = _chunk_paths(
        output_root, logical_shard=logical_shard, chunk_index=chunk_index
    )
    if not tensor_path.exists() and not receipt_path.exists():
        return None
    if not tensor_path.exists() or not receipt_path.exists():
        raise ValueError("incomplete contextual chunk/receipt pair cannot resume")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != CHUNK_STATUS
        or receipt.get("identity_sha256") != identity_sha256
        or receipt.get("sha256") != sha256_file(tensor_path)
        or receipt.get("byte_count") != tensor_path.stat().st_size
    ):
        raise ValueError("existing contextual hidden chunk drifted")
    return receipt


def _load_requirements(path: Path, receipt: dict[str, Any]) -> list[dict[str, Any]]:
    if receipt.get("status") != CONTEXTUAL_REQUIREMENT_STATUS:
        raise ValueError("contextual requirement receipt status drifted")
    if sha256_file(path) != receipt.get("sha256"):
        raise ValueError("contextual requirement shard drifted")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != receipt.get("context_count"):
        raise ValueError("contextual requirement count drifted")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--partition-index", type=int, required=True)
    parser.add_argument("--partition-count", type=int, required=True)
    parser.add_argument("--contexts-per-chunk", type=int, default=8)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-contexts", type=int)
    args = parser.parse_args()
    if not 0 <= args.partition_index < args.partition_count:
        raise ValueError("partition index is outside partition count")
    if not 1 <= args.contexts_per_chunk <= 64:
        raise ValueError("contexts per chunk must be in [1,64]")
    if args.max_contexts is not None and args.max_contexts <= 0:
        raise ValueError("max contexts must be positive")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is None:
        raise ValueError("source revision must be a full Git SHA")

    try:
        import torch
        from PIL import Image
        from pyarrow import parquet as pq
        from safetensors.torch import save_file
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "contextual extraction requires torch, Pillow, pyarrow and safetensors"
        ) from error

    input_manifest_path = args.input_root / "manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if (
        input_manifest.get("status") != CONTEXTUAL_INPUT_STATUS
        or input_manifest.get("evaluation_labels_included") is not False
        or input_manifest.get("contextual_profile_id") != CONTEXTUAL_PROFILE_ID
    ):
        raise ValueError("contextual hidden extraction requires frozen train/tune inputs")
    source_manifest_path = args.source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "COMPLETED_VARIABLE_HISTORY_SOURCE":
        raise ValueError("variable-history source is not complete")
    source_by_shard = {
        row["logical_shard"]: row for row in source_manifest["shards"]
    }
    requirement_receipts = {
        row["logical_shard"]: row for row in input_manifest["requirement_shards"]
    }
    selected_shards = tuple(
        shard
        for shard in range(256)
        if shard % args.partition_count == args.partition_index
    )
    identity_prefix = hashlib.sha256(
        (
            sha256_file(args.config)
            + sha256_file(input_manifest_path)
            + sha256_file(source_manifest_path)
            + args.source_revision
            + CONTEXTUAL_PROFILE_ID
        ).encode("utf-8")
    ).hexdigest()

    runtime = None
    completed_during_invocation = 0
    resumed_chunks = 0
    visited_contexts = 0
    metadata_example = None
    for logical_shard in selected_shards:
        requirement_receipt = requirement_receipts[logical_shard]
        requirement_path = args.input_root / requirement_receipt["path"]
        requirements = _load_requirements(requirement_path, requirement_receipt)
        if not requirements:
            continue
        source_receipt = source_by_shard[logical_shard]
        source_path = (
            args.source_root
            / "trajectory-shards"
            / f"shard-{logical_shard:03d}-of-256.parquet"
        )
        if sha256_file(source_path) != source_receipt["sha256"]:
            raise ValueError("contextual source trajectory shard drifted")
        source_rows = pq.read_table(
            source_path, columns=["images", "source_id"]
        ).to_pylist()
        images_by_trajectory = {
            str(row["source_id"]): row["images"] for row in source_rows
        }
        for chunk_index, start in enumerate(
            range(0, len(requirements), args.contexts_per_chunk)
        ):
            chunk = requirements[start : start + args.contexts_per_chunk]
            chunk_identity = hashlib.sha256(
                (
                    identity_prefix
                    + requirement_receipt["sha256"]
                    + source_receipt["sha256"]
                    + "".join(row["context_key"] for row in chunk)
                ).encode("utf-8")
            ).hexdigest()
            if _existing_chunk(
                args.output_root,
                logical_shard=logical_shard,
                chunk_index=chunk_index,
                identity_sha256=chunk_identity,
            ) is not None:
                resumed_chunks += 1
                visited_contexts += len(chunk)
                continue
            if args.max_contexts is not None and visited_contexts >= args.max_contexts:
                break
            if runtime is None:
                runtime = GUIOwlVariableHistoryRuntime(
                    model_dir=args.model_dir,
                    expected_snapshot_manifest=(
                        args.repository_root
                        / "code/configs/gui_owl_1_5_8b_snapshot.json"
                    ),
                    device=args.device,
                    target_effective_visual_tokens_per_image=(
                        VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
                    ),
                )
            tensors = {}
            text_counts = {}
            for requirement in chunk:
                trajectory_images = images_by_trajectory.get(
                    requirement["trajectory_id"]
                )
                step = requirement["observation_step"]
                if trajectory_images is None or not 0 <= step < len(trajectory_images):
                    raise ValueError("contextual requirement image identity drifted")
                payload = trajectory_images[step]["bytes"]
                if not isinstance(payload, bytes) or not payload:
                    raise ValueError("contextual source image payload is invalid")
                with Image.open(io.BytesIO(payload)) as image:
                    messages = build_contextual_entity_messages(
                        prompt_text=requirement["prompt_text"],
                        image=image.convert("RGB"),
                    )
                    visual, text, metadata = contextual_hidden_forward(
                        runtime=runtime, messages=messages
                    )
                key = requirement["context_key"]
                tensors[f"visual__{key}"] = visual
                tensors[f"text__{key}"] = text
                text_counts[key] = int(text.shape[0])
                metadata_example = metadata
                visited_contexts += 1
            tensor_path, receipt_path = _chunk_paths(
                args.output_root,
                logical_shard=logical_shard,
                chunk_index=chunk_index,
            )
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = tensor_path.with_suffix(
                tensor_path.suffix + f".{os.getpid()}.tmp"
            )
            save_file(tensors, str(temporary))
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, tensor_path)
            receipt = {
                "byte_count": tensor_path.stat().st_size,
                "chunk_index": chunk_index,
                "context_count": len(chunk),
                "context_keys": [row["context_key"] for row in chunk],
                "contextual_profile_id": CONTEXTUAL_PROFILE_ID,
                "identity_sha256": chunk_identity,
                "logical_shard": logical_shard,
                "sha256": sha256_file(tensor_path),
                "source_hidden_size": CONTEXTUAL_SOURCE_HIDDEN_SIZE,
                "source_revision": args.source_revision,
                "status": CHUNK_STATUS,
                "text_token_counts": text_counts,
                "text_token_limit": CONTEXTUAL_TEXT_TOKEN_LIMIT,
                "visual_token_count": CONTEXTUAL_VISUAL_TOKEN_COUNT,
            }
            _write_atomic(receipt_path, canonical_json_bytes(receipt, pretty=True))
            completed_during_invocation += 1
        if args.max_contexts is not None and visited_contexts >= args.max_contexts:
            break

    worker = {
        "completed_chunks_during_invocation": completed_during_invocation,
        "contextual_profile_id": CONTEXTUAL_PROFILE_ID,
        "metadata_example": metadata_example,
        "partition_count": args.partition_count,
        "partition_index": args.partition_index,
        "resumed_chunks": resumed_chunks,
        "schema_version": "1.0.0",
        "selected_logical_shard_count": len(selected_shards),
        "source_revision": args.source_revision,
        "status": WORKER_STATUS,
        "visited_context_count": visited_contexts,
    }
    _write_atomic(
        args.output_root
        / "workers"
        / f"worker-{args.partition_index:03d}-of-{args.partition_count:03d}.json",
        canonical_json_bytes(worker, pretty=True),
    )
    print(json.dumps(worker, sort_keys=True))


if __name__ == "__main__":
    main()
