#!/usr/bin/env python3
"""Extract resumable full-sequence GUI-Owl selector branch boundaries."""

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
    CONTEXTUAL_ALLOWED_VISUAL_TOKEN_COUNTS,
    CONTEXTUAL_SOURCE_HIDDEN_SIZE,
    CONTEXTUAL_TEXT_TOKEN_LIMIT,
    build_contextual_entity_messages,
)
from causalcache.set_utility_contextual_inputs import (
    CONTEXTUAL_INPUT_STATUS,
    CONTEXTUAL_PROFILE_ID,
    CONTEXTUAL_REQUIREMENT_STATUS,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file
from causalcache.set_utility_selector_boundary_cache import (
    SELECTOR_BOUNDARY_CHUNK_STATUS,
    SELECTOR_BOUNDARY_SCHEMA_VERSION,
    SELECTOR_BOUNDARY_TENSOR_ROLES,
    SELECTOR_BOUNDARY_WORKER_STATUS,
    existing_selector_boundary_chunk,
    load_selector_context_allowlist,
    selector_boundary_chunk_paths,
    selector_boundary_profile_id,
    selector_boundary_tensor_bundle,
    write_atomic,
)
from causalcache.set_utility_selector_branch import capture_selector_boundary_forward


def _requirements(path: Path, receipt: dict[str, Any]) -> list[dict[str, Any]]:
    if receipt.get("status") != CONTEXTUAL_REQUIREMENT_STATUS:
        raise ValueError("selector boundary requirement status drifted")
    if sha256_file(path) != receipt.get("sha256"):
        raise ValueError("selector boundary requirement shard drifted")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != receipt.get("context_count"):
        raise ValueError("selector boundary requirement count drifted")
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
    parser.add_argument("--trainable-layer-count", type=int, default=4)
    parser.add_argument("--context-allowlist", type=Path)
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
        from PIL import Image
        from pyarrow import parquet as pq
        from safetensors.torch import save_file
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "selector boundary extraction requires Pillow, pyarrow and safetensors"
        ) from error

    input_manifest_path = args.input_root / "manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if (
        input_manifest.get("status") != CONTEXTUAL_INPUT_STATUS
        or input_manifest.get("evaluation_labels_included") is not False
        or input_manifest.get("contextual_profile_id") != CONTEXTUAL_PROFILE_ID
    ):
        raise ValueError(
            "selector boundary extraction requires frozen contextual inputs"
        )
    source_manifest_path = args.source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "COMPLETED_VARIABLE_HISTORY_SOURCE":
        raise ValueError("selector boundary source manifest is not complete")
    source_by_shard = {row["logical_shard"]: row for row in source_manifest["shards"]}
    requirement_receipts = {
        row["logical_shard"]: row for row in input_manifest["requirement_shards"]
    }
    allowlist, allowlist_sha256 = load_selector_context_allowlist(
        args.context_allowlist
    )
    if allowlist is not None:
        available = set()
        for receipt in input_manifest["requirement_shards"]:
            available.update(
                row["context_key"]
                for row in _requirements(args.input_root / receipt["path"], receipt)
            )
        if not allowlist.issubset(available):
            raise ValueError("selector boundary allowlist is outside contextual inputs")
    selected_shards = tuple(
        shard
        for shard in range(256)
        if shard % args.partition_count == args.partition_index
    )
    profile_id = selector_boundary_profile_id(
        trainable_layer_count=args.trainable_layer_count
    )
    bindings = {
        "config_sha256": sha256_file(args.config),
        "context_allowlist_sha256": allowlist_sha256,
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "selector_boundary_profile_id": profile_id,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_revision": args.source_revision,
        "trainable_layer_count": args.trainable_layer_count,
    }
    identity_prefix = hashlib.sha256(canonical_json_bytes(bindings)).hexdigest()

    runtime = None
    completed_during_invocation = 0
    resumed_chunks = 0
    visited_contexts = 0
    image_token_id = None
    metadata_example = None
    for logical_shard in selected_shards:
        requirement_receipt = requirement_receipts[logical_shard]
        requirements = _requirements(
            args.input_root / requirement_receipt["path"], requirement_receipt
        )
        if allowlist is not None:
            requirements = [
                row for row in requirements if row["context_key"] in allowlist
            ]
        if not requirements:
            continue
        source_receipt = source_by_shard[logical_shard]
        source_path = (
            args.source_root
            / "trajectory-shards"
            / f"shard-{logical_shard:03d}-of-256.parquet"
        )
        if sha256_file(source_path) != source_receipt["sha256"]:
            raise ValueError("selector boundary source trajectory shard drifted")
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
            prior = existing_selector_boundary_chunk(
                args.output_root,
                logical_shard=logical_shard,
                chunk_index=chunk_index,
                identity_sha256=chunk_identity,
            )
            if prior is not None:
                resumed_chunks += 1
                visited_contexts += len(chunk)
                prior_image_token_id = prior.get("image_token_id")
                if image_token_id is None:
                    image_token_id = prior_image_token_id
                elif image_token_id != prior_image_token_id:
                    raise ValueError("resumed selector boundary image token id drifted")
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
                image_token_id = getattr(runtime.model.config, "image_token_id", None)
                if type(image_token_id) is not int:
                    raise ValueError("GUI-Owl config lacks a scalar image token id")
            tensors = {}
            sequence_lengths = {}
            text_counts = {}
            visual_counts = {}
            for requirement in chunk:
                trajectory_images = images_by_trajectory.get(
                    requirement["trajectory_id"]
                )
                step = requirement["observation_step"]
                if trajectory_images is None or not 0 <= step < len(trajectory_images):
                    raise ValueError("selector boundary image identity drifted")
                payload = trajectory_images[step]["bytes"]
                if not isinstance(payload, bytes) or not payload:
                    raise ValueError(
                        "selector boundary source image payload is invalid"
                    )
                with Image.open(io.BytesIO(payload)) as image:
                    messages = build_contextual_entity_messages(
                        prompt_text=requirement["prompt_text"],
                        image=image.convert("RGB"),
                    )
                    forward = capture_selector_boundary_forward(
                        runtime=runtime,
                        messages=messages,
                        trainable_layer_count=args.trainable_layer_count,
                        include_final_hidden_state=False,
                    )
                bundle = selector_boundary_tensor_bundle(
                    forward=forward, image_token_id=image_token_id
                )
                key = requirement["context_key"]
                for role, tensor in bundle.tensors.items():
                    tensors[f"{role}__{key}"] = tensor
                sequence_lengths[key] = bundle.sequence_length
                text_counts[key] = bundle.text_token_count
                visual_counts[key] = bundle.visual_token_count
                metadata_example = {
                    "boundary_hidden_dtype": str(
                        bundle.tensors["boundary_hidden"].dtype
                    ),
                    "sequence_length": bundle.sequence_length,
                    "text_token_count": bundle.text_token_count,
                    "visual_token_count": bundle.visual_token_count,
                }
                visited_contexts += 1
                del forward, bundle
            tensor_path, receipt_path = selector_boundary_chunk_paths(
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
                **bindings,
                "byte_count": tensor_path.stat().st_size,
                "chunk_index": chunk_index,
                "context_count": len(chunk),
                "context_keys": [row["context_key"] for row in chunk],
                "contexts_per_chunk": args.contexts_per_chunk,
                "identity_sha256": chunk_identity,
                "image_token_id": image_token_id,
                "logical_shard": logical_shard,
                "allowed_visual_token_counts": sorted(
                    CONTEXTUAL_ALLOWED_VISUAL_TOKEN_COUNTS
                ),
                "schema_version": SELECTOR_BOUNDARY_SCHEMA_VERSION,
                "sequence_lengths": sequence_lengths,
                "sha256": sha256_file(tensor_path),
                "source_hidden_size": CONTEXTUAL_SOURCE_HIDDEN_SIZE,
                "status": SELECTOR_BOUNDARY_CHUNK_STATUS,
                "tensor_roles": list(SELECTOR_BOUNDARY_TENSOR_ROLES),
                "text_token_counts": text_counts,
                "text_token_limit": CONTEXTUAL_TEXT_TOKEN_LIMIT,
                "visual_token_counts": visual_counts,
            }
            write_atomic(receipt_path, canonical_json_bytes(receipt, pretty=True))
            completed_during_invocation += 1
        if args.max_contexts is not None and visited_contexts >= args.max_contexts:
            break

    worker = {
        **bindings,
        "completed_chunks_during_invocation": completed_during_invocation,
        "context_scope": "allowlist" if allowlist is not None else "all_contexts",
        "image_token_id": image_token_id,
        "metadata_example": metadata_example,
        "partition_count": args.partition_count,
        "partition_index": args.partition_index,
        "resumed_chunks": resumed_chunks,
        "schema_version": "1.0.0",
        "selected_logical_shard_count": len(selected_shards),
        "status": SELECTOR_BOUNDARY_WORKER_STATUS,
        "visited_context_count": visited_contexts,
    }
    write_atomic(
        args.output_root
        / "workers"
        / f"worker-{args.partition_index:03d}-of-{args.partition_count:03d}.json",
        canonical_json_bytes(worker, pretty=True),
    )
    print(json.dumps(worker, sort_keys=True))


if __name__ == "__main__":
    main()
