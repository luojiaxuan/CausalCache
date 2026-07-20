#!/usr/bin/env python3
"""Validate and index a complete contextual hidden-state cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

from causalcache.set_utility_contextual_hidden import (
    CONTEXTUAL_ALLOWED_VISUAL_TOKEN_COUNTS,
    CONTEXTUAL_SOURCE_HIDDEN_SIZE,
    CONTEXTUAL_TEXT_TOKEN_LIMIT,
)
from causalcache.set_utility_contextual_inputs import (
    CONTEXTUAL_INPUT_STATUS,
    CONTEXTUAL_PROFILE_ID,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file


CACHE_STATUS = "COMPLETED_SET_UTILITY_CONTEXTUAL_HIDDEN_CACHE"
CHUNK_STATUS = "COMPLETED_SET_UTILITY_CONTEXTUAL_HIDDEN_CHUNK"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError("safetensors file is shorter than its header length")
        header = json.loads(handle.read(struct.unpack("<Q", raw_length)[0]))
    return {key: value for key, value in header.items() if key != "__metadata__"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    destination = args.cache_root / "manifest.json"
    if destination.exists():
        raise FileExistsError("contextual cache manifest already exists")
    input_manifest_path = args.input_root / "manifest.json"
    input_manifest = _read_json(input_manifest_path)
    if (
        input_manifest.get("status") != CONTEXTUAL_INPUT_STATUS
        or input_manifest.get("evaluation_labels_included") is not False
        or input_manifest.get("contextual_profile_id") != CONTEXTUAL_PROFILE_ID
    ):
        raise ValueError("contextual cache finalizer requires frozen train/tune inputs")
    expected = {
        json.loads(line)["context_key"]
        for receipt in input_manifest["requirement_shards"]
        for line in (args.input_root / receipt["path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    }
    if len(expected) != input_manifest["context_count"]:
        raise ValueError("contextual requirement inventory drifted")

    inventory = {}
    shard_records = []
    observed = set()
    receipt_paths = sorted((args.cache_root / "receipts").glob("**/chunk-*.json"))
    for receipt_path in receipt_paths:
        receipt = _read_json(receipt_path)
        if (
            receipt.get("status") != CHUNK_STATUS
            or receipt.get("contextual_profile_id") != CONTEXTUAL_PROFILE_ID
            or receipt.get("allowed_visual_token_counts")
            != sorted(CONTEXTUAL_ALLOWED_VISUAL_TOKEN_COUNTS)
            or receipt.get("text_token_limit") != CONTEXTUAL_TEXT_TOKEN_LIMIT
            or receipt.get("source_hidden_size") != CONTEXTUAL_SOURCE_HIDDEN_SIZE
        ):
            raise ValueError("contextual chunk receipt contract drifted")
        relative = (
            Path("context-shards")
            / receipt_path.parent.name
            / receipt_path.with_suffix(".safetensors").name
        )
        tensor_path = args.cache_root / relative
        if (
            not tensor_path.is_file()
            or sha256_file(tensor_path) != receipt.get("sha256")
            or tensor_path.stat().st_size != receipt.get("byte_count")
        ):
            raise ValueError("contextual chunk bytes drifted")
        header = _safetensors_header(tensor_path)
        context_keys = receipt["context_keys"]
        if (
            len(context_keys) != receipt["context_count"]
            or len(context_keys) != len(set(context_keys))
            or observed.intersection(context_keys)
        ):
            raise ValueError("contextual chunk inventory overlaps or drifts")
        expected_tensor_names = {
            f"{role}__{key}" for key in context_keys for role in ("visual", "text")
        }
        if set(header) != expected_tensor_names:
            raise ValueError("contextual tensor inventory differs from its receipt")
        for key in context_keys:
            visual = header[f"visual__{key}"]
            text = header[f"text__{key}"]
            visual_shape = visual.get("shape")
            text_shape = text.get("shape")
            if (
                visual.get("dtype") != "BF16"
                or not isinstance(visual_shape, list)
                or len(visual_shape) != 2
                or visual_shape[0] not in CONTEXTUAL_ALLOWED_VISUAL_TOKEN_COUNTS
                or visual_shape[1] != CONTEXTUAL_SOURCE_HIDDEN_SIZE
                or receipt["visual_token_counts"].get(key) != visual_shape[0]
            ):
                raise ValueError("contextual visual tensor geometry drifted")
            if (
                text.get("dtype") != "BF16"
                or not isinstance(text_shape, list)
                or len(text_shape) != 2
                or not 1 <= text_shape[0] <= CONTEXTUAL_TEXT_TOKEN_LIMIT
                or text_shape[1] != CONTEXTUAL_SOURCE_HIDDEN_SIZE
                or receipt["text_token_counts"].get(key) != text_shape[0]
            ):
                raise ValueError("contextual text tensor geometry drifted")
            partition = str(relative.parent)
            inventory[f"visual:{key}"] = {
                "dtype": "torch.bfloat16",
                "partition": partition,
                "shape": visual_shape,
                "shard": relative.name,
                "tensor": f"visual__{key}",
            }
            inventory[f"text:{key}"] = {
                "dtype": "torch.bfloat16",
                "partition": partition,
                "shape": text_shape,
                "shard": relative.name,
                "tensor": f"text__{key}",
            }
        observed.update(context_keys)
        shard_records.append(
            {
                "byte_count": tensor_path.stat().st_size,
                "context_count": len(context_keys),
                "logical_shard": receipt["logical_shard"],
                "path": str(relative),
                "sha256": receipt["sha256"],
            }
        )
    if observed != expected:
        raise ValueError("contextual hidden cache is incomplete")
    manifest = {
        "content_sha256": "",
        "context_count": len(observed),
        "contextual_profile_id": CONTEXTUAL_PROFILE_ID,
        "evaluation_labels_included": False,
        "input_content_sha256": input_manifest["content_sha256"],
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "schema_version": "3.0.0",
        "shards": shard_records,
        "source_hidden_size": CONTEXTUAL_SOURCE_HIDDEN_SIZE,
        "status": CACHE_STATUS,
        "tensor_inventory": inventory,
        "text_count": len(observed),
        "visual_count": len(observed),
        "visual_token_profile": CONTEXTUAL_PROFILE_ID,
    }
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    payload = canonical_json_bytes(manifest, pretty=True)
    temporary = destination.with_suffix(f".json.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "content_sha256": manifest["content_sha256"],
                "context_count": manifest["context_count"],
                "shard_count": len(shard_records),
                "status": manifest["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
