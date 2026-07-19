#!/usr/bin/env python3
"""Validate all token-cache partitions and publish one root manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from causalcache.set_utility_mvp import canonical_json_bytes


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--partition-count", type=int, required=True)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    cache_root = args.cache_root.resolve()
    destination = cache_root / "manifest.json"
    if destination.exists():
        raise FileExistsError(f"token cache manifest already exists: {destination}")
    input_manifest = _read_json(input_root / "manifest.json")
    states = tuple(
        json.loads(line)
        for line in (input_root / input_manifest["states_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    )
    expected_visual = {
        key
        for state in states
        for key in (state["current_image_key"], *state["event_image_keys"])
    }
    expected_text = {
        key
        for state in states
        for key in (state["instruction_text_key"], *state["event_text_keys"])
    }
    inventory: dict[str, dict[str, Any]] = {}
    parts = []
    for index in range(args.partition_count):
        part_root = cache_root / f"part-{index:02d}"
        manifest_path = part_root / "manifest.json"
        manifest = _read_json(manifest_path)
        if (
            manifest.get("status")
            != "COMPLETED_SET_UTILITY_TOKEN_CACHE_PARTITION"
            or manifest.get("partition_index") != index
            or manifest.get("partition_count") != args.partition_count
            or manifest.get("input_content_sha256")
            != input_manifest["content_sha256"]
        ):
            raise ValueError("token cache partition identity drifted")
        for shard in manifest["shards"]:
            path = part_root / shard["path"]
            payload = path.read_bytes()
            if (
                len(payload) != shard["byte_count"]
                or hashlib.sha256(payload).hexdigest() != shard["sha256"]
            ):
                raise ValueError("token cache shard bytes drifted")
        for key, record in manifest["tensor_inventory"].items():
            if key in inventory:
                raise ValueError(f"token cache key appears twice: {key}")
            inventory[key] = {
                **record,
                "partition": f"part-{index:02d}",
            }
        parts.append(
            {
                "content_sha256": manifest["content_sha256"],
                "manifest": f"part-{index:02d}/manifest.json",
                "partition_index": index,
                "text_count": manifest["text_count"],
                "visual_count": manifest["visual_count"],
            }
        )
    observed_visual = {key.split(":", 1)[1] for key in inventory if key.startswith("visual:")}
    observed_text = {key.split(":", 1)[1] for key in inventory if key.startswith("text:")}
    if observed_visual != expected_visual or observed_text != expected_text:
        raise ValueError("token cache does not exactly cover the input states")
    manifest = {
        "evaluation_labels_included": False,
        "input_content_sha256": input_manifest["content_sha256"],
        "parts": parts,
        "schema_version": "1.0.0",
        "status": "COMPLETED_SET_UTILITY_TOKEN_CACHE",
        "tensor_inventory": inventory,
        "text_count": len(expected_text),
        "visual_count": len(expected_visual),
    }
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    temporary = destination.with_suffix(f".json.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(manifest) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


if __name__ == "__main__":
    main()
