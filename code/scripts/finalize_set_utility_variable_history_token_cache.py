#!/usr/bin/env python3
"""Bind merged inputs to existing full visual and text token shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError("safetensors file is shorter than its header length")
        header_length = struct.unpack("<Q", length_bytes)[0]
        header = json.loads(handle.read(header_length))
    result = {
        key: value
        for key, value in header.items()
        if key != "__metadata__"
    }
    if any(
        not isinstance(value, dict)
        or not isinstance(value.get("dtype"), str)
        or not isinstance(value.get("shape"), list)
        for value in result.values()
    ):
        raise ValueError("safetensors header contains an invalid tensor entry")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--visual-subdir", default="visual-shards")
    parser.add_argument("--text-subdir", default="text-cache")
    args = parser.parse_args()
    destination = args.cache_root / "manifest.json"
    if destination.exists():
        raise FileExistsError("variable-history token cache manifest already exists")
    input_manifest = _read_json(args.input_root / "manifest.json")
    if (
        input_manifest.get("status")
        != "COMPLETED_MERGED_VARIABLE_HISTORY_TRAINING_INPUT_SNAPSHOT"
        or input_manifest.get("evaluation_labels_included") is not False
    ):
        raise ValueError("token cache finalizer input status or split firewall drifted")
    states = _read_jsonl(args.input_root / input_manifest["states_jsonl"])
    visual_by_shard: dict[int, set[str]] = {}
    expected_text = set()
    for state in states:
        logical_shard = int(state["logical_shard"])
        visual_by_shard.setdefault(logical_shard, set()).update(
            (state["current_image_key"], *state["event_image_keys"])
        )
        expected_text.update((state["instruction_text_key"], *state["event_text_keys"]))

    inventory: dict[str, dict[str, Any]] = {}
    shard_records = []
    for logical_shard, keys in sorted(visual_by_shard.items()):
        relative = Path(args.visual_subdir) / f"shard-{logical_shard:03d}-of-256.safetensors"
        path = args.cache_root / relative
        header = _safetensors_header(path)
        for key in sorted(keys):
            trajectory_id, _, step_text = key.partition(":observation:")
            if not trajectory_id or not step_text:
                raise ValueError("visual key does not encode trajectory and observation")
            tensor_name = f"tokens__{trajectory_id}__{int(step_text):03d}"
            try:
                tensor = header[tensor_name]
            except KeyError as error:
                raise ValueError("visual token shard omits a snapshot observation") from error
            if tensor["dtype"] != "BF16":
                raise ValueError("visual source tokens must remain BF16")
            inventory[f"visual:{key}"] = {
                "dtype": "torch.bfloat16",
                "partition": args.visual_subdir,
                "shape": tensor["shape"],
                "shard": relative.name,
                "tensor": tensor_name,
            }
        shard_records.append(
            {
                "byte_count": path.stat().st_size,
                "kind": "visual",
                "logical_shard": logical_shard,
                "path": str(relative),
                "sha256": _sha256_file(path),
            }
        )

    text_root = args.cache_root / args.text_subdir
    text_manifest = _read_json(text_root / "manifest.json")
    if (
        text_manifest.get("status") != "COMPLETED_VARIABLE_HISTORY_TEXT_TOKEN_CACHE"
        or text_manifest.get("input_content_sha256") != input_manifest["content_sha256"]
    ):
        raise ValueError("text cache identity differs from merged training input")
    observed_text = set()
    for path in sorted((text_root / "text-shards").glob("text-*.safetensors")):
        header = _safetensors_header(path)
        for tensor_name, tensor in header.items():
            if not tensor_name.startswith("t_"):
                raise ValueError("text token tensor name is invalid")
            if tensor["dtype"] != "BF16":
                raise ValueError("text source tokens must remain BF16")
            key = tensor_name[2:]
            if key in observed_text:
                raise ValueError("text token key appears in more than one shard")
            observed_text.add(key)
            inventory[f"text:{key}"] = {
                "dtype": "torch.bfloat16",
                "partition": f"{args.text_subdir}/text-shards",
                "shape": tensor["shape"],
                "shard": path.name,
                "tensor": tensor_name,
            }
        shard_records.append(
            {
                "byte_count": path.stat().st_size,
                "kind": "text",
                "path": str(Path(args.text_subdir) / "text-shards" / path.name),
                "sha256": _sha256_file(path),
            }
        )
    if observed_text != expected_text:
        raise ValueError("text cache does not exactly cover the training snapshot")
    expected_visual_count = sum(len(keys) for keys in visual_by_shard.values())
    if len(inventory) != expected_visual_count + len(expected_text):
        raise RuntimeError("token cache inventory count drifted")
    manifest = {
        "content_sha256": "",
        "evaluation_labels_included": False,
        "input_content_sha256": input_manifest["content_sha256"],
        "schema_version": "2.0.0",
        "shards": shard_records,
        "status": "COMPLETED_VARIABLE_HISTORY_TOKEN_CACHE",
        "tensor_inventory": inventory,
        "text_count": len(expected_text),
        "visual_count": expected_visual_count,
        "visual_token_profile": input_manifest["visual_token_profile"],
    }
    manifest["content_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    payload = _canonical_json(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
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
                "status": manifest["status"],
                "text_count": manifest["text_count"],
                "visual_count": manifest["visual_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
