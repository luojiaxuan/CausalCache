#!/usr/bin/env python3
"""Repack only visual tokens referenced by a frozen training snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_STATUS = "COMPLETED_MERGED_VARIABLE_HISTORY_TRAINING_INPUT_SNAPSHOT"


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


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_name(key: str) -> str:
    trajectory_id, separator, step = key.partition(":observation:")
    if not trajectory_id or not separator or not step.isdigit():
        raise ValueError("visual key does not encode trajectory and observation")
    return f"tokens__{trajectory_id}__{int(step):03d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-token-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("selected visual token output already exists")
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ModuleNotFoundError as error:
        raise RuntimeError("visual token repacking requires PyTorch and safetensors") from error

    input_manifest = _read_json(args.input_root / "manifest.json")
    if (
        input_manifest.get("status") != EXPECTED_STATUS
        or input_manifest.get("evaluation_labels_included") is not False
    ):
        raise ValueError("visual token repacking input or split firewall drifted")
    states = _read_jsonl(args.input_root / input_manifest["states_jsonl"])
    keys_by_shard: dict[int, set[str]] = {}
    for state in states:
        keys_by_shard.setdefault(int(state["logical_shard"]), set()).update(
            (state["current_image_key"], *state["event_image_keys"])
        )

    shard_records = []
    covered_keys = set()
    missing_shards = []
    for logical_shard, keys in sorted(keys_by_shard.items()):
        source = (
            args.source_token_root
            / "token-shards"
            / f"shard-{logical_shard:03d}-of-256.safetensors"
        )
        if not source.exists():
            missing_shards.append(logical_shard)
            continue
        source_tensors = load_file(str(source), device="cpu")
        selected = {}
        for key in sorted(keys):
            tensor_name = _tensor_name(key)
            try:
                tensor = source_tensors[tensor_name]
            except KeyError as error:
                raise ValueError("source token shard omits a referenced observation") from error
            if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
                raise ValueError("selected visual tokens must be rank-two BF16")
            selected[tensor_name] = tensor.contiguous()
            covered_keys.add(key)
        destination = (
            args.output_root
            / "visual-shards"
            / f"shard-{logical_shard:03d}-of-256.safetensors"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f".safetensors.{os.getpid()}.tmp")
        save_file(selected, str(temporary))
        os.replace(temporary, destination)
        shard_records.append(
            {
                "byte_count": destination.stat().st_size,
                "logical_shard": logical_shard,
                "path": str(destination.relative_to(args.output_root)),
                "sha256": _sha256_file(destination),
                "visual_count": len(selected),
            }
        )
    if missing_shards and not args.allow_partial:
        raise FileNotFoundError("source token root does not contain every required shard")
    if not shard_records:
        raise ValueError("source token root covers no selected visual token shard")
    manifest = {
        "content_sha256": "",
        "covered_visual_count": len(covered_keys),
        "evaluation_labels_included": False,
        "host_alias": args.host_alias,
        "input_content_sha256": input_manifest["content_sha256"],
        "missing_logical_shards": missing_shards,
        "schema_version": "1.0.0",
        "shards": shard_records,
        "status": "COMPLETED_PARTIAL_SELECTED_VISUAL_TOKEN_CACHE",
    }
    manifest["content_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    _write_atomic(args.output_root / "manifest.json", _canonical_json(manifest))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
