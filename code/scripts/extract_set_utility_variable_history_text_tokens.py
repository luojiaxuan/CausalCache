#!/usr/bin/env python3
"""Extract resumable frozen GUI-Owl input embeddings for snapshot text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


EMBEDDING_TENSOR = "model.language_model.embed_tokens.weight"


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--maximum-text-tokens", type=int, default=128)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is None:
        raise ValueError("source revision must be a full Git SHA")
    if min(args.batch_size, args.shard_size, args.maximum_text_tokens) <= 0:
        raise ValueError("text extraction sizes must be positive")
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from transformers import AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError("text extraction requires torch, safetensors and transformers") from error

    input_manifest_path = args.input_root / "manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if (
        input_manifest.get("status")
        != "COMPLETED_MERGED_VARIABLE_HISTORY_TRAINING_INPUT_SNAPSHOT"
        or input_manifest.get("evaluation_labels_included") is not False
    ):
        raise ValueError("text extraction input status or split firewall drifted")
    text_path = args.input_root / input_manifest["texts_jsonl"]
    if _sha256_file(text_path) != input_manifest["texts_sha256"]:
        raise ValueError("training snapshot text payload drifted")
    texts = {
        row["key"]: row["text"]
        for row in (
            json.loads(line)
            for line in text_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    if len(texts) != input_manifest["text_count"] or any(
        hashlib.sha256(value.encode("utf-8")).hexdigest() != key
        for key, value in texts.items()
    ):
        raise ValueError("training snapshot text inventory drifted")

    snapshot_path = args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    expected = {row["path"]: row for row in snapshot["files"]}
    index_path = args.model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_filename = index["weight_map"][EMBEDDING_TENSOR]
    weight_path = args.model_dir / weight_filename
    for path in (index_path, weight_path):
        record = expected[path.name]
        if path.stat().st_size != record["size"] or _sha256_file(path) != record["sha256"]:
            raise ValueError("GUI-Owl text embedding source differs from frozen snapshot")
    identity = hashlib.sha256(
        (
            _sha256_file(input_manifest_path)
            + _sha256_file(snapshot_path)
            + args.source_revision
            + str(args.maximum_text_tokens)
        ).encode("ascii")
    ).hexdigest()
    keys = sorted(texts)
    shard_groups = [
        keys[start : start + args.shard_size]
        for start in range(0, len(keys), args.shard_size)
    ]
    receipts = []
    pending = []
    for index_number, shard_keys in enumerate(shard_groups):
        tensor_path = args.output_root / "text-shards" / f"text-{index_number:05d}.safetensors"
        receipt_path = args.output_root / "receipts" / f"text-{index_number:05d}.json"
        shard_identity = hashlib.sha256(
            (identity + "".join(shard_keys)).encode("ascii")
        ).hexdigest()
        if tensor_path.exists() or receipt_path.exists():
            if not tensor_path.exists() or not receipt_path.exists():
                raise ValueError("incomplete text shard/receipt pair cannot resume")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                receipt.get("identity_sha256") != shard_identity
                or receipt.get("sha256") != _sha256_file(tensor_path)
                or receipt.get("byte_count") != tensor_path.stat().st_size
            ):
                raise ValueError("existing text token shard drifted")
            receipts.append(receipt)
        else:
            pending.append((index_number, shard_keys, shard_identity))

    if pending:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
            embedding_weight = handle.get_tensor(EMBEDDING_TENSOR)
        if embedding_weight.dtype is not torch.bfloat16 or embedding_weight.shape[1] != 4096:
            raise ValueError("GUI-Owl input embedding geometry drifted")
        for index_number, shard_keys, shard_identity in pending:
            tensors = {}
            for start in range(0, len(shard_keys), args.batch_size):
                batch_keys = shard_keys[start : start + args.batch_size]
                encoded = tokenizer(
                    [texts[key] for key in batch_keys],
                    add_special_tokens=True,
                    max_length=args.maximum_text_tokens,
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                )
                embedded = embedding_weight[encoded["input_ids"]]
                lengths = encoded["attention_mask"].sum(dim=1).tolist()
                for row, (key, length) in enumerate(zip(batch_keys, lengths, strict=True)):
                    tensors[f"t_{key}"] = embedded[row, : int(length)].contiguous()
            tensor_path = args.output_root / "text-shards" / f"text-{index_number:05d}.safetensors"
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = tensor_path.with_suffix(tensor_path.suffix + f".{os.getpid()}.tmp")
            save_file(tensors, str(temporary))
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, tensor_path)
            receipt = {
                "byte_count": tensor_path.stat().st_size,
                "identity_sha256": shard_identity,
                "keys": shard_keys,
                "sha256": _sha256_file(tensor_path),
                "status": "COMPLETED_VARIABLE_HISTORY_TEXT_TOKEN_SHARD",
                "tensor_count": len(tensors),
            }
            _write_atomic(
                args.output_root / "receipts" / f"text-{index_number:05d}.json",
                _canonical_json(receipt),
            )
            receipts.append(receipt)
        del embedding_weight
    receipts.sort(key=lambda row: row["keys"][0])
    if sum(row["tensor_count"] for row in receipts) != len(texts):
        raise RuntimeError("text token shards do not cover the snapshot")
    manifest = {
        "embedding_tensor": EMBEDDING_TENSOR,
        "input_content_sha256": input_manifest["content_sha256"],
        "maximum_text_tokens": args.maximum_text_tokens,
        "model_revision": snapshot["revision"],
        "schema_version": "1.0.0",
        "shards": receipts,
        "source_revision": args.source_revision,
        "status": "COMPLETED_VARIABLE_HISTORY_TEXT_TOKEN_CACHE",
        "text_count": len(texts),
    }
    _write_atomic(args.output_root / "manifest.json", _canonical_json(manifest))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
