#!/usr/bin/env python3
"""Merge disjoint compact visual-token exports into one cache directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
    parser.add_argument("--input-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input_roots) < 2:
        raise ValueError("selected visual token merge requires at least two inputs")
    if args.output_root.exists():
        raise FileExistsError("merged selected visual token output already exists")

    manifests = [_read_json(root / "manifest.json") for root in args.input_roots]
    if any(
        manifest.get("status") != "COMPLETED_PARTIAL_SELECTED_VISUAL_TOKEN_CACHE"
        or manifest.get("evaluation_labels_included") is not False
        for manifest in manifests
    ):
        raise ValueError("selected visual token input status drifted")
    input_hashes = {manifest["input_content_sha256"] for manifest in manifests}
    if len(input_hashes) != 1:
        raise ValueError("selected visual token inputs bind different snapshots")

    observed_shards = set()
    shard_records = []
    for root, manifest in zip(args.input_roots, manifests, strict=True):
        for record in manifest["shards"]:
            logical_shard = int(record["logical_shard"])
            if logical_shard in observed_shards:
                raise ValueError("selected visual token inputs overlap")
            observed_shards.add(logical_shard)
            source = root / record["path"]
            if _sha256_file(source) != record["sha256"]:
                raise ValueError("selected visual token shard hash drifted")
            destination = args.output_root / "visual-shards" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            shard_records.append(
                {
                    **record,
                    "path": str(destination.relative_to(args.output_root)),
                }
            )
    manifest = {
        "content_sha256": "",
        "evaluation_labels_included": False,
        "input_content_sha256": next(iter(input_hashes)),
        "input_records": [
            {
                "content_sha256": row["content_sha256"],
                "host_alias": row["host_alias"],
            }
            for row in manifests
        ],
        "schema_version": "1.0.0",
        "shards": sorted(shard_records, key=lambda row: row["logical_shard"]),
        "status": "COMPLETED_MERGED_SELECTED_VISUAL_TOKEN_CACHE",
        "visual_count": sum(row["visual_count"] for row in shard_records),
    }
    manifest["content_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    _write_atomic(args.output_root / "visual_manifest.json", _canonical_json(manifest))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
