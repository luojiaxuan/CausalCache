#!/usr/bin/env python3
"""Deterministically rebalance scorer inputs without changing sample rows."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from scripts.run_hgkv_readout_extraction_shards import (
    atomic_write_json,
    sha256_file,
)
from scripts.run_success_action_scoring_shards import score_key


def assignment_for(key: tuple[Any, ...], *, output_count: int) -> int:
    packed = json.dumps(
        key, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(packed).digest()[:8], "big") % output_count


def rebalance(
    *,
    input_root: Path,
    output_root: Path,
    input_glob: str,
    output_count: int,
    images_target: Path,
    source_commit: str,
) -> dict[str, Any]:
    if output_count < 1:
        raise ValueError("output-count must be positive")
    input_paths = sorted(input_root.glob(input_glob))
    if not input_paths:
        raise FileNotFoundError(
            f"{input_root} contains no inputs matching {input_glob!r}"
        )
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    partial = output_root.with_name(output_root.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"partial output already exists: {partial}")

    partial.mkdir(parents=True)
    output_paths = [
        partial / f"samples-balanced-shard{index:02d}.jsonl"
        for index in range(output_count)
    ]
    handles = [path.open("w", encoding="utf-8") for path in output_paths]
    raw_rows = [0] * output_count
    keys_by_output: list[set[tuple[Any, ...]]] = [
        set() for _ in range(output_count)
    ]
    owner_by_key: dict[tuple[Any, ...], str] = {}
    source_rows: dict[str, int] = {}
    try:
        for path in input_paths:
            count = 0
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    key = score_key(row)
                    previous_owner = owner_by_key.get(key)
                    if previous_owner is not None and previous_owner != path.name:
                        raise ValueError(
                            "score identity crosses physical input files: "
                            f"{key} in {previous_owner} and {path.name}"
                        )
                    owner_by_key[key] = path.name
                    output_index = assignment_for(
                        key, output_count=output_count
                    )
                    handles[output_index].write(line)
                    raw_rows[output_index] += 1
                    keys_by_output[output_index].add(key)
                    count += 1
            source_rows[path.name] = count
    except Exception:
        for handle in handles:
            handle.close()
        shutil.rmtree(partial)
        raise
    else:
        for handle in handles:
            handle.close()

    # note (luojiaxuan): the hash input is the exact scorer resume identity, so
    # duplicate render rows for one restored coalition remain in one file and
    # one process. This preserves in-process dedup while balancing unique
    # forwards, which raw-line or round-robin splitting would not guarantee.
    unique_count = len(owner_by_key)
    output_unique_count = sum(len(keys) for keys in keys_by_output)
    if output_unique_count != unique_count:
        shutil.rmtree(partial)
        raise ValueError(
            "score identity crossed balanced outputs: "
            f"observed={output_unique_count} expected={unique_count}"
        )
    input_raw_count = sum(source_rows.values())
    if sum(raw_rows) != input_raw_count:
        shutil.rmtree(partial)
        raise ValueError("raw row conservation failed")

    os.symlink(str(images_target), partial / "images")
    outputs = {
        path.name: {
            "bytes": path.stat().st_size,
            "raw_rows": raw_rows[index],
            "sha256": sha256_file(path),
            "unique_score_identities": len(keys_by_output[index]),
        }
        for index, path in enumerate(output_paths)
    }
    manifest = {
        "assignment": (
            "uint64_be(sha256(canonical_score_identity)[:8]) % output_count"
        ),
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_files": {
            path.name: {
                "raw_rows": source_rows[path.name],
                "sha256": sha256_file(path),
            }
            for path in input_paths
        },
        "input_glob": input_glob,
        "input_raw_rows": input_raw_count,
        "input_root": str(input_root),
        "input_unique_score_identities": unique_count,
        "output_count": output_count,
        "output_files": outputs,
        "output_raw_rows": sum(raw_rows),
        "output_unique_score_identities": output_unique_count,
        "schema_version": (
            "causalcache.success_action_scoring_balanced_file_shards.v1"
        ),
        "score_identity": [
            "pair_group",
            "variant",
            "singleton_event_step_id",
            "restored_set_key",
        ],
        "source_commit": source_commit,
        "status": "DONE",
    }
    atomic_write_json(partial / "DONE", manifest)
    partial.replace(output_root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-glob", default="samples-shard*.jsonl")
    parser.add_argument("--output-count", type=int, required=True)
    parser.add_argument("--images-target", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    manifest = rebalance(
        input_root=args.input_root,
        output_root=args.output_root,
        input_glob=args.input_glob,
        output_count=args.output_count,
        images_target=args.images_target,
        source_commit=args.source_commit,
    )
    print(
        json.dumps(
            {
                "output_raw_rows": manifest["output_raw_rows"],
                "output_unique_score_identities": manifest[
                    "output_unique_score_identities"
                ],
                "status": manifest["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

