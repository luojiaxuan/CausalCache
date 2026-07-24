#!/usr/bin/env python3
"""Launch and validate sharded full-history V2 HGKV readout extraction."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.extract_hgkv_readout_v2 import _sample_paths
from scripts.run_hgkv_readout_extraction_shards import (
    atomic_write_json,
    parse_gpu_indices,
    parse_shard_indices,
    resolve_shard_indices,
    sha256_file,
)
from scripts.validate_hgkv_readout_v2 import (
    load_expected,
    validate_features,
)


def input_manifest(
    paths: list[Path],
) -> tuple[dict[str, str], set[tuple[str, int]]]:
    hashes: dict[str, str] = {}
    keys: set[tuple[str, int]] = set()
    for path in paths:
        hashes[str(path)] = sha256_file(path)
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("variant") != "singleton":
                raise ValueError(f"{path}:{line_no} is not a singleton row")
            key = (
                str(row["pair_group"]),
                int(row["singleton_event_step_id"]),
            )
            if key in keys:
                raise ValueError(f"{path}:{line_no} duplicate input key {key}")
            keys.add(key)
    return hashes, keys


def build_child_command(
    args: argparse.Namespace, *, shard_index: int
) -> list[str]:
    return [
        sys.executable,
        str(
            args.repository_root
            / "code/scripts/extract_hgkv_readout_v2.py"
        ),
        "--repository-root",
        str(args.repository_root),
        "--model-dir",
        str(args.model_dir),
        "--dataset-root",
        str(args.dataset_root),
        "--samples-glob",
        args.samples_glob,
        "--output",
        str(args.output_root / f"features-fresh-shard{shard_index}.jsonl"),
        "--device",
        "cuda:0",
        "--lora-checkpoint",
        str(args.lora_checkpoint),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--adapter-layer-count",
        str(args.adapter_layer_count),
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(args.shard_count),
        "--shard-by-file",
    ]


def run(args: argparse.Namespace) -> int:
    shard_indices = resolve_shard_indices(
        args.shard_count, args.shard_indices
    )
    if len(args.gpu_local_indices) != len(shard_indices):
        raise ValueError(
            "gpu-local-indices count must equal selected shard count"
        )
    sample_paths = _sample_paths(args.dataset_root, args.samples_glob)
    if len(sample_paths) < args.shard_count:
        raise ValueError("file sharding requires at least one file per shard")
    hashes, expected_input_keys = input_manifest(sample_paths)
    expected = load_expected(args.state_paths)
    if not expected_input_keys.issubset(expected):
        raise ValueError("fresh readout input contains non-inventory keys")
    if len(expected_input_keys) != args.expected_rows:
        raise ValueError(
            f"input rows {len(expected_input_keys)} != {args.expected_rows}"
        )
    checkpoint_sha256 = sha256_file(args.lora_checkpoint)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError("HGKV checkpoint SHA256 drifted")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_paths = [
        args.output_root / f"features-fresh-shard{index}.jsonl"
        for index in range(args.shard_count)
    ]
    if (args.output_root / "DONE").is_file():
        validate_features(
            {key: expected[key] for key in expected_input_keys},
            output_paths,
        )
        return 0
    commands = [
        build_child_command(args, shard_index=index)
        for index in shard_indices
    ]
    aggregate = hashlib.sha256()
    for path in sorted(hashes):
        aggregate.update(f"{hashes[path]}  {path}\n".encode("utf-8"))
    atomic_write_json(
        args.output_root / "launch-manifest.json",
        {
            "checkpoint_sha256": checkpoint_sha256,
            "commands": commands,
            "expected_rows": args.expected_rows,
            "gpu_local_indices": args.gpu_local_indices,
            "input_manifest_sha256": aggregate.hexdigest(),
            "launched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "shard_count": args.shard_count,
            "shard_indices": shard_indices,
            "source_commit": args.source_commit,
        },
    )
    (args.output_root / "FAILED.json").unlink(missing_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes], Any]] = []
    python_path = str(args.repository_root / "code")
    for shard_index, gpu_index, command in zip(
        shard_indices, args.gpu_local_indices, commands, strict=True
    ):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        environment["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (python_path, environment.get("PYTHONPATH"))
            if value
        )
        log_handle = (
            args.output_root / f"shard{shard_index}.log"
        ).open("ab")
        processes.append(
            (
                shard_index,
                subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                ),
                log_handle,
            )
        )
    return_codes: dict[str, int] = {}
    for shard_index, process, log_handle in processes:
        return_codes[str(shard_index)] = process.wait()
        log_handle.close()
    if any(code != 0 for code in return_codes.values()):
        atomic_write_json(
            args.output_root / "FAILED.json",
            {
                "return_codes": return_codes,
                "status": "FAILED",
                "time": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        return 1
    observed = validate_features(
        {key: expected[key] for key in expected_input_keys},
        output_paths,
    )
    atomic_write_json(
        args.output_root / "DONE",
        {
            **observed,
            "checkpoint_sha256": checkpoint_sha256,
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input_manifest_sha256": aggregate.hexdigest(),
            "source_commit": args.source_commit,
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--samples-glob", default="*/samples-shard*.jsonl")
    parser.add_argument("--state-paths", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--adapter-layer-count", type=int, default=8)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument(
        "--gpu-local-indices", type=parse_gpu_indices, required=True
    )
    parser.add_argument(
        "--shard-indices", type=parse_shard_indices, default=None
    )
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        args.output_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            args.output_root / "FAILED.json",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "status": "FAILED",
                "time": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
