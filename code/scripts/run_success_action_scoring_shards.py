#!/usr/bin/env python3
"""Launch and verify sharded teacher-forced success-action scoring."""

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

from scripts.run_hgkv_readout_extraction_shards import (
    atomic_write_json,
    sha256_file,
)
from scripts.score_success_action_recovery import resolve_sample_paths


def score_key(row: dict[str, Any]) -> tuple[str, str, int | None, str | None]:
    return (
        str(row["pair_group"]),
        str(row.get("variant", "correct")),
        row.get("singleton_event_step_id"),
        row.get("restored_set_key"),
    )


def parse_gpu_slots(raw: str) -> list[int]:
    try:
        slots = [int(item) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU slots must be comma-separated integers"
        ) from exc
    if not slots or min(slots) < 0:
        raise argparse.ArgumentTypeError(
            "GPU slots must be non-negative integers"
        )
    return slots


def input_manifest(paths: list[Path]) -> tuple[dict[str, str], set[tuple]]:
    hashes = {}
    keys: set[tuple] = set()
    owner_by_key: dict[tuple, str] = {}
    for path in paths:
        hashes[path.name] = sha256_file(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                key = score_key(json.loads(line))
                previous_owner = owner_by_key.get(key)
                if previous_owner is not None and previous_owner != path.name:
                    raise ValueError(
                        "score identity crosses physical input files: "
                        f"{key} in {previous_owner} and {path.name}"
                    )
                owner_by_key[key] = path.name
                keys.add(key)
    return hashes, keys


def validate_score_outputs(
    paths: list[Path], *, expected_keys: set[tuple]
) -> dict[str, Any]:
    keys: set[tuple] = set()
    rows_per_shard = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing score shard: {path}")
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number} is not valid JSON"
                    ) from exc
                key = score_key(row)
                if key in keys:
                    raise ValueError(f"duplicate output score key: {key}")
                value = float(row["target_logprob_mean"])
                if not (-float("inf") < value < float("inf")):
                    raise ValueError(f"non-finite score for {key}")
                keys.add(key)
                count += 1
        rows_per_shard[path.name] = count
    missing = expected_keys - keys
    unexpected = keys - expected_keys
    if missing or unexpected:
        raise ValueError(
            f"score key mismatch: missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
    return {
        "row_count": len(keys),
        "rows_per_shard": rows_per_shard,
        "unique_key_count": len(keys),
    }


def build_child_command(
    args: argparse.Namespace, *, shard_index: int
) -> list[str]:
    scorer = (
        args.repository_root
        / "code"
        / "scripts"
        / "score_success_action_recovery.py"
    )
    command = [
        sys.executable,
        str(scorer),
        "--repository-root",
        str(args.repository_root),
        "--model-dir",
        str(args.model_dir),
        "--dataset-root",
        str(args.dataset_root),
        "--output",
        str(args.output_root / f"scores-shard{shard_index}.jsonl"),
        "--device",
        "cuda:0",
        "--lora-checkpoint",
        str(args.lora_checkpoint),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--adapter-type",
        "history_gated_kv",
        "--adapter-layer-count",
        str(args.adapter_layer_count),
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(args.shard_count),
        "--samples-glob",
        args.samples_glob,
    ]
    if args.shard_by_file:
        command.append("--shard-by-file")
    return command


def run(args: argparse.Namespace) -> int:
    if len(args.gpu_local_indices) != args.shard_count:
        raise ValueError(
            "gpu-local-indices count must equal shard-count for one process per shard"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    sample_paths = resolve_sample_paths(args.dataset_root, args.samples_glob)
    if args.shard_by_file and len(sample_paths) < args.shard_count:
        raise ValueError("file sharding requires at least one input file per shard")
    checkpoint_sha256 = sha256_file(args.lora_checkpoint)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint SHA256 {checkpoint_sha256} != "
            f"{args.expected_checkpoint_sha256}"
        )
    hashes, expected_keys = input_manifest(sample_paths)
    output_paths = [
        args.output_root / f"scores-shard{index}.jsonl"
        for index in range(args.shard_count)
    ]
    if (args.output_root / "DONE").is_file():
        validate_score_outputs(output_paths, expected_keys=expected_keys)
        return 0

    commands = [
        build_child_command(args, shard_index=index)
        for index in range(args.shard_count)
    ]
    aggregate = hashlib.sha256()
    for name in sorted(hashes):
        aggregate.update(f"{hashes[name]}  {name}\n".encode())
    atomic_write_json(
        args.output_root / "launch-manifest.json",
        {
            "adapter_type": "history_gated_kv",
            "checkpoint_sha256": checkpoint_sha256,
            "commands": commands,
            "expected_unique_keys": len(expected_keys),
            "gpu_local_indices": args.gpu_local_indices,
            "input_files": hashes,
            "input_manifest_sha256": aggregate.hexdigest(),
            "launched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "samples_glob": args.samples_glob,
            "shard_by_file": args.shard_by_file,
            "shard_count": args.shard_count,
            "source_commit": args.source_commit,
        },
    )
    (args.output_root / "FAILED.json").unlink(missing_ok=True)

    processes: list[tuple[subprocess.Popen[bytes], Any]] = []
    python_path = str(args.repository_root / "code")
    for shard_index, (gpu_index, command) in enumerate(
        zip(args.gpu_local_indices, commands, strict=True)
    ):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        environment["PYTHONPATH"] = os.pathsep.join(
            item
            for item in (python_path, environment.get("PYTHONPATH"))
            if item
        )
        log_handle = (
            args.output_root / f"shard{shard_index}.log"
        ).open("ab")
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log_handle))

    return_codes = []
    for process, log_handle in processes:
        return_codes.append(process.wait())
        log_handle.close()
    if any(code != 0 for code in return_codes):
        atomic_write_json(
            args.output_root / "FAILED.json",
            {
                "return_codes": return_codes,
                "status": "FAILED",
                "time": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        return 1

    validation = validate_score_outputs(
        output_paths, expected_keys=expected_keys
    )
    atomic_write_json(
        args.output_root / "DONE",
        {
            "checkpoint_sha256": checkpoint_sha256,
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input_manifest_sha256": aggregate.hexdigest(),
            "source_commit": args.source_commit,
            "status": "DONE",
            **validation,
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--adapter-layer-count", type=int, default=8)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument(
        "--gpu-local-indices", type=parse_gpu_slots, required=True
    )
    parser.add_argument("--samples-glob", required=True)
    parser.add_argument("--shard-by-file", action="store_true")
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
