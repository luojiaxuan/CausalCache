#!/usr/bin/env python3
"""Launch and verify independent HGKV-readout feature-extraction shards."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gpu_indices(raw: str) -> list[int]:
    try:
        indices = [int(item) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU indices must be comma-separated integers"
        ) from exc
    if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError(
            "GPU indices must be unique non-negative integers"
        )
    return indices


def parse_shard_indices(raw: str) -> list[int]:
    try:
        indices = [int(item) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shard indices must be comma-separated integers"
        ) from exc
    if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError(
            "shard indices must be unique non-negative integers"
        )
    return indices


def resolve_shard_indices(
    shard_count: int, requested: list[int] | None
) -> list[int]:
    indices = list(range(shard_count)) if requested is None else requested
    if shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if any(index >= shard_count for index in indices):
        raise ValueError(
            f"shard indices {indices} exceed shard-count {shard_count}"
        )
    return indices


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_feature_outputs(
    output_paths: list[Path], *, expected_rows: int
) -> dict[str, Any]:
    seen: set[tuple[str, int]] = set()
    rows_per_shard: dict[str, int] = {}
    feature_dims: set[int] = set()
    for path in output_paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing feature shard: {path}")
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number} is not valid JSON"
                    ) from exc
                key = (
                    str(row["pair_group"]),
                    int(row["singleton_event_step_id"]),
                )
                if key in seen:
                    raise ValueError(f"duplicate feature key: {key}")
                seen.add(key)
                feature_dim = int(row["feature_dim"])
                if len(row["feature"]) != feature_dim:
                    raise ValueError(
                        f"{path}:{line_number} feature length mismatch"
                    )
                if not all(
                    math.isfinite(float(value)) for value in row["feature"]
                ):
                    raise ValueError(
                        f"{path}:{line_number} contains non-finite features"
                    )
                feature_dims.add(feature_dim)
                count += 1
        rows_per_shard[path.name] = count
    if len(seen) != expected_rows:
        raise ValueError(
            f"feature row count {len(seen)} != expected {expected_rows}"
        )
    if len(feature_dims) != 1:
        raise ValueError(f"inconsistent feature dimensions: {feature_dims}")
    return {
        "row_count": len(seen),
        "unique_key_count": len(seen),
        "feature_dim": next(iter(feature_dims)),
        "rows_per_shard": rows_per_shard,
    }


def build_child_command(
    args: argparse.Namespace, *, shard_index: int
) -> list[str]:
    extractor = (
        args.repository_root
        / "code"
        / "scripts"
        / "extract_hgkv_readout_features.py"
    )
    return [
        sys.executable,
        str(extractor),
        "--repository-root",
        str(args.repository_root),
        "--model-dir",
        str(args.model_dir),
        "--dataset-root",
        str(args.dataset_root),
        "--output",
        str(args.output_root / f"features-shard{shard_index}.jsonl"),
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
    ]


def run(args: argparse.Namespace) -> int:
    shard_indices = resolve_shard_indices(
        args.shard_count, args.shard_indices
    )
    if len(args.gpu_local_indices) != len(shard_indices):
        raise ValueError(
            "gpu-local-indices count must equal selected shard count "
            "for one process per shard"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset_root / "samples.jsonl"
    dataset_sha256 = sha256_file(dataset_path)
    checkpoint_sha256 = sha256_file(args.lora_checkpoint)
    if dataset_sha256 != args.expected_dataset_sha256:
        raise ValueError(
            f"dataset SHA256 {dataset_sha256} != "
            f"{args.expected_dataset_sha256}"
        )
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint SHA256 {checkpoint_sha256} != "
            f"{args.expected_checkpoint_sha256}"
        )

    output_paths = [
        args.output_root / f"features-shard{index}.jsonl"
        for index in range(args.shard_count)
    ]
    if (args.output_root / "DONE").is_file():
        validate_feature_outputs(output_paths, expected_rows=args.expected_rows)
        return 0

    commands = [
        build_child_command(args, shard_index=index)
        for index in shard_indices
    ]
    atomic_write_json(
        args.output_root / "launch-manifest.json",
        {
            "adapter_type": "history_gated_kv",
            "checkpoint_sha256": checkpoint_sha256,
            "commands": commands,
            "dataset_sha256": dataset_sha256,
            "expected_rows": args.expected_rows,
            "gpu_local_indices": args.gpu_local_indices,
            "launched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "shard_count": args.shard_count,
            "shard_indices": shard_indices,
            "source_commit": args.source_commit,
        },
    )
    (args.output_root / "FAILED.json").unlink(missing_ok=True)

    processes: list[tuple[subprocess.Popen[bytes], Any]] = []
    python_path = str(args.repository_root / "code")
    for shard_index, gpu_index, command in zip(
        shard_indices, args.gpu_local_indices, commands, strict=True
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

    return_codes: dict[str, int] = {}
    for shard_index, (process, log_handle) in zip(
        shard_indices, processes, strict=True
    ):
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

    validation = validate_feature_outputs(
        output_paths, expected_rows=args.expected_rows
    )
    atomic_write_json(
        args.output_root / "DONE",
        {
            "checkpoint_sha256": checkpoint_sha256,
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dataset_sha256": dataset_sha256,
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
        "--gpu-local-indices", type=parse_gpu_indices, required=True
    )
    parser.add_argument(
        "--shard-indices",
        type=parse_shard_indices,
        default=None,
        help="comma-separated logical shards to resume; defaults to all shards",
    )
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
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
