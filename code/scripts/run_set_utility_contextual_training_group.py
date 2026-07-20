#!/usr/bin/env python3
"""Launch contextual predictor variants on independent visible GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _job(value: str) -> tuple[int, str]:
    raw_index, separator, variant = value.partition(":")
    if not separator or not raw_index.isdigit() or not variant:
        raise argparse.ArgumentTypeError("job must be CUDA_INDEX:VARIANT")
    return int(raw_index), variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--job", type=_job, action="append", required=True)
    args = parser.parse_args()
    if len(args.job) < 2:
        raise ValueError("contextual training group requires at least two jobs")
    cuda_indices = [index for index, _ in args.job]
    variants = [variant for _, variant in args.job]
    if len(cuda_indices) != len(set(cuda_indices)) or len(variants) != len(set(variants)):
        raise ValueError("contextual training jobs must have unique GPUs and variants")
    if args.output_root.exists():
        raise FileExistsError("contextual training group output already exists")
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True)
    trainer = Path(__file__).with_name("train_set_utility_token_predictor.py")
    processes = []
    handles = []
    for cuda_index, variant in args.job:
        handle = (log_root / f"{variant}.log").open("ab", buffering=0)
        command = [
            sys.executable,
            str(trainer),
            "--input-root",
            str(args.input_root),
            "--cache-root",
            str(args.cache_root),
            "--config",
            str(args.config),
            "--variant",
            variant,
            "--output-root",
            str(args.output_root / variant),
            "--device",
            "cuda:0",
        ]
        process = subprocess.Popen(
            command,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(cuda_index)},
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((variant, process))
        handles.append(handle)
    exits = {variant: process.wait() for variant, process in processes}
    for handle in handles:
        handle.close()
    result = {
        "exit_codes": exits,
        "status": "COMPLETED" if all(code == 0 for code in exits.values()) else "FAILED",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "COMPLETED":
        sys.exit(1)


if __name__ == "__main__":
    main()
