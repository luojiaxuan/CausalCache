#!/usr/bin/env python3
"""Launch frozen tune selectors on independent visible GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _job(value: str) -> tuple[int, str, Path]:
    pieces = value.split(":", 2)
    if len(pieces) != 3 or not pieces[0].isdigit() or not pieces[1] or not pieces[2]:
        raise argparse.ArgumentTypeError(
            "job must be CUDA_INDEX:VARIANT:TRAINING_ROOT"
        )
    return int(pieces[0]), pieces[1], Path(pieces[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "tune"), default="tune")
    parser.add_argument("--conditional-candidates-per-step", type=int, default=0)
    parser.add_argument("--job", type=_job, action="append", required=True)
    args = parser.parse_args()
    if len(args.job) < 2:
        raise ValueError("tune selector group requires at least two jobs")
    cuda_indices = [index for index, _, _ in args.job]
    variants = [variant for _, variant, _ in args.job]
    if len(cuda_indices) != len(set(cuda_indices)) or len(variants) != len(set(variants)):
        raise ValueError("tune selector jobs must have unique GPUs and variants")
    if args.output_root.exists():
        raise FileExistsError("tune selector group output already exists")
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True)
    selector = Path(__file__).with_name("run_set_utility_tune_selectors.py")
    processes = []
    handles = []
    for cuda_index, variant, training_root in args.job:
        summary = training_root / "summary.json"
        checkpoint = training_root / "best.safetensors"
        if not summary.is_file() or not checkpoint.is_file():
            raise FileNotFoundError("tune selector training artifact is incomplete")
        handle = (log_root / f"{variant}.log").open("ab", buffering=0)
        command = [
            sys.executable,
            str(selector),
            "--input-root",
            str(args.input_root),
            "--cache-root",
            str(args.cache_root),
            "--config",
            str(args.config),
            "--variant",
            variant,
            "--training-summary",
            str(summary),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(args.output_root / f"{variant}.json"),
            "--device",
            "cuda:0",
            "--role",
            args.role,
            "--conditional-candidates-per-step",
            str(args.conditional_candidates_per_step),
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
