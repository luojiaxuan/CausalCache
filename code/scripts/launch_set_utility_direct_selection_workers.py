#!/usr/bin/env python3
"""Launch direct train-selection shards on independent visible GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _job(value: str) -> tuple[int, int, Path]:
    pieces = value.split(":", 2)
    if (
        len(pieces) != 3
        or not pieces[0].isdigit()
        or not pieces[1].isdigit()
        or not pieces[2]
    ):
        raise argparse.ArgumentTypeError("job must be CUDA_INDEX:WORKER_ID:STATE_IDS")
    return int(pieces[0]), int(pieces[1]), Path(pieces[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--collection-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--job", type=_job, action="append", required=True)
    args = parser.parse_args()
    gpu_ids = [gpu for gpu, _, _ in args.job]
    worker_ids = [worker for _, worker, _ in args.job]
    if len(gpu_ids) != len(set(gpu_ids)) or len(worker_ids) != len(set(worker_ids)):
        raise ValueError("direct selection jobs must have unique GPUs and workers")
    if args.output_root.exists():
        raise FileExistsError("direct selection output already exists")
    for _, _, state_ids in args.job:
        if not state_ids.is_file():
            raise FileNotFoundError(state_ids)
        requested = tuple(
            line
            for line in state_ids.read_text(encoding="utf-8").splitlines()
            if line
        )
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("worker state ids must be non-empty and unique")
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True)
    selector = Path(__file__).with_name(
        "run_set_utility_direct_marginal_tune_selectors.py"
    )
    processes = []
    handles = []
    for gpu_id, worker_id, state_ids in args.job:
        handle = (log_root / f"worker-{worker_id:02d}.log").open("ab", buffering=0)
        command = [
            sys.executable,
            str(selector),
            "--input-root",
            str(args.input_root),
            "--cache-root",
            str(args.cache_root),
            "--config",
            str(args.training_config),
            "--variant",
            args.variant,
            "--training-summary",
            str(args.training_summary),
            "--checkpoint",
            str(args.checkpoint),
            "--collection-config",
            str(args.collection_config),
            "--role",
            "train",
            "--state-id-file",
            str(state_ids),
            "--output",
            str(args.output_root / f"worker-{worker_id:02d}.json"),
            "--device",
            "cuda:0",
        ]
        process = subprocess.Popen(
            command,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((worker_id, process))
        handles.append(handle)
    exits = {worker_id: process.wait() for worker_id, process in processes}
    for handle in handles:
        handle.close()
    result = {
        "exit_codes": exits,
        "status": "COMPLETED" if all(code == 0 for code in exits.values()) else "FAILED",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
