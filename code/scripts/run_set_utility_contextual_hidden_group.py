#!/usr/bin/env python3
"""Launch one contextual-hidden worker per visible GPU and aggregate exits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _integer_list(value: str, *, label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} must contain integers") from error
    if not parsed or len(set(parsed)) != len(parsed) or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{label} must contain unique non-negative integers")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-indices", required=True)
    parser.add_argument("--partition-indices", required=True)
    parser.add_argument("--partition-count", type=int, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("worker_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cuda_indices = _integer_list(args.cuda_indices, label="cuda indices")
    partition_indices = _integer_list(
        args.partition_indices, label="partition indices"
    )
    if len(cuda_indices) != len(partition_indices):
        raise ValueError("CUDA and partition counts differ")
    if args.partition_count <= max(partition_indices):
        raise ValueError("partition count does not cover every worker")
    worker_argv = list(args.worker_argv)
    if worker_argv and worker_argv[0] == "--":
        worker_argv.pop(0)
    if not worker_argv:
        raise ValueError("worker command is empty")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    for cuda_index, partition_index in zip(
        cuda_indices, partition_indices, strict=True
    ):
        log_path = args.log_dir / f"worker-{partition_index:03d}.log"
        handle = log_path.open("ab", buffering=0)
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cuda_index)}
        command = [
            *worker_argv,
            "--device",
            "cuda:0",
            "--partition-index",
            str(partition_index),
            "--partition-count",
            str(args.partition_count),
        ]
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((partition_index, process))
        handles.append(handle)
    exits = {}
    for partition_index, process in processes:
        exits[str(partition_index)] = process.wait()
    for handle in handles:
        handle.close()
    result = {
        "exit_codes": exits,
        "partition_count": args.partition_count,
        "status": "COMPLETED" if all(code == 0 for code in exits.values()) else "FAILED",
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "COMPLETED":
        sys.exit(1)


if __name__ == "__main__":
    main()
