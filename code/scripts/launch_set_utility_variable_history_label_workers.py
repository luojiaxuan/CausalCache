#!/usr/bin/env python3
"""Launch multiple resumable label workers inside one multi-GPU container."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _worker_spec(value: str) -> tuple[int, int]:
    try:
        gpu_index, partition_index = (int(item) for item in value.split(":"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("worker must be LOCAL_GPU:PARTITION") from error
    if gpu_index < 0 or partition_index < 0:
        raise argparse.ArgumentTypeError("worker indices must be non-negative")
    return gpu_index, partition_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--scientific-config", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--partition-count", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--worker", action="append", type=_worker_spec, required=True)
    args = parser.parse_args()
    if args.partition_count <= 0:
        raise ValueError("partition count must be positive")
    partitions = [partition for _, partition in args.worker]
    if len(set(partitions)) != len(partitions):
        raise ValueError("worker partitions must be unique")
    if any(partition >= args.partition_count for partition in partitions):
        raise ValueError("worker partition is outside partition count")
    args.log_root.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[int, subprocess.Popen[bytes], object]] = []
    for gpu_index, partition_index in args.worker:
        command = [
            sys.executable,
            "code/scripts/run_set_utility_variable_history_labels.py",
            "--repository-root",
            str(args.repository_root),
            "--scientific-config",
            str(args.scientific_config),
            "--execution-config",
            str(args.execution_config),
            "--source-root",
            str(args.source_root),
            "--schedule-root",
            str(args.schedule_root),
            "--model-dir",
            str(args.model_dir),
            "--output-root",
            str(args.output_root),
            "--partition-index",
            str(partition_index),
            "--partition-count",
            str(args.partition_count),
            "--source-revision",
            args.source_revision,
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        environment["PYTHONPATH"] = str(args.repository_root / "code")
        log_path = args.log_root / f"worker-{partition_index:03d}.log"
        log_handle = log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=args.repository_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((partition_index, process, log_handle))

    failures = []
    try:
        for partition_index, process, _ in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append((partition_index, return_code))
    except BaseException:
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for _, _, log_handle in processes:
            log_handle.close()
    if failures:
        raise SystemExit(f"label workers failed: {failures}")


if __name__ == "__main__":
    main()
