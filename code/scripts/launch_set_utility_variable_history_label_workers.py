#!/usr/bin/env python3
"""Launch multiple resumable label workers inside one multi-GPU container."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _worker_spec(value: str) -> tuple[int, int, int, int]:
    try:
        parts = tuple(int(item) for item in value.split(":"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "worker must be LOCAL_GPU:PARTITION[:LANE_INDEX:LANE_COUNT]"
        ) from error
    if len(parts) == 2:
        gpu_index, partition_index = parts
        lane_index, lane_count = 0, 1
    elif len(parts) == 4:
        gpu_index, partition_index, lane_index, lane_count = parts
    else:
        raise argparse.ArgumentTypeError(
            "worker must be LOCAL_GPU:PARTITION[:LANE_INDEX:LANE_COUNT]"
        )
    if (
        gpu_index < 0
        or partition_index < 0
        or lane_count <= 0
        or not 0 <= lane_index < lane_count
    ):
        raise argparse.ArgumentTypeError("worker indices must be non-negative")
    return gpu_index, partition_index, lane_index, lane_count


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
    worker_keys = [
        (partition, lane_index, lane_count)
        for _, partition, lane_index, lane_count in args.worker
    ]
    if len(set(worker_keys)) != len(worker_keys):
        raise ValueError("worker partition lanes must be unique")
    partitions = [partition for _, partition, _, _ in args.worker]
    if any(partition >= args.partition_count for partition in partitions):
        raise ValueError("worker partition is outside partition count")
    args.log_root.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[str, subprocess.Popen[bytes], object]] = []
    for gpu_index, partition_index, lane_index, lane_count in args.worker:
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
            "--state-lane-index",
            str(lane_index),
            "--state-lane-count",
            str(lane_count),
            "--source-revision",
            args.source_revision,
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        environment["PYTHONPATH"] = str(args.repository_root / "code")
        worker_id = f"{partition_index:03d}"
        if lane_count > 1:
            worker_id += f"-lane-{lane_index:02d}-of-{lane_count:02d}"
        log_path = args.log_root / f"worker-{worker_id}.log"
        log_handle = log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=args.repository_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((worker_id, process, log_handle))

    failures = []
    try:
        for worker_id, process, _ in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append((worker_id, return_code))
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
