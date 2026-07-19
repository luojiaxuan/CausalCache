#!/usr/bin/env python3
"""Launch one independent token-cache extractor per visible GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuids", required=True)
    parser.add_argument("--partition-count", type=int, required=True)
    parser.add_argument("--visual-shard-size", type=int, default=16)
    parser.add_argument("--text-shard-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--maximum-text-tokens", type=int, default=128)
    args = parser.parse_args()

    uuids = tuple(value for value in args.expected_gpu_uuids.split(",") if value)
    if len(uuids) != args.partition_count:
        raise ValueError("GPU UUID count must equal the token partition count")
    visible = tuple(
        value for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value
    )
    if visible and len(visible) != args.partition_count:
        raise ValueError("visible GPU count must equal the token partition count")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    processes = []
    logs = []
    for partition_index, gpu_uuid in enumerate(uuids):
        log = (output_root / f"extract-part-{partition_index:02d}.log").open("ab")
        command = [
            sys.executable,
            "-m",
            "scripts.extract_set_utility_token_cache",
            "--repository-root",
            str(args.repository_root.resolve()),
            "--input-root",
            str(args.input_root.resolve()),
            "--model-dir",
            str(args.model_dir.resolve()),
            "--output-root",
            str(output_root),
            "--device",
            f"cuda:{partition_index}",
            "--expected-gpu-uuid",
            gpu_uuid,
            "--partition-index",
            str(partition_index),
            "--partition-count",
            str(args.partition_count),
            "--visual-shard-size",
            str(args.visual_shard_size),
            "--text-shard-size",
            str(args.text_shard_size),
            "--text-batch-size",
            str(args.text_batch_size),
            "--maximum-text-tokens",
            str(args.maximum_text_tokens),
        ]
        processes.append(
            subprocess.Popen(
                command,
                cwd=args.repository_root.resolve(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )
        logs.append(log)
    failures = []
    try:
        for index, process in enumerate(processes):
            return_code = process.wait()
            if return_code:
                failures.append((index, return_code))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()
    if failures:
        raise RuntimeError(f"token cache extractors failed: {failures!r}")


if __name__ == "__main__":
    main()
