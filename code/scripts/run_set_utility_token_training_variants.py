#!/usr/bin/env python3
"""Train independent token-predictor variants on all selected GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", required=True)
    args = parser.parse_args()

    variants = tuple(value for value in args.variants.split(",") if value)
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("variants must be a non-empty unique comma-separated list")
    visible = tuple(
        value for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value
    )
    if visible and len(visible) < len(variants):
        raise ValueError("fewer visible GPUs than training variants")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    processes = []
    logs = []
    for device_index, variant in enumerate(variants):
        log = (output_root / f"{variant}.log").open("xb")
        command = [
            sys.executable,
            "-m",
            "scripts.train_set_utility_token_predictor",
            "--input-root",
            str(args.input_root.resolve()),
            "--cache-root",
            str(args.cache_root.resolve()),
            "--config",
            str(args.config.resolve()),
            "--variant",
            variant,
            "--output-root",
            str(output_root / variant),
            "--device",
            f"cuda:{device_index}",
        ]
        processes.append(
            subprocess.Popen(
                command,
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
                failures.append((variants[index], return_code))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()
    if failures:
        raise RuntimeError(f"token predictor variants failed: {failures!r}")


if __name__ == "__main__":
    main()
