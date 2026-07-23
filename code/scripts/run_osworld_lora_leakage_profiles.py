#!/usr/bin/env python3
"""Launch independent OSWorld leakage profiles on explicit visible GPU indices."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_assignment(value: str) -> tuple[str, str, Path | None]:
    parts = value.split(":", maxsplit=2)
    if len(parts) not in (2, 3) or not parts[0].isdigit() or not parts[1]:
        raise ValueError(f"invalid GPU:PROFILE[:CHECKPOINT] assignment: {value}")
    checkpoint = Path(parts[2]) if len(parts) == 3 and parts[2] != "-" else None
    return parts[0], parts[1], checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--assignment",
        action="append",
        required=True,
        help="Container-visible GPU_INDEX:PROFILE_ID[:CHECKPOINT]",
    )
    args = parser.parse_args()
    assignments = [parse_assignment(value) for value in args.assignment]
    gpu_ids = [gpu_id for gpu_id, _, _ in assignments]
    profile_ids = [profile_id for _, profile_id, _ in assignments]
    if len(gpu_ids) != len(set(gpu_ids)) or len(profile_ids) != len(set(profile_ids)):
        raise ValueError("GPU and profile assignments must be unique")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    children = []
    logs = []
    for gpu_id, profile_id, checkpoint in assignments:
        log = (output_root / f"{profile_id}.log").open("a", encoding="utf-8")
        logs.append(log)
        command = [
            sys.executable,
            str(
                args.repository_root.resolve()
                / "code/scripts/run_osworld_lora_leakage_profile.py"
            ),
            "--repository-root",
            str(args.repository_root.resolve()),
            "--config",
            str(args.config.resolve()),
            "--manifest",
            str(args.manifest.resolve()),
            "--profile-id",
            profile_id,
            "--model-dir",
            str(args.model_dir.resolve()),
            "--device",
            "cuda:0",
            "--output-root",
            str(output_root),
        ]
        if checkpoint is not None:
            command.extend(["--lora-checkpoint", str(checkpoint.resolve())])
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id}
        children.append(
            (
                profile_id,
                subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ),
            )
        )
    failures = []
    for profile_id, child in children:
        return_code = child.wait()
        if return_code != 0:
            failures.append((profile_id, return_code))
    for log in logs:
        log.close()
    if failures:
        raise RuntimeError(f"OSWorld leakage profiles failed: {failures}")


if __name__ == "__main__":
    main()
