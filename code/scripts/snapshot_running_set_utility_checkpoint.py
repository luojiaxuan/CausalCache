#!/usr/bin/env python3
"""Create an immutable selector-compatible snapshot of a running best checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from causalcache.set_utility_mvp import canonical_json_bytes


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_running_checkpoint(
    *,
    input_root: Path,
    cache_root: Path,
    config_path: Path,
    variant: str,
    training_root: Path,
    output_root: Path,
    best_epoch: int,
    best_tune_total: float,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"snapshot output already exists: {output_root}")
    if best_epoch <= 0 or not math.isfinite(best_tune_total):
        raise ValueError("best epoch and tune total must be valid")
    input_manifest = _read_json(input_root / "manifest.json")
    cache_manifest = _read_json(cache_root / "manifest.json")
    config = _read_json(config_path)
    if variant not in config["variants"]:
        raise ValueError("snapshot variant is absent from config")
    if (
        input_manifest.get("evaluation_labels_included") is not False
        or cache_manifest.get("evaluation_labels_included") is not False
        or config.get("input", {}).get("training_input_content_sha256")
        != input_manifest.get("content_sha256")
        or config.get("input", {}).get("contextual_cache_content_sha256")
        != cache_manifest.get("content_sha256")
    ):
        raise ValueError("snapshot input/cache/config firewall drifted")
    source = training_root / "best.safetensors"
    if not source.is_file():
        raise FileNotFoundError("running training root has no best checkpoint")
    staging = output_root.with_name(f".{output_root.name}.{os.getpid()}.tmp")
    staging.mkdir(parents=True)
    try:
        source_before = _sha256(source)
        destination = staging / "best.safetensors"
        shutil.copyfile(source, destination)
        with destination.open("rb") as handle:
            os.fsync(handle.fileno())
        source_after = _sha256(source)
        destination_sha256 = _sha256(destination)
        if not source_before == source_after == destination_sha256:
            raise RuntimeError("best checkpoint changed while snapshotting")
        summary = {
            "best_checkpoint": {
                "byte_count": destination.stat().st_size,
                "path": destination.name,
                "sha256": destination_sha256,
            },
            "best_epoch": best_epoch,
            "best_tune_total": best_tune_total,
            "cache_content_sha256": cache_manifest["content_sha256"],
            "config_sha256": _sha256(config_path),
            "evaluation_records_loaded": False,
            "input_content_sha256": input_manifest["content_sha256"],
            "schema_version": "1.0.0",
            "seed": int(config["training"]["seed"]),
            "source_training_root": str(training_root),
            "status": "SNAPSHOT_RUNNING_SET_UTILITY_CHECKPOINT",
            "variant": variant,
        }
        summary["content_sha256"] = hashlib.sha256(
            canonical_json_bytes(summary)
        ).hexdigest()
        (staging / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
        os.replace(staging, output_root)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--best-epoch", type=int, required=True)
    parser.add_argument("--best-tune-total", type=float, required=True)
    args = parser.parse_args()
    result = snapshot_running_checkpoint(
        input_root=args.input_root.resolve(),
        cache_root=args.cache_root.resolve(),
        config_path=args.config.resolve(),
        variant=args.variant,
        training_root=args.training_root.resolve(),
        output_root=args.output_root.resolve(),
        best_epoch=args.best_epoch,
        best_tune_total=args.best_tune_total,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
