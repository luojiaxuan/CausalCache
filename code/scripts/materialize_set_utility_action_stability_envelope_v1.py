#!/usr/bin/env python3
"""Materialize one fresh D1 four-H200 execution envelope."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from causalcache.set_utility_action_stability_envelope_v1 import (
    build_action_stability_envelope_v1,
    build_action_stability_preflight_evidence_v1,
    canonical_run_layout,
    validate_clean_pushed_source,
    write_canonical_envelope_pair_exclusive,
    write_preflight_evidence_exclusive,
)


def _gpu(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("--gpu-json must be strict JSON") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--gpu-json must contain one object")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--preflight-started-at-utc", required=True)
    parser.add_argument("--preflight-completed-at-utc", required=True)
    parser.add_argument("--raw-cleanup-log", type=Path, required=True)
    parser.add_argument("--host-alias", choices=("hyper00", "hyper01"), required=True)
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-image-reference", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--driver-version", required=True)
    parser.add_argument("--gpu-json", action="append", type=_gpu, required=True)
    parser.add_argument("--parent-execution-envelope", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--processor-root", type=Path, required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--transformers-version", required=True)
    parser.add_argument("--cuda-runtime-version", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    layout = canonical_run_layout(args.run_root)
    output = args.output or Path(layout["envelope_path"])
    if str(output) != layout["envelope_path"] or os.path.lexists(output):
        raise FileExistsError("D1 output must be the fresh canonical run envelope")
    validate_clean_pushed_source(
        repository_root=repository_root,
        expected_git_revision=args.source_git_revision,
    )
    evidence = build_action_stability_preflight_evidence_v1(
        raw_cleanup_log_path=args.raw_cleanup_log,
        started_at_utc=args.preflight_started_at_utc,
        completed_at_utc=args.preflight_completed_at_utc,
        host_alias=args.host_alias,
        hostname=args.hostname,
        container_id=args.container_id,
        driver_version=args.driver_version,
        selected_gpus=args.gpu_json,
    )
    receipt = write_preflight_evidence_exclusive(
        run_root=args.run_root,
        evidence=evidence,
    )
    evidence_path = Path(receipt["path"])
    software = {
        "cuda_runtime_version": args.cuda_runtime_version,
        "python_version": args.python_version,
        "torch_version": args.torch_version,
        "transformers_version": args.transformers_version,
    }
    try:
        envelope = build_action_stability_envelope_v1(
            repository_root=repository_root,
            source_git_revision=args.source_git_revision,
            run_root=args.run_root,
            python_executable=args.python_executable,
            preflight_evidence_path=evidence_path,
            parent_envelope_path=args.parent_execution_envelope,
            processor_root=args.processor_root,
            model_dir=args.model_dir,
            host_alias=args.host_alias,
            hostname=args.hostname,
            container_id=args.container_id,
            container_image_reference=args.container_image_reference,
            container_image_digest=args.container_image_digest,
            driver_version=args.driver_version,
            software_versions=software,
        )
        result = write_canonical_envelope_pair_exclusive(
            repository_root=repository_root,
            data_path=output,
            envelope=envelope,
        )
    except Exception:
        if evidence_path.is_file() and not evidence_path.is_symlink():
            evidence_path.unlink()
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
