#!/usr/bin/env python3
"""Materialize one fresh exact-84-call GPU execution envelope."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from causalcache.set_utility_throughput_pilot_envelope_v1 import (
    HOSTNAME,
    HOST_ALIAS,
    build_set_utility_throughput_pilot_preflight_evidence_v1,
    build_set_utility_throughput_pilot_envelope_v1,
    canonical_preflight_evidence_path,
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
        raise argparse.ArgumentTypeError("--gpu-json must contain one JSON object")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--preflight-started-at-utc", required=True)
    parser.add_argument("--preflight-completed-at-utc", required=True)
    parser.add_argument("--raw-cleanup-log", type=Path, required=True)
    parser.add_argument("--host-alias", default=HOST_ALIAS)
    parser.add_argument("--hostname", default=HOSTNAME)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-image-reference", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--driver-version", required=True)
    parser.add_argument("--gpu-json", action="append", type=_gpu, required=True)
    parser.add_argument("--model-local-path", type=Path, required=True)
    parser.add_argument("--processor-local-root", type=Path, required=True)
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
    if str(output) != layout["envelope_path"]:
        raise ValueError("--output must equal the canonical run-root envelope path")
    if os.path.lexists(output):
        raise FileExistsError(f"execution envelope already exists: {output}")
    validate_clean_pushed_source(
        repository_root=repository_root,
        expected_git_revision=args.source_git_revision,
    )
    software_versions = {
        "cuda_runtime_version": args.cuda_runtime_version,
        "python_version": args.python_version,
        "torch_version": args.torch_version,
        "transformers_version": args.transformers_version,
    }
    evidence = build_set_utility_throughput_pilot_preflight_evidence_v1(
        raw_cleanup_log_path=args.raw_cleanup_log,
        preflight_started_at_utc=args.preflight_started_at_utc,
        preflight_completed_at_utc=args.preflight_completed_at_utc,
        host_alias=args.host_alias,
        hostname=args.hostname,
        container_id=args.container_id,
        container_image_reference=args.container_image_reference,
        container_image_digest=args.container_image_digest,
        driver_version=args.driver_version,
        selected_gpus=args.gpu_json,
        python_executable=args.python_executable,
        software_versions=software_versions,
    )
    identity_keys = (
        "host_index",
        "name",
        "total_memory_bytes",
        "uuid",
        "visible_index",
    )
    identity_gpus = [
        {key: gpu[key] for key in identity_keys} for gpu in args.gpu_json
    ]
    evidence_receipt = write_preflight_evidence_exclusive(
        run_root=args.run_root,
        evidence=evidence,
    )
    evidence_path = canonical_preflight_evidence_path(args.run_root)
    try:
        envelope = build_set_utility_throughput_pilot_envelope_v1(
            repository_root=repository_root,
            source_git_revision=args.source_git_revision,
            run_root=args.run_root,
            python_executable=args.python_executable,
            preflight_started_at_utc=args.preflight_started_at_utc,
            preflight_completed_at_utc=args.preflight_completed_at_utc,
            preflight_evidence_path=evidence_path,
            host_alias=args.host_alias,
            hostname=args.hostname,
            container_id=args.container_id,
            container_image_reference=args.container_image_reference,
            container_image_digest=args.container_image_digest,
            driver_version=args.driver_version,
            gpus=identity_gpus,
            model_local_path=args.model_local_path,
            processor_local_root=args.processor_local_root,
            software_versions=software_versions,
            inspect_local_artifacts=True,
            verify_current_environment=True,
        )
    except BaseException:
        if (
            evidence_path.is_file()
            and not evidence_path.is_symlink()
            and evidence_path.stat().st_size == evidence_receipt["size_bytes"]
        ):
            evidence_path.unlink()
        raise
    receipt = write_canonical_envelope_pair_exclusive(
        repository_root=repository_root,
        data_path=output,
        envelope=envelope,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
