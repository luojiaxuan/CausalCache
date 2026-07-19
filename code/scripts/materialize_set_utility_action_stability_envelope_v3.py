#!/usr/bin/env python3
"""Materialize the direct-child strict-determinism D2 execution envelope."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from causalcache.set_utility_action_stability_execution_v3 import (
    build_action_stability_envelope_v3,
    validate_clean_pushed_source_v3,
    write_canonical_envelope_pair_v3,
)


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("value must be strict JSON") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be one JSON object")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--parent-envelope-path", required=True)
    parser.add_argument("--processor-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--preflight-path", required=True)
    parser.add_argument("--preflight-completed-at-utc", required=True)
    parser.add_argument("--host-alias", choices=("hyper00", "hyper01"), required=True)
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-image-reference", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--driver-version", required=True)
    parser.add_argument("--gpu-json", action="append", type=_json_object, required=True)
    parser.add_argument("--software-versions-json", type=_json_object, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    validate_clean_pushed_source_v3(
        repository_root=root, expected_git_revision=args.source_git_revision
    )
    envelope = build_action_stability_envelope_v3(
        repository_root=root,
        source_git_revision=args.source_git_revision,
        run_root=args.run_root,
        python_executable=args.python_executable,
        parent_envelope_path=args.parent_envelope_path,
        processor_root=args.processor_root,
        model_dir=args.model_dir,
        preflight_path=args.preflight_path,
        preflight_completed_at_utc=args.preflight_completed_at_utc,
        host_alias=args.host_alias,
        hostname=args.hostname,
        container_id=args.container_id,
        container_image_reference=args.container_image_reference,
        container_image_digest=args.container_image_digest,
        driver_version=args.driver_version,
        selected_gpus=args.gpu_json,
        software_versions=args.software_versions_json,
    )
    receipt = write_canonical_envelope_pair_v3(envelope, repository_root=root)
    print(json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
