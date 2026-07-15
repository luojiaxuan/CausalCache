"""Validate the complete GUIOdyssey restoration-v2 derived artifact offline."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from causalcache.data.guiodyssey_restoration_v2 import (
    build_ocr_backend_provenance,
    load_json_object,
    sha256_file,
    source_dataset_identity_from_v1_config,
    validate_artifact,
    validate_frozen_inputs,
    validate_ocr_runtime_identity,
)
from causalcache.restoration_v2_contract import RestorationV2Contract
from causalcache.restoration_v2_text_backend import load_backend_config
from scripts.validate_restoration_v2_ocr_backend import validate_artifact_source
from scripts.validate_restoration_v2_real_screen import runtime_identity_and_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v2-contract", type=Path, required=True)
    parser.add_argument("--v1-config", type=Path, required=True)
    parser.add_argument("--source-file-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--exposure-manifest", type=Path, required=True)
    parser.add_argument("--ocr-backend-config", type=Path, required=True)
    parser.add_argument("--ocr-backend-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--git-revision", required=True)
    return parser.parse_args()


def _verify_git_checkout(repository_root: Path, expected_revision: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
        raise ValueError("--git-revision must be a full 40-character Git SHA")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()
    if actual != expected_revision:
        raise ValueError("--git-revision does not match checkout HEAD")
    canonical = subprocess.check_output(
        ["git", "rev-parse", "origin/main"],
        cwd=repository_root,
        text=True,
    ).strip()
    if canonical != expected_revision:
        raise ValueError("artifact validation must use canonical origin/main")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        text=True,
    )
    if status:
        raise ValueError("formal artifact validation requires a clean checkout")


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    _verify_git_checkout(repository_root, args.git_revision)
    contract = RestorationV2Contract.load(args.v2_contract)
    _, v1_config = load_json_object(args.v1_config)
    load_json_object(args.source_file_manifest)
    _, selection_manifest = load_json_object(args.selection_manifest)
    _, exposure_manifest = load_json_object(args.exposure_manifest)
    backend_config = load_backend_config(args.ocr_backend_config)
    _, backend_manifest = load_json_object(args.ocr_backend_manifest)
    validate_artifact_source(
        backend_config_path=args.ocr_backend_config,
        artifact_manifest_path=args.ocr_backend_manifest,
        repository_root=repository_root,
    )
    validate_frozen_inputs(
        v2_contract=contract.data,
        selection_manifest=selection_manifest,
        selection_manifest_sha256=sha256_file(args.selection_manifest),
        v1_config_sha256=sha256_file(args.v1_config),
        source_file_manifest_sha256=sha256_file(args.source_file_manifest),
        exposure_manifest=exposure_manifest,
        backend_config=backend_config,
    )
    input_paths = {
        "scientific_contract": args.v2_contract,
        "v1_config": args.v1_config,
        "source_file_manifest": args.source_file_manifest,
        "selection_manifest": args.selection_manifest,
        "exposure_manifest": args.exposure_manifest,
        "ocr_backend_config": args.ocr_backend_config,
        "ocr_backend_manifest": args.ocr_backend_manifest,
    }
    ocr_runtime_identity, engine = runtime_identity_and_engine(
        backend_config=backend_config,
        model_dir=args.model_dir,
        wheel_dir=args.wheel_dir,
    )
    validate_ocr_runtime_identity(
        ocr_runtime_identity,
        backend_config=backend_config,
    )
    result = validate_artifact(
        output_dir=args.output_dir,
        backend_config=backend_config,
        backend_config_sha256=sha256_file(args.ocr_backend_config),
        selection_manifest=selection_manifest,
        expected_input_sha256={
            key: sha256_file(path) for key, path in input_paths.items()
        },
        expected_ocr_backend_provenance=build_ocr_backend_provenance(
            backend_config_sha256=sha256_file(args.ocr_backend_config),
            backend_manifest_sha256=sha256_file(args.ocr_backend_manifest),
            backend_manifest=backend_manifest,
        ),
        expected_ocr_runtime_identity=ocr_runtime_identity,
        expected_generator_git_revision=args.git_revision,
        expected_dataset_repo=str(
            contract.data["data"]["derived_artifact"]["repo"]
        ),
        expected_source_dataset=source_dataset_identity_from_v1_config(
            v1_config
        ),
        repository_root=repository_root,
        require_formal=True,
        ocr_engine=engine,
        require_ocr_replay=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
