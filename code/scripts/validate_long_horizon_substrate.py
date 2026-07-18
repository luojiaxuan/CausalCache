"""Validate and optionally OCR-replay the long-horizon development substrate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from causalcache.long_horizon_data import (
    DATASET_REPO,
    build_ocr_backend_provenance,
    canonical_json_bytes,
    load_development_source_pilots,
    load_json_object,
    sha256_file,
    validate_artifact,
)
from causalcache.restoration_v2_text_backend import load_backend_config
from scripts.build_long_horizon_substrate import verify_pushed_checkout
from scripts.validate_restoration_v2_ocr_backend import validate_artifact_source
from scripts.validate_restoration_v2_real_screen import runtime_identity_and_engine


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--v1-config", type=Path, required=True)
    parser.add_argument("--source-file-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--ocr-backend-config", type=Path, required=True)
    parser.add_argument("--ocr-backend-manifest", type=Path, required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    parser.add_argument("--require-ocr-replay", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--wheel-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    verify_pushed_checkout(repository_root, args.git_revision)
    if args.require_ocr_replay and (args.model_dir is None or args.wheel_dir is None):
        raise ValueError("OCR replay requires --model-dir and --wheel-dir")
    if not args.require_ocr_replay and (
        args.model_dir is not None or args.wheel_dir is not None
    ):
        raise ValueError("model/wheel paths require --require-ocr-replay")
    _, selection = load_json_object(args.selection_manifest)
    _, v1_config = load_json_object(args.v1_config)
    _, source_file_manifest = load_json_object(args.source_file_manifest)
    backend_config = load_backend_config(args.ocr_backend_config)
    _, backend_manifest = load_json_object(args.ocr_backend_manifest)
    validate_artifact_source(
        backend_config_path=args.ocr_backend_config,
        artifact_manifest_path=args.ocr_backend_manifest,
        repository_root=repository_root,
    )
    pilots, source_images = load_development_source_pilots(
        source_root=args.source_root,
        source_file_manifest=source_file_manifest,
        v1_config=v1_config,
        selection_manifest=selection,
        derived_repo=args.dataset_repo,
    )
    engine = None
    if args.require_ocr_replay:
        _, engine = runtime_identity_and_engine(
            backend_config=backend_config,
            model_dir=args.model_dir,
            wheel_dir=args.wheel_dir,
        )
    paths = {
        "selection_manifest": args.selection_manifest,
        "v1_config": args.v1_config,
        "source_file_manifest": args.source_file_manifest,
        "ocr_backend_config": args.ocr_backend_config,
        "ocr_backend_manifest": args.ocr_backend_manifest,
    }
    result = validate_artifact(
        output_dir=args.output_dir,
        backend_config=backend_config,
        backend_config_sha256=sha256_file(args.ocr_backend_config),
        selection_manifest=selection,
        expected_input_sha256={key: sha256_file(path) for key, path in paths.items()},
        expected_dataset_repo=args.dataset_repo,
        expected_ocr_backend_provenance=build_ocr_backend_provenance(
            backend_config_sha256=sha256_file(args.ocr_backend_config),
            backend_manifest_sha256=sha256_file(args.ocr_backend_manifest),
            backend_manifest=backend_manifest,
        ),
        expected_generator_git_revision=args.git_revision,
        repository_root=repository_root,
        pilots_by_source=pilots,
        source_image_payloads=source_images,
        ocr_engine=engine,
        require_ocr_replay=args.require_ocr_replay,
    )
    print(canonical_json_bytes(result).decode("utf-8"))


if __name__ == "__main__":
    main()
