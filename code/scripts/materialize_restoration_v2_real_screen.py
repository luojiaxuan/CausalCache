"""Materialize the frozen restoration-v2 real-screen OCR golden payload."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath

from causalcache.data.restoration_v2_real_screen import (
    IMAGE_TAR_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    OCR_JSONL_RELATIVE_PATH,
    artifact_gitattributes_bytes,
    artifact_readme_bytes,
    artifact_tree_identity,
    build_candidate_pool,
    build_ocr_jsonl,
    build_payload_manifest,
    load_screening_image_payloads,
    pretty_json_bytes,
    select_real_screen_golden,
    sha256_file,
    write_selected_image_tar,
)
from causalcache.restoration_v2_contract import RestorationV2Contract
from causalcache.restoration_v2_text_backend import (
    load_backend_config,
    run_rapidocr_record,
)
from scripts.validate_restoration_v2_real_screen import (
    BACKEND_CONFIG_SHA256,
    DERIVED_DATASET_REPO,
    EXPOSURE_MANIFEST_SHA256,
    SCIENTIFIC_CONTRACT_SHA256,
    SELECTION_MANIFEST_SHA256,
    SOURCE_FILE_MANIFEST_SHA256,
    V1_CONFIG_SHA256,
    generator_identity,
    load_json_object,
    runtime_identity_and_engine,
    validate_source_contract,
    verify_git_checkout,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-contract", type=Path, required=True)
    parser.add_argument("--v1-config", type=Path, required=True)
    parser.add_argument("--source-file-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--exposure-manifest", type=Path, required=True)
    parser.add_argument("--backend-config", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _payload_path(root: Path, relative: str) -> Path:
    return root.joinpath(*PurePosixPath(relative).parts)


def _require_input_hash(path: Path, expected_sha256: str) -> None:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"real-screen materializer input SHA256 drifted: {path}")


def main() -> int:
    args = _build_parser().parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.git_revision) is None:
        raise ValueError("--git-revision must be a full 40-character Git SHA")
    if args.output_dir.exists():
        raise FileExistsError("real-screen output directory must not already exist")
    repository_root = Path(__file__).resolve().parents[2]
    verify_git_checkout(repository_root, args.git_revision)
    source_result = validate_source_contract(
        args.source_contract,
        repository_root=repository_root,
    )
    expected_inputs = {
        args.v2_contract: SCIENTIFIC_CONTRACT_SHA256,
        args.v1_config: V1_CONFIG_SHA256,
        args.source_file_manifest: SOURCE_FILE_MANIFEST_SHA256,
        args.selection_manifest: SELECTION_MANIFEST_SHA256,
        args.exposure_manifest: EXPOSURE_MANIFEST_SHA256,
        args.backend_config: BACKEND_CONFIG_SHA256,
    }
    for path, digest in expected_inputs.items():
        _require_input_hash(path, digest)

    contract = RestorationV2Contract.load(args.v2_contract)
    if contract.source_sha256 != SCIENTIFIC_CONTRACT_SHA256:
        raise ValueError("real-screen scientific contract identity drifted")
    v1_config = load_json_object(args.v1_config)
    source_file_manifest = load_json_object(args.source_file_manifest)
    selection_manifest = load_json_object(args.selection_manifest)
    load_json_object(args.exposure_manifest)
    source_contract = load_json_object(args.source_contract)
    backend_config = load_backend_config(args.backend_config)

    image_payloads = load_screening_image_payloads(
        selection_manifest,
        source_root=args.source_root,
        source_file_manifest=source_file_manifest,
        v1_config=v1_config,
        derived_repo=DERIVED_DATASET_REPO,
    )
    candidate_pool = build_candidate_pool(
        selection_manifest,
        image_payloads=image_payloads,
        backend_config=backend_config,
    )
    selected = select_real_screen_golden(
        candidate_pool,
        backend_config=backend_config,
    )
    runtime, engine = runtime_identity_and_engine(
        backend_config=backend_config,
        model_dir=args.model_dir,
        wheel_dir=args.wheel_dir,
    )
    ocr_records = {}
    for record in selected:
        image_sha256 = record["image_sha256"]
        ocr_records[image_sha256] = run_rapidocr_record(
            engine=engine,
            backend_config=backend_config,
            image_member_path=record["image_member_path"],
            image_bytes=image_payloads[record["image_member_path"]],
            backend_config_sha256=BACKEND_CONFIG_SHA256,
        )
    ocr_jsonl = build_ocr_jsonl(
        selected,
        ocr_records_by_sha256=ocr_records,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_tar_path = _payload_path(args.output_dir, IMAGE_TAR_RELATIVE_PATH)
    image_tar_record = write_selected_image_tar(
        image_tar_path,
        selected,
        image_payloads=image_payloads,
    )
    ocr_jsonl_path = _payload_path(args.output_dir, OCR_JSONL_RELATIVE_PATH)
    _exclusive_write(ocr_jsonl_path, ocr_jsonl)
    ocr_jsonl_record = {
        "path": OCR_JSONL_RELATIVE_PATH,
        "size_bytes": len(ocr_jsonl),
        "sha256": sha256_file(ocr_jsonl_path),
        "record_count": len(selected),
    }
    manifest = build_payload_manifest(
        dataset_repo=DERIVED_DATASET_REPO,
        source_contract_sha256=source_result["source_contract_sha256"],
        inputs=source_contract["inputs"],
        generator=generator_identity(
            source_contract,
            git_revision=args.git_revision,
        ),
        runtime=runtime,
        backend_config=backend_config,
        candidate_pool=candidate_pool,
        selected=selected,
        ocr_records_by_sha256=ocr_records,
        image_tar_file=image_tar_record,
        ocr_jsonl_file=ocr_jsonl_record,
    )
    manifest_path = _payload_path(args.output_dir, MANIFEST_RELATIVE_PATH)
    _exclusive_write(manifest_path, pretty_json_bytes(manifest))
    readme_path = args.output_dir / "README.md"
    _exclusive_write(readme_path, artifact_readme_bytes())
    gitattributes_path = args.output_dir / ".gitattributes"
    _exclusive_write(gitattributes_path, artifact_gitattributes_bytes())
    tree = artifact_tree_identity(args.output_dir)
    print(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "protocol_id": "causalcache_restoration_v2",
                "outcome": "MATERIALIZED_POLICY_BLIND_REAL_SCREEN_GOLDEN",
                "git_revision": args.git_revision,
                "output_dir": str(args.output_dir),
                "source_contract_sha256": source_result[
                    "source_contract_sha256"
                ],
                "payload_index_sha256": manifest["payload_index_sha256"],
                "eligible_occurrence_count": 180,
                "unique_image_count": 75,
                "selected_image_count": len(selected),
                "files": tree["files"],
                "artifact_tree_sha256": tree["artifact_tree_sha256"],
                "confirm_images_used": False,
                "policy_output_used": False,
                "restoration_output_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
