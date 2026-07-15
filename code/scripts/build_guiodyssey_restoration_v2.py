"""Build the complete policy-blind GUIOdyssey restoration-v2 artifact."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from causalcache.data.guiodyssey_restoration_v2 import (
    EXPECTED_FORMAL_COUNTS,
    build_ocr_backend_provenance,
    build_trajectory_records,
    generate_ocr_records,
    load_json_object,
    load_selected_source_pilots,
    required_image_member_paths,
    sha256_bytes,
    sha256_file,
    source_dataset_identity_from_v1_config,
    validate_artifact,
    validate_formal_image_inventory,
    validate_frozen_inputs,
    validate_ocr_runtime_identity,
    write_artifact,
)
from causalcache.restoration_v2_contract import RestorationV2Contract
from causalcache.restoration_v2_text_backend import load_backend_config
from scripts.validate_restoration_v2_ocr_backend import validate_artifact_source
from scripts.validate_restoration_v2_real_screen import runtime_identity_and_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-contract", type=Path, required=True)
    parser.add_argument("--v1-config", type=Path, required=True)
    parser.add_argument("--source-file-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--exposure-manifest", type=Path, required=True)
    parser.add_argument("--ocr-backend-config", type=Path, required=True)
    parser.add_argument("--ocr-backend-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        text=True,
    )
    if status:
        raise ValueError("formal derived materialization requires a clean checkout")
    canonical = subprocess.check_output(
        ["git", "rev-parse", "origin/main"],
        cwd=repository_root,
        text=True,
    ).strip()
    if canonical != expected_revision:
        raise ValueError("formal derived materialization must use canonical origin/main")


def _identity(repository_root: Path, path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("formal input files must belong to the repository checkout") from error
    return {"path": relative, "sha256": sha256_file(resolved)}


def _generator(repository_root: Path, git_revision: str) -> dict[str, str]:
    paths = {
        "module": "code/causalcache/data/guiodyssey_restoration_v2.py",
        "build_cli": "code/scripts/build_guiodyssey_restoration_v2.py",
        "validator": "code/scripts/validate_guiodyssey_restoration_v2.py",
    }
    return {
        "git_revision": git_revision,
        **{f"{name}_path": path for name, path in paths.items()},
        **{
            f"{name}_sha256": sha256_file(repository_root / path)
            for name, path in paths.items()
        },
    }


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    _verify_git_checkout(repository_root, args.git_revision)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")

    contract = RestorationV2Contract.load(args.v2_contract)
    _, v1_config = load_json_object(args.v1_config)
    _, source_file_manifest = load_json_object(args.source_file_manifest)
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
    expected_dataset_repo = str(contract.data["data"]["derived_artifact"]["repo"])
    expected_source_dataset = source_dataset_identity_from_v1_config(v1_config)

    pilots_by_source, source_images = load_selected_source_pilots(
        source_root=args.source_root,
        source_file_manifest=source_file_manifest,
        v1_config=v1_config,
        selection_manifest=selection_manifest,
        derived_repo=expected_dataset_repo,
    )
    backend_config_sha256 = sha256_file(args.ocr_backend_config)
    ocr_backend_provenance = build_ocr_backend_provenance(
        backend_config_sha256=backend_config_sha256,
        backend_manifest_sha256=sha256_file(args.ocr_backend_manifest),
        backend_manifest=backend_manifest,
    )
    required_paths = required_image_member_paths(
        pilots_by_source=pilots_by_source,
        selection_manifest=selection_manifest,
    )
    validate_formal_image_inventory(
        required_paths,
        selection_manifest=selection_manifest,
    )
    image_inventory = {path: source_images[path] for path in required_paths}
    ocr_runtime_identity, engine = runtime_identity_and_engine(
        backend_config=backend_config,
        model_dir=args.model_dir,
        wheel_dir=args.wheel_dir,
    )
    validate_ocr_runtime_identity(
        ocr_runtime_identity,
        backend_config=backend_config,
    )
    ocr_records, generated_ocr_aggregate_sha256 = generate_ocr_records(
        engine=engine,
        backend_config=backend_config,
        backend_config_sha256=backend_config_sha256,
        image_payloads=image_inventory,
    )
    trajectories, image_payloads = build_trajectory_records(
        pilots_by_source=pilots_by_source,
        source_image_payloads=source_images,
        selection_manifest=selection_manifest,
        ocr_records_by_path=ocr_records,
        backend_config=backend_config,
        backend_config_sha256=backend_config_sha256,
    )
    inputs = {
        "scientific_contract": _identity(repository_root, args.v2_contract),
        "v1_config": _identity(repository_root, args.v1_config),
        "source_file_manifest": _identity(
            repository_root,
            args.source_file_manifest,
        ),
        "selection_manifest": _identity(repository_root, args.selection_manifest),
        "exposure_manifest": _identity(repository_root, args.exposure_manifest),
        "ocr_backend_config": _identity(repository_root, args.ocr_backend_config),
        "ocr_backend_manifest": _identity(
            repository_root,
            args.ocr_backend_manifest,
        ),
    }
    tree = write_artifact(
        output_dir=args.output_dir,
        dataset_repo=expected_dataset_repo,
        trajectories=trajectories,
        image_payloads=image_payloads,
        ocr_records_by_path=ocr_records,
        inputs=inputs,
        generator=_generator(repository_root, args.git_revision),
        ocr_backend_provenance=ocr_backend_provenance,
        ocr_runtime_identity=ocr_runtime_identity,
        formal_counts_enforced=True,
    )
    result = validate_artifact(
        output_dir=args.output_dir,
        backend_config=backend_config,
        backend_config_sha256=backend_config_sha256,
        selection_manifest=selection_manifest,
        expected_input_sha256={
            key: value["sha256"] for key, value in inputs.items()
        },
        expected_ocr_backend_provenance=ocr_backend_provenance,
        expected_ocr_runtime_identity=ocr_runtime_identity,
        expected_generator_git_revision=args.git_revision,
        expected_dataset_repo=expected_dataset_repo,
        expected_source_dataset=expected_source_dataset,
        repository_root=repository_root,
        require_formal=True,
        ocr_engine=engine,
        require_ocr_replay=True,
    )
    if result["artifact_tree_sha256"] != tree["artifact_tree_sha256"]:
        raise RuntimeError("post-write artifact tree identity drifted")
    if result["ocr_record_aggregate_sha256"] != generated_ocr_aggregate_sha256:
        raise RuntimeError("post-write OCR record aggregate identity drifted")
    if result["ocr_replay_record_count"] != EXPECTED_FORMAL_COUNTS[
        "ocr_record_count"
    ]:
        raise RuntimeError("post-write exact OCR replay count drifted")
    if result["ocr_replay_aggregate_sha256"] != (
        generated_ocr_aggregate_sha256
    ):
        raise RuntimeError("post-write exact OCR replay aggregate drifted")
    print(
        json.dumps(
            {
                **result,
                "output_dir": str(args.output_dir.resolve()),
                "generated_ocr_record_count": len(ocr_records),
                "generated_ocr_record_aggregate_sha256": (
                    generated_ocr_aggregate_sha256
                ),
                "selection_manifest_sha256": sha256_bytes(
                    args.selection_manifest.read_bytes()
                ),
                "exposure_manifest_sha256": sha256_bytes(
                    args.exposure_manifest.read_bytes()
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
