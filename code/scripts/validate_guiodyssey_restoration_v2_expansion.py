"""Validate the formal GUIOdyssey label-expansion artifact offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from causalcache.data.guiodyssey_restoration_v2_expansion import (
    MANIFEST_RELATIVE_PATH,
    _load_json_object,
    artifact_tree_identity,
    build_ocr_backend_provenance,
    pretty_json_bytes,
    sha256_file,
    source_dataset_identity_from_v1_config,
    validate_artifact,
    validate_frozen_inputs,
    validate_ocr_runtime_identity,
)
from causalcache.restoration_v2_text_backend import load_backend_config
from scripts.build_guiodyssey_restoration_v2_expansion import (
    EXPECTED_DATASET_REPO,
    _generator,
    _identity,
    _require_canonical_inputs,
    _verify_git_checkout,
)
from scripts.validate_restoration_v2_ocr_backend import validate_artifact_source
from scripts.validate_restoration_v2_real_screen import runtime_identity_and_engine


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-expansion-config", type=Path, required=True)
    parser.add_argument("--parent-selection-manifest", type=Path, required=True)
    parser.add_argument("--expansion-selection-manifest", type=Path, required=True)
    parser.add_argument("--v1-config", type=Path, required=True)
    parser.add_argument("--source-file-manifest", type=Path, required=True)
    parser.add_argument("--ocr-backend-config", type=Path, required=True)
    parser.add_argument("--ocr-backend-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--git-revision", required=True)
    return parser.parse_args(argv)


def _reject_symlinks_and_special_entries(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("expansion artifact root must be a real directory")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"expansion artifact cannot contain symlinks: {path}")
        if not path.is_dir() and not path.is_file():
            raise ValueError(f"expansion artifact contains a special entry: {path}")


def _preflight_artifact_projection(
    *,
    output_dir: Path,
    repository_root: Path,
    git_revision: str,
    input_paths: dict[str, Path],
) -> None:
    _reject_symlinks_and_special_entries(output_dir)
    artifact_tree_identity(output_dir)
    payload, manifest = _load_json_object(output_dir / MANIFEST_RELATIVE_PATH)
    if payload != pretty_json_bytes(manifest):
        raise ValueError("expansion artifact manifest must be canonical pretty JSON")
    if manifest.get("generator") != _generator(repository_root, git_revision):
        raise ValueError("expansion artifact generator source binding drifted")
    expected_inputs = {
        key: _identity(repository_root, path) for key, path in input_paths.items()
    }
    if manifest.get("inputs") != expected_inputs:
        raise ValueError("expansion artifact canonical input binding drifted")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    _verify_git_checkout(repository_root, args.git_revision)
    input_paths = _require_canonical_inputs(repository_root, args)
    _preflight_artifact_projection(
        output_dir=args.output_dir,
        repository_root=repository_root,
        git_revision=args.git_revision,
        input_paths=input_paths,
    )

    _, expansion_config = _load_json_object(input_paths["label_expansion_config"])
    _, parent_selection = _load_json_object(
        input_paths["parent_selection_manifest"]
    )
    _, expansion_selection = _load_json_object(
        input_paths["expansion_selection_manifest"]
    )
    _, v1_config = _load_json_object(input_paths["v1_config"])
    _load_json_object(input_paths["source_file_manifest"])
    backend_config = load_backend_config(input_paths["ocr_backend_config"])
    _, backend_manifest = _load_json_object(input_paths["ocr_backend_manifest"])
    validate_artifact_source(
        backend_config_path=input_paths["ocr_backend_config"],
        artifact_manifest_path=input_paths["ocr_backend_manifest"],
        repository_root=repository_root,
    )
    validate_frozen_inputs(
        expansion_manifest=expansion_selection,
        expansion_manifest_sha256=sha256_file(
            input_paths["expansion_selection_manifest"]
        ),
        expansion_config=expansion_config,
        expansion_config_sha256=sha256_file(input_paths["label_expansion_config"]),
        parent_selection=parent_selection,
        parent_selection_sha256=sha256_file(
            input_paths["parent_selection_manifest"]
        ),
        v1_config=v1_config,
        v1_config_sha256=sha256_file(input_paths["v1_config"]),
        source_file_manifest_sha256=sha256_file(
            input_paths["source_file_manifest"]
        ),
        backend_config=backend_config,
    )
    ocr_runtime_identity, engine = runtime_identity_and_engine(
        backend_config=backend_config,
        model_dir=args.model_dir,
        wheel_dir=args.wheel_dir,
    )
    validate_ocr_runtime_identity(
        ocr_runtime_identity, backend_config=backend_config
    )
    ocr_backend_provenance = build_ocr_backend_provenance(
        backend_config_sha256=sha256_file(input_paths["ocr_backend_config"]),
        backend_manifest_sha256=sha256_file(
            input_paths["ocr_backend_manifest"]
        ),
        backend_manifest=backend_manifest,
    )
    result = validate_artifact(
        output_dir=args.output_dir,
        backend_config=backend_config,
        backend_config_sha256=sha256_file(input_paths["ocr_backend_config"]),
        selection_manifest=expansion_selection,
        parent_selection=parent_selection,
        selection_manifest_sha256=sha256_file(
            input_paths["expansion_selection_manifest"]
        ),
        parent_selection_sha256=sha256_file(
            input_paths["parent_selection_manifest"]
        ),
        expected_input_sha256={
            key: sha256_file(path) for key, path in input_paths.items()
        },
        expected_ocr_backend_provenance=ocr_backend_provenance,
        expected_ocr_runtime_identity=ocr_runtime_identity,
        expected_generator_git_revision=args.git_revision,
        expected_dataset_repo=EXPECTED_DATASET_REPO,
        expected_source_dataset=source_dataset_identity_from_v1_config(v1_config),
        repository_root=repository_root,
        require_formal=True,
        ocr_engine=engine,
        require_ocr_replay=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
