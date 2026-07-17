"""Build the formal policy-blind GUIOdyssey label-expansion artifact."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Sequence

from causalcache.data.guiodyssey_restoration_v2_expansion import (
    EXPECTED_FORMAL_COUNTS,
    _load_json_object,
    build_ocr_backend_provenance,
    build_trajectory_records,
    generate_ocr_records,
    load_expansion_source_pilots,
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
from causalcache.restoration_v2_text_backend import load_backend_config
from scripts.validate_restoration_v2_ocr_backend import validate_artifact_source
from scripts.validate_restoration_v2_real_screen import runtime_identity_and_engine


EXPECTED_DATASET_REPO = "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile"
CANONICAL_INPUT_PATHS = {
    "label_expansion_config": (
        "code/configs/causalcache_restoration_v2_2_label_expansion_v1.json"
    ),
    "parent_selection_manifest": "data/manifests/restoration_v2_selection.json",
    "expansion_selection_manifest": (
        "data/manifests/restoration_v2_2_label_expansion_selection.json"
    ),
    "v1_config": "code/configs/independent_reference_gate_v1.json",
    "source_file_manifest": (
        "data/manifests/independent_reference_gate_v1_source_files.json"
    ),
    "ocr_backend_config": "code/configs/restoration_v2_ocr_backend.json",
    "ocr_backend_manifest": "data/manifests/restoration_v2_ocr_backend.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-expansion-config", type=Path, required=True)
    parser.add_argument("--parent-selection-manifest", type=Path, required=True)
    parser.add_argument("--expansion-selection-manifest", type=Path, required=True)
    parser.add_argument("--v1-config", type=Path, required=True)
    parser.add_argument("--source-file-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--ocr-backend-config", type=Path, required=True)
    parser.add_argument("--ocr-backend-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _verify_git_checkout(repository_root: Path, expected_revision: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
        raise ValueError("--git-revision must be a full 40-character Git SHA")

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=repository_root, text=True
        ).strip()

    if git("symbolic-ref", "--short", "HEAD") != "main":
        raise ValueError("formal expansion build requires symbolic branch main")
    if git("rev-parse", "HEAD") != expected_revision:
        raise ValueError("--git-revision does not match checkout HEAD")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("formal expansion build requires a clean checkout")
    if git("rev-parse", "refs/remotes/origin/main") != expected_revision:
        raise ValueError("formal expansion build requires canonical origin/main")
    advertised = git(
        "ls-remote", "--exit-code", "origin", "refs/heads/main"
    ).split()
    if len(advertised) != 2 or advertised[0] != expected_revision:
        raise ValueError("formal expansion build requires pushed canonical main")


def _require_canonical_inputs(
    repository_root: Path, args: argparse.Namespace
) -> dict[str, Path]:
    root = repository_root.resolve()
    result: dict[str, Path] = {}
    for name, relative in CANONICAL_INPUT_PATHS.items():
        supplied = Path(getattr(args, name))
        if supplied.is_symlink():
            raise ValueError(f"canonical input cannot be a symlink: {name}")
        expected = (root / relative).resolve(strict=True)
        try:
            observed = supplied.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(f"canonical input does not exist: {name}") from error
        if observed != expected or not observed.is_file():
            raise ValueError(f"canonical input path drifted: {name}")
        result[name] = observed
    return result


def _identity(repository_root: Path, path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("formal inputs must belong to the repository") from error
    return {"path": relative, "sha256": sha256_file(resolved)}


def _generator(repository_root: Path, git_revision: str) -> dict[str, str]:
    paths = {
        "module": (
            "code/causalcache/data/guiodyssey_restoration_v2_expansion.py"
        ),
        "build_cli": (
            "code/scripts/build_guiodyssey_restoration_v2_expansion.py"
        ),
        "validator": (
            "code/scripts/validate_guiodyssey_restoration_v2_expansion.py"
        ),
    }
    return {
        "git_revision": git_revision,
        **{f"{name}_path": path for name, path in paths.items()},
        **{
            f"{name}_sha256": sha256_file(repository_root / path)
            for name, path in paths.items()
        },
    }


def _output_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink() or os.path.lexists(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    _verify_git_checkout(repository_root, args.git_revision)
    input_paths = _require_canonical_inputs(repository_root, args)
    if _output_exists(args.output_dir):
        raise FileExistsError(f"output directory already exists: {args.output_dir}")

    _, expansion_config = _load_json_object(input_paths["label_expansion_config"])
    _, parent_selection = _load_json_object(
        input_paths["parent_selection_manifest"]
    )
    _, expansion_selection = _load_json_object(
        input_paths["expansion_selection_manifest"]
    )
    _, v1_config = _load_json_object(input_paths["v1_config"])
    _, source_file_manifest = _load_json_object(input_paths["source_file_manifest"])
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
    expected_source_dataset = source_dataset_identity_from_v1_config(v1_config)

    pilots_by_source, source_images = load_expansion_source_pilots(
        source_root=args.source_root,
        source_file_manifest=source_file_manifest,
        v1_config=v1_config,
        selection_manifest=expansion_selection,
        derived_repo=EXPECTED_DATASET_REPO,
    )
    backend_config_sha256 = sha256_file(input_paths["ocr_backend_config"])
    ocr_backend_provenance = build_ocr_backend_provenance(
        backend_config_sha256=backend_config_sha256,
        backend_manifest_sha256=sha256_file(
            input_paths["ocr_backend_manifest"]
        ),
        backend_manifest=backend_manifest,
    )
    required_paths = required_image_member_paths(
        pilots_by_source=pilots_by_source,
        selection_manifest=expansion_selection,
    )
    validate_formal_image_inventory(
        required_paths, selection_manifest=expansion_selection
    )
    image_inventory = {path: source_images[path] for path in required_paths}
    ocr_runtime_identity, engine = runtime_identity_and_engine(
        backend_config=backend_config,
        model_dir=args.model_dir,
        wheel_dir=args.wheel_dir,
    )
    validate_ocr_runtime_identity(
        ocr_runtime_identity, backend_config=backend_config
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
        selection_manifest=expansion_selection,
        ocr_records_by_path=ocr_records,
        backend_config=backend_config,
        backend_config_sha256=backend_config_sha256,
    )
    inputs = {
        key: _identity(repository_root, path) for key, path in input_paths.items()
    }
    tree = write_artifact(
        output_dir=args.output_dir,
        dataset_repo=EXPECTED_DATASET_REPO,
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
        selection_manifest=expansion_selection,
        parent_selection=parent_selection,
        selection_manifest_sha256=sha256_file(
            input_paths["expansion_selection_manifest"]
        ),
        parent_selection_sha256=sha256_file(
            input_paths["parent_selection_manifest"]
        ),
        expected_input_sha256={key: value["sha256"] for key, value in inputs.items()},
        expected_ocr_backend_provenance=ocr_backend_provenance,
        expected_ocr_runtime_identity=ocr_runtime_identity,
        expected_generator_git_revision=args.git_revision,
        expected_dataset_repo=EXPECTED_DATASET_REPO,
        expected_source_dataset=expected_source_dataset,
        repository_root=repository_root,
        require_formal=True,
        ocr_engine=engine,
        require_ocr_replay=True,
    )
    if result["artifact_tree_sha256"] != tree["artifact_tree_sha256"]:
        raise RuntimeError("post-write expansion tree identity drifted")
    if result["ocr_record_aggregate_sha256"] != generated_ocr_aggregate_sha256:
        raise RuntimeError("post-write expansion OCR identity drifted")
    if result["ocr_replay_record_count"] != EXPECTED_FORMAL_COUNTS[
        "ocr_record_count"
    ]:
        raise RuntimeError("post-write expansion OCR replay count drifted")
    if result["ocr_replay_aggregate_sha256"] != generated_ocr_aggregate_sha256:
        raise RuntimeError("post-write expansion OCR replay digest drifted")
    print(
        json.dumps(
            {
                **result,
                "output_dir": str(args.output_dir.resolve()),
                "generated_ocr_record_count": len(ocr_records),
                "generated_ocr_record_aggregate_sha256": (
                    generated_ocr_aggregate_sha256
                ),
                "expansion_selection_manifest_sha256": sha256_bytes(
                    input_paths["expansion_selection_manifest"].read_bytes()
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
