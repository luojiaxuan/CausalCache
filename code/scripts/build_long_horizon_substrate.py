"""Build the policy-blind long-horizon GUIOdyssey development substrate."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Sequence

from causalcache.long_horizon_data import (
    DATASET_REPO,
    build_long_horizon_records,
    build_ocr_backend_provenance,
    canonical_json_bytes,
    generate_ocr_records,
    load_development_source_pilots,
    load_json_object,
    required_image_member_paths,
    ocr_record_aggregate_sha256,
    sha256_file,
    validate_artifact,
    write_artifact,
)
from causalcache.long_horizon_contract import (
    CANONICAL_CONFIG_PATH,
    RUNNER_FREEZE_B_PATH,
    SELECTION_MANIFEST_FREEZE_PATH,
    LongHorizonContract,
)
from causalcache.restoration_v2_text_backend import load_backend_config
from scripts.validate_restoration_v2_ocr_backend import validate_artifact_source
from scripts.validate_restoration_v2_real_screen import runtime_identity_and_engine


GENERATOR_PATHS = {
    "module": "code/causalcache/long_horizon_data.py",
    "build_cli": "code/scripts/build_long_horizon_substrate.py",
    "validator": "code/scripts/validate_long_horizon_substrate.py",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=Path(CANONICAL_CONFIG_PATH))
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--v1-config", type=Path, required=True)
    parser.add_argument("--source-file-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--ocr-backend-config", type=Path, required=True)
    parser.add_argument("--ocr-backend-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    return parser.parse_args(argv)


def verify_pushed_checkout(repository_root: Path, revision: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("--git-revision must be a full Git SHA")

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=repository_root, text=True
        ).strip()

    branch = git("symbolic-ref", "--short", "HEAD")
    if git("rev-parse", "HEAD") != revision:
        raise ValueError("--git-revision differs from checkout HEAD")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("formal long-horizon build requires a clean checkout")
    remote_ref = f"refs/remotes/origin/{branch}"
    if git("rev-parse", remote_ref) != revision:
        raise ValueError("long-horizon build requires the pushed current branch")
    advertised = git(
        "ls-remote", "--exit-code", "origin", f"refs/heads/{branch}"
    ).split()
    if len(advertised) != 2 or advertised[0] != revision:
        raise ValueError("canonical remote branch does not advertise the checkout")


def _identity(repository_root: Path, path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("formal long-horizon inputs must belong to Git") from error
    return {"path": relative, "sha256": sha256_file(resolved)}


def generator_identity(repository_root: Path, revision: str) -> dict[str, str]:
    return {
        "git_revision": revision,
        **{f"{name}_path": path for name, path in GENERATOR_PATHS.items()},
        **{
            f"{name}_sha256": sha256_file(repository_root / path)
            for name, path in GENERATOR_PATHS.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    verify_pushed_checkout(repository_root, args.git_revision)
    contract = LongHorizonContract.load(
        args.contract,
        repository_root=repository_root,
        require_runner_absent=False,
    )
    if (repository_root / RUNNER_FREEZE_B_PATH).exists():
        raise ValueError("substrate must be built before Execution-B runner freeze")
    frozen_paths = {
        "selection_manifest": SELECTION_MANIFEST_FREEZE_PATH,
        "v1_config": contract.source_pool["parent_protocol_config"]["path"],
        "source_file_manifest": contract.source_pool["source_file_manifest"]["path"],
        "ocr_backend_config": contract.data["substrate_profile"]["ocr_backend_config"]["path"],
        "ocr_backend_manifest": contract.data["substrate_profile"]["ocr_backend_manifest"]["path"],
    }
    observed_paths = {
        "selection_manifest": args.selection_manifest,
        "v1_config": args.v1_config,
        "source_file_manifest": args.source_file_manifest,
        "ocr_backend_config": args.ocr_backend_config,
        "ocr_backend_manifest": args.ocr_backend_manifest,
    }
    for key, path in observed_paths.items():
        if path.resolve() != (repository_root / frozen_paths[key]).resolve():
            raise ValueError(f"{key} differs from the frozen Source-A path")
    if args.dataset_repo != contract.data["artifact_plan"]["repo"]:
        raise ValueError("dataset repo differs from the frozen artifact destination")
    input_paths = {
        "selection_manifest": args.selection_manifest,
        "v1_config": args.v1_config,
        "source_file_manifest": args.source_file_manifest,
        "ocr_backend_config": args.ocr_backend_config,
        "ocr_backend_manifest": args.ocr_backend_manifest,
    }
    _, selection = load_json_object(args.selection_manifest)
    _, v1_config = load_json_object(args.v1_config)
    _, source_file_manifest = load_json_object(args.source_file_manifest)
    _, backend_manifest = load_json_object(args.ocr_backend_manifest)
    backend_config = load_backend_config(args.ocr_backend_config)
    validate_artifact_source(
        backend_config_path=args.ocr_backend_config,
        artifact_manifest_path=args.ocr_backend_manifest,
        repository_root=repository_root,
    )
    inputs = {
        key: _identity(repository_root, path) for key, path in input_paths.items()
    }
    pilots, source_images = load_development_source_pilots(
        source_root=args.source_root,
        source_file_manifest=source_file_manifest,
        v1_config=v1_config,
        selection_manifest=selection,
        derived_repo=args.dataset_repo,
    )
    required_paths = required_image_member_paths(
        pilots_by_source=pilots,
        selection_manifest=selection,
    )
    images = {path: source_images[path] for path in required_paths}
    runtime, engine = runtime_identity_and_engine(
        backend_config=backend_config,
        model_dir=args.model_dir,
        wheel_dir=args.wheel_dir,
    )
    backend_config_sha256 = sha256_file(args.ocr_backend_config)
    ocr_records, generated_ocr_sha256 = generate_ocr_records(
        engine=engine,
        backend_config=backend_config,
        backend_config_sha256=backend_config_sha256,
        image_payloads=images,
    )
    trajectories, features, derived_images = build_long_horizon_records(
        pilots_by_source=pilots,
        source_image_payloads=source_images,
        selection_manifest=selection,
        ocr_records_by_path=ocr_records,
        backend_config=backend_config,
        backend_config_sha256=backend_config_sha256,
    )
    if derived_images != images:
        raise RuntimeError("long-horizon selected image inventory changed")
    provenance = build_ocr_backend_provenance(
        backend_config_sha256=backend_config_sha256,
        backend_manifest_sha256=sha256_file(args.ocr_backend_manifest),
        backend_manifest=backend_manifest,
    )
    generator = generator_identity(repository_root, args.git_revision)
    tree = write_artifact(
        output_dir=args.output_dir,
        dataset_repo=args.dataset_repo,
        trajectories=trajectories,
        feature_states=features,
        image_payloads=images,
        ocr_records_by_path=ocr_records,
        selection_manifest=selection,
        inputs=inputs,
        generator=generator,
        ocr_backend_provenance=provenance,
        ocr_runtime_identity=runtime,
    )
    result = validate_artifact(
        output_dir=args.output_dir,
        backend_config=backend_config,
        backend_config_sha256=backend_config_sha256,
        selection_manifest=selection,
        expected_input_sha256={key: value["sha256"] for key, value in inputs.items()},
        expected_dataset_repo=args.dataset_repo,
        expected_ocr_backend_provenance=provenance,
        expected_generator_git_revision=args.git_revision,
        repository_root=repository_root,
        pilots_by_source=pilots,
        source_image_payloads=source_images,
        ocr_engine=engine,
        require_ocr_replay=True,
    )
    if result["artifact_tree_sha256"] != tree["artifact_tree_sha256"]:
        raise RuntimeError("post-write long-horizon tree identity drifted")
    if generated_ocr_sha256 != ocr_record_aggregate_sha256(ocr_records):
        raise RuntimeError("generated OCR aggregate identity drifted")
    print(canonical_json_bytes(result).decode("utf-8"))


if __name__ == "__main__":
    main()
