"""Validate restoration-v2 real-screen OCR source and payload artifacts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from causalcache.data.restoration_v2_real_screen import (
    ARTIFACT_ID,
    EXPECTED_CONFIRM_INTERSECTION_COUNT,
    EXPECTED_CONFIRM_UNIQUE_IMAGE_COUNT,
    EXPECTED_OCCURRENCE_COUNT,
    EXPECTED_ORIENTATION_COUNTS,
    EXPECTED_SELECTED_IMAGES,
    EXPECTED_STATE_COUNT,
    EXPECTED_UNIQUE_IMAGE_COUNT,
    IMAGE_TAR_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    OCR_JSONL_RELATIVE_PATH,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    artifact_gitattributes_bytes,
    artifact_readme_bytes,
    artifact_tree_identity,
    build_candidate_pool,
    build_ocr_jsonl,
    build_payload_manifest,
    build_selected_image_tar_bytes,
    canonical_json_bytes,
    load_screening_image_payloads,
    select_real_screen_golden,
    sha256_file,
)
from causalcache.restoration_v2_contract import RestorationV2Contract
from causalcache.restoration_v2_text_backend import (
    create_rapidocr_engine,
    load_backend_config,
    run_rapidocr_record,
    sha256_bytes,
    verify_model_files,
    verify_rapidocr_package_files,
    verify_recognizer_character_inventory,
    verify_runtime_packages,
    verify_wheel_files,
)


SCIENTIFIC_CONTRACT_SHA256 = (
    "9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc"
)
SELECTION_MANIFEST_SHA256 = (
    "292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b"
)
EXPOSURE_MANIFEST_SHA256 = (
    "bc1224826e0642c2374f6cfbebd676389d8c17ce6bf8a789f08903740cf97a95"
)
BACKEND_CONFIG_SHA256 = (
    "51e08a9565f804ecb1ee7f12887cc12742b84c04996fffbbde7c0f304e4bd036"
)
V1_CONFIG_SHA256 = (
    "7b62aa31c80536f28bc4e8a3d684ff535bc2315d44a31cb5f002ed6dcb493fd8"
)
SOURCE_FILE_MANIFEST_SHA256 = (
    "46f2240a07f46b6e283cb90e1f478f0b14a2e1499c52e2cc8e144898065adffc"
)
RUNTIME_LOCK_SHA256 = (
    "1c36a767a69ab158101a686ee94d67f04d837df17301f6043b77c87060a82c4e"
)
OCR_MODEL_REPO = "gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en"
OCR_MODEL_REVISION = "0dbc766a73ee88d10d52285d434dbfec58617835"
DERIVED_DATASET_REPO = "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile"
SOURCE_CONTRACT_ARTIFACT_ID = (
    "causalcache-restoration-v2-real-screen-source-v1"
)
SOURCE_CONTRACT_STATUS = "SOURCE_FROZEN_BEFORE_REAL_SCREEN_OCR_OUTPUT"
SOURCE_CONTRACT_PATH = "data/manifests/restoration_v2_real_screen_source.json"
EXPECTED_SOURCE_PATHS = {
    "code/causalcache/data/guiodyssey.py",
    "code/causalcache/data/guiodyssey_independent.py",
    "code/causalcache/data/restoration_v2_real_screen.py",
    "code/causalcache/data/restoration_v2_selection.py",
    "code/causalcache/restoration_v2_contract.py",
    "code/causalcache/restoration_v2_text_backend.py",
    "code/causalcache/schema.py",
    "code/configs/causalcache_restoration_v2.json",
    "code/configs/independent_reference_gate_v1.json",
    "code/configs/restoration_v2_ocr_backend.json",
    "code/requirements/restoration_v2_ocr_lock.txt",
    "code/scripts/materialize_restoration_v2_real_screen.py",
    "code/scripts/validate_restoration_v2_real_screen.py",
    "code/tests/test_restoration_v2_real_screen.py",
    "data/manifests/independent_reference_gate_v1_source_files.json",
    "data/manifests/restoration_v2_exposure.json",
    "data/manifests/restoration_v2_selection.json",
}


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _require_git_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{field} must be a full Git SHA")
    return value


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("artifact path must be a safe relative path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or str(parsed) != value or any(
        part in {".", ".."} for part in parsed.parts
    ):
        raise ValueError("artifact path must be a safe relative path")
    return value


def _expected_input_identities() -> dict[str, Any]:
    return {
        "scientific_contract": {
            "path": "code/configs/causalcache_restoration_v2.json",
            "sha256": SCIENTIFIC_CONTRACT_SHA256,
        },
        "selection_manifest": {
            "path": "data/manifests/restoration_v2_selection.json",
            "sha256": SELECTION_MANIFEST_SHA256,
        },
        "exposure_manifest": {
            "path": "data/manifests/restoration_v2_exposure.json",
            "sha256": EXPOSURE_MANIFEST_SHA256,
        },
        "ocr_backend_config": {
            "path": "code/configs/restoration_v2_ocr_backend.json",
            "sha256": BACKEND_CONFIG_SHA256,
        },
        "v1_selection_config": {
            "path": "code/configs/independent_reference_gate_v1.json",
            "sha256": V1_CONFIG_SHA256,
        },
        "source_file_manifest": {
            "path": "data/manifests/independent_reference_gate_v1_source_files.json",
            "sha256": SOURCE_FILE_MANIFEST_SHA256,
            "repo": "cua-lite/GUIOdyssey",
            "revision": "ea08072b30e523fb4492e4f4597505879ffcd63b",
            "file_count": 16,
            "total_bytes": 2252923738,
        },
        "ocr_runtime_lock": {
            "path": "code/requirements/restoration_v2_ocr_lock.txt",
            "sha256": RUNTIME_LOCK_SHA256,
        },
        "ocr_model_artifact": {
            "repo": OCR_MODEL_REPO,
            "repo_type": "model",
            "visibility": "private",
            "immutable_revision": OCR_MODEL_REVISION,
            "models": {
                "ch_PP-OCRv5_det_mobile.onnx": (
                    "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae"
                ),
                "ch_ppocr_mobile_v2.0_cls_mobile.onnx": (
                    "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"
                ),
                "en_PP-OCRv5_rec_mobile.onnx": (
                    "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8"
                ),
            },
        },
        "derived_dataset_destination": {
            "repo": DERIVED_DATASET_REPO,
            "repo_type": "dataset",
            "visibility": "private",
            "payload_prefix": "golden/real-screen-v1",
        },
    }


def validate_source_contract(
    source_contract_path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    manifest = load_json_object(source_contract_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("real-screen source schema_version drifted")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("real-screen source protocol_id drifted")
    if manifest.get("artifact_id") != SOURCE_CONTRACT_ARTIFACT_ID:
        raise ValueError("real-screen source artifact_id drifted")
    if manifest.get("status") != SOURCE_CONTRACT_STATUS:
        raise ValueError("real-screen source status drifted")
    if manifest.get("policy_output_generated_before_source_manifest") is not False:
        raise ValueError("real-screen source manifest must precede policy output")
    if manifest.get("restoration_output_generated_before_source_manifest") is not False:
        raise ValueError("real-screen source manifest must precede restoration output")
    if manifest.get("confirm_images_may_be_materialized") is not False:
        raise ValueError("real-screen source manifest must exclude confirm images")
    if manifest.get("inputs") != _expected_input_identities():
        raise ValueError("real-screen source input identities drifted")

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {root}")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list):
        raise ValueError("real-screen source_files must be a list")
    observed_paths = set()
    for record in source_files:
        if not isinstance(record, Mapping):
            raise ValueError("real-screen source file records must be objects")
        relative = _safe_relative_path(record.get("path"))
        if relative in observed_paths:
            raise ValueError("real-screen source file paths must be unique")
        expected_sha = _require_sha256(record.get("sha256"), "source file SHA256")
        source_path = root.joinpath(*PurePosixPath(relative).parts)
        if not source_path.is_file() or sha256_file(source_path) != expected_sha:
            raise ValueError(f"real-screen source file SHA256 drifted: {relative}")
        observed_paths.add(relative)
    if observed_paths != EXPECTED_SOURCE_PATHS:
        raise ValueError("real-screen source file inventory drifted")

    expected_selection = {
        "eligible_state_count": EXPECTED_STATE_COUNT,
        "eligible_occurrence_count": EXPECTED_OCCURRENCE_COUNT,
        "unique_image_count": EXPECTED_UNIQUE_IMAGE_COUNT,
        "orientation_counts": EXPECTED_ORIENTATION_COUNTS,
        "confirm_unique_image_count": EXPECTED_CONFIRM_UNIQUE_IMAGE_COUNT,
        "confirm_sha_intersection_count": EXPECTED_CONFIRM_INTERSECTION_COUNT,
        "same_sha_representative_path": "minimum_image_member_path",
        "orientation_definition": {
            "portrait": "height>width",
            "landscape": "width>height",
            "square": "INVALID_DERIVED_ARTIFACT_BEFORE_POLICY_OUTPUT",
        },
        "selected_images": {
            orientation: [dict(record) for record in records]
            for orientation, records in EXPECTED_SELECTED_IMAGES.items()
        },
    }
    if manifest.get("expected_selection") != expected_selection:
        raise ValueError("real-screen expected selection drifted")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "outcome": "PASSED_REAL_SCREEN_SOURCE_VALIDATION",
        "source_contract_sha256": sha256_file(source_contract_path),
        "source_file_count": len(source_files),
        "eligible_occurrence_count": EXPECTED_OCCURRENCE_COUNT,
        "unique_image_count": EXPECTED_UNIQUE_IMAGE_COUNT,
        "selected_image_count": 6,
        "confirm_images_used": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
    }


def runtime_identity_and_engine(
    *,
    backend_config: Mapping[str, Any],
    model_dir: str | Path,
    wheel_dir: str | Path,
) -> tuple[dict[str, Any], Any]:
    packages = verify_runtime_packages(backend_config)
    package_files = verify_rapidocr_package_files(backend_config)
    wheels = verify_wheel_files(backend_config, wheel_dir)
    models = verify_model_files(backend_config, model_dir)
    engine = create_rapidocr_engine(backend_config, model_dir)
    characters = verify_recognizer_character_inventory(engine, backend_config)
    return (
        {
            "runtime_packages": packages,
            "rapidocr_package_file_sha256": package_files,
            "wheel_sha256": wheels,
            "model_sha256": models,
            "recognizer_character_inventory": characters,
        },
        engine,
    )


def verify_git_checkout(repository_root: str | Path, expected_revision: str) -> None:
    revision = _require_git_sha(expected_revision, "formal run Git revision")
    root = Path(repository_root)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if actual != revision:
        raise ValueError("formal run Git revision differs from checkout HEAD")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    )
    if status:
        raise ValueError("formal real-screen run requires a clean Git checkout")
    pushed = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=root, text=True
    ).strip()
    if pushed != revision:
        raise ValueError("formal run revision must equal canonical origin/main")


def generator_identity(
    source_contract: Mapping[str, Any],
    *,
    git_revision: str,
) -> dict[str, Any]:
    source_by_path = {
        str(record["path"]): str(record["sha256"])
        for record in source_contract["source_files"]
    }
    paths = {
        "module": "code/causalcache/data/restoration_v2_real_screen.py",
        "materializer": "code/scripts/materialize_restoration_v2_real_screen.py",
        "validator": "code/scripts/validate_restoration_v2_real_screen.py",
    }
    return {
        "git_revision": _require_git_sha(git_revision, "generator Git revision"),
        **{
            f"{name}_path": path
            for name, path in paths.items()
        },
        **{
            f"{name}_sha256": source_by_path[path]
            for name, path in paths.items()
        },
    }


def _payload_file_record(path: Path, relative: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing real-screen payload file: {path}")
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_image_tar(
    path: Path,
    selected: Sequence[Mapping[str, Any]],
    *,
    raw_payloads: Mapping[str, bytes],
) -> dict[str, bytes]:
    expected_bytes, _ = build_selected_image_tar_bytes(
        selected,
        image_payloads=raw_payloads,
    )
    if path.read_bytes() != expected_bytes:
        raise ValueError("real-screen image tar is not canonical USTAR bytes")
    expected_names = [str(record["image_tar_member"]) for record in selected]
    if expected_names != sorted(expected_names):
        raise ValueError("expected real-screen image tar order drifted")
    extracted: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != expected_names:
            raise ValueError("real-screen image tar member inventory or order drifted")
        for member, record in zip(members, selected, strict=True):
            _safe_relative_path(member.name)
            if (
                not member.isfile()
                or member.type != tarfile.REGTYPE
                or member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mode != 0o644
                or member.pax_headers
            ):
                raise ValueError("real-screen image tar metadata drifted")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("real-screen image tar member is unreadable")
            payload = handle.read()
            if len(payload) != member.size or sha256_bytes(payload) != record[
                "image_sha256"
            ]:
                raise ValueError("real-screen image tar payload drifted")
            if payload != raw_payloads[record["image_member_path"]]:
                raise ValueError("real-screen tar image differs from pinned source bytes")
            extracted[record["image_sha256"]] = payload
    return extracted


def validate_artifact(
    *,
    output_dir: str | Path,
    repository_root: str | Path,
    git_revision: str,
    source_contract_path: str | Path,
    source_root: str | Path,
    v2_contract_path: str | Path,
    v1_config_path: str | Path,
    source_file_manifest_path: str | Path,
    selection_manifest_path: str | Path,
    exposure_manifest_path: str | Path,
    backend_config_path: str | Path,
    model_dir: str | Path,
    wheel_dir: str | Path,
) -> dict[str, Any]:
    verify_git_checkout(repository_root, git_revision)
    source_result = validate_source_contract(
        source_contract_path,
        repository_root=repository_root,
    )
    source_contract = load_json_object(source_contract_path)
    path_identities = {
        Path(v2_contract_path): SCIENTIFIC_CONTRACT_SHA256,
        Path(v1_config_path): V1_CONFIG_SHA256,
        Path(source_file_manifest_path): SOURCE_FILE_MANIFEST_SHA256,
        Path(selection_manifest_path): SELECTION_MANIFEST_SHA256,
        Path(exposure_manifest_path): EXPOSURE_MANIFEST_SHA256,
        Path(backend_config_path): BACKEND_CONFIG_SHA256,
    }
    for path, expected_sha in path_identities.items():
        if sha256_file(path) != expected_sha:
            raise ValueError(f"real-screen formal input SHA256 drifted: {path}")

    contract = RestorationV2Contract.load(v2_contract_path)
    if contract.source_sha256 != SCIENTIFIC_CONTRACT_SHA256:
        raise ValueError("real-screen scientific contract identity drifted")
    v1_config = load_json_object(v1_config_path)
    source_file_manifest = load_json_object(source_file_manifest_path)
    selection_manifest = load_json_object(selection_manifest_path)
    load_json_object(exposure_manifest_path)
    backend_config = load_backend_config(backend_config_path)
    raw_payloads = load_screening_image_payloads(
        selection_manifest,
        source_root=source_root,
        source_file_manifest=source_file_manifest,
        v1_config=v1_config,
        derived_repo=DERIVED_DATASET_REPO,
    )
    candidate_pool = build_candidate_pool(
        selection_manifest,
        image_payloads=raw_payloads,
        backend_config=backend_config,
    )
    selected = select_real_screen_golden(
        candidate_pool,
        backend_config=backend_config,
    )

    root = Path(output_dir)
    tree = artifact_tree_identity(root)
    gitattributes_path = root / ".gitattributes"
    if gitattributes_path.read_bytes() != artifact_gitattributes_bytes():
        raise ValueError("real-screen dataset .gitattributes drifted")
    readme_path = root / "README.md"
    if readme_path.read_bytes() != artifact_readme_bytes():
        raise ValueError("real-screen dataset README drifted")
    manifest_path = root.joinpath(*PurePosixPath(MANIFEST_RELATIVE_PATH).parts)
    image_tar_path = root.joinpath(*PurePosixPath(IMAGE_TAR_RELATIVE_PATH).parts)
    ocr_jsonl_path = root.joinpath(*PurePosixPath(OCR_JSONL_RELATIVE_PATH).parts)
    manifest = load_json_object(manifest_path)
    tar_payloads = _validate_image_tar(
        image_tar_path,
        selected,
        raw_payloads=raw_payloads,
    )
    runtime, engine = runtime_identity_and_engine(
        backend_config=backend_config,
        model_dir=model_dir,
        wheel_dir=wheel_dir,
    )
    config_sha256 = sha256_file(backend_config_path)
    replay_records = {}
    for record in selected:
        image_sha256 = record["image_sha256"]
        replay_records[image_sha256] = run_rapidocr_record(
            engine=engine,
            backend_config=backend_config,
            image_member_path=record["image_member_path"],
            image_bytes=tar_payloads[image_sha256],
            backend_config_sha256=config_sha256,
        )
    expected_jsonl = build_ocr_jsonl(
        selected,
        ocr_records_by_sha256=replay_records,
    )
    actual_jsonl = ocr_jsonl_path.read_bytes()
    if actual_jsonl != expected_jsonl:
        raise ValueError("real-screen OCR JSONL differs from independent replay")

    image_tar_record = _payload_file_record(image_tar_path, IMAGE_TAR_RELATIVE_PATH)
    image_tar_record.update(
        {
            "member_count": 6,
            "members": [
                {
                    "path": record["image_tar_member"],
                    "size_bytes": len(tar_payloads[record["image_sha256"]]),
                    "sha256": record["image_sha256"],
                }
                for record in selected
            ],
        }
    )
    jsonl_record = _payload_file_record(ocr_jsonl_path, OCR_JSONL_RELATIVE_PATH)
    jsonl_record["record_count"] = 6
    expected_manifest = build_payload_manifest(
        dataset_repo=DERIVED_DATASET_REPO,
        source_contract_sha256=source_result["source_contract_sha256"],
        inputs=source_contract["inputs"],
        generator=generator_identity(source_contract, git_revision=git_revision),
        runtime=runtime,
        backend_config=backend_config,
        candidate_pool=candidate_pool,
        selected=selected,
        ocr_records_by_sha256=replay_records,
        image_tar_file=image_tar_record,
        ocr_jsonl_file=jsonl_record,
    )
    if manifest != expected_manifest:
        raise ValueError("real-screen manifest differs from independently rebuilt payload")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "outcome": "PASSED_REAL_SCREEN_ARTIFACT_VALIDATION",
        "artifact_id": ARTIFACT_ID,
        "manifest_sha256": sha256_file(manifest_path),
        "image_tar_sha256": image_tar_record["sha256"],
        "ocr_jsonl_sha256": jsonl_record["sha256"],
        "payload_index_sha256": manifest["payload_index_sha256"],
        "artifact_tree_sha256": tree["artifact_tree_sha256"],
        "files": tree["files"],
        "eligible_occurrence_count": EXPECTED_OCCURRENCE_COUNT,
        "unique_image_count": EXPECTED_UNIQUE_IMAGE_COUNT,
        "selected_image_count": 6,
        "ocr_replay_record_count": len(replay_records),
        "confirm_images_used": False,
        "policy_output_used": False,
        "restoration_output_used": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("source", "artifact"))
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--git-revision")
    parser.add_argument("--source-root")
    parser.add_argument("--v2-contract")
    parser.add_argument("--v1-config")
    parser.add_argument("--source-file-manifest")
    parser.add_argument("--selection-manifest")
    parser.add_argument("--exposure-manifest")
    parser.add_argument("--backend-config")
    parser.add_argument("--model-dir")
    parser.add_argument("--wheel-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "source":
        result = validate_source_contract(
            args.source_contract,
            repository_root=args.repository_root,
        )
    else:
        required = {
            "output_dir": args.output_dir,
            "git_revision": args.git_revision,
            "source_root": args.source_root,
            "v2_contract": args.v2_contract,
            "v1_config": args.v1_config,
            "source_file_manifest": args.source_file_manifest,
            "selection_manifest": args.selection_manifest,
            "exposure_manifest": args.exposure_manifest,
            "backend_config": args.backend_config,
            "model_dir": args.model_dir,
            "wheel_dir": args.wheel_dir,
        }
        missing = sorted(key for key, value in required.items() if value is None)
        if missing:
            raise ValueError(f"artifact mode missing arguments: {missing}")
        result = validate_artifact(
            output_dir=args.output_dir,
            repository_root=args.repository_root,
            git_revision=args.git_revision,
            source_contract_path=args.source_contract,
            source_root=args.source_root,
            v2_contract_path=args.v2_contract,
            v1_config_path=args.v1_config,
            source_file_manifest_path=args.source_file_manifest,
            selection_manifest_path=args.selection_manifest,
            exposure_manifest_path=args.exposure_manifest,
            backend_config_path=args.backend_config,
            model_dir=args.model_dir,
            wheel_dir=args.wheel_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
