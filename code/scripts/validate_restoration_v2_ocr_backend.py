"""Validate the pinned restoration-v2 OCR backend and golden records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from causalcache.restoration_v2_text_backend import (
    BACKEND_ID,
    canonical_json_bytes,
    create_rapidocr_engine,
    decode_fixture_image,
    load_backend_config,
    load_json,
    prepare_image_bytes,
    run_rapidocr_record,
    sha256_bytes,
    sha256_file,
    verify_model_files,
    verify_rapidocr_package_files,
    verify_recognizer_character_inventory,
    verify_runtime_packages,
    verify_wheel_files,
)


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
REAL_SCREEN_RUN_GIT_COMMIT = "dcc6e217b4885cef5f745d987a1ec74e57109717"
REAL_SCREEN_SOURCE_CONTRACT_SHA256 = (
    "374a38c997a1ee9a715a8cf6ce9b7ca26edc1cf56f503c2d42a97436afac16c5"
)
REAL_SCREEN_HF_REVISION = "9ebbbbbc4666e8a065f4ecb5240491c70f05e21b"
REAL_SCREEN_ARTIFACT_TREE_SHA256 = (
    "605d6396b0cde84697ff3f2407a3630d7ccdb822f55c2bdae56be82e942a7e25"
)
REAL_SCREEN_PAYLOAD_INDEX_SHA256 = (
    "5d3e061b7db176ff6d6e97939cb98c2a6b819a928cea78d35677cc54cd0b1eb5"
)
REAL_SCREEN_HF_FILES = {
    ".gitattributes": (42, "116c15b65825b94cdce725a6c380e58928b054dc861b6b7b07f5eae549a7c2e9"),
    "README.md": (464, "837b73784b25439eab058ff0c4c06ce3c24ebabb8f368a61e4e7574a39936d5a"),
    "golden/real-screen-v1/images-00000-of-00001.tar": (
        3102720,
        "9876d1e604634712374c3b53cc1e7ef40141daf415b6699941ccc55dbcc759f7",
    ),
    "golden/real-screen-v1/manifest.json": (
        145574,
        "c9652993d8a9854f0a35de596176b9c83ff127240ee21548180e20bd4df88cab",
    ),
    "golden/real-screen-v1/ocr-records-00000-of-00001.jsonl": (
        73930,
        "c646f1a182f1a6b6c43a17115f15c799e82ca283dc94844c01b8dd77b1ff2a70",
    ),
}


def _prepared_payload(image_bytes: bytes) -> dict[str, Any]:
    prepared = prepare_image_bytes(image_bytes)
    return {
        "source_format": prepared.source_format,
        "source_mode": prepared.source_mode,
        "width": prepared.width,
        "height": prepared.height,
        "exif_present": prepared.exif_present,
        "alpha_extrema": (
            list(prepared.alpha_extrema) if prepared.alpha_extrema is not None else None
        ),
        "rgb_bytes_sha256": prepared.rgb_bytes_sha256,
        "resized_rgb_256x256_sha256": prepared.resized_rgb_bytes_sha256,
    }


def _load_fixture(
    path: str | Path,
    *,
    validate_prepared: bool = True,
) -> dict[str, Any]:
    fixture = load_json(path)
    if fixture.get("schema_version") != "1.0.0":
        raise ValueError("unexpected OCR golden fixture schema_version")
    if fixture.get("backend_id") != BACKEND_ID:
        raise ValueError("OCR golden fixture backend_id mismatch")
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("OCR golden fixture cases must be a non-empty list")
    case_ids = [case.get("case_id") for case in cases if isinstance(case, Mapping)]
    if (
        len(case_ids) != len(cases)
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(set(case_ids)) != len(case_ids)
    ):
        raise ValueError("OCR golden fixture case ids must be unique strings")
    for case in cases:
        if not isinstance(case.get("run_ocr"), bool):
            raise ValueError("OCR golden fixture run_ocr must be boolean")
        image_bytes = decode_fixture_image(case)
        if validate_prepared and _prepared_payload(image_bytes) != case.get(
            "expected_prepared"
        ):
            raise ValueError(f"OCR golden prepared image drifted: {case['case_id']}")
    return fixture


def inspect_golden(
    *,
    backend_config_path: str | Path,
    model_dir: str | Path,
    wheel_dir: str | Path,
    fixture_path: str | Path,
) -> dict[str, Any]:
    config = load_backend_config(backend_config_path)
    fixture = _load_fixture(fixture_path)
    config_sha256 = sha256_file(backend_config_path)
    packages = verify_runtime_packages(config)
    package_files = verify_rapidocr_package_files(config)
    wheels = verify_wheel_files(config, wheel_dir)
    models = verify_model_files(config, model_dir)
    engine = create_rapidocr_engine(config, model_dir)
    character_inventory = verify_recognizer_character_inventory(engine, config)
    cases = []
    for case in fixture["cases"]:
        image_bytes = decode_fixture_image(case)
        payload: dict[str, Any] = {
            "case_id": case["case_id"],
            "image_sha256": sha256_bytes(image_bytes),
            "prepared": _prepared_payload(image_bytes),
            "run_ocr": case["run_ocr"],
        }
        if case["run_ocr"]:
            payload["ocr_record"] = run_rapidocr_record(
                engine=engine,
                backend_config=config,
                image_member_path=str(case["image_member_path"]),
                image_bytes=image_bytes,
                backend_config_sha256=config_sha256,
            )
        cases.append(payload)
    return {
        "schema_version": "1.0.0",
        "protocol_id": "causalcache_restoration_v2",
        "outcome": "INSPECTED_OCR_GOLDEN",
        "backend_id": BACKEND_ID,
        "backend_config_sha256": config_sha256,
        "fixture_sha256": sha256_file(fixture_path),
        "runtime_packages": packages,
        "rapidocr_package_file_sha256": package_files,
        "wheel_sha256": wheels,
        "model_sha256": models,
        "recognizer_character_inventory": character_inventory,
        "cases": cases,
    }


def validate_golden(
    *,
    backend_config_path: str | Path,
    model_dir: str | Path,
    wheel_dir: str | Path,
    fixture_path: str | Path,
) -> dict[str, Any]:
    fixture = _load_fixture(fixture_path)
    expected = fixture.get("expected_inspection")
    if not isinstance(expected, Mapping):
        raise ValueError("OCR golden fixture expected_inspection is not frozen")
    actual = inspect_golden(
        backend_config_path=backend_config_path,
        model_dir=model_dir,
        wheel_dir=wheel_dir,
        fixture_path=fixture_path,
    )
    expected_cases = expected.get("cases")
    if actual["cases"] != expected_cases:
        raise ValueError("OCR golden case output drifted")
    expected_config_sha = expected.get("backend_config_sha256")
    if actual["backend_config_sha256"] != expected_config_sha:
        raise ValueError("OCR golden backend config SHA256 drifted")
    expected_models = expected.get("model_sha256")
    if actual["model_sha256"] != expected_models:
        raise ValueError("OCR golden model identity drifted")
    expected_packages = expected.get("runtime_packages")
    if actual["runtime_packages"] != expected_packages:
        raise ValueError("OCR golden package identity drifted")
    expected_package_files = expected.get("rapidocr_package_file_sha256")
    if actual["rapidocr_package_file_sha256"] != expected_package_files:
        raise ValueError("OCR golden installed package resources drifted")
    expected_wheels = expected.get("wheel_sha256")
    if actual["wheel_sha256"] != expected_wheels:
        raise ValueError("OCR golden wheel identity drifted")
    expected_characters = expected.get("recognizer_character_inventory")
    if actual["recognizer_character_inventory"] != expected_characters:
        raise ValueError("OCR golden character inventory drifted")
    return {
        **actual,
        "outcome": "PASSED_OCR_GOLDEN_VALIDATION",
        "expected_cases_sha256": sha256_bytes(canonical_json_bytes(expected_cases)),
    }


def config_only(
    *,
    backend_config_path: str | Path,
    fixture_path: str | Path | None,
) -> dict[str, Any]:
    config = load_backend_config(backend_config_path)
    fixture_case_count = 0
    fixture_sha256 = None
    if fixture_path is not None:
        fixture = _load_fixture(fixture_path, validate_prepared=False)
        fixture_case_count = len(fixture["cases"])
        fixture_sha256 = sha256_file(fixture_path)
    return {
        "schema_version": "1.0.0",
        "protocol_id": config["protocol_id"],
        "outcome": "PASSED_OCR_CONFIG_SOURCE_VALIDATION",
        "backend_id": config["backend_id"],
        "backend_config_sha256": sha256_file(backend_config_path),
        "fixture_case_count": fixture_case_count,
        "fixture_sha256": fixture_sha256,
        "prepared_images_validated": False,
        "policy_loaded": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
    }


def validate_artifact_source(
    *,
    backend_config_path: str | Path,
    artifact_manifest_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    config = load_backend_config(backend_config_path)
    manifest = load_json(artifact_manifest_path)
    if manifest.get("schema_version") != "1.0.0":
        raise ValueError("unexpected OCR artifact manifest schema_version")
    if manifest.get("protocol_id") != "causalcache_restoration_v2":
        raise ValueError("unexpected OCR artifact manifest protocol_id")
    if manifest.get("artifact_id") != "causalcache-restoration-v2-ocr-backend-v1":
        raise ValueError("unexpected OCR artifact_id")
    if manifest.get("status") != (
        "hf_model_immutable_verified_synthetic_and_real_screen_golden_passed"
    ):
        raise ValueError("unexpected OCR artifact manifest status")
    implementation_commit = manifest.get("backend_implementation_git_commit")
    if not isinstance(implementation_commit, str) or GIT_SHA_PATTERN.fullmatch(
        implementation_commit
    ) is None:
        raise ValueError("OCR backend implementation commit must be a full SHA")

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {root}")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("OCR artifact source_files must be a non-empty list")
    observed_source: dict[str, str] = {}
    for record in source_files:
        if not isinstance(record, Mapping):
            raise TypeError("OCR artifact source file records must be objects")
        relative = record.get("path")
        expected_sha = record.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise ValueError("OCR artifact source path must be a string")
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or str(parsed) != relative or ".." in parsed.parts:
            raise ValueError("OCR artifact source path must be safe and canonical")
        if not isinstance(expected_sha, str) or SHA256_PATTERN.fullmatch(
            expected_sha
        ) is None:
            raise ValueError("OCR artifact source SHA256 must be lowercase hex")
        if relative in observed_source:
            raise ValueError("duplicate OCR artifact source path")
        path = root.joinpath(*parsed.parts)
        if not path.is_file():
            raise FileNotFoundError(f"missing OCR artifact source file: {path}")
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            raise ValueError(f"OCR artifact source SHA256 drifted: {relative}")
        observed_source[relative] = observed_sha
    expected_source_paths = {
        "Makefile",
        "code/causalcache/restoration_v2_text_backend.py",
        "code/causalcache/data/restoration_v2_real_screen.py",
        "code/configs/restoration_v2_ocr_backend.json",
        "code/requirements/restoration_v2_ocr_lock.txt",
        "code/scripts/materialize_restoration_v2_real_screen.py",
        "code/scripts/validate_restoration_v2_ocr_backend.py",
        "code/scripts/validate_restoration_v2_real_screen.py",
        "code/tests/test_restoration_v2_real_screen.py",
        "code/tests/test_restoration_v2_text_backend.py",
        "data/fixtures/restoration_v2_ocr_golden.json",
        "data/manifests/restoration_v2_real_screen_source.json",
        "data/results/restoration_v2_ocr_backend/real_screen_summary.json",
        "data/results/restoration_v2_ocr_backend/synthetic_summary.json",
    }
    if set(observed_source) != expected_source_paths:
        raise ValueError("OCR artifact source file inventory drifted")

    config_relative = "code/configs/restoration_v2_ocr_backend.json"
    if observed_source.get(config_relative) != sha256_file(backend_config_path):
        raise ValueError("OCR artifact manifest is not bound to backend config")
    synthetic = manifest.get("synthetic_golden")
    if not isinstance(synthetic, Mapping) or synthetic.get("outcome") != (
        "PASSED_SYNTHETIC_OCR_GOLDEN"
    ):
        raise ValueError("OCR artifact synthetic golden is not passed")
    summary_path = synthetic.get("summary_path")
    if observed_source.get(summary_path) != synthetic.get("summary_sha256"):
        raise ValueError("OCR artifact synthetic summary identity drifted")

    hf_artifact = manifest.get("hf_model_artifact")
    if not isinstance(hf_artifact, Mapping):
        raise ValueError("OCR HF model artifact must be an object")
    if hf_artifact.get("repo") != config["hf_model_artifact"]["repo"]:
        raise ValueError("OCR HF model repo drifted")
    if hf_artifact.get("repo_type") != "model" or hf_artifact.get("visibility") != (
        "private"
    ):
        raise ValueError("OCR HF model type or visibility drifted")
    revision = hf_artifact.get("immutable_revision")
    if not isinstance(revision, str) or GIT_SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError("OCR HF model immutable revision must be a full SHA")
    if hf_artifact.get("tag_resolved_revision") != revision:
        raise ValueError("OCR HF tag does not resolve to the immutable revision")
    uploaded_from_commit = hf_artifact.get("uploaded_from_git_commit")
    if not isinstance(uploaded_from_commit, str) or GIT_SHA_PATTERN.fullmatch(
        uploaded_from_commit
    ) is None:
        raise ValueError("OCR HF upload commit must be a full SHA")
    artifact_files = hf_artifact.get("files")
    if not isinstance(artifact_files, list) or len(artifact_files) != 6:
        raise ValueError("OCR HF artifact must contain six recorded files")
    artifact_paths = set()
    for record in artifact_files:
        if not isinstance(record, Mapping):
            raise TypeError("OCR HF file records must be objects")
        path = record.get("path")
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if not isinstance(path, str) or not path or path in artifact_paths:
            raise ValueError("OCR HF file paths must be unique strings")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("OCR HF file sizes must be positive integers")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(
            digest
        ) is None:
            raise ValueError("OCR HF file SHA256 must be lowercase hex")
        artifact_paths.add(path)
    expected_artifact_paths = {
        ".gitattributes",
        "README.md",
        "ch_PP-OCRv5_det_mobile.onnx",
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "en_PP-OCRv5_rec_mobile.onnx",
        "manifest.json",
    }
    if artifact_paths != expected_artifact_paths:
        raise ValueError("OCR HF file inventory drifted")
    artifact_files_by_path = {record["path"]: record for record in artifact_files}
    for model_role in ("detector", "recognizer", "inactive_classifier"):
        model_config = config["models"][model_role]
        filename = model_config["filename"]
        if artifact_files_by_path[filename]["sha256"] != model_config["sha256"]:
            raise ValueError(f"OCR HF {model_role} SHA256 drifted from backend config")
    redownload = hf_artifact.get("redownload")
    if not isinstance(redownload, Mapping) or redownload.get(
        "all_listed_file_hashes_verified"
    ) is not True:
        raise ValueError("OCR HF immutable re-download is not verified")
    expected_redownload_argv = (
        f"hf download {hf_artifact['repo']} --revision {revision} --repo-type model "
        f"--local-dir {redownload.get('fresh_local_dir')} --force-download --format json"
    )
    if redownload.get("argv") != expected_redownload_argv:
        raise ValueError("OCR HF immutable re-download argv drifted")
    if manifest.get("dependency_5_closed") is not True:
        raise ValueError("OCR dependency 5 must close after both goldens pass")
    real_screen = manifest.get("real_screen_golden")
    if not isinstance(real_screen, Mapping) or real_screen.get("status") != (
        "passed_policy_blind_six_image_materialization"
    ):
        raise ValueError("OCR real-screen golden passing state drifted")
    if real_screen.get("outcome") != "PASSED_REAL_SCREEN_OCR_GOLDEN":
        raise ValueError("OCR real-screen golden outcome drifted")
    if real_screen.get("confirm_images_used") is not False:
        raise ValueError("confirm images cannot enter OCR real-screen golden")
    if real_screen.get("destination") != config["golden_contract"][
        "real_screen_golden_destination"
    ]:
        raise ValueError("OCR real-screen golden destination drifted")
    if real_screen.get("repo_type") != "dataset" or real_screen.get(
        "visibility"
    ) != "private":
        raise ValueError("OCR real-screen HF dataset type or visibility drifted")
    if real_screen.get("tag") != "ocr-real-screen-golden-v1.0.0":
        raise ValueError("OCR real-screen HF tag drifted")
    if real_screen.get("immutable_revision") != REAL_SCREEN_HF_REVISION or (
        real_screen.get("tag_resolved_revision") != REAL_SCREEN_HF_REVISION
    ):
        raise ValueError("OCR real-screen HF immutable revision drifted")
    if real_screen.get("uploaded_from_git_commit") != REAL_SCREEN_RUN_GIT_COMMIT:
        raise ValueError("OCR real-screen source Git revision drifted")
    if real_screen.get("source_contract_sha256") != (
        REAL_SCREEN_SOURCE_CONTRACT_SHA256
    ):
        raise ValueError("OCR real-screen source contract identity drifted")
    if real_screen.get("artifact_tree_sha256") != (
        REAL_SCREEN_ARTIFACT_TREE_SHA256
    ):
        raise ValueError("OCR real-screen artifact tree identity drifted")
    if real_screen.get("payload_index_sha256") != (
        REAL_SCREEN_PAYLOAD_INDEX_SHA256
    ):
        raise ValueError("OCR real-screen payload index identity drifted")
    files = real_screen.get("files")
    if not isinstance(files, list) or len(files) != len(REAL_SCREEN_HF_FILES):
        raise ValueError("OCR real-screen HF file inventory drifted")
    files_by_path = {}
    for record in files:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ValueError("OCR real-screen HF file record drifted")
        path = record["path"]
        if path in files_by_path:
            raise ValueError("OCR real-screen HF file paths must be unique")
        files_by_path[path] = record
    if set(files_by_path) != set(REAL_SCREEN_HF_FILES):
        raise ValueError("OCR real-screen HF file inventory drifted")
    for path, (size_bytes, digest) in REAL_SCREEN_HF_FILES.items():
        if files_by_path[path].get("size_bytes") != size_bytes or files_by_path[
            path
        ].get("sha256") != digest:
            raise ValueError(f"OCR real-screen HF file identity drifted: {path}")
    redownload = real_screen.get("redownload")
    if not isinstance(redownload, Mapping) or redownload.get(
        "all_listed_file_hashes_verified"
    ) is not True or redownload.get("artifact_validator_replayed_on_hyper00") is not True:
        raise ValueError("OCR real-screen immutable re-download is not verified")
    if redownload.get("revision") != REAL_SCREEN_HF_REVISION:
        raise ValueError("OCR real-screen re-download revision drifted")
    summary_path = real_screen.get("summary_path")
    if observed_source.get(summary_path) != real_screen.get("summary_sha256"):
        raise ValueError("OCR real-screen summary identity drifted")
    summary = load_json(root.joinpath(*PurePosixPath(summary_path).parts))
    if summary.get("outcome") != "PASSED_REAL_SCREEN_OCR_GOLDEN" or summary.get(
        "dependency_5_closed"
    ) is not True:
        raise ValueError("OCR real-screen summary outcome drifted")
    if summary.get("source", {}).get("run_git_commit") != REAL_SCREEN_RUN_GIT_COMMIT:
        raise ValueError("OCR real-screen summary Git revision drifted")
    if summary.get("source", {}).get("source_contract_sha256") != (
        REAL_SCREEN_SOURCE_CONTRACT_SHA256
    ):
        raise ValueError("OCR real-screen summary source contract drifted")
    summary_hf = summary.get("hf_dataset_artifact")
    if not isinstance(summary_hf, Mapping) or summary_hf.get(
        "immutable_revision"
    ) != REAL_SCREEN_HF_REVISION:
        raise ValueError("OCR real-screen summary HF revision drifted")
    if summary_hf.get("artifact_tree_sha256") != REAL_SCREEN_ARTIFACT_TREE_SHA256:
        raise ValueError("OCR real-screen summary artifact tree drifted")
    summary_files = summary_hf.get("files")
    if not isinstance(summary_files, list) or len(summary_files) != len(
        REAL_SCREEN_HF_FILES
    ):
        raise ValueError("OCR real-screen summary HF file inventory drifted")
    summary_files_by_path = {
        record.get("path"): record
        for record in summary_files
        if isinstance(record, Mapping)
    }
    if set(summary_files_by_path) != set(REAL_SCREEN_HF_FILES):
        raise ValueError("OCR real-screen summary HF file inventory drifted")
    for path, (size_bytes, digest) in REAL_SCREEN_HF_FILES.items():
        if summary_files_by_path[path].get("size_bytes") != size_bytes or (
            summary_files_by_path[path].get("sha256") != digest
        ):
            raise ValueError(f"OCR real-screen summary HF file drifted: {path}")
    if real_screen.get("independent_materialization_repeat_count") != 2:
        raise ValueError("OCR real-screen materialization repeat count drifted")
    if real_screen.get("artifact_validation_count") != 3:
        raise ValueError("OCR real-screen artifact validation count drifted")
    materializations = summary.get("materialization_repeats")
    if not isinstance(materializations, list) or len(materializations) != 2:
        raise ValueError("OCR real-screen materialization evidence drifted")
    expected_materialization_paths = [
        "/data/tmp/restoration-v2-real-screen-dcc6e217-repeat-1",
        "/data/tmp/restoration-v2-real-screen-dcc6e217-repeat-2",
    ]
    if [record.get("repeat") for record in materializations] != [1, 2]:
        raise ValueError("OCR real-screen materialization repeat IDs drifted")
    for record, output_path in zip(
        materializations,
        expected_materialization_paths,
        strict=True,
    ):
        if record.get("outcome") != (
            "MATERIALIZED_POLICY_BLIND_REAL_SCREEN_GOLDEN"
        ) or record.get("artifact_tree_sha256") != (
            REAL_SCREEN_ARTIFACT_TREE_SHA256
        ):
            raise ValueError("OCR real-screen materialization outcome drifted")
        if record.get("output_path") != output_path:
            raise ValueError("OCR real-screen materialization output path drifted")
        argv = record.get("argv")
        if not isinstance(argv, str) or (
            f"--git-revision {REAL_SCREEN_RUN_GIT_COMMIT}" not in argv
            or f"--output-dir {output_path}" not in argv
        ):
            raise ValueError("OCR real-screen materialization argv drifted")
        started = record.get("started_at_utc")
        ended = record.get("ended_at_utc")
        if not isinstance(started, str) or not isinstance(ended, str) or not (
            started < ended
        ):
            raise ValueError("OCR real-screen materialization UTC bracket drifted")
    if materializations[0]["ended_at_utc"] >= materializations[1][
        "started_at_utc"
    ]:
        raise ValueError("OCR real-screen materialization order drifted")
    validations = summary.get("artifact_validations")
    expected_validation_paths = {
        "repeat-1-independent-replay": expected_materialization_paths[0],
        "repeat-2-independent-replay": expected_materialization_paths[1],
        "hf-immutable-redownload-independent-replay": (
            "/data/tmp/restoration-v2-real-screen-dcc6e217-hf-redownload"
        ),
    }
    if not isinstance(validations, list) or len(validations) != 3:
        raise ValueError("OCR real-screen validation evidence drifted")
    if [record.get("validation_id") for record in validations] != list(
        expected_validation_paths
    ):
        raise ValueError("OCR real-screen validation IDs drifted")
    previous_end = None
    for record in validations:
        validation_id = record["validation_id"]
        output_path = expected_validation_paths[validation_id]
        if record.get("outcome") != "PASSED_REAL_SCREEN_ARTIFACT_VALIDATION":
            raise ValueError("OCR real-screen validation outcome drifted")
        if record.get("ocr_replay_record_count") != 6 or record.get(
            "artifact_tree_sha256"
        ) != REAL_SCREEN_ARTIFACT_TREE_SHA256:
            raise ValueError("OCR real-screen replay evidence drifted")
        argv = record.get("argv")
        if not isinstance(argv, str) or (
            f"--git-revision {REAL_SCREEN_RUN_GIT_COMMIT}" not in argv
            or f"--output-dir {output_path}" not in argv
        ):
            raise ValueError("OCR real-screen validation argv drifted")
        started = record.get("started_at_utc")
        ended = record.get("ended_at_utc")
        if not isinstance(started, str) or not isinstance(ended, str) or not (
            started < ended
        ):
            raise ValueError("OCR real-screen validation UTC bracket drifted")
        if previous_end is not None and previous_end >= started:
            raise ValueError("OCR real-screen validation order drifted")
        previous_end = ended
    if summary.get("negative_declarations") != {
        "confirm_images_used": False,
        "policy_loaded": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
    }:
        raise ValueError("OCR real-screen summary negative declarations drifted")
    if manifest.get("policy_output_generated_before_manifest") is not False:
        raise ValueError("OCR artifact manifest must precede policy output")
    if manifest.get("restoration_output_generated_before_manifest") is not False:
        raise ValueError("OCR artifact manifest must precede restoration output")
    return {
        "schema_version": "1.0.0",
        "protocol_id": "causalcache_restoration_v2",
        "outcome": "PASSED_OCR_ARTIFACT_SOURCE_VALIDATION",
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
        "source_file_count": len(observed_source),
        "hf_model_repo": hf_artifact["repo"],
        "hf_model_immutable_revision": revision,
        "hf_file_count": len(artifact_files),
        "hf_dataset_file_count": len(files),
        "hf_dataset_immutable_revision": REAL_SCREEN_HF_REVISION,
        "model_hashes_bound_to_backend_config": True,
        "synthetic_golden_passed": True,
        "real_screen_golden_passed": True,
        "dependency_5_closed": True,
        "policy_output_generated_by_this_validation": False,
    }


def _write_exclusive(path: str | Path, value: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with output.open("xb") as target:
        target.write(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "config-only",
            "inspect-golden",
            "validate-golden",
            "artifact-source",
        ),
    )
    parser.add_argument("--backend-config", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--wheel-dir")
    parser.add_argument("--fixture")
    parser.add_argument("--artifact-manifest")
    parser.add_argument("--repository-root")
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "artifact-source":
        if args.artifact_manifest is None or args.repository_root is None:
            raise ValueError(
                "artifact-source mode requires --artifact-manifest and --repository-root"
            )
        result = validate_artifact_source(
            backend_config_path=args.backend_config,
            artifact_manifest_path=args.artifact_manifest,
            repository_root=args.repository_root,
        )
    elif args.mode == "config-only":
        result = config_only(
            backend_config_path=args.backend_config,
            fixture_path=args.fixture,
        )
    else:
        if args.model_dir is None or args.wheel_dir is None or args.fixture is None:
            raise ValueError(
                "golden modes require --model-dir, --wheel-dir, and --fixture"
            )
        if args.mode == "inspect-golden":
            result = inspect_golden(
                backend_config_path=args.backend_config,
                model_dir=args.model_dir,
                wheel_dir=args.wheel_dir,
                fixture_path=args.fixture,
            )
        else:
            result = validate_golden(
                backend_config_path=args.backend_config,
                model_dir=args.model_dir,
                wheel_dir=args.wheel_dir,
                fixture_path=args.fixture,
            )
    if args.output is not None:
        _write_exclusive(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
