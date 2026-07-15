"""Validate the pinned restoration-v2 OCR backend and golden records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
        choices=("config-only", "inspect-golden", "validate-golden"),
    )
    parser.add_argument("--backend-config", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--wheel-dir")
    parser.add_argument("--fixture")
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "config-only":
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
