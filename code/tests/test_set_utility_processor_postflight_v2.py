from __future__ import annotations

import io
import importlib.util
import json
import shutil
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from causalcache.restoration_v2_text_backend import (
    build_ocr_record,
    load_backend_config,
    sha256_bytes,
)
from causalcache.set_utility_processor_artifacts import (
    canonical_json_bytes,
    canonical_pretty_json_bytes,
)
from causalcache.set_utility_processor_image_contract_v2 import (
    PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS,
)
from causalcache.set_utility_processor_postflight_v2 import (
    IMAGE_CONTRACT_PATH,
    RUNNER_PATH,
    V1_RUNNER_PATH,
    ProcessorFreezePostflightContextV2,
    VALIDATION_STATUS,
    _validate_global_format_tally,
    _validate_record_without_image,
    validate_completed_processor_freeze_root_v2,
    validate_processor_only_source_v2,
)


ROOT = Path(__file__).resolve().parents[2]
BACKEND_SHA = "a" * 64
BACKEND_CONFIG_PATH = ROOT / "code/configs/restoration_v2_ocr_backend.json"


def _load_v1_postflight_fixture() -> ModuleType:
    name = "_causalcache_test_processor_postflight_v1_fixture"
    path = ROOT / "code/tests/test_set_utility_processor_postflight.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load the v1 postflight fixture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _png(mode: str) -> bytes:
    image = Image.new(mode, (5, 7), (1, 2, 3, 255) if mode == "RGBA" else (1, 2, 3))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_v2_source_audit_recursively_binds_both_runners() -> None:
    result = validate_processor_only_source_v2(ROOT)

    assert result["status"] == (
        "VALID_PROCESSOR_ONLY_SOURCE_V2_IMAGE_CONTRACT_REPAIR"
    )
    assert set(result["bound_sources"]) == {
        "image_contract_v2",
        "runner_v1",
        "runner_v2",
    }
    assert result["forbidden_image_mutation_call_count"] == 0
    assert result["forbidden_v1_execution_helper_call_count"] == 0
    assert result["transient_rgb_convert_call_count"] == 1
    assert result["bound_sources"]["runner_v1"]["path"] == V1_RUNNER_PATH
    assert result["bound_sources"]["runner_v2"]["path"] == RUNNER_PATH
    assert result["bound_sources"]["image_contract_v2"]["path"] == (
        IMAGE_CONTRACT_PATH
    )


def test_v2_source_audit_rejects_direct_runner_image_mutation(
    tmp_path: Path,
) -> None:
    for relative in (RUNNER_PATH, V1_RUNNER_PATH, IMAGE_CONTRACT_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_PATH
    runner.write_text(
        runner.read_text(encoding="utf-8") + "\nimage.save('forbidden.png')\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden image mutation"):
        validate_processor_only_source_v2(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    (
        "\nother.convert('RGB')\n",
        "\nother.convert('RGBA')\n",
        "\nother.putalpha(255)\n",
        "\nImageOps.exif_transpose(other)\n",
    ),
)
def test_v2_source_audit_rejects_extra_or_non_rgb_conversion(
    tmp_path: Path,
    mutation: str,
) -> None:
    for relative in (RUNNER_PATH, V1_RUNNER_PATH, IMAGE_CONTRACT_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_PATH
    runner.write_text(
        runner.read_text(encoding="utf-8") + mutation,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden image mutation"):
        validate_processor_only_source_v2(tmp_path)


def test_v2_source_audit_requires_literal_rgb_at_the_exact_replay_site(
    tmp_path: Path,
) -> None:
    for relative in (RUNNER_PATH, V1_RUNNER_PATH, IMAGE_CONTRACT_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_PATH
    source = runner.read_text(encoding="utf-8")
    assert source.count('source.convert("RGB")') == 1
    runner.write_text(
        source.replace('source.convert("RGB")', 'source.convert("RGBA")'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden image mutation"):
        validate_processor_only_source_v2(tmp_path)


def test_v2_source_audit_rejects_imported_image_mutation_alias(
    tmp_path: Path,
) -> None:
    for relative in (RUNNER_PATH, V1_RUNNER_PATH, IMAGE_CONTRACT_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_PATH
    runner.write_text(
        runner.read_text(encoding="utf-8")
        + "\nfrom PIL.ImageOps import exif_transpose as x\nx(object())\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden image mutation"):
        validate_processor_only_source_v2(tmp_path)


def test_v2_source_audit_rejects_v1_helper_attribute_alias(
    tmp_path: Path,
) -> None:
    for relative in (RUNNER_PATH, V1_RUNNER_PATH, IMAGE_CONTRACT_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_PATH
    runner.write_text(
        runner.read_text(encoding="utf-8") + "\nbad = _V1._run_ocr_worker\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden v1 image execution helper"):
        validate_processor_only_source_v2(tmp_path)


def test_v2_source_audit_rejects_v1_image_execution_helper_reuse(
    tmp_path: Path,
) -> None:
    for relative in (RUNNER_PATH, V1_RUNNER_PATH, IMAGE_CONTRACT_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    runner = tmp_path / RUNNER_PATH
    runner.write_text(
        runner.read_text(encoding="utf-8") + "\n_V1._run_ocr_worker(None, None, None)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden v1 image execution helper"):
        validate_processor_only_source_v2(tmp_path)


def test_global_format_tally_requires_exact_census_counts() -> None:
    assert _validate_global_format_tally(
        (
            {"PNG:RGBA": 10_000, "PNG:RGB": 4},
            {"PNG:RGBA": 8_768, "PNG:RGB": 20},
        )
    ) == dict(sorted(PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS.items()))

    with pytest.raises(ValueError, match="frozen v2 census"):
        _validate_global_format_tally(
            ({"PNG:RGBA": 18_768, "PNG:RGB": 23},)
        )
    with pytest.raises(ValueError, match="escaped the v2 contract"):
        _validate_global_format_tally(
            ({"PNG:RGBA": 18_768, "PNG:RGB": 24, "JPEG:RGB": 1},)
        )


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_record_semantics_accept_exact_v2_union(mode: str) -> None:
    payload = _png(mode)
    path = f"images/fixture/{mode.lower()}.png"
    record = build_ocr_record(
        image_member_path=path,
        image_bytes=payload,
        backend_config_sha256=BACKEND_SHA,
        boxes=None,
        texts=None,
        scores=None,
    )

    assert _validate_record_without_image(
        record,
        expected_path=path,
        backend_config_sha256=BACKEND_SHA,
    ) == ("PNG", mode)
    assert record["image_sha256"] == sha256_bytes(payload)


def test_record_semantics_rejects_canonical_sha_tamper() -> None:
    payload = _png("RGB")
    path = "images/fixture/rgb.png"
    record = build_ocr_record(
        image_member_path=path,
        image_bytes=payload,
        backend_config_sha256=BACKEND_SHA,
        boxes=None,
        texts=None,
        scores=None,
    )
    record["canonical_ocr_record_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="canonical SHA256 drifted"):
        _validate_record_without_image(
            record,
            expected_path=path,
            backend_config_sha256=BACKEND_SHA,
        )


def test_record_semantics_rejects_extra_field_even_with_recomputed_hash() -> None:
    payload = _png("RGB")
    path = "images/fixture/rgb.png"
    record = build_ocr_record(
        image_member_path=path,
        image_bytes=payload,
        backend_config_sha256=BACKEND_SHA,
        boxes=None,
        texts=None,
        scores=None,
    )
    record["evil"] = "self-hashed garbage"
    unsigned = dict(record)
    unsigned.pop("canonical_ocr_record_sha256")
    record["canonical_ocr_record_sha256"] = sha256_bytes(
        canonical_json_bytes(unsigned)
    )

    with pytest.raises(ValueError, match="fields drifted"):
        _validate_record_without_image(
            record,
            expected_path=path,
            backend_config_sha256=BACKEND_SHA,
        )


def test_v2_postflight_wraps_v1_structure_and_rebuilds_mixed_mode_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _load_v1_postflight_fixture()
    original_builder = fixture._trajectory_record

    def full_record(assignment: object):
        base = original_builder(assignment)
        queries = []
        for query in base.queries:
            all_paths = tuple(query.ocr_records_by_path)

            def payload_for(path: str) -> bytes:
                step = int(path.rsplit("-", 1)[1].removesuffix(".png"))
                return _png("RGB" if step % 2 == 0 else "RGBA")

            ocr = {
                path: build_ocr_record(
                    image_member_path=path,
                    image_bytes=payload_for(path),
                    backend_config_sha256="7" * 64,
                    boxes=None,
                    texts=None,
                    scores=None,
                )
                for path in all_paths
            }
            queries.append(
                replace(
                    query,
                    image_payloads={
                        path: payload_for(path) for path in query.image_payloads
                    },
                    ocr_records_by_path=ocr,
                )
            )
        return replace(base, queries=tuple(queries))

    fixture._trajectory_record = full_record
    root, base_context, _, worker_tallies = fixture._write_root(tmp_path)
    renamed = root.with_name(
        "causalcache-set-utility-processor-freeze-v2-image-contract-repair-"
        + "d" * 7
    )
    root.rename(renamed)
    root = renamed

    image_contract_sha = "6" * 64
    structural = replace(
        base_context,
        ocr_runtime_identity={
            **base_context.ocr_runtime_identity,
            "image_contract_sha256": image_contract_sha,
        },
    )
    for worker_index in range(4):
        path = root / "receipts" / f"ocr-worker-{worker_index:02d}.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["ocr_runtime_identity"] = structural.ocr_runtime_identity
        path.write_bytes(canonical_json_bytes(receipt))
    identity_path = root / "run-identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["runtime_cli"]["output_root"] = str(root.resolve())
    identity_path.write_bytes(canonical_json_bytes(identity))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_identity_sha256"] = sha256_bytes(canonical_json_bytes(identity))
    manifest_path.write_bytes(canonical_pretty_json_bytes(manifest))

    expected_tally: Counter[str] = Counter()
    for tally in worker_tallies:
        expected_tally.update(tally)
    import causalcache.set_utility_processor_postflight_v2 as postflight_v2

    monkeypatch.setattr(
        postflight_v2,
        "PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS",
        dict(expected_tally),
    )
    monkeypatch.setattr(
        postflight_v2,
        "PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_TOTAL",
        sum(expected_tally.values()),
    )
    context = ProcessorFreezePostflightContextV2(
        structural_context=structural,
        backend_config=load_backend_config(BACKEND_CONFIG_PATH),
        backend_config_sha256="7" * 64,
        image_contract_sha256=image_contract_sha,
    )

    result = validate_completed_processor_freeze_root_v2(root, context=context)

    assert result["status"] == VALIDATION_STATUS
    assert result["processor_image_contract_id"].endswith(
        "png_rgb_allowlist_repair"
    )
    assert result["global_format_mode_tally"] == dict(sorted(expected_tally.items()))
    assert result["stored_image_ocr_validation_count"] > 0
    assert result["metadata_only_ocr_validation_count"] == 4

    arbitrary = root.with_name(
        "causalcache-set-utility-processor-freeze-v2-image-contract-repair-formal"
    )
    root.rename(arbitrary)
    with pytest.raises(ValueError, match="exact Git-bound namespace"):
        validate_completed_processor_freeze_root_v2(arbitrary, context=context)
