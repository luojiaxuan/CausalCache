from __future__ import annotations

import copy
import io
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from causalcache.data.guiodyssey_restoration_v2 import validate_ocr_record
from causalcache.restoration_v2_text_backend import (
    load_backend_config,
    run_rapidocr_record,
    sha256_bytes,
)
from causalcache.set_utility_processor_image_contract_v2 import (
    PROCESSOR_IMAGE_CONTRACT_V2_ACCEPTED_INPUTS,
    PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS,
    PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_TOTAL,
    PROCESSOR_IMAGE_CONTRACT_V2_ID,
    build_validated_ocr_batch_v2,
    decode_processor_image_v2,
    prepare_processor_image_v2,
    run_processor_rapidocr_record_v2,
    validate_processor_image_v2,
    validate_processor_ocr_record_v2,
)


ROOT = Path(__file__).resolve().parents[2]
BACKEND_CONFIG_PATH = ROOT / "code/configs/restoration_v2_ocr_backend.json"
BACKEND_CONFIG_SHA256 = sha256_bytes(BACKEND_CONFIG_PATH.read_bytes())


def _png(mode: str, *, alpha: int = 255) -> bytes:
    color = (11, 22, 33, alpha) if mode == "RGBA" else (11, 22, 33)
    image = Image.new(mode, (7, 9), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _RecordingEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, dict[str, object]]] = []

    def __call__(self, payload: bytes, **kwargs: object) -> SimpleNamespace:
        self.calls.append((payload, kwargs))
        return SimpleNamespace(
            boxes=[[[0, 0], [6, 0], [6, 8], [0, 8]]],
            txts=["Alpha Beta"],
            scores=[0.75],
        )


def _run(image_bytes: bytes, *, path: str = "images/fixture/state.png"):
    engine = _RecordingEngine()
    backend = load_backend_config(BACKEND_CONFIG_PATH)
    record = run_processor_rapidocr_record_v2(
        engine=engine,
        backend_config=backend,
        image_member_path=path,
        image_bytes=image_bytes,
        backend_config_sha256=BACKEND_CONFIG_SHA256,
    )
    return engine, backend, record


def test_exact_png_rgba_and_rgb_allowlist_is_accepted() -> None:
    assert PROCESSOR_IMAGE_CONTRACT_V2_ID == (
        "causalcache_set_utility_processor_image_contract_v2_png_rgb_allowlist_repair"
    )
    assert PROCESSOR_IMAGE_CONTRACT_V2_ACCEPTED_INPUTS == (
        ("PNG", "RGBA", (255, 255), False),
        ("PNG", "RGB", None, False),
    )
    assert PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS == {
        "PNG:RGB": 24,
        "PNG:RGBA": 18_768,
    }
    assert PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_TOTAL == 18_792
    assert sum(PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS.values()) == (
        PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_TOTAL
    )

    rgba = prepare_processor_image_v2(_png("RGBA"))
    rgb = prepare_processor_image_v2(_png("RGB"))
    assert (rgba.source_mode, rgba.alpha_extrema) == ("RGBA", (255, 255))
    assert (rgb.source_mode, rgb.alpha_extrema) == ("RGB", None)

    decoded_rgba = decode_processor_image_v2(_png("RGBA"))
    decoded_rgb = decode_processor_image_v2(_png("RGB"))
    assert (decoded_rgba.mode, decoded_rgba.size) == ("RGBA", (7, 9))
    assert (decoded_rgb.mode, decoded_rgb.size) == ("RGB", (7, 9))


def test_neighboring_image_contracts_are_rejected_fail_closed() -> None:
    rgba = prepare_processor_image_v2(_png("RGBA"))
    rgb = prepare_processor_image_v2(_png("RGB"))
    mutations = (
        replace(rgba, source_format="JPEG"),
        replace(rgba, alpha_extrema=(254, 255)),
        replace(rgba, alpha_extrema=None),
        replace(rgba, exif_present=True),
        replace(rgb, source_mode="P"),
        replace(rgb, alpha_extrema=(255, 255)),
        replace(rgb, exif_present=True),
    )
    for mutated in mutations:
        with pytest.raises(ValueError, match="INVALID_PROCESSOR_IMAGE_CONTRACT_V2"):
            validate_processor_image_v2(mutated)

    with pytest.raises(ValueError, match="INVALID_PROCESSOR_IMAGE_CONTRACT_V2"):
        prepare_processor_image_v2(_png("RGBA", alpha=254))


@pytest.mark.parametrize("mode", ["RGBA", "RGB"])
def test_engine_and_record_receive_the_original_encoded_bytes(mode: str) -> None:
    image_bytes = _png(mode)
    engine, _, record = _run(image_bytes)

    assert len(engine.calls) == 1
    assert engine.calls[0][0] is image_bytes
    assert engine.calls[0][1] == {
        "use_det": True,
        "use_cls": False,
        "use_rec": True,
        "return_word_box": False,
        "return_single_char_box": False,
        "text_score": 0.5,
        "box_thresh": 0.5,
        "unclip_ratio": 1.6,
    }
    assert record["image_sha256"] == sha256_bytes(image_bytes)


def test_rgba_v2_record_is_identical_to_v1_canonical_record() -> None:
    image_bytes = _png("RGBA")
    backend = load_backend_config(BACKEND_CONFIG_PATH)
    v1_engine = _RecordingEngine()
    v2_engine = _RecordingEngine()

    v1_record = run_rapidocr_record(
        engine=v1_engine,
        backend_config=backend,
        image_member_path="images/fixture/state.png",
        image_bytes=image_bytes,
        backend_config_sha256=BACKEND_CONFIG_SHA256,
    )
    v2_record = run_processor_rapidocr_record_v2(
        engine=v2_engine,
        backend_config=backend,
        image_member_path="images/fixture/state.png",
        image_bytes=image_bytes,
        backend_config_sha256=BACKEND_CONFIG_SHA256,
    )

    assert v2_record == v1_record
    assert validate_ocr_record(
        v2_record,
        image_bytes=image_bytes,
        backend_config=backend,
        backend_config_sha256=BACKEND_CONFIG_SHA256,
    ).source_mode == "RGBA"


def test_rgb_record_and_encoded_byte_sha_are_stable_under_v2_validation() -> None:
    image_bytes = _png("RGB")
    _, backend, record = _run(image_bytes)
    expected_image_sha = sha256_bytes(image_bytes)
    expected_record_sha = record["canonical_ocr_record_sha256"]

    prepared = validate_processor_ocr_record_v2(
        record,
        image_bytes=image_bytes,
        backend_config=backend,
        backend_config_sha256=BACKEND_CONFIG_SHA256,
    )
    assert prepared.source_mode == "RGB"
    assert record["image_sha256"] == expected_image_sha
    assert record["canonical_ocr_record_sha256"] == expected_record_sha

    with pytest.raises(ValueError, match="INVALID_DERIVED_ARTIFACT"):
        validate_ocr_record(
            record,
            image_bytes=image_bytes,
            backend_config=backend,
            backend_config_sha256=BACKEND_CONFIG_SHA256,
        )

    mutated = copy.deepcopy(record)
    mutated["full_spatial_tokens"].append("drift")
    with pytest.raises(ValueError, match="pinned canonical reconstruction"):
        validate_processor_ocr_record_v2(
            mutated,
            image_bytes=image_bytes,
            backend_config=backend,
            backend_config_sha256=BACKEND_CONFIG_SHA256,
        )


def test_mixed_mode_batch_uses_v2_validator_for_every_record() -> None:
    rgba = _png("RGBA")
    rgb = _png("RGB")
    payloads = {
        "images/fixture/z-rgb.png": rgb,
        "images/fixture/a-rgba.png": rgba,
    }
    engine = _RecordingEngine()
    backend = load_backend_config(BACKEND_CONFIG_PATH)

    batch = build_validated_ocr_batch_v2(
        payloads,
        engine=engine,
        backend_config=backend,
        backend_config_sha256=BACKEND_CONFIG_SHA256,
    )

    assert tuple(batch.records_by_path) == tuple(sorted(payloads))
    assert batch.prepared_by_path["images/fixture/a-rgba.png"].source_mode == "RGBA"
    assert batch.prepared_by_path["images/fixture/z-rgb.png"].source_mode == "RGB"
    assert [call[0] for call in engine.calls] == [
        payloads[path] for path in sorted(payloads)
    ]
