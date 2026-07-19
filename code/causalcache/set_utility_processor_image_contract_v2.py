"""Mixed RGB/RGBA image contract for set-utility processor OCR."""

from __future__ import annotations

import io
import re
from collections.abc import Callable, Mapping
from typing import Any

from causalcache.data.guiodyssey_restoration_v2 import (
    OCR_RECORD_KEYS,
    generate_ocr_records,
)
from causalcache.restoration_v2_text_backend import (
    PreparedImage,
    build_ocr_record,
    prepare_image_bytes,
    validate_backend_config,
)
from causalcache.set_utility_processor_substrate import ValidatedOcrBatch


PROCESSOR_IMAGE_CONTRACT_V2_ID = (
    "causalcache_set_utility_processor_image_contract_v2_png_rgb_allowlist_repair"
)
PROCESSOR_IMAGE_CONTRACT_V2_ACCEPTED_INPUTS = (
    ("PNG", "RGBA", (255, 255), False),
    ("PNG", "RGB", None, False),
)
PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS = {
    "PNG:RGB": 24,
    "PNG:RGBA": 18_768,
}
PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_TOTAL = 18_792
PROCESSOR_IMAGE_CONTRACT_V2_VIOLATION_OUTCOME = (
    "INVALID_PROCESSOR_IMAGE_CONTRACT_V2"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RESIZED_RGB_BYTE_COUNT = 256 * 256 * 3


def validate_processor_image_v2(prepared: PreparedImage) -> None:
    """Validate one decoded image against the exact v2 accepted-input union."""
    if not isinstance(prepared, PreparedImage):
        raise TypeError("prepared must be a PreparedImage")
    observed = (
        prepared.source_format,
        prepared.source_mode,
        prepared.alpha_extrema,
        prepared.exif_present,
    )
    if observed not in PROCESSOR_IMAGE_CONTRACT_V2_ACCEPTED_INPUTS:
        raise ValueError(
            f"{PROCESSOR_IMAGE_CONTRACT_V2_VIOLATION_OUTCOME}: "
            "expected exact PNG/RGBA opaque or PNG/RGB without alpha/EXIF"
        )
    if (
        type(prepared.width) is not int
        or type(prepared.height) is not int
        or prepared.width <= 0
        or prepared.height <= 0
    ):
        raise ValueError("prepared image dimensions must be positive integers")
    if (
        not isinstance(prepared.rgb_bytes_sha256, str)
        or _SHA256_PATTERN.fullmatch(prepared.rgb_bytes_sha256) is None
    ):
        raise ValueError("prepared image RGB SHA256 must be lowercase canonical")
    if (
        not isinstance(prepared.resized_rgb_bytes, bytes)
        or len(prepared.resized_rgb_bytes) != _RESIZED_RGB_BYTE_COUNT
    ):
        raise ValueError("prepared image must contain exact 256x256 row-major RGB bytes")


def prepare_processor_image_v2(image_bytes: bytes) -> PreparedImage:
    """Decode for validation without changing the caller-owned encoded bytes."""
    prepared = prepare_image_bytes(image_bytes)
    validate_processor_image_v2(prepared)
    return prepared


def decode_processor_image_v2(image_bytes: bytes) -> Any:
    """Return a validated, loaded PIL image without pixel or encoding mutation."""
    prepare_processor_image_v2(image_bytes)
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    return image


def _canonical_record_from_observed_nodes(
    record: Mapping[str, Any],
    *,
    image_bytes: bytes,
    backend_config_sha256: str,
) -> dict[str, Any]:
    if set(record) != OCR_RECORD_KEYS:
        raise ValueError("OCR record fields drifted from the pinned full schema")
    image_member_path = record.get("image_member_path")
    if not isinstance(image_member_path, str) or not image_member_path.startswith(
        "images/"
    ):
        raise ValueError("OCR image_member_path must belong to images/")
    nodes = record.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("OCR record nodes must be a JSON array")
    try:
        boxes = [node["polygon_xy"] for node in nodes]
        texts = [node["raw_text"] for node in nodes]
        scores = [float(node["confidence_decimal_string"]) for node in nodes]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("OCR record node schema is invalid") from error
    try:
        return build_ocr_record(
            image_member_path=image_member_path,
            image_bytes=image_bytes,
            backend_config_sha256=backend_config_sha256,
            boxes=boxes,
            texts=texts,
            scores=scores,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("OCR record canonical reconstruction failed") from error


def validate_processor_ocr_record_v2(
    record: Mapping[str, Any],
    *,
    image_bytes: bytes,
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
) -> PreparedImage:
    """Validate the full canonical OCR schema under the v2 image union."""
    if not isinstance(record, Mapping):
        raise TypeError("OCR record must be a mapping")
    validate_backend_config(backend_config)
    prepared = prepare_processor_image_v2(image_bytes)
    expected = _canonical_record_from_observed_nodes(
        record,
        image_bytes=image_bytes,
        backend_config_sha256=backend_config_sha256,
    )
    if dict(record) != expected:
        raise ValueError("OCR record differs from the pinned canonical reconstruction")
    return prepared


def run_processor_rapidocr_record_v2(
    *,
    engine: Any,
    backend_config: Mapping[str, Any],
    image_member_path: str,
    image_bytes: bytes,
    backend_config_sha256: str,
) -> dict[str, Any]:
    """Run frozen RapidOCR on the exact original encoded byte object."""
    validate_backend_config(backend_config)
    prepare_processor_image_v2(image_bytes)
    result = engine(
        image_bytes,
        use_det=True,
        use_cls=False,
        use_rec=True,
        return_word_box=False,
        return_single_char_box=False,
        text_score=0.5,
        box_thresh=0.5,
        unclip_ratio=1.6,
    )
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is not None and hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    record = build_ocr_record(
        image_member_path=image_member_path,
        image_bytes=image_bytes,
        backend_config_sha256=backend_config_sha256,
        boxes=boxes,
        texts=texts,
        scores=scores,
    )
    validate_processor_ocr_record_v2(
        record,
        image_bytes=image_bytes,
        backend_config=backend_config,
        backend_config_sha256=backend_config_sha256,
    )
    return record


def build_validated_ocr_batch_v2(
    image_payloads: Mapping[str, bytes],
    *,
    engine: Any,
    backend_config: Mapping[str, Any],
    backend_config_sha256: str,
    record_runner: Callable[..., Mapping[str, Any]] = (
        run_processor_rapidocr_record_v2
    ),
) -> ValidatedOcrBatch:
    """Generate and validate one exact mixed-mode OCR batch."""
    records, _ = generate_ocr_records(
        engine=engine,
        backend_config=backend_config,
        backend_config_sha256=backend_config_sha256,
        image_payloads=image_payloads,
        record_runner=record_runner,
    )
    prepared = {
        path: validate_processor_ocr_record_v2(
            record,
            image_bytes=image_payloads[path],
            backend_config=backend_config,
            backend_config_sha256=backend_config_sha256,
        )
        for path, record in records.items()
    }
    return ValidatedOcrBatch(records_by_path=records, prepared_by_path=prepared)


__all__ = [
    "PROCESSOR_IMAGE_CONTRACT_V2_ACCEPTED_INPUTS",
    "PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS",
    "PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_TOTAL",
    "PROCESSOR_IMAGE_CONTRACT_V2_ID",
    "PROCESSOR_IMAGE_CONTRACT_V2_VIOLATION_OUTCOME",
    "build_validated_ocr_batch_v2",
    "decode_processor_image_v2",
    "prepare_processor_image_v2",
    "run_processor_rapidocr_record_v2",
    "validate_processor_image_v2",
    "validate_processor_ocr_record_v2",
]
