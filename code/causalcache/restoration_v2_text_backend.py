"""Pinned OCR and image preprocessing for restoration-v2 artifacts."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from importlib.util import find_spec
from importlib.metadata import PackageNotFoundError, version as package_version
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from causalcache.low_fidelity_v2 import normalize_screen_text


BACKEND_CONFIG_SCHEMA_VERSION = "1.0.0"
OCR_RECORD_SCHEMA_VERSION = "1.0.0"
BACKEND_ID = "rapidocr-3.8.4-ppocrv5-mobile-en-cpu-v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CONFIDENCE_DECIMAL_PLACES = 8
RESIZED_IMAGE_SIZE = (256, 256)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON document must contain an object")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def validate_backend_config(config: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "backend_id",
        "scientific_contract_sha256",
        "source",
        "upstream_package_files",
        "wheels",
        "models",
        "hf_model_artifact",
        "runtime_packages",
        "inference",
        "image_preprocessing",
        "ocr_input",
        "selected_guiodyssey_input_contract",
        "canonical_output",
        "golden_contract",
    }
    if set(config) != expected_keys:
        raise ValueError("OCR backend config top-level fields drifted")
    if config.get("schema_version") != BACKEND_CONFIG_SCHEMA_VERSION:
        raise ValueError("unexpected OCR backend config schema_version")
    if config.get("protocol_id") != "causalcache_restoration_v2":
        raise ValueError("unexpected OCR backend protocol_id")
    if config.get("backend_id") != BACKEND_ID:
        raise ValueError("unexpected OCR backend_id")
    if config.get("scientific_contract_sha256") != (
        "9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc"
    ):
        raise ValueError("OCR backend is not bound to the frozen scientific contract")

    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("source must be an object")
    expected_source = {
        "repo": "RapidAI/RapidOCR",
        "tag": "v3.8.4",
        "commit": "86b48d2c96818f4dd7897627b04cdaf2c1dc1172",
        "license": "Apache-2.0",
    }
    if dict(source) != expected_source:
        raise ValueError("RapidOCR source identity drifted")

    expected_upstream_package_files = {
        "config_yaml_sha256": (
            "253abd97e27627c00925b06c33cdc8c63fd56093f9a9a80c9a258bf1a2621d18"
        ),
        "default_models_yaml_sha256": (
            "0d734a5024997bdbbd84d275c633285d6e0084a2514a934825fd59772c6464ab"
        ),
        "load_image_py_sha256": (
            "f6de151fdce9fdf6890650dc3f7305455a2cade9bbf7652a30a558ee3fbe3f59"
        ),
        "onnxruntime_main_py_sha256": (
            "f62a6aaaf7edb0b71eba65bdd5e2ae2fed6194643c44382618eaeb757827e775"
        ),
        "recognizer_main_py_sha256": (
            "7e45ae163ae05973724943749a83b9c2ff3f8e37c838a6a3d253043f58ea5b58"
        ),
    }
    if config.get("upstream_package_files") != expected_upstream_package_files:
        raise ValueError("RapidOCR packaged configuration identity drifted")

    wheels = config.get("wheels")
    expected_wheels = {
        "rapidocr": {
            "version": "3.8.4",
            "filename": "rapidocr-3.8.4-py3-none-any.whl",
            "sha256": "1a8400df99ea2348a6d1902d1c6fa64c29edcf56b3c58cecd4cf1e5ff51b7ae2",
        },
        "onnxruntime": {
            "version": "1.24.4",
            "filename": "onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "sha256": "0d640eb9f3782689b55cfa715094474cd5662f2f137be6a6f847a594b6e9705c",
        },
    }
    if wheels != expected_wheels:
        raise ValueError("OCR wheel identity drifted")

    models = config.get("models")
    if not isinstance(models, Mapping) or set(models) != {
        "detector",
        "recognizer",
        "inactive_classifier",
    }:
        raise ValueError("OCR model inventory drifted")
    expected_models = {
        "detector": {
            "filename": "ch_PP-OCRv5_det_mobile.onnx",
            "sha256": "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
            "upstream_url": (
                "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/"
                "onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx"
            ),
        },
        "recognizer": {
            "filename": "en_PP-OCRv5_rec_mobile.onnx",
            "sha256": "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8",
            "upstream_url": (
                "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/"
                "onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx"
            ),
            "embedded_character_metadata_key": "character",
            "embedded_character_entry_count": 436,
            "embedded_character_utf8_sha256": (
                "e025a66d31f327ba0c232e03f407ae8d105e1e709e7ccb3f408aa778c24e70d6"
            ),
            "embedded_character_canonical_json_sha256": (
                "16d149d69e1f4202e7d5a703ca26edcc04070888cd59c2353bf4cc70406ab611"
            ),
        },
        "inactive_classifier": {
            "filename": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            "sha256": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
            "source": (
                "embedded_in_rapidocr_wheel_but_loaded_even_when_Global.use_cls_false"
            ),
        },
    }
    if models != expected_models:
        raise ValueError("OCR model identity drifted")
    for role, expected_record in expected_models.items():
        record = models[role]
        _require_sha256(record.get("sha256"), f"models.{role}.sha256")

    expected_hf_model_artifact = {
        "repo": "gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en",
        "repo_type": "model",
        "visibility": "private",
        "immutable_revision_source": "data/manifests/restoration_v2_ocr_backend.json",
        "required_before_policy_output": True,
    }
    if config.get("hf_model_artifact") != expected_hf_model_artifact:
        raise ValueError("OCR model artifact pre-evidence identity drifted")

    inference = config.get("inference")
    expected_inference = {
        "provider": "CPUExecutionProvider",
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "enable_cpu_mem_arena": False,
        "use_cuda": False,
        "use_dml": False,
        "use_cann": False,
        "use_coreml": False,
        "text_score": 0.5,
        "use_det": True,
        "use_cls": False,
        "use_rec": True,
        "return_word_box": False,
        "return_single_char_box": False,
        "min_height": 30,
        "width_height_ratio": 8,
        "max_side_len": 2000,
        "min_side_len": 30,
        "det_limit_side_len": 736,
        "det_limit_type": "min",
        "det_thresh": 0.3,
        "det_box_thresh": 0.5,
        "det_max_candidates": 1000,
        "det_unclip_ratio": 1.6,
        "det_use_dilation": True,
        "det_score_mode": "fast",
        "rec_batch_num": 6,
        "rec_img_shape": [3, 48, 320],
        "additional_confidence_filter": False,
    }
    if inference != expected_inference:
        raise ValueError("OCR inference parameters drifted")

    image = config.get("image_preprocessing")
    expected_image = {
        "decoder": "Pillow.Image.open(BytesIO(raw));load",
        "pillow_version": "12.2.0",
        "exif_transpose": False,
        "color_conversion": "convert_RGB",
        "resize": [256, 256],
        "resample": "PIL.Image.Resampling.BILINEAR",
        "resample_integer": 2,
        "box": None,
        "reducing_gap": None,
        "output": "row_major_uint8_RGB_bytes",
    }
    if image != expected_image:
        raise ValueError("image preprocessing identity drifted")

    expected_ocr_input = {
        "payload": "original_encoded_image_bytes",
        "pre_resize": False,
        "shares_resized_rgb_with_mad_or_rgb_baseline": False,
        "loader": "rapidocr.utils.load_image.LoadImage",
        "bytes_decode": (
            "PIL.Image.open(BytesIO(raw))_to_numpy_without_exif_transpose"
        ),
        "rgba_conversion": (
            "opaque_RGBA_float32_alpha_composite_then_cv2_COLOR_RGB2BGR"
        ),
        "load_image_source_sha256": (
            "f6de151fdce9fdf6890650dc3f7305455a2cade9bbf7652a30a558ee3fbe3f59"
        ),
    }
    if config.get("ocr_input") != expected_ocr_input:
        raise ValueError("OCR input transport drifted")

    output = config.get("canonical_output")
    expected_output = {
        "record_schema_version": OCR_RECORD_SCHEMA_VERSION,
        "polygon_coordinate_rounding": "nonnegative_floor_x_plus_0.5_then_clip_to_image_extent",
        "bbox_order": ["top", "left", "bottom", "right"],
        "node_order": [
            "top",
            "left",
            "bottom",
            "right",
            "normalized_text",
            "raw_text",
            "confidence_decimal_string",
            "polygon_xy",
        ],
        "text_normalization": "Unicode_NFKC_collapse_whitespace_strip",
        "tokenization": "normalized_node_text_split_on_ASCII_space",
        "confidence_decimal_places": CONFIDENCE_DECIMAL_PLACES,
        "full_spatial_tokens_are_uncapped": True,
        "strong_summary_delta_cap_per_side": 32,
        "record_hash_excludes_only_canonical_ocr_record_sha256_field": True,
    }
    if output != expected_output:
        raise ValueError("OCR canonical output contract drifted")

    expected_packages = {
        "antlr4-python3-runtime": "4.9.3",
        "certifi": "2026.6.17",
        "charset-normalizer": "3.4.9",
        "colorlog": "6.10.1",
        "flatbuffers": "25.12.19",
        "idna": "3.18",
        "mpmath": "1.3.0",
        "numpy": "2.3.5",
        "omegaconf": "2.3.1",
        "onnxruntime": "1.24.4",
        "opencv-python": "4.10.0.84",
        "packaging": "26.2",
        "Pillow": "12.2.0",
        "protobuf": "7.35.1",
        "pyarrow": "24.0.0",
        "pyclipper": "1.4.0",
        "PyYAML": "6.0.3",
        "rapidocr": "3.8.4",
        "requests": "2.34.2",
        "Shapely": "2.1.2",
        "six": "1.17.0",
        "sympy": "1.14.0",
        "tqdm": "4.68.4",
        "urllib3": "2.7.0",
    }
    if config.get("runtime_packages") != expected_packages:
        raise ValueError("OCR runtime package identity drifted")

    expected_selected_input = {
        "source_format": "PNG",
        "source_mode": "RGBA",
        "alpha_extrema": [255, 255],
        "exif_present": False,
        "violation_outcome": "INVALID_DERIVED_ARTIFACT_BEFORE_POLICY_OUTPUT",
    }
    if config.get("selected_guiodyssey_input_contract") != expected_selected_input:
        raise ValueError("selected GUIOdyssey image contract drifted")

    expected_golden = {
        "synthetic_fixture_required": True,
        "independent_process_repeats": 2,
        "canonical_records_must_be_byte_identical": True,
        "real_screen_golden_destination": (
            "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile"
        ),
        "confirm_images_may_be_used_for_golden_selection": False,
        "real_screen_selection": {
            "eligible_state_roles": ["v2_label_train", "v2_development"],
            "eligible_image_roles": ["candidate_post_state", "current_observation"],
            "deduplicate_by": "image_sha256",
            "orientation_strata": ["portrait", "landscape"],
            "count_per_stratum": 3,
            "within_stratum_order": ["image_sha256", "image_member_path"],
            "insufficient_stratum_outcome": (
                "INVALID_DERIVED_ARTIFACT_BEFORE_POLICY_OUTPUT"
            ),
        },
    }
    if config.get("golden_contract") != expected_golden:
        raise ValueError("OCR golden contract drifted")


def load_backend_config(path: str | Path) -> dict[str, Any]:
    config = load_json(path)
    validate_backend_config(config)
    return config


def verify_runtime_packages(config: Mapping[str, Any]) -> dict[str, str]:
    expected = config["runtime_packages"]
    observed: dict[str, str] = {}
    for distribution, expected_version in expected.items():
        try:
            observed_version = package_version(distribution)
        except PackageNotFoundError as error:
            raise RuntimeError(f"required OCR package is missing: {distribution}") from error
        if observed_version != expected_version:
            raise RuntimeError(
                f"OCR package version mismatch for {distribution}: "
                f"expected {expected_version}, observed {observed_version}"
            )
        observed[str(distribution)] = observed_version
    return observed


def verify_model_files(config: Mapping[str, Any], model_dir: str | Path) -> dict[str, str]:
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"OCR model directory does not exist: {root}")
    observed: dict[str, str] = {}
    for role, record in config["models"].items():
        path = root / str(record["filename"])
        if not path.is_file():
            raise FileNotFoundError(f"missing OCR {role} model: {path}")
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise ValueError(f"OCR {role} model SHA256 mismatch")
        observed[str(role)] = digest
    return observed


def verify_wheel_files(config: Mapping[str, Any], wheel_dir: str | Path) -> dict[str, str]:
    root = Path(wheel_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"OCR wheel directory does not exist: {root}")
    observed: dict[str, str] = {}
    for distribution, record in config["wheels"].items():
        path = root / str(record["filename"])
        if not path.is_file():
            raise FileNotFoundError(f"missing OCR {distribution} wheel: {path}")
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise ValueError(f"OCR {distribution} wheel SHA256 mismatch")
        observed[str(distribution)] = digest
    return observed


def verify_rapidocr_package_files(config: Mapping[str, Any]) -> dict[str, str]:
    spec = find_spec("rapidocr")
    if spec is None or spec.origin is None:
        raise RuntimeError("installed rapidocr package cannot be located")
    root = Path(spec.origin).resolve().parent
    expected = config["upstream_package_files"]
    files = {
        "config_yaml_sha256": root / "config.yaml",
        "default_models_yaml_sha256": root / "default_models.yaml",
        "load_image_py_sha256": root / "utils" / "load_image.py",
        "onnxruntime_main_py_sha256": (
            root / "inference_engine" / "onnxruntime" / "main.py"
        ),
        "recognizer_main_py_sha256": root / "ch_ppocr_rec" / "main.py",
    }
    observed: dict[str, str] = {}
    for field, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing installed RapidOCR resource: {path}")
        digest = sha256_file(path)
        if digest != expected[field]:
            raise ValueError(f"installed RapidOCR resource SHA256 drifted: {path.name}")
        observed[path.name] = digest
    return observed


def verify_recognizer_character_inventory(
    engine: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected = config["models"]["recognizer"]
    session = engine.text_rec.session
    metadata = session.session.get_modelmeta().custom_metadata_map
    key = expected["embedded_character_metadata_key"]
    raw = metadata.get(key)
    if not isinstance(raw, str):
        raise ValueError("recognizer ONNX embedded character metadata is missing")
    raw_characters = session.get_character_list()
    if not isinstance(raw_characters, Sequence) or isinstance(
        raw_characters, (str, bytes)
    ):
        raise TypeError("recognizer character inventory must be a sequence")
    characters = list(raw_characters)
    if any(
        not isinstance(character, str) for character in characters
    ):
        raise TypeError("recognizer character inventory must be a list of strings")
    observed = {
        "metadata_key": key,
        "entry_count": len(characters),
        "utf8_sha256": sha256_bytes(raw.encode("utf-8")),
        "canonical_json_sha256": sha256_bytes(canonical_json_bytes(characters)),
    }
    required = {
        "metadata_key": key,
        "entry_count": expected["embedded_character_entry_count"],
        "utf8_sha256": expected["embedded_character_utf8_sha256"],
        "canonical_json_sha256": expected[
            "embedded_character_canonical_json_sha256"
        ],
    }
    if observed != required:
        raise ValueError("recognizer embedded character inventory drifted")
    return observed


@dataclass(frozen=True)
class PreparedImage:
    source_format: str
    source_mode: str
    width: int
    height: int
    exif_present: bool
    alpha_extrema: tuple[int, int] | None
    rgb_bytes_sha256: str
    resized_rgb_bytes: bytes

    @property
    def resized_rgb_bytes_sha256(self) -> str:
        return sha256_bytes(self.resized_rgb_bytes)


def prepare_image_bytes(image_bytes: bytes) -> PreparedImage:
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise TypeError("image_bytes must be non-empty bytes")
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        source_format = str(image.format or "unknown")
        source_mode = image.mode
        width, height = image.size
        exif_present = bool(image.getexif())
        alpha_extrema: tuple[int, int] | None = None
        if "A" in image.getbands():
            extrema = image.getchannel("A").getextrema()
            alpha_extrema = (int(extrema[0]), int(extrema[1]))
        rgb = image.convert("RGB")
        rgb_bytes = rgb.tobytes()
        resized = rgb.resize(RESIZED_IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
        resized_rgb_bytes = resized.tobytes()
    if width <= 0 or height <= 0:
        raise ValueError("decoded image dimensions must be positive")
    if len(resized_rgb_bytes) != RESIZED_IMAGE_SIZE[0] * RESIZED_IMAGE_SIZE[1] * 3:
        raise RuntimeError("resized image did not produce exact row-major RGB bytes")
    return PreparedImage(
        source_format=source_format,
        source_mode=source_mode,
        width=width,
        height=height,
        exif_present=exif_present,
        alpha_extrema=alpha_extrema,
        rgb_bytes_sha256=sha256_bytes(rgb_bytes),
        resized_rgb_bytes=resized_rgb_bytes,
    )


def validate_selected_guiodyssey_image(
    prepared: PreparedImage,
    config: Mapping[str, Any],
) -> None:
    validate_backend_config(config)
    contract = config["selected_guiodyssey_input_contract"]
    observed = {
        "source_format": prepared.source_format,
        "source_mode": prepared.source_mode,
        "alpha_extrema": (
            list(prepared.alpha_extrema) if prepared.alpha_extrema is not None else None
        ),
        "exif_present": prepared.exif_present,
    }
    expected = {
        "source_format": contract["source_format"],
        "source_mode": contract["source_mode"],
        "alpha_extrema": contract["alpha_extrema"],
        "exif_present": contract["exif_present"],
    }
    if observed != expected:
        raise ValueError(f"{contract['violation_outcome']}: image contract drifted")


def prepare_selected_guiodyssey_image(
    image_bytes: bytes,
    config: Mapping[str, Any],
) -> PreparedImage:
    prepared = prepare_image_bytes(image_bytes)
    validate_selected_guiodyssey_image(prepared, config)
    return prepared


def mean_absolute_rgb_difference_from_prepared(
    before: PreparedImage,
    after: PreparedImage,
) -> float:
    if len(before.resized_rgb_bytes) != len(after.resized_rgb_bytes):
        raise ValueError("prepared image buffers must have equal lengths")
    difference = sum(
        abs(left - right)
        for left, right in zip(
            before.resized_rgb_bytes,
            after.resized_rgb_bytes,
            strict=True,
        )
    )
    return difference / (len(before.resized_rgb_bytes) * 255)


def _round_pixel(value: Any, *, extent: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("OCR polygon coordinates must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError("OCR polygon coordinates must be finite and non-negative")
    return min(extent - 1, int(math.floor(numeric + 0.5)))


def _confidence_string(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("OCR confidence must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError("OCR confidence must be finite and in [0, 1]")
    return f"{numeric:.{CONFIDENCE_DECIMAL_PLACES}f}"


def canonicalize_rapidocr_nodes(
    *,
    boxes: Sequence[Any] | None,
    texts: Sequence[Any] | None,
    scores: Sequence[Any] | None,
    width: int,
    height: int,
) -> tuple[dict[str, Any], ...]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if boxes is None and texts is None and scores is None:
        return ()
    if boxes is None or texts is None or scores is None:
        raise ValueError("RapidOCR boxes/texts/scores must be present together")
    if not (len(boxes) == len(texts) == len(scores)):
        raise ValueError("RapidOCR boxes/texts/scores length mismatch")

    nodes: list[dict[str, Any]] = []
    for raw_box, raw_text, raw_score in zip(boxes, texts, scores, strict=True):
        if hasattr(raw_box, "tolist"):
            raw_box = raw_box.tolist()
        if not isinstance(raw_box, Sequence) or isinstance(raw_box, (str, bytes)):
            raise TypeError("RapidOCR polygon must be a sequence")
        polygon: list[list[int]] = []
        for raw_point in raw_box:
            if (
                not isinstance(raw_point, Sequence)
                or isinstance(raw_point, (str, bytes))
                or len(raw_point) != 2
            ):
                raise ValueError("RapidOCR polygon points must contain x and y")
            polygon.append(
                [
                    _round_pixel(raw_point[0], extent=width),
                    _round_pixel(raw_point[1], extent=height),
                ]
            )
        if len(polygon) < 4:
            raise ValueError("RapidOCR polygon must contain at least four points")
        if not isinstance(raw_text, str):
            raise TypeError("RapidOCR text must be a string")
        normalized_text = normalize_screen_text(raw_text)
        if not normalized_text:
            raise ValueError("RapidOCR text cannot be empty after normalization")
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        bbox = [min(ys), min(xs), max(ys), max(xs)]
        nodes.append(
            {
                "polygon_xy": polygon,
                "bbox_top_left_bottom_right": bbox,
                "raw_text": raw_text,
                "normalized_text": normalized_text,
                "confidence_decimal_string": _confidence_string(raw_score),
            }
        )
    nodes.sort(
        key=lambda node: (
            *node["bbox_top_left_bottom_right"],
            node["normalized_text"],
            node["raw_text"],
            node["confidence_decimal_string"],
            node["polygon_xy"],
        )
    )
    return tuple(nodes)


def build_ocr_record(
    *,
    image_member_path: str,
    image_bytes: bytes,
    backend_config_sha256: str,
    boxes: Sequence[Any] | None,
    texts: Sequence[Any] | None,
    scores: Sequence[Any] | None,
) -> dict[str, Any]:
    if not isinstance(image_member_path, str) or not image_member_path:
        raise ValueError("image_member_path must be a safe relative path")
    parsed_member_path = PurePosixPath(image_member_path)
    if (
        "\\" in image_member_path
        or parsed_member_path.is_absolute()
        or str(parsed_member_path) != image_member_path
        or any(part in {".", ".."} for part in parsed_member_path.parts)
    ):
        raise ValueError("image_member_path must be a safe relative path")
    _require_sha256(backend_config_sha256, "backend_config_sha256")
    prepared = prepare_image_bytes(image_bytes)
    nodes = canonicalize_rapidocr_nodes(
        boxes=boxes,
        texts=texts,
        scores=scores,
        width=prepared.width,
        height=prepared.height,
    )
    full_spatial_tokens = tuple(
        token
        for node in nodes
        for token in str(node["normalized_text"]).split(" ")
    )
    record: dict[str, Any] = {
        "schema_version": OCR_RECORD_SCHEMA_VERSION,
        "backend_id": BACKEND_ID,
        "backend_config_sha256": backend_config_sha256,
        "image_member_path": image_member_path,
        "image_sha256": sha256_bytes(image_bytes),
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
        "nodes": list(nodes),
        "full_spatial_tokens": list(full_spatial_tokens),
        "full_spatial_tokens_sha256": sha256_bytes(
            canonical_json_bytes(list(full_spatial_tokens))
        ),
    }
    record["canonical_ocr_record_sha256"] = sha256_bytes(canonical_json_bytes(record))
    return record


def create_rapidocr_engine(config: Mapping[str, Any], model_dir: str | Path) -> Any:
    validate_backend_config(config)
    verify_runtime_packages(config)
    verify_model_files(config, model_dir)
    from rapidocr import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    root = Path(model_dir)
    params = {
        "Global.text_score": 0.5,
        "Global.use_det": True,
        "Global.use_cls": False,
        "Global.use_rec": True,
        "Global.return_word_box": False,
        "Global.return_single_char_box": False,
        "Global.min_height": 30,
        "Global.width_height_ratio": 8,
        "Global.max_side_len": 2000,
        "Global.min_side_len": 30,
        "EngineConfig.onnxruntime.intra_op_num_threads": 1,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.enable_cpu_mem_arena": False,
        "EngineConfig.onnxruntime.use_cuda": False,
        "EngineConfig.onnxruntime.use_dml": False,
        "EngineConfig.onnxruntime.use_cann": False,
        "EngineConfig.onnxruntime.use_coreml": False,
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.lang_type": LangDet.CH,
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.model_path": str(root / config["models"]["detector"]["filename"]),
        "Det.limit_side_len": 736,
        "Det.limit_type": "min",
        "Det.thresh": 0.3,
        "Det.box_thresh": 0.5,
        "Det.max_candidates": 1000,
        "Det.unclip_ratio": 1.6,
        "Det.use_dilation": True,
        "Det.score_mode": "fast",
        "Cls.engine_type": EngineType.ONNXRUNTIME,
        "Cls.lang_type": LangCls.CH,
        "Cls.model_type": ModelType.MOBILE,
        "Cls.ocr_version": OCRVersion.PPOCRV4,
        "Cls.model_path": str(root / config["models"]["inactive_classifier"]["filename"]),
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": LangRec.EN,
        "Rec.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.model_path": str(root / config["models"]["recognizer"]["filename"]),
        "Rec.rec_batch_num": 6,
        "Rec.rec_img_shape": [3, 48, 320],
    }
    engine = RapidOCR(params=params)
    sessions = (
        engine.text_det.session.session,
        engine.text_cls.session.session,
        engine.text_rec.session.session,
    )
    for session in sessions:
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError(
                f"OCR session provider drifted: {session.get_providers()}"
            )
    return engine


def run_rapidocr_record(
    *,
    engine: Any,
    backend_config: Mapping[str, Any],
    image_member_path: str,
    image_bytes: bytes,
    backend_config_sha256: str,
) -> dict[str, Any]:
    prepare_selected_guiodyssey_image(image_bytes, backend_config)
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
    return build_ocr_record(
        image_member_path=image_member_path,
        image_bytes=image_bytes,
        backend_config_sha256=backend_config_sha256,
        boxes=boxes,
        texts=texts,
        scores=scores,
    )


def decode_fixture_image(case: Mapping[str, Any]) -> bytes:
    value = case.get("image_png_base64")
    if not isinstance(value, str):
        raise ValueError("golden fixture image_png_base64 must be a string")
    try:
        payload = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ValueError("golden fixture image_png_base64 is invalid") from error
    expected = _require_sha256(case.get("image_sha256"), "golden image_sha256")
    if sha256_bytes(payload) != expected:
        raise ValueError("golden fixture image SHA256 mismatch")
    return payload
