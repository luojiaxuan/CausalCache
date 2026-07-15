"""Vision-only GUI-Owl features for the frozen restoration-v2 baseline."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.restoration_v2_baselines import (
    BaselineSelection,
    CANDIDATE_EVENT_STEP_IDS,
    select_top_two,
)


MODEL_CLASS_NAME = "Qwen3VLForConditionalGeneration"
MODEL_REPO = "mPLUG/GUI-Owl-1.5-8B-Instruct"
MODEL_REVISION = "06d5faecff74840bab2be2425e9c42667a5d04fc"
SNAPSHOT_MANIFEST_SHA256 = (
    "50b675ec31c5c46dbb0d44c137a808fffb9d054916d39b596648d4eb9df7cbc3"
)
MODEL_FILE_COUNT = 14
MODEL_TOTAL_BYTES = 17_545_907_171
TRANSFORMERS_VERSION = "5.6.0"
VISION_DEPTH = 27
VISION_HIDDEN_SIZE = 1152
VISION_OUTPUT_SIZE = 4096
VISION_PATCH_SIZE = 16
VISION_TEMPORAL_PATCH_SIZE = 2
VISION_SPATIAL_MERGE_SIZE = 2
DEEPSTACK_VISUAL_INDEXES = (8, 16, 24)
REDUCTION_DTYPE = "torch.float32"
OUTPUT_DTYPE = "torch.bfloat16"
L2_NORM_EPSILON = 1e-12
TRANSFORMERS_SOURCE_SHA256 = {
    "modeling_qwen3_vl.py": (
        "c37abc40a3744dc116cf1e06eceb7ff79bd384cfbfe0d41773f5efc074f31523"
    ),
    "processing_qwen3_vl.py": (
        "b51f77d783b4a88d503bf9198828d46f71e3ead3833da2631748188b159253ba"
    ),
    "image_processing_qwen2_vl.py": (
        "de5859892f04c64f3e79c9d318842ac0040000788e43870997bd1ec6565c02bb"
    ),
}
TRANSFORMERS_MODULES = {
    "modeling_qwen3_vl.py": "transformers.models.qwen3_vl.modeling_qwen3_vl",
    "processing_qwen3_vl.py": "transformers.models.qwen3_vl.processing_qwen3_vl",
    "image_processing_qwen2_vl.py": (
        "transformers.models.qwen2_vl.image_processing_qwen2_vl"
    ),
}
_VERIFIED_RUNTIME_TOKEN = object()


@dataclass(frozen=True)
class VerifiedVisionRuntimeIdentity:
    """Identity returned only after hashing the complete model and runtime source."""

    model_dir: str
    model_repo: str
    model_revision: str
    snapshot_manifest_sha256: str
    verified_model_file_count: int
    verified_model_total_bytes: int
    transformers_version: str
    transformers_source_sha256: tuple[tuple[str, str], ...]
    _verification_token: object


@dataclass(frozen=True)
class VisionEmbeddingBatch:
    """GPU-resident normalized embeddings and auditable per-image geometry."""

    normalized_embeddings: Any
    image_grid_thw: tuple[tuple[int, int, int], ...]
    raw_patch_boundaries: tuple[int, ...]
    merged_token_boundaries: tuple[int, ...]
    merged_token_counts: tuple[int, ...]
    output_dtype: str
    reduction_dtype: str
    hidden_size: int
    feature_field: str
    deepstack_features_excluded: bool


def _positive_integer(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _safe_model_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("model snapshot path must be a safe relative path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or str(parsed) != value or any(
        part in {".", ".."} for part in parsed.parts
    ):
        raise ValueError("model snapshot path must be a safe relative path")
    return value


def verify_frozen_vision_runtime(
    *,
    model_dir: str | Path,
    expected_snapshot_manifest: str | Path,
) -> VerifiedVisionRuntimeIdentity:
    """Hash every model file and the exact Transformers vision implementation."""
    root = Path(model_dir).resolve()
    manifest_path = Path(expected_snapshot_manifest).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"GUI-Owl model directory does not exist: {root}")
    if _sha256_file(manifest_path) != SNAPSHOT_MANIFEST_SHA256:
        raise ValueError("Git GUI-Owl snapshot manifest SHA256 drifted")
    expected = _load_json_object(manifest_path)
    local_snapshot_path = root / ".snapshot.json"
    if not local_snapshot_path.is_file():
        raise FileNotFoundError("GUI-Owl model directory is missing .snapshot.json")
    local = _load_json_object(local_snapshot_path)
    if local != expected:
        raise ValueError("local GUI-Owl snapshot identity differs from the Git manifest")
    if set(expected) != {"repo", "revision", "files"}:
        raise ValueError("GUI-Owl snapshot manifest schema drifted")
    if expected["repo"] != MODEL_REPO or expected["revision"] != MODEL_REVISION:
        raise ValueError("GUI-Owl snapshot repo or revision drifted")
    files = expected["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("GUI-Owl snapshot files must be a non-empty list")
    seen: set[str] = set()
    total_bytes = 0
    for record in files:
        if not isinstance(record, Mapping) or set(record) != {"path", "size", "sha256"}:
            raise ValueError("GUI-Owl snapshot file schema drifted")
        relative = _safe_model_relative_path(record["path"])
        if relative in seen:
            raise ValueError("GUI-Owl snapshot paths must be unique")
        if type(record["size"]) is not int or record["size"] < 0:
            raise ValueError("GUI-Owl snapshot file size must be non-negative")
        if (
            not isinstance(record["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        ):
            raise ValueError("GUI-Owl snapshot file SHA256 is invalid")
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError(f"GUI-Owl snapshot file size drifted: {relative}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"GUI-Owl snapshot file SHA256 drifted: {relative}")
        seen.add(relative)
        total_bytes += int(record["size"])
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != seen.union({".snapshot.json"}):
        raise ValueError("GUI-Owl model directory contains an unexpected file inventory")

    actual_transformers_version = importlib.metadata.version("transformers")
    if actual_transformers_version != TRANSFORMERS_VERSION:
        raise ValueError("Transformers version drifted for the vision baseline")
    verified_sources: list[tuple[str, str]] = []
    for filename, module_name in TRANSFORMERS_MODULES.items():
        module = importlib.import_module(module_name)
        source_path = Path(str(module.__file__)).resolve()
        if source_path.name != filename:
            raise ValueError("Transformers vision source filename drifted")
        digest = _sha256_file(source_path)
        if digest != TRANSFORMERS_SOURCE_SHA256[filename]:
            raise ValueError(f"Transformers vision source SHA256 drifted: {filename}")
        verified_sources.append((filename, digest))
    return VerifiedVisionRuntimeIdentity(
        model_dir=str(root),
        model_repo=MODEL_REPO,
        model_revision=MODEL_REVISION,
        snapshot_manifest_sha256=SNAPSHOT_MANIFEST_SHA256,
        verified_model_file_count=len(files),
        verified_model_total_bytes=total_bytes,
        transformers_version=TRANSFORMERS_VERSION,
        transformers_source_sha256=tuple(sorted(verified_sources)),
        _verification_token=_VERIFIED_RUNTIME_TOKEN,
    )


def canonical_image_grid_thw(image_grid_thw: Any) -> tuple[tuple[int, int, int], ...]:
    if hasattr(image_grid_thw, "detach"):
        raw = image_grid_thw.detach().to(device="cpu").tolist()
    else:
        raw = image_grid_thw
    if isinstance(raw, (str, bytes, bytearray, Mapping)):
        raise TypeError("image_grid_thw must be a rank-2 integer sequence")
    try:
        rows = tuple(raw)
    except TypeError as error:
        raise TypeError("image_grid_thw must be a rank-2 integer sequence") from error
    if not rows:
        raise ValueError("image_grid_thw must contain at least one image")
    result: list[tuple[int, int, int]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise TypeError(f"image_grid_thw[{index}] must be a length-three sequence")
        if len(row) != 3:
            raise ValueError(f"image_grid_thw[{index}] must contain t, h, and w")
        temporal, height, width = (
            _positive_integer(value, f"image_grid_thw[{index}]") for value in row
        )
        if (
            height % VISION_SPATIAL_MERGE_SIZE
            or width % VISION_SPATIAL_MERGE_SIZE
        ):
            raise ValueError("visual grid height and width must divide by merge size")
        result.append((temporal, height, width))
    return tuple(result)


def visual_token_geometry(
    image_grid_thw: Any,
) -> dict[str, tuple[int, ...] | tuple[tuple[int, int, int], ...]]:
    grids = canonical_image_grid_thw(image_grid_thw)
    raw_counts = tuple(t * h * w for t, h, w in grids)
    merged_counts = tuple(
        value // (VISION_SPATIAL_MERGE_SIZE**2) for value in raw_counts
    )

    def cumulative(values: tuple[int, ...]) -> tuple[int, ...]:
        boundaries = [0]
        for value in values:
            boundaries.append(boundaries[-1] + value)
        return tuple(boundaries)

    return {
        "image_grid_thw": grids,
        "raw_patch_counts": raw_counts,
        "raw_patch_boundaries": cumulative(raw_counts),
        "merged_token_counts": merged_counts,
        "merged_token_boundaries": cumulative(merged_counts),
    }


def _validate_model_identity(
    model: Any,
    runtime_identity: VerifiedVisionRuntimeIdentity,
) -> None:
    if (
        not isinstance(runtime_identity, VerifiedVisionRuntimeIdentity)
        or runtime_identity._verification_token is not _VERIFIED_RUNTIME_TOKEN
    ):
        raise ValueError("vision extraction requires a verified frozen runtime identity")
    if runtime_identity.model_repo != MODEL_REPO or runtime_identity.model_revision != MODEL_REVISION:
        raise ValueError("verified GUI-Owl repo or revision drifted")
    if runtime_identity.snapshot_manifest_sha256 != SNAPSHOT_MANIFEST_SHA256:
        raise ValueError("verified GUI-Owl snapshot manifest drifted")
    if (
        runtime_identity.verified_model_file_count != MODEL_FILE_COUNT
        or runtime_identity.verified_model_total_bytes != MODEL_TOTAL_BYTES
    ):
        raise ValueError("verified GUI-Owl model file inventory drifted")
    if runtime_identity.transformers_version != TRANSFORMERS_VERSION:
        raise ValueError("verified Transformers version drifted")
    if dict(runtime_identity.transformers_source_sha256) != TRANSFORMERS_SOURCE_SHA256:
        raise ValueError("verified Transformers source identity drifted")
    if model.__class__.__name__ != MODEL_CLASS_NAME:
        raise TypeError(f"vision baseline requires {MODEL_CLASS_NAME}")
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("GUI-Owl model is missing config")
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if architectures != (MODEL_CLASS_NAME,):
        raise ValueError("GUI-Owl model architecture identity drifted")
    name_or_path = getattr(config, "_name_or_path", None)
    if not isinstance(name_or_path, str) or Path(name_or_path).resolve() != Path(
        runtime_identity.model_dir
    ):
        raise ValueError("loaded GUI-Owl model path differs from verified snapshot")
    vision = getattr(config, "vision_config", None)
    expected = {
        "depth": VISION_DEPTH,
        "hidden_size": VISION_HIDDEN_SIZE,
        "out_hidden_size": VISION_OUTPUT_SIZE,
        "patch_size": VISION_PATCH_SIZE,
        "temporal_patch_size": VISION_TEMPORAL_PATCH_SIZE,
        "spatial_merge_size": VISION_SPATIAL_MERGE_SIZE,
        "deepstack_visual_indexes": list(DEEPSTACK_VISUAL_INDEXES),
    }
    if vision is None or any(getattr(vision, key, None) != value for key, value in expected.items()):
        raise ValueError("GUI-Owl vision config identity drifted")


def extract_normalized_spatial_merger_embeddings(
    *,
    model: Any,
    pixel_values: Any,
    image_grid_thw: Any,
    runtime_identity: VerifiedVisionRuntimeIdentity,
) -> VisionEmbeddingBatch:
    """Run only the final main vision merger and reduce each image on device."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("GUI-Owl vision extraction requires PyTorch") from error

    _validate_model_identity(model, runtime_identity)
    geometry = visual_token_geometry(image_grid_thw)
    grids = geometry["image_grid_thw"]
    assert isinstance(grids, tuple)
    raw_counts = geometry["raw_patch_counts"]
    assert isinstance(raw_counts, tuple)
    if not hasattr(pixel_values, "ndim") or pixel_values.ndim != 2:
        raise ValueError("pixel_values must be a rank-2 packed-patch tensor")
    if int(pixel_values.shape[0]) != sum(raw_counts):
        raise ValueError("pixel_values patch count differs from image_grid_thw")
    if int(pixel_values.shape[1]) != (
        3 * VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE
    ):
        raise ValueError("pixel_values packed patch width drifted")

    with torch.inference_mode():
        outputs = model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        features = getattr(outputs, "pooler_output", None)
        if not isinstance(features, (tuple, list)) or len(features) != len(grids):
            raise ValueError("pooler_output must contain one tensor per input image")
        merged_counts = geometry["merged_token_counts"]
        assert isinstance(merged_counts, tuple)
        normalized = []
        for image_index, (feature, token_count) in enumerate(
            zip(features, merged_counts, strict=True)
        ):
            if not isinstance(feature, torch.Tensor) or feature.ndim != 2:
                raise TypeError(f"pooler_output[{image_index}] must be a rank-2 tensor")
            if tuple(feature.shape) != (token_count, VISION_OUTPUT_SIZE):
                raise ValueError("post-merger visual output shape drifted")
            if feature.dtype is not torch.bfloat16:
                raise ValueError("post-merger visual output must remain bfloat16")
            if feature.device != pixel_values.device:
                raise ValueError("post-merger visual output changed device")
            feature_float = feature.float()
            if not torch.isfinite(feature_float).all():
                raise ValueError("post-merger visual output contains non-finite values")
            pooled = feature_float.mean(dim=0)
            norm = torch.linalg.vector_norm(pooled)
            if not torch.isfinite(norm) or float(norm.item()) <= L2_NORM_EPSILON:
                raise ValueError("mean-pooled visual output has invalid L2 norm")
            normalized.append(pooled / norm)
        embedding_tensor = torch.stack(normalized, dim=0)
        if embedding_tensor.dtype is not torch.float32:
            raise RuntimeError("vision baseline reduction must use float32")
        if not torch.isfinite(embedding_tensor).all():
            raise ValueError("normalized visual embeddings contain non-finite values")

    return VisionEmbeddingBatch(
        normalized_embeddings=embedding_tensor,
        image_grid_thw=grids,
        raw_patch_boundaries=geometry["raw_patch_boundaries"],
        merged_token_boundaries=geometry["merged_token_boundaries"],
        merged_token_counts=geometry["merged_token_counts"],
        output_dtype=str(features[0].dtype),
        reduction_dtype=str(embedding_tensor.dtype),
        hidden_size=int(embedding_tensor.shape[1]),
        feature_field="pooler_output",
        deepstack_features_excluded=True,
    )


def frozen_policy_vision_similarity_from_batch(
    batch: VisionEmbeddingBatch,
    *,
    event_image_indices: Mapping[int, int],
    current_image_index: int,
) -> BaselineSelection:
    """Compute four on-device cosines and transfer only scalar scores to CPU."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("GUI-Owl vision similarity requires PyTorch") from error
    if set(event_image_indices) != set(CANDIDATE_EVENT_STEP_IDS):
        raise ValueError("event_image_indices must cover exactly events 1,2,3,4")
    if any(type(value) is not int for value in event_image_indices.values()):
        raise TypeError("event image indices must be integers")
    if type(current_image_index) is not int:
        raise TypeError("current image index must be an integer")
    embeddings = batch.normalized_embeddings
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise TypeError("normalized_embeddings must be a rank-2 tensor")
    if embeddings.dtype is not torch.float32 or int(embeddings.shape[1]) != VISION_OUTPUT_SIZE:
        raise ValueError("normalized embedding dtype or hidden size drifted")
    if (
        batch.feature_field != "pooler_output"
        or not batch.deepstack_features_excluded
        or batch.output_dtype != OUTPUT_DTYPE
        or batch.reduction_dtype != REDUCTION_DTYPE
        or batch.hidden_size != VISION_OUTPUT_SIZE
        or len(batch.image_grid_thw) != int(embeddings.shape[0])
    ):
        raise ValueError("vision embedding batch metadata drifted")
    norms = torch.linalg.vector_norm(embeddings, dim=1)
    if not torch.isfinite(norms).all() or not torch.allclose(
        norms,
        torch.ones_like(norms),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("vision embeddings must remain L2-normalized")
    indices = tuple(event_image_indices[step] for step in CANDIDATE_EVENT_STEP_IDS)
    all_indices = (*indices, current_image_index)
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("candidate and current image indices must be distinct")
    if any(index < 0 or index >= int(embeddings.shape[0]) for index in all_indices):
        raise ValueError("vision baseline image index is out of range")
    with torch.inference_mode():
        event_vectors = embeddings[list(indices)]
        current = embeddings[current_image_index]
        scores_tensor = torch.matmul(event_vectors, current)
        if not torch.isfinite(scores_tensor).all():
            raise ValueError("vision baseline cosine contains non-finite values")
        scores = [float(value) for value in scores_tensor.detach().cpu().tolist()]
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("vision baseline scalar cosine contains non-finite values")
    return select_top_two(
        dict(zip(CANDIDATE_EVENT_STEP_IDS, scores, strict=True))
    )
