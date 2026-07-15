"""Fail-closed validator for restoration-v2 baseline implementation identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from causalcache.policy.gui_owl_v2_vision import (
    DEEPSTACK_VISUAL_INDEXES,
    L2_NORM_EPSILON,
    MODEL_CLASS_NAME,
    MODEL_FILE_COUNT,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_TOTAL_BYTES,
    OUTPUT_DTYPE,
    REDUCTION_DTYPE,
    SNAPSHOT_MANIFEST_SHA256,
    TRANSFORMERS_SOURCE_SHA256,
    TRANSFORMERS_VERSION,
    VISION_DEPTH,
    VISION_HIDDEN_SIZE,
    VISION_OUTPUT_SIZE,
    VISION_PATCH_SIZE,
    VISION_SPATIAL_MERGE_SIZE,
    VISION_TEMPORAL_PATCH_SIZE,
)
from causalcache.restoration_v2_baselines import (
    CANDIDATE_EVENT_STEP_IDS,
    OCR_JACCARD_WEIGHT,
    PRIMARY_BUDGET_EVENT_CAPACITY,
    RECENT_EVENT_STEP_IDS,
    RGB_HISTOGRAM_BINS_PER_CHANNEL,
    RGB_HISTOGRAM_WEIGHT,
    SCORE_TIE_ABS_TOL,
    SCORE_TIE_REL_TOL,
    uniform_random_subsets,
)
from causalcache.restoration_v2_contract import (
    FROZEN_RESTORATION_V2_SHA256,
    validate_restoration_v2_contract,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_restoration_v2"
ARTIFACT_ID = "causalcache-restoration-v2-baselines-v1"
STATUS = "PASSED_PREOUTPUT_BASELINE_IMPLEMENTATION"
EXPECTED_SOURCE_PATHS = {
    "code/causalcache/low_fidelity_v2.py",
    "code/causalcache/policy/gui_owl_v2_vision.py",
    "code/causalcache/restoration_v2_baselines.py",
    "code/causalcache/restoration_v2_text_backend.py",
    "code/configs/causalcache_restoration_v2.json",
    "code/configs/gui_owl_1_5_8b_snapshot.json",
    "code/configs/restoration_v2_ocr_backend.json",
    "code/scripts/validate_restoration_v2_baselines.py",
    "code/tests/test_gui_owl_v2_vision.py",
    "code/tests/test_restoration_v2_baseline_manifest.py",
    "code/tests/test_restoration_v2_baselines.py",
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("baseline source path must be a safe relative path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or str(parsed) != value or any(
        part in {".", ".."} for part in parsed.parts
    ):
        raise ValueError("baseline source path must be a safe relative path")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def expected_implementation_contract(
    scientific_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "scientific_baseline_contract": scientific_contract["restoration_gate"][
            "baseline_contract"
        ],
        "candidate_event_step_ids": list(CANDIDATE_EVENT_STEP_IDS),
        "primary_budget_event_capacity": PRIMARY_BUDGET_EVENT_CAPACITY,
        "summary_only_selection": [],
        "recent_selection": list(RECENT_EVENT_STEP_IDS),
        "uniform_random_exact_subsets": [
            list(value) for value in uniform_random_subsets(CANDIDATE_EVENT_STEP_IDS)
        ],
        "uniform_random_seed": None,
        "ocr_and_rgb": {
            "ocr_normalization": "Unicode_NFKC_collapse_whitespace_strip_then_set",
            "ocr_jaccard_weight": OCR_JACCARD_WEIGHT,
            "rgb_resize": [256, 256],
            "rgb_joint_histogram_bins_per_channel": RGB_HISTOGRAM_BINS_PER_CHANNEL,
            "rgb_histogram_weight": RGB_HISTOGRAM_WEIGHT,
            "empty_ocr_set_jaccard": 1.0,
        },
        "score_tie": {
            "rel_tol": SCORE_TIE_REL_TOL,
            "abs_tol": SCORE_TIE_ABS_TOL,
            "secondary_key": "lower_event_step_id",
        },
        "policy_vision": {
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "snapshot_manifest_sha256": SNAPSHOT_MANIFEST_SHA256,
            "snapshot_file_count": MODEL_FILE_COUNT,
            "snapshot_total_bytes": MODEL_TOTAL_BYTES,
            "snapshot_verification": (
                "exact_local_snapshot_json_and_every_listed_file_size_sha256_and_"
                "exact_file_inventory_before_extraction"
            ),
            "model_class": MODEL_CLASS_NAME,
            "transformers_version": TRANSFORMERS_VERSION,
            "transformers_source_sha256": TRANSFORMERS_SOURCE_SHA256,
            "feature_call": (
                "Qwen3VLForConditionalGeneration.get_image_features"
                "(pixel_values,image_grid_thw,return_dict=True)"
            ),
            "feature_field": "pooler_output",
            "feature_semantics": "final_main_vision_merger_output",
            "pre_merger_last_hidden_state_excluded": True,
            "all_deepstack_features_excluded": True,
            "deepstack_visual_indexes": list(DEEPSTACK_VISUAL_INDEXES),
            "vision_depth": VISION_DEPTH,
            "vision_hidden_size_before_merger": VISION_HIDDEN_SIZE,
            "vision_output_size_after_merger": VISION_OUTPUT_SIZE,
            "patch_size": VISION_PATCH_SIZE,
            "temporal_patch_size": VISION_TEMPORAL_PATCH_SIZE,
            "spatial_merge_size": VISION_SPATIAL_MERGE_SIZE,
            "per_image_boundary": "t*h*w/(spatial_merge_size^2)",
            "feature_dtype": OUTPUT_DTYPE,
            "mean_and_l2_reduction_dtype": REDUCTION_DTYPE,
            "l2_norm_epsilon": L2_NORM_EPSILON,
            "normalized_embeddings_remain_on_accelerator": True,
            "full_visual_features_move_to_cpu": False,
            "cpu_transfer": (
                "grid_metadata_per_image_norm_validation_scalars_and_four_final_"
                "cosines_only"
            ),
            "language_model_called": False,
            "lm_head_called": False,
            "generate_called": False,
        },
    }


def validate_baseline_manifest(
    manifest_path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    manifest = load_json_object(manifest_path)
    if set(manifest) != {
        "schema_version",
        "protocol_id",
        "artifact_id",
        "status",
        "scientific_contract",
        "source_files",
        "implementation_contract",
        "negative_declarations",
        "dependency_6_closed",
    }:
        raise ValueError("baseline manifest top-level schema drifted")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("baseline manifest schema_version drifted")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("baseline manifest protocol_id drifted")
    if manifest.get("artifact_id") != ARTIFACT_ID:
        raise ValueError("baseline manifest artifact_id drifted")
    if manifest.get("status") != STATUS:
        raise ValueError("baseline manifest status drifted")
    expected_scientific = {
        "path": "code/configs/causalcache_restoration_v2.json",
        "sha256": FROZEN_RESTORATION_V2_SHA256,
    }
    if manifest.get("scientific_contract") != expected_scientific:
        raise ValueError("baseline scientific contract identity drifted")

    root = Path(repository_root).resolve()
    contract_path = root / expected_scientific["path"]
    if sha256_file(contract_path) != FROZEN_RESTORATION_V2_SHA256:
        raise ValueError("baseline scientific contract source bytes drifted")
    scientific_contract = load_json_object(contract_path)
    validate_restoration_v2_contract(scientific_contract)

    source_files = manifest.get("source_files")
    if not isinstance(source_files, list):
        raise ValueError("baseline source_files must be a list")
    observed_paths: set[str] = set()
    for record in source_files:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError("baseline source file record schema drifted")
        relative = _safe_relative_path(record.get("path"))
        if relative in observed_paths:
            raise ValueError("baseline source file paths must be unique")
        expected_sha = _require_sha256(record.get("sha256"), "baseline source SHA256")
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"baseline source file SHA256 drifted: {relative}")
        observed_paths.add(relative)
    if observed_paths != EXPECTED_SOURCE_PATHS:
        raise ValueError("baseline source file inventory drifted")

    expected_implementation = expected_implementation_contract(scientific_contract)
    if manifest.get("implementation_contract") != expected_implementation:
        raise ValueError("baseline implementation contract drifted")
    if manifest.get("negative_declarations") != {
        "policy_loaded": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "confirm_scores_observed": False,
        "sampled_random_selector_implemented": False,
    }:
        raise ValueError("baseline negative declarations drifted")
    if manifest.get("dependency_6_closed") is not True:
        raise ValueError("baseline dependency 6 must be explicitly closed")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "outcome": "PASSED_BASELINE_SOURCE_VALIDATION",
        "manifest_sha256": sha256_file(manifest_path),
        "source_file_count": len(source_files),
        "baseline_count": len(scientific_contract["restoration_gate"]["baselines"]),
        "dependency_6_closed": True,
        "policy_output_generated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_baseline_manifest(
                args.manifest,
                repository_root=args.repository_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
