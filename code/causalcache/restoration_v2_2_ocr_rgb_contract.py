"""Fail-closed source contract for the restoration-v2.2 OCR/RGB baseline."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL_ID = "causalcache_restoration_v2_2_ocr_rgb_baseline_v1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline.json"
)
FROZEN_CONFIG_SHA256 = (
    "08f57505d71e603d81d6915ef27cf008d0fe097a56d70a05f8b3519211e2e6f9"
)
PASS_STATUS = "VALID_RESTORATION_V2_2_OCR_RGB_BASELINE_SOURCE_CONTRACT"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
ALLOWED_ROLES = ("v2_label_train", "v2_development")
EXPECTED_OPERATION_CEILING = {
    "validated_label_state_count": 45,
    "validated_label_distance_row_count": 420,
    "primary_label_state_count": 15,
    "primary_label_distance_row_count": 240,
    "geometry_primary_record_count": 15,
    "geometry_nonprimary_semantic_parse_count": 0,
    "semantic_trajectory_record_count": 15,
    "semantic_ocr_record_count": 75,
    "image_payload_extract_count": 75,
    "image_decode_resize_count": 75,
    "candidate_similarity_comparison_count": 60,
    "ocr_similarity_comparison_count": 60,
    "rgb_similarity_comparison_count": 60,
    "combined_similarity_score_count": 60,
    "selected_coalition_distance_lookup_count": 15,
    "ocr_inference_count": 0,
    "ocr_model_load_count": 0,
    "policy_model_load_count": 0,
    "gpu_count": 0,
    "policy_forward_count": 0,
    "policy_vision_feature_forward_count": 0,
    "generation_count": 0,
    "teacher_forward_count": 0,
    "kl_measurement_count": 0,
    "gate_training_example_count": 0,
    "gate_model_forward_count": 0,
    "matched_nll_evaluation_count": 0,
    "closed_loop_episode_count": 0,
    "confirm_state_access_count": 0,
    "confirm_feature_score_count": 0,
    "derived_nonselected_semantic_parse_count": 0,
    "sealed_test_state_access_count": 0,
}
EXPECTED_OUTPUT_FILES = ("README.md", "state_scores.jsonl", "summary.json")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def strict_json_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=pairs_hook)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} key schema drifted: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe relative path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or str(parsed) != value
        or any(part in {".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"{label} must be a safe relative path")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _git_bytes(root: Path, revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _validate_file_record(
    root: Path,
    record: Mapping[str, Any],
    *,
    prefix: str = "",
    revision: str | None = None,
) -> None:
    _exact_keys(record, {"path", "size_bytes", "sha256"}, "file record")
    relative = _safe_relative_path(record["path"], "file path")
    size = record["size_bytes"]
    digest = _require_sha256(record["sha256"], "file SHA256")
    if type(size) is not int or size < 0:
        raise ValueError("file size must be a non-negative integer")
    repository_path = f"{prefix}/{relative}" if prefix else relative
    path = root.joinpath(*PurePosixPath(repository_path).parts)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"bound source file is missing or symlinked: {repository_path}")
    payload = path.read_bytes()
    if len(payload) != size or sha256_bytes(payload) != digest:
        raise ValueError(f"bound source file identity drifted: {repository_path}")
    if revision is not None and _git_bytes(root, revision, repository_path) != payload:
        raise ValueError(f"bound source differs from its frozen commit: {repository_path}")


def _validate_contract_semantics(data: Mapping[str, Any]) -> None:
    _exact_keys(
        data,
        {
            "schema_version",
            "protocol_id",
            "preregistration_status",
            "authorization",
            "immutable_inputs",
            "scope",
            "feature_contract",
            "statistics_contract",
            "operation_ceiling",
            "output_contract",
        },
        "OCR/RGB contract",
    )
    if data["schema_version"] != "0.1.0" or data["protocol_id"] != PROTOCOL_ID:
        raise ValueError("OCR/RGB contract protocol identity drifted")
    if data["preregistration_status"] != (
        "source_only_frozen_before_any_formal_ocr_rgb_primary_aggregate"
    ):
        raise ValueError("OCR/RGB source-freeze status drifted")
    authorization = _mapping(data["authorization"], "authorization")
    if authorization.get("cpu_feature_reduction_allowed") is not True or any(
        value is not False
        for key, value in authorization.items()
        if key != "cpu_feature_reduction_allowed"
    ):
        raise ValueError("OCR/RGB authorization must remain CPU-only and policy-blind")
    scope = _mapping(data["scope"], "scope")
    expected_scope = {
        "allowed_roles": list(ALLOWED_ROLES),
        "primary_candidate_event_count": 4,
        "primary_budget_event_capacity": 2,
        "primary_decision_step_id": 6,
        "candidate_event_step_ids": [1, 2, 3, 4],
        "current_equivalent_event_step_id": 5,
        "expected_trajectory_count": 15,
        "expected_state_count": 15,
        "expected_candidate_comparison_count": 60,
        "expected_unique_image_count": 75,
        "expected_role_trajectory_counts": {
            "v2_label_train": 10,
            "v2_development": 5,
        },
        "candidate_image_role": "event_observation_after_post_action_state",
        "query_image_role": "decision_step_6_current_observation",
        "derived_nonselected_records_semantic_parse_allowed": False,
    }
    if dict(scope) != expected_scope:
        raise ValueError("OCR/RGB primary scope drifted")
    if dict(_mapping(data["operation_ceiling"], "operation ceiling")) != (
        EXPECTED_OPERATION_CEILING
    ):
        raise ValueError("OCR/RGB operation ceiling drifted")
    feature = _mapping(data["feature_contract"], "feature contract")
    required_feature_values = {
        "ocr_source": "archived_full_spatial_tokens_only",
        "rgb_source": "prepare_image_bytes_resized_rgb_bytes",
        "rgb_resize": [256, 256],
        "rgb_joint_histogram_bins_per_channel": 16,
        "ocr_weight": 0.5,
        "rgb_weight": 0.5,
        "selection": "top_2_by_combined_score",
        "score_tie_rel_tol": 1e-09,
        "score_tie_abs_tol": 1e-12,
        "score_tie_secondary_key": "lower_event_step_id",
    }
    for key, expected in required_feature_values.items():
        if feature.get(key) != expected:
            raise ValueError(f"OCR/RGB feature contract drifted: {key}")
    statistics = _mapping(data["statistics_contract"], "statistics contract")
    expected_statistics = {
        "resampling_unit": "trajectory_id",
        "state_weighting_within_trajectory": "equal",
        "trajectory_weighting_within_role": "equal",
        "overall_resampling": "stratified_within_role",
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 271_828,
        "bootstrap_confidence": 0.9,
        "interval": "percentile_linear_interpolation",
        "tie_epsilon": 1e-12,
        "paired_recovery_comparisons": [
            "exact_subset",
            "exact_cardinality_oracle",
            "true_conditional_greedy",
            "budget_conditioned_independent",
            "full_shapley_independent",
            "dynamic_recent",
            "analytic_exact_cardinality_random",
        ],
    }
    if dict(statistics) != expected_statistics:
        raise ValueError("OCR/RGB statistics contract drifted")
    output = _mapping(data["output_contract"], "output contract")
    if output.get("canonical_result_directory") != (
        "data/results/restoration_v2_2_ocr_rgb_baseline_v1"
    ) or tuple(output.get("exact_files", ())) != EXPECTED_OUTPUT_FILES:
        raise ValueError("OCR/RGB output identity drifted")
    if any(
        output.get(key) is not True
        for key in (
            "formal_run_requires_clean_pushed_main",
            "run_requires_exclusive_new_output_directory",
            "validate_rebuilds_every_output_byte_from_immutable_inputs",
        )
    ):
        raise ValueError("OCR/RGB output safety rules drifted")


@dataclass(frozen=True)
class RestorationV22OcrRgbContract:
    path: Path
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        validate_bound_sources: bool = True,
    ) -> "RestorationV22OcrRgbContract":
        root = Path(repository_root).resolve()
        resolved = Path(path).resolve()
        canonical = (root / CANONICAL_CONFIG_PATH).resolve()
        if resolved != canonical or not resolved.is_file() or resolved.is_symlink():
            raise ValueError("OCR/RGB contract must use the canonical config")
        digest = sha256_file(resolved)
        if digest != FROZEN_CONFIG_SHA256:
            raise ValueError("OCR/RGB frozen config SHA256 drifted")
        data = strict_json_object_bytes(resolved.read_bytes(), label="OCR/RGB contract")
        _validate_contract_semantics(data)
        contract = cls(resolved, data, digest, root)
        if validate_bound_sources:
            contract.validate_bound_sources()
        return contract

    def validate_bound_sources(self) -> None:
        inputs = _mapping(self.data["immutable_inputs"], "immutable inputs")
        _exact_keys(
            inputs,
            {
                "selector_geometry_result",
                "restoration_labels",
                "derived_dataset",
                "baseline_implementation",
            },
            "immutable inputs",
        )
        geometry = _mapping(inputs["selector_geometry_result"], "geometry result")
        revision = geometry.get("git_commit")
        if not isinstance(revision, str) or GIT_SHA_PATTERN.fullmatch(revision) is None:
            raise ValueError("geometry result commit must be a full Git SHA")
        expected_geometry_lineage = {
            "execution_source_git_commit": (
                "9a4eca5a53c2a9a3340c6274b9fa5ff9012a5a64"
            ),
            "artifact_git_commit": (
                "d0f25d812869d5fc7b58284a25abe3aa8049b0aa"
            ),
            "validation_git_commit": (
                "5e2e45089d5bfa1a0f0e538f337a51b9b5297a4e"
            ),
            "repair_contract_sha256": (
                "2d312f54559f67aafe7efec2656d23171000e8b0f41d2d3c923f6a3c8b43be4c"
            ),
            "parent_contract_sha256": (
                "8022dcdec272916b7975d696a3ce6b54022c7414cd348a715c55b0d3d694dad5"
            ),
        }
        if revision != expected_geometry_lineage["artifact_git_commit"] or any(
            geometry.get(key) != value
            for key, value in expected_geometry_lineage.items()
        ):
            raise ValueError("geometry execution/artifact/validation lineage drifted")
        for commit in (
            expected_geometry_lineage["execution_source_git_commit"],
            expected_geometry_lineage["artifact_git_commit"],
            expected_geometry_lineage["validation_git_commit"],
        ):
            subprocess.run(
                ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                cwd=self.repository_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        for ancestor, descendant in (
            (
                expected_geometry_lineage["execution_source_git_commit"],
                expected_geometry_lineage["artifact_git_commit"],
            ),
            (
                expected_geometry_lineage["artifact_git_commit"],
                expected_geometry_lineage["validation_git_commit"],
            ),
        ):
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", ancestor, descendant],
                cwd=self.repository_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        directory = _safe_relative_path(geometry.get("directory"), "geometry directory")
        records = _sequence(geometry.get("files"), "geometry files")
        if tuple(record.get("path") for record in records) != (
            "README.md",
            "state_budget_records.jsonl",
            "summary.json",
        ):
            raise ValueError("geometry result exact file inventory drifted")
        for raw_record in records:
            record = _mapping(raw_record, "geometry file")
            _validate_file_record(
                self.repository_root,
                record,
                prefix=directory,
                revision=revision,
            )
            repository_path = f"{directory}/{record['path']}"
            if _git_bytes(
                self.repository_root,
                expected_geometry_lineage["validation_git_commit"],
                repository_path,
            ) != _git_bytes(self.repository_root, revision, repository_path):
                raise ValueError(
                    f"geometry validation commit changed artifact bytes: {repository_path}"
                )
        frozen_contracts = {
            "code/configs/causalcache_restoration_v2_2_selector_geometry_v2_repair.json": (
                expected_geometry_lineage["repair_contract_sha256"]
            ),
            "code/configs/causalcache_restoration_v2_2_selector_geometry.json": (
                expected_geometry_lineage["parent_contract_sha256"]
            ),
        }
        for path, digest in frozen_contracts.items():
            payload = _git_bytes(
                self.repository_root,
                expected_geometry_lineage["execution_source_git_commit"],
                path,
            )
            if sha256_bytes(payload) != digest:
                raise ValueError(f"geometry contract lineage hash drifted: {path}")

        labels = _mapping(inputs["restoration_labels"], "label input")
        label_binding = self._validate_binding(labels, "binding_path", "binding_sha256")
        raw = _mapping(label_binding.get("raw_archive"), "label raw archive")
        artifact = _mapping(label_binding.get("hf_artifact"), "label HF artifact")
        expected_label_values = {
            "hf_repo": artifact.get("repo"),
            "hf_revision": artifact.get("immutable_revision"),
            "hf_tag": artifact.get("tag"),
            "hf_path": artifact.get("path"),
            "raw_archive_sha256": raw.get("sha256"),
            "raw_archive_size_bytes": raw.get("size_bytes"),
            "raw_archive_file_count": raw.get("file_count"),
            "raw_tree_inventory_sha256": raw.get("tree_inventory_sha256"),
        }
        for key, observed in expected_label_values.items():
            if labels.get(key) != observed:
                raise ValueError(f"label immutable identity drifted: {key}")

        derived = _mapping(inputs["derived_dataset"], "derived dataset")
        derived_binding = self._validate_binding(
            derived,
            "binding_path",
            "binding_sha256",
        )
        hf = _mapping(derived_binding.get("hf_dataset_artifact"), "derived HF artifact")
        if (
            derived.get("hf_repo") != hf.get("repo")
            or derived.get("hf_revision") != hf.get("immutable_revision")
            or derived.get("hf_tag") != hf.get("tag")
            or derived.get("artifact_tree_sha256") != hf.get("artifact_tree_sha256")
            or derived.get("counts") != hf.get("counts")
            or derived.get("exact_files") != hf.get("files")
        ):
            raise ValueError("derived immutable HF identity drifted")

        baseline = _mapping(inputs["baseline_implementation"], "baseline input")
        source_closure = _mapping(
            baseline.get("source_closure"),
            "baseline transitive source closure",
        )
        if dict(source_closure) != {
            "revision": "formal_source_git_commit",
            "tracked_python_roots": ["code/causalcache", "code/scripts"],
            "suffix": ".py",
            "validation": "all_source_commit_paths_must_remain_byte_unchanged",
        }:
            raise ValueError("baseline transitive source closure drifted")
        manifest_path = _safe_relative_path(
            baseline.get("manifest_path"),
            "baseline manifest path",
        )
        manifest = self._read_bound_json(
            manifest_path,
            baseline.get("manifest_sha256"),
            label="baseline manifest",
        )
        if (
            manifest.get("artifact_id") != baseline.get("artifact_id")
            or manifest.get("status") != baseline.get("required_status")
        ):
            raise ValueError("baseline manifest identity drifted")
        manifest_sources = {
            record["path"]: record["sha256"]
            for record in _sequence(manifest.get("source_files"), "baseline sources")
        }
        for raw_record in _sequence(baseline.get("source_files"), "source files"):
            record = _mapping(raw_record, "source file")
            _exact_keys(record, {"path", "sha256"}, "source file")
            path = _safe_relative_path(record["path"], "source file path")
            digest = _require_sha256(record["sha256"], "source file SHA256")
            actual = self.repository_root.joinpath(*PurePosixPath(path).parts)
            if not actual.is_file() or actual.is_symlink() or sha256_file(actual) != digest:
                raise ValueError(f"baseline source identity drifted: {path}")
            if path in manifest_sources and manifest_sources[path] != digest:
                raise ValueError(f"baseline manifest/source mismatch: {path}")

    def _validate_binding(
        self,
        record: Mapping[str, Any],
        path_key: str,
        sha_key: str,
    ) -> dict[str, Any]:
        path = _safe_relative_path(record.get(path_key), path_key)
        return self._read_bound_json(path, record.get(sha_key), label=path)

    def _read_bound_json(
        self,
        relative_path: str,
        expected_sha256: Any,
        *,
        label: str,
    ) -> dict[str, Any]:
        digest = _require_sha256(expected_sha256, f"{label} SHA256")
        path = self.repository_root.joinpath(*PurePosixPath(relative_path).parts)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise ValueError(f"bound JSON identity drifted: {relative_path}")
        return strict_json_object_bytes(path.read_bytes(), label=label)


def validate_contract(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    contract = RestorationV22OcrRgbContract.load(
        path,
        repository_root=repository_root,
        validate_bound_sources=True,
    )
    return {
        "status": PASS_STATUS,
        "protocol_id": PROTOCOL_ID,
        "config_sha256": contract.sha256,
        "geometry_result_git_commit": contract.data["immutable_inputs"][
            "selector_geometry_result"
        ]["git_commit"],
        "expected_state_count": 15,
        "expected_candidate_comparison_count": 60,
        "expected_unique_image_count": 75,
        "cpu_only": True,
        "gpu_count": 0,
        "ocr_inference_count": 0,
        "policy_forward_count": 0,
        "confirm_locked": True,
        "sealed_test_locked": True,
        "formal_aggregate_generated": False,
    }
