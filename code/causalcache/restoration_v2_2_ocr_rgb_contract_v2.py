"""Fail-closed contract for the OCR/RGB v2 identity-scanner repair."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.restoration_v2_2_ocr_rgb_contract import (
    CANONICAL_CONFIG_PATH as PARENT_CONFIG_PATH,
    EXPECTED_OUTPUT_FILES,
    FROZEN_CONFIG_SHA256 as PARENT_CONFIG_SHA256,
    PROTOCOL_ID as PARENT_PROTOCOL_ID,
    RestorationV22OcrRgbContract,
    sha256_bytes,
    sha256_file,
    strict_json_object_bytes,
)


PROTOCOL_ID = (
    "causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair"
)
CANONICAL_CONFIG_PATH = (
    "code/configs/"
    "causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.json"
)
FROZEN_CONFIG_SHA256 = (
    "d68cb032ef3c56c330d57329507d409b20f878b7510bd09d88e2eff1f3f898f3"
)
PASS_STATUS = "VALID_RESTORATION_V2_2_OCR_RGB_BASELINE_V2_IDENTITY_REPAIR"
PARENT_SOURCE_GIT_COMMIT = "aa5898f25e2e7d647363fe701ac90134bf744a5c"
FAILURE_GIT_COMMIT = "2870d8ae26542a184647e8b6d97b8c79e4e12641"
FAILURE_DIRECTORY = (
    "data/results/restoration_v2_2_ocr_rgb_baseline_v1_attempt"
)
V1_CANONICAL_RESULT_DIRECTORY = (
    "data/results/restoration_v2_2_ocr_rgb_baseline_v1"
)
V2_CANONICAL_RESULT_DIRECTORY = (
    "data/results/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair"
)
TRAJECTORY_IDENTITY_OCCURRENCES_PER_LINE = 2
OCR_IDENTITY_OCCURRENCES_PER_LINE = 1
SHARED_REDUCER_PATH = "code/causalcache/restoration_v2_2_ocr_rgb.py"
PARENT_SHARED_REDUCER_SHA256 = (
    "65e9f8e0f0769f88245652f6567182bcb6724b7143c7e34c8ff251584c91120b"
)
REPAIR_SHARED_REDUCER_SHA256 = (
    "db276c6f7ad6f5d29b63d7cabd66b86bd0e6c65df7715d022727726cb5eca6a0"
)
PARENT_TO_REPAIR_SOURCE_CHANGED_PATHS = (
    "README.md",
    "code/README.md",
    SHARED_REDUCER_PATH,
    "code/causalcache/restoration_v2_2_ocr_rgb_contract_v2.py",
    CANONICAL_CONFIG_PATH,
    "code/scripts/run_restoration_v2_2_ocr_rgb_baseline_v2.py",
    "code/scripts/validate_restoration_v2_2_ocr_rgb_contract_v2.py",
    "code/tests/test_restoration_v2_2_ocr_rgb_v2.py",
    f"{FAILURE_DIRECTORY}/README.md",
    f"{FAILURE_DIRECTORY}/summary.json",
    "docs/progress.md",
    "docs/restoration_v2_2_ocr_rgb_baseline.md",
    "docs/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.md",
)

EXPECTED_PARENT = {
    "path": PARENT_CONFIG_PATH,
    "sha256": PARENT_CONFIG_SHA256,
    "protocol_id": PARENT_PROTOCOL_ID,
    "source_git_commit": PARENT_SOURCE_GIT_COMMIT,
}
EXPECTED_FAILURE_FILES = (
    {
        "path": "README.md",
        "size_bytes": 3881,
        "sha256": (
            "9b3b6d2eab20d997bd26f12845de9ba7"
            "849b7fa0f5905f501fdb3b9179b9e6be"
        ),
    },
    {
        "path": "summary.json",
        "size_bytes": 4798,
        "sha256": (
            "fa422e4c71f9d25af4359ac222cadfe2"
            "49d952c4a626685f787595077b686b14"
        ),
    },
)
EXPECTED_FAILED_ATTEMPT = {
    "directory": FAILURE_DIRECTORY,
    "git_commit": FAILURE_GIT_COMMIT,
    "protocol_id": PARENT_PROTOCOL_ID,
    "status": "OCR_RGB_BASELINE_ATTEMPT_INVALID",
    "outcome": "INVALID_RESTORATION_V2_2_OCR_RGB_BASELINE_V1",
    "error_class": "ValueError",
    "error_message": "derived trajectories line 0 has invalid identity encoding",
    "formal_state_score_row_count": 0,
    "scientific_payload_generated": False,
    "exact_files": list(EXPECTED_FAILURE_FILES),
}
EXPECTED_REPAIR_SCOPE = {
    "identity_scanner_only": True,
    "parent_scientific_contract_changed": False,
    "immutable_inputs_changed": False,
    "feature_contract_changed": False,
    "selection_contract_changed": False,
    "statistics_contract_changed": False,
    "operation_ceiling_changed": False,
    "trajectory_identity": {
        "file": (
            "derived/restoration-v2-v1/"
            "trajectories-00000-of-00001.jsonl"
        ),
        "identity_field": "source_id",
        "expected_total_line_count": 35,
        "expected_occurrences_per_line": (
            TRAJECTORY_IDENTITY_OCCURRENCES_PER_LINE
        ),
        "all_occurrences_must_decode_to_same_utf8_value": True,
        "escaped_identity_allowed": False,
        "nonselected_record_semantic_parse_allowed": False,
    },
    "ocr_identity": {
        "file": (
            "derived/restoration-v2-v1/"
            "ocr-records-00000-of-00001.jsonl"
        ),
        "identity_field": "image_member_path",
        "expected_total_line_count": 210,
        "expected_occurrences_per_line": OCR_IDENTITY_OCCURRENCES_PER_LINE,
        "all_occurrences_must_decode_to_same_utf8_value": True,
        "escaped_identity_allowed": False,
        "nonselected_record_semantic_parse_allowed": False,
    },
    "selected_record_identity_post_parse_crosscheck": True,
    "zero_occurrence_fails_closed": True,
    "occurrence_count_drift_fails_closed": True,
    "inconsistent_occurrence_value_fails_closed": True,
    "v1_default_identity_scanner_semantics_preserved": True,
    "shared_reducer_source": {
        "path": SHARED_REDUCER_PATH,
        "parent_source_sha256": PARENT_SHARED_REDUCER_SHA256,
        "repair_source_sha256": REPAIR_SHARED_REDUCER_SHA256,
    },
    "parent_to_repair_source_changed_paths": list(
        PARENT_TO_REPAIR_SOURCE_CHANGED_PATHS
    ),
}
EXPECTED_OUTPUT = {
    "canonical_result_directory": V2_CANONICAL_RESULT_DIRECTORY,
    "exact_files": list(EXPECTED_OUTPUT_FILES),
    "invalid_v1_canonical_result_directory": V1_CANONICAL_RESULT_DIRECTORY,
    "invalid_v1_canonical_result_and_staging_must_remain_absent": True,
    "failed_attempt_directory_must_remain_byte_unchanged": True,
    "raw_labels_images_ocr_or_histograms_copied_to_git": False,
    "formal_run_requires_clean_pushed_main": True,
    "run_requires_exclusive_new_output_directory": True,
    "validate_rebuilds_every_output_byte_from_immutable_inputs": True,
    "result_status_before_execution": "not_generated",
}


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


def _git_bytes(root: Path, revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _validate_commit(root: Path, revision: str) -> None:
    subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _validate_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _validate_exact_failure_files(root: Path) -> None:
    directory = root.joinpath(*PurePosixPath(FAILURE_DIRECTORY).parts)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("v1 failure directory is missing or symlinked")
    entries = tuple(directory.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise ValueError("v1 failure directory may contain only regular files")
    observed = sorted(path.name for path in entries)
    expected = [record["path"] for record in EXPECTED_FAILURE_FILES]
    if observed != sorted(expected):
        raise ValueError("v1 failure exact file inventory drifted")
    for record in EXPECTED_FAILURE_FILES:
        path = directory / record["path"]
        if not path.is_file() or path.is_symlink():
            raise ValueError("v1 failure record is missing or symlinked")
        payload = path.read_bytes()
        if (
            len(payload) != record["size_bytes"]
            or sha256_bytes(payload) != record["sha256"]
            or _git_bytes(
                root,
                FAILURE_GIT_COMMIT,
                f"{FAILURE_DIRECTORY}/{record['path']}",
            )
            != payload
        ):
            raise ValueError(f"v1 failure record identity drifted: {record['path']}")


def _validate_failure_summary(root: Path) -> None:
    summary = strict_json_object_bytes(
        (root / FAILURE_DIRECTORY / "summary.json").read_bytes(),
        label="OCR/RGB v1 failure summary",
    )
    expected_scalars = {
        "protocol_id": PARENT_PROTOCOL_ID,
        "git_source_commit": PARENT_SOURCE_GIT_COMMIT,
        "contract_sha256": PARENT_CONFIG_SHA256,
        "status": "OCR_RGB_BASELINE_ATTEMPT_INVALID",
        "outcome": "INVALID_RESTORATION_V2_2_OCR_RGB_BASELINE_V1",
        "artifact_status": "implementation_invalid",
        "valid_for_scientific_result": False,
        "scientific_conclusion": None,
        "retry_allowed": False,
        "resume_allowed": False,
        "top_up_count": 0,
        "attempt_ledger_created": False,
    }
    if any(summary.get(key) != value for key, value in expected_scalars.items()):
        raise ValueError("v1 failure summary scalar identity drifted")
    failure = _mapping(summary.get("failure"), "v1 failure")
    if dict(failure) != {
        "category": "identity_lexer_repeated_equal_field_mismatch",
        "class": "ValueError",
        "failed_implementation_assumption": (
            "exactly_one_identity_field_occurrence_per_jsonl_line"
        ),
        "immutable_input_encoding": (
            "compact_json_with_equal_top_level_and_nested_source_id_occurrences"
        ),
        "message": "derived trajectories line 0 has invalid identity encoding",
        "stage": (
            "selected_trajectory_identity_scan_before_feature_materialization"
        ),
    }:
        raise ValueError("v1 failure diagnosis drifted")
    outputs = _mapping(summary.get("formal_outputs"), "v1 formal outputs")
    if (
        any(
            outputs.get(key) != 0
            for key in (
                "aggregate_count",
                "combined_similarity_score_count",
                "match_rate_value_count",
                "ocr_similarity_score_count",
                "recovery_value_count",
                "rgb_similarity_score_count",
                "selected_coalition_distance_lookup_count",
                "state_score_row_count",
            )
        )
        or outputs.get("scientific_payload_generated") is not False
        or outputs.get("canonical_output_directory_exists") is not False
        or outputs.get("staging_directory_exists") is not False
    ):
        raise ValueError("v1 zero-score output boundary drifted")
    negative = _mapping(
        summary.get("negative_operation_counts"),
        "v1 negative operations",
    )
    if any(value != 0 for value in negative.values()):
        raise ValueError("v1 prohibited operation count drifted")
    forensics = _mapping(
        summary.get("post_failure_identity_lexer_forensics"),
        "v1 identity forensics",
    )
    expected_forensics = {
        "ocr_records": {
            "all_values_equal_within_line": True,
            "identity_field": "image_member_path",
            "line_count": 210,
            "occurrence_count_histogram": {"1": 210},
            "semantic_parse_count": 0,
            "unique_identity_count": 210,
        },
        "trajectories": {
            "all_values_equal_within_line": True,
            "identity_field": "source_id",
            "line_count": 35,
            "occurrence_count_histogram": {"2": 35},
            "semantic_parse_count": 0,
            "unique_identity_count": 35,
        },
    }
    if dict(forensics) != expected_forensics:
        raise ValueError("v1 identity forensic profile drifted")
    replacement = _mapping(
        summary.get("replacement_constraints"),
        "v1 replacement constraints",
    )
    if dict(replacement) != {
        "exact_per_file_identity_occurrence_repair_only": True,
        "immutable_inputs_unchanged": True,
        "new_attempt_identity_required": True,
        "scientific_contract_unchanged": True,
    }:
        raise ValueError("v1 replacement boundary drifted")


@dataclass(frozen=True)
class RestorationV22OcrRgbIdentityRepairContract:
    path: Path
    data: Mapping[str, Any]
    sha256: str
    parent: RestorationV22OcrRgbContract
    repository_root: Path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
    ) -> "RestorationV22OcrRgbIdentityRepairContract":
        root = Path(repository_root).resolve()
        resolved = Path(path).resolve()
        canonical = (root / CANONICAL_CONFIG_PATH).resolve()
        if resolved != canonical or not resolved.is_file() or resolved.is_symlink():
            raise ValueError("identity-repair contract must use the canonical config")
        digest = sha256_file(resolved)
        if digest != FROZEN_CONFIG_SHA256:
            raise ValueError("identity-repair frozen config SHA256 drifted")
        parent = RestorationV22OcrRgbContract.load(
            root / PARENT_CONFIG_PATH,
            repository_root=root,
            validate_bound_sources=True,
        )
        contract = cls(
            path=resolved,
            data=strict_json_object_bytes(
                resolved.read_bytes(),
                label="OCR/RGB identity-repair contract",
            ),
            sha256=digest,
            parent=parent,
            repository_root=root,
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        _exact_keys(
            self.data,
            {
                "schema_version",
                "protocol_id",
                "repair_status",
                "parent_contract",
                "failed_attempt",
                "repair_scope",
                "authorization",
                "output_contract",
            },
            "OCR/RGB identity-repair contract",
        )
        if (
            self.data["schema_version"] != "0.2.0"
            or self.data["protocol_id"] != PROTOCOL_ID
            or self.data["repair_status"]
            != "source_only_frozen_before_any_v2_identity_repair_aggregate"
        ):
            raise ValueError("identity-repair protocol identity drifted")
        if dict(_mapping(self.data["parent_contract"], "parent contract")) != (
            EXPECTED_PARENT
        ):
            raise ValueError("identity-repair parent contract drifted")
        if dict(_mapping(self.data["failed_attempt"], "failed attempt")) != (
            EXPECTED_FAILED_ATTEMPT
        ):
            raise ValueError("identity-repair failed-attempt binding drifted")
        if dict(_mapping(self.data["repair_scope"], "repair scope")) != (
            EXPECTED_REPAIR_SCOPE
        ):
            raise ValueError("identity-repair scope drifted")
        if self.data["authorization"] != self.parent.data["authorization"]:
            raise ValueError("identity-repair authorization differs from parent")
        if dict(_mapping(self.data["output_contract"], "output contract")) != (
            EXPECTED_OUTPUT
        ):
            raise ValueError("identity-repair output contract drifted")
        if self.parent.sha256 != PARENT_CONFIG_SHA256:
            raise ValueError("identity-repair parent SHA256 drifted")

        _validate_commit(self.repository_root, PARENT_SOURCE_GIT_COMMIT)
        _validate_commit(self.repository_root, FAILURE_GIT_COMMIT)
        _validate_ancestor(
            self.repository_root,
            PARENT_SOURCE_GIT_COMMIT,
            FAILURE_GIT_COMMIT,
        )
        parent_source_bytes = _git_bytes(
            self.repository_root,
            PARENT_SOURCE_GIT_COMMIT,
            PARENT_CONFIG_PATH,
        )
        if (
            sha256_bytes(parent_source_bytes) != PARENT_CONFIG_SHA256
            or parent_source_bytes != self.parent.path.read_bytes()
        ):
            raise ValueError("identity-repair parent source bytes drifted")
        parent_reducer_bytes = _git_bytes(
            self.repository_root,
            PARENT_SOURCE_GIT_COMMIT,
            SHARED_REDUCER_PATH,
        )
        repair_reducer_path = self.repository_root / SHARED_REDUCER_PATH
        if (
            sha256_bytes(parent_reducer_bytes) != PARENT_SHARED_REDUCER_SHA256
            or not repair_reducer_path.is_file()
            or repair_reducer_path.is_symlink()
            or sha256_file(repair_reducer_path) != REPAIR_SHARED_REDUCER_SHA256
        ):
            raise ValueError("identity-repair shared reducer source drifted")
        _validate_exact_failure_files(self.repository_root)
        _validate_failure_summary(self.repository_root)

        v1_output = self.repository_root / V1_CANONICAL_RESULT_DIRECTORY
        if v1_output.exists() or v1_output.with_name(v1_output.name + ".tmp").exists():
            raise ValueError("invalid v1 canonical output or staging must remain absent")


def validate_contract(
    path: str | Path,
    *,
    repository_root: str | Path,
    require_output_absent: bool = False,
) -> dict[str, Any]:
    contract = RestorationV22OcrRgbIdentityRepairContract.load(
        path,
        repository_root=repository_root,
    )
    output = contract.repository_root / V2_CANONICAL_RESULT_DIRECTORY
    output_exists = output.exists()
    staging_exists = output.with_name(output.name + ".tmp").exists()
    if require_output_absent and (output_exists or staging_exists):
        raise ValueError("source-only validation requires absent v2 output and staging")
    return {
        "status": PASS_STATUS,
        "protocol_id": PROTOCOL_ID,
        "config_sha256": contract.sha256,
        "parent_config_sha256": contract.parent.sha256,
        "parent_source_git_commit": PARENT_SOURCE_GIT_COMMIT,
        "failed_attempt_git_commit": FAILURE_GIT_COMMIT,
        "trajectory_line_count": 35,
        "trajectory_identity_occurrences_per_line": (
            TRAJECTORY_IDENTITY_OCCURRENCES_PER_LINE
        ),
        "ocr_line_count": 210,
        "ocr_identity_occurrences_per_line": OCR_IDENTITY_OCCURRENCES_PER_LINE,
        "scientific_contract_changed": False,
        "formal_aggregate_generated": output_exists,
        "staging_output_exists": staging_exists,
        "confirm_locked": True,
        "sealed_test_locked": True,
    }
