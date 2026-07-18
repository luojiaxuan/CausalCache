"""Fail-closed Source-A contract for long-horizon development selection."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_long_horizon_development_v1"
SOURCE_STATUS = (
    "source_a_placeholder_frozen_before_long_horizon_selection_or_semantic_access"
)
VALIDATION_STATUS = "VALID_CAUSALCACHE_LONG_HORIZON_SOURCE_A_V1"
CANONICAL_CONFIG_PATH = "code/configs/causalcache_long_horizon_development_v1.json"
RUNNER_FREEZE_B_PATH = (
    "code/configs/causalcache_long_horizon_development_runner_v1.json"
)
SELECTION_MANIFEST_FREEZE_PATH = (
    "data/manifests/causalcache_long_horizon_development_v1_selection.json"
)
FROZEN_CONFIG_SHA256 = (
    "82965790bf541dea348997ecd1849473d5cfb89f53da9f5a615568f57ca67a96"
)
SOURCE_A_INVENTORY_PLACEHOLDER = "0" * 64
SELECTION_SALT = "causalcache-long-horizon-development-v1"

REQUIRED_SOURCE_A_PATHS = (
    "code/causalcache/long_horizon_contract.py",
    "code/causalcache/long_horizon_data.py",
    "code/causalcache/long_horizon_evaluation.py",
    "code/causalcache/long_horizon_execution.py",
    "code/causalcache/long_horizon_prepare.py",
    "code/causalcache/long_horizon_runtime.py",
    "code/causalcache/long_horizon_selection.py",
    "code/causalcache/long_horizon_selectors.py",
    "code/causalcache/long_horizon_v4.py",
    "code/causalcache/policy/gui_owl_v2.py",
    "code/causalcache/policy/gui_owl_v2_1.py",
    CANONICAL_CONFIG_PATH,
    "code/scripts/build_long_horizon_selection.py",
    "code/scripts/build_long_horizon_selector_seal.py",
    "code/scripts/build_long_horizon_substrate.py",
    "code/scripts/manage_long_horizon_execution.py",
    "code/scripts/run_long_horizon_worker.py",
    "code/scripts/validate_long_horizon_contract.py",
    "code/scripts/validate_long_horizon_selector_seal.py",
    "code/scripts/validate_long_horizon_substrate.py",
    "code/tests/test_gui_owl_v2_1.py",
    "code/tests/test_gui_owl_v2_prompt.py",
    "code/tests/test_long_horizon_contract.py",
    "code/tests/test_long_horizon_data.py",
    "code/tests/test_long_horizon_evaluation.py",
    "code/tests/test_long_horizon_execution.py",
    "code/tests/test_long_horizon_prepare.py",
    "code/tests/test_long_horizon_runtime.py",
    "code/tests/test_long_horizon_selection.py",
    "code/tests/test_long_horizon_selectors.py",
    "code/tests/test_long_horizon_v4.py",
)

TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "source_pool",
    "substrate_profile",
    "historical_role_firewall",
    "selection",
    "state_geometry",
    "selector_matrix",
    "reference_contract",
    "validity_and_go_contract",
    "policy_context_profile",
    "operation_accounting",
    "execution_b_plan",
    "artifact_plan",
    "learned_model_artifacts",
    "access_firewall",
    "source_freeze",
    "authorization",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
LEARNED_MODEL_ARTIFACTS_SHA256 = (
    "004e85603e0ddb028bf9b3f0b5c4b454b8bc05fb6ff87f84e3f85c7086367f06"
)


@dataclass(frozen=True)
class LongHorizonContract:
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path

    @classmethod
    def load(
        cls,
        path: str | Path = CANONICAL_CONFIG_PATH,
        *,
        repository_root: str | Path = ".",
        require_runner_absent: bool = True,
    ) -> "LongHorizonContract":
        root = Path(repository_root).resolve()
        source_path = _resolve_regular(root, path, "long-horizon contract")
        payload = source_path.read_bytes()
        digest = sha256_bytes(payload)
        if digest != FROZEN_CONFIG_SHA256:
            raise ValueError(
                "long-horizon contract bytes drifted: "
                f"expected {FROZEN_CONFIG_SHA256}, got {digest}"
            )
        data = load_strict_json_bytes(payload, label="long-horizon contract")
        validate_contract(data, repository_root=root)
        runner = root / RUNNER_FREEZE_B_PATH
        selection = root / SELECTION_MANIFEST_FREEZE_PATH
        if require_runner_absent and (
            os.path.lexists(runner) or os.path.lexists(selection)
        ):
            raise ValueError(
                "selection and Execution-B freezes must be absent during Source-A"
            )
        return cls(data=data, sha256=digest, repository_root=root, source_path=source_path)

    @property
    def source_pool(self) -> Mapping[str, Any]:
        return _mapping(self.data["source_pool"], "source pool")

    @property
    def historical_firewall(self) -> Mapping[str, Any]:
        return _mapping(self.data["historical_role_firewall"], "historical firewall")

    @property
    def selection(self) -> Mapping[str, Any]:
        return _mapping(self.data["selection"], "selection")

    @property
    def geometry(self) -> Mapping[str, Any]:
        return _mapping(self.data["state_geometry"], "state geometry")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_strict_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def load_strict_json(path: Path, *, label: str) -> dict[str, Any]:
    return load_strict_json_bytes(path.read_bytes(), label=label)


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
            f"{label} keys drifted: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} drifted: expected {expected!r}, got {actual!r}")


def _strict_int(value: Any, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} must be integer {expected}")


def _strict_bool(value: Any, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{label} must be {expected}")


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _resolve_regular(root: Path, relative: str | Path, label: str) -> Path:
    value = _relative_path(str(relative), label)
    path = root.joinpath(*PurePosixPath(value).parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if root not in path.resolve().parents:
        raise ValueError(f"{label} escapes repository root")
    return path


def source_ids_sha256(source_ids: Sequence[str]) -> str:
    return sha256_bytes(canonical_json_bytes(sorted(set(source_ids))))


def _extract_splits_source_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    splits = _mapping(manifest.get("splits"), "artifact splits")
    result: list[str] = []
    for role_name in sorted(splits):
        role = _mapping(splits[role_name], f"artifact split {role_name}")
        ids = _sequence(role.get("source_ids"), f"artifact split {role_name} IDs")
        if any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("artifact split source IDs must be non-empty strings")
        if "trajectory_count" in role:
            _strict_int(
                role["trajectory_count"],
                len(ids),
                f"artifact split {role_name} trajectory count",
            )
        result.extend(ids)
    return tuple(result)


def _extract_trajectory_roles(
    manifest: Mapping[str, Any], *, section_name: str
) -> tuple[str, ...]:
    roles = _mapping(manifest.get(section_name), f"manifest {section_name}")
    result: list[str] = []
    for role_name in sorted(roles):
        role = _mapping(roles[role_name], f"{section_name} role {role_name}")
        trajectories = _sequence(
            role.get("trajectories"), f"{section_name} role {role_name} trajectories"
        )
        role_ids: list[str] = []
        for record in trajectories:
            source_id = _mapping(record, "trajectory record").get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("trajectory record source_id must be non-empty text")
            role_ids.append(source_id)
        if len(role_ids) != len(set(role_ids)):
            raise ValueError(f"{section_name} role {role_name} contains duplicate IDs")
        result.extend(role_ids)
    return tuple(result)


def extract_historical_role_source_ids(
    manifest: Mapping[str, Any], *, extractor: str
) -> tuple[str, ...]:
    if extractor == "splits_source_ids":
        return _extract_splits_source_ids(manifest)
    if extractor == "roles_trajectory_source_ids":
        return _extract_trajectory_roles(manifest, section_name="roles")
    if extractor == "splits_trajectory_source_ids":
        return _extract_trajectory_roles(manifest, section_name="splits")
    raise ValueError(f"unsupported historical role extractor: {extractor}")


def historical_role_inventory(
    contract: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    firewall = _mapping(contract["historical_role_firewall"], "historical firewall")
    records = _sequence(firewall["manifests"], "historical manifests")
    manifests: list[dict[str, Any]] = []
    union: set[str] = set()
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"historical manifest {index}")
        path_text = _relative_path(record["path"], "historical manifest path")
        path = _resolve_regular(repository_root, path_text, "historical manifest")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"historical manifest bytes drifted: {path_text}")
        ids = extract_historical_role_source_ids(
            load_strict_json(path, label=path_text),
            extractor=str(record["extractor"]),
        )
        unique = tuple(sorted(set(ids)))
        _strict_int(
            record["expected_source_id_count"],
            len(unique),
            f"historical manifest {path_text} source-ID count",
        )
        _equal(
            record["expected_source_ids_sha256"],
            source_ids_sha256(unique),
            f"historical manifest {path_text} source-ID digest",
        )
        union.update(unique)
        manifests.append(
            {
                "path": path_text,
                "sha256": record["sha256"],
                "extractor": record["extractor"],
                "source_id_count": len(unique),
                "source_ids_sha256": source_ids_sha256(unique),
                "source_ids": list(unique),
            }
        )
    union_ids = tuple(sorted(union))
    _strict_int(
        firewall["expected_union_source_id_count"],
        len(union_ids),
        "historical union source-ID count",
    )
    _equal(
        firewall["expected_union_source_ids_sha256"],
        source_ids_sha256(union_ids),
        "historical union source-ID digest",
    )
    return {
        "manifests": manifests,
        "union_source_id_count": len(union_ids),
        "union_source_ids_sha256": source_ids_sha256(union_ids),
        "union_source_ids": list(union_ids),
    }


def _validate_source_pool(data: Mapping[str, Any], root: Path) -> None:
    pool = _mapping(data["source_pool"], "source pool")
    _exact_keys(
        pool,
        {
            "parent_protocol_config",
            "source_file_manifest",
            "transport_files_must_equal_parent_exactly",
            "structural_eligibility",
        },
        "source pool",
    )
    parent = _mapping(pool["parent_protocol_config"], "parent protocol")
    parent_path = _resolve_regular(root, parent["path"], "parent protocol")
    _equal(sha256_file(parent_path), parent["sha256"], "parent protocol SHA256")
    parent_data = load_strict_json(parent_path, label="parent protocol")
    _equal(parent_data.get("protocol_id"), parent["protocol_id"], "parent protocol ID")

    source = _mapping(pool["source_file_manifest"], "source file manifest")
    source_path = _resolve_regular(root, source["path"], "source file manifest")
    _equal(sha256_file(source_path), source["sha256"], "source manifest SHA256")
    source_data = load_strict_json(source_path, label="source file manifest")
    _strict_int(source["file_count"], len(source_data.get("files", [])), "file count")
    _equal(source_data.get("repo"), source["repo"], "source repo")
    _equal(source_data.get("revision"), source["revision"], "source revision")
    _equal(parent_data["source_pool"]["transport_repo"], source["repo"], "parent repo")
    _equal(
        parent_data["source_pool"]["transport_revision"],
        source["revision"],
        "parent revision",
    )
    _equal(
        [item["path"] for item in source_data["files"]],
        parent_data["source_pool"]["transport_files"],
        "pinned 16 source files",
    )
    _strict_bool(pool["transport_files_must_equal_parent_exactly"], True, "source equality")

    eligibility = _mapping(pool["structural_eligibility"], "eligibility")
    _strict_int(eligibility["minimum_decisions_per_trajectory"], 18, "minimum decisions")
    _strict_int(eligibility["maximum_decisions_per_trajectory"], 60, "maximum decisions")
    _strict_int(eligibility["expected_eligible_trajectory_count"], 42, "eligible count")
    _strict_bool(
        eligibility[
            "all_other_source_platform_terminal_action_parser_app_and_embedded_image_checks_inherited_exactly"
        ],
        True,
        "full structural inheritance",
    )
    _strict_bool(
        eligibility["open_ended_exception_filtering_allowed"],
        False,
        "open-ended filtering",
    )

    substrate = _mapping(data["substrate_profile"], "substrate profile")
    for key, expected_path, expected_sha in (
        (
            "ocr_backend_config",
            "code/configs/restoration_v2_ocr_backend.json",
            "51e08a9565f804ecb1ee7f12887cc12742b84c04996fffbbde7c0f304e4bd036",
        ),
        (
            "ocr_backend_manifest",
            "data/manifests/restoration_v2_ocr_backend.json",
            "107478672438b52e9c2ccc9ef8de5d13d4e16329ab1afbe95772e2d3df5bd2a9",
        ),
    ):
        record = _mapping(substrate[key], key)
        _equal(record["path"], expected_path, f"{key} path")
        _equal(record["sha256"], expected_sha, f"{key} SHA256")
        _equal(
            sha256_file(_resolve_regular(root, expected_path, key)),
            expected_sha,
            f"{key} live SHA256",
        )
    for key, expected in (
        ("development_trajectory_count", 24),
        ("image_member_count", 432),
        ("ocr_record_count", 432),
        ("event_count", 408),
        ("state_count", 48),
        ("candidate_occurrence_count", 576),
        ("reserve_python_row_materialization_count", 0),
        ("reserve_semantic_content_access_count", 0),
        ("reserve_ocr_or_image_access_count", 0),
    ):
        _strict_int(substrate[key], expected, key)
    _strict_bool(
        substrate["full_ocr_replay_required_before_upload"],
        True,
        "full OCR replay",
    )
    _strict_bool(
        substrate["source_transport_row_group_overread_possible"],
        True,
        "source row-group overread disclosure",
    )


def _validate_selection_and_geometry(data: Mapping[str, Any]) -> None:
    selection = _mapping(data["selection"], "selection")
    _equal(selection["trajectory_salt"], SELECTION_SALT, "selection salt")
    _equal(
        selection["trajectory_hash_input"],
        "UTF8(salt + NUL + source_id)",
        "hash input",
    )
    _equal(
        selection["trajectory_order"],
        ["sha256_hex", "source_id", "transport_file", "transport_row_index"],
        "trajectory order",
    )
    _strict_int(selection["development_trajectory_count"], 24, "development count")
    _strict_int(selection["unopened_reserve_trajectory_count"], 18, "reserve count")
    for key in (
        "padding_allowed",
        "trajectory_splicing_allowed",
        "post_output_top_up_allowed",
        "post_selection_filtering_allowed",
        "reserve_policy_ocr_restoration_or_selector_access_allowed",
    ):
        _strict_bool(selection[key], False, key)

    geometry = _mapping(data["state_geometry"], "state geometry")
    _strict_int(geometry["development_states_per_trajectory"], 2, "states per trajectory")
    _equal(
        geometry["states"],
        [
            {
                "decision_step_id": 10,
                "candidate_event_count": 8,
                "candidate_event_step_ids": list(range(1, 9)),
                "current_equivalent_event_step_id": 9,
            },
            {
                "decision_step_id": 18,
                "candidate_event_count": 16,
                "candidate_event_step_ids": list(range(1, 17)),
                "current_equivalent_event_step_id": 17,
            },
        ],
        "long-horizon state geometry",
    )
    for key in (
        "history_is_strict_unspliced_prefix",
        "candidate_archive_padding_allowed",
        "reserve_state_materialization_allowed",
    ):
        expected = key == "history_is_strict_unspliced_prefix"
        _strict_bool(geometry[key], expected, key)


def _validate_selector_and_reference_contract(data: Mapping[str, Any]) -> None:
    matrix = _mapping(data["selector_matrix"], "selector matrix")
    _equal(
        matrix["budget_2"],
        [
            "restoration_independent_gate",
            "recent",
            "ocr_rgb_v2",
            "v1_conditional",
            "v4_safe_frozen_base_residual",
            "random",
            "summary_only",
        ],
        "B2 selectors",
    )
    _equal(
        matrix["budget_4"],
        [
            "restoration_independent_gate",
            "recent",
            "ocr_rgb_v2",
            "random",
            "summary_only",
        ],
        "B4 selectors",
    )
    _equal(matrix["random_seed_roster"], [271828], "random seed roster")
    _strict_bool(
        matrix["random_seed_roster_frozen_in_source_a"],
        True,
        "random seed freeze",
    )
    _strict_int(matrix["summary_only_high_fidelity_event_count"], 0, "summary-only budget")
    _strict_bool(
        matrix["selector_sets_must_be_label_blind_sealed_before_any_restoration_distance"],
        True,
        "selector label-blind seal",
    )

    reference = _mapping(data["reference_contract"], "reference contract")
    n8 = _mapping(reference["n8"], "n8 reference")
    _equal(n8["reference_type"], "exact_subset_oracle", "n8 reference type")
    _strict_bool(n8["evaluate_every_subset_with_cardinality_at_most_budget"], True, "n8 exhaustive")
    _strict_int(n8["budget_2_subset_count"], 37, "n8 B2 subset count")
    _strict_int(n8["budget_4_subset_count"], 163, "n8 B4 subset count")
    _strict_bool(n8["global_optimum_claim_allowed_within_frozen_candidate_archive_and_budget"], True, "n8 oracle claim")

    n16 = _mapping(reference["n16"], "n16 reference")
    _equal(n16["reference_type"], "frozen_selector_pair_union", "n16 reference type")
    _strict_bool(n16["global_oracle_or_search_claim_allowed"], False, "n16 oracle claim")
    _strict_bool(n16["enumeration_or_shortlist_allowed"], False, "n16 search")
    _equal(
        n16["budget_2_comparator_pairs"],
        [
            ["restoration_independent_gate", "recent"],
            ["restoration_independent_gate", "ocr_rgb_v2"],
            ["v1_conditional", "restoration_independent_gate"],
            ["v4_safe_frozen_base_residual", "restoration_independent_gate"],
        ],
        "n16 B2 pairs",
    )
    _equal(
        n16["budget_4_comparator_pairs"],
        [
            ["restoration_independent_gate", "recent"],
            ["restoration_independent_gate", "ocr_rgb_v2"],
        ],
        "n16 B4 pairs",
    )
    _strict_int(n16["exact_subset_oracle_count"], 0, "n16 exact oracle count")


def _validate_validity_and_go_contract(data: Mapping[str, Any]) -> None:
    contract = _mapping(data["validity_and_go_contract"], "validity and GO")
    _exact_keys(
        contract,
        {
            "canonical_action_validity",
            "policy_coverage_thresholds",
            "fixed_denominator_reduction",
            "normalization",
            "set_aware_vs_independent_development_go",
            "independent_long_horizon_development_go",
        },
        "validity and GO",
    )
    validity = _mapping(
        contract["canonical_action_validity"], "canonical action validity"
    )
    _strict_int(validity["independent_parse_repetitions"], 2, "parse repetitions")
    for key in (
        "both_parses_must_succeed",
        "canonical_actions_must_be_exactly_identical",
        "n8_full_reference_applies",
        "each_n16_pair_union_reference_applies_independently",
        "invalid_counts_must_be_retained_by_state_geometry_and_comparator_pair",
    ):
        _strict_bool(validity[key], True, key)
    for key in (
        "invalid_trajectory_replacement_allowed",
        "invalid_trajectory_top_up_allowed",
    ):
        _strict_bool(validity[key], False, key)

    coverage = _mapping(
        contract["policy_coverage_thresholds"], "policy coverage thresholds"
    )
    _strict_int(coverage["selected_development_trajectory_denominator"], 24, "coverage denominator")
    _strict_int(coverage["minimum_valid_n8_trajectories"], 18, "n8 valid minimum")
    _strict_int(
        coverage["minimum_valid_n16_trajectories_per_comparator_pair"],
        18,
        "n16 per-pair valid minimum",
    )
    _equal(
        coverage["failure_outcome"],
        "NO_GO_INSUFFICIENT_LONG_HORIZON_POLICY_COVERAGE",
        "coverage failure outcome",
    )
    _strict_bool(coverage["overall_go_claim_allowed_below_any_threshold"], False, "coverage GO")

    fixed = _mapping(
        contract["fixed_denominator_reduction"], "fixed-denominator reduction"
    )
    _exact_keys(
        fixed,
        {
            "selected_trajectory_denominator",
            "invalid_state_contribution",
            "n8_paired_delta_zero_fill",
            "n16_pair_delta_zero_fill",
            "n8_exact_utility_ratio_zero_fill",
            "zero_exact_oracle_utility_ratio",
            "bootstrap_uses_all_selected_trajectories",
            "positive_support_uses_all_selected_trajectories",
            "valid_state_missing_or_duplicate_selector_outcome",
        },
        "fixed-denominator reduction",
    )
    _strict_int(
        fixed["selected_trajectory_denominator"],
        24,
        "fixed-denominator selected trajectory count",
    )
    if (
        type(fixed["invalid_state_contribution"]) is not float
        or fixed["invalid_state_contribution"] != 0.0
        or type(fixed["zero_exact_oracle_utility_ratio"]) is not float
        or fixed["zero_exact_oracle_utility_ratio"] != 0.0
    ):
        raise ValueError("fixed-denominator zero-fill values drifted")
    for key in (
        "n8_paired_delta_zero_fill",
        "n16_pair_delta_zero_fill",
        "n8_exact_utility_ratio_zero_fill",
        "bootstrap_uses_all_selected_trajectories",
        "positive_support_uses_all_selected_trajectories",
    ):
        _strict_bool(fixed[key], True, f"fixed-denominator {key}")
    _equal(
        fixed["valid_state_missing_or_duplicate_selector_outcome"],
        "INVALID_LONG_HORIZON_SELECTOR_JOIN",
        "fixed-denominator selector-join failure",
    )

    normalization = _mapping(contract["normalization"], "normalization")
    _equal(normalization["denominator"], "D(empty)", "normalization denominator")
    if (
        type(normalization["zero_threshold"]) is not float
        or normalization["zero_threshold"] != 1e-12
        or type(normalization["normalized_value_when_denominator_at_or_below_threshold"])
        is not float
        or normalization["normalized_value_when_denominator_at_or_below_threshold"]
        != 0.0
    ):
        raise ValueError("zero-denominator normalization rule drifted")
    _strict_bool(normalization["zero_denominator_record_must_be_retained"], True, "zero record retention")
    _strict_bool(normalization["zero_denominator_state_filtering_allowed"], False, "zero state filtering")

    development = _mapping(
        contract["set_aware_vs_independent_development_go"], "set-aware GO"
    )
    _strict_int(development["applies_to_budget"], 2, "set-aware GO budget")
    _equal(
        development["set_aware_selectors"],
        ["v1_conditional", "v4_safe_frozen_base_residual"],
        "set-aware GO selectors",
    )
    _equal(development["reference_selector"], "restoration_independent_gate", "GO reference")
    if (
        type(development["n8_trajectory_equal_mean_normalized_delta_minimum"])
        is not float
        or development["n8_trajectory_equal_mean_normalized_delta_minimum"] != 0.01
        or type(development["n8_paired_trajectory_bootstrap_confidence"])
        is not float
        or development["n8_paired_trajectory_bootstrap_confidence"] != 0.9
    ):
        raise ValueError("set-aware continuous GO thresholds drifted")
    _strict_int(development["n8_paired_trajectory_bootstrap_iterations"], 10000, "bootstrap iterations")
    _strict_int(development["n8_paired_trajectory_bootstrap_seed"], 20260718, "bootstrap seed")
    _strict_int(development["n8_minimum_positive_trajectories"], 14, "positive trajectories")
    _strict_int(development["n8_fixed_selected_trajectory_denominator"], 24, "GO denominator")
    for key in (
        "n8_trajectory_equal_mean_raw_delta_must_be_strictly_positive",
        "n8_paired_bootstrap_lower_bound_must_be_strictly_positive",
        "corresponding_n16_pair_mean_normalized_delta_must_be_strictly_positive",
        "all_n8_and_n16_conditions_are_jointly_necessary",
        "conditions_apply_separately_to_each_named_set_aware_selector",
        "overall_set_aware_go_requires_at_least_one_selector_to_pass_all_its_conditions",
    ):
        _strict_bool(development[key], True, key)
    _strict_bool(
        development["post_result_best_of_threshold_or_selector_redefinition_allowed"],
        False,
        "set-aware post-result redefinition",
    )
    _equal(
        development["failure_outcome"],
        "NO_GO_SET_AWARE_LONG_HORIZON_DEVELOPMENT_V1",
        "set-aware failure outcome",
    )

    independent = _mapping(
        contract["independent_long_horizon_development_go"],
        "independent long-horizon GO",
    )
    _strict_int(independent["primary_budget"], 4, "independent primary budget")
    _equal(
        independent["n8_exact_reference"],
        "exact_subset_oracle_with_cardinality_at_most_4",
        "independent n8 exact reference",
    )
    if (
        type(independent["n8_trajectory_equal_mean_raw_utility_ratio_to_exact_minimum"])
        is not float
        or independent[
            "n8_trajectory_equal_mean_raw_utility_ratio_to_exact_minimum"
        ]
        != 0.85
        or type(independent["n8_paired_trajectory_bootstrap_confidence"])
        is not float
        or independent["n8_paired_trajectory_bootstrap_confidence"] != 0.9
    ):
        raise ValueError("independent continuous GO thresholds drifted")
    _equal(
        independent["n8_comparators"],
        ["recent", "ocr_rgb_v2"],
        "independent n8 comparators",
    )
    _strict_int(independent["n8_paired_trajectory_bootstrap_iterations"], 10000, "independent bootstrap iterations")
    _strict_int(independent["n8_paired_trajectory_bootstrap_seed"], 20260718, "independent bootstrap seed")
    _strict_int(independent["n8_per_comparator_minimum_positive_trajectories"], 14, "independent positive trajectories")
    _strict_int(independent["n8_fixed_selected_trajectory_denominator"], 24, "independent denominator")
    _equal(
        independent["n16_budget_4_comparator_pairs"],
        [
            ["restoration_independent_gate", "recent"],
            ["restoration_independent_gate", "ocr_rgb_v2"],
        ],
        "independent n16 pairs",
    )
    for key in (
        "n8_per_comparator_trajectory_equal_mean_normalized_delta_must_be_strictly_positive",
        "n8_per_comparator_trajectory_equal_mean_raw_delta_must_be_strictly_positive",
        "n8_per_comparator_paired_bootstrap_lower_bound_must_be_strictly_positive",
        "n16_per_pair_mean_normalized_delta_must_be_strictly_positive",
        "n16_per_pair_mean_raw_delta_must_be_strictly_positive",
        "budget_2_is_secondary_report_only",
        "all_n8_and_n16_conditions_are_jointly_necessary",
    ):
        _strict_bool(independent[key], True, key)
    _equal(
        independent["pass_outcome"],
        "AUTHORIZE_NEW_CLOSED_LOOP_SOURCE_A_ONLY",
        "independent pass outcome",
    )
    _strict_bool(
        independent["old_closed_loop_or_test_direct_unlock_allowed"],
        False,
        "old closed-loop unlock",
    )
    _equal(
        independent["failure_outcome"],
        "NO_GO_LONG_HORIZON_INDEPENDENT_DEVELOPMENT_V1",
        "independent failure outcome",
    )


def _validate_policy_execution_and_artifact_plan(
    data: Mapping[str, Any], root: Path
) -> None:
    policy = _mapping(data["policy_context_profile"], "policy context profile")
    _equal(policy["repo"], "mPLUG/GUI-Owl-1.5-8B-Instruct", "policy repo")
    _equal(
        policy["revision"],
        "06d5faecff74840bab2be2425e9c42667a5d04fc",
        "policy revision",
    )
    _equal(
        policy["snapshot_manifest_path"],
        "code/configs/gui_owl_1_5_8b_snapshot.json",
        "policy snapshot",
    )
    _equal(
        policy["snapshot_manifest_sha256"],
        "50b675ec31c5c46dbb0d44c137a808fffb9d054916d39b596648d4eb9df7cbc3",
        "policy snapshot SHA256",
    )
    _equal(
        sha256_file(
            _resolve_regular(
                root,
                policy["snapshot_manifest_path"],
                "policy snapshot manifest",
            )
        ),
        policy["snapshot_manifest_sha256"],
        "policy snapshot live SHA256",
    )
    _equal(
        policy["runtime_class"],
        "causalcache.policy.gui_owl_v2_2_eager_runtime.GUIOwlV22EagerRuntime",
        "policy runtime class",
    )
    _equal(
        policy["runtime_id"],
        "causalcache_restoration_v2_2_eager_runtime",
        "policy runtime ID",
    )
    _equal(
        policy["official_tool_interface"],
        "causalcache_restoration_v2_1_official_tool_interface",
        "official-tool interface",
    )
    _equal(policy["dtype"], "torch.bfloat16", "policy dtype")
    _equal(policy["attention_implementation"], "eager", "policy attention")
    for key, expected in (
        ("effective_visual_tokens_per_image", 2560),
        ("maximum_context_tokens", 32768),
        ("generation_max_new_tokens", 256),
        ("processor_preflight_reserved_completion_tokens", 256),
        ("n8_full_reference_history_images", 8),
        ("n8_full_reference_total_images_including_current", 9),
        ("n16_budget_2_pair_union_maximum_history_images", 4),
        ("n16_budget_2_pair_union_maximum_total_images_including_current", 5),
        ("n16_budget_4_pair_union_maximum_history_images", 8),
        ("n16_budget_4_pair_union_maximum_total_images_including_current", 9),
    ):
        _strict_int(policy[key], expected, key)
    _strict_bool(policy["n16_full_reference_allowed"], False, "n16 full reference")
    _strict_bool(
        policy["pair_union_limits_include_overlap_without_requiring_disjointness"],
        True,
        "pair-union overlap",
    )

    operations = _mapping(data["operation_accounting"], "operation accounting")
    for key, expected in (
        ("fixed_selected_trajectory_denominator", 24),
        ("n8_canonical_action_parse_attempts_per_trajectory", 2),
        ("n8_total_requested_canonical_action_parse_attempts", 48),
        ("n8_budget_2_projected_subset_rows_per_valid_trajectory", 37),
        ("n8_budget_2_total_requested_projected_subset_rows", 888),
        ("n8_budget_4_unique_subset_rows_per_valid_trajectory", 163),
        ("n8_budget_4_total_requested_unique_subset_rows", 3912),
        ("n16_comparator_pair_count_per_trajectory", 6),
        ("n16_pair_reference_parse_attempts_per_pair", 2),
        ("n16_total_requested_pair_reference_parse_attempts", 288),
        ("n16_selector_arms_per_valid_pair", 2),
        ("n16_total_requested_selector_arm_distances", 288),
        ("n16_maximum_nonself_distance_rows_per_valid_pair", 3),
        ("n16_total_requested_maximum_nonself_distances", 432),
        ("maximum_generation_attempt_count", 336),
        ("maximum_teacher_forward_count", 4512),
        ("maximum_gpu_kl_count", 4344),
    ):
        _strict_int(operations[key], expected, key)
    for key in (
        "n8_budget_2_rows_are_projection_of_budget_4_rows_not_additional_forwards",
        "n16_nonself_rows_are_unique_empty_A_B_excluding_any_row_equal_to_R",
        "n16_pair_union_self_distance_is_analytic_zero",
        "maximum_counts_assume_every_case_is_valid_and_no_A_B_R_duplicate_rows",
        "requested_denominators_include_invalid_cases",
        "invalid_cases_preserve_partial_completed_operations_and_explicit_missing_counts",
    ):
        _strict_bool(operations[key], True, key)
    for key in (
        "invalid_cases_may_be_dropped_from_fixed_selected_denominator",
        "invalid_cases_may_be_replaced_or_topped_up",
    ):
        _strict_bool(operations[key], False, key)

    execution = _mapping(data["execution_b_plan"], "Execution-B plan")
    _equal(execution["host_alias"], "hyper00", "Execution-B host")
    for key, expected in (
        ("gpu_count", 4),
        ("worker_count", 4),
        ("development_trajectories_per_worker", 6),
        ("reserve_worker_count", 0),
    ):
        _strict_int(execution[key], expected, key)
    _equal(
        execution["worker_selection_rank_ranges"],
        [[0, 5], [6, 11], [12, 17], [18, 23]],
        "worker rank shards",
    )
    for key in (
        "gpu_ids_selected_only_after_required_preflight",
        "each_worker_binds_one_selected_gpu",
        "worker_shards_cover_each_development_trajectory_exactly_once",
    ):
        _strict_bool(execution[key], True, key)
    _strict_bool(
        execution["execution_b_may_change_scientific_threshold_geometry_or_comparator_semantics"],
        False,
        "Execution-B science mutation",
    )

    artifact = _mapping(data["artifact_plan"], "artifact plan")
    _equal(
        artifact["repo"],
        "gavinlaw/causalcache-long-horizon-development-mobile",
        "artifact repo",
    )
    _equal(artifact["repo_type"], "dataset", "artifact repo type")
    _equal(artifact["visibility"], "private", "artifact visibility")
    _equal(artifact["tag"], "long-horizon-development-v1", "artifact tag")
    _strict_bool(
        artifact["immutable_revision_and_manifest_required_before_git_result_commit"],
        True,
        "artifact immutable publication",
    )
    _strict_bool(artifact["reserve_payload_publication_allowed"], False, "reserve publication")

    learned = _mapping(data["learned_model_artifacts"], "learned model artifacts")
    _equal(
        sha256_bytes(canonical_json_bytes(learned)),
        LEARNED_MODEL_ARTIFACTS_SHA256,
        "learned model artifact roster digest",
    )
    formal = _mapping(
        learned["formal58_base_and_conditional"], "formal58 learned artifact"
    )
    _equal(
        formal["repo"],
        "gavinlaw/causalcache-gate-v1-formal58-selector-mobile",
        "formal58 repo",
    )
    _equal(
        formal["revision"],
        "23f6786075c7bff91f93fd7e8a878e070efb72a9",
        "formal58 revision",
    )
    _strict_bool(formal["private"], True, "formal58 privacy")
    manifests = _sequence(formal["ensemble_manifests"], "ensemble manifests")
    _equal(
        [record["family"] for record in manifests],
        ["conditional", "independent"],
        "ensemble manifest families",
    )
    checkpoints = _sequence(formal["checkpoints"], "formal58 checkpoints")
    _strict_int(len(checkpoints), 10, "formal58 checkpoint count")
    _equal(
        [(record["family"], record["seed"]) for record in checkpoints],
        [(family, seed) for family in ("conditional", "independent") for seed in range(5)],
        "formal58 checkpoint roster",
    )
    for record in (*manifests, *checkpoints):
        if SHA256_RE.fullmatch(str(record["sha256"])) is None:
            raise ValueError("formal58 artifact SHA256 is invalid")
    for record in checkpoints:
        if SHA256_RE.fullmatch(str(record["model_state_sha256"])) is None:
            raise ValueError("formal58 model-state SHA256 is invalid")

    residual = _mapping(
        learned["v4_safe_frozen_base_residual"], "v4 learned artifact"
    )
    _equal(
        residual["repo"],
        "gavinlaw/causalcache-set-conditioned-v4-frozen-base-residual-exploration-mobile",
        "v4 repo",
    )
    _equal(
        residual["revision"],
        "1641b90a4ebb05037e4710738a78e2c79f81cdc3",
        "v4 revision",
    )
    _equal(
        residual["manifest_path"],
        "manifest.json",
        "v4 manifest path",
    )
    _equal(
        residual["manifest_sha256"],
        "3ecc5e0a27e8ff57774788d71da6dbe4ff2a3bf08001747ab9a0de79607b9544",
        "v4 manifest SHA256",
    )
    _equal(
        residual["label_blind_seal_path"],
        "label-blind-seal.json",
        "v4 label-blind seal path",
    )
    _equal(
        residual["label_blind_seal_sha256"],
        "02b4b41eb9d180dd22e3c51575bfcdb9a0fca470d51fff3a7ee9cec38faa4439",
        "v4 label-blind seal SHA256",
    )
    _strict_bool(residual["private"], True, "v4 privacy")
    residual_checkpoints = _sequence(
        residual["residual_checkpoints"], "v4 residual checkpoints"
    )
    _equal(
        [record["seed"] for record in residual_checkpoints],
        list(range(5)),
        "v4 residual checkpoint seeds",
    )
    if any(SHA256_RE.fullmatch(str(record["sha256"])) is None for record in residual_checkpoints):
        raise ValueError("v4 residual checkpoint SHA256 is invalid")
    _strict_bool(
        learned[
            "execution_b_must_fresh_download_and_verify_exact_rosters_before_selector_scoring"
        ],
        True,
        "learned artifact verification",
    )
    _strict_bool(
        learned["training_optimizer_or_artifact_substitution_allowed"],
        False,
        "learned artifact substitution",
    )


def _validate_firewalls_and_freeze(data: Mapping[str, Any], root: Path) -> None:
    firewall = _mapping(data["historical_role_firewall"], "historical role firewall")
    _strict_bool(firewall["selected_intersection_must_be_empty"], True, "role overlap")
    _strict_bool(firewall["source_ids_must_be_parsed_from_the_bound_manifest_roles"], True, "role parsing")
    _strict_bool(firewall["literal_historical_source_id_roster_in_this_contract_allowed"], False, "literal role roster")
    historical_role_inventory(data, repository_root=root)

    access = _mapping(data["access_firewall"], "access firewall")
    _equal(
        access["source_a_allowed_operations"],
        [
            "strict_contract_validation",
            "pinned_raw_parquet_structural_scan",
            "deterministic_selection_manifest_construction",
        ],
        "Source-A allowed operations",
    )
    for key in (
        "old_confirm_locked",
        "old_closed_loop_locked",
        "sealed_test_locked",
        "matched_nll_locked",
        "reserve_locked",
    ):
        _strict_bool(access[key], True, key)

    freeze = _mapping(data["source_freeze"], "source freeze")
    _equal(freeze["required_source_a_paths"], list(REQUIRED_SOURCE_A_PATHS), "Source-A paths")
    _equal(
        freeze["source_a_inventory_sha256"],
        SOURCE_A_INVENTORY_PLACEHOLDER,
        "Source-A inventory placeholder",
    )
    _equal(
        freeze["source_a_inventory_hash_status"],
        "placeholder_must_be_bound_by_execution_b_runner_without_mutating_source_a",
        "Source-A placeholder status",
    )
    runner = _mapping(freeze["execution_b_runner_freeze"], "runner freeze")
    selection = _mapping(
        freeze["selection_manifest_freeze"], "selection manifest freeze"
    )
    _equal(
        selection["path"],
        SELECTION_MANIFEST_FREEZE_PATH,
        "selection manifest freeze path",
    )
    for key in (
        "must_be_absent_during_source_a_validation",
        "only_allowed_selection_freeze_diff",
        "direct_single_parent_child_of_source_a",
        "bind_source_a_git_commit",
        "bind_contract_sha256",
        "bind_required_source_a_inventory_sha256",
    ):
        _strict_bool(selection[key], True, f"selection manifest freeze {key}")
    _equal(runner["path"], RUNNER_FREEZE_B_PATH, "runner freeze path")
    for key in (
        "must_be_absent_during_source_a_validation",
        "only_allowed_execution_b_source_tree_diff",
        "direct_single_parent_child_of_selection_freeze",
        "bind_source_a_git_commit",
        "bind_contract_sha256",
        "bind_required_source_a_inventory_sha256",
        "bind_selection_freeze_git_commit",
        "bind_selection_manifest_sha256",
        "bind_substrate_immutable_revision_and_manifest",
        "bind_selector_preparation_manifest_immutable_revision_and_sha256",
        "bind_label_blind_seal_immutable_revision_and_sha256",
    ):
        _strict_bool(runner[key], True, f"runner freeze {key}")

    authorization = _mapping(data["authorization"], "authorization")
    if not authorization or any(type(value) is not bool or value for value in authorization.values()):
        raise ValueError("every Source-A authorization flag must be exactly false")


def validate_contract(data: Mapping[str, Any], *, repository_root: Path) -> None:
    _exact_keys(data, TOP_LEVEL_KEYS, "long-horizon contract")
    _equal(data["schema_version"], SCHEMA_VERSION, "schema version")
    _equal(data["protocol_id"], PROTOCOL_ID, "protocol ID")
    _equal(data["status"], SOURCE_STATUS, "Source-A status")
    _validate_source_pool(data, repository_root)
    _validate_selection_and_geometry(data)
    _validate_selector_and_reference_contract(data)
    _validate_validity_and_go_contract(data)
    _validate_policy_execution_and_artifact_plan(data, repository_root)
    _validate_firewalls_and_freeze(data, repository_root)


def build_source_a_inventory(repository_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in REQUIRED_SOURCE_A_PATHS:
        path = _resolve_regular(repository_root, relative, f"Source-A path {relative}")
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def validate_source_a(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    contract = LongHorizonContract.load(
        path,
        repository_root=repository_root,
        require_runner_absent=True,
    )
    inventory = build_source_a_inventory(contract.repository_root)
    historical = historical_role_inventory(
        contract.data,
        repository_root=contract.repository_root,
    )
    return {
        "status": VALIDATION_STATUS,
        "protocol_id": PROTOCOL_ID,
        "contract_sha256": contract.sha256,
        "source_a_inventory_placeholder": SOURCE_A_INVENTORY_PLACEHOLDER,
        "source_a_inventory": inventory,
        "source_a_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "runner_freeze_b_path": RUNNER_FREEZE_B_PATH,
        "runner_freeze_b_present": False,
        "historical_role_union_count": historical["union_source_id_count"],
        "historical_role_union_sha256": historical["union_source_ids_sha256"],
        "policy_load_or_forward_count": 0,
        "ocr_access_count": 0,
        "restoration_access_count": 0,
        "selector_scoring_count": 0,
        "reserve_access_count": 0,
        "old_confirm_access_count": 0,
        "closed_loop_access_count": 0,
        "sealed_test_access_count": 0,
    }
