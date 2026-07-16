"""Fail-closed contract and coverage validator for selector-geometry v2 repair."""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.restoration_v2_2_selector_geometry_contract import (
    CANONICAL_CONFIG_PATH as PARENT_CONFIG_PATH,
    FROZEN_CONFIG_SHA256 as PARENT_CONFIG_SHA256,
    RestorationV22SelectorGeometryContract,
    load_strict_json_object,
    sha256_file,
)


PROTOCOL_ID = "causalcache_restoration_v2_2_selector_geometry_v2_repair"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_selector_geometry_v2_repair.json"
)
FROZEN_CONFIG_SHA256 = (
    "2d312f54559f67aafe7efec2656d23171000e8b0f41d2d3c923f6a3c8b43be4c"
)
PASS_STATUS = "VALID_RESTORATION_V2_2_SELECTOR_GEOMETRY_V2_REPAIR_CONTRACT"
EXPECTED_SCHEMA_VERSION = "1.1.0"
EXPECTED_SCIENCE_STATUS = (
    "COMPLETED_POLICY_FREE_SELECTOR_GEOMETRY_V2_REPAIR_REDUCTION"
)
EXPECTED_STATE_COUNT = 45
EXPECTED_TRAJECTORY_COUNT = 15
EXPECTED_RECORD_COUNT = 180
EXPECTED_CELL_COUNT = 144
ROLES = ("v2_label_train", "v2_development")
EVENT_COUNTS = (2, 3, 4)
STRENGTHS = ("low", "mid", "high")
NEGATIVE_FLAGS = (False, True)
METHODS = (
    "exact_subset",
    "exact_cardinality_oracle",
    "true_conditional_greedy",
    "forced_fill_true_conditional_greedy",
    "budget_conditioned_independent",
    "forced_fill_budget_conditioned_independent",
    "full_shapley_independent",
    "forced_fill_full_shapley_independent",
    "dynamic_recent",
    "analytic_exact_cardinality_random",
)
SUMMARY_KEYS = {
    "status",
    "state_count",
    "trajectory_count",
    "mean_actual_utility",
    "mean_normalized_recovery",
    "recovery_ratio_of_means_to_exact_subset",
    "mean_selected_cardinality",
    "exact_coalition_match_rate",
    "mean_jaccard_to_exact_coalition",
    "selected_cardinality_histogram",
    "selection_metric_semantics",
}
EXPECTED_OPERATION_COUNTS = {
    "policy_forward": 0,
    "generation": 0,
    "teacher_forward": 0,
    "kl_measurement": 0,
    "gate_training_example": 0,
    "gate_model_forward": 0,
    "matched_nll_evaluation": 0,
    "closed_loop_episode": 0,
    "confirm_state_access": 0,
    "sealed_test_state_access": 0,
    "policy_vision_feature_forward": 0,
}

EXPECTED_PARENT = {
    "path": PARENT_CONFIG_PATH,
    "sha256": PARENT_CONFIG_SHA256,
    "protocol_id": "causalcache_restoration_v2_2_selector_geometry_v1",
    "source_git_commit": "7a1a8cc28a9a8f6a745f372406fe751b9bf4ff53",
}
EXPECTED_AUTHORIZATION = {
    "policy_free_reduction_allowed": True,
    "restoration_teacher_forward_allowed": False,
    "new_kl_measurement_allowed": False,
    "gate_training_allowed": False,
    "matched_nll_allowed": False,
    "closed_loop_allowed": False,
    "confirm_access_allowed": False,
    "sealed_test_access_allowed": False,
    "policy_vision_feature_extraction_allowed": False,
}
EXPECTED_OUTPUT = {
    "canonical_result_directory": (
        "data/results/restoration_v2_2_selector_geometry_v2_repair"
    ),
    "superseded_result_directory": (
        "data/results/restoration_v2_2_selector_geometry_v1"
    ),
    "superseded_result_must_not_be_modified": True,
    "raw_labels_or_images_copied_to_git": False,
    "result_must_bind_parent_contract_and_immutable_labels": True,
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
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} key schema drifted: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} drifted: expected {expected!r}, got {actual!r}")


def _close(actual: Any, expected: float, label: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(float(actual)) or not math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"{label} drifted: expected {expected}, got {actual}")


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile input cannot be empty")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _expected_repair_scope() -> dict[str, Any]:
    return {
        "reporting_only": True,
        "selector_algorithms_changed": False,
        "selector_scientific_values_changed": False,
        "bootstrap_values_changed": False,
        "required_interaction_factor_cross_product": [
            "role",
            "candidate_event_count",
            "budget",
            "interaction_strength_stratum",
            "has_strict_negative_deployment_marginal",
        ],
        "required_roles": list(ROLES),
        "required_candidate_event_counts": list(EVENT_COUNTS),
        "required_interaction_strength_strata": list(STRENGTHS),
        "required_negative_marginal_flags": list(NEGATIVE_FLAGS),
        "expected_interaction_factor_cell_count": EXPECTED_CELL_COUNT,
        "empty_cells_materialized": True,
        "empty_cell_numeric_values": None,
        "required_selectors_per_cell": list(METHODS),
        "analytic_random_required_metrics": [
            "exact_cardinality",
            "expected_exact_coalition_match",
            "expected_jaccard_to_exact_coalition",
        ],
        "analytic_random_match_definition": (
            "one_over_n_choose_B_if_exact_subset_cardinality_equals_B_else_zero"
        ),
        "analytic_random_jaccard_definition": (
            "uniform_mean_over_all_exact_B_coalitions"
        ),
        "analytic_random_budget_zero_rule": {
            "exact_cardinality": 0,
            "expected_exact_coalition_match": 1.0,
            "expected_jaccard_to_exact_coalition": 1.0,
        },
    }


@dataclass(frozen=True)
class RestorationV22SelectorGeometryRepairContract:
    path: Path
    data: Mapping[str, Any]
    sha256: str
    parent: RestorationV22SelectorGeometryContract

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
    ) -> "RestorationV22SelectorGeometryRepairContract":
        root = Path(repository_root).resolve()
        resolved = Path(path).resolve()
        canonical = (root / CANONICAL_CONFIG_PATH).resolve()
        if resolved != canonical or not resolved.is_file() or resolved.is_symlink():
            raise ValueError("repair contract must use the canonical config")
        digest = sha256_file(resolved)
        _equal(digest, FROZEN_CONFIG_SHA256, "repair config SHA256")
        parent = RestorationV22SelectorGeometryContract.load(
            root / PARENT_CONFIG_PATH,
            repository_root=root,
        )
        contract = cls(
            path=resolved,
            data=load_strict_json_object(resolved),
            sha256=digest,
            parent=parent,
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        data = _mapping(self.data, "repair contract")
        _exact_keys(
            data,
            {
                "schema_version",
                "protocol_id",
                "repair_status",
                "parent_contract",
                "repair_scope",
                "authorization",
                "output_contract",
            },
            "repair contract",
        )
        _equal(data["schema_version"], "0.2.0", "repair schema version")
        _equal(data["protocol_id"], PROTOCOL_ID, "repair protocol id")
        _equal(
            data["repair_status"],
            "source_only_frozen_before_v2_repair_reduction",
            "repair status",
        )
        _equal(data["parent_contract"], EXPECTED_PARENT, "parent contract")
        _equal(data["repair_scope"], _expected_repair_scope(), "repair scope")
        _equal(data["authorization"], EXPECTED_AUTHORIZATION, "authorization")
        _equal(data["output_contract"], EXPECTED_OUTPUT, "output contract")
        _equal(self.parent.sha256, PARENT_CONFIG_SHA256, "loaded parent SHA256")


def validate_contract(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    contract = RestorationV22SelectorGeometryRepairContract.load(
        path,
        repository_root=repository_root,
    )
    return {
        "status": PASS_STATUS,
        "protocol_id": PROTOCOL_ID,
        "config_sha256": contract.sha256,
        "parent_config_sha256": contract.parent.sha256,
        "expected_state_budget_record_count": EXPECTED_RECORD_COUNT,
        "expected_interaction_factor_cell_count": EXPECTED_CELL_COUNT,
        "reporting_only_repair": True,
        "selector_values_changed": False,
        "confirm_locked": True,
        "sealed_test_locked": True,
        "gate_training_locked": True,
    }


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = frozenset(left)
    right_set = frozenset(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def _validate_random_record(record: Mapping[str, Any], label: str) -> None:
    state = _mapping(record["state"], f"{label}.state")
    methods = _mapping(record["methods"], f"{label}.methods")
    random_method = _mapping(
        methods["analytic_exact_cardinality_random"],
        f"{label}.random",
    )
    exact_method = _mapping(methods["exact_subset"], f"{label}.exact")
    budget = record["budget_event_capacity"]
    event_ids = tuple(state["candidate_event_step_ids"])
    exact = tuple(exact_method["selected_coalition"])
    if type(budget) is not int or budget < 0 or budget > len(event_ids):
        raise ValueError(f"{label} has invalid budget")
    coalitions = tuple(itertools.combinations(event_ids, budget))
    expected_count = math.comb(len(event_ids), budget)
    _equal(random_method.get("coalition_count"), expected_count, f"{label} random C(n,B)")
    _equal(random_method.get("exact_cardinality"), budget, f"{label} random k")
    _equal(random_method.get("selected_cardinality"), budget, f"{label} random cardinality")
    expected_match = 1.0 / expected_count if len(exact) == budget else 0.0
    expected_jaccard = math.fsum(
        _jaccard(coalition, exact) for coalition in coalitions
    ) / expected_count
    _close(
        random_method.get("expected_exact_coalition_match"),
        expected_match,
        f"{label} random expected match",
    )
    _close(
        random_method.get("exact_coalition_match"),
        expected_match,
        f"{label} random generic match",
    )
    _close(
        random_method.get("expected_jaccard_to_exact_coalition"),
        expected_jaccard,
        f"{label} random expected Jaccard",
    )
    _close(
        random_method.get("jaccard_to_exact_coalition"),
        expected_jaccard,
        f"{label} random generic Jaccard",
    )
    if budget == 0:
        _close(random_method["expected_exact_coalition_match"], 1.0, f"{label} B0 match")
        _close(random_method["expected_jaccard_to_exact_coalition"], 1.0, f"{label} B0 Jaccard")


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    method: str,
    state_count: int,
    trajectory_count: int,
    label: str,
) -> None:
    _exact_keys(summary, SUMMARY_KEYS, label)
    _equal(summary["state_count"], state_count, f"{label}.state_count")
    _equal(
        summary["trajectory_count"],
        trajectory_count,
        f"{label}.trajectory_count",
    )
    histogram = _mapping(
        summary["selected_cardinality_histogram"],
        f"{label}.selected_cardinality_histogram",
    )
    expected_semantics = (
        "analytic_expectation_over_uniform_exact_cardinality_coalitions"
        if method == "analytic_exact_cardinality_random"
        else "deterministic_selected_coalition"
    )
    _equal(
        summary["selection_metric_semantics"],
        expected_semantics,
        f"{label}.selection_metric_semantics",
    )
    if state_count == 0:
        _equal(summary["status"], "EMPTY_STRATUM", f"{label}.status")
        for key in SUMMARY_KEYS - {
            "status",
            "state_count",
            "trajectory_count",
            "selected_cardinality_histogram",
            "selection_metric_semantics",
        }:
            _equal(summary[key], None, f"{label}.{key}")
        _equal(histogram, {}, f"{label}.histogram")
        return
    _equal(summary["status"], "MEASURED", f"{label}.status")
    if sum(histogram.values()) != state_count:
        raise ValueError(f"{label} cardinality histogram does not cover its states")
    for key in (
        "mean_actual_utility",
        "mean_normalized_recovery",
        "mean_selected_cardinality",
        "exact_coalition_match_rate",
        "mean_jaccard_to_exact_coalition",
    ):
        value = summary[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}.{key} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{label}.{key} must be finite")
    for key in ("exact_coalition_match_rate", "mean_jaccard_to_exact_coalition"):
        if not 0.0 <= float(summary[key]) <= 1.0:
            raise ValueError(f"{label}.{key} must be in [0,1]")


def validate_repaired_scientific_payload(payload: Mapping[str, Any]) -> None:
    """Independently reject omitted cells and incorrect random expectations."""
    science = _mapping(payload, "science")
    _equal(science.get("schema_version"), EXPECTED_SCHEMA_VERSION, "science schema")
    _equal(science.get("status"), EXPECTED_SCIENCE_STATUS, "science status")
    _equal(
        science.get("repair_identity"),
        {
            "protocol_id": PROTOCOL_ID,
            "supersedes_incomplete_result": (
                "causalcache_restoration_v2_2_selector_geometry_v1"
            ),
            "selector_values_changed": False,
            "bootstrap_values_changed": False,
            "reporting_only_repair": True,
        },
        "science repair identity",
    )
    operation_counts = _mapping(science.get("operation_counts"), "operation counts")
    _equal(operation_counts, EXPECTED_OPERATION_COUNTS, "operation counts")
    denominator = _mapping(science.get("denominator"), "science denominator")
    _exact_keys(
        denominator,
        {"state_count", "trajectory_count", "state_budget_record_count"},
        "science denominator",
    )
    _equal(denominator.get("state_count"), EXPECTED_STATE_COUNT, "state count")
    _equal(
        denominator.get("trajectory_count"),
        EXPECTED_TRAJECTORY_COUNT,
        "trajectory count",
    )
    _equal(
        denominator.get("state_budget_record_count"),
        EXPECTED_RECORD_COUNT,
        "record count",
    )
    records = _sequence(science.get("state_budget_records"), "state records")
    _equal(len(records), EXPECTED_RECORD_COUNT, "state-record length")

    state_metadata: dict[str, tuple[str, str, int, tuple[int, ...], float, bool]] = {}
    budgets_by_state: dict[str, set[int]] = {}
    record_keys: set[tuple[str, int]] = set()
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"record[{index}]")
        state = _mapping(record.get("state"), f"record[{index}].state")
        methods = _mapping(record.get("methods"), f"record[{index}].methods")
        _equal(set(methods), set(METHODS), f"record[{index}] methods")
        state_id = state.get("state_id")
        if not isinstance(state_id, str) or not state_id:
            raise ValueError(f"record[{index}] state_id is invalid")
        role = state.get("role")
        trajectory_id = state.get("trajectory_id")
        event_ids = tuple(state.get("candidate_event_step_ids", ()))
        event_count = state.get("candidate_event_count")
        if role not in ROLES:
            raise ValueError(f"record[{index}] role is invalid")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError(f"record[{index}] trajectory_id is invalid")
        if type(event_count) is not int or event_count not in EVENT_COUNTS:
            raise ValueError(f"record[{index}] candidate count is invalid")
        if (
            len(event_ids) != event_count
            or any(type(event_id) is not int for event_id in event_ids)
            or len(set(event_ids)) != event_count
            or tuple(sorted(event_ids)) != event_ids
        ):
            raise ValueError(f"record[{index}] candidate event ids are invalid")
        interaction = _mapping(
            record.get("interaction_metrics"),
            f"record[{index}].interaction_metrics",
        )
        negative_marginal = interaction["has_strict_negative_marginal"]
        if type(negative_marginal) is not bool:
            raise ValueError(f"record[{index}] negative-marginal flag is invalid")
        metadata = (
            role,
            trajectory_id,
            event_count,
            event_ids,
            float(interaction["normalized_mean_absolute_interaction"]),
            negative_marginal,
        )
        if state_id in state_metadata and state_metadata[state_id] != metadata:
            raise ValueError(f"state metadata drifted across budgets: {state_id}")
        state_metadata[state_id] = metadata
        budget = record.get("budget_event_capacity")
        key = (state_id, budget)
        if key in record_keys:
            raise ValueError(f"duplicate state-budget record: {key}")
        record_keys.add(key)
        budgets_by_state.setdefault(state_id, set()).add(budget)
        _validate_random_record(record, f"record[{index}]")

    _equal(len(state_metadata), EXPECTED_STATE_COUNT, "unique state count")
    for state_id, metadata in state_metadata.items():
        n = metadata[2]
        _equal(budgets_by_state[state_id], set(range(n + 1)), f"{state_id} budgets")
    _equal(
        Counter(metadata[0] for metadata in state_metadata.values()),
        Counter({"v2_label_train": 30, "v2_development": 15}),
        "state roles",
    )
    _equal(
        Counter(metadata[2] for metadata in state_metadata.values()),
        Counter({2: 15, 3: 15, 4: 15}),
        "candidate-event counts",
    )
    trajectories = {(metadata[0], metadata[1]) for metadata in state_metadata.values()}
    _equal(len(trajectories), EXPECTED_TRAJECTORY_COUNT, "unique trajectories")

    cutpoints = _mapping(
        science.get("train_interaction_tertile_cutpoints_by_n"),
        "interaction cutpoints",
    )
    _equal(set(cutpoints), {"2", "3", "4"}, "interaction cutpoint inventory")
    for n in EVENT_COUNTS:
        train_values = [
            metadata[4]
            for metadata in state_metadata.values()
            if metadata[0] == "v2_label_train" and metadata[2] == n
        ]
        observed = _mapping(cutpoints.get(str(n)), f"cutpoints[{n}]")
        _exact_keys(
            observed,
            {"lower_tertile", "upper_tertile", "train_state_count"},
            f"cutpoints[{n}]",
        )
        _close(
            observed.get("lower_tertile"),
            _linear_quantile(train_values, 1.0 / 3.0),
            f"cutpoints[{n}].lower",
        )
        _close(
            observed.get("upper_tertile"),
            _linear_quantile(train_values, 2.0 / 3.0),
            f"cutpoints[{n}].upper",
        )
        _equal(observed.get("train_state_count"), 10, f"cutpoints[{n}].count")

    assignments = _sequence(
        science.get("interaction_state_assignments"),
        "interaction assignments",
    )
    _equal(len(assignments), EXPECTED_STATE_COUNT, "assignment count")
    assignment_map: dict[str, tuple[str, int, str, bool, str]] = {}
    for index, raw_assignment in enumerate(assignments):
        assignment = _mapping(raw_assignment, f"assignment[{index}]")
        _exact_keys(
            assignment,
            {
                "role",
                "trajectory_id",
                "state_id",
                "candidate_event_count",
                "normalized_mean_absolute_interaction",
                "interaction_strength_stratum",
                "has_strict_negative_deployment_marginal",
            },
            f"assignment[{index}]",
        )
        state_id = assignment.get("state_id")
        if state_id in assignment_map:
            raise ValueError(f"duplicate interaction assignment: {state_id}")
        if state_id not in state_metadata:
            raise ValueError(f"unknown interaction assignment: {state_id}")
        if type(
            assignment.get("has_strict_negative_deployment_marginal")
        ) is not bool:
            raise ValueError(f"assignment[{index}] negative flag is invalid")
        role, trajectory_id, n, _, value, negative = state_metadata[state_id]
        observed_cutpoints = cutpoints[str(n)]
        expected_strength = (
            "low"
            if value <= observed_cutpoints["lower_tertile"]
            else "mid"
            if value <= observed_cutpoints["upper_tertile"]
            else "high"
        )
        expected = (role, n, expected_strength, negative, trajectory_id)
        observed_value = (
            assignment.get("role"),
            assignment.get("candidate_event_count"),
            assignment.get("interaction_strength_stratum"),
            assignment.get("has_strict_negative_deployment_marginal"),
            assignment.get("trajectory_id"),
        )
        _equal(observed_value, expected, f"assignment[{index}]")
        _close(
            assignment.get("normalized_mean_absolute_interaction"),
            value,
            f"assignment[{index}].interaction",
        )
        assignment_map[state_id] = expected

    reports = _mapping(
        science.get("interaction_selector_reports"),
        "interaction selector reports",
    )
    _exact_keys(
        reports,
        {
            "dimensions",
            "cutpoint_source",
            "development_application",
            "expected_cell_count",
            "observed_cell_count",
            "nonempty_cell_count",
            "empty_cell_count",
            "cells",
        },
        "interaction selector reports",
    )
    _equal(
        reports.get("dimensions"),
        {
            "roles": list(ROLES),
            "candidate_event_counts": list(EVENT_COUNTS),
            "budget_domain": "zero_through_matching_candidate_count",
            "interaction_strength_strata": list(STRENGTHS),
            "strict_negative_marginal_flags": list(NEGATIVE_FLAGS),
        },
        "interaction report dimensions",
    )
    _equal(
        reports.get("cutpoint_source"),
        "train_only_tertiles_separately_within_n",
        "interaction cutpoint source",
    )
    _equal(
        reports.get("development_application"),
        "frozen_matching_n_train_cutpoints",
        "interaction development application",
    )
    _equal(reports.get("expected_cell_count"), EXPECTED_CELL_COUNT, "expected cells")
    _equal(reports.get("observed_cell_count"), EXPECTED_CELL_COUNT, "observed cells")
    cells = _sequence(reports.get("cells"), "interaction cells")
    _equal(len(cells), EXPECTED_CELL_COUNT, "cell length")
    expected_keys = {
        (role, n, budget, strength, negative)
        for role in ROLES
        for n in EVENT_COUNTS
        for budget in range(n + 1)
        for strength in STRENGTHS
        for negative in NEGATIVE_FLAGS
    }
    observed_keys: set[tuple[str, int, int, str, bool]] = set()
    nonempty_count = 0
    for index, raw_cell in enumerate(cells):
        cell = _mapping(raw_cell, f"cell[{index}]")
        _exact_keys(
            cell,
            {
                "cell_id",
                "role",
                "candidate_event_count",
                "budget_event_capacity",
                "interaction_strength_stratum",
                "has_strict_negative_deployment_marginal",
                "state_count",
                "trajectory_count",
                "methods",
            },
            f"cell[{index}]",
        )
        key = (
            cell.get("role"),
            cell.get("candidate_event_count"),
            cell.get("budget_event_capacity"),
            cell.get("interaction_strength_stratum"),
            cell.get("has_strict_negative_deployment_marginal"),
        )
        if (
            type(key[1]) is not int
            or type(key[2]) is not int
            or type(key[4]) is not bool
        ):
            raise ValueError(f"cell[{index}] factor types are invalid")
        if key in observed_keys:
            raise ValueError(f"duplicate interaction cell: {key}")
        observed_keys.add(key)
        role, n, budget, strength, negative = key
        expected_state_ids = {
            state_id
            for state_id, assignment in assignment_map.items()
            if assignment[:4] == (role, n, strength, negative)
        }
        expected_trajectories = {
            assignment_map[state_id][4] for state_id in expected_state_ids
        }
        state_count = len(expected_state_ids)
        trajectory_count = len(expected_trajectories)
        _equal(cell.get("state_count"), state_count, f"cell[{index}].state_count")
        _equal(
            cell.get("trajectory_count"),
            trajectory_count,
            f"cell[{index}].trajectory_count",
        )
        expected_id = (
            f"{role}:n{n}:b{budget}:{strength}:negative_{str(negative).lower()}"
        )
        _equal(cell.get("cell_id"), expected_id, f"cell[{index}].cell_id")
        methods = _mapping(cell.get("methods"), f"cell[{index}].methods")
        _equal(set(methods), set(METHODS), f"cell[{index}] methods")
        for method in METHODS:
            _validate_summary(
                _mapping(methods[method], f"cell[{index}].{method}"),
                method=method,
                state_count=state_count,
                trajectory_count=trajectory_count,
                label=f"cell[{index}].{method}",
            )
        nonempty_count += state_count > 0
    _equal(observed_keys, expected_keys, "joint interaction Cartesian cells")
    _equal(reports.get("nonempty_cell_count"), nonempty_count, "nonempty cells")
    _equal(
        reports.get("empty_cell_count"),
        EXPECTED_CELL_COUNT - nonempty_count,
        "empty cells",
    )


def validate_repair_preserves_legacy_payload(
    repaired: Mapping[str, Any],
    legacy: Mapping[str, Any],
) -> None:
    """Verify the repair changes reporting geometry, not frozen selector values."""
    shared_top_level = (
        "operation_counts",
        "denominator",
        "bootstrap",
        "visual_baselines",
        "train_interaction_tertile_cutpoints_by_n",
        "primary_interaction_strata",
        "internal_method_shaping",
    )
    for key in shared_top_level:
        _equal(repaired.get(key), legacy.get(key), f"legacy-preservation {key}")
    for section in ("slices", "budget_curve"):
        repaired_section = _mapping(repaired.get(section), f"repaired {section}")
        legacy_section = _mapping(legacy.get(section), f"legacy {section}")
        _equal(set(repaired_section), set(legacy_section), f"{section} inventory")
        for slice_id in legacy_section:
            repaired_slice = _mapping(repaired_section[slice_id], "repaired slice")
            legacy_slice = _mapping(legacy_section[slice_id], "legacy slice")
            for key in (
                "slice_id",
                "state_count",
                "trajectory_count",
                "candidate_event_counts",
                "budgets",
            ):
                _equal(
                    repaired_slice.get(key),
                    legacy_slice.get(key),
                    f"{section}.{slice_id}.{key}",
                )
            _equal(
                repaired_slice.get("paired_bootstrap"),
                legacy_slice.get("paired_bootstrap"),
                f"{section}.{slice_id}.bootstrap",
            )
            for comparison, legacy_deltas in legacy_slice[
                "development_trajectory_deltas"
            ].items():
                repaired_deltas = repaired_slice[
                    "development_trajectory_deltas"
                ][comparison]
                _equal(len(repaired_deltas), len(legacy_deltas), "delta count")
                for repaired_delta, legacy_delta in zip(
                    repaired_deltas,
                    legacy_deltas,
                    strict=True,
                ):
                    for key, value in legacy_delta.items():
                        _equal(
                            repaired_delta.get(key),
                            value,
                            f"{section}.{slice_id}.{comparison}.{key}",
                        )
            for method in METHODS:
                for role, legacy_summary in legacy_slice["methods"][method].items():
                    repaired_summary = repaired_slice["methods"][method][role]
                    for key, value in legacy_summary.items():
                        _equal(
                            repaired_summary.get(key),
                            value,
                            f"{section}.{slice_id}.{method}.{role}.{key}",
                        )
    repaired_records = _sequence(repaired.get("state_budget_records"), "repaired records")
    legacy_records = _sequence(legacy.get("state_budget_records"), "legacy records")
    _equal(len(repaired_records), len(legacy_records), "legacy record count")
    random_reporting_keys = {
        "exact_cardinality",
        "selected_cardinality",
        "expected_exact_coalition_match",
        "expected_jaccard_to_exact_coalition",
        "exact_coalition_match",
        "jaccard_to_exact_coalition",
        "metric_semantics",
    }
    for index, (repaired_record, legacy_record) in enumerate(
        zip(repaired_records, legacy_records, strict=True)
    ):
        repaired_copy = dict(repaired_record)
        legacy_copy = dict(legacy_record)
        repaired_methods = {
            method: dict(values)
            for method, values in repaired_copy["methods"].items()
        }
        legacy_methods = {
            method: dict(values) for method, values in legacy_copy["methods"].items()
        }
        for key in random_reporting_keys:
            repaired_methods["analytic_exact_cardinality_random"].pop(key, None)
            legacy_methods["analytic_exact_cardinality_random"].pop(key, None)
        repaired_copy["methods"] = repaired_methods
        legacy_copy["methods"] = legacy_methods
        _equal(repaired_copy, legacy_copy, f"legacy state record[{index}]")
