"""Versioned repair for complete selector-geometry scientific reporting."""

from __future__ import annotations

import copy
import itertools
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from causalcache import restoration_v2_2_selector_geometry_result as legacy
from causalcache.restoration_v2_2_geometry_stats import (
    DEFAULT_BOOTSTRAP_CONFIDENCE,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_TIE_EPSILON,
    StateMetricRow,
    aggregate_trajectory_first,
    paired_trajectory_bootstrap,
    ratio_of_means,
)
from causalcache.restoration_v2_2_selector_geometry import SelectorGeometryState


SCHEMA_VERSION = "1.1.0"
STATUS = "COMPLETED_POLICY_FREE_SELECTOR_GEOMETRY_V2_REPAIR_REDUCTION"
METHODS = legacy.METHODS
PRIMARY_SLICE_ID = legacy.PRIMARY_SLICE_ID
ROLES = ("v2_label_train", "v2_development")
INTERACTION_STRENGTHS = ("low", "mid", "high")
NEGATIVE_MARGINAL_FLAGS = (False, True)
EXPECTED_INTERACTION_FACTOR_CELL_COUNT = 144


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = frozenset(left)
    right_set = frozenset(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def _repair_random_metrics(record: dict[str, Any]) -> None:
    random_method = record["methods"]["analytic_exact_cardinality_random"]
    budget = int(record["budget_event_capacity"])
    event_ids = tuple(record["state"]["candidate_event_step_ids"])
    exact = tuple(record["methods"]["exact_subset"]["selected_coalition"])
    coalitions = tuple(itertools.combinations(event_ids, budget))
    if len(coalitions) != int(random_method["coalition_count"]):
        raise RuntimeError("analytic random coalition denominator drifted")
    random_method.update(
        {
            "exact_cardinality": budget,
            "selected_cardinality": budget,
            "expected_exact_coalition_match": _mean(
                tuple(float(coalition == exact) for coalition in coalitions)
            ),
            "expected_jaccard_to_exact_coalition": _mean(
                tuple(_jaccard(coalition, exact) for coalition in coalitions)
            ),
            "metric_semantics": (
                "analytic_expectation_over_uniform_exact_cardinality_coalitions"
            ),
        }
    )
    random_method["exact_coalition_match"] = random_method[
        "expected_exact_coalition_match"
    ]
    random_method["jaccard_to_exact_coalition"] = random_method[
        "expected_jaccard_to_exact_coalition"
    ]


def build_state_records(
    states: Sequence[SelectorGeometryState],
    *,
    enforce_formal_denominator: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Preserve v1 selector values while repairing analytic-random metadata."""
    legacy_records = legacy.build_state_records(
        states,
        enforce_formal_denominator=enforce_formal_denominator,
    )
    repaired = copy.deepcopy(legacy_records)
    for record in repaired:
        _repair_random_metrics(record)
    return tuple(repaired)


def _trajectory_metrics(
    records: Sequence[Mapping[str, Any]],
    method_name: str,
) -> tuple[Any, ...]:
    rows = []
    for record in records:
        method = record["methods"][method_name]
        exact = record["methods"]["exact_subset"]
        recovery = method["normalized_recovery"]
        exact_recovery = exact["normalized_recovery"]
        if recovery is None or exact_recovery is None:
            raise ValueError("formal selector summary encountered null recovery")
        rows.append(
            StateMetricRow(
                role=record["state"]["role"],
                trajectory_id=record["state"]["trajectory_id"],
                state_id=record["state"]["state_id"],
                metrics={
                    "utility": float(method["utility"]),
                    "recovery": float(recovery),
                    "exact_recovery": float(exact_recovery),
                    "cardinality": float(method["selected_cardinality"]),
                    "exact_match": float(method["exact_coalition_match"]),
                    "jaccard": float(method["jaccard_to_exact_coalition"]),
                },
            )
        )
    return aggregate_trajectory_first(
        rows,
        (
            "utility",
            "recovery",
            "exact_recovery",
            "cardinality",
            "exact_match",
            "jaccard",
        ),
    )


def _method_summary(
    records: Sequence[Mapping[str, Any]],
    method_name: str,
) -> dict[str, Any]:
    if method_name not in METHODS:
        raise ValueError(f"unknown selector method: {method_name}")
    trajectories = _trajectory_metrics(records, method_name)
    result: dict[str, Any] = {}
    for role in (*ROLES, "overall_stratified"):
        selected_trajectories = tuple(
            row
            for row in trajectories
            if role == "overall_stratified" or row.role == role
        )
        if not selected_trajectories:
            continue
        selected_records = tuple(
            record
            for record in records
            if role == "overall_stratified" or record["state"]["role"] == role
        )
        recoveries = tuple(
            row.metric("recovery") for row in selected_trajectories
        )
        exact_recoveries = tuple(
            row.metric("exact_recovery") for row in selected_trajectories
        )
        cardinalities = Counter(
            int(record["methods"][method_name]["selected_cardinality"])
            for record in selected_records
        )
        result[role] = {
            "status": "MEASURED",
            "state_count": len(selected_records),
            "trajectory_count": len(selected_trajectories),
            "mean_actual_utility": _mean(
                tuple(row.metric("utility") for row in selected_trajectories)
            ),
            "mean_normalized_recovery": _mean(recoveries),
            "recovery_ratio_of_means_to_exact_subset": ratio_of_means(
                recoveries,
                exact_recoveries,
            ),
            "mean_selected_cardinality": _mean(
                tuple(
                    row.metric("cardinality") for row in selected_trajectories
                )
            ),
            "exact_coalition_match_rate": _mean(
                tuple(
                    row.metric("exact_match") for row in selected_trajectories
                )
            ),
            "mean_jaccard_to_exact_coalition": _mean(
                tuple(row.metric("jaccard") for row in selected_trajectories)
            ),
            "selected_cardinality_histogram": {
                str(key): cardinalities[key] for key in sorted(cardinalities)
            },
            "selection_metric_semantics": (
                "analytic_expectation_over_uniform_exact_cardinality_coalitions"
                if method_name == "analytic_exact_cardinality_random"
                else "deterministic_selected_coalition"
            ),
        }
    return result


def _serialize_bootstrap(report: Any) -> dict[str, Any]:
    value = asdict(report)
    for split in ("train", "development", "overall_stratified"):
        value[split]["win_tie_loss"]["count"] = sum(
            value[split]["win_tie_loss"][key]
            for key in ("wins", "ties", "losses")
        )
    return value


def _development_delta_payload(
    rows: Sequence[StateMetricRow],
    *,
    left: str,
    right: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    trajectories = tuple(
        row
        for row in aggregate_trajectory_first(rows, (left, right))
        if row.role == "v2_development"
    )
    values = []
    counts = {"wins": 0, "ties": 0, "losses": 0}
    for row in trajectories:
        difference = row.metric(left) - row.metric(right)
        if difference > DEFAULT_TIE_EPSILON:
            sign = "win"
            counts["wins"] += 1
        elif difference < -DEFAULT_TIE_EPSILON:
            sign = "loss"
            counts["losses"] += 1
        else:
            sign = "tie"
            counts["ties"] += 1
        values.append(
            {
                "trajectory_id": row.trajectory_id,
                "left": row.metric(left),
                "right": row.metric(right),
                "difference": difference,
                "sign": sign,
            }
        )
    return values, {**counts, "count": sum(counts.values())}


def _slice_summary(
    slice_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    paired_rows = legacy._paired_rows(records)
    comparisons = {
        "search_gap": ("exact", "greedy"),
        "objective_projection_gap": ("greedy", "independent"),
        "full_shapley_projection_gap": ("greedy", "shapley"),
    }
    paired = {}
    development = {}
    development_sign_counts = {}
    for name, (left, right) in comparisons.items():
        paired[name] = _serialize_bootstrap(
            paired_trajectory_bootstrap(
                paired_rows,
                left,
                right,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
                confidence=DEFAULT_BOOTSTRAP_CONFIDENCE,
            )
        )
        deltas, counts = _development_delta_payload(
            paired_rows,
            left=left,
            right=right,
        )
        development[name] = deltas
        development_sign_counts[name] = counts
    return {
        "slice_id": slice_id,
        "state_count": len(records),
        "trajectory_count": len(
            {
                (record["state"]["role"], record["state"]["trajectory_id"])
                for record in records
            }
        ),
        "candidate_event_counts": sorted(
            {record["state"]["candidate_event_count"] for record in records}
        ),
        "budgets": sorted(
            {record["budget_event_capacity"] for record in records}
        ),
        "methods": {
            method: _method_summary(records, method) for method in METHODS
        },
        "paired_bootstrap": paired,
        "development_trajectory_deltas": development,
        "development_sign_counts": development_sign_counts,
    }


def _interaction_label(
    record: Mapping[str, Any],
    cutpoints: Mapping[str, Mapping[str, float]],
) -> str:
    event_count = str(record["state"]["candidate_event_count"])
    value = float(
        record["interaction_metrics"][
            "normalized_mean_absolute_interaction"
        ]
    )
    lower = float(cutpoints[event_count]["lower_tertile"])
    upper = float(cutpoints[event_count]["upper_tertile"])
    if value <= lower:
        return "low"
    if value <= upper:
        return "mid"
    return "high"


def _empty_method_summary(method_name: str) -> dict[str, Any]:
    return {
        "status": "EMPTY_STRATUM",
        "state_count": 0,
        "trajectory_count": 0,
        "mean_actual_utility": None,
        "mean_normalized_recovery": None,
        "recovery_ratio_of_means_to_exact_subset": None,
        "mean_selected_cardinality": None,
        "exact_coalition_match_rate": None,
        "mean_jaccard_to_exact_coalition": None,
        "selected_cardinality_histogram": {},
        "selection_metric_semantics": (
            "analytic_expectation_over_uniform_exact_cardinality_coalitions"
            if method_name == "analytic_exact_cardinality_random"
            else "deterministic_selected_coalition"
        ),
    }


def _interaction_assignments(
    records: Sequence[Mapping[str, Any]],
    cutpoints: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    by_state: dict[str, Mapping[str, Any]] = {}
    for record in records:
        by_state[record["state"]["state_id"]] = record
    return [
        {
            "role": record["state"]["role"],
            "trajectory_id": record["state"]["trajectory_id"],
            "state_id": state_id,
            "candidate_event_count": record["state"][
                "candidate_event_count"
            ],
            "normalized_mean_absolute_interaction": record[
                "interaction_metrics"
            ]["normalized_mean_absolute_interaction"],
            "interaction_strength_stratum": _interaction_label(
                record,
                cutpoints,
            ),
            "has_strict_negative_deployment_marginal": bool(
                record["interaction_metrics"][
                    "has_strict_negative_marginal"
                ]
            ),
        }
        for state_id, record in sorted(by_state.items())
    ]


def _interaction_factor_reports(
    records: Sequence[Mapping[str, Any]],
    cutpoints: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    cells = []
    for role in ROLES:
        for event_count in (2, 3, 4):
            for budget in range(event_count + 1):
                for strength in INTERACTION_STRENGTHS:
                    for negative_flag in NEGATIVE_MARGINAL_FLAGS:
                        selected = tuple(
                            record
                            for record in records
                            if record["state"]["role"] == role
                            and record["state"]["candidate_event_count"]
                            == event_count
                            and record["budget_event_capacity"] == budget
                            and _interaction_label(record, cutpoints) == strength
                            and bool(
                                record["interaction_metrics"][
                                    "has_strict_negative_marginal"
                                ]
                            )
                            is negative_flag
                        )
                        methods = {
                            method: (
                                _method_summary(selected, method)[role]
                                if selected
                                else _empty_method_summary(method)
                            )
                            for method in METHODS
                        }
                        cells.append(
                            {
                                "cell_id": (
                                    f"{role}:n{event_count}:b{budget}:"
                                    f"{strength}:negative_{str(negative_flag).lower()}"
                                ),
                                "role": role,
                                "candidate_event_count": event_count,
                                "budget_event_capacity": budget,
                                "interaction_strength_stratum": strength,
                                "has_strict_negative_deployment_marginal": (
                                    negative_flag
                                ),
                                "state_count": len(selected),
                                "trajectory_count": len(
                                    {
                                        record["state"]["trajectory_id"]
                                        for record in selected
                                    }
                                ),
                                "methods": methods,
                            }
                        )
    if len(cells) != EXPECTED_INTERACTION_FACTOR_CELL_COUNT:
        raise RuntimeError("interaction factor cross-product count drifted")
    return {
        "dimensions": {
            "roles": list(ROLES),
            "candidate_event_counts": [2, 3, 4],
            "budget_domain": "zero_through_matching_candidate_count",
            "interaction_strength_strata": list(INTERACTION_STRENGTHS),
            "strict_negative_marginal_flags": [False, True],
        },
        "cutpoint_source": "train_only_tertiles_separately_within_n",
        "development_application": "frozen_matching_n_train_cutpoints",
        "expected_cell_count": EXPECTED_INTERACTION_FACTOR_CELL_COUNT,
        "observed_cell_count": len(cells),
        "nonempty_cell_count": sum(cell["state_count"] > 0 for cell in cells),
        "empty_cell_count": sum(cell["state_count"] == 0 for cell in cells),
        "cells": cells,
    }


def build_selector_geometry_scientific_payload(
    states: Sequence[SelectorGeometryState],
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    enforce_formal_denominator: bool = True,
) -> dict[str, Any]:
    """Build the complete v2 repair payload without changing selector values."""
    if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if type(bootstrap_seed) is not int:
        raise TypeError("bootstrap_seed must be an integer")
    records = build_state_records(
        states,
        enforce_formal_denominator=enforce_formal_denominator,
    )
    slice_specs = {
        PRIMARY_SLICE_ID: (frozenset({4}), 2),
        "secondary_n3_b2": (frozenset({3}), 2),
        "hard_n_ge3_b2": (frozenset({3, 4}), 2),
        "overall_b2_secondary": (frozenset({2, 3, 4}), 2),
    }
    slices = {
        slice_id: _slice_summary(
            slice_id,
            legacy._slice_records(
                records,
                candidate_counts=event_counts,
                budget=budget,
            ),
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        for slice_id, (event_counts, budget) in slice_specs.items()
    }
    budget_curve = {}
    for event_count in (2, 3, 4):
        for budget in range(event_count + 1):
            slice_id = f"n{event_count}_b{budget}"
            budget_curve[slice_id] = _slice_summary(
                slice_id,
                legacy._slice_records(
                    records,
                    candidate_counts=frozenset({event_count}),
                    budget=budget,
                ),
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            )
    cutpoints = legacy._train_interaction_cutpoints(records)
    primary_records = legacy._slice_records(
        records,
        candidate_counts=frozenset({4}),
        budget=2,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "repair_identity": {
            "protocol_id": (
                "causalcache_restoration_v2_2_selector_geometry_v2_repair"
            ),
            "supersedes_incomplete_result": (
                "causalcache_restoration_v2_2_selector_geometry_v1"
            ),
            "selector_values_changed": False,
            "bootstrap_values_changed": False,
            "reporting_only_repair": True,
        },
        "operation_counts": {
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
        },
        "denominator": {
            "state_count": len(states),
            "trajectory_count": len(
                {(state.role, state.trajectory_id) for state in states}
            ),
            "state_budget_record_count": len(records),
        },
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "confidence": DEFAULT_BOOTSTRAP_CONFIDENCE,
            "resampling_unit": "trajectory_id",
            "overall": "stratified_within_role",
        },
        "visual_baselines": {
            "ocr_and_rgb_similarity": "pending_feature_stage",
            "frozen_policy_vision_similarity": (
                "pending_separate_feature_only_source_freeze"
            ),
        },
        "train_interaction_tertile_cutpoints_by_n": cutpoints,
        "interaction_state_assignments": _interaction_assignments(
            records,
            cutpoints,
        ),
        "interaction_selector_reports": _interaction_factor_reports(
            records,
            cutpoints,
        ),
        "primary_interaction_strata": legacy._primary_interaction_strata(
            primary_records,
            cutpoints,
        ),
        "slices": slices,
        "budget_curve": budget_curve,
        "internal_method_shaping": legacy._method_shaping(
            slices[PRIMARY_SLICE_ID]
        ),
        "state_budget_records": list(records),
    }
