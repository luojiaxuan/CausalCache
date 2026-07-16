"""Deterministic scientific reduction for restoration-v2.2 selector geometry."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from causalcache.restoration_v2_2_geometry_stats import (
    DEFAULT_BOOTSTRAP_CONFIDENCE,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_TIE_EPSILON,
    StateMetricRow,
    aggregate_trajectory_first,
    linear_quantile,
    paired_trajectory_bootstrap,
    ratio_of_means,
)
from causalcache.restoration_v2_2_selector_geometry import (
    NORMALIZATION_EPSILON,
    SelectorGeometryState,
    evaluate_all_feasible_budgets,
)


SCHEMA_VERSION = "1.0.0"
STATUS = "COMPLETED_POLICY_FREE_SELECTOR_GEOMETRY_REDUCTION"
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
DETERMINISTIC_METHODS = frozenset(METHODS) - {
    "analytic_exact_cardinality_random"
}
PRIMARY_SLICE_ID = "primary_n4_b2"


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


def _validate_formal_states(states: Sequence[SelectorGeometryState]) -> None:
    if len(states) != 45:
        raise ValueError("formal selector geometry requires exactly 45 states")
    if tuple(state.index for state in states) != tuple(range(45)):
        raise ValueError("formal selector geometry states must be index ordered")
    role_counts = Counter(state.role for state in states)
    if role_counts != {"v2_label_train": 30, "v2_development": 15}:
        raise ValueError("formal selector geometry role denominator drifted")
    event_counts = Counter(len(state.candidate_event_step_ids) for state in states)
    if event_counts != {2: 15, 3: 15, 4: 15}:
        raise ValueError("formal selector geometry event-count denominator drifted")
    trajectories: dict[tuple[str, str], list[SelectorGeometryState]] = {}
    for state in states:
        if len(state.candidate_event_step_ids) != state.decision_step_id - 2:
            raise ValueError("decision-step/event-count identity drifted")
        trajectories.setdefault((state.role, state.trajectory_id), []).append(state)
    trajectory_roles = Counter(role for role, _ in trajectories)
    if trajectory_roles != {"v2_label_train": 10, "v2_development": 5}:
        raise ValueError("formal selector geometry trajectory denominator drifted")
    if any(
        sorted(state.decision_step_id for state in trajectory) != [4, 5, 6]
        for trajectory in trajectories.values()
    ):
        raise ValueError("every formal trajectory must contain decision steps 4,5,6")


def _compact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    exact_coalition = record["methods"]["exact_subset"]["selected_coalition"]
    methods: dict[str, dict[str, Any]] = {}
    if set(record["methods"]) != set(METHODS):
        raise ValueError("selector method inventory drifted")
    for method_name in METHODS:
        method = record["methods"][method_name]
        coalition = method["selected_coalition"]
        compact = {
            "selected_coalition": coalition,
            "selected_cardinality": None if coalition is None else len(coalition),
            "utility": method["utility"],
            "normalized_recovery": method["normalized_recovery"],
            "absolute_regret_to_exact_subset": method[
                "absolute_regret_to_exact_subset"
            ],
            "normalized_recovery_regret_to_exact_subset": method[
                "normalized_recovery_regret_to_exact_subset"
            ],
            "utility_ratio_to_exact_subset": method[
                "utility_ratio_to_exact_subset"
            ],
            "exact_coalition_match": (
                None if coalition is None else coalition == exact_coalition
            ),
            "jaccard_to_exact_coalition": (
                None if coalition is None else _jaccard(coalition, exact_coalition)
            ),
        }
        if method_name == "analytic_exact_cardinality_random":
            compact["coalition_count"] = method["coalition_count"]
        methods[method_name] = compact
    return {
        "state": dict(record["state"]),
        "budget_event_capacity": record["budget_event_capacity"],
        "baseline_summary_only_distance": record[
            "baseline_summary_only_distance"
        ],
        "interaction_metrics": dict(record["interaction_metrics"]),
        "methods": methods,
        "gaps": {
            key: dict(value) for key, value in record["gaps"].items()
        },
    }


def build_state_records(
    states: Sequence[SelectorGeometryState],
    *,
    enforce_formal_denominator: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Evaluate and compact every state at every capacity from zero through n."""
    ordered = tuple(states)
    if not ordered:
        raise ValueError("selector geometry requires at least one state")
    if any(not isinstance(state, SelectorGeometryState) for state in ordered):
        raise TypeError("states must contain only SelectorGeometryState objects")
    if enforce_formal_denominator:
        _validate_formal_states(ordered)
    records = tuple(
        _compact_record(record)
        for state in ordered
        for record in evaluate_all_feasible_budgets(state)
    )
    expected_count = sum(
        len(state.candidate_event_step_ids) + 1 for state in ordered
    )
    if len(records) != expected_count:
        raise RuntimeError("selector geometry budget-grid row count drifted")
    return records


def _slice_records(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_counts: frozenset[int],
    budget: int,
) -> tuple[Mapping[str, Any], ...]:
    selected = tuple(
        record
        for record in records
        if record["state"]["candidate_event_count"] in candidate_counts
        and record["budget_event_capacity"] == budget
    )
    if not selected:
        raise ValueError("selector geometry slice cannot be empty")
    return selected


def _trajectory_values(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> tuple[tuple[str, str, int, float], ...]:
    rows = [
        StateMetricRow(
            role=record["state"]["role"],
            trajectory_id=record["state"]["trajectory_id"],
            state_id=record["state"]["state_id"],
            metrics={"value": float(record["_summary_metrics"][metric])},
        )
        for record in records
    ]
    return tuple(
        (row.role, row.trajectory_id, row.state_count, row.metric("value"))
        for row in aggregate_trajectory_first(rows, ("value",))
    )


def _role_groups(
    values: Sequence[tuple[str, str, int, float]],
) -> dict[str, tuple[tuple[str, str, int, float], ...]]:
    roles = sorted({value[0] for value in values})
    result = {
        role: tuple(value for value in values if value[0] == role)
        for role in roles
    }
    result["overall_stratified"] = tuple(values)
    return result


def _method_summary(
    records: Sequence[Mapping[str, Any]],
    method_name: str,
) -> dict[str, Any]:
    prepared: list[dict[str, Any]] = []
    for record in records:
        method = record["methods"][method_name]
        exact = record["methods"]["exact_subset"]
        recovery = method["normalized_recovery"]
        exact_recovery = exact["normalized_recovery"]
        if recovery is None or exact_recovery is None:
            raise ValueError("formal selector summary encountered null recovery")
        summary_metrics = {
            "utility": float(method["utility"]),
            "recovery": float(recovery),
            "exact_recovery": float(exact_recovery),
        }
        if method["selected_cardinality"] is not None:
            summary_metrics["cardinality"] = float(method["selected_cardinality"])
            summary_metrics["exact_match"] = float(method["exact_coalition_match"])
            summary_metrics["jaccard"] = float(method["jaccard_to_exact_coalition"])
        prepared.append({**record, "_summary_metrics": summary_metrics})

    recovery_values = _trajectory_values(prepared, metric="recovery")
    exact_values = _trajectory_values(prepared, metric="exact_recovery")
    utility_values = _trajectory_values(prepared, metric="utility")
    exact_lookup = {
        (role, trajectory_id): value
        for role, trajectory_id, _, value in exact_values
    }
    utility_lookup = {
        (role, trajectory_id): value
        for role, trajectory_id, _, value in utility_values
    }
    groups = _role_groups(recovery_values)
    result: dict[str, Any] = {}
    for role, values in groups.items():
        recoveries = tuple(value[3] for value in values)
        exact_recoveries = tuple(
            exact_lookup[(value[0], value[1])] for value in values
        )
        utilities = tuple(
            utility_lookup[(value[0], value[1])] for value in values
        )
        summary: dict[str, Any] = {
            "state_count": sum(value[2] for value in values),
            "trajectory_count": len(values),
            "mean_actual_utility": _mean(utilities),
            "mean_normalized_recovery": _mean(recoveries),
            "recovery_ratio_of_means_to_exact_subset": ratio_of_means(
                recoveries,
                exact_recoveries,
            ),
        }
        if method_name in DETERMINISTIC_METHODS:
            for metric, output_key in (
                ("cardinality", "mean_selected_cardinality"),
                ("exact_match", "exact_coalition_match_rate"),
                ("jaccard", "mean_jaccard_to_exact_coalition"),
            ):
                metric_records = [
                    {**record, "_summary_metrics": record["_summary_metrics"]}
                    for record in prepared
                ]
                trajectory_metric = _trajectory_values(
                    metric_records,
                    metric=metric,
                )
                lookup = {
                    (item[0], item[1]): item[3] for item in trajectory_metric
                }
                summary[output_key] = _mean(
                    tuple(lookup[(item[0], item[1])] for item in values)
                )
            cardinalities = Counter(
                int(record["methods"][method_name]["selected_cardinality"])
                for record in records
                if role == "overall_stratified"
                or record["state"]["role"] == role
            )
            summary["selected_cardinality_histogram"] = {
                str(key): cardinalities[key] for key in sorted(cardinalities)
            }
        result[role] = summary
    return result


def _paired_rows(
    records: Sequence[Mapping[str, Any]],
) -> tuple[StateMetricRow, ...]:
    rows = []
    for record in records:
        methods = record["methods"]
        values = {
            "exact": methods["exact_subset"]["normalized_recovery"],
            "greedy": methods["true_conditional_greedy"]["normalized_recovery"],
            "independent": methods["budget_conditioned_independent"][
                "normalized_recovery"
            ],
            "shapley": methods["full_shapley_independent"][
                "normalized_recovery"
            ],
        }
        if any(value is None for value in values.values()):
            raise ValueError("paired selector geometry encountered null recovery")
        rows.append(
            StateMetricRow(
                role=record["state"]["role"],
                trajectory_id=record["state"]["trajectory_id"],
                state_id=record["state"]["state_id"],
                metrics={key: float(value) for key, value in values.items()},
            )
        )
    return tuple(rows)


def _serialize_bootstrap(report: Any) -> dict[str, Any]:
    value = asdict(report)
    for split in ("train", "development", "overall_stratified"):
        value[split]["win_tie_loss"]["count"] = sum(
            value[split]["win_tie_loss"][key]
            for key in ("wins", "ties", "losses")
        )
    return value


def _development_deltas(
    rows: Sequence[StateMetricRow],
    *,
    left: str,
    right: str,
) -> list[dict[str, Any]]:
    trajectories = aggregate_trajectory_first(rows, (left, right))
    return [
        {
            "trajectory_id": row.trajectory_id,
            "left": row.metric(left),
            "right": row.metric(right),
            "difference": row.metric(left) - row.metric(right),
        }
        for row in trajectories
        if row.role == "v2_development"
    ]


def _slice_summary(
    slice_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows = _paired_rows(records)
    comparisons = {
        "search_gap": ("exact", "greedy"),
        "objective_projection_gap": ("greedy", "independent"),
        "full_shapley_projection_gap": ("greedy", "shapley"),
    }
    paired = {}
    development = {}
    for name, (left, right) in comparisons.items():
        paired[name] = _serialize_bootstrap(
            paired_trajectory_bootstrap(
                rows,
                left,
                right,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
                confidence=DEFAULT_BOOTSTRAP_CONFIDENCE,
            )
        )
        development[name] = _development_deltas(rows, left=left, right=right)
    return {
        "slice_id": slice_id,
        "state_count": len(records),
        "trajectory_count": len(
            {(record["state"]["role"], record["state"]["trajectory_id"]) for record in records}
        ),
        "candidate_event_counts": sorted(
            {record["state"]["candidate_event_count"] for record in records}
        ),
        "budgets": sorted({record["budget_event_capacity"] for record in records}),
        "methods": {
            method: _method_summary(records, method) for method in METHODS
        },
        "paired_bootstrap": paired,
        "development_trajectory_deltas": development,
    }


def _train_interaction_cutpoints(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    result = {}
    for event_count in (2, 3, 4):
        values_by_state: dict[str, float] = {}
        for record in records:
            if (
                record["state"]["role"] == "v2_label_train"
                and record["state"]["candidate_event_count"] == event_count
            ):
                values_by_state[record["state"]["state_id"]] = float(
                    record["interaction_metrics"][
                        "normalized_mean_absolute_interaction"
                    ]
                )
        values = tuple(values_by_state.values())
        if not values:
            raise ValueError("train interaction cutpoint stratum is empty")
        result[str(event_count)] = {
            "lower_tertile": linear_quantile(values, 1.0 / 3.0),
            "upper_tertile": linear_quantile(values, 2.0 / 3.0),
            "train_state_count": len(values),
        }
    return result


def _interaction_label(
    record: Mapping[str, Any],
    cutpoints: Mapping[str, Mapping[str, float]],
) -> str:
    count = str(record["state"]["candidate_event_count"])
    value = float(
        record["interaction_metrics"]["normalized_mean_absolute_interaction"]
    )
    lower = float(cutpoints[count]["lower_tertile"])
    upper = float(cutpoints[count]["upper_tertile"])
    if value <= lower:
        return "low"
    if value <= upper:
        return "mid"
    return "high"


def _descriptive_gap_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ("v2_label_train", "v2_development", "overall_stratified"):
        selected = tuple(
            record
            for record in records
            if role == "overall_stratified" or record["state"]["role"] == role
        )
        if not selected:
            result[role] = {"state_count": 0, "trajectory_count": 0}
            continue
        rows = _paired_rows(selected)
        trajectories = aggregate_trajectory_first(
            rows,
            ("exact", "greedy", "independent", "shapley"),
        )
        result[role] = {
            "state_count": len(selected),
            "trajectory_count": len(trajectories),
            "mean_search_gap": _mean(
                tuple(row.metric("exact") - row.metric("greedy") for row in trajectories)
            ),
            "mean_objective_projection_gap": _mean(
                tuple(row.metric("greedy") - row.metric("independent") for row in trajectories)
            ),
            "mean_full_shapley_projection_gap": _mean(
                tuple(row.metric("greedy") - row.metric("shapley") for row in trajectories)
            ),
        }
    return result


def _primary_interaction_strata(
    primary_records: Sequence[Mapping[str, Any]],
    cutpoints: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    by_strength = {}
    for label in ("low", "mid", "high"):
        selected = tuple(
            record
            for record in primary_records
            if _interaction_label(record, cutpoints) == label
        )
        by_strength[label] = _descriptive_gap_summary(selected)
    by_negative = {}
    for flag in (False, True):
        selected = tuple(
            record
            for record in primary_records
            if bool(
                record["interaction_metrics"]["has_strict_negative_marginal"]
            )
            is flag
        )
        by_negative[str(flag).lower()] = _descriptive_gap_summary(selected)
    return {
        "by_train_tertile_strength": by_strength,
        "by_strict_negative_deployment_marginal": by_negative,
    }


def _method_shaping(primary: Mapping[str, Any]) -> dict[str, Any]:
    projection = primary["paired_bootstrap"]["objective_projection_gap"][
        "development"
    ]
    search = primary["paired_bootstrap"]["search_gap"]["development"]
    projection_mean = float(projection["mean_difference"])
    projection_wins = int(projection["win_tie_loss"]["wins"])
    projection_lower = float(projection["mean_difference_interval"]["lower"])
    if projection_mean <= 0.01:
        conditioning = "prefer_independent_gate"
    elif (
        projection_mean >= 0.05
        and projection_wins >= 4
        and projection_lower > 0.0
    ):
        conditioning = "set_conditioned_main_candidate"
    else:
        conditioning = "dual_architecture_ablation_required"

    exact_mean = float(search["left_mean"])
    greedy_mean = float(search["right_mean"])
    greedy_ratio = (
        None
        if exact_mean <= NORMALIZATION_EPSILON
        else greedy_mean / exact_mean
    )
    search_gap = float(search["mean_difference"])
    if greedy_ratio is not None and greedy_ratio >= 0.90 and search_gap <= 0.05:
        search_decision = "online_greedy_sufficient"
    elif (greedy_ratio is not None and greedy_ratio < 0.85) or search_gap > 0.10:
        search_decision = "search_gap_requires_strong_diagnosis"
    else:
        search_decision = "search_gap_yellow"
    return {
        "paper_claim_gate": False,
        "confirm_unlock_gate": False,
        "conditioning_decision": conditioning,
        "conditioning_inputs": {
            "development_mean_objective_projection_gap": projection_mean,
            "positive_direction_trajectory_count_out_of_5": projection_wins,
            "paired_90_percent_bootstrap_lower_bound": projection_lower,
        },
        "search_decision": search_decision,
        "search_inputs": {
            "development_true_greedy_to_exact_recovery_ratio": greedy_ratio,
            "development_mean_normalized_recovery_search_gap": search_gap,
        },
    }


def build_selector_geometry_scientific_payload(
    states: Sequence[SelectorGeometryState],
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    enforce_formal_denominator: bool = True,
) -> dict[str, Any]:
    """Build the complete JSON-safe table-only selector-geometry payload."""
    if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if type(bootstrap_seed) is not int:
        raise TypeError("bootstrap_seed must be an integer")
    records = build_state_records(
        states,
        enforce_formal_denominator=enforce_formal_denominator,
    )
    slices: dict[str, dict[str, Any]] = {}
    slice_specs = {
        PRIMARY_SLICE_ID: (frozenset({4}), 2),
        "secondary_n3_b2": (frozenset({3}), 2),
        "hard_n_ge3_b2": (frozenset({3, 4}), 2),
        "overall_b2_secondary": (frozenset({2, 3, 4}), 2),
    }
    for slice_id, (event_counts, budget) in slice_specs.items():
        slices[slice_id] = _slice_summary(
            slice_id,
            _slice_records(records, candidate_counts=event_counts, budget=budget),
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    budget_curve = {}
    for event_count in (2, 3, 4):
        for budget in range(event_count + 1):
            slice_id = f"n{event_count}_b{budget}"
            budget_curve[slice_id] = _slice_summary(
                slice_id,
                _slice_records(
                    records,
                    candidate_counts=frozenset({event_count}),
                    budget=budget,
                ),
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            )

    cutpoints = _train_interaction_cutpoints(records)
    primary_records = _slice_records(
        records,
        candidate_counts=frozenset({4}),
        budget=2,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
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
            "frozen_policy_vision_similarity": "pending_separate_feature_only_source_freeze",
        },
        "train_interaction_tertile_cutpoints_by_n": cutpoints,
        "primary_interaction_strata": _primary_interaction_strata(
            primary_records,
            cutpoints,
        ),
        "slices": slices,
        "budget_curve": budget_curve,
        "internal_method_shaping": _method_shaping(slices[PRIMARY_SLICE_ID]),
        "state_budget_records": list(records),
    }
