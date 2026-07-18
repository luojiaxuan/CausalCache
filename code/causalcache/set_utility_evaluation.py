"""True-utility evaluation for joint set selectors."""

from __future__ import annotations

import itertools
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.set_utility_data import (
    SetUtilityState,
    UTILITY_TIE_EPSILON,
    build_padded_set_utility_batch,
)
from causalcache.set_utility_search import exact_utility_oracle


def _states(value: Sequence[SetUtilityState]) -> tuple[SetUtilityState, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not value:
        raise ValueError("set-utility evaluation requires states")
    result = tuple(value)
    if any(not isinstance(state, SetUtilityState) for state in result):
        raise TypeError("set-utility evaluation contains an invalid state")
    if len({state.state_id for state in result}) != len(result):
        raise ValueError("evaluation state ids must be unique")
    return result


def _selection(
    value: Sequence[int],
    *,
    state: SetUtilityState,
    budget: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("selector output must be an event-id sequence")
    result = tuple(value)
    if (
        any(type(event_id) is not int for event_id in result)
        or result != tuple(sorted(result))
        or len(set(result)) != len(result)
        or len(result) > budget
        or not set(result).issubset(state.event_ids)
    ):
        raise ValueError("selector output is not a feasible canonical subset")
    state.utility(result)
    return result


def trajectory_equal_values(
    states: Sequence[SetUtilityState],
    state_values: Mapping[str, float],
) -> dict[str, float]:
    """Average states within trajectory before any cross-trajectory reduction."""
    selected = _states(states)
    if set(state_values) != {state.state_id for state in selected}:
        raise ValueError("state-value inventory differs from the evaluation states")
    by_trajectory: dict[str, list[float]] = defaultdict(list)
    for state in selected:
        value = float(state_values[state.state_id])
        if not math.isfinite(value):
            raise ValueError("state metric is non-finite")
        by_trajectory[state.trajectory_id].append(value)
    return {
        trajectory_id: statistics.fmean(values)
        for trajectory_id, values in sorted(by_trajectory.items())
    }


def trajectory_equal_mean(
    states: Sequence[SetUtilityState],
    state_values: Mapping[str, float],
) -> float:
    values = trajectory_equal_values(states, state_values)
    return statistics.fmean(values.values())


def paired_trajectory_win_tie_loss(
    states: Sequence[SetUtilityState],
    left_state_values: Mapping[str, float],
    right_state_values: Mapping[str, float],
    *,
    tie_epsilon: float = UTILITY_TIE_EPSILON,
) -> dict[str, Any]:
    """Compare selectors after state-equal aggregation within each trajectory."""
    if not math.isfinite(tie_epsilon) or tie_epsilon < 0.0:
        raise ValueError("paired tie epsilon must be finite and non-negative")
    left = trajectory_equal_values(states, left_state_values)
    right = trajectory_equal_values(states, right_state_values)
    if set(left) != set(right):
        raise RuntimeError("paired trajectory inventories differ")
    deltas = {
        trajectory_id: left[trajectory_id] - right[trajectory_id]
        for trajectory_id in left
    }
    return {
        "trajectory_count": len(deltas),
        "wins": sum(delta > tie_epsilon for delta in deltas.values()),
        "ties": sum(abs(delta) <= tie_epsilon for delta in deltas.values()),
        "losses": sum(delta < -tie_epsilon for delta in deltas.values()),
        "trajectory_equal_mean_delta": statistics.fmean(deltas.values()),
        "deltas_by_trajectory": deltas,
    }


def _method_summary(
    states: tuple[SetUtilityState, ...],
    *,
    budget: int,
    selections: Mapping[str, tuple[int, ...]],
    exact_utilities: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, float]]:
    utilities = {
        state.state_id: state.utility(selections[state.state_id]) for state in states
    }
    mean_utility = trajectory_equal_mean(states, utilities)
    exact_mean = trajectory_equal_mean(states, exact_utilities)
    histogram = Counter(len(subset) for subset in selections.values())
    recovery_by_trajectory: dict[str, list[float]] = defaultdict(list)
    excluded_state_count = 0
    for state in states:
        exact = exact_utilities[state.state_id]
        if exact <= UTILITY_TIE_EPSILON:
            excluded_state_count += 1
            continue
        recovery_by_trajectory[state.trajectory_id].append(
            utilities[state.state_id] / exact
        )
    recovery_trajectory_means = tuple(
        statistics.fmean(values) for values in recovery_by_trajectory.values()
    )
    cardinalities_by_trajectory: dict[str, list[int]] = defaultdict(list)
    for state in states:
        cardinalities_by_trajectory[state.trajectory_id].append(
            len(selections[state.state_id])
        )
    return (
        {
            "trajectory_equal_mean_raw_utility": mean_utility,
            "trajectory_equal_mean_exact_utility": exact_mean,
            "trajectory_equal_mean_exact_recovery_ratio": (
                None
                if not recovery_trajectory_means
                else statistics.fmean(recovery_trajectory_means)
            ),
            "exact_recovery_excluded_state_count": excluded_state_count,
            "exact_recovery_excluded_trajectory_count": len(
                {state.trajectory_id for state in states}
                - set(recovery_by_trajectory)
            ),
            "ratio_of_trajectory_equal_mean_raw_utility_to_exact": (
                None if exact_mean <= UTILITY_TIE_EPSILON else mean_utility / exact_mean
            ),
            "selected_cardinality_distribution": {
                "counts": {
                    str(cardinality): histogram.get(cardinality, 0)
                    for cardinality in range(budget + 1)
                },
                "proportions": {
                    str(cardinality): histogram.get(cardinality, 0) / len(states)
                    for cardinality in range(budget + 1)
                },
                "trajectory_equal_proportions": {
                    str(cardinality): statistics.fmean(
                        sum(value == cardinality for value in values) / len(values)
                        for values in cardinalities_by_trajectory.values()
                    )
                    for cardinality in range(budget + 1)
                },
            },
        },
        utilities,
    )


def evaluate_joint_selectors(
    states: Sequence[SetUtilityState],
    *,
    budget: int,
    selections_by_method: Mapping[str, Mapping[str, Sequence[int]]],
    tie_epsilon: float = UTILITY_TIE_EPSILON,
) -> dict[str, Any]:
    """Score every selected set with true U, never with its predicted score."""
    selected_states = _states(states)
    if type(budget) is not int or budget < 0:
        raise ValueError("evaluation budget must be a non-negative integer")
    if any(
        budget > state.maximum_labeled_cardinality for state in selected_states
    ):
        raise ValueError("evaluation budget exceeds a state's label-cardinality cap")
    if not isinstance(selections_by_method, Mapping) or not selections_by_method:
        raise ValueError("evaluation requires at least one selector")
    if "exact_subset_oracle" in selections_by_method:
        raise ValueError("exact_subset_oracle is reserved for the evaluator")
    expected_state_ids = {state.state_id for state in selected_states}

    selections: dict[str, dict[str, tuple[int, ...]]] = {}
    for method, values in selections_by_method.items():
        if not isinstance(method, str) or not method:
            raise ValueError("selector names must be non-empty text")
        if not isinstance(values, Mapping) or set(values) != expected_state_ids:
            raise ValueError(f"{method} selection inventory drifted")
        selections[method] = {
            state.state_id: _selection(
                values[state.state_id], state=state, budget=budget
            )
            for state in selected_states
        }

    exact_results = {
        state.state_id: exact_utility_oracle(state, budget=budget)
        for state in selected_states
    }
    selections["exact_subset_oracle"] = {
        state_id: result.selected_subset
        for state_id, result in exact_results.items()
    }
    exact_utilities = {
        state_id: result.selected_predicted_utility
        for state_id, result in exact_results.items()
    }

    summaries: dict[str, Any] = {}
    utilities_by_method: dict[str, dict[str, float]] = {}
    for method, method_selections in selections.items():
        summary, utilities = _method_summary(
            selected_states,
            budget=budget,
            selections=method_selections,
            exact_utilities=exact_utilities,
        )
        summaries[method] = summary
        utilities_by_method[method] = utilities

    comparisons = {
        f"{left}_minus_{right}": paired_trajectory_win_tie_loss(
            selected_states,
            utilities_by_method[left],
            utilities_by_method[right],
            tie_epsilon=tie_epsilon,
        )
        for left, right in itertools.permutations(selections, 2)
    }
    records = {
        state.state_id: {
            "trajectory_id": state.trajectory_id,
            "methods": {
                method: {
                    "selected_subset": list(method_selections[state.state_id]),
                    "selected_cardinality": len(method_selections[state.state_id]),
                    "true_utility": utilities_by_method[method][state.state_id],
                }
                for method, method_selections in selections.items()
            },
        }
        for state in selected_states
    }
    return {
        "budget": budget,
        "state_count": len(selected_states),
        "trajectory_count": len(
            {state.trajectory_id for state in selected_states}
        ),
        "all_selected_sets_scored_by_true_utility": True,
        "methods": summaries,
        "paired_trajectory_comparisons": comparisons,
        "records": records,
    }


def evaluate_prediction_table(
    states: Sequence[SetUtilityState],
    predictions_by_state: Mapping[str, Mapping[tuple[int, ...], float]],
    *,
    tie_epsilon: float = UTILITY_TIE_EPSILON,
) -> dict[str, Any]:
    """Report balanced regression and within-state subset-ranking quality."""
    selected_states = _states(states)
    if not isinstance(predictions_by_state, Mapping):
        raise TypeError("prediction table must be state indexed")
    if set(predictions_by_state) != {state.state_id for state in selected_states}:
        raise ValueError("prediction state inventory drifted")
    batch = build_padded_set_utility_batch(
        selected_states,
        ranking_tie_epsilon=tie_epsilon,
    )

    prediction_rows: list[list[float]] = []
    for state in selected_states:
        state_predictions = predictions_by_state[state.state_id]
        if not isinstance(state_predictions, Mapping):
            raise TypeError("state predictions must be subset indexed")
        expected = {target.subset for target in state.targets}
        if set(state_predictions) != expected:
            raise ValueError("prediction subset inventory drifted")
        values = [float(state_predictions[target.subset]) for target in state.targets]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("prediction table contains a non-finite value")
        prediction_rows.append(values)

    raw_absolute = 0.0
    raw_squared = 0.0
    normalized_absolute = 0.0
    normalized_squared = 0.0
    for state_index, state in enumerate(selected_states):
        for row_index, target in enumerate(state.targets):
            prediction = prediction_rows[state_index][row_index]
            raw_error = prediction - target.raw_utility
            raw_weight = batch.raw_regression_weights[state_index][row_index]
            raw_absolute += raw_weight * abs(raw_error)
            raw_squared += raw_weight * raw_error * raw_error
            normalized_weight = batch.normalized_regression_weights[state_index][
                row_index
            ]
            if normalized_weight:
                assert state.normalization_scale is not None
                normalized_error = raw_error / state.normalization_scale
                normalized_absolute += normalized_weight * abs(normalized_error)
                normalized_squared += (
                    normalized_weight * normalized_error * normalized_error
                )

    ranking_accuracy = 0.0
    ranking_tie_credit = 0.0
    for row in batch.ranking_rows:
        delta = (
            prediction_rows[row.batch_index][row.left_row]
            - prediction_rows[row.batch_index][row.right_row]
        )
        signed = delta * row.target_sign
        credit = 1.0 if signed > tie_epsilon else 0.5 if abs(delta) <= tie_epsilon else 0.0
        ranking_accuracy += row.weight * credit
        if abs(delta) <= tie_epsilon:
            ranking_tie_credit += row.weight
    normalized_available = any(batch.normalized_state_mask)
    return {
        "weighting": "state_equal_then_cardinality_equal_within_state",
        "raw_regression": {
            "mae": raw_absolute,
            "rmse": math.sqrt(raw_squared),
        },
        "normalized_regression": (
            {
                "mae": normalized_absolute,
                "rmse": math.sqrt(normalized_squared),
            }
            if normalized_available
            else None
        ),
        "within_state_subset_ranking": {
            "pair_count": len(batch.ranking_rows),
            "balanced_accuracy_with_half_credit_for_prediction_ties": (
                ranking_accuracy if batch.ranking_rows else None
            ),
            "prediction_tie_weight": (
                ranking_tie_credit if batch.ranking_rows else None
            ),
        },
    }


__all__ = [
    "evaluate_joint_selectors",
    "evaluate_prediction_table",
    "paired_trajectory_win_tie_loss",
    "trajectory_equal_mean",
    "trajectory_equal_values",
]
