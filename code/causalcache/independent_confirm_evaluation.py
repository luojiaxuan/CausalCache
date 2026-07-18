"""Fixed-denominator evaluation for the independent-gate confirm-20 slice."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.gate_v1_data import GateState
from causalcache.gate_v1_evaluation import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    paired_bootstrap_lower,
)
from causalcache.restoration_v2_2_label_table import primary_exact_subset_oracle


EXPECTED_CONFIRM_STATE_COUNT = 20
EXPECTED_CANDIDATE_EVENT_COUNT = 4
DEPLOYMENT_BUDGET = 2
MEMORY_SENSITIVE_EPSILON = 1e-12

HEURISTIC_ORDER = ("dynamic_recent", "ocr_rgb_v2", "policy_vision_v3")

MINIMUM_MEMORY_SENSITIVE_STATE_COUNT = 8
MINIMUM_ENSEMBLE_EXACT_RAW_RATIO = 0.85
MINIMUM_RETAINED_BASELINE_MASS = 0.50
MINIMUM_STRONGEST_HEURISTIC_POSITIVE_COUNT = 12
MINIMUM_PASSING_SEED_COUNT = 4
MINIMUM_SEED_EXACT_RAW_RATIO = 0.75
MAXIMUM_SEED_RATIO_POPULATION_STD = 0.10


def _reference_coverage_failure(reference_failure_count: int) -> dict[str, Any]:
    return {
        "status": "NO_GO_REFERENCE_COVERAGE",
        "evaluation_performed": False,
        "fixed_state_denominator": EXPECTED_CONFIRM_STATE_COUNT,
        "reference": {
            "expected_count": EXPECTED_CONFIRM_STATE_COUNT,
            "success_count": EXPECTED_CONFIRM_STATE_COUNT - reference_failure_count,
            "failure_count": reference_failure_count,
        },
        "bootstrap": {
            "unit": "trajectory",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "interval": "percentile",
            "quantile": "Hyndman_Fan_type_7",
        },
        "metrics": {},
        "records": [],
        "gate_checks": {"reference_coverage_20_of_20": False},
        "go": False,
    }


def _validate_reference_failure_count(value: int) -> int:
    if type(value) is not int or not 0 <= value <= EXPECTED_CONFIRM_STATE_COUNT:
        raise ValueError("reference_failure_count must be an integer from 0 through 20")
    return value


def _validate_states(states: Sequence[GateState]) -> tuple[GateState, ...]:
    if isinstance(states, (str, bytes, bytearray, Mapping)):
        raise ValueError("confirm states must be an ordered sequence")
    result = tuple(states)
    if len(result) != EXPECTED_CONFIRM_STATE_COUNT:
        raise ValueError("independent confirm requires the fixed 20-state denominator")
    if any(not isinstance(state, GateState) for state in result):
        raise TypeError("confirm state inventory must contain only GateState values")
    if len({state.source_id for state in result}) != EXPECTED_CONFIRM_STATE_COUNT:
        raise ValueError("independent confirm requires exactly one state per trajectory")
    if len({state.state_id for state in result}) != EXPECTED_CONFIRM_STATE_COUNT:
        raise ValueError("confirm state ids must be unique")
    for state in result:
        if (
            len(state.candidate_event_step_ids) != EXPECTED_CANDIDATE_EVENT_COUNT
            or state.candidate_event_step_ids != state.table.event_ids
        ):
            raise ValueError("every confirm state must contain the frozen four candidates")
    return result


def _selection_map(
    states: Sequence[GateState],
    value: Mapping[str, Sequence[int]],
    *,
    label: str,
) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a state-indexed mapping")
    expected = {state.state_id for state in states}
    if set(value) != expected:
        raise ValueError(f"{label} state inventory drifted from confirm-20")
    selections: dict[str, tuple[int, ...]] = {}
    for state in states:
        raw = value[state.state_id]
        if (
            isinstance(raw, (str, bytes, bytearray, Mapping))
            or not isinstance(raw, Sequence)
            or any(type(event_id) is not int for event_id in raw)
        ):
            raise ValueError(f"{label} contains a malformed selection")
        selected = tuple(raw)
        if (
            selected != tuple(sorted(selected))
            or len(set(selected)) != len(selected)
            or len(selected) > DEPLOYMENT_BUDGET
            or not set(selected).issubset(state.candidate_event_step_ids)
        ):
            raise ValueError(f"{label} contains an infeasible B=2 selection")
        selections[state.state_id] = selected
    return selections


def _seed_selection_maps(
    states: Sequence[GateState],
    value: Sequence[Mapping[str, Sequence[int]]],
) -> tuple[dict[str, tuple[int, ...]], ...]:
    if (
        isinstance(value, (str, bytes, bytearray, Mapping))
        or not isinstance(value, Sequence)
        or len(value) != 5
    ):
        raise ValueError("independent confirm requires exactly five sealed seed maps")
    return tuple(
        _selection_map(states, seed, label=f"independent seed {index}")
        for index, seed in enumerate(value)
    )


def _heuristic_selection_maps(
    states: Sequence[GateState],
    value: Mapping[str, Mapping[str, Sequence[int]]],
) -> dict[str, dict[str, tuple[int, ...]]]:
    if not isinstance(value, Mapping) or set(value) != set(HEURISTIC_ORDER):
        raise ValueError("confirm heuristic inventory drifted")
    return {
        name: _selection_map(states, value[name], label=f"heuristic {name}")
        for name in HEURISTIC_ORDER
    }


def _metric(state: GateState, selected: Sequence[int]) -> dict[str, Any]:
    baseline = state.table.distance(())
    utility = state.table.utility(selected)
    return {
        "selected": list(selected),
        "distance": state.table.distance(selected),
        "raw_utility": utility,
        "normalized_recovery": (
            utility / baseline if baseline > MEMORY_SENSITIVE_EPSILON else None
        ),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(denominator) or denominator <= MEMORY_SENSITIVE_EPSILON:
        return None
    return numerator / denominator


def _fixed20_zero_imputed_mean(values: Sequence[float | None]) -> float:
    if len(values) != EXPECTED_CONFIRM_STATE_COUNT:
        raise ValueError("normalized confirm metric requires the fixed 20 states")
    return math.fsum(0.0 if value is None else value for value in values) / (
        EXPECTED_CONFIRM_STATE_COUNT
    )


def _eligible_only_mean(values: Sequence[float | None]) -> float | None:
    eligible = [value for value in values if value is not None]
    return math.fsum(eligible) / len(eligible) if eligible else None


def evaluate_independent_confirm(
    states: Sequence[GateState],
    *,
    independent_ensemble_selections: Mapping[str, Sequence[int]],
    independent_seed_selections: Sequence[Mapping[str, Sequence[int]]],
    heuristic_selections: Mapping[str, Mapping[str, Sequence[int]]],
    reference_failure_count: int,
) -> dict[str, Any]:
    """Evaluate sealed independent selections only after reference coverage is 20/20."""
    failures = _validate_reference_failure_count(reference_failure_count)
    if failures:
        return _reference_coverage_failure(failures)

    frozen_states = _validate_states(states)
    ensemble = _selection_map(
        frozen_states,
        independent_ensemble_selections,
        label="independent ensemble",
    )
    seeds = _seed_selection_maps(frozen_states, independent_seed_selections)
    heuristics = _heuristic_selection_maps(frozen_states, heuristic_selections)

    records: list[dict[str, Any]] = []
    independent_utilities: list[float] = []
    exact_utilities: list[float] = []
    baselines: list[float] = []
    seed_utilities = [[] for _ in range(5)]
    heuristic_utilities = {name: [] for name in HEURISTIC_ORDER}
    independent_normalized: list[float | None] = []
    exact_normalized: list[float | None] = []
    seed_normalized: list[list[float | None]] = [[] for _ in range(5)]
    heuristic_normalized: dict[str, list[float | None]] = {
        name: [] for name in HEURISTIC_ORDER
    }
    memory_sensitive_count = 0

    for state in frozen_states:
        baseline = state.table.distance(())
        exact = primary_exact_subset_oracle(state.table)
        independent = _metric(state, ensemble[state.state_id])
        exact_metric = _metric(state, exact.coalition)
        seed_metrics = [
            _metric(state, seed[state.state_id]) for seed in seeds
        ]
        heuristic_metrics = {
            name: _metric(state, heuristics[name][state.state_id])
            for name in HEURISTIC_ORDER
        }
        memory_sensitive = baseline > MEMORY_SENSITIVE_EPSILON
        memory_sensitive_count += int(memory_sensitive)
        baselines.append(baseline)
        independent_utilities.append(float(independent["raw_utility"]))
        exact_utilities.append(float(exact_metric["raw_utility"]))
        independent_normalized.append(independent["normalized_recovery"])
        exact_normalized.append(exact_metric["normalized_recovery"])
        for index, metric in enumerate(seed_metrics):
            seed_utilities[index].append(float(metric["raw_utility"]))
            seed_normalized[index].append(metric["normalized_recovery"])
        for name, metric in heuristic_metrics.items():
            heuristic_utilities[name].append(float(metric["raw_utility"]))
            heuristic_normalized[name].append(metric["normalized_recovery"])
        records.append(
            {
                "source_id": state.source_id,
                "state_id": state.state_id,
                "decision_step_id": state.decision_step_id,
                "candidate_event_step_ids": list(state.candidate_event_step_ids),
                "baseline_distance": baseline,
                "memory_sensitive": memory_sensitive,
                "exact": exact_metric,
                "independent": independent,
                "seed_independent": seed_metrics,
                "heuristics": heuristic_metrics,
            }
        )

    independent_sum = math.fsum(independent_utilities)
    exact_sum = math.fsum(exact_utilities)
    baseline_sum = math.fsum(baselines)
    ensemble_exact_ratio = _ratio(independent_sum, exact_sum)
    retained_baseline_mass = _ratio(independent_sum, baseline_sum)
    seed_exact_ratios = [
        _ratio(math.fsum(values), exact_sum) for values in seed_utilities
    ]
    finite_seed_ratios = [
        value for value in seed_exact_ratios if value is not None
    ]
    seed_ratio_population_std = (
        statistics.pstdev(finite_seed_ratios)
        if len(finite_seed_ratios) == len(seed_exact_ratios)
        else None
    )
    fixed20_independent_normalized_mean = _fixed20_zero_imputed_mean(
        independent_normalized
    )
    fixed20_exact_normalized_mean = _fixed20_zero_imputed_mean(exact_normalized)
    fixed20_normalized_ratio_to_exact = _ratio(
        fixed20_independent_normalized_mean,
        fixed20_exact_normalized_mean,
    )
    eligible_independent_normalized_mean = _eligible_only_mean(
        independent_normalized
    )
    eligible_exact_normalized_mean = _eligible_only_mean(exact_normalized)
    eligible_normalized_ratio_to_exact = (
        None
        if eligible_independent_normalized_mean is None
        or eligible_exact_normalized_mean is None
        else _ratio(
            eligible_independent_normalized_mean,
            eligible_exact_normalized_mean,
        )
    )

    heuristic_comparisons: dict[str, Any] = {}
    for name in HEURISTIC_ORDER:
        deltas = [
            learned - heuristic
            for learned, heuristic in zip(
                independent_utilities, heuristic_utilities[name], strict=True
            )
        ]
        trajectory_deltas = {
            state.source_id: delta
            for state, delta in zip(frozen_states, deltas, strict=True)
        }
        heuristic_comparisons[name] = {
            "heuristic_raw_utility_sum": math.fsum(heuristic_utilities[name]),
            "independent_minus_heuristic_raw_mean": (
                math.fsum(deltas) / EXPECTED_CONFIRM_STATE_COUNT
            ),
            "independent_minus_heuristic_raw_deltas": deltas,
            "positive_trajectory_count": sum(delta > 0.0 for delta in deltas),
            "paired_bootstrap_lower": paired_bootstrap_lower(trajectory_deltas),
            "fixed20_zero_imputed_heuristic_mean_normalized_recovery": (
                _fixed20_zero_imputed_mean(heuristic_normalized[name])
            ),
            "fixed20_zero_imputed_independent_minus_heuristic_mean_normalized_delta": (
                _fixed20_zero_imputed_mean(
                    [
                        None
                        if learned is None or heuristic is None
                        else learned - heuristic
                        for learned, heuristic in zip(
                            independent_normalized,
                            heuristic_normalized[name],
                            strict=True,
                        )
                    ]
                )
            ),
            "legacy_eligible_only_heuristic_mean_normalized_recovery": (
                _eligible_only_mean(heuristic_normalized[name])
            ),
        }

    strongest_heuristic = max(
        HEURISTIC_ORDER,
        key=lambda name: (
            heuristic_comparisons[name]["heuristic_raw_utility_sum"],
            -HEURISTIC_ORDER.index(name),
        ),
    )
    passing_seed_count = sum(
        ratio is not None and ratio >= MINIMUM_SEED_EXACT_RAW_RATIO
        for ratio in seed_exact_ratios
    )
    metrics = {
        "memory_sensitive_state_count": memory_sensitive_count,
        "independent_raw_utility_sum": independent_sum,
        "exact_raw_utility_sum": exact_sum,
        "baseline_distance_sum": baseline_sum,
        "ensemble_exact_raw_utility_ratio": ensemble_exact_ratio,
        "retained_baseline_mass": retained_baseline_mass,
        "normalized_reporting_affects_go": False,
        "normalized_eligible_state_count": memory_sensitive_count,
        "fixed20_zero_imputed_independent_mean_normalized_recovery": (
            fixed20_independent_normalized_mean
        ),
        "fixed20_zero_imputed_exact_mean_normalized_recovery": (
            fixed20_exact_normalized_mean
        ),
        "fixed20_normalized_ratio_to_exact": fixed20_normalized_ratio_to_exact,
        "legacy_eligible_only_independent_mean_normalized_recovery": (
            eligible_independent_normalized_mean
        ),
        "legacy_eligible_only_exact_mean_normalized_recovery": (
            eligible_exact_normalized_mean
        ),
        "legacy_eligible_only_normalized_ratio_to_exact": (
            eligible_normalized_ratio_to_exact
        ),
        "heuristic_comparisons": heuristic_comparisons,
        "strongest_heuristic": strongest_heuristic,
        "strongest_heuristic_positive_trajectory_count": (
            heuristic_comparisons[strongest_heuristic]["positive_trajectory_count"]
        ),
        "strongest_heuristic_paired_bootstrap_lower": (
            heuristic_comparisons[strongest_heuristic]["paired_bootstrap_lower"]
        ),
        "individual_seed_exact_raw_utility_ratios": seed_exact_ratios,
        "individual_seed_fixed20_normalized_ratios_to_exact": [
            _ratio(
                _fixed20_zero_imputed_mean(values),
                fixed20_exact_normalized_mean,
            )
            for values in seed_normalized
        ],
        "individual_seed_legacy_eligible_only_normalized_ratios_to_exact": [
            (
                None
                if _eligible_only_mean(values) is None
                or eligible_exact_normalized_mean is None
                else _ratio(
                    _eligible_only_mean(values),
                    eligible_exact_normalized_mean,
                )
            )
            for values in seed_normalized
        ],
        "individual_seed_ratio_at_least_0_75_count": passing_seed_count,
        "seed_exact_ratio_population_std": seed_ratio_population_std,
    }
    checks = {
        "reference_coverage_20_of_20": True,
        "minimum_memory_sensitive_states": (
            memory_sensitive_count >= MINIMUM_MEMORY_SENSITIVE_STATE_COUNT
        ),
        "ensemble_exact_raw_utility_ratio": (
            ensemble_exact_ratio is not None
            and ensemble_exact_ratio >= MINIMUM_ENSEMBLE_EXACT_RAW_RATIO
        ),
        "retained_baseline_mass": (
            retained_baseline_mass is not None
            and retained_baseline_mass >= MINIMUM_RETAINED_BASELINE_MASS
        ),
        "independent_minus_every_heuristic_raw_mean": all(
            comparison["independent_minus_heuristic_raw_mean"] > 0.0
            for comparison in heuristic_comparisons.values()
        ),
        "strongest_heuristic_bootstrap_lower": (
            heuristic_comparisons[strongest_heuristic]["paired_bootstrap_lower"]
            > 0.0
        ),
        "strongest_heuristic_positive_trajectory_count": (
            heuristic_comparisons[strongest_heuristic]["positive_trajectory_count"]
            >= MINIMUM_STRONGEST_HEURISTIC_POSITIVE_COUNT
        ),
        "individual_seed_exact_ratio_pass_count": (
            passing_seed_count >= MINIMUM_PASSING_SEED_COUNT
        ),
        "seed_exact_ratio_population_std": (
            seed_ratio_population_std is not None
            and seed_ratio_population_std <= MAXIMUM_SEED_RATIO_POPULATION_STD
        ),
    }
    go = all(checks.values())
    return {
        "status": "GO_TO_PAIRED_CLOSED_LOOP" if go else "NO_GO_INDEPENDENT_CONFIRM",
        "evaluation_performed": True,
        "fixed_state_denominator": EXPECTED_CONFIRM_STATE_COUNT,
        "reference": {
            "expected_count": EXPECTED_CONFIRM_STATE_COUNT,
            "success_count": EXPECTED_CONFIRM_STATE_COUNT,
            "failure_count": 0,
        },
        "bootstrap": {
            "unit": "trajectory",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "interval": "percentile",
            "quantile": "Hyndman_Fan_type_7",
        },
        "metrics": metrics,
        "records": records,
        "gate_checks": checks,
        "go": go,
    }
