"""Pure failure decomposition over sealed fresh-16 labels and state records."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.gate_v1_data import (
    DEPLOYMENT_BUDGET,
    LABEL_TIE_EPSILON,
    NORMALIZATION_EPSILON,
    LabelState,
)
from causalcache.restoration_v2_2_label_table import (
    ValidatedDistanceTable,
    pair_interactions,
    primary_exact_subset_oracle,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_gate_v1_fresh16_failure_decomposition_v1"
STATUS = "FROZEN_GATE_V1_FRESH16_FAILURE_DECOMPOSITION_V1"
SOURCE_COUNT = 16
STATE_COUNT = 48
STATES_PER_SOURCE = 3
SEED_COUNT = 5
EVENT_COUNTS = (2, 3, 4)
METHOD_ORDER = ("exact", "true_greedy", "oracle_independent", "conditional", "independent")
INTERACTION_TERTILES = ("low", "mid", "high")
BASELINE_RANK_STRATA = ("bottom_quartile", "upper_three_quartiles")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 271_828
BOOTSTRAP_CONFIDENCE = 0.9
HEURISTIC_ORDER = ("dynamic_recent", "ocr_rgb_v2", "policy_vision_v3")

DECISION_THRESHOLDS: Mapping[str, Any] = {
    "bootstrap": {
        "confidence": BOOTSTRAP_CONFIDENCE,
        "interval": "percentile",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "unit": "trajectory",
    },
    "search_pass": {
        "all_required": [
            "normalized_recovery_G_over_E_at_least_0_95",
            "raw_utility_G_over_E_at_least_0_95",
            "n3_raw_utility_G_over_E_at_least_0_90",
            "n4_raw_utility_G_over_E_at_least_0_90",
        ],
        "n3_raw_utility_G_over_E_minimum": 0.90,
        "n4_raw_utility_G_over_E_minimum": 0.90,
        "normalized_recovery_G_over_E_minimum": 0.95,
        "raw_utility_G_over_E_minimum": 0.95,
    },
    "oracle_set_headroom_pass": {
        "all_required": [
            "normalized_mean_G_minus_J_at_least_0_02",
            "paired_trajectory_bootstrap_90_lower_strictly_positive",
            "positive_trajectory_count_at_least_12_of_16",
        ],
        "normalized_mean_delta_G_minus_J_minimum": 0.02,
        "paired_trajectory_bootstrap_lower_strictly_greater_than": 0.0,
        "positive_trajectory_count_minimum": 12,
        "trajectory_count": SOURCE_COUNT,
    },
    "material_student_gap": {
        "any_required": [
            "C_over_G_normalized_strictly_below_0_90",
            "C_over_G_raw_strictly_below_0_95",
        ],
        "normalized_recovery_C_over_G_maximum_exclusive": 0.90,
        "raw_utility_C_over_G_maximum_exclusive": 0.95,
    },
    "denominator_diagnostic": {
        "bottom_within_n_quartile_exact_raw_utility_share_maximum": 0.20,
        "bottom_within_n_quartile_positive_normalized_regret_share_minimum": 0.50,
        "diagnostic_only_cannot_change_routing": True,
        "raw_minus_normalized_ratio_gap_minimum": 0.15,
        "raw_utility_over_exact_raw_minimum": 0.90,
        "remove_bottom_within_n_quartile_normalized_ratio_improvement_minimum": 0.10,
    },
    "seed_isolated_diagnostic": {
        "drop_worst_seed_population_std_maximum": 0.08,
        "individual_seed_E_ratio_minimum": 0.75,
        "individual_seed_pass_count_minimum": 4,
        "individual_seed_total_count": SEED_COUNT,
        "median_individual_seed_E_ratio_minimum": 0.75,
        "not_a_routing_gate": True,
    },
    "prefix_completion_dominance": {
        "diagnostic_only": True,
        "minimum_share": 0.60,
        "share_definition": (
            "positive_distillation_regret_from_prefix_or_completion_errors_divided_by_"
            "total_positive_distillation_regret"
        ),
    },
    "learned_C_vs_I_replay": {
        "diagnostic_only": True,
        "must_equal_parent_sealed_report": True,
    },
    "final_routing": {
        "all_required": [
            "search_pass",
            "oracle_set_headroom_pass",
            "material_student_gap",
        ],
        "diagnostics_not_routing_gates": [
            "prefix_completion_dominance",
            "denominator_diagnostic",
            "seed_isolated_diagnostic",
            "learned_C_vs_I_replay",
        ],
        "fail_status": "NO_V2_CONDITIONAL_RESCUE",
        "fresh16_already_consumed": True,
        "fresh16_may_be_used_as_v2_holdout": False,
        "parent_v1_verdict_may_change": False,
        "pass_status": "ONE_V2_CONDITIONAL_RESCUE",
    },
}


@dataclass(frozen=True)
class _StateResult:
    label: LabelState
    selections: Mapping[str, tuple[int, ...]]
    utilities: Mapping[str, float]
    normalized: Mapping[str, float | None]
    seed_conditional_selections: tuple[tuple[int, ...], ...]
    seed_independent_selections: tuple[tuple[int, ...], ...]
    seed_conditional_utilities: tuple[float, ...]
    seed_independent_utilities: tuple[float, ...]
    heuristic_selections: Mapping[str, tuple[int, ...]]
    heuristic_utilities: Mapping[str, float]
    trace_rows: tuple[Mapping[str, Any], ...]
    search_gap: float
    student_gap: float
    total_gap: float
    prefix_lock_gap: float
    completion_stop_gap: float
    interaction_strength: float

    @property
    def baseline(self) -> float:
        return self.label.table.distance(())

    @property
    def event_count(self) -> int:
        return len(self.label.table.event_ids)


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


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be a finite number")
    return converted


def _contract_sha256(contract: Any) -> str:
    value = getattr(contract, "sha256", None)
    if value is None and isinstance(contract, Mapping):
        value = contract.get("sha256") or contract.get("contract_sha256")
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("failure-decomposition contract SHA256 is malformed")
    return value


def _contract_data(contract: Any) -> Mapping[str, Any]:
    value = getattr(contract, "data", contract)
    return _mapping(value, "failure-decomposition contract")


def _validate_contract_thresholds(contract: Any) -> Mapping[str, Any]:
    thresholds = _mapping(
        _contract_data(contract).get("routing_contract"), "routing contract"
    )
    projection = {
        key: thresholds.get(key)
        for key in DECISION_THRESHOLDS
    }
    if json.dumps(
        projection,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ) != json.dumps(
        DECISION_THRESHOLDS,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ):
        raise ValueError("failure-decomposition decision thresholds drifted")
    return thresholds


def _canonical_selection(
    value: Any,
    *,
    table: ValidatedDistanceTable,
    label: str,
) -> tuple[int, ...]:
    raw = _sequence(value, label)
    if any(type(item) is not int for item in raw):
        raise ValueError(f"{label} must contain integer event ids")
    selected = tuple(raw)
    if (
        selected != tuple(sorted(selected))
        or len(set(selected)) != len(selected)
        or len(selected) > DEPLOYMENT_BUDGET
        or not set(selected).issubset(table.event_ids)
    ):
        raise ValueError(f"{label} is infeasible")
    return selected


def _normalized(utility: float, baseline: float) -> float | None:
    return utility / baseline if baseline > NORMALIZATION_EPSILON else None


def _metric_selection(
    raw: Any,
    *,
    table: ValidatedDistanceTable,
    label: str,
) -> tuple[int, ...]:
    record = _mapping(raw, label)
    if set(record) != {"selected", "raw_utility", "normalized_recovery"}:
        raise ValueError(f"{label} schema drifted")
    selected = _canonical_selection(record["selected"], table=table, label=label)
    utility = table.utility(selected)
    normalized = _normalized(utility, table.distance(()))
    if record["raw_utility"] != utility or record["normalized_recovery"] != normalized:
        raise ValueError(f"{label} does not replay the validated LabelState")
    return selected


def _strict_positive_true_greedy(
    table: ValidatedDistanceTable,
) -> tuple[tuple[int, ...], tuple[Mapping[str, Any], ...]]:
    selected: tuple[int, ...] = ()
    trace = []
    for round_index in range(DEPLOYMENT_BUDGET):
        scores = tuple(
            (
                event,
                table.distance(selected)
                - table.distance(tuple(sorted((*selected, event)))),
            )
            for event in table.event_ids
            if event not in selected
        )
        event, gain = min(scores, key=lambda item: (-item[1], item[0]))
        if gain <= 0.0:
            trace.append(
                {
                    "round": round_index,
                    "selected_before": list(selected),
                    "decision": "stop",
                    "event_step_id": None,
                    "marginal_gain": gain,
                    "selected_after": list(selected),
                }
            )
            break
        restored = tuple(sorted((*selected, event)))
        trace.append(
            {
                "round": round_index,
                "selected_before": list(selected),
                "decision": "add",
                "event_step_id": event,
                "marginal_gain": gain,
                "selected_after": list(restored),
            }
        )
        selected = restored
    return selected, tuple(trace)


def budget_conditioned_oracle_independent_scores(
    table: ValidatedDistanceTable,
) -> Mapping[int, float]:
    """Replay exactly the independent raw target in gate_v1_data."""
    if not isinstance(table, ValidatedDistanceTable):
        raise TypeError("oracle independent scores require a ValidatedDistanceTable")
    scores = {}
    for event_id in table.event_ids:
        empty_gain = table.distance(()) - table.distance((event_id,))
        singleton_gains = [
            table.distance((other,))
            - table.distance(tuple(sorted((other, event_id))))
            for other in table.event_ids
            if other != event_id
        ]
        if not singleton_gains:
            raise ValueError("oracle independent target lacks singleton-conditioned gains")
        scores[event_id] = 0.5 * (
            empty_gain + math.fsum(singleton_gains) / len(singleton_gains)
        )
    return scores


def _oracle_independent_selection(table: ValidatedDistanceTable) -> tuple[int, ...]:
    scores = budget_conditioned_oracle_independent_scores(table)
    ranked = sorted(
        (event for event, value in scores.items() if value > 0.0),
        key=lambda event: (-scores[event], event),
    )
    return tuple(sorted(ranked[:DEPLOYMENT_BUDGET]))


def _interaction_strength(table: ValidatedDistanceTable) -> float:
    values = tuple(item.interaction for item in pair_interactions(table))
    if not values:
        return 0.0
    return (
        math.fsum(abs(value) for value in values)
        / len(values)
        / max(table.distance(()), NORMALIZATION_EPSILON)
    )


def _prefix_split(
    table: ValidatedDistanceTable,
    conditional: tuple[int, ...],
    exact_utility: float,
    first_selected_event: int | None,
) -> tuple[float, float]:
    conditional_utility = table.utility(conditional)
    if first_selected_event is None:
        if conditional:
            raise ValueError("nonempty conditional set has no first selected event")
        return 0.0, exact_utility - conditional_utility
    if first_selected_event not in conditional:
        raise ValueError("first selected event is absent from final conditional set")
    feasible = tuple(
        row
        for row in table.rows
        if first_selected_event in row.coalition
        and len(row.coalition) <= DEPLOYMENT_BUDGET
    )
    prefix_ceiling = max(row.utility for row in feasible)
    prefix_gap = exact_utility - prefix_ceiling
    completion_gap = prefix_ceiling - conditional_utility
    if prefix_gap < -1e-12 or completion_gap < -1e-12:
        raise ValueError("prefix-lock decomposition produced a negative regret")
    return max(0.0, prefix_gap), max(0.0, completion_gap)


def _parse_trace(
    raw: Any,
    *,
    table: ValidatedDistanceTable,
    expected_selection: tuple[int, ...],
) -> tuple[Mapping[str, Any], ...]:
    trace = _sequence(raw, "conditional selection trace")
    if not 1 <= len(trace) <= DEPLOYMENT_BUDGET:
        raise ValueError("conditional selection trace has an invalid round count")
    selected: tuple[int, ...] = ()
    stopped = False
    prior_diverged = False
    result = []
    for round_index, raw_round in enumerate(trace):
        record = _mapping(raw_round, "conditional trace round")
        if set(record) != {
            "round",
            "selected_before",
            "candidate_predicted_marginal_gains",
            "decision",
            "selected_event_step_id",
            "selected_after",
        }:
            raise ValueError("conditional trace round schema drifted")
        before = _canonical_selection(
            record["selected_before"], table=table, label="trace selected-before"
        )
        if before != selected or record["round"] != round_index or stopped:
            raise ValueError("conditional trace round order drifted")
        remaining = tuple(event for event in table.event_ids if event not in selected)
        score_rows = _sequence(
            record["candidate_predicted_marginal_gains"], "predicted score rows"
        )
        if len(score_rows) != len(remaining):
            raise ValueError("predicted score denominator drifted")
        predicted: dict[int, float] = {}
        for expected_event, raw_score in zip(remaining, score_rows, strict=True):
            score = _mapping(raw_score, "predicted score row")
            if set(score) != {"event_step_id", "predicted_marginal_gain"}:
                raise ValueError("predicted score row schema drifted")
            if score.get("event_step_id") != expected_event:
                raise ValueError("predicted score event order drifted")
            predicted[expected_event] = _finite(
                score.get("predicted_marginal_gain"), "predicted marginal gain"
            )
        predicted_event = min(remaining, key=lambda event: (-predicted[event], event))
        predicted_best = predicted[predicted_event]
        true_scores = {
            event: table.distance(selected)
            - table.distance(tuple(sorted((*selected, event))))
            for event in remaining
        }
        true_event = min(remaining, key=lambda event: (-true_scores[event], event))
        true_best = max(0.0, true_scores[true_event])
        decision = record.get("decision")
        chosen = record.get("selected_event_step_id")
        if predicted_best > 0.0:
            if decision != "add" or chosen != predicted_event:
                raise ValueError("conditional trace contradicts predicted positive argmax")
            after = tuple(sorted((*selected, predicted_event)))
            chosen_true_gain = true_scores[predicted_event]
        else:
            if decision != "stop" or chosen is not None:
                raise ValueError("conditional trace contradicts predicted stopping rule")
            after = selected
            chosen_true_gain = 0.0
            stopped = True
        if prior_diverged:
            error_type = "unreached_after_prior_divergence"
        elif decision == "stop" and true_best > 0.0:
            error_type = "premature_stop"
        elif decision == "add" and chosen_true_gain <= 0.0:
            error_type = "nonpositive_addition"
        elif decision == "add" and chosen != true_event:
            error_type = "wrong_event"
        else:
            error_type = "agree"
        observed_after = _canonical_selection(
            record["selected_after"], table=table, label="trace selected-after"
        )
        if observed_after != after:
            raise ValueError("conditional trace selected-after drifted")
        result.append(
            {
                "round": round_index,
                "selected_before": list(selected),
                "decision": decision,
                "selected_event_step_id": chosen,
                "predicted_best_event_step_id": predicted_event,
                "predicted_best_gain": predicted_best,
                "true_best_event_step_id": true_event if true_best > 0.0 else None,
                "true_best_gain": true_best,
                "chosen_true_gain": chosen_true_gain,
                "predicted_ranking_regret": (
                    predicted_best - predicted[true_event] if true_best > 0.0 else 0.0
                ),
                "true_regret": true_best - chosen_true_gain,
                "normalized_true_regret": _normalized(
                    true_best - chosen_true_gain, table.distance(())
                ),
                "false_add": decision == "add" and chosen_true_gain <= 0.0,
                "false_stop": decision == "stop" and true_best > 0.0,
                "student_error_type": error_type,
                "selected_after": list(after),
            }
        )
        prior_diverged = prior_diverged or error_type != "agree"
        selected = after
    if selected != expected_selection:
        raise ValueError("conditional trace final selection drifted")
    if not stopped and len(selected) != DEPLOYMENT_BUDGET:
        raise ValueError("conditional trace ended before budget without a stop")
    return tuple(result)


def _records_by_state(
    raw: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    if isinstance(raw, Mapping):
        values = tuple(raw.values())
        supplied_keys = tuple(raw.keys())
    else:
        values = tuple(_sequence(raw, "parent state records"))
        supplied_keys = ()
    result = {}
    for item in values:
        record = _mapping(item, "parent state record")
        state_id = record.get("state_id")
        if not isinstance(state_id, str) or not state_id or state_id in result:
            raise ValueError("parent state records contain an invalid or duplicate state id")
        result[state_id] = record
    if supplied_keys and set(supplied_keys) != set(result):
        raise ValueError("parent state-record mapping keys drifted from record state ids")
    return result


def _validate_fresh_roster(labels: Sequence[LabelState]) -> tuple[LabelState, ...]:
    states = tuple(labels)
    if len(states) != STATE_COUNT or any(not isinstance(item, LabelState) for item in states):
        raise ValueError("failure decomposition requires exactly 48 validated LabelStates")
    keys = {(item.source_id, item.state_id) for item in states}
    if len(keys) != STATE_COUNT:
        raise ValueError("fresh LabelState join keys are duplicated")
    sources = tuple(dict.fromkeys(item.source_id for item in states))
    if len(sources) != SOURCE_COUNT:
        raise ValueError("failure decomposition requires exactly 16 trajectories")
    for source_id in sources:
        source_states = tuple(item for item in states if item.source_id == source_id)
        if (
            len(source_states) != STATES_PER_SOURCE
            or tuple(len(item.table.event_ids) for item in source_states) != EVENT_COUNTS
            or tuple(item.decision_step_id for item in source_states) != (4, 5, 6)
        ):
            raise ValueError("fresh trajectory must contain ordered n=2/3/4 states")
    return states


def _state_result(label: LabelState, wrapper: Mapping[str, Any]) -> _StateResult:
    table = label.table
    if (
        wrapper.get("source_id") != label.source_id
        or wrapper.get("state_id") != label.state_id
        or wrapper.get("decision_step_id") != label.decision_step_id
        or wrapper.get("candidate_event_step_ids") != list(table.event_ids)
    ):
        raise ValueError("parent state-record identity differs from LabelState")
    evaluation = _mapping(wrapper.get("evaluation"), "parent state evaluation")
    if (
        evaluation.get("source_id") != label.source_id
        or evaluation.get("event_count") != len(table.event_ids)
        or evaluation.get("baseline") != table.distance(())
    ):
        raise ValueError("parent state evaluation identity or baseline drifted")
    exact = primary_exact_subset_oracle(table)
    exact_record = _metric_selection(evaluation.get("exact"), table=table, label="exact")
    if exact_record != exact.coalition:
        raise ValueError("parent state exact coalition drifted")
    conditional = _metric_selection(
        evaluation.get("conditional"), table=table, label="conditional"
    )
    independent = _metric_selection(
        evaluation.get("independent"), table=table, label="independent"
    )
    seed_conditional_raw = _sequence(
        evaluation.get("seed_conditional"), "conditional seed metrics"
    )
    seed_independent_raw = _sequence(
        evaluation.get("seed_independent"), "independent seed metrics"
    )
    if len(seed_conditional_raw) != SEED_COUNT or len(seed_independent_raw) != SEED_COUNT:
        raise ValueError("parent state seed denominator must be exactly five")
    seed_conditional = tuple(
        _metric_selection(item, table=table, label=f"conditional seed {seed}")
        for seed, item in enumerate(seed_conditional_raw)
    )
    seed_independent = tuple(
        _metric_selection(item, table=table, label=f"independent seed {seed}")
        for seed, item in enumerate(seed_independent_raw)
    )
    heuristic_raw = _mapping(evaluation.get("heuristics"), "heuristic metrics")
    if set(heuristic_raw) != set(HEURISTIC_ORDER):
        raise ValueError("parent heuristic inventory drifted")
    heuristics = {
        name: _metric_selection(
            heuristic_raw[name], table=table, label=f"heuristic {name}"
        )
        for name in HEURISTIC_ORDER
    }
    true_greedy, _ = _strict_positive_true_greedy(table)
    oracle_independent = _oracle_independent_selection(table)
    selections = {
        "exact": exact.coalition,
        "true_greedy": true_greedy,
        "oracle_independent": oracle_independent,
        "conditional": conditional,
        "independent": independent,
    }
    utilities = {name: table.utility(selected) for name, selected in selections.items()}
    normalized = {
        name: _normalized(utility, table.distance(()))
        for name, utility in utilities.items()
    }
    search_gap = utilities["exact"] - utilities["true_greedy"]
    student_gap = utilities["true_greedy"] - utilities["conditional"]
    total_gap = utilities["exact"] - utilities["conditional"]
    if not math.isclose(
        total_gap,
        search_gap + student_gap,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("state exact-search-student gap is not additive")
    trace = _parse_trace(
        wrapper.get("conditional_selection_trace"),
        table=table,
        expected_selection=conditional,
    )
    first_selected_event = next(
        (
            int(row["selected_event_step_id"])
            for row in trace
            if row["decision"] == "add"
        ),
        None,
    )
    prefix_gap, completion_gap = _prefix_split(
        table,
        conditional,
        utilities["exact"],
        first_selected_event,
    )
    if not math.isclose(
        total_gap,
        prefix_gap + completion_gap,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("state prefix/completion gap is not additive")
    addition_count = evaluation.get("conditional_selected_addition_count")
    nonpositive_count = evaluation.get(
        "conditional_true_nonpositive_addition_count"
    )
    observed_nonpositive = sum(
        row["decision"] == "add" and row["chosen_true_gain"] <= 0.0
        for row in trace
    )
    if (
        type(addition_count) is not int
        or addition_count != len(conditional)
        or type(nonpositive_count) is not int
        or nonpositive_count != observed_nonpositive
    ):
        raise ValueError("parent conditional addition counters drifted")
    return _StateResult(
        label=label,
        selections=selections,
        utilities=utilities,
        normalized=normalized,
        seed_conditional_selections=seed_conditional,
        seed_independent_selections=seed_independent,
        seed_conditional_utilities=tuple(table.utility(item) for item in seed_conditional),
        seed_independent_utilities=tuple(table.utility(item) for item in seed_independent),
        heuristic_selections=heuristics,
        heuristic_utilities={name: table.utility(item) for name, item in heuristics.items()},
        trace_rows=trace,
        search_gap=search_gap,
        student_gap=student_gap,
        total_gap=total_gap,
        prefix_lock_gap=prefix_gap,
        completion_stop_gap=completion_gap,
        interaction_strength=_interaction_strength(table),
    )


def _trajectory_equal_mean(
    states: Sequence[_StateResult],
    values: Mapping[str, float | None],
) -> float | None:
    by_source: dict[str, list[float]] = defaultdict(list)
    for state in states:
        value = values[state.label.state_id]
        if value is not None:
            by_source[state.label.source_id].append(float(value))
    source_means = [math.fsum(items) / len(items) for items in by_source.values() if items]
    return math.fsum(source_means) / len(source_means) if source_means else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= NORMALIZATION_EPSILON:
        return None
    return numerator / denominator


def _slice_summary(states: Sequence[_StateResult]) -> Mapping[str, Any]:
    selected = tuple(states)
    methods = {}
    for method in METHOD_ORDER:
        raw = _trajectory_equal_mean(
            selected, {item.label.state_id: item.utilities[method] for item in selected}
        )
        normalized = _trajectory_equal_mean(
            selected, {item.label.state_id: item.normalized[method] for item in selected}
        )
        methods[method] = {
            "mean_raw_utility": raw,
            "mean_normalized_recovery": normalized,
        }
    exact = methods["exact"]
    for method in METHOD_ORDER:
        methods[method]["raw_ratio_to_exact"] = _ratio(
            methods[method]["mean_raw_utility"], exact["mean_raw_utility"]
        )
        methods[method]["normalized_ratio_to_exact"] = _ratio(
            methods[method]["mean_normalized_recovery"],
            exact["mean_normalized_recovery"],
        )
    raw_values = tuple(
        methods[name]["mean_raw_utility"]
        for name in ("exact", "true_greedy", "conditional")
    )
    if any(value is None for value in raw_values):
        search_raw = student_raw = total_raw = None
    else:
        exact_raw, greedy_raw, conditional_raw = raw_values
        search_raw = exact_raw - greedy_raw
        student_raw = greedy_raw - conditional_raw
        total_raw = exact_raw - conditional_raw
        if not math.isclose(
            total_raw,
            search_raw + student_raw,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("aggregate raw exact-search-student gap is not additive")
    normalized_values = tuple(
        methods[name]["mean_normalized_recovery"] for name in ("exact", "true_greedy", "conditional")
    )
    if any(value is None for value in normalized_values):
        search_normalized = student_normalized = total_normalized = None
    else:
        exact_norm, greedy_norm, conditional_norm = normalized_values
        search_normalized = exact_norm - greedy_norm
        student_normalized = greedy_norm - conditional_norm
        total_normalized = exact_norm - conditional_norm
        if not math.isclose(
            total_normalized,
            search_normalized + student_normalized,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("aggregate normalized exact-search-student gap is not additive")
    return {
        "state_count": len(selected),
        "trajectory_count": len({item.label.source_id for item in selected}),
        "methods": methods,
        "gaps": {
            "raw": {
                "exact_minus_true_greedy": search_raw,
                "true_greedy_minus_conditional": student_raw,
                "exact_minus_conditional": total_raw,
                "additive_identity_valid": total_raw is not None,
            },
            "normalized": {
                "exact_minus_true_greedy": search_normalized,
                "true_greedy_minus_conditional": student_normalized,
                "exact_minus_conditional": total_normalized,
                "additive_identity_valid": total_normalized is not None,
            },
        },
    }


def _rank_assignments(
    states: Sequence[_StateResult],
) -> tuple[Mapping[str, str], Mapping[str, str]]:
    baseline: dict[str, str] = {}
    interaction: dict[str, str] = {}
    for event_count in EVENT_COUNTS:
        group = tuple(item for item in states if item.event_count == event_count)
        if len(group) != SOURCE_COUNT:
            raise ValueError("each n stratum must contain exactly 16 states")
        baseline_order = sorted(
            group,
            key=lambda item: (
                item.baseline,
                item.label.source_id,
                item.label.state_id,
            ),
        )
        bottom_count = math.ceil(len(group) / 4)
        for index, item in enumerate(baseline_order):
            baseline[item.label.state_id] = (
                "bottom_quartile" if index < bottom_count else "upper_three_quartiles"
            )
        interaction_order = sorted(
            group,
            key=lambda item: (
                item.interaction_strength,
                item.label.source_id,
                item.label.state_id,
            ),
        )
        for index, item in enumerate(interaction_order):
            rank = min(2, index * 3 // len(group))
            interaction[item.label.state_id] = INTERACTION_TERTILES[rank]
    return baseline, interaction


def _trajectory_deltas(
    states: Sequence[_StateResult],
    left: str,
    right: str,
) -> Mapping[str, float]:
    by_source: dict[str, list[float]] = defaultdict(list)
    for state in states:
        left_value = state.normalized[left]
        right_value = state.normalized[right]
        if left_value is not None and right_value is not None:
            by_source[state.label.source_id].append(left_value - right_value)
    if set(by_source) != {item.label.source_id for item in states}:
        raise ValueError("paired normalized trajectory delta lost a trajectory")
    return {
        source: math.fsum(values) / len(values)
        for source, values in by_source.items()
    }


def _type7_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile input is empty or malformed")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _paired_bootstrap_lower(deltas: Mapping[str, float]) -> float:
    values = tuple(deltas[source] for source in sorted(deltas))
    generator = random.Random(BOOTSTRAP_SEED)
    means = [
        math.fsum(values[generator.randrange(len(values))] for _ in values)
        / len(values)
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return _type7_quantile(means, (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0)


def _round_summary(states: Sequence[_StateResult]) -> Mapping[str, Any]:
    result = {}
    for round_index in range(DEPLOYMENT_BUDGET):
        rows = tuple(
            (state, row)
            for state in states
            for row in state.trace_rows
            if row["round"] == round_index
        )
        if not rows:
            result[str(round_index)] = {
                "state_count": 0,
                "trajectory_count": 0,
                "mean_predicted_ranking_regret": None,
                "mean_true_regret": None,
                "mean_normalized_true_regret": None,
                "false_add_count": 0,
                "false_stop_count": 0,
                "student_error_type_counts": {},
            }
            continue
        pseudo_states = tuple(item[0] for item in rows)
        predicted_values = {
            state.label.state_id: float(row["predicted_ranking_regret"])
            for state, row in rows
        }
        true_values = {
            state.label.state_id: float(row["true_regret"])
            for state, row in rows
        }
        normalized_values = {
            state.label.state_id: row["normalized_true_regret"] for state, row in rows
        }
        result[str(round_index)] = {
            "state_count": len(rows),
            "trajectory_count": len({state.label.source_id for state, _ in rows}),
            "mean_predicted_ranking_regret": _trajectory_equal_mean(
                pseudo_states, predicted_values
            ),
            "mean_true_regret": _trajectory_equal_mean(pseudo_states, true_values),
            "mean_normalized_true_regret": _trajectory_equal_mean(
                pseudo_states, normalized_values
            ),
            "false_add_count": sum(bool(row["false_add"]) for _, row in rows),
            "false_stop_count": sum(bool(row["false_stop"]) for _, row in rows),
            "student_error_type_counts": {
                name: sum(row["student_error_type"] == name for _, row in rows)
                for name in (
                    "agree",
                    "wrong_event",
                    "premature_stop",
                    "nonpositive_addition",
                    "unreached_after_prior_divergence",
                )
            },
        }
    return result


def _denominator_diagnostic(
    states: Sequence[_StateResult],
    baseline_assignments: Mapping[str, str],
    overall: Mapping[str, Any],
) -> Mapping[str, Any]:
    eligible_by_source: dict[str, int] = defaultdict(int)
    for state in states:
        if state.baseline > NORMALIZATION_EPSILON:
            eligible_by_source[state.label.source_id] += 1
    eligible_sources = len(eligible_by_source)
    normalized_regret_total = 0.0
    normalized_regret_bottom = 0.0
    for state in states:
        if state.baseline <= NORMALIZATION_EPSILON:
            continue
        if eligible_sources == 0:
            raise ValueError("denominator diagnostic has no normalization-eligible source")
        weight = 1.0 / eligible_sources / eligible_by_source[state.label.source_id]
        contribution = weight * max(0.0, state.total_gap / state.baseline)
        normalized_regret_total += contribution
        if baseline_assignments[state.label.state_id] == "bottom_quartile":
            normalized_regret_bottom += contribution
    raw_weight = 1.0 / SOURCE_COUNT / STATES_PER_SOURCE
    exact_raw_total = math.fsum(raw_weight * state.utilities["exact"] for state in states)
    exact_raw_bottom = math.fsum(
        raw_weight * state.utilities["exact"]
        for state in states
        if baseline_assignments[state.label.state_id] == "bottom_quartile"
    )
    upper = tuple(
        state
        for state in states
        if baseline_assignments[state.label.state_id] == "upper_three_quartiles"
    )
    upper_summary = _slice_summary(upper)
    full_ratio = overall["methods"]["conditional"]["normalized_ratio_to_exact"]
    upper_ratio = upper_summary["methods"]["conditional"]["normalized_ratio_to_exact"]
    return {
        "bottom_quartile_positive_normalized_regret_share": (
            normalized_regret_bottom / normalized_regret_total
            if normalized_regret_total > 0.0
            else 0.0
        ),
        "bottom_quartile_exact_raw_utility_share": (
            exact_raw_bottom / exact_raw_total if exact_raw_total > 0.0 else 0.0
        ),
        "full_normalized_ratio_to_exact": full_ratio,
        "upper_three_quartiles_normalized_ratio_to_exact": upper_ratio,
        "drop_bottom_quartile_normalized_ratio_improvement": (
            upper_ratio - full_ratio
            if upper_ratio is not None and full_ratio is not None
            else None
        ),
    }


def _seed_summary(states: Sequence[_StateResult], overall: Mapping[str, Any]) -> Mapping[str, Any]:
    exact_norm = overall["methods"]["exact"]["mean_normalized_recovery"]
    exact_raw = overall["methods"]["exact"]["mean_raw_utility"]
    normalized_means = []
    raw_means = []
    for seed in range(SEED_COUNT):
        normalized = {
            state.label.state_id: _normalized(
                state.seed_conditional_utilities[seed], state.baseline
            )
            for state in states
        }
        raw = {
            state.label.state_id: state.seed_conditional_utilities[seed]
            for state in states
        }
        normalized_means.append(_trajectory_equal_mean(states, normalized))
        raw_means.append(_trajectory_equal_mean(states, raw))
    normalized_ratios = [_ratio(value, exact_norm) for value in normalized_means]
    raw_ratios = [_ratio(value, exact_raw) for value in raw_means]
    if any(value is None for value in normalized_ratios):
        raise ValueError("seed normalized ratio denominator is invalid")
    worst = min(range(SEED_COUNT), key=lambda seed: (normalized_ratios[seed], seed))
    leave_worst = [value for seed, value in enumerate(normalized_means) if seed != worst]
    threshold = DECISION_THRESHOLDS["seed_isolated_diagnostic"]
    pass_count = sum(
        value >= threshold["individual_seed_E_ratio_minimum"]
        for value in normalized_ratios
    )
    median = statistics.median(normalized_ratios)
    leave_worst_std = statistics.pstdev(leave_worst)
    isolated = (
        median >= threshold["median_individual_seed_E_ratio_minimum"]
        and pass_count >= threshold["individual_seed_pass_count_minimum"]
        and leave_worst_std
        <= threshold["drop_worst_seed_population_std_maximum"]
    )
    return {
        "conditional_seed_mean_normalized_recovery": normalized_means,
        "conditional_seed_mean_raw_utility": raw_means,
        "conditional_seed_normalized_ratio_to_exact": normalized_ratios,
        "conditional_seed_raw_ratio_to_exact": raw_ratios,
        "individual_seed_pass_count": pass_count,
        "median_normalized_ratio_to_exact": median,
        "worst_seed": worst,
        "leave_worst_seed_normalized_recovery_population_std": leave_worst_std,
        "instability_classification": "isolated" if isolated else "systemic",
    }


def _decision_tree(
    states: Sequence[_StateResult],
    overall: Mapping[str, Any],
    by_n: Mapping[str, Mapping[str, Any]],
    baseline_assignments: Mapping[str, str],
    thresholds: Mapping[str, Any],
) -> Mapping[str, Any]:
    greedy = overall["methods"]["true_greedy"]
    search_n = {
        key: by_n[key]["methods"]["true_greedy"]["raw_ratio_to_exact"]
        for key in ("3", "4")
    }
    search_threshold = _mapping(thresholds["search_pass"], "search thresholds")
    search_pass = (
        greedy["normalized_ratio_to_exact"] is not None
        and greedy["normalized_ratio_to_exact"]
        >= search_threshold["normalized_recovery_G_over_E_minimum"]
        and greedy["raw_ratio_to_exact"] is not None
        and greedy["raw_ratio_to_exact"]
        >= search_threshold["raw_utility_G_over_E_minimum"]
        and search_n["3"] is not None
        and search_n["3"] >= search_threshold["n3_raw_utility_G_over_E_minimum"]
        and search_n["4"] is not None
        and search_n["4"] >= search_threshold["n4_raw_utility_G_over_E_minimum"]
    )
    teacher_deltas = _trajectory_deltas(states, "true_greedy", "oracle_independent")
    teacher_mean = math.fsum(teacher_deltas.values()) / len(teacher_deltas)
    teacher_lower = _paired_bootstrap_lower(teacher_deltas)
    teacher_positive = sum(value > 0.0 for value in teacher_deltas.values())
    teacher_threshold = _mapping(
        thresholds["oracle_set_headroom_pass"], "oracle set-headroom thresholds"
    )
    teacher_pass = (
        teacher_mean
        >= teacher_threshold["normalized_mean_delta_G_minus_J_minimum"]
        and teacher_lower
        > teacher_threshold[
            "paired_trajectory_bootstrap_lower_strictly_greater_than"
        ]
        and teacher_positive
        >= teacher_threshold["positive_trajectory_count_minimum"]
    )
    conditional = overall["methods"]["conditional"]
    true_greedy = overall["methods"]["true_greedy"]
    student_normalized_ratio = _ratio(
        conditional["mean_normalized_recovery"],
        true_greedy["mean_normalized_recovery"],
    )
    student_raw_ratio = _ratio(
        conditional["mean_raw_utility"], true_greedy["mean_raw_utility"]
    )
    student_threshold = _mapping(
        thresholds["material_student_gap"], "material student-gap thresholds"
    )
    student_material = (
        student_normalized_ratio is not None
        and student_raw_ratio is not None
        and (
            student_normalized_ratio
            < student_threshold["normalized_recovery_C_over_G_maximum_exclusive"]
            or student_raw_ratio
            < student_threshold["raw_utility_C_over_G_maximum_exclusive"]
        )
    )
    denominator = _denominator_diagnostic(states, baseline_assignments, overall)
    exact_conditional = conditional
    ratio_values = (
        exact_conditional["raw_ratio_to_exact"],
        exact_conditional["normalized_ratio_to_exact"],
    )
    raw_minus_normalized = (
        ratio_values[0] - ratio_values[1]
        if all(value is not None for value in ratio_values)
        else None
    )
    denominator_threshold = _mapping(
        thresholds["denominator_diagnostic"], "denominator thresholds"
    )
    denominator_dominated = (
        exact_conditional["raw_ratio_to_exact"] is not None
        and exact_conditional["raw_ratio_to_exact"]
        >= denominator_threshold["raw_utility_over_exact_raw_minimum"]
        and raw_minus_normalized is not None
        and raw_minus_normalized
        >= denominator_threshold["raw_minus_normalized_ratio_gap_minimum"]
        and denominator["bottom_quartile_positive_normalized_regret_share"]
        >= denominator_threshold[
            "bottom_within_n_quartile_positive_normalized_regret_share_minimum"
        ]
        and denominator["bottom_quartile_exact_raw_utility_share"]
        <= denominator_threshold[
            "bottom_within_n_quartile_exact_raw_utility_share_maximum"
        ]
        and denominator["drop_bottom_quartile_normalized_ratio_improvement"]
        is not None
        and denominator["drop_bottom_quartile_normalized_ratio_improvement"]
        >= denominator_threshold[
            "remove_bottom_within_n_quartile_normalized_ratio_improvement_minimum"
        ]
    )
    seeds = _seed_summary(states, overall)
    prefix_positive = _trajectory_equal_mean(
        states,
        {
            state.label.state_id: max(0.0, state.prefix_lock_gap)
            for state in states
        },
    )
    completion_positive = _trajectory_equal_mean(
        states,
        {
            state.label.state_id: max(0.0, state.completion_stop_gap)
            for state in states
        },
    )
    total_positive = _trajectory_equal_mean(
        states,
        {
            state.label.state_id: max(0.0, state.total_gap)
            for state in states
        },
    )
    prefix_share = (
        prefix_positive / total_positive
        if total_positive is not None and total_positive > 0.0
        else 0.0
    )
    completion_share = (
        completion_positive / total_positive
        if total_positive is not None and total_positive > 0.0
        else 0.0
    )
    prefix_threshold = _mapping(
        thresholds["prefix_completion_dominance"],
        "prefix/completion diagnostic thresholds",
    )
    c_i_deltas = _trajectory_deltas(states, "conditional", "independent")
    c_i_mean = math.fsum(c_i_deltas.values()) / len(c_i_deltas)
    final_routing = _mapping(thresholds["final_routing"], "final routing")
    route_pass = search_pass and teacher_pass and student_material
    if route_pass:
        outcome = final_routing["pass_status"]
        failed_at = None
    elif not search_pass:
        outcome = final_routing["fail_status"]
        failed_at = "search_pass"
    elif not teacher_pass:
        outcome = final_routing["fail_status"]
        failed_at = "oracle_set_headroom_pass"
    else:
        outcome = final_routing["fail_status"]
        failed_at = "material_student_gap"
    return {
        "thresholds": {key: thresholds[key] for key in DECISION_THRESHOLDS},
        "search": {
            "normalized_ratio_to_exact": greedy["normalized_ratio_to_exact"],
            "raw_ratio_to_exact": greedy["raw_ratio_to_exact"],
            "n3_n4_raw_ratio_to_exact": search_n,
            "pass": search_pass,
        },
        "oracle_set_conditioning_headroom": {
            "mean_normalized_delta": teacher_mean,
            "paired_bootstrap_lower": teacher_lower,
            "positive_trajectory_count": teacher_positive,
            "pass": teacher_pass,
        },
        "student": {
            "conditional_normalized_ratio_to_true_greedy": student_normalized_ratio,
            "conditional_raw_ratio_to_true_greedy": student_raw_ratio,
            "material_gap": student_material,
        },
        "denominator": {
            **denominator,
            "raw_minus_normalized_ratio_to_exact": raw_minus_normalized,
            "denominator_dominated": denominator_dominated,
            "diagnostic_only_cannot_change_v1_verdict": True,
        },
        "seed_stability": seeds,
        "prefix_completion": {
            "mean_positive_prefix_lock_gap_raw": prefix_positive,
            "mean_positive_completion_stop_gap_raw": completion_positive,
            "mean_positive_total_exact_student_gap_raw": total_positive,
            "prefix_share": prefix_share,
            "completion_share": completion_share,
            "prefix_dominant": prefix_share >= prefix_threshold["minimum_share"],
            "completion_dominant": (
                completion_share >= prefix_threshold["minimum_share"]
            ),
            "diagnostic_only": True,
        },
        "learned_C_vs_I_replay": {
            "mean_normalized_delta": c_i_mean,
            "paired_bootstrap_lower": _paired_bootstrap_lower(c_i_deltas),
            "positive_trajectory_count": sum(
                value > 0.0 for value in c_i_deltas.values()
            ),
            "mean_raw_utility_delta": (
                conditional["mean_raw_utility"]
                - overall["methods"]["independent"]["mean_raw_utility"]
            ),
            "recomputed_from_parent_sealed_state_records": True,
            "diagnostic_only": True,
        },
        "outcome": outcome,
        "failed_at": failed_at,
        "v2_conditional_rescue_authorized": route_pass,
        "v1_verdict_unchanged": True,
        "fresh16_is_consumed_diagnostic_only": True,
        "confirm_matched_nll_closed_loop_authorized": False,
        "untouched_holdout_required_before_any_post_primary_stage": True,
    }


def _state_row(
    state: _StateResult,
    *,
    ordinal: int,
    baseline_rank: str,
    interaction_tertile: str,
) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": STATUS,
        "ordinal": ordinal,
        "source_id": state.label.source_id,
        "state_id": state.label.state_id,
        "decision_step_id": state.label.decision_step_id,
        "candidate_event_step_ids": list(state.label.table.event_ids),
        "event_count": state.event_count,
        "baseline_distance": state.baseline,
        "baseline_rank_within_n": baseline_rank,
        "interaction_strength": state.interaction_strength,
        "interaction_rank_tertile_within_n": interaction_tertile,
        "methods": {
            method: {
                "selected": list(state.selections[method]),
                "raw_utility": state.utilities[method],
                "normalized_recovery": state.normalized[method],
            }
            for method in METHOD_ORDER
        },
        "seed_conditional": [
            {
                "seed": seed,
                "selected": list(state.seed_conditional_selections[seed]),
                "raw_utility": state.seed_conditional_utilities[seed],
                "normalized_recovery": _normalized(
                    state.seed_conditional_utilities[seed], state.baseline
                ),
            }
            for seed in range(SEED_COUNT)
        ],
        "seed_independent": [
            {
                "seed": seed,
                "selected": list(state.seed_independent_selections[seed]),
                "raw_utility": state.seed_independent_utilities[seed],
                "normalized_recovery": _normalized(
                    state.seed_independent_utilities[seed], state.baseline
                ),
            }
            for seed in range(SEED_COUNT)
        ],
        "heuristics": {
            name: {
                "selected": list(state.heuristic_selections[name]),
                "raw_utility": state.heuristic_utilities[name],
                "normalized_recovery": _normalized(
                    state.heuristic_utilities[name], state.baseline
                ),
            }
            for name in HEURISTIC_ORDER
        },
        "gap_decomposition": {
            "search_gap_raw": state.search_gap,
            "student_gap_raw": state.student_gap,
            "total_exact_minus_conditional_raw": state.total_gap,
            "search_plus_student_equals_total": True,
            "prefix_lock_gap_raw": state.prefix_lock_gap,
            "completion_stop_gap_raw": state.completion_stop_gap,
            "prefix_plus_completion_equals_total": True,
            "first_selected_event_step_id": next(
                (
                    row["selected_event_step_id"]
                    for row in state.trace_rows
                    if row["decision"] == "add"
                ),
                None,
            ),
        },
        "rounds": [dict(item) for item in state.trace_rows],
    }


def _trajectory_row(states: Sequence[_StateResult]) -> Mapping[str, Any]:
    selected = tuple(states)
    if (
        len(selected) != STATES_PER_SOURCE
        or len({item.label.source_id for item in selected}) != 1
    ):
        raise ValueError("trajectory row requires one complete three-state source")
    summary = _slice_summary(selected)

    def mean_delta(left: str, right: str, *, normalized: bool) -> float | None:
        values = []
        for state in selected:
            if normalized:
                left_value = state.normalized[left]
                right_value = state.normalized[right]
                if left_value is None or right_value is None:
                    continue
            else:
                left_value = state.utilities[left]
                right_value = state.utilities[right]
            values.append(left_value - right_value)
        return math.fsum(values) / len(values) if values else None

    return {
        "source_id": selected[0].label.source_id,
        "state_count": len(selected),
        "state_ids": [item.label.state_id for item in selected],
        "methods": summary["methods"],
        "gaps": summary["gaps"],
        "oracle_set_conditioning_headroom": {
            "G_minus_J_mean_raw_utility": mean_delta(
                "true_greedy", "oracle_independent", normalized=False
            ),
            "G_minus_J_mean_normalized_recovery": mean_delta(
                "true_greedy", "oracle_independent", normalized=True
            ),
        },
        "learned_C_vs_I_replay": {
            "C_minus_I_mean_raw_utility": mean_delta(
                "conditional", "independent", normalized=False
            ),
            "C_minus_I_mean_normalized_recovery": mean_delta(
                "conditional", "independent", normalized=True
            ),
        },
    }


def _denominator_bin(state: _StateResult) -> str:
    if state.baseline <= NORMALIZATION_EPSILON:
        return "normalization_ineligible"
    if state.baseline <= 0.001:
        return "tiny"
    if state.baseline <= 0.01:
        return "small"
    return "regular"


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            _mapping(row, "failure-decomposition JSONL row"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def build_failure_decomposition(
    label_states: Sequence[LabelState],
    parent_state_records: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    contract: Any,
) -> Mapping[str, Any]:
    """Reduce sealed fresh-16 objects without file, network, model, or policy access."""
    contract_sha256 = _contract_sha256(contract)
    routing_contract = _validate_contract_thresholds(contract)
    labels = _validate_fresh_roster(label_states)
    records = _records_by_state(parent_state_records)
    if set(records) != {item.state_id for item in labels}:
        raise ValueError("LabelState and parent state-record inventories differ")
    states = tuple(_state_result(item, records[item.state_id]) for item in labels)
    baseline_assignments, interaction_assignments = _rank_assignments(states)
    overall = _slice_summary(states)
    by_n = {
        str(event_count): _slice_summary(
            tuple(item for item in states if item.event_count == event_count)
        )
        for event_count in EVENT_COUNTS
    }
    baseline_strata = {
        str(event_count): {
            name: _slice_summary(
                tuple(
                    item
                    for item in states
                    if item.event_count == event_count
                    and baseline_assignments[item.label.state_id] == name
                )
            )
            for name in BASELINE_RANK_STRATA
        }
        for event_count in EVENT_COUNTS
    }
    interaction_strata = {
        str(event_count): {
            name: _slice_summary(
                tuple(
                    item
                    for item in states
                    if item.event_count == event_count
                    and interaction_assignments[item.label.state_id] == name
                )
            )
            for name in INTERACTION_TERTILES
        }
        for event_count in EVENT_COUNTS
    }
    denominator_strata = {
        name: _slice_summary(
            tuple(item for item in states if _denominator_bin(item) == name)
        )
        for name in ("normalization_ineligible", "tiny", "small", "regular")
    }
    heuristic_means = {
        name: _trajectory_equal_mean(
            states,
            {
                item.label.state_id: _normalized(
                    item.heuristic_utilities[name], item.baseline
                )
                for item in states
            },
        )
        for name in HEURISTIC_ORDER
    }
    strongest = max(
        HEURISTIC_ORDER,
        key=lambda name: (heuristic_means[name], -HEURISTIC_ORDER.index(name)),
    )
    state_rows = [
        _state_row(
            item,
            ordinal=ordinal,
            baseline_rank=baseline_assignments[item.label.state_id],
            interaction_tertile=interaction_assignments[item.label.state_id],
        )
        for ordinal, item in enumerate(states)
    ]
    trajectory_rows = [
        _trajectory_row(
            tuple(item for item in states if item.label.source_id == source_id)
        )
        for source_id in dict.fromkeys(item.label.source_id for item in states)
    ]
    state_rows_bytes = _canonical_jsonl_bytes(state_rows)
    trajectory_rows_bytes = _canonical_jsonl_bytes(trajectory_rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": STATUS,
        "contract_sha256": contract_sha256,
        "source_count": SOURCE_COUNT,
        "state_count": STATE_COUNT,
        "budget_event_capacity": DEPLOYMENT_BUDGET,
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "label_tie_epsilon": LABEL_TIE_EPSILON,
        "method_order": list(METHOD_ORDER),
        "aggregation": {
            "unit": "trajectory_equal_then_state_equal_within_trajectory",
            "raw_includes_small_baseline_states": True,
            "normalized_excludes_baseline_at_or_below_epsilon": True,
        },
        "overall": overall,
        "heuristics": {
            "order": list(HEURISTIC_ORDER),
            "mean_normalized_recovery": heuristic_means,
            "strongest": strongest,
        },
        "round_summary": _round_summary(states),
        "strata": {
            "by_n": by_n,
            "baseline_rank_within_n": baseline_strata,
            "interaction_rank_tertile_within_n": interaction_strata,
            "denominator": denominator_strata,
        },
        "decision_tree": _decision_tree(
            states,
            overall,
            by_n,
            baseline_assignments,
            routing_contract,
        ),
        "state_rows": state_rows,
        "state_rows_sha256": hashlib.sha256(state_rows_bytes).hexdigest(),
        "state_rows_size_bytes": len(state_rows_bytes),
        "trajectory_rows": trajectory_rows,
        "trajectory_rows_sha256": hashlib.sha256(trajectory_rows_bytes).hexdigest(),
        "trajectory_rows_size_bytes": len(trajectory_rows_bytes),
        "operation_counts": {
            "scope": "pure_reducer_after_two_sealed_48_record_decodes",
            "file_read": 0,
            "file_write": 0,
            "network": 0,
            "torch_import": 0,
            "gpu": 0,
            "model_load": 0,
            "model_forward": 0,
            "policy_forward": 0,
            "sealed_fresh16_label_state_decode_count": STATE_COUNT,
            "sealed_parent_state_record_decode_count": STATE_COUNT,
            "fresh16_raw_source_access_count": 0,
            "upstream_raw_label_read_count": 0,
            "legacy_dev5_semantic_decode_count": 0,
            "confirm20_access_count": 0,
            "matched_nll_evaluation_count": 0,
            "closed_loop_episode_count": 0,
            "gate_training_step_count": 0,
            "hf_mutation": 0,
        },
    }
    return payload


def failure_decomposition_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize one already reduced payload as canonical newline-terminated JSON."""
    return (
        json.dumps(
            _mapping(payload, "failure-decomposition payload"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


__all__ = [
    "DECISION_THRESHOLDS",
    "PROTOCOL_ID",
    "STATUS",
    "budget_conditioned_oracle_independent_scores",
    "build_failure_decomposition",
    "failure_decomposition_json_bytes",
]
