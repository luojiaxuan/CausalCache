"""Policy-free selector geometry over validated restoration-v2.2 distance tables."""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.attribution import exact_permutation_restoration
from causalcache.restoration_v2_2_eager_artifact import strict_json_object_bytes
from causalcache.restoration_v2_2_label_artifact import (
    LabelEvidence,
    read_label_evidence_archive,
)
from causalcache.restoration_v2_2_label_table import (
    ValidatedDistanceTable,
    deployment_conditional_edges,
    exact_permutation_average_attribution,
    pair_interactions,
    validate_complete_distance_table,
)
from causalcache.subset_search import (
    SearchResult,
    SearchStep,
    conditional_marginal_greedy,
    exact_subset_search,
)


SCHEMA_VERSION = "1.0.0"
NORMALIZATION_EPSILON = 1e-12
_STATE_MEMBER = re.compile(r"workers/(?:even|odd)/states/[0-9]{3}\.json")


@dataclass(frozen=True)
class SelectorGeometryState:
    """One validated state and its complete canonical distance table."""

    member_name: str
    index: int
    role: str
    trajectory_id: str
    state_id: str
    decision_step_id: int
    candidate_event_step_ids: tuple[int, ...]
    table: ValidatedDistanceTable


def _state_from_record(member_name: str, record: Mapping[str, Any]) -> SelectorGeometryState:
    state = record.get("state")
    rows = record.get("distance_rows")
    if not isinstance(state, Mapping) or not isinstance(rows, list):
        raise ValueError(f"{member_name} lacks state metadata or distance rows")
    required = {
        "index",
        "role",
        "trajectory_id",
        "state_id",
        "decision_step_id",
        "candidate_event_step_ids",
    }
    if not required.issubset(state):
        raise ValueError(f"{member_name} state metadata is incomplete")
    index = state["index"]
    decision_step_id = state["decision_step_id"]
    event_ids = state["candidate_event_step_ids"]
    if type(index) is not int or index < 0:
        raise ValueError(f"{member_name} has an invalid state index")
    if type(decision_step_id) is not int or decision_step_id <= 0:
        raise ValueError(f"{member_name} has an invalid decision step")
    if not isinstance(event_ids, list):
        raise ValueError(f"{member_name} has invalid candidate event ids")
    string_fields = {
        field: state[field]
        for field in ("role", "trajectory_id", "state_id")
    }
    if any(not isinstance(value, str) or not value for value in string_fields.values()):
        raise ValueError(f"{member_name} has invalid string state metadata")

    distances: dict[tuple[int, ...], Any] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{member_name} contains a non-object distance row")
        coalition = row.get("coalition_event_step_ids")
        if not isinstance(coalition, list):
            raise ValueError(f"{member_name} contains invalid coalition ids")
        key = tuple(coalition)
        if key in distances:
            raise ValueError(f"{member_name} contains duplicate coalition rows")
        distances[key] = row.get("distance_kl")
    table = validate_complete_distance_table(tuple(event_ids), distances)
    if table.event_ids != tuple(event_ids):
        raise RuntimeError("validated event identity changed during table construction")
    return SelectorGeometryState(
        member_name=member_name,
        index=index,
        role=string_fields["role"],
        trajectory_id=string_fields["trajectory_id"],
        state_id=string_fields["state_id"],
        decision_step_id=decision_step_id,
        candidate_event_step_ids=table.event_ids,
        table=table,
    )


def selector_geometry_states(evidence: LabelEvidence) -> tuple[SelectorGeometryState, ...]:
    """Project validated evidence into index-ordered state distance tables."""
    if not isinstance(evidence, LabelEvidence):
        raise TypeError("evidence must be validated LabelEvidence")
    names = tuple(
        sorted(name for name in evidence.files if _STATE_MEMBER.fullmatch(name))
    )
    if not names:
        raise ValueError("validated label evidence contains no state records")
    states = tuple(
        _state_from_record(
            name,
            strict_json_object_bytes(evidence.files[name], label=name),
        )
        for name in names
    )
    indices = [state.index for state in states]
    state_ids = [state.state_id for state in states]
    if len(indices) != len(set(indices)) or len(state_ids) != len(set(state_ids)):
        raise ValueError("selector geometry states must have unique indices and ids")
    ordered = tuple(sorted(states, key=lambda state: state.index))
    if [state.index for state in ordered] != list(range(len(ordered))):
        raise ValueError("selector geometry state indices must be contiguous from zero")
    return ordered


def load_selector_geometry_states(
    archive_path: str | Path,
) -> tuple[SelectorGeometryState, ...]:
    """Validate a canonical raw archive before exposing any scientific table."""
    return selector_geometry_states(read_label_evidence_archive(archive_path))


def _validate_budget(state: SelectorGeometryState, budget: int) -> None:
    if not isinstance(state, SelectorGeometryState):
        raise TypeError("state must be a SelectorGeometryState")
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a non-negative integer")
    if budget > len(state.candidate_event_step_ids):
        raise ValueError("budget cannot exceed the state's candidate event count")


def _unit_costs(state: SelectorGeometryState) -> dict[int, int]:
    return {event_id: 1 for event_id in state.candidate_event_step_ids}


def _empty_search_result(algorithm: str) -> SearchResult:
    return SearchResult(
        algorithm=algorithm,
        coalition=frozenset(),
        cost=0,
        utility=0.0,
        unique_set_evaluations=1,
        candidate_score_evaluations=1,
        frontier_expansions=0,
        sequential_rounds=0,
        converged=True,
        trace=(),
    )


def exact_selector(state: SelectorGeometryState, *, budget: int) -> SearchResult:
    """Return the exact at-most-B subset under strict, zero-epsilon ties."""
    _validate_budget(state, budget)
    if budget == 0:
        return _empty_search_result("exact_subset")
    return exact_subset_search(
        lambda coalition: state.table.utility(coalition),
        _unit_costs(state),
        budget=budget,
        epsilon=0.0,
        max_feasible_coalitions=2 ** len(state.candidate_event_step_ids),
    )


def true_conditional_greedy_selector(
    state: SelectorGeometryState,
    *,
    budget: int,
) -> SearchResult:
    """Recompute true conditional gains and stop before every nonpositive gain."""
    _validate_budget(state, budget)
    if budget == 0:
        return _empty_search_result("true_conditional_greedy_raw_gain")
    return conditional_marginal_greedy(
        lambda coalition: state.table.utility(coalition),
        _unit_costs(state),
        budget=budget,
        score_rule="raw_gain",
        stopping_threshold=0.0,
        epsilon=0.0,
    )


def exact_cardinality_oracle(
    state: SelectorGeometryState,
    *,
    budget: int,
) -> SearchResult:
    """Return the exact best coalition whose cardinality is exactly B."""
    _validate_budget(state, budget)
    feasible = tuple(
        row for row in state.table.rows if len(row.coalition) == budget
    )
    if not feasible:
        raise RuntimeError("validated power set lacks an exact-cardinality coalition")
    selected = min(feasible, key=lambda row: (row.distance, row.coalition))
    return SearchResult(
        algorithm="exact_cardinality_oracle",
        coalition=frozenset(selected.coalition),
        cost=budget,
        utility=selected.utility,
        unique_set_evaluations=len(feasible),
        candidate_score_evaluations=len(feasible),
        frontier_expansions=0,
        sequential_rounds=1 if budget > 0 else 0,
        converged=True,
        trace=(),
    )


def forced_fill_true_conditional_greedy_selector(
    state: SelectorGeometryState,
    *,
    budget: int,
) -> SearchResult:
    """Greedily rescore true marginals but fill B slots even when gains are nonpositive."""
    _validate_budget(state, budget)
    if budget == 0:
        return _empty_search_result("forced_fill_true_conditional_greedy")
    selected = frozenset()
    current_utility = state.table.utility(selected)
    evaluated: set[frozenset[int]] = {selected}
    candidate_evaluations = 0
    trace: list[SearchStep] = []
    for round_index in range(1, budget + 1):
        best_event: int | None = None
        best_gain: float | None = None
        best_utility: float | None = None
        candidates = tuple(
            event_id
            for event_id in state.candidate_event_step_ids
            if event_id not in selected
        )
        for event_id in candidates:
            candidate = selected | {event_id}
            candidate_utility = state.table.utility(candidate)
            evaluated.add(candidate)
            candidate_evaluations += 1
            gain = candidate_utility - current_utility
            if (
                best_event is None
                or gain > best_gain
                or (gain == best_gain and event_id < best_event)
            ):
                best_event = event_id
                best_gain = gain
                best_utility = candidate_utility
        if best_event is None or best_gain is None or best_utility is None:
            raise RuntimeError("forced-fill greedy exhausted candidates before its budget")
        selected = selected | {best_event}
        current_utility = best_utility
        trace.append(
            SearchStep(
                round_index=round_index,
                coalition=tuple(sorted(selected)),
                cost=len(selected),
                utility=current_utility,
                selected_event=best_event,
                marginal_gain=best_gain,
                evaluated_candidates=len(candidates),
            )
        )
    return SearchResult(
        algorithm="forced_fill_true_conditional_greedy",
        coalition=selected,
        cost=len(selected),
        utility=current_utility,
        unique_set_evaluations=len(evaluated),
        candidate_score_evaluations=candidate_evaluations,
        frontier_expansions=0,
        sequential_rounds=budget,
        converged=True,
        trace=tuple(trace),
    )


def _select_strict_positive_scores(
    scores: Mapping[int, float],
    *,
    event_ids: Sequence[int],
    budget: int,
) -> tuple[int, ...]:
    if set(scores) != set(event_ids):
        raise ValueError("static selector scores must cover every candidate event")
    converted = {event_id: float(scores[event_id]) for event_id in event_ids}
    if any(not math.isfinite(value) for value in converted.values()):
        raise ValueError("static selector scores must be finite")
    ranked = tuple(
        sorted(
            (event_id for event_id in event_ids if converted[event_id] > 0.0),
            key=lambda event_id: (-converted[event_id], event_id),
        )
    )
    return tuple(sorted(ranked[:budget]))


def _select_forced_fill_scores(
    scores: Mapping[int, float],
    *,
    event_ids: Sequence[int],
    budget: int,
) -> tuple[int, ...]:
    if set(scores) != set(event_ids):
        raise ValueError("static selector scores must cover every candidate event")
    converted = {event_id: float(scores[event_id]) for event_id in event_ids}
    if any(not math.isfinite(value) for value in converted.values()):
        raise ValueError("static selector scores must be finite")
    ranked = tuple(
        sorted(
            event_ids,
            key=lambda event_id: (-converted[event_id], event_id),
        )
    )
    return tuple(sorted(ranked[:budget]))


def _budget_conditioned_scores(
    state: SelectorGeometryState,
    *,
    budget: int,
) -> dict[int, float]:
    if budget == 0:
        return {event_id: 0.0 for event_id in state.candidate_event_step_ids}
    original_ids = state.candidate_event_step_ids
    index_to_event = dict(enumerate(original_ids))
    event_to_index = {event_id: index for index, event_id in index_to_event.items()}

    def indexed_distance(coalition: frozenset[int]) -> float:
        original = frozenset(index_to_event[index] for index in coalition)
        return state.table.distance(original)

    indexed_scores = exact_permutation_restoration(
        indexed_distance,
        {index: 1 for index in index_to_event},
        budget=budget,
        max_events=len(original_ids),
    )
    return {
        event_id: indexed_scores[event_to_index[event_id]]
        for event_id in original_ids
    }


def budget_conditioned_independent_selector(
    state: SelectorGeometryState,
    *,
    budget: int,
) -> tuple[tuple[int, ...], dict[int, float]]:
    """Select by exact near-budget average marginals without set conditioning."""
    _validate_budget(state, budget)
    original_ids = state.candidate_event_step_ids
    scores = _budget_conditioned_scores(state, budget=budget)
    selected = _select_strict_positive_scores(
        scores,
        event_ids=original_ids,
        budget=budget,
    )
    return selected, scores


def forced_fill_budget_conditioned_independent_selector(
    state: SelectorGeometryState,
    *,
    budget: int,
) -> tuple[tuple[int, ...], dict[int, float]]:
    """Select exactly B events by frozen budget-conditioned independent scores."""
    _validate_budget(state, budget)
    scores = _budget_conditioned_scores(state, budget=budget)
    selected = _select_forced_fill_scores(
        scores,
        event_ids=state.candidate_event_step_ids,
        budget=budget,
    )
    return selected, scores


def _full_shapley_scores(state: SelectorGeometryState) -> dict[int, float]:
    return {
        item.event_id: item.mean_marginal_gain
        for item in exact_permutation_average_attribution(state.table)
    }


def full_shapley_independent_selector(
    state: SelectorGeometryState,
    *,
    budget: int,
) -> tuple[tuple[int, ...], dict[int, float]]:
    """Select by full-hypercube permutation-average event values."""
    _validate_budget(state, budget)
    scores = _full_shapley_scores(state)
    selected = _select_strict_positive_scores(
        scores,
        event_ids=state.candidate_event_step_ids,
        budget=budget,
    )
    return selected, scores


def forced_fill_full_shapley_independent_selector(
    state: SelectorGeometryState,
    *,
    budget: int,
) -> tuple[tuple[int, ...], dict[int, float]]:
    """Select exactly B events by frozen full-Shapley independent scores."""
    _validate_budget(state, budget)
    scores = _full_shapley_scores(state)
    selected = _select_forced_fill_scores(
        scores,
        event_ids=state.candidate_event_step_ids,
        budget=budget,
    )
    return selected, scores


def dynamic_recent_selector(
    state: SelectorGeometryState,
    *,
    budget: int,
) -> tuple[int, ...]:
    """Select exactly the B chronologically most recent candidate events."""
    _validate_budget(state, budget)
    if budget == 0:
        return ()
    return state.candidate_event_step_ids[-budget:]


def analytic_exact_cardinality_random(
    state: SelectorGeometryState,
    *,
    budget: int,
) -> dict[str, Any]:
    """Return the analytic expectation over all exact-B coalitions."""
    _validate_budget(state, budget)
    coalitions = tuple(
        itertools.combinations(state.candidate_event_step_ids, budget)
    )
    utilities = tuple(state.table.utility(coalition) for coalition in coalitions)
    mean_utility = math.fsum(utilities) / len(utilities)
    return {
        "selection_type": "analytic_exact_cardinality_expectation",
        "selected_coalition": None,
        "exact_cardinality": budget,
        "coalition_count": len(coalitions),
        "mean_utility": mean_utility,
        "utility_by_coalition": [
            {"coalition": list(coalition), "utility": utility}
            for coalition, utility in zip(coalitions, utilities, strict=True)
        ],
    }


def _normalized_recovery(utility: float, baseline_distance: float) -> float | None:
    if baseline_distance <= NORMALIZATION_EPSILON:
        return None
    return utility / baseline_distance


def _ratio_to_exact(utility: float, exact_utility: float) -> float | None:
    if exact_utility <= NORMALIZATION_EPSILON:
        return None
    return utility / exact_utility


def _score_rows(scores: Mapping[int, float] | None) -> list[dict[str, Any]] | None:
    if scores is None:
        return None
    return [
        {"event_step_id": event_id, "score": float(scores[event_id])}
        for event_id in sorted(scores)
    ]


def _method_record(
    *,
    method: str,
    coalition: Sequence[int] | None,
    utility: float,
    baseline_distance: float,
    exact_utility: float,
    scores: Mapping[int, float] | None = None,
    selection_type: str = "deterministic_subset",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = _normalized_recovery(utility, baseline_distance)
    exact_normalized = _normalized_recovery(exact_utility, baseline_distance)
    result: dict[str, Any] = {
        "method": method,
        "selection_type": selection_type,
        "selected_coalition": None if coalition is None else list(coalition),
        "utility": utility,
        "normalized_recovery": normalized,
        "absolute_regret_to_exact_subset": exact_utility - utility,
        "normalized_recovery_regret_to_exact_subset": (
            None
            if normalized is None or exact_normalized is None
            else exact_normalized - normalized
        ),
        "utility_ratio_to_exact_subset": _ratio_to_exact(utility, exact_utility),
        "scores": _score_rows(scores),
    }
    if extra is not None:
        result.update(extra)
    return result


def _paired_gap(
    left_utility: float,
    right_utility: float,
    *,
    baseline_distance: float,
) -> dict[str, float | None]:
    left_normalized = _normalized_recovery(left_utility, baseline_distance)
    right_normalized = _normalized_recovery(right_utility, baseline_distance)
    return {
        "raw_utility": left_utility - right_utility,
        "normalized_recovery": (
            None
            if left_normalized is None or right_normalized is None
            else left_normalized - right_normalized
        ),
    }


def state_interaction_metrics(state: SelectorGeometryState) -> dict[str, Any]:
    """Return state-normalized interaction and two negative-marginal sign audits."""
    if not isinstance(state, SelectorGeometryState):
        raise TypeError("state must be a SelectorGeometryState")
    interaction_values = tuple(
        item.interaction for item in pair_interactions(state.table)
    )
    marginal_values = tuple(
        edge.marginal_gain for edge in deployment_conditional_edges(state.table)
    )
    absolute_mass = math.fsum(abs(value) for value in interaction_values)
    complementarity_mass = math.fsum(
        value for value in interaction_values if value > 0.0
    )
    redundancy_mass = math.fsum(
        -value for value in interaction_values if value < 0.0
    )
    strict_negative_count = sum(value < 0.0 for value in marginal_values)
    epsilon_negative_count = sum(value < -NORMALIZATION_EPSILON for value in marginal_values)
    marginal_count = len(marginal_values)
    baseline = state.table.distance(())
    return {
        "interaction_count": len(interaction_values),
        "positive_interaction_count": sum(
            value > 0.0 for value in interaction_values
        ),
        "negative_interaction_count": sum(
            value < 0.0 for value in interaction_values
        ),
        "zero_interaction_count": sum(
            value == 0.0 for value in interaction_values
        ),
        "complementarity_mass": complementarity_mass,
        "redundancy_mass": redundancy_mass,
        "absolute_interaction_mass": absolute_mass,
        "mean_absolute_interaction": (
            math.fsum(abs(value) for value in interaction_values)
            / len(interaction_values)
            if interaction_values
            else 0.0
        ),
        "normalized_mean_absolute_interaction": (
            math.fsum(abs(value) for value in interaction_values)
            / len(interaction_values)
            / max(baseline, NORMALIZATION_EPSILON)
            if interaction_values
            else 0.0
        ),
        "redundancy_share": (
            redundancy_mass / absolute_mass
            if absolute_mass > 0.0
            else None
        ),
        "deployment_marginal_count": marginal_count,
        "strict_negative_marginal_count": strict_negative_count,
        "strict_negative_marginal_rate": (
            strict_negative_count / marginal_count if marginal_count else 0.0
        ),
        "has_strict_negative_marginal": strict_negative_count > 0,
        "epsilon_1e12_negative_marginal_count": epsilon_negative_count,
        "epsilon_1e12_negative_marginal_rate": (
            epsilon_negative_count / marginal_count if marginal_count else 0.0
        ),
        "has_epsilon_1e12_negative_marginal": epsilon_negative_count > 0,
    }


def _search_extra(result: SearchResult) -> dict[str, Any]:
    return {
        "unique_set_utility_evaluations": result.unique_set_evaluations,
        "candidate_score_evaluations": result.candidate_score_evaluations,
        "sequential_rounds": result.sequential_rounds,
        "trace": [
            {
                "round_index": step.round_index,
                "coalition": list(step.coalition),
                "utility": step.utility,
                "selected_event": step.selected_event,
                "marginal_gain": step.marginal_gain,
                "evaluated_candidates": step.evaluated_candidates,
            }
            for step in result.trace
        ],
    }


def evaluate_selector_geometry_state(
    state: SelectorGeometryState,
    *,
    budget: int,
) -> dict[str, Any]:
    """Evaluate all policy-free selectors for one state and one feasible budget."""
    _validate_budget(state, budget)
    baseline = state.table.distance(())
    exact = exact_selector(state, budget=budget)
    exact_cardinality = exact_cardinality_oracle(state, budget=budget)
    greedy = true_conditional_greedy_selector(state, budget=budget)
    forced_greedy = forced_fill_true_conditional_greedy_selector(
        state,
        budget=budget,
    )
    budget_independent, budget_scores = budget_conditioned_independent_selector(
        state,
        budget=budget,
    )
    forced_budget_independent, _ = (
        forced_fill_budget_conditioned_independent_selector(
            state,
            budget=budget,
        )
    )
    shapley_independent, shapley_scores = full_shapley_independent_selector(
        state,
        budget=budget,
    )
    forced_shapley_independent, _ = forced_fill_full_shapley_independent_selector(
        state,
        budget=budget,
    )
    recent = dynamic_recent_selector(state, budget=budget)
    random = analytic_exact_cardinality_random(state, budget=budget)
    exact_utility = exact.utility
    exact_cardinality_utility = exact_cardinality.utility
    budget_independent_utility = state.table.utility(budget_independent)
    forced_budget_independent_utility = state.table.utility(
        forced_budget_independent
    )
    shapley_independent_utility = state.table.utility(shapley_independent)
    forced_shapley_independent_utility = state.table.utility(
        forced_shapley_independent
    )
    recent_utility = state.table.utility(recent)
    random_utility = float(random["mean_utility"])
    methods = {
        "exact_subset": _method_record(
            method="exact_subset",
            coalition=tuple(sorted(exact.coalition)),
            utility=exact.utility,
            baseline_distance=baseline,
            exact_utility=exact_utility,
            extra=_search_extra(exact),
        ),
        "exact_cardinality_oracle": _method_record(
            method="exact_cardinality_oracle",
            coalition=tuple(sorted(exact_cardinality.coalition)),
            utility=exact_cardinality.utility,
            baseline_distance=baseline,
            exact_utility=exact_utility,
            extra=_search_extra(exact_cardinality),
        ),
        "true_conditional_greedy": _method_record(
            method="true_conditional_greedy",
            coalition=tuple(sorted(greedy.coalition)),
            utility=greedy.utility,
            baseline_distance=baseline,
            exact_utility=exact_utility,
            extra=_search_extra(greedy),
        ),
        "forced_fill_true_conditional_greedy": _method_record(
            method="forced_fill_true_conditional_greedy",
            coalition=tuple(sorted(forced_greedy.coalition)),
            utility=forced_greedy.utility,
            baseline_distance=baseline,
            exact_utility=exact_utility,
            extra=_search_extra(forced_greedy),
        ),
        "budget_conditioned_independent": _method_record(
            method="budget_conditioned_independent",
            coalition=budget_independent,
            utility=budget_independent_utility,
            baseline_distance=baseline,
            exact_utility=exact_utility,
            scores=budget_scores,
        ),
        "forced_fill_budget_conditioned_independent": _method_record(
            method="forced_fill_budget_conditioned_independent",
            coalition=forced_budget_independent,
            utility=forced_budget_independent_utility,
            baseline_distance=baseline,
            exact_utility=exact_utility,
            scores=budget_scores,
        ),
        "full_shapley_independent": _method_record(
            method="full_shapley_independent",
            coalition=shapley_independent,
            utility=shapley_independent_utility,
            baseline_distance=baseline,
            exact_utility=exact_utility,
            scores=shapley_scores,
        ),
        "forced_fill_full_shapley_independent": _method_record(
            method="forced_fill_full_shapley_independent",
            coalition=forced_shapley_independent,
            utility=forced_shapley_independent_utility,
            baseline_distance=baseline,
            exact_utility=exact_utility,
            scores=shapley_scores,
        ),
        "dynamic_recent": _method_record(
            method="dynamic_recent",
            coalition=recent,
            utility=recent_utility,
            baseline_distance=baseline,
            exact_utility=exact_utility,
        ),
        "analytic_exact_cardinality_random": _method_record(
            method="analytic_exact_cardinality_random",
            coalition=None,
            utility=random_utility,
            baseline_distance=baseline,
            exact_utility=exact_utility,
            selection_type=str(random["selection_type"]),
            extra={
                "exact_cardinality": random["exact_cardinality"],
                "coalition_count": random["coalition_count"],
                "utility_by_coalition": random["utility_by_coalition"],
            },
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "state": {
            "member_name": state.member_name,
            "index": state.index,
            "role": state.role,
            "trajectory_id": state.trajectory_id,
            "state_id": state.state_id,
            "decision_step_id": state.decision_step_id,
            "candidate_event_step_ids": list(state.candidate_event_step_ids),
            "candidate_event_count": len(state.candidate_event_step_ids),
        },
        "budget_event_capacity": budget,
        "baseline_summary_only_distance": baseline,
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "interaction_metrics": state_interaction_metrics(state),
        "methods": methods,
        "gaps": {
            "search_gap": _paired_gap(
                exact.utility,
                greedy.utility,
                baseline_distance=baseline,
            ),
            "objective_projection_gap": _paired_gap(
                greedy.utility,
                budget_independent_utility,
                baseline_distance=baseline,
            ),
            "total_gap": _paired_gap(
                exact.utility,
                budget_independent_utility,
                baseline_distance=baseline,
            ),
            "full_shapley_projection_gap": _paired_gap(
                greedy.utility,
                shapley_independent_utility,
                baseline_distance=baseline,
            ),
            "exact_cardinality_forced_fill_gap": _paired_gap(
                exact.utility,
                exact_cardinality_utility,
                baseline_distance=baseline,
            ),
            "true_greedy_forced_fill_gap": _paired_gap(
                greedy.utility,
                forced_greedy.utility,
                baseline_distance=baseline,
            ),
            "budget_independent_forced_fill_gap": _paired_gap(
                budget_independent_utility,
                forced_budget_independent_utility,
                baseline_distance=baseline,
            ),
            "full_shapley_forced_fill_gap": _paired_gap(
                shapley_independent_utility,
                forced_shapley_independent_utility,
                baseline_distance=baseline,
            ),
        },
    }


def evaluate_all_feasible_budgets(
    state: SelectorGeometryState,
) -> tuple[dict[str, Any], ...]:
    """Return one policy-free geometry record for every budget from zero through n."""
    if not isinstance(state, SelectorGeometryState):
        raise TypeError("state must be a SelectorGeometryState")
    return tuple(
        evaluate_selector_geometry_state(state, budget=budget)
        for budget in range(len(state.candidate_event_step_ids) + 1)
    )
