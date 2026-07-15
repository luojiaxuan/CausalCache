"""Deterministic budget-constrained search over a black-box set utility."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping


Coalition = frozenset[int]
SetUtility = Callable[[Coalition], float]


@dataclass(frozen=True)
class SearchStep:
    round_index: int
    coalition: tuple[int, ...]
    cost: int
    utility: float
    selected_event: int | None
    marginal_gain: float | None
    evaluated_candidates: int


@dataclass(frozen=True)
class SearchResult:
    algorithm: str
    coalition: Coalition
    cost: int
    utility: float
    unique_set_evaluations: int
    candidate_score_evaluations: int
    frontier_expansions: int
    sequential_rounds: int
    converged: bool
    trace: tuple[SearchStep, ...]


def validate_search_problem(event_costs: Mapping[int, int], budget: int) -> tuple[int, ...]:
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if not event_costs:
        raise ValueError("search requires at least one event")
    for event_id, cost in event_costs.items():
        if not isinstance(event_id, int) or isinstance(event_id, bool):
            raise ValueError("event ids must be integers")
        if not isinstance(cost, int) or isinstance(cost, bool) or cost <= 0:
            raise ValueError("event costs must be positive integers")
    return tuple(sorted(event_costs))


def coalition_cost(coalition: Coalition, event_costs: Mapping[int, int]) -> int:
    unknown = set(coalition) - set(event_costs)
    if unknown:
        raise ValueError(f"coalition contains unknown events: {sorted(unknown)}")
    return sum(event_costs[event_id] for event_id in coalition)


def feasible_coalitions(event_costs: Mapping[int, int], budget: int) -> Iterable[Coalition]:
    event_ids = validate_search_problem(event_costs, budget)
    for subset_size in range(len(event_ids) + 1):
        for subset in itertools.combinations(event_ids, subset_size):
            coalition = frozenset(subset)
            if coalition_cost(coalition, event_costs) <= budget:
                yield coalition


class MemoizedSetUtility:
    """Validate, memoize, and count unique black-box utility evaluations."""

    def __init__(
        self,
        utility: SetUtility,
        event_costs: Mapping[int, int],
        *,
        budget: int,
    ) -> None:
        self._utility = utility
        self.event_costs = dict(event_costs)
        self.event_ids = validate_search_problem(self.event_costs, budget)
        self.budget = budget
        self._cache: dict[Coalition, float] = {}

    def __call__(self, coalition: Coalition) -> float:
        normalized = frozenset(coalition)
        cost = coalition_cost(normalized, self.event_costs)
        if cost > self.budget:
            raise ValueError("cannot evaluate an over-budget coalition")
        if normalized not in self._cache:
            value = float(self._utility(normalized))
            if not math.isfinite(value):
                raise ValueError("set utility must be finite")
            self._cache[normalized] = value
        return self._cache[normalized]

    @property
    def unique_evaluation_count(self) -> int:
        return len(self._cache)

    @property
    def evaluated_coalitions(self) -> tuple[Coalition, ...]:
        return tuple(sorted(self._cache, key=lambda coalition: (len(coalition), tuple(sorted(coalition)))))


def _coalition_key(
    coalition: Coalition,
    utility: float,
    event_costs: Mapping[int, int],
) -> tuple[float, int, int, tuple[int, ...]]:
    return (-utility, coalition_cost(coalition, event_costs), len(coalition), tuple(sorted(coalition)))


def _better_coalition(
    candidate: Coalition,
    candidate_utility: float,
    incumbent: Coalition,
    incumbent_utility: float,
    event_costs: Mapping[int, int],
    *,
    epsilon: float,
) -> bool:
    if candidate_utility > incumbent_utility + epsilon:
        return True
    if candidate_utility < incumbent_utility - epsilon:
        return False
    return _coalition_key(candidate, candidate_utility, event_costs)[1:] < _coalition_key(
        incumbent, incumbent_utility, event_costs
    )[1:]


def _result(
    *,
    algorithm: str,
    coalition: Coalition,
    oracle: MemoizedSetUtility,
    candidate_score_evaluations: int,
    frontier_expansions: int,
    sequential_rounds: int,
    converged: bool,
    trace: list[SearchStep],
) -> SearchResult:
    return SearchResult(
        algorithm=algorithm,
        coalition=coalition,
        cost=coalition_cost(coalition, oracle.event_costs),
        utility=oracle(coalition),
        unique_set_evaluations=oracle.unique_evaluation_count,
        candidate_score_evaluations=candidate_score_evaluations,
        frontier_expansions=frontier_expansions,
        sequential_rounds=sequential_rounds,
        converged=converged,
        trace=tuple(trace),
    )


def exact_subset_search(
    utility: SetUtility,
    event_costs: Mapping[int, int],
    *,
    budget: int,
    epsilon: float = 1e-12,
    max_feasible_coalitions: int | None = None,
) -> SearchResult:
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and non-negative")
    oracle = MemoizedSetUtility(utility, event_costs, budget=budget)
    coalitions = tuple(feasible_coalitions(event_costs, budget))
    if max_feasible_coalitions is not None and len(coalitions) > max_feasible_coalitions:
        raise ValueError("feasible coalition count exceeds the frozen exact-search limit")
    incumbent = frozenset()
    incumbent_utility = oracle(incumbent)
    for coalition in coalitions[1:]:
        value = oracle(coalition)
        if _better_coalition(
            coalition,
            value,
            incumbent,
            incumbent_utility,
            oracle.event_costs,
            epsilon=epsilon,
        ):
            incumbent = coalition
            incumbent_utility = value
    return _result(
        algorithm="exact_subset",
        coalition=incumbent,
        oracle=oracle,
        candidate_score_evaluations=len(coalitions),
        frontier_expansions=0,
        sequential_rounds=1,
        converged=True,
        trace=[],
    )


def _candidate_is_better(
    candidate: tuple[int, float, float],
    incumbent: tuple[int, float, float] | None,
    event_costs: Mapping[int, int],
    *,
    epsilon: float,
) -> bool:
    if incumbent is None:
        return True
    event_id, gain, score = candidate
    incumbent_id, incumbent_gain, incumbent_score = incumbent
    if score > incumbent_score + epsilon:
        return True
    if score < incumbent_score - epsilon:
        return False
    if gain > incumbent_gain + epsilon:
        return True
    if gain < incumbent_gain - epsilon:
        return False
    return (event_costs[event_id], event_id) < (event_costs[incumbent_id], incumbent_id)


def _conditional_greedy_with_oracle(
    oracle: MemoizedSetUtility,
    *,
    score_rule: str,
    stopping_threshold: float,
    epsilon: float,
) -> tuple[Coalition, int, int, list[SearchStep]]:
    if score_rule not in {"raw_gain", "gain_per_cost"}:
        raise ValueError("score_rule must be raw_gain or gain_per_cost")
    if not math.isfinite(stopping_threshold):
        raise ValueError("stopping_threshold must be finite")
    selected: Coalition = frozenset()
    current_utility = oracle(selected)
    candidate_evaluations = 0
    rounds = 0
    trace: list[SearchStep] = []
    while True:
        feasible = [
            event_id
            for event_id in oracle.event_ids
            if event_id not in selected
            and coalition_cost(selected | {event_id}, oracle.event_costs) <= oracle.budget
        ]
        if not feasible:
            break
        rounds += 1
        best: tuple[int, float, float] | None = None
        for event_id in feasible:
            candidate = selected | {event_id}
            gain = oracle(candidate) - current_utility
            score = gain if score_rule == "raw_gain" else gain / oracle.event_costs[event_id]
            evaluated = (event_id, gain, score)
            candidate_evaluations += 1
            if _candidate_is_better(evaluated, best, oracle.event_costs, epsilon=epsilon):
                best = evaluated
        assert best is not None
        event_id, gain, _ = best
        if gain <= stopping_threshold + epsilon:
            break
        selected = selected | {event_id}
        current_utility = oracle(selected)
        trace.append(
            SearchStep(
                round_index=rounds,
                coalition=tuple(sorted(selected)),
                cost=coalition_cost(selected, oracle.event_costs),
                utility=current_utility,
                selected_event=event_id,
                marginal_gain=gain,
                evaluated_candidates=len(feasible),
            )
        )
    return selected, candidate_evaluations, rounds, trace


def conditional_marginal_greedy(
    utility: SetUtility,
    event_costs: Mapping[int, int],
    *,
    budget: int,
    score_rule: str = "raw_gain",
    stopping_threshold: float = 0.0,
    epsilon: float = 1e-12,
) -> SearchResult:
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and non-negative")
    oracle = MemoizedSetUtility(utility, event_costs, budget=budget)
    selected, evaluations, rounds, trace = _conditional_greedy_with_oracle(
        oracle,
        score_rule=score_rule,
        stopping_threshold=stopping_threshold,
        epsilon=epsilon,
    )
    return _result(
        algorithm=f"true_conditional_greedy_{score_rule}",
        coalition=selected,
        oracle=oracle,
        candidate_score_evaluations=evaluations,
        frontier_expansions=0,
        sequential_rounds=rounds,
        converged=True,
        trace=trace,
    )


def _exchange_neighbors(
    coalition: Coalition,
    event_ids: tuple[int, ...],
    event_costs: Mapping[int, int],
    *,
    budget: int,
    max_remove: int,
    max_add: int,
) -> tuple[Coalition, ...]:
    outside = tuple(event_id for event_id in event_ids if event_id not in coalition)
    neighbors: set[Coalition] = set()
    for remove_count in range(min(max_remove, len(coalition)) + 1):
        for removed in itertools.combinations(sorted(coalition), remove_count):
            base = coalition - frozenset(removed)
            for add_count in range(min(max_add, len(outside)) + 1):
                for added in itertools.combinations(outside, add_count):
                    candidate = base | frozenset(added)
                    if candidate == coalition:
                        continue
                    if coalition_cost(candidate, event_costs) <= budget:
                        neighbors.add(candidate)
    return tuple(sorted(neighbors, key=lambda item: (len(item), tuple(sorted(item)))))


def greedy_with_bounded_exchange(
    utility: SetUtility,
    event_costs: Mapping[int, int],
    *,
    budget: int,
    score_rule: str = "raw_gain",
    stopping_threshold: float = 0.0,
    epsilon: float = 1e-12,
    max_remove: int = 2,
    max_add: int = 2,
    max_passes: int = 8,
) -> SearchResult:
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and non-negative")
    if max_remove <= 0 or max_add <= 0 or max_passes <= 0:
        raise ValueError("exchange bounds and max_passes must be positive")
    oracle = MemoizedSetUtility(utility, event_costs, budget=budget)
    selected, candidate_evaluations, greedy_rounds, trace = _conditional_greedy_with_oracle(
        oracle,
        score_rule=score_rule,
        stopping_threshold=stopping_threshold,
        epsilon=epsilon,
    )
    current_utility = oracle(selected)
    passes = 0
    converged = False
    while passes < max_passes:
        passes += 1
        neighbors = _exchange_neighbors(
            selected,
            oracle.event_ids,
            oracle.event_costs,
            budget=budget,
            max_remove=max_remove,
            max_add=max_add,
        )
        candidate_evaluations += len(neighbors)
        best = selected
        best_utility = current_utility
        for candidate in neighbors:
            value = oracle(candidate)
            if _better_coalition(
                candidate,
                value,
                best,
                best_utility,
                oracle.event_costs,
                epsilon=epsilon,
            ):
                best = candidate
                best_utility = value
        if best == selected:
            converged = True
            break
        selected = best
        current_utility = best_utility
        trace.append(
            SearchStep(
                round_index=greedy_rounds + passes,
                coalition=tuple(sorted(selected)),
                cost=coalition_cost(selected, oracle.event_costs),
                utility=current_utility,
                selected_event=None,
                marginal_gain=None,
                evaluated_candidates=len(neighbors),
            )
        )
    return _result(
        algorithm=f"true_conditional_greedy_{score_rule}_exchange_{max_remove}x{max_add}",
        coalition=selected,
        oracle=oracle,
        candidate_score_evaluations=candidate_evaluations,
        frontier_expansions=passes,
        sequential_rounds=greedy_rounds + passes,
        converged=converged,
        trace=trace,
    )


def beam_subset_search(
    utility: SetUtility,
    event_costs: Mapping[int, int],
    *,
    budget: int,
    width: int,
    epsilon: float = 1e-12,
    allow_nonpositive_prefixes: bool = True,
) -> SearchResult:
    if width <= 0:
        raise ValueError("beam width must be positive")
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and non-negative")
    oracle = MemoizedSetUtility(utility, event_costs, budget=budget)
    empty = frozenset()
    empty_utility = oracle(empty)
    frontier: tuple[Coalition, ...] = (empty,)
    incumbent = empty
    incumbent_utility = empty_utility
    candidate_evaluations = 1
    expansions = 0
    rounds = 0
    trace: list[SearchStep] = []
    while frontier:
        children: set[Coalition] = set()
        for coalition in frontier:
            for event_id in oracle.event_ids:
                if event_id in coalition:
                    continue
                child = coalition | {event_id}
                if coalition_cost(child, oracle.event_costs) <= budget:
                    children.add(child)
        if not children:
            break
        rounds += 1
        expansions += len(frontier)
        scored: list[tuple[Coalition, float]] = []
        for coalition in sorted(children, key=lambda item: (len(item), tuple(sorted(item)))):
            value = oracle(coalition)
            candidate_evaluations += 1
            if _better_coalition(
                coalition,
                value,
                incumbent,
                incumbent_utility,
                oracle.event_costs,
                epsilon=epsilon,
            ):
                incumbent = coalition
                incumbent_utility = value
            if allow_nonpositive_prefixes or value > empty_utility + epsilon:
                scored.append((coalition, value))
        scored.sort(key=lambda item: _coalition_key(item[0], item[1], oracle.event_costs))
        frontier = tuple(coalition for coalition, _ in scored[:width])
        trace.append(
            SearchStep(
                round_index=rounds,
                coalition=tuple(sorted(incumbent)),
                cost=coalition_cost(incumbent, oracle.event_costs),
                utility=incumbent_utility,
                selected_event=None,
                marginal_gain=None,
                evaluated_candidates=len(children),
            )
        )
    return _result(
        algorithm=f"true_utility_beam_{width}",
        coalition=incumbent,
        oracle=oracle,
        candidate_score_evaluations=candidate_evaluations,
        frontier_expansions=expansions,
        sequential_rounds=rounds,
        converged=True,
        trace=trace,
    )


def second_order_interaction(
    utility: SetUtility,
    coalition: Coalition,
    left: int,
    right: int,
) -> float:
    if left == right or left in coalition or right in coalition:
        raise ValueError("interaction events must be distinct and outside the conditioning coalition")
    base = float(utility(coalition))
    left_value = float(utility(coalition | {left}))
    right_value = float(utility(coalition | {right}))
    pair_value = float(utility(coalition | {left, right}))
    values = (base, left_value, right_value, pair_value)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("interaction utilities must be finite")
    return pair_value - left_value - right_value + base
