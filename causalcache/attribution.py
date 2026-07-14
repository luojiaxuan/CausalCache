"""Budget-conditioned restoration estimators."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence


Coalition = frozenset[int]
DistanceFunction = Callable[[Coalition], float]


@dataclass(frozen=True)
class AttributionEstimate:
    event_id: int
    mean: float
    standard_error: float
    samples: tuple[float, ...]


def validate_problem(event_costs: Mapping[int, int], budget: int) -> tuple[int, ...]:
    event_ids = tuple(sorted(event_costs))
    if not event_ids:
        raise ValueError("attribution requires at least one event")
    if event_ids != tuple(range(len(event_ids))):
        raise ValueError("event ids must be contiguous integers starting at zero")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if any(cost <= 0 for cost in event_costs.values()):
        raise ValueError("event costs must be positive")
    if any(cost > budget for cost in event_costs.values()):
        raise ValueError("every attributed event must fit within the requested budget")
    return event_ids


def shared_antithetic_permutations(
    event_ids: Sequence[int],
    *,
    sample_count: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rng = random.Random(seed)
    result: list[tuple[int, ...]] = []
    while len(result) < sample_count:
        permutation = list(event_ids)
        rng.shuffle(permutation)
        result.append(tuple(permutation))
        if len(result) < sample_count:
            result.append(tuple(reversed(permutation)))
    return tuple(result)


def near_budget_coalition(
    permutation: Sequence[int],
    *,
    excluded_event: int,
    event_costs: Mapping[int, int],
    capacity: int,
) -> Coalition:
    if capacity < 0:
        raise ValueError("capacity cannot be negative")
    selected: set[int] = set()
    used = 0
    for event_id in permutation:
        if event_id == excluded_event:
            continue
        event_cost = event_costs[event_id]
        if used + event_cost <= capacity:
            selected.add(event_id)
            used += event_cost
    return frozenset(selected)


def is_maximal_feasible_coalition(
    coalition: Coalition,
    *,
    excluded_event: int,
    event_costs: Mapping[int, int],
    capacity: int,
) -> bool:
    used = sum(event_costs[event_id] for event_id in coalition)
    if used > capacity or excluded_event in coalition:
        return False
    return all(
        event_id == excluded_event
        or event_id in coalition
        or used + event_costs[event_id] > capacity
        for event_id in event_costs
    )


def _standard_error(samples: Sequence[float]) -> float:
    if len(samples) <= 1:
        return 0.0
    mean = sum(samples) / len(samples)
    variance = sum((sample - mean) ** 2 for sample in samples) / (len(samples) - 1)
    return math.sqrt(variance / len(samples))


def estimate_budget_conditioned_restoration(
    distance: DistanceFunction,
    event_costs: Mapping[int, int],
    *,
    budget: int,
    sample_count: int,
    seed: int,
) -> dict[int, AttributionEstimate]:
    event_ids = validate_problem(event_costs, budget)
    permutations = shared_antithetic_permutations(event_ids, sample_count=sample_count, seed=seed)
    distance_cache: dict[Coalition, float] = {}

    def cached_distance(coalition: Coalition) -> float:
        if coalition not in distance_cache:
            value = float(distance(coalition))
            if not math.isfinite(value) or value < 0:
                raise ValueError("distance must be finite and non-negative")
            distance_cache[coalition] = value
        return distance_cache[coalition]

    estimates: dict[int, AttributionEstimate] = {}
    for event_id in event_ids:
        capacity = budget - event_costs[event_id]
        samples: list[float] = []
        for permutation in permutations:
            coalition = near_budget_coalition(
                permutation,
                excluded_event=event_id,
                event_costs=event_costs,
                capacity=capacity,
            )
            restored = coalition | {event_id}
            samples.append(cached_distance(coalition) - cached_distance(restored))
        mean = sum(samples) / len(samples)
        estimates[event_id] = AttributionEstimate(
            event_id=event_id,
            mean=mean,
            standard_error=_standard_error(samples),
            samples=tuple(samples),
        )
    return estimates


def exact_permutation_restoration(
    distance: DistanceFunction,
    event_costs: Mapping[int, int],
    *,
    budget: int,
    max_events: int = 9,
) -> dict[int, float]:
    event_ids = validate_problem(event_costs, budget)
    if len(event_ids) > max_events:
        raise ValueError(f"exact permutation enumeration is limited to {max_events} events")
    totals = {event_id: 0.0 for event_id in event_ids}
    count = 0
    distance_cache: dict[Coalition, float] = {}

    def cached_distance(coalition: Coalition) -> float:
        if coalition not in distance_cache:
            distance_cache[coalition] = float(distance(coalition))
        return distance_cache[coalition]

    for permutation in itertools.permutations(event_ids):
        count += 1
        for event_id in event_ids:
            coalition = near_budget_coalition(
                permutation,
                excluded_event=event_id,
                event_costs=event_costs,
                capacity=budget - event_costs[event_id],
            )
            totals[event_id] += cached_distance(coalition) - cached_distance(coalition | {event_id})
    return {event_id: total / count for event_id, total in totals.items()}


def feasible_coalitions(event_costs: Mapping[int, int], budget: int) -> Iterable[Coalition]:
    event_ids = validate_problem(event_costs, budget)
    for subset_size in range(len(event_ids) + 1):
        for subset in itertools.combinations(event_ids, subset_size):
            if sum(event_costs[event_id] for event_id in subset) <= budget:
                yield frozenset(subset)
