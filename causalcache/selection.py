"""Budget-aware selection and ranking metrics."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from causalcache.attribution import Coalition, DistanceFunction, feasible_coalitions, validate_problem


def select_positive_value_knapsack(
    scores: Mapping[int, float],
    event_costs: Mapping[int, int],
    *,
    budget: int,
    threshold: float = 0.0,
) -> Coalition:
    event_ids = validate_problem(event_costs, budget)
    if set(scores) != set(event_ids):
        raise ValueError("scores and event_costs must have identical event ids")
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for event_id in event_ids:
        adjusted_value = float(scores[event_id]) - threshold
        if adjusted_value <= 0:
            continue
        event_cost = event_costs[event_id]
        updated = dict(states)
        for used_cost, (value, selected) in states.items():
            next_cost = used_cost + event_cost
            if next_cost > budget:
                continue
            candidate = (value + adjusted_value, selected + (event_id,))
            incumbent = updated.get(next_cost)
            if incumbent is None or candidate[0] > incumbent[0] or (
                math.isclose(candidate[0], incumbent[0]) and candidate[1] < incumbent[1]
            ):
                updated[next_cost] = candidate
        states = updated
    best_cost, (_, best_events) = max(
        states.items(),
        key=lambda item: (item[1][0], -item[0], tuple(-event_id for event_id in item[1][1])),
    )
    del best_cost
    return frozenset(best_events)


def minimum_distance_coalition(
    distance: DistanceFunction,
    event_costs: Mapping[int, int],
    *,
    budget: int,
) -> Coalition:
    return min(
        feasible_coalitions(event_costs, budget),
        key=lambda coalition: (float(distance(coalition)), sum(event_costs[event] for event in coalition), tuple(coalition)),
    )


def restoration_utility(distance: DistanceFunction, coalition: Coalition) -> float:
    return float(distance(frozenset())) - float(distance(coalition))


def jaccard(left: Coalition, right: Coalition) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and math.isclose(values[order[start]], values[order[end]]):
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def spearman(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    if set(left) != set(right) or len(left) < 2:
        raise ValueError("rankings require matching ids and at least two events")
    event_ids = sorted(left)
    left_ranks = _average_ranks([left[event_id] for event_id in event_ids])
    right_ranks = _average_ranks([right[event_id] for event_id in event_ids])
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks))
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left_ranks))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right_ranks))
    if left_scale == 0 or right_scale == 0:
        return 0.0
    return numerator / (left_scale * right_scale)
