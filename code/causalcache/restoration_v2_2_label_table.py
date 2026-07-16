"""Exact restoration-v2.2 distance-table labels and subset oracles."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any


Coalition = frozenset[int]

SUPPORTED_EVENT_COUNTS = (2, 3, 4)
PRIMARY_BUDGET_EVENT_CAPACITY = 2
EXACT_TIE_EPSILON = 0.0
STATES_PER_EVENT_COUNT = 15

FULL_DISTANCE_ROWS_BY_EVENT_COUNT = {2: 4, 3: 8, 4: 16}
DEPLOYMENT_EDGE_COUNTS_BY_EVENT_COUNT = {2: 4, 3: 9, 4: 16}
FULL_EDGE_COUNTS_BY_EVENT_COUNT = {2: 4, 3: 12, 4: 32}
PAIR_INTERACTION_COUNTS_BY_EVENT_COUNT = {2: 1, 3: 6, 4: 24}
ATTRIBUTION_ROW_COUNTS_BY_EVENT_COUNT = {2: 2, 3: 3, 4: 4}

EXPECTED_FULL_45_DISTANCE_ROW_COUNT = 420
EXPECTED_FULL_45_DEPLOYMENT_EDGE_COUNT = 435
EXPECTED_FULL_45_FULL_EDGE_COUNT = 720
EXPECTED_FULL_45_PAIR_INTERACTION_COUNT = 465
EXPECTED_FULL_45_ATTRIBUTION_ROW_COUNT = 135


@dataclass(frozen=True)
class DistanceRow:
    coalition: tuple[int, ...]
    distance: float
    utility: float


@dataclass(frozen=True)
class ValidatedDistanceTable:
    event_ids: tuple[int, ...]
    rows: tuple[DistanceRow, ...]

    def distance(self, coalition: Iterable[int]) -> float:
        normalized = _normalize_coalition(
            coalition,
            event_ids=self.event_ids,
            name="coalition",
        )
        key = tuple(sorted(normalized))
        for row in self.rows:
            if row.coalition == key:
                return row.distance
        raise ValueError("coalition is absent from the validated full distance table")

    def utility(self, coalition: Iterable[int]) -> float:
        return self.distance(()) - self.distance(coalition)


@dataclass(frozen=True)
class ExactSubsetOracle:
    budget_event_capacity: int
    tie_epsilon: float
    coalition: tuple[int, ...]
    distance: float
    utility: float
    evaluated_coalition_count: int


@dataclass(frozen=True)
class ConditionalMarginalEdge:
    base_coalition: tuple[int, ...]
    event_id: int
    restored_coalition: tuple[int, ...]
    base_distance: float
    restored_distance: float
    marginal_gain: float


@dataclass(frozen=True)
class PairInteraction:
    conditioning_coalition: tuple[int, ...]
    left_event_id: int
    right_event_id: int
    interaction: float


@dataclass(frozen=True)
class PermutationAverageAttribution:
    event_id: int
    mean_marginal_gain: float
    permutation_count: int
    marginal_samples: tuple[float, ...]


def _normalize_event_ids(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError("candidate_event_step_ids must be an ordered sequence")
    try:
        event_ids = tuple(values)
    except TypeError as error:
        raise TypeError("candidate_event_step_ids must be an ordered sequence") from error
    if len(event_ids) not in SUPPORTED_EVENT_COUNTS:
        raise ValueError("candidate event count must be exactly 2, 3, or 4")
    if any(type(event_id) is not int for event_id in event_ids):
        raise TypeError("candidate event step ids must be integers")
    if any(event_id <= 0 for event_id in event_ids):
        raise ValueError("candidate event step ids must be positive")
    if event_ids != tuple(sorted(event_ids)) or len(set(event_ids)) != len(event_ids):
        raise ValueError("candidate event step ids must be unique and strictly increasing")
    return event_ids


def _normalize_coalition(
    value: Iterable[int],
    *,
    event_ids: Sequence[int],
    name: str,
) -> Coalition:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{name} must be an event-id iterable")
    try:
        raw = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an event-id iterable") from error
    if any(type(event_id) is not int for event_id in raw):
        raise TypeError(f"{name} must contain only integer event ids")
    if len(raw) != len(set(raw)):
        raise ValueError(f"{name} cannot contain duplicate event ids")
    normalized = frozenset(raw)
    unknown = normalized - frozenset(event_ids)
    if unknown:
        raise ValueError(f"{name} contains unknown events: {sorted(unknown)}")
    return normalized


def _all_coalitions(event_ids: Sequence[int]) -> tuple[Coalition, ...]:
    return tuple(
        frozenset(subset)
        for size in range(len(event_ids) + 1)
        for subset in itertools.combinations(event_ids, size)
    )


def _finite_nonnegative_distance(value: Any, *, coalition: Coalition) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"distance for {sorted(coalition)} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"distance for {sorted(coalition)} must be finite and non-negative")
    return converted


def validate_complete_distance_table(
    candidate_event_step_ids: Sequence[int],
    distances: Mapping[Iterable[int], Real],
) -> ValidatedDistanceTable:
    """Validate all and only the power-set distances for one frozen state."""
    event_ids = _normalize_event_ids(candidate_event_step_ids)
    if not isinstance(distances, Mapping):
        raise TypeError("distances must be a coalition-to-distance mapping")

    normalized: dict[Coalition, float] = {}
    for raw_coalition, raw_distance in distances.items():
        coalition = _normalize_coalition(
            raw_coalition,
            event_ids=event_ids,
            name="distance-table coalition",
        )
        if coalition in normalized:
            raise ValueError(
                "distance table contains duplicate coalitions after normalization"
            )
        normalized[coalition] = _finite_nonnegative_distance(
            raw_distance,
            coalition=coalition,
        )

    expected_order = _all_coalitions(event_ids)
    expected = frozenset(expected_order)
    observed = frozenset(normalized)
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise ValueError(
            "distance table must contain the complete power set; "
            f"missing={[sorted(item) for item in sorted(missing, key=_coalition_sort_key)]}, "
            f"extra={[sorted(item) for item in sorted(extra, key=_coalition_sort_key)]}"
        )
    expected_count = FULL_DISTANCE_ROWS_BY_EVENT_COUNT[len(event_ids)]
    if len(normalized) != expected_count:
        raise ValueError("distance-table row count drifted from the frozen power set")

    empty_distance = normalized[frozenset()]
    rows = tuple(
        DistanceRow(
            coalition=tuple(sorted(coalition)),
            distance=normalized[coalition],
            utility=empty_distance - normalized[coalition],
        )
        for coalition in expected_order
    )
    return ValidatedDistanceTable(event_ids=event_ids, rows=rows)


def _coalition_sort_key(coalition: Coalition) -> tuple[int, tuple[int, ...]]:
    return len(coalition), tuple(sorted(coalition))


def restoration_utility(
    table: ValidatedDistanceTable,
    coalition: Iterable[int],
) -> float:
    _require_table(table)
    return table.utility(coalition)


def primary_exact_subset_oracle(
    table: ValidatedDistanceTable,
) -> ExactSubsetOracle:
    """Minimize distance over empty, singleton, and pair coalitions exactly."""
    _require_table(table)
    feasible = tuple(
        row for row in table.rows if len(row.coalition) <= PRIMARY_BUDGET_EVENT_CAPACITY
    )
    selected = min(
        feasible,
        key=lambda row: (row.distance, len(row.coalition), row.coalition),
    )
    return ExactSubsetOracle(
        budget_event_capacity=PRIMARY_BUDGET_EVENT_CAPACITY,
        tie_epsilon=EXACT_TIE_EPSILON,
        coalition=selected.coalition,
        distance=selected.distance,
        utility=selected.utility,
        evaluated_coalition_count=len(feasible),
    )


def deployment_conditional_edges(
    table: ValidatedDistanceTable,
) -> tuple[ConditionalMarginalEdge, ...]:
    """Return every conditional edge reachable under the primary B=2 budget."""
    return _conditional_edges(table, maximum_restored_events=PRIMARY_BUDGET_EVENT_CAPACITY)


def full_conditional_edges(
    table: ValidatedDistanceTable,
) -> tuple[ConditionalMarginalEdge, ...]:
    """Return every upward directed edge of the complete subset hypercube."""
    _require_table(table)
    return _conditional_edges(table, maximum_restored_events=len(table.event_ids))


def _conditional_edges(
    table: ValidatedDistanceTable,
    *,
    maximum_restored_events: int,
) -> tuple[ConditionalMarginalEdge, ...]:
    _require_table(table)
    if type(maximum_restored_events) is not int or maximum_restored_events <= 0:
        raise ValueError("maximum_restored_events must be a positive integer")
    distance_by_coalition = {
        frozenset(row.coalition): row.distance for row in table.rows
    }
    edges: list[ConditionalMarginalEdge] = []
    for base in sorted(distance_by_coalition, key=_coalition_sort_key):
        if len(base) >= maximum_restored_events:
            continue
        for event_id in table.event_ids:
            if event_id in base:
                continue
            restored = base | {event_id}
            if len(restored) > maximum_restored_events:
                continue
            base_distance = distance_by_coalition[base]
            restored_distance = distance_by_coalition[restored]
            edges.append(
                ConditionalMarginalEdge(
                    base_coalition=tuple(sorted(base)),
                    event_id=event_id,
                    restored_coalition=tuple(sorted(restored)),
                    base_distance=base_distance,
                    restored_distance=restored_distance,
                    marginal_gain=base_distance - restored_distance,
                )
            )
    return tuple(edges)


def pair_interactions(
    table: ValidatedDistanceTable,
) -> tuple[PairInteraction, ...]:
    """Return all pair interactions at every valid conditioning coalition."""
    _require_table(table)
    distance_by_coalition = {
        frozenset(row.coalition): row.distance for row in table.rows
    }
    interactions: list[PairInteraction] = []
    for left, right in itertools.combinations(table.event_ids, 2):
        remaining = tuple(
            event_id
            for event_id in table.event_ids
            if event_id not in {left, right}
        )
        for conditioning in _all_coalitions(remaining):
            left_value = distance_by_coalition[conditioning | {left}]
            right_value = distance_by_coalition[conditioning | {right}]
            base_value = distance_by_coalition[conditioning]
            pair_value = distance_by_coalition[conditioning | {left, right}]
            interactions.append(
                PairInteraction(
                    conditioning_coalition=tuple(sorted(conditioning)),
                    left_event_id=left,
                    right_event_id=right,
                    interaction=(
                        left_value + right_value - base_value - pair_value
                    ),
                )
            )
    return tuple(interactions)


def exact_permutation_average_attribution(
    table: ValidatedDistanceTable,
) -> tuple[PermutationAverageAttribution, ...]:
    """Average exact conditional gains over all event-order permutations."""
    _require_table(table)
    samples: dict[int, list[float]] = {
        event_id: [] for event_id in table.event_ids
    }
    permutation_count = 0
    for permutation in itertools.permutations(table.event_ids):
        permutation_count += 1
        prefix: Coalition = frozenset()
        for event_id in permutation:
            restored = prefix | {event_id}
            samples[event_id].append(
                table.distance(prefix) - table.distance(restored)
            )
            prefix = restored
    expected_permutations = math.factorial(len(table.event_ids))
    if permutation_count != expected_permutations:
        raise RuntimeError("exact permutation count drifted")
    return tuple(
        PermutationAverageAttribution(
            event_id=event_id,
            mean_marginal_gain=math.fsum(samples[event_id]) / permutation_count,
            permutation_count=permutation_count,
            marginal_samples=tuple(samples[event_id]),
        )
        for event_id in table.event_ids
    )


def _require_table(table: ValidatedDistanceTable) -> None:
    if not isinstance(table, ValidatedDistanceTable):
        raise TypeError("table must be a ValidatedDistanceTable")
