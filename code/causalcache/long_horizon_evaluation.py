"""Sparse long-horizon restoration evaluation after label-blind selection sealing."""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from causalcache.long_horizon_selectors import (
    LabelBlindSelectionSeal,
    SelectionDecision,
)


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 271_828
BOOTSTRAP_CONFIDENCE = 0.9
NORMALIZATION_EPSILON = 1e-12
EXACT_ORACLE_BUDGETS = (2, 4)
HORIZON_BINS = (
    ("n_le_4", 1, 4),
    ("n_5_to_8", 5, 8),
    ("n_9_to_16", 9, 16),
    ("n_gt_16", 17, None),
)


def _event_ids(values: Any) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError("distance-table event ids must be an ordered sequence")
    try:
        result = tuple(values)
    except TypeError as error:
        raise TypeError("distance-table event ids must be an ordered sequence") from error
    if not result or any(type(event) is not int for event in result):
        raise ValueError("distance-table event ids must be non-empty integers")
    if result != tuple(sorted(result)) or len(set(result)) != len(result) or result[0] <= 0:
        raise ValueError("distance-table event ids must be positive and strictly increasing")
    return result


def _coalition(
    values: Iterable[int],
    *,
    event_ids: Sequence[int],
    label: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{label} must be an event-id iterable")
    try:
        raw = tuple(values)
    except TypeError as error:
        raise TypeError(f"{label} must be an event-id iterable") from error
    if any(type(event) is not int for event in raw):
        raise TypeError(f"{label} must contain integer event ids")
    if len(set(raw)) != len(raw):
        raise ValueError(f"{label} contains duplicate event ids")
    result = tuple(sorted(raw))
    if not set(result).issubset(event_ids):
        raise ValueError(f"{label} contains an unknown event id")
    return result


def _distance(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _required_budget_coalitions(
    event_ids: Sequence[int],
    maximum_budget: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        coalition
        for size in range(maximum_budget + 1)
        for coalition in itertools.combinations(event_ids, size)
    )


@dataclass(frozen=True)
class SparseReferenceDistanceTable:
    """All measured at-most-B distances plus an unmeasured self-reference invariant."""

    event_ids: tuple[int, ...]
    maximum_enumerated_budget: int
    rows: tuple[tuple[tuple[int, ...], float], ...]
    full_reference_distance: float

    @property
    def full_reference_coalition(self) -> tuple[int, ...]:
        return self.event_ids

    def distance(self, coalition: Iterable[int]) -> float:
        key = _coalition(coalition, event_ids=self.event_ids, label="distance lookup")
        if key == self.full_reference_coalition:
            return self.full_reference_distance
        try:
            return dict(self.rows)[key]
        except KeyError as error:
            raise ValueError("coalition is absent from the sparse distance table") from error

    @property
    def reference_recoverable_utility(self) -> float:
        return self.distance(()) - self.distance(self.full_reference_coalition)

    def raw_utility(self, coalition: Iterable[int]) -> float:
        return self.distance(()) - self.distance(coalition)

    def normalized_recovery(self, coalition: Iterable[int]) -> float:
        denominator = self.distance(())
        if denominator <= NORMALIZATION_EPSILON:
            return 0.0
        return self.raw_utility(coalition) / denominator


def build_sparse_reference_distance_table(
    event_ids: Sequence[int],
    distances: Mapping[Iterable[int], Real],
    *,
    maximum_enumerated_budget: int,
    full_reference_distance: Real = 0.0,
) -> SparseReferenceDistanceTable:
    """Validate exactly the measured ``|S|<=B`` rows and a zero self invariant."""
    events = _event_ids(event_ids)
    if (
        type(maximum_enumerated_budget) is not int
        or maximum_enumerated_budget <= 0
        or maximum_enumerated_budget >= len(events)
    ):
        raise ValueError("sparse-table maximum budget must be positive and smaller than n")
    if not isinstance(distances, Mapping):
        raise TypeError("sparse distances must be a coalition-to-distance mapping")
    normalized: dict[tuple[int, ...], float] = {}
    for raw_coalition, raw_distance in distances.items():
        coalition = _coalition(
            raw_coalition,
            event_ids=events,
            label="sparse distance coalition",
        )
        if coalition in normalized:
            raise ValueError("sparse distance table contains a duplicate coalition")
        normalized[coalition] = _distance(raw_distance, f"distance for {coalition}")
    required = _required_budget_coalitions(events, maximum_enumerated_budget)
    required_set = set(required)
    if set(normalized) != required_set:
        missing = sorted(required_set - set(normalized), key=lambda item: (len(item), item))
        extra = sorted(set(normalized) - required_set, key=lambda item: (len(item), item))
        raise ValueError(
            "sparse table must contain all and only measured budget-feasible "
            f"coalitions; missing={missing}, extra={extra}"
        )
    reference_distance = _distance(
        full_reference_distance, "complete-history self-reference distance"
    )
    if reference_distance != 0.0:
        raise ValueError("complete-history self-reference distance must equal zero")
    table = SparseReferenceDistanceTable(
        event_ids=events,
        maximum_enumerated_budget=maximum_enumerated_budget,
        rows=tuple((coalition, normalized[coalition]) for coalition in required),
        full_reference_distance=reference_distance,
    )
    return table


@dataclass(frozen=True)
class ExactAtMostBOracle:
    budget_event_capacity: int
    coalition: tuple[int, ...]
    distance: float
    raw_utility: float
    normalized_recovery: float
    evaluated_coalition_count: int
    exact_within_budget: bool = True
    full_power_set_enumerated: bool = False


def exact_at_most_b_oracle(
    table: SparseReferenceDistanceTable,
    *,
    budget_event_capacity: int,
) -> ExactAtMostBOracle:
    """Return the exact oracle among all subsets satisfying ``|S|<=B``."""
    if not isinstance(table, SparseReferenceDistanceTable):
        raise TypeError("exact oracle requires a SparseReferenceDistanceTable")
    if type(budget_event_capacity) is not int or budget_event_capacity not in EXACT_ORACLE_BUDGETS:
        raise ValueError(f"exact long-horizon oracle budget must be one of {EXACT_ORACLE_BUDGETS}")
    if budget_event_capacity > table.maximum_enumerated_budget:
        raise ValueError("exact oracle budget exceeds the enumerated sparse-table budget")
    feasible = tuple(
        (coalition, distance)
        for coalition, distance in table.rows
        if len(coalition) <= budget_event_capacity
    )
    coalition, distance = min(feasible, key=lambda row: (row[1], len(row[0]), row[0]))
    return ExactAtMostBOracle(
        budget_event_capacity=budget_event_capacity,
        coalition=coalition,
        distance=distance,
        raw_utility=table.raw_utility(coalition),
        normalized_recovery=table.normalized_recovery(coalition),
        evaluated_coalition_count=len(feasible),
        full_power_set_enumerated=(
            table.maximum_enumerated_budget >= len(table.event_ids)
        ),
    )


@dataclass(frozen=True)
class StateSelectionMetric:
    selector_name: str
    source_id: str
    state_id: str
    horizon_n: int
    budget_event_capacity: int
    selected_event_step_ids: tuple[int, ...]
    distance: float
    raw_utility: float
    normalized_recovery: float


def _metric_from_decision(
    decision: SelectionDecision,
    table: SparseReferenceDistanceTable,
) -> StateSelectionMetric:
    if decision.candidate_event_step_ids != table.event_ids:
        raise ValueError("sealed selection and distance table candidate geometries differ")
    return StateSelectionMetric(
        selector_name=decision.selector_name,
        source_id=decision.source_id,
        state_id=decision.state_id,
        horizon_n=len(table.event_ids),
        budget_event_capacity=decision.budget_event_capacity,
        selected_event_step_ids=decision.selected_event_step_ids,
        distance=table.distance(decision.selected_event_step_ids),
        raw_utility=table.raw_utility(decision.selected_event_step_ids),
        normalized_recovery=table.normalized_recovery(decision.selected_event_step_ids),
    )


def evaluate_sparse_tables_after_seal(
    seal: LabelBlindSelectionSeal,
    tables_by_state_id: Mapping[str, SparseReferenceDistanceTable],
) -> tuple[StateSelectionMetric, ...]:
    """Join label tables only through an already-constructed opaque selection seal."""
    if not isinstance(seal, LabelBlindSelectionSeal):
        raise TypeError("evaluation requires a LabelBlindSelectionSeal")
    if not isinstance(tables_by_state_id, Mapping):
        raise TypeError("distance tables must be indexed by state id")
    expected = {record.state_id for record in seal.records}
    if set(tables_by_state_id) != expected:
        raise ValueError("distance-table state inventory differs from the sealed roster")
    return tuple(
        _metric_from_decision(record, tables_by_state_id[record.state_id])
        for record in seal.records
    )


@dataclass(frozen=True)
class ComparatorPairUnionDistanceTable:
    """One comparator pair under its union reference; never a subset oracle."""

    source_id: str
    state_id: str
    event_ids: tuple[int, ...]
    selections: tuple[tuple[str, tuple[int, ...]], ...]
    rows: tuple[tuple[tuple[int, ...], float], ...]
    pair_union_reference_coalition: tuple[int, ...]
    selector_budget_event_capacity: int
    supports_exact_subset_oracle: bool = False

    def distance(self, coalition: Iterable[int]) -> float:
        key = _coalition(coalition, event_ids=self.event_ids, label="pair-union lookup")
        try:
            return dict(self.rows)[key]
        except KeyError as error:
            raise ValueError(
                "coalition was not measured under this comparator-pair union reference"
            ) from error

    def selector_distance(self, selector_name: str) -> float:
        try:
            selected = dict(self.selections)[selector_name]
        except KeyError as error:
            raise ValueError("unknown pair-union comparator") from error
        return self.distance(selected)

    def pair_union_distance(self, left: str, right: str) -> float:
        selections = dict(self.selections)
        if left not in selections or right not in selections:
            raise ValueError("unknown pair-union comparator")
        return self.distance(tuple(sorted(set(selections[left]) | set(selections[right]))))

    @property
    def reference_recoverable_utility(self) -> float:
        return self.distance(()) - self.distance(self.pair_union_reference_coalition)

    def selector_metric(self, selector_name: str) -> tuple[float, float]:
        raw = self.distance(()) - self.selector_distance(selector_name)
        denominator = self.distance(())
        normalized = raw / denominator if denominator > NORMALIZATION_EPSILON else 0.0
        return raw, normalized


def build_comparator_pair_union_distance_table(
    *,
    source_id: str,
    state_id: str,
    event_ids: Sequence[int],
    selections: Mapping[str, Sequence[int]],
    distances: Mapping[Iterable[int], Real],
    maximum_selector_budget: int = 4,
) -> ComparatorPairUnionDistanceTable:
    """Validate empty, A, B, and R=A-union-B for exactly one comparator pair."""
    events = _event_ids(event_ids)
    if (
        not isinstance(source_id, str)
        or not source_id
        or not isinstance(state_id, str)
        or not state_id
    ):
        raise ValueError("pair-union source/state identities must be non-empty")
    if type(maximum_selector_budget) is not int or maximum_selector_budget <= 0:
        raise ValueError("pair-union selector budget must be positive")
    if not isinstance(selections, Mapping) or len(selections) != 2:
        raise ValueError("each pair-union table requires exactly two comparators")
    normalized_selections: dict[str, tuple[int, ...]] = {}
    for name, raw in selections.items():
        if not isinstance(name, str) or not name:
            raise ValueError("pair-union comparator names must be non-empty")
        selected = _coalition(raw, event_ids=events, label=f"{name} selection")
        if len(selected) > maximum_selector_budget:
            raise ValueError("pair-union comparator selection exceeds its budget")
        normalized_selections[name] = selected
    left, right = sorted(normalized_selections)
    reference = tuple(
        sorted(set(normalized_selections[left]) | set(normalized_selections[right]))
    )
    required = {(), *normalized_selections.values(), reference}
    if not isinstance(distances, Mapping):
        raise TypeError("pair-union distances must be a mapping")
    normalized_distances: dict[tuple[int, ...], float] = {}
    for raw, value in distances.items():
        coalition = _coalition(raw, event_ids=events, label="pair-union distance coalition")
        if coalition in normalized_distances:
            raise ValueError("pair-union distance table contains a duplicate coalition")
        normalized_distances[coalition] = _distance(value, f"distance for {coalition}")
    if set(normalized_distances) != required:
        raise ValueError(
            "pair-union table must contain exactly empty, A, B, and R=A-union-B"
        )
    ordered_rows = tuple(
        (coalition, normalized_distances[coalition])
        for coalition in sorted(required, key=lambda item: (len(item), item))
    )
    result = ComparatorPairUnionDistanceTable(
        source_id=source_id,
        state_id=state_id,
        event_ids=events,
        selections=tuple(sorted(normalized_selections.items())),
        rows=ordered_rows,
        pair_union_reference_coalition=reference,
        selector_budget_event_capacity=maximum_selector_budget,
    )
    if result.distance(reference) != 0.0:
        raise ValueError("pair-union reference self-distance must equal zero")
    return result


def evaluate_pair_union_table(
    table: ComparatorPairUnionDistanceTable,
) -> tuple[StateSelectionMetric, ...]:
    if not isinstance(table, ComparatorPairUnionDistanceTable):
        raise TypeError("pair-union evaluation requires a comparator-specific table")
    result = []
    for name, selected in table.selections:
        raw, normalized = table.selector_metric(name)
        result.append(
            StateSelectionMetric(
                selector_name=name,
                source_id=table.source_id,
                state_id=table.state_id,
                horizon_n=len(table.event_ids),
                budget_event_capacity=table.selector_budget_event_capacity,
                selected_event_step_ids=selected,
                distance=table.selector_distance(name),
                raw_utility=raw,
                normalized_recovery=normalized,
            )
        )
    return tuple(result)


def type7_quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("type-7 quantile input is invalid")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("type-7 quantile values must be finite")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def trajectory_equal_selector_mean(
    records: Sequence[StateSelectionMetric],
    *,
    selector_name: str,
    metric: str,
    budget_event_capacity: int | None = None,
) -> tuple[float, int, int]:
    """Average states within trajectory first, then average trajectories equally."""
    if metric not in {"raw_utility", "normalized_recovery"}:
        raise ValueError("trajectory-equal metric is unsupported")
    if budget_event_capacity is not None and (
        type(budget_event_capacity) is not int or budget_event_capacity <= 0
    ):
        raise ValueError("trajectory-equal budget filter must be positive")
    matching = [
        record
        for record in records
        if record.selector_name == selector_name
        and (
            budget_event_capacity is None
            or record.budget_event_capacity == budget_event_capacity
        )
    ]
    if not matching:
        raise ValueError("trajectory-equal aggregation has no matching selector records")
    keys = [(record.source_id, record.state_id) for record in matching]
    if len(set(keys)) != len(keys):
        raise ValueError("trajectory-equal aggregation requires an explicit budget filter")
    by_source: dict[str, list[float]] = defaultdict(list)
    excluded_states = 0
    for record in matching:
        value = getattr(record, metric)
        if value is None:
            excluded_states += 1
        else:
            by_source[record.source_id].append(float(value))
    means = [statistics.fmean(values) for _, values in sorted(by_source.items()) if values]
    if not means:
        raise ValueError("trajectory-equal aggregation has no eligible trajectory")
    all_sources = {record.source_id for record in matching}
    return statistics.fmean(means), excluded_states, len(all_sources) - len(means)


def paired_trajectory_deltas(
    records: Sequence[StateSelectionMetric],
    *,
    left_selector: str,
    right_selector: str,
    metric: str,
    budget_event_capacity: int | None = None,
) -> Mapping[str, float]:
    """Build left-minus-right paired state means within each trajectory."""
    if metric not in {"raw_utility", "normalized_recovery"}:
        raise ValueError("paired trajectory metric is unsupported")
    if budget_event_capacity is not None and (
        type(budget_event_capacity) is not int or budget_event_capacity <= 0
    ):
        raise ValueError("paired trajectory budget filter must be positive")
    by_key: dict[tuple[str, str], dict[str, StateSelectionMetric]] = defaultdict(dict)
    for record in records:
        if record.selector_name in {left_selector, right_selector} and (
            budget_event_capacity is None
            or record.budget_event_capacity == budget_event_capacity
        ):
            key = (record.source_id, record.state_id)
            if record.selector_name in by_key[key]:
                raise ValueError("paired metric records contain a duplicate selector/state")
            by_key[key][record.selector_name] = record
    by_source: dict[str, list[float]] = defaultdict(list)
    for (source_id, _), pair in sorted(by_key.items()):
        if set(pair) != {left_selector, right_selector}:
            continue
        left = getattr(pair[left_selector], metric)
        right = getattr(pair[right_selector], metric)
        if left is not None and right is not None:
            by_source[source_id].append(float(left) - float(right))
    result = {
        source_id: statistics.fmean(values)
        for source_id, values in sorted(by_source.items())
        if values
    }
    if not result:
        raise ValueError("paired metric has no eligible trajectories")
    return result


@dataclass(frozen=True)
class PairedBootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    seed: int
    trajectory_count: int
    quantile_method: str = "type7_linear"
    resampling_unit: str = "trajectory"


def paired_trajectory_bootstrap(
    trajectory_deltas: Mapping[str, float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> PairedBootstrapInterval:
    """Return a deterministic two-sided percentile interval over trajectories."""
    if not isinstance(trajectory_deltas, Mapping) or not trajectory_deltas:
        raise ValueError("paired bootstrap requires trajectory-indexed deltas")
    if type(resamples) is not int or resamples <= 0:
        raise ValueError("paired bootstrap resamples must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("paired bootstrap seed must be a non-negative integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("paired bootstrap confidence must lie in (0, 1)")
    values = tuple(
        _distance_like_signed(trajectory_deltas[source], f"delta for {source}")
        for source in sorted(trajectory_deltas)
    )
    generator = random.Random(seed)
    bootstrap_means = tuple(
        math.fsum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    return PairedBootstrapInterval(
        estimate=statistics.fmean(values),
        lower=type7_quantile(bootstrap_means, tail),
        upper=type7_quantile(bootstrap_means, 1.0 - tail),
        confidence=confidence,
        resamples=resamples,
        seed=seed,
        trajectory_count=len(values),
    )


def _distance_like_signed(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class HorizonIncidenceBin:
    name: str
    minimum_n: int
    maximum_n: int | None
    decision_count: int
    decision_fraction: float
    trajectory_count: int
    trajectory_fraction: float


@dataclass(frozen=True)
class HorizonIncidenceReport:
    total_decision_count: int
    total_trajectory_count: int
    minimum_n: int
    maximum_n: int
    exact_n_decision_counts: tuple[tuple[int, int], ...]
    bins: tuple[HorizonIncidenceBin, ...]


def horizon_incidence(states: Sequence[Any]) -> HorizonIncidenceReport:
    """Count actual candidate-history exposure by decision and by trajectory."""
    rows = tuple(states)
    if not rows:
        raise ValueError("horizon incidence requires at least one state")
    identities: set[tuple[str, str]] = set()
    exact: dict[int, int] = defaultdict(int)
    trajectory_ns: dict[str, set[int]] = defaultdict(set)
    for state in rows:
        source_id = getattr(state, "source_id", None)
        state_id = getattr(state, "state_id", None)
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(state_id, str)
            or not state_id
        ):
            raise ValueError("horizon incidence state identity is malformed")
        identity = (source_id, state_id)
        if identity in identities:
            raise ValueError("horizon incidence contains a duplicate state")
        identities.add(identity)
        n = len(_event_ids(getattr(state, "candidate_event_step_ids", None)))
        exact[n] += 1
        trajectory_ns[source_id].add(n)
    total_decisions = len(rows)
    total_trajectories = len(trajectory_ns)
    bins: list[HorizonIncidenceBin] = []
    for name, minimum, maximum in HORIZON_BINS:
        contains = lambda n: n >= minimum and (maximum is None or n <= maximum)
        decision_count = sum(count for n, count in exact.items() if contains(n))
        trajectory_count = sum(
            any(contains(n) for n in values) for values in trajectory_ns.values()
        )
        bins.append(
            HorizonIncidenceBin(
                name=name,
                minimum_n=minimum,
                maximum_n=maximum,
                decision_count=decision_count,
                decision_fraction=decision_count / total_decisions,
                trajectory_count=trajectory_count,
                trajectory_fraction=trajectory_count / total_trajectories,
            )
        )
    return HorizonIncidenceReport(
        total_decision_count=total_decisions,
        total_trajectory_count=total_trajectories,
        minimum_n=min(exact),
        maximum_n=max(exact),
        exact_n_decision_counts=tuple(sorted(exact.items())),
        bins=tuple(bins),
    )


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "ComparatorPairUnionDistanceTable",
    "ExactAtMostBOracle",
    "HORIZON_BINS",
    "HorizonIncidenceBin",
    "HorizonIncidenceReport",
    "PairedBootstrapInterval",
    "SparseReferenceDistanceTable",
    "StateSelectionMetric",
    "build_comparator_pair_union_distance_table",
    "build_sparse_reference_distance_table",
    "evaluate_pair_union_table",
    "evaluate_sparse_tables_after_seal",
    "exact_at_most_b_oracle",
    "horizon_incidence",
    "paired_trajectory_bootstrap",
    "paired_trajectory_deltas",
    "trajectory_equal_selector_mean",
    "type7_quantile",
]
