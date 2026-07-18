"""Cardinality-capped restoration tables for budget-agnostic set utility."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any


MAXIMUM_CANDIDATE_EVENT_COUNT = 16
PRIMARY_MAXIMUM_LABELED_CARDINALITY = 2
SUPPORTED_MAXIMUM_LABELED_CARDINALITIES = (2, 3, 4)
EXACT_ORACLE_TIE_BREAK = (
    "higher_utility",
    "smaller_cardinality",
    "lexicographically_smaller_event_id_tuple",
)
EQUAL_WEIGHTING_SCHEME = (
    "split_normalized_state_equal_cardinality_equal_subset_equal"
)


@dataclass(frozen=True)
class SetUtilityDistanceInputRow:
    """One untrusted distance row before table-level validation."""

    split: str
    state_id: str
    candidate_event_step_ids: tuple[int, ...]
    maximum_labeled_cardinality: int
    coalition_event_step_ids: tuple[int, ...]
    distance: float


@dataclass(frozen=True)
class SetUtilityLabelRow:
    """One canonical non-negative distance and its signed derived utility."""

    split: str
    state_id: str
    coalition_event_step_ids: tuple[int, ...]
    cardinality: int
    distance: float
    utility: float


@dataclass(frozen=True)
class ValidatedSetUtilityTable:
    """All and only the subsets up to one state's label-cardinality cap."""

    split: str
    state_id: str
    candidate_event_step_ids: tuple[int, ...]
    maximum_labeled_cardinality: int
    rows: tuple[SetUtilityLabelRow, ...]

    @property
    def candidate_event_count(self) -> int:
        return len(self.candidate_event_step_ids)

    @property
    def labeled_cardinalities(self) -> tuple[int, ...]:
        return tuple(
            range(min(self.candidate_event_count, self.maximum_labeled_cardinality) + 1)
        )

    def distance(self, coalition: Iterable[int]) -> float:
        key = _canonical_coalition(
            coalition,
            candidate_event_step_ids=self.candidate_event_step_ids,
            maximum_cardinality=self.maximum_labeled_cardinality,
            label="queried coalition",
        )
        for row in self.rows:
            if row.coalition_event_step_ids == key:
                return row.distance
        raise AssertionError("validated capped table is internally incomplete")

    def utility(self, coalition: Iterable[int]) -> float:
        return self.distance(()) - self.distance(coalition)


@dataclass(frozen=True)
class ExactAtMostBudgetOracle:
    split: str
    state_id: str
    budget: int
    coalition_event_step_ids: tuple[int, ...]
    cardinality: int
    distance: float
    utility: float
    evaluated_coalition_count: int
    tie_break: tuple[str, ...] = EXACT_ORACLE_TIE_BREAK


@dataclass(frozen=True)
class EqualWeightingRow:
    split: str
    state_id: str
    coalition_event_step_ids: tuple[int, ...]
    cardinality: int
    state_weight: float
    cardinality_weight_within_state: float
    subset_weight_within_cardinality: float
    normalized_row_weight: float


@dataclass(frozen=True)
class EqualWeightingMetadata:
    """Auditable loss weights normalized independently inside each split."""

    scheme: str
    normalization_scope: str
    split_state_counts: tuple[tuple[str, int], ...]
    rows: tuple[EqualWeightingRow, ...]

    def row_weight(
        self,
        *,
        split: str,
        state_id: str,
        coalition_event_step_ids: Iterable[int],
    ) -> float:
        coalition = tuple(sorted(coalition_event_step_ids))
        matches = tuple(
            row
            for row in self.rows
            if row.split == split
            and row.state_id == state_id
            and row.coalition_event_step_ids == coalition
        )
        if len(matches) != 1:
            raise ValueError("weight lookup must identify exactly one validated subset row")
        return matches[0].normalized_row_weight


@dataclass(frozen=True)
class ValidatedSetUtilityDataset:
    tables: tuple[ValidatedSetUtilityTable, ...]
    weighting: EqualWeightingMetadata


def expected_capped_row_count(
    candidate_event_count: int,
    maximum_labeled_cardinality: int,
) -> int:
    _validate_candidate_event_count(candidate_event_count)
    maximum_labeled_cardinality = _validate_maximum_labeled_cardinality(
        maximum_labeled_cardinality
    )
    return sum(
        math.comb(candidate_event_count, cardinality)
        for cardinality in range(
            min(candidate_event_count, maximum_labeled_cardinality) + 1
        )
    )


def validate_cardinality_capped_distance_table(
    *,
    split: str,
    state_id: str,
    candidate_event_step_ids: Sequence[int],
    maximum_labeled_cardinality: int,
    distances: Mapping[Iterable[int], Real],
) -> ValidatedSetUtilityTable:
    """Validate exactly all ``|S| <= K`` rows without requiring a power set."""
    split = _nonempty_identity(split, "split")
    state_id = _nonempty_identity(state_id, "state_id")
    event_ids = _candidate_event_ids(candidate_event_step_ids)
    cap = _validate_maximum_labeled_cardinality(maximum_labeled_cardinality)
    if not isinstance(distances, Mapping):
        raise TypeError("distances must be a coalition-to-distance mapping")

    normalized: dict[tuple[int, ...], float] = {}
    for raw_coalition, raw_distance in distances.items():
        coalition = _canonical_coalition(
            raw_coalition,
            candidate_event_step_ids=event_ids,
            maximum_cardinality=cap,
            label="distance-table coalition",
        )
        if coalition in normalized:
            raise ValueError("distance table contains a duplicate normalized coalition")
        normalized[coalition] = _finite_distance(
            raw_distance,
            coalition=coalition,
        )

    expected_order = _expected_coalitions(event_ids, cap)
    expected = frozenset(expected_order)
    observed = frozenset(normalized)
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise ValueError(
            "distance table must contain exactly every cardinality-capped subset; "
            f"missing={_display_coalitions(missing)}, "
            f"extra={_display_coalitions(extra)}"
        )
    expected_count = expected_capped_row_count(len(event_ids), cap)
    if len(normalized) != expected_count:
        raise AssertionError("cardinality-capped row count is internally inconsistent")

    empty_distance = normalized[()]
    rows = tuple(
        SetUtilityLabelRow(
            split=split,
            state_id=state_id,
            coalition_event_step_ids=coalition,
            cardinality=len(coalition),
            distance=normalized[coalition],
            utility=empty_distance - normalized[coalition],
        )
        for coalition in expected_order
    )
    return ValidatedSetUtilityTable(
        split=split,
        state_id=state_id,
        candidate_event_step_ids=event_ids,
        maximum_labeled_cardinality=cap,
        rows=rows,
    )


def validate_cardinality_capped_label_rows(
    rows: Sequence[SetUtilityDistanceInputRow],
) -> ValidatedSetUtilityDataset:
    """Bind serialized rows to one state and split, then validate each table."""
    if isinstance(rows, (str, bytes, bytearray, Mapping)) or not isinstance(
        rows, Sequence
    ):
        raise TypeError("label rows must be a non-empty sequence")
    if not rows:
        raise ValueError("label rows cannot be empty")
    if any(not isinstance(row, SetUtilityDistanceInputRow) for row in rows):
        raise TypeError("every label row must be SetUtilityDistanceInputRow")

    split_by_state: dict[str, str] = {}
    grouped: dict[tuple[str, str], list[SetUtilityDistanceInputRow]] = defaultdict(list)
    for row in rows:
        split = _nonempty_identity(row.split, "split")
        state_id = _nonempty_identity(row.state_id, "state_id")
        previous_split = split_by_state.setdefault(state_id, split)
        if previous_split != split:
            raise ValueError("one state_id cannot contribute subset rows to multiple splits")
        grouped[(split, state_id)].append(row)

    tables: list[ValidatedSetUtilityTable] = []
    for (split, state_id), state_rows in sorted(grouped.items()):
        first = state_rows[0]
        candidate_event_step_ids = _candidate_event_ids(
            first.candidate_event_step_ids
        )
        cap = _validate_maximum_labeled_cardinality(
            first.maximum_labeled_cardinality
        )
        distances: dict[tuple[int, ...], float] = {}
        for row in state_rows:
            if _candidate_event_ids(row.candidate_event_step_ids) != candidate_event_step_ids:
                raise ValueError("subset rows from one state disagree on candidate events")
            if _validate_maximum_labeled_cardinality(
                row.maximum_labeled_cardinality
            ) != cap:
                raise ValueError("subset rows from one state disagree on cardinality cap")
            coalition = _canonical_coalition(
                row.coalition_event_step_ids,
                candidate_event_step_ids=candidate_event_step_ids,
                maximum_cardinality=cap,
                label="serialized subset row",
            )
            if coalition in distances:
                raise ValueError("one state contains a duplicate normalized subset row")
            distances[coalition] = row.distance
        tables.append(
            validate_cardinality_capped_distance_table(
                split=split,
                state_id=state_id,
                candidate_event_step_ids=candidate_event_step_ids,
                maximum_labeled_cardinality=cap,
                distances=distances,
            )
        )
    canonical_tables = tuple(tables)
    return ValidatedSetUtilityDataset(
        tables=canonical_tables,
        weighting=build_equal_weighting_metadata(canonical_tables),
    )


def exact_at_most_budget_oracle(
    table: ValidatedSetUtilityTable,
    *,
    budget: int,
) -> ExactAtMostBudgetOracle:
    """Select by true capped utility with the frozen exact tie-break."""
    _require_validated_table(table)
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a non-negative integer")
    if budget > table.maximum_labeled_cardinality:
        raise ValueError("budget exceeds this table's exact labeled-cardinality cap")
    feasible = tuple(row for row in table.rows if row.cardinality <= budget)
    selected = min(
        feasible,
        key=lambda row: (
            -row.utility,
            row.cardinality,
            row.coalition_event_step_ids,
        ),
    )
    return ExactAtMostBudgetOracle(
        split=table.split,
        state_id=table.state_id,
        budget=budget,
        coalition_event_step_ids=selected.coalition_event_step_ids,
        cardinality=selected.cardinality,
        distance=selected.distance,
        utility=selected.utility,
        evaluated_coalition_count=len(feasible),
    )


def build_equal_weighting_metadata(
    tables: Sequence[ValidatedSetUtilityTable],
) -> EqualWeightingMetadata:
    """Give every state and every observed cardinality equal split-local mass."""
    if isinstance(tables, (str, bytes, bytearray, Mapping)) or not isinstance(
        tables, Sequence
    ):
        raise TypeError("tables must be a non-empty sequence")
    if not tables:
        raise ValueError("tables cannot be empty")
    if any(not isinstance(table, ValidatedSetUtilityTable) for table in tables):
        raise TypeError("weighting inputs must be validated set-utility tables")
    identities = [(table.split, table.state_id) for table in tables]
    if len(identities) != len(set(identities)):
        raise ValueError("weighting inputs contain a duplicate state table")
    split_by_state: dict[str, str] = {}
    for split, state_id in identities:
        previous = split_by_state.setdefault(state_id, split)
        if previous != split:
            raise ValueError("one state_id cannot be weighted in multiple splits")

    state_counts: dict[str, int] = defaultdict(int)
    for table in tables:
        state_counts[table.split] += 1
    weights: list[EqualWeightingRow] = []
    for table in sorted(tables, key=lambda item: (item.split, item.state_id)):
        state_weight = 1.0 / state_counts[table.split]
        cardinalities = table.labeled_cardinalities
        cardinality_weight = 1.0 / len(cardinalities)
        rows_by_cardinality: dict[int, list[SetUtilityLabelRow]] = defaultdict(list)
        for row in table.rows:
            rows_by_cardinality[row.cardinality].append(row)
        if tuple(sorted(rows_by_cardinality)) != cardinalities:
            raise AssertionError("validated table cardinality inventory drifted")
        for cardinality in cardinalities:
            cardinality_rows = rows_by_cardinality[cardinality]
            expected_count = math.comb(table.candidate_event_count, cardinality)
            if len(cardinality_rows) != expected_count:
                raise AssertionError("validated table cardinality count drifted")
            subset_weight = 1.0 / expected_count
            for row in cardinality_rows:
                weights.append(
                    EqualWeightingRow(
                        split=table.split,
                        state_id=table.state_id,
                        coalition_event_step_ids=row.coalition_event_step_ids,
                        cardinality=cardinality,
                        state_weight=state_weight,
                        cardinality_weight_within_state=cardinality_weight,
                        subset_weight_within_cardinality=subset_weight,
                        normalized_row_weight=(
                            state_weight * cardinality_weight * subset_weight
                        ),
                    )
                )
    return EqualWeightingMetadata(
        scheme=EQUAL_WEIGHTING_SCHEME,
        normalization_scope="independently_within_each_split",
        split_state_counts=tuple(sorted(state_counts.items())),
        rows=tuple(weights),
    )


def _validate_candidate_event_count(value: int) -> int:
    if type(value) is not int or not 0 <= value <= MAXIMUM_CANDIDATE_EVENT_COUNT:
        raise ValueError(
            f"candidate event count must be between 0 and {MAXIMUM_CANDIDATE_EVENT_COUNT}"
        )
    return value


def _validate_maximum_labeled_cardinality(value: int) -> int:
    if type(value) is not int or value not in SUPPORTED_MAXIMUM_LABELED_CARDINALITIES:
        raise ValueError("maximum labeled cardinality must be exactly 2, 3, or 4")
    return value


def _candidate_event_ids(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)) or not isinstance(
        values, Sequence
    ):
        raise TypeError("candidate event step ids must be an ordered sequence")
    event_ids = tuple(values)
    _validate_candidate_event_count(len(event_ids))
    if any(type(event_id) is not int for event_id in event_ids):
        raise TypeError("candidate event step ids must be integers")
    if any(event_id <= 0 for event_id in event_ids):
        raise ValueError("candidate event step ids must be positive")
    if event_ids != tuple(sorted(event_ids)) or len(event_ids) != len(set(event_ids)):
        raise ValueError("candidate event step ids must be unique and strictly increasing")
    return event_ids


def _nonempty_identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty canonical text")
    return value


def _canonical_coalition(
    value: Iterable[int],
    *,
    candidate_event_step_ids: Sequence[int],
    maximum_cardinality: int,
    label: str,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{label} must be an event-id iterable")
    try:
        raw = tuple(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an event-id iterable") from error
    if any(type(event_id) is not int for event_id in raw):
        raise TypeError(f"{label} must contain only integer event ids")
    if len(raw) != len(set(raw)):
        raise ValueError(f"{label} cannot contain duplicate event ids")
    coalition = tuple(sorted(raw))
    unknown = frozenset(coalition) - frozenset(candidate_event_step_ids)
    if unknown:
        raise ValueError(f"{label} contains events outside its state: {sorted(unknown)}")
    if len(coalition) > maximum_cardinality:
        raise ValueError(f"{label} exceeds the exact labeled-cardinality cap")
    return coalition


def _finite_distance(value: Any, *, coalition: tuple[int, ...]) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"distance for {coalition} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"distance for {coalition} must be finite and non-negative")
    return converted


def _expected_coalitions(
    event_ids: Sequence[int],
    maximum_cardinality: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        coalition
        for cardinality in range(min(len(event_ids), maximum_cardinality) + 1)
        for coalition in itertools.combinations(event_ids, cardinality)
    )


def _display_coalitions(
    coalitions: Iterable[tuple[int, ...]],
) -> list[list[int]]:
    return [
        list(coalition)
        for coalition in sorted(coalitions, key=lambda item: (len(item), item))
    ]


def _require_validated_table(table: Any) -> None:
    if not isinstance(table, ValidatedSetUtilityTable):
        raise TypeError("table must be a validated set-utility table")
