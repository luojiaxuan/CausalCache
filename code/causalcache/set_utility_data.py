"""Validated state tables and padded batches for set-utility learning."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

UTILITY_TIE_EPSILON = 1e-12


def _finite_vector(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{label} must be a numeric sequence")
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite values")
    return result


def _canonical_subset(
    value: Sequence[int],
    *,
    event_ids: tuple[int, ...],
    label: str,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{label} must be an event-id sequence")
    result = tuple(value)
    if (
        any(type(event_id) is not int for event_id in result)
        or result != tuple(sorted(result))
        or len(set(result)) != len(result)
        or not set(result).issubset(event_ids)
    ):
        raise ValueError(f"{label} must be a sorted unique subset of the state events")
    return result


@dataclass(frozen=True)
class SubsetUtilityTarget:
    """One exact set-utility target."""

    subset: tuple[int, ...]
    raw_utility: float


@dataclass(frozen=True)
class SetUtilityState:
    """One query state with label-blind features and an exact utility table."""

    trajectory_id: str
    state_id: str
    event_ids: tuple[int, ...]
    maximum_labeled_cardinality: int
    query_features: tuple[float, ...]
    context_features: tuple[float, ...]
    event_features: tuple[tuple[float, ...], ...]
    targets: tuple[SubsetUtilityTarget, ...]
    normalization_scale: float | None
    pair_features: tuple[tuple[tuple[float, ...], ...], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_id, str) or not self.trajectory_id:
            raise ValueError("trajectory id must be non-empty text")
        if not isinstance(self.state_id, str) or not self.state_id:
            raise ValueError("state id must be non-empty text")
        if (
            not self.event_ids
            or any(type(event_id) is not int for event_id in self.event_ids)
            or self.event_ids != tuple(sorted(self.event_ids))
            or len(set(self.event_ids)) != len(self.event_ids)
        ):
            raise ValueError("event ids must be a non-empty sorted unique integer tuple")
        if len(self.event_ids) > 16:
            raise ValueError("set-utility state cannot exceed 16 candidate events")
        if type(self.maximum_labeled_cardinality) is not int or (
            self.maximum_labeled_cardinality not in (2, 3, 4)
        ):
            raise ValueError("maximum labeled cardinality must be 2, 3, or 4")

        query = _finite_vector(self.query_features, label="query features")
        context = _finite_vector(self.context_features, label="context features")
        events = tuple(
            _finite_vector(values, label="event features")
            for values in self.event_features
        )
        if len(events) != len(self.event_ids):
            raise ValueError("event feature count differs from the event-id count")
        event_dimension = len(events[0])
        if any(len(values) != event_dimension for values in events):
            raise ValueError("event feature dimensions differ within a state")
        object.__setattr__(self, "query_features", query)
        object.__setattr__(self, "context_features", context)
        object.__setattr__(self, "event_features", events)

        if self.normalization_scale is not None:
            scale = float(self.normalization_scale)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("normalization scale must be finite and positive")
            object.__setattr__(self, "normalization_scale", scale)

        canonical_targets: list[SubsetUtilityTarget] = []
        seen: set[tuple[int, ...]] = set()
        for target in self.targets:
            if not isinstance(target, SubsetUtilityTarget):
                raise TypeError("set-utility targets must be SubsetUtilityTarget values")
            subset = _canonical_subset(
                target.subset,
                event_ids=self.event_ids,
                label="target subset",
            )
            utility = float(target.raw_utility)
            if subset in seen or not math.isfinite(utility):
                raise ValueError("set-utility targets must be unique and finite")
            seen.add(subset)
            canonical_targets.append(SubsetUtilityTarget(subset, utility))
        canonical_targets.sort(key=lambda item: (len(item.subset), item.subset))
        if not canonical_targets or canonical_targets[0].subset != ():
            raise ValueError("the exact target table must include the empty set")
        if abs(canonical_targets[0].raw_utility) > UTILITY_TIE_EPSILON:
            raise ValueError("empty-set utility must be zero")
        expected_subsets = tuple(
            subset
            for cardinality in range(
                min(len(self.event_ids), self.maximum_labeled_cardinality) + 1
            )
            for subset in itertools.combinations(self.event_ids, cardinality)
        )
        if tuple(target.subset for target in canonical_targets) != expected_subsets:
            raise ValueError(
                "set-utility targets must contain exactly every subset up to the "
                "maximum labeled cardinality"
            )
        object.__setattr__(self, "targets", tuple(canonical_targets))

        if self.pair_features is not None:
            pair_rows = tuple(
                tuple(
                    _finite_vector(values, label="pair features")
                    for values in row
                )
                for row in self.pair_features
            )
            event_count = len(self.event_ids)
            if len(pair_rows) != event_count or any(
                len(row) != event_count for row in pair_rows
            ):
                raise ValueError("pair features must form an event-by-event matrix")
            pair_dimension = len(pair_rows[0][0])
            if any(
                len(values) != pair_dimension
                for row in pair_rows
                for values in row
            ):
                raise ValueError("pair feature dimensions differ within a state")
            object.__setattr__(self, "pair_features", pair_rows)

    @property
    def target_by_subset(self) -> dict[tuple[int, ...], float]:
        return {target.subset: target.raw_utility for target in self.targets}

    def utility(self, subset: Sequence[int]) -> float:
        canonical = _canonical_subset(
            subset,
            event_ids=self.event_ids,
            label="queried subset",
        )
        try:
            return self.target_by_subset[canonical]
        except KeyError as error:
            raise KeyError(f"utility is not labeled for subset {canonical}") from error


@dataclass(frozen=True)
class RankingRow:
    batch_index: int
    left_row: int
    right_row: int
    target_sign: int
    weight: float
    cardinality_bucket: tuple[int, int]


@dataclass(frozen=True)
class PaddedSetUtilityBatch:
    """Dependency-free padded representation; tensor conversion is optional."""

    trajectory_ids: tuple[str, ...]
    state_ids: tuple[str, ...]
    event_ids: tuple[tuple[int | None, ...], ...]
    query_features: tuple[tuple[float, ...], ...]
    context_features: tuple[tuple[float, ...], ...]
    event_features: tuple[tuple[tuple[float, ...], ...], ...]
    event_mask: tuple[tuple[bool, ...], ...]
    subsets: tuple[tuple[tuple[int, ...] | None, ...], ...]
    subset_masks: tuple[tuple[tuple[bool, ...], ...], ...]
    subset_row_mask: tuple[tuple[bool, ...], ...]
    raw_targets: tuple[tuple[float, ...], ...]
    normalized_targets: tuple[tuple[float, ...], ...]
    normalization_scales: tuple[float, ...]
    normalized_state_mask: tuple[bool, ...]
    raw_regression_weights: tuple[tuple[float, ...], ...]
    normalized_regression_weights: tuple[tuple[float, ...], ...]
    ranking_rows: tuple[RankingRow, ...]
    pair_features: tuple[
        tuple[tuple[tuple[float, ...], ...], ...], ...
    ] | None

    @property
    def batch_size(self) -> int:
        return len(self.state_ids)

    @property
    def maximum_event_count(self) -> int:
        return len(self.event_mask[0])

    @property
    def maximum_subset_count(self) -> int:
        return len(self.subset_row_mask[0])


@dataclass(frozen=True)
class TensorSetUtilityBatch:
    trajectory_ids: tuple[str, ...]
    state_ids: tuple[str, ...]
    subsets: tuple[tuple[tuple[int, ...] | None, ...], ...]
    query_features: Any
    context_features: Any
    event_features: Any
    event_mask: Any
    subset_masks: Any
    subset_row_mask: Any
    raw_targets: Any
    normalized_targets: Any
    normalization_scales: Any
    normalized_state_mask: Any
    raw_regression_weights: Any
    normalized_regression_weights: Any
    ranking_batch_indices: Any
    ranking_left_rows: Any
    ranking_right_rows: Any
    ranking_signs: Any
    ranking_weights: Any
    pair_features: Any | None


def _validate_state_dimensions(states: Sequence[SetUtilityState]) -> None:
    query_dimension = len(states[0].query_features)
    context_dimension = len(states[0].context_features)
    event_dimension = len(states[0].event_features[0])
    pair_dimensions = {
        None
        if state.pair_features is None
        else len(state.pair_features[0][0])
        for state in states
    }
    if any(
        len(state.query_features) != query_dimension
        or len(state.context_features) != context_dimension
        or len(state.event_features[0]) != event_dimension
        for state in states
    ):
        raise ValueError("feature dimensions differ across set-utility states")
    if len(pair_dimensions) != 1:
        raise ValueError("pair-feature availability or dimensions differ across states")


def _balanced_regression_weights(
    states: Sequence[SetUtilityState],
    *,
    normalized: bool,
) -> tuple[tuple[float, ...], ...]:
    eligible = [
        index
        for index, state in enumerate(states)
        if not normalized or state.normalization_scale is not None
    ]
    weights = [[0.0] * len(state.targets) for state in states]
    if not eligible:
        return tuple(tuple(row) for row in weights)
    state_weight = 1.0 / len(eligible)
    for state_index in eligible:
        state = states[state_index]
        by_cardinality: dict[int, list[int]] = defaultdict(list)
        for row_index, target in enumerate(state.targets):
            by_cardinality[len(target.subset)].append(row_index)
        cardinality_weight = state_weight / len(by_cardinality)
        for row_indices in by_cardinality.values():
            row_weight = cardinality_weight / len(row_indices)
            for row_index in row_indices:
                weights[state_index][row_index] = row_weight
    return tuple(tuple(row) for row in weights)


def _balanced_ranking_rows(
    states: Sequence[SetUtilityState],
    *,
    tie_epsilon: float,
) -> tuple[RankingRow, ...]:
    if not math.isfinite(tie_epsilon) or tie_epsilon < 0.0:
        raise ValueError("ranking tie epsilon must be finite and non-negative")
    by_state: dict[int, dict[tuple[int, int], list[tuple[int, int, int]]]] = {}
    for batch_index, state in enumerate(states):
        buckets: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
        for left, right in itertools.combinations(range(len(state.targets)), 2):
            delta = (
                state.targets[left].raw_utility
                - state.targets[right].raw_utility
            )
            if abs(delta) <= tie_epsilon:
                continue
            cardinalities = tuple(
                sorted(
                    (
                        len(state.targets[left].subset),
                        len(state.targets[right].subset),
                    )
                )
            )
            buckets[cardinalities].append(
                (left, right, 1 if delta > 0.0 else -1)
            )
        if buckets:
            by_state[batch_index] = dict(buckets)
    if not by_state:
        return ()

    result: list[RankingRow] = []
    state_weight = 1.0 / len(by_state)
    for batch_index, buckets in sorted(by_state.items()):
        bucket_weight = state_weight / len(buckets)
        for cardinality_bucket, pairs in sorted(buckets.items()):
            pair_weight = bucket_weight / len(pairs)
            result.extend(
                RankingRow(
                    batch_index=batch_index,
                    left_row=left,
                    right_row=right,
                    target_sign=sign,
                    weight=pair_weight,
                    cardinality_bucket=cardinality_bucket,
                )
                for left, right, sign in pairs
            )
    return tuple(result)


def build_padded_set_utility_batch(
    states: Sequence[SetUtilityState],
    *,
    ranking_tie_epsilon: float = UTILITY_TIE_EPSILON,
) -> PaddedSetUtilityBatch:
    """Pad events and exact subset rows with state/cardinality-balanced weights."""
    if isinstance(states, (str, bytes, bytearray, Mapping)) or not states:
        raise ValueError("set-utility batch must contain states")
    selected = tuple(states)
    if any(not isinstance(state, SetUtilityState) for state in selected):
        raise TypeError("set-utility batch contains an invalid state")
    if len({state.state_id for state in selected}) != len(selected):
        raise ValueError("set-utility state ids must be unique within a batch")
    _validate_state_dimensions(selected)

    maximum_events = max(len(state.event_ids) for state in selected)
    maximum_rows = max(len(state.targets) for state in selected)
    event_dimension = len(selected[0].event_features[0])
    raw_weights = _balanced_regression_weights(selected, normalized=False)
    normalized_weights = _balanced_regression_weights(selected, normalized=True)
    ranking_rows = _balanced_ranking_rows(
        selected,
        tie_epsilon=ranking_tie_epsilon,
    )

    padded_event_ids: list[tuple[int | None, ...]] = []
    event_features: list[tuple[tuple[float, ...], ...]] = []
    event_masks: list[tuple[bool, ...]] = []
    subsets: list[tuple[tuple[int, ...] | None, ...]] = []
    subset_masks: list[tuple[tuple[bool, ...], ...]] = []
    subset_row_masks: list[tuple[bool, ...]] = []
    raw_targets: list[tuple[float, ...]] = []
    normalized_targets: list[tuple[float, ...]] = []
    padded_raw_weights: list[tuple[float, ...]] = []
    padded_normalized_weights: list[tuple[float, ...]] = []
    padded_pairs: list[tuple[tuple[tuple[float, ...], ...], ...]] = []

    pair_dimension = (
        None
        if selected[0].pair_features is None
        else len(selected[0].pair_features[0][0])
    )
    for state_index, state in enumerate(selected):
        pad_events = maximum_events - len(state.event_ids)
        padded_event_ids.append((*state.event_ids, *((None,) * pad_events)))
        event_features.append(
            (*state.event_features, *((0.0,) * event_dimension,) * pad_events)
        )
        event_masks.append(
            (True,) * len(state.event_ids) + (False,) * pad_events
        )
        index_by_event = {
            event_id: index for index, event_id in enumerate(state.event_ids)
        }
        row_masks: list[tuple[bool, ...]] = []
        state_subsets: list[tuple[int, ...] | None] = []
        state_raw: list[float] = []
        state_normalized: list[float] = []
        state_rows_mask: list[bool] = []
        for target in state.targets:
            selected_indices = {index_by_event[event_id] for event_id in target.subset}
            row_masks.append(
                tuple(index in selected_indices for index in range(maximum_events))
            )
            state_subsets.append(target.subset)
            state_raw.append(target.raw_utility)
            state_normalized.append(
                0.0
                if state.normalization_scale is None
                else target.raw_utility / state.normalization_scale
            )
            state_rows_mask.append(True)
        pad_rows = maximum_rows - len(state.targets)
        row_masks.extend(((False,) * maximum_events,) * pad_rows)
        state_subsets.extend((None,) * pad_rows)
        state_raw.extend((0.0,) * pad_rows)
        state_normalized.extend((0.0,) * pad_rows)
        state_rows_mask.extend((False,) * pad_rows)
        subset_masks.append(tuple(row_masks))
        subsets.append(tuple(state_subsets))
        raw_targets.append(tuple(state_raw))
        normalized_targets.append(tuple(state_normalized))
        subset_row_masks.append(tuple(state_rows_mask))
        padded_raw_weights.append((*raw_weights[state_index], *((0.0,) * pad_rows)))
        padded_normalized_weights.append(
            (*normalized_weights[state_index], *((0.0,) * pad_rows))
        )

        if pair_dimension is not None:
            assert state.pair_features is not None
            pair_zero = (0.0,) * pair_dimension
            rows = [
                [pair_zero for _ in range(maximum_events)]
                for _ in range(maximum_events)
            ]
            for left in range(len(state.event_ids)):
                for right in range(len(state.event_ids)):
                    rows[left][right] = state.pair_features[left][right]
            padded_pairs.append(tuple(tuple(row) for row in rows))

    batch = PaddedSetUtilityBatch(
        trajectory_ids=tuple(state.trajectory_id for state in selected),
        state_ids=tuple(state.state_id for state in selected),
        event_ids=tuple(padded_event_ids),
        query_features=tuple(state.query_features for state in selected),
        context_features=tuple(state.context_features for state in selected),
        event_features=tuple(event_features),
        event_mask=tuple(event_masks),
        subsets=tuple(subsets),
        subset_masks=tuple(subset_masks),
        subset_row_mask=tuple(subset_row_masks),
        raw_targets=tuple(raw_targets),
        normalized_targets=tuple(normalized_targets),
        normalization_scales=tuple(
            0.0 if state.normalization_scale is None else state.normalization_scale
            for state in selected
        ),
        normalized_state_mask=tuple(
            state.normalization_scale is not None for state in selected
        ),
        raw_regression_weights=tuple(padded_raw_weights),
        normalized_regression_weights=tuple(padded_normalized_weights),
        ranking_rows=ranking_rows,
        pair_features=tuple(padded_pairs) if pair_dimension is not None else None,
    )
    if not math.isclose(
        math.fsum(math.fsum(row) for row in batch.raw_regression_weights),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("raw regression weights do not sum to one")
    normalized_sum = math.fsum(
        math.fsum(row) for row in batch.normalized_regression_weights
    )
    if any(batch.normalized_state_mask) and not math.isclose(
        normalized_sum, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("normalized regression weights do not sum to one")
    if batch.ranking_rows and not math.isclose(
        math.fsum(row.weight for row in batch.ranking_rows),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("ranking weights do not sum to one")
    return batch


def set_utility_state_from_feature_and_table(
    feature_state: Any,
    label_table: Any,
    *,
    normalization_epsilon: float = UTILITY_TIE_EPSILON,
) -> SetUtilityState:
    """Join the canonical label-blind feature state to its exact utility table."""
    from causalcache.set_utility_features import SetUtilityFeatureState
    from causalcache.set_utility_label_table import ValidatedSetUtilityTable

    if not isinstance(feature_state, SetUtilityFeatureState):
        raise TypeError("feature_state must be SetUtilityFeatureState")
    if not isinstance(label_table, ValidatedSetUtilityTable):
        raise TypeError("label_table must be ValidatedSetUtilityTable")
    if (
        not math.isfinite(normalization_epsilon)
        or normalization_epsilon < 0.0
    ):
        raise ValueError("normalization epsilon must be finite and non-negative")
    if feature_state.state_id != label_table.state_id:
        raise ValueError("feature and label state ids differ")
    if feature_state.event_step_ids != label_table.candidate_event_step_ids:
        raise ValueError("feature and label candidate inventories differ")
    event_count = len(feature_state.event_step_ids)
    expected_mask = (True,) * event_count + (False,) * (
        len(feature_state.event_mask) - event_count
    )
    if feature_state.event_mask != expected_mask:
        raise ValueError("feature-state padding mask is not canonical")
    baseline = label_table.distance(())
    return SetUtilityState(
        trajectory_id=feature_state.source_id,
        state_id=feature_state.state_id,
        event_ids=feature_state.event_step_ids,
        maximum_labeled_cardinality=label_table.maximum_labeled_cardinality,
        query_features=feature_state.query_features,
        context_features=feature_state.context_features,
        event_features=feature_state.event_features[:event_count],
        targets=tuple(
            SubsetUtilityTarget(
                subset=row.coalition_event_step_ids,
                raw_utility=row.utility,
            )
            for row in label_table.rows
        ),
        normalization_scale=(
            baseline if baseline > normalization_epsilon else None
        ),
        pair_features=tuple(
            tuple(row[:event_count])
            for row in feature_state.pair_features[:event_count]
        ),
    )


def tensorize_set_utility_batch(
    batch: PaddedSetUtilityBatch,
    *,
    device: Any = "cpu",
    dtype: Any | None = None,
) -> TensorSetUtilityBatch:
    """Convert a validated padded batch to PyTorch without making PyTorch mandatory."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("tensorizing set-utility batches requires PyTorch") from error
    if not isinstance(batch, PaddedSetUtilityBatch):
        raise TypeError("batch must be PaddedSetUtilityBatch")
    selected_dtype = torch.float32 if dtype is None else dtype
    ranking = batch.ranking_rows
    return TensorSetUtilityBatch(
        trajectory_ids=batch.trajectory_ids,
        state_ids=batch.state_ids,
        subsets=batch.subsets,
        query_features=torch.tensor(
            batch.query_features, dtype=selected_dtype, device=device
        ),
        context_features=torch.tensor(
            batch.context_features, dtype=selected_dtype, device=device
        ),
        event_features=torch.tensor(
            batch.event_features, dtype=selected_dtype, device=device
        ),
        event_mask=torch.tensor(batch.event_mask, dtype=torch.bool, device=device),
        subset_masks=torch.tensor(
            batch.subset_masks, dtype=torch.bool, device=device
        ),
        subset_row_mask=torch.tensor(
            batch.subset_row_mask, dtype=torch.bool, device=device
        ),
        raw_targets=torch.tensor(
            batch.raw_targets, dtype=selected_dtype, device=device
        ),
        normalized_targets=torch.tensor(
            batch.normalized_targets, dtype=selected_dtype, device=device
        ),
        normalization_scales=torch.tensor(
            batch.normalization_scales, dtype=selected_dtype, device=device
        ),
        normalized_state_mask=torch.tensor(
            batch.normalized_state_mask, dtype=torch.bool, device=device
        ),
        raw_regression_weights=torch.tensor(
            batch.raw_regression_weights, dtype=selected_dtype, device=device
        ),
        normalized_regression_weights=torch.tensor(
            batch.normalized_regression_weights,
            dtype=selected_dtype,
            device=device,
        ),
        ranking_batch_indices=torch.tensor(
            [row.batch_index for row in ranking], dtype=torch.long, device=device
        ),
        ranking_left_rows=torch.tensor(
            [row.left_row for row in ranking], dtype=torch.long, device=device
        ),
        ranking_right_rows=torch.tensor(
            [row.right_row for row in ranking], dtype=torch.long, device=device
        ),
        ranking_signs=torch.tensor(
            [row.target_sign for row in ranking],
            dtype=selected_dtype,
            device=device,
        ),
        ranking_weights=torch.tensor(
            [row.weight for row in ranking], dtype=selected_dtype, device=device
        ),
        pair_features=(
            None
            if batch.pair_features is None
            else torch.tensor(
                batch.pair_features, dtype=selected_dtype, device=device
            )
        ),
    )


__all__ = [
    "PaddedSetUtilityBatch",
    "RankingRow",
    "SetUtilityState",
    "SubsetUtilityTarget",
    "TensorSetUtilityBatch",
    "UTILITY_TIE_EPSILON",
    "build_padded_set_utility_batch",
    "set_utility_state_from_feature_and_table",
    "tensorize_set_utility_batch",
]
