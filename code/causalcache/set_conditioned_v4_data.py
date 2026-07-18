"""Pair-only supervision for the frozen-base residual v4 study."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from causalcache.gate_v1_data import (
    LABEL_TIE_EPSILON,
    NORMALIZATION_EPSILON,
    FeatureState,
    GateState,
)
from causalcache.set_conditioned_v3_data import (
    feasible_coalitions,
    v3_independent_input,
)

BASE_HIDDEN_DIMENSION = 88
PAIR_FEATURE_DIMENSION = 4 * BASE_HIDDEN_DIMENSION + 64


@dataclass(frozen=True)
class V4EventRow:
    source_id: str
    state_id: str
    event_step_id: int
    input_vector: tuple[float, ...]


@dataclass(frozen=True)
class V4PairTarget:
    source_id: str
    state_id: str
    coalition: tuple[int, int]
    left_event_index: int
    right_event_index: int
    q64: tuple[float, ...]
    raw_pair_utility: float
    normalized_pair_utility: float | None
    regression_weight: float


@dataclass(frozen=True)
class V4RankingPair:
    state_id: str
    left_coalition: tuple[int, ...]
    right_coalition: tuple[int, ...]
    target_sign: int
    weight: float


@dataclass(frozen=True)
class V4TrainingBatch:
    states: tuple[GateState, ...]
    event_rows: tuple[V4EventRow, ...]
    pair_targets: tuple[V4PairTarget, ...]
    ranking_pairs: tuple[V4RankingPair, ...]

    @property
    def regression_weight_sum(self) -> float:
        return math.fsum(row.regression_weight for row in self.pair_targets)

    @property
    def ranking_weight_sum(self) -> float:
        return math.fsum(row.weight for row in self.ranking_pairs)

    @property
    def eligible_pair_count(self) -> int:
        return sum(row.normalized_pair_utility is not None for row in self.pair_targets)


def v4_independent_input(
    state: FeatureState | GateState,
    event_step_id: int,
) -> tuple[float, ...]:
    """Replay the exact 200-dimensional input consumed by gate v1."""
    return v3_independent_input(state, event_step_id)


def _validate_states(states: Sequence[GateState]) -> tuple[GateState, ...]:
    if isinstance(states, (str, bytes, bytearray)):
        raise TypeError("v4 states must be a sequence of GateState values")
    result = tuple(states)
    if not result or any(not isinstance(state, GateState) for state in result):
        raise ValueError("v4 training requires a non-empty GateState sequence")
    state_ids = tuple(state.state_id for state in result)
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("v4 training states contain duplicate state ids")
    for state in result:
        feasible_coalitions(state)
        for event_id in state.candidate_event_step_ids:
            v4_independent_input(state, event_id)
    return result


def _equal_state_weights(states: Sequence[GateState]) -> dict[tuple[str, str], float]:
    by_source: dict[str, list[str]] = defaultdict(list)
    for state in states:
        by_source[state.source_id].append(state.state_id)
    if not by_source:
        raise ValueError("v4 weighting requires at least one trajectory")
    return {
        (source_id, state_id): 1.0 / len(by_source) / len(state_ids)
        for source_id, state_ids in by_source.items()
        for state_id in state_ids
    }


def build_training_batch(states: Sequence[GateState]) -> V4TrainingBatch:
    """Build normalized pair corrections and pair-involving ranking rows."""
    frozen_states = _validate_states(states)
    eligible_states = tuple(
        state
        for state in frozen_states
        if state.table.distance(()) > NORMALIZATION_EPSILON
    )
    if not eligible_states:
        raise ValueError("v4 training has no state eligible for normalized regression")
    regression_state_weights = _equal_state_weights(eligible_states)

    event_rows: list[V4EventRow] = []
    event_indices: dict[tuple[str, int], int] = {}
    pair_targets: list[V4PairTarget] = []
    ranking_groups: list[
        tuple[
            GateState,
            tuple[tuple[tuple[int, ...], tuple[int, ...], int], ...],
        ]
    ] = []

    for state in frozen_states:
        for event_id in state.candidate_event_step_ids:
            key = (state.state_id, event_id)
            if key in event_indices:
                raise RuntimeError("v4 event index inventory contains a duplicate")
            event_indices[key] = len(event_rows)
            event_rows.append(
                V4EventRow(
                    source_id=state.source_id,
                    state_id=state.state_id,
                    event_step_id=event_id,
                    input_vector=v4_independent_input(state, event_id),
                )
            )

        baseline = state.table.distance(())
        eligible = baseline > NORMALIZATION_EPSILON
        pairs = tuple(itertools.combinations(state.candidate_event_step_ids, 2))
        state_regression_weight = (
            regression_state_weights[(state.source_id, state.state_id)]
            if eligible
            else 0.0
        )
        for coalition in pairs:
            raw_utility = state.table.utility(coalition)
            normalized = raw_utility / baseline if eligible else None
            if not math.isfinite(raw_utility) or (
                normalized is not None and not math.isfinite(normalized)
            ):
                raise ValueError("v4 pair target is non-finite")
            pair_targets.append(
                V4PairTarget(
                    source_id=state.source_id,
                    state_id=state.state_id,
                    coalition=coalition,
                    left_event_index=event_indices[(state.state_id, coalition[0])],
                    right_event_index=event_indices[(state.state_id, coalition[1])],
                    q64=state.q64,
                    raw_pair_utility=raw_utility,
                    normalized_pair_utility=normalized,
                    regression_weight=(
                        state_regression_weight / len(pairs) if eligible else 0.0
                    ),
                )
            )

        utility_by_coalition = {
            coalition: state.table.utility(coalition)
            for coalition in feasible_coalitions(state)
        }
        comparisons: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
        for left, right in itertools.combinations(feasible_coalitions(state), 2):
            if len(left) != 2 and len(right) != 2:
                continue
            delta = utility_by_coalition[left] - utility_by_coalition[right]
            if abs(delta) <= LABEL_TIE_EPSILON:
                continue
            comparisons.append((left, right, 1 if delta > 0.0 else -1))
        if comparisons:
            ranking_groups.append((state, tuple(comparisons)))

    ranking_states = tuple(state for state, _ in ranking_groups)
    ranking_state_weights = (
        _equal_state_weights(ranking_states) if ranking_states else {}
    )
    ranking_pairs = tuple(
        V4RankingPair(
            state_id=state.state_id,
            left_coalition=left,
            right_coalition=right,
            target_sign=sign,
            weight=(
                ranking_state_weights[(state.source_id, state.state_id)]
                / len(comparisons)
            ),
        )
        for state, comparisons in ranking_groups
        for left, right, sign in comparisons
    )

    batch = V4TrainingBatch(
        states=frozen_states,
        event_rows=tuple(event_rows),
        pair_targets=tuple(pair_targets),
        ranking_pairs=ranking_pairs,
    )
    if not math.isclose(
        batch.regression_weight_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("v4 normalized pair-regression weights do not sum to one")
    if batch.ranking_pairs and not math.isclose(
        batch.ranking_weight_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("v4 pair-involving ranking weights do not sum to one")
    return batch


__all__ = [
    "BASE_HIDDEN_DIMENSION",
    "PAIR_FEATURE_DIMENSION",
    "V4EventRow",
    "V4PairTarget",
    "V4RankingPair",
    "V4TrainingBatch",
    "build_training_batch",
    "feasible_coalitions",
    "v4_independent_input",
]
