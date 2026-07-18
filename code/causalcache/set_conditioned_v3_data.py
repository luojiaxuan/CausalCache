"""Training rows and exact B=2 geometry for set-conditioned v3."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from causalcache.gate_v1_data import (
    DEPLOYMENT_BUDGET,
    HASH_DIMENSION,
    INDEPENDENT_INPUT_DIMENSION,
    LABEL_TIE_EPSILON,
    FeatureState,
    GateState,
)


TARGET_SCALE_EPSILON = 1e-12
PAIR_FEATURE_DIMENSION = 320


@dataclass(frozen=True)
class V3TargetScale:
    """One fold-train-only scale shared by every utility target."""

    rms: float
    unclamped_rms: float
    clamped: bool

    def normalize(self, raw_value: float) -> float:
        value = float(raw_value) / self.rms
        if not math.isfinite(value):
            raise ValueError("normalized v3 target is non-finite")
        return value

    def to_raw(self, normalized_value: float) -> float:
        value = float(normalized_value) * self.rms
        if not math.isfinite(value):
            raise ValueError("raw v3 prediction is non-finite")
        return value


@dataclass(frozen=True)
class V3SingletonTarget:
    source_id: str
    state_id: str
    event_step_id: int
    input_vector: tuple[float, ...]
    raw_utility: float
    normalized_utility: float
    weight: float


@dataclass(frozen=True)
class V3PairTarget:
    source_id: str
    state_id: str
    coalition: tuple[int, int]
    left_singleton_index: int
    right_singleton_index: int
    q64: tuple[float, ...]
    raw_utility: float
    normalized_utility: float
    raw_residual: float
    normalized_residual: float
    utility_weight: float
    residual_weight: float


@dataclass(frozen=True)
class V3RankingPair:
    state_id: str
    left_coalition: tuple[int, ...]
    right_coalition: tuple[int, ...]
    target_sign: int
    weight: float


@dataclass(frozen=True)
class V3TrainingBatch:
    states: tuple[GateState, ...]
    target_scale: V3TargetScale
    singleton_targets: tuple[V3SingletonTarget, ...]
    pair_targets: tuple[V3PairTarget, ...]
    ranking_pairs: tuple[V3RankingPair, ...]

    @property
    def singleton_weight_sum(self) -> float:
        return math.fsum(row.weight for row in self.singleton_targets)

    @property
    def pair_weight_sum(self) -> float:
        return math.fsum(row.utility_weight for row in self.pair_targets)

    @property
    def residual_weight_sum(self) -> float:
        return math.fsum(row.residual_weight for row in self.pair_targets)

    @property
    def ranking_weight_sum(self) -> float:
        return math.fsum(row.weight for row in self.ranking_pairs)


@dataclass(frozen=True)
class _RawStateTargets:
    state: GateState
    singleton_utilities: tuple[tuple[int, float], ...]
    pair_utilities: tuple[tuple[tuple[int, int], float, float], ...]


def feasible_coalitions(
    state: FeatureState | GateState,
    *,
    budget: int = DEPLOYMENT_BUDGET,
) -> tuple[tuple[int, ...], ...]:
    """Return empty, singleton, then chronological pair coalitions."""
    if type(budget) is not int or budget != 2:
        raise ValueError("set-conditioned v3 freezes the deployment budget at two")
    event_ids = state.candidate_event_step_ids
    if event_ids != tuple(sorted(event_ids)) or len(set(event_ids)) != len(event_ids):
        raise ValueError("candidate event ids must be sorted and unique")
    if not 2 <= len(event_ids) <= 4:
        raise ValueError("set-conditioned v3 requires the frozen n=2/3/4 geometry")
    if isinstance(state, GateState) and state.table.event_ids != event_ids:
        raise ValueError("set-conditioned v3 feature/label geometry differs")
    result = (
        (),
        *((event_id,) for event_id in event_ids),
        *itertools.combinations(event_ids, 2),
    )
    if len(result) > 11:
        raise RuntimeError("B=2 feasible-set enumeration exceeded eleven coalitions")
    return tuple(result)


def v3_independent_input(
    state: FeatureState | GateState, event_step_id: int
) -> tuple[float, ...]:
    """Replay the v1 independent feature without requiring labels."""
    if not isinstance(state, (FeatureState, GateState)):
        raise TypeError("v3 inference state must be FeatureState or GateState")
    candidates = {
        candidate.event_step_id: candidate for candidate in state.candidates
    }
    if tuple(candidates) != state.candidate_event_step_ids or event_step_id not in candidates:
        raise ValueError("v3 candidate feature geometry is invalid")
    candidate = candidates[event_step_id]
    if (
        len(state.q64) != HASH_DIMENSION
        or len(candidate.h64) != HASH_DIMENSION
        or len(candidate.g8) != 8
    ):
        raise ValueError("v3 label-blind feature dimension drifted")
    vector = (
        *state.q64,
        *candidate.h64,
        *(
            left * right
            for left, right in zip(state.q64, candidate.h64, strict=True)
        ),
        *candidate.g8,
    )
    if len(vector) != INDEPENDENT_INPUT_DIMENSION or any(
        not math.isfinite(value) for value in vector
    ):
        raise ValueError("v3 independent input must contain 200 finite values")
    return vector


def _validate_states(states: Sequence[GateState]) -> tuple[GateState, ...]:
    if isinstance(states, (str, bytes, bytearray)):
        raise TypeError("v3 states must be a sequence of GateState values")
    result = tuple(states)
    if not result or any(not isinstance(state, GateState) for state in result):
        raise ValueError("v3 training states must be a non-empty GateState sequence")
    state_ids = [state.state_id for state in result]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("v3 training states contain duplicate state ids")
    for state in result:
        feasible_coalitions(state)
        if len(state.q64) != 64 or any(not math.isfinite(value) for value in state.q64):
            raise ValueError("v3 q64 must contain exactly 64 finite values")
        for event_id in state.candidate_event_step_ids:
            vector = v3_independent_input(state, event_id)
            if len(vector) != INDEPENDENT_INPUT_DIMENSION:
                raise RuntimeError("v3 independent input dimension drifted")
    return result


def _raw_state_targets(states: Sequence[GateState]) -> tuple[_RawStateTargets, ...]:
    result: list[_RawStateTargets] = []
    for state in _validate_states(states):
        singleton = tuple(
            (event_id, state.table.utility((event_id,)))
            for event_id in state.candidate_event_step_ids
        )
        singleton_by_event = dict(singleton)
        pairs = tuple(
            (
                pair,
                state.table.utility(pair),
                state.table.utility(pair)
                - singleton_by_event[pair[0]]
                - singleton_by_event[pair[1]],
            )
            for pair in itertools.combinations(state.candidate_event_step_ids, 2)
        )
        values = (
            *(value for _, value in singleton),
            *(value for _, value, _ in pairs),
            *(value for _, _, value in pairs),
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("v3 utility target is non-finite")
        result.append(
            _RawStateTargets(
                state=state,
                singleton_utilities=singleton,
                pair_utilities=pairs,
            )
        )
    return tuple(result)


def _source_state_weights(
    raw_states: Sequence[_RawStateTargets],
) -> dict[tuple[str, str], float]:
    by_source: dict[str, list[str]] = defaultdict(list)
    for row in raw_states:
        by_source[row.state.source_id].append(row.state.state_id)
    source_count = len(by_source)
    return {
        (source_id, state_id): 1.0 / source_count / len(state_ids)
        for source_id, state_ids in by_source.items()
        for state_id in state_ids
    }


def fit_target_scale(states: Sequence[GateState]) -> V3TargetScale:
    """Fit one shared RMS using only the supplied training-fold states."""
    raw_states = _raw_state_targets(states)
    state_weights = _source_state_weights(raw_states)
    singleton_square = 0.0
    pair_square = 0.0
    residual_square = 0.0
    for row in raw_states:
        state_weight = state_weights[(row.state.source_id, row.state.state_id)]
        singleton_square += state_weight * math.fsum(
            value * value for _, value in row.singleton_utilities
        ) / len(row.singleton_utilities)
        pair_square += state_weight * math.fsum(
            utility * utility for _, utility, _ in row.pair_utilities
        ) / len(row.pair_utilities)
        residual_square += state_weight * math.fsum(
            residual * residual for _, _, residual in row.pair_utilities
        ) / len(row.pair_utilities)
    unclamped = math.sqrt((singleton_square + pair_square + residual_square) / 3.0)
    if not math.isfinite(unclamped):
        raise ValueError("v3 target RMS is non-finite")
    rms = max(unclamped, TARGET_SCALE_EPSILON)
    return V3TargetScale(
        rms=rms,
        unclamped_rms=unclamped,
        clamped=unclamped < TARGET_SCALE_EPSILON,
    )


def _validate_scale(scale: V3TargetScale) -> None:
    if not isinstance(scale, V3TargetScale):
        raise TypeError("target_scale must be a V3TargetScale")
    if (
        not math.isfinite(scale.rms)
        or scale.rms < TARGET_SCALE_EPSILON
        or not math.isfinite(scale.unclamped_rms)
        or scale.unclamped_rms < 0.0
        or scale.clamped != (scale.unclamped_rms < TARGET_SCALE_EPSILON)
    ):
        raise ValueError("v3 target scale is malformed")


def build_training_batch(
    states: Sequence[GateState],
    *,
    target_scale: V3TargetScale | None = None,
) -> V3TrainingBatch:
    """Build equal trajectory/state/group weights and fold-scaled targets."""
    raw_states = _raw_state_targets(states)
    scale = fit_target_scale(tuple(row.state for row in raw_states)) if target_scale is None else target_scale
    _validate_scale(scale)
    state_weights = _source_state_weights(raw_states)
    singletons: list[V3SingletonTarget] = []
    pairs: list[V3PairTarget] = []
    singleton_index: dict[tuple[str, int], int] = {}

    for row in raw_states:
        state = row.state
        state_weight = state_weights[(state.source_id, state.state_id)]
        singleton_weight = state_weight / len(row.singleton_utilities)
        for event_id, raw_utility in row.singleton_utilities:
            index = len(singletons)
            singleton_index[(state.state_id, event_id)] = index
            singletons.append(
                V3SingletonTarget(
                    source_id=state.source_id,
                    state_id=state.state_id,
                    event_step_id=event_id,
                    input_vector=v3_independent_input(state, event_id),
                    raw_utility=raw_utility,
                    normalized_utility=scale.normalize(raw_utility),
                    weight=singleton_weight,
                )
            )
        pair_weight = state_weight / len(row.pair_utilities)
        for coalition, raw_utility, raw_residual in row.pair_utilities:
            pairs.append(
                V3PairTarget(
                    source_id=state.source_id,
                    state_id=state.state_id,
                    coalition=coalition,
                    left_singleton_index=singleton_index[(state.state_id, coalition[0])],
                    right_singleton_index=singleton_index[(state.state_id, coalition[1])],
                    q64=state.q64,
                    raw_utility=raw_utility,
                    normalized_utility=scale.normalize(raw_utility),
                    raw_residual=raw_residual,
                    normalized_residual=scale.normalize(raw_residual),
                    utility_weight=pair_weight,
                    residual_weight=pair_weight,
                )
            )

    ranking_by_state: list[tuple[_RawStateTargets, list[tuple[tuple[int, ...], tuple[int, ...], int]]]] = []
    for row in raw_states:
        utilities = {
            (): 0.0,
            **{(event_id,): value for event_id, value in row.singleton_utilities},
            **{coalition: value for coalition, value, _ in row.pair_utilities},
        }
        comparisons: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
        for left, right in itertools.combinations(feasible_coalitions(row.state), 2):
            delta = utilities[left] - utilities[right]
            if abs(delta) <= LABEL_TIE_EPSILON:
                continue
            comparisons.append((left, right, 1 if delta > 0.0 else -1))
        if comparisons:
            ranking_by_state.append((row, comparisons))

    ranking: list[V3RankingPair] = []
    if ranking_by_state:
        eligible_state_weights = _source_state_weights(
            tuple(row for row, _ in ranking_by_state)
        )
        for row, comparisons in ranking_by_state:
            group_weight = eligible_state_weights[(row.state.source_id, row.state.state_id)]
            comparison_weight = group_weight / len(comparisons)
            ranking.extend(
                V3RankingPair(
                    state_id=row.state.state_id,
                    left_coalition=left,
                    right_coalition=right,
                    target_sign=sign,
                    weight=comparison_weight,
                )
                for left, right, sign in comparisons
            )

    batch = V3TrainingBatch(
        states=tuple(row.state for row in raw_states),
        target_scale=scale,
        singleton_targets=tuple(singletons),
        pair_targets=tuple(pairs),
        ranking_pairs=tuple(ranking),
    )
    for name, value in (
        ("singleton", batch.singleton_weight_sum),
        ("pair", batch.pair_weight_sum),
        ("residual", batch.residual_weight_sum),
    ):
        if not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"v3 {name} weights do not sum to one")
    if batch.ranking_pairs and not math.isclose(
        batch.ranking_weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("v3 ranking weights do not sum to one")
    return batch


def raw_utility(state: GateState, coalition: Iterable[int]) -> float:
    normalized = tuple(coalition)
    if normalized not in feasible_coalitions(state):
        raise ValueError("coalition is outside the frozen B=2 feasible set")
    value = state.table.utility(normalized)
    if not math.isfinite(value):
        raise ValueError("v3 raw utility is non-finite")
    return value


__all__ = [
    "PAIR_FEATURE_DIMENSION",
    "TARGET_SCALE_EPSILON",
    "V3PairTarget",
    "V3RankingPair",
    "V3SingletonTarget",
    "V3TargetScale",
    "V3TrainingBatch",
    "build_training_batch",
    "feasible_coalitions",
    "fit_target_scale",
    "raw_utility",
    "v3_independent_input",
]
