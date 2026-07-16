"""Trajectory-first statistics for restoration-v2.2 selector geometry."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any


DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 271_828
DEFAULT_BOOTSTRAP_CONFIDENCE = 0.90
DEFAULT_TIE_EPSILON = 1e-12
DEFAULT_NORMALIZATION_EPSILON = 1e-12
LINEAR_PERCENTILE_INTERVAL = "percentile_linear_interpolation"


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _positive_finite(value: Any, *, name: str) -> float:
    converted = _finite_float(value, name=name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


@dataclass(frozen=True)
class StateMetricRow:
    """One state's named metrics and its trajectory/split identity."""

    role: str
    trajectory_id: str
    state_id: str
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        _nonempty_string(self.role, name="role")
        _nonempty_string(self.trajectory_id, name="trajectory_id")
        _nonempty_string(self.state_id, name="state_id")
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise ValueError("metrics must be a non-empty mapping")
        normalized: dict[str, float] = {}
        for raw_name, raw_value in self.metrics.items():
            name = _nonempty_string(raw_name, name="metric name")
            if name in normalized:
                raise ValueError(f"duplicate metric name: {name}")
            normalized[name] = _finite_float(
                raw_value,
                name=f"metric {name}",
            )
        object.__setattr__(self, "metrics", MappingProxyType(normalized))


@dataclass(frozen=True)
class TrajectoryMetricRow:
    """State-equal metrics for one trajectory."""

    role: str
    trajectory_id: str
    state_count: int
    metrics: Mapping[str, float]

    def metric(self, name: str) -> float:
        try:
            return self.metrics[name]
        except KeyError as error:
            raise KeyError(f"trajectory metric is absent: {name}") from error


@dataclass(frozen=True)
class WinTieLoss:
    wins: int
    ties: int
    losses: int

    @property
    def count(self) -> int:
        return self.wins + self.ties + self.losses


@dataclass(frozen=True)
class BootstrapInterval:
    confidence: float
    lower: float
    upper: float
    method: str = LINEAR_PERCENTILE_INTERVAL


@dataclass(frozen=True)
class PairedSplitSummary:
    role: str
    state_count: int
    trajectory_count: int
    left_mean: float
    right_mean: float
    mean_difference: float
    left_to_right_ratio_of_means: float | None
    win_tie_loss: WinTieLoss
    mean_difference_interval: BootstrapInterval


@dataclass(frozen=True)
class PairedBootstrapReport:
    left_metric: str
    right_metric: str
    resamples: int
    seed: int
    confidence: float
    train: PairedSplitSummary
    development: PairedSplitSummary
    overall_stratified: PairedSplitSummary


@dataclass(frozen=True)
class InteractionGeometry:
    interaction_count: int
    normalized_mean_absolute_interaction: float
    redundancy_share: float | None
    marginal_count: int
    negative_marginal_count: int
    negative_marginal_rate: float
    has_negative_marginal: bool


StateMetricInput = StateMetricRow | Mapping[str, Any]


def _coerce_state_metric(value: StateMetricInput) -> StateMetricRow:
    if isinstance(value, StateMetricRow):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("state metrics must be StateMetricRow objects or mappings")
    required = {"role", "trajectory_id", "state_id", "metrics"}
    if set(value) != required:
        raise ValueError(
            "state metric mappings must contain exactly role, trajectory_id, "
            "state_id, and metrics"
        )
    return StateMetricRow(
        role=value["role"],
        trajectory_id=value["trajectory_id"],
        state_id=value["state_id"],
        metrics=value["metrics"],
    )


def _metric_names(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("metric_names must be a sequence of names")
    names = tuple(_nonempty_string(value, name="metric name") for value in values)
    if not names:
        raise ValueError("metric_names cannot be empty")
    if len(set(names)) != len(names):
        raise ValueError("metric_names cannot contain duplicates")
    return names


def aggregate_trajectory_first(
    states: Sequence[StateMetricInput],
    metric_names: Sequence[str],
) -> tuple[TrajectoryMetricRow, ...]:
    """Average states within each trajectory without weighting by row count."""
    names = _metric_names(metric_names)
    normalized = tuple(_coerce_state_metric(value) for value in states)
    if not normalized:
        raise ValueError("states cannot be empty")

    grouped: dict[tuple[str, str], list[StateMetricRow]] = {}
    seen_states: set[tuple[str, str, str]] = set()
    for state in normalized:
        identity = (state.role, state.trajectory_id, state.state_id)
        if identity in seen_states:
            raise ValueError(f"duplicate state identity: {identity}")
        seen_states.add(identity)
        missing = [name for name in names if name not in state.metrics]
        if missing:
            raise ValueError(
                f"state {state.state_id} lacks requested metrics: {missing}"
            )
        grouped.setdefault((state.role, state.trajectory_id), []).append(state)

    result: list[TrajectoryMetricRow] = []
    for (role, trajectory_id), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: row.state_id)
        metrics = {
            name: _mean(tuple(row.metrics[name] for row in ordered))
            for name in names
        }
        result.append(
            TrajectoryMetricRow(
                role=role,
                trajectory_id=trajectory_id,
                state_count=len(ordered),
                metrics=MappingProxyType(metrics),
            )
        )
    return tuple(result)


def linear_quantile(values: Sequence[Real], probability: float) -> float:
    """Return the deterministic type-7/linear empirical quantile."""
    probability = _finite_float(probability, name="probability")
    if probability < 0.0 or probability > 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(
        _finite_float(value, name="quantile value") for value in values
    )
    if not ordered:
        raise ValueError("quantile values cannot be empty")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def ratio_of_means(
    numerators: Sequence[Real],
    denominators: Sequence[Real],
    *,
    denominator_epsilon: float = DEFAULT_NORMALIZATION_EPSILON,
) -> float | None:
    """Compute mean(numerator) / mean(denominator), never mean of ratios."""
    epsilon = _positive_finite(
        denominator_epsilon,
        name="denominator_epsilon",
    )
    left = tuple(_finite_float(value, name="numerator") for value in numerators)
    right = tuple(
        _finite_float(value, name="denominator") for value in denominators
    )
    if not left or not right or len(left) != len(right):
        raise ValueError("numerators and denominators must have equal non-zero length")
    denominator_mean = _mean(right)
    if abs(denominator_mean) <= epsilon:
        return None
    return _mean(left) / denominator_mean


def win_tie_loss(
    left_values: Sequence[Real],
    right_values: Sequence[Real],
    *,
    tie_epsilon: float = DEFAULT_TIE_EPSILON,
) -> WinTieLoss:
    """Count paired outcomes; callers should pass trajectory-level values."""
    epsilon = _positive_finite(tie_epsilon, name="tie_epsilon")
    left = tuple(_finite_float(value, name="left value") for value in left_values)
    right = tuple(
        _finite_float(value, name="right value") for value in right_values
    )
    if not left or len(left) != len(right):
        raise ValueError("paired values must have equal non-zero length")
    differences = tuple(
        left_value - right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    return WinTieLoss(
        wins=sum(value > epsilon for value in differences),
        ties=sum(abs(value) <= epsilon for value in differences),
        losses=sum(value < -epsilon for value in differences),
    )


def _summary_without_interval(
    trajectories: Sequence[TrajectoryMetricRow],
    *,
    role: str,
    left_metric: str,
    right_metric: str,
    tie_epsilon: float,
) -> tuple[float, float, float, float | None, WinTieLoss]:
    if not trajectories:
        raise ValueError(f"role has no trajectories: {role}")
    left = tuple(row.metric(left_metric) for row in trajectories)
    right = tuple(row.metric(right_metric) for row in trajectories)
    left_mean = _mean(left)
    right_mean = _mean(right)
    return (
        left_mean,
        right_mean,
        left_mean - right_mean,
        ratio_of_means(left, right),
        win_tie_loss(left, right, tie_epsilon=tie_epsilon),
    )


def _resampled_difference(
    rows: Sequence[TrajectoryMetricRow],
    indices: Sequence[int],
    *,
    left_metric: str,
    right_metric: str,
) -> float:
    return _mean(
        tuple(
            rows[index].metric(left_metric) - rows[index].metric(right_metric)
            for index in indices
        )
    )


def paired_trajectory_bootstrap(
    states: Sequence[StateMetricInput],
    left_metric: str,
    right_metric: str,
    *,
    train_role: str = "v2_label_train",
    development_role: str = "v2_development",
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
    tie_epsilon: float = DEFAULT_TIE_EPSILON,
) -> PairedBootstrapReport:
    """Run paired split-specific and overall stratified cluster bootstrap."""
    left_metric = _nonempty_string(left_metric, name="left_metric")
    right_metric = _nonempty_string(right_metric, name="right_metric")
    if left_metric == right_metric:
        raise ValueError("left_metric and right_metric must differ")
    train_role = _nonempty_string(train_role, name="train_role")
    development_role = _nonempty_string(
        development_role,
        name="development_role",
    )
    if train_role == development_role:
        raise ValueError("train_role and development_role must differ")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    confidence = _finite_float(confidence, name="confidence")
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    tie_epsilon = _positive_finite(tie_epsilon, name="tie_epsilon")

    trajectories = aggregate_trajectory_first(
        states,
        (left_metric, right_metric),
    )
    unexpected = sorted(
        {row.role for row in trajectories} - {train_role, development_role}
    )
    if unexpected:
        raise ValueError(f"bootstrap input contains unexpected roles: {unexpected}")
    train = tuple(row for row in trajectories if row.role == train_role)
    development = tuple(
        row for row in trajectories if row.role == development_role
    )
    if not train or not development:
        raise ValueError("both train and development trajectories are required")

    rng = random.Random(seed)
    train_samples: list[float] = []
    development_samples: list[float] = []
    overall_samples: list[float] = []
    total_trajectories = len(train) + len(development)
    for _ in range(resamples):
        train_indices = tuple(rng.randrange(len(train)) for _ in train)
        development_indices = tuple(
            rng.randrange(len(development)) for _ in development
        )
        train_difference = _resampled_difference(
            train,
            train_indices,
            left_metric=left_metric,
            right_metric=right_metric,
        )
        development_difference = _resampled_difference(
            development,
            development_indices,
            left_metric=left_metric,
            right_metric=right_metric,
        )
        train_samples.append(train_difference)
        development_samples.append(development_difference)
        overall_samples.append(
            (
                train_difference * len(train)
                + development_difference * len(development)
            )
            / total_trajectories
        )

    tail = (1.0 - confidence) / 2.0

    def interval(samples: Sequence[float]) -> BootstrapInterval:
        return BootstrapInterval(
            confidence=confidence,
            lower=linear_quantile(samples, tail),
            upper=linear_quantile(samples, 1.0 - tail),
        )

    def summarize(
        rows: Sequence[TrajectoryMetricRow],
        *,
        role: str,
        samples: Sequence[float],
    ) -> PairedSplitSummary:
        left_mean, right_mean, difference, ratio, outcomes = (
            _summary_without_interval(
                rows,
                role=role,
                left_metric=left_metric,
                right_metric=right_metric,
                tie_epsilon=tie_epsilon,
            )
        )
        return PairedSplitSummary(
            role=role,
            state_count=sum(row.state_count for row in rows),
            trajectory_count=len(rows),
            left_mean=left_mean,
            right_mean=right_mean,
            mean_difference=difference,
            left_to_right_ratio_of_means=ratio,
            win_tie_loss=outcomes,
            mean_difference_interval=interval(samples),
        )

    overall = tuple(sorted(train + development, key=lambda row: (row.role, row.trajectory_id)))
    return PairedBootstrapReport(
        left_metric=left_metric,
        right_metric=right_metric,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        train=summarize(train, role=train_role, samples=train_samples),
        development=summarize(
            development,
            role=development_role,
            samples=development_samples,
        ),
        overall_stratified=summarize(
            overall,
            role="overall_stratified",
            samples=overall_samples,
        ),
    )


def interaction_geometry(
    interactions: Sequence[Real],
    marginal_gains: Sequence[Real],
    *,
    summary_only_distance: Real,
    normalization_epsilon: float = DEFAULT_NORMALIZATION_EPSILON,
    sign_epsilon: float = DEFAULT_TIE_EPSILON,
) -> InteractionGeometry:
    """Normalize interaction magnitude and summarize redundancy/non-monotonicity."""
    baseline = _finite_float(
        summary_only_distance,
        name="summary_only_distance",
    )
    if baseline < 0.0:
        raise ValueError("summary_only_distance cannot be negative")
    normalization_epsilon = _positive_finite(
        normalization_epsilon,
        name="normalization_epsilon",
    )
    sign_epsilon = _positive_finite(sign_epsilon, name="sign_epsilon")
    interaction_values = tuple(
        _finite_float(value, name="interaction") for value in interactions
    )
    marginal_values = tuple(
        _finite_float(value, name="marginal gain") for value in marginal_gains
    )

    absolute_mass = math.fsum(abs(value) for value in interaction_values)
    negative_mass = math.fsum(
        -value for value in interaction_values if value < 0.0
    )
    normalized_mean = (
        _mean(tuple(abs(value) for value in interaction_values))
        / max(baseline, normalization_epsilon)
        if interaction_values
        else 0.0
    )
    negative_count = sum(value < -sign_epsilon for value in marginal_values)
    marginal_count = len(marginal_values)
    return InteractionGeometry(
        interaction_count=len(interaction_values),
        normalized_mean_absolute_interaction=normalized_mean,
        redundancy_share=(
            negative_mass / absolute_mass
            if absolute_mass > normalization_epsilon
            else None
        ),
        marginal_count=marginal_count,
        negative_marginal_count=negative_count,
        negative_marginal_rate=(
            negative_count / marginal_count if marginal_count else 0.0
        ),
        has_negative_marginal=negative_count > 0,
    )
