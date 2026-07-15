"""Pure-CPU metrics for the preregistered restoration diagnostic."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any


Coalition = frozenset[int]

_ALLOWED_OUTCOME_STATUSES = frozenset(
    {
        "INVALID",
        "NO_GO_DIAGNOSTIC",
        "INCONCLUSIVE_NEGATIVE",
        "INCONCLUSIVE_POSITIVE",
    }
)
_POSITIVE_THRESHOLD_PREFIX = (
    "at_least_one_state_with_budget_1024_normalized_oracle_recovery_at_least_"
)


@dataclass(frozen=True)
class CoalitionMetric:
    """Observed distance and recovery for one deterministic coalition."""

    coalition: tuple[int, ...]
    cost: int
    distance: float
    utility: float
    normalized_recovery: float


@dataclass(frozen=True)
class RandomExpectation:
    """Uniform expectation over budget-maximal feasible coalitions."""

    coalitions: tuple[tuple[int, ...], ...]
    expected_cost: float
    expected_distance: float
    expected_utility: float
    expected_normalized_recovery: float


@dataclass(frozen=True)
class ExactCoalitionDiagnostic:
    """Selections and metrics derived from a complete feasible distance table."""

    event_ids: tuple[int, ...]
    budget: int
    epsilon: float
    summary_distance: float
    recent: CoalitionMetric
    similarity: CoalitionMetric
    oracle: CoalitionMetric
    random: RandomExpectation


@dataclass(frozen=True)
class DiagnosticOutcomeRow:
    """The frozen state-level fields consumed by the v1 outcome rule."""

    decision_step_id: int
    memory_sensitive: bool
    budget_1024_oracle_utility: float
    epsilon: float
    budget_1024_normalized_oracle_recovery: float
    positive_non_recent_exact_gain: bool


@dataclass(frozen=True)
class DiagnosticOutcome:
    """Auditable result of applying the preregistered v1 outcome rule."""

    status: str
    reasons: tuple[str, ...]
    evaluated_step_ids: tuple[int, ...]
    positive_step_ids: tuple[int, ...]


def extract_first_json_object(text: str) -> dict[str, Any]:
    """Return the first decodable JSON object embedded in ``text``."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("text does not contain a JSON object")


def canonical_json_action(text: str) -> str:
    """Extract the first JSON object and serialize it as sorted compact JSON."""

    return json.dumps(
        extract_first_json_object(text),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _rgb_pixels(image_or_pixels: Any) -> Iterable[tuple[int, int, int]]:
    if hasattr(image_or_pixels, "convert") and hasattr(image_or_pixels, "getdata"):
        pixels = image_or_pixels.convert("RGB").getdata()
    else:
        pixels = image_or_pixels
    try:
        iterator = iter(pixels)
    except TypeError as error:
        raise TypeError("image must be image-like or an iterable of RGB pixels") from error
    for pixel in iterator:
        if isinstance(pixel, (str, bytes)):
            raise ValueError("each pixel must contain exactly three RGB channels")
        try:
            channels = tuple(pixel)
        except TypeError as error:
            raise ValueError("each pixel must contain exactly three RGB channels") from error
        if len(channels) != 3:
            raise ValueError("each pixel must contain exactly three RGB channels")
        if any(isinstance(channel, bool) or not isinstance(channel, int) for channel in channels):
            raise ValueError("RGB channels must be integers")
        if any(channel < 0 or channel > 255 for channel in channels):
            raise ValueError("RGB channels must be between 0 and 255")
        yield channels


def _joint_rgb_histogram(image_or_pixels: Any, bins_per_channel: int) -> dict[int, int]:
    if isinstance(bins_per_channel, bool) or not isinstance(bins_per_channel, int):
        raise TypeError("bins_per_channel must be an integer")
    if bins_per_channel <= 0 or bins_per_channel > 256:
        raise ValueError("bins_per_channel must be between 1 and 256")
    histogram: dict[int, int] = {}
    pixel_count = 0
    for red, green, blue in _rgb_pixels(image_or_pixels):
        red_bin = min(red * bins_per_channel // 256, bins_per_channel - 1)
        green_bin = min(green * bins_per_channel // 256, bins_per_channel - 1)
        blue_bin = min(blue * bins_per_channel // 256, bins_per_channel - 1)
        index = (red_bin * bins_per_channel + green_bin) * bins_per_channel + blue_bin
        histogram[index] = histogram.get(index, 0) + 1
        pixel_count += 1
    if pixel_count == 0:
        raise ValueError("RGB histogram requires at least one pixel")
    return histogram


def rgb_histogram_cosine(
    left_image_or_pixels: Any,
    right_image_or_pixels: Any,
    *,
    bins_per_channel: int = 16,
) -> float:
    """Compute cosine similarity between sparse joint RGB histograms."""

    left = _joint_rgb_histogram(left_image_or_pixels, bins_per_channel)
    right = _joint_rgb_histogram(right_image_or_pixels, bins_per_channel)
    dot = sum(count * right.get(index, 0) for index, count in left.items())
    left_norm = math.sqrt(sum(count * count for count in left.values()))
    right_norm = math.sqrt(sum(count * count for count in right.values()))
    value = dot / (left_norm * right_norm)
    return min(1.0, max(0.0, value))


def normalized_recovery(*, summary_distance: float, distance: float, epsilon: float) -> float:
    """Normalize recovered distance by ``max(D(empty), epsilon)``."""

    for name, value in (
        ("summary_distance", summary_distance),
        ("distance", distance),
        ("epsilon", epsilon),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if summary_distance < 0 or distance < 0:
        raise ValueError("distances must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return (summary_distance - distance) / max(summary_distance, epsilon)


def _validate_event_costs(event_costs: Mapping[int, int], budget: int) -> tuple[int, ...]:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if not event_costs:
        raise ValueError("event_costs must not be empty")
    event_ids: list[int] = []
    for event_id, cost in event_costs.items():
        if isinstance(event_id, bool) or not isinstance(event_id, int):
            raise TypeError("event ids must be integers")
        if isinstance(cost, bool) or not isinstance(cost, int) or cost <= 0:
            raise ValueError("event costs must be positive integers")
        if cost > budget:
            raise ValueError("every event must fit within the requested budget")
        event_ids.append(event_id)
    return tuple(sorted(event_ids))


def _feasible_coalitions(
    event_ids: Sequence[int], event_costs: Mapping[int, int], budget: int
) -> tuple[Coalition, ...]:
    coalitions: list[Coalition] = []
    for subset_size in range(len(event_ids) + 1):
        for subset in combinations(event_ids, subset_size):
            if sum(event_costs[event_id] for event_id in subset) <= budget:
                coalitions.append(frozenset(subset))
    return tuple(coalitions)


def _normalize_distance_table(
    distance_table: Mapping[Iterable[int], float],
    *,
    event_ids: tuple[int, ...],
    feasible: tuple[Coalition, ...],
) -> dict[Coalition, float]:
    known_ids = frozenset(event_ids)
    normalized: dict[Coalition, float] = {}
    for raw_coalition, raw_distance in distance_table.items():
        try:
            coalition = frozenset(raw_coalition)
        except TypeError as error:
            raise TypeError("distance-table keys must be iterable coalitions") from error
        if not coalition.issubset(known_ids):
            raise ValueError("distance table contains an unknown event id")
        if coalition in normalized:
            raise ValueError("distance table contains duplicate normalized coalitions")
        distance = float(raw_distance)
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("coalition distances must be finite and non-negative")
        normalized[coalition] = distance
    required = frozenset(feasible)
    observed = frozenset(normalized)
    if observed != required:
        missing = sorted((tuple(sorted(value)) for value in required - observed))
        extra = sorted((tuple(sorted(value)) for value in observed - required))
        raise ValueError(f"distance table must cover exactly the feasible coalitions; missing={missing}, extra={extra}")
    return normalized


def _greedy_ranked_coalition(
    ranked_event_ids: Sequence[int], event_costs: Mapping[int, int], budget: int
) -> Coalition:
    selected: set[int] = set()
    used_cost = 0
    for event_id in ranked_event_ids:
        event_cost = event_costs[event_id]
        if used_cost + event_cost <= budget:
            selected.add(event_id)
            used_cost += event_cost
    return frozenset(selected)


def _is_budget_maximal(
    coalition: Coalition, event_ids: Sequence[int], event_costs: Mapping[int, int], budget: int
) -> bool:
    used_cost = sum(event_costs[event_id] for event_id in coalition)
    return all(
        event_id in coalition or used_cost + event_costs[event_id] > budget
        for event_id in event_ids
    )


def _metric(
    coalition: Coalition,
    *,
    distances: Mapping[Coalition, float],
    event_costs: Mapping[int, int],
    summary_distance: float,
    epsilon: float,
) -> CoalitionMetric:
    distance = distances[coalition]
    return CoalitionMetric(
        coalition=tuple(sorted(coalition)),
        cost=sum(event_costs[event_id] for event_id in coalition),
        distance=distance,
        utility=summary_distance - distance,
        normalized_recovery=normalized_recovery(
            summary_distance=summary_distance,
            distance=distance,
            epsilon=epsilon,
        ),
    )


def evaluate_exact_coalition_table(
    distance_table: Mapping[Iterable[int], float],
    event_costs: Mapping[int, int],
    *,
    budget: int,
    similarity_scores: Mapping[int, float],
    epsilon: float,
) -> ExactCoalitionDiagnostic:
    """Evaluate frozen baselines and the global oracle on an exact table.

    Recent ranks larger event ids first. Similarity ranks by descending score
    and then ascending event id. Random is uniform over budget-maximal feasible
    coalitions, which is exact-k selection when all costs are equal.
    """

    event_ids = _validate_event_costs(event_costs, budget)
    if set(similarity_scores) != set(event_ids):
        raise ValueError("similarity_scores and event_costs must have identical event ids")
    normalized_similarity: dict[int, float] = {}
    for event_id, raw_score in similarity_scores.items():
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("similarity scores must be finite")
        normalized_similarity[event_id] = score
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    feasible = _feasible_coalitions(event_ids, event_costs, budget)
    distances = _normalize_distance_table(
        distance_table,
        event_ids=event_ids,
        feasible=feasible,
    )
    summary_distance = distances[frozenset()]

    recent_coalition = _greedy_ranked_coalition(
        tuple(reversed(event_ids)), event_costs, budget
    )
    similarity_order = tuple(
        sorted(event_ids, key=lambda event_id: (-normalized_similarity[event_id], event_id))
    )
    similarity_coalition = _greedy_ranked_coalition(
        similarity_order, event_costs, budget
    )
    oracle_coalition = min(
        feasible,
        key=lambda coalition: (
            distances[coalition],
            sum(event_costs[event_id] for event_id in coalition),
            tuple(sorted(coalition)),
        ),
    )

    random_coalitions = tuple(
        coalition
        for coalition in feasible
        if _is_budget_maximal(coalition, event_ids, event_costs, budget)
    )
    # note (luojiaxuan): Random is an analytic expectation, so this diagnostic
    # has neither Monte Carlo variance nor a hidden seed-dependent result.
    expected_cost = sum(
        sum(event_costs[event_id] for event_id in coalition)
        for coalition in random_coalitions
    ) / len(random_coalitions)
    expected_distance = sum(distances[coalition] for coalition in random_coalitions) / len(
        random_coalitions
    )
    random_expectation = RandomExpectation(
        coalitions=tuple(tuple(sorted(coalition)) for coalition in random_coalitions),
        expected_cost=expected_cost,
        expected_distance=expected_distance,
        expected_utility=summary_distance - expected_distance,
        expected_normalized_recovery=normalized_recovery(
            summary_distance=summary_distance,
            distance=expected_distance,
            epsilon=epsilon,
        ),
    )

    return ExactCoalitionDiagnostic(
        event_ids=event_ids,
        budget=budget,
        epsilon=epsilon,
        summary_distance=summary_distance,
        recent=_metric(
            recent_coalition,
            distances=distances,
            event_costs=event_costs,
            summary_distance=summary_distance,
            epsilon=epsilon,
        ),
        similarity=_metric(
            similarity_coalition,
            distances=distances,
            event_costs=event_costs,
            summary_distance=summary_distance,
            epsilon=epsilon,
        ),
        oracle=_metric(
            oracle_coalition,
            distances=distances,
            event_costs=event_costs,
            summary_distance=summary_distance,
            epsilon=epsilon,
        ),
        random=random_expectation,
    )


def has_positive_non_recent_gain(
    exact_gains: Mapping[int, float],
    *,
    recent_coalition: Iterable[int],
    epsilon: float,
) -> bool:
    """Return whether any event outside recent memory has exact gain above epsilon."""

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    recent = frozenset(recent_coalition)
    for event_id, raw_gain in exact_gains.items():
        gain = float(raw_gain)
        if not math.isfinite(gain):
            raise ValueError("exact gains must be finite")
        if event_id not in recent and gain > epsilon:
            return True
    return False


def _invalid_outcome(*reasons: str) -> DiagnosticOutcome:
    return DiagnosticOutcome(
        status="INVALID",
        reasons=tuple(reasons),
        evaluated_step_ids=(),
        positive_step_ids=(),
    )


def _positive_recovery_threshold(outcome_rule: Mapping[str, Any]) -> float:
    conditions = outcome_rule.get("inconclusive_positive_requires")
    if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)):
        raise ValueError("inconclusive_positive_requires must be a sequence")
    matches = [
        condition[len(_POSITIVE_THRESHOLD_PREFIX) :]
        for condition in conditions
        if isinstance(condition, str) and condition.startswith(_POSITIVE_THRESHOLD_PREFIX)
    ]
    if len(matches) != 1 or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", matches[0]) is None:
        raise ValueError("config must encode exactly one normalized-recovery threshold")
    threshold = float(matches[0])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("normalized-recovery threshold must be finite and non-negative")
    return threshold


def _coerce_outcome_row(value: DiagnosticOutcomeRow | Mapping[str, Any]) -> DiagnosticOutcomeRow:
    if isinstance(value, DiagnosticOutcomeRow):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("outcome rows must be mappings or DiagnosticOutcomeRow values")
    return DiagnosticOutcomeRow(
        decision_step_id=value["decision_step_id"],
        memory_sensitive=value["memory_sensitive"],
        budget_1024_oracle_utility=value["budget_1024_oracle_utility"],
        epsilon=value["epsilon"],
        budget_1024_normalized_oracle_recovery=value[
            "budget_1024_normalized_oracle_recovery"
        ],
        positive_non_recent_exact_gain=value["positive_non_recent_exact_gain"],
    )


def classify_diagnostic_outcome(
    rows: Sequence[DiagnosticOutcomeRow | Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> DiagnosticOutcome:
    """Apply the frozen v1 status rule without post-result threshold choices."""

    try:
        outcome_rule = config["outcome_rule"]
        if not isinstance(outcome_rule, Mapping):
            return _invalid_outcome("config outcome_rule is not an object")
        allowed_statuses = outcome_rule.get("allowed_statuses")
        if not isinstance(allowed_statuses, Sequence) or isinstance(
            allowed_statuses, (str, bytes)
        ):
            return _invalid_outcome("config allowed_statuses is not a sequence")
        if frozenset(allowed_statuses) != _ALLOWED_OUTCOME_STATUSES:
            return _invalid_outcome("config allowed_statuses do not match frozen v1 statuses")
        positive_threshold = _positive_recovery_threshold(outcome_rule)
        coerced_rows = tuple(_coerce_outcome_row(row) for row in rows)
    except (KeyError, TypeError, ValueError) as error:
        return _invalid_outcome(str(error))

    if not coerced_rows:
        return _invalid_outcome("at least one outcome row is required")
    step_ids: list[int] = []
    for row in coerced_rows:
        if isinstance(row.decision_step_id, bool) or not isinstance(row.decision_step_id, int):
            return _invalid_outcome("decision_step_id must be an integer")
        if type(row.memory_sensitive) is not bool:
            return _invalid_outcome("memory_sensitive must be boolean")
        if type(row.positive_non_recent_exact_gain) is not bool:
            return _invalid_outcome("positive_non_recent_exact_gain must be boolean")
        numeric_values = (
            row.budget_1024_oracle_utility,
            row.epsilon,
            row.budget_1024_normalized_oracle_recovery,
        )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric_values):
            return _invalid_outcome("outcome numeric fields must be numbers")
        if any(not math.isfinite(float(value)) for value in numeric_values):
            return _invalid_outcome("outcome numeric fields must be finite")
        if row.epsilon <= 0:
            return _invalid_outcome("epsilon must be positive")
        step_ids.append(row.decision_step_id)
    if len(step_ids) != len(set(step_ids)):
        return _invalid_outcome("decision_step_id values must be unique")

    expected_states = config.get("states")
    if expected_states is not None:
        try:
            expected_step_ids = tuple(
                sorted(int(state["decision_step_id"]) for state in expected_states)
            )
        except (KeyError, TypeError, ValueError) as error:
            return _invalid_outcome(f"invalid config states: {error}")
        if tuple(sorted(step_ids)) != expected_step_ids:
            return _invalid_outcome("outcome rows do not exactly match configured decision states")

    evaluated_step_ids = tuple(sorted(step_ids))
    if all(not row.memory_sensitive for row in coerced_rows):
        return DiagnosticOutcome(
            status="NO_GO_DIAGNOSTIC",
            reasons=("all configured states are memory-insensitive",),
            evaluated_step_ids=evaluated_step_ids,
            positive_step_ids=(),
        )
    if all(row.budget_1024_oracle_utility <= row.epsilon for row in coerced_rows):
        return DiagnosticOutcome(
            status="NO_GO_DIAGNOSTIC",
            reasons=("no configured state has budget-1024 oracle utility above its epsilon",),
            evaluated_step_ids=evaluated_step_ids,
            positive_step_ids=(),
        )

    positive_rows = tuple(
        row
        for row in coerced_rows
        if row.memory_sensitive
        and row.budget_1024_normalized_oracle_recovery >= positive_threshold
        and row.positive_non_recent_exact_gain
    )
    if positive_rows:
        positive_step_ids = tuple(sorted(row.decision_step_id for row in positive_rows))
        return DiagnosticOutcome(
            status="INCONCLUSIVE_POSITIVE",
            reasons=(
                "at least one state is memory-sensitive",
                f"states {positive_step_ids} meet normalized recovery >= {positive_threshold:g} and non-recent gain",
            ),
            evaluated_step_ids=evaluated_step_ids,
            positive_step_ids=positive_step_ids,
        )
    return DiagnosticOutcome(
        status="INCONCLUSIVE_NEGATIVE",
        reasons=("valid diagnostic does not meet the frozen no-go or positive rule",),
        evaluated_step_ids=evaluated_step_ids,
        positive_step_ids=(),
    )
