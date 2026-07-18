"""Budget-separated baselines for cardinality-capped set utility."""

from __future__ import annotations

import math
from collections.abc import Mapping

from causalcache.set_utility_features import (
    SetUtilityFeatureState,
    event_ocr_rgb_score,
)
from causalcache.set_utility_label_table import ValidatedSetUtilityTable


def strictly_positive_top_b(
    scores: Mapping[int, float],
    *,
    budget: int,
) -> tuple[int, ...]:
    """Apply the frozen at-most-B rule; budget never changes an event score."""
    if type(budget) is not int or budget < 0:
        raise ValueError("baseline budget must be a non-negative integer")
    if not isinstance(scores, Mapping):
        raise TypeError("baseline scores must be an event-indexed mapping")
    canonical: list[tuple[int, float]] = []
    for event_id, raw_score in scores.items():
        if type(event_id) is not int or event_id <= 0:
            raise ValueError("baseline event ids must be positive integers")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("baseline scores must be finite")
        canonical.append((event_id, score))
    if len({event_id for event_id, _ in canonical}) != len(canonical):
        raise ValueError("baseline event ids must be unique")
    selected = sorted(
        (item for item in canonical if item[1] > 0.0),
        key=lambda item: (-item[1], item[0]),
    )[:budget]
    return tuple(sorted(event_id for event_id, _ in selected))


def ocr_rgb_scores(state: SetUtilityFeatureState) -> dict[int, float]:
    """Recover the frozen equal-weight current-state OCR/RGB event scores."""
    if not isinstance(state, SetUtilityFeatureState):
        raise TypeError("OCR/RGB baseline requires a set-utility feature state")
    return {
        event_id: event_ocr_rgb_score(state.event_features[index])
        for index, event_id in enumerate(state.event_step_ids)
    }


def recent_selection(
    state: SetUtilityFeatureState,
    *,
    budget: int,
) -> tuple[int, ...]:
    """Select the most recent chronological events without reading any label."""
    if not isinstance(state, SetUtilityFeatureState):
        raise TypeError("recent baseline requires a set-utility feature state")
    if type(budget) is not int or budget < 0:
        raise ValueError("baseline budget must be a non-negative integer")
    return tuple(sorted(state.event_step_ids)[-budget:]) if budget else ()


def ocr_rgb_selection(
    state: SetUtilityFeatureState,
    *,
    budget: int,
) -> tuple[int, ...]:
    return strictly_positive_top_b(ocr_rgb_scores(state), budget=budget)


def oracle_independent_j_scores(
    table: ValidatedSetUtilityTable,
) -> dict[int, float]:
    """Project exact singleton/pair distances to the frozen independent J target."""
    if not isinstance(table, ValidatedSetUtilityTable):
        raise TypeError("oracle-independent J requires a validated distance table")
    if table.maximum_labeled_cardinality < 2:
        raise ValueError("oracle-independent J requires exact singleton and pair labels")
    event_ids = table.candidate_event_step_ids
    empty_distance = table.distance(())
    scores: dict[int, float] = {}
    for event_id in event_ids:
        singleton_gain = empty_distance - table.distance((event_id,))
        conditional = [
            table.distance((other,))
            - table.distance(tuple(sorted((event_id, other))))
            for other in event_ids
            if other != event_id
        ]
        conditional_mean = (
            math.fsum(conditional) / len(conditional)
            if conditional
            else singleton_gain
        )
        scores[event_id] = 0.5 * (singleton_gain + conditional_mean)
    return scores


def oracle_independent_j_selection(
    table: ValidatedSetUtilityTable,
    *,
    budget: int,
) -> tuple[int, ...]:
    return strictly_positive_top_b(
        oracle_independent_j_scores(table),
        budget=budget,
    )


__all__ = [
    "ocr_rgb_scores",
    "ocr_rgb_selection",
    "oracle_independent_j_scores",
    "oracle_independent_j_selection",
    "recent_selection",
    "strictly_positive_top_b",
]
