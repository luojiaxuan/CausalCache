from __future__ import annotations

import math

import pytest

from causalcache.set_utility_baselines import (
    ocr_rgb_scores,
    ocr_rgb_selection,
    oracle_independent_j_scores,
    oracle_independent_j_selection,
    recent_selection,
    strictly_positive_top_b,
)
from causalcache.set_utility_features import (
    EVENT_NUMERIC_FEATURE_NAMES,
    HASH_DIMENSION,
    SetUtilityFeatureState,
)
from causalcache.set_utility_label_table import (
    validate_cardinality_capped_distance_table,
)


def _feature_state() -> SetUtilityFeatureState:
    def row(ocr: float, rgb: float) -> tuple[float, ...]:
        explicit = [0.0] * len(EVENT_NUMERIC_FEATURE_NAMES)
        explicit[EVENT_NUMERIC_FEATURE_NAMES.index("current_ocr_jaccard")] = ocr
        explicit[EVENT_NUMERIC_FEATURE_NAMES.index("current_rgb_histogram_cosine")] = rgb
        return (0.0,) * HASH_DIMENSION + tuple(explicit)

    zero_pair = (0.0,) * 8
    return SetUtilityFeatureState(
        source_id="trajectory-a",
        state_id="state-a",
        decision_step_id=4,
        event_step_ids=(1, 2, 3),
        query_features=(0.0,) * 64,
        context_features=(0.0,) * 4,
        event_features=(row(0.9, 0.7), row(0.2, 0.4), row(0.6, 0.6)),
        pair_features=tuple(
            tuple(zero_pair for _ in range(3)) for _ in range(3)
        ),
        event_mask=(True, True, True),
    )


def test_ocr_rgb_is_equal_weight_and_budget_only_changes_selection() -> None:
    state = _feature_state()
    assert ocr_rgb_scores(state) == pytest.approx({1: 0.8, 2: 0.3, 3: 0.6})
    assert ocr_rgb_selection(state, budget=1) == (1,)
    assert ocr_rgb_selection(state, budget=2) == (1, 3)
    assert ocr_rgb_selection(state, budget=4) == (1, 2, 3)
    assert recent_selection(state, budget=2) == (2, 3)


def test_oracle_independent_j_uses_singleton_and_conditional_pair_gains() -> None:
    table = validate_cardinality_capped_distance_table(
        split="evaluation",
        state_id="state-j",
        candidate_event_step_ids=(1, 2, 3),
        maximum_labeled_cardinality=2,
        distances={
            (): 10.0,
            (1,): 6.0,
            (2,): 8.0,
            (3,): 9.0,
            (1, 2): 5.0,
            (1, 3): 4.0,
            (2, 3): 7.0,
        },
    )
    # note (luojiaxuan): the three rows hand-check the frozen J projection.
    # j1: 0.5 * ((10-6) + mean(8-5, 9-4)) = 4.0
    # j2: 0.5 * ((10-8) + mean(6-5, 9-7)) = 1.75
    # j3: 0.5 * ((10-9) + mean(6-4, 8-7)) = 1.25
    assert oracle_independent_j_scores(table) == pytest.approx(
        {1: 4.0, 2: 1.75, 3: 1.25}
    )
    assert oracle_independent_j_selection(table, budget=2) == (1, 2)


def test_at_most_b_rule_keeps_empty_and_has_deterministic_ties() -> None:
    assert strictly_positive_top_b({1: -1.0, 2: 0.0}, budget=2) == ()
    assert strictly_positive_top_b({3: 1.0, 1: 1.0, 2: 0.5}, budget=2) == (1, 3)
    with pytest.raises(ValueError, match="finite"):
        strictly_positive_top_b({1: math.nan}, budget=1)
