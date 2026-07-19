from __future__ import annotations

from causalcache.set_utility_dense import (
    dense_decision_steps,
    dense_legacy_coverage,
    dense_required_event_step_ids,
    legacy_available_event_step_ids,
)


def test_dense_state_count_uses_every_eligible_step() -> None:
    assert dense_decision_steps(decision_count=6) == (6, 7)
    assert dense_decision_steps(decision_count=17) == tuple(range(6, 19))
    assert dense_required_event_step_ids(6) == frozenset({1, 2, 3, 4, 5})
    assert dense_required_event_step_ids(18) == frozenset({13, 14, 15, 16, 17})


def test_legacy_anchor_terminal_images_cover_short_trajectories() -> None:
    total, covered, missing = dense_legacy_coverage(decision_count=29)
    assert (total, covered, missing) == (25, 25, ())
    assert legacy_available_event_step_ids(decision_count=29) == frozenset(range(1, 30))


def test_legacy_anchor_terminal_images_expose_long_trajectory_gap() -> None:
    total, covered, missing = dense_legacy_coverage(decision_count=38)
    assert total == 34
    assert covered == 26
    assert missing == tuple(range(19, 27))
