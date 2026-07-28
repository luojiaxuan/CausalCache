from __future__ import annotations

from scripts.reduce_mobileworld_budget_curve import (
    _paired_comparison,
    _wilson_interval,
)


def test_wilson_interval_contains_observed_rate() -> None:
    interval = _wilson_interval(34, 117)
    assert interval[0] < 34 / 117 < interval[1]
    assert 0.0 <= interval[0] < interval[1] <= 1.0


def test_paired_comparison_reports_right_minus_left() -> None:
    scores = {
        "B0": {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0},
        "B1": {"a": 1.0, "b": 0.0, "c": 1.0, "d": 1.0},
    }
    result = _paired_comparison(
        "B0",
        "B1",
        ["a", "b", "c", "d"],
        scores,
        iterations=100,
        seed=7,
    )
    assert result["left_successes"] == 2
    assert result["right_successes"] == 3
    assert result["right_minus_left"] == 0.25
    assert result["discordant"] == {"left_only": 1, "right_only": 2}
    assert result["ties"] == 1
