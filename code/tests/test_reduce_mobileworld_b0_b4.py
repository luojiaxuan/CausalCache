from __future__ import annotations

from scripts.reduce_mobileworld_b0_b4 import (
    _mcnemar_exact,
    _paired_comparison,
)


def test_mobileworld_strict_pairing_counts_missing_as_zero() -> None:
    result = _paired_comparison(
        ["a", "b", "c", "d"],
        {"a": 1.0, "b": 1.0, "c": 0.0},
        {"a": 1.0, "b": 0.0, "c": 1.0},
        missing_as_zero=True,
        iterations=100,
        seed=7,
    )
    assert result["denominator"] == 4
    assert result["b4_successes"] == 2
    assert result["b0_successes"] == 2
    assert result["discordant"] == {"b0_only": 1, "b4_only": 1}
    assert result["ties"] == 2
    assert result["b0_minus_b4"] == 0.0


def test_mobileworld_mcnemar_exact_is_two_sided() -> None:
    assert _mcnemar_exact(0, 0) == 1.0
    assert _mcnemar_exact(1, 1) == 1.0
    assert _mcnemar_exact(0, 5) == 0.0625
