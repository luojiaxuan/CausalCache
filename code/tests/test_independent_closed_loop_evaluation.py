from __future__ import annotations

import math
import random

import pytest

from causalcache.independent_closed_loop_evaluation import (
    BOOTSTRAP_CONFIDENCE,
    INDEPENDENT_ARM,
    RECENT_ARM,
    evaluate_closed_loop_partition,
    evaluate_matched_nll_partition,
    template_cluster_bootstrap_lower,
    type7_quantile,
)


def _templates(partition: str) -> tuple[str, ...]:
    count = 60 if partition == "train" else 25
    return tuple(f"Task{index:02d}" for index in range(count))


def _indices(partition: str) -> tuple[int, ...]:
    return (0,) if partition == "train" else (0, 1, 2)


def _episodes(
    partition: str,
    *,
    independent_success: float = 1.0,
    recent_success: float = 0.0,
    state_count: int | None = None,
) -> list[dict]:
    rows = []
    for task_type in _templates(partition):
        for task_index in _indices(partition):
            for arm, success in (
                (INDEPENDENT_ARM, independent_success),
                (RECENT_ARM, recent_success),
            ):
                row = {
                    "partition": partition,
                    "task_type": task_type,
                    "task_index": task_index,
                    "arm": arm,
                    "official_terminal_success": success,
                }
                if state_count is not None:
                    row["policy_decision_state_count"] = state_count
                rows.append(row)
    return rows


def _states(
    partition: str,
    episodes: list[dict],
    *,
    nll_independent: float = 1.0,
    nll_recent: float = 1.03,
    restoration_independent: float = 2.0,
    restoration_recent: float = 1.0,
) -> list[dict]:
    rows = []
    for episode in episodes:
        for decision_index in range(episode["policy_decision_state_count"]):
            rows.append(
                {
                    "partition": partition,
                    "task_type": episode["task_type"],
                    "task_index": episode["task_index"],
                    "origin_arm": episode["arm"],
                    "decision_index": decision_index,
                    "reference_exact_agreement": True,
                    "nll_kl_finite": True,
                    "selection_budget_audit_complete": True,
                    "nll_independent": nll_independent,
                    "nll_recent": nll_recent,
                    "restoration_independent": restoration_independent,
                    "restoration_recent": restoration_recent,
                }
            )
    return rows


def test_closed_loop_test_averages_three_pairs_before_equal_templates() -> None:
    episodes = _episodes("test", independent_success=0.0, recent_success=0.0)
    for row in episodes:
        if row["arm"] == INDEPENDENT_ARM and row["task_index"] in {0, 2}:
            row["official_terminal_success"] = 1.0
        if row["arm"] == RECENT_ARM and row["task_index"] == 2:
            row["official_terminal_success"] = 1.0

    report = evaluate_closed_loop_partition(
        episodes,
        partition="test",
        template_order=_templates("test"),
        bootstrap_resamples=50,
    )

    assert report["template_count"] == 25
    assert report["instance_indices"] == [0, 1, 2]
    assert report["paired_instance_count"] == 75
    assert report["primary_arm_episode_count"] == 150
    assert report["paired_effect"] == pytest.approx(1.0 / 3.0)
    assert all(
        item["paired_effect"] == pytest.approx(1.0 / 3.0)
        for item in report["template_records"]
    )
    assert report["bootstrap"]["unit"] == "template_cluster"
    assert report["bootstrap"]["lower"] == pytest.approx(1.0 / 3.0)


def test_closed_loop_train_has_sixty_pairs_and_never_drops_failures() -> None:
    episodes = _episodes("train")
    episodes[0]["official_terminal_success"] = 0.0
    report = evaluate_closed_loop_partition(
        episodes,
        partition="train",
        template_order=_templates("train"),
        bootstrap_resamples=20,
    )

    assert report["paired_instance_count"] == 60
    assert report["primary_arm_episode_count"] == 120
    assert len(report["instance_records"]) == 60
    assert report["paired_effect"] == pytest.approx(59.0 / 60.0)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected"])
def test_closed_loop_fails_closed_on_unit_inventory_drift(mutation: str) -> None:
    episodes = _episodes("test")
    if mutation == "missing":
        episodes.pop()
    elif mutation == "duplicate":
        episodes.append(dict(episodes[0]))
    else:
        episodes[0]["task_index"] = 9

    with pytest.raises(ValueError, match="episode unit"):
        evaluate_closed_loop_partition(
            episodes,
            partition="test",
            template_order=_templates("test"),
            bootstrap_resamples=5,
        )


def test_template_bootstrap_uses_exact_stdlib_draw_stream_and_type7() -> None:
    values = (0.0, 1.0, 3.0)
    resamples = 11
    seed = 271_828
    generator = random.Random(seed)
    means = [
        math.fsum(values[generator.randrange(3)] for _ in range(3)) / 3
        for _ in range(resamples)
    ]
    expected = type7_quantile(means, (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0)

    assert template_cluster_bootstrap_lower(
        values,
        resamples=resamples,
        seed=seed,
    ) == expected


def test_matched_nll_test_hierarchy_and_primary_claim() -> None:
    episodes = _episodes("test", state_count=2)
    states = _states("test", episodes)
    report = evaluate_matched_nll_partition(
        episodes,
        states,
        partition="test",
        template_order=_templates("test"),
        bootstrap_resamples=50,
    )

    assert report["role"] == "primary_mechanism_evidence"
    assert report["mechanism_eligible_template_count"] == 25
    assert report["primary_caliper"]["matched_template_count"] == 25
    assert report["primary_caliper"]["mean_z"] == 1.0
    assert report["primary_caliper"]["bootstrap_lower"] == 1.0
    assert report["primary_claim_supported"] is True
    first = report["template_records"][0]
    assert first["instance_pair_count"] == 3
    assert first["nll_independent"] == 1.0
    assert first["nll_recent"] == 1.03
    assert first["restoration_difference"] == 1.0
    assert first["success_difference"] == 1.0
    assert first["z"] == 1.0


def test_matched_nll_requires_all_three_test_instances_eligible() -> None:
    episodes = _episodes("test", state_count=1)
    states = _states("test", episodes)
    failed = next(
        row
        for row in states
        if row["task_type"] == "Task00"
        and row["task_index"] == 2
        and row["origin_arm"] == RECENT_ARM
    )
    failed["reference_exact_agreement"] = False
    for field in (
        "nll_independent",
        "nll_recent",
        "restoration_independent",
        "restoration_recent",
    ):
        failed[field] = None

    report = evaluate_matched_nll_partition(
        episodes,
        states,
        partition="test",
        template_order=_templates("test"),
        bootstrap_resamples=20,
    )

    assert report["mechanism_eligible_template_count"] == 24
    first = report["template_records"][0]
    assert first["mechanism_eligible"] is False
    assert first["ineligible_instance_indices"] == [2]
    assert first["nll_independent"] is None
    assert "Task00" not in report["primary_caliper"]["matched_task_types"]


def test_caliper_is_applied_after_three_instance_template_aggregation() -> None:
    episodes = _episodes("test", state_count=1)
    states = _states("test", episodes, nll_independent=1.0, nll_recent=1.0)
    differences = (0.10, 0.10, -0.10)
    for row in states:
        if row["task_type"] == "Task00":
            row["nll_recent"] = 1.0 + differences[row["task_index"]]

    report = evaluate_matched_nll_partition(
        episodes,
        states,
        partition="test",
        template_order=_templates("test"),
        bootstrap_resamples=20,
    )

    first = report["template_records"][0]
    assert abs(first["nll_independent"] - first["nll_recent"]) == pytest.approx(
        1.0 / 30.0
    )
    assert "Task00" in report["primary_caliper"]["matched_task_types"]
    assert "Task00" not in report["sensitivity_calipers"][0]["matched_task_types"]


def test_train_matched_nll_is_diagnostic_even_when_statistic_passes() -> None:
    episodes = _episodes("train", state_count=1)
    states = _states("train", episodes)
    report = evaluate_matched_nll_partition(
        episodes,
        states,
        partition="train",
        template_order=_templates("train"),
        bootstrap_resamples=20,
    )

    assert report["role"] == "development_diagnostic_only"
    assert report["primary_caliper"]["matched_template_count"] == 60
    assert report["primary_caliper"]["bootstrap_lower"] == 1.0
    assert report["primary_claim_supported"] is False


def test_restoration_tie_contributes_zero() -> None:
    episodes = _episodes("test", state_count=1)
    states = _states(
        "test",
        episodes,
        restoration_independent=1.0 + 5e-13,
        restoration_recent=1.0,
    )
    report = evaluate_matched_nll_partition(
        episodes,
        states,
        partition="test",
        template_order=_templates("test"),
        bootstrap_resamples=20,
    )

    assert report["primary_caliper"]["mean_z"] == 0.0
    assert report["primary_caliper"]["bootstrap_lower"] == 0.0
    assert report["primary_claim_supported"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected"])
def test_matched_nll_fails_closed_on_state_inventory_drift(mutation: str) -> None:
    episodes = _episodes("test", state_count=1)
    states = _states("test", episodes)
    if mutation == "missing":
        states.pop()
    elif mutation == "duplicate":
        states.append(dict(states[0]))
    else:
        states[0]["decision_index"] = 7

    with pytest.raises(ValueError, match="matched-NLL state"):
        evaluate_matched_nll_partition(
            episodes,
            states,
            partition="test",
            template_order=_templates("test"),
            bootstrap_resamples=5,
        )


def test_zero_state_episode_is_recorded_as_mechanism_ineligible() -> None:
    episodes = _episodes("test", state_count=1)
    target = next(
        row
        for row in episodes
        if row["task_type"] == "Task00"
        and row["task_index"] == 0
        and row["arm"] == INDEPENDENT_ARM
    )
    target["policy_decision_state_count"] = 0
    states = _states("test", episodes)

    report = evaluate_matched_nll_partition(
        episodes,
        states,
        partition="test",
        template_order=_templates("test"),
        bootstrap_resamples=5,
    )

    instance = report["instance_records"][0]
    assert instance["mechanism_eligible"] is False
    assert instance["ineligible_reasons"] == [
        f"zero_policy_decision_states:{INDEPENDENT_ARM}"
    ]
    assert report["template_records"][0]["mechanism_eligible"] is False
