from __future__ import annotations

import math
import random

import pytest

from causalcache.exploratory_closed_loop_evaluation import (
    ADVANCE_ROUTE,
    ALL_CONTRASTS,
    ARMS,
    INCONCLUSIVE_ROUTE,
    INDEPENDENT_ARM,
    INVALID_ROUTE,
    OCR_RGB_ARM,
    RECENT_ARM,
    STOP_ROUTE,
    STRATUM_BY_TEMPLATE,
    SUMMARY_ARM,
    TEMPLATE_ORDER,
    exact_two_sided_sign_test,
    evaluate_exploratory_closed_loop,
    paired_bootstrap_interval,
    type7_quantile,
)


def _audit(**updates: int) -> dict[str, int]:
    result = {
        "hidden_retry_count": 0,
        "top_up_count": 0,
        "template_or_arm_deletion_count": 0,
        "memory_binding_decision_count": 3,
        "independent_recent_selector_disagreement_count": 2,
    }
    result.update(updates)
    return result


def _episodes() -> list[dict]:
    return [
        {
            "partition": "validation",
            "task_type": task_type,
            "task_index": 0,
            "horizon_stratum": STRATUM_BY_TEMPLATE[task_type],
            "arm": arm,
            "official_terminal_success": 0.0,
            "action_parse_attempt_count": 2,
            "action_parse_success_count": 2,
            "infrastructure_failure": False,
            "normal_environment_chain": True,
            "required_audits_complete": True,
            "failure_classification": "terminal_failure",
        }
        for task_type in TEMPLATE_ORDER
        for arm in ARMS
    ]


def _row(rows: list[dict], task_type: str, arm: str) -> dict:
    return next(
        item
        for item in rows
        if item["task_type"] == task_type and item["arm"] == arm
    )


def test_advance_route_uses_frozen_overall_and_long_net_win_rules() -> None:
    rows = _episodes()
    short_task = next(
        task for task in TEMPLATE_ORDER if STRATUM_BY_TEMPLATE[task] == "short"
    )
    long_task = next(
        task for task in TEMPLATE_ORDER if STRATUM_BY_TEMPLATE[task] == "long"
    )
    _row(rows, short_task, INDEPENDENT_ARM)["official_terminal_success"] = 1.0
    _row(rows, short_task, INDEPENDENT_ARM)[
        "failure_classification"
    ] = "official_success"
    _row(rows, long_task, INDEPENDENT_ARM)["official_terminal_success"] = 1.0
    _row(rows, long_task, INDEPENDENT_ARM)[
        "failure_classification"
    ] = "official_success"

    report = evaluate_exploratory_closed_loop(
        rows,
        execution_audit=_audit(),
        bootstrap_resamples=30,
    )

    assert report["execution_validity"]["valid"] is True
    assert report["routing"]["route"] == ADVANCE_ROUTE
    overall = report["statistics"]["overall"]
    primary = overall["paired_contrasts"][
        "independent_B2_minus_recent_B2"
    ]
    assert primary["wins"] == 2
    assert primary["ties"] == 10
    assert primary["losses"] == 0
    assert primary["wins_minus_losses"] == 2
    assert primary["paired_mean_difference"] == pytest.approx(2.0 / 12.0)
    long_primary = report["statistics"]["horizon_strata"]["long"][
        "paired_contrasts"
    ]["independent_B2_minus_recent_B2"]
    assert long_primary["wins_minus_losses"] == 1


def test_all_ties_stop_the_current_independent_direction() -> None:
    report = evaluate_exploratory_closed_loop(
        _episodes(),
        execution_audit=_audit(),
        bootstrap_resamples=20,
    )

    assert report["routing"]["route"] == STOP_ROUTE
    assert all(report["routing"]["stop_criteria"].values())
    assert report["statistics"]["overall"]["arm_success"][RECENT_ARM] == {
        "success_count": 0,
        "success_rate": 0.0,
    }


def test_one_short_primary_win_is_inconclusive_not_advance_or_stop() -> None:
    rows = _episodes()
    short_task = next(
        task for task in TEMPLATE_ORDER if STRATUM_BY_TEMPLATE[task] == "short"
    )
    _row(rows, short_task, INDEPENDENT_ARM)["official_terminal_success"] = 1.0
    _row(rows, short_task, INDEPENDENT_ARM)[
        "failure_classification"
    ] = "official_success"

    report = evaluate_exploratory_closed_loop(
        rows,
        execution_audit=_audit(),
        bootstrap_resamples=20,
    )

    assert report["routing"]["route"] == INCONCLUSIVE_ROUTE
    assert (
        report["routing"]["advance_criteria"][
            "independent_vs_recent_overall_wins_minus_losses_at_least_2"
        ]
        is False
    )
    assert (
        report["routing"]["stop_criteria"][
            "independent_vs_recent_overall_not_positive"
        ]
        is False
    )


def test_conditional_success_cannot_rescue_independent_route() -> None:
    rows = _episodes()
    for task_type in TEMPLATE_ORDER:
        _row(rows, task_type, "conditional_B2")["official_terminal_success"] = 1.0
        _row(rows, task_type, "conditional_B2")[
            "failure_classification"
        ] = "official_success"

    report = evaluate_exploratory_closed_loop(
        rows,
        execution_audit=_audit(),
        bootstrap_resamples=10,
    )

    assert report["routing"]["route"] == STOP_ROUTE
    assert (
        report["routing"]["conditional_success_may_rescue_independent_route"]
        is False
    )


def test_reports_each_arm_success_and_every_method_versus_recent_by_stratum() -> None:
    report = evaluate_exploratory_closed_loop(
        _episodes(),
        execution_audit=_audit(),
        bootstrap_resamples=10,
    )

    assert set(report["statistics"]["overall"]["arm_success"]) == set(ARMS)
    expected_contrasts = {
        f"{left}_minus_{right}" for left, right in ALL_CONTRASTS
    }
    assert (
        set(report["statistics"]["overall"]["paired_contrasts"])
        == expected_contrasts
    )
    for stratum in ("short", "medium", "long"):
        region = report["statistics"]["horizon_strata"][stratum]
        assert region["template_count"] == 4
        assert set(region["paired_contrasts"]) == expected_contrasts
    assert report["statistics"]["horizon_definition"] == (
        "frozen_pre_treatment_max_steps"
    )
    assert (
        report["statistics"]["realized_episode_steps_used_for_stratification"]
        is False
    )


def test_missing_episode_is_invalid_and_is_not_imputed_for_statistics() -> None:
    rows = _episodes()
    rows.pop()

    report = evaluate_exploratory_closed_loop(
        rows,
        execution_audit=_audit(),
        bootstrap_resamples=10,
    )

    assert report["routing"]["route"] == INVALID_ROUTE
    assert report["statistics"] is None
    assert report["execution_validity"]["checks"]["exact_episode_record_count"] is False
    assert len(report["execution_validity"]["inventory"]["missing"]) == 1


def test_duplicate_episode_is_invalid_instead_of_changing_the_denominator() -> None:
    rows = _episodes()
    rows.append(dict(rows[0]))

    report = evaluate_exploratory_closed_loop(
        rows,
        execution_audit=_audit(),
        bootstrap_resamples=10,
    )

    assert report["routing"]["route"] == INVALID_ROUTE
    assert report["statistics"] is None
    assert len(report["execution_validity"]["inventory"]["duplicates"]) == 1


@pytest.mark.parametrize(
    "mutation",
    ["parse", "audit", "infrastructure", "normal_chain", "hidden_retry"],
)
def test_execution_validity_gates_override_favorable_outcomes(mutation: str) -> None:
    rows = _episodes()
    long_task = next(
        task for task in TEMPLATE_ORDER if STRATUM_BY_TEMPLATE[task] == "long"
    )
    short_task = next(
        task for task in TEMPLATE_ORDER if STRATUM_BY_TEMPLATE[task] == "short"
    )
    _row(rows, long_task, INDEPENDENT_ARM)["official_terminal_success"] = 1.0
    _row(rows, long_task, INDEPENDENT_ARM)[
        "failure_classification"
    ] = "official_success"
    _row(rows, short_task, INDEPENDENT_ARM)["official_terminal_success"] = 1.0
    _row(rows, short_task, INDEPENDENT_ARM)[
        "failure_classification"
    ] = "official_success"
    audit = _audit()

    if mutation == "parse":
        target = _row(rows, TEMPLATE_ORDER[0], INDEPENDENT_ARM)
        target["action_parse_success_count"] = 0
    elif mutation == "audit":
        _row(rows, TEMPLATE_ORDER[0], INDEPENDENT_ARM)[
            "required_audits_complete"
        ] = False
    elif mutation == "infrastructure":
        for task_type in TEMPLATE_ORDER[:2]:
            target = _row(rows, task_type, INDEPENDENT_ARM)
            target["official_terminal_success"] = 0.0
            target["failure_classification"] = "infrastructure_failure"
            target["infrastructure_failure"] = True
            target["normal_environment_chain"] = False
    elif mutation == "normal_chain":
        for task_type in TEMPLATE_ORDER[:3]:
            _row(rows, task_type, INDEPENDENT_ARM)[
                "normal_environment_chain"
            ] = False
    else:
        audit["hidden_retry_count"] = 1

    report = evaluate_exploratory_closed_loop(
        rows,
        execution_audit=audit,
        bootstrap_resamples=10,
    )

    assert report["execution_validity"]["valid"] is False
    assert report["routing"]["route"] == INVALID_ROUTE
    assert report["statistics"] is not None


def test_horizon_identity_drift_is_invalid_but_outcomes_remain_descriptive() -> None:
    rows = _episodes()
    rows[0]["horizon_stratum"] = "long"

    report = evaluate_exploratory_closed_loop(
        rows,
        execution_audit=_audit(),
        bootstrap_resamples=10,
    )

    assert report["execution_validity"]["valid"] is False
    assert report["routing"]["route"] == INVALID_ROUTE
    assert report["statistics"] is not None
    assert report["execution_validity"]["inventory"]["identity_issues"]


def test_recorded_infrastructure_failure_must_have_itt_zero() -> None:
    rows = _episodes()
    rows[0]["infrastructure_failure"] = True
    rows[0]["normal_environment_chain"] = False
    rows[0]["official_terminal_success"] = 1.0
    rows[0]["failure_classification"] = "infrastructure_failure"

    with pytest.raises(ValueError, match="ITT success zero"):
        evaluate_exploratory_closed_loop(
            rows,
            execution_audit=_audit(),
            bootstrap_resamples=10,
        )


def test_exact_sign_test_and_type7_quantile() -> None:
    assert exact_two_sided_sign_test(3, 1)["p_value"] == pytest.approx(0.625)
    assert exact_two_sided_sign_test(0, 0)["p_value"] == 1.0
    assert type7_quantile((0.0, 10.0), 0.25) == pytest.approx(2.5)


def test_paired_bootstrap_matches_the_frozen_stdlib_draw_stream() -> None:
    effects = (0.0, 1.0, 3.0)
    resamples = 11
    seed = 271_828
    generator = random.Random(seed)
    means = [
        math.fsum(effects[generator.randrange(3)] for _ in range(3)) / 3
        for _ in range(resamples)
    ]
    expected_lower = type7_quantile(means, 0.05)
    expected_upper = type7_quantile(means, 0.95)

    interval = paired_bootstrap_interval(
        effects,
        resamples=resamples,
        seed=seed,
    )

    assert interval["lower"] == pytest.approx(expected_lower)
    assert interval["upper"] == pytest.approx(expected_upper)
    assert interval["role"] == "descriptive_only"
