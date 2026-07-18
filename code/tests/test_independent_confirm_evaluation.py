from __future__ import annotations

import itertools

import pytest

import causalcache.independent_confirm_evaluation as evaluation
from causalcache.gate_v1_data import CandidateFeatures, GateState
from causalcache.restoration_v2_2_label_table import validate_complete_distance_table


def _state(index: int) -> GateState:
    event_ids = (1, 2, 3, 4)
    baseline = 12.0
    weights = {1: 5.0, 2: 4.0, 3: 2.0, 4: 1.0}
    distances = {
        coalition: baseline - sum(weights[event] for event in coalition)
        for size in range(5)
        for coalition in itertools.combinations(event_ids, size)
    }
    source_id = f"confirm-{index:02d}"
    state_id = f"{source_id}:decision_step:006"
    return GateState(
        source_id=source_id,
        state_id=state_id,
        decision_step_id=6,
        candidate_event_step_ids=event_ids,
        q64=(0.0,) * 64,
        candidates=tuple(
            CandidateFeatures(
                event_step_id=event,
                h64=(0.0,) * 64,
                g8=(0.0,) * 8,
            )
            for event in event_ids
        ),
        table=validate_complete_distance_table(event_ids, distances),
    )


def _fixture():
    states = tuple(_state(index) for index in range(20))

    def selection(selected):
        return {state.state_id: selected for state in states}

    ensemble = selection((1, 2))
    seeds = tuple(
        selection((1, 2) if index < 4 else (1, 3)) for index in range(5)
    )
    heuristics = {
        "dynamic_recent": selection((3, 4)),
        "ocr_rgb_v2": selection((2, 4)),
        "policy_vision_v3": selection((2, 3)),
    }
    return states, ensemble, seeds, heuristics


def test_go_report_uses_aggregate_raw_ratios_and_all_twenty_states():
    states, ensemble, seeds, heuristics = _fixture()
    report = evaluation.evaluate_independent_confirm(
        states,
        independent_ensemble_selections=ensemble,
        independent_seed_selections=seeds,
        heuristic_selections=heuristics,
        reference_failure_count=0,
    )

    assert report["status"] == "GO_TO_PAIRED_CLOSED_LOOP"
    assert report["evaluation_performed"] is True
    assert report["fixed_state_denominator"] == 20
    assert len(report["records"]) == 20
    assert report["reference"] == {
        "expected_count": 20,
        "success_count": 20,
        "failure_count": 0,
    }
    assert report["bootstrap"] == {
        "unit": "trajectory",
        "resamples": 10_000,
        "seed": 271_828,
        "confidence": 0.9,
        "interval": "percentile",
        "quantile": "Hyndman_Fan_type_7",
    }
    metrics = report["metrics"]
    assert metrics["memory_sensitive_state_count"] == 20
    assert metrics["independent_raw_utility_sum"] == pytest.approx(180.0)
    assert metrics["exact_raw_utility_sum"] == pytest.approx(180.0)
    assert metrics["baseline_distance_sum"] == pytest.approx(240.0)
    assert metrics["ensemble_exact_raw_utility_ratio"] == pytest.approx(1.0)
    assert metrics["retained_baseline_mass"] == pytest.approx(0.75)
    assert metrics["normalized_reporting_affects_go"] is False
    assert metrics["normalized_eligible_state_count"] == 20
    assert metrics[
        "fixed20_zero_imputed_independent_mean_normalized_recovery"
    ] == pytest.approx(0.75)
    assert metrics[
        "fixed20_zero_imputed_exact_mean_normalized_recovery"
    ] == pytest.approx(0.75)
    assert metrics["fixed20_normalized_ratio_to_exact"] == pytest.approx(1.0)
    assert metrics[
        "legacy_eligible_only_independent_mean_normalized_recovery"
    ] == pytest.approx(0.75)
    assert metrics[
        "legacy_eligible_only_exact_mean_normalized_recovery"
    ] == pytest.approx(0.75)
    assert metrics["legacy_eligible_only_normalized_ratio_to_exact"] == pytest.approx(
        1.0
    )
    assert metrics["strongest_heuristic"] == "policy_vision_v3"
    assert metrics["strongest_heuristic_positive_trajectory_count"] == 20
    assert metrics["strongest_heuristic_paired_bootstrap_lower"] == pytest.approx(
        3.0
    )
    assert metrics["individual_seed_exact_raw_utility_ratios"] == pytest.approx(
        [1.0, 1.0, 1.0, 1.0, 7.0 / 9.0]
    )
    assert metrics[
        "individual_seed_fixed20_normalized_ratios_to_exact"
    ] == pytest.approx([1.0, 1.0, 1.0, 1.0, 7.0 / 9.0])
    assert metrics[
        "individual_seed_legacy_eligible_only_normalized_ratios_to_exact"
    ] == pytest.approx([1.0, 1.0, 1.0, 1.0, 7.0 / 9.0])
    assert metrics["individual_seed_ratio_at_least_0_75_count"] == 5
    assert metrics["seed_exact_ratio_population_std"] == pytest.approx(
        0.08888888888888889
    )
    assert all(report["gate_checks"].values())
    assert all(type(value) is bool for value in report["gate_checks"].values())
    assert report["go"] is True
    for record in report["records"]:
        assert record["candidate_event_step_ids"] == [1, 2, 3, 4]
        assert record["exact"]["selected"] == [1, 2]
        assert record["independent"]["selected"] == [1, 2]
        assert record["exact"]["normalized_recovery"] == pytest.approx(0.75)
        assert record["independent"]["normalized_recovery"] == pytest.approx(0.75)


def test_population_std_is_over_five_aggregate_seed_exact_ratios():
    states, ensemble, seeds, heuristics = _fixture()
    empty_seed = {state.state_id: () for state in states}
    report = evaluation.evaluate_independent_confirm(
        states,
        independent_ensemble_selections=ensemble,
        independent_seed_selections=(*seeds[:4], empty_seed),
        heuristic_selections=heuristics,
        reference_failure_count=0,
    )

    metrics = report["metrics"]
    assert metrics["individual_seed_exact_raw_utility_ratios"] == pytest.approx(
        [1.0, 1.0, 1.0, 1.0, 0.0]
    )
    assert metrics["individual_seed_ratio_at_least_0_75_count"] == 4
    assert metrics["seed_exact_ratio_population_std"] == pytest.approx(0.4)
    assert report["gate_checks"]["individual_seed_exact_ratio_pass_count"] is True
    assert report["gate_checks"]["seed_exact_ratio_population_std"] is False
    assert report["status"] == "NO_GO_INDEPENDENT_CONFIRM"
    assert report["go"] is False


def test_normalized_descriptives_report_fixed20_and_legacy_eligible_denominators():
    states, ensemble, seeds, heuristics = _fixture()
    first = states[0]
    insensitive = GateState(
        source_id=first.source_id,
        state_id=first.state_id,
        decision_step_id=first.decision_step_id,
        candidate_event_step_ids=first.candidate_event_step_ids,
        q64=first.q64,
        candidates=first.candidates,
        table=validate_complete_distance_table(
            first.candidate_event_step_ids,
            {
                coalition: 0.0
                for size in range(5)
                for coalition in itertools.combinations(
                    first.candidate_event_step_ids, size
                )
            },
        ),
    )
    report = evaluation.evaluate_independent_confirm(
        (insensitive, *states[1:]),
        independent_ensemble_selections=ensemble,
        independent_seed_selections=seeds,
        heuristic_selections=heuristics,
        reference_failure_count=0,
    )

    metrics = report["metrics"]
    assert metrics["normalized_eligible_state_count"] == 19
    assert metrics[
        "fixed20_zero_imputed_independent_mean_normalized_recovery"
    ] == pytest.approx(0.75 * 19 / 20)
    assert metrics[
        "legacy_eligible_only_independent_mean_normalized_recovery"
    ] == pytest.approx(0.75)
    assert metrics["fixed20_normalized_ratio_to_exact"] == pytest.approx(1.0)
    assert metrics["legacy_eligible_only_normalized_ratio_to_exact"] == pytest.approx(
        1.0
    )
    assert report["records"][0]["independent"]["normalized_recovery"] is None


def test_reference_failure_short_circuits_before_state_or_label_access(monkeypatch):
    class ExplodingStates:
        def __iter__(self):
            raise AssertionError("states must not be read after a reference failure")

    def forbidden_oracle(_table):
        raise AssertionError("exact oracle must not run after a reference failure")

    monkeypatch.setattr(evaluation, "primary_exact_subset_oracle", forbidden_oracle)
    report = evaluation.evaluate_independent_confirm(
        ExplodingStates(),
        independent_ensemble_selections=None,
        independent_seed_selections=None,
        heuristic_selections=None,
        reference_failure_count=1,
    )

    assert report["status"] == "NO_GO_REFERENCE_COVERAGE"
    assert report["evaluation_performed"] is False
    assert report["reference"]["success_count"] == 19
    assert report["metrics"] == {}
    assert report["records"] == []
    assert report["gate_checks"] == {"reference_coverage_20_of_20": False}
    assert report["go"] is False


def test_fixed_denominator_and_one_state_per_trajectory_fail_closed():
    states, ensemble, seeds, heuristics = _fixture()
    with pytest.raises(ValueError, match="fixed 20-state denominator"):
        evaluation.evaluate_independent_confirm(
            states[:-1],
            independent_ensemble_selections=ensemble,
            independent_seed_selections=seeds,
            heuristic_selections=heuristics,
            reference_failure_count=0,
        )

    duplicate_source = tuple(
        [states[0], states[0]] + list(states[2:])
    )
    with pytest.raises(ValueError, match="one state per trajectory"):
        evaluation.evaluate_independent_confirm(
            duplicate_source,
            independent_ensemble_selections=ensemble,
            independent_seed_selections=seeds,
            heuristic_selections=heuristics,
            reference_failure_count=0,
        )


@pytest.mark.parametrize("failure_count", [-1, 21, 1.0, True])
def test_reference_failure_count_is_exact(failure_count):
    with pytest.raises(ValueError, match="reference_failure_count"):
        evaluation.evaluate_independent_confirm(
            (),
            independent_ensemble_selections={},
            independent_seed_selections=(),
            heuristic_selections={},
            reference_failure_count=failure_count,
        )
