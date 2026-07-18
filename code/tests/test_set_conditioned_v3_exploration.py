from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.run_set_conditioned_v3_exploration as v3_runner

from causalcache.gate_v1_data import CandidateFeatures, FeatureState, GateState
from causalcache.restoration_v2_2_label_table import validate_complete_distance_table
from causalcache.set_conditioned_v3_exploration import (
    METHODS,
    RawStatePrediction,
    OOFTrial,
    bootstrap_summary,
    build_evaluation_report,
    choose_oof_selection,
    claim_fresh_label_access,
    feasible_coalitions_from_ids,
    formal_oracle_interaction_headroom,
    oracle_additive_selection,
    paired_trajectory_bootstrap,
    prediction_artifact_bytes,
    read_prediction_artifact,
    seal_label_blind_outputs,
    selection_metric_summary,
    select_predicted_utility,
    state_prediction_record,
    verify_fresh_label_access_claim,
    verify_label_blind_seal,
)


def _state(
    source: str,
    event_count: int,
    *,
    utility_by_coalition: dict[tuple[int, ...], float] | None = None,
) -> GateState:
    event_ids = tuple(range(1, event_count + 1))
    utilities = {}
    for size in range(event_count + 1):
        for coalition in itertools.combinations(event_ids, size):
            utilities[coalition] = (
                utility_by_coalition[coalition]
                if utility_by_coalition is not None
                else float(sum(coalition))
            )
    baseline = 20.0
    table = validate_complete_distance_table(
        event_ids,
        {coalition: baseline - utility for coalition, utility in utilities.items()},
    )
    decision_step = event_count + 2
    return GateState(
        source_id=source,
        state_id=f"{source}:decision_step:{decision_step:03d}",
        decision_step_id=decision_step,
        candidate_event_step_ids=event_ids,
        q64=(0.0,) * 64,
        candidates=tuple(
            CandidateFeatures(event_step_id=event, h64=(0.0,) * 64, g8=(0.0,) * 8)
            for event in event_ids
        ),
        table=table,
    )


def _feature(state: GateState) -> FeatureState:
    return FeatureState(
        source_id=state.source_id,
        state_id=state.state_id,
        decision_step_id=state.decision_step_id,
        candidate_event_step_ids=state.candidate_event_step_ids,
        q64=state.q64,
        candidates=state.candidates,
    )


def _prediction(
    event_ids: tuple[int, ...],
    *,
    singletons: tuple[float, ...],
    residual: float,
) -> RawStatePrediction:
    return RawStatePrediction(
        singleton_utilities=dict(zip(event_ids, singletons, strict=True)),
        pair_residuals={
            pair: residual for pair in itertools.combinations(event_ids, 2)
        },
    )


def test_feasible_enumeration_and_tie_break_are_exact() -> None:
    coalitions = feasible_coalitions_from_ids((1, 2, 3, 4))
    assert len(coalitions) == 11
    assert coalitions[:5] == ((), (1,), (2,), (3,), (4,))
    assert select_predicted_utility({(): 0.0, (1, 2): 1.0, (2,): 1.0}) == (2,)


def test_oracle_additive_can_differ_from_exact_set_utility() -> None:
    state = _state(
        "source-a",
        2,
        utility_by_coalition={(): 0.0, (1,): 6.0, (2,): 5.0, (1, 2): 4.0},
    )
    assert oracle_additive_selection(state) == (1, 2)
    assert state.table.utility((1,)) > state.table.utility((1, 2))


def test_four_of_five_consensus_accepts_unguarded() -> None:
    state = _state("source-a", 2)
    feature = _feature(state)
    pair = _prediction((1, 2), singletons=(0.1, 0.1), residual=2.0)
    singleton = _prediction((1, 2), singletons=(2.0, -1.0), residual=-2.0)
    record = state_prediction_record(feature, (pair, pair, pair, pair, singleton))
    assert record["v3_safe_selected"] == [1, 2]
    assert record["safe_decision"]["used_pair_candidate"] is True
    assert record["safe_decision"]["used_fallback"] is False
    assert record["safe_decision"]["pair_candidate_vote_count"] == 4
    assert record["safe_decision"]["strictly_positive_margin_count"] == 4
    assert record["safe_decision"]["pair_candidate"] == [1, 2]


def test_identical_pair_and_additive_is_neither_switch_nor_fallback() -> None:
    state = _state("source-a", 2)
    feature = _feature(state)
    prediction = _prediction((1, 2), singletons=(1.0, 0.5), residual=0.0)
    record = state_prediction_record(feature, (prediction,) * 5)
    assert record["v3_safe_selected"] == record["v3_additive_selected"]
    assert record["safe_decision"]["used_pair_candidate"] is False
    assert record["safe_decision"]["used_fallback"] is False


def test_low_consensus_falls_back_to_ensemble_additive() -> None:
    state = _state("source-a", 3)
    feature = _feature(state)
    predictions = (
        _prediction((1, 2, 3), singletons=(0.4, 0.3, 0.2), residual=2.0),
        _prediction((1, 2, 3), singletons=(0.4, 0.3, 0.2), residual=2.0),
        _prediction((1, 2, 3), singletons=(2.0, -1.0, -1.0), residual=-3.0),
        _prediction((1, 2, 3), singletons=(-1.0, 2.0, -1.0), residual=-3.0),
        _prediction((1, 2, 3), singletons=(-1.0, -1.0, 2.0), residual=-3.0),
    )
    record = state_prediction_record(feature, predictions)
    assert record["safe_decision"]["used_pair_candidate"] is False
    assert record["safe_decision"]["pair_candidate_vote_count"] < 4
    assert record["v3_safe_selected"] == record["v3_additive_selected"]


def test_prediction_artifact_is_canonical_and_feature_only() -> None:
    state = _state("source-a", 2)
    feature = _feature(state)
    prediction = _prediction((1, 2), singletons=(0.1, 0.2), residual=0.3)
    payload = prediction_artifact_bytes(
        (feature,),
        ((prediction,) * 5,),
        model_inventory=({"seed": 0},),
    )
    assert b"distance" not in payload
    replay = read_prediction_artifact(payload, features=(feature,))
    assert tuple(replay) == (state.state_id,)
    assert replay[state.state_id]["v3_safe_selected"] == [1, 2]


def test_label_blind_seal_precedes_access_claim(tmp_path: Path) -> None:
    seal = seal_label_blind_outputs(
        tmp_path,
        {"fresh16-predictions.json": b"predictions\n"},
        training_report={"path": "training.json", "sha256": "0" * 64},
        source_a_git_commit="1" * 40,
        execution_b_git_commit="2" * 40,
        runner_freeze_sha256="3" * 64,
        contract_sha256="4" * 64,
    )
    assert seal["fresh_label_access_count"] == 0
    assert verify_label_blind_seal(tmp_path)["confirm_access_count"] == 0
    assert seal["execution_b_git_commit"] == "2" * 40
    assert seal["runner_freeze_sha256"] == "3" * 64
    claim = claim_fresh_label_access(tmp_path)
    assert claim["fresh_label_access_claim_count"] == 1
    assert verify_fresh_label_access_claim(tmp_path)["fresh_label_decode_count"] == 0


def test_train_lineage_failure_precedes_every_input_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"read": 0, "train": 0}
    monkeypatch.setattr(
        v3_runner,
        "load_frozen_set_conditioned_v3_contract",
        lambda *_args, **_kwargs: SimpleNamespace(sha256="1" * 64),
    )
    monkeypatch.setattr(
        v3_runner,
        "validate_execution_b",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad B")),
    )
    monkeypatch.setattr(
        v3_runner,
        "_read",
        lambda _path: calls.__setitem__("read", calls["read"] + 1),
    )
    monkeypatch.setattr(
        v3_runner,
        "train_and_seal",
        lambda **_kwargs: calls.__setitem__("train", calls["train"] + 1),
    )
    args = SimpleNamespace(
        command="train-seal",
        input_dir=tmp_path,
        output_dir=tmp_path / "output",
        repository_root=tmp_path,
        contract="contract.json",
        source_a_git_commit="1" * 40,
        execution_b_git_commit="2" * 40,
        execution_b_freeze=v3_runner.EXECUTION_B_RUNNER_FREEZE_PATH,
    )
    with pytest.raises(ValueError, match="bad B"):
        v3_runner._run(args)
    assert calls == {"read": 0, "train": 0}


def test_evaluate_lineage_or_seal_failure_precedes_claim_and_label_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"claim": 0, "read": 0, "evaluate": 0}
    validation = {
        "execution_b_git_commit": "2" * 40,
        "runner_freeze_sha256": "3" * 64,
        "contract_sha256": "4" * 64,
    }
    monkeypatch.setattr(
        v3_runner,
        "validate_execution_b",
        lambda *_args, **_kwargs: validation,
    )
    monkeypatch.setattr(
        v3_runner,
        "verify_label_blind_seal",
        lambda _path: {
            "source_a_git_commit": "1" * 40,
            "execution_b_git_commit": "2" * 40,
            "runner_freeze_sha256": "f" * 64,
            "contract_sha256": "4" * 64,
        },
    )
    monkeypatch.setattr(
        v3_runner,
        "claim_fresh_label_access",
        lambda _path: calls.__setitem__("claim", calls["claim"] + 1),
    )
    monkeypatch.setattr(
        v3_runner,
        "_read",
        lambda _path: calls.__setitem__("read", calls["read"] + 1),
    )
    monkeypatch.setattr(
        v3_runner,
        "evaluate_sealed_fresh16",
        lambda **_kwargs: calls.__setitem__("evaluate", calls["evaluate"] + 1),
    )
    args = SimpleNamespace(
        command="evaluate",
        input_dir=tmp_path,
        output_dir=tmp_path / "output",
        repository_root=tmp_path,
        contract="contract.json",
        source_a_git_commit="1" * 40,
        execution_b_git_commit="2" * 40,
        execution_b_freeze=v3_runner.EXECUTION_B_RUNNER_FREEZE_PATH,
    )
    with pytest.raises(ValueError, match="seal differs"):
        v3_runner._run(args)
    assert calls == {"claim": 0, "read": 0, "evaluate": 0}


def test_metrics_are_trajectory_equal_and_report_both_scales() -> None:
    states = tuple(_state(source, count) for source in ("a", "b") for count in (2, 3, 4))
    selections = {
        method: {
            state.state_id: (
                state.candidate_event_step_ids[-2:]
                if method in {"exact", "v3_unguarded", "v3_safe"}
                else (state.candidate_event_step_ids[-1],)
            )
            for state in states
        }
        for method in METHODS
    }
    summary = selection_metric_summary(states, selections)
    assert summary["overall"]["trajectory_count"] == 2
    assert summary["overall"]["methods"]["v3_safe"]["raw_ratio_to_exact"] == 1.0
    assert summary["overall"]["methods"]["v3_safe"]["normalized_ratio_to_exact"] == 1.0
    bootstraps = bootstrap_summary(states, selections)
    assert bootstraps["v3_safe_minus_v3_additive"]["raw"]["trajectory_count"] == 2


def test_formal_oracle_interaction_audit_reports_exact_minus_additive() -> None:
    states = tuple(
        _state(source, count) for source in ("a", "b") for count in (2, 3, 4)
    )
    audit = formal_oracle_interaction_headroom(states)
    assert audit["definition"] == "exact_subset_oracle_minus_oracle_singleton_additive"
    assert audit["trajectory_paired_bootstrap"]["raw"]["trajectory_count"] == 2


def test_paired_bootstrap_is_deterministic() -> None:
    first = paired_trajectory_bootstrap({"a": 1.0, "b": -0.5}, resamples=100)
    second = paired_trajectory_bootstrap({"b": -0.5, "a": 1.0}, resamples=100)
    assert first == second


def test_oof_learning_rate_tie_chooses_lower_rate() -> None:
    trials = []
    for learning_rate, score in ((0.0003, 0.8), (0.001, 0.80005)):
        for seed in range(5):
            trials.append(
                OOFTrial(
                    learning_rate=learning_rate,
                    seed=seed,
                    selected_epoch=1,
                    selected_score=score,
                    selected_raw_ratio=score,
                    selected_normalized_ratio=score,
                    epochs_run=1,
                    score_by_epoch=(score,),
                    raw_ratio_by_epoch=(score,),
                    normalized_ratio_by_epoch=(score,),
                    selected_predictions={},
                )
            )
    assert choose_oof_selection(trials).learning_rate == 0.0003


def test_evaluation_report_records_firewall_and_switches() -> None:
    states = tuple(_state("source-a", count) for count in (2, 3, 4))
    records = {}
    historical = {}
    for state in states:
        selected = list(state.candidate_event_step_ids[-2:])
        records[state.state_id] = {
            "v3_additive_selected": selected,
            "v3_unguarded_selected": selected,
            "v3_safe_selected": selected,
            "safe_decision": {
                "used_pair_candidate": True,
                "reason": "pair_candidate_equals_additive",
                "used_fallback": False,
                "pair_candidate": selected,
                "base_candidate": selected,
                "pair_candidate_equals_additive": True,
                "pair_candidate_vote_count": 5,
                "strictly_positive_margin_count": 0,
                "seedwise_pair_minus_base_margins": [0.0] * 5,
                "seedwise_unguarded_argmax": [selected] * 5,
            },
        }
        historical[state.state_id] = (state.candidate_event_step_ids[-1],)
    report = build_evaluation_report(
        states,
        prediction_records=records,
        historical_v1_independent=historical,
        label_blind_seal_sha256="1" * 64,
        label_access_claim_sha256="2" * 64,
    )
    assert report["scope"] == "consumed_development_exploration_not_confirmatory"
    assert report["fresh_label_decode_count"] == 1
    assert report["confirm_access_count"] == 0
    assert report["gpu_operation_count"] == 0
    assert report["development_interpretation"]["value"] in {
        "PROMISING_DEVELOPMENT_SIGNAL_FOR_A_SEPARATELY_FROZEN_FUTURE_STUDY",
        "NO_DEVELOPMENT_EVIDENCE_TO_CONTINUE_SET_CONDITIONING",
    }
    assert len(report["report_sha256"]) == 64


def test_rejects_noncanonical_candidate_ids() -> None:
    with pytest.raises(ValueError, match="candidate ids"):
        feasible_coalitions_from_ids((2, 1))
