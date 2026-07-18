from __future__ import annotations

import builtins
import copy
import itertools
import socket

import pytest

from causalcache.gate_v1_data import CandidateFeatures, GateState
from causalcache.independent_confirm_artifact import (
    fixed_report_bytes,
    state_records_jsonl_bytes,
)
from causalcache.independent_confirm_data import CONFIRM_SOURCE_IDS
from causalcache.independent_confirm_evaluation import evaluate_independent_confirm
from causalcache.independent_confirm_failure_decomposition import (
    build_independent_confirm_failure_decomposition,
    oracle_independent_scores,
    oracle_independent_selection,
)
from causalcache.restoration_v2_2_label_table import validate_complete_distance_table


def _table():
    event_ids = (1, 2, 3, 4)
    baseline = 20.0
    utility = {
        (): 0.0,
        (1,): 4.0,
        (2,): 3.5,
        (3,): 0.0,
        (4,): 0.0,
        (1, 2): 5.0,
        (1, 3): 7.0,
        (1, 4): 7.0,
        (2, 3): 6.5,
        (2, 4): 6.5,
        (3, 4): 9.0,
        (1, 2, 3): 12.0,
        (1, 2, 4): 12.0,
        (1, 3, 4): 13.0,
        (2, 3, 4): 13.0,
        (1, 2, 3, 4): 20.0,
    }
    return validate_complete_distance_table(
        event_ids, {coalition: baseline - value for coalition, value in utility.items()}
    )


def _state(index: int, table) -> GateState:
    source_id = CONFIRM_SOURCE_IDS[index]
    return GateState(
        source_id=source_id,
        state_id=f"{source_id}:decision_step:006",
        decision_step_id=6,
        candidate_event_step_ids=(1, 2, 3, 4),
        q64=(0.0,) * 64,
        candidates=tuple(
            CandidateFeatures(
                event_step_id=event,
                h64=(0.0,) * 64,
                g8=(0.0,) * 8,
            )
            for event in (1, 2, 3, 4)
        ),
        table=table,
    )


def _raw_record(index: int, table):
    source_id = CONFIRM_SOURCE_IDS[index]
    rows = []
    for row in table.rows:
        rows.append(
            {
                "coalition": list(row.coalition),
                "coalition_mask": sum(
                    1 << offset
                    for offset, event in enumerate(table.event_ids)
                    if event in row.coalition
                ),
                "distance": row.distance,
                "utility": row.utility,
            }
        )
    return {
        "schema_version": "1.0.0",
        "protocol_id": "causalcache_independent_confirm20_runner_v1",
        "outcome": "VALID_INDEPENDENT_CONFIRM_STATE",
        "failure": None,
        "state": {
            "ordinal": index,
            "source_id": source_id,
            "state_id": f"{source_id}:decision_step:006",
            "decision_step_id": 6,
            "candidate_event_step_ids": [1, 2, 3, 4],
        },
        "distance_rows": rows,
    }


def _fixture_bytes(*, learned_selection=(3,), ocr_selection=(3, 4)):
    table = _table()
    states = tuple(_state(index, table) for index in range(20))

    def selections(value):
        return {state.state_id: value for state in states}

    learned = selections(learned_selection)
    seeds = tuple(learned for _ in range(5))
    heuristics = {
        "dynamic_recent": selections((1, 2)),
        "ocr_rgb_v2": selections(ocr_selection),
        "policy_vision_v3": selections((1, 3)),
    }
    report = evaluate_independent_confirm(
        states,
        independent_ensemble_selections=learned,
        independent_seed_selections=seeds,
        heuristic_selections=heuristics,
        reference_failure_count=0,
    )
    records = [_raw_record(index, table) for index in range(20)]
    return state_records_jsonl_bytes(records), fixed_report_bytes(report), records, report


def test_pure_reducer_replays_case_a_and_student_gap():
    raw_bytes, report_bytes, _, _ = _fixture_bytes()
    result = build_independent_confirm_failure_decomposition(raw_bytes, report_bytes)

    assert result["protocol_id"] == (
        "causalcache_independent_confirm20_failure_decomposition_v1"
    )
    assert result["status"] == (
        "FROZEN_INDEPENDENT_CONFIRM20_ORACLE_INDEPENDENT_DECOMPOSITION_V1"
    )
    methods = result["aggregate"]["methods"]
    assert methods["exact"]["raw_utility_sum"] == pytest.approx(180.0)
    assert methods["oracle_independent"]["raw_utility_sum"] == pytest.approx(100.0)
    assert methods["learned_independent"]["raw_utility_sum"] == pytest.approx(0.0)
    assert methods["ocr_rgb_v2"]["raw_utility_sum"] == pytest.approx(180.0)
    assert methods["oracle_independent"]["raw_ratio_to_exact"] == pytest.approx(
        5.0 / 9.0
    )
    assert methods["oracle_independent"][
        "fixed20_zero_imputed_mean_normalized_recovery"
    ] == pytest.approx(0.25)

    j_ocr = result["comparisons"]["oracle_independent_minus_ocr_rgb_v2"]
    assert j_ocr["raw"]["mean_delta"] == pytest.approx(-4.0)
    assert j_ocr["raw"]["paired_bootstrap_90_interval"] == pytest.approx(
        [-4.0, -4.0]
    )
    assert j_ocr["raw"]["negative_trajectory_count"] == 20
    j_i = result["comparisons"][
        "oracle_independent_minus_learned_independent"
    ]
    assert j_i["raw"]["mean_delta"] == pytest.approx(5.0)
    assert j_i["raw"]["paired_bootstrap_90_interval"] == pytest.approx([5.0, 5.0])
    i_ocr = result["comparisons"]["learned_independent_minus_ocr_rgb_v2"]
    assert i_ocr["raw"]["mean_delta"] == pytest.approx(-9.0)
    decision = result["decision"]
    assert decision["primary_metric"] == "raw_utility_sum"
    assert decision["status"] == "CASE_A_ORACLE_INDEPENDENT_LOSES_TO_OCR_RGB"
    assert decision["case_a_j_minus_ocr_raw_mean_at_most_zero"] is True
    assert decision["oracle_independent_student_gap_supported"] is True
    assert decision["same_objective_student_rescue_supported"] is False
    assert decision["normalized_metrics_can_change_primary_case"] is False
    assert decision["parent_confirm_verdict"] == "NO_GO_INDEPENDENT_CONFIRM"
    assert decision["parent_confirm_verdict_locked"] is True
    assert all(value is False for value in decision["authorizations"].values())
    assert len(result["state_rows"]) == 20
    assert result["state_rows"][0]["methods"]["oracle_independent"]["selected"] == [
        1,
        2,
    ]
    assert all(value == 0 for value in result["operation_counts"].values())


def test_oracle_independent_uses_strict_positive_stop_and_event_tie_break():
    event_ids = (1, 2, 3, 4)
    weights = {1: 2.0, 2: 2.0, 3: 0.0, 4: -1.0}
    table = validate_complete_distance_table(
        event_ids,
        {
            coalition: 20.0 - sum(weights[event] for event in coalition)
            for size in range(5)
            for coalition in itertools.combinations(event_ids, size)
        },
    )
    scores = oracle_independent_scores(table)
    assert scores[1] == pytest.approx(scores[2])
    assert scores[3] == pytest.approx(0.0)
    assert scores[4] < 0.0
    assert oracle_independent_selection(table, budget=1) == (1,)
    assert oracle_independent_selection(table, budget=4) == (1, 2)


def test_raw_routing_requires_every_case_b_condition_and_otherwise_is_inconclusive():
    raw_bytes, case_b_report, _, _ = _fixture_bytes(ocr_selection=(1,))
    case_b = build_independent_confirm_failure_decomposition(
        raw_bytes, case_b_report
    )["decision"]
    assert case_b["status"] == "CASE_B_TEACHER_VALID_STUDENT_DISTILLATION_GAP"
    assert case_b["same_objective_student_rescue_supported"] is True
    assert case_b["case_b_j_minus_ocr_bootstrap_lower_strictly_positive"] is True
    assert case_b[
        "case_b_j_minus_ocr_positive_trajectory_count_at_least_12"
    ] is True
    assert case_b["case_b_i_minus_ocr_raw_mean_strictly_negative"] is True

    _, inconclusive_report, _, _ = _fixture_bytes(
        learned_selection=(1, 2), ocr_selection=(1,)
    )
    inconclusive = build_independent_confirm_failure_decomposition(
        raw_bytes, inconclusive_report
    )["decision"]
    assert inconclusive["status"] == (
        "INCONCLUSIVE_ORACLE_INDEPENDENT_CONFIRM_DECOMPOSITION"
    )
    assert inconclusive["case_b_i_minus_ocr_raw_mean_strictly_negative"] is False
    assert inconclusive["same_objective_student_rescue_supported"] is False


def test_reducer_rejects_fixed_selection_and_raw_utility_drift():
    raw_bytes, report_bytes, records, report = _fixture_bytes()

    changed_report = copy.deepcopy(report)
    changed_report["records"][0]["exact"]["selected"] = [1, 2]
    with pytest.raises(ValueError, match="exact selected coalition"):
        build_independent_confirm_failure_decomposition(
            raw_bytes, fixed_report_bytes(changed_report)
        )

    changed_records = copy.deepcopy(records)
    changed_records[0]["distance_rows"][0]["utility"] = 1.0
    with pytest.raises(ValueError, match="raw coalition utility"):
        build_independent_confirm_failure_decomposition(
            state_records_jsonl_bytes(changed_records), report_bytes
        )


def test_reducer_performs_no_io_or_network(monkeypatch):
    raw_bytes, report_bytes, _, _ = _fixture_bytes()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pure reducer attempted external IO")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    result = build_independent_confirm_failure_decomposition(raw_bytes, report_bytes)
    assert result["inputs"]["state_count"] == 20


def test_reducer_requires_exact_bytes():
    raw_bytes, report_bytes, _, _ = _fixture_bytes()
    with pytest.raises(TypeError, match="exact bytes"):
        build_independent_confirm_failure_decomposition(raw_bytes.decode(), report_bytes)
