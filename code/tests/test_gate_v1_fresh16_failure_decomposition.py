from __future__ import annotations

import builtins
import copy
import hashlib
import itertools
import json
import math
import socket
from types import SimpleNamespace

import pytest

import causalcache.gate_v1_fresh16_failure_decomposition as decomposition
from causalcache.gate_v1_data import (
    CandidateFeatures,
    GateState,
    LabelState,
    build_training_batch,
)
from causalcache.gate_v1_fresh16_failure_decomposition import (
    DECISION_THRESHOLDS,
    budget_conditioned_oracle_independent_scores,
    build_failure_decomposition,
    failure_decomposition_json_bytes,
)
from causalcache.restoration_v2_2_label_table import (
    primary_exact_subset_oracle,
    validate_complete_distance_table,
)


def _contract():
    return SimpleNamespace(
        sha256="a" * 64,
        data={
            "routing_contract": copy.deepcopy(DECISION_THRESHOLDS),
        },
    )


def _table(source_index: int, event_count: int):
    event_ids = tuple(range(1, event_count + 1))
    baseline = 20.0 + source_index + event_count / 10.0
    weights = {1: 5.0, 2: 4.0, 3: 2.0, 4: 1.0}
    pair_bonus = 0.01 * (source_index + 1) * event_count
    distances = {}
    for size in range(event_count + 1):
        for coalition in itertools.combinations(event_ids, size):
            utility = sum(weights[event] for event in coalition)
            utility += pair_bonus * math.comb(len(coalition), 2)
            distances[coalition] = baseline - utility
    return validate_complete_distance_table(event_ids, distances)


def _metric(table, selected):
    selected = tuple(sorted(selected))
    utility = table.utility(selected)
    baseline = table.distance(())
    return {
        "selected": list(selected),
        "raw_utility": utility,
        "normalized_recovery": utility / baseline,
    }


def _trace_for_order(table, order):
    selected = ()
    result = []
    for round_index, chosen in enumerate(order):
        remaining = tuple(event for event in table.event_ids if event not in selected)
        true_scores = {
            event: table.distance(selected)
            - table.distance(tuple(sorted((*selected, event))))
            for event in remaining
        }
        predicted = {
            event: true_scores[event] + (100.0 if event == chosen else 0.0)
            for event in remaining
        }
        after = tuple(sorted((*selected, chosen)))
        result.append(
            {
                "round": round_index,
                "selected_before": list(selected),
                "candidate_predicted_marginal_gains": [
                    {
                        "event_step_id": event,
                        "predicted_marginal_gain": predicted[event],
                    }
                    for event in remaining
                ],
                "decision": "add",
                "selected_event_step_id": chosen,
                "selected_after": list(after),
            }
        )
        selected = after
    return result


def _fixture():
    labels = []
    records = []
    for source_index in range(16):
        source_id = f"source-{source_index:02d}"
        for event_count in (2, 3, 4):
            decision_step = event_count + 2
            state_id = f"{source_id}:decision_step:{decision_step:03d}"
            table = _table(source_index, event_count)
            label = LabelState(
                source_id=source_id,
                state_id=state_id,
                decision_step_id=decision_step,
                table=table,
            )
            exact = primary_exact_subset_oracle(table).coalition
            trace = _trace_for_order(table, exact)
            evaluation = {
                "source_id": source_id,
                "event_count": event_count,
                "baseline": table.distance(()),
                "exact": _metric(table, exact),
                "conditional": _metric(table, exact),
                "independent": _metric(table, exact),
                "seed_conditional": [_metric(table, exact) for _ in range(5)],
                "seed_independent": [_metric(table, exact) for _ in range(5)],
                "heuristics": {
                    name: _metric(table, exact)
                    for name in (
                        "dynamic_recent",
                        "ocr_rgb_v2",
                        "policy_vision_v3",
                    )
                },
                "conditional_selected_addition_count": len(exact),
                "conditional_true_nonpositive_addition_count": 0,
            }
            labels.append(label)
            records.append(
                {
                    "schema_version": "1.0.0",
                    "protocol_id": "parent",
                    "status": "parent-state-record",
                    "ordinal": len(records),
                    "source_id": source_id,
                    "state_id": state_id,
                    "decision_step_id": decision_step,
                    "candidate_event_step_ids": list(table.event_ids),
                    "conditional_selection_trace": trace,
                    "evaluation": evaluation,
                }
            )
    return tuple(labels), records


@pytest.fixture(scope="module")
def valid_fixture():
    return _fixture()


@pytest.fixture(scope="module")
def valid_payload(valid_fixture):
    labels, records = valid_fixture
    return build_failure_decomposition(labels, records, contract=_contract())


def _canonical_jsonl(rows):
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for row in rows
    )


def test_full_reducer_is_deterministic_and_trajectory_equal(valid_fixture, valid_payload):
    labels, records = valid_fixture
    replay = build_failure_decomposition(
        labels,
        {row["state_id"]: row for row in records},
        contract=_contract(),
    )
    assert failure_decomposition_json_bytes(replay) == failure_decomposition_json_bytes(
        valid_payload
    )
    assert valid_payload["source_count"] == 16
    assert valid_payload["state_count"] == 48
    assert len(valid_payload["state_rows"]) == 48
    assert len(valid_payload["trajectory_rows"]) == 16
    assert valid_payload["state_rows_sha256"] == hashlib.sha256(
        _canonical_jsonl(valid_payload["state_rows"])
    ).hexdigest()
    assert valid_payload["trajectory_rows_sha256"] == hashlib.sha256(
        _canonical_jsonl(valid_payload["trajectory_rows"])
    ).hexdigest()
    assert failure_decomposition_json_bytes(valid_payload).endswith(b"\n")

    methods = valid_payload["overall"]["methods"]
    for method in ("exact", "true_greedy", "oracle_independent", "conditional", "independent"):
        assert methods[method]["raw_ratio_to_exact"] == pytest.approx(1.0)
        assert methods[method]["normalized_ratio_to_exact"] == pytest.approx(1.0)
    assert valid_payload["overall"]["gaps"]["raw"]["additive_identity_valid"]
    assert valid_payload["heuristics"]["strongest"] == "dynamic_recent"
    decision = valid_payload["decision_tree"]
    assert decision["search"]["pass"] is True
    assert decision["oracle_set_conditioning_headroom"]["pass"] is False
    assert decision["outcome"] == "NO_V2_CONDITIONAL_RESCUE"
    assert decision["failed_at"] == "oracle_set_headroom_pass"


def test_rank_strata_and_zero_operation_boundary(valid_payload):
    strata = valid_payload["strata"]
    for event_count in (2, 3, 4):
        assert strata["by_n"][str(event_count)]["state_count"] == 16
        baseline = strata["baseline_rank_within_n"][str(event_count)]
        assert baseline["bottom_quartile"]["state_count"] == 4
        assert baseline["upper_three_quartiles"]["state_count"] == 12
        interaction = strata["interaction_rank_tertile_within_n"][str(event_count)]
        assert interaction["low"]["state_count"] == 6
        assert interaction["mid"]["state_count"] == 5
        assert interaction["high"]["state_count"] == 5
    assert strata["denominator"]["regular"]["state_count"] == 48
    for name in ("normalization_ineligible", "tiny", "small"):
        assert strata["denominator"][name]["state_count"] == 0

    operations = valid_payload["operation_counts"]
    assert operations["sealed_fresh16_label_state_decode_count"] == 48
    assert operations["sealed_parent_state_record_decode_count"] == 48
    for name in (
        "file_read",
        "file_write",
        "network",
        "torch_import",
        "gpu",
        "model_load",
        "model_forward",
        "policy_forward",
        "fresh16_raw_source_access_count",
        "upstream_raw_label_read_count",
        "legacy_dev5_semantic_decode_count",
        "confirm20_access_count",
        "matched_nll_evaluation_count",
        "closed_loop_episode_count",
        "gate_training_step_count",
        "hf_mutation",
    ):
        assert operations[name] == 0


def test_prefix_split_uses_trace_first_choice_not_sorted_final_set(valid_fixture):
    labels, original = valid_fixture
    records = copy.deepcopy(original)
    target = next(
        row
        for row in records
        if row["source_id"] == "source-00"
        and len(row["candidate_event_step_ids"]) == 4
    )
    label = next(item for item in labels if item.state_id == target["state_id"])
    target["evaluation"]["conditional"] = _metric(label.table, (3, 4))
    target["evaluation"]["conditional_selected_addition_count"] = 2
    target["conditional_selection_trace"] = _trace_for_order(label.table, (4, 3))

    payload = build_failure_decomposition(labels, records, contract=_contract())
    state = next(row for row in payload["state_rows"] if row["state_id"] == label.state_id)
    gaps = state["gap_decomposition"]
    assert state["methods"]["conditional"]["selected"] == [3, 4]
    assert gaps["first_selected_event_step_id"] == 4
    assert gaps["prefix_lock_gap_raw"] == pytest.approx(3.0)
    assert gaps["completion_stop_gap_raw"] == pytest.approx(3.0)
    assert gaps["prefix_lock_gap_raw"] + gaps["completion_stop_gap_raw"] == pytest.approx(
        gaps["total_exact_minus_conditional_raw"]
    )
    assert state["rounds"][0]["student_error_type"] == "wrong_event"
    assert (
        state["rounds"][1]["student_error_type"]
        == "unreached_after_prior_divergence"
    )


def test_first_round_stop_is_all_completion_regret(valid_fixture):
    labels, original = valid_fixture
    records = copy.deepcopy(original)
    target = next(
        row
        for row in records
        if row["source_id"] == "source-00"
        and len(row["candidate_event_step_ids"]) == 4
    )
    label = next(item for item in labels if item.state_id == target["state_id"])
    target["evaluation"]["conditional"] = _metric(label.table, ())
    target["evaluation"]["conditional_selected_addition_count"] = 0
    target["conditional_selection_trace"] = [
        {
            "round": 0,
            "selected_before": [],
            "candidate_predicted_marginal_gains": [
                {"event_step_id": event, "predicted_marginal_gain": -float(event)}
                for event in label.table.event_ids
            ],
            "decision": "stop",
            "selected_event_step_id": None,
            "selected_after": [],
        }
    ]
    payload = build_failure_decomposition(labels, records, contract=_contract())
    state = next(row for row in payload["state_rows"] if row["state_id"] == label.state_id)
    gaps = state["gap_decomposition"]
    assert gaps["first_selected_event_step_id"] is None
    assert gaps["prefix_lock_gap_raw"] == 0.0
    assert gaps["completion_stop_gap_raw"] == pytest.approx(
        gaps["total_exact_minus_conditional_raw"]
    )
    assert state["rounds"][0]["false_stop"] is True
    assert state["rounds"][0]["student_error_type"] == "premature_stop"


def test_oracle_independent_target_exactly_matches_training_batch(valid_fixture):
    label = valid_fixture[0][-1]
    state = GateState(
        source_id=label.source_id,
        state_id=label.state_id,
        decision_step_id=label.decision_step_id,
        candidate_event_step_ids=label.table.event_ids,
        q64=(0.0,) * 64,
        candidates=tuple(
            CandidateFeatures(
                event_step_id=event,
                h64=(0.0,) * 64,
                g8=(0.0,) * 8,
            )
            for event in label.table.event_ids
        ),
        table=label.table,
    )
    expected = budget_conditioned_oracle_independent_scores(label.table)
    batch = build_training_batch((state,), family="independent")
    observed = {example.event_step_id: example.raw_target for example in batch.examples}
    assert observed == expected


def test_true_greedy_and_independent_oracle_require_strict_positive_gain():
    table = validate_complete_distance_table(
        (1, 2),
        {
            (): 1.0,
            (1,): 2.0,
            (2,): 3.0,
            (1, 2): 4.0,
        },
    )
    assert decomposition._strict_positive_true_greedy(table)[0] == ()
    assert decomposition._oracle_independent_selection(table) == ()


def test_reducer_fails_closed_on_inventory_metric_trace_and_threshold_drift(valid_fixture):
    labels, original = valid_fixture
    with pytest.raises(ValueError, match="inventories differ"):
        build_failure_decomposition(labels, original[:-1], contract=_contract())

    records = copy.deepcopy(original)
    records[0]["evaluation"]["conditional"]["raw_utility"] += 1.0
    with pytest.raises(ValueError, match="does not replay"):
        build_failure_decomposition(labels, records, contract=_contract())

    records = copy.deepcopy(original)
    records[0]["conditional_selection_trace"][0]["selected_event_step_id"] = 2
    with pytest.raises(ValueError, match="positive argmax"):
        build_failure_decomposition(labels, records, contract=_contract())

    contract = _contract()
    contract.data["routing_contract"]["search_pass"][
        "raw_utility_G_over_E_minimum"
    ] = 0.94
    with pytest.raises(ValueError, match="thresholds drifted"):
        build_failure_decomposition(labels, original, contract=contract)


def test_reducer_does_not_open_files_or_sockets(monkeypatch, valid_fixture):
    labels, records = valid_fixture

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pure reducer crossed an I/O boundary")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    payload = build_failure_decomposition(labels, records, contract=_contract())
    assert payload["operation_counts"]["file_read"] == 0
    assert payload["operation_counts"]["network"] == 0
