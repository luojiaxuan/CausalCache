from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.run_set_conditioned_v3_parser_repair as runner
import scripts.run_set_conditioned_v3_parser_repair_v2 as runner_v2

from causalcache.gate_v1_data import CandidateFeatures, FeatureState
from causalcache.gate_v1_provenance import canonical_selection_sha256
from causalcache.set_conditioned_v3_exploration import (
    FRESH_FEATURE_SHA256,
    FRESH_LABEL_SHA256,
    HISTORICAL_INDEPENDENT_SHA256,
)
from causalcache.set_conditioned_v3_parser_repair import (
    evaluate_parser_repair,
    evaluate_parser_repair_v2,
    load_historical_v1_independent_parser_repair_v1,
    load_historical_v1_independent_parser_repair_v2,
    read_parser_repair_report,
    read_parser_repair_v2_report,
)
from causalcache.set_conditioned_v3_parser_repair_contract import (
    CANONICAL_CONFIG_PATH,
    EXACT_HISTORICAL_PROTOCOL_ID,
    FROZEN_CONFIG_SHA256,
    RUNNER_FREEZE_PATH,
    canonical_json_bytes,
    load_frozen_parser_repair_contract,
    parser_repair_execution_b_freeze_payload,
    validate_contract_data,
)
from causalcache.set_conditioned_v3_parser_repair_v2_contract import (
    FROZEN_CONFIG_SHA256 as FROZEN_CONFIG_SHA256_V2,
    load_frozen_parser_repair_v2_contract,
    validate_contract_data as validate_v2_contract_data,
)


ROOT = Path(__file__).resolve().parents[2]


def _features() -> tuple[FeatureState, ...]:
    rows = []
    for index in range(48):
        rows.append(
            FeatureState(
                source_id=f"source-{index // 3:02d}",
                state_id=f"source-{index // 3:02d}:decision_step:{index % 3 + 4:03d}",
                decision_step_id=index % 3 + 4,
                candidate_event_step_ids=(1, 2),
                q64=(0.0,) * 64,
                candidates=(
                    CandidateFeatures(
                        event_step_id=1,
                        h64=(0.0,) * 64,
                        g8=(0.0,) * 8,
                    ),
                    CandidateFeatures(
                        event_step_id=2,
                        h64=(0.0,) * 64,
                        g8=(0.0,) * 8,
                    ),
                ),
            )
        )
    return tuple(rows)


def _historical_payload(
    features: tuple[FeatureState, ...],
    *,
    protocol_id: str = EXACT_HISTORICAL_PROTOCOL_ID,
    sort_records: bool = False,
) -> bytes:
    selections = {feature.state_id: (1, 2) for feature in features}
    records = [
        {
            "state_id": feature.state_id,
            "ensemble_selected_event_step_ids": [1, 2],
            "seed_selected_event_step_ids": [
                {"seed": seed, "selected_event_step_ids": [1, 2]}
                for seed in range(5)
            ],
        }
        for feature in features
    ]
    if sort_records:
        records.sort(key=lambda item: item["state_id"], reverse=True)
    value = {
        "schema_version": "1.0.0",
        "protocol_id": protocol_id,
        "status": "FROZEN_GATE_V1_FRESH16_LEARNED_SELECTIONS_V1",
        "name": "independent",
        "selection_sha256": canonical_selection_sha256(selections),
        "decision_contract": "FEATURE_ONLY_ALL_ENSEMBLE_AND_SEED_DECISIONS_V1",
        "records": records,
    }
    return canonical_json_bytes(value) + b"\n"


def _fake_evaluation_report(
    *_args: object,
    label_blind_seal_sha256: str,
    label_access_claim_sha256: str,
    **_kwargs: object,
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "protocol_id": "parent",
        "status": "parent",
        "scope": "parent",
        "label_blind_seal_sha256": label_blind_seal_sha256,
        "label_access_claim_sha256": label_access_claim_sha256,
        "fresh_feature_sha256": FRESH_FEATURE_SHA256,
        "fresh_label_sha256": FRESH_LABEL_SHA256,
        "historical_v1_independent_sha256": HISTORICAL_INDEPENDENT_SHA256,
        "trajectory_count": 16,
        "state_count": 48,
        "metrics": {},
        "bootstrap": {},
        "switch_diagnostics": {},
        "development_interpretation": {
            "value": "NO_DEVELOPMENT_EVIDENCE_TO_CONTINUE_SET_CONDITIONING",
            "primary_contrast": "v3_safe_minus_v3_additive",
            "mean_normalized_delta": 0.0,
            "mean_raw_delta": 0.0,
            "n4_mean_normalized_delta": 0.0,
            "checks": {},
            "may_authorize_confirm": False,
            "may_change_v1_verdict": False,
        },
        "state_records": [{} for _ in range(48)],
        "fresh_label_access_claim_count": 1,
        "fresh_label_decode_count": 1,
        "confirm_access_count": 0,
        "legacy_development_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
        "raw_gui_access_count": 0,
        "policy_forward_count": 0,
        "gpu_operation_count": 0,
        "report_sha256": "f" * 64,
    }


def test_contract_freezes_exact_parser_only_boundary() -> None:
    contract = load_frozen_parser_repair_contract(repository_root=ROOT)
    assert contract.sha256 == FROZEN_CONFIG_SHA256
    repair = contract.data["parser_repair"]
    assert repair["exact_required_protocol_id"] == EXACT_HISTORICAL_PROTOCOL_ID
    assert repair["protocol_allowlist_forbidden"] is True
    assert repair["checkpoint_change_allowed"] is False
    assert contract.data["access_accounting"][
        "fresh_label_semantic_decode_attempt_count_total"
    ] == 2
    assert contract.data["runtime_contract"]["training_entrypoint_allowed"] is False


def test_contract_rejects_protocol_allowlist_or_decode_rewrite() -> None:
    contract = load_frozen_parser_repair_contract(repository_root=ROOT)
    changed = copy.deepcopy(contract.data)
    changed["parser_repair"]["protocol_allowlist_forbidden"] = False
    with pytest.raises(ValueError, match="historical parser repair"):
        validate_contract_data(changed)


def test_v2_overlay_freezes_exact_state_id_join_and_zero_v1_label_decode() -> None:
    contract = load_frozen_parser_repair_v2_contract(repository_root=ROOT)
    assert contract.sha256 == FROZEN_CONFIG_SHA256_V2
    join = contract.data["state_join_repair"]
    assert join["v2_exact_rule"] == (
        "require_unique_state_ids_and_equal_state_id_sets_then_join_by_state_id"
    )
    assert join["observed_positional_mismatch_count"] == 45
    accounting = contract.data["access_accounting"]
    assert accounting[
        "parser_repair_v1_fresh_label_semantic_decode_attempt_count"
    ] == 0
    assert accounting["fresh_label_semantic_decode_attempt_count_total"] == 2
    assert accounting["historical_parse_attempt_count_total"] == 3
    assert accounting["parser_repair_v2_real_artifact_dry_run_count"] == 0
    changed = copy.deepcopy(contract.data)
    changed["access_accounting"][
        "fresh_label_semantic_decode_attempt_count_total"
    ] = 1
    with pytest.raises(ValueError, match="access accounting"):
        validate_v2_contract_data(changed)


def test_runner_freeze_binds_parent_seal_claim_predictions_and_zero_confirm() -> None:
    contract = load_frozen_parser_repair_contract(repository_root=ROOT)
    value = parser_repair_execution_b_freeze_payload(
        contract,
        source_a_git_commit="1" * 40,
        source_inventory_sha256="2" * 64,
    )
    assert value["runner_freeze_path"] == RUNNER_FREEZE_PATH
    assert value["parent_label_blind_seal_sha256"].startswith("9029f0dc")
    assert value["parent_label_access_claim_sha256"].startswith("7bb9a95b")
    assert value["parent_prediction_sha256"].startswith("23235c59")
    assert value["confirm20_access_count"] == 0
    assert value["gpu_count"] == 0


def test_exact_evaluation_protocol_accepts_and_primary_protocol_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _features()
    payload = _historical_payload(features)
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.HISTORICAL_INDEPENDENT_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    selections = load_historical_v1_independent_parser_repair_v1(
        payload,
        feature_states=features,
    )
    assert len(selections) == 48

    wrong = _historical_payload(
        features,
        protocol_id="causalcache_gate_v1_fresh16_primary_v1",
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.HISTORICAL_INDEPENDENT_SHA256",
        hashlib.sha256(wrong).hexdigest(),
    )
    with pytest.raises(ValueError, match="schema drifted"):
        load_historical_v1_independent_parser_repair_v1(
            wrong,
            feature_states=features,
        )


def test_v2_joins_equal_unique_state_sets_despite_different_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _features()
    payload = _historical_payload(features, sort_records=True)
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.HISTORICAL_INDEPENDENT_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(ValueError, match="state order drifted"):
        load_historical_v1_independent_parser_repair_v1(
            payload,
            feature_states=features,
        )
    selections = load_historical_v1_independent_parser_repair_v2(
        payload,
        feature_states=features,
    )
    assert set(selections) == {feature.state_id for feature in features}

    value = json.loads(payload)
    value["records"][0]["state_id"] = "missing-from-features"
    changed = canonical_json_bytes(value) + b"\n"
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.HISTORICAL_INDEPENDENT_SHA256",
        hashlib.sha256(changed).hexdigest(),
    )
    with pytest.raises(ValueError, match="sets differ"):
        load_historical_v1_independent_parser_repair_v2(
            changed,
            feature_states=features,
        )


@pytest.mark.parametrize("malformed_field", ["seed", "event"])
def test_v2_rejects_boolean_seed_or_event_ids(
    monkeypatch: pytest.MonkeyPatch,
    malformed_field: str,
) -> None:
    features = _features()
    value = json.loads(_historical_payload(features, sort_records=True))
    seed_row = value["records"][0]["seed_selected_event_step_ids"][0]
    if malformed_field == "seed":
        seed_row["seed"] = False
    else:
        seed_row["selected_event_step_ids"] = [True, 2]
    payload = canonical_json_bytes(value) + b"\n"
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.HISTORICAL_INDEPENDENT_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(ValueError, match="seed record drifted"):
        load_historical_v1_independent_parser_repair_v2(
            payload,
            feature_states=features,
        )


def test_repair_report_discloses_two_decodes_and_preserves_zero_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "label-blind-seal.json").write_bytes(b"seal\n")
    (tmp_path / "fresh-label-access-claim.json").write_bytes(b"claim\n")
    (tmp_path / "fresh16-predictions.json").write_bytes(b"predictions\n")
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.verify_label_blind_seal",
        lambda _path: {
            "source_a_git_commit": "1" * 40,
            "execution_b_git_commit": "2" * 40,
            "runner_freeze_sha256": "3" * 64,
        },
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.verify_fresh_label_access_claim",
        lambda _path: {"fresh_label_access_claim_count": 1},
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.load_fresh_features",
        lambda _payload: (),
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.read_prediction_artifact",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.load_historical_v1_independent_parser_repair_v1",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.load_fresh_states",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.build_evaluation_report",
        lambda *_args, **_kwargs: {
            "schema_version": "1.0.0",
            "protocol_id": "parent",
            "status": "parent",
            "scope": "parent",
            "report_sha256": "f" * 64,
            "fresh_label_decode_count": 1,
            "fresh_label_access_claim_count": 1,
            "confirm_access_count": 0,
        },
    )
    accounting = {
        "fresh_label_access_claim_count_total": 1,
        "parent_fresh_label_semantic_decode_attempt_count": 1,
        "repair_fresh_label_semantic_decode_attempt_count": 1,
        "fresh_label_semantic_decode_attempt_count_total": 2,
        "parent_historical_parse_attempt_count": 1,
        "repair_historical_parse_attempt_count": 1,
        "historical_parse_attempt_count_total": 2,
        "training_run_count_total": 1,
        "prediction_generation_count_total": 1,
        "development_report_completion_count_total": 1,
        "confirm20_access_count": 0,
        "legacy_dev5_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
        "policy_forward_count": 0,
        "gpu_operation_count": 0,
    }
    contract = SimpleNamespace(sha256="4" * 64, data={"access_accounting": accounting})
    report = evaluate_parser_repair(
        contract=contract,
        output_dir=tmp_path,
        feature_payload=b"feature",
        label_payload=b"label",
        historical_independent_payload=b"historical",
        repair_source_a_git_commit="5" * 40,
        repair_execution_b_git_commit="6" * 40,
        repair_runner_freeze_sha256="7" * 64,
    )
    assert "fresh_label_decode_count" not in report
    assert report["access_accounting"][
        "fresh_label_semantic_decode_attempt_count_total"
    ] == 2
    assert report["confirm_access_count"] == 0
    assert read_parser_repair_report(tmp_path) == report


def test_v2_report_records_v1_prelabel_failure_and_total_two_decodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "label-blind-seal.json").write_bytes(b"seal\n")
    (tmp_path / "fresh-label-access-claim.json").write_bytes(b"claim\n")
    (tmp_path / "fresh16-predictions.json").write_bytes(b"predictions\n")
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.verify_label_blind_seal",
        lambda _path: {
            "source_a_git_commit": "1" * 40,
            "execution_b_git_commit": "2" * 40,
            "runner_freeze_sha256": "3" * 64,
        },
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.verify_fresh_label_access_claim",
        lambda _path: {"fresh_label_access_claim_count": 1},
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.load_fresh_features",
        lambda _payload: (),
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.read_prediction_artifact",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.load_historical_v1_independent_parser_repair_v2",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.load_fresh_states",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.build_evaluation_report",
        _fake_evaluation_report,
    )
    accounting = {
        "fresh_label_access_claim_count_total": 1,
        "parent_fresh_label_semantic_decode_attempt_count": 1,
        "parser_repair_v1_fresh_label_semantic_decode_attempt_count": 0,
        "parser_repair_v2_fresh_label_semantic_decode_attempt_count": 1,
        "fresh_label_semantic_decode_attempt_count_total": 2,
        "parent_historical_parse_attempt_count": 1,
        "parser_repair_v1_historical_parse_attempt_count": 1,
        "parser_repair_v2_historical_parse_attempt_count": 1,
        "historical_parse_attempt_count_total": 3,
        "parser_repair_v2_real_artifact_dry_run_count": 0,
        "training_run_count_total": 1,
        "prediction_generation_count_total": 1,
        "development_report_completion_count_total": 1,
        "confirm20_access_count": 0,
        "legacy_dev5_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
        "policy_forward_count": 0,
        "gpu_operation_count": 0,
    }
    contract = SimpleNamespace(sha256="4" * 64, data={"access_accounting": accounting})
    report = evaluate_parser_repair_v2(
        contract=contract,
        output_dir=tmp_path,
        feature_payload=b"feature",
        label_payload=b"label",
        historical_independent_payload=b"historical",
        repair_source_a_git_commit="5" * 40,
        repair_execution_b_git_commit="6" * 40,
        repair_runner_freeze_sha256="7" * 64,
    )
    assert report["access_accounting"][
        "parser_repair_v1_fresh_label_semantic_decode_attempt_count"
    ] == 0
    assert report["access_accounting"][
        "fresh_label_semantic_decode_attempt_count_total"
    ] == 2
    read_kwargs = {
        "contract": contract,
        "expected_source_a_git_commit": "5" * 40,
        "expected_execution_b_git_commit": "6" * 40,
        "expected_runner_freeze_sha256": "7" * 64,
    }
    assert read_parser_repair_v2_report(
        tmp_path,
        expected_report_sha256=report["report_sha256"],
        **read_kwargs,
    ) == report

    for field in ("top_authority", "nested_authority", "lineage", "gpu_count"):
        tampered = copy.deepcopy(report)
        if field == "top_authority":
            tampered["may_authorize_confirm"] = True
        elif field == "nested_authority":
            tampered["development_interpretation"]["may_authorize_confirm"] = True
        elif field == "lineage":
            tampered["repair_provenance"]["repair_execution_b_git_commit"] = "bad"
        else:
            tampered["gpu_operation_count"] = 999
        tampered.pop("report_sha256")
        tampered["report_sha256"] = hashlib.sha256(
            canonical_json_bytes(tampered)
        ).hexdigest()
        (tmp_path / "fresh16-development-report-parser-repair-v2.json").write_bytes(
            canonical_json_bytes(tampered) + b"\n"
        )
        with pytest.raises(ValueError, match="schema drifted"):
            read_parser_repair_v2_report(
                tmp_path,
                expected_report_sha256=tampered["report_sha256"],
                **read_kwargs,
            )


def test_v2_historical_failure_precedes_semantic_label_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "label-blind-seal.json").write_bytes(b"seal\n")
    (tmp_path / "fresh-label-access-claim.json").write_bytes(b"claim\n")
    (tmp_path / "fresh16-predictions.json").write_bytes(b"predictions\n")
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.verify_label_blind_seal",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.verify_fresh_label_access_claim",
        lambda _path: {"fresh_label_access_claim_count": 1},
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.load_fresh_features",
        lambda _payload: (),
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.read_prediction_artifact",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.load_historical_v1_independent_parser_repair_v2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad history")),
    )
    label_calls = {"count": 0}
    monkeypatch.setattr(
        "causalcache.set_conditioned_v3_parser_repair.load_fresh_states",
        lambda *_args, **_kwargs: label_calls.__setitem__("count", 1),
    )
    contract = SimpleNamespace(sha256="4" * 64, data={"access_accounting": {}})
    with pytest.raises(ValueError, match="bad history"):
        evaluate_parser_repair_v2(
            contract=contract,
            output_dir=tmp_path,
            feature_payload=b"feature",
            label_payload=b"label",
            historical_independent_payload=b"historical",
            repair_source_a_git_commit="5" * 40,
            repair_execution_b_git_commit="6" * 40,
            repair_runner_freeze_sha256="7" * 64,
        )
    assert label_calls["count"] == 0
    assert not (tmp_path / "fresh16-development-report-parser-repair-v2.json").exists()


def test_lineage_failure_precedes_parent_artifact_or_input_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"parent": 0, "input": 0, "evaluate": 0}
    monkeypatch.setattr(
        runner,
        "load_frozen_parser_repair_contract",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        runner,
        "validate_parser_repair_execution_b",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad repair B")),
    )
    monkeypatch.setattr(
        runner,
        "validate_parent_sealed_artifacts",
        lambda *_args, **_kwargs: calls.__setitem__("parent", calls["parent"] + 1),
    )
    monkeypatch.setattr(
        runner,
        "exact_input_bytes",
        lambda *_args, **_kwargs: calls.__setitem__("input", calls["input"] + 1),
    )
    monkeypatch.setattr(
        runner,
        "evaluate_parser_repair",
        lambda *_args, **_kwargs: calls.__setitem__(
            "evaluate", calls["evaluate"] + 1
        ),
    )
    args = SimpleNamespace(
        command="evaluate",
        repository_root=tmp_path,
        contract=CANONICAL_CONFIG_PATH,
        source_a_git_commit="1" * 40,
        execution_b_git_commit="2" * 40,
        execution_b_freeze=RUNNER_FREEZE_PATH,
        input_dir=tmp_path,
        output_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="bad repair B"):
        runner._run(args)
    assert calls == {"parent": 0, "input": 0, "evaluate": 0}


def test_v2_validate_lineage_failure_precedes_report_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"parent": 0, "report": 0}
    monkeypatch.setattr(
        runner_v2,
        "load_frozen_parser_repair_v2_contract",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        runner_v2,
        "validate_parser_repair_v2_execution_b",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad v2 B")),
    )
    monkeypatch.setattr(
        runner_v2,
        "validate_v2_parent_artifacts",
        lambda *_args, **_kwargs: calls.__setitem__("parent", calls["parent"] + 1),
    )
    monkeypatch.setattr(
        runner_v2,
        "read_parser_repair_v2_report",
        lambda *_args, **_kwargs: calls.__setitem__("report", calls["report"] + 1),
    )
    args = SimpleNamespace(
        command="validate",
        repository_root=tmp_path,
        contract="contract.json",
        source_a_git_commit="1" * 40,
        execution_b_git_commit="2" * 40,
        execution_b_freeze=(
            "code/configs/causalcache_set_conditioned_v3_"
            "historical_protocol_parser_repair_v2_runner.json"
        ),
        expected_report_sha256="3" * 64,
        output_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="bad v2 B"):
        runner_v2._run(args)
    assert calls == {"parent": 0, "report": 0}


def test_cli_has_no_training_entrypoint() -> None:
    with pytest.raises(SystemExit):
        runner._parser().parse_args(["train-seal"])
    with pytest.raises(SystemExit):
        runner_v2._parser().parse_args(["train-seal"])
