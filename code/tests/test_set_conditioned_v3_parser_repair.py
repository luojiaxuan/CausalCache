from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.run_set_conditioned_v3_parser_repair as runner

from causalcache.gate_v1_data import CandidateFeatures, FeatureState
from causalcache.gate_v1_provenance import canonical_selection_sha256
from causalcache.set_conditioned_v3_parser_repair import (
    evaluate_parser_repair,
    load_historical_v1_independent_parser_repair_v1,
    read_parser_repair_report,
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
    changed = copy.deepcopy(contract.data)
    changed["access_accounting"][
        "fresh_label_semantic_decode_attempt_count_total"
    ] = 1
    with pytest.raises(ValueError, match="access accounting"):
        validate_contract_data(changed)


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


def test_cli_has_no_training_entrypoint() -> None:
    with pytest.raises(SystemExit):
        runner._parser().parse_args(["train-seal"])
