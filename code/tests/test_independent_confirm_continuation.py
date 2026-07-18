from __future__ import annotations

import inspect
from types import MappingProxyType, SimpleNamespace

import pytest

import causalcache.independent_confirm_continuation as continuation
from causalcache.gate_v1_data import CandidateFeatures, FeatureState
from causalcache.independent_confirm_artifact import build_label_blind_payload
from causalcache.independent_confirm_artifact import PayloadPublicationReceipt, REPORT_TARGETS
from causalcache.independent_confirm_execution import RestorationPhaseResult
from causalcache.independent_confirm_runner import ConfirmWorkerResult
from causalcache.independent_confirm_data import (
    CONFIRM_SOURCE_IDS,
    Confirm20LabelBlindBundle,
)


def _features() -> tuple[FeatureState, ...]:
    return tuple(
        FeatureState(
            source_id=source_id,
            state_id=f"{source_id}:decision_step:006",
            decision_step_id=6,
            candidate_event_step_ids=(1, 2, 3, 4),
            q64=(0.0,) * 64,
            candidates=tuple(
                CandidateFeatures(
                    event_step_id=event,
                    h64=(float(event),) + (0.0,) * 63,
                    g8=(float(event),) + (0.0,) * 7,
                )
                for event in (1, 2, 3, 4)
            ),
        )
        for source_id in CONFIRM_SOURCE_IDS
    )


def _score_records(scores):
    ranked = sorted((1, 2, 3, 4), key=lambda event: (-scores[event - 1], event))
    return tuple(
        {
            "ordinal": ordinal,
            "source_id": source_id,
            "state_id": f"{source_id}:decision_step:006",
            "decision_step_id": 6,
            "candidate_event_step_ids": [1, 2, 3, 4],
            "scores_by_event_step": [
                {"event_step_id": event, "score": scores[event - 1]}
                for event in (1, 2, 3, 4)
            ],
            "ranked_event_step_ids": ranked,
            "selected_event_step_ids": sorted(ranked[:2]),
        }
        for ordinal, source_id in enumerate(CONFIRM_SOURCE_IDS)
    )


def _score_map(scores):
    return {
        f"{source_id}:decision_step:006": {
            event: scores[event - 1] for event in (1, 2, 3, 4)
        }
        for source_id in CONFIRM_SOURCE_IDS
    }


def _payload_fixture():
    features = _features()
    state_ids = [feature.state_id for feature in features]
    ocr_records = _score_records((0.1, 0.4, 0.3, 0.2))
    vision_records = _score_records((0.8, 0.2, 0.6, 0.1))
    seal = build_label_blind_payload(
        feature_states=features,
        selections={
            "dynamic_recent": {state: (3, 4) for state in state_ids},
            "ocr_rgb_v2": {state: (2, 3) for state in state_ids},
            "policy_vision_v3": {state: (1, 3) for state in state_ids},
        },
        score_records={
            "ocr_rgb_v2": ocr_records,
            "policy_vision_v3": vision_records,
        },
        ensemble_scores=_score_map((0.9, -0.1, 0.4, 0.0)),
        seed_scores=tuple(
            _score_map((0.9 - seed / 10.0, -0.1, 0.4, 0.0))
            for seed in range(5)
        ),
    )
    bundle = Confirm20LabelBlindBundle(
        feature_states=features,
        work_items=(),
        image_requirements=(),
        image_payloads_by_path=MappingProxyType({}),
        dynamic_recent=MappingProxyType({state: (3, 4) for state in state_ids}),
        ocr_rgb_v2=MappingProxyType({state: (2, 3) for state in state_ids}),
        ocr_rgb_score_records=ocr_records,
    )
    contract = SimpleNamespace(
        payload={
            "targets": [
                {
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "size_bytes": item["size_bytes"],
                }
                for item in seal.inventory
            ],
            "payload_inventory_sha256": seal.inventory_sha256,
        }
    )
    return seal, bundle, contract


def test_replays_exact_payload_as_selector_truth_without_model_inputs(monkeypatch) -> None:
    seal, bundle, contract = _payload_fixture()
    monkeypatch.setattr(
        continuation,
        "EXPECTED_PAYLOAD_INVENTORY_SHA256",
        seal.inventory_sha256,
    )
    calls = []
    monkeypatch.setattr(
        continuation,
        "build_confirm20_label_blind_bundle",
        lambda _payloads: calls.append("deterministic-parent-replay") or bundle,
    )

    view = continuation.replay_payload_against_parent(
        contract=contract,
        payloads=object(),
        payload_files=seal.files,
    )

    assert calls == ["deterministic-parent-replay"]
    assert view.independent.selection_sha256
    assert view.heuristics["policy_vision_v3"][bundle.feature_states[0].state_id] == (
        1,
        3,
    )
    parameters = inspect.signature(
        continuation.execute_independent_confirm_continuation
    ).parameters
    assert "ensemble" not in parameters
    assert "independent_scorer" not in parameters
    assert "policy_vision_phase" not in parameters


def test_rejects_any_payload_byte_drift_before_parent_replay(monkeypatch) -> None:
    seal, _bundle, contract = _payload_fixture()
    monkeypatch.setattr(
        continuation,
        "EXPECTED_PAYLOAD_INVENTORY_SHA256",
        seal.inventory_sha256,
    )
    files = dict(seal.files)
    path = next(iter(files))
    files[path] += b"drift"

    with pytest.raises(ValueError, match="payload bytes drifted"):
        continuation.validate_expected_payload_files(contract, files)


def test_continuation_path_only_restores_and_publishes_report(monkeypatch, tmp_path) -> None:
    events = []
    fake_independent = SimpleNamespace(selection_sha256="7" * 64)
    fake_view = SimpleNamespace(
        seal=SimpleNamespace(
            inventory_sha256=continuation.EXPECTED_PAYLOAD_INVENTORY_SHA256,
            files={"sealed": b"payload"},
        ),
        bundle=Confirm20LabelBlindBundle(
            feature_states=(),
            work_items=(),
            image_requirements=(),
            image_payloads_by_path=MappingProxyType({}),
            dynamic_recent=MappingProxyType({}),
            ocr_rgb_v2=MappingProxyType({}),
            ocr_rgb_score_records=(),
        ),
        feature_states=(),
        independent=fake_independent,
        heuristics=MappingProxyType({}),
    )
    monkeypatch.setattr(
        continuation,
        "validate_continuation_run_contract",
        lambda *_args, **_kwargs: MappingProxyType(
            {
                "execution_b_commit": "b" * 40,
                "runtime": {"devices": [], "gpu_uuids": []},
            }
        ),
    )
    monkeypatch.setattr(
        continuation,
        "replay_payload_against_parent",
        lambda **_kwargs: events.append("payload-replay") or fake_view,
    )

    class Output:
        def __init__(self, _path):
            events.append("output-open")

        def persist_label_blind(self, _files, inventory):
            events.append("local-payload-seal")
            return {
                "durable": True,
                "file_count": len(inventory),
                "inventory_sha256": "8" * 64,
                "persistence_id": "fixture",
            }

        def persist_report(self, _files):
            events.append("local-report")

        def write_completion(self, _value):
            events.append("completion")

        def write_failure(self, _value):
            events.append("failure")

    monkeypatch.setattr(continuation, "DurableConfirmOutput", Output)
    monkeypatch.setattr(
        continuation,
        "persist_and_seal_label_blind_payload",
        lambda *_args, **_kwargs: events.append("typed-local-seal")
        or SimpleNamespace(),
    )
    runner_receipt = SimpleNamespace(payload_commit=continuation.EXPECTED_PAYLOAD_COMMIT)
    monkeypatch.setattr(
        continuation,
        "authorize_adopted_payload_commit",
        lambda *_args, **_kwargs: events.append("adopted-runner-receipt")
        or runner_receipt,
    )
    monkeypatch.setattr(continuation, "confirm_worker_assignments", lambda _items: ())
    restoration_result = RestorationPhaseResult(worker_results=(), worker_metadata=())
    monkeypatch.setattr(
        continuation,
        "validate_restoration_phase",
        lambda result, **_kwargs: result,
    )
    monkeypatch.setattr(
        continuation,
        "aggregate_independent_confirm_workers",
        lambda *_args, **_kwargs: {
            "status": "NO_GO_INDEPENDENT_CONFIRM",
            "go": False,
            "execution": {"operation_counts": {}},
        },
    )
    monkeypatch.setattr(continuation, "_ordered_state_records", lambda *_args, **_kwargs: ())
    report_files = {path: b"report" for path in REPORT_TARGETS}
    monkeypatch.setattr(
        continuation,
        "build_report_files",
        lambda **_kwargs: report_files,
    )

    def phase_factory(_bundle):
        events.append("restoration-factory")

        def phase(_assignments, _receipt, _digest):
            events.append("restoration")
            return restoration_result

        return phase

    class Publisher:
        def verify_payload(self, *_args):
            events.append("payload-fresh-verify")
            return True

        def publish_report(self, _files, *, payload_commit):
            events.append("report-publish")
            assert payload_commit == continuation.EXPECTED_PAYLOAD_COMMIT
            return {
                "base_commit": continuation.EXPECTED_BASE_COMMIT,
                "payload_commit": continuation.EXPECTED_PAYLOAD_COMMIT,
                "report_commit": "c" * 40,
                "annotated_tag_object": "d" * 40,
                "created_tag_count": 1,
                "byte_identical_fresh_replay": True,
            }

        def publish_payload(self, *_args, **_kwargs):
            pytest.fail("continuation must never publish payload")

    adopted = PayloadPublicationReceipt(
        base_commit=continuation.EXPECTED_BASE_COMMIT,
        payload_commit=continuation.EXPECTED_PAYLOAD_COMMIT,
        payload_inventory_sha256=continuation.EXPECTED_PAYLOAD_INVENTORY_SHA256,
    )
    result = continuation.execute_independent_confirm_continuation(
        contract=SimpleNamespace(sha256="e" * 64),
        parent_contract=SimpleNamespace(sha256="f" * 64),
        payloads=object(),
        adopted_payload=adopted,
        payload_files={"sealed": b"payload"},
        run_contract={},
        output_dir=tmp_path / "out",
        publisher=Publisher(),
        restoration_phase_factory=phase_factory,
        after_restoration=lambda: events.append("release-tripwire"),
    )

    assert result.completion["confirm_go"] is False
    assert result.completion["payload_publish_call_count"] == 0
    assert result.completion["policy_vision_call_count"] == 0
    assert result.completion["selector_checkpoint_load_count"] == 0
    assert events.index("adopted-runner-receipt") < events.index("restoration")
    assert events.index("restoration") < events.index("release-tripwire")
    assert events.index("release-tripwire") < events.index("report-publish")
    assert "failure" not in events
