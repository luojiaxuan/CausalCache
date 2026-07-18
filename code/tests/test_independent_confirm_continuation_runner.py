from __future__ import annotations

import hashlib
import json

import pytest

import causalcache.independent_confirm_runner as runner
from causalcache.gate_v1_contract import canonical_json_bytes
from causalcache.independent_confirm_artifact import independent_decisions_bytes
from causalcache.independent_confirm_data import (
    CONFIRM_CANDIDATE_EVENT_STEP_IDS,
    CONFIRM_DECISION_STEP_ID,
    CONFIRM_SOURCE_IDS,
)
from causalcache.independent_confirm_runner import (
    authorize_adopted_payload_commit,
    independent_selection_payload_bytes,
    persist_and_seal_label_blind_payload,
    replay_independent_selection_bundle,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decision_payload() -> bytes:
    ensemble_scores = {}
    seed_scores = [dict() for _ in range(5)]
    for ordinal, source_id in enumerate(CONFIRM_SOURCE_IDS):
        state_id = f"{source_id}:decision_step:{CONFIRM_DECISION_STEP_ID:03d}"
        ensemble_scores[state_id] = {
            event: (-0.1, 0.4, 0.3, 0.2)[event - 1] + ordinal / 1000.0
            for event in CONFIRM_CANDIDATE_EVENT_STEP_IDS
        }
        for seed in range(5):
            seed_scores[seed][state_id] = {
                event: (-0.2, 0.5, 0.25, 0.1)[event - 1] + seed / 100.0
                for event in CONFIRM_CANDIDATE_EVENT_STEP_IDS
            }
    return independent_decisions_bytes(
        ensemble_scores=ensemble_scores,
        seed_scores=seed_scores,
    )


def _persisted_seal():
    payload = _decision_payload()
    bundle = replay_independent_selection_bundle(payload)

    def persist(_files, inventory):
        return {
            "durable": True,
            "file_count": len(inventory),
            "inventory_sha256": _sha256(canonical_json_bytes(list(inventory))),
            "persistence_id": "continuation-read-only-fixture",
        }

    return persist_and_seal_label_blind_payload(
        {"confirm/independent-decisions-v1.json": payload},
        independent_selections=bundle,
        independent_selection_path="confirm/independent-decisions-v1.json",
        persist_fn=persist,
    )


def test_replay_rebuilds_opaque_bundle_and_binds_full_file_sha256() -> None:
    payload = _decision_payload()
    bundle = replay_independent_selection_bundle(payload)

    assert independent_selection_payload_bytes(bundle) == payload
    assert bundle.selection_sha256 == _sha256(payload)
    assert bundle.ensemble_selections[
        f"{CONFIRM_SOURCE_IDS[0]}:decision_step:006"
    ] == (2, 3)
    assert len(bundle.seed_selections) == 5
    assert len(bundle.score_records) == len(CONFIRM_SOURCE_IDS)
    assert json.loads(payload)["selection_sha256"] != bundle.selection_sha256
    with pytest.raises(TypeError):
        bundle.ensemble_selections["new-state"] = (1,)  # type: ignore[index]


def test_replay_rejects_noncanonical_decision_bytes() -> None:
    payload = _decision_payload()
    noncanonical = json.dumps(
        json.loads(payload), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    assert noncanonical != payload

    with pytest.raises(ValueError, match="canonical JSON"):
        replay_independent_selection_bundle(noncanonical)
    with pytest.raises(TypeError, match="must be bytes"):
        replay_independent_selection_bundle(bytearray(payload))  # type: ignore[arg-type]


def test_authorize_adopted_payload_commit_uses_only_fresh_replay() -> None:
    seal = _persisted_seal()
    calls = []

    def verify(commit, inventory):
        calls.append((commit, tuple(inventory)))
        return True

    receipt = authorize_adopted_payload_commit(seal, "a" * 40, verify)

    assert len(calls) == 1
    assert calls[0][0] == "a" * 40
    assert receipt.payload_commit == "a" * 40
    assert receipt.inventory_sha256 == seal.inventory_sha256
    assert receipt.selection_sha256 == seal.selection_sha256


def test_adoption_verify_failure_does_not_issue_receipt(monkeypatch) -> None:
    seal = _persisted_seal()
    issued = []
    monkeypatch.setattr(
        runner,
        "PayloadCommitReceipt",
        lambda **kwargs: issued.append(kwargs),
    )

    with pytest.raises(ValueError, match="fresh replay verification failed"):
        authorize_adopted_payload_commit(
            seal,
            "b" * 40,
            lambda _commit, _inventory: False,
        )
    assert issued == []


def test_adoption_rejects_tampered_seal_before_remote_verify() -> None:
    seal = _persisted_seal()
    seal.inventory_sha256 = "0" * 64
    calls = []

    with pytest.raises(ValueError, match="persisted seal binding is invalid"):
        authorize_adopted_payload_commit(
            seal,
            "c" * 40,
            lambda commit, inventory: calls.append((commit, inventory)) is None,
        )
    assert calls == []
