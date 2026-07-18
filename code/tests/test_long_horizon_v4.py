from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from causalcache.gate_v1_data import CandidateFeatures, FeatureState
from causalcache.gate_v1_formal_train import (
    load_safetensors_checkpoint,
    serialize_safetensors_checkpoint,
)
from causalcache.gate_v1_provenance import canonical_model_state_sha256
from causalcache.gate_v1_training import build_model
from causalcache.long_horizon_contract import LongHorizonContract
from causalcache.long_horizon_selectors import residual_b2_selections
from causalcache.long_horizon_v4 import (
    BASE_CHECKPOINT_SIZE_BYTES,
    BASE_LEARNING_RATE,
    BASE_SELECTED_EPOCHS,
    FROZEN_SEEDS,
    RESIDUAL_LEARNING_RATE,
    RESIDUAL_SELECTED_EPOCHS,
    FrozenV4Ensemble,
    build_zero_residual_head,
    freeze_v4_seed,
    frozen_artifact_roster,
    load_residual_checkpoint,
    predict_residual_seed_scores,
    predict_seed_scores,
    serialize_residual_checkpoint,
    validate_frozen_v4_seed,
    zero_residual_seed_scores,
)


ROOT = Path(__file__).resolve().parents[2]


def _contract() -> LongHorizonContract:
    return LongHorizonContract.load(repository_root=ROOT)


def _state(candidate_count: int) -> FeatureState:
    events = tuple(range(1, candidate_count + 1))
    return FeatureState(
        source_id=f"fixture-{candidate_count}",
        state_id=f"fixture-{candidate_count}-state",
        decision_step_id=candidate_count + 2,
        candidate_event_step_ids=events,
        q64=tuple((position + 1) / 97.0 for position in range(64)),
        candidates=tuple(
            CandidateFeatures(
                event_step_id=event,
                h64=tuple(
                    (event * 67 + position + 1) / 4096.0
                    for position in range(64)
                ),
                g8=tuple((event * 11 + position + 1) / 512.0 for position in range(8)),
            )
            for event in events
        ),
    )


def _seed_fixture(seed: int, *, residual_bias: float = 0.0):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    base = build_model("independent", seed=seed)
    base_state_sha256 = canonical_model_state_sha256(base)
    base_payload = serialize_safetensors_checkpoint(base)
    assert len(base_payload) == BASE_CHECKPOINT_SIZE_BYTES
    base_checkpoint_sha256 = hashlib.sha256(base_payload).hexdigest()
    replayed_base = load_safetensors_checkpoint(
        base_payload,
        family="independent",
        seed=seed,
        expected_model_state_sha256=base_state_sha256,
        expected_checkpoint_sha256=base_checkpoint_sha256,
    )

    residual = build_zero_residual_head(seed=seed)
    with torch.no_grad():
        residual.network[2].bias.fill_(residual_bias)
    residual_payload = serialize_residual_checkpoint(residual)
    residual_checkpoint_sha256 = hashlib.sha256(residual_payload).hexdigest()
    replayed_residual = load_residual_checkpoint(
        residual_payload,
        seed=seed,
        expected_checkpoint_sha256=residual_checkpoint_sha256,
        expected_size_bytes=len(residual_payload),
    )
    frozen = freeze_v4_seed(
        seed=seed,
        base_model=replayed_base,
        residual_head=replayed_residual,
        base_selected_epoch=BASE_SELECTED_EPOCHS[seed],
        base_learning_rate=BASE_LEARNING_RATE,
        residual_selected_epoch=RESIDUAL_SELECTED_EPOCHS[seed],
        residual_learning_rate=RESIDUAL_LEARNING_RATE,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_checkpoint_size_bytes=len(base_payload),
        base_model_state_sha256=base_state_sha256,
        residual_checkpoint_sha256=residual_checkpoint_sha256,
        residual_checkpoint_size_bytes=len(residual_payload),
    )
    return frozen, base_payload, residual_payload


def _ensemble_fixture() -> FrozenV4Ensemble:
    return FrozenV4Ensemble(
        seeds=tuple(_seed_fixture(seed)[0] for seed in FROZEN_SEEDS),
        formal_manifest_sha256="1" * 64,
        residual_manifest_sha256="2" * 64,
        residual_metadata_sha256="3" * 64,
    )


def test_frozen_artifact_roster_is_exact_source_a_five_by_five() -> None:
    roster = frozen_artifact_roster(_contract())
    assert (
        tuple(record["seed"] for record in roster["independent_checkpoints"])
        == FROZEN_SEEDS
    )
    assert (
        tuple(record["seed"] for record in roster["residual_checkpoints"])
        == FROZEN_SEEDS
    )
    assert len(roster["independent_checkpoints"]) == 5
    assert len(roster["residual_checkpoints"]) == 5
    with pytest.raises(TypeError, match="validated LongHorizonContract"):
        frozen_artifact_roster(_contract().data)


@pytest.mark.parametrize(("candidate_count", "pair_count"), ((8, 28), (16, 120)))
def test_zero_residual_replays_base_and_emits_every_pair(
    candidate_count: int,
    pair_count: int,
) -> None:
    frozen, base_payload, _ = _seed_fixture(0)
    state = _state(candidate_count)
    predicted = predict_seed_scores(frozen, state)
    base_only = zero_residual_seed_scores(frozen, state)
    assert predicted.seed == 0
    assert predicted.singleton_scores == base_only.singleton_scores
    assert predicted.pair_residual_scores == base_only.pair_residual_scores
    assert len(predicted.singleton_scores) == candidate_count
    assert len(predicted.pair_residual_scores) == pair_count
    assert set(predicted.pair_residual_scores.values()) == {0.0}
    assert serialize_safetensors_checkpoint(frozen.base_model) == base_payload
    assert canonical_model_state_sha256(frozen.base_model) == frozen.base_model_state_sha256


def test_nonzero_residual_checkpoint_scores_every_pair_without_changing_base() -> None:
    frozen, base_payload, residual_payload = _seed_fixture(0, residual_bias=0.25)
    prediction = predict_seed_scores(frozen, _state(8))
    assert set(prediction.pair_residual_scores.values()) == {0.25}
    assert serialize_safetensors_checkpoint(frozen.base_model) == base_payload
    assert serialize_residual_checkpoint(frozen.residual_head) == residual_payload


@pytest.mark.parametrize(("candidate_count", "pair_count"), ((8, 28), (16, 120)))
def test_five_seed_adapter_feeds_residual_selector(
    candidate_count: int,
    pair_count: int,
) -> None:
    state = _state(candidate_count)
    predictions = predict_residual_seed_scores(_ensemble_fixture(), state)
    assert tuple(value.seed for value in predictions) == FROZEN_SEEDS
    assert {len(value.pair_residual_scores) for value in predictions} == {pair_count}
    result = residual_b2_selections(state, predictions)
    assert result.safe.selected_event_step_ids == result.frozen_base.selected_event_step_ids
    assert result.used_pair_candidate is False
    assert result.used_fallback is False


def test_residual_checkpoint_tamper_and_size_drift_are_rejected() -> None:
    frozen, _, payload = _seed_fixture(0)
    digest = frozen.residual_checkpoint_sha256
    tampered = payload[:-1] + bytes([payload[-1] ^ 1])
    with pytest.raises(ValueError, match="SHA256"):
        load_residual_checkpoint(
            tampered,
            seed=0,
            expected_checkpoint_sha256=digest,
            expected_size_bytes=len(payload),
        )
    with pytest.raises(ValueError, match="size"):
        load_residual_checkpoint(
            payload,
            seed=0,
            expected_checkpoint_sha256=digest,
            expected_size_bytes=len(payload) + 1,
        )


def test_frozen_base_state_and_training_metadata_drift_are_rejected() -> None:
    torch = pytest.importorskip("torch")
    frozen, _, _ = _seed_fixture(0)
    with pytest.raises(ValueError, match="metadata"):
        validate_frozen_v4_seed(replace(frozen, base_learning_rate=1e-4))
    with torch.no_grad():
        next(frozen.base_model.parameters()).view(-1)[0].add_(1.0)
    with pytest.raises(ValueError, match="base state changed"):
        validate_frozen_v4_seed(frozen)
