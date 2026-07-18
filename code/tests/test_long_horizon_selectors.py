from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from causalcache.gate_v1_data import CandidateFeatures, FeatureState
from causalcache.gate_v1_provenance import (
    FORMAL_TRAIN_SOURCE_IDS_SHA256,
    GATE_V1_CONFIG_SHA256,
    ArtifactBinding,
    FrozenEnsembleProvenance,
    FrozenTrainingProvenance,
    SeedCheckpointProvenance,
)
from causalcache.long_horizon_selectors import (
    ConditionalB2Result,
    LabelBlindSelectionSeal,
    ResidualSeedScores,
    conditional_b2_selection,
    deterministic_random_selection,
    feature_view,
    independent_selection,
    load_frozen_formal58_ensembles,
    ocr_rgb_score_selection,
    ocr_rgb_similarity_scores,
    read_label_blind_selection_seal,
    recent_selection,
    residual_b2_selections,
    seal_label_blind_selections,
    summary_only_selection,
    validate_frozen_formal58_ensembles,
)


def _state(source: str, state_id: str, n: int) -> FeatureState:
    event_ids = tuple(range(1, n + 1))
    return FeatureState(
        source_id=source,
        state_id=state_id,
        decision_step_id=n + 2,
        candidate_event_step_ids=event_ids,
        q64=(0.0,) * 64,
        candidates=tuple(
            CandidateFeatures(
                event_step_id=event,
                h64=(event / 100.0,) * 64,
                g8=(event / 100.0,) * 8,
            )
            for event in event_ids
        ),
    )


def _residual_predictions(
    state: FeatureState,
    residuals: tuple[float, float, float, float, float],
) -> tuple[ResidualSeedScores, ...]:
    pairs = tuple(
        (left, right)
        for left in state.candidate_event_step_ids
        for right in state.candidate_event_step_ids
        if left < right
    )
    return tuple(
        ResidualSeedScores(
            seed=seed,
            singleton_scores={
                event: 1.0 if event == 1 else -1.0
                for event in state.candidate_event_step_ids
            },
            pair_residual_scores={pair: residual for pair in pairs},
        )
        for seed, residual in enumerate(residuals)
    )


def test_feature_view_supports_arbitrary_n_and_drops_nonfeature_fields() -> None:
    state = _state("source", "n16", 16)
    wrapped = SimpleNamespace(**state.__dict__, table=object())
    view = feature_view(wrapped)
    assert len(view.candidate_event_step_ids) == 16
    assert view.candidate(16).event_step_id == 16
    assert not hasattr(view, "table")

    broken = replace(state, candidate_event_step_ids=tuple(range(1, 15)) + (16, 15))
    with pytest.raises(ValueError, match="strictly increasing"):
        feature_view(broken)


def test_independent_b2_b4_positive_stop_and_exact_tie_break() -> None:
    state = _state("source", "n8", 8)
    values = {1: 2.0, 2: 2.0, 3: 1.0, 4: 0.5, 5: 0.0, 6: -1.0, 7: -2.0, 8: -3.0}
    score = lambda _state, event, _coalition: values[event]

    b2 = independent_selection(state, score, budget_event_capacity=2)
    b4 = independent_selection(state, score, budget_event_capacity=4)
    assert b2.selector_name == "restoration_independent_gate"
    assert b2.selected_event_step_ids == (1, 2)
    assert b4.selected_event_step_ids == (1, 2, 3, 4)

    stopped = independent_selection(
        state,
        lambda _state, _event, _coalition: 0.0,
        budget_event_capacity=4,
    )
    assert stopped.selected_event_step_ids == ()


def test_conditional_b2_rescores_stops_and_forbids_b4() -> None:
    state = _state("source", "conditional", 8)
    calls: list[tuple[int, tuple[int, ...]]] = []

    def score(_state, event: int, coalition: tuple[int, ...]) -> float:
        calls.append((event, coalition))
        if not coalition:
            return 2.0 if event in {1, 2} else -1.0
        return 3.0 if event == 3 else -1.0

    result = conditional_b2_selection(state, score)
    assert isinstance(result, ConditionalB2Result)
    assert result.decision.selected_event_step_ids == (1, 3)
    assert result.trace[0].selected_event_step_id == 1
    assert result.trace[1].selected_before == (1,)
    assert any(coalition == (1,) for _, coalition in calls)

    stopped = conditional_b2_selection(
        state, lambda _state, _event, _coalition: 0.0
    )
    assert stopped.decision.selected_event_step_ids == ()
    assert len(stopped.trace) == 1
    with pytest.raises(ValueError, match="B=2"):
        conditional_b2_selection(state, score, budget_event_capacity=4)


def test_recent_random_and_ocr_rgb_support_b2_and_b4() -> None:
    state = _state("source", "heuristics", 16)
    assert recent_selection(state, budget_event_capacity=2).selected_event_step_ids == (15, 16)
    assert recent_selection(
        state, budget_event_capacity=4
    ).selected_event_step_ids == (13, 14, 15, 16)

    first = deterministic_random_selection(state, budget_event_capacity=4, seed=271828)
    second = deterministic_random_selection(state, budget_event_capacity=4, seed=271828)
    assert first == second
    assert len(first.selected_event_step_ids) == 4

    scores = {event: 1.0 for event in state.candidate_event_step_ids}
    ocr = ocr_rgb_score_selection(state, scores, budget_event_capacity=4)
    assert ocr.selected_event_step_ids == (1, 2, 3, 4)
    scores.pop(16)
    with pytest.raises(ValueError, match="inventory"):
        ocr_rgb_score_selection(state, scores, budget_event_capacity=2)


def test_ocr_rgb_similarity_scores_reuse_frozen_arbitrary_n_semantics() -> None:
    events = tuple(range(1, 9))
    black = bytes(256 * 256 * 3)
    white = bytes([255]) * (256 * 256 * 3)
    scores = ocr_rgb_similarity_scores(
        event_step_ids=events,
        event_ocr_tokens={
            event: (("target",) if event == 8 else (f"event-{event}",))
            for event in events
        },
        current_ocr_tokens=("target",),
        event_resized_rgb_bytes={
            event: (white if event == 8 else black) for event in events
        },
        current_resized_rgb_bytes=white,
    )
    assert set(scores) == set(events)
    assert scores[8] == pytest.approx(1.0)
    state = _state("source", "n8-ocr", 8)
    assert ocr_rgb_score_selection(
        state, scores, budget_event_capacity=2
    ).selected_event_step_ids[-1] == 8


def test_v4_unguarded_and_safe_four_of_five_guard() -> None:
    state = _state("source", "residual", 2)
    accepted = residual_b2_selections(
        state,
        _residual_predictions(state, (2.0, 2.0, 2.0, 2.0, -2.0)),
    )
    assert accepted.frozen_base.selected_event_step_ids == (1,)
    assert accepted.unguarded.selected_event_step_ids == (1, 2)
    assert accepted.safe.selected_event_step_ids == (1, 2)
    assert accepted.pair_candidate_vote_count == 4
    assert accepted.strictly_positive_margin_count == 4
    assert accepted.used_pair_candidate
    assert not accepted.used_fallback

    fallback = residual_b2_selections(
        state,
        _residual_predictions(state, (3.0, 3.0, 3.0, -1.0, -1.0)),
    )
    assert fallback.unguarded.selected_event_step_ids == (1, 2)
    assert fallback.safe.selected_event_step_ids == (1,)
    assert fallback.pair_candidate_vote_count == 3
    assert fallback.used_fallback

    with pytest.raises(ValueError, match="B=2"):
        residual_b2_selections(
            state,
            _residual_predictions(state, (0.0,) * 5),
            budget_event_capacity=4,
        )


def test_v4_empty_set_wins_nonpositive_scores_and_ties() -> None:
    state = _state("source", "negative", 2)
    predictions = tuple(
        ResidualSeedScores(
            seed=seed,
            singleton_scores={1: 0.0, 2: -1.0},
            pair_residual_scores={(1, 2): 1.0},
        )
        for seed in range(5)
    )
    result = residual_b2_selections(state, predictions)
    assert result.frozen_base.selected_event_step_ids == ()
    assert result.unguarded.selected_event_step_ids == ()
    assert result.safe.selected_event_step_ids == ()


def test_selection_seal_is_canonical_complete_and_replayable() -> None:
    states = (_state("s1", "a", 8), _state("s2", "b", 16))
    decisions = []
    for state in states:
        decisions.extend(
            (
                recent_selection(state, budget_event_capacity=4),
                independent_selection(
                    state,
                    lambda _state, event, _coalition: float(event),
                    budget_event_capacity=4,
                ),
            )
        )
    seal = seal_label_blind_selections(states, decisions)
    shuffled = seal_label_blind_selections(tuple(reversed(states)), tuple(reversed(decisions)))
    assert isinstance(seal, LabelBlindSelectionSeal)
    assert seal.payload_bytes == shuffled.payload_bytes
    assert seal.sha256 == shuffled.sha256
    payload = json.loads(seal.payload_bytes)
    assert payload["state_count"] == 2
    assert payload["record_count"] == 4

    replay = read_label_blind_selection_seal(
        seal.payload_bytes,
        states,
        expected_sha256=seal.sha256,
    )
    assert replay.payload_bytes == seal.payload_bytes
    assert replay.selections_by_selector["recent"]["a"] == (5, 6, 7, 8)

    with pytest.raises(ValueError, match="full feature-state roster"):
        seal_label_blind_selections(states, decisions[:-1])
    with pytest.raises(ValueError, match="canonical"):
        read_label_blind_selection_seal(seal.payload_bytes.rstrip(), states)


def test_seal_covers_the_n8_b2_and_n16_b4_config_rosters() -> None:
    n8 = _state("source", "n8-matrix", 8)
    scores = lambda _state, event, _coalition: float(event)
    residual = residual_b2_selections(
        n8,
        _residual_predictions(n8, (0.0, 0.0, 0.0, 0.0, 0.0)),
    )
    n8_decisions = (
        independent_selection(n8, scores, budget_event_capacity=2),
        recent_selection(n8, budget_event_capacity=2),
        ocr_rgb_score_selection(
            n8,
            {event: float(event) for event in n8.candidate_event_step_ids},
            budget_event_capacity=2,
        ),
        conditional_b2_selection(n8, scores).decision,
        residual.safe,
        deterministic_random_selection(n8, budget_event_capacity=2, seed=271828),
        summary_only_selection(n8, budget_event_capacity=2),
    )
    n8_roster = (
        "restoration_independent_gate",
        "recent",
        "ocr_rgb_v2",
        "v1_conditional",
        "v4_safe_frozen_base_residual",
        "random",
        "summary_only",
    )
    n8_seal = seal_label_blind_selections(
        (n8,), n8_decisions, expected_selector_names=n8_roster
    )
    assert set(n8_seal.selections_by_selector) == set(n8_roster)

    n16 = _state("source", "n16-matrix", 16)
    n16_residual = residual_b2_selections(
        n16,
        _residual_predictions(n16, (0.0, 0.0, 0.0, 0.0, 0.0)),
    )
    n16_b2_decisions = (
        independent_selection(n16, scores, budget_event_capacity=2),
        recent_selection(n16, budget_event_capacity=2),
        ocr_rgb_score_selection(
            n16,
            {event: float(event) for event in n16.candidate_event_step_ids},
            budget_event_capacity=2,
        ),
        conditional_b2_selection(n16, scores).decision,
        n16_residual.safe,
        deterministic_random_selection(n16, budget_event_capacity=2, seed=271828),
        summary_only_selection(n16, budget_event_capacity=2),
    )
    n16_b2_seal = seal_label_blind_selections(
        (n16,), n16_b2_decisions, expected_selector_names=n8_roster
    )
    assert set(n16_b2_seal.selections_by_selector) == set(n8_roster)

    n16_decisions = (
        independent_selection(n16, scores, budget_event_capacity=4),
        recent_selection(n16, budget_event_capacity=4),
        ocr_rgb_score_selection(
            n16,
            {event: float(event) for event in n16.candidate_event_step_ids},
            budget_event_capacity=4,
        ),
        deterministic_random_selection(n16, budget_event_capacity=4, seed=271828),
        summary_only_selection(n16, budget_event_capacity=4),
    )
    n16_roster = (
        "restoration_independent_gate",
        "recent",
        "ocr_rgb_v2",
        "random",
        "summary_only",
    )
    n16_seal = seal_label_blind_selections(
        (n16,), n16_decisions, expected_selector_names=n16_roster
    )
    assert set(n16_seal.selections_by_selector) == set(n16_roster)

    with pytest.raises(ValueError, match="expected matrix"):
        seal_label_blind_selections(
            (n16,),
            n16_decisions,
            expected_selector_names=(*n16_roster, "v1_conditional"),
        )

    with pytest.raises(ValueError, match="noncanonical"):
        seal_label_blind_selections((n8,), (residual.unguarded,))


def test_one_seal_binds_the_complete_mixed_budget_matrix_without_overwrite() -> None:
    states = (_state("s1", "n8", 8), _state("s2", "n16", 16))
    score = lambda _state, event, _coalition: float(event)
    decisions = []
    for state in states:
        residual = residual_b2_selections(
            state,
            _residual_predictions(state, (0.0, 0.0, 0.0, 0.0, 0.0)),
        )
        decisions.extend(
            (
                independent_selection(state, score, budget_event_capacity=2),
                recent_selection(state, budget_event_capacity=2),
                ocr_rgb_score_selection(
                    state,
                    {
                        event: float(event)
                        for event in state.candidate_event_step_ids
                    },
                    budget_event_capacity=2,
                ),
                conditional_b2_selection(state, score).decision,
                residual.safe,
                deterministic_random_selection(
                    state, budget_event_capacity=2, seed=271828
                ),
                summary_only_selection(state, budget_event_capacity=2),
                independent_selection(state, score, budget_event_capacity=4),
                recent_selection(state, budget_event_capacity=4),
                ocr_rgb_score_selection(
                    state,
                    {
                        event: float(event)
                        for event in state.candidate_event_step_ids
                    },
                    budget_event_capacity=4,
                ),
                deterministic_random_selection(
                    state, budget_event_capacity=4, seed=271828
                ),
                summary_only_selection(state, budget_event_capacity=4),
            )
        )
    matrix = {
        2: (
            "restoration_independent_gate",
            "recent",
            "ocr_rgb_v2",
            "v1_conditional",
            "v4_safe_frozen_base_residual",
            "random",
            "summary_only",
        ),
        4: (
            "restoration_independent_gate",
            "recent",
            "ocr_rgb_v2",
            "random",
            "summary_only",
        ),
    }
    seal = seal_label_blind_selections(
        states,
        decisions,
        expected_selector_names_by_budget=matrix,
    )
    assert len(seal.records) == len(states) * (7 + 5)
    assert seal.selections_by_budget[2]["recent"]["n8"] == (7, 8)
    assert seal.selections_by_budget[4]["recent"]["n8"] == (5, 6, 7, 8)
    with pytest.raises(ValueError, match="mixed-budget"):
        _ = seal.selections_by_selector

    replay = read_label_blind_selection_seal(
        seal.payload_bytes,
        states,
        expected_sha256=seal.sha256,
    )
    assert replay.payload_bytes == seal.payload_bytes
    assert replay.selections_by_budget[4]["summary_only"]["n16"] == ()

    with pytest.raises(ValueError, match="expected matrix"):
        seal_label_blind_selections(
            states,
            decisions,
            expected_selector_names_by_budget={2: matrix[2]},
        )


def test_formal58_validation_replays_both_provenances(monkeypatch) -> None:
    calls = []
    ensembles = SimpleNamespace(
        checkpoint_load_count=10,
        conditional=SimpleNamespace(family="conditional", seeds=tuple(range(5))),
        conditional_provenance=object(),
        independent=SimpleNamespace(family="independent", seeds=tuple(range(5))),
        independent_provenance=object(),
    )
    monkeypatch.setattr(
        "causalcache.long_horizon_selectors.validate_frozen_ensemble",
        lambda ensemble, provenance: calls.append((ensemble.family, provenance)),
    )
    assert validate_frozen_formal58_ensembles(ensembles) is ensembles
    assert [family for family, _ in calls] == ["conditional", "independent"]

    ensembles.checkpoint_load_count = 9
    with pytest.raises(ValueError, match="ten loaded checkpoints"):
        validate_frozen_formal58_ensembles(ensembles)


def test_long_horizon_formal58_loader_accepts_the_frozen_config_schema(
    monkeypatch,
) -> None:
    repository = "owner/formal58"
    revision = "a" * 40
    manifests = []
    checkpoint_records = []
    payloads = {}
    for family in ("conditional", "independent"):
        selected_epochs = (1, 2, 3, 4, 5)
        training = FrozenTrainingProvenance(
            family=family,
            gate_config_sha256=GATE_V1_CONFIG_SHA256,
            formal_train_source_ids_sha256=FORMAL_TRAIN_SOURCE_IDS_SHA256,
            learning_rate=0.0003,
            selected_epochs=selected_epochs,
            oof_selection_sha256=hashlib.sha256(
                f"{family}-selection".encode()
            ).hexdigest(),
            feature_artifact=ArtifactBinding(
                repository=repository,
                revision=revision,
                path=f"{family}/features.jsonl",
                sha256=hashlib.sha256(f"{family}-features".encode()).hexdigest(),
            ),
            label_artifact=ArtifactBinding(
                repository=repository,
                revision=revision,
                path=f"{family}/labels.jsonl",
                sha256=hashlib.sha256(f"{family}-labels".encode()).hexdigest(),
            ),
            training_report_artifact=ArtifactBinding(
                repository=repository,
                revision=revision,
                path=f"{family}/report.json",
                sha256=hashlib.sha256(f"{family}-report".encode()).hexdigest(),
            ),
        )
        checkpoints = []
        for seed, epoch in enumerate(selected_epochs):
            path = f"checkpoints/{family}-{seed}.safetensors"
            payload = f"{family}-{seed}".encode()
            artifact_sha = hashlib.sha256(payload).hexdigest()
            state_sha = hashlib.sha256(f"{family}-{seed}-state".encode()).hexdigest()
            checkpoints.append(
                SeedCheckpointProvenance(
                    seed=seed,
                    selected_epoch=epoch,
                    model_state_sha256=state_sha,
                    checkpoint_artifact=ArtifactBinding(
                        repository=repository,
                        revision=revision,
                        path=path,
                        sha256=artifact_sha,
                    ),
                )
            )
            checkpoint_records.append(
                {
                    "family": family,
                    "seed": seed,
                    "path": path,
                    "sha256": artifact_sha,
                    "model_state_sha256": state_sha,
                }
            )
            payloads[path] = payload
        provenance = FrozenEnsembleProvenance(
            training=training,
            checkpoints=tuple(checkpoints),
        )
        manifest_path = f"manifests/{family}.json"
        manifest_payload = json.dumps(
            provenance.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        manifests.append(
            {
                "family": family,
                "path": manifest_path,
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "provenance_sha256": provenance.sha256,
            }
        )
        payloads[manifest_path] = manifest_payload
    contract = {
        "learned_model_artifacts": {
            "formal58_base_and_conditional": {
                "repo": repository,
                "revision": revision,
                "ensemble_manifests": manifests,
                "checkpoints": checkpoint_records,
            }
        }
    }
    load_calls = []
    monkeypatch.setattr(
        "causalcache.gate_v1_formal_train.load_safetensors_checkpoint",
        lambda payload, **kwargs: load_calls.append((payload, kwargs)) or object(),
    )
    validation_calls = []
    monkeypatch.setattr(
        "causalcache.long_horizon_selectors.validate_frozen_ensemble",
        lambda ensemble, provenance: validation_calls.append(
            (ensemble.family, provenance.training.family)
        ),
    )
    loaded = load_frozen_formal58_ensembles(contract, payloads)
    assert loaded.checkpoint_load_count == 10
    assert loaded.conditional.seeds == tuple(range(5))
    assert loaded.independent.seeds == tuple(range(5))
    assert len(load_calls) == 10
    assert validation_calls == [
        ("conditional", "conditional"),
        ("independent", "independent"),
    ]
