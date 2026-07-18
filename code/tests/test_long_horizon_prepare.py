from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from causalcache.gate_v1_data import CandidateFeatures, FeatureState
from causalcache.long_horizon_contract import LongHorizonContract
from causalcache.long_horizon_prepare import (
    EXPECTED_SELECTION_RECORD_COUNT,
    LoadedPreparationInputs,
    MANIFEST_RELATIVE_PATH,
    OCRRGBStateInput,
    SELECTION_RELATIVE_PATH,
    _generator_provenance,
    _load_formal_payloads,
    artifact_tree_identity,
    score_and_seal_label_blind_selectors,
    validate_selector_seal_artifact,
    write_selector_seal_artifact,
)
from causalcache.long_horizon_selectors import ResidualSeedScores


MATRIX = {
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


def _state(source: str, state_id: str, n: int) -> FeatureState:
    events = tuple(range(1, n + 1))
    return FeatureState(
        source_id=source,
        state_id=state_id,
        decision_step_id=n + 2,
        candidate_event_step_ids=events,
        q64=(0.0,) * 64,
        candidates=tuple(
            CandidateFeatures(
                event_step_id=event,
                h64=(event / 100.0,) * 64,
                g8=(event / 100.0,) * 8,
            )
            for event in events
        ),
    )


def _visual(state: FeatureState) -> OCRRGBStateInput:
    black = bytes(256 * 256 * 3)
    return OCRRGBStateInput(
        event_ocr_tokens=MappingProxyType(
            {event: (f"event-{event}",) for event in state.candidate_event_step_ids}
        ),
        current_ocr_tokens=("current",),
        event_resized_rgb_bytes=MappingProxyType(
            {event: black for event in state.candidate_event_step_ids}
        ),
        current_resized_rgb_bytes=black,
    )


class _FakeV4:
    def predict(self, state: FeatureState) -> tuple[ResidualSeedScores, ...]:
        pairs = tuple(itertools.combinations(state.candidate_event_step_ids, 2))
        return tuple(
            ResidualSeedScores(
                seed=seed,
                singleton_scores={
                    event: float(event) / 10.0
                    for event in state.candidate_event_step_ids
                },
                pair_residual_scores={pair: 0.0 for pair in pairs},
            )
            for seed in range(5)
        )


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 7
        return self.value


def _fake_formal() -> SimpleNamespace:
    return SimpleNamespace(
        independent=SimpleNamespace(family="independent"),
        conditional=SimpleNamespace(family="conditional"),
    )


def _patch_fast_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "causalcache.long_horizon_prepare.ensemble_score",
        lambda ensemble: (
            lambda _state, event, coalition: (
                float(event) / 10.0
                if ensemble.family == "independent"
                else float(event) / 10.0 - len(coalition) / 100.0
            )
        ),
    )
    monkeypatch.setattr(
        "causalcache.long_horizon_prepare.ocr_rgb_similarity_scores",
        lambda *, event_step_ids, **_kwargs: {
            event: float(event) for event in event_step_ids
        },
    )


def test_mixed_budget_seal_never_overwrites_same_selector_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_scoring(monkeypatch)
    states = (_state("s1", "s1-n8", 8), _state("s2", "s2-n16", 16))
    visual = {state.state_id: _visual(state) for state in states}
    scored = score_and_seal_label_blind_selectors(
        states=states,
        ocr_rgb_by_state=visual,
        formal_ensembles=_fake_formal(),
        v4_ensemble=_FakeV4(),
        random_seed=271828,
        selector_names_by_budget=MATRIX,
        clock_ns=_Clock(),
        require_formal=False,
    )

    assert len(scored.seal.records) == 2 * (7 + 5)
    assert scored.seal.selections_by_budget[2]["recent"]["s1-n8"] == (7, 8)
    assert scored.seal.selections_by_budget[4]["recent"]["s1-n8"] == (
        5,
        6,
        7,
        8,
    )
    with pytest.raises(ValueError, match="mixed-budget"):
        _ = scored.seal.selections_by_selector
    payload = json.loads(scored.seal.payload_bytes)
    assert payload["record_count"] == 24
    assert all(
        "distance_kl" not in record and "restoration_label" not in record
        for record in payload["records"]
    )


def test_formal_roster_scores_exactly_576_label_blind_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_scoring(monkeypatch)
    states = tuple(
        state
        for source_index in range(24)
        for state in (
            _state(f"source-{source_index}", f"source-{source_index}-n8", 8),
            _state(f"source-{source_index}", f"source-{source_index}-n16", 16),
        )
    )
    scored = score_and_seal_label_blind_selectors(
        states=states,
        ocr_rgb_by_state={state.state_id: _visual(state) for state in states},
        formal_ensembles=_fake_formal(),
        v4_ensemble=_FakeV4(),
        random_seed=271828,
        selector_names_by_budget=MATRIX,
        clock_ns=_Clock(),
        require_formal=True,
    )

    assert len(scored.seal.records) == EXPECTED_SELECTION_RECORD_COUNT == 576
    assert len(scored.latency_by_arm) == 12
    assert {record["invocation_count"] for record in scored.latency_by_arm} == {48}
    assert {record["total_nanoseconds"] for record in scored.latency_by_arm} == {48 * 7}


def test_validator_byte_replays_all_576_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_scoring(monkeypatch)
    states = tuple(
        state
        for source_index in range(24)
        for state in (
            _state(f"source-{source_index}", f"source-{source_index}-n8", 8),
            _state(f"source-{source_index}", f"source-{source_index}-n16", 16),
        )
    )
    loaded = LoadedPreparationInputs(
        states=states,
        ocr_rgb_by_state=MappingProxyType(
            {state.state_id: _visual(state) for state in states}
        ),
        formal_ensembles=_fake_formal(),
        v4_ensemble=_FakeV4(),
        contract_provenance=MappingProxyType(
            {"protocol_id": "contract", "source_a": {}, "selection_freeze": {}}
        ),
        substrate_provenance=MappingProxyType({"manifest": {"revision": "e" * 40}}),
        model_provenance=MappingProxyType(
            {
                "formal58": {"files": []},
                "v4_frozen_base_residual": {"files": []},
            }
        ),
    )
    scored = score_and_seal_label_blind_selectors(
        states=states,
        ocr_rgb_by_state=loaded.ocr_rgb_by_state,
        formal_ensembles=loaded.formal_ensembles,
        v4_ensemble=loaded.v4_ensemble,
        random_seed=271828,
        selector_names_by_budget=MATRIX,
        clock_ns=_Clock(),
        require_formal=True,
    )
    repository_root = Path(__file__).resolve().parents[2]
    git_revision = "c" * 40
    generator = _generator_provenance(
        repository_root=repository_root,
        git_revision=git_revision,
    )
    output = tmp_path / "formal-seal"
    write_selector_seal_artifact(
        output_dir=output,
        loaded=loaded,
        scored=scored,
        random_seed=271828,
        selector_matrix=MATRIX,
        generator=generator,
    )
    contract = LongHorizonContract(
        data={
            "protocol_id": "contract",
            "selector_matrix": {
                "budget_2": list(MATRIX[2]),
                "budget_4": list(MATRIX[4]),
                "random_seed_roster": [271828],
                "random_seed_roster_frozen_in_source_a": True,
                "summary_only_high_fidelity_event_count": 0,
                "selector_sets_must_be_label_blind_sealed_before_any_restoration_distance": True,
            },
        },
        sha256="d" * 64,
        repository_root=repository_root,
        source_path=repository_root
        / "code/configs/causalcache_long_horizon_development_v1.json",
    )
    monkeypatch.setattr(
        "causalcache.long_horizon_prepare.load_preparation_inputs",
        lambda **_kwargs: loaded,
    )
    result = validate_selector_seal_artifact(
        output_dir=output,
        contract=contract,
        substrate_dir=tmp_path / "substrate",
        substrate_revision="e" * 40,
        formal_model_dir=tmp_path / "formal",
        v4_model_dir=tmp_path / "v4",
        random_seed=271828,
        expected_git_revision=git_revision,
    )
    assert result["record_count"] == 576
    assert result["state_count"] == 48
    assert result["formal58_checkpoint_count"] == 10
    assert result["v4_residual_checkpoint_count"] == 5


def test_selection_bytes_are_deterministic_while_latency_is_external(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_scoring(monkeypatch)
    states = (_state("s1", "n8", 8), _state("s2", "n16", 16))
    kwargs = {
        "states": states,
        "ocr_rgb_by_state": {state.state_id: _visual(state) for state in states},
        "formal_ensembles": _fake_formal(),
        "v4_ensemble": _FakeV4(),
        "random_seed": 271828,
        "selector_names_by_budget": MATRIX,
        "require_formal": False,
    }
    first = score_and_seal_label_blind_selectors(clock_ns=_Clock(), **kwargs)

    class DifferentClock:
        def __init__(self) -> None:
            self.value = 0

        def __call__(self) -> int:
            self.value += 31
            return self.value

    second = score_and_seal_label_blind_selectors(clock_ns=DifferentClock(), **kwargs)
    assert first.seal.payload_bytes == second.seal.payload_bytes
    assert first.seal.sha256 == second.seal.sha256
    assert first.latency_by_arm != second.latency_by_arm


def test_writer_persists_exact_two_files_and_binds_latency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_scoring(monkeypatch)
    states = (_state("s1", "n8", 8), _state("s2", "n16", 16))
    scored = score_and_seal_label_blind_selectors(
        states=states,
        ocr_rgb_by_state={state.state_id: _visual(state) for state in states},
        formal_ensembles=_fake_formal(),
        v4_ensemble=_FakeV4(),
        random_seed=271828,
        selector_names_by_budget=MATRIX,
        clock_ns=_Clock(),
        require_formal=False,
    )
    loaded = LoadedPreparationInputs(
        states=states,
        ocr_rgb_by_state=MappingProxyType(
            {state.state_id: _visual(state) for state in states}
        ),
        formal_ensembles=_fake_formal(),
        v4_ensemble=_FakeV4(),
        contract_provenance=MappingProxyType(
            {"protocol_id": "contract", "path": "config.json", "sha256": "a" * 64}
        ),
        substrate_provenance=MappingProxyType({"artifact_tree_sha256": "b" * 64}),
        model_provenance=MappingProxyType(
            {"formal58": {}, "v4_frozen_base_residual": {}}
        ),
    )
    output = tmp_path / "seal"
    tree = write_selector_seal_artifact(
        output_dir=output,
        loaded=loaded,
        scored=scored,
        random_seed=271828,
        selector_matrix=MATRIX,
        generator=MappingProxyType({"git_revision": "c" * 40, "source_files": []}),
    )
    assert tree == artifact_tree_identity(output)
    assert {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    } == {
        SELECTION_RELATIVE_PATH,
        MANIFEST_RELATIVE_PATH,
    }
    manifest = json.loads((output / MANIFEST_RELATIVE_PATH).read_bytes())
    assert manifest["selection_seal"]["sha256"] == scored.seal.sha256
    assert manifest["latency"]["selection_sha256_excludes_operational_latency"]
    assert manifest["restoration_label_access_count"] == 0
    with pytest.raises(FileExistsError):
        write_selector_seal_artifact(
            output_dir=output,
            loaded=loaded,
            scored=scored,
            random_seed=271828,
            selector_matrix=MATRIX,
            generator=MappingProxyType({"git_revision": "c" * 40, "source_files": []}),
        )


def test_formal_payload_reader_verifies_exact_twelve_bound_files(
    tmp_path: Path,
) -> None:
    manifests = []
    checkpoints = []
    for family in ("conditional", "independent"):
        path = f"manifests/{family}.json"
        payload = family.encode()
        manifests.append(
            {
                "family": family,
                "path": path,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        for seed in range(5):
            path = f"checkpoints/{family}-{seed}.safetensors"
            payload = f"{family}-{seed}".encode()
            checkpoints.append(
                {
                    "family": family,
                    "seed": seed,
                    "path": path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    contract = SimpleNamespace(
        data={
            "learned_model_artifacts": {
                "formal58_base_and_conditional": {
                    "ensemble_manifests": manifests,
                    "checkpoints": checkpoints,
                }
            }
        }
    )
    payloads, inventory = _load_formal_payloads(contract, tmp_path)  # type: ignore[arg-type]
    assert len(payloads) == len(inventory) == 12
    first = tmp_path / manifests[0]["path"]
    first.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 drifted"):
        _load_formal_payloads(contract, tmp_path)  # type: ignore[arg-type]


def test_scoring_rejects_matrix_or_visual_inventory_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fast_scoring(monkeypatch)
    state = _state("s", "n8", 8)
    kwargs = {
        "states": (state,),
        "formal_ensembles": _fake_formal(),
        "v4_ensemble": _FakeV4(),
        "random_seed": 271828,
        "clock_ns": _Clock(),
        "require_formal": False,
    }
    with pytest.raises(ValueError, match="OCR/RGB state inventory"):
        score_and_seal_label_blind_selectors(
            ocr_rgb_by_state={},
            selector_names_by_budget=MATRIX,
            **kwargs,
        )
    with pytest.raises(ValueError, match="differs from Source-A"):
        score_and_seal_label_blind_selectors(
            ocr_rgb_by_state={state.state_id: _visual(state)},
            selector_names_by_budget={2: MATRIX[2]},
            **kwargs,
        )
