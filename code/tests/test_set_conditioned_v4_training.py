from __future__ import annotations

import hashlib
import importlib.util
import itertools
import math

import pytest

from causalcache.gate_v1_data import CandidateFeatures, FeatureState, GateState
from causalcache.gate_v1_provenance import canonical_model_state_sha256
from causalcache.gate_v1_training import build_model as build_gate_v1_model
from causalcache.restoration_v2_2_label_table import validate_complete_distance_table
from causalcache.set_conditioned_v4_data import build_training_batch
from causalcache.set_conditioned_v4_training import (
    EXPECTED_BASE_PARAMETER_COUNT,
    EXPECTED_RESIDUAL_PARAMETER_COUNT,
    assert_zero_residual_identity,
    build_optimizer,
    build_residual_head,
    freeze_independent_base,
    independent_selection_from_scores,
    load_residual_checkpoint,
    optimization_step,
    predict_state,
    replay_fold_clean_base,
    residual_loss,
    select_base,
    select_from_scores,
    serialize_residual_checkpoint,
    train_fixed_epochs,
    validate_frozen_base,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


def _state(
    source: str = "source-a",
    event_count: int = 3,
    *,
    utility_by_coalition: dict[tuple[int, ...], float] | None = None,
) -> GateState:
    event_ids = tuple(range(1, event_count + 1))
    utilities = {
        coalition: (
            utility_by_coalition[coalition]
            if utility_by_coalition is not None
            else math.fsum(coalition)
            + math.fsum(
                left * right / 4.0
                for left, right in itertools.combinations(coalition, 2)
            )
        )
        for size in range(event_count + 1)
        for coalition in itertools.combinations(event_ids, size)
    }
    baseline = 20.0
    table = validate_complete_distance_table(
        event_ids,
        {coalition: baseline - utility for coalition, utility in utilities.items()},
    )
    return GateState(
        source_id=source,
        state_id=f"{source}:n{event_count}",
        decision_step_id=event_count + 2,
        candidate_event_step_ids=event_ids,
        q64=(0.125,) * 64,
        candidates=tuple(
            CandidateFeatures(
                event_step_id=event,
                h64=tuple(1.0 if index == event - 1 else 0.0 for index in range(64)),
                g8=(event / 10.0,) * 8,
            )
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


def _frozen_base(*, seed: int = 0, zero: bool = False):
    import torch

    model = build_gate_v1_model("independent", seed=seed)
    if zero:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
    return freeze_independent_base(
        model,
        seed=seed,
        selected_epoch=1,
        learning_rate=0.0003,
    )


def _base_rows(scores: tuple[tuple[int, float], ...]):
    values = dict(scores)
    return (
        ((), 0.0),
        *(((event,), value) for event, value in scores),
        *(
            (pair, values[pair[0]] + values[pair[1]])
            for pair in itertools.combinations(values, 2)
        ),
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is unavailable")
def test_frozen_base_and_residual_architectures_are_exact() -> None:
    import torch

    base = _frozen_base()
    head = build_residual_head(seed=0)
    assert (
        sum(parameter.numel() for parameter in base.model.parameters())
        == EXPECTED_BASE_PARAMETER_COUNT
    )
    assert (
        sum(parameter.numel() for parameter in head.parameters())
        == EXPECTED_RESIDUAL_PARAMETER_COUNT
    )
    assert head.network[0].in_features == 416
    assert head.network[0].out_features == 64
    assert torch.count_nonzero(head.network[2].weight) == 0
    assert torch.count_nonzero(head.network[2].bias) == 0
    assert all(not parameter.requires_grad for parameter in base.model.parameters())
    assert validate_frozen_base(base) == base.model_state_sha256


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is unavailable")
def test_epoch_zero_residual_is_exact_and_reproduces_independent() -> None:
    state = _state()
    base = _frozen_base()
    head = build_residual_head(seed=0)
    prediction = predict_state(base, head, _feature(state))
    assert all(value == 0.0 for _, value in prediction.pair_residual_scores)
    selections = assert_zero_residual_identity(base, head, (_feature(state),))
    assert selections == {state.state_id: select_base(base, head, state)}


@pytest.mark.parametrize(
    "scores",
    (
        ((1, -1.0), (2, -2.0), (3, -3.0)),
        ((1, 0.0), (2, -1.0), (3, -2.0)),
        ((1, 1.0), (2, 0.0), (3, -1.0)),
        ((1, 1.0), (2, 1.0), (3, 1.0)),
        ((1, 3.0), (2, 2.0), (3, 1.0)),
        ((1, 3.0), (2, 3.0), (3, -1.0)),
    ),
)
def test_exhaustive_zero_residual_ties_match_positive_top2(scores) -> None:
    assert select_from_scores(_base_rows(scores)) == independent_selection_from_scores(
        scores
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is unavailable")
def test_loss_is_normalized_pair_correction_plus_pair_ranking() -> None:
    utilities = {(): 0.0, (1,): 2.0, (2,): 3.0, (1, 2): 10.0}
    state = _state(event_count=2, utility_by_coalition=utilities)
    batch = build_training_batch((state,))
    base = _frozen_base(zero=True)
    head = build_residual_head(seed=0)
    total, scalars = residual_loss(base, head, batch)
    expected_regression = 0.5 * (10.0 / 20.0) ** 2
    expected_ranking = math.log(2.0)
    assert scalars["regression"] == pytest.approx(expected_regression, abs=1e-7)
    assert scalars["ranking"] == pytest.approx(expected_ranking, abs=1e-7)
    assert float(total.detach()) == pytest.approx(
        expected_regression + 0.25 * expected_ranking,
        abs=1e-7,
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is unavailable")
def test_residual_step_changes_only_head_and_never_base_hash_or_grad() -> None:
    base = _frozen_base()
    head = build_residual_head(seed=0)
    batch = build_training_batch((_state(),))
    optimizer = build_optimizer(head, learning_rate=0.001)
    before_head = tuple(parameter.detach().clone() for parameter in head.parameters())
    before_base = base.model_state_sha256
    scalars = optimization_step(base, head, optimizer, batch)
    assert math.isfinite(scalars["total"])
    assert any(
        not before.equal(after)
        for before, after in zip(before_head, head.parameters(), strict=True)
    )
    assert validate_frozen_base(base) == before_base
    assert all(parameter.grad is None for parameter in base.model.parameters())


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is unavailable")
def test_optimizer_with_any_non_head_parameter_fails_closed() -> None:
    import torch

    base = _frozen_base()
    head = build_residual_head(seed=0)
    extra = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([*head.parameters(), extra], lr=0.001)
    with pytest.raises(ValueError, match="residual-head-only"):
        optimization_step(base, head, optimizer, build_training_batch((_state(),)))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is unavailable")
def test_fold_clean_replay_trains_then_freezes_base() -> None:
    initial = build_gate_v1_model("independent", seed=0)
    initial_sha = canonical_model_state_sha256(initial)
    base = replay_fold_clean_base(
        (_state(),),
        seed=0,
        selected_epoch=1,
        learning_rate=0.0003,
    )
    assert base.model_state_sha256 != initial_sha
    assert base.selected_epoch == 1
    assert validate_frozen_base(base) == base.model_state_sha256


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is unavailable")
def test_fixed_epoch_fit_runs_only_residual_optimizer_steps() -> None:
    base = _frozen_base(seed=1)
    fit = train_fixed_epochs(
        (_state(),),
        base=base,
        seed=1,
        epochs=2,
        learning_rate=0.001,
    )
    assert tuple(row.epoch for row in fit.history) == (1, 2)
    assert validate_frozen_base(base) == base.model_state_sha256


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is unavailable")
def test_epoch_zero_final_fit_returns_exact_frozen_base_head() -> None:
    state = _state()
    base = _frozen_base(seed=3)
    fit = train_fixed_epochs(
        (state,),
        base=base,
        seed=3,
        epochs=0,
        learning_rate=0.0003,
    )
    assert fit.history == ()
    assert_zero_residual_identity(base, fit.head, (_feature(state),))
    assert validate_frozen_base(base) == base.model_state_sha256


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_SAFETENSORS,
    reason="PyTorch or safetensors is unavailable",
)
def test_residual_checkpoint_contains_only_head_and_replays_exactly() -> None:
    head = build_residual_head(seed=2)
    payload = serialize_residual_checkpoint(head)
    replay = load_residual_checkpoint(
        payload,
        seed=2,
        expected_checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert replay.state_dict().keys() == head.state_dict().keys()
    assert serialize_residual_checkpoint(replay) == payload
    assert all(not name.startswith("base") for name in replay.state_dict())
