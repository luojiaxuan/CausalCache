"""Frozen independent base and pair-only residual training for v4."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.gate_v1_data import (
    INDEPENDENT_INPUT_DIMENSION,
    FeatureState,
    GateState,
    build_training_batch as build_gate_v1_training_batch,
)
from causalcache.gate_v1_formal_train import load_safetensors_checkpoint
from causalcache.gate_v1_provenance import canonical_model_state_sha256
from causalcache.gate_v1_training import (
    GRADIENT_NORM,
    LEARNING_RATES,
    MAXIMUM_EPOCHS,
    RANKING_COEFFICIENT,
    SEEDS,
    SMOOTH_L1_BETA,
    WEIGHT_DECAY,
    build_model as build_gate_v1_model,
    build_optimizer as build_gate_v1_optimizer,
    configure_deterministic_cpu,
    optimization_step as gate_v1_optimization_step,
)
from causalcache.set_conditioned_v4_data import (
    BASE_HIDDEN_DIMENSION,
    PAIR_FEATURE_DIMENSION,
    V4TrainingBatch,
    build_training_batch,
    feasible_coalitions,
    v4_independent_input,
)

RESIDUAL_HIDDEN_DIMENSION = 64
EXPECTED_BASE_PARAMETER_COUNT = 25_609
EXPECTED_RESIDUAL_PARAMETER_COUNT = 26_753


def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("set-conditioned v4 training requires PyTorch") from error
    return torch


try:
    import torch as _torch_module
except ModuleNotFoundError:
    _torch_module = None


if _torch_module is not None:

    class SetConditionedV4ResidualHead(_torch_module.nn.Module):
        """Chronological pair correction over frozen gate-v1 embeddings."""

        def __init__(self) -> None:
            super().__init__()
            self.network = _torch_module.nn.Sequential(
                _torch_module.nn.Linear(
                    PAIR_FEATURE_DIMENSION,
                    RESIDUAL_HIDDEN_DIMENSION,
                ),
                _torch_module.nn.GELU(approximate="none"),
                _torch_module.nn.Linear(RESIDUAL_HIDDEN_DIMENSION, 1),
            )
            # note (luojiaxuan): Only the terminal layer is zeroed so epoch zero
            # is exactly the frozen base while the random first layer can learn.
            _torch_module.nn.init.zeros_(self.network[2].weight)
            _torch_module.nn.init.zeros_(self.network[2].bias)

        def forward(
            self,
            left_embeddings: Any,
            right_embeddings: Any,
            q64: Any,
        ) -> Any:
            if (
                left_embeddings.shape != right_embeddings.shape
                or left_embeddings.ndim != 2
                or left_embeddings.shape[1] != BASE_HIDDEN_DIMENSION
                or q64.ndim != 2
                or q64.shape[0] != left_embeddings.shape[0]
                or q64.shape[1] != 64
            ):
                raise ValueError("v4 residual tensor geometry is invalid")
            pair_features = _torch_module.cat(
                (
                    left_embeddings,
                    right_embeddings,
                    left_embeddings * right_embeddings,
                    _torch_module.abs(left_embeddings - right_embeddings),
                    q64,
                ),
                dim=1,
            )
            if pair_features.shape[1] != PAIR_FEATURE_DIMENSION:
                raise RuntimeError("v4 pair feature dimension drifted")
            return self.network(pair_features).squeeze(-1)

else:

    class SetConditionedV4ResidualHead:
        """Import placeholder for environments without PyTorch."""

        def __init__(self) -> None:
            _torch()


@dataclass(frozen=True)
class FrozenIndependentBase:
    seed: int
    selected_epoch: int
    learning_rate: float
    model: Any
    model_state_sha256: str


@dataclass(frozen=True)
class V4StatePrediction:
    singleton_scores: tuple[tuple[int, float], ...]
    pair_residual_scores: tuple[tuple[tuple[int, int], float], ...]
    base_set_scores: tuple[tuple[tuple[int, ...], float], ...]
    total_set_scores: tuple[tuple[tuple[int, ...], float], ...]

    def score(self, coalition: Sequence[int], *, use_residual: bool = True) -> float:
        key = tuple(coalition)
        rows = self.total_set_scores if use_residual else self.base_set_scores
        for candidate, value in rows:
            if candidate == key:
                return value
        raise KeyError(key)


@dataclass(frozen=True)
class V4EpochResult:
    epoch: int
    regression: float
    ranking: float
    total: float
    preclip_gradient_norm: float


@dataclass(frozen=True)
class V4FixedEpochFit:
    head: Any
    batch: V4TrainingBatch
    history: tuple[V4EpochResult, ...]


def _validate_base_architecture(model: Any) -> None:
    torch = _torch()
    if not isinstance(model, torch.nn.Sequential) or len(model) != 5:
        raise ValueError("frozen independent base must be the gate-v1 Sequential")
    first, gelu1, second, gelu2, output = tuple(model)
    if (
        not isinstance(first, torch.nn.Linear)
        or first.in_features != INDEPENDENT_INPUT_DIMENSION
        or first.out_features != BASE_HIDDEN_DIMENSION
        or not isinstance(gelu1, torch.nn.GELU)
        or gelu1.approximate != "none"
        or not isinstance(second, torch.nn.Linear)
        or second.in_features != BASE_HIDDEN_DIMENSION
        or second.out_features != BASE_HIDDEN_DIMENSION
        or not isinstance(gelu2, torch.nn.GELU)
        or gelu2.approximate != "none"
        or not isinstance(output, torch.nn.Linear)
        or output.in_features != BASE_HIDDEN_DIMENSION
        or output.out_features != 1
    ):
        raise ValueError("frozen independent base architecture drifted")
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != EXPECTED_BASE_PARAMETER_COUNT:
        raise ValueError("frozen independent base parameter count drifted")


def freeze_independent_base(
    model: Any,
    *,
    seed: int,
    selected_epoch: int,
    learning_rate: float,
) -> FrozenIndependentBase:
    """Freeze one final or fold-clean independent model and bind its state."""
    if seed not in SEEDS:
        raise ValueError("frozen base seed is outside the five-seed contract")
    if (
        type(selected_epoch) is not int
        or not 1 <= selected_epoch <= MAXIMUM_EPOCHS
        or learning_rate not in LEARNING_RATES
    ):
        raise ValueError("frozen base training metadata drifted")
    _validate_base_architecture(model)
    model.eval()
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    frozen = FrozenIndependentBase(
        seed=seed,
        selected_epoch=selected_epoch,
        learning_rate=learning_rate,
        model=model,
        model_state_sha256=canonical_model_state_sha256(model),
    )
    validate_frozen_base(frozen)
    return frozen


def validate_frozen_base(base: FrozenIndependentBase) -> str:
    if not isinstance(base, FrozenIndependentBase):
        raise TypeError("base must be a FrozenIndependentBase")
    if (
        base.seed not in SEEDS
        or type(base.selected_epoch) is not int
        or not 1 <= base.selected_epoch <= MAXIMUM_EPOCHS
        or base.learning_rate not in LEARNING_RATES
    ):
        raise ValueError("frozen base metadata is invalid")
    _validate_base_architecture(base.model)
    if base.model.training:
        raise ValueError("frozen base must remain in eval mode")
    if any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in base.model.parameters()
    ):
        raise ValueError("frozen base acquired a gradient or trainable parameter")
    observed = canonical_model_state_sha256(base.model)
    if observed != base.model_state_sha256:
        raise ValueError("frozen base model state changed")
    return observed


def load_frozen_independent_base(
    checkpoint_payload: bytes,
    *,
    seed: int,
    selected_epoch: int,
    learning_rate: float,
    expected_model_state_sha256: str,
    expected_checkpoint_sha256: str,
) -> FrozenIndependentBase:
    """Strictly load one canonical gate-v1 checkpoint and freeze it."""
    model = load_safetensors_checkpoint(
        checkpoint_payload,
        family="independent",
        seed=seed,
        expected_model_state_sha256=expected_model_state_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    return freeze_independent_base(
        model,
        seed=seed,
        selected_epoch=selected_epoch,
        learning_rate=learning_rate,
    )


def replay_fold_clean_base(
    states: Sequence[GateState],
    *,
    seed: int,
    selected_epoch: int,
    learning_rate: float,
) -> FrozenIndependentBase:
    """Replay gate v1 on fold-train states, then freeze before residual fitting."""
    if seed not in SEEDS or learning_rate not in LEARNING_RATES:
        raise ValueError("fold-clean base replay metadata drifted")
    if type(selected_epoch) is not int or not 1 <= selected_epoch <= MAXIMUM_EPOCHS:
        raise ValueError("fold-clean base epoch is invalid")
    training_states = tuple(states)
    batch = build_gate_v1_training_batch(training_states, family="independent")
    model = build_gate_v1_model("independent", seed=seed)
    optimizer = build_gate_v1_optimizer(model, learning_rate=learning_rate)
    for _ in range(selected_epoch):
        gate_v1_optimization_step(model, optimizer, batch)
    return freeze_independent_base(
        model,
        seed=seed,
        selected_epoch=selected_epoch,
        learning_rate=learning_rate,
    )


def build_residual_head(*, seed: int) -> SetConditionedV4ResidualHead:
    if seed not in SEEDS:
        raise ValueError("v4 residual seed is outside the five-seed contract")
    torch = configure_deterministic_cpu(seed)
    head = SetConditionedV4ResidualHead()
    head.seed = seed
    head.to(device="cpu", dtype=torch.float32)
    observed = sum(parameter.numel() for parameter in head.parameters())
    if observed != EXPECTED_RESIDUAL_PARAMETER_COUNT:
        raise RuntimeError("v4 residual parameter count drifted")
    return head


def _validate_base_head_pair(base: FrozenIndependentBase, head: Any) -> None:
    validate_frozen_base(base)
    if getattr(head, "seed", None) != base.seed:
        raise ValueError("v4 residual head seed differs from frozen base seed")


def build_optimizer(head: Any, *, learning_rate: float) -> Any:
    if learning_rate not in LEARNING_RATES:
        raise ValueError("v4 residual learning rate is outside the frozen grid")
    torch = _torch()
    return torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def _base_embeddings_and_scores(
    base: FrozenIndependentBase, inputs: Any
) -> tuple[Any, Any]:
    validate_frozen_base(base)
    torch = _torch()
    if (
        inputs.ndim != 2
        or inputs.shape[1] != INDEPENDENT_INPUT_DIMENSION
        or inputs.dtype != torch.float32
        or inputs.device.type != "cpu"
    ):
        raise ValueError("v4 frozen-base input tensor geometry drifted")
    base.model.eval()
    # note (luojiaxuan): The base forward is detached twice by contract:
    # requires_grad is false on every parameter and autograd is disabled here.
    with torch.no_grad():
        hidden = base.model[0](inputs)
        hidden = base.model[1](hidden)
        hidden = base.model[2](hidden)
        hidden = base.model[3](hidden)
        scores = base.model[4](hidden).squeeze(-1)
    return hidden.detach(), scores.detach()


def _forward_batch(
    base: FrozenIndependentBase,
    head: Any,
    batch: V4TrainingBatch,
) -> tuple[Any, Any, Any, Mapping[tuple[str, tuple[int, ...]], Any]]:
    torch = _torch()
    _validate_base_head_pair(base, head)
    inputs = torch.tensor(
        [row.input_vector for row in batch.event_rows],
        dtype=torch.float32,
        device="cpu",
    )
    embeddings, singleton_scores = _base_embeddings_and_scores(base, inputs)
    left_indices = torch.tensor(
        [row.left_event_index for row in batch.pair_targets],
        dtype=torch.long,
        device="cpu",
    )
    right_indices = torch.tensor(
        [row.right_event_index for row in batch.pair_targets],
        dtype=torch.long,
        device="cpu",
    )
    q64 = torch.tensor(
        [row.q64 for row in batch.pair_targets],
        dtype=torch.float32,
        device="cpu",
    )
    residual_scores = head(
        embeddings[left_indices],
        embeddings[right_indices],
        q64,
    )
    base_pair_scores = singleton_scores[left_indices] + singleton_scores[right_indices]
    pair_scores = base_pair_scores + residual_scores

    lookup: dict[tuple[str, tuple[int, ...]], Any] = {}
    zero = residual_scores.sum() * 0.0
    for state in batch.states:
        lookup[(state.state_id, ())] = zero
    for index, row in enumerate(batch.event_rows):
        lookup[(row.state_id, (row.event_step_id,))] = singleton_scores[index]
    for index, row in enumerate(batch.pair_targets):
        lookup[(row.state_id, row.coalition)] = pair_scores[index]
    return residual_scores, base_pair_scores, pair_scores, lookup


def residual_loss(
    base: FrozenIndependentBase,
    head: Any,
    batch: V4TrainingBatch,
) -> tuple[Any, dict[str, float]]:
    torch = _torch()
    residual_scores, base_pair_scores, _pair_scores, lookup = _forward_batch(
        base,
        head,
        batch,
    )
    eligible_indices = tuple(
        index
        for index, row in enumerate(batch.pair_targets)
        if row.normalized_pair_utility is not None
    )
    if not eligible_indices:
        raise ValueError("v4 batch contains no normalized pair target")
    indices = torch.tensor(eligible_indices, dtype=torch.long, device="cpu")
    targets = (
        torch.tensor(
            [
                float(batch.pair_targets[index].normalized_pair_utility)
                for index in eligible_indices
            ],
            dtype=torch.float32,
            device="cpu",
        )
        - base_pair_scores[indices]
    )
    weights = torch.tensor(
        [batch.pair_targets[index].regression_weight for index in eligible_indices],
        dtype=torch.float32,
        device="cpu",
    )
    regression_rows = torch.nn.functional.smooth_l1_loss(
        residual_scores[indices],
        targets,
        reduction="none",
        beta=SMOOTH_L1_BETA,
    )
    regression = torch.sum(regression_rows * weights)

    if batch.ranking_pairs:
        left = torch.stack(
            [lookup[(row.state_id, row.left_coalition)] for row in batch.ranking_pairs]
        )
        right = torch.stack(
            [lookup[(row.state_id, row.right_coalition)] for row in batch.ranking_pairs]
        )
        signs = torch.tensor(
            [row.target_sign for row in batch.ranking_pairs],
            dtype=torch.float32,
            device="cpu",
        )
        ranking_weights = torch.tensor(
            [row.weight for row in batch.ranking_pairs],
            dtype=torch.float32,
            device="cpu",
        )
        ranking = torch.sum(
            torch.nn.functional.softplus(-(left - right) * signs) * ranking_weights
        )
    else:
        ranking = residual_scores.sum() * 0.0
    total = regression + RANKING_COEFFICIENT * ranking
    if not bool(torch.isfinite(total)):
        raise ValueError("v4 residual loss is non-finite")
    return total, {
        "regression": float(regression.detach()),
        "ranking": float(ranking.detach()),
        "total": float(total.detach()),
    }


def optimization_step(
    base: FrozenIndependentBase,
    head: Any,
    optimizer: Any,
    batch: V4TrainingBatch,
) -> dict[str, float]:
    torch = _torch()
    before = validate_frozen_base(base)
    expected_parameters = {id(parameter) for parameter in head.parameters()}
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_parameters != expected_parameters:
        raise ValueError("v4 optimizer is not residual-head-only")
    head.train()
    optimizer.zero_grad(set_to_none=True)
    loss, scalars = residual_loss(base, head, batch)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), GRADIENT_NORM)
    if not bool(torch.isfinite(norm)):
        raise ValueError("v4 residual gradient norm is non-finite")
    optimizer.step()
    if any(
        not bool(torch.isfinite(parameter).all()) for parameter in head.parameters()
    ):
        raise ValueError("v4 residual parameter became non-finite")
    after = validate_frozen_base(base)
    if after != before:
        raise RuntimeError("v4 optimization changed the frozen independent base")
    return {**scalars, "preclip_gradient_norm": float(norm)}


def _prediction_from_components(
    state: FeatureState | GateState,
    singleton_values: Sequence[float],
    residual_values: Sequence[float],
) -> V4StatePrediction:
    event_ids = state.candidate_event_step_ids
    pair_ids = tuple(itertools.combinations(event_ids, 2))
    if len(singleton_values) != len(event_ids) or len(residual_values) != len(pair_ids):
        raise ValueError("v4 prediction component count differs from state geometry")
    singleton = tuple(zip(event_ids, map(float, singleton_values), strict=True))
    residual = tuple(zip(pair_ids, map(float, residual_values), strict=True))
    if any(not math.isfinite(value) for _, value in singleton) or any(
        not math.isfinite(value) for _, value in residual
    ):
        raise ValueError("v4 prediction is non-finite")
    singleton_by_event = dict(singleton)
    residual_by_pair = dict(residual)
    base_rows: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    base_rows.extend(((event,), value) for event, value in singleton)
    base_rows.extend(
        (pair, singleton_by_event[pair[0]] + singleton_by_event[pair[1]])
        for pair in pair_ids
    )
    total_rows = tuple(
        (
            coalition,
            value if len(coalition) != 2 else value + residual_by_pair[coalition],
        )
        for coalition, value in base_rows
    )
    return V4StatePrediction(
        singleton_scores=singleton,
        pair_residual_scores=residual,
        base_set_scores=tuple(base_rows),
        total_set_scores=total_rows,
    )


def predict_state(
    base: FrozenIndependentBase,
    head: Any,
    state: FeatureState | GateState,
) -> V4StatePrediction:
    feasible_coalitions(state)
    _validate_base_head_pair(base, head)
    torch = _torch()
    event_ids = state.candidate_event_step_ids
    pair_ids = tuple(itertools.combinations(event_ids, 2))
    inputs = torch.tensor(
        [v4_independent_input(state, event_id) for event_id in event_ids],
        dtype=torch.float32,
        device="cpu",
    )
    embeddings, singleton = _base_embeddings_and_scores(base, inputs)
    left_indices = torch.tensor(
        [event_ids.index(pair[0]) for pair in pair_ids],
        dtype=torch.long,
        device="cpu",
    )
    right_indices = torch.tensor(
        [event_ids.index(pair[1]) for pair in pair_ids],
        dtype=torch.long,
        device="cpu",
    )
    q64 = torch.tensor(
        [state.q64 for _ in pair_ids],
        dtype=torch.float32,
        device="cpu",
    )
    head.eval()
    with torch.no_grad():
        residual = head(
            embeddings[left_indices],
            embeddings[right_indices],
            q64,
        )
    return _prediction_from_components(state, singleton.tolist(), residual.tolist())


def _mean_prediction(
    predictions: Sequence[V4StatePrediction],
    state: FeatureState | GateState,
) -> V4StatePrediction:
    rows = tuple(predictions)
    if not rows:
        raise ValueError("v4 prediction ensemble cannot be empty")
    singleton = tuple(
        math.fsum(dict(row.singleton_scores)[event] for row in rows) / len(rows)
        for event in state.candidate_event_step_ids
    )
    residual = tuple(
        math.fsum(dict(row.pair_residual_scores)[pair] for row in rows) / len(rows)
        for pair in itertools.combinations(state.candidate_event_step_ids, 2)
    )
    return _prediction_from_components(state, singleton, residual)


def ensemble_prediction(
    bases: Sequence[FrozenIndependentBase],
    heads: Sequence[Any],
    state: FeatureState | GateState,
) -> V4StatePrediction:
    frozen_bases = tuple(bases)
    residual_heads = tuple(heads)
    if not frozen_bases or len(frozen_bases) != len(residual_heads):
        raise ValueError("v4 ensemble base/head inventory differs")
    return _mean_prediction(
        tuple(
            predict_state(base, head, state)
            for base, head in zip(frozen_bases, residual_heads, strict=True)
        ),
        state,
    )


def select_from_scores(
    rows: Sequence[tuple[tuple[int, ...], float]],
) -> tuple[int, ...]:
    values = tuple(rows)
    if not values or any(not math.isfinite(value) for _, value in values):
        raise ValueError("v4 score rows must be non-empty and finite")
    coalitions = tuple(coalition for coalition, _ in values)
    if len(set(coalitions)) != len(coalitions):
        raise ValueError("v4 score rows contain duplicate coalitions")
    return min(values, key=lambda row: (-row[1], len(row[0]), row[0]))[0]


def independent_selection_from_scores(
    singleton_scores: Sequence[tuple[int, float]],
    *,
    budget: int = 2,
) -> tuple[int, ...]:
    rows = tuple(singleton_scores)
    if type(budget) is not int or budget != 2:
        raise ValueError("v4 freezes independent selection budget at two")
    if not rows or any(not math.isfinite(value) for _, value in rows):
        raise ValueError("independent singleton scores are invalid")
    scores = dict(rows)
    if len(scores) != len(rows) or tuple(scores) != tuple(sorted(scores)):
        raise ValueError("independent singleton score inventory is not canonical")
    ranked = sorted(
        (event for event, value in rows if value > 0.0),
        key=lambda event: (-scores[event], event),
    )
    return tuple(sorted(ranked[:budget]))


def select_base(
    base: FrozenIndependentBase,
    head: Any,
    state: FeatureState | GateState,
) -> tuple[int, ...]:
    return select_from_scores(predict_state(base, head, state).base_set_scores)


def select_unguarded(
    base: FrozenIndependentBase,
    head: Any,
    state: FeatureState | GateState,
) -> tuple[int, ...]:
    return select_from_scores(predict_state(base, head, state).total_set_scores)


def select_ensemble_base(
    bases: Sequence[FrozenIndependentBase],
    heads: Sequence[Any],
    state: FeatureState | GateState,
) -> tuple[int, ...]:
    return select_from_scores(ensemble_prediction(bases, heads, state).base_set_scores)


def select_ensemble_unguarded(
    bases: Sequence[FrozenIndependentBase],
    heads: Sequence[Any],
    state: FeatureState | GateState,
) -> tuple[int, ...]:
    return select_from_scores(ensemble_prediction(bases, heads, state).total_set_scores)


def assert_zero_residual_identity(
    base: FrozenIndependentBase,
    head: Any,
    states: Sequence[FeatureState | GateState],
) -> Mapping[str, tuple[int, ...]]:
    selections: dict[str, tuple[int, ...]] = {}
    for state in states:
        prediction = predict_state(base, head, state)
        if any(value != 0.0 for _, value in prediction.pair_residual_scores):
            raise ValueError("v4 residual head is not exact-zero")
        exhaustive = select_from_scores(prediction.base_set_scores)
        independent = independent_selection_from_scores(prediction.singleton_scores)
        if exhaustive != independent:
            raise RuntimeError("zero residual does not reproduce independent selection")
        selections[state.state_id] = exhaustive
    return selections


def train_fixed_epochs(
    states: Sequence[GateState],
    *,
    base: FrozenIndependentBase,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> V4FixedEpochFit:
    if type(epochs) is not int or epochs < 0:
        raise ValueError("v4 residual epoch count must be non-negative")
    if seed != base.seed:
        raise ValueError("v4 final residual seed must match its frozen base seed")
    batch = build_training_batch(states)
    head = build_residual_head(seed=seed)
    if epochs == 0:
        return V4FixedEpochFit(head=head, batch=batch, history=())
    optimizer = build_optimizer(head, learning_rate=learning_rate)
    history = tuple(
        V4EpochResult(
            epoch=epoch,
            **optimization_step(base, head, optimizer, batch),
        )
        for epoch in range(1, epochs + 1)
    )
    return V4FixedEpochFit(head=head, batch=batch, history=history)


def serialize_residual_checkpoint(head: Any) -> bytes:
    try:
        from safetensors.torch import load, save
    except ModuleNotFoundError as error:
        raise RuntimeError("v4 residual checkpoint requires safetensors") from error
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in sorted(head.state_dict().items())
    }
    payload = save(tensors)
    replay = load(payload)
    canonical = {name: replay[name].contiguous() for name in sorted(replay)}
    if save(canonical) != payload:
        raise RuntimeError("v4 residual checkpoint is not canonical")
    return payload


def load_residual_checkpoint(
    payload: bytes,
    *,
    seed: int,
    expected_checkpoint_sha256: str,
) -> Any:
    if hashlib.sha256(payload).hexdigest() != expected_checkpoint_sha256:
        raise ValueError("v4 residual checkpoint SHA256 drifted")
    try:
        from safetensors.torch import load, save
    except ModuleNotFoundError as error:
        raise RuntimeError("v4 residual checkpoint requires safetensors") from error
    values = load(payload)
    canonical = {name: values[name].contiguous() for name in sorted(values)}
    if save(canonical) != payload:
        raise ValueError("v4 residual checkpoint encoding is not canonical")
    head = build_residual_head(seed=seed)
    if set(canonical) != set(head.state_dict()):
        raise ValueError("v4 residual checkpoint tensor inventory drifted")
    head.load_state_dict(canonical, strict=True)
    if serialize_residual_checkpoint(head) != payload:
        raise ValueError("v4 residual checkpoint replay changed bytes")
    return head


__all__ = [
    "EXPECTED_BASE_PARAMETER_COUNT",
    "EXPECTED_RESIDUAL_PARAMETER_COUNT",
    "FrozenIndependentBase",
    "RESIDUAL_HIDDEN_DIMENSION",
    "SetConditionedV4ResidualHead",
    "V4EpochResult",
    "V4FixedEpochFit",
    "V4StatePrediction",
    "assert_zero_residual_identity",
    "build_optimizer",
    "build_residual_head",
    "ensemble_prediction",
    "freeze_independent_base",
    "independent_selection_from_scores",
    "load_frozen_independent_base",
    "load_residual_checkpoint",
    "optimization_step",
    "predict_state",
    "replay_fold_clean_base",
    "residual_loss",
    "select_base",
    "select_ensemble_base",
    "select_ensemble_unguarded",
    "select_from_scores",
    "select_unguarded",
    "serialize_residual_checkpoint",
    "train_fixed_epochs",
    "validate_frozen_base",
]
