"""Deterministic CPU training and selection for set-conditioned v3."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.gate_v1_data import INDEPENDENT_INPUT_DIMENSION, FeatureState, GateState
from causalcache.set_conditioned_v3_data import (
    PAIR_FEATURE_DIMENSION,
    V3TargetScale,
    V3TrainingBatch,
    build_training_batch,
    feasible_coalitions,
    v3_independent_input,
)


ENCODER_DIMENSION = 64
PAIR_HIDDEN_DIMENSION = 64
EXPECTED_PARAMETER_COUNT = 33_538
RANKING_COEFFICIENT = 0.25
SMOOTH_L1_BETA = 1.0
WEIGHT_DECAY = 1e-4
GRADIENT_NORM = 1.0
ENSEMBLE_SIZE = 5
SAFE_CONSENSUS_COUNT = 4


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("set-conditioned v3 training requires PyTorch") from error
    return torch


try:
    import torch as _torch_module
except ModuleNotFoundError:
    _torch_module = None


if _torch_module is not None:

    class SetConditionedV3Model(_torch_module.nn.Module):
        """Shared event encoder with singleton and chronological residual heads."""

        def __init__(self) -> None:
            super().__init__()
            self.event_encoder = _torch_module.nn.Sequential(
                _torch_module.nn.Linear(INDEPENDENT_INPUT_DIMENSION, ENCODER_DIMENSION),
                _torch_module.nn.GELU(approximate="none"),
            )
            self.singleton_head = _torch_module.nn.Linear(ENCODER_DIMENSION, 1)
            self.pair_residual_head = _torch_module.nn.Sequential(
                _torch_module.nn.Linear(PAIR_FEATURE_DIMENSION, PAIR_HIDDEN_DIMENSION),
                _torch_module.nn.GELU(approximate="none"),
                _torch_module.nn.Linear(PAIR_HIDDEN_DIMENSION, 1),
            )

        def encode(self, independent_vectors: Any) -> Any:
            return self.event_encoder(independent_vectors)

        def singleton_scores(self, embeddings: Any) -> Any:
            return self.singleton_head(embeddings).squeeze(-1)

        def pair_residual_scores(
            self,
            left_embeddings: Any,
            right_embeddings: Any,
            q64: Any,
        ) -> Any:
            if (
                left_embeddings.shape != right_embeddings.shape
                or left_embeddings.ndim != 2
                or left_embeddings.shape[1] != ENCODER_DIMENSION
                or q64.shape != left_embeddings.shape
            ):
                raise ValueError("v3 pair residual tensor geometry is invalid")
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
                raise RuntimeError("v3 pair feature dimension drifted")
            return self.pair_residual_head(pair_features).squeeze(-1)

else:

    class SetConditionedV3Model:
        """Import placeholder for environments that do not install PyTorch."""

        def __init__(self) -> None:
            _import_torch()


@dataclass(frozen=True)
class V3StatePrediction:
    singleton_scores: tuple[tuple[int, float], ...]
    pair_residual_scores: tuple[tuple[tuple[int, int], float], ...]
    additive_set_utilities: tuple[tuple[tuple[int, ...], float], ...]
    unguarded_set_utilities: tuple[tuple[tuple[int, ...], float], ...]

    def utility(self, coalition: Sequence[int], *, use_residual: bool = True) -> float:
        key = tuple(coalition)
        rows = self.unguarded_set_utilities if use_residual else self.additive_set_utilities
        for candidate, value in rows:
            if candidate == key:
                return value
        raise KeyError(key)

    def to_raw(self, target_scale: V3TargetScale) -> "V3StatePrediction":
        if not isinstance(target_scale, V3TargetScale):
            raise TypeError("target_scale must be a V3TargetScale")
        return V3StatePrediction(
            singleton_scores=tuple(
                (event_id, target_scale.to_raw(value))
                for event_id, value in self.singleton_scores
            ),
            pair_residual_scores=tuple(
                (coalition, target_scale.to_raw(value))
                for coalition, value in self.pair_residual_scores
            ),
            additive_set_utilities=tuple(
                (coalition, target_scale.to_raw(value))
                for coalition, value in self.additive_set_utilities
            ),
            unguarded_set_utilities=tuple(
                (coalition, target_scale.to_raw(value))
                for coalition, value in self.unguarded_set_utilities
            ),
        )


@dataclass(frozen=True)
class V3SafeSelection:
    coalition: tuple[int, ...]
    used_fallback: bool
    pair_candidate: tuple[int, ...]
    agreeing_seed_count: int
    positive_margin_seed_count: int
    seed_margins: tuple[float, ...]
    seed_selections: tuple[tuple[int, ...], ...]
    fallback_coalition: tuple[int, ...]


@dataclass(frozen=True)
class V3EpochResult:
    epoch: int
    singleton: float
    pair: float
    residual: float
    ranking: float
    total: float
    preclip_gradient_norm: float


@dataclass(frozen=True)
class V3FixedEpochFit:
    model: Any
    batch: V3TrainingBatch
    history: tuple[V3EpochResult, ...]


def configure_deterministic_cpu(seed: int) -> Any:
    if type(seed) is not int or seed < 0:
        raise ValueError("v3 seed must be a non-negative integer")
    torch = _import_torch()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError("v3 requires one PyTorch inter-op thread") from error
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return torch


def build_model(*, seed: int) -> SetConditionedV3Model:
    configure_deterministic_cpu(seed)
    model = SetConditionedV3Model()
    model.to(device="cpu", dtype=_import_torch().float32)
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("v3 model parameter count drifted")
    return model


def build_optimizer(model: Any, *, learning_rate: float) -> Any:
    value = float(learning_rate)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("v3 learning rate must be finite and positive")
    torch = _import_torch()
    return torch.optim.AdamW(
        model.parameters(),
        lr=value,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def _forward_batch(
    model: Any, batch: V3TrainingBatch
) -> tuple[Any, Any, Any, dict[tuple[str, tuple[int, ...]], Any]]:
    torch = _import_torch()
    singleton_inputs = torch.tensor(
        [row.input_vector for row in batch.singleton_targets],
        dtype=torch.float32,
        device="cpu",
    )
    embeddings = model.encode(singleton_inputs)
    singleton_predictions = model.singleton_scores(embeddings)
    left_indices = torch.tensor(
        [row.left_singleton_index for row in batch.pair_targets], dtype=torch.long
    )
    right_indices = torch.tensor(
        [row.right_singleton_index for row in batch.pair_targets], dtype=torch.long
    )
    q64 = torch.tensor(
        [row.q64 for row in batch.pair_targets], dtype=torch.float32, device="cpu"
    )
    residual_predictions = model.pair_residual_scores(
        embeddings[left_indices], embeddings[right_indices], q64
    )
    pair_predictions = (
        singleton_predictions[left_indices]
        + singleton_predictions[right_indices]
        + residual_predictions
    )
    lookup: dict[tuple[str, tuple[int, ...]], Any] = {}
    zero = singleton_predictions.sum() * 0.0
    for state in batch.states:
        lookup[(state.state_id, ())] = zero
    for index, row in enumerate(batch.singleton_targets):
        lookup[(row.state_id, (row.event_step_id,))] = singleton_predictions[index]
    for index, row in enumerate(batch.pair_targets):
        lookup[(row.state_id, row.coalition)] = pair_predictions[index]
    return singleton_predictions, pair_predictions, residual_predictions, lookup


def gate_loss(model: Any, batch: V3TrainingBatch) -> tuple[Any, dict[str, float]]:
    torch = _import_torch()
    singleton_predictions, pair_predictions, residual_predictions, lookup = _forward_batch(
        model, batch
    )

    def weighted_smooth_l1(predictions: Any, targets: list[float], weights: list[float]) -> Any:
        target_tensor = torch.tensor(targets, dtype=torch.float32, device="cpu")
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device="cpu")
        rows = torch.nn.functional.smooth_l1_loss(
            predictions,
            target_tensor,
            reduction="none",
            beta=SMOOTH_L1_BETA,
        )
        return torch.sum(rows * weight_tensor)

    singleton = weighted_smooth_l1(
        singleton_predictions,
        [row.normalized_utility for row in batch.singleton_targets],
        [row.weight for row in batch.singleton_targets],
    )
    pair = weighted_smooth_l1(
        pair_predictions,
        [row.normalized_utility for row in batch.pair_targets],
        [row.utility_weight for row in batch.pair_targets],
    )
    residual = weighted_smooth_l1(
        residual_predictions,
        [row.normalized_residual for row in batch.pair_targets],
        [row.residual_weight for row in batch.pair_targets],
    )
    if batch.ranking_pairs:
        left = torch.stack(
            [lookup[(row.state_id, row.left_coalition)] for row in batch.ranking_pairs]
        )
        right = torch.stack(
            [lookup[(row.state_id, row.right_coalition)] for row in batch.ranking_pairs]
        )
        signs = torch.tensor(
            [row.target_sign for row in batch.ranking_pairs], dtype=torch.float32
        )
        weights = torch.tensor(
            [row.weight for row in batch.ranking_pairs], dtype=torch.float32
        )
        ranking = torch.sum(torch.nn.functional.softplus(-(left - right) * signs) * weights)
    else:
        ranking = singleton_predictions.sum() * 0.0
    total = (singleton + pair + residual) / 3.0 + RANKING_COEFFICIENT * ranking
    if not bool(torch.isfinite(total)):
        raise ValueError("v3 loss is non-finite")
    return total, {
        "singleton": float(singleton.detach()),
        "pair": float(pair.detach()),
        "residual": float(residual.detach()),
        "ranking": float(ranking.detach()),
        "total": float(total.detach()),
    }


def optimization_step(
    model: Any, optimizer: Any, batch: V3TrainingBatch
) -> dict[str, float]:
    torch = _import_torch()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, scalars = gate_loss(model, batch)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM)
    if not bool(torch.isfinite(norm)):
        raise ValueError("v3 gradient norm is non-finite")
    optimizer.step()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise ValueError("v3 parameter became non-finite")
    return {**scalars, "preclip_gradient_norm": float(norm)}


def _prediction_from_components(
    state: FeatureState | GateState,
    singleton_values: Sequence[float],
    residual_values: Sequence[float],
) -> V3StatePrediction:
    event_ids = state.candidate_event_step_ids
    pair_ids = tuple(itertools.combinations(event_ids, 2))
    if len(singleton_values) != len(event_ids) or len(residual_values) != len(pair_ids):
        raise ValueError("v3 prediction component count differs from state geometry")
    singleton = tuple(zip(event_ids, map(float, singleton_values), strict=True))
    residual = tuple(zip(pair_ids, map(float, residual_values), strict=True))
    if any(not math.isfinite(value) for _, value in singleton) or any(
        not math.isfinite(value) for _, value in residual
    ):
        raise ValueError("v3 prediction is non-finite")
    singleton_by_event = dict(singleton)
    residual_by_pair = dict(residual)
    additive: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    additive.extend(((event_id,), value) for event_id, value in singleton)
    additive.extend(
        (pair, singleton_by_event[pair[0]] + singleton_by_event[pair[1]])
        for pair in pair_ids
    )
    unguarded = tuple(
        (coalition, value if len(coalition) < 2 else value + residual_by_pair[coalition])
        for coalition, value in additive
    )
    return V3StatePrediction(
        singleton_scores=singleton,
        pair_residual_scores=residual,
        additive_set_utilities=tuple(additive),
        unguarded_set_utilities=unguarded,
    )


def predict_state(
    model: Any, state: FeatureState | GateState
) -> V3StatePrediction:
    feasible_coalitions(state)
    torch = _import_torch()
    model.eval()
    event_ids = state.candidate_event_step_ids
    pair_ids = tuple(itertools.combinations(event_ids, 2))
    with torch.no_grad():
        inputs = torch.tensor(
            [v3_independent_input(state, event_id) for event_id in event_ids],
            dtype=torch.float32,
            device="cpu",
        )
        embeddings = model.encode(inputs)
        singleton = model.singleton_scores(embeddings)
        left_indices = torch.tensor(
            [event_ids.index(pair[0]) for pair in pair_ids], dtype=torch.long
        )
        right_indices = torch.tensor(
            [event_ids.index(pair[1]) for pair in pair_ids], dtype=torch.long
        )
        q64 = torch.tensor(
            [state.q64 for _ in pair_ids], dtype=torch.float32, device="cpu"
        )
        residual = model.pair_residual_scores(
            embeddings[left_indices], embeddings[right_indices], q64
        )
    return _prediction_from_components(state, singleton.tolist(), residual.tolist())


def select_from_utilities(
    rows: Sequence[tuple[tuple[int, ...], float]],
) -> tuple[int, ...]:
    if not rows or any(not math.isfinite(value) for _, value in rows):
        raise ValueError("v3 set utility rows must be non-empty and finite")
    coalitions = [coalition for coalition, _ in rows]
    if len(set(coalitions)) != len(coalitions):
        raise ValueError("v3 set utility rows contain duplicate coalitions")
    coalition, _ = min(rows, key=lambda row: (-row[1], len(row[0]), row[0]))
    return coalition


def select_additive(
    model: Any, state: FeatureState | GateState
) -> tuple[int, ...]:
    return select_from_utilities(predict_state(model, state).additive_set_utilities)


def select_unguarded(
    model: Any, state: FeatureState | GateState
) -> tuple[int, ...]:
    return select_from_utilities(predict_state(model, state).unguarded_set_utilities)


def _mean_prediction(
    predictions: Sequence[V3StatePrediction],
    state: FeatureState | GateState,
) -> V3StatePrediction:
    predictions = tuple(predictions)
    if not predictions:
        raise ValueError("v3 prediction ensemble cannot be empty")
    singleton = tuple(
        math.fsum(dict(prediction.singleton_scores)[event_id] for prediction in predictions)
        / len(predictions)
        for event_id in state.candidate_event_step_ids
    )
    pair_ids = tuple(itertools.combinations(state.candidate_event_step_ids, 2))
    residual = tuple(
        math.fsum(dict(prediction.pair_residual_scores)[pair] for prediction in predictions)
        / len(predictions)
        for pair in pair_ids
    )
    return _prediction_from_components(state, singleton, residual)


def ensemble_prediction(
    models: Sequence[Any], state: FeatureState | GateState
) -> V3StatePrediction:
    members = tuple(models)
    if not members:
        raise ValueError("v3 ensemble cannot be empty")
    return _mean_prediction(
        tuple(predict_state(model, state) for model in members), state
    )


def select_ensemble_additive(
    models: Sequence[Any], state: FeatureState | GateState
) -> tuple[int, ...]:
    return select_from_utilities(ensemble_prediction(models, state).additive_set_utilities)


def select_ensemble_unguarded(
    models: Sequence[Any], state: FeatureState | GateState
) -> tuple[int, ...]:
    return select_from_utilities(ensemble_prediction(models, state).unguarded_set_utilities)


def select_ensemble_safe(
    models: Sequence[Any],
    state: FeatureState | GateState,
    *,
    consensus_count: int = SAFE_CONSENSUS_COUNT,
) -> V3SafeSelection:
    members = tuple(models)
    if len(members) != ENSEMBLE_SIZE:
        raise ValueError("safe v3 selection requires exactly five seed models")
    if type(consensus_count) is not int or consensus_count != SAFE_CONSENSUS_COUNT:
        raise ValueError("safe v3 selection freezes consensus at four of five")
    predictions = tuple(predict_state(model, state) for model in members)
    ensemble = _mean_prediction(predictions, state)
    pair_candidate = select_from_utilities(ensemble.unguarded_set_utilities)
    fallback = select_from_utilities(ensemble.additive_set_utilities)
    seed_selections = tuple(
        select_from_utilities(prediction.unguarded_set_utilities)
        for prediction in predictions
    )
    agreement = sum(
        selection == pair_candidate for selection in seed_selections
    )
    seed_margins = tuple(
        prediction.utility(pair_candidate)
        - prediction.utility(fallback)
        for prediction in predictions
    )
    if any(not math.isfinite(value) for value in seed_margins):
        raise ValueError("safe v3 seed margin is non-finite")
    positive_margin_count = sum(value > 0.0 for value in seed_margins)
    same_selection = pair_candidate == fallback
    use_pair_candidate = same_selection or (
        agreement >= consensus_count
        and positive_margin_count >= consensus_count
    )
    return V3SafeSelection(
        coalition=pair_candidate if use_pair_candidate else fallback,
        used_fallback=not use_pair_candidate,
        pair_candidate=pair_candidate,
        agreeing_seed_count=agreement,
        positive_margin_seed_count=positive_margin_count,
        seed_margins=seed_margins,
        seed_selections=seed_selections,
        fallback_coalition=fallback,
    )


def train_fixed_epochs(
    states: Sequence[GateState],
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    target_scale: V3TargetScale | None = None,
) -> V3FixedEpochFit:
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("v3 epoch count must be a positive integer")
    batch = build_training_batch(states, target_scale=target_scale)
    model = build_model(seed=seed)
    optimizer = build_optimizer(model, learning_rate=learning_rate)
    history: list[V3EpochResult] = []
    for epoch in range(1, epochs + 1):
        scalars = optimization_step(model, optimizer, batch)
        history.append(V3EpochResult(epoch=epoch, **scalars))
    return V3FixedEpochFit(model=model, batch=batch, history=tuple(history))


__all__ = [
    "ENCODER_DIMENSION",
    "ENSEMBLE_SIZE",
    "EXPECTED_PARAMETER_COUNT",
    "PAIR_HIDDEN_DIMENSION",
    "RANKING_COEFFICIENT",
    "SAFE_CONSENSUS_COUNT",
    "SetConditionedV3Model",
    "V3EpochResult",
    "V3FixedEpochFit",
    "V3SafeSelection",
    "V3StatePrediction",
    "build_model",
    "build_optimizer",
    "configure_deterministic_cpu",
    "ensemble_prediction",
    "gate_loss",
    "optimization_step",
    "predict_state",
    "select_additive",
    "select_ensemble_additive",
    "select_ensemble_safe",
    "select_ensemble_unguarded",
    "select_from_utilities",
    "select_unguarded",
    "train_fixed_epochs",
]
