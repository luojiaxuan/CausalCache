"""Deterministic CPU trainer for the frozen CausalCache gate v1 contract."""

from __future__ import annotations

import io
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.gate_v1_contract import (
    EXPECTED_ASSIGNMENT_DIGEST,
    EXPECTED_FOLD_DIGESTS,
    canonical_json_bytes,
    derive_oof_folds,
    sha256_bytes,
)
from causalcache.gate_v1_data import (
    CONDITIONAL_INPUT_DIMENSION,
    INDEPENDENT_INPUT_DIMENSION,
    GateState,
    TrainingBatch,
    build_training_batch,
    conditional_input,
    independent_input,
    select_conditional,
    select_independent,
    validate_canonical_gate_state_roster,
)
from causalcache.restoration_v2_2_label_table import primary_exact_subset_oracle


FAMILIES = ("conditional", "independent")
LEARNING_RATES = (0.0003, 0.001)
SEEDS = (0, 1, 2, 3, 4)
MAXIMUM_EPOCHS = 500
OOF_PATIENCE_EPOCHS = 50
EPOCH_IMPROVEMENT_EPSILON = 1e-4
RANKING_COEFFICIENT = 0.25
SMOOTH_L1_BETA = 1.0
WEIGHT_DECAY = 1e-4
GRADIENT_NORM = 1.0


@dataclass(frozen=True)
class OOFTrial:
    family: str
    learning_rate: float
    seed: int
    selected_epoch: int
    best_raw_utility_ratio: float
    epochs_run: int
    metric_by_epoch: tuple[float, ...]


@dataclass(frozen=True)
class FamilySelection:
    family: str
    learning_rate: float
    five_seed_mean_oof_ratio: float
    trials: tuple[OOFTrial, ...]
    grid_trials: tuple[OOFTrial, ...]


@dataclass(frozen=True)
class FittedEnsemble:
    family: str
    learning_rate: float
    seeds: tuple[int, ...]
    selected_epochs: tuple[int, ...]
    selection_sha256: str
    models: tuple[Any, ...]


def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "gate v1 training requires PyTorch; feature/schema validation does not"
        ) from error
    return torch


def configure_deterministic_cpu(seed: int) -> Any:
    if type(seed) is not int or seed < 0:
        raise ValueError("gate seed must be a non-negative integer")
    torch = _torch()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError(
                "gate v1 requires one PyTorch inter-op thread"
            ) from error
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return torch


def build_model(family: str, *, seed: int) -> Any:
    if family not in FAMILIES:
        raise ValueError("unknown gate family")
    torch = configure_deterministic_cpu(seed)
    input_dimension, hidden_dimension = (
        (CONDITIONAL_INPUT_DIMENSION, 64)
        if family == "conditional"
        else (INDEPENDENT_INPUT_DIMENSION, 88)
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dimension, hidden_dimension),
        torch.nn.GELU(approximate="none"),
        torch.nn.Linear(hidden_dimension, hidden_dimension),
        torch.nn.GELU(approximate="none"),
        torch.nn.Linear(hidden_dimension, 1),
    )
    model.to(device="cpu", dtype=torch.float32)
    expected = 25_409 if family == "conditional" else 25_609
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != expected:
        raise RuntimeError("gate model parameter count drifted")
    return model


def _batch_tensors(batch: TrainingBatch) -> tuple[Any, Any, Any]:
    torch = _torch()
    inputs = torch.tensor(
        [example.input_vector for example in batch.examples],
        dtype=torch.float32,
        device="cpu",
    )
    targets = torch.tensor(
        [
            0.0 if example.normalized_target is None else example.normalized_target
            for example in batch.examples
        ],
        dtype=torch.float32,
        device="cpu",
    )
    weights = torch.tensor(
        [example.regression_weight for example in batch.examples],
        dtype=torch.float32,
        device="cpu",
    )
    if inputs.shape != (len(batch.examples), batch.input_dimension):
        raise RuntimeError("gate input tensor shape drifted")
    return inputs, targets, weights


def gate_loss(model: Any, batch: TrainingBatch) -> tuple[Any, dict[str, float]]:
    torch = _torch()
    inputs, targets, weights = _batch_tensors(batch)
    predictions = model(inputs).squeeze(-1)
    regression_rows = torch.nn.functional.smooth_l1_loss(
        predictions,
        targets,
        reduction="none",
        beta=SMOOTH_L1_BETA,
    )
    regression = torch.sum(regression_rows * weights)
    if batch.ranking_pairs:
        left = torch.tensor(
            [pair.left_index for pair in batch.ranking_pairs], dtype=torch.long
        )
        right = torch.tensor(
            [pair.right_index for pair in batch.ranking_pairs], dtype=torch.long
        )
        signs = torch.tensor(
            [pair.target_sign for pair in batch.ranking_pairs], dtype=torch.float32
        )
        pair_weights = torch.tensor(
            [pair.weight for pair in batch.ranking_pairs], dtype=torch.float32
        )
        ranking_rows = torch.nn.functional.softplus(
            -(predictions[left] - predictions[right]) * signs
        )
        ranking = torch.sum(ranking_rows * pair_weights)
    else:
        ranking = predictions.sum() * 0.0
    total = regression + RANKING_COEFFICIENT * ranking
    if not bool(torch.isfinite(total)):
        raise ValueError("gate loss is non-finite")
    return total, {
        "regression": float(regression.detach()),
        "ranking": float(ranking.detach()),
        "total": float(total.detach()),
    }


def build_optimizer(model: Any, *, learning_rate: float) -> Any:
    if learning_rate not in LEARNING_RATES:
        raise ValueError("learning rate is outside the frozen candidate set")
    torch = _torch()
    return torch.optim.AdamW(
        model.parameters(),
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


def optimization_step(model: Any, optimizer: Any, batch: TrainingBatch) -> dict[str, float]:
    torch = _torch()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, scalars = gate_loss(model, batch)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM)
    if not bool(torch.isfinite(norm)):
        raise ValueError("gate gradient norm is non-finite")
    optimizer.step()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise ValueError("gate parameter became non-finite")
    return {**scalars, "preclip_gradient_norm": float(norm)}


def predict_vectors(model: Any, vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        return ()
    torch = _torch()
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(vectors, dtype=torch.float32, device="cpu")
        predictions = model(inputs).squeeze(-1)
    values = tuple(float(value) for value in predictions.tolist())
    if any(not math.isfinite(value) for value in values):
        raise ValueError("gate prediction is non-finite")
    return values


def model_score(model: Any, family: str):
    def score(state: GateState, event_step_id: int, coalition: tuple[int, ...]) -> float:
        vector = (
            conditional_input(state, event_step_id, coalition)
            if family == "conditional"
            else independent_input(state, event_step_id)
        )
        return predict_vectors(model, (vector,))[0]

    return score


def ensemble_score(ensemble: FittedEnsemble):
    if len(ensemble.models) != 5 or ensemble.seeds != SEEDS:
        raise ValueError("formal gate ensemble must contain the five frozen seeds")
    scorers = tuple(model_score(model, ensemble.family) for model in ensemble.models)

    def score(state: GateState, event_step_id: int, coalition: tuple[int, ...]) -> float:
        return math.fsum(
            scorer(state, event_step_id, coalition) for scorer in scorers
        ) / len(scorers)

    return score


def select_with_model(state: GateState, model: Any, family: str) -> tuple[int, ...]:
    scorer = model_score(model, family)
    if family == "conditional":
        return select_conditional(state, scorer)[0]
    return select_independent(state, scorer)


def batched_model_selections(
    states: Sequence[GateState], model: Any, family: str
) -> dict[str, tuple[int, ...]]:
    """Replay deployment selection while batching candidates within each round."""
    if family not in FAMILIES:
        raise ValueError("unknown gate family")
    if family == "independent":
        keys = [
            (state, event)
            for state in states
            for event in state.candidate_event_step_ids
        ]
        values = predict_vectors(
            model,
            tuple(independent_input(state, event) for state, event in keys),
        )
        scores: dict[str, dict[int, float]] = {}
        for (state, event), value in zip(keys, values, strict=True):
            scores.setdefault(state.state_id, {})[event] = value
        return {
            state.state_id: tuple(
                sorted(
                    sorted(
                        (
                            event
                            for event, value in scores[state.state_id].items()
                            if value > 0.0
                        ),
                        key=lambda event: (-scores[state.state_id][event], event),
                    )[:2]
                )
            )
            for state in states
        }

    selected = {state.state_id: () for state in states}
    active = list(states)
    for _ in range(2):
        keys = [
            (state, event, selected[state.state_id])
            for state in active
            for event in state.candidate_event_step_ids
            if event not in selected[state.state_id]
        ]
        values = predict_vectors(
            model,
            tuple(
                conditional_input(state, event, coalition)
                for state, event, coalition in keys
            ),
        )
        scores: dict[str, list[tuple[int, float]]] = {}
        for (state, event, _), value in zip(keys, values, strict=True):
            scores.setdefault(state.state_id, []).append((event, value))
        next_active = []
        for state in active:
            event, value = min(
                scores[state.state_id], key=lambda item: (-item[1], item[0])
            )
            if value > 0.0:
                selected[state.state_id] = tuple(
                    sorted((*selected[state.state_id], event))
                )
                next_active.append(state)
        active = next_active
        if not active:
            break
    return selected


def _n4_states(states: Sequence[GateState]) -> tuple[GateState, ...]:
    by_source: dict[str, list[GateState]] = {}
    for state in states:
        by_source.setdefault(state.source_id, []).append(state)
    selected: list[GateState] = []
    for source_id, source_states in by_source.items():
        matching = [state for state in source_states if len(state.candidate_event_step_ids) == 4]
        if len(matching) != 1:
            raise ValueError(f"OOF source {source_id} must contain exactly one n=4 state")
        selected.extend(matching)
    return tuple(selected)


def validate_formal_training_roster(
    states: Sequence[GateState], expected_source_ids: Sequence[str]
) -> tuple[tuple[str, ...], ...]:
    validate_canonical_gate_state_roster(
        states,
        expected_source_ids,
        expected_source_count=58,
    )
    source_order = tuple(expected_source_ids)
    assignments, folds = derive_oof_folds(source_order)
    fold_digests = tuple(
        sha256_bytes(canonical_json_bytes(list(fold))) for fold in folds
    )
    assignment_payload = [
        {"source_id": source_id, "fold": fold}
        for source_id, fold in assignments
    ]
    if (
        fold_digests != EXPECTED_FOLD_DIGESTS
        or sha256_bytes(canonical_json_bytes(assignment_payload))
        != EXPECTED_ASSIGNMENT_DIGEST
    ):
        raise ValueError("formal OOF fold identities differ from preregistration")
    return folds


def raw_utility_ratio(
    states: Sequence[GateState],
    selections: Mapping[str, Sequence[int]],
) -> float:
    numerator = math.fsum(
        state.table.utility(selections[state.state_id]) for state in states
    )
    denominator = math.fsum(
        primary_exact_subset_oracle(state.table).utility for state in states
    )
    if not math.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError("exact raw-utility denominator is not strictly positive")
    value = numerator / denominator
    if not math.isfinite(value):
        raise ValueError("raw utility ratio is non-finite")
    return value


def run_oof_trial(
    states: Sequence[GateState],
    *,
    family: str,
    learning_rate: float,
    seed: int,
    folds: Sequence[Sequence[str]],
    maximum_epochs: int = MAXIMUM_EPOCHS,
    patience_epochs: int = OOF_PATIENCE_EPOCHS,
) -> OOFTrial:
    """Train all fold models in lockstep and score the held-out n=4 rollout each epoch."""
    if family not in FAMILIES or learning_rate not in LEARNING_RATES or seed not in SEEDS:
        raise ValueError("OOF trial parameters differ from the frozen grid")
    if maximum_epochs <= 0 or patience_epochs <= 0:
        raise ValueError("OOF epoch limits must be positive")
    all_sources = tuple(dict.fromkeys(state.source_id for state in states))
    flattened = tuple(source_id for fold in folds for source_id in fold)
    if (
        len(folds) != 5
        or len(flattened) != len(set(flattened))
        or set(flattened) != set(all_sources)
    ):
        raise ValueError("OOF folds must partition the training trajectories exactly")

    fold_models = []
    fold_optimizers = []
    fold_batches = []
    fold_eval_states = []
    for heldout in folds:
        heldout_set = set(heldout)
        training = tuple(state for state in states if state.source_id not in heldout_set)
        evaluation = _n4_states(
            tuple(state for state in states if state.source_id in heldout_set)
        )
        model = build_model(family, seed=seed)
        fold_models.append(model)
        fold_optimizers.append(build_optimizer(model, learning_rate=learning_rate))
        fold_batches.append(build_training_batch(training, family=family))
        fold_eval_states.append(evaluation)

    metrics: list[float] = []
    best_metric = -math.inf
    best_epoch = 0
    for epoch in range(1, maximum_epochs + 1):
        for model, optimizer, batch in zip(
            fold_models, fold_optimizers, fold_batches, strict=True
        ):
            optimization_step(model, optimizer, batch)
        all_evaluation: list[GateState] = []
        selections: dict[str, tuple[int, ...]] = {}
        for model, evaluation in zip(fold_models, fold_eval_states, strict=True):
            all_evaluation.extend(evaluation)
            selections.update(batched_model_selections(evaluation, model, family))
        metric = raw_utility_ratio(all_evaluation, selections)
        metrics.append(metric)
        if best_epoch == 0 or metric > best_metric + EPOCH_IMPROVEMENT_EPSILON:
            best_metric = metric
            best_epoch = epoch
        if epoch - best_epoch >= patience_epochs:
            break
    if best_epoch == 0:
        raise RuntimeError("OOF trial failed to select an epoch")
    return OOFTrial(
        family=family,
        learning_rate=learning_rate,
        seed=seed,
        selected_epoch=best_epoch,
        best_raw_utility_ratio=best_metric,
        epochs_run=len(metrics),
        metric_by_epoch=tuple(metrics),
    )


def _validate_oof_trial(trial: OOFTrial) -> None:
    if (
        trial.family not in FAMILIES
        or trial.learning_rate not in LEARNING_RATES
        or trial.seed not in SEEDS
        or type(trial.selected_epoch) is not int
        or type(trial.epochs_run) is not int
        or not 1 <= trial.selected_epoch <= trial.epochs_run <= MAXIMUM_EPOCHS
        or len(trial.metric_by_epoch) != trial.epochs_run
        or not math.isfinite(trial.best_raw_utility_ratio)
        or any(not math.isfinite(value) for value in trial.metric_by_epoch)
    ):
        raise ValueError("OOF trial metadata is malformed or non-finite")
    best_metric = -math.inf
    best_epoch = 0
    expected_stop_epoch: int | None = None
    for epoch, metric in enumerate(trial.metric_by_epoch, start=1):
        if best_epoch == 0 or metric > best_metric + EPOCH_IMPROVEMENT_EPSILON:
            best_metric = metric
            best_epoch = epoch
        if epoch - best_epoch >= OOF_PATIENCE_EPOCHS:
            expected_stop_epoch = epoch
            break
    expected_epochs_run = (
        expected_stop_epoch if expected_stop_epoch is not None else MAXIMUM_EPOCHS
    )
    if trial.epochs_run != expected_epochs_run:
        raise ValueError("OOF trial does not replay frozen patience or maximum epochs")
    if trial.selected_epoch != best_epoch or trial.best_raw_utility_ratio != best_metric:
        raise ValueError("OOF trial does not replay the frozen earliest-epoch rule")


def family_selection_payload(selection: FamilySelection) -> dict[str, Any]:
    _validate_family_selection(selection)

    def trial_payload(trial: OOFTrial) -> dict[str, Any]:
        return {
            "family": trial.family,
            "learning_rate": trial.learning_rate,
            "seed": trial.seed,
            "selected_epoch": trial.selected_epoch,
            "best_raw_utility_ratio": trial.best_raw_utility_ratio,
            "epochs_run": trial.epochs_run,
            "metric_by_epoch": list(trial.metric_by_epoch),
        }

    return {
        "family": selection.family,
        "learning_rate": selection.learning_rate,
        "five_seed_mean_oof_ratio": selection.five_seed_mean_oof_ratio,
        "selected_trials": [trial_payload(trial) for trial in selection.trials],
        "grid_trials": [trial_payload(trial) for trial in selection.grid_trials],
    }


def family_selection_sha256(selection: FamilySelection) -> str:
    return sha256_bytes(canonical_json_bytes(family_selection_payload(selection)))


def _validate_family_selection(selection: FamilySelection) -> None:
    if selection.family not in FAMILIES or selection.learning_rate not in LEARNING_RATES:
        raise ValueError("family selection metadata is invalid")
    if len(selection.grid_trials) != len(LEARNING_RATES) * len(SEEDS):
        raise ValueError("family selection does not retain the complete frozen OOF grid")
    for trial in selection.grid_trials:
        _validate_oof_trial(trial)
    expected_grid = tuple(
        (learning_rate, seed) for learning_rate in LEARNING_RATES for seed in SEEDS
    )
    observed_grid = tuple(
        (trial.learning_rate, trial.seed) for trial in selection.grid_trials
    )
    if (
        observed_grid != expected_grid
        or any(trial.family != selection.family for trial in selection.grid_trials)
    ):
        raise ValueError("family selection OOF grid order or family drifted")
    expected_trials = tuple(
        trial
        for trial in selection.grid_trials
        if trial.learning_rate == selection.learning_rate
    )
    if selection.trials != expected_trials:
        raise ValueError("family selection seed trials differ from the selected learning rate")
    expected_mean = statistics.fmean(
        trial.best_raw_utility_ratio for trial in expected_trials
    )
    if (
        not math.isfinite(selection.five_seed_mean_oof_ratio)
        or selection.five_seed_mean_oof_ratio != expected_mean
    ):
        raise ValueError("family selection five-seed mean does not replay")
    means = {
        learning_rate: statistics.fmean(
            trial.best_raw_utility_ratio
            for trial in selection.grid_trials
            if trial.learning_rate == learning_rate
        )
        for learning_rate in LEARNING_RATES
    }
    expected_learning_rate = (
        LEARNING_RATES[0]
        if abs(means[LEARNING_RATES[0]] - means[LEARNING_RATES[1]]) <= 1e-4
        else max(LEARNING_RATES, key=lambda value: means[value])
    )
    if selection.learning_rate != expected_learning_rate:
        raise ValueError("family selection learning rate does not replay the frozen rule")


def choose_learning_rate(trials: Sequence[OOFTrial], *, family: str) -> FamilySelection:
    for trial in trials:
        _validate_oof_trial(trial)
    expected = {(learning_rate, seed) for learning_rate in LEARNING_RATES for seed in SEEDS}
    observed = {
        (trial.learning_rate, trial.seed)
        for trial in trials
        if trial.family == family
    }
    if observed != expected or len(trials) != len(expected):
        raise ValueError("learning-rate selection requires the complete frozen grid")
    means = {
        learning_rate: statistics.fmean(
            trial.best_raw_utility_ratio
            for trial in trials
            if trial.learning_rate == learning_rate
        )
        for learning_rate in LEARNING_RATES
    }
    if abs(means[LEARNING_RATES[0]] - means[LEARNING_RATES[1]]) <= 1e-4:
        selected = LEARNING_RATES[0]
    else:
        selected = max(LEARNING_RATES, key=lambda value: means[value])
    selected_trials = tuple(
        sorted(
            (trial for trial in trials if trial.learning_rate == selected),
            key=lambda trial: trial.seed,
        )
    )
    grid_trials = tuple(
        sorted(
            trials,
            key=lambda trial: (LEARNING_RATES.index(trial.learning_rate), trial.seed),
        )
    )
    selection = FamilySelection(
        family=family,
        learning_rate=selected,
        five_seed_mean_oof_ratio=means[selected],
        trials=selected_trials,
        grid_trials=grid_trials,
    )
    _validate_family_selection(selection)
    return selection


def run_formal_oof(
    states: Sequence[GateState],
    *,
    family: str,
    expected_source_ids: Sequence[str],
) -> FamilySelection:
    """Run the immutable 2-LR x 5-seed train-only OOF selection."""
    folds = validate_formal_training_roster(states, expected_source_ids)
    if tuple(len(fold) for fold in folds) != (12, 12, 12, 11, 11):
        raise ValueError("formal OOF fold sizes drifted")
    trials = tuple(
        run_oof_trial(
            states,
            family=family,
            learning_rate=learning_rate,
            seed=seed,
            folds=folds,
        )
        for learning_rate in LEARNING_RATES
        for seed in SEEDS
    )
    return choose_learning_rate(trials, family=family)


def fit_final_ensemble(
    states: Sequence[GateState],
    selection: FamilySelection,
    *,
    expected_source_ids: Sequence[str],
) -> FittedEnsemble:
    validate_formal_training_roster(states, expected_source_ids)
    _validate_family_selection(selection)
    if (
        selection.family not in FAMILIES
        or selection.learning_rate not in LEARNING_RATES
        or len(selection.trials) != 5
        or tuple(trial.seed for trial in selection.trials) != SEEDS
        or any(
            trial.family != selection.family
            or trial.learning_rate != selection.learning_rate
            or trial.selected_epoch <= 0
            or trial.selected_epoch > MAXIMUM_EPOCHS
            for trial in selection.trials
        )
    ):
        raise ValueError("final-fit selection differs from the frozen OOF output")
    batch = build_training_batch(states, family=selection.family)
    models = []
    epochs = []
    for trial in selection.trials:
        model = build_model(selection.family, seed=trial.seed)
        optimizer = build_optimizer(model, learning_rate=selection.learning_rate)
        for _ in range(trial.selected_epoch):
            optimization_step(model, optimizer, batch)
        models.append(model)
        epochs.append(trial.selected_epoch)
    return FittedEnsemble(
        family=selection.family,
        learning_rate=selection.learning_rate,
        seeds=tuple(trial.seed for trial in selection.trials),
        selected_epochs=tuple(epochs),
        selection_sha256=family_selection_sha256(selection),
        models=tuple(models),
    )


def _model_snapshot(model: Any) -> tuple[tuple[str, tuple[float, ...]], ...]:
    return tuple(
        (name, tuple(float(value) for value in tensor.detach().reshape(-1).tolist()))
        for name, tensor in model.state_dict().items()
    )


def _two_step_replay(
    states: Sequence[GateState], family: str
) -> tuple[Any, tuple[dict[str, float], ...]]:
    model = build_model(family, seed=0)
    optimizer = build_optimizer(model, learning_rate=LEARNING_RATES[0])
    batch = build_training_batch(states, family=family)
    scalars = tuple(optimization_step(model, optimizer, batch) for _ in range(2))
    return model, scalars


def run_synthetic_two_step_smoke(states: Sequence[GateState]) -> dict[str, Any]:
    """Exercise the frozen trainer without reading or retaining any formal artifact."""
    if not states or len(tuple(dict.fromkeys(state.source_id for state in states))) > 10:
        raise ValueError("trainer smoke accepts only a bounded synthetic/legacy-sized roster")
    reports: dict[str, Any] = {}
    for family in FAMILIES:
        first, first_scalars = _two_step_replay(states, family)
        second, second_scalars = _two_step_replay(states, family)
        if first_scalars != second_scalars or _model_snapshot(first) != _model_snapshot(second):
            raise RuntimeError("two-step deterministic replay differs")
        batch = build_training_batch(states, family=family)
        if not any(example.raw_target < 0.0 for example in batch.examples):
            raise ValueError("smoke fixture must retain at least one negative target")
        if family == "conditional":
            targets_by_state_event: dict[tuple[str, int], set[float]] = {}
            for example in batch.examples:
                targets_by_state_event.setdefault(
                    (example.state_id, example.event_step_id), set()
                ).add(example.raw_target)
            if not any(len(values) > 1 for values in targets_by_state_event.values()):
                raise ValueError(
                    "smoke fixture must exercise a coalition-dependent marginal target"
                )
        probe_vectors = tuple(example.input_vector for example in batch.examples[:4])
        before = predict_vectors(first, probe_vectors)
        buffer = io.BytesIO()
        torch = _torch()
        torch.save(first.state_dict(), buffer)
        buffer.seek(0)
        reloaded = build_model(family, seed=0)
        reloaded.load_state_dict(
            torch.load(buffer, map_location="cpu", weights_only=True)
        )
        after = predict_vectors(reloaded, probe_vectors)
        if before != after:
            raise RuntimeError("temporary checkpoint reload changed exact scores")
        batched = batched_model_selections(states, first, family)
        scalar = {
            state.state_id: select_with_model(state, first, family)
            for state in states
        }
        if batched != scalar:
            raise RuntimeError("batched and scalar deployment selection differ")
        reports[family] = {
            "input_dimension": batch.input_dimension,
            "regression_weight_sum": batch.regression_weight_sum,
            "ranking_pair_count": len(batch.ranking_pairs),
            "optimization_steps": 2,
            "finite_forward_backward": True,
            "negative_target_not_clipped": True,
            "interaction_target_exercised": family == "conditional",
            "deterministic_two_step_replay": True,
            "temporary_checkpoint_save_load_exact_score": True,
            "batched_scalar_selection_exact_replay": True,
        }

    probe = max(states, key=lambda state: len(state.candidate_event_step_ids))
    calls: list[tuple[int, tuple[int, ...]]] = []

    def conditional_probe(
        _state: GateState, event: int, coalition: tuple[int, ...]
    ) -> float:
        calls.append((event, coalition))
        if not coalition:
            return {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.5}.get(event, -1.0)
        return 2.0 if event == 3 else -1.0

    selected, _ = select_conditional(probe, conditional_probe)
    if selected != (1, 3) or not any(coalition == (1,) for _, coalition in calls):
        raise RuntimeError("conditional selector did not rescore after its first choice")
    stopped, _ = select_conditional(probe, lambda _state, _event, _coalition: 0.0)
    if stopped:
        raise RuntimeError("conditional selector violated the nonpositive stop rule")
    independent_calls = 0

    def independent_probe(_state: GateState, event: int, coalition: tuple[int, ...]) -> float:
        nonlocal independent_calls
        independent_calls += 1
        if coalition:
            raise RuntimeError("independent selector attempted set-conditioned rescoring")
        return float(event)

    select_independent(probe, independent_probe)
    if independent_calls != len(probe.candidate_event_step_ids):
        raise RuntimeError("independent selector was not one-shot")
    return {
        "status": "VALID_GATE_V1_SYNTHETIC_TWO_STEP_SMOKE",
        "paper_metric_count": 0,
        "persistent_checkpoint_count": 0,
        "development_semantic_access_count": 0,
        "conditional_rescoring": True,
        "tau0_stop": True,
        "independent_one_shot": True,
        "families": reports,
    }


__all__ = [
    "FAMILIES",
    "FittedEnsemble",
    "FamilySelection",
    "LEARNING_RATES",
    "OOFTrial",
    "SEEDS",
    "build_model",
    "batched_model_selections",
    "choose_learning_rate",
    "ensemble_score",
    "family_selection_payload",
    "family_selection_sha256",
    "fit_final_ensemble",
    "gate_loss",
    "model_score",
    "optimization_step",
    "predict_vectors",
    "raw_utility_ratio",
    "run_formal_oof",
    "run_oof_trial",
    "run_synthetic_two_step_smoke",
    "select_with_model",
    "validate_formal_training_roster",
]
