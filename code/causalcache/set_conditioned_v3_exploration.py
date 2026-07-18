"""CPU-only orchestration for the non-mainline set-conditioned v3 study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.gate_v1_contract import canonical_json_bytes, derive_oof_folds
from causalcache.gate_v1_data import (
    DEPLOYMENT_BUDGET,
    NORMALIZATION_EPSILON,
    FeatureState,
    GateState,
    join_feature_and_label_states,
    validate_canonical_gate_state_roster,
)
from causalcache.gate_v1_formal_cache import read_feature_cache, read_label_cache
from causalcache.gate_v1_formal_train import (
    FEATURE_CACHE_SHA256,
    FORMAL_JOIN_AUDIT_SHA256,
    LABEL_CACHE_SHA256,
)
from causalcache.gate_v1_fresh16 import (
    FRESH_STATE_COUNT,
    LEARNED_STATUS as FRESH_LEARNED_STATUS,
    PROTOCOL_ID as FRESH_PROTOCOL_ID,
    read_feature_states_jsonl,
    read_label_states_jsonl,
)
from causalcache.gate_v1_provenance import canonical_selection_sha256
from causalcache.restoration_v2_2_label_table import primary_exact_subset_oracle


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_conditioned_v3_pair_residual_exploration_v1"
SEEDS = (0, 1, 2, 3, 4)
LEARNING_RATES = (0.0003, 0.001)
MAXIMUM_EPOCHS = 500
PATIENCE_EPOCHS = 50
IMPROVEMENT_EPSILON = 1e-4
CONSENSUS_MINIMUM = 4
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 271_828
BOOTSTRAP_CONFIDENCE = 0.90
FRESH_FEATURE_SHA256 = (
    "deacc63480ed706d44e7690d974250860d1bc3e5967d4c2a3315090fe4b93939"
)
FRESH_LABEL_SHA256 = (
    "832096b98011264a6b7fee74b2e78798876ea0d5ffc5c9c9baf7642be8e92382"
)
HISTORICAL_INDEPENDENT_SHA256 = (
    "2359c0f4b03e7009291c8912bea7193a4dfebd5f01d4876efd312bc595404dd2"
)
METHODS = (
    "exact",
    "oracle_additive",
    "v3_additive",
    "v3_unguarded",
    "v3_safe",
    "historical_v1_independent",
)


@dataclass(frozen=True)
class OOFTrial:
    learning_rate: float
    seed: int
    selected_epoch: int
    selected_score: float
    selected_raw_ratio: float
    selected_normalized_ratio: float
    epochs_run: int
    score_by_epoch: tuple[float, ...]
    raw_ratio_by_epoch: tuple[float, ...]
    normalized_ratio_by_epoch: tuple[float, ...]
    selected_predictions: Mapping[str, RawStatePrediction]


@dataclass(frozen=True)
class OOFSelection:
    learning_rate: float
    five_seed_mean_score: float
    selected_trials: tuple[OOFTrial, ...]
    grid_trials: tuple[OOFTrial, ...]


@dataclass(frozen=True)
class FinalSeedModel:
    seed: int
    selected_epoch: int
    fitted: Any


@dataclass(frozen=True)
class TrainedV3:
    selection: OOFSelection
    models: tuple[FinalSeedModel, ...]


@dataclass(frozen=True)
class RawStatePrediction:
    singleton_utilities: Mapping[int, float]
    pair_residuals: Mapping[tuple[int, int], float]


def sha256_bytes(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise TypeError("SHA256 payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def canonical_json_line(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def load_formal58_archives(
    feature_archive: bytes,
    label_archive: bytes,
) -> tuple[tuple[GateState, ...], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Load and join the immutable formal-58 caches without any GUI decode."""
    if sha256_bytes(feature_archive) != FEATURE_CACHE_SHA256:
        raise ValueError("formal feature archive SHA256 drifted")
    if sha256_bytes(label_archive) != LABEL_CACHE_SHA256:
        raise ValueError("formal label archive SHA256 drifted")
    features = read_feature_cache(feature_archive)
    labels = read_label_cache(label_archive)
    source_ids = tuple(dict.fromkeys(state.source_id for state in features))
    states = join_feature_and_label_states(
        features,
        labels,
        expected_source_ids=source_ids,
    )
    validate_canonical_gate_state_roster(
        states,
        source_ids,
        expected_source_count=58,
    )
    if len(states) != 174 or FORMAL_JOIN_AUDIT_SHA256 != (
        "551e70b7e99f7a761f9c50d2adae3be72b0933044982a015a65e93372e41d77b"
    ):
        raise ValueError("formal joined denominator or audit binding drifted")
    _, folds = derive_oof_folds(source_ids)
    if tuple(len(fold) for fold in folds) != (12, 12, 12, 11, 11):
        raise ValueError("formal five-fold geometry drifted")
    return states, source_ids, folds


def load_fresh_features(payload: bytes) -> tuple[FeatureState, ...]:
    if sha256_bytes(payload) != FRESH_FEATURE_SHA256:
        raise ValueError("fresh-16 feature SHA256 drifted")
    features = read_feature_states_jsonl(payload)
    if len(features) != 48:
        raise ValueError("fresh-16 feature denominator drifted")
    return features


def load_fresh_states(
    feature_payload: bytes,
    label_payload: bytes,
) -> tuple[GateState, ...]:
    if sha256_bytes(feature_payload) != FRESH_FEATURE_SHA256:
        raise ValueError("fresh-16 feature SHA256 drifted")
    if sha256_bytes(label_payload) != FRESH_LABEL_SHA256:
        raise ValueError("fresh-16 label SHA256 drifted")
    features = read_feature_states_jsonl(feature_payload)
    labels = read_label_states_jsonl(label_payload)
    source_ids = tuple(dict.fromkeys(state.source_id for state in features))
    states = join_feature_and_label_states(
        features,
        labels,
        expected_source_ids=source_ids,
    )
    validate_canonical_gate_state_roster(
        states,
        source_ids,
        expected_source_count=16,
    )
    return states


def feasible_coalitions_from_ids(
    event_ids: Sequence[int],
    *,
    budget: int = DEPLOYMENT_BUDGET,
) -> tuple[tuple[int, ...], ...]:
    ids = tuple(event_ids)
    if (
        not ids
        or ids != tuple(sorted(ids))
        or len(ids) != len(set(ids))
        or any(type(item) is not int for item in ids)
        or type(budget) is not int
        or budget < 0
    ):
        raise ValueError("candidate ids or budget are malformed")
    result: list[tuple[int, ...]] = [()]
    result.extend((item,) for item in ids if budget >= 1)
    if budget >= 2:
        result.extend(
            (left, right)
            for left_index, left in enumerate(ids)
            for right in ids[left_index + 1 :]
        )
    return tuple(result)


def select_predicted_utility(
    utilities: Mapping[tuple[int, ...], float],
) -> tuple[int, ...]:
    if not utilities or () not in utilities:
        raise ValueError("predicted utility map must include the empty coalition")
    if any(not math.isfinite(float(value)) for value in utilities.values()):
        raise ValueError("predicted utilities must be finite")
    return min(
        utilities,
        key=lambda coalition: (-float(utilities[coalition]), len(coalition), coalition),
    )


def utility_maps_from_components(
    event_ids: Sequence[int],
    prediction: RawStatePrediction,
) -> tuple[Mapping[tuple[int, ...], float], Mapping[tuple[int, ...], float]]:
    feasible = feasible_coalitions_from_ids(event_ids)
    ids = tuple(event_ids)
    if set(prediction.singleton_utilities) != set(ids):
        raise ValueError("singleton prediction inventory drifted")
    expected_pairs = {item for item in feasible if len(item) == 2}
    if set(prediction.pair_residuals) != expected_pairs:
        raise ValueError("pair-residual prediction inventory drifted")
    additive: dict[tuple[int, ...], float] = {}
    total: dict[tuple[int, ...], float] = {}
    for coalition in feasible:
        base = math.fsum(prediction.singleton_utilities[event] for event in coalition)
        residual = prediction.pair_residuals[coalition] if len(coalition) == 2 else 0.0
        if not math.isfinite(base) or not math.isfinite(residual):
            raise ValueError("v3 prediction is non-finite")
        additive[coalition] = base
        total[coalition] = base + residual
    return additive, total


def aggregate_seed_predictions(
    event_ids: Sequence[int],
    predictions: Sequence[RawStatePrediction],
) -> RawStatePrediction:
    if len(predictions) != len(SEEDS):
        raise ValueError("v3 ensemble requires exactly five seed predictions")
    ids = tuple(event_ids)
    pairs = tuple(
        coalition
        for coalition in feasible_coalitions_from_ids(ids)
        if len(coalition) == 2
    )
    for prediction in predictions:
        utility_maps_from_components(ids, prediction)
    return RawStatePrediction(
        singleton_utilities={
            event: statistics.fmean(
                float(prediction.singleton_utilities[event]) for prediction in predictions
            )
            for event in ids
        },
        pair_residuals={
            pair: statistics.fmean(
                float(prediction.pair_residuals[pair]) for prediction in predictions
            )
            for pair in pairs
        },
    )


def state_prediction_record(
    feature: FeatureState,
    seed_predictions: Sequence[RawStatePrediction],
) -> Mapping[str, Any]:
    if len(seed_predictions) != len(SEEDS):
        raise ValueError("feature prediction must contain exactly five seeds")
    event_ids = feature.candidate_event_step_ids
    seed_rows = []
    unguarded_by_seed = []
    total_by_seed: list[Mapping[tuple[int, ...], float]] = []
    for seed, prediction in zip(SEEDS, seed_predictions, strict=True):
        additive, total = utility_maps_from_components(event_ids, prediction)
        additive_selected = select_predicted_utility(additive)
        unguarded_selected = select_predicted_utility(total)
        unguarded_by_seed.append(unguarded_selected)
        total_by_seed.append(total)
        seed_rows.append(
            {
                "seed": seed,
                "singleton_raw_utilities": [
                    {
                        "event_step_id": event,
                        "predicted_raw_utility": prediction.singleton_utilities[event],
                    }
                    for event in event_ids
                ],
                "pair_raw_residuals": [
                    {
                        "coalition": list(pair),
                        "predicted_raw_residual": prediction.pair_residuals[pair],
                    }
                    for pair in sorted(prediction.pair_residuals)
                ],
                "additive_selected": list(additive_selected),
                "unguarded_selected": list(unguarded_selected),
            }
        )
    ensemble = aggregate_seed_predictions(event_ids, seed_predictions)
    additive, total = utility_maps_from_components(event_ids, ensemble)
    additive_selected = select_predicted_utility(additive)
    unguarded_selected = select_predicted_utility(total)
    agreement_count = sum(
        selected == unguarded_selected for selected in unguarded_by_seed
    )
    seed_margins = tuple(
        values[unguarded_selected] - values[additive_selected]
        for values in total_by_seed
    )
    positive_margin_count = sum(value > 0.0 for value in seed_margins)
    same_selection = unguarded_selected == additive_selected
    accepted = (not same_selection) and (
        agreement_count >= CONSENSUS_MINIMUM
        and positive_margin_count >= CONSENSUS_MINIMUM
    )
    if same_selection:
        safe_selected = unguarded_selected
        reason = "pair_candidate_equals_additive"
        used_pair_candidate = False
        used_fallback = False
    elif accepted:
        safe_selected = unguarded_selected
        reason = "argmax_and_positive_margin_consensus_at_least_4_of_5"
        used_pair_candidate = True
        used_fallback = False
    else:
        safe_selected = additive_selected
        reason = "dual_consensus_below_4_of_5"
        used_pair_candidate = False
        used_fallback = True
    return {
        "source_id": feature.source_id,
        "state_id": feature.state_id,
        "decision_step_id": feature.decision_step_id,
        "candidate_event_step_ids": list(event_ids),
        "seed_predictions": seed_rows,
        "ensemble_predicted_set_utilities": [
            {
                "coalition": list(coalition),
                "additive_raw_utility": additive[coalition],
                "pair_raw_residual": (
                    ensemble.pair_residuals[coalition] if len(coalition) == 2 else 0.0
                ),
                "total_raw_utility": total[coalition],
            }
            for coalition in feasible_coalitions_from_ids(event_ids)
        ],
        "v3_additive_selected": list(additive_selected),
        "v3_unguarded_selected": list(unguarded_selected),
        "v3_safe_selected": list(safe_selected),
        "safe_decision": {
            "used_pair_candidate": used_pair_candidate,
            "used_fallback": used_fallback,
            "reason": reason,
            "pair_candidate": list(unguarded_selected),
            "base_candidate": list(additive_selected),
            "pair_candidate_equals_additive": same_selection,
            "pair_candidate_vote_count": agreement_count,
            "strictly_positive_margin_count": positive_margin_count,
            "seedwise_pair_minus_base_margins": list(seed_margins),
            "seedwise_unguarded_argmax": [list(item) for item in unguarded_by_seed],
        },
    }


def prediction_artifact_bytes(
    features: Sequence[FeatureState],
    predictions: Sequence[Sequence[RawStatePrediction]],
    *,
    model_inventory: Sequence[Mapping[str, Any]],
) -> bytes:
    if len(features) != len(predictions):
        raise ValueError("feature/prediction state denominator drifted")
    records = [
        state_prediction_record(feature, state_predictions)
        for feature, state_predictions in zip(features, predictions, strict=True)
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "SEALED_FEATURE_ONLY_SET_CONDITIONED_V3_PREDICTIONS_V1",
        "decision_contract": (
            "EXHAUSTIVE_EMPTY_SINGLETON_PAIR_ENSEMBLE_WITH_DUAL_4_OF_5_FALLBACK_V1"
        ),
        "model_inventory": list(model_inventory),
        "state_count": len(records),
        "records": records,
        "fresh_label_access_count": 0,
        "confirm_access_count": 0,
    }
    return canonical_json_line(payload)


def read_prediction_artifact(
    payload: bytes,
    *,
    features: Sequence[FeatureState],
) -> Mapping[str, Mapping[str, Any]]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("v3 prediction artifact is not JSON") from error
    records = value.get("records") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or payload != canonical_json_line(value)
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status")
        != "SEALED_FEATURE_ONLY_SET_CONDITIONED_V3_PREDICTIONS_V1"
        or value.get("fresh_label_access_count") != 0
        or value.get("confirm_access_count") != 0
        or not isinstance(records, list)
        or len(records) != len(features)
    ):
        raise ValueError("v3 prediction artifact schema or firewall status drifted")
    result = {}
    for feature, record in zip(features, records, strict=True):
        if not isinstance(record, Mapping) or (
            record.get("source_id") != feature.source_id
            or record.get("state_id") != feature.state_id
            or record.get("decision_step_id") != feature.decision_step_id
            or record.get("candidate_event_step_ids")
            != list(feature.candidate_event_step_ids)
        ):
            raise ValueError("v3 feature-only prediction identity drifted")
        feasible = set(feasible_coalitions_from_ids(feature.candidate_event_step_ids))
        for key in (
            "v3_additive_selected",
            "v3_unguarded_selected",
            "v3_safe_selected",
        ):
            selected = tuple(record.get(key, ()))
            if selected not in feasible:
                raise ValueError("v3 feature-only selection is infeasible")
        seed_rows = record.get("seed_predictions")
        if not isinstance(seed_rows, list) or len(seed_rows) != len(SEEDS):
            raise ValueError("v3 feature-only seed prediction denominator drifted")
        replay_predictions = []
        expected_pairs = tuple(
            coalition
            for coalition in feasible_coalitions_from_ids(
                feature.candidate_event_step_ids
            )
            if len(coalition) == 2
        )
        for seed, row in zip(SEEDS, seed_rows, strict=True):
            if not isinstance(row, Mapping) or row.get("seed") != seed:
                raise ValueError("v3 feature-only seed identity drifted")
            singleton_rows = row.get("singleton_raw_utilities")
            residual_rows = row.get("pair_raw_residuals")
            if not isinstance(singleton_rows, list) or not isinstance(residual_rows, list):
                raise ValueError("v3 feature-only component rows are malformed")
            singletons = {
                int(item["event_step_id"]): float(item["predicted_raw_utility"])
                for item in singleton_rows
                if isinstance(item, Mapping)
            }
            residuals = {
                tuple(item["coalition"]): float(item["predicted_raw_residual"])
                for item in residual_rows
                if isinstance(item, Mapping)
            }
            if (
                tuple(singletons) != feature.candidate_event_step_ids
                or tuple(residuals) != expected_pairs
            ):
                raise ValueError("v3 feature-only component inventory drifted")
            replay_predictions.append(
                RawStatePrediction(
                    singleton_utilities=singletons,
                    pair_residuals=residuals,
                )
            )
        if record != state_prediction_record(feature, replay_predictions):
            raise ValueError("v3 feature-only decisions do not replay component scores")
        if feature.state_id in result:
            raise ValueError("v3 prediction artifact contains duplicate state ids")
        result[feature.state_id] = record
    return result


def oracle_additive_selection(state: GateState) -> tuple[int, ...]:
    singleton = {
        event: state.table.utility((event,))
        for event in state.candidate_event_step_ids
    }
    utilities = {
        coalition: math.fsum(singleton[event] for event in coalition)
        for coalition in feasible_coalitions_from_ids(state.candidate_event_step_ids)
    }
    return select_predicted_utility(utilities)


def _n4_dual_ratio(
    states: Sequence[GateState],
    selections: Mapping[str, Sequence[int]],
) -> tuple[float, float, float]:
    n4 = tuple(state for state in states if len(state.candidate_event_step_ids) == 4)
    sources = {state.source_id for state in n4}
    if len(n4) != len(sources) or not n4:
        raise ValueError("OOF metric requires exactly one n=4 state per trajectory")
    selected_raw = {
        state.state_id: state.table.utility(selections[state.state_id]) for state in n4
    }
    exact_raw = {
        state.state_id: primary_exact_subset_oracle(state.table).utility for state in n4
    }
    selected_normalized = {}
    exact_normalized = {}
    for state in n4:
        baseline = state.table.distance(())
        selected_normalized[state.state_id] = (
            selected_raw[state.state_id] / baseline
            if baseline > NORMALIZATION_EPSILON
            else None
        )
        exact_normalized[state.state_id] = (
            exact_raw[state.state_id] / baseline
            if baseline > NORMALIZATION_EPSILON
            else None
        )
    selected_raw_mean, _, _ = _trajectory_equal_mean(n4, selected_raw)
    exact_raw_mean, _, _ = _trajectory_equal_mean(n4, exact_raw)
    selected_norm_mean, _, _ = _trajectory_equal_mean(n4, selected_normalized)
    exact_norm_mean, _, _ = _trajectory_equal_mean(n4, exact_normalized)
    raw_ratio = _ratio(selected_raw_mean, exact_raw_mean)
    normalized_ratio = _ratio(selected_norm_mean, exact_norm_mean)
    if raw_ratio is None or normalized_ratio is None:
        raise ValueError("OOF n=4 exact denominator is not positive")
    return min(raw_ratio, normalized_ratio), raw_ratio, normalized_ratio


def _raw_prediction(model: Any, target_scale: Any, state: Any) -> RawStatePrediction:
    from causalcache.set_conditioned_v3_training import predict_state

    prediction = predict_state(model, state).to_raw(target_scale)
    return RawStatePrediction(
        singleton_utilities=dict(prediction.singleton_scores),
        pair_residuals=dict(prediction.pair_residual_scores),
    )


def _unguarded_selection(model: Any, target_scale: Any, state: GateState) -> tuple[int, ...]:
    prediction = _raw_prediction(model, target_scale, state)
    _, total = utility_maps_from_components(
        state.candidate_event_step_ids,
        prediction,
    )
    return select_predicted_utility(total)


def run_oof_trial(
    states: Sequence[GateState],
    *,
    folds: Sequence[Sequence[str]],
    learning_rate: float,
    seed: int,
    maximum_epochs: int = MAXIMUM_EPOCHS,
    patience_epochs: int = PATIENCE_EPOCHS,
) -> OOFTrial:
    """Train five folds in lockstep and early-stop on the frozen dual n=4 score."""
    from causalcache.set_conditioned_v3_data import build_training_batch, fit_target_scale
    from causalcache.set_conditioned_v3_training import (
        build_model,
        build_optimizer,
        optimization_step,
    )

    if learning_rate not in LEARNING_RATES or seed not in SEEDS:
        raise ValueError("OOF hyperparameter is outside the frozen grid")
    if (
        type(maximum_epochs) is not int
        or maximum_epochs <= 0
        or type(patience_epochs) is not int
        or patience_epochs <= 0
    ):
        raise ValueError("OOF epoch limits must be positive integers")
    source_ids = tuple(dict.fromkeys(state.source_id for state in states))
    flattened = tuple(item for fold in folds for item in fold)
    if (
        len(folds) != 5
        or len(flattened) != len(set(flattened))
        or set(flattened) != set(source_ids)
    ):
        raise ValueError("OOF folds must partition trajectories exactly")
    tracks = []
    for heldout in folds:
        heldout_set = set(heldout)
        training = tuple(state for state in states if state.source_id not in heldout_set)
        evaluation = tuple(
            state
            for state in states
            if state.source_id in heldout_set
            and len(state.candidate_event_step_ids) == 4
        )
        scale = fit_target_scale(training)
        batch = build_training_batch(training, target_scale=scale)
        model = build_model(seed=seed)
        optimizer = build_optimizer(model, learning_rate=learning_rate)
        tracks.append((model, optimizer, batch, scale, evaluation))

    scores: list[float] = []
    raw_ratios: list[float] = []
    normalized_ratios: list[float] = []
    best_score = -math.inf
    best_raw = -math.inf
    best_normalized = -math.inf
    best_epoch = 0
    best_predictions: dict[str, RawStatePrediction] = {}
    for epoch in range(1, maximum_epochs + 1):
        for model, optimizer, batch, _scale, _evaluation in tracks:
            optimization_step(model, optimizer, batch)
        evaluation_states: list[GateState] = []
        selections = {}
        epoch_predictions: dict[str, RawStatePrediction] = {}
        for model, _optimizer, _batch, scale, evaluation in tracks:
            evaluation_states.extend(evaluation)
            for state in evaluation:
                prediction = _raw_prediction(model, scale, state)
                epoch_predictions[state.state_id] = prediction
                _, total = utility_maps_from_components(
                    state.candidate_event_step_ids,
                    prediction,
                )
                selections[state.state_id] = select_predicted_utility(total)
        score, raw_ratio, normalized_ratio = _n4_dual_ratio(
            evaluation_states,
            selections,
        )
        scores.append(score)
        raw_ratios.append(raw_ratio)
        normalized_ratios.append(normalized_ratio)
        if best_epoch == 0 or score > best_score + IMPROVEMENT_EPSILON:
            best_score = score
            best_raw = raw_ratio
            best_normalized = normalized_ratio
            best_epoch = epoch
            best_predictions = epoch_predictions
        if epoch - best_epoch >= patience_epochs:
            break
    if best_epoch == 0:
        raise RuntimeError("OOF trial did not select an epoch")
    return OOFTrial(
        learning_rate=learning_rate,
        seed=seed,
        selected_epoch=best_epoch,
        selected_score=best_score,
        selected_raw_ratio=best_raw,
        selected_normalized_ratio=best_normalized,
        epochs_run=len(scores),
        score_by_epoch=tuple(scores),
        raw_ratio_by_epoch=tuple(raw_ratios),
        normalized_ratio_by_epoch=tuple(normalized_ratios),
        selected_predictions=best_predictions,
    )


def choose_oof_selection(trials: Sequence[OOFTrial]) -> OOFSelection:
    trials = tuple(trials)
    expected = {(learning_rate, seed) for learning_rate in LEARNING_RATES for seed in SEEDS}
    observed = {(trial.learning_rate, trial.seed) for trial in trials}
    if len(trials) != len(expected) or observed != expected:
        raise ValueError("OOF selection requires the complete LR/seed grid")
    means = {
        learning_rate: statistics.fmean(
            trial.selected_score
            for trial in trials
            if trial.learning_rate == learning_rate
        )
        for learning_rate in LEARNING_RATES
    }
    if abs(means[LEARNING_RATES[0]] - means[LEARNING_RATES[1]]) <= IMPROVEMENT_EPSILON:
        selected_lr = LEARNING_RATES[0]
    else:
        selected_lr = max(LEARNING_RATES, key=lambda value: means[value])
    grid = tuple(
        sorted(trials, key=lambda trial: (LEARNING_RATES.index(trial.learning_rate), trial.seed))
    )
    selected = tuple(trial for trial in grid if trial.learning_rate == selected_lr)
    return OOFSelection(
        learning_rate=selected_lr,
        five_seed_mean_score=means[selected_lr],
        selected_trials=selected,
        grid_trials=grid,
    )


def train_v3(
    states: Sequence[GateState],
    *,
    folds: Sequence[Sequence[str]],
    maximum_epochs: int = MAXIMUM_EPOCHS,
    patience_epochs: int = PATIENCE_EPOCHS,
) -> TrainedV3:
    from causalcache.set_conditioned_v3_data import fit_target_scale
    from causalcache.set_conditioned_v3_training import train_fixed_epochs

    trials = tuple(
        run_oof_trial(
            states,
            folds=folds,
            learning_rate=learning_rate,
            seed=seed,
            maximum_epochs=maximum_epochs,
            patience_epochs=patience_epochs,
        )
        for learning_rate in LEARNING_RATES
        for seed in SEEDS
    )
    selection = choose_oof_selection(trials)
    full_scale = fit_target_scale(states)
    models = tuple(
        FinalSeedModel(
            seed=trial.seed,
            selected_epoch=trial.selected_epoch,
            fitted=train_fixed_epochs(
                states,
                seed=trial.seed,
                epochs=trial.selected_epoch,
                learning_rate=selection.learning_rate,
                target_scale=full_scale,
            ),
        )
        for trial in selection.selected_trials
    )
    if tuple(item.seed for item in models) != SEEDS:
        raise RuntimeError("final v3 ensemble seed order drifted")
    return TrainedV3(selection=selection, models=models)


def _trial_payload(trial: OOFTrial) -> Mapping[str, Any]:
    prediction_witness = [
        {
            "state_id": state_id,
            "singleton_raw_utilities": [
                [event, value]
                for event, value in sorted(prediction.singleton_utilities.items())
            ],
            "pair_raw_residuals": [
                [list(pair), value]
                for pair, value in sorted(prediction.pair_residuals.items())
            ],
        }
        for state_id, prediction in sorted(trial.selected_predictions.items())
    ]
    return {
        "learning_rate": trial.learning_rate,
        "seed": trial.seed,
        "selected_epoch": trial.selected_epoch,
        "selected_score": trial.selected_score,
        "selected_raw_ratio": trial.selected_raw_ratio,
        "selected_normalized_ratio": trial.selected_normalized_ratio,
        "epochs_run": trial.epochs_run,
        "score_by_epoch": list(trial.score_by_epoch),
        "raw_ratio_by_epoch": list(trial.raw_ratio_by_epoch),
        "normalized_ratio_by_epoch": list(trial.normalized_ratio_by_epoch),
        "selected_prediction_count": len(prediction_witness),
        "selected_predictions_sha256": sha256_bytes(
            canonical_json_bytes(prediction_witness)
        ),
    }


def oof_selection_payload(selection: OOFSelection) -> Mapping[str, Any]:
    return {
        "learning_rate": selection.learning_rate,
        "five_seed_mean_score": selection.five_seed_mean_score,
        "selected_trials": [_trial_payload(trial) for trial in selection.selected_trials],
        "grid_trials": [_trial_payload(trial) for trial in selection.grid_trials],
    }


def formal_oof_variant_diagnostics(
    states: Sequence[GateState],
    selection: OOFSelection,
) -> Mapping[str, Any]:
    n4 = tuple(state for state in states if len(state.candidate_event_step_ids) == 4)
    if len(n4) != 58 or tuple(trial.seed for trial in selection.selected_trials) != SEEDS:
        raise ValueError("formal OOF variant diagnostics require 58 n=4 states and five seeds")
    records = {}
    for state in n4:
        predictions = tuple(
            trial.selected_predictions[state.state_id]
            for trial in selection.selected_trials
        )
        records[state.state_id] = state_prediction_record(_feature_from_gate(state), predictions)
    variants = {
        "additive": {
            state.state_id: tuple(records[state.state_id]["v3_additive_selected"])
            for state in n4
        },
        "unguarded_pair_residual": {
            state.state_id: tuple(records[state.state_id]["v3_unguarded_selected"])
            for state in n4
        },
        "safe_pair_residual": {
            state.state_id: tuple(records[state.state_id]["v3_safe_selected"])
            for state in n4
        },
    }
    metrics = {}
    for name, selected in variants.items():
        score, raw_ratio, normalized_ratio = _n4_dual_ratio(n4, selected)
        metrics[name] = {
            "minimum_ratio": score,
            "raw_utility_ratio_to_exact": raw_ratio,
            "normalized_recovery_ratio_to_exact": normalized_ratio,
        }
    safe_accepted = sum(
        bool(records[state.state_id]["safe_decision"]["used_pair_candidate"])
        for state in n4
    )
    safe_fallback = sum(
        bool(records[state.state_id]["safe_decision"]["used_fallback"])
        for state in n4
    )
    return {
        "state_count": len(n4),
        "trajectory_count": len({state.source_id for state in n4}),
        "variants": metrics,
        "safe_accepted_unguarded_count": safe_accepted,
        "safe_fallback_count": safe_fallback,
        "safe_pair_equals_additive_count": len(n4) - safe_accepted - safe_fallback,
    }


def _feature_from_gate(state: GateState) -> FeatureState:
    return FeatureState(
        source_id=state.source_id,
        state_id=state.state_id,
        decision_step_id=state.decision_step_id,
        candidate_event_step_ids=state.candidate_event_step_ids,
        q64=state.q64,
        candidates=state.candidates,
    )


def _serialize_checkpoint(model: Any) -> bytes:
    try:
        from safetensors.torch import load, save
    except ModuleNotFoundError as error:
        raise RuntimeError("v3 checkpoint serialization requires safetensors") from error
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in sorted(model.state_dict().items())
    }
    payload = save(tensors)
    replay = load(payload)
    if save({name: replay[name].contiguous() for name in sorted(replay)}) != payload:
        raise RuntimeError("v3 safetensors checkpoint is not deterministic")
    return payload


def serialize_trained_v3(
    trained: TrainedV3,
) -> tuple[Mapping[str, bytes], tuple[Mapping[str, Any], ...]]:
    payloads: dict[str, bytes] = {}
    inventory = []
    for item in trained.models:
        checkpoint = _serialize_checkpoint(item.fitted.model)
        path = f"checkpoints/seed-{item.seed}.safetensors"
        payloads[path] = checkpoint
        scale = item.fitted.batch.target_scale
        inventory.append(
            {
                "seed": item.seed,
                "selected_epoch": item.selected_epoch,
                "learning_rate": trained.selection.learning_rate,
                "target_scale_rms": scale.rms,
                "target_scale_unclamped_rms": scale.unclamped_rms,
                "target_scale_clamped": scale.clamped,
                "checkpoint_path": path,
                "checkpoint_sha256": sha256_bytes(checkpoint),
                "checkpoint_size_bytes": len(checkpoint),
            }
        )
    if tuple(record["seed"] for record in inventory) != SEEDS:
        raise RuntimeError("serialized v3 model inventory seed order drifted")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "SERIALIZED_SET_CONDITIONED_V3_ENSEMBLE_V1",
        "device": "cpu",
        "dtype": "float32",
        "architecture": {
            "pair_order": "chronological",
            "pair_feature_dimension": 320,
            "target": "raw_utility_with_fold_train_shared_rms",
            "loss": (
                "mean(singleton_regression,pair_regression,residual_regression)"
                "+0.25*all_set_ranking"
            ),
        },
        "models": inventory,
        "fresh_label_access_count": 0,
        "confirm_access_count": 0,
        "gpu_operation_count": 0,
    }
    payloads["model-metadata.json"] = canonical_json_line(metadata)
    return payloads, tuple(inventory)


def training_report(
    trained: TrainedV3,
    *,
    formal_states: Sequence[GateState],
) -> Mapping[str, Any]:
    oof_epochs = sum(trial.epochs_run for trial in trained.selection.grid_trials)
    final_epochs = sum(item.selected_epoch for item in trained.models)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETED_SET_CONDITIONED_V3_FORMAL58_TRAIN_V1",
        "scope": "non_mainline_exploration",
        "formal_feature_sha256": FEATURE_CACHE_SHA256,
        "formal_label_sha256": LABEL_CACHE_SHA256,
        "formal_join_audit_sha256": FORMAL_JOIN_AUDIT_SHA256,
        "formal_trajectory_count": 58,
        "formal_state_count": len(formal_states),
        "selection_metric": (
            "single_seed_unguarded_n4_min_trajectory_equal_raw_ratio_"
            "normalized_ratio_to_exact"
        ),
        "post_selection_formal_variants": [
            "five_seed_additive",
            "five_seed_unguarded",
            "four_of_five_safe_with_additive_fallback",
        ],
        "oracle_interaction_headroom": formal_oracle_interaction_headroom(
            formal_states
        ),
        "oof": oof_selection_payload(trained.selection),
        "formal_oof_variant_diagnostics": formal_oof_variant_diagnostics(
            formal_states,
            trained.selection,
        ),
        "operation_counts": {
            "oof_grid_trial_count": len(trained.selection.grid_trials),
            "oof_fold_track_count": 5 * len(trained.selection.grid_trials),
            "oof_epoch_metric_count": oof_epochs,
            "oof_optimizer_step_count": 5 * oof_epochs,
            "final_fit_count": len(trained.models),
            "final_optimizer_step_count": final_epochs,
            "optimizer_step_count": 5 * oof_epochs + final_epochs,
            "checkpoint_count": len(trained.models),
        },
        "formal_label_decode_count": 1,
        "fresh_feature_decode_count": 1,
        "fresh_label_access_count": 0,
        "confirm_access_count": 0,
        "legacy_development_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
        "raw_gui_access_count": 0,
        "policy_forward_count": 0,
        "gpu_operation_count": 0,
    }


def formal_oracle_interaction_headroom(
    states: Sequence[GateState],
) -> Mapping[str, Any]:
    """Measure how much exact B=2 utility is lost by deleting pair residuals."""
    exact = {
        state.state_id: primary_exact_subset_oracle(state.table).coalition
        for state in states
    }
    additive = {
        state.state_id: oracle_additive_selection(state) for state in states
    }
    selections = {
        name: (exact if name == "exact" else additive) for name in METHODS
    }
    metrics = selection_metric_summary(states, selections)
    raw_deltas = _trajectory_deltas(
        states,
        selections,
        left="exact",
        right="oracle_additive",
        normalized=False,
    )
    normalized_deltas = _trajectory_deltas(
        states,
        selections,
        left="exact",
        right="oracle_additive",
        normalized=True,
    )
    return {
        "definition": "exact_subset_oracle_minus_oracle_singleton_additive",
        "overall_metrics": {
            "exact": metrics["overall"]["methods"]["exact"],
            "oracle_additive": metrics["overall"]["methods"]["oracle_additive"],
        },
        "n4_metrics": {
            "exact": metrics["by_event_count"]["4"]["methods"]["exact"],
            "oracle_additive": metrics["by_event_count"]["4"]["methods"][
                "oracle_additive"
            ],
        },
        "trajectory_paired_bootstrap": {
            "raw": paired_trajectory_bootstrap(raw_deltas),
            "normalized": paired_trajectory_bootstrap(normalized_deltas),
        },
    }


def train_and_seal(
    *,
    formal_feature_archive: bytes,
    formal_label_archive: bytes,
    fresh_feature_payload: bytes,
    output_dir: Path,
    source_a_git_commit: str,
    execution_b_git_commit: str,
    runner_freeze_sha256: str,
    contract_sha256: str,
    maximum_epochs: int = MAXIMUM_EPOCHS,
    patience_epochs: int = PATIENCE_EPOCHS,
) -> Mapping[str, Any]:
    """Train on formal-58, predict fresh features, and seal all label-blind bytes."""
    states, _source_ids, folds = load_formal58_archives(
        formal_feature_archive,
        formal_label_archive,
    )
    features = load_fresh_features(fresh_feature_payload)
    trained = train_v3(
        states,
        folds=folds,
        maximum_epochs=maximum_epochs,
        patience_epochs=patience_epochs,
    )
    model_payloads, model_inventory = serialize_trained_v3(trained)
    predictions = tuple(
        tuple(
            _raw_prediction(
                item.fitted.model,
                item.fitted.batch.target_scale,
                feature,
            )
            for item in trained.models
        )
        for feature in features
    )
    prediction_payload = prediction_artifact_bytes(
        features,
        predictions,
        model_inventory=model_inventory,
    )
    report = training_report(trained, formal_states=states)
    report_payload = canonical_json_line(report)
    payloads = {
        **model_payloads,
        "fresh16-predictions.json": prediction_payload,
        "formal58-training-report.json": report_payload,
    }
    expected_payloads = {
        *(f"checkpoints/seed-{seed}.safetensors" for seed in SEEDS),
        "model-metadata.json",
        "fresh16-predictions.json",
        "formal58-training-report.json",
    }
    if set(payloads) != expected_payloads:
        raise RuntimeError("label-blind v3 bundle inventory drifted")
    return seal_label_blind_outputs(
        output_dir,
        payloads,
        training_report={
            "path": "formal58-training-report.json",
            "sha256": sha256_bytes(report_payload),
            "selected_learning_rate": trained.selection.learning_rate,
            "five_seed_mean_oof_score": trained.selection.five_seed_mean_score,
        },
        source_a_git_commit=source_a_git_commit,
        execution_b_git_commit=execution_b_git_commit,
        runner_freeze_sha256=runner_freeze_sha256,
        contract_sha256=contract_sha256,
    )


def _trajectory_equal_mean(
    states: Sequence[GateState],
    values: Mapping[str, float | None],
) -> tuple[float | None, int, int]:
    by_source: dict[str, list[float]] = defaultdict(list)
    excluded_states = 0
    for state in states:
        value = values[state.state_id]
        if value is None:
            excluded_states += 1
        else:
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError("metric value is non-finite")
            by_source[state.source_id].append(converted)
    means = [statistics.fmean(items) for items in by_source.values() if items]
    all_sources = {state.source_id for state in states}
    return (
        statistics.fmean(means) if means else None,
        excluded_states,
        len(all_sources) - len(means),
    )


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= NORMALIZATION_EPSILON:
        return None
    value = numerator / denominator
    if not math.isfinite(value):
        raise ValueError("metric ratio is non-finite")
    return value


def selection_metric_summary(
    states: Sequence[GateState],
    selections: Mapping[str, Mapping[str, Sequence[int]]],
) -> Mapping[str, Any]:
    """Compute trajectory-equal raw/normalized metrics overall and by event count."""
    if tuple(selections) != METHODS:
        raise ValueError("selection method inventory or order drifted")
    state_ids = {state.state_id for state in states}
    if any(set(method) != state_ids for method in selections.values()):
        raise ValueError("selection state inventory drifted")

    def summarize(selected_states: Sequence[GateState]) -> Mapping[str, Any]:
        selected_states = tuple(selected_states)
        method_payload: dict[str, Any] = {}
        for name in METHODS:
            raw: dict[str, float] = {}
            normalized: dict[str, float | None] = {}
            for state in selected_states:
                coalition = tuple(selections[name][state.state_id])
                if coalition not in feasible_coalitions_from_ids(
                    state.candidate_event_step_ids
                ):
                    raise ValueError(f"{name} emitted an infeasible coalition")
                utility = state.table.utility(coalition)
                baseline = state.table.distance(())
                raw[state.state_id] = utility
                normalized[state.state_id] = (
                    utility / baseline if baseline > NORMALIZATION_EPSILON else None
                )
            raw_mean, _, _ = _trajectory_equal_mean(selected_states, raw)
            norm_mean, excluded_states, excluded_trajectories = _trajectory_equal_mean(
                selected_states,
                normalized,
            )
            method_payload[name] = {
                "mean_raw_utility": raw_mean,
                "mean_normalized_recovery": norm_mean,
                "normalized_excluded_state_count": excluded_states,
                "normalized_excluded_trajectory_count": excluded_trajectories,
            }
        exact = method_payload["exact"]
        for payload in method_payload.values():
            payload["raw_ratio_to_exact"] = _ratio(
                payload["mean_raw_utility"], exact["mean_raw_utility"]
            )
            payload["normalized_ratio_to_exact"] = _ratio(
                payload["mean_normalized_recovery"],
                exact["mean_normalized_recovery"],
            )
        return {
            "state_count": len(selected_states),
            "trajectory_count": len({state.source_id for state in selected_states}),
            "methods": method_payload,
        }

    return {
        "overall": summarize(states),
        "by_event_count": {
            str(event_count): summarize(
                tuple(
                    state
                    for state in states
                    if len(state.candidate_event_step_ids) == event_count
                )
            )
            for event_count in (2, 3, 4)
        },
    }


def _type7_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile input is malformed")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def paired_trajectory_bootstrap(
    deltas: Mapping[str, float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> Mapping[str, Any]:
    ordered = tuple(float(deltas[key]) for key in sorted(deltas))
    if (
        not ordered
        or any(not math.isfinite(value) for value in ordered)
        or type(resamples) is not int
        or resamples <= 0
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("paired bootstrap input is malformed")
    generator = random.Random(seed)
    means = tuple(
        math.fsum(ordered[generator.randrange(len(ordered))] for _ in ordered)
        / len(ordered)
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "trajectory_count": len(ordered),
        "mean_delta": statistics.fmean(ordered),
        "positive_trajectory_count": sum(value > 0.0 for value in ordered),
        "negative_trajectory_count": sum(value < 0.0 for value in ordered),
        "zero_trajectory_count": sum(value == 0.0 for value in ordered),
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "lower": _type7_quantile(means, tail),
        "upper": _type7_quantile(means, 1.0 - tail),
    }


def _trajectory_deltas(
    states: Sequence[GateState],
    selections: Mapping[str, Mapping[str, Sequence[int]]],
    *,
    left: str,
    right: str,
    normalized: bool,
) -> Mapping[str, float]:
    by_source: dict[str, list[float]] = defaultdict(list)
    for state in states:
        baseline = state.table.distance(())
        if normalized and baseline <= NORMALIZATION_EPSILON:
            continue
        left_utility = state.table.utility(selections[left][state.state_id])
        right_utility = state.table.utility(selections[right][state.state_id])
        delta = left_utility - right_utility
        by_source[state.source_id].append(delta / baseline if normalized else delta)
    return {
        source_id: statistics.fmean(values)
        for source_id, values in by_source.items()
        if values
    }


def bootstrap_summary(
    states: Sequence[GateState],
    selections: Mapping[str, Mapping[str, Sequence[int]]],
) -> Mapping[str, Any]:
    comparisons = (
        ("v3_unguarded", "v3_additive"),
        ("v3_safe", "v3_additive"),
        ("v3_safe", "historical_v1_independent"),
    )
    return {
        f"{left}_minus_{right}": {
            metric: paired_trajectory_bootstrap(
                _trajectory_deltas(
                    states,
                    selections,
                    left=left,
                    right=right,
                    normalized=metric == "normalized",
                )
            )
            for metric in ("raw", "normalized")
        }
        for left, right in comparisons
    }


def switch_diagnostics(
    states: Sequence[GateState],
    selections: Mapping[str, Mapping[str, Sequence[int]]],
    prediction_records: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    rows = []
    for state in states:
        record = prediction_records[state.state_id]
        additive = tuple(selections["v3_additive"][state.state_id])
        unguarded = tuple(selections["v3_unguarded"][state.state_id])
        safe = tuple(selections["v3_safe"][state.state_id])
        delta = state.table.utility(safe) - state.table.utility(additive)
        rows.append(
            {
                "state_id": state.state_id,
                "event_count": len(state.candidate_event_step_ids),
                "safe_used_pair_candidate": bool(
                    record["safe_decision"]["used_pair_candidate"]
                ),
                "safe_used_fallback": bool(
                    record["safe_decision"]["used_fallback"]
                ),
                "unguarded_changed_additive": unguarded != additive,
                "safe_changed_additive": safe != additive,
                "safe_delta_raw": delta,
                "outcome": "improved" if delta > 0.0 else "harmed" if delta < 0.0 else "equal",
                "reason": str(record["safe_decision"]["reason"]),
            }
        )

    def summary(selected: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        selected = tuple(selected)
        return {
            "state_count": len(selected),
            "used_pair_candidate_count": sum(
                bool(row["safe_used_pair_candidate"]) for row in selected
            ),
            "fallback_count": sum(
                bool(row["safe_used_fallback"]) for row in selected
            ),
            "pair_equals_additive_count": sum(
                not bool(row["safe_used_pair_candidate"])
                and not bool(row["safe_used_fallback"])
                for row in selected
            ),
            "unguarded_changed_additive_count": sum(
                bool(row["unguarded_changed_additive"]) for row in selected
            ),
            "safe_changed_additive_count": sum(
                bool(row["safe_changed_additive"]) for row in selected
            ),
            "improved_count": sum(row["outcome"] == "improved" for row in selected),
            "harmed_count": sum(row["outcome"] == "harmed" for row in selected),
            "equal_count": sum(row["outcome"] == "equal" for row in selected),
            "mean_safe_minus_additive_raw": (
                statistics.fmean(float(row["safe_delta_raw"]) for row in selected)
                if selected
                else None
            ),
            "reason_counts": dict(sorted(Counter(row["reason"] for row in selected).items())),
        }

    return {
        "overall": summary(rows),
        "by_event_count": {
            str(event_count): summary(
                tuple(row for row in rows if row["event_count"] == event_count)
            )
            for event_count in (2, 3, 4)
        },
    }


def _regular_file_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input is missing or unsafe: {path}")
    return path.read_bytes()


def _write_exclusive_or_identical(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError(f"pre-existing output differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal_label_blind_outputs(
    output_dir: Path,
    payloads: Mapping[str, bytes],
    *,
    training_report: Mapping[str, Any],
    source_a_git_commit: str,
    execution_b_git_commit: str,
    runner_freeze_sha256: str,
    contract_sha256: str,
) -> Mapping[str, Any]:
    """Persist model/prediction bytes and seal them before labels may be opened."""
    if not payloads or any(
        Path(name).is_absolute() or ".." in Path(name).parts for name in payloads
    ):
        raise ValueError("label-blind payload paths are unsafe")
    if (
        len(source_a_git_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_a_git_commit)
        or len(execution_b_git_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in execution_b_git_commit
        )
        or len(runner_freeze_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in runner_freeze_sha256
        )
        or len(contract_sha256) != 64
        or any(character not in "0123456789abcdef" for character in contract_sha256)
    ):
        raise ValueError("Git or contract execution binding is malformed")
    for name in sorted(payloads):
        _write_exclusive_or_identical(output_dir / name, payloads[name])
    inventory = [
        {
            "path": name,
            "sha256": sha256_bytes(payloads[name]),
            "size_bytes": len(payloads[name]),
        }
        for name in sorted(payloads)
    ]
    seal = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "SEALED_LABEL_BLIND_SET_CONDITIONED_V3_V1",
        "training_report": training_report,
        "source_a_git_commit": source_a_git_commit,
        "execution_b_git_commit": execution_b_git_commit,
        "runner_freeze_sha256": runner_freeze_sha256,
        "contract_sha256": contract_sha256,
        "payload_inventory": inventory,
        "fresh_label_access_count": 0,
        "confirm_access_count": 0,
        "legacy_development_access_count": 0,
        "raw_gui_access_count": 0,
        "policy_forward_count": 0,
        "gpu_operation_count": 0,
    }
    seal_payload = canonical_json_line(seal)
    _write_exclusive_or_identical(output_dir / "label-blind-seal.json", seal_payload)
    return {**seal, "seal_sha256": sha256_bytes(seal_payload)}


def verify_label_blind_seal(output_dir: Path) -> Mapping[str, Any]:
    payload = _regular_file_bytes(output_dir / "label-blind-seal.json")
    try:
        seal = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("label-blind seal is not JSON") from error
    if (
        not isinstance(seal, Mapping)
        or payload != canonical_json_line(seal)
        or seal.get("status") != "SEALED_LABEL_BLIND_SET_CONDITIONED_V3_V1"
        or seal.get("fresh_label_access_count") != 0
        or seal.get("confirm_access_count") != 0
        or not isinstance(seal.get("source_a_git_commit"), str)
        or len(seal["source_a_git_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in seal["source_a_git_commit"]
        )
        or not isinstance(seal.get("execution_b_git_commit"), str)
        or len(seal["execution_b_git_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in seal["execution_b_git_commit"]
        )
        or not isinstance(seal.get("runner_freeze_sha256"), str)
        or len(seal["runner_freeze_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in seal["runner_freeze_sha256"]
        )
        or not isinstance(seal.get("contract_sha256"), str)
        or len(seal["contract_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in seal["contract_sha256"]
        )
    ):
        raise ValueError("label-blind seal schema or firewall status drifted")
    for record in seal.get("payload_inventory", ()):
        if not isinstance(record, Mapping):
            raise ValueError("label-blind payload inventory is malformed")
        observed = _regular_file_bytes(output_dir / str(record["path"]))
        if (
            sha256_bytes(observed) != record.get("sha256")
            or len(observed) != record.get("size_bytes")
        ):
            raise ValueError("sealed label-blind payload changed")
    return seal


def claim_fresh_label_access(output_dir: Path) -> Mapping[str, Any]:
    """Persist the one development-label claim without opening label bytes."""
    seal = verify_label_blind_seal(output_dir)
    seal_payload = _regular_file_bytes(output_dir / "label-blind-seal.json")
    claim = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "CLAIMED_FRESH16_DEVELOPMENT_LABEL_ACCESS_SET_CONDITIONED_V3_V1",
        "label_blind_seal_sha256": sha256_bytes(seal_payload),
        "expected_fresh_label_sha256": FRESH_LABEL_SHA256,
        "fresh_label_access_claim_count": 1,
        "fresh_label_decode_count": 0,
        "confirm_access_count": 0,
        "legacy_development_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
        "raw_gui_access_count": 0,
        "policy_forward_count": 0,
        "gpu_operation_count": 0,
        "sealed_payload_count": len(seal["payload_inventory"]),
    }
    payload = canonical_json_line(claim)
    _write_exclusive_or_identical(output_dir / "fresh-label-access-claim.json", payload)
    return {**claim, "claim_sha256": sha256_bytes(payload)}


def verify_fresh_label_access_claim(output_dir: Path) -> Mapping[str, Any]:
    verify_label_blind_seal(output_dir)
    seal_payload = _regular_file_bytes(output_dir / "label-blind-seal.json")
    claim_payload = _regular_file_bytes(output_dir / "fresh-label-access-claim.json")
    try:
        claim = json.loads(claim_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("fresh label access claim is not JSON") from error
    if (
        not isinstance(claim, Mapping)
        or claim_payload != canonical_json_line(claim)
        or claim.get("status")
        != "CLAIMED_FRESH16_DEVELOPMENT_LABEL_ACCESS_SET_CONDITIONED_V3_V1"
        or claim.get("label_blind_seal_sha256") != sha256_bytes(seal_payload)
        or claim.get("expected_fresh_label_sha256") != FRESH_LABEL_SHA256
        or claim.get("fresh_label_access_claim_count") != 1
        or claim.get("fresh_label_decode_count") != 0
        or claim.get("confirm_access_count") != 0
    ):
        raise ValueError("fresh label access claim schema or firewall status drifted")
    return claim


def build_evaluation_report(
    states: Sequence[GateState],
    *,
    prediction_records: Mapping[str, Mapping[str, Any]],
    historical_v1_independent: Mapping[str, Sequence[int]],
    label_blind_seal_sha256: str,
    label_access_claim_sha256: str,
) -> Mapping[str, Any]:
    state_ids = {state.state_id for state in states}
    if set(prediction_records) != state_ids or set(historical_v1_independent) != state_ids:
        raise ValueError("evaluation decision inventory differs from fresh labels")
    selections: dict[str, dict[str, tuple[int, ...]]] = {
        name: {} for name in METHODS
    }
    for state in states:
        exact = primary_exact_subset_oracle(state.table).coalition
        record = prediction_records[state.state_id]
        selections["exact"][state.state_id] = exact
        selections["oracle_additive"][state.state_id] = oracle_additive_selection(state)
        selections["v3_additive"][state.state_id] = tuple(
            record["v3_additive_selected"]
        )
        selections["v3_unguarded"][state.state_id] = tuple(
            record["v3_unguarded_selected"]
        )
        selections["v3_safe"][state.state_id] = tuple(record["v3_safe_selected"])
        selections["historical_v1_independent"][state.state_id] = tuple(
            historical_v1_independent[state.state_id]
        )
    ordered_selections = {name: selections[name] for name in METHODS}
    metrics = selection_metric_summary(states, ordered_selections)
    bootstraps = bootstrap_summary(states, ordered_selections)
    switches = switch_diagnostics(states, ordered_selections, prediction_records)
    primary_bootstrap = bootstraps["v3_safe_minus_v3_additive"]
    safe_overall = metrics["overall"]["methods"]["v3_safe"]
    additive_overall = metrics["overall"]["methods"]["v3_additive"]
    safe_n4 = metrics["by_event_count"]["4"]["methods"]["v3_safe"]
    additive_n4 = metrics["by_event_count"]["4"]["methods"]["v3_additive"]
    normalized_delta = (
        safe_overall["mean_normalized_recovery"]
        - additive_overall["mean_normalized_recovery"]
    )
    raw_delta = safe_overall["mean_raw_utility"] - additive_overall["mean_raw_utility"]
    n4_normalized_delta = (
        safe_n4["mean_normalized_recovery"]
        - additive_n4["mean_normalized_recovery"]
    )
    interpretation_checks = {
        "mean_normalized_delta_at_least_0_01": normalized_delta >= 0.01,
        "normalized_bootstrap_lower_strictly_positive": (
            primary_bootstrap["normalized"]["lower"] > 0.0
        ),
        "mean_raw_delta_strictly_positive": raw_delta > 0.0,
        "n4_mean_normalized_delta_strictly_positive": n4_normalized_delta > 0.0,
        "at_least_8_positive_trajectories": (
            primary_bootstrap["normalized"]["positive_trajectory_count"] >= 8
        ),
    }
    promising = all(interpretation_checks.values())
    development_interpretation = (
        "PROMISING_DEVELOPMENT_SIGNAL_FOR_A_SEPARATELY_FROZEN_FUTURE_STUDY"
        if promising
        else "NO_DEVELOPMENT_EVIDENCE_TO_CONTINUE_SET_CONDITIONING"
    )
    state_rows = []
    for state in states:
        baseline = state.table.distance(())
        state_rows.append(
            {
                "source_id": state.source_id,
                "state_id": state.state_id,
                "event_count": len(state.candidate_event_step_ids),
                "baseline_distance": baseline,
                "methods": {
                    name: {
                        "selected": list(ordered_selections[name][state.state_id]),
                        "raw_utility": state.table.utility(
                            ordered_selections[name][state.state_id]
                        ),
                        "normalized_recovery": (
                            state.table.utility(ordered_selections[name][state.state_id])
                            / baseline
                            if baseline > NORMALIZATION_EPSILON
                            else None
                        ),
                    }
                    for name in METHODS
                },
                "safe_decision": dict(prediction_records[state.state_id]["safe_decision"]),
            }
        )
    report_without_hash = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETED_SET_CONDITIONED_V3_FRESH16_DEVELOPMENT_V1",
        "scope": "consumed_development_exploration_not_confirmatory",
        "label_blind_seal_sha256": label_blind_seal_sha256,
        "label_access_claim_sha256": label_access_claim_sha256,
        "fresh_feature_sha256": FRESH_FEATURE_SHA256,
        "fresh_label_sha256": FRESH_LABEL_SHA256,
        "historical_v1_independent_sha256": HISTORICAL_INDEPENDENT_SHA256,
        "trajectory_count": len({state.source_id for state in states}),
        "state_count": len(states),
        "metrics": metrics,
        "bootstrap": {
            "unit": "trajectory",
            "interval": "two_sided_percentile",
            "comparisons": bootstraps,
        },
        "switch_diagnostics": switches,
        "development_interpretation": {
            "value": development_interpretation,
            "primary_contrast": "v3_safe_minus_v3_additive",
            "mean_normalized_delta": normalized_delta,
            "mean_raw_delta": raw_delta,
            "n4_mean_normalized_delta": n4_normalized_delta,
            "checks": interpretation_checks,
            "may_authorize_confirm": False,
            "may_change_v1_verdict": False,
        },
        "state_records": state_rows,
        "fresh_label_access_claim_count": 1,
        "fresh_label_decode_count": 1,
        "confirm_access_count": 0,
        "legacy_development_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
        "raw_gui_access_count": 0,
        "policy_forward_count": 0,
        "gpu_operation_count": 0,
    }
    report_sha256 = sha256_bytes(canonical_json_bytes(report_without_hash))
    return {**report_without_hash, "report_sha256": report_sha256}


def evaluate_sealed_fresh16(
    *,
    output_dir: Path,
    feature_payload: bytes,
    label_payload: bytes,
    historical_independent_payload: bytes,
) -> Mapping[str, Any]:
    """Verify the prior seal/claim, then perform the sole fresh-label decode."""
    verify_label_blind_seal(output_dir)
    claim = verify_fresh_label_access_claim(output_dir)
    seal_payload = _regular_file_bytes(output_dir / "label-blind-seal.json")
    claim_payload = _regular_file_bytes(output_dir / "fresh-label-access-claim.json")
    prediction_payload = _regular_file_bytes(output_dir / "fresh16-predictions.json")
    features = load_fresh_features(feature_payload)
    predictions = read_prediction_artifact(prediction_payload, features=features)
    states = load_fresh_states(feature_payload, label_payload)
    historical = load_historical_v1_independent(
        historical_independent_payload,
        feature_states=features,
    )
    report = build_evaluation_report(
        states,
        prediction_records=predictions,
        historical_v1_independent=historical,
        label_blind_seal_sha256=sha256_bytes(seal_payload),
        label_access_claim_sha256=sha256_bytes(claim_payload),
    )
    if claim.get("expected_fresh_label_sha256") != sha256_bytes(label_payload):
        raise ValueError("decoded fresh label differs from the pre-access claim")
    report_payload = canonical_json_line(report)
    _write_exclusive_or_identical(output_dir / "fresh16-development-report.json", report_payload)
    return report


def load_historical_v1_independent(
    payload: bytes,
    *,
    feature_states: Sequence[FeatureState],
) -> Mapping[str, tuple[int, ...]]:
    if sha256_bytes(payload) != HISTORICAL_INDEPENDENT_SHA256:
        raise ValueError("historical v1 independent decision SHA256 drifted")
    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("historical v1 independent artifact has duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("historical v1 independent artifact is not JSON") from error
    records = value.get("records") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or payload != canonical_json_line(value)
        or set(value)
        != {
            "schema_version",
            "protocol_id",
            "status",
            "name",
            "selection_sha256",
            "decision_contract",
            "records",
        }
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != FRESH_PROTOCOL_ID
        or value.get("status") != FRESH_LEARNED_STATUS
        or value.get("name") != "independent"
        or value.get("decision_contract")
        != "FEATURE_ONLY_ALL_ENSEMBLE_AND_SEED_DECISIONS_V1"
        or not isinstance(records, list)
        or len(records) != FRESH_STATE_COUNT
        or len(feature_states) != FRESH_STATE_COUNT
    ):
        raise ValueError("historical v1 independent artifact schema drifted")
    features = {state.state_id: state for state in feature_states}
    if len(features) != FRESH_STATE_COUNT:
        raise ValueError("historical v1 feature inventory drifted")
    selections: dict[str, tuple[int, ...]] = {}
    for feature, record in zip(feature_states, records, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "state_id",
            "ensemble_selected_event_step_ids",
            "seed_selected_event_step_ids",
        }:
            raise ValueError("historical v1 independent state schema drifted")
        if record.get("state_id") != feature.state_id:
            raise ValueError("historical v1 independent state order drifted")
        selected_raw = record.get("ensemble_selected_event_step_ids")
        if not isinstance(selected_raw, list) or any(
            type(event) is not int for event in selected_raw
        ):
            raise ValueError("historical v1 independent selection is malformed")
        selected = tuple(selected_raw)
        if (
            selected != tuple(sorted(selected))
            or len(set(selected)) != len(selected)
            or len(selected) > DEPLOYMENT_BUDGET
            or not set(selected).issubset(feature.candidate_event_step_ids)
        ):
            raise ValueError("historical v1 independent selection is infeasible")
        seed_rows = record.get("seed_selected_event_step_ids")
        if not isinstance(seed_rows, list) or len(seed_rows) != len(SEEDS):
            raise ValueError("historical v1 independent seed inventory drifted")
        for seed, seed_record in zip(SEEDS, seed_rows, strict=True):
            if (
                not isinstance(seed_record, Mapping)
                or set(seed_record) != {"seed", "selected_event_step_ids"}
                or seed_record.get("seed") != seed
            ):
                raise ValueError("historical v1 independent seed record drifted")
            seed_selected = seed_record.get("selected_event_step_ids")
            if not isinstance(seed_selected, list) or tuple(seed_selected) not in set(
                feasible_coalitions_from_ids(feature.candidate_event_step_ids)
            ):
                raise ValueError("historical v1 independent seed selection is infeasible")
        selections[feature.state_id] = selected
    if value.get("selection_sha256") != canonical_selection_sha256(selections):
        raise ValueError("historical v1 independent selection digest drifted")
    return selections


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONSENSUS_MINIMUM",
    "FRESH_FEATURE_SHA256",
    "FRESH_LABEL_SHA256",
    "HISTORICAL_INDEPENDENT_SHA256",
    "IMPROVEMENT_EPSILON",
    "LEARNING_RATES",
    "MAXIMUM_EPOCHS",
    "METHODS",
    "OOFSelection",
    "OOFTrial",
    "PATIENCE_EPOCHS",
    "PROTOCOL_ID",
    "SEEDS",
    "TrainedV3",
    "RawStatePrediction",
    "bootstrap_summary",
    "build_evaluation_report",
    "canonical_json_line",
    "choose_oof_selection",
    "claim_fresh_label_access",
    "evaluate_sealed_fresh16",
    "feasible_coalitions_from_ids",
    "load_formal58_archives",
    "load_fresh_features",
    "load_fresh_states",
    "load_historical_v1_independent",
    "oracle_additive_selection",
    "formal_oracle_interaction_headroom",
    "formal_oof_variant_diagnostics",
    "oof_selection_payload",
    "paired_trajectory_bootstrap",
    "seal_label_blind_outputs",
    "selection_metric_summary",
    "select_predicted_utility",
    "sha256_bytes",
    "state_prediction_record",
    "switch_diagnostics",
    "train_and_seal",
    "train_v3",
    "utility_maps_from_components",
    "read_prediction_artifact",
    "run_oof_trial",
    "serialize_trained_v3",
    "training_report",
    "verify_fresh_label_access_claim",
    "verify_label_blind_seal",
]
