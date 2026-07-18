"""Sealed CPU orchestration for the frozen-base residual v4 development study."""

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

from causalcache.gate_v1_contract import canonical_json_bytes
from causalcache.gate_v1_data import (
    DEPLOYMENT_BUDGET,
    NORMALIZATION_EPSILON,
    FeatureState,
    GateState,
    join_feature_and_label_states,
    validate_canonical_gate_state_roster,
)
from causalcache.gate_v1_fresh16 import read_label_states_jsonl
from causalcache.gate_v1_provenance import (
    FrozenEnsembleProvenance,
    canonical_model_state_sha256,
    canonical_selection_sha256,
    frozen_ensemble_provenance_from_manifest,
)
from causalcache.gate_v1_training import model_score
from causalcache.restoration_v2_2_label_table import primary_exact_subset_oracle
from causalcache.set_conditioned_v3_exploration import (
    FRESH_FEATURE_SHA256,
    FRESH_LABEL_SHA256,
    HISTORICAL_INDEPENDENT_SHA256,
    feasible_coalitions_from_ids,
    load_formal58_archives,
    load_fresh_features,
    select_predicted_utility,
)
from causalcache.set_conditioned_v3_parser_repair import (
    load_historical_v1_independent_parser_repair_v2,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_conditioned_v4_frozen_base_residual_development_v1"
SEEDS = (0, 1, 2, 3, 4)
LEARNING_RATES = (0.0003, 0.001)
BASE_LEARNING_RATE = 0.0003
BASE_SELECTED_EPOCHS = (60, 51, 6, 56, 54)
BASE_OOF_RAW_RATIOS = (
    0.9012143403172107,
    0.9044738982091675,
    0.9164728647614823,
    0.9105838341486948,
    0.8991374084054389,
)
BASE_OOF_MEAN_RAW_RATIO = 0.9063764691683989
MAXIMUM_EPOCHS = 500
PATIENCE_EPOCHS = 50
IMPROVEMENT_EPSILON = 1e-4
CONSENSUS_MINIMUM = 4
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 271_828
BOOTSTRAP_CONFIDENCE = 0.90
BASE_MODEL_REPOSITORY = "gavinlaw/causalcache-gate-v1-formal58-selector-mobile"
BASE_MODEL_MANIFEST_REVISION = "23f6786075c7bff91f93fd7e8a878e070efb72a9"
BASE_MODEL_PAYLOAD_REVISION = "a6c9e7f6dab6bc27794438b5b66da07fd59b2889"
BASE_MANIFEST_SHA256 = (
    "38809f1ef4da9636ef6779733cee4926b36b38e3bd86eab199450e685c63da24"
)
BASE_PROVENANCE_SHA256 = (
    "238c2e64bdb22ccebd261961200021e74a688f912d530f486b00bbb853a8ea57"
)
BASE_CHECKPOINT_RECORDS = (
    {
        "seed": 0,
        "selected_epoch": 60,
        "path": "formal58-train/v1/checkpoints/independent-seed-0.safetensors",
        "sha256": "7088a1cb4224dd1e7013868249ed1c99c20782eb63c2a762e3059cfb939d4a49",
        "model_state_sha256": "0198ad3cca7bd9e4eef2f6c40c9ac6349732f21935c61ec3434f058ee40b972a",
        "size_bytes": 102860,
    },
    {
        "seed": 1,
        "selected_epoch": 51,
        "path": "formal58-train/v1/checkpoints/independent-seed-1.safetensors",
        "sha256": "8a25b8bad27e4c6f20a9a93273602e82a276ee7f9275ff672d6b0ed64dd8fd79",
        "model_state_sha256": "454b0732389eac920f2aa69edb68317e3746dc57e1d2c6c7269f792257255af7",
        "size_bytes": 102860,
    },
    {
        "seed": 2,
        "selected_epoch": 6,
        "path": "formal58-train/v1/checkpoints/independent-seed-2.safetensors",
        "sha256": "b8c38619db06bd479d278da62663a3b451cff951dff0747bf156835234216211",
        "model_state_sha256": "c4d508933928dc7c6a24501090e3afd2a7d50a4977f37b3b3cb6cd818d28eb13",
        "size_bytes": 102860,
    },
    {
        "seed": 3,
        "selected_epoch": 56,
        "path": "formal58-train/v1/checkpoints/independent-seed-3.safetensors",
        "sha256": "b7b175c0b513508f10479bb228fad6ffb6b4e330b050c981d415c2a54b8c11a2",
        "model_state_sha256": "1f0d319411319dd2348e0a3da8e7d2b18e5b566e468278853d890a761639da60",
        "size_bytes": 102860,
    },
    {
        "seed": 4,
        "selected_epoch": 54,
        "path": "formal58-train/v1/checkpoints/independent-seed-4.safetensors",
        "sha256": "8e5eee31c85653a210b9bf05c4cf4d94a8b4fabdb6bdab455caa446cc655df38",
        "model_state_sha256": "51fc5aaa157ec9feb97a7c40cd7d532cc248ecb076feacd880dc9bba38007def",
        "size_bytes": 102860,
    },
)
METHODS = (
    "exact",
    "frozen_base",
    "v4_unguarded",
    "v4_safe",
    "historical_v1_independent",
)


@dataclass(frozen=True)
class FrozenBaseEnsemble:
    provenance: FrozenEnsembleProvenance
    bases: tuple[Any, ...]
    manifest_sha256: str
    checkpoint_inventory: tuple[Mapping[str, Any], ...]
    checkpoint_payloads: tuple[bytes, ...]


@dataclass(frozen=True)
class ResidualStatePrediction:
    singleton_scores: Mapping[int, float]
    pair_residual_scores: Mapping[tuple[int, int], float]


@dataclass(frozen=True)
class OOFTrial:
    learning_rate: float
    seed: int
    selected_epoch: int
    selected_score: float
    selected_raw_ratio: float
    selected_normalized_ratio: float
    optimization_epochs_run: int
    score_by_epoch: tuple[float, ...]
    raw_ratio_by_epoch: tuple[float, ...]
    normalized_ratio_by_epoch: tuple[float, ...]
    selected_predictions: Mapping[str, ResidualStatePrediction]


@dataclass(frozen=True)
class OOFSelection:
    learning_rate: float
    five_seed_mean_score: float
    selected_trials: tuple[OOFTrial, ...]
    grid_trials: tuple[OOFTrial, ...]


@dataclass(frozen=True)
class FinalResidualModel:
    seed: int
    selected_epoch: int
    base: Any
    head: Any


@dataclass(frozen=True)
class TrainedV4:
    selection: OOFSelection
    models: tuple[FinalResidualModel, ...]
    fold_base_replay: Mapping[str, Any]


def sha256_bytes(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise TypeError("SHA256 payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def canonical_json_line(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(value, Mapping) or payload != canonical_json_line(value):
        raise ValueError(f"{label} is not canonical JSON plus newline")
    return value


def load_frozen_base_ensemble(
    manifest_payload: bytes,
    checkpoint_payloads: Mapping[str, bytes],
) -> FrozenBaseEnsemble:
    """Strictly bind and load the five immutable historical independent bases."""
    if sha256_bytes(manifest_payload) != BASE_MANIFEST_SHA256:
        raise ValueError("frozen independent ensemble manifest SHA256 drifted")
    manifest = _strict_json(manifest_payload, label="frozen independent manifest")
    provenance = frozen_ensemble_provenance_from_manifest(manifest)
    if (
        provenance.training.family != "independent"
        or provenance.sha256 != BASE_PROVENANCE_SHA256
        or provenance.training.learning_rate != BASE_LEARNING_RATE
        or provenance.training.selected_epochs != BASE_SELECTED_EPOCHS
    ):
        raise ValueError("frozen independent ensemble provenance drifted")
    expected_paths = {str(record["path"]) for record in BASE_CHECKPOINT_RECORDS}
    if set(checkpoint_payloads) != expected_paths:
        raise ValueError("frozen independent checkpoint inventory drifted")
    bases = []
    for record, bound in zip(
        BASE_CHECKPOINT_RECORDS,
        provenance.checkpoints,
        strict=True,
    ):
        payload = checkpoint_payloads[str(record["path"])]
        if (
            len(payload) != record["size_bytes"]
            or sha256_bytes(payload) != record["sha256"]
            or bound.seed != record["seed"]
            or bound.selected_epoch != record["selected_epoch"]
            or bound.model_state_sha256 != record["model_state_sha256"]
            or bound.checkpoint_artifact.repository != BASE_MODEL_REPOSITORY
            or bound.checkpoint_artifact.revision != BASE_MODEL_PAYLOAD_REVISION
            or bound.checkpoint_artifact.path != record["path"]
            or bound.checkpoint_artifact.sha256 != record["sha256"]
        ):
            raise ValueError("frozen independent checkpoint binding drifted")
        from causalcache.set_conditioned_v4_training import (
            load_frozen_independent_base,
        )

        bases.append(
            load_frozen_independent_base(
                payload,
                seed=int(record["seed"]),
                selected_epoch=int(record["selected_epoch"]),
                learning_rate=BASE_LEARNING_RATE,
                expected_model_state_sha256=str(record["model_state_sha256"]),
                expected_checkpoint_sha256=str(record["sha256"]),
            )
        )
    if tuple(getattr(base, "seed", None) for base in bases) != SEEDS:
        raise ValueError("frozen independent base seed order drifted")
    return FrozenBaseEnsemble(
        provenance=provenance,
        bases=tuple(bases),
        manifest_sha256=BASE_MANIFEST_SHA256,
        checkpoint_inventory=BASE_CHECKPOINT_RECORDS,
        checkpoint_payloads=tuple(
            checkpoint_payloads[str(record["path"])]
            for record in BASE_CHECKPOINT_RECORDS
        ),
    )


def frozen_base_scores(base: Any, state: FeatureState | GateState) -> Mapping[int, float]:
    scorer = model_score(base.model, "independent")
    scores = {
        event: float(scorer(state, event, ()))
        for event in state.candidate_event_step_ids
    }
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("frozen independent base emitted a non-finite score")
    return scores


def set_score_maps(
    event_ids: Sequence[int],
    prediction: ResidualStatePrediction,
) -> tuple[Mapping[tuple[int, ...], float], Mapping[tuple[int, ...], float]]:
    feasible = feasible_coalitions_from_ids(event_ids)
    if set(prediction.singleton_scores) != set(event_ids):
        raise ValueError("frozen-base singleton score inventory drifted")
    expected_pairs = {coalition for coalition in feasible if len(coalition) == 2}
    if set(prediction.pair_residual_scores) != expected_pairs:
        raise ValueError("frozen-base pair residual inventory drifted")
    base_scores = {}
    total_scores = {}
    for coalition in feasible:
        base = math.fsum(prediction.singleton_scores[event] for event in coalition)
        residual = (
            prediction.pair_residual_scores[coalition]
            if len(coalition) == 2
            else 0.0
        )
        if not math.isfinite(base) or not math.isfinite(residual):
            raise ValueError("v4 predicted set score is non-finite")
        base_scores[coalition] = base
        total_scores[coalition] = base + residual
    return base_scores, total_scores


def zero_residual_prediction(
    event_scores: Mapping[int, float],
) -> ResidualStatePrediction:
    event_ids = tuple(event_scores)
    feasible = feasible_coalitions_from_ids(event_ids)
    return ResidualStatePrediction(
        singleton_scores=dict(event_scores),
        pair_residual_scores={
            coalition: 0.0 for coalition in feasible if len(coalition) == 2
        },
    )


def base_selection_from_scores(event_scores: Mapping[int, float]) -> tuple[int, ...]:
    prediction = zero_residual_prediction(event_scores)
    base, total = set_score_maps(tuple(event_scores), prediction)
    if base != total:
        raise RuntimeError("zero residual changed frozen-base set scores")
    selected = select_predicted_utility(base)
    ranked = sorted(
        (event for event, value in event_scores.items() if value > 0.0),
        key=lambda event: (-float(event_scores[event]), event),
    )
    expected = tuple(sorted(ranked[:DEPLOYMENT_BUDGET]))
    if selected != expected:
        raise RuntimeError("set enumeration does not replay independent positive top-B")
    return selected


def _prediction_from_training(value: Any) -> ResidualStatePrediction:
    prediction = ResidualStatePrediction(
        singleton_scores=dict(value.singleton_scores),
        pair_residual_scores=dict(value.pair_residual_scores),
    )
    set_score_maps(tuple(prediction.singleton_scores), prediction)
    return prediction


def aggregate_seed_predictions(
    event_ids: Sequence[int],
    predictions: Sequence[ResidualStatePrediction],
) -> ResidualStatePrediction:
    if len(predictions) != len(SEEDS):
        raise ValueError("v4 ensemble requires exactly five seed predictions")
    ids = tuple(event_ids)
    pairs = tuple(
        coalition for coalition in feasible_coalitions_from_ids(ids) if len(coalition) == 2
    )
    for prediction in predictions:
        set_score_maps(ids, prediction)
    return ResidualStatePrediction(
        singleton_scores={
            event: statistics.fmean(
                prediction.singleton_scores[event] for prediction in predictions
            )
            for event in ids
        },
        pair_residual_scores={
            pair: statistics.fmean(
                prediction.pair_residual_scores[pair] for prediction in predictions
            )
            for pair in pairs
        },
    )


def state_prediction_record(
    feature: FeatureState,
    seed_predictions: Sequence[ResidualStatePrediction],
) -> Mapping[str, Any]:
    if len(seed_predictions) != len(SEEDS):
        raise ValueError("v4 feature prediction requires exactly five seeds")
    seed_rows = []
    seed_unguarded = []
    seed_totals = []
    for seed, prediction in zip(SEEDS, seed_predictions, strict=True):
        base_scores, total_scores = set_score_maps(
            feature.candidate_event_step_ids,
            prediction,
        )
        base_selected = select_predicted_utility(base_scores)
        zero_selected = base_selection_from_scores(prediction.singleton_scores)
        if base_selected != zero_selected:
            raise RuntimeError("zero residual failed to replay one frozen base seed")
        unguarded = select_predicted_utility(total_scores)
        seed_unguarded.append(unguarded)
        seed_totals.append(total_scores)
        seed_rows.append(
            {
                "seed": seed,
                "frozen_event_scores": [
                    {"event_step_id": event, "score": prediction.singleton_scores[event]}
                    for event in feature.candidate_event_step_ids
                ],
                "pair_residual_scores": [
                    {"coalition": list(pair), "score": prediction.pair_residual_scores[pair]}
                    for pair in sorted(prediction.pair_residual_scores)
                ],
                "frozen_base_selected": list(base_selected),
                "zero_residual_selected": list(zero_selected),
                "unguarded_selected": list(unguarded),
            }
        )
    ensemble = aggregate_seed_predictions(
        feature.candidate_event_step_ids,
        seed_predictions,
    )
    base_scores, total_scores = set_score_maps(
        feature.candidate_event_step_ids,
        ensemble,
    )
    base_selected = select_predicted_utility(base_scores)
    zero_selected = base_selection_from_scores(ensemble.singleton_scores)
    if zero_selected != base_selected:
        raise RuntimeError("zero residual failed to replay frozen base ensemble")
    pair_candidate = select_predicted_utility(total_scores)
    vote_count = sum(item == pair_candidate for item in seed_unguarded)
    margins = tuple(
        values[pair_candidate] - values[base_selected] for values in seed_totals
    )
    positive_margin_count = sum(value > 0.0 for value in margins)
    identical = pair_candidate == base_selected
    accepted = (
        not identical
        and vote_count >= CONSENSUS_MINIMUM
        and positive_margin_count >= CONSENSUS_MINIMUM
    )
    if identical:
        safe = base_selected
        reason = "pair_candidate_equals_frozen_base"
        used_pair = False
        fallback = False
    elif accepted:
        safe = pair_candidate
        reason = "argmax_and_positive_margin_consensus_at_least_4_of_5"
        used_pair = True
        fallback = False
    else:
        safe = base_selected
        reason = "dual_consensus_below_4_of_5"
        used_pair = False
        fallback = True
    return {
        "source_id": feature.source_id,
        "state_id": feature.state_id,
        "decision_step_id": feature.decision_step_id,
        "candidate_event_step_ids": list(feature.candidate_event_step_ids),
        "seed_predictions": seed_rows,
        "ensemble_predicted_set_scores": [
            {
                "coalition": list(coalition),
                "frozen_base_score": base_scores[coalition],
                "pair_residual_score": (
                    ensemble.pair_residual_scores[coalition]
                    if len(coalition) == 2
                    else 0.0
                ),
                "total_score": total_scores[coalition],
            }
            for coalition in feasible_coalitions_from_ids(
                feature.candidate_event_step_ids
            )
        ],
        "frozen_base_selected": list(base_selected),
        "zero_residual_selected": list(zero_selected),
        "v4_unguarded_selected": list(pair_candidate),
        "v4_safe_selected": list(safe),
        "base_invariant": {
            "zero_residual_replays_frozen_base": True,
            "all_seed_zero_residual_replays_frozen_base": True,
        },
        "safe_decision": {
            "used_pair_candidate": used_pair,
            "used_fallback": fallback,
            "reason": reason,
            "pair_candidate": list(pair_candidate),
            "base_candidate": list(base_selected),
            "pair_candidate_equals_base": identical,
            "pair_candidate_vote_count": vote_count,
            "strictly_positive_margin_count": positive_margin_count,
            "seedwise_pair_minus_base_margins": list(margins),
            "seedwise_unguarded_argmax": [list(item) for item in seed_unguarded],
        },
    }


def prediction_artifact_bytes(
    features: Sequence[FeatureState],
    predictions: Sequence[Sequence[ResidualStatePrediction]],
    *,
    base_inventory: Sequence[Mapping[str, Any]],
    residual_inventory: Sequence[Mapping[str, Any]],
) -> bytes:
    if len(features) != len(predictions):
        raise ValueError("v4 feature/prediction denominator drifted")
    records = [
        state_prediction_record(feature, state_predictions)
        for feature, state_predictions in zip(features, predictions, strict=True)
    ]
    if len(records) != 48:
        raise ValueError("v4 formal prediction artifact requires fresh16/48 states")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "SEALED_FEATURE_ONLY_SET_CONDITIONED_V4_FROZEN_BASE_RESIDUAL_V1",
        "decision_contract": (
            "FROZEN_FIVE_SEED_BASE_PLUS_PAIR_RESIDUAL_WITH_DUAL_4_OF_5_FALLBACK_V1"
        ),
        "frozen_base_manifest_sha256": BASE_MANIFEST_SHA256,
        "frozen_base_inventory": list(base_inventory),
        "residual_model_inventory": list(residual_inventory),
        "state_count": len(records),
        "zero_residual_replay_state_count": sum(
            bool(record["base_invariant"]["zero_residual_replays_frozen_base"])
            for record in records
        ),
        "records": records,
        "fresh_label_access_count": 0,
        "confirm20_access_count": 0,
        "policy_forward_count": 0,
        "gpu_operation_count": 0,
    }
    return canonical_json_line(payload)


def read_prediction_artifact(
    payload: bytes,
    *,
    features: Sequence[FeatureState] | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    value = _strict_json(payload, label="v4 feature-only predictions")
    records = value.get("records")
    if (
        value.get("protocol_id") != PROTOCOL_ID
        or value.get("status")
        != "SEALED_FEATURE_ONLY_SET_CONDITIONED_V4_FROZEN_BASE_RESIDUAL_V1"
        or value.get("frozen_base_manifest_sha256") != BASE_MANIFEST_SHA256
        or value.get("state_count") != 48
        or value.get("zero_residual_replay_state_count") != 48
        or value.get("fresh_label_access_count") != 0
        or value.get("confirm20_access_count") != 0
        or not isinstance(records, list)
        or len(records) != 48
    ):
        raise ValueError("v4 feature-only prediction schema or firewall drifted")
    if features is not None and len(features) != 48:
        raise ValueError("v4 prediction replay requires exactly 48 feature states")
    result: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("v4 feature-only prediction row is malformed")
        state_id = record.get("state_id")
        candidate_ids = record.get("candidate_event_step_ids")
        if (
            not isinstance(state_id, str)
            or state_id in result
            or not isinstance(candidate_ids, list)
            or any(type(event) is not int for event in candidate_ids)
        ):
            raise ValueError("v4 feature-only prediction identity is malformed")
        if features is not None:
            feature = features[index]
            if (
                record.get("source_id") != feature.source_id
                or state_id != feature.state_id
                or record.get("decision_step_id") != feature.decision_step_id
                or candidate_ids != list(feature.candidate_event_step_ids)
            ):
                raise ValueError("v4 feature-only prediction identity drifted")
        feasible = set(feasible_coalitions_from_ids(tuple(candidate_ids)))
        for key in (
            "frozen_base_selected",
            "zero_residual_selected",
            "v4_unguarded_selected",
            "v4_safe_selected",
        ):
            if tuple(record.get(key, ())) not in feasible:
                raise ValueError("v4 feature-only prediction is infeasible")
        if record["frozen_base_selected"] != record["zero_residual_selected"]:
            raise ValueError("v4 zero residual does not replay the frozen base")
        seed_rows = record.get("seed_predictions")
        if not isinstance(seed_rows, list) or len(seed_rows) != 5:
            raise ValueError("v4 feature-only seed denominator drifted")
        replay = []
        expected_pairs = tuple(
            coalition
            for coalition in feasible_coalitions_from_ids(tuple(candidate_ids))
            if len(coalition) == 2
        )
        for seed, row in zip(SEEDS, seed_rows, strict=True):
            if not isinstance(row, Mapping) or row.get("seed") != seed:
                raise ValueError("v4 feature-only seed identity drifted")
            singleton_rows = row.get("frozen_event_scores")
            residual_rows = row.get("pair_residual_scores")
            if not isinstance(singleton_rows, list) or not isinstance(
                residual_rows, list
            ):
                raise ValueError("v4 feature-only component rows are malformed")
            singletons = {
                int(item["event_step_id"]): float(item["score"])
                for item in singleton_rows
                if isinstance(item, Mapping)
            }
            residuals = {
                tuple(item["coalition"]): float(item["score"])
                for item in residual_rows
                if isinstance(item, Mapping)
            }
            if tuple(singletons) != tuple(candidate_ids) or tuple(residuals) != expected_pairs:
                raise ValueError("v4 feature-only component inventory drifted")
            replay.append(
                ResidualStatePrediction(
                    singleton_scores=singletons,
                    pair_residual_scores=residuals,
                )
            )
        if features is not None and record != state_prediction_record(features[index], replay):
            raise ValueError("v4 feature-only decisions do not replay sealed scores")
        result[state_id] = record
    return result


def _n4_dual_ratio(
    states: Sequence[GateState],
    selections: Mapping[str, Sequence[int]],
) -> tuple[float, float, float]:
    n4 = tuple(state for state in states if len(state.candidate_event_step_ids) == 4)
    if not n4 or len(n4) != len({state.source_id for state in n4}):
        raise ValueError("v4 OOF requires one n=4 state per trajectory")
    selected_raw = {
        state.state_id: state.table.utility(selections[state.state_id]) for state in n4
    }
    exact_raw = {
        state.state_id: primary_exact_subset_oracle(state.table).utility for state in n4
    }
    selected_norm: dict[str, float | None] = {}
    exact_norm: dict[str, float | None] = {}
    for state in n4:
        baseline = state.table.distance(())
        selected_norm[state.state_id] = (
            selected_raw[state.state_id] / baseline
            if baseline > NORMALIZATION_EPSILON
            else None
        )
        exact_norm[state.state_id] = (
            exact_raw[state.state_id] / baseline
            if baseline > NORMALIZATION_EPSILON
            else None
        )
    selected_raw_mean, _, _ = _trajectory_equal_mean(n4, selected_raw)
    exact_raw_mean, _, _ = _trajectory_equal_mean(n4, exact_raw)
    selected_norm_mean, _, _ = _trajectory_equal_mean(n4, selected_norm)
    exact_norm_mean, _, _ = _trajectory_equal_mean(n4, exact_norm)
    raw_ratio = _ratio(selected_raw_mean, exact_raw_mean)
    normalized_ratio = _ratio(selected_norm_mean, exact_norm_mean)
    if raw_ratio is None or normalized_ratio is None:
        raise ValueError("v4 OOF exact denominator is not positive")
    return min(raw_ratio, normalized_ratio), raw_ratio, normalized_ratio


def _base_state_digest(base: Any) -> str:
    digest = canonical_model_state_sha256(base.model)
    expected = getattr(base, "model_state_sha256", digest)
    if digest != expected:
        raise ValueError("frozen base model-state digest drifted")
    if any(parameter.requires_grad for parameter in base.model.parameters()):
        raise ValueError("frozen base unexpectedly contains trainable parameters")
    return digest


def prepare_fold_clean_bases(
    states: Sequence[GateState],
    *,
    folds: Sequence[Sequence[str]],
) -> tuple[Mapping[int, tuple[Mapping[str, Any], ...]], Mapping[str, Any]]:
    """Rebuild heldout-clean historical bases and replay their frozen OOF metric."""
    from causalcache.set_conditioned_v4_training import replay_fold_clean_base

    tracks_by_seed: dict[int, tuple[Mapping[str, Any], ...]] = {}
    seed_reports = []
    for seed, epoch, expected_ratio in zip(
        SEEDS,
        BASE_SELECTED_EPOCHS,
        BASE_OOF_RAW_RATIOS,
        strict=True,
    ):
        tracks = []
        all_evaluation = []
        selections = {}
        for heldout in folds:
            heldout_set = set(heldout)
            training = tuple(
                state for state in states if state.source_id not in heldout_set
            )
            evaluation = tuple(
                state
                for state in states
                if state.source_id in heldout_set
                and len(state.candidate_event_step_ids) == 4
            )
            base = replay_fold_clean_base(
                training,
                seed=seed,
                selected_epoch=epoch,
                learning_rate=BASE_LEARNING_RATE,
            )
            before = _base_state_digest(base)
            for state in evaluation:
                scores = frozen_base_scores(base, state)
                selections[state.state_id] = base_selection_from_scores(scores)
            all_evaluation.extend(evaluation)
            tracks.append(
                {
                    "training": training,
                    "evaluation": evaluation,
                    "base": base,
                    "base_state_sha256_before": before,
                }
            )
        _score, raw_ratio, normalized_ratio = _n4_dual_ratio(
            all_evaluation,
            selections,
        )
        if not math.isclose(raw_ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("fold-clean frozen base failed historical OOF replay")
        tracks_by_seed[seed] = tuple(tracks)
        seed_reports.append(
            {
                "seed": seed,
                "selected_epoch": epoch,
                "raw_utility_ratio_to_exact": raw_ratio,
                "normalized_recovery_ratio_to_exact": normalized_ratio,
                "expected_historical_raw_ratio": expected_ratio,
                "exact_replay_within_1e-12": True,
            }
        )
    mean_raw = statistics.fmean(
        float(record["raw_utility_ratio_to_exact"]) for record in seed_reports
    )
    if not math.isclose(
        mean_raw,
        BASE_OOF_MEAN_RAW_RATIO,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("fold-clean base five-seed OOF mean failed replay")
    return tracks_by_seed, {
        "status": "REPLAYED_FORMAL58_FOLD_CLEAN_FROZEN_INDEPENDENT_BASE_V1",
        "seed_reports": seed_reports,
        "five_seed_mean_raw_utility_ratio_to_exact": mean_raw,
        "expected_historical_mean": BASE_OOF_MEAN_RAW_RATIO,
        "full_fit_base_scored_heldout_fold_count": 0,
        "fold_clean_base_count": 25,
    }


def run_oof_trial(
    states: Sequence[GateState],
    *,
    fold_clean_tracks: Sequence[Mapping[str, Any]],
    learning_rate: float,
    seed: int,
    maximum_epochs: int = MAXIMUM_EPOCHS,
    patience_epochs: int = PATIENCE_EPOCHS,
) -> OOFTrial:
    """Fit residual-only folds with epoch zero as the exact frozen-base candidate."""
    from causalcache.set_conditioned_v4_data import build_training_batch
    from causalcache.set_conditioned_v4_training import (
        build_optimizer,
        build_residual_head,
        optimization_step,
        predict_state,
    )

    if learning_rate not in LEARNING_RATES or seed not in SEEDS:
        raise ValueError("v4 OOF hyperparameter is outside the frozen grid")
    if len(fold_clean_tracks) != 5:
        raise ValueError("v4 OOF requires five fold-clean base tracks")
    tracks = []
    for source in fold_clean_tracks:
        base = source["base"]
        if getattr(base, "seed", None) != seed:
            raise ValueError("v4 OOF base/head seed pairing drifted")
        head = build_residual_head(seed=seed)
        optimizer = build_optimizer(head, learning_rate=learning_rate)
        batch = build_training_batch(source["training"])
        tracks.append(
            {
                **dict(source),
                "head": head,
                "optimizer": optimizer,
                "batch": batch,
            }
        )

    scores = []
    raw_ratios = []
    normalized_ratios = []
    best_score = -math.inf
    best_raw = -math.inf
    best_normalized = -math.inf
    best_epoch = -1
    best_predictions: dict[str, ResidualStatePrediction] = {}
    optimization_epochs_run = 0
    for epoch in range(0, maximum_epochs + 1):
        if epoch > 0:
            for track in tracks:
                optimization_step(
                    track["base"],
                    track["head"],
                    track["optimizer"],
                    track["batch"],
                )
            optimization_epochs_run += 1
        all_evaluation = []
        selections = {}
        epoch_predictions = {}
        for track in tracks:
            evaluation = track["evaluation"]
            all_evaluation.extend(evaluation)
            for state in evaluation:
                prediction = _prediction_from_training(
                    predict_state(track["base"], track["head"], state)
                )
                epoch_predictions[state.state_id] = prediction
                _base, total = set_score_maps(
                    state.candidate_event_step_ids,
                    prediction,
                )
                selections[state.state_id] = select_predicted_utility(total)
        score, raw_ratio, normalized_ratio = _n4_dual_ratio(
            all_evaluation,
            selections,
        )
        scores.append(score)
        raw_ratios.append(raw_ratio)
        normalized_ratios.append(normalized_ratio)
        if best_epoch < 0 or score > best_score + IMPROVEMENT_EPSILON:
            best_score = score
            best_raw = raw_ratio
            best_normalized = normalized_ratio
            best_epoch = epoch
            best_predictions = epoch_predictions
        if epoch > 0 and epoch - best_epoch >= patience_epochs:
            break
    if best_epoch < 0:
        raise RuntimeError("v4 OOF did not select epoch zero or a residual epoch")
    for track in tracks:
        if _base_state_digest(track["base"]) != track["base_state_sha256_before"]:
            raise RuntimeError("residual optimization mutated a fold-clean frozen base")
    return OOFTrial(
        learning_rate=learning_rate,
        seed=seed,
        selected_epoch=best_epoch,
        selected_score=best_score,
        selected_raw_ratio=best_raw,
        selected_normalized_ratio=best_normalized,
        optimization_epochs_run=optimization_epochs_run,
        score_by_epoch=tuple(scores),
        raw_ratio_by_epoch=tuple(raw_ratios),
        normalized_ratio_by_epoch=tuple(normalized_ratios),
        selected_predictions=best_predictions,
    )


def choose_oof_selection(trials: Sequence[OOFTrial]) -> OOFSelection:
    trials = tuple(trials)
    expected = {
        (learning_rate, seed) for learning_rate in LEARNING_RATES for seed in SEEDS
    }
    if len(trials) != len(expected) or {
        (trial.learning_rate, trial.seed) for trial in trials
    } != expected:
        raise ValueError("v4 OOF selection requires the complete LR/seed grid")
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
        sorted(
            trials,
            key=lambda trial: (
                LEARNING_RATES.index(trial.learning_rate),
                trial.seed,
            ),
        )
    )
    return OOFSelection(
        learning_rate=selected_lr,
        five_seed_mean_score=means[selected_lr],
        selected_trials=tuple(
            trial for trial in grid if trial.learning_rate == selected_lr
        ),
        grid_trials=grid,
    )


def train_v4(
    states: Sequence[GateState],
    *,
    folds: Sequence[Sequence[str]],
    frozen_bases: FrozenBaseEnsemble,
    maximum_epochs: int = MAXIMUM_EPOCHS,
    patience_epochs: int = PATIENCE_EPOCHS,
) -> TrainedV4:
    from causalcache.set_conditioned_v4_training import train_fixed_epochs

    tracks_by_seed, replay = prepare_fold_clean_bases(states, folds=folds)
    trials = tuple(
        run_oof_trial(
            states,
            fold_clean_tracks=tracks_by_seed[seed],
            learning_rate=learning_rate,
            seed=seed,
            maximum_epochs=maximum_epochs,
            patience_epochs=patience_epochs,
        )
        for learning_rate in LEARNING_RATES
        for seed in SEEDS
    )
    selection = choose_oof_selection(trials)
    models = []
    for trial, base in zip(
        selection.selected_trials,
        frozen_bases.bases,
        strict=True,
    ):
        before = _base_state_digest(base)
        fitted = train_fixed_epochs(
            states,
            base=base,
            seed=trial.seed,
            epochs=trial.selected_epoch,
            learning_rate=selection.learning_rate,
        )
        fitted_base = getattr(fitted, "base", base)
        head = getattr(fitted, "head", None)
        if head is None:
            raise RuntimeError("v4 final fit did not return a residual head")
        if fitted_base is not base or _base_state_digest(base) != before:
            raise RuntimeError("v4 final residual fit mutated or replaced frozen base")
        models.append(
            FinalResidualModel(
                seed=trial.seed,
                selected_epoch=trial.selected_epoch,
                base=base,
                head=head,
            )
        )
    if tuple(item.seed for item in models) != SEEDS:
        raise RuntimeError("v4 final residual ensemble seed order drifted")
    return TrainedV4(
        selection=selection,
        models=tuple(models),
        fold_base_replay=replay,
    )


def _trial_payload(trial: OOFTrial) -> Mapping[str, Any]:
    witness = [
        {
            "state_id": state_id,
            "singleton_scores": [
                [event, value]
                for event, value in sorted(prediction.singleton_scores.items())
            ],
            "pair_residual_scores": [
                [list(pair), value]
                for pair, value in sorted(prediction.pair_residual_scores.items())
            ],
        }
        for state_id, prediction in sorted(trial.selected_predictions.items())
    ]
    return {
        "learning_rate": trial.learning_rate,
        "seed": trial.seed,
        "selected_epoch": trial.selected_epoch,
        "epoch_zero_was_candidate": True,
        "selected_score": trial.selected_score,
        "selected_raw_ratio": trial.selected_raw_ratio,
        "selected_normalized_ratio": trial.selected_normalized_ratio,
        "optimization_epochs_run": trial.optimization_epochs_run,
        "score_by_epoch_including_epoch_zero": list(trial.score_by_epoch),
        "raw_ratio_by_epoch_including_epoch_zero": list(trial.raw_ratio_by_epoch),
        "normalized_ratio_by_epoch_including_epoch_zero": list(
            trial.normalized_ratio_by_epoch
        ),
        "selected_prediction_count": len(witness),
        "selected_predictions_sha256": sha256_bytes(canonical_json_bytes(witness)),
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
        raise ValueError("v4 formal OOF diagnostics require 58 n4 states and five seeds")
    records = {}
    for state in n4:
        predictions = tuple(
            trial.selected_predictions[state.state_id]
            for trial in selection.selected_trials
        )
        records[state.state_id] = state_prediction_record(
            _feature_from_gate(state),
            predictions,
        )
    variants = {
        "frozen_base": {
            state.state_id: tuple(records[state.state_id]["frozen_base_selected"])
            for state in n4
        },
        "unguarded_frozen_base_residual": {
            state.state_id: tuple(records[state.state_id]["v4_unguarded_selected"])
            for state in n4
        },
        "safe_frozen_base_residual": {
            state.state_id: tuple(records[state.state_id]["v4_safe_selected"])
            for state in n4
        },
    }
    metrics = {}
    for name, selected in variants.items():
        score, raw, normalized = _n4_dual_ratio(n4, selected)
        metrics[name] = {
            "minimum_ratio": score,
            "raw_utility_ratio_to_exact": raw,
            "normalized_recovery_ratio_to_exact": normalized,
        }
    return {
        "state_count": 58,
        "variants": metrics,
        "safe_accepted_pair_count": sum(
            bool(records[state.state_id]["safe_decision"]["used_pair_candidate"])
            for state in n4
        ),
        "safe_fallback_count": sum(
            bool(records[state.state_id]["safe_decision"]["used_fallback"])
            for state in n4
        ),
        "selected_epoch_zero_count": sum(
            trial.selected_epoch == 0 for trial in selection.selected_trials
        ),
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


def serialize_trained_v4(
    trained: TrainedV4,
) -> tuple[Mapping[str, bytes], tuple[Mapping[str, Any], ...]]:
    from causalcache.set_conditioned_v4_training import serialize_residual_checkpoint

    payloads: dict[str, bytes] = {}
    inventory = []
    for item in trained.models:
        payload = serialize_residual_checkpoint(item.head)
        path = f"residual-checkpoints/seed-{item.seed}.safetensors"
        payloads[path] = payload
        inventory.append(
            {
                "seed": item.seed,
                "selected_epoch": item.selected_epoch,
                "learning_rate": trained.selection.learning_rate,
                "base_checkpoint_sha256": BASE_CHECKPOINT_RECORDS[item.seed]["sha256"],
                "base_model_state_sha256": BASE_CHECKPOINT_RECORDS[item.seed][
                    "model_state_sha256"
                ],
                "residual_checkpoint_path": path,
                "residual_checkpoint_sha256": sha256_bytes(payload),
                "residual_checkpoint_size_bytes": len(payload),
            }
        )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "SERIALIZED_SET_CONDITIONED_V4_FROZEN_BASE_RESIDUAL_ENSEMBLE_V1",
        "device": "cpu",
        "dtype": "float32",
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "base_parameters_trainable": False,
        "checkpoint_contains_residual_head_only": True,
        "models": inventory,
        "fresh_label_access_count": 0,
        "confirm20_access_count": 0,
        "gpu_operation_count": 0,
    }
    payloads["residual-model-metadata.json"] = canonical_json_line(metadata)
    return payloads, tuple(inventory)


def training_report(
    trained: TrainedV4,
    *,
    formal_states: Sequence[GateState],
) -> Mapping[str, Any]:
    optimization_epochs = sum(
        trial.optimization_epochs_run for trial in trained.selection.grid_trials
    )
    final_epochs = sum(item.selected_epoch for item in trained.models)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETED_SET_CONDITIONED_V4_FORMAL58_RESIDUAL_TRAIN_V1",
        "scope": "non_mainline_frozen_base_residual_development_exploration",
        "formal_trajectory_count": 58,
        "formal_state_count": len(formal_states),
        "base_model_revision": BASE_MODEL_MANIFEST_REVISION,
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "base_trainable_parameter_count": 0,
        "fold_clean_base_replay": trained.fold_base_replay,
        "epoch_zero": {
            "included_in_model_selection": True,
            "definition": "exact_frozen_base_with_pair_residual_identically_zero",
        },
        "selection_metric": (
            "single_seed_unguarded_n4_min_trajectory_equal_raw_ratio_"
            "normalized_ratio_to_exact"
        ),
        "oof": oof_selection_payload(trained.selection),
        "formal_oof_variant_diagnostics": formal_oof_variant_diagnostics(
            formal_states,
            trained.selection,
        ),
        "operation_counts": {
            "fold_clean_base_replay_count": 25,
            "fold_clean_base_optimizer_step_count": 5
            * sum(BASE_SELECTED_EPOCHS),
            "oof_grid_trial_count": 10,
            "oof_residual_fold_track_count": 50,
            "oof_epoch_zero_evaluation_count": 10,
            "oof_optimizer_step_count": 5 * optimization_epochs,
            "final_residual_fit_count": 5,
            "final_residual_optimizer_step_count": final_epochs,
            "residual_optimizer_step_count": 5 * optimization_epochs + final_epochs,
            "canonical_full_fit_frozen_base_optimizer_step_count": 0,
            "residual_checkpoint_count": 5,
        },
        "formal_label_decode_count": 1,
        "fresh_feature_decode_count": 1,
        "fresh_label_access_count": 0,
        "confirm20_access_count": 0,
        "legacy_dev5_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
        "raw_gui_access_count": 0,
        "policy_forward_count": 0,
        "gpu_operation_count": 0,
    }


def base_invariant_report(
    ensemble: FrozenBaseEnsemble,
    *,
    trained: TrainedV4,
    prediction_records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    from causalcache.gate_v1_formal_train import serialize_safetensors_checkpoint

    state_digests = tuple(_base_state_digest(base) for base in ensemble.bases)
    expected_digests = tuple(
        str(record["model_state_sha256"]) for record in BASE_CHECKPOINT_RECORDS
    )
    if state_digests != expected_digests:
        raise RuntimeError("v4 frozen full-fit base changed during residual training")
    post_training_checkpoint_payloads = tuple(
        serialize_safetensors_checkpoint(base.model) for base in ensemble.bases
    )
    if post_training_checkpoint_payloads != ensemble.checkpoint_payloads:
        raise RuntimeError("v4 frozen base checkpoint bytes changed during training")
    zero_count = sum(
        bool(record["base_invariant"]["zero_residual_replays_frozen_base"])
        for record in prediction_records
    )
    if zero_count != 48:
        raise RuntimeError("v4 zero residual did not replay all fresh base decisions")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "VERIFIED_SET_CONDITIONED_V4_FROZEN_BASE_INVARIANT_V1",
        "manifest_sha256": ensemble.manifest_sha256,
        "ensemble_provenance_sha256": ensemble.provenance.sha256,
        "checkpoint_input_inventory": list(ensemble.checkpoint_inventory),
        "post_training_model_state_sha256": list(state_digests),
        "post_training_checkpoint_sha256": [
            sha256_bytes(payload) for payload in post_training_checkpoint_payloads
        ],
        "pre_post_checkpoint_bytes_identical": True,
        "fold_clean_replay_status": trained.fold_base_replay["status"],
        "frozen_base_optimizer_step_count": 0,
        "fresh_zero_residual_replay_state_count": zero_count,
        "historical_fresh_decision_artifact_read_count": 0,
        "fresh_label_access_count": 0,
        "confirm20_access_count": 0,
    }


def train_and_seal(
    *,
    formal_feature_archive: bytes,
    formal_label_archive: bytes,
    fresh_feature_payload: bytes,
    base_manifest_payload: bytes,
    base_checkpoint_payloads: Mapping[str, bytes],
    output_dir: Path,
    source_a_git_commit: str,
    execution_b_git_commit: str,
    runner_freeze_sha256: str,
    contract_sha256: str,
    maximum_epochs: int = MAXIMUM_EPOCHS,
    patience_epochs: int = PATIENCE_EPOCHS,
) -> Mapping[str, Any]:
    """Replay frozen bases, fit only residual heads, predict features, then seal."""
    formal_states, _source_ids, folds = load_formal58_archives(
        formal_feature_archive,
        formal_label_archive,
    )
    features = load_fresh_features(fresh_feature_payload)
    ensemble = load_frozen_base_ensemble(
        base_manifest_payload,
        base_checkpoint_payloads,
    )
    trained = train_v4(
        formal_states,
        folds=folds,
        frozen_bases=ensemble,
        maximum_epochs=maximum_epochs,
        patience_epochs=patience_epochs,
    )
    residual_payloads, residual_inventory = serialize_trained_v4(trained)
    from causalcache.set_conditioned_v4_training import predict_state

    predictions = tuple(
        tuple(
            _prediction_from_training(predict_state(item.base, item.head, feature))
            for item in trained.models
        )
        for feature in features
    )
    prediction_records = tuple(
        state_prediction_record(feature, rows)
        for feature, rows in zip(features, predictions, strict=True)
    )
    prediction_payload = prediction_artifact_bytes(
        features,
        predictions,
        base_inventory=ensemble.checkpoint_inventory,
        residual_inventory=residual_inventory,
    )
    formal_report = training_report(trained, formal_states=formal_states)
    formal_report_payload = canonical_json_line(formal_report)
    invariant = base_invariant_report(
        ensemble,
        trained=trained,
        prediction_records=prediction_records,
    )
    invariant_payload = canonical_json_line(invariant)
    payloads = {
        **residual_payloads,
        "fresh16-feature-only-predictions.json": prediction_payload,
        "formal58-residual-training-report.json": formal_report_payload,
        "base-invariant-report.json": invariant_payload,
    }
    expected = {
        *(f"residual-checkpoints/seed-{seed}.safetensors" for seed in SEEDS),
        "residual-model-metadata.json",
        "fresh16-feature-only-predictions.json",
        "formal58-residual-training-report.json",
        "base-invariant-report.json",
    }
    if set(payloads) != expected:
        raise RuntimeError("v4 label-blind payload inventory drifted")
    return seal_label_blind_outputs(
        output_dir,
        payloads,
        training_report={
            "path": "formal58-residual-training-report.json",
            "sha256": sha256_bytes(formal_report_payload),
            "selected_learning_rate": trained.selection.learning_rate,
            "five_seed_mean_oof_score": trained.selection.five_seed_mean_score,
        },
        base_invariant_report={
            "path": "base-invariant-report.json",
            "sha256": sha256_bytes(invariant_payload),
            "fresh_zero_residual_replay_state_count": 48,
        },
        frozen_base_input_inventory=ensemble.checkpoint_inventory,
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
                raise ValueError("v4 metric value is non-finite")
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
        raise ValueError("v4 metric ratio is non-finite")
    return value


def selection_metric_summary(
    states: Sequence[GateState],
    selections: Mapping[str, Mapping[str, Sequence[int]]],
) -> Mapping[str, Any]:
    if tuple(selections) != METHODS:
        raise ValueError("v4 selection method inventory or order drifted")
    state_ids = {state.state_id for state in states}
    if any(set(rows) != state_ids for rows in selections.values()):
        raise ValueError("v4 selection state inventory drifted")

    def summarize(selected_states: Sequence[GateState]) -> Mapping[str, Any]:
        selected_states = tuple(selected_states)
        methods = {}
        for name in METHODS:
            raw: dict[str, float] = {}
            normalized: dict[str, float | None] = {}
            for state in selected_states:
                coalition = tuple(selections[name][state.state_id])
                if coalition not in feasible_coalitions_from_ids(
                    state.candidate_event_step_ids
                ):
                    raise ValueError(f"{name} emitted an infeasible v4 coalition")
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
            methods[name] = {
                "mean_raw_utility": raw_mean,
                "mean_normalized_recovery": norm_mean,
                "normalized_excluded_state_count": excluded_states,
                "normalized_excluded_trajectory_count": excluded_trajectories,
            }
        exact = methods["exact"]
        for value in methods.values():
            value["raw_ratio_to_exact"] = _ratio(
                value["mean_raw_utility"],
                exact["mean_raw_utility"],
            )
            value["normalized_ratio_to_exact"] = _ratio(
                value["mean_normalized_recovery"],
                exact["mean_normalized_recovery"],
            )
        return {
            "state_count": len(selected_states),
            "trajectory_count": len({state.source_id for state in selected_states}),
            "methods": methods,
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
        raise ValueError("v4 quantile input is malformed")
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
        raise ValueError("v4 paired bootstrap input is malformed")
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
        delta = state.table.utility(selections[left][state.state_id]) - state.table.utility(
            selections[right][state.state_id]
        )
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
    return {
        f"{left}_minus_{right}": {
            scale: paired_trajectory_bootstrap(
                _trajectory_deltas(
                    states,
                    selections,
                    left=left,
                    right=right,
                    normalized=scale == "normalized",
                )
            )
            for scale in ("raw", "normalized")
        }
        for left, right in (
            ("v4_unguarded", "frozen_base"),
            ("v4_safe", "frozen_base"),
            ("v4_safe", "historical_v1_independent"),
        )
    }


def frozen_base_residual_interpretation(
    metrics: Mapping[str, Any],
    bootstraps: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Apply the five pre-frozen development checks to safe-minus-base."""
    primary = bootstraps["v4_safe_minus_frozen_base"]
    safe_overall = metrics["overall"]["methods"]["v4_safe"]
    base_overall = metrics["overall"]["methods"]["frozen_base"]
    safe_n4 = metrics["by_event_count"]["4"]["methods"]["v4_safe"]
    base_n4 = metrics["by_event_count"]["4"]["methods"]["frozen_base"]
    normalized_delta = (
        safe_overall["mean_normalized_recovery"]
        - base_overall["mean_normalized_recovery"]
    )
    raw_delta = safe_overall["mean_raw_utility"] - base_overall["mean_raw_utility"]
    n4_delta = (
        safe_n4["mean_normalized_recovery"]
        - base_n4["mean_normalized_recovery"]
    )
    checks = {
        "mean_normalized_delta_at_least_0_01": normalized_delta >= 0.01,
        "normalized_bootstrap_lower_strictly_positive": (
            primary["normalized"]["lower"] > 0.0
        ),
        "mean_raw_delta_strictly_positive": raw_delta > 0.0,
        "n4_mean_normalized_delta_strictly_positive": n4_delta > 0.0,
        "at_least_8_positive_trajectories": (
            primary["normalized"]["positive_trajectory_count"] >= 8
        ),
    }
    value = (
        "PROMISING_DEVELOPMENT_SIGNAL_FOR_A_SEPARATELY_FROZEN_FUTURE_STUDY"
        if all(checks.values())
        else "NO_DEVELOPMENT_EVIDENCE_FOR_FROZEN_BASE_RESIDUAL"
    )
    return {
        "value": value,
        "primary_contrast": "v4_safe_minus_frozen_base",
        "mean_normalized_delta": normalized_delta,
        "mean_raw_delta": raw_delta,
        "n4_mean_normalized_delta": n4_delta,
        "checks": checks,
        "may_authorize_confirm20": False,
        "may_change_v1_or_v3_verdict": False,
        "may_support_another_v5_retune_on_fresh16": False,
    }


def switch_diagnostics(
    states: Sequence[GateState],
    selections: Mapping[str, Mapping[str, Sequence[int]]],
    prediction_records: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    rows = []
    for state in states:
        record = prediction_records[state.state_id]
        base = tuple(selections["frozen_base"][state.state_id])
        unguarded = tuple(selections["v4_unguarded"][state.state_id])
        safe = tuple(selections["v4_safe"][state.state_id])
        delta = state.table.utility(safe) - state.table.utility(base)
        rows.append(
            {
                "state_id": state.state_id,
                "event_count": len(state.candidate_event_step_ids),
                "used_pair_candidate": bool(
                    record["safe_decision"]["used_pair_candidate"]
                ),
                "used_fallback": bool(record["safe_decision"]["used_fallback"]),
                "unguarded_changed_base": unguarded != base,
                "safe_changed_base": safe != base,
                "safe_minus_base_raw": delta,
                "outcome": (
                    "improved" if delta > 0.0 else "harmed" if delta < 0.0 else "equal"
                ),
                "reason": str(record["safe_decision"]["reason"]),
            }
        )

    def summarize(selected: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        selected = tuple(selected)
        return {
            "state_count": len(selected),
            "used_pair_candidate_count": sum(
                bool(row["used_pair_candidate"]) for row in selected
            ),
            "fallback_count": sum(bool(row["used_fallback"]) for row in selected),
            "pair_equals_base_count": sum(
                not bool(row["used_pair_candidate"])
                and not bool(row["used_fallback"])
                for row in selected
            ),
            "unguarded_changed_base_count": sum(
                bool(row["unguarded_changed_base"]) for row in selected
            ),
            "safe_changed_base_count": sum(
                bool(row["safe_changed_base"]) for row in selected
            ),
            "improved_count": sum(row["outcome"] == "improved" for row in selected),
            "harmed_count": sum(row["outcome"] == "harmed" for row in selected),
            "equal_count": sum(row["outcome"] == "equal" for row in selected),
            "reason_counts": dict(
                sorted(Counter(row["reason"] for row in selected).items())
            ),
        }

    return {
        "overall": summarize(rows),
        "by_event_count": {
            str(event_count): summarize(
                tuple(row for row in rows if row["event_count"] == event_count)
            )
            for event_count in (2, 3, 4)
        },
    }


def _regular_file_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"v4 input is missing or unsafe: {path}")
    return path.read_bytes()


def _write_exclusive_or_identical(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError(f"pre-existing v4 output differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("v4 output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ValueError(f"one-shot v4 output already exists: {path}") from error
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("one-shot v4 output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _valid_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def seal_label_blind_outputs(
    output_dir: Path,
    payloads: Mapping[str, bytes],
    *,
    training_report: Mapping[str, Any],
    base_invariant_report: Mapping[str, Any],
    frozen_base_input_inventory: Sequence[Mapping[str, Any]],
    source_a_git_commit: str,
    execution_b_git_commit: str,
    runner_freeze_sha256: str,
    contract_sha256: str,
) -> Mapping[str, Any]:
    """Persist residual outputs and bind immutable base inputs before labels."""
    if not payloads or any(
        Path(name).is_absolute() or ".." in Path(name).parts for name in payloads
    ):
        raise ValueError("v4 label-blind payload paths are unsafe")
    if not (
        _valid_hex(source_a_git_commit, 40)
        and _valid_hex(execution_b_git_commit, 40)
        and _valid_hex(runner_freeze_sha256, 64)
        and _valid_hex(contract_sha256, 64)
    ):
        raise ValueError("v4 Git or contract execution binding is malformed")
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
        "status": "SEALED_LABEL_BLIND_SET_CONDITIONED_V4_FROZEN_BASE_RESIDUAL_V1",
        "training_report": dict(training_report),
        "base_invariant_report": dict(base_invariant_report),
        "frozen_base_manifest_sha256": BASE_MANIFEST_SHA256,
        "frozen_base_input_inventory": list(frozen_base_input_inventory),
        "source_a_git_commit": source_a_git_commit,
        "execution_b_git_commit": execution_b_git_commit,
        "runner_freeze_sha256": runner_freeze_sha256,
        "contract_sha256": contract_sha256,
        "payload_inventory": inventory,
        "fresh_label_access_count": 0,
        "fresh_label_semantic_decode_attempt_count": 0,
        "confirm20_access_count": 0,
        "legacy_dev5_access_count": 0,
        "raw_gui_access_count": 0,
        "policy_forward_count": 0,
        "gpu_operation_count": 0,
    }
    payload = canonical_json_line(seal)
    _write_exclusive_or_identical(output_dir / "label-blind-seal.json", payload)
    return {**seal, "seal_sha256": sha256_bytes(payload)}


def verify_label_blind_seal(output_dir: Path) -> Mapping[str, Any]:
    payload = _regular_file_bytes(output_dir / "label-blind-seal.json")
    seal = _strict_json(payload, label="v4 label-blind seal")
    if (
        seal.get("protocol_id") != PROTOCOL_ID
        or seal.get("status")
        != "SEALED_LABEL_BLIND_SET_CONDITIONED_V4_FROZEN_BASE_RESIDUAL_V1"
        or seal.get("frozen_base_manifest_sha256") != BASE_MANIFEST_SHA256
        or seal.get("fresh_label_access_count") != 0
        or seal.get("fresh_label_semantic_decode_attempt_count") != 0
        or seal.get("confirm20_access_count") != 0
        or not _valid_hex(seal.get("source_a_git_commit"), 40)
        or not _valid_hex(seal.get("execution_b_git_commit"), 40)
        or not _valid_hex(seal.get("runner_freeze_sha256"), 64)
        or not _valid_hex(seal.get("contract_sha256"), 64)
    ):
        raise ValueError("v4 label-blind seal schema or firewall drifted")
    inventory = seal.get("payload_inventory")
    if not isinstance(inventory, list):
        raise ValueError("v4 label-blind payload inventory is malformed")
    for record in inventory:
        if not isinstance(record, Mapping):
            raise ValueError("v4 label-blind payload record is malformed")
        observed = _regular_file_bytes(output_dir / str(record["path"]))
        if (
            sha256_bytes(observed) != record.get("sha256")
            or len(observed) != record.get("size_bytes")
        ):
            raise ValueError("sealed v4 label-blind payload changed")
    return seal


def claim_fresh_label_access(output_dir: Path) -> Mapping[str, Any]:
    """Persist the sole v4 consumed-development claim without opening labels."""
    seal = verify_label_blind_seal(output_dir)
    seal_payload = _regular_file_bytes(output_dir / "label-blind-seal.json")
    claim = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "CLAIMED_FRESH16_CONSUMED_DEVELOPMENT_ACCESS_SET_CONDITIONED_V4_V1",
        "label_blind_seal_sha256": sha256_bytes(seal_payload),
        "expected_fresh_label_sha256": FRESH_LABEL_SHA256,
        "expected_historical_independent_sha256": HISTORICAL_INDEPENDENT_SHA256,
        "fresh_label_access_claim_count": 1,
        "fresh_label_semantic_decode_attempt_count": 0,
        "confirm20_access_count": 0,
        "legacy_dev5_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
        "raw_gui_access_count": 0,
        "policy_forward_count": 0,
        "gpu_operation_count": 0,
        "sealed_payload_count": len(seal["payload_inventory"]),
    }
    payload = canonical_json_line(claim)
    _write_once(
        output_dir / "fresh-label-access-claim.json",
        payload,
    )
    return {**claim, "claim_sha256": sha256_bytes(payload)}


def verify_fresh_label_access_claim(output_dir: Path) -> Mapping[str, Any]:
    verify_label_blind_seal(output_dir)
    seal_payload = _regular_file_bytes(output_dir / "label-blind-seal.json")
    payload = _regular_file_bytes(output_dir / "fresh-label-access-claim.json")
    claim = _strict_json(payload, label="v4 fresh label access claim")
    if (
        claim.get("protocol_id") != PROTOCOL_ID
        or claim.get("status")
        != "CLAIMED_FRESH16_CONSUMED_DEVELOPMENT_ACCESS_SET_CONDITIONED_V4_V1"
        or claim.get("label_blind_seal_sha256") != sha256_bytes(seal_payload)
        or claim.get("expected_fresh_label_sha256") != FRESH_LABEL_SHA256
        or claim.get("expected_historical_independent_sha256")
        != HISTORICAL_INDEPENDENT_SHA256
        or claim.get("fresh_label_access_claim_count") != 1
        or claim.get("fresh_label_semantic_decode_attempt_count") != 0
        or claim.get("confirm20_access_count") != 0
    ):
        raise ValueError("v4 fresh label access claim schema or firewall drifted")
    return claim


def _historical_decision_details(
    payload: bytes,
    *,
    features: Sequence[FeatureState],
) -> tuple[Mapping[str, tuple[int, ...]], tuple[Mapping[str, tuple[int, ...]], ...]]:
    ensemble = load_historical_v1_independent_parser_repair_v2(
        payload,
        feature_states=features,
    )
    value = _strict_json(payload, label="historical v1 independent decisions")
    records = value["records"]
    records_by_id = {
        record["state_id"]: record
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("state_id"), str)
    }
    if len(records_by_id) != len(features) or set(records_by_id) != {
        feature.state_id for feature in features
    }:
        raise ValueError("historical seed decisions do not share the Fresh state set")
    per_seed = [dict() for _ in SEEDS]
    for feature in features:
        record = records_by_id[feature.state_id]
        for seed, seed_record in zip(
            SEEDS,
            record["seed_selected_event_step_ids"],
            strict=True,
        ):
            per_seed[seed][feature.state_id] = tuple(
                seed_record["selected_event_step_ids"]
            )
    return ensemble, tuple(per_seed)


def verify_historical_base_replay(
    prediction_records: Mapping[str, Mapping[str, Any]],
    historical_ensemble: Mapping[str, Sequence[int]],
    historical_by_seed: Sequence[Mapping[str, Sequence[int]]],
) -> Mapping[str, Any]:
    if set(prediction_records) != set(historical_ensemble) or len(historical_by_seed) != 5:
        raise ValueError("v4 historical base replay inventory drifted")
    for state_id, record in prediction_records.items():
        if tuple(record["frozen_base_selected"]) != tuple(
            historical_ensemble[state_id]
        ):
            raise ValueError("v4 frozen ensemble differs from historical independent")
        seed_rows = record["seed_predictions"]
        for seed, row in zip(SEEDS, seed_rows, strict=True):
            if tuple(row["frozen_base_selected"]) != tuple(
                historical_by_seed[seed][state_id]
            ):
                raise ValueError("v4 frozen seed differs from historical independent")
    return {
        "status": "REPLAYED_HISTORICAL_FRESH16_INDEPENDENT_DECISIONS_V1",
        "ensemble_state_count": len(prediction_records),
        "seed_state_decision_count": len(prediction_records) * 5,
        "ensemble_selection_sha256": canonical_selection_sha256(
            {
                state_id: tuple(record["frozen_base_selected"])
                for state_id, record in prediction_records.items()
            }
        ),
        "historical_artifact_sha256": HISTORICAL_INDEPENDENT_SHA256,
    }


def build_evaluation_report(
    states: Sequence[GateState],
    *,
    prediction_records: Mapping[str, Mapping[str, Any]],
    historical_v1_independent: Mapping[str, Sequence[int]],
    historical_replay: Mapping[str, Any],
    label_blind_seal_sha256: str,
    label_access_claim_sha256: str,
) -> Mapping[str, Any]:
    state_ids = {state.state_id for state in states}
    if set(prediction_records) != state_ids or set(historical_v1_independent) != state_ids:
        raise ValueError("v4 evaluation decision inventory differs from labels")
    selections: dict[str, dict[str, tuple[int, ...]]] = {
        name: {} for name in METHODS
    }
    for state in states:
        record = prediction_records[state.state_id]
        selections["exact"][state.state_id] = primary_exact_subset_oracle(
            state.table
        ).coalition
        selections["frozen_base"][state.state_id] = tuple(
            record["frozen_base_selected"]
        )
        selections["v4_unguarded"][state.state_id] = tuple(
            record["v4_unguarded_selected"]
        )
        selections["v4_safe"][state.state_id] = tuple(record["v4_safe_selected"])
        selections["historical_v1_independent"][state.state_id] = tuple(
            historical_v1_independent[state.state_id]
        )
    ordered = {name: selections[name] for name in METHODS}
    metrics = selection_metric_summary(states, ordered)
    frozen = metrics["overall"]["methods"]["frozen_base"]
    historical = metrics["overall"]["methods"]["historical_v1_independent"]
    if frozen != historical:
        raise ValueError("v4 frozen base metric differs from historical independent")
    if not math.isclose(
        float(frozen["mean_normalized_recovery"]),
        0.7123785840017873,
        rel_tol=0.0,
        abs_tol=1e-15,
    ) or not math.isclose(
        float(frozen["mean_raw_utility"]),
        0.04046445946535945,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("v4 frozen base failed the historical fresh16 metric replay")
    bootstraps = bootstrap_summary(states, ordered)
    switches = switch_diagnostics(states, ordered, prediction_records)
    interpretation = frozen_base_residual_interpretation(metrics, bootstraps)
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
                        "selected": list(ordered[name][state.state_id]),
                        "raw_utility": state.table.utility(
                            ordered[name][state.state_id]
                        ),
                        "normalized_recovery": (
                            state.table.utility(ordered[name][state.state_id]) / baseline
                            if baseline > NORMALIZATION_EPSILON
                            else None
                        ),
                    }
                    for name in METHODS
                },
                "safe_decision": dict(
                    prediction_records[state.state_id]["safe_decision"]
                ),
            }
        )
    without_hash = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETED_SET_CONDITIONED_V4_FRESH16_CONSUMED_DEVELOPMENT_V1",
        "scope": "consumed_development_exploration_not_holdout_not_test",
        "label_blind_seal_sha256": label_blind_seal_sha256,
        "label_access_claim_sha256": label_access_claim_sha256,
        "fresh_feature_sha256": FRESH_FEATURE_SHA256,
        "fresh_label_sha256": FRESH_LABEL_SHA256,
        "historical_v1_independent_sha256": HISTORICAL_INDEPENDENT_SHA256,
        "historical_base_replay": dict(historical_replay),
        "trajectory_count": 16,
        "state_count": 48,
        "metrics": metrics,
        "bootstrap": {
            "unit": "trajectory",
            "interval": "two_sided_percentile",
            "comparisons": bootstraps,
        },
        "switch_diagnostics": switches,
        "development_interpretation": interpretation,
        "state_records": state_rows,
        "v4_fresh_label_access_claim_count": 1,
        "v4_fresh_label_semantic_decode_attempt_count": 1,
        "accounting_scope": "v3_to_v4_lineage",
        "lineage_totals_are_project_global": False,
        "v3_to_v4_lineage_claim_count": 2,
        "v3_to_v4_lineage_semantic_decode_attempt_count": 3,
        "fresh_decoded_state_count": 48,
        "confirm20_access_count": 0,
        "legacy_dev5_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
        "raw_gui_access_count": 0,
        "policy_forward_count": 0,
        "gpu_operation_count": 0,
    }
    return {
        **without_hash,
        "report_sha256": sha256_bytes(canonical_json_bytes(without_hash)),
    }


def evaluate_sealed_fresh16(
    *,
    output_dir: Path,
    feature_payload: bytes,
    label_payload: bytes,
    historical_independent_payload: bytes,
    prevalidated_features: Sequence[FeatureState] | None = None,
) -> Mapping[str, Any]:
    """After seal and claim, perform one consumed-development join and report."""
    verify_label_blind_seal(output_dir)
    claim = verify_fresh_label_access_claim(output_dir)
    seal_payload = _regular_file_bytes(output_dir / "label-blind-seal.json")
    claim_payload = _regular_file_bytes(output_dir / "fresh-label-access-claim.json")
    prediction_payload = _regular_file_bytes(
        output_dir / "fresh16-feature-only-predictions.json"
    )
    if prevalidated_features is None:
        features = load_fresh_features(feature_payload)
    else:
        if sha256_bytes(feature_payload) != FRESH_FEATURE_SHA256:
            raise ValueError("v4 prevalidated Fresh feature bytes drifted")
        features = tuple(prevalidated_features)
        if len(features) != 48:
            raise ValueError("v4 prevalidated Fresh feature denominator drifted")
    predictions = read_prediction_artifact(prediction_payload, features=features)
    if claim.get("expected_fresh_label_sha256") != sha256_bytes(label_payload):
        raise ValueError("v4 fresh label differs from pre-access claim")
    if claim.get("expected_historical_independent_sha256") != sha256_bytes(
        historical_independent_payload
    ):
        raise ValueError("v4 historical decision differs from pre-access claim")
    historical, historical_by_seed = _historical_decision_details(
        historical_independent_payload,
        features=features,
    )
    labels = read_label_states_jsonl(label_payload)
    source_ids = tuple(dict.fromkeys(feature.source_id for feature in features))
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
    replay = verify_historical_base_replay(
        predictions,
        historical,
        historical_by_seed,
    )
    report = build_evaluation_report(
        states,
        prediction_records=predictions,
        historical_v1_independent=historical,
        historical_replay=replay,
        label_blind_seal_sha256=sha256_bytes(seal_payload),
        label_access_claim_sha256=sha256_bytes(claim_payload),
    )
    payload = canonical_json_line(report)
    _write_once(
        output_dir / "fresh16-consumed-development-report.json",
        payload,
    )
    return report


def verify_evaluation_report(output_dir: Path) -> Mapping[str, Any]:
    """Validate only sealed report bytes; never reopen feature, label, or history."""
    payload = _regular_file_bytes(
        output_dir / "fresh16-consumed-development-report.json"
    )
    report = _strict_json(payload, label="v4 consumed-development report")
    report_without_hash = dict(report)
    internal = report_without_hash.pop("report_sha256", None)
    interpretation = report.get("development_interpretation")
    valid_interpretations = {
        "NO_DEVELOPMENT_EVIDENCE_FOR_FROZEN_BASE_RESIDUAL",
        "PROMISING_DEVELOPMENT_SIGNAL_FOR_A_SEPARATELY_FROZEN_FUTURE_STUDY",
    }
    if (
        report.get("protocol_id") != PROTOCOL_ID
        or report.get("status")
        != "COMPLETED_SET_CONDITIONED_V4_FRESH16_CONSUMED_DEVELOPMENT_V1"
        or internal != sha256_bytes(canonical_json_bytes(report_without_hash))
        or not _valid_hex(report.get("label_blind_seal_sha256"), 64)
        or not _valid_hex(report.get("label_access_claim_sha256"), 64)
        or report.get("fresh_feature_sha256") != FRESH_FEATURE_SHA256
        or report.get("fresh_label_sha256") != FRESH_LABEL_SHA256
        or report.get("historical_v1_independent_sha256")
        != HISTORICAL_INDEPENDENT_SHA256
        or report.get("trajectory_count") != 16
        or report.get("state_count") != 48
        or report.get("v4_fresh_label_access_claim_count") != 1
        or report.get("v4_fresh_label_semantic_decode_attempt_count") != 1
        or report.get("accounting_scope") != "v3_to_v4_lineage"
        or report.get("lineage_totals_are_project_global") is not False
        or report.get("v3_to_v4_lineage_claim_count") != 2
        or report.get("v3_to_v4_lineage_semantic_decode_attempt_count") != 3
        or report.get("confirm20_access_count") != 0
        or report.get("legacy_dev5_access_count") != 0
        or report.get("matched_nll_evaluation_count") != 0
        or report.get("closed_loop_episode_count") != 0
        or report.get("raw_gui_access_count") != 0
        or report.get("policy_forward_count") != 0
        or report.get("gpu_operation_count") != 0
        or not isinstance(interpretation, Mapping)
        or interpretation.get("value") not in valid_interpretations
        or interpretation.get("primary_contrast")
        != "v4_safe_minus_frozen_base"
        or interpretation.get("may_authorize_confirm20") is not False
        or interpretation.get("may_change_v1_or_v3_verdict") is not False
        or interpretation.get("may_support_another_v5_retune_on_fresh16")
        is not False
    ):
        raise ValueError("v4 consumed-development report replay drifted")
    return report


__all__ = [
    "BASE_CHECKPOINT_RECORDS",
    "BASE_MANIFEST_SHA256",
    "BASE_OOF_MEAN_RAW_RATIO",
    "BASE_OOF_RAW_RATIOS",
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
    "ResidualStatePrediction",
    "SEEDS",
    "TrainedV4",
    "aggregate_seed_predictions",
    "base_invariant_report",
    "base_selection_from_scores",
    "bootstrap_summary",
    "build_evaluation_report",
    "canonical_json_line",
    "choose_oof_selection",
    "claim_fresh_label_access",
    "evaluate_sealed_fresh16",
    "frozen_base_residual_interpretation",
    "load_frozen_base_ensemble",
    "load_fresh_features",
    "oof_selection_payload",
    "paired_trajectory_bootstrap",
    "prediction_artifact_bytes",
    "prepare_fold_clean_bases",
    "read_prediction_artifact",
    "run_oof_trial",
    "seal_label_blind_outputs",
    "selection_metric_summary",
    "set_score_maps",
    "sha256_bytes",
    "state_prediction_record",
    "switch_diagnostics",
    "train_and_seal",
    "train_v4",
    "verify_evaluation_report",
    "verify_fresh_label_access_claim",
    "verify_historical_base_replay",
    "verify_label_blind_seal",
    "zero_residual_prediction",
]
