"""Label-blind CPU selectors for variable-length GUI histories.

This module deliberately has no restoration-label dependency.  It converts a
``FeatureState``-like object into a feature-only view, produces selections, and
seals those selections before any distance table is needed by evaluation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cmp_to_key
from numbers import Real
from types import MappingProxyType
from typing import Any

from causalcache.gate_v1_data import CandidateFeatures
from causalcache.gate_v1_provenance import (
    FROZEN_SEEDS,
    FrozenEnsembleProvenance,
    frozen_ensemble_provenance_from_manifest,
    validate_frozen_ensemble,
)
from causalcache.gate_v1_training import FittedEnsemble
from causalcache.restoration_v2_baselines import (
    OCR_JACCARD_WEIGHT,
    RGB_HISTOGRAM_WEIGHT,
    SCORE_TIE_ABS_TOL,
    SCORE_TIE_REL_TOL,
    joint_rgb_histogram_cosine,
    normalized_ocr_token_set,
)


SUPPORTED_BUDGETS = (2, 4)
CONDITIONAL_BUDGET = 2
RESIDUAL_BUDGET = 2
SAFE_VOTE_THRESHOLD = 4
SAFE_VOTE_TOTAL = 5
SELECTION_SEAL_SCHEMA_VERSION = "1.0.0"
SELECTION_SEAL_PROTOCOL_ID = "causalcache_long_horizon_label_blind_selection"
SELECTION_SEAL_STATUS = "SEALED_LONG_HORIZON_LABEL_BLIND_SELECTIONS_V1"
CANONICAL_SELECTOR_NAMES = (
    "restoration_independent_gate",
    "recent",
    "ocr_rgb_v2",
    "v1_conditional",
    "v4_safe_frozen_base_residual",
    "random",
    "summary_only",
)

Score = Callable[[Any, int, tuple[int, ...]], float]
_SEAL_TOKEN = object()


def _finite_vector(values: Any, dimension: int, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{label} must be a numeric sequence")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be a numeric sequence") from error
    if len(result) != dimension or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain {dimension} finite values")
    return result


def _event_ids(values: Any) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError("candidate event ids must be an ordered sequence")
    try:
        result = tuple(values)
    except TypeError as error:
        raise TypeError("candidate event ids must be an ordered sequence") from error
    if not result:
        raise ValueError("long-horizon state must contain at least one candidate")
    if any(type(event) is not int for event in result):
        raise TypeError("candidate event ids must be integers")
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise ValueError("candidate event ids must be unique and strictly increasing")
    if result[0] <= 0:
        raise ValueError("candidate event ids must be positive")
    return result


def _budget(value: Any) -> int:
    if type(value) is not int:
        raise TypeError("long-horizon budget must be an integer")
    if value not in SUPPORTED_BUDGETS:
        raise ValueError(f"long-horizon budget must be one of {SUPPORTED_BUDGETS}")
    return value


def _finite_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class LongHorizonFeatureView:
    """An arbitrary-``n`` view containing only frozen gate-v1 inputs."""

    source_id: str
    state_id: str
    decision_step_id: int
    candidate_event_step_ids: tuple[int, ...]
    q64: tuple[float, ...]
    candidates: tuple[CandidateFeatures, ...]

    def candidate(self, event_step_id: int) -> CandidateFeatures:
        for candidate in self.candidates:
            if candidate.event_step_id == event_step_id:
                return candidate
        raise KeyError(event_step_id)


def feature_view(state: Any) -> LongHorizonFeatureView:
    """Copy a ``FeatureState``-like value without carrying a label table."""
    source_id = getattr(state, "source_id", None)
    state_id = getattr(state, "state_id", None)
    decision_step_id = getattr(state, "decision_step_id", None)
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("feature view source_id must be non-empty text")
    if not isinstance(state_id, str) or not state_id:
        raise ValueError("feature view state_id must be non-empty text")
    if type(decision_step_id) is not int or decision_step_id <= 0:
        raise ValueError("feature view decision_step_id must be positive")
    event_ids = _event_ids(getattr(state, "candidate_event_step_ids", None))
    if event_ids[-1] >= decision_step_id:
        raise ValueError("candidate events must precede the decision step")
    q64 = _finite_vector(getattr(state, "q64", None), 64, "feature view q64")
    raw_candidates = getattr(state, "candidates", None)
    if isinstance(raw_candidates, (str, bytes, bytearray, Mapping)):
        raise TypeError("feature candidates must be an ordered sequence")
    try:
        candidates = tuple(raw_candidates)
    except TypeError as error:
        raise TypeError("feature candidates must be an ordered sequence") from error
    if tuple(getattr(item, "event_step_id", None) for item in candidates) != event_ids:
        raise ValueError("feature candidate inventory differs from candidate event ids")
    copied: list[CandidateFeatures] = []
    for candidate in candidates:
        copied.append(
            CandidateFeatures(
                event_step_id=candidate.event_step_id,
                h64=_finite_vector(candidate.h64, 64, "candidate h64"),
                g8=_finite_vector(candidate.g8, 8, "candidate g8"),
            )
        )
    return LongHorizonFeatureView(
        source_id=source_id,
        state_id=state_id,
        decision_step_id=decision_step_id,
        candidate_event_step_ids=event_ids,
        q64=q64,
        candidates=tuple(copied),
    )


@dataclass(frozen=True)
class ConditionalRound:
    round_index: int
    selected_before: tuple[int, ...]
    candidate_scores: tuple[tuple[int, float], ...]
    selected_event_step_id: int | None


@dataclass(frozen=True)
class SelectionDecision:
    selector_name: str
    source_id: str
    state_id: str
    decision_step_id: int
    candidate_event_step_ids: tuple[int, ...]
    budget_event_capacity: int
    selected_event_step_ids: tuple[int, ...]


@dataclass(frozen=True)
class ConditionalB2Result:
    decision: SelectionDecision
    trace: tuple[ConditionalRound, ...]


def _decision(
    view: LongHorizonFeatureView,
    *,
    selector_name: str,
    budget: int,
    selected: Sequence[int],
) -> SelectionDecision:
    capacity = _budget(budget)
    result = tuple(selected)
    if (
        result != tuple(sorted(result))
        or len(set(result)) != len(result)
        or len(result) > capacity
        or not set(result).issubset(view.candidate_event_step_ids)
    ):
        raise ValueError("selector produced an infeasible event subset")
    if not isinstance(selector_name, str) or not selector_name:
        raise ValueError("selector name must be non-empty text")
    return SelectionDecision(
        selector_name=selector_name,
        source_id=view.source_id,
        state_id=view.state_id,
        decision_step_id=view.decision_step_id,
        candidate_event_step_ids=view.candidate_event_step_ids,
        budget_event_capacity=capacity,
        selected_event_step_ids=result,
    )


def independent_selection(
    state: Any,
    score: Score,
    *,
    budget_event_capacity: int,
) -> SelectionDecision:
    """Apply positive-score top-B with lower event id as the exact tie break."""
    view = feature_view(state)
    budget = _budget(budget_event_capacity)
    scores = {
        event: _finite_score(score(view, event, ()), f"independent event {event} score")
        for event in view.candidate_event_step_ids
    }
    ranked = sorted(
        (event for event, value in scores.items() if value > 0.0),
        key=lambda event: (-scores[event], event),
    )
    return _decision(
        view,
        selector_name="restoration_independent_gate",
        budget=budget,
        selected=tuple(sorted(ranked[:budget])),
    )


def conditional_b2_selection(
    state: Any,
    score: Score,
    *,
    budget_event_capacity: int = CONDITIONAL_BUDGET,
) -> ConditionalB2Result:
    """Run the frozen v1 iterative selector; B=4 is intentionally unsupported."""
    view = feature_view(state)
    if type(budget_event_capacity) is not int:
        raise TypeError("conditional budget must be an integer")
    if budget_event_capacity != CONDITIONAL_BUDGET:
        raise ValueError("v1 conditional is an allowed B=2 comparator only")
    selected: tuple[int, ...] = ()
    trace: list[ConditionalRound] = []
    for round_index in range(CONDITIONAL_BUDGET):
        scores = tuple(
            (
                event,
                _finite_score(
                    score(view, event, selected),
                    f"conditional event {event} score",
                ),
            )
            for event in view.candidate_event_step_ids
            if event not in selected
        )
        event, predicted = min(scores, key=lambda item: (-item[1], item[0]))
        chosen = event if predicted > 0.0 else None
        trace.append(
            ConditionalRound(
                round_index=round_index,
                selected_before=selected,
                candidate_scores=scores,
                selected_event_step_id=chosen,
            )
        )
        if chosen is None:
            break
        selected = tuple(sorted((*selected, chosen)))
    return ConditionalB2Result(
        decision=_decision(
            view,
            selector_name="v1_conditional",
            budget=CONDITIONAL_BUDGET,
            selected=selected,
        ),
        trace=tuple(trace),
    )


def recent_selection(
    state: Any,
    *,
    budget_event_capacity: int,
) -> SelectionDecision:
    view = feature_view(state)
    budget = _budget(budget_event_capacity)
    return _decision(
        view,
        selector_name="recent",
        budget=budget,
        selected=view.candidate_event_step_ids[-min(budget, len(view.candidate_event_step_ids)) :],
    )


def deterministic_random_selection(
    state: Any,
    *,
    budget_event_capacity: int,
    seed: int,
) -> SelectionDecision:
    """Sample without replacement from a state-local deterministic RNG."""
    view = feature_view(state)
    budget = _budget(budget_event_capacity)
    if type(seed) is not int or seed < 0:
        raise ValueError("random selector seed must be a non-negative integer")
    identity = json.dumps(
        {
            "seed": seed,
            "source_id": view.source_id,
            "state_id": view.state_id,
            "candidate_event_step_ids": list(view.candidate_event_step_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(identity).digest(), "big")
    generator = random.Random(local_seed)
    count = min(budget, len(view.candidate_event_step_ids))
    selected = tuple(sorted(generator.sample(view.candidate_event_step_ids, count)))
    return _decision(
        view,
        selector_name="random",
        budget=budget,
        selected=selected,
    )


def ocr_rgb_score_selection(
    state: Any,
    scores_by_event_step: Mapping[int, Real],
    *,
    budget_event_capacity: int,
) -> SelectionDecision:
    """Consume precomputed label-blind OCR/RGB scores and force-fill top-B."""
    view = feature_view(state)
    budget = _budget(budget_event_capacity)
    if not isinstance(scores_by_event_step, Mapping):
        raise TypeError("OCR/RGB scores must be an event-indexed mapping")
    if set(scores_by_event_step) != set(view.candidate_event_step_ids):
        raise ValueError("OCR/RGB score inventory differs from feature candidates")
    scores = {
        event: _finite_score(scores_by_event_step[event], f"OCR/RGB event {event} score")
        for event in view.candidate_event_step_ids
    }
    def compare(left: int, right: int) -> int:
        if math.isclose(
            scores[left],
            scores[right],
            rel_tol=SCORE_TIE_REL_TOL,
            abs_tol=SCORE_TIE_ABS_TOL,
        ):
            return -1 if left < right else 1
        return -1 if scores[left] > scores[right] else 1

    ranking = sorted(view.candidate_event_step_ids, key=cmp_to_key(compare))
    selected = tuple(sorted(ranking[: min(budget, len(ranking))]))
    return _decision(
        view,
        selector_name="ocr_rgb_v2",
        budget=budget,
        selected=selected,
    )


def ocr_rgb_similarity_scores(
    *,
    event_step_ids: Sequence[int],
    event_ocr_tokens: Mapping[int, Sequence[str]],
    current_ocr_tokens: Sequence[str],
    event_resized_rgb_bytes: Mapping[int, bytes | bytearray | memoryview],
    current_resized_rgb_bytes: bytes | bytearray | memoryview,
) -> Mapping[int, float]:
    """Compute the frozen OCR-Jaccard/RGB-histogram score for arbitrary ``n``."""
    events = _event_ids(event_step_ids)
    if set(event_ocr_tokens) != set(events) or set(event_resized_rgb_bytes) != set(
        events
    ):
        raise ValueError("OCR/RGB feature inventory differs from candidate events")
    current_tokens = normalized_ocr_token_set(current_ocr_tokens)
    result: dict[int, float] = {}
    for event in events:
        post_tokens = normalized_ocr_token_set(event_ocr_tokens[event])
        if not post_tokens and not current_tokens:
            text_score = 1.0
        else:
            text_score = len(post_tokens & current_tokens) / len(
                post_tokens | current_tokens
            )
        rgb_score = joint_rgb_histogram_cosine(
            event_resized_rgb_bytes[event],
            current_resized_rgb_bytes,
        )
        score = OCR_JACCARD_WEIGHT * text_score + RGB_HISTOGRAM_WEIGHT * rgb_score
        result[event] = _finite_score(score, f"OCR/RGB event {event} score")
    return MappingProxyType(result)


@dataclass(frozen=True)
class ResidualSeedScores:
    seed: int
    singleton_scores: Mapping[int, float]
    pair_residual_scores: Mapping[tuple[int, int], float]


@dataclass(frozen=True)
class ResidualB2Result:
    frozen_base: SelectionDecision
    unguarded: SelectionDecision
    safe: SelectionDecision
    pair_candidate_vote_count: int
    strictly_positive_margin_count: int
    seedwise_pair_minus_base_margins: tuple[float, ...]
    seedwise_unguarded_selections: tuple[tuple[int, ...], ...]
    used_pair_candidate: bool
    used_fallback: bool


@dataclass(frozen=True)
class FrozenFormal58Ensembles:
    conditional: FittedEnsemble
    conditional_provenance: FrozenEnsembleProvenance
    independent: FittedEnsemble
    independent_provenance: FrozenEnsembleProvenance
    checkpoint_load_count: int = 10


def _residual_score_maps(
    view: LongHorizonFeatureView,
    value: ResidualSeedScores,
) -> tuple[dict[tuple[int, ...], float], dict[tuple[int, ...], float]]:
    if type(value.seed) is not int or value.seed not in FROZEN_SEEDS:
        raise ValueError("residual prediction seed is outside the frozen five seeds")
    events = view.candidate_event_step_ids
    pairs = tuple(itertools.combinations(events, 2))
    if set(value.singleton_scores) != set(events):
        raise ValueError("residual singleton score inventory differs from candidates")
    if set(value.pair_residual_scores) != set(pairs):
        raise ValueError("residual pair score inventory differs from all candidate pairs")
    singleton = {
        event: _finite_score(value.singleton_scores[event], f"seed {value.seed} event score")
        for event in events
    }
    residual = {
        pair: _finite_score(value.pair_residual_scores[pair], f"seed {value.seed} pair score")
        for pair in pairs
    }
    base: dict[tuple[int, ...], float] = {(): 0.0}
    base.update({(event,): singleton[event] for event in events})
    base.update({pair: singleton[pair[0]] + singleton[pair[1]] for pair in pairs})
    total = {
        coalition: score + (residual[coalition] if len(coalition) == 2 else 0.0)
        for coalition, score in base.items()
    }
    return base, total


def _predicted_set_argmax(scores: Mapping[tuple[int, ...], float]) -> tuple[int, ...]:
    if not scores or () not in scores:
        raise ValueError("predicted set scores must include the empty coalition")
    if any(not math.isfinite(float(value)) for value in scores.values()):
        raise ValueError("predicted set score is non-finite")
    return min(scores, key=lambda coalition: (-scores[coalition], len(coalition), coalition))


def residual_b2_selections(
    state: Any,
    seed_predictions: Sequence[ResidualSeedScores],
    *,
    budget_event_capacity: int = RESIDUAL_BUDGET,
) -> ResidualB2Result:
    """Consume five frozen residual predictions and apply the inherited dual guard."""
    view = feature_view(state)
    if type(budget_event_capacity) is not int:
        raise TypeError("residual budget must be an integer")
    if budget_event_capacity != RESIDUAL_BUDGET:
        raise ValueError("v4 residual is an allowed B=2 comparator only")
    predictions = tuple(seed_predictions)
    if tuple(item.seed for item in predictions) != FROZEN_SEEDS:
        raise ValueError("v4 residual predictions must be ordered seeds 0 through 4")
    seed_bases: list[dict[tuple[int, ...], float]] = []
    seed_totals: list[dict[tuple[int, ...], float]] = []
    for prediction in predictions:
        base, total = _residual_score_maps(view, prediction)
        seed_bases.append(base)
        seed_totals.append(total)
    coalitions = tuple(seed_bases[0])
    ensemble_base = {
        coalition: math.fsum(values[coalition] for values in seed_bases) / SAFE_VOTE_TOTAL
        for coalition in coalitions
    }
    ensemble_total = {
        coalition: math.fsum(values[coalition] for values in seed_totals) / SAFE_VOTE_TOTAL
        for coalition in coalitions
    }
    base_selected = _predicted_set_argmax(ensemble_base)
    pair_candidate = _predicted_set_argmax(ensemble_total)
    seed_unguarded = tuple(_predicted_set_argmax(values) for values in seed_totals)
    vote_count = sum(selected == pair_candidate for selected in seed_unguarded)
    margins = tuple(
        values[pair_candidate] - values[base_selected] for values in seed_totals
    )
    positive_margin_count = sum(value > 0.0 for value in margins)
    identical = pair_candidate == base_selected
    accepted = (
        not identical
        and vote_count >= SAFE_VOTE_THRESHOLD
        and positive_margin_count >= SAFE_VOTE_THRESHOLD
    )
    safe_selected = pair_candidate if accepted else base_selected
    return ResidualB2Result(
        frozen_base=_decision(
            view,
            selector_name="restoration_independent_gate",
            budget=RESIDUAL_BUDGET,
            selected=base_selected,
        ),
        unguarded=_decision(
            view,
            selector_name="v4_unguarded_frozen_base_residual",
            budget=RESIDUAL_BUDGET,
            selected=pair_candidate,
        ),
        safe=_decision(
            view,
            selector_name="v4_safe_frozen_base_residual",
            budget=RESIDUAL_BUDGET,
            selected=safe_selected,
        ),
        pair_candidate_vote_count=vote_count,
        strictly_positive_margin_count=positive_margin_count,
        seedwise_pair_minus_base_margins=margins,
        seedwise_unguarded_selections=seed_unguarded,
        used_pair_candidate=accepted,
        used_fallback=not identical and not accepted,
    )


def validate_frozen_formal58_ensembles(ensembles: Any) -> Any:
    """Replay both five-seed ensemble states against their frozen provenance."""
    if getattr(ensembles, "checkpoint_load_count", None) != 10:
        raise ValueError("formal58 long-horizon inference requires ten loaded checkpoints")
    for family in ("conditional", "independent"):
        ensemble = getattr(ensembles, family, None)
        provenance = getattr(ensembles, f"{family}_provenance", None)
        if ensemble is None or provenance is None:
            raise ValueError(f"formal58 {family} ensemble or provenance is missing")
        if getattr(ensemble, "family", None) != family:
            raise ValueError("formal58 ensemble family binding drifted")
        if tuple(getattr(ensemble, "seeds", ())) != FROZEN_SEEDS:
            raise ValueError("formal58 ensemble seed roster drifted")
        validate_frozen_ensemble(ensemble, provenance)
    return ensembles


def _formal58_model_config(contract: Any) -> Mapping[str, Any]:
    model = getattr(contract, "model", None)
    if isinstance(model, Mapping):
        return model
    data = getattr(contract, "data", contract)
    if not isinstance(data, Mapping):
        raise TypeError("formal58 loader contract must expose a mapping")
    learned = data.get("learned_model_artifacts")
    if isinstance(learned, Mapping):
        model = learned.get("formal58_base_and_conditional")
    else:
        model = data.get("formal_model_input")
    if not isinstance(model, Mapping):
        raise ValueError("formal58 model artifact config is missing")
    return model


def _strict_json_object(payload: bytes, label: str) -> Mapping[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {label}: {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _verified_payload(
    payloads: Mapping[str, bytes],
    record: Mapping[str, Any],
    *,
    label: str,
) -> bytes:
    path = record.get("path")
    if not isinstance(path, str) or not path or path.startswith("/") or ".." in path.split("/"):
        raise ValueError(f"{label} path is unsafe")
    try:
        payload = payloads[path]
    except KeyError as error:
        raise ValueError(f"{label} payload is missing: {path}") from error
    if (
        not isinstance(payload, bytes)
        or hashlib.sha256(payload).hexdigest() != record.get("sha256")
    ):
        raise ValueError(f"{label} payload digest drifted: {path}")
    if "size_bytes" in record and len(payload) != record["size_bytes"]:
        raise ValueError(f"{label} payload size drifted: {path}")
    return payload


def load_frozen_formal58_ensembles(
    contract: Any,
    payloads: Mapping[str, bytes],
) -> FrozenFormal58Ensembles:
    """Load the long-horizon artifact schema through frozen gate-v1 primitives."""
    from causalcache.gate_v1_formal_train import load_safetensors_checkpoint

    model_config = _formal58_model_config(contract)
    manifests = model_config.get("ensemble_manifests")
    checkpoints = model_config.get("checkpoints")
    if (
        isinstance(manifests, (str, bytes, bytearray, Mapping))
        or not isinstance(manifests, Sequence)
        or isinstance(checkpoints, (str, bytes, bytearray, Mapping))
        or not isinstance(checkpoints, Sequence)
        or len(manifests) != 2
        or len(checkpoints) != 10
    ):
        raise ValueError("formal58 artifact config must bind two manifests and ten checkpoints")
    expected_paths = {
        record.get("path")
        for record in (*tuple(manifests), *tuple(checkpoints))
        if isinstance(record, Mapping)
    }
    if len(expected_paths) != 12 or set(payloads) != expected_paths:
        raise ValueError("formal58 model payload inventory drifted")
    provenance_by_family: dict[str, FrozenEnsembleProvenance] = {}
    for raw_record in manifests:
        if not isinstance(raw_record, Mapping):
            raise ValueError("formal58 ensemble manifest record is malformed")
        payload = _verified_payload(payloads, raw_record, label="formal58 ensemble manifest")
        provenance = frozen_ensemble_provenance_from_manifest(
            _strict_json_object(payload, f"{raw_record.get('family')} ensemble manifest")
        )
        family = raw_record.get("family")
        if (
            family not in {"conditional", "independent"}
            or provenance.training.family != family
            or provenance.sha256 != raw_record.get("provenance_sha256")
            or family in provenance_by_family
        ):
            raise ValueError("formal58 ensemble manifest provenance drifted")
        provenance_by_family[family] = provenance
    if set(provenance_by_family) != {"conditional", "independent"}:
        raise ValueError("formal58 ensemble manifest family inventory drifted")
    repo = model_config.get("repo")
    models_by_family: dict[str, dict[int, Any]] = {
        "conditional": {},
        "independent": {},
    }
    for raw_record in checkpoints:
        if not isinstance(raw_record, Mapping):
            raise ValueError("formal58 checkpoint record is malformed")
        family = raw_record.get("family")
        seed = raw_record.get("seed")
        if family not in provenance_by_family or type(seed) is not int or seed not in FROZEN_SEEDS:
            raise ValueError("formal58 checkpoint family or seed drifted")
        provenance = provenance_by_family[family]
        bound = provenance.checkpoints[seed]
        payload = _verified_payload(payloads, raw_record, label="formal58 checkpoint")
        if (
            bound.seed != seed
            or bound.checkpoint_artifact.repository != repo
            or bound.checkpoint_artifact.path != raw_record.get("path")
            or bound.checkpoint_artifact.sha256 != raw_record.get("sha256")
            or bound.model_state_sha256 != raw_record.get("model_state_sha256")
            or (
                "selected_epoch" in raw_record
                and bound.selected_epoch != raw_record["selected_epoch"]
            )
            or seed in models_by_family[family]
        ):
            raise ValueError("formal58 checkpoint provenance binding drifted")
        models_by_family[family][seed] = load_safetensors_checkpoint(
            payload,
            family=family,
            seed=seed,
            expected_model_state_sha256=bound.model_state_sha256,
            expected_checkpoint_sha256=bound.checkpoint_artifact.sha256,
        )
    ensembles: dict[str, FittedEnsemble] = {}
    for family in ("conditional", "independent"):
        provenance = provenance_by_family[family]
        if set(models_by_family[family]) != set(FROZEN_SEEDS):
            raise ValueError("formal58 checkpoint seed inventory is incomplete")
        training = provenance.training
        ensembles[family] = FittedEnsemble(
            family=family,
            learning_rate=training.learning_rate,
            seeds=FROZEN_SEEDS,
            selected_epochs=training.selected_epochs,
            selection_sha256=training.oof_selection_sha256,
            models=tuple(models_by_family[family][seed] for seed in FROZEN_SEEDS),
        )
    loaded = FrozenFormal58Ensembles(
        conditional=ensembles["conditional"],
        conditional_provenance=provenance_by_family["conditional"],
        independent=ensembles["independent"],
        independent_provenance=provenance_by_family["independent"],
    )
    return validate_frozen_formal58_ensembles(loaded)


def _validate_decision_against_view(
    decision: SelectionDecision,
    view: LongHorizonFeatureView,
) -> None:
    expected = (
        view.source_id,
        view.state_id,
        view.decision_step_id,
        view.candidate_event_step_ids,
    )
    observed = (
        decision.source_id,
        decision.state_id,
        decision.decision_step_id,
        decision.candidate_event_step_ids,
    )
    if observed != expected:
        raise ValueError("label-blind selection geometry differs from its feature view")
    _decision(
        view,
        selector_name=decision.selector_name,
        budget=decision.budget_event_capacity,
        selected=decision.selected_event_step_ids,
    )
    if decision.selector_name == "v1_conditional" and decision.budget_event_capacity != 2:
        raise ValueError("conditional B=4 decisions are forbidden")
    if (
        decision.selector_name == "v4_safe_frozen_base_residual"
        and decision.budget_event_capacity != 2
    ):
        raise ValueError("v4 residual B=4 decisions are forbidden")


def summary_only_selection(
    state: Any,
    *,
    budget_event_capacity: int,
) -> SelectionDecision:
    """Represent the summary-only comparator as zero high-fidelity events."""
    view = feature_view(state)
    budget = _budget(budget_event_capacity)
    return _decision(
        view,
        selector_name="summary_only",
        budget=budget,
        selected=(),
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class LabelBlindSelectionSeal:
    """Opaque, immutable binding over canonical pre-label selection bytes."""

    __slots__ = ("_payload_bytes", "_records", "sha256", "_token")

    def __init__(
        self,
        payload_bytes: bytes,
        records: Sequence[SelectionDecision],
        *,
        _token: object,
    ) -> None:
        if _token is not _SEAL_TOKEN:
            raise TypeError("selection seals must be built by the canonical sealer")
        self._payload_bytes = bytes(payload_bytes)
        self._records = tuple(records)
        self.sha256 = hashlib.sha256(self._payload_bytes).hexdigest()
        self._token = _token

    @property
    def payload_bytes(self) -> bytes:
        return self._payload_bytes

    @property
    def records(self) -> tuple[SelectionDecision, ...]:
        return self._records

    @property
    def selections_by_selector(self) -> Mapping[str, Mapping[str, tuple[int, ...]]]:
        """Return a legacy single-budget view and reject ambiguous mixed budgets."""
        result: dict[str, dict[str, tuple[int, ...]]] = {}
        for record in self._records:
            by_state = result.setdefault(record.selector_name, {})
            if record.state_id in by_state:
                raise ValueError(
                    "mixed-budget selection seal requires selections_by_budget"
                )
            by_state[record.state_id] = record.selected_event_step_ids
        return MappingProxyType(
            {
                name: MappingProxyType(dict(values))
                for name, values in result.items()
            }
        )

    @property
    def selections_by_budget(
        self,
    ) -> Mapping[int, Mapping[str, Mapping[str, tuple[int, ...]]]]:
        result: dict[int, dict[str, dict[str, tuple[int, ...]]]] = {}
        for record in self._records:
            by_selector = result.setdefault(record.budget_event_capacity, {})
            by_state = by_selector.setdefault(record.selector_name, {})
            if record.state_id in by_state:
                raise ValueError("selection seal contains a duplicate budget/selector/state")
            by_state[record.state_id] = record.selected_event_step_ids
        return MappingProxyType(
            {
                budget: MappingProxyType(
                    {
                        name: MappingProxyType(dict(values))
                        for name, values in selectors.items()
                    }
                )
                for budget, selectors in result.items()
            }
        )


def _seal_payload(records: Sequence[SelectionDecision]) -> dict[str, Any]:
    selectors = tuple(sorted({record.selector_name for record in records}))
    state_ids = tuple(sorted({record.state_id for record in records}))
    budgets = tuple(sorted({record.budget_event_capacity for record in records}))
    selectors_by_budget = {
        str(budget): sorted(
            {
                record.selector_name
                for record in records
                if record.budget_event_capacity == budget
            }
        )
        for budget in budgets
    }
    return {
        "schema_version": SELECTION_SEAL_SCHEMA_VERSION,
        "protocol_id": SELECTION_SEAL_PROTOCOL_ID,
        "status": SELECTION_SEAL_STATUS,
        "selector_names": list(selectors),
        "budgets": list(budgets),
        "selector_names_by_budget": selectors_by_budget,
        "state_count": len(state_ids),
        "record_count": len(records),
        "records": [
            {
                "selector_name": record.selector_name,
                "source_id": record.source_id,
                "state_id": record.state_id,
                "decision_step_id": record.decision_step_id,
                "candidate_event_step_ids": list(record.candidate_event_step_ids),
                "budget_event_capacity": record.budget_event_capacity,
                "selected_event_step_ids": list(record.selected_event_step_ids),
            }
            for record in sorted(
                records,
                key=lambda item: (
                    item.budget_event_capacity,
                    item.selector_name,
                    item.state_id,
                ),
            )
        ],
    }


def seal_label_blind_selections(
    states: Sequence[Any],
    decisions: Sequence[SelectionDecision],
    *,
    expected_selector_names: Sequence[str] | None = None,
    expected_selector_names_by_budget: Mapping[int, Sequence[str]] | None = None,
) -> LabelBlindSelectionSeal:
    """Validate full selector/state coverage and emit deterministic JSON plus SHA256."""
    views = tuple(feature_view(state) for state in states)
    if not views or len({view.state_id for view in views}) != len(views):
        raise ValueError("selection seal feature-state roster is empty or duplicated")
    by_state = {view.state_id: view for view in views}
    rows = tuple(decisions)
    if not rows:
        raise ValueError("selection seal requires at least one selector decision")
    keys = tuple(
        (row.budget_event_capacity, row.selector_name, row.state_id) for row in rows
    )
    if len(set(keys)) != len(keys):
        raise ValueError("selection seal contains duplicate budget/selector/state records")
    selectors = {row.selector_name for row in rows}
    unknown = selectors - set(CANONICAL_SELECTOR_NAMES)
    if unknown:
        raise ValueError(f"selection seal contains noncanonical selector names: {sorted(unknown)}")
    observed_budgets = {row.budget_event_capacity for row in rows}
    if expected_selector_names is not None and expected_selector_names_by_budget is not None:
        raise ValueError("selection seal accepts only one expected-roster interface")
    if expected_selector_names is not None:
        expected_selectors = tuple(expected_selector_names)
        if (
            not expected_selectors
            or len(set(expected_selectors)) != len(expected_selectors)
            or any(name not in CANONICAL_SELECTOR_NAMES for name in expected_selectors)
            or len(observed_budgets) != 1
            or selectors != set(expected_selectors)
        ):
            raise ValueError("selection seal selector roster differs from the expected matrix")
    if expected_selector_names_by_budget is not None:
        if not isinstance(expected_selector_names_by_budget, Mapping):
            raise TypeError("expected budget-selector matrix must be a mapping")
        expected_matrix: dict[int, set[str]] = {}
        for budget, raw_names in expected_selector_names_by_budget.items():
            checked_budget = _budget(budget)
            if isinstance(raw_names, (str, bytes, bytearray, Mapping)):
                raise TypeError("expected selector roster must be a sequence")
            names = tuple(raw_names)
            if (
                not names
                or len(set(names)) != len(names)
                or any(name not in CANONICAL_SELECTOR_NAMES for name in names)
            ):
                raise ValueError("expected budget-selector matrix is malformed")
            expected_matrix[checked_budget] = set(names)
        observed_matrix = {
            budget: {
                row.selector_name
                for row in rows
                if row.budget_event_capacity == budget
            }
            for budget in observed_budgets
        }
        if observed_matrix != expected_matrix:
            raise ValueError("selection seal selector roster differs from the expected matrix")
    expected_states = set(by_state)
    for budget, selector in sorted(
        {(row.budget_event_capacity, row.selector_name) for row in rows}
    ):
        observed_states = {
            row.state_id
            for row in rows
            if row.budget_event_capacity == budget and row.selector_name == selector
        }
        if observed_states != expected_states:
            raise ValueError(
                "every sealed budget/selector must cover the full feature-state roster"
            )
    for row in rows:
        try:
            view = by_state[row.state_id]
        except KeyError as error:
            raise ValueError("selection decision references an unknown feature state") from error
        _validate_decision_against_view(row, view)
    payload = _canonical_json_bytes(_seal_payload(rows)) + b"\n"
    return LabelBlindSelectionSeal(payload, rows, _token=_SEAL_TOKEN)


def read_label_blind_selection_seal(
    payload: bytes,
    states: Sequence[Any],
    *,
    expected_sha256: str | None = None,
) -> LabelBlindSelectionSeal:
    """Strictly parse and byte-replay a persisted canonical selection seal."""
    if not isinstance(payload, bytes):
        raise TypeError("selection seal payload must be bytes")
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError("selection seal SHA256 differs from the expected digest")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("selection seal is not valid UTF-8 JSON") from error
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "protocol_id",
        "status",
        "selector_names",
        "budgets",
        "selector_names_by_budget",
        "state_count",
        "record_count",
        "records",
    }:
        raise ValueError("selection seal top-level schema drifted")
    if (
        value["schema_version"] != SELECTION_SEAL_SCHEMA_VERSION
        or value["protocol_id"] != SELECTION_SEAL_PROTOCOL_ID
        or value["status"] != SELECTION_SEAL_STATUS
        or not isinstance(value["records"], list)
    ):
        raise ValueError("selection seal identity or record array drifted")
    records: list[SelectionDecision] = []
    expected_record_keys = {
        "selector_name",
        "source_id",
        "state_id",
        "decision_step_id",
        "candidate_event_step_ids",
        "budget_event_capacity",
        "selected_event_step_ids",
    }
    for raw in value["records"]:
        if not isinstance(raw, Mapping) or set(raw) != expected_record_keys:
            raise ValueError("selection seal record schema drifted")
        if not isinstance(raw["candidate_event_step_ids"], list) or not isinstance(
            raw["selected_event_step_ids"], list
        ):
            raise ValueError("selection seal event arrays are malformed")
        records.append(
            SelectionDecision(
                selector_name=raw["selector_name"],
                source_id=raw["source_id"],
                state_id=raw["state_id"],
                decision_step_id=raw["decision_step_id"],
                candidate_event_step_ids=tuple(raw["candidate_event_step_ids"]),
                budget_event_capacity=raw["budget_event_capacity"],
                selected_event_step_ids=tuple(raw["selected_event_step_ids"]),
            )
        )
    seal = seal_label_blind_selections(states, records)
    if seal.payload_bytes != payload or seal.sha256 != observed_sha256:
        raise ValueError("selection seal is not in canonical byte encoding")
    if value["selector_names"] != sorted({row.selector_name for row in records}):
        raise ValueError("selection seal selector inventory drifted")
    expected_budgets = sorted({row.budget_event_capacity for row in records})
    if value["budgets"] != expected_budgets:
        raise ValueError("selection seal budget inventory drifted")
    expected_matrix = {
        str(budget): sorted(
            {
                row.selector_name
                for row in records
                if row.budget_event_capacity == budget
            }
        )
        for budget in expected_budgets
    }
    if value["selector_names_by_budget"] != expected_matrix:
        raise ValueError("selection seal budget-selector inventory drifted")
    if value["record_count"] != len(records) or value["state_count"] != len(tuple(states)):
        raise ValueError("selection seal denominators drifted")
    return seal


__all__ = [
    "CONDITIONAL_BUDGET",
    "CANONICAL_SELECTOR_NAMES",
    "ConditionalB2Result",
    "ConditionalRound",
    "LabelBlindSelectionSeal",
    "FrozenFormal58Ensembles",
    "LongHorizonFeatureView",
    "RESIDUAL_BUDGET",
    "ResidualB2Result",
    "ResidualSeedScores",
    "SAFE_VOTE_THRESHOLD",
    "SAFE_VOTE_TOTAL",
    "SUPPORTED_BUDGETS",
    "SelectionDecision",
    "conditional_b2_selection",
    "deterministic_random_selection",
    "feature_view",
    "independent_selection",
    "load_frozen_formal58_ensembles",
    "ocr_rgb_score_selection",
    "ocr_rgb_similarity_scores",
    "read_label_blind_selection_seal",
    "recent_selection",
    "residual_b2_selections",
    "seal_label_blind_selections",
    "summary_only_selection",
    "validate_frozen_formal58_ensembles",
]
