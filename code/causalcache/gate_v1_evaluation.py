"""Frozen aggregation and GO statistics for CausalCache gate v1."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from causalcache.gate_v1_data import (
    GateState,
    select_conditional,
    select_independent,
    validate_canonical_gate_state_roster,
)
from causalcache.gate_v1_provenance import (
    GATE_V1_CONFIG_SHA256,
    ArtifactBinding,
    FrozenEnsembleProvenance,
    HeuristicArtifactProvenance,
    artifact_binding_from_manifest,
    canonical_selection_sha256,
    frozen_ensemble_provenance_from_manifest,
    heuristic_artifact_provenance_from_manifest,
    validate_frozen_ensemble,
)
from causalcache.gate_v1_training import FittedEnsemble, model_score
from causalcache.restoration_v2_2_label_table import primary_exact_subset_oracle


Score = Callable[[GateState, int, tuple[int, ...]], float]
HEURISTIC_ORDER = ("dynamic_recent", "ocr_rgb_v2", "policy_vision_v3")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 271_828
BOOTSTRAP_CONFIDENCE = 0.9
FRESH_DEVELOPMENT_IDS_SHA256 = (
    "1c37cbf6b67b0ddee61b3efe27d33b30c8fbfb471ced12f12e49624a10b82454"
)
COMBINED_DEVELOPMENT_IDS_SHA256 = (
    "ac75b14490254f59b93ebbc66e15d11201a3d241f3cc59e5c34f4cf0d3b59b0c"
)


def canonical_report_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_evaluation_roster(
    states: Sequence[GateState],
    expected_source_ids: Sequence[str],
    *,
    frozen_source_ids_sha256: str,
) -> None:
    expected_digest = hashlib.sha256(
        json.dumps(
            list(expected_source_ids),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if expected_digest != frozen_source_ids_sha256:
        raise ValueError("evaluation expected roster is not the preregistered roster")
    expected_count = (
        16
        if frozen_source_ids_sha256 == FRESH_DEVELOPMENT_IDS_SHA256
        else 21
        if frozen_source_ids_sha256 == COMBINED_DEVELOPMENT_IDS_SHA256
        else -1
    )
    if expected_count < 0:
        raise ValueError("evaluation roster digest is outside the frozen gate v1 slices")
    validate_canonical_gate_state_roster(
        states,
        expected_source_ids,
        expected_source_count=expected_count,
    )


def _ensemble_score(scores: Sequence[Score]) -> Score:
    if len(scores) != 5:
        raise ValueError("formal evaluation requires five seed scorers")

    def score(state: GateState, event: int, coalition: tuple[int, ...]) -> float:
        return math.fsum(item(state, event, coalition) for item in scores) / 5

    return score


def _trajectory_equal_mean(
    states: Sequence[GateState], values: Mapping[str, float | None]
) -> tuple[float, int, int]:
    by_source: dict[str, list[float]] = defaultdict(list)
    excluded_states = 0
    for state in states:
        value = values[state.state_id]
        if value is None:
            excluded_states += 1
        else:
            by_source[state.source_id].append(float(value))
    source_means = [statistics.fmean(items) for items in by_source.values() if items]
    if not source_means:
        raise ValueError("metric has no eligible trajectories")
    all_sources = set(state.source_id for state in states)
    return statistics.fmean(source_means), excluded_states, len(all_sources) - len(source_means)


def _trajectory_deltas(
    states: Sequence[GateState],
    left: Mapping[str, float | None],
    right: Mapping[str, float | None],
) -> dict[str, float]:
    by_source: dict[str, list[float]] = defaultdict(list)
    for state in states:
        left_value = left[state.state_id]
        right_value = right[state.state_id]
        if left_value is not None and right_value is not None:
            by_source[state.source_id].append(float(left_value) - float(right_value))
    return {
        source_id: statistics.fmean(values)
        for source_id, values in by_source.items()
        if values
    }


def type7_quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("type-7 quantile input is invalid")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("quantile values must be finite")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def paired_bootstrap_lower(
    trajectory_deltas: Mapping[str, float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> float:
    values = tuple(float(value) for value in trajectory_deltas.values())
    if not values or type(resamples) is not int or resamples <= 0:
        raise ValueError("paired bootstrap input is empty or malformed")
    generator = random.Random(seed)
    means = [
        math.fsum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(resamples)
    ]
    return type7_quantile(means, (1.0 - confidence) / 2.0)


def _selector_records(
    states: Sequence[GateState],
    conditional_scores: Sequence[Score],
    independent_scores: Sequence[Score],
    heuristics: Mapping[str, Mapping[str, Sequence[int]]],
) -> dict[str, Any]:
    if tuple(heuristics) != HEURISTIC_ORDER:
        raise ValueError("heuristic comparator inventory or order drifted")
    conditional_ensemble = _ensemble_score(conditional_scores)
    independent_ensemble = _ensemble_score(independent_scores)
    records: dict[str, Any] = {}
    for state in states:
        conditional_selected, conditional_trace = select_conditional(
            state, conditional_ensemble
        )
        independent_selected = select_independent(state, independent_ensemble)
        seed_conditional = [
            select_conditional(state, score)[0] for score in conditional_scores
        ]
        seed_independent = [
            select_independent(state, score) for score in independent_scores
        ]
        heuristic_selected = {}
        for name in HEURISTIC_ORDER:
            try:
                selected = tuple(int(item) for item in heuristics[name][state.state_id])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"heuristic {name} lacks a valid state selection") from error
            if (
                tuple(sorted(selected)) != selected
                or len(set(selected)) != len(selected)
                or len(selected) > 2
                or not set(selected).issubset(state.candidate_event_step_ids)
            ):
                raise ValueError(f"heuristic {name} selection is infeasible")
            heuristic_selected[name] = selected
        baseline = state.table.distance(())
        exact = primary_exact_subset_oracle(state.table)
        eligible = baseline > 1e-12

        def metric(selected: Sequence[int]) -> dict[str, Any]:
            utility = state.table.utility(selected)
            return {
                "selected": list(selected),
                "raw_utility": utility,
                "normalized_recovery": utility / baseline if eligible else None,
            }

        selected_prefix: tuple[int, ...] = ()
        nonpositive = 0
        for event, _prediction in conditional_trace:
            restored = tuple(sorted((*selected_prefix, event)))
            true_gain = state.table.distance(selected_prefix) - state.table.distance(restored)
            if true_gain <= 0.0:
                nonpositive += 1
            selected_prefix = restored
        records[state.state_id] = {
            "source_id": state.source_id,
            "event_count": len(state.candidate_event_step_ids),
            "baseline": baseline,
            "exact": {
                "selected": list(exact.coalition),
                "raw_utility": exact.utility,
                "normalized_recovery": exact.utility / baseline if eligible else None,
            },
            "conditional": metric(conditional_selected),
            "independent": metric(independent_selected),
            "seed_conditional": [metric(selected) for selected in seed_conditional],
            "seed_independent": [metric(selected) for selected in seed_independent],
            "heuristics": {
                name: metric(heuristic_selected[name]) for name in HEURISTIC_ORDER
            },
            "conditional_selected_addition_count": len(conditional_trace),
            "conditional_true_nonpositive_addition_count": nonpositive,
        }
    return records


def _metric_map(records: Mapping[str, Any], method: str, metric: str) -> dict[str, Any]:
    return {state_id: record[method][metric] for state_id, record in records.items()}


def _raw_ratio(
    states: Sequence[GateState], records: Mapping[str, Any], method: str
) -> float:
    learned = _metric_map(records, method, "raw_utility")
    exact = _metric_map(records, "exact", "raw_utility")
    learned_mean, _, _ = _trajectory_equal_mean(states, learned)
    exact_mean, _, _ = _trajectory_equal_mean(states, exact)
    if not math.isfinite(exact_mean) or exact_mean <= 1e-12:
        raise ValueError("exact raw-utility denominator is invalid")
    return learned_mean / exact_mean


def _seed_metric_map(records: Mapping[str, Any], family: str, seed_index: int) -> dict[str, Any]:
    return {
        state_id: record[f"seed_{family}"][seed_index]["normalized_recovery"]
        for state_id, record in records.items()
    }


def _evaluate_primary_slice_from_scores(
    states: Sequence[GateState],
    *,
    expected_source_ids: Sequence[str],
    conditional_scores: Sequence[Score],
    independent_scores: Sequence[Score],
    heuristics: Mapping[str, Mapping[str, Sequence[int]]],
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Unsealed metric core; it cannot emit a formal or test report status."""
    validate_evaluation_roster(
        states,
        expected_source_ids,
        frozen_source_ids_sha256=FRESH_DEVELOPMENT_IDS_SHA256,
    )
    if len(expected_source_ids) != 16 or len(states) != 48:
        raise ValueError("formal primary evaluation requires fresh-16 / 48 states")
    records = _selector_records(
        states, conditional_scores, independent_scores, heuristics
    )
    conditional_normalized = _metric_map(
        records, "conditional", "normalized_recovery"
    )
    independent_normalized = _metric_map(
        records, "independent", "normalized_recovery"
    )
    conditional_mean, excluded_states, excluded_trajectories = _trajectory_equal_mean(
        states, conditional_normalized
    )
    exact_mean, _, _ = _trajectory_equal_mean(
        states, _metric_map(records, "exact", "normalized_recovery")
    )
    if not math.isfinite(exact_mean) or exact_mean <= 1e-12:
        raise ValueError("exact normalized-recovery denominator is invalid")
    heuristic_means: dict[str, float] = {}
    heuristic_deltas: dict[str, float] = {}
    for name in HEURISTIC_ORDER:
        values = {
            state_id: record["heuristics"][name]["normalized_recovery"]
            for state_id, record in records.items()
        }
        heuristic_mean, _, _ = _trajectory_equal_mean(states, values)
        heuristic_means[name] = heuristic_mean
        heuristic_deltas[name] = conditional_mean - heuristic_mean
    strongest = max(
        HEURISTIC_ORDER,
        key=lambda name: (heuristic_means[name], -HEURISTIC_ORDER.index(name)),
    )
    strongest_values = {
        state_id: record["heuristics"][strongest]["normalized_recovery"]
        for state_id, record in records.items()
    }
    strongest_trajectory_deltas = _trajectory_deltas(
        states, conditional_normalized, strongest_values
    )
    set_trajectory_deltas = _trajectory_deltas(
        states, conditional_normalized, independent_normalized
    )
    set_mean = statistics.fmean(set_trajectory_deltas.values())

    seed_normalized_means = []
    seed_exact_ratios = []
    seed_raw_ratios = []
    paired_seed_deltas = []
    for seed_index in range(5):
        conditional_seed = _seed_metric_map(records, "conditional", seed_index)
        independent_seed = _seed_metric_map(records, "independent", seed_index)
        seed_normalized_mean, _, _ = _trajectory_equal_mean(
            states, conditional_seed
        )
        seed_normalized_means.append(seed_normalized_mean)
        seed_exact_ratios.append(seed_normalized_mean / exact_mean)
        seed_raw = {
            state_id: record["seed_conditional"][seed_index]["raw_utility"]
            for state_id, record in records.items()
        }
        learned_mean, _, _ = _trajectory_equal_mean(states, seed_raw)
        exact_raw_mean, _, _ = _trajectory_equal_mean(
            states, _metric_map(records, "exact", "raw_utility")
        )
        seed_raw_ratios.append(learned_mean / exact_raw_mean)
        paired_seed_deltas.append(
            statistics.fmean(
                _trajectory_deltas(states, conditional_seed, independent_seed).values()
            )
        )

    hard_states = [state for state in states if len(state.candidate_event_step_ids) in {3, 4}]
    hard_learned, _, _ = _trajectory_equal_mean(
        hard_states, _metric_map(records, "conditional", "raw_utility")
    )
    hard_exact, _, _ = _trajectory_equal_mean(
        hard_states, _metric_map(records, "exact", "raw_utility")
    )
    if hard_exact <= 1e-12:
        raise ValueError("hard-slice exact raw utility is invalid")
    selected_count = sum(
        record["conditional_selected_addition_count"] for record in records.values()
    )
    nonpositive_count = sum(
        record["conditional_true_nonpositive_addition_count"]
        for record in records.values()
    )
    nonpositive_rate = nonpositive_count / selected_count if selected_count else 0.0
    conditional_raw_mean, _, _ = _trajectory_equal_mean(
        states, _metric_map(records, "conditional", "raw_utility")
    )
    independent_raw_mean, _, _ = _trajectory_equal_mean(
        states, _metric_map(records, "independent", "raw_utility")
    )
    metrics = {
        "ensemble_normalized_recovery": conditional_mean,
        "exact_normalized_recovery": exact_mean,
        "ensemble_raw_utility_over_exact_raw": _raw_ratio(
            states, records, "conditional"
        ),
        "normalized_excluded_state_count": excluded_states,
        "normalized_excluded_trajectory_count": excluded_trajectories,
        "raw_retained_small_Dempty_state_count": excluded_states,
        "heuristic_mean_normalized_recovery": heuristic_means,
        "conditional_minus_heuristic_mean_normalized_delta": heuristic_deltas,
        "strongest_heuristic": strongest,
        "strongest_heuristic_positive_trajectory_count": sum(
            value > 0.0 for value in strongest_trajectory_deltas.values()
        ),
        "strongest_heuristic_paired_bootstrap_lower": paired_bootstrap_lower(
            strongest_trajectory_deltas, resamples=bootstrap_resamples
        ),
        "individual_seed_mean_normalized_recovery": seed_normalized_means,
        "individual_seed_exact_normalized_ratios": seed_exact_ratios,
        "individual_seed_exact_raw_ratios": seed_raw_ratios,
        "individual_seed_ratio_at_least_0_75_count": sum(
            value >= 0.75 for value in seed_exact_ratios
        ),
        "seed_mean_recovery_population_std": statistics.pstdev(
            seed_normalized_means
        ),
        "hard_n3_n4_raw_utility_ratio": hard_learned / hard_exact,
        "selected_true_nonpositive_addition_rate": nonpositive_rate,
        "conditional_minus_independent_mean_normalized_delta": set_mean,
        "conditional_minus_independent_raw_utility_delta": (
            conditional_raw_mean - independent_raw_mean
        ),
        "conditional_minus_independent_positive_trajectory_count": sum(
            value > 0.0 for value in set_trajectory_deltas.values()
        ),
        "conditional_minus_independent_paired_bootstrap_lower": paired_bootstrap_lower(
            set_trajectory_deltas, resamples=bootstrap_resamples
        ),
        "paired_seed_normalized_deltas": paired_seed_deltas,
        "paired_seed_positive_count": sum(value > 0.0 for value in paired_seed_deltas),
    }
    selector_checks = {
        "ensemble_normalized_recovery_over_exact": conditional_mean / exact_mean >= 0.8,
        "ensemble_raw_utility_over_exact_raw": metrics[
            "ensemble_raw_utility_over_exact_raw"
        ]
        >= 0.8,
        "delta_vs_every_heuristic": all(value >= 0.05 for value in heuristic_deltas.values()),
        "strongest_positive_trajectory_count": metrics[
            "strongest_heuristic_positive_trajectory_count"
        ]
        >= 12,
        "strongest_bootstrap_lower": metrics[
            "strongest_heuristic_paired_bootstrap_lower"
        ]
        > 0.0,
        "individual_seed_pass_count": metrics[
            "individual_seed_ratio_at_least_0_75_count"
        ]
        >= 4,
        "seed_population_std": metrics["seed_mean_recovery_population_std"] <= 0.08,
        "hard_n3_n4_ratio": metrics["hard_n3_n4_raw_utility_ratio"] >= 0.7,
        "nonpositive_addition_rate": metrics[
            "selected_true_nonpositive_addition_rate"
        ]
        <= 0.1,
    }
    set_primary_checks = {
        "mean_normalized_delta": metrics[
            "conditional_minus_independent_mean_normalized_delta"
        ]
        >= 0.02,
        "raw_utility_delta": metrics[
            "conditional_minus_independent_raw_utility_delta"
        ]
        > 0.0,
        "positive_trajectory_count": metrics[
            "conditional_minus_independent_positive_trajectory_count"
        ]
        >= 12,
        "bootstrap_lower": metrics[
            "conditional_minus_independent_paired_bootstrap_lower"
        ]
        > 0.0,
        "paired_seed_positive_count": metrics["paired_seed_positive_count"] >= 4,
    }
    report = {
        "source_count": len(expected_source_ids),
        "state_count": len(states),
        "bootstrap": {
            "unit": "trajectory",
            "resamples": bootstrap_resamples,
            "seed": BOOTSTRAP_SEED,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "interval": "percentile",
            "quantile": "Hyndman_Fan_type_7",
        },
        "metrics": metrics,
        "go_selector_checks": selector_checks,
        "go_selector": all(selector_checks.values()),
        "go_set_conditioning_primary_checks": set_primary_checks,
        "go_set_conditioning_primary": all(set_primary_checks.values()),
        "combined21_compatibility_evaluated": False,
        "confirm_access_authorized": False,
        "matched_nll_authorized": False,
        "closed_loop_authorized": False,
    }
    return report


def _state_inventory_sha256(states: Sequence[GateState]) -> str:
    payload = [
        {
            "source_id": state.source_id,
            "state_id": state.state_id,
            "decision_step_id": state.decision_step_id,
            "candidate_event_step_ids": list(state.candidate_event_step_ids),
        }
        for state in states
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_formal_ensemble_pair(
    conditional_ensemble: FittedEnsemble,
    conditional_provenance: FrozenEnsembleProvenance,
    independent_ensemble: FittedEnsemble,
    independent_provenance: FrozenEnsembleProvenance,
) -> tuple[str, str]:
    if not isinstance(conditional_ensemble, FittedEnsemble) or not isinstance(
        independent_ensemble, FittedEnsemble
    ):
        raise TypeError("formal evaluation requires fitted gate ensembles, not callables")
    if (
        conditional_ensemble.family != "conditional"
        or independent_ensemble.family != "independent"
        or conditional_provenance.training.family != "conditional"
        or independent_provenance.training.family != "independent"
    ):
        raise ValueError("formal ensemble families are reversed or malformed")
    conditional_digest = validate_frozen_ensemble(
        conditional_ensemble, conditional_provenance
    )
    independent_digest = validate_frozen_ensemble(
        independent_ensemble, independent_provenance
    )
    if (
        conditional_provenance.training.feature_artifact
        != independent_provenance.training.feature_artifact
        or conditional_provenance.training.label_artifact
        != independent_provenance.training.label_artifact
    ):
        raise ValueError("conditional and independent models were not fit on identical caches")
    all_models = (*conditional_ensemble.models, *independent_ensemble.models)
    if len({id(model) for model in all_models}) != 10:
        raise ValueError("formal conditional/independent ensembles cannot share model objects")
    all_states = tuple(
        checkpoint.model_state_sha256
        for provenance in (conditional_provenance, independent_provenance)
        for checkpoint in provenance.checkpoints
    )
    if len(set(all_states)) != 10:
        raise ValueError("formal conditional/independent checkpoints must all be distinct")
    return conditional_digest, independent_digest


def _formal_primary_provenance_payload(
    states: Sequence[GateState],
    *,
    conditional_ensemble: FittedEnsemble,
    conditional_provenance: FrozenEnsembleProvenance,
    independent_ensemble: FittedEnsemble,
    independent_provenance: FrozenEnsembleProvenance,
    heuristics: Mapping[str, Mapping[str, Sequence[int]]],
    evaluation_feature_artifact: ArtifactBinding,
    evaluation_label_artifact: ArtifactBinding,
    heuristic_provenance: Sequence[HeuristicArtifactProvenance],
) -> dict[str, Any]:
    conditional_digest, independent_digest = _validate_formal_ensemble_pair(
        conditional_ensemble,
        conditional_provenance,
        independent_ensemble,
        independent_provenance,
    )
    evaluation_feature_artifact.validate(label="fresh-16 feature artifact")
    evaluation_label_artifact.validate(label="fresh-16 label artifact")
    if evaluation_feature_artifact == evaluation_label_artifact:
        raise ValueError("evaluation feature and label caches must be physically separated")
    bindings = tuple(heuristic_provenance)
    if tuple(binding.name for binding in bindings) != HEURISTIC_ORDER:
        raise ValueError("heuristic provenance inventory or order drifted")
    expected_state_ids = {state.state_id for state in states}
    for binding in bindings:
        binding.validate()
        selections = heuristics.get(binding.name)
        if not isinstance(selections, Mapping) or set(selections) != expected_state_ids:
            raise ValueError(f"{binding.name} heuristic state inventory drifted")
        if canonical_selection_sha256(selections) != binding.selection_sha256:
            raise ValueError(f"{binding.name} selections differ from the frozen artifact")
    return {
        "gate_config_sha256": GATE_V1_CONFIG_SHA256,
        "state_inventory_sha256": _state_inventory_sha256(states),
        "conditional_ensemble_sha256": conditional_digest,
        "conditional_ensemble": conditional_provenance.to_payload(),
        "independent_ensemble_sha256": independent_digest,
        "independent_ensemble": independent_provenance.to_payload(),
        "evaluation_feature_artifact": evaluation_feature_artifact.to_payload(),
        "evaluation_label_artifact": evaluation_label_artifact.to_payload(),
        "heuristic_artifacts": [binding.to_payload() for binding in bindings],
    }


def evaluate_primary_slice(
    states: Sequence[GateState],
    *,
    expected_source_ids: Sequence[str],
    conditional_ensemble: FittedEnsemble,
    conditional_provenance: FrozenEnsembleProvenance,
    independent_ensemble: FittedEnsemble,
    independent_provenance: FrozenEnsembleProvenance,
    heuristics: Mapping[str, Mapping[str, Sequence[int]]],
    evaluation_feature_artifact: ArtifactBinding,
    evaluation_label_artifact: ArtifactBinding,
    heuristic_provenance: Sequence[HeuristicArtifactProvenance],
) -> dict[str, Any]:
    """Run the immutable fresh-16 evaluation with model and artifact provenance."""
    validate_evaluation_roster(
        states,
        expected_source_ids,
        frozen_source_ids_sha256=FRESH_DEVELOPMENT_IDS_SHA256,
    )
    provenance_payload = _formal_primary_provenance_payload(
        states,
        conditional_ensemble=conditional_ensemble,
        conditional_provenance=conditional_provenance,
        independent_ensemble=independent_ensemble,
        independent_provenance=independent_provenance,
        heuristics=heuristics,
        evaluation_feature_artifact=evaluation_feature_artifact,
        evaluation_label_artifact=evaluation_label_artifact,
        heuristic_provenance=heuristic_provenance,
    )
    conditional_scores = tuple(
        model_score(model, "conditional") for model in conditional_ensemble.models
    )
    independent_scores = tuple(
        model_score(model, "independent") for model in independent_ensemble.models
    )
    report = _evaluate_primary_slice_from_scores(
        states,
        expected_source_ids=expected_source_ids,
        conditional_scores=conditional_scores,
        independent_scores=independent_scores,
        heuristics=heuristics,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    sealed = {
        "status": "FROZEN_GATE_V1_FRESH16_PRIMARY_EVALUATION",
        **report,
        "provenance": provenance_payload,
    }
    sealed["report_sha256"] = canonical_report_sha256(sealed)
    return sealed


def _evaluate_primary_slice_test_only(
    states: Sequence[GateState],
    *,
    expected_source_ids: Sequence[str],
    conditional_scores: Sequence[Score],
    independent_scores: Sequence[Score],
    heuristics: Mapping[str, Mapping[str, Sequence[int]]],
    bootstrap_resamples: int = 100,
) -> dict[str, Any]:
    report = _evaluate_primary_slice_from_scores(
        states,
        expected_source_ids=expected_source_ids,
        conditional_scores=conditional_scores,
        independent_scores=independent_scores,
        heuristics=heuristics,
        bootstrap_resamples=bootstrap_resamples,
    )
    sealed = {
        "status": "TEST_ONLY_GATE_V1_FRESH16_PRIMARY_EVALUATION",
        **report,
    }
    sealed["report_sha256"] = canonical_report_sha256(sealed)
    return sealed


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_primary_provenance_payload(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("primary report lacks provenance")
    expected_keys = {
        "gate_config_sha256",
        "state_inventory_sha256",
        "conditional_ensemble_sha256",
        "conditional_ensemble",
        "independent_ensemble_sha256",
        "independent_ensemble",
        "evaluation_feature_artifact",
        "evaluation_label_artifact",
        "heuristic_artifacts",
    }
    if set(value) != expected_keys or value.get("gate_config_sha256") != GATE_V1_CONFIG_SHA256:
        raise ValueError("primary provenance schema or gate config drifted")
    if not _is_sha256(value.get("state_inventory_sha256")):
        raise ValueError("primary state inventory digest is invalid")
    conditional = frozen_ensemble_provenance_from_manifest(
        value["conditional_ensemble"]
    )
    independent = frozen_ensemble_provenance_from_manifest(
        value["independent_ensemble"]
    )
    if (
        conditional.training.family != "conditional"
        or independent.training.family != "independent"
        or value["conditional_ensemble_sha256"] != conditional.sha256
        or value["independent_ensemble_sha256"] != independent.sha256
        or conditional.training.feature_artifact
        != independent.training.feature_artifact
        or conditional.training.label_artifact != independent.training.label_artifact
    ):
        raise ValueError("primary ensemble provenance does not replay")
    feature = artifact_binding_from_manifest(
        value["evaluation_feature_artifact"], label="primary feature artifact"
    )
    label = artifact_binding_from_manifest(
        value["evaluation_label_artifact"], label="primary label artifact"
    )
    if feature == label:
        raise ValueError("primary feature and label artifacts are not separated")
    heuristic_values = value["heuristic_artifacts"]
    if not isinstance(heuristic_values, list):
        raise ValueError("primary heuristic provenance must be an array")
    heuristics = tuple(
        heuristic_artifact_provenance_from_manifest(item)
        for item in heuristic_values
    )
    if tuple(item.name for item in heuristics) != HEURISTIC_ORDER:
        raise ValueError("primary heuristic provenance order drifted")
    return value


def _validate_combined_provenance_payload(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("combined report lacks provenance")
    if set(value) != {
        "gate_config_sha256",
        "state_inventory_sha256",
        "conditional_ensemble_sha256",
        "independent_ensemble_sha256",
        "evaluation_feature_artifact",
        "evaluation_label_artifact",
    } or value.get("gate_config_sha256") != GATE_V1_CONFIG_SHA256:
        raise ValueError("combined provenance schema or gate config drifted")
    if not all(
        _is_sha256(value.get(name))
        for name in (
            "state_inventory_sha256",
            "conditional_ensemble_sha256",
            "independent_ensemble_sha256",
        )
    ):
        raise ValueError("combined provenance contains an invalid digest")
    feature = artifact_binding_from_manifest(
        value["evaluation_feature_artifact"], label="combined feature artifact"
    )
    label = artifact_binding_from_manifest(
        value["evaluation_label_artifact"], label="combined label artifact"
    )
    if feature == label:
        raise ValueError("combined feature and label artifacts are not separated")
    return value


def _validate_frozen_primary_report(
    primary_report: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    unbound = dict(primary_report)
    observed_digest = unbound.pop("report_sha256", None)
    primary_provenance = primary_report.get("provenance")
    expected_bootstrap = {
        "unit": "trajectory",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "interval": "percentile",
        "quantile": "Hyndman_Fan_type_7",
    }
    if (
        primary_report.get("status") != "FROZEN_GATE_V1_FRESH16_PRIMARY_EVALUATION"
        or observed_digest != canonical_report_sha256(unbound)
        or primary_report.get("combined21_compatibility_evaluated") is not False
        or primary_report.get("bootstrap") != expected_bootstrap
    ):
        raise ValueError("primary fresh-16 report is not a frozen valid input")
    return observed_digest, _validate_primary_provenance_payload(primary_provenance)


def _finalize_combined_compatibility(
    primary_report: Mapping[str, Any],
    *,
    combined21_conditional_minus_independent_mean_normalized_delta: float,
    combined_provenance_payload: Mapping[str, Any],
    excluded_state_count: int,
    excluded_trajectory_count: int,
) -> dict[str, Any]:
    """Apply the post-primary compatibility guard without authorizing confirm."""
    observed_digest, primary_provenance = _validate_frozen_primary_report(
        primary_report
    )
    combined_provenance = _validate_combined_provenance_payload(
        combined_provenance_payload
    )
    if (
        combined_provenance.get("conditional_ensemble_sha256")
        != primary_provenance.get("conditional_ensemble_sha256")
        or combined_provenance.get("independent_ensemble_sha256")
        != primary_provenance.get("independent_ensemble_sha256")
    ):
        raise ValueError("combined-21 evaluation did not reuse the frozen primary ensembles")
    if (
        type(excluded_state_count) is not int
        or excluded_state_count < 0
        or type(excluded_trajectory_count) is not int
        or excluded_trajectory_count < 0
    ):
        raise ValueError("combined-21 excluded counts are invalid")
    value = float(combined21_conditional_minus_independent_mean_normalized_delta)
    if not math.isfinite(value):
        raise ValueError("combined-21 compatibility delta must be finite")
    set_go = bool(primary_report["go_set_conditioning_primary"] and value >= 0.0)
    result = {
        "status": "FINAL_GATE_V1_GO_DECISION",
        "primary_report_sha256": observed_digest,
        "combined_provenance": dict(combined_provenance),
        "combined21_conditional_minus_independent_mean_normalized_delta": value,
        "combined21_normalized_excluded_state_count": excluded_state_count,
        "combined21_normalized_excluded_trajectory_count": excluded_trajectory_count,
        "combined21_guard_pass": value >= 0.0,
        "go_selector": bool(primary_report["go_selector"]),
        "go_set_conditioning": set_go,
        "go_to_post_go_contract": bool(primary_report["go_selector"] and set_go),
        "confirm_access_authorized": False,
        "matched_nll_authorized": False,
        "closed_loop_authorized": False,
    }
    result["report_sha256"] = canonical_report_sha256(result)
    return result


def evaluate_combined21_compatibility(
    primary_report: Mapping[str, Any],
    states: Sequence[GateState],
    *,
    expected_source_ids: Sequence[str],
    conditional_ensemble: FittedEnsemble,
    conditional_provenance: FrozenEnsembleProvenance,
    independent_ensemble: FittedEnsemble,
    independent_provenance: FrozenEnsembleProvenance,
    evaluation_feature_artifact: ArtifactBinding,
    evaluation_label_artifact: ArtifactBinding,
) -> dict[str, Any]:
    """Read/evaluate combined-21 only after a frozen fresh-16 report exists."""
    _, primary_provenance = _validate_frozen_primary_report(primary_report)
    validate_evaluation_roster(
        states,
        expected_source_ids,
        frozen_source_ids_sha256=COMBINED_DEVELOPMENT_IDS_SHA256,
    )
    if len(expected_source_ids) != 21 or len(states) != 63:
        raise ValueError("combined compatibility requires 21 trajectories / 63 states")
    conditional_digest, independent_digest = _validate_formal_ensemble_pair(
        conditional_ensemble,
        conditional_provenance,
        independent_ensemble,
        independent_provenance,
    )
    evaluation_feature_artifact.validate(label="combined-21 feature artifact")
    evaluation_label_artifact.validate(label="combined-21 label artifact")
    if evaluation_feature_artifact == evaluation_label_artifact:
        raise ValueError("combined feature and label caches must be physically separated")
    combined_provenance = {
        "gate_config_sha256": GATE_V1_CONFIG_SHA256,
        "state_inventory_sha256": _state_inventory_sha256(states),
        "conditional_ensemble_sha256": conditional_digest,
        "independent_ensemble_sha256": independent_digest,
        "evaluation_feature_artifact": evaluation_feature_artifact.to_payload(),
        "evaluation_label_artifact": evaluation_label_artifact.to_payload(),
    }
    if (
        conditional_digest != primary_provenance.get("conditional_ensemble_sha256")
        or independent_digest != primary_provenance.get("independent_ensemble_sha256")
    ):
        raise ValueError("combined-21 must reuse the exact frozen primary ensembles")
    conditional_scores = tuple(
        model_score(model, "conditional") for model in conditional_ensemble.models
    )
    independent_scores = tuple(
        model_score(model, "independent") for model in independent_ensemble.models
    )
    empty_heuristics = {
        name: {state.state_id: () for state in states} for name in HEURISTIC_ORDER
    }
    records = _selector_records(
        states, conditional_scores, independent_scores, empty_heuristics
    )
    conditional = _metric_map(records, "conditional", "normalized_recovery")
    independent = _metric_map(records, "independent", "normalized_recovery")
    trajectory_deltas = _trajectory_deltas(states, conditional, independent)
    delta = statistics.fmean(trajectory_deltas.values())
    excluded_state_count = sum(
        conditional[state.state_id] is None or independent[state.state_id] is None
        for state in states
    )
    excluded_trajectory_count = len(set(state.source_id for state in states)) - len(
        trajectory_deltas
    )
    return _finalize_combined_compatibility(
        primary_report,
        combined21_conditional_minus_independent_mean_normalized_delta=delta,
        combined_provenance_payload=combined_provenance,
        excluded_state_count=excluded_state_count,
        excluded_trajectory_count=excluded_trajectory_count,
    )


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "COMBINED_DEVELOPMENT_IDS_SHA256",
    "FRESH_DEVELOPMENT_IDS_SHA256",
    "HEURISTIC_ORDER",
    "canonical_report_sha256",
    "evaluate_combined21_compatibility",
    "evaluate_primary_slice",
    "paired_bootstrap_lower",
    "type7_quantile",
    "validate_evaluation_roster",
]
