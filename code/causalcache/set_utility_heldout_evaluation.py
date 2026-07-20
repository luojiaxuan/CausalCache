"""Frozen held-out reduction for variable-history set-utility selectors."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from causalcache.set_utility_variable_history import history_bin


COMPLETED_LABEL = "COMPLETED_VARIABLE_HISTORY_LABEL_STATE"
SKIPPED_LABEL = "SKIPPED_VARIABLE_HISTORY_LABEL_STATE"
SEALED_SELECTIONS = "SEALED_SET_UTILITY_HELDOUT_SELECTIONS"
COMPLETED_EVALUATION = "COMPLETED_SET_UTILITY_HELDOUT_EVALUATION"
INCOMPLETE_EVALUATION = "INCOMPLETE_SET_UTILITY_HELDOUT_EVALUATION"
HEURISTICS = ("recent", "ocr_rgb")


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return json.dumps(value, **options).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{label} is outside its finite domain")
    return result


def _subset(value: Any, *, candidates: tuple[int, ...], label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be an event-id array")
    result = tuple(value)
    if (
        any(type(item) is not int for item in result)
        or result != tuple(sorted(result))
        or len(result) != len(set(result))
        or not set(result).issubset(candidates)
    ):
        raise ValueError(f"{label} is not a canonical candidate subset")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty metric")
    return fmean(values)


def percentile_type7(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability is outside [0,1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    fraction = location - lower
    upper = ordered[min(lower + 1, len(ordered) - 1)]
    return ordered[lower] + fraction * (upper - ordered[lower])


def paired_trajectory_bootstrap(
    trajectory_deltas: Mapping[str, float],
    *,
    resamples: int,
    seed: int,
    interval: float,
) -> dict[str, Any]:
    if resamples <= 0 or not 0.0 < interval < 1.0:
        raise ValueError("bootstrap settings are invalid")
    identities = tuple(sorted(trajectory_deltas))
    if not identities:
        raise ValueError("paired bootstrap has no trajectory clusters")
    values = tuple(
        _finite(trajectory_deltas[item], label=f"trajectory delta {item}")
        for item in identities
    )
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        draws.append(
            fmean(values[rng.randrange(len(values))] for _ in range(len(values)))
        )
    tail = (1.0 - interval) / 2.0
    return {
        "cluster_count": len(values),
        "confidence": interval,
        "lower": percentile_type7(draws, tail),
        "point_estimate": fmean(values),
        "resamples": resamples,
        "seed": seed,
        "upper": percentile_type7(draws, 1.0 - tail),
    }


def load_label_terminals(label_roots: Sequence[Path]) -> dict[str, dict[str, Any]]:
    if not label_roots:
        raise ValueError("at least one label root is required")
    terminals: dict[str, dict[str, Any]] = {}
    for root in label_roots:
        states_root = root / "states"
        if not states_root.is_dir():
            raise FileNotFoundError(f"label state directory does not exist: {states_root}")
        for path in sorted(states_root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            state_id = payload.get("state_id")
            if not isinstance(state_id, str) or not state_id:
                raise ValueError(f"terminal label has no state identity: {path}")
            if state_id in terminals:
                raise ValueError(f"duplicate terminal label state: {state_id}")
            terminals[state_id] = payload
    return terminals


def _trajectory_means(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    budgets: Sequence[int],
    metric: str,
    bins: frozenset[str] | None = None,
    exact_only: bool = False,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if bins is not None and row["history_bin"] not in bins:
            continue
        if exact_only and "exact_oracle" not in row["tracks"]:
            continue
        for budget in budgets:
            grouped[row["trajectory_id"]].append(
                float(row["methods"][method][str(budget)][metric])
            )
    return {
        trajectory_id: fmean(values)
        for trajectory_id, values in sorted(grouped.items())
    }


def _trajectory_equal_summary(values: Mapping[str, float]) -> dict[str, Any]:
    return {
        "mean": _mean(tuple(values.values())),
        "trajectory_count": len(values),
    }


def _oracle_trajectory_means(
    rows: Sequence[Mapping[str, Any]], *, budget: int, metric: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if "exact_oracle" not in row["tracks"]:
            continue
        grouped[row["trajectory_id"]].append(
            float(row["oracle"][str(budget)][metric])
        )
    return {
        trajectory_id: fmean(values)
        for trajectory_id, values in sorted(grouped.items())
    }


def _exact_oracle(
    distances: Mapping[tuple[int, ...], float], *, budget: int
) -> tuple[tuple[int, ...], float]:
    eligible = tuple(
        (subset, distance)
        for subset, distance in distances.items()
        if len(subset) <= budget
    )
    if not eligible:
        raise ValueError("exact oracle domain is empty")
    return min(eligible, key=lambda item: (item[1], len(item[0]), item[0]))


def _selection_content_is_valid(payload: Mapping[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    if not isinstance(claimed, str):
        return False
    unsigned = dict(payload)
    del unsigned["content_sha256"]
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() == claimed


def evaluate_heldout(
    *,
    selections: Mapping[str, Any],
    terminals: Mapping[str, Mapping[str, Any]],
    budgets: Sequence[int],
    normalization_floor: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    bootstrap_interval: float,
    reference_config_sha256: str | None = None,
    checkpoint_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if selections.get("status") != SEALED_SELECTIONS:
        raise ValueError("selector selections are not sealed")
    if not _selection_content_is_valid(selections):
        raise ValueError("sealed selector content hash drifted")
    normalized_budgets = tuple(budgets)
    if normalized_budgets != (1, 2, 3, 4):
        raise ValueError("held-out evaluation budgets must be exactly 1,2,3,4")
    floor = _finite(normalization_floor, label="normalization floor")
    if floor <= 0.0:
        raise ValueError("normalization floor must be positive")
    selection_rows = tuple(selections.get("records", ()))
    if not selection_rows:
        raise ValueError("sealed selections contain no states")
    expected_state_ids = {row["state_id"] for row in selection_rows}
    if len(expected_state_ids) != len(selection_rows):
        raise ValueError("sealed selections contain duplicate states")
    extra_terminal_ids = set(terminals) - expected_state_ids
    if extra_terminal_ids:
        raise ValueError("terminal labels contain states outside the sealed selection")

    model_artifacts = selections.get("model_artifacts")
    if not isinstance(model_artifacts, Mapping) or not model_artifacts:
        raise ValueError("sealed selections omit model artifacts")
    models = tuple(sorted(model_artifacts))
    methods = tuple(sorted((*models, "ocr_rgb", "random", "recent")))
    state_records = []
    skipped: dict[str, str] = {}
    missing = []
    all_predictions_finite = True
    for selection in sorted(selection_rows, key=lambda item: item["state_id"]):
        state_id = selection["state_id"]
        trajectory_id = selection["trajectory_id"]
        candidates = tuple(selection["candidate_event_ids"])
        if candidates != tuple(range(1, len(candidates) + 1)):
            raise ValueError(f"candidate universe drifted for {state_id}")
        if set(selection.get("methods", ())) != set(methods):
            raise ValueError(f"method inventory drifted for {state_id}")
        metrics = selection.get("model_metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(models):
            raise ValueError(f"model metric inventory drifted for {state_id}")
        for model in models:
            predicted = metrics[model].get("predicted_utilities")
            if not isinstance(predicted, Mapping) or set(predicted) != {
                str(item) for item in normalized_budgets
            }:
                raise ValueError(f"predicted utility inventory drifted for {state_id}")
            for budget in normalized_budgets:
                try:
                    _finite(
                        predicted[str(budget)],
                        label=f"{state_id} {model} predicted utility B{budget}",
                    )
                except ValueError:
                    all_predictions_finite = False

        terminal = terminals.get(state_id)
        if terminal is None:
            missing.append(state_id)
            continue
        status = terminal.get("status")
        if status == SKIPPED_LABEL:
            skipped[state_id] = str(terminal.get("failure_class", "UnknownFailure"))
            continue
        if status != COMPLETED_LABEL:
            raise ValueError(f"unknown terminal label status for {state_id}: {status}")
        if (
            terminal.get("state_id") != state_id
            or terminal.get("trajectory_id") != trajectory_id
            or terminal.get("role") != "evaluation"
            or tuple(terminal.get("candidate_event_step_ids", ())) != candidates
        ):
            raise ValueError(f"terminal label identity drifted for {state_id}")
        if (
            reference_config_sha256 is not None
            and terminal.get("scientific_config_sha256") != reference_config_sha256
        ):
            raise ValueError(f"terminal scientific profile drifted for {state_id}")
        distances: dict[tuple[int, ...], float] = {}
        for ordinal, raw in enumerate(terminal.get("distance_rows", ())):
            subset = _subset(
                raw.get("coalition_event_step_ids"),
                candidates=candidates,
                label=f"{state_id} distance row {ordinal}",
            )
            if subset in distances:
                raise ValueError(f"duplicate distance subset for {state_id}")
            distances[subset] = _finite(
                raw.get("distance"),
                label=f"{state_id} distance {subset}",
                nonnegative=True,
            )
        if () not in distances or candidates not in distances:
            raise ValueError(f"distance anchors are missing for {state_id}")
        if distances[candidates] != 0.0:
            raise ValueError(f"full-history distance is nonzero for {state_id}")
        tracks = tuple(selection["tracks"])
        if "exact_oracle" in tracks:
            expected_exact = (
                1
                + len(candidates)
                + len(candidates) * (len(candidates) - 1) // 2
            )
            observed_exact = sum(len(subset) <= 2 for subset in distances)
            if observed_exact != expected_exact:
                raise ValueError(f"exact table is incomplete for {state_id}")
        empty_distance = distances[()]
        denominator = max(empty_distance, floor)
        method_metrics: dict[str, dict[str, dict[str, Any]]] = {}
        for method in methods:
            by_budget = selection["methods"][method]
            if set(by_budget) != {str(item) for item in normalized_budgets}:
                raise ValueError(f"selection budgets drifted for {state_id} {method}")
            method_metrics[method] = {}
            for budget in normalized_budgets:
                subset = _subset(
                    by_budget[str(budget)],
                    candidates=candidates,
                    label=f"{state_id} {method} B{budget}",
                )
                if len(subset) > budget:
                    raise ValueError(f"selection exceeds B{budget} for {state_id}")
                if subset not in distances:
                    raise ValueError(
                        "selected subset lacks restoration truth: "
                        f"{state_id} {method} B{budget}"
                    )
                distance = distances[subset]
                utility = empty_distance - distance
                method_metrics[method][str(budget)] = {
                    "distance": distance,
                    "normalized_recovery": utility / denominator,
                    "selected_event_ids": list(subset),
                    "utility": utility,
                }
        oracle_metrics = {}
        if "exact_oracle" in tracks:
            for budget in (1, 2):
                subset, distance = _exact_oracle(distances, budget=budget)
                utility = empty_distance - distance
                oracle_metrics[str(budget)] = {
                    "distance": distance,
                    "normalized_recovery": utility / denominator,
                    "selected_event_ids": list(subset),
                    "utility": utility,
                }
                for method in methods:
                    method_metrics[method][str(budget)]["oracle_regret"] = (
                        method_metrics[method][str(budget)]["distance"] - distance
                    )
        state_records.append(
            {
                "empty_distance": empty_distance,
                "history_bin": history_bin(len(candidates)),
                "methods": method_metrics,
                "oracle": oracle_metrics,
                "state_id": state_id,
                "tracks": list(tracks),
                "trajectory_id": trajectory_id,
            }
        )

    completed_ids = {row["state_id"] for row in state_records}
    coverage_complete = completed_ids == expected_state_ids
    oracle_summary = {}
    for budget in (1, 2):
        oracle_summary[str(budget)] = {}
        for metric in ("normalized_recovery", "utility"):
            values = _oracle_trajectory_means(
                state_records, budget=budget, metric=metric
            )
            oracle_summary[str(budget)][metric] = (
                _trajectory_equal_summary(values) if values else None
            )
    method_summaries: dict[str, Any] = {}
    for method in methods:
        by_budget = {}
        for budget in normalized_budgets:
            recovery = _trajectory_means(
                state_records,
                method=method,
                budgets=(budget,),
                metric="normalized_recovery",
            )
            utility = _trajectory_means(
                state_records, method=method, budgets=(budget,), metric="utility"
            )
            by_budget[str(budget)] = {
                "normalized_recovery": _trajectory_equal_summary(recovery),
                "utility": _trajectory_equal_summary(utility),
            }
        macro = _trajectory_means(
            state_records,
            method=method,
            budgets=normalized_budgets,
            metric="normalized_recovery",
        )
        bins = {}
        for bin_name in ("short", "medium", "long", "very_long"):
            values = _trajectory_means(
                state_records,
                method=method,
                budgets=normalized_budgets,
                metric="normalized_recovery",
                bins=frozenset((bin_name,)),
            )
            bins[bin_name] = (
                _trajectory_equal_summary(values) if values else None
            )
        long_values = _trajectory_means(
            state_records,
            method=method,
            budgets=normalized_budgets,
            metric="normalized_recovery",
            bins=frozenset(("long", "very_long")),
        )
        exact_regret = {}
        for budget in (1, 2):
            values = _trajectory_means(
                state_records,
                method=method,
                budgets=(budget,),
                metric="oracle_regret",
                exact_only=True,
            )
            exact_regret[str(budget)] = (
                _trajectory_equal_summary(values) if values else None
            )
        method_summaries[method] = {
            "by_budget": by_budget,
            "exact_oracle_regret": exact_regret,
            "history_bins": bins,
            "long_plus_very_long_macro_normalized_recovery": (
                _trajectory_equal_summary(long_values) if long_values else None
            ),
            "primary_macro_B1_B4_trajectory_equal_normalized_recovery": (
                _trajectory_equal_summary(macro) if macro else None
            ),
        }

    comparisons: dict[str, Any] = {}
    go_models = []
    for model in models:
        comparisons[model] = {}
        bootstrap_pass = True
        for baseline in HEURISTICS:
            left = _trajectory_means(
                state_records,
                method=model,
                budgets=normalized_budgets,
                metric="normalized_recovery",
            )
            right = _trajectory_means(
                state_records,
                method=baseline,
                budgets=normalized_budgets,
                metric="normalized_recovery",
            )
            if set(left) != set(right):
                raise ValueError("paired trajectory inventories differ")
            delta = {key: left[key] - right[key] for key in left}
            result = paired_trajectory_bootstrap(
                delta,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
                interval=bootstrap_interval,
            )
            result["lower_strictly_positive"] = result["lower"] > 0.0
            comparisons[model][f"minus_{baseline}"] = result
            bootstrap_pass = bootstrap_pass and result["lower_strictly_positive"]
        regret_pass = True
        regret_checks = {}
        for budget in (1, 2):
            model_regret = method_summaries[model]["exact_oracle_regret"][str(budget)]
            if model_regret is None:
                passed = False
            else:
                passed = all(
                    model_regret["mean"]
                    < method_summaries[baseline]["exact_oracle_regret"][
                        str(budget)
                    ]["mean"]
                    for baseline in HEURISTICS
                )
            regret_checks[str(budget)] = passed
            regret_pass = regret_pass and passed
        long_value = method_summaries[model][
            "long_plus_very_long_macro_normalized_recovery"
        ]
        heuristic_long = tuple(
            method_summaries[item]["long_plus_very_long_macro_normalized_recovery"]
            for item in HEURISTICS
        )
        long_delta = None
        long_pass = False
        if long_value is not None and all(item is not None for item in heuristic_long):
            long_delta = long_value["mean"] - max(
                item["mean"] for item in heuristic_long
            )
            long_pass = long_delta > 0.0
        checks = {
            "all_predictions_and_true_distances_finite": (
                all_predictions_finite and coverage_complete
            ),
            "bootstrap_lower_above_zero_vs_recent_and_ocr_rgb": bootstrap_pass,
            "exact_B1_B2_oracle_regret_lower_than_recent_and_ocr_rgb": regret_pass,
            "exact_regret_by_budget": regret_checks,
            "long_plus_very_long_delta_vs_best_heuristic": long_delta,
            "long_plus_very_long_point_estimate_above_zero": long_pass,
        }
        go = all(
            (
                checks["all_predictions_and_true_distances_finite"],
                bootstrap_pass,
                regret_pass,
                long_pass,
            )
        )
        comparisons[model]["requirements"] = checks
        comparisons[model]["verdict"] = "GO" if go else "NO_GO"
        if go:
            go_models.append(model)

    latency: dict[str, Any] = {}
    for model in models:
        values_by_name: dict[str, list[float]] = defaultdict(list)
        for selection in selection_rows:
            for key, value in selection["model_metrics"][model]["latency_ms"].items():
                values_by_name[key].append(
                    _finite(value, label=f"{model} latency {key}", nonnegative=True)
                )
        latency[model] = {
            key: {
                "p50_ms": percentile_type7(values, 0.5),
                "p95_ms": percentile_type7(values, 0.95),
                "state_count": len(values),
            }
            for key, values in sorted(values_by_name.items())
        }
        method_summaries[model]["latency"] = latency[model]

    winner = None
    winner_reason = "no_model_satisfied_all_frozen_requirements"
    winner_diagnostics: dict[str, Any] = {
        "go_models": sorted(go_models),
        "primary_by_model": {},
        "shortlist_within_0.01": [],
    }
    if go_models:
        primary = {
            model: method_summaries[model][
                "primary_macro_B1_B4_trajectory_equal_normalized_recovery"
            ]["mean"]
            for model in go_models
        }
        maximum = max(primary.values())
        shortlist = tuple(
            model for model in go_models if maximum - primary[model] <= 0.01
        )
        sizes = dict(checkpoint_sizes or {})
        if any(
            model not in models
            or type(size) is not int
            or size <= 0
            for model, size in sizes.items()
        ):
            raise ValueError("checkpoint size bindings are invalid")
        winner = min(
            shortlist,
            key=lambda model: (
                method_summaries[model]["latency"][
                    "selector_total_from_cached_source_tokens"
                ]["p95_ms"],
                sizes.get(model, math.inf),
                model,
            ),
        )
        winner_reason = (
            "highest_primary_outside_0.01_or_lowest_selector_p95_within_0.01"
        )
        winner_diagnostics = {
            "checkpoint_size_bytes": {
                model: sizes.get(model) for model in sorted(shortlist)
            },
            "go_models": sorted(go_models),
            "primary_by_model": dict(sorted(primary.items())),
            "selector_p95_ms": {
                model: method_summaries[model]["latency"][
                    "selector_total_from_cached_source_tokens"
                ]["p95_ms"]
                for model in sorted(shortlist)
            },
            "shortlist_within_0.01": sorted(shortlist),
        }

    result = {
        "bootstrap": {
            "confidence": bootstrap_interval,
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "unit": "trajectory",
        },
        "comparisons": comparisons,
        "coverage": {
            "complete": coverage_complete,
            "completed_state_count": len(state_records),
            "expected_state_count": len(selection_rows),
            "missing_state_ids": sorted(missing),
            "skipped_state_failures": dict(sorted(skipped.items())),
            "skipped_state_count": len(skipped),
        },
        "method_summaries": method_summaries,
        "normalization_floor": floor,
        "oracle_summary": oracle_summary,
        "schema_version": "1.0.0",
        "state_records": state_records,
        "status": COMPLETED_EVALUATION if coverage_complete else INCOMPLETE_EVALUATION,
        "winner": {
            "model": winner,
            "reason": winner_reason,
            "selection_diagnostics": winner_diagnostics,
            "verdict": "GO" if winner is not None else "NO_GO",
        },
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


__all__ = [
    "COMPLETED_EVALUATION",
    "INCOMPLETE_EVALUATION",
    "canonical_json_bytes",
    "evaluate_heldout",
    "load_label_terminals",
    "paired_trajectory_bootstrap",
    "percentile_type7",
    "sha256_file",
]
