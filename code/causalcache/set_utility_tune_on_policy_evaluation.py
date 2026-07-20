"""Reduce tune-only on-policy restoration truth for contextual selectors."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from causalcache.set_utility_heldout_evaluation import (
    paired_trajectory_bootstrap,
    percentile_type7,
)
from causalcache.set_utility_heldout_inference import BUDGETS, canonical_json_bytes
from causalcache.set_utility_variable_history import history_bin


COMPLETED_LABEL = "COMPLETED_VARIABLE_HISTORY_LABEL_STATE"
SKIPPED_LABEL = "SKIPPED_VARIABLE_HISTORY_LABEL_STATE"
COMPLETED_RESULT = "COMPLETED_SET_UTILITY_TUNE_ON_POLICY_EVALUATION"
INCOMPLETE_RESULT = "INCOMPLETE_SET_UTILITY_TUNE_ON_POLICY_EVALUATION"


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{label} is outside its finite domain")
    return result


def _subset(value: Any, *, candidates: tuple[int, ...], label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be an event-id sequence")
    result = tuple(value)
    if (
        any(type(item) is not int for item in result)
        or result != tuple(sorted(result))
        or len(result) != len(set(result))
        or not set(result).issubset(candidates)
    ):
        raise ValueError(f"{label} is not a canonical candidate subset")
    return result


def _content_is_valid(payload: Mapping[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    if not isinstance(claimed, str):
        return False
    unsigned = dict(payload)
    del unsigned["content_sha256"]
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() == claimed


def _trajectory_means(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    budgets: Sequence[int],
    metric: str,
    bins: frozenset[str] | None = None,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if bins is not None and row["history_bin"] not in bins:
            continue
        for budget in budgets:
            grouped[row["trajectory_id"]].append(
                float(row["methods"][method][str(budget)][metric])
            )
    return {
        trajectory_id: fmean(values)
        for trajectory_id, values in sorted(grouped.items())
    }


def _summary(values: Mapping[str, float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "mean": fmean(values.values()),
        "trajectory_count": len(values),
    }


def evaluate_tune_on_policy(
    *,
    selections: Mapping[str, Mapping[str, Any]],
    terminals: Mapping[str, Mapping[str, Any]],
    normalization_floor: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    bootstrap_interval: float,
) -> dict[str, Any]:
    if len(selections) < 2 or "recent" in selections:
        raise ValueError("evaluation requires at least two named learned selectors")
    floor = _finite(normalization_floor, label="normalization floor")
    if floor <= 0.0:
        raise ValueError("normalization floor must be positive")

    records_by_model: dict[str, dict[str, Mapping[str, Any]]] = {}
    binding = None
    for name, payload in sorted(selections.items()):
        if (
            payload.get("status") != "COMPLETED_SET_UTILITY_TUNE_SELECTIONS"
            or not _content_is_valid(payload)
            or payload.get("variant") != name
        ):
            raise ValueError(f"selector payload is invalid: {name}")
        current_binding = (
            payload.get("cache_content_sha256"),
            payload.get("config_sha256"),
            payload.get("input_content_sha256"),
        )
        if binding is None:
            binding = current_binding
        elif binding != current_binding:
            raise ValueError("selector payload bindings differ")
        rows = {row["state_id"]: row for row in payload.get("records", ())}
        if len(rows) != len(payload.get("records", ())):
            raise ValueError("selector payload contains duplicate states")
        records_by_model[name] = rows
    inventories = {tuple(sorted(rows)) for rows in records_by_model.values()}
    if len(inventories) != 1:
        raise ValueError("selector state inventories differ")
    expected_ids = set(next(iter(inventories)))
    extra = set(terminals) - expected_ids
    if extra:
        raise ValueError("label terminals escaped the tune selector inventory")

    models = tuple(sorted(selections))
    methods = (*models, "recent")
    state_records = []
    missing = []
    skipped: dict[str, str] = {}
    for state_id in sorted(expected_ids):
        model_rows = {name: records_by_model[name][state_id] for name in models}
        reference = model_rows[models[0]]
        candidates = tuple(reference["candidate_event_ids"])
        if any(
            tuple(row["candidate_event_ids"]) != candidates
            or row["trajectory_id"] != reference["trajectory_id"]
            or row["recent"] != reference["recent"]
            for row in model_rows.values()
        ):
            raise ValueError(f"selector state identity drifted: {state_id}")
        terminal = terminals.get(state_id)
        if terminal is None:
            missing.append(state_id)
            continue
        if terminal.get("status") == SKIPPED_LABEL:
            skipped[state_id] = str(terminal.get("failure_class", "UnknownFailure"))
            continue
        if (
            terminal.get("status") != COMPLETED_LABEL
            or terminal.get("role") != "tune"
            or terminal.get("trajectory_id") != reference["trajectory_id"]
            or tuple(terminal.get("candidate_event_step_ids", ())) != candidates
        ):
            raise ValueError(f"label terminal identity drifted: {state_id}")
        distances = {}
        for row in terminal.get("distance_rows", ()):
            subset = _subset(
                row.get("coalition_event_step_ids"),
                candidates=candidates,
                label=f"{state_id} label subset",
            )
            if subset in distances:
                raise ValueError(f"duplicate label subset: {state_id} {subset}")
            distances[subset] = _finite(
                row.get("distance"),
                label=f"{state_id} distance",
                nonnegative=True,
            )
        if () not in distances or candidates not in distances or distances[candidates] != 0.0:
            raise ValueError(f"label anchors are invalid: {state_id}")
        baseline = distances[()]
        denominator = max(baseline, floor)
        method_metrics = {}
        for method in methods:
            selections_by_budget = (
                reference["recent"]
                if method == "recent"
                else model_rows[method]["learned"]
            )
            method_metrics[method] = {}
            for budget in BUDGETS:
                subset = _subset(
                    selections_by_budget[str(budget)],
                    candidates=candidates,
                    label=f"{state_id} {method} B{budget}",
                )
                if len(subset) > budget or subset not in distances:
                    raise ValueError(f"selected subset has no valid truth: {state_id} {method}")
                distance = distances[subset]
                utility = baseline - distance
                method_metrics[method][str(budget)] = {
                    "distance": distance,
                    "normalized_recovery": utility / denominator,
                    "selected_event_ids": list(subset),
                    "utility": utility,
                }
        state_records.append(
            {
                "empty_distance": baseline,
                "history_bin": history_bin(len(candidates)),
                "methods": method_metrics,
                "state_id": state_id,
                "trajectory_id": reference["trajectory_id"],
            }
        )

    summaries = {}
    for method in methods:
        by_budget = {}
        for budget in BUDGETS:
            by_budget[str(budget)] = {
                metric: _summary(
                    _trajectory_means(
                        state_records,
                        method=method,
                        budgets=(budget,),
                        metric=metric,
                    )
                )
                for metric in ("normalized_recovery", "utility")
            }
        primary = _trajectory_means(
            state_records,
            method=method,
            budgets=BUDGETS,
            metric="normalized_recovery",
        )
        long_history = _trajectory_means(
            state_records,
            method=method,
            budgets=BUDGETS,
            metric="normalized_recovery",
            bins=frozenset(("long", "very_long")),
        )
        summaries[method] = {
            "by_budget": by_budget,
            "long_plus_very_long_macro_normalized_recovery": _summary(long_history),
            "primary_macro_B1_B4_trajectory_equal_normalized_recovery": _summary(primary),
        }
        if method in selections:
            latencies: dict[str, list[float]] = defaultdict(list)
            for row in records_by_model[method].values():
                total = 0.0
                for key, value in row["latency_ms"].items():
                    measured = _finite(value, label=f"{method} latency {key}", nonnegative=True)
                    latencies[key].append(measured)
                    total += measured
                latencies["total"].append(total)
            summaries[method]["latency"] = {
                key: {
                    "p50_ms": percentile_type7(values, 0.5),
                    "p95_ms": percentile_type7(values, 0.95),
                }
                for key, values in sorted(latencies.items())
            }

    comparisons = {}
    for model in models:
        learned = _trajectory_means(
            state_records,
            method=model,
            budgets=BUDGETS,
            metric="normalized_recovery",
        )
        recent = _trajectory_means(
            state_records,
            method="recent",
            budgets=BUDGETS,
            metric="normalized_recovery",
        )
        if set(learned) != set(recent):
            raise ValueError("paired trajectory inventories differ")
        comparisons[model] = {
            "minus_recent": paired_trajectory_bootstrap(
                {key: learned[key] - recent[key] for key in learned},
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
                interval=bootstrap_interval,
            )
        }

    primary = {
        model: summaries[model][
            "primary_macro_B1_B4_trajectory_equal_normalized_recovery"
        ]["mean"]
        for model in models
    }
    best = max(primary.values())
    shortlist = tuple(model for model in models if best - primary[model] <= 0.01)
    winner = min(
        shortlist,
        key=lambda model: (summaries[model]["latency"]["total"]["p95_ms"], model),
    )
    coverage_complete = len(state_records) == len(expected_ids)
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
            "expected_state_count": len(expected_ids),
            "missing_state_ids": missing,
            "skipped_state_failures": dict(sorted(skipped.items())),
        },
        "method_summaries": summaries,
        "normalization_floor": floor,
        "schema_version": "1.0.0",
        "state_records": state_records,
        "status": COMPLETED_RESULT if coverage_complete else INCOMPLETE_RESULT,
        "winner": {
            "model": winner,
            "primary_by_model": dict(sorted(primary.items())),
            "reason": "highest_primary_or_lowest_total_p95_within_0.01",
            "shortlist_within_0.01": sorted(shortlist),
        },
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


__all__ = [
    "COMPLETED_RESULT",
    "INCOMPLETE_RESULT",
    "evaluate_tune_on_policy",
]
