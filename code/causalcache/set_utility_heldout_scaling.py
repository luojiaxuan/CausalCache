"""Post-hoc exact-track scaling diagnostics for frozen utility predictors."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from causalcache.set_utility_heldout_evaluation import (
    COMPLETED_LABEL,
    SKIPPED_LABEL,
    paired_trajectory_bootstrap,
)
from causalcache.set_utility_heldout_inference import canonical_json_bytes


COMPLETED_SCALING_DIAGNOSTIC = (
    "COMPLETED_SET_UTILITY_HELDOUT_SCALING_DIAGNOSTIC"
)
COMPLETED_SCALING_DIAGNOSTIC_WITH_SKIPS = (
    "COMPLETED_SET_UTILITY_HELDOUT_SCALING_DIAGNOSTIC_WITH_SKIPS"
)


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{label} is outside its finite domain")
    return result


def _subset(
    value: Any, *, candidates: tuple[int, ...], budget: int, label: str
) -> tuple[int, ...]:
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
        or len(result) > budget
    ):
        raise ValueError(f"{label} is not a valid at-most-B subset")
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
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for budget in budgets:
            grouped[row["trajectory_id"]].append(
                float(row["methods"][method][str(budget)][metric])
            )
    return {
        trajectory_id: fmean(values)
        for trajectory_id, values in sorted(grouped.items())
    }


def _summary(values: Mapping[str, float]) -> dict[str, Any]:
    if not values:
        raise ValueError("trajectory-equal summary has no values")
    return {"mean": fmean(values.values()), "trajectory_count": len(values)}


def evaluate_scaling_exact_track(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    selections: Mapping[str, Mapping[str, Any]],
    terminals: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidates_config = config["frozen_candidates"]["models"]
    if set(selections) != set(candidates_config):
        raise ValueError("scaling selection candidate inventory drifted")
    budgets = tuple(config["diagnostic_evaluation"]["budgets"])
    if budgets != (1, 2):
        raise ValueError("scaling diagnostic must use exact B1/B2 truth")
    floor = _finite(
        config["representation"]["normalization_floor"],
        label="normalization floor",
    )
    if floor <= 0.0:
        raise ValueError("normalization floor must be positive")

    records_by_model: dict[str, dict[str, Mapping[str, Any]]] = {}
    reference_state_ids: tuple[str, ...] | None = None
    for model_name, payload in sorted(selections.items()):
        candidate = candidates_config[model_name]
        if (
            payload.get("status")
            != "COMPLETED_SET_UTILITY_HELDOUT_MODEL_SELECTIONS"
            or payload.get("model_name") != model_name
            or payload.get("checkpoint_sha256") != candidate["sha256"]
            or payload.get("config_sha256") != config_sha256
            or not _content_is_valid(payload)
        ):
            raise ValueError(f"invalid scaling selection payload: {model_name}")
        rows = {row["state_id"]: row for row in payload.get("records", ())}
        if len(rows) != len(payload.get("records", ())):
            raise ValueError(f"duplicate scaling states: {model_name}")
        state_ids = tuple(sorted(rows))
        if reference_state_ids is None:
            reference_state_ids = state_ids
        elif state_ids != reference_state_ids:
            raise ValueError("scaling selection state inventories differ")
        records_by_model[model_name] = rows
    if reference_state_ids is None:
        raise ValueError("scaling diagnostic has no selection payloads")
    if len(reference_state_ids) != config["state_inventory"]["union_state_count"]:
        raise ValueError("scaling selection union count drifted")

    first_model = sorted(records_by_model)[0]
    exact_state_ids = tuple(
        state_id
        for state_id in reference_state_ids
        if "exact_oracle" in records_by_model[first_model][state_id]["tracks"]
    )
    if len(exact_state_ids) != config["state_inventory"]["exact_state_count"]:
        raise ValueError("scaling exact-track count drifted")

    state_records = []
    skipped: dict[str, str] = {}
    for state_id in exact_state_ids:
        reference = records_by_model[first_model][state_id]
        terminal = terminals.get(state_id)
        if terminal is None:
            raise ValueError(f"exact-track terminal is missing: {state_id}")
        if terminal.get("status") == SKIPPED_LABEL:
            skipped[state_id] = str(terminal.get("failure_class", "UnknownFailure"))
            continue
        if terminal.get("status") != COMPLETED_LABEL:
            raise ValueError(f"invalid exact-track terminal status: {state_id}")
        candidate_events = tuple(reference["candidate_event_ids"])
        if (
            terminal.get("state_id") != state_id
            or terminal.get("trajectory_id") != reference["trajectory_id"]
            or tuple(terminal.get("candidate_event_step_ids", ()))
            != candidate_events
        ):
            raise ValueError(f"scaling terminal identity drifted: {state_id}")
        distances: dict[tuple[int, ...], float] = {}
        for raw in terminal.get("distance_rows", ()):
            subset = tuple(raw["coalition_event_step_ids"])
            if subset in distances:
                raise ValueError(f"duplicate scaling distance: {state_id}")
            distances[subset] = _finite(
                raw["distance"], label=f"{state_id} distance", nonnegative=True
            )
        expected = 1 + len(candidate_events) + (
            len(candidate_events) * (len(candidate_events) - 1) // 2
        )
        if sum(len(subset) <= 2 for subset in distances) != expected:
            raise ValueError(f"exact B1/B2 truth is incomplete: {state_id}")
        empty_distance = distances[()]
        denominator = max(empty_distance, floor)
        methods: dict[str, dict[str, Any]] = {}
        model_names = tuple(sorted(records_by_model))
        for model_name in (*model_names, "recent", "ocr_rgb"):
            source = (
                records_by_model[model_name][state_id]
                if model_name in records_by_model
                else reference
            )
            if model_name in records_by_model:
                selected_by_budget = source["methods"][model_name]
            else:
                selected_by_budget = source["methods"][model_name]
                for other in model_names[1:]:
                    if (
                        records_by_model[other][state_id]["methods"][model_name]
                        != selected_by_budget
                    ):
                        raise ValueError("scaling baseline selections drifted")
            methods[model_name] = {}
            for budget in budgets:
                subset = _subset(
                    selected_by_budget[str(budget)],
                    candidates=candidate_events,
                    budget=budget,
                    label=f"{state_id} {model_name} B{budget}",
                )
                if subset not in distances:
                    raise ValueError(f"selected exact subset lacks truth: {state_id}")
                methods[model_name][str(budget)] = {
                    "distance": distances[subset],
                    "normalized_recovery": (
                        empty_distance - distances[subset]
                    )
                    / denominator,
                    "selected_cardinality": len(subset),
                    "utility": empty_distance - distances[subset],
                }
        oracle = {}
        for budget in budgets:
            subset, distance = min(
                (
                    (subset, distance)
                    for subset, distance in distances.items()
                    if len(subset) <= budget
                ),
                key=lambda item: (item[1], len(item[0]), item[0]),
            )
            oracle[str(budget)] = {
                "distance": distance,
                "normalized_recovery": (empty_distance - distance) / denominator,
                "selected_cardinality": len(subset),
                "utility": empty_distance - distance,
            }
            for model_name in methods:
                methods[model_name][str(budget)]["oracle_regret"] = (
                    methods[model_name][str(budget)]["distance"] - distance
                )
        state_records.append(
            {
                "methods": methods,
                "oracle": oracle,
                "state_id": state_id,
                "trajectory_id": reference["trajectory_id"],
            }
        )

    methods = (*sorted(records_by_model), "recent", "ocr_rgb")
    summaries = {}
    for method in methods:
        by_budget = {}
        for budget in budgets:
            by_budget[str(budget)] = {
                metric: _summary(
                    _trajectory_means(
                        state_records,
                        method=method,
                        budgets=(budget,),
                        metric=metric,
                    )
                )
                for metric in (
                    "normalized_recovery",
                    "oracle_regret",
                    "selected_cardinality",
                    "utility",
                )
            }
        macro = _trajectory_means(
            state_records,
            method=method,
            budgets=budgets,
            metric="normalized_recovery",
        )
        summaries[method] = {
            "by_budget": by_budget,
            "macro_B1_B2_normalized_recovery": _summary(macro),
        }
        if method in candidates_config:
            summaries[method].update(
                {
                    "best_tune_total": candidates_config[method]["best_tune_total"],
                    "family": candidates_config[method]["family"],
                    "fraction": candidates_config[method]["fraction"],
                    "train_state_count": candidates_config[method][
                        "train_state_count"
                    ],
                    "train_trajectory_count": candidates_config[method][
                        "train_trajectory_count"
                    ],
                }
            )

    oracle_summary = {}
    for budget in budgets:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in state_records:
            grouped[row["trajectory_id"]].append(
                row["oracle"][str(budget)]["normalized_recovery"]
            )
        oracle_summary[str(budget)] = _summary(
            {name: fmean(values) for name, values in sorted(grouped.items())}
        )

    bootstrap = config["diagnostic_evaluation"]["bootstrap"]
    comparisons = {}
    for model_name in sorted(records_by_model):
        model_values = _trajectory_means(
            state_records,
            method=model_name,
            budgets=budgets,
            metric="normalized_recovery",
        )
        comparisons[model_name] = {}
        for baseline in ("recent", "ocr_rgb"):
            baseline_values = _trajectory_means(
                state_records,
                method=baseline,
                budgets=budgets,
                metric="normalized_recovery",
            )
            delta = {
                key: model_values[key] - baseline_values[key]
                for key in model_values
            }
            comparisons[model_name][f"minus_{baseline}"] = paired_trajectory_bootstrap(
                delta,
                resamples=bootstrap["resamples"],
                seed=bootstrap["seed"],
                interval=bootstrap["interval"],
            )

    scaling = {}
    for family in ("deepsets", "set_transformer"):
        ordered = sorted(
            (
                name
                for name, candidate in candidates_config.items()
                if candidate["family"] == family
            ),
            key=lambda name: candidates_config[name]["fraction"],
        )
        if not ordered:
            continue
        values = [
            summaries[name]["macro_B1_B2_normalized_recovery"]["mean"]
            for name in ordered
        ]
        smallest_to_largest = {}
        for label, compared_budgets in (
            ("macro_B1_B2", budgets),
            ("B1", (1,)),
            ("B2", (2,)),
        ):
            smallest = _trajectory_means(
                state_records,
                method=ordered[0],
                budgets=compared_budgets,
                metric="normalized_recovery",
            )
            largest = _trajectory_means(
                state_records,
                method=ordered[-1],
                budgets=compared_budgets,
                metric="normalized_recovery",
            )
            if set(smallest) != set(largest):
                raise ValueError("scaling endpoint trajectory inventories differ")
            smallest_to_largest[label] = paired_trajectory_bootstrap(
                {key: largest[key] - smallest[key] for key in smallest},
                resamples=bootstrap["resamples"],
                seed=bootstrap["seed"],
                interval=bootstrap["interval"],
            )
        scaling[family] = {
            "candidate_order": ordered,
            "heldout_macro_B1_B2": values,
            "heldout_best_candidate": min(
                ordered,
                key=lambda name: (
                    -summaries[name]["macro_B1_B2_normalized_recovery"]["mean"],
                    candidates_config[name]["fraction"],
                ),
            ),
            "monotonic_nondecreasing": all(
                right >= left for left, right in zip(values, values[1:])
            ),
            "smallest_to_largest_delta": values[-1] - values[0],
            "smallest_to_largest_paired_bootstrap": smallest_to_largest,
            "tune_best_candidate": min(
                ordered, key=lambda name: candidates_config[name]["best_tune_total"]
            ),
        }

    result = {
        "claim_boundary": config["claim_boundary"],
        "comparisons": comparisons,
        "coverage": {
            "completed_exact_state_count": len(state_records),
            "expected_exact_state_count": len(exact_state_ids),
            "skipped_exact_state_count": len(skipped),
            "skipped_state_failures": dict(sorted(skipped.items())),
        },
        "method_summaries": summaries,
        "oracle_summary": oracle_summary,
        "scaling": scaling,
        "schema_version": "1.0.0",
        "state_records": state_records,
        "status": (
            COMPLETED_SCALING_DIAGNOSTIC_WITH_SKIPS
            if skipped
            else COMPLETED_SCALING_DIAGNOSTIC
        ),
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


__all__ = [
    "COMPLETED_SCALING_DIAGNOSTIC",
    "COMPLETED_SCALING_DIAGNOSTIC_WITH_SKIPS",
    "evaluate_scaling_exact_track",
]
