"""Self-contained exact-B4 reduction for frozen held-out selectors."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from causalcache.set_utility_heldout_evaluation import (
    COMPLETED_LABEL,
    SKIPPED_LABEL,
    canonical_json_bytes,
)


COMPLETED_B4_EVALUATION = "COMPLETED_SET_UTILITY_B4_ORACLE_EVALUATION"
INCOMPLETE_B4_EVALUATION = "INCOMPLETE_SET_UTILITY_B4_ORACLE_EVALUATION"
B4_METHODS = ("deepsets", "set_transformer", "recent", "ocr_rgb")


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{label} is outside its finite domain")
    return result


def _selection_content_is_valid(payload: Mapping[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    if not isinstance(claimed, str):
        return False
    unsigned = dict(payload)
    del unsigned["content_sha256"]
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() == claimed


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


def _exact_subset(
    distances: Mapping[tuple[int, ...], float], *, budget: int
) -> tuple[tuple[int, ...], float]:
    return min(
        (
            (subset, distance)
            for subset, distance in distances.items()
            if len(subset) <= budget
        ),
        key=lambda item: (item[1], len(item[0]), item[0]),
    )


def _true_conditional_greedy(
    distances: Mapping[tuple[int, ...], float],
    *,
    candidates: tuple[int, ...],
    budget: int,
) -> tuple[tuple[int, ...], float]:
    selected: tuple[int, ...] = ()
    distance = distances[selected]
    for _ in range(budget):
        extensions = []
        for event_id in candidates:
            if event_id in selected:
                continue
            subset = tuple(sorted((*selected, event_id)))
            extensions.append((subset, distances[subset]))
        subset, next_distance = min(
            extensions, key=lambda item: (item[1], item[0])
        )
        if next_distance >= distance:
            break
        selected, distance = subset, next_distance
    return selected, distance


def _metric(
    *, subset: tuple[int, ...], distance: float, empty_distance: float, floor: float
) -> dict[str, Any]:
    utility = empty_distance - distance
    return {
        "distance": distance,
        "normalized_recovery": utility / max(empty_distance, floor),
        "selected_cardinality": len(subset),
        "selected_event_ids": list(subset),
        "utility": utility,
    }


def _trajectory_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    budget: int,
    metric: str,
    method: str | None = None,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        source = row[family] if method is None else row[family][method]
        grouped[row["trajectory_id"]].append(float(source[str(budget)][metric]))
    return {
        trajectory_id: fmean(values)
        for trajectory_id, values in sorted(grouped.items())
    }


def _summary(values: Mapping[str, float]) -> dict[str, Any]:
    if not values:
        raise ValueError("trajectory-equal summary is empty")
    return {"mean": fmean(values.values()), "trajectory_count": len(values)}


def evaluate_b4_oracle(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    execution_config_sha256: str,
    selections: Mapping[str, Any],
    terminals: Mapping[str, Mapping[str, Any]],
    source_revision: str,
) -> dict[str, Any]:
    if not _selection_content_is_valid(selections):
        raise ValueError("B4 sealed selection content drifted")
    selection_contract = config["inputs"]["sealed_selections"]
    if selections.get("content_sha256") != selection_contract["content_sha256"]:
        raise ValueError("B4 sealed selection binding drifted")
    budgets = tuple(config["evaluation"]["budgets"])
    if budgets != (1, 2, 3, 4):
        raise ValueError("B4 budgets must be exactly 1,2,3,4")
    floor = _finite(
        config["evaluation"]["normalization_floor"], label="normalization floor"
    )
    if floor <= 0.0:
        raise ValueError("normalization floor must be positive")
    state_contract = config["state_selection"]
    selected = tuple(
        row
        for row in selections["records"]
        if state_contract["required_track"] in row["tracks"]
        and state_contract["minimum_candidate_count"]
        <= len(row["candidate_event_ids"])
        <= state_contract["maximum_candidate_count"]
    )
    if (
        len(selected) != state_contract["expected_state_count"]
        or len({row["trajectory_id"] for row in selected})
        != state_contract["expected_trajectory_count"]
    ):
        raise ValueError("B4 selected state inventory drifted")
    expected_ids = {row["state_id"] for row in selected}
    if len(expected_ids) != len(selected) or set(terminals) - expected_ids:
        raise ValueError("B4 terminal state inventory drifted")

    records = []
    skipped: dict[str, str] = {}
    missing = []
    for selection in sorted(selected, key=lambda row: row["state_id"]):
        state_id = selection["state_id"]
        terminal = terminals.get(state_id)
        if terminal is None:
            missing.append(state_id)
            continue
        if terminal.get("status") == SKIPPED_LABEL:
            skipped[state_id] = str(terminal.get("failure_class", "UnknownFailure"))
            continue
        if terminal.get("status") != COMPLETED_LABEL:
            raise ValueError(f"B4 terminal status is invalid: {state_id}")
        candidates = tuple(selection["candidate_event_ids"])
        if (
            terminal.get("state_id") != state_id
            or terminal.get("trajectory_id") != selection["trajectory_id"]
            or terminal.get("role") != "evaluation"
            or tuple(terminal.get("candidate_event_step_ids", ())) != candidates
            or terminal.get("scientific_config_sha256")
            != config["reference"]["scientific_config_sha256"]
            or terminal.get("execution_config_sha256") != execution_config_sha256
            or terminal.get("source_revision") != source_revision
        ):
            raise ValueError(f"B4 terminal identity drifted: {state_id}")
        distances: dict[tuple[int, ...], float] = {}
        for ordinal, raw in enumerate(terminal.get("distance_rows", ())):
            subset = _subset(
                raw.get("coalition_event_step_ids"),
                candidates=candidates,
                budget=len(candidates),
                label=f"{state_id} distance row {ordinal}",
            )
            if subset in distances:
                raise ValueError(f"duplicate B4 distance subset: {state_id}")
            distances[subset] = _finite(
                raw.get("distance"),
                label=f"{state_id} distance {subset}",
                nonnegative=True,
            )
        expected_subsets = {
            subset
            for cardinality in range(5)
            for subset in itertools.combinations(candidates, cardinality)
        }
        expected_subsets.add(candidates)
        if set(distances) != expected_subsets or distances[candidates] != 0.0:
            raise ValueError(f"self-contained exact B4 table is incomplete: {state_id}")
        empty_distance = distances[()]
        exact = {}
        greedy = {}
        for budget in budgets:
            subset, distance = _exact_subset(distances, budget=budget)
            exact[str(budget)] = _metric(
                subset=subset,
                distance=distance,
                empty_distance=empty_distance,
                floor=floor,
            )
            subset, distance = _true_conditional_greedy(
                distances, candidates=candidates, budget=budget
            )
            greedy[str(budget)] = _metric(
                subset=subset,
                distance=distance,
                empty_distance=empty_distance,
                floor=floor,
            )
        methods = {}
        for method in B4_METHODS:
            if method not in selection["methods"]:
                raise ValueError(f"B4 selection omits method {method}: {state_id}")
            methods[method] = {}
            for budget in budgets:
                subset = _subset(
                    selection["methods"][method][str(budget)],
                    candidates=candidates,
                    budget=budget,
                    label=f"{state_id} {method} B{budget}",
                )
                metric = _metric(
                    subset=subset,
                    distance=distances[subset],
                    empty_distance=empty_distance,
                    floor=floor,
                )
                metric["oracle_regret"] = metric["distance"] - exact[str(budget)][
                    "distance"
                ]
                methods[method][str(budget)] = metric
        records.append(
            {
                "exact": exact,
                "methods": methods,
                "state_id": state_id,
                "trajectory_id": selection["trajectory_id"],
                "true_conditional_greedy": greedy,
            }
        )

    coverage_complete = len(records) == len(selected)
    summaries: dict[str, Any] = {"methods": {}}
    for family in ("exact", "true_conditional_greedy"):
        summaries[family] = {
            str(budget): {
                metric: _summary(
                    _trajectory_values(
                        records,
                        family=family,
                        budget=budget,
                        metric=metric,
                    )
                )
                for metric in ("normalized_recovery", "selected_cardinality", "utility")
            }
            for budget in budgets
        }
    for method in B4_METHODS:
        summaries["methods"][method] = {
            str(budget): {
                metric: _summary(
                    _trajectory_values(
                        records,
                        family="methods",
                        method=method,
                        budget=budget,
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
            for budget in budgets
        }

    search_gap = {}
    distillation_gap = {method: {} for method in B4_METHODS}
    for budget in budgets:
        exact_values = _trajectory_values(
            records, family="exact", budget=budget, metric="normalized_recovery"
        )
        greedy_values = _trajectory_values(
            records,
            family="true_conditional_greedy",
            budget=budget,
            metric="normalized_recovery",
        )
        search_gap[str(budget)] = _summary(
            {key: exact_values[key] - greedy_values[key] for key in exact_values}
        )
        for method in B4_METHODS:
            method_values = _trajectory_values(
                records,
                family="methods",
                method=method,
                budget=budget,
                metric="normalized_recovery",
            )
            distillation_gap[method][str(budget)] = _summary(
                {key: exact_values[key] - method_values[key] for key in exact_values}
            )
    exact_b2 = _trajectory_values(
        records, family="exact", budget=2, metric="normalized_recovery"
    )
    exact_b4 = _trajectory_values(
        records, family="exact", budget=4, metric="normalized_recovery"
    )
    cardinalities = {
        str(budget): dict(
            sorted(
                Counter(
                    row["exact"][str(budget)]["selected_cardinality"]
                    for row in records
                ).items()
            )
        )
        for budget in budgets
    }
    result = {
        "bindings": {
            "config_sha256": config_sha256,
            "execution_config_sha256": execution_config_sha256,
            "source_revision": source_revision,
        },
        "coverage": {
            "complete": coverage_complete,
            "completed_state_count": len(records),
            "expected_state_count": len(selected),
            "missing_state_ids": sorted(missing),
            "skipped_state_count": len(skipped),
            "skipped_state_failures": dict(sorted(skipped.items())),
        },
        "diagnostics": {
            "B2_to_B4_exact_normalized_recovery_gain": _summary(
                {key: exact_b4[key] - exact_b2[key] for key in exact_b2}
            ),
            "distillation_gap_exact_minus_method_normalized_recovery": distillation_gap,
            "exact_minus_true_greedy_normalized_recovery": search_gap,
            "exact_selected_cardinality_state_counts": cardinalities,
        },
        "schema_version": "1.0.0",
        "state_records": records,
        "status": (
            COMPLETED_B4_EVALUATION
            if coverage_complete
            else INCOMPLETE_B4_EVALUATION
        ),
        "summaries": summaries,
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


__all__ = [
    "B4_METHODS",
    "COMPLETED_B4_EVALUATION",
    "INCOMPLETE_B4_EVALUATION",
    "evaluate_b4_oracle",
]
