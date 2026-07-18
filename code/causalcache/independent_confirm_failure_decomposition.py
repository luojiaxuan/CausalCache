"""Pure failure decomposition for the consumed independent confirm-20 report."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.gate_v1_data import label_state_from_restoration_record
from causalcache.gate_v1_evaluation import type7_quantile
from causalcache.independent_confirm_artifact import (
    read_fixed_report,
    read_state_records_jsonl,
)
from causalcache.independent_confirm_data import (
    CONFIRM_BUDGET_EVENT_CAPACITY,
    CONFIRM_CANDIDATE_EVENT_STEP_IDS,
    CONFIRM_DECISION_STEP_ID,
    CONFIRM_SOURCE_IDS,
    CONFIRM_STATE_COUNT,
)
from causalcache.restoration_v2_2_label_table import (
    ValidatedDistanceTable,
    primary_exact_subset_oracle,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_independent_confirm20_failure_decomposition_v1"
VALID_STATE_OUTCOME = "VALID_INDEPENDENT_CONFIRM_STATE"
MEMORY_SENSITIVE_EPSILON = 1e-12
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 271_828
BOOTSTRAP_CONFIDENCE = 0.90

METHOD_ORDER = (
    "exact",
    "oracle_independent",
    "learned_independent",
    "ocr_rgb_v2",
)


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be one finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be one finite scalar")
    return result


def _close(left: Any, right: float, *, label: str) -> float:
    value = _finite(left, label=label)
    if not math.isclose(value, right, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} differs from the distance-table replay")
    return value


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= MEMORY_SENSITIVE_EPSILON:
        return None
    return numerator / denominator


def oracle_independent_scores(
    table: ValidatedDistanceTable,
) -> Mapping[int, float]:
    """Return the exact empty/singleton-projected independent raw targets."""
    if not isinstance(table, ValidatedDistanceTable):
        raise TypeError("oracle independent scores require a ValidatedDistanceTable")
    if len(table.event_ids) < 2:
        raise ValueError("oracle independent projection requires at least two events")
    scores: dict[int, float] = {}
    for event_id in table.event_ids:
        empty_gain = table.distance(()) - table.distance((event_id,))
        singleton_gains = tuple(
            table.distance((other,))
            - table.distance(tuple(sorted((other, event_id))))
            for other in table.event_ids
            if other != event_id
        )
        scores[event_id] = 0.5 * (
            empty_gain + math.fsum(singleton_gains) / len(singleton_gains)
        )
    return scores


def oracle_independent_selection(
    table: ValidatedDistanceTable,
    *,
    budget: int = CONFIRM_BUDGET_EVENT_CAPACITY,
) -> tuple[int, ...]:
    """Apply the frozen static-positive top-B deployment rule to true targets."""
    if type(budget) is not int or budget <= 0:
        raise ValueError("oracle independent budget must be a positive integer")
    scores = oracle_independent_scores(table)
    ranked = sorted(
        (event for event, score in scores.items() if score > 0.0),
        key=lambda event: (-scores[event], event),
    )
    return tuple(sorted(ranked[:budget]))


def _coalition_mask(event_ids: Sequence[int], coalition: Sequence[int]) -> int:
    selected = set(coalition)
    return sum(1 << index for index, event in enumerate(event_ids) if event in selected)


def _validated_table(record: Mapping[str, Any], *, ordinal: int) -> ValidatedDistanceTable:
    state = record.get("state")
    if (
        not isinstance(state, Mapping)
        or record.get("outcome") != VALID_STATE_OUTCOME
        or record.get("failure") is not None
        or state.get("ordinal") != ordinal
        or state.get("source_id") != CONFIRM_SOURCE_IDS[ordinal]
        or state.get("state_id")
        != f"{CONFIRM_SOURCE_IDS[ordinal]}:decision_step:{CONFIRM_DECISION_STEP_ID:03d}"
        or state.get("decision_step_id") != CONFIRM_DECISION_STEP_ID
        or state.get("candidate_event_step_ids")
        != list(CONFIRM_CANDIDATE_EVENT_STEP_IDS)
    ):
        raise ValueError("confirm raw record identity, outcome, or geometry drifted")
    label = label_state_from_restoration_record(record, record_schema="expansion")
    table = label.table
    rows = record.get("distance_rows")
    if not isinstance(rows, list) or len(rows) != 2 ** len(table.event_ids):
        raise ValueError("confirm raw record lacks the complete powerset")
    if len(rows) != len(table.rows):
        raise ValueError("confirm raw distance-row denominator drifted")
    baseline = table.distance(())
    for raw, replay in zip(rows, table.rows, strict=True):
        if (
            not isinstance(raw, Mapping)
            or raw.get("coalition") != list(replay.coalition)
            or raw.get("coalition_mask")
            != _coalition_mask(table.event_ids, replay.coalition)
        ):
            raise ValueError("confirm raw coalition order or mask drifted")
        _close(raw.get("distance"), replay.distance, label="raw coalition distance")
        _close(
            raw.get("utility"),
            baseline - replay.distance,
            label="raw coalition utility",
        )
    return table


def _selection(
    raw: Any,
    *,
    table: ValidatedDistanceTable,
    label: str,
) -> tuple[int, ...]:
    if (
        not isinstance(raw, list)
        or any(type(event) is not int for event in raw)
        or raw != sorted(raw)
        or len(set(raw)) != len(raw)
        or len(raw) > CONFIRM_BUDGET_EVENT_CAPACITY
        or not set(raw).issubset(table.event_ids)
    ):
        raise ValueError(f"{label} selection is infeasible")
    return tuple(raw)


def _metric(
    table: ValidatedDistanceTable,
    selected: Sequence[int],
) -> dict[str, Any]:
    coalition = tuple(selected)
    baseline = table.distance(())
    distance = table.distance(coalition)
    utility = baseline - distance
    return {
        "selected": list(coalition),
        "distance": distance,
        "raw_utility": utility,
        "normalized_recovery": (
            utility / baseline if baseline > MEMORY_SENSITIVE_EPSILON else None
        ),
    }


def _validate_report_metric(
    raw: Any,
    *,
    table: ValidatedDistanceTable,
    selected: tuple[int, ...],
    label: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "selected",
        "distance",
        "raw_utility",
        "normalized_recovery",
    }:
        raise ValueError(f"{label} metric schema drifted")
    if _selection(raw.get("selected"), table=table, label=label) != selected:
        raise ValueError(f"{label} selected coalition differs from replay")
    expected = _metric(table, selected)
    _close(raw.get("distance"), expected["distance"], label=f"{label} distance")
    _close(
        raw.get("raw_utility"),
        expected["raw_utility"],
        label=f"{label} raw utility",
    )
    normalized = raw.get("normalized_recovery")
    if expected["normalized_recovery"] is None:
        if normalized is not None:
            raise ValueError(f"{label} normalized recovery must be null")
    else:
        _close(
            normalized,
            expected["normalized_recovery"],
            label=f"{label} normalized recovery",
        )
    return expected


def _paired_interval(values: Sequence[float]) -> tuple[float, float]:
    deltas = tuple(_finite(value, label="paired delta") for value in values)
    if len(deltas) != CONFIRM_STATE_COUNT:
        raise ValueError("paired bootstrap requires the fixed 20 trajectories")
    generator = random.Random(BOOTSTRAP_SEED)
    means = tuple(
        math.fsum(deltas[generator.randrange(len(deltas))] for _ in deltas)
        / len(deltas)
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return type7_quantile(means, tail), type7_quantile(means, 1.0 - tail)


def _method_summary(
    raw_values: Sequence[float],
    normalized_values: Sequence[float | None],
    cardinalities: Sequence[int],
    *,
    exact_raw_sum: float,
    exact_normalized_mean: float,
    baseline_sum: float,
) -> dict[str, Any]:
    raw_sum = math.fsum(raw_values)
    normalized_mean = math.fsum(
        0.0 if value is None else value for value in normalized_values
    ) / CONFIRM_STATE_COUNT
    return {
        "raw_utility_sum": raw_sum,
        "raw_ratio_to_exact": _ratio(raw_sum, exact_raw_sum),
        "retained_baseline_mass": _ratio(raw_sum, baseline_sum),
        "fixed20_zero_imputed_mean_normalized_recovery": normalized_mean,
        "normalized_ratio_to_exact": _ratio(normalized_mean, exact_normalized_mean),
        "mean_selected_cardinality": math.fsum(cardinalities)
        / CONFIRM_STATE_COUNT,
    }


def _comparison(
    left_raw: Sequence[float],
    right_raw: Sequence[float],
    left_normalized: Sequence[float | None],
    right_normalized: Sequence[float | None],
    left_selected: Sequence[tuple[int, ...]],
    right_selected: Sequence[tuple[int, ...]],
) -> dict[str, Any]:
    raw_deltas = tuple(
        left - right for left, right in zip(left_raw, right_raw, strict=True)
    )
    normalized_deltas = tuple(
        (0.0 if left is None else left) - (0.0 if right is None else right)
        for left, right in zip(left_normalized, right_normalized, strict=True)
    )

    def summary(values: tuple[float, ...]) -> dict[str, Any]:
        lower, upper = _paired_interval(values)
        return {
            "mean_delta": math.fsum(values) / CONFIRM_STATE_COUNT,
            "sum_delta": math.fsum(values),
            "positive_trajectory_count": sum(value > 0.0 for value in values),
            "equal_trajectory_count": sum(value == 0.0 for value in values),
            "negative_trajectory_count": sum(value < 0.0 for value in values),
            "paired_bootstrap_90_interval": [lower, upper],
        }

    return {
        "raw": summary(raw_deltas),
        "normalized": summary(normalized_deltas),
        "selection_agreement_count": sum(
            left == right
            for left, right in zip(left_selected, right_selected, strict=True)
        ),
    }


def build_independent_confirm_failure_decomposition(
    raw_state_records_bytes: bytes,
    fixed_report_bytes: bytes,
) -> dict[str, Any]:
    """Replay J and the sealed comparators from the consumed confirm bytes."""
    if not isinstance(raw_state_records_bytes, bytes) or not isinstance(
        fixed_report_bytes, bytes
    ):
        raise TypeError("failure decomposition inputs must be exact bytes")
    raw_records = read_state_records_jsonl(raw_state_records_bytes)
    report = read_fixed_report(fixed_report_bytes)
    if (
        len(raw_records) != CONFIRM_STATE_COUNT
        or report.get("evaluation_performed") is not True
        or report.get("fixed_state_denominator") != CONFIRM_STATE_COUNT
        or report.get("reference")
        != {
            "expected_count": CONFIRM_STATE_COUNT,
            "success_count": CONFIRM_STATE_COUNT,
            "failure_count": 0,
        }
        or not isinstance(report.get("records"), list)
        or len(report["records"]) != CONFIRM_STATE_COUNT
    ):
        raise ValueError("failure decomposition requires a complete confirm-20 report")

    method_raw: dict[str, list[float]] = {name: [] for name in METHOD_ORDER}
    method_normalized: dict[str, list[float | None]] = {
        name: [] for name in METHOD_ORDER
    }
    method_selected: dict[str, list[tuple[int, ...]]] = {
        name: [] for name in METHOD_ORDER
    }
    baselines: list[float] = []
    state_rows: list[dict[str, Any]] = []

    for ordinal, (raw_record, report_record) in enumerate(
        zip(raw_records, report["records"], strict=True)
    ):
        table = _validated_table(raw_record, ordinal=ordinal)
        source_id = CONFIRM_SOURCE_IDS[ordinal]
        state_id = f"{source_id}:decision_step:{CONFIRM_DECISION_STEP_ID:03d}"
        if (
            not isinstance(report_record, Mapping)
            or report_record.get("source_id") != source_id
            or report_record.get("state_id") != state_id
            or report_record.get("decision_step_id") != CONFIRM_DECISION_STEP_ID
            or report_record.get("candidate_event_step_ids")
            != list(CONFIRM_CANDIDATE_EVENT_STEP_IDS)
            or report_record.get("baseline_distance") != table.distance(())
            or report_record.get("memory_sensitive")
            is not (table.distance(()) > MEMORY_SENSITIVE_EPSILON)
        ):
            raise ValueError("fixed report state identity or baseline drifted")

        exact = primary_exact_subset_oracle(table).coalition
        oracle = oracle_independent_selection(table)
        learned_raw = report_record.get("independent")
        heuristics = report_record.get("heuristics")
        if not isinstance(learned_raw, Mapping) or not isinstance(heuristics, Mapping):
            raise ValueError("fixed report lacks learned or heuristic metrics")
        learned = _selection(
            learned_raw.get("selected"), table=table, label="learned independent"
        )
        if set(heuristics) != {
            "dynamic_recent",
            "ocr_rgb_v2",
            "policy_vision_v3",
        }:
            raise ValueError("fixed report heuristic inventory drifted")
        ocr_raw = heuristics["ocr_rgb_v2"]
        if not isinstance(ocr_raw, Mapping):
            raise ValueError("fixed report OCR/RGB metric is malformed")
        ocr = _selection(ocr_raw.get("selected"), table=table, label="OCR/RGB")

        expected_metrics = {
            "exact": _validate_report_metric(
                report_record.get("exact"),
                table=table,
                selected=exact,
                label="exact",
            ),
            "oracle_independent": _metric(table, oracle),
            "learned_independent": _validate_report_metric(
                learned_raw,
                table=table,
                selected=learned,
                label="learned independent",
            ),
            "ocr_rgb_v2": _validate_report_metric(
                ocr_raw,
                table=table,
                selected=ocr,
                label="OCR/RGB",
            ),
        }
        selections = {
            "exact": exact,
            "oracle_independent": oracle,
            "learned_independent": learned,
            "ocr_rgb_v2": ocr,
        }
        baseline = table.distance(())
        baselines.append(baseline)
        for name in METHOD_ORDER:
            metric = expected_metrics[name]
            method_raw[name].append(metric["raw_utility"])
            method_normalized[name].append(metric["normalized_recovery"])
            method_selected[name].append(selections[name])
        state_rows.append(
            {
                "ordinal": ordinal,
                "source_id": source_id,
                "state_id": state_id,
                "baseline_distance": baseline,
                "oracle_independent_scores": [
                    {"event_step_id": event, "raw_target": score}
                    for event, score in oracle_independent_scores(table).items()
                ],
                "methods": expected_metrics,
            }
        )

    baseline_sum = math.fsum(baselines)
    exact_raw_sum = math.fsum(method_raw["exact"])
    exact_normalized_mean = math.fsum(
        0.0 if value is None else value for value in method_normalized["exact"]
    ) / CONFIRM_STATE_COUNT
    methods = {
        name: _method_summary(
            method_raw[name],
            method_normalized[name],
            [len(selected) for selected in method_selected[name]],
            exact_raw_sum=exact_raw_sum,
            exact_normalized_mean=exact_normalized_mean,
            baseline_sum=baseline_sum,
        )
        for name in METHOD_ORDER
    }

    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("fixed report aggregate metrics are missing")
    _close(
        metrics.get("baseline_distance_sum"),
        baseline_sum,
        label="fixed report baseline sum",
    )
    _close(
        metrics.get("exact_raw_utility_sum"),
        exact_raw_sum,
        label="fixed report exact raw sum",
    )
    _close(
        metrics.get("independent_raw_utility_sum"),
        methods["learned_independent"]["raw_utility_sum"],
        label="fixed report learned raw sum",
    )
    heuristic_comparisons = metrics.get("heuristic_comparisons")
    if not isinstance(heuristic_comparisons, Mapping) or not isinstance(
        heuristic_comparisons.get("ocr_rgb_v2"), Mapping
    ):
        raise ValueError("fixed report OCR/RGB aggregate is missing")
    _close(
        heuristic_comparisons["ocr_rgb_v2"].get("heuristic_raw_utility_sum"),
        methods["ocr_rgb_v2"]["raw_utility_sum"],
        label="fixed report OCR/RGB raw sum",
    )

    comparisons = {
        f"oracle_independent_minus_{right}": _comparison(
            method_raw["oracle_independent"],
            method_raw[right],
            method_normalized["oracle_independent"],
            method_normalized[right],
            method_selected["oracle_independent"],
            method_selected[right],
        )
        for right in ("exact", "ocr_rgb_v2", "learned_independent")
    }
    comparisons["learned_independent_minus_ocr_rgb_v2"] = _comparison(
        method_raw["learned_independent"],
        method_raw["ocr_rgb_v2"],
        method_normalized["learned_independent"],
        method_normalized["ocr_rgb_v2"],
        method_selected["learned_independent"],
        method_selected["ocr_rgb_v2"],
    )
    j_minus_ocr = comparisons["oracle_independent_minus_ocr_rgb_v2"]["raw"]
    j_minus_i = comparisons["oracle_independent_minus_learned_independent"]["raw"]
    i_minus_ocr = comparisons["learned_independent_minus_ocr_rgb_v2"]["raw"]
    case_a = j_minus_ocr["mean_delta"] <= 0.0
    case_b = (
        j_minus_ocr["mean_delta"] > 0.0
        and j_minus_ocr["paired_bootstrap_90_interval"][0] > 0.0
        and j_minus_ocr["positive_trajectory_count"] >= 12
        and i_minus_ocr["mean_delta"] < 0.0
    )
    student_gap_supported = (
        j_minus_i["mean_delta"] > 0.0
        and j_minus_i["paired_bootstrap_90_interval"][0] > 0.0
    )
    route = (
        "CASE_A_ORACLE_INDEPENDENT_LOSES_TO_OCR_RGB"
        if case_a
        else "CASE_B_TEACHER_VALID_STUDENT_DISTILLATION_GAP"
        if case_b
        else "INCONCLUSIVE_ORACLE_INDEPENDENT_CONFIRM_DECOMPOSITION"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_INDEPENDENT_CONFIRM20_ORACLE_INDEPENDENT_DECOMPOSITION_V1",
        "inputs": {
            "raw_state_records_sha256": hashlib.sha256(
                raw_state_records_bytes
            ).hexdigest(),
            "fixed_report_sha256": hashlib.sha256(fixed_report_bytes).hexdigest(),
            "state_count": CONFIRM_STATE_COUNT,
        },
        "selector_contract": {
            "budget_event_capacity": CONFIRM_BUDGET_EVENT_CAPACITY,
            "positive_score_strictly_greater_than_zero": True,
            "score_order": "descending_raw_target_then_ascending_event_step_id",
            "final_coalition_order": "ascending_event_step_id",
            "exact_oracle": (
                "minimum_distance_at_most_B_then_smaller_cardinality_then_lexicographic"
            ),
        },
        "bootstrap": {
            "unit": "trajectory",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "interval": "two_sided_percentile",
            "quantile": "Hyndman_Fan_type_7",
        },
        "aggregate": {
            "baseline_distance_sum": baseline_sum,
            "methods": methods,
        },
        "comparisons": comparisons,
        "decision": {
            "primary_metric": "raw_utility_sum",
            "status": route,
            "case_a_j_minus_ocr_raw_mean_at_most_zero": case_a,
            "case_b_j_minus_ocr_raw_mean_strictly_positive": (
                j_minus_ocr["mean_delta"] > 0.0
            ),
            "case_b_j_minus_ocr_bootstrap_lower_strictly_positive": (
                j_minus_ocr["paired_bootstrap_90_interval"][0] > 0.0
            ),
            "case_b_j_minus_ocr_positive_trajectory_count_at_least_12": (
                j_minus_ocr["positive_trajectory_count"] >= 12
            ),
            "case_b_i_minus_ocr_raw_mean_strictly_negative": (
                i_minus_ocr["mean_delta"] < 0.0
            ),
            "oracle_independent_student_gap_supported": student_gap_supported,
            "same_objective_student_rescue_supported": case_b,
            "normalized_metrics_can_change_primary_case": False,
            "parent_confirm_verdict": "NO_GO_INDEPENDENT_CONFIRM",
            "parent_confirm_verdict_locked": True,
            "authorizations": {
                "confirm_reclassification": False,
                "closed_loop": False,
                "matched_nll": False,
                "sealed_androidworld_test": False,
                "gate_training_on_confirm20": False,
            },
        },
        "state_rows": state_rows,
        "operation_counts": {
            "file_read_count": 0,
            "network_call_count": 0,
            "gpu_call_count": 0,
            "model_forward_count": 0,
            "training_call_count": 0,
            "remote_mutation_count": 0,
        },
    }


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "PROTOCOL_ID",
    "build_independent_confirm_failure_decomposition",
    "oracle_independent_scores",
    "oracle_independent_selection",
]
