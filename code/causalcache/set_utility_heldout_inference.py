"""Pure selection and merge helpers for rich-token held-out inference."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


BUDGETS = (1, 2, 3, 4)
BASELINE_METHODS = ("recent", "ocr_rgb", "random")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _events(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError("candidate events must be an ordered sequence")
    result = tuple(values)
    if (
        not result
        or any(type(value) is not int or value <= 0 for value in result)
        or result != tuple(sorted(result))
        or len(result) != len(set(result))
    ):
        raise ValueError("candidate events must be sorted unique positive integers")
    return result


def recent_budget_selections(
    event_ids: Sequence[int],
) -> dict[str, list[int]]:
    events = _events(event_ids)
    return {
        str(budget): list(events[-min(budget, len(events)) :])
        for budget in BUDGETS
    }


def ocr_rgb_budget_selections(
    event_ids: Sequence[int], scores: Sequence[float]
) -> dict[str, list[int]]:
    events = _events(event_ids)
    values = tuple(float(value) for value in scores)
    if len(values) != len(events) or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values
    ):
        raise ValueError("OCR/RGB scores must be finite, bounded, and event-aligned")
    ranked = tuple(
        event_id
        for event_id, score in sorted(
            zip(events, values, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        if score > 0.0
    )
    return {
        str(budget): sorted(ranked[: min(budget, len(ranked))])
        for budget in BUDGETS
    }


def random_budget_selections(
    event_ids: Sequence[int], *, state_id: str, seed: int
) -> dict[str, list[int]]:
    events = _events(event_ids)
    if not isinstance(state_id, str) or not state_id:
        raise ValueError("state identity must be non-empty")
    if type(seed) is not int:
        raise TypeError("random-control seed must be an integer")
    permutation = sorted(
        events,
        key=lambda event_id: (
            hashlib.sha256(
                f"{seed}\0{state_id}\0{event_id}".encode("utf-8")
            ).digest(),
            event_id,
        ),
    )
    return {
        str(budget): sorted(permutation[: min(budget, len(permutation))])
        for budget in BUDGETS
    }


def conditional_greedy_budget_path(
    event_ids: Sequence[int],
    *,
    score_batch: Callable[[tuple[tuple[int, ...], ...]], Iterable[float]],
) -> tuple[dict[str, list[int]], dict[str, float], int]:
    """Run one nested greedy path and return every at-most-B prefix."""
    events = _events(event_ids)
    if not callable(score_batch):
        raise TypeError("score_batch must be callable")

    def score(subsets: tuple[tuple[int, ...], ...]) -> tuple[float, ...]:
        values = tuple(float(value) for value in score_batch(subsets))
        if len(values) != len(subsets) or any(
            not math.isfinite(value) for value in values
        ):
            raise ValueError("predicted subset utilities are invalid")
        return values

    current: tuple[int, ...] = ()
    current_utility = score((current,))[0]
    score_count = 1
    selections: dict[str, list[int]] = {}
    utilities: dict[str, float] = {}
    stopped = False
    for budget in BUDGETS:
        if not stopped and len(current) < min(budget, len(events)):
            candidates = tuple(
                tuple(sorted((*current, event_id)))
                for event_id in events
                if event_id not in current
            )
            values = score(candidates)
            score_count += len(candidates)
            best_subset, best_utility = min(
                zip(candidates, values, strict=True),
                key=lambda item: (-item[1], item[0]),
            )
            if best_utility > current_utility:
                current, current_utility = best_subset, best_utility
            else:
                stopped = True
        selections[str(budget)] = list(current)
        utilities[str(budget)] = current_utility
    return selections, utilities, score_count


def beam_budget_path(
    event_ids: Sequence[int],
    *,
    score_batch: Callable[[tuple[tuple[int, ...], ...]], Iterable[float]],
    width: int,
) -> tuple[
    dict[str, list[int]],
    dict[str, float],
    int,
    tuple[dict[str, Any], ...],
]:
    """Search one shared beam through cardinality four and expose its frontiers."""
    events = _events(event_ids)
    if type(width) is not int or width <= 0:
        raise ValueError("beam width must be a positive integer")
    if not callable(score_batch):
        raise TypeError("score_batch must be callable")

    def score(subsets: tuple[tuple[int, ...], ...]) -> tuple[float, ...]:
        values = tuple(float(value) for value in score_batch(subsets))
        if len(values) != len(subsets) or any(
            not math.isfinite(value) for value in values
        ):
            raise ValueError("predicted subset utilities are invalid")
        return values

    empty = ()
    empty_utility = score((empty,))[0]
    score_count = 1
    scored: dict[tuple[int, ...], float] = {empty: empty_utility}
    frontier = (empty,)
    selections: dict[str, list[int]] = {}
    utilities: dict[str, float] = {}
    trace = []
    for budget in BUDGETS:
        bases = frontier
        children = tuple(
            sorted(
                {
                    tuple(sorted((*base, event_id)))
                    for base in bases
                    for event_id in events
                    if event_id not in base
                }
            )
        )
        if children:
            values = score(children)
            score_count += len(children)
            scored.update(zip(children, values, strict=True))
            ranked = sorted(
                zip(children, values, strict=True),
                key=lambda item: (-item[1], item[0]),
            )
            frontier = tuple(subset for subset, _ in ranked[:width])
        else:
            frontier = ()
        feasible = tuple(
            (subset, utility)
            for subset, utility in scored.items()
            if len(subset) <= budget
        )
        selected, selected_utility = min(
            feasible,
            key=lambda item: (-item[1], len(item[0]), item[0]),
        )
        selections[str(budget)] = list(selected)
        utilities[str(budget)] = selected_utility
        trace.append(
            {
                "base_subsets": [list(base) for base in bases],
                "budget": budget,
                "candidate_count": len(children),
                "frontier_subsets": [list(subset) for subset in frontier],
            }
        )
    return selections, utilities, score_count, tuple(trace)


def merge_model_selection_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(payloads) < 2:
        raise ValueError("at least two model selection payloads are required")
    ordered = sorted(payloads, key=lambda value: str(value.get("model_name")))
    if any(
        value.get("status") != "COMPLETED_SET_UTILITY_HELDOUT_MODEL_SELECTIONS"
        for value in ordered
    ):
        raise ValueError("a held-out model selection payload is incomplete")
    binding_keys = (
        "cache_content_sha256",
        "config_sha256",
        "input_content_sha256",
        "inventory_sha256",
    )
    for key in binding_keys:
        if len({value.get(key) for value in ordered}) != 1:
            raise ValueError(f"held-out model selection binding drifted: {key}")
    records_by_model = {
        value["model_name"]: {row["state_id"]: row for row in value["records"]}
        for value in ordered
    }
    state_sets = {tuple(sorted(rows)) for rows in records_by_model.values()}
    if len(state_sets) != 1:
        raise ValueError("model selection payload state inventories differ")
    state_ids = next(iter(state_sets))
    merged_records = []
    for state_id in state_ids:
        rows = [records_by_model[name][state_id] for name in sorted(records_by_model)]
        reference = rows[0]
        identity_keys = (
            "candidate_event_ids",
            "logical_shard",
            "state_id",
            "tracks",
            "trajectory_id",
        )
        if any(
            any(row.get(key) != reference.get(key) for key in identity_keys)
            for row in rows[1:]
        ):
            raise ValueError("held-out model selection state identity drifted")
        for baseline in BASELINE_METHODS:
            if any(
                row["methods"][baseline] != reference["methods"][baseline]
                for row in rows[1:]
            ):
                raise ValueError("held-out baseline selections differ by model")
        methods = {
            baseline: reference["methods"][baseline]
            for baseline in BASELINE_METHODS
        }
        model_metrics = {}
        for row, payload in zip(rows, ordered, strict=True):
            model_name = payload["model_name"]
            methods[model_name] = row["methods"][model_name]
            model_metrics[model_name] = {
                "latency_ms": row["latency_ms"],
                "predicted_utilities": row["predicted_utilities"],
                "subset_score_count": row["subset_score_count"],
            }
        merged_records.append(
            {
                **{key: reference[key] for key in identity_keys},
                "methods": methods,
                "model_metrics": model_metrics,
            }
        )
    result = {
        "cache_content_sha256": ordered[0]["cache_content_sha256"],
        "config_sha256": ordered[0]["config_sha256"],
        "input_content_sha256": ordered[0]["input_content_sha256"],
        "inventory_sha256": ordered[0]["inventory_sha256"],
        "model_artifacts": {
            value["model_name"]: {
                "checkpoint_sha256": value["checkpoint_sha256"],
                "content_sha256": value["content_sha256"],
                "variant": value["variant"],
            }
            for value in ordered
        },
        "records": merged_records,
        "schema_version": "1.0.0",
        "status": "SEALED_SET_UTILITY_HELDOUT_SELECTIONS",
    }
    result["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def merge_model_selection_partitions(
    payloads: Sequence[Mapping[str, Any]],
    *,
    full_input_content_sha256: str,
    expected_state_ids: Sequence[str],
) -> dict[str, Any]:
    if len(payloads) < 2:
        raise ValueError("at least two model selection partitions are required")
    expected = tuple(sorted(expected_state_ids))
    if len(expected) != 805 or len(expected) != len(set(expected)):
        raise ValueError("full held-out input must contain 805 unique states")
    stable_keys = (
        "checkpoint_sha256",
        "config_sha256",
        "inventory_sha256",
        "model_name",
        "variant",
    )
    for key in stable_keys:
        if len({value.get(key) for value in payloads}) != 1:
            raise ValueError(f"held-out model partition binding drifted: {key}")
    if any(
        value.get("status")
        != "COMPLETED_SET_UTILITY_HELDOUT_MODEL_SELECTION_PARTITION"
        for value in payloads
    ):
        raise ValueError("a held-out model selection partition is incomplete")
    records: dict[str, Mapping[str, Any]] = {}
    for payload in payloads:
        for row in payload["records"]:
            if row["state_id"] in records:
                raise ValueError("held-out model selection partitions overlap")
            records[row["state_id"]] = row
    if tuple(sorted(records)) != expected:
        raise ValueError("held-out model selection partitions do not cover 805 states")
    cache_partition_sha256s = sorted(
        str(value["cache_content_sha256"]) for value in payloads
    )
    input_partition_sha256s = sorted(
        str(value["input_content_sha256"]) for value in payloads
    )
    result = {
        "cache_content_sha256": hashlib.sha256(
            canonical_json_bytes(cache_partition_sha256s)
        ).hexdigest(),
        "cache_partition_content_sha256s": cache_partition_sha256s,
        "checkpoint_sha256": payloads[0]["checkpoint_sha256"],
        "config_sha256": payloads[0]["config_sha256"],
        "input_content_sha256": full_input_content_sha256,
        "input_partition_content_sha256s": input_partition_sha256s,
        "inventory_sha256": payloads[0]["inventory_sha256"],
        "model_name": payloads[0]["model_name"],
        "records": [records[state_id] for state_id in expected],
        "schema_version": "1.0.0",
        "status": "COMPLETED_SET_UTILITY_HELDOUT_MODEL_SELECTIONS",
        "variant": payloads[0]["variant"],
    }
    result["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


__all__ = [
    "BASELINE_METHODS",
    "BUDGETS",
    "canonical_json_bytes",
    "conditional_greedy_budget_path",
    "merge_model_selection_payloads",
    "merge_model_selection_partitions",
    "ocr_rgb_budget_selections",
    "random_budget_selections",
    "recent_budget_selections",
]
