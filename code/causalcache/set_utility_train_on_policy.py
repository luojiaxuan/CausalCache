"""Deterministic train-only targeting for on-policy coalition enrichment."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.set_utility_variable_history import history_bin


TRAIN_SELECTION_STATUS = "COMPLETED_SET_UTILITY_TRAIN_SELECTIONS"
HISTORY_BINS = ("very_long", "long", "medium", "short")


def _subset(value: Any, *, candidates: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError("selector subset must be an event-id sequence")
    result = tuple(value)
    if (
        any(type(item) is not int for item in result)
        or result != tuple(sorted(result))
        or len(result) != len(set(result))
        or not set(result).issubset(candidates)
    ):
        raise ValueError("selector subset escaped the candidate universe")
    return result


def validate_train_selections(
    selections: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    if set(selections) != {"deepsets", "set_transformer"}:
        raise ValueError("train enrichment requires DeepSets and Set Transformer")
    binding = None
    records_by_model = {}
    for name, payload in sorted(selections.items()):
        if (
            payload.get("status") != TRAIN_SELECTION_STATUS
            or payload.get("role") != "train"
            or not isinstance(payload.get("variant"), str)
        ):
            raise ValueError(f"invalid train selection payload: {name}")
        current_binding = (
            payload.get("cache_content_sha256"),
            payload.get("config_sha256"),
            payload.get("input_content_sha256"),
        )
        if binding is None:
            binding = current_binding
        elif current_binding != binding:
            raise ValueError("train selector bindings differ")
        rows = {row["state_id"]: row for row in payload.get("records", ())}
        if len(rows) != len(payload.get("records", ())):
            raise ValueError("train selector payload contains duplicate states")
        records_by_model[name] = rows
    inventories = {tuple(sorted(rows)) for rows in records_by_model.values()}
    if len(inventories) != 1:
        raise ValueError("train selector state inventories differ")
    return records_by_model


def _priority(
    state_id: str,
    deepsets: Mapping[str, Any],
    set_transformer: Mapping[str, Any],
) -> tuple[Any, ...]:
    candidates = tuple(set_transformer["candidate_event_ids"])
    if (
        tuple(deepsets["candidate_event_ids"]) != candidates
        or deepsets["trajectory_id"] != set_transformer["trajectory_id"]
        or deepsets["recent"] != set_transformer["recent"]
    ):
        raise ValueError("train selector state identity drifted")
    model_disagreement = 0
    recent_disagreement = 0
    old_event_span = 0.0
    for budget in ("1", "2", "3", "4"):
        deep = set(_subset(deepsets["learned"][budget], candidates=candidates))
        learned = set(
            _subset(set_transformer["learned"][budget], candidates=candidates)
        )
        recent = set(_subset(set_transformer["recent"][budget], candidates=candidates))
        model_disagreement += len(deep.symmetric_difference(learned))
        recent_disagreement += len(recent.symmetric_difference(learned))
        if learned:
            old_event_span += max(candidates) - min(learned)
    margins = []
    for step in set_transformer.get("conditional_steps", ()):
        ranked = step.get("ranked_candidates", ())
        if len(ranked) >= 2:
            first = float(ranked[0]["predicted_utility"])
            second = float(ranked[1]["predicted_utility"])
            if not math.isfinite(first) or not math.isfinite(second):
                raise ValueError("conditional trace utility is non-finite")
            margins.append(abs(first - second))
    uncertainty_margin = min(margins) if margins else math.inf
    return (
        -model_disagreement,
        -recent_disagreement,
        uncertainty_margin,
        -old_event_span,
        hashlib.sha256(state_id.encode("utf-8")).hexdigest(),
    )


def _quotas(
    counts: Mapping[str, int],
    *,
    target: int,
    weights: Mapping[str, float],
) -> dict[str, int]:
    if set(weights) != set(HISTORY_BINS):
        raise ValueError("history-bin weights are incomplete")
    total_weight = sum(float(value) for value in weights.values())
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("history-bin weights must sum to one")
    raw = {name: target * float(weights[name]) for name in HISTORY_BINS}
    result = {name: min(counts.get(name, 0), math.floor(raw[name])) for name in HISTORY_BINS}
    while sum(result.values()) < target:
        available = [name for name in HISTORY_BINS if result[name] < counts.get(name, 0)]
        if not available:
            raise ValueError("target exceeds available train states")
        name = min(
            available,
            key=lambda item: (
                -round(raw[item] - result[item], 12),
                HISTORY_BINS.index(item),
            ),
        )
        result[name] += 1
    return result


def select_targeted_states(
    selections: Mapping[str, Mapping[str, Any]],
    *,
    target_fraction: float,
    history_bin_weights: Mapping[str, float],
    maximum_states_per_trajectory: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    records = validate_train_selections(selections)
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target fraction must be in (0, 1]")
    if maximum_states_per_trajectory <= 0:
        raise ValueError("trajectory cap must be positive")
    state_ids = sorted(records["set_transformer"])
    target = max(1, round(len(state_ids) * target_fraction))
    by_bin = {name: [] for name in HISTORY_BINS}
    for state_id in state_ids:
        row = records["set_transformer"][state_id]
        candidates = tuple(row["candidate_event_ids"])
        by_bin[history_bin(len(candidates))].append(state_id)
    counts = {name: len(values) for name, values in by_bin.items()}
    quotas = _quotas(counts, target=target, weights=history_bin_weights)
    for name in HISTORY_BINS:
        by_bin[name].sort(
            key=lambda state_id: _priority(
                state_id,
                records["deepsets"][state_id],
                records["set_transformer"][state_id],
            )
        )

    selected = []
    selected_set = set()
    trajectory_counts: Counter[str] = Counter()
    realized = Counter()
    for name in HISTORY_BINS:
        cap = maximum_states_per_trajectory
        while realized[name] < quotas[name]:
            added = False
            for state_id in by_bin[name]:
                if state_id in selected_set:
                    continue
                trajectory_id = records["set_transformer"][state_id]["trajectory_id"]
                if trajectory_counts[trajectory_id] >= cap:
                    continue
                selected.append(state_id)
                selected_set.add(state_id)
                trajectory_counts[trajectory_id] += 1
                realized[name] += 1
                added = True
                if realized[name] == quotas[name]:
                    break
            if not added:
                cap += 1
    return tuple(sorted(selected)), {
        "history_bin_available_counts": counts,
        "history_bin_selected_counts": dict(realized),
        "maximum_realized_states_per_trajectory": max(trajectory_counts.values()),
        "selected_state_count": len(selected),
        "target_fraction": target_fraction,
        "target_state_count": target,
        "trajectory_count": len(trajectory_counts),
    }


def enrichment_coalitions(
    state_id: str,
    records_by_model: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    set_row = records_by_model["set_transformer"][state_id]
    candidates = tuple(set_row["candidate_event_ids"])
    coalitions: list[tuple[int, ...]] = [()]
    sources = ["anchor_empty"]

    def add(value: Any, source: str) -> None:
        subset = _subset(value, candidates=candidates)
        if subset not in coalitions:
            coalitions.append(subset)
            sources.append(source)

    for model_name in ("deepsets", "set_transformer"):
        row = records_by_model[model_name][state_id]
        for budget in ("1", "2", "3", "4"):
            add(row["learned"][budget], f"learned_{model_name}_b{budget}")
        for step_index, step in enumerate(row.get("conditional_steps", ()), start=1):
            for rank, candidate in enumerate(step.get("ranked_candidates", ()), start=1):
                add(
                    candidate["subset"],
                    f"conditional_{model_name}_step{step_index}_rank{rank}",
                )
    for budget in ("1", "2", "3", "4"):
        add(set_row["recent"][budget], f"recent_b{budget}")
    add(candidates, "anchor_full")
    return tuple(
        {"event_ids": list(subset), "source": source}
        for subset, source in zip(coalitions, sources, strict=True)
    )
