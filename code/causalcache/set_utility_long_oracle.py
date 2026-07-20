"""Wave-based true-greedy oracle diagnostic for long-history states."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.set_utility_variable_history import (
    history_bin,
    logical_shard_for_trajectory,
    states_from_assignments,
)


DIAGNOSTIC_BINS = ("very_long", "long")
BUDGETS = (1, 2, 3, 4)
WAVES = (1, 2, 3, 4)
SELECTION_STATUS = "COMPLETED_SET_UTILITY_LONG_ORACLE_SELECTION"


def _canonical_subset(value: Any, *, candidates: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError("long-oracle subset must be an event-id sequence")
    subset = tuple(value)
    if (
        any(type(item) is not int for item in subset)
        or subset != tuple(sorted(subset))
        or len(subset) != len(set(subset))
        or not set(subset).issubset(candidates)
    ):
        raise ValueError("long-oracle subset escaped the candidate universe")
    return subset


def _state_order_key(state_id: str, *, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{state_id}".encode("utf-8")).hexdigest()


def select_long_oracle_states(
    assignments: Sequence[Mapping[str, Any]],
    *,
    bin_targets: Mapping[str, int],
    maximum_states_per_trajectory: int,
    salt: str,
) -> tuple[dict[str, Any], ...]:
    """Deterministically sample train long/very-long states with a trajectory cap."""
    if set(bin_targets) != set(DIAGNOSTIC_BINS):
        raise ValueError("bin targets must cover exactly long and very_long")
    if any(
        type(value) is not int or value <= 0 for value in bin_targets.values()
    ):
        raise ValueError("bin targets must be positive integers")
    if type(maximum_states_per_trajectory) is not int or (
        maximum_states_per_trajectory <= 0
    ):
        raise ValueError("trajectory cap must be positive")
    if not isinstance(salt, str) or not salt:
        raise ValueError("selection salt must be non-empty text")
    by_bin: dict[str, list[Any]] = {name: [] for name in DIAGNOSTIC_BINS}
    for state in states_from_assignments(assignments):
        if state.role != "train":
            continue
        name = history_bin(len(state.candidate_event_ids))
        if name in by_bin:
            by_bin[name].append(state)
    trajectory_counts: Counter[str] = Counter()
    selected: list[Any] = []
    for name in DIAGNOSTIC_BINS:
        ordered = sorted(
            by_bin[name],
            key=lambda state: _state_order_key(state.state_id, salt=salt),
        )
        taken = 0
        for state in ordered:
            if taken == bin_targets[name]:
                break
            if trajectory_counts[state.trajectory_id] >= maximum_states_per_trajectory:
                continue
            selected.append(state)
            trajectory_counts[state.trajectory_id] += 1
            taken += 1
        if taken != bin_targets[name]:
            raise ValueError(
                f"trajectory cap cannot satisfy the {name} bin target"
            )
    rows = tuple(
        {
            "candidate_event_ids": list(state.candidate_event_ids),
            "history_bin": history_bin(len(state.candidate_event_ids)),
            "logical_shard": logical_shard_for_trajectory(
                state.trajectory_id, shard_count=256
            ),
            "role": state.role,
            "state_id": state.state_id,
            "trajectory_id": state.trajectory_id,
        }
        for state in sorted(selected, key=lambda state: state.state_id)
    )
    identities = tuple(row["state_id"] for row in rows)
    if len(identities) != len(set(identities)):
        raise RuntimeError("long-oracle selection produced duplicate states")
    return rows


def _state_rng(state_id: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{state_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="big"))


def recent_prefix(candidates: tuple[int, ...], budget: int) -> tuple[int, ...]:
    return tuple(sorted(candidates)[-budget:])


def wave_one_coalitions(
    *,
    state_id: str,
    candidate_event_ids: Sequence[int],
    random_subsets_per_budget: int,
    random_seed: int,
) -> tuple[dict[str, Any], ...]:
    """Empty anchor, every singleton, recent prefixes, random floors, full anchor."""
    candidates = _canonical_subset(
        sorted(candidate_event_ids), candidates=tuple(sorted(candidate_event_ids))
    )
    if len(candidates) <= max(BUDGETS):
        raise ValueError("long-oracle states must exceed the maximum budget")
    if type(random_subsets_per_budget) is not int or random_subsets_per_budget < 0:
        raise ValueError("random subset count must be a non-negative integer")
    coalitions: list[tuple[int, ...]] = []
    sources: list[str] = []

    def add(values: Sequence[int], source: str) -> None:
        subset = _canonical_subset(tuple(sorted(values)), candidates=candidates)
        if subset not in coalitions:
            coalitions.append(subset)
            sources.append(source)

    add((), "anchor_empty")
    for event in candidates:
        add((event,), "singleton_complete")
    for budget in BUDGETS:
        add(recent_prefix(candidates, budget), f"recent_b{budget}")
    rng = _state_rng(state_id, random_seed)
    for budget in BUDGETS:
        for _ in range(random_subsets_per_budget):
            add(sorted(rng.sample(candidates, budget)), f"random_b{budget}")
    add(candidates, "anchor_full")
    return tuple(
        {"event_ids": list(subset), "source": source}
        for subset, source in zip(coalitions, sources, strict=True)
    )


def _distance_lookup(
    distance_rows: Sequence[Mapping[str, Any]],
    *,
    candidates: tuple[int, ...],
) -> dict[tuple[int, ...], float]:
    result: dict[tuple[int, ...], float] = {}
    for row in distance_rows:
        subset = _canonical_subset(
            row["coalition_event_step_ids"], candidates=candidates
        )
        value = float(row["distance"])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("long-oracle distance must be finite and non-negative")
        if subset in result and not math.isclose(
            result[subset], value, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("long-oracle duplicate coalition distances drifted")
        result[subset] = value
    return result


def greedy_selection_from_distances(
    *,
    candidate_event_ids: Sequence[int],
    distance_rows: Sequence[Mapping[str, Any]],
    previous_selected: Sequence[int],
) -> tuple[int, ...]:
    """Pick the true-greedy expansion of ``previous_selected`` from labeled rows."""
    candidates = _canonical_subset(
        sorted(candidate_event_ids), candidates=tuple(sorted(candidate_event_ids))
    )
    previous = _canonical_subset(sorted(previous_selected), candidates=candidates)
    distances = _distance_lookup(distance_rows, candidates=candidates)
    expansions = tuple(
        tuple(sorted((*previous, event)))
        for event in candidates
        if event not in previous
    )
    missing = tuple(subset for subset in expansions if subset not in distances)
    if missing:
        raise ValueError(
            "labeled rows do not cover every one-event expansion of the prefix"
        )
    return min(expansions, key=lambda subset: (distances[subset], subset))


def additive_top_k(
    *,
    candidate_event_ids: Sequence[int],
    distance_rows: Sequence[Mapping[str, Any]],
    k: int,
) -> tuple[int, ...]:
    """Top-``k`` events ranked by singleton distance from complete wave-1 labels."""
    candidates = _canonical_subset(
        sorted(candidate_event_ids), candidates=tuple(sorted(candidate_event_ids))
    )
    if type(k) is not int or not 1 <= k <= len(candidates):
        raise ValueError("additive k must be a valid subset size")
    distances = _distance_lookup(distance_rows, candidates=candidates)
    singletons = tuple((event,) for event in candidates)
    missing = tuple(subset for subset in singletons if subset not in distances)
    if missing:
        raise ValueError("wave-one rows must cover every singleton")
    ranked = sorted(singletons, key=lambda subset: (distances[subset], subset))
    return tuple(sorted(subset[0] for subset in ranked[:k]))


def next_wave_coalitions(
    *,
    candidate_event_ids: Sequence[int],
    previous_selected: Sequence[int],
    additive_subset: Sequence[int] | None,
) -> tuple[dict[str, Any], ...]:
    """One-event expansions of the greedy prefix plus the additive comparison set."""
    candidates = _canonical_subset(
        sorted(candidate_event_ids), candidates=tuple(sorted(candidate_event_ids))
    )
    previous = _canonical_subset(sorted(previous_selected), candidates=candidates)
    if not previous or len(previous) >= max(BUDGETS):
        raise ValueError("greedy prefix size must be in [1, 3] for the next wave")
    coalitions: list[tuple[int, ...]] = []
    sources: list[str] = []

    def add(values: Sequence[int], source: str) -> None:
        subset = _canonical_subset(tuple(sorted(values)), candidates=candidates)
        if subset not in coalitions:
            coalitions.append(subset)
            sources.append(source)

    add((), "anchor_empty")
    step = len(previous) + 1
    for event in candidates:
        if event not in previous:
            add((*previous, event), f"greedy_step{step}_expansion")
    if additive_subset is not None:
        subset = _canonical_subset(sorted(additive_subset), candidates=candidates)
        if len(subset) != step:
            raise ValueError("additive subset size must match the wave step")
        add(subset, f"additive_top{step}")
    add(candidates, "anchor_full")
    return tuple(
        {"event_ids": list(subset), "source": source}
        for subset, source in zip(coalitions, sources, strict=True)
    )


def _recovery(
    *,
    empty_distance: float,
    distance: float,
) -> float:
    if not math.isfinite(empty_distance) or empty_distance <= 0.0:
        raise ValueError("empty-coalition distance must be finite and positive")
    return (empty_distance - distance) / empty_distance


def _trajectory_equal_mean(values: Mapping[str, list[float]]) -> float:
    if not values:
        raise ValueError("trajectory-equal mean requires at least one trajectory")
    return sum(
        sum(rows) / len(rows) for rows in values.values()
    ) / len(values)


def _paired_bootstrap(
    left_by_trajectory: Mapping[str, list[float]],
    right_by_trajectory: Mapping[str, list[float]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    trajectories = sorted(left_by_trajectory)
    if trajectories != sorted(right_by_trajectory):
        raise ValueError("paired bootstrap requires identical trajectory sets")
    deltas = {
        trajectory: (
            sum(left_by_trajectory[trajectory]) / len(left_by_trajectory[trajectory])
            - sum(right_by_trajectory[trajectory])
            / len(right_by_trajectory[trajectory])
        )
        for trajectory in trajectories
    }
    point = sum(deltas.values()) / len(deltas)
    rng = random.Random(seed)
    resampled = []
    for _ in range(resamples):
        chosen = [deltas[rng.choice(trajectories)] for _ in trajectories]
        resampled.append(sum(chosen) / len(chosen))
    resampled.sort()

    def quantile(fraction: float) -> float:
        index = min(
            len(resampled) - 1, max(0, round(fraction * (len(resampled) - 1)))
        )
        return resampled[index]

    return {
        "point_estimate": point,
        "lower_95": quantile(0.025),
        "upper_95": quantile(0.975),
    }


def reduce_long_oracle_metrics(
    *,
    states: Sequence[Mapping[str, Any]],
    wave_terminals: Mapping[int, Mapping[str, Mapping[str, Any]]],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    random_subsets_per_budget: int,
    random_seed: int,
) -> dict[str, Any]:
    """Aggregate oracle-greedy/recent/additive/random recovery over completed states."""
    if set(wave_terminals) != set(WAVES):
        raise ValueError("reducer requires terminals for waves one through four")
    completed: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for state in states:
        state_id = state["state_id"]
        candidates = _canonical_subset(
            sorted(state["candidate_event_ids"]),
            candidates=tuple(sorted(state["candidate_event_ids"])),
        )
        terminals = []
        skip_reason = None
        for wave in WAVES:
            terminal = wave_terminals[wave].get(state_id)
            if terminal is None:
                skip_reason = f"missing_wave_{wave}"
                break
            if terminal.get("status") != "COMPLETED_VARIABLE_HISTORY_LABEL_STATE":
                skip_reason = (
                    f"wave_{wave}_{terminal.get('failure_class', 'skipped')}"
                )
                break
            terminals.append(terminal)
        if skip_reason is not None:
            skipped[state_id] = skip_reason
            continue
        references = {
            terminal["reference"]["serialized_action"] for terminal in terminals
        }
        if len(references) != 1:
            skipped[state_id] = "reference_action_drifted_across_waves"
            continue
        merged_rows = [
            row for terminal in terminals for row in terminal["distance_rows"]
        ]
        try:
            distances = _distance_lookup(merged_rows, candidates=candidates)
        except ValueError:
            skipped[state_id] = "cross_wave_distance_drift"
            continue
        empty = distances.get(())
        if empty is None:
            raise ValueError(f"{state_id} is missing the empty-coalition anchor")
        if empty <= 0.0:
            skipped[state_id] = "empty_distance_not_positive"
            continue

        greedy_prefixes: list[tuple[int, ...]] = []
        previous: tuple[int, ...] = ()
        for wave in WAVES:
            rows = terminals[wave - 1]["distance_rows"]
            previous = greedy_selection_from_distances(
                candidate_event_ids=candidates,
                distance_rows=rows,
                previous_selected=previous,
            )
            greedy_prefixes.append(previous)

        def recovery_of(subset: tuple[int, ...]) -> float:
            if subset not in distances:
                raise ValueError(
                    f"{state_id} is missing a labeled comparison subset"
                )
            return _recovery(empty_distance=empty, distance=distances[subset])

        rng = _state_rng(state_id, random_seed)
        random_subsets: dict[int, list[tuple[int, ...]]] = {}
        for budget in BUDGETS:
            random_subsets[budget] = [
                tuple(sorted(rng.sample(candidates, budget)))
                for _ in range(random_subsets_per_budget)
            ]
        methods: dict[str, dict[int, float]] = {
            "oracle_greedy": {},
            "recent": {},
            "additive": {},
            "random": {},
        }
        for budget in BUDGETS:
            methods["oracle_greedy"][budget] = max(
                recovery_of(prefix) for prefix in greedy_prefixes[:budget]
            )
            methods["recent"][budget] = recovery_of(
                recent_prefix(candidates, budget)
            )
            additive_subset = additive_top_k(
                candidate_event_ids=candidates,
                distance_rows=terminals[0]["distance_rows"],
                k=budget,
            )
            methods["additive"][budget] = recovery_of(additive_subset)
            random_values = [
                recovery_of(subset) for subset in random_subsets[budget]
            ]
            methods["random"][budget] = sum(random_values) / len(random_values)
        completed.append(
            {
                "history_bin": state["history_bin"],
                "methods": methods,
                "state_id": state_id,
                "trajectory_id": state["trajectory_id"],
                "best_singleton": greedy_prefixes[0],
                "best_singleton_in_recent_4": bool(
                    set(greedy_prefixes[0]).issubset(recent_prefix(candidates, 4))
                ),
                "best_singleton_age_fraction": (
                    (max(candidates) - greedy_prefixes[0][0]) / max(candidates)
                ),
            }
        )
    if not completed:
        raise ValueError("no state completed all four waves")

    def by_trajectory(
        method: str, budgets: Sequence[int], *, bins: frozenset[str] | None = None
    ) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        for row in completed:
            if bins is not None and row["history_bin"] not in bins:
                continue
            values = [row["methods"][method][budget] for budget in budgets]
            result.setdefault(row["trajectory_id"], []).append(
                sum(values) / len(values)
            )
        return result

    summary_methods: dict[str, Any] = {}
    for method in ("oracle_greedy", "recent", "additive", "random"):
        per_budget = {
            f"B{budget}": _trajectory_equal_mean(by_trajectory(method, (budget,)))
            for budget in BUDGETS
        }
        summary_methods[method] = {
            **per_budget,
            "macro_B1_B4": _trajectory_equal_mean(by_trajectory(method, BUDGETS)),
            "very_long_macro": (
                _trajectory_equal_mean(
                    by_trajectory(method, BUDGETS, bins=frozenset(("very_long",)))
                )
                if any(row["history_bin"] == "very_long" for row in completed)
                else None
            ),
        }
    comparisons = {
        "oracle_greedy_minus_recent_macro": _paired_bootstrap(
            by_trajectory("oracle_greedy", BUDGETS),
            by_trajectory("recent", BUDGETS),
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "additive_minus_recent_macro": _paired_bootstrap(
            by_trajectory("additive", BUDGETS),
            by_trajectory("recent", BUDGETS),
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
    }
    outside_recent = [
        row for row in completed if not row["best_singleton_in_recent_4"]
    ]
    return {
        "comparisons": comparisons,
        "completed_state_count": len(completed),
        "completed_states": {
            row["state_id"]: {
                "best_singleton": list(row["best_singleton"]),
                "history_bin": row["history_bin"],
                "methods": {
                    method: {
                        f"B{budget}": row["methods"][method][budget]
                        for budget in BUDGETS
                    }
                    for method in ("oracle_greedy", "recent", "additive")
                },
            }
            for row in completed
        },
        "best_singleton_outside_recent4_fraction": len(outside_recent)
        / len(completed),
        "best_singleton_age_fraction_mean": sum(
            row["best_singleton_age_fraction"] for row in completed
        )
        / len(completed),
        "methods": summary_methods,
        "skipped_states": dict(sorted(skipped.items())),
        "trajectory_count": len({row["trajectory_id"] for row in completed}),
    }


__all__ = [
    "BUDGETS",
    "DIAGNOSTIC_BINS",
    "SELECTION_STATUS",
    "WAVES",
    "additive_top_k",
    "greedy_selection_from_distances",
    "next_wave_coalitions",
    "recent_prefix",
    "reduce_long_oracle_metrics",
    "select_long_oracle_states",
    "wave_one_coalitions",
]
