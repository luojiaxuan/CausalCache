"""Independent stdlib-only math audit for expansion restoration labels.

The input is a non-empty sequence of normalized raw state records.  Each record
must contain a ``state`` mapping with ``state_id``, ``source_id``, ``role``, and
strictly increasing ``candidate_event_step_ids`` (two through four events), plus
``distance_rows`` containing one ``coalition``/``distance`` mapping for every
member of the candidate power set.  Extra raw evidence fields are ignored.

This module deliberately does not import the production reducer.  It rebuilds
the mathematical projection from raw D(S) scalars and can compare that
projection with a reducer result without inheriting reducer implementation
state.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


PRIMARY_BUDGET_EVENT_CAPACITY = 2
ALGEBRA_RESIDUAL_TOLERANCE = 1e-12
PASS_STATUS = "PASS_INDEPENDENT_EXPANSION_MATH_AUDIT"

_STATE_MATH_FIELDS = (
    "state",
    "baseline_summary_only_distance",
    "full_history_distance",
    "deployment_conditional_edges",
    "full_hypercube_edges",
    "pair_interactions",
    "exact_permutation_attribution",
    "primary_exact_subset_oracle",
    "telescoping_max_abs_residual",
    "shapley_efficiency_abs_residual",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _candidate_event_ids(value: Any, state_index: int) -> tuple[int, ...]:
    raw = tuple(_sequence(value, f"state {state_index} candidate event ids"))
    if len(raw) not in {2, 3, 4}:
        raise ValueError(f"state {state_index} must have exactly two, three, or four candidates")
    if any(type(event_id) is not int or event_id <= 0 for event_id in raw):
        raise TypeError(f"state {state_index} candidate event ids must be positive integers")
    if raw != tuple(sorted(raw)) or len(set(raw)) != len(raw):
        raise ValueError(f"state {state_index} candidate event ids must be strictly increasing")
    return raw


def _coalitions(event_ids: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        coalition
        for size in range(len(event_ids) + 1)
        for coalition in itertools.combinations(event_ids, size)
    )


def _normalize_state_record(
    raw_record: Any,
    state_index: int,
) -> tuple[dict[str, Any], tuple[int, ...], dict[tuple[int, ...], float]]:
    record = _mapping(raw_record, f"state record {state_index}")
    state = dict(_mapping(record.get("state"), f"state record {state_index} identity"))
    for field in ("state_id", "source_id", "role"):
        if not isinstance(state.get(field), str) or not state[field]:
            raise ValueError(f"state {state_index} {field} must be a non-empty string")
    event_ids = _candidate_event_ids(state.get("candidate_event_step_ids"), state_index)
    expected = _coalitions(event_ids)
    allowed = frozenset(event_ids)
    rows = _sequence(record.get("distance_rows"), f"state {state_index} distance rows")
    if len(rows) != len(expected):
        raise ValueError(f"state {state_index} distance rows must contain the complete power set")

    distances: dict[tuple[int, ...], float] = {}
    for row_index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"state {state_index} distance row {row_index}")
        coalition_raw = tuple(
            _sequence(row.get("coalition"), f"state {state_index} coalition {row_index}")
        )
        if any(type(event_id) is not int for event_id in coalition_raw):
            raise TypeError(f"state {state_index} coalitions must contain only integer event ids")
        if (
            coalition_raw != tuple(sorted(coalition_raw))
            or len(set(coalition_raw)) != len(coalition_raw)
            or not set(coalition_raw).issubset(allowed)
        ):
            raise ValueError(f"state {state_index} contains a noncanonical coalition")
        if coalition_raw in distances:
            raise ValueError(f"state {state_index} contains a duplicate coalition")
        distance_raw = row.get("distance")
        if isinstance(distance_raw, bool) or not isinstance(distance_raw, Real):
            raise TypeError(f"state {state_index} distances must be real numbers")
        distance = float(distance_raw)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError(f"state {state_index} distances must be finite and non-negative")
        distances[coalition_raw] = distance

    if frozenset(distances) != frozenset(expected):
        raise ValueError(f"state {state_index} distance rows do not cover the complete power set")
    distances = {coalition: distances[coalition] for coalition in expected}
    if distances[event_ids] != 0.0:
        raise ValueError(f"state {state_index} full-history distance must be canonical zero")
    return state, event_ids, distances


def _edges(
    event_ids: tuple[int, ...],
    distances: Mapping[tuple[int, ...], float],
    maximum_restored_events: int,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for base in _coalitions(event_ids):
        if len(base) >= maximum_restored_events:
            continue
        for event_id in event_ids:
            if event_id in base:
                continue
            restored = tuple(sorted((*base, event_id)))
            if len(restored) > maximum_restored_events:
                continue
            edges.append(
                {
                    "base_coalition": list(base),
                    "event_id": event_id,
                    "restored_coalition": list(restored),
                    "base_distance": distances[base],
                    "restored_distance": distances[restored],
                    "marginal_gain": distances[base] - distances[restored],
                }
            )
    return edges


def _interactions(
    event_ids: tuple[int, ...],
    distances: Mapping[tuple[int, ...], float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for left, right in itertools.combinations(event_ids, 2):
        remaining = tuple(event_id for event_id in event_ids if event_id not in {left, right})
        for conditioning in _coalitions(remaining):
            with_left = tuple(sorted((*conditioning, left)))
            with_right = tuple(sorted((*conditioning, right)))
            with_pair = tuple(sorted((*conditioning, left, right)))
            result.append(
                {
                    "conditioning_coalition": list(conditioning),
                    "left_event_id": left,
                    "right_event_id": right,
                    "interaction": (
                        distances[with_left]
                        + distances[with_right]
                        - distances[conditioning]
                        - distances[with_pair]
                    ),
                }
            )
    return result


def _attributions(
    event_ids: tuple[int, ...],
    distances: Mapping[tuple[int, ...], float],
) -> list[dict[str, Any]]:
    samples: dict[int, list[float]] = {event_id: [] for event_id in event_ids}
    permutations = tuple(itertools.permutations(event_ids))
    for permutation in permutations:
        prefix: tuple[int, ...] = ()
        for event_id in permutation:
            restored = tuple(sorted((*prefix, event_id)))
            samples[event_id].append(distances[prefix] - distances[restored])
            prefix = restored
    return [
        {
            "event_id": event_id,
            "mean_marginal_gain": math.fsum(samples[event_id]) / len(permutations),
            "permutation_count": len(permutations),
            "marginal_samples": samples[event_id],
        }
        for event_id in event_ids
    ]


def _oracle(
    event_ids: tuple[int, ...],
    distances: Mapping[tuple[int, ...], float],
) -> dict[str, Any]:
    feasible = [
        coalition
        for coalition in _coalitions(event_ids)
        if len(coalition) <= PRIMARY_BUDGET_EVENT_CAPACITY
    ]
    selected = min(
        feasible,
        key=lambda coalition: (distances[coalition], len(coalition), coalition),
    )
    return {
        "budget_event_capacity": PRIMARY_BUDGET_EVENT_CAPACITY,
        "tie_epsilon": 0.0,
        "coalition": list(selected),
        "distance": distances[selected],
        "utility": distances[()] - distances[selected],
        "evaluated_coalition_count": len(feasible),
    }


def _algebra_residuals(
    event_ids: tuple[int, ...],
    distances: Mapping[tuple[int, ...], float],
    attributions: Sequence[Mapping[str, Any]],
) -> tuple[float, float]:
    total = distances[()] - distances[event_ids]
    telescoping: list[float] = []
    for permutation in itertools.permutations(event_ids):
        prefix: tuple[int, ...] = ()
        gains: list[float] = []
        for event_id in permutation:
            restored = tuple(sorted((*prefix, event_id)))
            gains.append(distances[prefix] - distances[restored])
            prefix = restored
        telescoping.append(abs(math.fsum(gains) - total))
    shapley = abs(
        math.fsum(float(row["mean_marginal_gain"]) for row in attributions) - total
    )
    return max(telescoping), shapley


def _derive_state(raw_record: Any, state_index: int) -> dict[str, Any]:
    state, event_ids, distances = _normalize_state_record(raw_record, state_index)
    attributions = _attributions(event_ids, distances)
    telescoping, shapley = _algebra_residuals(event_ids, distances, attributions)
    if telescoping > ALGEBRA_RESIDUAL_TOLERANCE or shapley > ALGEBRA_RESIDUAL_TOLERANCE:
        raise ValueError(
            f"state {state_index} restoration algebra residual exceeds tolerance"
        )
    return {
        "state": state,
        "baseline_summary_only_distance": distances[()],
        "full_history_distance": distances[event_ids],
        "deployment_conditional_edges": _edges(
            event_ids, distances, PRIMARY_BUDGET_EVENT_CAPACITY
        ),
        "full_hypercube_edges": _edges(event_ids, distances, len(event_ids)),
        "pair_interactions": _interactions(event_ids, distances),
        "exact_permutation_attribution": attributions,
        "primary_exact_subset_oracle": _oracle(event_ids, distances),
        "telescoping_max_abs_residual": telescoping,
        "shapley_efficiency_abs_residual": shapley,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def audit_normalized_raw_states(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute the expansion label math solely from normalized raw D(S)."""
    raw_records = _sequence(records, "normalized raw state records")
    if not raw_records:
        raise ValueError("normalized raw state records cannot be empty")
    states = [_derive_state(record, index) for index, record in enumerate(raw_records)]
    state_ids = [state["state"]["state_id"] for state in states]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("normalized raw state records contain duplicate state ids")

    deployment = [edge for state in states for edge in state["deployment_conditional_edges"]]
    full = [edge for state in states for edge in state["full_hypercube_edges"]]
    interactions = [item for state in states for item in state["pair_interactions"]]
    attributions = [item for state in states for item in state["exact_permutation_attribution"]]
    raw_row_count = sum(
        len(_sequence(record["distance_rows"], "distance rows"))
        for record in raw_records
    )
    counts = {
        "trajectory_count": len({state["state"]["source_id"] for state in states}),
        "state_count": len(states),
        "raw_distance_row_count": raw_row_count,
        "deployment_conditional_edge_count": len(deployment),
        "full_hypercube_edge_count": len(full),
        "pair_interaction_count": len(interactions),
        "exact_permutation_attribution_count": len(attributions),
        "primary_exact_subset_oracle_count": len(states),
    }
    marginal_values = [float(edge["marginal_gain"]) for edge in full]
    interaction_values = [float(item["interaction"]) for item in interactions]
    summary = {
        "role_state_counts": dict(
            sorted(Counter(state["state"]["role"] for state in states).items())
        ),
        "marginal_sign_counts": {
            "negative": sum(value < 0 for value in marginal_values),
            "zero": sum(value == 0 for value in marginal_values),
            "positive": sum(value > 0 for value in marginal_values),
        },
        "interaction_sign_counts": {
            "negative": sum(value < 0 for value in interaction_values),
            "zero": sum(value == 0 for value in interaction_values),
            "positive": sum(value > 0 for value in interaction_values),
        },
        "nonmonotone_state_count": sum(
            any(float(edge["marginal_gain"]) < 0 for edge in state["full_hypercube_edges"])
            for state in states
        ),
        "mean_summary_only_distance": math.fsum(
            float(state["baseline_summary_only_distance"]) for state in states
        )
        / len(states),
        "mean_primary_exact_oracle_utility": math.fsum(
            float(state["primary_exact_subset_oracle"]["utility"]) for state in states
        )
        / len(states),
        "algebra_residual_gate": {
            "tolerance": ALGEBRA_RESIDUAL_TOLERANCE,
            "maximum_telescoping_abs_residual": max(
                float(state["telescoping_max_abs_residual"]) for state in states
            ),
            "maximum_shapley_efficiency_abs_residual": max(
                float(state["shapley_efficiency_abs_residual"]) for state in states
            ),
            "passed": True,
        },
    }
    payload = {"counts": counts, "summary": summary, "states": states}
    return {
        **payload,
        "math_projection_sha256": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
    }


def _first_mismatch(left: Any, right: Any, path: str) -> str | None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return f"{path} keys differ: audit={sorted(left)}, reducer={sorted(right)}"
        for key in sorted(left):
            mismatch = _first_mismatch(left[key], right[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path} length differs: audit={len(left)}, reducer={len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            mismatch = _first_mismatch(left_item, right_item, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    if type(left) is not type(right) or left != right:
        return f"{path} differs: audit={left!r}, reducer={right!r}"
    return None


def compare_with_reducer_output(
    audit_result: Mapping[str, Any],
    reducer_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless all independently audited math matches the reducer."""
    audit = _mapping(audit_result, "audit result")
    reducer = _mapping(reducer_result, "reducer result")
    audit_counts = dict(_mapping(audit.get("counts"), "audit counts"))
    reducer_counts = _mapping(reducer.get("counts"), "reducer counts")
    reducer_count_projection = {key: reducer_counts.get(key) for key in audit_counts}

    audit_summary = dict(_mapping(audit.get("summary"), "audit summary"))
    reducer_summary = _mapping(reducer.get("summary"), "reducer summary")
    reducer_summary_projection = {key: reducer_summary.get(key) for key in audit_summary}

    audit_states = list(_sequence(audit.get("states"), "audit states"))
    raw_reducer_states = _sequence(reducer.get("states"), "reducer states")
    reducer_states: list[dict[str, Any]] = []
    for index, raw_state in enumerate(raw_reducer_states):
        state = _mapping(raw_state, f"reducer state {index}")
        missing = [field for field in _STATE_MATH_FIELDS if field not in state]
        if missing:
            raise ValueError(f"reducer state {index} is missing math fields: {missing}")
        reducer_states.append({field: state[field] for field in _STATE_MATH_FIELDS})

    projection = {
        "counts": audit_counts,
        "summary": audit_summary,
        "states": audit_states,
    }
    reducer_projection = {
        "counts": reducer_count_projection,
        "summary": reducer_summary_projection,
        "states": reducer_states,
    }
    mismatch = _first_mismatch(projection, reducer_projection, "math_projection")
    if mismatch is not None:
        raise ValueError(f"independent expansion math audit disagrees with reducer: {mismatch}")
    projection_sha256 = hashlib.sha256(_canonical_json_bytes(projection)).hexdigest()
    expected_sha256 = audit.get("math_projection_sha256")
    if expected_sha256 != projection_sha256:
        raise ValueError("audit math projection SHA256 does not match its payload")
    return {
        "status": PASS_STATUS,
        "compared_state_count": len(audit_states),
        "math_projection_sha256": projection_sha256,
    }


__all__ = [
    "ALGEBRA_RESIDUAL_TOLERANCE",
    "PASS_STATUS",
    "PRIMARY_BUDGET_EVENT_CAPACITY",
    "audit_normalized_raw_states",
    "compare_with_reducer_output",
]
