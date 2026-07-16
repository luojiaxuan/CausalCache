"""Deterministic scientific summaries from validated v2.2 label evidence."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.restoration_v2_2_eager_artifact import strict_json_object_bytes
from causalcache.restoration_v2_2_label_artifact import (
    SCHEMA_VERSION,
    LabelEvidence,
    read_label_evidence_archive,
)
from causalcache.restoration_v2_2_label_contract import PROTOCOL_ID
from causalcache.restoration_v2_2_label_table import (
    PRIMARY_BUDGET_EVENT_CAPACITY,
    ValidatedDistanceTable,
    deployment_conditional_edges,
    pair_interactions,
    primary_exact_subset_oracle,
    validate_complete_distance_table,
)


SUMMARY_STATUS = "SUMMARIZED_RESTORATION_V2_2_EAGER_LABEL_SCIENCE"
SIGN_RULE = "strict_float_comparison_to_zero"
NORMALIZATION_EPSILON = 1e-12
_STATE_MEMBER = re.compile(r"workers/(?:even|odd)/states/[0-9]{3}\.json")


@dataclass(frozen=True)
class _StateMeasures:
    role: str
    table: ValidatedDistanceTable

    @property
    def raw_utilities(self) -> tuple[float, ...]:
        return tuple(row.utility for row in self.table.rows)

    @property
    def deployment_marginals(self) -> tuple[float, ...]:
        return tuple(
            edge.marginal_gain for edge in deployment_conditional_edges(self.table)
        )

    @property
    def interactions(self) -> tuple[float, ...]:
        return tuple(item.interaction for item in pair_interactions(self.table))


def _finite_sum(values: Sequence[float]) -> float:
    total = math.fsum(values)
    return 0.0 if total == 0.0 else total


def _numeric_histogram(values: Sequence[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(key): counts[key] for key in sorted(counts)}


def _sign_counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "positive_count": sum(value > 0.0 for value in values),
        "negative_count": sum(value < 0.0 for value in values),
        "zero_count": sum(value == 0.0 for value in values),
    }


def _raw_utility_summary(states: Sequence[_StateMeasures]) -> dict[str, Any]:
    values_by_state = [state.raw_utilities for state in states]
    values = tuple(value for state_values in values_by_state for value in state_values)
    return {
        "row_count": len(values),
        **_sign_counts(values),
        "mean_utility": _finite_sum(values) / len(values),
        "states_with_positive_utility_count": sum(
            any(value > 0.0 for value in state_values)
            for state_values in values_by_state
        ),
        "states_with_negative_utility_count": sum(
            any(value < 0.0 for value in state_values)
            for state_values in values_by_state
        ),
    }


def _deployment_marginal_summary(
    states: Sequence[_StateMeasures],
) -> dict[str, Any]:
    values_by_state = [state.deployment_marginals for state in states]
    values = tuple(value for state_values in values_by_state for value in state_values)
    return {
        "edge_count": len(values),
        **_sign_counts(values),
        "mean_marginal_gain": _finite_sum(values) / len(values),
        "states_with_positive_marginal_count": sum(
            any(value > 0.0 for value in state_values)
            for state_values in values_by_state
        ),
        "states_with_negative_marginal_count": sum(
            any(value < 0.0 for value in state_values)
            for state_values in values_by_state
        ),
    }


def _oracle_summary(states: Sequence[_StateMeasures]) -> dict[str, Any]:
    oracles = [primary_exact_subset_oracle(state.table) for state in states]
    eligible_by_cardinality: Counter[int] = Counter()
    for state in states:
        for row in state.table.rows:
            if len(row.coalition) <= PRIMARY_BUDGET_EVENT_CAPACITY:
                eligible_by_cardinality[len(row.coalition)] += 1
    utilities = tuple(oracle.utility for oracle in oracles)
    baseline_distances = tuple(state.table.distance(()) for state in states)
    normalized_recoveries = tuple(
        utility / baseline
        for utility, baseline in zip(utilities, baseline_distances, strict=True)
        if baseline > NORMALIZATION_EPSILON
    )
    return {
        "state_count": len(states),
        "budget_event_capacity": PRIMARY_BUDGET_EVENT_CAPACITY,
        "selected_cardinality_histogram": _numeric_histogram(
            [len(oracle.coalition) for oracle in oracles]
        ),
        "eligible_coalition_count": sum(
            oracle.evaluated_coalition_count for oracle in oracles
        ),
        "eligible_coalition_count_histogram": _numeric_histogram(
            [oracle.evaluated_coalition_count for oracle in oracles]
        ),
        "eligible_coalition_cardinality_histogram": {
            str(key): eligible_by_cardinality[key]
            for key in sorted(eligible_by_cardinality)
        },
        "positive_utility_state_count": sum(value > 0.0 for value in utilities),
        "negative_utility_state_count": sum(value < 0.0 for value in utilities),
        "zero_utility_state_count": sum(value == 0.0 for value in utilities),
        "mean_summary_only_distance": _finite_sum(baseline_distances) / len(states),
        "mean_utility": _finite_sum(utilities) / len(states),
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "normalized_recovery_eligible_state_count": len(normalized_recoveries),
        "normalized_recovery_excluded_state_count": (
            len(states) - len(normalized_recoveries)
        ),
        "mean_normalized_recovery": (
            _finite_sum(normalized_recoveries) / len(normalized_recoveries)
            if normalized_recoveries
            else None
        ),
    }


def _interaction_summary(states: Sequence[_StateMeasures]) -> dict[str, Any]:
    values_by_state = [state.interactions for state in states]
    values = tuple(value for state_values in values_by_state for value in state_values)
    absolute_values = tuple(abs(value) for value in values)
    state_absolute_masses = tuple(
        _finite_sum(tuple(abs(value) for value in state_values))
        for state_values in values_by_state
    )
    positive_values = tuple(value for value in values if value > 0.0)
    negative_absolute_values = tuple(-value for value in values if value < 0.0)
    return {
        "row_count": len(values),
        **_sign_counts(values),
        "states_with_nonzero_interaction_count": sum(
            any(value != 0.0 for value in state_values)
            for state_values in values_by_state
        ),
        "signed_mass": _finite_sum(values),
        "positive_mass": _finite_sum(positive_values),
        "negative_absolute_mass": _finite_sum(negative_absolute_values),
        "absolute_mass": _finite_sum(absolute_values),
        "mean_absolute_magnitude": (
            _finite_sum(absolute_values) / len(absolute_values)
            if absolute_values
            else 0.0
        ),
        "maximum_absolute_magnitude": max(absolute_values, default=0.0),
        "mean_state_absolute_mass": (
            _finite_sum(state_absolute_masses) / len(state_absolute_masses)
            if state_absolute_masses
            else 0.0
        ),
        "maximum_state_absolute_mass": max(state_absolute_masses, default=0.0),
    }


def _group_summary(states: Sequence[_StateMeasures]) -> dict[str, Any]:
    if not states:
        raise ValueError("scientific summary groups cannot be empty")
    return {
        "state_count": len(states),
        "candidate_event_count_histogram": _numeric_histogram(
            [len(state.table.event_ids) for state in states]
        ),
        "raw_restoration_utility": _raw_utility_summary(states),
        "deployment_conditional_marginal": _deployment_marginal_summary(states),
        "primary_exact_subset_oracle": _oracle_summary(states),
        "pair_interaction": _interaction_summary(states),
    }


def _state_measures(evidence: LabelEvidence) -> tuple[_StateMeasures, ...]:
    names = sorted(name for name in evidence.files if _STATE_MEMBER.fullmatch(name))
    if not names:
        raise ValueError("validated label evidence contains no raw state records")
    measures = []
    for name in names:
        record = strict_json_object_bytes(evidence.files[name], label=name)
        state = record.get("state")
        rows = record.get("distance_rows")
        if not isinstance(state, Mapping) or not isinstance(rows, list):
            raise ValueError(f"{name} lacks raw state distance rows")
        role = state.get("role")
        event_ids = state.get("candidate_event_step_ids")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{name} has an invalid role")
        if not isinstance(event_ids, list):
            raise ValueError(f"{name} has invalid candidate event ids")
        distances: dict[tuple[int, ...], Any] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"{name} contains a non-object distance row")
            coalition = row.get("coalition_event_step_ids")
            if not isinstance(coalition, list):
                raise ValueError(f"{name} contains invalid coalition ids")
            key = tuple(coalition)
            if key in distances:
                raise ValueError(f"{name} contains duplicate coalition rows")
            distances[key] = row.get("distance_kl")
        table = validate_complete_distance_table(tuple(event_ids), distances)
        measures.append(_StateMeasures(role=role, table=table))
    return tuple(measures)


def build_label_scientific_summary(evidence: LabelEvidence) -> dict[str, Any]:
    """Recompute compact statistics from a validated archive's raw distances."""
    if not isinstance(evidence, LabelEvidence):
        raise TypeError("evidence must be validated LabelEvidence")
    states = _state_measures(evidence)
    roles = sorted({state.role for state in states})
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "attempt_id": evidence.profile.attempt_id,
        "attempt_revision": evidence.profile.attempt_revision,
        "status": SUMMARY_STATUS,
        "outcome": evidence.outcome,
        "sign_rule": SIGN_RULE,
        "source_evidence": {
            "source_git_commit": evidence.source_git_commit,
            "run_contract_sha256": evidence.run_contract_sha256,
            "tree_inventory_sha256": evidence.tree_inventory_sha256,
            "archive_validation": "read_label_evidence_archive_passed",
        },
        "overall": _group_summary(states),
        "by_role": {
            role: _group_summary(
                tuple(state for state in states if state.role == role)
            )
            for role in roles
        },
    }


def summarize_label_evidence_archive(path: str | Path) -> dict[str, Any]:
    """Validate a canonical archive before reducing its raw distance rows."""
    return build_label_scientific_summary(read_label_evidence_archive(path))
