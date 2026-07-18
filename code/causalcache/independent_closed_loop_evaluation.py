"""Frozen closed-loop and matched-NLL aggregation for the independent gate."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


INDEPENDENT_ARM = "independent_B2"
RECENT_ARM = "recent_B2"
PRIMARY_ARMS = (INDEPENDENT_ARM, RECENT_ARM)
PARTITION_INSTANCE_INDICES = {"train": (0,), "test": (0, 1, 2)}
PARTITION_TEMPLATE_COUNTS = {"train": 60, "test": 25}

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 271_828
BOOTSTRAP_CONFIDENCE = 0.90
PRIMARY_CALIPER = 0.05
SENSITIVITY_CALIPERS = (0.02, 0.10)
RESTORATION_TIE_EPSILON = 1e-12
MINIMUM_MATCHED_TEST_TEMPLATES = 15


def _ordered_sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be an ordered sequence")
    return tuple(value)


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _binary(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if result not in {0.0, 1.0}:
        raise ValueError(f"{label} must be binary")
    return result


def _strict_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a bool")
    return value


def _partition_spec(
    partition: str,
    template_order: Sequence[str],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if partition not in PARTITION_INSTANCE_INDICES:
        raise ValueError("partition must be train or test")
    templates = _ordered_sequence(template_order, label="template_order")
    if (
        len(templates) != PARTITION_TEMPLATE_COUNTS[partition]
        or any(not isinstance(item, str) or not item for item in templates)
        or len(set(templates)) != len(templates)
    ):
        raise ValueError("template_order differs from the frozen partition inventory")
    return templates, PARTITION_INSTANCE_INDICES[partition]


def type7_quantile(values: Sequence[float], probability: float) -> float:
    """Return the Hyndman--Fan type-7 sample quantile."""
    ordered = tuple(sorted(_finite(item, label="quantile value") for item in values))
    if not ordered or isinstance(probability, bool) or not 0.0 <= probability <= 1.0:
        raise ValueError("type-7 quantile input is invalid")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def template_cluster_bootstrap_lower(
    template_values: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> float:
    """Resample complete template records with the frozen stdlib RNG."""
    values = tuple(_finite(item, label="template value") for item in template_values)
    if not values or type(resamples) is not int or resamples <= 0:
        raise ValueError("bootstrap values or resample count are invalid")
    if type(seed) is not int:
        raise TypeError("bootstrap seed must be an int")
    confidence = _finite(confidence, label="bootstrap confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie strictly between zero and one")
    generator = random.Random(seed)
    count = len(values)
    means = tuple(
        math.fsum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    return type7_quantile(means, (1.0 - confidence) / 2.0)


def _episode_index(
    records: Sequence[Mapping[str, Any]],
    *,
    partition: str,
    templates: Sequence[str],
    instance_indices: Sequence[int],
    require_state_count: bool,
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    rows = _ordered_sequence(records, label="episode records")
    expected = {
        (task_type, task_index, arm)
        for task_type in templates
        for task_index in instance_indices
        for arm in PRIMARY_ARMS
    }
    indexed: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("episode records must contain mappings")
        if row.get("partition") != partition:
            raise ValueError("episode partition differs from the requested partition")
        task_type = row.get("task_type")
        task_index = row.get("task_index")
        arm = row.get("arm")
        if (
            not isinstance(task_type, str)
            or type(task_index) is not int
            or arm not in PRIMARY_ARMS
        ):
            raise ValueError("episode unit identity is malformed")
        key = (task_type, task_index, arm)
        if key in indexed:
            raise ValueError(f"duplicate episode unit: {key}")
        _binary(
            row.get("official_terminal_success"),
            label=f"episode {ordinal} official_terminal_success",
        )
        if require_state_count:
            state_count = row.get("policy_decision_state_count")
            if type(state_count) is not int or state_count < 0:
                raise ValueError("policy_decision_state_count must be a nonnegative int")
        indexed[key] = row
    missing = expected - set(indexed)
    unexpected = set(indexed) - expected
    if missing or unexpected:
        raise ValueError(
            f"episode unit inventory drifted: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return indexed


def evaluate_closed_loop_partition(
    episode_records: Sequence[Mapping[str, Any]],
    *,
    partition: str,
    template_order: Sequence[str],
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Compute the fixed equal-instance, equal-template closed-loop ITT effect."""
    templates, instance_indices = _partition_spec(partition, template_order)
    episodes = _episode_index(
        episode_records,
        partition=partition,
        templates=templates,
        instance_indices=instance_indices,
        require_state_count=False,
    )
    instance_records = []
    template_records = []
    for task_type in templates:
        effects = []
        for task_index in instance_indices:
            independent = _binary(
                episodes[(task_type, task_index, INDEPENDENT_ARM)][
                    "official_terminal_success"
                ],
                label="independent terminal success",
            )
            recent = _binary(
                episodes[(task_type, task_index, RECENT_ARM)][
                    "official_terminal_success"
                ],
                label="recent terminal success",
            )
            effect = independent - recent
            effects.append(effect)
            instance_records.append(
                {
                    "task_type": task_type,
                    "task_index": task_index,
                    "independent_success": independent,
                    "recent_success": recent,
                    "paired_effect": effect,
                }
            )
        template_effect = math.fsum(effects) / len(effects)
        template_records.append(
            {
                "task_type": task_type,
                "instance_pair_count": len(effects),
                "paired_effect": template_effect,
            }
        )
    template_effects = tuple(item["paired_effect"] for item in template_records)
    point_estimate = math.fsum(template_effects) / len(template_effects)
    return {
        "partition": partition,
        "template_count": len(templates),
        "instance_indices": list(instance_indices),
        "paired_instance_count": len(instance_records),
        "primary_arm_episode_count": len(episodes),
        "instance_records": instance_records,
        "template_records": template_records,
        "paired_effect": point_estimate,
        "bootstrap": {
            "unit": "template_cluster",
            "resamples": bootstrap_resamples,
            "seed": BOOTSTRAP_SEED,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "quantile": "Hyndman_Fan_type_7",
            "lower": template_cluster_bootstrap_lower(
                template_effects,
                resamples=bootstrap_resamples,
            ),
        },
    }


def _state_index(
    records: Sequence[Mapping[str, Any]],
    *,
    partition: str,
    expected_episodes: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> dict[tuple[str, int, str, int], Mapping[str, Any]]:
    rows = _ordered_sequence(records, label="matched-NLL state records")
    expected = {
        (*episode_key, decision_index)
        for episode_key, episode in expected_episodes.items()
        for decision_index in range(episode["policy_decision_state_count"])
    }
    indexed: dict[tuple[str, int, str, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("matched-NLL state records must contain mappings")
        if row.get("partition") != partition:
            raise ValueError("matched-NLL state partition drifted")
        task_type = row.get("task_type")
        task_index = row.get("task_index")
        origin_arm = row.get("origin_arm")
        decision_index = row.get("decision_index")
        if (
            not isinstance(task_type, str)
            or type(task_index) is not int
            or origin_arm not in PRIMARY_ARMS
            or type(decision_index) is not int
            or decision_index < 0
        ):
            raise ValueError("matched-NLL state identity is malformed")
        key = (task_type, task_index, origin_arm, decision_index)
        if key in indexed:
            raise ValueError(f"duplicate matched-NLL state unit: {key}")
        for field in (
            "reference_exact_agreement",
            "nll_kl_finite",
            "selection_budget_audit_complete",
        ):
            _strict_bool(row.get(field), label=field)
        audit_ok = (
            row["reference_exact_agreement"]
            and row["nll_kl_finite"]
            and row["selection_budget_audit_complete"]
        )
        metric_fields = (
            "nll_independent",
            "nll_recent",
            "restoration_independent",
            "restoration_recent",
        )
        if audit_ok:
            for field in metric_fields:
                _finite(row.get(field), label=field)
        else:
            for field in metric_fields:
                if row.get(field) is not None:
                    _finite(row.get(field), label=field)
        indexed[key] = row
    missing = expected - set(indexed)
    unexpected = set(indexed) - expected
    if missing or unexpected:
        raise ValueError(
            f"matched-NLL state inventory drifted: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return indexed


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _sign_epsilon(value: float) -> float:
    if value > RESTORATION_TIE_EPSILON:
        return 1.0
    if value < -RESTORATION_TIE_EPSILON:
        return -1.0
    return 0.0


def _matched_caliper_report(
    template_records: Sequence[Mapping[str, Any]],
    *,
    caliper: float,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    caliper = _finite(caliper, label="NLL caliper")
    if caliper < 0.0:
        raise ValueError("NLL caliper must be nonnegative")
    matched = tuple(
        record
        for record in template_records
        if record["mechanism_eligible"]
        and abs(record["nll_independent"] - record["nll_recent"]) <= caliper
    )
    z_values = tuple(record["z"] for record in matched)
    return {
        "caliper_nats_per_token": caliper,
        "matched_template_count": len(matched),
        "matched_task_types": [record["task_type"] for record in matched],
        "mean_z": _mean(z_values) if z_values else None,
        "bootstrap_lower": (
            template_cluster_bootstrap_lower(
                z_values,
                resamples=bootstrap_resamples,
            )
            if z_values
            else None
        ),
    }


def evaluate_matched_nll_partition(
    episode_records: Sequence[Mapping[str, Any]],
    state_records: Sequence[Mapping[str, Any]],
    *,
    partition: str,
    template_order: Sequence[str],
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Compute the frozen hierarchical matched-NLL mechanism statistic."""
    templates, instance_indices = _partition_spec(partition, template_order)
    episodes = _episode_index(
        episode_records,
        partition=partition,
        templates=templates,
        instance_indices=instance_indices,
        require_state_count=True,
    )
    states = _state_index(
        state_records,
        partition=partition,
        expected_episodes=episodes,
    )

    metric_fields = (
        "nll_independent",
        "nll_recent",
        "restoration_independent",
        "restoration_recent",
    )
    instance_records = []
    template_records = []
    by_template: dict[str, list[dict[str, Any]]] = {item: [] for item in templates}
    for task_type in templates:
        for task_index in instance_indices:
            reasons = []
            origin_means: dict[str, dict[str, float]] = {}
            for origin_arm in PRIMARY_ARMS:
                episode = episodes[(task_type, task_index, origin_arm)]
                count = episode["policy_decision_state_count"]
                if count == 0:
                    reasons.append(f"zero_policy_decision_states:{origin_arm}")
                    continue
                origin_states = tuple(
                    states[(task_type, task_index, origin_arm, decision_index)]
                    for decision_index in range(count)
                )
                failed = tuple(
                    decision_index
                    for decision_index, state in enumerate(origin_states)
                    if not (
                        state["reference_exact_agreement"]
                        and state["nll_kl_finite"]
                        and state["selection_budget_audit_complete"]
                    )
                )
                if failed:
                    reasons.append(
                        f"state_audit_failed:{origin_arm}:"
                        + ",".join(str(item) for item in failed)
                    )
                    continue
                origin_means[origin_arm] = {
                    field: _mean(
                        tuple(_finite(state[field], label=field) for state in origin_states)
                    )
                    for field in metric_fields
                }
            eligible = not reasons and set(origin_means) == set(PRIMARY_ARMS)
            metrics = {
                field: (
                    0.5
                    * (
                        origin_means[INDEPENDENT_ARM][field]
                        + origin_means[RECENT_ARM][field]
                    )
                    if eligible
                    else None
                )
                for field in metric_fields
            }
            record = {
                "task_type": task_type,
                "task_index": task_index,
                "mechanism_eligible": eligible,
                "ineligible_reasons": reasons,
                **metrics,
            }
            instance_records.append(record)
            by_template[task_type].append(record)

    for task_type in templates:
        instances = by_template[task_type]
        eligible = all(item["mechanism_eligible"] for item in instances)
        independent_success = _mean(
            tuple(
                _binary(
                    episodes[(task_type, task_index, INDEPENDENT_ARM)][
                        "official_terminal_success"
                    ],
                    label="independent success",
                )
                for task_index in instance_indices
            )
        )
        recent_success = _mean(
            tuple(
                _binary(
                    episodes[(task_type, task_index, RECENT_ARM)][
                        "official_terminal_success"
                    ],
                    label="recent success",
                )
                for task_index in instance_indices
            )
        )
        metrics = {
            field: (
                _mean(tuple(_finite(item[field], label=field) for item in instances))
                if eligible
                else None
            )
            for field in metric_fields
        }
        restoration_delta = (
            metrics["restoration_independent"] - metrics["restoration_recent"]
            if eligible
            else None
        )
        template_records.append(
            {
                "task_type": task_type,
                "instance_pair_count": len(instances),
                "mechanism_eligible": eligible,
                "ineligible_instance_indices": [
                    item["task_index"]
                    for item in instances
                    if not item["mechanism_eligible"]
                ],
                **metrics,
                "independent_success": independent_success,
                "recent_success": recent_success,
                "success_difference": independent_success - recent_success,
                "restoration_difference": restoration_delta,
                "z": (
                    _sign_epsilon(restoration_delta)
                    * (independent_success - recent_success)
                    if eligible
                    else None
                ),
            }
        )

    primary = _matched_caliper_report(
        template_records,
        caliper=PRIMARY_CALIPER,
        bootstrap_resamples=bootstrap_resamples,
    )
    sensitivities = [
        _matched_caliper_report(
            template_records,
            caliper=caliper,
            bootstrap_resamples=bootstrap_resamples,
        )
        for caliper in SENSITIVITY_CALIPERS
    ]
    primary_claim_supported = (
        partition == "test"
        and primary["matched_template_count"] >= MINIMUM_MATCHED_TEST_TEMPLATES
        and primary["bootstrap_lower"] is not None
        and primary["bootstrap_lower"] > 0.0
    )
    return {
        "partition": partition,
        "role": (
            "development_diagnostic_only"
            if partition == "train"
            else "primary_mechanism_evidence"
        ),
        "template_count": len(templates),
        "instance_indices": list(instance_indices),
        "instance_records": instance_records,
        "template_records": template_records,
        "mechanism_eligible_template_count": sum(
            item["mechanism_eligible"] for item in template_records
        ),
        "primary_caliper": primary,
        "sensitivity_calipers": sensitivities,
        "minimum_matched_test_templates": MINIMUM_MATCHED_TEST_TEMPLATES,
        "primary_claim_supported": primary_claim_supported,
        "interpretation": (
            "post_treatment_mechanism_association_on_controller_conditioned_"
            "state_distributions_not_a_mediated_causal_effect"
        ),
    }


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "INDEPENDENT_ARM",
    "MINIMUM_MATCHED_TEST_TEMPLATES",
    "PARTITION_INSTANCE_INDICES",
    "PARTITION_TEMPLATE_COUNTS",
    "PRIMARY_CALIPER",
    "RECENT_ARM",
    "RESTORATION_TIE_EPSILON",
    "SENSITIVITY_CALIPERS",
    "evaluate_closed_loop_partition",
    "evaluate_matched_nll_partition",
    "template_cluster_bootstrap_lower",
    "type7_quantile",
]
