"""Pure evaluation for the frozen validation-12 exploratory closed loop."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


PROTOCOL_ID = "causalcache_exploratory_closed_loop_validation12_v1"
PARTITION = "validation"
TASK_INDEX = 0

INDEPENDENT_ARM = "independent_B2"
RECENT_ARM = "recent_B2"
OCR_RGB_ARM = "ocr_rgb_B2"
CONDITIONAL_ARM = "conditional_B2"
SUMMARY_ARM = "summary_B0"
ARMS = (
    INDEPENDENT_ARM,
    RECENT_ARM,
    OCR_RGB_ARM,
    CONDITIONAL_ARM,
    SUMMARY_ARM,
)

STRATUM_ORDER = ("short", "medium", "long")
TEMPLATE_STRATA = (
    ("SimpleSmsSendClipboardContent", "short"),
    ("MarkorDeleteNote", "short"),
    ("MarkorChangeNoteContent", "short"),
    ("MarkorAddNoteHeader", "short"),
    ("RecipeDeleteSingleWithRecipeWithNoise", "medium"),
    ("TurnOnWifiAndOpenApp", "medium"),
    ("SimpleCalendarDeleteEvents", "medium"),
    ("SimpleSmsSendReceivedAddress", "medium"),
    ("ExpenseAddMultiple", "long"),
    ("RetroSavePlaylist", "long"),
    ("RetroPlayingQueue", "long"),
    ("ExpenseAddMultipleFromGallery", "long"),
)
TEMPLATE_ORDER = tuple(item[0] for item in TEMPLATE_STRATA)
STRATUM_BY_TEMPLATE = dict(TEMPLATE_STRATA)

REGISTERED_CONTRASTS = (
    (INDEPENDENT_ARM, RECENT_ARM),
    (INDEPENDENT_ARM, OCR_RGB_ARM),
    (INDEPENDENT_ARM, SUMMARY_ARM),
    (CONDITIONAL_ARM, RECENT_ARM),
    (CONDITIONAL_ARM, OCR_RGB_ARM),
)
VERSUS_RECENT_CONTRASTS = tuple(
    (arm, RECENT_ARM) for arm in ARMS if arm != RECENT_ARM
)
ALL_CONTRASTS = tuple(dict.fromkeys((*REGISTERED_CONTRASTS, *VERSUS_RECENT_CONTRASTS)))

EXPECTED_TEMPLATE_COUNT = 12
EXPECTED_ARM_COUNT = 5
EXPECTED_EPISODE_COUNT = EXPECTED_TEMPLATE_COUNT * EXPECTED_ARM_COUNT

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 271_828
BOOTSTRAP_CONFIDENCE = 0.90

ADVANCE_ROUTE = "ADVANCE_TO_NEW_TRAIN60_PROTOCOL"
STOP_ROUTE = "STOP_CURRENT_INDEPENDENT_CLOSED_LOOP_DIRECTION"
INCONCLUSIVE_ROUTE = "INCONCLUSIVE_DEVELOPMENT_ONLY"
INVALID_ROUTE = "INVALID_EXPLORATORY_CLOSED_LOOP_EXECUTION_V1"

FAILURE_CLASSIFICATIONS = (
    "official_success",
    "terminal_failure",
    "parse_failure",
    "selector_failure",
    "executor_failure",
    "infrastructure_failure",
)


def _sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise TypeError(f"{label} must be an ordered sequence")
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


def _nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be a nonnegative int")
    return value


def _strict_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a bool")
    return value


def type7_quantile(values: Sequence[float], probability: float) -> float:
    """Return the Hyndman--Fan type-7 quantile."""
    ordered = tuple(sorted(_finite(item, label="quantile value") for item in values))
    probability = _finite(probability, label="quantile probability")
    if not ordered:
        raise ValueError("quantile values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be in [0, 1]")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def paired_bootstrap_interval(
    paired_effects: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Bootstrap complete paired-template effects with the frozen RNG family."""
    effects = tuple(
        _finite(item, label="paired bootstrap effect") for item in paired_effects
    )
    if not effects:
        raise ValueError("paired bootstrap effects must not be empty")
    if type(resamples) is not int or resamples <= 0:
        raise TypeError("bootstrap resamples must be a positive int")
    if type(seed) is not int:
        raise TypeError("bootstrap seed must be an int")
    confidence = _finite(confidence, label="bootstrap confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie strictly between zero and one")

    generator = random.Random(seed)
    count = len(effects)
    means = tuple(
        math.fsum(effects[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "unit": "template_pair",
        "resamples": resamples,
        "seed": seed,
        "confidence": confidence,
        "interval": "percentile",
        "quantile": "Hyndman_Fan_type_7",
        "lower": type7_quantile(means, tail),
        "upper": type7_quantile(means, 1.0 - tail),
        "role": "descriptive_only",
    }


def exact_two_sided_sign_test(wins: int, losses: int) -> dict[str, Any]:
    """Return the exact doubled-tail sign test over discordant pairs."""
    wins = _nonnegative_int(wins, label="sign-test wins")
    losses = _nonnegative_int(losses, label="sign-test losses")
    discordant = wins + losses
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(wins, losses)
        tail_mass = math.fsum(
            math.comb(discordant, index) for index in range(smaller + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail_mass)
    return {
        "test": "exact_two_sided_sign_test",
        "wins": wins,
        "losses": losses,
        "discordant_pair_count": discordant,
        "p_value": p_value,
        "role": "descriptive_only",
    }


def _expected_keys() -> set[tuple[str, int, str]]:
    return {
        (task_type, TASK_INDEX, arm)
        for task_type in TEMPLATE_ORDER
        for arm in ARMS
    }


def _key_record(key: tuple[str, int, str]) -> dict[str, Any]:
    return {"task_type": key[0], "task_index": key[1], "arm": key[2]}


def _normalize_episode(row: Mapping[str, Any], *, ordinal: int) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError(f"episode {ordinal} must be a mapping")
    task_type = row.get("task_type")
    task_index = row.get("task_index")
    arm = row.get("arm")
    horizon_stratum = row.get("horizon_stratum")
    partition = row.get("partition")
    if not isinstance(task_type, str) or not task_type:
        raise TypeError(f"episode {ordinal} task_type must be nonempty text")
    if type(task_index) is not int:
        raise TypeError(f"episode {ordinal} task_index must be an int")
    if not isinstance(arm, str) or not arm:
        raise TypeError(f"episode {ordinal} arm must be nonempty text")
    if not isinstance(horizon_stratum, str) or not horizon_stratum:
        raise TypeError(f"episode {ordinal} horizon_stratum must be nonempty text")
    if not isinstance(partition, str) or not partition:
        raise TypeError(f"episode {ordinal} partition must be nonempty text")

    parse_attempt_count = _nonnegative_int(
        row.get("action_parse_attempt_count"),
        label=f"episode {ordinal} action_parse_attempt_count",
    )
    parse_success_count = _nonnegative_int(
        row.get("action_parse_success_count"),
        label=f"episode {ordinal} action_parse_success_count",
    )
    if parse_success_count > parse_attempt_count:
        raise ValueError("action parse successes cannot exceed attempts")

    success = _binary(
        row.get("official_terminal_success"),
        label=f"episode {ordinal} official_terminal_success",
    )
    infrastructure_failure = _strict_bool(
        row.get("infrastructure_failure"),
        label=f"episode {ordinal} infrastructure_failure",
    )
    normal_environment_chain = _strict_bool(
        row.get("normal_environment_chain"),
        label=f"episode {ordinal} normal_environment_chain",
    )
    audits_complete = _strict_bool(
        row.get("required_audits_complete"),
        label=f"episode {ordinal} required_audits_complete",
    )
    failure_classification = row.get("failure_classification")
    if failure_classification not in FAILURE_CLASSIFICATIONS:
        raise ValueError(f"episode {ordinal} failure_classification is invalid")
    if success == 1.0 and failure_classification != "official_success":
        raise ValueError("recorded failure must have ITT success zero")
    if success == 0.0 and failure_classification == "official_success":
        raise ValueError("official_success classification requires success one")
    if infrastructure_failure != (
        failure_classification == "infrastructure_failure"
    ):
        raise ValueError("infrastructure failure flag and classification disagree")
    if infrastructure_failure and normal_environment_chain:
        raise ValueError(
            "infrastructure failure cannot have a normal environment chain"
        )

    return {
        "partition": partition,
        "task_type": task_type,
        "task_index": task_index,
        "horizon_stratum": horizon_stratum,
        "arm": arm,
        "official_terminal_success": success,
        "action_parse_attempt_count": parse_attempt_count,
        "action_parse_success_count": parse_success_count,
        "infrastructure_failure": infrastructure_failure,
        "normal_environment_chain": normal_environment_chain,
        "required_audits_complete": audits_complete,
        "failure_classification": failure_classification,
    }


def _normalize_execution_audit(value: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("execution_audit must be a mapping")
    fields = (
        "hidden_retry_count",
        "top_up_count",
        "template_or_arm_deletion_count",
        "memory_binding_decision_count",
        "independent_recent_selector_disagreement_count",
    )
    return {
        field: _nonnegative_int(value.get(field), label=f"execution audit {field}")
        for field in fields
    }


def _index_episodes(
    rows: Sequence[dict[str, Any]],
) -> tuple[
    dict[tuple[str, int, str], dict[str, Any]],
    list[tuple[str, int, str]],
    list[tuple[str, int, str]],
    list[tuple[str, int, str]],
    list[str],
]:
    expected = _expected_keys()
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    duplicates: list[tuple[str, int, str]] = []
    unexpected: list[tuple[str, int, str]] = []
    identity_issues: list[str] = []
    for row in rows:
        key = (row["task_type"], row["task_index"], row["arm"])
        if key not in expected:
            unexpected.append(key)
            continue
        if key in indexed:
            duplicates.append(key)
            continue
        indexed[key] = row
        if row["partition"] != PARTITION:
            identity_issues.append(f"{key!r} partition drifted")
        expected_stratum = STRATUM_BY_TEMPLATE[row["task_type"]]
        if row["horizon_stratum"] != expected_stratum:
            identity_issues.append(f"{key!r} horizon stratum drifted")
    missing = sorted(expected - set(indexed))
    return (
        indexed,
        missing,
        sorted(set(duplicates)),
        sorted(set(unexpected)),
        identity_issues,
    )


def _validity_report(
    rows: Sequence[dict[str, Any]],
    indexed: Mapping[tuple[str, int, str], dict[str, Any]],
    missing: Sequence[tuple[str, int, str]],
    duplicates: Sequence[tuple[str, int, str]],
    unexpected: Sequence[tuple[str, int, str]],
    identity_issues: Sequence[str],
    execution_audit: Mapping[str, int],
) -> dict[str, Any]:
    inventory_complete = not (
        missing or duplicates or unexpected or identity_issues
    ) and len(indexed) == EXPECTED_EPISODE_COUNT
    audit_complete_count = sum(
        row["required_audits_complete"] for row in indexed.values()
    )
    audit_coverage = audit_complete_count / EXPECTED_EPISODE_COUNT

    parse_coverage: dict[str, float] = {}
    non_infrastructure_count: dict[str, int] = {}
    for arm in ARMS:
        arm_rows = [row for row in indexed.values() if row["arm"] == arm]
        attempts = sum(row["action_parse_attempt_count"] for row in arm_rows)
        successes = sum(row["action_parse_success_count"] for row in arm_rows)
        parse_coverage[arm] = successes / attempts if attempts else 0.0
        non_infrastructure_count[arm] = sum(
            not row["infrastructure_failure"] for row in arm_rows
        )

    normal_template_count = 0
    for task_type in TEMPLATE_ORDER:
        arm_rows = [indexed.get((task_type, TASK_INDEX, arm)) for arm in ARMS]
        if all(
            row is not None and row["normal_environment_chain"] for row in arm_rows
        ):
            normal_template_count += 1

    checks = {
        "exact_episode_record_count": len(rows) == EXPECTED_EPISODE_COUNT,
        "paired_arm_inventory_complete": inventory_complete,
        "hidden_retry_count": execution_audit["hidden_retry_count"] == 0,
        "top_up_count": execution_audit["top_up_count"] == 0,
        "template_or_arm_deletion_count": (
            execution_audit["template_or_arm_deletion_count"] == 0
        ),
        "required_audit_coverage": audit_coverage == 1.0,
        "per_arm_action_parse_coverage": all(
            parse_coverage[arm] >= 0.95 for arm in ARMS
        ),
        "per_arm_non_infrastructure_episode_count": all(
            non_infrastructure_count[arm] >= 11 for arm in ARMS
        ),
        "all_five_arms_normal_environment_chain_template_count": (
            normal_template_count >= 10
        ),
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "valid": not reasons,
        "checks": checks,
        "reasons": reasons,
        "inventory": {
            "expected_episode_record_count": EXPECTED_EPISODE_COUNT,
            "observed_episode_record_count": len(rows),
            "unique_expected_episode_record_count": len(indexed),
            "missing": [_key_record(item) for item in missing],
            "duplicates": [_key_record(item) for item in duplicates],
            "unexpected": [_key_record(item) for item in unexpected],
            "identity_issues": list(identity_issues),
        },
        "required_audit_coverage": audit_coverage,
        "per_arm_action_parse_coverage": parse_coverage,
        "per_arm_non_infrastructure_episode_count": non_infrastructure_count,
        "all_five_arms_normal_environment_chain_template_count": (
            normal_template_count
        ),
    }


def _contrast_name(left_arm: str, right_arm: str) -> str:
    return f"{left_arm}_minus_{right_arm}"


def _paired_contrast(
    index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    templates: Sequence[str],
    left_arm: str,
    right_arm: str,
    *,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    effects = tuple(
        index[(task_type, TASK_INDEX, left_arm)]["official_terminal_success"]
        - index[(task_type, TASK_INDEX, right_arm)]["official_terminal_success"]
        for task_type in templates
    )
    wins = sum(effect > 0.0 for effect in effects)
    ties = sum(effect == 0.0 for effect in effects)
    losses = sum(effect < 0.0 for effect in effects)
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "template_count": len(effects),
        "paired_mean_difference": math.fsum(effects) / len(effects),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "wins_minus_losses": wins - losses,
        "bootstrap": paired_bootstrap_interval(
            effects,
            resamples=bootstrap_resamples,
        ),
        "sign_test": exact_two_sided_sign_test(wins, losses),
    }


def _region_statistics(
    index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    templates: Sequence[str],
    *,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    template_tuple = tuple(templates)
    arm_success = {}
    for arm in ARMS:
        successes = math.fsum(
            index[(task_type, TASK_INDEX, arm)]["official_terminal_success"]
            for task_type in template_tuple
        )
        arm_success[arm] = {
            "success_count": int(successes),
            "success_rate": successes / len(template_tuple),
        }
    contrasts = {
        _contrast_name(left_arm, right_arm): _paired_contrast(
            index,
            template_tuple,
            left_arm,
            right_arm,
            bootstrap_resamples=bootstrap_resamples,
        )
        for left_arm, right_arm in ALL_CONTRASTS
    }
    return {
        "template_count": len(template_tuple),
        "task_types": list(template_tuple),
        "arm_success": arm_success,
        "paired_contrasts": contrasts,
        "versus_recent_contrast_names": [
            _contrast_name(left_arm, right_arm)
            for left_arm, right_arm in VERSUS_RECENT_CONTRASTS
        ],
        "registered_contrast_names": [
            _contrast_name(left_arm, right_arm)
            for left_arm, right_arm in REGISTERED_CONTRASTS
        ],
    }


def _statistics(
    index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    strata = {
        stratum: _region_statistics(
            index,
            tuple(
                task_type
                for task_type, task_stratum in TEMPLATE_STRATA
                if task_stratum == stratum
            ),
            bootstrap_resamples=bootstrap_resamples,
        )
        for stratum in STRATUM_ORDER
    }
    return {
        "overall": _region_statistics(
            index,
            TEMPLATE_ORDER,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "horizon_strata": strata,
        "horizon_definition": "frozen_pre_treatment_max_steps",
        "realized_episode_steps_used_for_stratification": False,
    }


def _wins_minus_losses(
    statistics: Mapping[str, Any],
    region: str,
    left_arm: str,
    right_arm: str,
) -> int:
    if region == "overall":
        region_statistics = statistics["overall"]
    else:
        region_statistics = statistics["horizon_strata"][region]
    return int(
        region_statistics["paired_contrasts"][_contrast_name(left_arm, right_arm)][
            "wins_minus_losses"
        ]
    )


def _route(
    statistics: Mapping[str, Any] | None,
    validity: Mapping[str, Any],
    execution_audit: Mapping[str, int],
) -> dict[str, Any]:
    if statistics is None:
        return {
            "route": INVALID_ROUTE,
            "advance_criteria": None,
            "stop_criteria": None,
        }

    independent_recent_overall = _wins_minus_losses(
        statistics, "overall", INDEPENDENT_ARM, RECENT_ARM
    )
    independent_recent_long = _wins_minus_losses(
        statistics, "long", INDEPENDENT_ARM, RECENT_ARM
    )
    independent_ocr_overall = _wins_minus_losses(
        statistics, "overall", INDEPENDENT_ARM, OCR_RGB_ARM
    )
    independent_ocr_long = _wins_minus_losses(
        statistics, "long", INDEPENDENT_ARM, OCR_RGB_ARM
    )
    independent_summary_overall = _wins_minus_losses(
        statistics, "overall", INDEPENDENT_ARM, SUMMARY_ARM
    )

    advance_criteria = {
        "valid_execution": bool(validity["valid"]),
        "independent_vs_recent_overall_wins_minus_losses_at_least_2": (
            independent_recent_overall >= 2
        ),
        "independent_vs_recent_long_wins_minus_losses_at_least_1": (
            independent_recent_long >= 1
        ),
        "independent_vs_ocr_overall_not_negative": independent_ocr_overall >= 0,
        "independent_vs_ocr_long_not_negative": independent_ocr_long >= 0,
        "independent_vs_summary_overall_not_negative": (
            independent_summary_overall >= 0
        ),
        "memory_binding_decision_count_strictly_positive": (
            execution_audit["memory_binding_decision_count"] > 0
        ),
        "independent_recent_selector_disagreement_count_strictly_positive": (
            execution_audit["independent_recent_selector_disagreement_count"] > 0
        ),
    }
    stop_criteria = {
        "independent_vs_recent_overall_not_positive": (
            independent_recent_overall <= 0
        ),
        "independent_vs_recent_long_not_positive": independent_recent_long <= 0,
        "independent_vs_ocr_overall_not_positive": independent_ocr_overall <= 0,
        "independent_vs_ocr_long_not_positive": independent_ocr_long <= 0,
    }

    if not validity["valid"]:
        route = INVALID_ROUTE
    elif all(advance_criteria.values()):
        route = ADVANCE_ROUTE
    elif all(stop_criteria.values()):
        route = STOP_ROUTE
    else:
        route = INCONCLUSIVE_ROUTE
    return {
        "route": route,
        "advance_criteria": advance_criteria,
        "stop_criteria": stop_criteria,
        "conditional_success_may_rescue_independent_route": False,
        "route_is_paper_claim": False,
    }


def evaluate_exploratory_closed_loop(
    episode_records: Sequence[Mapping[str, Any]],
    *,
    execution_audit: Mapping[str, Any],
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Evaluate the fixed 12-template, five-arm development-only probe."""
    supplied = _sequence(episode_records, label="episode_records")
    normalized = tuple(
        _normalize_episode(row, ordinal=ordinal) for ordinal, row in enumerate(supplied)
    )
    audit = _normalize_execution_audit(execution_audit)
    if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
        raise TypeError("bootstrap_resamples must be a positive int")

    indexed, missing, duplicates, unexpected, identity_issues = _index_episodes(
        normalized
    )
    validity = _validity_report(
        normalized,
        indexed,
        missing,
        duplicates,
        unexpected,
        identity_issues,
        audit,
    )
    complete_for_statistics = not (missing or duplicates or unexpected)
    statistics = (
        _statistics(indexed, bootstrap_resamples=bootstrap_resamples)
        if complete_for_statistics
        else None
    )
    routing = _route(statistics, validity, audit)
    return {
        "schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "evidence_role": "development_only_resource_routing",
        "fixed_denominator": {
            "partition": PARTITION,
            "template_count": EXPECTED_TEMPLATE_COUNT,
            "arm_count": EXPECTED_ARM_COUNT,
            "episode_count": EXPECTED_EPISODE_COUNT,
            "task_index": TASK_INDEX,
            "horizon_stratum_template_counts": {
                stratum: sum(item[1] == stratum for item in TEMPLATE_STRATA)
                for stratum in STRATUM_ORDER
            },
        },
        "execution_audit": audit,
        "execution_validity": validity,
        "statistics": statistics,
        "routing": routing,
    }
