"""Validation and aggregation for the frozen AndroidWorld zero-shot matrix."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


POLICIES = ("frozen", "full_layer", "history_gated")
ARMS = ("summary_B0", "recent_B4", "recent_B8")
TASK_INDICES = (0, 1)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_724
BOOTSTRAP_CONFIDENCE = 0.95

FULL_LAYER_SHA256 = (
    "75670fa5bb0a492f5ead26212de49b7e1fe8ddf7c25f27a2a2a82a934f14fdc3"
)
HISTORY_GATED_SHA256 = (
    "8f2cc49e1aa0b06ce231eb54937d813317f5274a799c97b09be7fdb22be46317"
)

DRAWING_TASKS = frozenset({"BrowserDraw", "SimpleDrawProCreateDrawing"})
TRIVIAL_VERIFY_TASKS = frozenset(
    {
        "ClockStopWatchPausedVerify",
        "SystemBluetoothTurnOffVerify",
        "SystemBluetoothTurnOnVerify",
        "SystemBrightnessMaxVerify",
        "SystemBrightnessMinVerify",
        "SystemWifiTurnOffVerify",
        "SystemWifiTurnOnVerify",
    }
)
ACTION_INCOMPATIBLE_TASKS = DRAWING_TASKS | TRIVIAL_VERIFY_TASKS


@dataclass(frozen=True)
class Attempt:
    source: str
    path: str
    sha256: str
    key: tuple[str, str, str, int]
    success: float
    infrastructure_failure: bool
    model_step_count: int
    failure_classification: str
    started_at: str
    row: Mapping[str, Any]

    @property
    def void(self) -> bool:
        return self.infrastructure_failure and self.model_step_count == 0

    @property
    def started_infrastructure_failure(self) -> bool:
        return self.infrastructure_failure and self.model_step_count > 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def type7_quantile(values: Sequence[float], probability: float) -> float:
    ordered = tuple(sorted(float(value) for value in values))
    if not ordered:
        raise ValueError("quantile input must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0, 1]")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def paired_bootstrap_interval(
    effects: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    values = tuple(float(value) for value in effects)
    if not values:
        raise ValueError("paired bootstrap requires at least one template")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    generator = random.Random(seed)
    count = len(values)
    draws = [
        math.fsum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    ]
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return [
        type7_quantile(draws, tail),
        type7_quantile(draws, 1.0 - tail),
    ]


def load_roster(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "full":
        raise ValueError("matrix roster must declare split='full'")
    if payload.get("task_type_count") != 116:
        raise ValueError("matrix roster must contain 116 task templates")
    if payload.get("task_instance_count") != 232:
        raise ValueError("matrix roster must contain 232 task instances")
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for row in payload["instances"]:
        required_fields = {
            "goal",
            "max_steps",
            "origin_split",
            "task_index",
            "task_type",
        }
        missing_fields = sorted(required_fields - set(row))
        if missing_fields:
            raise ValueError(
                f"roster row is missing required fields: {missing_fields}"
            )
        key = (str(row["task_type"]), int(row["task_index"]))
        if key in records:
            raise ValueError(f"duplicate roster key: {key}")
        records[key] = dict(row)
    templates = {key[0] for key in records}
    if len(templates) != 116:
        raise ValueError("matrix roster template inventory drifted")
    for task_type in templates:
        indices = {index for name, index in records if name == task_type}
        if indices != set(TASK_INDICES):
            raise ValueError(f"roster task indices drifted for {task_type}: {indices}")
    if not ACTION_INCOMPATIBLE_TASKS <= templates:
        missing = sorted(ACTION_INCOMPATIBLE_TASKS - templates)
        raise ValueError(f"action-incompatible roster tasks are missing: {missing}")
    return records


def classify_policy(row: Mapping[str, Any]) -> str:
    policy = row.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("episode policy must be a mapping")
    checkpoint_sha256 = policy.get("lora_checkpoint_sha256")
    adapter_type = policy.get("adapter_type")
    if checkpoint_sha256 is None and adapter_type is None:
        return "frozen"
    if checkpoint_sha256 == FULL_LAYER_SHA256 and adapter_type in {
        None,
        "full_policy_lora",
    }:
        return "full_layer"
    if (
        checkpoint_sha256 == HISTORY_GATED_SHA256
        and adapter_type == "history_gated_kv"
    ):
        return "history_gated"
    raise ValueError(
        "unregistered matrix policy: "
        f"adapter_type={adapter_type!r}, checkpoint={checkpoint_sha256!r}"
    )


def _binary(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if result not in {0.0, 1.0}:
        raise ValueError(f"{label} must be binary")
    return result


def normalize_attempt(
    row: Mapping[str, Any],
    *,
    source: str,
    path: Path,
    file_sha256: str,
    roster: Mapping[tuple[str, int], Mapping[str, Any]],
) -> Attempt:
    task_type = str(row.get("task_type", ""))
    task_index = row.get("task_index")
    if not task_type or type(task_index) is not int:
        raise ValueError("episode task identity is malformed")
    roster_key = (task_type, task_index)
    if roster_key not in roster:
        raise ValueError(f"episode is outside the frozen roster: {roster_key}")
    arm = row.get("arm")
    if arm not in ARMS:
        raise ValueError(f"episode arm is not registered: {arm!r}")
    policy = classify_policy(row)

    instance = row.get("instance")
    if not isinstance(instance, Mapping):
        raise ValueError("episode instance must be a mapping")
    expected = roster[roster_key]
    for field in ("task_type", "task_index", "goal", "max_steps"):
        if instance.get(field) != expected.get(field):
            raise ValueError(
                f"episode instance {field} disagrees with roster for {roster_key}"
            )
    partition = row.get("partition")
    if partition not in {"full", expected["origin_split"]}:
        raise ValueError(
            f"episode partition {partition!r} is invalid for {roster_key}"
        )
    if row.get("allow_sealed_split") is not True:
        raise ValueError("episode did not opt into the sealed roster")
    if row.get("shared_early_decisions") != 2:
        raise ValueError("shared_early_decisions drifted")
    if row.get("parse_retries_allowed") != 1:
        raise ValueError("parse_retries_allowed drifted")

    policy_payload = row["policy"]
    if policy_payload.get("target_effective_visual_tokens_per_image") != 2560:
        raise ValueError("effective visual-token budget drifted")
    infrastructure_failure = row.get("infrastructure_failure")
    if type(infrastructure_failure) is not bool:
        raise TypeError("infrastructure_failure must be a bool")
    model_step_count = row.get("model_step_count")
    if type(model_step_count) is not int or model_step_count < 0:
        raise TypeError("model_step_count must be a nonnegative int")
    success = _binary(
        row.get("official_terminal_success"),
        label="official_terminal_success",
    )
    failure_classification = row.get("failure_classification")
    if not isinstance(failure_classification, str) or not failure_classification:
        raise ValueError("failure_classification must be nonempty text")
    if infrastructure_failure != (
        failure_classification == "infrastructure_failure"
    ):
        raise ValueError("infrastructure flag and failure classification disagree")
    if success == 1.0 and failure_classification != "official_success":
        raise ValueError("successful episode must be classified official_success")
    if success == 0.0 and failure_classification == "official_success":
        raise ValueError("official_success requires terminal success")
    producer_success = (
        row.get("score_after") == 1.0
        and row.get("termination_reason") == "policy_terminated"
    )
    if bool(success) != producer_success:
        raise ValueError(
            "official_terminal_success disagrees with "
            "score_after==1 and policy_terminated"
        )
    if not infrastructure_failure and row.get("required_audits_complete") is not True:
        raise ValueError("formal non-infrastructure episode lacks required audits")
    started_at = row.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError("episode started_at must be nonempty text")
    return Attempt(
        source=source,
        path=str(path),
        sha256=file_sha256,
        key=(policy, str(arm), task_type, task_index),
        success=success,
        infrastructure_failure=infrastructure_failure,
        model_step_count=model_step_count,
        failure_classification=failure_classification,
        started_at=started_at,
        row=row,
    )


def load_attempts(
    inputs: Sequence[tuple[str, Path]],
    *,
    roster: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[list[Attempt], list[dict[str, str]]]:
    attempts: list[Attempt] = []
    rejected: list[dict[str, str]] = []
    for source, root in inputs:
        if not root.is_dir():
            raise FileNotFoundError(f"matrix input root does not exist: {root}")
        for path in sorted(root.rglob("*.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                rejected.append(
                    {"source": source, "path": str(path), "reason": str(error)}
                )
                continue
            if not isinstance(row, Mapping) or "task_type" not in row:
                continue
            roster_key = (str(row.get("task_type", "")), row.get("task_index"))
            # note (luojiaxuan): 同一 run root 还保留旧 sealed-25 的 index-2
            # appendix 记录；它们不属于 116×2 headline roster，既不是坏文件也不能
            # 混进 formal-cell inventory。
            if roster_key not in roster:
                continue
            try:
                attempt = normalize_attempt(
                    row,
                    source=source,
                    path=path,
                    file_sha256=sha256_file(path),
                    roster=roster,
                )
            except (TypeError, ValueError) as error:
                rejected.append(
                    {"source": source, "path": str(path), "reason": str(error)}
                )
                continue
            attempts.append(attempt)
    return attempts, rejected


def expected_keys(
    roster: Mapping[tuple[str, int], Mapping[str, Any]],
) -> set[tuple[str, str, str, int]]:
    return {
        (policy, arm, task_type, task_index)
        for policy in POLICIES
        for arm in ARMS
        for task_type, task_index in roster
    }


def resolve_attempts(
    attempts: Iterable[Attempt],
    *,
    roster: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str, str, int], Attempt],
    dict[str, Any],
]:
    grouped: dict[tuple[str, str, str, int], list[Attempt]] = defaultdict(list)
    exact_duplicate_count = 0
    seen_sha_by_key: dict[tuple[str, str, str, int], set[str]] = defaultdict(set)
    for attempt in attempts:
        if attempt.sha256 in seen_sha_by_key[attempt.key]:
            exact_duplicate_count += 1
            continue
        seen_sha_by_key[attempt.key].add(attempt.sha256)
        grouped[attempt.key].append(attempt)

    selected: dict[tuple[str, str, str, int], Attempt] = {}
    unauthorized_noninfra_duplicates: dict[str, list[dict[str, Any]]] = {}
    void_attempts = 0
    started_infra_attempts = 0
    for key, rows in grouped.items():
        rows.sort(key=lambda item: (item.started_at, item.source, item.path))
        void_attempts += sum(item.void for item in rows)
        started_infra_attempts += sum(
            item.started_infrastructure_failure for item in rows
        )
        formal = [item for item in rows if not item.infrastructure_failure]
        if len(formal) == 1:
            selected[key] = formal[0]
        elif len(formal) > 1:
            unauthorized_noninfra_duplicates[json.dumps(key)] = [
                {
                    "path": item.path,
                    "sha256": item.sha256,
                    "source": item.source,
                    "started_at": item.started_at,
                    "success": item.success,
                }
                for item in formal
            ]

    expected = expected_keys(roster)
    missing = sorted(expected - set(selected))
    unexpected = sorted(set(grouped) - expected)

    # note (luojiaxuan): hard-delete 视图只认同一 instance 的九个
    # policy×arm 都留下零步 env-init void、且没有任何真实开跑记录的证据。
    # 这一定义是 policy-agnostic、跑后可审计，不会按 success 选择样本。
    env_init_failures: list[dict[str, Any]] = []
    for task_type, task_index in sorted(roster):
        instance_keys = [
            (policy, arm, task_type, task_index)
            for policy in POLICIES
            for arm in ARMS
        ]
        all_void_only = all(
            grouped.get(key)
            and all(attempt.void for attempt in grouped[key])
            for key in instance_keys
        )
        if all_void_only:
            env_init_failures.append(
                {"task_type": task_type, "task_index": task_index}
            )

    audit = {
        "attempt_count": sum(len(rows) for rows in grouped.values()),
        "exact_duplicate_count": exact_duplicate_count,
        "expected_cell_count": len(expected),
        "formal_cell_count": len(selected),
        "missing_cell_count": len(missing),
        "missing_cells": [
            {
                "policy": key[0],
                "arm": key[1],
                "task_type": key[2],
                "task_index": key[3],
            }
            for key in missing
        ],
        "policy_agnostic_env_init_failure_instances": env_init_failures,
        "rejected_or_malformed_count": 0,
        "started_infrastructure_attempt_count": started_infra_attempts,
        "unauthorized_noninfra_duplicate_count": len(
            unauthorized_noninfra_duplicates
        ),
        "unauthorized_noninfra_duplicates": unauthorized_noninfra_duplicates,
        "unexpected_cell_count": len(unexpected),
        "unexpected_cells": [list(key) for key in unexpected],
        "void_attempt_count": void_attempts,
    }
    return selected, audit


def _paired_summary(
    effects: Sequence[float],
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    values = tuple(float(value) for value in effects)
    wins = sum(value > 0.0 for value in values)
    ties = sum(value == 0.0 for value in values)
    losses = sum(value < 0.0 for value in values)
    return {
        "template_count": len(values),
        "paired_template_macro_difference": math.fsum(values) / len(values),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "bootstrap_95_ci": paired_bootstrap_interval(
            values,
            resamples=bootstrap_resamples,
            seed=seed,
        ),
    }


def aggregate_view(
    selected: Mapping[tuple[str, str, str, int], Attempt],
    *,
    template_indices: Mapping[str, Sequence[int]],
    bootstrap_resamples: int,
) -> dict[str, Any]:
    templates = tuple(sorted(template_indices))
    required = {
        (policy, arm, task_type, task_index)
        for policy in POLICIES
        for arm in ARMS
        for task_type, indices in template_indices.items()
        for task_index in indices
    }
    missing = sorted(required - set(selected))
    if missing:
        return {
            "ready": False,
            "template_count": len(templates),
            "instance_count": sum(len(indices) for indices in template_indices.values()),
            "missing_cell_count": len(missing),
        }

    template_values: dict[tuple[str, str, str], float] = {}
    cell_metrics: dict[str, Any] = {}
    for policy in POLICIES:
        cell_metrics[policy] = {}
        for arm in ARMS:
            values = []
            failure_counts: Counter[str] = Counter()
            episode_success = 0.0
            episode_count = 0
            for task_type in templates:
                per_instance = [
                    selected[(policy, arm, task_type, task_index)].success
                    for task_index in template_indices[task_type]
                ]
                value = math.fsum(per_instance) / len(per_instance)
                template_values[(policy, arm, task_type)] = value
                values.append(value)
                for task_index in template_indices[task_type]:
                    attempt = selected[(policy, arm, task_type, task_index)]
                    failure_counts[attempt.failure_classification] += 1
                    episode_success += attempt.success
                    episode_count += 1
            cell_metrics[policy][arm] = {
                "episode_count": episode_count,
                "episode_success_count": int(episode_success),
                "episode_success_rate": episode_success / episode_count,
                "failure_classification_counts": dict(sorted(failure_counts.items())),
                "template_macro_success": math.fsum(values) / len(values),
            }

    comparisons: dict[str, Any] = {
        "within_policy_arms": {},
        "fixed_arm_policies": {},
    }
    arm_pairs = (
        ("recent_B4", "summary_B0"),
        ("recent_B8", "summary_B0"),
        ("recent_B8", "recent_B4"),
    )
    policy_pairs = (
        ("full_layer", "frozen"),
        ("history_gated", "frozen"),
        ("history_gated", "full_layer"),
    )
    contrast_index = 0
    for policy in POLICIES:
        for left, right in arm_pairs:
            effects = [
                template_values[(policy, left, task_type)]
                - template_values[(policy, right, task_type)]
                for task_type in templates
            ]
            name = f"{policy}:{left}-minus-{right}"
            comparisons["within_policy_arms"][name] = _paired_summary(
                effects,
                seed=BOOTSTRAP_SEED + contrast_index,
                bootstrap_resamples=bootstrap_resamples,
            )
            contrast_index += 1
    for arm in ARMS:
        for left, right in policy_pairs:
            effects = [
                template_values[(left, arm, task_type)]
                - template_values[(right, arm, task_type)]
                for task_type in templates
            ]
            name = f"{arm}:{left}-minus-{right}"
            comparisons["fixed_arm_policies"][name] = _paired_summary(
                effects,
                seed=BOOTSTRAP_SEED + contrast_index,
                bootstrap_resamples=bootstrap_resamples,
            )
            contrast_index += 1
    return {
        "ready": True,
        "template_count": len(templates),
        "instance_count": sum(len(indices) for indices in template_indices.values()),
        "cell_metrics": cell_metrics,
        "paired_comparisons": comparisons,
    }


def aggregate_matrix(
    attempts: Sequence[Attempt],
    *,
    roster: Mapping[tuple[str, int], Mapping[str, Any]],
    rejected: Sequence[Mapping[str, str]] = (),
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    selected, audit = resolve_attempts(attempts, roster=roster)
    audit["rejected_or_malformed_count"] = len(rejected)
    audit["rejected_or_malformed"] = list(rejected)
    templates = sorted({task_type for task_type, _ in roster})
    headline_indices = {task_type: TASK_INDICES for task_type in templates}
    action_indices = {
        task_type: TASK_INDICES
        for task_type in templates
        if task_type not in ACTION_INCOMPATIBLE_TASKS
    }
    hard_deleted = {
        (row["task_type"], row["task_index"])
        for row in audit["policy_agnostic_env_init_failure_instances"]
    }
    hard_delete_indices = {
        task_type: tuple(
            index
            for index in TASK_INDICES
            if (task_type, index) not in hard_deleted
        )
        for task_type in templates
    }
    hard_delete_indices = {
        task_type: indices
        for task_type, indices in hard_delete_indices.items()
        if indices
    }
    views = {
        "headline_all_116": aggregate_view(
            selected,
            template_indices=headline_indices,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "action_compatible_107": aggregate_view(
            selected,
            template_indices=action_indices,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "hard_delete_env_init_only": aggregate_view(
            selected,
            template_indices=hard_delete_indices,
            bootstrap_resamples=bootstrap_resamples,
        ),
    }
    complete = (
        audit["formal_cell_count"] == audit["expected_cell_count"]
        and audit["missing_cell_count"] == 0
        and audit["unauthorized_noninfra_duplicate_count"] == 0
        and audit["unexpected_cell_count"] == 0
        and audit["rejected_or_malformed_count"] == 0
        and views["headline_all_116"]["ready"]
    )
    return {
        "schema_version": "causalcache.sealed_zero_shot_matrix_v1",
        "status": (
            "COMPLETE_SEALED_ZERO_SHOT_MATRIX_V1"
            if complete
            else "INCOMPLETE_SEALED_ZERO_SHOT_MATRIX_V1"
        ),
        "protocol": {
            "policies": list(POLICIES),
            "arms": list(ARMS),
            "task_indices": list(TASK_INDICES),
            "expected_templates": 116,
            "expected_instances": 232,
            "expected_cells": 2088,
            "bootstrap": {
                "confidence": BOOTSTRAP_CONFIDENCE,
                "interval": "percentile",
                "quantile": "Hyndman_Fan_type_7",
                "resamples": bootstrap_resamples,
                "seed_family_start": BOOTSTRAP_SEED,
                "unit": "template_with_within_template_instance_mean",
            },
            "action_incompatible_tasks": sorted(ACTION_INCOMPATIBLE_TASKS),
        },
        "audit": audit,
        "views": views,
    }


__all__ = [
    "ACTION_INCOMPATIBLE_TASKS",
    "ARMS",
    "Attempt",
    "POLICIES",
    "TASK_INDICES",
    "aggregate_matrix",
    "aggregate_view",
    "classify_policy",
    "expected_keys",
    "load_attempts",
    "load_roster",
    "normalize_attempt",
    "paired_bootstrap_interval",
    "resolve_attempts",
    "sha256_file",
    "type7_quantile",
]
