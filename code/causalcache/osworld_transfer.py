"""Reduction helpers for paired OSWorld policy-transfer experiments."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "causalcache.osworld_transfer_pilot.v1"
SUMMARY_SCHEMA_VERSION = "causalcache.osworld_transfer_summary.v1"


def load_transfer_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("OSWorld transfer config schema drifted")
    arms = [profile["arm"] for profile in config["profiles"]]
    if arms != ["frozen", "terminal_s60"]:
        raise ValueError("OSWorld transfer arms drifted")
    if config["selection"]["task_count"] <= 0:
        raise ValueError("OSWorld transfer task count must be positive")
    return config


def load_arm_results(
    root: Path,
    *,
    expected_profile_id: str,
    expected_task_count: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(root.glob("*/*/result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "COMPLETE_OSWORLD_EPISODE":
            raise ValueError(f"incomplete OSWorld result: {path}")
        task = value["task"]
        identity = (str(task["domain"]), str(task["task_id"]))
        if identity in results:
            raise ValueError(f"duplicate OSWorld task result: {identity}")
        step_profiles = {
            step["policy_response"].get("source") for step in value["steps"]
        }
        if step_profiles != {expected_profile_id}:
            raise ValueError(
                f"OSWorld policy profile drifted for {identity}: {step_profiles}"
            )
        results[identity] = value
    failures_by_task: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in sorted(root.glob("*/*/attempts/*/failure.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        task = value["task"]
        identity = (str(task["domain"]), str(task["task_id"]))
        failures_by_task.setdefault(identity, []).append(value)
    for identity, failures in failures_by_task.items():
        if identity in results:
            continue
        latest = max(failures, key=lambda value: str(value["failed_at"]))
        results[identity] = {
            "status": "FAILED_OSWORLD_EPISODE",
            "task": {"domain": identity[0], "task_id": identity[1]},
            "score": 0.0,
            "success": False,
            "completed_steps": int(latest["completed_steps"]),
            "termination_reason": f"run_failure:{latest['error_type']}",
            "steps": [],
            "counted_failure": {
                "error_type": latest["error_type"],
                "error_message": latest["error_message"],
                "failed_at": latest["failed_at"],
            },
        }
    if len(results) != expected_task_count:
        raise ValueError(
            f"OSWorld result count drifted: {len(results)} != {expected_task_count}"
        )
    return results


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    seed: int = 0,
    draws: int = 10000,
) -> list[float]:
    if not differences or draws <= 0:
        raise ValueError("paired bootstrap needs data and positive draws")
    generator = random.Random(seed)
    estimates = [
        sum(generator.choice(differences) for _ in differences) / len(differences)
        for _ in range(draws)
    ]
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _arm_summary(results: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    scores = [float(result["score"]) for result in results.values()]
    successes = [bool(result["success"]) for result in results.values()]
    steps = [int(result["completed_steps"]) for result in results.values()]
    return {
        "task_count": len(results),
        "mean_osworld_score": sum(scores) / len(scores),
        "task_success_count": sum(successes),
        "task_success_rate": sum(successes) / len(successes),
        "mean_completed_steps": sum(steps) / len(steps),
        "counted_failure_count": sum(
            "counted_failure" in result for result in results.values()
        ),
        "counted_failure_type_counts": dict(
            sorted(
                Counter(
                    result["counted_failure"]["error_type"]
                    for result in results.values()
                    if "counted_failure" in result
                ).items()
            )
        ),
        "termination_reason_counts": dict(
            sorted(Counter(result["termination_reason"] for result in results.values()).items())
        ),
    }


def reduce_transfer_results(
    *,
    config: Mapping[str, Any],
    arm_results: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
) -> dict[str, Any]:
    profiles = {profile["arm"]: profile for profile in config["profiles"]}
    if set(arm_results) != set(profiles):
        raise ValueError("OSWorld transfer result arms drifted")
    identities = set(arm_results["frozen"])
    if identities != set(arm_results["terminal_s60"]):
        raise ValueError("OSWorld transfer task identities are not paired")
    differences = [
        float(arm_results["terminal_s60"][identity]["score"])
        - float(arm_results["frozen"][identity]["score"])
        for identity in sorted(identities)
    ]
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "COMPLETE_OSWORLD_TRANSFER_PILOT",
        "task_count": len(identities),
        "arms": {
            arm: _arm_summary(arm_results[arm])
            for arm in ("frozen", "terminal_s60")
        },
        "paired": {
            "mean_score_delta_terminal_s60_minus_frozen": sum(differences)
            / len(differences),
            "mean_score_delta_bootstrap_95_ci": paired_bootstrap_ci(differences),
            "terminal_s60_wins": wins,
            "terminal_s60_losses": losses,
            "ties": len(differences) - wins - losses,
            "task_scores": [
                {
                    "domain": identity[0],
                    "task_id": identity[1],
                    "frozen": float(arm_results["frozen"][identity]["score"]),
                    "terminal_s60": float(
                        arm_results["terminal_s60"][identity]["score"]
                    ),
                    "delta": difference,
                }
                for identity, difference in zip(
                    sorted(identities), differences, strict=True
                )
            ],
        },
    }


__all__ = [
    "SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "load_arm_results",
    "load_transfer_config",
    "paired_bootstrap_ci",
    "reduce_transfer_results",
]
