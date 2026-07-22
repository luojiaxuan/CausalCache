"""Fail-closed OSWorld benchmark roster and policy-pool helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalcache.osworld import (
    OSWORLD_POLICY_REQUEST_SCHEMA_VERSION,
    load_osworld_inventory,
    osworld_git_revision,
)


OSWORLD_BENCHMARK_CONFIG_SCHEMA_VERSION = "causalcache.osworld.benchmark_config.v1"


@dataclass(frozen=True)
class ValidatedOSWorldRoster:
    full: tuple[tuple[str, str], ...]
    selected: tuple[tuple[str, str], ...]
    excluded: tuple[tuple[str, str], ...]


def load_osworld_benchmark_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != OSWORLD_BENCHMARK_CONFIG_SCHEMA_VERSION:
        raise ValueError("OSWorld benchmark config schema version drifted")
    return payload


def validate_osworld_benchmark_roster(
    osworld_root: str | Path,
    config: Mapping[str, Any],
) -> ValidatedOSWorldRoster:
    root = Path(osworld_root).expanduser().resolve()
    roster = config["roster"]
    revision = osworld_git_revision(root)
    if revision != config["osworld_revision"]:
        raise ValueError("OSWorld benchmark revision drifted")
    full = load_osworld_inventory(root, meta_path=roster["full_meta_path"])
    selected = load_osworld_inventory(root, meta_path=roster["selected_meta_path"])
    if len(full) != roster["expected_full_task_count"]:
        raise ValueError("OSWorld full roster count drifted")
    if len(selected) != roster["expected_selected_task_count"]:
        raise ValueError("OSWorld selected roster count drifted")
    full_set = set(full)
    selected_set = set(selected)
    if not selected_set < full_set:
        raise ValueError("OSWorld selected roster must be a strict subset of full")
    excluded = tuple(record for record in full if record not in selected_set)
    expected_excluded = tuple(
        (record["domain"], record["task_id"])
        for record in roster["excluded_tasks"]
    )
    if excluded != expected_excluded:
        raise ValueError("OSWorld excluded task identities drifted")
    if len(excluded) != roster["expected_excluded_task_count"]:
        raise ValueError("OSWorld excluded task count drifted")
    return ValidatedOSWorldRoster(full=full, selected=selected, excluded=excluded)


def policy_endpoint_for_worker(endpoints: Sequence[str], worker_id: int) -> str:
    if not endpoints or any(
        not isinstance(endpoint, str)
        or not endpoint.startswith(("http://", "https://"))
        for endpoint in endpoints
    ):
        raise ValueError("policy endpoints must be non-empty HTTP(S) URLs")
    if type(worker_id) is not int or worker_id < 0:
        raise ValueError("worker_id must be a non-negative integer")
    return endpoints[worker_id % len(endpoints)]


def smoke_policy_action(request: Mapping[str, Any]) -> dict[str, str]:
    if request.get("schema_version") != OSWORLD_POLICY_REQUEST_SCHEMA_VERSION:
        raise ValueError("OSWorld policy request schema version drifted")
    step_id = request.get("step_id")
    if type(step_id) is not int or step_id <= 0:
        raise ValueError("OSWorld policy request step_id must be positive")
    return {"type": "wait" if step_id == 1 else "done"}
