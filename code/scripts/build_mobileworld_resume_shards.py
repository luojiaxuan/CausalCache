#!/usr/bin/env python3
"""Freeze disjoint MobileWorld task subsets around already completed results."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_SCHEMA = "causalcache.mobileworld.benchmark_config.v1"
TASK_SUBSET_SCHEMA = "causalcache.mobileworld.task_subset.v1"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _task_names_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--completed-trajectories", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("code/configs/causalcache_mobileworld_memory_v1.json"),
    )
    args = parser.parse_args()

    if args.shard_count < 2:
        raise ValueError("MobileWorld resume requires at least two shards")
    repository_root = args.repository_root.expanduser().resolve()
    completed_root = args.completed_trajectories.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repository_root / config_path
    config = _load_object(config_path.resolve())
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("MobileWorld benchmark config schema drifted")
    plan_path = repository_root / config["upstream"]["memory_split"]
    plan = _load_object(plan_path)
    roster = list(plan["benchmark_profiles"]["frozen_gui_owl_gui_only"])
    if len(roster) != config["benchmark"]["expected_denominator"]:
        raise ValueError("MobileWorld frozen GUI-only denominator drifted")

    completed = sorted(
        path.parent.name for path in completed_root.glob("*/result.txt")
    )
    if len(completed) != len(set(completed)):
        raise ValueError("MobileWorld completed task inventory contains duplicates")
    unknown = sorted(set(completed) - set(roster))
    if unknown:
        raise ValueError(f"completed inventory contains unknown tasks: {unknown}")
    completed_set = set(completed)
    pending = [task for task in roster if task not in completed_set]
    if not pending:
        raise ValueError("MobileWorld full roster is already complete")

    created_at = datetime.now(timezone.utc).isoformat()
    common = {
        "schema_version": TASK_SUBSET_SCHEMA,
        "created_at": created_at,
        "full_roster_count": len(roster),
        "full_roster_sha256": _task_names_sha256(roster),
        "completed_count_at_freeze": len(completed),
        "completed_task_names_sha256": _task_names_sha256(completed),
        "pending_count_at_freeze": len(pending),
        "pending_task_names_sha256": _task_names_sha256(pending),
        "shard_count": args.shard_count,
    }
    for shard_index in range(args.shard_count):
        tasks = pending[shard_index :: args.shard_count]
        _atomic_json(
            output_root / f"shard-{shard_index}.json",
            {
                **common,
                "shard_index": shard_index,
                "task_count": len(tasks),
                "task_names_sha256": _task_names_sha256(tasks),
                "tasks": tasks,
            },
        )
    print(
        json.dumps(
            {
                "completed_count": len(completed),
                "pending_count": len(pending),
                "shard_counts": [
                    len(pending[index :: args.shard_count])
                    for index in range(args.shard_count)
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
