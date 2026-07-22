#!/usr/bin/env python3
"""Run resumable CausalCache episodes against the pinned OSWorld DesktopEnv."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from causalcache.osworld import (
    HTTPOSWorldPolicy,
    SUPPORTED_MEMORY_ARMS,
    ScriptedOSWorldPolicy,
    create_osworld_environment,
    import_osworld_desktop_env,
    load_osworld_inventory,
    load_osworld_task,
    osworld_git_revision,
    run_osworld_episode,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "causalcache.osworld.runner_config.v1":
        raise ValueError("OSWorld runner config schema version drifted")
    return payload


def _selected_tasks(
    inventory: tuple[tuple[str, str], ...],
    *,
    domain: str | None,
    task_id: str | None,
    run_all: bool,
    limit: int | None,
    shard_index: int,
    num_shards: int,
) -> tuple[tuple[str, str], ...]:
    if not run_all and task_id is None:
        raise ValueError("pass --task-id for one task or --all for a suite")
    if task_id is not None and run_all:
        raise ValueError("--task-id and --all are mutually exclusive")
    if task_id is not None and domain is None:
        raise ValueError("--task-id requires --domain")
    if type(num_shards) is not int or num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if type(shard_index) is not int or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must fall within [0, num_shards)")
    selected = [
        record
        for record in inventory
        if (domain is None or record[0] == domain)
        and (task_id is None or record[1] == task_id)
    ]
    selected = [
        record for index, record in enumerate(selected) if index % num_shards == shard_index
    ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("task selection is empty")
    return tuple(selected)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("code/configs/causalcache_osworld_runner_v1.json"),
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--domain")
    parser.add_argument("--task-id")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--provider")
    parser.add_argument("--region")
    parser.add_argument("--path-to-vm")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction)
    parser.add_argument("--memory-arm", choices=SUPPORTED_MEMORY_ARMS)
    parser.add_argument("--memory-budget", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--pause-seconds", type=float)
    parser.add_argument("--policy", choices=("http", "scripted"), default="http")
    parser.add_argument("--policy-url")
    parser.add_argument("--policy-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--scripted-actions", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    repository_root = args.repository_root.expanduser().resolve()
    osworld_root = args.osworld_root.expanduser().resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repository_root / config_path
    config = _load_config(config_path)
    revision = osworld_git_revision(osworld_root)
    expected_revision = config["osworld"]["revision"]
    if revision != expected_revision:
        raise ValueError(
            f"OSWorld revision drifted: expected {expected_revision}, observed {revision}"
        )
    inventory = load_osworld_inventory(
        osworld_root, meta_path=config["osworld"]["task_meta_path"]
    )
    if args.preflight:
        desktop_env = import_osworld_desktop_env(osworld_root)
        print(
            json.dumps(
                {
                    "status": "VALID_OSWORLD_PREFLIGHT",
                    "osworld_revision": revision,
                    "task_count": len(inventory),
                    "domain_count": len({domain for domain, _ in inventory}),
                    "desktop_env_module": desktop_env.__module__,
                },
                sort_keys=True,
            )
        )
        return

    selected = _selected_tasks(
        inventory,
        domain=args.domain,
        task_id=args.task_id,
        run_all=args.all,
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.dry_run:
        tasks = [
            load_osworld_task(osworld_root, domain=domain, task_id=task_id)
            for domain, task_id in selected
        ]
        print(
            json.dumps(
                {
                    "status": "VALID_OSWORLD_DRY_RUN",
                    "osworld_revision": revision,
                    "selected_tasks": len(tasks),
                    "tasks": [
                        {"domain": task.domain, "task_id": task.task_id}
                        for task in tasks
                    ],
                },
                sort_keys=True,
            )
        )
        return

    runner = config["runner"]
    output_root = (args.output_root or Path(runner["output_root"])).expanduser().resolve()
    cache_dir = (args.cache_dir or Path(runner["cache_dir"])).expanduser().resolve()
    provider = args.provider or runner["provider"]
    screen_size = tuple(runner["screen_size"])
    headless = runner["headless"] if args.headless is None else args.headless
    memory_arm = args.memory_arm or runner["memory_arm"]
    memory_budget = (
        runner["memory_budget"] if args.memory_budget is None else args.memory_budget
    )
    max_steps = runner["max_steps"] if args.max_steps is None else args.max_steps
    pause_seconds = (
        runner["pause_seconds"]
        if args.pause_seconds is None
        else args.pause_seconds
    )
    if args.policy == "http":
        if not args.policy_url:
            raise ValueError("HTTP policy requires --policy-url")
        policy = HTTPOSWorldPolicy(
            args.policy_url, timeout_seconds=args.policy_timeout_seconds
        )
    else:
        if args.scripted_actions is None:
            raise ValueError("scripted policy requires --scripted-actions")
        policy = ScriptedOSWorldPolicy.from_json(args.scripted_actions)

    provenance = {
        "causalcache_git_revision": _git_revision(repository_root),
        "osworld_git_revision": revision,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "provider": provider,
        "screen_size": list(screen_size),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "policy_transport": args.policy,
    }
    pending: list[tuple[str, str]] = []
    for record in selected:
        completion_path = output_root / record[0] / record[1] / "result.json"
        if not completion_path.exists():
            pending.append(record)
            continue
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") != "COMPLETE_OSWORLD_EPISODE":
            raise ValueError(f"invalid completion marker: {completion_path}")
    if not pending:
        print(
            json.dumps(
                {
                    "status": "COMPLETE_OSWORLD_SHARD",
                    "selected_tasks": len(selected),
                    "completed": len(selected),
                    "resumed_skips": len(selected),
                },
                sort_keys=True,
            )
        )
        return

    environment = create_osworld_environment(
        osworld_root,
        provider_name=provider,
        region=args.region,
        path_to_vm=args.path_to_vm,
        cache_dir=cache_dir,
        screen_size=screen_size,
        headless=headless,
    )
    completed = 0
    resumed_skips = 0
    failures: list[dict[str, str]] = []
    try:
        for domain, task_id in selected:
            task = load_osworld_task(osworld_root, domain=domain, task_id=task_id)
            try:
                result = run_osworld_episode(
                    environment=environment,
                    policy=policy,
                    task=task,
                    output_root=output_root,
                    memory_arm=memory_arm,
                    memory_budget=memory_budget,
                    max_steps=max_steps,
                    pause_seconds=pause_seconds,
                    screen_size=screen_size,
                    provenance=provenance,
                )
                completed += 1
                resumed_skips += int(result.get("resumed_skip", False))
            except Exception as error:
                failures.append(
                    {
                        "domain": domain,
                        "task_id": task_id,
                        "error_type": error.__class__.__name__,
                        "error_message": str(error),
                    }
                )
    finally:
        environment.close()
    summary = {
        "status": (
            "COMPLETE_OSWORLD_SHARD" if not failures else "PARTIAL_OSWORLD_SHARD"
        ),
        "selected_tasks": len(selected),
        "completed": completed,
        "resumed_skips": resumed_skips,
        "failures": failures,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
