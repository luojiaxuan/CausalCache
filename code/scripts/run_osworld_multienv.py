#!/usr/bin/env python3
"""Run a resumable OSWorld roster over concurrent KVM environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import queue
import subprocess
import time
from pathlib import Path
from typing import Any

from causalcache.osworld import (
    HTTPOSWorldPolicy,
    create_osworld_environment,
    load_osworld_task,
    run_osworld_episode,
)
from causalcache.osworld_benchmark import (
    load_osworld_benchmark_config,
    policy_endpoint_for_worker,
    validate_osworld_benchmark_roster,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _worker(
    worker_id: int,
    task_queue: Any,
    result_queue: Any,
    spec: dict[str, Any],
) -> None:
    endpoint = policy_endpoint_for_worker(spec["policy_endpoints"], worker_id)
    policy = HTTPOSWorldPolicy(endpoint, timeout_seconds=spec["policy_timeout_seconds"])
    environment = None
    try:
        while True:
            record = task_queue.get()
            if record is None:
                break
            domain, task_id = record
            try:
                if environment is None:
                    environment = create_osworld_environment(
                        spec["osworld_root"],
                        provider_name=spec["provider"],
                        region=None,
                        path_to_vm=spec["path_to_vm"],
                        cache_dir=spec["cache_dir"],
                        screen_size=tuple(spec["screen_size"]),
                        headless=True,
                        docker_dns_server=spec["docker_dns_server"],
                        docker_cpu_model=spec["docker_cpu_model"],
                    )
                task = load_osworld_task(
                    spec["osworld_root"], domain=domain, task_id=task_id
                )
                result = run_osworld_episode(
                    environment=environment,
                    policy=policy,
                    task=task,
                    output_root=spec["output_root"],
                    memory_arm=spec["memory_arm"],
                    memory_budget=spec["memory_budget"],
                    max_steps=spec["max_steps"],
                    pause_seconds=spec["pause_seconds"],
                    screen_size=tuple(spec["screen_size"]),
                    provenance={
                        **spec["provenance"],
                        "worker_id": worker_id,
                        "policy_endpoint": endpoint,
                    },
                    evaluate_at_end=spec["evaluate_at_end"],
                )
                result_queue.put(
                    {
                        "status": "completed",
                        "worker_id": worker_id,
                        "domain": domain,
                        "task_id": task_id,
                        "resumed_skip": bool(result.get("resumed_skip")),
                        "elapsed_seconds": float(result["elapsed_seconds"]),
                        "completed_steps": int(result["completed_steps"]),
                        "success": result["success"],
                        "policy_latencies_seconds": [
                            float(step["policy_latency_seconds"])
                            for step in result["steps"]
                        ],
                        "policy_queue_seconds": [
                            float(step["policy_response"]["queue_seconds"])
                            for step in result["steps"]
                            if "queue_seconds" in step["policy_response"]
                        ],
                        "policy_generation_seconds": [
                            float(step["policy_response"]["runtime"]["generation_seconds"])
                            for step in result["steps"]
                            if isinstance(step["policy_response"].get("runtime"), dict)
                            and "generation_seconds" in step["policy_response"]["runtime"]
                        ],
                    }
                )
            except Exception as error:
                result_queue.put(
                    {
                        "status": "failed",
                        "worker_id": worker_id,
                        "domain": domain,
                        "task_id": task_id,
                        "error_type": error.__class__.__name__,
                        "error_message": str(error),
                    }
                )
                if environment is not None:
                    environment.close()
                    environment = None
    except BaseException as error:
        result_queue.put(
            {
                "status": "worker_failed",
                "worker_id": worker_id,
                "error_type": error.__class__.__name__,
                "error_message": str(error),
            }
        )
        raise
    finally:
        if environment is not None:
            environment.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("code/configs/causalcache_osworld_benchmark_h100_v1.json"),
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--selection-mode", choices=("prefix", "evenly_spaced"), default="prefix"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--pause-seconds", type=float)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--policy-endpoint", action="append")
    return parser


def main() -> None:
    args = _parser().parse_args()
    repository_root = args.repository_root.expanduser().resolve()
    osworld_root = args.osworld_root.expanduser().resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repository_root / config_path
    config = load_osworld_benchmark_config(config_path)
    roster = validate_osworld_benchmark_roster(osworld_root, config)
    execution = config["execution"]
    policy_endpoints = tuple(
        args.policy_endpoint or config["policy_pool"]["endpoints"]
    )
    num_envs = execution["num_envs"] if args.num_envs is None else args.num_envs
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if args.policy_endpoint is None and len(policy_endpoints) != config["policy_pool"]["replica_count"]:
        raise ValueError("policy endpoint count drifted from replica_count")
    selected = roster.selected[: args.limit] if args.limit else roster.selected
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.selection_mode == "evenly_spaced" and args.limit:
        if args.limit > len(roster.selected):
            raise ValueError("evenly-spaced limit exceeds the selected roster")
        if args.limit == 1:
            selected = (roster.selected[0],)
        else:
            indices = [
                round(index * (len(roster.selected) - 1) / (args.limit - 1))
                for index in range(args.limit)
            ]
            selected = tuple(roster.selected[index] for index in indices)
    assignments = [
        {
            "worker_id": worker_id,
            "policy_endpoint": policy_endpoint_for_worker(
                policy_endpoints, worker_id
            ),
        }
        for worker_id in range(num_envs)
    ]
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "VALID_OSWORLD_MULTIENV_PREFLIGHT",
                    "full_tasks": len(roster.full),
                    "selected_tasks": len(roster.selected),
                    "excluded_tasks": len(roster.excluded),
                    "num_envs": num_envs,
                    "policy_replicas": len(policy_endpoints),
                    "worker_assignments": assignments,
                },
                sort_keys=True,
            )
        )
        return

    output_root = (args.output_root or Path(execution["output_root"])).resolve()
    cache_dir = (args.cache_dir or Path(execution["cache_dir"])).resolve()
    pending = []
    resumed_skips = 0
    for domain, task_id in selected:
        completion = output_root / domain / task_id / "result.json"
        if not completion.exists():
            pending.append((domain, task_id))
            continue
        value = json.loads(completion.read_text(encoding="utf-8"))
        if value.get("status") != "COMPLETE_OSWORLD_EPISODE":
            raise ValueError(f"invalid completion marker: {completion}")
        resumed_skips += 1
    if not pending:
        print(
            json.dumps(
                {
                    "status": "COMPLETE_OSWORLD_MULTIENV_BENCHMARK",
                    "selected_tasks": len(selected),
                    "completed": len(selected),
                    "resumed_skips": resumed_skips,
                },
                sort_keys=True,
            )
        )
        return

    active_envs = min(num_envs, len(pending))
    context = multiprocessing.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    for record in pending:
        task_queue.put(record)
    for _ in range(active_envs):
        task_queue.put(None)
    spec = {
        "repository_root": str(repository_root),
        "osworld_root": str(osworld_root),
        "output_root": str(output_root),
        "cache_dir": str(cache_dir),
        "provider": execution["provider"],
        "path_to_vm": execution["path_to_vm"],
        "screen_size": execution["screen_size"],
        "docker_dns_server": execution["docker_dns_server"],
        "docker_cpu_model": execution["docker_cpu_model"],
        "memory_arm": execution["memory_arm"],
        "memory_budget": execution["memory_budget"],
        "max_steps": (
            execution["max_steps"] if args.max_steps is None else args.max_steps
        ),
        "pause_seconds": (
            execution["pause_seconds"]
            if args.pause_seconds is None
            else args.pause_seconds
        ),
        "evaluate_at_end": (
            False
            if args.skip_evaluation
            else bool(execution.get("evaluate_at_end", True))
        ),
        "policy_endpoints": policy_endpoints,
        "policy_timeout_seconds": config["policy_pool"]["timeout_seconds"],
        "provenance": {
            "causalcache_git_revision": _git_revision(repository_root),
            "osworld_git_revision": config["osworld_revision"],
            "benchmark_config_path": str(config_path),
            "benchmark_config_sha256": _sha256_file(config_path),
            "roster_meta_path": config["roster"]["selected_meta_path"],
            "roster_task_count": len(roster.selected),
            "num_envs": num_envs,
            "policy_replica_count": len(policy_endpoints),
            "selection_mode": args.selection_mode,
            "evaluate_at_end": (
                False
                if args.skip_evaluation
                else bool(execution.get("evaluate_at_end", True))
            ),
        },
    }
    workers = [
        context.Process(
            target=_worker,
            args=(worker_id, task_queue, result_queue, spec),
            name=f"osworld-env-{worker_id}",
        )
        for worker_id in range(active_envs)
    ]
    benchmark_started = time.perf_counter()
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    messages = []
    while True:
        try:
            messages.append(result_queue.get_nowait())
        except queue.Empty:
            break
    benchmark_wall_seconds = time.perf_counter() - benchmark_started
    completed_messages = [message for message in messages if message["status"] == "completed"]
    failures = [message for message in messages if message["status"] != "completed"]
    worker_exit_codes = [worker.exitcode for worker in workers]
    complete = len(completed_messages) == len(pending) and not failures and all(
        code == 0 for code in worker_exit_codes
    )
    policy_latencies = [
        value
        for message in completed_messages
        for value in message["policy_latencies_seconds"]
    ]
    queue_latencies = [
        value
        for message in completed_messages
        for value in message["policy_queue_seconds"]
    ]
    generation_latencies = [
        value
        for message in completed_messages
        for value in message["policy_generation_seconds"]
    ]
    fresh_task_count = len(completed_messages)
    summary = {
        "status": (
            "COMPLETE_OSWORLD_MULTIENV_BENCHMARK"
            if complete
            else "PARTIAL_OSWORLD_MULTIENV_BENCHMARK"
        ),
        "selected_tasks": len(selected),
        "pending_at_start": len(pending),
        "completed": resumed_skips + len(completed_messages),
        "resumed_skips": resumed_skips,
        "active_envs": active_envs,
        "policy_replicas": len(policy_endpoints),
        "benchmark_wall_seconds": benchmark_wall_seconds,
        "fresh_tasks_per_hour": (
            fresh_task_count / benchmark_wall_seconds * 3600.0
            if benchmark_wall_seconds > 0
            else None
        ),
        "policy_requests": len(policy_latencies),
        "policy_requests_per_second": (
            len(policy_latencies) / benchmark_wall_seconds
            if benchmark_wall_seconds > 0
            else None
        ),
        "task_elapsed_seconds_sum": sum(
            message["elapsed_seconds"] for message in completed_messages
        ),
        "completed_steps": sum(
            message["completed_steps"] for message in completed_messages
        ),
        "successful_tasks": sum(
            message["success"] is True for message in completed_messages
        ),
        "policy_latency_seconds": {
            "p50": _percentile(policy_latencies, 0.50),
            "p95": _percentile(policy_latencies, 0.95),
            "max": max(policy_latencies) if policy_latencies else None,
        },
        "server_queue_seconds": {
            "p50": _percentile(queue_latencies, 0.50),
            "p95": _percentile(queue_latencies, 0.95),
            "max": max(queue_latencies) if queue_latencies else None,
        },
        "generation_seconds": {
            "p50": _percentile(generation_latencies, 0.50),
            "p95": _percentile(generation_latencies, 0.95),
            "max": max(generation_latencies) if generation_latencies else None,
        },
        "worker_assignments": assignments[:active_envs],
        "worker_exit_codes": worker_exit_codes,
        "failures": failures,
    }
    _atomic_json(output_root / "benchmark-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
