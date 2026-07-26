#!/usr/bin/env python3
"""Run OSWorld 2.0 tasks with concurrent environments and one shared GUI-Owl."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import queue
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causalcache.osworld_v2 import (
    GUIOwlOSWorldV2Agent,
    load_osworld_v2_memory_plan,
    validate_osworld_v2_checkout,
)
from causalcache.osworld import configure_osworld_docker_runtime


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "causalcache.osworld_v2.benchmark_config.v1":
        raise ValueError("OSWorld 2.0 benchmark config schema drifted")
    if value.get("release", {}).get("task_count") != 108:
        raise ValueError("OSWorld 2.0 benchmark task count drifted")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gpu_snapshot() -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return [{"error": result.stderr.strip()}]
    keys = (
        "index",
        "name",
        "uuid",
        "memory_total_mib",
        "memory_used_mib",
        "utilization_percent",
    )
    return [
        dict(zip(keys, (part.strip() for part in line.split(",")), strict=True))
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _policy_health(endpoint: str) -> dict[str, Any]:
    url = endpoint.rstrip("/")
    if url.endswith("/act"):
        url = url[:-4]
    with urllib.request.urlopen(f"{url}/health", timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    if response.status != 200 or value.get("status") != "ready":
        raise RuntimeError("OSWorld 2.0 shared policy is not ready")
    return value


def _safe_policy_health(endpoint: str) -> dict[str, Any]:
    try:
        return _policy_health(endpoint)
    except Exception as error:  # noqa: BLE001
        return {
            "status": "unavailable",
            "error_type": error.__class__.__name__,
            "error": str(error),
        }


def _logical_shard(
    task_ids: list[str],
    *,
    shard_count: int,
    shard_index: int,
) -> list[str]:
    if shard_count <= 0:
        raise ValueError("OSWorld 2.0 shard count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("OSWorld 2.0 shard index must be within shard count")
    return task_ids[shard_index::shard_count]


def _task_selection(
    task_ids: list[str],
    *,
    shard_count: int,
    shard_index: int,
    explicit_task_ids: list[str] | None,
) -> list[str]:
    if explicit_task_ids is None:
        return _logical_shard(
            task_ids,
            shard_count=shard_count,
            shard_index=shard_index,
        )
    if shard_count != 1 or shard_index != 0:
        raise ValueError("explicit OSWorld 2.0 task ids cannot be combined with shards")
    if len(explicit_task_ids) != len(set(explicit_task_ids)):
        raise ValueError("explicit OSWorld 2.0 task ids must be unique")
    unknown = sorted(set(explicit_task_ids) - set(task_ids))
    if unknown:
        raise ValueError(f"explicit OSWorld 2.0 task ids are outside split: {unknown}")
    return list(explicit_task_ids)


def _runtime_identity() -> dict[str, Any]:
    versions = {}
    for package in ("gymnasium", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": versions,
    }


def _local_asset_base_url(path: str | Path) -> str:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("OSWorld 2.0 local asset root must be a directory")
    return root.as_uri()


def _import_upstream(
    root: Path,
    *,
    docker_dns_server: str,
    docker_cpu_model: str | None,
    docker_port_lock_timeout_seconds: int,
) -> tuple[Any, Any, Any, Any]:
    configure_osworld_docker_runtime(
        root,
        dns_server=docker_dns_server,
        cpu_model=docker_cpu_model,
        port_lock_timeout_seconds=docker_port_lock_timeout_seconds,
    )
    upstream_paths = (str(root / "scripts/python"), str(root))
    for path in reversed(upstream_paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    # note (luojiaxuan): gated V2 task classes import evaluation_examples lazily
    # during load_task_config, so both upstream paths must remain visible for the
    # full worker lifetime rather than only while importing the runner modules.
    desktop_module = importlib.import_module("desktop_env.desktop_env")
    loader_module = importlib.import_module("task_loader")
    runner_module = importlib.import_module("lib_run_single")
    return (
        desktop_module.DesktopEnv,
        loader_module.resolve_task_json_path,
        loader_module.load_task_config,
        runner_module.run_single_example,
    )


def _result_directory(
    root: Path,
    *,
    split: str,
    task_id: str,
) -> Path:
    return root / "pyautogui" / "screenshot" / "frozen_gui_owl" / split / task_id


def _worker(
    worker_id: int,
    task_queue: Any,
    result_queue: Any,
    spec: dict[str, Any],
) -> None:
    osworld_root = Path(spec["osworld_root"])
    os.environ["OSWORLD_FILE_BASE_URL"] = _local_asset_base_url(spec["assets_root"])
    os.environ["WEBSITE_HOST_SUFFIX"] = spec["website_host_suffix"]
    if spec["docker_host"] is not None:
        os.environ["DOCKER_HOST"] = spec["docker_host"]
    DesktopEnv, resolve_task_json_path, load_task_config, run_single_example = (
        _import_upstream(
            osworld_root,
            docker_dns_server=spec["docker_dns_server"],
            docker_cpu_model=spec["docker_cpu_model"],
            docker_port_lock_timeout_seconds=spec[
                "docker_port_lock_timeout_seconds"
            ],
        )
    )
    env = None
    try:
        env = DesktopEnv(
            provider_name=spec["provider"],
            path_to_vm=spec["path_to_vm"],
            region=spec["region"],
            action_space="pyautogui",
            screen_size=tuple(spec["screen_size"]),
            headless=True,
            os_type="Ubuntu",
            require_a11y_tree=False,
            enable_proxy=True,
            client_password=spec["client_password"],
            force_disable_vnc=True,
            force_disable_recording=True,
        )
        agent = GUIOwlOSWorldV2Agent(
            policy_endpoint=spec["policy_endpoint"],
            memory_arm=spec["memory_arm"],
            memory_budget=spec["memory_budget"],
            screen_size=tuple(spec["screen_size"]),
            timeout_seconds=spec["policy_timeout_seconds"],
        )
        while True:
            task_id = task_queue.get()
            if task_id is None:
                break
            result_dir = _result_directory(
                Path(spec["result_root"]),
                split=spec["split"],
                task_id=task_id,
            )
            result_file = result_dir / "result.txt"
            if result_file.exists():
                result_queue.put(
                    {
                        "status": "resumed_skip",
                        "worker_id": worker_id,
                        "task_id": task_id,
                    }
                )
                continue
            started = time.perf_counter()
            try:
                config_file = resolve_task_json_path(
                    task_id=task_id,
                    base_dir=str(osworld_root / "evaluation_examples"),
                    domain="tasks",
                    eval_version="v2",
                )
                example = load_task_config(
                    config_file,
                    task_id=task_id,
                    base_dir=str(osworld_root / "evaluation_examples"),
                    domain="tasks",
                    eval_version="v2",
                )
                result_dir.mkdir(parents=True, exist_ok=True)
                scores: list[float] = []
                run_args = argparse.Namespace(
                    sleep_after_execution=spec["sleep_after_execution_seconds"],
                    result_dir=spec["result_root"],
                    checkpoint_eval_mode="off",
                    trace_guest=False,
                )
                run_single_example(
                    agent,
                    env,
                    example,
                    spec["max_steps"],
                    example["instruction"],
                    run_args,
                    str(result_dir),
                    scores,
                )
                score = scores[-1] if scores else None
                result_queue.put(
                    {
                        "status": "completed",
                        "worker_id": worker_id,
                        "task_id": task_id,
                        "score": score,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
            except Exception as error:  # noqa: BLE001
                result_queue.put(
                    {
                        "status": "failed",
                        "worker_id": worker_id,
                        "task_id": task_id,
                        "error_type": error.__class__.__name__,
                        "error": str(error),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
    except BaseException as error:
        result_queue.put(
            {
                "status": "worker_failed",
                "worker_id": worker_id,
                "error_type": error.__class__.__name__,
                "error": str(error),
            }
        )
        raise
    finally:
        if env is not None:
            env.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("code/configs/causalcache_osworld_v2_memory_v1.json"),
    )
    parser.add_argument("--split", choices=("full", "memory_core", "memory_stress_union", "non_memory_control"), default="memory_core")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--policy-endpoint", required=True)
    parser.add_argument("--policy-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--docker-host")
    parser.add_argument("--path-to-vm")
    parser.add_argument("--region")
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output-root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    repository_root = args.repository_root.expanduser().resolve()
    osworld_root = args.osworld_root.expanduser().resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repository_root / config_path
    config = _load_config(config_path)
    execution = config["execution"]
    assets_root = (
        args.assets_root or Path(execution["assets_root"])
    ).expanduser().resolve()
    split_path = repository_root / config["release"]["memory_split"]
    plan = load_osworld_v2_memory_plan(split_path)
    readiness = validate_osworld_v2_checkout(
        osworld_root,
        require_task_classes=not args.preflight,
        require_assets=not args.preflight,
        assets_root=assets_root,
    )
    task_ids = (
        [record["task_id"] for record in plan["records"]]
        if args.split == "full"
        else list(plan["splits"][args.split])
    )
    unsharded_task_count = len(task_ids)
    task_ids = _task_selection(
        task_ids,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        explicit_task_ids=args.task_ids,
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("OSWorld 2.0 limit must be positive")
        task_ids = task_ids[: args.limit]
    num_envs = args.num_envs or execution["num_envs"]
    if num_envs <= 0:
        raise ValueError("OSWorld 2.0 environment count must be positive")
    output_root = (
        args.output_root or Path(execution["result_root"])
    ).expanduser().resolve()
    preflight = {
        "status": (
            "VALID_OSWORLD_V2_MEMORY_PREFLIGHT"
            if readiness["task_classes_ready"] and readiness["assets_ready"]
            else "BLOCKED_OSWORLD_V2_GATED_SUBSTRATE"
        ),
        "readiness": readiness,
        "split": args.split,
        "unsharded_task_count": unsharded_task_count,
        "task_count": len(task_ids),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "num_envs": num_envs,
        "policy_endpoint": args.policy_endpoint,
        "docker_host": args.docker_host,
        "output_root": str(output_root),
        "assets_root": str(assets_root),
    }
    if args.preflight:
        print(json.dumps(preflight, sort_keys=True))
        return

    context = multiprocessing.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    for task_id in task_ids:
        task_queue.put(task_id)
    active_envs = min(num_envs, len(task_ids))
    for _ in range(active_envs):
        task_queue.put(None)
    spec = {
        "osworld_root": str(osworld_root),
        "result_root": str(output_root),
        "split": args.split,
        "provider": execution["provider"],
        "docker_host": args.docker_host,
        "docker_dns_server": execution["docker_dns_server"],
        "docker_cpu_model": execution["docker_cpu_model"],
        "docker_port_lock_timeout_seconds": execution[
            "docker_port_lock_timeout_seconds"
        ],
        "path_to_vm": args.path_to_vm,
        "region": args.region,
        "screen_size": execution["screen_size"],
        "client_password": execution["client_password"],
        "max_steps": execution["max_steps"],
        "sleep_after_execution_seconds": execution[
            "sleep_after_execution_seconds"
        ],
        "policy_endpoint": args.policy_endpoint,
        "policy_timeout_seconds": args.policy_timeout_seconds,
        "memory_arm": config["policy"]["memory_arm"],
        "memory_budget": config["policy"]["memory_budget"],
        "assets_root": str(assets_root),
        "website_host_suffix": execution["website_host_suffix"],
    }
    workers = [
        context.Process(
            target=_worker,
            args=(worker_id, task_queue, result_queue, spec),
            name=f"osworld-v2-env-{worker_id}",
        )
        for worker_id in range(active_envs)
    ]
    started = time.perf_counter()
    started_at = _utc_now()
    policy_before = _policy_health(args.policy_endpoint)
    gpu_before = _gpu_snapshot()
    for worker in workers:
        worker.start()
    messages = []
    terminal_task_ids: set[str] = set()
    while len(terminal_task_ids) < len(task_ids):
        try:
            message = result_queue.get(timeout=10)
        except queue.Empty:
            if any(worker.is_alive() for worker in workers):
                continue
            break
        messages.append(message)
        if message["status"] in {"completed", "failed", "resumed_skip"}:
            terminal_task_ids.add(message["task_id"])
    for worker in workers:
        worker.join()
    while True:
        try:
            messages.append(result_queue.get_nowait())
        except queue.Empty:
            break
    missing_task_ids = sorted(set(task_ids) - terminal_task_ids)
    messages.extend(
        {
            "status": "failed",
            "task_id": task_id,
            "error_type": "WorkerPoolExited",
            "error": "all OSWorld 2.0 workers exited before returning this task",
        }
        for task_id in missing_task_ids
    )
    summary = {
        "status": "COMPLETE_OSWORLD_V2_MEMORY_RUN",
        "split": args.split,
        "unsharded_task_count": unsharded_task_count,
        "task_count": len(task_ids),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "completed": sum(message["status"] == "completed" for message in messages),
        "resumed_skips": sum(
            message["status"] == "resumed_skip" for message in messages
        ),
        "failures": sum(message["status"] == "failed" for message in messages),
        "worker_failures": sum(
            message["status"] == "worker_failed" for message in messages
        ),
        "worker_exit_codes": [worker.exitcode for worker in workers],
        "elapsed_seconds": time.perf_counter() - started,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "repository_revision": _git_revision(repository_root),
        "osworld_revision": readiness["code_revision"],
        "argv": list(sys.argv),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "memory_plan_path": str(split_path),
        "memory_plan_sha256": _sha256(split_path),
        "assets_root": str(assets_root),
        "runtime_identity": _runtime_identity(),
        "gpu_before": gpu_before,
        "gpu_after": _gpu_snapshot(),
        "policy_before": policy_before,
        "policy_after": _safe_policy_health(args.policy_endpoint),
        "messages": messages,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_name = (
        f"summary-{args.split}.json"
        if args.shard_count == 1
        else (
            f"summary-{args.split}-shard-{args.shard_index:03d}"
            f"-of-{args.shard_count:03d}.json"
        )
    )
    (output_root / summary_name).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    if summary["failures"] or summary["worker_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
