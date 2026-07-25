#!/usr/bin/env python3
"""Run a frozen MobileWorld roster against one shared GUI-Owl policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_SCHEMA = "causalcache.mobileworld.benchmark_config.v1"
EXECUTION_SCHEMA = "causalcache.mobileworld.execution_record.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    keys = ("index", "name", "uuid", "memory_total_mib", "memory_used_mib", "utilization_percent")
    return [
        dict(zip(keys, (part.strip() for part in line.split(",")), strict=True))
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _runtime_identity() -> dict[str, Any]:
    versions = {}
    for package in ("mobile-world", "torch", "transformers"):
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


def _policy_health(endpoint: str) -> dict[str, Any]:
    url = endpoint.rstrip("/")
    if url.endswith("/act"):
        url = url[:-4]
    with urllib.request.urlopen(f"{url}/health", timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    if response.status != 200 or value.get("status") != "ready":
        raise RuntimeError("MobileWorld shared policy is not ready")
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _evenly_spaced(values: list[str], count: int) -> list[str]:
    if count <= 0 or count > len(values):
        raise ValueError("capacity task count must be within the roster")
    if count == 1:
        return [values[0]]
    return [
        values[round(index * (len(values) - 1) / (count - 1))]
        for index in range(count)
    ]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--mobileworld-root", type=Path, required=True)
    parser.add_argument("--fleet-manifest", type=Path, required=True)
    parser.add_argument("--policy-endpoint", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", choices=("capacity", "full"), required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("code/configs/causalcache_mobileworld_memory_v1.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    repository_root = args.repository_root.expanduser().resolve()
    mobileworld_root = args.mobileworld_root.expanduser().resolve()
    fleet_path = args.fleet_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repository_root / config_path
    config_path = config_path.resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("MobileWorld benchmark config schema drifted")
    if _git_revision(mobileworld_root) != config["upstream"]["revision"]:
        raise ValueError("MobileWorld upstream revision drifted")
    plan_path = repository_root / config["upstream"]["memory_split"]
    plan = _load_json(plan_path)
    roster = list(plan["benchmark_profiles"]["frozen_gui_owl_gui_only"])
    if len(roster) != config["benchmark"]["expected_denominator"]:
        raise ValueError("MobileWorld frozen GUI-only denominator drifted")
    fleet = _load_json(fleet_path)
    if fleet.get("image") != config["environment"]["image"]:
        raise ValueError("MobileWorld fleet image differs from frozen config")
    containers = fleet.get("containers")
    if not isinstance(containers, list) or len(containers) < args.num_envs:
        raise ValueError("MobileWorld fleet has fewer environments than requested")
    selected_containers = containers[: args.num_envs]
    if any(record.get("ready") is not True for record in selected_containers):
        raise RuntimeError("MobileWorld fleet contains an unready environment")
    if args.profile == "full":
        tasks = roster
        profile = config["benchmark"]
    else:
        profile = config["capacity"]
        tasks = _evenly_spaced(roster, profile["task_count_per_point"])
    aw_urls = [record["backend_url"] for record in selected_containers]
    agent_path = (
        repository_root / "code/integrations/mobileworld_gui_owl_agent.py"
    ).resolve()
    log_root = output_root / "trajectories"
    invocation = {
        "agent_type": str(agent_path),
        "model_name": "frozen-gui-owl-1.5-8b",
        "llm_base_url": args.policy_endpoint,
        "log_file_root": str(log_root),
        "tasks": tasks,
        "max_step": profile["max_round"],
        "aw_urls": aw_urls,
        "api_key": "empty",
        "device": "emulator-5554",
        "step_wait_time": profile["step_wait_time_seconds"],
        "suite_family": "mobile_world",
        "enable_mcp": False,
        "enable_user_interaction": False,
        "max_concurrency": args.num_envs,
        "shuffle_tasks": False,
        "auto_retry": profile.get("auto_retry", 0),
        "memory_arm": config["policy"]["memory_arm"],
        "memory_budget": config["policy"]["memory_budget"],
    }
    record: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA,
        "status": "RUNNING_MOBILEWORLD_GUI_OWL",
        "profile": args.profile,
        "started_at": _utc_now(),
        "repository_revision": _git_revision(repository_root),
        "mobileworld_revision": _git_revision(mobileworld_root),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "memory_plan_path": str(plan_path),
        "memory_plan_sha256": _sha256(plan_path),
        "fleet_manifest_path": str(fleet_path),
        "fleet_manifest_sha256": _sha256(fleet_path),
        "task_count": len(tasks),
        "task_names_sha256": hashlib.sha256(
            json.dumps(tasks, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "num_envs": args.num_envs,
        "policy_endpoint": args.policy_endpoint,
        "argv": list(sys.argv),
        "runtime_identity": _runtime_identity(),
        "policy_before": _policy_health(args.policy_endpoint),
        "gpu_before": _gpu_snapshot(),
        "invocation": invocation,
    }
    _atomic_json(output_root / "execution-record.json", record)
    sys.path.insert(0, str(mobileworld_root / "src"))
    from mobile_world.core.runner import run_agent_with_evaluation

    started = time.perf_counter()
    try:
        results, missing = run_agent_with_evaluation(**invocation)
    except BaseException as error:
        record.update(
            {
                "status": "FAILED_MOBILEWORLD_GUI_OWL",
                "error_type": error.__class__.__name__,
                "error": str(error),
            }
        )
        raise
    else:
        scores = [float(result["score"]) for result in results]
        record.update(
            {
                "status": "COMPLETE_MOBILEWORLD_GUI_OWL",
                "tasks_with_results": len(results),
                "tasks_without_results": len(missing),
                "missing_tasks": missing,
                "successful_tasks": sum(score > 0.99 for score in scores),
                "positive_score_tasks": sum(score > 0 for score in scores),
                "mean_score": sum(scores) / len(scores) if scores else None,
                "results": results,
            }
        )
    finally:
        record.update(
            {
                "finished_at": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "policy_after": _safe_policy_health(args.policy_endpoint),
                "gpu_after": _gpu_snapshot(),
            }
        )
        _atomic_json(output_root / "execution-record.json", record)
    print(json.dumps(record, sort_keys=True))
    if record.get("tasks_without_results"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
