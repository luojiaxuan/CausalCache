#!/usr/bin/env python3
"""Run the H100 OSWorld capacity matrix with per-point host telemetry."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "causalcache.osworld.capacity_sweep.v1"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _container_ip(name: str) -> str:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "-f",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            name,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not value:
        raise RuntimeError("policy container has no bridge IP")
    return value


def _health(endpoint: str) -> dict[str, Any]:
    with urllib.request.urlopen(endpoint, timeout=5) as response:
        value = json.loads(response.read().decode("utf-8"))
    if value.get("status") != "ready":
        raise RuntimeError(f"policy replica is not ready: {endpoint}")
    return value


def _stop(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--venv-python", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--driver-image", default="hongccc/sglang-omni:dev")
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    repository_root = args.repository_root.resolve()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("capacity sweep schema version drifted")
    if config.get("evaluate_at_end") is not False:
        raise ValueError("capacity sweep must disable task evaluators")
    ports = config["policy_ports"]
    if len(ports) != config["maximum_policy_replicas"] or len(set(ports)) != len(ports):
        raise ValueError("capacity policy port inventory drifted")
    policy_ip = _container_ip(config["policy_container"])
    health = [
        _health(f"http://{policy_ip}:{port}/health")
        for port in ports
    ]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING_OSWORLD_CAPACITY_SWEEP",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "causalcache_git_revision": revision,
        "policy_ip": policy_ip,
        "policy_health": health,
        "points": [],
        "invalid_point_count": 0,
    }
    _atomic_json(output_root / "capacity-sweep.json", aggregate)

    for point_index, point in enumerate(config["points"]):
        replicas = int(point["policy_replicas"])
        num_envs = int(point["num_envs"])
        tasks = int(point["tasks"])
        if not 1 <= replicas <= config["maximum_policy_replicas"]:
            raise ValueError("capacity point exceeds the policy replica cap")
        if tasks < num_envs:
            raise ValueError("capacity point must give every environment at least one task")
        point_id = f"{point_index:02d}-{point['axis']}-g{replicas}-e{num_envs}-t{tasks}"
        point_root = output_root / "points" / point_id
        point_root.mkdir(parents=True, exist_ok=False)
        gpu_log = (point_root / "nvidia-smi.csv").open("w", encoding="utf-8")
        cpu_log = (point_root / "vmstat.txt").open("w", encoding="utf-8")
        runner_log = (point_root / "runner.log").open("w", encoding="utf-8")
        gpu_sampler: subprocess.Popen[Any] | None = None
        cpu_sampler: subprocess.Popen[Any] | None = None
        started = time.perf_counter()
        return_code = -1
        try:
            gpu_sampler = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                    "-l",
                    "1",
                ],
                stdout=gpu_log,
                stderr=subprocess.STDOUT,
            )
            cpu_sampler = subprocess.Popen(
                ["vmstat", "1"], stdout=cpu_log, stderr=subprocess.STDOUT
            )
            policy_arguments = [
                value
                for port in ports[:replicas]
                for value in (
                    "--policy-endpoint",
                    f"http://{policy_ip}:{port}/act",
                )
            ]
            capacity_config = repository_root / config["capacity_config"]
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                f"causalcache-jaxan-osworld-capacity-{point_id}",
                "--network",
                "host",
                "-v",
                "/data:/data",
                "-v",
                "/var/run/docker.sock:/var/run/docker.sock",
                "-v",
                "/dev/kvm:/dev/kvm",
                args.driver_image,
                "/bin/bash",
                "-lc",
                "exec \"$@\"",
                "capacity-driver",
                str(args.venv_python),
                str(repository_root / "code/scripts/run_osworld_multienv.py"),
                "--repository-root",
                str(repository_root),
                "--osworld-root",
                str(args.osworld_root.resolve()),
                "--config",
                str(capacity_config),
                "--num-envs",
                str(num_envs),
                "--limit",
                str(tasks),
                "--selection-mode",
                config["selection_mode"],
                "--output-root",
                str(point_root / "episodes"),
                "--cache-dir",
                str(args.cache_dir.resolve()),
                "--max-steps",
                str(config["max_steps"]),
                "--pause-seconds",
                str(config["pause_seconds"]),
                "--skip-evaluation",
                *policy_arguments,
            ]
            shell_command = (
                "git config --global --add safe.directory "
                f"{shlex.quote(str(repository_root))} && "
                "git config --global --add safe.directory "
                f"{shlex.quote(str(args.osworld_root.resolve()))} && "
                f"cd {shlex.quote(str(args.runtime_root.resolve()))} && "
                f"PYTHONPATH={shlex.quote(str(repository_root / 'code'))} "
                + shlex.join(command[18:])
            )
            command = command[:14] + ["/bin/bash", "-lc", shell_command]
            return_code = subprocess.run(
                command,
                stdout=runner_log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        finally:
            _stop(cpu_sampler)
            _stop(gpu_sampler)
            for handle in (gpu_log, cpu_log, runner_log):
                handle.close()
        wall_seconds = time.perf_counter() - started
        summary_path = point_root / "episodes/benchmark-summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else None
        )
        record = {
            **point,
            "point_id": point_id,
            "driver_return_code": return_code,
            "outer_wall_seconds": wall_seconds,
            "benchmark_summary": summary,
            "valid": (
                return_code == 0
                and summary is not None
                and not summary.get("failures")
                and summary.get("completed") == tasks
            ),
        }
        aggregate["points"].append(record)
        if not record["valid"]:
            aggregate["invalid_point_count"] += 1
        _atomic_json(output_root / "capacity-sweep.json", aggregate)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        if not record["valid"] and not config.get("continue_after_invalid_point", False):
            aggregate["status"] = "PARTIAL_OSWORLD_CAPACITY_SWEEP"
            _atomic_json(output_root / "capacity-sweep.json", aggregate)
            raise SystemExit(1)

    aggregate["status"] = (
        "COMPLETE_OSWORLD_CAPACITY_SWEEP"
        if aggregate["invalid_point_count"] == 0
        else "COMPLETE_OSWORLD_CAPACITY_SWEEP_WITH_INVALID_POINTS"
    )
    aggregate["completed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(output_root / "capacity-sweep.json", aggregate)


if __name__ == "__main__":
    main()
