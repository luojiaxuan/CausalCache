#!/usr/bin/env python3
"""Reduce an OSWorld capacity sweep and its host telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


SCHEMA_VERSION = "causalcache.osworld.capacity_summary.v1"


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _gpu_telemetry(path: Path, replicas: int) -> dict[str, Any]:
    utilization: list[float] = []
    memory_mib: list[float] = []
    power_watts: list[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 5 or int(row[1].strip()) >= replicas:
                continue
            utilization.append(float(row[2].strip()))
            memory_mib.append(float(row[3].strip()))
            power_watts.append(float(row[4].strip()))
    return {
        "sample_count": len(utilization),
        "utilization_mean_percent": mean(utilization),
        "utilization_p95_percent": _percentile(utilization, 0.95),
        "utilization_nonzero_fraction": sum(value > 0 for value in utilization)
        / len(utilization),
        "memory_mean_mib": mean(memory_mib),
        "memory_max_mib": max(memory_mib),
        "power_mean_watts": mean(power_watts),
    }


def _cpu_telemetry(path: Path) -> dict[str, Any]:
    samples: list[list[float]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 17 or not all(field.lstrip("-").isdigit() for field in fields):
                continue
            samples.append([float(value) for value in fields])
    samples = samples[1:]
    return {
        "sample_count": len(samples),
        "runnable_mean": mean(row[0] for row in samples),
        "cpu_user_mean_percent": mean(row[-5] for row in samples),
        "cpu_system_mean_percent": mean(row[-4] for row in samples),
        "cpu_idle_mean_percent": mean(row[-3] for row in samples),
        "cpu_iowait_mean_percent": mean(row[-2] for row in samples),
    }


def summarize(raw_root: Path, *, source_raw_root: str | None = None) -> dict[str, Any]:
    source = json.loads((raw_root / "capacity-sweep.json").read_text(encoding="utf-8"))
    if source.get("status") != "COMPLETE_OSWORLD_CAPACITY_SWEEP":
        raise ValueError(f"capacity sweep is not complete: {source.get('status')}")
    points = []
    for point in source["points"]:
        benchmark = point["benchmark_summary"]
        point_root = raw_root / "points" / point["point_id"]
        points.append(
            {
                "point_id": point["point_id"],
                "axis": point["axis"],
                "policy_replicas": point["policy_replicas"],
                "num_envs": point["num_envs"],
                "tasks": point["tasks"],
                "valid": point["valid"],
                "wall_seconds": benchmark["benchmark_wall_seconds"],
                "tasks_per_hour": benchmark["fresh_tasks_per_hour"],
                "policy_requests_per_second": benchmark["policy_requests_per_second"],
                "client_p50_seconds": benchmark["policy_latency_seconds"]["p50"],
                "client_p95_seconds": benchmark["policy_latency_seconds"]["p95"],
                "queue_p50_seconds": benchmark["server_queue_seconds"]["p50"],
                "queue_p95_seconds": benchmark["server_queue_seconds"]["p95"],
                "generation_p50_seconds": benchmark["generation_seconds"]["p50"],
                "generation_p95_seconds": benchmark["generation_seconds"]["p95"],
                "failure_count": len(benchmark["failures"]),
                "gpu": _gpu_telemetry(
                    point_root / "nvidia-smi.csv", point["policy_replicas"]
                ),
                "cpu": _cpu_telemetry(point_root / "vmstat.txt"),
            }
        )
    env_points = [point for point in points if point["axis"] == "env" and point["valid"]]
    gpu_points = [
        point for point in points if point["num_envs"] == 12 and point["valid"]
    ]
    env_knee = max(env_points, key=lambda point: point["tasks_per_hour"])
    latency_point = min(gpu_points, key=lambda point: point["queue_p95_seconds"])
    return {
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": source["schema_version"],
        "source_status": source["status"],
        "source_git_revision": source["causalcache_git_revision"],
        "source_raw_root": source_raw_root or str(raw_root),
        "started_at": source["started_at"],
        "completed_at": source["completed_at"],
        "environment_throughput_knee": {
            "policy_replicas": env_knee["policy_replicas"],
            "num_envs": env_knee["num_envs"],
            "tasks_per_hour": env_knee["tasks_per_hour"],
        },
        "lowest_gpu_axis_queue_p95": {
            "policy_replicas": latency_point["policy_replicas"],
            "num_envs": latency_point["num_envs"],
            "queue_p95_seconds": latency_point["queue_p95_seconds"],
        },
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-raw-root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(
        args.raw_root.resolve(), source_raw_root=args.source_raw_root
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
