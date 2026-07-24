#!/usr/bin/env python3
"""Plan policy-isolated AndroidWorld matrix continuation workers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.sealed_matrix_v1 import ARMS, POLICIES, load_roster, sha256_file


@dataclass(frozen=True)
class WorkerSpec:
    policy: str
    gpu: int
    port: int
    ordinal: int


def parse_worker(value: str) -> WorkerSpec:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--worker must use POLICY:GPU:PORT"
        )
    policy, raw_gpu, raw_port = parts
    if policy not in POLICIES:
        raise argparse.ArgumentTypeError(f"unknown policy: {policy}")
    try:
        gpu = int(raw_gpu)
        port = int(raw_port)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "worker GPU and port must be integers"
        ) from error
    if gpu < 0 or not 1024 <= port <= 65535:
        raise argparse.ArgumentTypeError("worker GPU or port is out of range")
    return WorkerSpec(policy=policy, gpu=gpu, port=port, ordinal=-1)


def load_missing(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (
                str(row["policy"]),
                str(row["arm"]),
                str(row["task_type"]),
                int(row["task_index"]),
            )
            if key in seen:
                raise ValueError(f"duplicate missing cell: {key}")
            if key[0] not in POLICIES or key[1] not in ARMS:
                raise ValueError(f"unregistered missing cell: {key}")
            seen.add(key)
            rows.append(
                {
                    "policy": key[0],
                    "arm": key[1],
                    "task_type": key[2],
                    "task_index": key[3],
                }
            )
    return rows


def assign_workers(
    missing: list[dict[str, Any]],
    *,
    workers: list[WorkerSpec],
    roster: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    if len({worker.gpu for worker in workers}) != len(workers):
        raise ValueError("worker GPUs must be unique")
    if len({worker.port for worker in workers}) != len(workers):
        raise ValueError("worker ports must be unique")
    normalized_workers = [
        WorkerSpec(
            policy=worker.policy,
            gpu=worker.gpu,
            port=worker.port,
            ordinal=index,
        )
        for index, worker in enumerate(workers)
    ]
    assignments = [
        {
            "cells": [],
            "estimated_max_steps": 0,
            "gpu": worker.gpu,
            "ordinal": worker.ordinal,
            "policy": worker.policy,
            "port": worker.port,
        }
        for worker in normalized_workers
    ]
    for policy in POLICIES:
        policy_workers = [
            assignment
            for assignment in assignments
            if assignment["policy"] == policy
        ]
        policy_cells = [row for row in missing if row["policy"] == policy]
        if policy_cells and not policy_workers:
            raise ValueError(f"missing cells have no {policy} worker")
        weighted = []
        for row in policy_cells:
            roster_key = (row["task_type"], row["task_index"])
            if roster_key not in roster:
                raise ValueError(f"missing cell is outside roster: {roster_key}")
            weighted.append((int(roster[roster_key]["max_steps"]), row))
        # note (luojiaxuan): 同 policy 模型只加载一次；policy 内用 LPT 按
        # frozen max_steps 贪心分配，避免长任务偶然集中到一个 emulator。
        weighted.sort(
            key=lambda item: (
                -item[0],
                item[1]["arm"],
                item[1]["task_type"],
                item[1]["task_index"],
            )
        )
        for max_steps, row in weighted:
            target = min(
                policy_workers,
                key=lambda assignment: (
                    assignment["estimated_max_steps"],
                    len(assignment["cells"]),
                    assignment["ordinal"],
                ),
            )
            target["cells"].append(row)
            target["estimated_max_steps"] += max_steps
    return [assignment for assignment in assignments if assignment["cells"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--missing-cells", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--worker", type=parse_worker, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    roster = load_roster(args.roster)
    missing = load_missing(args.missing_cells)
    assignments = assign_workers(
        missing,
        workers=args.worker,
        roster=roster,
    )
    payload = {
        "assignments": assignments,
        "missing_cell_count": len(missing),
        "planned_cell_count": sum(
            len(assignment["cells"]) for assignment in assignments
        ),
        "roster_sha256": sha256_file(args.roster),
        "source_commit": args.source_commit,
        "worker_count": len(assignments),
    }
    if payload["planned_cell_count"] != payload["missing_cell_count"]:
        raise RuntimeError("worker assignment coverage drifted")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
