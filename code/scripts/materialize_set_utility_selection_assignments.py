#!/usr/bin/env python3
"""Map train states onto a frozen rollout plan by logical shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--rollout-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "tune"), required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("selection assignment output already exists")
    input_manifest = _read_json(args.input_root / "manifest.json")
    plan = _read_json(args.rollout_plan)
    if plan.get("logical_shard_count") != 256:
        raise ValueError("selection rollout plan must cover 256 logical shards")
    plan_workers = plan.get("workers")
    if not isinstance(plan_workers, list) or not plan_workers:
        raise ValueError("selection rollout plan must contain workers")
    shard_to_worker = {}
    workers = {}
    host_gpus = set()
    for worker in plan_workers:
        if not isinstance(worker, dict):
            raise ValueError("selection rollout plan contains an invalid worker")
        worker_id = worker.get("worker_id")
        host = worker.get("host")
        physical_gpu_id = worker.get("physical_gpu_id")
        logical_shard_ids = worker.get("logical_shard_ids")
        if type(worker_id) is not int or worker_id < 0:
            raise ValueError("selection rollout plan has an invalid worker id")
        if not isinstance(host, str) or not host:
            raise ValueError("selection rollout plan has an invalid host")
        if type(physical_gpu_id) is not int or physical_gpu_id < 0:
            raise ValueError("selection rollout plan has an invalid physical GPU")
        if not isinstance(logical_shard_ids, list):
            raise ValueError("selection rollout plan has invalid logical shards")
        if worker_id in workers:
            raise ValueError("selection rollout plan duplicates a worker")
        host_gpu = (host, physical_gpu_id)
        if host_gpu in host_gpus:
            raise ValueError("selection rollout plan duplicates a host/GPU mapping")
        host_gpus.add(host_gpu)
        workers[worker_id] = worker
        for shard in logical_shard_ids:
            if type(shard) is not int or not 0 <= shard < 256:
                raise ValueError("selection rollout plan has an invalid logical shard")
            if shard in shard_to_worker:
                raise ValueError("selection rollout plan duplicates a logical shard")
            shard_to_worker[shard] = worker_id
    if set(shard_to_worker) != set(range(256)):
        raise ValueError("selection rollout plan does not cover every logical shard")
    assigned = {worker_id: [] for worker_id in workers}
    candidate_counts = {worker_id: 0 for worker_id in workers}
    seen_state_ids = set()
    for line in (args.input_root / input_manifest["states_jsonl"]).read_text(
        encoding="utf-8"
    ).splitlines():
        if not line:
            continue
        state = json.loads(line)
        state_id = state.get("state_id")
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("selection input contains an invalid state id")
        if state_id in seen_state_ids:
            raise ValueError("selection input contains a duplicate state")
        seen_state_ids.add(state_id)
        if state["role"] != args.role:
            continue
        logical_shard = state.get("logical_shard")
        if type(logical_shard) is not int or logical_shard not in shard_to_worker:
            raise ValueError("selection input contains an invalid logical shard")
        candidates = state.get("candidate_event_step_ids")
        if not isinstance(candidates, list):
            raise ValueError("selection input contains invalid candidate events")
        worker_id = shard_to_worker[logical_shard]
        assigned[worker_id].append(state_id)
        candidate_counts[worker_id] += len(state["candidate_event_step_ids"])
    if any(not state_ids for state_ids in assigned.values()):
        raise ValueError("selection rollout plan produced an empty worker")
    receipts = []
    for worker_id, worker in sorted(workers.items()):
        state_ids = sorted(assigned[worker_id])
        payload = ("\n".join(state_ids) + "\n").encode("utf-8")
        relative = f"worker-{worker_id:02d}-state-ids.txt"
        digest = _write(args.output_root / relative, payload)
        receipts.append(
            {
                "candidate_event_count": candidate_counts[worker_id],
                "file_sha256": digest,
                "host": worker["host"],
                "logical_shard_ids": worker["logical_shard_ids"],
                "path": relative,
                "physical_gpu_id": worker["physical_gpu_id"],
                "state_count": len(state_ids),
                "worker_id": worker_id,
            }
        )
    manifest = {
        "input_content_sha256": input_manifest["content_sha256"],
        "role": args.role,
        "rollout_plan": plan,
        "schema_version": "causalcache.selection_assignments.v1",
        "state_count": sum(row["state_count"] for row in receipts),
        "status": "COMPLETED_SET_UTILITY_SELECTION_ASSIGNMENTS",
        "workers": receipts,
    }
    unsigned = json.dumps(
        manifest, allow_nan=False, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    manifest["content_sha256"] = hashlib.sha256(unsigned).hexdigest()
    _write(
        args.output_root / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
