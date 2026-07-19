#!/usr/bin/env python3
"""Validate that legacy shards plus backfill expand to every dense state."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from causalcache import set_utility_processor_artifacts as artifacts
from causalcache.set_utility_dense import derive_dense_states_from_query_pair


def _read_backfill(root: Path) -> dict[str, dict[str, bytes]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED_DENSE_IMAGE_BACKFILL":
        raise ValueError("dense image backfill is incomplete")
    result: dict[str, dict[str, bytes]] = defaultdict(dict)
    for record in manifest["files"]:
        payload = (root / record["reference"]).read_bytes()
        if (
            len(payload) != record["byte_count"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise ValueError("dense backfill payload drifted")
        result[record["source_id"]][record["reference"]] = payload
    return result


def _validate_worker(
    worker_index: int,
    processor_root: Path,
    backfill: dict[str, dict[str, bytes]],
) -> dict[str, Any]:
    path = (
        processor_root
        / "substrate"
        / f"processor-substrate-worker-{worker_index:02d}.tar"
    )
    with tarfile.open(path, mode="r:") as archive:
        expected_worker = artifacts._read_manifest(archive, expected_worker=None)
    iterator = artifacts.iter_processor_query_artifact_records(
        path, expected_worker=expected_worker
    )
    counts = Counter()
    trajectory_count = 0
    while True:
        try:
            pair = (next(iterator), next(iterator))
        except StopIteration:
            break
        trajectory_count += 1
        states = derive_dense_states_from_query_pair(
            pair,
            supplemental_image_payloads=backfill.get(pair[0].trajectory_id),
        )
        expected = len(pair[1].history_events) - 4
        if len(states) != expected:
            raise RuntimeError(
                f"dense expansion is incomplete for {pair[0].trajectory_id}"
            )
        counts[pair[0].role] += len(states)
    return {
        "state_count": sum(counts.values()),
        "state_count_by_role": dict(sorted(counts.items())),
        "trajectory_count": trajectory_count,
        "worker_index": worker_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor-root", type=Path, required=True)
    parser.add_argument("--backfill-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backfill = _read_backfill(args.backfill_root)
    with ThreadPoolExecutor(max_workers=4) as executor:
        workers = tuple(
            executor.map(
                lambda index: _validate_worker(index, args.processor_root, backfill),
                range(4),
            )
        )
    roles = Counter()
    for worker in workers:
        roles.update(worker["state_count_by_role"])
    payload = {
        "backfill_file_count": sum(len(value) for value in backfill.values()),
        "state_count": sum(worker["state_count"] for worker in workers),
        "state_count_by_role": dict(sorted(roles.items())),
        "status": "VALIDATED_DENSE_PER_STEP_INPUTS",
        "trajectory_count": sum(worker["trajectory_count"] for worker in workers),
        "workers": workers,
    }
    if (
        payload["state_count"] != 12_792
        or payload["trajectory_count"] != 1_200
        or payload["state_count_by_role"]
        != {"evaluation": 1_046, "train": 10_680, "tune": 1_066}
    ):
        raise RuntimeError("dense input aggregate differs from the census")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "workers"}))


if __name__ == "__main__":
    main()
