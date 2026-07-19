#!/usr/bin/env python3
"""Freeze completed train/tune labels for a non-evaluative token-model pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from causalcache.set_utility_mvp import (
    COMPLETED_STATE_STATUS,
    canonical_json_bytes,
)
from causalcache.set_utility_dense import dense_trajectory_partition


SNAPSHOT_STATUS = "FROZEN_SET_UTILITY_TOKEN_PILOT_SNAPSHOT"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_source(value: str) -> tuple[str, Path, int | None, int | None]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError(
            "sources must use NAME=/absolute/state/root[@INDEX/COUNT]"
        )
    partition_index = None
    partition_count = None
    match = re.fullmatch(r"(.+)@([0-9]+)/([1-9][0-9]*)", raw_path)
    if match is not None:
        raw_path = match.group(1)
        partition_index = int(match.group(2))
        partition_count = int(match.group(3))
        if partition_index >= partition_count:
            raise argparse.ArgumentTypeError("source partition index is out of range")
    path = Path(raw_path).resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"state root does not exist: {path}")
    return name, path, partition_index, partition_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=_parse_source, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"snapshot output already exists: {output_root}")
    state_root = output_root / "states"
    state_root.mkdir(parents=True)

    selected: dict[str, dict[str, Any]] = {}
    bytes_by_state: dict[str, bytes] = {}
    source_by_state: dict[str, str] = {}
    observed_roles = Counter()
    observed_statuses = Counter()
    for source_name, source_root, partition_index, partition_count in args.source:
        for path in sorted(source_root.glob("*.json")):
            payload = path.read_bytes()
            record = json.loads(payload)
            if not isinstance(record, dict):
                raise ValueError(f"{path} must contain one JSON object")
            status = str(record.get("status"))
            role = str(record.get("role"))
            trajectory_id = str(record.get("trajectory_id"))
            if partition_index is not None and dense_trajectory_partition(
                trajectory_id,
                partition_count=partition_count,
            ) != partition_index:
                continue
            observed_statuses[status] += 1
            observed_roles[role] += 1
            if status != COMPLETED_STATE_STATUS or role not in {"train", "tune"}:
                continue
            state_id = str(record["state_id"])
            canonical = canonical_json_bytes(record) + b"\n"
            prior = bytes_by_state.get(state_id)
            if prior is not None and prior != canonical:
                raise ValueError(f"duplicate state differs across sources: {state_id}")
            selected[state_id] = record
            bytes_by_state[state_id] = canonical
            source_by_state.setdefault(state_id, source_name)

    if not selected or not any(x["role"] == "tune" for x in selected.values()):
        raise RuntimeError("snapshot requires completed train and tune states")
    records = []
    role_counts = Counter()
    trajectory_roles: dict[str, str] = {}
    for state_id in sorted(selected):
        record = selected[state_id]
        role = str(record["role"])
        trajectory_id = str(record["trajectory_id"])
        prior_role = trajectory_roles.setdefault(trajectory_id, role)
        if prior_role != role:
            raise ValueError("one trajectory crossed train/tune roles")
        filename = f"{state_id.replace(':', '_')}.json"
        destination = state_root / filename
        with destination.open("xb") as handle:
            handle.write(bytes_by_state[state_id])
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(bytes_by_state[state_id]).hexdigest()
        role_counts[role] += 1
        records.append(
            {
                "file": f"states/{filename}",
                "role": role,
                "sha256": digest,
                "source": source_by_state[state_id],
                "state_id": state_id,
                "trajectory_id": trajectory_id,
            }
        )
    manifest = {
        "evaluation_labels_included": False,
        "observed_role_counts": dict(sorted(observed_roles.items())),
        "observed_status_counts": dict(sorted(observed_statuses.items())),
        "records": records,
        "role_counts": dict(sorted(role_counts.items())),
        "schema_version": "1.0.0",
        "source_roots": [
            {
                "name": name,
                "path": str(path),
                "trajectory_partition_count": count,
                "trajectory_partition_index": index,
            }
            for name, path, index, count in args.source
        ],
        "state_count": len(records),
        "status": SNAPSHOT_STATUS,
        "trajectory_count": len(trajectory_roles),
    }
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    _write_json(output_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
