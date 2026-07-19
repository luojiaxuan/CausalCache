#!/usr/bin/env python3
"""Build resumable trajectory shards with full images and processed metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from causalcache.set_utility_variable_history import logical_shard_for_trajectory


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _processor_terminal_records(processor_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    paths = sorted((processor_root / "substrate").glob("processor-substrate-worker-*.tar"))
    if len(paths) != 4:
        raise ValueError("processor source must contain four worker tar files")
    for path in paths:
        with tarfile.open(path, mode="r:") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith("/record.json"):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError("processor record member is unreadable")
                record = json.load(handle)
                if record.get("query_kind") != "terminal":
                    continue
                trajectory_id = record.get("trajectory_id")
                if not isinstance(trajectory_id, str) or trajectory_id in records:
                    raise ValueError("processor terminal trajectory identity is invalid")
                records[trajectory_id] = record
    return records


def _validated_existing_shard(path: Path, receipt_path: Path, identity: str) -> dict[str, Any] | None:
    if not path.exists() and not receipt_path.exists():
        return None
    if not path.exists() or not receipt_path.exists():
        raise ValueError(f"incomplete source shard/receipt pair: {path.name}")
    receipt = _read_json(receipt_path)
    if (
        receipt.get("identity_sha256") != identity
        or receipt.get("sha256") != _sha256_file(path)
        or receipt.get("byte_count") != path.stat().st_size
    ):
        raise ValueError(f"existing source shard receipt drifted: {path.name}")
    return receipt


def _build_shard(
    shard_index: int,
    assignments: tuple[dict[str, Any], ...],
    *,
    terminal_records: dict[str, dict[str, Any]],
    source_root: Path,
    output_root: Path,
    contract_identity: str,
) -> dict[str, Any]:
    try:
        import pyarrow as pa
        from pyarrow import parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError("variable-history source materialization requires pyarrow") from error

    identity_payload = {
        "assignments": [
            {
                key: assignment[key]
                for key in (
                    "decision_count",
                    "role",
                    "source_id",
                    "transport_file",
                    "transport_row_index",
                )
            }
            for assignment in assignments
        ],
        "contract_identity": contract_identity,
        "logical_shard": shard_index,
    }
    identity = hashlib.sha256(_canonical_json(identity_payload).encode("utf-8")).hexdigest()
    path = output_root / "trajectory-shards" / f"shard-{shard_index:03d}-of-256.parquet"
    receipt_path = output_root / "receipts" / f"shard-{shard_index:03d}-of-256.json"
    existing = _validated_existing_shard(path, receipt_path, identity)
    if existing is not None:
        return existing

    raw_by_file: dict[str, Any] = {}
    rows = []
    for assignment in assignments:
        transport_file = assignment["transport_file"]
        table = raw_by_file.get(transport_file)
        if table is None:
            table = pq.read_table(
                source_root / transport_file,
                columns=["images", "messages", "metadata"],
            )
            raw_by_file[transport_file] = table
        raw = table.slice(assignment["transport_row_index"], 1).to_pylist()[0]
        terminal = terminal_records[assignment["source_id"]]
        if (
            terminal["role"] != assignment["role"]
            or len(terminal["history_events"]) != assignment["decision_count"]
            or len(raw["images"]) != assignment["decision_count"] + 1
        ):
            raise ValueError("raw, processor, and assignment trajectory metadata differ")
        rows.append(
            {
                "decision_count": assignment["decision_count"],
                "history_events_json": _canonical_json(terminal["history_events"]),
                "images": raw["images"],
                "ocr_records_json": _canonical_json(terminal["ocr_records_by_path"]),
                "raw_messages": raw["messages"],
                "raw_metadata": raw["metadata"],
                "role": assignment["role"],
                "source_id": assignment["source_id"],
                "task_instruction": terminal["task_instruction"],
                "transport_file": transport_file,
                "transport_row_index": assignment["transport_row_index"],
            }
        )
    raw_by_file.clear()
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    pq.write_table(table, temporary, compression="zstd", compression_level=3)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    receipt = {
        "byte_count": path.stat().st_size,
        "identity_sha256": identity,
        "logical_shard": shard_index,
        "row_count": len(rows),
        "sha256": _sha256_file(path),
        "status": "COMPLETED_VARIABLE_HISTORY_SOURCE_SHARD",
        "trajectory_ids": [assignment["source_id"] for assignment in assignments],
    }
    _write_atomic(receipt_path, (_canonical_json(receipt) + "\n").encode("utf-8"))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--processor-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("worker count must be positive")

    config = _read_json(args.config)
    if config["resumability"]["dynamic_logical_shards"] != 256:
        raise ValueError("source materializer requires the frozen 256 logical shards")
    assignment_payload = _read_json(args.assignments)
    assignments = tuple(assignment_payload["assignments"])
    terminals = _processor_terminal_records(args.processor_root)
    expected_ids = {assignment["source_id"] for assignment in assignments}
    if set(terminals) != expected_ids:
        raise ValueError("processor terminals do not exactly cover frozen trajectories")

    by_shard: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for assignment in assignments:
        by_shard[
            logical_shard_for_trajectory(assignment["source_id"], shard_count=256)
        ].append(assignment)
    contract_identity = hashlib.sha256(
        (
            _sha256_file(args.config)
            + _sha256_file(args.assignments)
            + config["source"]["transport_revision"]
        ).encode("utf-8")
    ).hexdigest()

    def build(index: int) -> dict[str, Any]:
        selected = tuple(sorted(by_shard.get(index, ()), key=lambda row: row["source_id"]))
        return _build_shard(
            index,
            selected,
            terminal_records=terminals,
            source_root=args.source_root,
            output_root=args.output_root,
            contract_identity=contract_identity,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        receipts = list(executor.map(build, range(256)))
    if sum(receipt["row_count"] for receipt in receipts) != len(assignments):
        raise RuntimeError("materialized source row count differs from assignments")
    manifest = {
        "assignment_manifest_sha256": _sha256_file(args.assignments),
        "config_sha256": _sha256_file(args.config),
        "logical_shard_count": 256,
        "schema_version": "1.0.0",
        "shards": receipts,
        "status": "COMPLETED_VARIABLE_HISTORY_SOURCE",
        "total_bytes": sum(receipt["byte_count"] for receipt in receipts),
        "trajectory_count": len(assignments),
        "transport_revision": config["source"]["transport_revision"],
    }
    _write_atomic(
        args.output_root / "manifest.json",
        (_canonical_json(manifest) + "\n").encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "logical_shard_count": len(receipts),
                "status": manifest["status"],
                "total_bytes": manifest["total_bytes"],
                "trajectory_count": manifest["trajectory_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
