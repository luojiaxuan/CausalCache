#!/usr/bin/env python3
"""Extract only the legacy processor artifact's missing trajectory images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from causalcache.set_utility_dense import (
    dense_decision_steps,
    dense_required_event_step_ids,
    legacy_available_event_step_ids,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing backfill payload differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _required_missing_steps(decision_count: int) -> tuple[int, ...]:
    available = legacy_available_event_step_ids(decision_count=decision_count)
    required: set[int] = set()
    for decision_step in dense_decision_steps(decision_count=decision_count):
        missing = dense_required_event_step_ids(decision_step) - available
        required.update(missing)
    return tuple(sorted(required))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    from pyarrow import parquet as pq

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assignment in manifest["assignments"]:
        missing = _required_missing_steps(assignment["decision_count"])
        if missing:
            item = {**assignment, "missing_event_step_ids": missing}
            selected.append(item)
            grouped[assignment["transport_file"]].append(item)

    records = []
    for transport_file in sorted(grouped):
        table = pq.read_table(args.source_root / transport_file, columns=["images"])
        for assignment in sorted(
            grouped[transport_file], key=lambda item: item["transport_row_index"]
        ):
            row_index = assignment["transport_row_index"]
            images = table.slice(row_index, 1).to_pylist()[0]["images"]
            if len(images) != assignment["decision_count"] + 1:
                raise ValueError("raw trajectory image count drifted")
            for step in assignment["missing_event_step_ids"]:
                value = images[step]
                payload = value.get("bytes") if isinstance(value, dict) else None
                if not isinstance(payload, bytes) or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("backfill source image is not encoded PNG bytes")
                relative = f"images/{assignment['source_id']}/observation-{step:03d}.png"
                _write_once(args.output_root / relative, payload)
                records.append(
                    {
                        "byte_count": len(payload),
                        "reference": relative,
                        "sha256": _sha256(payload),
                        "source_id": assignment["source_id"],
                        "transport_file": transport_file,
                        "transport_row_index": row_index,
                    }
                )
    output = {
        "file_count": len(records),
        "files": sorted(records, key=lambda item: item["reference"]),
        "source_trajectory_count": len(selected),
        "status": "COMPLETED_DENSE_IMAGE_BACKFILL",
    }
    _write_once(
        args.output_root / "manifest.json",
        (json.dumps(output, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({key: value for key, value in output.items() if key != "files"}))


if __name__ == "__main__":
    main()
