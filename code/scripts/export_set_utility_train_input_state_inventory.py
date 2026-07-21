#!/usr/bin/env python3
"""Export a lightweight train-only state inventory from a frozen input snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

from causalcache.set_utility_train_heldout_contract import (
    build_train_input_state_inventory,
    canonical_json_bytes,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain one JSON object")
            yield value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-input-content-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("train input state inventory output already exists")
    input_root = args.input_root.resolve()
    manifest_path = input_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("content_sha256") != args.expect_input_content_sha256:
        raise ValueError("training input content differs from the frozen expectation")
    states_path = input_root / manifest["states_jsonl"]
    if _sha256_file(states_path) != manifest.get("states_sha256"):
        raise ValueError("training states JSONL differs from its manifest")
    result = build_train_input_state_inventory(
        manifest,
        _read_jsonl(states_path),
        input_manifest_sha256=_sha256_file(manifest_path),
    )
    _write_atomic(args.output, result)
    print(
        json.dumps(
            {
                "content_sha256": result["content_sha256"],
                "status": result["status"],
                "train_state_count": result["census"]["train_state_count"],
                "train_trajectory_count": result["census"][
                    "train_trajectory_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
