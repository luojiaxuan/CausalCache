#!/usr/bin/env python3
"""Materialize a deterministic trajectory-disjoint train-heldout contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from causalcache.set_utility_train_heldout_contract import (
    canonical_json_bytes,
    materialize_train_heldout_contract,
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
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("train-heldout contract output already exists")
    repository_root = args.repository_root.resolve()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    assignment_path = repository_root / config["source"]["assignment_manifest"]
    assignment_sha = _sha256_file(assignment_path)
    if assignment_sha != config["source"]["assignment_manifest_sha256"]:
        raise ValueError("assignment manifest differs from the frozen config")
    assignment_manifest = _read_json(assignment_path)
    state_inventory_path = repository_root / config["source"][
        "train_state_inventory"
    ]
    state_inventory = _read_json(state_inventory_path)
    result = materialize_train_heldout_contract(
        assignment_manifest["assignments"],
        train_state_inventory=state_inventory,
        config=config,
        assignment_manifest_sha256=assignment_sha,
        config_sha256=_sha256_file(config_path),
    )
    _write_atomic(args.output, result)
    print(
        json.dumps(
            {
                "checkpoint_state_count": result["census"][
                    "checkpoint_state_count"
                ],
                "content_sha256": result["content_sha256"],
                "heldout_trajectory_count": result["census"][
                    "heldout_trajectory_count"
                ],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
