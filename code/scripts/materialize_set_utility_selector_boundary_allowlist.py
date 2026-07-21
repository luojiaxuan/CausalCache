#!/usr/bin/env python3
"""Select only optimizer and checkpoint contexts for selector LoRA caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_evaluation import (
    canonical_json_bytes,
    sha256_file,
)


ALLOWLIST_STATUS = "COMPLETED_SET_UTILITY_SELECTOR_BOUNDARY_ALLOWLIST"


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def materialize_selector_boundary_allowlist(
    *,
    training_input_root: Path,
    contextual_input_root: Path,
    split_manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("selector boundary allowlist output already exists")
    training_manifest_path = training_input_root / "manifest.json"
    contextual_manifest_path = contextual_input_root / "manifest.json"
    training_manifest = _read_json(training_manifest_path)
    contextual_manifest = _read_json(contextual_manifest_path)
    split_manifest = _read_json(split_manifest_path)
    states_path = training_input_root / training_manifest["states_jsonl"]
    if sha256_file(states_path) != training_manifest["states_sha256"]:
        raise ValueError("selector allowlist training states drifted")
    states = _read_jsonl(states_path)
    by_state = {row["state_id"]: row for row in states}
    if len(by_state) != len(states):
        raise ValueError("selector allowlist source contains duplicate states")
    optimizer_ids = tuple(split_manifest["optimizer_state_ids"])
    checkpoint_ids = tuple(split_manifest["checkpoint_state_ids"])
    if set(optimizer_ids).intersection(checkpoint_ids):
        raise ValueError("selector allowlist optimizer/checkpoint states overlap")
    optimizer_id_set = set(optimizer_ids)
    selected_ids = (*optimizer_ids, *checkpoint_ids)
    if len(selected_ids) != len(set(selected_ids)) or any(
        state_id not in by_state for state_id in selected_ids
    ):
        raise ValueError("selector allowlist state inventory drifted")

    requirements = {}
    for receipt in contextual_manifest["requirement_shards"]:
        path = contextual_input_root / receipt["path"]
        if sha256_file(path) != receipt["sha256"]:
            raise ValueError("selector allowlist contextual requirements drifted")
        for row in _read_jsonl(path):
            key = row["context_key"]
            if key in requirements:
                raise ValueError("selector allowlist requirement key is duplicated")
            requirements[key] = row
    if len(requirements) != contextual_manifest["context_count"]:
        raise ValueError("selector allowlist requirement count drifted")

    context_keys = set()
    role_by_state = Counter()
    for state_id in selected_ids:
        state = by_state[state_id]
        if state.get("role") != "train":
            raise ValueError("selector allowlist crossed the train-only firewall")
        role_by_state[
            "optimizer" if state_id in optimizer_id_set else "checkpoint"
        ] += 1
        keys = {
            state["current_image_key"],
            state["instruction_text_key"],
            *state["event_image_keys"],
            *state["event_text_keys"],
        }
        if not keys.issubset(requirements):
            raise ValueError("selector allowlist state references an unknown context")
        context_keys.update(keys)
    ordered_keys = tuple(sorted(context_keys))
    payload = b"".join(key.encode("ascii") + b"\n" for key in ordered_keys)
    role_counts = Counter(requirements[key]["entity_role"] for key in ordered_keys)
    _write_atomic(output_root / "context_keys.txt", payload)
    manifest: dict[str, Any] = {
        "checkpoint_state_count": len(checkpoint_ids),
        "context_count": len(ordered_keys),
        "context_input_content_sha256": contextual_manifest["content_sha256"],
        "context_keys_file": "context_keys.txt",
        "context_keys_sha256": hashlib.sha256(payload).hexdigest(),
        "entity_role_counts": dict(sorted(role_counts.items())),
        "evaluation_labels_included": False,
        "optimizer_state_count": len(optimizer_ids),
        "schema_version": "1.0.0",
        "split_manifest_content_sha256": split_manifest["content_sha256"],
        "state_role_counts": dict(sorted(role_by_state.items())),
        "status": ALLOWLIST_STATUS,
        "training_input_content_sha256": training_manifest["content_sha256"],
    }
    manifest["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    _write_atomic(
        output_root / "manifest.json", canonical_json_bytes(manifest, pretty=True)
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-input-root", type=Path, required=True)
    parser.add_argument("--contextual-input-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_selector_boundary_allowlist(
        training_input_root=args.training_input_root.resolve(),
        contextual_input_root=args.contextual_input_root.resolve(),
        split_manifest_path=args.split_manifest.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
