#!/usr/bin/env python3
"""Merge disjoint host snapshots into one immutable train/tune input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    args = parser.parse_args()
    if len(args.input_roots) < 2:
        raise ValueError("snapshot merge requires at least two input roots")
    if args.output_root.exists():
        raise FileExistsError("merged snapshot output already exists")

    manifests = [_read_json(root / "manifest.json") for root in args.input_roots]
    if any(
        manifest.get("status")
        != "COMPLETED_VARIABLE_HISTORY_TRAINING_INPUT_SNAPSHOT"
        or manifest.get("evaluation_labels_included") is not False
        for manifest in manifests
    ):
        raise ValueError("input snapshot status or split firewall drifted")
    for field in (
        "source_manifest_sha256",
        "label_source_revisions",
        "numeric_feature_names",
        "visual_token_profile",
    ):
        if len({json.dumps(manifest[field], sort_keys=True) for manifest in manifests}) != 1:
            raise ValueError(f"input snapshots disagree on {field}")

    states: dict[str, dict[str, Any]] = {}
    texts: dict[str, str] = {}
    input_records = []
    for root, manifest in zip(args.input_roots, manifests, strict=True):
        state_path = root / manifest["states_jsonl"]
        text_path = root / manifest["texts_jsonl"]
        if (
            hashlib.sha256(state_path.read_bytes()).hexdigest()
            != manifest["states_sha256"]
            or hashlib.sha256(text_path.read_bytes()).hexdigest()
            != manifest["texts_sha256"]
        ):
            raise ValueError("input snapshot payload hash drifted")
        for row in _read_jsonl(state_path):
            if row["state_id"] in states:
                raise ValueError("input snapshots contain a duplicate state")
            states[row["state_id"]] = row
        for row in _read_jsonl(text_path):
            prior = texts.setdefault(row["key"], row["text"])
            if prior != row["text"]:
                raise ValueError("input snapshots contain a text SHA256 collision")
        input_records.append(
            {
                "content_sha256": manifest["content_sha256"],
                "snapshot_id": manifest["snapshot_id"],
                "state_count": manifest["state_count"],
            }
        )
    if not states or not texts:
        raise ValueError("merged snapshot cannot be empty")
    trajectory_roles: dict[str, str] = {}
    for row in states.values():
        prior = trajectory_roles.setdefault(row["trajectory_id"], row["role"])
        if prior != row["role"] or row["role"] not in {"train", "tune"}:
            raise ValueError("trajectory split integrity drifted while merging")

    state_payload = b"".join(_canonical_json(states[key]) for key in sorted(states))
    text_payload = b"".join(
        _canonical_json({"key": key, "text": texts[key]}) for key in sorted(texts)
    )
    _write_atomic(args.output_root / "states.jsonl", state_payload)
    _write_atomic(args.output_root / "texts.jsonl", text_payload)
    manifest = {
        "content_sha256": hashlib.sha256(state_payload + text_payload).hexdigest(),
        "evaluation_labels_included": False,
        "input_snapshots": input_records,
        "label_source_revisions": manifests[0]["label_source_revisions"],
        "numeric_feature_names": manifests[0]["numeric_feature_names"],
        "role_counts": dict(sorted(Counter(row["role"] for row in states.values()).items())),
        "schema_version": "1.0.0",
        "snapshot_id": args.snapshot_id,
        "source_manifest_sha256": manifests[0]["source_manifest_sha256"],
        "state_count": len(states),
        "states_jsonl": "states.jsonl",
        "states_sha256": hashlib.sha256(state_payload).hexdigest(),
        "status": "COMPLETED_MERGED_VARIABLE_HISTORY_TRAINING_INPUT_SNAPSHOT",
        "text_count": len(texts),
        "texts_jsonl": "texts.jsonl",
        "texts_sha256": hashlib.sha256(text_payload).hexdigest(),
        "trajectory_count": len(trajectory_roles),
        "visual_logical_shards": sorted({row["logical_shard"] for row in states.values()}),
        "visual_token_profile": manifests[0]["visual_token_profile"],
    }
    _write_atomic(args.output_root / "manifest.json", _canonical_json(manifest))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
