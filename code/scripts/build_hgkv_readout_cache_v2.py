#!/usr/bin/env python3
"""Reuse exact V1 HGKV readouts and append frozen V2 temporal metadata."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from causalcache.hgkv_selector_v2 import (
    HGKV_READOUT_DIM,
    V2_FEATURE_DIM,
    temporal_features,
)


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(Path(value) for value in glob.glob(pattern))
    return sorted(paths)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected(
    paths: Iterable[Path],
) -> dict[tuple[str, int], dict[str, Any]]:
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            state = json.loads(line)
            pair_group = str(state["pair_group"])
            for candidate in state["candidate_event_step_ids"]:
                key = (pair_group, int(candidate))
                if key in expected:
                    raise ValueError(f"{path}:{line_no} duplicate key {key}")
                expected[key] = state
    return expected


def convert_v1_row(
    row: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    event_step_id: int,
) -> dict[str, Any]:
    readout = [float(value) for value in row["feature"]]
    if len(readout) != HGKV_READOUT_DIM:
        raise ValueError("V1 feature is not a 1280-d HGKV readout")
    if not all(math.isfinite(value) for value in readout):
        raise ValueError("V1 feature contains non-finite values")
    temporal = temporal_features(
        event_step_id=event_step_id,
        decision_step=int(state["decision_step"]),
        history_length=int(state["history_length"]),
        candidate_event_step_ids=state["candidate_event_step_ids"],
    )
    feature = [*readout, *temporal]
    if len(feature) != V2_FEATURE_DIM:
        raise AssertionError("V2 feature concatenation produced wrong dimension")
    return {
        "schema_version": "causalcache.hgkv_selector_v2.feature.v1",
        "episode": str(state["episode"]),
        "step_index": int(state["decision_step"]),
        "pair_group": str(state["pair_group"]),
        "variant": "singleton",
        "singleton_event_step_id": event_step_id,
        "restored_event_step_ids": [event_step_id],
        "candidate_event_step_ids": [
            int(value) for value in state["candidate_event_step_ids"]
        ],
        "history_length": int(state["history_length"]),
        "layer_indices": row["layer_indices"],
        "hgkv_readout_dim": HGKV_READOUT_DIM,
        "temporal_feature_dim": V2_FEATURE_DIM - HGKV_READOUT_DIM,
        "feature_dim": V2_FEATURE_DIM,
        "feature": feature,
        "scalars": row.get("scalars", {}),
        "readout_source": "v1_exact_pair_group_event_step_id",
    }


def convert_rows(
    *,
    expected: Mapping[tuple[str, int], Mapping[str, Any]],
    v1_paths: Iterable[Path],
) -> tuple[list[dict[str, Any]], set[tuple[str, int]]]:
    converted: dict[tuple[str, int], dict[str, Any]] = {}
    for path in v1_paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row["pair_group"]),
                int(row["singleton_event_step_id"]),
            )
            state = expected.get(key)
            if state is None:
                continue
            if key in converted:
                raise ValueError(f"{path}:{line_no} duplicate V1 readout {key}")
            converted[key] = convert_v1_row(
                row,
                state=state,
                event_step_id=key[1],
            )
    return [converted[key] for key in sorted(converted)], set(expected) - set(
        converted
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", action="append", required=True)
    parser.add_argument("--v1-features", action="append", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    state_paths = expand_paths(args.states)
    v1_paths = expand_paths(args.v1_features)
    if not state_paths or not v1_paths:
        parser.error("state and V1 feature patterns must match files")
    expected = load_expected(state_paths)
    rows, missing = convert_rows(expected=expected, v1_paths=v1_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema_version": (
            "causalcache.hgkv_selector_v2.reused_readout_manifest.v1"
        ),
        "source_commit": args.source_commit,
        "expected_rows": len(expected),
        "reused_rows": len(rows),
        "missing_rows_for_fresh_extraction": len(missing),
        "feature_dim": V2_FEATURE_DIM,
        "state_sha256": {
            str(path): sha256_file(path) for path in state_paths
        },
        "v1_feature_sha256": {
            str(path): sha256_file(path) for path in v1_paths
        },
        "output_sha256": sha256_file(args.output),
        "status": "PARTIAL" if missing else "DONE",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
