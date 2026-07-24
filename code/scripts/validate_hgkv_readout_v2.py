#!/usr/bin/env python3
"""Validate exact full-history coverage and schema of V2 HGKV readout features."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from causalcache.hgkv_selector_v2 import (
    HGKV_READOUT_DIM,
    V2_FEATURE_DIM,
    temporal_features,
)


def _expand(patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            paths.update(path.glob("*.jsonl"))
        else:
            paths.update(Path(value) for value in glob.glob(pattern))
    return sorted(paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected(paths: Iterable[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            state = json.loads(line)
            pair_group = str(state["pair_group"])
            candidates = tuple(
                int(value) for value in state["candidate_event_step_ids"]
            )
            if tuple(sorted(candidates)) != candidates:
                raise ValueError(f"{path}:{line_no} candidates are not canonical")
            for candidate in candidates:
                key = (pair_group, candidate)
                if key in expected:
                    raise ValueError(f"duplicate expected feature key {key}")
                expected[key] = state
    return expected


def validate_features(
    expected: dict[tuple[str, int], dict[str, Any]],
    feature_paths: Iterable[Path],
) -> dict[str, Any]:
    observed: set[tuple[str, int]] = set()
    dimensions: set[int] = set()
    for path in feature_paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row["pair_group"]),
                int(row["singleton_event_step_id"]),
            )
            if key not in expected:
                raise ValueError(f"{path}:{line_no} unexpected feature key {key}")
            if key in observed:
                raise ValueError(f"{path}:{line_no} duplicate feature key {key}")
            observed.add(key)
            feature = row.get("feature")
            if not isinstance(feature, list):
                raise ValueError(f"{path}:{line_no} feature must be a list")
            dimension = int(row.get("feature_dim", len(feature)))
            dimensions.add(dimension)
            if dimension != V2_FEATURE_DIM or len(feature) != V2_FEATURE_DIM:
                raise ValueError(f"{path}:{line_no} V2 feature dimension drifted")
            if int(row.get("hgkv_readout_dim", HGKV_READOUT_DIM)) != HGKV_READOUT_DIM:
                raise ValueError(f"{path}:{line_no} HGKV readout dimension drifted")
            values = [float(value) for value in feature]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{line_no} feature contains non-finite values")
            state = expected[key]
            temporal = temporal_features(
                event_step_id=key[1],
                decision_step=int(state["decision_step"]),
                history_length=int(state["history_length"]),
                candidate_event_step_ids=state["candidate_event_step_ids"],
            )
            if any(
                abs(left - right) > 1e-6
                for left, right in zip(values[-len(temporal) :], temporal)
            ):
                raise ValueError(f"{path}:{line_no} temporal features drifted")
    missing = sorted(set(expected) - observed)
    if missing:
        raise ValueError(f"missing {len(missing)} full-history feature keys")
    return {
        "expected_rows": len(expected),
        "observed_rows": len(observed),
        "unique_rows": len(observed),
        "feature_dimensions": sorted(dimensions),
        "states_with_early_candidates": len(
            {
                pair_group
                for pair_group, _candidate in expected
                if len(
                    expected[(pair_group, _candidate)][
                        "candidate_event_step_ids"
                    ]
                )
                > 8
            }
        ),
        "recent_8_truncation_count": 0,
        "status": "DONE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", action="append", required=True)
    parser.add_argument("--features", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state_paths = _expand(args.states)
    feature_paths = _expand(args.features)
    if not state_paths or not feature_paths:
        parser.error("state and feature patterns must match files")
    expected = load_expected(state_paths)
    result = validate_features(expected, feature_paths)
    result["schema_version"] = "causalcache.hgkv_selector_v2.feature_done.v1"
    result["state_sha256"] = {
        str(path): _sha256(path) for path in state_paths
    }
    result["feature_sha256"] = {
        str(path): _sha256(path) for path in feature_paths
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
