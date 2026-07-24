#!/usr/bin/env python3
"""Validate exact missing-key coverage of V2 full-history singleton renders."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from causalcache.hgkv_selector_v2 import temporal_features
from scripts.render_hgkv_selector_v2_singletons import (
    load_cached_singletons,
    load_states,
)


def _expand(patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(Path(value) for value in glob.glob(pattern))
    return sorted(paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_paths(sample: dict[str, Any], root: Path) -> set[Path]:
    images: set[Path] = set()
    for message in sample["messages"]:
        for content in message["content"]:
            if content.get("type") == "image":
                images.add(root / str(content["path"]))
    return images


def validate(
    *,
    state_paths: list[Path],
    sample_paths: list[Path],
    coalition_cache_pattern: str,
) -> dict[str, Any]:
    states_by_episode: dict[str, dict[str, Any]] = {}
    for path in state_paths:
        for episode, state in load_states(path).items():
            if episode in states_by_episode:
                raise ValueError(f"duplicate inventory episode {episode}")
            states_by_episode[episode] = state
    states = {
        str(state["pair_group"]): state for state in states_by_episode.values()
    }
    cached = load_cached_singletons(coalition_cache_pattern)
    required = {
        (pair_group, int(candidate))
        for pair_group, state in states.items()
        for candidate in state["candidate_event_step_ids"]
    }
    expected_missing = required - cached
    observed: set[tuple[str, int]] = set()
    images: set[Path] = set()
    for sample_path in sample_paths:
        for line_no, line in enumerate(sample_path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("variant") != "singleton":
                raise ValueError(f"{sample_path}:{line_no} is not singleton")
            key = (
                str(sample["pair_group"]),
                int(sample["singleton_event_step_id"]),
            )
            if key not in expected_missing:
                raise ValueError(f"{sample_path}:{line_no} unexpected key {key}")
            if key in observed:
                raise ValueError(f"{sample_path}:{line_no} duplicate key {key}")
            observed.add(key)
            state = states[key[0]]
            if sample["candidate_event_step_ids"] != state[
                "candidate_event_step_ids"
            ]:
                raise ValueError(f"{sample_path}:{line_no} inventory drift")
            if sample["memory_config"]["restored_event_step_ids"] != [key[1]]:
                raise ValueError(f"{sample_path}:{line_no} restored set drift")
            expected_temporal = temporal_features(
                event_step_id=key[1],
                decision_step=int(state["decision_step"]),
                history_length=int(state["history_length"]),
                candidate_event_step_ids=state["candidate_event_step_ids"],
            )
            if any(
                abs(float(left) - right) > 1e-12
                for left, right in zip(
                    sample["temporal_features"], expected_temporal
                )
            ):
                raise ValueError(f"{sample_path}:{line_no} temporal drift")
            images.update(_image_paths(sample, sample_path.parent))
    missing = expected_missing - observed
    if missing:
        raise ValueError(f"render is missing {len(missing)} exact singleton keys")
    image_sha256: dict[str, str] = {}
    for path in sorted(images):
        if not path.is_file():
            raise ValueError(f"render references missing image {path}")
        image_sha256[str(path)] = _sha256(path)
    aggregate = hashlib.sha256(
        json.dumps(image_sha256, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "causalcache.hgkv_selector_v2.singleton_render_done.v1",
        "states": len(states),
        "required_singletons": len(required),
        "cached_singletons": len(required & cached),
        "rendered_missing_singletons": len(observed),
        "complete_singletons": len((required & cached) | observed),
        "unexpected_rows": 0,
        "duplicate_rows": 0,
        "missing_rows": 0,
        "referenced_images": len(images),
        "image_manifest_sha256": aggregate,
        "state_sha256": {str(path): _sha256(path) for path in state_paths},
        "sample_sha256": {str(path): _sha256(path) for path in sample_paths},
        "status": "DONE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", action="append", required=True)
    parser.add_argument("--samples", action="append", required=True)
    parser.add_argument("--coalition-cache", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state_paths = _expand(args.states)
    sample_paths = _expand(args.samples)
    if not state_paths or not sample_paths:
        parser.error("state and sample patterns must match files")
    result = validate(
        state_paths=state_paths,
        sample_paths=sample_paths,
        coalition_cache_pattern=args.coalition_cache,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
