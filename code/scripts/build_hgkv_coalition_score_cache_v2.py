#!/usr/bin/env python3
"""Build the V2 coalition cache with exact scientific keys and B0 joins."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from causalcache.hgkv_selector_v2 import (
    CoalitionCacheKey,
)


def _paths(patterns: Iterable[str]) -> list[Path]:
    result: set[Path] = set()
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            result.update(path.glob("*.jsonl"))
        else:
            result.update(Path(value) for value in glob.glob(pattern))
    return sorted(result)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(paths: Iterable[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_no} invalid JSON") from error
                yield path, line_no, row


def load_target_actions(paths: Iterable[Path]) -> dict[str, str]:
    targets: dict[str, str] = {}
    for path, line_no, row in _read_rows(paths):
        pair_group = row.get("pair_group")
        target = row.get("target_text")
        if pair_group is None or target is None:
            continue
        pair_group = str(pair_group)
        target = str(target)
        previous = targets.get(pair_group)
        if previous is not None and previous != target:
            raise ValueError(f"{path}:{line_no} target drift for {pair_group}")
        targets[pair_group] = target
    return targets


def _restored_ids(row: Mapping[str, Any]) -> tuple[int, ...]:
    memory = row.get("memory_config")
    if isinstance(memory, Mapping) and isinstance(
        memory.get("restored_event_step_ids"), list
    ):
        return tuple(int(value) for value in memory["restored_event_step_ids"])
    key = row.get("restored_set_key")
    if key == "" or key is None and str(row.get("variant", "")).lower() == "b0":
        return ()
    if isinstance(key, str):
        return tuple(int(value) for value in key.split("-") if value)
    singleton = row.get("singleton_event_step_id")
    if singleton is not None:
        return (int(singleton),)
    raise ValueError("score row does not identify its restored coalition")


def load_b0_scores(paths: Iterable[Path]) -> dict[str, float]:
    b0: dict[str, float] = {}
    for path, line_no, row in _read_rows(paths):
        pair_group = row.get("pair_group")
        score = row.get("target_logprob_mean")
        if pair_group is None or score is None:
            continue
        try:
            restored = _restored_ids(row)
        except ValueError:
            continue
        if restored:
            continue
        value = float(score)
        if not math.isfinite(value):
            raise ValueError(f"{path}:{line_no} non-finite B0 score")
        pair_group = str(pair_group)
        previous = b0.get(pair_group)
        if previous is not None and abs(previous - value) > 1e-8:
            raise ValueError(f"conflicting B0 scores for {pair_group}")
        b0[pair_group] = value
    return b0


def build_cache_rows(
    *,
    targets: Mapping[str, str],
    b0_scores: Mapping[str, float],
    score_sources: Sequence[tuple[str, Sequence[Path]]],
    prompt_revision: str,
    hgkv_checkpoint_sha256: str,
    b0_policy_sha256: str,
) -> list[dict[str, Any]]:
    cache: dict[CoalitionCacheKey, dict[str, Any]] = {}
    for pair_group, b0_score in sorted(b0_scores.items()):
        if pair_group not in targets:
            raise KeyError(f"missing target for B0 pair group {pair_group}")
        key = CoalitionCacheKey.build(
            pair_group=pair_group,
            restored_event_step_ids=(),
            target_text=targets[pair_group],
            prompt_revision=prompt_revision,
            hgkv_checkpoint_sha256=hgkv_checkpoint_sha256,
            b0_policy_sha256=b0_policy_sha256,
        )
        cache[key] = {
            **key.to_mapping(),
            "restored_event_step_ids": [],
            "target_logprob_mean": float(b0_score),
            "b0_target_logprob_mean": float(b0_score),
            "u_act": 0.0,
            "source_artifacts": ["b0_exact_join"],
            "source_versions": ["frozen_b0"],
        }
    for source_version, paths in score_sources:
        for path, line_no, row in _read_rows(paths):
            pair_group_value = row.get("pair_group")
            score_value = row.get("target_logprob_mean")
            if pair_group_value is None or score_value is None:
                continue
            pair_group = str(pair_group_value)
            if pair_group not in targets:
                raise KeyError(f"{path}:{line_no} missing target for {pair_group}")
            if pair_group not in b0_scores:
                raise KeyError(f"{path}:{line_no} missing B0 for {pair_group}")
            restored = _restored_ids(row)
            score = float(score_value)
            if not math.isfinite(score):
                raise ValueError(f"{path}:{line_no} non-finite score")
            key = CoalitionCacheKey.build(
                pair_group=pair_group,
                restored_event_step_ids=restored,
                target_text=targets[pair_group],
                prompt_revision=prompt_revision,
                hgkv_checkpoint_sha256=hgkv_checkpoint_sha256,
                b0_policy_sha256=b0_policy_sha256,
            )
            u_act = score - float(b0_scores[pair_group])
            candidate = {
                **key.to_mapping(),
                "restored_event_step_ids": list(restored),
                "target_logprob_mean": score,
                "b0_target_logprob_mean": float(b0_scores[pair_group]),
                "u_act": u_act,
                "source_artifacts": [str(path)],
                "source_versions": [source_version],
            }
            previous = cache.get(key)
            if previous is None:
                cache[key] = candidate
                continue
            if (
                abs(previous["target_logprob_mean"] - score) > 1e-8
                or abs(previous["u_act"] - u_act) > 1e-8
            ):
                raise ValueError(
                    f"conflicting exact-key coalition score for {key}"
                )
            if str(path) not in previous["source_artifacts"]:
                previous["source_artifacts"].append(str(path))
            if source_version not in previous["source_versions"]:
                previous["source_versions"].append(source_version)
    rows = [cache[key] for key in sorted(cache)]
    for row in rows:
        row["source_artifacts"].sort()
        row["source_versions"].sort()
    return rows


def _parse_source(value: str) -> tuple[str, list[Path]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--score-source must be VERSION=GLOB")
    version, pattern = value.split("=", 1)
    if not version or not pattern:
        raise argparse.ArgumentTypeError("--score-source must be VERSION=GLOB")
    paths = _paths([pattern])
    if not paths:
        raise argparse.ArgumentTypeError(f"score source matched no files: {pattern}")
    return version, paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-samples", action="append", required=True)
    parser.add_argument("--b0-scores", action="append", required=True)
    parser.add_argument(
        "--score-source", action="append", type=_parse_source, required=True
    )
    parser.add_argument("--prompt-revision", required=True)
    parser.add_argument("--hgkv-checkpoint-sha256", required=True)
    parser.add_argument("--b0-policy-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    target_paths = _paths(args.target_samples)
    b0_paths = _paths(args.b0_scores)
    if not target_paths or not b0_paths:
        parser.error("target samples and B0 patterns must match files")
    targets = load_target_actions(target_paths)
    b0 = load_b0_scores(b0_paths)
    rows = build_cache_rows(
        targets=targets,
        b0_scores=b0,
        score_sources=args.score_source,
        prompt_revision=args.prompt_revision,
        hgkv_checkpoint_sha256=args.hgkv_checkpoint_sha256,
        b0_policy_sha256=args.b0_policy_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    all_inputs = sorted(
        set(target_paths + b0_paths + [
            path for _version, paths in args.score_source for path in paths
        ])
    )
    manifest = {
        "schema_version": "causalcache.hgkv_selector_v2.coalition_cache.v1",
        "rows": len(rows),
        "unique_keys": len(rows),
        "target_pair_groups": len(targets),
        "b0_pair_groups": len(b0),
        "prompt_revision": args.prompt_revision,
        "hgkv_checkpoint_sha256": args.hgkv_checkpoint_sha256,
        "b0_policy_sha256": args.b0_policy_sha256,
        "source_commit": args.source_commit,
        "input_sha256": {
            str(path): _file_sha256(path) for path in all_inputs
        },
        "output_sha256": _file_sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
