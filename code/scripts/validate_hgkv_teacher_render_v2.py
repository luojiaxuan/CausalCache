#!/usr/bin/env python3
"""Validate exact missing-coalition coverage of a V2 teacher render depth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from scripts.plan_hgkv_teacher_beam_v2 import expand_paths, sha256_file
from scripts.render_hgkv_teacher_beam_v2 import load_missing_coalitions


def image_paths(sample: dict[str, Any], root: Path) -> set[Path]:
    paths: set[Path] = set()
    for message in sample["messages"]:
        for content in message["content"]:
            if content.get("type") == "image":
                paths.add(root / str(content["path"]))
    return paths


def validate_render(
    *,
    plan_paths: Iterable[Path],
    sample_paths: Iterable[Path],
) -> dict[str, Any]:
    expected = load_missing_coalitions(plan_paths)
    observed: set[str] = set()
    images: set[Path] = set()
    sample_paths = list(sample_paths)
    for path in sample_paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            key = (
                f"{sample['pair_group']}/"
                f"{sample['restored_set_key']}"
            )
            declared = expected.get(key)
            if declared is None:
                raise ValueError(f"{path}:{line_no} unexpected coalition {key}")
            if key in observed:
                raise ValueError(f"{path}:{line_no} duplicate coalition {key}")
            if sample.get("schema_version") != (
                "causalcache.hgkv_selector_v2.teacher_coalition_sample.v1"
            ):
                raise ValueError(f"{path}:{line_no} schema drifted")
            if (
                sample["memory_config"]["restored_event_step_ids"]
                != declared["selected_event_step_ids"]
                or int(sample["teacher_depth"]) != int(declared["depth"])
                or int(sample["decision_step_id"])
                != int(declared["decision_step"])
            ):
                raise ValueError(f"{path}:{line_no} coalition metadata drifted")
            observed.add(key)
            images.update(image_paths(sample, path.parent))
    missing = set(expected) - observed
    if missing:
        raise ValueError(f"teacher render is missing {len(missing)} coalitions")
    image_hashes: dict[str, str] = {}
    for path in sorted(images):
        if not path.is_file():
            raise ValueError(f"teacher render references missing image {path}")
        image_hashes[str(path)] = sha256_file(path)
    image_manifest_sha256 = hashlib.sha256(
        json.dumps(image_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": (
            "causalcache.hgkv_selector_v2.teacher_render_done.v1"
        ),
        "depth": (
            None
            if not expected
            else next(iter(expected.values()))["depth"]
        ),
        "expected_coalitions": len(expected),
        "observed_coalitions": len(observed),
        "unique_coalitions": len(observed),
        "duplicate_rows": 0,
        "unexpected_rows": 0,
        "missing_rows": 0,
        "referenced_images": len(images),
        "image_manifest_sha256": image_manifest_sha256,
        "sample_sha256": {
            str(path): sha256_file(path) for path in sample_paths
        },
        "status": "DONE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="append", required=True)
    parser.add_argument("--samples", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_paths = expand_paths(args.plan)
    sample_paths = expand_paths(args.samples)
    if not plan_paths or not sample_paths:
        parser.error("plan and sample patterns must match files")
    result = validate_render(
        plan_paths=plan_paths,
        sample_paths=sample_paths,
    )
    result["plan_sha256"] = {
        str(path): sha256_file(path) for path in plan_paths
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
