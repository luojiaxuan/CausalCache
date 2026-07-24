#!/usr/bin/env python3
"""Validate selected-set renders against the frozen gate plan."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.build_odyssey_sft_dataset import restored_set_key
from scripts.train_hgkv_singleton_selector_v1 import (
    atomic_write_json,
    sha256_file,
)


def expected_from_plan(
    plan_path: Path,
) -> dict[tuple[str, str], dict[str, set[Any]]]:
    expected: dict[tuple[str, str], dict[str, set[Any]]] = defaultdict(
        lambda: {"budgets": set(), "methods": set(), "rows": set()}
    )
    seen_rows = set()
    with plan_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pair_group = str(row["pair_group"])
            method = str(row["method"])
            budget = int(row["budget"])
            identity = (pair_group, method, budget)
            if identity in seen_rows:
                raise ValueError(f"duplicate gate-plan row: {identity}")
            seen_rows.add(identity)
            selected_key = restored_set_key(
                row["selected_event_step_ids"]
            )
            item = expected[(pair_group, selected_key)]
            item["budgets"].add(budget)
            item["methods"].add(method)
            item["rows"].add((method, budget))
    return expected


def validate(
    *,
    plan_path: Path,
    render_dir: Path,
    samples_glob: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    expected = expected_from_plan(plan_path)
    observed = set()
    input_hashes = {}
    missing_images = []
    for path in sorted(render_dir.glob(samples_glob)):
        input_hashes[path.name] = sha256_file(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if row["variant"] != "selected_set":
                    raise ValueError(
                        f"unexpected variant in {path}:{line_number}"
                    )
                restored = row["memory_config"][
                    "restored_event_step_ids"
                ]
                selected_key = restored_set_key(restored)
                if str(row["restored_set_key"]) != selected_key:
                    raise ValueError(
                        f"restored_set_key mismatch in {path}:{line_number}"
                    )
                key = (str(row["pair_group"]), selected_key)
                if key in observed:
                    raise ValueError(f"duplicate selected-set render: {key}")
                if key not in expected:
                    raise ValueError(f"unexpected selected-set render: {key}")
                expected_item = expected[key]
                if set(row["selected_set_budgets"]) != expected_item["budgets"]:
                    raise ValueError(f"selected-set budget mismatch: {key}")
                if set(row["selected_set_methods"]) != expected_item["methods"]:
                    raise ValueError(f"selected-set method mismatch: {key}")
                for message in row["messages"]:
                    for part in message["content"]:
                        if part.get("type") != "image":
                            continue
                        image_path = render_dir / part["path"]
                        if not image_path.is_file() or image_path.stat().st_size == 0:
                            missing_images.append(str(image_path))
                observed.add(key)
    if not input_hashes:
        raise FileNotFoundError(
            f"{render_dir} contains no inputs matching {samples_glob!r}"
        )
    missing = set(expected) - observed
    if missing:
        raise ValueError(f"selected-set render coverage miss: {len(missing)}")
    if missing_images:
        raise ValueError(f"missing rendered images: {len(missing_images)}")
    return (
        {
            "expected_unique_sets": len(expected),
            "missing_images": 0,
            "observed_unique_sets": len(observed),
            "plan_rows": sum(len(item["rows"]) for item in expected.values()),
            "status": "DONE",
        },
        input_hashes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--samples-glob", default="samples-shard*.jsonl")
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    summary, input_hashes = validate(
        plan_path=args.plan,
        render_dir=args.render_dir,
        samples_glob=args.samples_glob,
    )
    summary.update(
        {
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input_files": input_hashes,
            "plan_sha256": sha256_file(args.plan),
            "source_commit": args.source_commit,
        }
    )
    atomic_write_json(args.render_dir / "DONE", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
