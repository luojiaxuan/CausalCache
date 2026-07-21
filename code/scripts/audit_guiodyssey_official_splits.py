#!/usr/bin/env python3
"""Cross-reference frozen trajectory roles against official GUI-Odyssey splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download


OFFICIAL_REPO = "OpenGVLab/GUI-Odyssey"
OFFICIAL_SPLITS = ("random_split", "task_split", "device_split", "app_split")
ROLES = ("train", "tune", "evaluation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--official-revision", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.assignment_manifest.read_text(encoding="utf-8"))
    assignments = payload["assignments"]
    role_counts = Counter(row["role"] for row in assignments)
    if set(role_counts) != set(ROLES):
        raise ValueError("assignment manifest roles drifted")

    audit: dict[str, object] = {
        "assignment_manifest_sha256": hashlib.sha256(
            args.assignment_manifest.read_bytes()
        ).hexdigest(),
        "official_repo": OFFICIAL_REPO,
        "our_role_counts": dict(sorted(role_counts.items())),
        "schema_version": "causalcache.guiodyssey_official_split_audit.v1",
    }
    splits: dict[str, object] = {}
    missing_ids: set[str] = set()
    for name in OFFICIAL_SPLITS:
        path = hf_hub_download(
            OFFICIAL_REPO,
            repo_type="dataset",
            revision=args.official_revision,
            filename=f"splits/{name}.json",
        )
        split = json.loads(Path(path).read_text(encoding="utf-8"))
        train = {value.removesuffix(".json") for value in split["train"]}
        test = {value.removesuffix(".json") for value in split["test"]}
        if train & test:
            raise ValueError(f"official {name} train/test lists overlap")
        table: dict[str, Counter[str]] = defaultdict(Counter)
        for row in assignments:
            trajectory_id = row["trajectory_id"]
            if trajectory_id in train:
                location = "official_train"
            elif trajectory_id in test:
                location = "official_test"
            else:
                location = "missing_from_official_lists"
                missing_ids.add(trajectory_id)
            table[row["role"]][location] += 1
        splits[name] = {
            "official_train_count": len(train),
            "official_test_count": len(test),
            "our_roles": {
                role: dict(sorted(table[role].items())) for role in ROLES
            },
        }
    audit["official_splits"] = splits
    audit["missing_trajectory_ids"] = sorted(missing_ids)
    audit["missing_trajectory_count"] = len(missing_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["official_splits"], sort_keys=True))


if __name__ == "__main__":
    main()
