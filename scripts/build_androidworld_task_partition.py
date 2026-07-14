"""Build the preregistered AndroidWorld task-template partition manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.run_androidworld_environment_smoke import request_json


SPLIT_NAMES = ("train", "validation", "test")


def task_bucket(task_type: str) -> int:
    digest = hashlib.sha256(task_type.encode("utf-8")).hexdigest()
    return int(digest, 16) % 10


def registry_sha256(task_types: list[str]) -> str:
    payload = "\n".join(sorted(task_types)).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def build_manifest(task_types: list[str], stack: dict[str, Any]) -> dict[str, Any]:
    if len(task_types) != len(set(task_types)):
        raise ValueError("registry contains duplicate task types")
    partition = stack["task_partition"]
    bucket_to_split: dict[int, str] = {}
    for split in SPLIT_NAMES:
        for bucket in partition[f"{split}_buckets"]:
            if bucket in bucket_to_split:
                raise ValueError(f"bucket {bucket} appears in multiple splits")
            bucket_to_split[bucket] = split
    if set(bucket_to_split) != set(range(10)):
        raise ValueError("task partition must cover buckets 0 through 9")

    split_records: dict[str, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        split_records[split] = {
            "buckets": partition[f"{split}_buckets"],
            "suite_seed": partition[f"{split}_suite_seed"],
            "task_combinations": partition[f"{split}_task_combinations"],
            "task_types": [],
        }

    tasks = []
    for task_type in sorted(task_types):
        bucket = task_bucket(task_type)
        split = bucket_to_split[bucket]
        split_records[split]["task_types"].append(task_type)
        tasks.append(
            {
                "task_type": task_type,
                "bucket": bucket,
                "split": split,
                "suite_seed": split_records[split]["suite_seed"],
                "task_combinations": split_records[split]["task_combinations"],
            }
        )

    for split in SPLIT_NAMES:
        split_records[split]["task_type_count"] = len(
            split_records[split]["task_types"]
        )
        split_records[split]["task_instance_count"] = (
            split_records[split]["task_type_count"]
            * split_records[split]["task_combinations"]
        )

    return {
        "schema_version": "0.1.0",
        "source": stack["benchmark"],
        "assignment": partition["assignment"],
        "registry": {
            "task_type_count": len(task_types),
            "sorted_task_types_sha256": registry_sha256(task_types),
        },
        "splits": split_records,
        "tasks": tasks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--expected-task-types", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stack = json.loads(args.stack_config.read_text(encoding="utf-8"))
    task_types = request_json(
        args.base_url, "/suite/task_list", params={"max_index": -1}
    )["task_list"]
    if len(task_types) != args.expected_task_types:
        raise ValueError(
            f"expected {args.expected_task_types} task types, received {len(task_types)}"
        )
    manifest = build_manifest(task_types, stack)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["registry"], indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                split: {
                    "task_type_count": manifest["splits"][split]["task_type_count"],
                    "task_instance_count": manifest["splits"][split][
                        "task_instance_count"
                    ],
                }
                for split in SPLIT_NAMES
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
