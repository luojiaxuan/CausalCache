"""Build an official-budget execution plan for a non-test AndroidWorld split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_SPLITS = ("train", "validation")


def records_sha256(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_plan(
    *,
    stack: dict[str, Any],
    partition: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"refusing to instantiate sealed split: {split}")

    from android_world import registry as aw_registry_module
    from android_world import suite_utils

    split_config = partition["splits"][split]
    task_types = split_config["task_types"]
    task_registry = aw_registry_module.TaskRegistry()
    registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
    suite = suite_utils.create_suite(
        task_registry=registry,
        n_task_combinations=split_config["task_combinations"],
        seed=split_config["suite_seed"],
        tasks=task_types,
    )
    if set(suite.keys()) != set(task_types):
        raise ValueError("created suite does not match frozen task types")

    instances = []
    for task_type in sorted(task_types):
        for task_index, task in enumerate(suite[task_type]):
            complexity = float(task.complexity)
            instances.append(
                {
                    "task_type": task_type,
                    "task_index": task_index,
                    "goal": task.goal,
                    "template": task.template,
                    "complexity": complexity,
                    "max_steps": int(10 * complexity),
                    "start_on_home_screen": bool(task.start_on_home_screen),
                }
            )

    if len(instances) != split_config["task_instance_count"]:
        raise ValueError("created suite instance count does not match manifest")
    return {
        "schema_version": "0.1.0",
        "split": split,
        "source": stack["benchmark"],
        "registry_sha256": partition["registry"]["sorted_task_types_sha256"],
        "suite_seed": split_config["suite_seed"],
        "task_combinations": split_config["task_combinations"],
        "task_type_count": split_config["task_type_count"],
        "task_instance_count": len(instances),
        "instance_records_sha256": records_sha256(instances),
        "instances": instances,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--split", choices=ALLOWED_SPLITS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stack = json.loads(args.stack_config.read_text(encoding="utf-8"))
    partition = json.loads(args.partition.read_text(encoding="utf-8"))
    plan = build_plan(stack=stack, partition=partition, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "split": plan["split"],
                "task_type_count": plan["task_type_count"],
                "task_instance_count": plan["task_instance_count"],
                "instance_records_sha256": plan["instance_records_sha256"],
                "maximum_step_budget": max(
                    instance["max_steps"] for instance in plan["instances"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
