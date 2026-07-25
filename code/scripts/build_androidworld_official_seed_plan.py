#!/usr/bin/env python3
"""Build the official-seed roster: task_random_seed=30, n_task_combinations=1.

# note (luojiaxuan): 官方 run_ma35.py 的默认值(seed 30 / 1 combination / 全部任务)。
# 用它可以拿到与官方逐任务同一批实例,消除"任务变体不同"这个复现差异来源。
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--templates-from", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=30)
    p.add_argument("--combinations", type=int, default=1)
    args = p.parse_args()

    from android_world import registry as aw_registry_module
    from android_world import suite_utils

    ref = json.loads(args.templates_from.read_text(encoding="utf-8"))
    task_types = sorted({i["task_type"] for i in ref["instances"]})
    reg = aw_registry_module.TaskRegistry()
    registry = reg.get_registry(reg.ANDROID_WORLD_FAMILY)
    suite = suite_utils.create_suite(
        task_registry=registry,
        n_task_combinations=args.combinations,
        seed=args.seed,
        tasks=task_types,
    )
    instances = []
    for tt in sorted(task_types):
        for idx, task in enumerate(suite[tt]):
            c = float(task.complexity)
            instances.append({
                "task_type": tt, "task_index": idx, "goal": task.goal,
                "template": task.template, "complexity": c,
                "max_steps": int(10 * c),
                "start_on_home_screen": bool(task.start_on_home_screen),
                "origin_split": "official_seed30",
            })
    plan = {
        "schema_version": "0.1.0", "split": "full",
        "source": ref.get("source"), "registry_sha256": ref.get("registry_sha256"),
        "suite_seed": args.seed, "task_combinations": args.combinations,
        "task_type_count": len(task_types), "task_instance_count": len(instances),
        "instances": instances,
    }
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    diff = sum(1 for i in instances
               if any(r["task_type"] == i["task_type"] and r["task_index"] == i["task_index"]
                      and r["goal"] != i["goal"] for r in ref["instances"]))
    print(json.dumps({"instances": len(instances), "templates": len(task_types),
                      "goals_differing_from_our_roster": diff}))


if __name__ == "__main__":
    main()
