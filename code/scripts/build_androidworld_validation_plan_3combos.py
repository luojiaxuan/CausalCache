#!/usr/bin/env python3
"""Regenerate the validation split with 3 task combinations (adds task_index 2).

# note (luojiaxuan): 主表改为每模板 3 instance 降方差;test/train 源本就含
# index 2,只有 validation 源(v2)是 2 combinations。此脚本用 validation 自身冻结的
# suite_seed=271828 重新 create_suite(n=3),index 0/1 必与 v2 逐字节一致(脚本内断言),
# 仅新增 index 2。生成纯 python(registry+seed),不需 emulator。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_androidworld_validation_plan import build_plan, records_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v2 = json.loads(args.validation_v2.read_text(encoding="utf-8"))
    task_types = sorted({i["task_type"] for i in v2["instances"]})
    partition = {
        "splits": {
            "validation": {
                "task_types": task_types,
                "task_combinations": 3,
                "suite_seed": v2["suite_seed"],
                "task_type_count": len(task_types),
                "task_instance_count": len(task_types) * 3,
            }
        },
        "registry": {"sorted_task_types_sha256": v2["registry_sha256"]},
    }
    stack = {"benchmark": v2["source"]}
    plan = build_plan(stack=stack, partition=partition, split="validation")

    # note (luojiaxuan): parity 断言——新生成的 index 0/1 必与 v2 完全一致,
    # 否则"跨方法固定 instance"承诺被破坏,fail loud。
    v2_by_key = {(i["task_type"], i["task_index"]): i for i in v2["instances"]}
    for inst in plan["instances"]:
        if inst["task_index"] in (0, 1):
            ref = v2_by_key[(inst["task_type"], inst["task_index"])]
            for field in ("goal", "template", "max_steps", "complexity"):
                if inst[field] != ref[field]:
                    raise SystemExit(
                        f"PARITY FAIL {inst['task_type']} idx{inst['task_index']} "
                        f"{field}: {inst[field]!r} != {ref[field]!r}"
                    )
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    idx2 = [i for i in plan["instances"] if i["task_index"] == 2]
    print(
        json.dumps(
            {
                "parity_index01": "PASS",
                "total_instances": len(plan["instances"]),
                "new_index2": len(idx2),
                "records_sha256": records_sha256(plan["instances"]),
            }
        )
    )


if __name__ == "__main__":
    main()
