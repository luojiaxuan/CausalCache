"""Merge the three AndroidWorld split plans into the full 116-template suite.

AndroidWorld evaluation is now pure zero-shot for every policy in the paper
(frozen / full-layer LoRA / history-gated); none of them trains on AndroidWorld.
The historical train/dev/test split is therefore a vestige for reporting: the
whole 116-template benchmark is the test set. This script produces a single
``split="full"`` plan holding two instances per template (task_index 0 and 1),
merged verbatim from the three committed source plans. Two variations per
template are kept because official AndroidWorld guidance is that task variation
materially affects completion rate, so one instance per template is too noisy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# note (luojiaxuan): 复用 validation plan 的 records_sha256,保证 full plan 的
# instance_records_sha256 与现有三个 plan 用完全相同的规范序列化 + sha256 约定。
from scripts.build_androidworld_validation_plan import records_sha256


# note (luojiaxuan): 合并顺序固定为 (split -> 源文件),按 split 名字典序,保证
# merged_from / source_splits 的确定性;实例本身另按 task_type 排序。
SOURCE_PLANS: tuple[tuple[str, str], ...] = (
    ("test", "androidworld_test_plan_v1.json"),
    ("train", "androidworld_train_plan_v1.json"),
    ("validation", "androidworld_validation_plan_v3.json"),
)

# note (luojiaxuan): full plan 跨三个不同 suite_seed,不存在单一可复现的生成
# seed;权威 per-split seed 保存在 source_splits。顶层 suite_seed 取仓库默认常量
# 271828 作为名义占位,task_combinations 固定为 2(每模板恰好两个实例
# task_index 0 与 1),以维持
# task_instance_count == task_type_count * task_combinations 不变量。环境
# initialize() 自带 live-goal 身份校验,任何 seed 误配都会显式报错而非静默跑错任务。
NOMINAL_SUITE_SEED = 271828
EXPECTED_TEMPLATE_COUNT = 116
# note (luojiaxuan): 主表改为每模板 3 instance 降方差(2026-07-24);三源均含
# index 0/1/2(test/train 原生 3 combos,val 由 build_androidworld_validation_plan_3combos
# 用 seed 271828 重生成 v3,index 0/1 与 v2 逐字节一致)。缺任一 index 的模板 fail loud。
KEPT_TASK_INDICES: tuple[int, ...] = (0, 1, 2)
EXPECTED_INSTANCE_COUNT = EXPECTED_TEMPLATE_COUNT * len(KEPT_TASK_INDICES)


def _load_plan(manifests_dir: Path, file_name: str) -> dict[str, Any]:
    return json.loads((manifests_dir / file_name).read_text(encoding="utf-8"))


def build_full_suite_plan(manifests_dir: Path) -> dict[str, Any]:
    instances: list[dict[str, Any]] = []
    source_splits: dict[str, dict[str, Any]] = {}
    merged_from: list[str] = []
    registry_sha256: str | None = None
    source: dict[str, Any] | None = None

    for split, file_name in SOURCE_PLANS:
        plan = _load_plan(manifests_dir, file_name)
        if plan["split"] != split:
            raise ValueError(
                f"{file_name} declares split {plan['split']!r}, expected {split!r}"
            )
        if registry_sha256 is None:
            registry_sha256 = plan["registry_sha256"]
            source = plan["source"]
        elif plan["registry_sha256"] != registry_sha256:
            raise ValueError("source plans disagree on registry_sha256")
        elif plan["source"] != source:
            raise ValueError("source plans disagree on benchmark source")

        merged_from.append(f"{file_name}@{split}")
        source_splits[split] = {
            "source_file": file_name,
            "suite_seed": int(plan["suite_seed"]),
            "task_combinations": int(plan["task_combinations"]),
            "task_type_count": int(plan["task_type_count"]),
            "instance_records_sha256": plan["instance_records_sha256"],
        }

        # note (luojiaxuan): 先按 task_type 归拢源 plan 的可用 task_index,便于
        # 逐模板校验 KEPT_TASK_INDICES 是否齐全,缺失时能报出具体 template 名。
        available: dict[str, dict[int, dict[str, Any]]] = {}
        for record in plan["instances"]:
            available.setdefault(record["task_type"], {})[
                record["task_index"]
            ] = record

        missing: list[str] = []
        for task_type, by_index in sorted(available.items()):
            for wanted in KEPT_TASK_INDICES:
                if wanted not in by_index:
                    missing.append(f"{split}:{task_type}[{wanted}]")
        if missing:
            raise ValueError(
                "source plan is missing required task variations: "
                + ", ".join(missing)
            )

        for task_type, by_index in sorted(available.items()):
            for wanted in KEPT_TASK_INDICES:
                merged = dict(by_index[wanted])
                # note (luojiaxuan): 记录原始 split,合并后不丢任何来源信息。
                merged["origin_split"] = split
                instances.append(merged)

    # note (luojiaxuan): 三个 split 的 task_type 互不相交;先按 task_type、再按
    # task_index 排序,得到确定性全序。
    instances.sort(key=lambda item: (item["task_type"], item["task_index"]))

    task_types = [item["task_type"] for item in instances]
    unique_task_types = sorted(set(task_types))
    if len(unique_task_types) != EXPECTED_TEMPLATE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_TEMPLATE_COUNT} templates, "
            f"got {len(unique_task_types)}"
        )
    if len(instances) != EXPECTED_INSTANCE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_INSTANCE_COUNT} instances, got {len(instances)}"
        )
    # note (luojiaxuan): 每模板必须恰好持有 KEPT_TASK_INDICES 这组 index。
    per_template: dict[str, set[int]] = {}
    for item in instances:
        per_template.setdefault(item["task_type"], set()).add(item["task_index"])
    bad = sorted(
        task_type
        for task_type, indices in per_template.items()
        if indices != set(KEPT_TASK_INDICES)
    )
    if bad:
        raise ValueError(
            "templates without exactly the kept task indices: " + ", ".join(bad)
        )

    return {
        "schema_version": "0.1.0",
        "split": "full",
        "source": source,
        "merged_from": merged_from,
        "source_splits": source_splits,
        "registry_sha256": registry_sha256,
        "suite_seed": NOMINAL_SUITE_SEED,
        "task_combinations": len(KEPT_TASK_INDICES),
        "task_type_count": len(unique_task_types),
        "task_instance_count": len(instances),
        "instance_records_sha256": records_sha256(instances),
        "instances": instances,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_full_suite_plan(args.manifests_dir)
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
                "origin_split_breakdown": {
                    split: sum(
                        1
                        for item in plan["instances"]
                        if item["origin_split"] == split
                    )
                    for split, _ in SOURCE_PLANS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
