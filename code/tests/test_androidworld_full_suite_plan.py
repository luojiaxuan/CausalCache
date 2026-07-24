from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from scripts.build_androidworld_validation_plan import records_sha256
from scripts.run_exploratory_closed_loop_episode import _load_plan_instance


ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "data" / "manifests"
FULL_PLAN_PATH = MANIFESTS / "androidworld_full_suite_plan_v1.json"
SOURCE_PLANS = {
    "test": "androidworld_test_plan_v1.json",
    "train": "androidworld_train_plan_v1.json",
    "validation": "androidworld_validation_plan_v2.json",
}
EXPECTED_TEMPLATE_COUNT = 116
KEPT_TASK_INDICES = {0, 1}
EXPECTED_INSTANCE_COUNT = EXPECTED_TEMPLATE_COUNT * len(KEPT_TASK_INDICES)
EXPECTED_ORIGIN_BREAKDOWN = {"test": 50, "train": 120, "validation": 62}


def _full_plan() -> dict:
    return json.loads(FULL_PLAN_PATH.read_text(encoding="utf-8"))


def test_full_plan_holds_two_instances_per_116_templates() -> None:
    plan = _full_plan()
    assert plan["split"] == "full"
    assert plan["task_type_count"] == EXPECTED_TEMPLATE_COUNT
    assert plan["task_instance_count"] == EXPECTED_INSTANCE_COUNT
    # note (luojiaxuan): 每模板两个变体,维持
    # task_instance_count == task_type_count * task_combinations 不变量。
    assert plan["task_combinations"] == 2
    assert (
        plan["task_type_count"] * plan["task_combinations"]
        == plan["task_instance_count"]
    )

    instances = plan["instances"]
    assert len(instances) == EXPECTED_INSTANCE_COUNT
    task_types = [item["task_type"] for item in instances]
    assert len(set(task_types)) == EXPECTED_TEMPLATE_COUNT

    per_template: dict[str, set[int]] = defaultdict(set)
    for item in instances:
        per_template[item["task_type"]].add(item["task_index"])
    # note (luojiaxuan): 每 template 恰好持有 task_index {0, 1}。
    assert all(
        indices == KEPT_TASK_INDICES for indices in per_template.values()
    )

    # note (luojiaxuan): 实例先按 task_type、再按 task_index 确定性排序。
    keys = [(item["task_type"], item["task_index"]) for item in instances]
    assert keys == sorted(keys)


def test_instance_records_sha256_recomputes_with_shared_convention() -> None:
    plan = _full_plan()
    assert plan["instance_records_sha256"] == records_sha256(plan["instances"])


def test_union_of_source_split_templates_equals_full_plan() -> None:
    plan = _full_plan()
    full_templates = {item["task_type"] for item in plan["instances"]}

    union: set[str] = set()
    for file_name in SOURCE_PLANS.values():
        source = json.loads((MANIFESTS / file_name).read_text(encoding="utf-8"))
        split_templates = {
            item["task_type"]
            for item in source["instances"]
            if item["task_index"] == 0
        }
        # note (luojiaxuan): 三个 split 的模板互不相交。
        assert union.isdisjoint(split_templates)
        union |= split_templates

    assert union == full_templates
    assert len(union) == EXPECTED_TEMPLATE_COUNT


def test_origin_split_is_present_and_consistent_with_source_plans() -> None:
    plan = _full_plan()
    assert all("origin_split" in item for item in plan["instances"])

    template_to_origin = {
        item["task_type"]: item["origin_split"] for item in plan["instances"]
    }
    assert set(template_to_origin.values()) == set(SOURCE_PLANS)

    for split, file_name in SOURCE_PLANS.items():
        source = json.loads((MANIFESTS / file_name).read_text(encoding="utf-8"))
        for item in source["instances"]:
            if item["task_index"] not in KEPT_TASK_INDICES:
                continue
            assert template_to_origin[item["task_type"]] == split

    # note (luojiaxuan): origin_split 分布应为 test 50 / train 120 / validation 62。
    breakdown = {
        split: sum(
            1 for item in plan["instances"] if item["origin_split"] == split
        )
        for split in SOURCE_PLANS
    }
    assert breakdown == EXPECTED_ORIGIN_BREAKDOWN
    assert sum(breakdown.values()) == EXPECTED_INSTANCE_COUNT


def test_full_plan_instances_preserve_source_records_verbatim() -> None:
    plan = _full_plan()
    full_by_key = {
        (item["task_type"], item["task_index"]): item
        for item in plan["instances"]
    }
    seen = 0
    for file_name in SOURCE_PLANS.values():
        source = json.loads((MANIFESTS / file_name).read_text(encoding="utf-8"))
        for item in source["instances"]:
            if item["task_index"] not in KEPT_TASK_INDICES:
                continue
            merged = full_by_key[(item["task_type"], item["task_index"])]
            # note (luojiaxuan): 除新增 origin_split 外,其余字段逐字保留。
            assert {k: merged[k] for k in item} == item
            assert set(merged) == set(item) | {"origin_split"}
            seen += 1
    assert seen == EXPECTED_INSTANCE_COUNT


@pytest.mark.parametrize("task_index", sorted(KEPT_TASK_INDICES))
def test_guard_loads_full_split_only_with_allow_sealed(task_index: int) -> None:
    plan = _full_plan()
    task_type = plan["instances"][0]["task_type"]

    with pytest.raises(ValueError, match="sealed"):
        _load_plan_instance(
            FULL_PLAN_PATH, task_type=task_type, task_index=task_index
        )

    loaded = _load_plan_instance(
        FULL_PLAN_PATH,
        task_type=task_type,
        task_index=task_index,
        allow_sealed=True,
    )
    assert loaded["split"] == "full"
    assert loaded["instance"]["task_type"] == task_type
    assert loaded["instance"]["task_index"] == task_index


def test_guard_still_refuses_unknown_split_even_with_allow_sealed(
    tmp_path: Path,
) -> None:
    plan = _full_plan()
    plan["split"] = "canary"
    unknown = tmp_path / "androidworld_canary_plan.json"
    unknown.write_text(json.dumps(plan), encoding="utf-8")
    task_type = plan["instances"][0]["task_type"]
    with pytest.raises(ValueError, match="sealed"):
        _load_plan_instance(
            unknown, task_type=task_type, task_index=0, allow_sealed=True
        )
