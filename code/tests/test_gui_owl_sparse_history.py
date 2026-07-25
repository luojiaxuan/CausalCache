#!/usr/bin/env python3
"""Prompt-builder 与五臂样本构造的契约测试(pytest,默认无需 torch/GPU)。

# note (luojiaxuan): 第二轮重写。第一轮这套用例是与实现并行写的,pair-group 构造入口
# 靠一张候选名表(build_pair_group_samples / build_arm_samples / build_group_samples)
# 去 getattr 试探 —— 三个名字当时一个都不存在,17 PASS / 22 FAIL。候选名表这种写法本身
# 也有问题:它让"函数改名了"和"函数没写"看起来一样,而且测试永远不会因为 API 漂移
# 而失败。现在直接 import ``build_pair_group_samples`` 并对**真实签名**下断言,名字或
# 参数一变,失败点就是那一条签名用例。
#
# 被测对象始终是仓库里的真代码;只对 torch / PIL / pyarrow / 重量级 runtime 做
# "仅在缺失时"的桩替换,好让纯逻辑部分在无 GPU 机器上也能跑。断言总数交给 pytest 计数,
# 不再硬编码(第一轮写死 "14 - len(fails)" 而实际 15 项,报表永远错位)。
"""

from __future__ import annotations

import importlib
import inspect
import random
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# 仓库路径与重依赖桩(在 import 被测模块之前完成)
# ---------------------------------------------------------------------------
def _repo_code_root() -> Path:
    """Locate the repository ``code/`` root that holds causalcache/ and scripts/."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "causalcache").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository code root from the test file")


_CODE_ROOT = _repo_code_root()
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

STUBBED: set[str] = set()


def stub_if_missing(name: str, **attributes: Any) -> None:
    """Register a stand-in module only when the real dependency is unavailable."""
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - 任何 import 失败都退回桩
        pass
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    STUBBED.add(name)
    parent_name, _, leaf = name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, leaf, module)


stub_if_missing("torch")
stub_if_missing("PIL")
stub_if_missing("PIL.Image", open=lambda *args, **kwargs: None)
stub_if_missing("pyarrow")
stub_if_missing("pyarrow.parquet")
stub_if_missing(
    "causalcache.policy.gui_owl_v2_1_runtime", GUIOwlV21OfficialToolsRuntime=object
)
stub_if_missing(
    "scripts.run_exploratory_closed_loop_episode",
    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE=2560,
)

from causalcache.policy.gui_owl_official import (  # noqa: E402
    OFFICIAL_SYSTEM_PROMPT,
)
from causalcache.policy.gui_owl_sparse_history import (  # noqa: E402
    SPARSE_PROTOCOL_ID,
    build_sparse_history_messages,
    sparse_image_count,
)
from scripts import build_sparse_history_dataset as producer  # noqa: E402
from scripts.build_sparse_history_dataset import (  # noqa: E402
    GroupRejected,
    build_pair_group_samples,
)


# ---------------------------------------------------------------------------
# 五臂契约(FIELD CONTRACT v2):slot -> (arm_id, role, prompt_format,
# selection_mode, adapter_mode)。消费侧只认这些显式字段,不认 variant 前缀。
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "causalcache.sparse_history_sample.v2"
ARM_SPEC: dict[str, tuple[str, str, str, str, str]] = {
    "N0": ("N0", "deployment_baseline", "official_multiturn", "recent", "bypass"),
    "R0": ("R0", "reference", "sparse_single_turn", "recent", "bypass"),
    "S0": ("S0", "measurement", "sparse_single_turn", "sparse", "bypass"),
    "RA": ("RA", "measurement", "sparse_single_turn", "recent", "active"),
    "SA": ("SA", "positive", "sparse_single_turn", "sparse", "active"),
    "SA_neg_step_shuffled": (
        "SA", "negative", "sparse_single_turn", "sparse", "active",
    ),
    "SA_neg_irrelevant": ("SA", "negative", "sparse_single_turn", "sparse", "active"),
    "SA_neg_duplicate": ("SA", "negative", "sparse_single_turn", "sparse", "active"),
}
RECENT_SLOTS = ("N0", "R0", "RA")
SPARSE_SLOTS = (
    "S0", "SA", "SA_neg_step_shuffled", "SA_neg_irrelevant", "SA_neg_duplicate",
)
NEGATIVE_KIND = {
    "SA_neg_step_shuffled": "step_shuffled",
    "SA_neg_irrelevant": "irrelevant",
    "SA_neg_duplicate": "duplicate",
}
FORBIDDEN_KEYS = ("sparse", "reference_variant", "_needs_donor")

DECISION_STEP = 12
EPISODE = "EP-A"
DONOR_EPISODE = "EP-B"
TARGET_TEXT = (
    'Action: Open the conversation.\n<tool_call>\n{"name": "mobile_use", '
    '"arguments": {"action": "click", "coordinate": [391, 392]}}\n</tool_call>'
)
ACTION_TEXTS = [f"do-{i}" for i in range(1, 18)]
DONOR_IMAGES = [f"images/{DONOR_EPISODE}/obs-{i:03d}.png" for i in range(8)]


def texts(messages: list[dict[str, Any]]) -> list[str]:
    return [
        part["text"]
        for message in messages
        for part in message["content"]
        if part.get("type") == "text"
    ]


def image_parts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        part
        for message in messages
        for part in message["content"]
        if part.get("type") == "image"
    ]


def part_order(messages: list[dict[str, Any]]) -> list[str]:
    return [
        part.get("type") for message in messages for part in message["content"]
    ]


# ---------------------------------------------------------------------------
# 1. 源码导入与 API 签名
# ---------------------------------------------------------------------------
def test_prompt_builder_imports_from_repository() -> None:
    module = importlib.import_module("causalcache.policy.gui_owl_sparse_history")
    assert Path(module.__file__).resolve().is_relative_to(_CODE_ROOT)
    assert callable(module.build_sparse_history_messages)
    assert callable(module.sparse_image_count)


def test_dataset_builder_imports_from_repository() -> None:
    module = importlib.import_module("scripts.build_sparse_history_dataset")
    assert Path(module.__file__).resolve().is_relative_to(_CODE_ROOT)
    assert callable(module.main)


def test_pair_group_entry_point_signature() -> None:
    """五臂构造必须能在 main() 之外复核 —— 签名断言前置,API 一漂移就在这里失败。

    # note (luojiaxuan): 第一轮用候选名表 getattr 试探三个名字,结果三个都不存在
    # 而测试只报一句"缺少构造函数"。现在直接 import,名字错 = import 期就崩,
    # 参数错 = 这一条失败,不会牵连下面二十多条语义用例。
    """
    parameters = inspect.signature(build_pair_group_samples).parameters
    assert set(parameters) == {
        "episode", "decision_step", "instruction", "action_texts", "selected_steps",
        "target_text", "split", "donor_episode", "donor_images", "full_responses",
    }
    for name in parameters:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} 必须是 keyword-only,位置参数在五臂构造这种宽签名上极易错位"
        )
    assert parameters["split"].default == "train"
    for optional in ("donor_episode", "donor_images", "full_responses"):
        assert parameters[optional].default is None


def test_builder_module_declares_the_frozen_protocol_id() -> None:
    assert SPARSE_PROTOCOL_ID == "causalcache_sparse_history_v1"
    assert producer.SCHEMA_VERSION == SCHEMA_VERSION
    assert producer.REFERENCE_ARM_ID == "R0"
    assert producer.DEPLOYMENT_BASELINE_ARM_ID == "N0"


def test_arm_specs_match_the_frozen_five_arm_contract() -> None:
    """ARM_SPECS 是生产侧唯一的臂表,必须与本文件的契约副本逐字相同。"""
    declared = {
        slot: (arm_id, role, prompt_format, selection_mode, adapter_mode)
        for slot, arm_id, role, prompt_format, selection_mode, adapter_mode
        in producer.ARM_SPECS
    }
    assert declared == {
        slot: spec for slot, spec in ARM_SPEC.items() if not slot.startswith("SA_neg_")
    }
    assert {slot for slot, _kind, _scale, _min in producer.NEGATIVE_SPECS} == set(
        NEGATIVE_KIND
    )


# ---------------------------------------------------------------------------
# 2. sparse prompt 的冻结不变量
# ---------------------------------------------------------------------------
@pytest.fixture()
def rendered() -> list[dict[str, Any]]:
    return build_sparse_history_messages(
        instruction="book a flight",
        action_texts=ACTION_TEXTS,
        selected_steps=[3, 11, 16],
        selected_images=["IMG3", "IMG11", "IMG16"],
        current_step=18,
        current_image="CUR",
    )


def test_history_images_follow_original_step_order(
    rendered: list[dict[str, Any]],
) -> None:
    blob = "\n".join(texts(rendered))
    labels = [
        int(x) for x in re.findall(r"Historical screenshot from Step(\d+)", blob)
    ]
    assert labels == [3, 11, 16]


def test_image_payload_order_matches_steps(rendered: list[dict[str, Any]]) -> None:
    assert [part["image"] for part in image_parts(rendered)] == [
        "IMG3", "IMG11", "IMG16", "CUR",
    ]


def test_current_screenshot_is_last_part(rendered: list[dict[str, Any]]) -> None:
    assert part_order(rendered)[-1] == "image"


def test_skipped_steps_survive_as_text_exactly_once(
    rendered: list[dict[str, Any]],
) -> None:
    blob = "\n".join(texts(rendered))
    skipped = {1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17}
    mentioned = {int(x) for x in re.findall(r"Step(\d+):", blob)}
    assert skipped <= mentioned
    assert all(blob.count(f"Step{i}: do-{i}") == 1 for i in sorted(skipped))


def test_each_history_image_is_followed_by_its_own_action(
    rendered: list[dict[str, Any]],
) -> None:
    blob = "\n".join(texts(rendered))
    assert all(f"Action at Step{s}: do-{s}" in blob for s in (3, 11, 16))


def test_prompt_declares_non_consecutive_selection(
    rendered: list[dict[str, Any]],
) -> None:
    blob = "\n".join(texts(rendered))
    assert "sparse, non-consecutive" in blob
    assert "not necessarily adjacent" in blob


def test_official_system_prompt_is_reused(rendered: list[dict[str, Any]]) -> None:
    assert rendered[0]["role"] == "system"
    assert rendered[0]["content"][0]["text"] == OFFICIAL_SYSTEM_PROMPT


def test_image_count_is_budget_plus_current(rendered: list[dict[str, Any]]) -> None:
    assert sparse_image_count(rendered) == 4


def test_zero_budget_degrades_to_text_only_history() -> None:
    """K=0 退化:没有历史图,历史只以纯文本存在,图数恰好剩当前观测那一张。"""
    messages = build_sparse_history_messages(
        instruction="x",
        action_texts=ACTION_TEXTS,
        selected_steps=[],
        selected_images=[],
        current_step=6,
        current_image="CUR",
    )
    assert sparse_image_count(messages) == 1
    assert "Step5: do-5" in "\n".join(texts(messages))
    assert "Historical screenshot from Step" not in "\n".join(texts(messages))


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    (
        (dict(selected_steps=[11, 3], selected_images=["a", "b"]), "步号乱序"),
        (dict(selected_steps=[3, 20], selected_images=["a", "b"]), "越过当前步"),
        (dict(selected_steps=[3], selected_images=["a", "b"]), "图数不匹配"),
        (dict(selected_steps=[3, 3], selected_images=["a", "b"]), "步号重复"),
        (dict(selected_steps=[0], selected_images=["a"]), "步号小于 1"),
        (dict(selected_steps=[18], selected_images=["a"]), "选点等于当前步"),
    ),
)
def test_builder_is_fail_closed(kwargs: dict[str, Any], reason: str) -> None:
    with pytest.raises(ValueError):
        build_sparse_history_messages(
            instruction="x",
            action_texts=ACTION_TEXTS,
            current_step=18,
            current_image="C",
            **kwargs,
        )


def test_builder_rejects_empty_instruction() -> None:
    with pytest.raises(ValueError):
        build_sparse_history_messages(
            instruction="   ",
            action_texts=ACTION_TEXTS,
            selected_steps=[3],
            selected_images=["a"],
            current_step=18,
            current_image="C",
        )


# ---------------------------------------------------------------------------
# 3. 五臂 + 三负样本的集成构造(真实调用生产侧,一个 pair-group 全臂)
# ---------------------------------------------------------------------------
def _sparse_selection(budget: int) -> list[int]:
    """Strictly increasing, pairwise non-adjacent steps older than the decision."""
    return [2, 5, 8, 10][:budget]


def build_group(
    *, budget: int, split: str = "train", **overrides: Any
) -> dict[str, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "episode": EPISODE,
        "decision_step": DECISION_STEP,
        "instruction": "book a flight",
        "action_texts": ACTION_TEXTS[: DECISION_STEP - 1],
        "selected_steps": _sparse_selection(budget),
        "target_text": TARGET_TEXT,
        "split": split,
        "donor_episode": DONOR_EPISODE,
        "donor_images": DONOR_IMAGES,
    }
    kwargs.update(overrides)
    samples = build_pair_group_samples(**kwargs)
    group = {sample["arm_slot"]: sample for sample in samples}
    assert len(group) == len(samples), "同一 pair-group 内 arm_slot 必须唯一"
    return group


def assert_sample_contract(
    sample: dict[str, Any], *, slot: str, budget: int
) -> None:
    arm_id, role, prompt_format, selection_mode, adapter_mode = ARM_SPEC[slot]
    assert sample["schema_version"] == SCHEMA_VERSION
    assert sample["arm_id"] == arm_id
    assert sample["role"] == role
    assert sample["prompt_format"] == prompt_format
    assert sample["selection_mode"] == selection_mode
    assert sample["adapter_mode"] == adapter_mode
    assert sample["sample_id"] == f"{sample['pair_group']}|{slot}"
    assert sample["pair_group"] == f"{sample['episode']}:{sample['decision_step']}"
    assert sample["decision_step"] == DECISION_STEP
    assert sample["reference_arm_id"] == "R0"
    assert sample["deployment_baseline_arm_id"] == "N0"
    assert sample["split"] in ("train", "heldout")
    assert sample["target_text"] == TARGET_TEXT

    steps = sample["selected_steps"]
    assert sample["budget"] == budget == len(steps) == len(sample["selected_images"])
    assert all(b > a for a, b in zip(steps, steps[1:])), "selected_steps 必须严格递增"
    assert all(1 <= s < sample["decision_step"] for s in steps)
    assert isinstance(sample["current_image"], str)

    parts = image_parts(sample["messages"])
    assert len(parts) == budget + 1, "图像 part 数必须是 K 张历史 + 1 张当前"
    for part in parts:
        assert "path" in part and isinstance(part["path"], str)
        assert "image" not in part, "落盘样本的图像 part 只带 path,不带内存对象"

    for key in FORBIDDEN_KEYS:
        assert key not in sample, f"废弃字段 {key} 不得再写入(FIELD CONTRACT v2)"

    if role == "negative":
        assert sample["negative_kind"] == NEGATIVE_KIND[slot]
        assert float(sample["negative_scale"]) > 0.0
        if sample["negative_kind"] == "irrelevant":
            assert sample["donor_episode"] != sample["episode"]


def test_pair_group_covers_the_five_arms_and_three_negatives() -> None:
    group = build_group(budget=3)
    assert set(group) == set(ARM_SPEC)
    assert {s["arm_id"] for s in group.values()} == {"N0", "R0", "S0", "RA", "SA"}
    for slot, sample in group.items():
        assert_sample_contract(sample, slot=slot, budget=3)


def test_pair_group_shares_one_target_and_one_group_key() -> None:
    group = build_group(budget=3)
    assert len({s["target_text"] for s in group.values()}) == 1
    assert len({s["pair_group"] for s in group.values()}) == 1
    assert len({s["current_image"] for s in group.values()}) == 1
    assert len({s["sample_id"] for s in group.values()}) == len(group)


def test_recent_arms_share_the_contiguous_window() -> None:
    budget = 3
    group = build_group(budget=budget)
    recent = list(range(DECISION_STEP - budget, DECISION_STEP))
    for slot in RECENT_SLOTS:
        assert group[slot]["selected_steps"] == recent
        assert group[slot]["selection_mode"] == "recent"


def test_sparse_arms_share_the_non_contiguous_selection() -> None:
    budget = 3
    group = build_group(budget=budget)
    selection = _sparse_selection(budget)
    for slot in SPARSE_SLOTS:
        assert group[slot]["selected_steps"] == selection
        assert group[slot]["selection_mode"] == "sparse"
    assert any(
        b - a > 1 for a, b in zip(selection, selection[1:])
    ), "sparse 选点必须真的非连续"


def test_adapter_mode_split_matches_the_five_arm_contract() -> None:
    group = build_group(budget=3)
    bypass = {s for s, v in group.items() if v["adapter_mode"] == "bypass"}
    active = {s for s, v in group.items() if v["adapter_mode"] == "active"}
    assert bypass == {"N0", "R0", "S0"}
    assert active == {"RA", "SA", *NEGATIVE_KIND}


def test_the_main_claim_pair_differs_only_in_adapter_mode() -> None:
    """主 claim 是 SA-RA:两臂的 prompt 必须逐字不同于选点、逐字同于格式与预算。

    # note (luojiaxuan): RA 与 R0、SA 与 S0 分别是"同 messages、只差 adapter_mode"
    # 的两对。这一条把它钉死 —— 一旦哪个臂的渲染多带了什么(例如给 active 臂偷偷
    # 换了 system prompt),SA-RA 就不再是纯粹的 adapter 效应。
    """
    group = build_group(budget=3)
    assert group["RA"]["messages"] == group["R0"]["messages"]
    assert group["SA"]["messages"] == group["S0"]["messages"]
    assert group["RA"]["adapter_mode"] != group["R0"]["adapter_mode"]
    assert group["SA"]["adapter_mode"] != group["S0"]["adapter_mode"]
    assert group["SA"]["messages"] != group["RA"]["messages"], (
        "SA 与 RA 只差选点(sparse vs recent),prompt 必须真的不同"
    )


def test_only_the_deployment_arm_uses_the_official_multiturn_format() -> None:
    group = build_group(budget=3)
    official = {
        s for s, v in group.items() if v["prompt_format"] == "official_multiturn"
    }
    assert official == {"N0"}


def test_official_arm_keeps_full_assistant_responses() -> None:
    """P0-1:官方臂的保留轮必须带 Action + <tool_call>,不得退化成裸描述。"""
    group = build_group(budget=3)
    assistant = [
        part["text"]
        for message in group["N0"]["messages"]
        if message["role"] == "assistant"
        for part in message["content"]
        if part.get("type") == "text"
    ]
    assert assistant, "官方多轮臂必须有 assistant 轮"
    for text in assistant:
        assert text.startswith("Action: ")
        assert "<tool_call>" in text and "</tool_call>" in text


def test_official_arm_rejects_unreconstructable_history_responses() -> None:
    """保留轮的完整响应重建不出来时整组丢弃,而不是静默降级。"""
    responses: list[str | None] = [
        producer.official_response(text, {"action": "system_button", "button": "Home"})
        for text in ACTION_TEXTS[: DECISION_STEP - 1]
    ]
    responses[-1] = None
    with pytest.raises(GroupRejected):
        build_group(budget=3, full_responses=responses)


def test_negative_images_differ_from_the_positive_arm() -> None:
    group = build_group(budget=3)
    positive = group["SA"]["selected_images"]
    for slot in NEGATIVE_KIND:
        assert group[slot]["selected_images"] != positive
    irrelevant = group["SA_neg_irrelevant"]["selected_images"]
    assert all(f"/{EPISODE}/" not in path for path in irrelevant), (
        "irrelevant 负样本不得混入本轨迹的图"
    )
    assert all(f"/{DONOR_EPISODE}/" in path for path in irrelevant)


def test_negative_scales_follow_the_frozen_contract() -> None:
    group = build_group(budget=3)
    assert float(group["SA_neg_step_shuffled"]["negative_scale"]) == 1.0
    assert float(group["SA_neg_irrelevant"]["negative_scale"]) == 1.0
    assert float(group["SA_neg_duplicate"]["negative_scale"]) == 0.5


def test_group_negative_slots_is_a_self_check_not_a_control_channel() -> None:
    """组内自检字段必须与"逐行 role == negative"的判定完全一致。"""
    group = build_group(budget=3)
    declared = {slot for sample in group.values() for slot in sample["group_negative_slots"]}
    by_role = {slot for slot, sample in group.items() if sample["role"] == "negative"}
    assert declared == by_role == set(NEGATIVE_KIND)
    assert len({tuple(s["group_negative_slots"]) for s in group.values()}) == 1


def test_split_is_written_verbatim_onto_every_arm() -> None:
    """split 是样本的只读字段,构造期写一次,消费侧不再重算 hash。"""
    for split in ("train", "heldout"):
        group = build_group(budget=3, split=split)
        assert {s["split"] for s in group.values()} == {split}


def test_irrelevant_donor_shares_the_split_of_its_group() -> None:
    group = build_group(budget=3, split="heldout")
    irrelevant = group["SA_neg_irrelevant"]
    assert irrelevant["donor_episode"] == DONOR_EPISODE
    # donor_split 只是构造期的内部字段,不落到样本上(避免与样本自身 split 混淆)
    assert "donor_split" not in irrelevant
    assert irrelevant["split"] == "heldout"


# ---------------------------------------------------------------------------
# 4. K=1/2/3/4 的臂库存与负样本几何
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("budget", (1, 2, 3, 4))
def test_arm_inventory_by_budget(budget: int) -> None:
    group = build_group(budget=budget)
    if budget == 1:
        # note (luojiaxuan): K=1 时"倒序错配"与"同图重复"都退化成 SA 自身,
        # 留着它们等于给 rank 项喂两个逐字相同的条件,梯度恒为 hinge 常数噪声。
        assert "SA_neg_step_shuffled" not in group
        assert "SA_neg_duplicate" not in group
        assert set(group) == {"N0", "R0", "S0", "RA", "SA", "SA_neg_irrelevant"}
    else:
        assert set(group) == set(ARM_SPEC)
    for slot, sample in group.items():
        assert_sample_contract(sample, slot=slot, budget=budget)


@pytest.mark.parametrize("budget", (2, 3, 4))
def test_step_shuffled_is_a_derangement(budget: int) -> None:
    group = build_group(budget=budget)
    positive = group["SA"]["selected_images"]
    shuffled = group["SA_neg_step_shuffled"]["selected_images"]
    assert sorted(shuffled) == sorted(positive), "shuffled 只错配位置,不换图源"
    assert all(a != b for a, b in zip(positive, shuffled)), (
        "shuffled 必须是 derangement:每个位置的图都必须与 SA 不同,"
        "否则奇数 K 的中间位置(如倒序 K=3)仍是对的,负样本被稀释"
    )
    assert group["SA_neg_step_shuffled"]["selected_steps"] == group["SA"]["selected_steps"]


@pytest.mark.parametrize("budget", (2, 3, 4))
def test_duplicate_repeats_a_single_history_image(budget: int) -> None:
    group = build_group(budget=budget)
    duplicate = group["SA_neg_duplicate"]["selected_images"]
    assert len(set(duplicate)) == 1
    assert duplicate[0] in group["SA"]["selected_images"]
    assert len(duplicate) == budget


@pytest.mark.parametrize("budget", (1, 2, 3, 4))
def test_budget_matched_reference_window(budget: int) -> None:
    group = build_group(budget=budget)
    # 预算对齐:R0 的连续窗口长度必须等于 SA 的选点数,否则 ℓc-ℓr 混入"图更多"效应
    assert group["R0"]["budget"] == group["SA"]["budget"] == budget
    assert len(image_parts(group["R0"]["messages"])) == budget + 1
    assert len(image_parts(group["N0"]["messages"])) == budget + 1


@pytest.mark.parametrize("budget", (1, 2, 3, 4))
def test_every_arm_carries_exactly_budget_plus_one_images(budget: int) -> None:
    group = build_group(budget=budget)
    for slot, sample in group.items():
        assert producer.count_images(sample["messages"]) == budget + 1, slot


# ---------------------------------------------------------------------------
# 5. 构造期的 fail-closed
# ---------------------------------------------------------------------------
def test_sparse_equal_to_recent_is_rejected() -> None:
    """S0 与 R0 会变成同一条 prompt,这一组量不出"冻结选图效应",整组丢弃。"""
    recent = list(range(DECISION_STEP - 3, DECISION_STEP))
    with pytest.raises(GroupRejected):
        build_group(budget=3, selected_steps=recent)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    (
        (dict(selected_steps=[]), "K 至少为 1"),
        (dict(selected_steps=[8, 2, 5]), "选点必须严格递增"),
        (dict(selected_steps=[2, 2]), "选点重复"),
        (dict(selected_steps=[0, 5]), "选点必须 1-based"),
        (dict(selected_steps=[2, DECISION_STEP]), "选点必须早于决策步"),
        (dict(action_texts=ACTION_TEXTS[:3]), "动作文本不足以覆盖决策步之前"),
    ),
)
def test_pair_group_builder_is_fail_closed(
    overrides: dict[str, Any], reason: str
) -> None:
    with pytest.raises(ValueError):
        build_group(budget=2, **overrides)


def test_the_shallowest_legal_decision_still_builds_a_complete_group() -> None:
    """K 只能取到"预算对齐的 recent 窗口装得下"为止;边界上仍须造出完整一组。

    # note (luojiaxuan): decision_step=6、K=2 时 recent 窗口是 [4,5],sparse 取
    # [1,3] —— 恰好是 recent 窗口还不越界、而 sparse 又不与它相等的最浅决策点。
    # 边界组能不能造出来,是"K 分布尾部会不会整段消失"的直接证据。
    """
    group = build_pair_group_samples(
        episode=EPISODE,
        decision_step=6,
        instruction="x",
        action_texts=ACTION_TEXTS[:5],
        selected_steps=[1, 3],
        target_text=TARGET_TEXT,
        donor_episode=DONOR_EPISODE,
        donor_images=DONOR_IMAGES,
    )
    by_slot = {sample["arm_slot"]: sample for sample in group}
    assert set(by_slot) == set(ARM_SPEC)
    assert by_slot["R0"]["selected_steps"] == [4, 5]
    assert by_slot["SA"]["selected_steps"] == [1, 3]
    for slot, sample in by_slot.items():
        assert producer.count_images(sample["messages"]) == 3, slot


def test_donor_episode_must_differ_from_the_group_episode() -> None:
    with pytest.raises(ValueError):
        build_group(budget=2, donor_episode=EPISODE, donor_images=None)


def test_donor_images_must_follow_the_frozen_naming_convention() -> None:
    with pytest.raises(ValueError):
        build_group(budget=2, donor_images=["images/EP-B/frame_0.jpg"])


def test_donor_pool_smaller_than_the_budget_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_group(budget=4, donor_images=DONOR_IMAGES[:2])


def test_donor_record_reports_the_ages_it_actually_matched() -> None:
    """退化成池尾 K 张时必须如实记录实际 age 向量,不得假装匹配上了。"""
    group = build_group(budget=3)
    irrelevant = group["SA_neg_irrelevant"]
    anchor = irrelevant["donor_decision_step"]
    assert irrelevant["donor_matched_ages"] == [
        anchor - step for step in irrelevant["donor_steps"]
    ]
    assert len(irrelevant["donor_steps"]) == 3
    assert irrelevant["donor_size_matches"] == 0, "内存构造拿不到分辨率,如实记 0"


# ---------------------------------------------------------------------------
# 6. 稀疏选点采样器本身的不变量(choose_sparse / max_sparse_budget)
# ---------------------------------------------------------------------------
def test_max_sparse_budget_matches_the_non_adjacent_capacity() -> None:
    # pool = cur-1;两两不相邻子集的最大规模是 ceil(P/2);P<5 时没有 age>4 的老事件
    assert producer.max_sparse_budget(5) == 0
    assert producer.max_sparse_budget(6) == 3
    assert producer.max_sparse_budget(9) == 4
    assert all(producer.max_sparse_budget(cur) == 0 for cur in range(1, 6))


@pytest.mark.parametrize("current_step", (9, 12, 17, 25))
def test_choose_sparse_never_degrades_into_a_recent_window(current_step: int) -> None:
    """审计的核心退化点:池子小时旧实现会关掉相邻性检查,sparse 变成连续窗口。

    # note (luojiaxuan): 这里对每个 (cur, K) 采样若干次并逐次断言,而不是统计"含
    # 相邻对的比例低于某阈值" —— 阈值型断言在实现退化时会先变成 flaky 再变成失败,
    # 而不变量型断言当场就失败。
    """
    rng = random.Random(20260724)
    for k in range(1, producer.max_sparse_budget(current_step) + 1):
        for _ in range(200):
            picked = producer.choose_sparse(rng, current_step, k)
            if not picked:
                continue
            assert len(picked) == k
            assert picked == sorted(set(picked))
            assert picked[0] >= 1 and picked[-1] < current_step
            assert all(b - a >= 2 for a, b in zip(picked, picked[1:])), (
                f"cur={current_step} K={k} 采到了相邻步 {picked}"
            )
            assert picked[0] <= current_step - 5, (
                f"cur={current_step} K={k} 选点 {picked} 不含 age>4 的老事件"
            )
            assert picked != list(range(current_step - k, current_step))


def test_choose_sparse_returns_empty_beyond_the_capacity() -> None:
    rng = random.Random(1)
    for current_step in (6, 9, 12):
        over = producer.max_sparse_budget(current_step) + 1
        assert producer.choose_sparse(rng, current_step, over) == []
        assert producer.choose_sparse(rng, current_step, 0) == []


def test_episode_image_path_is_the_single_naming_authority() -> None:
    """donor 路径与本轨迹路径必须由同一个模板生成,否则 irrelevant 臂指向不存在的文件。"""
    assert producer.episode_image_path("EP-A", 1) == "images/EP-A/obs-000.png"
    assert producer.episode_image_path("EP-A", 12) == "images/EP-A/obs-011.png"
    group = build_group(budget=3)
    assert group["SA"]["selected_images"] == [
        producer.episode_image_path(EPISODE, step)
        for step in group["SA"]["selected_steps"]
    ]
    assert group["SA"]["current_image"] == producer.episode_image_path(
        EPISODE, DECISION_STEP
    )
