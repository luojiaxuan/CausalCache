#!/usr/bin/env python3
"""六臂/五臂替换语料(V6)构建器的契约测试(pytest,无需 GPU)。

# note (luojiaxuan): 这套用例守的是 build_replacement_corpus_v6.py 里**一旦静默失守就
# 会让主张失去意义**的几条约束:
#   1. **配对渲染**:R0/RA、S0/SA 每一对的 messages 必须逐字节相同,只有 adapter_mode
#      不同。主张 A_c - A_r 是差中差,prompt 一旦有差异,差里就混进了 prompt 效应——
#      而这种差异不会报错,只会让数字变好看。
#   2. **matched wrong 的定义**:负样本集与 S 集必须替换掉**同一个** recent 位置、恢复
#      **不同**的旧帧。少了前半条,两者差的就不止"选了哪张旧帧";少了后半条,负样本
#      干脆就是正样本。
#   3. **不伪造正例**:recent_sufficient(k=0)组只出 N0/R0/RA,绝不给它编 oracle 集。
#   4. **没有合格 distractor 时退成五臂**(N0/R0/RA/S0/SA),而不是整组丢弃,也不是塞
#      一个随机帧。整组丢弃会系统性截掉高 gain 的正例(实测被丢组 gain 中位数 0.053 vs
#      保留组 0.034);塞随机帧或改用相对门槛则会让负样本本身就该有正增益,
#      ``|A_n| < 0.02`` 这条 gate 随之失去可证伪性。判别只准读 ``has_content_control``,
#      不准数臂数。真正的硬失败(缺分数/缺图/对账不符)仍然整组拒绝并计数。
#   5. **K=0 退化**:渲染路径与官方基线逐字节一致(与 test_gui_owl_sparse_multiturn.py
#      的同名用例同一条判据),确保我们没有在渲染层引入自己的措辞。
#   6. **N0 不是冗余臂**:它必须与原生 build_official_messages 逐字节相同,且与 R0
#      **不同**。后一条(反向断言)才是真正抓错的那条 —— 两者相同意味着 renderer 冻结
#      没生效,而 format_effect = R0 - N0 会安静地恒等于 0。
#   7. **trainer 真的认这些槽位**:负样本的 arm_slot 必须过 trainer 的 validate_* 与
#      损失分发器的 negatives 收集。只验证 encode_sample 能跑是不够的 —— v5 的
#      ``if slot != f"SA_neg_{kind}"`` 契约会让不合规的名字**静默**变成"这组没有负样本"。
"""

from __future__ import annotations

import importlib
import json
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
    parent_name, _, leaf = name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, leaf, module)


stub_if_missing("torch")
stub_if_missing("PIL")
stub_if_missing("PIL.Image", open=lambda *args, **kwargs: None)

from causalcache.policy.gui_owl_official import build_official_messages  # noqa: E402
from causalcache.policy.gui_owl_sparse_multiturn import (  # noqa: E402
    SPARSE_MULTITURN_PROMPT_FORMAT,
)
import scripts.train_success_sft_lora as trainer  # noqa: E402
from scripts.build_sparse_history_dataset import (  # noqa: E402
    official_response,
    target_arguments,
)
from scripts.build_replacement_corpus_v6 import (  # noqa: E402
    ADAPTER_PAIRS,
    ARM_SPECS,
    CLASS_RECENT_SUFFICIENT,
    CLASS_UTILITY_POSITIVE,
    BASE_ARM_SLOTS,
    CONTROL_ABSENT_NO_QUALIFYING,
    CONTROL_ABSENT_RECENT_SUFFICIENT,
    DEPLOYMENT_FORMAT,
    NEGATIVE_ARM_SLOT,
    RECENT_ONLY_ARM_SLOTS,
    RESERVED_ARM_SLOTS,
    SCHEMA_VERSION,
    GroupRejected,
    build_corpus,
    build_replacement_group_samples,
    messages_bytes,
    render_messages,
)
from scripts.train_success_sft_lora import (  # noqa: E402
    _prompt_encoding_path,
    effective_adapter_mode,
)


# ---------------------------------------------------------------------------
# 固定夹具:一个 12 步决策点、预算 2
# ---------------------------------------------------------------------------
EPISODE = "0000000000000042"
CURRENT_STEP = 12
BUDGET = 2
DELTA_RECENT = 0.01
INSTRUCTION = "Open the settings app and switch the display theme to dark"
# 1 基:ACTION_TEXTS[i - 1] 是 Step i 的动作描述,最后一条是目标步本身。
ACTION_TEXTS = [f"tap the list item on row {index}" for index in range(1, CURRENT_STEP + 1)]
PREFIX_ACTIONS = ACTION_TEXTS[: CURRENT_STEP - 1]
TARGET_TEXT = official_response(
    ACTION_TEXTS[CURRENT_STEP - 1], {"action": "click", "coordinate": [420, 880]}
)
RECENT = [CURRENT_STEP - BUDGET, CURRENT_STEP - 1]  # [10, 11]
ORACLE_OLD = 4
DROPPED = 10
ORACLE_STEPS = sorted([step for step in RECENT if step != DROPPED] + [ORACLE_OLD])
# 旧帧池 1..9。3 与 5 是 age 上最近的两张,但它们**有用**,所以必须被筛掉;
# 剩下 2 与 6 并列最近,按"并列取更老"的规则应当选中 2。
UTILITY: dict[int, float] = {
    1: -0.02,
    2: 0.0,
    3: 0.50,
    5: 0.50,
    6: 0.001,
    7: -0.01,
    8: 0.0,
    9: 0.03,
}
EXPECTED_WRONG_OLD = 2
POSITIVE_DECISION = {
    "k": 1,
    "klass": CLASS_UTILITY_POSITIVE,
    "steps": ORACLE_STEPS,
    "gain": 0.042,
    "pool_size": 18,
}
RECENT_SUFFICIENT_DECISION = {
    "k": 0,
    "klass": CLASS_RECENT_SUFFICIENT,
    "steps": RECENT,
    "gain": 0.001,
    "reason": "gain below delta_recent",
}


def build_positive(utility: dict[int, float | None] | None = None) -> list[dict]:
    return build_replacement_group_samples(
        episode=EPISODE,
        decision_step=CURRENT_STEP,
        instruction=INSTRUCTION,
        action_texts=PREFIX_ACTIONS,
        target_text=TARGET_TEXT,
        budget=BUDGET,
        decision=POSITIVE_DECISION,
        utility=dict(UTILITY) if utility is None else utility,
        delta_recent=DELTA_RECENT,
    )


def by_slot(rows: list[dict]) -> dict[str, dict]:
    return {row["arm_slot"]: row for row in rows}


def _synthetic_full_responses() -> list[str]:
    """The history responses the in-memory entry point synthesizes for tests.

    # note (luojiaxuan): 与 build_replacement_group_samples 的合成规则必须一致,否则
    # N0 的对照渲染会拿到另一套 assistant 轮,断言就变成"两个错的东西相等"。
    """
    return [
        official_response(text, {"action": "system_button", "button": "Home"})
        for text in PREFIX_ACTIONS
    ]


# ---------------------------------------------------------------------------
# 1. 臂契约与命名
# ---------------------------------------------------------------------------
def test_the_arm_table_is_the_frozen_one() -> None:
    assert ARM_SPECS == (
        ("N0", "N0", "deployment_baseline", "official_multiturn", "recent", "bypass"),
        ("R0", "R0", "reference", SPARSE_MULTITURN_PROMPT_FORMAT, "recent", "bypass"),
        ("RA", "RA", "measurement", SPARSE_MULTITURN_PROMPT_FORMAT, "recent", "active"),
        (
            "S0", "S0", "measurement", SPARSE_MULTITURN_PROMPT_FORMAT,
            "replacement", "bypass",
        ),
        (
            "SA", "SA", "positive", SPARSE_MULTITURN_PROMPT_FORMAT,
            "replacement", "active",
        ),
        (
            "SA_neg_age_matched", "SA", "negative", SPARSE_MULTITURN_PROMPT_FORMAT,
            "wrong", "active",
        ),
    )
    assert RECENT_ONLY_ARM_SLOTS == ("N0", "R0", "RA")
    assert BASE_ARM_SLOTS == ("N0", "R0", "RA", "S0", "SA")
    # 每一列都必须与 trainer 的 v6 契约表逐字段相同,否则 validate_sparse_sample 会
    # 在训练启动时才炸(或者更糟:某个槽位不被认成负样本而静默失效)。
    for slot, arm_id, role, prompt_format, mode, adapter_mode in ARM_SPECS:
        assert trainer.SPARSE_REPLACEMENT_ARM_CONTRACT[slot] == (
            arm_id, role, prompt_format, mode, adapter_mode,
        ), slot


def test_the_unused_active_baseline_name_stays_reserved() -> None:
    """NA = 官方 renderer + adapter active,没有任何主张需要它。"""
    assert set(RESERVED_ARM_SLOTS) == {"NA"}
    slots = {spec[0] for spec in ARM_SPECS}
    assert slots & set(RESERVED_ARM_SLOTS) == set()
    assert "N0" not in RESERVED_ARM_SLOTS, "N0 是部署基线臂,本语料照常产出"
    assert {row["arm_slot"] for row in build_positive()} & set(RESERVED_ARM_SLOTS) == set()


def test_a_positive_group_emits_all_six_arms_once_each() -> None:
    rows = build_positive()
    assert [row["arm_slot"] for row in rows] == [spec[0] for spec in ARM_SPECS]
    assert len({row["sample_id"] for row in rows}) == 6
    for row in rows:
        assert row["sample_id"] == f"{row['pair_group']}|{row['arm_slot']}"
        assert row["reference_arm_id"] == "R0"
        assert row["deployment_baseline_arm_id"] == "N0"
        assert row["label_class"] == CLASS_UTILITY_POSITIVE
        assert row["label_k"] == 1
        assert row["has_content_control"] is True
        assert row["content_control_absent_reason"] is None
        assert row["group_arm_slots"] == [spec[0] for spec in ARM_SPECS]
    formats = {row["arm_slot"]: row["prompt_format"] for row in rows}
    assert formats.pop("N0") == DEPLOYMENT_FORMAT
    assert set(formats.values()) == {SPARSE_MULTITURN_PROMPT_FORMAT}


def test_pair_group_key_separates_the_two_budgets_of_one_decision() -> None:
    """v5 的 <episode>:<step> 在 v6 会让 B=1 与 B=2 两组撞在一起。"""
    rows = build_positive()
    assert rows[0]["pair_group"] == f"{EPISODE}:{CURRENT_STEP}:b{BUDGET}"
    assert rows[0]["source_pair_group"] == f"{EPISODE}:{CURRENT_STEP}"


# ---------------------------------------------------------------------------
# 2. (a) 两对臂的 messages 逐字节相同
# ---------------------------------------------------------------------------
def test_each_adapter_pair_renders_byte_for_byte_identically() -> None:
    rows = by_slot(build_positive())
    for bypass_slot, active_slot in ADAPTER_PAIRS:
        bypass, active = rows[bypass_slot], rows[active_slot]
        assert messages_bytes(bypass["messages"]) == messages_bytes(active["messages"]), (
            f"{bypass_slot}/{active_slot} 的 prompt 不同,差中差里会混进 prompt 效应"
        )
        # 落盘形态也必须同字节:samples.jsonl 里就是这么写出去的。
        assert json.dumps(bypass["messages"], ensure_ascii=False) == json.dumps(
            active["messages"], ensure_ascii=False
        )
        assert bypass["messages_sha256"] == active["messages_sha256"]
        assert (bypass["adapter_mode"], active["adapter_mode"]) == ("bypass", "active")
        assert bypass["selected_steps"] == active["selected_steps"]
        assert bypass["selected_images"] == active["selected_images"]


def test_the_distinct_history_arms_differ_from_one_another() -> None:
    """配对相同是要求,跨历史内容相同则说明这一组什么都量不出来。"""
    rows = by_slot(build_positive())
    digests = {
        slot: rows[slot]["messages_sha256"]
        for slot in ("N0", "R0", "S0", NEGATIVE_ARM_SLOT)
    }
    assert len(set(digests.values())) == 4, digests


def test_the_negative_arm_has_no_bypass_twin() -> None:
    """A_n 由**同一行**跑 active + bypass 两次前向得到,孪生臂纯属浪费前向。"""
    assert ADAPTER_PAIRS == (("R0", "RA"), ("S0", "SA"))
    rows = by_slot(build_positive())
    assert rows[NEGATIVE_ARM_SLOT]["adapter_mode"] == "active"
    assert not [slot for slot in rows if slot.startswith("SA_neg_") and slot != NEGATIVE_ARM_SLOT]


# ---------------------------------------------------------------------------
# 2b. N0 部署基线:与原生官方渲染逐字节相同,且与 R0 **不同**
# ---------------------------------------------------------------------------
def test_deployment_baseline_matches_the_native_official_renderer() -> None:
    rows = by_slot(build_positive())
    n0 = rows["N0"]
    official = _serialise(
        build_official_messages(
            goal=INSTRUCTION,
            past_action_texts=PREFIX_ACTIONS,
            recent_images=[
                {"__path__": path} for path in rows["R0"]["selected_images"]
            ],
            current_image={"__path__": rows["R0"]["current_image"]},
            past_full_responses=_synthetic_full_responses(),
        )
    )
    assert n0["messages"] == official
    assert messages_bytes(n0["messages"]) == messages_bytes(official)
    assert n0["selected_steps"] == RECENT
    assert n0["adapter_mode"] == "bypass"
    assert n0["role"] == "deployment_baseline"


def test_deployment_baseline_is_not_the_reference_arm() -> None:
    """反向断言:N0 == R0 意味着 renderer 冻结没生效,format_effect 会恒等于 0。"""
    rows = by_slot(build_positive())
    assert rows["N0"]["selected_steps"] == rows["R0"]["selected_steps"]
    assert rows["N0"]["selected_images"] == rows["R0"]["selected_images"]
    assert messages_bytes(rows["N0"]["messages"]) != messages_bytes(
        rows["R0"]["messages"]
    ), "N0 与 R0 是同一批 recent 帧的两种渲染,逐字节相同说明冻结 renderer 没生效"
    assert rows["N0"]["messages_sha256"] != rows["R0"]["messages_sha256"]


# ---------------------------------------------------------------------------
# 3. (b) matched wrong 集:同一个被替换位置,不同的旧帧
# ---------------------------------------------------------------------------
def test_wrong_set_replaces_the_same_recent_slot_with_a_different_old_frame() -> None:
    rows = by_slot(build_positive())
    oracle = rows["S0"]["selected_steps"]
    wrong = rows[NEGATIVE_ARM_SLOT]["selected_steps"]
    assert oracle == ORACLE_STEPS
    assert set(RECENT) - set(oracle) == set(RECENT) - set(wrong) == {DROPPED}
    restored_oracle = set(oracle) - set(RECENT)
    restored_wrong = set(wrong) - set(RECENT)
    assert restored_oracle == {ORACLE_OLD}
    assert restored_wrong == {EXPECTED_WRONG_OLD}
    assert restored_oracle != restored_wrong
    assert len(oracle) == len(wrong) == BUDGET


def test_the_wrong_frame_is_the_age_nearest_frame_that_is_still_useless() -> None:
    """先按效用筛,再按 age 取最近:相邻帧往往是同样有用的替身。"""
    rows = by_slot(build_positive())
    diagnostics = rows[NEGATIVE_ARM_SLOT]["replacement"]
    assert diagnostics["dropped_recent_step"] == DROPPED
    assert diagnostics["oracle_old_step"] == ORACLE_OLD
    assert diagnostics["wrong_old_step"] == EXPECTED_WRONG_OLD
    assert diagnostics["wrong_utility"] < DELTA_RECENT
    # 3 与 5 在 age 上更近(gap=1),但效用 0.5 >= delta,必须被排除。
    assert diagnostics["wrong_age_gap"] == abs(EXPECTED_WRONG_OLD - ORACLE_OLD) == 2
    # 候选 8 张(旧帧池 1..9 去掉 oracle 帧 4),其中 3/5/9 有用,合格的是 1/2/6/7/8。
    assert diagnostics["distractor_candidates"] == 8
    assert diagnostics["distractor_qualifying"] == 5
    assert diagnostics["distractor_qualifying_fraction"] == 5 / 8


def test_ties_in_age_are_broken_towards_the_older_frame() -> None:
    """2 与 6 距 oracle(4)同为 2,规则取更老的 2,结果必须可复现。"""
    rows = by_slot(build_positive())
    assert rows[NEGATIVE_ARM_SLOT]["replacement"]["wrong_old_step"] == 2
    assert UTILITY[2] < DELTA_RECENT and UTILITY[6] < DELTA_RECENT


def test_the_wrong_frame_comes_from_the_same_episode_and_decision_point() -> None:
    """同决策点取帧 = app / 设备 / 分辨率 / 视觉 token 数自动匹配。"""
    rows = by_slot(build_positive())
    for slot in ("S0", NEGATIVE_ARM_SLOT):
        assert rows[slot]["episode"] == EPISODE
        assert rows[slot]["decision_step"] == CURRENT_STEP
        for path in rows[slot]["selected_images"]:
            assert path.startswith(f"images/{EPISODE}/")


# ---------------------------------------------------------------------------
# 4. (c) recent_sufficient 组只出两臂
# ---------------------------------------------------------------------------
def test_recent_sufficient_groups_emit_only_the_baseline_and_recent_arms() -> None:
    rows = build_replacement_group_samples(
        episode=EPISODE,
        decision_step=CURRENT_STEP,
        instruction=INSTRUCTION,
        action_texts=PREFIX_ACTIONS,
        target_text=TARGET_TEXT,
        budget=BUDGET,
        decision=RECENT_SUFFICIENT_DECISION,
        utility={},
        delta_recent=DELTA_RECENT,
    )
    assert [row["arm_slot"] for row in rows] == ["N0", "R0", "RA"]
    assert all(row["selected_steps"] == RECENT for row in rows)
    assert all(row["label_class"] == CLASS_RECENT_SUFFICIENT for row in rows)
    assert all(row["label_k"] == 0 for row in rows)
    assert all(row["group_arm_slots"] == ["N0", "R0", "RA"] for row in rows)
    # 基线臂对所有组都有意义:k=0 组同样要能报 deployment_delta。
    assert by_slot(rows)["N0"]["role"] == "deployment_baseline"
    assert all(row["has_content_control"] is False for row in rows)
    assert all(
        row["content_control_absent_reason"] == CONTROL_ABSENT_RECENT_SUFFICIENT
        for row in rows
    )
    # 绝不给 k=0 组伪造 oracle / wrong 集。
    assert {row["selection_mode"] for row in rows} == {"recent"}
    arms = by_slot(rows)
    # 三臂同一批 recent 帧,但 R0/RA 是同一份 prompt,N0 是另一个 renderer。
    assert messages_bytes(arms["R0"]["messages"]) == messages_bytes(
        arms["RA"]["messages"]
    )
    assert messages_bytes(arms["N0"]["messages"]) != messages_bytes(
        arms["R0"]["messages"]
    )


def test_recent_sufficient_needs_no_utility_table_at_all() -> None:
    rows = build_replacement_group_samples(
        episode=EPISODE,
        decision_step=CURRENT_STEP,
        instruction=INSTRUCTION,
        action_texts=PREFIX_ACTIONS,
        target_text=TARGET_TEXT,
        budget=BUDGET,
        decision=RECENT_SUFFICIENT_DECISION,
        utility=None,
        delta_recent=DELTA_RECENT,
    )
    assert len(rows) == 3


def test_a_recent_sufficient_label_whose_steps_are_not_the_recent_window_is_rejected() -> None:
    with pytest.raises(GroupRejected) as error:
        build_replacement_group_samples(
            episode=EPISODE,
            decision_step=CURRENT_STEP,
            instruction=INSTRUCTION,
            action_texts=PREFIX_ACTIONS,
            target_text=TARGET_TEXT,
            budget=BUDGET,
            decision={**RECENT_SUFFICIENT_DECISION, "steps": [3, 11]},
            utility={},
            delta_recent=DELTA_RECENT,
        )
    assert error.value.reason == "recent_steps_disagreement"


# ---------------------------------------------------------------------------
# 5. (d) 找不到合格 distractor -> 退成五臂(不是整组丢弃,也不计入 reject)
# ---------------------------------------------------------------------------
def test_a_group_without_any_useless_old_frame_keeps_five_arms() -> None:
    """整组丢弃会系统性截掉高 gain 的正例;这里只丢内容对照臂。"""
    useful_everywhere = {step: 0.5 for step in UTILITY}
    rows = build_positive(utility=useful_everywhere)
    assert [row["arm_slot"] for row in rows] == list(BASE_ARM_SLOTS)
    for row in rows:
        assert row["has_content_control"] is False
        assert row["content_control_absent_reason"] == CONTROL_ABSENT_NO_QUALIFYING
        assert row["label_class"] == CLASS_UTILITY_POSITIVE
        assert row["label_k"] == 1
        assert row["group_arm_slots"] == list(BASE_ARM_SLOTS)
        assert not row["arm_slot"].startswith("SA_neg_")
    arms = by_slot(rows)
    # 五臂组仍然是一个完整可用的 DiD 单元:两对臂照样逐字节相同。
    for bypass_slot, active_slot in (("R0", "RA"), ("S0", "SA")):
        assert messages_bytes(arms[bypass_slot]["messages"]) == messages_bytes(
            arms[active_slot]["messages"]
        )
    assert arms["S0"]["selected_steps"] == ORACLE_STEPS
    assert arms["R0"]["messages_sha256"] != arms["S0"]["messages_sha256"]
    # 缺席原因来自诊断,而不是"数臂数"推断出来的。
    assert arms["S0"]["replacement"]["wrong_old_step"] is None
    assert arms["S0"]["replacement"]["distractor_qualifying"] == 0


def test_the_content_control_flag_is_explicit_on_every_class() -> None:
    """下游判别只准读字段:六臂 True、五臂 False+原因、k=0 False+原因。"""
    six = build_positive()
    five = build_positive(utility={step: 0.5 for step in UTILITY})
    three = build_replacement_group_samples(
        episode=EPISODE,
        decision_step=CURRENT_STEP,
        instruction=INSTRUCTION,
        action_texts=PREFIX_ACTIONS,
        target_text=TARGET_TEXT,
        budget=BUDGET,
        decision=RECENT_SUFFICIENT_DECISION,
        utility={},
        delta_recent=DELTA_RECENT,
    )

    def shape(rows: list[dict]) -> tuple[int, bool, str | None]:
        return (
            len(rows),
            rows[0]["has_content_control"],
            rows[0]["content_control_absent_reason"],
        )

    assert shape(six) == (6, True, None)
    assert shape(five) == (5, False, CONTROL_ABSENT_NO_QUALIFYING)
    assert shape(three) == (3, False, CONTROL_ABSENT_RECENT_SUFFICIENT)


def test_a_missing_candidate_score_rejects_the_group_instead_of_guessing() -> None:
    partial = dict(UTILITY)
    partial[6] = None  # 拿不到分数的候选帧
    with pytest.raises(GroupRejected) as error:
        build_positive(utility=partial)
    assert error.value.reason == "distractor_utility_unavailable"


def test_an_oracle_set_that_is_not_a_single_swap_is_rejected() -> None:
    with pytest.raises(GroupRejected) as error:
        build_replacement_group_samples(
            episode=EPISODE,
            decision_step=CURRENT_STEP,
            instruction=INSTRUCTION,
            action_texts=PREFIX_ACTIONS,
            target_text=TARGET_TEXT,
            budget=BUDGET,
            decision={**POSITIVE_DECISION, "steps": [3, 4]},
            utility=dict(UTILITY),
            delta_recent=DELTA_RECENT,
        )
    assert error.value.reason == "replacement_is_not_a_single_swap"


# ---------------------------------------------------------------------------
# 6. (e) K=0 时渲染与官方基线逐字节一致
# ---------------------------------------------------------------------------
def _serialise(messages: list[dict]) -> list[dict]:
    """Mirror the builder's image serialisation so the two are comparable."""
    return [
        {
            "role": message["role"],
            "content": [
                {"type": "image", "path": part["image"]["__path__"]}
                if part.get("type") == "image"
                else dict(part)
                for part in message["content"]
            ],
        }
        for message in messages
    ]


def test_zero_budget_rendering_degrades_to_the_official_layout_byte_for_byte() -> None:
    current_path = f"images/{EPISODE}/obs-{CURRENT_STEP - 1:03d}.png"
    mine = render_messages(
        instruction=INSTRUCTION,
        action_texts=PREFIX_ACTIONS,
        full_responses=None,
        steps=[],
        image_paths=[],
        current_step=CURRENT_STEP,
        current_path=current_path,
    )
    official = _serialise(
        build_official_messages(
            goal=INSTRUCTION,
            past_action_texts=PREFIX_ACTIONS,
            recent_images=[],
            current_image={"__path__": current_path},
        )
    )
    assert mine == official
    assert messages_bytes(mine) == messages_bytes(official)


# ---------------------------------------------------------------------------
# 6b. recent_frames_kept:B=1 时替换臂里一张最近帧都没有,必须可判别
# ---------------------------------------------------------------------------
def test_recent_frames_kept_is_explicit_on_every_row() -> None:
    rows = by_slot(build_positive())
    assert rows["R0"]["recent_frames_kept"] == rows["RA"]["recent_frames_kept"] == BUDGET
    assert rows["S0"]["recent_frames_kept"] == BUDGET - 1
    assert rows[NEGATIVE_ARM_SLOT]["recent_frames_kept"] == BUDGET - 1


def test_budget_one_replacement_arms_keep_no_recent_frame_at_all() -> None:
    """B=1 的 S0-R0 差的不只是'哪张历史图',还有'有没有最近帧'——必须读得出来。"""
    budget = 1
    recent_step = CURRENT_STEP - 1  # 11
    oracle_old = 5
    # 旧帧池 1..10。4/6 有用(age 上最近),3/7 无用并列次近,规则取更老的 3。
    utility = {step: -0.02 for step in range(1, CURRENT_STEP - 1) if step != oracle_old}
    utility[4] = 0.5
    utility[6] = 0.5
    rows = by_slot(
        build_replacement_group_samples(
            episode=EPISODE,
            decision_step=CURRENT_STEP,
            instruction=INSTRUCTION,
            action_texts=PREFIX_ACTIONS,
            target_text=TARGET_TEXT,
            budget=budget,
            decision={
                "k": 1,
                "klass": CLASS_UTILITY_POSITIVE,
                "steps": [oracle_old],
                "gain": 0.03,
            },
            utility=utility,
            delta_recent=DELTA_RECENT,
        )
    )
    assert rows["R0"]["selected_steps"] == [recent_step]
    assert rows["R0"]["recent_frames_kept"] == 1
    assert rows["S0"]["selected_steps"] == [oracle_old]
    assert rows[NEGATIVE_ARM_SLOT]["selected_steps"] == [3]
    for slot in ("S0", "SA", NEGATIVE_ARM_SLOT):
        assert rows[slot]["recent_frames_kept"] == 0, slot
    # 基线臂保留了那唯一一张最近帧,所以 B=1 的 deployment_delta 仍有意义。
    assert rows["N0"]["recent_frames_kept"] == 1
    assert rows["S0"]["replacement"]["dropped_recent_step"] == recent_step


def test_every_arm_carries_exactly_budget_plus_one_images() -> None:
    for row in build_positive():
        images = [
            part
            for message in row["messages"]
            for part in message["content"]
            if part["type"] == "image"
        ]
        assert len(images) == BUDGET + 1
        assert [part["path"] for part in images[:-1]] == row["selected_images"]
        assert images[-1]["path"] == row["current_image"]


# ---------------------------------------------------------------------------
# 7. trainer 可消费性(encode_sample / adapter_context_for_sample 的入口条件)
# ---------------------------------------------------------------------------
def test_rows_are_consumable_by_the_trainers_encoder_and_adapter_dispatch() -> None:
    for row in build_positive():
        assert _prompt_encoding_path(row) == "chat_template"
        assert effective_adapter_mode(row) == row["adapter_mode"]
        # history_sample_context 的 K 只认 memory_config 或 v5 schema 常量,缺则报错。
        assert row["memory_config"]["restored_event_step_ids"] == row["selected_steps"]
        assert len(row["selected_steps"]) == row["budget"] == BUDGET
        assert row["target_text"] == TARGET_TEXT


def test_the_negative_arm_declares_its_kind_scale_and_source_steps() -> None:
    rows = by_slot(build_positive())
    negative = rows[NEGATIVE_ARM_SLOT]
    assert negative["role"] == "negative"
    assert negative["arm_id"] == "SA"
    assert negative["negative_kind"] == "age_matched"
    assert negative["negative_scale"] == 1.0
    # age_matched 的供体是**同一条 episode**(与 irrelevant 正好相反)。
    assert negative["donor_episode"] == EPISODE == negative["episode"]
    assert negative["distractor_source_step"] == EXPECTED_WRONG_OLD
    assert negative["oracle_source_step"] == ORACLE_OLD
    assert negative["distractor_source_step"] in negative["selected_steps"]
    assert negative["oracle_source_step"] not in negative["selected_steps"]
    for slot in ("N0", "R0", "RA", "S0", "SA"):
        assert "negative_kind" not in rows[slot]


# ---------------------------------------------------------------------------
# 8. 端到端:build_corpus 的拒绝计数、配比与分母对账
# ---------------------------------------------------------------------------
# 构建器只做 is_file() 存在性检查,不解码像素,所以夹具用最小 PNG 头即可。
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _annotation(current_step: int) -> dict:
    steps = [
        {"step": index, "action": "CLICK", "info": [[(index + 1) * 7, (index + 3) * 11]]}
        for index in range(current_step)
    ]
    steps.append({"step": current_step, "action": "COMPLETE", "info": None})
    return {"steps": steps}


def _corpus_group(root: Path, annotations: Path, episode: str, current_step: int) -> dict:
    """Write one v5-style decision point (N0/R0/S0 rows + screenshots + annotation)."""
    payload = _annotation(current_step)
    annotations.mkdir(parents=True, exist_ok=True)
    (annotations / f"{episode}.json").write_text(json.dumps(payload), encoding="utf-8")
    directory = root / "images" / episode
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(current_step):
        (directory / f"obs-{index:03d}.png").write_bytes(_PNG)

    actions = [f"tap the list item on row {i}" for i in range(1, current_step + 1)]
    prefix = actions[: current_step - 1]
    responses = [
        official_response(prefix[index], target_arguments(payload["steps"][index]))
        for index in range(current_step - 1)
    ]

    def image(step: int) -> str:
        return f"images/{episode}/obs-{step - 1:03d}.png"

    recent = list(range(current_step - 4, current_step))
    n0_messages = [
        {
            "role": message["role"],
            "content": [
                {"type": "image", "path": part["image"]["__path__"]}
                if part.get("type") == "image"
                else dict(part)
                for part in message["content"]
            ],
        }
        for message in build_official_messages(
            goal=INSTRUCTION,
            past_action_texts=prefix,
            recent_images=[{"__path__": image(step)} for step in recent],
            current_image={"__path__": image(current_step)},
            past_full_responses=responses,
        )
    ]
    common = {
        "episode": episode,
        "decision_step": current_step,
        "instruction": INSTRUCTION,
        "action_texts": prefix,
        "target_text": official_response(
            actions[current_step - 1], {"action": "click", "coordinate": [420, 880]}
        ),
        "current_image": image(current_step),
        "split": "train",
        "pair_group": f"{episode}:{current_step}",
    }
    sparse = [2, 5, 7, 9]
    return {
        "N0": {
            **common,
            "arm_slot": "N0",
            "selected_steps": recent,
            "selected_images": [image(step) for step in recent],
            "messages": n0_messages,
        },
        "R0": {
            **common,
            "arm_slot": "R0",
            "selected_steps": recent,
            "selected_images": [image(step) for step in recent],
        },
        "S0": {
            **common,
            "arm_slot": "S0",
            "selected_steps": sparse,
            "selected_images": [image(step) for step in sparse],
        },
    }


_ANCHOR_MEAN = -0.7312


def _score(mean: float) -> dict:
    """A cache record whose fold-balanced gain over the anchor is ``mean - anchor``."""
    return {
        "mean": mean,
        "n_tok": 4,
        "n_even": 2,
        "n_odd": 2,
        "sum_even": 2 * mean,
        "sum_odd": 2 * mean,
        "prompt_tokens": 100,
        "total_tokens": 104,
    }


def _label_inputs(pair_group: str, utility: dict[int, float], *, positive: bool) -> dict:
    """Raw-label block + score-cache entries consistent with one utility table."""
    fmt = SPARSE_MULTITURN_PROMPT_FORMAT
    scores = {f"{pair_group}|{fmt}|{'-'.join(map(str, RECENT))}": _score(_ANCHOR_MEAN)}
    kept = [step for step in RECENT if step != DROPPED]
    for old, value in utility.items():
        steps = sorted(kept + [old])
        key = f"{pair_group}|{fmt}|{'-'.join(map(str, steps))}"
        scores[key] = _score(_ANCHOR_MEAN + value)
    block: dict[str, Any] = {
        "recent_steps": RECENT,
        "anchor_mean": _ANCHOR_MEAN,
        "n_old": CURRENT_STEP - BUDGET - 1,
    }
    if positive:
        block["k1"] = {
            "crossfit_gain": POSITIVE_DECISION["gain"],
            "both_folds_positive": True,
            "in_sample_best_steps": ORACLE_STEPS,
            "pool_size": POSITIVE_DECISION["pool_size"],
        }
        block["k1_drop_positions"] = "all"
    return {"scores": scores, "block": block}


def test_build_corpus_degrades_to_four_arms_and_preserves_the_natural_class_ratio(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v5"
    annotations = tmp_path / "annotations"
    episodes = {
        "positive": "0000000000000101",
        "no_control": "0000000000000202",
        "sufficient": "0000000000000303",
    }
    corpus_index: dict[str, dict] = {}
    raw_labels: dict[str, dict] = {}
    scores: dict[str, dict] = {}
    decided: list[dict] = []

    oracle_utility = dict(UTILITY)
    oracle_utility[ORACLE_OLD] = POSITIVE_DECISION["gain"]
    tables = {
        "positive": (oracle_utility, POSITIVE_DECISION, True),
        # 每张旧帧都有用 -> 找不到合格 distractor -> 退成五臂,**不**计入 reject
        "no_control": ({step: 0.5 for step in oracle_utility}, POSITIVE_DECISION, True),
        "sufficient": ({}, RECENT_SUFFICIENT_DECISION, False),
    }
    for name, episode in episodes.items():
        utility, decision, positive = tables[name]
        pair_group = f"{episode}:{CURRENT_STEP}"
        corpus_index[pair_group] = _corpus_group(
            root, annotations, episode, CURRENT_STEP
        )
        payload = _label_inputs(pair_group, utility, positive=positive)
        scores.update(payload["scores"])
        raw_labels[pair_group] = {
            "pair_group": pair_group,
            "episode": episode,
            "current_step": CURRENT_STEP,
            "n_candidates": CURRENT_STEP - 1,
            "format": SPARSE_MULTITURN_PROMPT_FORMAT,
            "budgets": {str(BUDGET): payload["block"]},
        }
        decided.append(
            {
                "pair_group": pair_group,
                "episode": episode,
                "current_step": CURRENT_STEP,
                "format": SPARSE_MULTITURN_PROMPT_FORMAT,
                "budgets": {str(BUDGET): decision},
            }
        )

    rows, manifest = build_corpus(
        decided_rows=decided,
        corpus_index=corpus_index,
        raw_labels=raw_labels,
        scores=scores,
        annotations=annotations,
        budgets=[BUDGET],
        delta_recent=DELTA_RECENT,
        image_root=root,
    )

    assert manifest["decisions_seen"] == 3
    assert manifest["groups"] == 3
    # 六臂 + 五臂 + 三臂 —— 没有一组因为缺 distractor 被丢掉。
    assert manifest["samples"] == len(rows) == 6 + 5 + 3
    assert manifest["rejected_reasons"] == {}
    assert manifest["rejected_groups"] == 0
    assert manifest["groups_by_class"] == {
        CLASS_RECENT_SUFFICIENT: 1,
        CLASS_UTILITY_POSITIVE: 2,
    }
    assert manifest["class_fractions"][CLASS_UTILITY_POSITIVE] == 2 / 3
    assert manifest["oversampled"] is False
    assert manifest["class_balancing"].startswith("none")
    assert manifest["deployment_baseline_arm_id"] == "N0"
    assert manifest["rows_per_group"] == {"3": 1, "5": 1, "6": 1}
    assert manifest["content_control"]["six_arm_groups"] == 1
    assert manifest["content_control"]["five_arm_groups"] == 1
    assert manifest["content_control"]["negative_arm_slot"] == NEGATIVE_ARM_SLOT
    assert manifest["content_control"]["six_arm_fraction_of_positives"] == 0.5
    assert manifest["content_control"]["absent_reasons"] == {
        CONTROL_ABSENT_NO_QUALIFYING: 1,
        CONTROL_ABSENT_RECENT_SUFFICIENT: 1,
    }
    # 六臂组与五臂组的 gain 分开报,选择效应不会被合并掩盖。
    assert manifest["oracle_gain_by_arms"]["six_arm"]["n"] == 1
    assert manifest["oracle_gain_by_arms"]["five_arm"]["n"] == 1
    assert manifest["oracle_gain_by_arms"]["all_positive"]["n"] == 2
    assert manifest["distractor"]["age_gap"]["n"] == 1

    emitted = {row["pair_group"] for row in rows}
    assert emitted == {
        f"{episode}:{CURRENT_STEP}:b{BUDGET}" for episode in episodes.values()
    }
    no_control_rows = [
        row for row in rows if row["episode"] == episodes["no_control"]
    ]
    assert {row["arm_slot"] for row in no_control_rows} == set(BASE_ARM_SLOTS)
    assert all(row["has_content_control"] is False for row in no_control_rows)
    positive_rows = by_slot(
        [row for row in rows if row["episode"] == episodes["positive"]]
    )
    assert set(positive_rows) == {spec[0] for spec in ARM_SPECS}
    assert positive_rows["S0"]["selected_steps"] == ORACLE_STEPS
    assert positive_rows[NEGATIVE_ARM_SLOT]["selected_steps"] == sorted(
        [step for step in RECENT if step != DROPPED] + [EXPECTED_WRONG_OLD]
    )
    for bypass_slot, active_slot in ADAPTER_PAIRS:
        assert messages_bytes(positive_rows[bypass_slot]["messages"]) == messages_bytes(
            positive_rows[active_slot]["messages"]
        )
    # 历史响应来自真实 annotation 重建,并与 N0 的 assistant 轮对过账。
    assistant_texts = [
        message["content"][0]["text"]
        for message in positive_rows["S0"]["messages"]
        if message["role"] == "assistant"
    ]
    assert len(assistant_texts) == BUDGET
    assert all("<tool_call>" in text for text in assistant_texts)


# ---------------------------------------------------------------------------
# 9. trainer 集成:validate_* 与损失分发器真的认这些槽位
# ---------------------------------------------------------------------------
# note (luojiaxuan): 这一节是为了钉死一类**静默失效**:v5 的 validate_sparse_sample 里
# 写着 ``if slot != f"SA_neg_{kind}": raise``,而 DiD 损失的 negatives 字典是按
# ``role == "negative"`` 收的。名字不合契约时,轻则启动报错(还算好),重则该槽位压根
# 不被当成负样本 —— negatives 为空,L_content 与 A_n 一起消失,损失照常下降,日志里
# 什么都看不出来。所以"encode_sample 能跑"远远不够,必须让**损失分发器**表态。
class _Recorder:
    """Stand-in for the trainer's per-arm forward: returns a fixed value per slot."""

    def __init__(self, values: dict[str, float]) -> None:
        self.values = values
        self.calls: list[tuple[str, bool, str | None]] = []

    def __call__(
        self,
        slot: str,
        *,
        grad: bool = False,
        adapter_mode: str | None = None,
        backward_weight: float | None = None,
    ) -> float | None:
        self.calls.append((slot, grad, adapter_mode))
        if adapter_mode == "bypass":
            # 同一行负样本在冻结 policy 上的分数 —— A_n 的基准点。
            return self.values.get(f"{slot}|bypass", 0.0)
        return self.values.get(slot, 0.0)


def _trainer_group(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    return rows, {row["arm_slot"]: index for index, row in enumerate(rows)}


def test_v6_rows_pass_the_trainers_sample_and_group_validation() -> None:
    rows = build_positive()
    for index, row in enumerate(rows):
        trainer.validate_sparse_sample(row, index=index)
    samples, group = _trainer_group(rows)
    assert (
        trainer.validate_sparse_group(samples, pair_group=rows[0]["pair_group"], group=group)
        == "R0"
    )


def test_the_negative_arm_is_collected_by_the_did_loss_dispatcher() -> None:
    """负样本字典必须非空且 key 是 SA_neg_age_matched —— 否则 L_content 静默消失。"""
    samples, group = _trainer_group(build_positive())
    values = {"SA": 0.10, "S0": 0.05, "RA": 0.02, "R0": 0.0}
    values[NEGATIVE_ARM_SLOT] = -0.30
    values[f"{NEGATIVE_ARM_SLOT}|bypass"] = -0.32
    recorder = _Recorder(values)
    diagnostics: dict[str, float] = {}
    total = trainer._sparse_history_group_loss(
        samples=samples,
        group=group,
        forward=recorder,
        training={},
        adapter_parameters=[],
        accumulation=1,
        torch=None,
        objective_kind=trainer.SPARSE_OBJECTIVE_DID_RA_AWARE,
        diagnostics_out=diagnostics,
    )
    assert total is not None
    # 负样本真的被收进来了(而不是"这组没有负样本")。
    assert diagnostics["negatives"] == 1.0
    # 诊断键按 kind 生成,与 gate 词表同一处定义。
    gap_key, drift_key = trainer.sparse_diagnostic_keys("age_matched")
    assert gap_key == "SA_minus_SA_neg_age_matched"
    assert drift_key == "age_matched_drift_abs"
    assert diagnostics[gap_key] == pytest.approx(0.10 - (-0.30), abs=1e-12)
    assert diagnostics[drift_key] == pytest.approx(abs(-0.30 - (-0.32)), abs=1e-12)
    assert diagnostics[trainer.sparse_content_diagnostic_key("age_matched")] == (
        pytest.approx((0.10 - 0.05) - (-0.30 - (-0.32)), abs=1e-12)
    )
    # A_n 的两次前向来自**同一行**:一次 active、一次显式 bypass 覆盖。
    negative_calls = [call for call in recorder.calls if call[0] == NEGATIVE_ARM_SLOT]
    assert (NEGATIVE_ARM_SLOT, False, None) in negative_calls
    assert (NEGATIVE_ARM_SLOT, False, "bypass") in negative_calls
    # 诊断键必须在 gate 词表里,否则留出集报告里这一项会静默消失。
    quantities = set(diagnostics) - set(trainer.SPARSE_RUNTIME_DIAGNOSTIC_KEYS)
    assert quantities <= set(trainer.SPARSE_GATE_VOCABULARY), sorted(
        quantities - set(trainer.SPARSE_GATE_VOCABULARY)
    )


def test_a_five_arm_group_still_runs_the_did_loss_without_negatives() -> None:
    """没有内容对照臂时 DiD 主项照常成立,只是 L_content 这一项为空。"""
    samples, group = _trainer_group(
        build_positive(utility={step: 0.5 for step in UTILITY})
    )
    diagnostics: dict[str, float] = {}
    # A_c = 0.02、A_r = 0.05 -> did_select 为负,L_select 被激活,损失非零。
    total = trainer._sparse_history_group_loss(
        samples=samples,
        group=group,
        forward=_Recorder({"SA": 0.02, "S0": 0.0, "RA": 0.05, "R0": 0.0}),
        training={},
        adapter_parameters=[],
        accumulation=1,
        torch=None,
        objective_kind=trainer.SPARSE_OBJECTIVE_DID_RA_AWARE,
        diagnostics_out=diagnostics,
    )
    assert total is not None and total > 0.0
    assert diagnostics["negatives"] == 0.0
    assert diagnostics["did_select"] == pytest.approx(0.02 - 0.05, abs=1e-12)
    assert diagnostics["loss_content"] == 0.0


def test_the_v5_negative_kinds_keep_their_frozen_gate_requirements() -> None:
    """新增 kind 不得改动 v5 的必需 gate —— 否则既有 config 会当场报错。"""
    assert trainer.SPARSE_NEGATIVE_KINDS == ("step_shuffled", "irrelevant", "duplicate")
    # note (luojiaxuan): 桌面 DiD schema 追加了 wrong;冻结的是 v5 kinds 与 v5/v6 的
    # 必需 gate,不是这个并集的长度 —— 并集本就是"词表认识哪些名字"的来源。
    assert trainer.SPARSE_ALL_NEGATIVE_KINDS == (
        "step_shuffled", "irrelevant", "duplicate", "age_matched", "wrong",
    )
    assert "age_matched_drift_abs" not in trainer.SPARSE_REQUIRED_GATES
    assert "wrong_drift_abs" not in trainer.SPARSE_REQUIRED_GATES
    for kind in trainer.SPARSE_NEGATIVE_KINDS:
        _gap, drift = trainer.sparse_diagnostic_keys(kind)
        assert drift in trainer.SPARSE_REQUIRED_GATES
    # 但词表必须认识它,否则打分脚本报告里这一项会静默消失。
    assert "age_matched_drift_abs" in trainer.SPARSE_GATE_VOCABULARY
    assert "SA_minus_SA_neg_age_matched" in trainer.SPARSE_GATE_VOCABULARY
    assert "wrong_drift_abs" in trainer.SPARSE_GATE_VOCABULARY
    assert "SA_minus_WA" in trainer.SPARSE_GATE_VOCABULARY


def test_the_age_matched_negative_rejects_a_foreign_donor() -> None:
    """跨 episode 供体是 irrelevant 的语义;age_matched 要求同 episode。"""
    rows = build_positive()
    negative = dict(by_slot(rows)[NEGATIVE_ARM_SLOT])
    negative["donor_episode"] = "some-other-episode"
    with pytest.raises(ValueError, match="its own episode"):
        trainer.validate_sparse_sample(negative, index=0)


def test_the_age_matched_negative_rejects_the_oracle_frame_as_its_own_source() -> None:
    rows = build_positive()
    negative = dict(by_slot(rows)[NEGATIVE_ARM_SLOT])
    negative["distractor_source_step"] = negative["oracle_source_step"]
    with pytest.raises(ValueError, match="oracle frame itself"):
        trainer.validate_sparse_sample(negative, index=0)


def test_the_age_matched_negative_rejects_source_steps_inside_the_recent_window() -> None:
    rows = build_positive()
    negative = dict(by_slot(rows)[NEGATIVE_ARM_SLOT])
    negative["oracle_source_step"] = RECENT[-1]
    with pytest.raises(ValueError, match="Recent-"):
        trainer.validate_sparse_sample(negative, index=0)


# ---------------------------------------------------------------------------
# 10. data.train_on_label_classes:k=0 三臂组必须**显式**排除,不是静默跳过
# ---------------------------------------------------------------------------
def _recent_sufficient_group(episode: str) -> list[dict]:
    return build_replacement_group_samples(
        episode=episode,
        decision_step=CURRENT_STEP,
        instruction=INSTRUCTION,
        action_texts=PREFIX_ACTIONS,
        target_text=TARGET_TEXT,
        budget=BUDGET,
        decision=RECENT_SUFFICIENT_DECISION,
        utility={},
        delta_recent=DELTA_RECENT,
    )


def _mixed_corpus() -> list[dict]:
    """One six-arm positive group + one three-arm recent_sufficient group."""
    return build_positive() + _recent_sufficient_group("0000000000000909")


def test_without_the_config_key_a_three_arm_group_still_fails_closed() -> None:
    """缺臂静默跳过正是 fail-closed 该拦的;不声明就必须报错。"""
    assert trainer.resolve_train_on_label_classes({"data": {}}) is None
    assert trainer.resolve_train_on_label_classes({}) is None
    with pytest.raises(ValueError, match="lacks the SA arm"):
        trainer.build_sparse_history_units(
            _mixed_corpus(), objective_kind=trainer.SPARSE_OBJECTIVE_DID_RA_AWARE
        )


def test_declaring_the_key_excludes_k0_groups_with_a_counted_record() -> None:
    config = {"data": {"train_on_label_classes": ["utility_positive"]}}
    allowed = trainer.resolve_train_on_label_classes(config)
    assert allowed == ("utility_positive",)
    selection: dict[str, object] = {}
    kept = trainer.filter_samples_by_label_class(
        _mixed_corpus(), allowed=allowed, selection_out=selection
    )
    assert selection == {
        "filter": ["utility_positive"],
        "kept_groups": {"train": {CLASS_UTILITY_POSITIVE: 1}},
        "skipped_groups": {"train": {CLASS_RECENT_SUFFICIENT: 1}},
        "kept_samples": 6,
        "skipped_samples": 3,
    }
    assert {row["label_class"] for row in kept} == {CLASS_UTILITY_POSITIVE}
    # 过滤后单元构造不再报错,组数与计数一致。
    units, heldout = trainer.build_sparse_history_units(
        kept, objective_kind=trainer.SPARSE_OBJECTIVE_DID_RA_AWARE
    )
    assert len(units) == 1 and heldout == set()


def test_a_filter_that_excludes_every_train_group_is_an_error() -> None:
    with pytest.raises(ValueError, match="excluded every train pair-group"):
        trainer.filter_samples_by_label_class(
            _recent_sufficient_group("0000000000000910"),
            allowed=("utility_positive",),
        )


def test_the_filter_refuses_a_corpus_without_label_class() -> None:
    """声明了该键却喂 v5 语料 —— 全丢与全留都是错的,只能报错。"""
    rows = [dict(row) for row in build_positive()]
    for row in rows:
        row.pop("label_class")
    with pytest.raises(ValueError, match="carries no label_class field"):
        trainer.filter_samples_by_label_class(rows, allowed=("utility_positive",))


def test_an_unknown_label_class_in_the_config_is_rejected() -> None:
    """拼错类名会静默把整批组过滤掉,训练照跑、组数变少、日志看不出来。"""
    with pytest.raises(ValueError, match="unknown label classes"):
        trainer.resolve_train_on_label_classes(
            {"data": {"train_on_label_classes": ["utility_postive"]}}
        )
    with pytest.raises(ValueError, match="non-empty list"):
        trainer.resolve_train_on_label_classes({"data": {"train_on_label_classes": []}})


# ---------------------------------------------------------------------------
# 11. 必需 gate 集合随语料 schema 变(两个方向都覆盖)
# ---------------------------------------------------------------------------
V6_CONFIG_PATH = _CODE_ROOT / "configs" / "causalcache_replacement_v6_a36.json"


def _v6_config() -> dict:
    return json.loads(V6_CONFIG_PATH.read_text(encoding="utf-8"))


def test_the_shipped_v6_config_passes_all_four_validators() -> None:
    import argparse

    config = _v6_config()
    adapter_type, options = trainer.adapter_settings(config)
    assert adapter_type == "history_gated_kv"
    assert options == {"layer_count": 36, "rank": 8, "alpha": 16}
    kind, _objective = trainer.validate_sparse_objective(config)
    assert kind == trainer.SPARSE_OBJECTIVE_DID_RA_AWARE
    gates = trainer.validate_sparse_gates(config)
    assert set(gates["must_pass"]) == {
        "did_select", "adapter_on_sparse", "adapter_on_recent_abs",
        "age_matched_drift_abs",
    }
    controls = trainer.resolve_sparse_training(
        config["training"],
        args=argparse.Namespace(max_steps=None, checkpoint_every_steps=None),
        objective_kind=kind,
    )
    assert controls == {"max_steps": 50, "checkpoint_every_steps": 25}
    # 语料侧与 config 侧必须指向同一个 schema,否则必需 gate 会按另一套算。
    assert config["data"]["sample_schema_version"] == SCHEMA_VERSION
    assert config["data"]["train_on_label_classes"] == [CLASS_UTILITY_POSITIVE]


def test_required_gates_follow_the_corpus_schema_in_both_directions() -> None:
    v2, v6 = trainer.SPARSE_SAMPLE_SCHEMA, trainer.SPARSE_REPLACEMENT_SAMPLE_SCHEMA
    assert trainer.SPARSE_REQUIRED_GATES_BY_SCHEMA[v2] == {
        "did_select", "adapter_on_sparse", "adapter_on_recent_abs",
        "step_shuffled_drift_abs", "irrelevant_drift_abs", "duplicate_drift_abs",
    }
    assert trainer.SPARSE_REQUIRED_GATES_BY_SCHEMA[v6] == {
        "did_select", "adapter_on_sparse", "adapter_on_recent_abs",
        "age_matched_drift_abs",
    }
    # v5 的取值逐字不变 —— 既有 config 与既有测试都靠这个常量。
    assert trainer.SPARSE_REQUIRED_GATES == trainer.SPARSE_REQUIRED_GATES_BY_SCHEMA[v2]


def test_a_v6_config_missing_the_age_matched_gate_is_rejected() -> None:
    config = _v6_config()
    config["gates"]["must_pass"].pop("age_matched_drift_abs")
    with pytest.raises(ValueError, match="misses required gates"):
        trainer.validate_sparse_gates(config)


def test_a_v6_config_declaring_v5_negatives_is_rejected() -> None:
    """声明了语料里不存在的量 = 打分时拿到 None,提前到训练启动前拦。"""
    config = _v6_config()
    config["gates"]["must_pass"]["irrelevant_drift_abs"] = "< 0.02"
    with pytest.raises(ValueError, match="never produces those negatives"):
        trainer.validate_sparse_gates(config)


def test_a_v2_config_keeps_its_frozen_gate_requirements() -> None:
    """缺 data.sample_schema_version 时按 v2 处理,既有 config 行为逐字不变。"""
    config = _v6_config()
    config["data"].pop("sample_schema_version")
    config["gates"]["must_pass"] = {
        "adapter_on_recent_abs": "< 0.02",
        "adapter_on_sparse": "> 0",
        "did_select": "> 0 @ci_low",
        "duplicate_drift_abs": "< 0.02",
        "irrelevant_drift_abs": "< 0.02",
        "step_shuffled_drift_abs": "< 0.02",
    }
    assert trainer.validate_sparse_gates(config)["must_pass"]
    # 反方向:v2 语料不产出 age_matched,声明它同样被拦。
    config["gates"]["must_pass"]["age_matched_drift_abs"] = "< 0.02"
    with pytest.raises(ValueError, match="never produces those negatives"):
        trainer.validate_sparse_gates(config)


def test_a_misspelled_sample_schema_version_is_not_silently_defaulted() -> None:
    config = _v6_config()
    config["data"]["sample_schema_version"] = "causalcache.replacement_corpus_sample.v7"
    with pytest.raises(ValueError, match="is not one of"):
        trainer.validate_sparse_gates(config)


def test_build_corpus_rejects_a_score_cache_from_another_run(tmp_path: Path) -> None:
    """anchor_mean 对不上就说明 glob 到的分数不是产出这批标签的那一份。"""
    root = tmp_path / "v5"
    annotations = tmp_path / "annotations"
    episode = "0000000000000404"
    pair_group = f"{episode}:{CURRENT_STEP}"
    corpus_index = {pair_group: _corpus_group(root, annotations, episode, CURRENT_STEP)}
    utility = dict(UTILITY)
    utility[ORACLE_OLD] = POSITIVE_DECISION["gain"]
    payload = _label_inputs(pair_group, utility, positive=True)
    payload["block"]["anchor_mean"] = _ANCHOR_MEAN + 0.5  # 另一批数
    rows, manifest = build_corpus(
        decided_rows=[
            {
                "pair_group": pair_group,
                "episode": episode,
                "current_step": CURRENT_STEP,
                "format": SPARSE_MULTITURN_PROMPT_FORMAT,
                "budgets": {str(BUDGET): POSITIVE_DECISION},
            }
        ],
        corpus_index=corpus_index,
        raw_labels={
            pair_group: {
                "pair_group": pair_group,
                "episode": episode,
                "current_step": CURRENT_STEP,
                "format": SPARSE_MULTITURN_PROMPT_FORMAT,
                "budgets": {str(BUDGET): payload["block"]},
            }
        },
        scores=payload["scores"],
        annotations=annotations,
        budgets=[BUDGET],
        delta_recent=DELTA_RECENT,
        image_root=root,
    )
    assert rows == []
    assert manifest["rejected_reasons"] == {"score_cache_provenance_mismatch": 1}
