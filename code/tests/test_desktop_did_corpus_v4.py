"""v4 官方多轮语料的冻结不变量。

# note (luojiaxuan): v3 组代数(窗口/替换槽位/age≥B+2/split)已由 v3 测试锁定,
# 本文件只锁 v4 的新增面:
#   1. 语料侧臂契约与 trainer 的 SPARSE_DESKTOP_ARM_CONTRACT_V2 逐字段一致,
#      prompt_format 在 trainer 的 chat_template 白名单里;
#   2. 行结构:R/S/W 图数恒为 B+1(含当前),R0/RA 与 S0/SA 孪生同 prompt,
#      行过 trainer 校验(normalize_desktop_splits 后);
#   3. 量纲:消息里每个 <tool_call> 坐标 ∈ [0,999] —— 单轮结构的像素混用不复现;
#   4. target_text = 官方完整响应(Action 行 + tool_call);
#   5. 保留步缺官方响应(tripleClick)→ 整组弃用并计数;
#   6. B0 parity 行 = 官方 kept=0 单 user 轮。
"""

from __future__ import annotations

import json

import pytest

import scripts.train_success_sft_lora as trainer
from causalcache.agentnet_actions import parse_agentnet_step
from scripts.build_desktop_did_corpus_v2 import recent_window
from scripts.build_desktop_did_corpus_v4 import (
    B0_SAMPLE_SCHEMA_V2,
    DESKTOP_ARM_CONTRACT_V2,
    PROMPT_FORMAT_V2,
    SAMPLE_SCHEMA_V2,
    build_b_groups,
)


def tool(action: str, **kwargs):
    return {"name": "computer_use", "arguments": {"action": action, **kwargs}}


def synthetic_record(*, steps: int, match_steps: tuple[int, ...],
                     dp_id="dp-1", task_id="traj-1",
                     triple_click_steps: tuple[int, ...] = ()):
    """决策点 fixture:history 的 action 一律由 raw_code 经冻结解析器导出。"""
    screen = (1000, 1000)
    relpaths = [f"{task_id}/obs-{index:03d}.png" for index in range(steps)]
    history = []
    for step_id in range(1, steps):
        if step_id in triple_click_steps:
            raw = "pyautogui.click(x=0.7, y=0.7, clicks=3)"
        elif step_id in match_steps:
            raw = "pyautogui.click(x=0.5005, y=0.5005)"
        elif step_id % 2:
            x = round(0.03 + 0.001 * step_id, 6)
            raw = f"pyautogui.click(x={x}, y=0.04)"
        else:
            raw = "pyautogui.press('tab')"
        parsed = parse_agentnet_step(raw, screen_size=screen)
        assert parsed.unparseable_reason is None
        history.append({
            "step_id": step_id,
            "action": parsed.history_action,
            "osworld_action": parsed.osworld_action,
            "raw_code": raw,
        })
    return {
        "dp_id": dp_id,
        "task_id": task_id,
        "os": "ubuntu",
        "step": steps,
        "screen_size": list(screen),
        "instruction": "demo task",
        "target_tool_call": tool("left_click", coordinate=[500, 500]),
        "target_text": '<tool_call>legacy</tool_call>',
        "history": history,
        "image_relpaths": relpaths,
    }


def image_paths(messages) -> list[str]:
    return [
        part["path"]
        for message in messages
        for part in message["content"]
        if part.get("type") == "image"
    ]


def test_contract_matches_trainer_v2_tables() -> None:
    assert DESKTOP_ARM_CONTRACT_V2 == trainer.SPARSE_DESKTOP_ARM_CONTRACT_V2
    assert SAMPLE_SCHEMA_V2 == trainer.SPARSE_DESKTOP_SAMPLE_SCHEMA_V2
    assert PROMPT_FORMAT_V2 == trainer.SPARSE_DESKTOP_PROMPT_FORMAT_V2
    assert PROMPT_FORMAT_V2 in trainer._CHAT_TEMPLATE_PROMPT_FORMATS
    assert SAMPLE_SCHEMA_V2 in trainer.SPARSE_SAMPLE_SCHEMAS
    assert trainer.SPARSE_DEPLOYMENT_BASELINE_BY_SCHEMA[SAMPLE_SCHEMA_V2] == "R0"


@pytest.mark.parametrize("b", [1, 2, 4])
def test_fixed_budget_official_rows(b) -> None:
    record = synthetic_record(steps=16, match_steps=(3,))
    rows, parity, stats = build_b_groups(
        [record], b=b, coordinate_tolerance=2, seed=7, emit_parity=(b == 1)
    )
    assert stats["counters"]["groups"] == 1
    by_slot = {row["arm_slot"]: row for row in rows}
    assert set(by_slot) == {"R0", "RA", "S0", "SA", "WA"}
    window = recent_window(16, b)
    # R 臂 = 官方连续窗口;S/W = 替换最老槽位;图数恒 B+1(含当前帧)。
    assert by_slot["R0"]["selected_steps"] == window
    for slot, row in by_slot.items():
        assert row["schema_version"] == SAMPLE_SCHEMA_V2
        assert row["prompt_format"] == PROMPT_FORMAT_V2
        assert len(image_paths(row["messages"])) == b + 1
        assert image_paths(row["messages"])[-1] == record["image_relpaths"][15]
        assert image_paths(row["messages"])[:-1] == [
            record["image_relpaths"][step] for step in row["selected_steps"]
        ]
    assert by_slot["R0"]["messages_sha256"] == by_slot["RA"]["messages_sha256"]
    assert by_slot["S0"]["messages_sha256"] == by_slot["SA"]["messages_sha256"]
    assert by_slot["WA"]["messages_sha256"] != by_slot["SA"]["messages_sha256"]
    # 正例旧帧 = 事件 2(匹配动作步 3 的前状态),恒替换最老槽位。
    assert by_slot["SA"]["selected_steps"] == sorted([2, *window[1:]])
    # target = 官方完整响应格式。
    assert by_slot["SA"]["target_text"].startswith("Action: Click at (500, 500).")
    assert "<tool_call>" in by_slot["SA"]["target_text"]


def test_rows_pass_trainer_validation_after_split_normalization() -> None:
    record = synthetic_record(steps=16, match_steps=(3,))
    rows, _, _ = build_b_groups(
        [record], b=4, coordinate_tolerance=2, seed=7, emit_parity=False
    )
    normalized, counters = trainer.normalize_desktop_splits(rows)
    for index, row in enumerate(normalized):
        trainer.validate_sparse_sample(row, index=index)
    assert len(normalized) + counters["desktop_test_dropped"] == len(rows)


def test_all_rendered_coordinates_are_normalized() -> None:
    record = synthetic_record(steps=16, match_steps=(3,))
    rows, _, _ = build_b_groups(
        [record], b=4, coordinate_tolerance=2, seed=7, emit_parity=False
    )
    for row in rows:
        # system prompt 里的 <tool_call> 模板是占位符不是 JSON,跳过 system 轮。
        for message in row["messages"]:
            if message["role"] == "system":
                continue
            for part in message["content"]:
                if part.get("type") != "text":
                    continue
                text = part["text"]
                start = 0
                while "<tool_call>" in text[start:]:
                    payload = text.split("<tool_call>", 1)[1].split(
                        "</tool_call>", 1)[0]
                    call = json.loads(payload)
                    for key in ("coordinate", "coordinate2"):
                        for value in call["arguments"].get(key, []):
                            assert 0 <= value <= 999
                    text = text.split("</tool_call>", 1)[1]


def test_triple_click_in_retained_window_drops_group() -> None:
    # B=1 的唯一保留步是 s-1;把它做成 tripleClick → target 形态缺失,整组弃用。
    record = synthetic_record(
        steps=16, match_steps=(3,), triple_click_steps=(15,)
    )
    rows, _, stats = build_b_groups(
        [record], b=1, coordinate_tolerance=2, seed=7, emit_parity=False
    )
    assert rows == []
    assert stats["counters"]["dropped_retained_unrenderable"] == 1
    # 同一步只被折叠(B=1 窗口不含它)时正常产组:tripleClick 放在窗口外。
    record2 = synthetic_record(
        steps=16, match_steps=(3,), triple_click_steps=(9,),
        dp_id="dp-2", task_id="traj-2",
    )
    rows2, _, stats2 = build_b_groups(
        [record2], b=1, coordinate_tolerance=2, seed=7, emit_parity=False
    )
    assert stats2["counters"]["groups"] == 1
    folded = "\n".join(
        part["text"]
        for row in rows2
        for message in row["messages"]
        for part in message["content"]
        if part.get("type") == "text"
    )
    assert "Triple-click at" in folded


def test_b0_parity_row_is_official_kept_zero() -> None:
    record = synthetic_record(steps=16, match_steps=(3,))
    _, parity, _ = build_b_groups(
        [record], b=1, coordinate_tolerance=2, seed=7, emit_parity=True
    )
    assert len(parity) == 1
    row = parity[0]
    assert row["schema_version"] == B0_SAMPLE_SCHEMA_V2
    assert row["selected_steps"] == []
    assert len(row["messages"]) == 2  # system + 单 user 轮
    assert image_paths(row["messages"]) == [record["image_relpaths"][15]]
    # 全部 15 步都折叠进第一轮文本。
    first_text = row["messages"][1]["content"][0]["text"]
    assert "Step15:" in first_text and "Step1:" in first_text
