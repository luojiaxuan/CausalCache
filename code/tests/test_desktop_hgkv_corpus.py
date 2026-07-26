"""Desktop DiD 六臂语料构建器的冻结不变量。

# note (luojiaxuan): 上半部锁纯函数(完整动作等价 + 正/错对照事件选择,与四臂
# prototype 语义逐字不变);下半部用真 PNG 走 build_corpus,锁交接 §6 的组结构:
# R0/RA 与 S0/SA 孪生同 prompt、WA 单行负臂账目字段、B0 独立 schema 单图、
# recent 臂 = 冻结部署选择器、确定性重建。
"""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from scripts.build_desktop_hgkv_corpus import (
    B0_SAMPLE_SCHEMA,
    DESKTOP_ARM_CONTRACT,
    SAMPLE_SCHEMA,
    build_corpus,
    history_action_matches_target,
    select_contrast_events,
)


def tool(action: str, **kwargs):
    return {"name": "computer_use", "arguments": {"action": action, **kwargs}}


def test_full_click_equivalence_uses_coordinates() -> None:
    screen = (1920, 1080)
    assert history_action_matches_target(
        {"type": "click", "x": 960, "y": 540},
        tool("left_click", coordinate=[500, 500]),
        screen_size=screen,
        coordinate_tolerance=2,
    )
    assert not history_action_matches_target(
        {"type": "click", "x": 1300, "y": 540},
        tool("left_click", coordinate=[500, 500]),
        screen_size=screen,
        coordinate_tolerance=25,
    )


def test_text_and_key_equivalence_include_parameters() -> None:
    assert history_action_matches_target(
        {"type": "type_text", "text": "hello"},
        tool("type", text="hello"),
        screen_size=(100, 100),
        coordinate_tolerance=0,
    )
    assert not history_action_matches_target(
        {"type": "type_text", "text": "bye"},
        tool("type", text="hello"),
        screen_size=(100, 100),
        coordinate_tolerance=0,
    )
    assert history_action_matches_target(
        {"type": "hotkey", "keys": ["CTRL", "S"]},
        tool("key", keys=["ctrl", "s"]),
        screen_size=(100, 100),
        coordinate_tolerance=0,
    )


def test_selects_pre_state_and_age_matched_wrong_state() -> None:
    record = {
        "dp_id": "dp-1",
        "step": 8,
        "screen_size": [1000, 1000],
        "target_tool_call": tool("left_click", coordinate=[500, 500]),
        "history": [
            {"step_id": 1, "action": {"type": "click", "x": 10, "y": 10}},
            {"step_id": 2, "action": {"type": "click", "x": 500, "y": 500}},
            {"step_id": 3, "action": {"type": "press", "key": "tab"}},
            {"step_id": 4, "action": {"type": "click", "x": 50, "y": 50}},
            {"step_id": 5, "action": {"type": "click", "x": 500, "y": 500}},
            {"step_id": 6, "action": {"type": "press", "key": "enter"}},
            {"step_id": 7, "action": {"type": "click", "x": 80, "y": 80}},
        ],
    }
    selected = select_contrast_events(
        record, coordinate_tolerance=2, min_age=3, seed=7
    )
    assert selected is not None
    positive_event, positive_action_step, wrong_event = selected
    assert (positive_event, positive_action_step) in {(1, 2), (4, 5)}
    assert wrong_event != positive_event
    assert 8 - wrong_event >= 3


# ---------------------------------------------------------------------------
# build_corpus 端到端(真 PNG + 冻结 renderer)
# ---------------------------------------------------------------------------

STEPS = 8


def png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def synthetic_record(tmp_path, *, dp_id="dp-1", task_id="traj-1", recurrence=True):
    relpaths = []
    (tmp_path / task_id).mkdir(exist_ok=True)
    for index in range(STEPS):
        relpath = f"{task_id}/obs-{index:03d}.png"
        (tmp_path / relpath).write_bytes(png_bytes((index * 25 % 255, 40, 70)))
        relpaths.append(relpath)
    match_xy = {"x": 500, "y": 500} if recurrence else {"x": 900, "y": 900}
    history = [
        {"step_id": 1, "action": {"type": "click", "x": 10, "y": 10}},
        {"step_id": 2, "action": {"type": "click", **match_xy}},
        {"step_id": 3, "action": {"type": "press", "key": "tab"}},
        {"step_id": 4, "action": {"type": "click", "x": 50, "y": 50}},
        {"step_id": 5, "action": {"type": "click", **match_xy}},
        {"step_id": 6, "action": {"type": "press", "key": "enter"}},
        {"step_id": 7, "action": {"type": "click", "x": 80, "y": 80}},
    ]
    for entry in history:
        action = entry["action"]
        entry["osworld_action"] = f"pyautogui.{action['type']}()"
    return {
        "dp_id": dp_id,
        "task_id": task_id,
        "os": "ubuntu",
        "step": STEPS,
        "screen_size": [1000, 1000],
        "instruction": "demo task",
        "target_tool_call": tool("left_click", coordinate=[500, 500]),
        "target_text": '{"action": "left_click", "coordinate": [500, 500]}',
        "history": history,
        "image_relpaths": relpaths,
    }


def build_one_group(tmp_path):
    record = synthetic_record(tmp_path)
    return build_corpus(
        [record],
        image_root=tmp_path,
        coordinate_tolerance=2,
        min_age=3,
        seed=7,
    )


def image_paths(messages) -> list[str]:
    return [
        part["path"]
        for message in messages
        for part in message["content"]
        if part.get("type") == "image"
    ]


def test_group_emits_did_arms_with_twin_prompts(tmp_path) -> None:
    rows, b0_rows, manifest = build_one_group(tmp_path)
    by_slot = {row["arm_slot"]: row for row in rows}
    assert sorted(by_slot) == sorted(DESKTOP_ARM_CONTRACT)
    assert len(rows) == 5 and len(b0_rows) == 1
    for row in rows:
        assert row["schema_version"] == SAMPLE_SCHEMA
        contract = DESKTOP_ARM_CONTRACT[row["arm_slot"]]
        assert (
            row["arm_id"], row["role"], row["prompt_format"],
            row["selection_mode"], row["adapter_mode"],
        ) == contract
        assert row["budget"] == 1
        assert len(image_paths(row["messages"])) == 2
        assert row["reference_arm_id"] == "R0"
        assert row["deployment_baseline_arm_id"] == "R0"
    # §8.6:bypass/active 孪生臂除 adapter_mode 外逐字节同 prompt
    assert by_slot["R0"]["messages"] == by_slot["RA"]["messages"]
    assert by_slot["S0"]["messages"] == by_slot["SA"]["messages"]
    assert by_slot["S0"]["messages_sha256"] != by_slot["WA"]["messages_sha256"]
    assert manifest["group_count"] == 1
    assert manifest["counters"]["groups_left_click"] == 1
    assert manifest["stats"]["positive_age"]["min"] >= 3


def test_recent_arm_uses_frozen_deployment_selection(tmp_path) -> None:
    rows, _b0, _manifest = build_one_group(tmp_path)
    by_slot = {row["arm_slot"]: row for row in rows}
    current_path = f"traj-1/obs-{STEPS - 1:03d}.png"
    for slot in ("R0", "RA"):
        row = by_slot[slot]
        # k=1 的冻结 recent 选择器取最后一个事件,其 post 帧与当前截图同帧
        assert row["selected_steps"] == [STEPS - 1]
        assert row["recent_frames_kept"] == 1
        assert image_paths(row["messages"]) == [current_path, current_path]
    for slot in ("S0", "SA", "WA"):
        row = by_slot[slot]
        assert row["recent_frames_kept"] == 0
        assert image_paths(row["messages"])[-1] == current_path


def test_wrong_arm_bookkeeping_and_ages(tmp_path) -> None:
    rows, _b0, _manifest = build_one_group(tmp_path)
    by_slot = {row["arm_slot"]: row for row in rows}
    positive_event = by_slot["SA"]["selected_steps"][0]
    wrong = by_slot["WA"]
    assert wrong["negative_kind"] == "wrong"
    assert wrong["negative_scale"] == 1.0
    assert wrong["donor_episode"] == wrong["episode"]
    assert wrong["distractor_source_step"] == wrong["selected_steps"][0]
    assert wrong["oracle_source_step"] == positive_event
    assert wrong["selected_steps"][0] != positive_event
    assert STEPS - positive_event >= 3
    assert STEPS - wrong["selected_steps"][0] >= 3
    assert wrong["memory_config"]["restored_event_step_ids"] == wrong["selected_steps"]


def test_b0_parity_row_is_isolated_and_imageless(tmp_path) -> None:
    rows, b0_rows, manifest = build_one_group(tmp_path)
    b0 = b0_rows[0]
    assert b0["schema_version"] == B0_SAMPLE_SCHEMA
    assert b0["arm_slot"] == "B0"
    assert b0["role"] == "parity_baseline"
    assert b0["adapter_mode"] == "bypass"
    assert b0["selected_steps"] == []
    assert len(image_paths(b0["messages"])) == 1
    assert b0["target_text"] == rows[0]["target_text"]
    assert b0["pair_group"] == rows[0]["pair_group"]
    assert manifest["b0_sample_count"] == 1
    # B0 决不能混进训练 samples
    assert all(row["schema_version"] == SAMPLE_SCHEMA for row in rows)


def test_rebuild_is_deterministic(tmp_path) -> None:
    first = build_one_group(tmp_path)
    second = build_one_group(tmp_path)
    for left, right in zip(first[:2], second[:2]):
        assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def test_record_without_recurrence_is_rejected(tmp_path) -> None:
    record = synthetic_record(tmp_path, recurrence=False)
    rows, b0_rows, manifest = build_corpus(
        [record],
        image_root=tmp_path,
        coordinate_tolerance=2,
        min_age=3,
        seed=7,
    )
    assert rows == [] and b0_rows == []
    assert manifest["counters"]["rejected_no_recurrence_or_negative"] == 1
    assert manifest["group_count"] == 0
    assert manifest["stats"] == {}


def test_min_age_gate_blocks_recent_masquerade(tmp_path) -> None:
    record = synthetic_record(tmp_path)
    # 把唯一等价旧动作压到 age < min_age:event 6 的 post 帧太新,不得当正例
    for entry in record["history"]:
        if entry["action"].get("x") == 500:
            entry["action"]["x"] = 901
    record["history"][6]["action"] = {"type": "click", "x": 500, "y": 500}
    selected = select_contrast_events(
        record, coordinate_tolerance=2, min_age=3, seed=7
    )
    assert selected is None
