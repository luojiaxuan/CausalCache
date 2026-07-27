"""Stage-1 cheap 特征的基本不变量。"""

from __future__ import annotations

from causalcache.agentnet_actions import parse_agentnet_step
from causalcache.selector_v4_features import FEATURE_NAMES, candidate_features


def build_record(steps=12, match_steps=(3, 7)):
    screen = (1000, 1000)
    history = []
    for step_id in range(1, steps):
        if step_id in match_steps:
            raw = "pyautogui.click(x=0.5005, y=0.5005)"
        elif step_id % 2:
            raw = f"pyautogui.click(x={round(0.03 + 0.001 * step_id, 6)}, y=0.04)"
        else:
            raw = "pyautogui.press('tab')"
        parsed = parse_agentnet_step(raw, screen_size=screen)
        history.append({
            "step_id": step_id,
            "action": parsed.history_action,
            "osworld_action": parsed.osworld_action,
            "raw_code": raw,
        })
    return {
        "dp_id": "dp-1",
        "task_id": "traj-1",
        "step": steps,
        "screen_size": list(screen),
        "target_tool_call": {
            "name": "computer_use",
            "arguments": {"action": "left_click", "coordinate": [500, 500]},
        },
        "history": history,
        "image_relpaths": [f"t/o{i:03d}.png" for i in range(steps)],
    }


def test_feature_vector_shape_and_witness_flags() -> None:
    record = build_record()
    pool = list(range(1, 11))  # 事件 1..s-2
    by_name = dict(zip(FEATURE_NAMES, candidate_features(
        record, candidate_pool=pool, duplicates={}, event=2,
    )))
    # 事件 2 = 匹配动作步 3 的前状态 → witness
    assert by_name["witness_match"] == 1.0
    assert by_name["witness_count"] == 2.0  # 事件 2 与 6
    assert by_name["age"] == 10.0
    assert by_name["coord_distance"] >= 0.0 and by_name["coord_distance"] < 2.0
    # 非 witness 事件
    by_name4 = dict(zip(FEATURE_NAMES, candidate_features(
        record, candidate_pool=pool, duplicates={}, event=4,
    )))
    assert by_name4["witness_match"] == 0.0
    assert by_name4["witness_rank"] == -1.0
    assert by_name4["coord_distance"] == -1.0
    # 新近度:事件 10(=s-2)在所有窗口内
    by_name10 = dict(zip(FEATURE_NAMES, candidate_features(
        record, candidate_pool=pool, duplicates={}, event=10,
    )))
    assert by_name10["in_recent_1"] == 1.0
    assert by_name10["recency_rank"] == 0.0
    assert by_name10["age"] == 2.0
    # witness 新近度名次:事件 6 比 2 新
    by_name6 = dict(zip(FEATURE_NAMES, candidate_features(
        record, candidate_pool=pool, duplicates={}, event=6,
    )))
    assert by_name6["witness_rank"] == 0.0
    assert dict(zip(FEATURE_NAMES, candidate_features(
        record, candidate_pool=pool, duplicates={}, event=2,
    )))["witness_rank"] == 1.0
    # dedup:两个更早副本指向事件 4
    by_dup = dict(zip(FEATURE_NAMES, candidate_features(
        record, candidate_pool=pool, duplicates={"1": 4, "3": 4}, event=4,
    )))
    assert by_dup["dedup_multiplicity"] == 3.0
