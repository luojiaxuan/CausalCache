"""第五条拒绝(证据值已在动作文本里)的双向覆盖。

# note (luojiaxuan): 这个测试要钉死的是**差分**,不是绝对计数 —— 同一份合成轨迹同时
# 造出两个 v3 候选,其中恰好一个的值提前出现在某个已完成步骤的 action 文本里。
# 若第五条写错方向(比如只看证据帧那一步的动作、或者把决策点自己那一步也算进去),
# 两个方向的断言至少有一个会红。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.mine_c_cases_image_only import mine

RECENT_K = 4

# 只在**截图描述**里出现过的值:恢复那张图是唯一拿到它的途径 -> 必须保留
IMAGE_ONLY_VALUE = "8829"
# 既在截图描述里、又被某个已完成步骤的动作文本写出来过 -> 必须被第五条拒掉
TEXT_AVAILABLE_VALUE = "7431"


def _episode() -> dict:
    """12 步合成轨迹;决策点是 index 5..11(range(RECENT_K + 1, 12))。"""
    blank_screen = "This is a screenshot of a plain settings page with a list of rows."
    steps = [
        # 0:image-only 证据帧。值只出现在 description 里,动作文本不提它。
        {
            "description": f"This is a screenshot of a locker panel showing pickup code {IMAGE_ONLY_VALUE} in a card.",
            "low_level_instruction": "Open the locker panel.",
            "info": [[10, 10], [10, 10]],
        },
        # 1:text-available 证据帧。
        {
            "description": f"This is a screenshot of a receipt page showing order reference {TEXT_AVAILABLE_VALUE} at the top.",
            "low_level_instruction": "Scroll down to the receipt.",
            "info": [[11, 11], [11, 11]],
        },
        {"description": blank_screen, "low_level_instruction": "Go back to the home screen.", "info": [[12, 12], [12, 12]]},
        # 3:把 TEXT_AVAILABLE_VALUE 写进已完成步骤的动作文本(不是决策点,d 从 5 开始)。
        {
            "description": blank_screen,
            "low_level_instruction": f"Write down the reference {TEXT_AVAILABLE_VALUE} in the notes app.",
            "info": [[13, 13], [13, 13]],
        },
        {"description": blank_screen, "low_level_instruction": "Open the delivery app.", "info": [[14, 14], [14, 14]]},
        {"description": blank_screen, "low_level_instruction": "Tap the account tab.", "info": [[15, 15], [15, 15]]},
        # 6:image-only 决策点 —— 值来自 step 0 的截图,之前没有任何动作文本提过它。
        {
            "description": blank_screen,
            "low_level_instruction": f'Type "{IMAGE_ONLY_VALUE}" into the pickup code field.',
            "info": [[16, 16], [16, 16]],
        },
        # 7:text-available 决策点 —— 值来自 step 1 的截图,但 step 3 的动作文本已写过。
        {
            "description": blank_screen,
            "low_level_instruction": f'Type "{TEXT_AVAILABLE_VALUE}" into the reference field.',
            "info": [[17, 17], [17, 17]],
        },
        {"description": blank_screen, "low_level_instruction": "Tap the submit button.", "info": [[18, 18], [18, 18]]},
        {"description": blank_screen, "low_level_instruction": "Tap the confirm button.", "info": [[19, 19], [19, 19]]},
        {"description": blank_screen, "low_level_instruction": "Scroll to the bottom.", "info": [[20, 20], [20, 20]]},
        {"description": blank_screen, "low_level_instruction": "Task completed.", "info": [[21, 21], [21, 21]]},
    ]
    for index, step in enumerate(steps):
        step["step"] = index
    return {
        "episode_id": "synthetic",
        "task_info": {"task": "Collect the parcel and confirm the delivery."},
        "steps": steps,
    }


@pytest.fixture()
def mined(tmp_path: Path) -> dict:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "EP1.json").write_text(
        json.dumps(_episode(), ensure_ascii=False), encoding="utf-8"
    )
    return mine([{"source_id": "EP1"}], annotations, RECENT_K)


def test_both_values_are_v3_candidates(mined: dict) -> None:
    """前四条对两个值都放行 —— 差分才有意义。"""
    assert mined["trajectories_scanned"] == 1
    assert mined["v3_candidate_decision_points"] == 2
    assert mined["v3_events"] == 2


def test_value_in_earlier_action_text_is_rejected(mined: dict) -> None:
    """方向一:值已被某个已完成步骤的动作文本写出来 -> 恢复截图冗余 -> 拒。"""
    assert mined["counters"]["reject_value_in_action_text"] == 1
    rejected = {event["value"] for event in mined["all_text_available_events"]}
    assert rejected == {TEXT_AVAILABLE_VALUE}
    (event,) = mined["all_text_available_events"]
    assert event["evidence_step"] == 1
    assert event["first_use_step"] == 7
    # 命中的是 step 3 的动作文本,不是证据帧那一步、也不是决策点自己那一步。
    assert event["action_text_steps"] == [3]


def test_value_only_in_screenshot_is_kept(mined: dict) -> None:
    """方向二:值只在截图描述里出现过 -> 只有恢复那张图才能拿到 -> 保留。"""
    assert mined["image_only_decision_points"] == 1
    kept = {event["value"] for event in mined["all_events"]}
    assert kept == {IMAGE_ONLY_VALUE}
    (event,) = mined["all_events"]
    assert event["episode"] == "EP1"
    assert event["evidence_step"] == 0
    assert event["first_use_step"] == 6
    assert event["age_at_first_use"] == 6
    assert "action_text_steps" not in event


def test_repeated_use_becomes_text_available(tmp_path: Path) -> None:
    """同一个值第二次被用到时必然已经文本可得 —— 只有首次使用能留在 image-only 侧。

    # note (luojiaxuan): 这是第五条最重要的结构性后果:v3 里一个事件平均覆盖 2.87 个
    # 决策点,而这些重复使用**全部**是文本冗余的。所以 image-only 决策点的上界是
    # v3 的事件数,不是 v3 的候选决策点数。
    """
    episode = _episode()
    # step 8 再用一次 IMAGE_ONLY_VALUE;此时 step 6 的动作文本已经写过它。
    episode["steps"][8]["low_level_instruction"] = (
        f'Type "{IMAGE_ONLY_VALUE}" into the confirmation field.'
    )
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "EP1.json").write_text(
        json.dumps(episode, ensure_ascii=False), encoding="utf-8"
    )
    mined = mine([{"source_id": "EP1"}], annotations, RECENT_K)

    assert mined["v3_candidate_decision_points"] == 3
    assert mined["image_only_decision_points"] == 1
    assert mined["counters"]["reject_value_in_action_text"] == 2
    # 首次使用仍在 image-only 侧,重复使用被划到 text-available 侧
    (kept,) = mined["all_events"]
    assert kept["value"] == IMAGE_ONLY_VALUE
    assert kept["first_use_step"] == 6 and kept["uses"] == 1
    repeated = [e for e in mined["all_text_available_events"] if e["value"] == IMAGE_ONLY_VALUE]
    assert [e["first_use_step"] for e in repeated] == [8]


def test_first_four_rejections_match_v3(tmp_path: Path) -> None:
    """前四条与 v3 逐条同序 —— 差分能归因到第五条,而不是抽取器漂移。"""
    from scripts import mine_c_cases_v3

    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "EP1.json").write_text(
        json.dumps(_episode(), ensure_ascii=False), encoding="utf-8"
    )
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selected": [{"source_id": "EP1"}]}), encoding="utf-8")
    output = tmp_path / "v3.json"
    argv = sys.argv
    sys.argv = [
        "mine_c_cases_v3",
        "--selection", str(selection),
        "--annotations", str(annotations),
        "--output", str(output),
        "--recent-k", str(RECENT_K),
    ]
    try:
        mine_c_cases_v3.main()
    finally:
        sys.argv = argv
    v3 = json.loads(output.read_text(encoding="utf-8"))
    ours = mine([{"source_id": "EP1"}], annotations, RECENT_K)

    for name in (
        "decision_points",
        "reject_value_in_goal",
        "reject_visible_now",
        "reject_inside_recent_k",
        "reject_never_visible",
        "candidate_decision_points",
    ):
        assert v3["counters"].get(name, 0) == ours["counters"].get(name, 0), name
    assert v3["events"] == ours["v3_events"]
