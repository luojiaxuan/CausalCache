"""r-条件化 Desktop DiD 语料 v2 的冻结不变量。

# note (luojiaxuan): 锁 2026-07-27 冻结口径:
#   1. Recent-r = 最近 r 张不同帧(事件 s-2..s-1-r),不含与当前截图同帧的 s-1;
#   2. next-recent = 事件 s-2-r;正/错旧帧 age ≥ r+3(r=0 退化为 v1 min_age=3);
#   3. 四臂共享 C_r,行内 matched budget(r+1 张恢复图 + 当前图);
#   4. Base(C_r)进 sidecar,r=0 的 Base 即 B0 parity 行;
#   5. 行结构过 trainer 契约(schema 不变),pair_group 带 r 后缀,split 跨 r 一致。
"""

from __future__ import annotations

import io

import pytest
import torch  # noqa: F401  (trainer 契约校验依赖)
from PIL import Image

import scripts.train_success_sft_lora as trainer
from scripts.build_desktop_did_corpus_v2 import (
    BASE_SAMPLE_SCHEMA,
    build_r_groups,
    eligible_events,
    next_recent_event,
    recent_window,
)
from scripts.build_desktop_hgkv_corpus import B0_SAMPLE_SCHEMA, SAMPLE_SCHEMA


def tool(action: str, **kwargs):
    return {"name": "computer_use", "arguments": {"action": action, **kwargs}}


def png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def synthetic_record(tmp_path, *, steps: int, match_steps: tuple[int, ...],
                     dp_id="dp-1", task_id="traj-1"):
    """steps 步轨迹;match_steps 处的历史动作与 target 完整等价。"""
    (tmp_path / task_id).mkdir(exist_ok=True)
    relpaths = []
    for index in range(steps):
        relpath = f"{task_id}/obs-{index:03d}.png"
        (tmp_path / relpath).write_bytes(png_bytes((index * 17 % 255, 40, 70)))
        relpaths.append(relpath)
    history = []
    for step_id in range(1, steps):
        if step_id in match_steps:
            action = {"type": "click", "x": 500, "y": 500}
        elif step_id % 2:
            action = {"type": "click", "x": 30 + step_id, "y": 40}
        else:
            action = {"type": "press", "key": "tab"}
        history.append({
            "step_id": step_id,
            "action": action,
            "osworld_action": f"pyautogui.{action['type']}()",
        })
    return {
        "dp_id": dp_id,
        "task_id": task_id,
        "os": "ubuntu",
        "step": steps,
        "screen_size": [1000, 1000],
        "instruction": "demo task",
        "target_tool_call": tool("left_click", coordinate=[500, 500]),
        "target_text": '{"action": "left_click", "coordinate": [500, 500]}',
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


def test_recent_window_is_distinct_frames_not_the_duplicate() -> None:
    # s=12:窗口不含事件 s-1=11(其 post 与当前截图同帧)
    assert recent_window(12, 0) == []
    assert recent_window(12, 1) == [10]
    assert recent_window(12, 4) == [7, 8, 9, 10]
    assert next_recent_event(12, 0) == 10
    assert next_recent_event(12, 2) == 8


def test_eligibility_requires_age_r_plus_3() -> None:
    # 20 步轨迹,唯一等价旧动作在 step 6(事件 5,age=15)→ r=8(需 age≥11)合格
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        rec = synthetic_record(Path(tmp), steps=20, match_steps=(6,))
        for r in (0, 1, 2, 4, 8):
            selected = eligible_events(rec, r=r, coordinate_tolerance=2, seed=7)
            assert selected is not None, f"r={r} should be eligible"
            positive, _, wrong = selected
            assert 20 - positive >= r + 3
            assert 20 - wrong >= r + 3
        # 等价旧动作太新(step 18,事件 17,age=3)→ 只有 r=0 合格
        rec2 = synthetic_record(Path(tmp), steps=20, match_steps=(18,),
                                dp_id="dp-2", task_id="traj-2")
        assert eligible_events(rec2, r=0, coordinate_tolerance=2, seed=7) is not None
        assert eligible_events(rec2, r=1, coordinate_tolerance=2, seed=7) is None


@pytest.mark.parametrize("r", [0, 1, 4])
def test_group_arms_share_cr_and_match_budget(tmp_path, r) -> None:
    record = synthetic_record(tmp_path, steps=16, match_steps=(3,))
    rows, base_rows, stats = build_r_groups(
        [record], image_root=tmp_path, r=r, coordinate_tolerance=2, seed=7,
    )
    assert stats["counters"]["groups"] == 1
    by_slot = {row["arm_slot"]: row for row in rows}
    assert sorted(by_slot) == ["R0", "RA", "S0", "SA", "WA"]
    window = recent_window(16, r)
    for row in rows:
        assert row["budget"] == r + 1
        assert row["recent_r"] == r
        assert row["base_steps"] == window
        assert len(image_paths(row["messages"])) == r + 2
        assert set(window) <= set(row["selected_steps"])
        assert row["pair_group"].endswith(f":r{r}")
    # 孪生臂同 prompt;R 臂新增 = next-recent;S/W 新增互异且不在窗口
    assert by_slot["R0"]["messages"] == by_slot["RA"]["messages"]
    assert by_slot["S0"]["messages"] == by_slot["SA"]["messages"]
    extra = lambda row: (set(row["selected_steps"]) - set(window)).pop()
    assert extra(by_slot["R0"]) == next_recent_event(16, r)
    assert extra(by_slot["SA"]) != extra(by_slot["WA"])
    assert 16 - extra(by_slot["SA"]) >= r + 3
    # Base sidecar:恰为 C_r,r=0 时是 B0 parity 行
    base = base_rows[0]
    assert base["selected_steps"] == window
    assert base["budget"] == r
    assert len(image_paths(base["messages"])) == r + 1
    if r == 0:
        assert base["schema_version"] == B0_SAMPLE_SCHEMA
        assert base["arm_slot"] == "B0"
    else:
        assert base["schema_version"] == BASE_SAMPLE_SCHEMA
        assert base["arm_slot"] == "BASE"


def test_rows_pass_trainer_validation_at_r4(tmp_path) -> None:
    record = synthetic_record(tmp_path, steps=16, match_steps=(3,))
    rows, _base, _stats = build_r_groups(
        [record], image_root=tmp_path, r=4, coordinate_tolerance=2, seed=7,
    )
    normalized = [{**row, "split": "train"} for row in rows]
    for index, row in enumerate(normalized):
        trainer.validate_sparse_sample(row, index=index)
    group = {row["arm_slot"]: index for index, row in enumerate(normalized)}
    assert trainer.validate_sparse_group(
        normalized, pair_group=normalized[0]["pair_group"], group=group
    ) == "R0"
    units, _ = trainer.build_sparse_history_units(
        normalized, objective_kind=trainer.SPARSE_OBJECTIVE_DID_RA_AWARE
    )
    assert len(units) == 1


def test_split_is_consistent_across_r(tmp_path) -> None:
    record = synthetic_record(tmp_path, steps=16, match_steps=(3,))
    splits = set()
    for r in (0, 2, 8):
        rows, _b, _s = build_r_groups(
            [record], image_root=tmp_path, r=r, coordinate_tolerance=2, seed=7,
        )
        splits.update(row["split"] for row in rows)
    assert len(splits) == 1


def test_v1_schema_id_is_preserved(tmp_path) -> None:
    record = synthetic_record(tmp_path, steps=16, match_steps=(3,))
    rows, _b, _s = build_r_groups(
        [record], image_root=tmp_path, r=2, coordinate_tolerance=2, seed=7,
    )
    assert all(row["schema_version"] == SAMPLE_SCHEMA for row in rows)
    assert all(
        row["source"]["recent_definition"] == "distinct_frames_v2" for row in rows
    )
