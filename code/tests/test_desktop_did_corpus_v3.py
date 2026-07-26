"""固定预算替换语料 v3 的冻结不变量(2026-07-27 用户确认口径)。

# note (luojiaxuan): 锁五条:
#   1. R = Recent-B 不同帧窗口(s-2..s-1-B);S/W = Recent-(B-1) + 旧帧,
#      替换的是**最老槽位** s-1-B;行内图数恒为 B(+当前图);
#   2. 正/错旧帧 age ≥ B+2(严格老于整个窗口);B=1 退化为真前一帧口径的 v1;
#   3. B=1 分片顺带产零图 parity 行;
#   4. 行过 trainer 契约;pair_group 带 :b 后缀;split 跨 B 一致;
#   5. merge 只并 B∈{1,2,4} 进 samples.jsonl(B=8 只建不训)。
"""

from __future__ import annotations

import io
import json

import pytest
import torch  # noqa: F401
from PIL import Image

import scripts.train_success_sft_lora as trainer
from scripts.build_desktop_did_corpus_v3 import (
    TRAIN_B_VALUES,
    build_b_groups,
    eligible_events_b,
    merge_outputs,
    replaced_slot_event,
)
from scripts.build_desktop_did_corpus_v2 import recent_window
from scripts.build_desktop_hgkv_corpus import B0_SAMPLE_SCHEMA


def tool(action: str, **kwargs):
    return {"name": "computer_use", "arguments": {"action": action, **kwargs}}


def png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def synthetic_record(tmp_path, *, steps: int, match_steps: tuple[int, ...],
                     dp_id="dp-1", task_id="traj-1"):
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


def test_replaced_slot_is_the_oldest_recent_frame() -> None:
    assert replaced_slot_event(16, 1) == 14   # B=1:唯一槽位 = 前一帧
    assert replaced_slot_event(16, 4) == 11
    assert recent_window(16, 4) == [11, 12, 13, 14]
    assert recent_window(16, 3) == [12, 13, 14]  # kept = 去掉最老的 11


def test_eligibility_requires_age_b_plus_2(tmp_path) -> None:
    rec = synthetic_record(tmp_path, steps=20, match_steps=(6,))  # age 15
    for b in (1, 2, 4, 8):
        assert eligible_events_b(rec, b=b, coordinate_tolerance=2, seed=7) is not None
    rec2 = synthetic_record(tmp_path, steps=20, match_steps=(17,),  # age 4
                            dp_id="dp-2", task_id="traj-2")
    assert eligible_events_b(rec2, b=1, coordinate_tolerance=2, seed=7) is not None
    assert eligible_events_b(rec2, b=2, coordinate_tolerance=2, seed=7) is not None
    assert eligible_events_b(rec2, b=4, coordinate_tolerance=2, seed=7) is None


@pytest.mark.parametrize("b", [1, 2, 4])
def test_fixed_budget_arms(tmp_path, b) -> None:
    record = synthetic_record(tmp_path, steps=16, match_steps=(3,))
    rows, parity, stats = build_b_groups(
        [record], image_root=tmp_path, b=b, coordinate_tolerance=2, seed=7,
        emit_parity=(b == 1),
    )
    assert stats["counters"]["groups"] == 1
    by_slot = {row["arm_slot"]: row for row in rows}
    window = recent_window(16, b)
    kept = recent_window(16, b - 1)
    for row in rows:
        assert row["budget"] == b
        assert len(row["selected_steps"]) == b          # 行内图数恒为 B
        assert len(image_paths(row["messages"])) == b + 1
        assert row["kept_recent_steps"] == kept
        assert row["replaced_slot_event"] == replaced_slot_event(16, b)
        assert row["k_replaced"] == 1
        assert row["pair_group"].endswith(f":b{b}")
    assert by_slot["R0"]["selected_steps"] == window
    assert by_slot["R0"]["messages"] == by_slot["RA"]["messages"]
    assert by_slot["S0"]["messages"] == by_slot["SA"]["messages"]
    extra = lambda row: (set(row["selected_steps"]) - set(kept)).pop()
    positive = extra(by_slot["SA"])
    wrong = extra(by_slot["WA"])
    assert positive != wrong
    assert 16 - positive >= b + 2 and 16 - wrong >= b + 2
    assert positive not in window and wrong not in window
    assert by_slot["R0"]["recent_frames_kept"] == b
    assert by_slot["SA"]["recent_frames_kept"] == b - 1
    if b == 1:
        # B=1 = 真前一帧口径的 v1;parity 行零图
        assert by_slot["R0"]["selected_steps"] == [14]
        assert parity and parity[0]["schema_version"] == B0_SAMPLE_SCHEMA
        assert parity[0]["selected_steps"] == []
        assert len(image_paths(parity[0]["messages"])) == 1


def test_rows_pass_trainer_validation_at_b4(tmp_path) -> None:
    record = synthetic_record(tmp_path, steps=16, match_steps=(3,))
    rows, _p, _s = build_b_groups(
        [record], image_root=tmp_path, b=4, coordinate_tolerance=2, seed=7,
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


def test_merge_concats_only_train_budgets(tmp_path) -> None:
    record = synthetic_record(tmp_path, steps=20, match_steps=(3,))
    out = tmp_path / "corpus"
    out.mkdir()
    for b in (1, 2, 4, 8):
        rows, parity, stats = build_b_groups(
            [record], image_root=tmp_path, b=b, coordinate_tolerance=2, seed=7,
            emit_parity=(b == 1),
        )
        with (out / f"samples-b{b}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        if b == 1:
            with (out / "parity_b0.jsonl").open("w") as handle:
                for row in parity:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        (out / f"manifest-b{b}.json").write_text(json.dumps({
            "schema_version": "causalcache.desktop_did_corpus_b.v1",
            "seed": 7, "coordinate_tolerance": 2,
            "per_b": {str(b): {"sample_count": len(rows),
                                "counters": stats["counters"]}},
            "inputs": {},
        }))
    merge_outputs(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["b_values_trained"] == list(TRAIN_B_VALUES)
    assert manifest["b_values_built"] == [1, 2, 4, 8]
    merged = [json.loads(l) for l in (out / "samples.jsonl").open()]
    assert {row["budget"] for row in merged} == {1, 2, 4}   # B=8 不进训练
    assert manifest["sample_count"] == len(merged) == 15
    assert (out / "samples-b8.jsonl").exists()


def test_split_is_consistent_across_b(tmp_path) -> None:
    record = synthetic_record(tmp_path, steps=20, match_steps=(3,))
    splits = set()
    for b in (1, 4, 8):
        rows, _p, _s = build_b_groups(
            [record], image_root=tmp_path, b=b, coordinate_tolerance=2, seed=7,
        )
        splits.update(row["split"] for row in rows)
    assert len(splits) == 1
