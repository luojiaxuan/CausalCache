"""Selector singleton-label rendering for the GUI-Odyssey SFT renderer.

# note (luojiaxuan): 合同第 13 步——对每个候选 ≥2 的决策点发 b0 + 逐候选单图
# 恢复(variant="singleton" + singleton_event_step_id),同 pair_group 自洽;
# 本测试用合成轨迹验证条数、restored ids、pair_group 完整性、cap 行为与
# train/heldout split 过滤,不触真实数据与模型。
"""

from __future__ import annotations

import io
import json

import pytest

from scripts.build_odyssey_sft_dataset import (
    load_heldout_episodes,
    render_trajectory,
    split_allows,
)

SOURCE_ID = "synthetic-odyssey-0001"
OBSERVATIONS = 12  # observations 0..11, events 1..11, terminal annotation step 11


def _png_bytes(index: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 64), (index * 9 % 256, 64, 128)).save(buffer, format="PNG")
    return buffer.getvalue()


def _event_payload(step_id: int) -> dict[str, object]:
    return {
        "event_step_id": step_id,
        "low_fidelity_summary": {
            "step_id": step_id,
            "action_type": "click",
            "action_argument": "coordinate_bin:x5_y5",
            "foreground_app": "unknown",
            "screen_text_added": [],
            "screen_text_removed": [],
            "screen_change": "low",
            "executor_result": "unknown",
        },
    }


@pytest.fixture()
def synthetic_setup(tmp_path):
    annotations_root = tmp_path / "annotations"
    annotations_root.mkdir()
    steps = [
        {"step": index, "action": "CLICK", "info": [[500.0, 600.0]]}
        for index in range(OBSERVATIONS - 1)
    ]
    steps.append({"step": OBSERVATIONS - 1, "action": "COMPLETE", "info": None})
    (annotations_root / f"{SOURCE_ID}.json").write_text(
        json.dumps(
            {
                "device_info": {"device_resolution": [1080, 2400]},
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )
    row = {
        "source_id": SOURCE_ID,
        "task_instruction": "Open the settings app and enable dark mode.",
        "history_events_json": json.dumps(
            [_event_payload(step_id) for step_id in range(1, OBSERVATIONS)]
        ),
        "images": [_png_bytes(index) for index in range(OBSERVATIONS)],
        "ocr_records_json": json.dumps([]),
        "raw_metadata": json.dumps({}),
    }
    output_root = tmp_path / "output"
    return row, annotations_root, output_root


def _render(row, annotations_root, output_root, *, decisions, **kwargs):
    return render_trajectory(
        row,
        annotations_root=annotations_root,
        decisions=decisions,
        output_root=output_root,
        shared_early_decisions=2,
        contrast_variants=False,
        donor_paths=None,
        **kwargs,
    )


def _image_parts(sample):
    return [
        part
        for message in sample["messages"]
        for part in message["content"]
        if part["type"] == "image"
    ]


def test_singleton_emission_counts_and_ids(synthetic_setup):
    row, annotations_root, output_root = synthetic_setup
    # note (luojiaxuan): d=3 只有 1 个候选(<2,整组跳过);d=4 两候选;
    # d=12 十候选,cap 8 只保最近 (3..10)。
    samples = _render(
        row,
        annotations_root,
        output_root,
        decisions=[3, 4, 12],
        selector_singletons=True,
        singleton_max_candidates=8,
    )
    assert len(samples) == 12
    by_group: dict[str, list[dict]] = {}
    for sample in samples:
        by_group.setdefault(sample["pair_group"], []).append(sample)
    assert set(by_group) == {f"{SOURCE_ID}:4", f"{SOURCE_ID}:12"}
    assert len(by_group[f"{SOURCE_ID}:4"]) == 3
    assert len(by_group[f"{SOURCE_ID}:12"]) == 9
    for group_key, group in by_group.items():
        b0 = [sample for sample in group if sample["variant"] == "b0"]
        singles = [sample for sample in group if sample["variant"] == "singleton"]
        assert len(b0) == 1
        assert len(b0) + len(singles) == len(group)
        assert b0[0]["singleton_event_step_id"] is None
        assert b0[0]["memory_config"]["restored_event_step_ids"] == []
        assert b0[0]["memory_config"]["budget"] == 0
        assert len(_image_parts(b0[0])) == 1  # current observation only
        targets = {sample["target_text"] for sample in group}
        assert len(targets) == 1 and next(iter(targets))
        for sample in group:
            assert sample["schema_version"] == "causalcache.odyssey_sft_sample.v1"
            assert sample["memory_config"]["mode"] == "selector_singleton"
        for sample in singles:
            event_id = sample["singleton_event_step_id"]
            assert sample["memory_config"]["restored_event_step_ids"] == [event_id]
            assert sample["memory_config"]["budget"] == 1
            parts = _image_parts(sample)
            assert len(parts) == 2  # restored history image + current observation
            assert parts[0]["path"] == (
                f"images/{SOURCE_ID}/observation-{event_id:03d}.png"
            )
    assert [
        sample["singleton_event_step_id"]
        for sample in by_group[f"{SOURCE_ID}:4"]
        if sample["variant"] == "singleton"
    ] == [1, 2]
    assert [
        sample["singleton_event_step_id"]
        for sample in by_group[f"{SOURCE_ID}:12"]
        if sample["variant"] == "singleton"
    ] == [3, 4, 5, 6, 7, 8, 9, 10]


def test_singleton_cap_is_tunable(synthetic_setup):
    row, annotations_root, output_root = synthetic_setup
    samples = _render(
        row,
        annotations_root,
        output_root,
        decisions=[12],
        selector_singletons=True,
        singleton_max_candidates=3,
    )
    assert [sample["variant"] for sample in samples] == [
        "b0",
        "singleton",
        "singleton",
        "singleton",
    ]
    assert [
        sample["singleton_event_step_id"]
        for sample in samples
        if sample["variant"] == "singleton"
    ] == [8, 9, 10]


def test_default_mode_unchanged_by_selector_fields(synthetic_setup):
    row, annotations_root, output_root = synthetic_setup
    samples = _render(row, annotations_root, output_root, decisions=[4, 12])
    assert samples
    for sample in samples:
        assert "singleton_event_step_id" not in sample
        assert sample["variant"] == "correct"
        assert sample["memory_config"]["mode"] != "selector_singleton"


def test_split_allows_and_heldout_file(tmp_path):
    heldout_path = tmp_path / "heldout.txt"
    heldout_path.write_text("traj-a\ntraj-b\n", encoding="utf-8")
    heldout = load_heldout_episodes(heldout_path)
    assert heldout == {"traj-a", "traj-b"}
    assert split_allows("traj-a", split="all", heldout_episodes=None)
    assert split_allows("traj-a", split="heldout", heldout_episodes=heldout)
    assert not split_allows("traj-a", split="train", heldout_episodes=heldout)
    assert split_allows("traj-c", split="train", heldout_episodes=heldout)
    assert not split_allows("traj-c", split="heldout", heldout_episodes=heldout)
    with pytest.raises(ValueError):
        split_allows("traj-a", split="train", heldout_episodes=None)
