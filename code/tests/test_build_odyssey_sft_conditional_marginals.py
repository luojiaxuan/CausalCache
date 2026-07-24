"""Conditional-marginal label rendering for the GUI-Odyssey SFT renderer.

# note (luojiaxuan): 协议「Conditional marginal 标签生成(近线性)」——用 hg-s100
# singleton 分做 shortlist/anchor/beam,为每个决策态发 cond_base / cond_edge1 /
# cond_edge2 三族样本,供下游按 restored_set_key join 出 Δ(j|S)=U(S∪{j})−U(S)。
# 本测试用合成分数与合成轨迹验证:发射条数对公式、restored_set_key 正确与排序、
# anchor de-dup、shortlist top-K、--no-second-layer 只砍第二层、|S|=2 edge2 的
# m+1 图块结构、以及渲染确定性。不触真实数据与模型。
"""

from __future__ import annotations

import io
import json

import pytest

from scripts.build_odyssey_sft_dataset import (
    conditional_marginal_plan,
    render_trajectory,
    restored_set_key,
)

SOURCE_ID = "synthetic-odyssey-0001"
OBSERVATIONS = 12  # observations 0..11, events 1..11, terminal annotation step 11
PAIR_GROUP = f"{SOURCE_ID}:12"

# note (luojiaxuan): decision=12 时候选 = 事件 1..10;取其中 8 个赋两两不同的
# singleton logprob,使 ranked 顺序完全确定:
#   logprob desc → [5, 8, 3, 7, 9, 4, 6, 10];shortlist(K=6) = {3,4,5,7,8,9}。
# oracle-top1=5,Recent-1=10,diverse=ranked[4]=9,random=id_sorted[seed%8]。
SCORES = {5: -0.10, 8: -0.20, 3: -0.30, 7: -0.40, 9: -0.50, 4: -0.60, 6: -0.70, 10: -0.80}
EXPECTED_SHORTLIST = {3, 4, 5, 7, 8, 9}


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


def _user_content(sample):
    return next(
        message["content"]
        for message in sample["messages"]
        if message["role"] == "user"
    )


def _image_parts(sample):
    return [part for part in _user_content(sample) if part["type"] == "image"]


# ---------------------------------------------------------------------------
# Pure-plan tests (no image / model I/O)
# ---------------------------------------------------------------------------


def test_restored_set_key_sorts_numerically():
    assert restored_set_key([10, 3]) == "3-10"
    assert restored_set_key((5, 8)) == "5-8"
    assert restored_set_key([7]) == "7"
    assert restored_set_key([]) == ""


def test_plan_emission_counts_match_formula():
    rows = conditional_marginal_plan(
        PAIR_GROUP, SCORES, shortlist_k=6, num_anchors=4, second_layer=True
    )
    by_variant: dict[str, list] = {}
    for row in rows:
        by_variant.setdefault(row[0], []).append(row)

    anchor_bases = [r for r in by_variant["cond_base"] if len(r[1]) == 1]
    pair_bases = [r for r in by_variant["cond_base"] if len(r[1]) == 2]
    anchors = sorted(r[1][0] for r in anchor_bases)
    # oracle-top1=5, Recent-1=10, diverse=9, random=id_sorted[seed%8] all distinct.
    assert anchors == [5, 6, 9, 10]
    assert 5 in anchors and 10 in anchors  # oracle + Recent
    shortlist = EXPECTED_SHORTLIST

    # edge1 = sum over anchors of |shortlist \ {i}| = 5+6+6+5 = 22.
    expected_edge1 = sum(len(shortlist - {i}) for i in anchors)
    assert expected_edge1 == 22
    assert len(by_variant["cond_edge1"]) == expected_edge1

    # beam-2 by singleton logprob sum -> {5,8} and {3,5}; both fully in shortlist.
    beam_pairs = sorted(r[1] for r in pair_bases)
    assert beam_pairs == [(3, 5), (5, 8)]
    expected_edge2 = sum(len(shortlist - set(p)) for p in beam_pairs)
    assert expected_edge2 == 8
    assert len(by_variant["cond_edge2"]) == expected_edge2

    assert len(anchor_bases) == 4
    assert len(pair_bases) == 2
    assert len(rows) == 4 + 2 + 22 + 8 == 36


def test_plan_restored_and_anchor_semantics():
    rows = conditional_marginal_plan(PAIR_GROUP, SCORES)
    for variant, restored, anchor, candidate in rows:
        assert tuple(sorted(restored)) == restored  # restored ids sorted ascending
        assert tuple(sorted(anchor)) == anchor
        if variant == "cond_base":
            assert candidate is None
            assert set(anchor) == set(restored)
        elif variant == "cond_edge1":
            assert len(anchor) == 1 and candidate is not None
            assert set(restored) == set(anchor) | {candidate}
            assert candidate in EXPECTED_SHORTLIST and candidate != anchor[0]
        elif variant == "cond_edge2":
            assert len(anchor) == 2 and candidate is not None
            assert set(restored) == set(anchor) | {candidate}
            assert candidate in EXPECTED_SHORTLIST and candidate not in anchor
        else:  # pragma: no cover - guards against a new variant slipping in
            raise AssertionError(f"unexpected variant {variant}")


def test_plan_shortlist_top_k_respected():
    # note (luojiaxuan): shortlist 只应含 top-K singleton;K=6 时候选 6(logprob 倒数
    # 第二)与 10(最低)是否入 shortlist 只取决于排名,不取决于 id/recency。
    edge1_candidates = {
        candidate
        for variant, _restored, _anchor, candidate in conditional_marginal_plan(
            PAIR_GROUP, SCORES, shortlist_k=6
        )
        if variant == "cond_edge1"
    }
    assert edge1_candidates == EXPECTED_SHORTLIST
    assert 6 not in edge1_candidates and 10 not in edge1_candidates  # rank 7,8

    small = {
        candidate
        for variant, _restored, _anchor, candidate in conditional_marginal_plan(
            PAIR_GROUP, SCORES, shortlist_k=3
        )
        if variant == "cond_edge1"
    }
    assert small == {5, 8, 3}  # ranked[:3]


def test_plan_anchor_dedup_when_oracle_equals_recent():
    # note (luojiaxuan): 令最大 id(10)同时为最高 singleton logprob → oracle-top1
    # 与 Recent-1 重合;de-dup 后 distinct anchor 数应 < num_anchors,且无重复 base。
    scores = {10: -0.10, 8: -0.20, 3: -0.30, 7: -0.40, 9: -0.50, 4: -0.60, 6: -0.70, 5: -0.80}
    rows = conditional_marginal_plan(PAIR_GROUP, scores, num_anchors=4)
    anchor_ids = [r[1][0] for r in rows if r[0] == "cond_base" and len(r[1]) == 1]
    assert len(anchor_ids) == len(set(anchor_ids))  # no duplicate base row
    assert len(set(anchor_ids)) < 4  # coincidence collapsed at least one kind
    assert anchor_ids.count(10) == 1  # oracle == Recent emitted once


def test_plan_no_second_layer_drops_edge2_only():
    full = conditional_marginal_plan(PAIR_GROUP, SCORES, second_layer=True)
    first = conditional_marginal_plan(PAIR_GROUP, SCORES, second_layer=False)
    # cond_edge1 rows byte-identical; anchor (|S|=1) base rows byte-identical.
    assert [r for r in full if r[0] == "cond_edge1"] == [
        r for r in first if r[0] == "cond_edge1"
    ]
    assert [r for r in full if r[0] == "cond_base" and len(r[1]) == 1] == [
        r for r in first if r[0] == "cond_base" and len(r[1]) == 1
    ]
    # only the second layer (edge2 + its |S|=2 beam bases) is removed.
    assert not any(r[0] == "cond_edge2" for r in first)
    assert not any(r[0] == "cond_base" and len(r[1]) == 2 for r in first)


def test_plan_is_deterministic():
    first = conditional_marginal_plan(PAIR_GROUP, SCORES)
    second = conditional_marginal_plan(PAIR_GROUP, SCORES)
    assert first == second


def test_plan_requires_two_candidates():
    assert conditional_marginal_plan(PAIR_GROUP, {5: -0.1}) == []
    assert conditional_marginal_plan(PAIR_GROUP, {}) == []


# ---------------------------------------------------------------------------
# End-to-end render tests (real PIL image blocks, no model)
# ---------------------------------------------------------------------------


def test_render_emits_conditional_family_and_passthrough(synthetic_setup):
    row, annotations_root, output_root = synthetic_setup
    samples = _render(
        row,
        annotations_root,
        output_root,
        decisions=[12],
        conditional_marginals=True,
        conditional_scores={PAIR_GROUP: SCORES},
    )
    assert len(samples) == 36
    for sample in samples:
        assert sample["pair_group"] == PAIR_GROUP
        assert sample["schema_version"] == "causalcache.odyssey_sft_sample.v1"
        assert sample["memory_config"]["mode"] == "conditional_marginal"
        restored = sample["memory_config"]["restored_event_step_ids"]
        assert restored == sorted(restored)
        assert sample["memory_config"]["budget"] == len(restored)
        assert sample["restored_set_key"] == restored_set_key(restored)
        assert "singleton_event_step_id" not in sample
        # image blocks = |restored| history + 1 current, all separated by text.
        assert len(_image_parts(sample)) == len(restored) + 1

    variants: dict[str, int] = {}
    for sample in samples:
        variants[sample["variant"]] = variants.get(sample["variant"], 0) + 1
    assert variants == {"cond_base": 6, "cond_edge1": 22, "cond_edge2": 8}

    # passthrough Δ(j|S) join fields.
    for sample in samples:
        anchor = sample["conditional_anchor_set"]
        candidate = sample["conditional_candidate"]
        if sample["variant"] == "cond_base":
            assert candidate is None
            assert anchor == sample["memory_config"]["restored_event_step_ids"]
        else:
            assert candidate is not None
            assert set(anchor) | {candidate} == set(
                sample["memory_config"]["restored_event_step_ids"]
            )


def test_render_edge2_has_m_plus_one_separated_image_blocks(synthetic_setup):
    row, annotations_root, output_root = synthetic_setup
    samples = _render(
        row,
        annotations_root,
        output_root,
        decisions=[12],
        conditional_marginals=True,
        conditional_scores={PAIR_GROUP: SCORES},
    )
    edge2 = [s for s in samples if s["variant"] == "cond_edge2"]
    assert edge2
    sample = edge2[0]
    restored = sample["memory_config"]["restored_event_step_ids"]
    assert len(restored) == 3  # |S|=2 anchor + 1 candidate
    content = _user_content(sample)
    image_indices = [i for i, part in enumerate(content) if part["type"] == "image"]
    # m+1 = 4 image blocks so build_history_token_mask sees K=3 restored + current.
    assert len(image_indices) == len(restored) + 1 == 4
    # every image block is separated by at least one text block (no adjacency).
    for earlier, later in zip(image_indices, image_indices[1:]):
        assert later - earlier >= 2
        assert any(
            content[between]["type"] == "text"
            for between in range(earlier + 1, later)
        )
    # restored history images precede the current observation, in sorted order.
    restored_paths = [
        content[i]["path"] for i in image_indices[:-1]
    ]
    assert restored_paths == [
        f"images/{SOURCE_ID}/observation-{sid:03d}.png" for sid in restored
    ]


def test_render_no_second_layer_drops_edge2(synthetic_setup):
    row, annotations_root, output_root = synthetic_setup
    samples = _render(
        row,
        annotations_root,
        output_root,
        decisions=[12],
        conditional_marginals=True,
        conditional_scores={PAIR_GROUP: SCORES},
        second_layer=False,
    )
    variants = {s["variant"] for s in samples}
    assert "cond_edge2" not in variants
    assert variants == {"cond_base", "cond_edge1"}
    assert len(samples) == 4 + 22  # anchor bases + edge1 only


def test_render_skips_states_without_singleton_scores(synthetic_setup):
    row, annotations_root, output_root = synthetic_setup
    # note (luojiaxuan): 只给 decision 12 的分;decision 4 无分应整态跳过。
    samples = _render(
        row,
        annotations_root,
        output_root,
        decisions=[4, 12],
        conditional_marginals=True,
        conditional_scores={PAIR_GROUP: SCORES},
    )
    assert {s["pair_group"] for s in samples} == {PAIR_GROUP}


def test_render_conditional_is_deterministic(synthetic_setup):
    row, annotations_root, output_root = synthetic_setup
    first = _render(
        row,
        annotations_root,
        output_root,
        decisions=[12],
        conditional_marginals=True,
        conditional_scores={PAIR_GROUP: SCORES},
    )
    second = _render(
        row,
        annotations_root,
        output_root,
        decisions=[12],
        conditional_marginals=True,
        conditional_scores={PAIR_GROUP: SCORES},
    )
    dump = lambda rows: [json.dumps(r, sort_keys=True) for r in rows]
    assert dump(first) == dump(second)


def test_render_selected_sets_deduplicates_shared_coalitions(synthetic_setup):
    row, annotations_root, output_root = synthetic_setup
    candidates = tuple(range(1, 11))
    plan = {
        PAIR_GROUP: [
            {
                "budget": 2,
                "candidate_event_step_ids": candidates,
                "method": "recent",
                "selected_event_step_ids": (9, 10),
            },
            {
                "budget": 2,
                "candidate_event_step_ids": candidates,
                "method": "hgkv_set_conditioned",
                "selected_event_step_ids": (9, 10),
            },
            {
                "budget": 4,
                "candidate_event_step_ids": candidates,
                "method": "random",
                "selected_event_step_ids": (1, 3, 5, 7),
            },
        ]
    }

    samples = _render(
        row,
        annotations_root,
        output_root,
        decisions=[12],
        selected_set_plan=plan,
    )

    assert len(samples) == 2
    by_key = {sample["restored_set_key"]: sample for sample in samples}
    shared = by_key["9-10"]
    assert shared["selected_set_budgets"] == [2]
    assert shared["selected_set_methods"] == [
        "hgkv_set_conditioned",
        "recent",
    ]
    assert shared["memory_config"] == {
        "budget": 2,
        "mode": "selected_set_gate",
        "restored_event_step_ids": [9, 10],
    }
    assert len(_image_parts(shared)) == 3
