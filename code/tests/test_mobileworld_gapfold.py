"""Mobile gap-fold 渲染器与部署侧特征的冻结不变量。"""

from __future__ import annotations

from causalcache.mobileworld_gapfold import build_mobile_official_messages_gapfold
from causalcache.mobileworld_selector_features import (
    actions_equivalent,
    dedup_pool,
    mobile_candidate_features,
    mobile_set_context_features,
    parse_history_actions,
)
from causalcache.policy.gui_owl_official import build_official_messages

# 训练端 selector_v4_features 的布局契约(desktop 分支);两端必须同步改
CHEAP_DIM = 20
SET_DIM = 8
CHEAP_NAMES_PROBE = {"witness_match": 10, "age": 0}  # 名称→索引抽查


def make_steps(total):
    action_texts = [f"Tap item {i}." for i in range(1, total + 1)]
    full_responses = [
        f"Action: Tap item {i}.\n<tool_call>\n"
        + '{"name": "mobile_use", "arguments": {"action": "click", '
        + f'"coordinate": [{100 + i}, {200 + i}]'
        + "}}\n</tool_call>"
        for i in range(1, total + 1)
    ]
    return action_texts, full_responses


def test_contiguous_tail_degenerates_to_official() -> None:
    texts, resps = make_steps(9)
    shown = [6, 7, 8, 9]
    ours = build_mobile_official_messages_gapfold(
        goal="open settings",
        action_texts=texts,
        full_responses=resps,
        shown_steps=shown,
        step_images={s: f"img-{s}" for s in shown},
        current_image="cur",
    )
    official = build_official_messages(
        goal="open settings",
        past_action_texts=texts,
        recent_images=[f"img-{s}" for s in shown],
        current_image="cur",
        past_full_responses=resps,
    )
    assert ours == official


def test_gapfold_covers_each_step_once() -> None:
    texts, resps = make_steps(11)
    shown = [2, 7, 9]
    msgs = build_mobile_official_messages_gapfold(
        goal="g", action_texts=texts, full_responses=resps,
        shown_steps=shown, step_images={s: f"i{s}" for s in shown},
        current_image="cur",
    )
    folded, retained, images = [], [], []
    for m in msgs[1:]:
        for p in m["content"]:
            if p["type"] == "text" and m["role"] == "user":
                for line in p["text"].splitlines():
                    if line.startswith("Step") and ":" in line:
                        folded.append(int(line.split(":")[0][4:]))
            if m["role"] == "assistant":
                retained.append(p["text"])
            if p["type"] == "image":
                images.append(p["image"])
    assert sorted(folded + shown) == list(range(1, 12))
    assert retained == [resps[s - 1] for s in shown]
    assert images == [f"i{s}" for s in shown] + ["cur"]


def test_feature_layout_matches_training_contract() -> None:
    texts, resps = make_steps(8)
    parsed = parse_history_actions(resps)
    assert all(p is not None and p["action"] == "click" for p in parsed)
    pool, alias, counts = dedup_pool({i: f"hash{i}" for i in range(1, 9)})
    assert pool == list(range(1, 9)) and not alias
    proposal = {"action": "click", "coordinate": (105, 205), "coordinate2": None,
                "text": None, "button": None}
    feats = mobile_candidate_features(
        step=5, pool=pool, total_steps=8, parsed_actions=parsed,
        proposal=proposal, duplicate_counts=counts,
    )
    assert len(feats) == CHEAP_DIM
    # 步骤 5 的动作是 (105,205) → 与 proposal 完全等价 → witness(索引 10)
    assert feats[CHEAP_NAMES_PROBE["witness_match"]] == 1.0
    assert feats[CHEAP_NAMES_PROBE["age"]] == 4.0  # decision = 9
    set_feats = mobile_set_context_features(
        step=5, selected=[7, 8], total_steps=8, parsed_actions=parsed,
        proposal=proposal, duplicate_alias=alias,
    )
    assert len(set_feats) == SET_DIM


def test_action_equivalence_tolerance() -> None:
    a = {"action": "click", "coordinate": (500, 500), "coordinate2": None,
         "text": None, "button": None}
    near = dict(a, coordinate=(510, 490))
    far = dict(a, coordinate=(600, 500))
    assert actions_equivalent(a, near)
    assert not actions_equivalent(a, far)
    assert not actions_equivalent(a, dict(a, action="type"))
