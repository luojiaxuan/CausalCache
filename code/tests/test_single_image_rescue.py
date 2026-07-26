"""Unit tests for the single-image rescue miner (no GPU, no model weights).

# note (luojiaxuan): 四件事必须被钉死,否则报出来的产出率是错的而看不出来。
#   (1) **图片数匹配**:三臂在同一个 r 下必须各持 r+1 张历史图,且 sparse 的候选 j
#       严格老于 recent 臂新加的那一步 —— 否则 sparse 会和 recent 撞成同一个集合,
#       「sparse 优于 recent」退化成 0 > 0 却仍然被计成一次比较。
#   (2) **G_select 的参照系是 recent 而不是 base**。参照写错会把「多一张图」的
#       增益整个算进 selection 效应里,那正是本设计要排除的东西。
#   (3) **第一类 rescue 的四个条件缺一不可**,尤其是「recent+1 仍错」与「无关图仍错」
#       这两条否定条件 —— 它们才是把「内容有用」与「有张旧图就行」区分开的地方。
#   (4) **cross-fit 的选择与评估必须落在不同折**。用合成分数造一个 in-sample 最优与
#       两折 cross-fit 最优**完全不同**的例子来断言,而不是靠代码审查。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_argmax_agreement import agreement_record  # noqa: E402
from scripts.mine_single_image_rescue import (  # noqa: E402
    CRITERIA,
    analyse_block,
    arm_plan,
    class1_block,
    crossfit_block,
    episode_stratum,
    matched_irrelevant,
    reduce_radius,
    sample_episode_subset,
)

# 合成目标串:8 个 token,语义段的 token 下标区间与真实渲染同构。
SPANS = {
    "description": (0, 3),
    "tool_call": (3, 8),
    "action_type": (4, 5),
    "coordinate": (6, 8),
}
N_TOK = 8


def rec(*, action_ok: bool, coord_ok: bool, values: list[float] | None = None) -> dict:
    """一条合成 agreement_record:action_type / coordinate 是否命中可独立指定。"""
    hits = [True] * N_TOK
    hits[4] = action_ok
    hits[6] = coord_ok
    hits[7] = coord_ok
    return agreement_record(
        values if values is not None else [-1.0] * N_TOK,
        hits,
        total_tokens=N_TOK + 100,
        spans=SPANS,
    )


def block(
    *,
    base,
    recent,
    sparse: dict[int, dict],
    current_step: int = 10,
    radius: int = 2,
    pair_group: str = "ep-a:10",
    episode: str = "ep-a",
) -> dict:
    plan = arm_plan(current_step, radius)
    return {
        "pair_group": pair_group,
        "episode": episode,
        "current_step": current_step,
        "r": radius,
        "base_steps": list(plan["base"]),
        "recent_steps": list(plan["recent"]),
        "candidates": sorted(sparse),
        "records": {
            "base": base,
            "recent": recent,
            "sparse": {str(j): value for j, value in sparse.items()},
        },
    }


# ---------------------------------------------------------------------------
# (1) 图片数匹配
# ---------------------------------------------------------------------------
def test_all_three_arms_hold_exactly_r_plus_one_history_images():
    for radius in (0, 1, 2, 3):
        plan = arm_plan(20, radius)
        assert len(plan["base"]) == radius
        assert len(plan["recent"]) == radius + 1
        for steps in plan["sparse"].values():
            assert len(steps) == radius + 1


def test_base_is_recent_r_and_recent_arm_is_recent_r_plus_one():
    plan = arm_plan(20, 2)
    assert plan["base"] == (18, 19)
    assert plan["recent"] == (17, 18, 19)


def test_candidates_exclude_the_step_the_recent_arm_adds():
    plan = arm_plan(20, 2)
    # recent 臂加的是 17,候选必须严格老于 17,否则两臂会撞成同一个集合。
    assert plan["candidates"] == list(range(1, 17))
    assert 17 not in plan["candidates"]
    assert all(plan["sparse"][j] != plan["recent"] for j in plan["candidates"])


def test_sparse_arm_is_base_plus_exactly_one_old_step():
    plan = arm_plan(20, 2)
    for j, steps in plan["sparse"].items():
        assert set(steps) == set(plan["base"]) | {j}
        assert steps == tuple(sorted(steps))


def test_radius_zero_has_an_empty_base_and_a_single_image_recent_arm():
    plan = arm_plan(10, 0)
    assert plan["base"] == ()
    assert plan["recent"] == (9,)
    assert plan["candidates"] == list(range(1, 9))


def test_arm_plan_returns_none_when_no_candidate_survives():
    # cur=4, r=2 -> recent=(1,2,3),候选区间 [1, 1) 为空。
    assert arm_plan(4, 2) is None
    assert arm_plan(3, 0) is not None
    assert arm_plan(2, 0) is None


# ---------------------------------------------------------------------------
# matched irrelevant 对照臂
# ---------------------------------------------------------------------------
def test_matched_irrelevant_picks_the_age_nearest_other_candidate():
    assert matched_irrelevant(5, [1, 2, 3, 4, 5, 6, 7]) == 4
    assert matched_irrelevant(1, [1, 2, 3]) == 2
    assert matched_irrelevant(3, [1, 3]) == 1


def test_matched_irrelevant_breaks_ties_towards_the_older_frame():
    # 4 与 6 到 5 的距离相同;取更老的 4,免得对照因为更靠近 current 而占便宜。
    assert matched_irrelevant(5, [4, 5, 6]) == 4


def test_matched_irrelevant_is_none_without_another_candidate():
    assert matched_irrelevant(3, [3]) is None


# ---------------------------------------------------------------------------
# (2) G_select 的参照系
# ---------------------------------------------------------------------------
def test_g_select_is_measured_against_recent_and_g_rescue_against_base():
    base = rec(action_ok=False, coord_ok=False, values=[-3.0] * N_TOK)
    recent = rec(action_ok=False, coord_ok=False, values=[-2.0] * N_TOK)
    sparse = {j: rec(action_ok=False, coord_ok=False, values=[-1.0] * N_TOK) for j in (1, 2, 3)}
    out = analyse_block(block(base=base, recent=recent, sparse=sparse), thresholds=THRESH)
    # sparse -1.0,recent -2.0,base -3.0
    assert out["gains"]["G_select_crossfit"] == pytest.approx(1.0)
    assert out["gains"]["G_rescue_crossfit"] == pytest.approx(2.0)
    assert out["gains"]["G_select_insample"] == pytest.approx(1.0)
    assert out["gains"]["G_rescue_insample"] == pytest.approx(2.0)
    assert out["gains"]["G_recent_over_base"] == pytest.approx(1.0)


def test_g_content_compares_against_the_age_matched_control_not_the_pool_mean():
    base = rec(action_ok=False, coord_ok=False, values=[-3.0] * N_TOK)
    recent = rec(action_ok=False, coord_ok=False, values=[-3.0] * N_TOK)
    sparse = {
        1: rec(action_ok=False, coord_ok=False, values=[-9.0] * N_TOK),
        2: rec(action_ok=False, coord_ok=False, values=[-2.0] * N_TOK),
        3: rec(action_ok=False, coord_ok=False, values=[-1.0] * N_TOK),
    }
    out = analyse_block(block(base=base, recent=recent, sparse=sparse), thresholds=THRESH)
    # in-sample 最优是 j=3,其 age 最近的对照是 j=2(-2.0),不是池均值(-4.0)。
    assert out["in_sample_best_j"] == 3
    assert out["gains"]["G_content_insample"] == pytest.approx(1.0)
    assert out["gains"]["G_select_random_old"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# (3) 第一类 rescue 的四个条件
# ---------------------------------------------------------------------------
THRESH = {"delta_select": 0.01, "delta_rescue": 0.01, "delta_content": 0.01}


def _class1(base, recent, sparse, criterion="action_type_all_agree"):
    from scripts.mine_single_image_rescue import _metrics

    return class1_block(
        _metrics(base),
        _metrics(recent),
        {j: _metrics(value) for j, value in sparse.items()},
        sorted(sparse),
        [],
    )[criterion]


def test_class1_keeps_a_tuple_that_meets_all_four_conditions():
    base = rec(action_ok=False, coord_ok=True)
    recent = rec(action_ok=False, coord_ok=True)
    sparse = {
        1: rec(action_ok=False, coord_ok=True),
        2: rec(action_ok=True, coord_ok=True),  # 唯一修好的
        3: rec(action_ok=False, coord_ok=True),
    }
    out = _class1(base, recent, sparse)
    assert out["status"] == "eligible"
    assert out["kept"] == [2]
    assert out["discarded_by_irrelevant"] == []


def test_class1_rejects_the_group_when_base_is_already_correct():
    base = rec(action_ok=True, coord_ok=True)
    recent = rec(action_ok=False, coord_ok=True)
    sparse = {1: rec(action_ok=True, coord_ok=True), 2: rec(action_ok=True, coord_ok=True)}
    assert _class1(base, recent, sparse)["status"] == "base_already_correct"


def test_class1_rejects_the_group_when_the_recent_arm_already_fixes_it():
    # 这是本设计的要害:连续图就够的话,sparse 修好不构成任何证据。
    base = rec(action_ok=False, coord_ok=True)
    recent = rec(action_ok=True, coord_ok=True)
    sparse = {1: rec(action_ok=True, coord_ok=True), 2: rec(action_ok=False, coord_ok=True)}
    assert _class1(base, recent, sparse)["status"] == "recent_already_fixes"


def test_class1_discards_a_tuple_whose_age_matched_control_also_repairs():
    base = rec(action_ok=False, coord_ok=True)
    recent = rec(action_ok=False, coord_ok=True)
    sparse = {
        1: rec(action_ok=True, coord_ok=True),
        2: rec(action_ok=True, coord_ok=True),  # j=1 的 age 对照,也修好了
        5: rec(action_ok=False, coord_ok=True),
    }
    out = _class1(base, recent, sparse)
    # 1 的对照是 2(也修好)-> 丢弃;2 的对照是 1(也修好)-> 丢弃。
    assert out["kept"] == []
    assert out["discarded_by_irrelevant"] == [1, 2]
    assert out["n_repaired"] == 2


def test_class1_repair_fraction_exposes_the_any_old_image_explanation():
    base = rec(action_ok=False, coord_ok=True)
    recent = rec(action_ok=False, coord_ok=True)
    sparse = {j: rec(action_ok=True, coord_ok=True) for j in (1, 2, 3, 4)}
    out = _class1(base, recent, sparse)
    assert out["repair_fraction"] == 1.0
    assert out["kept"] == []  # 每个 j 的对照也都修好了,全部丢弃


def test_class1_criteria_are_independent_across_semantic_spans():
    base = rec(action_ok=True, coord_ok=False)
    recent = rec(action_ok=True, coord_ok=False)
    sparse = {1: rec(action_ok=True, coord_ok=True), 2: rec(action_ok=True, coord_ok=False)}
    from scripts.mine_single_image_rescue import _metrics

    out = class1_block(
        _metrics(base),
        _metrics(recent),
        {j: _metrics(value) for j, value in sparse.items()},
        [1, 2],
        [],
    )
    # action_type 本来就对 -> 不合格;coordinate 被 j=1 修好 -> 合格
    assert out["action_type_all_agree"]["status"] == "base_already_correct"
    assert out["coordinate_all_agree"]["kept"] == [1]
    # tool_call 整段含 action_type 与 coordinate,坐标错则整段错,同样被修好
    assert out["tool_call_all_agree"]["kept"] == [1]


# ---------------------------------------------------------------------------
# 分级判据:坐标段的二值命中率实测只有 0.0196,是个地板,必须另配分级量
# ---------------------------------------------------------------------------
def coord_rec(hit_count: int) -> dict:
    """坐标段 4 个 token 里命中 ``hit_count`` 个,其余段全中。"""
    spans = {"description": (0, 2), "tool_call": (2, 8), "action_type": (2, 3), "coordinate": (4, 8)}
    hits = [True] * N_TOK
    for index in range(4 + hit_count, 8):
        hits[index] = False
    return agreement_record([-1.0] * N_TOK, hits, total_tokens=N_TOK + 100, spans=spans)


def _graded(base, recent, sparse, criterion="coordinate_agree"):
    from scripts.mine_single_image_rescue import _metrics, class1_graded_block

    return class1_graded_block(
        _metrics(base),
        _metrics(recent),
        {j: _metrics(value) for j, value in sparse.items()},
        sorted(sparse),
    )[criterion]


def test_graded_criterion_catches_partial_coordinate_improvement():
    # 二值判据下 base/recent/sparse 全是 0(没有一个坐标是逐 token 全对),测不出差别。
    base = coord_rec(1)
    recent = coord_rec(1)
    sparse = {1: coord_rec(3), 2: coord_rec(1), 3: coord_rec(0)}
    for record in (base, recent, sparse[1]):
        from scripts.mine_single_image_rescue import _metrics

        assert _metrics(record)["coordinate_all_agree"] == 0.0
    out = _graded(base, recent, sparse)
    assert out["status"] == "eligible"
    assert out["kept"] == [1]
    assert out["base_value"] == pytest.approx(0.25)
    assert out["best_sparse_value"] == pytest.approx(0.75)


def test_graded_criterion_requires_beating_recent_not_just_base():
    base = coord_rec(1)
    recent = coord_rec(3)  # 连续图已经把坐标推到 0.75
    sparse = {1: coord_rec(2), 2: coord_rec(1), 3: coord_rec(0)}
    out = _graded(base, recent, sparse)
    # j=1 比 base 好但没比 recent 好 -> 不算
    assert out["kept"] == []
    assert out["n_improved"] == 0


def test_graded_criterion_discards_when_the_control_is_at_least_as_good():
    base = coord_rec(0)
    recent = coord_rec(0)
    sparse = {1: coord_rec(3), 2: coord_rec(3), 5: coord_rec(0)}
    out = _graded(base, recent, sparse)
    # 1 的 age 对照是 2(同样 0.75)-> 不严格优于对照 -> 丢弃;2 同理
    assert out["kept"] == []
    assert out["discarded_by_irrelevant"] == [1, 2]
    assert out["n_improved"] == 2


def test_graded_criterion_skips_a_base_that_is_already_perfect():
    base = coord_rec(4)
    recent = coord_rec(4)
    sparse = {1: coord_rec(4), 2: coord_rec(2)}
    assert _graded(base, recent, sparse)["status"] == "base_already_perfect"


def test_graded_block_is_reported_for_every_graded_criterion():
    from scripts.mine_single_image_rescue import GRADED_CRITERIA

    base = coord_rec(1)
    recent = coord_rec(1)
    entries = [
        analyse_block(
            block(
                base=base,
                recent=recent,
                sparse={1: coord_rec(3), 2: coord_rec(1), 3: coord_rec(0)},
                pair_group=f"ep-{index}:10",
                episode=f"ep-{index}",
            ),
            thresholds=THRESH,
        )
        for index in range(3)
    ]
    out = reduce_radius(entries)
    assert set(out["class1_graded"]) == set(GRADED_CRITERIA)
    assert out["class1_graded"]["coordinate_agree"]["improved_groups"] == 3
    assert out["class1_graded"]["coordinate_agree"]["improve_tuples"] == 3


# ---------------------------------------------------------------------------
# (4) cross-fit:选择与评估落在不同折
# ---------------------------------------------------------------------------
def _split(even: float, odd: float) -> list[float]:
    return [even if index % 2 == 0 else odd for index in range(N_TOK)]


def test_crossfit_selects_on_one_fold_and_scores_on_the_other():
    base = rec(action_ok=False, coord_ok=False, values=_split(-5.0, -5.0))
    recent = rec(action_ok=False, coord_ok=False, values=_split(-4.0, -4.0))
    sparse = {
        # j=1 在偶折上最好、奇折上最差;j=2 反之;j=3 两折都中等。
        1: rec(action_ok=False, coord_ok=False, values=_split(-1.0, -9.0)),
        2: rec(action_ok=False, coord_ok=False, values=_split(-9.0, -1.0)),
        3: rec(action_ok=False, coord_ok=False, values=_split(-3.0, -3.0)),
    }
    out = crossfit_block(base, recent, sparse, [1, 2, 3])
    assert [d["j_star"] for d in out["directions"]] == [1, 2]
    # 方向 1:偶折选 j=1,奇折评估 -> -9.0 vs recent 奇折 -4.0 -> -5.0
    # 方向 2:奇折选 j=2,偶折评估 -> -9.0 vs recent 偶折 -4.0 -> -5.0
    assert out["select"] == pytest.approx(-5.0)
    assert out["both_folds_select_positive"] is False


def test_crossfit_differs_from_the_in_sample_optimum():
    base = rec(action_ok=False, coord_ok=False, values=_split(-5.0, -5.0))
    recent = rec(action_ok=False, coord_ok=False, values=_split(-5.0, -5.0))
    sparse = {
        1: rec(action_ok=False, coord_ok=False, values=_split(-1.0, -9.0)),
        2: rec(action_ok=False, coord_ok=False, values=_split(-9.0, -1.0)),
        3: rec(action_ok=False, coord_ok=False, values=_split(-3.0, -3.0)),
    }
    out = analyse_block(block(base=base, recent=recent, sparse=sparse), thresholds=THRESH)
    # 全 token 均值:j=1 与 j=2 都是 -5.0,j=3 是 -3.0 -> in-sample 选 j=3
    assert out["in_sample_best_j"] == 3
    assert out["gains"]["G_select_insample"] == pytest.approx(2.0)
    # cross-fit 选到的却是两个被噪声顶上来的候选,增益为负 —— 乐观上界与诚实量分离
    assert out["gains"]["G_select_crossfit"] == pytest.approx(-4.0)
    assert out["crossfit_j_stars"] == [1, 2]


def test_class2_requires_all_three_thresholds_and_both_folds():
    base = rec(action_ok=False, coord_ok=False, values=_split(-5.0, -5.0))
    recent = rec(action_ok=False, coord_ok=False, values=_split(-3.0, -3.0))
    sparse = {
        1: rec(action_ok=False, coord_ok=False, values=_split(-1.0, -1.0)),
        2: rec(action_ok=False, coord_ok=False, values=_split(-2.0, -2.0)),
        3: rec(action_ok=False, coord_ok=False, values=_split(-2.5, -2.5)),
    }
    out = analyse_block(block(base=base, recent=recent, sparse=sparse), thresholds=THRESH)
    assert out["class2"]["positive"] is True
    assert out["class2"]["select_pass"] and out["class2"]["rescue_pass"]
    assert out["class2"]["content_pass"] and out["class2"]["both_folds_pass"]

    # 把对照臂抬到与 j* 同分,content 条件就该失败(收益不再来自内容)。
    sparse[2] = rec(action_ok=False, coord_ok=False, values=_split(-1.0, -1.0))
    out = analyse_block(block(base=base, recent=recent, sparse=sparse), thresholds=THRESH)
    assert out["class2"]["content_pass"] is False
    assert out["class2"]["positive"] is False


# ---------------------------------------------------------------------------
# 按 episode 的分层抽样
# ---------------------------------------------------------------------------
def _group_steps(n_episodes: int = 60) -> dict[str, tuple[str, int]]:
    out: dict[str, tuple[str, int]] = {}
    for index in range(n_episodes):
        episode = f"ep{index:03d}"
        step = 10 + (index % 16)
        for slot in range(3):
            out[f"{episode}:{step + slot}"] = (episode, step + slot)
    return out


def test_sampling_takes_whole_episodes_never_partial_ones():
    steps = _group_steps()
    chosen = set(sample_episode_subset(steps, 30, seed=7))
    kept = {g for g, (ep, _) in steps.items() if ep in chosen}
    by_episode: dict[str, int] = {}
    for _, (ep, _) in steps.items():
        by_episode[ep] = by_episode.get(ep, 0) + 1
    for episode in chosen:
        picked = sum(1 for g, (ep, _) in steps.items() if ep == episode and g in kept)
        assert picked == by_episode[episode]


def test_sampling_reaches_the_target_group_count_and_is_deterministic():
    steps = _group_steps()
    first = sample_episode_subset(steps, 30, seed=7)
    second = sample_episode_subset(steps, 30, seed=7)
    assert first == second
    covered = sum(1 for _, (ep, _) in steps.items() if ep in set(first))
    assert covered >= 30
    assert sample_episode_subset(steps, 30, seed=8) != first


def test_exclusion_drops_whole_episodes_never_individual_groups():
    from scripts.mine_single_image_rescue import drop_excluded_episodes

    steps = _group_steps(10)
    excluded = {"ep000", "ep003"}
    kept = drop_excluded_episodes(steps, excluded)
    assert {ep for ep, _ in kept.values()}.isdisjoint(excluded)
    # 未被剔除的 episode 必须整条留下,不能只留其中一部分决策点。
    for episode in {ep for ep, _ in steps.values()} - excluded:
        before = sum(1 for ep, _ in steps.values() if ep == episode)
        after = sum(1 for ep, _ in kept.values() if ep == episode)
        assert before == after


def test_a_second_batch_sampled_after_exclusion_is_episode_disjoint():
    from scripts.mine_single_image_rescue import drop_excluded_episodes

    steps = _group_steps(60)
    first = set(sample_episode_subset(steps, 30, seed=7))
    rest = drop_excluded_episodes(steps, first)
    second = set(sample_episode_subset(rest, 30, seed=7))
    assert second and not (second & first)
    # 两批的组也必然不相交,因为剔除单元是 episode。
    g1 = {g for g, (ep, _) in steps.items() if ep in first}
    g2 = {g for g, (ep, _) in steps.items() if ep in second}
    assert not (g1 & g2)


def test_sampling_returns_every_episode_when_the_target_exceeds_the_split():
    steps = _group_steps(5)
    assert len(sample_episode_subset(steps, 10**6, seed=1)) == 5


def test_sampling_spans_every_trajectory_length_stratum():
    steps = _group_steps()
    chosen = set(sample_episode_subset(steps, 30, seed=7))
    strata = {
        episode_stratum(sorted(s for g, (ep, s) in steps.items() if ep == episode)[1])
        for episode in chosen
    }
    assert len(strata) >= 3


# ---------------------------------------------------------------------------
# 跨机器归约:多个 glob 源 + 重复 (组, r) 必须硬失败
# ---------------------------------------------------------------------------
def test_expand_accepts_several_comma_separated_globs(tmp_path):
    from scripts.mine_single_image_rescue import expand

    a = tmp_path / "hostA"
    b = tmp_path / "hostB"
    a.mkdir()
    b.mkdir()
    (a / "raw.shard000.jsonl").write_text("", encoding="utf-8")
    (b / "raw.shard000.jsonl").write_text("", encoding="utf-8")
    out = expand(f"{a}/raw.*.jsonl,{b}/raw.*.jsonl")
    assert len(out) == 2
    assert {p.parent.name for p in out} == {"hostA", "hostB"}


def test_expand_rejects_a_glob_that_matches_nothing(tmp_path):
    from scripts.mine_single_image_rescue import expand

    (tmp_path / "raw.jsonl").write_text("", encoding="utf-8")
    # 拼错一个路径必须炸,不能安静地只归约另一批。
    with pytest.raises(SystemExit):
        expand(f"{tmp_path}/raw.jsonl,{tmp_path}/typo.*.jsonl")


def test_expand_deduplicates_paths_matched_by_two_globs(tmp_path):
    from scripts.mine_single_image_rescue import expand

    (tmp_path / "raw.shard000.jsonl").write_text("", encoding="utf-8")
    out = expand(f"{tmp_path}/raw.*.jsonl,{tmp_path}/raw.shard000.jsonl")
    assert len(out) == 1


def test_merging_two_batches_rejects_a_duplicated_group_radius(tmp_path):
    from scripts.mine_single_image_rescue import RAW_SCHEMA, write_report

    base = rec(action_ok=False, coord_ok=True)
    recent = rec(action_ok=False, coord_ok=True)
    shared = block(
        base=base,
        recent=recent,
        sparse={j: rec(action_ok=False, coord_ok=True) for j in (1, 2, 3)},
        pair_group="ep-dup:10",
        episode="ep-dup",
    )
    shared["schema_version"] = RAW_SCHEMA
    with pytest.raises(SystemExit):
        write_report(
            [shared, dict(shared)],
            thresholds=THRESH,
            common={},
            output_path=tmp_path / "out.json",
        )


# ---------------------------------------------------------------------------
# 归约计数
# ---------------------------------------------------------------------------
def test_reduce_radius_counts_groups_tuples_and_discards_separately():
    base = rec(action_ok=False, coord_ok=True)
    recent = rec(action_ok=False, coord_ok=True)
    rescued = block(
        base=base,
        recent=recent,
        sparse={
            1: rec(action_ok=True, coord_ok=True),
            2: rec(action_ok=False, coord_ok=True),
            3: rec(action_ok=False, coord_ok=True),
        },
        pair_group="ep-a:10",
        episode="ep-a",
    )
    discarded = block(
        base=base,
        recent=recent,
        sparse={
            1: rec(action_ok=True, coord_ok=True),
            2: rec(action_ok=True, coord_ok=True),
            3: rec(action_ok=False, coord_ok=True),
        },
        pair_group="ep-b:10",
        episode="ep-b",
    )
    inert = block(
        base=base,
        recent=recent,
        sparse={j: rec(action_ok=False, coord_ok=True) for j in (1, 2, 3)},
        pair_group="ep-c:10",
        episode="ep-c",
    )
    out = reduce_radius(
        [analyse_block(b, thresholds=THRESH) for b in (rescued, discarded, inert)]
    )
    stats = out["class1"]["action_type_all_agree"]
    assert stats["groups_scored"] == 3
    assert stats["eligible_groups"] == 3
    assert stats["rescued_groups"] == 1
    assert stats["rescue_tuples"] == 1
    assert stats["discarded_by_irrelevant_tuples"] == 2
    assert stats["discarded_by_irrelevant_groups"] == 1
    assert stats["yield_over_scored"] == pytest.approx(1 / 3)
    assert out["episodes"] == 3


def test_every_criterion_is_reported_for_every_radius_block():
    base = rec(action_ok=False, coord_ok=True)
    recent = rec(action_ok=False, coord_ok=True)
    entries = [
        analyse_block(
            block(
                base=base,
                recent=recent,
                sparse={j: rec(action_ok=False, coord_ok=True) for j in (1, 2, 3)},
                pair_group=f"ep-{index}:10",
                episode=f"ep-{index}",
            ),
            thresholds=THRESH,
        )
        for index in range(3)
    ]
    out = reduce_radius(entries)
    assert set(out["class1"]) == set(CRITERIA)
    assert out["baseline"]["action_type_all_agree"]["point"] == pytest.approx(0.0)
