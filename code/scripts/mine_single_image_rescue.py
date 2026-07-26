#!/usr/bin/env python3
"""Single-image rescue:在**图片数完全相同**的前提下,一张关键旧图能不能纠正当前动作。

# note (luojiaxuan): 本脚本只服务一个主张 ——「当过去事件只剩文字 summary 时,恢复少量
# 任务相关的高保真历史截图可以纠正当前 action;在相同图片数量下,它优于恢复连续 Recent
# 截图或无关旧图」。三臂的最终图片数严格相等,所以收益不能被归因于「给了更多视觉 token」。
#
# ---------------------------------------------------------------------------
# 上下文族(current 截图恒在,不计入历史预算)
# ---------------------------------------------------------------------------
#   C_r          = current + **完整** action summaries + Recent-r 张历史图
#   C_recent+1   = C_r + 下一张连续 Recent 图        == Recent-(r+1)
#   C_sparse+1   = C_r + Recent-(r+1) 之外的某张旧图 j
#   C_irrelevant = C_r + 与 j 同 age 的另一张旧图 j⁻(同 episode、age 最接近、排除 j)
#
# 三臂都是 r+1 张历史图。核心量 ``G_select(r) = Q(C_sparse+1) - Q(C_recent+1)``。
#
# 冻结渲染器 ``official_style_sparse_multiturn`` 的不变量 2 保证:1..cur-1 每一步恰好
# 出现一次 —— 被选中的步以 assistant 完整响应 + 截图出现,未选中的步以
# ``Intervening actions`` 文本行出现。所以 C_r 里「完整 action summaries」是结构性的,
# 不需要另做处理;r=0 时渲染器退化成官方布局本身(逐字节相同)。
#
# ---------------------------------------------------------------------------
# 为什么 r=2 是主设置而不是 r=0
# ---------------------------------------------------------------------------
# r=0 时两臂**各缺一半信息**:Recent-1 给「我刚操作到哪」但没有关键值,Sparse-1 给关键值
# 但没有局部连续性。互有短板,sparse 未必赢,所以 r=0 出现 null 是预期之内。
# r=2 时局部连续性已由 Recent-2 保证,最后一个槽位才是公平较量:
# 「再给一张高度相似的表单页」vs「早先短信里的验证码」。测的是**条件边际价值**。
# 预注册的形状预测:r=0 弱 -> r=1/2 最大 -> r=3 起饱和。形状对上了才算机制被验证。
#
# ---------------------------------------------------------------------------
# 判据必须用 argmax,不能只用 logprob
# ---------------------------------------------------------------------------
# 实测 logprob 增益与 argmax 翻转几乎不相关(回归 r=0.055),+0.0075/token 的 oracle
# 增益在 argmax 层面完全不动。所以第一类筛选直接看动作是否被修复。决策落在目标串**末尾**
# 的 ``<tool_call>`` JSON 里(开头第一个 token 是常量 ``Action``),因此看的是
# ``action_type`` / ``coordinate`` / ``tool_call``,不是 first-k。
#
# ---------------------------------------------------------------------------
# 与既有打分逐位同源
# ---------------------------------------------------------------------------
# 复用 ``probe_argmax_agreement.ArgmaxSetScorer``(它自带 ``verify_agreement_parity``,
# 每次启动实测三条路径的均值,漂了就 fail-closed),以及它的 ``target_token_stats`` /
# ``agreement_record`` / ``metrics_from_record`` / 语义段拆分。本脚本**不重写任何前向或
# 指标算术**,只负责选臂、落盘、归约。
#
# 原始量全部落盘(每个臂一条完整 ``agreement_record``),阈值只在归约阶段生效,下游可以
# 用别的阈值重算而不必重跑前向。
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_sparse_multiturn import SPARSE_MULTITURN_PROMPT_FORMAT
from scripts.budget_replacement_sweep import cluster_bootstrap
from scripts.probe_argmax_agreement import (
    CROSSFIT_DIRECTIONS,
    ArgmaxSetScorer,
    fold_logprob_mean,
    metrics_from_record,
)
from scripts.probe_oracle_headroom import (
    BEAM_WIDTH,
    DecisionPoint,
    SetScoreCache,
    group_dev_rows,
    load_config,
    load_sparse_samples,
    resolve_sample_files,
    set_key,
    sha256_of,
)

REPORT_SCHEMA = "causalcache.single_image_rescue.v1"
SHARD_REPORT_SCHEMA = "causalcache.single_image_rescue_shard.v1"
RAW_SCHEMA = "causalcache.single_image_rescue_raw.v1"
FROZEN_FORMAT = SPARSE_MULTITURN_PROMPT_FORMAT

DEFAULT_RADII: tuple[int, ...] = (0, 1, 2, 3)
MAIN_RADIUS = 2

# note (luojiaxuan): 三个判据都用 ``*_all_agree``(该语义段的每个 token 都是 argmax 命中)
# 而不是比例 —— 「动作被修复」在闭环里是全有或全无的:``click`` 的 ``cl`` 对了但 ``ick``
# 错了不会走对分支。``coordinate_agree``(比例)仍然逐条落盘,只是不当判据。
CRITERIA: tuple[str, ...] = (
    "action_type_all_agree",
    "tool_call_all_agree",
    "coordinate_all_agree",
)
# note (luojiaxuan): ``coordinate_all_agree`` 在 dev 60 组上实测只有 **0.0196** —— 坐标
# 逐 token 精确命中本来就几乎不发生,拿它当二值判据等于在地板上找抬升,测不出东西来。
# ``tool_call_all_agree``(0.13-0.17)含坐标段,同样被这个地板拖着。所以除了二值判据,
# 再按**分级**判据算一份:sparse 臂的段内命中比例是否严格高于 base / recent / 无关图。
# 这仍然是 argmax 层面的量(不是 logprob),但不要求坐标一次就对到底,能把「往正确方向
# 挪了」这类改变露出来。两套判据并列上报,不合并成一个数。
GRADED_CRITERIA: tuple[str, ...] = (
    "coordinate_agree",
    "action_type_agree",
    "tool_call_agree",
    "token_agree",
)
# 报告里逐 r 汇报的基线量;前两个是天花板,没有它们任何 delta 都没法解读。
BASELINE_METRICS: tuple[str, ...] = (
    "action_type_all_agree",
    "tool_call_all_agree",
    "coordinate_all_agree",
    "coordinate_agree",
    "action_type_agree",
    "tool_call_agree",
    "token_agree",
    "all_agree",
)
GAIN_NAMES: tuple[str, ...] = (
    "G_select_crossfit",
    "G_rescue_crossfit",
    "G_content_crossfit",
    "G_select_insample",
    "G_rescue_insample",
    "G_content_insample",
    "G_select_random_old",
    "G_recent_over_base",
)
ARGMAX_DELTA_NAMES: tuple[str, ...] = tuple(
    f"delta_{name}_at_crossfit_j" for name in CRITERIA
)
# 分层抽样的 episode 分层键上界(按该 episode 各组 decision_step 的中位数)。
# train 实测 cur ∈ [10, 45],中位 16。
STRATA_UPPER: tuple[int, ...] = (12, 16, 20, 10**9)


# ---------------------------------------------------------------------------
# 臂的构造(纯函数)
# ---------------------------------------------------------------------------
def arm_plan(current_step: int, radius: int) -> dict[str, Any] | None:
    """三臂 + 候选池。``None`` 表示这个 (组, r) 连一个合法候选都凑不出来。

    # note (luojiaxuan): 候选 j 必须**严格老于** recent 臂新加的那一步 ``cur-r-1``,
    # 否则 j == cur-r-1 时 sparse 臂与 recent 臂是同一个集合,「sparse 优于 recent」
    # 就退化成 0 > 0。所以 j ∈ [1, cur-r-2],候选数 n_old = cur-r-2。
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if current_step < 1:
        raise ValueError("current_step must be 1-based and positive")
    base = tuple(range(current_step - radius, current_step))
    if base and base[0] < 1:
        return None
    recent = tuple(range(current_step - radius - 1, current_step))
    if recent[0] < 1:
        return None
    candidates = list(range(1, current_step - radius - 1))
    if not candidates:
        return None
    return {
        "base": base,
        "recent": recent,
        "candidates": candidates,
        "sparse": {j: tuple(sorted((*base, j))) for j in candidates},
    }


def matched_irrelevant(j: int, candidates: list[int]) -> int | None:
    """同 episode 内 age 最接近、排除 j 本身的旧图。

    # note (luojiaxuan): 因为本脚本把**每一个** j 都打了分,「age 最接近的另一张」必然
    # 也在已打分集合里 —— 这个对照臂是**零额外前向**的。平手时取更老的那张(j-1):
    # 让对照至少和 j 一样老,免得对照因为更靠近 current 而白占便宜。
    """
    others = [other for other in candidates if other != j]
    if not others:
        return None
    return min(others, key=lambda other: (abs(other - j), other))


# ---------------------------------------------------------------------------
# 按 episode 的分层抽样(纯函数)
# ---------------------------------------------------------------------------
def drop_excluded_episodes(
    group_steps: dict[str, tuple[str, int]], excluded: set[str]
) -> dict[str, tuple[str, int]]:
    """按 **episode** 剔除,不是按 group。

    # note (luojiaxuan): 扩样本量时第二批必须与第一批 episode 级不相交。同一 episode
    # 内的决策点共享轨迹与目标分布,把一条 episode 劈到两批里,两批就不再是独立样本,
    # 下游按 episode 聚簇的 bootstrap 会低估方差,「两批结论是否一致」这个复现检验也
    # 就失去意义。所以剔除的单位只能是 episode。
    """
    return {
        group: value
        for group, value in group_steps.items()
        if value[0] not in excluded
    }


def episode_stratum(median_step: int) -> int:
    for index, upper in enumerate(STRATA_UPPER):
        if median_step <= upper:
            return index
    return len(STRATA_UPPER) - 1


def sample_episode_subset(
    group_steps: dict[str, tuple[str, int]], target_groups: int, seed: int
) -> list[str]:
    """抽 episode(不是抽组),直到累计组数达到 target。

    ``group_steps`` 是 ``pair_group -> (episode, decision_step)``。分层键是该 episode 各组
    decision_step 的中位数,配额按各层的**组数**比例分配,某层抽干了就顺延到别层。

    # note (luojiaxuan): 抽样单元必须是 episode:同一条轨迹里的多个决策点共享历史与目标
    # 分布,按组抽会让 episode-cluster bootstrap 的簇结构失真,也会让子集里出现「同一条
    # 轨迹只进来一个决策点」的偏斜。整条 episode 进出保证子集本身仍是 episode 的样本。
    """
    if target_groups <= 0:
        raise ValueError("target_groups must be positive")
    by_episode: dict[str, list[int]] = {}
    for _, (episode, step) in sorted(group_steps.items()):
        by_episode.setdefault(episode, []).append(step)
    strata: dict[int, list[str]] = {}
    for episode, steps in by_episode.items():
        ordered = sorted(steps)
        strata.setdefault(
            episode_stratum(ordered[len(ordered) // 2]), []
        ).append(episode)
    total_groups = sum(len(steps) for steps in by_episode.values())
    if target_groups >= total_groups:
        return sorted(by_episode)

    rng = random.Random(seed)
    queues: dict[int, list[str]] = {}
    quota: dict[int, int] = {}
    for index in sorted(strata):
        members = sorted(strata[index])
        rng.shuffle(members)
        queues[index] = members
        size = sum(len(by_episode[episode]) for episode in members)
        quota[index] = int(round(target_groups * size / total_groups))

    chosen: list[str] = []
    taken = 0
    # 第一轮:各层按组数配额独立填,层内顺序已用 seed 打乱。
    for index in sorted(queues):
        filled = 0
        while queues[index] and filled < quota[index]:
            episode = queues[index].pop(0)
            chosen.append(episode)
            filled += len(by_episode[episode])
            taken += len(by_episode[episode])
    # 第二轮:配额取整后可能仍差几组,按层轮转补齐,保证最终 >= target。
    order = sorted(queues)
    position = 0
    while taken < target_groups and any(queues[index] for index in order):
        index = order[position % len(order)]
        position += 1
        if not queues[index]:
            continue
        episode = queues[index].pop(0)
        chosen.append(episode)
        taken += len(by_episode[episode])
    return sorted(chosen)


# ---------------------------------------------------------------------------
# 归约:从原始 record 到每个 (组, r) 的量(纯函数,无 GPU)
# ---------------------------------------------------------------------------
def _metrics(record: dict[str, Any]) -> dict[str, float]:
    return metrics_from_record(record)


def crossfit_block(
    base: dict[str, Any],
    recent: dict[str, Any],
    sparse: dict[int, dict[str, Any]],
    candidates: list[int],
) -> dict[str, Any] | None:
    """两折 cross-fit:在一折的 logprob 上挑 j*,在**另一折**上算三个 delta。

    # note (luojiaxuan): 候选池有十几个 j,在同一批 token 上取 max 必然吃 winner's
    # curse。所以主结论一律用 cross-fit:选择只发生在 select 折,评估只发生在 evaluate
    # 折,两个方向各算一次再平均。in-sample 版本单独以 ``*_insample`` 出现,只当乐观上界。
    """
    directions: list[dict[str, Any]] = []
    for select, evaluate in CROSSFIT_DIRECTIONS:
        scored = [
            (value, j)
            for j in candidates
            if (value := fold_logprob_mean(sparse[j], select)) is not None
        ]
        if not scored:
            return None
        best = max(scored)[1]
        chosen = fold_logprob_mean(sparse[best], evaluate)
        anchor = fold_logprob_mean(base, evaluate)
        rival = fold_logprob_mean(recent, evaluate)
        if chosen is None or anchor is None or rival is None:
            return None
        control = matched_irrelevant(best, candidates)
        control_value = (
            None if control is None else fold_logprob_mean(sparse[control], evaluate)
        )
        directions.append(
            {
                "select_fold": select,
                "evaluate_fold": evaluate,
                "j_star": best,
                "j_control": control,
                "select": chosen - rival,
                "rescue": chosen - anchor,
                "content": None if control_value is None else chosen - control_value,
            }
        )
    if not directions:
        return None

    def average(name: str) -> float | None:
        values = [d[name] for d in directions if d[name] is not None]
        return None if not values else math.fsum(values) / len(values)

    return {
        "directions": directions,
        "j_stars": [d["j_star"] for d in directions],
        "select": average("select"),
        "rescue": average("rescue"),
        "content": average("content"),
        "both_folds_select_positive": all(d["select"] > 0 for d in directions),
        "both_folds_rescue_positive": all(d["rescue"] > 0 for d in directions),
        "both_folds_content_positive": all(
            d["content"] is not None and d["content"] > 0 for d in directions
        ),
    }


def class1_block(
    base_metrics: dict[str, float],
    recent_metrics: dict[str, float],
    sparse_metrics: dict[int, dict[str, float]],
    candidates: list[int],
    crossfit_j: list[int],
) -> dict[str, Any]:
    """第一类 —— 真实 action 修复。base 错 -> sparse 变对 -> recent 仍错 -> 无关图仍错。"""
    out: dict[str, Any] = {}
    for criterion in CRITERIA:
        anchor = base_metrics.get(criterion)
        rival = recent_metrics.get(criterion)
        if anchor is None or rival is None:
            out[criterion] = {"status": "span_missing"}
            continue
        # 「base 已经对」或「recent+1 也修好了」都让这个 (组, r) 整体不合格 ——
        # 前者没有可修的东西,后者说明连续图就够,不构成 sparse 的证据。
        if anchor >= 1.0:
            out[criterion] = {"status": "base_already_correct"}
            continue
        if rival >= 1.0:
            out[criterion] = {"status": "recent_already_fixes"}
            continue
        kept: list[int] = []
        discarded: list[int] = []
        repaired: list[int] = []
        for j in candidates:
            value = sparse_metrics[j].get(criterion)
            if value is None or value < 1.0:
                continue
            repaired.append(j)
            control = matched_irrelevant(j, candidates)
            control_value = (
                None if control is None else sparse_metrics[control].get(criterion)
            )
            if control_value is not None and control_value >= 1.0:
                discarded.append(j)
            else:
                kept.append(j)
        out[criterion] = {
            "status": "eligible",
            "n_candidates": len(candidates),
            "repaired": repaired,
            "kept": kept,
            "discarded_by_irrelevant": discarded,
            "n_repaired": len(repaired),
            "n_kept": len(kept),
            "n_discarded": len(discarded),
            # note (luojiaxuan): 「随便哪张旧图都能修」的直接度量。它高就说明收益来自
            # 「有张旧图」而不是内容,下游可以用它设更严的门槛而不必重跑。
            "repair_fraction": len(repaired) / len(candidates),
            "crossfit_j_kept": sorted(set(crossfit_j) & set(kept)),
            "crossfit_j_repairs": bool(set(crossfit_j) & set(repaired)),
        }
    return out


def class1_graded_block(
    base_metrics: dict[str, float],
    recent_metrics: dict[str, float],
    sparse_metrics: dict[int, dict[str, float]],
    candidates: list[int],
) -> dict[str, Any]:
    """分级版第一类:段内命中比例严格优于 base / recent / 无关图。

    # note (luojiaxuan): 与二值版共用同一批数字,只是把「变对」放宽成「严格变好」。
    # base 已经满分(1.0)时没有可改进的余地,同样按不合格计,免得把天花板算成 0 收益。
    """
    out: dict[str, Any] = {}
    for criterion in GRADED_CRITERIA:
        anchor = base_metrics.get(criterion)
        rival = recent_metrics.get(criterion)
        if anchor is None or rival is None:
            out[criterion] = {"status": "span_missing"}
            continue
        if anchor >= 1.0:
            out[criterion] = {"status": "base_already_perfect"}
            continue
        kept: list[int] = []
        discarded: list[int] = []
        improved: list[int] = []
        for j in candidates:
            value = sparse_metrics[j].get(criterion)
            if value is None or not (value > anchor and value > rival):
                continue
            improved.append(j)
            control = matched_irrelevant(j, candidates)
            control_value = (
                None if control is None else sparse_metrics[control].get(criterion)
            )
            if control_value is not None and value <= control_value:
                discarded.append(j)
            else:
                kept.append(j)
        out[criterion] = {
            "status": "eligible",
            "n_candidates": len(candidates),
            "kept": kept,
            "discarded_by_irrelevant": discarded,
            "n_improved": len(improved),
            "n_kept": len(kept),
            "n_discarded": len(discarded),
            "improve_fraction": len(improved) / len(candidates),
            "base_value": anchor,
            "recent_value": rival,
            "best_sparse_value": max(
                (
                    sparse_metrics[j][criterion]
                    for j in candidates
                    if criterion in sparse_metrics[j]
                ),
                default=None,
            ),
        }
    return out


def analyse_block(raw: dict[str, Any], *, thresholds: dict[str, float]) -> dict[str, Any]:
    """一个 (组, r) 的全部归约量。原始 record 已在盘上,本函数不做任何前向。"""
    base = raw["records"]["base"]
    recent = raw["records"]["recent"]
    sparse = {int(j): record for j, record in raw["records"]["sparse"].items()}
    candidates = sorted(sparse)
    base_metrics = _metrics(base)
    recent_metrics = _metrics(recent)
    sparse_metrics = {j: _metrics(record) for j, record in sparse.items()}

    crossfit = crossfit_block(base, recent, sparse, candidates)
    in_sample_best = max(candidates, key=lambda j: sparse[j]["mean"])
    control = matched_irrelevant(in_sample_best, candidates)
    mean_over_candidates = math.fsum(sparse[j]["mean"] for j in candidates) / len(
        candidates
    )

    gains: dict[str, float | None] = {
        "G_select_crossfit": None if crossfit is None else crossfit["select"],
        "G_rescue_crossfit": None if crossfit is None else crossfit["rescue"],
        "G_content_crossfit": None if crossfit is None else crossfit["content"],
        "G_select_insample": sparse[in_sample_best]["mean"] - recent["mean"],
        "G_rescue_insample": sparse[in_sample_best]["mean"] - base["mean"],
        "G_content_insample": (
            None
            if control is None
            else sparse[in_sample_best]["mean"] - sparse[control]["mean"]
        ),
        "G_select_random_old": mean_over_candidates - recent["mean"],
        "G_recent_over_base": recent["mean"] - base["mean"],
    }

    crossfit_j = [] if crossfit is None else list(dict.fromkeys(crossfit["j_stars"]))
    argmax_deltas: dict[str, float | None] = {}
    for criterion in CRITERIA:
        rival = recent_metrics.get(criterion)
        values = [
            sparse_metrics[j][criterion]
            for j in crossfit_j
            if criterion in sparse_metrics[j]
        ]
        argmax_deltas[f"delta_{criterion}_at_crossfit_j"] = (
            None
            if rival is None or not values
            else math.fsum(values) / len(values) - rival
        )

    class2 = None
    if crossfit is not None and crossfit["content"] is not None:
        class2 = {
            "select_pass": crossfit["select"] > thresholds["delta_select"],
            "rescue_pass": crossfit["rescue"] > thresholds["delta_rescue"],
            "content_pass": crossfit["content"] > thresholds["delta_content"],
            "both_folds_pass": (
                crossfit["both_folds_select_positive"]
                and crossfit["both_folds_rescue_positive"]
            ),
        }
        class2["positive"] = all(
            class2[name]
            for name in ("select_pass", "rescue_pass", "content_pass", "both_folds_pass")
        )

    return {
        "pair_group": raw["pair_group"],
        "episode": raw["episode"],
        "current_step": raw["current_step"],
        "r": raw["r"],
        "n_candidates": len(candidates),
        "n_tok": int(base["n_tok"]),
        "base_steps": raw["base_steps"],
        "recent_steps": raw["recent_steps"],
        "baseline": {
            name: base_metrics[name] for name in BASELINE_METRICS if name in base_metrics
        },
        "recent_baseline": {
            name: recent_metrics[name]
            for name in BASELINE_METRICS
            if name in recent_metrics
        },
        "base_mean_logprob": base["mean"],
        "recent_mean_logprob": recent["mean"],
        "gains": gains,
        "argmax_deltas": argmax_deltas,
        "crossfit_j_stars": crossfit_j,
        "in_sample_best_j": in_sample_best,
        "class1": class1_block(
            base_metrics, recent_metrics, sparse_metrics, candidates, crossfit_j
        ),
        "class1_graded": class1_graded_block(
            base_metrics, recent_metrics, sparse_metrics, candidates
        ),
        "class2": class2,
    }


def reduce_radius(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """把一个 r 下的全部 (组) 归约成报告里的一块。"""
    episode = {block["pair_group"]: block["episode"] for block in blocks}

    def boot(name: str, getter: Any) -> dict[str, Any] | None:
        pairs = [
            (episode[block["pair_group"]], value)
            for block in blocks
            if (value := getter(block)) is not None
        ]
        return cluster_bootstrap(pairs)

    baseline = {
        name: boot(name, lambda b, n=name: b["baseline"].get(n))
        for name in BASELINE_METRICS
    }
    recent_baseline = {
        name: boot(name, lambda b, n=name: b["recent_baseline"].get(n))
        for name in BASELINE_METRICS
    }
    gains = {name: boot(name, lambda b, n=name: b["gains"].get(n)) for name in GAIN_NAMES}
    argmax = {
        name: boot(name, lambda b, n=name: b["argmax_deltas"].get(n))
        for name in ARGMAX_DELTA_NAMES
    }

    class1: dict[str, Any] = {}
    for criterion in CRITERIA:
        entries = [block["class1"].get(criterion, {}) for block in blocks]
        eligible = [e for e in entries if e.get("status") == "eligible"]
        rescued = [e for e in eligible if e["n_kept"] > 0]
        statuses: dict[str, int] = {}
        for entry in entries:
            statuses[entry.get("status", "missing")] = (
                statuses.get(entry.get("status", "missing"), 0) + 1
            )
        class1[criterion] = {
            "groups_scored": len(blocks),
            "status_counts": statuses,
            "eligible_groups": len(eligible),
            "rescued_groups": len(rescued),
            "rescued_episodes": len(
                {
                    block["episode"]
                    for block, entry in zip(blocks, entries)
                    if entry.get("status") == "eligible" and entry["n_kept"] > 0
                }
            ),
            "rescue_tuples": sum(e["n_kept"] for e in eligible),
            "discarded_by_irrelevant_tuples": sum(e["n_discarded"] for e in eligible),
            "discarded_by_irrelevant_groups": sum(
                1 for e in eligible if e["n_discarded"] > 0 and e["n_kept"] == 0
            ),
            "yield_over_scored": len(rescued) / len(blocks) if blocks else None,
            "yield_over_eligible": (
                len(rescued) / len(eligible) if eligible else None
            ),
            "crossfit_j_rescued_groups": sum(
                1 for e in eligible if e["crossfit_j_kept"]
            ),
            "mean_repair_fraction": (
                math.fsum(e["repair_fraction"] for e in eligible) / len(eligible)
                if eligible
                else None
            ),
            "unique_j_rescued_groups": sum(1 for e in rescued if e["n_kept"] == 1),
        }

    class1_graded: dict[str, Any] = {}
    for criterion in GRADED_CRITERIA:
        entries = [block["class1_graded"].get(criterion, {}) for block in blocks]
        eligible = [e for e in entries if e.get("status") == "eligible"]
        improved = [e for e in eligible if e["n_kept"] > 0]
        class1_graded[criterion] = {
            "groups_scored": len(blocks),
            "eligible_groups": len(eligible),
            "improved_groups": len(improved),
            "improve_tuples": sum(e["n_kept"] for e in eligible),
            "discarded_by_irrelevant_tuples": sum(e["n_discarded"] for e in eligible),
            "yield_over_scored": len(improved) / len(blocks) if blocks else None,
            "yield_over_eligible": len(improved) / len(eligible) if eligible else None,
            "mean_improve_fraction": (
                math.fsum(e["improve_fraction"] for e in eligible) / len(eligible)
                if eligible
                else None
            ),
        }

    class2_entries = [b["class2"] for b in blocks if b["class2"] is not None]
    class2 = {
        "groups_scored": len(blocks),
        "groups_evaluable": len(class2_entries),
        "positive_groups": sum(1 for e in class2_entries if e["positive"]),
        "select_pass_groups": sum(1 for e in class2_entries if e["select_pass"]),
        "rescue_pass_groups": sum(1 for e in class2_entries if e["rescue_pass"]),
        "content_pass_groups": sum(1 for e in class2_entries if e["content_pass"]),
        "both_folds_pass_groups": sum(1 for e in class2_entries if e["both_folds_pass"]),
        "yield_over_scored": (
            sum(1 for e in class2_entries if e["positive"]) / len(blocks)
            if blocks
            else None
        ),
    }

    return {
        "groups": len(blocks),
        "episodes": len(set(episode.values())),
        "mean_candidates": math.fsum(b["n_candidates"] for b in blocks) / len(blocks),
        "baseline": baseline,
        "recent_baseline": recent_baseline,
        "gains": gains,
        "argmax_deltas": argmax,
        "class1": class1,
        "class1_graded": class1_graded,
        "class2": class2,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mine single-image rescue tuples under a matched visual budget: does one "
            "task-relevant old screenshot fix the current action where the next "
            "consecutive Recent screenshot does not?"
        )
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--episode-filter", type=Path, required=True)
    parser.add_argument("--episode-filter-key", required=True)
    parser.add_argument(
        "--exclude-episodes",
        type=Path,
        default=None,
        help=(
            "JSON holding a list of episodes to drop before sampling, so a second "
            "batch is episode-disjoint from an earlier one"
        ),
    )
    parser.add_argument("--exclude-episodes-key", default="episodes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, default=None)
    parser.add_argument("--score-cache", type=Path, default=None)
    parser.add_argument("--radii", default="0,1,2,3")
    parser.add_argument("--target-groups", type=int, default=300)
    parser.add_argument("--sample-seed", type=int, default=20260726)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-reduction", action="store_true")
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument("--heartbeat", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--delta-select", type=float, default=0.01)
    parser.add_argument("--delta-rescue", type=float, default=0.01)
    parser.add_argument("--delta-content", type=float, default=0.01)
    parser.add_argument(
        "--reduce-from-raw",
        default=None,
        help=(
            "CPU-only reduction: one or more comma-separated globs for the raw "
            "per-(group, r) JSONL shards; skips every forward pass and only recomputes "
            "the report. Multiple globs let one reduction span batches produced on "
            "different hosts; a duplicated (pair_group, r) is a hard error."
        ),
    )
    return parser


def expand(pattern: str) -> list[Path]:
    """Expand one or more comma-separated globs into a de-duplicated path list.

    # note (luojiaxuan): 支持多个 glob 是为了把**跨机器**的两批原始产物喂进同一次归约
    # (pilot 在 hyper00、batch2 在 hyper01,落盘目录不同)。每个 glob 必须各自命中至少
    # 一个文件 —— 否则拼错一个路径会安静地把那一批整个丢掉,而归约照常给出一份「看起来
    # 很正常」的报告。去重按解析后的绝对路径做,免得同一个文件被两个 glob 命中而进两次。
    """
    seen: dict[Path, None] = {}
    for part in (piece.strip() for piece in pattern.split(",")):
        if not part:
            continue
        matches = sorted(Path(match).resolve() for match in globlib.glob(part))
        if not matches:
            raise SystemExit(f"pattern {part!r} matched no file")
        for match in matches:
            seen.setdefault(match, None)
    if not seen:
        raise SystemExit(f"pattern {pattern!r} matched no file")
    return list(seen)


def load_raw(paths: list[Path]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            lines = handle.readlines()
        for position, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # 只容忍无换行结尾的最后一行(被掐断的写入);中间坏行是真损坏。
                if position == len(lines) and not line.endswith("\n"):
                    break
                raise
            if record.get("schema_version") != RAW_SCHEMA:
                raise SystemExit(
                    f"{path}:{position} carries schema {record.get('schema_version')!r}, "
                    f"expected {RAW_SCHEMA!r}"
                )
            blocks.append(record)
    return blocks


def write_report(
    blocks: list[dict[str, Any]],
    *,
    thresholds: dict[str, float],
    common: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    analysed = [analyse_block(block, thresholds=thresholds) for block in blocks]
    # 同一个 (组, r) 只能出现一次;分片重叠会悄悄把分母做大。
    seen: set[tuple[str, int]] = set()
    for entry in analysed:
        key = (entry["pair_group"], entry["r"])
        if key in seen:
            raise SystemExit(
                f"{key} appears twice in the raw shards; the shard split overlapped "
                "and the denominator would be inflated"
            )
        seen.add(key)

    by_radius: dict[str, Any] = {}
    for radius in sorted({entry["r"] for entry in analysed}):
        members = [entry for entry in analysed if entry["r"] == radius]
        by_radius[str(radius)] = reduce_radius(members)

    report = {
        "schema_version": REPORT_SCHEMA,
        "main_radius": MAIN_RADIUS,
        "thresholds": thresholds,
        "claim": (
            "With the number of images held exactly equal, restoring one task-relevant "
            "high-fidelity old screenshot beats restoring the next consecutive Recent "
            "screenshot, and beats an age-matched but content-irrelevant old screenshot."
        ),
        "arm_algebra": {
            "C_r": "current + full action summaries + Recent-r history images",
            "C_recent+1": "C_r + the next consecutive Recent image == Recent-(r+1)",
            "C_sparse+1": "C_r + one old image j, with j strictly older than cur-r-1",
            "C_irrelevant+1": (
                "C_r + the age-nearest other candidate j-, tie-broken to the older one"
            ),
            "G_select(r)": "Q(C_sparse+1) - Q(C_recent+1); all three arms hold r+1 images",
            "crossfit": (
                "j* is argmax-ed on one token fold and evaluated on the other; the two "
                "directions are averaged. *_insample is the same quantity with j* chosen "
                "on all tokens and is an OPTIMISTIC upper bound."
            ),
            "class1": (
                "argmax repair: base wrong -> C_sparse+1 correct -> C_recent+1 still "
                "wrong -> age-matched irrelevant still wrong"
            ),
        },
        "bootstrap": {
            "cluster_unit": "episode",
            "implementation": "scripts.budget_replacement_sweep.cluster_bootstrap",
        },
        "by_radius": by_radius,
        "per_group": analysed,
        **common,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def show(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "n/a".rjust(36)
    star = "*" if (entry["ci_low"] > 0 or entry["ci_high"] < 0) else " "
    return "%+.4f [%+.4f, %+.4f] n=%-4d%s" % (
        entry["point"], entry["ci_low"], entry["ci_high"], entry["n"], star
    )


def print_report(report: dict[str, Any]) -> None:
    for radius, block in sorted(report["by_radius"].items(), key=lambda kv: int(kv[0])):
        tag = "  <== MAIN" if int(radius) == report["main_radius"] else ""
        print("")
        print(
            f"=== r={radius}  B={int(radius) + 1}  groups={block['groups']}  "
            f"episodes={block['episodes']}  mean_candidates="
            f"{block['mean_candidates']:.1f}{tag}"
        )
        print("-- baseline correctness of C_r (the ceiling)")
        for name in ("action_type_all_agree", "tool_call_all_agree", "coordinate_all_agree"):
            print(f"   base   {name:<26}{show(block['baseline'].get(name))}")
            print(f"   recent {name:<26}{show(block['recent_baseline'].get(name))}")
        print("-- G(r) log-prob per token")
        for name in GAIN_NAMES:
            print(f"   {name:<28}{show(block['gains'].get(name))}")
        print("-- class 1 (argmax repair)")
        for criterion in CRITERIA:
            stats = block["class1"][criterion]
            print(
                f"   {criterion:<26} eligible={stats['eligible_groups']:<4} "
                f"rescued_groups={stats['rescued_groups']:<4} "
                f"tuples={stats['rescue_tuples']:<4} "
                f"discarded={stats['discarded_by_irrelevant_tuples']:<4} "
                f"yield={('%.3f' % stats['yield_over_scored']) if stats['yield_over_scored'] is not None else 'n/a'}"
            )
        print("-- class 1 graded (strictly better span agreement than base/recent/control)")
        for criterion in GRADED_CRITERIA:
            stats = block["class1_graded"][criterion]
            print(
                f"   {criterion:<26} eligible={stats['eligible_groups']:<4} "
                f"improved_groups={stats['improved_groups']:<4} "
                f"tuples={stats['improve_tuples']:<4} "
                f"discarded={stats['discarded_by_irrelevant_tuples']:<4} "
                f"yield={('%.3f' % stats['yield_over_scored']) if stats['yield_over_scored'] is not None else 'n/a'}"
            )
        stats = block["class2"]
        print(
            f"-- class 2 (utility-positive) positive={stats['positive_groups']}/"
            f"{stats['groups_evaluable']} "
            f"select={stats['select_pass_groups']} rescue={stats['rescue_pass_groups']} "
            f"content={stats['content_pass_groups']}"
        )


def main() -> None:
    args = build_argument_parser().parse_args()
    thresholds = {
        "delta_select": args.delta_select,
        "delta_rescue": args.delta_rescue,
        "delta_content": args.delta_content,
    }

    if args.reduce_from_raw:
        blocks = load_raw(expand(args.reduce_from_raw))
        report = write_report(
            blocks,
            thresholds=thresholds,
            common={
                "reduced_from": args.reduce_from_raw,
                "format": FROZEN_FORMAT,
                "raw_blocks": len(blocks),
            },
            output_path=args.output,
        )
        print_report(report)
        print("")
        print(f"report: {args.output}")
        return

    if args.shard_count < 1:
        raise SystemExit("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit(
            f"--shard-index {args.shard_index} must lie inside "
            f"[0, --shard-count={args.shard_count})"
        )
    if args.shard_count > 1 and args.score_cache is None:
        raise SystemExit("--shard-count > 1 requires --score-cache")
    if args.shard_count > 1 and not args.skip_reduction:
        raise SystemExit(
            "a sharded process only holds 1/--shard-count of the sample; reducing on "
            "that slice would silently shrink the denominator. Pass --skip-reduction on "
            "every shard, then reduce once with --reduce-from-raw."
        )
    radii = tuple(int(part) for part in args.radii.split(",") if part.strip())
    if not radii or any(r < 0 for r in radii):
        raise SystemExit(f"--radii must be non-negative ints, got {args.radii!r}")

    config = load_config(args.config)
    sample_files = resolve_sample_files(args.dataset_root)
    samples = load_sparse_samples(sample_files)

    payload = json.loads(args.episode_filter.read_text(encoding="utf-8"))
    if args.episode_filter_key not in payload:
        available = sorted(k for k, v in payload.items() if isinstance(v, list))
        raise SystemExit(
            f"--episode-filter-key {args.episode_filter_key!r} absent from "
            f"{args.episode_filter}; list-valued keys are {available}"
        )
    allowed = set(payload[args.episode_filter_key])
    if not allowed:
        raise SystemExit("episode filter selected an empty episode set")
    rows = group_dev_rows(samples, allowed)
    if not rows:
        raise SystemExit("episode filter removed every group; wrong corpus?")

    group_steps = {
        group: (row["N0"]["episode"], int(row["N0"]["decision_step"]))
        for group, row in rows.items()
    }
    episodes_in_split = len({ep for ep, _ in group_steps.values()})
    excluded: set[str] = set()
    if args.exclude_episodes is not None:
        payload = json.loads(args.exclude_episodes.read_text(encoding="utf-8"))
        if args.exclude_episodes_key not in payload:
            available = sorted(k for k, v in payload.items() if isinstance(v, list))
            raise SystemExit(
                f"--exclude-episodes-key {args.exclude_episodes_key!r} absent from "
                f"{args.exclude_episodes}; list-valued keys are {available}"
            )
        excluded = set(payload[args.exclude_episodes_key])
        group_steps = drop_excluded_episodes(group_steps, excluded)
        if not group_steps:
            raise SystemExit("the exclusion list removed every group from this split")

    episodes = set(sample_episode_subset(group_steps, args.target_groups, args.sample_seed))
    # 不相交是硬约束,不是「注意事项」—— 一旦漏了,两批就不再是独立样本。
    overlap = sorted(episodes & excluded)
    if overlap:
        raise SystemExit(
            f"{len(overlap)} sampled episodes are also in the exclusion list "
            f"({overlap[:5]}); the two batches would not be independent"
        )
    chosen_groups = sorted(g for g, (ep, _) in group_steps.items() if ep in episodes)

    plan = {
        "episode_filter_key": args.episode_filter_key,
        "groups_in_split": len(rows),
        "episodes_in_split": episodes_in_split,
        "excluded_episodes": len(excluded),
        "excluded_episodes_source": (
            None if args.exclude_episodes is None else str(args.exclude_episodes)
        ),
        "groups_after_exclusion": len(group_steps),
        "episodes_after_exclusion": len({ep for ep, _ in group_steps.values()}),
        "target_groups": args.target_groups,
        "sampled_groups": len(chosen_groups),
        "sampled_episodes": len(episodes),
        "radii": list(radii),
        "sample_seed": args.sample_seed,
        "expected_forwards": sum(
            len(
                {
                    steps
                    for radius in radii
                    if (p := arm_plan(group_steps[g][1], radius)) is not None
                    for steps in (p["base"], p["recent"], *p["sparse"].values())
                }
            )
            for g in chosen_groups
        ),
    }
    print(json.dumps({"rescue_plan": plan}, ensure_ascii=False), flush=True)
    if args.plan_only:
        return

    import torch

    shard_groups = chosen_groups
    if args.shard_count > 1:
        shard_groups = chosen_groups[args.shard_index :: args.shard_count]
        if not shard_groups:
            raise SystemExit(
                f"shard {args.shard_index}/{args.shard_count} covers no pair-group"
            )

    image_root = args.image_root or args.dataset_root

    def per_shard(path: Path) -> Path:
        if args.shard_count <= 1:
            return path
        return path.with_name(
            f"{path.stem}.shard{args.shard_index:03d}"
            f"-of-{args.shard_count:03d}{path.suffix}"
        )

    output_path = per_shard(args.output)
    raw_path = per_shard(
        args.raw_output or args.output.with_suffix(".raw.jsonl")
    )
    heartbeat_path = None if args.heartbeat is None else per_shard(args.heartbeat)

    cache_fingerprint = {
        "probe": REPORT_SCHEMA,
        "record_fields": "v2_target_spans",
        "config_sha256": sha256_of(args.config),
        "model_dir": str(args.model_dir),
        "dataset_root": str(args.dataset_root),
        "image_root": str(image_root),
        "annotations": str(args.annotations),
        "format": FROZEN_FORMAT,
        "beam_width": BEAM_WIDTH,
        "adapter": "none",
    }
    cache = SetScoreCache(
        args.score_cache,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        fingerprint=cache_fingerprint,
    )
    scorer = ArgmaxSetScorer(
        args=args, config=config, image_root=image_root, cache=cache, torch=torch
    )

    started = time.time()
    every = max(int(args.progress_every), 0)

    # note (luojiaxuan): 续跑靠 ``--score-cache``(前向结果)+ 重写 raw JSONL。raw 不做
    # 追加式续写:重启后已缓存的臂是 cache_hit,几秒就能把 raw 重新写全,比维护一份
    # 「哪些组已落 raw」的边状态更不容易出错。
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_handle = raw_path.open("w", encoding="utf-8")

    def progress(position: int, total: int) -> None:
        if position != total and (every <= 0 or position % every):
            return
        state = {
            "stage": "scoring_single_image_rescue",
            "groups_done": position,
            "groups_total": total,
            "forwards": scorer.forwards,
            "cache_hits": scorer.cache_hits,
            "max_prompt_tokens": scorer.max_prompt_tokens,
            "elapsed_seconds": round(time.time() - started, 1),
        }
        print(json.dumps({"rescue_progress": state}), flush=True)
        if heartbeat_path is not None:
            heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            heartbeat_path.write_text(
                json.dumps(state, sort_keys=True) + "\n", encoding="utf-8"
            )

    annotation_cache: dict[str, list[dict]] = {}
    written = 0
    skipped: dict[str, int] = {}

    def run_group(pair_group: str) -> int:
        nonlocal written
        point = DecisionPoint(
            rows[pair_group],
            annotations=args.annotations,
            annotation_cache=annotation_cache,
            image_root=image_root,
        )
        produced = 0
        for radius in radii:
            plan_r = arm_plan(point.current_step, radius)
            if plan_r is None:
                skipped[f"r{radius}_no_candidate"] = (
                    skipped.get(f"r{radius}_no_candidate", 0) + 1
                )
                continue
            if plan_r["base"] != point.recent(radius):
                raise SystemExit(
                    f"{pair_group}: arm_plan anchors r={radius} at {plan_r['base']} but "
                    f"the corpus decision step gives {point.recent(radius)}"
                )
            records = {
                "base": scorer.agree(point, plan_r["base"]),
                "recent": scorer.agree(point, plan_r["recent"]),
                "sparse": {
                    str(j): scorer.agree(point, steps)
                    for j, steps in plan_r["sparse"].items()
                },
            }
            raw_handle.write(
                json.dumps(
                    {
                        "schema_version": RAW_SCHEMA,
                        "pair_group": pair_group,
                        "episode": point.episode,
                        "current_step": point.current_step,
                        "r": radius,
                        "base_steps": list(plan_r["base"]),
                        "recent_steps": list(plan_r["recent"]),
                        "candidates": plan_r["candidates"],
                        "set_keys": {
                            "base": set_key(plan_r["base"]),
                            "recent": set_key(plan_r["recent"]),
                            "sparse": {
                                str(j): set_key(steps)
                                for j, steps in plan_r["sparse"].items()
                            },
                        },
                        "records": records,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            raw_handle.flush()
            written += 1
            produced += 1
        return produced

    try:
        if not args.require_cached:
            probe_point = DecisionPoint(
                rows[shard_groups[0]],
                annotations=args.annotations,
                annotation_cache=annotation_cache,
                image_root=image_root,
            )
            print(
                json.dumps(
                    {
                        "encoder_parity": scorer.verify_agreement_parity(
                            probe_point, probe_point.recent(1)
                        )
                    }
                ),
                flush=True,
            )
        for position, pair_group in enumerate(shard_groups, start=1):
            run_group(pair_group)
            progress(position, len(shard_groups))
    finally:
        raw_handle.close()
        cache.close()

    provenance = {
        "device": args.device,
        "device_name": None,
        "max_prompt_tokens": scorer.max_prompt_tokens,
        "max_total_tokens": scorer.max_total_tokens,
        "encoder_parity": scorer.mean_parity,
    }
    if scorer.runtime is not None:
        try:
            provenance["device_name"] = torch.cuda.get_device_name(
                scorer.runtime.model.device
            )
        except Exception:  # noqa: BLE001 - provenance must never break a run
            pass

    common = {
        "repository_root": str(args.repository_root),
        "config": str(args.config),
        "model_dir": str(args.model_dir),
        "dataset_root": str(args.dataset_root),
        "annotations": str(args.annotations),
        "episode_filter": str(args.episode_filter),
        "episode_filter_key": args.episode_filter_key,
        "sample_files": [
            {"path": str(path), "sha256": sha256_of(path)} for path in sample_files
        ],
        "format": FROZEN_FORMAT,
        "radii": list(radii),
        "adapter": "none (frozen policy; no injection, no checkpoint load)",
        "plan": plan,
        "cache_fingerprint": cache_fingerprint,
        "cache": cache.stats(),
        "provenance": provenance,
        "forward_passes": scorer.forwards,
        "cache_hits": scorer.cache_hits,
        "span_failures": scorer.span_failures,
        "raw_output": str(raw_path),
        "raw_blocks_written": written,
        "skipped": skipped,
        "elapsed_seconds": round(time.time() - started, 1),
    }

    if args.skip_reduction:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": SHARD_REPORT_SCHEMA,
                    "shard": f"{args.shard_index}/{args.shard_count}",
                    "pair_groups_scored": len(shard_groups),
                    **common,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "shard_complete": f"{args.shard_index}/{args.shard_count}",
                    "pair_groups": len(shard_groups),
                    "raw_blocks": written,
                    "forwards": scorer.forwards,
                    "output": str(output_path),
                }
            ),
            flush=True,
        )
        return

    report = write_report(
        load_raw([raw_path]),
        thresholds=thresholds,
        common=common,
        output_path=output_path,
    )
    print_report(report)
    print("")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
