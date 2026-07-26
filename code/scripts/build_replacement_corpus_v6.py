#!/usr/bin/env python3
"""Turn the frozen-policy replacement labels into the **six-arm** V6 training corpus.

# note (luojiaxuan): 与 v5 五臂 builder(``build_sparse_history_dataset.py``)的关系,
# 逐条说清楚,免得被当成"同一个东西加了两条臂"——
#   * **沿用**:样本字段结构(schema/sample_id/pair_group/arm_slot/adapter_mode/
#     selected_steps/selected_images/messages/target_text)、图像路径写法(相对
#     ``--dataset-root`` 的字符串,不内嵌像素)、fail-closed 风格(任何一项对不上就
#     **整组丢弃**并计入 ``rejected_reasons``,绝不静默降级)。
#   * **不沿用选点**:v5 的 sparse 集是 ``choose_sparse`` 随机抽的非相邻集合;v6 的
#     替换集来自冻结 policy 实测的 k=1 效用标签(``generate_replacement_labels.py``
#     打分 + ``decide_replacement_labels.py`` 判类)。
#   * **不沿用 renderer**:v5 是 ``sparse_single_turn``,已由 ``docs/renderer_freeze_v1.md``
#     判定出局;v6 一律走 ``official_style_sparse_multiturn``
#     (``causalcache/policy/gui_owl_sparse_multiturn.py``),且不提供任何切换开关。
#   * **不沿用负样本族**:v5 的 step_shuffled / duplicate / irrelevant 三个负样本测的是
#     "图错位 / 图重复 / 图来自别的轨迹";v6 的负样本是 **matched wrong 替换集**,测的是
#     "在同一决策点、同一预算、同一被替换位置上,换成一张**内容不相关**的旧帧"。
#
# ---------------------------------------------------------------------------
# 臂契约(arm_slot 是唯一主键,且必须与 trainer 的 SPARSE_ARM_CONTRACTS 逐列一致)
# ---------------------------------------------------------------------------
#   N0                  Recent-B      bypass  deployment_baseline
#   R0                  Recent-B      bypass  reference
#   RA                  Recent-B      active  measurement
#   S0                  oracle 替换集  bypass  measurement
#   SA                  oracle 替换集  active  positive
#   SA_neg_age_matched  age 匹配无关帧 active  negative
#
# N0 与 R0 用的是**同一批 recent 帧**,区别只在 renderer:N0 走原生
# ``build_official_messages``,R0 走冻结的 ``official_style_sparse_multiturn``。
# 两者不是同一个 prompt(Gate 1 实测 ``R_multi - N0 = +0.0261``),所以 N0 不是冗余臂,
# 它撑起两个派生量:``format_effect = R0 - N0``(我们的 renderer 相对原生的收益)与
# ``deployment_delta = SA - N0``(相对**标准做法**的总提升 —— 读者最关心的那个数)。
# 没有 N0,就只能报"相对我们自己 R0 的提升",而那不回答"相对标准做法呢"。
# ``NA``(官方 renderer + adapter active)保留不用:没有任何主张需要它,出现它多半
# 意味着有人把部署基线和负样本搞混了,故 ``RESERVED_ARM_SLOTS`` 把它钉成运行期断言。
#
# 负样本的 ``arm_slot`` 是 **trainer 的硬契约**(``SA_neg_<kind>``),不是本脚本的口味:
# ``validate_sparse_sample`` 按这个前缀核对 negative_kind,而损失里的 negatives 字典按
# ``role == "negative"`` 收集。名字不合契约不报错,只会让 L_content 与 A_n 静默消失。
# 负样本**只有 active 一行**:``A_n`` 由同一行跑 active + bypass 两次前向得到,bypass
# 孪生臂纯属浪费前向。
#
# 每组实际出几臂由**标签类别**与**有没有合格的内容对照帧**共同决定,三种形态:
#   6 臂  utility_positive 且找得到 age 匹配的无关帧
#   5 臂  utility_positive 但找不到(N0/R0/RA/S0/SA;见下面"matched wrong 集怎么造")
#   3 臂  recent_sufficient(k=0),只有 N0/R0/RA
# 判别只准读 ``has_content_control`` / ``content_control_absent_reason`` 两个显式字段,
# **不准数臂数**——单条样本根本数不出来,而数错不会报错。
#
# 两对臂的 ``messages`` 必须**逐字节相同**,只有 ``adapter_mode`` 不同。主张
# ``A_c - A_r``(差中差)完全依赖这一点:prompt 一旦有差异,差中差里就混进了 prompt
# 效应。每行带 ``messages_sha256``,组内配对不等即整组丢弃(见 ``_validate_group``),
# 且外部审计不必重新序列化就能复核。
#
# ---------------------------------------------------------------------------
# matched wrong 集怎么造(本脚本最需要被复核的设计)
# ---------------------------------------------------------------------------
# 从**同一决策点**的旧帧池里选,所以 app / 设备 / 分辨率 / 每图视觉 token 数自动匹配,
# 不需要任何显式对齐:
#   1. 同样是 k=1,且替换掉**同一个** recent 位置(``dropped_recent_step`` 与 oracle
#      集逐位相同)——于是 W 与 S 只差"恢复了哪一张旧帧"这一个自由度;
#   2. 该旧帧的 k=1 效用必须 ``< delta_recent``(与判类同一道闸、同一个阈值),
#      即"这张旧帧对当前决策不相关";
#   3. 在全部合格帧里取 ``|step - oracle_step|`` 最小的那张(age 匹配),排除 oracle
#      帧本身;并列时取更老的那张,保证确定性。
# 于是"错误"的定义是**内容不相关**,而不是"更老"或"跨 app"——后两者会把 age/域偏移
# 混进负样本,量出来的就不是"选错帧"的代价。绝不退化成随机帧。
#
# 找不到合格帧时**只丢负样本一臂**,保留 N0/R0/RA/S0/SA(五臂组),整组不丢。理由与代价:
#   * 实测代价不小:已标注样本上约 **26%** 的 ``utility_positive`` 决策点找不到合格帧,
#     而且丢掉的不是随机一批——被丢组的 oracle gain 中位数(0.053)明显**高于**保留组
#     (0.034)。物理解释:冻结 policy 认定"替换收益大"的点,往往是 Recent 窗口本身退化
#     的点,于是很多旧帧都有正收益,反而找不到"无用帧"。整组丢弃会把正例的 gain 分布
#     截掉上尾,这是对正例类的**选择效应**,不是随机损失。
#   * 为什么不改成相对门槛 ``U_wrong < U_oracle - margin``:那会**污染验收判据**。若
#     "wrong" 只是"不如 oracle"而非"内容无关",负样本本身就该有正增益,于是
#     ``|A_n| < 0.02`` 这条 gate 失败时,原因不再是 adapter 发生 common-mode 放大,
#     判据失去可证伪性,整个 DiD 框架白搭。所以"错误 = 内容不相关"的定义原样保住,
#     宁可让一部分组没有负样本臂。
#   * 每行显式带 ``has_content_control`` 与 ``content_control_absent_reason``,下游**不得**
#     靠"数臂数"推断;manifest 分别报六臂/五臂组数与两者的 oracle gain 分位数。
#   * ``rejected_reasons`` 只留给真正的硬失败(缺图、缺分数、对账不符);"没有合格
#     distractor"是数据性质,不是失败,不进拒绝计数。
#
# ---------------------------------------------------------------------------
# 每个集合的分数从哪里来(与任务书的口径差异,如实记录)
# ---------------------------------------------------------------------------
# ``labels.shard*.jsonl``(``generate_replacement_labels.py`` 的输出)**没有**逐集合
# 分数:它只存 cross-fit 两折各自的 argmax 集合与增益、in-sample 最优集合、池大小。
# 逐集合分数在同一个作业写的 **score cache**(``--score-cache`` 指向的
# ``label_cache.shard*.jsonl``)里,键是 ``"<pair_group>|<format>|<set_key>"``。
# 所以本脚本需要两个 glob:
#   * ``--raw-labels-glob``  —— 取 ``anchor_mean`` / ``recent_steps`` / ``n_old`` /
#     ``k1.crossfit_gain`` 做**溯源交叉核对**(证明 score cache 与这批标签同源);
#   * ``--set-score-glob``   —— 取每个候选替换集的两折分数,算 distractor 的 k=1 效用。
# 候选集缺分数即整组拒绝(``distractor_utility_unavailable``),不猜、不外推。
#
# distractor 的效用与 oracle 的 ``gain`` 用**同一把尺**:两折各自算增益再平均
# (``fold_balanced_gain``)。对**固定集合**而言它与 ``crossfit_gain`` 逐位同义
# (cross-fit 的两个方向此时都在评同一个集合),所以两者可以直接跟同一个
# ``delta_recent`` 比。
#
# ---------------------------------------------------------------------------
# 类别配比:不过采样
# ---------------------------------------------------------------------------
# ``utility_positive``(k=1)出六臂,没有合格内容对照帧时退成五臂(见上);
# ``recent_sufficient``(k=0)**只出 N0/R0/RA**,
# 且绝不为它伪造 oracle 集。实测自然率约 25% / 75%,本脚本原样保留:过采样正例等于教
# 模型"总是该替换",正是 DiD 的 L_cap 要压住的 common-mode 放大。manifest 里如实报告
# 交付后的最终配比(``class_fractions``)与标签侧的自然率,任何人都能对上号。
#
# 预算只做 B ∈ {1,2}:``docs/gate3_replacement_audit.md` 判定选择的价值随预算单调衰减,
# B=4 已不显著。k 只做 0/1:同一份审计判定 ``marginal_k2_over_k1`` 点估计为负。
#
# ---------------------------------------------------------------------------
# 输出布局
# ---------------------------------------------------------------------------
# ``--output-dir/{samples.jsonl,manifest.json}`` 加上指向 ``--image-root`` 的 ``part*``
# 符号链接(``sparse-v5-final`` 自己就是这么拼的)。trainer 用
# ``args.dataset_root / "samples.jsonl"`` 读样本、用同一个 root 解析图片路径,所以链接
# 必须建,否则语料只有配上原 dataset-root 才能用。图片一张都不复制。
"""

from __future__ import annotations

import argparse
import collections
import glob as globlib
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_official import build_official_messages
from causalcache.policy.gui_owl_sparse_multiturn import (
    SPARSE_MULTITURN_PROMPT_FORMAT,
    build_sparse_multiturn_messages,
)

# note (luojiaxuan): 这两个 helper 一律**复用**,不重写。``set_key`` 决定 score cache
# 的键格式,重写一份就等于给缓存键开了第二个真相来源(拼错不会报错,只会全部 miss 然后
# 整批组被拒);``rebuild_full_responses`` 是"从原始 annotation 重建完整响应 + 与 N0 的
# assistant 轮逐字节对账"的唯一实现,多轮 renderer 对每个保留步都强制要求非空完整响应。
from scripts.probe_oracle_headroom import set_key
from scripts.probe_prompt_format import rebuild_full_responses
from scripts.score_sparse_history_arms import load_sparse_samples, resolve_sample_files

SCHEMA_VERSION = "causalcache.replacement_corpus_sample.v6"
DECISION_SCHEMA_VERSION = "causalcache.replacement_label_decision.v1"
LABEL_FORMAT = SPARSE_MULTITURN_PROMPT_FORMAT
# 部署基线用**原生官方** renderer(build_official_messages),不是冻结的那个。
DEPLOYMENT_FORMAT = "official_multiturn"
REFERENCE_ARM_ID = "R0"
DEPLOYMENT_BASELINE_ARM_ID = "N0"
POSITIVE_ARM_SLOT = "SA"
# note (luojiaxuan): 负样本的 arm_slot 是 trainer 的**硬契约**:validate_sparse_sample
# 里写着 ``if slot != f"SA_neg_{kind}": raise``,而损失里的 negatives 字典是按
# ``role == "negative"`` 收的。名字不合契约不会报错,只会让 L_content 与 A_n 静默消失。
NEGATIVE_ARM_SLOT = "SA_neg_age_matched"
NEGATIVE_KIND = "age_matched"
NEGATIVE_SCALE = 1.0

# 臂契约:(arm_slot, arm_id, role, prompt_format, selection_mode, adapter_mode)
# —— 字段顺序与 v5 的 ARM_SPECS 及 trainer 的 SPARSE_ARM_CONTRACT 逐列一致。
ARM_SPECS: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("N0", "N0", "deployment_baseline", DEPLOYMENT_FORMAT, "recent", "bypass"),
    ("R0", "R0", "reference", LABEL_FORMAT, "recent", "bypass"),
    ("RA", "RA", "measurement", LABEL_FORMAT, "recent", "active"),
    ("S0", "S0", "measurement", LABEL_FORMAT, "replacement", "bypass"),
    ("SA", "SA", "positive", LABEL_FORMAT, "replacement", "active"),
    (NEGATIVE_ARM_SLOT, "SA", "negative", LABEL_FORMAT, "wrong", "active"),
)
# recent_sufficient(k=0)组只出这三臂:这类样本教的是"当前 Recent 已够用"。
# 基线臂对所有组都有意义,所以 N0 也在。
RECENT_ONLY_ARM_SLOTS: tuple[str, ...] = ("N0", "R0", "RA")
# utility_positive 但找不到合格 distractor 时的五臂:主张 SA-RA 的 DiD 仍然成立,
# 只是这一组没有内容对照臂。整组丢弃会系统性截掉高 gain 的正例,见模块 docstring。
BASE_ARM_SLOTS: tuple[str, ...] = ("N0", "R0", "RA", "S0", "SA")
CONTENT_CONTROL_ARM_SLOTS: tuple[str, ...] = (NEGATIVE_ARM_SLOT,)
# (bypass 臂, active 臂):每一对的 messages 必须逐字节相同。负样本**没有** bypass
# 孪生臂 —— A_n 由同一行跑 active + bypass 两次前向得到(trainer 的 DiD 损失里
# ``forward(slot, grad=False, adapter_mode="bypass")``),多出一条只会浪费前向。
ADAPTER_PAIRS: tuple[tuple[str, str], ...] = (("R0", "RA"), ("S0", "SA"))
# note (luojiaxuan): ``NA``(官方 renderer + adapter active)仍然保留不用:没有任何
# 主张需要它,出现它多半意味着有人把部署基线和负样本搞混了。N0 不在此列 —— 它就是
# v5 语义的部署基线臂,本语料照常产出。
RESERVED_ARM_SLOTS: tuple[str, ...] = ("NA",)

CLASS_UTILITY_POSITIVE = "utility_positive"
CLASS_RECENT_SUFFICIENT = "recent_sufficient"
LABEL_CLASSES = (CLASS_UTILITY_POSITIVE, CLASS_RECENT_SUFFICIENT)
# ``content_control_absent_reason`` 的闭集。None 表示该组带负样本臂。
CONTROL_ABSENT_RECENT_SUFFICIENT = "recent_sufficient_group"
CONTROL_ABSENT_NO_QUALIFYING = "no_qualifying_distractor"
CONTROL_ABSENT_NO_OTHER_OLD_FRAME = "no_other_old_frame"
CONTROL_ABSENT_REASONS = (
    CONTROL_ABSENT_RECENT_SUFFICIENT,
    CONTROL_ABSENT_NO_QUALIFYING,
    CONTROL_ABSENT_NO_OTHER_OLD_FRAME,
)

# 审计判定的有效预算与 k 上界,见 docs/gate3_replacement_audit.md。
SUPPORTED_BUDGETS: tuple[int, ...] = (1, 2)
SUPPORTED_K: tuple[int, ...] = (0, 1)
DEFAULT_DELTA_RECENT = 0.01
# 溯源核对的浮点容差:两个文件里的同一个数只经过 json round-trip,不该有真实误差。
PROVENANCE_TOLERANCE = 1e-9

# v5 语料里本脚本要读的臂:N0 提供 assistant 轮(重建完整响应时对账用),
# R0/S0 只用来核对 (step -> 图片路径) 的推导规则。
CORPUS_SLOTS: tuple[str, ...] = ("N0", "R0", "S0")


class GroupRejected(ValueError):
    """A group failed a fail-closed check; the whole group is dropped."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# 分数与效用
# ---------------------------------------------------------------------------
def cache_key(pair_group: str, steps: Sequence[int]) -> str:
    """The frozen scorer's cache key for one ``(group, format, set)`` triple."""
    return f"{pair_group}|{LABEL_FORMAT}|{set_key(tuple(sorted(int(s) for s in steps)))}"


def fold_balanced_gain(record: Mapping[str, Any], anchor: Mapping[str, Any]) -> float:
    """Two-fold-balanced gain of one **fixed** set over the Recent anchor.

    # note (luojiaxuan): 与 ``crossfit_gain`` 对固定集合逐位同义(cross-fit 的两个方向
    # 此时评的是同一个集合),所以 distractor 的效用与 oracle 的 ``gain`` 可以跟同一个
    # ``delta_recent`` 比。不用 ``mean`` 之差是因为 n_even != n_odd 时两者会差一点点,
    # 而"差一点点"正好会在阈值附近改变去留。
    """
    for block in (record, anchor):
        if not block.get("n_even") or not block.get("n_odd"):
            raise GroupRejected("degenerate_token_folds")
    even = record["sum_even"] / record["n_even"] - anchor["sum_even"] / anchor["n_even"]
    odd = record["sum_odd"] / record["n_odd"] - anchor["sum_odd"] / anchor["n_odd"]
    return (even + odd) / 2.0


# ---------------------------------------------------------------------------
# 替换集的分解与 matched wrong 的选取
# ---------------------------------------------------------------------------
def decompose_replacement(
    recent: Sequence[int], steps: Sequence[int]
) -> tuple[int, int]:
    """``(dropped_recent_step, restored_old_step)`` of a k=1 replacement set.

    审计把 k 的上界定在 1,所以合法的替换集 = Recent 窗口换掉**恰好一个**位置。
    多换一个或没换,都说明标签与本脚本对 k 的理解不一致,整组丢弃而不是猜。
    """
    chosen = [int(step) for step in steps]
    if len(set(chosen)) != len(chosen):
        raise GroupRejected("replacement_set_has_duplicates")
    if len(chosen) != len(recent):
        raise GroupRejected("replacement_budget_mismatch")
    dropped = sorted(set(recent) - set(chosen))
    restored = sorted(set(chosen) - set(recent))
    if len(dropped) != 1 or len(restored) != 1:
        raise GroupRejected("replacement_is_not_a_single_swap")
    return dropped[0], restored[0]


def choose_matched_wrong(
    *,
    oracle_old: int,
    old_pool: Sequence[int],
    utility: Mapping[int, float | None],
    delta_recent: float,
) -> tuple[int | None, dict[str, Any]]:
    """Pick the age-nearest **irrelevant** old frame, or report that none exists.

    返回 ``(step, diagnostics)``;``step is None`` 表示这一组没有内容对照臂,原因写在
    ``diagnostics["content_control_absent_reason"]``——调用方据此产出五臂组,**不**拒绝
    整组(见模块 docstring:整组丢弃会系统性截掉高 gain 的正例)。

    # note (luojiaxuan): 顺序是"先筛效用,再按 age 取最近",不能反过来。相邻帧往往与
    # oracle 帧内容几乎相同,先取最近再看效用的话,拿到的多半是一个同样有用的替身,
    # 那 W 臂就不再是负样本了。并列时取 step 更小(更老)的一张,只为确定性。
    #
    # 软硬两类结果必须分开:"池子里没有别的旧帧"和"合格帧一个都没有"是**数据性质**,
    # 退化成五臂;而 ``utility`` 里出现 None(候选帧的集合分数拿不到)是**硬失败**,
    # 少一个候选就无法断言"最近的合格帧"是谁,拿次近的顶上等于偷偷改了 distractor 的
    # 定义,所以仍然整组拒绝并计入 rejected_reasons。
    """
    candidates = [int(step) for step in old_pool if int(step) != int(oracle_old)]
    diagnostics: dict[str, Any] = {
        "old_pool_size": len(candidates) + 1,
        "distractor_candidates": len(candidates),
    }
    if not candidates:
        diagnostics["content_control_absent_reason"] = CONTROL_ABSENT_NO_OTHER_OLD_FRAME
        return None, diagnostics
    if any(utility.get(step) is None for step in candidates):
        raise GroupRejected("distractor_utility_unavailable")
    qualifying = [step for step in candidates if float(utility[step]) < delta_recent]
    diagnostics["distractor_qualifying"] = len(qualifying)
    diagnostics["distractor_qualifying_fraction"] = len(qualifying) / len(candidates)
    if not qualifying:
        diagnostics["content_control_absent_reason"] = CONTROL_ABSENT_NO_QUALIFYING
        return None, diagnostics
    chosen = min(qualifying, key=lambda step: (abs(step - int(oracle_old)), step))
    diagnostics.update(
        {
            "content_control_absent_reason": None,
            "wrong_utility": float(utility[chosen]),
            "wrong_age_gap": abs(chosen - int(oracle_old)),
        }
    )
    return chosen, diagnostics


def plan_group(
    *,
    current_step: int,
    budget: int,
    decision: Mapping[str, Any],
    utility: Mapping[int, float | None],
    delta_recent: float,
) -> dict[str, Any]:
    """Resolve one decided label into the arms' selections, or reject the group.

    返回 ``{"label_class", "label_k", "selections", "diagnostics",
    "has_content_control", "content_control_absent_reason"}``。``selections`` 的键是
    selection_mode(``recent`` / ``replacement`` / ``wrong``),值是 1 基 step 列表;
    没有内容对照臂时 ``wrong`` 直接缺席,而不是塞一个凑数的集合。
    """
    if budget < 1:
        raise GroupRejected("non_positive_budget")
    recent = list(range(current_step - budget, current_step))
    if recent[0] < 1:
        raise GroupRejected("insufficient_history_for_budget")
    label_class = decision.get("klass")
    if label_class not in LABEL_CLASSES:
        raise GroupRejected("unknown_label_class")
    label_k = int(decision.get("k", -1))
    if label_k not in SUPPORTED_K:
        raise GroupRejected("unsupported_k")
    old_pool = list(range(1, current_step - budget))

    if label_class == CLASS_RECENT_SUFFICIENT:
        if label_k != 0:
            raise GroupRejected("label_class_k_disagreement")
        if [int(step) for step in decision.get("steps", ())] != recent:
            raise GroupRejected("recent_steps_disagreement")
        return {
            "label_class": label_class,
            "label_k": 0,
            "selections": {"recent": recent},
            "diagnostics": {
                "old_pool_size": len(old_pool),
                "reason": decision.get("reason"),
                "gain": float(decision.get("gain", 0.0)),
            },
            "has_content_control": False,
            "content_control_absent_reason": CONTROL_ABSENT_RECENT_SUFFICIENT,
        }

    if label_k != 1:
        raise GroupRejected("label_class_k_disagreement")
    oracle = sorted(int(step) for step in decision.get("steps", ()))
    dropped, oracle_old = decompose_replacement(recent, oracle)
    if oracle_old not in old_pool:
        raise GroupRejected("oracle_step_outside_old_pool")
    wrong_old, diagnostics = choose_matched_wrong(
        oracle_old=oracle_old,
        old_pool=old_pool,
        utility=utility,
        delta_recent=delta_recent,
    )
    diagnostics.update(
        {
            "dropped_recent_step": dropped,
            "oracle_old_step": oracle_old,
            "wrong_old_step": wrong_old,
            "oracle_age": current_step - oracle_old,
            "wrong_age": None if wrong_old is None else current_step - wrong_old,
            "gain": float(decision.get("gain", 0.0)),
            "pool_size": decision.get("pool_size"),
        }
    )
    selections: dict[str, list[int]] = {"recent": recent, "replacement": oracle}
    if wrong_old is not None:
        kept = [step for step in recent if step != dropped]
        selections["wrong"] = sorted(kept + [wrong_old])
    return {
        "label_class": label_class,
        "label_k": 1,
        "selections": selections,
        "diagnostics": diagnostics,
        "has_content_control": wrong_old is not None,
        "content_control_absent_reason": diagnostics["content_control_absent_reason"],
    }


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def render_messages(
    *,
    instruction: str,
    action_texts: Sequence[str],
    full_responses: Sequence[str | None] | None,
    steps: Sequence[int],
    image_paths: Sequence[str],
    current_step: int,
    current_path: str,
    prompt_format: str = LABEL_FORMAT,
) -> list[dict[str, Any]]:
    """Render one arm; images travel as paths.

    # note (luojiaxuan): encode_sample 期望 ``messages[*].content`` 里 ``type=="image"``
    # 的部分带 ``path``(相对 dataset_root),训练时才去开图。
    #
    # 只有**两个**合法取值,且它们不可互换:历史臂一律走冻结的
    # ``official_style_sparse_multiturn``(换它必须改 docs/renderer_freeze_v1.md);
    # 部署基线臂 N0 走**原生** ``build_official_messages``,因为 deployment_delta
    # 要回答的是"相对标准做法提升多少",拿我们自己的 renderer 当基线就答非所问了。
    # Gate 1 实测两者不等价(R_multi - N0 = +0.0261),所以 N0 不是冗余臂。
    """
    if prompt_format == LABEL_FORMAT:
        messages = build_sparse_multiturn_messages(
            instruction=instruction,
            action_texts=list(action_texts),
            full_responses=None if full_responses is None else list(full_responses),
            selected_steps=list(steps),
            selected_images=[{"__path__": path} for path in image_paths],
            current_step=current_step,
            current_image={"__path__": current_path},
        )
    elif prompt_format == DEPLOYMENT_FORMAT:
        messages = build_official_messages(
            goal=instruction,
            past_action_texts=list(action_texts),
            recent_images=[{"__path__": path} for path in image_paths],
            current_image={"__path__": current_path},
            past_full_responses=(
                None if full_responses is None else list(full_responses)
            ),
        )
    else:
        raise ValueError(f"unknown prompt_format {prompt_format!r}")
    rendered: list[dict[str, Any]] = []
    for message in messages:
        content: list[dict[str, Any]] = []
        for part in message["content"]:
            if part.get("type") == "image":
                content.append({"type": "image", "path": part["image"]["__path__"]})
            else:
                content.append(dict(part))
        rendered.append({"role": message["role"], "content": content})
    return rendered


def messages_bytes(messages: Sequence[Mapping[str, Any]]) -> bytes:
    """The exact bytes a rendered prompt contributes to ``samples.jsonl``."""
    return json.dumps(messages, ensure_ascii=False).encode("utf-8")


def messages_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(messages_bytes(messages)).hexdigest()


def count_images(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for message in messages
        for part in message["content"]
        if part.get("type") == "image"
    )


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def group_id(episode: str, current_step: int, budget: int) -> str:
    """The v6 pair-group key.

    # note (luojiaxuan): v5 的 pair_group 是 ``<episode>:<step>``,而 v6 在同一个决策点
    # 上会分别产出 B=1 与 B=2 两组(两个预算的替换集不同)。沿用旧键会让两组撞在一起,
    # 而 trainer 是按 pair_group 聚合的——撞了不会报错,只会把两组的臂混成一组。
    """
    return f"{episode}:{current_step}:b{budget}"


def build_group_rows(
    *,
    episode: str,
    split: str,
    instruction: str,
    action_texts: Sequence[str],
    full_responses: Sequence[str | None],
    current_step: int,
    budget: int,
    target_text: str,
    current_image: str,
    image_path: Any,
    label_class: str,
    label_k: int,
    selections: Mapping[str, Sequence[int]],
    diagnostics: Mapping[str, Any],
    has_content_control: bool,
    content_control_absent_reason: str | None,
    source_pair_group: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble every row of one v6 group, or raise ``GroupRejected``."""
    prefix_actions = list(action_texts[: current_step - 1])
    prefix_responses = list(full_responses[: current_step - 1])
    if len(prefix_actions) != current_step - 1:
        raise GroupRejected("history_prefix_too_short")
    if len(prefix_responses) != current_step - 1 or any(
        not isinstance(response, str) or not response.strip()
        for response in prefix_responses
    ):
        # 多轮 renderer 的保留轮必须带 Action + <tool_call>,重建不出来就整组丢弃。
        raise GroupRejected("unreconstructable_history_response")

    if label_class == CLASS_UTILITY_POSITIVE:
        if has_content_control:
            arms = ARM_SPECS
            required_modes = {"recent", "replacement", "wrong"}
        else:
            # 五臂:主张 SA-RA 的 DiD 照常成立,只是这一组没有内容对照臂。
            arms = tuple(spec for spec in ARM_SPECS if spec[0] in BASE_ARM_SLOTS)
            required_modes = {"recent", "replacement"}
    elif label_class == CLASS_RECENT_SUFFICIENT:
        if has_content_control:
            raise GroupRejected("recent_sufficient_cannot_carry_content_control")
        arms = tuple(spec for spec in ARM_SPECS if spec[0] in RECENT_ONLY_ARM_SLOTS)
        required_modes = {"recent"}
    else:
        raise GroupRejected("unknown_label_class")
    if set(selections) != required_modes:
        raise GroupRejected("selection_modes_mismatch")
    if has_content_control != (content_control_absent_reason is None):
        raise GroupRejected("content_control_flag_disagreement")
    if content_control_absent_reason is not None and (
        content_control_absent_reason not in CONTROL_ABSENT_REASONS
    ):
        raise GroupRejected("unknown_content_control_absent_reason")

    steps_by_mode: dict[str, list[int]] = {}
    images_by_mode: dict[str, list[str]] = {}
    for mode, raw_steps in selections.items():
        steps = [int(step) for step in raw_steps]
        if len(steps) != budget:
            raise GroupRejected("budget_selection_mismatch")
        if any(later <= earlier for earlier, later in zip(steps, steps[1:])):
            raise GroupRejected("selection_not_strictly_increasing")
        if steps[0] < 1 or steps[-1] >= current_step:
            raise GroupRejected("selection_outside_history")
        steps_by_mode[mode] = steps
        images_by_mode[mode] = [image_path(step) for step in steps]

    # note (luojiaxuan): 渲染按 (prompt_format, selection_mode) 缓存 —— N0 与 R0 的
    # selection_mode 同为 recent 但 renderer 不同,只用 mode 当键会让 N0 拿到 R0 的
    # prompt,format_effect = R0 - N0 于是恒等于 0 而**不报错**。
    rendered: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for _slot, _arm_id, _role, arm_format, mode, _adapter_mode in arms:
        key = (arm_format, mode)
        if key in rendered:
            continue
        rendered[key] = render_messages(
            instruction=instruction,
            action_texts=prefix_actions,
            full_responses=prefix_responses,
            steps=steps_by_mode[mode],
            image_paths=images_by_mode[mode],
            current_step=current_step,
            current_path=current_image,
            prompt_format=arm_format,
        )

    identifier = group_id(episode, current_step, budget)
    common: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pair_group": identifier,
        "source_pair_group": source_pair_group or f"{episode}:{current_step}",
        "episode": episode,
        "decision_step": current_step,
        "budget": budget,
        "split": split,
        "instruction": instruction,
        "action_texts": prefix_actions,
        "current_image": current_image,
        "target_text": target_text,
        "reference_arm_id": REFERENCE_ARM_ID,
        "deployment_baseline_arm_id": DEPLOYMENT_BASELINE_ARM_ID,
        "label_class": label_class,
        "label_k": label_k,
        # note (luojiaxuan): 下游**不得**靠"数臂数"推断有没有内容对照臂,必须读这两个
        # 显式字段:六臂组 True/None,五臂组 False + 缺席原因,k=0 组 False +
        # recent_sufficient_group。数臂数在只读单条样本时根本做不到。
        "has_content_control": has_content_control,
        "content_control_absent_reason": content_control_absent_reason,
        "group_arm_slots": [spec[0] for spec in arms],
        "replacement": dict(diagnostics),
    }

    recent_window = set(range(current_step - budget, current_step))
    rows: list[dict[str, Any]] = []
    for slot, arm_id, role, arm_format, mode, adapter_mode in arms:
        messages = rendered[(arm_format, mode)]
        row = {
            **common,
            "sample_id": f"{identifier}|{slot}",
            "arm_slot": slot,
            "arm_id": arm_id,
            "role": role,
            "prompt_format": arm_format,
            "selection_mode": mode,
            "adapter_mode": adapter_mode,
            "selected_steps": list(steps_by_mode[mode]),
            "selected_images": list(images_by_mode[mode]),
            # note (luojiaxuan): 该臂的选中步里有几张落在 Recent-B 窗口内。B=1 时
            # S/负样本臂恒为 0——唯一的最近帧被换走,prompt 里一张最近帧都没有,于是
            # B=1 的 S0-R0 差的不只是"哪张历史图",还有"有没有最近帧"。分层报告必须按
            # 预算分开写,所以这个数直接落到样本行上,而不是让读结果的人自己去推。
            "recent_frames_kept": sum(
                1 for step in steps_by_mode[mode] if step in recent_window
            ),
            # note (luojiaxuan): trainer 的 history_sample_context 用
            # ``memory_config.restored_event_step_ids`` 或 v5 的 schema 常量来定 K,
            # 二者皆无就 fail-closed 报错。v6 是新 schema,所以显式带上 memory_config
            # ——语义正是"恢复了哪几步的截图",与 selected_steps 恒等(下面断言)。
            "memory_config": {"restored_event_step_ids": list(steps_by_mode[mode])},
            "messages": messages,
            "messages_sha256": messages_sha256(messages),
            "variant": f"{slot}_{mode}{budget}",
        }
        if slot == NEGATIVE_ARM_SLOT:
            row["negative_kind"] = NEGATIVE_KIND
            row["negative_scale"] = NEGATIVE_SCALE
            # trainer 的 age_matched 分支要求供体**同 episode**(与 irrelevant 相反),
            # 并按这两个来源步核对"确实换掉了 oracle 帧"。
            row["donor_episode"] = episode
            row["distractor_source_step"] = int(diagnostics["wrong_old_step"])
            row["oracle_source_step"] = int(diagnostics["oracle_old_step"])
        rows.append(row)

    _validate_group(
        rows,
        budget=budget,
        current_image=current_image,
        target_text=target_text,
        has_content_control=has_content_control,
    )
    return rows


def _validate_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    current_image: str,
    target_text: str,
    has_content_control: bool,
) -> None:
    """Fail-closed in-group checks — the hard constraints, asserted at build time."""
    by_slot = {row["arm_slot"]: row for row in rows}
    control_present = set(CONTENT_CONTROL_ARM_SLOTS) <= set(by_slot)
    if control_present != has_content_control:
        raise GroupRejected("content_control_arms_disagree_with_flag")
    if len(by_slot) != len(rows):
        raise GroupRejected("duplicate_arm_slot")
    if set(by_slot) & set(RESERVED_ARM_SLOTS):
        # NA(官方 renderer + adapter active)没有任何主张需要它,出现它多半意味着
        # 有人把部署基线和负样本搞混了。
        raise GroupRejected("reserved_arm_slot_reused")
    if DEPLOYMENT_BASELINE_ARM_ID not in by_slot:
        raise GroupRejected("deployment_baseline_arm_absent")
    for row in rows:
        if row["current_image"] != current_image:
            raise GroupRejected("current_image_disagreement")
        if row["target_text"].encode("utf-8") != target_text.encode("utf-8"):
            raise GroupRejected("target_text_disagreement")
        if len(row["selected_steps"]) != budget or len(row["selected_images"]) != budget:
            raise GroupRejected("row_budget_mismatch")
        if count_images(row["messages"]) != budget + 1:
            raise GroupRejected("image_count_mismatch")
        if row["memory_config"]["restored_event_step_ids"] != row["selected_steps"]:
            raise GroupRejected("memory_config_disagreement")
        if row["messages_sha256"] != messages_sha256(row["messages"]):
            raise GroupRejected("messages_digest_disagreement")
    for bypass_slot, active_slot in ADAPTER_PAIRS:
        if bypass_slot not in by_slot:
            continue
        if active_slot not in by_slot:
            raise GroupRejected("adapter_pair_incomplete")
        bypass, active = by_slot[bypass_slot], by_slot[active_slot]
        if messages_bytes(bypass["messages"]) != messages_bytes(active["messages"]):
            # 主张 A_c - A_r 依赖这一点:prompt 有差异,差中差就混进了 prompt 效应。
            raise GroupRejected("paired_arms_render_differently")
        if (bypass["adapter_mode"], active["adapter_mode"]) != ("bypass", "active"):
            raise GroupRejected("paired_arms_share_adapter_mode")
    # note (luojiaxuan): 反向断言 —— N0 与 R0 是**同一批 recent 帧**,只有 renderer 不同。
    # 若两者逐字节相同,说明冻结 renderer 没生效(或 recent 帧被当成别的模式渲染了),
    # 而 format_effect = R0 - N0 会安静地恒等于 0。这条比"N0 等于官方渲染"更能抓错:
    # 后者在两个 renderer 意外相同时照样成立。
    if messages_bytes(by_slot["N0"]["messages"]) == messages_bytes(
        by_slot["R0"]["messages"]
    ):
        raise GroupRejected("deployment_baseline_equals_reference")
    if "S0" in by_slot:
        if messages_bytes(by_slot["R0"]["messages"]) == messages_bytes(
            by_slot["S0"]["messages"]
        ):
            raise GroupRejected("replacement_equals_recent")
        if NEGATIVE_ARM_SLOT in by_slot and messages_bytes(
            by_slot["S0"]["messages"]
        ) == messages_bytes(by_slot[NEGATIVE_ARM_SLOT]["messages"]):
            raise GroupRejected("wrong_equals_replacement")


# ---------------------------------------------------------------------------
# 纯内存入口(复核者与单测用,不碰磁盘)
# ---------------------------------------------------------------------------
_HISTORY_IMAGE_TEMPLATE = "{image_dir}/obs-{index:03d}.png"
# 合成历史响应用的动作:只保证**格式**与 target_text 同函数生成,不声称是该步的真实
# 动作。生产路径 main() 始终传由 annotation 重建并与 N0 assistant 轮对过账的响应。
_SYNTHETIC_HISTORY_ARGUMENTS = {"action": "system_button", "button": "Home"}


def episode_image_path(image_dir: str, step: int) -> str:
    """1-based step -> the on-disk relative path used by every arm."""
    return _HISTORY_IMAGE_TEMPLATE.format(image_dir=image_dir, index=step - 1)


def build_replacement_group_samples(
    *,
    episode: str,
    decision_step: int,
    instruction: str,
    action_texts: Sequence[str],
    target_text: str,
    budget: int,
    decision: Mapping[str, Any],
    utility: Mapping[int, float | None] | None = None,
    delta_recent: float = DEFAULT_DELTA_RECENT,
    split: str = "train",
    full_responses: Sequence[str | None] | None = None,
    image_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Build one complete v6 group from an in-memory decision + utility table.

    ``decision`` is one ``decide_replacement_labels.py`` budget block. ``utility``
    maps a candidate old step to its k=1 utility (``None`` = unavailable, which
    rejects the group). ``full_responses`` may stay ``None`` in tests — history
    responses are then synthesized with the *same* formatter as ``target_text``.
    Raises ``GroupRejected`` exactly where the production path does.
    """
    from scripts.build_sparse_history_dataset import official_response

    directory = image_dir or f"images/{episode}"
    if full_responses is None:
        full_responses = [
            official_response(text, dict(_SYNTHETIC_HISTORY_ARGUMENTS))
            for text in action_texts
        ]
    plan = plan_group(
        current_step=decision_step,
        budget=budget,
        decision=decision,
        utility=utility or {},
        delta_recent=delta_recent,
    )
    return build_group_rows(
        episode=episode,
        split=split,
        instruction=instruction,
        action_texts=list(action_texts),
        full_responses=list(full_responses),
        current_step=decision_step,
        budget=budget,
        target_text=target_text,
        current_image=episode_image_path(directory, decision_step),
        image_path=lambda step: episode_image_path(directory, step),
        label_class=plan["label_class"],
        label_k=plan["label_k"],
        selections=plan["selections"],
        diagnostics=plan["diagnostics"],
        has_content_control=plan["has_content_control"],
        content_control_absent_reason=plan["content_control_absent_reason"],
    )


# ---------------------------------------------------------------------------
# 输入装载
# ---------------------------------------------------------------------------
def load_decided_labels(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read ``decide_replacement_labels.py`` output: summary line + decision rows."""
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if summary is None and "summary" in record:
                summary = record["summary"]
                continue
            key = record["pair_group"]
            if key in seen:
                raise SystemExit(f"{path}: pair-group {key!r} appears twice")
            seen.add(key)
            rows.append(record)
    if summary is None:
        raise SystemExit(f"{path} has no summary line; is it a decided-labels file?")
    if summary.get("schema_version") != DECISION_SCHEMA_VERSION:
        raise SystemExit(
            f"{path} declares schema {summary.get('schema_version')!r}, expected "
            f"{DECISION_SCHEMA_VERSION}"
        )
    if not rows:
        raise SystemExit(f"{path} carries no decision rows")
    return summary, rows


def load_raw_labels(patterns: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Merge ``generate_replacement_labels.py`` shards, refusing any overlap.

    # note (luojiaxuan): 标签作业仍在跑时最后一行可能是残行(正在写 / 被 SIGKILL 掐断),
    # 只容忍**没有换行结尾的最后一行**;中间的坏行是真损坏,静默跳过等于凭空少掉一批组。
    """
    merged: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        for name in sorted(globlib.glob(pattern)):
            with open(name, encoding="utf-8") as handle:
                lines = handle.readlines()
            for position, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if position == len(lines) and not line.endswith("\n"):
                        break
                    raise SystemExit(f"{name}:{position} is not JSON")
                key = record["pair_group"]
                if key in merged:
                    raise SystemExit(
                        f"pair-group {key!r} appears in two raw-label files; "
                        "sharding is supposed to partition, so a stale file is "
                        "being globbed alongside a fresh one"
                    )
                merged[key] = record
    if not merged:
        raise SystemExit(f"no raw label rows matched {list(patterns)}")
    return merged


def load_set_scores(patterns: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Merge the frozen scorer's per-set cache shards.

    # note (luojiaxuan): 分片本该分区,但合并文件与分片同时被 glob 到是常事,所以重复
    # 键**值相同**时放行、不同则报错——后者意味着两批分数来自不同的模型/语料/配置,
    # 静默合并会让"哪张旧帧无用"这个判定悄悄换掉依据。残尾行(作业仍在写)允许跳过。
    """
    merged: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        for name in sorted(globlib.glob(pattern)):
            with open(name, encoding="utf-8") as handle:
                lines = handle.readlines()
            for position, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if position == len(lines) and not line.endswith("\n"):
                        break  # 正在写入的残行
                    raise SystemExit(f"{name}:{position} is not JSON")
                if "cache_key" not in record:
                    continue  # cache_fingerprint 行
                key = record["cache_key"]
                value = record["value"]
                if key in merged and merged[key] != value:
                    raise SystemExit(
                        f"score cache key {key!r} carries two different values; the "
                        "globbed files were produced under different scoring setups"
                    )
                merged[key] = value
    if not merged:
        raise SystemExit(f"no per-set scores matched {list(patterns)}")
    return merged


def index_corpus_rows(
    samples: Iterable[Mapping[str, Any]], wanted: set[str]
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Index the v5 corpus rows this builder reads, keyed by v5 pair_group."""
    index: dict[str, dict[str, Mapping[str, Any]]] = {}
    for sample in samples:
        key = sample.get("pair_group")
        if key not in wanted:
            continue
        slot = sample.get("arm_slot")
        if slot not in CORPUS_SLOTS:
            continue
        bucket = index.setdefault(key, {})
        if slot in bucket:
            raise SystemExit(f"pair-group {key!r} carries two {slot!r} rows")
        bucket[slot] = sample
    return index


# ---------------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------------
def _decision_context(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    annotations: Path,
    annotation_cache: dict[str, list[dict]],
) -> dict[str, Any]:
    """Everything one decision point contributes, or ``GroupRejected``."""
    missing = [slot for slot in CORPUS_SLOTS if slot not in rows]
    if missing:
        raise GroupRejected("corpus_group_incomplete")
    n0 = rows["N0"]
    episode = str(n0["episode"])
    current_step = int(n0["decision_step"])
    current_image = str(n0["current_image"])
    image_dir = str(Path(current_image).parent)
    if Path(current_image).name != f"obs-{current_step - 1:03d}.png":
        raise GroupRejected("image_naming_convention_violated")
    if Path(image_dir).name != episode:
        raise GroupRejected("image_directory_is_not_the_episode")

    def step_image(step: int) -> str:
        return episode_image_path(image_dir, int(step))

    # 推导规则必须与语料里现成的 (step, path) 对逐条一致,否则历史图会指错帧。
    for slot in CORPUS_SLOTS:
        row = rows[slot]
        for step, path in zip(row["selected_steps"], row["selected_images"]):
            if path != step_image(int(step)):
                raise GroupRejected("image_path_derivation_mismatch")
    try:
        full_responses = rebuild_full_responses(
            dict(rows), annotations, annotation_cache
        )
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        raise GroupRejected("unreconstructable_history_response") from error
    return {
        "episode": episode,
        "current_step": current_step,
        "current_image": current_image,
        "image_dir": image_dir,
        "step_image": step_image,
        "instruction": str(n0["instruction"]),
        "action_texts": list(n0["action_texts"]),
        "target_text": str(n0["target_text"]),
        "split": str(n0["split"]),
        "full_responses": full_responses,
    }


def _verify_label_provenance(
    *,
    pair_group: str,
    budget: int,
    recent: Sequence[int],
    current_step: int,
    raw_record: Mapping[str, Any] | None,
    decision: Mapping[str, Any],
    scores: Mapping[str, Mapping[str, Any]],
    old_pool_size: int,
) -> Mapping[str, Any] | None:
    """Tie the decided labels, the raw labels and the score cache to one another.

    返回 Recent 锚点的分数记录(``utility_positive`` 组必有;``recent_sufficient`` 组
    不需要逐集合分数,返回 ``None``)。

    # note (luojiaxuan): 这一层不是形式主义。distractor 的"无用"判定完全来自 score
    # cache,而 cache 是按 ``<pair_group>|<format>|<set_key>`` 寻址的裸字典——换一个模型
    # 或换一份语料重跑,键**一模一样**,值却是另一批数。所以用 ``anchor_mean``(标签文件
    # 里现成的)与 cache 里 Recent 集合的 mean 对账:对不上就说明 glob 到的分数不是产出
    # 这批标签的那一份,整组拒绝。
    """
    if raw_record is None:
        raise GroupRejected("raw_label_missing")
    if raw_record.get("format") != LABEL_FORMAT:
        raise GroupRejected("raw_label_format_mismatch")
    if int(raw_record.get("current_step", -1)) != current_step:
        raise GroupRejected("raw_label_step_mismatch")
    block = (raw_record.get("budgets") or {}).get(str(budget))
    if block is None:
        raise GroupRejected("raw_label_budget_missing")
    if [int(step) for step in block.get("recent_steps", ())] != list(recent):
        raise GroupRejected("recent_window_disagreement")
    if decision.get("klass") != CLASS_UTILITY_POSITIVE:
        return None
    if int(block.get("n_old", -1)) != old_pool_size:
        raise GroupRejected("old_pool_disagreement")
    k1 = block.get("k1")
    if not k1:
        raise GroupRejected("raw_label_k1_missing")
    if abs(float(k1["crossfit_gain"]) - float(decision["gain"])) > PROVENANCE_TOLERANCE:
        raise GroupRejected("label_gain_disagreement")
    anchor = scores.get(cache_key(pair_group, recent))
    if anchor is None:
        raise GroupRejected("anchor_score_missing")
    if abs(float(anchor["mean"]) - float(block["anchor_mean"])) > PROVENANCE_TOLERANCE:
        raise GroupRejected("score_cache_provenance_mismatch")
    return anchor


def _distractor_utilities(
    *,
    pair_group: str,
    recent: Sequence[int],
    dropped: int,
    old_pool: Sequence[int],
    anchor: Mapping[str, Any],
    scores: Mapping[str, Mapping[str, Any]],
) -> dict[int, float | None]:
    """k=1 utility of every candidate old frame at the oracle's drop position."""
    kept = [step for step in recent if step != dropped]
    utilities: dict[int, float | None] = {}
    for old in old_pool:
        record = scores.get(cache_key(pair_group, sorted(kept + [old])))
        utilities[old] = None if record is None else fold_balanced_gain(record, anchor)
    return utilities


def build_corpus(
    *,
    decided_rows: Sequence[Mapping[str, Any]],
    corpus_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
    raw_labels: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    annotations: Path,
    budgets: Sequence[int],
    delta_recent: float,
    image_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join labels + v5 corpus + annotations into v6 rows, counting every rejection.

    ``image_root`` is only used to assert that every referenced screenshot exists;
    pass ``None`` in tests that build rows without a filesystem.
    """
    rows: list[dict[str, Any]] = []
    rejected: collections.Counter = collections.Counter()
    per_class: collections.Counter = collections.Counter()
    per_budget_class: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    per_split: collections.Counter = collections.Counter()
    rows_per_group: collections.Counter = collections.Counter()
    control_absent_reasons: collections.Counter = collections.Counter()
    recent_kept: dict[str, dict[str, collections.Counter]] = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter)
    )
    annotation_cache: dict[str, list[dict]] = {}
    budgets_absent = 0
    decisions_seen = 0
    age_gaps: list[int] = []
    wrong_utilities: list[float] = []
    qualifying_fractions: list[float] = []
    six_arm_gains: list[float] = []
    five_arm_gains: list[float] = []

    for decided in sorted(decided_rows, key=lambda record: record["pair_group"]):
        pair_group = decided["pair_group"]
        blocks = decided.get("budgets") or {}
        present = [budget for budget in budgets if str(budget) in blocks]
        budgets_absent += len(budgets) - len(present)
        decisions_seen += len(present)
        if not present:
            continue

        def reject_all(reason: str) -> None:
            for _ in present:
                rejected[reason] += 1

        corpus_rows = corpus_index.get(pair_group)
        if corpus_rows is None:
            reject_all("corpus_group_missing")
            continue
        try:
            context = _decision_context(
                corpus_rows, annotations=annotations, annotation_cache=annotation_cache
            )
        except GroupRejected as error:
            reject_all(error.reason)
            continue
        if int(decided["current_step"]) != context["current_step"]:
            reject_all("decision_step_disagreement")
            continue
        if decided.get("format") != LABEL_FORMAT:
            reject_all("decided_label_format_mismatch")
            continue

        for budget in present:
            decision = blocks[str(budget)]
            current_step = context["current_step"]
            recent = list(range(current_step - budget, current_step))
            old_pool = list(range(1, current_step - budget))
            try:
                anchor = _verify_label_provenance(
                    pair_group=pair_group,
                    budget=budget,
                    recent=recent,
                    current_step=current_step,
                    raw_record=raw_labels.get(pair_group),
                    decision=decision,
                    scores=scores,
                    old_pool_size=len(old_pool),
                )
                utility: dict[int, float | None] = {}
                if anchor is not None:
                    dropped, _oracle_old = decompose_replacement(
                        recent, [int(step) for step in decision.get("steps", ())]
                    )
                    utility = _distractor_utilities(
                        pair_group=pair_group,
                        recent=recent,
                        dropped=dropped,
                        old_pool=old_pool,
                        anchor=anchor,
                        scores=scores,
                    )
                plan = plan_group(
                    current_step=current_step,
                    budget=budget,
                    decision=decision,
                    utility=utility,
                    delta_recent=delta_recent,
                )
                group_rows = build_group_rows(
                    episode=context["episode"],
                    split=context["split"],
                    instruction=context["instruction"],
                    action_texts=context["action_texts"],
                    full_responses=context["full_responses"],
                    current_step=current_step,
                    budget=budget,
                    target_text=context["target_text"],
                    current_image=context["current_image"],
                    image_path=context["step_image"],
                    label_class=plan["label_class"],
                    label_k=plan["label_k"],
                    selections=plan["selections"],
                    diagnostics=plan["diagnostics"],
                    has_content_control=plan["has_content_control"],
                    content_control_absent_reason=plan["content_control_absent_reason"],
                    source_pair_group=pair_group,
                )
            except GroupRejected as error:
                rejected[error.reason] += 1
                continue
            except ValueError as error:
                rejected[f"render_error:{str(error)[:64]}"] += 1
                continue
            if image_root is not None:
                absent = [
                    path
                    for row in group_rows
                    for path in [*row["selected_images"], row["current_image"]]
                    if not (image_root / path).is_file()
                ]
                if absent:
                    rejected["missing_image_file"] += 1
                    continue
            rows.extend(group_rows)
            label_class = plan["label_class"]
            per_class[label_class] += 1
            per_budget_class[str(budget)][label_class] += 1
            per_split[context["split"]] += 1
            rows_per_group[str(len(group_rows))] += 1
            if plan["content_control_absent_reason"] is not None:
                control_absent_reasons[plan["content_control_absent_reason"]] += 1
            for row in group_rows:
                recent_kept[str(budget)][row["arm_slot"]][
                    str(row["recent_frames_kept"])
                ] += 1
            diagnostics = plan["diagnostics"]
            if label_class == CLASS_UTILITY_POSITIVE:
                if plan["has_content_control"]:
                    six_arm_gains.append(float(diagnostics["gain"]))
                    age_gaps.append(int(diagnostics["wrong_age_gap"]))
                    wrong_utilities.append(float(diagnostics["wrong_utility"]))
                    qualifying_fractions.append(
                        float(diagnostics["distractor_qualifying_fraction"])
                    )
                else:
                    five_arm_gains.append(float(diagnostics["gain"]))

    groups = sum(per_class.values())
    if groups + sum(rejected.values()) != decisions_seen:
        raise SystemExit(
            "rejection accounting does not reconcile: "
            f"{groups} kept + {sum(rejected.values())} rejected != "
            f"{decisions_seen} decisions seen"
        )
    natural_positive = per_class[CLASS_UTILITY_POSITIVE] / groups if groups else 0.0
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "prompt_format": LABEL_FORMAT,
        "renderer_freeze": "docs/renderer_freeze_v1.md",
        "label_audit": "docs/gate3_replacement_audit.md",
        "arms": [spec[0] for spec in ARM_SPECS],
        "arms_by_class": {
            f"{CLASS_UTILITY_POSITIVE}_with_content_control": [
                spec[0] for spec in ARM_SPECS
            ],
            f"{CLASS_UTILITY_POSITIVE}_without_content_control": list(BASE_ARM_SLOTS),
            CLASS_RECENT_SUFFICIENT: list(RECENT_ONLY_ARM_SLOTS),
        },
        "reference_arm_id": REFERENCE_ARM_ID,
        # N0 与 v5 语义逐字段相同(原生 build_official_messages + recent + bypass),
        # 所以 format_effect = R0 - N0 与 deployment_delta = SA - N0 在本语料上可算。
        "deployment_baseline_arm_id": DEPLOYMENT_BASELINE_ARM_ID,
        "budgets": list(budgets),
        "delta_recent": delta_recent,
        "decisions_seen": decisions_seen,
        "budget_blocks_absent": budgets_absent,
        "groups": groups,
        "samples": len(rows),
        "groups_by_class": dict(sorted(per_class.items())),
        "class_fractions": {
            name: count / groups for name, count in sorted(per_class.items())
        } if groups else {},
        "positive_fraction_delivered": natural_positive,
        # 不过采样:自然分布本身就是 selector STOP 要学的东西。
        "class_balancing": "none (natural label rate preserved)",
        "oversampled": False,
        "per_budget": {b: dict(c) for b, c in sorted(per_budget_class.items())},
        "groups_by_split": dict(sorted(per_split.items())),
        "rows_per_group": dict(sorted(rows_per_group.items())),
        "distractor": {
            "age_gap": _quantiles(age_gaps),
            "adjacent_fraction": (
                sum(1 for gap in age_gaps if gap == 1) / len(age_gaps)
                if age_gaps
                else 0.0
            ),
            "wrong_utility": _quantiles(wrong_utilities),
            "qualifying_fraction_of_old_pool": _quantiles(qualifying_fractions),
        },
        # note (luojiaxuan): 六臂组与五臂组的 oracle gain **必须分开报**。五臂组不是
        # 随机缺失:找不到"无用旧帧"的多半是 Recent 窗口本身退化、很多旧帧都有正收益的
        # 决策点,它们的 gain 系统性偏高。合并成一个分布会把这个选择效应藏起来。
        "oracle_gain_by_arms": {
            "six_arm": _quantiles(six_arm_gains),
            "five_arm": _quantiles(five_arm_gains),
            "all_positive": _quantiles(six_arm_gains + five_arm_gains),
        },
        "recent_frames_kept": {
            "note": (
                "每行的 recent_frames_kept 是该臂选中的步里落在 Recent-B 窗口内的张数。"
                "B=1 时 S 臂与负样本臂恒为 0:唯一的最近帧被换走,prompt 里一张最近帧都没有,"
                "所以 B=1 的 S0-R0 差的不只是'哪张历史图',还有'有没有最近帧'。"
                "分层报告必须按预算分开写,不要把 B=1 与 B=2 合并成一个 selection effect。"
            ),
            "by_budget": {
                budget_key: {
                    slot: dict(sorted(counts.items()))
                    for slot, counts in sorted(slots.items())
                }
                for budget_key, slots in sorted(recent_kept.items())
            },
            "replacement_arm_rows_without_any_recent_frame": sum(
                1
                for row in rows
                if row["selection_mode"] in ("replacement", "wrong")
                and row["recent_frames_kept"] == 0
            ),
        },
        "rejected_groups": sum(rejected.values()),
        "rejected_reasons": dict(sorted(rejected.items())),
    }
    positives = per_class[CLASS_UTILITY_POSITIVE]
    six_arm_groups = len(six_arm_gains)
    manifest["content_control"] = {
        "negative_arm_slot": NEGATIVE_ARM_SLOT,
        "negative_kind": NEGATIVE_KIND,
        "six_arm_groups": six_arm_groups,
        "five_arm_groups": len(five_arm_gains),
        "recent_sufficient_groups": per_class[CLASS_RECENT_SUFFICIENT],
        "six_arm_fraction_of_positives": (
            six_arm_groups / positives if positives else 0.0
        ),
        "absent_reasons": dict(sorted(control_absent_reasons.items())),
        "policy": (
            f"no qualifying content control -> drop {NEGATIVE_ARM_SLOT} only and keep "
            "N0/R0/RA/S0/SA; the group is never dropped for this, and it is never "
            "counted as a rejection. Read has_content_control on the sample row, "
            "never the arm count."
        ),
    }
    return rows, manifest


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    """n / p10 / p50 / p90 / max, in the same style as decide_replacement_labels."""
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}
    return {
        "n": len(ordered),
        "p10": ordered[int(0.10 * len(ordered))],
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[int(0.90 * len(ordered))],
        "max": ordered[-1],
    }


def link_image_roots(
    paths: Iterable[str], *, image_root: Path, output_dir: Path
) -> list[str]:
    """Symlink the top-level image directories the corpus references.

    # note (luojiaxuan): trainer 用 ``args.dataset_root`` 同时解析 samples.jsonl 与图片
    # 路径,而 v6 一张图都不复制(v5 语料 200GB 量级)。所以输出目录里给每个被引用的
    # 顶层目录(实测是 ``part0``..``part63``)建一个指向 --image-root 的符号链接——
    # ``sparse-v5-final`` 自己就是这么拼出来的。
    """
    linked: list[str] = []
    for name in sorted({Path(path).parts[0] for path in paths}):
        source = image_root / name
        target = output_dir / name
        if target.exists() or target.is_symlink():
            continue
        if not source.exists():
            raise SystemExit(f"image root {image_root} has no {name!r} to link")
        os.symlink(os.path.realpath(source), target)
        linked.append(name)
    return linked


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decided-labels", type=Path, required=True)
    parser.add_argument("--raw-labels-glob", action="append", required=True)
    parser.add_argument(
        "--set-score-glob",
        action="append",
        required=True,
        help=(
            "glob for the frozen scorer's per-set cache (label_cache.shard*.jsonl); "
            "labels.jsonl does NOT carry per-set scores, only the argmax sets"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--delta-recent",
        type=float,
        default=None,
        help="default: the value recorded in the decided-labels summary",
    )
    parser.add_argument("--budgets", default=",".join(str(b) for b in SUPPORTED_BUDGETS))
    parser.add_argument(
        "--link-images", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    budgets = [int(part) for part in args.budgets.split(",") if part.strip()]
    unsupported = sorted(set(budgets) - set(SUPPORTED_BUDGETS))
    if not budgets or unsupported:
        raise SystemExit(
            f"--budgets must be a subset of {list(SUPPORTED_BUDGETS)}; "
            "docs/gate3_replacement_audit.md measured selection value decaying to "
            f"nothing by B=4, so {unsupported} carries no learnable signal"
        )

    summary, decided_rows = load_decided_labels(args.decided_labels)
    delta_recent = (
        float(summary["delta_recent"]) if args.delta_recent is None else args.delta_recent
    )
    if abs(delta_recent - float(summary["delta_recent"])) > PROVENANCE_TOLERANCE:
        raise SystemExit(
            f"--delta-recent {delta_recent} differs from the threshold the labels were "
            f"decided under ({summary['delta_recent']}); the 'wrong' frames would then "
            "be defined by a different bar than the positive class"
        )
    if not summary.get("require_both_folds", False):
        raise SystemExit(
            "the decided labels were produced with --no-require-both-folds; the "
            "positive class would then include single-fold flukes"
        )

    raw_labels = load_raw_labels(args.raw_labels_glob)
    scores = load_set_scores(args.set_score_glob)
    image_root = args.image_root or args.dataset_root
    wanted = {record["pair_group"] for record in decided_rows}
    corpus_index = index_corpus_rows(
        load_sparse_samples(resolve_sample_files(args.dataset_root)), wanted
    )

    rows, manifest = build_corpus(
        decided_rows=decided_rows,
        corpus_index=corpus_index,
        raw_labels=raw_labels,
        scores=scores,
        annotations=args.annotations,
        budgets=budgets,
        delta_recent=delta_recent,
        image_root=image_root,
    )
    if not rows:
        raise SystemExit(
            "every group was rejected; see the manifest's rejected_reasons before "
            "loosening anything"
        )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    linked: list[str] = []
    if args.link_images:
        linked = link_image_roots(
            (row["current_image"] for row in rows),
            image_root=image_root,
            output_dir=output,
        )
    manifest["inputs"] = {
        "decided_labels": str(args.decided_labels),
        "decided_labels_sha256": sha256_of(args.decided_labels),
        "raw_labels_glob": list(args.raw_labels_glob),
        "raw_label_groups": len(raw_labels),
        "set_score_glob": list(args.set_score_glob),
        "set_scores": len(scores),
        "dataset_root": str(args.dataset_root),
        "image_root": str(image_root),
        "annotations": str(args.annotations),
        "corpus_groups_indexed": len(corpus_index),
        "linked_image_roots": linked,
    }
    manifest["label_decision_summary"] = summary
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
