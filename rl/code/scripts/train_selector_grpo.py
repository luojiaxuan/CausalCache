#!/usr/bin/env python3
"""Phase 3:selector-only GRPO 训练器(路线 `rl/docs/agentic_memory_rl_roadmap_20260813.md` §4)。

# note (luojiaxuan): 本文件只做一件事 —— **固定 memory-aware executor,只训 selector**。
# GUI-Owl(executor)完全不在计算图里:它在 rollout 阶段已经把 terminal success
# 烤进 JSONL,训练侧只读那个 0/1。可学参数只有 selector。
#
# 目标函数(与路线 §4 逐字对应):
#
#   同一 (task_id, group_id) 的 G 条 rollout 构成一组,R_i = 1[success_i]:
#       A_i = (R_i - mean_G(R)) / (std_G(R) + eps)                 # 组内相对优势
#   逐步重要性比(logπ_old 来自 rollout 时刻的 JSONL,是常数):
#       r_{i,t} = exp( logπ_θ(S_t | x_{i,t}) - logπ_old(S_t | x_{i,t}) )
#   PPO-clip 代理:
#       L_pg  = - mean_i mean_t [ min( r·A_i , clip(r, 1-ε, 1+ε)·A_i ) ]
#       L     = L_pg - β_ent · H(π_θ) [ + β_kl · KL(π_θ ‖ π_ref) ]
#   H 与 KL 都在**整个动作空间** C(n, B) 上精确求和(B=2、n≤max_steps,
#   最多几百个子集,不需要采样估计)。π_ref 默认关闭;开启时 = 初始 checkpoint。
#
# **组内 reward 全同(std=0)时该组不产生任何梯度**,本实现显式跳过并计数
# (`groups_degenerate_all0/all1`)。路线 §4 明确点名"若大量 group 全 0/全 1"
# 是要靠 curriculum / SFT 初始能力 / group size / rollout 数量来解决的问题,
# **不许引入 milestone reward** —— 所以这个比例必须始终可观测,而不是被
# 静默吃掉。`--dry-run` 不加载任何模型就能把这张表打出来。
#
# 路线 §9 禁止清单在本文件的落地(逐条自查):
#   * milestone reward / step-level 正确性:奖励**只**来自 JSONL 顶层 `success`,
#     本文件不读任何 per-step reward/正确性字段,也不提供开关去读;
#   * gold 帧监督:不存在 gold subset 标签通路,loss 里没有任何交叉熵项;
#   * pass-1 draft:不构造、不读取 draft 动作;
#   * B=1 探针:动作空间恒为 C(n, b),b 由 rollout 记录的 |S| 决定,
#     不做逐帧边际打分;
#   * Gumbel-softmax / straight-through 软选帧:选帧在 rollout 侧就已经采样成
#     离散事实,训练侧只重算它的 log-prob,梯度**不穿过图片选择、也不穿过环境**。
#
# 与并行开发中的 `selector_model.py` / `features.py` 的集成缝见
# `_resolve_*` 三个函数的 docstring:全部走"按签名过滤 kwargs"的鸭子类型探测,
# 并在缺件时打印它找过哪些入口,便于对齐而不是静默走错分支。
"""

from __future__ import annotations

import argparse
import glob
import importlib
import inspect
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# note (luojiaxuan): 脚本既可能被 `python3 rl/code/scripts/train_selector_grpo.py`
# 直接执行,也可能带 PYTHONPATH=rl/code:code 运行;这里把两条包根都顶到 sys.path
# (rl/code 提供 causalcache_agentic,仓库根 code 提供 causalcache),保证前一种
# 用法也能 import。
_RL_CODE = Path(__file__).resolve().parent.parent
_REPO_CODE = _RL_CODE.parent.parent / "code"
for _p in (str(_RL_CODE), str(_REPO_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from causalcache_agentic.policy_io import (  # noqa: E402
    DEFAULT_BUDGET,
    HistoryFrameBank,
    SelectorState,
)

DEFAULT_SELECTOR_MODULE = "causalcache_agentic.selector_model"
DEFAULT_FEATURES_MODULE = "causalcache_agentic.features"
CKPT_FORMAT = "causalcache_agentic.selector_grpo.v1"

# note (luojiaxuan): 下面三张表与 `scripts/rollout_selector.py` 的同名常量**逐字
# 相同**。两侧必须发现同一个 selector 类、同一个 state builder、同一套 kwargs 别名
# ——否则训练端重算的 log π 来自另一个分布,ratio 全错而且没人会察觉。改一侧就要
# 改另一侧,`--logprob-check` 是这条纪律的自动哨兵。
_SELECTOR_MODULES = (DEFAULT_SELECTOR_MODULE, "causalcache_agentic.selector",
                     "causalcache_agentic.selector_policy",
                     "causalcache_agentic.subset_selector")
_SELECTOR_FACTORIES = ("load_selector", "load_selector_bundle", "build_selector",
                       "make_selector", "load_checkpoint", "load_from_checkpoint")
_SELECTOR_CLASSES = ("SubsetSelectorPolicy", "SubsetSelector", "SelectorPolicyNet",
                     "SelectorNet", "FrameSelector", "SubsetScorer")
# 非 learned 臂(recent/random)与 greedy 评测跑的 log π 恒为 0 或点质量,
# 结构上没有可训信号,训练端一律剔除并计数,不允许混进优势里。
NON_TRAINABLE_ARMS = ("recent", "random")

# note (luojiaxuan): 动作空间是 C(n, b);b=2 时 n=64 就已经 2016 个子集。真实
# rollout 的 n ≤ max_steps-1(默认 23 → 253),超过这个量级说明 JSONL 的
# candidates 记账出了问题,宁可 fail-loud 也不要静默烧显存。
MAX_SUBSETS_HARD = 20000


# =============================================================== CLI

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3 selector-only GRPO(executor 冻结且不在计算图内)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- 数据 ----
    g = p.add_argument_group("数据")
    g.add_argument("--rollouts", nargs="+", required=True,
                   help="rollout_selector.py 产出的 JSONL(可多个,支持 glob)")
    g.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                   help="预算 B;每步实际基数以 JSONL 的 |chosen_subset| 为准并交叉校验")
    g.add_argument("--arm", default="",
                   help="只训练该 arm 的 rollout(空 = 全收 learned 臂)")
    g.add_argument("--limit-groups", type=int, default=0,
                   help="只取前 N 个组(冒烟用;0=关)")
    g.add_argument("--min-group-size", type=int, default=2,
                   help="组内 rollout 数少于该值直接跳过(std 无定义)")
    g.add_argument("--temperature", type=float, default=0.0,
                   help="重算 log π 的温度。0 = **从每行 JSONL 的 `temperature` 读**"
                        "(正确做法:必须与采样时刻同温,否则 ratio 全错);"
                        ">0 = 强制覆盖,仅供排障")
    g.add_argument("--sampling-mode", default="auto",
                   choices=("auto", "enumerate", "delegate"),
                   help="auto = 读 JSONL 的 `sampling_mode`。enumerate:"
                        "log_softmax(score_subsets / T_json);delegate:selector 自带"
                        "分布(用它自己的 self.temperature)")

    # ---- 模型与集成缝 ----
    g = p.add_argument_group("模型")
    g.add_argument("--selector-ckpt", default="",
                   help="初始/继续训练的 selector checkpoint(--dry-run 时可省)")
    g.add_argument("--selector-module", default=DEFAULT_SELECTOR_MODULE,
                   help="提供 SelectorStateBuilder / selector 类的模块名"
                        "(**必须与 rollout_selector.py 的同名参数一致**)")
    g.add_argument("--selector-factory", default="",
                   help="MODULE:ATTR,显式指定 selector 工厂(自动发现失败时用)")
    g.add_argument("--state-builder", default="",
                   help="MODULE:ATTR,显式指定 state builder;不给则用 "
                        "<--selector-module>.SelectorStateBuilder,再退回 "
                        "policy_io.SelectorState.from_bank")
    g.add_argument("--features-module", default=DEFAULT_FEATURES_MODULE,
                   help="提供 FeatureCache / build_cache 的模块名")
    g.add_argument("--model-dir", default="",
                   help="冻结 GUI-Owl 目录:FrozenTextEmbedder 与冻结视觉特征都用它")
    g.add_argument("--snapshot-manifest", default="",
                   help="透传给 features.build_extractor 的权重快照清单")
    g.add_argument("--out", default="",
                   help="输出 checkpoint 路径(--dry-run 时可省)")
    g.add_argument("--ref-ckpt", default="",
                   help="π_ref 的 checkpoint;留空且 --kl-coef>0 时用初始 checkpoint")

    # ---- 优化 ----
    g = p.add_argument_group("优化")
    g.add_argument("--epochs", type=int, default=1)
    g.add_argument("--lr", type=float, default=1e-5)
    g.add_argument("--clip-eps", type=float, default=0.2,
                   help="PPO 比值裁剪半径 ε")
    g.add_argument("--ent-coef", type=float, default=0.01,
                   help="β_ent:子集分布熵奖励(防塌缩,路线 §4 指定手段之一)")
    g.add_argument("--kl-coef", type=float, default=0.0,
                   help="β_kl:KL(π_θ‖π_ref),默认关")
    g.add_argument("--adv-eps", type=float, default=1e-6,
                   help="优势归一分母 (std + eps);调大 = 抑制小方差组的放大")
    g.add_argument("--grad-accum", type=int, default=8,
                   help="累积多少个**有效组**再 optimizer.step()")
    g.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="梯度裁剪阈值(≤0 关闭,但仍报告 grad norm)")
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=20260813)
    g.add_argument("--no-shuffle", action="store_true",
                   help="按文件顺序训练(默认每 epoch 按 seed 洗组)")
    g.add_argument("--train-dropout", action="store_true",
                   help="前向时用 nn.Module 的 train 模式(打开 backbone dropout)。"
                        "**默认关**:rollout 是 eval 模式采的样,logπ_old 里没有"
                        "dropout 噪声;这边开着会让 ratio=π_θ/π_old 带上一层与"
                        "策略无关的随机偏移,裁剪与更新方向一起失真")

    # ---- 特征 ----
    g = p.add_argument_group("特征")
    g.add_argument("--features", choices=("frozen", "dummy"), default="frozen",
                   help="frozen=features.build_cache 的冻结视觉特征;"
                        "dummy=features.DummyFeatureExtractor 的确定性假特征"
                        "(CPU 冒烟;与 rollout 侧 dry-run 同一实现)")
    g.add_argument("--feature-cache-dir", default="",
                   help="特征磁盘缓存目录(透传 features.build_cache)")
    g.add_argument("--feature-cache-items", type=int, default=0,
                   help="特征 LRU 上限(0 = 用 features 的默认值)")
    g.add_argument("--feature-dim", type=int, default=0,
                   help="dummy 特征维度(0 = 用 features 的默认值)。"
                        "**只有 rollout 侧也改过时才改**:假特征由 "
                        "(dim, tokens, seed, dtype) 唯一决定,任一项不同 log π 就对不上")
    g.add_argument("--feature-tokens", type=int, default=0,
                   help="dummy 特征 token 数(0 = 用 features 的默认值)")
    g.add_argument("--dummy-seed", type=int, default=-1,
                   help="dummy 特征的 seed(<0 = 用 features 的默认值 0,"
                        "与 rollout `--feature-source module` 一致)")
    g.add_argument("--visual-tokens", type=int, default=2560,
                   help="冻结视觉塔的 token 预算(透传)")
    g.add_argument("--feature-dtype", default="")

    # ---- 运行 ----
    g = p.add_argument_group("运行")
    g.add_argument("--device", default="cpu")
    g.add_argument("--rehydrate-frames", default="auto",
                   choices=("auto", "never"),
                   help="auto=截图缺失时按 task_source + 逐步动作确定性重放渲染"
                        "(rollout 开 --cleanup-shots 后必需);never=缺图直接报错")
    g.add_argument("--frame-dir", default="",
                   help="重放渲染的落图目录(默认 <--out 同级>/rehydrated_frames)")
    g.add_argument("--dry-run", action="store_true",
                   help="只读 JSONL 并打印分组/奖励/有效组诊断,不加载模型、不训练")
    g.add_argument("--log-every", type=int, default=10,
                   help="每 N 个**有效组**打印一次训练诊断")
    g.add_argument("--log-jsonl", default="",
                   help="把每次诊断同时以 JSONL 追加到该文件")
    g.add_argument("--logprob-check", type=int, default=64,
                   help="训练前重算前 N 个步的 logπ 与 JSONL 的 logπ_old 比对"
                        "(0=关)。同一 checkpoint 下必须一致,否则 state 重建错了")
    g.add_argument("--logprob-tol", type=float, default=1e-4,
                   help="logπ 一致性容差")
    g.add_argument("--strict-logprob", action="store_true",
                   help="一致性超差直接退出(继续训练一个已更新过的 ckpt 时应关闭)")
    return p.parse_args(argv)


# =============================================================== 数据

@dataclass
class StepSample:
    """一条 rollout 里**可产生梯度**的一步。

    # note (luojiaxuan): **编号口径(与 rollout_selector 写 steps 那段逐字一致)**:
    # ``candidates`` / ``chosen`` 里的整数是 **env 的 0-based 决策步号**(== policy_io
    # 的官方事件号,两者同值、不做偏移),**不是**候选列表里的下标。所以
    # ``enumerate_subsets(candidates, b)`` 直接在事件号上做组合、``subsets.index(chosen)``
    # 直接按事件号元组查找,全程不需要任何"下标↔事件号"换算 —— 一旦哪一侧改成
    # 下标口径,重算的 log π 就对应到另一批历史帧,而两边的 JSON 都看不出异样
    # (``--logprob-check`` 是唯一能抓到它的哨兵)。
    """

    step: int                       # env 0-based 决策序号
    candidates: tuple[int, ...]     # 该步的合法事件号(升序,已含 --max-candidates 截断)
    chosen: tuple[int, ...]         # 采样得到的 subset(升序,同为事件号)
    logp_old: float                 # rollout 时刻 selector 给的 log π_old(S)
    budget: int                     # 该步实际基数 |S|


@dataclass
class RolloutRecord:
    """JSONL 的一行。除 `success` 外不携带任何奖励信号。"""

    task_id: str
    group_id: str
    rollout_id: str
    arm: str
    reward: float                              # = float(success),terminal-only
    instruction: str
    frames: dict[int, str]                     # env step → 截图路径
    history_actions: list[dict[str, Any]]      # env 0-based 步 k 的 computer_use 参数
    history_lines: list[str]                   # 同步号的官方历史文本(Action: ...)
    steps: list[StepSample]
    n_steps: int
    temperature: float = 1.0                   # 采样时刻的温度(重算必须同温)
    sampling_mode: str = "enumerate"
    state_builder_origin: str = ""             # rollout 侧记的 state 构造路径
    task_source: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    _bank: HistoryFrameBank | None = field(default=None, repr=False)
    _rehydrated: bool = field(default=False, repr=False)

    def bank(self, feature_provider: Callable[[int, str], Any] | None
             ) -> HistoryFrameBank:
        """按需构造(并缓存)帧仓库 —— 每条 rollout 一份,跨 epoch 复用特征缓存。"""
        if self._bank is None:
            self._bank = HistoryFrameBank(
                frames=dict(self.frames), feature_provider=feature_provider)
        return self._bank

    def missing_frames(self) -> list[int]:
        """需要但磁盘上不存在的帧步号(rollout 开 `--cleanup-shots` 后是常态)。"""
        need = {s.step for s in self.steps}
        need.update(j for s in self.steps for j in s.candidates)
        return sorted(k for k in need
                      if k not in self.frames or not Path(self.frames[k]).exists())


@dataclass
class Group:
    """同一 (task_id, group_id) 的一组 rollout —— GRPO 的优势基准单位。"""

    task_id: str
    group_id: str
    rollouts: list[RolloutRecord]

    @property
    def rewards(self) -> list[float]:
        return [r.reward for r in self.rollouts]


def _has_magic(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


def _expand_paths(patterns: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        hit = [Path(x) for x in sorted(glob.glob(pat))] if _has_magic(pat) else []
        if not hit:
            cand = Path(pat)
            if not cand.exists():
                raise FileNotFoundError(f"rollout 文件不存在:{pat}")
            hit = [cand]
        out.extend(hit)
    if not out:
        raise FileNotFoundError(f"--rollouts 没有匹配到任何文件:{list(patterns)}")
    return out


def _as_int_tuple(raw: Any, *, what: str) -> tuple[int, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{what} 必须是整数列表,收到 {raw!r}")
    vals = [int(v) for v in raw]
    if any(b <= a for a, b in zip(vals, vals[1:])):
        vals = sorted(set(vals))
    return tuple(vals)


def _pick(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _frames_from_record(rec: dict[str, Any], steps_raw: list[dict[str, Any]]
                        ) -> dict[int, str]:
    """帧仓库:优先顶层 `frames`(dict/list),否则从每步的 `screenshot` 收集。"""
    frames: dict[int, str] = {}
    raw = _pick(rec, "frames", "screenshots")
    if isinstance(raw, dict):
        for k, v in raw.items():
            frames[int(k)] = str(v)
    elif isinstance(raw, (list, tuple)):
        for i, v in enumerate(raw):
            frames[int(i)] = str(v)
    for s in steps_raw:
        shot = _pick(s, "screenshot", "frame", "image")
        if shot:
            frames.setdefault(int(_pick(s, "step", default=0)), str(shot))
    return frames


def load_rollouts(paths: Sequence[Path], *, budget: int, arm_filter: str = "",
                  ) -> tuple[list[RolloutRecord], dict[str, int]]:
    """读 rollout JSONL。**只从顶层 `success` 取奖励**,不看任何 per-step 奖励字段。"""
    records: list[RolloutRecord] = []
    counts: dict[str, int] = defaultdict(int)
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                counts["lines"] += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno} 不是合法 JSON:{exc}") from exc
                arm = str(_pick(rec, "arm", default=""))
                if arm_filter and arm != arm_filter:
                    counts["skipped_arm"] += 1
                    continue
                sel_meta = rec.get("selector") or {}
                trainable = sel_meta.get("trainable")
                if arm in NON_TRAINABLE_ARMS or trainable is False:
                    # note (luojiaxuan): recent/random 臂是评测对照,其 log π 是点
                    # 质量或均匀常数;混进 GRPO 会用一个根本不是 π_θ 的分布算 ratio。
                    counts["skipped_nontrainable_arm"] += 1
                    continue
                if bool(_pick(rec, "greedy", default=False)):
                    # note (luojiaxuan): --selector-greedy 的 rollout 里 log π ≡ 0
                    # (rollout_selector.choose_subset 的约定),不是真实采样概率,
                    # 拿它当 log π_old 会得到荒谬的 ratio。整条丢弃并计数。
                    counts["skipped_greedy"] += 1
                    continue
                if "success" not in rec:
                    raise ValueError(
                        f"{path}:{lineno} 缺少顶层 `success` —— 本训练器只接受 "
                        "terminal-only reward,不会从 step 字段合成奖励")
                reward = 1.0 if bool(rec["success"]) else 0.0
                steps_raw = list(rec.get("steps") or [])
                frames = _frames_from_record(rec, steps_raw)
                history_actions: list[dict[str, Any]] = []
                history_lines: list[str] = []
                samples: list[StepSample] = []
                for s in steps_raw:
                    step = int(_pick(s, "step", default=len(history_actions)))
                    act = _pick(s, "action", "arguments", default=None)
                    while len(history_actions) <= step:
                        history_actions.append({})
                        history_lines.append("")
                    if isinstance(act, dict):
                        history_actions[step] = dict(act)
                    # note (luojiaxuan): 官方历史文本直接取 rollout 记的 `action_line`,
                    # 不在训练端重渲染 —— 重渲染多一次口径漂移的机会,而 state builder
                    # 的文本嵌入是 log π 的输入之一,差一个字就对不上。
                    history_lines[step] = str(_pick(s, "action_line", default=""))
                    cands = _as_int_tuple(
                        _pick(s, "candidates", "candidate_events"),
                        what="candidates")
                    chosen = _as_int_tuple(
                        _pick(s, "chosen_subset", "subset", "shown_subset"),
                        what="chosen_subset")
                    logp = _pick(s, "selector_logprob", "logprob", "logp")
                    if not chosen:
                        counts["steps_no_choice"] += 1
                        continue
                    if len(cands) < 2:
                        # note (luojiaxuan): 候选 ≤1 时 C(n,b) 只有一个子集,
                        # log π 恒为 0、熵恒为 0 —— 结构上不可能有梯度,计数后跳过。
                        counts["steps_trivial"] += 1
                        continue
                    b = len(chosen)
                    want = min(budget, len(cands))
                    if b != want:
                        raise ValueError(
                            f"{path}:{lineno} step={step}:|chosen_subset|={b} 与 "
                            f"min(B={budget}, |candidates|={len(cands)})={want} 不符 —— "
                            "预算口径对不上,拒绝静默训练")
                    if b == len(cands):
                        # note (luojiaxuan): C(n, b) == 1 ⟺ b == n(b≥1 时)。
                        # 唯一合法动作 ⇒ log π ≡ 0、熵 ≡ 0,同样无梯度。
                        counts["steps_trivial"] += 1
                        continue
                    if logp is None:
                        counts["steps_missing_logprob"] += 1
                        continue
                    if not set(chosen) <= set(cands):
                        raise ValueError(
                            f"{path}:{lineno} step={step}:chosen_subset {chosen} "
                            f"不在 candidates {cands} 内")
                    samples.append(StepSample(step=step, candidates=cands,
                                              chosen=chosen, logp_old=float(logp),
                                              budget=b))
                    counts["steps_used"] += 1
                task_source = dict(rec.get("task_source") or {})
                instruction = str(_pick(rec, "instruction", "task_instruction",
                                        "goal", default=""))
                if not instruction and task_source:
                    # note (luojiaxuan): rollout 行里没有 instruction 字段,只有
                    # task_source(template_id/seed/regime)。instruction 是 state
                    # builder 的文本输入之一 —— 拿空串顶替会让重算的 log π 悄悄偏
                    # 掉几个 1e-4,看起来"差不多"实则不是同一个分布。这里按
                    # tasks.generate_task 确定性重建,与 rollout 侧逐字同源。
                    instruction = _instruction_from_source(task_source)
                    counts["instruction_rebuilt"] += 1
                records.append(RolloutRecord(
                    task_id=str(_pick(rec, "task_id", default="?")),
                    group_id=str(_pick(rec, "group_id", default="?")),
                    rollout_id=str(_pick(rec, "rollout_id", default=str(lineno))),
                    arm=arm,
                    reward=reward,
                    instruction=instruction,
                    frames=frames,
                    history_actions=history_actions,
                    history_lines=history_lines,
                    steps=samples,
                    n_steps=int(_pick(rec, "n_steps", default=len(steps_raw))),
                    temperature=float(_pick(rec, "temperature", default=1.0)),
                    sampling_mode=str(_pick(rec, "sampling_mode",
                                            default="enumerate")),
                    state_builder_origin=str(sel_meta.get("state_builder", "")),
                    task_source=task_source,
                    source_file=str(path),
                ))
                counts["rollouts"] += 1
    return records, dict(counts)


_TASK_CACHE: dict[tuple[str, int, str], Any] = {}


def task_from_source(task_source: Mapping[str, Any]) -> Any:
    """`task_source` → `TaskSpec`,按 (template_id, seed, regime) 记忆化。

    `tasks.generate_task` 是确定性的(同 seed 同实例),这正是 rollout 侧不落
    instruction / 允许 `--cleanup-shots` 删图的前提。
    """
    key = (str(task_source.get("template_id", "")),
           int(task_source.get("seed", 0)),
           str(task_source.get("regime", "")))
    if key not in _TASK_CACHE:
        from causalcache_agentic import tasks as tasks_module

        _TASK_CACHE[key] = tasks_module.generate_task(
            template_id=key[0], seed=key[1], regime=key[2])
    return _TASK_CACHE[key]


def _instruction_from_source(task_source: Mapping[str, Any]) -> str:
    try:
        return str(task_from_source(task_source).instruction)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"rollout 行既没有 `instruction` 也无法从 task_source={dict(task_source)!r} "
            f"重建({type(exc).__name__}: {exc})。没有 instruction 就重建不出同一个 "
            "state,log π 必然对不上 —— 拒绝用空串顶替。") from exc


def rehydrate_frames(rec: RolloutRecord, frame_root: Path) -> int:
    """按 task_source + 逐步动作**确定性重放**,把缺失的 PNG 重新渲染出来。

    # note (luojiaxuan): rollout 的 `--cleanup-shots`(其帮助文本写着"默认建议开")
    # 会把 PNG 全删掉,而冻结视觉特征必须读图。环境是纯函数式的:同 TaskSpec 同
    # 动作序列 ⇒ 同状态 ⇒ 同渲染(contract.py 的 E1),所以重放能逐位还原当时的帧。
    # 这不是"近似重建",是同一函数的再求值。
    """
    from causalcache_agentic.env import GUIEnv

    if rec._rehydrated:
        return 0
    rec._rehydrated = True
    need = set(rec.missing_frames())
    if not need:
        return 0
    task = task_from_source(rec.task_source)
    out_dir = frame_root / _safe_name(rec.task_id) / _safe_name(str(rec.rollout_id))
    env = GUIEnv(auto_stop_on_success=False)
    env.reset(task)
    made = 0
    for k in range(rec.n_steps):
        if k in need:
            # note (luojiaxuan): **优先写回 JSONL 里记的原路径**。不只是整洁问题:
            # `--features dummy` 的假特征由路径字符串决定,换个目录就换了一组特征,
            # log π 一致性检查会红得莫名其妙。原路径不可写时才落到 --frame-dir。
            path = rec.frames.get(k)
            try:
                if path is None:
                    raise OSError("JSONL 未记该帧路径")
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                env.render(path)
            except OSError:
                out_dir.mkdir(parents=True, exist_ok=True)
                path = str(out_dir / f"step{k:02d}.png")
                env.render(path)
            rec.frames[k] = path
            made += 1
        if k < len(rec.history_actions):
            env.step(rec.history_actions[k])
    rec._bank = None          # 帧路径可能变了,已建的 bank(含特征缓存)必须作废
    return made


def _safe_name(name: str) -> str:
    return name.replace("::", "__").replace("/", "_")


def group_rollouts(records: Sequence[RolloutRecord]) -> list[Group]:
    """按 (task_id, group_id) 聚组,保持首次出现顺序。"""
    buckets: dict[tuple[str, str], Group] = {}
    order: list[tuple[str, str]] = []
    for r in records:
        key = (r.task_id, r.group_id)
        if key not in buckets:
            buckets[key] = Group(task_id=r.task_id, group_id=r.group_id, rollouts=[])
            order.append(key)
        buckets[key].rollouts.append(r)
    return [buckets[k] for k in order]


def advantages(group: Group, *, adv_eps: float, min_group_size: int
               ) -> tuple[list[float] | None, str]:
    """组内相对优势。返回 (A_i 列表 或 None, 理由标签)。

    **奖励全同(std=0)⇒ 返回 None**:这类组数学上不产生任何梯度,必须显式
    跳过并计数,而不是让 0 优势混进平均值里把有效样本冲淡。
    """
    rewards = group.rewards
    if len(rewards) < min_group_size:
        return None, "too_small"
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    if std <= 0.0:
        return None, "all1" if mean >= 1.0 - 1e-9 else "all0"
    denom = std + adv_eps
    return [(r - mean) / denom for r in rewards], "effective"


def group_report(groups: Sequence[Group], *, adv_eps: float, min_group_size: int
                 ) -> dict[str, Any]:
    """路线 §4 点名要监控的组级诊断(--dry-run 也打印这张表)。"""
    tally: dict[str, int] = defaultdict(int)
    sizes: list[int] = []
    rewards: list[float] = []
    steps_eff = 0
    for g in groups:
        adv, why = advantages(g, adv_eps=adv_eps, min_group_size=min_group_size)
        tally[why] += 1
        sizes.append(len(g.rollouts))
        rewards.extend(g.rewards)
        if adv is not None:
            steps_eff += sum(len(r.steps) for r in g.rollouts)
    n = max(len(groups), 1)
    return {
        "groups_total": len(groups),
        "groups_effective": tally["effective"],
        "groups_degenerate_all0": tally["all0"],
        "groups_degenerate_all1": tally["all1"],
        "groups_too_small": tally["too_small"],
        "frac_effective_groups": tally["effective"] / n,
        "group_size_mean": statistics.fmean(sizes) if sizes else 0.0,
        "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
        "n_rollouts": len(rewards),
        "trainable_steps_in_effective_groups": steps_eff,
    }


# =============================================================== 动作空间

def enumerate_subsets(candidates: Sequence[int], b: int) -> list[tuple[int, ...]]:
    """动作空间 = C(n, b) 全枚举(recent-B 只是其中一个普通合法动作)。"""
    # note (luojiaxuan): 先用 comb 算基数再枚举 —— 否则病态的 candidates 会在
    # 触发上限检查之前就把内存吃掉。
    total = math.comb(len(candidates), b)
    if total > MAX_SUBSETS_HARD:
        raise ValueError(
            f"候选 {len(candidates)} 取 {b} 得到 {total} 个子集,超过硬上限 "
            f"{MAX_SUBSETS_HARD};检查 rollout 的 candidates 记账")
    return list(combinations(tuple(candidates), b))


def recent_subset(candidates: Sequence[int], b: int) -> tuple[int, ...]:
    k = min(int(b), len(candidates))
    return tuple(candidates[-k:]) if k else ()


# =============================================================== 集成缝

def _call_filtered(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """按被调方签名过滤 kwargs 再调用(并行开发期的命名容差)。

    # note (luojiaxuan): selector_model.py / features.py 由并行 agent 实现,
    # 参数名可能与本文件的猜测不完全一致。这里不硬编码调用形态:能接受
    # **kwargs 的直接全给,否则只递交签名里出现过的键。多余的猜测键被丢掉,
    # 缺失的必填键仍会由被调方自己报错 —— 错在哪一侧一目了然。
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    allowed = {name for name, p in sig.parameters.items()
               if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                             inspect.Parameter.KEYWORD_ONLY)}
    return fn(**{k: v for k, v in kwargs.items() if k in allowed})


def _import_optional(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


class DummyFeatureCache:
    """`--features dummy`:路径哈希 → 确定性假特征向量(CPU 冒烟专用)。

    数值毫无语义,只用来验证 state 重建 / 打分 / 反传这条链路在没有 GPU、
    没有真实截图的机器上也能跑通。**禁止用它产出任何实验结论**。
    """

    def __init__(self, dim: int = 512, dtype: str = "float32") -> None:
        self.dim = int(dim)
        self.dtype = dtype
        self._cache: dict[str, Any] = {}

    def __call__(self, step: int, path: str) -> Any:
        import numpy as np
        import torch

        key = f"{step}:{path}"
        if key not in self._cache:
            import hashlib

            seed = int.from_bytes(
                hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")
            rng = np.random.default_rng(seed % (2 ** 63))
            vec = rng.standard_normal(self.dim).astype("float32")
            vec /= (float(np.linalg.norm(vec)) + 1e-8)
            self._cache[key] = torch.from_numpy(vec).to(getattr(torch, self.dtype))
        return self._cache[key]

    # note (luojiaxuan): 兼容以方法名调用的 state builder。
    def get(self, step: int, path: str) -> Any:
        return self(step, path)

    def feature(self, step: int, path: str) -> Any:
        return self(step, path)


def feature_spec(args: argparse.Namespace) -> dict[str, Any]:
    """`features.build_cache` 的入参(它用 `_pick` 从 Mapping / Namespace 取字段)。

    # note (luojiaxuan): 这里**不能**直接把本文件的 argparse.Namespace 递过去 ——
    # 我们的 `--dry-run` 意思是"不训练",而 features 侧的 `dry_run` 意思是"用假
    # extractor";同名不同义,必须显式翻译。`--features dummy` 才映射成 dry_run。
    """
    # note (luojiaxuan): dummy_* 一律**只在显式给了才传** —— features 侧的默认值
    # (dim 4096 / tokens 64 / seed 0)就是 rollout `--feature-source module` 拿到的
    # 那一组;这里若擅自填自己的默认值,假特征立刻分叉,log π 一致性检查会红。
    spec: dict[str, Any] = {
        "dry_run": args.features == "dummy",
        "device": args.device,
        "visual_tokens": args.visual_tokens,
    }
    if args.feature_dim:
        spec["dummy_dim"] = args.feature_dim
    if args.dummy_seed >= 0:
        spec["dummy_seed"] = args.dummy_seed
    if args.model_dir:
        spec["model_dir"] = args.model_dir
    if args.snapshot_manifest:
        spec["snapshot_manifest"] = args.snapshot_manifest
    if args.feature_dtype:
        spec["feature_dtype"] = args.feature_dtype
    if args.feature_tokens:
        spec["dummy_tokens"] = args.feature_tokens
    if args.feature_cache_dir:
        spec["feature_cache_dir"] = args.feature_cache_dir
    if args.feature_cache_items:
        spec["feature_cache_items"] = args.feature_cache_items
    return spec


def _resolve_feature_cache(args: argparse.Namespace) -> Callable[[int, str], Any]:
    """帧特征提供者。首选 `features.build_cache(spec)` —— **与 rollout 同一个工厂**。

    退路顺序:`build_cache` → `FeatureCache(extractor)` → 本文件的
    `DummyFeatureCache`(仅当 features 模块不可用且 `--features dummy`)。
    返回值必须能 `provider(step, path) -> 特征`;`features.FeatureCache.__call__`
    正是这个签名。
    """
    mod = _import_optional(args.features_module)
    if mod is None:
        if args.features == "dummy":
            print(f"[grpo] warn: 无 {args.features_module},退回本文件的假特征",
                  flush=True)
            return DummyFeatureCache(dim=args.feature_dim or 512,
                                     dtype=args.feature_dtype or "float32")
        raise ImportError(
            f"--features frozen 需要模块 {args.features_module};"
            "在它就位之前可用 --features dummy 跑结构冒烟")
    for name in ("build_cache", "build_feature_cache"):
        factory = getattr(mod, name, None)
        if callable(factory):
            return _as_provider(factory(feature_spec(args)))
    cls = getattr(mod, "FeatureCache", None)
    if cls is None:
        raise ImportError(
            f"{args.features_module} 里找不到 build_cache / FeatureCache")
    maker = getattr(mod, "build_extractor", None)
    extractor = maker(feature_spec(args)) if callable(maker) else None
    return _as_provider(_call_filtered(
        cls, extractor=extractor, cache_dir=args.feature_cache_dir or None,
        device=args.device, dim=args.feature_dim or None,
        dtype=args.feature_dtype or None))


def _as_provider(obj: Any) -> Callable[[int, str], Any]:
    if callable(obj):
        return obj
    for name in ("get", "feature", "__getitem__"):
        if hasattr(obj, name):
            fn = getattr(obj, name)
            return lambda step, path, _fn=fn: _fn(step, path)
    raise TypeError(f"特征缓存对象 {type(obj).__name__} 既不可调用也没有 get/feature")


def _resolve_embedder(args: argparse.Namespace, selector: Any) -> Any:
    """state builder 要的冻结文本嵌入器。与 rollout 的 `_resolve_embedder` 同序:
    selector 自带 > `FrozenTextEmbedder.from_model_dir` > `DummyTextEmbedder`。"""
    for name in ("embedder", "text_embedder"):
        got = getattr(selector, name, None)
        if got is not None:
            return got
    maker = getattr(selector, "make_embedder", None)
    if callable(maker):
        return maker()
    mod = _import_optional(args.selector_module)
    if mod is None:
        return None
    frozen = getattr(mod, "FrozenTextEmbedder", None)
    if frozen is not None and args.model_dir:
        return _call_filtered(frozen.from_model_dir, model_dir=args.model_dir,
                              device=args.device)
    dummy = getattr(mod, "DummyTextEmbedder", None)
    if dummy is not None and args.features == "dummy":
        # note (luojiaxuan): 假文本嵌入只配假视觉特征 —— 真跑一定要 --model-dir,
        # 否则 state 的文本通道与 rollout 侧不是同一个东西。
        return dummy()
    return None


def _resolve_state_builder(args: argparse.Namespace, selector: Any,
                           provider: Callable[..., Any] | None
                           ) -> tuple[Callable[..., Any], str]:
    """state_repr 重建器,返回 (调用入口, origin 描述)。

    构造顺序与 `rollout_selector.resolve_state_factory` **完全对应**:
    `--state-builder MODULE:ATTR` > `<--selector-module>.SelectorStateBuilder` >
    `policy_io.SelectorState.from_bank`(Phase 0 冻结契约的最小载体)。
    origin 会与 JSONL 里 rollout 记的 `selector.state_builder` 对账。
    """
    cls: Any = None
    origin = ""
    if args.state_builder:
        mod_name, _, attr = args.state_builder.partition(":")
        if not attr:
            raise SystemExit(
                f"--state-builder 需要 MODULE:ATTR 形式,收到 {args.state_builder!r}")
        cls = getattr(importlib.import_module(mod_name), attr, None)
        if cls is None:
            raise SystemExit(f"{mod_name} 里没有 {attr!r}")
        origin = f"{mod_name}:{attr}"
    else:
        mod = _import_optional(args.selector_module)
        cls = getattr(mod, "SelectorStateBuilder", None) if mod else None
        origin = f"{args.selector_module}:SelectorStateBuilder"

    if cls is None:
        def _fallback(**kw: Any) -> SelectorState:
            return _call_filtered(SelectorState.from_bank, **kw)
        return _fallback, "policy_io.SelectorState.from_bank"

    builder = _call_filtered(
        cls, embedder=_resolve_embedder(args, selector),
        budget=args.budget, b=args.budget, device=args.device,
        feature_cache=provider, feature_provider=provider, features=provider)
    for name in ("build", "build_state", "__call__"):
        if hasattr(builder, name):
            return (lambda _fn=getattr(builder, name), **kw:
                    _call_filtered(_fn, **kw)), origin
    raise TypeError(f"{origin} 没有 build/build_state/__call__")


def _discover_loaders(args: argparse.Namespace) -> list[tuple[Any, str]]:
    """候选 selector 加载入口,顺序与 `rollout_selector._discover_factory` 一致。"""
    if args.selector_factory:
        mod_name, _, attr = args.selector_factory.partition(":")
        if not attr:
            raise SystemExit(
                f"--selector-factory 需要 MODULE:ATTR 形式,收到 {args.selector_factory!r}")
        obj = getattr(importlib.import_module(mod_name), attr, None)
        if obj is None:
            raise SystemExit(f"{mod_name} 里没有 {attr!r}")
        return [(obj, f"{mod_name}:{attr}")]
    ordered = [args.selector_module] + [m for m in _SELECTOR_MODULES
                                        if m != args.selector_module]
    out: list[tuple[Any, str]] = []
    for mod_name in ordered:
        mod = _import_optional(mod_name)
        if mod is None:
            continue
        for cls_name in _SELECTOR_CLASSES:
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            for name in ("load", "from_checkpoint"):
                fn = getattr(cls, name, None)
                if callable(fn):
                    out.append((fn, f"{mod_name}:{cls_name}.{name}"))
        for name in _SELECTOR_FACTORIES:
            fn = getattr(mod, name, None)
            if callable(fn):
                out.append((fn, f"{mod_name}:{name}"))
    return out


def _resolve_selector(args: argparse.Namespace, ckpt_path: str
                      ) -> tuple[Any, dict, str]:
    """加载 selector(nn.Module)。返回 (module, bundle 非权重字段, origin)。

    优先走 selector 模块自己的类方法 `load` / `from_checkpoint`(`SubsetSelectorPolicy`
    的 checkpoint 是 `{cfg, temperature, state_dict}`,只有它自己能复原结构),
    再试模块级工厂,最后才试 `torch.load` 出来直接就是 nn.Module 的形态。
    """
    import torch

    tried: list[str] = []
    errors: list[str] = []
    for fn, origin in _discover_loaders(args):
        tried.append(origin)
        try:
            obj = _call_filtered(
                fn, path=ckpt_path, ckpt=ckpt_path, checkpoint=ckpt_path,
                ckpt_path=ckpt_path, device=args.device, budget=args.budget,
                seed=args.seed, map_location=args.device)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{origin}: {type(exc).__name__}: {exc}")
            continue
        model, extra = _unwrap_selector(obj)
        if model is not None:
            return model.to(args.device), extra, origin
    tried.append("torch.load(ckpt) -> nn.Module")
    blob = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model, extra = _unwrap_selector(blob)
    if model is not None:
        return model.to(args.device), extra, "torch.load"
    raise TypeError(
        "无法从 checkpoint 得到一个 nn.Module selector。已尝试:\n  - "
        + "\n  - ".join(tried)
        + (("\n错误:\n  - " + "\n  - ".join(errors)) if errors else "")
        + f"\ncheckpoint 顶层类型 = {type(blob).__name__}"
        + (f",键 = {sorted(blob)}" if isinstance(blob, dict) else ""))


def _unwrap_selector(obj: Any) -> tuple[Any, dict]:
    """从各种返回形态里取出 nn.Module,并保留 bundle 的非权重字段(cfg 等)。"""
    import torch
    from torch import nn

    if isinstance(obj, nn.Module):
        return obj, {}
    if isinstance(obj, (tuple, list)) and obj:
        for item in obj:
            if isinstance(item, nn.Module):
                extra = next((x for x in obj if isinstance(x, dict)), {})
                return item, dict(extra)
    if isinstance(obj, dict):
        for key in ("model", "selector", "module", "policy"):
            if isinstance(obj.get(key), nn.Module):
                extra = {k: v for k, v in obj.items()
                         if k != key and not isinstance(v, (nn.Module, torch.Tensor))}
                return obj[key], extra
    for key in ("model", "selector", "module"):
        cand = getattr(obj, key, None)
        if isinstance(cand, nn.Module):
            return cand, {}
    return None, {}


def _score_fn(selector: Any) -> Callable[[Any, list[tuple[int, ...]]], Any]:
    """取 selector 的子集打分入口(policy_io.SelectorPolicy 的 `score_subsets`)。"""
    for name in ("score_subsets", "subset_logits", "forward"):
        if hasattr(selector, name):
            return getattr(selector, name)
    raise TypeError(
        f"selector {type(selector).__name__} 没有 score_subsets / subset_logits / forward")


# =============================================================== 单步前向

@dataclass
class StepForward:
    """一步重算的结果(全部是 torch 标量,除 index)。"""

    logp_new: Any
    entropy: Any
    logits: Any
    log_probs: Any
    index: int
    recent_index: int


def build_state(selector: Any, state_builder: Callable[..., Any],
                rec: RolloutRecord, sample: StepSample,
                provider: Callable[[int, str], Any] | None) -> Any:
    """重建某一步的 state_repr。

    # note (luojiaxuan): 下面这个 kwargs 池与 `rollout_selector.StateFactory.__call__`
    # 的池**逐字相同**(键名、取值口径、recent 的算法)。两边都靠 `_call_filtered`
    # 按签名挑自己要的键,所以只要池一致,不同 state builder 在两侧拿到的输入
    # 必然相同 —— 这就是 log π 能对上的机械保证。
    """
    bank = rec.bank(provider)
    now = sample.step
    return state_builder(
        policy=selector, selector=selector,
        task_instruction=rec.instruction, instruction=rec.instruction,
        goal=rec.instruction,
        history_actions=list(rec.history_actions[:now]),
        actions=list(rec.history_actions[:now]),
        history_action_lines=list(rec.history_lines[:now]),
        bank=bank, frame_bank=bank,
        frame_features=(lambda j: bank.feature(j)),
        current_feature=(bank.feature(now)
                         if bank.feature_provider is not None else None),
        current_step=now, step=now,
        candidates=list(sample.candidates),
        recent=list(recent_subset(sample.candidates, sample.budget)),
        budget=sample.budget, b=sample.budget,
    )


def effective_temperature(args: argparse.Namespace, selector: Any,
                          rec: RolloutRecord) -> float:
    """重算 log π 用的温度 —— 必须与采样时刻同温,否则整条 ratio 无意义。

    * `--temperature > 0`:强制覆盖(排障用);
    * `sampling_mode == "delegate"`:采样走的是 selector 自己的 `sample()`,
      用的是模型属性 `self.temperature`;
    * 否则(`enumerate`,rollout 默认):rollout 用的是它 CLI 上的温度,
      已写进 JSONL 每行的 `temperature`。
    """
    if args.temperature > 0:
        return float(args.temperature)
    mode = args.sampling_mode if args.sampling_mode != "auto" else rec.sampling_mode
    if mode == "delegate":
        return float(getattr(selector, "temperature", 1.0) or 1.0)
    return float(rec.temperature or 1.0)


def forward_step(selector: Any, state_builder: Callable[..., Any],
                 rec: RolloutRecord, sample: StepSample,
                 provider: Callable[[int, str], Any] | None,
                 temperature: float = 1.0) -> StepForward:
    """重建 state_repr → 全动作空间打分 → 温度归一 → 取被选 subset 的 log π_θ。

    动作空间与熵都在 C(n, b) 上精确计算;梯度只经过 selector 参数,
    **不穿过图片选择、不穿过环境、不涉及 GUI-Owl**。
    """
    import torch

    subsets = enumerate_subsets(sample.candidates, sample.budget)
    state = build_state(selector, state_builder, rec, sample, provider)
    raw = _score_fn(selector)(state, subsets)
    logits = raw if isinstance(raw, torch.Tensor) else torch.as_tensor(raw)
    logits = logits.reshape(-1).float()
    if logits.shape[0] != len(subsets):
        raise ValueError(
            f"selector 返回 {logits.shape[0]} 个分数,与子集数 {len(subsets)} 不符")
    if temperature <= 0:
        raise ValueError(f"温度必须 > 0,收到 {temperature!r}")
    logits = logits / temperature
    log_probs = torch.log_softmax(logits, dim=0)
    probs = log_probs.exp()
    # note (luojiaxuan): logits 里可能出现 -inf(点质量基线臂),0·(-inf)=nan,
    # 故用 where 把零概率项显式置 0 再求和。
    ent_terms = torch.where(probs > 0, probs * log_probs,
                            torch.zeros_like(probs))
    entropy = -ent_terms.sum()
    try:
        idx = subsets.index(sample.chosen)
    except ValueError:
        raise ValueError(
            f"chosen_subset {sample.chosen} 不在枚举出的动作空间里"
            f"(candidates={sample.candidates}, b={sample.budget})") from None
    rec_sub = recent_subset(sample.candidates, sample.budget)
    r_idx = subsets.index(rec_sub) if rec_sub in subsets else -1
    return StepForward(logp_new=log_probs[idx], entropy=entropy, logits=logits,
                       log_probs=log_probs, index=idx, recent_index=r_idx)


# =============================================================== 诊断累加器

class Meter:
    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, key: str, value: float, n: int = 1) -> None:
        self.sums[key] += float(value) * n
        self.counts[key] += n

    def mean(self, key: str) -> float:
        c = self.counts.get(key, 0)
        return self.sums[key] / c if c else float("nan")

    def snapshot(self, keys: Iterable[str]) -> dict[str, float]:
        # note (luojiaxuan): 没有样本的键直接不出现,而不是打一个 NaN —— 既避免
        # 把 `kl=nan`(β_kl=0 时的常态)当成数值异常,也让日志始终是合法 JSON。
        return {k: self.mean(k) for k in keys if self.counts.get(k, 0) > 0}

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()


LOG_KEYS = ("mean_reward", "frac_effective_groups", "mean_ratio", "clip_frac",
            "clip_active_frac", "entropy", "grad_norm", "pi_recent",
            "pi_chosen", "recent_is_argmax", "kl", "adv_abs", "loss")


# =============================================================== logπ 一致性

def check_state_builder_origin(groups: Sequence[Group], origin: str,
                               *, strict: bool) -> dict[str, Any]:
    """把 rollout 记的 `selector.state_builder` 与本次训练用的构造路径对账。

    # note (luojiaxuan): rollout_selector 专门把 origin 写进每行记录,就是为了
    # 让这条比对成为可能。两侧走不同 state builder 时 log π 一定不同,但那种
    # 错误在数值上很像"训练不收敛",不点名就查不出来。
    """
    seen = sorted({r.state_builder_origin for g in groups for r in g.rollouts
                   if r.state_builder_origin})
    ok = (not seen) or (origin in seen and len(seen) == 1)
    out = {"trainer_origin": origin, "rollout_origins": seen, "match": bool(ok)}
    print("[state-builder] " + json.dumps(out, ensure_ascii=False), flush=True)
    if not ok:
        msg = (f"state builder 口径不一致:训练端 {origin!r},rollout 端 {seen}。"
               "两侧 state 不同 ⇒ 重算的 log π 不是同一个分布。")
        if strict:
            raise SystemExit(msg)
        print("[state-builder] warn: " + msg, flush=True)
    return out


def check_logprob_consistency(selector: Any, state_builder: Callable[..., Any],
                              provider: Callable[[int, str], Any] | None,
                              groups: Sequence[Group], *, limit: int, tol: float,
                              strict: bool, args: argparse.Namespace
                              ) -> dict[str, Any]:
    """训练前把 logπ_θ 重算一遍,与 JSONL 的 logπ_old 比对。

    **这是 state 重建正确性的唯一硬证据**:同一个 checkpoint 下,若重建的
    state_repr 与 rollout 当时喂给 selector 的一致,两者必须逐位近似相等。
    不一致 = 特征、候选顺序、历史动作切片或步号口径至少错了一个。
    继续训练一个已更新过的 ckpt 时该检查天然不成立,故 `--strict-logprob` 默认关。
    """
    import torch

    if limit <= 0:
        return {"checked": 0, "skipped": True}
    diffs: list[float] = []
    worst: tuple[float, str] | None = None
    with torch.no_grad():
        for g in groups:
            for rec in g.rollouts:
                temp = effective_temperature(args, selector, rec)
                for s in rec.steps:
                    fw = forward_step(selector, state_builder, rec, s, provider,
                                      temperature=temp)
                    d = abs(float(fw.logp_new) - s.logp_old)
                    diffs.append(d)
                    if worst is None or d > worst[0]:
                        worst = (d, f"{rec.task_id}/{rec.rollout_id}@step{s.step}")
                    if len(diffs) >= limit:
                        break
                if len(diffs) >= limit:
                    break
            if len(diffs) >= limit:
                break
    if not diffs:
        return {"checked": 0, "skipped": True}
    out = {
        "checked": len(diffs),
        "max_abs_diff": max(diffs),
        "mean_abs_diff": statistics.fmean(diffs),
        "tol": tol,
        "worst_step": worst[1] if worst else "",
        "passed": max(diffs) <= tol,
        "skipped": False,
    }
    msg = ("[logprob-check] n=%d max|Δ|=%.3e mean|Δ|=%.3e tol=%.1e worst=%s -> %s"
           % (out["checked"], out["max_abs_diff"], out["mean_abs_diff"], tol,
              out["worst_step"], "PASS" if out["passed"] else "FAIL"))
    print(msg, flush=True)
    if not out["passed"] and strict:
        raise SystemExit(
            "logπ 重算与 JSONL 的 logπ_old 不一致 —— state 重建错了(或 ckpt 不是"
            "产出该 rollout 的那个)。去掉 --strict-logprob 可强行继续。")
    return out


# =============================================================== 训练

def ensure_frames(args: argparse.Namespace, groups: Sequence[Group],
                  stats_sink: dict[str, Any]) -> None:
    """训练前保证每条 rollout 需要的 PNG 都在盘上(缺就确定性重放渲染)。

    # note (luojiaxuan): `--features dummy` 的假特征只由**路径字符串**决定、
    # 根本不打开文件,所以缺图无害,且此时重放到别的目录反而会改掉特征。
    # 重放渲染是为冻结视觉塔服务的,只在 frozen 下才有意义。
    """
    if args.features == "dummy":
        stats_sink["frames_rehydrated"] = 0
        return
    missing = [r for g in groups for r in g.rollouts if r.missing_frames()]
    stats_sink["rollouts_missing_frames"] = len(missing)
    if not missing:
        return
    if args.rehydrate_frames == "never":
        sample = missing[0]
        raise SystemExit(
            f"{len(missing)} 条 rollout 的截图不存在(如 {sample.task_id}/"
            f"{sample.rollout_id} 缺步 {sample.missing_frames()[:5]});"
            "rollout 若开了 --cleanup-shots 请用 --rehydrate-frames auto")
    root = Path(args.frame_dir) if args.frame_dir else (
        Path(args.out).parent / "rehydrated_frames")
    made = sum(rehydrate_frames(r, root) for r in missing)
    stats_sink["frames_rehydrated"] = made
    print(f"[grpo] 重放渲染 {made} 帧({len(missing)} 条 rollout)-> {root}",
          flush=True)


def train(args: argparse.Namespace, groups: list[Group], report: dict[str, Any]
          ) -> dict[str, Any]:
    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ensure_frames(args, groups, stats_sink=report)
    provider = _resolve_feature_cache(args)
    selector, bundle_extra, sel_origin = _resolve_selector(args, args.selector_ckpt)
    state_builder, sb_origin = _resolve_state_builder(args, selector, provider)
    # note (luojiaxuan): **默认 eval() 而不是 train()**。梯度与 requires_grad 无关
    # 于 train/eval,eval 只是关掉 dropout/BN 的随机性 —— 而 rollout 侧正是在
    # eval 下采的样。两侧模式不一致时 ratio = exp(logπ_θ - logπ_old) 在第一次
    # 更新前就已经 ≠1(纯 dropout 噪声),clip 会大量触发,更新方向被污染。
    selector.train(bool(args.train_dropout))
    print(f"[grpo] selector <- {sel_origin} ({args.selector_ckpt}) "
          f"train_mode={bool(args.train_dropout)}", flush=True)
    origin_report = check_state_builder_origin(groups, sb_origin,
                                               strict=args.strict_logprob)

    params = [p for p in selector.parameters() if p.requires_grad]
    if not params:
        raise ValueError("selector 没有任何 requires_grad 的参数,无法训练")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    ref_selector = None
    if args.kl_coef > 0.0:
        ref_path = args.ref_ckpt or args.selector_ckpt
        ref_selector, _, _ = _resolve_selector(args, ref_path)
        ref_selector.eval()
        for p in ref_selector.parameters():
            p.requires_grad_(False)

    consistency = check_logprob_consistency(
        selector, state_builder, provider, groups, limit=args.logprob_check,
        tol=args.logprob_tol, strict=args.strict_logprob, args=args)

    log_fh = open(args.log_jsonl, "a", encoding="utf-8") if args.log_jsonl else None
    meter = Meter()
    running = Meter()
    stats: dict[str, Any] = {
        "groups_seen": 0, "groups_effective": 0,
        "groups_degenerate_all0": 0, "groups_degenerate_all1": 0,
        "groups_too_small": 0, "steps_used": 0, "optimizer_steps": 0,
        "epochs": args.epochs,
    }
    order = list(range(len(groups)))
    rng = random.Random(args.seed)
    pending = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        if not args.no_shuffle:
            rng.shuffle(order)
        for gi in order:
            group = groups[gi]
            stats["groups_seen"] += 1
            adv, why = advantages(group, adv_eps=args.adv_eps,
                                  min_group_size=args.min_group_size)
            for r in group.rewards:
                meter.add("mean_reward", r)
                running.add("mean_reward", r)
            if adv is None:
                key = {"all0": "groups_degenerate_all0",
                       "all1": "groups_degenerate_all1",
                       "too_small": "groups_too_small"}[why]
                stats[key] += 1
                meter.add("frac_effective_groups", 0.0)
                running.add("frac_effective_groups", 0.0)
                continue
            stats["groups_effective"] += 1
            meter.add("frac_effective_groups", 1.0)
            running.add("frac_effective_groups", 1.0)

            # note (luojiaxuan): **逐步 backward,不保留整组计算图**。
            # 原写法把一组(8 条 rollout × ~10 步)的 surrogate 全部 append 后
            # 才 backward —— 每步要在 C(n,2) 个子集上过 128-latent 帧,累起来
            # 实测吃满 140GB 显存(2026-08-13 两次 OOM)。GRPO 的目标
            # -mean_i mean_t[term] 对步与 rollout 都是**线性平均**,优势 a_i 是
            # 常数,所以把 term/(n_roll*n_step) 就地 backward 与整组一次
            # backward 的梯度**逐位等价**,只是峰值显存降到单步。
            live_rollouts = [(rec, a_i) for rec, a_i in zip(group.rollouts, adv)
                             if rec.steps]
            n_roll = len(live_rollouts)
            loss_acc = 0.0
            for rec, a_i in live_rollouts:
                surrogates: list[Any] = []
                temp = effective_temperature(args, selector, rec)
                for s in rec.steps:
                    fw = forward_step(selector, state_builder, rec, s, provider,
                                      temperature=temp)
                    ratio = torch.exp(fw.logp_new - s.logp_old)
                    unclipped = ratio * a_i
                    clipped = torch.clamp(ratio, 1.0 - args.clip_eps,
                                          1.0 + args.clip_eps) * a_i
                    surrogate = torch.min(unclipped, clipped)
                    term = surrogate + args.ent_coef * fw.entropy
                    if ref_selector is not None:
                        with torch.no_grad():
                            ref_fw = forward_step(ref_selector, state_builder,
                                                  rec, s, provider,
                                                  temperature=temp)
                        kl = torch.sum(fw.log_probs.exp()
                                       * (fw.log_probs - ref_fw.log_probs))
                        term = term - args.kl_coef * kl
                        for m in (meter, running):
                            m.add("kl", float(kl.detach()))
                    # 就地 backward(scale = 1/(组内 rollout 数 × 该 rollout 步数)
                    # × 1/grad_accum),再丢弃该步的图
                    n_step = max(len(rec.steps), 1)
                    scale = 1.0 / (n_roll * n_step * max(args.grad_accum, 1))
                    (-(term) * scale).backward()
                    loss_acc += float(term.detach()) / (n_roll * n_step)
                    surrogates.append(1)
                    # note (luojiaxuan): 诊断读数全部 detach —— 绝不让日志统计
                    # 意外把张量挂回计算图(也消掉 float(requires_grad) 的警告)。
                    with torch.no_grad():
                        r_val = float(ratio.detach())
                        probs = fw.log_probs.detach().exp()
                        is_rec = (1.0 if fw.recent_index >= 0
                                  and int(torch.argmax(fw.logits.detach()))
                                  == fw.recent_index else 0.0)
                        for m in (meter, running):
                            m.add("mean_ratio", r_val)
                            m.add("clip_frac",
                                  1.0 if abs(r_val - 1.0) > args.clip_eps else 0.0)
                            m.add("clip_active_frac",
                                  1.0 if float(clipped.detach())
                                  < float(unclipped.detach()) else 0.0)
                            m.add("entropy", float(fw.entropy.detach()))
                            m.add("pi_chosen", float(probs[fw.index]))
                            m.add("adv_abs", abs(a_i))
                            m.add("recent_is_argmax", is_rec)
                            if fw.recent_index >= 0:
                                m.add("pi_recent", float(probs[fw.recent_index]))
                    stats["steps_used"] += 1
                if not surrogates:
                    continue
            if loss_acc == 0.0 and n_roll == 0:
                continue
            loss_value = -loss_acc
            for m in (meter, running):
                m.add("loss", loss_value)
            pending += 1

            if pending >= args.grad_accum:
                gn = torch.nn.utils.clip_grad_norm_(
                    params, args.max_grad_norm if args.max_grad_norm > 0
                    else float("inf"))
                opt.step()
                opt.zero_grad(set_to_none=True)
                pending = 0
                stats["optimizer_steps"] += 1
                for m in (meter, running):
                    m.add("grad_norm", float(gn))
                if (args.log_every > 0
                        and stats["optimizer_steps"] % args.log_every == 0):
                    _emit(epoch, stats, running, t0, log_fh)
                    running.reset()

    if pending > 0:
        gn = torch.nn.utils.clip_grad_norm_(
            params, args.max_grad_norm if args.max_grad_norm > 0 else float("inf"))
        opt.step()
        opt.zero_grad(set_to_none=True)
        stats["optimizer_steps"] += 1
        for m in (meter, running):
            m.add("grad_norm", float(gn))
    _emit("final", stats, meter, t0, log_fh)
    if log_fh is not None:
        log_fh.close()

    stats["final_metrics"] = meter.snapshot(LOG_KEYS)
    stats["logprob_consistency"] = consistency
    stats["state_builder"] = origin_report
    stats["selector_origin"] = sel_origin
    stats["dataset_report"] = report
    stats["wall_seconds"] = time.time() - t0
    if args.out:
        save_checkpoint(args, selector, bundle_extra, stats)
    return stats


def _emit(epoch: Any, stats: dict[str, Any], meter: Meter, t0: float,
          log_fh: Any) -> None:
    row = {"epoch": epoch, "opt_steps": stats["optimizer_steps"],
           "groups_seen": stats["groups_seen"],
           "groups_effective": stats["groups_effective"],
           "groups_all0": stats["groups_degenerate_all0"],
           "groups_all1": stats["groups_degenerate_all1"],
           "steps_used": stats["steps_used"],
           "elapsed_s": round(time.time() - t0, 1)}
    row.update({k: round(v, 6) for k, v in meter.snapshot(LOG_KEYS).items()})
    print("[grpo] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    if log_fh is not None:
        log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log_fh.flush()


def save_checkpoint(args: argparse.Namespace, selector: Any,
                    bundle_extra: dict, stats: dict[str, Any]) -> None:
    """落盘。

    # note (luojiaxuan): **优先调 selector 自己的 `save()`**,再把训练元数据补写进
    # 同一个文件。理由是硬的:`SubsetSelectorPolicy.load` 要求 `{cfg, state_dict}`,
    # 而下一轮 rollout 正是用它来读这个 checkpoint。若这里自作主张写成
    # `{model_state: ...}`,产出的 ckpt 谁也加载不了,训练→采样的闭环当场断掉。
    """
    import torch

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    saver = getattr(selector, "save", None)
    native = False
    if callable(saver):
        try:
            saver(out)
            native = True
        except Exception as exc:  # noqa: BLE001
            print(f"[grpo] warn: selector.save 失败({type(exc).__name__}: {exc}),"
                  "退回通用 bundle 格式", flush=True)
    if native:
        blob = torch.load(out, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict):
            blob = {"model_state": selector.state_dict()}
    else:
        blob = dict(bundle_extra)
        blob["state_dict"] = selector.state_dict()
        blob["model_state"] = blob["state_dict"]
        blob.setdefault("cfg", bundle_extra.get("cfg")
                        or getattr(selector, "cfg", None))
    blob.update({
        "trainer_format": CKPT_FORMAT,
        "train_cfg": vars(args),
        "train_stats": stats,
        "selector_cfg": blob.get("cfg") or getattr(selector, "cfg", None),
        "reward": "terminal_success_only",
        "source_ckpt": args.selector_ckpt,
        "native_save": native,
    })
    torch.save(blob, out)
    print(f"[grpo] checkpoint -> {out} (native_save={native})", flush=True)


# =============================================================== main

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print("[grpo] cfg " + json.dumps(vars(args), ensure_ascii=False), flush=True)
    paths = _expand_paths(args.rollouts)
    records, counts = load_rollouts(paths, budget=args.budget,
                                    arm_filter=args.arm)
    groups = group_rollouts(records)
    if args.limit_groups > 0:
        groups = groups[:args.limit_groups]
    report = group_report(groups, adv_eps=args.adv_eps,
                          min_group_size=args.min_group_size)
    report["files"] = [str(p) for p in paths]
    report.update({f"load_{k}": v for k, v in counts.items()})
    print("[grpo] dataset " + json.dumps(report, ensure_ascii=False), flush=True)

    if args.dry_run:
        print("[grpo] --dry-run:只做数据与组诊断,不加载模型、不训练", flush=True)
        return 0
    if not args.selector_ckpt:
        raise SystemExit("--selector-ckpt 是必填(除非 --dry-run)")
    if not args.out:
        raise SystemExit("--out 是必填(除非 --dry-run)")
    if report["groups_effective"] == 0:
        raise SystemExit(
            "没有任何有效组(组内 reward 全同 ⇒ 优势恒 0 ⇒ 无梯度)。"
            "按路线 §4 应调难度 curriculum / group size / rollout 数量,"
            "**不得**引入 milestone reward 来制造信号。")
    stats = train(args, groups, report)
    print("[grpo] done " + json.dumps(
        {k: v for k, v in stats.items() if k != "dataset_report"},
        ensure_ascii=False, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
