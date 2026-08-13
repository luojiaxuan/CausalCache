#!/usr/bin/env python3
"""Phase 3 分组 rollout worker —— selector-only GRPO 的数据引擎。

# note (luojiaxuan): 路线见 rl/docs/agentic_memory_rl_roadmap_20260813.md §5(Phase 3)。
# 一句话:**固定一个 memory-aware 的冻结 executor,只训 selector**;selector 每步
# 从全部历史帧里采样 B=2 张喂给 executor;奖励只有任务最终成功 0/1;同一 task
# 用**同一初始状态**产 G 条 rollout,交给 GRPO 在组内做相对优势。
#
# 本文件只负责"产数据",不含任何参数更新。四条硬纪律:
#   1. **奖励只有终局 verifier 的 0/1**。不写 milestone、不写 step-level 正确性、
#      不读 gold 动作 —— 路线 §9 明令禁止(dry-run 的假 executor 是唯一例外,
#      它读 memory_probe 只为在无 GPU 时造出组内 reward 方差,见 DryRunPolicy);
#   2. **selector 看不到诊断信息**。selector 的输入恒为
#      ``SelectorState.from_bank(instruction, history_actions, bank, step)``,
#      结构上就装不下 memory_probe / required_steps;后者只出现在输出行的
#      ``diagnostics`` 字段里,供事后分层统计;
#   3. **同 group 同初始状态**。每条 rollout 用同一个 TaskSpec 重新 ``env.reset``,
#      并对 reset 后的 state 取摘要,组内不一致直接 fail-loud —— 差异只允许来自
#      selector 采样(executor 是 greedy 的);
#   4. **subset 打分是整体打分**。动作空间 = 全部 C(N,B) subset(recent-B 只是其中
#      一个普通合法动作),按 ``SelectorPolicy.score_subsets`` 批式打分后由本文件
#      统一做温度 softmax 与采样 —— 所以 ``selector_logprob`` 恒是**采样时刻**的
#      log π(S|state),而不是事后重算的近似。
#
# 与 GRPO trainer 的接口约定:输出 JSONL 每行一条 rollout,含 ``task_source``
# (template_id/seed/regime)与每步 ``action`` —— 环境是确定性的,故即使用了
# ``--cleanup-shots`` 删图,trainer 也能用 ``tasks.generate_task(**task_source)``
# + 重放动作序列**逐位重建**每一帧,再重算带梯度的 log π。

用法(下面命令都以 ``PYTHONPATH=rl/code:code python3 <本文件>`` 开头):

    # 无 GPU 的管线自检(macOS 开发机)
    --dry-run --n-tasks 3 --group-size 4 --out /tmp/rl_dry/rollouts.jsonl

    # GPU 真跑(容器内,8 shard 之一):learned selector 采样产 GRPO 数据
    --tasks-split train --n-tasks 512 --group-size 8 --budget 2 \\
    --model-dir /data/models/GUI-Owl-1.5-8B \\
    --snapshot-manifest /data/models/GUI-Owl-1.5-8B.manifest.json \\
    --adapter /data/mem_sft/adapter.pt --selector-ckpt /data/selector/step0.pt \\
    --out /data/g3/rollouts.jsonl --shot-dir /data/g3/shots --cleanup-shots \\
    --device cuda:0 --shard-index 0 --shard-count 8

    # 基线臂评测(确定性 ⇒ 组自动收敛到 1 条,不浪费显卡)
    --selector-arm recent --tasks-split syn_ood --n-tasks 200 ...
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]
for _extra in (_ROOT / "rl" / "code", _ROOT / "code", _HERE.parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from causalcache_agentic.contract import TaskSpec
from causalcache_agentic.env import GUIEnv, action_line, normalize_action
from causalcache_agentic.expert import subset_is_solvable
from causalcache_agentic.policy_io import (
    DEFAULT_BUDGET,
    HistoryFrameBank,
    PolicyInputBuilder,
    RandomBSelector,
    RecentBSelector,
    SelectorPolicy,
    SelectorState,
)

# note (luojiaxuan): 解码配置、冻结守卫、假策略、稳定 seed 全部**复用**
# agentic_memory_sensitivity —— 不抄一份第二实现。抄贴会随时间漂移,import
# 则保证 Phase 1 验收、Phase 2 闭环评测与 Phase 3 rollout 用的是逐字同一条
# 生成路径(do_sample=False / eos=tool_call_close / suppress=standard_eos)。
from agentic_memory_sensitivity import (  # noqa: E402
    FALLBACK_ACTION,
    DryRunPolicy,
    FrozenGUIOwlPolicy,
    SubsetChoice,
    read_probe,
    stable_seed,
)

ARMS = ("learned", "recent", "random")
# note (luojiaxuan): 自动发现 Phase 3 selector 实现的位置与工厂名。找不到就要求
# 显式 --selector-factory MODULE:ATTR —— 宁可报错,也不猜一个不对的类默默跑。
DEFAULT_SELECTOR_MODULE = "causalcache_agentic.selector_model"
_SELECTOR_MODULES = (DEFAULT_SELECTOR_MODULE, "causalcache_agentic.selector",
                     "causalcache_agentic.selector_policy",
                     "causalcache_agentic.subset_selector")
_SELECTOR_FACTORIES = ("load_selector", "load_selector_bundle", "build_selector",
                       "make_selector", "load_checkpoint", "load_from_checkpoint")
_SELECTOR_CLASSES = ("SubsetSelectorPolicy", "SubsetSelector", "SelectorPolicyNet",
                     "SelectorNet", "FrameSelector", "SubsetScorer")
_FACTORY_KWARGS = ("checkpoint", "ckpt", "ckpt_path", "path", "device", "budget",
                   "seed", "map_location")


# ---------------------------------------------------------------- 小工具

def state_digest(state: Any) -> str:
    """初始状态摘要:同 group 的 rollout 必须给出同一个值,否则记账已经串台。

    # note (luojiaxuan): 用 dataclass 递归 ``repr`` 而不是渲染 PNG —— 前者不花
    # 渲染时间,且能抓到"executor 原地改了入参 state"这类隐蔽污染(它会让第二条
    # rollout 从被上一条改脏的状态开局)。
    """
    import hashlib

    return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()[:16]


def _log_softmax(scores: np.ndarray) -> np.ndarray:
    """带 -inf 容忍的 log-softmax(基线臂用 -inf 表达"这个 subset 概率为 0")。"""
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise ValueError("selector 给出的 subset 分数全是 -inf/NaN,无法归一")
    shifted = scores - float(np.max(finite))
    total = float(np.exp(np.where(np.isfinite(shifted), shifted, -np.inf)).sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"softmax 归一因子非法({total!r})")
    return shifted - math.log(total)


def _entropy(logp: np.ndarray) -> float:
    p = np.exp(np.where(np.isfinite(logp), logp, -np.inf))
    keep = p > 0.0
    return float(-(p[keep] * logp[keep]).sum())


def _as_scores(raw: Any, n: int) -> np.ndarray:
    """selector 的打分 → float64 一维 numpy(torch.Tensor / numpy / list 都收)。"""
    if hasattr(raw, "detach"):
        raw = raw.detach().to("cpu").float().numpy()
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"score_subsets 返回 {arr.shape[0]} 个分数,期望 {n} 个")
    if np.isnan(arr).any():
        raise ValueError("score_subsets 返回了 NaN")
    return arr


def _safe_name(task_id: str) -> str:
    return task_id.replace("::", "__").replace("/", "_")


@contextlib.contextmanager
def inference_mode() -> Any:
    """selector 前向的统一上下文:没装 torch 就是空操作。

    # note (luojiaxuan): 两件事必须在这里做,少一件都会静默毁掉 GRPO:
    #   1. **no_grad** —— 参数化 selector 每步都过一遍 encode_state + subset
    #      scorer,不切图的话一条 rollout 会攒下几十步的 autograd 图(而这些图
    #      永远不会被 backward,纯浪费,长轨迹直接 OOM);
    #   2. **eval()**(在 resolve_selector 里一次性设好)—— backbone 带 dropout
    #      p=0.1,train 模式下同一 state 两次打分不同:记进 JSONL 的 log π_old
    #      不可复现,训练端重算的 ratio 会带上一层与策略无关的噪声。
    """
    torch = sys.modules.get("torch")
    if torch is None:
        yield
        return
    with torch.no_grad():
        yield


# ---------------------------------------------------------------- selector 侧

class StubHashSelector:
    """**dry-run 专用**的假 selector:哈希打分 + 温和 recent 先验 + 假特征读取。

    它不是模型,只是让 subset 枚举 / 温度采样 / log-prob 记账 / 特征 provider
    这几条管线在无 GPU 的机器上被真正走一遍。真跑一律走 ``--selector-ckpt``
    或 ``--selector-factory`` 指到的 Phase 3 实现。
    """

    def __init__(self, seed: int = 0, recent_bias: float = 0.5,
                 feature_weight: float = 0.25) -> None:
        self.seed = int(seed)
        self.recent_bias = float(recent_bias)
        self.feature_weight = float(feature_weight)
        self.metadata = {"selector_profile_id": "stub_hash_fake", "frozen": False,
                         "recent_bias": recent_bias}

    # -------------------------------------------------------- Protocol
    def score_subsets(self, state_repr: Any, subsets: list[tuple[int, ...]]
                      ) -> np.ndarray:
        instruction = getattr(state_repr, "task_instruction", "")
        now = int(getattr(state_repr, "current_step", 0))
        pool = list(getattr(state_repr, "candidates", ()) or ())
        recent = tuple(pool[-len(subsets[0]):]) if (subsets and pool and
                                                    subsets[0]) else ()
        out = np.zeros((len(subsets),), dtype=np.float64)
        for i, sub in enumerate(subsets):
            raw = stable_seed(self.seed, instruction, now, sub) % 1_000_003
            score = (raw / 1_000_003.0) * 2.0 - 1.0
            if sub and tuple(sub) == recent:
                score += self.recent_bias
            score += self.feature_weight * self._feature_term(state_repr, sub)
            out[i] = score
        return out

    def sample(self, state_repr: Any, candidates: Sequence[int], b: int
               ) -> tuple[tuple[int, ...], float]:
        subsets = enumerate_subsets(candidates, b)
        logp = _log_softmax(self.score_subsets(state_repr, subsets))
        idx = int(np.argmax(logp))
        return subsets[idx], float(logp[idx])

    def argmax(self, state_repr: Any, candidates: Sequence[int], b: int
               ) -> tuple[int, ...]:
        subsets = enumerate_subsets(candidates, b)
        return subsets[int(np.argmax(self.score_subsets(state_repr, subsets)))]

    # -------------------------------------------------------- 内部
    def _feature_term(self, state_repr: Any, subset: Sequence[int]) -> float:
        bank = getattr(state_repr, "bank", None)
        if bank is None or not subset:
            return 0.0
        try:
            vals = [float(np.asarray(bank.feature(j)).mean()) for j in subset]
        except NotImplementedError:
            return 0.0
        return sum(vals) / len(vals)


def enumerate_subsets(candidates: Sequence[int], b: int) -> list[tuple[int, ...]]:
    """动作空间:候选集上全部 C(n, b_eff) 个升序 subset(候选不足时自然退化)。

    n 为 0/1 或 b_eff == 0 时返回 ``[()]`` / 单元素列表 —— 早期步只有一个合法
    动作,不是错误。
    """
    pool = [int(j) for j in candidates]
    k = min(int(b), len(pool))
    return [tuple(c) for c in itertools.combinations(pool, k)]


def feature_spec(args: argparse.Namespace, *, dummy: bool) -> dict[str, Any]:
    """``causalcache_agentic.features.build_cache`` 的入参。

    # note (luojiaxuan): 这张键表与 ``train_selector_grpo.feature_spec`` **逐字同一
    # 套**,而且共享同一条纪律:**"没显式给就不传",让 features 模块自己的默认值
    # (dummy dim 4096 / tokens 64 / seed 0 / dtype float16)成为两侧唯一的真相**。
    # 曾经踩过的坑:本文件自带一份 ``make_fake_feature_provider``(路径哈希 → 一维
    # [64] 向量),trainer 侧却用 ``features.DummyFeatureExtractor``([64,4096]),
    # 于是 ①selector 直接拒收一维特征,②即便形状对了,两侧假特征也是两组不同的
    # 随机数 —— 重算的 log π 与记录的 log π_old 永远对不上,而症状看起来只是
    # "GRPO 不收敛"。假特征只允许有一个实现,就是 features 模块里的那个。
    """
    spec: dict[str, Any] = {"dry_run": bool(dummy), "device": args.device,
                            "visual_tokens": args.visual_tokens}
    if args.model_dir is not None:
        spec["model_dir"] = args.model_dir
    if args.snapshot_manifest is not None:
        spec["snapshot_manifest"] = args.snapshot_manifest
    if args.feature_dtype:
        spec["feature_dtype"] = args.feature_dtype
    if args.feature_dim:
        spec["dummy_dim"] = args.feature_dim
    if args.feature_tokens:
        spec["dummy_tokens"] = args.feature_tokens
    if args.dummy_seed >= 0:
        spec["dummy_seed"] = args.dummy_seed
    if args.feature_cache_dir is not None:
        spec["feature_cache_dir"] = args.feature_cache_dir
    if args.feature_cache_items is not None:
        spec["feature_cache_items"] = args.feature_cache_items
    return spec


def _call_filtered(fn: Any, /, **kwargs: Any) -> Any:
    """按 ``fn`` 的签名过滤 kwargs 后调用(别名超集进、只留它认识的)。

    # note (luojiaxuan): 与 train_selector_grpo.py 的同名工具**同一口径** ——
    # rollout 侧与训练侧必须用同一套 kwargs 别名去构造 state,否则训练端重算的
    # log π 与本文件记录的 log π 会对不上,GRPO 的重要性比整体失真。
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def _call_factory(factory: Any, args: argparse.Namespace) -> Any:
    """按签名投喂 checkpoint/device/budget/seed,避免对工厂签名做强假设。"""
    pool = {"checkpoint": args.selector_ckpt, "ckpt": args.selector_ckpt,
            "ckpt_path": args.selector_ckpt, "path": args.selector_ckpt,
            "map_location": args.device, "device": args.device,
            "budget": args.budget, "seed": args.rollout_seed}
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(args.selector_ckpt)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return factory(checkpoint=args.selector_ckpt, device=args.device,
                       budget=args.budget, seed=args.rollout_seed)
    kwargs = {k: v for k, v in pool.items()
              if k in sig.parameters and k in _FACTORY_KWARGS}
    return factory(**kwargs)


def _discover_factory(spec: str | None, module_hint: str = "") -> tuple[Any, str] | None:
    """(显式 MODULE:ATTR)或(在候选模块里找工厂/类)→ (callable, 来源描述)。

    ``module_hint`` 即 ``--selector-module``,**排在内置候选之前** —— 它同时决定
    state builder 的来源,两者必须落在同一个模块里,否则 state 与打分器不配套。
    """
    if spec:
        mod_name, _, attr = spec.partition(":")
        if not attr:
            raise SystemExit(f"--selector-factory 需要 MODULE:ATTR 形式,收到 {spec!r}")
        module = importlib.import_module(mod_name)
        obj = getattr(module, attr, None)
        if obj is None:
            raise SystemExit(f"{mod_name} 里没有 {attr!r}")
        return obj, f"{mod_name}:{attr}"
    ordered = ([module_hint] if module_hint else []) + [
        m for m in _SELECTOR_MODULES if m != module_hint]
    for mod_name in ordered:
        try:
            module = importlib.import_module(mod_name)
        except ImportError:
            continue
        for name in _SELECTOR_FACTORIES:
            obj = getattr(module, name, None)
            if callable(obj):
                return obj, f"{mod_name}:{name}"
        for cls_name in _SELECTOR_CLASSES:
            cls = getattr(module, cls_name, None)
            if cls is None:
                continue
            for name in ("from_checkpoint", "load"):
                obj = getattr(cls, name, None)
                if callable(obj):
                    return obj, f"{mod_name}:{cls_name}.{name}"
            return cls, f"{mod_name}:{cls_name}"
    return None


def _import_optional(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _preferred_selector(args: argparse.Namespace) -> tuple[Any, str] | None:
    """首选构造路径:``<--selector-module>`` 里的 Phase 3 实现。

    有 checkpoint 就 ``cls.load(path)``;没有就 ``cls(config_from_args(args))``
    —— 后者正是"没给 --selector-ckpt 则用随机初始化的新 selector"。架构旋钮由
    该模块自己的 ``SelectorConfig`` 默认值决定,本文件只透传同名 CLI 值。
    """
    module = _import_optional(args.selector_module)
    if module is None:
        return None
    cls = next((getattr(module, n) for n in _SELECTOR_CLASSES
                if getattr(module, n, None) is not None), None)
    if cls is None:
        return None
    name = cls.__name__
    if args.selector_ckpt is not None:
        loader = next((getattr(cls, n) for n in ("load", "from_checkpoint")
                       if hasattr(cls, n)), None)
        if loader is None:
            return None
        selector = _call_filtered(loader, path=args.selector_ckpt,
                                  ckpt=args.selector_ckpt,
                                  checkpoint=args.selector_ckpt,
                                  map_location=args.device, device=args.device)
        origin = f"{args.selector_module}:{name}.load"
    else:
        cfg_fn = getattr(module, "config_from_args", None)
        if callable(cfg_fn):
            # note (luojiaxuan): config_from_args 用 getattr(args, k, default) 取值,
            # 故先滤掉 None —— 否则我这边"没给就是 None"的可选项会把对方的默认值
            # 覆盖成 None,SelectorConfig 校验直接炸。
            clean = argparse.Namespace(
                **{k: v for k, v in vars(args).items() if v is not None})
            selector = cls(cfg_fn(clean))
        else:
            selector = cls()
        origin = f"{args.selector_module}:{name}(new)"
    if args.recent_strength is not None:
        init = getattr(selector, "init_recent_bias", None)
        if callable(init):
            # note (luojiaxuan): 路线 §4 的 "modest recent-B 初始化"(不是极强 KEEP
            # bias)。默认不动 —— 强度归 selector 自己的 SelectorConfig 默认值管。
            _call_filtered(init, strength=args.recent_strength)
    mover = getattr(selector, "to", None)
    if callable(mover):
        selector = mover(args.device)
    # note (luojiaxuan): rollout 是**推理**:eval() 关掉 backbone 的 dropout。
    # 不关的话同一 state 两次打分不同,记进 JSONL 的 log π_old 既不可复现,
    # 也不是训练端(同 checkpoint 重算)能对上的那个数。
    trainer_mode = getattr(selector, "eval", None)
    if callable(trainer_mode):
        trainer_mode()
    return selector, origin


def resolve_selector(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """按 ``--selector-arm`` 造出 selector,并返回一份写进每行记录的 meta。"""
    if args.selector_arm != "learned" and args.selector_ckpt is not None:
        raise SystemExit(
            f"--selector-arm {args.selector_arm} 是无参数基线臂,不接受 --selector-ckpt")
    if args.selector_arm == "recent":
        return RecentBSelector(), {"kind": "recent_b", "trainable": False}
    if args.selector_arm == "random":
        return (RandomBSelector(seed=args.rollout_seed),
                {"kind": "random_b", "trainable": False, "seed": args.rollout_seed})

    stub_meta = {"kind": "stub_hash_fake", "trainable": False,
                 "note": "dry-run 专用假 selector,真跑禁止使用"}
    found = None
    try:
        if args.selector_factory is None:
            found = _preferred_selector(args)
            if found is not None:
                selector, origin = found
                if not isinstance(selector, SelectorPolicy):
                    raise SystemExit(
                        f"{origin} 造出的对象没有实现 SelectorPolicy(需要 "
                        "score_subsets/sample/argmax 三个方法)")
                return selector, {
                    "kind": "learned", "trainable": True, "origin": origin,
                    "checkpoint": (str(args.selector_ckpt) if args.selector_ckpt
                                   else None),
                    "randomly_initialized": args.selector_ckpt is None,
                    "model": (getattr(selector, "describe", lambda: {})()
                              if callable(getattr(selector, "describe", None))
                              else getattr(selector, "metadata", {}))}
        found = _discover_factory(args.selector_factory, args.selector_module)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        if not args.dry_run:
            raise
        print(json.dumps({"tag": "selector_discovery_failed", "error": str(exc)},
                         ensure_ascii=False), flush=True)
    if found is None:
        if args.dry_run:
            return StubHashSelector(seed=args.rollout_seed), stub_meta
        raise SystemExit(
            "找不到 Phase 3 selector 实现。请用 --selector-factory MODULE:ATTR 指定"
            f"(自动查找过 {list(_SELECTOR_MODULES)}),或加 --dry-run 走假 selector")
    factory, origin = found
    try:
        selector = _call_factory(factory, args)
    except Exception as exc:  # noqa: BLE001
        # note (luojiaxuan): dry-run 的职责是验**本文件**的管线,不是验别人的模型能不能
        # 在 CPU 上构造出来;真实 selector 造不出来时退回假 selector 并如实记账,
        # 免得一个并行开发中的模块把管线自检整个卡死。真跑照常炸。
        if not args.dry_run:
            raise
        print(json.dumps({"tag": "selector_build_failed", "origin": origin,
                          "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False), flush=True)
        return StubHashSelector(seed=args.rollout_seed), dict(stub_meta,
                                                              failed_origin=origin)
    if not isinstance(selector, SelectorPolicy):
        raise SystemExit(
            f"{origin} 造出的对象没有实现 SelectorPolicy(需要 score_subsets/"
            "sample/argmax 三个方法)")
    eval_mode = getattr(selector, "eval", None)
    if callable(eval_mode):
        eval_mode()          # note (luojiaxuan): 同上 —— rollout 恒推理,关 dropout。
    meta: dict[str, Any] = {
        "kind": "learned", "trainable": True, "origin": origin,
        "checkpoint": str(args.selector_ckpt) if args.selector_ckpt else None,
        "randomly_initialized": args.selector_ckpt is None}
    extra = getattr(selector, "metadata", None)
    if isinstance(extra, dict):
        meta["model"] = extra
    return selector, meta


class StateFactory:
    """每步 ``state_repr`` 的构造口径 —— **必须与 GRPO trainer 完全一致**。

    优先用 selector 模块自带的 ``SelectorStateBuilder``(它知道自己的 state 长什么
    样);拿不到就退回 Phase 0 冻结契约里的 ``policy_io.SelectorState.from_bank``。
    两条路都按签名过滤 kwargs 超集,与 ``train_selector_grpo.forward_step`` 传的
    别名集合逐字对齐,所以训练端重算 log π 时能重建出同一个 state。

    ``origin`` 会写进每行记录的 ``selector.state_builder`` —— 训练端只要比对这个
    字段就知道两侧是不是同一条构造路径,不至于默默用两个 state 训一个策略。
    """

    def __init__(self, builder_call: Callable[..., Any] | None, origin: str,
                 *, dry_run: bool) -> None:
        self.builder_call = builder_call
        self.origin = origin
        self.dry_run = dry_run

    def __call__(self, *, selector: Any, task_instruction: str,
                 history_actions: Sequence[dict[str, Any]],
                 history_lines: Sequence[str], bank: HistoryFrameBank,
                 current_step: int, candidates: Sequence[int], budget: int) -> Any:
        if self.builder_call is None:
            return SelectorState.from_bank(
                task_instruction=task_instruction, history_actions=history_actions,
                bank=bank, current_step=current_step)
        pool = {
            "policy": selector, "selector": selector,
            "task_instruction": task_instruction, "instruction": task_instruction,
            "goal": task_instruction,
            "history_actions": list(history_actions),
            "actions": list(history_actions),
            "history_action_lines": list(history_lines),
            "bank": bank, "frame_bank": bank,
            "frame_features": (lambda j: bank.feature(j)),
            "current_feature": (bank.feature(current_step)
                                if bank.feature_provider is not None else None),
            "current_step": current_step, "step": current_step,
            "candidates": list(candidates),
            "recent": list(bank.recent(budget, current_step)),
            "budget": budget, "b": budget,
        }
        try:
            return _call_filtered(self.builder_call, **pool)
        except Exception as exc:  # noqa: BLE001
            # note (luojiaxuan): **不再有"dry-run 时退回最小载体"这条路** —— 那条
            # 退路在集成时反噬过一次:参数化 selector 的 score_subsets 只认自己
            # builder 产出的 state,退化后的 policy_io.SelectorState 一进去就抛,
            # 于是真正的病因(帧特征形状不对)被一条无关的 "state_repr 类型错误"
            # 盖住。state 口径是 log π 的定义域,dry-run 也不许悄悄换,只许炸。
            raise RuntimeError(
                f"{self.origin} 造不出 state(step={current_step}, "
                f"candidates={list(candidates)}):{type(exc).__name__}: {exc} —— "
                "state 口径决定 log π 的定义域,两侧必须同一条构造路径,故此处不做"
                "退化;请修好 state builder 或用 --state-builder 指定正确的实现"
            ) from exc


def persist_selector(args: argparse.Namespace, selector: Any,
                     selector_meta: dict[str, Any]) -> Path | None:
    """把本次实际使用的 selector 权重落盘,返回路径(基线臂返回 None)。

    # note (luojiaxuan): 这是"训练→采样"闭环里最容易漏掉的一环。没给
    # ``--selector-ckpt`` 时本文件会**当场随机初始化**一个 selector 跑完整批
    # rollout;若不把这份权重存下来,那批 JSONL 里的 log π_old 就成了孤儿 ——
    # train_selector_grpo 只能拿另一个随机初始化的模型去重算 log π_θ,
    # ratio 从第一步起就是两个无关分布的比值,而表面上训练照跑不误。
    # 故随机初始化时**默认**存到 ``<out>.init.pt``,不需要用户记得加开关。
    """
    if selector_meta.get("kind") != "learned":
        return None
    target = args.save_selector
    if target is None:
        if args.selector_ckpt is not None:
            return None                      # 权重来自现成 ckpt,无须再存一份
        target = args.out.with_suffix(args.out.suffix + ".init.pt")
    target = Path(target).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    saver = getattr(selector, "save", None)
    if callable(saver):
        saver(target)
    else:
        import torch

        torch.save({"state_dict": selector.state_dict()}, target)
    print(json.dumps({"tag": "selector_saved", "path": str(target),
                      "reason": ("explicit --save-selector" if args.save_selector
                                 else "randomly initialized selector")},
                     ensure_ascii=False), flush=True)
    return target


def resolve_state_factory(args: argparse.Namespace, selector: Any,
                          selector_meta: dict[str, Any],
                          provider: Callable[[int, str], Any] | None) -> StateFactory:
    """挑 state 构造路径。基线臂与假 selector 恒走冻结契约的最小载体。"""
    if selector_meta.get("kind") != "learned":
        # note (luojiaxuan): recent/random/stub 是按 policy_io 的最小 state 定义的
        # (只读 candidates),给它们套模型侧的 state builder 既无意义又会改行为。
        return StateFactory(None, "policy_io.SelectorState.from_bank",
                            dry_run=args.dry_run)
    cls: Any = None
    origin = ""
    if args.state_builder:
        mod_name, _, attr = args.state_builder.partition(":")
        if not attr:
            raise SystemExit(f"--state-builder 需要 MODULE:ATTR 形式,收到 {args.state_builder!r}")
        cls = getattr(importlib.import_module(mod_name), attr, None)
        if cls is None:
            raise SystemExit(f"{mod_name} 里没有 {attr!r}")
        origin = f"{mod_name}:{attr}"
    else:
        try:
            module = importlib.import_module(args.selector_module)
        except ImportError:
            module = None
        cls = getattr(module, "SelectorStateBuilder", None) if module else None
        origin = f"{args.selector_module}:SelectorStateBuilder"
    if cls is None:
        return StateFactory(None, "policy_io.SelectorState.from_bank",
                            dry_run=args.dry_run)
    try:
        builder = _call_filtered(
            cls, embedder=_resolve_embedder(args, selector), budget=args.budget,
            b=args.budget, device=args.device, feature_cache=provider,
            feature_provider=provider, features=provider)
    except Exception as exc:  # noqa: BLE001
        if not args.dry_run:
            raise SystemExit(
                f"{origin} 构造失败({type(exc).__name__}: {exc})。它决定 state 口径,"
                "与训练端必须一致,故真跑不做退化;可用 --state-builder 显式指定")
        print(json.dumps({"tag": "state_builder_unavailable", "origin": origin,
                          "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False), flush=True)
        return StateFactory(None, "policy_io.SelectorState.from_bank",
                            dry_run=args.dry_run)
    call = next((getattr(builder, n) for n in ("build", "build_state", "__call__")
                 if hasattr(builder, n)), None)
    if call is None:
        raise SystemExit(f"{origin} 没有 build/build_state/__call__")
    return StateFactory(call, origin, dry_run=args.dry_run)


def _resolve_embedder(args: argparse.Namespace, selector: Any) -> Any:
    """state builder 需要的冻结文本嵌入器:selector 自带 > 冻结基座 > dry-run 假的。"""
    for name in ("embedder", "text_embedder"):
        got = getattr(selector, name, None)
        if got is not None:
            return got
    maker = getattr(selector, "make_embedder", None)
    if callable(maker):
        return maker()
    try:
        module = importlib.import_module(args.selector_module)
    except ImportError:
        return None
    frozen = getattr(module, "FrozenTextEmbedder", None)
    if frozen is not None and args.model_dir is not None:
        return _call_filtered(frozen.from_model_dir, model_dir=args.model_dir,
                              device=args.device)
    dummy = getattr(module, "DummyTextEmbedder", None)
    if dummy is not None and args.dry_run:
        return dummy()
    return None


def resolve_feature_provider(selector: Any, args: argparse.Namespace
                             ) -> Callable[[int, str], Any] | None:
    """帧特征来源(``--feature-source``)。

    ``auto``:selector 自带 > dry-run 走 ``features`` 的 ``DummyFeatureExtractor``
    / 真跑走冻结视觉塔;两条路都由 ``causalcache_agentic.features.build_cache``
    产出(与训练端**同一个工厂、同一份 spec**),故重算 log π 时特征逐位相同。
    ``fake``(强制假特征)/ ``module``(强制冻结特征)/ ``none`` 便于排障。
    """
    src = args.feature_source
    if src == "none":
        return None
    if src == "auto":
        provider = getattr(selector, "feature_provider", None)
        if callable(provider):
            return provider
        maker = getattr(selector, "make_feature_provider", None)
        if callable(maker):
            return maker()
    dummy = (src == "fake") or (src == "auto" and args.dry_run)
    module = _import_optional("causalcache_agentic.features")
    builder = getattr(module, "build_cache", None) if module else None
    if not callable(builder):
        raise SystemExit(
            "causalcache_agentic.features.build_cache 不可用,取不到帧特征。"
            "它是 rollout 与训练端共用的唯一特征入口,没有第二实现可退 —— "
            "要跑没有特征的最小管线请用 --feature-source none")
    return builder(feature_spec(args, dummy=dummy))


# ---------------------------------------------------------------- 单步决策

def choose_subset(selector: Any, state_repr: SelectorState,
                  candidates: Sequence[int], budget: int, *, temperature: float,
                  greedy: bool, delegate: bool, rng: random.Random
                  ) -> tuple[tuple[int, ...], float, dict[str, Any]]:
    """选一个 subset,返回 (升序 subset, **采样时刻**的 log π(S|state), 诊断)。

    默认 ``enumerate`` 模式:枚举全部 C(n, B) subset → ``score_subsets`` 批式打分
    → 除以温度 → log-softmax → 采样。好处有三:温度对任何 selector 实现统一生效;
    log-prob 是精确值而不是事后近似;基线臂(recent 的 0/-inf、random 的全 0)
    在同一条路径上分别退化成点质量与均匀分布,评测与训练不走两套代码。

    ``greedy=True``(评测)时执行的策略本身是确定性的,故 log π ≡ 0;此时把
    tempered 分布下的 log-prob 另记为 ``softmax_logprob`` 供诊断。
    """
    subsets = enumerate_subsets(candidates, budget)
    diag: dict[str, Any] = {"n_subsets": len(subsets), "entropy": 0.0,
                            "softmax_logprob": 0.0, "mode": "delegate" if delegate
                            else "enumerate"}
    if len(subsets) <= 1:
        # note (luojiaxuan): 早期步只有一个合法动作 —— 不是采样,log-prob 恒为 0。
        diag["forced"] = True
        return (subsets[0] if subsets else ()), 0.0, diag

    if delegate:
        if greedy:
            picked = tuple(int(j) for j in selector.argmax(
                state_repr, list(candidates), budget))
            return picked, 0.0, diag
        picked_raw, logp = selector.sample(state_repr, list(candidates), budget)
        picked = tuple(int(j) for j in picked_raw)
        diag["softmax_logprob"] = float(logp)
        return picked, float(logp), diag

    scores = _as_scores(selector.score_subsets(state_repr, subsets), len(subsets))
    if temperature <= 0.0:
        raise ValueError(f"--temperature 必须 > 0,收到 {temperature!r}")
    logp = _log_softmax(scores / temperature)
    diag["entropy"] = round(_entropy(logp), 6)
    if greedy:
        idx = int(np.argmax(logp))
        diag["softmax_logprob"] = float(logp[idx])
        return subsets[idx], 0.0, diag
    probs = np.exp(np.where(np.isfinite(logp), logp, -np.inf))
    draw = rng.random() * float(probs.sum())
    idx = int(np.searchsorted(np.cumsum(probs), draw, side="left"))
    idx = min(max(idx, 0), len(subsets) - 1)
    chosen = float(logp[idx])
    if not math.isfinite(chosen):
        raise ValueError(f"采到了零概率 subset {subsets[idx]}(log-prob={chosen})")
    diag["softmax_logprob"] = chosen
    return subsets[idx], chosen, diag


# ---------------------------------------------------------------- 单条 rollout

def rollout_dir(shot_root: Path, task_id: str, arm: str, rollout_id: int) -> Path:
    """**每条 rollout 一个独立目录** —— 同组不同 rollout 会分叉,共用目录必然串图。"""
    return shot_root / _safe_name(task_id) / arm / f"r{rollout_id:03d}"


def _dry_choice(task: TaskSpec, step: int, subset: Sequence[int],
                logprob: float) -> SubsetChoice:
    """给 dry-run 假 executor 用的 ctx。**只在 --dry-run 下构造**,真跑不碰 probe。"""
    required, decision = read_probe(task)
    return SubsetChoice(
        subset=tuple(subset), logprob=logprob,
        required_available=tuple(j for j in required if j < step),
        required_missing=(), is_decision_step=(decision == step),
        covered=subset_is_solvable(task, step, tuple(subset)))


def run_rollout(task: TaskSpec, *, env: GUIEnv, policy: Any, selector: Any,
                parse: Callable[[str], dict[str, Any] | None], args: argparse.Namespace,
                arm: str, group_id: str, rollout_id: int,
                feature_provider: Callable[[int, str], Any] | None,
                selector_meta: dict[str, Any],
                state_factory: StateFactory) -> dict[str, Any]:
    """闭环跑一条 rollout,返回一行可直接落 JSONL 的记录。

    每步:渲染当前帧 → 登记进 bank → selector 采样 subset → 官方 prompt →
    executor greedy 生成 → 解析动作 → env.step。终止于 policy 自己 terminate
    或步数上限;成功与否**只**由 ``env.verify()`` 的声明式断言给出(terminal-only)。
    """
    builder = PolicyInputBuilder(budget=args.budget)
    bank = HistoryFrameBank(feature_provider=feature_provider)
    shot_dir = rollout_dir(args.shot_dir, task.task_id, arm, rollout_id)
    shot_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(stable_seed(args.rollout_seed, task.task_id, arm, rollout_id))

    env.reset(task)
    digest = state_digest(env.state)
    history_actions: list[dict[str, Any]] = []
    history_lines: list[str] = []
    steps: list[dict[str, Any]] = []
    limit = task.max_steps if args.max_steps is None else min(task.max_steps,
                                                              args.max_steps)
    parse_failures = step_errors = 0
    logprob_sum = 0.0
    started = time.perf_counter()

    for k in range(limit):
        shot = str(shot_dir / f"step{k:02d}.png")
        if args.skip_render:
            # note (luojiaxuan): dry-run 提速开关。policy_io 只传路径不读图,假
            # executor 也不读图,故跳过 PNG 编码不改变管线结构;真跑禁止使用。
            Path(shot).parent.mkdir(parents=True, exist_ok=True)
        else:
            env.render(shot)
        bank.add(k, shot)

        cands = bank.candidates(k)
        if args.max_candidates:
            cands = cands[-args.max_candidates:]
        with inference_mode():
            state_repr = state_factory(
                selector=selector, task_instruction=task.instruction,
                history_actions=history_actions, history_lines=history_lines,
                bank=bank, current_step=k, candidates=cands, budget=args.budget)
            subset, logprob, sel_diag = choose_subset(
                selector, state_repr, cands, args.budget,
                temperature=args.temperature, greedy=args.selector_greedy,
                delegate=(args.sampling_mode == "delegate"), rng=rng)
        # note (luojiaxuan): fail-loud —— selector 给出非法 subset(越界/重复/基数
        # 不对)必须当场炸,否则 prompt 会静默错位,整批 RL 数据作废。
        builder.validate_subset(subset, bank, k)
        if not math.isfinite(logprob) or logprob > 1e-9:
            raise ValueError(
                f"{task.task_id} step {k}: log-prob 必须是有限非正数,得到 {logprob!r}")
        logprob_sum += logprob

        messages = builder.build(task.instruction, history_actions, subset, bank, k)
        ctx: dict[str, Any] = {"task": task, "arm": arm, "step": k,
                               "choice": _dry_choice(task, k, subset, logprob)
                               if args.dry_run else None}
        raw = policy.generate(messages, ctx)
        parsed = parse(raw)
        if not isinstance(parsed, dict) or "action" not in parsed:
            parse_failures += 1
            action = dict(FALLBACK_ACTION)
        else:
            action = normalize_action(parsed)
        try:
            line = action_line(action)
        except (KeyError, TypeError, ValueError):
            line = f"Action: {json.dumps(action, ensure_ascii=False)}"
        _obs, done, info = env.step(action)
        if info["last_error"]:
            step_errors += 1

        recent = bank.recent(args.budget, k)
        # note (luojiaxuan): **编号口径(与 train_selector_grpo.StepSample 逐字一致)**:
        # ``candidates`` / ``chosen_subset`` 里的整数是 **env 的 0-based 决策步号**
        # (== policy_io 的官方事件号,两者同值不做偏移),**不是**候选列表里的下标。
        # 于是恒有 candidates == [1, step-1] ∩ 已入 bank 的帧(帧 0 被折叠成文本、
        # 当前帧 step 恒在末轮,两者都不可选),且 screenshot 名里的 stepKK == step。
        # 训练端按同一口径读回来重算 log π;哪一侧改成下标口径,subset 就会静默
        # 指向另一批历史帧,而 JSONL 看上去完全正常。
        steps.append({
            "step": k, "chosen_subset": list(subset), "candidates": list(cands),
            "selector_logprob": logprob, "action": dict(action),
            "action_line": line, "screenshot": shot,
            "n_candidates": len(cands), "n_subsets": sel_diag["n_subsets"],
            "softmax_logprob": sel_diag["softmax_logprob"],
            "entropy": sel_diag["entropy"],
            "is_recent": tuple(subset) == recent,
            "active_app": info["active_app"], "last_error": info["last_error"]})
        history_actions.append(dict(action))
        history_lines.append(line)
        if done:
            break

    success = env.verify()
    probe_required, probe_decision = read_probe(task)
    row = {
        "task_id": task.task_id, "template_id": task.template_id,
        "family": task.family, "regime": task.regime, "seed": task.seed,
        # note (luojiaxuan): instruction 是 state builder 的文本输入之一,直接落盘
        # 而不是让训练端从 task_source 再 generate_task 一次。重建虽然确定性,但多
        # 一次口径漂移的机会(模板改一个字、seed 混合函数动一下,重建出来的
        # instruction 就不是当时喂给 selector 的那句),log π 会静默偏掉。
        "instruction": task.instruction,
        "group_id": group_id, "rollout_id": rollout_id, "arm": arm,
        "success": bool(success), "reason": env.reason or "actions_exhausted",
        "n_steps": len(steps), "steps": steps,
        "selector_logprob_sum": logprob_sum,
        # ---- 以下为附加字段(schema 的必需部分在上面) ----
        "reward": 1.0 if success else 0.0,          # terminal-only,GRPO 直接读
        "budget": args.budget, "temperature": args.temperature,
        "greedy": bool(args.selector_greedy), "sampling_mode": args.sampling_mode,
        "group_size": args.group_size,
        "initial_state_digest": digest, "max_steps": limit,
        "parse_failures": parse_failures, "step_errors": step_errors,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "shot_dir": str(shot_dir), "shots_kept": not args.cleanup_shots,
        "final_flags": dict(env.state.flags),
        "n_recent_steps": sum(1 for s in steps if s["is_recent"]),
        # note (luojiaxuan): trainer 用它 + steps[*].action 逐位重建每一帧
        # (环境确定性),所以 --cleanup-shots 删图不影响重算 log π。
        "task_source": {"template_id": task.template_id, "seed": task.seed,
                        "regime": task.regime},
        "selector": selector_meta,
        "policy": getattr(policy, "metadata", {}),
        # note (luojiaxuan): 诊断专用 —— 结构上不进 selector 输入,也不作 reward。
        "diagnostics": {"required_steps": list(probe_required),
                        "decision_step": probe_decision},
    }
    if args.cleanup_shots:
        for png in shot_dir.glob("*.png"):
            png.unlink(missing_ok=True)
    return row


# ---------------------------------------------------------------- 分组

def effective_group_size(args: argparse.Namespace) -> int:
    """确定性臂(recent / greedy)组内 G 条完全同轨,自动收敛到 1 条以免烧显卡。"""
    deterministic = args.selector_arm == "recent" or args.selector_greedy
    if deterministic and not args.allow_degenerate_group:
        return 1
    return args.group_size


def run_group(task: TaskSpec, *, env: GUIEnv, policy: Any, selector: Any,
              parse: Callable[[str], dict[str, Any] | None],
              args: argparse.Namespace, group_size: int, done: dict[tuple, str],
              sink: Any, feature_provider: Callable[[int, str], Any] | None,
              selector_meta: dict[str, Any],
              state_factory: StateFactory) -> list[dict[str, Any]]:
    """同一 TaskSpec 产 group_size 条 rollout,**逐条校验初始状态摘要一致**。"""
    group_id = task.task_id
    rows: list[dict[str, Any]] = []
    baseline = done.get(("digest", group_id))
    for rid in range(group_size):
        key = (task.task_id, group_id, rid, args.selector_arm)
        if key in done:
            continue
        row = run_rollout(task, env=env, policy=policy, selector=selector,
                          parse=parse, args=args, arm=args.selector_arm,
                          group_id=group_id, rollout_id=rid,
                          feature_provider=feature_provider,
                          selector_meta=selector_meta, state_factory=state_factory)
        digest = row["initial_state_digest"]
        if baseline is None:
            baseline = digest
        elif digest != baseline:
            # note (luojiaxuan): 同组必须同初始状态 —— 不一致意味着 reset 没隔离干净
            # (executor 原地改了 state,或 TaskSpec 被复用后污染),此时组内相对
            # 优势在比较两个不同的环境,GRPO 学到的全是噪声。fail-loud。
            raise RuntimeError(
                f"group {group_id} 初始状态不一致:rollout {rid} 摘要 {digest} "
                f"≠ 组基准 {baseline}")
        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        sink.flush()
        rows.append(row)
    return rows


# ---------------------------------------------------------------- IO

def read_done(path: Path) -> dict[tuple, str]:
    """断点续跑索引:(task_id, group_id, rollout_id, arm) → 1,外加各组初始摘要。"""
    done: dict[tuple, str] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not {"task_id", "group_id", "rollout_id"} <= set(row):
                continue
            done[(str(row["task_id"]), str(row["group_id"]),
                  int(row["rollout_id"]), str(row.get("arm", "")))] = "1"
            digest = row.get("initial_state_digest")
            if digest:
                done.setdefault(("digest", str(row["group_id"])), str(digest))
    return done


def build_tasks(args: argparse.Namespace) -> list[TaskSpec]:
    """实例化任务(随机只发生在这一步,带显式 seed;同 seed 跨进程可复现)。"""
    from causalcache_agentic import tasks as tasks_module

    if args.families:
        fams = [f for item in args.families for f in item.split(",") if f]
        out = tasks_module.generate_batch(args.n_tasks, args.task_seed, families=fams)
    else:
        out = tasks_module.make_dataset(args.tasks_split, args.n_tasks, args.task_seed)
    return out[: args.limit_tasks] if args.limit_tasks else out


# ---------------------------------------------------------------- CLI

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rollout_selector",
        description="Phase 3 分组 rollout worker(selector-only GRPO 数据引擎)")
    # 任务
    p.add_argument("--tasks-split", default="train",
                   choices=("train", "syn_iid", "syn_ood"))
    p.add_argument("--families", nargs="+", default=None,
                   help="按 family/template_id 过滤(逗号或空格分隔),给了就忽略 --tasks-split")
    p.add_argument("--n-tasks", type=int, default=64)
    p.add_argument("--task-seed", type=int, default=0)
    p.add_argument("--limit-tasks", type=int, default=0, help="0 = 不限")
    # 分组与预算
    p.add_argument("--group-size", type=int, default=8, help="G:同 task 同初始状态的 rollout 数")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="B(primary=2)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="额外步数上限(与 task.max_steps 取小)")
    p.add_argument("--allow-degenerate-group", action="store_true",
                   help="确定性臂也照跑 G 条(默认自动收敛到 1 条)")
    # executor(冻结 GUI-Owl + 可选 Phase 2 adapter)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--snapshot-manifest", type=Path, default=None)
    p.add_argument("--adapter", type=Path, default=None,
                   help="Phase 2 policy_mem_sft 的 LoRA adapter;不给 = 原始 GUI-Owl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--visual-tokens", type=int, default=2560)
    p.add_argument("--max-new-tokens", type=int, default=128)
    # selector
    p.add_argument("--selector-arm", default="learned", choices=ARMS)
    p.add_argument("--selector-ckpt", type=Path, default=None,
                   help="learned 臂的 checkpoint;不给 = 随机初始化的新 selector")
    p.add_argument("--save-selector", type=Path, default=None,
                   help="把本次实际使用的 selector 存到该路径。**随机初始化时**"
                        "(没给 --selector-ckpt)即使不指定也会自动存到 "
                        "<out>.init.pt —— GRPO 必须从产出这批 log π_old 的**同一份"
                        "权重**起训,否则重要性比从第一步就是错的")
    p.add_argument("--selector-factory", default=None,
                   help="MODULE:ATTR,显式指定 selector 工厂(自动发现失败时用)")
    p.add_argument("--selector-module", default=DEFAULT_SELECTOR_MODULE,
                   help="提供 SelectorStateBuilder / load_selector 的模块名"
                        "(与 train_selector_grpo.py 的同名参数必须一致)")
    p.add_argument("--state-builder", default=None,
                   help="MODULE:ATTR,显式指定 state builder;不给则用 "
                        "<--selector-module>.SelectorStateBuilder,再退回 "
                        "policy_io.SelectorState.from_bank")
    p.add_argument("--sampling-mode", default="enumerate",
                   choices=("enumerate", "delegate"),
                   help="enumerate=本文件枚举 C(n,B) 并做温度采样(默认);"
                        "delegate=直接调 selector.sample/argmax")
    p.add_argument("--selector-greedy", action="store_true",
                   help="评测用:argmax 而非采样(此时 log π ≡ 0)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="selector 采样温度(只作用于选帧;executor 恒 greedy)。"
                        "≠1 时训练端重算 log π 必须用同一温度")
    p.add_argument("--recent-strength", type=float, default=None,
                   help="新建 selector 时的 modest recent-B 先验强度;"
                        "不给 = 用 selector 自己的默认值")
    p.add_argument("--max-candidates", type=int, default=0,
                   help="只保留最近 M 个候选帧(0 = 全部);M 必须 ≥ B")
    p.add_argument("--feature-source", default="auto",
                   choices=("auto", "module", "fake", "none"),
                   help="帧特征来源:auto=selector 自带 > dry-run 假特征 / 真跑 "
                        "causalcache_agentic.features;module/fake/none 为强制指定")
    # note (luojiaxuan): 下面三个 dummy 旋钮**只有显式给了才透传**(0 / 负数 = 用
    # features 模块的默认值)。它们与 train_selector_grpo.py 的 --feature-dim /
    # --feature-tokens / --dummy-seed 一一对应,**必须两侧同时改**:假特征由
    # (dim, tokens, seed, dtype) 唯一决定,差一项 log π 一致性检查就会红。
    p.add_argument("--feature-dim", type=int, default=0,
                   help="假特征维度(0 = 用 features 的默认值;冻结特征维度由视觉塔决定)")
    p.add_argument("--feature-tokens", type=int, default=0,
                   help="假特征 token 数(0 = 用 features 的默认值)")
    p.add_argument("--dummy-seed", type=int, default=-1,
                   help="假特征 seed(<0 = 用 features 的默认值)")
    p.add_argument("--feature-dtype", default=None,
                   help="透传给 causalcache_agentic.features(如 float16)")
    p.add_argument("--feature-cache-items", type=int, default=None,
                   help="透传:特征 LRU 上限")
    p.add_argument("--feature-cache-dir", type=Path, default=None,
                   help="透传:特征落盘缓存目录")
    p.add_argument("--rollout-seed", type=int, default=0,
                   help="selector 采样 RNG 的根 seed(同 seed 逐条可复现)")
    # 输出
    p.add_argument("--out", type=Path, required=True, help="rollout JSONL")
    p.add_argument("--shot-dir", type=Path, default=None,
                   help="截图根目录(默认 <out 同级>/shots)")
    p.add_argument("--summary", type=Path, default=None, help="可选:运行汇总 JSON")
    p.add_argument("--cleanup-shots", action="store_true",
                   help="每条 rollout 跑完即删其 PNG(RL 阶段图很多,默认建议开)")
    p.add_argument("--skip-render", action="store_true",
                   help="不落 PNG(仅 --dry-run 提速用,真跑会拒绝)")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=20)
    # 假运行
    p.add_argument("--dry-run", action="store_true",
                   help="不加载模型,用假 executor + 假特征验证管线(无需 GPU)")
    p.add_argument("--dry-run-mode", default="informed", choices=("informed", "random"))
    return p.parse_args(list(argv) if argv is not None else None)


def _validate(args: argparse.Namespace) -> None:
    if args.group_size < 1:
        raise SystemExit("--group-size 必须 ≥ 1")
    if args.budget < 0:
        raise SystemExit("--budget 必须 ≥ 0")
    if args.temperature <= 0.0:
        raise SystemExit("--temperature 必须 > 0")
    if args.max_candidates and args.max_candidates < args.budget:
        raise SystemExit(f"--max-candidates({args.max_candidates}) 必须 ≥ --budget"
                         f"({args.budget}),否则 recent-B 会变成非法动作")
    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        raise SystemExit("--shard-index 必须落在 [0, --shard-count)")
    if args.sampling_mode == "delegate" and args.temperature != 1.0:
        # note (luojiaxuan): delegate 模式下采样发生在 selector 内部,本文件没有
        # 插手的余地 —— 静默忽略温度会让"我调了温度"变成幻觉,故直接拒绝。
        raise SystemExit("--sampling-mode delegate 下 --temperature 不生效(采样在"
                         " selector 内部);要调温度请用默认的 enumerate 模式")
    if not args.dry_run:
        if args.model_dir is None or args.snapshot_manifest is None:
            raise SystemExit("真跑必须给 --model-dir 与 --snapshot-manifest(冻结守卫);"
                             "只想验管线请加 --dry-run")
        if args.skip_render:
            raise SystemExit("--skip-render 只允许配合 --dry-run(真跑必须有真截图)")
    else:
        # note (luojiaxuan): dry-run 的前提是"本机没有 GPU",device 一律压成 cpu,
        # 免得把 cuda:0 传给一个真的 selector 实现后在 macOS 上直接崩。
        args.device = "cpu"
    args.out = args.out.expanduser()
    args.shot_dir = (args.shot_dir or args.out.parent / "shots").expanduser()
    if args.shard_count > 1:
        args.out = args.out.with_name(
            f"{args.out.stem}.shard{args.shard_index:02d}{args.out.suffix}")
        if args.summary:
            args.summary = args.summary.with_name(
                f"{args.summary.stem}.shard{args.shard_index:02d}{args.summary.suffix}")


def _progress(tag: str, n_new: int, rows: Sequence[dict[str, Any]], t0: float,
              extra: dict[str, Any] | None = None) -> dict[str, Any]:
    elapsed = max(time.perf_counter() - t0, 1e-9)
    steps = [r["n_steps"] for r in rows]
    payload = {"tag": tag, "rollouts": n_new,
               "rollouts_per_sec": round(n_new / elapsed, 3),
               "mean_steps": round(sum(steps) / len(steps), 2) if steps else None,
               "success_rate": (round(sum(1 for r in rows if r["success"]) / len(rows), 4)
                                if rows else None),
               "elapsed_sec": round(elapsed, 1)}
    payload.update(extra or {})
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def _group_reward_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """GRPO 关心的组级健康度:多少组的 reward 有方差(全 0 / 全 1 的组学不到东西)。"""
    by_group: dict[str, list[float]] = {}
    for row in rows:
        by_group.setdefault(str(row["group_id"]), []).append(float(row["reward"]))
    usable = sum(1 for v in by_group.values() if len(v) > 1 and 0 < sum(v) < len(v))
    return {"n_groups": len(by_group), "groups_with_reward_variance": usable,
            "frac_groups_with_variance": (round(usable / len(by_group), 4)
                                          if by_group else None),
            "all_zero_groups": sum(1 for v in by_group.values() if sum(v) == 0),
            "all_one_groups": sum(1 for v in by_group.values()
                                  if v and sum(v) == len(v))}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate(args)

    tasks = build_tasks(args)
    shard = [t for i, t in enumerate(tasks) if i % args.shard_count == args.shard_index]
    # note (luojiaxuan): **按 task 分片,不按 rollout 分片** —— 一个 group 必须整组
    # 落在同一个 shard 里,否则 GRPO 的组内相对优势会被拆到两个进程各算一半。
    group_size = effective_group_size(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.shot_dir.mkdir(parents=True, exist_ok=True)
    done = read_done(args.out)

    from rl_oracle_enumerate import parse_tool_call  # 与冻结评测逐字同源的解析器

    selector, selector_meta = resolve_selector(args)
    saved_ckpt = persist_selector(args, selector, selector_meta)
    if saved_ckpt is not None:
        selector_meta = dict(selector_meta, initial_checkpoint=str(saved_ckpt))
    feature_provider = resolve_feature_provider(selector, args)
    state_factory = resolve_state_factory(args, selector, selector_meta,
                                          feature_provider)
    # note (luojiaxuan): state 口径写进每行记录 —— 训练端比对这一个字段就能发现
    # "两侧用了不同 state 构造路径"这种会静默毁掉重要性比的错配。
    selector_meta = dict(selector_meta, state_builder=state_factory.origin,
                         has_feature_provider=feature_provider is not None)
    policy: Any = (
        DryRunPolicy(args.dry_run_mode, args.rollout_seed) if args.dry_run
        else FrozenGUIOwlPolicy(args.model_dir, args.snapshot_manifest, args.device,
                                args.visual_tokens, args.max_new_tokens,
                                adapter=args.adapter))
    env = GUIEnv(auto_stop_on_success=False)

    print(json.dumps({
        "tag": "launch", "arm": args.selector_arm, "selector": selector_meta,
        "group_size_requested": args.group_size, "group_size_effective": group_size,
        "budget": args.budget, "temperature": args.temperature,
        "greedy": bool(args.selector_greedy), "sampling_mode": args.sampling_mode,
        "n_tasks_in_shard": len(shard), "shard": [args.shard_index, args.shard_count],
        "resume_rows": sum(1 for k in done if k[0] != "digest"),
        "out": str(args.out), "dry_run": bool(args.dry_run),
        "reward": "terminal_verifier_only"}, ensure_ascii=False), flush=True)
    if args.temperature != 1.0:
        # note (luojiaxuan): selector 的 score_subsets 返回**未除温度**的 logits,
        # 温度是本文件加的。训练端若按 T=1 重算 log π_θ,重要性比 π_θ/π_old 会
        # 系统性偏,GRPO 的裁剪与更新方向一起失真 —— 故这里显式喊出来。
        print(json.dumps({
            "tag": "warning", "temperature": args.temperature,
            "message": "log π_old 记的是该温度下的分布;train_selector_grpo.py "
                       "重算 log π_θ 时必须用同一温度,否则重要性比失真"},
            ensure_ascii=False), flush=True)

    rows: list[dict[str, Any]] = []
    reported = 0
    t0 = time.perf_counter()
    with args.out.open("a", encoding="utf-8") as sink:
        for task in shard:
            fresh = run_group(task, env=env, policy=policy, selector=selector,
                              parse=parse_tool_call, args=args, group_size=group_size,
                              done=done, sink=sink, feature_provider=feature_provider,
                              selector_meta=selector_meta,
                              state_factory=state_factory)
            rows.extend(fresh)
            if args.progress_every and len(rows) - reported >= args.progress_every:
                reported = len(rows)
                _progress("progress", len(rows), rows, t0)

    summary = _progress("done", len(rows), rows, t0,
                        {"groups": _group_reward_stats(rows),
                         "arm": args.selector_arm, "out": str(args.out),
                         "parse_failures": sum(r["parse_failures"] for r in rows),
                         "recent_step_frac": _recent_frac(rows)})
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps({"meta": {
                "arm": args.selector_arm, "selector": selector_meta,
                "policy": getattr(policy, "metadata", {}),
                "tasks_split": args.tasks_split, "families": args.families,
                "n_tasks": args.n_tasks, "task_seed": args.task_seed,
                "group_size": group_size, "budget": args.budget,
                "temperature": args.temperature, "greedy": bool(args.selector_greedy),
                "rollout_seed": args.rollout_seed,
                "shard": [args.shard_index, args.shard_count],
                "reward": "terminal_verifier_only"},
                "summary": summary}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    return 0


def _recent_frac(rows: Sequence[dict[str, Any]]) -> float | None:
    """选中 recent-B 的步比例(路线验收表里的 "选 recent-2 比例")。"""
    total = sum(r["n_steps"] for r in rows)
    if not total:
        return None
    return round(sum(r["n_recent_steps"] for r in rows) / total, 4)


if __name__ == "__main__":
    raise SystemExit(main())
