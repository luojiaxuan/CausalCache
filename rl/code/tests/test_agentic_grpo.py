"""Phase 3 验收测试套件(selector-only GRPO 组件)。

运行方式(仓库根目录执行;`rl/`、`rl/code/` 都不是 Python 包,本文件自己修正 sys.path,
下面两种入口等价;**必须带 PYTHONDONTWRITEBYTECODE=1**,否则本机沙箱写 __pycache__ 会挂起):

    cd /Users/luojiaxuan/Documents/CausalCache/.claude/worktrees/sweet-easley-a4876e && \
        PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=rl/code:code \
        python3 rl/code/tests/test_agentic_grpo.py -v
    # 等价:… python3 -m unittest discover -s rl/code/tests -p 'test_agentic_grpo.py' -v

覆盖 `rl/docs/agentic_memory_rl_roadmap_20260813.md` **§5 Phase 3**(与 §9 禁止清单)的
验收条款(测试函数 → 条款):

| 测试 | 路线文档条款 |
|---|---|
| `test_selector_distribution`      | §5「动作空间 = 全部 C(N,2) subset」「接口必须是 z_θ(S given state)、基数 = B,不得写死 score_pair」「训练采样、测试 argmax」 |
| `test_permutation_equivariance`   | §5「subset 联合表示」:π(S) 只能由帧内容与 frame age/step index 决定,不得由候选在数组里的位置决定 |
| `test_recent_is_ordinary_action`  | §5「recent-2 是其中普通合法动作」「modest recent-B 初始化(**不是**极强 KEEP bias)」「防塌缩:entropy bonus、温度」 |
| `test_no_diagnostic_leak`         | §5「**不用** gold / pass-1 分布 / B=1 probe / milestone」+ §2「哪帧含哪个变量可留作诊断,**不得**作 selector primary RL reward」+ contract.py 对 `memory_probe` 的注释 |
| `test_grpo_zero_variance_group`   | §5「GRPO 用同组相对 reward」「entropy 与 group reward variance 监控」;全同 reward 组无相对信息 ⇒ 优势恒 0 ⇒ 零梯度,且必须被计数(§5 验收要报「有 reward variance 的 group 比例」) |
| `test_grpo_advantage_sign`        | §5「同 task instance 同初始 state 产 G 条、GRPO 用同组相对 reward」—— 成功优势 > 失败优势,且一次更新后成功 subset 的 logπ 上升 |
| `test_rollout_schema`             | §2「RL 数据:selector sampled subset 与 log-prob、terminal success、metadata」+ §5「同 task instance 同初始 state 产 G 条」+ recent-2 对照臂 |
| `test_terminal_reward_only`       | §0「训练主奖励**只有**最终任务成功 R∈{0,1}」+ §9 禁止清单(milestone reward / step-level 正确性 / gold 帧监督 / pass-1 draft / Gumbel-softmax 软选帧) |

**设计说明(为什么这份测试大量用反射)**:`features.py` / `selector_model.py` /
`scripts/rollout_selector.py` / `scripts/train_selector_grpo.py` 由并行 agent 实现。
本文件对**命名宽容、对行为严格**:所有调用都按参数名(经别名表 `_ALIAS`)绑定,
绑定不上必填参数时抛 `ContractError` 并打印签名与可提供的量,方便实现方对齐;
断言本身一律钉在**可观测行为**上(概率归一、置换不变、梯度方向、JSONL 字段)。
任一被测模块尚未落地或 import 失败时,相关 TestCase **skip 并打印真实 ImportError**。

整套在 CPU 上跑,无 GPU 依赖(`DummyFeatureExtractor` 假特征 + `--dry-run` 假 policy)。
注意:**luojiaxuan 的 macOS 上没有可用 torch**,依赖 torch 的 6 个用例在那台机器上恒
skip(只剩 rollout schema 与源码扫描两条),完整套必须在装了 torch 的机器上跑。

自检记录(2026-08-13,torch 2.13.0 CPU / Python 3.12 临时 venv,对**真实**并行实现):
8/8 通过,整套约 15 秒(渲染 21 帧 + 9 次 state 构建 + 2 次 rollout --dry-run 子进程)。
另做了 24 项**突变检验**(故意把被测行为改坏,确认对应用例真的会红),24/24 全部被
抓到:logprob_of 与 logits 不同源、argmax 恒返回 recent-B、打分混入数组位置、recent
先验默认过强(均值 0.675)、构造期即塌成点质量、state 挂 required_steps / regime、
零方差组给非零优势、跳过零方差组却不给原因、优势符号反、JSONL 缺 reward /
缺 selector_logprob / reward 非 0-1 / recent 臂选了非 recent-B / 同组 candidates 不一致、
以及 milestone/step_reward/required_steps-as-reward/gumbel 四类源码禁项(带否定语境的
自证注释与字符串正确豁免)。
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib
import importlib.util
import inspect
import itertools
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import tokenize
import unittest
from collections.abc import Mapping, Sequence
from typing import Any, Callable

# note (luojiaxuan): 本机沙箱写 __pycache__ 会挂起;子进程另经 env 传同名变量。
sys.dont_write_bytecode = True

_RL_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # …/rl/code
_REPO = os.path.dirname(os.path.dirname(_RL_CODE))                          # 仓库根
_CODE = os.path.join(_REPO, "code")
for _p in (_RL_CODE, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# note (luojiaxuan): 以下全是 **Phase 3 验收门槛**(路线文档 §5),不是科学指标;
# 要调整必须先改路线文档再改这里。
BUDGET_B = 2                    # primary budget B=2(路线 §0)
PROB_TOL = 1e-4                 # 概率归一 / 置换不变性容差(验收指定 1e-4)
LOGP_TOL = 1e-4                 # sample().logprob 与 logprob_of() 的一致性容差
RECENT_PRIOR_LO = 0.15          # 初始 π(recent-B) 下界(modest 先验,不是塌到 0)
RECENT_PRIOR_HI = 0.65          # 初始 π(recent-B) 上界(不是极强 KEEP bias)
RECENT_PRIOR_LOOSE = (0.10, 0.75)   # 单点观测的宽松带(均值走上面的窄带)
RECENT_POINTMASS_MAX = 0.85     # 任何候选规模下都不允许塌成点质量
ENTROPY_FRAC_MIN = 0.35         # 初始策略熵 ≥ 该比例 × log C(n,B)(防初始化即塌缩)
GRPO_LR = 0.05                  # 方向性验证用的大 lr(验收指定「大 lr 验证方向性」),配 SGD
ZERO_ADV_TOL = 1e-6             # 全同 reward 组的优势/梯度判零阈值
FIXTURE_FRAMES = 7              # 每个 fixture 真渲染的帧数(⇒ 候选 1..5,C(5,2)=10)
FIXTURE_TASKS = 3               # fixture 任务数(控制整套耗时;每帧渲染约 50ms)
SMALL_STEP = 4                  # 小候选集观测点:current_step=4 ⇒ 候选 1..3,C(3,2)=3
PRIOR_SEEDS = (0, 1, 2)         # 初始化先验的观测种子
SUBPROC_TIMEOUT = 240.0         # rollout --dry-run 子进程超时(秒)
DRY_RUN_TASKS = "2"             # --dry-run 采样的任务数(如脚本支持相应 flag)
DRY_RUN_GROUP = "2"             # --dry-run 的 group size(同上)

_MOD: dict[str, Any] = {}
_ERR: dict[str, str] = {}
for _n in ("contract", "policy_io", "tasks", "render", "executor", "env",
           "features", "selector_model"):
    try:
        _MOD[_n] = importlib.import_module(f"causalcache_agentic.{_n}")
    except Exception as exc:                                        # noqa: BLE001
        _MOD[_n], _ERR[_n] = None, f"{type(exc).__name__}: {exc}"
try:
    import numpy as np
except Exception as exc:                                            # noqa: BLE001
    np, _ERR["numpy"] = None, f"{type(exc).__name__}: {exc}"        # type: ignore[assignment]
try:
    import torch
except Exception as exc:                                            # noqa: BLE001
    torch, _ERR["torch"] = None, f"{type(exc).__name__}: {exc}"     # type: ignore[assignment]

_SCRIPT_PATH = {
    "rollout": os.path.join(_RL_CODE, "scripts", "rollout_selector.py"),
    "grpo": os.path.join(_RL_CODE, "scripts", "train_selector_grpo.py"),
}
_SCRIPT: dict[str, Any] = {}


def _load_script(key: str) -> Any:
    """按路径加载 scripts/ 下的脚本(该目录不是包,只能走 spec_from_file_location)。"""
    if key in _SCRIPT:
        return _SCRIPT[key]
    path = _SCRIPT_PATH[key]
    if not os.path.exists(path):
        _SCRIPT[key], _ERR[f"scripts.{key}"] = None, f"文件不存在: {path}"
        return None
    try:
        spec = importlib.util.spec_from_file_location(f"_agentic_script_{key}", path)
        mod = importlib.util.module_from_spec(spec)                 # type: ignore[arg-type]
        sys.modules[spec.name] = mod                                # type: ignore[union-attr]
        spec.loader.exec_module(mod)                                # type: ignore[union-attr]
    except Exception as exc:                                        # noqa: BLE001
        _SCRIPT[key], _ERR[f"scripts.{key}"] = None, f"{type(exc).__name__}: {exc}"
        return None
    _SCRIPT[key] = mod
    return mod


for _k in ("rollout", "grpo"):
    _load_script(_k)


def requires(*names: str) -> Callable[[Any], Any]:
    """缺依赖时把整个 TestCase 标成 skip,并打印真实 ImportError。

    names 取值:``causalcache_agentic`` 子模块名、``numpy``/``torch``、
    ``script:rollout``/``script:grpo``。
    """
    missing: list[str] = []
    for n in names:
        if n.startswith("script:"):
            key = n.split(":", 1)[1]
            if _SCRIPT.get(key) is None:
                missing.append(f"scripts.{key}")
        elif n == "numpy":
            if np is None:
                missing.append("numpy")
        elif n == "torch":
            if torch is None:
                missing.append("torch")
        elif _MOD.get(n) is None:
            missing.append(n)
    if not missing:
        return lambda obj: obj
    return unittest.skip("依赖尚未就绪:" + "; ".join(
        f"{n} -> {_ERR.get(n, '未实现')}" for n in missing))


# ------------------------------------------------------------------ 反射调用工具

class ContractError(AssertionError):
    """被测对象的公开接口与 Phase 3 契约不符(当作测试失败,不当作 skip)。"""


_MISS = object()

# note (luojiaxuan): 并行 agent 的参数命名不可预知,这里把「同义参数名」折叠成少数
# canonical key。绑定不上必填参数时报 ContractError 并列出签名,让实现方一眼看出
# 该改哪个名字 —— 测试对命名宽容、对行为严格。
_ALIAS: dict[str, str] = {
    "state": "state", "state_repr": "state", "selector_state": "state", "st": "state",
    "subsets": "subsets", "candidate_subsets": "subsets", "subset_list": "subsets",
    "subset": "subset", "chosen_subset": "subset", "chosen": "subset",
    "candidates": "candidates", "cands": "candidates", "candidate_events": "candidates",
    "events": "candidates", "candidate_steps": "candidates",
    "b": "b", "budget": "b", "k": "b", "n_select": "b", "b_budget": "b",
    "task": "task", "task_spec": "task", "spec": "task",
    "instruction": "instruction", "task_instruction": "instruction", "goal": "instruction",
    "bank": "bank", "frame_bank": "bank", "history_bank": "bank",
    "current_step": "current_step", "now": "current_step", "step": "current_step",
    "step_index": "current_step", "cur_step": "current_step",
    "history_actions": "history_actions", "actions": "history_actions",
    "action_history": "history_actions",
    "history_action_lines": "history_action_lines", "action_lines": "history_action_lines",
    "frame_features": "frame_features", "features": "frame_features",
    "candidate_features": "frame_features",
    "current_feature": "current_feature", "cur_feature": "current_feature",
    "recent": "recent", "recent_subset": "recent",
    "policy": "policy", "selector": "policy", "model": "policy",
    "cfg": "cfg", "config": "cfg", "selector_cfg": "cfg",
    "extractor": "extractor", "feature_extractor": "extractor", "cache": "extractor",
    "feature_cache": "extractor", "provider": "extractor", "feature_provider": "extractor",
    "embedder": "embedder", "text_embedder": "embedder",
    "feature_dim": "feature_dim", "feat_dim": "feature_dim", "dim": "feature_dim",
    "d_model": "feature_dim", "in_dim": "feature_dim", "input_dim": "feature_dim",
    "d_in": "feature_dim", "embed_dim": "feature_dim",
    "device": "device", "seed": "seed",
    "rewards": "rewards", "reward": "rewards", "group_rewards": "rewards", "r": "rewards",
    "group": "group", "groups": "groups",
}


# note (luojiaxuan): 实现方常把数值旋钮写成**无默认值的 keyword-only 参数**
# (例:advantages(group, *, adv_eps, min_group_size))。这些不是科学口径,测试给
# 一组中性值补上即可 —— 仍然只对**必填且 pool 里没有**的参数生效。
_KNOBS: dict[str, Any] = {
    "adv_eps": 1e-6, "eps": 1e-6, "min_group_size": 2, "normalize": True,
    "temperature": 1.0, "entropy_coef": 0.0, "clip_ratio": 0.2, "strict": False,
}


def _bind(fn: Any, pool: Mapping[str, Any], ctx: str) -> tuple[list[Any], dict[str, Any]]:
    """按参数名(经别名表)从 pool 取值绑定;必填参数取不到即 ContractError。"""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:                          # pragma: no cover
        raise ContractError(f"{ctx}: 无法读取签名({exc})") from exc
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for name, p in sig.parameters.items():
        if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        val = pool.get(_ALIAS.get(name, name), _MISS)
        if val is _MISS:
            if p.default is not inspect.Parameter.empty:
                continue
            if name in _KNOBS:
                val = _KNOBS[name]
            else:
                missing.append(name)
                continue
        if p.kind is p.POSITIONAL_ONLY:
            args.append(val)
        else:
            kwargs[name] = val
    if missing:
        raise ContractError(
            f"{ctx}: 必填参数 {missing} 无法绑定 —— 签名 {sig};测试能提供的量为 "
            f"{sorted(pool)}(别名见 tests/test_agentic_grpo.py::_ALIAS)。"
            "请改用契约里的参数名,或给这些参数默认值。")
    return args, kwargs


def _call(fn: Any, pool: Mapping[str, Any], ctx: str) -> Any:
    args, kwargs = _bind(fn, pool, ctx)
    return fn(*args, **kwargs)


def _construct(cls: Any, pool: Mapping[str, Any], ctx: str) -> Any:
    args, kwargs = _bind(cls.__init__, pool, ctx)
    return cls(*args, **kwargs)


def _attr(obj: Any, names: Sequence[str], ctx: str) -> Any:
    """取第一个存在的属性/方法;都没有就报出全部候选名与实际公开成员。"""
    for n in names:
        got = getattr(obj, n, None)
        if got is not None:
            return got
    raise ContractError(
        f"{ctx}: 未找到下列任一成员 {list(names)};实际暴露 "
        f"{sorted(n for n in dir(obj) if not n.startswith('_'))[:40]}")


def _to_float_list(x: Any) -> list[float]:
    if torch is not None and torch.is_tensor(x):
        return [float(v) for v in x.detach().reshape(-1).tolist()]
    if np is not None and isinstance(x, np.ndarray):
        return [float(v) for v in x.reshape(-1).tolist()]
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    if isinstance(x, (int, float)):
        return [float(x)]
    raise ContractError(f"无法把 {type(x).__name__} 解释成一维实数序列:{x!r}")


def _obj_fields(obj: Any) -> dict[str, Any]:
    """对象的「字段视图」:dataclass 字段 + 实例 __dict__ 里的**额外**属性。

    # note (luojiaxuan): 必须把 __dict__ 里的临时属性也算进来 —— dataclass 的
    # ``fields()`` 与 ``repr()`` 都看不见 ``object.__setattr__(st, "required_steps", …)``
    # 这种事后挂上去的东西,只查声明字段会漏掉最容易发生的泄露方式。
    """
    out: dict[str, Any] = {}
    if isinstance(obj, Mapping):
        return {str(k): v for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out.update({f.name: getattr(obj, f.name, None) for f in dataclasses.fields(obj)})
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        out.update({k: v for k, v in d.items() if not k.startswith("__")})
    return out


def _around(text: str, needle: str, span: int = 60) -> str:
    i = text.find(needle)
    return text[max(0, i - span): i + len(needle) + span] if i >= 0 else ""


# ------------------------------------------------------------------ fixture 构造

_TMP: tempfile.TemporaryDirectory | None = None
_FIX: list[dict[str, Any]] | None = None


def tearDownModule() -> None:                                       # noqa: N802
    global _TMP
    if _TMP is not None:
        _TMP.cleanup()
        _TMP = None


def _tmpdir() -> str:
    global _TMP
    if _TMP is None:
        _TMP = tempfile.TemporaryDirectory(prefix="agentic_grpo_")
    return _TMP.name


def _pick_tasks(n_tasks: int, n_frames: int) -> list[Any]:
    """挑 expert_actions 足够长的任务实例(要够 n_frames 帧才凑得出多元候选集)。"""
    tasks = _MOD["tasks"]
    out: list[Any] = []
    regimes = ("one_old_frame", "two_frame_complementary", "distractor_heavy")
    tids = [t.template_id for t in tasks.TEMPLATES]
    for i, tid in enumerate(tids * 3):
        spec = tasks.generate_task(template_id=tid, seed=3300 + 7 * i,
                                   regime=regimes[i % len(regimes)])
        if len(spec.expert_actions) >= n_frames:
            out.append(spec)
        if len(out) >= n_tasks:
            break
    if len(out) < n_tasks:
        raise ContractError(
            f"tasks.py 没有 {n_tasks} 个 expert_actions ≥ {n_frames} 的模板实例,"
            f"凑不出 Phase 3 需要的候选集(实得 {len(out)} 个)")
    return out


def _render_frames(spec: Any, n_frames: int, out_dir: str) -> list[str]:
    """按专家动作前缀真渲染 n_frames 张 1920×1080 截图(每步**决策前**观测)。"""
    os.makedirs(out_dir, exist_ok=True)
    state = copy.deepcopy(spec.initial_state)
    paths: list[str] = []
    for i in range(n_frames):
        path = os.path.join(out_dir, f"f{i:02d}.png")
        _MOD["render"].render_to_file(state, path)
        paths.append(path)
        if getattr(state, "terminated", False):
            break
        if i < len(spec.expert_actions):
            state = _MOD["executor"].apply_action(state, dict(spec.expert_actions[i]))
    while len(paths) < n_frames:                                    # 提前 terminate 时补齐
        paths.append(paths[-1])
    return paths


def _make_provider() -> Any:
    """`DummyFeatureExtractor`(+ `FeatureCache`)组成 (step, path) → 特征的 provider。

    一律走**默认维度**:features.DummyFeatureExtractor.dim 与 selector_model 的
    d_vis 默认值本来就对齐,手工指定反而会把两边错开。
    """
    feats = _MOD["features"]
    dummy = _construct(_attr(feats, ("DummyFeatureExtractor",), "features 模块"),
                       {"seed": 0}, "features.DummyFeatureExtractor(...)")
    cache_cls = getattr(feats, "FeatureCache", None)
    if cache_cls is None:
        return dummy
    return _construct(cache_cls, {"extractor": dummy}, "features.FeatureCache(...)")


def _attach(bank: Any, provider: Any) -> None:
    fn = getattr(_MOD["features"], "attach_to_bank", None)
    if fn is None:
        bank.feature_provider = provider
        return
    _call(fn, {"bank": bank, "extractor": provider}, "features.attach_to_bank(...)")


def _shape_of(x: Any) -> tuple[int, ...]:
    if torch is not None and torch.is_tensor(x):
        return tuple(int(v) for v in x.shape)
    if np is not None and isinstance(x, np.ndarray):
        return tuple(int(v) for v in x.shape)
    if isinstance(x, (list, tuple)):
        return (len(x),)
    raise ContractError(f"帧特征类型 {type(x).__name__} 无法取形状:{x!r}")


def _fixtures() -> list[dict[str, Any]]:
    """模块级 fixture:真任务 + 真截图 + HistoryFrameBank + Dummy 冻结特征。

    每个 fixture 渲 ``FIXTURE_FRAMES`` 帧(0..6),取 ``current_step = 6``,故候选事件号
    = [1..5](帧 0 恒被折叠成文本、当前帧恒在末轮,见 policy_io 编号口径),动作空间
    C(5,2)=10 —— 足够区分「均匀」「modest 先验」「塌缩」三种初始化。
    """
    global _FIX
    if _FIX is not None:
        return _FIX
    pio, env = _MOD["policy_io"], _MOD["env"]
    provider = _make_provider()
    out: list[dict[str, Any]] = []
    for i, spec in enumerate(_pick_tasks(FIXTURE_TASKS, FIXTURE_FRAMES)):
        paths = _render_frames(spec, FIXTURE_FRAMES,
                               os.path.join(_tmpdir(), f"task{i:02d}"))
        bank = pio.HistoryFrameBank()
        bank.extend(paths)
        _attach(bank, provider)
        now = FIXTURE_FRAMES - 1
        acts = [dict(a) for a in spec.expert_actions[:now]]
        out.append({
            "task": spec, "bank": bank, "paths": paths, "provider": provider,
            "current_step": now, "history_actions": acts,
            "history_action_lines": [env.action_line(a) for a in acts],
            "candidates": tuple(int(c) for c in bank.candidates(now)),
            "feature_dim": _shape_of(bank.feature(1))[-1],
        })
    _FIX = out
    return _FIX


def _make_selector(seed: int = 0, *, recency_prior: bool = True) -> Any:
    """构造 `selector_model.SubsetSelectorPolicy`(带 SelectorConfig 时先造 cfg)。

    ``recency_prior=False`` 用于让 argmax 检查**可证伪**:带 recency 先验时最大 logit
    恒是 recent-B,「argmax 恒返回 recent-B」这种退化实现就查不出来。
    """
    sm = _MOD["selector_model"]
    cls = _attr(sm, ("SubsetSelectorPolicy",), "causalcache_agentic.selector_model")
    torch.manual_seed(seed)
    pool: dict[str, Any] = {"b": BUDGET_B, "device": "cpu", "seed": seed}
    cfg_cls = getattr(sm, "SelectorConfig", None)
    if cfg_cls is not None:
        # note (luojiaxuan): 只覆盖 budget(与可选的 recency 先验开关);维度一律用
        # 默认值 —— 与 features 的 DummyFeatureExtractor 默认维度天然对齐,手工
        # 指定反而错开。
        cfg_pool: dict[str, Any] = {"b": BUDGET_B}
        if not recency_prior:
            cfg_pool["use_recency_prior"] = False
        pool["cfg"] = _construct(cfg_cls, cfg_pool, "selector_model.SelectorConfig(...)")
    sel = _construct(cls, pool, "selector_model.SubsetSelectorPolicy(...)")
    if isinstance(sel, torch.nn.Module):
        # note (luojiaxuan): **必须 eval()** —— selector 默认带 dropout=0.1,train 模式下
        # 同一 state 两次前向就不是同一个分布,logits/logprob_of/sample/recent_prob 之间
        # 的一致性检查会被 dropout 噪声(实测 |Δlogit| 达 0.07)全部污染。eval 不影响
        # 反传,梯度方向性检查照常成立。
        sel.eval()
    return sel


def _make_state_builder() -> Any:
    sm = _MOD["selector_model"]
    cls = _attr(sm, ("SelectorStateBuilder",), "causalcache_agentic.selector_model")
    pool: dict[str, Any] = {"b": BUDGET_B, "device": "cpu", "seed": 0}
    emb_cls = getattr(sm, "DummyTextEmbedder", None)
    if emb_cls is not None:
        pool["embedder"] = _construct(emb_cls, {"seed": 0, "device": "cpu"},
                                      "selector_model.DummyTextEmbedder(...)")
    return _construct(cls, pool, "selector_model.SelectorStateBuilder(...)")


def _state_of(fix: Mapping[str, Any], sel: Any, *, current_step: int | None = None,
              candidates: Sequence[int] | None = None) -> Any:
    """用 `SelectorStateBuilder` 造 selector 的 state_repr(帧特征取自 bank 缓存)。"""
    builder = _make_state_builder()
    build = _attr(builder, ("build", "build_state", "__call__", "from_bank"),
                  "SelectorStateBuilder 实例")
    bank = fix["bank"]
    now = int(fix["current_step"] if current_step is None else current_step)
    cands = tuple(int(c) for c in (bank.candidates(now) if candidates is None
                                   else candidates))
    feats = {j: bank.feature(j) for j in cands}
    pool: dict[str, Any] = {
        "policy": sel, "task": fix["task"], "instruction": fix["task"].instruction,
        "bank": bank, "extractor": fix["provider"], "b": BUDGET_B, "device": "cpu",
        "current_step": now, "candidates": cands,
        "history_actions": fix["history_actions"][:now],
        "history_action_lines": fix["history_action_lines"][:now],
        "frame_features": feats, "current_feature": bank.feature(now),
        "recent": _recent(cands),
    }
    try:
        return _call(build, pool, "SelectorStateBuilder.build(...)")
    except ContractError:
        raise
    except Exception:                                               # noqa: BLE001
        # note (luojiaxuan): 容忍「frame_features 要的是与 candidates 对齐的序列」
        # 这种口径差异;仍失败就把原始异常抛出去(那是真的接口不兼容)。
        pool["frame_features"] = [feats[j] for j in cands]
        return _call(build, pool, "SelectorStateBuilder.build(frame_features=list)")


# ------------------------------------------------------------------ 分布工具

def _recent(candidates: Sequence[int], b: int = BUDGET_B) -> tuple[int, ...]:
    """recent-B(升序);候选不足 B 时自然退化,候选为空时返回空元组。

    # note (luojiaxuan): 先排序再取尾部 —— recent-B 的定义是「步号最大的 B 帧」,
    # 与候选**传入顺序**无关。置换不变性测试正是靠这条才不会把自己的记账错误
    # (给 selector 传了个错的 recent)误判成 selector 在用数组位置。
    """
    cands = sorted(int(c) for c in candidates)
    k = min(int(b), len(cands))
    return tuple(cands[-k:]) if k > 0 else ()


def _subsets(candidates: Sequence[int], b: int = BUDGET_B) -> list[tuple[int, ...]]:
    """全部 C(n,b) 个合法 subset;**每个 subset 内部升序**,与候选传入顺序无关。

    # note (luojiaxuan): 必须先排序再组合 —— 否则给一个逆序候选表时
    # combinations 会产出 (5,4) 这种降序元组,置换不变性测试会误判成接口不符。
    """
    cands = sorted(int(c) for c in candidates)
    return [tuple(s) for s in itertools.combinations(cands, min(int(b), len(cands)))]


def _action_space(sel: Any, candidates: Sequence[int],
                  b: int = BUDGET_B) -> list[tuple[int, ...]]:
    """优先用 selector 自己的 `action_space`,并校验它就是 C(n,B) 的全枚举。"""
    mine = _subsets(candidates, b)
    fn = getattr(sel, "action_space", None)
    if fn is None:
        return mine
    theirs = [tuple(int(j) for j in s)
              for s in _call(fn, {"candidates": [int(c) for c in candidates], "b": b},
                             "SubsetSelectorPolicy.action_space(...)")]
    if sorted(theirs) != sorted(mine):
        raise ContractError(
            f"action_space({list(candidates)}, {b}) = {theirs};契约要求恰好是 "
            f"C({len(candidates)},{b}) 的全部子集 {mine}(路线 §5)")
    for s in theirs:
        if tuple(sorted(s)) != s:
            raise ContractError(f"action_space 返回未升序的 subset {s}")
    return theirs


def _logits(sel: Any, state: Any, subsets: Sequence[tuple[int, ...]],
            candidates: Sequence[int], b: int = BUDGET_B) -> Any:
    fn = _attr(sel, ("score_subsets",), "SubsetSelectorPolicy 实例")
    out = _call(fn, {"state": state, "subsets": [tuple(int(j) for j in s) for s in subsets],
                     "candidates": [int(c) for c in candidates], "b": b},
                "SubsetSelectorPolicy.score_subsets(...)")
    if torch.is_tensor(out):
        t = out
    elif np is not None and isinstance(out, np.ndarray):
        t = torch.as_tensor(out)
    elif isinstance(out, (list, tuple)):
        t = torch.as_tensor([float(v) for v in out])
    else:
        raise ContractError(
            f"score_subsets 返回 {type(out).__name__};契约要求与入参同序的一维 logits")
    t = t.reshape(-1)
    if t.numel() != len(subsets):
        raise ContractError(
            f"score_subsets 返回 {t.numel()} 个分数,入参有 {len(subsets)} 个 subset "
            "—— 契约要求「顺序与入参一致」的逐 subset 打分")
    return t.float()


def _probs(sel: Any, state: Any, candidates: Sequence[int],
           b: int = BUDGET_B) -> dict[tuple[int, ...], float]:
    subs = _action_space(sel, candidates, b)
    p = torch.softmax(_logits(sel, state, subs, candidates, b).detach().double(), dim=-1)
    return {s: float(p[i]) for i, s in enumerate(subs)}


def _logps(sel: Any, state: Any, subsets: Sequence[tuple[int, ...]],
           candidates: Sequence[int], b: int = BUDGET_B) -> dict[tuple[int, ...], Any]:
    """一次前向拿到若干 subset 的**可反传** logπ(S|state)。

    统一从 score_subsets 的 log_softmax 取(不依赖实现方 logprob_of 是否保留计算图;
    两者数值一致已由 test_selector_distribution 钉死)。
    """
    space = _action_space(sel, candidates, b)
    logits = _logits(sel, state, space, candidates, b)
    if not logits.requires_grad:
        raise ContractError(
            "score_subsets 返回的 logits 不可反传(requires_grad=False)—— "
            "Phase 3 要训 selector,subset 打分必须在计算图里")
    logp = torch.log_softmax(logits, dim=-1)
    idx = {s: i for i, s in enumerate(space)}
    return {tuple(int(j) for j in s): logp[idx[tuple(int(j) for j in s)]] for s in subsets}


def _logprob_of(sel: Any, state: Any, subset: Sequence[int],
                candidates: Sequence[int], b: int = BUDGET_B) -> float:
    fn = _attr(sel, ("logprob_of", "log_prob_of", "logprob", "log_prob", "subset_logprob"),
               "SubsetSelectorPolicy 实例")
    out = _call(fn, {"state": state, "subset": tuple(int(j) for j in subset),
                     "candidates": [int(c) for c in candidates], "b": b},
                "SubsetSelectorPolicy.logprob_of(...)")
    if isinstance(out, tuple):                                      # 容忍 (logp, extra)
        out = out[0]
    return float(out.detach() if torch.is_tensor(out) else out)


# ------------------------------------------------------------------ 源码扫描工具

def _source_views(path: str) -> tuple[str, list[tuple[int, str]]]:
    """返回 (code_text, prose_literals)。

    * ``code_text``:去掉注释与 docstring 的源码;字符串字面量若形如标识符
      (``"required_steps"`` 这类 dict key)则**保留内容**,否则清空 —— 这样既能抓到
      ``rec["step_reward"]``,又不会被中文说明文字误伤;
    * ``prose_literals``:被清空的那些说明性字符串((行号, 原文)),交给
      「带否定词豁免」的单独扫描。
    """
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    lines = src.splitlines()
    grid = [list(ln) for ln in lines]
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:                                      # pragma: no cover
        raise ContractError(f"{path} 语法错误,无法扫描:{exc}") from exc
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        for child in body if isinstance(body, list) else ():
            if (isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant)
                    and isinstance(child.value.value, str)):
                doc_lines.update(range(child.lineno, (child.end_lineno or child.lineno) + 1))

    def blank(sr: int, sc: int, er: int, ec: int) -> None:
        for r in range(sr, min(er, len(grid)) + 1):
            row = grid[r - 1]
            a = sc if r == sr else 0
            bnd = ec if r == er else len(row)
            for i in range(a, min(bnd, len(row))):
                row[i] = " "

    ident = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    prose: list[tuple[int, str]] = []
    str_types = {tokenize.STRING}
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        if hasattr(tokenize, name):
            str_types.add(getattr(tokenize, name))
    try:
        with open(path, "rb") as fb:
            for tok in tokenize.tokenize(fb.readline):
                if tok.type == tokenize.COMMENT:
                    blank(tok.start[0], tok.start[1], tok.end[0], tok.end[1])
                elif tok.type in str_types and tok.start[0] not in doc_lines:
                    try:
                        val = ast.literal_eval(tok.string)
                    except Exception:                               # noqa: BLE001
                        val = tok.string
                    if not (isinstance(val, str) and ident.match(val)):
                        prose.append((tok.start[0], tok.string))
                        blank(tok.start[0], tok.start[1], tok.end[0], tok.end[1])
    except tokenize.TokenError:                                     # pragma: no cover
        pass
    text = "\n".join("" if (i + 1) in doc_lines else "".join(row)
                     for i, row in enumerate(grid))
    return text, prose


# note (luojiaxuan): 路线 §9 硬禁项 —— 在被测脚本的**代码**里出现即违规。
_BANNED_TOKENS = (
    "milestone", "step_reward", "stepwise_reward", "per_step_reward",
    "process_reward", "reward_shaping", "shaped_reward", "dense_reward",
    "step_correct", "action_correct", "gold_subset", "gold_frame", "gold_pair",
    "pass1_draft", "pass_1_draft", "gumbel", "straight_through",
)
# 与 reward/advantage 同行出现即视为「拿诊断字段当 reward」
_DIAG_TOKENS = ("required_steps", "memory_probe", "frame_vars", "decision_step",
                "distractor_steps", "milestone", "probe")
_REWARD_RE = re.compile(r"\breward|\badvantage|\breturns?\b", re.I)
# 说明性文字里出现禁词,只要带否定/引用语境就豁免(自证注释不该被误伤)
_NEGATION_RE = re.compile(r"不得|禁止|不引入|不用|不再|无|非|not |never|no |avoid|"
                          r"without|forbid|§ ?9|禁", re.I)


# ------------------------------------------------------------------ 测试:分布契约

@requires("contract", "policy_io", "tasks", "render", "executor", "env", "features",
          "selector_model", "torch", "numpy")
class SelectorDistributionTest(unittest.TestCase):
    """路线 §5:动作空间 = 全部 C(N,2) subset;`z_θ(S|state)` 对整个 subset 打分;
    训练采样、测试 argmax。本用例把「logits / logprob_of / sample / argmax」四者钉成
    同一个分布 —— 三者不自洽的实现会让 GRPO 记的 logπ 与实际采样分布对不上。"""

    def test_selector_distribution(self) -> None:
        fix = _fixtures()[0]
        sel = _make_selector(seed=0)
        state = _state_of(fix, sel)
        cands = fix["candidates"]
        self.assertGreaterEqual(len(cands), 4, f"fixture 候选太少:{cands}")
        subs = _action_space(sel, cands)
        self.assertEqual(math.comb(len(cands), BUDGET_B), len(subs),
                         f"动作空间大小 {len(subs)} ≠ C({len(cands)},{BUDGET_B})")

        # (1) C(n,2) 上概率和为 1(用 logprob_of 自己给的数,不是我算的 softmax)
        logps = [_logprob_of(sel, state, s, cands) for s in subs]
        for s, lp in zip(subs, logps):
            self.assertTrue(math.isfinite(lp), f"logprob_of({s}) = {lp} 不是有限数")
            self.assertLessEqual(lp, 1e-6, f"logprob_of({s}) = {lp} > 0,不是 log 概率")
        total = sum(math.exp(lp) for lp in logps)
        self.assertAlmostEqual(
            1.0, total, delta=PROB_TOL,
            msg=f"π 在 C({len(cands)},{BUDGET_B})={len(subs)} 个 subset 上求和 = "
                f"{total:.6f} ≠ 1 —— selector 的动作空间必须恰好是全部 B-子集(路线 §5)")

        # (2) logits 与 logprob_of 必须是同一个分布
        probs = _probs(sel, state, cands)
        for s, lp in zip(subs, logps):
            self.assertAlmostEqual(
                probs[s], math.exp(lp), delta=PROB_TOL,
                msg=f"subset {s}: softmax(score_subsets)={probs[s]:.6f} 与 "
                    f"exp(logprob_of)={math.exp(lp):.6f} 不一致 —— 两条路径必须同分布")

        # (3) sample 的 logprob 与 logprob_of 一致(GRPO 的 logπ 记账靠这条)
        sample = _attr(sel, ("sample",), "SubsetSelectorPolicy 实例")
        torch.manual_seed(1234)
        for i in range(8):
            got = _call(sample, {"state": state, "candidates": list(cands), "b": BUDGET_B},
                        "SubsetSelectorPolicy.sample(...)")
            self.assertIsInstance(got, tuple, f"第 {i} 次 sample 返回 "
                                              f"{type(got).__name__},契约要求 "
                                              "(subset, logprob) 二元组")
            self.assertEqual(2, len(got), f"第 {i} 次 sample 返回 {len(got)} 元组")
            sub = tuple(int(j) for j in got[0])
            lp = float(got[1].detach() if torch.is_tensor(got[1]) else got[1])
            self.assertEqual(tuple(sorted(sub)), sub, f"sample 返回的 subset 未升序:{sub}")
            self.assertEqual(BUDGET_B, len(sub), f"sample 返回 {len(sub)} 帧,契约要求 B=2")
            self.assertIn(sub, subs, f"sample 返回候选集外的 subset:{sub},候选={cands}")
            self.assertAlmostEqual(
                _logprob_of(sel, state, sub, cands), lp, delta=LOGP_TOL,
                msg=f"第 {i} 次 sample 的 logprob={lp:.6f} 与 logprob_of({sub}) 不一致")

        # (4) argmax = 最大 logit 对应的 subset
        # note (luojiaxuan): 带 recency 先验时最大 logit 恒是 recent-B,只测默认模型
        # 的话「argmax 恒返回 recent-B」这种退化实现查不出来。所以再关掉先验测几个
        # 种子 —— 那时 argmax 应当随内容走,至少有一次不等于 recent-B。
        non_recent = 0
        cases = [("默认(带 recency 先验)", sel, state)]
        probe = _make_selector(seed=0, recency_prior=False)
        if getattr(getattr(probe, "cfg", None), "use_recency_prior", None) is False:
            for seed in PRIOR_SEEDS:
                p = _make_selector(seed=seed, recency_prior=False)
                cases.append((f"关闭 recency 先验(seed={seed})", p, _state_of(fix, p)))
        for label, s_i, st_i in cases:
            logits = _logits(s_i, st_i, subs, cands).detach()
            best = subs[int(torch.argmax(logits))]
            raw = _call(_attr(s_i, ("argmax",), "SubsetSelectorPolicy 实例"),
                        {"state": st_i, "candidates": list(cands), "b": BUDGET_B},
                        "SubsetSelectorPolicy.argmax(...)")
            arg = tuple(int(j) for j in (raw[0] if isinstance(raw, tuple) and raw
                                         and not isinstance(raw[0], int) else raw))
            self.assertEqual(best, arg,
                             f"{label}: argmax 返回 {arg},最大 logit 的 subset 是 {best}"
                             f"(logits={[round(float(v), 4) for v in logits]})")
            non_recent += int(arg != _recent(cands))
        if len(cases) > 1:
            self.assertGreater(
                non_recent, 0,
                f"{len(cases)} 组观测里 argmax 全都恰好等于 recent-B {_recent(cands)}"
                "(含关闭 recency 先验的),这条检查退化成恒真 —— argmax 必须真的读 logits")


@requires("contract", "policy_io", "tasks", "render", "executor", "env", "features",
          "selector_model", "torch", "numpy")
class PermutationTest(unittest.TestCase):
    """路线 §5「subset 联合表示」的硬约束:π(S) 只能由**帧内容 + frame age/step index**
    决定,不得由候选在数组里的**位置**决定。位置一旦可用,selector 就能靠「永远选最后
    两个槽位」白拿 recent-2 的分数而不学任何记忆策略,论文的因果对照当场失效。"""

    def test_permutation_equivariance(self) -> None:
        fix = _fixtures()[0]
        sel = _make_selector(seed=0)
        state = _state_of(fix, sel)
        cands = list(fix["candidates"])
        n = len(cands)
        subs = _action_space(sel, cands)
        base = _probs(sel, state, cands)

        # (a) batch 内 subset 换位置不得改变逐 subset 的分数(禁止跨 subset 位置耦合)
        order = list(range(len(subs)))[::-1]
        s_logits = _logits(sel, state, [subs[i] for i in order], cands).detach()
        d_logits = _logits(sel, state, subs, cands).detach()
        for pos, i in enumerate(order):
            self.assertAlmostEqual(
                float(d_logits[i]), float(s_logits[pos]), delta=1e-4,
                msg=f"subset {subs[i]} 在 batch 里换个位置分数就变了 "
                    f"({float(d_logits[i]):.6f} → {float(s_logits[pos]):.6f})")

        # (b) 候选顺序置换:同一物理 subset 的概率必须不变
        perm = [n - 1 - i for i in range(n)]                        # 逆序,最强扰动
        shuffled = [cands[i] for i in perm]
        variants: list[tuple[str, Any, list[int]]] = []
        errors: list[str] = []
        # note (luojiaxuan): 结构性置换按**优先级取第一个可用的**,不是全都测。
        # builder 级是走公开 API 的诚实做法(ages/recency_rank/index 都由实现自己
        # 重新推导);state 级是反射硬改字段的兜底,只在 builder 拒收乱序候选时才用
        # —— 反射改出来的 state 可能违反实现内部「位置==时间序」的不变量,拿它判
        # 失败会冤枉人。
        for label, make in (
                ("builder 级置换(candidates 逆序重建 state)",
                 lambda: _state_of(fix, sel, candidates=shuffled)),
                ("state 级置换(候选与其特征同步逆序)",
                 lambda: _permute_state(state, perm, cands))):
            try:
                st = make()
            except Exception as exc:                                # noqa: BLE001
                errors.append(f"{label} -> {type(exc).__name__}: {exc}")
                continue
            if st is not None:
                variants.append((label, st, shuffled))
                break
        try:
            _logits(sel, state, subs, shuffled)
        except Exception as exc:                                    # noqa: BLE001
            errors.append(f"调用级置换 -> {type(exc).__name__}: {exc}")
        else:
            variants.append(("调用级置换(candidates 逆序传入)", state, shuffled))
        if not variants:
            self.fail("无法构造置换后的输入,三条路都不通:" + " | ".join(errors) +
                      " —— 契约要求至少一条可行,否则无法证明 selector 没有靠数组位置作弊")
        for label, st, cs in variants:
            probs = _probs(sel, st, cs)
            for s in subs:
                self.assertIn(s, probs, f"{label}: 置换后丢了 subset {s}")
                self.assertAlmostEqual(
                    base[s], probs[s], delta=PROB_TOL,
                    msg=f"{label}: 物理 subset {s} 的概率从 {base[s]:.6f} 变成 "
                        f"{probs[s]:.6f}(容差 {PROB_TOL})—— selector 在用数组位置")


def _permute_state(state: Any, perm: Sequence[int], cands: Sequence[int]) -> Any | None:
    """反射地把 state 里所有「与候选对齐」的结构一起置换;做不到返回 None。

    对齐判据:序列/张量第 0 维长度 == 候选数;dict 的 key 集 == 候选集(位置索引表)。
    必须至少命中一处「就是候选本身」的结构,否则说明这个 state 没暴露候选顺序,
    置换无意义。
    """
    n = len(cands)
    ints = tuple(int(c) for c in cands)
    hit = [False]

    def conv(val: Any, depth: int = 0) -> Any:
        if torch is not None and torch.is_tensor(val):
            return val[list(perm)] if val.dim() >= 1 and val.shape[0] == n else None
        if np is not None and isinstance(val, np.ndarray):
            return val[list(perm)] if val.ndim >= 1 and val.shape[0] == n else None
        if isinstance(val, Mapping):
            keys = list(val)
            if (len(keys) == n and all(isinstance(k, (int, np.integer)) for k in keys)
                    and tuple(sorted(int(k) for k in keys)) == tuple(sorted(ints))):
                hit[0] = True
                vals = [val[k] for k in ints]                       # 按候选序取
                if all(isinstance(v, (int, np.integer)) for v in vals) and \
                        sorted(int(v) for v in vals) == list(range(n)):
                    return {ints[perm[i]]: i for i in range(n)}     # 位置索引表
                return {ints[perm[i]]: vals[perm[i]] for i in range(n)}
            if depth >= 2:
                return None
            sub = {k: conv(v, depth + 1) for k, v in val.items()}
            if any(v is not None for v in sub.values()):
                return {k: (sub[k] if sub[k] is not None else v) for k, v in val.items()}
            return None
        if isinstance(val, (list, tuple)) and len(val) == n:
            if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                   for v in val) and tuple(int(v) for v in val) == ints:
                hit[0] = True
            seq = [val[i] for i in perm]
            return tuple(seq) if isinstance(val, tuple) else seq
        return None

    fields = _obj_fields(state)
    if not fields:
        return None
    new = {k: v for k, v in ((k, conv(v)) for k, v in fields.items()) if v is not None}
    if not hit[0] or not new:
        return None
    try:
        return dataclasses.replace(state, **new)                    # type: ignore[type-var]
    except Exception:                                               # noqa: BLE001
        clone = copy.copy(state)
        for k, v in new.items():
            try:
                object.__setattr__(clone, k, v)
            except Exception:                                       # noqa: BLE001
                return None
        return clone


@requires("contract", "policy_io", "tasks", "render", "executor", "env", "features",
          "selector_model", "torch", "numpy")
class RecentPriorTest(unittest.TestCase):
    """路线 §5:recent-2 是 C(N,2) 里的**普通合法动作**,初始化只能是 modest 先验
    (原文:「modest recent-B 初始化(**不是**极强 KEEP bias)」),同时不能塌到 0。
    前者过强会让 selector 学不动(离线战役 §0.14 的 always-KEEP 退化就是这么来的),
    后者会让 RL 一开始就丢掉 recent-sufficient 那一类任务的分。

    门槛读法:主观测点是 C(5,2)=10 的候选规模,π(recent-2) 的**均值**必须落在
    [0.15, 0.65],单点放宽到 [0.10, 0.75](随机初始化噪声)。若实现提供了 recent 先验
    的强度旋钮(`SelectorConfig.recent_strength/recent_tau`、`init_recent_bias(strength)`),
    这条就是在钉它的**默认值**。另在 C(3,2)=3 的小候选规模上只查两件事:
    π(recent) ≥ 均匀(先验方向对)、≤ 0.85(没塌成点质量)。
    """

    def test_recent_is_ordinary_action(self) -> None:
        fixes = _fixtures()
        observed: dict[str, list[float]] = {}
        for seed in PRIOR_SEEDS:
            for i, fix in enumerate(fixes):
                sel = _make_selector(seed=seed)
                state = _state_of(fix, sel)
                cands = fix["candidates"]
                probs = _probs(sel, state, cands)
                rec = _recent(cands)
                self.assertIn(rec, probs, f"recent-B {rec} 不在动作空间里 —— "
                                          "路线要求 recent-2 始终是合法动作")
                ent = -sum(q * math.log(max(q, 1e-12)) for q in probs.values())
                floor = ENTROPY_FRAC_MIN * math.log(len(probs))
                self.assertGreater(
                    ent, floor,
                    f"seed={seed} fixture={i}: 初始策略熵 {ent:.4f} ≤ {floor:.4f} "
                    f"(= {ENTROPY_FRAC_MIN}×log C)—— 初始化即塌缩,GRPO 探索不出东西")
                # note (luojiaxuan): 两个观测点都要查:①**刚构造出来**的模型(rollout /
                # train 默认就用它);②若实现把 modest 先验做成显式初始化步骤,再按它
                # 的**默认强度**施加一次 —— 后者正是在钉 recent_strength/tau 的默认值。
                stages = [("构造后", probs)]
                init = getattr(sel, "init_recent_bias", None)
                if callable(init):
                    _call(init, {}, "SubsetSelectorPolicy.init_recent_bias(...)")
                    stages.append(("init_recent_bias() 默认强度后", _probs(sel, state, cands)))
                lo, hi = RECENT_PRIOR_LOOSE
                for stage, dist in stages:
                    p = dist[rec]
                    observed.setdefault(stage, []).append(p)
                    self.assertTrue(
                        lo <= p <= hi,
                        f"seed={seed} fixture={i} {stage}: 初始 π(recent-2)={p:.4f} 越出"
                        f"单点带 [{lo}, {hi}](候选 {cands},C={len(dist)})")
                own = getattr(sel, "recent_prob", None)
                if callable(own):
                    mine = float(_call(own, {"state": state},
                                       "SubsetSelectorPolicy.recent_prob(...)"))
                    self.assertAlmostEqual(
                        stages[-1][1][rec], mine, delta=1e-3,
                        msg=f"selector 自报 recent_prob={mine:.4f},与 softmax 口径 "
                            f"{stages[-1][1][rec]:.4f} 不一致 —— 监控量与真实分布必须同源")
        # note (luojiaxuan): **逐 stage** 判均值,不把两个阶段混在一起平均 —— 混算会让
        # 「构造期正常、init_recent_bias 默认强度过头」这种错配被互相抵消掉。
        for stage, values in observed.items():
            mean = sum(values) / len(values)
            self.assertTrue(
                RECENT_PRIOR_LO <= mean <= RECENT_PRIOR_HI,
                f"{stage}:π(recent-2) 均值 {mean:.4f} 越出 modest 先验带 "
                f"[{RECENT_PRIOR_LO}, {RECENT_PRIOR_HI}](逐点值 "
                f"{[round(p, 3) for p in values]});旋钮见 SelectorConfig."
                "recent_strength / recent_tau 或 init_recent_bias(strength, tau)")

        # 小候选规模:只查方向与非点质量
        fix = fixes[0]
        sel = _make_selector(seed=0)
        cands = tuple(int(c) for c in fix["bank"].candidates(SMALL_STEP))
        if len(cands) >= BUDGET_B + 1:
            state = _state_of(fix, sel, current_step=SMALL_STEP)
            probs = _probs(sel, state, cands)
            p = probs[_recent(cands)]
            uniform = 1.0 / len(probs)
            self.assertGreaterEqual(
                p, uniform - PROB_TOL,
                f"候选 {cands}(C={len(probs)}): π(recent-2)={p:.4f} < 均匀 "
                f"{uniform:.4f} —— modest **recent** 先验的方向反了")
            self.assertLessEqual(
                p, RECENT_POINTMASS_MAX,
                f"候选 {cands}(C={len(probs)}): π(recent-2)={p:.4f} > "
                f"{RECENT_POINTMASS_MAX},小候选集上塌成点质量")


@requires("contract", "policy_io", "tasks", "render", "executor", "env", "features",
          "selector_model", "torch", "numpy")
class DiagnosticLeakTest(unittest.TestCase):
    """路线 §2/§5 + contract.py:`memory_probe`(required_steps / regime / decision_step /
    frame_vars / variables.answer)**只准做诊断**,不得进 policy 输入、不得作 RL reward。
    selector 的 state 就是「进模型的东西」,一旦携带这些字段,Phase 3 的结论会退化成
    「用 gold 监督选帧」,与 §9 禁止清单直接冲突。"""

    FORBIDDEN = ("required_steps", "memory_probe", "regime", "answer", "frame_vars",
                 "decision_step", "decision_steps", "distractor_steps", "distractor",
                 "recent2", "gold", "oracle", "solvable", "probe", "assertions",
                 "expert_actions")

    def test_no_diagnostic_leak(self) -> None:
        fix = _fixtures()[0]
        task = fix["task"]
        self.assertTrue(task.memory_probe,
                        f"fixture 任务 {task.task_id} 没有 memory_probe,本用例失去意义")
        self.assertIn("required_steps", task.memory_probe)
        sel = _make_selector(seed=0)
        state = _state_of(fix, sel)

        # (1) 对象图:不得挂着 TaskSpec,也不得有诊断字段名
        for path, node in _walk(state):
            name = path.rsplit(".", 1)[-1].split("[")[0].lower()
            for bad in self.FORBIDDEN:
                self.assertNotIn(
                    bad, name,
                    f"selector state 里出现诊断字段 {path!r}(命中禁词 {bad!r})—— "
                    "memory_probe 类信息禁止进 policy/selector 输入")
            self.assertNotIsInstance(
                node, _MOD["contract"].TaskSpec,
                f"selector state 的 {path!r} 直接挂着 TaskSpec —— 它带着 memory_probe/"
                "regime/assertions/expert_actions,等于把答案塞进模型输入")

        # (2) 序列化字符串:regime 标签与诊断 key 都不得出现
        # note (luojiaxuan): tasks.py 的 task_id 形如 "<template>::<regime>::<seed>",
        # 天然含 regime 字样;task_id 作为记账元数据可以留,故先剔除再扫描。
        text = repr(state)[:200_000].replace(task.task_id, "<task_id>")
        low = text.lower()
        for bad in ("required_steps", "memory_probe", "frame_vars", "decision_step",
                    "distractor_steps"):
            self.assertNotIn(bad, low,
                             f"repr(state) 里出现诊断字段 {bad!r}:…{_around(low, bad)}…")
        self.assertNotIn(
            task.regime.lower(), low,
            f"repr(state) 里出现 regime 标签 {task.regime!r} —— regime 是诊断分层标签,"
            f"进了输入就等于告诉模型该不该捞旧帧:…{_around(low, task.regime.lower())}…")


def _walk(root: Any, max_nodes: int = 4000, max_depth: int = 6) -> list[tuple[str, Any]]:
    """广度优先遍历对象图,返回 (路径, 对象);张量/数组/标量当叶子。"""
    out: list[tuple[str, Any]] = []
    seen: set[int] = set()
    queue: list[tuple[str, Any, int]] = [("state", root, 0)]
    while queue and len(out) < max_nodes:
        path, node, depth = queue.pop(0)
        if id(node) in seen:
            continue
        seen.add(id(node))
        out.append((path, node))
        if depth >= max_depth or isinstance(node, (str, bytes, int, float, bool,
                                                   type(None))):
            continue
        if torch is not None and torch.is_tensor(node):
            continue
        if np is not None and isinstance(node, np.ndarray):
            continue
        if isinstance(node, Mapping):
            for k, v in list(node.items())[:200]:
                queue.append((f"{path}[{k}]", v, depth + 1))
            continue
        if isinstance(node, (list, tuple, set)):
            for i, v in enumerate(list(node)[:200]):
                queue.append((f"{path}[{i}]", v, depth + 1))
            continue
        for k, v in _obj_fields(node).items():
            queue.append((f"{path}.{k}", v, depth + 1))
    return out


# ------------------------------------------------------------------ 测试:GRPO

_ADV_NAMES = ("advantages", "group_advantages", "compute_advantages", "grpo_advantages",
              "compute_group_advantages", "group_advantage", "normalize_advantages",
              "rewards_to_advantages", "make_advantages")
_STAT_NAMES = ("group_report", "group_stats", "group_reward_stats", "reward_variance",
               "group_reward_variance", "group_variance", "has_reward_variance",
               "group_has_variance", "is_degenerate_group", "degenerate_group",
               "zero_variance", "is_zero_variance")
# note (luojiaxuan): 「零方差组被计数」的可见形态很多:统计字段名(zero_variance_groups /
# groups_with_reward_variance)、跳过原因串(all0 / all1 / degenerate)。这里一并认。
_VAR_KEY_RE = re.compile(r"zero[_-]?var|no[_-]?variance|degener|has[_-]?variance|"
                         r"reward[_-]?var|var(iance)?[_-]?group|group[_-]?var|"
                         r"all[_-]?(zero|one|0|1)\b", re.I)


class _RewardArray(np.ndarray if np is not None else object):       # type: ignore[misc]
    """既能当序列用(`group.rewards` 是 property)又能被调用(`group.rewards()`)。"""

    def __call__(self) -> "_RewardArray":
        return self


class _FakeRollout:
    """duck-typed rollout(实现方没暴露 RolloutRecord 时的兜底)。"""

    def __init__(self, index: int, reward: float) -> None:
        self.task_id, self.group_id, self.rollout_id = "t0", "g0", str(index)
        self.arm, self.reward, self.n_steps = "learned", float(reward), 1


class _FakeGroup:
    def __init__(self, rewards: Sequence[float]) -> None:
        self.task_id, self.group_id = "t0", "g0"
        self.rollouts = [_FakeRollout(i, r) for i, r in enumerate(rewards)]

    @property
    def rewards(self) -> Any:
        return np.asarray([r.reward for r in self.rollouts],
                          dtype=np.float64).view(_RewardArray)


def _make_group(rewards: Sequence[float]) -> Any:
    """优先用训练脚本自己的 Group/RolloutRecord 构造(口径一致),否则用 duck-type。"""
    mod = _SCRIPT.get("grpo")
    g_cls = getattr(mod, "Group", None)
    r_cls = getattr(mod, "RolloutRecord", None)
    s_cls = getattr(mod, "StepSample", None)
    if g_cls is not None and r_cls is not None:
        try:
            rolls = []
            for i, rw in enumerate(rewards):
                steps = [_construct(s_cls, {"current_step": 3, "candidates": (1, 2),
                                            "subset": (1, 2), "logp_old": 0.0,
                                            "b": BUDGET_B},
                                    "train_selector_grpo.StepSample(...)")] \
                    if s_cls is not None else []
                rolls.append(_construct(r_cls, {
                    "task_id": "t0", "group_id": "g0", "rollout_id": str(i),
                    "arm": "learned", "rewards": float(rw), "instruction": "x",
                    "frames": {}, "history_actions": [], "history_lines": [],
                    "steps": steps, "n_steps": len(steps), "source_file": "",
                }, "train_selector_grpo.RolloutRecord(...)"))
            return _construct(g_cls, {"task_id": "t0", "group_id": "g0",
                                      "rollouts": rolls},
                              "train_selector_grpo.Group(...)")
        except Exception:                                           # noqa: BLE001
            pass
    return _FakeGroup(rewards)


def _adv_callable() -> Any:
    mod = _SCRIPT["grpo"]
    for name in _ADV_NAMES:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    for cls_name in ("GRPOTrainer", "SelectorGRPOTrainer", "Trainer"):
        cls = getattr(mod, cls_name, None)
        for name in _ADV_NAMES:
            fn = getattr(cls, name, None) if cls is not None else None
            if callable(fn):
                return fn
    raise ContractError(
        f"{_SCRIPT_PATH['grpo']} 里找不到组内优势函数;接受的名字:{list(_ADV_NAMES)}"
        "(路线 §5「GRPO 用同组相对 reward」)")


def _advantages(rewards: Sequence[float]) -> tuple[list[float] | None, Any]:
    """调用被测的组内优势函数,返回 (优势列表或 None, 附带信息)。

    兼容两种入参口径:``f(rewards)`` 与 ``f(group)``(后者是 rollout JSONL 分组后的
    Group 对象);也兼容两种返回口径:直接给优势,或给 ``(优势 | None, reason)`` ——
    ``None`` 表示「这组被整体跳过」(零方差组的合法处理方式,等价于零梯度)。
    """
    fn = _adv_callable()
    pools: list[dict[str, Any]] = [
        {"rewards": np.asarray(list(rewards), dtype=np.float64)},
        {"group": _make_group(rewards)},
        {"rewards": [float(r) for r in rewards]},
        {"rewards": torch.tensor([float(r) for r in rewards])},
    ]
    last: Exception | None = None
    for pool in pools:
        try:
            out = _call(fn, pool, "train_selector_grpo 组内优势函数")
        except Exception as exc:                                    # noqa: BLE001
            last = exc
            continue
        info: Any = None
        if isinstance(out, tuple) and len(out) == 2 and (
                out[1] is None or isinstance(out[1], (Mapping, str))):
            out, info = out[0], out[1]
        return (None if out is None else _to_float_list(out)), info
    raise ContractError(
        f"组内优势函数对 rewards(ndarray/list/tensor)与 group 四种入参都失败:{last}")


def _says_zero_variance(info: Any) -> bool:
    """附带信息(reason 字符串或 stats dict)是否点明了「这组没有 reward 方差」。"""
    if info is None:
        return False
    if isinstance(info, Mapping):
        return any(_VAR_KEY_RE.search(str(k)) for k in info)
    return bool(_VAR_KEY_RE.search(str(info)))


def _params(sel: Any) -> list[Any]:
    fn = getattr(sel, "parameters", None)
    return [p for p in fn()] if callable(fn) else []


def _zero_variance_counted(rewards: Sequence[float]) -> tuple[bool, str]:
    """零方差组是否被某种可见机制计数(接受三类实现,见失败信息)。"""
    mod = _SCRIPT["grpo"]
    _adv, info = _advantages(rewards)
    if _says_zero_variance(info):
        return True, f"优势函数返回 info={info!r}"
    for name in _STAT_NAMES:
        fn = getattr(mod, name, None)
        if not callable(fn):
            continue
        for pool in ({"groups": [_make_group(rewards)]},
                     {"group": _make_group(rewards)},
                     {"rewards": np.asarray(list(rewards), dtype=np.float64)}):
            try:
                out = _call(fn, pool, f"train_selector_grpo.{name}")
            except Exception:                                       # noqa: BLE001
                continue
            if isinstance(out, Mapping):
                if any(_VAR_KEY_RE.search(str(k)) for k in out):
                    return True, f"{name}() 返回统计 dict:{sorted(out)[:8]}"
                break
            low = name.lower()
            if isinstance(out, bool):
                if (out is False) if ("has" in low) else (out is True):
                    return True, f"{name}() = {out}"
                break
            try:
                val = float(out)
            except (TypeError, ValueError):
                break
            if "var" in low and abs(val) <= ZERO_ADV_TOL:
                return True, f"{name}() = {val}"
            break
    code, _prose = _source_views(_SCRIPT_PATH["grpo"])
    m = _VAR_KEY_RE.search(code)
    if m:
        return True, f"训练脚本代码里的统计量 {m.group(0)!r}"
    return False, ""


@requires("script:grpo", "torch", "numpy", "contract", "policy_io", "tasks", "render",
          "executor", "env", "features", "selector_model")
class GRPOZeroVarianceTest(unittest.TestCase):
    """路线 §5:GRPO 用同组相对 reward。全同 reward 的组内没有任何相对信息,优势必须
    恒 0(⇒ 不产生梯度),且必须**被计数** —— §5 验收明确要报「有 reward variance 的
    group 比例」,不统计就看不见「大量 group 全 0/全 1」这个该调 curriculum 的信号
    (而不是该加 milestone reward 的信号,§9)。"""

    def test_grpo_zero_variance_group(self) -> None:
        numeric: list[float] | None = None
        for rewards in ([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0]):
            adv, info = _advantages(rewards)
            if adv is None:
                # 实现选择「整组直接跳过」:这本身就是零梯度,但必须说明原因
                self.assertTrue(
                    _says_zero_variance(info),
                    f"rewards={rewards}(全同)的优势返回 None,但附带信息 {info!r} 没有"
                    "点明是零方差 —— 跳过的组必须能被统计出来(路线 §5 要报有 reward "
                    "variance 的 group 比例)")
                continue
            self.assertEqual(len(rewards), len(adv),
                             f"rewards={rewards} 得到 {len(adv)} 个优势")
            for i, a in enumerate(adv):
                self.assertFalse(math.isnan(a) or math.isinf(a),
                                 f"rewards={rewards} 的优势 {adv} 出现 NaN/Inf —— "
                                 "零方差组多半除了个没有 eps 的 std")
                self.assertLessEqual(
                    abs(a), ZERO_ADV_TOL,
                    f"rewards={rewards}(全同)第 {i} 条的优势 {a} ≠ 0 —— "
                    "同组相对 reward 在零方差组必须给 0")
            numeric = adv

        # 零优势 ⇒ 零梯度(拿真 selector 走一遍 PG loss);整组跳过时无梯度可查
        if numeric is not None:
            fix = _fixtures()[0]
            sel = _make_selector(seed=0)
            state = _state_of(fix, sel)
            cands = fix["candidates"]
            params = [p for p in _params(sel) if p.requires_grad]
            self.assertTrue(params, "SubsetSelectorPolicy 没有可训练参数(应为 nn.Module)")
            subs = _action_space(sel, cands)
            chosen = [subs[i] for i in (0, 1, 2, -1)][:len(numeric)]
            logps = _logps(sel, state, chosen, cands)
            loss = -(torch.stack([logps[s] for s in chosen])
                     * torch.tensor(numeric[:len(chosen)], dtype=torch.float32)).mean()
            for p in params:
                p.grad = None
            loss.backward()
            worst = max((float(p.grad.abs().max()) for p in params if p.grad is not None),
                        default=0.0)
            self.assertLessEqual(
                worst, ZERO_ADV_TOL,
                f"全同 reward 组仍产生梯度(最大 |grad| = {worst:.3e})—— "
                "零优势必须 ⇒ 零梯度,否则塌缩风险直接来自记账错误")

        counted, how = _zero_variance_counted([1.0, 1.0, 1.0, 1.0])
        self.assertTrue(
            counted,
            "零方差组没有被任何可见机制计数。接受下列任一:(a) 组内优势函数额外返回带 "
            f"variance/degenerate 关键字的 info dict;(b) 模块级统计函数 {list(_STAT_NAMES)};"
            f"(c) 训练脚本代码里出现 /{_VAR_KEY_RE.pattern}/ 的统计量。"
            "路线 §5 验收要求报『有 reward variance 的 group 比例』")
        self.assertTrue(how)


@requires("script:grpo", "torch", "numpy", "contract", "policy_io", "tasks", "render",
          "executor", "env", "features", "selector_model")
class GRPOAdvantageSignTest(unittest.TestCase):
    """路线 §5:同 task instance 同初始 state 产 G 条、GRPO 用同组相对 reward。
    混合组里成功轨迹的优势必须严格大于失败轨迹;一次(大 lr)更新后,成功轨迹所选
    subset 的 logπ 必须上升 —— 这是「terminal-only reward 真的能把梯度送进 selector」
    的端到端方向性证明(dummy 特征 + 固定种子 + 单组)。"""

    def test_grpo_advantage_sign(self) -> None:
        rewards = [1.0, 1.0, 0.0, 0.0]
        adv, info = _advantages(rewards)
        self.assertIsNotNone(
            adv, f"混合组 rewards={rewards} 拿不到优势(返回 None,info={info!r})—— "
                 "有 reward variance 的组必须产生可用的组内相对优势")
        assert adv is not None                                      # for type checkers
        self.assertEqual(4, len(adv))
        for i in (0, 1):
            for j in (2, 3):
                self.assertGreater(
                    adv[i], adv[j],
                    f"成功轨迹的优势 {adv[i]} 未大于失败轨迹 {adv[j]}"
                    f"(rewards={rewards}, advantages={adv})")
        self.assertGreater(adv[0], 0.0, f"混合组里成功轨迹的优势应为正,实得 {adv}")
        self.assertLess(adv[3], 0.0, f"混合组里失败轨迹的优势应为负,实得 {adv}")

        torch.manual_seed(20260813)
        fix = _fixtures()[0]
        sel = _make_selector(seed=0)
        state = _state_of(fix, sel)
        cands = fix["candidates"]
        params = [p for p in _params(sel) if p.requires_grad]
        self.assertTrue(params, "SubsetSelectorPolicy 没有可训练参数(应为 nn.Module)")
        subs = _action_space(sel, cands)
        win, lose_a, lose_b = subs[0], subs[1], subs[2]
        chosen = [win, win, lose_a, lose_b]
        watch = (win, lose_a, lose_b)
        logps = _logps(sel, state, tuple(chosen) + watch, cands)
        before = {s: float(logps[s].detach()) for s in watch}

        # note (luojiaxuan): 用 **SGD** 而不是 Adam。方向性断言的依据是一阶展开
        # (Δloss<0 ⇒ Σ adv·Δlogπ>0),只有沿真实梯度走才成立;Adam 的首步近似
        # lr·sign(g) 完全丢掉梯度大小,实测 lr=0.1 单步会把「成功 vs 失败」的差距
        # 推反(dgap=-0.57),那是优化器伪影,不是 selector 学不动。
        opt = torch.optim.SGD(params, lr=GRPO_LR)
        loss = -(torch.stack([logps[s] for s in chosen])
                 * torch.tensor(adv, dtype=torch.float32)).mean()
        opt.zero_grad()
        loss.backward()
        gnorm = math.sqrt(sum(float((p.grad ** 2).sum()) for p in params
                              if p.grad is not None))
        self.assertGreater(gnorm, 0.0, "混合组更新的梯度范数为 0 —— logπ 没接进计算图")
        opt.step()

        # note (luojiaxuan): 更新后必须**重建** state —— 旧 state 里的张量是用旧参数
        # 算出来的,拿它评估会漏掉表示层的变化,也可能触发二次反传。
        after_state = _state_of(fix, sel)
        after_logps = _logps(sel, after_state, watch, cands)
        after = {s: float(after_logps[s].detach()) for s in watch}
        self.assertGreater(
            after[win], before[win],
            f"一次更新(lr={GRPO_LR})后,成功 rollout 所选 subset {win} 的 logπ 从 "
            f"{before[win]:.5f} 变成 {after[win]:.5f},没有上升(advantages={adv})")
        # note (luojiaxuan): 失败臂的**绝对** logπ 未必下降 —— 三个 subset 共享参数,
        # 抬高 win 会顺带抬高相邻 subset。真正该成立的是**相对差距拉开**:
        # logπ(win) − mean logπ(losers) 必须变大,否则负优势没起作用。
        gap0 = before[win] - (before[lose_a] + before[lose_b]) / 2
        gap1 = after[win] - (after[lose_a] + after[lose_b]) / 2
        self.assertGreater(
            gap1, gap0,
            f"一次更新后,logπ({win}) 与失败臂 {lose_a}/{lose_b} 的平均差距从 "
            f"{gap0:+.5f} 变成 {gap1:+.5f},没有拉开 —— 负优势没有把概率推下去"
            f"(advantages={adv})")
        obj0 = sum(a * before[s] for a, s in zip(adv, chosen))
        obj1 = sum(a * after[s] for a, s in zip(adv, chosen))
        self.assertGreater(
            obj1, obj0,
            f"一次更新后 GRPO 目标 Σ adv·logπ 从 {obj0:+.5f} 掉到 {obj1:+.5f} —— "
            "沿梯度走一步却让目标变差,说明 logπ 与优势的记账错位")


# ------------------------------------------------------------------ 测试:rollout schema

_LINE_FIELDS = {
    "task_id": ("task_id", "task", "instance_id"),
    "group_id": ("group_id", "group", "group_index", "group_key"),
    "rollout_id": ("rollout_id", "rollout", "rollout_index", "sample_id",
                   "sample_index", "member", "g_index", "traj_id"),
    "reward": ("reward", "success", "terminal_reward", "task_success", "ret"),
}
_STEP_FIELDS = {
    "step": ("step", "step_index", "t"),
    "candidates": ("candidates", "candidate_events", "cands", "candidate_steps"),
    "chosen_subset": ("chosen_subset", "subset", "shown_subset", "selected_subset",
                      "chosen"),
    "selector_logprob": ("selector_logprob", "selector_log_prob", "logprob", "log_prob"),
}
_ARM_FIELDS = ("arm", "selector_arm", "policy_arm", "arm_name")


def _get(rec: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for a in aliases:
        if a in rec:
            return rec[a]
    return _MISS


def _step_views(rec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """一行 JSONL 归一成「若干 step 视图」:既接受一行一步,也接受一行一条轨迹。"""
    steps = rec.get("steps")
    if isinstance(steps, list) and steps and all(isinstance(s, Mapping) for s in steps):
        merged = []
        for s in steps:
            view = {k: v for k, v in rec.items() if k != "steps"}
            view.update(s)
            merged.append(view)
        return merged
    return [rec]


@unittest.skipIf(not os.path.exists(_SCRIPT_PATH["rollout"]),
                 f"scripts/rollout_selector.py 尚未落地({_SCRIPT_PATH['rollout']})")
class RolloutSchemaTest(unittest.TestCase):
    """路线 §2「RL 数据:selector sampled subset 与 log-prob、terminal success、metadata」
    + §5「同 task instance 同初始 state 产 G 条」。JSONL 是 rollout 与 GRPO trainer 之间
    唯一的数据面:字段缺一项 trainer 就得靠猜;同组 candidates 不一致则说明组内不是同一
    task instance / 同一套帧记账,组内相对 reward 立刻失去意义。"""

    def test_rollout_schema(self) -> None:
        help_text = self._help()
        flags = {f.replace("_", "-")
                 for f in re.findall(r"--([a-zA-Z0-9][a-zA-Z0-9_-]*)", help_text)}
        self.assertIn("dry-run", flags,
                      f"rollout_selector.py 必须支持 --dry-run;--help 里的开关:"
                      f"{sorted(flags)}")
        arms = self._arm_choices(help_text)
        recent_arm = next((a for a in arms if "recent" in a.lower()), None)
        stoch_arm = next((a for a in arms if "random" in a.lower()), None) or \
            next((a for a in arms if "learn" in a.lower()), None)

        records, used_arm = self._dry_run(flags, recent_arm)
        self._check_records(records, used_arm)
        groups = self._grouped(records)
        if not any(len(v) >= 2 for v in groups.values()) and stoch_arm:
            # note (luojiaxuan): recent 是确定性臂,实现可以合理地把同组去重成 1 条;
            # 「同组 G 条」的检查改用随机臂跑第二遍。
            records2, used2 = self._dry_run(flags, stoch_arm)
            self._check_records(records2, used2)
            groups = self._grouped(records2)
        sized = {k: v for k, v in groups.items() if len(v) >= 2}
        self.assertTrue(
            sized,
            f"没有任何 (task_id, group_id) 出现 ≥2 条 rollout(共 {len(groups)} 组)—— "
            "GRPO 需要同 task instance 的一组轨迹才能算组内相对 reward")
        for key, recs in sized.items():
            by_step: dict[Any, list[tuple[int, list[int]]]] = {}
            for i, rec in enumerate(recs):
                for view in _step_views(rec):
                    by_step.setdefault(_get(view, _STEP_FIELDS["step"]), []).append(
                        (i, [int(c) for c in _get(view, _STEP_FIELDS["candidates"])]))
            for step, items in by_step.items():
                first = items[0][1]
                for i, cands in items[1:]:
                    self.assertEqual(
                        first, cands,
                        f"group {key} 的第 {step} 步:rollout#{items[0][0]} 候选 {first} "
                        f"与 rollout#{i} 候选 {cands} 不一致 —— 同组必须是同一 task "
                        "instance、同一初始 state、同一套帧记账")

    # ---------------------------------------------------------- 检查
    def _check_records(self, records: Sequence[Mapping[str, Any]],
                       used_arm: str | None) -> None:
        self.assertGreaterEqual(len(records), 2,
                                f"--dry-run 只产出 {len(records)} 行 JSONL,验不了 schema")
        recent_steps = 0
        for idx, rec in enumerate(records):
            ctx = f"第 {idx} 行"
            for canon, aliases in _LINE_FIELDS.items():
                self.assertIsNot(_get(rec, aliases), _MISS,
                                 f"{ctx}缺字段 {canon}(接受别名 {list(aliases)});"
                                 f"实际键 = {sorted(rec)}")
            reward = _get(rec, _LINE_FIELDS["reward"])
            if reward is not None:
                self.assertIn(float(reward), (0.0, 1.0),
                              f"{ctx}的 terminal reward = {reward!r};路线 §0 规定主奖励"
                              "只有最终任务成功 R∈{0,1}")
            arm = _get(rec, _ARM_FIELDS)
            arm_txt = arm if isinstance(arm, str) else (used_arm or "")
            is_recent = "recent" in str(arm_txt).lower()
            for k, view in enumerate(_step_views(rec)):
                vctx = f"{ctx} step-view {k}"
                for canon, aliases in _STEP_FIELDS.items():
                    self.assertIsNot(_get(view, aliases), _MISS,
                                     f"{vctx}缺字段 {canon}(接受别名 {list(aliases)});"
                                     f"实际键 = {sorted(view)}")
                cands = [int(c) for c in _get(view, _STEP_FIELDS["candidates"])]
                sub = tuple(int(c) for c in _get(view, _STEP_FIELDS["chosen_subset"]))
                self.assertEqual(tuple(sorted(cands)), tuple(cands),
                                 f"{vctx}: candidates 未升序 {cands}")
                self.assertEqual(tuple(sorted(set(sub))), sub,
                                 f"{vctx}: chosen_subset 未严格递增 {sub}")
                self.assertTrue(set(sub) <= set(cands),
                                f"{vctx}: chosen_subset {sub} ⊄ candidates {cands}")
                self.assertEqual(min(BUDGET_B, len(cands)), len(sub),
                                 f"{vctx}: chosen_subset 基数 {len(sub)} ≠ "
                                 f"min(B={BUDGET_B}, |candidates|={len(cands)})")
                lp = _get(view, _STEP_FIELDS["selector_logprob"])
                if lp is not None and lp is not _MISS:
                    self.assertLessEqual(float(lp), 1e-6,
                                         f"{vctx}: selector_logprob={lp} > 0,不是 log 概率")
                if is_recent:
                    recent_steps += 1
                    self.assertEqual(
                        _recent(cands), sub,
                        f"{vctx}: recent 臂选了 {sub},recent-B 应为 {_recent(cands)}"
                        f"(candidates={cands})")
        if used_arm and "recent" in used_arm.lower():
            self.assertGreater(
                recent_steps, 0,
                f"以 --…arm {used_arm} 跑出来的记录里,没有任何一步能被认定为 recent 臂"
                f"(JSONL 也没有 {list(_ARM_FIELDS)} 字段)。recent-2 是路线 §5 必须存在"
                "的对照臂")

    @staticmethod
    def _grouped(records: Sequence[Mapping[str, Any]]
                 ) -> dict[tuple[Any, Any], list[Mapping[str, Any]]]:
        out: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
        for rec in records:
            key = (_get(rec, _LINE_FIELDS["task_id"]), _get(rec, _LINE_FIELDS["group_id"]))
            out.setdefault(key, []).append(rec)
        return out

    # ---------------------------------------------------------- 子进程封装
    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([_RL_CODE, _CODE, env.get("PYTHONPATH", "")])
        return env

    def _run(self, argv: Sequence[str], timeout: float) -> Any:
        cmd = [sys.executable, _SCRIPT_PATH["rollout"], *argv]
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                  cwd=_REPO, env=self._env(), check=False)
        except subprocess.TimeoutExpired as exc:                    # pragma: no cover
            self.fail(f"`{' '.join(cmd)}` 超过 {timeout}s 未结束 —— --dry-run 必须是轻量"
                      f"自检(不加载 GUI-Owl)。stdout 尾部:{str(exc.stdout)[-2000:]}")

    def _help(self) -> str:
        proc = self._run(["--help"], timeout=90.0)
        self.assertEqual(0, proc.returncode,
                         f"`rollout_selector.py --help` 退出码 {proc.returncode}\n"
                         f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
        return proc.stdout + proc.stderr

    @staticmethod
    def _arm_choices(help_text: str) -> list[str]:
        """从 argparse 的 choices 块里取臂名(找不到就退化成 recent/random)。"""
        out: list[str] = []
        for block in re.findall(r"\{([^{}]+)\}", help_text):
            toks = [t.strip() for t in block.split(",")]
            if any("recent" in t.lower() for t in toks):
                out.extend(toks)
        return out or ["recent", "random"]

    def _dry_run(self, flags: set[str],
                 arm: str | None) -> tuple[list[Mapping[str, Any]], str | None]:
        with tempfile.TemporaryDirectory(prefix="rollout_dry_") as tmp:
            out_path = os.path.join(tmp, "rollout.jsonl")
            argv = ["--dry-run"]
            used_arm: str | None = None

            def pick(cands: Sequence[str], value: str) -> bool:
                for c in cands:
                    if c in flags:
                        argv.extend([f"--{c}", value])
                        return True
                return False

            pick(("out", "output", "out-path", "out-file", "jsonl", "out-jsonl", "save"),
                 out_path)
            pick(("n-tasks", "num-tasks", "tasks", "limit-tasks", "limit"), DRY_RUN_TASKS)
            pick(("group-size", "group", "n-rollouts", "rollouts"), DRY_RUN_GROUP)
            pick(("shot-dir",), os.path.join(tmp, "shots"))
            if arm and pick(("selector-arm", "arm", "policy-arm"), arm):
                used_arm = arm
            proc = self._run(argv, timeout=SUBPROC_TIMEOUT)
            self.assertEqual(
                0, proc.returncode,
                f"`rollout_selector.py {' '.join(argv)}` 退出码 {proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout[-4000:]}\n"
                f"--- stderr ---\n{proc.stderr[-4000:]}")
            return self._records(tmp, out_path, proc.stdout), used_arm

    def _records(self, tmp: str, out_path: str,
                 stdout: str) -> list[Mapping[str, Any]]:
        paths = [out_path] if os.path.exists(out_path) else []
        if not paths:
            for root, _dirs, files in os.walk(tmp):
                paths.extend(os.path.join(root, f) for f in files
                             if f.endswith((".jsonl", ".json")))
        recs: list[Any] = []
        for path in sorted(paths):
            with open(path, "r", encoding="utf-8") as fh:
                for ln, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        self.fail(f"{path}:{ln} 不是合法 JSON 行:{exc}\n{line[:400]}")
        if not recs:                                                # 退化:JSONL 打到 stdout
            for line in stdout.splitlines():
                if line.strip().startswith("{"):
                    try:
                        recs.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        self.assertTrue(recs, "--dry-run 既没写出 JSONL 文件,stdout 里也没有 JSON 行 —— "
                              "路线 §2 要求 rollout 落 RL 数据(subset / log-prob / "
                              "terminal success / metadata)")
        return [r for r in recs if isinstance(r, Mapping)]


# ------------------------------------------------------------------ 测试:闭环口径

@requires("torch", "numpy", "selector_model", "features", "tasks", "env",
          "script:rollout", "script:grpo")
class LearnedArmClosedLoopTest(unittest.TestCase):
    """**learned 臂**的跨模块口径闸门:rollout 记的 log π_old 必须被 GRPO 逐位重算出来。

    为什么单独立一条:`RolloutSchemaTest` 只跑基线臂(recent/random)—— 它们不碰
    帧特征、也不用参数化 selector 的 state,于是 rollout↔trainer 之间真正的接缝
    (帧特征实现、state builder、候选编号口径、温度、动作空间枚举顺序)一条都没被
    走到。集成时这里确实是坏的:rollout 侧自带一份"路径哈希 → 一维向量"的假特征,
    trainer 侧用 `features.DummyFeatureExtractor` 的 `[T, D]`,两者既形状不同、
    数值也是两组无关随机数。

    本用例断言的是**唯一的硬证据**:同一份 checkpoint 下,trainer 重建 state 后
    重算的 log π_θ 与 JSONL 里的 log π_old 必须近似相等(容差 `LOGP_TOL`)。
    对不上就说明特征、候选顺序、历史动作切片、步号或温度至少错了一个 —— 而这类
    错误在训练曲线上只表现为"不收敛",不点名就查不出来。

    路线条款:§5「同 task instance 同初始 state 产 G 条;selector 采样;GRPO 用同组
    相对 reward」的前提是 π_old 与 π_θ 是同一个分布族;§2「JSONL 是两者之间唯一的
    数据面」。
    """

    def test_learned_arm_logprob_roundtrip(self) -> None:
        import pathlib

        grpo = _load_script("grpo")

        missing = [n for n in ("load_rollouts", "group_rollouts", "parse_args",
                               "_resolve_feature_cache", "_resolve_selector",
                               "_resolve_state_builder", "check_logprob_consistency")
                   if not hasattr(grpo, n)]
        self.assertFalse(missing, f"train_selector_grpo.py 缺少入口 {missing};"
                                  "本用例需要它们来复现训练端的 state 重建路径")

        with tempfile.TemporaryDirectory(prefix="grpo_loop_") as tmp:
            out = os.path.join(tmp, "rollout.jsonl")
            argv = ["--dry-run", "--selector-arm", "learned",
                    "--n-tasks", DRY_RUN_TASKS, "--group-size", DRY_RUN_GROUP,
                    "--skip-render", "--rollout-seed", "7",
                    "--shot-dir", os.path.join(tmp, "shots"), "--out", out]
            proc = self._run(argv)
            self.assertEqual(0, proc.returncode,
                             f"`rollout_selector.py {' '.join(argv)}` 退出码 "
                             f"{proc.returncode}\n--- stdout ---\n{proc.stdout[-4000:]}"
                             f"\n--- stderr ---\n{proc.stderr[-4000:]}")
            # 随机初始化的 selector 必须被落盘 —— 否则这批 log π_old 是孤儿,
            # 训练端只能拿另一组随机权重去算 ratio(而且不会报错)。
            init = out + ".init.pt"
            self.assertTrue(
                os.path.exists(init),
                "rollout 用随机初始化的 selector 采了样却没存权重(期望 "
                f"{init});GRPO 必须从产出该批 log π_old 的同一份权重起训")

            records, counts = grpo.load_rollouts([pathlib.Path(out)], budget=BUDGET_B)
            self.assertTrue(records, f"trainer 从 {out} 没读出任何 learned 臂 rollout")
            self.assertGreater(
                counts.get("steps_used", 0), 0,
                f"没有任何可训练的步(计数:{counts});C(n,2)>1 的步才有梯度")

            args = grpo.parse_args([
                "--rollouts", out, "--features", "dummy", "--device", "cpu",
                "--selector-ckpt", init, "--out", os.path.join(tmp, "trained.pt")])
            provider = grpo._resolve_feature_cache(args)
            selector = grpo._resolve_selector(args, init)[0]
            # note (luojiaxuan): 这里**照抄训练端 train() 里的那一行**,而不是无脑
            # eval() —— 这样 dropout 的开关也被本用例守住:rollout 是 eval 模式采的
            # 样,若哪天训练端把 train 模式设成默认,重算的 log π 会带上 dropout
            # 噪声,ratio 在第一次更新前就 ≠1,这条断言会红。
            selector.train(bool(args.train_dropout))
            builder, sb_origin = grpo._resolve_state_builder(args, selector, provider)
            groups = grpo.group_rollouts(records)

            seen = sorted({r.state_builder_origin for r in records
                           if r.state_builder_origin})
            self.assertEqual(
                [sb_origin], seen,
                f"state builder 口径不一致:训练端 {sb_origin!r},rollout 端 {seen} "
                "—— 两侧 state 不同就不是同一个 π,重要性比全错")

            report = grpo.check_logprob_consistency(
                selector, builder, provider, groups, limit=16, tol=LOGP_TOL,
                strict=False, args=args)
            self.assertFalse(report.get("skipped"),
                             f"log π 一致性检查被跳过了:{report}")
            self.assertTrue(
                report.get("passed"),
                f"trainer 重算的 log π_θ 与 JSONL 的 log π_old 对不上:{report} —— "
                "帧特征 / 候选顺序 / 历史动作切片 / 步号口径 / 温度 至少错了一个")

    @staticmethod
    def _run(argv: Sequence[str]) -> Any:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [_RL_CODE, _CODE, env.get("PYTHONPATH", "")])
        return subprocess.run([sys.executable, _SCRIPT_PATH["rollout"], *argv],
                              capture_output=True, text=True, cwd=_REPO, env=env,
                              timeout=SUBPROC_TIMEOUT, check=False)


# ------------------------------------------------------------------ 测试:源码禁令

class TerminalRewardOnlyTest(unittest.TestCase):
    """路线 §0「训练主奖励**只有**最终任务成功 R∈{0,1}」+ §9 禁止清单。

    这是一道**回归闸门**:今天两个脚本是干净的,不代表半年后加特性时还干净。扫描去掉
    注释与 docstring 的代码文本(所以「本脚本不引入 milestone reward」这类自证说明不会
    误伤;说明性字符串另按「带否定词豁免」单独判),命中即失败。"""

    TARGETS = ("rollout", "grpo")

    def test_terminal_reward_only(self) -> None:
        scanned = 0
        for key in self.TARGETS:
            path = _SCRIPT_PATH[key]
            if not os.path.exists(path):
                continue
            scanned += 1
            code, prose = _source_views(path)
            low = code.lower()
            name = os.path.basename(path)
            for bad in _BANNED_TOKENS:
                self.assertNotIn(
                    bad, low,
                    f"{name} 的代码里出现 §9 禁止项 {bad!r}:…{_around(low, bad)}… —— "
                    "路线 §9 禁止 milestone reward / step-level 正确性 / gold 帧监督 / "
                    "pass-1 draft / Gumbel-softmax 软选帧")
            for lineno, text in prose:
                if _NEGATION_RE.search(text):
                    continue
                for bad in _BANNED_TOKENS:
                    self.assertNotIn(
                        bad, text.lower(),
                        f"{name}:{lineno} 的字符串字面量里出现 §9 禁止项 {bad!r} 且不带"
                        f"否定语境:{text[:160]}")
            for lineno, line in enumerate(code.splitlines(), 1):
                if not _REWARD_RE.search(line):
                    continue
                hit = [d for d in _DIAG_TOKENS if d in line.lower()]
                self.assertFalse(
                    hit,
                    f"{name}:{lineno} 把诊断字段 {hit} 与 reward/advantage 写在同一行:"
                    f"{line.strip()[:160]!r} —— memory_probe 类标注只能做诊断,不得作 "
                    "selector 的 RL reward(路线 §2)")
        # 额外守一道 selector 侧的 §9(软选帧 / gold 监督),模块缺席则跳过
        sm_path = os.path.join(_RL_CODE, "causalcache_agentic", "selector_model.py")
        if os.path.exists(sm_path):
            low = _source_views(sm_path)[0].lower()
            for bad in ("gumbel", "straight_through", "gold_subset", "gold_frame",
                        "milestone"):
                self.assertNotIn(bad, low, f"selector_model.py 出现 §9 禁止项 {bad!r}:"
                                           f"…{_around(low, bad)}…")
        if not scanned:
            self.skipTest("两个脚本都尚未落地:" + ", ".join(
                _SCRIPT_PATH[k] for k in self.TARGETS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
