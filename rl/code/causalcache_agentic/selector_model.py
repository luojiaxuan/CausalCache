"""Phase 3 参数化 subset selector:``z_θ(S | state), |S| = B`` 的 GRPO 策略。

# note (luojiaxuan): 路线见 rl/docs/agentic_memory_rl_roadmap_20260813.md §4
# (文档标题写作 "Phase 3:Selector-only GRPO")。本文件是该阶段的主角:固定
# memory-aware executor,只训 selector;selector 每步从**全部**历史帧里选 B 张。
#
# 三条来自路线的硬约束在此落实:
#   1. **动作空间 = 全部 C(n, B)**,recent-B 只是其中一个**普通合法动作** ——
#      没有 KEEP 特殊动作、没有人工 move cost、没有"是否换子集"的外层门。
#      (这与老离线线 DirectDCST 的 G(KEEP)=0 外部规则**明确不同**,老规则属于
#      台账 §0.14 判死的 selector 线,不得带进来。)
#   2. **非加性打分**:唯一的标量出口是 subset 级 ``SubsetScorer.head_z``,
#      整个子集一次进 attention 出一个 logit,**不是**逐帧分相加。唯一的加性项
#      是第 3 条的 recency 先验,它是可学的小偏置,训练可以把它压平或翻转。
#   3. **modest recent-B 初始化**(防塌缩,不是极强 KEEP bias):在 z 上加一个
#      "越靠近 recent 越高"的**可学**先验,默认强度让初始 π(recent-B) ≈ 0.3-0.5
#      —— 既保证起点不比 recent-B 基线差太多,又留足探索概率给旧帧。
#
# 结构件全部复用老线已验证的 ``causalcache_rl.direct_dcst``(路线 §1 明列的
# 可复用资产):``FrameResampler``(每帧 [T,d_vis] → K 个 latent)、
# ``ContextEncoder``(instruction/动作行冻结词嵌入 → 上下文 token)、
# ``CandidateSetEncoder``(帧间交互)、``SubsetScorer``(非加性子集打分 +
# 标量策略头 ``head_z``)。**弃用**的部件在此显式关掉:``DirectDCST.fail``
# (recent-failure / pass-1 探针头)整块摘除,``head_rescue`` / ``head_harm``
# 冻结且输出丢弃 —— 路线 §9 禁止 pass-1 draft 与 B=1 逐帧 probe。
#
# 本文件**不含**、且不得引入:milestone reward、step-level 正确性、gold 帧监督、
# pass-1 draft、B=1 探针、Gumbel-softmax / straight-through 软选帧。选择是**硬**
# 采样,梯度只经 ``log π(S)`` 回流(REINFORCE/GRPO),不穿过图片。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from causalcache_rl.direct_dcst import DirectDCST

# note (luojiaxuan): direct_dcst 的输入维度写死 4096(老线 token 缓存的口径)。
# 本文件允许换维度,做法是在构造后替换那三处输入投影,而**不修改** direct_dcst.py。
DCST_NATIVE_D_IN = 4096
SELECTOR_FORMAT = 1


class SelectorError(ValueError):
    """selector 侧的输入错误(候选/子集/预算不自洽)。继承 ValueError。"""


# ---------------------------------------------------------------- 配置

@dataclass(frozen=True)
class SelectorConfig:
    """selector 的全部结构与策略超参(随 checkpoint 一起存,评测端自恢复)。

    ``dim`` / ``k_spatial`` / ``k_global`` 是容量旋钮,``temperature`` /
    ``recent_strength`` / ``recent_tau`` 是策略旋钮。实测参数量(d_vis=d_text=4096):

    ==================  ========
    dim/k_spatial/k_global  参数量
    ==================  ========
    256 / 48 / 16       17.2M
    384 / 48 / 16       28.4M
    **512 / 96 / 32**   **41.2M**(默认)
    640 / 96 / 32       55.8M(**超出 15-50M 区间,慎用**)
    ==================  ========

    字段:``d_vis`` 冻结视觉特征维度(每帧 [T, d_vis]);``d_text`` 冻结词嵌入
    维度;``budget`` = primary B(路线 §0);``temperature`` 即 π=softmax(z/T)
    的 T;``recent_strength`` / ``recent_tau`` 见 ``init_recent_bias``;
    ``n_rank_buckets`` recency rank 桶数(超出并入最后一桶);``z_init_scale``
    head_z 权重初始缩放(<1 让起点更贴近纯先验);``subset_chunk`` 单次前向最多
    打多少个 subset(只影响显存,不影响数值)。
    """

    dim: int = 512
    k_spatial: int = 96
    k_global: int = 32
    dropout: float = 0.1
    d_vis: int = DCST_NATIVE_D_IN
    d_text: int = DCST_NATIVE_D_IN
    budget: int = 2
    temperature: float = 1.0
    recent_strength: float = 4.0
    recent_tau: float = 2.0
    n_rank_buckets: int = 32
    use_recency_prior: bool = True
    z_init_scale: float = 1.0
    subset_chunk: int = 64

    def validated(self) -> "SelectorConfig":
        if self.dim % 8 != 0:
            # note (luojiaxuan): direct_dcst 的 AttnPool 写死 8 头,dim 必须整除 8。
            raise SelectorError(f"dim 必须是 8 的倍数(AttnPool 写死 8 头),收到 {self.dim}")
        if self.budget < 0:
            raise SelectorError(f"budget 必须 ≥ 0,收到 {self.budget}")
        if self.temperature <= 0:
            raise SelectorError(f"temperature 必须 > 0,收到 {self.temperature}")
        if self.n_rank_buckets < 1:
            raise SelectorError(f"n_rank_buckets 必须 ≥ 1,收到 {self.n_rank_buckets}")
        if self.subset_chunk < 1:
            raise SelectorError(f"subset_chunk 必须 ≥ 1,收到 {self.subset_chunk}")
        return self

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SelectorConfig":
        """容忍旧/新 checkpoint 的字段增删:只取本类认识的字段。"""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known}).validated()


# ---------------------------------------------------------------- 冻结文本嵌入

def load_frozen_embedding(model_dir: str | Path, device: Any = "cpu") -> torch.Tensor:
    """只捞 GUI-Owl 的 ``embed_tokens.weight``,不加载策略模型。

    # note (luojiaxuan): 逐行照搬 rl/code/scripts/train_direct_dcst.py 的
    # ``frozen_embed``(路线要求复用同一个冻结词嵌入,不另训文本编码器)。
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    idx_file = model_dir / "model.safetensors.index.json"
    if idx_file.exists():
        weight_map = json.loads(idx_file.read_text())["weight_map"]
        name = next(k for k in weight_map if k.endswith("embed_tokens.weight"))
        shard = model_dir / weight_map[name]
    else:
        shard = model_dir / "model.safetensors"
        with safe_open(str(shard), framework="pt") as handle:
            name = next(k for k in handle.keys() if k.endswith("embed_tokens.weight"))
    with safe_open(str(shard), framework="pt") as handle:
        weight = handle.get_tensor(name)
    return weight.to(device=device, dtype=torch.float32).requires_grad_(False)


class FrozenTextEmbedder:
    """(tokenizer + 冻结词嵌入表)→ ``embed(text, max_tokens) -> [n, d_text]``。

    整表冻结、不进优化器。同一条 instruction 在一条 rollout 里会被查很多次,
    故内部带一个上限受控的字典缓存。
    """

    def __init__(self, weight: torch.Tensor, tokenizer: Any,
                 *, cache_size: int = 4096) -> None:
        self.weight = weight
        self.tokenizer = tokenizer
        self.cache_size = int(cache_size)
        self._cache: dict[tuple[str, int], torch.Tensor] = {}

    @property
    def dim(self) -> int:
        return int(self.weight.shape[-1])

    @property
    def device(self) -> Any:
        return self.weight.device

    @classmethod
    def from_model_dir(cls, model_dir: str | Path, device: Any = "cpu",
                       *, cache_size: int = 4096) -> "FrozenTextEmbedder":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        return cls(load_frozen_embedding(model_dir, device), tokenizer,
                   cache_size=cache_size)

    def embed(self, text: str, max_tokens: int) -> torch.Tensor:
        key = (text, int(max_tokens))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        ids = self.tokenizer(text[:4000], add_special_tokens=False,
                             return_tensors="pt")["input_ids"][0][: int(max_tokens)]
        if ids.numel() == 0:
            # note (luojiaxuan): 空段会让 AttnPool 的 key 全空 → NaN,这里补一个
            # pad/eos token 保证每段至少一个 token。
            fallback = getattr(self.tokenizer, "pad_token_id", None)
            if fallback is None:
                fallback = getattr(self.tokenizer, "eos_token_id", None) or 0
            ids = torch.tensor([int(fallback)], dtype=torch.long)
        out = self.weight[ids.to(self.weight.device)]
        if len(self._cache) < self.cache_size:
            self._cache[key] = out
        return out


class DummyTextEmbedder:
    """**仅供 CPU 自检**:哈希分词 + 随机词嵌入,替代冻结 GUI-Owl 词嵌入。

    不训练、不进优化器,只是让结构自检不必下载 8B 权重。生产路径必须用
    ``FrozenTextEmbedder``。
    """

    def __init__(self, dim: int = DCST_NATIVE_D_IN, vocab: int = 4096,
                 seed: int = 0, device: Any = "cpu") -> None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.weight = (torch.randn(vocab, dim, generator=gen) * 0.02).to(device)
        self.vocab = int(vocab)
        self._cache: dict[tuple[str, int], torch.Tensor] = {}

    @property
    def dim(self) -> int:
        return int(self.weight.shape[-1])

    @property
    def device(self) -> Any:
        return self.weight.device

    def embed(self, text: str, max_tokens: int) -> torch.Tensor:
        key = (text, int(max_tokens))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        words = text.replace("(", " ").replace(")", " ").split() or ["<empty>"]
        # note (luojiaxuan): 用 crc32 而不是内置 hash() —— 后者对 str 每进程加盐
        # (PYTHONHASHSEED),会让自检结果跨进程漂移,曾把一个数值断言弄成随机通过。
        ids = [zlib.crc32(w.encode("utf-8")) % self.vocab
               for w in words[: int(max_tokens)]]
        out = self.weight[torch.tensor(ids, dtype=torch.long,
                                       device=self.weight.device)]
        self._cache[key] = out
        return out


# ---------------------------------------------------------------- state 载体

@dataclass(frozen=True)
class SelectorStateRepr:
    """一步 rollout 的 selector 现场(``SelectorPolicy`` 的 ``state_repr``)。

    ``candidates`` 是**全局事件号**(= env 0-based 帧号,口径见 policy_io 模块
    docstring),局部下标 = 它在本元组中的位置。**允许任意顺序** —— 打分对候选
    顺序等变(见类型注释),``recency_rank`` 由事件号而非局部下标算出。

    ``candidates`` 字段名与 ``policy_io.SelectorState`` 一致,故 policy_io 的
    ``_candidate_pool`` / 基线臂可以直接吃本对象。

    **``st`` 携带一张活的 autograd 图**:一个 state 只能 backward 一次。rollout
    期请在 ``torch.no_grad()`` 下构造(或用 ``detached()``),GRPO 更新期为每个
    被更新的 step **重新** ``build`` 一次 —— 这与"训练采样、测试 argmax、梯度
    只经 log π 回流"的设计一致,不需要跨 update 复用图。

    字段:``st`` 是 ``DirectDCST.encode_state`` 的输出;``recent`` 是 recent-B 的
    全局事件号(升序);``ages`` = ``current_step - event``、``recency_rank``
    (0 = 最新)两者都与 ``candidates`` 同序;``index_of`` 是全局事件号 → 局部下标。
    """

    st: dict[str, Any]
    candidates: tuple[int, ...]
    current_step: int
    budget: int
    recent: tuple[int, ...]
    ages: tuple[int, ...]
    recency_rank: tuple[int, ...]
    index_of: Mapping[int, int] = field(default_factory=dict)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    def detached(self) -> "SelectorStateRepr":
        """切断 autograd 图的副本 —— rollout 期缓存 state 用,避免持有大图。"""
        st = {k: (v.detach() if isinstance(v, torch.Tensor) else v)
              for k, v in self.st.items()}
        return replace(self, st=st)

    def local(self, subset: Sequence[int]) -> tuple[int, ...]:
        """全局事件号 → 局部下标。越界即 fail-loud。"""
        out = []
        for event in subset:
            try:
                out.append(self.index_of[int(event)])
            except KeyError:
                raise SelectorError(
                    f"事件 {event} 不在候选集 {self.candidates} 中"
                ) from None
        if len(set(out)) != len(out):
            raise SelectorError(f"subset 有重复事件:{tuple(subset)}")
        return tuple(out)


# ---------------------------------------------------------------- state 组装

def _as_feature(value: Any, device: Any, dtype: torch.dtype) -> torch.Tensor:
    """任意 array-like → [T, d] float 张量(接 features.FeatureCache 的产物)。"""
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        # note (luojiaxuan): 容忍 features 侧带 batch 维的 [1,T,d] 缓存形态。
        tensor = tensor[0]
    if tensor.dim() != 2:
        raise SelectorError(f"帧特征必须是 [T, d] 或 [1, T, d],收到 shape {tuple(tensor.shape)}")
    return tensor


@dataclass
class SelectorStateBuilder:
    """把一步 rollout 的现场组装成 ``SelectorStateRepr``。

    输入:instruction 文本、历史动作行文本、候选帧冻结特征、当前帧特征、帧龄、
    recent-B —— 内部用**冻结词嵌入**编码文本(``FrozenTextEmbedder``,不重训
    文本编码器),产出 ``DirectDCST.encode_state`` 需要的六个入参并调用之。

    帧/事件编号口径与 ``policy_io`` 完全一致:帧索引 k == 官方事件号 k;
    事件 j 的 post 帧由**第 j 步(1-based)**的动作产生,故它的动作行是
    ``history_action_lines[j - 1]``;``age = current_step - j``。
    """

    # note (luojiaxuan): embedder 只需实现 embed(text, max_tokens) -> [n, d_text]
    # (``FrozenTextEmbedder`` 是生产实现,``DummyTextEmbedder`` 仅供自检)。
    embedder: Any
    max_instruction_tokens: int = 256
    max_action_tokens: int = 64
    instruction_chars: int = 2000
    action_chars: int = 200

    def build(
        self,
        policy: "SubsetSelectorPolicy",
        *,
        task_instruction: str,
        history_action_lines: Sequence[str],
        candidates: Sequence[int],
        frame_features: Callable[[int], Any] | Mapping[int, Any] | Sequence[Any],
        current_step: int,
        current_feature: Any | None = None,
        budget: int | None = None,
        recent: Sequence[int] | None = None,
    ) -> SelectorStateRepr:
        """组装并调用 ``policy.backbone.encode_state``,返回不可变 state。

        ``frame_features`` 可以是 callable(事件号 → 特征)、mapping、或与
        ``candidates`` 等长的序列。生产路径直接传 ``bank.feature``:
        ``features.attach_to_bank(bank, FeatureCache(...))`` 之后
        ``HistoryFrameBank.feature(step)`` 就是"事件号 → 冻结特征"的 callable
        (dtype 不限,这里统一转成策略的 dtype)。``current_feature`` 省略时按当前
        帧号(= ``current_step``)去取。

        # note (luojiaxuan): 已知小偏差 —— ``DirectDCST.encode_state`` 内部用
        # ``infer_grid(T)`` 把 token 数因子分解成行列,而 features.FeatureCache
        # 其实存了真实网格(``cache.grid(path)``)。1920×1080→2584 时两者一致
        # (38×68),但 T 为质数样的形状会退化成单行。真要用精确网格,得绕开
        # encode_state 直接调 ``resampler.forward_batch(xs, grids)``;此处按任务
        # 约定调 encode_state,先记下这个偏差而不擅自改老线接口。
        """
        cand = tuple(int(j) for j in candidates)
        if len(set(cand)) != len(cand):
            raise SelectorError(f"candidates 有重复:{cand}")
        now = int(current_step)
        if any(j >= now or j < 0 for j in cand):
            raise SelectorError(
                f"候选事件号必须落在 [0, {now - 1}](current_step={now}),收到 {cand}")
        lines = [str(x) for x in history_action_lines]
        if len(lines) != now:
            raise SelectorError(
                f"history_action_lines 长度 {len(lines)} 必须等于 current_step={now}")
        if any(j - 1 >= len(lines) or j - 1 < 0 for j in cand):
            raise SelectorError(
                f"候选事件 {cand} 缺少对应动作行(需要 history_action_lines[j-1])")

        b = policy.cfg.budget if budget is None else int(budget)
        if not cand:
            # note (luojiaxuan): early step(current_step ≤ 1)没有合法历史帧 ——
            # 动作空间退化成单个空集,不必也不能过 backbone(CandidateSetEncoder
            # 在 N=0 时会在 stack/MHA 上炸)。
            return SelectorStateRepr(
                st={}, candidates=(), current_step=now, budget=0, recent=(),
                ages=(), recency_rank=(), index_of={})
        device = policy.device
        dtype = policy.dtype
        lookup = _feature_lookup(frame_features, cand)

        toks = [_as_feature(lookup(j), device, dtype) for j in cand]
        cur_raw = lookup(now) if current_feature is None else current_feature
        cur = _as_feature(cur_raw, device, dtype)

        ctx_segs = [self._text(task_instruction, self.instruction_chars,
                               self.max_instruction_tokens, device, dtype)]
        for t, line in enumerate(lines, start=1):
            ctx_segs.append(self._text(f"Step{t}: {line}", self.action_chars,
                                       self.max_action_tokens, device, dtype))
        act_embs = [self._text(lines[j - 1], self.action_chars,
                               self.max_action_tokens, device, dtype) for j in cand]
        ages = tuple(now - j for j in cand)

        order = sorted(range(len(cand)), key=lambda i: cand[i], reverse=True)
        rank = [0] * len(cand)
        for position, local in enumerate(order):
            rank[local] = position
        eff = max(0, min(b, len(cand)))
        recent_local = order[:eff] if recent is None else [
            _require_index(cand, int(j)) for j in recent]
        recent_events = tuple(sorted(cand[i] for i in recent_local))

        st = policy.backbone.encode_state(
            toks, cur, ctx_segs, list(ages), act_embs, list(recent_local))
        return SelectorStateRepr(
            st=st, candidates=cand, current_step=now, budget=eff,
            recent=recent_events, ages=ages, recency_rank=tuple(rank),
            index_of={j: i for i, j in enumerate(cand)})

    def _text(self, text: str, max_chars: int, max_tokens: int,
              device: Any, dtype: torch.dtype) -> torch.Tensor:
        out = self.embedder.embed(str(text)[:max_chars], max_tokens)
        return out.to(device=device, dtype=dtype)


def _feature_lookup(source: Any, candidates: Sequence[int]) -> Callable[[int], Any]:
    if callable(source):
        return source
    if isinstance(source, Mapping):
        def _from_map(event: int) -> Any:
            try:
                return source[event]
            except KeyError:
                raise SelectorError(f"frame_features 缺少事件 {event} 的特征") from None
        return _from_map
    seq = list(source)
    if len(seq) != len(candidates):
        raise SelectorError(
            f"frame_features 是序列时必须与 candidates 等长(还需给当前帧特征),"
            f"收到 {len(seq)} vs {len(candidates)}")
    table = {int(j): seq[i] for i, j in enumerate(candidates)}

    def _from_seq(event: int) -> Any:
        try:
            return table[int(event)]
        except KeyError:
            raise SelectorError(
                f"frame_features 序列里没有事件 {event}(当前帧需显式传 current_feature)"
            ) from None
    return _from_seq


def _require_index(pool: Sequence[int], event: int) -> int:
    try:
        return list(pool).index(event)
    except ValueError:
        raise SelectorError(f"recent 里的事件 {event} 不在候选集 {tuple(pool)} 中") from None


# ---------------------------------------------------------------- 策略

class SubsetSelectorPolicy(nn.Module):
    """固定预算 B 的 subset 策略:``π(S | state) = softmax(z_θ(S) / T)``。

    满足 ``policy_io.SelectorPolicy`` Protocol(``score_subsets`` / ``sample`` /
    ``argmax``),另加 GRPO 需要的 ``logprob_of``(带梯度)与 ``entropy``。

    **动作空间**恒为全部 ``C(n, min(B, n))`` 个子集,枚举按事件号字典序;
    recent-B 是其中普通一项,没有额外的 KEEP 动作、没有 move cost。
    """

    def __init__(self, cfg: SelectorConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        cfg = (cfg or SelectorConfig())
        if overrides:
            cfg = replace(cfg, **overrides)
        self.cfg = cfg.validated()

        self.backbone = DirectDCST(d=cfg.dim, k_spatial=cfg.k_spatial,
                                   k_global=cfg.k_global, dropout=cfg.dropout,
                                   draft_dim=0)
        # note (luojiaxuan): 摘掉 recent-failure 头 —— 它是老线 pass-1 / B=1 探针
        # 那条被路线 §9 禁掉的支线,留着就是 7.4M 死参数。
        self.backbone.fail = None
        # note (luojiaxuan): rescue/harm 是老线离线分类目标,Phase 3 只用 head_z;
        # 冻结它们,免得 weight decay 白动一堆用不上的参数。
        self.backbone.scorer.head_rescue.requires_grad_(False)
        self.backbone.scorer.head_harm.requires_grad_(False)
        if cfg.d_vis != DCST_NATIVE_D_IN:
            self.backbone.resampler.ln_in = nn.LayerNorm(cfg.d_vis)
            self.backbone.resampler.w_in = nn.Linear(cfg.d_vis, cfg.dim)
        if cfg.d_text != DCST_NATIVE_D_IN:
            self.backbone.context.proj = nn.Sequential(
                nn.LayerNorm(cfg.d_text), nn.Linear(cfg.d_text, cfg.dim))
            self.backbone.set_enc.act_proj = nn.Sequential(
                nn.LayerNorm(cfg.d_text), nn.Linear(cfg.d_text, cfg.dim))
        # note (luojiaxuan): 实测随机初始化下 z 在各 subset 上的 std ≈ 0.07,远小于
        # 下面 recency 先验的量级(~1-4),所以初始 π 由先验决定而不是初始化的运气
        # (8 seed 实测 π(recent-2) ∈ [0.358, 0.483])。z_init_scale 默认 1.0 =
        # 保留 DirectDCST 原始 head_z 初始化;设 <1 可把起点压得更贴近纯先验,
        # 代价是回传到 body 的那一路梯度同比变小。
        with torch.no_grad():
            self.backbone.scorer.head_z.weight.mul_(cfg.z_init_scale)
            self.backbone.scorer.head_z.bias.zero_()

        self.temperature = float(cfg.temperature)
        # note (luojiaxuan): 唯一的加性项 —— 按 recency rank(0=最新)查表的**可学**
        # 偏置。它只是先验:训练可以把它压平、甚至翻成"偏好旧帧"。
        self.recency_bias = nn.Parameter(torch.zeros(cfg.n_rank_buckets))
        self.init_recent_bias()
        self._generators: dict[str, torch.Generator] = {}

    # ------------------------------------------------------------ 基本属性
    @property
    def device(self) -> Any:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def param_count(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)

    def describe(self) -> dict[str, Any]:
        return {"params": self.param_count(),
                "trainable": self.param_count(trainable_only=True),
                "cfg": asdict(self.cfg), "temperature": self.temperature}

    # ------------------------------------------------------------ recency 先验
    def init_recent_bias(self, strength: float | None = None,
                         tau: float | None = None) -> torch.Tensor:
        """写入"越靠近 recent 越高"的**可学**先验:``bias[r] = strength·e^{−r/τ}``。

        ``r`` 是 recency rank(0 = 最新候选)。子集先验 = 各成员 bias 之和,加在
        ``head_z`` 的 logit 上。默认 ``strength=4.0, τ=2.0`` 让 n=8/B=2 时初始
        π(recent-2) ≈ 0.4 —— **modest**,不是极强 KEEP bias(那是台账 §0.14
        判死的旧做法);``strength=0`` 即关掉先验、起点均匀。

        π(recent-B) 随候选数 n 自然下降(同一 strength 下 n=4→0.57、n=8→0.42、
        n=20→0.25),这是**想要**的:候选越多越该多探索,而相对 uniform 的偏好
        倍数仍有几十倍。要按别的 n 标定,用 ``prior_recent_prob`` 反解 strength。
        """
        s = self.cfg.recent_strength if strength is None else float(strength)
        t = self.cfg.recent_tau if tau is None else float(tau)
        ranks = torch.arange(self.cfg.n_rank_buckets, dtype=torch.float32)
        profile = torch.exp(-ranks / max(t, 1e-6))
        with torch.no_grad():
            self.recency_bias.copy_((s * profile).to(self.recency_bias.dtype))
        return self.recency_bias.detach()

    def _prior(self, state: SelectorStateRepr,
               local_subsets: Sequence[Sequence[int]]) -> torch.Tensor:
        n_rows = len(local_subsets)
        if not self.cfg.use_recency_prior or n_rows == 0 or not local_subsets[0]:
            return torch.zeros(n_rows, device=self.recency_bias.device,
                               dtype=self.recency_bias.dtype)
        cap = self.cfg.n_rank_buckets - 1
        idx = torch.tensor(
            [[min(state.recency_rank[i], cap) for i in row] for row in local_subsets],
            dtype=torch.long, device=self.recency_bias.device)
        return self.recency_bias[idx].sum(-1)

    # ------------------------------------------------------------ 打分
    def forward(self, state_repr: SelectorStateRepr,
                subsets: Sequence[Sequence[int]]) -> torch.Tensor:
        return self.score_subsets(state_repr, subsets)

    def score_subsets(self, state_repr: SelectorStateRepr,
                      subsets: Sequence[Sequence[int]]) -> torch.Tensor:
        """对候选 subset 批量打分,返回 **未除温度** 的 logits ``z``,顺序同入参。

        ``subsets`` 用**全局事件号**(与 policy_io 一致)。非加性:每个子集整体
        进 ``SubsetScorer``(fine latents + role + H_S/H_R)出一个标量 ``head_z``,
        **不存在**任何逐帧标量再求和的路径。

        实测(CPU 自检,n=4/B=2):随机初始化时交互项很弱 —— 四元组残差
        ``z(1,2)+z(3,4)−z(1,4)−z(3,2) ≈ 1e-4``,而 z 在 28 个子集上的 std ≈ 7e-2,
        即**起点近似可加**;但拿一个"任何逐帧可加打分都拟合不了"的目标训练时,
        本 scorer 的 MSE 降到 0.16,而最优可加打分的下界是 0.667 —— **交互能力
        是靠训练长出来的,不是初始化就有的**。GRPO 早期看不到明显非加性行为属正常。
        """
        state = _require_state(state_repr)
        rows = [tuple(int(j) for j in s) for s in subsets]
        if not rows:
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        sizes = {len(r) for r in rows}
        if len(sizes) != 1:
            raise SelectorError(f"同一批 subset 必须等基数,收到 {sorted(sizes)}")
        local = [state.local(r) for r in rows]
        if not local[0]:
            # note (luojiaxuan): 空预算(B=0 或无候选)—— 只有一个空动作,logit 恒 0,
            # 但保留与参数的零梯度连接,免得上层 backward 在退化态崩掉。
            return self._zeros_with_graph(len(rows))
        parts = []
        for start in range(0, len(local), self.cfg.subset_chunk):
            chunk = local[start:start + self.cfg.subset_chunk]
            # note (luojiaxuan): 只取第三个输出 head_z;rescue/harm 丢弃(老线目标)。
            _, _, z = self.backbone.score_subsets(state.st, list(chunk))
            parts.append(z)
        return torch.cat(parts) + self._prior(state, local)

    def _zeros_with_graph(self, n: int) -> torch.Tensor:
        base = self.recency_bias.sum() * 0.0
        return base.expand(n) if n else base.new_zeros(0)

    # ------------------------------------------------------------ 分布
    def action_space(self, candidates: Sequence[int], b: int) -> list[tuple[int, ...]]:
        """全部 ``C(n, min(b, n))`` 个子集,按事件号字典序(升序元组)。"""
        pool = sorted(int(j) for j in candidates)
        if len(set(pool)) != len(pool):
            raise SelectorError(f"candidates 有重复:{tuple(candidates)}")
        k = max(0, min(int(b), len(pool)))
        return list(itertools.combinations(pool, k))

    def distribution(self, state_repr: SelectorStateRepr,
                     candidates: Sequence[int] | None = None,
                     b: int | None = None
                     ) -> tuple[list[tuple[int, ...]], torch.Tensor, torch.Tensor]:
        """一次算清 ``(全部动作, 未归一 logits, log π)``。

        其他四个接口(sample/argmax/logprob_of/entropy)全部走这里,**保证同分布**。
        """
        state = _require_state(state_repr)
        pool = state.candidates if candidates is None else candidates
        _check_pool(state, pool)
        budget = state.budget if b is None else int(b)
        subsets = self.action_space(pool, budget)
        logits = self.score_subsets(state, subsets) / self.temperature
        return subsets, logits, torch.log_softmax(logits, dim=0)

    def sample(self, state_repr: SelectorStateRepr,
               candidates: Sequence[int] | None = None, b: int | None = None,
               *, generator: torch.Generator | None = None
               ) -> tuple[tuple[int, ...], float]:
        """训练期采样:按 ``softmax(z / temperature)`` 抽一个子集。

        返回 ``(升序子集, log π(S))``;log-prob 是 **float**(已 detach)。GRPO
        更新时用 ``logprob_of`` 重算带梯度的同一数值。
        """
        subsets, _, logprobs = self.distribution(state_repr, candidates, b)
        if len(subsets) <= 1:
            return (subsets[0] if subsets else ()), 0.0
        probs = logprobs.detach().exp()
        gen = generator or self._generator(probs.device)
        idx = int(torch.multinomial(probs, 1, generator=gen).item())
        return subsets[idx], float(logprobs[idx].detach().item())

    def argmax(self, state_repr: SelectorStateRepr,
               candidates: Sequence[int] | None = None,
               b: int | None = None) -> tuple[int, ...]:
        """评测期确定性动作:logits 最大的子集(温度 > 0 时与 log π 的 argmax 同解)。"""
        subsets, logits, _ = self.distribution(state_repr, candidates, b)
        if not subsets:
            return ()
        return subsets[int(torch.argmax(logits).item())]

    def logprob_of(self, state_repr: SelectorStateRepr,
                   candidates: Sequence[int] | None = None,
                   b: int | None = None,
                   subset: Sequence[int] | None = None) -> torch.Tensor:
        """``log π(subset | state)``,**带梯度**(GRPO 的 ratio 分子)。

        与 ``sample`` 严格同分布:同一个 ``distribution`` 出口,同一个温度。
        """
        if subset is None:
            raise SelectorError("logprob_of 必须给 subset")
        subsets, _, logprobs = self.distribution(state_repr, candidates, b)
        want = tuple(sorted(int(j) for j in subset))
        if tuple(int(j) for j in subset) != want:
            # note (luojiaxuan): 与 policy_io.validate_subset 同口径 —— 子集必须
            # 严格递增,乱序视为记账错乱而不是等价输入。
            raise SelectorError(f"subset 必须严格递增,收到 {tuple(subset)}")
        try:
            idx = subsets.index(want)
        except ValueError:
            raise SelectorError(
                f"subset {want} 不在动作空间里(候选 {tuple(subsets[0]) if subsets else ()} …,"
                f"共 {len(subsets)} 个)") from None
        return logprobs[idx]

    def entropy(self, state_repr: SelectorStateRepr,
                candidates: Sequence[int] | None = None,
                b: int | None = None) -> torch.Tensor:
        """策略熵(nat),带梯度 —— GRPO 的 entropy bonus 与塌缩监控都读它。"""
        subsets, _, logprobs = self.distribution(state_repr, candidates, b)
        if len(subsets) <= 1:
            return self._zeros_with_graph(0).sum()
        # note (luojiaxuan): 按约定 0·log0 := 0。原式 `p*logp` 在 p 下溢到 0 时
        # 得 0*(-inf) = **nan**,nan 顺着 entropy bonus 进梯度,grad_norm 变
        # Infinity,一个优化步就把权重打坏。Phase 3 没暴露是因为那时 selector
        # 刚初始化、分布接近均匀;Phase 4 从**已训练的尖锐分布**出发,尾部子集
        # 概率真的会下溢,于是第一次联合更新就炸了(ratio 均值 4.4e+22)。
        probs = logprobs.exp()
        terms = torch.where(probs > 0, probs * logprobs,
                            torch.zeros_like(probs))
        return -terms.sum()

    def recent_prob(self, state_repr: SelectorStateRepr) -> float:
        """诊断:当前 π(recent-B)。塌缩监控与"选 recent-B 比例"报表用。"""
        state = _require_state(state_repr)
        subsets, _, logprobs = self.distribution(state)
        if not state.recent:
            return 1.0
        try:
            idx = subsets.index(tuple(state.recent))
        except ValueError:
            return 0.0
        return float(logprobs[idx].detach().exp().item())

    # ------------------------------------------------------------ 随机源
    def seed(self, value: int) -> None:
        """固定采样随机源(每设备一条)。"""
        self._generators = {}
        self._seed = int(value)

    def _generator(self, device: Any) -> torch.Generator:
        key = str(device)
        gen = self._generators.get(key)
        if gen is None:
            gen = torch.Generator(device=device)
            base = getattr(self, "_seed", None)
            if base is not None:
                gen.manual_seed(base)
            else:
                gen.seed()
            self._generators[key] = gen
        return gen

    # ------------------------------------------------------------ 存取
    def save(self, path: str | Path) -> Path:
        """存 ``{cfg, state_dict}`` —— 评测端只给路径即可自恢复结构。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": SELECTOR_FORMAT, "cfg": asdict(self.cfg),
                    "temperature": self.temperature,
                    "state_dict": self.state_dict()}, target)
        return target

    @classmethod
    def load(cls, path: str | Path, map_location: Any = "cpu",
             *, strict: bool = True) -> "SubsetSelectorPolicy":
        blob = torch.load(str(path), map_location=map_location, weights_only=False)
        if "cfg" not in blob or "state_dict" not in blob:
            raise SelectorError(f"{path} 不是 selector checkpoint(缺 cfg/state_dict)")
        model = cls(SelectorConfig.from_dict(blob["cfg"]))
        model.load_state_dict(blob["state_dict"], strict=strict)
        model.temperature = float(blob.get("temperature", model.cfg.temperature))
        return model.to(map_location)

    def load_pretrained_backbone(self, path: str | Path,
                                 map_location: Any = "cpu") -> dict[str, int]:
        """用老线 DirectDCST checkpoint 热启 backbone(路线 §1 的"初始化资产")。

        非严格加载:形状不合或本模型已摘掉的键(``fail.*``)自动跳过,返回
        ``{"matched": …, "skipped": …}`` 供 run docs 记账。**纯初始化**,不引入
        任何老目标(rescue/harm/pass-1)进训练。
        """
        blob = torch.load(str(path), map_location=map_location, weights_only=False)
        raw = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        own = self.backbone.state_dict()
        take = {k: v for k, v in raw.items()
                if k in own and tuple(own[k].shape) == tuple(v.shape)}
        self.backbone.load_state_dict(take, strict=False)
        return {"matched": len(take), "skipped": len(raw) - len(take)}


def _require_state(state_repr: Any) -> SelectorStateRepr:
    if not isinstance(state_repr, SelectorStateRepr):
        raise SelectorError(
            "state_repr 必须是 SelectorStateBuilder.build 产出的 SelectorStateRepr,"
            f"收到 {type(state_repr).__name__}")
    return state_repr


def _check_pool(state: SelectorStateRepr, pool: Sequence[int]) -> None:
    given = {int(j) for j in pool}
    if given != set(state.candidates):
        raise SelectorError(
            f"candidates {sorted(given)} 与 state 的候选集 {sorted(state.candidates)} "
            "不一致(state 是按候选特征编码的,不能事后换候选池)")


# ---------------------------------------------------------------- CLI 配置

def add_selector_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """把 SelectorConfig 的旋钮挂到训练/评测脚本的 argparse 上。"""
    g = parser.add_argument_group("selector")
    d = SelectorConfig()
    g.add_argument("--dim", type=int, default=d.dim)
    g.add_argument("--k-spatial", type=int, default=d.k_spatial)
    g.add_argument("--k-global", type=int, default=d.k_global)
    g.add_argument("--dropout", type=float, default=d.dropout)
    g.add_argument("--d-vis", type=int, default=d.d_vis)
    g.add_argument("--d-text", type=int, default=d.d_text)
    g.add_argument("--budget", type=int, default=d.budget)
    g.add_argument("--temperature", type=float, default=d.temperature)
    g.add_argument("--recent-strength", type=float, default=d.recent_strength,
                   help="modest recent-B 先验强度;0 = 关掉先验")
    g.add_argument("--recent-tau", type=float, default=d.recent_tau)
    g.add_argument("--z-init-scale", type=float, default=d.z_init_scale)
    g.add_argument("--subset-chunk", type=int, default=d.subset_chunk)
    g.add_argument("--no-recency-prior", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> SelectorConfig:
    d = SelectorConfig()
    return SelectorConfig(
        dim=getattr(args, "dim", d.dim),
        k_spatial=getattr(args, "k_spatial", d.k_spatial),
        k_global=getattr(args, "k_global", d.k_global),
        dropout=getattr(args, "dropout", d.dropout),
        d_vis=getattr(args, "d_vis", d.d_vis),
        d_text=getattr(args, "d_text", d.d_text),
        budget=getattr(args, "budget", d.budget),
        temperature=getattr(args, "temperature", d.temperature),
        recent_strength=getattr(args, "recent_strength", d.recent_strength),
        recent_tau=getattr(args, "recent_tau", d.recent_tau),
        z_init_scale=getattr(args, "z_init_scale", d.z_init_scale),
        subset_chunk=getattr(args, "subset_chunk", d.subset_chunk),
        use_recency_prior=not getattr(args, "no_recency_prior", False),
    ).validated()


def prior_recent_prob(strength: float, tau: float, n: int, b: int) -> float:
    """解析诊断:只算 recency 先验(z≡0)时的 π(recent-b),用于标定 strength。"""
    if n <= 0 or b <= 0 or b >= n:
        return 1.0
    bias = [strength * math.exp(-r / max(tau, 1e-6)) for r in range(n)]
    scores = [sum(bias[r] for r in combo)
              for combo in itertools.combinations(range(n), b)]
    top = math.exp(sum(bias[:b]) - max(scores))
    return top / sum(math.exp(s - max(scores)) for s in scores)


__all__ = [
    "DummyTextEmbedder",
    "FrozenTextEmbedder",
    "SELECTOR_FORMAT",
    "SelectorConfig",
    "SelectorError",
    "SelectorStateBuilder",
    "SelectorStateRepr",
    "SubsetSelectorPolicy",
    "add_selector_cli_args",
    "config_from_args",
    "load_frozen_embedding",
    "prior_recent_prob",
]
