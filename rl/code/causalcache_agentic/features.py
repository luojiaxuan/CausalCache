"""Phase 3 selector 的冻结视觉特征提取与缓存。

# note (luojiaxuan): 路线见 rl/docs/agentic_memory_rl_roadmap_20260813.md §4
# (Phase 3);共享契约见 contract.py(只 import 不改)。selector 的输入之一是
# "全部候选历史帧的缓存特征",本模块就是产出这份特征的唯一入口。
#
# **特征口径与老线逐行同源**(rl/code/scripts/cache_fullres_features.py 的
# ``encode_frame``,``--regions raw`` 分支),这样 Phase 3 的 selector 与老线
# selector 共用同一份特征定义、可直接比较:
#   1. 每帧**单独**过一次单图 prompt(``[{"role":"user","content":[{"type":
#      "image","image":path}]}]``),``add_generation_prompt=False``;
#   2. 取 ``out.hidden_states[0]``,即**嵌入层**输出 —— 视觉塔 + 投影之后、
#      任何 LLM 层之前,故**不含跨帧上下文、不含文本上下文**;
#   3. 用 ``mm_token_type_ids == 1`` 定位图像 token 段(取首尾之间的连续段);
#   4. 用 ``image_grid_thw`` 还原二维网格 ``R=(h//merge)*max(t,1)``、
#      ``C=w//merge``,并断言 ``R*C == 段长``(错位即抛,不静默容忍);
#   5. **不做第二次池化**:逐帧保留全部 ~2584 个 token(老线 raw 模式)。
#      老线的 10×16 区域平均把 16 个 token 混成一个向量,文字级细节在特征层
#      被二次毁掉,是"selector 停在 recency"的头号嫌疑,故这里不复现它;
#   6. 落地为 ``[T, 4096]`` fp16 **CPU** 张量(老线 ``.half().cpu()`` 同口径)。
#
# **逐帧独立**是部署故事的关键:一帧在自己当"当前屏"的那一步本来就被冻结
# policy 全清编码过,rollout 里缓存复用即可,推理期零额外编码成本。
#
# 本模块**不含**、且按路线 §9 永久不得引入:pass-1 draft、B=1 逐帧 probe、
# gold 帧监督、milestone reward、step-level 正确性、Gumbel-softmax 软选帧。
# 这里只做"把像素变成冻结向量"这一件事,不产生任何监督信号。
#
# 依赖姿势:torch / transformers / causalcache.osworld_gui_owl **全部惰性
# import**,故本模块在没有 GPU、甚至没装 torch 的机器上也能 import,
# ``DummyFeatureExtractor`` + ``FeatureCache`` 的结构逻辑可在 CPU 上跑通
# (无 torch 时后端自动退化为 numpy,见 ``backend()``)。
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from causalcache_agentic.policy_io import HistoryFrameBank

if TYPE_CHECKING:  # pragma: no cover
    import torch


DEFAULT_VISUAL_TOKENS = 2560     # 全清预算,与冻结 policy 的动作遍一致
DEFAULT_FEATURE_DTYPE = "float16"
DEFAULT_MAX_ITEMS = 64           # ≈64 × 21MB ≈ 1.3GB 常驻(见 estimate_bytes)
DEFAULT_DUMMY_DIM = 4096
DEFAULT_DUMMY_TOKENS = 64
_GRID_MEMO_SIZE = 512


class FeatureError(ValueError):
    """本模块统一的错误基类(继承 ValueError,调用方可只捕 ValueError)。"""


# ------------------------------------------------------------------ 后端选择

_TORCH: Any = None
_TORCH_PROBED = False


def _torch_or_none() -> Any:
    """惰性探测 torch。没装就返回 None(本机 macOS 走 numpy 后端)。"""
    global _TORCH, _TORCH_PROBED
    if not _TORCH_PROBED:
        _TORCH_PROBED = True
        try:
            import torch as _t
        except ImportError:
            _TORCH = None
        else:
            _TORCH = _t
    return _TORCH


def backend() -> str:
    """当前张量后端:``"torch"`` 或 ``"numpy"``。"""
    return "numpy" if _torch_or_none() is None else "torch"


_DTYPE_ALIASES = {
    "float16": "float16", "fp16": "float16", "half": "float16",
    "float32": "float32", "fp32": "float32", "float": "float32",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
}


def _canonical_dtype(name: str) -> str:
    key = str(name).strip().lower()
    try:
        return _DTYPE_ALIASES[key]
    except KeyError:
        raise FeatureError(
            f"未知 dtype {name!r},可选 {sorted(set(_DTYPE_ALIASES.values()))}"
        ) from None


def _torch_dtype(torch_mod: Any, name: str) -> Any:
    return getattr(torch_mod, _canonical_dtype(name))


def _numpy_dtype(name: str) -> Any:
    canonical = _canonical_dtype(name)
    if canonical == "bfloat16":
        # note (luojiaxuan): numpy 没有 bfloat16;无 torch 的纯 CPU 结构测试里
        # 退化成 float32,并在 signature 里保留 bf16 字样,避免与真跑的盘缓存混用。
        return np.float32
    return getattr(np, canonical)


def _to_backend_array(arr: np.ndarray, dtype_name: str) -> Any:
    """numpy float32 → 目标后端 + 目标 dtype(torch 在则给 torch.Tensor)。"""
    torch_mod = _torch_or_none()
    if torch_mod is None:
        return np.ascontiguousarray(arr, dtype=_numpy_dtype(dtype_name))
    tensor = torch_mod.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    return tensor.to(_torch_dtype(torch_mod, dtype_name)).contiguous()


def tensor_bytes(value: Any) -> int:
    """张量占用字节数(torch / numpy 通吃),用于缓存体积统计。"""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel()) * int(element_size())
    return 0


def _shape_of(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise FeatureError(f"特征必须是张量(带 .shape),收到 {type(value).__name__}")
    return tuple(int(x) for x in shape)


def estimate_bytes(
    tokens: int = 2584, dim: int = 4096, dtype: str = DEFAULT_FEATURE_DTYPE
) -> int:
    """单帧特征的字节数估计(默认 ~21MB),用于设 ``max_items``。"""
    width = 4 if _canonical_dtype(dtype) == "float32" else 2
    return int(tokens) * int(dim) * width


def _sanitize(text: str) -> str:
    """把任意字符串压成文件系统安全的短标识(缓存命名空间用)。"""
    keep = [c if (c.isalnum() or c in "-_.") else "-" for c in str(text)]
    return "".join(keep)[:64] or "none"


# ------------------------------------------------------------------ 提取器接口

@runtime_checkable
class FeatureExtractor(Protocol):
    """``FeatureCache`` 只依赖这三件东西,故真提取器与 Dummy 可互换。"""

    signature: str
    feature_dim: int

    def encode(self, path: str) -> Any:
        """单帧 → ``[T, D]`` 张量(CPU)。"""
        ...

    def grid(self, path: str) -> tuple[int, int]:
        """单帧 → 视觉 token 的二维网格 ``(rows, cols)``。"""
        ...


# ------------------------------------------------------------------ 真提取器

class FrozenFeatureExtractor:
    """冻结 GUI-Owl 的逐帧独立视觉特征(口径见模块 docstring)。

    参数
    ----
    model_dir / snapshot_manifest
        与 harness 同款:传给 ``GUIOwlOSWorldRuntime``,冻结守卫会校验权重身份。
    device
        ``"cuda:0"`` 之类。特征最终恒落到 **CPU**(缓存要能跨 step 常驻)。
    visual_tokens
        全清预算,默认 2560 —— 与冻结 policy 的动作遍一致。改它会改变 T,
        故它进 ``signature``,盘缓存不会与别的预算串味。
    dtype
        **输出特征**的 dtype,默认 fp16(老线同口径)。模型本身的计算精度由
        ``GUIOwlOSWorldRuntime`` 固定为 bf16,本参数不改动它。
    runtime
        可选:复用同 host 上已构造好的 ``GUIOwlOSWorldRuntime``(rollout 里
        executor 已经加载了同一个 8B 基座),避免第二份权重多占约 16GB 显存。
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        snapshot_manifest: str | Path | None = None,
        device: str = "cuda:0",
        visual_tokens: int = DEFAULT_VISUAL_TOKENS,
        dtype: str = DEFAULT_FEATURE_DTYPE,
        *,
        max_new_tokens: int = 16,
        runtime: Any | None = None,
    ) -> None:
        import torch  # note (luojiaxuan): 惰性 —— 无 torch 的机器仍可 import 本模块。

        from causalcache.osworld_gui_owl import (
            VISION_SPATIAL_MERGE_SIZE,
            GUIOwlOSWorldRuntime,
        )

        if int(visual_tokens) <= 0:
            raise FeatureError(f"visual_tokens 必须为正,收到 {visual_tokens!r}")
        self._torch = torch
        self._merge = int(VISION_SPATIAL_MERGE_SIZE)
        self.device = str(device)
        self.visual_tokens = int(visual_tokens)
        self.dtype_name = _canonical_dtype(dtype)
        self._dtype = _torch_dtype(torch, self.dtype_name)
        if runtime is None:
            if model_dir is None or snapshot_manifest is None:
                raise FeatureError(
                    "未注入 runtime 时,model_dir 与 snapshot_manifest 均为必填"
                )
            runtime = GUIOwlOSWorldRuntime(
                model_dir=model_dir,
                expected_snapshot_manifest=snapshot_manifest,
                device=self.device,
                effective_visual_tokens_per_image=self.visual_tokens,
                max_new_tokens=int(max_new_tokens),
            )
        self.runtime = runtime
        self.metadata = dict(getattr(runtime, "metadata", {}))
        self._feature_dim = self._infer_hidden_size()
        self._grid_memo: OrderedDict[str, tuple[int, int]] = OrderedDict()

    # ---------------------------------------------------------- 身份/元信息
    @property
    def signature(self) -> str:
        """盘缓存命名空间:模型版本 + 视觉预算 + 输出 dtype。

        任何一项变了,特征就不是同一件东西,必须换目录 —— 否则会读到别的
        口径的旧张量,而且**静默**(形状可能恰好一样)。
        """
        revision = str(self.metadata.get("model_revision") or "unknown")
        return _sanitize(f"guiowl-{revision[:12]}-vt{self.visual_tokens}"
                         f"-{self.dtype_name}")

    @property
    def feature_dim(self) -> int:
        return int(self._feature_dim)

    def _infer_hidden_size(self) -> int:
        config = getattr(getattr(self.runtime, "model", None), "config", None)
        for holder in (getattr(config, "text_config", None), config):
            size = getattr(holder, "hidden_size", None)
            if isinstance(size, int) and size > 0:
                return int(size)
        # note (luojiaxuan): 拿不到就先记 0,第一次 encode 之后按实测填上;
        # 猜一个 4096 会让下游的维度断言变成"看起来对",反而更难查。
        return 0

    # ---------------------------------------------------------- 主接口
    def encode(self, path: str) -> "torch.Tensor":
        """单帧 → ``[T, D]`` fp16 CPU 张量(T ≈ 2584 @ 1920×1080, 2560 预算)。"""
        torch = self._torch
        enc = self._encode_inputs(path)
        with torch.inference_mode():
            out = self.runtime.model(**enc, output_hidden_states=True)
        # note (luojiaxuan): hidden_states[0] = 嵌入层输出 = 视觉塔+投影后的
        # 纯视觉特征,不含任何 LLM 层的跨帧/文本上下文。老线同一行。
        hidden = out.hidden_states[0][0]
        segment = self._image_segment(hidden, enc, path)
        rows, cols = self._grid_from_enc(enc, path)
        if rows * cols != int(segment.shape[0]):
            raise FeatureError(
                f"{path}: grid {rows}x{cols} 与图像段长 {int(segment.shape[0])} 不符"
            )
        self._remember_grid(path, (rows, cols))
        feat = segment.detach().float().to(self._dtype).cpu().contiguous()
        self._feature_dim = int(feat.shape[1])
        return feat

    def grid(self, path: str) -> tuple[int, int]:
        """单帧的 ``(rows, cols)``。命中 memo 时零成本,否则只跑 processor。"""
        key = str(path)
        cached = self._grid_memo.get(key)
        if cached is not None:
            self._grid_memo.move_to_end(key)
            return cached
        enc = self._encode_inputs(key)
        value = self._grid_from_enc(enc, key)
        self._remember_grid(key, value)
        return value

    # ---------------------------------------------------------- 内部
    def _encode_inputs(self, path: str) -> Any:
        """单图 prompt → processor 张量(与老线 encode_frame 逐行同构)。"""
        messages = [{"role": "user",
                     "content": [{"type": "image", "image": str(path)}]}]
        return self.runtime.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.runtime.device)

    @staticmethod
    def _image_segment(hidden: Any, enc: Any, path: str) -> Any:
        mm = enc.get("mm_token_type_ids")
        if mm is None:
            raise FeatureError(f"{path}: processor 没给 mm_token_type_ids")
        index = (mm[0] == 1).nonzero(as_tuple=True)[0]
        if index.numel() == 0:
            raise FeatureError(f"{path}: prompt 里没有图像 token")
        return hidden[index[0]: index[-1] + 1]

    def _grid_from_enc(self, enc: Any, path: str) -> tuple[int, int]:
        thw = enc.get("image_grid_thw")
        if thw is None:
            raise FeatureError(f"{path}: processor 没给 image_grid_thw")
        t, h, w = (int(x) for x in thw[0])
        return (h // self._merge) * max(t, 1), w // self._merge

    def _remember_grid(self, path: str, value: tuple[int, int]) -> None:
        self._grid_memo[path] = value
        self._grid_memo.move_to_end(path)
        while len(self._grid_memo) > _GRID_MEMO_SIZE:
            self._grid_memo.popitem(last=False)


# ------------------------------------------------------------------ 假提取器

@dataclass
class DummyFeatureExtractor:
    """确定性假特征(CPU 单测与 ``--dry-run`` 用),**不碰模型**。

    确定性保证:特征只由 ``sha1(f"{seed}::{path}")`` 决定 —— 与 ``PYTHONHASHSEED``
    无关(不用内建 ``hash()``)、与调用顺序无关、与进程无关。同一 ``(path, dim,
    tokens, seed, dtype)`` 在任何进程/任何次序下逐位相同;不同 path 几乎必然
    不同。跨 numpy 主版本的比特级一致依赖 ``default_rng``(PCG64)的流稳定性,
    这在单机自检的语境下足够。
    """

    dim: int = DEFAULT_DUMMY_DIM
    tokens: int = DEFAULT_DUMMY_TOKENS
    seed: int = 0
    dtype: str = DEFAULT_FEATURE_DTYPE
    encode_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if int(self.dim) <= 0 or int(self.tokens) <= 0:
            raise FeatureError(f"dim/tokens 必须为正,收到 {self.dim}/{self.tokens}")
        self.dim = int(self.dim)
        self.tokens = int(self.tokens)
        self.seed = int(self.seed)
        self.dtype = _canonical_dtype(self.dtype)

    @property
    def signature(self) -> str:
        return _sanitize(f"dummy-d{self.dim}-t{self.tokens}-s{self.seed}"
                         f"-{self.dtype}")

    @property
    def feature_dim(self) -> int:
        return self.dim

    def encode(self, path: str) -> Any:
        self.encode_calls += 1
        digest = hashlib.sha1(f"{self.seed}::{path}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        raw = rng.standard_normal((self.tokens, self.dim), dtype=np.float32)
        return _to_backend_array(raw, self.dtype)

    def grid(self, path: str) -> tuple[int, int]:
        """把 ``tokens`` 分解成最接近正方形的 ``rows × cols``(确定性)。"""
        rows = 1
        for candidate in range(1, int(self.tokens ** 0.5) + 1):
            if self.tokens % candidate == 0:
                rows = candidate
        return rows, self.tokens // rows


# ------------------------------------------------------------------ 缓存

@dataclass
class _Entry:
    feat: Any
    grid: tuple[int, int] | None
    nbytes: int


class FeatureCache:
    """路径 → 特征的 LRU 内存缓存 + 可选盘缓存。

    * ``max_items`` 控内存上限(每帧 ~21MB fp16,见 ``estimate_bytes``);
    * ``cache_dir`` 给定时落盘到 ``<cache_dir>/<namespace>/<sha1(path)>.pt``,
      ``namespace`` 默认取 extractor 的 ``signature``(模型版本+视觉预算+dtype),
      **不同口径不会互相污染**;写盘走 ``.tmp`` + rename,多 shard 并发安全;
    * 统计口径(``stats()``):``hits`` 只数内存命中;``misses`` 数内存未命中;
      ``disk_hits ⊆ misses``;``encodes = misses − disk_hits`` 才是真正过模型的
      次数。这样"同一路径只编码一次"可以被 ``encodes`` 直接证伪。

    线程安全**不做保证**(路线里 rollout 是单进程顺序推进的)。
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        *,
        max_items: int = DEFAULT_MAX_ITEMS,
        cache_dir: str | Path | None = None,
        namespace: str | None = None,
    ) -> None:
        if not hasattr(extractor, "encode"):
            raise FeatureError("extractor 必须提供 encode(path)")
        if int(max_items) < 1:
            raise FeatureError(f"max_items 必须 ≥ 1,收到 {max_items!r}")
        self.extractor = extractor
        self.max_items = int(max_items)
        self.namespace = _sanitize(
            namespace if namespace is not None
            else getattr(extractor, "signature", "default")
        )
        self._backend = backend()
        self.cache_dir: Path | None = None
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir) / self.namespace
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lru: "OrderedDict[str, _Entry]" = OrderedDict()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "encodes": 0,
                       "disk_hits": 0, "disk_writes": 0, "disk_errors": 0}

    # ---------------------------------------------------------- 主接口
    def get(self, path: str) -> Any:
        """取特征。内存 → 盘 → 编码,三级回退。"""
        key = str(path)
        entry = self._lru.get(key)
        if entry is not None:
            self._lru.move_to_end(key)
            self._stats["hits"] += 1
            return entry.feat
        self._stats["misses"] += 1
        entry = self._load_disk(key)
        if entry is None:
            feat = self.extractor.encode(key)
            self._stats["encodes"] += 1
            shape = _shape_of(feat)
            if len(shape) != 2:
                raise FeatureError(
                    f"{key}: 特征必须是 [T, D] 二维张量,收到 shape={shape}"
                )
            grid = self._grid_of(key)
            entry = _Entry(feat=feat, grid=grid, nbytes=tensor_bytes(feat))
            self._write_disk(key, entry)
        self._insert(key, entry)
        return entry.feat

    def grid(self, path: str) -> tuple[int, int] | None:
        """``(rows, cols)``。不计入 hit/miss —— 它不消耗编码预算。"""
        key = str(path)
        entry = self._lru.get(key)
        if entry is not None and entry.grid is not None:
            return entry.grid
        return self._grid_of(key)

    def preload(self, paths: "list[str] | tuple[str, ...]") -> int:
        """批量预热,返回本次真正过模型的次数(便于日志里核对编码成本)。"""
        before = self._stats["encodes"]
        for path in paths:
            self.get(path)
        return self._stats["encodes"] - before

    def stats(self) -> dict[str, Any]:
        out = dict(self._stats)
        out["size"] = len(self._lru)
        out["max_items"] = self.max_items
        out["bytes"] = sum(e.nbytes for e in self._lru.values())
        out["backend"] = self._backend
        out["namespace"] = self.namespace
        out["cache_dir"] = str(self.cache_dir) if self.cache_dir else None
        return out

    def clear(self) -> None:
        """清空内存缓存(**不动盘**,盘上的是可复用的昂贵产物)。"""
        self._lru.clear()

    def __call__(self, step: int, path: str) -> Any:
        """``HistoryFrameBank.feature_provider`` 的签名:``(step, path) -> 特征``。"""
        return self.get(path)

    def __len__(self) -> int:
        return len(self._lru)

    # ---------------------------------------------------------- 内部
    def _grid_of(self, key: str) -> tuple[int, int] | None:
        getter = getattr(self.extractor, "grid", None)
        if getter is None:
            return None
        value = getter(key)
        return None if value is None else (int(value[0]), int(value[1]))

    def _insert(self, key: str, entry: _Entry) -> None:
        self._lru[key] = entry
        self._lru.move_to_end(key)
        while len(self._lru) > self.max_items:
            self._lru.popitem(last=False)
            self._stats["evictions"] += 1

    def _disk_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        suffix = ".pt" if self._backend == "torch" else ".npz"
        return self.cache_dir / f"{digest}{suffix}"

    def _load_disk(self, key: str) -> _Entry | None:
        target = self._disk_path(key)
        if target is None or not target.exists():
            return None
        try:
            if self._backend == "torch":
                payload = self._torch_load(target)
                feat = payload["feat"]
                grid = payload.get("grid")
                signature = str(payload.get("signature", ""))
            else:
                with np.load(target, allow_pickle=False) as bundle:
                    feat = bundle["feat"]
                    grid = bundle["grid"].tolist()
                    signature = str(bundle["signature"].item())
            if signature and signature != self.namespace:
                raise FeatureError(f"盘缓存 signature {signature!r} 与当前口径不符")
            shape = _shape_of(feat)
            if len(shape) != 2:
                raise FeatureError(f"盘缓存 shape={shape} 不是 [T, D]")
        except (OSError, EOFError, KeyError, TypeError, ValueError,
                RuntimeError) as exc:
            # note (luojiaxuan): 盘缓存损坏/串味不是致命错误 —— 删掉重编码即可,
            # 但必须计数,disk_errors 长期非零说明有并发或口径问题要查。
            self._stats["disk_errors"] += 1
            _ = exc
            try:
                target.unlink()
            except OSError:
                pass
            return None
        self._stats["disk_hits"] += 1
        parsed = None if grid is None or int(grid[0]) < 0 else (int(grid[0]),
                                                               int(grid[1]))
        return _Entry(feat=feat, grid=parsed, nbytes=tensor_bytes(feat))

    def _torch_load(self, target: Path) -> Any:
        torch_mod = _torch_or_none()
        try:
            return torch_mod.load(target, map_location="cpu", weights_only=True)
        except TypeError:
            # note (luojiaxuan): 老 torch 没有 weights_only 形参。
            return torch_mod.load(target, map_location="cpu")

    def _write_disk(self, key: str, entry: _Entry) -> None:
        target = self._disk_path(key)
        if target is None:
            return
        grid = list(entry.grid) if entry.grid is not None else [-1, -1]
        tmp = target.with_name(f"{target.stem}.{os.getpid()}.tmp")
        try:
            if self._backend == "torch":
                _torch_or_none().save(
                    {"feat": entry.feat, "grid": grid,
                     "signature": self.namespace}, tmp)
            else:
                with open(tmp, "wb") as handle:
                    np.savez(handle, feat=entry.feat,
                             grid=np.asarray(grid, dtype=np.int64),
                             signature=np.asarray(self.namespace))
            tmp.replace(target)
        except (OSError, RuntimeError, ValueError):
            self._stats["disk_errors"] += 1
            try:
                tmp.unlink()
            except OSError:
                pass
            return
        self._stats["disk_writes"] += 1


def attach_to_bank(
    bank: HistoryFrameBank, cache: FeatureCache, *, replace: bool = False
) -> None:
    """把 cache 挂成 ``bank.feature_provider``(签名 ``(step, path) -> 特征``)。

    已挂了**别的** provider 时默认报错:两套特征混进同一条轨迹会静默错位,
    宁可 fail-loud。确实要换时传 ``replace=True``。
    """
    if not isinstance(bank, HistoryFrameBank):
        raise FeatureError(f"bank 必须是 HistoryFrameBank,收到 {type(bank).__name__}")
    if not callable(cache):
        raise FeatureError("cache 必须可调用((step, path) -> 特征)")
    existing = bank.feature_provider
    if existing is not None and existing is not cache and not replace:
        raise FeatureError(
            "bank 已挂了另一个 feature_provider;确认要替换请传 replace=True"
        )
    bank.feature_provider = cache


# ------------------------------------------------------------------ 工厂

def _pick(args_like: Any, *names: str, default: Any = None) -> Any:
    """从 argparse.Namespace / dataclass / dict 里按优先级取第一个非 None 值。"""
    for name in names:
        if isinstance(args_like, Mapping):
            value = args_like.get(name)
        else:
            value = getattr(args_like, name, None)
        if value is not None:
            return value
    return default


def build_extractor(args_like: Any) -> "FrozenFeatureExtractor | DummyFeatureExtractor":
    """``--dry-run`` → Dummy,否则真的加载冻结 GUI-Owl。

    识别的字段(Namespace / dict 均可):``dry_run``、``model_dir``、
    ``snapshot_manifest``、``device``、``visual_tokens``、``feature_dtype``
    (或 ``dtype``)、``dummy_dim``、``dummy_tokens``、``dummy_seed``(或 ``seed``)、
    ``feature_runtime``(复用已加载的 GUIOwlOSWorldRuntime)。
    """
    dtype = _pick(args_like, "feature_dtype", "dtype", default=DEFAULT_FEATURE_DTYPE)
    if bool(_pick(args_like, "dry_run", default=False)):
        return DummyFeatureExtractor(
            dim=int(_pick(args_like, "dummy_dim", default=DEFAULT_DUMMY_DIM)),
            tokens=int(_pick(args_like, "dummy_tokens",
                             default=DEFAULT_DUMMY_TOKENS)),
            seed=int(_pick(args_like, "dummy_seed", "seed", default=0)),
            dtype=str(dtype),
        )
    runtime = _pick(args_like, "feature_runtime")
    model_dir = _pick(args_like, "model_dir")
    manifest = _pick(args_like, "snapshot_manifest")
    if runtime is None and (model_dir is None or manifest is None):
        raise FeatureError(
            "非 dry-run 需要 model_dir 与 snapshot_manifest(或注入 feature_runtime)"
        )
    return FrozenFeatureExtractor(
        model_dir=model_dir,
        snapshot_manifest=manifest,
        device=str(_pick(args_like, "device", default="cuda:0")),
        visual_tokens=int(_pick(args_like, "visual_tokens",
                                default=DEFAULT_VISUAL_TOKENS)),
        dtype=str(dtype),
        runtime=runtime,
    )


def build_cache(
    args_like: Any, extractor: FeatureExtractor | None = None
) -> FeatureCache:
    """同一份 args 造出缓存;识别 ``feature_cache_dir``、``feature_cache_items``。"""
    chosen = extractor if extractor is not None else build_extractor(args_like)
    return FeatureCache(
        chosen,
        max_items=int(_pick(args_like, "feature_cache_items", "max_items",
                            default=DEFAULT_MAX_ITEMS)),
        cache_dir=_pick(args_like, "feature_cache_dir", "cache_dir"),
    )


__all__ = [
    "DEFAULT_DUMMY_DIM",
    "DEFAULT_DUMMY_TOKENS",
    "DEFAULT_FEATURE_DTYPE",
    "DEFAULT_MAX_ITEMS",
    "DEFAULT_VISUAL_TOKENS",
    "DummyFeatureExtractor",
    "FeatureCache",
    "FeatureError",
    "FeatureExtractor",
    "FrozenFeatureExtractor",
    "attach_to_bank",
    "backend",
    "build_cache",
    "build_extractor",
    "estimate_bytes",
    "tensor_bytes",
]
