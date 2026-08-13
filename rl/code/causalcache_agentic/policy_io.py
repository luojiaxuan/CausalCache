"""环境轨迹 ↔ GUI-Owl 输入的桥接层(Phase 0 统一接口的 policy-io 模块)。

# note (luojiaxuan): 路线见 rl/docs/agentic_memory_rl_roadmap_20260813.md,共享
# 契约见 causalcache_agentic/contract.py(本模块只 import,不修改)。本文件提供
# 三件东西,别的模块(rollout / GRPO trainer / evaluator)只依赖这里的公共 API:
#   * ``HistoryFrameBank``  —— 一条轨迹到目前为止的全部帧(step → 截图路径),
#     负责回答"现在能选哪些历史事件"与"事件号 → 帧路径";
#   * ``PolicyInputBuilder`` —— (instruction, 历史动作, 被选 subset) → 冻结
#     ``build_desktop_official_messages`` 的官方多轮 messages;
#   * ``SelectorPolicy``     —— selector 的最小接口 + 两个 Phase 3 基线臂
#     (``RecentBSelector`` / ``RandomBSelector``)。
#
# 编号口径(全模块唯一真相,写错就会静默错位,故在此写死):
#   * env 侧 ``StepRecord.step`` 是 **0-based 决策序号**,其 ``screenshot`` 是该步
#     **决策前**的观测。故帧索引 k 的图 == 第 k 步(0-based)决策前看到的屏幕;
#   * 官方 builder 侧步号是 **1-based**:当前步 s = current_step + 1,历史形态
#     ``steps`` 覆盖步骤 1..s-1,即 env 的 0-based 步 0..current_step-1;
#   * 官方"事件 j"的 post 帧 == 步骤 j+1(1-based)的决策前观测 == env 0-based 帧 j。
#     **因此事件号与 env 帧索引是同一个整数**,本模块不做任何偏移换算;
#   * 官方合约要求 shown_events ⊆ [1, s-2] = [1, current_step-1] —— 帧 0(初始屏)
#     不可作历史保留图(它被折叠进第一轮的 "Previous actions" 文本),当前帧
#     current_step 也不可(它恒在末轮出现)。
#
# 本模块**不含**、且按路线 §9 永久不得引入:pass-1 draft、B=1 逐帧 probe、
# gold 帧监督。selector 只从 terminal reward 学。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from causalcache.agentnet_desktop_official import (
    OfficialStepForms,
    build_desktop_official_messages,
    render_official_action_line,
    render_official_response,
)

if TYPE_CHECKING:  # pragma: no cover
    # note (luojiaxuan): torch 只作类型注解出现;本模块运行期不依赖它,基线臂用
    # numpy 打分,真正的参数化 selector(Phase 3)返回 torch.Tensor。
    import torch


DEFAULT_BUDGET = 2          # Primary budget B=2(路线 §0)
DEFAULT_TOOL_NAME = "computer_use"


class PolicyIOError(ValueError):
    """本模块统一的输入错误基类(全部继承 ValueError,调用方可只捕 ValueError)。"""


class SubsetError(PolicyIOError):
    """subset 非法:非严格递增、越界、基数不等于预算。"""


# ------------------------------------------------------------------ 帧仓库

@dataclass
class HistoryFrameBank:
    """一条轨迹到目前为止的全部帧:env 0-based 步号 → 截图路径。

    帧索引与官方事件号同值(见模块 docstring)。本类不读图、不渲染,只做索引与
    可选的特征缓存,故单测可以用不存在的假路径。
    """

    frames: dict[int, str] = field(default_factory=dict)
    feature_provider: Callable[[int, str], Any] | None = None
    _feature_cache: dict[int, Any] = field(default_factory=dict, repr=False)

    # -------------------------------------------------------------- 写入
    def add(self, step: int, path: str) -> None:
        """登记第 ``step`` 步决策前的截图路径。重复登记同路径幂等,改路径报错。"""
        key = _as_index(step, name="step")
        if key < 0:
            raise PolicyIOError(f"step 必须 ≥ 0,收到 {step!r}")
        if not isinstance(path, str) or not path:
            raise PolicyIOError(f"step={key} 的截图路径必须是非空字符串,收到 {path!r}")
        existing = self.frames.get(key)
        if existing is not None and existing != path:
            # note (luojiaxuan): fail-loud —— 同一步两个不同帧意味着 rollout 记账错乱,
            # 静默覆盖会让后续 subset 指向错误的历史内容。
            raise PolicyIOError(
                f"step={key} 已登记帧 {existing!r},不允许改写为 {path!r}"
            )
        self.frames[key] = path
        self._feature_cache.pop(key, None)

    def extend(self, paths: Sequence[str], *, start: int = 0) -> None:
        """批量登记连续帧(便于从已有 Trajectory 重建 bank)。"""
        for offset, path in enumerate(paths):
            self.add(start + offset, path)

    # -------------------------------------------------------------- 读取
    def has(self, step: int) -> bool:
        return _as_index(step, name="step") in self.frames

    def path(self, step: int) -> str:
        key = _as_index(step, name="step")
        try:
            return self.frames[key]
        except KeyError:
            raise PolicyIOError(
                f"bank 中没有 step={key} 的帧(已有 {sorted(self.frames)})"
            ) from None

    def candidates(self, now: int) -> list[int]:
        """``now``(0-based 当前决策序号)时刻可被选作历史保留图的事件号,升序。

        合法区间 [1, now-1]:帧 0 恒被折叠成文本,当前帧 ``now`` 恒在末轮。
        """
        cutoff = _as_index(now, name="now")
        return [j for j in range(1, cutoff) if j in self.frames]

    def recent(self, b: int, now: int) -> tuple[int, ...]:
        """recent-B 基线动作:最近的 ``b`` 个合法事件号,**升序**返回。

        候选不足 ``b`` 时返回全部候选(early step 的自然退化,不报错)。
        """
        budget = _as_index(b, name="b")
        cands = self.candidates(now)
        if budget == 0:
            return ()
        return tuple(cands[-budget:])

    def feature(self, step: int) -> Any:
        """Phase 3 冻结视觉特征缓存挂点(未接 provider 时显式 NotImplemented)。"""
        key = _as_index(step, name="step")
        if key in self._feature_cache:
            return self._feature_cache[key]
        if self.feature_provider is None:
            raise NotImplementedError(
                "HistoryFrameBank.feature 需要 feature_provider(Phase 3 接冻结视觉"
                "特征缓存时注入 Callable[[step, path], Any])"
            )
        value = self.feature_provider(key, self.path(key))
        self._feature_cache[key] = value
        return value

    def __len__(self) -> int:
        return len(self.frames)


def frames_for_subset(
    bank: HistoryFrameBank, subset: Sequence[int]
) -> dict[int, str]:
    """subset → ``build_desktop_official_messages`` 需要的 ``event_images`` 映射。"""
    return {_as_index(j, name="event"): bank.path(j) for j in subset}


# ------------------------------------------------------------------ 官方形态

# note (luojiaxuan): **动作别名归一** —— 冻结 GUI-Owl 的 tool spec 里 click 与
# left_click 都是合法枚举值,executor 也两者都认;但官方历史渲染器
# (render_official_action_line)只认规范名。实测:RL rollout 中策略偶尔输出
# {"action": "click"},历史渲染直接抛错 → 整个分片崩掉(2026-08-13 敏感性
# 跑挂了 2/4 分片)。策略输出是不可控的,归一必须在**进入历史之前**做,
# 且只做同义改名,不改变任何语义。
ACTION_ALIASES = {
    "click": "left_click",
    "drag": "left_click_drag",
    "leftclick": "left_click",
    "doubleclick": "double_click",
    "rightclick": "right_click",
    "middleclick": "middle_click",
}


def canonical_action(action: Mapping[str, Any]) -> dict[str, Any]:
    """把动作 dict 的 action 名归一到官方渲染器认识的枚举值(不改其它字段)。"""
    out = dict(action)
    name = str(out.get("action", "")).strip()
    key = name.lower().replace("-", "_")
    out["action"] = ACTION_ALIASES.get(key, ACTION_ALIASES.get(
        key.replace("_", ""), name))
    return out


def official_forms_from_actions(
    history_actions: Sequence[Mapping[str, Any]],
    *,
    tool_name: str = DEFAULT_TOOL_NAME,
) -> list[OfficialStepForms]:
    """env 的 computer_use 动作序列 → 步骤 1..n 的官方双形态。

    入参既接受裸 ``arguments``(``contract.StepRecord.action`` 的形态,如
    ``{"action": "left_click", "coordinate": [500, 300]}``),也接受已包好的
    tool call(``{"name": ..., "arguments": {...}}``)。**直接从 tool call 渲染**,
    不经 pyautogui ``raw_code`` 那条为 AgentNet 语料准备的老路。
    """
    forms: list[OfficialStepForms] = []
    for step_id, entry in enumerate(history_actions, start=1):
        call = _as_tool_call(entry, tool_name=tool_name, step_id=step_id)
        call = {**call, "arguments": canonical_action(call["arguments"])}
        try:
            action_line = render_official_action_line(call)
            full_response = render_official_response(call)
        except (KeyError, TypeError, ValueError) as exc:
            raise PolicyIOError(
                f"步骤 {step_id} 的动作无法渲染成官方形态({entry!r}): {exc}"
            ) from exc
        forms.append(
            OfficialStepForms(
                step_id=step_id,
                action_line=action_line,
                full_response=full_response,
                target_unavailable_reason=None,
            )
        )
    return forms


def _as_tool_call(
    entry: Mapping[str, Any], *, tool_name: str, step_id: int
) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise PolicyIOError(
            f"步骤 {step_id} 的动作必须是 mapping(computer_use arguments),"
            f"收到 {type(entry).__name__}"
        )
    if "arguments" in entry:
        arguments = entry["arguments"]
        if not isinstance(arguments, Mapping):
            raise PolicyIOError(
                f"步骤 {step_id} 的 arguments 必须是 mapping,收到 {arguments!r}"
            )
        name = entry.get("name") or tool_name
        return {"name": str(name), "arguments": dict(arguments)}
    if "action" in entry:
        return {"name": tool_name, "arguments": dict(entry)}
    raise PolicyIOError(
        f"步骤 {step_id} 的动作既没有 'arguments' 也没有 'action' 字段:{entry!r}"
    )


# ------------------------------------------------------------------ prompt 组装

@dataclass
class PolicyInputBuilder:
    """(轨迹前缀 + 被选 subset) → 冻结 GUI-Owl 的官方多轮 messages。

    ``budget`` 即 B。候选不足 B 的 early step 走 ``effective_budget``(= min(B,
    候选数))退化:此时 subset 必须恰好用满可用候选,仍然拒绝"能选却少选"。
    """

    budget: int = DEFAULT_BUDGET
    tool_name: str = DEFAULT_TOOL_NAME

    def effective_budget(self, n_candidates: int) -> int:
        return min(self.budget, max(0, n_candidates))

    def validate_subset(
        self, subset: Sequence[int], bank: HistoryFrameBank, current_step: int
    ) -> list[int]:
        """校验并规范化 subset,返回 int 列表。不合法即抛 ``SubsetError``。"""
        events = [_as_index(j, name="subset element") for j in subset]
        if any(b <= a for a, b in zip(events, events[1:])):
            raise SubsetError(f"subset 必须严格递增且无重复,收到 {tuple(events)}")
        cands = bank.candidates(current_step)
        want = self.effective_budget(len(cands))
        if len(events) != want:
            raise SubsetError(
                f"subset 基数必须为 {want}(budget={self.budget},current_step="
                f"{current_step} 时可选候选 {cands}),收到 {tuple(events)}"
            )
        allowed = set(cands)
        illegal = [j for j in events if j not in allowed]
        if illegal:
            raise SubsetError(
                f"subset 越界 {illegal}:current_step={current_step} 的合法事件号为 "
                f"{cands}(区间 [1, {current_step - 1}] 且帧已入 bank)"
            )
        return events

    def build(
        self,
        task_instruction: str,
        history_actions: Sequence[Mapping[str, Any]],
        subset: Sequence[int],
        bank: HistoryFrameBank,
        current_step: int,
    ) -> list[dict[str, Any]]:
        """组装 messages。图片数恒为 ``len(subset) + 1``(被选事件帧 + 当前帧)。

        ``history_actions``:env 0-based 步 0..current_step-1 的 computer_use 动作,
        长度必须恰为 ``current_step``。
        """
        now = _as_index(current_step, name="current_step")
        if not isinstance(task_instruction, str) or not task_instruction.strip():
            raise PolicyIOError("task_instruction 必须是非空文本")
        actions = list(history_actions)
        if len(actions) != now:
            raise PolicyIOError(
                f"history_actions 长度 {len(actions)} 必须等于 current_step={now}"
                "(即步骤 1..current_step 的历史动作)"
            )
        if not bank.has(now):
            raise PolicyIOError(
                f"bank 中缺少当前帧 step={now}(已有 {sorted(bank.frames)})"
            )
        events = self.validate_subset(subset, bank, now)
        steps = official_forms_from_actions(actions, tool_name=self.tool_name)
        return build_desktop_official_messages(
            goal=task_instruction,
            steps=steps,
            shown_events=events,
            event_images=frames_for_subset(bank, events),
            current_image=bank.path(now),
        )


# ------------------------------------------------------------------ selector 接口

@dataclass(frozen=True)
class SelectorState:
    """``state_repr`` 的最小可选载体(Protocol 对 state_repr 本身是不透明的)。

    基线臂只在 ``score_subsets`` 里读 ``candidates``(用于确定 recent 参照系);
    Phase 3 的参数化 selector 可以换成携带视觉特征的自定义对象。
    """

    task_instruction: str
    current_step: int
    candidates: tuple[int, ...]
    history_actions: tuple[Mapping[str, Any], ...] = ()
    bank: HistoryFrameBank | None = None

    @classmethod
    def from_bank(
        cls,
        *,
        task_instruction: str,
        history_actions: Sequence[Mapping[str, Any]],
        bank: HistoryFrameBank,
        current_step: int,
    ) -> "SelectorState":
        now = _as_index(current_step, name="current_step")
        return cls(
            task_instruction=task_instruction,
            current_step=now,
            candidates=tuple(bank.candidates(now)),
            history_actions=tuple(history_actions),
            bank=bank,
        )


@runtime_checkable
class SelectorPolicy(Protocol):
    """固定预算 B 的记忆选择器接口。

    路线 §4 硬约束:接口必须是 ``z(S | state), |S| = B`` —— 对**整个 subset**
    打分,不得退化成 ``score_pair(f1, f2)`` 这类写死基数的实现。
    """

    def score_subsets(
        self, state_repr: Any, subsets: list[tuple[int, ...]]
    ) -> "torch.Tensor":
        """对候选 subset 批量打分(未归一 logits,顺序与入参一致)。"""
        ...

    def sample(
        self, state_repr: Any, candidates: Sequence[int], b: int
    ) -> tuple[tuple[int, ...], float]:
        """训练期采样:返回(升序 subset, 该 subset 的 log-prob)。"""
        ...

    def argmax(
        self, state_repr: Any, candidates: Sequence[int], b: int
    ) -> tuple[int, ...]:
        """评测期确定性动作:返回升序 subset。"""
        ...


@dataclass
class RecentBSelector:
    """基线臂 A:恒选 recent-B(点质量策略,log-prob = 0.0)。

    这是路线里"必须被超过"的对照。``score_subsets`` 返回点质量的 log-prob:
    recent 子集 0.0,其余 -inf(softmax 后即 one-hot)。
    """

    def argmax(
        self, state_repr: Any, candidates: Sequence[int], b: int
    ) -> tuple[int, ...]:
        return _recent_subset(candidates, b)

    def sample(
        self, state_repr: Any, candidates: Sequence[int], b: int
    ) -> tuple[tuple[int, ...], float]:
        return self.argmax(state_repr, candidates, b), 0.0

    def score_subsets(
        self, state_repr: Any, subsets: list[tuple[int, ...]]
    ) -> np.ndarray:
        # note (luojiaxuan): 基线臂无参数、不进反传,故用 numpy 而不引入 torch 依赖;
        # 下游 rollout 只用到 argmax / 索引,两者行为一致。
        rows = [tuple(int(j) for j in s) for s in subsets]
        if not rows:
            return np.zeros((0,), dtype=np.float32)
        pool = _candidate_pool(state_repr, rows)
        scores = np.full((len(rows),), -np.inf, dtype=np.float32)
        for index, row in enumerate(rows):
            if row == _recent_subset(pool, len(row)):
                scores[index] = 0.0
        return scores


@dataclass
class RandomBSelector:
    """基线臂 B:在全部 C(n, B) 个 subset 上均匀采样(Phase 6 的 sanity 臂)。

    ``seed`` 固定 → 采样序列可复现;``reset()`` 回到初始 seed。
    """

    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self._rng = random.Random(self.seed)

    def sample(
        self, state_repr: Any, candidates: Sequence[int], b: int
    ) -> tuple[tuple[int, ...], float]:
        pool = _normalize_candidates(candidates)
        k = min(_as_index(b, name="b"), len(pool))
        if k == 0:
            return (), 0.0
        picked = tuple(sorted(self._rng.sample(pool, k)))
        # note (luojiaxuan): random.sample 在 k-子集上均匀,故 log-prob 恒为
        # -log C(n, k);n == k 时只有一个子集,log-prob = 0。
        return picked, -math.log(math.comb(len(pool), k))

    def argmax(
        self, state_repr: Any, candidates: Sequence[int], b: int
    ) -> tuple[int, ...]:
        # note (luojiaxuan): 均匀分布没有众数;这里取字典序最小的子集作确定性
        # tie-break,而**不是** recent-B —— 否则 random 臂在评测期会悄悄退化成
        # recent 臂,让 sanity 对照失效。random 臂应当用 sample() 评测。
        pool = _normalize_candidates(candidates)
        k = min(_as_index(b, name="b"), len(pool))
        return tuple(pool[:k])

    def score_subsets(
        self, state_repr: Any, subsets: list[tuple[int, ...]]
    ) -> np.ndarray:
        # note (luojiaxuan): 均匀策略的 logits 恒等;归一后的 log-prob 依赖候选集
        # 大小,由 sample() 精确给出。
        return np.zeros((len(subsets),), dtype=np.float32)


# ------------------------------------------------------------------ 内部工具

def _as_index(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise PolicyIOError(f"{name} 必须是整数,收到 {value!r}")
    return int(value)


def _normalize_candidates(candidates: Sequence[int]) -> list[int]:
    pool = [_as_index(j, name="candidate") for j in candidates]
    if any(b <= a for a, b in zip(pool, pool[1:])):
        raise SubsetError(f"candidates 必须严格递增且无重复,收到 {tuple(pool)}")
    return pool


def _recent_subset(candidates: Sequence[int], b: int) -> tuple[int, ...]:
    pool = _normalize_candidates(candidates)
    k = min(_as_index(b, name="b"), len(pool))
    if k == 0:
        return ()
    return tuple(pool[-k:])


def _candidate_pool(state_repr: Any, rows: Sequence[tuple[int, ...]]) -> list[int]:
    """recent 参照系:优先用 state 携带的候选集,否则退化成给定 subset 的并集。"""
    raw: Any = None
    if isinstance(state_repr, Mapping):
        raw = state_repr.get("candidates")
    else:
        raw = getattr(state_repr, "candidates", None)
    if raw is not None:
        return _normalize_candidates(list(raw))
    return sorted({int(j) for row in rows for j in row})


__all__ = [
    "DEFAULT_BUDGET",
    "DEFAULT_TOOL_NAME",
    "HistoryFrameBank",
    "OfficialStepForms",
    "PolicyIOError",
    "PolicyInputBuilder",
    "RandomBSelector",
    "RecentBSelector",
    "SelectorPolicy",
    "SelectorState",
    "SubsetError",
    "frames_for_subset",
    "official_forms_from_actions",
]
