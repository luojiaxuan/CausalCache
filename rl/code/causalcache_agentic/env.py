"""程序化 GUI 环境:reset / step / render / verify + 专家轨迹与吞吐基准。

# note (luojiaxuan): 本模块是 contract.py 的**使用方**,不定义新状态语义。
# 三条纪律:
#   1. **深拷贝**:reset 深拷贝 TaskSpec.initial_state,同一 TaskSpec 可重复
#      reset,两次 reset 得到的 state 必须相等(dataclass 递归 __eq__);
#   2. **step/render 内无随机、无时间依赖**:随机只允许在 task 实例化阶段;
#   3. **依赖惰性解析**:render.py / executor.py 在首次使用时才 import,
#      并允许构造期注入(便于单测与替换实现,见决策 E1/E2)。
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .contract import (
    ScreenState,
    StepRecord,
    TaskSpec,
    Trajectory,
    verify as verify_assertions,
)

# note (luojiaxuan): 终止类动作(policy 主动结束)。动作**照常下发给 executor**
# (它才知道 answer 文本落哪个 flag);env 只做兜底:执行完仍未 terminated 就强制
# 置位,保证换任何 executor 实现都不会跑飞。
TERMINAL_ACTIONS = ("terminate", "answer")

# note (luojiaxuan): 与 executor.py 的 ERROR_KEY / ANSWER_KEY 同名兜底常量 ——
# 优先从实际 executor 对象读同名属性,读不到才用这里的字面量。
ERROR_FLAG_KEY = "__last_error__"
ANSWER_FLAG_KEY = "__answer__"

ApplyFn = Callable[[ScreenState, dict[str, Any]], ScreenState]
SubsetBuilder = Callable[[int, "list[StepRecord]"], Sequence[int]]

_ACTION_LINE_FN: Callable[[dict[str, Any]], str] | None = None


# ---------------------------------------------------------------- 动作规范化

def normalize_action(action: Any) -> dict[str, Any]:
    """把 {"name": "computer_use", "arguments": {...}} 或裸 arguments 统一为 arguments。"""
    if not isinstance(action, dict):
        raise TypeError(f"action 必须是 dict,收到 {type(action).__name__}")
    if "action" not in action and isinstance(action.get("arguments"), dict):
        action = action["arguments"]
    if "action" not in action:
        raise ValueError(f"action 缺少 'action' 字段: {sorted(action)}")
    return action


def as_tool_call(action: dict[str, Any]) -> dict[str, Any]:
    """arguments → 官方 tool_call 形态(render_official_action_line 的入参口径)。"""
    return {"name": "computer_use", "arguments": normalize_action(action)}


def action_line(action: dict[str, Any]) -> str:
    """官方历史文本的 ``Action:`` 行(复用冻结渲染器,保证与语料逐字同源)。"""
    global _ACTION_LINE_FN
    if _ACTION_LINE_FN is None:
        from causalcache.agentnet_desktop_official import render_official_action_line

        _ACTION_LINE_FN = render_official_action_line
    return _ACTION_LINE_FN(as_tool_call(action))


def is_terminal_action(action: dict[str, Any]) -> bool:
    return normalize_action(action)["action"] in TERMINAL_ACTIONS


# ---------------------------------------------------------------- 依赖解析

def _executor_module() -> Any:
    from . import executor as executor_module

    return executor_module


def _render_module() -> Any:
    from . import render as render_module

    return render_module


def _resolve_apply(obj: Any = None) -> ApplyFn:
    """解析 executor:模块级 apply / apply_action / Executor 类 / 裸可调用皆可。"""
    if obj is None:
        obj = _executor_module()
    for name in ("apply", "apply_action"):
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn
    for name in ("GUIExecutor", "Executor"):
        cls = getattr(obj, name, None)
        if cls is not None:
            return cls().apply
    if callable(obj):
        return obj
    raise TypeError(f"无法从 {obj!r} 解析出 apply(state, action) 接口")


@dataclass
class _RenderFns:
    render: Callable[[ScreenState], Any]
    render_to_file: Callable[[ScreenState, str], Any]


def _resolve_render(obj: Any = None) -> _RenderFns:
    """解析 renderer:需要 render(state);render_to_file 缺失时退化为 img.save。"""
    if obj is None:
        obj = _render_module()
    fn = getattr(obj, "render", None)
    if not callable(fn):
        if callable(obj):
            fn = obj
        else:
            raise TypeError(f"无法从 {obj!r} 解析出 render(state) 接口")
    to_file = getattr(obj, "render_to_file", None)
    if not callable(to_file):
        def to_file(state: ScreenState, path: str, _fn=fn) -> Any:
            image = _fn(state)
            image.save(path)
            return image

    return _RenderFns(render=fn, render_to_file=to_file)


# ---------------------------------------------------------------- 环境

class GUIEnv:
    """单实例 GUI 环境。**不持有任何跨 episode 的隐藏状态**(除计数器外)。

    参数
    ----
    executor / renderer
        可选注入;默认惰性 import 同包 ``executor`` / ``render`` 模块。
    auto_stop_on_success
        默认 False —— 训练时必须由 policy 自己 terminate;数据生成脚本可置 True,
        让专家动作序列跑完断言成立即停。
    """

    def __init__(
        self,
        executor: Any = None,
        renderer: Any = None,
        auto_stop_on_success: bool = False,
    ) -> None:
        self._executor_obj = executor
        self._renderer_obj = renderer
        self.auto_stop_on_success = auto_stop_on_success
        self._apply: ApplyFn | None = None
        self._render_fns: _RenderFns | None = None
        self._error_key = ERROR_FLAG_KEY
        self._answer_key = ANSWER_FLAG_KEY
        self._task: TaskSpec | None = None
        self._state: ScreenState | None = None
        self._steps = 0
        self._done = False
        self._reason = ""
        self._last_error = ""

    # ------------------------------------------------------------ 属性
    @property
    def task(self) -> TaskSpec:
        if self._task is None:
            raise RuntimeError("必须先 reset(task)")
        return self._task

    @property
    def state(self) -> ScreenState:
        if self._state is None:
            raise RuntimeError("必须先 reset(task)")
        return self._state

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def done(self) -> bool:
        return self._done

    @property
    def reason(self) -> str:
        return self._reason

    # ------------------------------------------------------------ 生命周期
    def reset(self, task: TaskSpec) -> ScreenState:
        """深拷贝初始状态开局。同一 TaskSpec 可反复 reset,结果逐字段相等。"""
        self._task = task
        self._state = copy.deepcopy(task.initial_state)
        self._steps = 0
        self._done = False
        self._reason = ""
        self._last_error = ""
        return self._state

    def step(self, action: dict[str, Any]) -> tuple[ScreenState, bool, dict[str, Any]]:
        """执行一个 computer_use 动作,返回 (obs, done, info)。

        info = {"step", "success", "last_error", "active_app", "reason", "terminated"}。
        错误有两条通道,都汇总进 ``last_error``:executor 抛异常(状态保持不变),
        或 executor 降级成 no-op 并写 ``flags[ERROR_KEY]``。两种情况都**不提前判死**、
        步数照常推进(单步失败不等于任务失败,policy 还有机会补救)。
        """
        if self._task is None or self._state is None:
            raise RuntimeError("必须先 reset(task)")
        if self._done:
            return self._state, True, self._info()

        args = normalize_action(action)
        self._last_error = ""
        self._steps += 1

        try:
            self._state = self._apply_fn()(self._state, args)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
        flagged = self._state.flags.get(self._error_key, "")
        if flagged and not self._last_error:
            self._last_error = str(flagged)

        if args["action"] in TERMINAL_ACTIONS and not self._state.terminated:
            # note (luojiaxuan): executor 没实现终止语义时的兜底,保证 done 一定成立。
            self._state.terminated = True
            if args["action"] == "answer":
                self._state.terminate_status = "success"
                self._state.flags.setdefault(self._answer_key,
                                             str(args.get("text", "")))
            else:
                self._state.terminate_status = str(args.get("status", "success"))
        if self._state.terminated:
            self._done = True
            self._reason = "terminated"

        self._state.step = self._steps
        if not self._done and self.auto_stop_on_success and self.verify():
            self._done = True
            self._reason = "success"
        if not self._done and self._steps >= self._task.max_steps:
            self._done = True
            self._reason = "max_steps"
        return self._state, self._done, self._info()

    def render(self, path: str | None = None) -> Any:
        """渲染当前状态。path 为空返回 PIL Image,否则写 PNG 并返回该路径。"""
        fns = self._render_fns_or_resolve()
        if path is None:
            return fns.render(self.state)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        fns.render_to_file(self.state, path)
        return path

    def verify(self) -> bool:
        """终局判定(声明式断言,无模型参与)。任意时刻可调用。"""
        return verify_assertions(self.state, self.task.assertions)

    # ------------------------------------------------------------ 内部
    def _apply_fn(self) -> ApplyFn:
        if self._apply is None:
            obj = self._executor_obj
            if obj is None:
                obj = self._executor_obj = _executor_module()
            self._error_key = str(getattr(obj, "ERROR_KEY", ERROR_FLAG_KEY))
            self._answer_key = str(getattr(obj, "ANSWER_KEY", ANSWER_FLAG_KEY))
            self._apply = _resolve_apply(obj)
        return self._apply

    def _render_fns_or_resolve(self) -> _RenderFns:
        if self._render_fns is None:
            self._render_fns = _resolve_render(self._renderer_obj)
        return self._render_fns

    def _info(self) -> dict[str, Any]:
        state = self.state
        return {
            "step": self._steps,
            "success": self.verify(),
            "last_error": self._last_error,
            "active_app": state.active_app,
            "reason": self._reason,
            "terminated": state.terminated,
        }


# ---------------------------------------------------------------- 无渲染回放

@dataclass
class ReplayResult:
    """replay 的详细结果(供 expert.py 定位失败到底出在 tasks 还是 executor)。"""

    success: bool
    state: ScreenState
    steps_executed: int
    reason: str = ""
    last_error: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)


def replay_detailed(
    task: TaskSpec,
    actions: Sequence[dict[str, Any]],
    *,
    executor: Any = None,
    with_trace: bool = False,
) -> ReplayResult:
    """无渲染回放并保留每步诊断。with_trace=True 时记录 flags 增量(仅诊断用)。"""
    env = GUIEnv(executor=executor)
    env.reset(task)
    trace: list[dict[str, Any]] = []
    executed = 0
    last_error = ""
    for act in actions:
        before = dict(env.state.flags) if with_trace else {}
        _obs, done, info = env.step(act)
        executed += 1
        if info["last_error"]:
            last_error = info["last_error"]
        if with_trace:
            after = env.state.flags
            delta = {k: v for k, v in after.items() if before.get(k) != v}
            dropped = [k for k in before if k not in after]
            args = normalize_action(act)
            trace.append({
                "step": executed - 1,
                "action": args.get("action"),
                "coordinate": list(args.get("coordinate", ())) or None,
                "text": args.get("text"),
                "keys": list(args.get("keys", ())) or None,
                "active_app": info["active_app"],
                "focus": list(env.state.focus) if env.state.focus else None,
                "flags_delta": delta,
                "flags_dropped": dropped,
                "error": info["last_error"],
            })
        if done:
            break
    reason = env.reason or ("actions_exhausted" if not env.done else "")
    return ReplayResult(
        success=env.verify(), state=env.state, steps_executed=executed,
        reason=reason, last_error=last_error, trace=trace,
    )


def replay(task: TaskSpec, actions: list[dict[str, Any]], *, executor: Any = None
           ) -> tuple[bool, ScreenState]:
    """**不渲染**的快速回放,用于 tasks 自检与 CI(吞吐关键路径)。"""
    res = replay_detailed(task, actions, executor=executor)
    return res.success, res.state


# ---------------------------------------------------------------- 专家轨迹

def _recent_subset(step_index: int, budget: int = 2) -> tuple[int, ...]:
    """recent-B:取当前决策步之前最近 B 个**合法**历史帧步号(升序)。

    # note (luojiaxuan): 合法区间是 [1, step_index-1],与
    # ``policy_io.HistoryFrameBank.candidates`` 逐字同源 —— 帧 0(初始屏)恒被
    # 折叠进首轮 "Previous actions" 文本,选中它组不出合法官方 prompt。早期步
    # 候选不足 B 时自然退化成更少的帧,不报错。
    """
    lo = max(1, step_index - budget)
    return tuple(range(lo, max(lo, step_index)))


def rollout_expert(
    task: TaskSpec,
    env: GUIEnv,
    shot_dir: str,
    builder: SubsetBuilder | Any = None,
) -> Trajectory:
    """按 task.expert_actions 逐步执行,**每步决策前**渲染一张 PNG。

    截图命名 ``{task_id}_step{k:02d}.png``(k = 0-based 决策序号),
    与 StepRecord.step 一一对应,便于 SFT / RL 侧按步号索引历史帧。
    builder 可选:``builder(step_index, records_so_far) -> Sequence[int]``,
    或带 ``select_subset`` 同签名方法;缺省用 recent-2。
    """
    os.makedirs(shot_dir, exist_ok=True)
    select = None
    if builder is not None:
        select = builder if callable(builder) else getattr(builder, "select_subset", None)
        if select is None:
            raise TypeError("builder 需可调用或提供 select_subset(step, records)")

    env.reset(task)
    records: list[StepRecord] = []
    done = False
    info: dict[str, Any] = {}
    for k, raw in enumerate(task.expert_actions):
        if done:
            break
        shot = os.path.join(shot_dir, f"{task.task_id}_step{k:02d}.png")
        env.render(shot)
        args = normalize_action(raw)
        try:
            line = action_line(args)
        except ValueError as exc:
            raise ValueError(
                f"task {task.task_id} step {k} 的动作无法渲染为官方历史行: {exc}"
            ) from exc
        subset = tuple(select(k, records)) if select is not None else _recent_subset(k)
        _obs, done, info = env.step(args)
        records.append(StepRecord(
            step=k, screenshot=shot, action=dict(args), action_line=line,
            source="expert", shown_subset=subset,
            info={"active_app": info["active_app"], "last_error": info["last_error"]},
        ))

    success = env.verify()
    reason = env.reason or "actions_exhausted"
    if info.get("last_error"):
        reason = f"{reason}+error"
    return Trajectory(
        task_id=task.task_id, template_id=task.template_id, family=task.family,
        regime=task.regime, seed=task.seed, steps=records, success=success,
        reason=reason, final_flags=dict(env.state.flags),
    )


# ---------------------------------------------------------------- 吞吐基准

def benchmark(tasks: list[TaskSpec], with_render: bool, *, executor: Any = None,
              renderer: Any = None) -> dict[str, Any]:
    """跑完给定任务的专家动作,测 steps/s 与 tasks/s。

    with_render=True 时每步决策前渲染一帧到内存(不落盘,隔离磁盘 I/O 噪声)。
    """
    env = GUIEnv(executor=executor, renderer=renderer)
    n_steps = 0
    n_success = 0
    t0 = time.perf_counter()
    for task in tasks:
        env.reset(task)
        for raw in task.expert_actions:
            if with_render:
                env.render(None)
            _obs, done, _info = env.step(raw)
            n_steps += 1
            if done:
                break
        if env.verify():
            n_success += 1
    elapsed = max(time.perf_counter() - t0, 1e-9)
    return {
        "n_tasks": len(tasks),
        "n_steps": n_steps,
        "n_success": n_success,
        "elapsed_sec": elapsed,
        "steps_per_sec": n_steps / elapsed,
        "tasks_per_sec": len(tasks) / elapsed,
        "with_render": with_render,
    }
