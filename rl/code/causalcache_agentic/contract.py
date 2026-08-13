"""Agentic memory RL —— 全线接口契约(Phase 0 冻结,实现方不得修改本文件)。

# note (luojiaxuan): 路线见 rl/docs/agentic_memory_rl_roadmap_20260813.md。
# 本文件是 renderer / executor / apps / tasks / expert / policy-io 六个模块的
# **唯一共享契约**:状态模型 + 效果词表 + 动作口径 + 数据 schema。设计原则:
#   * **状态模型即契约**:widget 携带声明式 effect,executor 通用解释,
#     app 语义声明化 —— 三者可并行实现、独立替换(决策 E2);
#   * **渲染是纯函数** ScreenState → Image:同 state 必须逐位相同(决策 E1);
#   * **动作口径复用冻结路径**:policy 输出 [0,999] 归一 computer_use 调用,
#     env 在此单点换算到 1920×1080 像素(决策 E3);
#   * **step 内禁止任何随机**:随机只允许出现在 task 实例化(带 seed)阶段。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

# ---------------------------------------------------------------- 屏幕与坐标

SCREEN_W = 1920
SCREEN_H = 1080
NORM_MAX = 999          # policy 侧归一坐标上界([0,999],与冻结语料同口径)
TASKBAR_H = 48          # 屏幕底部任务栏高度(切应用的可点区域)


def norm_to_pixel(x: int, y: int) -> tuple[int, int]:
    """[0,999] 归一 → 像素。与 _norm999_from_pixel 互为逆(取整误差 ≤1px)。"""
    px = round(float(x) / NORM_MAX * (SCREEN_W - 1))
    py = round(float(y) / NORM_MAX * (SCREEN_H - 1))
    return max(0, min(SCREEN_W - 1, px)), max(0, min(SCREEN_H - 1, py))


def pixel_to_norm(px: int, py: int) -> tuple[int, int]:
    x = round(float(px) / max(SCREEN_W - 1, 1) * NORM_MAX)
    y = round(float(py) / max(SCREEN_H - 1, 1) * NORM_MAX)
    return max(0, min(NORM_MAX, x)), max(0, min(NORM_MAX, y))


# ---------------------------------------------------------------- 状态模型

@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


# effect 词表(executor 必须全部实现;新增 effect 只加解释分支,不改已有语义)
#   set_value      target=widget_id  value=文本            —— 直接写 widget.value
#   set_text       target=widget_id  value=文本            —— 改 widget.text(显示名)
#   set_visible    target=widget_id  value="0"/"1"
#   set_enabled    target=widget_id  value="0"/"1"
#   focus_app      target=app_id                            —— 置顶窗口(切应用)
#   open_app       target=app_id                            —— 打开并置顶
#   close_app      target=app_id
#   open_dialog    target=app_id     payload={"dialog": id} —— 显示同 app 内 dialog 组
#   close_dialog   target=app_id     payload={"dialog": id}
#   set_flag       target=flag_key   value=字面量           —— 写 state.flags(verifier 读)
#   flag_from      target=flag_key   payload={"widget": id} —— flags[key] ← 该 widget.value
#   copy_widget    payload={"widget": id}                   —— clipboard ← 该 widget.value
#   paste_into     target=widget_id                         —— widget.value ← clipboard
#   append_value   target=widget_id  value=后缀
#   clear_value    target=widget_id
#   scroll_to      target=widget_id                         —— 令其所在滚动区滚到可见
#   switch_tab     target=app_id     payload={"tab": tab_id}—— 同 app 内切页(改可见组)
#   noop
@dataclass(frozen=True)
class Effect:
    kind: str
    target: str = ""
    value: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Widget:
    """一个可见/可点控件。rect 为**屏幕绝对像素**(窗口内布局在构造期算好)。

    kind ∈ {button, label, text_field, text_area, list_item, cell, tab,
            menu_item, icon, checkbox, title, badge, divider}
    group:同 app 内的可见性分组(如 dialog id / tab id);None = 常驻。
    meta:任务级标注(如 {"carries": "order_id"}),仅供诊断与数据生成,
         **不得进入 policy 输入,也不得作 RL reward**。
    """

    wid: str
    kind: str
    rect: Rect
    text: str = ""
    value: str = ""
    enabled: bool = True
    visible: bool = True
    group: str | None = None
    style: dict[str, Any] = field(default_factory=dict)
    on_click: tuple[Effect, ...] = ()
    on_double_click: tuple[Effect, ...] = ()
    on_enter: tuple[Effect, ...] = ()          # 焦点在此 widget 时按 Return
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Window:
    app_id: str
    title: str
    rect: Rect
    widgets: list[Widget]
    open: bool = True
    scroll_y: int = 0
    scroll_area: Rect | None = None            # 可滚动区域(None = 不可滚)
    scroll_max: int = 0
    active_groups: tuple[str, ...] = ()        # 当前可见的 group(dialog/tab)


@dataclass
class ScreenState:
    """整屏状态。**渲染与执行都只依赖它**,不得有隐藏全局态。"""

    windows: dict[str, Window]
    z_order: list[str]                          # 底→顶,末位 = 活动窗口
    flags: dict[str, str] = field(default_factory=dict)
    clipboard: str = ""
    focus: tuple[str, str] | None = None        # (app_id, widget_id)
    select_all: bool = False                    # ctrl+a 后下一次 type 覆盖
    step: int = 0
    terminated: bool = False
    terminate_status: str = ""

    @property
    def active_app(self) -> str | None:
        return self.z_order[-1] if self.z_order else None


# ---------------------------------------------------------------- 任务与验收

MEMORY_REGIMES = (
    "recent_sufficient",
    "one_old_frame",
    "two_frame_complementary",
    "distractor_heavy",
    "history_irrelevant",
)


@dataclass(frozen=True)
class Assertion:
    """终局 verifier 的一条声明式断言(禁止用自然语言或 LLM 判定)。

    kind ∈ {flag_equals, flag_contains, widget_value_equals,
            widget_value_contains, flag_absent}
    """

    kind: str
    target: str
    expected: str = ""


@dataclass
class TaskSpec:
    task_id: str
    template_id: str                     # workflow family(划分单位)
    family: str                          # 更粗的组:用于 template-level split
    regime: str                          # MEMORY_REGIMES 之一
    seed: int
    instruction: str
    initial_state: ScreenState
    assertions: tuple[Assertion, ...]
    expert_actions: tuple[dict[str, Any], ...]   # computer_use arguments 序列
    max_steps: int = 24
    # 诊断用(禁止进 policy 输入 / 禁止作 reward):
    memory_probe: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


def verify(state: ScreenState, assertions: Sequence[Assertion]) -> bool:
    """终局判定:全部断言成立才算成功。**无歧义、无模型参与**。"""
    for a in assertions:
        if a.kind == "flag_equals":
            if state.flags.get(a.target, "") != a.expected:
                return False
        elif a.kind == "flag_contains":
            if a.expected not in state.flags.get(a.target, ""):
                return False
        elif a.kind == "flag_absent":
            if a.target in state.flags:
                return False
        elif a.kind in ("widget_value_equals", "widget_value_contains"):
            found = None
            for win in state.windows.values():
                for w in win.widgets:
                    if w.wid == a.target:
                        found = w.value
                        break
            if found is None:
                return False
            if a.kind == "widget_value_equals" and found != a.expected:
                return False
            if a.kind == "widget_value_contains" and a.expected not in found:
                return False
        else:
            raise ValueError(f"未知断言类型 {a.kind!r}")
    return True


# ---------------------------------------------------------------- 轨迹 schema

@dataclass
class StepRecord:
    step: int                                    # 0-based 决策序号
    screenshot: str                              # 该步决策前观测的 PNG 路径
    action: dict[str, Any]                       # computer_use arguments
    action_line: str                             # 官方历史文本(Action: ...)
    source: str = "expert"                       # expert | policy
    shown_subset: tuple[int, ...] = ()           # 该步喂给 policy 的历史帧步号
    selector_logprob: float | None = None
    policy_logprob: float | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    task_id: str
    template_id: str
    family: str
    regime: str
    seed: int
    steps: list[StepRecord]
    success: bool = False
    reason: str = ""                             # terminated | max_steps | error
    final_flags: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "task_id": self.task_id, "template_id": self.template_id,
            "family": self.family, "regime": self.regime, "seed": self.seed,
            "success": self.success, "reason": self.reason,
            "final_flags": self.final_flags,
            "steps": [{"step": s.step, "screenshot": s.screenshot,
                       "action": s.action, "action_line": s.action_line,
                       "source": s.source, "shown_subset": list(s.shown_subset),
                       "selector_logprob": s.selector_logprob,
                       "policy_logprob": s.policy_logprob, "info": s.info}
                      for s in self.steps]}, ensure_ascii=False)


# ---------------------------------------------------------------- 模块 Protocol

class Renderer(Protocol):
    """render.py 提供。必须确定性:同 state 两次渲染字节相同。"""

    def render(self, state: ScreenState) -> Any:  # -> PIL.Image.Image (RGB 1920x1080)
        ...


class Executor(Protocol):
    """executor.py 提供。纯函数式:返回**新** state,不原地改入参。"""

    def apply(self, state: ScreenState, action: dict[str, Any]) -> ScreenState:
        ...


class TaskTemplate(Protocol):
    """tasks.py 中每个 workflow family 的实例化器。"""

    template_id: str
    family: str

    def build(self, seed: int, regime: str) -> TaskSpec:
        ...


# apps.py 必须提供的窗口构造器(tasks.py 只依赖这些签名):
#   build_files_app(spec)   spec={"files":[{"name","size","modified","content"}...],
#                                 "path": str}
#   build_writer_app(spec)  spec={"title","paragraphs":[str],"editable_id":str}
#   build_calc_app(spec)    spec={"title","rows":[[str]],"headers":[str]}
#   build_browser_app(spec) spec={"url","title","lines":[str],"links":[{"text","url"}]}
#   build_mail_app(spec)    spec={"folder","messages":[{"from","subject","body"}],
#                                 "compose":bool}
#   build_settings_app(spec) spec={"sections":[{"name","items":[{"label","value"}]}]}
# 每个构造器返回 Window;窗口矩形由 apps.WINDOW_RECT 统一(全屏减任务栏),
# 控件 rect 为屏幕绝对像素;需要滚动的长列表设置 window.scroll_area/scroll_max。
# 每个构造器必须支持 spec["app_id"] 覆盖默认 app_id,以便同 app 多实例。

APP_BUILDERS = (
    "build_files_app", "build_writer_app", "build_calc_app",
    "build_browser_app", "build_mail_app", "build_settings_app",
)

# 任务 DSL primitive(tasks.py 实现;每个 primitive 产出
# (状态变更, 专家动作序列, 断言, memory_probe 片段)):
TASK_PRIMITIVES = (
    "READ", "WRITE", "COPY", "COMPARE", "TRANSFORM", "SEARCH",
    "SAVE", "SEND", "SWITCH_APP", "DISTRACT", "VERIFY",
)


def make_taskbar(app_ids: Sequence[str], titles: Sequence[str]) -> list[Widget]:
    """底部任务栏:每个 app 一个按钮,点击 focus_app。apps.py 与 tasks.py 共用。"""
    out: list[Widget] = []
    slot = 240
    for i, (aid, title) in enumerate(zip(app_ids, titles)):
        out.append(Widget(
            wid=f"taskbar::{aid}", kind="button",
            rect=Rect(8 + i * slot, SCREEN_H - TASKBAR_H + 6, slot - 12,
                      TASKBAR_H - 12),
            text=title, group=None,
            on_click=(Effect("focus_app", target=aid),),
            style={"role": "taskbar"}))
    return out
