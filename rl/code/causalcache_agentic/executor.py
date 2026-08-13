"""GUI 动作执行器 —— 纯函数式 ScreenState 转移(Phase 1 环境三件套之二)。

# note (luojiaxuan): 契约见 contract.py(冻结,本文件只 import 不修改)。设计要点:
#   * **纯函数**:apply_action 先整体克隆 state 再改克隆体,入参绝不被原地修改;
#   * **无随机、无时间、无哈希序**:遍历只走 list/tuple/dict 插入序,集合只用于
#     成员判定、绝不迭代 —— 同 (state, action) 必得同一新 state;
#   * **策略输出是脏的**:RL rollout 里 policy 会点空、对着按钮打字、拼错动作名,
#     这些一律降级为 no-op 并把原因写进 flags["__last_error__"](每步开头清空),
#     **不抛异常**;而未知 effect kind / 目标控件不存在是**我们自己**的构造 bug,
#     fail fast 抛 ValueError,不吞。
# 坐标口径:policy 给 [0,999] 归一,本文件在 norm_to_pixel 单点换算到 1920×1080。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from causalcache_agentic.contract import (
    Effect, Rect, ScreenState, Widget, Window, norm_to_pixel,
)

# note (luojiaxuan): 诊断用保留 flag 键。双下划线命名与任务自定义 flag 隔离;
# verifier 的断言不应该指向它们,renderer 也不应该把它们画到屏幕上。
ERROR_KEY = "__last_error__"
ANSWER_KEY = "__answer__"

# note (luojiaxuan): 可编辑控件种类 —— type / ctrl+v / BackSpace / Tab 只认这三种。
EDITABLE_KINDS = frozenset({"text_field", "text_area", "cell"})

_MODIFIERS = frozenset({"ctrl", "shift", "alt"})
_KEY_ALIAS = {"control": "ctrl", "ctl": "ctrl", "cmd": "ctrl", "command": "ctrl",
              "meta": "ctrl", "super": "ctrl", "esc": "escape", "enter": "return",
              "kp_enter": "return", "back_space": "backspace", "del": "delete"}

# note (luojiaxuan): scroll 符号的**单点定义**,取值必须与冻结的 computer_use 工具
# 描述一致:"Positive values scroll up, negative values scroll down"。故
# pixels<0(向下滚)⇒ scroll_y 增大(看到列表更靠后的内容),即 scroll_y += -pixels。
# 这条口径同时被 render_official_action_line 渲染成 "Scroll down.",历史文本与真实
# 转移方向因此自洽 —— 早期版本取 +1,会让冻结 policy 想往下翻时反而滚到顶(踩过)。
SCROLL_SIGN = -1


def _clone_widget(w: Widget) -> Widget:
    """浅拷贝控件:rect / Effect 元组都是 frozen 不可变,可安全共享;可变的
    style / meta 单独复制,避免新旧 state 之间串改。"""
    return Widget(
        wid=w.wid, kind=w.kind, rect=w.rect, text=w.text, value=w.value,
        enabled=w.enabled, visible=w.visible, group=w.group,
        style=dict(w.style), on_click=w.on_click, meta=dict(w.meta),
        on_double_click=w.on_double_click, on_enter=w.on_enter)


def _clone_window(win: Window) -> Window:
    return Window(
        app_id=win.app_id, title=win.title, rect=win.rect,
        widgets=[_clone_widget(w) for w in win.widgets], open=win.open,
        scroll_y=win.scroll_y, scroll_area=win.scroll_area,
        scroll_max=win.scroll_max, active_groups=win.active_groups)


def clone_state(state: ScreenState) -> ScreenState:
    """深拷贝到"可安全改写"的程度:不可变对象共享(focus 是 tuple),可变容器新建。

    # note (luojiaxuan): 不用 copy.deepcopy —— 它连 Rect/Effect 一起复制,几百个
    # 控件下慢一个量级,而这里每步都要克隆一次(吞吐是 Phase 1 验收项之一)。
    """
    return ScreenState(
        windows={aid: _clone_window(win) for aid, win in state.windows.items()},
        z_order=list(state.z_order), flags=dict(state.flags),
        clipboard=state.clipboard, focus=state.focus,
        select_all=state.select_all, step=state.step,
        terminated=state.terminated, terminate_status=state.terminate_status)


# ---------------------------------------------------------------- 几何与可见性
# note (luojiaxuan): 以下三个函数是**命中判定与渲染的共享几何口径**,render.py
# 应当 import 它们而不是各写一份,否则"画在哪"与"点得到哪"会悄悄分叉。

def group_visible(win: Window, group: str | None) -> bool:
    """group=None 常驻可见;否则要求它在窗口当前 active_groups 里。"""
    return group is None or group in win.active_groups


def is_scroll_content(win: Window, widget: Widget) -> bool:
    """该控件是否随窗口滚动:style["scroll"] 显式优先,否则"水平重叠且不完全在
    区域上方"。

    # note (luojiaxuan): 这条规则必须与 render.py 的 _in_scroll 逐字一致,否则
    # "画在哪"与"点得到哪"会分叉(点击落空却看得见,是最难查的一类环境 bug)。
    """
    if win.scroll_area is None or str(widget.style.get("role", "")) == "taskbar":
        return False
    if "scroll" in widget.style:
        return bool(widget.style["scroll"])
    area, r = win.scroll_area, widget.rect
    h_hit = r.x < area.x + area.w and r.x + r.w > area.x
    return h_hit and (r.y + r.h) > area.y


def widget_screen_rect(win: Window, widget: Widget) -> Rect:
    """控件在屏幕上的实际矩形(滚动内容按 scroll_y 上移)。"""
    if not is_scroll_content(win, widget):
        return widget.rect
    r = widget.rect
    return Rect(r.x, r.y - win.scroll_y, r.w, r.h)


def widget_hittable(win: Window, widget: Widget) -> bool:
    return widget.visible and widget.enabled and group_visible(win, widget.group)


# ---------------------------------------------------------------- 命中测试

def _hit_in_window(win: Window, px: int, py: int) -> Widget | None:
    """窗口内命中:widgets **逆序**(后加的画在上面,所以先被点到)。"""
    for w in reversed(win.widgets):
        if not widget_hittable(win, w) \
                or not widget_screen_rect(win, w).contains(px, py):
            continue
        # note (luojiaxuan): 滚出视口的内容虽然 rect 命中,但被裁掉了,不可点。
        if is_scroll_content(win, w) and win.scroll_area is not None \
                and not win.scroll_area.contains(px, py):
            continue
        return w
    return None


def _taskbar_hit(state: ScreenState, px: int, py: int) -> tuple[str, Widget] | None:
    """任务栏永远在最上层:它的 rect 落在窗口矩形之外,必须单独扫一遍。"""
    for app_id in reversed(state.z_order):
        win = state.windows.get(app_id)
        if win is None or not win.open:
            continue
        for w in reversed(win.widgets):
            if str(w.style.get("role", "")) == "taskbar" and widget_hittable(win, w) \
                    and w.rect.contains(px, py):
                return app_id, w
    return None


def hit_test(state: ScreenState, px: int, py: int) -> tuple[str, Widget] | None:
    """屏幕像素 → (app_id, widget)。点到空白返回 None。

    顺序:任务栏(常驻置顶)→ z_order 从顶到底第一个 rect 含该点的开着的窗口。
    最上层窗口**吃掉**落在它矩形内的点击:即便窗口内没有控件命中也不再往下穿透。
    """
    taskbar = _taskbar_hit(state, px, py)
    if taskbar is not None:
        return taskbar
    for app_id in reversed(state.z_order):
        win = state.windows.get(app_id)
        if win is None or not win.open or not win.rect.contains(px, py):
            continue
        hit = _hit_in_window(win, px, py)
        return (app_id, hit) if hit is not None else None
    return None


def _window_at(state: ScreenState, px: int, py: int) -> Window | None:
    """z_order 从顶到底,第一个 rect 含该点且开着的窗口(scroll 动作用)。"""
    return next((w for aid in reversed(state.z_order)
                 if (w := state.windows.get(aid)) is not None
                 and w.open and w.rect.contains(px, py)), None)


def find_widget(state: ScreenState, wid: str) -> tuple[Window, Widget] | None:
    """按 wid 全局查找(wid 在一个 state 内唯一,与 contract.verify 同口径)。"""
    for win in state.windows.values():
        for w in win.widgets:
            if w.wid == wid:
                return win, w
    return None


def _focused(state: ScreenState) -> tuple[Window, Widget] | None:
    """把 state.focus 解析成 (window, widget);焦点悬空则 None。"""
    if state.focus is None:
        return None
    win = state.windows.get(state.focus[0])
    if win is None:
        return None
    return next(((win, w) for w in win.widgets if w.wid == state.focus[1]), None)


def _focus_window(state: ScreenState) -> Window | None:
    """焦点所在窗口;无焦点(或其窗口已关)时回落到最上层开着的窗口。"""
    app_id = state.focus[0] if state.focus is not None else state.active_app
    win = state.windows.get(app_id) if app_id else None
    if win is not None and win.open:
        return win
    for aid in reversed(state.z_order):
        cand = state.windows.get(aid)
        if cand is not None and cand.open:
            return cand
    return None


def _drop_hidden_focus(state: ScreenState) -> None:
    """焦点控件因关 dialog / 切 tab / 置 invisible 而看不见时收回焦点。"""
    found = _focused(state)
    if found is None or not found[1].visible \
            or not group_visible(found[0], found[1].group):
        state.focus = None


# ---------------------------------------------------------------- effect 解释器

def _as_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"布尔 effect 值无法解析: {value!r}")


def _need_widget(state: ScreenState, wid: str, kind: str) -> tuple[Window, Widget]:
    found = find_widget(state, wid)
    if found is None:
        raise ValueError(f"effect {kind!r} 的目标控件不存在: {wid!r}")
    return found


def _need_window(state: ScreenState, app_id: str, kind: str) -> Window:
    win = state.windows.get(app_id)
    if win is None:
        raise ValueError(f"effect {kind!r} 的目标 app 不存在: {app_id!r}")
    return win


def _payload_str(effect: Effect, key: str) -> str:
    if key not in effect.payload:
        raise ValueError(f"effect {effect.kind!r} 缺少 payload[{key!r}]")
    return str(effect.payload[key])


def _raise_to_top(state: ScreenState, app_id: str) -> None:
    if app_id in state.z_order:
        state.z_order.remove(app_id)
    state.z_order.append(app_id)


def _scroll_into_view(win: Window, widget: Widget) -> None:
    if win.scroll_area is None or not is_scroll_content(win, widget):
        return
    area, r = win.scroll_area, widget.rect
    top = r.y - win.scroll_y
    new_y = win.scroll_y
    if top < area.y:
        new_y = r.y - area.y
    elif top + r.h > area.y + area.h:
        new_y = r.y + r.h - (area.y + area.h)
    win.scroll_y = max(0, min(win.scroll_max, new_y))


def _active_tab_ids(win: Window) -> frozenset[str]:
    """窗口里所有"由 switch_tab 管理"的 group id —— 用于切页时互斥关闭旧页。"""
    ids: set[str] = set()
    for w in win.widgets:
        for bundle in (w.on_click, w.on_double_click, w.on_enter):
            for e in bundle:
                if e.kind == "switch_tab" and "tab" in e.payload:
                    ids.add(str(e.payload["tab"]))
    return frozenset(ids)


def apply_effect(state: ScreenState, effect: Effect) -> None:
    """就地在**已克隆**的 state 上执行一条 effect。未知 kind 抛 ValueError。"""
    kind = effect.kind
    if kind == "noop":
        return
    if kind == "set_value":
        _need_widget(state, effect.target, kind)[1].value = effect.value
    elif kind == "set_text":
        _need_widget(state, effect.target, kind)[1].text = effect.value
    elif kind == "set_visible":
        _need_widget(state, effect.target, kind)[1].visible = _as_bool(effect.value)
        _drop_hidden_focus(state)
    elif kind == "set_enabled":
        _need_widget(state, effect.target, kind)[1].enabled = _as_bool(effect.value)
    elif kind == "focus_app":
        _need_window(state, effect.target, kind)
        _raise_to_top(state, effect.target)
    elif kind == "open_app":
        _need_window(state, effect.target, kind).open = True
        _raise_to_top(state, effect.target)
    elif kind == "close_app":
        _need_window(state, effect.target, kind).open = False
        # note (luojiaxuan): 关掉的窗口沉到 z_order 最底,保证 active_app(末位)
        # 始终指向一个开着的窗口;不从 z_order 移除,方便之后 open_app 复位。
        if effect.target in state.z_order:
            state.z_order.remove(effect.target)
            state.z_order.insert(0, effect.target)
        if state.focus is not None and state.focus[0] == effect.target:
            state.focus = None
    elif kind == "open_dialog":
        win = _need_window(state, effect.target, kind)
        dialog = _payload_str(effect, "dialog")
        if dialog not in win.active_groups:
            win.active_groups = win.active_groups + (dialog,)
    elif kind == "close_dialog":
        win = _need_window(state, effect.target, kind)
        dialog = _payload_str(effect, "dialog")
        win.active_groups = tuple(g for g in win.active_groups if g != dialog)
        _drop_hidden_focus(state)
    elif kind == "switch_tab":
        win = _need_window(state, effect.target, kind)
        tab = _payload_str(effect, "tab")
        known = _active_tab_ids(win)
        kept = tuple(g for g in win.active_groups if g not in known and g != tab)
        win.active_groups = kept + (tab,)
        _drop_hidden_focus(state)
    elif kind == "set_flag":
        state.flags[effect.target] = effect.value
    elif kind == "flag_from":
        _, w = _need_widget(state, _payload_str(effect, "widget"), kind)
        state.flags[effect.target] = w.value
    elif kind == "copy_widget":
        _, w = _need_widget(state, _payload_str(effect, "widget"), kind)
        state.clipboard = w.value
    elif kind == "paste_into":
        _need_widget(state, effect.target, kind)[1].value = state.clipboard
    elif kind == "append_value":
        _, w = _need_widget(state, effect.target, kind)
        w.value = w.value + effect.value
    elif kind == "clear_value":
        _need_widget(state, effect.target, kind)[1].value = ""
    elif kind == "scroll_to":
        win, w = _need_widget(state, effect.target, kind)
        _scroll_into_view(win, w)
    else:
        raise ValueError(f"未知 effect kind {kind!r}(环境构造 bug,不吞)")


def apply_effects(state: ScreenState, effects: Sequence[Effect]) -> None:
    for e in effects:
        apply_effect(state, e)


def _unwrap(action: Any) -> dict[str, Any] | None:
    """容忍两种入参:computer_use 的 arguments,或整包 {"name","arguments"}。"""
    if not isinstance(action, dict):
        return None
    if "action" not in action and isinstance(action.get("arguments"), dict):
        return action["arguments"]
    return action


def _coordinate(args: dict[str, Any],
                key: str = "coordinate") -> tuple[int, int] | None:
    v = args.get(key)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            return int(round(float(v[0]))), int(round(float(v[1])))
        except (TypeError, ValueError):
            return None
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _fail(state: ScreenState, reason: str) -> ScreenState:
    state.flags[ERROR_KEY] = reason
    return state


def _normalize_keys(raw: Any) -> tuple[str, ...]:
    """键名归一:大小写、'control'/'cmd' 别名、以及 ["ctrl+a"] 这种粘连写法。"""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(_KEY_ALIAS.get(p, p) for item in raw
                 for part in str(item).replace("-", "+").split("+")
                 if (p := part.strip().lower()))


# ---------------------------------------------------------------- 各动作实现

def _do_click(state: ScreenState, args: dict[str, Any], mode: str) -> ScreenState:
    point = _coordinate(args)
    if point is None:
        return _fail(state, "bad_coordinate")
    px, py = norm_to_pixel(*point)
    hit = hit_test(state, px, py)
    if hit is None:
        return _fail(state, "miss")
    app_id, widget = hit
    state.focus = (app_id, widget.wid)
    if widget.kind in EDITABLE_KINDS:
        state.select_all = False
    if mode == "right":
        # note (luojiaxuan): 右键只移焦点,不触发任何 effect(不做上下文菜单)。
        return state
    # note (luojiaxuan): 没声明 on_double_click 的控件,双击退化为单击一次。
    double = mode == "double" and bool(widget.on_double_click)
    apply_effects(state, widget.on_double_click if double else widget.on_click)
    return state


def _editable_focus(state: ScreenState) -> tuple[Widget | None, str]:
    """取"可编辑且此刻真的能编辑"的焦点控件;失败时返回 (None, 错误原因)。"""
    found = _focused(state)
    if found is None:
        return None, "no_focus"
    win, w = found
    if w.kind not in EDITABLE_KINDS or not widget_hittable(win, w):
        return None, "not_editable"
    return w, ""


def _do_type(state: ScreenState, args: dict[str, Any]) -> ScreenState:
    text = args.get("text")
    if not isinstance(text, str):
        return _fail(state, "missing_text")
    w, err = _editable_focus(state)
    if w is None:
        return _fail(state, err)
    w.value = text if state.select_all else w.value + text
    state.select_all = False
    return state


def _do_tab(state: ScreenState) -> ScreenState:
    win = _focus_window(state)
    if win is None:
        return _fail(state, "no_window")
    cands = [w for w in win.widgets
             if w.kind in EDITABLE_KINDS and widget_hittable(win, w)]
    if not cands:
        return _fail(state, "no_editable")
    index = -1
    if state.focus is not None and state.focus[0] == win.app_id:
        for i, w in enumerate(cands):
            if w.wid == state.focus[1]:
                index = i
                break
    nxt = cands[(index + 1) % len(cands)]
    state.focus = (win.app_id, nxt.wid)
    state.select_all = False
    return state


def _do_key(state: ScreenState, args: dict[str, Any]) -> ScreenState:
    keys = _normalize_keys(args.get("keys"))
    if not keys:
        return _fail(state, "missing_keys")
    mods = frozenset(k for k in keys if k in _MODIFIERS)
    mains = [k for k in keys if k not in _MODIFIERS]
    main = mains[0] if mains else ""
    combo = "+".join(sorted(mods) + ([main] if main else []))
    if "ctrl" in mods:
        if main == "a":
            state.select_all = True
            return state
        if main == "c":
            found = _focused(state)
            if found is None:
                return _fail(state, "no_focus")
            state.clipboard = found[1].value
            return state
        if main == "v":
            w, err = _editable_focus(state)
            if w is None:
                return _fail(state, err)
            w.value = state.clipboard if state.select_all else w.value + state.clipboard
            state.select_all = False
            return state
        if main == "s":
            win = _focus_window(state)
            if win is None:
                return _fail(state, "no_window")
            for w in win.widgets:
                # note (luojiaxuan): 约定 wid 以 "::save" 结尾的按钮是该窗口的保存
                # 入口;取列表序第一个可见可用者,保证确定性。
                if w.wid.endswith("::save") and widget_hittable(win, w):
                    apply_effects(state, w.on_click)
                    return state
            return _fail(state, "no_save_button")
        return _fail(state, f"unsupported_key:{combo}")
    if main == "return":
        found = _focused(state)
        if found is None:
            return _fail(state, "no_focus")
        apply_effects(state, found[1].on_enter)
        return state
    if main == "escape":
        win = _focus_window(state)
        if win is None or not win.active_groups:
            return _fail(state, "no_active_group")
        win.active_groups = win.active_groups[:-1]
        _drop_hidden_focus(state)
        return state
    if main == "tab":
        return _do_tab(state)
    if main == "backspace":
        w, err = _editable_focus(state)
        if w is None:
            return _fail(state, err)
        # note (luojiaxuan): select_all 下退格清空整格(真实 GUI 行为),否则删末字符。
        w.value = "" if state.select_all else w.value[:-1]
        state.select_all = False
        return state
    return _fail(state, f"unsupported_key:{combo}")


def _do_scroll(state: ScreenState, args: dict[str, Any]) -> ScreenState:
    pixels = _as_int(args.get("pixels", 0))
    if pixels is None:
        return _fail(state, "bad_pixels")
    if args.get("coordinate") is not None:
        point = _coordinate(args)
        if point is None:
            return _fail(state, "bad_coordinate")
        win = _window_at(state, *norm_to_pixel(*point))
        if win is None:
            return _fail(state, "miss")
    else:
        app_id = state.active_app
        win = state.windows.get(app_id) if app_id else None
        if win is None or not win.open:
            return _fail(state, "no_window")
    if win.scroll_area is None or win.scroll_max <= 0:
        return _fail(state, "not_scrollable")
    win.scroll_y = max(0, min(win.scroll_max, win.scroll_y + SCROLL_SIGN * pixels))
    return state


def _do_end(state: ScreenState, args: dict[str, Any], name: str) -> ScreenState:
    """terminate / answer 的共同出口。"""
    if name == "answer":
        # note (luojiaxuan): answer ≡ terminate(success),附带把答案文本落到保留
        # flag,让"问答型"任务也能用 flag_equals 断言,不必额外造控件。
        state.flags[ANSWER_KEY] = str(args.get("text", ""))
        status = "success"
    else:
        status = str(args.get("status", "success"))
        if status not in ("success", "failure"):
            state.flags[ERROR_KEY] = f"bad_status:{status}"
    state.terminated = True
    state.terminate_status = status
    return state


# ---------------------------------------------------------------- 顶层 API

def apply_action(state: ScreenState, action: dict[str, Any]) -> ScreenState:
    """执行一个 computer_use 动作,返回**新** state(入参不被修改)。

    坐标为 [0,999] 归一。任何非法输入都降级成 no-op 并写
    flags["__last_error__"],不抛异常;step 无论成败都 +1。
    """
    new = clone_state(state)
    new.step = state.step + 1
    new.flags.pop(ERROR_KEY, None)

    args = _unwrap(action)
    if args is None:
        return _fail(new, "action_not_dict")
    if new.terminated:
        return _fail(new, "already_terminated")
    name = args.get("action")
    if not isinstance(name, str):
        return _fail(new, "missing_action")
    name = name.strip()

    if name in ("left_click", "click", "double_click", "right_click"):
        mode = {"double_click": "double", "right_click": "right"}.get(name, "left")
        return _do_click(new, args, mode)
    if name == "type":
        return _do_type(new, args)
    if name == "key":
        return _do_key(new, args)
    if name == "scroll":
        return _do_scroll(new, args)
    if name in ("terminate", "answer"):
        return _do_end(new, args, name)
    if name == "wait":
        return new                       # note (luojiaxuan): 唯一"静默 no-op"动作
    return _fail(new, f"unsupported_action:{name}")


@dataclass(frozen=True)
class GUIExecutor:
    """contract.Executor 的实现(薄包装,便于依赖注入与替换)。"""

    def apply(self, state: ScreenState, action: dict[str, Any]) -> ScreenState:
        return apply_action(state, action)


__all__ = ["ANSWER_KEY", "EDITABLE_KINDS", "ERROR_KEY", "SCROLL_SIGN",
           "GUIExecutor", "apply_action", "apply_effect", "apply_effects",
           "clone_state", "find_widget", "group_visible", "hit_test",
           "is_scroll_content", "widget_hittable", "widget_screen_rect"]
