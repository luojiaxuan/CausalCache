"""AgentNet pyautogui 步码到冻结 computer_use 合约的确定性映射。

# note (luojiaxuan): AgentNet 每步的 ``value.code`` 是 pyautogui/computer.* 语句,
# 坐标是 0-1 归一化小数。挖掘管线要把每步转成两样东西:
#   1. **history_action**:``DesktopAction.to_mapping()`` 形态(像素坐标),进
#      ``_compact_history`` 的 C_r 文字历史 —— 与冻结桌面 runtime 的事件语义逐字节
#      一致(默认字段 button/clicks/duration 由 DesktopAction 本身补齐);
#   2. **target_tool_call**:GUI-Owl 输出空间的 ``computer_use`` tool-call JSON
#      (坐标 [0,999] 归一化整数),作 teacher-forced 打分的 target。
# fail-closed:14 个已勘察原语之外的代码、越界坐标、意外 kwargs,一律整步标记
# unparseable 并给出 reason,不猜。两条自检闭环:history_action 必须能过
# ``DesktopAction.from_mapping``,target 必须能过 ``parse_gui_owl_osworld_action``,
# 保证产出永远落在冻结合约内。
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

from causalcache.osworld import DesktopAction

# note (luojiaxuan): 循环依赖规避 —— osworld_gui_owl 只在自检时惰性 import。

SCROLL_CLICKS_TO_PIXELS = 1  # 冻结 runtime 里 "pixels" 直接落 pyautogui.scroll,1:1。


@dataclass(frozen=True)
class ParsedAgentNetStep:
    """一步 AgentNet 代码的双形态产物。

    # note (luojiaxuan): 三种终态,层层收窄而不是一刀切:
    #   unparseable_reason 非空       —— 整步不可用(历史都进不了);
    #   target_unavailable_reason 非空 —— 可进文字历史,但不能作 teacher target
    #                                    (如 tripleClick 不在 computer_use 枚举里);
    #   两者皆空                       —— 双形态齐备。
    """

    raw_code: str
    history_action: dict[str, Any] | None
    osworld_action: str | None
    target_tool_call: dict[str, Any] | None
    unparseable_reason: str | None
    target_unavailable_reason: str | None = None

    @property
    def parseable(self) -> bool:
        return self.unparseable_reason is None

    @property
    def target_available(self) -> bool:
        return self.parseable and self.target_tool_call is not None


class _Unparseable(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def render_target_text(tool_call: dict[str, Any]) -> str:
    """target tool-call 的 teacher-forced 文本形态(与 GUI-Owl 输出协议一致)。"""
    payload = json.dumps(tool_call, ensure_ascii=False)
    return f"<tool_call>\n{payload}\n</tool_call>"


def _fraction(value: Any, *, label: str) -> float:
    if type(value) not in (int, float):
        raise _Unparseable(f"{label}_not_numeric")
    fraction = float(value)
    if not 0.0 <= fraction <= 1.0:
        # note (luojiaxuan): AgentNet 坐标应为 0-1 小数;出现 >1 的值说明该轨迹混入
        # 像素坐标或标注损坏,fail-closed 而不是按猜测的分辨率除回去。
        raise _Unparseable(f"{label}_out_of_range")
    return fraction


def _norm999(fraction: float) -> int:
    return round(fraction * 999)


def _pixel(fraction: float, size: int) -> int:
    return round(fraction * (size - 1))


def _call_parts(node: ast.expr) -> tuple[str, list[Any], dict[str, Any]]:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        raise _Unparseable("not_a_known_call")
    if not isinstance(node.func.value, ast.Name):
        raise _Unparseable("not_a_known_call")
    qualified = f"{node.func.value.id}.{node.func.attr}"
    try:
        args = [ast.literal_eval(arg) for arg in node.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
    except (ValueError, SyntaxError) as error:
        raise _Unparseable("non_literal_argument") from error
    if any(kw.arg is None for kw in node.keywords):
        raise _Unparseable("star_kwargs")
    return qualified, args, kwargs


def _coordinate_args(
    args: list[Any], kwargs: dict[str, Any], *, allowed_extra: frozenset[str]
) -> tuple[float, float, dict[str, Any]]:
    extra = dict(kwargs)
    if "x" in extra or "y" in extra:
        if len(args) != 0 or "x" not in extra or "y" not in extra:
            raise _Unparseable("coordinate_signature")
        x = extra.pop("x")
        y = extra.pop("y")
    elif len(args) == 2:
        x, y = args
    else:
        raise _Unparseable("missing_coordinate")
    unknown = set(extra) - allowed_extra
    if unknown:
        raise _Unparseable("unexpected_kwargs")
    return _fraction(x, label="x"), _fraction(y, label="y"), extra


_CLICK_TARGET_BY_BUTTON = {
    "left": "left_click",
    "right": "right_click",
    "middle": "middle_click",
}


def _map_single(
    qualified: str,
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    screen_size: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    """返回 (history_mapping, target_arguments, target_none_reason)。"""
    width, height = screen_size

    if qualified == "pyautogui.click":
        fx, fy, extra = _coordinate_args(
            args, kwargs, allowed_extra=frozenset({"button", "clicks", "interval"})
        )
        button = extra.get("button", "left")
        if button not in _CLICK_TARGET_BY_BUTTON:
            raise _Unparseable("unknown_button")
        clicks = extra.get("clicks", 1)
        if type(clicks) is not int or not 1 <= clicks <= 3:
            raise _Unparseable("unknown_clicks")
        if clicks == 2:
            history = {"type": "double_click", "x": _pixel(fx, width), "y": _pixel(fy, height)}
            target = {"action": "double_click", "coordinate": [_norm999(fx), _norm999(fy)]}
        elif clicks == 3:
            history = {
                "type": "click",
                "x": _pixel(fx, width),
                "y": _pixel(fy, height),
                "button": button,
                "clicks": 3,
            }
            return history, None, "triple_click_not_in_computer_use"
        else:
            history = {
                "type": "right_click" if button == "right" else "click",
                "x": _pixel(fx, width),
                "y": _pixel(fy, height),
            }
            if button == "middle":
                history["button"] = "middle"
            target = {
                "action": _CLICK_TARGET_BY_BUTTON[button],
                "coordinate": [_norm999(fx), _norm999(fy)],
            }
        return history, target, None

    if qualified == "pyautogui.doubleClick":
        fx, fy, _ = _coordinate_args(args, kwargs, allowed_extra=frozenset({"interval"}))
        history = {"type": "double_click", "x": _pixel(fx, width), "y": _pixel(fy, height)}
        return history, {"action": "double_click", "coordinate": [_norm999(fx), _norm999(fy)]}, None

    if qualified == "pyautogui.rightClick":
        fx, fy, _ = _coordinate_args(args, kwargs, allowed_extra=frozenset())
        history = {"type": "right_click", "x": _pixel(fx, width), "y": _pixel(fy, height)}
        return history, {"action": "right_click", "coordinate": [_norm999(fx), _norm999(fy)]}, None

    if qualified == "pyautogui.middleClick":
        fx, fy, _ = _coordinate_args(args, kwargs, allowed_extra=frozenset())
        history = {
            "type": "click",
            "x": _pixel(fx, width),
            "y": _pixel(fy, height),
            "button": "middle",
        }
        return history, {"action": "middle_click", "coordinate": [_norm999(fx), _norm999(fy)]}, None

    if qualified == "computer.tripleClick":
        fx, fy, _ = _coordinate_args(args, kwargs, allowed_extra=frozenset())
        history = {
            "type": "click",
            "x": _pixel(fx, width),
            "y": _pixel(fy, height),
            "button": "left",
            "clicks": 3,
        }
        return history, None, "triple_click_not_in_computer_use"

    if qualified == "pyautogui.moveTo":
        fx, fy, _ = _coordinate_args(args, kwargs, allowed_extra=frozenset({"duration"}))
        history = {"type": "move", "x": _pixel(fx, width), "y": _pixel(fy, height)}
        return history, {"action": "mouse_move", "coordinate": [_norm999(fx), _norm999(fy)]}, None

    if qualified == "pyautogui.dragTo":
        fx, fy, extra = _coordinate_args(
            args, kwargs, allowed_extra=frozenset({"button", "duration"})
        )
        if extra.get("button", "left") != "left":
            raise _Unparseable("drag_non_left_button")
        history = {"type": "drag", "x": _pixel(fx, width), "y": _pixel(fy, height)}
        target = {"action": "left_click_drag", "coordinate2": [_norm999(fx), _norm999(fy)]}
        return history, target, None

    if qualified == "pyautogui.write":
        if "message" in kwargs and not args:
            text = kwargs.pop("message")
        elif len(args) == 1 and "message" not in kwargs:
            text = args[0]
        else:
            raise _Unparseable("write_signature")
        if set(kwargs) - {"interval"}:
            raise _Unparseable("unexpected_kwargs")
        if not isinstance(text, str) or len(text) > 10000:
            raise _Unparseable("write_text_invalid")
        return {"type": "type_text", "text": text}, {"action": "type", "text": text}, None

    if qualified == "pyautogui.press":
        if kwargs.get("presses", 1) != 1:
            raise _Unparseable("press_repeat")
        if set(kwargs) - {"presses", "interval"}:
            raise _Unparseable("unexpected_kwargs")
        if len(args) != 1:
            raise _Unparseable("press_signature")
        key = args[0]
        if isinstance(key, list):
            if len(key) != 1:
                raise _Unparseable("press_key_list")
            key = key[0]
        if not isinstance(key, str) or not key or len(key) > 32:
            raise _Unparseable("press_key_invalid")
        return {"type": "press", "key": key}, {"action": "key", "keys": [key.casefold()]}, None

    if qualified == "pyautogui.hotkey":
        if kwargs:
            raise _Unparseable("unexpected_kwargs")
        if len(args) == 1 and isinstance(args[0], list):
            keys = args[0]
        else:
            keys = list(args)
        if not keys or any(
            not isinstance(key, str) or not key or len(key) > 32 for key in keys
        ):
            raise _Unparseable("hotkey_keys_invalid")
        keys = [key.casefold() for key in keys]
        if len(keys) == 1:
            return {"type": "press", "key": keys[0]}, {"action": "key", "keys": keys}, None
        if len(keys) > 5:
            raise _Unparseable("hotkey_too_many_keys")
        return {"type": "hotkey", "keys": keys}, {"action": "key", "keys": keys}, None

    if qualified in {"pyautogui.scroll", "pyautogui.hscroll"}:
        if kwargs:
            raise _Unparseable("unexpected_kwargs")
        if len(args) != 1 or type(args[0]) is not int or args[0] == 0:
            raise _Unparseable("scroll_amount_invalid")
        amount = args[0] * SCROLL_CLICKS_TO_PIXELS
        if qualified == "pyautogui.scroll":
            return {"type": "scroll", "dy": amount}, {"action": "scroll", "pixels": amount}, None
        return (
            {"type": "scroll", "dy": 0, "dx": amount},
            {"action": "hscroll", "pixels": amount},
            None,
        )

    if qualified == "computer.wait":
        if args or kwargs:
            raise _Unparseable("wait_signature")
        return {"type": "wait"}, {"action": "wait"}, None

    if qualified == "computer.terminate":
        status = kwargs.get("status") if not args else args[0]
        if len(args) > 1 or set(kwargs) - {"status"}:
            raise _Unparseable("terminate_signature")
        if status not in {"success", "failure"}:
            raise _Unparseable("terminate_status_invalid")
        history = {"type": "done" if status == "success" else "fail"}
        return history, {"action": "terminate", "status": status}, None

    raise _Unparseable("unknown_primitive")


def parse_agentnet_step(
    code: str, *, screen_size: tuple[int, int]
) -> ParsedAgentNetStep:
    """把一步 AgentNet code 解析为 history/target 双形态;任何意外 fail-closed。"""

    def failed(reason: str) -> ParsedAgentNetStep:
        return ParsedAgentNetStep(
            raw_code=code,
            history_action=None,
            osworld_action=None,
            target_tool_call=None,
            unparseable_reason=reason,
        )

    if not isinstance(code, str) or not code.strip():
        return failed("empty_code")
    try:
        module = ast.parse(code.strip())
    except SyntaxError:
        return failed("syntax_error")
    statements = module.body
    if not statements or any(not isinstance(stmt, ast.Expr) for stmt in statements):
        return failed("not_expression_statements")

    try:
        calls = [_call_parts(stmt.value) for stmt in statements]  # type: ignore[attr-defined]
        if len(calls) == 1:
            qualified, args, kwargs = calls[0]
            history, target_args, none_reason = _map_single(
                qualified, args, kwargs, screen_size=screen_size
            )
        elif (
            len(calls) == 2
            and calls[0][0] == "pyautogui.moveTo"
            and calls[1][0] == "pyautogui.dragTo"
        ):
            # note (luojiaxuan): AgentNet 拖拽惯用 moveTo+dragTo 两句;冻结合约里
            # 这是一个 left_click_drag(coordinate=起点, coordinate2=终点)。history
            # 侧用 drag(终点)—— 与 parse_gui_owl_osworld_action 对 drag 的语义一致。
            sx, sy, _ = _coordinate_args(
                calls[0][1], calls[0][2], allowed_extra=frozenset({"duration"})
            )
            ex, ey, extra = _coordinate_args(
                calls[1][1], calls[1][2], allowed_extra=frozenset({"button", "duration"})
            )
            if extra.get("button", "left") != "left":
                raise _Unparseable("drag_non_left_button")
            width, height = screen_size
            history = {"type": "drag", "x": _pixel(ex, width), "y": _pixel(ey, height)}
            target_args = {
                "action": "left_click_drag",
                "coordinate": [_norm999(sx), _norm999(sy)],
                "coordinate2": [_norm999(ex), _norm999(ey)],
            }
            none_reason = None
        else:
            return failed("multi_statement")
    except _Unparseable as error:
        return failed(error.reason)

    # note (luojiaxuan): 自检闭环 1 —— history mapping 必须能过冻结的
    # DesktopAction.from_mapping,并以补齐默认值后的 to_mapping() 入库,保证与
    # 桌面 runtime 事件里的 action 字段逐字节同构。
    try:
        desktop_action = DesktopAction.from_mapping(history, screen_size=screen_size)
    except (ValueError, TypeError):
        return failed("history_contract_rejected")
    history_mapping = desktop_action.to_mapping()
    osworld_action = desktop_action.to_osworld()

    tool_call = None
    if target_args is not None:
        tool_call = {"name": "computer_use", "arguments": target_args}
        # note (luojiaxuan): 自检闭环 2 —— target 渲染成文本后必须能被冻结的
        # parse_gui_owl_osworld_action 解析回来;任何解析失败都说明 target 落在
        # 合约之外,fail-closed。
        from causalcache.osworld_gui_owl import parse_gui_owl_osworld_action

        try:
            parse_gui_owl_osworld_action(
                render_target_text(tool_call), screen_size=screen_size
            )
        except (ValueError, TypeError):
            return failed("target_contract_rejected")
    elif none_reason is None:
        return failed("target_missing_without_reason")

    return ParsedAgentNetStep(
        raw_code=code,
        history_action=history_mapping,
        osworld_action=osworld_action,
        target_tool_call=tool_call,
        unparseable_reason=None,
        target_unavailable_reason=none_reason,
    )


__all__ = [
    "ParsedAgentNetStep",
    "SCROLL_CLICKS_TO_PIXELS",
    "parse_agentnet_step",
    "render_target_text",
]
