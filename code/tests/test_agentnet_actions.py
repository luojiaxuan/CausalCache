"""AgentNet pyautogui → computer_use 解析器的冻结不变量。

# note (luojiaxuan): 覆盖 14 个已勘察原语各至少一例、0.0/1.0 坐标边界、以及
# fail-closed 逐项理由;target 一律回代冻结的 parse_gui_owl_osworld_action
# 验证合约闭环(解析器内部本就自检,这里再显式锁一层)。
"""

from __future__ import annotations

import pytest

from causalcache.agentnet_actions import (
    parse_agentnet_step,
    render_target_text,
)
from causalcache.osworld_gui_owl import parse_gui_owl_osworld_action

SCREEN = (1440, 900)


def parse(code: str):
    return parse_agentnet_step(code, screen_size=SCREEN)


def test_click_maps_to_left_click_with_norm999_coordinate() -> None:
    step = parse("pyautogui.click(x=0.018, y=0.508)")
    assert step.parseable and step.target_available
    assert step.target_tool_call == {
        "name": "computer_use",
        "arguments": {"action": "left_click", "coordinate": [18, 507]},
    }
    assert step.history_action["type"] == "click"
    assert step.history_action["x"] == round(0.018 * 1439)
    assert step.history_action["y"] == round(0.508 * 899)
    action = parse_gui_owl_osworld_action(
        render_target_text(step.target_tool_call), screen_size=SCREEN
    )
    assert action.type == "click"


def test_click_with_right_button_kw() -> None:
    step = parse("pyautogui.click(x=0.5, y=0.5, button='right')")
    assert step.target_tool_call["arguments"]["action"] == "right_click"
    assert step.history_action["type"] == "right_click"


def test_double_click() -> None:
    step = parse("pyautogui.doubleClick(x=0.153, y=0.283)")
    assert step.target_tool_call["arguments"]["action"] == "double_click"
    assert step.history_action["type"] == "double_click"
    assert step.history_action["clicks"] == 2


def test_right_click() -> None:
    step = parse("pyautogui.rightClick(x=0.304, y=0.684)")
    assert step.target_tool_call["arguments"]["action"] == "right_click"


def test_middle_click() -> None:
    step = parse("pyautogui.middleClick(x=0.4661, y=0.575)")
    assert step.target_tool_call["arguments"]["action"] == "middle_click"
    assert step.history_action["button"] == "middle"


def test_triple_click_history_only() -> None:
    step = parse("computer.tripleClick(x=0.226, y=0.311)")
    assert step.parseable
    assert not step.target_available
    assert step.target_unavailable_reason == "triple_click_not_in_computer_use"
    assert step.history_action["clicks"] == 3


def test_move_to() -> None:
    step = parse("pyautogui.moveTo(x=0.394, y=0.604)")
    assert step.target_tool_call["arguments"]["action"] == "mouse_move"
    assert step.history_action["type"] == "move"


def test_drag_to_alone_uses_coordinate2() -> None:
    step = parse("pyautogui.dragTo(x=0.085, y=0.361, button='left')")
    assert step.target_tool_call["arguments"] == {
        "action": "left_click_drag",
        "coordinate2": [round(0.085 * 999), round(0.361 * 999)],
    }
    assert step.history_action["type"] == "drag"


def test_move_then_drag_combo_keeps_both_endpoints() -> None:
    step = parse("pyautogui.moveTo(x=0.1, y=0.2)\npyautogui.dragTo(x=0.3, y=0.4)")
    args = step.target_tool_call["arguments"]
    assert args["action"] == "left_click_drag"
    assert args["coordinate"] == [round(0.1 * 999), round(0.2 * 999)]
    assert args["coordinate2"] == [round(0.3 * 999), round(0.4 * 999)]
    action = parse_gui_owl_osworld_action(
        render_target_text(step.target_tool_call), screen_size=SCREEN
    )
    assert action.type == "drag"


def test_write_message_kw() -> None:
    step = parse("pyautogui.write(message='Send Keybindings To Shell')")
    assert step.target_tool_call["arguments"] == {
        "action": "type",
        "text": "Send Keybindings To Shell",
    }
    assert step.history_action == {
        "type": "type_text",
        "text": "Send Keybindings To Shell",
    }


def test_press_single_key() -> None:
    step = parse("pyautogui.press('pagedown')")
    assert step.target_tool_call["arguments"] == {"action": "key", "keys": ["pagedown"]}
    assert step.history_action == {"type": "press", "key": "pagedown"}


def test_hotkey_list_form_and_casefold() -> None:
    step = parse("pyautogui.hotkey(['Ctrl', 'O'])")
    assert step.target_tool_call["arguments"] == {"action": "key", "keys": ["ctrl", "o"]}
    assert step.history_action == {"type": "hotkey", "keys": ("ctrl", "o")}


def test_hotkey_varargs_form() -> None:
    step = parse("pyautogui.hotkey('ctrl', 'shift', 't')")
    assert step.target_tool_call["arguments"]["keys"] == ["ctrl", "shift", "t"]


def test_scroll_maps_clicks_one_to_one() -> None:
    step = parse("pyautogui.scroll(-4)")
    assert step.target_tool_call["arguments"] == {"action": "scroll", "pixels": -4}
    assert step.history_action == {"type": "scroll", "dx": 0, "dy": -4}


def test_hscroll() -> None:
    step = parse("pyautogui.hscroll(13)")
    assert step.target_tool_call["arguments"] == {"action": "hscroll", "pixels": 13}
    assert step.history_action["dx"] == 13
    assert step.history_action["dy"] == 0


def test_wait() -> None:
    step = parse("computer.wait()")
    assert step.target_tool_call["arguments"] == {"action": "wait"}
    assert step.history_action == {"type": "wait"}


def test_terminate_failure_and_success() -> None:
    failure = parse("computer.terminate(status='failure')")
    assert failure.target_tool_call["arguments"] == {
        "action": "terminate",
        "status": "failure",
    }
    assert failure.history_action == {"type": "fail"}
    success = parse("computer.terminate(status='success')")
    assert success.history_action == {"type": "done"}


@pytest.mark.parametrize("fraction,expected", [(0.0, 0), (1.0, 999)])
def test_coordinate_norm999_boundaries(fraction: float, expected: int) -> None:
    step = parse(f"pyautogui.click(x={fraction}, y={fraction})")
    assert step.target_tool_call["arguments"]["coordinate"] == [expected, expected]


def test_coordinate_pixel_boundaries() -> None:
    step = parse("pyautogui.click(x=1.0, y=1.0)")
    assert step.history_action["x"] == SCREEN[0] - 1
    assert step.history_action["y"] == SCREEN[1] - 1


@pytest.mark.parametrize(
    "code,reason",
    [
        ("pyautogui.click(x=1.2, y=0.5)", "x_out_of_range"),
        ("pyautogui.click(x=0.5, y=-0.1)", "y_out_of_range"),
        ("pyautogui.click()", "missing_coordinate"),
        ("pyautogui.locateOnScreen('a.png')", "unknown_primitive"),
        ("import os", "not_expression_statements"),
        ("pyautogui.click(x=0.1, y=0.1)\npyautogui.click(x=0.2, y=0.2)", "multi_statement"),
        ("pyautogui.scroll(0)", "scroll_amount_invalid"),
        ("computer.terminate(status='partial')", "terminate_status_invalid"),
        ("pyautogui.click(x=0.1, y=0.1, weird=1)", "unexpected_kwargs"),
        ("", "empty_code"),
        ("pyautogui.click(x=0.1,", "syntax_error"),
    ],
)
def test_fail_closed_reasons(code: str, reason: str) -> None:
    step = parse(code)
    assert not step.parseable
    assert step.unparseable_reason == reason
    assert step.history_action is None
    assert step.target_tool_call is None


def test_render_target_text_shape() -> None:
    step = parse("pyautogui.click(x=0.5, y=0.5)")
    text = render_target_text(step.target_tool_call)
    assert text.startswith("<tool_call>\n") and text.endswith("\n</tool_call>")
