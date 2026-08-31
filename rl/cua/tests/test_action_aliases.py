# note (luojiaxuan): 动作方言别名回归。这些名字是模型先验带入的同义词,
# 不接受它们只会让该步变成 unknown 空转 —— 实测训练中 double_tap 166 次、
# pull 25 次、open_app 22 次、scroll 4 次,而官方评测侧对同样的输出直接抛
# ValueError(CheckInvoiceTask3 因此 37 次全败)。
import pytest

from sglang_omni_rl.gui_owl.action_space import GuiOwlMobileActionSpace


def _convert(payload):
    space = GuiOwlMobileActionSpace()
    return space._convert_single_from_agent(
        {"name": "mobile_use", "arguments": payload}
    )


@pytest.mark.parametrize("name", ["scroll", "pull"])
def test_swipe_aliases_map_like_swipe(name):
    """scroll/pull 的 wire 签名与 swipe 一致,应产生相同的 Lite 动作。"""
    coords = {"coordinate": [500, 870], "coordinate2": [500, 670]}
    assert _convert({"action": name, **coords}) == _convert(
        {"action": "swipe", **coords})


@pytest.mark.parametrize("payload", [
    {"action": "open", "text": "mattermost"},
    {"action": "open_app", "text": "mattermost"},
    {"action": "open_app", "app_name": "mattermost"},
])
def test_open_app_alias_and_param_names(payload):
    out = _convert(payload)
    assert out and "mattermost" in str(out)


def test_unknown_action_counted_and_not_crashing():
    """MobileWorld 无双击面,double_tap 仍走 unknown 反馈而非抛错,但要被计数。"""
    from sglang_omni_rl.gui_owl import action_space as mod
    before = mod._UNKNOWN_ACTION_COUNTS.get("double_tap", 0)
    out = _convert({"action": "double_tap", "coordinate": [626, 315]})
    assert out is not None
    assert mod._UNKNOWN_ACTION_COUNTS.get("double_tap", 0) == before + 1
