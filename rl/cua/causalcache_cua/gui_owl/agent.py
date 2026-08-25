# note (luojiaxuan): GUI-Owl-1.5 agent registry 行。渲染/解析全在 adapter,
# 此处仅注册 AutoAdapterAgent(与 mai_ui/agent.py 同法);chat template 无
# 额外参数需求,不覆写 build_generation_prompt。
from __future__ import annotations

from lite.agents.core.agent.base import AutoAdapterAgent


class GuiOwlMobileAgent(AutoAdapterAgent, key="gui_owl@mobile@use"):
    pass
