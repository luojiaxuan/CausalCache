# note (luojiaxuan): GUI-Owl-1.5 action space(CausalCache out-of-tree family)。
# wire 形态 = MobileWorld 官方 gui_owl_1_5 prompts 的单一 mobile_use 工具:
# action ∈ {key, click, long_press, swipe, type, system_button, open, wait,
#           answer, interact, terminate},坐标 0-1000 标注、解析按 ÷999
# (官方 parse_action_to_structure_output 实测 108→0.108108=108/999,
#  与 MAI-UI 同族,复用其缩放助手)。
# 与 MAI-UI 的差异:swipe 是精确端点(coordinate→coordinate2)而非方向枚举;
# terminate.status 枚举 success/failure(与 Lite 一致,无需映射);
# 多出 key(adb keyevent,canonical 无对应→unknown 反馈)与
# interact(→ ask_user extra tool);button 枚举大写 Back/Home/Menu/Enter。
from __future__ import annotations

import logging
from typing import Any, Literal

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.action_space.base import LiteMobileActionSpace
from lite.agents.core.action_space.utils.geometry import required_coord
from lite.agents.core.action_space.utils.unknown_wrapper_action import (
    unknown_wrapper_action_batch,
)
from lite.agents.models.mai_ui.action_space import (
    _required_from_mai,
    _scale_to_mai,
)
from lite.core.tools.action_space import (
    LITE_MOBILE_ACTION_BATCH_TOOL_NAME,
    merge_adjacent_lite_action_batches,
)
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_name,
)
from lite.core.tools.extra_tools import (
    LiteFinishToolSet,
    extra_tool_name_and_arguments_are_admitted,
)
from lite.core.tools.schemas import tool

logger = logging.getLogger(__name__)

_GUI_OWL_NATIVE_TOOL_NAME = "mobile_use"

# note (luojiaxuan): 与官方 system prompt 的 enum 顺序一致(字节级无关紧要,
# schema 不进 prompt——system prompt 是硬烤模板;此 enum 只服务 valid_actions
# 过滤与内部一致性)。
_GUI_OWL_ACTIONS = (
    "key", "click", "long_press", "swipe", "type", "system_button",
    "open", "wait", "answer", "interact", "terminate",
)


def _action_values(cls) -> set[str]:
    return set(_GUI_OWL_ACTIONS)


class GuiOwlMobileActionSpace(BaseActionSpace, key="gui_owl@mobile"):
    """GUI-Owl-1.5 mobile action space(单 mobile_use 包装工具)。

    坐标:wire [0,999] 语义(标注 0-1000,官方解析 ÷999)↔ Lite [0,1000]。
    system prompt 为 SFT 文本硬烤,不走 schema 渲染;因此
    ``filter_tool_schemas_for_valid_actions`` 原样返回(与 MAI-UI 同策)。
    """

    @staticmethod
    @tool(
        action="The action to perform.",
        coordinate="(x, y) for click/long_press or swipe start.",
        coordinate2="(x, y) swipe end point.",
        text="Text for key/type/open/answer/interact.",
        time="Seconds for long_press/wait.",
        button="System button: Back/Home/Menu/Enter.",
        status="Terminal status for terminate.",
    )
    def mobile_use(
        action: Literal[
            "key", "click", "long_press", "swipe", "type", "system_button",
            "open", "wait", "answer", "interact", "terminate",
        ],
        coordinate: list[int] | None = None,
        coordinate2: list[int] | None = None,
        text: str | None = None,
        time: float | None = None,
        button: Literal["Back", "Home", "Menu", "Enter"] | None = None,
        status: Literal["success", "failure"] | None = None,
    ) -> dict[str, Any]:
        """Use a touchscreen to interact with a mobile device."""
        args: dict[str, Any] = {"action": action}
        for k, v in (
            ("coordinate", coordinate), ("coordinate2", coordinate2),
            ("text", text), ("time", time), ("button", button),
            ("status", status),
        ):
            if v is not None:
                args[k] = v
        return make_tool_call(_GUI_OWL_NATIVE_TOOL_NAME, args)

    @classmethod
    def filter_tool_schemas_for_valid_actions(
        cls, schemas: list[dict[str, Any]], valid_actions: list[str]
    ) -> list[dict[str, Any]]:
        # system prompt 是 SFT 文本,不裁剪 native 工具面(同 mai_ui)。
        return schemas

    # ------------------------------------------------------------------
    # Lite -> GUI-Owl wire(SFT 导出/replay 用)
    # ------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        if name == LITE_MOBILE_ACTION_BATCH_TOOL_NAME:
            result: list[dict[str, Any]] = []
            for child in args["actions"]:
                action = child["action"]
                child_args = {k: v for k, v in child.items() if k != "action"}
                result.extend(
                    self._convert_single_to_agent(make_tool_call(action, child_args))
                )
            return result

        mk = GuiOwlMobileActionSpace.mobile_use
        if name == "tap":
            clicks = args.get("clicks", 1)
            if clicks != 1:
                raise ValueError(
                    f"GUI-Owl cannot render tap(clicks={clicks}): wire has no "
                    "double-tap spelling"
                )
            return [mk(action="click",
                       coordinate=_scale_to_mai(
                           required_coord(args.get("coordinate"), dimensions=2)))["function"]]
        if name == "long_press":
            return [mk(action="long_press",
                       coordinate=_scale_to_mai(
                           required_coord(args.get("coordinate"), dimensions=2)),
                       time=args.get("duration"))["function"]]
        if name in ("swipe", "drag"):
            return [mk(action="swipe",
                       coordinate=_scale_to_mai(
                           required_coord(args.get("start_coordinate"),
                                          name="start_coordinate", dimensions=2)),
                       coordinate2=_scale_to_mai(
                           required_coord(args.get("coordinate"), dimensions=2)))["function"]]
        if name == "type":
            return [mk(action="type", text=args.get("text", ""))["function"]]
        if name == "system_button":
            btn = args.get("button", "")
            if btn not in ("Back", "Home", "Menu", "Enter"):
                raise ValueError(
                    f"GUI-Owl has no spelling for system_button={btn!r}"
                )
            return [mk(action="system_button", button=btn)["function"]]
        if name == "wait":
            return [mk(action="wait", time=args.get("duration", 1.0))["function"]]
        if name == "open_app":
            return [mk(action="open", text=args.get("app_name", ""))["function"]]
        if name == "response":
            return [mk(action="answer", text=args.get("text", ""))["function"]]
        if name == "terminate":
            return [mk(action="terminate",
                       status=args.get("status", "success"))["function"]]
        if name == "ask_user":
            return [mk(action="interact", text=args.get("text", ""))["function"]]
        raise ValueError(f"GUI-Owl cannot render Lite action {name!r}")

    # ------------------------------------------------------------------
    # GUI-Owl wire -> Lite
    # ------------------------------------------------------------------

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        *,
        resolution: Any = None,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for tc in agent_tool_calls:
            result.extend(self._convert_single_from_agent(
                tc,
                active_extra_tool_names=active_extra_tool_names,
                active_extra_tool_schemas=active_extra_tool_schemas,
            ))
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(
        self,
        agent_tool_call: dict[str, Any],
        *,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        name = agent_tool_call["name"]
        args = agent_tool_call["arguments"]
        action = args.get("action", "")

        if name != _GUI_OWL_NATIVE_TOOL_NAME:
            admitted = extra_tool_name_and_arguments_are_admitted(
                name, args,
                active_extra_tool_names=active_extra_tool_names,
                active_extra_tool_schemas=active_extra_tool_schemas,
            )
            # 模型丢掉 wrapper 直接用 action 值当工具名(同 mai_ui 语义)。
            if not action and not admitted and name in _action_values(type(self)):
                action = name
            elif not action or admitted:
                return [make_tool_call(name, args)]

        if action == "click":
            return [LiteMobileActionSpace.tap(
                coordinate=_required_from_mai(args.get("coordinate")),
            )]
        if action == "long_press":
            t = args.get("time")
            return [LiteMobileActionSpace.long_press(
                coordinate=_required_from_mai(args.get("coordinate")),
                duration=float(t) if t is not None else None,
            )]
        if action == "swipe":
            return [LiteMobileActionSpace.swipe(
                start_coordinate=_required_from_mai(args.get("coordinate")),
                coordinate=_required_from_mai(
                    args.get("coordinate2"), name="coordinate2"),
            )]
        if action == "type":
            return [LiteMobileActionSpace.type(text=args.get("text", ""))]
        if action == "system_button":
            return [LiteMobileActionSpace.system_button(
                button=args.get("button", ""))]
        if action == "open":
            return [make_tool_call("open_app", {"app_name": args.get("text", "")})]
        if action == "wait":
            t = args.get("time")
            return [LiteMobileActionSpace.wait(
                duration=float(t) if t is not None else 1.0,
            )]
        if action == "answer":
            return [LiteFinishToolSet.response(text=args.get("text", ""))]
        if action == "terminate":
            return [LiteFinishToolSet.terminate(
                status=args.get("status", "success"))]
        if action == "interact":
            return [make_tool_call("ask_user", {"text": args.get("text", "")})]
        if action == "key":
            # canonical 无 adb keyevent 面;交给 env 以 unknown 反馈,
            # 不静默删除(否则模型收不到任何回应)。
            logger.warning("GUI-Owl key action has no Lite counterpart: %s", args)
            return [unknown_wrapper_action_batch(
                LITE_MOBILE_ACTION_BATCH_TOOL_NAME, args)]

        logger.warning("Unknown GUI-Owl action: %s(%s)", action, args)
        return [unknown_wrapper_action_batch(
            LITE_MOBILE_ACTION_BATCH_TOOL_NAME, args)]
