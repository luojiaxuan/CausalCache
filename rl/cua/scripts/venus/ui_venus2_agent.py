# note (luojiaxuan): UI-Venus-2 官方 mobile 多轮推理协议(inclusionAI/UI-Venus@UI-Venus-2
# models/mobile/mobile_example.py)移植到 MobileWorld harness:system prompt、历史消息拼装
# (最近 N_IMG 张历史截图 + 全部历史 assistant 原文)、<think>/<action> 解析、0–999 归一化
# 坐标与动作名映射均照搬;N_IMG 由 CC_VENUS_N_IMG 注入(-1 = 全部历史截图)。
import ast
import base64
import os
import re
from io import BytesIO
from typing import Any

from loguru import logger
from PIL import Image

from mobile_world.agents.base import BaseAgent
from mobile_world.runtime.utils.models import (
    ANSWER,
    CLICK,
    DOUBLE_TAP,
    DRAG,
    FINISHED,
    INPUT_TEXT,
    KEYBOARD_ENTER,
    LONG_PRESS,
    NAVIGATE_BACK,
    NAVIGATE_HOME,
    OPEN_APP,
    UNKNOWN,
    WAIT,
    JSONAction,
)

SYSTEM_PROMPT = """**You are a GUI Agent.** Your role is to analyze the user's task, provide clear and accurate answers to their questions, and execute the task with precise actions.

### Available Actions
You may execute one of the following functions:

- Click(point=(x1, y1))
> Perform a tap action at the specified screen coordinate. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- Drag(start=(x1, y1), end=(x2, y2))
> Perform a drag action by long-pressing at the start coordinate for a few seconds and then dragging to the end coordinate. This is typically used for adjusting app layouts, moving sliders, solving slider captchas, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- Swipe(start=(x1, y1), end=(x2, y2))
> Perform a swipe action by dragging from the start coordinate to the end coordinate. This is typically used for scrolling to find content, switching tabs, pulling down the notification shade, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- DoubleClick(point=(x1, y1))
> Perform a double tap action at the specified screen coordinate. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- LongPress(point=(x1, y1))
> Perform a long-press action at the specified screen coordinate for a certain duration. This can be used to trigger additional options, such as copy, forward, delete, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- Type(content='')
> Enter the specified text into the currently active input field.

- LaunchApp(app='')
> Launch the target app. Use this action when the target app is not currently visible on the screen.

- Wait()
> Wait for the current page, animation, or content to finish loading.

- CallUser(content='')
> Request user takeover or additional information when needed, for example, when there are multiple on-screen options that satisfy the requirement.

- GetScreenshot()
> Take a screenshot and save it to the device's photo album.

- PressBack()
> Return to the previous screen.

- PressHome()
> Return to the system home screen.

- PressEnter()
> Perform an Enter key action.

- PressRecent()
> Open the system recent apps screen.

- Answer(content='')
> Answer the user's questions as requested.

- Finished(content='')
> Mark the task as completed and inform the user of the task execution status.

### Instructions
- Make sure you understand the task goal to avoid wrong actions.
- Make sure you carefully examine the current screenshot. Sometimes the summarized history might not be reliable, over-claiming some effects.
- If additional information is needed during task execution, use `CallUser` to interact with the user.
- Consider exploring the screen by using the `Swipe` action with different directions to reveal additional content.
- To copy text: first select the exact text you want to copy, which usually also brings up the text selection bar, then click the `copy` button in bar.
- To paste text into a text box, first long press the text box, then usually the text selection bar will appear with a `paste` button in it.

### Output Format
<think> your thinking process </think>
<action> the next action </action>

### User Task
{user_task}"""

NORM = 1000


def parse_response(response: str) -> tuple[str, str]:
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    action_match = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""
    action = action_match.group(1).strip() if action_match else ""
    return think, action


def parse_action_call(action: str) -> tuple[str, dict]:
    match = re.match(r"(\w+)\((.*)\)\s*$", action.strip(), re.DOTALL)
    if not match:
        return action.strip(), {}
    name, params_text = match.group(1), match.group(2).strip()
    if not params_text:
        return name, {}
    escaped = params_text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    try:
        tree = ast.parse(f"_({escaped})", mode="eval")
        params = {item.arg: ast.literal_eval(item.value) for item in tree.body.keywords}
    except (SyntaxError, ValueError):
        params = {}
    return name, params


def to_pixel(point, width: int, height: int) -> tuple[int, int]:
    return int(point[0] / NORM * width), int(point[1] / NORM * height)


def venus_action_to_json(name: str, params: dict, width: int, height: int) -> dict:
    if name in ("Click", "DoubleClick", "LongPress"):
        x, y = to_pixel(params.get("point", params.get("box")), width, height)
        return {"action_type": {"Click": CLICK, "DoubleClick": DOUBLE_TAP, "LongPress": LONG_PRESS}[name],
                "x": x, "y": y}
    if name in ("Swipe", "Drag"):
        sx, sy = to_pixel(params["start"], width, height)
        ex, ey = to_pixel(params["end"], width, height)
        return {"action_type": DRAG, "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey}
    if name == "Type":
        return {"action_type": INPUT_TEXT, "text": params.get("content", "")}
    if name == "LaunchApp":
        return {"action_type": OPEN_APP, "app_name": params.get("app", "")}
    if name in ("Wait", "GetScreenshot"):
        return {"action_type": WAIT}
    if name in ("CallUser", "Answer"):
        return {"action_type": ANSWER, "text": params.get("content", "")}
    if name == "Finished":
        return {"action_type": FINISHED, "text": params.get("content", "")}
    if name == "PressBack":
        return {"action_type": NAVIGATE_BACK}
    if name == "PressHome":
        return {"action_type": NAVIGATE_HOME}
    if name == "PressEnter":
        return {"action_type": KEYBOARD_ENTER}
    raise ValueError(f"Unsupported UI-Venus-2 action: {name}({params})")


class Venus2Agent(BaseAgent):
    """UI-Venus-2 mobile agent following the official multi-turn protocol."""

    def __init__(self, llm_base_url: str, model_name: str, api_key: str = "EMPTY",
                 model_config: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        config = {"max_tokens": 16384, "temperature": 0.0, **(model_config or {})}
        self.model_name = model_name
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
        self.n_img = int(os.environ.get("CC_VENUS_N_IMG", "2"))
        self.history: list[dict] = []
        self.build_openai_client(llm_base_url, api_key)

    def initialize_hook(self, instruction: str) -> None:
        logger.info(f"Venus2Agent init n_img={self.n_img} instruction={instruction}")
        self.reset()

    def reset(self) -> None:
        self.history = []

    @staticmethod
    def _image_content(b64: str, label: str) -> list[dict]:
        return [{"type": "text", "text": label},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]

    def _build_messages(self, cur_b64: str) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(user_task=self.instruction)}]
        n_img = len(self.history) if self.n_img < 0 else self.n_img
        image_start = max(0, len(self.history) - n_img)
        for index, turn in enumerate(self.history):
            content: Any = ""
            if n_img > 0 and index >= image_start:
                content = self._image_content(turn["b64"], "History Screenshot:")
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": turn["raw_response"]})
        messages.append({"role": "user", "content": self._image_content(cur_b64, "Current Screenshot:\n")})
        return messages

    def predict(self, observation: dict[str, Any]) -> tuple[str, JSONAction]:
        if self.instruction is None:
            raise ValueError("Agent not initialized. Call initialize() first.")
        shot = observation["screenshot"]
        img = shot if isinstance(shot, Image.Image) else Image.open(BytesIO(shot))
        img = img.convert("RGB")
        width, height = img.width, img.height
        buf = BytesIO()
        img.save(buf, format="PNG")
        cur_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        messages = self._build_messages(cur_b64)
        generated_text = self.openai_chat_completions_create(
            model=self.model_name, messages=messages, retry_times=3,
            max_tokens=self.max_tokens, temperature=self.temperature,
            extra_body={"repetition_penalty": 1.05, "frequency_penalty": 0.3},
        )
        if generated_text is None:
            raise ValueError("LLM call failed after retries.")
        logger.info(f"CC_TRACE venus2 n_img={self.n_img} hist={len(self.history)} "
                    f"imgs={min(len(self.history), len(self.history) if self.n_img < 0 else self.n_img) + 1}")
        logger.info(f"Response: {repr(generated_text[-600:])}")

        think, action = parse_response(generated_text)
        self.history.append({"b64": cur_b64, "raw_response": generated_text})
        name, params = parse_action_call(action)
        try:
            aw = venus_action_to_json(name, params, width, height)
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Venus2 action parse failed: {e} | action={action!r}")
            return generated_text, JSONAction(action_type=UNKNOWN, text=str(e))
        logger.info(f"Action: {action!r} -> {aw!r}")
        return generated_text, JSONAction(**aw)
