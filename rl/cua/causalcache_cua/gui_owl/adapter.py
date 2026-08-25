# note (luojiaxuan): GUI-Owl-1.5 adapter(CausalCache out-of-tree family)。
# 职责:官方(补丁版)消息布局渲染(protocol)+ S 选帧三策略
# (recent/random/learned)+ <tool_call> 解析。与 mai_ui 的关键差异:
# assistant 的 Lite content 保存**全量 raw 文本**(含 <tool_call> 块)——
# 官方 history bubble 渲染的就是 prediction.strip() 原文,拆开再重拼无法
# 保证字节还原。episode 对账键 cc_episode 写进 sample.metadata.others,
# 随 segmenter 流到 slime Sample.metadata["others"]。
from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import io
import json
import logging
import os
import random as _random
import re
import urllib.request
import uuid
from typing import Any

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.adapter.base import BaseAgentAdapter
from lite.agents.models.mai_ui.adapter import _parse_mai_tool_call_json
from lite.agents.types import AgentMessage, AgentStep
from lite.core import LiteSample
from lite.core.messages.content import peel_system_message
from lite.core.messages.final import mark_model_output_error
from lite.core.messages.turns import group_into_turns, truncate_sample_to_turn

from .action_space import GuiOwlMobileActionSpace
from .prompts import SYSTEM_PROMPT
from .protocol import GuiOwlHistoryProtocol, _image_parts

logger = logging.getLogger(__name__)


def _pil_to_b64_png(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@dataclasses.dataclass
class GuiOwlMobileUseAdapter(BaseAgentAdapter, key="gui_owl@mobile@use"):
    """GUI-Owl-1.5 mobile-use adapter。

    Attributes:
        history_n: 官方语义,进 prompt 图像总数上限(含当前帧);
            预算 B = history_n - 1。默认读 CC_HISTORY_N,再退 3。
        frame_policy: recent | random | learned。默认读 CC_FRAME_POLICY。
        selector_url: learned 策略的 selector 服务(默认 CC_SELECTOR_URL)。
        random_seed: random 策略种子盐(默认 CC_SEED)。
    """

    action_space: BaseActionSpace = dataclasses.field(
        default_factory=GuiOwlMobileActionSpace
    )
    protocol: GuiOwlHistoryProtocol = dataclasses.field(
        default_factory=GuiOwlHistoryProtocol
    )
    history_n: int | None = None
    frame_policy: str | None = None
    selector_url: str | None = None
    random_seed: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.history_n is None:
            self.history_n = int(os.environ.get("CC_HISTORY_N", "3"))
        if self.frame_policy is None:
            self.frame_policy = os.environ.get("CC_FRAME_POLICY", "recent")
        if self.selector_url is None:
            self.selector_url = os.environ.get("CC_SELECTOR_URL", "")
        if self.random_seed is None:
            self.random_seed = os.environ.get("CC_SEED", "0")
        self.protocol.history_n = self.history_n
        # note (luojiaxuan): adapter 实例 per-agent 新建(registry 不缓存),
        # 实例生命周期 = 一条 episode → uuid 即 episode 对账键。
        self._episode_id = uuid.uuid4().hex

    # ------------------------------------------------------------------
    # S 选帧
    # ------------------------------------------------------------------

    def _select_frames(
        self, total: int, budget: int, instruction: str,
        history_frames_b64: list[str],
    ) -> list[int] | None:
        """返回 S(含 total)或 None(recent=协议默认后缀窗)。
        与补丁版 _cc_select_frames 语义逐一对应。"""
        policy = self.frame_policy
        budget = max(0, min(budget, total))
        if policy == "recent":
            return None
        if policy == "random":
            key = f"{self.random_seed}|{instruction}|{total}|{total}".encode()
            rng = _random.Random(int(hashlib.md5(key).hexdigest()[:16], 16))
            past = rng.sample(range(total), budget) if total > 0 else []
            return sorted(past) + [total]
        if policy == "learned":
            if total == 0 or budget == 0:
                # 无候选或无预算:S=[total] 与 recent 后缀窗等价,免打服务
                # (空帧曾令服务端 torch.stack 崩 500,smoke 事故 #4)。
                return None
            if not self.selector_url:
                raise RuntimeError("frame_policy=learned 但 selector_url 为空")
            payload = json.dumps({
                "frames_b64": history_frames_b64, "step": total,
                "budget": budget, "task": instruction[:80],
                "episode": self._episode_id,
            }).encode()
            req = urllib.request.Request(
                self.selector_url.rstrip("/") + "/select", data=payload,
                headers={"Content-Type": "application/json"},
            )
            # fail loud:selector 失败不回退 recent,静默回退会污染训练数据面。
            with urllib.request.urlopen(req, timeout=30) as r:
                idx = json.loads(r.read())["indices"]
            return sorted(int(i) for i in idx) + [total]
        raise ValueError(f"unknown frame_policy: {policy!r}")

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        truncated = truncate_sample_to_turn(sample, k)
        _, content = peel_system_message(copy.deepcopy(truncated.messages))
        turns = group_into_turns(content)
        completed = [t for t in turns if t.get("assistant") is not None]
        total = len(completed) if (turns and turns[-1].get("assistant") is None) \
            else max(0, len(turns) - 1)
        budget = max(0, min(self.history_n - 1, total))

        instruction = self.protocol._extract_instruction(turns[0]) if turns else ""

        history_b64: list[str] = []
        if self.frame_policy == "learned" and total > 0:
            for t in range(total):
                img_parts = _image_parts(completed[t]["observations"])
                if not img_parts:
                    raise ValueError(f"turn {t} 无图像,learned 选帧无法特征化")
                img = processed[img_parts[-1]["index"]]
                if img is None:
                    raise ValueError(f"turn {t} 图像未 prepare(index="
                                     f"{img_parts[-1]['index']})")
                history_b64.append(_pil_to_b64_png(img))

        S = self._select_frames(total, budget, instruction, history_b64)
        # CC_TRACE:smoke 验收第 2 条(选帧分布在变)的观测口径,勿删。
        logger.warning(
            "CC_TRACE policy=%s hist_n=%s total=%s S=%s episode=%s",
            self.frame_policy, self.history_n, total,
            S if S is not None else list(range(total - budget, total + 1)),
            self._episode_id,
        )

        # episode 对账键流入 slime Sample.metadata["others"]。
        try:
            sample.metadata.others["cc_episode"] = self._episode_id
            sample.metadata.others["cc_frame_policy"] = self.frame_policy
        except Exception:  # noqa: BLE001 — metadata 缺席时不阻塞渲染
            pass

        self.protocol.pending_keep_frames = S
        messages = self.protocol.process_messages(truncated.messages)

        result: AgentStep = [{
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        }]
        for msg in messages:
            result.append(self.convert_message_to_agent(msg))
        return result

    # ------------------------------------------------------------------
    # per-message conversion
    # ------------------------------------------------------------------

    def _convert_message_to_agent(
        self,
        message: Any,
        **kwargs,
    ) -> AgentMessage:
        """protocol 已把历史重建为纯 text/image part 消息,这里做直通;
        防御性剥掉结构化字段(不应出现)。"""
        result = copy.deepcopy(message)
        result.pop("tool_calls", None)
        result.pop("reasoning_content", None)
        return result

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> Any:
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result
        if "tool_calls" in result:
            converted = self._route_agent_tool_calls_to_lite(result["tool_calls"])
            if result["tool_calls"] and not converted:
                mark_model_output_error(
                    result,
                    "tool call did not satisfy the active tool schema or "
                    "native action grammar",
                )
            result["tool_calls"] = converted
        # content 保持全量 raw 文本(官方 history bubble = prediction 原文)。
        return result

    # ------------------------------------------------------------------
    # raw parse
    # ------------------------------------------------------------------

    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs,
    ) -> AgentMessage:
        """提取 <tool_call>{json}</tool_call>;content 保留**全量原文**
        (含 tool_call 块与 Action 行),供历史 bubble 与折叠文本复刻。"""
        result: AgentMessage = {"role": "assistant"}
        tool_calls: list[dict[str, Any]] = []
        for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>",
                             response, re.DOTALL):
            tc_json = _parse_mai_tool_call_json(m.group(1).strip())
            if tc_json is None:
                continue
            name, arguments = tc_json.get("name"), tc_json.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, dict):
                logger.warning("Ignoring non-flat GUI-Owl tool_call: %s", tc_json)
                continue
            tool_calls.append({"name": name, "arguments": arguments})
        if tool_calls:
            result["tool_calls"] = tool_calls
        elif "<tool_call>" in response or "</tool_call>" in response:
            mark_model_output_error(result, "malformed <tool_call> JSON")
        if response.strip():
            result["content"] = [{"type": "text", "text": response}]
        return result
