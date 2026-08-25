# note (luojiaxuan): GUI-Owl-1.5 历史协议 —— 复刻**打过补丁的**官方
# gui_owl_1_5.predict 消息布局(S 参数化选帧)。ground truth 与偏差记录见
# rl/cua/docs/adapter_contract_notes.md 与 cua_lite_integration_20260825.md。
# 布局(S ⊆ [0..total],恒含 total=当前观察):
#   [user: USER_PROMPT(_WITH_HISTSTEPS){instruction, 折叠文本} + image(S[0])]
#   for k in 0..len(S)-2:
#     [assistant: raw_response(S[k]).strip()]
#     [user: "<tool_response>\n" + (tool_text|"None") + image(S[k+1]) + "\n</tool_response>"]
# 折叠文本(非 S 的历史步,补丁版格式):
#   "Step{i+1}: {add_period_robustly(conclusion_i)}"
#   + ("  Tool response: {t}" if t truthy)
# 已知与官方运行时的偏差(GUI-only smoke 不受影响,记录在案):
#   - ask_user 回复在 CUA-Lite 是 env note 文本,无法区分出
#     "(Ask_user_response)" 前缀与 "  User response:" 折叠两种官方拼法;
#   - 官方 bubble 的 tool 文本来自 MCP 工具结果,GUI-only 任务恒 None,
#     CUA-Lite 的 env note(动作错误反馈等)非空时按原文渲染。
from __future__ import annotations

import copy
import dataclasses
from typing import Any

from lite.agents.core.protocol.base import TurnWindowProtocol
from lite.core import LiteMessage
from lite.core.messages import USER_ROLE, instruction_text

from .prompts import USER_PROMPT_TEMPLATE, USER_PROMPT_WITH_HISTSTEPS_TEMPLATE

# ---------------------------------------------------------------------------
# 官方 helpers 的精确复刻(mobile_world/agents/utils/helpers.py)
# ---------------------------------------------------------------------------

# 官方 END_PUNCTUATIONS 逐项转义抄录(顺序同源码,集合语义)。
_END_PUNCTUATIONS = {
    "。", "！", "？", "…", "；",
    ".", "!", "?", ";",
    "~", "～", "》", "」", "』", "）", ")", "]", "}",
}


def add_period_robustly(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    text = text.strip()
    if not text:
        return text
    if text[-1] in _END_PUNCTUATIONS:
        return text
    chinese_count = sum(1 for ch in text if "一" <= ch <= "鿿")
    english_count = sum(1 for ch in text if ch.isalpha() and ord(ch) < 128)
    return text + ("。" if chinese_count > english_count else ".")


def extract_conclusion(response_text: str) -> str:
    """官方 parse_tagged_text 的 conclusion 分支(容错版:无 tool_call 块时
    取 Action: 后全文)。"""
    action_parts = response_text.strip().split("Action:")
    action_content = action_parts[1] if len(action_parts) > 1 else response_text
    tool_parts = action_content.split("<tool_call>")
    conclusion = tool_parts[0].strip()
    if conclusion.startswith('"') and conclusion.endswith('"'):
        conclusion = conclusion[1:-1]
    return conclusion


# ---------------------------------------------------------------------------
# 消息内容抽取
# ---------------------------------------------------------------------------


def _text_parts(msg: LiteMessage) -> str:
    parts = msg.get("content") or []
    if isinstance(parts, str):
        return parts
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def _image_parts(msgs: list[LiteMessage]) -> list[dict[str, Any]]:
    out = []
    for m in msgs:
        c = m.get("content") or []
        if isinstance(c, list):
            out.extend(p for p in c if p.get("type") == "image")
    return out


def _turn_frame(turn: dict[str, Any]) -> dict[str, Any] | None:
    imgs = _image_parts(turn["observations"])
    return copy.deepcopy(imgs[-1]) if imgs else None


def _turn_tool_text(turn: dict[str, Any], is_first: bool) -> str | None:
    """观察块的 model-visible 文本 = 官方 tool_call_res 语义。首 turn 的文本是
    instruction,不算工具结果。"""
    if is_first:
        return None
    txt = "".join(_text_parts(m) for m in turn["observations"]
                  if m.get("role") == USER_ROLE).strip()
    return txt or None


@dataclasses.dataclass
class GuiOwlHistoryProtocol(TurnWindowProtocol, key="gui_owl.history"):
    """S 参数化的 GUI-Owl 官方(补丁版)历史布局。

    Attributes:
        history_n: 官方语义——进 prompt 的图像总数上限(含当前帧);
            预算 B = history_n - 1。CC_HISTORY_N 的对应物。
        pending_keep_frames: adapter 每次 render 前写入的 S(0-based turn
            索引,含 total);None → recent 后缀窗(官方默认行为)。
            读后即清,防跨 render 泄漏。
    """

    history_n: int = 1
    pending_keep_frames: list[int] | None = None

    def _select_messages(
        self,
        content: list[LiteMessage],
        turns: list[dict[str, Any]],
    ) -> list[LiteMessage]:
        if not turns:
            return content

        # 官方索引系:total = 已完成 turn 数;current = 尾部待决观察。
        completed = [t for t in turns if t.get("assistant") is not None]
        total = len(completed)
        # 尾 turn 应为 pending 观察;若最后 turn 已含 assistant(SFT 全轨迹
        # 渲染的中间步)也允许——current 即最后一个 turn 的观察。
        current_turn = turns[-1] if turns[-1].get("assistant") is None else None
        if current_turn is None:
            # SFT/unroll 路径:render 到第 k turn 时截断样本尾即观察块,
            # 正常不会走到这里;稳妥起见把最后完成 turn 当 current。
            current_turn = turns[-1]
            completed = completed[:-1]
            total = len(completed)

        S = self.pending_keep_frames
        self.pending_keep_frames = None
        budget = max(0, min(self.history_n - 1, total))
        if S is None:
            S = list(range(total - budget, total + 1))
        else:
            S = sorted(set(int(i) for i in S))
            if total not in S:
                S = S + [total]
        if any(i < 0 or i > total for i in S):
            raise ValueError(f"keep_frames 越界: S={S}, total={total}")

        kept = set(S)
        text_idx = [i for i in range(total) if i not in kept]

        def turn_at(i: int) -> dict[str, Any]:
            return current_turn if i == total else completed[i]

        instruction = self._extract_instruction(turns[0])

        # ── 折叠文本(补丁版 _cc_format_steps_by_index)──
        folded_lines = []
        for i in text_idx:
            resp = _text_parts(completed[i]["assistant"])
            line = f"Step{i + 1}: {add_period_robustly(extract_conclusion(resp))}"
            tool_text = _turn_tool_text(completed[i], is_first=(i == 0))
            if tool_text:
                line += f"  Tool response: {tool_text}"
            folded_lines.append(line)

        # ── 首条 user:模板 + image(S[0]) ──
        if folded_lines:
            first_text = USER_PROMPT_WITH_HISTSTEPS_TEMPLATE.format(
                instruction=instruction,
                previous_steps="\n".join(folded_lines),
            )
        else:
            first_text = USER_PROMPT_TEMPLATE.format(instruction=instruction)

        first_frame = _turn_frame(turn_at(S[0]))
        if first_frame is None:
            raise ValueError(f"turn {S[0]} 无图像,无法作为 S[0]")
        result: list[LiteMessage] = [{
            "role": "user",
            "content": [{"type": "text", "text": first_text}, first_frame],
        }]

        # ── 交错 assistant / user(image)对 ──
        for k in range(len(S) - 1):
            t_resp = S[k]
            if t_resp < total:
                resp_text = _text_parts(completed[t_resp]["assistant"]).strip()
                result.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": resp_text}],
                })
            nxt = S[k + 1]
            nxt_turn = turn_at(nxt)
            frame = _turn_frame(nxt_turn)
            if frame is None:
                raise ValueError(f"turn {nxt} 无图像,无法进入 S")
            tool_text = _turn_tool_text(nxt_turn, is_first=(nxt == 0))
            result.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "<tool_response>\n"},
                    {"type": "text", "text": tool_text if tool_text else "None"},
                    frame,
                    {"type": "text", "text": "\n</tool_response>"},
                ],
            })

        return result
