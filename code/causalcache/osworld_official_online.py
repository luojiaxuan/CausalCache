"""OSWorld 在线服务 → 官方多轮渲染 + 桌面 selector 特征的桥接。

# note (luojiaxuan): HGKV v4 与 selector v4 都在 desktop_official_multiturn 口径上
# 训练/判定,OSWorld 闭环 serve 必须走同一渲染与同一特征代码路径:
#   * prompt:build_desktop_official_messages(gap-fold),事件 j 的 post 帧 =
#     步骤 j+1 的保留轮观测,当前帧恒在末轮 —— 与离线 corpus 完全同构;
#   * 特征:把在线 request 组装成 screening 记录形制(history-action 像素 schema
#     = DesktopAction.to_mapping),直接调用 selector_v4_features 的
#     candidate_features/set_context_features —— 不做第二套实现,witness 伪目标
#     由 CAUSALCACHE_WITNESS_PSEUDO_TARGET=last_action 走与离线判定同一分支。
# 候选池 = [1, t-2] 且保留轮 full_response 非空的事件(builder fail-closed 约束)。
"""

from __future__ import annotations

import base64
import io
from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.agentnet_desktop_official import (
    OfficialStepForms,
    build_desktop_official_messages,
    render_official_action_line,
)

_FOLD_FALLBACK_LINE = "Wait for the screen to update."


def decode_png_b64(payload: str) -> Any:
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def official_forms_from_history(
    history: Sequence[Mapping[str, Any]],
) -> list[OfficialStepForms]:
    """request.history(步骤 1..t-1)→ 官方双形态;不可渲染步折叠兜底、不可保留。"""
    forms: list[OfficialStepForms] = []
    for event in history:
        arguments = event.get("official_arguments")
        line = None
        if isinstance(arguments, Mapping):
            try:
                line = render_official_action_line({"arguments": arguments})
            except (ValueError, KeyError, TypeError):
                line = None
        full = event.get("full_response") or None
        if line is None:
            line = _FOLD_FALLBACK_LINE
            full = None  # 动作行都渲不出来的步不允许作保留轮
        forms.append(OfficialStepForms(
            step_id=int(event["step_id"]),
            action_line=line,
            full_response=full,
            target_unavailable_reason=None if full else "online_unrenderable_step",
        ))
    return forms


def eligible_pool(
    history: Sequence[Mapping[str, Any]],
    forms: Sequence[OfficialStepForms],
) -> list[int]:
    """可作保留轮的事件号:∈[1, t-2]、有截图载荷、保留轮响应非空。

    # note (luojiaxuan): 保留轮展示的是步骤 sid+1 的完整响应(事件 sid 的 post
    # 帧作为步骤 sid+1 的观测,builder 取 steps[sid] 即 step sid+1)——所以
    # 资格检查必须查 forms[sid],不是 forms[sid-1](后者曾放行"下一步响应缺失"
    # 的事件,builder fail-closed 抛 ValueError)。
    """
    total = len(history)
    pool = []
    for event in history:
        sid = int(event["step_id"])
        if sid > total - 1:
            continue  # 事件 t-1 的 post 帧即当前观测
        if event.get("restored_post_screenshot_png_base64") is None:
            continue
        if not forms[sid].full_response:
            continue
        pool.append(sid)
    return pool


def synthetic_selector_record(request: Mapping[str, Any]) -> dict[str, Any]:
    history = request["history"]
    return {
        "step": len(history) + 1,
        "screen_size": list(request["screen_size"]),
        # last_action 模式下不会被读取;显式给一个永不匹配的占位。
        "target_tool_call": {"arguments": {"action": "__none__"}},
        "history": [
            {"step_id": int(e["step_id"]), "action": e.get("action") or {}}
            for e in history
        ],
        "task_id": str(request["task"]["task_id"]),
    }


def dedup_alias(history: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    """post 截图 sha 相同的事件 → 首个事件为 canonical(与离线 duplicates 同构)。"""
    canon_by_sha: dict[str, int] = {}
    alias: dict[int, int] = {}
    for event in history:
        sha = event.get("post_screenshot_sha256")
        sid = int(event["step_id"])
        if not sha:
            continue
        if sha in canon_by_sha:
            alias[sid] = canon_by_sha[sha]
        else:
            canon_by_sha[sha] = sid
    return alias


def build_official_messages_for_request(
    request: Mapping[str, Any], shown_events: Sequence[int]
) -> list[dict[str, Any]]:
    history = request["history"]
    forms = official_forms_from_history(history)
    shown = sorted(int(e) for e in shown_events)
    event_images: dict[int, Any] = {}
    for event in history:
        sid = int(event["step_id"])
        if sid in shown:
            encoded = event.get("restored_post_screenshot_png_base64")
            if encoded is None:
                raise ValueError(f"shown event {sid} lacks its screenshot payload")
            event_images[sid] = decode_png_b64(encoded)
    return build_desktop_official_messages(
        goal=str(request["task"]["instruction"]),
        steps=forms,
        shown_events=shown,
        event_images=event_images,
        current_image=decode_png_b64(request["current_screenshot_png_base64"]),
    )


__all__ = [
    "build_official_messages_for_request",
    "decode_png_b64",
    "dedup_alias",
    "eligible_pool",
    "official_forms_from_history",
    "synthetic_selector_record",
]
