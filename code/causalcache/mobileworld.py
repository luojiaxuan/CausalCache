"""MobileWorld task-memory and shared frozen-policy adapters."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    GUI_OWL_V2_1_SYSTEM_PROMPT,
    validate_gui_owl_v2_1_native_messages,
)


MOBILEWORLD_POLICY_REQUEST_SCHEMA_VERSION = (
    "causalcache.mobileworld.policy_request.v1"
)
MOBILEWORLD_POLICY_RESPONSE_SCHEMA_VERSION = (
    "causalcache.mobileworld.policy_response.v1"
)
MOBILEWORLD_SUPPORTED_MEMORY_ARMS = ("summary", "recent", "full")
MOBILEWORLD_SUPPORTED_ACTION_TYPES = frozenset(
    {
        "answer",
        "click",
        "drag",
        "finished",
        "input_text",
        "keyboard_enter",
        "long_press",
        "navigate_back",
        "navigate_home",
        "open_app",
        "wait",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decode_png(encoded: str) -> Any:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("MobileWorld screenshot must be non-empty base64 text")
    from PIL import Image

    image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
    image.load()
    return image.convert("RGB")


@dataclass(frozen=True)
class MobileWorldHistoryEvent:
    """One executed action and the post-action screenshot it produced."""

    step_id: int
    action: Mapping[str, Any]
    screen_changed: bool
    post_screenshot: bytes

    def summary(self) -> dict[str, Any]:
        if type(self.step_id) is not int or self.step_id <= 0:
            raise ValueError("MobileWorld history step ids must be positive integers")
        if not isinstance(self.action, Mapping):
            raise TypeError("MobileWorld history action must be a mapping")
        if not isinstance(self.screen_changed, bool):
            raise TypeError("MobileWorld screen_changed must be boolean")
        if not isinstance(self.post_screenshot, bytes) or not self.post_screenshot:
            raise ValueError("MobileWorld post screenshot must be non-empty bytes")
        return {
            "step_id": self.step_id,
            "action": dict(self.action),
            "screen_changed": self.screen_changed,
            "post_screenshot_sha256": _sha256_bytes(self.post_screenshot),
        }


def select_mobileworld_memory(
    history: Sequence[MobileWorldHistoryEvent],
    *,
    arm: str,
    budget: int,
) -> tuple[int, ...]:
    """Select an ordered at-most-budget subset from executed history."""
    if arm not in MOBILEWORLD_SUPPORTED_MEMORY_ARMS:
        raise ValueError(f"unsupported MobileWorld memory arm: {arm}")
    if type(budget) is not int or budget < 0:
        raise ValueError("MobileWorld memory budget must be non-negative")
    event_ids = tuple(event.step_id for event in history)
    if (
        any(type(step_id) is not int or step_id <= 0 for step_id in event_ids)
        or event_ids != tuple(range(1, len(event_ids) + 1))
    ):
        raise ValueError("MobileWorld history ids must be contiguous from one")
    if arm == "summary" or budget == 0:
        return ()
    if arm == "full":
        return event_ids
    return event_ids[-min(budget, len(event_ids)) :]


def build_mobileworld_policy_request(
    *,
    instruction: str,
    step_id: int,
    screenshot: bytes,
    history: Sequence[MobileWorldHistoryEvent],
    selected_event_step_ids: Sequence[int],
    screen_size: tuple[int, int],
) -> dict[str, Any]:
    """Build one stateless request for a shared MobileWorld policy replica."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("MobileWorld instruction must be non-empty text")
    if type(step_id) is not int or step_id != len(history) + 1:
        raise ValueError("MobileWorld policy step id must follow history")
    if not isinstance(screenshot, bytes) or not screenshot:
        raise ValueError("MobileWorld current screenshot must be non-empty bytes")
    width, height = screen_size
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("MobileWorld screen size must contain positive integers")
    selected_ids = tuple(selected_event_step_ids)
    if (
        any(type(value) is not int for value in selected_ids)
        or selected_ids != tuple(sorted(set(selected_ids)))
    ):
        raise ValueError("MobileWorld selected event ids must be sorted and unique")
    known_ids = {event.step_id for event in history}
    if not set(selected_ids).issubset(known_ids):
        raise ValueError("MobileWorld selected memory refers to an unknown event")
    return {
        "schema_version": MOBILEWORLD_POLICY_REQUEST_SCHEMA_VERSION,
        "task": {
            "instruction": instruction,
            "instruction_sha256": _sha256_bytes(instruction.encode("utf-8")),
        },
        "step_id": step_id,
        "screen_size": [width, height],
        "current_screenshot_png_base64": base64.b64encode(screenshot).decode("ascii"),
        "history": [
            {
                **event.summary(),
                "restored_post_screenshot_png_base64": (
                    base64.b64encode(event.post_screenshot).decode("ascii")
                    if event.step_id in selected_ids
                    else None
                ),
            }
            for event in history
        ],
        "selected_event_step_ids": list(selected_ids),
        "action_schema": {
            "supported_action_types": sorted(MOBILEWORLD_SUPPORTED_ACTION_TYPES),
            "contract": "Return exactly one MobileWorld JSONAction mapping.",
        },
    }


def build_mobileworld_gui_owl_messages(
    request: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Render a MobileWorld request into the pinned GUI-Owl mobile envelope."""
    if request.get("schema_version") != MOBILEWORLD_POLICY_REQUEST_SCHEMA_VERSION:
        raise ValueError("MobileWorld policy request schema version drifted")
    task = request.get("task")
    history = request.get("history")
    selected = request.get("selected_event_step_ids")
    if not isinstance(task, Mapping) or not isinstance(task.get("instruction"), str):
        raise ValueError("MobileWorld policy request lacks its instruction")
    if not isinstance(history, list) or not isinstance(selected, list):
        raise ValueError("MobileWorld policy request lacks history selection")
    selected_ids = set(selected)
    summaries = [
        {
            "step_id": event.get("step_id"),
            "action": event.get("action"),
            "screen_changed": event.get("screen_changed"),
        }
        for event in history
    ]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Generate the next mobile action for this task.\n"
                f"Instruction: {task['instruction']}\n"
                "Previous event summaries: "
                f"{json.dumps(summaries, ensure_ascii=False, separators=(',', ':'))}"
            ),
        }
    ]
    observed_selected: set[int] = set()
    for event in history:
        step_id = event.get("step_id")
        if step_id not in selected_ids:
            continue
        encoded = event.get("restored_post_screenshot_png_base64")
        if encoded is None:
            raise ValueError("selected MobileWorld event lacks its restored screenshot")
        content.extend(
            (
                {"type": "text", "text": f"Restored screenshot after step {step_id}:"},
                {"type": "image", "image": _decode_png(encoded)},
            )
        )
        observed_selected.add(step_id)
    if observed_selected != selected_ids:
        raise ValueError("MobileWorld selected identities and screenshots differ")
    content.extend(
        (
            {"type": "text", "text": "Current observation:"},
            {
                "type": "image",
                "image": _decode_png(request["current_screenshot_png_base64"]),
            },
            {"type": "text", "text": GUI_OWL_V2_1_FINAL_USER_INSTRUCTION},
        )
    )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_V2_1_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]
    validate_gui_owl_v2_1_native_messages(messages)
    return messages


def mobileworld_action_from_gui_owl(
    action: GUIOwlV2Action,
    *,
    screen_size: tuple[int, int],
) -> dict[str, Any]:
    """Translate one canonical GUI-Owl mobile action to MobileWorld JSONAction."""
    if not isinstance(action, GUIOwlV2Action):
        raise TypeError("action must be a GUIOwlV2Action")
    width, height = screen_size

    def pixel(coordinate: tuple[int, int] | None) -> tuple[int, int]:
        if coordinate is None:
            raise ValueError("GUI-Owl action is missing a coordinate")
        return (
            round(coordinate[0] / 999 * (width - 1)),
            round(coordinate[1] / 999 * (height - 1)),
        )

    if action.action in {"click", "long_press"}:
        x, y = pixel(action.coordinate)
        return {"action_type": action.action, "x": x, "y": y}
    if action.action == "swipe":
        start_x, start_y = pixel(action.coordinate)
        end_x, end_y = pixel(action.coordinate2)
        return {
            "action_type": "drag",
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
        }
    if action.action == "type":
        return {"action_type": "input_text", "text": action.text}
    if action.action == "system_button":
        buttons = {
            "Back": "navigate_back",
            "Home": "navigate_home",
            "Enter": "keyboard_enter",
        }
        return {"action_type": buttons[str(action.button)]}
    if action.action == "open":
        return {"action_type": "open_app", "app_name": action.text}
    if action.action == "wait":
        return {"action_type": "wait"}
    if action.action == "answer":
        return {"action_type": "answer", "text": action.text}
    if action.action == "terminate":
        return {"action_type": "finished", "text": action.status}
    raise ValueError(f"unsupported GUI-Owl MobileWorld action: {action.action}")


@dataclass(frozen=True)
class MobileWorldPolicyDecision:
    action: Mapping[str, Any]
    raw_response: Mapping[str, Any]
    latency_seconds: float


class HTTPMobileWorldPolicy:
    """Minimal HTTP client used by the upstream-compatible MobileWorld agent."""

    def __init__(self, endpoint: str, *, timeout_seconds: float = 300.0) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("MobileWorld policy endpoint must use HTTP")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def act(self, request: Mapping[str, Any]) -> MobileWorldPolicyDecision:
        payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                http_request, timeout=self.timeout_seconds
            ) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body)
            except json.JSONDecodeError:
                detail = body
            raise RuntimeError(
                f"MobileWorld policy HTTP {error.code}: "
                f"{json.dumps(detail, ensure_ascii=False, sort_keys=True)}"
            ) from error
        latency = time.perf_counter() - started
        if not isinstance(raw, Mapping):
            raise TypeError("MobileWorld policy response must be a mapping")
        if raw.get("schema_version") != MOBILEWORLD_POLICY_RESPONSE_SCHEMA_VERSION:
            raise ValueError("MobileWorld policy response schema version drifted")
        action = raw.get("action")
        if not isinstance(action, Mapping):
            raise TypeError("MobileWorld policy response action must be a mapping")
        if action.get("action_type") not in MOBILEWORLD_SUPPORTED_ACTION_TYPES:
            raise ValueError("MobileWorld policy returned an unsupported action")
        return MobileWorldPolicyDecision(
            action=dict(action),
            raw_response=raw,
            latency_seconds=latency,
        )
