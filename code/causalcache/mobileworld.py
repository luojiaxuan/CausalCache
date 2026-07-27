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
from causalcache.policy.gui_owl_official import (
    OFFICIAL_PROTOCOL_ID,
    build_official_messages,
)


MOBILEWORLD_POLICY_REQUEST_SCHEMA_VERSION = (
    "causalcache.mobileworld.policy_request.v2"
)
MOBILEWORLD_POLICY_RESPONSE_SCHEMA_VERSION = (
    "causalcache.mobileworld.policy_response.v2"
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
    """One completed official-protocol policy step."""

    step_id: int
    action: Mapping[str, Any]
    action_text: str
    full_response: str
    policy_parsed: bool
    screen_changed: bool
    observation_screenshot: bytes
    post_screenshot: bytes

    def summary(self) -> dict[str, Any]:
        if type(self.step_id) is not int or self.step_id <= 0:
            raise ValueError("MobileWorld history step ids must be positive integers")
        if not isinstance(self.action, Mapping):
            raise TypeError("MobileWorld history action must be a mapping")
        if not isinstance(self.action_text, str) or not self.action_text.strip():
            raise ValueError("MobileWorld history action text must be non-empty")
        if not isinstance(self.full_response, str) or not self.full_response.strip():
            raise ValueError("MobileWorld history full response must be non-empty")
        if not isinstance(self.policy_parsed, bool):
            raise TypeError("MobileWorld history policy_parsed must be boolean")
        if not isinstance(self.screen_changed, bool):
            raise TypeError("MobileWorld screen_changed must be boolean")
        if (
            not isinstance(self.observation_screenshot, bytes)
            or not self.observation_screenshot
        ):
            raise ValueError(
                "MobileWorld observation screenshot must be non-empty bytes"
            )
        if not isinstance(self.post_screenshot, bytes) or not self.post_screenshot:
            raise ValueError("MobileWorld post screenshot must be non-empty bytes")
        return {
            "step_id": self.step_id,
            "action": dict(self.action),
            "action_text": self.action_text,
            "full_response": self.full_response,
            "policy_parsed": self.policy_parsed,
            "screen_changed": self.screen_changed,
            "observation_screenshot_sha256": _sha256_bytes(
                self.observation_screenshot
            ),
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
                "restored_observation_screenshot_png_base64": (
                    base64.b64encode(event.observation_screenshot).decode("ascii")
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
        "prompt_protocol": OFFICIAL_PROTOCOL_ID,
    }


def build_mobileworld_gui_owl_messages(
    request: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Render a MobileWorld request with the official-faithful mobile envelope."""
    if request.get("schema_version") != MOBILEWORLD_POLICY_REQUEST_SCHEMA_VERSION:
        raise ValueError("MobileWorld policy request schema version drifted")
    task = request.get("task")
    history = request.get("history")
    selected = request.get("selected_event_step_ids")
    if not isinstance(task, Mapping) or not isinstance(task.get("instruction"), str):
        raise ValueError("MobileWorld policy request lacks its instruction")
    if not isinstance(history, list) or not isinstance(selected, list):
        raise ValueError("MobileWorld policy request lacks history selection")
    if request.get("prompt_protocol") != OFFICIAL_PROTOCOL_ID:
        raise ValueError("MobileWorld prompt protocol drifted")
    selected_ids = tuple(selected)
    event_ids = tuple(event.get("step_id") for event in history)
    expected_suffix = event_ids[len(event_ids) - len(selected_ids) :] if selected_ids else ()
    if selected_ids != expected_suffix:
        raise ValueError(
            "official MobileWorld prompt supports only a contiguous recent suffix"
        )
    recent_images: list[Any] = []
    for event in history:
        step_id = event.get("step_id")
        if step_id not in selected_ids:
            continue
        encoded = event.get("restored_observation_screenshot_png_base64")
        if encoded is None:
            raise ValueError(
                "selected MobileWorld event lacks its observation screenshot"
            )
        recent_images.append(_decode_png(encoded))
    action_texts = [event.get("action_text") for event in history]
    full_responses = [event.get("full_response") for event in history]
    if any(not isinstance(value, str) or not value.strip() for value in action_texts):
        raise ValueError("MobileWorld history lacks official action text")
    if any(
        not isinstance(value, str) or not value.strip() for value in full_responses
    ):
        raise ValueError("MobileWorld history lacks full assistant responses")
    return build_official_messages(
        goal=task["instruction"],
        past_action_texts=action_texts,
        past_full_responses=full_responses,
        recent_images=recent_images,
        current_image=_decode_png(request["current_screenshot_png_base64"]),
    )


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
    action_text: str
    full_response: str
    policy_parsed: bool
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
        action_text = raw.get("action_text")
        full_response = raw.get("full_response")
        policy_parsed = raw.get("policy_parsed")
        if not isinstance(action_text, str) or not action_text.strip():
            raise ValueError("MobileWorld policy response lacks action text")
        if not isinstance(full_response, str) or not full_response.strip():
            raise ValueError("MobileWorld policy response lacks full response")
        if not isinstance(policy_parsed, bool):
            raise TypeError("MobileWorld policy response lacks parser status")
        return MobileWorldPolicyDecision(
            action=dict(action),
            action_text=action_text,
            full_response=full_response,
            policy_parsed=policy_parsed,
            raw_response=raw,
            latency_seconds=latency,
        )
