"""Deterministic strong low-fidelity records for restoration v2."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


LOW_FIDELITY_V2_KEYS = (
    "step_id",
    "action_type",
    "action_argument",
    "foreground_app",
    "screen_text_added",
    "screen_text_removed",
    "screen_change",
    "executor_result",
)
LOW_FIDELITY_V2_ACTIONS = (
    "click",
    "long_press",
    "swipe",
    "type",
    "system_button",
    "open",
    "wait",
    "answer",
    "terminate",
)
SCREEN_CHANGE_VALUES = ("none", "low", "medium", "high")
EXECUTOR_RESULT_VALUES = ("accepted", "failed", "unknown")
MAXIMUM_SCREEN_TEXT_TOKENS = 32
SCREEN_CHANGE_THRESHOLDS = (0.005, 0.05, 0.20)
_ACTION_ARGUMENT_PARAMETERS = {
    "click": ("coordinate",),
    "long_press": ("coordinate",),
    "swipe": ("coordinate", "coordinate2"),
    "type": ("text",),
    "system_button": ("button",),
    "open": ("text",),
    "wait": (),
    "answer": ("text",),
    "terminate": ("status",),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_screen_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("screen text must be a string")
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_foreground_app(value: str | None) -> str:
    if value is None:
        return "unknown"
    normalized = normalize_screen_text(value).casefold()
    return normalized or "unknown"


def select_foreground_app(
    *,
    source_event_app_label: str | None,
    executor_package_name: str | None,
) -> str:
    if source_event_app_label is not None and normalize_screen_text(source_event_app_label):
        return normalize_foreground_app(source_event_app_label)
    return normalize_foreground_app(executor_package_name)


@dataclass(frozen=True)
class ScreenTextNode:
    text: str
    top: float
    left: float
    bottom: float
    right: float

    def __post_init__(self) -> None:
        normalized = normalize_screen_text(self.text)
        if not normalized:
            raise ValueError("screen text nodes cannot be empty after normalization")
        values = (self.top, self.left, self.bottom, self.right)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise TypeError("screen text bounds must be numeric")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("screen text bounds must be finite")
        if self.bottom < self.top or self.right < self.left:
            raise ValueError("screen text bounds must be ordered")

    def normalized_tokens(self) -> tuple[str, ...]:
        return tuple(normalize_screen_text(self.text).split(" "))

    def sort_key(self) -> tuple[float, float, float, float, str]:
        return (
            float(self.top),
            float(self.left),
            float(self.bottom),
            float(self.right),
            normalize_screen_text(self.text),
        )


@dataclass(frozen=True)
class ScreenTextDelta:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    added_discarded_count: int
    removed_discarded_count: int


def _spatial_tokens(nodes: Iterable[ScreenTextNode]) -> tuple[str, ...]:
    ordered = sorted(nodes, key=ScreenTextNode.sort_key)
    return tuple(token for node in ordered for token in node.normalized_tokens())


def _ordered_multiset_difference(
    source: Sequence[str],
    reference: Sequence[str],
) -> tuple[str, ...]:
    remaining = Counter(reference)
    result = []
    for token in source:
        if remaining[token] > 0:
            remaining[token] -= 1
        else:
            result.append(token)
    return tuple(result)


def screen_text_delta(
    before: Iterable[ScreenTextNode],
    after: Iterable[ScreenTextNode],
    *,
    maximum_added_tokens: int = MAXIMUM_SCREEN_TEXT_TOKENS,
    maximum_removed_tokens: int = MAXIMUM_SCREEN_TEXT_TOKENS,
) -> ScreenTextDelta:
    if (
        type(maximum_added_tokens) is not int
        or type(maximum_removed_tokens) is not int
        or maximum_added_tokens <= 0
        or maximum_removed_tokens <= 0
    ):
        raise ValueError("screen text token limits must be positive")
    before_tokens = _spatial_tokens(before)
    after_tokens = _spatial_tokens(after)
    added_all = _ordered_multiset_difference(after_tokens, before_tokens)
    removed_all = _ordered_multiset_difference(before_tokens, after_tokens)
    return ScreenTextDelta(
        added=added_all[:maximum_added_tokens],
        removed=removed_all[:maximum_removed_tokens],
        added_discarded_count=max(0, len(added_all) - maximum_added_tokens),
        removed_discarded_count=max(0, len(removed_all) - maximum_removed_tokens),
    )


def screen_change_from_mean_absolute_rgb_difference(value: float) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("mean absolute RGB difference must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("mean absolute RGB difference must be finite and in [0, 1]")
    if value <= SCREEN_CHANGE_THRESHOLDS[0]:
        return "none"
    if value <= SCREEN_CHANGE_THRESHOLDS[1]:
        return "low"
    if value <= SCREEN_CHANGE_THRESHOLDS[2]:
        return "medium"
    return "high"


def mean_absolute_rgb_difference_from_resized_pixels(
    before: Iterable[Sequence[int]],
    after: Iterable[Sequence[int]],
    *,
    expected_pixels: int = 256 * 256,
) -> float:
    if type(expected_pixels) is not int or expected_pixels <= 0:
        raise ValueError("expected_pixels must be a positive integer")
    before_pixels = tuple(tuple(pixel) for pixel in before)
    after_pixels = tuple(tuple(pixel) for pixel in after)
    if len(before_pixels) != expected_pixels or len(after_pixels) != expected_pixels:
        raise ValueError("resized RGB inputs must contain exactly expected_pixels pixels")
    difference = 0
    for left, right in zip(before_pixels, after_pixels, strict=True):
        if len(left) != 3 or len(right) != 3:
            raise ValueError("resized RGB pixels must contain exactly three channels")
        if any(type(channel) is not int or channel < 0 or channel > 255 for channel in (*left, *right)):
            raise ValueError("resized RGB channels must be integers in [0, 255]")
        difference += sum(abs(a - b) for a, b in zip(left, right, strict=True))
    return difference / (expected_pixels * 3 * 255)


def coordinate_bin(coordinate: Sequence[int]) -> str:
    if not isinstance(coordinate, Sequence) or isinstance(coordinate, (str, bytes)):
        raise TypeError("coordinate must be a sequence")
    if len(coordinate) != 2 or any(type(value) is not int for value in coordinate):
        raise ValueError("coordinate must contain exactly two integers")
    if any(value < 0 or value > 999 for value in coordinate):
        raise ValueError("coordinate values must be in [0, 999]")
    return f"coordinate_bin:x{coordinate[0] * 10 // 1000}_y{coordinate[1] * 10 // 1000}"


def swipe_argument(start: Sequence[int], end: Sequence[int]) -> str:
    coordinate_bin(start)
    coordinate_bin(end)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    displacement = max(abs(dx), abs(dy))
    if displacement == 0:
        return "viewport_stationary:zero"
    if abs(dy) >= abs(dx):
        direction = "down" if dy < 0 else "up"
    else:
        direction = "right" if dx < 0 else "left"
    if displacement <= 332:
        displacement_bin = "short"
    elif displacement <= 665:
        displacement_bin = "medium"
    else:
        displacement_bin = "long"
    return f"viewport_{direction}:{displacement_bin}"


def action_argument(arguments: Mapping[str, Any]) -> str:
    action = arguments.get("action")
    if action not in LOW_FIDELITY_V2_ACTIONS:
        raise ValueError(f"unsupported canonical action arguments: {action}")
    expected_keys = {"action", *_ACTION_ARGUMENT_PARAMETERS[action]}
    if set(arguments) != expected_keys:
        raise ValueError(f"{action} summary arguments must contain exactly {sorted(expected_keys)}")
    if action in {"click", "long_press"}:
        return coordinate_bin(arguments["coordinate"])
    if action == "swipe":
        return swipe_argument(arguments["coordinate"], arguments["coordinate2"])
    if action in {"type", "open", "answer"}:
        text = arguments["text"]
        if type(text) is not str:
            raise ValueError("text action arguments must be strings")
        return unicodedata.normalize("NFKC", text)
    if action == "system_button":
        button = arguments["button"]
        if button not in {"Back", "Home", "Enter"}:
            raise ValueError("unsupported v2 system button")
        return str(button)
    if action == "wait":
        return "wait"
    if action == "terminate" and arguments.get("status") == "success":
        return "success"
    raise AssertionError(f"unhandled canonical action arguments: {action}")


def executor_result(executor_accepted: bool | None) -> str:
    if executor_accepted is None:
        return "unknown"
    if type(executor_accepted) is not bool:
        raise TypeError("executor acceptance must be bool or None")
    return "accepted" if executor_accepted else "failed"


def _validate_cross_field_action_argument(action_type: str, value: str) -> None:
    if action_type in {"click", "long_press"}:
        if re.fullmatch(r"coordinate_bin:x[0-9]_y[0-9]", value) is None:
            raise ValueError("spatial action_argument must be a frozen 10x10 coordinate bin")
        return
    if action_type == "swipe":
        moving = re.fullmatch(
            r"viewport_(up|down|left|right):(short|medium|long)",
            value,
        )
        if moving is None and value != "viewport_stationary:zero":
            raise ValueError("swipe action_argument must use the frozen direction/displacement grammar")
        return
    if action_type in {"type", "open", "answer"}:
        return
    if action_type == "system_button" and value in {"Back", "Home", "Enter"}:
        return
    if action_type == "wait" and value == "wait":
        return
    if action_type == "terminate" and value == "success":
        return
    raise ValueError("action_argument does not match action_type")


@dataclass(frozen=True)
class LowFidelityEventV2:
    step_id: int
    action_type: str
    action_argument: str
    foreground_app: str
    screen_text_added: tuple[str, ...]
    screen_text_removed: tuple[str, ...]
    screen_change: str
    executor_result: str

    def __post_init__(self) -> None:
        if type(self.step_id) is not int or self.step_id < 1:
            raise ValueError("step_id must be a positive integer")
        if self.action_type not in LOW_FIDELITY_V2_ACTIONS:
            raise ValueError("action_type must be a canonical GUI-Owl v2 action")
        if not isinstance(self.action_argument, str):
            raise TypeError("action_argument must be a string")
        if self.action_argument != unicodedata.normalize("NFKC", self.action_argument):
            raise ValueError("action_argument must use Unicode NFKC normalization")
        _validate_cross_field_action_argument(self.action_type, self.action_argument)
        if self.foreground_app != normalize_foreground_app(self.foreground_app):
            raise ValueError("foreground_app must use the frozen normalization")
        for name, values in (
            ("screen_text_added", self.screen_text_added),
            ("screen_text_removed", self.screen_text_removed),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple")
            if len(values) > MAXIMUM_SCREEN_TEXT_TOKENS:
                raise ValueError(f"{name} exceeds the frozen 32-token cap")
            if any(not value or value != normalize_screen_text(value) for value in values):
                raise ValueError(f"{name} values must be non-empty normalized strings")
        if self.screen_change not in SCREEN_CHANGE_VALUES:
            raise ValueError("screen_change must use a frozen categorical bin")
        if self.executor_result not in EXECUTOR_RESULT_VALUES:
            raise ValueError("executor_result must be accepted, failed, or unknown")

    def to_ordered_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action_type": self.action_type,
            "action_argument": self.action_argument,
            "foreground_app": self.foreground_app,
            "screen_text_added": list(self.screen_text_added),
            "screen_text_removed": list(self.screen_text_removed),
            "screen_change": self.screen_change,
            "executor_result": self.executor_result,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LowFidelityEventV2":
        if not isinstance(value, Mapping) or set(value) != set(LOW_FIDELITY_V2_KEYS):
            raise ValueError("v2 low-fidelity object must contain exactly the frozen eight fields")
        added = value["screen_text_added"]
        removed = value["screen_text_removed"]
        if not isinstance(added, list) or not isinstance(removed, list):
            raise ValueError("screen text deltas must be JSON arrays")
        return cls(
            step_id=value["step_id"],
            action_type=value["action_type"],
            action_argument=value["action_argument"],
            foreground_app=value["foreground_app"],
            screen_text_added=tuple(added),
            screen_text_removed=tuple(removed),
            screen_change=value["screen_change"],
            executor_result=value["executor_result"],
        )


def serialize_low_fidelity_v2(event: LowFidelityEventV2) -> bytes:
    serialized = json.dumps(
        event.to_ordered_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (serialized + "\n").encode("utf-8")
