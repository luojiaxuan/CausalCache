"""Versioned event and decision records shared by labeling and evaluation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


class ActionType(str, Enum):
    TAP = "tap"
    LONG_PRESS = "long_press"
    TYPE_TEXT = "type_text"
    SWIPE = "swipe"
    BACK = "back"
    HOME = "home"
    ENTER = "enter"
    WAIT = "wait"
    STOP = "stop"


class ResultStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    NO_CHANGE = "no_change"
    UNKNOWN = "unknown"


def normalize_text(value: str | None, *, case_sensitive: bool = False) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized if case_sensitive else normalized.casefold()


@dataclass(frozen=True)
class ExecutableAction:
    action_type: ActionType
    target: str | None = None
    text_argument: str | None = None
    text_case_sensitive: bool = False

    def canonical_signature(self) -> tuple[str, str | None, str | None]:
        return (
            self.action_type.value,
            normalize_text(self.target),
            normalize_text(self.text_argument, case_sensitive=self.text_case_sensitive),
        )

    def executable_match(self, other: "ExecutableAction") -> bool:
        return self.canonical_signature() == other.canonical_signature()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutableAction":
        return cls(
            action_type=ActionType(str(data["action_type"])),
            target=data.get("target"),
            text_argument=data.get("text_argument"),
            text_case_sensitive=bool(data.get("text_case_sensitive", False)),
        )


@dataclass(frozen=True)
class LowFidelityEvent:
    step_id: int
    action_type: ActionType
    target_text_or_coordinate_bin: str
    deterministic_ui_delta: str
    result_status: ResultStatus

    def __post_init__(self) -> None:
        if self.step_id < 1:
            raise ValueError("step_id must be positive")
        if not self.target_text_or_coordinate_bin:
            raise ValueError("target_text_or_coordinate_bin cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action_type": self.action_type.value,
            "target_text_or_coordinate_bin": self.target_text_or_coordinate_bin,
            "deterministic_ui_delta": self.deterministic_ui_delta,
            "result_status": self.result_status.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LowFidelityEvent":
        return cls(
            step_id=int(data["step_id"]),
            action_type=ActionType(str(data["action_type"])),
            target_text_or_coordinate_bin=str(data["target_text_or_coordinate_bin"]),
            deterministic_ui_delta=str(data["deterministic_ui_delta"]),
            result_status=ResultStatus(str(data["result_status"])),
        )


@dataclass(frozen=True)
class ArchivedEvent:
    trajectory_id: str
    low_fidelity: LowFidelityEvent
    observation_before_uri: str
    observation_after_uri: str
    executed_action: ExecutableAction
    visual_token_cost: int
    visual_embedding_uri: str | None = None

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory_id cannot be empty")
        if self.visual_token_cost <= 0:
            raise ValueError("visual_token_cost must be positive")
        if not self.observation_before_uri or not self.observation_after_uri:
            raise ValueError("archived observations require stable URIs")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArchivedEvent":
        return cls(
            trajectory_id=str(data["trajectory_id"]),
            low_fidelity=LowFidelityEvent.from_dict(data["low_fidelity"]),
            observation_before_uri=str(data["observation_before_uri"]),
            observation_after_uri=str(data["observation_after_uri"]),
            executed_action=ExecutableAction.from_dict(data["executed_action"]),
            visual_token_cost=int(data["visual_token_cost"]),
            visual_embedding_uri=data.get("visual_embedding_uri"),
        )


@dataclass(frozen=True)
class DecisionRecord:
    trajectory_id: str
    decision_step_id: int
    instruction: str
    current_observation_uri: str
    history: tuple[ArchivedEvent, ...]
    full_history_action: ExecutableAction
    validated_action: ExecutableAction
    trajectory_success: bool

    def __post_init__(self) -> None:
        if self.decision_step_id <= 1:
            raise ValueError("decision_step_id must follow at least one historical event")
        if not self.instruction or not self.current_observation_uri:
            raise ValueError("decision records require instruction and current observation")
        step_ids = tuple(event.low_fidelity.step_id for event in self.history)
        if step_ids != tuple(sorted(set(step_ids))):
            raise ValueError("history step ids must be unique and sorted")
        if any(event.trajectory_id != self.trajectory_id for event in self.history):
            raise ValueError("history events must belong to the decision trajectory")
        if any(step_id >= self.decision_step_id for step_id in step_ids):
            raise ValueError("history must precede the decision step")

    def reference_is_valid(self, validation_mode: str) -> bool:
        executable_match = self.full_history_action.executable_match(self.validated_action)
        if validation_mode == "executable_match":
            return executable_match
        if validation_mode == "successful_trajectory":
            return self.trajectory_success
        if validation_mode == "either":
            return executable_match or self.trajectory_success
        raise ValueError(f"unknown validation_mode: {validation_mode}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for event in data["history"]:
            event["low_fidelity"]["action_type"] = event["low_fidelity"]["action_type"].value
            event["low_fidelity"]["result_status"] = event["low_fidelity"]["result_status"].value
            event["executed_action"]["action_type"] = event["executed_action"]["action_type"].value
        data["full_history_action"]["action_type"] = data["full_history_action"]["action_type"].value
        data["validated_action"]["action_type"] = data["validated_action"]["action_type"].value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionRecord":
        return cls(
            trajectory_id=str(data["trajectory_id"]),
            decision_step_id=int(data["decision_step_id"]),
            instruction=str(data["instruction"]),
            current_observation_uri=str(data["current_observation_uri"]),
            history=tuple(ArchivedEvent.from_dict(item) for item in data["history"]),
            full_history_action=ExecutableAction.from_dict(data["full_history_action"]),
            validated_action=ExecutableAction.from_dict(data["validated_action"]),
            trajectory_success=bool(data["trajectory_success"]),
        )
