"""Full-history query and mixed-fidelity prompt construction."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.set_utility_label_producer import (
    MixedFidelityPromptPlan,
    PromptHistoryEvent,
    UtilityHistoryEvent,
)
from causalcache.set_utility_processor_prompt import (
    build_set_utility_gui_owl_v2_1_messages,
)
from causalcache.set_utility_variable_history import SPLIT_ROLES


@dataclass(frozen=True)
class VariableHistoryTrajectory:
    role: str
    trajectory_id: str
    task_instruction: str
    history_events: tuple[UtilityHistoryEvent, ...]
    image_payloads: Mapping[str, bytes]
    ocr_records_by_path: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        if self.role not in SPLIT_ROLES:
            raise ValueError("variable-history trajectory role is invalid")
        if not self.trajectory_id or not self.task_instruction:
            raise ValueError("trajectory identity and instruction must be non-empty")
        event_ids = tuple(event.event_step_id for event in self.history_events)
        if event_ids != tuple(range(1, len(event_ids) + 1)) or len(event_ids) < 6:
            raise ValueError("trajectory events must be a contiguous prefix of length at least six")
        expected_images = {
            f"images/{self.trajectory_id}/observation-{step:03d}.png"
            for step in range(len(self.history_events) + 1)
        }
        if set(self.image_payloads) != expected_images or any(
            not isinstance(payload, bytes) or not payload
            for payload in self.image_payloads.values()
        ):
            raise ValueError("trajectory images do not exactly cover every observation")
        required_ocr = {
            reference
            for event in self.history_events
            for reference in (
                event.high_fidelity_observation_ref,
                f"images/{self.trajectory_id}/observation-{event.event_step_id - 1:03d}.png",
            )
        }
        if not required_ocr.issubset(self.ocr_records_by_path):
            raise ValueError("trajectory OCR does not cover every event transition")

    @property
    def decision_count(self) -> int:
        return len(self.history_events)


@dataclass(frozen=True)
class VariableHistoryUtilityQuery:
    role: str
    trajectory_id: str
    state_id: str
    task_instruction: str
    decision_step_id: int
    history_events: tuple[UtilityHistoryEvent, ...]
    candidate_event_step_ids: tuple[int, ...]
    current_observation_ref: str

    def __post_init__(self) -> None:
        if self.role not in SPLIT_ROLES:
            raise ValueError("variable-history query role is invalid")
        if self.decision_step_id < 6:
            raise ValueError("variable-history query starts at decision step six")
        event_ids = tuple(event.event_step_id for event in self.history_events)
        expected = tuple(range(1, self.decision_step_id))
        if event_ids != expected or self.candidate_event_step_ids != expected:
            raise ValueError("query candidates must be the complete prior event history")
        expected_state_id = f"{self.trajectory_id}:decision:{self.decision_step_id:03d}"
        if self.state_id != expected_state_id:
            raise ValueError("query state identity drifted")
        if self.current_observation_ref != self.history_events[-1].high_fidelity_observation_ref:
            raise ValueError("query current observation must follow its final prior event")


def trajectory_from_source_row(row: Mapping[str, Any]) -> VariableHistoryTrajectory:
    history_payload = json.loads(row["history_events_json"])
    ocr = json.loads(row["ocr_records_json"])
    trajectory_id = str(row["source_id"])
    image_payloads = {}
    for step, image in enumerate(row["images"]):
        payload = image.get("bytes") if isinstance(image, Mapping) else None
        if not isinstance(payload, bytes):
            raise ValueError("source row contains an invalid image payload")
        image_payloads[
            f"images/{trajectory_id}/observation-{step:03d}.png"
        ] = payload
    events = tuple(
        UtilityHistoryEvent(
            event_step_id=event["event_step_id"],
            low_fidelity_summary=LowFidelityEventV2.from_mapping(
                event["low_fidelity_summary"]
            ),
            high_fidelity_observation_ref=event["high_fidelity_observation_ref"],
        )
        for event in history_payload
    )
    return VariableHistoryTrajectory(
        role=str(row["role"]),
        trajectory_id=trajectory_id,
        task_instruction=str(row["task_instruction"]),
        history_events=events,
        image_payloads=image_payloads,
        ocr_records_by_path=ocr,
    )


def build_variable_history_queries_from_source_row(
    row: Mapping[str, Any],
) -> tuple[VariableHistoryUtilityQuery, ...]:
    history_payload = json.loads(row["history_events_json"])
    trajectory_id = str(row["source_id"])
    events = tuple(
        UtilityHistoryEvent(
            event_step_id=event["event_step_id"],
            low_fidelity_summary=LowFidelityEventV2.from_mapping(
                event["low_fidelity_summary"]
            ),
            high_fidelity_observation_ref=event["high_fidelity_observation_ref"],
        )
        for event in history_payload
    )
    if int(row["decision_count"]) != len(events):
        raise ValueError("source-row decision count differs from its history")
    return tuple(
        VariableHistoryUtilityQuery(
            role=str(row["role"]),
            trajectory_id=trajectory_id,
            state_id=f"{trajectory_id}:decision:{decision_step:03d}",
            task_instruction=str(row["task_instruction"]),
            decision_step_id=decision_step,
            history_events=events[: decision_step - 1],
            candidate_event_step_ids=tuple(range(1, decision_step)),
            current_observation_ref=events[
                decision_step - 2
            ].high_fidelity_observation_ref,
        )
        for decision_step in range(6, len(events) + 2)
    )


def build_variable_history_queries(
    trajectory: VariableHistoryTrajectory,
) -> tuple[VariableHistoryUtilityQuery, ...]:
    if not isinstance(trajectory, VariableHistoryTrajectory):
        raise TypeError("trajectory must be VariableHistoryTrajectory")
    return tuple(
        VariableHistoryUtilityQuery(
            role=trajectory.role,
            trajectory_id=trajectory.trajectory_id,
            state_id=f"{trajectory.trajectory_id}:decision:{decision_step:03d}",
            task_instruction=trajectory.task_instruction,
            decision_step_id=decision_step,
            history_events=trajectory.history_events[: decision_step - 1],
            candidate_event_step_ids=tuple(range(1, decision_step)),
            current_observation_ref=trajectory.history_events[
                decision_step - 2
            ].high_fidelity_observation_ref,
        )
        for decision_step in range(6, trajectory.decision_count + 2)
    )


def build_variable_history_prompt_plan(
    query: VariableHistoryUtilityQuery,
    restored_event_step_ids: Sequence[int],
) -> MixedFidelityPromptPlan:
    if not isinstance(query, VariableHistoryUtilityQuery):
        raise TypeError("query must be VariableHistoryUtilityQuery")
    restored = tuple(restored_event_step_ids)
    if (
        restored != tuple(sorted(restored))
        or len(restored) != len(set(restored))
        or not set(restored).issubset(query.candidate_event_step_ids)
    ):
        raise ValueError("restored coalition is invalid")
    restored_set = frozenset(restored)
    return MixedFidelityPromptPlan(
        state_id=query.state_id,
        task_instruction=query.task_instruction,
        restored_event_step_ids=restored,
        history_events=tuple(
            PromptHistoryEvent(
                event_step_id=event.event_step_id,
                low_fidelity_summary=event.low_fidelity_summary,
                is_frozen_candidate=True,
                high_fidelity_observation_ref=(
                    event.high_fidelity_observation_ref
                    if event.event_step_id in restored_set
                    else None
                ),
            )
            for event in query.history_events
        ),
        current_observation_ref=query.current_observation_ref,
    )


def build_variable_history_messages(
    query: VariableHistoryUtilityQuery,
    restored_event_step_ids: Sequence[int],
    *,
    image_bytes_loader: Callable[[str], bytes],
    image_decoder: Callable[[bytes], Any],
) -> tuple[Mapping[str, Any], ...]:
    plan = build_variable_history_prompt_plan(query, restored_event_step_ids)
    return tuple(
        build_set_utility_gui_owl_v2_1_messages(
            plan,
            image_bytes_loader=image_bytes_loader,
            image_decoder=image_decoder,
        )
    )


__all__ = [
    "VariableHistoryTrajectory",
    "VariableHistoryUtilityQuery",
    "build_variable_history_messages",
    "build_variable_history_prompt_plan",
    "build_variable_history_queries",
    "build_variable_history_queries_from_source_row",
    "trajectory_from_source_row",
]
