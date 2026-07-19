"""Dense per-step set-utility inputs derived from the existing trajectory artifact."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.restoration_v2_text_backend import prepare_image_bytes
from causalcache.set_utility_features import (
    SetUtilityEventInput,
    SetUtilityFeatureState,
    build_set_utility_feature_state,
)
from causalcache.set_utility_label_producer import (
    MixedFidelityPromptPlan,
    PromptHistoryEvent,
    UtilityHistoryEvent,
)
from causalcache.set_utility_processor_artifacts import ProcessorQueryArtifactRecord
from causalcache.set_utility_processor_prompt import (
    build_set_utility_gui_owl_v2_1_messages,
)
from causalcache.set_utility_processor_substrate import SPLIT_ROLES


DENSE_CANDIDATE_COUNT = 4
DENSE_MINIMUM_DECISION_STEP = DENSE_CANDIDATE_COUNT + 2


@dataclass(frozen=True)
class DenseUtilityQuery:
    """One decision step with the newest four non-current events as candidates."""

    split: str
    trajectory_id: str
    state_id: str
    task_instruction: str
    decision_step_id: int
    history_events: tuple[UtilityHistoryEvent, ...]
    candidate_event_step_ids: tuple[int, ...]
    maximum_labeled_cardinality: int
    current_observation_ref: str

    def __post_init__(self) -> None:
        if self.split not in SPLIT_ROLES:
            raise ValueError("dense query split is invalid")
        if not self.trajectory_id or not self.state_id or not self.task_instruction:
            raise ValueError("dense query identities and instruction must be non-empty")
        history_ids = tuple(event.event_step_id for event in self.history_events)
        if history_ids != tuple(range(1, self.decision_step_id)):
            raise ValueError("dense query history must be an exact decision prefix")
        expected = history_ids[:-1][-DENSE_CANDIDATE_COUNT:]
        if (
            self.decision_step_id < DENSE_MINIMUM_DECISION_STEP
            or self.candidate_event_step_ids != expected
            or len(expected) != DENSE_CANDIDATE_COUNT
        ):
            raise ValueError("dense query candidates must be the newest four events")
        if self.maximum_labeled_cardinality != 2:
            raise ValueError("dense query label cap must be two")
        if self.current_observation_ref != self.history_events[-1].high_fidelity_observation_ref:
            raise ValueError("dense current observation must be the final prefix post-state")


@dataclass(frozen=True)
class DenseUtilityStateInput:
    query: DenseUtilityQuery
    image_payloads: Mapping[str, bytes]
    ocr_records_by_path: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        if not isinstance(self.query, DenseUtilityQuery):
            raise TypeError("dense state requires a dense query")
        by_step = {event.event_step_id: event for event in self.query.history_events}
        expected_images = {
            self.query.current_observation_ref,
            *(by_step[step].high_fidelity_observation_ref for step in self.query.candidate_event_step_ids),
        }
        if set(self.image_payloads) != expected_images or any(
            not isinstance(payload, bytes) or not payload
            for payload in self.image_payloads.values()
        ):
            raise ValueError("dense image inventory is not exactly candidates plus current")
        if not isinstance(self.ocr_records_by_path, Mapping):
            raise TypeError("dense OCR records must be a mapping")
        required_ocr = {
            event.high_fidelity_observation_ref for event in self.query.history_events
        }
        if not required_ocr.issubset(self.ocr_records_by_path):
            raise ValueError("dense OCR inventory does not cover the history prefix")


@dataclass(frozen=True)
class DenseReferenceInput:
    messages: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.messages or any(
            not isinstance(message, Mapping) for message in self.messages
        ):
            raise ValueError("dense reference messages must be a non-empty tuple")


def legacy_available_event_step_ids(*, decision_count: int) -> frozenset[int]:
    """Return event images retained by the old anchor+terminal processor artifact."""
    if type(decision_count) is not int or decision_count < 6:
        raise ValueError("decision count must be at least six")
    anchor_step = 6 if decision_count <= 9 else 10 if decision_count <= 17 else 18
    anchor_current = anchor_step - 1
    anchor_candidates = tuple(range(1, anchor_current))[-16:]
    terminal_candidates = tuple(range(1, decision_count))[-16:]
    return frozenset(
        (*anchor_candidates, anchor_current, *terminal_candidates, decision_count)
    )


def dense_decision_steps(*, decision_count: int) -> tuple[int, ...]:
    if type(decision_count) is not int or decision_count < 6:
        raise ValueError("decision count must be at least six")
    return tuple(range(DENSE_MINIMUM_DECISION_STEP, decision_count + 2))


def dense_required_event_step_ids(decision_step_id: int) -> frozenset[int]:
    if type(decision_step_id) is not int or decision_step_id < DENSE_MINIMUM_DECISION_STEP:
        raise ValueError("dense decision step is too early")
    candidates = tuple(range(1, decision_step_id - 1))[-DENSE_CANDIDATE_COUNT:]
    return frozenset((*candidates, decision_step_id - 1))


def dense_legacy_coverage(*, decision_count: int) -> tuple[int, int, tuple[int, ...]]:
    steps = dense_decision_steps(decision_count=decision_count)
    available = legacy_available_event_step_ids(decision_count=decision_count)
    missing = tuple(
        step
        for step in steps
        if not dense_required_event_step_ids(step).issubset(available)
    )
    return len(steps), len(steps) - len(missing), missing


def derive_dense_states_from_query_pair(
    queries: Sequence[ProcessorQueryArtifactRecord],
    *,
    supplemental_image_payloads: Mapping[str, bytes] | None = None,
) -> tuple[DenseUtilityStateInput, ...]:
    """Expand one legacy anchor+terminal pair into every image-covered step."""
    if len(queries) != 2 or any(
        not isinstance(query, ProcessorQueryArtifactRecord) for query in queries
    ):
        raise ValueError("dense expansion requires one validated two-query pair")
    by_kind = {query.query_kind: query for query in queries}
    if set(by_kind) != {"stratum_anchor", "terminal"}:
        raise ValueError("dense expansion requires anchor and terminal queries")
    anchor = by_kind["stratum_anchor"]
    terminal = by_kind["terminal"]
    if anchor.trajectory_id != terminal.trajectory_id or anchor.role != terminal.role:
        raise ValueError("dense query pair identity drifted")

    available_images: dict[str, bytes] = {}
    for query in (anchor, terminal):
        for reference, payload in query.image_payloads.items():
            previous = available_images.setdefault(reference, payload)
            if previous != payload:
                raise ValueError("duplicate dense image reference has different bytes")
    for reference, payload in (supplemental_image_payloads or {}).items():
        if not isinstance(reference, str) or not isinstance(payload, bytes) or not payload:
            raise ValueError("supplemental dense image payload is invalid")
        previous = available_images.setdefault(reference, payload)
        if previous != payload:
            raise ValueError("supplemental dense image differs from retained bytes")

    master_history = terminal.history_events
    master_ocr = terminal.ocr_records_by_path
    states: list[DenseUtilityStateInput] = []
    for decision_step in dense_decision_steps(decision_count=len(master_history)):
        prefix = master_history[: decision_step - 1]
        candidate_ids = tuple(range(1, decision_step - 1))[-DENSE_CANDIDATE_COUNT:]
        references = {
            prefix[-1]["high_fidelity_observation_ref"],
            *(prefix[step - 1]["high_fidelity_observation_ref"] for step in candidate_ids),
        }
        if not references.issubset(available_images):
            continue
        history = tuple(
            UtilityHistoryEvent(
                event_step_id=event["event_step_id"],
                low_fidelity_summary=LowFidelityEventV2.from_mapping(
                    event["low_fidelity_summary"]
                ),
                high_fidelity_observation_ref=event["high_fidelity_observation_ref"],
            )
            for event in prefix
        )
        query = DenseUtilityQuery(
            split=terminal.role,
            trajectory_id=terminal.trajectory_id,
            state_id=f"{terminal.trajectory_id}:decision:{decision_step:03d}",
            task_instruction=terminal.task_instruction,
            decision_step_id=decision_step,
            history_events=history,
            candidate_event_step_ids=candidate_ids,
            maximum_labeled_cardinality=2,
            current_observation_ref=prefix[-1]["high_fidelity_observation_ref"],
        )
        prefix_ocr_paths = {
            str(event[field])
            for event in prefix
            for field in ("observation_before_ref", "observation_after_ref")
        }
        states.append(
            DenseUtilityStateInput(
                query=query,
                image_payloads={ref: available_images[ref] for ref in sorted(references)},
                ocr_records_by_path={
                    ref: master_ocr[ref] for ref in sorted(prefix_ocr_paths)
                },
            )
        )
    return tuple(states)


def build_dense_prompt_plan(
    query: DenseUtilityQuery,
    restored_event_step_ids: Sequence[int],
) -> MixedFidelityPromptPlan:
    if not isinstance(query, DenseUtilityQuery):
        raise TypeError("dense prompt requires a dense query")
    restored = tuple(restored_event_step_ids)
    if (
        restored != tuple(sorted(restored))
        or len(set(restored)) != len(restored)
        or not set(restored).issubset(query.candidate_event_step_ids)
    ):
        raise ValueError("dense restoration coalition is invalid")
    candidates = frozenset(query.candidate_event_step_ids)
    upgraded = frozenset(restored)
    return MixedFidelityPromptPlan(
        state_id=query.state_id,
        task_instruction=query.task_instruction,
        restored_event_step_ids=restored,
        history_events=tuple(
            PromptHistoryEvent(
                event_step_id=event.event_step_id,
                low_fidelity_summary=event.low_fidelity_summary,
                is_frozen_candidate=event.event_step_id in candidates,
                high_fidelity_observation_ref=(
                    event.high_fidelity_observation_ref
                    if event.event_step_id in upgraded
                    else None
                ),
            )
            for event in query.history_events
        ),
        current_observation_ref=query.current_observation_ref,
    )


def build_dense_messages(
    state: DenseUtilityStateInput,
    plan: MixedFidelityPromptPlan,
    *,
    image_decoder: Any,
) -> DenseReferenceInput:
    if not isinstance(state, DenseUtilityStateInput):
        raise TypeError("dense messages require a dense state")
    expected = build_dense_prompt_plan(state.query, plan.restored_event_step_ids)
    if plan != expected:
        raise ValueError("dense prompt plan differs from its query")
    return DenseReferenceInput(
        messages=tuple(
            build_set_utility_gui_owl_v2_1_messages(
                plan,
                image_bytes_loader=lambda reference: state.image_payloads[reference],
                image_decoder=image_decoder,
            )
        )
    )


def build_dense_feature_state(state: DenseUtilityStateInput) -> SetUtilityFeatureState:
    if not isinstance(state, DenseUtilityStateInput):
        raise TypeError("dense feature construction requires a dense state")
    query = state.query
    history = {event.event_step_id: event for event in query.history_events}
    ocr = {
        path: tuple(record["full_spatial_tokens"])
        for path, record in state.ocr_records_by_path.items()
    }
    prepared = {
        path: prepare_image_bytes(payload).resized_rgb_bytes
        for path, payload in state.image_payloads.items()
    }
    events = tuple(
        SetUtilityEventInput(
            low_fidelity_v2=history[step].low_fidelity_summary,
            post_ocr_spatial_tokens=ocr[history[step].high_fidelity_observation_ref],
            post_resized_rgb_bytes=prepared[history[step].high_fidelity_observation_ref],
        )
        for step in query.candidate_event_step_ids
    )
    return build_set_utility_feature_state(
        source_id=query.trajectory_id,
        state_id=query.state_id,
        decision_step_id=query.decision_step_id,
        instruction=query.task_instruction,
        current_ocr_spatial_tokens=ocr[query.current_observation_ref],
        current_resized_rgb_bytes=prepared[query.current_observation_ref],
        events=events,
        pad_to=DENSE_CANDIDATE_COUNT,
    )


__all__ = [
    "DENSE_CANDIDATE_COUNT",
    "DENSE_MINIMUM_DECISION_STEP",
    "DenseReferenceInput",
    "DenseUtilityQuery",
    "DenseUtilityStateInput",
    "build_dense_feature_state",
    "build_dense_messages",
    "build_dense_prompt_plan",
    "dense_decision_steps",
    "dense_legacy_coverage",
    "dense_required_event_step_ids",
    "derive_dense_states_from_query_pair",
    "legacy_available_event_step_ids",
]
