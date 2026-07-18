"""Pure structural core for variable-history capped restoration labels."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import serialize_gui_owl_v2_1_teacher_target
from causalcache.set_utility_label_schedule import (
    LabelStateRequest,
    MAXIMUM_CANDIDATE_COUNT,
    TeacherOperationBudget,
    build_state_label_schedule,
    enumerate_cardinality_capped_subsets,
)
from causalcache.set_utility_split_validation import SPLIT_ROLES


SUCCESS_ROW_STATUSES = ("measured_kl", "reference_identity_zero")
FAILURE_ROW_STATUS = "state_failure"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty canonical text")
    return value


def _positive_event_ids(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{name} must be an ordered sequence")
    try:
        result = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an ordered sequence") from error
    if (
        not result
        or any(type(value) is not int or value <= 0 for value in result)
        or result != tuple(sorted(result))
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} must be sorted unique positive integers")
    return result


def _coalition(
    values: Sequence[int],
    *,
    candidates: tuple[int, ...],
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError("coalition event ids must be an ordered sequence")
    try:
        coalition = tuple(values)
    except TypeError as error:
        raise TypeError("coalition event ids must be an ordered sequence") from error
    if (
        any(type(value) is not int for value in coalition)
        or coalition != tuple(sorted(coalition))
        or len(set(coalition)) != len(coalition)
    ):
        raise ValueError("coalition event ids must be sorted unique integers")
    if not set(coalition).issubset(candidates):
        raise ValueError("coalition contains an event outside the frozen candidates")
    return coalition


def _sha256_identity(value: str, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _finite_nonnegative(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class CandidateLengthAttempt:
    candidate_event_step_ids: tuple[int, ...]
    processor_input_token_count: int

    def __post_init__(self) -> None:
        if self.candidate_event_step_ids:
            _positive_event_ids(
                self.candidate_event_step_ids,
                "candidate length-attempt event ids",
            )
        if (
            type(self.processor_input_token_count) is not int
            or self.processor_input_token_count <= 0
        ):
            raise ValueError("processor input token count must be a positive integer")


@dataclass(frozen=True)
class FrozenCandidateContext:
    initial_candidate_event_step_ids: tuple[int, ...]
    candidate_event_step_ids: tuple[int, ...]
    dropped_event_step_ids: tuple[int, ...]
    processor_input_token_count: int
    reserved_action_tokens: int
    context_limit: int
    attempts: tuple[CandidateLengthAttempt, ...]

    def __post_init__(self) -> None:
        initial = _positive_event_ids(
            self.initial_candidate_event_step_ids,
            "initial candidate event ids",
        )
        selected = _positive_event_ids(
            self.candidate_event_step_ids,
            "frozen candidate event ids",
        )
        if len(initial) > MAXIMUM_CANDIDATE_COUNT:
            raise ValueError("initial candidate event count exceeds 16")
        if selected != initial[len(self.dropped_event_step_ids) :]:
            raise ValueError("candidate freezing may only remove the oldest event")
        if self.dropped_event_step_ids != initial[: len(self.dropped_event_step_ids)]:
            raise ValueError("dropped candidates are not the oldest initial events")
        if (
            type(self.reserved_action_tokens) is not int
            or self.reserved_action_tokens <= 0
            or type(self.context_limit) is not int
            or self.context_limit <= 0
        ):
            raise ValueError("reserved action tokens and context limit must be positive")
        if not self.attempts or self.attempts[-1].candidate_event_step_ids != selected:
            raise ValueError("candidate length attempts must end at the frozen candidates")
        if self.attempts[0].candidate_event_step_ids != initial:
            raise ValueError("candidate length attempts must start at the initial candidates")
        for previous, current in zip(self.attempts, self.attempts[1:], strict=False):
            if current.candidate_event_step_ids != previous.candidate_event_step_ids[1:]:
                raise ValueError("candidate length attempts must drop one oldest event")
            if (
                previous.processor_input_token_count + self.reserved_action_tokens
                <= self.context_limit
            ):
                raise ValueError("candidate freezing continued after a fitting attempt")
        if self.processor_input_token_count != self.attempts[-1].processor_input_token_count:
            raise ValueError("frozen processor length differs from the final attempt")
        if self.processor_input_token_count + self.reserved_action_tokens > self.context_limit:
            raise ValueError("frozen candidates do not fit the context limit")


class CandidateContextDoesNotFit(ValueError):
    """Raised when no non-empty candidate suffix fits the processor context."""

    def __init__(self, attempts: tuple[CandidateLengthAttempt, ...]) -> None:
        super().__init__("no non-empty candidate suffix fits the processor context")
        self.attempts = attempts


def freeze_recent_candidates(
    eligible_candidate_event_step_ids: Sequence[int],
    *,
    processor_only_length: Callable[[tuple[int, ...]], int],
    reserved_action_tokens: int,
    context_limit: int,
) -> FrozenCandidateContext:
    """Freeze the newest fitting suffix using only processor input length."""
    eligible = _positive_event_ids(
        eligible_candidate_event_step_ids,
        "eligible candidate event ids",
    )
    if not callable(processor_only_length):
        raise TypeError("processor_only_length must be callable")
    if (
        type(reserved_action_tokens) is not int
        or reserved_action_tokens <= 0
        or type(context_limit) is not int
        or context_limit <= 0
        or reserved_action_tokens >= context_limit
    ):
        raise ValueError("context limit must exceed a positive action-token reserve")

    initial = eligible[-MAXIMUM_CANDIDATE_COUNT:]
    candidates = initial
    attempts: list[CandidateLengthAttempt] = []
    while candidates:
        length = processor_only_length(candidates)
        attempt = CandidateLengthAttempt(candidates, length)
        attempts.append(attempt)
        if length + reserved_action_tokens <= context_limit:
            dropped_count = len(initial) - len(candidates)
            return FrozenCandidateContext(
                initial_candidate_event_step_ids=initial,
                candidate_event_step_ids=candidates,
                dropped_event_step_ids=initial[:dropped_count],
                processor_input_token_count=length,
                reserved_action_tokens=reserved_action_tokens,
                context_limit=context_limit,
                attempts=tuple(attempts),
            )
        candidates = candidates[1:]
    raise CandidateContextDoesNotFit(tuple(attempts))


@dataclass(frozen=True)
class UtilityHistoryEvent:
    event_step_id: int
    low_fidelity_summary: LowFidelityEventV2
    high_fidelity_observation_ref: str

    def __post_init__(self) -> None:
        if type(self.event_step_id) is not int or self.event_step_id <= 0:
            raise ValueError("history event step id must be a positive integer")
        if not isinstance(self.low_fidelity_summary, LowFidelityEventV2):
            raise TypeError("history summary must be LowFidelityEventV2")
        if self.low_fidelity_summary.step_id != self.event_step_id:
            raise ValueError("history event and low-fidelity step ids differ")
        _identity(self.high_fidelity_observation_ref, "history observation reference")


@dataclass(frozen=True)
class UtilityQuerySpec:
    split: str
    trajectory_id: str
    source_id: str
    state_id: str
    instruction_app_group_sha256: str
    task_instruction: str
    decision_step_id: int
    history_events: tuple[UtilityHistoryEvent, ...]
    current_equivalent_event_step_id: int
    candidate_context: FrozenCandidateContext
    maximum_labeled_cardinality: int
    current_observation_ref: str
    source_artifact_sha256: str
    request_manifest_sha256: str
    slice_witness_sha256: str

    def __post_init__(self) -> None:
        if self.split not in SPLIT_ROLES:
            raise ValueError(f"utility query split must be one of {SPLIT_ROLES!r}")
        _identity(self.trajectory_id, "trajectory id")
        _identity(self.source_id, "source id")
        _identity(self.state_id, "state id")
        _sha256_identity(
            self.instruction_app_group_sha256,
            "instruction-app group SHA256",
        )
        _identity(self.task_instruction, "task instruction")
        _identity(self.current_observation_ref, "current observation reference")
        for value, name in (
            (self.source_artifact_sha256, "source artifact SHA256"),
            (self.request_manifest_sha256, "request manifest SHA256"),
            (self.slice_witness_sha256, "slice witness SHA256"),
        ):
            _sha256_identity(value, name)
        if not isinstance(self.history_events, tuple) or not self.history_events:
            raise ValueError("utility query history must be a non-empty tuple")
        if any(not isinstance(event, UtilityHistoryEvent) for event in self.history_events):
            raise TypeError("utility query history contains an invalid event")
        history_ids = tuple(event.event_step_id for event in self.history_events)
        _positive_event_ids(history_ids, "utility query history event ids")
        if (
            type(self.current_equivalent_event_step_id) is not int
            or self.current_equivalent_event_step_id != history_ids[-1]
        ):
            raise ValueError(
                "current-equivalent event must be the final history event"
            )
        if (
            type(self.decision_step_id) is not int
            or self.decision_step_id != self.current_equivalent_event_step_id + 1
        ):
            raise ValueError(
                "decision step must immediately follow the current-equivalent event"
            )
        if (
            self.history_events[-1].high_fidelity_observation_ref
            != self.current_observation_ref
        ):
            raise ValueError(
                "current observation must equal the final history post-state"
            )
        if not isinstance(self.candidate_context, FrozenCandidateContext):
            raise TypeError("candidate context must be FrozenCandidateContext")
        eligible_candidates = history_ids[:-1]
        if not eligible_candidates:
            raise ValueError("utility query needs a candidate before the current event")
        expected_initial = eligible_candidates[-MAXIMUM_CANDIDATE_COUNT:]
        if self.candidate_context.initial_candidate_event_step_ids != expected_initial:
            raise ValueError(
                "candidate context must start from the newest 16 non-current events"
            )
        LabelStateRequest(
            state_id=self.state_id,
            candidate_event_ids=self.candidate_event_step_ids,
            maximum_cardinality=self.maximum_labeled_cardinality,
        )

    @property
    def candidate_event_step_ids(self) -> tuple[int, ...]:
        return self.candidate_context.candidate_event_step_ids


@dataclass(frozen=True)
class PromptHistoryEvent:
    event_step_id: int
    low_fidelity_summary: LowFidelityEventV2
    is_frozen_candidate: bool
    high_fidelity_observation_ref: str | None

    def __post_init__(self) -> None:
        if type(self.event_step_id) is not int or self.event_step_id <= 0:
            raise ValueError("prompt history event step id must be a positive integer")
        if not isinstance(self.low_fidelity_summary, LowFidelityEventV2):
            raise TypeError("prompt history summary must be LowFidelityEventV2")
        if self.low_fidelity_summary.step_id != self.event_step_id:
            raise ValueError("prompt event and summary step ids differ")
        if type(self.is_frozen_candidate) is not bool:
            raise TypeError("prompt candidate flag must be bool")
        if self.high_fidelity_observation_ref is not None:
            _identity(
                self.high_fidelity_observation_ref,
                "prompt history observation reference",
            )
            if not self.is_frozen_candidate:
                raise ValueError("a non-candidate history event cannot be high fidelity")


@dataclass(frozen=True)
class MixedFidelityPromptPlan:
    state_id: str
    task_instruction: str
    restored_event_step_ids: tuple[int, ...]
    history_events: tuple[PromptHistoryEvent, ...]
    current_observation_ref: str
    current_observation_high_fidelity: bool = True

    def __post_init__(self) -> None:
        _identity(self.state_id, "prompt-plan state id")
        _identity(self.task_instruction, "prompt-plan task instruction")
        _identity(self.current_observation_ref, "prompt-plan current observation")
        if self.current_observation_high_fidelity is not True:
            raise ValueError("prompt-plan current observation must be high fidelity")
        if not isinstance(self.history_events, tuple) or not self.history_events:
            raise ValueError("prompt plan must retain a non-empty history tuple")
        if any(not isinstance(event, PromptHistoryEvent) for event in self.history_events):
            raise TypeError("prompt plan contains an invalid history event")
        history_ids = tuple(event.event_step_id for event in self.history_events)
        _positive_event_ids(history_ids, "prompt-plan history event ids")
        candidate_ids = tuple(
            event.event_step_id
            for event in self.history_events
            if event.is_frozen_candidate
        )
        if candidate_ids:
            _positive_event_ids(candidate_ids, "prompt-plan candidate event ids")
        restored = _coalition(
            self.restored_event_step_ids,
            candidates=candidate_ids,
        )
        high_fidelity = tuple(
            event.event_step_id
            for event in self.history_events
            if event.high_fidelity_observation_ref is not None
        )
        if high_fidelity != restored:
            raise ValueError("prompt-plan high-fidelity history differs from its coalition")

    @property
    def summary_event_step_ids(self) -> tuple[int, ...]:
        return tuple(event.event_step_id for event in self.history_events)

    @property
    def high_fidelity_history_event_step_ids(self) -> tuple[int, ...]:
        return tuple(
            event.event_step_id
            for event in self.history_events
            if event.high_fidelity_observation_ref is not None
        )


def build_mixed_fidelity_prompt_plan(
    query: UtilityQuerySpec,
    restored_event_step_ids: Sequence[int],
) -> MixedFidelityPromptPlan:
    """Keep every summary and upgrade only restored frozen candidates."""
    if not isinstance(query, UtilityQuerySpec):
        raise TypeError("query must be UtilityQuerySpec")
    restored = _coalition(
        restored_event_step_ids,
        candidates=query.candidate_event_step_ids,
    )
    candidates = frozenset(query.candidate_event_step_ids)
    upgraded = frozenset(restored)
    events = tuple(
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
    )
    plan = MixedFidelityPromptPlan(
        state_id=query.state_id,
        task_instruction=query.task_instruction,
        restored_event_step_ids=restored,
        history_events=events,
        current_observation_ref=query.current_observation_ref,
    )
    if plan.summary_event_step_ids != tuple(
        event.event_step_id for event in query.history_events
    ):
        raise RuntimeError("mixed-fidelity plan dropped a history summary")
    if plan.high_fidelity_history_event_step_ids != restored:
        raise RuntimeError("mixed-fidelity plan upgraded a non-coalition event")
    if not plan.current_observation_high_fidelity:
        raise RuntimeError("current observation must remain high fidelity")
    return plan


def prompt_plan_sha256(plan: MixedFidelityPromptPlan) -> str:
    if not isinstance(plan, MixedFidelityPromptPlan):
        raise TypeError("plan must be MixedFidelityPromptPlan")
    return _sha256(
        {
            "state_id": plan.state_id,
            "task_instruction": plan.task_instruction,
            "restored_event_step_ids": list(plan.restored_event_step_ids),
            "history_events": [
                {
                    "event_step_id": event.event_step_id,
                    "low_fidelity_summary": event.low_fidelity_summary.to_ordered_dict(),
                    "is_frozen_candidate": event.is_frozen_candidate,
                    "high_fidelity_observation_ref": (
                        event.high_fidelity_observation_ref
                    ),
                }
                for event in plan.history_events
            ],
            "current_observation_ref": plan.current_observation_ref,
            "current_observation_high_fidelity": (
                plan.current_observation_high_fidelity
            ),
        }
    )


@dataclass(frozen=True)
class TeacherTargetRecord:
    state_id: str
    reference_restored_event_step_ids: tuple[int, ...]
    first_generated_action: GUIOwlV2Action
    repeated_generated_action: GUIOwlV2Action
    serialized_teacher_target: str
    reference_prompt_sha256: str
    reference_input_token_count: int
    action_token_count: int
    vocabulary_size: int
    reference_repeat_kl: float

    def __post_init__(self) -> None:
        _identity(self.state_id, "teacher-target state id")
        _positive_event_ids(
            self.reference_restored_event_step_ids,
            "reference restored event ids",
        )
        if not isinstance(self.first_generated_action, GUIOwlV2Action) or not isinstance(
            self.repeated_generated_action, GUIOwlV2Action
        ):
            raise TypeError("generated actions must be canonical GUIOwlV2Action values")
        if self.first_generated_action != self.repeated_generated_action:
            raise ValueError("the two reference generations must produce the same action")
        expected_target = serialize_gui_owl_v2_1_teacher_target(
            self.first_generated_action
        )
        if self.serialized_teacher_target != expected_target:
            raise ValueError("serialized teacher target differs from the canonical action")
        _sha256_identity(self.reference_prompt_sha256, "reference prompt SHA256")
        for value, name in (
            (self.reference_input_token_count, "reference input token count"),
            (self.action_token_count, "action token count"),
            (self.vocabulary_size, "vocabulary size"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _finite_nonnegative(self.reference_repeat_kl, "reference repeat KL")

    @property
    def canonical_action(self) -> GUIOwlV2Action:
        return self.first_generated_action


@dataclass(frozen=True)
class CappedRestorationRow:
    state_id: str
    candidate_event_step_ids: tuple[int, ...]
    maximum_labeled_cardinality: int
    coalition_event_step_ids: tuple[int, ...]
    status: str
    distance: float | None
    prompt_plan_sha256: str | None
    candidate_teacher_forward_count: int
    candidate_kl_measurement_count: int
    failure_stage: str | None = None
    failure_class: str | None = None

    def __post_init__(self) -> None:
        _identity(self.state_id, "restoration-row state id")
        candidates = _positive_event_ids(
            self.candidate_event_step_ids,
            "restoration-row candidate event ids",
        )
        LabelStateRequest(
            state_id=self.state_id,
            candidate_event_ids=candidates,
            maximum_cardinality=self.maximum_labeled_cardinality,
        )
        coalition = _coalition(
            self.coalition_event_step_ids,
            candidates=candidates,
        )
        if len(coalition) > self.maximum_labeled_cardinality:
            raise ValueError("restoration row exceeds its labeled-cardinality cap")
        if self.status == "measured_kl":
            _finite_nonnegative(self.distance, "measured restoration distance")
            _sha256_identity(self.prompt_plan_sha256, "candidate prompt SHA256")
            if (
                self.candidate_teacher_forward_count != 1
                or self.candidate_kl_measurement_count != 1
                or self.failure_stage is not None
                or self.failure_class is not None
            ):
                raise ValueError("measured restoration-row receipts drifted")
        elif self.status == "reference_identity_zero":
            if (
                self.distance != 0.0
                or self.candidate_teacher_forward_count != 0
                or self.candidate_kl_measurement_count != 0
                or self.failure_stage is not None
                or self.failure_class is not None
            ):
                raise ValueError("reference-identity row must be a zero-cost zero distance")
            _sha256_identity(self.prompt_plan_sha256, "reference prompt SHA256")
        elif self.status == FAILURE_ROW_STATUS:
            if (
                self.distance is not None
                or self.prompt_plan_sha256 is not None
                or self.candidate_teacher_forward_count != 0
                or self.candidate_kl_measurement_count != 0
            ):
                raise ValueError("failed state rows cannot publish partial measurements")
            _identity(self.failure_stage, "failure stage")
            _identity(self.failure_class, "failure class")
        else:
            raise ValueError("unsupported capped restoration-row status")


def build_successful_capped_rows(
    query: UtilityQuerySpec,
    measured_distances: Mapping[Sequence[int], Real],
) -> tuple[CappedRestorationRow, ...]:
    """Build exact capped rows, deriving an in-cap full reference as D=0."""
    if not isinstance(query, UtilityQuerySpec):
        raise TypeError("query must be UtilityQuerySpec")
    if not isinstance(measured_distances, Mapping):
        raise TypeError("measured distances must be a coalition mapping")
    candidates = query.candidate_event_step_ids
    expected = enumerate_cardinality_capped_subsets(
        candidates,
        maximum_cardinality=query.maximum_labeled_cardinality,
    )
    identity = candidates if len(candidates) <= query.maximum_labeled_cardinality else None
    normalized: dict[tuple[int, ...], float] = {}
    for raw_coalition, distance in measured_distances.items():
        coalition = _coalition(raw_coalition, candidates=candidates)
        if len(coalition) > query.maximum_labeled_cardinality:
            raise ValueError("measured distance coalition exceeds the label cap")
        if coalition == identity:
            raise ValueError("full-candidate identity distance must be derived, not measured")
        if coalition in normalized:
            raise ValueError("measured distances contain a duplicate normalized coalition")
        normalized[coalition] = _finite_nonnegative(
            distance,
            "measured restoration distance",
        )
    required_measured = set(expected) - ({identity} if identity is not None else set())
    if set(normalized) != required_measured:
        raise ValueError("measured distances do not match the capped non-identity inventory")

    rows: list[CappedRestorationRow] = []
    for coalition in expected:
        plan_sha = prompt_plan_sha256(
            build_mixed_fidelity_prompt_plan(query, coalition)
        )
        if coalition == identity:
            rows.append(
                CappedRestorationRow(
                    state_id=query.state_id,
                    candidate_event_step_ids=candidates,
                    maximum_labeled_cardinality=query.maximum_labeled_cardinality,
                    coalition_event_step_ids=coalition,
                    status="reference_identity_zero",
                    distance=0.0,
                    prompt_plan_sha256=plan_sha,
                    candidate_teacher_forward_count=0,
                    candidate_kl_measurement_count=0,
                )
            )
        else:
            rows.append(
                CappedRestorationRow(
                    state_id=query.state_id,
                    candidate_event_step_ids=candidates,
                    maximum_labeled_cardinality=query.maximum_labeled_cardinality,
                    coalition_event_step_ids=coalition,
                    status="measured_kl",
                    distance=normalized[coalition],
                    prompt_plan_sha256=plan_sha,
                    candidate_teacher_forward_count=1,
                    candidate_kl_measurement_count=1,
                )
            )
    return tuple(rows)


def build_failed_capped_rows(
    query: UtilityQuerySpec,
    *,
    failure_stage: str,
    failure_class: str,
) -> tuple[CappedRestorationRow, ...]:
    """Retain every planned row identity while publishing no partial label."""
    if not isinstance(query, UtilityQuerySpec):
        raise TypeError("query must be UtilityQuerySpec")
    _identity(failure_stage, "failure stage")
    _identity(failure_class, "failure class")
    return tuple(
        CappedRestorationRow(
            state_id=query.state_id,
            candidate_event_step_ids=query.candidate_event_step_ids,
            maximum_labeled_cardinality=query.maximum_labeled_cardinality,
            coalition_event_step_ids=coalition,
            status=FAILURE_ROW_STATUS,
            distance=None,
            prompt_plan_sha256=None,
            candidate_teacher_forward_count=0,
            candidate_kl_measurement_count=0,
            failure_stage=failure_stage,
            failure_class=failure_class,
        )
        for coalition in enumerate_cardinality_capped_subsets(
            query.candidate_event_step_ids,
            maximum_cardinality=query.maximum_labeled_cardinality,
        )
    )


def _sum_operation_budgets(
    queries: Sequence[UtilityQuerySpec],
) -> TeacherOperationBudget:
    fields = tuple(TeacherOperationBudget.__dataclass_fields__)
    values = {field: 0 for field in fields}
    for query in queries:
        operations = build_state_label_schedule(
            LabelStateRequest(
                state_id=query.state_id,
                candidate_event_ids=query.candidate_event_step_ids,
                maximum_cardinality=query.maximum_labeled_cardinality,
            )
        ).operations
        for field in fields:
            values[field] += getattr(operations, field)
    return TeacherOperationBudget(**values)


@dataclass(frozen=True)
class ValidatedCappedRestorationBatch:
    maximum_reference_repeat_kl: float
    state_denominator: int
    successful_state_count: int
    failed_state_count: int
    raw_row_denominator: int
    successful_label_row_count: int
    failed_row_count: int
    failed_state_ids: tuple[str, ...]
    planned_operations: TeacherOperationBudget
    committed_success_operations: TeacherOperationBudget


def validate_capped_restoration_batch(
    queries: Sequence[UtilityQuerySpec],
    teacher_targets: Sequence[TeacherTargetRecord],
    rows: Sequence[CappedRestorationRow],
    *,
    maximum_reference_repeat_kl: float,
) -> ValidatedCappedRestorationBatch:
    """Validate exact inventory, state-atomic publication, and operation arithmetic."""
    _finite_nonnegative(
        maximum_reference_repeat_kl,
        "maximum reference repeat KL",
    )
    if (
        isinstance(queries, (str, bytes, bytearray, Mapping))
        or not isinstance(queries, Sequence)
        or not queries
        or any(not isinstance(query, UtilityQuerySpec) for query in queries)
    ):
        raise ValueError("queries must be a non-empty UtilityQuerySpec sequence")
    if (
        isinstance(teacher_targets, (str, bytes, bytearray, Mapping))
        or not isinstance(teacher_targets, Sequence)
        or any(not isinstance(target, TeacherTargetRecord) for target in teacher_targets)
    ):
        raise TypeError("teacher targets must be a TeacherTargetRecord sequence")
    if (
        isinstance(rows, (str, bytes, bytearray, Mapping))
        or not isinstance(rows, Sequence)
        or any(not isinstance(row, CappedRestorationRow) for row in rows)
    ):
        raise TypeError("restoration rows must be a CappedRestorationRow sequence")

    query_by_state = {query.state_id: query for query in queries}
    if len(query_by_state) != len(queries):
        raise ValueError("utility query batch contains a duplicate state id")
    target_by_state = {target.state_id: target for target in teacher_targets}
    if len(target_by_state) != len(teacher_targets):
        raise ValueError("teacher target batch contains a duplicate state id")
    unknown_targets = set(target_by_state) - set(query_by_state)
    if unknown_targets:
        raise ValueError("teacher target batch contains an unknown state")

    rows_by_state: dict[str, dict[tuple[int, ...], CappedRestorationRow]] = {
        state_id: {} for state_id in query_by_state
    }
    for row in rows:
        if row.state_id not in rows_by_state:
            raise ValueError("restoration rows contain an unknown state")
        state_rows = rows_by_state[row.state_id]
        if row.coalition_event_step_ids in state_rows:
            raise ValueError("restoration rows contain a duplicate state-coalition row")
        state_rows[row.coalition_event_step_ids] = row

    successful_queries: list[UtilityQuerySpec] = []
    failed_state_ids: list[str] = []
    successful_rows = 0
    failed_rows = 0
    for query in queries:
        expected = enumerate_cardinality_capped_subsets(
            query.candidate_event_step_ids,
            maximum_cardinality=query.maximum_labeled_cardinality,
        )
        state_rows = rows_by_state[query.state_id]
        if set(state_rows) != set(expected) or len(state_rows) != len(expected):
            raise ValueError("one state does not retain its exact capped row inventory")
        canonical_rows = tuple(state_rows[coalition] for coalition in expected)
        for row in canonical_rows:
            if (
                row.candidate_event_step_ids != query.candidate_event_step_ids
                or row.maximum_labeled_cardinality
                != query.maximum_labeled_cardinality
            ):
                raise ValueError("restoration row disagrees with its utility query")
        statuses = {row.status for row in canonical_rows}
        is_failure = statuses == {FAILURE_ROW_STATUS}
        is_success = statuses.issubset(SUCCESS_ROW_STATUSES)
        if not is_failure and not is_success:
            raise ValueError("one state must publish atomically as success or failure")

        target = target_by_state.get(query.state_id)
        if is_failure:
            if target is not None:
                raise ValueError("failed states cannot publish a teacher target")
            failure_identities = {
                (row.failure_stage, row.failure_class) for row in canonical_rows
            }
            if len(failure_identities) != 1:
                raise ValueError("failed state rows disagree on failure identity")
            failed_state_ids.append(query.state_id)
            failed_rows += len(canonical_rows)
            continue

        if target is None:
            raise ValueError("successful states require exactly one teacher target")
        if target.reference_repeat_kl > maximum_reference_repeat_kl:
            raise ValueError("reference repeat KL exceeds the frozen stability threshold")
        reference_plan = build_mixed_fidelity_prompt_plan(
            query,
            query.candidate_event_step_ids,
        )
        if (
            target.reference_restored_event_step_ids
            != query.candidate_event_step_ids
            or target.reference_prompt_sha256 != prompt_plan_sha256(reference_plan)
            or target.reference_input_token_count
            != query.candidate_context.processor_input_token_count
        ):
            raise ValueError("teacher target differs from the frozen full-candidate reference")
        identity = (
            query.candidate_event_step_ids
            if len(query.candidate_event_step_ids)
            <= query.maximum_labeled_cardinality
            else None
        )
        for row in canonical_rows:
            expected_sha = prompt_plan_sha256(
                build_mixed_fidelity_prompt_plan(
                    query,
                    row.coalition_event_step_ids,
                )
            )
            if row.prompt_plan_sha256 != expected_sha:
                raise ValueError("restoration row prompt differs from the frozen plan")
            if row.coalition_event_step_ids == identity:
                if row.status != "reference_identity_zero":
                    raise ValueError("in-cap full candidates must use the D=0 identity row")
            elif row.status != "measured_kl":
                raise ValueError("non-identity capped rows must use measured KL")
        successful_queries.append(query)
        successful_rows += len(canonical_rows)

    successful_ids = {query.state_id for query in successful_queries}
    if set(target_by_state) != successful_ids:
        raise ValueError("teacher target states differ from successful label states")
    planned = _sum_operation_budgets(queries)
    committed = _sum_operation_budgets(successful_queries)
    return ValidatedCappedRestorationBatch(
        maximum_reference_repeat_kl=float(maximum_reference_repeat_kl),
        state_denominator=len(queries),
        successful_state_count=len(successful_queries),
        failed_state_count=len(failed_state_ids),
        raw_row_denominator=len(rows),
        successful_label_row_count=successful_rows,
        failed_row_count=failed_rows,
        failed_state_ids=tuple(sorted(failed_state_ids)),
        planned_operations=planned,
        committed_success_operations=committed,
    )


__all__ = [
    "CandidateContextDoesNotFit",
    "CandidateLengthAttempt",
    "CappedRestorationRow",
    "FrozenCandidateContext",
    "MixedFidelityPromptPlan",
    "PromptHistoryEvent",
    "TeacherTargetRecord",
    "UtilityHistoryEvent",
    "UtilityQuerySpec",
    "ValidatedCappedRestorationBatch",
    "build_failed_capped_rows",
    "build_mixed_fidelity_prompt_plan",
    "build_successful_capped_rows",
    "freeze_recent_candidates",
    "prompt_plan_sha256",
    "validate_capped_restoration_batch",
]
