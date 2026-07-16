"""Confirm-safe arbitrary-coalition inputs for restoration-v2.2 labels."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.data.restoration_v2_screening import (
    EXPECTED_SCREENING_STATES,
    SCREENING_ROLES,
    ScreeningState,
    ValidatedScreeningArtifact,
)
from causalcache.policy.gui_owl_v2_1 import (
    build_gui_owl_v2_1_mixed_fidelity_messages,
    validate_gui_owl_v2_1_native_messages,
)


LABEL_PROMPT_INVENTORY_SCHEMA_VERSION = "1.0.0"


def _validate_artifact(artifact: ValidatedScreeningArtifact) -> None:
    if not isinstance(artifact, ValidatedScreeningArtifact):
        raise TypeError("artifact must be a ValidatedScreeningArtifact")
    if len(artifact.states) != EXPECTED_SCREENING_STATES:
        raise ValueError("v2.2 label inputs require exactly 45 screening states")


def _validate_state(
    artifact: ValidatedScreeningArtifact,
    state: ScreeningState,
) -> tuple[int, ...]:
    if not isinstance(state, ScreeningState):
        raise TypeError("state must be a ScreeningState")
    if state.role not in SCREENING_ROLES:
        raise ValueError("confirm-like state access is forbidden for v2.2 labels")
    if state not in artifact.states:
        raise ValueError("state is not a member of the validated screening artifact")
    candidates = state.candidate_event_step_ids
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or any(
            not isinstance(event_id, int)
            or isinstance(event_id, bool)
            or event_id <= 0
            for event_id in candidates
        )
        or candidates != tuple(sorted(set(candidates)))
    ):
        raise ValueError("candidate event step ids are not canonical sorted integers")
    return candidates


def _canonical_coalition(
    state: ScreeningState,
    restored_event_step_ids: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(restored_event_step_ids, (str, bytes, bytearray)) or not isinstance(
        restored_event_step_ids, Sequence
    ):
        raise TypeError("restored event step ids must be an ordered sequence")
    restored = tuple(restored_event_step_ids)
    if any(
        not isinstance(event_id, int) or isinstance(event_id, bool)
        for event_id in restored
    ):
        raise ValueError("restored event step ids must be integers")
    if len(restored) != len(set(restored)):
        raise ValueError("restoration coalition contains duplicate event step ids")
    if restored != tuple(sorted(restored)):
        raise ValueError("restoration coalition must be canonical sorted")
    candidates = state.candidate_event_step_ids
    if not set(restored).issubset(candidates):
        raise ValueError("restoration coalition contains an unknown candidate event")
    return restored


def _validate_budget_metadata(budget_event_capacity: int | None) -> None:
    if budget_event_capacity is not None and (
        not isinstance(budget_event_capacity, int)
        or isinstance(budget_event_capacity, bool)
        or budget_event_capacity < 0
    ):
        raise ValueError("budget_event_capacity must be a non-negative integer or None")


@dataclass(frozen=True)
class V22LabelPromptSpec:
    """One screening state and one canonical restoration coalition."""

    state: ScreeningState
    restored_event_step_ids: tuple[int, ...]
    budget_event_capacity: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, ScreeningState):
            raise TypeError("state must be a ScreeningState")
        if self.state.role not in SCREENING_ROLES:
            raise ValueError("confirm-like state access is forbidden for v2.2 labels")
        canonical = _canonical_coalition(
            self.state,
            self.restored_event_step_ids,
        )
        if canonical != self.restored_event_step_ids:
            raise ValueError("restoration coalition is not canonical")
        _validate_budget_metadata(self.budget_event_capacity)

    @property
    def coalition_mask(self) -> int:
        restored = set(self.restored_event_step_ids)
        return sum(
            1 << index
            for index, event_id in enumerate(self.state.candidate_event_step_ids)
            if event_id in restored
        )

    @property
    def fidelity(self) -> str:
        if not self.restored_event_step_ids:
            return "summary_only"
        if self.restored_event_step_ids == self.state.candidate_event_step_ids:
            return "full_reference"
        return "mixed_fidelity"


def make_v2_2_label_prompt_spec(
    artifact: ValidatedScreeningArtifact,
    state: ScreeningState,
    *,
    restored_event_step_ids: Sequence[int],
    budget_event_capacity: int | None = None,
) -> V22LabelPromptSpec:
    """Validate membership and create one budget-agnostic prompt specification."""
    _validate_artifact(artifact)
    _validate_state(artifact, state)
    restored = _canonical_coalition(state, restored_event_step_ids)
    _validate_budget_metadata(budget_event_capacity)
    return V22LabelPromptSpec(
        state=state,
        restored_event_step_ids=restored,
        budget_event_capacity=budget_event_capacity,
    )


def iter_v2_2_label_prompt_specs(
    artifact: ValidatedScreeningArtifact,
    *,
    budget_event_capacity: int | None = None,
) -> Iterator[V22LabelPromptSpec]:
    """Yield every coalition in state order and ascending coalition-mask order."""
    _validate_artifact(artifact)
    _validate_budget_metadata(budget_event_capacity)
    for expected_index, state in enumerate(artifact.states):
        candidates = _validate_state(artifact, state)
        if state.index != expected_index:
            raise ValueError("screening state order drifted")
        for mask in range(1 << len(candidates)):
            restored = tuple(
                event_id
                for index, event_id in enumerate(candidates)
                if mask & (1 << index)
            )
            yield V22LabelPromptSpec(
                state=state,
                restored_event_step_ids=restored,
                budget_event_capacity=budget_event_capacity,
            )


def build_v2_2_label_messages(
    artifact: ValidatedScreeningArtifact,
    spec: V22LabelPromptSpec,
    *,
    image_decoder: Callable[[bytes], Any],
) -> list[dict[str, Any]]:
    """Build one native v2.1 prompt without exposing budget metadata."""
    _validate_artifact(artifact)
    if not isinstance(spec, V22LabelPromptSpec):
        raise TypeError("spec must be a V22LabelPromptSpec")
    _validate_state(artifact, spec.state)
    _canonical_coalition(spec.state, spec.restored_event_step_ids)

    # note (luojiaxuan): The validated artifact already owns a screening-only
    # manifest and image inventory. Reusing those private bytes preserves the
    # confirm barrier while leaving the frozen v2/v2.1 builders unchanged.
    manifest = json.loads(artifact._screening_manifest_json)
    if not isinstance(manifest, Mapping):
        raise ValueError("screening manifest must be a JSON object")
    messages = build_gui_owl_v2_1_mixed_fidelity_messages(
        manifest,
        trajectory_id=spec.state.trajectory_id,
        decision_step_id=spec.state.decision_step_id,
        restored_event_step_ids=spec.restored_event_step_ids,
        image_bytes_loader=artifact.image_bytes,
        image_decoder=image_decoder,
    )
    validate_gui_owl_v2_1_native_messages(messages)
    return messages


def label_prompt_image_inventory(
    spec: V22LabelPromptSpec,
    messages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a deterministic, image-object-independent prompt inventory."""
    if not isinstance(spec, V22LabelPromptSpec):
        raise TypeError("spec must be a V22LabelPromptSpec")
    image_count = validate_gui_owl_v2_1_native_messages(messages)
    expected_image_count = 1 + len(spec.restored_event_step_ids)
    if image_count != expected_image_count:
        raise ValueError("prompt image count differs from the restoration coalition")

    text_records: list[dict[str, str]] = []
    user_block_types: list[str] = []
    for message in messages:
        role = str(message["role"])
        for block in message["content"]:
            block_type = str(block["type"])
            if role == "user":
                user_block_types.append(block_type)
            if block_type == "text":
                text_records.append({"role": role, "text": str(block["text"])})
    text_payload = json.dumps(
        text_records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": LABEL_PROMPT_INVENTORY_SCHEMA_VERSION,
        "state_id": spec.state.state_id,
        "role": spec.state.role,
        "trajectory_id": spec.state.trajectory_id,
        "decision_step_id": spec.state.decision_step_id,
        "candidate_event_step_ids": list(spec.state.candidate_event_step_ids),
        "coalition_mask": spec.coalition_mask,
        "fidelity": spec.fidelity,
        "high_fidelity_restored_event_step_ids": list(
            spec.restored_event_step_ids
        ),
        "high_fidelity_image_count": len(spec.restored_event_step_ids),
        "current_observation_image_count": 1,
        "image_count": image_count,
        "text_block_count": len(text_records),
        "policy_visible_text_sha256": hashlib.sha256(text_payload).hexdigest(),
        "user_block_types": user_block_types,
    }
