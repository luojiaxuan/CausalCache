"""Arbitrary-coalition inputs for restoration-v2.2 expansion exact labels."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.data.restoration_v2_2_expansion_label_parent import (
    EXPANSION_LABEL_ROLES,
    V22ExpansionLabelParentState,
)
from causalcache.policy.gui_owl_v2_1 import (
    build_gui_owl_v2_1_mixed_fidelity_messages,
    validate_gui_owl_v2_1_native_messages,
)
from causalcache.restoration_v2_2_expansion_substrate_inputs import (
    build_decision_view_input,
    validate_request_manifest,
)
from scripts.run_restoration_v2_2_expansion_substrate import (
    ExpansionArtifact,
    MessageBundle,
    build_expansion_messages,
)


EXPANSION_LABEL_PROMPT_INVENTORY_SCHEMA_VERSION = "1.0.0"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_budget_metadata(budget_event_capacity: int | None) -> None:
    if budget_event_capacity is not None and (
        not isinstance(budget_event_capacity, int)
        or isinstance(budget_event_capacity, bool)
        or budget_event_capacity < 0
    ):
        raise ValueError("budget_event_capacity must be a non-negative integer or None")


def _validate_artifact(artifact: ExpansionArtifact) -> str:
    if not isinstance(artifact, ExpansionArtifact):
        raise TypeError("artifact must be a validated immutable ExpansionArtifact")
    if len(artifact.states) != 192:
        raise ValueError("expansion label inputs require exactly 192 manifest states")
    validation = artifact.validation
    if (
        not isinstance(validation, Mapping)
        or validation.get("outcome")
        != "PASSED_GUIODYSSEY_RESTORATION_V2_EXPANSION_VALIDATION"
        or validation.get("counts", {}).get("state_count") != 192
        or validation.get("per_decision_view_current_expert_action_payload_included")
        is not False
        or validation.get("consumer_must_slice_events_by_history_event_step_ids")
        is not True
    ):
        raise ValueError("expansion label inputs require a safe immutable validation")
    tree_sha = artifact.tree.get("artifact_tree_sha256")
    if not isinstance(tree_sha, str) or re.fullmatch(r"[0-9a-f]{64}", tree_sha) is None:
        raise ValueError("immutable artifact tree SHA256 is invalid")
    return tree_sha


def _artifact_state(artifact: ExpansionArtifact, parent: V22ExpansionLabelParentState) -> Any:
    if not isinstance(parent, V22ExpansionLabelParentState):
        raise TypeError("parent must be a V22ExpansionLabelParentState")
    if parent.role not in EXPANSION_LABEL_ROLES:
        raise ValueError("legacy 45-state roles are forbidden for expansion labels")
    tree_sha = _validate_artifact(artifact)
    if parent.immutable_artifact_tree_sha256 != tree_sha:
        raise ValueError("parent state belongs to a different immutable artifact")
    if parent.index < 0 or parent.index >= len(artifact.states):
        raise ValueError("parent state index is outside the immutable denominator")
    state = artifact.states[parent.index]
    decision = state.decision
    expected = {
        "index": parent.index,
        "role": parent.role,
        "source_id": parent.source_id,
        "state_id": parent.state_id,
        "decision_step_id": parent.decision_step_id,
        "history_event_step_ids": parent.history_event_step_ids,
        "candidate_event_step_ids": parent.candidate_event_step_ids,
        "current_equivalent_event_step_id": parent.current_equivalent_event_step_id,
    }
    observed = {
        "index": state.index,
        "role": state.role,
        "source_id": state.source_id,
        "state_id": state.state_id,
        "decision_step_id": state.decision_step_id,
        "history_event_step_ids": tuple(decision.get("history_event_step_ids", ())),
        "candidate_event_step_ids": tuple(
            decision.get("candidate_event_step_ids", ())
        ),
        "current_equivalent_event_step_id": decision.get(
            "current_equivalent_event_step_id"
        ),
    }
    if observed != expected:
        raise ValueError("parent identity differs from the immutable manifest state")
    if (
        len(parent.candidate_event_step_ids) not in {2, 3, 4}
        or parent.candidate_event_step_ids
        != tuple(range(1, parent.decision_step_id - 1))
        or parent.history_event_step_ids
        != tuple(range(1, parent.decision_step_id))
    ):
        raise ValueError("expansion parent n=2/3/4 history geometry drifted")
    return state


def _canonical_coalition(
    parent: V22ExpansionLabelParentState,
    restored_event_step_ids: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(restored_event_step_ids, (str, bytes, bytearray)) or not isinstance(
        restored_event_step_ids, Sequence
    ):
        raise TypeError("restored event step ids must be an ordered sequence")
    restored = tuple(restored_event_step_ids)
    if any(type(event_id) is not int for event_id in restored):
        raise ValueError("restored event step ids must be integers")
    if len(restored) != len(set(restored)):
        raise ValueError("restoration coalition contains duplicate event step ids")
    if restored != tuple(sorted(restored)):
        raise ValueError("restoration coalition must be canonical sorted")
    if not set(restored).issubset(parent.candidate_event_step_ids):
        raise ValueError("restoration coalition contains an unknown candidate event")
    return restored


@dataclass(frozen=True)
class V22ExpansionLabelPromptSpec:
    """One expansion parent and one canonical arbitrary restoration coalition."""

    parent: V22ExpansionLabelParentState
    restored_event_step_ids: tuple[int, ...]
    budget_event_capacity: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.parent, V22ExpansionLabelParentState):
            raise TypeError("parent must be a V22ExpansionLabelParentState")
        if self.parent.role not in EXPANSION_LABEL_ROLES:
            raise ValueError("legacy 45-state roles are forbidden for expansion labels")
        if _canonical_coalition(self.parent, self.restored_event_step_ids) != (
            self.restored_event_step_ids
        ):
            raise ValueError("restoration coalition is not canonical")
        _validate_budget_metadata(self.budget_event_capacity)

    @property
    def coalition_mask(self) -> int:
        restored = set(self.restored_event_step_ids)
        return sum(
            1 << index
            for index, event_id in enumerate(self.parent.candidate_event_step_ids)
            if event_id in restored
        )

    @property
    def fidelity(self) -> str:
        if not self.restored_event_step_ids:
            return "summary_only"
        if self.restored_event_step_ids == self.parent.candidate_event_step_ids:
            return "full_reference"
        return "mixed_fidelity"


def make_v2_2_expansion_label_prompt_spec(
    artifact: ExpansionArtifact,
    parent: V22ExpansionLabelParentState,
    *,
    restored_event_step_ids: Sequence[int],
    budget_event_capacity: int | None = None,
) -> V22ExpansionLabelPromptSpec:
    """Validate manifest membership and construct one budget-agnostic spec."""
    _artifact_state(artifact, parent)
    restored = _canonical_coalition(parent, restored_event_step_ids)
    _validate_budget_metadata(budget_event_capacity)
    return V22ExpansionLabelPromptSpec(
        parent=parent,
        restored_event_step_ids=restored,
        budget_event_capacity=budget_event_capacity,
    )


def iter_v2_2_expansion_label_prompt_specs(
    artifact: ExpansionArtifact,
    parents: Sequence[V22ExpansionLabelParentState],
    *,
    budget_event_capacity: int | None = None,
) -> Iterator[V22ExpansionLabelPromptSpec]:
    """Yield every state power set in index, cardinality, lexicographic order."""
    _validate_artifact(artifact)
    _validate_budget_metadata(budget_event_capacity)
    if isinstance(parents, (str, bytes, bytearray, Mapping)) or not isinstance(
        parents, Sequence
    ):
        raise TypeError("parents must be the ordered 192-state parent sequence")
    if len(parents) != 192:
        raise ValueError("expansion prompt enumeration requires all 192 parents")
    for index, parent in enumerate(parents):
        if not isinstance(parent, V22ExpansionLabelParentState) or parent.index != index:
            raise ValueError("expansion parents must preserve canonical 192-state order")
        _artifact_state(artifact, parent)
        candidates = parent.candidate_event_step_ids
        for size in range(len(candidates) + 1):
            for restored in itertools.combinations(candidates, size):
                yield V22ExpansionLabelPromptSpec(
                    parent=parent,
                    restored_event_step_ids=restored,
                    budget_event_capacity=budget_event_capacity,
                )


def _mixed_bundle(
    artifact: ExpansionArtifact,
    state: Any,
    restored: tuple[int, ...],
    image_decoder: Callable[[bytes], Any],
) -> MessageBundle:
    trajectory = artifact.trajectory(state)
    decision = state.decision
    view = build_decision_view_input(trajectory, decision)
    included_events = tuple(view["events"])
    validate_request_manifest(
        view["request_manifest"],
        decision,
        included_events=included_events,
    )
    sliced_trajectory = dict(trajectory)
    sliced_trajectory["events"] = list(included_events)
    sliced_trajectory["decisions"] = [dict(decision)]
    messages = build_gui_owl_v2_1_mixed_fidelity_messages(
        {"trajectories": [sliced_trajectory]},
        trajectory_id=state.source_id,
        decision_step_id=state.decision_step_id,
        restored_event_step_ids=restored,
        image_bytes_loader=artifact.image_bytes,
        image_decoder=image_decoder,
    )
    validate_gui_owl_v2_1_native_messages(messages)
    return MessageBundle(
        messages=messages,
        request_manifest=dict(view["request_manifest"]),
        slice_witness_sha256=str(view["slice_witness_sha256"]),
        included_events=included_events,
        restored_event_step_ids=restored,
    )


def _validate_bundle(
    spec: V22ExpansionLabelPromptSpec,
    bundle: MessageBundle,
) -> None:
    parent = spec.parent
    image_count = validate_gui_owl_v2_1_native_messages(bundle.messages)
    included_steps = tuple(event.get("step_id") for event in bundle.included_events)
    request_sha = hashlib.sha256(
        _canonical_json_bytes(bundle.request_manifest)
    ).hexdigest()
    if (
        bundle.restored_event_step_ids != spec.restored_event_step_ids
        or image_count != 1 + len(spec.restored_event_step_ids)
        or included_steps != parent.history_event_step_ids
        or bundle.request_manifest.get("source_id") != parent.source_id
        or bundle.request_manifest.get("state_id") != parent.state_id
        or bundle.request_manifest.get("decision_step_id") != parent.decision_step_id
        or tuple(bundle.request_manifest.get("included_event_step_ids", ()))
        != parent.history_event_step_ids
        or bundle.request_manifest.get("current_or_future_event_action_exposure_count")
        != 0
        or any(step >= parent.decision_step_id for step in included_steps)
        or request_sha != parent.request_manifest_sha256
        or bundle.slice_witness_sha256 != parent.slice_witness_sha256
    ):
        raise ValueError("expansion label bundle history/request identity drifted")


def build_v2_2_expansion_label_messages(
    artifact: ExpansionArtifact,
    spec: V22ExpansionLabelPromptSpec,
    *,
    image_decoder: Callable[[bytes], Any],
) -> MessageBundle:
    """Build a full, empty, or arbitrary mixed-fidelity expansion request."""
    if not isinstance(spec, V22ExpansionLabelPromptSpec):
        raise TypeError("spec must be a V22ExpansionLabelPromptSpec")
    state = _artifact_state(artifact, spec.parent)
    restored = _canonical_coalition(spec.parent, spec.restored_event_step_ids)

    # note (luojiaxuan): Endpoint calls go through the exact public substrate
    # builder, making full and empty requests byte-identical to the substrate.
    # Only strict interior coalitions invoke the same frozen GUI-Owl builder on
    # the same public decision-view slice.
    if not restored:
        bundle = build_expansion_messages(
            artifact,
            state,
            "summary_only",
            image_decoder,
        )
    elif restored == spec.parent.candidate_event_step_ids:
        bundle = build_expansion_messages(
            artifact,
            state,
            "reference",
            image_decoder,
        )
    else:
        bundle = _mixed_bundle(artifact, state, restored, image_decoder)
    _validate_bundle(spec, bundle)
    return bundle


def expansion_label_prompt_inventory(
    spec: V22ExpansionLabelPromptSpec,
    bundle: MessageBundle,
) -> dict[str, Any]:
    """Return a deterministic inventory without serializing decoded images."""
    if not isinstance(spec, V22ExpansionLabelPromptSpec):
        raise TypeError("spec must be a V22ExpansionLabelPromptSpec")
    _validate_bundle(spec, bundle)
    text_records: list[dict[str, str]] = []
    user_block_types: list[str] = []
    for message in bundle.messages:
        role = str(message["role"])
        for block in message["content"]:
            block_type = str(block["type"])
            if role == "user":
                user_block_types.append(block_type)
            if block_type == "text":
                text_records.append({"role": role, "text": str(block["text"])})
    text_sha = hashlib.sha256(_canonical_json_bytes(text_records)).hexdigest()
    return {
        "schema_version": EXPANSION_LABEL_PROMPT_INVENTORY_SCHEMA_VERSION,
        "state_index": spec.parent.index,
        "state_id": spec.parent.state_id,
        "role": spec.parent.role,
        "source_id": spec.parent.source_id,
        "decision_step_id": spec.parent.decision_step_id,
        "history_event_step_ids": list(spec.parent.history_event_step_ids),
        "candidate_event_step_ids": list(spec.parent.candidate_event_step_ids),
        "coalition_mask": spec.coalition_mask,
        "fidelity": spec.fidelity,
        "high_fidelity_restored_event_step_ids": list(
            spec.restored_event_step_ids
        ),
        "high_fidelity_image_count": len(spec.restored_event_step_ids),
        "current_observation_image_count": 1,
        "image_count": 1 + len(spec.restored_event_step_ids),
        "request_manifest_sha256": spec.parent.request_manifest_sha256,
        "slice_witness_sha256": spec.parent.slice_witness_sha256,
        "current_or_future_event_action_exposure_count": 0,
        "text_block_count": len(text_records),
        "policy_visible_text_sha256": text_sha,
        "user_block_types": user_block_types,
    }


__all__ = [
    "EXPANSION_LABEL_PROMPT_INVENTORY_SCHEMA_VERSION",
    "V22ExpansionLabelPromptSpec",
    "build_v2_2_expansion_label_messages",
    "expansion_label_prompt_inventory",
    "iter_v2_2_expansion_label_prompt_specs",
    "make_v2_2_expansion_label_prompt_spec",
]
