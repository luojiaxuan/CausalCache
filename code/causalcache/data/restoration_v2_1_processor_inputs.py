"""Confirm-safe v2.1 processor inputs for the frozen screening denominator."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
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


PROCESSOR_FIDELITIES = ("reference", "summary_only")
EXPECTED_PROCESSOR_PROMPTS = EXPECTED_SCREENING_STATES * len(PROCESSOR_FIDELITIES)


@dataclass(frozen=True)
class V21ProcessorPromptSpec:
    """One immutable state/fidelity member of the 90-prompt preflight."""

    prompt_index: int
    state: ScreeningState
    fidelity: str
    restored_event_step_ids: tuple[int, ...]


def iter_v2_1_processor_prompt_specs(
    artifact: ValidatedScreeningArtifact,
) -> Iterator[V21ProcessorPromptSpec]:
    """Yield reference then summary-only for each exact screening state."""
    if not isinstance(artifact, ValidatedScreeningArtifact):
        raise TypeError("artifact must be a ValidatedScreeningArtifact")
    if len(artifact.states) != EXPECTED_SCREENING_STATES:
        raise ValueError("v2.1 processor preflight requires exactly 45 states")
    prompt_index = 0
    for state_index, state in enumerate(artifact.states):
        if state.index != state_index or state.role not in SCREENING_ROLES:
            raise ValueError("screening state order or role drifted")
        for fidelity in PROCESSOR_FIDELITIES:
            restored = (
                state.candidate_event_step_ids if fidelity == "reference" else ()
            )
            yield V21ProcessorPromptSpec(
                prompt_index=prompt_index,
                state=state,
                fidelity=fidelity,
                restored_event_step_ids=tuple(restored),
            )
            prompt_index += 1
    if prompt_index != EXPECTED_PROCESSOR_PROMPTS:
        raise ValueError("v2.1 processor prompt denominator drifted")


def build_v2_1_processor_messages(
    artifact: ValidatedScreeningArtifact,
    spec: V21ProcessorPromptSpec,
    *,
    image_decoder: Callable[[bytes], Any],
) -> list[dict[str, Any]]:
    """Build one prompt from the loader's screening-only manifest and images."""
    if not isinstance(artifact, ValidatedScreeningArtifact):
        raise TypeError("artifact must be a ValidatedScreeningArtifact")
    if not isinstance(spec, V21ProcessorPromptSpec):
        raise TypeError("spec must be a V21ProcessorPromptSpec")
    if spec.state not in artifact.states or spec.state.role not in SCREENING_ROLES:
        raise ValueError("prompt state is outside the confirm-safe screening view")
    if spec.fidelity not in PROCESSOR_FIDELITIES:
        raise ValueError("unknown v2.1 processor fidelity")
    expected_restored = (
        spec.state.candidate_event_step_ids if spec.fidelity == "reference" else ()
    )
    if spec.restored_event_step_ids != tuple(expected_restored):
        raise ValueError("prompt restored-event set differs from its fidelity")

    # note (luojiaxuan): The private bytes are already a screening-only copy made by
    # the fail-closed loader. Reading them here avoids reopening the full artifact or
    # adding a generic builder hook to the frozen v2 loader.
    manifest = json.loads(artifact._screening_manifest_json)
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
