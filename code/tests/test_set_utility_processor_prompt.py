from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import pytest

from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    GUI_OWL_V2_1_SYSTEM_PROMPT,
    build_gui_owl_v2_1_mixed_fidelity_messages,
    validate_gui_owl_v2_1_native_messages,
)
from causalcache.set_utility_label_producer import (
    MixedFidelityPromptPlan,
    PromptHistoryEvent,
)
from causalcache.set_utility_processor_prompt import (
    build_set_utility_gui_owl_v2_1_messages,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROMPT_FIXTURE = (
    REPOSITORY_ROOT / "data/fixtures/restoration_v2_prompt_low_fidelity.json"
)


def _fixture_plan(
    manifest: dict[str, object],
    *,
    decision_step_id: int,
    restored_event_step_ids: tuple[int, ...],
) -> MixedFidelityPromptPlan:
    trajectory = manifest["trajectories"][0]
    decision = next(
        item
        for item in trajectory["decisions"]
        if item["decision_step_id"] == decision_step_id
    )
    events = {item["step_id"]: item for item in trajectory["events"]}
    candidates = frozenset(decision["candidate_event_step_ids"])
    restored = frozenset(restored_event_step_ids)
    return MixedFidelityPromptPlan(
        state_id=f"fixture-step-{decision_step_id}",
        task_instruction=trajectory["instruction"],
        restored_event_step_ids=restored_event_step_ids,
        history_events=tuple(
            PromptHistoryEvent(
                event_step_id=step_id,
                low_fidelity_summary=LowFidelityEventV2.from_mapping(
                    events[step_id]["low_fidelity_v2"]
                ),
                is_frozen_candidate=step_id in candidates,
                high_fidelity_observation_ref=(
                    events[step_id]["observation_after_path"]
                    if step_id in restored
                    else None
                ),
            )
            for step_id in decision["history_event_step_ids"]
        ),
        current_observation_ref=decision["current_observation_path"],
    )


def _render(plan: MixedFidelityPromptPlan) -> list[dict[str, object]]:
    return build_set_utility_gui_owl_v2_1_messages(
        plan,
        image_bytes_loader=lambda reference: reference.encode("utf-8"),
        image_decoder=lambda raw: raw.decode("utf-8"),
    )


def test_old_step_4_5_6_fixture_is_byte_and_structure_compatible() -> None:
    manifest = json.loads(PROMPT_FIXTURE.read_text(encoding="utf-8"))
    for decision_step_id in (4, 5, 6):
        candidates = tuple(range(1, decision_step_id - 1))
        for cardinality in range(len(candidates) + 1):
            for coalition in itertools.combinations(candidates, cardinality):
                expected = build_gui_owl_v2_1_mixed_fidelity_messages(
                    manifest,
                    trajectory_id="fixture-step-6",
                    decision_step_id=decision_step_id,
                    restored_event_step_ids=coalition,
                    image_bytes_loader=lambda reference: reference.encode("utf-8"),
                    image_decoder=lambda raw: raw.decode("utf-8"),
                )
                actual = _render(
                    _fixture_plan(
                        manifest,
                        decision_step_id=decision_step_id,
                        restored_event_step_ids=coalition,
                    )
                )
                assert actual == expected
                assert json.dumps(
                    actual,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8") == json.dumps(
                    expected,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")


def test_arbitrary_decision_keeps_all_summaries_and_only_restored_images() -> None:
    manifest = json.loads(PROMPT_FIXTURE.read_text(encoding="utf-8"))
    source = _fixture_plan(
        manifest,
        decision_step_id=6,
        restored_event_step_ids=(),
    )
    arbitrary_step_ids = (3, 8, 13, 21, 34)
    plan = MixedFidelityPromptPlan(
        state_id="arbitrary-decision-35",
        task_instruction=source.task_instruction,
        restored_event_step_ids=(8, 21),
        history_events=tuple(
            PromptHistoryEvent(
                event_step_id=target_step_id,
                low_fidelity_summary=replace(
                    source_event.low_fidelity_summary,
                    step_id=target_step_id,
                ),
                is_frozen_candidate=target_step_id != 34,
                high_fidelity_observation_ref=(
                    f"fixture://post-{target_step_id:03d}.png"
                    if target_step_id in {8, 21}
                    else None
                ),
            )
            for source_event, target_step_id in zip(
                source.history_events,
                arbitrary_step_ids,
                strict=True,
            )
        ),
        current_observation_ref=source.current_observation_ref,
    )
    messages = _render(plan)
    content = messages[1]["content"]

    assert messages[0] == {
        "role": "system",
        "content": [{"type": "text", "text": GUI_OWL_V2_1_SYSTEM_PROMPT}],
    }
    assert [block["image"] for block in content if block["type"] == "image"] == [
        "fixture://post-008.png",
        "fixture://post-021.png",
        "fixture://current.png",
    ]
    summary_text = "\n".join(
        block["text"]
        for block in content
        if block["type"] == "text" and block["text"].startswith("Event summary:")
    )
    assert tuple(
        summary_text.count(f'"step_id":{step_id},') for step_id in arbitrary_step_ids
    ) == (
        1,
        1,
        1,
        1,
        1,
    )
    assert content[-1] == {
        "type": "text",
        "text": GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    }
    assert validate_gui_owl_v2_1_native_messages(messages) == 3


def test_current_observation_is_loaded_once_even_with_empty_coalition() -> None:
    manifest = json.loads(PROMPT_FIXTURE.read_text(encoding="utf-8"))
    plan = _fixture_plan(
        manifest,
        decision_step_id=6,
        restored_event_step_ids=(),
    )
    loaded: list[str] = []
    messages = build_set_utility_gui_owl_v2_1_messages(
        plan,
        image_bytes_loader=lambda reference: (
            loaded.append(reference) or reference.encode("utf-8")
        ),
        image_decoder=lambda raw: raw.decode("utf-8"),
    )
    assert loaded == ["fixture://current.png"]
    assert validate_gui_owl_v2_1_native_messages(messages) == 1


def test_renderer_fails_closed_before_or_at_image_boundary() -> None:
    manifest = json.loads(PROMPT_FIXTURE.read_text(encoding="utf-8"))
    plan = _fixture_plan(
        manifest,
        decision_step_id=4,
        restored_event_step_ids=(1,),
    )
    with pytest.raises(TypeError, match="MixedFidelityPromptPlan"):
        build_set_utility_gui_owl_v2_1_messages(
            object(),
            image_bytes_loader=lambda reference: reference.encode("utf-8"),
            image_decoder=lambda raw: raw,
        )
    with pytest.raises(TypeError, match="must return bytes"):
        build_set_utility_gui_owl_v2_1_messages(
            plan,
            image_bytes_loader=lambda reference: reference,
            image_decoder=lambda raw: raw,
        )
