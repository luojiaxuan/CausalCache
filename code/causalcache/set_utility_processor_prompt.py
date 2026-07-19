"""Render set-utility prompt plans with the frozen GUI-Owl v2.1 contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from causalcache.low_fidelity_v2 import serialize_low_fidelity_v2
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    GUI_OWL_V2_1_SYSTEM_PROMPT,
    validate_gui_owl_v2_1_native_messages,
)
from causalcache.set_utility_label_producer import MixedFidelityPromptPlan


def build_set_utility_gui_owl_v2_1_messages(
    plan: MixedFidelityPromptPlan,
    *,
    image_bytes_loader: Callable[[str], bytes],
    image_decoder: Callable[[bytes], Any],
) -> list[dict[str, Any]]:
    """Render every summary, coalition images, and the high-fidelity current state."""
    if not isinstance(plan, MixedFidelityPromptPlan):
        raise TypeError("plan must be a MixedFidelityPromptPlan")
    if not callable(image_bytes_loader):
        raise TypeError("image_bytes_loader must be callable")
    if not callable(image_decoder):
        raise TypeError("image_decoder must be callable")

    def load_image(reference: str) -> Any:
        raw = image_bytes_loader(reference)
        if not isinstance(raw, bytes):
            raise TypeError("set-utility image_bytes_loader must return bytes")
        return image_decoder(raw)

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Please generate the next move from the task, event summaries, restored "
                "post-action states, and current observation.\n\n"
                f"Instruction: {plan.task_instruction}"
            ),
        }
    ]
    for event in plan.history_events:
        summary = serialize_low_fidelity_v2(event.low_fidelity_summary).decode("utf-8")
        content.append({"type": "text", "text": f"Event summary:\n{summary}"})
        if event.high_fidelity_observation_ref is not None:
            content.append(
                {
                    "type": "image",
                    "image": load_image(event.high_fidelity_observation_ref),
                }
            )
    content.extend(
        [
            {"type": "text", "text": "Current observation:"},
            {
                "type": "image",
                "image": load_image(plan.current_observation_ref),
            },
            {
                "type": "text",
                "text": GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
            },
        ]
    )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_V2_1_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]
    validate_gui_owl_v2_1_native_messages(messages)
    return messages


__all__ = ["build_set_utility_gui_owl_v2_1_messages"]
