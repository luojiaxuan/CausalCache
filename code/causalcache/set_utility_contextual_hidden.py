"""Contextual GUI-Owl hidden-state extraction for utility predictor v3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    GUI_OWL_V2_1_SYSTEM_PROMPT,
    validate_gui_owl_v2_1_native_messages,
)


CONTEXTUAL_TEXT_TOKEN_LIMIT = 64
CONTEXTUAL_VISUAL_TOKEN_COUNT = 480
CONTEXTUAL_SOURCE_HIDDEN_SIZE = 4096


def build_contextual_entity_messages(
    *, prompt_text: str, image: Any
) -> list[dict[str, Any]]:
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("contextual entity prompt must be non-empty")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_V2_1_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": image},
                {"type": "text", "text": GUI_OWL_V2_1_FINAL_USER_INSTRUCTION},
            ],
        },
    ]
    validate_gui_owl_v2_1_native_messages(messages)
    return messages


def select_contextual_hidden_tokens(
    *,
    input_ids: Any,
    final_hidden_state: Any,
    image_token_id: int,
    text_token_limit: int = CONTEXTUAL_TEXT_TOKEN_LIMIT,
) -> tuple[Any, Any]:
    """Select image positions and their immediately preceding text context."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("contextual hidden selection requires PyTorch") from error
    if type(image_token_id) is not int or image_token_id < 0:
        raise ValueError("image token id must be a non-negative integer")
    if type(text_token_limit) is not int or text_token_limit <= 0:
        raise ValueError("text token limit must be positive")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("contextual input ids must have batch size one")
    if (
        final_hidden_state.ndim != 3
        or final_hidden_state.shape[0] != 1
        or final_hidden_state.shape[1] != input_ids.shape[1]
        or final_hidden_state.shape[2] != CONTEXTUAL_SOURCE_HIDDEN_SIZE
    ):
        raise ValueError("contextual final hidden-state geometry drifted")
    image_positions = torch.nonzero(
        input_ids[0] == image_token_id, as_tuple=False
    ).flatten()
    if image_positions.numel() != CONTEXTUAL_VISUAL_TOKEN_COUNT:
        raise ValueError("contextual image token count drifted")
    first_image_position = int(image_positions[0].item())
    text_start = max(0, first_image_position - text_token_limit)
    text_positions = torch.arange(
        text_start,
        first_image_position,
        device=input_ids.device,
        dtype=torch.long,
    )
    text_positions = text_positions[
        input_ids[0].index_select(0, text_positions) != image_token_id
    ]
    if text_positions.numel() == 0:
        raise ValueError("contextual text selection is empty")
    visual = final_hidden_state[0].index_select(0, image_positions)
    text = final_hidden_state[0].index_select(0, text_positions)
    visual = visual.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    text = text.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    if tuple(visual.shape) != (
        CONTEXTUAL_VISUAL_TOKEN_COUNT,
        CONTEXTUAL_SOURCE_HIDDEN_SIZE,
    ):
        raise RuntimeError("contextual visual output geometry drifted")
    if not 1 <= text.shape[0] <= text_token_limit or text.shape[1] != (
        CONTEXTUAL_SOURCE_HIDDEN_SIZE
    ):
        raise RuntimeError("contextual text output geometry drifted")
    return visual, text


def contextual_hidden_forward(
    *, runtime: Any, messages: Sequence[Mapping[str, Any]]
) -> tuple[Any, Any, dict[str, Any]]:
    """Run one frozen GUI-Owl forward and return selected final-layer tokens."""
    model_inputs, image_counts = runtime._encode_exact_batch((messages,))
    if image_counts != (1,):
        raise ValueError("contextual entity forward requires exactly one image")
    image_token_id = getattr(runtime.model.config, "image_token_id", None)
    if type(image_token_id) is not int:
        raise ValueError("GUI-Owl config lacks a scalar image_token_id")
    with runtime.torch.inference_mode():
        outputs = runtime.model(
            **model_inputs,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
    hidden_states = getattr(outputs, "hidden_states", None)
    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
        raise RuntimeError("GUI-Owl forward did not return language-model hidden states")
    visual, text = select_contextual_hidden_tokens(
        input_ids=model_inputs["input_ids"],
        final_hidden_state=hidden_states[-1],
        image_token_id=image_token_id,
    )
    metadata = {
        "final_hidden_state_dtype": str(hidden_states[-1].dtype),
        "image_token_id": image_token_id,
        "input_sequence_length": int(model_inputs["input_ids"].shape[1]),
        "language_hidden_state_count": len(hidden_states),
        "text_token_count": int(text.shape[0]),
        "visual_token_count": int(visual.shape[0]),
    }
    del outputs, hidden_states, model_inputs
    return visual, text, metadata


__all__ = [
    "CONTEXTUAL_SOURCE_HIDDEN_SIZE",
    "CONTEXTUAL_TEXT_TOKEN_LIMIT",
    "CONTEXTUAL_VISUAL_TOKEN_COUNT",
    "build_contextual_entity_messages",
    "contextual_hidden_forward",
    "select_contextual_hidden_tokens",
]
