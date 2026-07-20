from __future__ import annotations

import unittest

from causalcache.set_utility_contextual_hidden import (
    CONTEXTUAL_ALLOWED_VISUAL_TOKEN_COUNTS,
    CONTEXTUAL_SOURCE_HIDDEN_SIZE,
    build_contextual_entity_messages,
    select_contextual_hidden_tokens,
)


class ContextualMessagesTest(unittest.TestCase):
    def test_native_message_contains_one_image(self) -> None:
        marker = object()
        messages = build_contextual_entity_messages(
            prompt_text="Task instruction:\nDo it", image=marker
        )
        self.assertEqual(messages[1]["content"][1]["image"], marker)


try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class ContextualSelectionTorchTest(unittest.TestCase):
    def test_selects_all_visual_and_bounded_preceding_text_tokens(self) -> None:
        image_token_id = 99
        prefix = 80
        suffix = 3
        visual_token_count = max(CONTEXTUAL_ALLOWED_VISUAL_TOKEN_COUNTS)
        input_ids = torch.arange(prefix + visual_token_count + suffix)[
            None
        ]
        input_ids[:, prefix : prefix + visual_token_count] = image_token_id
        hidden = torch.arange(
            input_ids.shape[1] * CONTEXTUAL_SOURCE_HIDDEN_SIZE,
            dtype=torch.float32,
        ).reshape(1, input_ids.shape[1], CONTEXTUAL_SOURCE_HIDDEN_SIZE)
        visual, text = select_contextual_hidden_tokens(
            input_ids=input_ids,
            final_hidden_state=hidden,
            image_token_id=image_token_id,
        )
        self.assertEqual(
            tuple(visual.shape),
            (visual_token_count, CONTEXTUAL_SOURCE_HIDDEN_SIZE),
        )
        self.assertEqual(tuple(text.shape), (64, CONTEXTUAL_SOURCE_HIDDEN_SIZE))
        self.assertEqual(visual.dtype, torch.bfloat16)
        self.assertEqual(text.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(text[-1], hidden[0, prefix - 1].to(torch.bfloat16)))


if __name__ == "__main__":
    unittest.main()
