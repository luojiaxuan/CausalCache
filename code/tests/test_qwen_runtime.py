import json
import math
import tempfile
import unittest
from pathlib import Path

from causalcache.policy.qwen_runtime import (
    QwenPolicyRuntime,
    _teacher_forced_action_layout,
    effective_visual_tokens,
    full_vocab_action_path_kl,
    visual_patch_factor,
)


class _FakeTokenizer:
    all_special_ids = [0, 99]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("canonical actions must disable special tokens")
        if text == "special":
            return [99]
        return [7, 8]


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


class QwenRuntimeTest(unittest.TestCase):
    def test_canonical_action_tokenization_disables_special_tokens(self) -> None:
        runtime = object.__new__(QwenPolicyRuntime)
        runtime.processor = _FakeProcessor()
        self.assertEqual(
            runtime.tokenize_canonical_action('{"action_type":"tap"}'),
            [7, 8],
        )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            runtime.tokenize_canonical_action("  ")
        with self.assertRaisesRegex(ValueError, "special"):
            runtime.tokenize_canonical_action("special")

    def test_teacher_forced_layout_uses_strict_causal_shift(self) -> None:
        layout = _teacher_forced_action_layout([10, 11, 12], [20, 21, 22])
        self.assertEqual(layout["model_input_ids"], [10, 11, 12, 20, 21])
        self.assertEqual(layout["action_logit_positions"], [2, 3, 4])
        self.assertEqual(layout["action_token_ids"], [20, 21, 22])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            _teacher_forced_action_layout([10], [])

    def test_full_vocab_action_path_kl_self_and_known_case(self) -> None:
        reference = [[math.log(0.75), math.log(0.25)]]
        candidate = [[math.log(0.5), math.log(0.5)]]
        self_kl = full_vocab_action_path_kl(reference, reference)
        known_kl = full_vocab_action_path_kl(reference, candidate)
        self.assertEqual(self_kl, {"per_token": [0.0], "sum": 0.0, "mean": 0.0})
        expected = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
        self.assertAlmostEqual(known_kl["per_token"][0], expected)
        self.assertAlmostEqual(known_kl["sum"], expected)
        self.assertAlmostEqual(known_kl["mean"], expected)

    def test_effective_visual_tokens_matches_native_gui_owl_grid(self) -> None:
        self.assertEqual(
            effective_visual_tokens([[1, 150, 68]], merge_size=2),
            2550,
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            effective_visual_tokens([[1, 150, 68]], merge_size=0)

    def test_visual_patch_factor_tracks_model_preprocessor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            config_path = model_dir / "preprocessor_config.json"
            config_path.write_text(
                json.dumps({"patch_size": 14, "merge_size": 2}),
                encoding="utf-8",
            )
            self.assertEqual(visual_patch_factor(model_dir), 28)
            config_path.write_text(
                json.dumps({"patch_size": 16, "merge_size": 2}),
                encoding="utf-8",
            )
            self.assertEqual(visual_patch_factor(model_dir), 32)


if __name__ == "__main__":
    unittest.main()
