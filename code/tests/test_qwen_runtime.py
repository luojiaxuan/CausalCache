import json
import tempfile
import unittest
from pathlib import Path

from causalcache.policy.qwen_runtime import effective_visual_tokens, visual_patch_factor


class QwenRuntimeTest(unittest.TestCase):
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
