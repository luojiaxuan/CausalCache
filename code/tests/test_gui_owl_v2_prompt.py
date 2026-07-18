import copy
import itertools
import json
import unittest
from pathlib import Path

from causalcache.policy.gui_owl_v2 import build_gui_owl_v2_mixed_fidelity_messages


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPOSITORY_ROOT / "data/fixtures/restoration_v2_prompt_low_fidelity.json"


def _items(messages, item_type):
    return [
        item
        for message in messages
        for item in message["content"]
        if item["type"] == item_type
    ]


class GUIOwlV2PromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def _messages(self, restored, *, decision_step_id=6):
        return build_gui_owl_v2_mixed_fidelity_messages(
            self.manifest,
            trajectory_id="fixture-step-6",
            decision_step_id=decision_step_id,
            restored_event_step_ids=restored,
            image_bytes_loader=lambda path: path.encode("utf-8"),
            image_decoder=lambda raw: raw.decode("utf-8"),
        )

    def test_all_sixteen_coalitions_change_only_post_state_image_blocks(self) -> None:
        reference_texts = None
        for size in range(5):
            for coalition in itertools.combinations((1, 2, 3, 4), size):
                with self.subTest(coalition=coalition):
                    messages = self._messages(coalition)
                    self.assertEqual([message["role"] for message in messages], ["system", "user"])
                    texts = [item["text"] for item in _items(messages, "text")]
                    if reference_texts is None:
                        reference_texts = texts
                    self.assertEqual(texts, reference_texts)
                    images = [item["image"] for item in _items(messages, "image")]
                    expected = [f"fixture://post-{step_id:03d}.png" for step_id in coalition]
                    expected.append("fixture://current.png")
                    self.assertEqual(images, expected)
                    self.assertEqual(len(images), 1 + len(coalition))
                    self.assertNotIn("fixture://before-001.png", images)
                    expected_types = ["text"]
                    for step_id in (1, 2, 3, 4, 5):
                        expected_types.append("text")
                        if step_id in coalition:
                            expected_types.append("image")
                    expected_types.extend(("text", "image", "text"))
                    self.assertEqual(
                        [item["type"] for item in messages[1]["content"]],
                        expected_types,
                    )

    def test_development_steps_four_five_and_six_share_the_same_prefix_rule(self) -> None:
        for decision_step_id in (4, 5, 6):
            candidates = tuple(range(1, decision_step_id - 1))
            for size in range(len(candidates) + 1):
                for coalition in itertools.combinations(candidates, size):
                    with self.subTest(step=decision_step_id, coalition=coalition):
                        images = [
                            item["image"]
                            for item in _items(
                                self._messages(coalition, decision_step_id=decision_step_id),
                                "image",
                            )
                        ]
                        expected = [f"fixture://post-{step_id:03d}.png" for step_id in coalition]
                        current = (
                            "fixture://current.png"
                            if decision_step_id == 6
                            else f"fixture://post-{decision_step_id - 1:03d}.png"
                        )
                        self.assertEqual(images, [*expected, current])

    def test_reference_has_four_history_post_states_plus_current_once(self) -> None:
        images = [item["image"] for item in _items(self._messages((1, 2, 3, 4)), "image")]
        self.assertEqual(
            images,
            [
                "fixture://post-001.png",
                "fixture://post-002.png",
                "fixture://post-003.png",
                "fixture://post-004.png",
                "fixture://current.png",
            ],
        )
        text = "\n".join(item["text"] for item in _items(self._messages(()), "text"))
        self.assertEqual(text.count('"step_id":5'), 1)
        self.assertNotIn("source_tool_call", text)
        self.assertNotIn("observation_before", text)

    def test_unknown_duplicate_and_current_equivalent_restorations_are_rejected(self) -> None:
        for restored, error in (((1, 1), "unique"), ((5,), "candidate"), ((99,), "candidate")):
            with self.subTest(restored=restored):
                with self.assertRaisesRegex(ValueError, error):
                    self._messages(restored)

    def test_summary_or_identity_tampering_is_rejected_before_loading_images(self) -> None:
        cases = []
        summary_tamper = copy.deepcopy(self.manifest)
        summary_tamper["trajectories"][0]["events"][0]["low_fidelity_v2"]["screen_change"] = "low"
        cases.append((summary_tamper, "serialization"))
        identity_tamper = copy.deepcopy(self.manifest)
        identity_tamper["trajectories"][0]["events"][4]["observation_after_sha256"] = "0" * 64
        cases.append((identity_tamper, "identical"))
        for manifest, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    build_gui_owl_v2_mixed_fidelity_messages(
                        manifest,
                        trajectory_id="fixture-step-6",
                        decision_step_id=6,
                        restored_event_step_ids=(),
                        image_bytes_loader=lambda path: self.fail(
                            f"loaded image before validation: {path}"
                        ),
                        image_decoder=lambda raw: raw,
                    )

    def test_image_bytes_are_verified_against_the_archived_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "artifact image SHA256 mismatch"):
            build_gui_owl_v2_mixed_fidelity_messages(
                self.manifest,
                trajectory_id="fixture-step-6",
                decision_step_id=6,
                restored_event_step_ids=(1,),
                image_bytes_loader=lambda path: b"substituted-image",
                image_decoder=lambda raw: raw,
            )

    def test_decision_steps_outside_the_frozen_development_set_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit allowlist"):
            self._messages((), decision_step_id=7)

    def test_explicit_decision_step_allowlist_is_strict_and_default_is_identical(self) -> None:
        default = self._messages((1, 2), decision_step_id=6)
        explicit = build_gui_owl_v2_mixed_fidelity_messages(
            self.manifest,
            trajectory_id="fixture-step-6",
            decision_step_id=6,
            restored_event_step_ids=(1, 2),
            image_bytes_loader=lambda path: path.encode("utf-8"),
            image_decoder=lambda raw: raw.decode("utf-8"),
            allowed_decision_steps=(4, 5, 6),
        )
        self.assertEqual(explicit, default)
        cases = (
            ([], TypeError, "non-empty tuple"),
            ((), TypeError, "non-empty tuple"),
            ((4, 4), ValueError, "unique"),
            ((4, 0), ValueError, "positive"),
            ((4, True), ValueError, "positive"),
        )
        for allowlist, error_type, message in cases:
            with self.subTest(allowlist=allowlist):
                with self.assertRaisesRegex(error_type, message):
                    build_gui_owl_v2_mixed_fidelity_messages(
                        self.manifest,
                        trajectory_id="fixture-step-6",
                        decision_step_id=6,
                        restored_event_step_ids=(),
                        image_bytes_loader=lambda path: path.encode("utf-8"),
                        image_decoder=lambda raw: raw.decode("utf-8"),
                        allowed_decision_steps=allowlist,
                    )


if __name__ == "__main__":
    unittest.main()
