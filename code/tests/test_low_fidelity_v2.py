import json
import unittest
from pathlib import Path

from causalcache.low_fidelity_v2 import (
    LowFidelityEventV2,
    ScreenTextNode,
    action_argument,
    coordinate_bin,
    executor_result,
    mean_absolute_rgb_difference_from_resized_pixels,
    normalize_foreground_app,
    screen_change_from_mean_absolute_rgb_difference,
    screen_text_delta,
    select_foreground_app,
    serialize_low_fidelity_v2,
    sha256_bytes,
    swipe_argument,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPOSITORY_ROOT / "data/fixtures/restoration_v2_prompt_low_fidelity.json"


class LowFidelityV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_fixture_serializations_are_exact_utf8_and_hash_pinned(self) -> None:
        for event in self.fixture["trajectories"][0]["events"]:
            with self.subTest(step_id=event["step_id"]):
                value = LowFidelityEventV2.from_mapping(event["low_fidelity_v2"])
                serialized = serialize_low_fidelity_v2(value)
                self.assertEqual(serialized.decode("utf-8"), event["low_fidelity_v2_serialized"])
                self.assertEqual(sha256_bytes(serialized), event["low_fidelity_v2_sha256"])
                self.assertTrue(serialized.endswith(b"\n"))
                self.assertNotIn(b"\\u", serialized)
                self.assertEqual(
                    list(json.loads(serialized)),
                    [
                        "step_id",
                        "action_type",
                        "action_argument",
                        "foreground_app",
                        "screen_text_added",
                        "screen_text_removed",
                        "screen_change",
                        "executor_result",
                    ],
                )

    def test_screen_text_delta_uses_spatial_order_multiset_difference_and_cap(self) -> None:
        before = [
            ScreenTextNode("old repeated", 20, 0, 30, 20),
            ScreenTextNode("same", 0, 0, 10, 20),
        ]
        after = [
            ScreenTextNode("same", 0, 0, 10, 20),
            ScreenTextNode("new repeated extra", 10, 0, 20, 20),
        ]
        delta = screen_text_delta(
            before,
            after,
            maximum_added_tokens=2,
            maximum_removed_tokens=1,
        )
        self.assertEqual(delta.added, ("new", "extra"))
        self.assertEqual(delta.removed, ("old",))
        self.assertEqual(delta.added_discarded_count, 0)
        self.assertEqual(delta.removed_discarded_count, 0)
        capped = screen_text_delta(
            [],
            [ScreenTextNode(" ".join(f"token-{index}" for index in range(33)), 0, 0, 1, 1)],
        )
        self.assertEqual(len(capped.added), 32)
        self.assertEqual(capped.added_discarded_count, 1)

    def test_foreground_and_executor_provenance_helpers_do_not_infer(self) -> None:
        self.assertEqual(normalize_foreground_app("  Markor\nBeta "), "markor beta")
        self.assertEqual(
            select_foreground_app(
                source_event_app_label=None,
                executor_package_name=" Com.Example.Notes ",
            ),
            "com.example.notes",
        )
        self.assertEqual(
            select_foreground_app(source_event_app_label=None, executor_package_name=None),
            "unknown",
        )
        self.assertEqual(executor_result(True), "accepted")
        self.assertEqual(executor_result(False), "failed")
        self.assertEqual(executor_result(None), "unknown")

    def test_screen_change_thresholds_and_resized_rgb_metric_are_exact(self) -> None:
        expected = {
            0.0: "none",
            0.005: "none",
            0.0050001: "low",
            0.05: "low",
            0.050001: "medium",
            0.20: "medium",
            0.200001: "high",
            1.0: "high",
        }
        for value, category in expected.items():
            self.assertEqual(screen_change_from_mean_absolute_rgb_difference(value), category)
        self.assertEqual(
            mean_absolute_rgb_difference_from_resized_pixels(
                [(0, 0, 0)],
                [(255, 127, 0)],
                expected_pixels=1,
            ),
            382 / 765,
        )

    def test_action_arguments_freeze_spatial_swipe_text_and_terminal_semantics(self) -> None:
        self.assertEqual(coordinate_bin([0, 999]), "coordinate_bin:x0_y9")
        self.assertEqual(coordinate_bin([999, 0]), "coordinate_bin:x9_y0")
        self.assertEqual(swipe_argument([500, 800], [500, 200]), "viewport_down:medium")
        self.assertEqual(swipe_argument([500, 500], [500, 500]), "viewport_stationary:zero")
        self.assertEqual(swipe_argument([800, 500], [100, 500]), "viewport_right:long")
        cases = (
            ({"action": "click", "coordinate": [0, 999]}, "coordinate_bin:x0_y9"),
            ({"action": "long_press", "coordinate": [999, 0]}, "coordinate_bin:x9_y0"),
            (
                {"action": "swipe", "coordinate": [500, 800], "coordinate2": [500, 200]},
                "viewport_down:medium",
            ),
            ({"action": "type", "text": "Café"}, "Café"),
            ({"action": "system_button", "button": "Home"}, "Home"),
            ({"action": "open", "text": "Markor"}, "Markor"),
            ({"action": "wait"}, "wait"),
            ({"action": "answer", "text": "42"}, "42"),
            ({"action": "terminate", "status": "success"}, "success"),
        )
        for arguments, expected in cases:
            with self.subTest(action=arguments["action"]):
                self.assertEqual(action_argument(arguments), expected)

    def test_action_type_and_argument_must_obey_the_same_frozen_grammar(self) -> None:
        with self.assertRaisesRegex(ValueError, "10x10 coordinate bin"):
            LowFidelityEventV2(
                step_id=1,
                action_type="click",
                action_argument="EXACT_SECRET_COORDINATE:[987,654]",
                foreground_app="unknown",
                screen_text_added=(),
                screen_text_removed=(),
                screen_change="none",
                executor_result="unknown",
            )


if __name__ == "__main__":
    unittest.main()
