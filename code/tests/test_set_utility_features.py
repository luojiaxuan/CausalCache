from __future__ import annotations

import inspect
import unittest

from causalcache.gate_v1_data import HASH_DIMENSION, signed_hash64
from causalcache.low_fidelity_v2 import LOW_FIDELITY_V2_KEYS, LowFidelityEventV2
from causalcache.restoration_v2_baselines import (
    RESIZED_RGB_HEIGHT,
    RESIZED_RGB_WIDTH,
    joint_rgb_histogram_cosine,
    ocr_token_set_jaccard,
)
from causalcache.set_utility_features import (
    CONTEXT_FEATURE_DIMENSION,
    EVENT_FEATURE_DIMENSION,
    EVENT_NUMERIC_FEATURE_NAMES,
    MAX_CANDIDATE_EVENTS,
    PAIR_FEATURE_DIMENSION,
    PAIR_FEATURE_NAMES,
    QUERY_FEATURE_DIMENSION,
    SetUtilityEventInput,
    build_set_utility_feature_state,
    event_ocr_rgb_score,
    event_recency_score,
)


def _solid_rgb(red: int, green: int, blue: int) -> bytes:
    return bytes((red, green, blue)) * (RESIZED_RGB_WIDTH * RESIZED_RGB_HEIGHT)


BLACK = _solid_rgb(0, 0, 0)
RED = _solid_rgb(255, 0, 0)
BLUE = _solid_rgb(0, 0, 255)


def _event(
    step_id: int,
    *,
    tokens: tuple[str, ...],
    pixels: bytes,
    action_type: str = "click",
    action_argument: str = "coordinate_bin:x1_y2",
    app: str = "settings",
    result: str = "accepted",
) -> SetUtilityEventInput:
    return SetUtilityEventInput(
        low_fidelity_v2=LowFidelityEventV2(
            step_id=step_id,
            action_type=action_type,
            action_argument=action_argument,
            foreground_app=app,
            screen_text_added=("added",),
            screen_text_removed=(),
            screen_change="medium",
            executor_result=result,
        ),
        post_ocr_spatial_tokens=tokens,
        post_resized_rgb_bytes=pixels,
    )


def _state(
    events: tuple[SetUtilityEventInput, ...],
    *,
    pad_to: int = MAX_CANDIDATE_EVENTS,
):
    return build_set_utility_feature_state(
        source_id="source",
        state_id="source:decision:020",
        decision_step_id=20,
        instruction="Open Settings and enable Wi-Fi",
        current_ocr_spatial_tokens=("Settings", "Wi-Fi"),
        current_resized_rgb_bytes=RED,
        events=events,
        pad_to=pad_to,
    )


class SetUtilityFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = (
            _event(3, tokens=("old", "screen"), pixels=BLACK),
            _event(11, tokens=("Settings", "Wi-Fi"), pixels=RED),
            _event(
                17,
                tokens=("Settings", "other"),
                pixels=BLUE,
                action_type="wait",
                action_argument="wait",
                app="launcher",
                result="unknown",
            ),
        )

    def test_dimensions_mask_and_zero_padding(self) -> None:
        state = _state(self.events)
        self.assertEqual(len(state.query_features), QUERY_FEATURE_DIMENSION)
        self.assertEqual(QUERY_FEATURE_DIMENSION, HASH_DIMENSION)
        self.assertEqual(len(state.context_features), CONTEXT_FEATURE_DIMENSION)
        self.assertEqual(len(state.event_features), MAX_CANDIDATE_EVENTS)
        self.assertEqual(len(state.pair_features), MAX_CANDIDATE_EVENTS)
        self.assertEqual(state.event_mask[:3], (True, True, True))
        self.assertEqual(state.event_mask[3:], (False,) * 13)
        self.assertTrue(
            all(len(row) == EVENT_FEATURE_DIMENSION for row in state.event_features)
        )
        self.assertTrue(
            all(
                len(row) == MAX_CANDIDATE_EVENTS
                and all(len(cell) == PAIR_FEATURE_DIMENSION for cell in row)
                for row in state.pair_features
            )
        )
        self.assertEqual(
            state.event_features[3:],
            ((0.0,) * EVENT_FEATURE_DIMENSION,) * 13,
        )
        self.assertTrue(
            all(
                cell == (0.0,) * PAIR_FEATURE_DIMENSION
                for row in state.pair_features[3:]
                for cell in row
            )
        )

    def test_query_and_event_semantic_hash_replay_gate_v1(self) -> None:
        state = _state(self.events, pad_to=3)
        expected_query = signed_hash64(
            (
                ("instruction", "Open Settings and enable Wi-Fi"),
                ("current_ocr_spatial_token", ("Settings", "Wi-Fi")),
            )
        )
        self.assertEqual(state.query_features, expected_query)
        low = self.events[0].low_fidelity_v2.to_ordered_dict()
        expected_event = signed_hash64(
            tuple((name, low[name]) for name in LOW_FIDELITY_V2_KEYS)
            + (
                (
                    "candidate_post_ocr_spatial_token",
                    self.events[0].post_ocr_spatial_tokens,
                ),
            )
        )
        self.assertEqual(state.event_features[0][:HASH_DIMENSION], expected_event)

    def test_explicit_ocr_rgb_reproduces_frozen_baseline(self) -> None:
        state = _state(self.events, pad_to=3)
        ocr_index = EVENT_NUMERIC_FEATURE_NAMES.index("current_ocr_jaccard")
        rgb_index = EVENT_NUMERIC_FEATURE_NAMES.index(
            "current_rgb_histogram_cosine"
        )
        for index, event in enumerate(self.events):
            row = state.event_features[index]
            expected_ocr = ocr_token_set_jaccard(
                event.post_ocr_spatial_tokens,
                ("Settings", "Wi-Fi"),
            )
            expected_rgb = joint_rgb_histogram_cosine(
                event.post_resized_rgb_bytes,
                RED,
            )
            self.assertAlmostEqual(row[HASH_DIMENSION + ocr_index], expected_ocr)
            self.assertAlmostEqual(row[HASH_DIMENSION + rgb_index], expected_rgb)
            self.assertAlmostEqual(
                event_ocr_rgb_score(row),
                0.5 * (expected_ocr + expected_rgb),
            )

    def test_explicit_recency_reproduces_recent_ranking(self) -> None:
        state = _state(self.events, pad_to=3)
        scores = {
            event_id: event_recency_score(state.event_features[index])
            for index, event_id in enumerate(state.event_step_ids)
        }
        ranking = tuple(sorted(scores, key=lambda item: (-scores[item], item)))
        self.assertEqual(ranking, (17, 11, 3))
        self.assertEqual(scores[17], 1.0 / 3.0)
        self.assertEqual(scores[11], 1.0 / 9.0)

    def test_pair_features_match_event_event_ocr_rgb_and_time(self) -> None:
        state = _state(self.events, pad_to=3)
        pair = state.pair_feature(3, 11)
        self.assertAlmostEqual(
            pair[PAIR_FEATURE_NAMES.index("event_ocr_jaccard")],
            ocr_token_set_jaccard(("old", "screen"), ("Settings", "Wi-Fi")),
        )
        self.assertAlmostEqual(
            pair[PAIR_FEATURE_NAMES.index("event_rgb_histogram_cosine")],
            joint_rgb_histogram_cosine(BLACK, RED),
        )
        self.assertEqual(
            pair[PAIR_FEATURE_NAMES.index("absolute_step_gap_fraction")],
            8.0 / 20.0,
        )
        self.assertEqual(
            pair[PAIR_FEATURE_NAMES.index("reciprocal_step_gap")],
            1.0 / 9.0,
        )
        self.assertEqual(pair, state.pair_feature(11, 3))

    def test_event_permutation_is_exactly_equivariant(self) -> None:
        original = _state(self.events, pad_to=5)
        permutation = (2, 0, 1)
        permuted = _state(tuple(self.events[index] for index in permutation), pad_to=5)
        self.assertEqual(permuted.query_features, original.query_features)
        self.assertEqual(permuted.context_features, original.context_features)
        self.assertEqual(
            permuted.event_step_ids,
            tuple(original.event_step_ids[index] for index in permutation),
        )
        for new_index, old_index in enumerate(permutation):
            self.assertEqual(
                permuted.event_features[new_index],
                original.event_features[old_index],
            )
            for new_right, old_right in enumerate(permutation):
                self.assertEqual(
                    permuted.pair_features[new_index][new_right],
                    original.pair_features[old_index][old_right],
                )
        self.assertEqual(permuted.event_features[3:], original.event_features[3:])
        self.assertEqual(permuted.event_mask, original.event_mask)

    def test_one_to_sixteen_events_and_padding_geometry(self) -> None:
        one = _state((self.events[0],), pad_to=1)
        self.assertEqual(one.event_mask, (True,))
        sixteen = tuple(
            _event(step, tokens=(f"token-{step}",), pixels=BLACK)
            for step in range(1, 17)
        )
        state = build_set_utility_feature_state(
            source_id="source",
            state_id="state",
            decision_step_id=18,
            instruction="do task",
            current_ocr_spatial_tokens=("current",),
            current_resized_rgb_bytes=BLACK,
            events=sixteen,
            pad_to=16,
        )
        self.assertEqual(state.event_count, 16)
        self.assertEqual(state.event_mask, (True,) * 16)
        with self.assertRaisesRegex(ValueError, "exceeds.*16"):
            build_set_utility_feature_state(
                source_id="source",
                state_id="state",
                decision_step_id=20,
                instruction="do task",
                current_ocr_spatial_tokens=("current",),
                current_resized_rgb_bytes=BLACK,
                events=sixteen + (_event(17, tokens=("x",), pixels=BLACK),),
            )
        with self.assertRaisesRegex(ValueError, "pad_to"):
            _state(self.events, pad_to=2)

    def test_mapping_firewall_rejects_label_and_budget_fields(self) -> None:
        event = self.events[0]
        mapping = {
            "low_fidelity_v2": event.low_fidelity_v2.to_ordered_dict(),
            "post_ocr_spatial_tokens": list(event.post_ocr_spatial_tokens),
            "post_resized_rgb_bytes": event.post_resized_rgb_bytes,
        }
        parsed = SetUtilityEventInput.from_mapping(mapping)
        self.assertEqual(parsed, event)
        for forbidden in (
            "distance",
            "distance_kl",
            "utility",
            "budget",
            "instruction_app_group_sha256",
        ):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(ValueError, "label-blind"):
                    SetUtilityEventInput.from_mapping({**mapping, forbidden: 1})

    def test_public_builder_contract_has_no_budget_or_label_input(self) -> None:
        parameters = inspect.signature(build_set_utility_feature_state).parameters
        self.assertFalse(
            set(parameters)
            & {"budget", "remaining_budget", "distance", "distance_kl", "utility"}
        )

    def test_rejects_duplicate_or_future_events(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            _state((self.events[0], self.events[0]), pad_to=2)
        future = _event(20, tokens=("future",), pixels=BLACK)
        with self.assertRaisesRegex(ValueError, "strictly precede"):
            _state((future,), pad_to=1)


if __name__ == "__main__":
    unittest.main()
