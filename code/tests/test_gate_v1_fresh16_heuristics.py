from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import causalcache.gate_v1_fresh16_heuristics as heuristic_module
from causalcache.gate_v1_fresh16_heuristics import (
    FRESH16_BUDGET_EVENT_CAPACITY,
    FRESH16_NATURAL_CANDIDATE_PREFIXES,
    ocr_rgb_similarity_selection,
    policy_vision_similarity_selection,
    recent_selection,
    select_similarity_top_b,
    validate_natural_candidate_event_step_ids,
)
from causalcache.restoration_v2_baselines import (
    CANDIDATE_EVENT_STEP_IDS,
    RESIZED_RGB_BYTE_COUNT,
    ocr_and_rgb_similarity_baseline,
    recent_selection as legacy_recent_selection,
    select_top_two,
)


def _solid_rgb(red: int, green: int, blue: int) -> bytes:
    return bytes((red, green, blue)) * (RESIZED_RGB_BYTE_COUNT // 3)


class Fresh16VariableNHeuristicTest(unittest.TestCase):
    def test_natural_prefix_and_budget_contract_fail_closed(self) -> None:
        self.assertEqual(FRESH16_BUDGET_EVENT_CAPACITY, 2)
        self.assertEqual(
            FRESH16_NATURAL_CANDIDATE_PREFIXES,
            ((1, 2), (1, 2, 3), (1, 2, 3, 4)),
        )
        for candidates in FRESH16_NATURAL_CANDIDATE_PREFIXES:
            with self.subTest(candidates=candidates):
                self.assertEqual(
                    validate_natural_candidate_event_step_ids(candidates),
                    candidates,
                )
        for invalid in ((1,), (2, 3), (1, 3, 2), (1, 2, 3, 4, 5)):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "natural prefixes"):
                    validate_natural_candidate_event_step_ids(invalid)
        with self.assertRaisesRegex(TypeError, "ordered iterable"):
            validate_natural_candidate_event_step_ids("1,2")
        with self.assertRaisesRegex(TypeError, "integers"):
            validate_natural_candidate_event_step_ids((1, True))
        with self.assertRaisesRegex(ValueError, "must equal 2"):
            recent_selection((1, 2), budget_event_capacity=1)
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            recent_selection((1, 2), budget_event_capacity=True)

    def test_recent_uses_last_min_budget_n_across_variable_batch_geometry(self) -> None:
        batch = FRESH16_NATURAL_CANDIDATE_PREFIXES
        self.assertEqual(
            tuple(recent_selection(candidates) for candidates in batch),
            ((1, 2), (2, 3), (3, 4)),
        )

    def test_similarity_scores_every_candidate_and_force_fills_min_budget_n(self) -> None:
        score_batch = (
            ((1, 2), {1: -10.0, 2: -20.0}),
            ((1, 2, 3), {1: 0.1, 2: 0.9, 3: 0.8}),
            ((1, 2, 3, 4), {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}),
        )
        results = tuple(
            select_similarity_top_b(scores, event_step_ids=candidates)
            for candidates, scores in score_batch
        )
        self.assertEqual(
            tuple(result.selected_event_step_ids for result in results),
            ((1, 2), (2, 3), (1, 2)),
        )
        self.assertEqual(
            tuple(len(result.scores_by_event_step) for result in results),
            (2, 3, 4),
        )
        self.assertEqual(results[0].ranked_event_step_ids, (1, 2))

    def test_frozen_tolerance_ties_prefer_lower_event_step_id(self) -> None:
        result = select_similarity_top_b(
            {1: 0.9, 2: 0.9000000005, 3: 0.8},
            event_step_ids=(1, 2, 3),
        )
        self.assertEqual(result.ranked_event_step_ids, (1, 2, 3))
        self.assertEqual(result.selected_event_step_ids, (1, 2))

    def test_similarity_mapping_and_numeric_values_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, r"missing=\[3\]"):
            select_similarity_top_b({1: 1.0, 2: 0.0}, event_step_ids=(1, 2, 3))
        with self.assertRaisesRegex(ValueError, r"extra=\[4\]"):
            select_similarity_top_b(
                {1: 1.0, 2: 0.0, 3: -1.0, 4: -2.0},
                event_step_ids=(1, 2, 3),
            )
        with self.assertRaisesRegex(TypeError, "real number"):
            select_similarity_top_b({1: True, 2: 0.0}, event_step_ids=(1, 2))
        with self.assertRaisesRegex(ValueError, "finite"):
            select_similarity_top_b({1: math.nan, 2: 0.0}, event_step_ids=(1, 2))

    def test_ocr_rgb_scores_all_candidates_for_each_variable_geometry(self) -> None:
        red = _solid_rgb(255, 0, 0)
        blue = _solid_rgb(0, 0, 255)
        for candidates in FRESH16_NATURAL_CANDIDATE_PREFIXES:
            with self.subTest(candidates=candidates), patch.object(
                heuristic_module,
                "joint_rgb_histogram_cosine",
                wraps=heuristic_module.joint_rgb_histogram_cosine,
            ) as rgb_score:
                result = ocr_rgb_similarity_selection(
                    event_step_ids=candidates,
                    event_ocr_tokens={step: ["Settings"] for step in candidates},
                    current_ocr_tokens=["Settings"],
                    event_resized_rgb_bytes={
                        step: red if step % 2 else blue for step in candidates
                    },
                    current_resized_rgb_bytes=red,
                )
                self.assertEqual(rgb_score.call_count, len(candidates))
                self.assertEqual(len(result.scores_by_event_step), len(candidates))
                self.assertEqual(len(result.selected_event_step_ids), 2)

    def test_ocr_rgb_requires_exact_candidate_coverage(self) -> None:
        black = _solid_rgb(0, 0, 0)
        with self.assertRaisesRegex(ValueError, r"missing=\[3\]"):
            ocr_rgb_similarity_selection(
                event_step_ids=(1, 2, 3),
                event_ocr_tokens={1: [], 2: []},
                current_ocr_tokens=[],
                event_resized_rgb_bytes={1: black, 2: black, 3: black},
                current_resized_rgb_bytes=black,
            )
        with self.assertRaisesRegex(ValueError, r"extra=\[3\]"):
            ocr_rgb_similarity_selection(
                event_step_ids=(1, 2),
                event_ocr_tokens={1: [], 2: []},
                current_ocr_tokens=[],
                event_resized_rgb_bytes={1: black, 2: black, 3: black},
                current_resized_rgb_bytes=black,
            )


class Fresh16N4CompatibilityTest(unittest.TestCase):
    def test_recent_n4_exactly_matches_legacy_behavior(self) -> None:
        self.assertEqual(
            recent_selection(CANDIDATE_EVENT_STEP_IDS),
            legacy_recent_selection(CANDIDATE_EVENT_STEP_IDS),
        )

    def test_generic_and_policy_vision_n4_match_legacy_top_two(self) -> None:
        score_cases = (
            {1: 0.1, 2: 0.4, 3: 0.3, 4: 0.2},
            {1: 0.9, 2: 0.9000000005, 3: 0.8, 4: 0.7},
            {1: -1.0, 2: -2.0, 3: -3.0, 4: -4.0},
        )
        for scores in score_cases:
            with self.subTest(scores=scores):
                expected = select_top_two(scores)
                self.assertEqual(
                    select_similarity_top_b(
                        scores,
                        event_step_ids=CANDIDATE_EVENT_STEP_IDS,
                    ),
                    expected,
                )
                self.assertEqual(
                    policy_vision_similarity_selection(
                        scores,
                        event_step_ids=CANDIDATE_EVENT_STEP_IDS,
                    ),
                    expected,
                )

    def test_ocr_rgb_n4_exactly_matches_legacy_behavior(self) -> None:
        red = _solid_rgb(255, 0, 0)
        blue = _solid_rgb(0, 0, 255)
        arguments = {
            "event_ocr_tokens": {
                1: ["Settings", "WiFi"],
                2: ["Other"],
                3: ["Settings", "WiFi"],
                4: ["Other"],
            },
            "current_ocr_tokens": ["Settings", "WiFi"],
            "event_resized_rgb_bytes": {1: blue, 2: red, 3: red, 4: blue},
            "current_resized_rgb_bytes": red,
        }
        expected = ocr_and_rgb_similarity_baseline(**arguments)
        actual = ocr_rgb_similarity_selection(
            event_step_ids=CANDIDATE_EVENT_STEP_IDS,
            **arguments,
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
