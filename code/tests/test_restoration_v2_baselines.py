from __future__ import annotations

import inspect
import json
import math
import unittest
from pathlib import Path

from causalcache.restoration_v2_baselines import (
    CANDIDATE_EVENT_STEP_IDS,
    RESIZED_RGB_BYTE_COUNT,
    joint_rgb_histogram,
    joint_rgb_histogram_cosine,
    normalized_ocr_token_set,
    ocr_and_rgb_similarity_baseline,
    ocr_token_set_jaccard,
    recent_selection,
    select_top_two,
    summary_only_selection,
    uniform_random_exact_expectation,
    uniform_random_subsets,
)
from causalcache.restoration_v2_contract import validate_restoration_v2_contract


ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_CONFIG_PATH = ROOT / "code/configs/causalcache_restoration_v2.json"


def _solid_rgb(red: int, green: int, blue: int) -> bytes:
    return bytes((red, green, blue)) * (RESIZED_RGB_BYTE_COUNT // 3)


class RestorationV2BaselineContractTest(unittest.TestCase):
    def test_implementation_matches_frozen_baseline_contract(self) -> None:
        config = json.loads(SCIENTIFIC_CONFIG_PATH.read_text(encoding="utf-8"))
        validate_restoration_v2_contract(config)
        gate = config["restoration_gate"]
        self.assertEqual(gate["primary_budget_event_capacity"], 2)
        self.assertEqual(
            gate["baseline_contract"],
            {
                "summary_only": "select_empty",
                "recent": "select_events_3_and_4",
                "uniform_random_exact_expectation": (
                    "mean_normalized_recovery_over_all_2_of_4_subsets"
                ),
                "ocr_and_rgb_similarity": (
                    "score=0.5*set_jaccard_of_whitespace_token_sets_after_frozen_ocr_"
                    "normalization(post,current)+0.5*cosine_joint_rgb_histogram_16x16x16"
                    "(post,current);select_top_2"
                ),
                "frozen_policy_vision_embedding_similarity": (
                    "cosine_of_l2_normalized_mean_last_visual_encoder_output_after_"
                    "spatial_merger(post,current);select_top_2"
                ),
                "tie_break": "higher_score_then_lower_event_step_id",
                "formula_source_hashes_required_before_confirm": True,
            },
        )

    def test_summary_recent_and_candidate_identity_are_exact(self) -> None:
        self.assertEqual(summary_only_selection(CANDIDATE_EVENT_STEP_IDS), ())
        self.assertEqual(recent_selection(CANDIDATE_EVENT_STEP_IDS), (3, 4))
        for invalid in ((1, 2, 3), (1, 2, 3, 5), (4, 3, 2, 1)):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "must equal"):
                    recent_selection(invalid)

    def test_random_is_exact_six_subset_expectation_without_seed(self) -> None:
        subsets = uniform_random_subsets(CANDIDATE_EVENT_STEP_IDS)
        self.assertEqual(
            subsets,
            ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)),
        )
        recoveries = {subset: float(index) for index, subset in enumerate(subsets)}
        result = uniform_random_exact_expectation(
            recoveries,
            event_step_ids=CANDIDATE_EVENT_STEP_IDS,
        )
        self.assertEqual(result.subsets, subsets)
        self.assertEqual(result.normalized_recovery_by_subset[3], ((2, 3), 3.0))
        self.assertEqual(result.mean_normalized_recovery, 2.5)
        self.assertNotIn(
            "seed",
            inspect.signature(uniform_random_exact_expectation).parameters,
        )

    def test_random_expectation_rejects_incomplete_duplicate_and_nonfinite_tables(self) -> None:
        subsets = uniform_random_subsets(CANDIDATE_EVENT_STEP_IDS)
        complete = {subset: 0.1 for subset in subsets}
        incomplete = dict(complete)
        incomplete.pop((3, 4))
        with self.assertRaisesRegex(ValueError, "all and only"):
            uniform_random_exact_expectation(
                incomplete,
                event_step_ids=CANDIDATE_EVENT_STEP_IDS,
            )
        duplicate = dict(complete)
        duplicate[(2, 1)] = 0.2
        with self.assertRaisesRegex(ValueError, "duplicate normalized"):
            uniform_random_exact_expectation(
                duplicate,
                event_step_ids=CANDIDATE_EVENT_STEP_IDS,
            )
        nonfinite = dict(complete)
        nonfinite[(1, 2)] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            uniform_random_exact_expectation(
                nonfinite,
                event_step_ids=CANDIDATE_EVENT_STEP_IDS,
            )


class RestorationV2OcrRgbBaselineTest(unittest.TestCase):
    def test_ocr_normalization_set_jaccard_and_empty_identity(self) -> None:
        self.assertEqual(
            normalized_ocr_token_set([" Cafe\u0301 ", "Wi\nFi", "Wi Fi"]),
            frozenset({"Caf\u00e9", "Wi", "Fi"}),
        )
        self.assertEqual(ocr_token_set_jaccard([], []), 1.0)
        self.assertEqual(ocr_token_set_jaccard(["A", "B"], ["B", "C"]), 1 / 3)
        with self.assertRaisesRegex(TypeError, "token strings"):
            normalized_ocr_token_set("not-a-token-sequence")

    def test_joint_rgb_histogram_is_frozen_16_cubed_and_cosine_is_exact(self) -> None:
        red = _solid_rgb(255, 0, 0)
        blue = _solid_rgb(0, 0, 255)
        histogram = joint_rgb_histogram(red)
        self.assertEqual(len(histogram), 16**3)
        self.assertEqual(sum(histogram), 256 * 256)
        self.assertEqual(joint_rgb_histogram_cosine(red, red), 1.0)
        self.assertEqual(joint_rgb_histogram_cosine(red, blue), 0.0)
        with self.assertRaisesRegex(ValueError, "exactly"):
            joint_rgb_histogram(red[:-1])
        with self.assertRaisesRegex(TypeError, "bytes-like"):
            joint_rgb_histogram([(255, 0, 0)])

    def test_ocr_rgb_uses_equal_weights_and_deterministic_top_two(self) -> None:
        red = _solid_rgb(255, 0, 0)
        blue = _solid_rgb(0, 0, 255)
        result = ocr_and_rgb_similarity_baseline(
            event_ocr_tokens={
                1: ["Settings", "WiFi"],
                2: ["Other"],
                3: ["Settings", "WiFi"],
                4: ["Other"],
            },
            current_ocr_tokens=["Settings", "WiFi"],
            event_resized_rgb_bytes={1: blue, 2: red, 3: red, 4: blue},
            current_resized_rgb_bytes=red,
        )
        self.assertEqual(result.score(1), 0.5)
        self.assertEqual(result.score(2), 0.5)
        self.assertEqual(result.score(3), 1.0)
        self.assertEqual(result.score(4), 0.0)
        self.assertEqual(result.ranked_event_step_ids, (3, 1, 2, 4))
        self.assertEqual(result.selected_event_step_ids, (1, 3))

    def test_ocr_rgb_missing_or_extra_event_fields_fail_closed(self) -> None:
        rgb = _solid_rgb(0, 0, 0)
        with self.assertRaisesRegex(ValueError, r"missing=\[4\]"):
            ocr_and_rgb_similarity_baseline(
                event_ocr_tokens={1: [], 2: [], 3: []},
                current_ocr_tokens=[],
                event_resized_rgb_bytes={1: rgb, 2: rgb, 3: rgb, 4: rgb},
                current_resized_rgb_bytes=rgb,
            )
        with self.assertRaisesRegex(ValueError, r"extra=\[5\]"):
            ocr_and_rgb_similarity_baseline(
                event_ocr_tokens={1: [], 2: [], 3: [], 4: []},
                current_ocr_tokens=[],
                event_resized_rgb_bytes={1: rgb, 2: rgb, 3: rgb, 4: rgb, 5: rgb},
                current_resized_rgb_bytes=rgb,
            )


class RestorationV2PolicyVisionTieBreakTest(unittest.TestCase):
    def test_score_ties_use_frozen_tolerance_then_lower_step(self) -> None:
        result = select_top_two(
            {1: 0.9, 2: 0.9000000005, 3: 0.8, 4: 0.7}
        )
        self.assertEqual(result.ranked_event_step_ids[:2], (1, 2))
        self.assertEqual(result.selected_event_step_ids, (1, 2))

if __name__ == "__main__":
    unittest.main()
