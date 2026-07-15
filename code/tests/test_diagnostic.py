import unittest

from causalcache.diagnostic import (
    DiagnosticOutcomeRow,
    canonical_json_action,
    classify_diagnostic_outcome,
    evaluate_exact_coalition_table,
    extract_first_json_object,
    has_positive_non_recent_gain,
    normalized_recovery,
    rgb_histogram_cosine,
)


def _outcome_config() -> dict:
    return {
        "states": [{"decision_step_id": 4}, {"decision_step_id": 8}],
        "outcome_rule": {
            "allowed_statuses": [
                "INVALID",
                "NO_GO_DIAGNOSTIC",
                "INCONCLUSIVE_NEGATIVE",
                "INCONCLUSIVE_POSITIVE",
            ],
            "inconclusive_positive_requires": [
                "at_least_one_memory_sensitive_state",
                "at_least_one_state_with_budget_1024_normalized_oracle_recovery_at_least_0.25",
                "that_state_has_at_least_one_positive_exact_gain_for_a_non_recent_event",
            ],
        },
    }


class CanonicalActionTest(unittest.TestCase):
    def test_extracts_first_valid_object_and_sorts_nested_keys(self) -> None:
        text = 'prefix {not json} {"z":2,"action":{"文":"值","a":1}} {"later":true}'
        self.assertEqual(
            extract_first_json_object(text),
            {"z": 2, "action": {"文": "值", "a": 1}},
        )
        self.assertEqual(
            canonical_json_action(text),
            '{"action":{"a":1,"文":"值"},"z":2}',
        )

    def test_rejects_text_without_json_object(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not contain"):
            canonical_json_action("[1, 2, 3]")


class HistogramSimilarityTest(unittest.TestCase):
    def test_joint_rgb_histogram_is_order_invariant(self) -> None:
        left = [(255, 0, 0), (0, 255, 0), (255, 0, 0)]
        right = [(0, 255, 0), (255, 0, 0), (255, 0, 0)]
        self.assertAlmostEqual(rgb_histogram_cosine(left, right), 1.0)

    def test_disjoint_colors_have_zero_similarity(self) -> None:
        self.assertEqual(
            rgb_histogram_cosine([(255, 0, 0)], [(0, 0, 255)]),
            0.0,
        )

    def test_invalid_or_empty_pixels_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one pixel"):
            rgb_histogram_cosine([], [(0, 0, 0)])
        with self.assertRaisesRegex(ValueError, "between 0 and 255"):
            rgb_histogram_cosine([(256, 0, 0)], [(0, 0, 0)])


class ExactCoalitionDiagnosticTest(unittest.TestCase):
    def test_baselines_random_expectation_and_similarity_tie_break(self) -> None:
        costs = {1: 1, 2: 1, 3: 1}
        distances = {
            frozenset(): 10.0,
            frozenset({1}): 8.0,
            frozenset({2}): 9.0,
            frozenset({3}): 7.0,
            frozenset({1, 2}): 6.0,
            frozenset({1, 3}): 5.0,
            frozenset({2, 3}): 4.0,
        }
        result = evaluate_exact_coalition_table(
            distances,
            costs,
            budget=2,
            similarity_scores={1: 0.9, 2: 0.9, 3: 0.1},
            epsilon=1e-4,
        )
        self.assertEqual(result.recent.coalition, (2, 3))
        self.assertEqual(result.similarity.coalition, (1, 2))
        self.assertEqual(result.oracle.coalition, (2, 3))
        self.assertEqual(result.random.coalitions, ((1, 2), (1, 3), (2, 3)))
        self.assertAlmostEqual(result.random.expected_distance, 5.0)
        self.assertAlmostEqual(result.random.expected_normalized_recovery, 0.5)

    def test_oracle_tie_prefers_lower_cost_then_lexicographic(self) -> None:
        result = evaluate_exact_coalition_table(
            {
                frozenset(): 10.0,
                frozenset({1}): 3.0,
                frozenset({2}): 3.0,
                frozenset({3}): 4.0,
                frozenset({1, 2}): 3.0,
                frozenset({1, 3}): 5.0,
                frozenset({2, 3}): 5.0,
            },
            {1: 1, 2: 1, 3: 1},
            budget=2,
            similarity_scores={1: 0.0, 2: 0.0, 3: 0.0},
            epsilon=1e-4,
        )
        self.assertEqual(result.oracle.coalition, (1,))
        self.assertEqual(result.similarity.coalition, (1, 2))

    def test_negative_restoration_does_not_force_oracle_memory(self) -> None:
        result = evaluate_exact_coalition_table(
            {
                frozenset(): 1.0,
                frozenset({0}): 2.0,
                frozenset({1}): 3.0,
            },
            {0: 1, 1: 1},
            budget=1,
            similarity_scores={0: 1.0, 1: 0.0},
            epsilon=1e-4,
        )
        self.assertEqual(result.oracle.coalition, ())
        self.assertEqual(result.oracle.normalized_recovery, 0.0)
        self.assertEqual(result.recent.normalized_recovery, -2.0)
        self.assertAlmostEqual(result.random.expected_normalized_recovery, -1.5)

    def test_normalization_uses_epsilon_floor(self) -> None:
        self.assertAlmostEqual(
            normalized_recovery(summary_distance=1e-5, distance=0.0, epsilon=1e-4),
            0.1,
        )

    def test_positive_non_recent_gain_is_strictly_above_epsilon(self) -> None:
        self.assertTrue(
            has_positive_non_recent_gain(
                {1: 1e-4, 2: 2e-4},
                recent_coalition={1},
                epsilon=1e-4,
            )
        )
        self.assertFalse(
            has_positive_non_recent_gain(
                {1: 1.0, 2: 1e-4},
                recent_coalition={1},
                epsilon=1e-4,
            )
        )


class DiagnosticOutcomeTest(unittest.TestCase):
    def test_no_go_when_all_states_are_insensitive(self) -> None:
        outcome = classify_diagnostic_outcome(
            [
                DiagnosticOutcomeRow(4, False, 1.0, 1e-4, 1.0, True),
                DiagnosticOutcomeRow(8, False, 1.0, 1e-4, 1.0, True),
            ],
            config=_outcome_config(),
        )
        self.assertEqual(outcome.status, "NO_GO_DIAGNOSTIC")

    def test_no_go_when_no_oracle_utility_exceeds_epsilon(self) -> None:
        outcome = classify_diagnostic_outcome(
            [
                DiagnosticOutcomeRow(4, True, 1e-4, 1e-4, 0.5, True),
                DiagnosticOutcomeRow(8, False, -0.1, 1e-4, -1.0, False),
            ],
            config=_outcome_config(),
        )
        self.assertEqual(outcome.status, "NO_GO_DIAGNOSTIC")

    def test_positive_status_uses_frozen_threshold_boundary(self) -> None:
        outcome = classify_diagnostic_outcome(
            [
                DiagnosticOutcomeRow(4, True, 0.1, 1e-4, 0.25, True),
                DiagnosticOutcomeRow(8, False, 0.0, 1e-4, 0.0, False),
            ],
            config=_outcome_config(),
        )
        self.assertEqual(outcome.status, "INCONCLUSIVE_POSITIVE")
        self.assertEqual(outcome.positive_step_ids, (4,))

    def test_valid_non_positive_status_is_negative(self) -> None:
        outcome = classify_diagnostic_outcome(
            [
                DiagnosticOutcomeRow(4, True, 0.1, 1e-4, 0.249, True),
                DiagnosticOutcomeRow(8, False, 0.0, 1e-4, 0.0, False),
            ],
            config=_outcome_config(),
        )
        self.assertEqual(outcome.status, "INCONCLUSIVE_NEGATIVE")

    def test_positive_evidence_must_come_from_a_sensitive_state(self) -> None:
        outcome = classify_diagnostic_outcome(
            [
                DiagnosticOutcomeRow(4, True, 0.1, 1e-4, 0.1, False),
                DiagnosticOutcomeRow(8, False, 0.1, 1e-4, 0.5, True),
            ],
            config=_outcome_config(),
        )
        self.assertEqual(outcome.status, "INCONCLUSIVE_NEGATIVE")

    def test_missing_configured_state_is_invalid(self) -> None:
        outcome = classify_diagnostic_outcome(
            [DiagnosticOutcomeRow(4, True, 0.1, 1e-4, 1.0, True)],
            config=_outcome_config(),
        )
        self.assertEqual(outcome.status, "INVALID")


if __name__ == "__main__":
    unittest.main()
