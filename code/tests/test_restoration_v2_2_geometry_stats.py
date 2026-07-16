from __future__ import annotations

import math
import unittest

from causalcache.restoration_v2_2_geometry_stats import (
    DEFAULT_BOOTSTRAP_CONFIDENCE,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    StateMetricRow,
    WinTieLoss,
    aggregate_trajectory_first,
    interaction_geometry,
    linear_quantile,
    paired_trajectory_bootstrap,
    ratio_of_means,
    win_tie_loss,
)


def _state(
    role: str,
    trajectory_id: str,
    state_id: str,
    *,
    left: float,
    right: float,
) -> StateMetricRow:
    return StateMetricRow(
        role=role,
        trajectory_id=trajectory_id,
        state_id=state_id,
        metrics={"left": left, "right": right},
    )


class RestorationV22GeometryStatsTest(unittest.TestCase):
    def test_trajectory_first_aggregation_prevents_state_pseudoreplication(
        self,
    ) -> None:
        states = [
            _state("train", "a", "a-1", left=0.0, right=4.0),
            _state("train", "a", "a-2", left=2.0, right=6.0),
            _state("train", "b", "b-1", left=9.0, right=1.0),
        ]
        result = aggregate_trajectory_first(states, ("left", "right"))
        self.assertEqual([row.trajectory_id for row in result], ["a", "b"])
        self.assertEqual(result[0].state_count, 2)
        self.assertEqual(result[0].metrics, {"left": 1.0, "right": 5.0})
        self.assertEqual(result[1].metrics["left"], 9.0)
        self.assertEqual(
            sum(row.metrics["left"] for row in result) / len(result),
            5.0,
        )
        self.assertNotEqual(5.0, (0.0 + 2.0 + 9.0) / 3.0)

    def test_aggregation_accepts_plain_mappings_and_is_order_invariant(self) -> None:
        rows = [
            {
                "role": "dev",
                "trajectory_id": "z",
                "state_id": "2",
                "metrics": {"score": 3.0},
            },
            {
                "role": "dev",
                "trajectory_id": "z",
                "state_id": "1",
                "metrics": {"score": 1.0},
            },
        ]
        self.assertEqual(
            aggregate_trajectory_first(rows, ("score",)),
            aggregate_trajectory_first(tuple(reversed(rows)), ("score",)),
        )
        self.assertEqual(
            aggregate_trajectory_first(rows, ("score",))[0].metrics["score"],
            2.0,
        )

    def test_aggregation_rejects_duplicate_or_missing_state_metrics(self) -> None:
        duplicate = _state("train", "a", "1", left=1.0, right=0.0)
        with self.assertRaisesRegex(ValueError, "duplicate state identity"):
            aggregate_trajectory_first([duplicate, duplicate], ("left",))
        with self.assertRaisesRegex(ValueError, "lacks requested metrics"):
            aggregate_trajectory_first([duplicate], ("missing",))
        with self.assertRaisesRegex(ValueError, "exactly role"):
            aggregate_trajectory_first(
                [
                    {
                        "role": "train",
                        "trajectory_id": "a",
                        "state_id": "1",
                        "metrics": {"left": 1.0},
                        "extra": 1,
                    }
                ],
                ("left",),
            )

    def test_linear_quantile_uses_deterministic_linear_interpolation(self) -> None:
        self.assertEqual(linear_quantile([10.0, 0.0], 0.25), 2.5)
        self.assertEqual(linear_quantile([0.0, 10.0, 20.0, 30.0], 0.25), 7.5)
        self.assertAlmostEqual(
            linear_quantile([0.0, 10.0, 20.0, 30.0], 0.90),
            27.0,
        )
        self.assertEqual(linear_quantile([4.0], 0.37), 4.0)
        with self.assertRaises(ValueError):
            linear_quantile([], 0.5)
        with self.assertRaises(ValueError):
            linear_quantile([1.0], 1.1)
        with self.assertRaises(ValueError):
            linear_quantile([math.nan], 0.5)

    def test_ratio_of_means_is_not_mean_of_per_pair_ratios(self) -> None:
        self.assertEqual(ratio_of_means([1.0, 9.0], [1.0, 3.0]), 2.5)
        self.assertNotEqual(2.5, ((1.0 / 1.0) + (9.0 / 3.0)) / 2.0)
        self.assertIsNone(ratio_of_means([1.0, 2.0], [1e-13, -1e-13]))
        with self.assertRaises(ValueError):
            ratio_of_means([1.0], [1.0, 2.0])

    def test_win_tie_loss_uses_paired_trajectory_values(self) -> None:
        result = win_tie_loss(
            [2.0, 1.0 + 5e-13, 0.0],
            [1.0, 1.0, 1.0],
        )
        self.assertEqual(result, WinTieLoss(wins=1, ties=1, losses=1))
        self.assertEqual(result.count, 3)

    def test_bootstrap_is_split_specific_and_overall_stratified(self) -> None:
        states = [
            _state("train", "a", "a-1", left=2.0, right=1.0),
            _state("train", "a", "a-2", left=4.0, right=3.0),
            _state("train", "b", "b-1", left=4.0, right=1.0),
            _state("dev", "c", "c-1", left=7.0, right=2.0),
        ]
        result = paired_trajectory_bootstrap(
            states,
            "left",
            "right",
            train_role="train",
            development_role="dev",
        )
        repeated = paired_trajectory_bootstrap(
            tuple(reversed(states)),
            "left",
            "right",
            train_role="train",
            development_role="dev",
        )
        self.assertEqual(result, repeated)
        self.assertEqual(result.resamples, DEFAULT_BOOTSTRAP_RESAMPLES)
        self.assertEqual(result.seed, DEFAULT_BOOTSTRAP_SEED)
        self.assertEqual(result.confidence, DEFAULT_BOOTSTRAP_CONFIDENCE)

        self.assertEqual(result.train.state_count, 3)
        self.assertEqual(result.train.trajectory_count, 2)
        self.assertEqual(result.train.mean_difference, 2.0)
        self.assertEqual(result.train.mean_difference_interval.lower, 1.0)
        self.assertEqual(result.train.mean_difference_interval.upper, 3.0)
        self.assertEqual(result.train.win_tie_loss, WinTieLoss(2, 0, 0))

        self.assertEqual(result.development.mean_difference, 5.0)
        self.assertEqual(result.development.mean_difference_interval.lower, 5.0)
        self.assertEqual(result.development.mean_difference_interval.upper, 5.0)
        self.assertEqual(result.overall_stratified.trajectory_count, 3)
        self.assertEqual(result.overall_stratified.state_count, 4)
        self.assertEqual(result.overall_stratified.mean_difference, 3.0)
        self.assertAlmostEqual(
            result.overall_stratified.mean_difference_interval.lower,
            7.0 / 3.0,
        )
        self.assertAlmostEqual(
            result.overall_stratified.mean_difference_interval.upper,
            11.0 / 3.0,
        )

    def test_bootstrap_rejects_state_rows_outside_the_two_strata(self) -> None:
        states = [
            _state("train", "a", "1", left=1.0, right=0.0),
            _state("dev", "b", "1", left=1.0, right=0.0),
            _state("confirm", "c", "1", left=1.0, right=0.0),
        ]
        with self.assertRaisesRegex(ValueError, "unexpected roles"):
            paired_trajectory_bootstrap(
                states,
                "left",
                "right",
                train_role="train",
                development_role="dev",
                resamples=10,
            )

    def test_interaction_geometry_normalizes_per_row_and_tracks_redundancy(
        self,
    ) -> None:
        result = interaction_geometry(
            [2.0, -1.0, -3.0],
            [1.0, -1e-11, -1e-13],
            summary_only_distance=2.0,
        )
        self.assertEqual(result.interaction_count, 3)
        self.assertEqual(result.normalized_mean_absolute_interaction, 1.0)
        self.assertAlmostEqual(result.redundancy_share, 2.0 / 3.0)
        self.assertEqual(result.negative_marginal_count, 1)
        self.assertEqual(result.negative_marginal_rate, 1.0 / 3.0)
        self.assertTrue(result.has_negative_marginal)

        empty = interaction_geometry(
            [],
            [],
            summary_only_distance=0.0,
        )
        self.assertEqual(empty.normalized_mean_absolute_interaction, 0.0)
        self.assertIsNone(empty.redundancy_share)
        self.assertEqual(empty.negative_marginal_rate, 0.0)
        self.assertFalse(empty.has_negative_marginal)

    def test_input_validation_rejects_nonfinite_metrics_and_bad_bootstrap(self) -> None:
        with self.assertRaises(ValueError):
            StateMetricRow("train", "a", "1", {"x": math.inf})
        states = [
            _state("train", "a", "1", left=1.0, right=0.0),
            _state("dev", "b", "1", left=1.0, right=0.0),
        ]
        with self.assertRaises(ValueError):
            paired_trajectory_bootstrap(
                states,
                "left",
                "right",
                train_role="train",
                development_role="dev",
                resamples=0,
            )
        with self.assertRaises(ValueError):
            interaction_geometry(
                [1.0],
                [1.0],
                summary_only_distance=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
