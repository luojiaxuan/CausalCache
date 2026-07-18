from __future__ import annotations

import gzip
import json
import unittest

from causalcache.long_horizon_incidence import (
    candidate_count_for_decision_index,
    horizon_bin,
    summarize_gzip_trace_shard,
    summarize_horizon_incidence,
)


def _record(plan_index: int, step_count: int) -> dict[str, object]:
    return {
        "plan_index": plan_index,
        "model_step_count": step_count,
        "steps": [{} for _ in range(step_count)],
    }


class LongHorizonIncidenceTest(unittest.TestCase):
    def test_candidate_geometry_and_bins(self) -> None:
        self.assertEqual(
            [candidate_count_for_decision_index(index) for index in range(5)],
            [0, 0, 1, 2, 3],
        )
        self.assertEqual(horizon_bin(4), "n_le_4")
        self.assertEqual(horizon_bin(5), "n_5_8")
        self.assertEqual(horizon_bin(8), "n_5_8")
        self.assertEqual(horizon_bin(9), "n_9_15")
        self.assertEqual(horizon_bin(16), "n_ge_16")

    def test_summary_keeps_zero_step_records(self) -> None:
        summary = summarize_horizon_incidence(
            (_record(0, 0), _record(1, 10), _record(2, 18)),
            artifact_label="fixture",
        )
        self.assertEqual(summary["record_count"], 3)
        self.assertEqual(summary["zero_step_episode_count"], 1)
        self.assertEqual(summary["decision_count"], 28)
        self.assertEqual(summary["n_ge_8"]["episode_count_among_nonempty"], 2)
        self.assertEqual(summary["n_ge_16"]["episode_count_among_nonempty"], 1)
        self.assertEqual(
            sum(summary["horizon_bin_counts"].values()),
            summary["decision_count"],
        )

    def test_gzip_jsonl_and_sha_are_deterministic(self) -> None:
        raw = b"".join(
            json.dumps(_record(index, index), sort_keys=True).encode() + b"\n"
            for index in range(3)
        )
        payload = gzip.compress(raw, mtime=0)
        first = summarize_gzip_trace_shard(payload, artifact_label="fixture")
        second = summarize_gzip_trace_shard(payload, artifact_label="fixture")
        self.assertEqual(first, second)
        self.assertEqual(first["record_count"], 3)

    def test_unsorted_or_duplicate_plan_indices_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            summarize_horizon_incidence(
                (_record(1, 1), _record(0, 1)), artifact_label="fixture"
            )
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            summarize_horizon_incidence(
                (_record(0, 1), _record(0, 1)), artifact_label="fixture"
            )

    def test_zero_step_exception_may_omit_steps(self) -> None:
        summary = summarize_horizon_incidence(
            (
                {
                    "plan_index": 0,
                    "run_status": "exception",
                    "model_step_count": 0,
                },
            ),
            artifact_label="fixture",
        )
        self.assertEqual(summary["zero_step_episode_count"], 1)
        with self.assertRaisesRegex(ValueError, "only zero-step exception"):
            summarize_horizon_incidence(
                ({"plan_index": 0, "model_step_count": 0},),
                artifact_label="fixture",
            )


if __name__ == "__main__":
    unittest.main()
