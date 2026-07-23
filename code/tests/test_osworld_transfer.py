from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from causalcache.osworld_transfer import (
    load_arm_results,
    paired_bootstrap_ci,
    reduce_transfer_results,
)


def _result(score: float) -> dict[str, object]:
    return {
        "score": score,
        "success": score > 0,
        "completed_steps": 2,
        "termination_reason": "policy_done",
        "steps": [],
    }


class OSWorldTransferTests(unittest.TestCase):
    def test_counts_terminal_run_failure_as_zero_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failure = root / "chrome/task/attempts/attempt/failure.json"
            failure.parent.mkdir(parents=True)
            (failure.parent / "checkpoint.json").write_text(
                json.dumps(
                    {
                        "completed_steps": 1,
                        "steps": [
                            {
                                "policy_latency_seconds": 2.0,
                                "policy_response": {
                                    "source": "failed_profile",
                                    "queue_seconds": 1.0,
                                    "runtime": {
                                        "generation_seconds": 0.5,
                                        "peak_gpu_memory_allocated_bytes": 123,
                                    },
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            failure.write_text(
                json.dumps(
                    {
                        "task": {"domain": "chrome", "task_id": "task"},
                        "completed_steps": 1,
                        "error_type": "HTTPError",
                        "error_message": "HTTP Error 500",
                        "failed_at": "2026-01-01T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            results = load_arm_results(
                root,
                expected_profile_id="failed_profile",
                expected_task_count=1,
            )
            self.assertEqual(results[("chrome", "task")]["score"], 0.0)
            self.assertEqual(len(results[("chrome", "task")]["steps"]), 1)
            self.assertEqual(
                results[("chrome", "task")]["termination_reason"],
                "run_failure:HTTPError",
            )

    def test_reduces_paired_scores(self) -> None:
        config = {
            "profiles": [
                {"arm": "frozen"},
                {"arm": "terminal_s60"},
            ]
        }
        identities = (("chrome", "a"), ("os", "b"))
        summary = reduce_transfer_results(
            config=config,
            arm_results={
                "frozen": {
                    identities[0]: _result(0.0),
                    identities[1]: _result(1.0),
                },
                "terminal_s60": {
                    identities[0]: _result(1.0),
                    identities[1]: _result(0.0),
                },
            },
        )
        self.assertEqual(summary["paired"]["terminal_s60_wins"], 1)
        self.assertEqual(summary["paired"]["terminal_s60_losses"], 1)
        self.assertEqual(
            summary["paired"]["mean_score_delta_terminal_s60_minus_frozen"], 0.0
        )
        self.assertEqual(summary["paired"]["adapted_arm"], "terminal_s60")

    def test_reduces_generic_adapted_arm(self) -> None:
        config = {
            "profiles": [
                {"arm": "frozen"},
                {"arm": "v3_e1"},
            ]
        }
        identity = ("chrome", "a")
        summary = reduce_transfer_results(
            config=config,
            arm_results={
                "frozen": {identity: _result(0.0)},
                "v3_e1": {identity: _result(1.0)},
            },
        )
        self.assertEqual(summary["paired"]["adapted_arm"], "v3_e1")
        self.assertEqual(summary["paired"]["adapted_wins"], 1)
        self.assertEqual(
            summary["paired"]["mean_score_delta_adapted_minus_frozen"], 1.0
        )
        self.assertEqual(
            summary["paired"]["task_scores"][0]["adapted"], 1.0
        )

    def test_bootstrap_is_deterministic(self) -> None:
        first = paired_bootstrap_ci([1.0, 0.0, -1.0], seed=7, draws=100)
        second = paired_bootstrap_ci([1.0, 0.0, -1.0], seed=7, draws=100)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
