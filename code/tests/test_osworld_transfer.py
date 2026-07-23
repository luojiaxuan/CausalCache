from __future__ import annotations

import unittest

from causalcache.osworld_transfer import paired_bootstrap_ci, reduce_transfer_results


def _result(score: float) -> dict[str, object]:
    return {
        "score": score,
        "success": score > 0,
        "completed_steps": 2,
        "termination_reason": "policy_done",
    }


class OSWorldTransferTests(unittest.TestCase):
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

    def test_bootstrap_is_deterministic(self) -> None:
        first = paired_bootstrap_ci([1.0, 0.0, -1.0], seed=7, draws=100)
        second = paired_bootstrap_ci([1.0, 0.0, -1.0], seed=7, draws=100)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
