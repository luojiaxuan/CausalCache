import tempfile
import unittest
from pathlib import Path

from causalcache.diagnostic_runner import (
    _attribution_for_budget,
    canonical_json_sha256,
    coalition_key,
    enumerate_bounded_coalitions,
    sha256_file,
    validate_visual_accounting,
)


class DiagnosticRunnerTest(unittest.TestCase):
    def test_frozen_state_coalition_counts(self) -> None:
        step_four = enumerate_bounded_coalitions(
            [1, 2, 3],
            maximum_restored_events=2,
        )
        step_eight = enumerate_bounded_coalitions(
            list(range(1, 8)),
            maximum_restored_events=2,
        )
        self.assertEqual(len(step_four), 7)
        self.assertEqual(len(step_eight), 29)
        self.assertEqual(len(step_four) + len(step_eight) + 2, 38)
        self.assertEqual(coalition_key(frozenset({3, 1})), "1,3")

    def test_visual_accounting_uses_effective_not_target_tokens(self) -> None:
        intervention = {
            "images_per_restored_event": 2,
            "effective_visual_tokens_per_image": 238,
            "event_cost_visual_tokens": 476,
        }
        result = validate_visual_accounting(
            {"image_count": 5, "effective_visual_tokens": 1190},
            restored_event_count=2,
            intervention=intervention,
        )
        self.assertEqual(result["incremental_memory_visual_tokens"], 952)
        with self.assertRaisesRegex(ValueError, "effective visual-token mismatch"):
            validate_visual_accounting(
                {"image_count": 5, "effective_visual_tokens": 1280},
                restored_event_count=2,
                intervention=intervention,
            )

    def test_hash_helpers_are_deterministic(self) -> None:
        self.assertEqual(
            canonical_json_sha256({"b": 2, "a": 1}),
            canonical_json_sha256({"a": 1, "b": 2}),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload"
            path.write_bytes(b"causalcache")
            self.assertEqual(
                sha256_file(path),
                "7d8b316f4170596bd299c2e2e7a6f4eff66c66af92c05c9e3dacdb27880c9457",
            )

    def test_exact_and_sampled_attribution_consume_cached_coalitions(self) -> None:
        event_ids = (1, 2, 3)
        values = {1: 3.0, 2: 2.0, 3: 1.0}
        distances = {}
        for coalition in enumerate_bounded_coalitions(
            event_ids,
            maximum_restored_events=2,
        ):
            distances[coalition] = 10.0 - sum(values[event_id] for event_id in coalition)
        result = _attribution_for_budget(
            distances=distances,
            event_ids=event_ids,
            event_cost=1,
            budget=2,
            epsilon=1e-4,
            sample_counts=[4],
            seeds=[0],
        )
        self.assertEqual(result["exact_scores"], values)
        self.assertEqual(result["exact_selected"]["coalition"], [1, 2])
        self.assertEqual(result["sample_sweep"][0]["top_budget_jaccard_vs_exact"], 1.0)


if __name__ == "__main__":
    unittest.main()
