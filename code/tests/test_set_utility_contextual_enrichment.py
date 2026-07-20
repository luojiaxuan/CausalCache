from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from causalcache.set_utility_contextual_enrichment import (
    materialize_contextual_enriched_inputs,
)
from causalcache.set_utility_contextual_inputs import (
    CONTEXTUAL_INPUT_STATUS,
    CONTEXTUAL_REQUIREMENT_STATUS,
)
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes
from scripts.run_set_utility_tune_selectors import _binding_covers_input


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


class ContextualEnrichmentTest(unittest.TestCase):
    def _fixture(self, root: Path, *, role: str = "train") -> tuple[Path, Path, Path]:
        base = root / "base"
        requirement = b'{"context_key":"x"}\n'
        requirement_path = base / "requirement-shards/shard-000-of-256.jsonl"
        requirement_path.parent.mkdir(parents=True)
        requirement_path.write_bytes(requirement)
        state = {
            "candidate_event_step_ids": [1, 2, 3],
            "distance_rows": [
                {"coalition_event_step_ids": [], "distance": 1.0},
                {"coalition_event_step_ids": [1], "distance": 0.8},
                {"coalition_event_step_ids": [1, 2, 3], "distance": 0.0},
            ],
            "maximum_labeled_cardinality": 1,
            "role": role,
            "state_id": "trajectory:decision:004",
            "trajectory_id": "trajectory",
        }
        state_payload = canonical_json_bytes(state) + b"\n"
        (base / "states.jsonl").write_bytes(state_payload)
        base_content = hashlib.sha256(
            state_payload + hashlib.sha256(requirement).hexdigest().encode("ascii")
        ).hexdigest()
        _write_json(
            base / "manifest.json",
            {
                "content_sha256": base_content,
                "evaluation_labels_included": False,
                "requirement_shards": [
                    {
                        "path": "requirement-shards/shard-000-of-256.jsonl",
                        "sha256": hashlib.sha256(requirement).hexdigest(),
                        "status": CONTEXTUAL_REQUIREMENT_STATUS,
                    }
                ],
                "state_count": 1,
                "states_jsonl": "states.jsonl",
                "states_sha256": hashlib.sha256(state_payload).hexdigest(),
                "status": CONTEXTUAL_INPUT_STATUS,
            },
        )
        schedule = root / "schedule"
        schedule_row = {
            "candidate_event_ids": [1, 2, 3],
            "coalitions": [
                {"event_ids": [], "source": "anchor_empty"},
                {"event_ids": [1], "source": "old_duplicate"},
                {"event_ids": [2, 3], "source": "on_policy"},
            ],
            "role": "train",
            "state_id": state["state_id"],
            "trajectory_id": state["trajectory_id"],
        }
        schedule_payload = canonical_json_bytes(schedule_row) + b"\n"
        shard = schedule / "schedule-shards/shard-000-of-256.jsonl"
        shard.parent.mkdir(parents=True)
        shard.write_bytes(schedule_payload)
        _write_json(
            schedule / "manifest.json",
            {
                "content_sha256": "s" * 64,
                "input_content_sha256": base_content,
                "state_count": 1,
                "status": "COMPLETED_SET_UTILITY_TRAIN_ON_POLICY_SCHEDULES",
            },
        )
        labels = root / "labels"
        _write_json(
            labels / "states/trajectory_decision_004.json",
            {
                "candidate_event_step_ids": [1, 2, 3],
                "distance_rows": [
                    {"coalition_event_step_ids": [], "distance": 1.0 + 1e-8},
                    {"coalition_event_step_ids": [1], "distance": 0.8},
                    {"coalition_event_step_ids": [2, 3], "distance": 0.2},
                    {"coalition_event_step_ids": [1, 2, 3], "distance": 0.0},
                ],
                "role": "train",
                "source_revision": "a" * 40,
                "state_id": state["state_id"],
                "status": "COMPLETED_VARIABLE_HISTORY_LABEL_STATE",
                "trajectory_id": state["trajectory_id"],
            },
        )
        return base, schedule, labels

    def test_merges_novel_rows_and_preserves_duplicate_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, schedule, labels = self._fixture(root)
            manifest = materialize_contextual_enriched_inputs(
                base_input_root=base,
                schedule_root=schedule,
                label_roots=(labels,),
                output_root=root / "output",
            )
            state = json.loads((root / "output/states.jsonl").read_text())
            table = {
                tuple(row["coalition_event_step_ids"]): row["distance"]
                for row in state["distance_rows"]
            }
            self.assertEqual(table[(2, 3)], 0.2)
            self.assertEqual(table[()], 1.0)
            self.assertEqual(manifest["enrichment"]["added_distance_row_count"], 1)
            self.assertEqual(manifest["enrichment"]["duplicate_distance_row_count"], 3)
            self.assertEqual(manifest["parent_content_sha256"], json.loads((base / "manifest.json").read_text())["content_sha256"])

    def test_rejects_evaluation_state_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, schedule, labels = self._fixture(root, role="tune")
            with self.assertRaisesRegex(ValueError, "evaluation firewall"):
                materialize_contextual_enriched_inputs(
                    base_input_root=base,
                    schedule_root=schedule,
                    label_roots=(labels,),
                    output_root=root / "output",
                )

    def test_selector_binding_accepts_direct_or_parent_input(self) -> None:
        manifest = {"content_sha256": "enriched", "parent_content_sha256": "base"}
        self.assertTrue(_binding_covers_input(manifest, "enriched"))
        self.assertTrue(_binding_covers_input(manifest, "base"))
        self.assertFalse(_binding_covers_input(manifest, "other"))


if __name__ == "__main__":
    unittest.main()
