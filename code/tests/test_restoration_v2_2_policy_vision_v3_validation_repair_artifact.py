from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract import (
    RUN_STATUS,
    pretty_json_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT = (
    ROOT
    / "data/results/"
    "restoration_v2_2_policy_vision_baseline_v3_validation_repair_v1"
)
SOURCE_COMMIT = "dbb45637cf79c3573bbbc6051b8b480e3f76d69d"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class PolicyVisionV3ValidationRepairArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = (RESULT / "README.md").read_bytes()
        cls.summary_bytes = (RESULT / "summary.json").read_bytes()
        cls.summary = json.loads(cls.summary_bytes)

    def test_exact_two_inventory_sizes_hashes_and_canonical_json(self) -> None:
        expected = {
            "README.md": (
                836,
                "3492dbae9a13d3d1d70e7aacf2cd7b405d1f9cc5956af980c519c8fb3ceee7e9",
            ),
            "summary.json": (
                10461,
                "f7a5ff63a754d06d7b61dcc46516ee2ed22e0a6a3b92b8b0be8a68869cf362b1",
            ),
        }
        children = tuple(RESULT.iterdir())
        self.assertEqual(sorted(path.name for path in children), sorted(expected))
        self.assertTrue(
            all(path.is_file() and not path.is_symlink() for path in children)
        )
        for name, (size, digest) in expected.items():
            with self.subTest(name=name):
                payload = (RESULT / name).read_bytes()
                self.assertEqual(len(payload), size)
                self.assertEqual(_sha256(payload), digest)
        self.assertEqual(self.summary_bytes, pretty_json_bytes(self.summary))

    def test_valid_source_and_exact_three_replay_are_locked(self) -> None:
        self.assertEqual(self.summary["status"], RUN_STATUS)
        source = self.summary["source_execution"]
        self.assertEqual(source["validation_source_git_commit"], SOURCE_COMMIT)
        self.assertEqual(
            source["validation_python_source_closure"],
            {
                "rule": (
                    "all_tracked_python_under_code_causalcache_and_code_scripts_"
                    "at_source_commit"
                ),
                "path_count": 156,
                "inventory_sha256": (
                    "e8daf215ca51b4c4d9e5f8a82399ec70d7d68700351968cace3386e235160a1d"
                ),
            },
        )
        replay = self.summary["exact_byte_replay"]
        self.assertTrue(replay["all_files_byte_equal"])
        self.assertTrue(replay["producer_artifact_bytes_unchanged_before_after"])
        self.assertEqual(replay["feature_record_count"], 15)
        self.assertEqual(replay["candidate_score_count"], 60)
        self.assertEqual(
            [row["path"] for row in replay["producer_files"]],
            ["README.md", "state_scores.jsonl", "summary.json"],
        )
        for row in replay["producer_files"]:
            self.assertTrue(row["byte_equal"], row["path"])
            self.assertEqual(row["size_bytes"], row["expected_size_bytes"])
            self.assertEqual(row["sha256"], row["expected_sha256"])

    def test_cpu_no_nvidia_and_all_forbidden_operations_are_zero(self) -> None:
        counts = self.summary["operation_counts"]
        self.assertTrue(counts)
        self.assertTrue(all(value == 0 for value in counts.values()))
        runtime = self.summary["runtime"]
        claim = self.summary["formal_attempt_claim"]
        identity = claim["record"]["runtime_identity"]
        self.assertEqual(runtime["device"], "cpu")
        for key, value in identity.items():
            self.assertEqual(runtime[key], value, key)
        self.assertEqual(
            runtime["no_nvidia_runtime"],
            {
                "cuda_runtime_import_count": 0,
                "forbidden_runtime_modules_checked": [
                    "torch",
                    "transformers",
                    "PIL",
                ],
                "forbidden_runtime_modules_imported": [],
                "nvidia_device_node_count": 0,
                "nvidia_device_nodes": [],
                "nvidia_smi_invocation_count": 0,
            },
        )
        self.assertEqual(
            claim["sha256"],
            "6b65bef9d6edb73ee4275e89923bfd0e6123fb275fccb2b5dda9e57245f7b5d3",
        )

    def test_only_projection_change_is_recorded(self) -> None:
        repair = self.summary["repair"]
        self.assertEqual(
            repair,
            {
                "all_non_state_reconstruction_fields_unchanged": True,
                "evaluated_state_keys_exact": [
                    "budget_event_capacity",
                    "candidate_event_step_ids",
                    "decision_step_id",
                    "index",
                    "role",
                    "state_id",
                    "trajectory_id",
                ],
                "feature_state_keys_exact_in_output_order": [
                    "index",
                    "role",
                    "trajectory_id",
                    "state_id",
                ],
                "only_semantic_change": (
                    "project_exact_seven_key_evaluated_state_to_exact_four_key_"
                    "feature_state"
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
