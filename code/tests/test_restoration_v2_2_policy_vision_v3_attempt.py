from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_policy_vision import (
    load_identity_witness,
    validate_feature_worker_provenance,
)
from causalcache.restoration_v2_2_policy_vision_v3_contract import (
    CANONICAL_CONFIG_PATH,
    RestorationV22PolicyVisionV3Contract,
)
from scripts.run_restoration_v2_2_policy_vision_baseline import (
    _feature_record_from_evaluated_row,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT = (
    ROOT
    / "data/results/"
    "restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair"
)
ATTEMPT = (
    ROOT
    / "data/results/"
    "restoration_v2_2_policy_vision_baseline_v3_cpu_validation_attempt"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RestorationV22PolicyVisionV3AttemptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((RESULT / "summary.json").read_text())
        cls.failure = json.loads((ATTEMPT / "failure.json").read_text())
        cls.rows = tuple(
            json.loads(line)
            for line in (RESULT / "state_scores.jsonl").read_text().splitlines()
        )

    def test_formal_artifact_exact_three_bytes_are_locked(self) -> None:
        expected = {
            "README.md": "70ce9496983bfafeab04dd5184e5b7d5c8f02a70df9812893af0fe0aed47ed1f",
            "state_scores.jsonl": "8c5977260bb1a8618b43cd877d963bd637c197e589db532c7ccbcab9c55f58b9",
            "summary.json": "5ed21d6c397839bc60987bb3ee1db3a0560b6e923557cc1e8e524de921af1c0e",
        }
        self.assertEqual(sorted(path.name for path in RESULT.iterdir()), sorted(expected))
        self.assertEqual(
            {name: _sha256(RESULT / name) for name in expected},
            expected,
        )
        self.assertFalse(
            RESULT.with_name(f".{RESULT.name}.staging").exists()
        )

    def test_gpu_schedule_completed_without_forbidden_operations(self) -> None:
        self.assertEqual(
            self.summary["status"],
            "COMPLETED_RESTORATION_V2_2_POLICY_VISION_BASELINE_"
            "V3_SIZE_DICT_INTERFACE_REPAIR",
        )
        counts = self.summary["aggregate"]["operation_counts"]
        self.assertEqual(counts["policy_vision_feature_forward_count"], 31)
        self.assertEqual(counts["canonical_state_score_count"], 15)
        for key in (
            "policy_forward_count",
            "language_model_forward_count",
            "lm_head_forward_count",
            "generation_count",
            "gate_model_forward_count",
            "gate_training_example_count",
            "matched_nll_evaluation_count",
            "closed_loop_episode_count",
            "confirm_state_access_count",
            "sealed_test_state_access_count",
        ):
            self.assertEqual(counts[key], 0, key)
        verification = self.summary["execution"]["verification"]
        self.assertEqual(
            verification["same_device"]["maximum_absolute_score_difference"],
            0.0,
        )
        self.assertEqual(
            verification["cross_device_sentinel"][
                "maximum_absolute_score_difference"
            ],
            0.0,
        )

    def test_recorded_rows_contain_the_complete_projection_repair_inputs(self) -> None:
        self.assertEqual(len(self.rows), 15)
        expected_state_keys = {
            "budget_event_capacity",
            "candidate_event_step_ids",
            "decision_step_id",
            "index",
            "role",
            "state_id",
            "trajectory_id",
        }
        for row in self.rows:
            self.assertEqual(set(row["state"]), expected_state_keys)
            expected_worker = "even" if row["state"]["index"] % 2 == 0 else "odd"
            expected_device = "cuda:0" if expected_worker == "even" else "cuda:1"
            self.assertEqual(row["worker"]["worker_id"], expected_worker)
            self.assertEqual(row["worker"]["device"], expected_device)
            self.assertTrue(row["worker"]["gpu_uuid"].startswith("GPU-"))

    def test_legacy_failure_and_four_key_projection_are_reproduced(self) -> None:
        contract = RestorationV22PolicyVisionV3Contract.load(
            ROOT / CANONICAL_CONFIG_PATH,
            repository_root=ROOT,
            validate_bound_sources=False,
        )
        work_items, _ = load_identity_witness(contract)
        feature_records = tuple(
            _feature_record_from_evaluated_row(row) for row in self.rows
        )
        with self.assertRaisesRegex(
            ValueError,
            "row-to-worker provenance drifted",
        ):
            validate_feature_worker_provenance(feature_records, work_items)

        state_keys = ("index", "role", "trajectory_id", "state_id")
        projected = []
        for feature_record in feature_records:
            record = dict(feature_record)
            record["state"] = {
                key: feature_record["state"][key] for key in state_keys
            }
            projected.append(record)
        self.assertIsNone(
            validate_feature_worker_provenance(projected, work_items)
        )

    def test_cpu_validation_failure_is_scoped_and_gpu_retry_is_forbidden(self) -> None:
        self.assertEqual(
            _sha256(ATTEMPT / "failure.json"),
            "8b1c517838256665d378d100df1b0023b6360a71fe30f08296c54805f3443c29",
        )
        self.assertEqual(
            self.failure["status"],
            "INVALID_POLICY_VISION_V3_CPU_VALIDATE_STATE_PROJECTION",
        )
        self.assertEqual(
            self.failure["artifact_status"],
            "COMPLETED_PENDING_VERSIONED_CPU_REPLAY_VALIDATION",
        )
        self.assertEqual(
            self.failure["validation_failure"]["gpu_model_or_feature_operation_count"],
            0,
        )
        self.assertTrue(
            all(
                self.failure["bounded_cpu_diagnostic"][
                    "rebuilt_exact_file_matches"
                ].values()
            )
        )
        self.assertFalse(
            self.failure["retry_boundary"]["same_gpu_protocol_retry_allowed"]
        )
        self.assertTrue(
            self.failure["retry_boundary"]["formal_artifact_bytes_must_be_preserved"]
        )

    def test_bounded_diagnostic_source_is_exactly_bound(self) -> None:
        diagnostic = (
            ROOT
            / "code/scripts/"
            "diagnose_restoration_v2_2_policy_vision_v3_cpu_replay.py"
        )
        self.assertEqual(
            _sha256(diagnostic),
            "84e12ce1ccff68667e7008d9eb76f080230369bd75c0f4b3b04249cdf2f778e4",
        )
        self.assertEqual(diagnostic.stat().st_size, 1866)


if __name__ == "__main__":
    unittest.main()
