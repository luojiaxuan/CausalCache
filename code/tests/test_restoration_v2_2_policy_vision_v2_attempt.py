from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = (
    ROOT
    / "data/results/"
    "restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair_attempt"
)
V2_OUTPUT = (
    ROOT
    / "data/results/"
    "restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair"
)


class RestorationV22PolicyVisionV2AttemptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.failure = json.loads((DIRECTORY / "failure.json").read_text())

    def test_failure_is_zero_feature_and_not_a_scientific_result(self) -> None:
        self.assertEqual(
            hashlib.sha256((DIRECTORY / "failure.json").read_bytes()).hexdigest(),
            "3fe5c7fd6ea082501627fcbe52d4ca2751dc00f6f9a2270a9fdf20987097e5b7",
        )
        self.assertEqual(
            self.failure["status"],
            "INVALID_POLICY_VISION_V2_ZERO_FEATURE_SIZE_DICT_INTERFACE",
        )
        counts = self.failure["formal_operation_bounds_before_failure"]
        for key in (
            "policy_model_load_count",
            "image_processor_batch_count",
            "policy_vision_feature_forward_count",
            "cosine_scalar_transfer_count",
            "canonical_state_score_count",
            "selection_count",
            "restoration_label_semantic_load_count",
            "result_file_count",
            "gate_training_example_count",
            "matched_nll_evaluation_count",
            "closed_loop_episode_count",
            "confirm_state_access_count",
            "sealed_test_state_access_count",
        ):
            self.assertEqual(counts[key], 0, key)
        self.assertFalse(V2_OUTPUT.exists())
        self.assertFalse(V2_OUTPUT.with_name(f".{V2_OUTPUT.name}.staging").exists())

    def test_root_cause_preserves_the_frozen_numeric_geometry(self) -> None:
        cause = self.failure["root_cause"]
        self.assertEqual(cause["size_python_type"], "transformers.image_utils.SizeDict")
        self.assertFalse(cause["is_collections_abc_mapping"])
        self.assertTrue(cause["dict_conversion_equals_expected"])
        self.assertEqual(cause["dict_conversion"], cause["expected_dict"])
        self.assertFalse(cause["scientific_preprocessing_value_drift_observed"])

    def test_attempt_is_durably_tombstoned(self) -> None:
        attempt = self.failure["formal_attempt"]
        self.assertEqual(attempt["attempt_ledger_mode"], "0600")
        self.assertEqual(
            attempt["attempt_ledger_sha256"],
            "492e7c7aea0539fc5bc65c0d0be41b5659563a140d8e411a3b347b4c18584502",
        )
        retry = self.failure["retry_boundary"]
        self.assertFalse(retry["same_protocol_retry_allowed"])
        self.assertTrue(retry["attempt_ledger_must_be_preserved"])
        self.assertTrue(retry["container_preserved_stopped"])


if __name__ == "__main__":
    unittest.main()
