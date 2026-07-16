from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_policy_vision_contract import (
    sha256_bytes,
    strict_json_object_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
ATTEMPT = (
    ROOT
    / "data/results/restoration_v2_2_policy_vision_baseline_v1_attempt/failure.json"
)
SOURCE_COMMIT = "c0937056e94d110cd67e593288f9e0c3a3b24809"
FAILURE_SHA256 = "1cea24506b14b9dcd4bdfff77c82024417f65e1416a418caeec0d1192c2b0311"


class RestorationV22PolicyVisionV1AttemptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = ATTEMPT.read_bytes()
        cls.record = strict_json_object_bytes(
            cls.payload,
            label="policy-vision v1 failure evidence",
        )

    def test_failure_identity_and_zero_feature_boundary_are_frozen(self) -> None:
        self.assertEqual(sha256_bytes(self.payload), FAILURE_SHA256)
        self.assertEqual(
            self.record["status"],
            "INVALID_POLICY_VISION_V1_ZERO_FEATURE_GPU_UUID_TYPE",
        )
        self.assertEqual(self.record["source_git_commit"], SOURCE_COMMIT)
        self.assertEqual(
            self.record["root_cause"]["observed_python_type"],
            "torch._C._CUuuid",
        )
        counts = self.record["operation_counts_before_failure"]
        for key in (
            "model_snapshot_full_hash_count",
            "image_processor_load_count",
            "policy_model_load_count",
            "image_processor_batch_count",
            "vision_feature_forward_count",
            "cosine_scalar_transfer_count",
            "selection_count",
            "restoration_label_semantic_load_count",
            "gate_training_example_count",
            "matched_nll_evaluation_count",
            "closed_loop_episode_count",
            "confirm_state_access_count",
            "sealed_test_state_access_count",
            "result_file_count",
        ):
            self.assertEqual(counts[key], 0, key)

    def test_v1_output_identity_remains_unused(self) -> None:
        output = ROOT / "data/results/restoration_v2_2_policy_vision_baseline_v1"
        staging = output.with_name(f".{output.name}.staging")
        self.assertFalse(output.exists())
        self.assertFalse(staging.exists())
        self.assertFalse(
            self.record["retry_boundary"]["same_protocol_retry_allowed"]
        )

    def test_failure_source_commit_contains_the_rejected_type_boundary(self) -> None:
        source = subprocess.run(
            [
                "git",
                "show",
                f"{SOURCE_COMMIT}:code/causalcache/policy/"
                "gui_owl_v2_2_vision_runtime.py",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        identity_probe = source.index("gpu_identity = _validated_gpu_identity")
        snapshot_hash = source.index("identity = verify_frozen_vision_runtime")
        processor_load = source.index("AutoImageProcessor.from_pretrained")
        model_load = source.index("AutoModelForImageTextToText.from_pretrained")
        self.assertLess(identity_probe, snapshot_hash)
        self.assertLess(identity_probe, processor_load)
        self.assertLess(identity_probe, model_load)
        self.assertIn(
            'raise ValueError("GPU UUID must be a non-empty string")',
            source,
        )


if __name__ == "__main__":
    unittest.main()
