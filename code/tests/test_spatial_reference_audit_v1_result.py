from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "data/results/spatial_reference_audit_v1"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError("committed spatial-audit result must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SpatialReferenceAuditV1ResultTest(unittest.TestCase):
    def test_compact_summary_preserves_frozen_decision_boundary(self) -> None:
        summary = _load(RESULT_ROOT / "summary.json")
        self.assertEqual(
            summary["status"],
            "VALID_SPATIAL_REFERENCE_AUDIT_V1_VALIDATION_REPAIR_V1",
        )
        self.assertEqual(
            summary["decision"], "EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY"
        )
        self.assertEqual(
            [
                (
                    item["profile_id"],
                    item["exact_generated_token_sequence_stable_count"],
                    item["exact_canonical_action_stable_count"],
                    item["state_count"],
                )
                for item in summary["profiles"]
            ],
            [
                ("bf16_auto", 7, 7, 13),
                ("bf16_eager_control", 13, 13, 13),
                ("fp32_eager_control", 4, 4, 4),
            ],
        )
        self.assertEqual(summary["profiles"][2]["decision_role"], "descriptive_only")
        self.assertFalse(summary["claim_boundary"]["strict_cuda_determinism_claimed"])
        self.assertEqual(
            summary["claim_boundary"]["parent_v2_1_outcome_unchanged"],
            "NO_GO_V2_1_FULL_45_SUBSTRATE",
        )
        self.assertEqual(
            summary["operation_counts"],
            {
                "confirm_state_accesses": 0,
                "gate_training_examples": 0,
                "generation_calls": 60,
                "restoration_coalition_constructions": 0,
                "teacher_forwards": 120,
            },
        )
        self.assertEqual(
            summary["shape_validation"],
            {
                "generation_shape_node_count": 60,
                "observed_effective_visual_tokens_per_image": [2516, 2560, 2584],
                "teacher_shape_node_count": 120,
                "total_shape_node_count": 180,
            },
        )

    def test_artifact_binds_git_files_and_immutable_hf_bytes(self) -> None:
        artifact = _load(RESULT_ROOT / "artifact.json")
        self.assertEqual(
            artifact["status"], "VERIFIED_SPATIAL_REFERENCE_AUDIT_V1_ARTIFACT"
        )
        self.assertEqual(
            artifact["formal_failure_and_repair"]["raw_audit_git_commit"],
            "c093bd8f92ab97427acb427bd2d66fb6b20b556a",
        )
        self.assertEqual(
            artifact["formal_failure_and_repair"]["repair_git_commit"],
            "a2528d7e95e73c25639568650f63abce58e4e491",
        )
        self.assertEqual(
            artifact["git_result"]["summary_sha256"],
            _sha256(RESULT_ROOT / "summary.json"),
        )
        self.assertEqual(
            artifact["git_result"]["readme_sha256"],
            _sha256(RESULT_ROOT / "README.md"),
        )
        self.assertEqual(
            artifact["hf_artifact"]["immutable_revision"],
            "d6b2312e458ce3b2b1dc8463a323a8d7dbc945c1",
        )
        self.assertEqual(
            artifact["hf_artifact"]["exact_file_allowlist"],
            ["README.md", "manifest.json", "raw/spatial-reference-audit-v1.tar"],
        )
        self.assertTrue(artifact["hf_artifact"]["tag_resolution_verified"])
        self.assertEqual(artifact["raw_archive"]["file_count"], 72)
        self.assertEqual(
            artifact["raw_archive"]["sha256"],
            "d62ad05f6fdef06a3551f2ebe9f83f28327068f0020ff3e61891da4f46ce5ecc",
        )
        self.assertEqual(artifact["raw_archive"]["size_bytes"], 1_873_920)
        self.assertEqual(
            artifact["raw_archive"]["tree_inventory_sha256"],
            "6423c13f4b7067f5a2139442bab0be11a1ef5da9f880f51a1131768afa963b82",
        )
        self.assertEqual(
            artifact["raw_archive"]["summary_sha256"],
            "1d6dc90c602a3cb97ff58da8fafe03eefe31b19d1673084a132cc328c9229bf4",
        )
        self.assertEqual(
            artifact["raw_archive"]["summary_scientific_payload_sha256"],
            "2d73e395d5ee91349adf904b344b7ae25569e79ce8bfa35be9df7902ab8dc79e",
        )
        self.assertEqual(
            artifact["hf_artifact"]["manifest_sha256"],
            "37f89b40e5a75b6c3073d32b2d3b4aeff90c07b3ada20075a21e428d99cbcf1a",
        )
        self.assertEqual(
            artifact["hf_artifact"]["readme_sha256"],
            "c8d86701f6eb28c43e181b386a77fee7cfe69a0070228adac255be82690a4c92",
        )
        self.assertTrue(
            artifact["raw_archive"][
                "fresh_immutable_download_rebuilt_canonical_bytes"
            ]
        )


if __name__ == "__main__":
    unittest.main()
