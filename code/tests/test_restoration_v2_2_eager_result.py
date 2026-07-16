from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "data/results/restoration_v2_2_eager_full_45_substrate"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError("committed v2.2-eager result must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RestorationV22EagerResultTest(unittest.TestCase):
    def test_compact_summary_preserves_frozen_pass_boundary(self) -> None:
        summary = _load(RESULT_ROOT / "summary.json")
        self.assertEqual(
            summary["status"], "COMPLETED_FIXED_45_STATE_V2_2_EAGER_SUBSTRATE"
        )
        self.assertEqual(summary["outcome"], "PASS_V2_2_EAGER_FULL_45_SUBSTRATE")
        self.assertTrue(summary["gate_passed"])
        self.assertTrue(all(summary["checks"].values()))
        self.assertFalse(summary["restoration_work_performed"])
        self.assertFalse(summary["confirm_role_used"])
        metrics = summary["metrics"]
        self.assertEqual(metrics["fixed_state_denominator"], 45)
        self.assertEqual(metrics["parse_success_count"], 45)
        self.assertEqual(metrics["repeat_canonical_action_agreement_count"], 45)
        self.assertEqual(metrics["finite_logit_state_count"], 45)
        self.assertEqual(metrics["memory_sensitive_state_count"], 45)
        self.assertEqual(metrics["mean_repeat_kl"], 0.0)
        self.assertEqual(metrics["generation_call_count"], 90)
        self.assertEqual(metrics["teacher_forward_count"], 135)
        self.assertEqual(metrics["kl_measurement_count"], 90)
        self.assertEqual(metrics["retry_count"], 0)
        self.assertEqual(metrics["top_up_count"], 0)
        self.assertEqual(metrics["restoration_label_count"], 0)
        self.assertEqual(metrics["gate_training_example_count"], 0)
        self.assertEqual(metrics["confirm_state_access_count"], 0)
        worker_state_indices = [
            index
            for worker in summary["worker_topology"]
            for index in worker["state_indices"]
        ]
        self.assertEqual(
            [len(worker["state_indices"]) for worker in summary["worker_topology"]],
            [23, 22],
        )
        self.assertEqual(sorted(worker_state_indices), list(range(45)))
        self.assertEqual(summary["merge_order"], list(range(45)))

    def test_artifact_binds_summary_source_and_immutable_hf_bytes(self) -> None:
        summary = _load(RESULT_ROOT / "summary.json")
        artifact = _load(RESULT_ROOT / "artifact.json")
        self.assertEqual(
            artifact["status"], "VERIFIED_RESTORATION_V2_2_EAGER_ARTIFACT"
        )
        self.assertEqual(
            artifact["source_execution"]["source_git_commit"],
            "8ae07519f14ac3635f292ee93a7b6d624507427e",
        )
        self.assertEqual(
            artifact["hf_artifact"]["immutable_revision"],
            "3577099d505b8c652d764f41269df911128ec767",
        )
        self.assertTrue(
            artifact["hf_artifact"]["fresh_immutable_download_verified"]
        )
        self.assertEqual(artifact["raw_archive"]["file_count"], 102)
        self.assertEqual(artifact["raw_archive"]["size_bytes"], 1_269_760)
        self.assertEqual(
            artifact["raw_archive"]["sha256"],
            "b22827e6e2d8d33b03686fc177dc8f9c55c5133470fbe40f9fb3e33cce809fb5",
        )
        self.assertEqual(
            artifact["raw_archive"]["tree_inventory_sha256"],
            "e7f0047b20c27bac1f4d61d7c0bff2e63c7760469cf9ccaa188dca948b7a4359",
        )
        self.assertEqual(
            artifact["hf_artifact"]["fresh_archive_sha256"],
            artifact["raw_archive"]["sha256"],
        )
        self.assertEqual(
            artifact["result"],
            {
                "attempted_state_count": 45,
                "completed_state_count": 45,
                "outcome": "PASS_V2_2_EAGER_FULL_45_SUBSTRATE",
            },
        )
        self.assertEqual(artifact["protocol_id"], summary["protocol_id"])
        self.assertEqual(artifact["result"]["outcome"], summary["outcome"])
        self.assertEqual(
            artifact["source_execution"]["run_contract_sha256"],
            summary["run_contract_sha256"],
        )
        self.assertEqual(
            _sha256(RESULT_ROOT / "summary.json"),
            "38c53dbaa587450ef09ddbd54c483c5f6120e8ce003cfca6fd8298280cad809e",
        )
        self.assertEqual(
            _sha256(RESULT_ROOT / "artifact.json"),
            "77cfebc01909b1a018f7fe4ab300b843695ca9726fb96d3035587803bb4c2711",
        )


if __name__ == "__main__":
    unittest.main()
