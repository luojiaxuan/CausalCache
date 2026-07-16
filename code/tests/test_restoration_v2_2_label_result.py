from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPOSITORY_ROOT / "data/results/restoration_v2_2_eager_labels_v2"


class RestorationV22LabelResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((RESULT_ROOT / "summary.json").read_bytes())
        cls.artifact = json.loads((RESULT_ROOT / "artifact.json").read_bytes())

    def test_committed_artifact_is_bound_to_the_sealed_hf_revision(self) -> None:
        artifact = self.artifact
        self.assertEqual(
            artifact["status"], "VERIFIED_RESTORATION_V2_2_EAGER_LABEL_ARTIFACT"
        )
        self.assertEqual(
            artifact["result"]["outcome"],
            "PASS_RESTORATION_V2_2_EAGER_LABELS_V2",
        )
        self.assertEqual(
            artifact["source_execution"]["source_git_commit"],
            "5ae40d4aed4eb20b931216776b379bc6ae55629d",
        )
        self.assertEqual(
            artifact["source_execution"]["run_contract_sha256"],
            "b78ca1e70652c7eef68efc3472adf78cec4510e507a0c00cfcee2c2cf05285d9",
        )
        self.assertEqual(artifact["raw_archive"]["file_count"], 101)
        self.assertEqual(
            artifact["raw_archive"]["sha256"],
            "99120d5444d31962d9f4254c3e40bc5f06d3e4d3d90a74b1749e7ccd7aefb29e",
        )
        self.assertEqual(
            artifact["raw_archive"]["tree_inventory_sha256"],
            "c5104594b0741810f3d49d007a63a74f16ee4236dd137d1dea92b9373065c45f",
        )

        hf = artifact["hf_artifact"]
        self.assertEqual(
            hf["repo"], "gavinlaw/causalcache-restoration-labels-mobile"
        )
        self.assertEqual(
            hf["immutable_revision"],
            "8f6baae5c0b23b08915fa1b0fb848dd519b4c8db",
        )
        self.assertEqual(hf["tag"], "v2.2-eager-train-dev-exact-v2")
        self.assertEqual(hf["tag_resolved_revision"], hf["immutable_revision"])
        self.assertIs(hf["private"], True)
        self.assertIs(hf["tag_resolution_verified"], True)
        self.assertIs(hf["fresh_immutable_download_verified"], True)
        self.assertEqual(
            [record["path"] for record in hf["exact_file_records"]],
            [
                ".gitattributes",
                "README.md",
                "manifest.json",
                "raw/v2.2-eager-train-dev-exact-v2.tar",
            ],
        )

    def test_committed_summary_has_the_fixed_label_denominators(self) -> None:
        summary = self.summary
        artifact = self.artifact
        self.assertEqual(
            summary["status"], "SUMMARIZED_RESTORATION_V2_2_EAGER_LABEL_SCIENCE"
        )
        self.assertEqual(summary["outcome"], artifact["result"]["outcome"])
        self.assertEqual(
            summary["source_evidence"]["source_git_commit"],
            artifact["source_execution"]["source_git_commit"],
        )
        self.assertEqual(
            summary["source_evidence"]["tree_inventory_sha256"],
            artifact["raw_archive"]["tree_inventory_sha256"],
        )

        overall = summary["overall"]
        self.assertEqual(overall["state_count"], 45)
        self.assertEqual(overall["candidate_event_count_histogram"], {"2": 15, "3": 15, "4": 15})
        self.assertEqual(overall["raw_restoration_utility"]["row_count"], 420)
        self.assertEqual(
            overall["deployment_conditional_marginal"]["edge_count"], 435
        )
        self.assertEqual(overall["pair_interaction"]["row_count"], 465)
        self.assertEqual(
            overall["primary_exact_subset_oracle"]["state_count"], 45
        )

    def test_committed_scientific_reduction_keeps_nonmonotone_rows(self) -> None:
        overall = self.summary["overall"]
        self.assertEqual(
            {
                key: overall["raw_restoration_utility"][key]
                for key in ("positive_count", "zero_count", "negative_count")
            },
            {"positive_count": 338, "zero_count": 45, "negative_count": 37},
        )
        self.assertEqual(
            {
                key: overall["deployment_conditional_marginal"][key]
                for key in ("positive_count", "zero_count", "negative_count")
            },
            {"positive_count": 360, "zero_count": 0, "negative_count": 75},
        )
        self.assertEqual(
            overall["deployment_conditional_marginal"][
                "states_with_negative_marginal_count"
            ],
            22,
        )
        self.assertEqual(
            {
                key: overall["pair_interaction"][key]
                for key in ("positive_count", "zero_count", "negative_count")
            },
            {"positive_count": 188, "zero_count": 0, "negative_count": 277},
        )

    def test_committed_exact_oracle_recovery_and_split_gap(self) -> None:
        overall = self.summary["overall"]["primary_exact_subset_oracle"]
        self.assertEqual(
            overall["selected_cardinality_histogram"], {"0": 1, "1": 3, "2": 41}
        )
        self.assertEqual(overall["positive_utility_state_count"], 44)
        self.assertEqual(overall["zero_utility_state_count"], 1)
        self.assertEqual(overall["normalized_recovery_eligible_state_count"], 45)
        self.assertAlmostEqual(
            overall["mean_normalized_recovery"], 0.8735119444361525, places=15
        )
        self.assertAlmostEqual(
            self.summary["by_role"]["v2_label_train"][
                "primary_exact_subset_oracle"
            ]["mean_normalized_recovery"],
            0.9028414852633441,
            places=15,
        )
        self.assertAlmostEqual(
            self.summary["by_role"]["v2_development"][
                "primary_exact_subset_oracle"
            ]["mean_normalized_recovery"],
            0.8148528627817694,
            places=15,
        )


if __name__ == "__main__":
    unittest.main()
