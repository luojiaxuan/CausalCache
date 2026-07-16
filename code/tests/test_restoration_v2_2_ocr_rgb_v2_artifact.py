from __future__ import annotations

import hashlib
import json
import math
import unittest
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "data/results/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair"
V1_RESULT = ROOT / "data/results/restoration_v2_2_ocr_rgb_baseline_v1"
V1_FAILURE = ROOT / "data/results/restoration_v2_2_ocr_rgb_baseline_v1_attempt"
PROTOCOL_ID = "causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair"
EXPECTED_FILES = {
    "README.md": (
        1603,
        "526347e1928510a3b5f3791631fd35d08e6c53420b9994fd37a4b89fc592b4bf",
    ),
    "state_scores.jsonl": (
        57465,
        "07e29538232620bbea3bb12d1c0ce2649ee505285f53daf006dea3f670ba2d0a",
    ),
    "summary.json": (
        38130,
        "3ae9c27c93c3813f3458d31e1740ff9a97d0c9af6beab9f926f627d8611c7b29",
    ),
}
EXPECTED_OPERATIONS = {
    "candidate_similarity_comparison_count": 60,
    "closed_loop_episode_count": 0,
    "combined_similarity_score_count": 60,
    "confirm_feature_score_count": 0,
    "confirm_state_access_count": 0,
    "derived_nonselected_semantic_parse_count": 0,
    "gate_model_forward_count": 0,
    "gate_training_example_count": 0,
    "generation_count": 0,
    "geometry_nonprimary_semantic_parse_count": 0,
    "geometry_primary_record_count": 15,
    "gpu_count": 0,
    "image_decode_resize_count": 75,
    "image_payload_extract_count": 75,
    "kl_measurement_count": 0,
    "matched_nll_evaluation_count": 0,
    "ocr_inference_count": 0,
    "ocr_model_load_count": 0,
    "ocr_similarity_comparison_count": 60,
    "policy_forward_count": 0,
    "policy_model_load_count": 0,
    "policy_vision_feature_forward_count": 0,
    "primary_label_distance_row_count": 240,
    "primary_label_state_count": 15,
    "rgb_similarity_comparison_count": 60,
    "sealed_test_state_access_count": 0,
    "selected_coalition_distance_lookup_count": 15,
    "semantic_ocr_record_count": 75,
    "semantic_trajectory_record_count": 15,
    "teacher_forward_count": 0,
    "validated_label_distance_row_count": 420,
    "validated_label_state_count": 45,
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_json(raw: bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=reject_duplicates)


class RestorationV22OcrRgbV2ArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary_bytes = (RESULT / "summary.json").read_bytes()
        cls.summary = _strict_json(cls.summary_bytes)
        cls.record_bytes = (RESULT / "state_scores.jsonl").read_bytes()
        cls.records = tuple(
            _strict_json(line) for line in cls.record_bytes.splitlines()
        )

    def test_exact_three_inventory_hashes_and_sizes(self) -> None:
        self.assertTrue(RESULT.is_dir())
        self.assertFalse(RESULT.is_symlink())
        self.assertEqual(
            sorted(path.name for path in RESULT.iterdir()),
            sorted(EXPECTED_FILES),
        )
        for name, (size, expected_hash) in EXPECTED_FILES.items():
            path = RESULT / name
            self.assertTrue(path.is_file(), msg=name)
            self.assertFalse(path.is_symlink(), msg=name)
            payload = path.read_bytes()
            self.assertEqual(len(payload), size, msg=name)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_hash, msg=name)

    def test_jsonl_is_canonical_and_denominator_is_exact(self) -> None:
        self.assertTrue(self.record_bytes.endswith(b"\n"))
        rebuilt = b"".join(
            _canonical_json_bytes(record) + b"\n" for record in self.records
        )
        self.assertEqual(rebuilt, self.record_bytes)
        self.assertEqual(len(self.records), 15)
        self.assertEqual(
            self.summary["state_scores"],
            {
                "path": "state_scores.jsonl",
                "record_count": 15,
                "sha256": EXPECTED_FILES["state_scores.jsonl"][1],
                "size_bytes": EXPECTED_FILES["state_scores.jsonl"][0],
            },
        )
        self.assertEqual(
            Counter(record["state"]["role"] for record in self.records),
            {"v2_label_train": 10, "v2_development": 5},
        )
        self.assertEqual(
            [record["state"]["index"] for record in self.records],
            list(range(2, 45, 3)),
        )
        self.assertEqual(
            len({record["state"]["state_id"] for record in self.records}), 15
        )
        self.assertEqual(
            len({record["state"]["trajectory_id"] for record in self.records}),
            15,
        )

        image_identities: set[tuple[str, str]] = set()
        for record in self.records:
            self.assertEqual(record["schema_version"], "1.0.0")
            self.assertEqual(record["protocol_id"], PROTOCOL_ID)
            state = record["state"]
            self.assertEqual(state["candidate_event_step_ids"], [1, 2, 3, 4])
            self.assertEqual(state["budget_event_capacity"], 2)
            self.assertEqual(len(record["candidate_scores"]), 4)
            self.assertEqual(
                [row["event_step_id"] for row in record["candidate_scores"]],
                [1, 2, 3, 4],
            )
            self.assertEqual(sorted(record["ranked_event_step_ids"]), [1, 2, 3, 4])
            self.assertEqual(len(record["selected_coalition"]), 2)
            image_identities.add(
                (
                    record["current_image"]["image_member_path"],
                    record["current_image"]["image_sha256"],
                )
            )
            for candidate in record["candidate_scores"]:
                image_identities.add(
                    (
                        candidate["post_image_member_path"],
                        candidate["post_image_sha256"],
                    )
                )
        self.assertEqual(sum(len(row["candidate_scores"]) for row in self.records), 60)
        self.assertEqual(len(image_identities), 75)

    def test_source_input_and_scientific_payload_identity(self) -> None:
        self.assertEqual(self.summary["schema_version"], "1.1.0")
        self.assertEqual(
            self.summary["status"],
            "COMPLETED_RESTORATION_V2_2_OCR_RGB_BASELINE_V2_IDENTITY_REPAIR",
        )
        self.assertEqual(self.summary["protocol_id"], PROTOCOL_ID)
        source = self.summary["source_execution"]
        self.assertEqual(
            source["source_git_commit"],
            "a9bede85ab8bd10623c5755b944b3c26865c6485",
        )
        self.assertEqual(
            source["contract_sha256"],
            "d68cb032ef3c56c330d57329507d409b20f878b7510bd09d88e2eff1f3f898f3",
        )
        self.assertEqual(
            source["parent_contract_sha256"],
            "08f57505d71e603d81d6915ef27cf008d0fe097a56d70a05f8b3519211e2e6f9",
        )
        self.assertEqual(
            source["formal_python_source_closure"],
            {
                "inventory_sha256": (
                    "3d60b7b1ddb60c7a1c377c90ac171304273ecd97fe14109532f3297fff474541"
                ),
                "path_count": 140,
                "rule": (
                    "all_tracked_python_under_code_causalcache_and_code_scripts_at_source_commit"
                ),
            },
        )
        inputs = self.summary["input_identity"]
        self.assertEqual(
            inputs["restoration_labels"]["hf_revision"],
            "8f6baae5c0b23b08915fa1b0fb848dd519b4c8db",
        )
        self.assertEqual(
            inputs["derived_dataset"]["hf_revision"],
            "89f136abaff797e14fe758a198996e51032a10a6",
        )
        self.assertEqual(
            inputs["derived_dataset"]["artifact_tree_sha256"],
            "475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e",
        )
        self.assertEqual(
            inputs["selector_geometry"]["scientific_payload_sha256"],
            "cc505443a7efdc68c8eeca754f24c9143cabf72a090f024fde05011e783cbb21",
        )
        full_payload = {
            "protocol_id": PROTOCOL_ID,
            "source_git_commit": source["source_git_commit"],
            "contract_sha256": source["contract_sha256"],
            "parent_contract_sha256": source["parent_contract_sha256"],
            "repair_identity": self.summary["repair_identity"],
            "input_identity": inputs,
            "aggregate": self.summary["aggregate"],
            "state_scores": list(self.records),
        }
        self.assertEqual(
            hashlib.sha256(_canonical_json_bytes(full_payload)).hexdigest(),
            "5942519bdff8f3e8a64bbc8b32a5d42a64abff37daf0804fcce0f098a63765b2",
        )
        self.assertEqual(
            self.summary["scientific_payload_sha256"],
            "5942519bdff8f3e8a64bbc8b32a5d42a64abff37daf0804fcce0f098a63765b2",
        )

    def test_operation_ceiling_is_exact(self) -> None:
        observed = self.summary["aggregate"]["operation_counts"]
        self.assertEqual(observed, EXPECTED_OPERATIONS)
        self.assertTrue(
            all(type(value) is int and value >= 0 for value in observed.values())
        )

    def test_headline_metrics_and_bootstrap_contract_are_locked(self) -> None:
        by_role = self.summary["aggregate"]["by_role"]
        expected = {
            "v2_label_train": (10, 0.771188782430142, 0.3, 0.3),
            "v2_development": (5, 0.019571013102546025, 0.0, 0.0),
            "overall_stratified": (15, 0.52064952598761, 0.2, 0.2),
        }
        for role, values in expected.items():
            count, recovery, exact_match, exact_cardinality_match = values
            self.assertEqual(by_role[role]["state_count"], count)
            self.assertAlmostEqual(
                by_role[role]["mean_normalized_recovery"], recovery
            )
            self.assertAlmostEqual(
                by_role[role]["exact_coalition_match_rate"], exact_match
            )
            self.assertAlmostEqual(
                by_role[role]["exact_cardinality_coalition_match_rate"],
                exact_cardinality_match,
            )

        development_deltas = {
            "budget_conditioned_independent": -0.7467323524656541,
            "exact_subset": -0.9057709454098476,
            "dynamic_recent": -9.492037172558268e-05,
            "analytic_exact_cardinality_random": -0.119899874315825,
        }
        reports = self.summary["aggregate"]["paired_trajectory_bootstrap"]
        self.assertEqual(
            set(reports),
            {
                "analytic_exact_cardinality_random",
                "budget_conditioned_independent",
                "dynamic_recent",
                "exact_cardinality_oracle",
                "exact_subset",
                "full_shapley_independent",
                "true_conditional_greedy",
            },
        )
        for name, report in reports.items():
            self.assertEqual(report["resamples"], 10_000, msg=name)
            self.assertEqual(report["seed"], 271_828, msg=name)
            self.assertEqual(report["confidence"], 0.9, msg=name)
            self.assertEqual(report["development"]["trajectory_count"], 5, msg=name)
            self.assertEqual(
                len(self.summary["aggregate"]["development_paired_deltas"][name]),
                5,
                msg=name,
            )
        for name, expected_delta in development_deltas.items():
            self.assertAlmostEqual(
                reports[name]["development"]["mean_difference"],
                expected_delta,
                msg=name,
            )

    def test_negative_outlier_and_v1_failure_boundary_are_preserved(self) -> None:
        negative = [
            record for record in self.records if record["normalized_recovery"] < 0.0
        ]
        self.assertEqual(len(negative), 1)
        record = negative[0]
        self.assertEqual(record["state"]["trajectory_id"], "0141544666483837")
        self.assertEqual(record["state"]["index"], 41)
        self.assertEqual(record["selected_coalition"], [3, 4])
        self.assertEqual(record["exact_subset_coalition"], [1, 2])
        self.assertAlmostEqual(
            record["baseline_summary_only_distance"], 0.0006160458433441818
        )
        self.assertAlmostEqual(record["selected_distance"], 0.002243123482912779)
        self.assertAlmostEqual(record["actual_utility"], -0.001627077639568597)
        self.assertAlmostEqual(record["normalized_recovery"], -2.6411632464494317)
        self.assertAlmostEqual(
            record["exact_subset_normalized_recovery"], 0.9236395231597756
        )
        development = [
            row
            for row in self.records
            if row["state"]["role"] == "v2_development"
        ]
        self.assertEqual(len(development), 5)
        self.assertAlmostEqual(
            math.fsum(row["normalized_recovery"] for row in development) / 5,
            self.summary["aggregate"]["by_role"]["v2_development"][
                "mean_normalized_recovery"
            ],
        )

        self.assertFalse(V1_RESULT.exists())
        self.assertFalse(V1_RESULT.with_name(V1_RESULT.name + ".tmp").exists())
        self.assertFalse(RESULT.with_name(RESULT.name + ".tmp").exists())
        self.assertEqual(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in V1_FAILURE.iterdir()
            },
            {
                "README.md": (
                    "9b3b6d2eab20d997bd26f12845de9ba7849b7fa0f5905f501fdb3b9179b9e6be"
                ),
                "summary.json": (
                    "fa422e4c71f9d25af4359ac222cadfe249d952c4a626685f787595077b686b14"
                ),
            },
        )
        repair = self.summary["repair_identity"]
        self.assertEqual(
            repair["failed_attempt_git_commit"],
            "2870d8ae26542a184647e8b6d97b8c79e4e12641",
        )
        self.assertEqual(
            repair["parent_source_git_commit"],
            "aa5898f25e2e7d647363fe701ac90134bf744a5c",
        )
        self.assertEqual(repair["v1_formal_state_score_row_count"], 0)
        self.assertFalse(repair["v1_scientific_payload_generated"])
        self.assertFalse(repair["v1_scientific_comparison_available"])
        self.assertEqual(repair["trajectory_identity_occurrences_per_line"], 2)
        self.assertEqual(repair["ocr_identity_occurrences_per_line"], 1)
        for key in (
            "parent_scientific_contract_changed",
            "immutable_inputs_changed",
            "feature_contract_changed",
            "selection_contract_changed",
            "statistics_contract_changed",
            "operation_ceiling_changed",
        ):
            self.assertFalse(repair[key], msg=key)


if __name__ == "__main__":
    unittest.main()
