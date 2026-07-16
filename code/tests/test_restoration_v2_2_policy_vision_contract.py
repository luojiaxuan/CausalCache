from __future__ import annotations

import copy
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_policy_vision_contract import (
    CANONICAL_CONFIG_PATH,
    EXPECTED_OPERATION_CEILING,
    EXPECTED_OUTPUT_FILES,
    EXPECTED_PRIMARY_STATE_INDICES,
    FROZEN_CONFIG_SHA256,
    PASS_STATUS,
    PROTOCOL_ID,
    RestorationV22PolicyVisionContract,
    _validate_semantics,
    sha256_file,
    strict_json_object_bytes,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


class RestorationV22PolicyVisionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = RestorationV22PolicyVisionContract.load(
            CONFIG,
            repository_root=ROOT,
        )

    def test_real_contract_is_frozen_bound_and_output_aware(self) -> None:
        self.assertEqual(sha256_file(CONFIG), FROZEN_CONFIG_SHA256)
        output_dir = ROOT / self.contract.data["output_contract"][
            "canonical_result_directory"
        ]
        output_exists = output_dir.exists()
        result = validate_contract(
            CONFIG,
            repository_root=ROOT,
            require_output_absent=not output_exists,
        )
        self.assertEqual(result["status"], PASS_STATUS)
        self.assertEqual(result["protocol_id"], PROTOCOL_ID)
        self.assertEqual(result["primary_state_count"], 15)
        self.assertEqual(result["unique_image_count"], 75)
        self.assertEqual(result["policy_vision_feature_forward_count"], 31)
        self.assertEqual(result["output_absent"], not output_exists)
        self.assertTrue(result["staging_output_absent"])

    def test_scope_uses_actual_primary_indices_and_fixed_worker_partition(self) -> None:
        scope = self.contract.data["scope"]
        execution = self.contract.data["execution_contract"]
        self.assertEqual(
            tuple(scope["primary_state_indices"]),
            EXPECTED_PRIMARY_STATE_INDICES,
        )
        workers = execution["runtime"]["workers"]
        self.assertEqual(workers[0]["state_indices"], list(range(2, 45, 6)))
        self.assertEqual(workers[1]["state_indices"], list(range(5, 42, 6)))
        assigned = sorted(workers[0]["state_indices"] + workers[1]["state_indices"])
        self.assertEqual(assigned, list(EXPECTED_PRIMARY_STATE_INDICES))
        sentinel = execution["verification"]["cross_device_sentinel"]
        self.assertEqual(sentinel["primary_ordinal"], 0)
        self.assertEqual(sentinel["state_index"], 2)
        self.assertEqual(
            sentinel["state_id"],
            "0131649930078879:decision_step:006",
        )

    def test_canonical_and_verification_operation_arithmetic(self) -> None:
        counts = self.contract.data["operation_ceiling"]
        self.assertEqual(counts, EXPECTED_OPERATION_CEILING)
        self.assertEqual(
            counts["policy_vision_feature_forward_count"],
            counts["canonical_vision_feature_forward_count"]
            + counts["same_device_replay_vision_feature_forward_count"]
            + counts["cross_device_sentinel_vision_feature_forward_count"],
        )
        self.assertEqual(
            counts["image_processor_batch_count"],
            counts["canonical_image_processor_batch_count"]
            + counts["verification_image_processor_batch_count"],
        )
        self.assertEqual(
            counts["image_processing_assignment_count"],
            counts["canonical_image_processing_assignment_count"]
            + counts["verification_image_processing_assignment_count"],
        )
        self.assertEqual(
            counts["cosine_scalar_transfer_count"],
            counts["canonical_cosine_scalar_transfer_count"]
            + counts["verification_cosine_scalar_transfer_count"],
        )

    def test_direct_image_processor_and_feature_worker_blinding(self) -> None:
        preprocessing = self.contract.data["execution_contract"]["preprocessing"]
        self.assertEqual(
            preprocessing["loader"],
            "transformers.AutoImageProcessor.from_pretrained",
        )
        self.assertFalse(preprocessing["auto_processor_or_tokenizer_allowed"])
        self.assertEqual(
            preprocessing["output_keys_exact"],
            ["pixel_values", "image_grid_thw"],
        )
        self.assertEqual(
            set(preprocessing["forbidden_inputs_to_feature_worker"]),
            {
                "goal",
                "text",
                "ocr",
                "restoration_labels",
                "geometry_utilities",
                "selected_coalitions",
            },
        )
        self.assertEqual(preprocessing["canonical_raw_patch_rows"], 767020)
        self.assertEqual(preprocessing["canonical_merged_visual_tokens"], 191755)

    def test_feature_path_excludes_language_and_deepstack_outputs(self) -> None:
        feature = self.contract.data["feature_contract"]
        self.assertEqual(feature["feature_field"], "pooler_output")
        self.assertEqual(feature["feature_dtype"], "torch.bfloat16")
        self.assertEqual(feature["reduction"], "per_image_FP32_mean_then_L2_normalize")
        self.assertTrue(feature["pre_merger_last_hidden_state_excluded"])
        self.assertTrue(feature["all_deepstack_features_excluded"])
        self.assertEqual(feature["deepstack_visual_indexes"], [8, 16, 24])
        self.assertFalse(feature["language_model_called"])
        self.assertFalse(feature["lm_head_called"])
        self.assertFalse(feature["generate_called"])
        for key in (
            "policy_forward_count",
            "language_model_forward_count",
            "lm_head_forward_count",
            "generation_count",
            "teacher_forward_count",
            "kl_measurement_count",
            "gate_training_example_count",
            "gate_model_forward_count",
            "matched_nll_evaluation_count",
            "closed_loop_episode_count",
            "confirm_state_access_count",
            "sealed_test_state_access_count",
        ):
            self.assertEqual(self.contract.data["operation_ceiling"][key], 0)

    def test_identity_witness_and_model_snapshot_are_exactly_bound(self) -> None:
        inputs = self.contract.data["immutable_inputs"]
        witness = inputs["ocr_rgb_identity_witness"]
        self.assertEqual(
            tuple(record["path"] for record in witness["files"]),
            EXPECTED_OUTPUT_FILES,
        )
        self.assertEqual(
            witness["scientific_payload_sha256"],
            "5942519bdff8f3e8a64bbc8b32a5d42a64abff37daf0804fcce0f098a63765b2",
        )
        snapshot = inputs["model_snapshot"]
        self.assertEqual(snapshot["file_count"], 14)
        self.assertEqual(snapshot["total_bytes"], 17545907171)
        self.assertEqual(
            snapshot["revision"],
            "06d5faecff74840bab2be2425e9c42667a5d04fc",
        )

    def test_semantic_drift_fails_closed_before_runtime(self) -> None:
        mutations = []
        feature = copy.deepcopy(self.contract.data)
        feature["feature_contract"]["feature_field"] = "last_hidden_state"
        mutations.append(feature)
        worker = copy.deepcopy(self.contract.data)
        worker["execution_contract"]["runtime"]["workers"][0]["state_indices"][0] = 0
        mutations.append(worker)
        sentinel = copy.deepcopy(self.contract.data)
        sentinel["execution_contract"]["verification"]["cross_device_sentinel"][
            "state_index"
        ] = 0
        mutations.append(sentinel)
        labels = copy.deepcopy(self.contract.data)
        labels["authorization"]["restoration_labels_to_feature_worker_allowed"] = True
        mutations.append(labels)
        count = copy.deepcopy(self.contract.data)
        count["operation_ceiling"]["language_model_forward_count"] = 1
        mutations.append(count)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                _validate_semantics(mutation)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            strict_json_object_bytes(
                b'{"protocol_id":"a","protocol_id":"b"}',
                label="synthetic contract",
            )


if __name__ == "__main__":
    unittest.main()
