from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from causalcache.gate_v1_fresh16_evaluation_contract import (
    CANONICAL_CONFIG_PATH,
    FRESH_SOURCE_IDS_SHA256,
    FROZEN_CONFIG_SHA256,
    PAYLOAD_TARGETS,
    REPORT_TARGETS,
    RUNNER_FREEZE_B_PATH,
    validate_fresh16_evaluation_contract_data,
    validate_fresh16_evaluation_source_only_contract,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


class Fresh16EvaluationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_frozen_data_contract_closes_geometry_models_and_outputs(self) -> None:
        validated = validate_fresh16_evaluation_contract_data(self.config)
        self.assertIs(validated, self.config)
        self.assertEqual(
            validated["fresh_geometry"]["source_ids_sha256"],
            FRESH_SOURCE_IDS_SHA256,
        )
        self.assertEqual(
            tuple(validated["output_contract"]["payload_commit"]["exact_targets"]),
            PAYLOAD_TARGETS,
        )
        self.assertEqual(
            tuple(validated["output_contract"]["report_commit"]["exact_new_targets"]),
            REPORT_TARGETS,
        )
        self.assertEqual(len(validated["formal_model_input"]["checkpoints"]), 10)
        self.assertEqual(
            validated["runtime_contract"]["policy_phase"],
            {
                "gpu_count": 2,
                "worker_count": 2,
                "states_per_worker": 24,
                "dtype": "bfloat16",
                "feature_repeats": 2,
                "processor_batch_count": 49,
                "vision_forward_count": 97,
                "cosine_scalar_transfer_count": 292,
            },
        )

    def test_scientific_and_access_mutations_fail_closed(self) -> None:
        mutations = (
            (
                lambda value: value["fresh_geometry"].__setitem__(
                    "source_ids_sha256", "0" * 64
                ),
                "geometry",
            ),
            (
                lambda value: value["formal_model_input"]["checkpoints"][0].__setitem__(
                    "model_state_sha256",
                    value["formal_model_input"]["checkpoints"][1][
                        "model_state_sha256"
                    ],
                ),
                "distinctness|checkpoint",
            ),
            (
                lambda value: value["formal_model_input"][
                    "remote_tree_auxiliary_files"
                ][0].__setitem__("sha256", "0" * 64),
                "remote-tree auxiliary inventory",
            ),
            (
                lambda value: value["derived_input"]["files"][3].__setitem__(
                    "record_count", 63
                ),
                "derived",
            ),
            (
                lambda value: value["derived_input"].__setitem__(
                    "generic_full_semantic_reader_allowed", True
                ),
                "derived generic_full_semantic_reader_allowed",
            ),
            (
                lambda value: value["label_input"]["archive"].__setitem__(
                    "sha256", "0" * 64
                ),
                "label archive binding",
            ),
            (
                lambda value: value["label_input"]["sidecar"].__setitem__(
                    "sha256", "0" * 64
                ),
                "label sidecar binding",
            ),
            (
                lambda value: value["label_input"]["raw_states_member"].__setitem__(
                    "run_contract_sha256", "0" * 64
                ),
                "raw-state",
            ),
            (
                lambda value: value["policy_vision_input"].__setitem__(
                    "immutable_revision", "0" * 40
                ),
                "policy vision input binding",
            ),
            (
                lambda value: value["runtime_contract"]["policy_phase"].__setitem__(
                    "gpu_count", 1
                ),
                "policy runtime phase",
            ),
            (
                lambda value: value["local_first_state_machine"][
                    "ordered_states"
                ].reverse(),
                "local state-machine contract",
            ),
            (
                lambda value: value["evaluation_contract"]["bootstrap"].__setitem__(
                    "resamples", 9999
                ),
                "bootstrap",
            ),
            (
                lambda value: value["evaluation_contract"][
                    "go_selector_all"
                ].__setitem__(
                    "ensemble_normalized_recovery_over_exact_minimum", 0.79
                ),
                "selector GO thresholds",
            ),
            (
                lambda value: value["evaluation_contract"][
                    "go_set_conditioning_all"
                ].__setitem__(
                    "paired_seed_positive_count_minimum", 3
                ),
                "set-conditioning GO thresholds",
            ),
            (
                lambda value: value["source_only_operation_contract"].__setitem__(
                    "fresh16_label_semantic_decode_count", 1
                ),
                "source-only operation contract",
            ),
            (
                lambda value: value["source_only_operation_contract"].pop(
                    "model_forward_count"
                ),
                "source-only operation contract",
            ),
            (
                lambda value: value["execution_planned_operation_contract"].pop(
                    "exact_subset_oracle_invocation_count"
                ),
                "execution operation contract",
            ),
            (
                lambda value: value["access_firewall"].__setitem__(
                    "all_heuristics_and_learned_selections_sealed_before_label_access",
                    False,
                ),
                "access firewall contract",
            ),
            (
                lambda value: value["authorization"].__setitem__(
                    "fresh16_access_authorized", True
                ),
                "Source-A authorization contract",
            ),
            (
                lambda value: value["authorization"].__setitem__(
                    "unknown_authorized", False
                ),
                "Source-A authorization contract",
            ),
        )
        for mutate, message in mutations:
            value = copy.deepcopy(self.config)
            mutate(value)
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                validate_fresh16_evaluation_contract_data(value)

    def test_source_only_validator_reports_zero_access_and_no_runner_b(self) -> None:
        self.assertFalse((ROOT / RUNNER_FREEZE_B_PATH).exists())
        result = validate_fresh16_evaluation_source_only_contract(
            CANONICAL_CONFIG_PATH,
            repository_root=ROOT,
        )
        self.assertEqual(result["config_sha256"], FROZEN_CONFIG_SHA256)
        self.assertFalse(result["runner_freeze_b_present"])
        self.assertTrue(result["gate_trained"])
        for key, value in result.items():
            if key.endswith("_count"):
                self.assertEqual(value, 0, key)
        for key in (
            "evaluation_executed",
            "execution_authorized",
            "fresh16_access_authorized",
            "label_access_authorized",
            "legacy_dev5_access_authorized",
            "confirm20_access_authorized",
            "matched_nll_authorized",
            "closed_loop_authorized",
        ):
            self.assertFalse(result[key], key)


if __name__ == "__main__":
    unittest.main()
