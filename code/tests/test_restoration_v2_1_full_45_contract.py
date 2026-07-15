from __future__ import annotations

import argparse
import copy
import json
import unittest
from pathlib import Path

from causalcache.restoration_v2_1_full_45_contract import (
    CANONICAL_ARCHIVE_PATH,
    CANONICAL_LEDGER_PATH,
    CANONICAL_OUTPUT_DIR,
    FROZEN_RESTORATION_V2_1_FULL_45_SHA256,
    FULL_45_PROJECTION_SHA256,
    INVALID_OUTCOME,
    NO_GO_OUTCOME,
    PASS_OUTCOME,
    RestorationV21Full45Contract,
    validate_restoration_v2_1_full_45_contract,
)
from scripts.validate_restoration_v2_1_full_45_contract import validate_from_args


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPOSITORY_ROOT
    / "code/configs/causalcache_restoration_v2_1_full_45.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


class RestorationV21Full45ContractTest(unittest.TestCase):
    def test_frozen_contract_loads_exact_30_plus_15_projection(self) -> None:
        contract = RestorationV21Full45Contract.load(
            CONFIG,
            repository_root=REPOSITORY_ROOT,
        )
        self.assertEqual(
            contract.source_sha256,
            FROZEN_RESTORATION_V2_1_FULL_45_SHA256,
        )
        self.assertEqual(contract.validation["fixed_state_denominator"], 45)
        self.assertEqual(
            contract.validation["role_state_counts"],
            {"v2_label_train": 30, "v2_development": 15},
        )
        self.assertEqual(
            contract.validation["state_projection_sha256"],
            FULL_45_PROJECTION_SHA256,
        )
        state_ids = contract.data["data"]["state_ids_in_exact_order"]
        self.assertEqual(len(state_ids), 45)
        self.assertEqual(
            state_ids[29],
            "0050955459888018:decision_step:006",
        )
        self.assertEqual(
            state_ids[30],
            "0119685762769531:decision_step:004",
        )

    def test_parent_pass_evidence_is_committed_and_exactly_bound(self) -> None:
        parent = RestorationV21Full45Contract.load(
            CONFIG,
            repository_root=REPOSITORY_ROOT,
        ).data["parent_authorization"]
        pilot = parent["fixed_15_pass_manifest"]
        self.assertEqual(pilot["required_outcome"], "PASS_V2_1_INTERFACE_PILOT")
        self.assertEqual(pilot["required_completed_state_count"], 15)
        self.assertEqual(
            pilot["hf_immutable_revision"],
            "bdff8ca71f150afd80d6291b4ecec76cbf9e7432",
        )
        processor = parent["processor_pass_manifest"]
        self.assertEqual(
            processor["required_status"],
            "PASSED_RESTORATION_V2_1_90_PROMPT_PROCESSOR_PREFLIGHT",
        )
        self.assertEqual(processor["required_prompt_count"], 90)
        self.assertIs(
            processor[
                "fresh_immutable_evidence_validation_required_before_runtime_import"
            ],
            True,
        )

    def test_schedule_has_exact_planned_and_maximum_counts(self) -> None:
        schedule = RestorationV21Full45Contract.load(
            CONFIG,
            repository_root=REPOSITORY_ROOT,
        ).data["computation_schedule"]
        expected = {
            "generation_call_count": 90,
            "teacher_forward_count": 135,
            "kl_measurement_count": 90,
        }
        self.assertEqual(
            schedule["planned_counts_if_all_45_states_reach_success_stage"],
            expected,
        )
        self.assertEqual(schedule["maximum_counts_for_entire_attempt"], expected)
        self.assertEqual(
            schedule["per_state_maximum"],
            {
                "full_history_generation_calls": 2,
                "teacher_forward_calls_on_success": 3,
                "kl_measurements_on_success": 2,
            },
        )

    def test_resume_is_not_retry(self) -> None:
        schedule = RestorationV21Full45Contract.load(
            CONFIG,
            repository_root=REPOSITORY_ROOT,
        ).data["computation_schedule"]
        self.assertIs(schedule["resume_allowed"], True)
        self.assertIs(
            schedule["resume_only_skips_existing_terminal_state_records"],
            True,
        )
        self.assertIs(
            schedule["resume_may_attempt_only_states_without_any_attempt_marker"],
            True,
        )
        self.assertIs(
            schedule["attempt_marker_without_terminal_state_invalidates_entire_attempt"],
            True,
        )
        self.assertIs(schedule["incomplete_attempt_retry_allowed"], False)
        self.assertEqual(schedule["state_retry_count"], 0)

    def test_gate_is_parent_numeric_gate_with_versioned_outcomes(self) -> None:
        gate = RestorationV21Full45Contract.load(
            CONFIG,
            repository_root=REPOSITORY_ROOT,
        ).data["substrate_gate"]
        self.assertEqual(gate["minimum_screening_states"], 20)
        self.assertEqual(gate["minimum_parse_coverage"], 0.99)
        self.assertEqual(gate["minimum_finite_logit_coverage"], 1.0)
        self.assertEqual(gate["minimum_repeat_canonical_action_agreement"], 1.0)
        self.assertEqual(gate["minimum_memory_sensitive_states"], 8)
        self.assertEqual(
            gate["repeat_noise_epsilon"],
            "max(1e-4,10*mean_repeat_kl)",
        )
        self.assertEqual(gate["pass_outcome"], PASS_OUTCOME)
        self.assertEqual(gate["fail_outcome"], NO_GO_OUTCOME)
        self.assertEqual(gate["invalid_outcome"], INVALID_OUTCOME)

    def test_confirm_expert_and_restoration_work_remain_zero(self) -> None:
        prohibited = RestorationV21Full45Contract.load(
            CONFIG,
            repository_root=REPOSITORY_ROOT,
        ).data["prohibited_work"]
        for field, value in prohibited.items():
            if field.endswith("_allowed"):
                self.assertIs(value, False, field)
            else:
                self.assertEqual(value, 0, field)

    def test_execution_and_hf_identities_are_independent_from_pilot(self) -> None:
        contract = RestorationV21Full45Contract.load(
            CONFIG,
            repository_root=REPOSITORY_ROOT,
        ).data
        execution = contract["execution"]
        self.assertEqual(execution["attempt_id"], "restoration-v2-1-full-45-substrate-v1")
        self.assertEqual(
            execution["canonical_persistent_output_dir"],
            str(CANONICAL_OUTPUT_DIR),
        )
        self.assertEqual(
            execution["canonical_global_attempt_ledger"],
            str(CANONICAL_LEDGER_PATH),
        )
        self.assertEqual(execution["canonical_raw_archive"], str(CANONICAL_ARCHIVE_PATH))
        self.assertNotEqual(
            execution["attempt_id"],
            contract["parent_authorization"]["fixed_15_pass_manifest"][
                "hf_tag"
            ],
        )
        self.assertEqual(
            contract["artifact_destination"]["repo"],
            "gavinlaw/causalcache-restoration-v2-1-full-45-substrate-mobile",
        )

    def test_source_only_cli_never_authorizes_policy_or_gpu(self) -> None:
        result = validate_from_args(
            argparse.Namespace(
                repository_root=REPOSITORY_ROOT,
                config=CONFIG,
            )
        )
        self.assertIs(result["contract_valid"], True)
        self.assertIs(
            result["policy_execution_authorized_by_this_validator"],
            False,
        )
        self.assertIs(
            result["gpu_execution_authorized_by_this_validator"],
            False,
        )
        self.assertEqual(
            result["external_runtime_evidence"]["status"],
            "NOT_CONSUMED_BY_SOURCE_ONLY_VALIDATOR",
        )

    def test_validator_rejects_parent_pass_or_schedule_drift(self) -> None:
        parent = copy.deepcopy(_config())
        parent["parent_authorization"]["fixed_15_pass_manifest"][
            "required_outcome"
        ] = "NO_GO_V2_1_INTERFACE_PILOT"
        with self.assertRaisesRegex(ValueError, "pilot binding required_outcome"):
            validate_restoration_v2_1_full_45_contract(
                parent,
                repository_root=REPOSITORY_ROOT,
            )

        schedule = copy.deepcopy(_config())
        schedule["computation_schedule"]["maximum_counts_for_entire_attempt"][
            "teacher_forward_count"
        ] = 136
        with self.assertRaisesRegex(ValueError, "maximum attempt schedule"):
            validate_restoration_v2_1_full_45_contract(
                schedule,
                repository_root=REPOSITORY_ROOT,
            )

    def test_validator_rejects_resume_gate_or_prohibited_work_drift(self) -> None:
        resume = copy.deepcopy(_config())
        resume["computation_schedule"][
            "attempt_marker_without_terminal_state_invalidates_entire_attempt"
        ] = False
        with self.assertRaisesRegex(ValueError, "schedule.*must be true"):
            validate_restoration_v2_1_full_45_contract(
                resume,
                repository_root=REPOSITORY_ROOT,
            )

        gate = copy.deepcopy(_config())
        gate["substrate_gate"]["minimum_memory_sensitive_states"] = 7
        with self.assertRaisesRegex(ValueError, "minimum_memory_sensitive_states"):
            validate_restoration_v2_1_full_45_contract(
                gate,
                repository_root=REPOSITORY_ROOT,
            )

        coalition = copy.deepcopy(_config())
        coalition["prohibited_work"][
            "maximum_restoration_coalition_construction_count"
        ] = 1
        with self.assertRaisesRegex(ValueError, "prohibited_work"):
            validate_restoration_v2_1_full_45_contract(
                coalition,
                repository_root=REPOSITORY_ROOT,
            )


if __name__ == "__main__":
    unittest.main()
