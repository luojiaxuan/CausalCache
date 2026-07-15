from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

from causalcache.restoration_v2_1_contract import (
    FROZEN_RESTORATION_V2_1_PILOT_SHA256,
    FULL_45_PROJECTION_SHA256,
    PILOT_PROJECTION_SHA256,
    RestorationV21PilotContract,
    validate_restoration_v2_1_pilot_contract,
    validate_restoration_v2_1_processor_preflight,
)
from scripts.validate_restoration_v2_1_contract import validate_from_args


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY_ROOT / "code/configs/causalcache_restoration_v2_1_pilot.json"


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _preflight() -> dict[str, object]:
    return {
        "schema_version": "0.1.0",
        "protocol_id": "causalcache_restoration_v2_1_official_tool_interface",
        "status": "PASSED_RESTORATION_V2_1_90_PROMPT_PROCESSOR_PREFLIGHT",
        "contract_sha256": FROZEN_RESTORATION_V2_1_PILOT_SHA256,
        "selection_manifest_sha256": "292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b",
        "policy_interface_source_sha256": "9c2f992d1b67dfc7f825b0bea6c076784d640cdd6abc87ddeb2bd2ff808c5a2c",
        "runtime_source_sha256": "a34b4417c23c085fb6a01e29dcc6398a00804c07eff4945812f9ba4688de6ba5",
        "canonical_tool_schema_sha256": "8f92f2fe5eeda45af1e41852f364d2fa4738fbb502eb4fb80f6ef1025691f6c7",
        "chat_template_file_sha256": "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
        "chat_template_text_sha256": "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4",
        "assistant_prefix_token_ids": [151644, 77091, 198],
        "tool_call_open_token_id": 151657,
        "tool_call_close_token_id": 151658,
        "standard_eos_token_ids": [151645, 151643],
        "pad_token_id": 151643,
        "state_count": 45,
        "prompt_count": 90,
        "official_tools_injected_prompt_count": 90,
        "fidelity_counts": {"reference": 45, "summary_only": 45},
        "role_state_counts": {"v2_label_train": 30, "v2_development": 15},
        "image_count_distribution": {"1": 45, "3": 15, "4": 15, "5": 15},
        "teacher_boundary_validated": True,
        "full_45_state_projection_sha256": FULL_45_PROJECTION_SHA256,
        "reserved_generation_tokens": 256,
        "maximum_context_tokens": 32768,
        "context_overflow_count": 0,
        "shape_records_sha256": "1" * 64,
        "policy_model_loaded": False,
        "policy_forward_executed": False,
        "policy_generation_executed": False,
        "confirm_accessed": False,
        "restoration_output_generated": False,
    }


class RestorationV21PilotContractTest(unittest.TestCase):
    def test_frozen_contract_loads_and_derives_exact_denominators(self) -> None:
        contract = RestorationV21PilotContract.load(
            CONFIG,
            repository_root=REPOSITORY_ROOT,
        )
        self.assertEqual(
            contract.protocol_id,
            "causalcache_restoration_v2_1_official_tool_interface",
        )
        self.assertEqual(
            contract.source_sha256,
            FROZEN_RESTORATION_V2_1_PILOT_SHA256,
        )
        self.assertEqual(contract.validation["pilot_state_count"], 15)
        self.assertEqual(contract.validation["planned_generation_call_count"], 15)
        self.assertEqual(
            contract.validation["pilot_state_projection_sha256"],
            PILOT_PROJECTION_SHA256,
        )
        self.assertEqual(
            contract.data["promotion"]["full_45_state_projection_sha256"],
            FULL_45_PROJECTION_SHA256,
        )

    def test_contract_declares_new_interface_without_relabelling_v2(self) -> None:
        contract = RestorationV21PilotContract.load(
            CONFIG,
            repository_root=REPOSITORY_ROOT,
        ).data
        change = contract["change_control"]
        self.assertEqual(
            change["change_class"],
            "new_policy_interface_protocol_not_format_only_adapter",
        )
        self.assertIs(change["v2_results_may_be_relabelled"], False)
        self.assertEqual(
            change["parent_substrate_result"]["outcome_must_remain"],
            "NO_GO_V2_SUBSTRATE",
        )
        self.assertEqual(
            change["parent_parser_replay"]["outcome_must_remain"],
            "NO_GO_ADAPTER_ONLY",
        )

    def test_generation_is_one_greedy_official_tool_call_without_fallback(self) -> None:
        contract = RestorationV21PilotContract.load(
            CONFIG,
            repository_root=REPOSITORY_ROOT,
        ).data
        generation = contract["primary_policy"]["generation"]
        self.assertEqual(generation["planned_generations_per_state"], 1)
        self.assertIs(generation["do_sample"], False)
        self.assertEqual(generation["eos_token_id"], 151658)
        self.assertEqual(generation["suppress_token_ids"], [151643, 151645])
        self.assertIsNone(generation["fallback_generation_interface"])
        interface = contract["policy_interface"]
        self.assertIs(interface["action_line_carrier_present"], False)
        self.assertIs(interface["model_emitted_closer_required"], True)
        self.assertIs(interface["host_appended_closer_allowed"], False)
        self.assertIs(interface["compatibility_parser_fallback_allowed"], False)
        self.assertIs(
            interface["wrapper"]["text_arguments_must_already_be_unicode_nfkc"],
            True,
        )
        self.assertIs(
            interface["wrapper"]["non_nfkc_text_arguments_rejected_by_parser"],
            True,
        )
        teacher = interface["teacher_target"]
        self.assertEqual(
            teacher["suppressed_standard_eos_token_ids"],
            [151645, 151643],
        )
        self.assertEqual(
            teacher["generation_suppression_semantics"],
            "negative_infinity_via_transformers_suppress_tokens",
        )
        self.assertEqual(
            teacher["teacher_suppression_semantics"],
            "torch_finfo_bfloat16_min_finite",
        )
        self.assertEqual(
            teacher["teacher_mask_application"],
            "same_mask_on_every_reference_and_candidate_action_path_position_"
            "before_float32_log_softmax",
        )
        self.assertIs(
            teacher["target_disjoint_from_suppressed_standard_eos_required"],
            True,
        )

    def test_gate_requires_all_15_and_forbids_hidden_work(self) -> None:
        contract = RestorationV21PilotContract.load(
            CONFIG,
            repository_root=REPOSITORY_ROOT,
        ).data
        gate = contract["pilot_gate"]
        self.assertEqual(gate["required_exact_whole_output_parse_count"], 15)
        self.assertEqual(gate["required_model_emitted_closer_count"], 15)
        self.assertEqual(gate["required_androidworld_bridge_count"], 15)
        self.assertEqual(gate["maximum_max_token_truncation_count"], 0)
        self.assertEqual(gate["pass_outcome"], "PASS_V2_1_INTERFACE_PILOT")
        self.assertEqual(gate["fail_outcome"], "NO_GO_V2_1_INTERFACE_PILOT")
        execution = contract["pilot_execution"]
        for field in (
            "state_retry_count",
            "top_up_count",
            "teacher_forward_count",
            "kl_measurement_count",
            "restoration_label_count",
            "expert_action_read_count",
            "confirm_state_generation_count",
            "label_train_state_generation_count",
        ):
            self.assertEqual(execution[field], 0)
        self.assertIs(contract["promotion"]["pilot_pass_does_not_authorize_confirm"], True)
        self.assertEqual(
            contract["processor_preflight"]["required_success_status"],
            "PASSED_RESTORATION_V2_1_90_PROMPT_PROCESSOR_PREFLIGHT",
        )
        self.assertEqual(
            contract["processor_preflight"]["runtime_source_sha256"],
            "a34b4417c23c085fb6a01e29dcc6398a00804c07eff4945812f9ba4688de6ba5",
        )
        self.assertEqual(
            contract["processor_preflight"]["official_tools_injected_prompt_count"],
            90,
        )
        self.assertEqual(
            contract["processor_preflight"]["image_count_distribution"],
            {"1": 45, "3": 15, "4": 15, "5": 15},
        )
        self.assertIs(
            contract["processor_preflight"]["teacher_boundary_validated"],
            True,
        )
        self.assertIs(contract["processor_preflight"]["confirm_accessed"], False)
        self.assertIs(
            contract["processor_preflight"]["restoration_output_generated"],
            False,
        )
        self.assertNotIn("status", contract["processor_preflight"])

    def test_semantic_validator_rejects_retry_or_threshold_drift(self) -> None:
        retry = copy.deepcopy(_config())
        retry["pilot_execution"]["state_retry_count"] = 1
        with self.assertRaisesRegex(ValueError, "pilot_execution"):
            validate_restoration_v2_1_pilot_contract(
                retry,
                repository_root=REPOSITORY_ROOT,
            )

        threshold = copy.deepcopy(_config())
        threshold["pilot_gate"]["required_exact_whole_output_parse_count"] = 14
        with self.assertRaisesRegex(ValueError, "pilot_gate"):
            validate_restoration_v2_1_pilot_contract(
                threshold,
                repository_root=REPOSITORY_ROOT,
            )

    def test_semantic_validator_rejects_interface_source_drift(self) -> None:
        changed = copy.deepcopy(_config())
        changed["policy_interface"]["source"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source SHA256 drifted"):
            validate_restoration_v2_1_pilot_contract(
                changed,
                repository_root=REPOSITORY_ROOT,
            )

    def test_semantic_validator_rejects_nfkc_or_teacher_mask_drift(self) -> None:
        nfkc = copy.deepcopy(_config())
        nfkc["policy_interface"]["wrapper"][
            "non_nfkc_text_arguments_rejected_by_parser"
        ] = False
        with self.assertRaisesRegex(ValueError, "policy_interface.wrapper"):
            validate_restoration_v2_1_pilot_contract(
                nfkc,
                repository_root=REPOSITORY_ROOT,
            )

        teacher_mask = copy.deepcopy(_config())
        teacher_mask["policy_interface"]["teacher_target"][
            "teacher_mask_application"
        ] = "candidate_only_after_log_softmax"
        with self.assertRaisesRegex(ValueError, "policy_interface.teacher_target"):
            validate_restoration_v2_1_pilot_contract(
                teacher_mask,
                repository_root=REPOSITORY_ROOT,
            )

    def test_processor_preflight_requires_all_90_without_policy(self) -> None:
        result = validate_restoration_v2_1_processor_preflight(
            _preflight(),
            contract_sha256=FROZEN_RESTORATION_V2_1_PILOT_SHA256,
            selection_manifest_sha256="292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b",
            policy_interface_source_sha256="9c2f992d1b67dfc7f825b0bea6c076784d640cdd6abc87ddeb2bd2ff808c5a2c",
        )
        self.assertEqual(result["prompt_count"], 90)

        forward = _preflight()
        forward["policy_forward_executed"] = True
        with self.assertRaisesRegex(ValueError, "policy_forward_executed"):
            validate_restoration_v2_1_processor_preflight(
                forward,
                contract_sha256=FROZEN_RESTORATION_V2_1_PILOT_SHA256,
                selection_manifest_sha256="292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b",
                policy_interface_source_sha256="9c2f992d1b67dfc7f825b0bea6c076784d640cdd6abc87ddeb2bd2ff808c5a2c",
            )

        missing_tool_injection = _preflight()
        missing_tool_injection["official_tools_injected_prompt_count"] = 89
        with self.assertRaisesRegex(ValueError, "official-tools prompt count"):
            validate_restoration_v2_1_processor_preflight(
                missing_tool_injection,
                contract_sha256=FROZEN_RESTORATION_V2_1_PILOT_SHA256,
                selection_manifest_sha256="292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b",
                policy_interface_source_sha256="9c2f992d1b67dfc7f825b0bea6c076784d640cdd6abc87ddeb2bd2ff808c5a2c",
            )

        confirm = _preflight()
        confirm["confirm_accessed"] = True
        with self.assertRaisesRegex(ValueError, "confirm_accessed"):
            validate_restoration_v2_1_processor_preflight(
                confirm,
                contract_sha256=FROZEN_RESTORATION_V2_1_PILOT_SHA256,
                selection_manifest_sha256="292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b",
                policy_interface_source_sha256="9c2f992d1b67dfc7f825b0bea6c076784d640cdd6abc87ddeb2bd2ff808c5a2c",
            )

        wrong_runtime = _preflight()
        wrong_runtime["runtime_source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "runtime SHA256"):
            validate_restoration_v2_1_processor_preflight(
                wrong_runtime,
                contract_sha256=FROZEN_RESTORATION_V2_1_PILOT_SHA256,
                selection_manifest_sha256="292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b",
                policy_interface_source_sha256="9c2f992d1b67dfc7f825b0bea6c076784d640cdd6abc87ddeb2bd2ff808c5a2c",
            )

    def test_cli_reports_pending_or_validated_processor_preflight(self) -> None:
        pending = validate_from_args(
            argparse.Namespace(
                repository_root=REPOSITORY_ROOT,
                config=CONFIG,
                processor_preflight=None,
            )
        )
        self.assertIs(pending["contract_valid"], True)
        self.assertIs(pending["processor_preflight"]["valid"], False)
        self.assertIs(pending["policy_generation_authorized_by_this_validator"], False)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processor-preflight.json"
            path.write_text(
                json.dumps(_preflight(), ensure_ascii=False),
                encoding="utf-8",
            )
            validated = validate_from_args(
                argparse.Namespace(
                    repository_root=REPOSITORY_ROOT,
                    config=CONFIG,
                    processor_preflight=path,
                )
            )
        self.assertIs(validated["processor_preflight"]["valid"], True)
        self.assertIs(validated["policy_generation_authorized_by_this_validator"], False)


if __name__ == "__main__":
    unittest.main()
