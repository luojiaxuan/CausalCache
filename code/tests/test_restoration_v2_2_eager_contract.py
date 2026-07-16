from __future__ import annotations

import argparse
import copy
import json
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_eager_contract import (
    FROZEN_RESTORATION_V2_2_EAGER_SHA256,
    INVALID_OUTCOME,
    NO_GO_OUTCOME,
    PASS_OUTCOME,
    RestorationV22EagerContract,
    validate_restoration_v2_2_eager_contract,
)
from scripts.validate_restoration_v2_2_eager_contract import validate_from_args


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/configs/causalcache_restoration_v2_2_eager.json"


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


class RestorationV22EagerContractTest(unittest.TestCase):
    def test_frozen_contract_loads_and_binds_authorizing_results(self) -> None:
        contract = RestorationV22EagerContract.load(CONFIG, repository_root=ROOT)
        self.assertEqual(
            contract.source_sha256, FROZEN_RESTORATION_V2_2_EAGER_SHA256
        )
        authorization = contract.data["authorization"]
        self.assertEqual(
            authorization["source_parent_git_commit"],
            "8a88b530bd72bb65f35d569aac1ea6e9263cc1a6",
        )
        self.assertEqual(
            authorization["spatial_reference_summary"]["required_decision"],
            "EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY",
        )
        self.assertEqual(
            authorization["v2_1_full_45_no_go_artifact"]["required_outcome"],
            "NO_GO_V2_1_FULL_45_SUBSTRATE",
        )

    def test_attempt_requires_fresh_45_outputs(self) -> None:
        data = RestorationV22EagerContract.load(CONFIG, repository_root=ROOT).data[
            "data"
        ]
        self.assertEqual(data["fixed_state_denominator"], 45)
        self.assertIs(data["fresh_state_output_required"], True)
        self.assertIs(
            data["v2_1_raw_state_or_aggregate_output_may_be_imported"], False
        )
        self.assertIs(data["v2_1_raw_output_may_count_toward_denominator"], False)
        self.assertEqual(data["v2_1_state_output_reuse_count"], 0)

    def test_runtime_is_exact_eager_control_without_strict_claim(self) -> None:
        runtime = RestorationV22EagerContract.load(
            CONFIG, repository_root=ROOT
        ).data["runtime"]
        self.assertEqual(runtime["dtype"], "bfloat16")
        self.assertEqual(runtime["attention_implementation_requested"], "eager")
        self.assertEqual(
            runtime["attention_implementation_observed_must_equal"], "eager"
        )
        self.assertEqual(
            runtime["eager_control_flags"],
            {
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
                "seed": 0,
            },
        )
        self.assertIs(runtime["pytorch_strict_deterministic_algorithms_enabled"], False)
        self.assertEqual(
            runtime["expected_execution_stack"],
            {
                "container_image_digest": (
                    "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
                ),
                "python_version": "3.12.3",
                "torch_version": "2.11.0+cu130",
                "torch_cuda_version": "13.0",
                "cudnn_version": 91900,
                "transformers_version": "5.6.0",
                "nvidia_driver_version": "570.172.08",
                "gpu_name": "NVIDIA H200",
            },
        )
        self.assertEqual(
            runtime["worker_identity_fields_required"],
            [
                "device",
                "gpu_name",
                "gpu_uuid",
                "gpu_pci_bus_id",
                "logical_device_index",
                "nvidia_smi_index",
            ],
        )
        self.assertEqual(
            runtime["worker_identity_fields_allowed_to_differ"],
            [
                "device",
                "gpu_uuid",
                "gpu_pci_bus_id",
                "logical_device_index",
                "nvidia_smi_index",
            ],
        )

    def test_exact_two_h200_workers_use_fixed_parity_shards(self) -> None:
        topology = RestorationV22EagerContract.load(
            CONFIG, repository_root=ROOT
        ).data["execution"]["worker_topology"]
        self.assertEqual(topology["worker_count"], 2)
        self.assertEqual(topology["gpu_model"], "NVIDIA H200")
        self.assertEqual(topology["workers"][0]["state_indices"], list(range(0, 45, 2)))
        self.assertEqual(topology["workers"][1]["state_indices"], list(range(1, 45, 2)))
        self.assertIs(topology["cross_worker_state_stealing_allowed"], False)
        self.assertIs(topology["worker_failure_invalidates_entire_attempt"], True)

    def test_scientific_schedule_is_unchanged_and_attempt_is_not_resumable(self) -> None:
        contract = RestorationV22EagerContract.load(CONFIG, repository_root=ROOT).data
        schedule = contract["computation_schedule"]
        self.assertEqual(
            schedule["maximum_counts_for_entire_attempt"],
            {
                "generation_call_count": 90,
                "teacher_forward_count": 135,
                "kl_measurement_count": 90,
            },
        )
        self.assertIs(schedule["resume_allowed"], False)
        self.assertIs(
            schedule["resume_only_skips_existing_terminal_state_records"], False
        )
        self.assertIs(
            schedule["resume_may_attempt_only_states_without_any_attempt_marker"],
            False,
        )
        gate = contract["substrate_gate"]
        self.assertEqual(gate["minimum_memory_sensitive_states"], 8)
        self.assertEqual(gate["minimum_repeat_canonical_action_agreement"], 1.0)
        self.assertEqual(gate["pass_outcome"], PASS_OUTCOME)
        self.assertEqual(gate["fail_outcome"], NO_GO_OUTCOME)
        self.assertEqual(gate["invalid_outcome"], INVALID_OUTCOME)

    def test_confirm_restoration_and_gate_operations_are_zero(self) -> None:
        prohibited = RestorationV22EagerContract.load(
            CONFIG, repository_root=ROOT
        ).data["prohibited_work"]
        for field, value in prohibited.items():
            if field.endswith("_allowed"):
                self.assertIs(value, False, field)
            else:
                self.assertEqual(value, 0, field)

    def test_inventory_reserves_new_formal_modules(self) -> None:
        inventory = RestorationV22EagerContract.load(
            CONFIG, repository_root=ROOT
        ).data["scientific_inheritance"]["formal_run_source_inventory_paths"]
        for path in (
            "code/causalcache/policy/gui_owl_v2_2_eager_runtime.py",
            "code/causalcache/restoration_v2_2_eager_artifact.py",
            "code/scripts/run_restoration_v2_2_eager_substrate.py",
            "code/scripts/manage_restoration_v2_2_eager_artifact.py",
        ):
            self.assertIn(path, inventory)

    def test_source_only_cli_does_not_authorize_execution(self) -> None:
        result = validate_from_args(
            argparse.Namespace(repository_root=ROOT, config=CONFIG)
        )
        self.assertIs(result["contract_valid"], True)
        self.assertIs(result["policy_execution_authorized_by_this_validator"], False)
        self.assertIs(result["gpu_execution_authorized_by_this_validator"], False)
        self.assertEqual(result["worker_state_counts"], {"even": 23, "odd": 22})

    def test_mutations_of_freshness_runtime_or_shards_fail_closed(self) -> None:
        freshness = copy.deepcopy(_config())
        freshness["data"]["v2_1_raw_output_may_count_toward_denominator"] = True
        with self.assertRaisesRegex(ValueError, "v2_1_raw_output"):
            validate_restoration_v2_2_eager_contract(freshness, repository_root=ROOT)

        runtime = copy.deepcopy(_config())
        runtime["runtime"]["eager_control_flags"]["cuda_matmul_allow_tf32"] = True
        with self.assertRaisesRegex(ValueError, "eager numerical controls"):
            validate_restoration_v2_2_eager_contract(runtime, repository_root=ROOT)

        stack = copy.deepcopy(_config())
        stack["runtime"]["expected_execution_stack"]["torch_version"] = "2.12.0"
        with self.assertRaisesRegex(ValueError, "recovered execution stack"):
            validate_restoration_v2_2_eager_contract(stack, repository_root=ROOT)

        shards = copy.deepcopy(_config())
        shards["execution"]["worker_topology"]["workers"][0]["state_indices"][0] = 1
        with self.assertRaisesRegex(ValueError, "parity shard"):
            validate_restoration_v2_2_eager_contract(shards, repository_root=ROOT)

        resumable = copy.deepcopy(_config())
        resumable["computation_schedule"]["resume_allowed"] = True
        with self.assertRaisesRegex(ValueError, "non-resumable"):
            validate_restoration_v2_2_eager_contract(resumable, repository_root=ROOT)


if __name__ == "__main__":
    unittest.main()
