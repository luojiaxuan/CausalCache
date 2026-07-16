from __future__ import annotations

import copy
import io
import json
import unittest
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from causalcache.restoration_v2_2_policy_vision_contract import (
    CANONICAL_CONFIG_PATH as PARENT_CONFIG_PATH,
    FROZEN_CONFIG_SHA256 as PARENT_CONFIG_SHA256,
    PROTOCOL_ID as PARENT_PROTOCOL_ID,
    sha256_file,
    strict_json_object_bytes,
)
from causalcache.restoration_v2_2_policy_vision_v2_contract import (
    CANONICAL_CONFIG_PATH,
    CANONICAL_OUTPUT_DIRECTORY,
    FAILURE_DIRECTORY,
    FAILURE_GIT_COMMIT,
    FORMAL_ATTEMPT_LEDGER_PATH,
    FROZEN_CONFIG_SHA256,
    GPU_UUID_RUNTIME_TYPE_PROFILE,
    GPU_UUID_VALUE_TYPE_PROFILE,
    GPU_UUID_VALUE_TYPE_PROFILE_MAPPING,
    PASS_STATUS,
    PROTOCOL_ID,
    RUN_STATUS,
    UNCHANGED_PARENT_SECTIONS,
    VALID_STATUS,
    V1_CANONICAL_OUTPUT_DIRECTORY,
    RestorationV22PolicyVisionV2Contract,
    _validate_overlay_semantics,
    validate_contract,
)
from scripts.validate_restoration_v2_2_policy_vision_v2_contract import (
    main as validate_main,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH
PARENT_CONFIG = ROOT / PARENT_CONFIG_PATH


class RestorationV22PolicyVisionV2ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_data = strict_json_object_bytes(
            CONFIG.read_bytes(),
            label="policy-vision v2 test config",
        )
        cls.contract = RestorationV22PolicyVisionV2Contract.load(
            CONFIG,
            repository_root=ROOT,
        )

    def test_frozen_contract_validates_with_new_identity_and_output(self) -> None:
        self.assertEqual(sha256_file(CONFIG), FROZEN_CONFIG_SHA256)
        result = validate_contract(CONFIG, repository_root=ROOT)
        self.assertEqual(result["status"], PASS_STATUS)
        self.assertEqual(result["protocol_id"], PROTOCOL_ID)
        self.assertEqual(result["run_status"], RUN_STATUS)
        self.assertEqual(result["valid_status"], VALID_STATUS)
        self.assertEqual(
            result["gpu_uuid_value_type_profile"],
            GPU_UUID_VALUE_TYPE_PROFILE,
        )
        self.assertEqual(
            result["canonical_output_directory"],
            CANONICAL_OUTPUT_DIRECTORY,
        )
        self.assertFalse(result["scientific_contract_changed"])
        self.assertTrue(result["output_absent"])
        self.assertTrue(result["staging_output_absent"])

    def test_runner_compatible_data_changes_only_protocol_prereg_output_identity(self) -> None:
        contract = self.contract
        self.assertEqual(contract.path, CONFIG.resolve())
        self.assertEqual(contract.sha256, FROZEN_CONFIG_SHA256)
        self.assertEqual(contract.repository_root, ROOT.resolve())
        self.assertIsInstance(contract.data, Mapping)
        self.assertEqual(contract.data["protocol_id"], PROTOCOL_ID)
        self.assertEqual(
            contract.data["preregistration_status"],
            self.config_data["preregistration_status"],
        )
        self.assertEqual(
            contract.data["output_contract"],
            self.config_data["output_contract"],
        )
        for section in UNCHANGED_PARENT_SECTIONS:
            self.assertEqual(
                contract.data[section],
                contract.parent.data[section],
                section,
            )
        expected_keys = set(contract.parent.data) | {"repair_identity"}
        self.assertEqual(set(contract.data), expected_keys)

    def test_repair_identity_is_self_contained_for_summary(self) -> None:
        repair = self.contract.repair_identity
        self.assertIsInstance(repair, Mapping)
        self.assertEqual(repair, self.contract.data["repair_identity"])
        self.assertEqual(
            repair["repair_config"],
            {
                "path": CANONICAL_CONFIG_PATH,
                "sha256": FROZEN_CONFIG_SHA256,
                "protocol_id": PROTOCOL_ID,
            },
        )
        self.assertEqual(
            repair["parent_contract"]["sha256"],
            PARENT_CONFIG_SHA256,
        )
        self.assertEqual(
            repair["parent_contract"]["protocol_id"],
            PARENT_PROTOCOL_ID,
        )
        self.assertEqual(
            repair["failed_attempt"]["git_commit"],
            FAILURE_GIT_COMMIT,
        )
        self.assertEqual(
            repair["failed_attempt"]["status"],
            "INVALID_POLICY_VISION_V1_ZERO_FEATURE_GPU_UUID_TYPE",
        )
        self.assertEqual(
            [record["sha256"] for record in repair["failed_attempt"]["exact_files"]],
            [
                "50dada41703c45523839c2904ec5d32f86553a71d48b46c815dd5d4dbfb9e144",
                "1cea24506b14b9dcd4bdfff77c82024417f65e1416a418caeec0d1192c2b0311",
            ],
        )
        self.assertEqual(
            repair["gpu_uuid_value_type_profile"],
            GPU_UUID_VALUE_TYPE_PROFILE,
        )
        self.assertEqual(
            repair["gpu_uuid_runtime_type_profile"],
            GPU_UUID_RUNTIME_TYPE_PROFILE,
        )
        self.assertEqual(
            repair["canonical_output_directory"],
            CANONICAL_OUTPUT_DIRECTORY,
        )
        self.assertEqual(
            repair["formal_attempt_ledger_path"],
            FORMAL_ATTEMPT_LEDGER_PATH,
        )
        self.assertIn(
            "code/scripts/run_restoration_v2_2_policy_vision_baseline.py",
            repair["implementation_changed_paths_allowlist"],
        )

    def test_repair_is_strictly_cuuid_to_str_then_existing_checks(self) -> None:
        repair = self.config_data["repair_scope"]
        semantic = repair["semantic_repair"]
        implementation = repair["implementation_boundary"]
        self.assertTrue(repair["gpu_uuid_value_type_only"])
        self.assertEqual(repair["pinned_torch_version"], "2.11.0+cu130")
        self.assertEqual(
            repair["runtime_type_profile"],
            GPU_UUID_RUNTIME_TYPE_PROFILE,
        )
        self.assertEqual(
            repair["contract_to_runtime_profile_mapping"],
            GPU_UUID_VALUE_TYPE_PROFILE_MAPPING,
        )
        self.assertEqual(
            semantic["newly_accepted_runtime_type"],
            "torch._C._CUuuid",
        )
        self.assertEqual(
            semantic["acceptance_predicate"],
            "type(value) is loaded torch._C._CUuuid",
        )
        self.assertEqual(semantic["conversion"], "value = str(value)")
        self.assertTrue(semantic["conversion_before_existing_parser"])
        self.assertTrue(semantic["existing_str_and_bytes_paths_unchanged"])
        self.assertTrue(semantic["all_other_runtime_types_rejected"])
        self.assertEqual(
            semantic["post_conversion_checks_reused_exactly"],
            [
                "canonical_gpu_uuid_format_and_lowercase_GPU_prefix",
                "observed_uuid_equals_expected_gpu_uuid",
                "nvidia_smi_uuid_matches_exactly_one_row",
                "nvidia_smi_pci_bus_id_matches_frozen_format",
                "nvidia_smi_index_is_nonnegative_integer",
            ],
        )
        self.assertEqual(
            set(implementation["existing_symbols_allowed_to_change"]),
            {
                "code/causalcache/policy/gui_owl_v2_2_vision_runtime.py",
                "code/causalcache/restoration_v2_2_policy_vision.py",
                "code/scripts/run_restoration_v2_2_policy_vision_baseline.py",
            },
        )
        self.assertTrue(
            implementation[
                "source_freeze_must_match_exact_changed_paths_before_formal_run"
            ]
        )
        self.assertEqual(
            implementation["formal_source_changed_paths_exact"],
            sorted(implementation["formal_source_changed_paths_exact"]),
        )
        self.assertIn(
            "code/scripts/run_restoration_v2_2_policy_vision_baseline.py",
            implementation["formal_source_changed_paths_exact"],
        )
        changed_flags = {
            key: value
            for key, value in repair.items()
            if key.endswith("_changed")
        }
        self.assertTrue(changed_flags)
        self.assertTrue(all(value is False for value in changed_flags.values()))

    def test_output_validation_and_attempt_claim_boundaries_are_exact(self) -> None:
        output = self.config_data["output_contract"]
        self.assertNotIn(
            "validate_rebuilds_every_output_byte_from_immutable_inputs",
            output,
        )
        self.assertTrue(
            output[
                "validate_rebuilds_every_output_byte_from_recorded_feature_rows_"
                "and_immutable_labels_witness"
            ]
        )
        self.assertFalse(output["validate_recomputes_policy_vision_features"])
        self.assertEqual(
            output["formal_attempt_ledger_path"],
            FORMAL_ATTEMPT_LEDGER_PATH,
        )
        self.assertTrue(
            output[
                "formal_attempt_ledger_created_with_o_excl_before_model_or_"
                "feature_access"
            ]
        )

    def test_parent_and_failed_v1_bytes_remain_unchanged(self) -> None:
        self.assertEqual(sha256_file(PARENT_CONFIG), PARENT_CONFIG_SHA256)
        expected = {
            "README.md": "50dada41703c45523839c2904ec5d32f86553a71d48b46c815dd5d4dbfb9e144",
            "failure.json": "1cea24506b14b9dcd4bdfff77c82024417f65e1416a418caeec0d1192c2b0311",
        }
        for name, digest in expected.items():
            self.assertEqual(sha256_file(ROOT / FAILURE_DIRECTORY / name), digest)
        v1_output = ROOT / V1_CANONICAL_OUTPUT_DIRECTORY
        self.assertFalse(v1_output.exists())
        self.assertFalse(
            v1_output.with_name(f".{v1_output.name}.staging").exists()
        )

    def test_formal_source_diff_must_match_the_exact_frozen_inventory(self) -> None:
        expected = self.contract.repair_identity[
            "formal_source_changed_paths_exact"
        ]
        with patch(
            "causalcache.restoration_v2_2_policy_vision_v2_contract."
            "_require_commit"
        ), patch(
            "causalcache.restoration_v2_2_policy_vision_v2_contract."
            "_require_ancestor"
        ), patch(
            "causalcache.restoration_v2_2_policy_vision_v2_contract._git_text",
            return_value="\n".join(reversed(expected)),
        ):
            self.contract.validate_formal_source_diff("a" * 40)
        with patch(
            "causalcache.restoration_v2_2_policy_vision_v2_contract."
            "_require_commit"
        ), patch(
            "causalcache.restoration_v2_2_policy_vision_v2_contract."
            "_require_ancestor"
        ), patch(
            "causalcache.restoration_v2_2_policy_vision_v2_contract._git_text",
            return_value="\n".join(expected[:-1]),
        ), self.assertRaisesRegex(ValueError, "changed-path boundary"):
            self.contract.validate_formal_source_diff("a" * 40)

    def test_overlay_mutations_fail_closed(self) -> None:
        mutations = []
        parent = copy.deepcopy(self.config_data)
        parent["parent_contract"]["sha256"] = "0" * 64
        mutations.append(parent)
        failure = copy.deepcopy(self.config_data)
        failure["failed_attempt"]["exact_files"][1]["sha256"] = "1" * 64
        mutations.append(failure)
        type_scope = copy.deepcopy(self.config_data)
        type_scope["repair_scope"]["semantic_repair"]["acceptance_predicate"] = (
            "isinstance(value, object)"
        )
        mutations.append(type_scope)
        checks = copy.deepcopy(self.config_data)
        checks["repair_scope"]["semantic_repair"][
            "post_conversion_checks_reused_exactly"
        ].pop()
        mutations.append(checks)
        science = copy.deepcopy(self.config_data)
        science["repair_scope"]["statistics_contract_changed"] = True
        mutations.append(science)
        output = copy.deepcopy(self.config_data)
        output["output_contract"]["canonical_result_directory"] = (
            V1_CANONICAL_OUTPUT_DIRECTORY
        )
        mutations.append(output)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                _validate_overlay_semantics(mutation)

    def test_validator_cli_emits_the_locked_source_contract(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            validate_main(
                [
                    "--contract",
                    str(CONFIG),
                    "--repository-root",
                    str(ROOT),
                ]
            )
        result = json.loads(stream.getvalue())
        self.assertEqual(result["status"], PASS_STATUS)
        self.assertEqual(result["config_sha256"], FROZEN_CONFIG_SHA256)


if __name__ == "__main__":
    unittest.main()
