from __future__ import annotations

import copy
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

from causalcache.restoration_v2_2_policy_vision_contract import (
    sha256_file,
    strict_json_object_bytes,
)
from causalcache.restoration_v2_2_policy_vision_v2_contract import (
    FROZEN_CONFIG_SHA256 as PARENT_CONFIG_SHA256,
    GPU_UUID_RUNTIME_TYPE_PROFILE,
    GPU_UUID_VALUE_TYPE_PROFILE,
    PROTOCOL_ID as PARENT_PROTOCOL_ID,
)
from causalcache.restoration_v2_2_policy_vision_v3_contract import (
    CANONICAL_CONFIG_PATH,
    CANONICAL_OUTPUT_DIRECTORY,
    COMBINED_RUNTIME_PROFILE_ID,
    FAILURE_DIRECTORY,
    FAILURE_GIT_COMMIT,
    FORMAL_ATTEMPT_LEDGER_PATH,
    FORMAL_SOURCE_CHANGED_PATHS_EXACT,
    FROZEN_CONFIG_SHA256,
    IMAGE_PROCESSOR_SIZE_RUNTIME_TYPE_PROFILE,
    IMAGE_PROCESSOR_SIZE_VALUE_TYPE_PROFILE,
    IMAGE_PROCESSOR_SIZE_VALUE_TYPE_PROFILE_MAPPING,
    PASS_STATUS,
    PARENT_CONFIG_PATH,
    PROTOCOL_ID,
    RUN_STATUS,
    SOURCE_DIFF_BASELINE_GIT_COMMIT,
    UNCHANGED_PARENT_SECTIONS,
    V1_CANONICAL_OUTPUT_DIRECTORY,
    V2_CANONICAL_OUTPUT_DIRECTORY,
    VALID_STATUS,
    RestorationV22PolicyVisionV3Contract,
    _validate_overlay_semantics,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH
PARENT_CONFIG = ROOT / PARENT_CONFIG_PATH


class RestorationV22PolicyVisionV3ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_data = strict_json_object_bytes(
            CONFIG.read_bytes(),
            label="policy-vision v3 test config",
        )
        cls.contract = RestorationV22PolicyVisionV3Contract.load(
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
            result["image_processor_size_runtime_type_profile"],
            IMAGE_PROCESSOR_SIZE_RUNTIME_TYPE_PROFILE,
        )
        self.assertEqual(
            result["combined_runtime_profile_id"],
            COMBINED_RUNTIME_PROFILE_ID,
        )
        self.assertEqual(
            result["canonical_output_directory"],
            CANONICAL_OUTPUT_DIRECTORY,
        )
        self.assertFalse(result["scientific_contract_changed"])
        self.assertTrue(result["prior_outputs_absent"])
        self.assertTrue(result["output_absent"])
        self.assertTrue(result["staging_output_absent"])

    def test_runner_data_changes_only_versioned_identity_fields(self) -> None:
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
        self.assertEqual(set(contract.data), set(contract.parent.data))

    def test_repair_identity_is_complete_for_runner_and_summary(self) -> None:
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
            "INVALID_POLICY_VISION_V2_ZERO_FEATURE_SIZE_DICT_INTERFACE",
        )
        self.assertEqual(
            [record["sha256"] for record in repair["failed_attempt"]["exact_files"]],
            [
                "9ddcf6eb8374a76c053622d3e16b772246f4c9aea17bc2578fe43f73762e3e90",
                "3fe5c7fd6ea082501627fcbe52d4ca2751dc00f6f9a2270a9fdf20987097e5b7",
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
            repair["image_processor_size_value_type_profile"],
            IMAGE_PROCESSOR_SIZE_VALUE_TYPE_PROFILE,
        )
        self.assertEqual(
            repair["image_processor_size_runtime_type_profile"],
            IMAGE_PROCESSOR_SIZE_RUNTIME_TYPE_PROFILE,
        )
        self.assertEqual(
            repair["image_processor_size_contract_to_runtime_profile_mapping"],
            IMAGE_PROCESSOR_SIZE_VALUE_TYPE_PROFILE_MAPPING,
        )
        self.assertEqual(
            repair["combined_runtime_profile_id"],
            COMBINED_RUNTIME_PROFILE_ID,
        )
        self.assertEqual(
            repair["source_diff_baseline_git_commit"],
            SOURCE_DIFF_BASELINE_GIT_COMMIT,
        )
        self.assertEqual(
            repair["formal_source_changed_paths_exact"],
            list(FORMAL_SOURCE_CHANGED_PATHS_EXACT),
        )
        self.assertEqual(
            repair["canonical_output_directory"],
            CANONICAL_OUTPUT_DIRECTORY,
        )
        self.assertEqual(
            repair["formal_attempt_ledger_path"],
            FORMAL_ATTEMPT_LEDGER_PATH,
        )

    def test_repair_is_exact_loaded_size_dict_to_frozen_plain_dict(self) -> None:
        repair = self.config_data["repair_scope"]
        semantic = repair["semantic_repair"]
        implementation = repair["implementation_boundary"]
        self.assertTrue(repair["image_processor_size_interface_only"])
        self.assertEqual(repair["pinned_transformers_version"], "5.6.0")
        self.assertEqual(
            repair["gpu_uuid_runtime_type_profile"],
            GPU_UUID_RUNTIME_TYPE_PROFILE,
        )
        self.assertEqual(
            repair["image_processor_size_runtime_type_profile"],
            IMAGE_PROCESSOR_SIZE_RUNTIME_TYPE_PROFILE,
        )
        self.assertEqual(
            semantic["newly_accepted_runtime_type"],
            "transformers.image_utils.SizeDict",
        )
        self.assertEqual(
            semantic["acceptance_predicate"],
            "type(value) is loaded transformers.image_utils.SizeDict",
        )
        self.assertFalse(semantic["mapping_membership_required"])
        self.assertTrue(
            semantic["mapping_implementation_explicitly_rejected_for_v3"]
        )
        self.assertEqual(semantic["conversion"], "normalized = dict(value)")
        self.assertEqual(
            semantic["exact_normalized_values"],
            {"longest_edge": 2621440, "shortest_edge": 2621440},
        )
        self.assertTrue(semantic["existing_v1_mapping_profile_unchanged"])
        self.assertTrue(semantic["existing_v2_gpu_uuid_profile_unchanged"])
        self.assertEqual(
            implementation["source_diff_baseline_git_commit"],
            FAILURE_GIT_COMMIT,
        )
        self.assertEqual(
            implementation["formal_source_changed_paths_exact"],
            sorted(implementation["formal_source_changed_paths_exact"]),
        )
        self.assertEqual(
            len(implementation["formal_source_changed_paths_exact"]),
            len(set(implementation["formal_source_changed_paths_exact"])),
        )
        changed_flags = {
            key: value
            for key, value in repair.items()
            if key.endswith("_changed")
        }
        self.assertTrue(changed_flags)
        self.assertTrue(all(value is False for value in changed_flags.values()))

    def test_parent_failure_and_prior_output_boundaries_are_exact(self) -> None:
        self.assertEqual(sha256_file(PARENT_CONFIG), PARENT_CONFIG_SHA256)
        expected = {
            "README.md": "9ddcf6eb8374a76c053622d3e16b772246f4c9aea17bc2578fe43f73762e3e90",
            "failure.json": "3fe5c7fd6ea082501627fcbe52d4ca2751dc00f6f9a2270a9fdf20987097e5b7",
        }
        for name, digest in expected.items():
            self.assertEqual(sha256_file(ROOT / FAILURE_DIRECTORY / name), digest)
        for relative in (
            V1_CANONICAL_OUTPUT_DIRECTORY,
            V2_CANONICAL_OUTPUT_DIRECTORY,
        ):
            output = ROOT / relative
            self.assertFalse(output.exists())
            self.assertFalse(
                output.with_name(f".{output.name}.staging").exists()
            )

    def test_output_and_attempt_claim_boundaries_are_exact(self) -> None:
        output = self.config_data["output_contract"]
        self.assertTrue(
            output["invalid_v1_and_v2_output_and_staging_must_remain_absent"]
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
        self.assertFalse(output["retry_allowed"])
        self.assertFalse(output["resume_allowed"])

    def test_formal_source_diff_uses_b1d0755_and_exact_inventory(self) -> None:
        expected = self.contract.repair_identity[
            "formal_source_changed_paths_exact"
        ]
        with patch(
            "causalcache.restoration_v2_2_policy_vision_v3_contract."
            "_require_commit"
        ), patch(
            "causalcache.restoration_v2_2_policy_vision_v3_contract."
            "_require_ancestor"
        ) as ancestor, patch(
            "causalcache.restoration_v2_2_policy_vision_v3_contract._git_text",
            return_value="\n".join(reversed(expected)),
        ):
            self.contract.validate_formal_source_diff("a" * 40)
        ancestor.assert_called_once_with(
            ROOT.resolve(),
            FAILURE_GIT_COMMIT,
            "a" * 40,
        )
        with patch(
            "causalcache.restoration_v2_2_policy_vision_v3_contract."
            "_require_commit"
        ), patch(
            "causalcache.restoration_v2_2_policy_vision_v3_contract."
            "_require_ancestor"
        ), patch(
            "causalcache.restoration_v2_2_policy_vision_v3_contract._git_text",
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
        type_scope["repair_scope"]["semantic_repair"][
            "acceptance_predicate"
        ] = "isinstance(value, object)"
        mutations.append(type_scope)
        geometry = copy.deepcopy(self.config_data)
        geometry["repair_scope"]["semantic_repair"]["exact_normalized_values"][
            "longest_edge"
        ] = 2621439
        mutations.append(geometry)
        science = copy.deepcopy(self.config_data)
        science["repair_scope"]["statistics_contract_changed"] = True
        mutations.append(science)
        baseline = copy.deepcopy(self.config_data)
        baseline["repair_scope"]["implementation_boundary"][
            "source_diff_baseline_git_commit"
        ] = "0" * 40
        mutations.append(baseline)
        output = copy.deepcopy(self.config_data)
        output["output_contract"]["canonical_result_directory"] = (
            V2_CANONICAL_OUTPUT_DIRECTORY
        )
        mutations.append(output)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                _validate_overlay_semantics(mutation)


if __name__ == "__main__":
    unittest.main()
