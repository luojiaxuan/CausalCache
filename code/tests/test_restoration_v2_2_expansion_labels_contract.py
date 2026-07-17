from __future__ import annotations

import copy
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_expansion_labels_contract import (
    ATTEMPT_ID,
    CANONICAL_CONFIG_PATH,
    CANONICAL_HF_PATH,
    CANONICAL_HF_REPO,
    CANONICAL_HF_TAG,
    DERIVED_HF_REVISION,
    EXPECTED_ATTRIBUTIONS,
    EXPECTED_DEPLOYMENT_EDGES,
    EXPECTED_DISTANCE_ROWS,
    EXPECTED_FULL_EDGES,
    EXPECTED_INTERACTIONS,
    EXPECTED_ORACLES,
    EXPECTED_STATES,
    EXPECTED_TEACHER_FORWARDS,
    FROZEN_CONFIG_SHA256,
    PROTOCOL_ID,
    SUBSTRATE_ARTIFACT_PATH,
    SUBSTRATE_HF_REVISION,
    build_contract,
    load_json_object,
    pretty_json_bytes,
    sha256_file,
    validate_contract_data,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


class ExpansionExactLabelsContractTest(unittest.TestCase):
    def _data(self) -> dict:
        payload, data = load_json_object(CONFIG)
        self.assertEqual(payload, pretty_json_bytes(data))
        return data

    def test_frozen_config_matches_deterministic_contract_and_exact_counts(self) -> None:
        self.assertEqual(sha256_file(CONFIG), FROZEN_CONFIG_SHA256)
        data = self._data()
        self.assertEqual(data, build_contract(repository_root=ROOT))
        validation = validate_contract_data(
            data,
            repository_root=ROOT,
            require_git_blobs=False,
        )
        self.assertEqual(validation["state_count"], EXPECTED_STATES)
        self.assertEqual(
            validation["raw_distance_row_count"], EXPECTED_DISTANCE_ROWS
        )
        self.assertEqual(
            validation["deployment_conditional_label_count"],
            EXPECTED_DEPLOYMENT_EDGES,
        )
        self.assertEqual(validation["full_hypercube_edge_count"], EXPECTED_FULL_EDGES)
        self.assertEqual(
            validation["full_pair_interaction_count"], EXPECTED_INTERACTIONS
        )
        self.assertEqual(
            validation["exact_permutation_attribution_count"],
            EXPECTED_ATTRIBUTIONS,
        )
        self.assertEqual(
            validation["primary_exact_subset_oracle_count"], EXPECTED_ORACLES
        )
        self.assertEqual(
            validation["teacher_forward_count"], EXPECTED_TEACHER_FORWARDS
        )
        self.assertEqual(validation["worker_state_counts"], {"even": 96, "odd": 96})

    def test_substrate_and_derived_revisions_are_immutably_bound(self) -> None:
        data = self._data()
        substrate = data["immutable_inputs"]["substrate_artifact"]
        derived = data["immutable_inputs"]["derived_artifact"]
        self.assertEqual(substrate["path"], SUBSTRATE_ARTIFACT_PATH)
        self.assertEqual(substrate["required_hf_revision"], SUBSTRATE_HF_REVISION)
        self.assertEqual(derived["immutable_revision"], DERIVED_HF_REVISION)
        validate_contract_data(
            data,
            repository_root=ROOT,
            require_git_blobs=True,
        )

        for section, key in (
            ("substrate_artifact", "required_hf_revision"),
            ("derived_artifact", "immutable_revision"),
        ):
            mutated = copy.deepcopy(data)
            mutated["immutable_inputs"][section][key] = "0" * 40
            with self.subTest(section=section), self.assertRaises(ValueError):
                validate_contract_data(
                    mutated,
                    repository_root=ROOT,
                    require_git_blobs=False,
                )

    def test_source_inventory_inherits_67_and_reserves_six_new_sources(self) -> None:
        source_lock = self._data()["scientific_source_lock"]
        sources = source_lock["source_files"]
        self.assertEqual(len(sources), 69)
        self.assertEqual(
            [record["path"] for record in sources],
            sorted(record["path"] for record in sources),
        )
        self.assertTrue(
            all(set(record) == {"path", "sha256", "size_bytes"} for record in sources)
        )
        self.assertEqual(
            source_lock["reserved_execution_source_paths"],
            [
                "code/causalcache/data/restoration_v2_2_expansion_label_parent.py",
                "code/causalcache/data/restoration_v2_2_expansion_label_inputs.py",
                "code/causalcache/restoration_v2_2_expansion_labels_contract.py",
                "code/causalcache/restoration_v2_2_expansion_labels_artifact.py",
                "code/scripts/run_restoration_v2_2_expansion_labels.py",
                "code/scripts/manage_restoration_v2_2_expansion_labels_artifact.py",
            ],
        )
        source_paths = {record["path"] for record in sources}
        self.assertNotIn(
            "code/causalcache/data/restoration_v2_2_expansion_label_parent.py",
            source_paths,
        )
        self.assertNotIn(
            "code/causalcache/data/restoration_v2_2_expansion_label_inputs.py",
            source_paths,
        )

    def test_every_scientific_denominator_drift_fails_closed(self) -> None:
        original = self._data()
        mutations = (
            lambda value: value["data_projection"].__setitem__(
                "fixed_state_denominator", 191
            ),
            lambda value: value["enumeration"]["raw_distance_row_counts"].__setitem__(
                "total", 1791
            ),
            lambda value: value["enumeration"][
                "deployment_conditional_edge_counts"
            ].__setitem__("total", 1855),
            lambda value: value["enumeration"]["full_hypercube_edge_counts"].__setitem__(
                "total", 3071
            ),
            lambda value: value["enumeration"]["full_pair_interaction_counts"].__setitem__(
                "total", 1983
            ),
            lambda value: value["enumeration"][
                "exact_permutation_attribution_counts"
            ].__setitem__("total", 575),
            lambda value: value["enumeration"][
                "primary_exact_subset_oracle_counts"
            ].__setitem__("total", 191),
            lambda value: value["operation_schedule"].__setitem__(
                "total_teacher_forward_call_count", 1983
            ),
        )
        for index, mutate in enumerate(mutations):
            data = copy.deepcopy(original)
            mutate(data)
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_contract_data(
                    data,
                    repository_root=ROOT,
                    require_git_blobs=False,
                )

    def test_legacy_tie_and_signed_negative_marginal_semantics_are_frozen(self) -> None:
        reduction = self._data()["reduction"]
        estimand = self._data()["estimand"]
        self.assertEqual(reduction["exact_oracle_tie_epsilon"], 0.0)
        self.assertEqual(
            reduction["exact_oracle_tie_break"],
            [
                "lower_distance",
                "lower_cardinality",
                "lexicographically_lower_sorted_event_step_ids",
            ],
        )
        self.assertFalse(reduction["negative_utility_or_marginal_clipped"])
        self.assertTrue(
            reduction["negative_conditional_marginals_are_retained_as_signed_values"]
        )
        self.assertFalse(estimand["negative_utility_or_marginal_clipped"])

    def test_source_only_validation_never_authorizes_gpu_or_later_phases(self) -> None:
        data = self._data()
        validation = validate_contract_data(
            data,
            repository_root=ROOT,
            require_git_blobs=False,
        )
        self.assertFalse(
            validation["policy_or_gpu_execution_authorized_by_this_validator"]
        )
        self.assertFalse(validation["label_execution_authorized_by_this_validator"])
        self.assertFalse(validation["confirm_access_authorized_by_this_validator"])
        self.assertFalse(
            data["freeze"]["source_only_validator_authorizes_policy_or_gpu_execution"]
        )
        self.assertFalse(data["execution"]["source_only_contract_authorizes_execution"])
        self.assertTrue(
            data["promotion"]["gate_training_matched_nll_closed_loop_and_confirm_remain_locked"]
        )

        for key in data["prohibited_work"]:
            mutated = copy.deepcopy(data)
            mutated["prohibited_work"][key] = (
                True if key.endswith("allowed") else 1
            )
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_contract_data(
                    mutated,
                    repository_root=ROOT,
                    require_git_blobs=False,
                )

    def test_protocol_output_and_hf_identity_are_new_and_versioned(self) -> None:
        data = self._data()
        self.assertEqual(data["protocol_id"], PROTOCOL_ID)
        self.assertEqual(data["execution"]["attempt_id"], ATTEMPT_ID)
        self.assertIn(ATTEMPT_ID, data["execution"]["canonical_persistent_output_dir"])
        destination = data["artifact_destination"]
        self.assertEqual(destination["repo"], CANONICAL_HF_REPO)
        self.assertEqual(destination["tag"], CANONICAL_HF_TAG)
        self.assertEqual(destination["raw_path"], CANONICAL_HF_PATH)
        self.assertIsNone(destination["immutable_revision"])
        self.assertNotEqual(
            destination["repo"],
            data["immutable_inputs"]["substrate_artifact"]["required_hf_repo"],
        )


if __name__ == "__main__":
    unittest.main()
