import copy
import json
import tempfile
import unittest
from pathlib import Path

from causalcache.restoration_v2_contract import (
    FROZEN_RESTORATION_V2_SHA256,
    RestorationV2Contract,
    validate_restoration_v2_contract,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
CONFIG = ROOT / "configs" / "causalcache_restoration_v2.json"


def _config() -> dict:
    with CONFIG.open(encoding="utf-8") as handle:
        return json.load(handle)


class RestorationV2ContractTest(unittest.TestCase):
    def test_frozen_contract_loads_with_expected_hash(self) -> None:
        contract = RestorationV2Contract.load(CONFIG)
        self.assertEqual(contract.protocol_id, "causalcache_restoration_v2")
        self.assertEqual(contract.source_sha256, FROZEN_RESTORATION_V2_SHA256)

    def test_source_of_truth_docs_pin_the_contract_hash(self) -> None:
        for relative_path in ("README.md", "docs/restoration_v2.md", "docs/progress.md"):
            with self.subTest(path=relative_path):
                text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn(FROZEN_RESTORATION_V2_SHA256, text)

    def test_reference_must_cover_every_distinct_post_state(self) -> None:
        invalid = _config()
        invalid["reference_definition"]["reference_high_fidelity_event_step_ids"] = [2, 3, 4]
        with self.assertRaisesRegex(ValueError, "every non-current distinct post-state"):
            validate_restoration_v2_contract(invalid)

    def test_latest_current_equivalent_event_cannot_be_candidate(self) -> None:
        invalid = _config()
        invalid["high_fidelity_event"]["latest_event_whose_post_state_is_current_is_not_a_candidate"] = False
        with self.assertRaisesRegex(ValueError, "latest_event"):
            validate_restoration_v2_contract(invalid)

    def test_high_fidelity_cannot_add_action_text(self) -> None:
        invalid = _config()
        invalid["high_fidelity_event"]["include_additional_action_text"] = True
        with self.assertRaisesRegex(ValueError, "include_additional_action_text"):
            validate_restoration_v2_contract(invalid)

    def test_removed_action_cannot_reenter_prompt_inventory(self) -> None:
        invalid = _config()
        invalid["action_contract"]["canonical_prompt_actions"].append("key")
        with self.assertRaisesRegex(ValueError, "canonical_prompt_actions"):
            validate_restoration_v2_contract(invalid)

    def test_action_argument_contract_cannot_drift(self) -> None:
        invalid = _config()
        invalid["action_contract"]["parameter_contract"]["wait"] = ["time"]
        with self.assertRaisesRegex(ValueError, "parameter_contract"):
            validate_restoration_v2_contract(invalid)

    def test_coordinate_grid_is_zero_through_999(self) -> None:
        invalid = _config()
        invalid["action_contract"]["coordinate_contract"]["normalized_maximum"] = 1000
        with self.assertRaisesRegex(ValueError, "normalized_maximum"):
            validate_restoration_v2_contract(invalid)

    def test_historical_data_roles_must_be_disjoint(self) -> None:
        invalid = _config()
        invalid["data"]["roles"]["v2_development"]["source_ids"][0] = invalid["data"]["roles"]["v2_label_train"]["source_ids"][0]
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            validate_restoration_v2_contract(invalid)

    def test_confirm_states_cannot_be_filtered_after_output(self) -> None:
        invalid = _config()
        invalid["data"]["roles"]["v2_confirm_primary"][
            "filter_by_parse_stability_quality_or_sensitivity_allowed"
        ] = True
        with self.assertRaisesRegex(ValueError, "filter_by_parse"):
            validate_restoration_v2_contract(invalid)

    def test_confirm_permutation_count_has_one_source_of_truth(self) -> None:
        invalid = _config()
        invalid["attribution_compute"]["confirmatory_permutation_samples"] = 8
        with self.assertRaisesRegex(ValueError, "confirmatory_permutation_samples"):
            validate_restoration_v2_contract(invalid)

    def test_gate_threshold_and_seed_are_exactly_frozen(self) -> None:
        threshold_drift = _config()
        threshold_drift["restoration_gate"]["minimum_mean_oracle_recovery"] = 0.29
        with self.assertRaisesRegex(ValueError, "minimum_mean_oracle_recovery"):
            validate_restoration_v2_contract(threshold_drift)

        seed_drift = _config()
        seed_drift["restoration_gate"]["confirmatory_seed"] += 1
        with self.assertRaisesRegex(ValueError, "confirmatory_seed"):
            validate_restoration_v2_contract(seed_drift)

    def test_baseline_set_and_formula_are_exactly_frozen(self) -> None:
        invalid = _config()
        invalid["restoration_gate"]["baselines"].pop()
        with self.assertRaisesRegex(ValueError, "restoration_gate.baselines"):
            validate_restoration_v2_contract(invalid)

    def test_default_loader_rejects_any_source_byte_drift(self) -> None:
        changed = CONFIG.read_text(encoding="utf-8").replace(
            '"minimum_mean_oracle_recovery": 0.3',
            '"minimum_mean_oracle_recovery": 0.30',
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "contract.json"
            temporary.write_text(changed, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                RestorationV2Contract.load(temporary)

    def test_hard_no_go_boundary_cannot_be_reinterpreted(self) -> None:
        invalid = _config()
        invalid["restoration_gate"]["hard_no_go_if"] = ["any_failed_threshold"]
        with self.assertRaisesRegex(ValueError, "hard_no_go_if"):
            validate_restoration_v2_contract(invalid)

    def test_all_preoutput_dependencies_are_mandatory(self) -> None:
        invalid = _config()
        invalid["implementation_dependencies_before_any_v2_policy_output"].pop()
        with self.assertRaisesRegex(ValueError, "implementation_dependencies"):
            validate_restoration_v2_contract(invalid)


if __name__ == "__main__":
    unittest.main()
