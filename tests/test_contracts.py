import copy
import json
import unittest
from pathlib import Path

from causalcache.contracts import ExperimentContract


ROOT = Path(__file__).resolve().parents[1]


class ExperimentContractTest(unittest.TestCase):
    def test_phase0_contract_loads(self) -> None:
        contract = ExperimentContract.load(ROOT / "configs" / "phase0_contract.json")
        self.assertEqual(contract.version, "0.3.0")
        self.assertEqual(
            contract.distance.target_canonicalization,
            "ui_element_or_coordinate_bin_or_scroll_direction",
        )
        self.assertEqual(contract.memory.visual_token_budgets, (256, 512, 1024, 2048))
        self.assertEqual(contract.attribution.samples_primary, 16)

    def test_storage_budget_is_rejected(self) -> None:
        with (ROOT / "configs" / "phase0_contract.json").open(encoding="utf-8") as handle:
            data = json.load(handle)
        invalid = copy.deepcopy(data)
        invalid["memory"]["budget_type"] = "persistent_storage"
        with self.assertRaisesRegex(ValueError, "policy_visible_context"):
            ExperimentContract.from_dict(invalid)

    def test_forced_top_b_is_rejected(self) -> None:
        with (ROOT / "configs" / "phase0_contract.json").open(encoding="utf-8") as handle:
            data = json.load(handle)
        invalid = copy.deepcopy(data)
        invalid["memory"]["selection_rule"] = "forced_top_b"
        with self.assertRaisesRegex(ValueError, "fewer than"):
            ExperimentContract.from_dict(invalid)


if __name__ == "__main__":
    unittest.main()
