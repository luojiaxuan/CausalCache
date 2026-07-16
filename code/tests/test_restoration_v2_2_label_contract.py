from __future__ import annotations

import copy
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_label_contract import (
    FROZEN_CONFIG_SHA256,
    RestorationV22LabelContract,
    load_strict_json_object,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/configs/causalcache_restoration_v2_2_labels.json"


class RestorationV22LabelContractTest(unittest.TestCase):
    def _contract(self, data=None) -> RestorationV22LabelContract:
        return RestorationV22LabelContract(
            path=CONFIG,
            data=load_strict_json_object(CONFIG) if data is None else data,
            sha256=sha256_file(CONFIG),
        )

    def test_source_contract_has_frozen_identity_and_exact_denominators(self) -> None:
        self.assertEqual(sha256_file(CONFIG), FROZEN_CONFIG_SHA256)
        contract = self._contract()
        contract.validate(repository_root=ROOT)
        self.assertEqual(
            contract.data["enumeration"]["raw_distance_row_counts"]["total"],
            420,
        )
        self.assertEqual(
            contract.data["enumeration"]["deployment_conditional_edge_counts"][
                "total"
            ],
            435,
        )
        self.assertEqual(
            contract.data["operation_schedule"]["total_teacher_forward_call_count"],
            465,
        )

    def test_distance_edge_or_operation_count_drift_fails_closed(self) -> None:
        original = load_strict_json_object(CONFIG)
        mutations = (
            ("raw rows", lambda value: value["enumeration"]["raw_distance_row_counts"].__setitem__("total", 419)),
            ("edges", lambda value: value["enumeration"]["deployment_conditional_edge_counts"].__setitem__("total", 434)),
            ("forwards", lambda value: value["operation_schedule"].__setitem__("total_teacher_forward_call_count", 464)),
        )
        for label, mutate in mutations:
            data = copy.deepcopy(original)
            mutate(data)
            with self.subTest(label=label), self.assertRaises(ValueError):
                self._contract(data).validate(repository_root=ROOT)

    def test_confirm_generation_and_gate_training_remain_forbidden(self) -> None:
        original = load_strict_json_object(CONFIG)
        for key in (
            "maximum_confirm_generation_count",
            "maximum_gate_training_example_count",
            "maximum_matched_nll_evaluation_count",
            "maximum_closed_loop_episode_count",
        ):
            data = copy.deepcopy(original)
            data["prohibited_work"][key] = 1
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError,
                "prohibited",
            ):
                self._contract(data).validate(repository_root=ROOT)


if __name__ == "__main__":
    unittest.main()
