from __future__ import annotations

import copy
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_label_contract import (
    CANONICAL_ATTEMPT_ID,
    CANONICAL_CONFIG_PATH,
    FROZEN_CONFIG_SHA256,
    RestorationV22LabelContract,
    V2_REPAIR_ATTEMPT_ID,
    V2_REPAIR_CONFIG_PATH,
    V2_REPAIR_FROZEN_CONFIG_SHA256,
    V2_REPAIR_SOURCE_PARENT_GIT_COMMIT,
    label_attempt_profile_for_config_path,
    load_strict_json_object,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/configs/causalcache_restoration_v2_2_labels.json"
REPAIR_CONFIG = ROOT / V2_REPAIR_CONFIG_PATH


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

    def test_v1_identity_is_unchanged_and_v2_repair_is_independent(self) -> None:
        self.assertEqual(CANONICAL_CONFIG_PATH, CONFIG.relative_to(ROOT).as_posix())
        self.assertEqual(
            FROZEN_CONFIG_SHA256,
            "56b29f6879ef14167b20a39d0d61ebd0e698e3bbf0457062b63453180e71cf87",
        )
        self.assertEqual(CANONICAL_ATTEMPT_ID, "restoration-v2-2-eager-labels-v1")
        self.assertNotIn("repair_lineage", load_strict_json_object(CONFIG))

        profile = label_attempt_profile_for_config_path(V2_REPAIR_CONFIG_PATH)
        contract = RestorationV22LabelContract(
            path=REPAIR_CONFIG,
            data=load_strict_json_object(REPAIR_CONFIG),
            sha256=sha256_file(REPAIR_CONFIG),
            profile=profile,
        )
        contract.validate(repository_root=ROOT)
        self.assertEqual(sha256_file(REPAIR_CONFIG), V2_REPAIR_FROZEN_CONFIG_SHA256)
        self.assertEqual(profile.attempt_id, V2_REPAIR_ATTEMPT_ID)
        self.assertEqual(
            profile.source_parent_git_commit,
            V2_REPAIR_SOURCE_PARENT_GIT_COMMIT,
        )
        self.assertEqual(contract.data["repair_lineage"]["superseded_teacher_forward_count"], 0)
        self.assertIn(CANONICAL_CONFIG_PATH, profile.expected_source_paths)
        self.assertIn(V2_REPAIR_CONFIG_PATH, profile.expected_source_paths)
        self.assertIn(
            "data/results/restoration_v2_2_eager_labels_v1_attempt/summary.json",
            profile.expected_source_paths,
        )


if __name__ == "__main__":
    unittest.main()
