import copy
import json
import tempfile
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_label_expansion import (
    CONFIG_SHA256,
    DEVELOPMENT_IDS,
    DEVELOPMENT_IDS_SHA256,
    EXPANSION_IDS_SHA256,
    PARENT_SHA256,
    TRAIN_IDS,
    TRAIN_IDS_SHA256,
    build_expansion_manifest,
    canonical_json_bytes,
    load_json_object,
    pretty_json_bytes,
    sha256_bytes,
    validate_expansion_manifest,
)
from scripts.materialize_restoration_v2_2_label_expansion import _exclusive_write


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/configs/causalcache_restoration_v2_2_label_expansion_v1.json"
PARENT = ROOT / "data/manifests/restoration_v2_selection.json"


def _generator() -> dict:
    return {
        "git_revision": "a" * 40,
        "module_path": "code/causalcache/restoration_v2_2_label_expansion.py",
        "module_sha256": "b" * 64,
        "cli_path": "code/scripts/materialize_restoration_v2_2_label_expansion.py",
        "cli_sha256": "c" * 64,
        "validator_path": "code/scripts/validate_restoration_v2_2_label_expansion.py",
        "validator_sha256": "d" * 64,
    }


def _inputs() -> tuple[dict, dict]:
    config_payload, config = load_json_object(CONFIG)
    parent_payload, parent = load_json_object(PARENT)
    if sha256_bytes(config_payload) != CONFIG_SHA256:
        raise AssertionError("test config identity drifted")
    if sha256_bytes(parent_payload) != PARENT_SHA256:
        raise AssertionError("test parent identity drifted")
    return config, parent


def _build() -> dict:
    config, parent = _inputs()
    return build_expansion_manifest(
        config=config,
        config_sha256=CONFIG_SHA256,
        parent=parent,
        parent_sha256=PARENT_SHA256,
        generator=_generator(),
    )


class LabelExpansionSplitTest(unittest.TestCase):
    def test_exact_split_ids_order_and_digests(self) -> None:
        manifest = _build()
        train_ids = tuple(
            record["source_id"]
            for record in manifest["splits"]["gate_train_expansion"][
                "trajectories"
            ]
        )
        development_ids = tuple(
            record["source_id"]
            for record in manifest["splits"]["gate_development_expansion"][
                "trajectories"
            ]
        )
        self.assertEqual(train_ids, TRAIN_IDS)
        self.assertEqual(development_ids, DEVELOPMENT_IDS)
        self.assertEqual(
            sha256_bytes(canonical_json_bytes(list(train_ids))),
            TRAIN_IDS_SHA256,
        )
        self.assertEqual(
            sha256_bytes(canonical_json_bytes(list(development_ids))),
            DEVELOPMENT_IDS_SHA256,
        )
        self.assertEqual(
            sha256_bytes(canonical_json_bytes(list(train_ids + development_ids))),
            EXPANSION_IDS_SHA256,
        )
        self.assertEqual(
            manifest["derivation"]["expansion_global_order_indices"],
            [
                44,
                45,
                46,
                47,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
            ],
        )

    def test_label_geometry_and_workload_are_exact(self) -> None:
        manifest = _build()
        self.assertEqual(
            manifest["expected_workload"]["total"],
            {
                "trajectories": 64,
                "states": 192,
                "full_subset_distance_rows": 1792,
                "deployment_conditional_edges": 1856,
                "teacher_forwards": 1984,
            },
        )
        states = [
            state
            for split in manifest["splits"].values()
            for state in split["states"]
        ]
        self.assertEqual(len(states), 192)
        by_step = {
            step: [state for state in states if state["decision_step_id"] == step]
            for step in (4, 5, 6)
        }
        for step, candidate_count, rows, edges in (
            (4, 2, 4, 4),
            (5, 3, 8, 9),
            (6, 4, 16, 16),
        ):
            self.assertEqual(len(by_step[step]), 64)
            self.assertTrue(
                all(
                    state["candidate_event_count"] == candidate_count
                    and state["full_subset_distance_rows"] == rows
                    and state["deployment_conditional_edges"] == edges
                    and state["teacher_forwards"] == rows + 1
                    for state in by_step[step]
                )
            )

    def test_prior_and_confirm_content_is_not_emitted(self) -> None:
        config, parent = _inputs()
        manifest = _build()
        prior_ids = {
            trajectory["source_id"]
            for role_name in (
                "v1_reference_contract_audit_only",
                "v2_label_train",
                "v2_development",
                "v2_confirm_primary",
            )
            for trajectory in parent["roles"][role_name]["trajectories"]
        }
        strings = set(json.dumps(manifest, sort_keys=True).replace('"', " ").split())
        self.assertFalse(prior_ids.intersection(strings))
        serialized = pretty_json_bytes(manifest)
        for forbidden_key in (
            b"normalized_app_labels",
            b"action_type_counts",
            b"instruction_sha256",
            b"current_observation",
            b"candidate_event_post_states",
            b"current_equivalence_witness",
            b"validated_action_sha256",
        ):
            self.assertNotIn(forbidden_key, serialized)
        validate_expansion_manifest(
            manifest,
            config=config,
            config_sha256=CONFIG_SHA256,
            parent=parent,
            parent_sha256=PARENT_SHA256,
        )

    def test_fail_closed_on_split_or_semantic_mutation(self) -> None:
        config, parent = _inputs()
        manifest = _build()
        invalid = copy.deepcopy(manifest)
        invalid["splits"]["gate_train_expansion"]["trajectories"][0][
            "instruction_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "records differ|schema|semantic"):
            validate_expansion_manifest(
                invalid,
                config=config,
                config_sha256=CONFIG_SHA256,
                parent=parent,
                parent_sha256=PARENT_SHA256,
            )
        invalid = copy.deepcopy(manifest)
        invalid["splits"]["gate_development_expansion"]["trajectories"].reverse()
        with self.assertRaisesRegex(ValueError, "records differ"):
            validate_expansion_manifest(
                invalid,
                config=config,
                config_sha256=CONFIG_SHA256,
                parent=parent,
                parent_sha256=PARENT_SHA256,
            )

    def test_fail_closed_on_parent_and_config_identity_drift(self) -> None:
        config, parent = _inputs()
        with self.assertRaisesRegex(ValueError, "config SHA256"):
            build_expansion_manifest(
                config=config,
                config_sha256="0" * 64,
                parent=parent,
                parent_sha256=PARENT_SHA256,
                generator=_generator(),
            )
        invalid_parent = copy.deepcopy(parent)
        invalid_parent["roles"]["v2_confirm_primary"]["trajectories"].reverse()
        with self.assertRaisesRegex(ValueError, "confirm role"):
            build_expansion_manifest(
                config=config,
                config_sha256=CONFIG_SHA256,
                parent=invalid_parent,
                parent_sha256=PARENT_SHA256,
                generator=_generator(),
            )

    def test_exclusive_output_write_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "split.json"
            _exclusive_write(output, b"first")
            self.assertEqual(output.read_bytes(), b"first")
            with self.assertRaises(FileExistsError):
                _exclusive_write(output, b"second")


if __name__ == "__main__":
    unittest.main()
