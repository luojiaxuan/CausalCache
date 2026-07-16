from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from causalcache.data.guiodyssey_restoration_v2 import canonical_json_bytes
from causalcache.spatial_reference_audit_v1 import (
    ParentMismatch,
    action_from_mapping,
    coordinate_delta,
    first_divergent_index,
    load_and_validate_config,
    load_json_object,
    profile_operation_counts,
    retokenize_parent_mismatch,
    validate_audit_config,
    validate_exposure_ledger,
    validate_repository_inputs,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "code/configs/spatial_reference_audit_v1.json"
FIXTURE_PATH = ROOT / "data/fixtures/spatial_reference_audit_v1_parent_mismatches.json"
EXPOSURE_PATH = ROOT / "data/manifests/spatial_reference_audit_v1_exposure.json"


class _Tokenizer:
    def __init__(self, rows: dict[str, list[int]]) -> None:
        self.rows = rows

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("audit retokenization must disable special tokens")
        return list(self.rows[text])


def _mismatch(
    *,
    actions: tuple[dict[str, object], dict[str, object]],
    bridges: tuple[dict[str, object], dict[str, object]],
    texts: tuple[str, str] = ("left", "right"),
    token_rows: tuple[list[int], list[int]] = ([1, 2, 3], [1, 4, 3]),
) -> ParentMismatch:
    return ParentMismatch(
        index=1,
        role="v2_label_train",
        state_id="state",
        screen_dimensions={"width": 100, "height": 200},
        actions=actions,
        output_texts=texts,
        generated_token_ids_sha256=tuple(
            hashlib.sha256(canonical_json_bytes(row)).hexdigest() for row in token_rows
        ),
        androidworld_bridges=bridges,
        state_member_sha256="0" * 64,
    )


class SpatialReferenceAuditContractTest(unittest.TestCase):
    def test_frozen_config_and_repository_hashes_validate(self) -> None:
        config = load_and_validate_config(CONFIG_PATH)
        validate_repository_inputs(ROOT, config)
        self.assertEqual(
            [profile.profile_id for profile in validate_audit_config(config)],
            [
                "bf16_auto",
                "bf16_eager_control",
                "fp32_eager_control",
            ],
        )

    def test_fixture_has_exact_parent_denominator(self) -> None:
        fixture = load_json_object(FIXTURE_PATH)
        records = fixture["mismatches"]
        self.assertEqual(
            [record["index"] for record in records],
            [1, 10, 16, 17, 20, 23, 24, 25, 26, 28, 30, 35, 39],
        )
        self.assertEqual(
            sum(record["role"] == "v2_label_train" for record in records),
            10,
        )
        self.assertEqual(
            sum(record["actions"][0]["action"] == "click" for record in records),
            12,
        )

    def test_profile_operation_counts_close_exact_total(self) -> None:
        config = load_and_validate_config(CONFIG_PATH)
        profiles = validate_audit_config(config)
        totals = {
            key: sum(profile_operation_counts(profile)[key] for profile in profiles)
            for key in config["operation_limits"]
        }
        self.assertEqual(totals, config["operation_limits"])

    def test_config_rejects_scope_or_decision_relaxation(self) -> None:
        config = load_and_validate_config(CONFIG_PATH)
        relaxed = copy.deepcopy(config)
        relaxed["scope"]["confirm_policy_output_access_allowed"] = True
        with self.assertRaisesRegex(ValueError, "scope"):
            validate_audit_config(relaxed)
        thresholded = copy.deepcopy(config)
        thresholded["decision_rule"]["margin_threshold_used_for_pass_fail"] = True
        with self.assertRaisesRegex(ValueError, "decision"):
            validate_audit_config(thresholded)
        runtime_drift = copy.deepcopy(config)
        runtime_drift["runtime_constraints"]["torch_version"] = "2.11.1"
        with self.assertRaisesRegex(ValueError, "runtime"):
            validate_audit_config(runtime_drift)

    def test_exposure_ledger_rejects_confirm_or_schedule_mutation(self) -> None:
        config = load_and_validate_config(CONFIG_PATH)
        exposure = load_json_object(EXPOSURE_PATH)
        validate_exposure_ledger(exposure, config=config)
        opened = copy.deepcopy(exposure)
        opened["role_state"]["v2_confirm_primary"]["known_policy_output"] = True
        with self.assertRaisesRegex(ValueError, "confirm"):
            validate_exposure_ledger(opened, config=config)
        expanded = copy.deepcopy(exposure)
        expanded["planned_audit_event"]["planned_generation_call_count"] = 61
        with self.assertRaisesRegex(ValueError, "planned"):
            validate_exposure_ledger(expanded, config=config)
        rewritten = copy.deepcopy(exposure)
        rewritten["semantics"]["v2_1_no_go_is_immutable"] = False
        with self.assertRaisesRegex(ValueError, "semantics"):
            validate_exposure_ledger(rewritten, config=config)

    def test_action_mapping_remains_exact(self) -> None:
        click = action_from_mapping({"action": "click", "coordinate": [12, 34]})
        swipe = action_from_mapping(
            {
                "action": "swipe",
                "coordinate": [1, 2],
                "coordinate2": [3, 4],
            }
        )
        self.assertEqual(click.coordinate, (12, 34))
        self.assertEqual(swipe.coordinate2, (3, 4))
        with self.assertRaisesRegex(ValueError, "schema"):
            action_from_mapping(
                {"action": "click", "coordinate": [12, 34], "coordinate2": [3, 4]}
            )

    def test_click_and_swipe_delta_use_recorded_androidworld_bridge(self) -> None:
        click = _mismatch(
            actions=(
                {"action": "click", "coordinate": [390, 820]},
                {"action": "click", "coordinate": [360, 816]},
            ),
            bridges=(
                {"action_type": "click", "x": 421, "y": 1969},
                {"action_type": "click", "x": 389, "y": 1960},
            ),
        )
        self.assertEqual(
            coordinate_delta(click)["pixel_delta"]["coordinate"],
            [-32, -9],
        )
        swipe = _mismatch(
            actions=(
                {
                    "action": "swipe",
                    "coordinate": [529, 837],
                    "coordinate2": [511, 334],
                },
                {
                    "action": "swipe",
                    "coordinate": [529, 862],
                    "coordinate2": [511, 373],
                },
            ),
            bridges=(
                {"action_type": "swipe", "direction": [381, 1072, 368, 428]},
                {"action_type": "swipe", "direction": [381, 1104, 368, 478]},
            ),
        )
        self.assertEqual(
            coordinate_delta(swipe)["pixel_delta"],
            {"coordinate": [0, 32], "coordinate2": [0, 50]},
        )

    def test_retokenization_checks_parent_hash_before_divergence(self) -> None:
        mismatch = _mismatch(
            actions=(
                {"action": "click", "coordinate": [1, 2]},
                {"action": "click", "coordinate": [3, 4]},
            ),
            bridges=(
                {"action_type": "click", "x": 1, "y": 2},
                {"action_type": "click", "x": 3, "y": 4},
            ),
        )
        result = retokenize_parent_mismatch(
            mismatch,
            _Tokenizer({"left": [1, 2, 3], "right": [1, 4, 3]}),
        )
        self.assertEqual(result["first_divergent_token_index"], 1)
        self.assertEqual(result["first_divergent_token_ids"], [2, 4])
        with self.assertRaisesRegex(ValueError, "hash"):
            retokenize_parent_mismatch(
                mismatch,
                _Tokenizer({"left": [9], "right": [1, 4, 3]}),
            )

    def test_first_divergence_handles_prefix_and_identity(self) -> None:
        self.assertEqual(first_divergent_index([1, 2], [1, 3]), 1)
        self.assertEqual(first_divergent_index([1], [1, 2]), 1)
        self.assertIsNone(first_divergent_index([1, 2], [1, 2]))


if __name__ == "__main__":
    unittest.main()
