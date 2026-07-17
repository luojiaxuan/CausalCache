import copy
import json
import tempfile
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_label_expansion import build_expansion_manifest
from causalcache.restoration_v2_2_label_expansion_exposure import (
    CONFIG_SHA256,
    EXPANSION_IDS_SHA256,
    PARENT_EXPOSURE_SHA256,
    PARENT_SHA256,
    PRIOR_OUTPUT_EXPOSURE_SHA256,
    SOURCE_MANIFEST_SHA256,
    build_exposure_ledger,
    load_json_object,
    pretty_json_bytes,
    sha256_bytes,
    validate_exposure_ledger,
)
from scripts.materialize_restoration_v2_2_label_expansion_exposure import (
    _exclusive_write,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/configs/causalcache_restoration_v2_2_label_expansion_v1.json"
PARENT = ROOT / "data/manifests/restoration_v2_selection.json"
SOURCE = ROOT / "data/manifests/independent_reference_gate_v1_source_files.json"
PARENT_EXPOSURE = ROOT / "data/manifests/restoration_v2_exposure.json"
PRIOR_OUTPUT_EXPOSURE = (
    ROOT / "data/manifests/spatial_reference_audit_v1_exposure.json"
)


def _generator() -> dict:
    return {
        "git_revision": "a" * 40,
        "module_path": (
            "code/causalcache/restoration_v2_2_label_expansion_exposure.py"
        ),
        "module_sha256": "b" * 64,
        "cli_path": (
            "code/scripts/materialize_restoration_v2_2_label_expansion_exposure.py"
        ),
        "cli_sha256": "c" * 64,
        "validator_path": (
            "code/scripts/validate_restoration_v2_2_label_expansion_exposure.py"
        ),
        "validator_sha256": "d" * 64,
    }


def _split_generator() -> dict:
    return {
        "git_revision": "e" * 40,
        "module_path": "code/causalcache/restoration_v2_2_label_expansion.py",
        "module_sha256": "1" * 64,
        "cli_path": "code/scripts/materialize_restoration_v2_2_label_expansion.py",
        "cli_sha256": "2" * 64,
        "validator_path": "code/scripts/validate_restoration_v2_2_label_expansion.py",
        "validator_sha256": "3" * 64,
    }


def _inputs() -> dict:
    paths = {
        "config": CONFIG,
        "parent": PARENT,
        "source_manifest": SOURCE,
        "parent_exposure": PARENT_EXPOSURE,
        "prior_output_exposure": PRIOR_OUTPUT_EXPOSURE,
    }
    loaded = {}
    for name, path in paths.items():
        payload, value = load_json_object(path)
        loaded[name] = value
        loaded[f"{name}_sha256"] = sha256_bytes(payload)
    expected = {
        "config_sha256": CONFIG_SHA256,
        "parent_sha256": PARENT_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "parent_exposure_sha256": PARENT_EXPOSURE_SHA256,
        "prior_output_exposure_sha256": PRIOR_OUTPUT_EXPOSURE_SHA256,
    }
    for key, digest in expected.items():
        if loaded[key] != digest:
            raise AssertionError(f"test input identity drifted: {key}")
    return loaded


def _build(
    *,
    expansion_selection: dict | None = None,
    expansion_selection_sha256: str | None = None,
) -> tuple[dict, dict]:
    inputs = _inputs()
    ledger = build_exposure_ledger(
        **inputs,
        expansion_selection=expansion_selection,
        expansion_selection_sha256=expansion_selection_sha256,
        generator=_generator(),
    )
    return ledger, inputs


class LabelExpansionExposureTest(unittest.TestCase):
    def test_exact_cohorts_and_all_required_intersections(self) -> None:
        ledger, inputs = _build()
        inventory = ledger["cohort_inventory"]
        self.assertEqual(
            inventory["prior_output_exposed"]["union"]["trajectory_count"], 23
        )
        self.assertEqual(inventory["sealed_confirm"]["trajectory_count"], 20)
        self.assertEqual(inventory["label_expansion"]["trajectory_count"], 64)
        self.assertEqual(
            inventory["label_expansion"]["source_ids_sha256"],
            EXPANSION_IDS_SHA256,
        )
        self.assertEqual(inventory["all_reserved_union"]["trajectory_count"], 43)
        comparisons = ledger["overlap_proof"]["comparisons"]
        self.assertEqual(
            [record["reserved_cohort"] for record in comparisons],
            [
                "v1_reference_contract_audit_only",
                "v2_label_train",
                "v2_development",
                "prior_output_exposed_union",
                "sealed_confirm",
                "all_reserved_union",
            ],
        )
        self.assertTrue(all(record["intersection_count"] == 0 for record in comparisons))
        self.assertFalse(ledger["selection_freeze"]["top_up_allowed"])
        self.assertFalse(ledger["selection_freeze"]["top_up_performed"])
        self.assertFalse(
            ledger["selection_freeze"]["policy_output_accessed_for_membership"]
        )
        self.assertFalse(
            ledger["selection_freeze"]["restoration_output_accessed_for_membership"]
        )
        validate_exposure_ledger(
            ledger,
            **inputs,
            expansion_selection=None,
            expansion_selection_sha256=None,
        )

    def test_ledger_withholds_all_parent_and_expansion_source_ids(self) -> None:
        ledger, inputs = _build()
        payload = pretty_json_bytes(ledger)
        parent_ids = [
            trajectory["source_id"]
            for role in inputs["parent"]["roles"].values()
            for trajectory in role["trajectories"]
        ]
        expansion_ids = [
            source_id
            for split in inputs["config"]["selection"].values()
            if isinstance(split, dict) and "source_ids" in split
            for source_id in split["source_ids"]
        ]
        for source_id in set(parent_ids + expansion_ids):
            self.assertNotIn(source_id.encode("utf-8"), payload)
        for forbidden in (
            b'"source_ids"',
            b'"instruction"',
            b'"action"',
            b'"image_sha256"',
            b'"ocr_text"',
        ):
            self.assertNotIn(forbidden, payload)

    def test_optional_structural_manifest_is_validated_and_bound(self) -> None:
        inputs = _inputs()
        selection = build_expansion_manifest(
            config=inputs["config"],
            config_sha256=inputs["config_sha256"],
            parent=inputs["parent"],
            parent_sha256=inputs["parent_sha256"],
            generator=_split_generator(),
        )
        selection_sha = sha256_bytes(pretty_json_bytes(selection))
        ledger, _ = _build(
            expansion_selection=selection,
            expansion_selection_sha256=selection_sha,
        )
        binding = ledger["inputs"]["expansion_selection_manifest"]
        self.assertTrue(binding["available_at_freeze"])
        self.assertEqual(binding["sha256"], selection_sha)
        self.assertEqual(binding["generator_git_revision"], "e" * 40)
        validate_exposure_ledger(
            ledger,
            **inputs,
            expansion_selection=selection,
            expansion_selection_sha256=selection_sha,
        )
        with self.assertRaisesRegex(ValueError, "differs from the frozen contract"):
            validate_exposure_ledger(
                ledger,
                **inputs,
                expansion_selection=None,
                expansion_selection_sha256=None,
            )

    def test_fail_closed_on_overlap_freeze_or_evidence_mutation(self) -> None:
        ledger, inputs = _build()
        invalid = copy.deepcopy(ledger)
        invalid["overlap_proof"]["comparisons"][0]["intersection_count"] = 1
        with self.assertRaisesRegex(ValueError, "differs from the frozen contract"):
            validate_exposure_ledger(
                invalid,
                **inputs,
                expansion_selection=None,
                expansion_selection_sha256=None,
            )
        invalid = copy.deepcopy(ledger)
        invalid["selection_freeze"]["top_up_performed"] = True
        with self.assertRaisesRegex(ValueError, "differs from the frozen contract"):
            validate_exposure_ledger(
                invalid,
                **inputs,
                expansion_selection=None,
                expansion_selection_sha256=None,
            )
        changed_inputs = copy.deepcopy(inputs)
        changed_inputs["prior_output_exposure"]["role_state"]["v2_label_train"][
            "known_policy_output"
        ] = False
        with self.assertRaisesRegex(ValueError, "role state"):
            validate_exposure_ledger(
                ledger,
                **changed_inputs,
                expansion_selection=None,
                expansion_selection_sha256=None,
            )

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_json_object(duplicate)
            nonfinite = Path(directory) / "nonfinite.json"
            nonfinite.write_text('{"a": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                load_json_object(nonfinite)

    def test_exclusive_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exposure.json"
            _exclusive_write(output, b"first")
            self.assertEqual(output.read_bytes(), b"first")
            with self.assertRaises(FileExistsError):
                _exclusive_write(output, b"second")


if __name__ == "__main__":
    unittest.main()
