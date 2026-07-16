from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from causalcache.restoration_v2_2_ocr_rgb import (
    SelectedImageFeature,
    score_ocr_rgb_state,
    select_jsonl_records_by_identity,
)
from causalcache.restoration_v2_2_ocr_rgb_contract import (
    PROTOCOL_ID as V1_PROTOCOL_ID,
    canonical_json_bytes,
)
from causalcache.restoration_v2_2_ocr_rgb_contract_v2 import (
    CANONICAL_CONFIG_PATH,
    FAILURE_DIRECTORY,
    FROZEN_CONFIG_SHA256,
    OCR_IDENTITY_OCCURRENCES_PER_LINE,
    PASS_STATUS,
    PROTOCOL_ID,
    TRAJECTORY_IDENTITY_OCCURRENCES_PER_LINE,
    V1_CANONICAL_RESULT_DIRECTORY,
    V2_CANONICAL_RESULT_DIRECTORY,
    RestorationV22OcrRgbIdentityRepairContract,
    sha256_file,
    validate_contract,
)
from scripts import run_restoration_v2_2_ocr_rgb_baseline_v2 as runner


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


def _write_lines(*lines: bytes) -> Path:
    temporary = tempfile.NamedTemporaryFile(delete=False)
    temporary.write(b"".join(line + b"\n" for line in lines))
    temporary.close()
    return Path(temporary.name)


def _feature(name: str, tokens: tuple[str, ...], rgb: bytes) -> SelectedImageFeature:
    return SelectedImageFeature(
        image_member_path=f"images/synthetic/{name}.png",
        image_sha256=name * 64,
        canonical_ocr_record_sha256=name.upper() * 64,
        full_spatial_tokens=tokens,
        resized_rgb_bytes=rgb,
    )


class RestorationV22OcrRgbIdentityRepairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = RestorationV22OcrRgbIdentityRepairContract.load(
            CONFIG,
            repository_root=ROOT,
        )

    def test_contract_binds_parent_failure_and_distinct_output(self) -> None:
        self.assertEqual(sha256_file(CONFIG), FROZEN_CONFIG_SHA256)
        result = validate_contract(
            CONFIG,
            repository_root=ROOT,
        )
        self.assertEqual(result["status"], PASS_STATUS)
        self.assertEqual(result["protocol_id"], PROTOCOL_ID)
        self.assertEqual(
            result["trajectory_identity_occurrences_per_line"],
            TRAJECTORY_IDENTITY_OCCURRENCES_PER_LINE,
        )
        self.assertEqual(
            result["ocr_identity_occurrences_per_line"],
            OCR_IDENTITY_OCCURRENCES_PER_LINE,
        )
        self.assertFalse(result["scientific_contract_changed"])
        self.assertTrue(result["formal_aggregate_generated"])
        self.assertFalse(result["staging_output_exists"])
        self.assertNotEqual(V1_CANONICAL_RESULT_DIRECTORY, V2_CANONICAL_RESULT_DIRECTORY)
        self.assertFalse((ROOT / V1_CANONICAL_RESULT_DIRECTORY).exists())
        self.assertTrue((ROOT / FAILURE_DIRECTORY / "summary.json").is_file())

    def test_exact_two_equal_identities_parse_selected_and_skip_opaque(self) -> None:
        selected = canonical_json_bytes(
            {
                "selection": {"source_id": "train-1"},
                "source_id": "train-1",
                "value": 7,
            }
        )
        opaque = (
            b'{"selection":{"source_id":"confirm-locked"},'
            b'"source_id":"confirm-locked",not-valid-json}'
        )
        path = _write_lines(selected, opaque)
        self.addCleanup(path.unlink)
        records = select_jsonl_records_by_identity(
            path,
            identity_field="source_id",
            allowed_identities={"train-1"},
            expected_total_line_count=2,
            expected_identity_occurrences_per_line=2,
            label="synthetic trajectories",
        )
        self.assertEqual(records["train-1"]["value"], 7)

    def test_v1_default_still_rejects_repeated_identity(self) -> None:
        line = canonical_json_bytes(
            {
                "selection": {"source_id": "train-1"},
                "source_id": "train-1",
            }
        )
        path = _write_lines(line)
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "invalid identity encoding"):
            select_jsonl_records_by_identity(
                path,
                identity_field="source_id",
                allowed_identities={"train-1"},
                expected_total_line_count=1,
                label="v1 synthetic trajectories",
            )

    def test_count_mismatch_inconsistent_escaped_and_zero_fail_closed(self) -> None:
        cases = (
            canonical_json_bytes({"source_id": "train-1"}),
            (
                b'{"a":{"source_id":"train-1"},'
                b'"b":{"source_id":"train-1"},"source_id":"train-1"}'
            ),
            canonical_json_bytes(
                {
                    "selection": {"source_id": "train-2"},
                    "source_id": "train-1",
                }
            ),
            (
                b'{"selection":{"source_id":"train\\u002d1"},'
                b'"source_id":"train-1"}'
            ),
            (
                b'{"a":{"source_id":"train-1"},'
                b'"b":{"source_id":"train\\u002d1"},'
                b'"source_id":"train-1"}'
            ),
            canonical_json_bytes({"different_field": "train-1"}),
        )
        for index, line in enumerate(cases):
            with self.subTest(index=index):
                path = _write_lines(line)
                try:
                    with self.assertRaises(ValueError):
                        select_jsonl_records_by_identity(
                            path,
                            identity_field="source_id",
                            allowed_identities={"train-1"},
                            expected_total_line_count=1,
                            expected_identity_occurrences_per_line=2,
                            label="invalid synthetic trajectories",
                        )
                finally:
                    path.unlink()

    def test_exact_one_ocr_identity_remains_supported(self) -> None:
        line = canonical_json_bytes(
            {"image_member_path": "images/train-1.png", "tokens": []}
        )
        path = _write_lines(line)
        self.addCleanup(path.unlink)
        records = select_jsonl_records_by_identity(
            path,
            identity_field="image_member_path",
            allowed_identities={"images/train-1.png"},
            expected_total_line_count=1,
            expected_identity_occurrences_per_line=1,
            label="synthetic OCR records",
        )
        self.assertEqual(records["images/train-1.png"]["tokens"], [])

    def test_v2_protocol_parameter_changes_only_row_identity(self) -> None:
        red = bytes((255, 0, 0)) * (256 * 256)
        blue = bytes((0, 0, 255)) * (256 * 256)
        features = {
            1: _feature("a", ("Settings",), red),
            2: _feature("b", ("Settings",), red),
            3: _feature("c", ("Other",), blue),
            4: _feature("d", ("Other",), blue),
        }
        current = _feature("q", ("Settings",), red)
        distances = {
            coalition: 5.0
            for size in range(5)
            for coalition in itertools.combinations((1, 2, 3, 4), size)
        }
        distances[()] = 10.0
        distances[(1, 2)] = 1.0
        methods = {
            method: {
                "selected_coalition": [1, 2],
                "normalized_recovery": 0.9,
            }
            for method in (
                "exact_subset",
                "exact_cardinality_oracle",
                "true_conditional_greedy",
                "budget_conditioned_independent",
                "full_shapley_independent",
                "dynamic_recent",
                "analytic_exact_cardinality_random",
            )
        }
        geometry = {
            "state": {
                "index": 0,
                "role": "v2_label_train",
                "trajectory_id": "synthetic",
                "state_id": "synthetic:decision_step:006",
                "decision_step_id": 6,
                "candidate_event_step_ids": [1, 2, 3, 4],
            },
            "budget_event_capacity": 2,
            "baseline_summary_only_distance": 10.0,
            "methods": methods,
        }
        v1 = score_ocr_rgb_state(
            geometry_record=geometry,
            event_features=features,
            current_feature=current,
            distance_by_coalition=distances,
        )
        v2 = score_ocr_rgb_state(
            geometry_record=geometry,
            event_features=features,
            current_feature=current,
            distance_by_coalition=distances,
            output_protocol_id=PROTOCOL_ID,
        )
        self.assertEqual(v1.pop("protocol_id"), V1_PROTOCOL_ID)
        self.assertEqual(v2.pop("protocol_id"), PROTOCOL_ID)
        self.assertEqual(v1, v2)

    def test_runner_identity_and_source_closure_inputs_are_frozen(self) -> None:
        self.assertEqual(runner.PROTOCOL_ID, PROTOCOL_ID)
        self.assertEqual(
            runner.RUN_STATUS,
            "COMPLETED_RESTORATION_V2_2_OCR_RGB_BASELINE_V2_IDENTITY_REPAIR",
        )
        self.assertEqual(
            runner.VALID_STATUS,
            "VALID_RESTORATION_V2_2_OCR_RGB_BASELINE_V2_IDENTITY_REPAIR",
        )
        self.assertEqual(len(runner.FORMAL_SOURCE_PATHS), len(set(runner.FORMAL_SOURCE_PATHS)))
        self.assertIn(CANONICAL_CONFIG_PATH, runner.FORMAL_SOURCE_PATHS)
        self.assertIn(
            "code/causalcache/restoration_v2_2_ocr_rgb_contract_v2.py",
            runner.FORMAL_SOURCE_PATHS,
        )
        self.assertIn(
            "code/scripts/run_restoration_v2_2_ocr_rgb_baseline_v2.py",
            runner.FORMAL_SOURCE_PATHS,
        )
        for relative in runner.FORMAL_SOURCE_PATHS:
            self.assertTrue((ROOT / relative).is_file(), msg=relative)

    def test_runner_threads_exact_repair_parameters_into_shared_reducer(self) -> None:
        records = tuple(
            {"protocol_id": PROTOCOL_ID, "synthetic_index": index}
            for index in range(15)
        )
        aggregate = {
            "by_role": {
                role: {
                    "state_count": count,
                    "mean_normalized_recovery": 0.5,
                    "exact_coalition_match_rate": 0.25,
                    "exact_cardinality_coalition_match_rate": 0.5,
                }
                for role, count in (
                    ("v2_label_train", 10),
                    ("v2_development", 5),
                    ("overall_stratified", 15),
                )
            }
        }
        with patch.object(
            runner,
            "build_state_score_records",
            return_value=records,
        ) as build, patch.object(
            runner,
            "summarize_state_scores",
            return_value=aggregate,
        ), patch.object(
            runner,
            "_input_identity",
            return_value={"synthetic": True},
        ), patch.object(
            runner,
            "_formal_python_source_closure",
            return_value={
                "rule": "synthetic",
                "path_count": 1,
                "inventory_sha256": "a" * 64,
            },
        ):
            files = runner._expected_files(
                contract=self.contract,
                labels_archive=Path("/synthetic/labels.tar"),
                derived_root=Path("/synthetic/derived"),
                source_commit="b" * 40,
            )
        self.assertEqual(
            build.call_args.kwargs["trajectory_identity_occurrences_per_line"],
            2,
        )
        self.assertEqual(
            build.call_args.kwargs["ocr_identity_occurrences_per_line"],
            1,
        )
        self.assertEqual(
            build.call_args.kwargs["output_protocol_id"],
            PROTOCOL_ID,
        )
        self.assertEqual(set(files), {"README.md", "state_scores.jsonl", "summary.json"})


if __name__ == "__main__":
    unittest.main()
