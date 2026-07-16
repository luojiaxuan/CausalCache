from __future__ import annotations

import copy
import itertools
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import causalcache.restoration_v2_2_ocr_rgb as ocr_rgb_module
from causalcache.restoration_v2_2_ocr_rgb import (
    SelectedImageFeature,
    score_ocr_rgb_state,
    select_jsonl_records_by_identity,
    summarize_state_scores,
)
from causalcache.restoration_v2_2_ocr_rgb_contract import (
    FROZEN_CONFIG_SHA256,
    PASS_STATUS,
    validate_contract,
)
from scripts.run_restoration_v2_2_ocr_rgb_baseline import (
    _formal_python_source_closure,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline.json"


def _solid_rgb(red: int, green: int, blue: int) -> bytes:
    return bytes((red, green, blue)) * (256 * 256)


def _feature(name: str, tokens: tuple[str, ...], rgb: bytes) -> SelectedImageFeature:
    return SelectedImageFeature(
        image_member_path=f"images/synthetic/{name}.png",
        image_sha256=name * 64,
        canonical_ocr_record_sha256=name.upper() * 64,
        full_spatial_tokens=tokens,
        resized_rgb_bytes=rgb,
    )


class RestorationV22OcrRgbContractTest(unittest.TestCase):
    def test_frozen_contract_and_bound_sources_pass(self) -> None:
        result = validate_contract(CONTRACT, repository_root=ROOT)
        self.assertEqual(result["status"], PASS_STATUS)
        self.assertEqual(result["config_sha256"], FROZEN_CONFIG_SHA256)
        self.assertEqual(result["expected_state_count"], 15)
        self.assertEqual(result["expected_candidate_comparison_count"], 60)
        self.assertEqual(result["expected_unique_image_count"], 75)
        self.assertEqual(result["gpu_count"], 0)
        self.assertEqual(result["ocr_inference_count"], 0)
        self.assertFalse(result["formal_aggregate_generated"])

        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        closure = _formal_python_source_closure(ROOT, head)
        self.assertGreater(closure["path_count"], 50)
        self.assertIn("code/causalcache/low_fidelity_v2.py", closure["paths"])
        self.assertIn(
            "code/causalcache/restoration_v2_2_geometry_stats.py",
            closure["paths"],
        )

    def test_nonselected_jsonl_record_is_never_semantically_parsed(self) -> None:
        selected = json.dumps(
            {"source_id": "train-1", "value": 7},
            sort_keys=True,
            separators=(",", ":"),
        )
        opaque_nonselected = '{"source_id":"confirm-locked",not-valid-json}'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            path.write_text(
                selected + "\n" + opaque_nonselected + "\n",
                encoding="utf-8",
            )
            records = select_jsonl_records_by_identity(
                path,
                identity_field="source_id",
                allowed_identities={"train-1"},
                expected_total_line_count=2,
                label="synthetic trajectories",
            )
        self.assertEqual(records, {"train-1": {"source_id": "train-1", "value": 7}})

    def test_synthetic_ocr_rgb_scoring_uses_frozen_top_two(self) -> None:
        red = _solid_rgb(255, 0, 0)
        blue = _solid_rgb(0, 0, 255)
        features = {
            1: _feature("a", ("Settings",), blue),
            2: _feature("b", ("Other",), red),
            3: _feature("c", ("Settings",), red),
            4: _feature("d", ("Other",), blue),
        }
        current = _feature("q", ("Settings",), red)
        distances = {
            coalition: 5.0
            for cardinality in range(5)
            for coalition in itertools.combinations((1, 2, 3, 4), cardinality)
        }
        distances[()] = 10.0
        distances[(1, 3)] = 1.0
        distances[(3,)] = 0.5
        methods = {
            name: {
                "selected_coalition": [1, 3] if name != "analytic_exact_cardinality_random" else None,
                "normalized_recovery": 0.9,
            }
            for name in (
                "exact_subset",
                "exact_cardinality_oracle",
                "true_conditional_greedy",
                "budget_conditioned_independent",
                "full_shapley_independent",
                "dynamic_recent",
                "analytic_exact_cardinality_random",
            )
        }
        methods["exact_subset"] = {
            "selected_coalition": [3],
            "normalized_recovery": 0.95,
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
        with patch(
            "causalcache.restoration_v2_2_ocr_rgb.ocr_token_set_jaccard",
            wraps=ocr_rgb_module.ocr_token_set_jaccard,
        ) as ocr_score, patch(
            "causalcache.restoration_v2_2_ocr_rgb.joint_rgb_histogram_cosine",
            wraps=ocr_rgb_module.joint_rgb_histogram_cosine,
        ) as rgb_score:
            record = score_ocr_rgb_state(
                geometry_record=geometry,
                event_features=features,
                current_feature=current,
                distance_by_coalition=distances,
            )
        self.assertEqual(ocr_score.call_count, 4)
        self.assertEqual(rgb_score.call_count, 4)
        self.assertEqual(record["ranked_event_step_ids"], [3, 1, 2, 4])
        self.assertEqual(record["selected_coalition"], [1, 3])
        self.assertEqual(record["exact_subset_coalition"], [3])
        self.assertFalse(record["exact_coalition_match"])
        self.assertEqual(record["exact_cardinality_oracle_coalition"], [1, 3])
        self.assertTrue(record["exact_cardinality_coalition_match"])
        self.assertAlmostEqual(record["normalized_recovery"], 0.9)

        formal_rows = []
        for index in range(15):
            row = copy.deepcopy(record)
            role = "v2_label_train" if index < 10 else "v2_development"
            trajectory_id = f"trajectory-{index:02d}"
            row["state"].update(
                {
                    "index": index,
                    "role": role,
                    "trajectory_id": trajectory_id,
                    "state_id": f"{trajectory_id}:decision_step:006",
                }
            )
            row["geometry_comparators"]["dynamic_recent"][
                "normalized_recovery"
            ] = 0.5
            formal_rows.append(row)
        summary = summarize_state_scores(formal_rows)
        self.assertEqual(summary["by_role"]["v2_development"]["state_count"], 5)
        recent = summary["paired_trajectory_bootstrap"]["dynamic_recent"]
        self.assertAlmostEqual(recent["development"]["mean_difference"], 0.4)
        self.assertEqual(recent["development"]["win_tie_loss"]["wins"], 5)
        self.assertEqual(recent["development"]["win_tie_loss"]["count"], 5)
        self.assertEqual(
            len(summary["development_paired_deltas"]["dynamic_recent"]),
            5,
        )


if __name__ == "__main__":
    unittest.main()
