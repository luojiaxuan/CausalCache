from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_selector_geometry_contract_v2 import (
    validate_repaired_scientific_payload,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "data/results/restoration_v2_2_selector_geometry_v2_repair"
EXPECTED_FILE_HASHES = {
    "README.md": "dd54af30e7cb07e3661f38aadce4d9008483b2980baea4eb6d9807f07d3a9c01",
    "state_budget_records.jsonl": (
        "b3f67714bb5667ceda945a3cb953b8987108aef607d5819a617272d48450cb03"
    ),
    "summary.json": "57eecbba5326cabf430f0a739118e986781a595e98ea6a32ddaeef118abc6811",
}


class RestorationV22SelectorGeometryV2ArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((RESULT / "summary.json").read_bytes())
        cls.record_bytes = (RESULT / "state_budget_records.jsonl").read_bytes()
        cls.records = tuple(
            json.loads(line) for line in cls.record_bytes.splitlines()
        )

    def test_exact_three_file_inventory_and_hashes(self) -> None:
        self.assertEqual(
            sorted(path.name for path in RESULT.iterdir()),
            sorted(EXPECTED_FILE_HASHES),
        )
        for name, expected in EXPECTED_FILE_HASHES.items():
            self.assertEqual(
                hashlib.sha256((RESULT / name).read_bytes()).hexdigest(),
                expected,
                msg=name,
            )

    def test_summary_binds_source_input_and_zero_operations(self) -> None:
        self.assertEqual(
            self.summary["status"],
            "COMPLETED_RESTORATION_V2_2_SELECTOR_GEOMETRY_V2_REPAIR",
        )
        self.assertEqual(
            self.summary["source_execution"]["source_git_commit"],
            "9a4eca5a53c2a9a3340c6274b9fa5ff9012a5a64",
        )
        self.assertEqual(
            self.summary["immutable_label_input"]["hf_revision"],
            "8f6baae5c0b23b08915fa1b0fb848dd519b4c8db",
        )
        self.assertEqual(
            self.summary["scientific_payload_sha256"],
            "cc505443a7efdc68c8eeca754f24c9143cabf72a090f024fde05011e783cbb21",
        )
        self.assertTrue(
            all(
                type(value) is int and value == 0
                for value in self.summary["science"]["operation_counts"].values()
            )
        )

    def test_jsonl_is_canonical_and_independent_schema_validation_passes(self) -> None:
        rebuilt = b"".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
            for record in self.records
        )
        self.assertEqual(rebuilt, self.record_bytes)
        self.assertEqual(len(self.records), 180)
        self.assertEqual(
            self.summary["state_budget_records"],
            {
                "path": "state_budget_records.jsonl",
                "record_count": 180,
                "sha256": EXPECTED_FILE_HASHES["state_budget_records.jsonl"],
                "size_bytes": len(self.record_bytes),
            },
        )
        science = dict(self.summary["science"])
        science["state_budget_records"] = list(self.records)
        validate_repaired_scientific_payload(science)

    def test_primary_method_shaping_and_claim_boundary_are_exact(self) -> None:
        primary = self.summary["science"]["slices"]["primary_n4_b2"]
        exact = primary["methods"]["exact_subset"]["v2_development"]
        independent = primary["methods"]["budget_conditioned_independent"][
            "v2_development"
        ]
        self.assertAlmostEqual(exact["mean_normalized_recovery"], 0.9253419585123936)
        self.assertAlmostEqual(
            independent["mean_normalized_recovery"],
            0.7663033655682001,
        )
        shaping = self.summary["science"]["internal_method_shaping"]
        self.assertEqual(
            shaping["conditioning_decision"],
            "set_conditioned_main_candidate",
        )
        self.assertEqual(shaping["search_decision"], "online_greedy_sufficient")
        self.assertFalse(shaping["paper_claim_gate"])
        self.assertFalse(shaping["confirm_unlock_gate"])


if __name__ == "__main__":
    unittest.main()
