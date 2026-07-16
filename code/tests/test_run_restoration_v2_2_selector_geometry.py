from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from causalcache.restoration_v2_2_selector_geometry_contract import (
    CANONICAL_CONFIG_PATH,
    RestorationV22SelectorGeometryContract,
)
from causalcache.restoration_v2_2_selector_geometry_result import (
    build_selector_geometry_scientific_payload,
)
from scripts.run_restoration_v2_2_selector_geometry import (
    RESULT_FILES,
    _expected_files,
    _validate_existing_result,
    _write_new_result,
)
from tests.test_restoration_v2_2_selector_geometry_result import (
    _two_trajectory_fixture,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


class RunRestorationV22SelectorGeometryTest(unittest.TestCase):
    def test_expected_result_files_are_deterministic_and_self_bound(self) -> None:
        contract = RestorationV22SelectorGeometryContract.load(
            CONFIG,
            repository_root=ROOT,
        )
        small_payload = build_selector_geometry_scientific_payload(
            _two_trajectory_fixture(),
            bootstrap_resamples=20,
            bootstrap_seed=7,
            enforce_formal_denominator=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "labels.tar"
            archive.write_bytes(b"test archive")
            with patch(
                "scripts.run_restoration_v2_2_selector_geometry."
                "load_selector_geometry_states",
                return_value=(),
            ), patch(
                "scripts.run_restoration_v2_2_selector_geometry."
                "build_selector_geometry_scientific_payload",
                return_value=small_payload,
            ):
                files = _expected_files(
                    contract=contract,
                    label_archive=archive,
                    source_commit="1" * 40,
                )
                repeated = _expected_files(
                    contract=contract,
                    label_archive=archive,
                    source_commit="1" * 40,
                )
            self.assertEqual(files, repeated)
            self.assertEqual(sorted(files), sorted(RESULT_FILES))
            summary = json.loads(files["summary.json"])
            records = files["state_budget_records.jsonl"].splitlines()
            self.assertEqual(
                summary["state_budget_records"]["record_count"],
                len(records),
            )
            self.assertEqual(len(records), 24)
            self.assertIn(b"Primary `n=4,B=2`", files["README.md"])
            self.assertNotIn("state_budget_records", summary["science"])

    def test_new_result_write_and_exact_validation_reject_drift(self) -> None:
        files = {
            "README.md": b"readme\n",
            "state_budget_records.jsonl": b"{}\n",
            "summary.json": b"{}\n",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            _write_new_result(output, files)
            _validate_existing_result(output, files)
            with self.assertRaises(FileExistsError):
                _write_new_result(output, files)
            (output / "summary.json").write_bytes(b"drift\n")
            with self.assertRaisesRegex(ValueError, "summary.json"):
                _validate_existing_result(output, files)


if __name__ == "__main__":
    unittest.main()
