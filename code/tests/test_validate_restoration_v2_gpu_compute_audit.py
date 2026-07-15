from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_restoration_v2_gpu_compute_audit import (
    VALIDATION_OUTCOME,
    _load_json_object,
    _write_exclusive,
    validate_summary,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = (
    REPOSITORY_ROOT
    / "data/results/restoration_v2_gpu_compute_audit/summary.json"
)
SUMMARY_SHA256 = "dce797694194cd88ac749dcc357c3259c41b69a5bde7a99fc93c675cab2e9dac"
RUN_COMMIT = "47062741a950b7c6050a6223b91f4bbae65332e7"
CONTAINER_ID = "69f2b1742e8fd9baac5080b2b97ee1f3c7c1df520f908a4541cadde9d28194df"
CONTAINER_DIGEST = (
    "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
)
GPU_UUID = "e19275bf-adc5-9fc3-42d7-9a3d4b666b81"


def _expected() -> argparse.Namespace:
    return argparse.Namespace(
        expected_run_git_commit=RUN_COMMIT,
        expected_host_alias="hyper00",
        expected_host_hostname="node-radixark-16-0000",
        expected_container_id=CONTAINER_ID,
        expected_container_image_digest=CONTAINER_DIGEST,
        expected_gpu_name="NVIDIA H200",
        expected_gpu_uuid=GPU_UUID,
        expected_nvidia_smi_gpu_uuid=f"GPU-{GPU_UUID}",
        expected_nvidia_driver_version="570.172.08",
        expected_torch_version="2.11.0+cu130",
        expected_torch_cuda_build_version="13.0",
    )


class RestorationV2GPUComputeAuditValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = _load_json_object(SUMMARY_PATH)

    def _validate(self, summary: dict[str, object]) -> dict[str, object]:
        return validate_summary(
            summary,
            summary_sha256=SUMMARY_SHA256,
            repository_root=REPOSITORY_ROOT,
            expected=_expected(),
        )

    def test_committed_formal_summary_passes_independent_validation(self) -> None:
        result = self._validate(copy.deepcopy(self.summary))
        self.assertEqual(result["outcome"], VALIDATION_OUTCOME)
        self.assertEqual(result["summary_sha256"], SUMMARY_SHA256)
        self.assertEqual(result["run_git_commit"], RUN_COMMIT)
        self.assertIs(result["policy_loaded"], False)
        self.assertIs(result["policy_output_generated"], False)
        self.assertIs(result["restoration_output_generated"], False)
        self.assertIs(result["dependency_8_closed"], False)
        self.assertIs(result["screening_unlocked"], False)
        compute = result["compute_validation"]
        self.assertLess(compute["maximum_batch1_cpu_oracle_absolute_error"], 1e-7)
        self.assertEqual(compute["batch2_maximum_absolute_error"], 0.0)
        self.assertEqual(compute["validation_scalar_host_reads"], 0)
        self.assertEqual(compute["full_tensor_host_transfers"], 0)

    def test_policy_or_restoration_exposure_fails_closed(self) -> None:
        for key in (
            "policy_loaded",
            "policy_output_generated",
            "restoration_output_generated",
        ):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.summary)
                changed[key] = True
                with self.assertRaisesRegex(ValueError, "touched policy"):
                    self._validate(changed)

    def test_compute_host_read_and_equivalence_drift_fail_closed(self) -> None:
        changed = copy.deepcopy(self.summary)
        changed["compute_audit"]["host_read_intermediate_tensor_value_count"] = 1
        with self.assertRaisesRegex(ValueError, "intermediate tensor"):
            self._validate(changed)

        changed = copy.deepcopy(self.summary)
        record = changed["compute_audit"]["batch2_vs_two_independent_batch1_gpu_calls"][0]
        record["passed"] = False
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self._validate(changed)

    def test_source_runtime_and_planner_drift_fail_closed(self) -> None:
        changed = copy.deepcopy(self.summary)
        changed["source_files"][
            "code/causalcache/restoration_v2_gpu_kl.py"
        ]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs from run Git commit"):
            self._validate(changed)

        changed = copy.deepcopy(self.summary)
        changed["runtime_identity"]["gpu_uuid"] = "0" * 36
        with self.assertRaisesRegex(ValueError, "gpu_uuid"):
            self._validate(changed)

        changed = copy.deepcopy(self.summary)
        changed["planner_audit"]["planning_scope"] = "cross_state"
        with self.assertRaisesRegex(ValueError, "planner audit"):
            self._validate(changed)

    def test_json_loader_rejects_duplicates_and_nonstandard_constants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"outcome":"a","outcome":"b"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                _load_json_object(duplicate)
            nonstandard = Path(directory) / "nan.json"
            nonstandard.write_text('{"distance":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-standard JSON"):
                _load_json_object(nonstandard)

    def test_validation_output_is_standard_json_and_exclusive(self) -> None:
        result = self._validate(copy.deepcopy(self.summary))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "validation.json"
            _write_exclusive(output, result)
            parsed = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(parsed["outcome"], VALIDATION_OUTCOME)
            with self.assertRaises(FileExistsError):
                _write_exclusive(output, result)


if __name__ == "__main__":
    unittest.main()
