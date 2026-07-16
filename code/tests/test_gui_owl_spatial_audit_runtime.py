from __future__ import annotations

import importlib.util
import os
import types
import unittest
from unittest import mock

from causalcache.policy.gui_owl_spatial_audit_runtime import (
    GUIOwlSpatialAuditRuntime,
    _attention_metadata,
    _validate_observed_attention,
    audit_absent_scientific_environment,
)


class SpatialAuditRuntimeContractTest(unittest.TestCase):
    def test_constructor_rejects_unvalidated_profile_before_gpu_import(self) -> None:
        with self.assertRaisesRegex(TypeError, "ProfileSpec"):
            GUIOwlSpatialAuditRuntime(
                model_dir="/model",
                expected_snapshot_manifest="/manifest",
                device="cuda:0",
                profile=object(),  # type: ignore[arg-type]
            )

    def test_attention_metadata_records_top_and_nested_configs(self) -> None:
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                _attn_implementation="eager",
                text_config=types.SimpleNamespace(_attn_implementation="eager"),
                vision_config=types.SimpleNamespace(_attn_implementation="eager"),
            )
        )
        self.assertEqual(
            _attention_metadata(model),
            {"model": "eager", "text_config": "eager", "vision_config": "eager"},
        )

    def test_attention_profile_rejects_auto_eager_or_partial_eager(self) -> None:
        auto = types.SimpleNamespace(attention_implementation="default")
        eager = types.SimpleNamespace(attention_implementation="eager")
        _validate_observed_attention(
            auto,  # type: ignore[arg-type]
            {"model": "sdpa", "text_config": "sdpa", "vision_config": None},
        )
        _validate_observed_attention(
            eager,  # type: ignore[arg-type]
            {"model": "eager", "text_config": "eager", "vision_config": None},
        )
        with self.assertRaisesRegex(RuntimeError, "unexpectedly resolved to eager"):
            _validate_observed_attention(
                auto,  # type: ignore[arg-type]
                {"model": "eager", "text_config": "sdpa"},
            )
        with self.assertRaisesRegex(RuntimeError, "not applied everywhere"):
            _validate_observed_attention(
                eager,  # type: ignore[arg-type]
                {"model": "eager", "text_config": "sdpa"},
            )

    def test_scientific_environment_audit_records_absence_and_rejects_presence(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            audit = audit_absent_scientific_environment()
        self.assertTrue(audit["all_absent"])
        self.assertEqual(audit["present_names"], [])
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", audit["audited_names"])
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", audit["audited_names"])
        with mock.patch.dict(
            os.environ,
            {"NVIDIA_TF32_OVERRIDE": "0"},
            clear=True,
        ), self.assertRaisesRegex(RuntimeError, "NVIDIA_TF32_OVERRIDE"):
            audit_absent_scientific_environment()

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch unavailable")
    def test_vector_summary_keeps_only_top_two_and_target_scalars(self) -> None:
        import torch

        runtime = object.__new__(GUIOwlSpatialAuditRuntime)
        runtime.torch = torch
        result = runtime._vector_summary(
            torch.tensor([1.0, 4.0, 2.0], dtype=torch.float32),
            target_token_id=2,
        )
        self.assertEqual(result["top1_token_id"], 1)
        self.assertEqual(result["top2_token_id"], 2)
        self.assertEqual(result["target_rank_strict"], 2)
        self.assertEqual(result["top1_minus_top2_margin"], 2.0)
        self.assertNotIn("full_vector", result)


if __name__ == "__main__":
    unittest.main()
