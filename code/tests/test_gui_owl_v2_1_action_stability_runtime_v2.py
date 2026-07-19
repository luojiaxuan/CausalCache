from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import causalcache.policy.gui_owl_v2_1_action_stability_runtime_v2 as runtime_v2
from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v1 import (
    GUIOwlV21AutoActionStabilityRuntimeV1,
)
from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v2 import (
    GUI_OWL_V2_1_ACTION_STABILITY_SDPA_NUMERICAL_CONTROL_CLAIM_V2,
    GUI_OWL_V2_1_ACTION_STABILITY_SDPA_NUMERICAL_CONTROL_RUNTIME_V2_ID,
    GUIOwlV21SDPANumericalControlActionStabilityRuntimeV2,
    validate_sdpa_numerical_control_attention_v2,
)


def test_runtime_reuses_parent_memory_safe_sdpa_loader() -> None:
    assert issubclass(
        GUIOwlV21SDPANumericalControlActionStabilityRuntimeV2,
        GUIOwlV21AutoActionStabilityRuntimeV1,
    )
    assert "sdpa" in GUI_OWL_V2_1_ACTION_STABILITY_SDPA_NUMERICAL_CONTROL_RUNTIME_V2_ID
    assert "not_strict_cuda_determinism" in (
        GUI_OWL_V2_1_ACTION_STABILITY_SDPA_NUMERICAL_CONTROL_CLAIM_V2
    )


def test_attention_requires_exact_top_text_and_vision_sdpa() -> None:
    expected = {"text": "sdpa", "top": "sdpa", "vision": "sdpa"}
    assert validate_sdpa_numerical_control_attention_v2(expected) == expected
    with pytest.raises(RuntimeError, match="resolve to SDPA"):
        validate_sdpa_numerical_control_attention_v2(
            {"text": "sdpa", "top": "sdpa", "vision": "eager"}
        )
    with pytest.raises(RuntimeError, match="resolve to SDPA"):
        validate_sdpa_numerical_control_attention_v2(
            {"text": "sdpa", "top": "sdpa"}
        )
    with pytest.raises(RuntimeError, match="inventory is invalid"):
        validate_sdpa_numerical_control_attention_v2("sdpa")  # type: ignore[arg-type]


def test_bootstrap_rejects_a_process_that_already_initialized_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())
    monkeypatch.setattr(
        runtime_v2,
        "audit_gui_owl_v2_2_scientific_environment",
        lambda: {"all_absent": True},
    )
    with pytest.raises(RuntimeError, match="fresh pre-CUDA process"):
        runtime_v2._bootstrap_numerical_controls_before_cuda_v2()


def test_bootstrap_applies_controls_before_stack_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: False),
    )
    fake_transformers = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        runtime_v2,
        "audit_gui_owl_v2_2_scientific_environment",
        lambda: events.append("environment_audit") or {"all_absent": True},
    )
    monkeypatch.setattr(
        runtime_v2,
        "_configure_and_validate_numerical_controls",
        lambda torch: events.append("numerical_controls") or {"cudnn_benchmark": False},
    )
    monkeypatch.setattr(
        runtime_v2,
        "_validate_software_stack",
        lambda torch, transformers: events.append("software_stack")
        or {"torch_version": "frozen"},
    )
    bootstrap = runtime_v2._bootstrap_numerical_controls_before_cuda_v2()
    assert events == ["environment_audit", "numerical_controls", "software_stack"]
    assert bootstrap["torch"] is fake_torch


def test_runtime_bootstraps_before_parent_cuda_loader_and_binds_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_name=lambda device: events.append("gpu_identity") or "NVIDIA H200"
        )
    )

    def bootstrap():
        events.append("bootstrap")
        return {
            "numerical_controls": {"cudnn_benchmark": False},
            "scientific_environment_audit": {"all_absent": True},
            "software_stack": {"torch_cuda_version": "13.0"},
            "torch": fake_torch,
        }

    def parent_init(self, **kwargs):
        events.append("parent_cuda_loader")
        self.torch = fake_torch
        self.device = "cuda:0"
        self.model = object()
        self.metadata = {"runtime_profile_id": "historical_auto"}

    monkeypatch.setattr(runtime_v2, "_bootstrap_numerical_controls_before_cuda_v2", bootstrap)
    monkeypatch.setattr(GUIOwlV21AutoActionStabilityRuntimeV1, "__init__", parent_init)
    monkeypatch.setattr(
        runtime_v2,
        "gui_owl_v2_2_observed_attention",
        lambda model: events.append("attention_readback")
        or {"text": "sdpa", "top": "sdpa", "vision": "sdpa"},
    )
    runtime = GUIOwlV21SDPANumericalControlActionStabilityRuntimeV2(
        model_dir="opaque",
        expected_snapshot_manifest="opaque",
        device="cuda:0",
        target_effective_visual_tokens_per_image=2560,
    )
    assert events == [
        "bootstrap",
        "parent_cuda_loader",
        "attention_readback",
        "gpu_identity",
    ]
    assert runtime.metadata["runtime_profile_id"] == (
        GUI_OWL_V2_1_ACTION_STABILITY_SDPA_NUMERICAL_CONTROL_RUNTIME_V2_ID
    )
    assert runtime.metadata["observed_attention_implementation"] == {
        "text": "sdpa",
        "top": "sdpa",
        "vision": "sdpa",
    }
    assert runtime.metadata["strict_cuda_determinism_claimed"] is False
    assert runtime.metadata["fresh_process_per_state_required"] is True
