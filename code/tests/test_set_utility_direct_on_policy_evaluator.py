from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from causalcache.set_utility_direct_marginal_replay import (
    FIXED_ZERO_STOP,
    LEARNED_STOP,
)
from causalcache.set_utility_direct_on_policy_evaluator import (
    latency_summary,
    load_set_transformer_selector,
    load_structured_deepsets_selector,
    marginal_budget_path,
    project_inference_state,
    validate_expected_rollout,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
SAFETENSORS_AVAILABLE = importlib.util.find_spec("safetensors") is not None


class AccessGuard(Mapping[str, Any]):
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values
        self.accessed: list[str] = []

    def __getitem__(self, key: str) -> Any:
        if key == "distance_rows":
            raise AssertionError("inference accessed restoration truth")
        self.accessed.append(key)
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def test_inference_projection_never_requests_truth_labels() -> None:
    row = AccessGuard(
        {
            "candidate_event_step_ids": [1, 2],
            "current_image_key": "current",
            "distance_rows": object(),
            "event_image_keys": ["image-1", "image-2"],
            "event_numeric_features": [[0.0, 1.0], [1.0, 2.0]],
            "event_text_keys": ["text-1", "text-2"],
            "instruction_text_key": "instruction",
            "state_id": "trajectory:decision:003",
            "trajectory_id": "trajectory",
        }
    )
    projected = project_inference_state(row)
    assert "distance_rows" not in projected
    assert "distance_rows" not in row.accessed


def test_native_stop_and_confidence_gated_recent_fallback() -> None:
    scores = {
        (): (0.0, 0.5005, 0.5),
        (1,): (0.0, 0.5005, -0.2),
        (2,): (0.0, -0.2, 0.5),
    }
    learned = marginal_budget_path(
        (1, 2),
        score=lambda selected: scores[selected],
        stop_semantics=FIXED_ZERO_STOP,
    )
    hybrid = marginal_budget_path(
        (1, 2),
        score=lambda selected: scores[selected],
        stop_semantics=FIXED_ZERO_STOP,
        recent_fallback_threshold=0.001,
    )
    assert learned["selections"]["1"] == [1]
    assert hybrid["selections"]["1"] == [2]
    assert hybrid["trace"][0]["reason"] == "recent_fallback"
    assert hybrid["selections"]["3"] == [2]
    assert hybrid["selections"]["4"] == [2]
    assert hybrid["trace"][-1]["reason"] == "confidence_gated_stop"


def test_learned_stop_is_not_replaced_by_zero() -> None:
    result = marginal_budget_path(
        (1, 2),
        score=lambda _: (0.4, 0.3, 0.2),
        stop_semantics=LEARNED_STOP,
        recent_fallback_threshold=0.001,
    )
    assert result["selections"] == {"1": [], "2": [], "3": [], "4": []}
    assert result["trace"][0]["stop_threshold"] == pytest.approx(0.4)
    borderline = marginal_budget_path(
        (1, 2),
        score=lambda _: (0.3005, 0.1, 0.3),
        stop_semantics=LEARNED_STOP,
        recent_fallback_threshold=0.001,
    )
    assert borderline["selections"]["1"] == [2]
    assert borderline["trace"][0]["reason"] == "recent_fallback"
    with pytest.raises(ValueError, match="must remain zero"):
        marginal_budget_path(
            (1,),
            score=lambda _: (0.1, 0.2),
            stop_semantics=FIXED_ZERO_STOP,
        )


def test_expected_rollout_requires_exact_native_selections() -> None:
    records = [
        {
            "candidate_event_ids": [1, 2],
            "methods": {"set_transformer": {"1": [1], "2": [1]}},
            "model_name": "set_transformer",
            "state_id": "s1",
        }
    ]
    expected = {
        "checkpoint": {"sha256": "a" * 64},
        "records": [
            {
                "candidate_event_ids": [1, 2],
                "selections": {"1": [1], "2": [1]},
                "state_id": "s1",
            }
        ],
    }
    validate_expected_rollout(
        records, expected, expected_checkpoint_sha256="a" * 64
    )
    expected["records"][0]["selections"]["1"] = [2]
    with pytest.raises(ValueError, match="differs from frozen"):
        validate_expected_rollout(
            records, expected, expected_checkpoint_sha256="a" * 64
        )


def test_latency_summary_uses_only_warm_state_records() -> None:
    records = [
        {"latency_ms": {"set_transformer": {"search": value}}}
        for value in (1.0, 2.0, 9.0)
    ]
    summary = latency_summary(records, ("set_transformer",))
    assert summary["set_transformer"]["search"] == {
        "p50_ms": 2.0,
        "p95_ms": pytest.approx(8.3),
        "state_count": 3,
    }


@pytest.mark.skipif(
    not TORCH_AVAILABLE or not SAFETENSORS_AVAILABLE,
    reason="PyTorch or safetensors is unavailable",
)
def test_checkpoint_loaders_rebuild_exact_architectures(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    from causalcache.set_utility_structured_marginal import (
        StructuredMarginalHeadConfig,
        StructuredTokenConditionalMarginalPredictor,
    )
    from causalcache.set_utility_token_models import (
        TokenConditionalMarginalPredictor,
        TokenUtilityModelConfig,
    )

    set_model_config = {
        "family": "set_transformer",
        "source_hidden_size": 8,
        "numeric_feature_size": 2,
        "hidden_size": 8,
        "latent_count": 2,
        "resampler_layers": 1,
        "set_layers": 1,
        "num_heads": 2,
        "dropout": 0.0,
        "preserve_entity_latents": True,
    }
    set_model = TokenConditionalMarginalPredictor(
        TokenUtilityModelConfig(**set_model_config)
    )
    set_path = tmp_path / "set.safetensors"
    save_file(
        {key: value.contiguous() for key, value in set_model.state_dict().items()},
        str(set_path),
    )
    from causalcache.set_utility_direct_on_policy_evaluator import sha256_file

    set_adapter = load_set_transformer_selector(
        variant={"model": set_model_config},
        checkpoint_path=set_path,
        expected_checkpoint_sha256=sha256_file(set_path),
        device="cpu",
        load_file=load_file,
    )
    assert set_adapter.stop_semantics == FIXED_ZERO_STOP
    assert set(set_adapter.model.state_dict()) == set(set_model.state_dict())

    encoder_config = {
        **set_model_config,
        "family": "deepsets",
        "preserve_entity_latents": False,
    }
    head_config = {
        "input_hidden_size": 8,
        "hidden_size": 8,
        "pair_rank": 2,
        "dropout": 0.0,
    }
    deepsets = StructuredTokenConditionalMarginalPredictor(
        TokenUtilityModelConfig(**encoder_config),
        StructuredMarginalHeadConfig(**head_config),
    )
    deepsets_path = tmp_path / "deepsets.safetensors"
    save_file(
        {key: value.contiguous() for key, value in deepsets.state_dict().items()},
        str(deepsets_path),
    )
    deepsets_adapter = load_structured_deepsets_selector(
        variant={"encoder": encoder_config, "head": head_config},
        checkpoint_path=deepsets_path,
        expected_checkpoint_sha256=sha256_file(deepsets_path),
        device="cpu",
        load_file=load_file,
    )
    assert deepsets_adapter.stop_semantics == LEARNED_STOP
    assert set(deepsets_adapter.model.state_dict()) == set(deepsets.state_dict())
    assert isinstance(torch.tensor(1), torch.Tensor)
