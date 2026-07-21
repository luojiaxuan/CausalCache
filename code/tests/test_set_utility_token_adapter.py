from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from causalcache.set_utility_token_adapter import (
    TokenAdapterConditionalMarginalPredictor,
    TokenAdapterConfig,
)
from causalcache.set_utility_token_models import (
    TokenConditionalMarginalPredictor,
    TokenUtilityModelConfig,
)


def _predictor_config() -> TokenUtilityModelConfig:
    return TokenUtilityModelConfig(
        family="set_transformer",
        source_hidden_size=16,
        numeric_feature_size=5,
        hidden_size=16,
        latent_count=2,
        resampler_layers=1,
        set_layers=1,
        num_heads=4,
        dropout=0.0,
        preserve_entity_latents=True,
    )


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "query_visual_tokens": torch.randn(2, 3, 16),
        "query_visual_mask": torch.ones(2, 3, dtype=torch.bool),
        "query_text_tokens": torch.randn(2, 2, 16),
        "query_text_mask": torch.ones(2, 2, dtype=torch.bool),
        "event_visual_tokens": torch.randn(2, 3, 3, 16),
        "event_visual_mask": torch.ones(2, 3, 3, dtype=torch.bool),
        "event_text_tokens": torch.randn(2, 3, 2, 16),
        "event_text_mask": torch.ones(2, 3, 2, dtype=torch.bool),
        "event_numeric_features": torch.randn(2, 3, 5),
        "event_mask": torch.tensor(
            [[True, True, True], [True, True, False]], dtype=torch.bool
        ),
        "selected_masks": torch.tensor(
            [
                [[False, False, False], [True, False, False]],
                [[False, False, False], [False, True, False]],
            ],
            dtype=torch.bool,
        ),
    }


def test_zero_initialized_adapter_exactly_reproduces_legacy_predictor() -> None:
    torch.manual_seed(19)
    config = _predictor_config()
    legacy = TokenConditionalMarginalPredictor(config).eval()
    wrapped = TokenAdapterConditionalMarginalPredictor(
        config,
        TokenAdapterConfig(source_hidden_size=16, bottleneck_size=4),
    ).eval()
    wrapped.load_predictor_state_dict(legacy.state_dict())
    inputs = _inputs()
    with torch.inference_mode():
        expected = legacy(**inputs)
        actual = wrapped(**inputs)
    assert torch.equal(actual, expected)
    for name in (
        "query_visual_tokens",
        "query_text_tokens",
        "event_visual_tokens",
        "event_text_tokens",
    ):
        assert torch.equal(wrapped.adapt_source_tokens(inputs[name]), inputs[name])


def test_legacy_checkpoint_loading_is_strict_but_excludes_adapter() -> None:
    config = _predictor_config()
    legacy = TokenConditionalMarginalPredictor(config)
    wrapped = TokenAdapterConditionalMarginalPredictor(
        config,
        TokenAdapterConfig(source_hidden_size=16, bottleneck_size=4),
    )
    state = legacy.state_dict()
    wrapped.load_predictor_state_dict(state)
    for name, value in wrapped.predictor.state_dict().items():
        assert torch.equal(value, state[name])

    incomplete = dict(state)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(RuntimeError):
        wrapped.load_predictor_state_dict(incomplete)

    unexpected = dict(state)
    unexpected["unexpected.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        wrapped.load_predictor_state_dict(unexpected)


def test_adapter_and_head_parameter_groups_support_staged_training() -> None:
    model = TokenAdapterConditionalMarginalPredictor(
        _predictor_config(),
        TokenAdapterConfig(source_hidden_size=16, bottleneck_size=4),
    )
    initially_frozen = {
        name
        for name, parameter in model.predictor.named_parameters()
        if not parameter.requires_grad
    }
    assert initially_frozen

    groups = model.optimizer_parameter_groups(adapter_lr=2e-5, head_lr=2e-4)
    assert [group["name"] for group in groups] == ["adapter", "head"]
    assert [group["lr"] for group in groups] == [2e-5, 2e-4]
    adapter_ids = {id(parameter) for parameter in groups[0]["params"]}
    head_ids = {id(parameter) for parameter in groups[1]["params"]}
    assert adapter_ids
    assert head_ids
    assert adapter_ids.isdisjoint(head_ids)

    model.freeze_head()
    groups = model.optimizer_parameter_groups(adapter_lr=2e-5, head_lr=2e-4)
    assert [group["name"] for group in groups] == ["adapter"]
    model.unfreeze_head()
    assert {
        name
        for name, parameter in model.predictor.named_parameters()
        if not parameter.requires_grad
    } == initially_frozen

    model.freeze_adapter()
    groups = model.optimizer_parameter_groups(adapter_lr=2e-5, head_lr=2e-4)
    assert [group["name"] for group in groups] == ["head"]
    model.unfreeze_adapter()
    assert all(parameter.requires_grad for parameter in model.adapter.parameters())


def test_token_adapter_configs_and_learning_rates_are_validated() -> None:
    with pytest.raises(ValueError):
        TokenAdapterConfig(source_hidden_size=16, bottleneck_size=17)
    with pytest.raises(ValueError):
        TokenAdapterConditionalMarginalPredictor(
            _predictor_config(),
            TokenAdapterConfig(source_hidden_size=32, bottleneck_size=4),
        )
    model = TokenAdapterConditionalMarginalPredictor(
        _predictor_config(),
        TokenAdapterConfig(source_hidden_size=16, bottleneck_size=4),
    )
    with pytest.raises(ValueError):
        model.optimizer_parameter_groups(adapter_lr=0.0, head_lr=1e-4)
    with pytest.raises(TypeError):
        model.optimizer_parameter_groups(adapter_lr=True, head_lr=1e-4)
