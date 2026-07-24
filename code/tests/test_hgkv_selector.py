from __future__ import annotations

import torch

from causalcache.hgkv_selector import (
    HGKVReadoutEncoder,
    HGKVSetConditionedSelector,
    HGKVSingletonSelector,
    set_conditioned_multitask_loss,
    singleton_multitask_loss,
)


def test_readout_encoder_aggregates_layers_and_normalizes_weights() -> None:
    encoder = HGKVReadoutEncoder(
        layer_count=2, layer_feature_dim=3, hidden_dim=4
    )
    encoded, attention = encoder(torch.randn(5, 6))

    assert encoded.shape == (5, 4)
    assert attention.shape == (5, 2)
    torch.testing.assert_close(
        attention.sum(dim=-1), torch.ones(5), atol=1e-6, rtol=1e-6
    )


def test_singleton_selector_masks_padding_and_emits_all_heads() -> None:
    model = HGKVSingletonSelector(
        layer_count=2,
        layer_feature_dim=3,
        hidden_dim=4,
        head_hidden_dim=3,
    )
    features = torch.randn(2, 3, 6)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    outputs = model(features, mask)

    assert outputs["gain"].shape == (2, 3)
    assert outputs["positive_logit"].shape == (2, 3)
    assert outputs["rank_score"].shape == (2, 3)
    assert outputs["stop_logit"].shape == (2,)
    assert outputs["encoded"].shape == (2, 3, 4)


def test_singleton_multitask_loss_is_finite_with_single_candidate_state() -> None:
    model = HGKVSingletonSelector(
        layer_count=2,
        layer_feature_dim=3,
        hidden_dim=4,
        head_hidden_dim=3,
    )
    features = torch.randn(2, 3, 6)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    targets = torch.tensor([[0.2, -0.1, 0.0], [-0.2, 0.0, 0.0]])
    outputs = model(features, mask)

    loss, parts = singleton_multitask_loss(
        outputs,
        targets,
        mask,
        gain_weight=1.0,
        rank_weight=0.5,
        positive_weight=0.25,
        stop_weight=0.25,
        huber_beta=0.02,
        rank_min_delta=1e-6,
    )

    assert torch.isfinite(loss)
    assert set(parts) == {"gain", "positive", "rank", "stop", "total"}


def test_set_conditioned_selector_handles_empty_selected_set() -> None:
    model = HGKVSetConditionedSelector(
        layer_count=2,
        layer_feature_dim=3,
        hidden_dim=4,
        head_hidden_dim=3,
        attention_heads=2,
        max_budget=4,
    )
    candidates = torch.randn(2, 3, 6)
    candidate_mask = torch.tensor([[True, True, False], [True, False, False]])
    selected = torch.randn(2, 2, 6)
    selected_mask = torch.tensor([[True, False], [False, False]])
    remaining_budget = torch.tensor([1, 4])

    outputs = model(
        candidates,
        candidate_mask,
        selected,
        selected_mask,
        remaining_budget,
    )

    assert outputs["marginal"].shape == (2, 3)
    assert outputs["rank_score"].shape == (2, 3)
    assert outputs["positive_logit"].shape == (2, 3)
    assert outputs["stop_logit"].shape == (2,)
    assert torch.isfinite(outputs["marginal"]).all()


def test_set_conditioned_multitask_loss_is_finite() -> None:
    model = HGKVSetConditionedSelector(
        layer_count=2,
        layer_feature_dim=3,
        hidden_dim=4,
        head_hidden_dim=3,
        attention_heads=2,
        max_budget=4,
    )
    candidates = torch.randn(2, 3, 6)
    candidate_mask = torch.tensor([[True, True, False], [True, False, False]])
    selected = torch.randn(2, 2, 6)
    selected_mask = torch.tensor([[True, False], [True, True]])
    remaining_budget = torch.tensor([1, 2])
    targets = torch.tensor([[0.2, -0.1, 0.0], [-0.2, 0.0, 0.0]])
    outputs = model(
        candidates,
        candidate_mask,
        selected,
        selected_mask,
        remaining_budget,
    )

    loss, parts = set_conditioned_multitask_loss(
        outputs,
        targets,
        candidate_mask,
        marginal_weight=1.0,
        rank_weight=0.5,
        positive_weight=0.25,
        stop_weight=0.25,
        huber_beta=0.02,
        rank_min_delta=1e-6,
    )

    assert torch.isfinite(loss)
    assert set(parts) == {"marginal", "positive", "rank", "stop", "total"}
