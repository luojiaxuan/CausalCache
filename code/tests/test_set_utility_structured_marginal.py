from __future__ import annotations

import importlib.util

import pytest

from causalcache.set_utility_structured_marginal import (
    StructuredMarginalHeadConfig,
    StructuredMarginalLossConfig,
    structured_greedy_budget_path,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def test_structured_greedy_uses_calibrated_stop_and_at_most_budget() -> None:
    def score(selected: tuple[int, ...]) -> tuple[float, ...]:
        if not selected:
            return (0.15, 0.8, 0.4, 0.1)
        return (0.25, 0.8, 0.2, 0.1)

    result = structured_greedy_budget_path(score, (10, 20, 30))
    assert result["selections"] == {
        "1": [10],
        "2": [10],
        "3": [10],
        "4": [10],
    }
    assert tuple(result["predicted_utilities"].values()) == pytest.approx(
        (0.8, 0.8, 0.8, 0.8)
    )
    assert result["trace"][1]["stop_threshold"] == pytest.approx(0.25)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_structured_head_exposes_symmetric_signed_pair_term() -> None:
    import torch

    from causalcache.set_utility_structured_marginal import (
        StructuredConditionalMarginalHead,
    )

    torch.manual_seed(7)
    head = StructuredConditionalMarginalHead(
        StructuredMarginalHeadConfig(
            input_hidden_size=8,
            hidden_size=8,
            pair_rank=4,
            dropout=0.0,
        )
    ).eval()
    query = torch.randn(1, 8)
    events = torch.randn(1, 3, 8)
    event_mask = torch.ones(1, 3, dtype=torch.bool)
    selected = torch.tensor(
        [[[False, False, False], [True, False, False], [False, True, False]]]
    )
    with torch.inference_mode():
        terms = head.decompose(
            query=query,
            events=events,
            event_mask=event_mask,
            selected_masks=selected,
        )
        scores = head(
            query=query,
            events=events,
            event_mask=event_mask,
            selected_masks=selected,
        )
    assert scores.shape == (1, 3, 4)
    assert torch.equal(terms["pair"][:, 0], torch.zeros(1, 3))
    assert terms["pair"][0, 1, 1] == pytest.approx(
        float(terms["pair"][0, 2, 0]), abs=1e-6
    )
    assert torch.equal(terms["stop"][:, 0], torch.zeros(1))
    assert torch.allclose(
        terms["marginal"], terms["base"] + terms["pair"] + terms["context"]
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_structured_loss_learns_redundancy_complementarity_and_stop() -> None:
    import torch

    from causalcache.set_utility_structured_marginal import (
        structured_conditional_marginal_loss,
    )

    targets = torch.tensor(
        [
            [
                [0.0, 0.4, 0.3, 0.2],
                [0.0, 0.0, 0.1, 0.5],
                [0.0, -0.1, 0.0, -0.2],
            ]
        ]
    )
    scores = targets.clone().requires_grad_(True)
    action_mask = torch.tensor(
        [
            [
                [True, True, True, True],
                [True, False, True, True],
                [True, True, False, True],
            ]
        ]
    )
    selected_masks = torch.tensor(
        [
            [
                [False, False, False],
                [True, False, False],
                [False, True, False],
            ]
        ]
    )
    loss, metrics = structured_conditional_marginal_loss(
        scores,
        normalized_targets=targets,
        action_mask=action_mask,
        group_mask=torch.ones(1, 3, dtype=torch.bool),
        selected_masks=selected_masks,
        config=StructuredMarginalLossConfig(),
    )
    assert torch.isfinite(loss)
    assert metrics["action_top1_accuracy"] == 1.0
    assert metrics["stop_action_accuracy"] == 1.0
    assert metrics["interaction_count"] == 4.0
    assert metrics["interaction_sign_accuracy"] == 1.0
    assert metrics["interaction_regression"] == pytest.approx(0.0)
    loss.backward()
    assert scores.grad is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_structured_loss_keeps_candidate_complete_groups_without_empty_anchor() -> None:
    import torch

    from causalcache.set_utility_structured_marginal import (
        structured_conditional_marginal_loss,
    )

    targets = torch.tensor([[[0.0, 0.0, 0.2, -0.1]]])
    scores = targets.clone().requires_grad_(True)
    loss, metrics = structured_conditional_marginal_loss(
        scores,
        normalized_targets=targets,
        action_mask=torch.tensor([[[True, False, True, True]]]),
        group_mask=torch.ones(1, 1, dtype=torch.bool),
        selected_masks=torch.tensor([[[True, False, False]]]),
        config=StructuredMarginalLossConfig(),
    )
    assert torch.isfinite(loss)
    assert metrics["interaction_count"] == 0.0
    loss.backward()
    assert scores.grad is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_structured_token_model_reuses_direct_source_encoder() -> None:
    import torch

    from causalcache.set_utility_structured_marginal import (
        StructuredTokenConditionalMarginalPredictor,
    )
    from causalcache.set_utility_token_models import (
        TokenConditionalMarginalPredictor,
        TokenUtilityModelConfig,
    )

    shared = dict(
        source_hidden_size=16,
        numeric_feature_size=5,
        hidden_size=16,
        latent_count=2,
        resampler_layers=1,
        set_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    direct = TokenConditionalMarginalPredictor(
        TokenUtilityModelConfig(
            family="set_transformer",
            preserve_entity_latents=True,
            **shared,
        )
    )
    model = StructuredTokenConditionalMarginalPredictor(
        TokenUtilityModelConfig(
            family="deepsets",
            preserve_entity_latents=False,
            **shared,
        ),
        StructuredMarginalHeadConfig(
            input_hidden_size=16,
            hidden_size=8,
            pair_rank=4,
            dropout=0.0,
        ),
    ).eval()
    model.load_shared_encoder_from_direct_state(direct.state_dict())
    inputs = {
        "query_visual_tokens": torch.randn(1, 3, 16),
        "query_visual_mask": torch.ones(1, 3, dtype=torch.bool),
        "query_text_tokens": torch.randn(1, 2, 16),
        "query_text_mask": torch.ones(1, 2, dtype=torch.bool),
        "event_visual_tokens": torch.randn(1, 3, 3, 16),
        "event_visual_mask": torch.ones(1, 3, 3, dtype=torch.bool),
        "event_text_tokens": torch.randn(1, 3, 2, 16),
        "event_text_mask": torch.ones(1, 3, 2, dtype=torch.bool),
        "event_numeric_features": torch.randn(1, 3, 5),
        "event_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    with torch.inference_mode():
        scores = model(
            **inputs,
            selected_masks=torch.tensor([[[False, False, False], [True, False, False]]]),
        )
    assert scores.shape == (1, 2, 4)
    trainable = lambda module: sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    assert trainable(model) < trainable(direct)
    assert trainable(model.head) < trainable(direct) - trainable(direct.encoder)


def test_structured_configs_reject_invalid_calibration() -> None:
    with pytest.raises(ValueError, match="decision_temperature"):
        StructuredMarginalLossConfig(decision_temperature=0.0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_structured_group_sum_matches_global_mean_across_unequal_ranks() -> None:
    import torch

    from causalcache.set_utility_structured_marginal import (
        structured_conditional_marginal_loss,
    )
    from scripts.train_set_utility_structured_marginal import (
        _structured_group_sum_backward_scale,
    )

    config = StructuredMarginalLossConfig(
        marginal_regression=1.0,
        decision_ranking=0.0,
        stop_calibration=0.0,
        interaction_regression=0.0,
        interaction_sign=0.0,
        smooth_l1_beta=0.5,
    )

    def tensors(active_groups: int, target: float, candidate_count: int):
        group_mask = torch.arange(16)[None, :] < active_groups
        action_mask = torch.zeros((1, 16, 4), dtype=torch.bool)
        action_mask[:, :, 0] = group_mask
        action_mask[:, :, 2 : 2 + candidate_count] = group_mask[:, :, None]
        targets = torch.zeros((1, 16, 4), dtype=torch.float32)
        targets[:, :, 2 : 2 + candidate_count] = target
        selected = torch.zeros((1, 16, 3), dtype=torch.bool)
        selected[:, :, 0] = group_mask
        return group_mask, action_mask, targets, selected

    rank_inputs = (tensors(1, -1.0, 1), tensors(16, 1.0, 2))
    sharded_parameter = torch.tensor(0.0, requires_grad=True)
    sharded_losses = []
    for group_mask, action_mask, targets, selected in rank_inputs:
        scores = torch.zeros((1, 16, 4)) + sharded_parameter
        loss_sum, _ = structured_conditional_marginal_loss(
            scores,
            normalized_targets=targets,
            action_mask=action_mask,
            group_mask=group_mask,
            selected_masks=selected,
            config=config,
            reduction="sum",
        )
        sharded_losses.append(loss_sum)
    scale = _structured_group_sum_backward_scale(
        global_group_weight=17.0, world_size=2
    )
    (sum(loss * scale for loss in sharded_losses) / 2.0).backward()

    reference_parameter = torch.tensor(0.0, requires_grad=True)
    reference_scores = torch.zeros((2, 16, 4)) + reference_parameter
    reference_loss, _ = structured_conditional_marginal_loss(
        reference_scores,
        normalized_targets=torch.cat(
            (rank_inputs[0][2], rank_inputs[1][2]), dim=0
        ),
        action_mask=torch.cat((rank_inputs[0][1], rank_inputs[1][1]), dim=0),
        group_mask=torch.cat((rank_inputs[0][0], rank_inputs[1][0]), dim=0),
        selected_masks=torch.cat((rank_inputs[0][3], rank_inputs[1][3]), dim=0),
        config=config,
    )
    reference_loss.backward()
    assert sharded_parameter.grad == pytest.approx(reference_parameter.grad)
    assert float(reference_parameter.grad) < -0.8


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_structured_interaction_and_ranking_are_pack_invariant() -> None:
    import torch

    from causalcache.set_utility_structured_marginal import (
        structured_conditional_marginal_loss,
    )

    targets = torch.tensor(
        [
            [0.0, 0.4, 0.3],
            [0.0, 0.0, 0.8],
            [0.0, -0.2, 0.0],
        ]
    )
    action_mask = torch.tensor(
        [[True, True, True], [True, False, True], [True, True, False]]
    )
    selected = torch.tensor(
        [[False, False], [True, False], [False, True]]
    )
    config = StructuredMarginalLossConfig()

    packed_once_scores = torch.zeros((3, 3), requires_grad=True)
    packed_once_loss, packed_once_metrics = structured_conditional_marginal_loss(
        packed_once_scores[None, :, :],
        normalized_targets=targets[None, :, :],
        action_mask=action_mask[None, :, :],
        group_mask=torch.ones((1, 3), dtype=torch.bool),
        selected_masks=selected[None, :, :],
        group_weights=torch.ones((1, 3)),
        config=config,
    )
    packed_once_loss.backward()

    split_scores = torch.zeros((3, 3), requires_grad=True)
    split_index = torch.tensor([[0, 1], [0, 2]])
    split_loss, split_metrics = structured_conditional_marginal_loss(
        split_scores[split_index],
        normalized_targets=targets[split_index],
        action_mask=action_mask[split_index],
        group_mask=torch.ones((2, 2), dtype=torch.bool),
        selected_masks=selected[split_index],
        group_weights=torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
        config=config,
    )
    split_loss.backward()

    assert float(split_loss.detach()) == pytest.approx(
        float(packed_once_loss.detach())
    )
    assert torch.allclose(split_scores.grad, packed_once_scores.grad, atol=1e-6)
    for key in (
        "decision_ranking",
        "interaction_regression",
        "interaction_sign",
        "marginal_regression",
        "stop_calibration",
        "total",
    ):
        assert split_metrics[key] == pytest.approx(packed_once_metrics[key])
