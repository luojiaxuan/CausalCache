from __future__ import annotations

import importlib.util

import pytest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_decision_loss_uses_complete_expansion_groups_and_stop() -> None:
    import torch

    from scripts.train_set_utility_token_predictor import _loss

    predictions = torch.tensor(
        [[0.0, 0.2, 0.1, 0.35]], dtype=torch.float32, requires_grad=True
    )
    subset_masks = torch.tensor(
        [[[False, False], [True, False], [False, True], [True, True]]]
    )
    batch = {
        "model": {
            "event_mask": torch.tensor([[True, True]]),
            "subset_masks": subset_masks,
        },
        "raw_targets": torch.tensor([[0.0, 0.3, -0.1, 0.8]]),
        "normalized_targets": torch.tensor([[0.0, 0.3, -0.1, 0.8]]),
        "label_mask": torch.tensor([[True, True, True, True]]),
        "scales": torch.tensor([1.0]),
        "scale_mask": torch.tensor([True]),
    }
    total, metrics = _loss(
        predictions,
        batch,
        loss_config={
            "conditional_listwise": 1.0,
            "conditional_marginal": 0.0,
            "decision_regret": 1.0,
            "decision_temperature": 0.25,
            "normalized_regression": 0.0,
            "normalized_smooth_l1_beta": 0.5,
            "raw_regression": 0.0,
            "raw_smooth_l1_beta": 0.05,
            "within_state_ranking": 0.0,
        },
        trajectory_weights=torch.ones(1),
        torch=torch,
    )
    assert torch.isfinite(total)
    assert metrics["complete_decision_group_count"] == 3.0
    assert metrics["conditional_listwise"] > 0.0
    assert metrics["decision_regret"] > 0.0
    total.backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_decision_loss_is_not_diluted_by_rows_without_complete_groups() -> None:
    import torch

    from scripts.train_set_utility_token_predictor import _loss

    loss_config = {
        "conditional_listwise": 1.0,
        "conditional_marginal": 0.0,
        "decision_regret": 1.0,
        "decision_temperature": 0.25,
        "normalized_regression": 0.0,
        "normalized_smooth_l1_beta": 0.5,
        "raw_regression": 0.0,
        "raw_smooth_l1_beta": 0.05,
        "within_state_ranking": 0.0,
    }
    active_batch = {
        "model": {
            "event_mask": torch.tensor([[True, True]]),
            "subset_masks": torch.tensor(
                [[[False, False], [True, False], [False, True]]]
            ),
        },
        "raw_targets": torch.tensor([[0.0, 0.3, -0.1]]),
        "normalized_targets": torch.tensor([[0.0, 0.3, -0.1]]),
        "label_mask": torch.tensor([[True, True, True]]),
        "scales": torch.tensor([1.0]),
        "scale_mask": torch.tensor([True]),
    }
    active_predictions = torch.tensor([[0.0, 0.2, 0.1]])
    active_total, _ = _loss(
        active_predictions,
        active_batch,
        loss_config=loss_config,
        trajectory_weights=torch.ones(1),
        torch=torch,
    )
    mixed_batch = {
        "model": {
            "event_mask": torch.tensor([[True, True], [True, True]]),
            "subset_masks": torch.tensor(
                [
                    [[False, False], [True, False], [False, True]],
                    [[False, False], [True, False], [False, False]],
                ]
            ),
        },
        "raw_targets": torch.tensor([[0.0, 0.3, -0.1], [0.0, 0.2, 0.0]]),
        "normalized_targets": torch.tensor(
            [[0.0, 0.3, -0.1], [0.0, 0.2, 0.0]]
        ),
        "label_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "scales": torch.tensor([1.0, 1.0]),
        "scale_mask": torch.tensor([True, True]),
    }
    mixed_predictions = torch.tensor([[0.0, 0.2, 0.1], [0.0, 0.1, 0.0]])
    mixed_total, metrics = _loss(
        mixed_predictions,
        mixed_batch,
        loss_config=loss_config,
        trajectory_weights=torch.ones(2),
        torch=torch,
    )
    assert metrics["complete_decision_group_count"] == 1.0
    assert torch.allclose(mixed_total, active_total)
