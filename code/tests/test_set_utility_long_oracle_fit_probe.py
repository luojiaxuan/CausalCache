from __future__ import annotations

import importlib.util

import pytest

from scripts.train_set_utility_long_oracle_fit_probe import (
    _singleton_state,
    _summarize_predictions,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def test_singleton_state_requires_and_orders_candidate_complete_truth() -> None:
    state = {
        "candidate_event_step_ids": [2, 5],
        "distance_rows": [
            {"coalition_event_step_ids": [], "distance": 1.0},
            {"coalition_event_step_ids": [2, 5], "distance": 0.2},
            {"coalition_event_step_ids": [5], "distance": 0.4},
            {"coalition_event_step_ids": [2], "distance": 0.6},
        ],
        "state_id": "trajectory:decision:003",
    }
    filtered = _singleton_state(state)
    assert [
        row["coalition_event_step_ids"] for row in filtered["distance_rows"]
    ] == [[], [2], [5]]
    assert len(state["distance_rows"]) == 4


def test_fit_summary_uses_trajectory_equal_gate_metrics() -> None:
    rows = []
    for state_id, trajectory_id, spearman, recall in (
        ("a", "x", 1.0, True),
        ("b", "x", 1.0, True),
        ("c", "y", 0.0, False),
    ):
        rows.append(
            {
                "history_bin": "long",
                "learned_recovery": 0.5,
                "model_top1_is_true_best": recall,
                "oracle_recovery": 1.0,
                "recent_recovery": 0.0,
                "sign_accuracy": 1.0,
                "spearman": spearman,
                "state_id": state_id,
                "stop_accuracy": 1.0,
                "trajectory_id": trajectory_id,
                "true_best_in_predicted_top4": recall,
                "true_best_outside_recent4": True,
            }
        )
    summary = _summarize_predictions(rows)["overall"]
    assert summary["singleton_spearman_mean"] == 0.5
    assert summary["true_best_in_predicted_top4_rate"] == 0.5
    assert summary["fit_gate"]["passed"] is False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_direct_singleton_head_preserves_empty_and_backpropagates() -> None:
    import torch

    from causalcache.set_utility_token_models import (
        TokenSingletonMarginalPredictor,
        TokenUtilityModelConfig,
    )
    from scripts.train_set_utility_long_oracle_fit_probe import _singleton_loss

    config = TokenUtilityModelConfig(
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
    model = TokenSingletonMarginalPredictor(config)
    subset_masks = torch.tensor(
        [[[False, False], [True, False], [False, True]]]
    )
    predictions = model(
        query_visual_tokens=torch.randn(1, 3, 16),
        query_visual_mask=torch.ones(1, 3, dtype=torch.bool),
        query_text_tokens=torch.randn(1, 2, 16),
        query_text_mask=torch.ones(1, 2, dtype=torch.bool),
        event_visual_tokens=torch.randn(1, 2, 3, 16),
        event_visual_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        event_text_tokens=torch.randn(1, 2, 2, 16),
        event_text_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        event_numeric_features=torch.randn(1, 2, 5),
        event_mask=torch.ones(1, 2, dtype=torch.bool),
        subset_masks=subset_masks,
    )
    assert predictions.shape == (1, 3)
    assert predictions[0, 0] == 0.0
    encoded = model.encode_state_once(
        query_visual_tokens=torch.randn(1, 3, 16),
        query_visual_mask=torch.ones(1, 3, dtype=torch.bool),
        query_text_tokens=torch.randn(1, 2, 16),
        query_text_mask=torch.ones(1, 2, dtype=torch.bool),
        event_visual_tokens=torch.randn(1, 2, 3, 16),
        event_visual_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        event_text_tokens=torch.randn(1, 2, 2, 16),
        event_text_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        event_numeric_features=torch.randn(1, 2, 5),
        event_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    candidate_scores = model.score_encoded_candidates(
        encoded,
        torch.tensor(
            [[[False, False], [True, False]]], dtype=torch.bool
        ),
    )
    assert candidate_scores.shape == (1, 2, 3)
    assert torch.equal(candidate_scores[:, :, 0], torch.zeros(1, 2))
    assert not torch.equal(candidate_scores[0, 0, 2], candidate_scores[0, 1, 2])
    batch = {
        "model": {
            "event_mask": torch.ones(1, 2, dtype=torch.bool),
            "subset_masks": subset_masks,
        },
        "raw_targets": torch.tensor([[0.0, 0.5, -0.2]]),
        "normalized_targets": torch.tensor([[0.0, 0.5, -0.2]]),
        "label_mask": torch.ones(1, 3, dtype=torch.bool),
        "scales": torch.ones(1),
    }
    loss, metrics = _singleton_loss(
        predictions,
        batch,
        loss_config={
            "teacher_mode": "hard",
            "decision_temperature": 0.05,
            "raw_smooth_l1_beta": 0.05,
            "normalized_smooth_l1_beta": 0.5,
            "raw_regression": 0.25,
            "normalized_regression": 1.25,
            "within_state_ranking": 1.0,
            "conditional_listwise": 2.0,
            "decision_regret": 2.0,
            "sign_classification": 1.0,
        },
        torch=torch,
    )
    assert torch.isfinite(loss)
    assert metrics["sign_accuracy"] in {0.0, 0.5, 1.0}
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.marginal_head.parameters()
    )
