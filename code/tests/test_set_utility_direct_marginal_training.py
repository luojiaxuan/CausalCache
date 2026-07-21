from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from scripts.train_set_utility_direct_marginal_v3 import (
    _conditional_groups,
    _split_train_holdout,
)
from scripts.run_set_utility_direct_marginal_tune_selectors import (
    _direct_budget_path,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def test_conditional_groups_require_candidate_complete_expansions() -> None:
    state = {
        "candidate_event_step_ids": [1, 2, 3],
        "distance_rows": [
            {"coalition_event_step_ids": [], "distance": 1.0},
            {"coalition_event_step_ids": [1], "distance": 0.6},
            {"coalition_event_step_ids": [2], "distance": 0.7},
            {"coalition_event_step_ids": [3], "distance": 0.8},
            {"coalition_event_step_ids": [1, 2], "distance": 0.5},
            {"coalition_event_step_ids": [1, 3], "distance": 0.4},
        ],
    }
    groups = _conditional_groups(
        state, normalization_floor=0.01, maximum_base_cardinality=3
    )
    assert [group["selected_event_step_ids"] for group in groups] == [(), (1,)]
    assert groups[0]["raw_marginals"] == pytest.approx((0.0, 0.4, 0.3, 0.2))
    assert groups[1]["raw_marginals"] == pytest.approx((0.0, 0.0, 0.1, 0.2))
    assert groups[1]["action_mask"] == (True, False, True, True)


def test_train_holdout_split_is_trajectory_disjoint_and_deterministic() -> None:
    states = tuple(
        {
            "state_id": f"{trajectory}:{state}",
            "trajectory_id": f"trajectory-{trajectory}",
        }
        for trajectory in range(20)
        for state in range(2)
    )
    first = _split_train_holdout(states, fraction=0.1, salt="frozen")
    second = _split_train_holdout(states, fraction=0.1, salt="frozen")
    assert first == second
    assert len({row["trajectory_id"] for row in first[1]}) == 2
    assert not (
        {row["trajectory_id"] for row in first[0]}
        & {row["trajectory_id"] for row in first[1]}
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_conditional_loss_uses_stop_and_top_action() -> None:
    import torch

    from scripts.train_set_utility_direct_marginal_v3 import _conditional_loss

    batch = {
        "conditional_action_mask": torch.tensor(
            [[[True, True, True], [True, False, True]]]
        ),
        "conditional_group_mask": torch.tensor([[True, True]]),
        "conditional_raw_targets": torch.tensor([[[0.0, 0.4, -0.1], [0.0, 0.0, -0.2]]]),
        "conditional_normalized_targets": torch.tensor(
            [[[0.0, 0.4, -0.1], [0.0, 0.0, -0.2]]]
        ),
        "scales": torch.ones(1),
    }
    predictions = batch["conditional_raw_targets"].clone().requires_grad_(True)
    loss, metrics = _conditional_loss(
        predictions,
        batch,
        loss_config={
            "conditional_marginal": 0.5,
            "conditional_listwise": 2.0,
            "decision_regret": 2.0,
            "sign_classification": 0.25,
            "decision_temperature": 0.05,
            "conditional_marginal_smooth_l1_beta": 0.25,
        },
        torch=torch,
    )
    assert torch.isfinite(loss)
    assert metrics["action_top1_accuracy"] == 1.0
    assert metrics["stop_action_accuracy"] == 1.0
    assert metrics["teacher_stop_rate"] == 0.5
    loss.backward()
    assert predictions.grad is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_direct_model_forward_accepts_selected_masks() -> None:
    import torch

    from causalcache.set_utility_token_models import (
        TokenConditionalMarginalPredictor,
        TokenUtilityModelConfig,
    )

    model = TokenConditionalMarginalPredictor(
        TokenUtilityModelConfig(
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
    )
    common = {
        "query_visual_tokens": torch.randn(1, 3, 16),
        "query_visual_mask": torch.ones(1, 3, dtype=torch.bool),
        "query_text_tokens": torch.randn(1, 2, 16),
        "query_text_mask": torch.ones(1, 2, dtype=torch.bool),
        "event_visual_tokens": torch.randn(1, 2, 3, 16),
        "event_visual_mask": torch.ones(1, 2, 3, dtype=torch.bool),
        "event_text_tokens": torch.randn(1, 2, 2, 16),
        "event_text_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "event_numeric_features": torch.randn(1, 2, 5),
        "event_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    predictions = model(
        **common,
        selected_masks=torch.tensor([[[False, False], [True, False]]]),
    )
    assert predictions.shape == (1, 2, 3)
    assert torch.equal(predictions[:, :, 0], torch.zeros(1, 2))
    with pytest.raises(ValueError, match="exactly one"):
        model(
            **common,
            subset_masks=torch.zeros(1, 1, 2, dtype=torch.bool),
            selected_masks=torch.zeros(1, 1, 2, dtype=torch.bool),
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_direct_model_cached_source_conditioning_matches_full_encoding() -> None:
    import torch

    from causalcache.set_utility_token_models import (
        TokenConditionalMarginalPredictor,
        TokenUtilityModelConfig,
    )

    model = TokenConditionalMarginalPredictor(
        TokenUtilityModelConfig(
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
    ).eval()
    common = {
        "query_visual_tokens": torch.randn(1, 3, 16),
        "query_visual_mask": torch.ones(1, 3, dtype=torch.bool),
        "query_text_tokens": torch.randn(1, 2, 16),
        "query_text_mask": torch.ones(1, 2, dtype=torch.bool),
        "event_visual_tokens": torch.randn(1, 2, 3, 16),
        "event_visual_mask": torch.ones(1, 2, 3, dtype=torch.bool),
        "event_text_tokens": torch.randn(1, 2, 2, 16),
        "event_text_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "event_numeric_features": torch.randn(1, 2, 5),
        "event_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    with torch.inference_mode():
        full = model.encode_state_once(**common)
        query = model.encoder.encode_query_source(
            query_visual_tokens=common["query_visual_tokens"],
            query_visual_mask=common["query_visual_mask"],
            query_text_tokens=common["query_text_tokens"],
            query_text_mask=common["query_text_mask"],
        )
        events = model.encoder.encode_event_sources(
            event_visual_tokens=common["event_visual_tokens"],
            event_visual_mask=common["event_visual_mask"],
            event_text_tokens=common["event_text_tokens"],
            event_text_mask=common["event_text_mask"],
            event_mask=common["event_mask"],
        )
        cached = model.condition_encoded_state(
            query=query,
            event_sources=events,
            event_numeric_features=common["event_numeric_features"],
            event_mask=common["event_mask"],
        )
    assert torch.equal(full.event_mask, cached.event_mask)
    assert torch.allclose(full.query, cached.query)
    assert torch.allclose(full.events, cached.events)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_direct_budget_path_stops_before_budget_when_marginals_are_harmful() -> None:
    import torch

    class FakeModel:
        def score_encoded_candidates(self, encoded, selected_mask):
            selected = tuple(torch.nonzero(selected_mask[0]).flatten().tolist())
            if not selected:
                return torch.tensor([[0.0, 0.8, 0.4, 0.1]])
            return torch.tensor([[0.0, 0.8, -0.2, -0.1]])

    selections, utilities, score_count, trace = _direct_budget_path(
        FakeModel(),
        SimpleNamespace(),
        (10, 20, 30),
        device="cpu",
        torch=torch,
    )
    assert selections == {
        "1": [10],
        "2": [10],
        "3": [10],
        "4": [10],
    }
    assert tuple(utilities.values()) == pytest.approx((0.8, 0.8, 0.8, 0.8))
    assert score_count == 7
    assert trace[1]["ranked_actions"][0]["action"] == "STOP"
