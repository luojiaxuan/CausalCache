from __future__ import annotations

import numpy as np
import torch

from scripts.select_hgkv_sets_v1 import select_for_state


class _Stage1:
    def __call__(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        del mask
        rank = -features[..., 0]
        return {
            "gain": rank,
            "positive_logit": torch.zeros_like(rank),
            "rank_score": rank,
            "stop_logit": torch.full(
                (features.shape[0],), -10.0, device=features.device
            ),
        }


class _Stage2:
    def __init__(self, *, stop_after: int = 2) -> None:
        self.stop_after = stop_after
        self.selected_counts: list[int] = []

    def __call__(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        selected_features: torch.Tensor,
        selected_mask: torch.Tensor,
        remaining_budget: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del candidate_mask, selected_features, remaining_budget
        rank = -candidate_features[..., 0]
        selected_count = selected_mask.sum(dim=1)
        self.selected_counts.extend(int(value) for value in selected_count)
        stop = torch.where(
            selected_count >= self.stop_after,
            torch.full_like(selected_count, 10.0, dtype=torch.float32),
            torch.full_like(selected_count, -10.0, dtype=torch.float32),
        )
        return {
            "marginal": rank,
            "positive_logit": torch.zeros_like(rank),
            "rank_score": rank,
            "stop_logit": stop,
        }


def test_greedy_selection_maps_remaining_indices_and_honors_stop() -> None:
    results = select_for_state(
        pair_group="episode:6",
        event_ids=[1, 2, 3, 4],
        feature_rows=np.asarray(
            [[1.0], [2.0], [3.0], [4.0]], dtype=np.float32
        ),
        budgets=[1, 2, 4],
        stage1=_Stage1(),
        stage2=_Stage2(),
        device=torch.device("cpu"),
    )
    by_key = {
        (row["method"], row["budget"]): row for row in results
    }
    assert by_key[("hgkv_singleton", 4)][
        "selected_event_step_ids"
    ] == [1, 2, 3, 4]
    assert by_key[("hgkv_set_conditioned", 1)][
        "selected_event_step_ids"
    ] == [1]
    assert by_key[("hgkv_set_conditioned", 2)][
        "selected_event_step_ids"
    ] == [1, 2]
    conditioned_b4 = by_key[("hgkv_set_conditioned", 4)]
    assert conditioned_b4["selected_event_step_ids"] == [1, 2]
    assert conditioned_b4["stopped"] is True
    assert conditioned_b4["steps"][-1]["candidate_event_id"] == 3
    assert conditioned_b4["steps"][-1]["selected_before"] == [1, 2]


def test_b4_extrapolation_executes_three_selected_event_input() -> None:
    stage2 = _Stage2(stop_after=3)
    results = select_for_state(
        pair_group="episode:6",
        event_ids=[1, 2, 3, 4],
        feature_rows=np.asarray(
            [[1.0], [2.0], [3.0], [4.0]], dtype=np.float32
        ),
        budgets=[4],
        stage1=_Stage1(),
        stage2=stage2,
        device=torch.device("cpu"),
    )
    conditioned = next(
        row for row in results if row["method"] == "hgkv_set_conditioned"
    )
    assert stage2.selected_counts == [1, 2, 3]
    assert conditioned["selected_event_step_ids"] == [1, 2, 3]
    assert conditioned["steps"][-1]["candidate_event_id"] == 4
    assert conditioned["steps"][-1]["selected_before"] == [1, 2, 3]
    assert conditioned["stopped"] is True
