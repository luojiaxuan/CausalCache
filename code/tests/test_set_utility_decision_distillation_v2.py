from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_lazy_cpu_token_cache_reads_inventory_without_device_preload(
    tmp_path: Path,
) -> None:
    import torch
    from safetensors.torch import save_file

    from scripts.train_set_utility_token_predictor import _TokenCache

    partition = tmp_path / "context-shards" / "shard-000-of-001"
    partition.mkdir(parents=True)
    shard = partition / "chunk-00000.safetensors"
    visual = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
    text = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    save_file({"visual_tensor": visual, "text_tensor": text}, str(shard))
    manifest = {
        "tensor_inventory": {
            "visual:image": {
                "dtype": "torch.bfloat16",
                "partition": "context-shards/shard-000-of-001",
                "shape": [3, 8],
                "shard": shard.name,
                "tensor": "visual_tensor",
            },
            "text:instruction": {
                "dtype": "torch.bfloat16",
                "partition": "context-shards/shard-000-of-001",
                "shape": [2, 8],
                "shard": shard.name,
                "tensor": "text_tensor",
            },
        }
    }
    cache = _TokenCache(tmp_path, manifest, device="cpu", mode="lazy_cpu")
    assert cache.mode == "lazy_cpu"
    assert not cache.tensors
    assert torch.equal(cache.visual("image"), visual)
    assert torch.equal(cache.text("instruction"), text)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_lazy_cpu_token_cache_reports_corrupt_shard_path(tmp_path: Path) -> None:
    from scripts.train_set_utility_token_predictor import _TokenCache

    partition = tmp_path / "context-shards" / "shard-000-of-001"
    partition.mkdir(parents=True)
    shard = partition / "chunk-00000.safetensors"
    shard.write_bytes(b"truncated")
    manifest = {
        "tensor_inventory": {
            "visual:image": {
                "dtype": "torch.bfloat16",
                "partition": "context-shards/shard-000-of-001",
                "shape": [3, 8],
                "shard": shard.name,
                "tensor": "visual_tensor",
            }
        }
    }
    with pytest.raises(RuntimeError, match="chunk-00000.safetensors"):
        _TokenCache(tmp_path, manifest, device="cpu", mode="lazy_cpu")


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


def test_decision_supervision_census_counts_only_complete_groups() -> None:
    from scripts.train_set_utility_token_predictor import (
        _decision_supervision_census,
    )

    def state(state_id: str, count: int, subsets: list[list[int]]) -> dict:
        return {
            "candidate_event_step_ids": list(range(1, count + 1)),
            "distance_rows": [
                {"coalition_event_step_ids": subset, "distance": 1.0}
                for subset in subsets
            ],
            "state_id": state_id,
        }

    result = _decision_supervision_census(
        (
            state("long", 17, [[], *[[event] for event in range(1, 18)]]),
            state("incomplete", 9, [[], [1], [2]]),
        )
    )
    assert result["state_count"] == 1
    assert result["long_plus_very_long_state_count"] == 1
    assert result["complete_expansion_group_count"] == 1
    assert result["history_bin_state_counts"] == {"long": 1}


def test_distributed_epoch_shards_reconstruct_padded_global_batches() -> None:
    from scripts.train_set_utility_token_predictor import _distributed_epoch_shard

    states = tuple({"state_id": str(index)} for index in range(10))
    shards = []
    for rank in range(4):
        shard, padding = _distributed_epoch_shard(
            states,
            per_device_batch_size=2,
            rank=rank,
            world_size=4,
        )
        assert padding == 6
        assert len(shard) == 4
        shards.append(shard)
    reconstructed = []
    for step in range(2):
        for rank in range(4):
            reconstructed.extend(shards[rank][step * 2 : (step + 1) * 2])
    assert tuple(reconstructed) == states + states[:6]


def test_distributed_epoch_shards_preserve_accumulated_global_batch() -> None:
    from scripts.train_set_utility_token_predictor import _distributed_epoch_shard

    states = tuple({"state_id": str(index)} for index in range(10))
    shards = []
    for rank in range(2):
        shard, padding = _distributed_epoch_shard(
            states,
            per_device_batch_size=1,
            gradient_accumulation_steps=4,
            rank=rank,
            world_size=2,
        )
        assert padding == 6
        assert len(shard) == 8
        shards.append(shard)
    reconstructed = []
    for accumulation_index in range(4):
        for rank in range(2):
            reconstructed.append(shards[rank][accumulation_index])
    assert tuple(reconstructed) == states[:8]
    reconstructed = []
    for accumulation_index in range(4, 8):
        for rank in range(2):
            reconstructed.append(shards[rank][accumulation_index])
    assert tuple(reconstructed) == states[8:] + states[:6]
