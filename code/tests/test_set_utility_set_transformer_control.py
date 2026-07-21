from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import pytest

from causalcache.set_utility_recovery_checkpoint import RecoveryCheckpointManager
from causalcache.set_utility_set_transformer_control import (
    CONTINUE_TRAINING,
    CONTROL_STATUS,
    EARLY_STOP,
    REQUIRED_BALANCE_AXES,
    WAITING_FOR_TRUTH,
    balanced_control_epoch,
    build_control_epoch_truth,
    control_balancing_inventory,
    control_epoch_barrier,
    group_equal_direct_loss,
    merge_control_truth_schedule,
    prepare_control_examples,
    select_control_checkpoint,
    validate_control_config,
)
from causalcache.set_utility_variable_history import history_bin
from scripts.train_set_utility_set_transformer_control import (
    _epoch_examples,
    _group_sum_backward_scale,
    _optimizer_steps,
    _partition_resume_plans,
    _resume_rank_signature,
    _save_or_verify_epoch_checkpoint,
    _validate_resume_rank_signatures,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


def _config() -> dict[str, object]:
    return {
        "checkpoint_selection": {
            "budgets": [1, 2, 3, 4],
            "primary_metric": "trajectory_equal_true_B1_B4_recovery_macro",
            "require_complete_truth_each_epoch": True,
            "tie_breaker": (
                "long_plus_very_long_trajectory_equal_true_B1_B4_recovery_macro"
            ),
        },
        "comparison_contract": {
            "formal_world_size": 6,
            "shared_global_encoded_state_batch": 12,
        },
        "firewall": {
            "evaluation_access": False,
            "heldout_states_enter_optimizer": False,
            "optimization_role": "train",
            "prior_heldout_exposed_checkpoints_eligible": False,
            "tune_access": False,
        },
        "initialization": {"checkpoint_sha256": None, "mode": "fresh"},
        "input": {
            "contextual_cache_content_sha256": "a" * 64,
            "evaluation_access": False,
            "merged_on_policy_labels_required": True,
            "merged_on_policy_contract": {
                "added_distance_row_count": 10,
                "schedule_content_sha256": "e" * 64,
                "source_name": "direct_on_policy_v1",
                "state_count": 2,
                "total_complete_group_count": 4,
                "train_state_count": 2,
            },
            "required_ancestor_input_content_sha256": "d" * 64,
            "source_manifest_file_sha256": "9" * 64,
            "train_heldout_manifest_content_sha256": "b" * 64,
            "training_input_content_sha256": "c" * 64,
        },
        "schema_version": "1.0.0",
        "status": CONTROL_STATUS,
        "training": {
            "balance_axes": list(REQUIRED_BALANCE_AXES),
            "exact_optimizer_inventory": {
                "complete_group_count": 4,
                "content_sha256": "f" * 64,
                "optimizer_state_count": 2,
            },
            "balancing": {
                "loss_aggregation": (
                    "global_equal_weight_per_sampled_group_across_packing_"
                    "accumulation_and_ddp"
                )
            },
            "epoch_barrier": {
                "mode": "checkpoint_rollout_truth_resume_before_next_optimizer_epoch",
                "posthoc_union_role": "debug_only",
                "require_candidate_complete_truth": True,
                "waiting_status": "WAITING_FOR_HELDOUT_TRUTH",
            },
            "maximum_base_cardinality": 3,
        },
        "variants": {
            "set": {
                "allowed_world_sizes": [1, 2, 4, 6],
                "model": {
                    "family": "set_transformer",
                    "preserve_entity_latents": True,
                },
            }
        },
    }


def _state(
    trajectory_id: str,
    candidate_count: int,
    *,
    stop: bool,
    include_singleton_bases: bool = False,
) -> dict[str, object]:
    candidates = tuple(range(1, candidate_count + 1))
    distances: dict[tuple[int, ...], float] = {(): 10.0}
    singleton_distance = 10.1 if stop else 9.0
    distances.update({(event,): singleton_distance for event in candidates})
    if include_singleton_bases:
        for left in candidates:
            for right in candidates:
                if left < right:
                    distances[(left, right)] = singleton_distance - 0.25
    return {
        "candidate_event_step_ids": list(candidates),
        "distance_rows": [
            {"coalition_event_step_ids": list(subset), "distance": distance}
            for subset, distance in sorted(
                distances.items(), key=lambda item: (len(item[0]), item[0])
            )
        ],
        "role": "train",
        "state_id": f"{trajectory_id}:decision:{candidate_count + 1:03d}",
        "trajectory_id": trajectory_id,
    }


def _heldout_fixture(*, complete: bool) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, object],
]:
    rows = []
    states_by_id = {}
    bins = (("short", 5), ("medium", 9), ("long", 17), ("very_long", 33))
    for name, candidate_count in bins:
        for index in range(64):
            trajectory_id = f"{name}-trajectory-{index:03d}"
            state_id = f"{trajectory_id}:decision:{candidate_count + 1:03d}"
            candidates = tuple(range(1, candidate_count + 1))
            distance_rows = [{"coalition_event_step_ids": [], "distance": 10.0}]
            if complete:
                distance_rows.extend(
                    {
                        "coalition_event_step_ids": [event],
                        "distance": 9.0 + event / 1000.0,
                    }
                    for event in candidates
                )
                distance_rows.append(
                    {
                        "coalition_event_step_ids": list(candidates),
                        "distance": 0.0,
                    }
                )
            states_by_id[state_id] = {
                "candidate_event_step_ids": list(candidates),
                "distance_rows": distance_rows,
                "logical_shard": index,
                "role": "train",
                "state_id": state_id,
                "trajectory_id": trajectory_id,
            }
            rows.append(
                {
                    "candidate_event_ids": list(candidates),
                    "queried_bases": [[]],
                    "selections": {str(budget): [1] for budget in (1, 2, 3, 4)},
                    "state_id": state_id,
                    "trajectory_id": trajectory_id,
                }
            )
    checkpoint_states = [
        {
            "candidate_count": len(state["candidate_event_step_ids"]),
            "history_bin": history_bin(len(state["candidate_event_step_ids"])),
            "state_id": state_id,
            "trajectory_id": state["trajectory_id"],
        }
        for state_id, state in sorted(states_by_id.items())
    ]
    split = {
        "checkpoint_state_ids": [row["state_id"] for row in checkpoint_states],
        "checkpoint_states": checkpoint_states,
        "content_sha256": "b" * 64,
    }
    return rows, states_by_id, split


def test_config_requires_fresh_set_transformer_and_frozen_truth_metric() -> None:
    config = _config()
    assert validate_control_config(config, "set") == config["variants"]["set"]
    missing_inventory = copy.deepcopy(config)
    missing_inventory["training"].pop("exact_optimizer_inventory")
    with pytest.raises(ValueError, match="exact optimizer inventory"):
        validate_control_config(missing_inventory, "set")
    missing_merge = copy.deepcopy(config)
    missing_merge["input"].pop("merged_on_policy_contract")
    with pytest.raises(ValueError, match="merged on-policy contract"):
        validate_control_config(missing_merge, "set")
    leaked = copy.deepcopy(config)
    leaked["initialization"] = {"checkpoint_sha256": "d" * 64, "mode": "full"}
    with pytest.raises(ValueError, match="fresh"):
        validate_control_config(leaked, "set")
    proxy = copy.deepcopy(config)
    proxy["checkpoint_selection"]["primary_metric"] = "heldout_loss"
    with pytest.raises(ValueError, match="metric"):
        validate_control_config(proxy, "set")
    pending = copy.deepcopy(config)
    pending["status"] = "PENDING_INPUT_BINDING"
    with pytest.raises(ValueError, match="not frozen"):
        validate_control_config(pending, "set")
    ancestor_only = copy.deepcopy(config)
    ancestor_only["input"]["training_input_content_sha256"] = ancestor_only[
        "input"
    ]["required_ancestor_input_content_sha256"]
    with pytest.raises(ValueError, match="ancestor"):
        validate_control_config(ancestor_only, "set")
    row_mean = copy.deepcopy(config)
    row_mean["training"]["balancing"]["loss_aggregation"] = "packed_row_mean"
    with pytest.raises(ValueError, match="group-loss"):
        validate_control_config(row_mean, "set")


def test_examples_and_sampler_balance_exact_frozen_axes() -> None:
    states = (
        _state("short-stop", 5, stop=True, include_singleton_bases=True),
        _state("short-go", 5, stop=False, include_singleton_bases=True),
        _state("medium-go", 9, stop=False),
        _state("long-stop", 17, stop=True),
        _state("very-long-go", 33, stop=False),
    )
    examples = prepare_control_examples(
        states, normalization_floor=0.01, maximum_base_cardinality=3
    )
    inventory = control_balancing_inventory(examples)
    assert set(inventory) == set(REQUIRED_BALANCE_AXES)
    assert inventory["stop_all_negative"]["True"] > 0
    assert inventory["stop_all_negative"]["False"] > 0
    assert inventory["base_cardinality"]["1"] > 0
    first = balanced_control_epoch(
        examples, seed=17, sample_count=25, power=0.5, maximum_weight=8.0
    )
    second = balanced_control_epoch(
        examples, seed=17, sample_count=25, power=0.5, maximum_weight=8.0
    )
    assert [row["group_example_id"] for row in first] == [
        row["group_example_id"] for row in second
    ]
    trajectory_counts = Counter(row["trajectory_id"] for row in first)
    assert max(trajectory_counts.values()) - min(trajectory_counts.values()) == 0
    packed = _epoch_examples(
        examples,
        epoch=1,
        seed=17,
        balancing={
            "maximum_groups_per_encoded_state": 4,
            "maximum_weight": 8.0,
            "power": 0.5,
            "samples_per_epoch": 25,
        },
    )
    assert sum(len(row["conditional_groups"]) for row in packed) == 25
    assert _optimizer_steps(
        len(packed), batch_size=1, accumulation=2, world_size=2
    ) == (len(packed) + 3) // 4


def test_truth_plan_waits_for_all_actually_queried_candidate_expansions() -> None:
    rollouts, states, split = _heldout_fixture(complete=False)
    plan = build_control_epoch_truth(
        epoch=1,
        checkpoint={"path": "checkpoints/epoch-0001.safetensors", "sha256": "1" * 64},
        rollout_records=rollouts,
        states_by_id=states,
        split_manifest=split,
        normalization_floor=0.01,
    )
    assert plan["truth_complete"] is False
    assert plan["contract_epoch_truth"]["truth_complete"] is False
    assert plan["contract_epoch_truth"]["observed_row_count"] == 0
    assert plan["selected_missing_pair_count"] == 4 * 256
    schedule = merge_control_truth_schedule([plan])
    assert schedule["status"] == "PENDING_SET_TRANSFORMER_CONTROL_TRUTH"
    assert schedule["model_family"] == "set_transformer_direct_marginal"
    assert schedule["epoch_checkpoints"] == [
        {"checkpoint_sha256": "1" * 64, "epoch": 1}
    ]
    assert schedule["missing_coalition_count"] > 0
    action, decision = control_epoch_barrier(
        [plan], patience=1, minimum_delta=0.0001
    )
    assert action == WAITING_FOR_TRUTH
    assert decision["observed_complete_epoch_count"] == 0


def test_complete_truth_uses_authoritative_reducer_and_immutable_checkpoints() -> None:
    rollouts, states, split = _heldout_fixture(complete=True)
    plans = [
        build_control_epoch_truth(
            epoch=epoch,
            checkpoint={
                "path": f"checkpoints/epoch-{epoch:04d}.safetensors",
                "sha256": str(epoch) * 64,
            },
            rollout_records=rollouts,
            states_by_id=states,
            split_manifest=split,
            normalization_floor=0.01,
        )
        for epoch in (1, 2)
    ]
    assert all(plan["truth_complete"] for plan in plans)
    assert all(plan["contract_epoch_truth"]["truth_complete"] for plan in plans)
    decision = select_control_checkpoint(plans, patience=1, minimum_delta=0.0001)
    assert decision["selected_epoch"] == 1
    assert decision["stopped_early"] is True
    assert decision["stop_after_epoch"] == 2
    action, _ = control_epoch_barrier(
        plans[:1], patience=1, minimum_delta=0.0001
    )
    assert action == CONTINUE_TRAINING
    action, _ = control_epoch_barrier(
        plans, patience=1, minimum_delta=0.0001
    )
    assert action == EARLY_STOP
    duplicate = copy.deepcopy(plans)
    duplicate[1]["checkpoint"]["sha256"] = "1" * 64
    duplicate[1]["contract_epoch_truth"]["checkpoint_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="immutable"):
        select_control_checkpoint(duplicate, patience=1, minimum_delta=0.0001)


def test_fresh_wait_truth_resume_unlocks_exactly_one_next_epoch() -> None:
    rollouts, states, split = _heldout_fixture(complete=False)
    checkpoint = {
        "path": "checkpoints/epoch-0001.safetensors",
        "sha256": "1" * 64,
    }
    pending = build_control_epoch_truth(
        epoch=1,
        checkpoint=checkpoint,
        rollout_records=rollouts,
        states_by_id=states,
        split_manifest=split,
        normalization_floor=0.01,
    )
    action, _ = control_epoch_barrier(
        [pending], patience=2, minimum_delta=0.0001
    )
    assert action == WAITING_FOR_TRUTH

    supplemental = {}
    for state_id, state in states.items():
        candidates = tuple(state["candidate_event_step_ids"])
        supplemental[state_id] = {
            **{(event,): 9.0 + event / 1000.0 for event in candidates},
            candidates: 0.0,
        }
    resumed = build_control_epoch_truth(
        epoch=1,
        checkpoint=checkpoint,
        rollout_records=pending["records"],
        states_by_id=states,
        split_manifest=split,
        normalization_floor=0.01,
        supplemental=supplemental,
    )
    action, selection = control_epoch_barrier(
        [resumed], patience=2, minimum_delta=0.0001
    )
    assert action == CONTINUE_TRAINING
    assert selection["observed_complete_epoch_count"] == 1


def test_crash_window_reuses_immutable_checkpoint_and_one_ahead_plan(
    tmp_path: Path,
) -> None:
    def saver(payload: bytes, destination: Path) -> None:
        destination.write_bytes(payload)

    output_root = tmp_path / "run"
    manager = RecoveryCheckpointManager(
        output_root,
        identity={"input": "a" * 64},
        selection_split="train_trajectory_holdout",
        saver=saver,
    )
    first = _save_or_verify_epoch_checkpoint(
        manager, b"same", epoch=2, output_root=output_root
    )
    assert _save_or_verify_epoch_checkpoint(
        manager, b"same", epoch=2, output_root=output_root
    ) == first
    completed, crash_window = _partition_resume_plans(
        ({"epoch": 1}, {"epoch": 2}), snapshot_epoch=1
    )
    assert [row["epoch"] for row in completed] == [1]
    assert [row["epoch"] for row in crash_window] == [2]
    with pytest.raises(ValueError, match="inventory drifted"):
        _partition_resume_plans(
            ({"epoch": 1}, {"epoch": 2}, {"epoch": 3}), snapshot_epoch=1
        )


def test_ddp_resume_signature_validation_fails_fast_on_rank_drift() -> None:
    common = {
        "epoch": 3,
        "model_sha256": "a" * 64,
        "optimizer_sha256": "b" * 64,
        "optimizer_steps": [17],
        "scheduler_last_epoch": 17,
        "scheduler_sha256": "c" * 64,
        "scheduler_step_count": 18,
        "status": "ok",
    }
    assert _validate_resume_rank_signatures(
        ({**common, "rank": 0}, {**common, "rank": 1}), world_size=2
    )["epoch"] == 3
    drifted = ({**common, "rank": 0}, {**common, "rank": 1, "epoch": 2})
    with pytest.raises(ValueError, match="rank drift"):
        _validate_resume_rank_signatures(drifted, world_size=2)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_resume_signature_hashes_model_optimizer_and_scheduler_state() -> None:
    snapshot = {
        "epoch": 2,
        "model": {"weight": torch.arange(4, dtype=torch.bfloat16)},
        "optimizer": {
            "param_groups": [{"lr": 0.1, "params": [0]}],
            "state": {0: {"exp_avg": torch.ones(4), "step": torch.tensor(7.0)}},
        },
        "scheduler": {"_step_count": 8, "last_epoch": 7},
    }
    first = _resume_rank_signature(snapshot, rank=0, torch=torch)
    second = _resume_rank_signature(copy.deepcopy(snapshot), rank=1, torch=torch)
    assert _validate_resume_rank_signatures((first, second), world_size=2)[
        "optimizer_steps"
    ] == [7]
    snapshot["model"]["weight"][0] = 9
    drifted = _resume_rank_signature(snapshot, rank=1, torch=torch)
    with pytest.raises(ValueError, match="rank drift"):
        _validate_resume_rank_signatures((first, drifted), world_size=2)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is unavailable")
def test_group_sum_scaling_matches_true_global_group_mean_gradient() -> None:
    loss_config = {
        "conditional_listwise": 0.0,
        "conditional_marginal": 1.0,
        "conditional_marginal_smooth_l1_beta": 0.5,
        "decision_regret": 0.0,
        "decision_temperature": 1.0,
        "sign_classification": 0.0,
    }

    def batch(active_groups: int, target: float) -> dict[str, object]:
        group_mask = torch.arange(16)[None, :] < active_groups
        action_mask = group_mask[:, :, None].expand(-1, -1, 2).clone()
        targets = torch.zeros((1, 16, 2), dtype=torch.float32)
        targets[:, :, 1] = target
        return {
            "conditional_action_mask": action_mask,
            "conditional_group_mask": group_mask,
            "conditional_normalized_targets": targets,
            "conditional_raw_targets": targets,
            "scales": torch.ones(1),
        }

    rank_batches = (batch(1, -1.0), batch(16, 1.0))
    sharded_parameter = torch.tensor(0.0, requires_grad=True)
    sharded_losses = []
    for value in rank_batches:
        predictions = torch.stack(
            (
                torch.zeros((1, 16)),
                sharded_parameter.expand(1, 16),
            ),
            dim=2,
        )
        loss_sum, _ = group_equal_direct_loss(
            predictions,
            value,
            loss_config=loss_config,
            reduction="sum",
            torch=torch,
        )
        sharded_losses.append(loss_sum)
    scale = _group_sum_backward_scale(global_group_count=17.0, world_size=2)
    simulated_ddp_loss = sum(loss * scale for loss in sharded_losses) / 2.0
    simulated_ddp_loss.backward()

    reference_parameter = torch.tensor(0.0, requires_grad=True)
    combined = {
        key: torch.cat(tuple(value[key] for value in rank_batches), dim=0)
        for key in rank_batches[0]
    }
    reference_predictions = torch.stack(
        (
            torch.zeros((2, 16)),
            reference_parameter.expand(2, 16),
        ),
        dim=2,
    )
    reference_loss, _ = group_equal_direct_loss(
        reference_predictions,
        combined,
        loss_config=loss_config,
        reduction="mean",
        torch=torch,
    )
    reference_loss.backward()
    assert sharded_parameter.grad == pytest.approx(reference_parameter.grad)
    assert float(reference_parameter.grad) < -0.8
