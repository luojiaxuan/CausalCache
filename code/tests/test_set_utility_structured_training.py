from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from causalcache.set_utility_recovery_checkpoint import BudgetRecoverySummary
from causalcache.set_utility_structured_training import (
    EpochRecovery,
    apply_split_manifest,
    balanced_group_epoch,
    balancing_inventory,
    build_epoch_truth_plan,
    candidate_complete_coalitions,
    conditional_groups,
    evaluate_epoch_truth,
    merge_epoch_truth_plans,
    pack_group_examples,
    prepare_group_examples,
    replay_recovery_early_stopping,
)
from scripts.train_set_utility_structured_marginal import (
    STRUCTURED_CONTROL_STATUS,
    _contract_epoch_truth,
    _formal_epoch_action,
    _load_split_manifest,
    _publish_truth_schedule,
    _read_existing_epoch_plan,
    _save_or_verify_epoch_checkpoint,
    _signed,
    _validate_config,
    _write_atomic,
)
from causalcache.set_utility_recovery_checkpoint import RecoveryCheckpointManager


def _state(
    state_id: str = "t0:decision:006",
    trajectory_id: str = "t0",
) -> dict:
    return {
        "candidate_event_step_ids": [1, 2, 3, 4, 5],
        "distance_rows": [
            {"coalition_event_step_ids": [], "distance": 1.0},
            {"coalition_event_step_ids": [1], "distance": 0.6},
            {"coalition_event_step_ids": [2], "distance": 0.7},
            {"coalition_event_step_ids": [3], "distance": 1.1},
            {"coalition_event_step_ids": [4], "distance": 1.0},
            {"coalition_event_step_ids": [5], "distance": 0.8},
            {"coalition_event_step_ids": [1, 2], "distance": 0.5},
            {"coalition_event_step_ids": [1, 3], "distance": 0.7},
            {"coalition_event_step_ids": [1, 4], "distance": 0.6},
            {"coalition_event_step_ids": [1, 5], "distance": 0.55},
            {"coalition_event_step_ids": [1, 2, 3, 4, 5], "distance": 0.0},
        ],
        "logical_shard": 0,
        "role": "train",
        "state_id": state_id,
        "trajectory_id": trajectory_id,
    }


def test_conditional_groups_keep_only_candidate_complete_bases() -> None:
    groups = conditional_groups(
        _state(), normalization_floor=0.01, maximum_base_cardinality=3
    )
    assert [group["selected_event_step_ids"] for group in groups] == [(), (1,)]
    assert groups[0]["normalized_marginals"] == pytest.approx(
        (0.0, 0.4, 0.3, -0.1, 0.0, 0.2)
    )
    assert groups[1]["normalized_marginals"] == pytest.approx(
        (0.0, 0.0, 0.1, -0.1, 0.0, 0.05)
    )


def test_stop_balance_treats_exact_zero_best_marginal_as_stop() -> None:
    state = _state()
    state["distance_rows"] = [
        {"coalition_event_step_ids": [], "distance": 1.0},
        {"coalition_event_step_ids": [1], "distance": 1.0},
        {"coalition_event_step_ids": [2], "distance": 1.1},
        {"coalition_event_step_ids": [3], "distance": 1.2},
        {"coalition_event_step_ids": [4], "distance": 1.3},
        {"coalition_event_step_ids": [5], "distance": 1.4},
        {"coalition_event_step_ids": [1, 2, 3, 4, 5], "distance": 0.0},
    ]
    examples = prepare_group_examples(
        (state,),
        normalization_floor=0.01,
        maximum_base_cardinality=3,
        epsilon=1e-6,
    )
    assert examples[0]["conditional_groups"][0]["balance"][
        "stop_all_negative"
    ] is True


def test_split_manifest_is_trajectory_disjoint_and_state_bounded() -> None:
    states = [
        _state(f"t{trajectory}:decision:{step:03d}", f"t{trajectory}")
        for trajectory in range(4)
        for step in (6, 7)
    ]
    optimization, heldout = apply_split_manifest(
        states,
        {
            "optimizer_trajectory_ids": ["t0", "t1", "t2"],
            "heldout_trajectory_ids": ["t3"],
            "checkpoint_state_ids": ["t3:decision:007"],
        },
    )
    assert {row["trajectory_id"] for row in optimization} == {"t0", "t1", "t2"}
    assert [row["state_id"] for row in heldout] == ["t3:decision:007"]
    broken = {
        "optimizer_trajectory_ids": ["t0", "t1", "t2", "t3"],
        "heldout_trajectory_ids": ["t3"],
    }
    with pytest.raises(ValueError, match="invalid"):
        apply_split_manifest(states, broken)


def test_balanced_group_epoch_is_deterministic_and_covers_all_axes() -> None:
    states = [_state(f"t{i}:decision:006", f"t{i}") for i in range(4)]
    examples = prepare_group_examples(
        states,
        normalization_floor=0.01,
        maximum_base_cardinality=3,
        epsilon=1e-6,
    )
    inventory = balancing_inventory(examples)
    assert set(inventory) == {
        "cardinality",
        "class",
        "history",
        "stop_all_negative",
    }
    first = balanced_group_epoch(
        examples,
        seed=9,
        sample_count=20,
        axes=("class", "cardinality", "history", "stop_all_negative"),
        power=0.5,
        maximum_weight=4.0,
    )
    second = balanced_group_epoch(
        examples,
        seed=9,
        sample_count=20,
        axes=("class", "cardinality", "history", "stop_all_negative"),
        power=0.5,
        maximum_weight=4.0,
    )
    assert [row["group_example_id"] for row in first] == [
        row["group_example_id"] for row in second
    ]
    trajectory_counts = {
        trajectory: sum(row["trajectory_id"] == trajectory for row in first)
        for trajectory in {row["trajectory_id"] for row in first}
    }
    assert set(trajectory_counts.values()) == {5}
    packed = pack_group_examples(first, maximum_groups_per_state=3, seed=19)
    assert Counter(
        group_id for row in packed for group_id in row["group_example_ids"]
    ) == Counter(row["group_example_id"] for row in first)
    assert all(len(row["conditional_groups"]) <= 3 for row in packed)
    assert all(
        sum(not group["selected_event_step_ids"] for group in row["conditional_groups"])
        == 1
        for row in packed
    )
    assert sum(sum(row["conditional_group_weights"]) for row in packed) == pytest.approx(
        len(first)
    )
    packed_wide = pack_group_examples(first, maximum_groups_per_state=6, seed=19)
    assert sum(
        sum(row["conditional_group_weights"]) for row in packed_wide
    ) == pytest.approx(len(first))


@pytest.mark.parametrize("sample_empty", [False, True])
def test_packing_preserves_sampled_empty_group_multiplicity(
    sample_empty: bool,
) -> None:
    empty, singleton = prepare_group_examples(
        (_state(),),
        normalization_floor=0.01,
        maximum_base_cardinality=3,
        epsilon=1e-6,
    )
    sampled = (
        (empty, empty, singleton, singleton, singleton)
        if sample_empty
        else (singleton,) * 5
    )
    packed = pack_group_examples(
        sampled, maximum_groups_per_state=3, seed=23
    )

    assert Counter(
        group_id for row in packed for group_id in row["group_example_ids"]
    ) == Counter(row["group_example_id"] for row in sampled)
    assert sum(sum(row["conditional_group_weights"]) for row in packed) == len(
        sampled
    )
    weighted_empty = 0.0
    for row in packed:
        assert len(row["conditional_groups"]) == len(
            row["conditional_group_weights"]
        )
        for group, weight in zip(
            row["conditional_groups"],
            row["conditional_group_weights"],
            strict=True,
        ):
            assert weight in {0.0, 1.0}
            if weight == 0.0:
                assert not group["selected_event_step_ids"]
            elif not group["selected_event_step_ids"]:
                weighted_empty += weight
    assert weighted_empty == (2.0 if sample_empty else 0.0)


def _rollout(
    *,
    epoch: int,
    bases: list[list[int]],
    selections: dict[str, list[int]],
) -> dict:
    return {
        "candidate_event_ids": [1, 2, 3, 4, 5],
        "queried_bases": bases,
        "selections": selections,
        "state_id": "t0:decision:006",
        "trajectory_id": "t0",
    }


def test_epoch_truth_plan_is_candidate_complete_and_mergeable() -> None:
    state = _state()
    records = [_rollout(epoch=1, bases=[[]], selections={str(b): [1] for b in range(1, 5)})]
    plan1 = build_epoch_truth_plan(
        epoch=1,
        checkpoint={"sha256": "a" * 64},
        rollout_records=records,
        states_by_id={state["state_id"]: state},
    )
    assert plan1["truth_complete"] is True
    expected = candidate_complete_coalitions((1, 2, 3, 4, 5), ((),))
    assert {tuple(value) for value in plan1["records"][0]["desired_coalitions"]} == set(expected)

    state["distance_rows"] = [
        row
        for row in state["distance_rows"]
        if row["coalition_event_step_ids"] != [1, 5]
    ]
    plan2 = build_epoch_truth_plan(
        epoch=2,
        checkpoint={"sha256": "b" * 64},
        rollout_records=[
            _rollout(
                epoch=2,
                bases=[[], [1]],
                selections={"1": [1], "2": [1, 2], "3": [1, 2], "4": [1, 2]},
            )
        ],
        states_by_id={state["state_id"]: state},
    )
    assert plan2["truth_complete"] is False
    merged = merge_epoch_truth_plans((plan1, plan2))
    assert merged["epochs"] == [1, 2]
    assert merged["missing_coalition_count"] == len(
        plan2["records"][0]["missing_coalitions"]
    )


def test_deepsets_truth_schedule_binds_epoch_checkpoints(tmp_path: Path) -> None:
    state = _state()
    plan = build_epoch_truth_plan(
        epoch=1,
        checkpoint={"sha256": "a" * 64},
        rollout_records=[
            _rollout(
                epoch=1,
                bases=[[]],
                selections={str(budget): [1] for budget in range(1, 5)},
            )
        ],
        states_by_id={state["state_id"]: state},
    )
    schedule = _publish_truth_schedule(tmp_path, (plan,))
    assert schedule["model_family"] == "deepsets_structured_marginal"
    assert schedule["epoch_checkpoints"] == [
        {"checkpoint_sha256": "a" * 64, "epoch": 1}
    ]


def test_true_recovery_waits_for_supplemental_rollout_truth() -> None:
    state = _state()
    state["distance_rows"] = [
        row
        for row in state["distance_rows"]
        if row["coalition_event_step_ids"] != [1, 5]
    ]
    rollout = _rollout(
        epoch=1,
        bases=[[], [1]],
        selections={"1": [1], "2": [1, 2], "3": [1, 2], "4": [1, 2]},
    )
    plan = build_epoch_truth_plan(
        epoch=1,
        checkpoint={"sha256": "a" * 64},
        rollout_records=[rollout],
        states_by_id={state["state_id"]: state},
    )
    recovery, missing = evaluate_epoch_truth(
        plan,
        states_by_id={state["state_id"]: state},
        normalization_floor=0.01,
    )
    assert missing == ()
    assert recovery is not None
    assert recovery.recovery.budget_means == pytest.approx(
        {1: 0.4, 2: 0.5, 3: 0.5, 4: 0.5}
    )
    assert plan["truth_complete"] is False
    supplemental = {
        state["state_id"]: {
            tuple(value): 0.4
            for value in plan["records"][0]["missing_coalitions"]
        }
    }
    rebuilt = build_epoch_truth_plan(
        epoch=1,
        checkpoint={"sha256": "a" * 64},
        rollout_records=[rollout],
        states_by_id={state["state_id"]: state},
        supplemental=supplemental,
    )
    assert rebuilt["truth_complete"] is True


def test_frozen_contract_waits_for_candidate_complete_truth() -> None:
    state = _state()
    candidates = list(range(1, 18))
    state["candidate_event_step_ids"] = candidates
    state["distance_rows"] = [
        {"coalition_event_step_ids": [], "distance": 1.0},
        *(
            {"coalition_event_step_ids": [event], "distance": 0.6}
            for event in candidates
        ),
        *(
            {"coalition_event_step_ids": [1, event], "distance": 0.5}
            for event in candidates[1:-1]
        ),
        {"coalition_event_step_ids": candidates, "distance": 0.0},
    ]
    rollout = _rollout(
        epoch=1,
        bases=[[], [1]],
        selections={"1": [1], "2": [1, 2], "3": [1, 2], "4": [1, 2]},
    )
    rollout["candidate_event_ids"] = candidates
    plan = build_epoch_truth_plan(
        epoch=1,
        checkpoint={"sha256": "a" * 64},
        rollout_records=[rollout],
        states_by_id={state["state_id"]: state},
    )
    assert plan["truth_complete"] is False
    manifest = {
        "checkpoint_states": [
            {
                "history_bin": "long",
                "state_id": state["state_id"],
                "trajectory_id": state["trajectory_id"],
            }
        ],
        "content_sha256": "b" * 64,
    }
    waiting = _contract_epoch_truth(
        plan,
        split_manifest=manifest,
        states_by_id={state["state_id"]: state},
        normalization_floor=0.01,
        supplemental={},
    )
    assert waiting["truth_complete"] is False
    assert waiting["candidate_complete_truth"] is False
    assert waiting["observed_row_count"] == 0
    assert waiting["selected_truth_row_count"] == 4

    supplemental = {
        state["state_id"]: {
            tuple(value): 0.4
            for value in plan["records"][0]["missing_coalitions"]
        }
    }
    complete_plan = build_epoch_truth_plan(
        epoch=1,
        checkpoint={"sha256": "a" * 64},
        rollout_records=[rollout],
        states_by_id={state["state_id"]: state},
        supplemental=supplemental,
    )
    complete = _contract_epoch_truth(
        complete_plan,
        split_manifest=manifest,
        states_by_id={state["state_id"]: state},
        normalization_floor=0.01,
        supplemental=supplemental,
    )
    assert complete["truth_complete"] is True
    assert complete["candidate_complete_truth"] is True
    assert complete["observed_row_count"] == 4


def test_formal_epoch_barrier_waits_before_any_further_optimization() -> None:
    assert (
        _formal_epoch_action(
            {
                "decision_ready": False,
                "status": "WAITING_FOR_COMPLETE_TRAIN_HELDOUT_EPOCH_TRUTH",
            }
        )
        == "wait_for_heldout_truth"
    )
    assert (
        _formal_epoch_action(
            {
                "decision_ready": True,
                "status": "CONTINUE_TRAIN_HELDOUT_CHECKPOINT_SELECTION",
                "stopped_early": False,
            }
        )
        == "continue"
    )
    assert (
        _formal_epoch_action(
            {
                "decision_ready": True,
                "status": "EARLY_STOP_TRAIN_HELDOUT_CHECKPOINT_SELECTION",
                "stopped_early": True,
            }
        )
        == "early_stop"
    )


def _epoch(epoch: int, macro: float, long: float) -> EpochRecovery:
    return EpochRecovery(
        epoch=epoch,
        recovery=BudgetRecoverySummary(
            budget_means={budget: macro for budget in range(1, 5)},
            macro_b1_b4=macro,
            state_count=4,
            trajectory_count=2,
        ),
        long_plus_macro=long,
    )


def test_early_stopping_uses_macro_then_long_tie_and_ignores_later_epochs() -> None:
    result = replay_recovery_early_stopping(
        (
            _epoch(1, 0.50, 0.50),
            _epoch(2, 0.60, 0.60),
            _epoch(3, 0.60005, 0.70),
            _epoch(4, 0.55, 0.80),
            _epoch(5, 0.54, 0.90),
            _epoch(6, 0.90, 0.90),
        ),
        patience=2,
        minimum_delta=0.01,
        long_tie_tolerance=0.001,
    )
    assert result["best_epoch"] == 3
    assert result["stopped_epoch"] == 5
    assert result["eligible_epoch_count"] == 5
    assert [row["decision"] for row in result["trace"]] == [
        "BEST_MACRO",
        "BEST_MACRO",
        "BEST_LONG_TIE_BREAK",
        "STALE",
        "STALE",
    ]


def test_config_requires_true_recovery_and_all_balancing_axes() -> None:
    config = {
        "checkpoint_selection": {
            "budgets": [1, 2, 3, 4],
            "incomplete_epoch_action": "wait_without_selecting_or_advancing_patience",
            "metric": "trajectory_equal_true_B1_B4_normalized_recovery_macro",
        },
        "firewall": {"allowed_input_role": "train", "final_evaluation_access": False},
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
            "required_ancestor_input_content_sha256": "b" * 64,
            "source_manifest_file_sha256": "9" * 64,
            "train_heldout_manifest_content_sha256": "c" * 64,
            "training_input_content_sha256": "d" * 64,
        },
        "schema_version": "1.0.0",
        "status": STRUCTURED_CONTROL_STATUS,
        "training": {
            "exact_optimizer_inventory": {
                "complete_group_count": 4,
                "content_sha256": "f" * 64,
                "optimizer_state_count": 2,
            },
            "epoch_barrier": {
                "mode": "checkpoint_rollout_truth_resume_before_next_optimizer_epoch",
                "posthoc_union_role": "debug_only",
                "require_candidate_complete_truth": True,
                "waiting_status": "WAITING_FOR_HELDOUT_TRUTH",
            },
            "balancing": {
                "axes": ["class", "cardinality", "history", "stop_all_negative"],
                "loss_aggregation": (
                    "global_equal_weight_per_candidate_complete_group_across_"
                    "packing_accumulation_and_ddp"
                ),
                "maximum_groups_per_encoded_state": 8,
            }
        },
        "variants": {"small": {"encoder": {}, "head": {}}},
    }
    assert _validate_config(config, "small") == config["variants"]["small"]
    proxy = copy.deepcopy(config)
    proxy["checkpoint_selection"]["metric"] = "heldout_loss"
    with pytest.raises(ValueError, match="metric"):
        _validate_config(proxy, "small")
    missing_axis = copy.deepcopy(config)
    missing_axis["training"]["balancing"]["axes"].remove("stop_all_negative")
    with pytest.raises(ValueError, match="balancing"):
        _validate_config(missing_axis, "small")
    row_mean = copy.deepcopy(config)
    row_mean["training"]["balancing"]["loss_aggregation"] = "packed_row_mean"
    with pytest.raises(ValueError, match="group-loss"):
        _validate_config(row_mean, "small")
    missing_barrier = copy.deepcopy(config)
    missing_barrier["training"].pop("epoch_barrier")
    with pytest.raises(ValueError, match="epoch barrier"):
        _validate_config(missing_barrier, "small")
    missing_inventory = copy.deepcopy(config)
    missing_inventory["training"].pop("exact_optimizer_inventory")
    with pytest.raises(ValueError, match="exact optimizer inventory"):
        _validate_config(missing_inventory, "small")
    missing_merge = copy.deepcopy(config)
    missing_merge["input"].pop("merged_on_policy_contract")
    with pytest.raises(ValueError, match="merged on-policy contract"):
        _validate_config(missing_merge, "small")
    pending = copy.deepcopy(config)
    pending["status"] = "PENDING_MERGED_ON_POLICY_INPUT_BINDING"
    with pytest.raises(ValueError, match="not frozen"):
        _validate_config(pending, "small")
    ancestor_only = copy.deepcopy(config)
    ancestor_only["input"]["training_input_content_sha256"] = ancestor_only[
        "input"
    ]["required_ancestor_input_content_sha256"]
    with pytest.raises(ValueError, match="ancestor"):
        _validate_config(ancestor_only, "small")


def test_split_manifest_binding_is_external_and_content_addressed(tmp_path) -> None:
    from causalcache.set_utility_train_heldout_contract import sha256_json

    manifest = {
        "checkpoint_selection": {
            "budgets": [1, 2, 3, 4],
            "minimum_delta": 0.01,
            "patience": 2,
            "require_complete_truth_each_epoch": True,
        },
        "firewall": {
            "all_heldout_trajectory_states_excluded_from_optimizer": True,
            "allowed_role": "train",
            "evaluation_access": False,
            "tune_access": False,
        },
        "optimizer_trajectory_ids": ["train"],
        "heldout_trajectory_ids": ["heldout"],
    }
    manifest["content_sha256"] = sha256_json(manifest)
    path = tmp_path / "split.json"
    import json

    path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "checkpoint_selection": {
            "long_tie_tolerance": 0.01,
            "minimum_delta": 0.01,
            "patience": 2,
        },
        "input": {
            "train_heldout_manifest_content_sha256": manifest["content_sha256"]
        },
    }
    assert _load_split_manifest(path, config)["content_sha256"] == manifest[
        "content_sha256"
    ]
    config["input"]["train_heldout_manifest_content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding drifted"):
        _load_split_manifest(path, config)


def test_committed_structured_config_is_frozen_to_merged_on_policy_input() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "code/configs/causalcache_set_utility_structured_marginal_v1.json")
        .read_text(encoding="utf-8")
    )
    manifest_path = root / "data/manifests/set_utility_train_heldout_v2.json"
    variant = "structured_pairwise_deepsets_d256_l16_r2_h128_p32_lr3e4"
    _validate_config(config, variant)
    assert config["status"] == "FROZEN_STRUCTURED_DIRECT_MARGINAL_CONTROL"
    assert config["input"]["training_input_content_sha256"] == (
        "6c9243a2a2846180f717ea59e3692a3005ae153b8d206f7ad1ec4a8bf1a1bfad"
    )
    assert config["input"]["merged_on_policy_labels_required"] is True
    inventory = config["training"]["exact_optimizer_inventory"]
    assert inventory["content_sha256"] == (
        "b7452b1bad141d96407bf80c62b1d4778461af260f9837852cd4d8e480646109"
    )
    assert inventory["optimizer_trajectory_count"] == 900
    assert inventory["optimizer_state_count"] == 9287
    assert inventory["complete_group_count"] == 84441
    assert sum(inventory["base_cardinality_counts"].values()) == 84441
    assert sum(inventory["history_bin_counts"].values()) == 84441
    assert sum(inventory["joint_stratum_counts"].values()) == 84441
    assert sum(inventory["stop_all_negative_counts"].values()) == 84441
    assert config["training"]["minimum_inventory"] == {
        "complete_group_count": 60000,
        "heldout_state_count": 256,
        "optimizer_state_count": 8000,
    }
    manifest = _load_split_manifest(manifest_path, config)
    assert manifest is not None
    assert manifest["census"]["source_train_state_count"] == 10658
    assert manifest["census"]["optimizer_state_count"] == 9287
    assert manifest["census"]["checkpoint_state_count"] == 256


def test_epoch_checkpoint_and_rollout_resume_are_idempotent(tmp_path: Path) -> None:
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
        manager, b"same-weights", epoch=1, output_root=output_root
    )
    assert _save_or_verify_epoch_checkpoint(
        manager, b"same-weights", epoch=1, output_root=output_root
    ) == first
    with pytest.raises(ValueError, match="replay differs"):
        _save_or_verify_epoch_checkpoint(
            manager, b"different-weights", epoch=1, output_root=output_root
        )

    plan = {
        "checkpoint": first,
        "epoch": 1,
        "records": [],
        "schema_version": "causalcache.structured_epoch_truth_plan.v1",
    }
    _write_atomic(output_root / "heldout-rollouts/epoch-0001.json", _signed(plan))
    assert _read_existing_epoch_plan(
        output_root, epoch=1, checkpoint=first
    ) is not None
