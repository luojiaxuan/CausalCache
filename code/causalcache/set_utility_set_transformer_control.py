"""Contract adapter for the direct Set Transformer marginal control."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from causalcache.set_utility_structured_training import (
    build_epoch_truth_plan,
    conditional_groups,
    distance_table,
    merge_epoch_truth_plans,
)
from causalcache.set_utility_train_heldout_contract import (
    BUDGETS,
    COMPLETE_TRUTH_STATUS,
    reduce_epoch_truth,
    select_checkpoint_from_epoch_truth,
    sha256_json,
)
from causalcache.set_utility_variable_history import history_bin


CONTROL_STATUS = "FROZEN_SET_TRANSFORMER_DIRECT_MARGINAL_CONTROL"
WAITING_FOR_TRUTH = "WAITING_FOR_HELDOUT_TRUTH"
CONTINUE_TRAINING = "CONTINUE_AFTER_HELDOUT_TRUTH"
EARLY_STOP = "EARLY_STOP_AFTER_HELDOUT_TRUTH"
REQUIRED_BALANCE_AXES = (
    "base_cardinality",
    "history_bin",
    "stop_all_negative",
)


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def validate_control_config(
    config: Mapping[str, Any], variant_name: str
) -> dict[str, Any]:
    """Fail closed unless the direct control shares the frozen truth contract."""
    if config.get("schema_version") != "1.0.0":
        raise ValueError("Set Transformer control config schema drifted")
    if config.get("status") != CONTROL_STATUS:
        raise ValueError("Set Transformer control config is not frozen for execution")
    input_contract = config.get("input")
    training = config.get("training")
    selection = config.get("checkpoint_selection")
    firewall = config.get("firewall")
    initialization = config.get("initialization")
    variants = config.get("variants")
    comparison = config.get("comparison_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (
            input_contract,
            training,
            selection,
            firewall,
            initialization,
            variants,
            comparison,
        )
    ):
        raise ValueError("Set Transformer control config sections are incomplete")
    variant = variants.get(variant_name)
    if not isinstance(variant, Mapping):
        raise ValueError("unknown Set Transformer control variant")
    for key in (
        "training_input_content_sha256",
        "required_ancestor_input_content_sha256",
        "contextual_cache_content_sha256",
        "train_heldout_manifest_content_sha256",
        "source_manifest_file_sha256",
    ):
        _sha256(input_contract.get(key), key)
    if input_contract["training_input_content_sha256"] == input_contract[
        "required_ancestor_input_content_sha256"
    ]:
        raise ValueError("Set Transformer still points to the ancestor input")
    if input_contract.get("evaluation_access") is not False:
        raise ValueError("Set Transformer control may not access evaluation")
    if input_contract.get("merged_on_policy_labels_required") is not True:
        raise ValueError("Set Transformer control requires merged on-policy labels")
    merged = input_contract.get("merged_on_policy_contract")
    if not isinstance(merged, Mapping):
        raise ValueError("Set Transformer merged on-policy contract is absent")
    if merged.get("source_name") != "direct_on_policy_v1":
        raise ValueError("Set Transformer merged on-policy source drifted")
    _sha256(
        merged.get("schedule_content_sha256"),
        "Set Transformer on-policy schedule content SHA256",
    )
    for key in (
        "added_distance_row_count",
        "state_count",
        "total_complete_group_count",
        "train_state_count",
    ):
        if type(merged.get(key)) is not int or int(merged[key]) <= 0:
            raise ValueError(f"Set Transformer merged on-policy {key} is invalid")
    if tuple(selection.get("budgets", ())) != BUDGETS:
        raise ValueError("Set Transformer checkpoint selection requires B1--B4")
    if selection.get("primary_metric") != (
        "trajectory_equal_true_B1_B4_recovery_macro"
    ):
        raise ValueError("Set Transformer checkpoint metric drifted")
    if selection.get("tie_breaker") != (
        "long_plus_very_long_trajectory_equal_true_B1_B4_recovery_macro"
    ):
        raise ValueError("Set Transformer checkpoint tie-breaker drifted")
    if selection.get("require_complete_truth_each_epoch") is not True:
        raise ValueError("Set Transformer checkpoint truth must be complete")
    if tuple(training.get("balance_axes", ())) != REQUIRED_BALANCE_AXES:
        raise ValueError("Set Transformer balance axes drifted")
    exact_inventory = training.get("exact_optimizer_inventory")
    if not isinstance(exact_inventory, Mapping):
        raise ValueError("Set Transformer exact optimizer inventory is not frozen")
    _sha256(
        exact_inventory.get("content_sha256"),
        "Set Transformer optimizer inventory content SHA256",
    )
    if (
        type(exact_inventory.get("optimizer_state_count")) is not int
        or type(exact_inventory.get("complete_group_count")) is not int
    ):
        raise ValueError("Set Transformer exact optimizer inventory counts are invalid")
    if training.get("balancing", {}).get("loss_aggregation") != (
        "global_equal_weight_per_sampled_group_across_packing_accumulation_and_ddp"
    ):
        raise ValueError("Set Transformer group-loss aggregation drifted")
    if int(training.get("maximum_base_cardinality", -1)) != 3:
        raise ValueError("Set Transformer control must train base cardinalities 0--3")
    barrier = training.get("epoch_barrier")
    if (
        not isinstance(barrier, Mapping)
        or barrier.get("mode")
        != "checkpoint_rollout_truth_resume_before_next_optimizer_epoch"
        or barrier.get("require_candidate_complete_truth") is not True
        or barrier.get("waiting_status") != WAITING_FOR_TRUTH
        or barrier.get("posthoc_union_role") != "debug_only"
    ):
        raise ValueError("Set Transformer per-epoch truth barrier drifted")
    if firewall.get("optimization_role") != "train":
        raise ValueError("Set Transformer optimization role must be train")
    if firewall.get("heldout_states_enter_optimizer") is not False:
        raise ValueError("Set Transformer heldout states may not enter optimization")
    if firewall.get("prior_heldout_exposed_checkpoints_eligible") is not False:
        raise ValueError("heldout-exposed checkpoints are forbidden")
    if firewall.get("tune_access") is not False or firewall.get(
        "evaluation_access"
    ) is not False:
        raise ValueError("Set Transformer control firewall is open")
    if initialization.get("mode") != "fresh" or initialization.get(
        "checkpoint_sha256"
    ) is not None:
        raise ValueError("Set Transformer control must initialize fresh after resplitting")
    model = variant.get("model")
    if (
        not isinstance(model, Mapping)
        or model.get("family") != "set_transformer"
        or model.get("preserve_entity_latents") is not True
    ):
        raise ValueError("direct marginal control must use entity-latent Set Transformer")
    allowed_world_sizes = tuple(variant.get("allowed_world_sizes", ()))
    if (
        not allowed_world_sizes
        or any(type(value) is not int or value <= 0 for value in allowed_world_sizes)
    ):
        raise ValueError("Set Transformer allowed world sizes are invalid")
    if (
        comparison.get("formal_world_size") != 6
        or comparison.get("shared_global_encoded_state_batch") != 12
        or 6 not in allowed_world_sizes
    ):
        raise ValueError("Set Transformer comparison batch contract drifted")
    return dict(variant)


def prepare_control_examples(
    states: Sequence[Mapping[str, Any]],
    *,
    normalization_floor: float,
    maximum_base_cardinality: int,
) -> tuple[dict[str, Any], ...]:
    """Flatten candidate-complete groups and attach the three frozen strata."""
    examples = []
    for state in states:
        groups = conditional_groups(
            state,
            normalization_floor=normalization_floor,
            maximum_base_cardinality=maximum_base_cardinality,
        )
        candidates = tuple(state["candidate_event_step_ids"])
        bin_name = history_bin(len(candidates))
        for group_index, group in enumerate(groups):
            valid = tuple(
                value
                for value, keep in zip(
                    group["normalized_marginals"][1:],
                    group["action_mask"][1:],
                    strict=True,
                )
                if keep
            )
            if not valid:
                raise ValueError("conditional group has no remaining candidate")
            enriched = dict(group)
            enriched["balance"] = {
                "base_cardinality": len(group["selected_event_step_ids"]),
                "history_bin": bin_name,
                "stop_all_negative": max(valid) <= 0.0,
            }
            example = dict(state)
            example["conditional_groups"] = (enriched,)
            example["group_example_id"] = (
                f"{state['state_id']}:direct-g{group_index:04d}"
            )
            examples.append(example)
    if not examples:
        raise ValueError("Set Transformer optimization split has no complete groups")
    identities = [row["group_example_id"] for row in examples]
    if len(identities) != len(set(identities)):
        raise ValueError("Set Transformer group identities are duplicated")
    return tuple(examples)


def control_balancing_inventory(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts = {axis: Counter() for axis in REQUIRED_BALANCE_AXES}
    for example in examples:
        groups = tuple(example.get("conditional_groups", ()))
        if len(groups) != 1:
            raise ValueError("control balancing requires one group per example")
        balance = groups[0].get("balance")
        if not isinstance(balance, Mapping) or set(balance) != set(
            REQUIRED_BALANCE_AXES
        ):
            raise ValueError("control example balance metadata drifted")
        for axis in REQUIRED_BALANCE_AXES:
            counts[axis][balance[axis]] += 1
    return {
        axis: {
            str(key): value
            for key, value in sorted(
                axis_counts.items(), key=lambda item: str(item[0])
            )
        }
        for axis, axis_counts in counts.items()
    }


def balanced_control_epoch(
    examples: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    sample_count: int,
    power: float,
    maximum_weight: float,
) -> tuple[dict[str, Any], ...]:
    """Sample trajectories uniformly and inverse-weight all frozen strata."""
    if (
        not examples
        or sample_count <= 0
        or not math.isfinite(power)
        or power < 0.0
        or not math.isfinite(maximum_weight)
        or maximum_weight < 1.0
    ):
        raise ValueError("Set Transformer control sampler parameters are invalid")
    control_balancing_inventory(examples)
    by_trajectory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    counts = {axis: Counter() for axis in REQUIRED_BALANCE_AXES}
    for example in examples:
        by_trajectory[str(example["trajectory_id"])].append(example)
        balance = example["conditional_groups"][0]["balance"]
        for axis in REQUIRED_BALANCE_AXES:
            counts[axis][balance[axis]] += 1
    trajectories = sorted(by_trajectory)
    generator = random.Random(seed)
    trajectory_order = []
    while len(trajectory_order) < sample_count:
        cycle = list(trajectories)
        generator.shuffle(cycle)
        trajectory_order.extend(cycle)
    result = []
    for trajectory_id in trajectory_order[:sample_count]:
        rows = by_trajectory[trajectory_id]
        weights = []
        for row in rows:
            balance = row["conditional_groups"][0]["balance"]
            weight = 1.0
            for axis in REQUIRED_BALANCE_AXES:
                frequency = counts[axis][balance[axis]]
                inverse = len(examples) / (len(counts[axis]) * frequency)
                weight *= inverse**power
            weights.append(min(maximum_weight, weight))
        result.append(dict(generator.choices(rows, weights=weights, k=1)[0]))
    return tuple(result)


def pack_control_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    maximum_groups_per_state: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Pack balanced direct groups without dropping the base-cardinality-zero stratum."""
    if maximum_groups_per_state <= 0 or not examples:
        raise ValueError("Set Transformer packing parameters are invalid")
    by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for example in examples:
        groups = tuple(example.get("conditional_groups", ()))
        if len(groups) != 1:
            raise ValueError("Set Transformer packing expects one sampled group")
        by_state[str(example["state_id"])].append(example)
    generator = random.Random(seed)
    packed = []
    for state_id in sorted(by_state):
        rows = list(by_state[state_id])
        generator.shuffle(rows)
        reference = rows[0]
        if any(
            row["trajectory_id"] != reference["trajectory_id"]
            or row["candidate_event_step_ids"]
            != reference["candidate_event_step_ids"]
            for row in rows[1:]
        ):
            raise ValueError("Set Transformer packed state identity drifted")
        for offset in range(0, len(rows), maximum_groups_per_state):
            chunk = rows[offset : offset + maximum_groups_per_state]
            value = dict(reference)
            value["conditional_groups"] = tuple(
                row["conditional_groups"][0] for row in chunk
            )
            value["group_example_ids"] = tuple(
                row["group_example_id"] for row in chunk
            )
            value.pop("group_example_id", None)
            packed.append(value)
    generator.shuffle(packed)
    if sum(len(row["group_example_ids"]) for row in packed) != len(examples):
        raise AssertionError("Set Transformer packing changed sampled group count")
    return tuple(packed)


def group_equal_direct_loss(
    predictions: Any,
    batch: Mapping[str, Any],
    *,
    loss_config: Mapping[str, Any],
    reduction: str = "mean",
    torch: Any,
) -> tuple[Any, dict[str, float]]:
    """Keep every sampled group equally weighted after rich-state packing."""
    if reduction not in {"mean", "sum"}:
        raise ValueError("direct control loss reduction must be mean or sum")
    action_mask = batch["conditional_action_mask"]
    group_mask = batch["conditional_group_mask"]
    raw_targets = batch["conditional_raw_targets"]
    normalized_targets = batch["conditional_normalized_targets"]
    if predictions.shape != action_mask.shape:
        raise ValueError("direct control prediction geometry drifted")
    valid = action_mask & group_mask.unsqueeze(-1)
    candidates = valid.clone()
    candidates[:, :, 0] = False
    if not bool(candidates.any()):
        raise ValueError("direct control batch has no candidate targets")
    temperature = float(loss_config["decision_temperature"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("direct control decision temperature must be positive")
    normalized_predictions = predictions / batch["scales"][:, None, None]
    marginal_terms = torch.nn.functional.smooth_l1_loss(
        normalized_predictions,
        normalized_targets,
        reduction="none",
        beta=float(loss_config["conditional_marginal_smooth_l1_beta"]),
    )
    marginal_groups = (marginal_terms * candidates).sum(dim=2) / candidates.sum(
        dim=2
    ).clamp_min(1)
    group_count = group_mask.sum().clamp_min(1)
    marginal_sum = (marginal_groups * group_mask).sum()
    marginal_mean = marginal_sum / group_count
    logits = (normalized_predictions / temperature).masked_fill(~valid, -1e9)
    teacher_action = normalized_targets.masked_fill(~valid, -1e9).argmax(dim=2)
    listwise_rows = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        teacher_action.reshape(-1),
        reduction="none",
    ).reshape_as(group_mask)
    listwise_sum = (listwise_rows * group_mask).sum()
    listwise_mean = listwise_sum / group_count
    best_teacher = normalized_targets.masked_fill(~valid, -1e9).max(
        dim=2, keepdim=True
    ).values
    probabilities = torch.softmax(logits, dim=2)
    regret_rows = (
        probabilities
        * (best_teacher - normalized_targets).clamp_min(0.0)
        * valid
    ).sum(dim=2)
    regret_sum = (regret_rows * group_mask).sum()
    regret_mean = regret_sum / group_count
    signed = candidates & (torch.abs(normalized_targets) > 1e-6)
    sign_terms = torch.nn.functional.softplus(
        -normalized_predictions * torch.sign(normalized_targets)
    )
    sign_groups = (sign_terms * signed).sum(dim=2) / signed.sum(dim=2).clamp_min(1)
    sign_sum = (sign_groups * group_mask).sum()
    sign_mean = sign_sum / group_count
    if reduction == "sum":
        marginal = marginal_sum
        listwise = listwise_sum
        regret = regret_sum
        sign = sign_sum
    else:
        marginal = marginal_mean
        listwise = listwise_mean
        regret = regret_mean
        sign = sign_mean
    total = (
        float(loss_config["conditional_marginal"]) * marginal
        + float(loss_config["conditional_listwise"]) * listwise
        + float(loss_config["decision_regret"]) * regret
        + float(loss_config["sign_classification"]) * sign
    )
    predicted_action = logits.argmax(dim=2)
    correct = (predicted_action == teacher_action) & group_mask
    teacher_stop = (teacher_action == 0) & group_mask
    teacher_non_stop = (teacher_action != 0) & group_mask
    sign_correct = (
        (normalized_predictions > 0) == (normalized_targets > 0)
    ) & signed

    def ratio(numerator: Any, denominator: Any) -> float:
        return float(
            numerator.sum().detach() / denominator.sum().clamp_min(1).detach()
        )

    return total, {
        "action_top1_accuracy": ratio(correct, group_mask),
        "conditional_group_count": float(group_mask.sum().detach()),
        "conditional_listwise": float(listwise_mean.detach()),
        "conditional_marginal_regression": float(marginal_mean.detach()),
        "decision_regret": float(regret_mean.detach()),
        "non_stop_action_accuracy": ratio(
            correct & teacher_non_stop, teacher_non_stop
        ),
        "sign_accuracy": ratio(sign_correct, signed),
        "sign_classification": float(sign_mean.detach()),
        "stop_action_accuracy": ratio(correct & teacher_stop, teacher_stop),
        "teacher_stop_rate": ratio(teacher_stop, group_mask),
        "total": float(
            (
                float(loss_config["conditional_marginal"]) * marginal_mean
                + float(loss_config["conditional_listwise"]) * listwise_mean
                + float(loss_config["decision_regret"]) * regret_mean
                + float(loss_config["sign_classification"]) * sign_mean
            ).detach()
        ),
    }


def build_control_epoch_truth(
    *,
    epoch: int,
    checkpoint: Mapping[str, Any],
    rollout_records: Sequence[Mapping[str, Any]],
    states_by_id: Mapping[str, Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    normalization_floor: float,
    supplemental: Mapping[str, Mapping[tuple[int, ...], float]] | None = None,
) -> dict[str, Any]:
    """Bind actual queried bases and reduce selected B1--B4 truth authoritatively."""
    plan = build_epoch_truth_plan(
        epoch=epoch,
        checkpoint=checkpoint,
        rollout_records=rollout_records,
        states_by_id=states_by_id,
        supplemental=supplemental,
    )
    checkpoint_rows = split_manifest.get("checkpoint_states")
    checkpoint_ids = split_manifest.get("checkpoint_state_ids")
    if (
        not isinstance(checkpoint_rows, Sequence)
        or not isinstance(checkpoint_ids, Sequence)
        or len(checkpoint_rows) != 256
        or {row["state_id"] for row in checkpoint_rows} != set(checkpoint_ids)
        or {row["state_id"] for row in plan["records"]} != set(checkpoint_ids)
    ):
        raise ValueError("Set Transformer epoch truth escaped the 256-state denominator")
    selected_records = []
    selected_missing = []
    for record in plan["records"]:
        state = states_by_id[record["state_id"]]
        table = distance_table(state, supplemental)
        if () not in table:
            raise ValueError("Set Transformer heldout truth lacks an empty anchor")
        denominator = max(table[()], normalization_floor)
        for budget in BUDGETS:
            subset = tuple(record["selections"][str(budget)])
            if subset not in table:
                selected_missing.append((state["state_id"], budget))
                continue
            selected_records.append(
                {
                    "budget": budget,
                    "history_bin": history_bin(
                        len(state["candidate_event_step_ids"])
                    ),
                    "normalized_recovery": (table[()] - table[subset])
                    / denominator,
                    "state_id": state["state_id"],
                    "trajectory_id": state["trajectory_id"],
                }
            )
    authoritative_ready = bool(plan["truth_complete"] and not selected_missing)
    contract_truth = reduce_epoch_truth(
        selected_records if authoritative_ready else (),
        checkpoint_states=checkpoint_rows,
        epoch=epoch,
        checkpoint_sha256=checkpoint["sha256"],
        contract_content_sha256=split_manifest["content_sha256"],
    )
    if bool(contract_truth["truth_complete"]) != authoritative_ready:
        raise AssertionError("selected truth completeness disagrees with reducer")
    result = dict(plan)
    result["contract_epoch_truth"] = contract_truth
    result["selected_missing_pair_count"] = len(selected_missing)
    result["status"] = (
        "COMPLETE_SET_TRANSFORMER_CONTROL_EPOCH_TRUTH"
        if contract_truth["status"] == COMPLETE_TRUTH_STATUS
        else "PENDING_SET_TRANSFORMER_CONTROL_EPOCH_TRUTH"
    )
    return result


def merge_control_truth_schedule(
    plans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = merge_epoch_truth_plans(plans)
    result["epoch_checkpoints"] = [
        {
            "checkpoint_sha256": plan["checkpoint"]["sha256"],
            "epoch": int(plan["epoch"]),
        }
        for plan in plans
    ]
    result["model_family"] = "set_transformer_direct_marginal"
    result["source_plan_hashes"] = [sha256_json(dict(plan)) for plan in plans]
    result["status"] = (
        "COMPLETE_SET_TRANSFORMER_CONTROL_TRUTH"
        if result["missing_coalition_count"] == 0
        else "PENDING_SET_TRANSFORMER_CONTROL_TRUTH"
    )
    return result


def select_control_checkpoint(
    plans: Sequence[Mapping[str, Any]],
    *,
    patience: int,
    minimum_delta: float,
    tie_breaker_minimum_delta: float = 0.0,
) -> dict[str, Any]:
    """Use only frozen-reducer outputs; proxy train/eval losses never select."""
    if not plans:
        raise ValueError("Set Transformer checkpoint selection has no epoch plans")
    truths = []
    checkpoint_shas = set()
    for expected_epoch, plan in enumerate(plans, start=1):
        if int(plan.get("epoch", -1)) != expected_epoch:
            raise ValueError("Set Transformer epoch plans must be contiguous")
        checkpoint = plan.get("checkpoint")
        truth = plan.get("contract_epoch_truth")
        if not isinstance(checkpoint, Mapping) or not isinstance(truth, Mapping):
            raise ValueError("Set Transformer epoch plan is incomplete")
        checkpoint_sha = _sha256(checkpoint.get("sha256"), "checkpoint SHA256")
        if checkpoint_sha in checkpoint_shas:
            raise ValueError("Set Transformer epoch checkpoints are not immutable")
        checkpoint_shas.add(checkpoint_sha)
        if truth.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("Set Transformer truth/checkpoint binding drifted")
        truths.append(truth)
    return select_checkpoint_from_epoch_truth(
        truths,
        patience=patience,
        minimum_delta=minimum_delta,
        tie_breaker_minimum_delta=tie_breaker_minimum_delta,
    )


def control_epoch_barrier(
    plans: Sequence[Mapping[str, Any]],
    *,
    patience: int,
    minimum_delta: float,
    tie_breaker_minimum_delta: float = 0.0,
) -> tuple[str, dict[str, Any]]:
    """Gate every next optimizer epoch on authoritative truth for the prior epoch."""
    selection = select_control_checkpoint(
        plans,
        patience=patience,
        minimum_delta=minimum_delta,
        tie_breaker_minimum_delta=tie_breaker_minimum_delta,
    )
    latest = plans[-1].get("contract_epoch_truth")
    if not isinstance(latest, Mapping):
        raise ValueError("Set Transformer barrier has no latest epoch truth")
    if latest.get("truth_complete") is not True:
        if selection.get("decision_ready") is not False:
            raise AssertionError("incomplete epoch truth unexpectedly advanced selection")
        return WAITING_FOR_TRUTH, selection
    if selection.get("decision_ready") is not True:
        raise AssertionError("complete epoch truth did not unlock selection")
    if selection.get("stopped_early") is True:
        return EARLY_STOP, selection
    return CONTINUE_TRAINING, selection


def order_hash(examples: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(row["group_example_id"]) for row in examples).encode()
    ).hexdigest()


__all__ = [
    "CONTINUE_TRAINING",
    "CONTROL_STATUS",
    "EARLY_STOP",
    "REQUIRED_BALANCE_AXES",
    "balanced_control_epoch",
    "build_control_epoch_truth",
    "control_balancing_inventory",
    "control_epoch_barrier",
    "group_equal_direct_loss",
    "merge_control_truth_schedule",
    "order_hash",
    "pack_control_examples",
    "prepare_control_examples",
    "select_control_checkpoint",
    "validate_control_config",
    "WAITING_FOR_TRUTH",
]
