#!/usr/bin/env python3
"""Train the direct conditional-marginal v3 student on complete expansion groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from causalcache.set_utility_mvp import canonical_json_bytes
from causalcache.set_utility_token_models import (
    TokenConditionalMarginalPredictor,
    TokenUtilityModelConfig,
)
from causalcache.set_utility_variable_history import history_bin
from scripts.train_set_utility_token_predictor import (
    _TokenCache,
    _batch_stream,
    _cache_covers_input,
    _configure_attention_backend,
    _distributed_epoch_shard,
    _distributed_metric_average,
    _read_json,
    _read_jsonl,
    _resolve_normalization_floor,
    _save_checkpoint,
    _seed_training_runtime,
    _trajectory_uniform_epoch,
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _conditional_groups(
    state: dict[str, Any],
    *,
    normalization_floor: float,
    maximum_base_cardinality: int,
) -> tuple[dict[str, Any], ...]:
    candidates = tuple(state["candidate_event_step_ids"])
    if (
        not candidates
        or candidates != tuple(sorted(candidates))
        or len(candidates) != len(set(candidates))
    ):
        raise ValueError("candidate events must be sorted, unique, and non-empty")
    by_subset = {
        tuple(row["coalition_event_step_ids"]): float(row["distance"])
        for row in state["distance_rows"]
    }
    if len(by_subset) != len(state["distance_rows"]) or () not in by_subset:
        raise ValueError("conditional training requires one unique empty coalition")
    universe = set(candidates)
    if any(
        subset != tuple(sorted(subset))
        or len(subset) != len(set(subset))
        or not set(subset).issubset(universe)
        for subset in by_subset
    ):
        raise ValueError("conditional training received an invalid coalition")
    baseline = by_subset[()]
    if baseline < -1e-8 or normalization_floor < 0.0:
        raise ValueError("conditional utility normalization is invalid")
    scale = max(baseline, normalization_floor)
    scale = scale if scale > 0.0 else 1.0
    groups = []
    for selected in sorted(by_subset, key=lambda item: (len(item), item)):
        if len(selected) > maximum_base_cardinality:
            continue
        selected_set = set(selected)
        remaining = tuple(event for event in candidates if event not in selected_set)
        if not remaining:
            continue
        expansions = {event: tuple(sorted((*selected, event))) for event in remaining}
        if any(expansion not in by_subset for expansion in expansions.values()):
            continue
        raw = [0.0]
        action_mask = [True]
        for event in candidates:
            if event in expansions:
                raw.append(by_subset[selected] - by_subset[expansions[event]])
                action_mask.append(True)
            else:
                raw.append(0.0)
                action_mask.append(False)
        groups.append(
            {
                "action_mask": tuple(action_mask),
                "normalized_marginals": tuple(value / scale for value in raw),
                "raw_marginals": tuple(raw),
                "selected_event_step_ids": selected,
            }
        )
    return tuple(groups)


def _prepare_state(
    state: dict[str, Any],
    *,
    normalization_floor: float,
    maximum_base_cardinality: int,
) -> dict[str, Any] | None:
    groups = _conditional_groups(
        state,
        normalization_floor=normalization_floor,
        maximum_base_cardinality=maximum_base_cardinality,
    )
    if not groups:
        return None
    prepared = dict(state)
    prepared["conditional_groups"] = groups
    return prepared


def _split_train_holdout(
    states: tuple[dict[str, Any], ...],
    *,
    fraction: float,
    salt: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if not 0.0 < fraction < 0.5 or not salt:
        raise ValueError("train-holdout split contract is invalid")
    trajectories = sorted({state["trajectory_id"] for state in states})
    holdout_count = max(1, round(len(trajectories) * fraction))
    ranked = sorted(
        trajectories,
        key=lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest(),
    )
    holdout_trajectories = set(ranked[:holdout_count])
    optimization = tuple(
        state for state in states if state["trajectory_id"] not in holdout_trajectories
    )
    holdout = tuple(
        state for state in states if state["trajectory_id"] in holdout_trajectories
    )
    if not optimization or not holdout:
        raise ValueError("train-holdout split produced an empty side")
    if {state["trajectory_id"] for state in optimization} & {
        state["trajectory_id"] for state in holdout
    }:
        raise AssertionError("train-holdout trajectories overlap")
    return optimization, holdout


def _attach_conditional_supervision(
    batch: dict[str, Any],
    states: list[dict[str, Any]],
    *,
    torch: Any,
) -> dict[str, Any]:
    device = batch["raw_targets"].device
    event_mask = batch["model"]["event_mask"]
    batch_size, event_count = event_mask.shape
    group_count = max(len(state["conditional_groups"]) for state in states)
    selected_masks = torch.zeros(
        (batch_size, group_count, event_count), dtype=torch.bool, device=device
    )
    action_mask = torch.zeros(
        (batch_size, group_count, event_count + 1),
        dtype=torch.bool,
        device=device,
    )
    group_mask = torch.zeros((batch_size, group_count), dtype=torch.bool, device=device)
    raw_targets = torch.zeros(
        (batch_size, group_count, event_count + 1),
        dtype=torch.float32,
        device=device,
    )
    normalized_targets = torch.zeros_like(raw_targets)
    for batch_index, state in enumerate(states):
        candidates = tuple(state["candidate_event_step_ids"])
        by_event = {event: index for index, event in enumerate(candidates)}
        for group_index, group in enumerate(state["conditional_groups"]):
            group_mask[batch_index, group_index] = True
            for event in group["selected_event_step_ids"]:
                selected_masks[batch_index, group_index, by_event[event]] = True
            width = len(candidates) + 1
            action_mask[batch_index, group_index, :width] = torch.tensor(
                group["action_mask"], dtype=torch.bool, device=device
            )
            raw_targets[batch_index, group_index, :width] = torch.tensor(
                group["raw_marginals"], dtype=torch.float32, device=device
            )
            normalized_targets[batch_index, group_index, :width] = torch.tensor(
                group["normalized_marginals"], dtype=torch.float32, device=device
            )
    if bool((selected_masks & ~event_mask.unsqueeze(1)).any()):
        raise AssertionError("a conditional group selects a padded event")
    if bool((action_mask[:, :, 1:] & selected_masks).any()):
        raise AssertionError("a selected event remained a valid action")
    batch["model"].pop("subset_masks")
    batch["model"]["selected_masks"] = selected_masks
    batch["conditional_action_mask"] = action_mask
    batch["conditional_group_mask"] = group_mask
    batch["conditional_raw_targets"] = raw_targets
    batch["conditional_normalized_targets"] = normalized_targets
    return batch


def _conditional_loss(
    predictions: Any,
    batch: dict[str, Any],
    *,
    loss_config: dict[str, Any],
    torch: Any,
) -> tuple[Any, dict[str, float]]:
    action_mask = batch["conditional_action_mask"]
    group_mask = batch["conditional_group_mask"]
    raw_targets = batch["conditional_raw_targets"]
    normalized_targets = batch["conditional_normalized_targets"]
    if predictions.shape != action_mask.shape:
        raise ValueError("conditional prediction geometry drifted")
    valid = action_mask & group_mask.unsqueeze(-1)
    non_stop = valid.clone()
    non_stop[:, :, 0] = False
    temperature = float(loss_config["decision_temperature"])
    if temperature <= 0.0:
        raise ValueError("decision temperature must be positive")

    normalized_predictions = predictions / batch["scales"][:, None, None]
    marginal_terms = torch.nn.functional.smooth_l1_loss(
        normalized_predictions,
        normalized_targets,
        reduction="none",
        beta=float(loss_config["conditional_marginal_smooth_l1_beta"]),
    )
    marginal_groups = (marginal_terms * non_stop).sum(dim=2) / non_stop.sum(
        dim=2
    ).clamp_min(1)

    student_logits = (normalized_predictions / temperature).masked_fill(~valid, -1e9)
    teacher_best_action = normalized_targets.masked_fill(~valid, -1e9).argmax(dim=2)
    listwise_groups = torch.nn.functional.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        teacher_best_action.reshape(-1),
        reduction="none",
    ).reshape_as(group_mask)
    best_teacher = (
        normalized_targets.masked_fill(~valid, -1e9).max(dim=2, keepdim=True).values
    )
    probabilities = torch.softmax(student_logits, dim=2)
    regret_groups = (
        probabilities * (best_teacher - normalized_targets).clamp_min(0.0) * valid
    ).sum(dim=2)

    signed = non_stop & (torch.abs(normalized_targets) > 1e-6)
    sign_terms = torch.nn.functional.softplus(
        -normalized_predictions * torch.sign(normalized_targets)
    )
    sign_groups = (sign_terms * signed).sum(dim=2) / signed.sum(dim=2).clamp_min(1)
    denominator = group_mask.sum(dim=1).clamp_min(1)
    marginal_rows = (marginal_groups * group_mask).sum(dim=1) / denominator
    listwise_rows = (listwise_groups * group_mask).sum(dim=1) / denominator
    regret_rows = (regret_groups * group_mask).sum(dim=1) / denominator
    sign_rows = (sign_groups * group_mask).sum(dim=1) / denominator
    marginal = marginal_rows.mean()
    listwise = listwise_rows.mean()
    regret = regret_rows.mean()
    sign = sign_rows.mean()
    total = (
        float(loss_config["conditional_marginal"]) * marginal
        + float(loss_config["conditional_listwise"]) * listwise
        + float(loss_config["decision_regret"]) * regret
        + float(loss_config["sign_classification"]) * sign
    )

    predicted_best_action = student_logits.argmax(dim=2)
    correct = (predicted_best_action == teacher_best_action) & group_mask
    teacher_stop = (teacher_best_action == 0) & group_mask
    teacher_non_stop = (teacher_best_action != 0) & group_mask
    sign_correct = ((normalized_predictions > 0) == (normalized_targets > 0)) & signed

    def ratio(numerator: Any, denominator_mask: Any) -> float:
        return float(
            numerator.sum().detach() / denominator_mask.sum().clamp_min(1).detach()
        )

    return total, {
        "action_top1_accuracy": ratio(correct, group_mask),
        "conditional_group_count": float(group_mask.sum().detach()),
        "conditional_listwise": float(listwise.detach()),
        "conditional_marginal_regression": float(marginal.detach()),
        "decision_regret": float(regret.detach()),
        "non_stop_action_accuracy": ratio(correct & teacher_non_stop, teacher_non_stop),
        "sign_accuracy": ratio(sign_correct, signed),
        "sign_classification": float(sign.detach()),
        "stop_action_accuracy": ratio(correct & teacher_stop, teacher_stop),
        "teacher_stop_rate": ratio(teacher_stop, group_mask),
        "total": float(total.detach()),
    }


def _inventory(states: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    bins = Counter(
        history_bin(len(state["candidate_event_step_ids"])) for state in states
    )
    return {
        "complete_expansion_group_count": sum(
            len(state["conditional_groups"]) for state in states
        ),
        "history_bin_state_counts": dict(sorted(bins.items())),
        "state_count": len(states),
        "trajectory_count": len({state["trajectory_id"] for state in states}),
    }


def _evaluate(
    model: Any,
    states: tuple[dict[str, Any], ...],
    *,
    cache: _TokenCache,
    batch_size: int,
    device: Any,
    loss_config: dict[str, Any],
    normalization_floor: float,
    torch: Any,
) -> dict[str, float]:
    totals = defaultdict(float)
    denominator = 0.0
    model.eval()
    with torch.inference_mode():
        for selected, batch in _batch_stream(
            states,
            batch_size=batch_size,
            cache=cache,
            device=device,
            torch=torch,
            normalization_floor=normalization_floor,
        ):
            batch = _attach_conditional_supervision(batch, selected, torch=torch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(**batch["model"])
                _, metrics = _conditional_loss(
                    predictions, batch, loss_config=loss_config, torch=torch
                )
            weight = float(len(selected))
            denominator += weight
            for key, value in metrics.items():
                totals[key] += value * weight
    return {key: value / denominator for key, value in sorted(totals.items())}


def _resume_path(output_root: Path, rank: int) -> Path:
    return output_root / f"resume-rank-{rank:02d}.pt"


def _save_resume(
    path: Path,
    *,
    epoch: int,
    history: list[dict[str, Any]],
    model: Any,
    optimizer: Any,
    scheduler: Any,
    identity: dict[str, Any],
    torch: Any,
) -> None:
    temporary = path.with_suffix(f".pt.{os.getpid()}.tmp")
    torch.save(
        {
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "epoch": epoch,
            "history": history,
            "identity": identity,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "python_rng_state": random.getstate(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise RuntimeError("direct marginal training requires PyTorch") from error

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if args.device != "cuda":
            raise RuntimeError("distributed direct training requires --device cuda")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
    else:
        device = args.device
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("direct marginal training requires CUDA")

    config_path = args.config.resolve()
    config = _read_json(config_path)
    variant = config["variants"][args.variant]
    training = config["training"]
    output_root = args.output_root.resolve()
    if args.resume:
        if not _resume_path(output_root, rank).is_file():
            raise FileNotFoundError("rank-local resume snapshot is missing")
    else:
        exists = torch.tensor([int(output_root.exists())], device=device)
        if distributed:
            torch.distributed.all_reduce(exists, op=torch.distributed.ReduceOp.MAX)
        if int(exists.item()):
            raise FileExistsError(f"training output already exists: {output_root}")
        if rank == 0:
            output_root.mkdir(parents=True)
            (output_root / "checkpoints").mkdir()
        if distributed:
            torch.distributed.barrier()

    input_root = args.input_root.resolve()
    cache_root = args.cache_root.resolve()
    input_manifest = _read_json(input_root / "manifest.json")
    cache_manifest = _read_json(cache_root / "manifest.json")
    input_contract = config["input"]
    if (
        input_manifest.get("evaluation_labels_included") is not False
        or cache_manifest.get("evaluation_labels_included") is not False
        or not _cache_covers_input(input_manifest, cache_manifest)
        or input_manifest.get("content_sha256")
        != input_contract["training_input_content_sha256"]
        or cache_manifest.get("content_sha256")
        != input_contract["contextual_cache_content_sha256"]
    ):
        raise ValueError("direct training input/cache identity or firewall drifted")
    initial_checkpoint = args.initial_checkpoint.resolve()
    if _file_sha256(initial_checkpoint) != input_contract["initial_checkpoint_sha256"]:
        raise ValueError("Stage-A initialization checkpoint drifted")

    normalization_floor = _resolve_normalization_floor(training, None)
    maximum_base_cardinality = int(training["maximum_base_cardinality"])
    states = _read_jsonl(input_root / input_manifest["states_jsonl"])
    raw_train = tuple(state for state in states if state["role"] == "train")
    prepared = tuple(
        item
        for state in raw_train
        if (
            item := _prepare_state(
                state,
                normalization_floor=normalization_floor,
                maximum_base_cardinality=maximum_base_cardinality,
            )
        )
        is not None
    )
    optimization, holdout = _split_train_holdout(
        prepared,
        fraction=float(training["train_holdout_fraction"]),
        salt=str(training["train_holdout_salt"]),
    )
    minimums = training["minimum_inventory"]
    observed = _inventory(prepared)
    if observed["state_count"] < int(minimums["state_count"]) or observed[
        "complete_expansion_group_count"
    ] < int(minimums["complete_expansion_group_count"]):
        raise ValueError("direct training inventory is below the frozen minimum")

    attention_backend = _configure_attention_backend(torch, training)
    seed = int(training["seed"])
    _seed_training_runtime(torch, seed)
    torch.set_float32_matmul_precision("high")
    cache = _TokenCache(cache_root, cache_manifest, device="cpu", mode="lazy_cpu")
    model_config = TokenUtilityModelConfig(**variant["model"])
    checkpoint_model = TokenConditionalMarginalPredictor(model_config).to(device)
    checkpoint_model.load_state_dict(load_file(str(initial_checkpoint)), strict=True)
    if distributed:
        expected_world_size = int(variant["distributed_world_size"])
        if world_size != expected_world_size:
            raise ValueError("distributed world size conflicts with committed config")
        model = torch.nn.parallel.DistributedDataParallel(
            checkpoint_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed(seed + rank)
    else:
        model = checkpoint_model
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(variant["learning_rate"]),
        weight_decay=float(variant["weight_decay"]),
    )
    epochs = int(training["epochs"])
    batch_size = int(variant["per_device_batch_size"])
    accumulation = int(variant["gradient_accumulation_steps"])
    evaluation_batch_size = int(variant["evaluation_batch_size"])
    rows_per_epoch = len(_trajectory_uniform_epoch(optimization, seed=seed))
    padding = (-rows_per_epoch) % (batch_size * world_size * accumulation)
    local_rows = (rows_per_epoch + padding) // world_size
    steps_per_epoch = math.ceil(math.ceil(local_rows / batch_size) / accumulation)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, round(total_steps * float(training["warmup_ratio"])))
    minimum_lr_ratio = float(training["minimum_lr_ratio"])

    def learning_rate_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    identity = {
        "config_sha256": _file_sha256(config_path),
        "input_content_sha256": input_manifest["content_sha256"],
        "cache_content_sha256": cache_manifest["content_sha256"],
        "initial_checkpoint_sha256": input_contract["initial_checkpoint_sha256"],
        "variant": args.variant,
        "world_size": world_size,
    }
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_epoch = None
    best_holdout_regret = math.inf
    best_checkpoint = None
    patience = 0
    termination_reason = "maximum_epochs"
    if args.resume:
        snapshot = torch.load(
            _resume_path(output_root, rank),
            map_location=device,
            weights_only=False,
        )
        if snapshot["identity"] != identity:
            raise ValueError("resume identity drifted")
        checkpoint_model.load_state_dict(snapshot["model"], strict=True)
        optimizer.load_state_dict(snapshot["optimizer"])
        scheduler.load_state_dict(snapshot["scheduler"])
        random.setstate(snapshot["python_rng_state"])
        torch.set_rng_state(snapshot["torch_rng_state"])
        torch.cuda.set_rng_state(snapshot["cuda_rng_state"])
        history = snapshot["history"]
        start_epoch = int(snapshot["epoch"]) + 1
        if history:
            best_entry = min(history, key=lambda row: row["holdout"]["decision_regret"])
            best_epoch = int(best_entry["epoch"])
            best_holdout_regret = float(best_entry["holdout"]["decision_regret"])
            patience = int(history[-1]["patience"])
            best_checkpoint = {
                "path": "best.safetensors",
                "sha256": _file_sha256(output_root / "best.safetensors"),
                "byte_count": (output_root / "best.safetensors").stat().st_size,
            }

    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        global_order = _trajectory_uniform_epoch(optimization, seed=seed + epoch)
        if distributed:
            order, observed_padding = _distributed_epoch_shard(
                global_order,
                per_device_batch_size=batch_size,
                gradient_accumulation_steps=accumulation,
                rank=rank,
                world_size=world_size,
            )
            if observed_padding != padding:
                raise RuntimeError("distributed padding drifted")
        else:
            order = global_order
        optimizer.zero_grad(set_to_none=True)
        totals = defaultdict(float)
        denominator = 0.0
        pending = 0
        for batch_index, (selected, batch) in enumerate(
            _batch_stream(
                order,
                batch_size=batch_size,
                cache=cache,
                device=device,
                torch=torch,
                normalization_floor=normalization_floor,
            )
        ):
            batch = _attach_conditional_supervision(batch, selected, torch=torch)
            final_batch = (batch_index + 1) * batch_size >= len(order)
            synchronize = pending + 1 == accumulation or final_batch
            sync_context = (
                nullcontext() if not distributed or synchronize else model.no_sync()
            )
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predictions = model(**batch["model"])
                    loss, metrics = _conditional_loss(
                        predictions,
                        batch,
                        loss_config=training["loss"],
                        torch=torch,
                    )
                (loss / accumulation).backward()
            pending += 1
            if pending == accumulation or final_batch:
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(training["maximum_gradient_norm"])
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            weight = float(len(selected))
            denominator += weight
            for key, value in metrics.items():
                totals[key] += value * weight
        train_metrics = (
            _distributed_metric_average(totals, denominator, device=device, torch=torch)
            if distributed
            else {key: value / denominator for key, value in sorted(totals.items())}
        )
        holdout_metrics = None
        if rank == 0:
            holdout_metrics = _evaluate(
                checkpoint_model,
                holdout,
                cache=cache,
                batch_size=evaluation_batch_size,
                device=device,
                loss_config=training["loss"],
                normalization_floor=normalization_floor,
                torch=torch,
            )
        if distributed:
            payload = [holdout_metrics]
            torch.distributed.broadcast_object_list(payload, src=0)
            holdout_metrics = payload[0]
        if not isinstance(holdout_metrics, dict) or not all(
            math.isfinite(value)
            for metrics in (train_metrics, holdout_metrics)
            for value in metrics.values()
        ):
            raise RuntimeError("direct marginal training produced non-finite metrics")
        improved = holdout_metrics["decision_regret"] < (
            best_holdout_regret - float(training["minimum_delta"])
        )
        if improved:
            best_holdout_regret = holdout_metrics["decision_regret"]
            best_epoch = epoch
            patience = 0
            if rank == 0:
                best_checkpoint = _save_checkpoint(
                    checkpoint_model, output_root / "best.safetensors"
                )
        else:
            patience += 1
        history.append(
            {
                "epoch": epoch,
                "holdout": holdout_metrics,
                "learning_rate": scheduler.get_last_lr()[0],
                "patience": patience,
                "train": train_metrics,
            }
        )
        if rank == 0:
            print(json.dumps(history[-1], sort_keys=True), flush=True)
        if distributed:
            torch.distributed.barrier()
        _save_resume(
            _resume_path(output_root, rank),
            epoch=epoch,
            history=history,
            model=checkpoint_model,
            optimizer=optimizer,
            scheduler=scheduler,
            identity=identity,
            torch=torch,
        )
        if distributed:
            torch.distributed.barrier()
        if patience >= int(training["early_stopping_patience"]):
            termination_reason = "early_stopping"
            break

    if rank == 0:
        summary = {
            "attention_backend": attention_backend,
            "best_checkpoint": best_checkpoint,
            "best_epoch": best_epoch,
            "best_holdout_decision_regret": best_holdout_regret,
            "cache_content_sha256": cache_manifest["content_sha256"],
            "config_sha256": identity["config_sha256"],
            "distributed": {
                "backend": "nccl" if distributed else None,
                "effective_global_batch_size": batch_size * world_size * accumulation,
                "gradient_accumulation_steps": accumulation,
                "per_device_batch_size": batch_size,
                "world_size": world_size,
            },
            "elapsed_seconds_this_invocation": time.time() - started,
            "evaluation_records_loaded": False,
            "history": history,
            "holdout_inventory": _inventory(holdout),
            "initial_checkpoint_sha256": input_contract["initial_checkpoint_sha256"],
            "input_content_sha256": input_manifest["content_sha256"],
            "model": variant["model"],
            "optimization_inventory": _inventory(optimization),
            "resume_supported": True,
            "schema_version": "causalcache.direct_marginal_training.v1",
            "seed": seed,
            "status": "COMPLETED_DIRECT_MARGINAL_V3_STAGE_B_TRAINING",
            "termination_reason": termination_reason,
            "train_inventory": observed,
            "tune_labels_entered_training_or_selection": False,
            "variant": args.variant,
        }
        summary["content_sha256"] = hashlib.sha256(
            canonical_json_bytes(summary)
        ).hexdigest()
        _atomic_json(output_root / "summary.json", summary)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
