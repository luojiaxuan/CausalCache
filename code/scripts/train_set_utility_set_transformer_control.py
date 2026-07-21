#!/usr/bin/env python3
"""Train the fresh direct Set Transformer under the frozen heldout contract."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalcache.set_utility_recovery_checkpoint import RecoveryCheckpointManager
from causalcache.set_utility_formal_input_verification import (
    coordinate_formal_cache_verification,
)
from causalcache.set_utility_resume_generation import (
    has_published_resume_generation,
    load_resume_generation_collective,
    restore_rng_states,
    resume_rank_signature as _resume_rank_signature,
    save_resume_generation_collective,
    validate_resume_rank_signatures as _validate_resume_rank_signatures,
)
from causalcache.set_utility_set_transformer_control import (
    CONTINUE_TRAINING,
    EARLY_STOP,
    WAITING_FOR_TRUTH,
    balanced_control_epoch,
    build_control_epoch_truth,
    control_balancing_inventory,
    control_epoch_barrier,
    group_equal_direct_loss,
    merge_control_truth_schedule,
    pack_control_examples,
    prepare_control_examples,
    validate_control_config,
)
from causalcache.set_utility_token_models import (
    TokenConditionalMarginalPredictor,
    TokenUtilityModelConfig,
)
from causalcache.set_utility_structured_training import formal_group_inventory
from scripts.train_set_utility_direct_marginal_v3 import (
    _attach_conditional_supervision,
)
from scripts.train_set_utility_structured_marginal import (
    _epoch_plan_path,
    _load_split_manifest,
    _load_supplemental_truth,
    _load_training_inputs,
    _plan_checkpoint_bindings,
    _read_epoch_plans,
    _read_existing_epoch_plan,
    _read_signed_json,
    _rollout_heldout,
    _save_or_verify_epoch_checkpoint,
    _selected_checkpoint,
    _sha256_file,
    _signed,
    _truth_sources,
    _write_atomic,
)
from scripts.train_set_utility_token_predictor import (
    _TokenCache,
    _batch_stream,
    _configure_attention_backend,
    _distributed_epoch_shard,
    _distributed_metric_average,
    _read_json,
    _seed_training_runtime,
)


class SetTransformerControlRuntime:
    """Small model/runtime hook around the frozen control training loop."""

    model_family = "set_transformer_direct_marginal"
    progress_schema = "causalcache.set_transformer_control_progress.v1"
    summary_schema = "causalcache.set_transformer_control_training.v1"
    completed_status = "COMPLETED_SET_TRANSFORMER_CONTROL_SELECTED_BY_TRUE_RECOVERY"
    finalized_status = "COMPLETED_SET_TRANSFORMER_CONTROL_TRUE_RECOVERY_SELECTION"
    pending_finalized_status = "PENDING_SET_TRANSFORMER_CONTROL_TRUE_RECOVERY"
    selection_schema = "causalcache.set_transformer_control_selection.v1"

    def validate_config(
        self, config: Mapping[str, Any], variant_name: str
    ) -> dict[str, Any]:
        return validate_control_config(config, variant_name)

    def execution_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(dict(config))

    def build_model(
        self,
        *,
        variant: Mapping[str, Any],
        device: Any,
        torch: Any,
    ) -> Any:
        return TokenConditionalMarginalPredictor(
            TokenUtilityModelConfig(**variant["model"])
        ).to(device)

    def optimizer_parameter_groups(
        self, model: Any, *, variant: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        return [
            {
                "name": "direct_head",
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                "lr": float(variant["learning_rate"]),
            }
        ]

    def set_training_mode(self, model: Any) -> None:
        model.train()

    def load_resume_model_state(self, model: Any, state: Mapping[str, Any]) -> None:
        model.load_state_dict(state, strict=True)

    def identity_fields(self) -> dict[str, Any]:
        return {"initialization": "fresh"}

    def runtime_metadata(self) -> dict[str, Any]:
        return {}

    def checkpoint_manager_kwargs(self) -> dict[str, Any]:
        return {}

    def ddp_find_unused_parameters(self) -> bool:
        return False


def _epoch_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    epoch: int,
    seed: int,
    balancing: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    sampled = balanced_control_epoch(
        examples,
        seed=seed + epoch,
        sample_count=int(balancing["samples_per_epoch"]),
        power=float(balancing["power"]),
        maximum_weight=float(balancing["maximum_weight"]),
    )
    return pack_control_examples(
        sampled,
        maximum_groups_per_state=int(balancing["maximum_groups_per_encoded_state"]),
        seed=seed + 100_000 + epoch,
    )


def _optimizer_steps(
    row_count: int,
    *,
    batch_size: int,
    accumulation: int,
    world_size: int,
) -> int:
    global_batch = batch_size * accumulation * world_size
    padded = row_count + (-row_count) % global_batch
    local_rows = padded // world_size
    return math.ceil(math.ceil(local_rows / batch_size) / accumulation)


def _partition_resume_plans(
    plans: Sequence[Mapping[str, Any]], *, snapshot_epoch: int
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Allow one ahead plan from checkpoint/rollout-before-resume crash window."""
    if snapshot_epoch <= 0:
        raise ValueError("Set Transformer resume epoch must be positive")
    ordered = tuple(
        sorted((dict(plan) for plan in plans), key=lambda row: row["epoch"])
    )
    epochs = tuple(int(plan["epoch"]) for plan in ordered)
    if epochs != tuple(range(1, len(ordered) + 1)):
        raise ValueError("Set Transformer resume plans are not contiguous")
    if len(ordered) not in {snapshot_epoch, snapshot_epoch + 1}:
        raise ValueError("Set Transformer resume epoch-plan inventory drifted")
    completed = tuple(plan for plan in ordered if int(plan["epoch"]) <= snapshot_epoch)
    crash_window = tuple(
        plan for plan in ordered if int(plan["epoch"]) > snapshot_epoch
    )
    if len(completed) != snapshot_epoch or len(crash_window) > 1:
        raise ValueError("Set Transformer resume plan partition drifted")
    return completed, crash_window


def _group_sum_backward_scale(*, global_group_count: float, world_size: int) -> float:
    """Undo DDP's rank mean while normalizing by all groups in one update."""
    if (
        not math.isfinite(global_group_count)
        or global_group_count <= 0.0
        or world_size <= 0
    ):
        raise ValueError("global group-loss denominator is invalid")
    return world_size / global_group_count


def _publish_schedule(
    output_root: Path,
    plans: Sequence[Mapping[str, Any]],
    *,
    model_family: str = "set_transformer_direct_marginal",
) -> dict[str, Any]:
    schedule = merge_control_truth_schedule(plans)
    if not isinstance(model_family, str) or not model_family:
        raise ValueError("truth schedule model family must be named")
    schedule["model_family"] = model_family
    _write_atomic(output_root / "heldout-truth-schedule.json", _signed(schedule))
    return schedule


def _selection(
    plans: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    selection = config["checkpoint_selection"]
    return control_epoch_barrier(
        plans,
        patience=int(selection["patience"]),
        minimum_delta=float(selection["minimum_delta"]),
        tie_breaker_minimum_delta=float(
            selection.get("tie_breaker_minimum_delta", 0.0)
        ),
    )


def _rebuild_plans(
    plans: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    states_by_id: Mapping[str, Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    normalization_floor: float,
    supplemental: Mapping[str, Mapping[tuple[int, ...], float]],
) -> tuple[dict[str, Any], ...]:
    rebuilt = []
    for plan in plans:
        _verify_epoch_checkpoint(output_root, plan)
        value = build_control_epoch_truth(
            epoch=int(plan["epoch"]),
            checkpoint=plan["checkpoint"],
            rollout_records=plan["records"],
            states_by_id=states_by_id,
            split_manifest=split_manifest,
            normalization_floor=normalization_floor,
            supplemental=supplemental,
        )
        _write_atomic(_epoch_plan_path(output_root, int(plan["epoch"])), _signed(value))
        rebuilt.append(value)
    return tuple(rebuilt)


def _verify_epoch_checkpoint(output_root: Path, plan: Mapping[str, Any]) -> None:
    checkpoint = plan["checkpoint"]
    path = output_root / checkpoint["path"]
    if (
        not path.is_file()
        or path.stat().st_size != int(checkpoint["byte_count"])
        or _sha256_file(path) != checkpoint["sha256"]
    ):
        raise ValueError("immutable Set Transformer epoch checkpoint drifted")


def _fit(
    args: argparse.Namespace,
    *,
    runtime: SetTransformerControlRuntime | None = None,
) -> None:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Set Transformer control training requires PyTorch"
        ) from error
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Set Transformer control training requires CUDA")
    runtime = runtime or SetTransformerControlRuntime()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    variant = runtime.validate_config(config, args.variant)
    execution_config = runtime.execution_config(config)
    split_manifest = _load_split_manifest(args.split_manifest, config)
    if split_manifest is None:
        raise ValueError("Set Transformer control requires the frozen split manifest")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if world_size not in tuple(int(value) for value in variant["allowed_world_sizes"]):
        raise ValueError("Set Transformer DDP world size is outside the config")
    if distributed:
        if args.device != "cuda":
            raise RuntimeError(
                "distributed Set Transformer training needs --device cuda"
            )
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    cache_receipt_mode = coordinate_formal_cache_verification(
        args.cache_root.resolve(),
        expected_content_sha256=config["input"]["contextual_cache_content_sha256"],
        rank=rank,
        distributed=distributed,
        torch=torch,
    )
    input_manifest, cache_manifest, optimization_states, heldout_states = (
        _load_training_inputs(
            input_root=args.input_root.resolve(),
            cache_root=args.cache_root.resolve(),
            config=config,
            split_manifest=split_manifest,
            cache_receipt_mode=cache_receipt_mode,
        )
    )
    training = execution_config["training"]
    examples = prepare_control_examples(
        optimization_states,
        normalization_floor=float(training["normalization_floor"]),
        maximum_base_cardinality=int(training["maximum_base_cardinality"]),
    )
    observed_optimizer_inventory = formal_group_inventory(examples)
    if observed_optimizer_inventory != training["exact_optimizer_inventory"]:
        raise ValueError("Set Transformer exact optimizer inventory drifted")
    minimums = training["minimum_inventory"]
    if (
        len({row["state_id"] for row in examples})
        < int(minimums["optimizer_state_count"])
        or len(examples) < int(minimums["complete_group_count"])
        or len(heldout_states) != int(minimums["heldout_state_count"])
    ):
        raise ValueError("Set Transformer control inventory is below frozen minimums")
    seed = int(training["seed"])
    _seed_training_runtime(torch, seed)
    torch.set_float32_matmul_precision("high")
    attention_backend = _configure_attention_backend(torch, training)
    checkpoint_model = runtime.build_model(
        variant=variant,
        device=device,
        torch=torch,
    )
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            checkpoint_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=runtime.ddp_find_unused_parameters(),
        )
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed(seed + rank)
    else:
        model = checkpoint_model
    parameter_groups = runtime.optimizer_parameter_groups(
        checkpoint_model, variant=variant
    )
    parameters = tuple(
        parameter for group in parameter_groups for parameter in group["params"]
    )
    if not parameters or len({id(parameter) for parameter in parameters}) != len(
        parameters
    ):
        raise ValueError("control optimizer parameters are empty or duplicated")
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(variant["weight_decay"]),
    )
    epochs = int(training["epochs"])
    batch_size = int(variant["per_device_batch_size"])
    accumulation = int(variant["gradient_accumulation_steps"])
    balancing = training["balancing"]
    packed_counts = [
        len(
            _epoch_examples(
                examples,
                epoch=epoch,
                seed=seed,
                balancing=balancing,
            )
        )
        for epoch in range(1, epochs + 1)
    ]
    steps_by_epoch = [
        _optimizer_steps(
            count,
            batch_size=batch_size,
            accumulation=accumulation,
            world_size=world_size,
        )
        for count in packed_counts
    ]
    total_steps = sum(steps_by_epoch)
    warmup_steps = max(1, round(total_steps * float(training["warmup_ratio"])))
    minimum_lr_ratio = float(training["minimum_lr_ratio"])

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    identity = {
        "cache_content_sha256": cache_manifest["content_sha256"],
        "config_sha256": _sha256_file(config_path),
        "input_content_sha256": input_manifest["content_sha256"],
        "model_family": runtime.model_family,
        "optimizer_inventory_content_sha256": observed_optimizer_inventory[
            "content_sha256"
        ],
        "split_manifest_content_sha256": split_manifest["content_sha256"],
        "variant": args.variant,
        "world_size": world_size,
        **runtime.identity_fields(),
    }
    output_root = args.output_root.resolve()
    manager = None
    if rank == 0:
        manager = RecoveryCheckpointManager(
            output_root,
            identity=identity,
            selection_split="train_trajectory_holdout",
            **runtime.checkpoint_manager_kwargs(),
        )
        if not args.resume and manager.manifest["epochs"]:
            raise FileExistsError("Set Transformer checkpoint history needs --resume")
    if distributed:
        torch.distributed.barrier()
    (output_root / "heldout-rollouts").mkdir(parents=True, exist_ok=True)
    cache = _TokenCache(
        args.cache_root.resolve(),
        cache_manifest,
        device="cpu",
        mode=str(variant["cache_mode"]),
    )
    heldout_by_id = {row["state_id"]: row for row in heldout_states}
    truth_sources = _truth_sources(args.truth_source)
    if truth_sources and not args.resume:
        _load_supplemental_truth(
            truth_sources,
            allowed_state_ids=set(heldout_by_id),
            states_by_id=heldout_by_id,
            expected_model_family=runtime.model_family,
            expected_epoch_checkpoints={},
            expected_input_content_sha256=input_manifest["content_sha256"],
            expected_heldout_manifest_content_sha256=split_manifest["content_sha256"],
            expected_source_manifest_file_sha256=config["input"][
                "source_manifest_file_sha256"
            ],
        )
    supplemental: dict[str, dict[tuple[int, ...], float]] = {}
    start_epoch = 1
    history = []
    barrier_action = CONTINUE_TRAINING
    if args.resume:
        snapshot, _ = load_resume_generation_collective(
            output_root,
            rank=rank,
            world_size=world_size,
            identity=identity,
            map_location=device,
            distributed=distributed,
            torch=torch,
        )
        runtime.load_resume_model_state(checkpoint_model, snapshot["model"])
        optimizer.load_state_dict(snapshot["optimizer"])
        scheduler.load_state_dict(snapshot["scheduler"])
        random.setstate(snapshot["python_rng_state"])
        restore_rng_states(snapshot, torch=torch)
        start_epoch = int(snapshot["epoch"]) + 1
        completed, _ = _partition_resume_plans(
            _read_epoch_plans(output_root),
            snapshot_epoch=int(snapshot["epoch"]),
        )
        supplemental = _load_supplemental_truth(
            truth_sources,
            allowed_state_ids=set(heldout_by_id),
            states_by_id=heldout_by_id,
            expected_model_family=runtime.model_family,
            expected_epoch_checkpoints=_plan_checkpoint_bindings(completed),
            expected_input_content_sha256=input_manifest["content_sha256"],
            expected_heldout_manifest_content_sha256=split_manifest["content_sha256"],
            expected_source_manifest_file_sha256=config["input"][
                "source_manifest_file_sha256"
            ],
        )
        if rank == 0:
            progress_path = output_root / "training-progress.json"
            if not progress_path.is_file():
                raise FileNotFoundError("Set Transformer resume progress is missing")
            progress = _read_signed_json(progress_path)
            if progress.get("identity") != identity:
                raise ValueError("Set Transformer resume progress identity drifted")
            history = [
                dict(row)
                for row in progress.get("history", ())
                if int(row["epoch"]) <= int(snapshot["epoch"])
            ]
            if [int(row["epoch"]) for row in history] != list(
                range(1, int(snapshot["epoch"]) + 1)
            ):
                raise ValueError("Set Transformer resume progress epochs drifted")
            rebuilt = _rebuild_plans(
                completed,
                output_root=output_root,
                states_by_id=heldout_by_id,
                split_manifest=split_manifest,
                normalization_floor=float(
                    config["checkpoint_selection"]["normalization_floor"]
                ),
                supplemental=supplemental,
            )
            schedule = _publish_schedule(
                output_root, rebuilt, model_family=runtime.model_family
            )
            barrier_action, selection = _selection(rebuilt, execution_config)
            if history:
                history[-1]["selection"] = selection
                history[-1]["truth_missing_coalition_count"] = schedule[
                    "missing_coalition_count"
                ]
            progress = {
                "attention_backend": attention_backend,
                "balancing_inventory": control_balancing_inventory(examples),
                "history": history,
                "identity": identity,
                "runtime": runtime.runtime_metadata(),
                "schema_version": runtime.progress_schema,
                "status": barrier_action,
            }
            _write_atomic(output_root / "training-progress.json", _signed(progress))
        if distributed:
            payload = [barrier_action]
            torch.distributed.broadcast_object_list(payload, src=0)
            barrier_action = str(payload[0])
    elif has_published_resume_generation(output_root):
        raise FileExistsError(
            "Set Transformer resume generation exists without --resume"
        )

    termination_reason = "maximum_epochs"
    if barrier_action == WAITING_FOR_TRUTH:
        termination_reason = "heldout_truth_barrier"
        start_epoch = epochs + 1
    elif barrier_action == EARLY_STOP:
        termination_reason = "true_recovery_early_stopping"
        start_epoch = epochs + 1
    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        runtime.set_training_mode(checkpoint_model)
        global_examples = _epoch_examples(
            examples, epoch=epoch, seed=seed, balancing=balancing
        )
        if len(global_examples) != packed_counts[epoch - 1]:
            raise RuntimeError("Set Transformer packed epoch geometry drifted")
        if distributed:
            epoch_examples, _ = _distributed_epoch_shard(
                global_examples,
                per_device_batch_size=batch_size,
                gradient_accumulation_steps=accumulation,
                rank=rank,
                world_size=world_size,
            )
        else:
            padding = (-len(global_examples)) % (batch_size * accumulation)
            epoch_examples = global_examples + tuple(
                global_examples[index % len(global_examples)]
                for index in range(padding)
            )
        optimizer.zero_grad(set_to_none=True)
        totals = defaultdict(float)
        denominator = 0.0
        batches = iter(
            _batch_stream(
                epoch_examples,
                batch_size=batch_size,
                cache=cache,
                device=device,
                torch=torch,
                normalization_floor=float(training["normalization_floor"]),
            )
        )
        while True:
            raw_window = tuple(islice(batches, accumulation))
            if not raw_window:
                break
            window = tuple(
                (
                    selected,
                    _attach_conditional_supervision(batch, selected, torch=torch),
                )
                for selected, batch in raw_window
            )
            local_window_groups = sum(
                float(batch["conditional_group_mask"].sum()) for _, batch in window
            )
            group_denominator = torch.tensor(
                local_window_groups, dtype=torch.float64, device=device
            )
            if distributed:
                torch.distributed.all_reduce(
                    group_denominator, op=torch.distributed.ReduceOp.SUM
                )
            backward_scale = _group_sum_backward_scale(
                global_group_count=float(group_denominator.item()),
                world_size=world_size,
            )
            for window_index, (_, batch) in enumerate(window):
                synchronize = window_index + 1 == len(window)
                sync_context = (
                    nullcontext() if not distributed or synchronize else model.no_sync()
                )
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predictions = model(**batch["model"])
                        loss_sum, metrics = group_equal_direct_loss(
                            predictions,
                            batch,
                            loss_config=training["loss"],
                            reduction="sum",
                            torch=torch,
                        )
                    (loss_sum * backward_scale).backward()
                group_weight = float(batch["conditional_group_mask"].sum())
                denominator += group_weight
                for key, value in metrics.items():
                    if key != "conditional_group_count":
                        totals[key] += value * group_weight
            torch.nn.utils.clip_grad_norm_(
                parameters, float(training["maximum_gradient_norm"])
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        train_metrics = (
            _distributed_metric_average(totals, denominator, device=device, torch=torch)
            if distributed
            else {key: value / denominator for key, value in sorted(totals.items())}
        )
        epoch_group_count = torch.tensor(
            denominator, dtype=torch.float64, device=device
        )
        if distributed:
            torch.distributed.all_reduce(
                epoch_group_count, op=torch.distributed.ReduceOp.SUM
            )
        train_metrics["conditional_group_count"] = float(epoch_group_count.item())
        if not all(math.isfinite(value) for value in train_metrics.values()):
            raise RuntimeError("Set Transformer control produced non-finite metrics")

        barrier_action = CONTINUE_TRAINING
        if rank == 0:
            assert manager is not None
            checkpoint = _save_or_verify_epoch_checkpoint(
                manager,
                checkpoint_model,
                epoch=epoch,
                output_root=output_root,
            )
            existing_plan = _read_existing_epoch_plan(
                output_root, epoch=epoch, checkpoint=checkpoint
            )
            rollout_records = (
                existing_plan["records"]
                if existing_plan is not None
                else _rollout_heldout(
                    checkpoint_model,
                    heldout_states,
                    cache=cache,
                    batch_size=int(variant["heldout_batch_size"]),
                    device=device,
                    normalization_floor=float(training["normalization_floor"]),
                    torch=torch,
                )
            )
            plan = build_control_epoch_truth(
                epoch=epoch,
                checkpoint=checkpoint,
                rollout_records=rollout_records,
                states_by_id=heldout_by_id,
                split_manifest=split_manifest,
                normalization_floor=float(
                    config["checkpoint_selection"]["normalization_floor"]
                ),
                supplemental=supplemental,
            )
            _write_atomic(_epoch_plan_path(output_root, epoch), _signed(plan))
            plans = _read_epoch_plans(output_root)
            schedule = _publish_schedule(
                output_root, plans, model_family=runtime.model_family
            )
            barrier_action, selection = _selection(plans, execution_config)
            epoch_record = {
                "barrier_action": barrier_action,
                "epoch": epoch,
                "learning_rate": scheduler.get_last_lr()[0],
                "packed_state_encoding_count": len(global_examples),
                "sampled_group_count": int(balancing["samples_per_epoch"]),
                "selection": selection,
                "train": train_metrics,
                "truth_missing_coalition_count": schedule["missing_coalition_count"],
            }
            history.append(epoch_record)
            progress = {
                "attention_backend": attention_backend,
                "balancing_inventory": control_balancing_inventory(examples),
                "history": history,
                "identity": identity,
                "runtime": runtime.runtime_metadata(),
                "schema_version": runtime.progress_schema,
                "status": barrier_action,
            }
            _write_atomic(output_root / "training-progress.json", _signed(progress))
            print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if distributed:
            payload = [barrier_action]
            torch.distributed.broadcast_object_list(payload, src=0)
            barrier_action = str(payload[0])
            torch.distributed.barrier()
        save_resume_generation_collective(
            output_root,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            model=checkpoint_model,
            optimizer=optimizer,
            scheduler=scheduler,
            identity=identity,
            distributed=distributed,
            torch=torch,
        )
        if barrier_action == WAITING_FOR_TRUTH:
            termination_reason = "heldout_truth_barrier"
            break
        if barrier_action == EARLY_STOP:
            termination_reason = "true_recovery_early_stopping"
            break

    if rank == 0:
        plans = _read_epoch_plans(output_root)
        barrier_action, selection = _selection(plans, execution_config)
        summary = {
            "attention_backend": attention_backend,
            "balancing_inventory": control_balancing_inventory(examples),
            "checkpoint_count": len(plans),
            "elapsed_seconds": time.time() - started,
            "evaluation_or_test_records_loaded": False,
            "heldout_state_count": len(heldout_states),
            "identity": identity,
            "optimization_group_count": len(examples),
            "runtime": runtime.runtime_metadata(),
            "schema_version": runtime.summary_schema,
            "selection": selection,
            "selected_checkpoint": _selected_checkpoint(plans, selection),
            "status": (
                WAITING_FOR_TRUTH
                if barrier_action == WAITING_FOR_TRUTH
                else runtime.completed_status
            ),
            "termination_reason": termination_reason,
        }
        _write_atomic(output_root / "summary.json", _signed(summary))
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _finalize(
    args: argparse.Namespace,
    *,
    runtime: SetTransformerControlRuntime | None = None,
) -> None:
    runtime = runtime or SetTransformerControlRuntime()
    config = _read_json(args.config.resolve())
    runtime.validate_config(config, args.variant)
    execution_config = runtime.execution_config(config)
    split_manifest = _load_split_manifest(args.split_manifest, config)
    if split_manifest is None:
        raise ValueError("Set Transformer finalize requires the frozen split")
    input_manifest, _, _, heldout_states = _load_training_inputs(
        input_root=args.input_root.resolve(),
        cache_root=args.cache_root.resolve(),
        config=config,
        split_manifest=split_manifest,
    )
    heldout_by_id = {row["state_id"]: row for row in heldout_states}
    output_root = args.output_root.resolve()
    originals = _read_epoch_plans(output_root)
    if not originals:
        raise ValueError("Set Transformer finalize found no epoch plans")
    supplemental = _load_supplemental_truth(
        _truth_sources(args.truth_source),
        allowed_state_ids=set(heldout_by_id),
        states_by_id=heldout_by_id,
        expected_model_family=runtime.model_family,
        expected_epoch_checkpoints=_plan_checkpoint_bindings(originals),
        expected_input_content_sha256=input_manifest["content_sha256"],
        expected_heldout_manifest_content_sha256=split_manifest["content_sha256"],
        expected_source_manifest_file_sha256=config["input"][
            "source_manifest_file_sha256"
        ],
    )
    rebuilt = _rebuild_plans(
        originals,
        output_root=output_root,
        states_by_id=heldout_by_id,
        split_manifest=split_manifest,
        normalization_floor=float(
            config["checkpoint_selection"]["normalization_floor"]
        ),
        supplemental=supplemental,
    )
    schedule = _publish_schedule(
        output_root, rebuilt, model_family=runtime.model_family
    )
    barrier_action, selection = _selection(rebuilt, execution_config)
    complete = bool(
        barrier_action != WAITING_FOR_TRUTH and schedule["missing_coalition_count"] == 0
    )
    result = {
        "checkpoint_count": len(rebuilt),
        "input_content_sha256": input_manifest["content_sha256"],
        "missing_coalition_count": schedule["missing_coalition_count"],
        "runtime": runtime.runtime_metadata(),
        "schema_version": runtime.selection_schema,
        "selection": selection,
        "barrier_action": barrier_action,
        "selected_checkpoint": _selected_checkpoint(rebuilt, selection),
        "status": (
            runtime.finalized_status if complete else runtime.pending_finalized_status
        ),
    }
    _write_atomic(output_root / "delayed-selection.json", _signed(result))
    print(
        json.dumps({"selection": selection, "status": result["status"]}, sort_keys=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "finalize"):
        child = subparsers.add_parser(command)
        child.add_argument("--input-root", type=Path, required=True)
        child.add_argument("--cache-root", type=Path, required=True)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--variant", required=True)
        child.add_argument("--split-manifest", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--truth-source", action="append", default=[])
    fit = subparsers.choices["fit"]
    fit.add_argument("--device", required=True)
    fit.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "fit":
        _fit(args)
    else:
        _finalize(args)


if __name__ == "__main__":
    main()
