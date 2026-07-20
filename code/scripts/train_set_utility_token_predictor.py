#!/usr/bin/env python3
"""Train one rich Set Transformer or DeepSets variant on train/tune only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from causalcache.set_utility_mvp import canonical_json_bytes
from causalcache.set_utility_token_models import (
    TokenSetUtilityPredictor,
    TokenUtilityModelConfig,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("training state rows must be JSON objects")
            rows.append(value)
    return tuple(rows)


def _cache_covers_input(
    input_manifest: dict[str, Any], cache_manifest: dict[str, Any]
) -> bool:
    cache_input = cache_manifest.get("input_content_sha256")
    return cache_input in {
        input_manifest.get("content_sha256"),
        input_manifest.get("parent_content_sha256"),
    }


class _TokenCache:
    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        *,
        device: Any = "cpu",
    ) -> None:
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as error:
            raise RuntimeError("token predictor training requires safetensors") from error
        self.tensors: dict[str, Any] = {}
        by_shard: dict[Path, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for cache_key, record in manifest["tensor_inventory"].items():
            path = root / record["partition"] / record["shard"]
            by_shard[path].append((cache_key, record))
        for path in sorted(by_shard):
            loaded = load_file(str(path), device=str(device))
            for cache_key, record in by_shard[path]:
                tensor = loaded[record["tensor"]]
                if list(tensor.shape) != record["shape"] or str(tensor.dtype) != record["dtype"]:
                    raise ValueError("cached token tensor metadata drifted")
                self.tensors[cache_key] = tensor
        if len(self.tensors) != len(manifest["tensor_inventory"]):
            raise ValueError("token cache preload omitted an inventory entry")
        self.device = device

    def visual(self, key: str) -> Any:
        return self.tensors[f"visual:{key}"]

    def text(self, key: str) -> Any:
        return self.tensors[f"text:{key}"]


def _targets(
    state: dict[str, Any],
    torch: Any,
    *,
    normalization_floor: float = 0.0,
) -> tuple[Any, Any, Any, Any]:
    event_ids = tuple(state["candidate_event_step_ids"])
    if (
        not event_ids
        or event_ids != tuple(sorted(event_ids))
        or len(event_ids) != len(set(event_ids))
    ):
        raise ValueError("candidate events must be sorted, unique, and non-empty")
    rows = tuple(state["distance_rows"])
    observed = tuple(tuple(row["coalition_event_step_ids"]) for row in rows)
    if not rows or observed[0] != ():
        raise ValueError("the empty coalition must be the first distance row")
    if len(observed) != len(set(observed)):
        raise ValueError("distance rows contain duplicate coalitions")
    universe = set(event_ids)
    if any(
        subset != tuple(sorted(subset))
        or len(subset) != len(set(subset))
        or not set(subset).issubset(universe)
        for subset in observed
    ):
        raise ValueError("a distance-row coalition is invalid for its candidate universe")
    baseline = float(rows[0]["distance"])
    if baseline < -1e-8:
        raise ValueError("restoration distance cannot be negative")
    if normalization_floor < 0.0:
        raise ValueError("normalization floor cannot be negative")
    raw = torch.tensor(
        [baseline - float(row["distance"]) for row in rows], dtype=torch.float32
    )
    raw[0] = 0.0
    scale_is_valid = baseline > 1e-12 or normalization_floor > 0.0
    scale = max(baseline, normalization_floor) if scale_is_valid else 1.0
    normalized = raw / scale
    subset_masks = torch.tensor(
        [[event_id in subset for event_id in event_ids] for subset in observed],
        dtype=torch.bool,
    )
    return subset_masks, raw, normalized, (scale, scale_is_valid)


def _collate(
    states: list[dict[str, Any]],
    *,
    cache: _TokenCache,
    device: Any,
    torch: Any,
    normalization_floor: float = 0.0,
) -> dict[str, Any]:
    if not states:
        raise ValueError("cannot collate an empty state batch")
    batch_size = len(states)
    query_visual_rows = [
        cache.visual(state["current_image_key"]) for state in states
    ]
    event_visual_rows = [
        [cache.visual(key) for key in state["event_image_keys"]] for state in states
    ]
    query_text_rows = [cache.text(state["instruction_text_key"]) for state in states]
    event_text_rows = [
        [cache.text(key) for key in state["event_text_keys"]] for state in states
    ]
    event_counts = [len(row) for row in event_visual_rows]
    if any(count <= 0 for count in event_counts):
        raise ValueError("every state must contain at least one candidate event")
    for state, visual_rows, text_rows in zip(
        states, event_visual_rows, event_text_rows, strict=True
    ):
        count = len(state["candidate_event_step_ids"])
        if not (
            len(visual_rows)
            == len(text_rows)
            == len(state["event_numeric_features"])
            == count
        ):
            raise ValueError("event features do not align with the candidate universe")
    event_count = max(event_counts)
    query_text_length = max(row.shape[0] for row in query_text_rows)
    event_text_length = max(
        row.shape[0] for rows in event_text_rows for row in rows
    )
    hidden = query_visual_rows[0].shape[-1]
    query_visual_length = max(row.shape[0] for row in query_visual_rows)
    event_visual_length = max(
        row.shape[0] for rows in event_visual_rows for row in rows
    )
    cache_device = query_visual_rows[0].device
    query_visual = torch.zeros(
        (batch_size, query_visual_length, hidden),
        dtype=torch.bfloat16,
        device=cache_device,
    )
    query_visual_mask = torch.zeros(
        (batch_size, query_visual_length), dtype=torch.bool, device=cache_device
    )
    event_visual = torch.zeros(
        (batch_size, event_count, event_visual_length, hidden),
        dtype=torch.bfloat16,
        device=cache_device,
    )
    event_visual_mask = torch.zeros(
        (batch_size, event_count, event_visual_length),
        dtype=torch.bool,
        device=cache_device,
    )
    query_text = torch.zeros(
        (batch_size, query_text_length, hidden),
        dtype=torch.bfloat16,
        device=cache_device,
    )
    query_text_mask = torch.zeros(
        (batch_size, query_text_length), dtype=torch.bool, device=cache_device
    )
    event_text = torch.zeros(
        (batch_size, event_count, event_text_length, hidden),
        dtype=torch.bfloat16,
        device=cache_device,
    )
    event_text_mask = torch.zeros(
        (batch_size, event_count, event_text_length),
        dtype=torch.bool,
        device=cache_device,
    )
    for batch_index, row in enumerate(query_visual_rows):
        query_visual[batch_index, : row.shape[0]] = row
        query_visual_mask[batch_index, : row.shape[0]] = True
    for batch_index, rows in enumerate(event_visual_rows):
        for event_index, row in enumerate(rows):
            event_visual[batch_index, event_index, : row.shape[0]] = row
            event_visual_mask[batch_index, event_index, : row.shape[0]] = True
    for batch_index, row in enumerate(query_text_rows):
        query_text[batch_index, : row.shape[0]] = row
        query_text_mask[batch_index, : row.shape[0]] = True
    for batch_index, rows in enumerate(event_text_rows):
        for event_index, row in enumerate(rows):
            event_text[batch_index, event_index, : row.shape[0]] = row
            event_text_mask[batch_index, event_index, : row.shape[0]] = True
    target_rows = [
        _targets(state, torch, normalization_floor=normalization_floor)
        for state in states
    ]
    subset_count = max(row[0].shape[0] for row in target_rows)
    event_numeric_features = torch.zeros(
        (batch_size, event_count, len(states[0]["event_numeric_features"][0])),
        dtype=torch.float32,
        device=device,
    )
    event_mask = torch.zeros(
        (batch_size, event_count), dtype=torch.bool, device=device
    )
    subset_masks = torch.zeros(
        (batch_size, subset_count, event_count), dtype=torch.bool, device=device
    )
    raw_targets = torch.zeros(
        (batch_size, subset_count), dtype=torch.float32, device=device
    )
    normalized_targets = torch.zeros_like(raw_targets)
    label_mask = torch.zeros(
        (batch_size, subset_count), dtype=torch.bool, device=device
    )
    for batch_index, (state, target_row) in enumerate(
        zip(states, target_rows, strict=True)
    ):
        current_event_count = event_counts[batch_index]
        current_subset_count = target_row[0].shape[0]
        numeric = torch.tensor(
            state["event_numeric_features"], dtype=torch.float32, device=device
        )
        if numeric.ndim != 2 or numeric.shape[1] != event_numeric_features.shape[2]:
            raise ValueError("event numeric feature width drifted within a batch")
        event_numeric_features[batch_index, :current_event_count] = numeric
        event_mask[batch_index, :current_event_count] = True
        subset_masks[
            batch_index, :current_subset_count, :current_event_count
        ] = target_row[0].to(device)
        raw_targets[batch_index, :current_subset_count] = target_row[1].to(device)
        normalized_targets[batch_index, :current_subset_count] = target_row[2].to(
            device
        )
        label_mask[batch_index, :current_subset_count] = True
    return {
        "model": {
            "query_visual_tokens": query_visual.to(device),
            "query_visual_mask": query_visual_mask.to(device),
            "query_text_tokens": query_text.to(device),
            "query_text_mask": query_text_mask.to(device),
            "event_visual_tokens": event_visual.to(device),
            "event_visual_mask": event_visual_mask.to(device),
            "event_text_tokens": event_text.to(device),
            "event_text_mask": event_text_mask.to(device),
            "event_numeric_features": event_numeric_features,
            "event_mask": event_mask,
            "subset_masks": subset_masks,
        },
        "raw_targets": raw_targets,
        "normalized_targets": normalized_targets,
        "label_mask": label_mask,
        "scales": torch.tensor(
            [row[3][0] for row in target_rows], dtype=torch.float32, device=device
        ),
        "scale_mask": torch.tensor(
            [row[3][1] for row in target_rows], dtype=torch.bool, device=device
        ),
        "state_ids": tuple(state["state_id"] for state in states),
        "trajectory_ids": tuple(state["trajectory_id"] for state in states),
    }


def _loss(
    predictions: Any,
    batch: dict[str, Any],
    *,
    loss_config: dict[str, Any],
    trajectory_weights: Any,
    torch: Any,
) -> tuple[Any, dict[str, float]]:
    raw_targets = batch["raw_targets"]
    normalized_targets = batch["normalized_targets"]
    scales = batch["scales"]
    scale_mask = batch["scale_mask"]
    label_mask = batch["label_mask"]
    label_weights = label_mask.to(predictions.dtype)
    labels_per_state = label_weights.sum(dim=1).clamp_min(1)
    raw_terms = torch.nn.functional.smooth_l1_loss(
        predictions,
        raw_targets,
        reduction="none",
        beta=float(loss_config["raw_smooth_l1_beta"]),
    )
    raw_rows = (raw_terms * label_weights).sum(dim=1) / labels_per_state
    normalized_predictions = predictions / scales.unsqueeze(1)
    normalized_terms = torch.nn.functional.smooth_l1_loss(
        normalized_predictions,
        normalized_targets,
        reduction="none",
        beta=float(loss_config["normalized_smooth_l1_beta"]),
    )
    normalized_rows = (
        normalized_terms * label_weights
    ).sum(dim=1) / labels_per_state
    normalized_rows = normalized_rows * scale_mask.to(normalized_rows.dtype)
    pair_mask = torch.triu(
        torch.ones(
            predictions.shape[1],
            predictions.shape[1],
            dtype=torch.bool,
            device=predictions.device,
        ),
        diagonal=1,
    ).unsqueeze(0)
    target_differences = normalized_targets.unsqueeze(2) - normalized_targets.unsqueeze(1)
    signs = torch.sign(target_differences)
    valid_pairs = label_mask.unsqueeze(2) & label_mask.unsqueeze(1)
    untied = pair_mask & valid_pairs & (torch.abs(target_differences) > 1e-6)
    prediction_differences = normalized_predictions.unsqueeze(2) - normalized_predictions.unsqueeze(1)
    ranking_terms = torch.nn.functional.softplus(
        -prediction_differences * signs
    )
    ranking_rows = (ranking_terms * untied).sum(dim=(1, 2)) / untied.sum(
        dim=(1, 2)
    ).clamp_min(1)
    base_masks = batch["model"]["subset_masks"].unsqueeze(2)
    expanded_masks = batch["model"]["subset_masks"].unsqueeze(1)
    one_event_expansions = (
        ((base_masks & ~expanded_masks).sum(dim=3) == 0)
        & (
            expanded_masks.sum(dim=3).to(torch.int64)
            - base_masks.sum(dim=3).to(torch.int64)
            == 1
        )
        & label_mask.unsqueeze(2)
        & label_mask.unsqueeze(1)
    )
    target_marginals = normalized_targets.unsqueeze(1) - normalized_targets.unsqueeze(2)
    predicted_marginals = (
        normalized_predictions.unsqueeze(1) - normalized_predictions.unsqueeze(2)
    )
    marginal_terms = torch.nn.functional.smooth_l1_loss(
        predicted_marginals,
        target_marginals,
        reduction="none",
        beta=float(loss_config.get("conditional_marginal_smooth_l1_beta", 0.25)),
    )
    marginal_rows = (
        (marginal_terms * one_event_expansions).sum(dim=(1, 2))
        / one_event_expansions.sum(dim=(1, 2)).clamp_min(1)
    )
    marginal_rows = marginal_rows * scale_mask.to(marginal_rows.dtype)
    decision_temperature = float(loss_config.get("decision_temperature", 0.25))
    if decision_temperature <= 0.0:
        raise ValueError("decision temperature must be positive")
    base_cardinality = batch["model"]["subset_masks"].sum(dim=2)
    event_mask = batch["model"].get("event_mask")
    event_count = (
        event_mask.sum(dim=1, keepdim=True)
        if event_mask is not None
        else torch.full_like(
            base_cardinality[:, :1], batch["model"]["subset_masks"].shape[2]
        )
    )
    expansion_count = one_event_expansions.sum(dim=2)
    complete_groups = expansion_count == (event_count - base_cardinality)
    group_mask = (
        one_event_expansions.any(dim=2)
        & label_mask
        & complete_groups
        & scale_mask.unsqueeze(1)
    )
    masked_teacher = (target_marginals / decision_temperature).masked_fill(
        ~one_event_expansions, -1e9
    )
    masked_student = (predicted_marginals / decision_temperature).masked_fill(
        ~one_event_expansions, -1e9
    )
    stop_logits = torch.zeros(
        (*predictions.shape, 1),
        dtype=predictions.dtype,
        device=predictions.device,
    )
    teacher_logits = torch.cat((masked_teacher, stop_logits), dim=2)
    student_logits = torch.cat((masked_student, stop_logits), dim=2)
    teacher_probabilities = torch.softmax(teacher_logits, dim=2)
    student_log_probabilities = torch.log_softmax(student_logits, dim=2)
    listwise_groups = -(
        teacher_probabilities * student_log_probabilities
    ).sum(dim=2)
    listwise_rows = (
        (listwise_groups * group_mask).sum(dim=1)
        / group_mask.sum(dim=1).clamp_min(1)
    )
    teacher_values = torch.cat((target_marginals, stop_logits), dim=2)
    action_mask = torch.cat(
        (
            one_event_expansions,
            label_mask.unsqueeze(2),
        ),
        dim=2,
    )
    best_teacher = teacher_values.masked_fill(~action_mask, -1e9).max(
        dim=2, keepdim=True
    ).values
    student_probabilities = torch.softmax(
        student_logits.masked_fill(~action_mask, -1e9), dim=2
    )
    regret_groups = (
        student_probabilities
        * (best_teacher - teacher_values).clamp_min(0.0)
        * action_mask
    ).sum(dim=2)
    regret_rows = (
        (regret_groups * group_mask).sum(dim=1)
        / group_mask.sum(dim=1).clamp_min(1)
    )
    weights = trajectory_weights / trajectory_weights.sum()
    raw = torch.sum(raw_rows * weights)
    normalized = torch.sum(normalized_rows * weights)
    ranking = torch.sum(ranking_rows * weights)
    conditional_marginal = torch.sum(marginal_rows * weights)
    active_decision_rows = group_mask.any(dim=1).to(trajectory_weights.dtype)
    decision_weights = trajectory_weights * active_decision_rows
    decision_denominator = decision_weights.sum().clamp_min(1.0)
    conditional_listwise = torch.sum(
        listwise_rows * decision_weights
    ) / decision_denominator
    decision_regret = torch.sum(regret_rows * decision_weights) / decision_denominator
    total = (
        float(loss_config["raw_regression"]) * raw
        + float(loss_config["normalized_regression"]) * normalized
        + float(loss_config["within_state_ranking"]) * ranking
        + float(loss_config.get("conditional_marginal", 0.0))
        * conditional_marginal
        + float(loss_config.get("conditional_listwise", 0.0))
        * conditional_listwise
        + float(loss_config.get("decision_regret", 0.0)) * decision_regret
    )
    correct = ((prediction_differences * signs) > 0) & untied
    ranking_accuracy = correct.sum() / untied.sum().clamp_min(1)
    return total, {
        "complete_decision_group_count": float(group_mask.sum().detach()),
        "conditional_listwise": float(conditional_listwise.detach()),
        "conditional_marginal_regression": float(conditional_marginal.detach()),
        "decision_regret": float(decision_regret.detach()),
        "normalized_regression": float(normalized.detach()),
        "ranking_accuracy": float(ranking_accuracy.detach()),
        "raw_mae": float(
            (
                (torch.abs(predictions - raw_targets) * label_weights).sum()
                / label_weights.sum().clamp_min(1)
            ).detach()
        ),
        "raw_regression": float(raw.detach()),
        "total": float(total.detach()),
    }


def _trajectory_weights(states: tuple[dict[str, Any], ...]) -> dict[str, float]:
    counts = Counter(state["trajectory_id"] for state in states)
    normalizer = len(states) / len(counts)
    return {
        state_id: normalizer / counts[trajectory_id]
        for state_id, trajectory_id in (
            (state["state_id"], state["trajectory_id"]) for state in states
        )
    }


def _trajectory_uniform_epoch(
    states: tuple[dict[str, Any], ...],
    *,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        by_trajectory[state["trajectory_id"]].append(state)
    generator = random.Random(seed)
    trajectories = sorted(by_trajectory)
    generator.shuffle(trajectories)
    for trajectory in trajectories:
        generator.shuffle(by_trajectory[trajectory])
    maximum_count = max(len(rows) for rows in by_trajectory.values())
    return tuple(
        by_trajectory[trajectory][round_index % len(by_trajectory[trajectory])]
        for round_index in range(maximum_count)
        for trajectory in trajectories
    )


def _seed_training_runtime(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _configure_attention_backend(torch: Any, training: dict[str, Any]) -> str:
    profile = str(training.get("attention_backend", "default"))
    if profile == "default":
        return profile
    if profile != "disable_cudnn_sdp":
        raise ValueError("unknown training attention backend profile")
    enable = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
    enabled = getattr(torch.backends.cuda, "cudnn_sdp_enabled", None)
    if enable is None or enabled is None:
        raise RuntimeError("PyTorch does not expose the cuDNN SDP backend controls")
    enable(False)
    if enabled():
        raise RuntimeError("cuDNN SDP backend remained enabled")
    return profile


def _resolve_normalization_floor(
    training: dict[str, Any], override: float | None
) -> float:
    configured = training.get("normalization_floor")
    if configured is None:
        value = 0.0 if override is None else float(override)
    else:
        value = float(configured)
        if override is not None and not math.isclose(
            value, float(override), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("normalization floor conflicts with committed config")
    if value < 0.0:
        raise ValueError("normalization floor cannot be negative")
    return value


def _evaluate(
    model: Any,
    states: tuple[dict[str, Any], ...],
    *,
    cache: _TokenCache,
    batch_size: int,
    device: Any,
    loss_config: dict[str, Any],
    weights_by_state: dict[str, float],
    torch: Any,
    normalization_floor: float,
) -> dict[str, float]:
    totals = defaultdict(float)
    denominator = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            selected = list(states[start : start + batch_size])
            batch = _collate(
                selected,
                cache=cache,
                device=device,
                torch=torch,
                normalization_floor=normalization_floor,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(**batch["model"])
                sample_weights = torch.tensor(
                    [weights_by_state[state["state_id"]] for state in selected],
                    dtype=torch.float32,
                    device=device,
                )
                _, metrics = _loss(
                    predictions,
                    batch,
                    loss_config=loss_config,
                    trajectory_weights=sample_weights,
                    torch=torch,
                )
            weight = float(sample_weights.sum())
            denominator += weight
            for key, value in metrics.items():
                totals[key] += value * weight
    return {key: value / denominator for key, value in sorted(totals.items())}


def _save_checkpoint(model: Any, path: Path) -> dict[str, Any]:
    from safetensors.torch import save_file

    state = {
        key: value.detach().to("cpu").contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state, str(path))
    payload = path.read_bytes()
    return {
        "byte_count": len(payload),
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--overfit-state-count", type=int, default=0)
    parser.add_argument("--normalization-floor", type=float)
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("token predictor training requires PyTorch") from error
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise RuntimeError("token predictor training requires an explicit CUDA device")
    config_path = args.config.resolve()
    config = _read_json(config_path)
    try:
        variant = config["variants"][args.variant]
    except KeyError as error:
        raise ValueError(f"unknown training variant: {args.variant}") from error
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"training output already exists: {output_root}")
    output_root.mkdir(parents=True)

    input_root = args.input_root.resolve()
    input_manifest = _read_json(input_root / "manifest.json")
    cache_root = args.cache_root.resolve()
    cache_manifest = _read_json(cache_root / "manifest.json")
    if (
        input_manifest.get("evaluation_labels_included") is not False
        or cache_manifest.get("evaluation_labels_included") is not False
        or not _cache_covers_input(input_manifest, cache_manifest)
    ):
        raise ValueError("training input/cache identity or split firewall drifted")
    states = _read_jsonl(input_root / input_manifest["states_jsonl"])
    if any(state["role"] not in {"train", "tune"} for state in states):
        raise ValueError("token pilot received a non-train/tune state")
    train_states = tuple(state for state in states if state["role"] == "train")
    tune_states = tuple(state for state in states if state["role"] == "tune")
    if args.overfit_state_count:
        if args.overfit_state_count < 2:
            raise ValueError("overfit state count must be at least two")
        ranked = sorted(
            train_states,
            key=lambda state: hashlib.sha256(state["state_id"].encode()).hexdigest(),
        )[: args.overfit_state_count]
        train_states = tuple(ranked)
        tune_states = tuple(ranked)
    if not train_states or not tune_states:
        raise ValueError("token pilot requires non-empty train and tune roles")

    training = config["training"]
    attention_backend = _configure_attention_backend(torch, training)
    normalization_floor = _resolve_normalization_floor(
        training, args.normalization_floor
    )
    seed = int(variant.get("seed", training["seed"]))
    _seed_training_runtime(torch, seed)
    torch.set_float32_matmul_precision("high")
    cache = _TokenCache(cache_root, cache_manifest, device=args.device)
    model_config = TokenUtilityModelConfig(**variant["model"])
    model = TokenSetUtilityPredictor(model_config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(variant["learning_rate"]),
        weight_decay=float(variant["weight_decay"]),
    )
    epochs = int(training["epochs"])
    batch_size = int(variant["batch_size"])
    accumulation = int(variant["gradient_accumulation_steps"])
    optimization_rows_per_epoch = len(
        _trajectory_uniform_epoch(train_states, seed=seed)
    )
    steps_per_epoch = math.ceil(
        math.ceil(optimization_rows_per_epoch / batch_size) / accumulation
    )
    total_optimizer_steps = steps_per_epoch * epochs
    warmup_steps = max(
        1, round(total_optimizer_steps * float(training["warmup_ratio"]))
    )
    minimum_lr_ratio = float(training["minimum_lr_ratio"])

    def learning_rate_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(
            1, total_optimizer_steps - warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    tune_weights = _trajectory_weights(tune_states)
    history = []
    best_tune = math.inf
    best_epoch = None
    best_checkpoint = None
    patience = 0
    termination_reason = "maximum_epochs"
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(
            _trajectory_uniform_epoch(train_states, seed=seed + epoch)
        )
        optimizer.zero_grad(set_to_none=True)
        train_metric_sum = defaultdict(float)
        train_weight_sum = 0.0
        pending = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = _collate(
                selected,
                cache=cache,
                device=args.device,
                torch=torch,
                normalization_floor=normalization_floor,
            )
            sample_weights = torch.tensor(
                [1.0] * len(selected),
                dtype=torch.float32,
                device=args.device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(**batch["model"])
                loss, metrics = _loss(
                    predictions,
                    batch,
                    loss_config=training["loss"],
                    trajectory_weights=sample_weights,
                    torch=torch,
                )
            (loss / accumulation).backward()
            pending += 1
            if pending == accumulation or start + batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["maximum_gradient_norm"])
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            weight = float(sample_weights.sum())
            train_weight_sum += weight
            for key, value in metrics.items():
                train_metric_sum[key] += value * weight
        train_metrics = {
            key: value / train_weight_sum
            for key, value in sorted(train_metric_sum.items())
        }
        tune_metrics = _evaluate(
            model,
            tune_states,
            cache=cache,
            batch_size=batch_size,
            device=args.device,
            loss_config=training["loss"],
            weights_by_state=tune_weights,
            torch=torch,
            normalization_floor=normalization_floor,
        )
        if not all(
            math.isfinite(value)
            for metrics in (train_metrics, tune_metrics)
            for value in metrics.values()
        ):
            if best_checkpoint is None:
                raise RuntimeError("training became non-finite before a valid checkpoint")
            termination_reason = "nonfinite_metrics"
            break
        history.append(
            {
                "epoch": epoch,
                "learning_rate": scheduler.get_last_lr()[0],
                "train": train_metrics,
                "tune": tune_metrics,
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if tune_metrics["total"] < best_tune - float(training["minimum_delta"]):
            best_tune = tune_metrics["total"]
            best_epoch = epoch
            best_checkpoint = _save_checkpoint(
                model, output_root / "best.safetensors"
            )
            patience = 0
        else:
            patience += 1
        if patience >= int(training["early_stopping_patience"]):
            termination_reason = "early_stopping"
            break

    summary = {
        "best_checkpoint": best_checkpoint,
        "best_epoch": best_epoch,
        "best_tune_total": best_tune,
        "attention_backend": attention_backend,
        "cache_content_sha256": cache_manifest["content_sha256"],
        "cache_input_content_sha256": cache_manifest["input_content_sha256"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "evaluation_records_loaded": False,
        "history": history,
        "input_content_sha256": input_manifest["content_sha256"],
        "input_parent_content_sha256": input_manifest.get(
            "parent_content_sha256"
        ),
        "model": variant["model"],
        "normalization_floor": normalization_floor,
        "overfit_state_count": args.overfit_state_count,
        "optimization_rows_per_epoch": optimization_rows_per_epoch,
        "schema_version": "1.0.0",
        "seed": seed,
        "seed_applied_before_model_initialization": True,
        "status": "COMPLETED_SET_UTILITY_TOKEN_PREDICTOR_TRAINING",
        "termination_reason": termination_reason,
        "train_state_count": len(train_states),
        "train_trajectory_count": len({state["trajectory_id"] for state in train_states}),
        "tune_state_count": len(tune_states),
        "tune_trajectory_count": len({state["trajectory_id"] for state in tune_states}),
        "variant": args.variant,
    }
    summary["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(summary)
    ).hexdigest()
    destination = output_root / "summary.json"
    temporary = destination.with_suffix(f".json.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(summary) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


if __name__ == "__main__":
    main()
