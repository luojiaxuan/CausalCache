#!/usr/bin/env python3
"""Train one rich Set Transformer or DeepSets variant on train/tune only."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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


class _TokenCache:
    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
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
            loaded = load_file(str(path), device="cpu")
            for cache_key, record in by_shard[path]:
                tensor = loaded[record["tensor"]]
                if list(tensor.shape) != record["shape"] or str(tensor.dtype) != record["dtype"]:
                    raise ValueError("cached token tensor metadata drifted")
                self.tensors[cache_key] = tensor
        if len(self.tensors) != len(manifest["tensor_inventory"]):
            raise ValueError("token cache preload omitted an inventory entry")

    def visual(self, key: str) -> Any:
        return self.tensors[f"visual:{key}"]

    def text(self, key: str) -> Any:
        return self.tensors[f"text:{key}"]


def _targets(state: dict[str, Any], torch: Any) -> tuple[Any, Any, Any, Any]:
    event_ids = tuple(state["candidate_event_step_ids"])
    rows = tuple(
        sorted(
            state["distance_rows"],
            key=lambda row: (
                len(row["coalition_event_step_ids"]),
                tuple(row["coalition_event_step_ids"]),
            ),
        )
    )
    expected = tuple(
        subset
        for cardinality in range(3)
        for subset in itertools.combinations(event_ids, cardinality)
    )
    observed = tuple(tuple(row["coalition_event_step_ids"]) for row in rows)
    if observed != expected:
        raise ValueError("training state does not contain the exact |S|<=2 table")
    baseline = float(rows[0]["distance"])
    raw = torch.tensor(
        [baseline - float(row["distance"]) for row in rows], dtype=torch.float32
    )
    raw[0] = 0.0
    scale_is_valid = baseline > 1e-12
    scale = baseline if scale_is_valid else 1.0
    normalized = raw / scale
    subset_masks = torch.tensor(
        [[event_id in subset for event_id in event_ids] for subset in expected],
        dtype=torch.bool,
    )
    return subset_masks, raw, normalized, (scale, scale_is_valid)


def _collate(
    states: list[dict[str, Any]],
    *,
    cache: _TokenCache,
    device: Any,
    torch: Any,
) -> dict[str, Any]:
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
    query_text_length = max(row.shape[0] for row in query_text_rows)
    event_text_length = max(
        row.shape[0] for rows in event_text_rows for row in rows
    )
    hidden = query_visual_rows[0].shape[-1]
    query_visual_length = max(row.shape[0] for row in query_visual_rows)
    event_visual_length = max(
        row.shape[0] for rows in event_visual_rows for row in rows
    )
    query_visual = torch.zeros(
        (batch_size, query_visual_length, hidden), dtype=torch.bfloat16
    )
    query_visual_mask = torch.zeros(
        (batch_size, query_visual_length), dtype=torch.bool
    )
    event_visual = torch.zeros(
        (batch_size, 4, event_visual_length, hidden), dtype=torch.bfloat16
    )
    event_visual_mask = torch.zeros(
        (batch_size, 4, event_visual_length), dtype=torch.bool
    )
    query_text = torch.zeros(
        (batch_size, query_text_length, hidden), dtype=torch.bfloat16
    )
    query_text_mask = torch.zeros(
        (batch_size, query_text_length), dtype=torch.bool
    )
    event_text = torch.zeros(
        (batch_size, 4, event_text_length, hidden), dtype=torch.bfloat16
    )
    event_text_mask = torch.zeros(
        (batch_size, 4, event_text_length), dtype=torch.bool
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
    target_rows = [_targets(state, torch) for state in states]
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
            "event_numeric_features": torch.tensor(
                [state["event_numeric_features"] for state in states],
                dtype=torch.float32,
                device=device,
            ),
            "event_mask": torch.ones(
                (batch_size, 4), dtype=torch.bool, device=device
            ),
            "subset_masks": torch.stack([row[0] for row in target_rows]).to(device),
        },
        "raw_targets": torch.stack([row[1] for row in target_rows]).to(device),
        "normalized_targets": torch.stack([row[2] for row in target_rows]).to(device),
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
    raw_rows = torch.nn.functional.smooth_l1_loss(
        predictions,
        raw_targets,
        reduction="none",
        beta=float(loss_config["raw_smooth_l1_beta"]),
    ).mean(dim=1)
    normalized_predictions = predictions / scales.unsqueeze(1)
    normalized_rows = torch.nn.functional.smooth_l1_loss(
        normalized_predictions,
        normalized_targets,
        reduction="none",
        beta=float(loss_config["normalized_smooth_l1_beta"]),
    ).mean(dim=1)
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
    untied = pair_mask & (torch.abs(target_differences) > 1e-6)
    prediction_differences = normalized_predictions.unsqueeze(2) - normalized_predictions.unsqueeze(1)
    ranking_terms = torch.nn.functional.softplus(
        -prediction_differences * signs
    )
    ranking_rows = (ranking_terms * untied).sum(dim=(1, 2)) / untied.sum(
        dim=(1, 2)
    ).clamp_min(1)
    weights = trajectory_weights / trajectory_weights.sum()
    raw = torch.sum(raw_rows * weights)
    normalized = torch.sum(normalized_rows * weights)
    ranking = torch.sum(ranking_rows * weights)
    total = (
        float(loss_config["raw_regression"]) * raw
        + float(loss_config["normalized_regression"]) * normalized
        + float(loss_config["within_state_ranking"]) * ranking
    )
    correct = ((prediction_differences * signs) > 0) & untied
    ranking_accuracy = correct.sum() / untied.sum().clamp_min(1)
    return total, {
        "normalized_regression": float(normalized.detach()),
        "ranking_accuracy": float(ranking_accuracy.detach()),
        "raw_mae": float(torch.mean(torch.abs(predictions - raw_targets)).detach()),
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
) -> dict[str, float]:
    totals = defaultdict(float)
    denominator = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            selected = list(states[start : start + batch_size])
            batch = _collate(selected, cache=cache, device=device, torch=torch)
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
        or cache_manifest.get("input_content_sha256")
        != input_manifest.get("content_sha256")
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

    cache = _TokenCache(cache_root, cache_manifest)
    model_config = TokenUtilityModelConfig(**variant["model"])
    model = TokenSetUtilityPredictor(model_config).to(args.device)
    training = config["training"]
    seed = int(variant.get("seed", training["seed"]))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
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
            batch = _collate(selected, cache=cache, device=args.device, torch=torch)
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
        )
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
            break

    summary = {
        "best_checkpoint": best_checkpoint,
        "best_epoch": best_epoch,
        "best_tune_total": best_tune,
        "cache_content_sha256": cache_manifest["content_sha256"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "evaluation_records_loaded": False,
        "history": history,
        "input_content_sha256": input_manifest["content_sha256"],
        "model": variant["model"],
        "overfit_state_count": args.overfit_state_count,
        "optimization_rows_per_epoch": optimization_rows_per_epoch,
        "schema_version": "1.0.0",
        "seed": seed,
        "status": "COMPLETED_SET_UTILITY_TOKEN_PREDICTOR_TRAINING",
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
