#!/usr/bin/env python3
"""Fit-probe singleton restoration ranking on frozen long-oracle train states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from causalcache.set_utility_mvp import canonical_json_bytes
from causalcache.set_utility_token_models import (
    TokenSetUtilityPredictor,
    TokenSingletonMarginalPredictor,
    TokenUtilityModelConfig,
)
from causalcache.set_utility_variable_history import history_bin
from scripts.evaluate_set_utility_long_oracle_insample import _spearman
from scripts.train_set_utility_token_predictor import (
    _TokenCache,
    _batch_stream,
    _cache_covers_input,
    _configure_attention_backend,
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


def _selected_state_ids(path: Path, *, expected_sha256: str) -> tuple[str, ...]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("fit-probe state-id file content drifted")
    values = tuple(line for line in payload.decode("utf-8").splitlines() if line)
    if not values or len(values) != len(set(values)):
        raise ValueError("fit-probe state ids must be non-empty and unique")
    return values


def _singleton_state(state: dict[str, Any]) -> dict[str, Any]:
    candidates = tuple(state["candidate_event_step_ids"])
    by_subset = {
        tuple(row["coalition_event_step_ids"]): row for row in state["distance_rows"]
    }
    required = ((), *((event,) for event in candidates))
    if any(subset not in by_subset for subset in required):
        raise ValueError(f"{state['state_id']} lacks candidate-complete singleton truth")
    result = dict(state)
    result["distance_rows"] = [by_subset[subset] for subset in required]
    return result


def _singleton_loss(
    predictions: Any,
    batch: dict[str, Any],
    *,
    loss_config: dict[str, Any],
    torch: Any,
) -> tuple[Any, dict[str, float]]:
    raw_targets = batch["raw_targets"]
    normalized_targets = batch["normalized_targets"]
    label_mask = batch["label_mask"]
    scales = batch["scales"].unsqueeze(1)
    normalized_predictions = predictions / scales
    cardinalities = batch["model"]["subset_masks"].sum(dim=-1)
    singleton_mask = label_mask & (cardinalities == 1)
    action_count = label_mask.sum(dim=1).clamp_min(1)

    raw_terms = torch.nn.functional.smooth_l1_loss(
        predictions,
        raw_targets,
        reduction="none",
        beta=float(loss_config["raw_smooth_l1_beta"]),
    )
    raw_regression = (
        (raw_terms * singleton_mask).sum(dim=1)
        / singleton_mask.sum(dim=1).clamp_min(1)
    ).mean()
    normalized_terms = torch.nn.functional.smooth_l1_loss(
        normalized_predictions,
        normalized_targets,
        reduction="none",
        beta=float(loss_config["normalized_smooth_l1_beta"]),
    )
    normalized_regression = (
        (normalized_terms * singleton_mask).sum(dim=1)
        / singleton_mask.sum(dim=1).clamp_min(1)
    ).mean()

    target_differences = normalized_targets.unsqueeze(2) - normalized_targets.unsqueeze(1)
    prediction_differences = (
        normalized_predictions.unsqueeze(2) - normalized_predictions.unsqueeze(1)
    )
    pair_mask = torch.triu(
        torch.ones(
            predictions.shape[1],
            predictions.shape[1],
            dtype=torch.bool,
            device=predictions.device,
        ),
        diagonal=1,
    ).unsqueeze(0)
    valid_pairs = pair_mask & singleton_mask.unsqueeze(2) & singleton_mask.unsqueeze(1)
    signs = torch.sign(target_differences)
    untied = valid_pairs & (torch.abs(target_differences) > 1e-6)
    ranking_terms = torch.nn.functional.softplus(-prediction_differences * signs)
    ranking = (
        (ranking_terms * untied).sum(dim=(1, 2))
        / untied.sum(dim=(1, 2)).clamp_min(1)
    ).mean()

    temperature = float(loss_config["decision_temperature"])
    if temperature <= 0.0:
        raise ValueError("decision temperature must be positive")
    student_logits = (normalized_predictions / temperature).masked_fill(
        ~label_mask, -1e9
    )
    teacher_mode = str(loss_config["teacher_mode"])
    if teacher_mode == "soft":
        teacher_probabilities = torch.softmax(
            (normalized_targets / temperature).masked_fill(~label_mask, -1e9),
            dim=1,
        )
        listwise_rows = -(
            teacher_probabilities * torch.log_softmax(student_logits, dim=1)
        ).sum(dim=1)
    elif teacher_mode == "hard":
        best_actions = normalized_targets.masked_fill(~label_mask, -1e9).argmax(dim=1)
        listwise_rows = torch.nn.functional.cross_entropy(
            student_logits, best_actions, reduction="none"
        )
    else:
        raise ValueError(f"unknown teacher mode: {teacher_mode}")
    conditional_listwise = listwise_rows.mean()

    probabilities = torch.softmax(student_logits, dim=1)
    best_teacher = normalized_targets.masked_fill(~label_mask, -1e9).max(
        dim=1, keepdim=True
    ).values
    decision_regret = (
        probabilities
        * (best_teacher - normalized_targets).clamp_min(0.0)
        * label_mask
    ).sum(dim=1).mean()
    signed = singleton_mask & (torch.abs(normalized_targets) > 1e-6)
    signed_terms = torch.nn.functional.softplus(
        -normalized_predictions * torch.sign(normalized_targets)
    )
    sign_classification = (
        (signed_terms * signed).sum(dim=1) / signed.sum(dim=1).clamp_min(1)
    ).mean()

    total = (
        float(loss_config["raw_regression"]) * raw_regression
        + float(loss_config["normalized_regression"]) * normalized_regression
        + float(loss_config["within_state_ranking"]) * ranking
        + float(loss_config["conditional_listwise"]) * conditional_listwise
        + float(loss_config["decision_regret"]) * decision_regret
        + float(loss_config["sign_classification"]) * sign_classification
    )
    ranking_correct = ((prediction_differences * signs) > 0) & untied
    sign_correct = (
        (normalized_predictions > 0) == (normalized_targets > 0)
    ) & signed
    return total, {
        "action_count": float(action_count.float().mean().detach()),
        "conditional_listwise": float(conditional_listwise.detach()),
        "decision_regret": float(decision_regret.detach()),
        "normalized_regression": float(normalized_regression.detach()),
        "ranking": float(ranking.detach()),
        "ranking_accuracy": float(
            ranking_correct.sum().detach() / untied.sum().clamp_min(1)
        ),
        "raw_regression": float(raw_regression.detach()),
        "sign_accuracy": float(sign_correct.sum().detach() / signed.sum().clamp_min(1)),
        "sign_classification": float(sign_classification.detach()),
        "total": float(total.detach()),
    }


def _trajectory_equal(rows: list[dict[str, Any]], key: str) -> float:
    by_trajectory: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_trajectory[row["trajectory_id"]].append(float(row[key]))
    return sum(
        sum(values) / len(values) for values in by_trajectory.values()
    ) / len(by_trajectory)


def _summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {
                "b1_learned_recovery": None,
                "b1_oracle_recovery": None,
                "b1_recent_recovery": None,
                "model_top1_is_true_best_rate": None,
                "singleton_sign_accuracy": None,
                "singleton_spearman_mean": None,
                "state_count": 0,
                "stop_accuracy": None,
                "true_best_in_predicted_top4_rate": None,
                "true_best_outside_recent4_found_in_predicted_top4_rate": None,
            }
        outside = [row for row in selected if row["true_best_outside_recent4"]]
        return {
            "b1_learned_recovery": _trajectory_equal(selected, "learned_recovery"),
            "b1_oracle_recovery": _trajectory_equal(selected, "oracle_recovery"),
            "b1_recent_recovery": _trajectory_equal(selected, "recent_recovery"),
            "model_top1_is_true_best_rate": _trajectory_equal(
                selected, "model_top1_is_true_best"
            ),
            "singleton_sign_accuracy": _trajectory_equal(selected, "sign_accuracy"),
            "singleton_spearman_mean": _trajectory_equal(selected, "spearman"),
            "state_count": len(selected),
            "stop_accuracy": _trajectory_equal(selected, "stop_accuracy"),
            "true_best_in_predicted_top4_rate": _trajectory_equal(
                selected, "true_best_in_predicted_top4"
            ),
            "true_best_outside_recent4_found_in_predicted_top4_rate": (
                _trajectory_equal(outside, "true_best_in_predicted_top4")
                if outside
                else None
            ),
        }

    overall = summarize(rows)
    overall["fit_gate"] = {
        "passed": (
            overall["singleton_spearman_mean"] > 0.5
            and overall["true_best_in_predicted_top4_rate"] > 0.6
        ),
        "singleton_spearman_strictly_greater_than": 0.5,
        "top4_recall_strictly_greater_than": 0.6,
    }
    return {
        "by_bin": {
            name: summarize([row for row in rows if row["history_bin"] == name])
            for name in ("long", "very_long")
        },
        "overall": overall,
    }


def _evaluate_probe(
    model: Any,
    states: tuple[dict[str, Any], ...],
    *,
    cache: _TokenCache,
    batch_size: int,
    device: Any,
    normalization_floor: float,
    torch: Any,
) -> dict[str, Any]:
    rows = []
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(**batch["model"])
            normalized = predictions.float() / batch["scales"].unsqueeze(1)
            for index, state in enumerate(selected):
                candidates = tuple(state["candidate_event_step_ids"])
                count = len(candidates)
                predicted = normalized[index, : count + 1].cpu().tolist()
                targets = batch["normalized_targets"][index, : count + 1].cpu().tolist()
                singleton_predictions = predicted[1:]
                singleton_targets = targets[1:]
                predicted_order = sorted(
                    range(count),
                    key=lambda item: (-singleton_predictions[item], candidates[item]),
                )
                true_best = min(
                    range(count),
                    key=lambda item: (-singleton_targets[item], candidates[item]),
                )
                predicted_action = min(
                    range(count + 1),
                    key=lambda item: (
                        -predicted[item],
                        int(item != 0),
                        candidates[item - 1] if item else -1,
                    ),
                )
                true_action = min(
                    range(count + 1),
                    key=lambda item: (
                        -targets[item],
                        int(item != 0),
                        candidates[item - 1] if item else -1,
                    ),
                )
                recent4 = set(candidates[-4:])
                signed = [
                    (prediction > 0) == (target > 0)
                    for prediction, target in zip(
                        singleton_predictions, singleton_targets, strict=True
                    )
                    if abs(target) > 1e-6
                ]
                rows.append(
                    {
                        "history_bin": history_bin(count),
                        "learned_recovery": targets[predicted_action],
                        "model_top1_is_true_best": predicted_order[0] == true_best,
                        "oracle_recovery": max(targets),
                        "recent_recovery": singleton_targets[-1],
                        "sign_accuracy": sum(signed) / len(signed) if signed else 1.0,
                        "spearman": _spearman(
                            singleton_predictions, singleton_targets
                        ),
                        "state_id": state["state_id"],
                        "stop_accuracy": (predicted_action == 0) == (true_action == 0),
                        "trajectory_id": state["trajectory_id"],
                        "true_best_in_predicted_top4": true_best in predicted_order[:4],
                        "true_best_outside_recent4": candidates[true_best] not in recent4,
                    }
                )
    return _summarize_predictions(rows)


def _resume_snapshot(
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
    parser.add_argument("--state-id-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("long-oracle fit probe requires PyTorch") from error
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise RuntimeError("long-oracle fit probe requires an explicit CUDA device")

    config_path = args.config.resolve()
    config = _read_json(config_path)
    try:
        variant = config["variants"][args.variant]
    except KeyError as error:
        raise ValueError(f"unknown fit-probe variant: {args.variant}") from error
    output_root = args.output_root.resolve()
    if args.resume:
        if not (output_root / "resume.pt").is_file():
            raise FileNotFoundError("fit-probe resume snapshot is missing")
    elif output_root.exists():
        raise FileExistsError(f"fit-probe output already exists: {output_root}")
    else:
        output_root.mkdir(parents=True)
        (output_root / "checkpoints").mkdir()

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
        raise ValueError("fit-probe input/cache identity or split firewall drifted")
    selected_ids = _selected_state_ids(
        args.state_id_file.resolve(),
        expected_sha256=input_contract["state_id_file_sha256"],
    )
    if len(selected_ids) != int(input_contract["state_count"]):
        raise ValueError("fit-probe state count drifted")
    by_id = {
        state["state_id"]: state
        for state in _read_jsonl(input_root / input_manifest["states_jsonl"])
        if state["state_id"] in set(selected_ids)
    }
    if set(by_id) != set(selected_ids):
        raise ValueError("fit-probe selected states are absent from training input")
    states = tuple(_singleton_state(by_id[state_id]) for state_id in selected_ids)
    if any(state["role"] != "train" for state in states):
        raise ValueError("fit-probe may only consume train states")
    if any(history_bin(len(state["candidate_event_step_ids"])) not in {"long", "very_long"} for state in states):
        raise ValueError("fit-probe state history bin drifted")

    training = config["training"]
    attention_backend = _configure_attention_backend(torch, training)
    normalization_floor = _resolve_normalization_floor(training, None)
    seed = int(variant.get("seed", training["seed"]))
    _seed_training_runtime(torch, seed)
    torch.set_float32_matmul_precision("high")
    cache = _TokenCache(cache_root, cache_manifest, device="cpu", mode="lazy_cpu")
    model_config = TokenUtilityModelConfig(**variant["model"])
    if variant["head"] == "scalar_subset":
        model = TokenSetUtilityPredictor(model_config).to(args.device)
    elif variant["head"] == "direct_singleton_marginal":
        model = TokenSingletonMarginalPredictor(model_config).to(args.device)
    else:
        raise ValueError(f"unknown fit-probe head: {variant['head']}")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(variant["learning_rate"]),
        weight_decay=float(variant["weight_decay"]),
    )
    epochs = int(training["epochs"])
    batch_size = int(variant["batch_size"])
    accumulation = int(variant["gradient_accumulation_steps"])
    rows_per_epoch = len(_trajectory_uniform_epoch(states, seed=seed))
    optimizer_steps = math.ceil(math.ceil(rows_per_epoch / batch_size) / accumulation)
    total_steps = optimizer_steps * epochs
    warmup_steps = max(1, round(total_steps * float(training["warmup_ratio"])))
    minimum_lr_ratio = float(training["minimum_lr_ratio"])

    def learning_rate_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    identity = {
        "cache_content_sha256": cache_manifest["content_sha256"],
        "config_sha256": config_sha256,
        "input_content_sha256": input_manifest["content_sha256"],
        "state_id_file_sha256": input_contract["state_id_file_sha256"],
        "variant": args.variant,
    }
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if args.resume:
        snapshot = torch.load(
            output_root / "resume.pt", map_location=args.device, weights_only=False
        )
        if snapshot["identity"] != identity:
            raise ValueError("fit-probe resume identity drifted")
        model.load_state_dict(snapshot["model"])
        optimizer.load_state_dict(snapshot["optimizer"])
        scheduler.load_state_dict(snapshot["scheduler"])
        random.setstate(snapshot["python_rng_state"])
        torch.set_rng_state(snapshot["torch_rng_state"].cpu())
        torch.cuda.set_rng_state(snapshot["cuda_rng_state"].cpu())
        history = snapshot["history"]
        start_epoch = int(snapshot["epoch"]) + 1

    started = time.time()
    gate_streak = 0
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        order = _trajectory_uniform_epoch(states, seed=seed + epoch)
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = defaultdict(float)
        denominator = 0.0
        pending = 0
        for batch_index, (selected, batch) in enumerate(
            _batch_stream(
                order,
                batch_size=batch_size,
                cache=cache,
                device=args.device,
                torch=torch,
                normalization_floor=normalization_floor,
            )
        ):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(**batch["model"])
                loss, metrics = _singleton_loss(
                    predictions,
                    batch,
                    loss_config=variant["loss"],
                    torch=torch,
                )
            (loss / accumulation).backward()
            pending += 1
            final_batch = (batch_index + 1) * batch_size >= len(order)
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
        train_metrics = {
            key: value / denominator for key, value in sorted(totals.items())
        }
        probe = _evaluate_probe(
            model,
            states,
            cache=cache,
            batch_size=int(variant["evaluation_batch_size"]),
            device=args.device,
            normalization_floor=normalization_floor,
            torch=torch,
        )
        checkpoint = _save_checkpoint(
            model, output_root / "checkpoints" / f"epoch-{epoch:03d}.safetensors"
        )
        history.append(
            {
                "checkpoint": checkpoint,
                "epoch": epoch,
                "learning_rate": scheduler.get_last_lr()[0],
                "probe": probe,
                "train": train_metrics,
            }
        )
        progress = {
            "completed_epoch": epoch,
            "history": history,
            "identity": identity,
            "status": "RUNNING_SET_UTILITY_LONG_ORACLE_FIT_PROBE",
        }
        progress["content_sha256"] = hashlib.sha256(
            canonical_json_bytes(progress)
        ).hexdigest()
        _atomic_json(output_root / "progress.json", progress)
        _resume_snapshot(
            output_root / "resume.pt",
            epoch=epoch,
            history=history,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            identity=identity,
            torch=torch,
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if probe["overall"]["fit_gate"]["passed"]:
            gate_streak += 1
        else:
            gate_streak = 0
        if (
            epoch >= int(training.get("minimum_epochs", epochs))
            and gate_streak >= int(training.get("gate_patience", epochs + 1))
        ):
            break

    best = max(
        history,
        key=lambda row: (
            row["probe"]["overall"]["fit_gate"]["passed"],
            row["probe"]["overall"]["singleton_spearman_mean"],
            row["probe"]["overall"]["true_best_in_predicted_top4_rate"],
            row["probe"]["overall"]["b1_learned_recovery"],
            -row["epoch"],
        ),
    )
    summary = {
        "attention_backend": attention_backend,
        "best_epoch": best["epoch"],
        "best_probe": best["probe"],
        "cache_content_sha256": cache_manifest["content_sha256"],
        "config_sha256": config_sha256,
        "elapsed_seconds_this_invocation": time.time() - started,
        "evaluation_records_loaded": False,
        "head": variant["head"],
        "history": history,
        "input_content_sha256": input_manifest["content_sha256"],
        "model": variant["model"],
        "schema_version": "causalcache.set_utility_long_oracle_fit_probe.v1",
        "seed": seed,
        "state_count": len(states),
        "state_id_file_sha256": input_contract["state_id_file_sha256"],
        "status": "COMPLETED_SET_UTILITY_LONG_ORACLE_FIT_PROBE",
        "trajectory_count": len({state["trajectory_id"] for state in states}),
        "variant": args.variant,
    }
    summary["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(summary)
    ).hexdigest()
    _atomic_json(output_root / "summary.json", summary)


if __name__ == "__main__":
    main()
