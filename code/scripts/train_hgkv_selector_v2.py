#!/usr/bin/env python3
"""Train the fresh unified full-history HGKV selector V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from causalcache.hgkv_selector_v2 import (
    HGKVSetSelectorV2,
    V2_FEATURE_DIM,
    hgkv_selector_v2_loss,
)
from scripts.build_hgkv_selector_v2_training_data import (
    expand_paths,
    sha256_file,
)


def load_feature_tensor(
    paths: Iterable[Path],
) -> tuple[torch.Tensor, list[tuple[str, int]], str]:
    features: dict[tuple[str, int], list[float]] = {}
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row["pair_group"]),
                int(row["singleton_event_step_id"]),
            )
            if key in features:
                raise ValueError(f"{path}:{line_no} duplicate feature {key}")
            values = [float(value) for value in row["feature"]]
            if len(values) != V2_FEATURE_DIM or not all(
                math.isfinite(value) for value in values
            ):
                raise ValueError(f"{path}:{line_no} invalid V2 feature")
            features[key] = values
    keys = sorted(features)
    packed = "".join(
        f"{pair_group}\t{candidate}\n" for pair_group, candidate in keys
    )
    return (
        torch.tensor([features[key] for key in keys], dtype=torch.float32),
        keys,
        hashlib.sha256(packed.encode("utf-8")).hexdigest(),
    )


def load_groups(
    paths: Iterable[Path],
    *,
    feature_keys: Sequence[tuple[str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_group = str(row["pair_group"])
            for candidate, index in zip(
                row["candidate_event_step_ids"],
                row["candidate_feature_indices"],
                strict=True,
            ):
                if feature_keys[int(index)] != (pair_group, int(candidate)):
                    raise ValueError(
                        f"{path}:{line_no} candidate feature index drifted"
                    )
            for candidate, index in zip(
                row["selected_event_step_ids"],
                row["selected_feature_indices"],
                strict=True,
            ):
                if feature_keys[int(index)] != (pair_group, int(candidate)):
                    raise ValueError(
                        f"{path}:{line_no} selected feature index drifted"
                    )
            rows.append(row)
    return rows


def collate_groups(
    rows: Sequence[Mapping[str, Any]],
    features: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collate an empty prefix batch")
    candidate_width = max(len(row["candidate_feature_indices"]) for row in rows)
    selected_width = max(len(row["selected_feature_indices"]) for row in rows)
    if selected_width > 3:
        raise ValueError("V2 selected width must not exceed three during training")
    batch_size = len(rows)
    candidate_features = torch.zeros(
        (batch_size, candidate_width, V2_FEATURE_DIM),
        dtype=torch.float32,
    )
    selected_features = torch.zeros(
        (batch_size, selected_width, V2_FEATURE_DIM),
        dtype=torch.float32,
    )
    candidate_mask = torch.zeros(
        (batch_size, candidate_width), dtype=torch.bool
    )
    selected_mask = torch.zeros(
        (batch_size, selected_width), dtype=torch.bool
    )
    targets = torch.zeros(
        (batch_size, candidate_width), dtype=torch.float32
    )
    for index, row in enumerate(rows):
        candidate_indices = torch.tensor(
            row["candidate_feature_indices"], dtype=torch.long
        )
        selected_indices = torch.tensor(
            row["selected_feature_indices"], dtype=torch.long
        )
        candidate_count = int(candidate_indices.numel())
        selected_count = int(selected_indices.numel())
        candidate_features[index, :candidate_count] = features[
            candidate_indices
        ]
        candidate_mask[index, :candidate_count] = True
        targets[index, :candidate_count] = torch.tensor(
            row["marginal_targets"], dtype=torch.float32
        )
        if selected_count:
            selected_features[index, :selected_count] = features[
                selected_indices
            ]
            selected_mask[index, :selected_count] = True
    return {
        "candidate_features": candidate_features.to(device),
        "candidate_mask": candidate_mask.to(device),
        "selected_features": selected_features.to(device),
        "selected_mask": selected_mask.to(device),
        "targets": targets.to(device),
        "remaining_budget": torch.tensor(
            [int(row["remaining_budget"]) for row in rows],
            dtype=torch.long,
            device=device,
        ),
        "rows": rows,
    }


def trajectory_group_folds(
    rows: Sequence[Mapping[str, Any]], *, folds: int
) -> list[set[str]]:
    if folds < 2:
        raise ValueError("GroupKFold requires at least two folds")
    counts = Counter(str(row["episode"]) for row in rows)
    if len(counts) < folds:
        raise ValueError("fewer trajectory groups than folds")
    assignments: list[set[str]] = [set() for _ in range(folds)]
    loads = [0] * folds
    for episode, count in sorted(
        counts.items(), key=lambda item: (-item[1], item[0])
    ):
        target = min(range(folds), key=lambda index: (loads[index], index))
        assignments[target].add(episode)
        loads[target] += count
    return assignments


def _batches(
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> Iterable[list[dict[str, Any]]]:
    indices = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [rows[index] for index in indices[start : start + batch_size]]


def make_model(config: Mapping[str, Any]) -> HGKVSetSelectorV2:
    features = config["features"]
    candidate = config["student"]["candidate_encoder"]
    conditioning = config["student"]["set_conditioning"]
    return HGKVSetSelectorV2(
        layer_count=int(features["hgkv_layer_count"]),
        layer_feature_dim=int(features["hgkv_layer_feature_dim"]),
        hidden_dim=int(candidate["hidden_dim"]),
        temporal_hidden_dim=int(candidate["temporal_hidden_dim"]),
        head_hidden_dim=int(candidate["head_hidden_dim"]),
        attention_heads=int(conditioning["attention_heads"]),
        max_budget=int(config["teacher"]["max_budget"]),
    )


def _loss_kwargs(config: Mapping[str, Any]) -> dict[str, float]:
    training = config["training"]
    weights = training["loss_weights"]
    return {
        "marginal_weight": float(weights["marginal"]),
        "rank_weight": float(weights["rank"]),
        "positive_weight": float(weights["positive"]),
        "stop_weight": float(weights["stop"]),
        "huber_beta": float(training["huber_beta"]),
        "rank_min_delta": float(training["rank_min_delta"]),
    }


def train_epoch(
    model: HGKVSetSelectorV2,
    optimizer: torch.optim.Optimizer,
    rows: list[dict[str, Any]],
    features: torch.Tensor,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    epoch_seed: int,
) -> float:
    model.train()
    total = 0.0
    groups = 0
    batch_size = int(config["training"]["batch_groups"])
    for batch_rows in _batches(
        rows,
        batch_size=batch_size,
        seed=epoch_seed,
        shuffle=True,
    ):
        batch = collate_groups(batch_rows, features, device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["candidate_features"],
            batch["candidate_mask"],
            batch["selected_features"],
            batch["selected_mask"],
            batch["remaining_budget"],
        )
        loss, _parts = hgkv_selector_v2_loss(
            outputs,
            batch["targets"],
            batch["candidate_mask"],
            **_loss_kwargs(config),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(config["training"]["gradient_clip_norm"]),
        )
        optimizer.step()
        total += float(loss.detach()) * len(batch_rows)
        groups += len(batch_rows)
    return total / max(groups, 1)


def binary_metrics(
    targets: Sequence[bool], scores: Sequence[float]
) -> dict[str, float]:
    if len(targets) != len(scores) or not targets:
        raise ValueError("binary diagnostics require aligned non-empty values")
    predictions = [score >= 0.0 for score in scores]
    tp = sum(prediction and target for prediction, target in zip(predictions, targets))
    tn = sum(
        not prediction and not target
        for prediction, target in zip(predictions, targets)
    )
    fp = sum(
        prediction and not target
        for prediction, target in zip(predictions, targets)
    )
    fn = sum(
        not prediction and target
        for prediction, target in zip(predictions, targets)
    )
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    positives = [score for score, target in zip(scores, targets) if target]
    negatives = [score for score, target in zip(scores, targets) if not target]
    if positives and negatives:
        wins = sum(
            (positive > negative) + 0.5 * (positive == negative)
            for positive in positives
            for negative in negatives
        )
        auroc = wins / (len(positives) * len(negatives))
    else:
        auroc = float("nan")
    return {
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "predicted_stop_rate": sum(predictions) / len(predictions),
        "true_stop_rate": sum(targets) / len(targets),
    }


@torch.no_grad()
def evaluate(
    model: HGKVSetSelectorV2,
    rows: list[dict[str, Any]],
    features: torch.Tensor,
    *,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses = 0.0
    groups = 0
    realized: list[float] = []
    stop_targets: list[bool] = []
    stop_scores: list[float] = []
    by_depth: dict[int, dict[str, list[Any]]] = {
        depth: {"targets": [], "scores": [], "realized": []}
        for depth in range(4)
    }
    batch_size = int(config["training"]["batch_groups"])
    for batch_rows in _batches(
        rows,
        batch_size=batch_size,
        seed=0,
        shuffle=False,
    ):
        batch = collate_groups(batch_rows, features, device=device)
        outputs = model(
            batch["candidate_features"],
            batch["candidate_mask"],
            batch["selected_features"],
            batch["selected_mask"],
            batch["remaining_budget"],
        )
        loss, _parts = hgkv_selector_v2_loss(
            outputs,
            batch["targets"],
            batch["candidate_mask"],
            **_loss_kwargs(config),
        )
        losses += float(loss) * len(batch_rows)
        groups += len(batch_rows)
        for index, row in enumerate(batch_rows):
            width = len(row["marginal_targets"])
            predicted = outputs["marginal"][index, :width]
            best_value, best_index = predicted.max(dim=0)
            true_increment = (
                0.0
                if float(best_value) <= 0.0
                else float(row["marginal_targets"][int(best_index)])
            )
            realized_u = float(row["prefix_u_act"]) + true_increment
            stop_target = bool(row["stop_is_optimal"])
            stop_score = float(outputs["stop_logit"][index])
            depth = int(row["depth"])
            realized.append(realized_u)
            stop_targets.append(stop_target)
            stop_scores.append(stop_score)
            by_depth[depth]["targets"].append(stop_target)
            by_depth[depth]["scores"].append(stop_score)
            by_depth[depth]["realized"].append(realized_u)
    return {
        "loss": losses / max(groups, 1),
        "teacher_forced_learned_beam4_realized_u": statistics.fmean(realized),
        "stop": binary_metrics(stop_targets, stop_scores),
        "by_depth": {
            str(depth): {
                **binary_metrics(values["targets"], values["scores"]),
                "teacher_forced_realized_u": statistics.fmean(
                    values["realized"]
                ),
                "groups": len(values["targets"]),
            }
            for depth, values in by_depth.items()
            if values["targets"]
        },
        "groups": groups,
    }


def fit(
    *,
    rows: list[dict[str, Any]],
    features: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    epochs: int,
    validation_rows: list[dict[str, Any]] | None,
    patience: int | None,
) -> tuple[HGKVSetSelectorV2, int, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_epoch = epochs
    best_metric = -float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            optimizer,
            rows,
            features,
            config=config,
            device=device,
            epoch_seed=seed + epoch,
        )
        record: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
        if validation_rows is not None:
            validation = evaluate(
                model,
                validation_rows,
                features,
                config=config,
                device=device,
            )
            metric = float(
                validation["teacher_forced_learned_beam4_realized_u"]
            )
            record["validation"] = validation
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if validation_rows is not None and patience is not None and stale >= patience:
            break
    return model, best_epoch, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features", action="append", required=True)
    parser.add_argument("--training-data", action="append", required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    feature_paths = expand_paths(args.features)
    data_paths = expand_paths(args.training_data)
    if not feature_paths or not data_paths:
        parser.error("feature and training-data patterns must match files")
    features, feature_keys, feature_index_sha256 = load_feature_tensor(
        feature_paths
    )
    training_manifest = json.loads(
        args.training_manifest.read_text(encoding="utf-8")
    )
    if training_manifest["feature_index_sha256"] != feature_index_sha256:
        raise ValueError("training-data feature index is not reproducible")
    groups = load_groups(data_paths, feature_keys=feature_keys)
    train_rows = [row for row in groups if row["split"] == "train"]
    dev_rows = [row for row in groups if row["split"] == "dev"]
    if not train_rows or not dev_rows:
        raise ValueError("V2 training requires non-empty train and dev groups")
    device = torch.device(args.device)
    seed = int(config["seed"])
    folds = int(config["training"]["folds"])
    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["patience"])
    fold_groups = trajectory_group_folds(train_rows, folds=folds)
    fold_results: list[dict[str, Any]] = []
    selected_epochs: list[int] = []
    for fold_index, heldout_episodes in enumerate(fold_groups):
        fold_train = [
            row
            for row in train_rows
            if str(row["episode"]) not in heldout_episodes
        ]
        fold_validation = [
            row
            for row in train_rows
            if str(row["episode"]) in heldout_episodes
        ]
        _model, best_epoch, history = fit(
            rows=fold_train,
            features=features,
            config=config,
            device=device,
            seed=seed + fold_index,
            epochs=max_epochs,
            validation_rows=fold_validation,
            patience=patience,
        )
        selected_epochs.append(best_epoch)
        fold_results.append(
            {
                "fold": fold_index,
                "heldout_trajectories": len(heldout_episodes),
                "train_groups": len(fold_train),
                "validation_groups": len(fold_validation),
                "best_epoch": best_epoch,
                "history": history,
            }
        )
        del _model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    refit_epochs = int(statistics.median(selected_epochs))
    model, _unused_epoch, refit_history = fit(
        rows=train_rows,
        features=features,
        config=config,
        device=device,
        seed=seed + folds,
        epochs=refit_epochs,
        validation_rows=None,
        patience=None,
    )
    development = evaluate(
        model,
        dev_rows,
        features,
        config=config,
        device=device,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_root / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "feature_index_sha256": feature_index_sha256,
            "source_commit": args.source_commit,
            "refit_epochs": refit_epochs,
        },
        checkpoint,
    )
    result = {
        "schema_version": "causalcache.hgkv_selector_v2.training_result.v1",
        "source_commit": args.source_commit,
        "selection": (
            "train_only_trajectory_groupkfold_then_median_epoch_full_refit"
        ),
        "checkpoint_metric": (
            "teacher_forced_learned_beam4_realized_u"
        ),
        "selected_fold_epochs": selected_epochs,
        "refit_epochs": refit_epochs,
        "fold_results": fold_results,
        "refit_history": refit_history,
        "development_read_count": 1,
        "development": development,
        "train_groups": len(train_rows),
        "dev_groups": len(dev_rows),
        "feature_rows": len(feature_keys),
        "feature_index_sha256": feature_index_sha256,
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(args.config),
        "training_data_sha256": {
            str(path): sha256_file(path) for path in data_paths
        },
        "feature_sha256": {
            str(path): sha256_file(path) for path in feature_paths
        },
        "status": "DONE",
    }
    (args.output_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "DONE").write_text(
        json.dumps(
            {
                "status": "DONE",
                "checkpoint_sha256": result["checkpoint_sha256"],
                "result_sha256": sha256_file(args.output_root / "result.json"),
                "source_commit": args.source_commit,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(development, sort_keys=True))


if __name__ == "__main__":
    main()
