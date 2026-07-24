#!/usr/bin/env python3
"""Train the frozen Stage-2 HGKV set-conditioned marginal selector."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from causalcache.hgkv_selector import (
    HGKVSetConditionedSelector,
    HGKVSingletonSelector,
    model_config_from_dict,
    set_conditioned_multitask_loss,
    set_model_config_from_dict,
)
from scripts.train_hgkv_singleton_selector_v1 import (
    atomic_write_json,
    bootstrap_ci,
    load_features,
    load_scores,
    rankdata,
    sha256_file,
)


ConditionalScoreKey = tuple[str, str, str]
GroupKey = tuple[str, tuple[int, ...], int]


def load_conditional_scores(paths: list[Path]) -> dict[ConditionalScoreKey, float]:
    scores: dict[ConditionalScoreKey, float] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (
                    str(row["pair_group"]),
                    str(row["variant"]),
                    str(row["restored_set_key"]),
                )
                value = float(row["target_logprob_mean"])
                if key in scores:
                    raise ValueError(f"duplicate conditional score key: {key}")
                if not np.isfinite(value):
                    raise ValueError(f"non-finite conditional score: {key}")
                scores[key] = value
    return scores


def set_key(event_ids: Iterable[int]) -> str:
    return "-".join(str(value) for value in sorted(event_ids))


def build_conditional_groups(
    render_paths: list[Path],
    *,
    conditional_scores: dict[ConditionalScoreKey, float],
    singleton_scores: dict[tuple[str, int], float],
    budget_replication: dict[str, list[int]],
) -> tuple[dict[GroupKey, dict[int, float]], dict[str, Any]]:
    groups: dict[GroupKey, dict[int, float]] = defaultdict(dict)
    base_parity = []
    edge_counts: dict[str, int] = defaultdict(int)
    rendered_rows = 0
    for path in render_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rendered_rows += 1
                pair_group = str(row["pair_group"])
                variant = str(row["variant"])
                anchor = tuple(
                    int(value)
                    for value in (row.get("conditional_anchor_set") or [])
                )
                restored = tuple(
                    int(value)
                    for value in row["memory_config"][
                        "restored_event_step_ids"
                    ]
                )
                child_key = (
                    pair_group,
                    variant,
                    str(row["restored_set_key"]),
                )
                if child_key not in conditional_scores:
                    raise ValueError(f"conditional score join miss: {child_key}")
                child_score = conditional_scores[child_key]
                if variant == "cond_base":
                    if len(anchor) != 1:
                        raise ValueError("cond_base must have one anchor")
                    singleton_key = (pair_group, anchor[0])
                    if singleton_key not in singleton_scores:
                        raise ValueError(
                            f"singleton score join miss: {singleton_key}"
                        )
                    base_parity.append(
                        child_score - singleton_scores[singleton_key]
                    )
                    continue
                if variant not in {"cond_edge1", "cond_edge2"}:
                    raise ValueError(f"unknown conditional variant: {variant}")
                candidate = row.get("conditional_candidate")
                if candidate is None:
                    raise ValueError(f"{variant} lacks conditional_candidate")
                candidate = int(candidate)
                if candidate in anchor:
                    raise ValueError("conditional candidate already selected")
                if set(restored) != {*anchor, candidate}:
                    raise ValueError("restored set does not equal anchor plus candidate")
                if variant == "cond_edge1":
                    if len(anchor) != 1:
                        raise ValueError("cond_edge1 must have one anchor")
                    base_key = (pair_group, anchor[0])
                    if base_key not in singleton_scores:
                        raise ValueError(
                            f"singleton score join miss: {base_key}"
                        )
                    base_score = singleton_scores[base_key]
                else:
                    if len(anchor) != 2:
                        raise ValueError("cond_edge2 must have two anchors")
                    base_key = (
                        pair_group,
                        "cond_edge1",
                        set_key(anchor),
                    )
                    if base_key not in conditional_scores:
                        raise ValueError(
                            f"conditional base score join miss: {base_key}"
                        )
                    base_score = conditional_scores[base_key]
                marginal = child_score - base_score
                edge_counts[variant] += 1
                for final_budget in budget_replication[variant]:
                    if final_budget <= len(anchor):
                        raise ValueError("final budget must exceed anchor size")
                    group_key = (pair_group, tuple(sorted(anchor)), final_budget)
                    previous = groups[group_key].get(candidate)
                    if previous is not None and not np.isclose(
                        previous, marginal, atol=1e-8, rtol=0
                    ):
                        raise ValueError(
                            f"conflicting marginal for {group_key} candidate {candidate}"
                        )
                    groups[group_key][candidate] = marginal
    parity_array = np.asarray(base_parity, dtype=np.float64)
    audit = {
        "cond_base_count": len(base_parity),
        "cond_base_max_abs_singleton_delta": (
            float(np.max(np.abs(parity_array))) if len(parity_array) else None
        ),
        "edge_counts_before_group_dedup": dict(sorted(edge_counts.items())),
        "group_count": len(groups),
        "rendered_rows": rendered_rows,
    }
    return groups, audit


def build_group_arrays(
    groups: dict[GroupKey, dict[int, float]],
    *,
    feature_indices: dict[tuple[str, int], int],
    episode_by_state: dict[str, str],
    max_candidates: int = 8,
    max_selected: int = 3,
) -> dict[str, Any]:
    group_keys = sorted(groups)
    candidate_indices = np.full(
        (len(group_keys), max_candidates), -1, dtype=np.int64
    )
    selected_indices = np.full(
        (len(group_keys), max_selected), -1, dtype=np.int64
    )
    candidate_event_ids = np.full_like(candidate_indices, -1)
    targets = np.zeros((len(group_keys), max_candidates), dtype=np.float32)
    candidate_mask = np.zeros_like(targets, dtype=bool)
    selected_mask = np.zeros(
        (len(group_keys), max_selected), dtype=bool
    )
    remaining_budget = np.zeros(len(group_keys), dtype=np.int64)
    episodes = []
    for group_index, group_key in enumerate(group_keys):
        pair_group, anchor, final_budget = group_key
        candidates = sorted(groups[group_key], reverse=True)
        if not candidates or len(candidates) > max_candidates:
            raise ValueError(
                f"{group_key} has {len(candidates)} candidates, "
                f"expected 1..{max_candidates}"
            )
        if not anchor or len(anchor) > max_selected:
            raise ValueError(f"{group_key} has unsupported anchor size")
        if pair_group not in episode_by_state:
            raise ValueError(f"episode join miss for {pair_group}")
        episodes.append(episode_by_state[pair_group])
        remaining_budget[group_index] = final_budget - len(anchor)
        for selected_position, event_id in enumerate(anchor):
            feature_key = (pair_group, event_id)
            if feature_key not in feature_indices:
                raise ValueError(f"selected feature join miss: {feature_key}")
            selected_indices[group_index, selected_position] = feature_indices[
                feature_key
            ]
            selected_mask[group_index, selected_position] = True
        for candidate_position, event_id in enumerate(candidates):
            feature_key = (pair_group, event_id)
            if feature_key not in feature_indices:
                raise ValueError(f"candidate feature join miss: {feature_key}")
            candidate_indices[group_index, candidate_position] = feature_indices[
                feature_key
            ]
            candidate_event_ids[group_index, candidate_position] = event_id
            targets[group_index, candidate_position] = groups[group_key][event_id]
            candidate_mask[group_index, candidate_position] = True
    return {
        "candidate_event_ids": candidate_event_ids,
        "candidate_indices": candidate_indices,
        "candidate_mask": candidate_mask,
        "episodes": np.asarray(episodes),
        "group_keys": group_keys,
        "remaining_budget": remaining_budget,
        "selected_indices": selected_indices,
        "selected_mask": selected_mask,
        "targets": targets,
    }


def batches(
    indices: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator | None,
) -> Iterable[np.ndarray]:
    ordered = indices.copy()
    if rng is not None:
        rng.shuffle(ordered)
    for start in range(0, len(ordered), batch_size):
        yield ordered[start : start + batch_size]


def tensor_batch(
    arrays: dict[str, Any],
    feature_table: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_indices = arrays["candidate_indices"][indices]
    selected_indices = arrays["selected_indices"][indices]
    candidate_mask = arrays["candidate_mask"][indices]
    selected_mask = arrays["selected_mask"][indices]
    candidate_features = feature_table[np.maximum(candidate_indices, 0)].copy()
    selected_features = feature_table[np.maximum(selected_indices, 0)].copy()
    candidate_features[~candidate_mask] = 0
    selected_features[~selected_mask] = 0
    return (
        torch.from_numpy(candidate_features).to(device),
        torch.from_numpy(candidate_mask).to(device),
        torch.from_numpy(selected_features).to(device),
        torch.from_numpy(selected_mask).to(device),
        torch.from_numpy(arrays["remaining_budget"][indices]).to(device),
    )


def initialize_model(
    config: dict[str, Any],
    stage1_checkpoint: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
) -> HGKVSetConditionedSelector:
    torch.manual_seed(seed)
    singleton = HGKVSingletonSelector(
        **model_config_from_dict(stage1_checkpoint["config"])
    )
    singleton.load_state_dict(stage1_checkpoint["model_state_dict"])
    model = HGKVSetConditionedSelector(
        **set_model_config_from_dict(config)
    )
    model.encoder.load_state_dict(singleton.encoder.state_dict())
    return model.to(device)


def train_epoch(
    model: HGKVSetConditionedSelector,
    optimizer: torch.optim.Optimizer,
    arrays: dict[str, Any],
    feature_table: np.ndarray,
    indices: np.ndarray,
    *,
    config: dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    training = config["training"]
    weights = training["loss_weights"]
    totals: dict[str, list[float]] = defaultdict(list)
    for selected in batches(
        indices, batch_size=int(training["batch_groups"]), rng=rng
    ):
        (
            candidate_features,
            candidate_mask,
            selected_features,
            selected_mask,
            remaining_budget,
        ) = tensor_batch(
            arrays, feature_table, selected, device=device
        )
        targets = torch.from_numpy(arrays["targets"][selected]).to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            candidate_features,
            candidate_mask,
            selected_features,
            selected_mask,
            remaining_budget,
        )
        loss, parts = set_conditioned_multitask_loss(
            outputs,
            targets,
            candidate_mask,
            marginal_weight=float(weights["marginal"]),
            rank_weight=float(weights["rank"]),
            positive_weight=float(weights["positive"]),
            stop_weight=float(weights["stop"]),
            huber_beta=float(training["huber_beta"]),
            rank_min_delta=float(training["rank_min_delta"]),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        for key, value in parts.items():
            totals[key].append(float(value))
    return {key: float(np.mean(values)) for key, values in totals.items()}


@torch.no_grad()
def predict(
    model: HGKVSetConditionedSelector,
    arrays: dict[str, Any],
    feature_table: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    width = arrays["targets"].shape[1]
    marginal = np.zeros((len(indices), width), dtype=np.float32)
    positive = np.zeros_like(marginal)
    rank = np.zeros_like(marginal)
    stop = np.zeros(len(indices), dtype=np.float32)
    offset = 0
    for selected in batches(indices, batch_size=batch_size, rng=None):
        tensors = tensor_batch(
            arrays, feature_table, selected, device=device
        )
        outputs = model(*tensors)
        size = len(selected)
        marginal[offset : offset + size] = (
            outputs["marginal"].cpu().numpy()
        )
        positive[offset : offset + size] = (
            outputs["positive_logit"].sigmoid().cpu().numpy()
        )
        rank[offset : offset + size] = outputs["rank_score"].cpu().numpy()
        stop[offset : offset + size] = (
            outputs["stop_logit"].cpu().numpy()
        )
        offset += size
    return {
        "marginal": marginal,
        "positive": positive,
        "rank": rank,
        "stop": stop,
    }


def group_metrics(
    arrays: dict[str, Any],
    predictions: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rhos = []
    hit = []
    regret = []
    selected = []
    recent = []
    random_values = []
    oracle = []
    at_most = []
    stop_correct = []
    for group_index in range(len(arrays["group_keys"])):
        mask = arrays["candidate_mask"][group_index]
        truth = arrays["targets"][group_index][mask]
        marginal = predictions["marginal"][group_index][mask]
        rank_score = predictions["rank"][group_index][mask]
        true_ranks = rankdata(truth)
        pred_ranks = rankdata(rank_score)
        if np.std(true_ranks) > 0 and np.std(pred_ranks) > 0:
            rhos.append(float(np.corrcoef(true_ranks, pred_ranks)[0, 1]))
        chosen = int(np.argmax(rank_score))
        best = float(np.max(truth))
        realized = float(truth[chosen])
        hit.append(float(realized == best))
        regret.append(best - realized)
        selected.append(realized)
        recent.append(float(truth[0]))
        random_values.append(float(np.mean(truth)))
        oracle.append(best)
        predicted_stop = bool(
            predictions["stop"][group_index] >= float(rank_score[chosen])
        )
        at_most.append(0.0 if predicted_stop else realized)
        stop_correct.append(float(predicted_stop == (best <= 0)))
    values = {
        "at_most_selected": np.asarray(at_most),
        "hit": np.asarray(hit),
        "oracle": np.asarray(oracle),
        "random": np.asarray(random_values),
        "recent": np.asarray(recent),
        "regret": np.asarray(regret),
        "selected": np.asarray(selected),
    }
    metrics = {
        "at_most_realized_marginal_mean": float(
            np.mean(values["at_most_selected"])
        ),
        "mean_regret": float(np.mean(values["regret"])),
        "mean_spearman": float(np.mean(rhos)) if rhos else None,
        "median_spearman": float(np.median(rhos)) if rhos else None,
        "n_groups": len(arrays["group_keys"]),
        "n_valid_spearman": len(rhos),
        "stop_accuracy": float(np.mean(stop_correct)),
        "top1_hit_rate": float(np.mean(values["hit"])),
    }
    return metrics, values


def subset_arrays(
    arrays: dict[str, Any], indices: np.ndarray
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    group_count = len(arrays["group_keys"])
    for key, value in arrays.items():
        if isinstance(value, np.ndarray) and len(value) == group_count:
            output[key] = value[indices]
        elif key == "group_keys":
            output[key] = [value[index] for index in indices]
        else:
            output[key] = value
    return output


def fit_fold(
    arrays: dict[str, Any],
    feature_table: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    config: dict[str, Any],
    stage1_checkpoint: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, Any]]]:
    model = initialize_model(
        config, stage1_checkpoint, device=device, seed=seed
    )
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_key: tuple[float, float] | None = None
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    rng = np.random.default_rng(seed)
    validation = subset_arrays(arrays, validation_indices)
    for epoch in range(1, int(training["max_epochs"]) + 1):
        losses = train_epoch(
            model,
            optimizer,
            arrays,
            feature_table,
            train_indices,
            config=config,
            device=device,
            rng=rng,
        )
        prediction = predict(
            model,
            arrays,
            feature_table,
            validation_indices,
            batch_size=int(training["batch_groups"]),
            device=device,
        )
        metrics, _ = group_metrics(validation, prediction)
        key = (
            float(metrics["mean_spearman"] or -math.inf),
            -float(metrics["mean_regret"]),
        )
        history.append(
            {"epoch": epoch, "loss": losses, "validation": metrics}
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= int(training["patience"]):
            break
    if best_state is None:
        raise RuntimeError("fold produced no checkpoint")
    return best_state, best_epoch, history


def fit_full(
    arrays: dict[str, Any],
    feature_table: np.ndarray,
    *,
    epochs: int,
    config: dict[str, Any],
    stage1_checkpoint: dict[str, Any],
    device: torch.device,
    seed: int,
) -> HGKVSetConditionedSelector:
    model = initialize_model(
        config, stage1_checkpoint, device=device, seed=seed
    )
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    indices = np.arange(len(arrays["group_keys"]))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        train_epoch(
            model,
            optimizer,
            arrays,
            feature_table,
            indices,
            config=config,
            device=device,
            rng=rng,
        )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--singleton-scores-dir", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--train-render-dir", type=Path, required=True)
    parser.add_argument("--heldout-render-dir", type=Path, required=True)
    parser.add_argument(
        "--train-conditional-scores", type=Path, nargs="+", required=True
    )
    parser.add_argument(
        "--heldout-conditional-scores", type=Path, nargs="+", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_table, feature_keys, feature_indices = load_features(args.features)
    singleton_train_rows = load_scores(
        args.singleton_scores_dir / "singleton-train.jsonl"
    )
    singleton_heldout_rows = load_scores(
        args.singleton_scores_dir / "singleton-heldout.jsonl"
    )
    singleton_train = {
        row["key"]: row["target_logprob_mean"] for row in singleton_train_rows
    }
    singleton_heldout = {
        row["key"]: row["target_logprob_mean"]
        for row in singleton_heldout_rows
    }
    episode_train = {
        row["pair_group"]: row["episode"] for row in singleton_train_rows
    }
    episode_heldout = {
        row["pair_group"]: row["episode"] for row in singleton_heldout_rows
    }
    if set(feature_keys) != set(singleton_train) | set(singleton_heldout):
        raise ValueError("feature/singleton score key mismatch")

    train_scores = load_conditional_scores(args.train_conditional_scores)
    heldout_scores = load_conditional_scores(args.heldout_conditional_scores)
    budget_replication = config["training"]["budget_replication"]
    train_groups, train_audit = build_conditional_groups(
        sorted(args.train_render_dir.glob("samples-shard*.jsonl")),
        conditional_scores=train_scores,
        singleton_scores=singleton_train,
        budget_replication=budget_replication,
    )
    heldout_groups, heldout_audit = build_conditional_groups(
        sorted(args.heldout_render_dir.glob("samples-shard*.jsonl")),
        conditional_scores=heldout_scores,
        singleton_scores=singleton_heldout,
        budget_replication=budget_replication,
    )
    train = build_group_arrays(
        train_groups,
        feature_indices=feature_indices,
        episode_by_state=episode_train,
    )
    heldout = build_group_arrays(
        heldout_groups,
        feature_indices=feature_indices,
        episode_by_state=episode_heldout,
    )
    if set(train["episodes"]) & set(heldout["episodes"]):
        raise ValueError("episode leak between train and heldout")

    stage1_checkpoint = torch.load(
        args.stage1_checkpoint, map_location="cpu"
    )
    from sklearn.model_selection import GroupKFold

    fold_records = []
    oof_marginal = np.full_like(train["targets"], np.nan)
    oof_positive = np.full_like(train["targets"], np.nan)
    oof_rank = np.full_like(train["targets"], np.nan)
    oof_stop = np.full(
        len(train["group_keys"]), np.nan, dtype=np.float32
    )
    selected_epochs = []
    state_indices = np.arange(len(train["group_keys"]))
    splitter = GroupKFold(n_splits=5)
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(state_indices, groups=train["episodes"])
    ):
        best_state, best_epoch, history = fit_fold(
            train,
            feature_table,
            train_indices,
            validation_indices,
            config=config,
            stage1_checkpoint=stage1_checkpoint,
            device=device,
            seed=seed + fold,
        )
        model = initialize_model(
            config,
            stage1_checkpoint,
            device=device,
            seed=seed + fold,
        )
        model.load_state_dict(best_state)
        prediction = predict(
            model,
            train,
            feature_table,
            validation_indices,
            batch_size=int(config["training"]["batch_groups"]),
            device=device,
        )
        oof_marginal[validation_indices] = prediction["marginal"]
        oof_positive[validation_indices] = prediction["positive"]
        oof_rank[validation_indices] = prediction["rank"]
        oof_stop[validation_indices] = prediction["stop"]
        selected_epochs.append(best_epoch)
        fold_records.append(
            {
                "best_epoch": best_epoch,
                "fold": fold,
                "history": history,
                "train_groups": len(train_indices),
                "validation_groups": len(validation_indices),
            }
        )
    if not (
        np.isfinite(oof_marginal[train["candidate_mask"]]).all()
        and np.isfinite(oof_stop).all()
    ):
        raise RuntimeError("OOF prediction coverage is incomplete")
    oof_metrics, _ = group_metrics(
        train,
        {
            "marginal": oof_marginal,
            "positive": oof_positive,
            "rank": oof_rank,
            "stop": oof_stop,
        },
    )
    final_epochs = int(np.median(selected_epochs))
    final_model = fit_full(
        train,
        feature_table,
        epochs=final_epochs,
        config=config,
        stage1_checkpoint=stage1_checkpoint,
        device=device,
        seed=seed,
    )

    heldout_indices = np.arange(len(heldout["group_keys"]))
    heldout_prediction = predict(
        final_model,
        heldout,
        feature_table,
        heldout_indices,
        batch_size=int(config["training"]["batch_groups"]),
        device=device,
    )
    heldout_metrics, heldout_values = group_metrics(
        heldout, heldout_prediction
    )
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    delta_recent = heldout_values["selected"] - heldout_values["recent"]
    delta_random = heldout_values["selected"] - heldout_values["random"]
    heldout_metrics["selected_edge"] = {
        "model_mean_marginal": float(np.mean(heldout_values["selected"])),
        "model_minus_random_ci95": bootstrap_ci(
            delta_random, seed=seed + 1, replicates=replicates
        ),
        "model_minus_random_mean": float(np.mean(delta_random)),
        "model_minus_recent_ci95": bootstrap_ci(
            delta_recent, seed=seed, replicates=replicates
        ),
        "model_minus_recent_mean": float(np.mean(delta_recent)),
        "oracle_mean_marginal": float(np.mean(heldout_values["oracle"])),
        "random_mean_marginal": float(np.mean(heldout_values["random"])),
        "recent_mean_marginal": float(np.mean(heldout_values["recent"])),
    }
    positive_mask = heldout_prediction["marginal"][
        heldout["candidate_mask"]
    ] > 0
    truth_flat = heldout["targets"][heldout["candidate_mask"]]
    heldout_metrics["sign_precision"] = {
        "n_predicted_positive": int(positive_mask.sum()),
        "precision": (
            float(np.mean(truth_flat[positive_mask] > 0))
            if positive_mask.any()
            else None
        ),
    }

    checkpoint = {
        "config": config,
        "config_sha256": sha256_file(args.config),
        "feature_files": {
            str(path): sha256_file(path) for path in args.features
        },
        "final_epochs": final_epochs,
        "model_state_dict": final_model.cpu().state_dict(),
        "source_commit": args.source_commit,
        "stage1_checkpoint_sha256": sha256_file(args.stage1_checkpoint),
    }
    torch.save(checkpoint, args.output_dir / "stage2.pt")
    with (args.output_dir / "predictions-heldout.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for group_index, group_key in enumerate(heldout["group_keys"]):
            pair_group, anchor, final_budget = group_key
            for candidate_index in np.flatnonzero(
                heldout["candidate_mask"][group_index]
            ):
                handle.write(
                    json.dumps(
                        {
                            "anchor_set": list(anchor),
                            "candidate_event_id": int(
                                heldout["candidate_event_ids"][
                                    group_index, candidate_index
                                ]
                            ),
                            "final_budget": final_budget,
                            "pair_group": pair_group,
                            "positive_probability": float(
                                heldout_prediction["positive"][
                                    group_index, candidate_index
                                ]
                            ),
                            "predicted_marginal": float(
                                heldout_prediction["marginal"][
                                    group_index, candidate_index
                                ]
                            ),
                            "predicted_rank_score": float(
                                heldout_prediction["rank"][
                                    group_index, candidate_index
                                ]
                            ),
                            "stop_probability": float(
                                1
                                / (
                                    1
                                    + np.exp(
                                        -heldout_prediction["stop"][
                                            group_index
                                        ]
                                    )
                                )
                            ),
                            "stop_rank_score": float(
                                heldout_prediction["stop"][group_index]
                            ),
                            "true_marginal": float(
                                heldout["targets"][
                                    group_index, candidate_index
                                ]
                            ),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    metrics = {
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": config,
        "config_sha256": sha256_file(args.config),
        "data": {
            "feature_rows": len(feature_keys),
            "heldout_audit": heldout_audit,
            "heldout_groups": len(heldout["group_keys"]),
            "train_audit": train_audit,
            "train_groups": len(train["group_keys"]),
        },
        "device": str(device),
        "final_epochs": final_epochs,
        "folds": fold_records,
        "heldout": heldout_metrics,
        "oof_train": oof_metrics,
        "selected_epochs": selected_epochs,
        "source_commit": args.source_commit,
        "status": "DONE",
    }
    atomic_write_json(args.output_dir / "metrics.json", metrics)
    atomic_write_json(
        args.output_dir / "DONE",
        {
            "checkpoint": "stage2.pt",
            "completed_at": metrics["completed_at"],
            "final_epochs": final_epochs,
            "heldout_model_minus_recent_ci95": heldout_metrics[
                "selected_edge"
            ]["model_minus_recent_ci95"],
            "heldout_model_minus_recent_mean": heldout_metrics[
                "selected_edge"
            ]["model_minus_recent_mean"],
            "status": "DONE",
        },
    )


if __name__ == "__main__":
    main()
