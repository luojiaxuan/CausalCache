#!/usr/bin/env python3
"""Train the frozen Stage-1 HGKV-readout singleton selector."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from causalcache.hgkv_selector import (
    HGKVSingletonSelector,
    model_config_from_dict,
    singleton_multitask_loss,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_scores(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            restored = row["memory_config"]["restored_event_step_ids"]
            if len(restored) != 1:
                raise ValueError(f"{path} contains a non-singleton row")
            key = (str(row["pair_group"]), int(restored[0]))
            if key in seen:
                raise ValueError(f"duplicate score key: {key}")
            seen.add(key)
            rows.append(
                {
                    "episode": str(row["episode"]),
                    "event_id": key[1],
                    "key": key,
                    "pair_group": key[0],
                    "target_logprob_mean": float(row["target_logprob_mean"]),
                }
            )
    return rows


def load_b0(path: Path) -> dict[str, float]:
    rows: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["memory_config"]["restored_event_step_ids"]:
                raise ValueError(f"{path} contains a non-B0 row")
            pair_group = str(row["pair_group"])
            if pair_group in rows:
                raise ValueError(f"duplicate B0 key: {pair_group}")
            rows[pair_group] = float(row["target_logprob_mean"])
    return rows


def load_features(
    paths: list[Path],
) -> tuple[np.ndarray, list[tuple[str, int]], dict[tuple[str, int], int]]:
    keys: list[tuple[str, int]] = []
    features: list[np.ndarray] = []
    indices: dict[tuple[str, int], int] = {}
    feature_dim = None
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (
                    str(row["pair_group"]),
                    int(row["singleton_event_step_id"]),
                )
                if key in indices:
                    raise ValueError(f"duplicate feature key: {key}")
                vector = np.asarray(row["feature"], dtype=np.float32)
                if feature_dim is None:
                    feature_dim = len(vector)
                if len(vector) != feature_dim:
                    raise ValueError(f"inconsistent feature dim for {key}")
                indices[key] = len(keys)
                keys.append(key)
                features.append(vector)
    if not features:
        raise ValueError("no feature rows loaded")
    return np.stack(features), keys, indices


def build_state_arrays(
    score_rows: list[dict[str, Any]],
    *,
    b0: dict[str, float],
    feature_table: np.ndarray,
    feature_indices: dict[tuple[str, int], int],
    max_candidates: int = 8,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[row["pair_group"]].append(row)
    states = sorted(grouped)
    feature_dim = feature_table.shape[1]
    features = np.zeros(
        (len(states), max_candidates, feature_dim), dtype=np.float32
    )
    targets = np.zeros((len(states), max_candidates), dtype=np.float32)
    event_ids = np.full((len(states), max_candidates), -1, dtype=np.int64)
    mask = np.zeros((len(states), max_candidates), dtype=bool)
    episodes: list[str] = []
    for state_index, pair_group in enumerate(states):
        candidates = sorted(grouped[pair_group], key=lambda item: -item["event_id"])
        if not candidates or len(candidates) > max_candidates:
            raise ValueError(
                f"{pair_group} has {len(candidates)} candidates, expected 1..{max_candidates}"
            )
        if pair_group not in b0:
            raise ValueError(f"B0 join miss for {pair_group}")
        episodes.append(candidates[0]["episode"])
        for candidate_index, row in enumerate(candidates):
            if row["key"] not in feature_indices:
                raise ValueError(f"feature join miss for {row['key']}")
            features[state_index, candidate_index] = feature_table[
                feature_indices[row["key"]]
            ]
            targets[state_index, candidate_index] = (
                row["target_logprob_mean"] - b0[pair_group]
            )
            event_ids[state_index, candidate_index] = row["event_id"]
            mask[state_index, candidate_index] = True
    return {
        "episodes": np.asarray(episodes),
        "event_ids": event_ids,
        "features": features,
        "mask": mask,
        "states": states,
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
    indices: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.from_numpy(arrays["features"][indices]).to(device)
    targets = torch.from_numpy(arrays["targets"][indices]).to(device)
    mask = torch.from_numpy(arrays["mask"][indices]).to(device)
    return features, targets, mask


def train_epoch(
    model: HGKVSingletonSelector,
    optimizer: torch.optim.Optimizer,
    arrays: dict[str, Any],
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
        indices, batch_size=int(training["batch_states"]), rng=rng
    ):
        features, targets, mask = tensor_batch(
            arrays, selected, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features, mask)
        loss, parts = singleton_multitask_loss(
            outputs,
            targets,
            mask,
            gain_weight=float(weights["gain"]),
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
    model: HGKVSingletonSelector,
    arrays: dict[str, Any],
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    width = arrays["targets"].shape[1]
    gains = np.zeros((len(indices), width), dtype=np.float32)
    positives = np.zeros_like(gains)
    ranks = np.zeros_like(gains)
    stops = np.zeros(len(indices), dtype=np.float32)
    offset = 0
    for selected in batches(indices, batch_size=batch_size, rng=None):
        features, _, mask = tensor_batch(arrays, selected, device=device)
        outputs = model(features, mask)
        size = len(selected)
        gains[offset : offset + size] = outputs["gain"].cpu().numpy()
        positives[offset : offset + size] = (
            outputs["positive_logit"].sigmoid().cpu().numpy()
        )
        ranks[offset : offset + size] = outputs["rank_score"].cpu().numpy()
        stops[offset : offset + size] = (
            outputs["stop_logit"].sigmoid().cpu().numpy()
        )
        offset += size
    return {
        "gain": gains,
        "positive": positives,
        "rank": ranks,
        "stop": stops,
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return ranks


def state_metrics(
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
    for state_index in range(len(arrays["states"])):
        mask = arrays["mask"][state_index]
        truth = arrays["targets"][state_index][mask]
        gain = predictions["gain"][state_index][mask]
        rank_score = predictions.get("rank", predictions["gain"])[state_index][
            mask
        ]
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
        at_most.append(realized if float(gain[chosen]) > 0 else 0.0)
        predicted_stop = bool(predictions["stop"][state_index] >= 0.5)
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
    summary = {
        "at_most1_realized_u_mean": float(np.mean(values["at_most_selected"])),
        "mean_regret": float(np.mean(values["regret"])),
        "mean_spearman": float(np.mean(rhos)) if rhos else None,
        "median_spearman": float(np.median(rhos)) if rhos else None,
        "n_states": len(arrays["states"]),
        "n_valid_spearman": len(rhos),
        "stop_accuracy": float(np.mean(stop_correct)),
        "top1_hit_rate": float(np.mean(values["hit"])),
    }
    return summary, values


def bootstrap_ci(
    values: np.ndarray, *, seed: int, replicates: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(replicates, dtype=np.float64)
    chunk = 1000
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        indices = rng.integers(0, n, size=(size, n))
        means[start : start + size] = values[indices].mean(axis=1)
    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def fit_fold(
    arrays: dict[str, Any],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    model = HGKVSingletonSelector(**model_config_from_dict(config)).to(device)
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
    for epoch in range(1, int(training["max_epochs"]) + 1):
        losses = train_epoch(
            model,
            optimizer,
            arrays,
            train_indices,
            config=config,
            device=device,
            rng=rng,
        )
        prediction = predict(
            model,
            arrays,
            validation_indices,
            batch_size=int(training["batch_states"]),
            device=device,
        )
        validation = subset_arrays(arrays, validation_indices)
        metrics, _ = state_metrics(validation, prediction)
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
    *,
    epochs: int,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> HGKVSingletonSelector:
    torch.manual_seed(seed)
    model = HGKVSingletonSelector(**model_config_from_dict(config)).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    indices = np.arange(len(arrays["states"]))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        train_epoch(
            model,
            optimizer,
            arrays,
            indices,
            config=config,
            device=device,
            rng=rng,
        )
    return model


def subset_arrays(
    arrays: dict[str, Any], indices: np.ndarray
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    state_count = len(arrays["states"])
    for key, value in arrays.items():
        if isinstance(value, np.ndarray) and len(value) == state_count:
            output[key] = value[indices]
        elif key == "states":
            output[key] = [value[index] for index in indices]
        else:
            output[key] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_table, feature_keys, feature_indices = load_features(args.features)
    train_scores = load_scores(args.scores_dir / "singleton-train.jsonl")
    heldout_scores = load_scores(args.scores_dir / "singleton-heldout.jsonl")
    b0 = load_b0(args.scores_dir / "b0-frozen-all.jsonl")
    score_keys = {row["key"] for row in train_scores + heldout_scores}
    if set(feature_keys) != score_keys:
        raise ValueError(
            f"feature/score key mismatch: features={len(feature_keys)} "
            f"scores={len(score_keys)}"
        )
    train = build_state_arrays(
        train_scores,
        b0=b0,
        feature_table=feature_table,
        feature_indices=feature_indices,
    )
    heldout = build_state_arrays(
        heldout_scores,
        b0=b0,
        feature_table=feature_table,
        feature_indices=feature_indices,
    )
    del feature_table
    if set(train["episodes"]) & set(heldout["episodes"]):
        raise ValueError("episode leak between train and heldout")

    from sklearn.model_selection import GroupKFold

    fold_records = []
    oof_gain = np.full_like(train["targets"], np.nan)
    oof_positive = np.full_like(train["targets"], np.nan)
    oof_rank = np.full_like(train["targets"], np.nan)
    oof_stop = np.full(len(train["states"]), np.nan, dtype=np.float32)
    selected_epochs = []
    splitter = GroupKFold(n_splits=5)
    state_indices = np.arange(len(train["states"]))
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(state_indices, groups=train["episodes"])
    ):
        best_state, best_epoch, history = fit_fold(
            train,
            train_indices,
            validation_indices,
            config=config,
            device=device,
            seed=seed + fold,
        )
        model = HGKVSingletonSelector(**model_config_from_dict(config)).to(device)
        model.load_state_dict(best_state)
        fold_prediction = predict(
            model,
            train,
            validation_indices,
            batch_size=int(config["training"]["batch_states"]),
            device=device,
        )
        oof_gain[validation_indices] = fold_prediction["gain"]
        oof_positive[validation_indices] = fold_prediction["positive"]
        oof_rank[validation_indices] = fold_prediction["rank"]
        oof_stop[validation_indices] = fold_prediction["stop"]
        selected_epochs.append(best_epoch)
        fold_records.append(
            {
                "best_epoch": best_epoch,
                "fold": fold,
                "history": history,
                "train_states": len(train_indices),
                "validation_states": len(validation_indices),
            }
        )

    if not (
        np.isfinite(oof_gain[train["mask"]]).all()
        and np.isfinite(oof_stop).all()
    ):
        raise RuntimeError("OOF prediction coverage is incomplete")
    oof_metrics, _ = state_metrics(
        train,
        {
            "gain": oof_gain,
            "positive": oof_positive,
            "rank": oof_rank,
            "stop": oof_stop,
        },
    )
    final_epochs = int(np.median(selected_epochs))
    final_model = fit_full(
        train,
        epochs=final_epochs,
        config=config,
        device=device,
        seed=seed,
    )

    heldout_indices = np.arange(len(heldout["states"]))
    heldout_prediction = predict(
        final_model,
        heldout,
        heldout_indices,
        batch_size=int(config["training"]["batch_states"]),
        device=device,
    )
    heldout_metrics, heldout_values = state_metrics(
        heldout, heldout_prediction
    )
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    delta_recent = heldout_values["selected"] - heldout_values["recent"]
    delta_random = heldout_values["selected"] - heldout_values["random"]
    heldout_metrics["b1_exactly_one"] = {
        "model_mean_u": float(np.mean(heldout_values["selected"])),
        "model_minus_random_ci95": bootstrap_ci(
            delta_random, seed=seed + 1, replicates=replicates
        ),
        "model_minus_random_mean": float(np.mean(delta_random)),
        "model_minus_recent_ci95": bootstrap_ci(
            delta_recent, seed=seed, replicates=replicates
        ),
        "model_minus_recent_mean": float(np.mean(delta_recent)),
        "oracle_mean_u": float(np.mean(heldout_values["oracle"])),
        "random_mean_u": float(np.mean(heldout_values["random"])),
        "recent_mean_u": float(np.mean(heldout_values["recent"])),
    }
    positive_mask = heldout_prediction["gain"][heldout["mask"]] > 0
    true_flat = heldout["targets"][heldout["mask"]]
    heldout_metrics["sign_precision"] = {
        "n_predicted_positive": int(positive_mask.sum()),
        "precision": (
            float(np.mean(true_flat[positive_mask] > 0))
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
        "score_files": {
            name: sha256_file(args.scores_dir / name)
            for name in (
                "b0-frozen-all.jsonl",
                "singleton-heldout.jsonl",
                "singleton-train.jsonl",
            )
        },
        "source_commit": args.source_commit,
    }
    torch.save(checkpoint, args.output_dir / "stage1.pt")
    with (args.output_dir / "predictions-heldout.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for state_index, pair_group in enumerate(heldout["states"]):
            for candidate_index in np.flatnonzero(heldout["mask"][state_index]):
                handle.write(
                    json.dumps(
                        {
                            "event_id": int(
                                heldout["event_ids"][
                                    state_index, candidate_index
                                ]
                            ),
                            "pair_group": pair_group,
                            "positive_probability": float(
                                heldout_prediction["positive"][
                                    state_index, candidate_index
                                ]
                            ),
                            "predicted_gain": float(
                                heldout_prediction["gain"][
                                    state_index, candidate_index
                                ]
                            ),
                            "predicted_rank_score": float(
                                heldout_prediction["rank"][
                                    state_index, candidate_index
                                ]
                            ),
                            "stop_probability": float(
                                heldout_prediction["stop"][state_index]
                            ),
                            "true_gain": float(
                                heldout["targets"][
                                    state_index, candidate_index
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
            "heldout_candidates": int(heldout["mask"].sum()),
            "heldout_states": len(heldout["states"]),
            "train_candidates": int(train["mask"].sum()),
            "train_states": len(train["states"]),
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
            "checkpoint": "stage1.pt",
            "completed_at": metrics["completed_at"],
            "final_epochs": final_epochs,
            "heldout_model_minus_recent_ci95": heldout_metrics[
                "b1_exactly_one"
            ]["model_minus_recent_ci95"],
            "heldout_model_minus_recent_mean": heldout_metrics[
                "b1_exactly_one"
            ]["model_minus_recent_mean"],
            "status": "DONE",
        },
    )


if __name__ == "__main__":
    main()
