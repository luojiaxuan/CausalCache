#!/usr/bin/env python3
"""Train the three small set-utility predictors and score held-out states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from causalcache.set_utility_baselines import (
    ocr_rgb_selection,
    oracle_independent_j_selection,
    recent_selection,
)
from causalcache.set_utility_evaluation import (
    evaluate_joint_selectors,
    evaluate_prediction_table,
)
from causalcache.set_utility_label_table import (
    validate_cardinality_capped_distance_table,
)
from causalcache.set_utility_mvp import (
    COMPLETED_STATE_STATUS,
    SKIPPED_STATE_STATUS,
    canonical_json_bytes,
    feature_state_from_payload,
    state_from_completed_record,
)
from causalcache.set_utility_search import learned_joint_at_most_budget_search
from causalcache.set_utility_trainer_runner import (
    SetTransformerArchitectureConfig,
    TrainerConfig,
    UtilityModelConfig,
    run_cpu_utility_training,
)
from causalcache.set_utility_training import UtilityLossWeights


FAMILIES = ("pairwise_additive", "deepsets", "set_transformer")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _label_table(record: dict[str, Any]) -> Any:
    return validate_cardinality_capped_distance_table(
        split=record["role"],
        state_id=record["state_id"],
        candidate_event_step_ids=tuple(record["candidate_event_step_ids"]),
        maximum_labeled_cardinality=record["maximum_labeled_cardinality"],
        distances={
            tuple(row["coalition_event_step_ids"]): row["distance"]
            for row in record["distance_rows"]
        },
    )


def _trainer_config(config: dict[str, Any]) -> TrainerConfig:
    training = config["training"]
    loss = training["loss"]
    return TrainerConfig(
        seed=training["seed"],
        epochs=training["epochs"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        batch_strategy="trajectory_uniform_cycle",
        trajectories_per_batch=training["trajectories_per_batch"],
        states_per_trajectory_per_epoch=1,
        loss=UtilityLossWeights(
            raw_regression=loss["raw_regression"],
            normalized_regression=loss["normalized_regression"],
            within_state_ranking=loss["within_state_ranking"],
            smooth_l1_beta=loss["smooth_l1_beta"],
        ),
        maximum_gradient_norm=training["maximum_gradient_norm"],
        early_stopping_patience=training["early_stopping_patience"],
        early_stopping_min_delta=training["early_stopping_min_delta"],
    )


def _model_config(family: str, config: dict[str, Any]) -> UtilityModelConfig:
    hidden = config["training"]["hidden_dimension"]
    if family != "set_transformer":
        return UtilityModelConfig(family=family, hidden_dimension=hidden)
    architecture = config["training"]["set_transformer"]
    return UtilityModelConfig(
        family=family,
        hidden_dimension=hidden,
        set_transformer=SetTransformerArchitectureConfig(
            num_heads=architecture["num_heads"],
            num_layers=architecture["num_layers"],
            dropout=architecture["dropout"],
        ),
    )


def _evaluate(
    records: tuple[dict[str, Any], ...],
    *,
    models: dict[str, Any],
) -> dict[str, Any]:
    states = tuple(state_from_completed_record(record) for record in records)
    features = {
        record["state_id"]: feature_state_from_payload(record["feature"])
        for record in records
    }
    tables = {record["state_id"]: _label_table(record) for record in records}
    predictions = {}
    learned_selections_by_budget: dict[int, dict[str, dict[str, tuple[int, ...]]]] = {}
    for family, model in models.items():
        family_predictions = {}
        for state in states:
            search = learned_joint_at_most_budget_search(model, state, budget=2)
            family_predictions[state.state_id] = dict(search.scored_subsets)
        predictions[family] = evaluate_prediction_table(states, family_predictions)

    for budget in (1, 2):
        selections: dict[str, dict[str, tuple[int, ...]]] = {
            family: {} for family in models
        }
        selections.update(
            {f"{family}_fixed_B": {} for family in models}
        )
        selections.update({"recent": {}, "ocr_rgb": {}, "oracle_independent_J": {}})
        for state in states:
            state_id = state.state_id
            for family, model in models.items():
                search = learned_joint_at_most_budget_search(
                    model,
                    state,
                    budget=budget,
                )
                selections[family][state_id] = search.selected_subset
                fixed = tuple(
                    item for item in search.scored_subsets if len(item[0]) == budget
                )
                selections[f"{family}_fixed_B"][state_id] = min(
                    fixed,
                    key=lambda item: (-item[1], item[0]),
                )[0]
            selections["recent"][state_id] = recent_selection(
                features[state_id], budget=budget
            )
            selections["ocr_rgb"][state_id] = ocr_rgb_selection(
                features[state_id], budget=budget
            )
            selections["oracle_independent_J"][state_id] = (
                oracle_independent_j_selection(tables[state_id], budget=budget)
            )
        learned_selections_by_budget[budget] = selections
    return {
        "budgets": {
            str(budget): evaluate_joint_selectors(
                states,
                budget=budget,
                selections_by_method=learned_selections_by_budget[budget],
            )
            for budget in (1, 2)
        },
        "prediction": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    records = tuple(
        _read_json(path) for path in sorted((run_root / "states").glob("*.json"))
    )
    completed = tuple(
        record for record in records if record.get("status") == COMPLETED_STATE_STATUS
    )
    skipped = tuple(
        record
        for record in records
        if record.get("status") == SKIPPED_STATE_STATUS
    )
    counts = Counter(record["role"] for record in completed)
    minimum = config["training"]["minimum_accepted_states"]
    if any(counts[role] < minimum[role] for role in minimum):
        raise RuntimeError(f"accepted state counts are too small: {dict(counts)}")

    all_states = tuple(state_from_completed_record(record) for record in completed)
    trainable_state_ids = {
        state.state_id
        for state in all_states
        if state.normalization_scale is not None
        and any(abs(target.raw_utility) > 1e-12 for target in state.targets)
    }
    train_tune_states = tuple(
        state
        for state, record in zip(all_states, completed, strict=True)
        if record["role"] in {"train", "tune"}
        and state.state_id in trainable_state_ids
    )
    role_by_trajectory = {
        record["trajectory_id"]: record["role"]
        for record in completed
        if record["role"] in {"train", "tune"}
        and record["state_id"] in trainable_state_ids
    }
    trainable_counts = Counter(
        record["role"]
        for record in completed
        if record["state_id"] in trainable_state_ids
    )
    if not trainable_counts["train"] or not trainable_counts["tune"]:
        raise RuntimeError(
            f"no utility-varying train/tune states: {dict(trainable_counts)}"
        )
    trainer_config = _trainer_config(config)
    models = {}
    training_summaries = {}
    model_root = run_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    import torch

    started = time.time()
    for family in FAMILIES:
        result = run_cpu_utility_training(
            train_tune_states,
            role_by_trajectory=role_by_trajectory,
            family=family,
            hidden_dimension=config["training"]["hidden_dimension"],
            config=trainer_config,
            model_config=_model_config(family, config),
        )
        models[family] = result.model
        checkpoint_path = model_root / f"{family}.pt"
        torch.save(
            {
                "checkpoint_manifest": result.checkpoint_manifest,
                "checkpoint_payload": result.checkpoint_payload,
                "state_dict": result.state_dict,
            },
            checkpoint_path,
        )
        training_summaries[family] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "epoch_count": len(result.epoch_records),
            "selected_epoch": result.selected_epoch,
            "selected_tune_objective": result.selected_tune_objective,
            "state_dict_sha256": result.state_dict_manifest["state_dict_sha256"],
        }

    evaluation_records = tuple(
        record for record in completed if record["role"] == "evaluation"
    )
    evaluation = _evaluate(evaluation_records, models=models)
    summary = {
        "accepted_by_role": dict(sorted(counts.items())),
        "config_sha256": config_sha,
        "elapsed_training_and_evaluation_seconds": time.time() - started,
        "evaluation": evaluation,
        "failure_classes": dict(
            Counter(record.get("failure_class", "unknown") for record in skipped)
        ),
        "model_training": training_summaries,
        "planned_state_count": len(records),
        "schema_version": "1.0.0",
        "skipped_state_count": len(skipped),
        "status": "COMPLETED_SET_UTILITY_MVP_TRAINING_AND_EVALUATION",
        "utility_varying_by_role": dict(sorted(trainable_counts.items())),
    }
    _write_json(run_root / "summary.json", summary)
    print(
        json.dumps(
            {
                "accepted_by_role": summary["accepted_by_role"],
                "elapsed_training_and_evaluation_seconds": summary[
                    "elapsed_training_and_evaluation_seconds"
                ],
                "failure_classes": summary["failure_classes"],
                "status": summary["status"],
                "summary_path": str(run_root / "summary.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
