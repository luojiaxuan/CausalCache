#!/usr/bin/env python3
"""Run frozen Stage-1/Stage-2 HGKV selection for budgets B1, B2, and B4."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from causalcache.hgkv_selector import (
    HGKVSetConditionedSelector,
    HGKVSingletonSelector,
    model_config_from_dict,
    set_model_config_from_dict,
)
from scripts.train_hgkv_singleton_selector_v1 import (
    atomic_write_json,
    load_features,
    load_scores,
    sha256_file,
)


@torch.no_grad()
def select_for_state(
    *,
    pair_group: str,
    event_ids: list[int],
    feature_rows: np.ndarray,
    budgets: list[int],
    stage1: HGKVSingletonSelector,
    stage2: HGKVSetConditionedSelector,
    device: torch.device,
) -> list[dict[str, Any]]:
    feature_tensor = torch.from_numpy(feature_rows).to(device)
    candidate_mask = torch.ones(
        (1, len(event_ids)), dtype=torch.bool, device=device
    )
    stage1_outputs = stage1(feature_tensor.unsqueeze(0), candidate_mask)
    first_rank = stage1_outputs["rank_score"][0]
    first_gain = stage1_outputs["gain"][0]
    first_positive = stage1_outputs["positive_logit"][0].sigmoid()
    first_stop = float(stage1_outputs["stop_logit"][0])
    first_index = int(torch.argmax(first_rank))
    first_is_stop = first_stop >= float(first_rank[first_index])
    results = []
    for budget in budgets:
        independent_order = [
            int(index)
            for index in torch.argsort(first_rank, descending=True).tolist()
            if float(first_rank[index]) > first_stop
        ]
        independent_selected = independent_order[:budget]
        results.append(
            {
                "budget": budget,
                "candidate_event_step_ids": event_ids,
                "method": "hgkv_singleton",
                "pair_group": pair_group,
                "selected_event_step_ids": [
                    event_ids[index] for index in independent_selected
                ],
                "steps": [
                    {
                        "candidate_event_id": event_ids[index],
                        "candidate_positive_probability": float(
                            first_positive[index]
                        ),
                        "candidate_predicted_gain": float(first_gain[index]),
                        "candidate_rank_score": float(first_rank[index]),
                        "stage": 1,
                        "stop_rank_score": first_stop,
                    }
                    for index in independent_order
                ],
                "stopped": len(independent_selected) < budget,
            }
        )
        selected_indices: list[int] = []
        steps = [
            {
                "candidate_event_id": event_ids[first_index],
                "candidate_positive_probability": float(
                    first_positive[first_index]
                ),
                "candidate_predicted_gain": float(first_gain[first_index]),
                "candidate_rank_score": float(first_rank[first_index]),
                "selected_before": [],
                "stage": 1,
                "stop_rank_score": first_stop,
            }
        ]
        stopped = first_is_stop
        if not stopped:
            selected_indices.append(first_index)
        while not stopped and len(selected_indices) < budget:
            remaining = [
                index
                for index in range(len(event_ids))
                if index not in selected_indices
            ]
            if not remaining:
                break
            candidate_features = feature_tensor[remaining].unsqueeze(0)
            remaining_mask = torch.ones(
                (1, len(remaining)), dtype=torch.bool, device=device
            )
            selected_features = torch.zeros(
                (1, 3, feature_tensor.shape[1]),
                dtype=feature_tensor.dtype,
                device=device,
            )
            selected_mask = torch.zeros(
                (1, 3), dtype=torch.bool, device=device
            )
            for position, index in enumerate(selected_indices):
                selected_features[0, position] = feature_tensor[index]
                selected_mask[0, position] = True
            remaining_budget = torch.tensor(
                [budget - len(selected_indices)],
                dtype=torch.long,
                device=device,
            )
            outputs = stage2(
                candidate_features,
                remaining_mask,
                selected_features,
                selected_mask,
                remaining_budget,
            )
            rank_scores = outputs["rank_score"][0]
            marginals = outputs["marginal"][0]
            positives = outputs["positive_logit"][0].sigmoid()
            local_index = int(torch.argmax(rank_scores))
            stop_score = float(outputs["stop_logit"][0])
            stopped = stop_score >= float(rank_scores[local_index])
            global_index = remaining[local_index]
            steps.append(
                {
                    "candidate_event_id": event_ids[global_index],
                    "candidate_positive_probability": float(
                        positives[local_index]
                    ),
                    "candidate_predicted_marginal": float(
                        marginals[local_index]
                    ),
                    "candidate_rank_score": float(rank_scores[local_index]),
                    "selected_before": [
                        event_ids[index] for index in selected_indices
                    ],
                    "stage": 2,
                    "stop_rank_score": stop_score,
                }
            )
            if not stopped:
                selected_indices.append(global_index)
        results.append(
            {
                "budget": budget,
                "candidate_event_step_ids": event_ids,
                "method": "hgkv_set_conditioned",
                "pair_group": pair_group,
                "selected_event_step_ids": [
                    event_ids[index] for index in selected_indices
                ],
                "steps": steps,
                "stopped": stopped,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--singleton-scores", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--device", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    if sorted(set(args.budgets)) != sorted(args.budgets):
        raise ValueError("budgets must be unique and sorted")
    if not args.budgets or min(args.budgets) < 1 or max(args.budgets) > 4:
        raise ValueError("budgets must be within 1..4")
    device = torch.device(args.device)
    feature_table, _, feature_indices = load_features(args.features)
    score_rows = load_scores(args.singleton_scores)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[row["pair_group"]].append(row)

    stage1_checkpoint = torch.load(
        args.stage1_checkpoint, map_location="cpu"
    )
    stage1 = HGKVSingletonSelector(
        **model_config_from_dict(stage1_checkpoint["config"])
    ).to(device)
    stage1.load_state_dict(stage1_checkpoint["model_state_dict"])
    stage1.eval()
    stage2_checkpoint = torch.load(
        args.stage2_checkpoint, map_location="cpu"
    )
    stage2 = HGKVSetConditionedSelector(
        **set_model_config_from_dict(stage2_checkpoint["config"])
    ).to(device)
    stage2.load_state_dict(stage2_checkpoint["model_state_dict"])
    stage2.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    row_count = 0
    selected_size_counts: dict[str, int] = defaultdict(int)
    with temporary.open("w", encoding="utf-8") as handle:
        for pair_group in sorted(grouped):
            candidates = sorted(
                grouped[pair_group], key=lambda row: -row["event_id"]
            )
            event_ids = [row["event_id"] for row in candidates]
            indices = []
            for event_id in event_ids:
                key = (pair_group, event_id)
                if key not in feature_indices:
                    raise ValueError(f"feature join miss: {key}")
                indices.append(feature_indices[key])
            results = select_for_state(
                pair_group=pair_group,
                event_ids=event_ids,
                feature_rows=feature_table[indices],
                budgets=args.budgets,
                stage1=stage1,
                stage2=stage2,
                device=device,
            )
            for result in results:
                result["episode"] = candidates[0]["episode"]
                handle.write(json.dumps(result, sort_keys=True) + "\n")
                row_count += 1
                selected_size_counts[
                    f"B{result['budget']}:K{len(result['selected_event_step_ids'])}"
                ] += 1
    temporary.replace(args.output)
    atomic_write_json(
        args.output.parent / "DONE",
        {
            "budgets": args.budgets,
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "output": args.output.name,
            "row_count": row_count,
            "selected_size_counts": dict(sorted(selected_size_counts.items())),
            "source_commit": args.source_commit,
            "stage1_checkpoint_sha256": sha256_file(args.stage1_checkpoint),
            "stage2_checkpoint_sha256": sha256_file(args.stage2_checkpoint),
            "status": "DONE",
        },
    )


if __name__ == "__main__":
    main()
