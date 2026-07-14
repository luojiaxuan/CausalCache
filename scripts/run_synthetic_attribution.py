"""Run the deterministic phase-0 restoration estimator experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from causalcache.attribution import estimate_budget_conditioned_restoration, exact_permutation_restoration
from causalcache.contracts import ExperimentContract
from causalcache.selection import (
    jaccard,
    minimum_distance_coalition,
    restoration_utility,
    select_positive_value_knapsack,
    spearman,
)
from causalcache.synthetic import SyntheticBehaviorModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    return sum(materialized) / len(materialized)


def _sample_std(values: Iterable[float]) -> float:
    materialized = tuple(values)
    return statistics.stdev(materialized) if len(materialized) > 1 else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    if math.isclose(denominator, 0.0):
        return 1.0 if math.isclose(numerator, 0.0) else 0.0
    return numerator / denominator


def _round(value: float) -> float:
    return round(value, 6)


def run(contract: ExperimentContract, model: SyntheticBehaviorModel) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exact_scores = exact_permutation_restoration(model.distance, model.event_costs, budget=model.budget)
    exact_selection = select_positive_value_knapsack(exact_scores, model.event_costs, budget=model.budget)
    optimum_selection = minimum_distance_coalition(model.distance, model.event_costs, budget=model.budget)
    exact_utility = restoration_utility(model.distance, exact_selection)
    optimum_utility = restoration_utility(model.distance, optimum_selection)
    reconstruction_error = abs(exact_utility - sum(exact_scores[event_id] for event_id in exact_selection))
    rows: list[dict[str, Any]] = []

    for sample_count in contract.attribution.samples_sweep:
        for seed in contract.attribution.seeds:
            estimates = estimate_budget_conditioned_restoration(
                model.distance,
                model.event_costs,
                budget=model.budget,
                sample_count=sample_count,
                seed=seed,
            )
            scores = {event_id: estimate.mean for event_id, estimate in estimates.items()}
            selected = select_positive_value_knapsack(scores, model.event_costs, budget=model.budget)
            utility = restoration_utility(model.distance, selected)
            rows.append(
                {
                    "sample_count": sample_count,
                    "seed": seed,
                    "spearman": spearman(scores, exact_scores),
                    "top_budget_jaccard": jaccard(selected, exact_selection),
                    "mean_standard_error": _mean(estimate.standard_error for estimate in estimates.values()),
                    "actual_utility": utility,
                    "utility_ratio_to_exact_selection": _ratio(utility, exact_utility),
                    "utility_ratio_to_subset_optimum": _ratio(utility, optimum_utility),
                    "selected_events": " ".join(str(event_id) for event_id in sorted(selected)),
                }
            )

    aggregates: list[dict[str, Any]] = []
    for sample_count in contract.attribution.samples_sweep:
        matching = [row for row in rows if row["sample_count"] == sample_count]
        aggregates.append(
            {
                "sample_count": sample_count,
                "spearman_mean": _round(_mean(row["spearman"] for row in matching)),
                "spearman_std": _round(_sample_std(row["spearman"] for row in matching)),
                "jaccard_mean": _round(_mean(row["top_budget_jaccard"] for row in matching)),
                "jaccard_std": _round(_sample_std(row["top_budget_jaccard"] for row in matching)),
                "mean_standard_error": _round(_mean(row["mean_standard_error"] for row in matching)),
                "utility_ratio_to_exact_mean": _round(
                    _mean(row["utility_ratio_to_exact_selection"] for row in matching)
                ),
                "utility_ratio_to_optimum_mean": _round(
                    _mean(row["utility_ratio_to_subset_optimum"] for row in matching)
                ),
            }
        )

    summary = {
        "experiment": "synthetic_phase0_budget_conditioned_restoration",
        "scope": "implementation_validation_not_paper_evidence",
        "contract_version": contract.version,
        "budget": model.budget,
        "event_costs": {str(key): value for key, value in sorted(model.event_costs.items())},
        "exact_distribution": "all shared permutations with greedy maximal near-budget packing",
        "sampled_distribution": "shared antithetic permutations with greedy maximal near-budget packing",
        "exact_attribution": {str(key): _round(value) for key, value in sorted(exact_scores.items())},
        "negative_gain_events": [event_id for event_id, value in sorted(exact_scores.items()) if value <= 0],
        "exact_attribution_selection": sorted(exact_selection),
        "exact_attribution_selection_utility": _round(exact_utility),
        "exact_selection_reconstruction_error": _round(reconstruction_error),
        "global_subset_optimum": sorted(optimum_selection),
        "global_subset_optimum_utility": _round(optimum_utility),
        "exact_selection_ratio_to_subset_optimum": _round(_ratio(exact_utility, optimum_utility)),
        "aggregates": aggregates,
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _round(value) if isinstance(value, float) else value for key, value in row.items()})


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Synthetic Phase-0 Attribution Validation",
        "",
        "> 这只是 estimator implementation validation，不是论文效果证据，也不能替代真实 GUI policy 实验。",
        "",
        "## 设置",
        "",
        f"- Visual-token budget: `{summary['budget']}`",
        f"- Exact attribution selection: `{summary['exact_attribution_selection']}`",
        f"- Global subset optimum: `{summary['global_subset_optimum']}`",
        f"- Negative-gain events: `{summary['negative_gain_events']}`",
        f"- Exact-score selection / subset-optimum utility: `{summary['exact_selection_ratio_to_subset_optimum']:.3f}`",
        f"- Exact-selection reconstruction error: `{summary['exact_selection_reconstruction_error']:.3f}`",
        "",
        "## 采样稳定性（5 seeds）",
        "",
        "| K | Spearman mean +/- std | Jaccard mean +/- std | Mean SE | Utility / exact-score | Utility / subset optimum |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["aggregates"]:
        lines.append(
            f"| {row['sample_count']} | {row['spearman_mean']:.3f} +/- {row['spearman_std']:.3f} "
            f"| {row['jaccard_mean']:.3f} +/- {row['jaccard_std']:.3f} "
            f"| {row['mean_standard_error']:.3f} "
            f"| {row['utility_ratio_to_exact_mean']:.3f} "
            f"| {row['utility_ratio_to_optimum_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 判读边界",
            "",
            "该实验检查采样器、预算约束、负 gain 与选择器的接口是否闭合。它不检查真实视觉证据、模型 logits、teacher coverage、closed-loop success 或 matched-NLL 假设。",
            "",
            "该可控例子中 exact marginal-score selection 只达到 global subset optimum 的一部分，说明 event interaction 会破坏可加性；真实实验必须报告 reconstruction error，并保留 budget-aware set loss。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    contract = ExperimentContract.load(args.contract)
    model = SyntheticBehaviorModel.load(args.scenario)
    summary, rows = run(contract, model)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_report(args.output_report, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
