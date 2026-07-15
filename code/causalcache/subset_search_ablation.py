"""Frozen CPU-only subset-search ablation and cached-table replay."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from causalcache.attribution import exact_permutation_restoration
from causalcache.selection import select_positive_value_knapsack
from causalcache.subset_search import (
    Coalition,
    SearchResult,
    beam_subset_search,
    conditional_marginal_greedy,
    exact_subset_search,
    feasible_coalitions,
    greedy_with_bounded_exchange,
    second_order_interaction,
)
from causalcache.synthetic import SyntheticBehaviorModel


SetFunction = Callable[[Coalition], float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys differ from the frozen schema")


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def _sorted_positive_ints(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(_positive_int(item, name) for item in value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be unique and sorted")
    return result


@dataclass(frozen=True)
class InputBinding:
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], name: str) -> "InputBinding":
        _exact_keys(data, {"path", "sha256"}, name)
        path = str(data["path"])
        digest = str(data["sha256"])
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"{name}.path must be repository-relative")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{name}.sha256 must be lowercase SHA256")
        return cls(path=path, sha256=digest)


@dataclass(frozen=True)
class SubsetSearchAblationConfig:
    schema_version: str
    experiment_id: str
    scope: str
    utility_epsilon: float
    max_feasible_coalitions: int
    greedy_stopping_threshold: float
    greedy_score_rules: tuple[str, ...]
    omit_density_when_all_costs_equal: bool
    local_seed: str
    max_remove: int
    max_add: int
    max_passes: int
    beam_widths: tuple[int, ...]
    allow_nonpositive_prefixes: bool
    real_distance_column: str
    real_event_cost: int
    real_budgets: tuple[int, ...]
    real_search_stopping_threshold: float
    real_state_sensitivity_threshold: float
    scale_generator: str
    scale_event_counts: tuple[int, ...]
    scale_event_cost: int
    scale_slot_budget: int
    inputs: Mapping[str, InputBinding]
    claim_boundary: Mapping[str, bool]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubsetSearchAblationConfig":
        _exact_keys(
            data,
            {
                "schema_version",
                "experiment_id",
                "scope",
                "utility_epsilon",
                "exact_search",
                "greedy",
                "local_exchange",
                "beam_search",
                "real_table_replay",
                "scale_sweep",
                "inputs",
                "claim_boundary",
            },
            "config",
        )
        exact = _mapping(data["exact_search"], "exact_search")
        greedy = _mapping(data["greedy"], "greedy")
        local = _mapping(data["local_exchange"], "local_exchange")
        beam = _mapping(data["beam_search"], "beam_search")
        real = _mapping(data["real_table_replay"], "real_table_replay")
        scale = _mapping(data["scale_sweep"], "scale_sweep")
        inputs_raw = _mapping(data["inputs"], "inputs")
        boundary_raw = _mapping(data["claim_boundary"], "claim_boundary")
        _exact_keys(exact, {"max_feasible_coalitions"}, "exact_search")
        _exact_keys(
            greedy,
            {"stopping_threshold", "score_rules", "omit_density_when_all_costs_equal"},
            "greedy",
        )
        _exact_keys(local, {"seed", "max_remove", "max_add", "max_passes"}, "local_exchange")
        _exact_keys(beam, {"widths", "allow_nonpositive_prefixes"}, "beam_search")
        _exact_keys(
            real,
            {
                "distance_column",
                "event_cost",
                "budgets",
                "search_stopping_threshold",
                "state_sensitivity_threshold_for_separate_analysis",
            },
            "real_table_replay",
        )
        _exact_keys(scale, {"generator", "event_counts", "event_cost", "slot_budget"}, "scale_sweep")
        expected_inputs = {
            "controlled_scenarios",
            "phase0_scenario",
            "real_policy_coalition_table",
            "real_policy_compact_summary",
        }
        _exact_keys(inputs_raw, expected_inputs, "inputs")
        expected_boundary = {
            "new_policy_or_gpu_operations",
            "learned_gate_evaluated",
            "closed_loop_success_evaluated",
            "can_reopen_v2_1_no_go",
            "real_table_is_selection_biased_development_evidence",
            "true_utility_search_is_offline_oracle_diagnostic",
        }
        _exact_keys(boundary_raw, expected_boundary, "claim_boundary")

        epsilon = float(data["utility_epsilon"])
        stopping_threshold = float(greedy["stopping_threshold"])
        real_stopping_threshold = float(real["search_stopping_threshold"])
        state_threshold = float(real["state_sensitivity_threshold_for_separate_analysis"])
        if any(
            not math.isfinite(value) or value < 0
            for value in (epsilon, stopping_threshold, real_stopping_threshold, state_threshold)
        ):
            raise ValueError("all thresholds must be finite and non-negative")
        if real_stopping_threshold != stopping_threshold:
            raise ValueError("real replay must use the same canonical search stopping threshold")
        score_rules = tuple(str(item) for item in greedy["score_rules"])
        if score_rules != ("raw_gain", "gain_per_cost"):
            raise ValueError("greedy score rules must freeze raw then density")
        if str(local["seed"]) != "true_conditional_greedy_raw_gain":
            raise ValueError("local exchange must use the frozen raw-greedy seed")
        if str(scale["generator"]) != "deterministic_linear_pairwise_v1":
            raise ValueError("unsupported scale generator")
        if str(real["distance_column"]) != "distance_mean":
            raise ValueError("real table replay must use distance_mean")
        if (
            str(data["schema_version"]) != "1.0.0"
            or str(data["experiment_id"]) != "subset-search-ablation-v1"
            or str(data["scope"])
            != "synthetic_implementation_validation_and_posthoc_development_table_replay"
        ):
            raise ValueError("config identity or scope drifted")
        boundary = {str(key): _boolean(value, f"claim_boundary.{key}") for key, value in boundary_raw.items()}
        expected_truth = {
            "new_policy_or_gpu_operations": False,
            "learned_gate_evaluated": False,
            "closed_loop_success_evaluated": False,
            "can_reopen_v2_1_no_go": False,
            "real_table_is_selection_biased_development_evidence": True,
            "true_utility_search_is_offline_oracle_diagnostic": True,
        }
        if boundary != expected_truth:
            raise ValueError("claim boundary differs from the frozen negative declarations")
        return cls(
            schema_version=str(data["schema_version"]),
            experiment_id=str(data["experiment_id"]),
            scope=str(data["scope"]),
            utility_epsilon=epsilon,
            max_feasible_coalitions=_positive_int(exact["max_feasible_coalitions"], "exact limit"),
            greedy_stopping_threshold=stopping_threshold,
            greedy_score_rules=score_rules,
            omit_density_when_all_costs_equal=_boolean(
                greedy["omit_density_when_all_costs_equal"],
                "greedy.omit_density_when_all_costs_equal",
            ),
            local_seed=str(local["seed"]),
            max_remove=_positive_int(local["max_remove"], "max_remove"),
            max_add=_positive_int(local["max_add"], "max_add"),
            max_passes=_positive_int(local["max_passes"], "max_passes"),
            beam_widths=_sorted_positive_ints(beam["widths"], "beam widths"),
            allow_nonpositive_prefixes=_boolean(
                beam["allow_nonpositive_prefixes"],
                "beam_search.allow_nonpositive_prefixes",
            ),
            real_distance_column=str(real["distance_column"]),
            real_event_cost=_positive_int(real["event_cost"], "real event cost"),
            real_budgets=_sorted_positive_ints(real["budgets"], "real budgets"),
            real_search_stopping_threshold=real_stopping_threshold,
            real_state_sensitivity_threshold=state_threshold,
            scale_generator=str(scale["generator"]),
            scale_event_counts=_sorted_positive_ints(scale["event_counts"], "scale event counts"),
            scale_event_cost=_positive_int(scale["event_cost"], "scale event cost"),
            scale_slot_budget=_positive_int(scale["slot_budget"], "scale slot budget"),
            inputs={
                name: InputBinding.from_dict(_mapping(value, f"inputs.{name}"), f"inputs.{name}")
                for name, value in inputs_raw.items()
            },
            claim_boundary=boundary,
        )

    @classmethod
    def load(cls, path: Path) -> "SubsetSearchAblationConfig":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(_mapping(json.load(handle), "config"))


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    source_type: str
    event_costs: Mapping[int, int]
    budget: int
    utility: SetFunction
    distance: SetFunction | None = None
    metadata: Mapping[str, Any] | None = None


def verify_input_bindings(repository_root: Path, config: SubsetSearchAblationConfig) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, binding in sorted(config.inputs.items()):
        path = repository_root / binding.path
        if not path.is_file():
            raise ValueError(f"bound input is missing: {binding.path}")
        actual = sha256_file(path)
        if actual != binding.sha256:
            raise ValueError(f"bound input hash mismatch: {binding.path}")
        result[name] = {"path": binding.path, "sha256": actual}
    return result


def _parse_numeric_mapping(value: Any, name: str) -> dict[int, float]:
    raw = _mapping(value, name)
    result = {int(key): float(item) for key, item in raw.items()}
    if not result or any(not math.isfinite(item) for item in result.values()):
        raise ValueError(f"{name} must contain finite numeric values")
    return result


def _parse_costs(value: Any, name: str) -> dict[int, int]:
    raw = _mapping(value, name)
    result = {int(key): _positive_int(item, name) for key, item in raw.items()}
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _parse_pair_mapping(value: Any, name: str, event_ids: set[int]) -> dict[tuple[int, int], float]:
    raw = _mapping(value, name)
    result: dict[tuple[int, int], float] = {}
    for key, item in raw.items():
        parts = str(key).split(",")
        if len(parts) != 2:
            raise ValueError(f"{name} keys must be left,right")
        left, right = (int(part) for part in parts)
        value_float = float(item)
        if left >= right or left not in event_ids or right not in event_ids or not math.isfinite(value_float):
            raise ValueError(f"invalid pair in {name}: {key}")
        result[(left, right)] = value_float
    return result


def _polynomial_utility(
    linear: Mapping[int, float],
    pairwise: Mapping[tuple[int, int], float],
) -> SetFunction:
    def utility(coalition: Coalition) -> float:
        if not set(coalition).issubset(linear):
            raise ValueError("polynomial coalition contains an unknown event")
        return sum(linear[event_id] for event_id in coalition) + sum(
            value for (left, right), value in pairwise.items() if left in coalition and right in coalition
        )

    return utility


def load_controlled_scenarios(path: Path) -> list[Scenario]:
    with path.open("r", encoding="utf-8") as handle:
        data = _mapping(json.load(handle), "controlled fixture")
    _exact_keys(data, {"schema_version", "utility_family", "scenarios"}, "controlled fixture")
    if data["schema_version"] != "1.0.0" or data["utility_family"] != "linear_plus_pairwise":
        raise ValueError("unsupported controlled fixture schema")
    scenarios_raw = data["scenarios"]
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        raise ValueError("controlled scenarios must be a non-empty list")
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for index, item in enumerate(scenarios_raw):
        raw = _mapping(item, f"scenarios[{index}]")
        _exact_keys(
            raw,
            {"scenario_id", "budget", "event_costs", "linear_values", "pair_interactions"},
            f"scenarios[{index}]",
        )
        scenario_id = str(raw["scenario_id"])
        if not scenario_id or scenario_id in seen:
            raise ValueError("scenario ids must be non-empty and unique")
        seen.add(scenario_id)
        costs = _parse_costs(raw["event_costs"], f"{scenario_id}.event_costs")
        linear = _parse_numeric_mapping(raw["linear_values"], f"{scenario_id}.linear_values")
        if set(costs) != set(linear):
            raise ValueError("event costs and linear values must share ids")
        pairs = _parse_pair_mapping(raw["pair_interactions"], f"{scenario_id}.pair_interactions", set(costs))
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                source_type="controlled_synthetic",
                event_costs=costs,
                budget=_positive_int(raw["budget"], f"{scenario_id}.budget"),
                utility=_polynomial_utility(linear, pairs),
                metadata={"utility_family": "linear_plus_pairwise"},
            )
        )
    return scenarios


def load_phase0_scenario(path: Path) -> Scenario:
    model = SyntheticBehaviorModel.load(path)
    base_distance = model.distance(frozenset())
    return Scenario(
        scenario_id="existing_synthetic_phase0_mixed",
        source_type="existing_synthetic_phase0",
        event_costs=dict(model.event_costs),
        budget=model.budget,
        utility=lambda coalition: base_distance - model.distance(coalition),
        distance=model.distance,
        metadata={"base_distance": base_distance},
    )


def _parse_coalition(text: str) -> Coalition:
    if text == "empty":
        return frozenset()
    parts = text.split("+")
    coalition = frozenset(int(part) for part in parts)
    if len(coalition) != len(parts) or tuple(sorted(coalition)) != tuple(int(part) for part in parts):
        raise ValueError("coalition column must contain sorted unique event ids")
    return coalition


def load_real_table_scenarios(
    table_path: Path,
    summary_path: Path,
    config: SubsetSearchAblationConfig,
) -> list[Scenario]:
    with summary_path.open("r", encoding="utf-8") as handle:
        compact = _mapping(json.load(handle), "real compact summary")
    if compact.get("artifact_status") != "valid_selection_biased_diagnostic":
        raise ValueError("real replay input lost its selection-biased artifact declaration")
    outcome = _mapping(compact.get("outcome"), "real compact summary outcome")
    if outcome.get("status") != "INCONCLUSIVE_POSITIVE":
        raise ValueError("real replay input outcome drifted")
    by_step: dict[int, dict[Coalition, float]] = {}
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = [
            "decision_step_id",
            "coalition",
            "event_count",
            "incremental_visual_tokens",
            "distance_mean",
            "distance_sum",
        ]
        if reader.fieldnames != expected_fields:
            raise ValueError("real coalition table schema drifted")
        for row in reader:
            step_id = int(row["decision_step_id"])
            coalition = _parse_coalition(row["coalition"])
            if int(row["event_count"]) != len(coalition):
                raise ValueError("real coalition event_count mismatch")
            expected_tokens = len(coalition) * config.real_event_cost
            if int(row["incremental_visual_tokens"]) != expected_tokens:
                raise ValueError("real coalition visual-token accounting mismatch")
            distance = float(row[config.real_distance_column])
            if not math.isfinite(distance) or distance < 0:
                raise ValueError("real coalition distance must be finite and non-negative")
            if coalition in by_step.setdefault(step_id, {}):
                raise ValueError("duplicate real coalition row")
            by_step[step_id][coalition] = distance

    scenarios: list[Scenario] = []
    for step_id, table in sorted(by_step.items()):
        if frozenset() not in table:
            raise ValueError("real coalition table is missing its empty baseline")
        event_ids = sorted({event_id for coalition in table for event_id in coalition})
        costs = {event_id: config.real_event_cost for event_id in event_ids}
        base_distance = table[frozenset()]
        for budget in config.real_budgets:
            expected = set(feasible_coalitions(costs, budget))
            missing = expected - set(table)
            if missing:
                raise ValueError(f"real table misses feasible coalitions for step {step_id}: {sorted(map(sorted, missing))}")

            def distance(coalition: Coalition, *, _table: Mapping[Coalition, float] = table) -> float:
                try:
                    return _table[frozenset(coalition)]
                except KeyError as error:
                    raise ValueError("real replay requested an unevaluated coalition") from error

            def utility(
                coalition: Coalition,
                *,
                _distance: SetFunction = distance,
                _base: float = base_distance,
            ) -> float:
                return _base - _distance(coalition)

            scenarios.append(
                Scenario(
                    scenario_id=f"real_v1_step_{step_id}_budget_{budget}",
                    source_type="selection_biased_v1_cached_policy_table",
                    event_costs=costs,
                    budget=budget,
                    utility=utility,
                    distance=distance,
                    metadata={
                        "decision_step_id": step_id,
                        "base_distance": base_distance,
                        "replay_only": True,
                        "new_policy_forwards": 0,
                    },
                )
            )
    return scenarios


def build_scale_scenarios(config: SubsetSearchAblationConfig) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for event_count in config.scale_event_counts:
        event_ids = tuple(range(event_count))
        linear = {event_id: 8.0 - 0.5 * ((event_id * 7) % 9) for event_id in event_ids}
        pairwise: dict[tuple[int, int], float] = {}
        for left, right in itertools.combinations(event_ids, 2):
            if (left + right) % 4 == 0:
                pairwise[(left, right)] = 0.75 * (((left * 13 + right * 17) % 9) - 4)
        costs = {event_id: config.scale_event_cost for event_id in event_ids}
        scenarios.append(
            Scenario(
                scenario_id=f"scale_n{event_count}_slots{config.scale_slot_budget}",
                source_type="deterministic_synthetic_scale_sweep",
                event_costs=costs,
                budget=config.scale_slot_budget * config.scale_event_cost,
                utility=_polynomial_utility(linear, pairwise),
                metadata={
                    "generator": config.scale_generator,
                    "event_count": event_count,
                    "slot_budget": config.scale_slot_budget,
                },
            )
        )
    return scenarios


def _round_float(value: float) -> float:
    rounded = round(float(value), 15)
    return 0.0 if rounded == -0.0 else rounded


def _trace(result: SearchResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in result.trace:
        row = asdict(step)
        row["coalition"] = list(step.coalition)
        row["utility"] = _round_float(step.utility)
        if step.marginal_gain is not None:
            row["marginal_gain"] = _round_float(step.marginal_gain)
        rows.append(row)
    return rows


def _ratio(utility: float, oracle_utility: float, epsilon: float) -> tuple[float | None, str]:
    if oracle_utility <= epsilon:
        return None, "zero_or_nonpositive_oracle_utility"
    return _round_float(utility / oracle_utility), "informative_positive_oracle"


def _serialize_search_result(
    result: SearchResult,
    oracle: SearchResult,
    *,
    value_source: str,
    epsilon: float,
) -> dict[str, Any]:
    ratio, ratio_status = _ratio(result.utility, oracle.utility, epsilon)
    return {
        "method": result.algorithm,
        "value_source": value_source,
        "selected_coalition": sorted(result.coalition),
        "selected_cost": result.cost,
        "actual_utility": _round_float(result.utility),
        "utility_ratio_to_exact_subset": ratio,
        "utility_ratio_status": ratio_status,
        "absolute_regret_to_exact_subset": _round_float(oracle.utility - result.utility),
        "unique_set_utility_evaluations": result.unique_set_evaluations,
        "candidate_score_evaluations": result.candidate_score_evaluations,
        "frontier_expansions": result.frontier_expansions,
        "sequential_rounds": result.sequential_rounds,
        "converged": result.converged,
        "trace": _trace(result),
    }


def _static_average_value_knapsack(
    scenario: Scenario,
    exact: SearchResult,
    *,
    epsilon: float,
) -> dict[str, Any] | None:
    if scenario.distance is None:
        return None
    event_ids = tuple(sorted(scenario.event_costs))
    if len(event_ids) > 9:
        raise ValueError("static exact permutation replay is limited to nine events")
    index_to_event = dict(enumerate(event_ids))
    event_to_index = {event_id: index for index, event_id in index_to_event.items()}
    indexed_costs = {event_to_index[event_id]: scenario.event_costs[event_id] for event_id in event_ids}
    distance_calls = 0

    def indexed_distance(coalition: Coalition) -> float:
        nonlocal distance_calls
        distance_calls += 1
        original = frozenset(index_to_event[index] for index in coalition)
        return scenario.distance(original)

    indexed_scores = exact_permutation_restoration(
        indexed_distance,
        indexed_costs,
        budget=scenario.budget,
        max_events=9,
    )
    indexed_selection = select_positive_value_knapsack(
        indexed_scores,
        indexed_costs,
        budget=scenario.budget,
    )
    selected = frozenset(index_to_event[index] for index in indexed_selection)
    utility = float(scenario.utility(selected))
    ratio, ratio_status = _ratio(utility, exact.utility, epsilon)
    return {
        "method": "exact_average_marginal_positive_value_knapsack",
        "value_source": "exact_budget_conditioned_average_marginal",
        "selected_coalition": sorted(selected),
        "selected_cost": sum(scenario.event_costs[event_id] for event_id in selected),
        "actual_utility": _round_float(utility),
        "utility_ratio_to_exact_subset": ratio,
        "utility_ratio_status": ratio_status,
        "absolute_regret_to_exact_subset": _round_float(exact.utility - utility),
        "unique_set_utility_evaluations": None,
        "candidate_score_evaluations": len(event_ids),
        "teacher_distance_evaluations": distance_calls,
        "sequential_rounds": 1,
        "converged": True,
        "trace": [],
        "interpretation": "projection_objective_gap_not_greedy_search_gap",
    }


def _interaction_summary(scenario: Scenario, epsilon: float) -> dict[str, Any]:
    empty = frozenset()
    feasible_pairs = [
        (left, right)
        for left, right in itertools.combinations(sorted(scenario.event_costs), 2)
        if scenario.event_costs[left] + scenario.event_costs[right] <= scenario.budget
    ]
    rows: list[dict[str, Any]] = []
    for left, right in feasible_pairs:
        value = second_order_interaction(scenario.utility, empty, left, right)
        rows.append({"events": [left, right], "interaction": _round_float(value)})
    complementarity_mass = sum(max(row["interaction"], 0.0) for row in rows)
    redundancy_mass = sum(max(-row["interaction"], 0.0) for row in rows)
    marginal_mass = sum(
        abs(float(scenario.utility(frozenset({event_id}))) - float(scenario.utility(empty)))
        for event_id in scenario.event_costs
        if scenario.event_costs[event_id] <= scenario.budget
    )
    strongest_positive = sorted(rows, key=lambda row: (-row["interaction"], row["events"]))[:5]
    strongest_negative = sorted(rows, key=lambda row: (row["interaction"], row["events"]))[:5]
    return {
        "conditioning_coalition": [],
        "feasible_pair_count": len(rows),
        "positive_pair_count": sum(row["interaction"] > epsilon for row in rows),
        "negative_pair_count": sum(row["interaction"] < -epsilon for row in rows),
        "near_zero_pair_count": sum(abs(row["interaction"]) <= epsilon for row in rows),
        "complementarity_mass": _round_float(complementarity_mass),
        "redundancy_mass": _round_float(redundancy_mass),
        "absolute_interaction_mass": _round_float(complementarity_mass + redundancy_mass),
        "normalized_interaction_mass": (
            None if marginal_mass <= epsilon else _round_float((complementarity_mass + redundancy_mass) / marginal_mass)
        ),
        "strongest_positive_pairs": strongest_positive,
        "strongest_negative_pairs": strongest_negative,
        "boundary": "second_order_at_empty_only",
    }


def evaluate_scenario(scenario: Scenario, config: SubsetSearchAblationConfig) -> dict[str, Any]:
    stopping_threshold = (
        config.real_search_stopping_threshold
        if scenario.source_type == "selection_biased_v1_cached_policy_table"
        else config.greedy_stopping_threshold
    )
    exact = exact_subset_search(
        scenario.utility,
        scenario.event_costs,
        budget=scenario.budget,
        epsilon=config.utility_epsilon,
        max_feasible_coalitions=config.max_feasible_coalitions,
    )
    results = [
        _serialize_search_result(
            exact,
            exact,
            value_source="true_set_utility",
            epsilon=config.utility_epsilon,
        )
    ]
    skipped: list[dict[str, str]] = []
    all_costs_equal = len(set(scenario.event_costs.values())) == 1
    for score_rule in config.greedy_score_rules:
        if score_rule == "gain_per_cost" and all_costs_equal and config.omit_density_when_all_costs_equal:
            skipped.append(
                {
                    "method": "true_conditional_greedy_gain_per_cost",
                    "reason": "identical_to_raw_gain_under_equal_event_costs",
                }
            )
            continue
        greedy = conditional_marginal_greedy(
            scenario.utility,
            scenario.event_costs,
            budget=scenario.budget,
            score_rule=score_rule,
            stopping_threshold=stopping_threshold,
            epsilon=config.utility_epsilon,
        )
        results.append(
            _serialize_search_result(
                greedy,
                exact,
                value_source="true_conditional_marginal",
                epsilon=config.utility_epsilon,
            )
        )
    exchange = greedy_with_bounded_exchange(
        scenario.utility,
        scenario.event_costs,
        budget=scenario.budget,
        score_rule="raw_gain",
        stopping_threshold=stopping_threshold,
        epsilon=config.utility_epsilon,
        max_remove=config.max_remove,
        max_add=config.max_add,
        max_passes=config.max_passes,
    )
    results.append(
        _serialize_search_result(
            exchange,
            exact,
            value_source="true_set_utility_with_true_raw_greedy_seed",
            epsilon=config.utility_epsilon,
        )
    )
    for width in config.beam_widths:
        beam = beam_subset_search(
            scenario.utility,
            scenario.event_costs,
            budget=scenario.budget,
            width=width,
            epsilon=config.utility_epsilon,
            allow_nonpositive_prefixes=config.allow_nonpositive_prefixes,
        )
        results.append(
            _serialize_search_result(
                beam,
                exact,
                value_source="true_set_utility",
                epsilon=config.utility_epsilon,
            )
        )
    static = _static_average_value_knapsack(scenario, exact, epsilon=config.utility_epsilon)
    if static is not None:
        results.append(static)

    sensitivity: dict[str, Any] | None = None
    if scenario.source_type == "selection_biased_v1_cached_policy_table":
        thresholded = conditional_marginal_greedy(
            scenario.utility,
            scenario.event_costs,
            budget=scenario.budget,
            score_rule="raw_gain",
            stopping_threshold=config.real_state_sensitivity_threshold,
            epsilon=config.utility_epsilon,
        )
        sensitivity = _serialize_search_result(
            thresholded,
            exact,
            value_source="true_conditional_marginal",
            epsilon=config.utility_epsilon,
        )
        sensitivity["method"] = "threshold_sensitivity_true_greedy_raw_gain"
        sensitivity["stopping_threshold"] = config.real_state_sensitivity_threshold
        sensitivity["interpretation"] = "abstention_threshold_sensitivity_not_search_regret"

    return {
        "scenario_id": scenario.scenario_id,
        "source_type": scenario.source_type,
        "event_count": len(scenario.event_costs),
        "event_costs": {str(key): value for key, value in sorted(scenario.event_costs.items())},
        "budget": scenario.budget,
        "feasible_coalition_count": sum(1 for _ in feasible_coalitions(scenario.event_costs, scenario.budget)),
        "metadata": dict(scenario.metadata or {}),
        "interaction_at_empty": _interaction_summary(scenario, config.utility_epsilon),
        "methods": results,
        "skipped_methods": skipped,
        "threshold_sensitivity": sensitivity,
    }


def build_scientific_payload(
    repository_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    config_resolved = config_path.resolve()
    try:
        config_relative = config_resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("config must be inside the repository") from error
    config = SubsetSearchAblationConfig.load(config_resolved)
    bindings = verify_input_bindings(root, config)
    controlled = load_controlled_scenarios(root / config.inputs["controlled_scenarios"].path)
    phase0 = load_phase0_scenario(root / config.inputs["phase0_scenario"].path)
    real = load_real_table_scenarios(
        root / config.inputs["real_policy_coalition_table"].path,
        root / config.inputs["real_policy_compact_summary"].path,
        config,
    )
    scales = build_scale_scenarios(config)
    scenarios = controlled + [phase0] + real + scales
    results = [evaluate_scenario(scenario, config) for scenario in scenarios]
    return {
        "schema_version": config.schema_version,
        "experiment_id": config.experiment_id,
        "scope": config.scope,
        "config": {
            "path": config_relative,
            "sha256": sha256_file(config_resolved),
            "utility_epsilon": config.utility_epsilon,
            "exact_max_feasible_coalitions": config.max_feasible_coalitions,
            "greedy_stopping_threshold": config.greedy_stopping_threshold,
            "local_exchange": {
                "seed": config.local_seed,
                "max_remove": config.max_remove,
                "max_add": config.max_add,
                "max_passes": config.max_passes,
            },
            "beam_widths": list(config.beam_widths),
            "allow_nonpositive_prefixes": config.allow_nonpositive_prefixes,
        },
        "input_bindings": bindings,
        "scenarios": results,
        "claim_boundary": dict(config.claim_boundary),
        "terminology": {
            "exact_average_marginal_knapsack_gap": "interaction_or_objective_projection_gap",
            "true_utility_greedy_to_exact_gap": "search_regret",
            "true_utility_search_cost": "offline_policy_rerun_equivalent_queries_not_online_gate_latency",
            "learned_selector_residual": "not_measured",
            "exact_subset_optimality_scope": "single_step_restoration_utility_not_terminal_success",
        },
        "v2_1_outcome_unchanged": "NO_GO_V2_1_FULL_45_SUBSTRATE",
    }
