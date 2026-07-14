"""Machine-checked experiment contract for CausalCache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _positive_ints(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain positive integers")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be unique and sorted")
    return result


@dataclass(frozen=True)
class MemoryContract:
    budget_type: str
    cost_unit: str
    persistent_raw_archive: bool
    low_fidelity_schema: tuple[str, ...]
    visual_token_budgets: tuple[int, ...]
    selection_rule: str
    selection_threshold: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryContract":
        expected_schema = (
            "step_id",
            "action_type",
            "target_text_or_coordinate_bin",
            "deterministic_ui_delta",
            "result_status",
        )
        schema = tuple(str(item) for item in data["low_fidelity_schema"])
        if schema != expected_schema:
            raise ValueError("low_fidelity_schema must match the frozen five-field order")
        contract = cls(
            budget_type=str(data["budget_type"]),
            cost_unit=str(data["cost_unit"]),
            persistent_raw_archive=bool(data["persistent_raw_archive"]),
            low_fidelity_schema=schema,
            visual_token_budgets=_positive_ints(data["visual_token_budgets"], "visual_token_budgets"),
            selection_rule=str(data["selection_rule"]),
            selection_threshold=float(data["selection_threshold"]),
        )
        if contract.budget_type != "policy_visible_context":
            raise ValueError("budget_type must be policy_visible_context")
        if contract.cost_unit != "visual_tokens":
            raise ValueError("cost_unit must be visual_tokens")
        if not contract.persistent_raw_archive:
            raise ValueError("query-time retrieval requires a persistent raw archive")
        if contract.selection_rule != "positive_value_knapsack":
            raise ValueError("selection_rule must allow fewer than the maximum number of events")
        return contract


@dataclass(frozen=True)
class TeacherContract:
    validation_mode: str
    report_coverage: bool
    run_unfiltered_ablation: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TeacherContract":
        contract = cls(
            validation_mode=str(data["validation_mode"]),
            report_coverage=bool(data["report_coverage"]),
            run_unfiltered_ablation=bool(data["run_unfiltered_ablation"]),
        )
        if contract.validation_mode not in {"executable_match", "successful_trajectory", "either"}:
            raise ValueError("unsupported teacher validation_mode")
        if not contract.report_coverage:
            raise ValueError("validated teacher coverage must be reported")
        return contract


@dataclass(frozen=True)
class DistanceContract:
    components: tuple[str, ...]
    weights: Mapping[str, float]
    pathwise_teacher_forcing: bool
    target_canonicalization: str
    text_canonicalization: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DistanceContract":
        components = tuple(str(item) for item in data["components"])
        if components != ("action_type", "target", "text"):
            raise ValueError("distance components must be action_type, target, and text")
        weights_raw = _mapping(data["weights"], "distance.weights")
        weights = {name: float(weights_raw[name]) for name in components}
        if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("distance weights must be non-negative with positive total weight")
        contract = cls(
            components=components,
            weights=weights,
            pathwise_teacher_forcing=bool(data["pathwise_teacher_forcing"]),
            target_canonicalization=str(data["target_canonicalization"]),
            text_canonicalization=str(data["text_canonicalization"]),
        )
        if not contract.pathwise_teacher_forcing:
            raise ValueError("the first implementation requires pathwise teacher forcing")
        if contract.target_canonicalization != "ui_element_or_coordinate_bin_or_scroll_direction":
            raise ValueError("target canonicalization must preserve executable equivalence")
        return contract


@dataclass(frozen=True)
class AttributionContract:
    estimator: str
    coalition_mode: str
    samples_primary: int
    samples_sweep: tuple[int, ...]
    seeds: tuple[int, ...]
    shared_coalitions: bool
    report_metrics: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AttributionContract":
        contract = cls(
            estimator=str(data["estimator"]),
            coalition_mode=str(data["coalition_mode"]),
            samples_primary=int(data["samples_primary"]),
            samples_sweep=_positive_ints(data["samples_sweep"], "samples_sweep"),
            seeds=tuple(int(item) for item in data["seeds"]),
            shared_coalitions=bool(data["shared_coalitions"]),
            report_metrics=tuple(str(item) for item in data["report_metrics"]),
        )
        if contract.estimator != "budget_conditioned_restoration":
            raise ValueError("estimator must be budget_conditioned_restoration")
        if contract.coalition_mode != "near_budget":
            raise ValueError("coalition_mode must be near_budget")
        if contract.samples_primary not in contract.samples_sweep:
            raise ValueError("samples_primary must appear in samples_sweep")
        if not contract.seeds or len(set(contract.seeds)) != len(contract.seeds):
            raise ValueError("seeds must be non-empty and unique")
        required_metrics = {"standard_error", "spearman", "top_budget_jaccard", "oracle_utility"}
        if not required_metrics.issubset(contract.report_metrics):
            raise ValueError("attribution report_metrics are incomplete")
        return contract


@dataclass(frozen=True)
class EvaluationContract:
    primary_closed_loop_benchmark: str
    mechanism_test: str
    negative_control: str
    primary_axis: str
    control_variables: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationContract":
        contract = cls(
            primary_closed_loop_benchmark=str(data["primary_closed_loop_benchmark"]),
            mechanism_test=str(data["mechanism_test"]),
            negative_control=str(data["negative_control"]),
            primary_axis=str(data["primary_axis"]),
            control_variables=tuple(str(item) for item in data["control_variables"]),
        )
        if contract.primary_closed_loop_benchmark != "AndroidWorld":
            raise ValueError("AndroidWorld is the frozen primary closed-loop benchmark")
        if contract.mechanism_test != "matched_nll_conditional_success":
            raise ValueError("mechanism_test must be matched_nll_conditional_success")
        if contract.negative_control != "shuffle_restoration_labels":
            raise ValueError("negative_control must shuffle restoration labels")
        if contract.primary_axis != "success_vs_visual_tokens_and_latency":
            raise ValueError("primary_axis must expose the success-memory frontier")
        required_controls = {"successor_nll", "visual_tokens", "task_horizon"}
        if not required_controls.issubset(contract.control_variables):
            raise ValueError("mechanism test control variables are incomplete")
        return contract


@dataclass(frozen=True)
class ExperimentContract:
    version: str
    memory: MemoryContract
    teacher: TeacherContract
    distance: DistanceContract
    attribution: AttributionContract
    evaluation: EvaluationContract

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentContract":
        return cls(
            version=str(data["version"]),
            memory=MemoryContract.from_dict(_mapping(data["memory"], "memory")),
            teacher=TeacherContract.from_dict(_mapping(data["teacher"], "teacher")),
            distance=DistanceContract.from_dict(_mapping(data["distance"], "distance")),
            attribution=AttributionContract.from_dict(_mapping(data["attribution"], "attribution")),
            evaluation=EvaluationContract.from_dict(_mapping(data["evaluation"], "evaluation")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentContract":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(_mapping(data, "contract"))
