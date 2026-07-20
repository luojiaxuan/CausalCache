"""Fail-closed reduction for post-GO native-action replay."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from causalcache.policy.gui_owl_v2_1 import parse_gui_owl_v2_1_output
from causalcache.set_utility_heldout_evaluation import (
    COMPLETED_EVALUATION,
    SEALED_SELECTIONS,
    paired_trajectory_bootstrap,
    percentile_type7,
)
from causalcache.set_utility_native_replay import (
    NATIVE_REPLAY_METHODS,
    NATIVE_REPLAY_SCHEMA_VERSION,
    NATIVE_REPLAY_SHARD_STATUS,
    NATIVE_REPLAY_STATUS,
    canonical_json_bytes,
    compare_action_components,
    sha256_file,
    signed_content_hash_is_valid,
)


COMPLETED_NATIVE_REPLAY_EVALUATION = (
    "COMPLETED_SET_UTILITY_NATIVE_REPLAY_EVALUATION"
)
INCOMPLETE_NATIVE_REPLAY_EVALUATION = (
    "INCOMPLETE_SET_UTILITY_NATIVE_REPLAY_EVALUATION"
)
COMPLETED_REPLAY_STATE = "COMPLETED_SET_UTILITY_NATIVE_REPLAY_STATE"
COMPLETED_REPLAY_COALITION = "COMPLETED_SET_UTILITY_NATIVE_REPLAY_COALITION"
FAILED_REPLAY_PARSE = "FAILED_SET_UTILITY_NATIVE_REPLAY_PARSE"
EXPECTED_EXACT_STATE_COUNT = 320
EXPECTED_LOGICAL_SHARD_COUNT = 256
DEPLOYMENT_LATENCY_RATIO_MAXIMUM = 0.1
BEHAVIOR_PRIMARY_METRIC = "canonical_action_exact"
BEHAVIOR_BOOTSTRAP_RESAMPLES = 10_000
BEHAVIOR_BOOTSTRAP_SEED = 20_260_720
BEHAVIOR_BOOTSTRAP_INTERVAL = 0.95
_BUDGETS = (1, 2, 3, 4)
_RATE_METRICS = (
    "parse_coverage",
    "canonical_action_exact",
    "action_type_exact",
    "target_exact",
    "text_nfkc_exact",
)
_BEHAVIOR_METRICS = (
    "canonical_action_exact",
    "action_type_exact",
    "target_exact",
    "text_nfkc_exact",
)
_MANIFEST_BINDINGS = (
    "config_sha256",
    "heldout_result_sha256",
    "inventory_sha256",
    "reference_terminal_inventory_sha256",
    "selections_sha256",
    "source_manifest_sha256",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain one JSON object")
        rows.append(row)
    return tuple(rows)


def _finite(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        raise ValueError(f"{label} is outside its finite domain")
    return result


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)[:-1]).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_subset(
    value: Any, *, candidates: tuple[int, ...], budget: int, label: str
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be an event-id array")
    subset = tuple(value)
    if (
        any(type(item) is not int for item in subset)
        or subset != tuple(sorted(subset))
        or len(subset) != len(set(subset))
        or len(subset) > budget
        or not set(subset).issubset(candidates)
    ):
        raise ValueError(f"{label} is not a canonical at-most-{budget} subset")
    return subset


def _validate_inputs(
    *,
    manifest: Mapping[str, Any],
    selections: Mapping[str, Any],
    selections_sha256: str,
    heldout_result: Mapping[str, Any],
    heldout_result_sha256: str,
) -> tuple[str, dict[str, Mapping[str, Any]]]:
    if (
        manifest.get("status") != NATIVE_REPLAY_STATUS
        or manifest.get("schema_version") != NATIVE_REPLAY_SCHEMA_VERSION
        or not signed_content_hash_is_valid(manifest)
        or tuple(manifest.get("methods", ())) != NATIVE_REPLAY_METHODS
        or tuple(manifest.get("budgets", ())) != _BUDGETS
        or manifest.get("state_count") != EXPECTED_EXACT_STATE_COUNT
        or manifest.get("logical_shard_count") != EXPECTED_LOGICAL_SHARD_COUNT
    ):
        raise ValueError("native replay manifest is not the frozen 320-state schedule")
    if (
        selections.get("status") != SEALED_SELECTIONS
        or not signed_content_hash_is_valid(selections)
        or manifest.get("selections_sha256") != selections_sha256
    ):
        raise ValueError("native replay selections binding is invalid")
    coverage = heldout_result.get("coverage")
    winner = heldout_result.get("winner")
    heldout_bindings = heldout_result.get("bindings")
    if (
        heldout_result.get("status") != COMPLETED_EVALUATION
        or not signed_content_hash_is_valid(heldout_result)
        or not isinstance(coverage, Mapping)
        or coverage.get("complete") is not True
        or coverage.get("completed_state_count") != coverage.get("expected_state_count")
        or coverage.get("missing_state_ids") not in ([], ())
        or coverage.get("skipped_state_count") != 0
        or not isinstance(winner, Mapping)
        or winner.get("verdict") != "GO"
        or manifest.get("heldout_result_sha256") != heldout_result_sha256
    ):
        raise ValueError("held-out result does not authorize native replay reduction")
    if (
        not isinstance(heldout_bindings, Mapping)
        or heldout_bindings.get("selections_sha256") != selections_sha256
        or heldout_bindings.get("config_sha256")
        != manifest.get("config_sha256")
        or selections.get("config_sha256") != manifest.get("config_sha256")
        or selections.get("inventory_sha256") != manifest.get("inventory_sha256")
    ):
        raise ValueError("held-out result and native replay bindings drifted")
    winner_model = winner.get("model")
    model_artifacts = selections.get("model_artifacts")
    if (
        not isinstance(winner_model, str)
        or winner_model != manifest.get("winner_model")
        or not isinstance(model_artifacts, Mapping)
        or winner_model not in model_artifacts
    ):
        raise ValueError("held-out winner drifted from the native replay schedule")
    records = selections.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("sealed selections omit their state records")
    exact_records: dict[str, Mapping[str, Any]] = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("sealed selection state is not an object")
        if "exact_oracle" not in row.get("tracks", ()):
            continue
        state_id = row.get("state_id")
        if not isinstance(state_id, str) or state_id in exact_records:
            raise ValueError("sealed exact-track state identities are invalid")
        exact_records[state_id] = row
    if len(exact_records) != EXPECTED_EXACT_STATE_COUNT:
        raise ValueError("sealed selections do not contain exact-track 320")
    return winner_model, exact_records


def _expected_coalitions(
    selection: Mapping[str, Any], *, winner_model: str
) -> tuple[dict[str, Any], ...]:
    candidates = tuple(selection.get("candidate_event_ids", ()))
    if (
        not candidates
        or any(type(item) is not int for item in candidates)
        or candidates != tuple(sorted(candidates))
        or len(candidates) != len(set(candidates))
    ):
        raise ValueError("sealed selection candidate universe is invalid")
    methods = selection.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("sealed selection omits methods")
    mapping = {"winner": winner_model, "recent": "recent", "ocr_rgb": "ocr_rgb"}
    tags: dict[tuple[int, ...], set[str]] = {}
    for replay_method, selector_method in mapping.items():
        budgets = methods.get(selector_method)
        if not isinstance(budgets, Mapping) or set(budgets) != {
            str(item) for item in _BUDGETS
        }:
            raise ValueError(f"sealed selection method drifted: {selector_method}")
        for budget in _BUDGETS:
            subset = _canonical_subset(
                budgets[str(budget)],
                candidates=candidates,
                budget=budget,
                label=f"{selector_method} B{budget}",
            )
            tags.setdefault(subset, set()).add(f"{replay_method}:B{budget}")
    return tuple(
        {
            "event_ids": list(subset),
            "sources": sorted(sources),
        }
        for subset, sources in sorted(
            tags.items(), key=lambda item: (len(item[0]), item[0])
        )
    )


def load_and_validate_native_replay_schedule(
    *,
    schedule_root: Path,
    selections_path: Path,
    heldout_result_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = schedule_root / "manifest.json"
    manifest = _read_json(manifest_path)
    selections = _read_json(selections_path)
    heldout_result = _read_json(heldout_result_path)
    manifest_sha = sha256_file(manifest_path)
    selections_sha = sha256_file(selections_path)
    heldout_result_sha = sha256_file(heldout_result_path)
    winner_model, exact_records = _validate_inputs(
        manifest=manifest,
        selections=selections,
        selections_sha256=selections_sha,
        heldout_result=heldout_result,
        heldout_result_sha256=heldout_result_sha,
    )
    shard_records = manifest.get("shards")
    if not isinstance(shard_records, Sequence) or isinstance(
        shard_records, (str, bytes)
    ):
        raise ValueError("native replay manifest omits shard records")
    by_shard = {}
    for record in shard_records:
        if not isinstance(record, Mapping):
            raise ValueError("native replay shard manifest record is invalid")
        logical_shard = record.get("logical_shard")
        if type(logical_shard) is not int or logical_shard in by_shard:
            raise ValueError("native replay shard manifest identity is invalid")
        by_shard[logical_shard] = record
    if set(by_shard) != set(range(EXPECTED_LOGICAL_SHARD_COUNT)):
        raise ValueError("native replay shard inventory is incomplete")

    schedule_rows: dict[str, dict[str, Any]] = {}
    observed_coalitions = 0
    for logical_shard in range(EXPECTED_LOGICAL_SHARD_COUNT):
        schedule_path = (
            schedule_root
            / "schedule-shards"
            / f"shard-{logical_shard:03d}-of-{EXPECTED_LOGICAL_SHARD_COUNT}.jsonl"
        )
        receipt_path = (
            schedule_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-{EXPECTED_LOGICAL_SHARD_COUNT}.json"
        )
        receipt = _read_json(receipt_path)
        shard_record = by_shard[logical_shard]
        if (
            sha256_file(receipt_path) != shard_record.get("receipt_sha256")
            or sha256_file(schedule_path) != shard_record.get("schedule_sha256")
            or receipt.get("status") != NATIVE_REPLAY_SHARD_STATUS
            or receipt.get("logical_shard") != logical_shard
            or receipt.get("schedule_sha256") != shard_record.get("schedule_sha256")
        ):
            raise ValueError(f"native replay shard {logical_shard} receipt drifted")
        for binding in _MANIFEST_BINDINGS:
            if receipt.get(binding) != manifest.get(binding):
                raise ValueError(f"native replay shard {logical_shard} binding drifted")
        rows = _read_jsonl(schedule_path)
        coalition_count = sum(len(row.get("coalitions", ())) for row in rows)
        if (
            receipt.get("state_count") != len(rows)
            or shard_record.get("state_count") != len(rows)
            or receipt.get("coalition_count") != coalition_count
            or shard_record.get("coalition_count") != coalition_count
        ):
            raise ValueError(f"native replay shard {logical_shard} counts drifted")
        observed_coalitions += coalition_count
        for row in rows:
            state_id = row.get("state_id")
            selection = exact_records.get(state_id)
            if not isinstance(state_id, str) or state_id in schedule_rows:
                raise ValueError("native replay schedule state identity is invalid")
            if selection is None:
                raise ValueError("native replay schedule contains a non-exact state")
            expected_coalitions = _expected_coalitions(
                selection, winner_model=winner_model
            )
            if (
                row.get("logical_shard") != logical_shard
                or selection.get("logical_shard") != logical_shard
                or row.get("trajectory_id") != selection.get("trajectory_id")
                or row.get("candidate_event_ids")
                != selection.get("candidate_event_ids")
                or row.get("role") != "evaluation"
                or row.get("winner_model") != winner_model
                or tuple(row.get("coalitions", ())) != expected_coalitions
            ):
                raise ValueError(f"native replay schedule row drifted: {state_id}")
            reference = row.get("reference")
            if not isinstance(reference, Mapping):
                raise ValueError(f"native replay schedule reference is missing: {state_id}")
            parsed = parse_gui_owl_v2_1_output(
                reference.get("serialized_action", "")
            ).canonical_action
            if (
                parsed.arguments() != reference.get("canonical_action")
                or not _is_sha256(reference.get("terminal_sha256"))
            ):
                raise ValueError(f"native replay schedule reference drifted: {state_id}")
            schedule_rows[state_id] = row
    if (
        set(schedule_rows) != set(exact_records)
        or len(schedule_rows) != EXPECTED_EXACT_STATE_COUNT
        or observed_coalitions
        != manifest.get("coalition_count_after_deduplication")
    ):
        raise ValueError("native replay schedule does not cover exact-track 320")
    bindings = {
        "config_sha256": manifest["config_sha256"],
        "heldout_result_sha256": heldout_result_sha,
        "inventory_sha256": manifest["inventory_sha256"],
        "schedule_manifest_sha256": manifest_sha,
        "selections_sha256": selections_sha,
    }
    return manifest, schedule_rows, {
        "bindings": bindings,
        "heldout_result": heldout_result,
        "winner_model": winner_model,
    }


def load_native_replay_terminals(
    terminal_roots: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if not terminal_roots:
        raise ValueError("at least one native replay terminal root is required")
    terminals = {}
    hashes = {}
    for root in terminal_roots:
        states_root = root / "states"
        if not states_root.is_dir():
            raise FileNotFoundError(states_root)
        for path in sorted(states_root.glob("*.json")):
            terminal = _read_json(path)
            state_id = terminal.get("state_id")
            if not isinstance(state_id, str) or state_id in terminals:
                raise ValueError("native replay terminal identity is missing or duplicated")
            terminals[state_id] = terminal
            hashes[state_id] = sha256_file(path)
    return terminals, hashes


def _validate_terminal(
    terminal: Mapping[str, Any], *, schedule: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[float, ...]]]:
    state_id = schedule["state_id"]
    if (
        terminal.get("status") != COMPLETED_REPLAY_STATE
        or terminal.get("state_id") != state_id
        or terminal.get("trajectory_id") != schedule["trajectory_id"]
        or terminal.get("candidate_event_step_ids")
        != schedule["candidate_event_ids"]
        or terminal.get("role") != "evaluation"
        or terminal.get("winner_model") != schedule["winner_model"]
        or terminal.get("reference") != schedule["reference"]
        or not _is_sha256(terminal.get("state_identity_sha256"))
    ):
        raise ValueError(f"native replay terminal binding drifted: {state_id}")
    records = terminal.get("records")
    coalitions = schedule["coalitions"]
    if not isinstance(records, Sequence) or len(records) != len(coalitions):
        raise ValueError(f"native replay terminal coalition coverage drifted: {state_id}")
    reference_action = parse_gui_owl_v2_1_output(
        schedule["reference"]["serialized_action"]
    ).canonical_action
    applicability = compare_action_components(reference_action, reference_action)
    by_source = {}
    unique_latencies = []
    winner_unique_latencies = []
    for ordinal, (record, coalition) in enumerate(zip(records, coalitions, strict=True)):
        if not isinstance(record, Mapping):
            raise ValueError(f"native replay record is invalid: {state_id}")
        identity = hashlib.sha256(
            canonical_json_bytes(
                {
                    "coalition": coalition,
                    "ordinal": ordinal,
                    "state_identity_sha256": terminal["state_identity_sha256"],
                }
            )
        ).hexdigest()
        if (
            record.get("coalition_identity_sha256") != identity
            or record.get("coalition_event_step_ids") != coalition["event_ids"]
            or record.get("sources") != coalition["sources"]
        ):
            raise ValueError(f"native replay coalition binding drifted: {state_id}")
        metadata = record.get("generation_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"native replay generation metadata is missing: {state_id}")
        latency_ms = 1000.0 * _finite(
            metadata.get("latency_seconds"), label=f"{state_id} generation latency"
        )
        unique_latencies.append(latency_ms)
        if any(source.startswith("winner:B") for source in coalition["sources"]):
            winner_unique_latencies.append(latency_ms)
        status = record.get("status")
        parsed = status == COMPLETED_REPLAY_COALITION
        comparison = None
        native_output = record.get("native_output")
        if not isinstance(native_output, str):
            raise ValueError(f"native replay output is missing: {state_id}")
        if parsed:
            prediction = parse_gui_owl_v2_1_output(native_output).canonical_action
            comparison = compare_action_components(reference_action, prediction)
            if (
                prediction.arguments() != record.get("predicted_action")
                or comparison != record.get("comparison")
            ):
                raise ValueError(f"native replay action comparison drifted: {state_id}")
        elif status == FAILED_REPLAY_PARSE:
            try:
                parse_gui_owl_v2_1_output(native_output)
            except ValueError:
                pass
            else:
                raise ValueError(f"native replay parse failure now parses: {state_id}")
        else:
            raise ValueError(f"native replay coalition is incomplete: {state_id}")
        outcome = {
            "action_type_exact": bool(parsed and comparison["action_type_exact"]),
            "canonical_action_exact": bool(
                parsed and comparison["canonical_action_exact"]
            ),
            "generation_latency_ms": latency_ms,
            "parse_coverage": parsed,
            "target_applicable": applicability["target_applicable"],
            "target_exact": (
                bool(parsed and comparison["target_exact"])
                if applicability["target_applicable"]
                else None
            ),
            "text_applicable": applicability["text_applicable"],
            "text_nfkc_exact": (
                bool(parsed and comparison["text_nfkc_exact"])
                if applicability["text_applicable"]
                else None
            ),
        }
        for source in coalition["sources"]:
            if source in by_source:
                raise ValueError(f"native replay source tag is duplicated: {state_id}")
            by_source[source] = outcome
    expected_sources = {
        f"{method}:B{budget}" for method in NATIVE_REPLAY_METHODS for budget in _BUDGETS
    }
    if set(by_source) != expected_sources:
        raise ValueError(f"native replay source tags are incomplete: {state_id}")
    return by_source, {
        "all": tuple(unique_latencies),
        "winner": tuple(winner_unique_latencies),
    }


def _metric_summary(
    state_records: Sequence[Mapping[str, Any]],
    *,
    method: str,
    budget: int,
    metric: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    numerator = 0
    denominator = 0
    for row in state_records:
        outcome = row["methods"][method][str(budget)]
        value = outcome[metric]
        if value is None:
            continue
        denominator += 1
        numerator += int(value)
        grouped[row["trajectory_id"]].append(float(value))
    if denominator == 0:
        return {
            "denominator": 0,
            "micro_rate": None,
            "numerator": 0,
            "trajectory_count": 0,
            "trajectory_equal_rate": None,
        }, {}
    trajectory_values = {
        trajectory_id: fmean(values)
        for trajectory_id, values in sorted(grouped.items())
    }
    return {
        "denominator": denominator,
        "micro_rate": numerator / denominator,
        "numerator": numerator,
        "trajectory_count": len(trajectory_values),
        "trajectory_equal_rate": fmean(trajectory_values.values()),
    }, trajectory_values


def _macro_trajectory_values(
    state_records: Sequence[Mapping[str, Any]], *, method: str, metric: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in state_records:
        for budget in _BUDGETS:
            value = row["methods"][method][str(budget)][metric]
            if value is not None:
                grouped[row["trajectory_id"]].append(float(value))
    return {
        trajectory_id: fmean(values)
        for trajectory_id, values in sorted(grouped.items())
        if values
    }


def _budget_trajectory_values(
    state_records: Sequence[Mapping[str, Any]],
    *,
    method: str,
    budget: int,
    metric: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in state_records:
        value = row["methods"][method][str(budget)][metric]
        if value is not None:
            grouped[row["trajectory_id"]].append(float(value))
    return {
        trajectory_id: fmean(values)
        for trajectory_id, values in sorted(grouped.items())
        if values
    }


def _paired_metric_result(
    winner_values: Mapping[str, float], baseline_values: Mapping[str, float]
) -> dict[str, Any]:
    if set(winner_values) != set(baseline_values):
        raise ValueError("native replay behavior trajectory inventories drifted")
    judgable = bool(winner_values)
    delta = (
        fmean(
            winner_values[item] - baseline_values[item] for item in winner_values
        )
        if judgable
        else None
    )
    return {
        "baseline_trajectory_equal_rate": (
            fmean(baseline_values.values()) if judgable else None
        ),
        "judgable": judgable,
        "noninferior_at_zero_margin": bool(
            judgable and delta is not None and delta >= 0.0
        ),
        "paired_trajectory_count": len(winner_values),
        "trajectory_equal_delta": delta,
        "winner_trajectory_equal_rate": (
            fmean(winner_values.values()) if judgable else None
        ),
    }


def _behavior_recovery_gate(
    state_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    macro_values = {
        method: {
            metric: _macro_trajectory_values(
                state_records, method=method, metric=metric
            )
            for metric in _RATE_METRICS
        }
        for method in NATIVE_REPLAY_METHODS
    }
    comparisons = {}
    all_metrics_judgable = True
    all_metrics_noninferior = True
    primary_significant = True
    for baseline in ("recent", "ocr_rgb"):
        metric_results = {}
        for metric in _BEHAVIOR_METRICS:
            winner_values = macro_values["winner"][metric]
            baseline_values = macro_values[baseline][metric]
            metric_results[metric] = _paired_metric_result(
                winner_values, baseline_values
            )
        by_budget = {}
        for budget in _BUDGETS:
            by_budget[str(budget)] = {}
            for metric in _BEHAVIOR_METRICS:
                budget_result = _paired_metric_result(
                    _budget_trajectory_values(
                        state_records,
                        method="winner",
                        budget=budget,
                        metric=metric,
                    ),
                    _budget_trajectory_values(
                        state_records,
                        method=baseline,
                        budget=budget,
                        metric=metric,
                    ),
                )
                by_budget[str(budget)][metric] = budget_result
                all_metrics_judgable = (
                    all_metrics_judgable and budget_result["judgable"]
                )
                all_metrics_noninferior = (
                    all_metrics_noninferior
                    and budget_result["noninferior_at_zero_margin"]
                )
        primary_deltas = {
            trajectory_id: (
                macro_values["winner"][BEHAVIOR_PRIMARY_METRIC][trajectory_id]
                - macro_values[baseline][BEHAVIOR_PRIMARY_METRIC][trajectory_id]
            )
            for trajectory_id in macro_values["winner"][BEHAVIOR_PRIMARY_METRIC]
        }
        bootstrap = paired_trajectory_bootstrap(
            primary_deltas,
            resamples=BEHAVIOR_BOOTSTRAP_RESAMPLES,
            seed=BEHAVIOR_BOOTSTRAP_SEED,
            interval=BEHAVIOR_BOOTSTRAP_INTERVAL,
        )
        bootstrap["lower_strictly_positive"] = bootstrap["lower"] > 0.0
        primary_significant = (
            primary_significant and bootstrap["lower_strictly_positive"]
        )
        comparisons[f"winner_minus_{baseline}"] = {
            "by_budget": by_budget,
            "macro_B1_B4": metric_results,
            "primary_paired_trajectory_bootstrap": bootstrap,
        }
    requirements = {
        "all_B1_B4_action_metrics_judgable": all_metrics_judgable,
        "all_B1_B4_action_metric_deltas_nonnegative_vs_recent_and_ocr_rgb": (
            all_metrics_noninferior
        ),
        "complete_exact_track_320": len(state_records) == EXPECTED_EXACT_STATE_COUNT,
        "primary_bootstrap_lower_strictly_positive_vs_recent_and_ocr_rgb": (
            primary_significant
        ),
    }
    verdict = "GO" if all(requirements.values()) else "NO_GO"
    return {
        "bootstrap": {
            "confidence": BEHAVIOR_BOOTSTRAP_INTERVAL,
            "resamples": BEHAVIOR_BOOTSTRAP_RESAMPLES,
            "seed": BEHAVIOR_BOOTSTRAP_SEED,
            "unit": "trajectory",
        },
        "budgets": list(_BUDGETS),
        "comparisons": comparisons,
        "noninferiority_margin": 0.0,
        "primary_metric": "macro_B1_B4_trajectory_equal_canonical_action_exact",
        "profile": "native_behavior_recovery_v1",
        "requirements": requirements,
        "verdict": verdict,
    }, macro_values


def _complete_evaluation(
    *,
    schedules: Mapping[str, Mapping[str, Any]],
    terminals: Mapping[str, Mapping[str, Any]],
    terminal_hashes: Mapping[str, str],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    state_records = []
    all_unique_latencies = []
    winner_unique_latencies = []
    for state_id in sorted(schedules):
        schedule = schedules[state_id]
        by_source, unique_latencies = _validate_terminal(
            terminals[state_id], schedule=schedule
        )
        all_unique_latencies.extend(unique_latencies["all"])
        winner_unique_latencies.extend(unique_latencies["winner"])
        methods = {}
        for method in NATIVE_REPLAY_METHODS:
            methods[method] = {
                str(budget): by_source[f"{method}:B{budget}"] for budget in _BUDGETS
            }
        state_records.append(
            {
                "methods": methods,
                "state_id": state_id,
                "trajectory_id": schedule["trajectory_id"],
            }
        )
    summaries = {}
    trajectory_values = {}
    for method in NATIVE_REPLAY_METHODS:
        summaries[method] = {"by_budget": {}}
        trajectory_values[method] = {}
        for budget in _BUDGETS:
            metrics = {}
            trajectory_values[method][str(budget)] = {}
            for metric in _RATE_METRICS:
                metrics[metric], values = _metric_summary(
                    state_records, method=method, budget=budget, metric=metric
                )
                trajectory_values[method][str(budget)][metric] = values
            latencies = [
                row["methods"][method][str(budget)]["generation_latency_ms"]
                for row in state_records
            ]
            metrics["generation_latency_ms"] = {
                "attempt_count": len(latencies),
                "p50_ms": percentile_type7(latencies, 0.5),
                "p95_ms": percentile_type7(latencies, 0.95),
            }
            summaries[method]["by_budget"][str(budget)] = metrics
    behavior_gate, macro_values = _behavior_recovery_gate(state_records)
    for method in NATIVE_REPLAY_METHODS:
        summaries[method]["macro_B1_B4_trajectory_equal"] = {
            metric: {
                "trajectory_count": len(macro_values[method][metric]),
                "trajectory_equal_rate": (
                    fmean(macro_values[method][metric].values())
                    if macro_values[method][metric]
                    else None
                ),
            }
            for metric in _RATE_METRICS
        }
    comparisons = {}
    for baseline in ("recent", "ocr_rgb"):
        by_budget = {}
        for budget in _BUDGETS:
            metrics = {}
            for metric in _RATE_METRICS:
                winner_values = trajectory_values["winner"][str(budget)][metric]
                baseline_values = trajectory_values[baseline][str(budget)][metric]
                if set(winner_values) != set(baseline_values):
                    raise ValueError("native replay paired trajectory inventories drifted")
                metrics[metric] = {
                    "paired_trajectory_count": len(winner_values),
                    "trajectory_equal_delta": (
                        fmean(
                            winner_values[item] - baseline_values[item]
                            for item in winner_values
                        )
                        if winner_values
                        else None
                    ),
                }
            by_budget[str(budget)] = metrics
        comparisons[f"winner_minus_{baseline}"] = {"by_budget": by_budget}
    replay_latency = {
        "attempt_count": len(all_unique_latencies),
        "p50_ms": percentile_type7(all_unique_latencies, 0.5),
        "p95_ms": percentile_type7(all_unique_latencies, 0.95),
        "sample_unit": "deduplicated_all_method_native_generation",
    }
    policy_latency = {
        "attempt_count": len(winner_unique_latencies),
        "p50_ms": percentile_type7(winner_unique_latencies, 0.5),
        "p95_ms": percentile_type7(winner_unique_latencies, 0.95),
        "sample_unit": "deduplicated_winner_native_generation",
    }
    winner_model = context["winner_model"]
    heldout_result = context["heldout_result"]
    try:
        selector_p95_value = heldout_result["method_summaries"][winner_model][
            "latency"
        ]["selector_total_from_cached_source_tokens"]["p95_ms"]
    except (KeyError, TypeError) as error:
        raise ValueError("held-out winner omits warm selector p95") from error
    selector_p95 = _finite(selector_p95_value, label="held-out warm selector p95")
    action_p95 = _finite(
        policy_latency["p95_ms"], label="native full-policy generation p95", positive=True
    )
    ratio = selector_p95 / action_p95
    latency_gate = {
        "action_policy_generation_p95_ms": action_p95,
        "action_policy_generation_sample_unit": policy_latency["sample_unit"],
        "maximum_ratio": DEPLOYMENT_LATENCY_RATIO_MAXIMUM,
        "profile": "warm_shared_encoder",
        "ratio": ratio,
        "selector_p95_ms": selector_p95,
        "verdict": "GO" if ratio <= DEPLOYMENT_LATENCY_RATIO_MAXIMUM else "NO_GO",
    }
    closed_loop_authorized = (
        behavior_gate["verdict"] == "GO" and latency_gate["verdict"] == "GO"
    )
    closed_loop_authorization = {
        "authorized": closed_loop_authorized,
        "requires": [
            "behavior_recovery_gate.GO",
            "deployment_latency_gate.GO",
        ],
        "verdict": "GO" if closed_loop_authorized else "NO_GO",
    }
    result = {
        "bindings": {
            **context["bindings"],
            "terminal_inventory_sha256": _sha256_text(
                dict(sorted(terminal_hashes.items()))
            ),
        },
        "behavior_recovery_gate": behavior_gate,
        "closed_loop_authorization": closed_loop_authorization,
        "comparisons": comparisons,
        "coverage": {
            "complete": True,
            "completed_state_count": EXPECTED_EXACT_STATE_COUNT,
            "expected_state_count": EXPECTED_EXACT_STATE_COUNT,
            "extra_state_ids": [],
            "missing_state_ids": [],
        },
        "deployment_latency_gate": latency_gate,
        "all_native_replay_generation_latency_ms": replay_latency,
        "method_summaries": summaries,
        "native_full_policy_generation_latency_ms": policy_latency,
        "schema_version": "1.0.0",
        "state_records": state_records,
        "status": COMPLETED_NATIVE_REPLAY_EVALUATION,
        "winner_model": winner_model,
    }
    result["content_sha256"] = _sha256_text(result)
    return result


def evaluate_native_replay(
    *,
    schedule_root: Path,
    terminal_roots: Sequence[Path],
    selections_path: Path,
    heldout_result_path: Path,
) -> dict[str, Any]:
    _, schedules, context = load_and_validate_native_replay_schedule(
        schedule_root=schedule_root,
        selections_path=selections_path,
        heldout_result_path=heldout_result_path,
    )
    terminals, terminal_hashes = load_native_replay_terminals(terminal_roots)
    expected = set(schedules)
    observed = set(terminals)
    extra = sorted(observed - expected)
    if extra:
        raise ValueError("native replay terminal roots contain states outside the schedule")
    missing = sorted(expected - observed)
    if missing:
        for state_id in sorted(observed):
            _validate_terminal(terminals[state_id], schedule=schedules[state_id])
        result = {
            "bindings": {
                **context["bindings"],
                "terminal_inventory_sha256": _sha256_text(
                    dict(sorted(terminal_hashes.items()))
                ),
            },
            "behavior_recovery_gate": {
                "bootstrap": {
                    "confidence": BEHAVIOR_BOOTSTRAP_INTERVAL,
                    "resamples": BEHAVIOR_BOOTSTRAP_RESAMPLES,
                    "seed": BEHAVIOR_BOOTSTRAP_SEED,
                    "unit": "trajectory",
                },
                "budgets": list(_BUDGETS),
                "comparisons": {},
                "noninferiority_margin": 0.0,
                "primary_metric": (
                    "macro_B1_B4_trajectory_equal_canonical_action_exact"
                ),
                "profile": "native_behavior_recovery_v1",
                "reason": "incomplete_exact_track_coverage",
                "requirements": {
                    "all_B1_B4_action_metrics_judgable": False,
                    "all_B1_B4_action_metric_deltas_nonnegative_vs_recent_and_ocr_rgb": False,
                    "complete_exact_track_320": False,
                    "primary_bootstrap_lower_strictly_positive_vs_recent_and_ocr_rgb": False,
                },
                "verdict": "NO_GO",
            },
            "closed_loop_authorization": {
                "authorized": False,
                "reason": "incomplete_exact_track_coverage",
                "requires": [
                    "behavior_recovery_gate.GO",
                    "deployment_latency_gate.GO",
                ],
                "verdict": "NO_GO",
            },
            "comparisons": {},
            "coverage": {
                "complete": False,
                "completed_state_count": len(observed),
                "expected_state_count": EXPECTED_EXACT_STATE_COUNT,
                "extra_state_ids": [],
                "missing_state_ids": missing,
            },
            "deployment_latency_gate": {
                "maximum_ratio": DEPLOYMENT_LATENCY_RATIO_MAXIMUM,
                "reason": "incomplete_exact_track_coverage",
                "verdict": "NO_GO",
            },
            "all_native_replay_generation_latency_ms": None,
            "method_summaries": {},
            "native_full_policy_generation_latency_ms": None,
            "schema_version": "1.0.0",
            "state_records": [],
            "status": INCOMPLETE_NATIVE_REPLAY_EVALUATION,
            "winner_model": context["winner_model"],
        }
        result["content_sha256"] = _sha256_text(result)
        return result
    return _complete_evaluation(
        schedules=schedules,
        terminals=terminals,
        terminal_hashes=terminal_hashes,
        context=context,
    )


__all__ = [
    "COMPLETED_NATIVE_REPLAY_EVALUATION",
    "DEPLOYMENT_LATENCY_RATIO_MAXIMUM",
    "INCOMPLETE_NATIVE_REPLAY_EVALUATION",
    "evaluate_native_replay",
    "load_and_validate_native_replay_schedule",
    "load_native_replay_terminals",
]
