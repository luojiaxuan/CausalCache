"""Frozen budget-deferral selection and sealing for evaluation-role states."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.set_utility_direct_on_policy_evaluator import (
    project_inference_state,
    recent_budget_path,
)
from causalcache.set_utility_train_heldout_contract import sha256_json
from causalcache.set_utility_variable_history import history_bin


SELECTION_SCHEMA = "causalcache.budget_deferral_eval_selections.v1"
SELECTION_STATUS = "SEALED_BUDGET_DEFERRAL_EVALUATION_SELECTIONS"
SEAL_SCHEMA = "causalcache.budget_deferral_eval_selection_seal.v1"
SEAL_STATUS = "SEALED_BUDGET_DEFERRAL_EVALUATION_SELECTION_RECEIPT"
METHODS = ("recent", "structured_deepsets_budget_deferral")
SLICES = ("all", "state_new", "trajectory_new")
FORBIDDEN_STATE_KEYS = frozenset(
    {
        "distance_rows",
        "labels",
        "restoration_distances",
        "restoration_truth",
        "target",
        "target_utility",
        "truth",
        "utility_target",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_budget_deferral_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the one candidate frozen before evaluation access."""
    if (
        config.get("schema_version") != "1.0.0"
        or config.get("status") != "FROZEN_BEFORE_CANDIDATE_EVALUATION"
    ):
        raise ValueError("budget-deferral evaluation config is not frozen")
    predictor = config.get("predictor")
    deployment = config.get("deployment")
    evaluation = config.get("frozen_candidate_evaluation")
    firewall = config.get("firewall")
    if not all(
        isinstance(value, Mapping)
        for value in (predictor, deployment, evaluation, firewall)
    ):
        raise ValueError("budget-deferral config sections are incomplete")
    if (
        predictor.get("model_family") != "deepsets_structured_marginal"
        or predictor.get("stop_semantics") != "learned_threshold"
        or type(predictor.get("selected_epoch")) is not int
        or int(predictor["selected_epoch"]) <= 0
    ):
        raise ValueError("budget-deferral predictor identity drifted")
    _sha256(predictor.get("checkpoint_sha256"), label="predictor checkpoint SHA256")
    if tuple(deployment.get("budgets", ())) != (1, 2, 3, 4):
        raise ValueError("budget-deferral budgets drifted")
    if deployment.get("selection_by_budget") != {
        "1": "recent",
        "2": "recent",
        "3": "deepsets_direct_conditional_marginal",
        "4": "deepsets_direct_conditional_marginal",
    }:
        raise ValueError("budget-deferral route drifted")
    if (
        deployment.get("selection_cardinality") != "at_most_B"
        or deployment.get("cross_budget_nestedness_required") is not False
        or deployment.get("within_budget_iterative_selection") is not True
        or deployment.get("post_evaluation_route_or_threshold_changes_allowed")
        is not False
    ):
        raise ValueError("budget-deferral deployment semantics drifted")
    if (
        evaluation.get("role") != "evaluation"
        or evaluation.get("label_blind_selection_must_be_sealed_before_truth_access")
        is not True
        or firewall.get("evaluation_features_allowed_after_this_freeze") is not True
        or firewall.get("evaluation_truth_allowed_only_after_signed_selection_seal")
        is not True
        or firewall.get("evaluation_truth_for_training_or_calibration") is not False
    ):
        raise ValueError("budget-deferral evaluation firewall drifted")
    _positive_int(evaluation.get("state_count"), label="evaluation state count")
    _sha256(
        evaluation.get("state_inventory_sha256"),
        label="evaluation state inventory SHA256",
    )
    boundary = evaluation.get("historical_access_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("historical access boundary is absent")
    for key in (
        "previously_consumed_state_count",
        "previously_consumed_trajectory_count",
        "state_new_count",
        "trajectory_new_state_count",
        "trajectory_new_trajectory_count",
    ):
        _positive_int(boundary.get(key), label=key)
    return dict(config)


def project_label_blind_evaluation_state(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project only deployable features and reject any embedded target field."""
    forbidden = FORBIDDEN_STATE_KEYS.intersection(row)
    if forbidden:
        raise ValueError(
            "evaluation state contains forbidden truth fields: "
            + ",".join(sorted(forbidden))
        )
    if row.get("role") != "evaluation":
        raise ValueError("frozen selector received a non-evaluation state")
    state = project_inference_state(row)
    if not state["state_id"].startswith(
        f"{state['trajectory_id']}:decision:"
    ):
        raise ValueError("evaluation state/trajectory identity drifted")
    logical_shard = row.get("logical_shard")
    if type(logical_shard) is not int or not 0 <= logical_shard < 256:
        raise ValueError("evaluation state logical shard drifted")
    observed_bin = history_bin(len(state["candidate_event_step_ids"]))
    declared_bin = row.get("history_bin", observed_bin)
    if declared_bin != observed_bin:
        raise ValueError("evaluation state history bin drifted")
    state["history_bin"] = observed_bin
    state["logical_shard"] = logical_shard
    return state


def evaluation_slice_memberships(
    states: Sequence[Mapping[str, Any]],
    *,
    inventory: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, int]]]:
    """Derive all/state-new/trajectory-new without consulting evaluation truth."""
    frozen = validate_budget_deferral_config(config)
    evaluation = frozen["frozen_candidate_evaluation"]
    if (
        inventory.get("status") != "FROZEN_VARIABLE_HISTORY_STATE_INVENTORY"
        or inventory.get("state_identity_sha256")
        != evaluation["state_inventory_sha256"]
    ):
        raise ValueError("evaluation state inventory identity drifted")
    tracks = inventory.get("evaluation_tracks")
    if not isinstance(tracks, Mapping):
        raise ValueError("evaluation state inventory omits historical tracks")
    exact = tracks.get("exact_state_ids")
    large = tracks.get("large_history_state_ids")
    if (
        not isinstance(exact, Sequence)
        or isinstance(exact, (str, bytes, bytearray))
        or not isinstance(large, Sequence)
        or isinstance(large, (str, bytes, bytearray))
    ):
        raise ValueError("historical evaluation tracks are invalid")
    consumed_state_ids = set(exact) | set(large)
    if len(consumed_state_ids) != int(
        evaluation["historical_access_boundary"]["previously_consumed_state_count"]
    ):
        raise ValueError("historically consumed state count drifted")
    by_state = {str(state["state_id"]): state for state in states}
    if len(by_state) != len(states) or set(consumed_state_ids) - set(by_state):
        raise ValueError("evaluation state denominator is duplicate or incomplete")
    consumed_trajectories = {
        str(by_state[state_id]["trajectory_id"]) for state_id in consumed_state_ids
    }
    boundary = evaluation["historical_access_boundary"]
    if len(consumed_trajectories) != int(
        boundary["previously_consumed_trajectory_count"]
    ):
        raise ValueError("historically consumed trajectory count drifted")
    memberships: dict[str, tuple[str, ...]] = {}
    for state_id, state in by_state.items():
        names = ["all"]
        if state_id not in consumed_state_ids:
            names.append("state_new")
        if str(state["trajectory_id"]) not in consumed_trajectories:
            names.append("trajectory_new")
        memberships[state_id] = tuple(names)
    slice_counts = {}
    for name in SLICES:
        selected = [
            by_state[state_id]
            for state_id, names in memberships.items()
            if name in names
        ]
        slice_counts[name] = {
            "state_count": len(selected),
            "trajectory_count": len(
                {str(state["trajectory_id"]) for state in selected}
            ),
        }
    if (
        slice_counts["all"]["state_count"] != int(evaluation["state_count"])
        or slice_counts["state_new"]["state_count"]
        != int(boundary["state_new_count"])
        or slice_counts["trajectory_new"]["state_count"]
        != int(boundary["trajectory_new_state_count"])
        or slice_counts["trajectory_new"]["trajectory_count"]
        != int(boundary["trajectory_new_trajectory_count"])
    ):
        raise ValueError("evaluation slice census drifted")
    return memberships, slice_counts


def budget_deferral_methods(
    event_ids: Sequence[int],
    *,
    learned_selections: Mapping[str, Sequence[int]],
) -> dict[str, dict[str, list[int]]]:
    """Apply the frozen B1/B2 recent and B3/B4 direct-DeepSets route."""
    candidates = tuple(event_ids)
    recent = recent_budget_path(candidates)
    learned = {}
    for budget in range(1, 5):
        key = str(budget)
        raw = learned_selections.get(key)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise ValueError("learned selector budget path is incomplete")
        subset = tuple(raw)
        if (
            subset != tuple(sorted(subset))
            or len(subset) != len(set(subset))
            or len(subset) > budget
            or any(event_id not in candidates for event_id in subset)
        ):
            raise ValueError("learned selector emitted an invalid at-most-B subset")
        learned[key] = list(subset)
    deferred = {
        "1": list(recent["1"]),
        "2": list(recent["2"]),
        "3": learned["3"],
        "4": learned["4"],
    }
    return {
        "recent": recent,
        "structured_deepsets_budget_deferral": deferred,
    }


def build_selection_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    bindings: Mapping[str, Any],
    slice_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Build the portable signed selection payload consumed by truth scheduling."""
    required_bindings = {
        "cache_content_sha256",
        "cache_manifest_file_sha256",
        "checkpoint_sha256",
        "config_content_sha256",
        "config_file_sha256",
        "input_content_sha256",
        "input_manifest_file_sha256",
        "input_states_sha256",
        "predictor_config_file_sha256",
        "state_inventory_content_sha256",
        "state_inventory_file_sha256",
    }
    if set(bindings) != required_bindings:
        raise ValueError("evaluation selection bindings are incomplete")
    for key, value in bindings.items():
        _sha256(value, label=key)
    normalized = []
    seen = set()
    history_counts: Counter[str] = Counter()
    for raw in records:
        if set(raw) != {
            "candidate_event_ids",
            "history_bin",
            "latency_ms",
            "logical_shard",
            "methods",
            "slices",
            "state_id",
            "trajectory_id",
        }:
            raise ValueError("evaluation selection record schema drifted")
        state_id = raw["state_id"]
        trajectory_id = raw["trajectory_id"]
        logical_shard = raw["logical_shard"]
        if (
            not isinstance(state_id, str)
            or not state_id
            or state_id in seen
            or not isinstance(trajectory_id, str)
            or not trajectory_id
            or not state_id.startswith(f"{trajectory_id}:decision:")
            or type(logical_shard) is not int
            or not 0 <= logical_shard < 256
        ):
            raise ValueError("evaluation selection record identity drifted")
        seen.add(state_id)
        candidates = tuple(raw["candidate_event_ids"])
        if (
            not candidates
            or candidates != tuple(sorted(candidates))
            or len(candidates) != len(set(candidates))
            or any(type(value) is not int or value <= 0 for value in candidates)
        ):
            raise ValueError("evaluation selection candidate universe is invalid")
        observed_bin = history_bin(len(candidates))
        if raw["history_bin"] != observed_bin:
            raise ValueError("evaluation selection history bin drifted")
        history_counts[observed_bin] += 1
        slices = tuple(raw["slices"])
        if slices not in {
            ("all",),
            ("all", "state_new"),
            ("all", "state_new", "trajectory_new"),
        }:
            raise ValueError("evaluation selection slice membership drifted")
        if not isinstance(raw["methods"], Mapping) or set(raw["methods"]) != set(
            METHODS
        ):
            raise ValueError("evaluation selection method inventory drifted")
        for method in METHODS:
            expected = budget_deferral_methods(
                candidates,
                learned_selections=(
                    raw["methods"][method]
                    if method == "structured_deepsets_budget_deferral"
                    else raw["methods"]["recent"]
                ),
            )
            if method == "recent":
                if raw["methods"][method] != expected["recent"]:
                    raise ValueError("recent selection drifted")
            else:
                for budget in range(1, 5):
                    subset = raw["methods"][method].get(str(budget))
                    if (
                        not isinstance(subset, list)
                        or len(subset) > budget
                        or any(value not in candidates for value in subset)
                    ):
                        raise ValueError("budget-deferral selection is invalid")
                if (
                    raw["methods"][method]["1"]
                    != raw["methods"]["recent"]["1"]
                    or raw["methods"][method]["2"]
                    != raw["methods"]["recent"]["2"]
                ):
                    raise ValueError("budget-deferral recent budgets drifted")
        latency = raw["latency_ms"]
        if not isinstance(latency, Mapping) or not latency:
            raise ValueError("evaluation selection latency is absent")
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in latency.values()
        ):
            raise ValueError("evaluation selection latency is invalid")
        normalized.append(dict(raw))
    normalized.sort(key=lambda row: row["state_id"])
    denominator = {
        "history_bin_counts": dict(sorted(history_counts.items())),
        "slice_counts": {
            name: {
                "state_count": int(slice_counts[name]["state_count"]),
                "trajectory_count": int(slice_counts[name]["trajectory_count"]),
            }
            for name in SLICES
        },
        "state_count": len(normalized),
        "trajectory_count": len({row["trajectory_id"] for row in normalized}),
    }
    observed_slices = {
        name: {
            "state_count": sum(name in row["slices"] for row in normalized),
            "trajectory_count": len(
                {
                    row["trajectory_id"]
                    for row in normalized
                    if name in row["slices"]
                }
            ),
        }
        for name in SLICES
    }
    if denominator["slice_counts"] != observed_slices:
        raise ValueError("selection record slice census disagrees with denominator")
    unsigned = {
        "bindings": dict(bindings),
        "denominator": denominator,
        "method_semantics": {
            "recent": "deterministic_most_recent_at_most_B",
            "structured_deepsets_budget_deferral": (
                "B1_B2_recent_B3_B4_direct_conditional_marginal"
            ),
        },
        "methods": list(METHODS),
        "records": normalized,
        "schema_version": SELECTION_SCHEMA,
        "status": SELECTION_STATUS,
        "truth_accessed": False,
    }
    return {**unsigned, "content_sha256": sha256_json(unsigned)}


def write_selection_and_seal(
    selection: Mapping[str, Any],
    *,
    selection_path: Path,
    seal_path: Path,
) -> dict[str, Any]:
    """Atomically publish selections, then bind their exact file bytes in a seal."""
    if selection_path.exists() or seal_path.exists():
        raise FileExistsError("evaluation selection or seal already exists")
    unsigned = dict(selection)
    claimed = unsigned.pop("content_sha256", None)
    if (
        selection.get("schema_version") != SELECTION_SCHEMA
        or selection.get("status") != SELECTION_STATUS
        or selection.get("truth_accessed") is not False
        or claimed != sha256_json(unsigned)
    ):
        raise ValueError("evaluation selection content signature drifted")
    _write_atomic(selection_path, selection)
    seal_unsigned = {
        "bindings": selection["bindings"],
        "denominator": selection["denominator"],
        "schema_version": SEAL_SCHEMA,
        "selection_content_sha256": claimed,
        "selection_file_sha256": sha256_file(selection_path),
        "status": SEAL_STATUS,
        "truth_accessed": False,
    }
    seal = {**seal_unsigned, "content_sha256": sha256_json(seal_unsigned)}
    _write_atomic(seal_path, seal)
    return seal


def verify_selection_and_seal(
    selection_path: Path, seal_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the only artifacts allowed to unlock evaluation truth scheduling."""
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or not isinstance(seal, dict):
        raise ValueError("selection and seal must contain JSON objects")
    selection_unsigned = dict(selection)
    selection_content = selection_unsigned.pop("content_sha256", None)
    seal_unsigned = dict(seal)
    seal_content = seal_unsigned.pop("content_sha256", None)
    rebuilt = build_selection_payload(
        selection.get("records", ()),
        bindings=selection.get("bindings", {}),
        slice_counts=selection.get("denominator", {}).get("slice_counts", {}),
    )
    if (
        rebuilt != selection
        or selection.get("schema_version") != SELECTION_SCHEMA
        or selection.get("status") != SELECTION_STATUS
        or selection.get("truth_accessed") is not False
        or selection_content != sha256_json(selection_unsigned)
        or seal.get("schema_version") != SEAL_SCHEMA
        or seal.get("status") != SEAL_STATUS
        or seal.get("truth_accessed") is not False
        or seal_content != sha256_json(seal_unsigned)
        or seal.get("selection_content_sha256") != selection_content
        or seal.get("selection_file_sha256") != sha256_file(selection_path)
        or seal.get("bindings") != selection.get("bindings")
        or seal.get("denominator") != selection.get("denominator")
        or set(seal)
        != {
            "bindings",
            "content_sha256",
            "denominator",
            "schema_version",
            "selection_content_sha256",
            "selection_file_sha256",
            "status",
            "truth_accessed",
        }
    ):
        raise ValueError("evaluation selection seal drifted")
    return selection, seal


__all__ = [
    "FORBIDDEN_STATE_KEYS",
    "METHODS",
    "SEAL_SCHEMA",
    "SEAL_STATUS",
    "SELECTION_SCHEMA",
    "SELECTION_STATUS",
    "SLICES",
    "budget_deferral_methods",
    "build_selection_payload",
    "evaluation_slice_memberships",
    "project_label_blind_evaluation_state",
    "sha256_file",
    "validate_budget_deferral_config",
    "verify_selection_and_seal",
    "write_selection_and_seal",
]
