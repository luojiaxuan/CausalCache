"""Frozen train-heldout split and truth-based checkpoint-selection contract."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any

from causalcache.set_utility_variable_history import history_bin, states_from_assignments


HISTORY_BINS = ("short", "medium", "long", "very_long")
BUDGETS = (1, 2, 3, 4)
CONTRACT_STATUS = "FROZEN_SET_UTILITY_TRAIN_HELDOUT_CONTRACT"
COMPLETE_TRUTH_STATUS = "COMPLETED_SET_UTILITY_TRAIN_HELDOUT_EPOCH_TRUTH"
INCOMPLETE_TRUTH_STATUS = "INCOMPLETE_SET_UTILITY_TRAIN_HELDOUT_EPOCH_TRUTH"
INVENTORY_STATUS = "FROZEN_SET_UTILITY_TRAIN_INPUT_STATE_INVENTORY"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _config_fields(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    source = config.get("source")
    split = config.get("split")
    checkpoint = config.get("checkpoint_selection")
    firewall = config.get("firewall")
    if not all(isinstance(value, Mapping) for value in (source, split, checkpoint, firewall)):
        raise ValueError("train-heldout config sections are incomplete")
    if source.get("allowed_role") != "train":
        raise ValueError("train-heldout source role must be train")
    if firewall.get("tune_access") is not False or firewall.get("evaluation_access") is not False:
        raise ValueError("train-heldout firewall must forbid tune and evaluation access")
    if firewall.get("all_heldout_trajectory_states_excluded_from_optimizer") is not True:
        raise ValueError("heldout trajectories must be wholly excluded from optimization")
    expected = _positive_int(
        source.get("expected_train_trajectory_count"), "expected train trajectory count"
    )
    heldout = _positive_int(
        split.get("heldout_trajectory_count"), "heldout trajectory count"
    )
    if heldout >= expected:
        raise ValueError("heldout trajectory count must be below the train inventory")
    targets = split.get("checkpoint_state_targets_by_history_bin")
    if not isinstance(targets, Mapping) or set(targets) != set(HISTORY_BINS):
        raise ValueError("checkpoint state targets must cover all history bins")
    normalized_targets = {
        name: _positive_int(targets[name], f"{name} checkpoint state target")
        for name in HISTORY_BINS
    }
    salt = split.get("selection_salt")
    if not isinstance(salt, str) or not salt:
        raise ValueError("selection salt must be non-empty text")
    budgets = tuple(checkpoint.get("budgets", ()))
    if budgets != BUDGETS:
        raise ValueError("checkpoint-selection budgets must be exactly B1--B4")
    if checkpoint.get("require_complete_truth_each_epoch") is not True:
        raise ValueError("checkpoint selection must require complete per-epoch truth")
    if checkpoint.get("higher_is_better") is not True:
        raise ValueError("checkpoint selection must maximize true recovery")
    patience = _positive_int(checkpoint.get("patience"), "checkpoint patience")
    minimum_delta = _nonnegative_float(
        checkpoint.get("minimum_delta"), "checkpoint minimum delta"
    )
    normalized_checkpoint = {
        **dict(checkpoint),
        "patience": patience,
        "minimum_delta": minimum_delta,
    }
    if "tie_breaker_minimum_delta" in checkpoint:
        normalized_checkpoint["tie_breaker_minimum_delta"] = _nonnegative_float(
            checkpoint["tie_breaker_minimum_delta"],
            "checkpoint tie-breaker minimum delta",
        )
    return (
        dict(source),
        {
            **dict(split),
            "checkpoint_state_targets_by_history_bin": normalized_targets,
        },
        normalized_checkpoint,
        dict(firewall),
    )


def _order_digest(salt: str, namespace: str, identity: str) -> str:
    return hashlib.sha256(
        f"{salt}:{namespace}:{identity}".encode("utf-8")
    ).hexdigest()


def build_train_input_state_inventory(
    input_manifest: Mapping[str, Any],
    state_records: Iterable[Mapping[str, Any]],
    *,
    input_manifest_sha256: str,
) -> dict[str, Any]:
    """Export only train identities from one frozen structured-training input."""
    manifest_sha = _sha256(input_manifest_sha256, "training input manifest SHA256")
    if input_manifest.get("status") != "COMPLETED_SET_UTILITY_CONTEXTUAL_INPUT_SNAPSHOT":
        raise ValueError("training input snapshot status is invalid")
    input_content_sha = _sha256(
        input_manifest.get("content_sha256"), "training input content SHA256"
    )
    states_sha = _sha256(
        input_manifest.get("states_sha256"), "training states JSONL SHA256"
    )
    if input_manifest.get("evaluation_labels_included") is not False:
        raise ValueError("training input snapshot may not include evaluation labels")
    source_role_counts = input_manifest.get("role_counts")
    if not isinstance(source_role_counts, Mapping) or any(
        type(value) is not int or value < 0 for value in source_role_counts.values()
    ):
        raise ValueError("training input role census is invalid")
    expected_roles = {str(name): int(value) for name, value in source_role_counts.items()}
    observed_roles: Counter[str] = Counter()
    seen_state_ids: set[str] = set()
    train_rows: list[dict[str, Any]] = []
    for record in state_records:
        if not isinstance(record, Mapping):
            raise ValueError("training input state rows must be mappings")
        role = record.get("role")
        if role not in expected_roles:
            raise ValueError("training input state role escaped the manifest census")
        observed_roles[str(role)] += 1
        state_id = record.get("state_id")
        if not isinstance(state_id, str) or not state_id or state_id in seen_state_ids:
            raise ValueError("training input state identity is invalid or duplicated")
        seen_state_ids.add(state_id)
        if role != "train":
            continue
        trajectory_id = record.get("trajectory_id")
        candidates = record.get("candidate_event_step_ids")
        if (
            not isinstance(trajectory_id, str)
            or not trajectory_id
            or not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes, bytearray))
        ):
            raise ValueError("train state identity or candidate universe is invalid")
        try:
            prefix, decision_text = state_id.rsplit(":decision:", 1)
            decision_step = int(decision_text)
        except (TypeError, ValueError) as error:
            raise ValueError("train state identity is not a decision identity") from error
        candidate_ids = tuple(candidates)
        if (
            prefix != trajectory_id
            or state_id != f"{trajectory_id}:decision:{decision_step:03d}"
            or candidate_ids != tuple(range(1, decision_step))
        ):
            raise ValueError("train state is not complete prior-history context")
        train_rows.append(
            {
                "candidate_count": len(candidate_ids),
                "history_bin": history_bin(len(candidate_ids)),
                "state_id": state_id,
                "trajectory_id": trajectory_id,
            }
        )
    if dict(observed_roles) != expected_roles:
        raise ValueError("training input state rows differ from the manifest role census")
    if sum(observed_roles.values()) != input_manifest.get("state_count"):
        raise ValueError("training input state count differs from its manifest")
    train_rows.sort(key=lambda row: row["state_id"])
    train_state_ids = [row["state_id"] for row in train_rows]
    result = {
        "census": {
            "source_role_counts": expected_roles,
            "source_state_count": sum(observed_roles.values()),
            "train_history_bin_counts": dict(
                Counter(row["history_bin"] for row in train_rows)
            ),
            "train_state_count": len(train_rows),
            "train_trajectory_count": len(
                {row["trajectory_id"] for row in train_rows}
            ),
        },
        "schema_version": "1.0.0",
        "source": {
            "input_content_sha256": input_content_sha,
            "input_manifest_sha256": manifest_sha,
            "states_jsonl": input_manifest.get("states_jsonl"),
            "states_sha256": states_sha,
        },
        "status": INVENTORY_STATUS,
        "train_state_ids_sha256": sha256_json(train_state_ids),
        "train_states": train_rows,
    }
    result["content_sha256"] = sha256_json(result)
    return result


def materialize_train_heldout_contract(
    assignments: Sequence[Mapping[str, Any]],
    *,
    train_state_inventory: Mapping[str, Any],
    config: Mapping[str, Any],
    assignment_manifest_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Select whole heldout trajectories and a balanced checkpoint denominator."""
    source, split, checkpoint, firewall = _config_fields(config)
    source_sha = _sha256(assignment_manifest_sha256, "assignment manifest SHA256")
    bound_source_sha = _sha256(
        source.get("assignment_manifest_sha256"), "bound assignment manifest SHA256"
    )
    if source_sha != bound_source_sha:
        raise ValueError("assignment manifest drifted from the frozen config")
    if train_state_inventory.get("status") != INVENTORY_STATUS:
        raise ValueError("train state inventory status is invalid")
    inventory_unsigned = dict(train_state_inventory)
    inventory_claimed_sha = inventory_unsigned.pop("content_sha256", None)
    inventory_content_sha = _sha256(
        inventory_claimed_sha, "train state inventory content SHA256"
    )
    if sha256_json(inventory_unsigned) != inventory_content_sha:
        raise ValueError("train state inventory content hash is invalid")
    if inventory_content_sha != _sha256(
        source.get("train_state_inventory_content_sha256"),
        "bound train state inventory content SHA256",
    ):
        raise ValueError("train state inventory drifted from the frozen config")
    inventory_source = train_state_inventory.get("source")
    if not isinstance(inventory_source, Mapping):
        raise ValueError("train state inventory source binding is absent")
    for inventory_key, config_key, label in (
        ("input_content_sha256", "training_input_content_sha256", "input content"),
        ("input_manifest_sha256", "training_input_manifest_sha256", "input manifest"),
        ("states_sha256", "training_states_sha256", "training states"),
    ):
        observed_sha = _sha256(inventory_source.get(inventory_key), f"{label} SHA256")
        expected_sha = _sha256(source.get(config_key), f"bound {label} SHA256")
        if observed_sha != expected_sha:
            raise ValueError(f"{label} drifted from the frozen config")
    config_sha = _sha256(config_sha256, "config SHA256")
    if isinstance(assignments, (str, bytes, bytearray, Mapping)) or not isinstance(
        assignments, Sequence
    ):
        raise ValueError("assignments must be a sequence")

    train_assignments: dict[str, Mapping[str, Any]] = {}
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise ValueError("assignment rows must be mappings")
        if assignment.get("role") != "train":
            continue
        trajectory_id = assignment.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("train assignment trajectory id is invalid")
        if trajectory_id in train_assignments:
            raise ValueError("train assignment contains duplicate trajectories")
        train_assignments[trajectory_id] = assignment
    expected_train = int(source["expected_train_trajectory_count"])
    if len(train_assignments) != expected_train:
        raise ValueError("train trajectory inventory differs from the frozen count")

    assignment_states = tuple(
        state
        for state in states_from_assignments(tuple(train_assignments.values()))
        if state.role == "train"
    )
    expected_assignment_states = _positive_int(
        source.get("expected_assignment_derived_train_state_count"),
        "expected assignment-derived train state count",
    )
    if len(assignment_states) != expected_assignment_states:
        raise ValueError("assignment-derived train state inventory drifted")
    assignment_by_id = {state.state_id: state for state in assignment_states}
    inventory_rows = train_state_inventory.get("train_states")
    if not isinstance(inventory_rows, Sequence) or isinstance(
        inventory_rows, (str, bytes, bytearray)
    ):
        raise ValueError("train state inventory rows are absent")
    actual_by_id: dict[str, Any] = {}
    for row in inventory_rows:
        if not isinstance(row, Mapping):
            raise ValueError("train state inventory rows must be mappings")
        state_id = row.get("state_id")
        if not isinstance(state_id, str) or state_id in actual_by_id:
            raise ValueError("train state inventory identity is invalid or duplicated")
        state = assignment_by_id.get(state_id)
        if state is None:
            raise ValueError("training input contains a state outside frozen assignments")
        expected_row = {
            "candidate_count": len(state.candidate_event_ids),
            "history_bin": history_bin(len(state.candidate_event_ids)),
            "state_id": state.state_id,
            "trajectory_id": state.trajectory_id,
        }
        if dict(row) != expected_row:
            raise ValueError("training input state metadata differs from assignments")
        actual_by_id[state_id] = state
    expected_actual_states = _positive_int(
        source.get("expected_train_state_count"), "expected train state count"
    )
    if len(actual_by_id) != expected_actual_states:
        raise ValueError("training input train state inventory differs from frozen count")
    actual_ids = set(actual_by_id)
    assignment_only_ids = sorted(set(assignment_by_id) - actual_ids)
    expected_exclusion_count = _positive_int(
        source.get("expected_assignment_only_exclusion_count"),
        "expected assignment-only exclusion count",
    )
    if len(assignment_only_ids) != expected_exclusion_count:
        raise ValueError("assignment-only state exclusion census drifted")
    exclusions = [
        {
            "reason": "ABSENT_FROM_FROZEN_STRUCTURED_TRAINING_INPUT",
            "state_id": state_id,
            "trajectory_id": assignment_by_id[state_id].trajectory_id,
        }
        for state_id in assignment_only_ids
    ]
    states = tuple(actual_by_id[state_id] for state_id in sorted(actual_by_id))
    by_trajectory: dict[str, list[Any]] = defaultdict(list)
    for state in states:
        by_trajectory[state.trajectory_id].append(state)
    if set(by_trajectory) != set(train_assignments):
        raise ValueError("train state inventory omits a trajectory")

    targets = dict(split["checkpoint_state_targets_by_history_bin"])
    available = Counter(
        history_bin(len(state.candidate_event_ids)) for state in states
    )
    if any(available[name] < targets[name] for name in HISTORY_BINS):
        raise ValueError("train state inventory cannot satisfy balanced bin targets")
    salt = str(split["selection_salt"])
    heldout_count = int(split["heldout_trajectory_count"])
    deficits = dict(targets)
    selected: list[str] = []
    selected_set: set[str] = set()
    bin_counts_by_trajectory = {
        trajectory_id: Counter(
            history_bin(len(state.candidate_event_ids))
            for state in trajectory_states
        )
        for trajectory_id, trajectory_states in by_trajectory.items()
    }
    for _ in range(heldout_count):
        def selection_key(trajectory_id: str) -> tuple[Any, ...]:
            counts = bin_counts_by_trajectory[trajectory_id]
            coverage = sum(
                Fraction(min(counts[name], deficits[name]), targets[name])
                for name in HISTORY_BINS
            )
            return (
                -coverage,
                _order_digest(salt, "trajectory", trajectory_id),
                trajectory_id,
            )

        chosen = min(
            (
                trajectory_id
                for trajectory_id in sorted(by_trajectory)
                if trajectory_id not in selected_set
            ),
            key=selection_key,
        )
        selected.append(chosen)
        selected_set.add(chosen)
        counts = bin_counts_by_trajectory[chosen]
        for name in HISTORY_BINS:
            deficits[name] = max(0, deficits[name] - counts[name])
    if any(deficits.values()):
        raise ValueError("heldout trajectory count cannot satisfy balanced bin targets")

    heldout_all_states = tuple(
        state for trajectory_id in selected for state in by_trajectory[trajectory_id]
    )
    checkpoint_states: list[Any] = []
    for name in HISTORY_BINS:
        eligible = sorted(
            (
                state
                for state in heldout_all_states
                if history_bin(len(state.candidate_event_ids)) == name
            ),
            key=lambda state: (
                _order_digest(salt, "state", state.state_id),
                state.state_id,
            ),
        )
        checkpoint_states.extend(eligible[: targets[name]])
    checkpoint_states.sort(key=lambda state: state.state_id)

    heldout_trajectory_ids = sorted(selected_set)
    optimizer_trajectory_ids = sorted(set(by_trajectory) - selected_set)
    heldout_all_state_ids = sorted(state.state_id for state in heldout_all_states)
    checkpoint_state_ids = [state.state_id for state in checkpoint_states]
    optimizer_state_ids = sorted(
        state.state_id
        for trajectory_id in optimizer_trajectory_ids
        for state in by_trajectory[trajectory_id]
    )
    if (
        set(heldout_trajectory_ids) & set(optimizer_trajectory_ids)
        or set(heldout_all_state_ids) & set(optimizer_state_ids)
        or not set(checkpoint_state_ids).issubset(heldout_all_state_ids)
    ):
        raise RuntimeError("train-heldout split is not trajectory-disjoint")

    checkpoint_rows = [
        {
            "candidate_count": len(state.candidate_event_ids),
            "history_bin": history_bin(len(state.candidate_event_ids)),
            "state_id": state.state_id,
            "trajectory_id": state.trajectory_id,
        }
        for state in checkpoint_states
    ]
    realized_bins = Counter(row["history_bin"] for row in checkpoint_rows)
    if dict(realized_bins) != targets:
        raise RuntimeError("checkpoint state denominator is not exactly balanced")

    allowlists = {
        "checkpoint_state_ids": checkpoint_state_ids,
        "heldout_all_state_ids": heldout_all_state_ids,
        "heldout_trajectory_ids": heldout_trajectory_ids,
        "optimizer_state_ids": optimizer_state_ids,
        "optimizer_trajectory_ids": optimizer_trajectory_ids,
    }
    allowlist_sha256s = {
        f"{name}_sha256": sha256_json(values)
        for name, values in sorted(allowlists.items())
    }
    result = {
        "allowlist_sha256s": allowlist_sha256s,
        **allowlists,
        "census": {
            "assignment_derived_train_state_count": len(assignment_states),
            "assignment_only_exclusion_count": len(exclusions),
            "checkpoint_history_bin_counts": dict(realized_bins),
            "checkpoint_state_count": len(checkpoint_state_ids),
            "heldout_all_state_count": len(heldout_all_state_ids),
            "heldout_trajectory_count": len(heldout_trajectory_ids),
            "optimizer_state_count": len(optimizer_state_ids),
            "optimizer_trajectory_count": len(optimizer_trajectory_ids),
            "source_train_state_count": len(states),
            "source_train_trajectory_count": len(by_trajectory),
        },
        "checkpoint_selection": checkpoint,
        "checkpoint_states": checkpoint_rows,
        "config_sha256": config_sha,
        "firewall": {
            **firewall,
            "selected_role": "train",
            "tune_or_evaluation_state_count": 0,
        },
        "schema_version": "1.0.0",
        "selection": split,
        "source": {
            "assignment_manifest_sha256": source_sha,
            "assignment_only_exclusions": exclusions,
            "assignment_only_exclusions_sha256": sha256_json(exclusions),
            "expected_train_trajectory_count": expected_train,
            "train_state_inventory_content_sha256": inventory_content_sha,
            "training_input_content_sha256": inventory_source["input_content_sha256"],
            "training_input_manifest_sha256": inventory_source[
                "input_manifest_sha256"
            ],
            "training_states_sha256": inventory_source["states_sha256"],
        },
        "status": CONTRACT_STATUS,
        "trajectory_disjoint": True,
    }
    result["content_sha256"] = sha256_json(result)
    return result


def reduce_epoch_truth(
    records: Sequence[Mapping[str, Any]],
    *,
    checkpoint_states: Sequence[Mapping[str, Any]],
    epoch: int,
    checkpoint_sha256: str,
    contract_content_sha256: str,
) -> dict[str, Any]:
    """Reduce true B1--B4 recovery, retaining incomplete truth as a wait state."""
    epoch_index = _positive_int(epoch, "epoch")
    checkpoint_sha = _sha256(checkpoint_sha256, "checkpoint SHA256")
    contract_sha = _sha256(contract_content_sha256, "contract content SHA256")
    if isinstance(checkpoint_states, (str, bytes, bytearray, Mapping)) or not isinstance(
        checkpoint_states, Sequence
    ):
        raise ValueError("checkpoint states must be a sequence")
    expected: dict[str, tuple[str, str]] = {}
    for row in checkpoint_states:
        state_id = row.get("state_id")
        trajectory_id = row.get("trajectory_id")
        name = row.get("history_bin")
        if (
            not isinstance(state_id, str)
            or not isinstance(trajectory_id, str)
            or name not in HISTORY_BINS
            or state_id in expected
        ):
            raise ValueError("checkpoint state denominator is invalid")
        expected[state_id] = (trajectory_id, str(name))
    if not expected:
        raise ValueError("checkpoint state denominator is empty")
    if isinstance(records, (str, bytes, bytearray, Mapping)) or not isinstance(
        records, Sequence
    ):
        raise ValueError("epoch truth records must be a sequence")
    observed: dict[tuple[str, int], float] = {}
    for row in records:
        state_id = row.get("state_id")
        budget = row.get("budget")
        if state_id not in expected or budget not in BUDGETS:
            raise ValueError("epoch truth escaped the frozen denominator")
        key = (str(state_id), int(budget))
        if key in observed:
            raise ValueError("epoch truth contains a duplicate state-budget row")
        if "trajectory_id" in row and row["trajectory_id"] != expected[state_id][0]:
            raise ValueError("epoch truth trajectory identity drifted")
        if "history_bin" in row and row["history_bin"] != expected[state_id][1]:
            raise ValueError("epoch truth history bin drifted")
        value = float(row.get("normalized_recovery"))
        if not math.isfinite(value):
            raise ValueError("epoch truth recovery is non-finite")
        observed[key] = value
    expected_pairs = {(state_id, budget) for state_id in expected for budget in BUDGETS}
    missing = sorted(expected_pairs - set(observed))
    common = {
        "checkpoint_sha256": checkpoint_sha,
        "contract_content_sha256": contract_sha,
        "epoch": epoch_index,
        "expected_row_count": len(expected_pairs),
        "observed_row_count": len(observed),
        "truth_complete": not missing,
    }
    if missing:
        return {
            **common,
            "missing_pair_count": len(missing),
            "missing_pairs_sha256": sha256_json(
                [[state_id, budget] for state_id, budget in missing]
            ),
            "status": INCOMPLETE_TRUTH_STATUS,
        }

    def trajectory_equal(state_ids: set[str]) -> float:
        by_trajectory: dict[str, list[float]] = defaultdict(list)
        for state_id in sorted(state_ids):
            trajectory_id = expected[state_id][0]
            by_trajectory[trajectory_id].extend(
                observed[state_id, budget] for budget in BUDGETS
            )
        if not by_trajectory:
            raise ValueError("epoch truth slice contains no trajectories")
        return sum(
            sum(values) / len(values) for values in by_trajectory.values()
        ) / len(by_trajectory)

    all_ids = set(expected)
    long_ids = {
        state_id
        for state_id, (_, name) in expected.items()
        if name in {"long", "very_long"}
    }
    return {
        **common,
        "long_plus_trajectory_count": len(
            {expected[state_id][0] for state_id in long_ids}
        ),
        "long_plus_very_long_trajectory_equal_true_B1_B4_recovery_macro": (
            trajectory_equal(long_ids)
        ),
        "missing_pair_count": 0,
        "primary_trajectory_equal_true_B1_B4_recovery_macro": trajectory_equal(
            all_ids
        ),
        "status": COMPLETE_TRUTH_STATUS,
        "trajectory_count": len({value[0] for value in expected.values()}),
    }


def select_checkpoint_from_epoch_truth(
    epoch_truth: Sequence[Mapping[str, Any]],
    *,
    patience: int,
    minimum_delta: float,
    tie_breaker_minimum_delta: float = 0.0,
) -> dict[str, Any]:
    """Select by true recovery; incomplete epoch truth cannot advance patience."""
    patience_value = _positive_int(patience, "checkpoint patience")
    delta_value = _nonnegative_float(minimum_delta, "checkpoint minimum delta")
    tie_delta_value = _nonnegative_float(
        tie_breaker_minimum_delta, "checkpoint tie-breaker minimum delta"
    )
    if isinstance(epoch_truth, (str, bytes, bytearray, Mapping)) or not isinstance(
        epoch_truth, Sequence
    ) or not epoch_truth:
        raise ValueError("epoch truth must be a non-empty sequence")
    ordered = sorted(epoch_truth, key=lambda row: row.get("epoch", 0))
    epochs = tuple(row.get("epoch") for row in ordered)
    if epochs != tuple(range(1, len(ordered) + 1)):
        raise ValueError("epoch truth must be contiguous and start at epoch one")
    contract_hashes = {row.get("contract_content_sha256") for row in ordered}
    if len(contract_hashes) != 1:
        raise ValueError("epoch truth contract binding drifted")
    incomplete = next(
        (row for row in ordered if row.get("truth_complete") is not True), None
    )
    if incomplete is not None:
        return {
            "decision_ready": False,
            "first_incomplete_epoch": incomplete["epoch"],
            "observed_complete_epoch_count": incomplete["epoch"] - 1,
            "selected_checkpoint_sha256": None,
            "selected_epoch": None,
            "stopped_early": False,
            "status": "WAITING_FOR_COMPLETE_TRAIN_HELDOUT_EPOCH_TRUTH",
        }

    primary_key = "primary_trajectory_equal_true_B1_B4_recovery_macro"
    long_key = (
        "long_plus_very_long_trajectory_equal_true_B1_B4_recovery_macro"
    )
    for row in ordered:
        if row.get("status") != COMPLETE_TRUTH_STATUS:
            raise ValueError("complete epoch truth status is invalid")
        _sha256(row.get("checkpoint_sha256"), "epoch checkpoint SHA256")
        for key in (primary_key, long_key):
            value = float(row.get(key))
            if not math.isfinite(value):
                raise ValueError("checkpoint-selection metric is non-finite")

    best = ordered[0]
    stale = 0
    stop_epoch = None
    for row in ordered[1:]:
        primary_delta = float(row[primary_key]) - float(best[primary_key])
        primary_improvement = primary_delta > delta_value
        primary_tie = abs(primary_delta) <= delta_value
        long_improvement = (
            float(row[long_key]) - float(best[long_key]) > tie_delta_value
        )
        if primary_improvement or (primary_tie and long_improvement):
            best = row
            stale = 0
        else:
            stale += 1
            if stale >= patience_value:
                stop_epoch = int(row["epoch"])
                break
    return {
        "decision_ready": True,
        "minimum_delta": delta_value,
        "tie_breaker_minimum_delta": tie_delta_value,
        "observed_complete_epoch_count": (
            len(ordered) if stop_epoch is None else stop_epoch
        ),
        "patience": patience_value,
        "selected_checkpoint_sha256": best["checkpoint_sha256"],
        "selected_epoch": best["epoch"],
        "selected_long_plus_recovery": best[long_key],
        "selected_primary_recovery": best[primary_key],
        "stale_complete_epochs": stale,
        "stop_after_epoch": stop_epoch,
        "stopped_early": stop_epoch is not None,
        "status": (
            "EARLY_STOP_TRAIN_HELDOUT_CHECKPOINT_SELECTION"
            if stop_epoch is not None
            else "CONTINUE_TRAIN_HELDOUT_CHECKPOINT_SELECTION"
        ),
    }


__all__ = [
    "BUDGETS",
    "COMPLETE_TRUTH_STATUS",
    "CONTRACT_STATUS",
    "HISTORY_BINS",
    "INCOMPLETE_TRUTH_STATUS",
    "INVENTORY_STATUS",
    "build_train_input_state_inventory",
    "canonical_json_bytes",
    "materialize_train_heldout_contract",
    "reduce_epoch_truth",
    "select_checkpoint_from_epoch_truth",
    "sha256_json",
]
