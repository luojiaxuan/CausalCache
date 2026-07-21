"""Formal train-heldout protocol for structured conditional marginals."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.set_utility_recovery_checkpoint import (
    BUDGETS,
    BudgetRecoverySummary,
    trajectory_equal_budget_recovery,
)
from causalcache.set_utility_variable_history import history_bin


def conditional_groups(
    state: Mapping[str, Any],
    *,
    normalization_floor: float,
    maximum_base_cardinality: int,
) -> tuple[dict[str, Any], ...]:
    """Extract only candidate-complete conditional-marginal groups."""
    candidates = tuple(state["candidate_event_step_ids"])
    if (
        not candidates
        or candidates != tuple(sorted(candidates))
        or len(candidates) != len(set(candidates))
    ):
        raise ValueError("conditional candidates must be sorted and unique")
    if not math.isfinite(normalization_floor) or normalization_floor <= 0.0:
        raise ValueError("normalization floor must be finite and positive")
    if maximum_base_cardinality < 0:
        raise ValueError("maximum base cardinality cannot be negative")
    distances = {}
    for row in state["distance_rows"]:
        subset = tuple(row["coalition_event_step_ids"])
        distance = float(row["distance"])
        if (
            subset != tuple(sorted(subset))
            or len(subset) != len(set(subset))
            or not set(subset).issubset(candidates)
            or not math.isfinite(distance)
            or distance < 0.0
            or subset in distances
        ):
            raise ValueError("conditional distance table is invalid")
        distances[subset] = distance
    if () not in distances:
        raise ValueError("conditional distance table has no empty anchor")
    scale = max(distances[()], normalization_floor)
    result = []
    for selected in sorted(distances, key=lambda value: (len(value), value)):
        if len(selected) > maximum_base_cardinality:
            continue
        remaining = tuple(event for event in candidates if event not in selected)
        if not remaining:
            continue
        expansions = {
            event: tuple(sorted((*selected, event))) for event in remaining
        }
        if any(expansion not in distances for expansion in expansions.values()):
            continue
        raw = [0.0]
        action_mask = [True]
        for event in candidates:
            if event in expansions:
                raw.append(distances[selected] - distances[expansions[event]])
                action_mask.append(True)
            else:
                raw.append(0.0)
                action_mask.append(False)
        result.append(
            {
                "action_mask": tuple(action_mask),
                "normalized_marginals": tuple(value / scale for value in raw),
                "raw_marginals": tuple(raw),
                "selected_event_step_ids": selected,
            }
        )
    return tuple(result)


def fixed_trajectory_split(
    states: Sequence[Mapping[str, Any]],
    *,
    heldout_fraction: float,
    salt: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Fallback split when a separately frozen assignment is unavailable."""
    if not 0.0 < heldout_fraction < 0.5 or not salt:
        raise ValueError("train-heldout split parameters are invalid")
    rows = tuple(dict(state) for state in states)
    trajectories = sorted({row["trajectory_id"] for row in rows})
    heldout_count = max(1, round(len(trajectories) * heldout_fraction))
    ranked = sorted(
        trajectories,
        key=lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest(),
    )
    heldout_ids = set(ranked[:heldout_count])
    optimization = tuple(
        row for row in rows if row["trajectory_id"] not in heldout_ids
    )
    heldout = tuple(row for row in rows if row["trajectory_id"] in heldout_ids)
    validate_trajectory_disjoint_split(optimization, heldout)
    return optimization, heldout


def validate_trajectory_disjoint_split(
    optimization: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
) -> None:
    if not optimization or not heldout:
        raise ValueError("train-heldout split contains an empty side")
    optimization_trajectories = {row["trajectory_id"] for row in optimization}
    heldout_trajectories = {row["trajectory_id"] for row in heldout}
    overlap = optimization_trajectories & heldout_trajectories
    if overlap:
        raise ValueError(f"train-heldout trajectories overlap: {sorted(overlap)}")
    optimization_states = {row["state_id"] for row in optimization}
    heldout_states = {row["state_id"] for row in heldout}
    if len(optimization_states) != len(optimization) or len(heldout_states) != len(
        heldout
    ):
        raise ValueError("train-heldout states are duplicated")


def apply_split_manifest(
    states: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Apply a frozen trajectory split and optional heldout state subsample."""
    optimizer_ids = manifest.get("optimizer_trajectory_ids")
    heldout_ids = manifest.get("heldout_trajectory_ids")
    optimizer_state_ids = manifest.get("optimizer_state_ids")
    heldout_all_state_ids = manifest.get("heldout_all_state_ids")
    checkpoint_state_ids = manifest.get("checkpoint_state_ids")
    if not isinstance(optimizer_ids, Sequence) or isinstance(optimizer_ids, str):
        raise ValueError("split manifest omits optimizer trajectories")
    if not isinstance(heldout_ids, Sequence) or isinstance(heldout_ids, str):
        raise ValueError("split manifest omits heldout trajectories")
    optimizer = set(optimizer_ids)
    heldout = set(heldout_ids)
    if (
        not optimizer
        or not heldout
        or optimizer & heldout
        or any(not isinstance(value, str) or not value for value in optimizer | heldout)
    ):
        raise ValueError("split manifest trajectory identities are invalid")
    rows = tuple(dict(state) for state in states)
    available_trajectories = {row["trajectory_id"] for row in rows}
    if optimizer | heldout != available_trajectories:
        raise ValueError("split manifest does not partition the train trajectories")
    all_optimization_rows = tuple(
        row for row in rows if row["trajectory_id"] in optimizer
    )
    all_heldout_rows = tuple(row for row in rows if row["trajectory_id"] in heldout)
    if optimizer_state_ids is not None:
        optimizer_states = tuple(optimizer_state_ids)
        if (
            len(optimizer_states) != len(set(optimizer_states))
            or set(optimizer_states)
            != {row["state_id"] for row in all_optimization_rows}
        ):
            raise ValueError("optimizer state allowlist drifted")
    if heldout_all_state_ids is not None:
        heldout_states = tuple(heldout_all_state_ids)
        if (
            len(heldout_states) != len(set(heldout_states))
            or set(heldout_states)
            != {row["state_id"] for row in all_heldout_rows}
        ):
            raise ValueError("heldout all-state firewall drifted")
    optimization_rows = all_optimization_rows
    heldout_rows = all_heldout_rows
    if checkpoint_state_ids is not None:
        if not isinstance(checkpoint_state_ids, Sequence) or isinstance(
            checkpoint_state_ids, str
        ):
            raise ValueError("checkpoint state inventory must be a sequence")
        checkpoint_states = tuple(checkpoint_state_ids)
        if len(checkpoint_states) != len(set(checkpoint_states)):
            raise ValueError("checkpoint state inventory is duplicated")
        requested = set(checkpoint_states)
        available = {row["state_id"] for row in heldout_rows}
        if not requested or not requested.issubset(available):
            raise ValueError("checkpoint states escape heldout trajectories")
        heldout_rows = tuple(row for row in heldout_rows if row["state_id"] in requested)
        metadata = manifest.get("checkpoint_states")
        if metadata is not None:
            if not isinstance(metadata, Sequence) or isinstance(metadata, str):
                raise ValueError("checkpoint state metadata must be a sequence")
            by_state = {row.get("state_id"): row for row in metadata}
            if len(by_state) != len(metadata) or set(by_state) != requested:
                raise ValueError("checkpoint state metadata inventory drifted")
            for state in heldout_rows:
                row = by_state[state["state_id"]]
                candidate_count = len(state["candidate_event_step_ids"])
                if (
                    row.get("trajectory_id") != state["trajectory_id"]
                    or row.get("candidate_count") != candidate_count
                    or row.get("history_bin") != history_bin(candidate_count)
                ):
                    raise ValueError("checkpoint state metadata identity drifted")
    validate_trajectory_disjoint_split(optimization_rows, heldout_rows)
    return optimization_rows, heldout_rows


def prepare_group_examples(
    states: Sequence[Mapping[str, Any]],
    *,
    normalization_floor: float,
    maximum_base_cardinality: int,
    epsilon: float,
) -> tuple[dict[str, Any], ...]:
    """Flatten complete groups while retaining the full state feature record."""
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("balancing epsilon must be finite and nonnegative")
    examples = []
    for state in states:
        groups = conditional_groups(
            state,
            normalization_floor=normalization_floor,
            maximum_base_cardinality=maximum_base_cardinality,
        )
        empty_groups = [
            group for group in groups if not group["selected_event_step_ids"]
        ]
        if len(empty_groups) > 1:
            raise ValueError("structured state has duplicate empty anchor groups")
        empty_anchor = empty_groups[0] if empty_groups else None
        candidates = tuple(state["candidate_event_step_ids"])
        bin_name = history_bin(len(candidates))
        for group_index, group in enumerate(groups):
            valid = [
                value
                for value, keep in zip(
                    group["normalized_marginals"][1:],
                    group["action_mask"][1:],
                    strict=True,
                )
                if keep
            ]
            positive = sum(value > epsilon for value in valid)
            negative = sum(value < -epsilon for value in valid)
            if positive and negative:
                marginal_class = "mixed_sign"
            elif positive:
                marginal_class = "nonnegative"
            elif negative == len(valid):
                marginal_class = "all_negative"
            else:
                marginal_class = "nonpositive_with_tie"
            best = max(valid)
            example = dict(state)
            enriched = dict(group)
            enriched["balance"] = {
                "cardinality": len(group["selected_event_step_ids"]),
                "class": marginal_class,
                "history": bin_name,
                "stop_all_negative": best <= 0.0,
            }
            example["conditional_groups"] = (enriched,)
            example["group_example_id"] = f"{state['state_id']}:g{group_index:04d}"
            example["structured_empty_anchor"] = empty_anchor
            examples.append(example)
    if not examples:
        raise ValueError("optimization split contains no complete conditional groups")
    return tuple(examples)


def balancing_inventory(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[Any]] = {
        axis: Counter() for axis in ("cardinality", "class", "history", "stop_all_negative")
    }
    for example in examples:
        balance = example["conditional_groups"][0]["balance"]
        for axis in result:
            result[axis][balance[axis]] += 1
    return {
        axis: {
            str(key): value
            for key, value in sorted(
                counts.items(), key=lambda row: str(row[0])
            )
        }
        for axis, counts in result.items()
    }


def formal_group_inventory(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind optimizer groups; ``stop_all_negative`` includes exact-zero ties."""
    if not examples:
        raise ValueError("formal group inventory cannot be empty")
    identities = []
    state_ids = set()
    trajectory_ids = set()
    history_counts: Counter[str] = Counter()
    cardinality_counts: Counter[int] = Counter()
    stop_counts: Counter[bool] = Counter()
    joint_counts: Counter[str] = Counter()
    for example in examples:
        groups = tuple(example.get("conditional_groups", ()))
        if len(groups) != 1:
            raise ValueError("formal inventory expects one group per example")
        group = groups[0]
        selected = tuple(group["selected_event_step_ids"])
        valid = tuple(
            value
            for value, keep in zip(
                group["normalized_marginals"][1:],
                group["action_mask"][1:],
                strict=True,
            )
            if keep
        )
        if not valid:
            raise ValueError("formal inventory group has no remaining candidate")
        state_id = str(example["state_id"])
        trajectory_id = str(example["trajectory_id"])
        bin_name = history_bin(len(example["candidate_event_step_ids"]))
        stop = max(valid) <= 0.0
        balance = group.get("balance")
        if (
            not isinstance(balance, Mapping)
            or type(balance.get("stop_all_negative")) is not bool
            or balance["stop_all_negative"] is not stop
        ):
            raise ValueError("formal group STOP balance metadata drifted")
        identity = f"{state_id}|{','.join(str(value) for value in selected)}"
        identities.append(identity)
        state_ids.add(state_id)
        trajectory_ids.add(trajectory_id)
        history_counts[bin_name] += 1
        cardinality_counts[len(selected)] += 1
        stop_counts[stop] += 1
        joint_counts[f"{len(selected)}|{bin_name}|{str(stop).lower()}"] += 1
    if len(identities) != len(set(identities)):
        raise ValueError("formal optimizer group identities are duplicated")

    def digest(values: Iterable[str]) -> str:
        return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()

    unsigned = {
        "base_cardinality_counts": {
            str(key): value for key, value in sorted(cardinality_counts.items())
        },
        "complete_group_count": len(identities),
        "group_identity_sha256": digest(identities),
        "history_bin_counts": dict(sorted(history_counts.items())),
        "joint_stratum_counts": dict(sorted(joint_counts.items())),
        "optimizer_state_count": len(state_ids),
        "optimizer_state_identity_sha256": digest(state_ids),
        "optimizer_trajectory_count": len(trajectory_ids),
        "optimizer_trajectory_identity_sha256": digest(trajectory_ids),
        "stop_all_negative_counts": {
            str(key).lower(): value for key, value in sorted(stop_counts.items())
        },
    }
    return {
        **unsigned,
        "content_sha256": hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }


def balanced_group_epoch(
    examples: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    sample_count: int,
    axes: Sequence[str],
    power: float,
    maximum_weight: float,
) -> tuple[dict[str, Any], ...]:
    """Sample trajectories uniformly, then inverse-frequency group strata."""
    valid_axes = {"cardinality", "class", "history", "stop_all_negative"}
    normalized_axes = tuple(axes)
    if (
        sample_count <= 0
        or not normalized_axes
        or len(normalized_axes) != len(set(normalized_axes))
        or not set(normalized_axes).issubset(valid_axes)
        or not math.isfinite(power)
        or power < 0.0
        or not math.isfinite(maximum_weight)
        or maximum_weight < 1.0
    ):
        raise ValueError("group balancing parameters are invalid")
    by_trajectory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    counts = {axis: Counter() for axis in normalized_axes}
    for example in examples:
        by_trajectory[example["trajectory_id"]].append(example)
        balance = example["conditional_groups"][0]["balance"]
        for axis in normalized_axes:
            counts[axis][balance[axis]] += 1
    trajectories = sorted(by_trajectory)
    generator = random.Random(seed)
    result = []
    order = []
    while len(order) < sample_count:
        cycle = list(trajectories)
        generator.shuffle(cycle)
        order.extend(cycle)
    for trajectory in order[:sample_count]:
        rows = by_trajectory[trajectory]
        weights = []
        for row in rows:
            balance = row["conditional_groups"][0]["balance"]
            weight = 1.0
            for axis in normalized_axes:
                frequency = counts[axis][balance[axis]]
                category_count = len(counts[axis])
                inverse = len(examples) / (category_count * frequency)
                weight *= inverse**power
            weights.append(min(maximum_weight, weight))
        result.append(dict(generator.choices(rows, weights=weights, k=1)[0]))
    return tuple(result)


def pack_group_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    maximum_groups_per_state: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Pack sampled groups without changing their effective multiplicity."""
    if maximum_groups_per_state <= 1 or not examples:
        raise ValueError("packed group parameters are invalid")
    by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for example in examples:
        groups = tuple(example.get("conditional_groups", ()))
        if len(groups) != 1:
            raise ValueError("packing expects one conditional group per sample")
        by_state[example["state_id"]].append(example)
    generator = random.Random(seed)
    packed = []
    for state_id in sorted(by_state):
        rows = list(by_state[state_id])
        generator.shuffle(rows)
        reference = rows[0]
        if any(
            row["trajectory_id"] != reference["trajectory_id"]
            or row["candidate_event_step_ids"]
            != reference["candidate_event_step_ids"]
            for row in rows[1:]
        ):
            raise ValueError("packed group state identity drifted")
        anchor = reference.get("structured_empty_anchor")
        if anchor is not None and (
            not isinstance(anchor, Mapping) or anchor["selected_event_step_ids"]
        ):
            raise ValueError("packed state has an invalid empty anchor")
        if any(row.get("structured_empty_anchor") != anchor for row in rows[1:]):
            raise ValueError("packed state empty-anchor identity drifted")
        sampled_empty = [
            row
            for row in rows
            if not row["conditional_groups"][0]["selected_event_step_ids"]
        ]
        sampled_nonempty = [
            row
            for row in rows
            if row["conditional_groups"][0]["selected_event_step_ids"]
        ]
        if sampled_empty and anchor is None:
            raise ValueError("sampled empty groups have no canonical empty anchor")

        chunks: list[tuple[list[Mapping[str, Any]], bool]] = []
        while sampled_empty:
            chunk = [sampled_empty.pop()]
            take = min(maximum_groups_per_state - 1, len(sampled_nonempty))
            chunk.extend(sampled_nonempty[-take:] if take else ())
            if take:
                del sampled_nonempty[-take:]
            chunks.append((chunk, False))
        auxiliary_capacity = maximum_groups_per_state - int(anchor is not None)
        while sampled_nonempty:
            chunk = sampled_nonempty[-auxiliary_capacity:]
            del sampled_nonempty[-auxiliary_capacity:]
            chunks.append((chunk, anchor is not None))

        for chunk, add_auxiliary_anchor in chunks:
            value = dict(reference)
            value["conditional_groups"] = tuple(
                ([dict(anchor)] if add_auxiliary_anchor else [])
                + [row["conditional_groups"][0] for row in chunk]
            )
            value["group_example_ids"] = tuple(
                row["group_example_id"] for row in chunk
            )
            value["conditional_group_weights"] = tuple(
                ([0.0] if add_auxiliary_anchor else [])
                + [1.0] * len(chunk)
            )
            value.pop("group_example_id", None)
            value.pop("structured_empty_anchor", None)
            packed.append(value)
    generator.shuffle(packed)
    if sum(len(row["group_example_ids"]) for row in packed) != len(examples):
        raise AssertionError("packing changed the balanced sampled-group count")
    observed_weight = sum(
        sum(row["conditional_group_weights"]) for row in packed
    )
    if observed_weight != len(examples):
        raise AssertionError("packing changed sampled group effective weight")
    return tuple(packed)


def distance_table(
    state: Mapping[str, Any],
    supplemental: Mapping[str, Mapping[tuple[int, ...], float]] | None = None,
) -> dict[tuple[int, ...], float]:
    candidates = tuple(state["candidate_event_step_ids"])
    result = {}
    for row in state["distance_rows"]:
        subset = tuple(row["coalition_event_step_ids"])
        value = float(row["distance"])
        if subset in result or not set(subset).issubset(candidates):
            raise ValueError("state truth contains duplicate or invalid coalitions")
        result[subset] = value
    extra = (supplemental or {}).get(state["state_id"], {})
    for subset, value in extra.items():
        if subset in result and result[subset] != value:
            raise ValueError("supplemental truth conflicts with static truth")
        if not set(subset).issubset(candidates) or not math.isfinite(value):
            raise ValueError("supplemental truth escapes the state")
        result[subset] = value
    return result


def candidate_complete_coalitions(
    candidates: Sequence[int], bases: Iterable[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    candidate_tuple = tuple(candidates)
    if (
        not candidate_tuple
        or candidate_tuple != tuple(sorted(candidate_tuple))
        or len(candidate_tuple) != len(set(candidate_tuple))
    ):
        raise ValueError("candidate-complete schedule has invalid candidates")
    desired = {(), candidate_tuple}
    for raw_base in bases:
        base = tuple(raw_base)
        if (
            base != tuple(sorted(base))
            or len(base) != len(set(base))
            or not set(base).issubset(candidate_tuple)
        ):
            raise ValueError("candidate-complete schedule has an invalid base")
        desired.add(base)
        desired.update(
            tuple(sorted((*base, event)))
            for event in candidate_tuple
            if event not in base
        )
    return tuple(sorted(desired, key=lambda value: (len(value), value)))


def build_epoch_truth_plan(
    *,
    epoch: int,
    checkpoint: Mapping[str, Any],
    rollout_records: Sequence[Mapping[str, Any]],
    states_by_id: Mapping[str, Mapping[str, Any]],
    supplemental: Mapping[str, Mapping[tuple[int, ...], float]] | None = None,
) -> dict[str, Any]:
    """Bind one checkpoint rollout to candidate-complete truth requirements."""
    if epoch <= 0 or not rollout_records:
        raise ValueError("epoch truth plan inputs are invalid")
    records = []
    missing_total = 0
    desired_total = 0
    for rollout in sorted(rollout_records, key=lambda row: row["state_id"]):
        state_id = rollout["state_id"]
        state = states_by_id.get(state_id)
        if state is None:
            raise ValueError("rollout state is outside the fixed heldout split")
        candidates = tuple(state["candidate_event_step_ids"])
        if tuple(rollout["candidate_event_ids"]) != candidates:
            raise ValueError("rollout candidate identity drifted")
        selections = rollout["selections"]
        if set(selections) != {str(value) for value in BUDGETS}:
            raise ValueError("rollout does not contain B1--B4 selections")
        bases = tuple(tuple(value) for value in rollout["queried_bases"])
        desired = set(candidate_complete_coalitions(candidates, bases))
        desired.update(tuple(selections[str(budget)]) for budget in BUDGETS)
        desired_ordered = tuple(sorted(desired, key=lambda value: (len(value), value)))
        available = distance_table(state, supplemental)
        missing = tuple(subset for subset in desired_ordered if subset not in available)
        desired_total += len(desired_ordered)
        missing_total += len(missing)
        records.append(
            {
                "candidate_event_ids": list(candidates),
                "desired_coalitions": [list(value) for value in desired_ordered],
                "history_bin": history_bin(len(candidates)),
                "logical_shard": state.get("logical_shard"),
                "missing_coalitions": [list(value) for value in missing],
                "queried_bases": [list(value) for value in bases],
                "selections": selections,
                "state_id": state_id,
                "trajectory_id": state["trajectory_id"],
            }
        )
    return {
        "checkpoint": dict(checkpoint),
        "desired_coalition_count": desired_total,
        "epoch": epoch,
        "missing_coalition_count": missing_total,
        "records": records,
        "schema_version": "causalcache.structured_epoch_truth_plan.v1",
        "state_count": len(records),
        "truth_complete": missing_total == 0,
    }


def merge_epoch_truth_plans(plans: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Union all actually queried epoch bases into one resumable label schedule."""
    if not plans:
        raise ValueError("cannot merge an empty epoch plan inventory")
    union: dict[str, dict[str, Any]] = {}
    epochs = []
    for plan in sorted(plans, key=lambda value: int(value["epoch"])):
        epoch = int(plan["epoch"])
        if epoch in epochs:
            raise ValueError("epoch truth plans are duplicated")
        epochs.append(epoch)
        for record in plan["records"]:
            state_id = record["state_id"]
            current = union.setdefault(
                state_id,
                {
                    "candidate_event_ids": record["candidate_event_ids"],
                    "desired": set(),
                    "history_bin": record["history_bin"],
                    "logical_shard": record["logical_shard"],
                    "missing": set(),
                    "state_id": state_id,
                    "trajectory_id": record["trajectory_id"],
                },
            )
            if (
                current["candidate_event_ids"] != record["candidate_event_ids"]
                or current["trajectory_id"] != record["trajectory_id"]
            ):
                raise ValueError("epoch plans disagree on state identity")
            current["desired"].update(tuple(value) for value in record["desired_coalitions"])
            current["missing"].update(tuple(value) for value in record["missing_coalitions"])
    records = []
    for state_id, row in sorted(union.items()):
        desired = sorted(row.pop("desired"), key=lambda value: (len(value), value))
        missing = sorted(row.pop("missing"), key=lambda value: (len(value), value))
        records.append(
            {
                **row,
                "desired_coalitions": [list(value) for value in desired],
                "missing_coalitions": [
                    {"event_ids": list(value), "source": "structured_epoch_rollout"}
                    for value in missing
                ],
            }
        )
    return {
        "epoch_count": len(epochs),
        "epochs": epochs,
        "missing_coalition_count": sum(
            len(row["missing_coalitions"]) for row in records
        ),
        "records": records,
        "schema_version": "causalcache.structured_truth_schedule.v1",
        "state_count": len(records),
    }


@dataclass(frozen=True)
class EpochRecovery:
    epoch: int
    recovery: BudgetRecoverySummary
    long_plus_macro: float | None


def evaluate_epoch_truth(
    plan: Mapping[str, Any],
    *,
    states_by_id: Mapping[str, Mapping[str, Any]],
    normalization_floor: float,
    supplemental: Mapping[str, Mapping[tuple[int, ...], float]] | None = None,
) -> tuple[EpochRecovery | None, tuple[dict[str, Any], ...]]:
    """Return true recovery only when every selected B1--B4 subset is labeled."""
    recovery_rows = []
    missing = []
    for record in plan["records"]:
        state = states_by_id[record["state_id"]]
        truth = distance_table(state, supplemental)
        if () not in truth:
            raise ValueError("heldout truth has no empty anchor")
        denominator = max(truth[()], normalization_floor)
        for budget in BUDGETS:
            subset = tuple(record["selections"][str(budget)])
            if subset not in truth:
                missing.append(
                    {
                        "budget": budget,
                        "coalition_event_step_ids": list(subset),
                        "state_id": state["state_id"],
                    }
                )
                continue
            recovery_rows.append(
                {
                    "budget": budget,
                    "history_bin": history_bin(len(state["candidate_event_step_ids"])),
                    "normalized_recovery": (truth[()] - truth[subset]) / denominator,
                    "state_id": state["state_id"],
                    "trajectory_id": state["trajectory_id"],
                }
            )
    if missing:
        return None, tuple(missing)
    summary = trajectory_equal_budget_recovery(recovery_rows)
    long_rows = [
        row for row in recovery_rows if row["history_bin"] in {"long", "very_long"}
    ]
    long_macro = (
        trajectory_equal_budget_recovery(long_rows).macro_b1_b4
        if long_rows
        else None
    )
    return (
        EpochRecovery(
            epoch=int(plan["epoch"]),
            recovery=summary,
            long_plus_macro=long_macro,
        ),
        (),
    )


def replay_recovery_early_stopping(
    recoveries: Sequence[EpochRecovery],
    *,
    patience: int,
    minimum_delta: float,
    long_tie_tolerance: float,
) -> dict[str, Any]:
    """Replay the only legal early-stopping rule after delayed truth arrives."""
    if (
        patience <= 0
        or minimum_delta < 0.0
        or long_tie_tolerance < 0.0
        or not all(math.isfinite(value) for value in (minimum_delta, long_tie_tolerance))
    ):
        raise ValueError("recovery early-stopping parameters are invalid")
    ordered = sorted(recoveries, key=lambda value: value.epoch)
    if not ordered or len({value.epoch for value in ordered}) != len(ordered):
        raise ValueError("recovery epochs are empty or duplicated")
    if tuple(value.epoch for value in ordered) != tuple(range(1, len(ordered) + 1)):
        raise ValueError("recovery epochs must be contiguous and start at one")
    best: EpochRecovery | None = None
    stale = 0
    trace = []
    stopped_epoch = None
    for value in ordered:
        macro = value.recovery.macro_b1_b4
        improvement = best is None or macro > best.recovery.macro_b1_b4 + minimum_delta
        tie_break = False
        if (
            not improvement
            and best is not None
            and abs(macro - best.recovery.macro_b1_b4) <= long_tie_tolerance
            and value.long_plus_macro is not None
            and (
                best.long_plus_macro is None
                or value.long_plus_macro > best.long_plus_macro + minimum_delta
            )
        ):
            tie_break = True
        if improvement or tie_break:
            best = value
            stale = 0
            decision = "BEST_MACRO" if improvement else "BEST_LONG_TIE_BREAK"
        else:
            stale += 1
            decision = "STALE"
        trace.append(
            {
                "decision": decision,
                "epoch": value.epoch,
                "long_plus_macro": value.long_plus_macro,
                "macro_B1_B4": macro,
                "patience": stale,
            }
        )
        if stale >= patience:
            stopped_epoch = value.epoch
            break
    assert best is not None
    return {
        "best_epoch": best.epoch,
        "best_long_plus_macro": best.long_plus_macro,
        "best_macro_B1_B4": best.recovery.macro_b1_b4,
        "eligible_epoch_count": len(trace),
        "stopped_epoch": stopped_epoch,
        "trace": trace,
    }


__all__ = [
    "EpochRecovery",
    "apply_split_manifest",
    "balanced_group_epoch",
    "balancing_inventory",
    "build_epoch_truth_plan",
    "candidate_complete_coalitions",
    "conditional_groups",
    "distance_table",
    "evaluate_epoch_truth",
    "formal_group_inventory",
    "fixed_trajectory_split",
    "merge_epoch_truth_plans",
    "pack_group_examples",
    "prepare_group_examples",
    "replay_recovery_early_stopping",
    "validate_trajectory_disjoint_split",
]
