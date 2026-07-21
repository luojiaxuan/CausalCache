"""Truth planning and reduction for post-selection heldout comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file
from causalcache.set_utility_heldout_truth_schedule import (
    FORMAL_MANIFEST_STATUS,
    FORMAL_RECEIPT_STATUS,
    LABEL_STATUS,
)
from causalcache.set_utility_train_heldout_contract import sha256_json
from causalcache.set_utility_variable_history import history_bin


SELECTOR_SCHEMA = "causalcache.direct_on_policy_unified_selections.v1"
SELECTOR_STATUS = "COMPLETED_DIRECT_ON_POLICY_UNIFIED_SELECTIONS"
SCHEDULE_SCHEMA = "causalcache.structured_truth_schedule.v1"
RESULT_SCHEMA = "causalcache.post_selection_truth_result.v1"
RESULT_STATUS = "COMPLETED_POST_SELECTION_TRUTH_COMPARISON"
MODEL_FAMILY = "set_transformer_direct_marginal"
EXPECTED_STATE_COUNT = 256
BUDGETS = (1, 2, 3, 4)
METHODS = (
    "recent",
    "set_transformer",
    "set_transformer_hybrid",
    "structured_deepsets",
    "structured_deepsets_hybrid",
)
REDUCED_METHODS = ("summary_only", *METHODS)
SHARED_TRUTH_BINDINGS = (
    "execution_config_sha256",
    "heldout_manifest_content_sha256",
    "heldout_manifest_file_sha256",
    "input_content_sha256",
    "input_manifest_file_sha256",
    "scientific_config_sha256",
    "source_content_sha256",
    "source_manifest_file_sha256",
    "source_revision",
)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_or_verify(path: Path, value: Mapping[str, Any], *, pretty: bool) -> None:
    payload = canonical_json_bytes(value, pretty=pretty) + b"\n"
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing signed output drifted: {path}")
        return
    _write_atomic(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            result.append(row)
    return tuple(result)


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _read_compact_signed(path: Path, *, schema: str) -> dict[str, Any]:
    value = _read_json(path)
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256", None)
    if claimed != sha256_json(unsigned):
        raise ValueError(f"signed JSON content hash drifted: {path}")
    if value.get("schema_version") != schema:
        raise ValueError(f"signed JSON schema drifted: {path}")
    return value


def _compact_signed(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    return {**unsigned, "content_sha256": sha256_json(unsigned)}


def _pretty_signed(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    digest = hashlib.sha256(canonical_json_bytes(unsigned, pretty=True)).hexdigest()
    return {**unsigned, "content_sha256": digest}


def _canonical_candidates(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an event-id array")
    result = tuple(value)
    if (
        not result
        or any(type(event) is not int for event in result)
        or result != tuple(sorted(result))
        or len(result) != len(set(result))
    ):
        raise ValueError(f"{label} is not canonical")
    return result


def _canonical_subset(
    value: Any, *, candidates: tuple[int, ...], label: str
) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an event-id array")
    result = tuple(value)
    if (
        any(type(event) is not int for event in result)
        or result != tuple(sorted(result))
        or len(result) != len(set(result))
        or not set(result).issubset(candidates)
    ):
        raise ValueError(f"{label} is not canonical")
    return result


def _finite_distance(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _load_input_states(
    input_root: Path, *, expected_content_sha256: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    expected = _sha256(expected_content_sha256, label="input content SHA256")
    manifest_path = input_root / "manifest.json"
    manifest = _read_json(manifest_path)
    states_name = manifest.get("states_jsonl")
    if (
        manifest.get("content_sha256") != expected
        or manifest.get("evaluation_labels_included") is not False
        or not isinstance(states_name, str)
    ):
        raise ValueError("post-selection input identity or firewall drifted")
    states_path = (input_root / states_name).resolve()
    if input_root.resolve() not in states_path.parents:
        raise ValueError("post-selection input states path escapes its root")
    if not states_path.is_file() or sha256_file(states_path) != manifest.get(
        "states_sha256"
    ):
        raise ValueError("post-selection input state payload drifted")
    rows = _read_jsonl(states_path)
    if manifest.get("state_count") is not None and manifest["state_count"] != len(rows):
        raise ValueError("post-selection input state count drifted")
    states: dict[str, dict[str, Any]] = {}
    for row in rows:
        state_id = row.get("state_id")
        if not isinstance(state_id, str) or state_id in states:
            raise ValueError("post-selection input state identity drifted")
        if row.get("role") not in {"train", "tune"}:
            raise ValueError("post-selection input crossed the evaluation firewall")
        candidates = _canonical_candidates(
            row.get("candidate_event_step_ids"), label="input candidates"
        )
        distances: dict[tuple[int, ...], float] = {}
        raw_distances = row.get("distance_rows")
        if not isinstance(raw_distances, list):
            raise ValueError("post-selection input has no distance table")
        for index, raw in enumerate(raw_distances):
            if not isinstance(raw, Mapping):
                raise ValueError("post-selection input distance row is invalid")
            subset = _canonical_subset(
                raw.get("coalition_event_step_ids"),
                candidates=candidates,
                label=f"input distance row {index}",
            )
            distance = _finite_distance(raw.get("distance"), label="input distance")
            if subset in distances:
                raise ValueError("post-selection input duplicates a coalition")
            distances[subset] = distance
        states[state_id] = {**row, "_candidates": candidates, "_truth": distances}
    return manifest, states


def _validate_selector_output(path: Path) -> dict[str, Any]:
    value = _read_compact_signed(path, schema=SELECTOR_SCHEMA)
    if (
        value.get("status") != SELECTOR_STATUS
        or value.get("truth_labels_accessed") is not False
        or tuple(value.get("methods", ())) != METHODS
        or not isinstance(value.get("bindings"), Mapping)
        or not isinstance(value.get("checkpoints"), Mapping)
    ):
        raise ValueError("selector output contract drifted")
    bindings = value["bindings"]
    for name in (
        "cache_content_sha256",
        "heldout_manifest_content_sha256",
        "heldout_manifest_file_sha256",
        "input_content_sha256",
        "input_states_sha256",
        "set_transformer_config_sha256",
        "source_manifest_file_sha256",
        "structured_deepsets_config_sha256",
    ):
        _sha256(bindings.get(name), label=f"selector {name}")
    checkpoints = value["checkpoints"]
    expected_families = {
        "set_transformer": MODEL_FAMILY,
        "structured_deepsets": "deepsets_structured_marginal",
    }
    if set(checkpoints) != set(expected_families):
        raise ValueError("selector output checkpoint inventory drifted")
    for name, family in expected_families.items():
        checkpoint = checkpoints[name]
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("model_family") != family
            or not isinstance(checkpoint.get("variant"), str)
        ):
            raise ValueError(f"selector output {name} checkpoint drifted")
        _sha256(checkpoint.get("sha256"), label=f"selected {name} checkpoint SHA256")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_STATE_COUNT:
        raise ValueError("selector output denominator drifted")
    seen = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("selector output record is invalid")
        state_id = record.get("state_id")
        trajectory_id = record.get("trajectory_id")
        name = record.get("history_bin")
        candidates = _canonical_candidates(
            record.get("candidate_event_ids"), label="selector candidates"
        )
        if (
            not isinstance(state_id, str)
            or state_id in seen
            or not isinstance(trajectory_id, str)
            or name != history_bin(len(candidates))
            or not isinstance(record.get("methods"), Mapping)
            or set(record["methods"]) != set(METHODS)
        ):
            raise ValueError("selector output state identity drifted")
        seen.add(state_id)
        for method in METHODS:
            selections = record["methods"][method]
            if not isinstance(selections, Mapping) or set(selections) != {
                str(value) for value in BUDGETS
            }:
                raise ValueError("selector output budget inventory drifted")
            for budget in BUDGETS:
                subset = _canonical_subset(
                    selections[str(budget)],
                    candidates=candidates,
                    label=f"{method} B{budget} selection",
                )
                if len(subset) > budget:
                    raise ValueError("selector output exceeded at-most-B")
    return value


def _load_heldout_manifest(
    path: Path, *, expected_content_sha256: str
) -> dict[str, Any]:
    value = _read_compact_signed(path, schema="1.0.0")
    expected = _sha256(expected_content_sha256, label="heldout content SHA256")
    rows = value.get("checkpoint_states")
    ids = value.get("checkpoint_state_ids")
    firewall = value.get("firewall")
    if (
        value.get("content_sha256") != expected
        or not isinstance(rows, list)
        or not isinstance(ids, list)
        or len(rows) != EXPECTED_STATE_COUNT
        or len(ids) != EXPECTED_STATE_COUNT
        or len(set(ids)) != EXPECTED_STATE_COUNT
        or {row.get("state_id") for row in rows if isinstance(row, Mapping)}
        != set(ids)
        or not isinstance(firewall, Mapping)
        or firewall.get("allowed_role", "train") != "train"
        or firewall.get("tune_access") is not False
        or firewall.get("evaluation_access") is not False
    ):
        raise ValueError("heldout manifest denominator or firewall drifted")
    return value


def _merge_distance(
    table: dict[tuple[int, ...], float], subset: tuple[int, ...], distance: float
) -> None:
    previous = table.get(subset)
    if previous is not None and previous != distance:
        raise ValueError("formal truth roots conflict on a coalition distance")
    table[subset] = distance


def load_sealed_truth_union(
    truth_roots: Sequence[Path],
    *,
    states_by_id: Mapping[str, Mapping[str, Any]],
    allowed_state_ids: set[str],
) -> tuple[dict[str, dict[tuple[int, ...], float]], dict[str, Any]]:
    """Validate and union sealed roots without depending on their model schedule."""
    if not truth_roots:
        raise ValueError("at least one sealed formal truth root is required")
    if not allowed_state_ids or not allowed_state_ids.issubset(states_by_id):
        raise ValueError("formal truth denominator is invalid")
    result: dict[str, dict[tuple[int, ...], float]] = {
        state_id: dict(states_by_id[state_id]["_truth"])
        for state_id in allowed_state_ids
    }
    common_bindings: dict[str, Any] | None = None
    root_bindings = []
    seen_roots: set[Path] = set()
    for raw_root in truth_roots:
        root = raw_root.resolve()
        if root in seen_roots:
            raise ValueError("formal truth root inventory contains duplicates")
        seen_roots.add(root)
        manifest_path = root / "formal-truth-manifest.json"
        receipt_path = root / "formal-truth-receipt.json"
        manifest = _read_compact_signed(
            manifest_path, schema="causalcache.formal_heldout_truth_manifest.v1"
        )
        receipt = _read_compact_signed(
            receipt_path, schema="causalcache.formal_heldout_truth_receipt.v1"
        )
        if (
            manifest.get("status") != FORMAL_MANIFEST_STATUS
            or receipt.get("status") != FORMAL_RECEIPT_STATUS
            or receipt.get("formal_manifest_content_sha256")
            != manifest.get("content_sha256")
            or receipt.get("formal_manifest_file_sha256") != sha256_file(manifest_path)
            or receipt.get("state_count") != manifest.get("state_count")
            or receipt.get("coalition_count") != manifest.get("coalition_count")
            or receipt.get("truth_schedule_summary_content_sha256")
            != manifest.get("truth_schedule_summary_content_sha256")
            or receipt.get("truth_schedule_summary_file_sha256")
            != manifest.get("truth_schedule_summary_file_sha256")
            or receipt.get("input_content_sha256")
            != manifest.get("input_content_sha256")
            or receipt.get("heldout_manifest_content_sha256")
            != manifest.get("heldout_manifest_content_sha256")
            or receipt.get("source_manifest_file_sha256")
            != manifest.get("source_manifest_file_sha256")
        ):
            raise ValueError("formal truth manifest/receipt binding drifted")
        model_schedules = manifest.get("model_schedules")
        if not isinstance(model_schedules, Mapping) or not model_schedules:
            raise ValueError("formal truth omits model schedule provenance")
        for schedule_name, schedule in model_schedules.items():
            if not isinstance(schedule_name, str) or not isinstance(schedule, Mapping):
                raise ValueError("formal truth model schedule provenance is invalid")
            _sha256(
                schedule.get("content_sha256"),
                label="formal truth schedule content SHA256",
            )
            _sha256(
                schedule.get("file_sha256"),
                label="formal truth schedule file SHA256",
            )
        current = {name: manifest.get(name) for name in SHARED_TRUTH_BINDINGS}
        for name in SHARED_TRUTH_BINDINGS:
            if name == "source_revision":
                value = current[name]
                if (
                    not isinstance(value, str)
                    or len(value) != 40
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError("formal truth source revision drifted")
            elif name == "source_content_sha256" and current[name] is None:
                continue
            else:
                _sha256(current[name], label=f"formal truth {name}")
        if common_bindings is None:
            common_bindings = current
        elif current != common_bindings:
            raise ValueError("formal truth roots disagree on shared bindings")

        bindings = manifest.get("states")
        if (
            not isinstance(bindings, list)
            or manifest.get("state_count") != len(bindings)
        ):
            raise ValueError("formal truth manifest state inventory drifted")
        expected_paths: set[Path] = set()
        seen_states: set[str] = set()
        observed_rows = 0
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise ValueError("formal truth state binding is invalid")
            state_id = binding.get("state_id")
            relative = binding.get("terminal_relative_path")
            state = states_by_id.get(str(state_id))
            if (
                state_id not in allowed_state_ids
                or state_id in seen_states
                or state is None
                or not isinstance(relative, str)
                or binding.get("trajectory_id") != state.get("trajectory_id")
                or tuple(binding.get("candidate_event_ids", ()))
                != state["_candidates"]
                or binding.get("logical_shard") != state.get("logical_shard")
            ):
                raise ValueError("formal truth state binding escaped the denominator")
            seen_states.add(str(state_id))
            terminal_path = (root / relative).resolve()
            if root not in terminal_path.parents or terminal_path.parent != (
                root / "states"
            ).resolve():
                raise ValueError("formal truth terminal path escapes its root")
            expected_paths.add(terminal_path)
            if (
                not terminal_path.is_file()
                or sha256_file(terminal_path) != binding.get("terminal_file_sha256")
            ):
                raise ValueError("formal truth terminal file hash drifted")
            terminal = _read_json(terminal_path)
            if (
                terminal.get("status") != LABEL_STATUS
                or terminal.get("role") != "train"
                or terminal.get("state_id") != state_id
                or terminal.get("trajectory_id") != state.get("trajectory_id")
                or tuple(terminal.get("candidate_event_step_ids", ()))
                != state["_candidates"]
                or terminal.get("state_identity_sha256")
                != binding.get("state_identity_sha256")
                or terminal.get("source_revision") != current["source_revision"]
                or terminal.get("scientific_config_sha256")
                != current["scientific_config_sha256"]
                or terminal.get("execution_config_sha256")
                != current["execution_config_sha256"]
            ):
                raise ValueError("formal truth terminal provenance drifted")
            _sha256(
                terminal.get("state_identity_sha256"),
                label="formal truth state identity SHA256",
            )
            rows = terminal.get("distance_rows")
            if not isinstance(rows, list) or len(rows) != binding.get("coalition_count"):
                raise ValueError("formal truth terminal coalition count drifted")
            observed_rows += len(rows)
            observed_subsets = set()
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    raise ValueError("formal truth distance row is invalid")
                subset = _canonical_subset(
                    row.get("coalition_event_step_ids"),
                    candidates=state["_candidates"],
                    label=f"formal truth distance row {index}",
                )
                if subset in observed_subsets:
                    raise ValueError("formal truth terminal duplicates a coalition")
                observed_subsets.add(subset)
                distance = _finite_distance(
                    row.get("distance"), label="formal truth distance"
                )
                if subset == state["_candidates"] and distance != 0.0:
                    raise ValueError("formal truth full-history anchor is nonzero")
                _merge_distance(result[str(state_id)], subset, distance)
        actual_paths = {path.resolve() for path in (root / "states").glob("*.json")}
        if actual_paths != expected_paths or observed_rows != manifest.get(
            "coalition_count"
        ):
            raise ValueError("formal truth root no longer matches its sealed inventory")
        root_bindings.append(
            {
                "formal_manifest_content_sha256": manifest["content_sha256"],
                "formal_manifest_file_sha256": sha256_file(manifest_path),
                "formal_receipt_content_sha256": receipt["content_sha256"],
                "formal_receipt_file_sha256": sha256_file(receipt_path),
                "root": str(root),
                "state_count": manifest["state_count"],
                "coalition_count": manifest["coalition_count"],
            }
        )
    assert common_bindings is not None
    return result, {
        "roots": sorted(root_bindings, key=lambda value: value["root"]),
        "shared": common_bindings,
    }


def _validate_denominator(
    *,
    selector: Mapping[str, Any],
    heldout: Mapping[str, Any],
    states_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    heldout_rows = {
        row["state_id"]: row for row in heldout["checkpoint_states"]
    }
    if set(heldout_rows) != set(heldout["checkpoint_state_ids"]):
        raise ValueError("heldout checkpoint state inventory drifted")
    result = []
    for record in selector["records"]:
        state_id = record["state_id"]
        state = states_by_id.get(state_id)
        contract = heldout_rows.get(state_id)
        candidates = tuple(record["candidate_event_ids"])
        if (
            state is None
            or contract is None
            or state.get("role") != "train"
            or state.get("trajectory_id") != record["trajectory_id"]
            or contract.get("trajectory_id") != record["trajectory_id"]
            or state["_candidates"] != candidates
            or contract.get("candidate_count") != len(candidates)
            or contract.get("history_bin") != record["history_bin"]
        ):
            raise ValueError("selector/input/heldout state identity drifted")
        result.append(dict(record))
    if {row["state_id"] for row in result} != set(heldout_rows):
        raise ValueError("selector output escaped the frozen heldout denominator")
    return tuple(sorted(result, key=lambda value: value["state_id"]))


def _load_inputs(
    *,
    selector_output_path: Path,
    input_root: Path,
    heldout_manifest_path: Path,
    truth_roots: Sequence[Path],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    tuple[dict[str, Any], ...],
    dict[str, dict[tuple[int, ...], float]],
    dict[str, Any],
]:
    selector = _validate_selector_output(selector_output_path)
    input_manifest, states = _load_input_states(
        input_root,
        expected_content_sha256=selector["bindings"]["input_content_sha256"],
    )
    heldout = _load_heldout_manifest(
        heldout_manifest_path,
        expected_content_sha256=selector["bindings"][
            "heldout_manifest_content_sha256"
        ],
    )
    records = _validate_denominator(
        selector=selector, heldout=heldout, states_by_id=states
    )
    allowed = {record["state_id"] for record in records}
    truth, truth_bindings = load_sealed_truth_union(
        truth_roots, states_by_id=states, allowed_state_ids=allowed
    )
    shared = truth_bindings["shared"]
    if (
        shared["input_content_sha256"] != input_manifest["content_sha256"]
        or shared["input_manifest_file_sha256"]
        != sha256_file(input_root / "manifest.json")
        or selector["bindings"]["input_states_sha256"]
        != input_manifest["states_sha256"]
        or shared["heldout_manifest_content_sha256"] != heldout["content_sha256"]
        or shared["heldout_manifest_file_sha256"]
        != sha256_file(heldout_manifest_path)
        or selector["bindings"]["heldout_manifest_file_sha256"]
        != sha256_file(heldout_manifest_path)
        or selector["bindings"]["source_manifest_file_sha256"]
        != shared["source_manifest_file_sha256"]
    ):
        raise ValueError("formal truth roots disagree with selector inputs")
    return selector, heldout, states, records, truth, truth_bindings


def plan_post_selection_truth(
    *,
    selector_output_path: Path,
    input_root: Path,
    heldout_manifest_path: Path,
    truth_roots: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Write a selected-subset-only truth schedule for all fixed heldout states."""
    selector, _, states, records, truth, truth_bindings = _load_inputs(
        selector_output_path=selector_output_path,
        input_root=input_root,
        heldout_manifest_path=heldout_manifest_path,
        truth_roots=truth_roots,
    )
    schedule_records = []
    missing_count = 0
    for record in records:
        state_id = record["state_id"]
        sources: dict[tuple[int, ...], set[str]] = defaultdict(set)
        for method in METHODS:
            for budget in BUDGETS:
                subset = tuple(record["methods"][method][str(budget)])
                sources[subset].add(f"{method}:B{budget}")
        desired = sorted(sources, key=lambda value: (len(value), value))
        missing = [subset for subset in desired if subset not in truth[state_id]]
        missing_count += len(missing)
        state = states[state_id]
        schedule_records.append(
            {
                "candidate_event_ids": list(state["_candidates"]),
                "desired_coalitions": [list(subset) for subset in desired],
                "history_bin": record["history_bin"],
                "logical_shard": state["logical_shard"],
                "missing_coalitions": [
                    {
                        "event_ids": list(subset),
                        "source": "+".join(sorted(sources[subset])),
                    }
                    for subset in missing
                ],
                "state_id": state_id,
                "trajectory_id": record["trajectory_id"],
            }
        )
    checkpoint_sha = selector["checkpoints"]["set_transformer"]["sha256"]
    schedule = _pretty_signed(
        {
            "epoch_checkpoints": [
                {"checkpoint_sha256": checkpoint_sha, "epoch": 1}
            ],
            "epoch_count": 1,
            "epochs": [1],
            "missing_coalition_count": missing_count,
            "model_family": MODEL_FAMILY,
            "records": schedule_records,
            "schema_version": SCHEDULE_SCHEMA,
            "selected_checkpoint_sha256": checkpoint_sha,
            "selection_output_content_sha256": selector["content_sha256"],
            "selection_output_file_sha256": sha256_file(selector_output_path),
            "state_count": len(schedule_records),
            "status": (
                "COMPLETE_SET_TRANSFORMER_CONTROL_TRUTH"
                if missing_count == 0
                else "PENDING_SET_TRANSFORMER_CONTROL_TRUTH"
            ),
            "truth_roots": truth_bindings["roots"],
        }
    )
    _write_or_verify(output_path, schedule, pretty=True)
    return schedule


def _trajectory_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    budget: int | None,
    long_plus: bool,
) -> dict[str, float]:
    by_trajectory: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if long_plus and row["history_bin"] not in {"long", "very_long"}:
            continue
        budgets = (budget,) if budget is not None else BUDGETS
        for current in budgets:
            by_trajectory[row["trajectory_id"]].append(
                row["methods"][method][str(current)]["normalized_recovery"]
            )
    if not by_trajectory:
        raise ValueError("post-selection metric slice has no trajectories")
    return {
        trajectory: sum(values) / len(values)
        for trajectory, values in by_trajectory.items()
    }


def _mean(values: Mapping[str, float]) -> float:
    return sum(values.values()) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_bootstrap(
    challenger: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    resamples: int,
    seed: int,
    interval: float,
) -> dict[str, Any]:
    if set(challenger) != set(reference) or not challenger:
        raise ValueError("paired bootstrap trajectory inventory drifted")
    if resamples <= 0 or not 0.0 < interval < 1.0:
        raise ValueError("paired bootstrap parameters are invalid")
    trajectories = sorted(challenger)
    deltas = [challenger[name] - reference[name] for name in trajectories]
    observed = sum(deltas) / len(deltas)
    generator = random.Random(seed)
    samples = [
        sum(generator.choice(deltas) for _ in deltas) / len(deltas)
        for _ in range(resamples)
    ]
    alpha = (1.0 - interval) / 2.0
    return {
        "bootstrap_mean": sum(samples) / len(samples),
        "confidence_interval": [
            _quantile(samples, alpha),
            _quantile(samples, 1.0 - alpha),
        ],
        "observed_delta": observed,
        "probability_strictly_positive": sum(value > 0.0 for value in samples)
        / len(samples),
        "trajectory_count": len(trajectories),
    }


def reduce_post_selection_truth(
    *,
    selector_output_path: Path,
    input_root: Path,
    heldout_manifest_path: Path,
    truth_roots: Sequence[Path],
    output_path: Path,
    normalization_floor: float = 0.01,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260721,
    bootstrap_interval: float = 0.95,
) -> dict[str, Any]:
    """Reduce all selected methods on true distances and paired trajectories."""
    if not math.isfinite(normalization_floor) or normalization_floor <= 0.0:
        raise ValueError("normalization floor must be finite and positive")
    selector, _, _, records, truth, truth_bindings = _load_inputs(
        selector_output_path=selector_output_path,
        input_root=input_root,
        heldout_manifest_path=heldout_manifest_path,
        truth_roots=truth_roots,
    )
    state_rows = []
    missing = []
    for record in records:
        state_id = record["state_id"]
        distances = truth[state_id]
        if () not in distances:
            raise ValueError("post-selection truth has no empty anchor")
        denominator = max(distances[()], normalization_floor)
        methods = {
            "summary_only": {
                str(budget): {
                    "distance": distances[()],
                    "normalized_recovery": 0.0,
                    "selected_event_ids": [],
                }
                for budget in BUDGETS
            }
        }
        for method in METHODS:
            budgets = {}
            for budget in BUDGETS:
                subset = tuple(record["methods"][method][str(budget)])
                if subset not in distances:
                    missing.append((state_id, method, budget, subset))
                    continue
                budgets[str(budget)] = {
                    "distance": distances[subset],
                    "normalized_recovery": (
                        distances[()] - distances[subset]
                    )
                    / denominator,
                    "selected_event_ids": list(subset),
                }
            methods[method] = budgets
        state_rows.append(
            {
                "empty_distance": distances[()],
                "history_bin": record["history_bin"],
                "methods": methods,
                "normalization_denominator": denominator,
                "state_id": state_id,
                "trajectory_id": record["trajectory_id"],
            }
        )
    if missing:
        digest = sha256_json(
            [
                [state_id, method, budget, list(subset)]
                for state_id, method, budget, subset in missing
            ]
        )
        raise ValueError(
            f"post-selection truth is incomplete: count={len(missing)} sha256={digest}"
        )

    method_metrics = {}
    for method in REDUCED_METHODS:
        budget_metrics = {}
        for budget in BUDGETS:
            values = _trajectory_values(
                state_rows, method=method, budget=budget, long_plus=False
            )
            budget_metrics[str(budget)] = {
                "trajectory_count": len(values),
                "trajectory_equal_normalized_recovery": _mean(values),
            }
        macro = _trajectory_values(
            state_rows, method=method, budget=None, long_plus=False
        )
        long_macro = _trajectory_values(
            state_rows, method=method, budget=None, long_plus=True
        )
        method_metrics[method] = {
            "budgets": budget_metrics,
            "long_plus_trajectory_count": len(long_macro),
            "long_plus_trajectory_equal_B1_B4_macro": _mean(long_macro),
            "primary_trajectory_count": len(macro),
            "primary_trajectory_equal_B1_B4_macro": _mean(macro),
        }

    comparisons = {}
    for challenger_index in range(1, len(REDUCED_METHODS)):
        challenger = REDUCED_METHODS[challenger_index]
        for reference_index in range(challenger_index):
            reference = REDUCED_METHODS[reference_index]
            slices = {}
            for budget in BUDGETS:
                slices[f"B{budget}"] = _paired_bootstrap(
                    _trajectory_values(
                        state_rows,
                        method=challenger,
                        budget=budget,
                        long_plus=False,
                    ),
                    _trajectory_values(
                        state_rows,
                        method=reference,
                        budget=budget,
                        long_plus=False,
                    ),
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed + 100 * challenger_index + 10 * reference_index + budget,
                    interval=bootstrap_interval,
                )
            slices["primary_macro_B1_B4"] = _paired_bootstrap(
                _trajectory_values(
                    state_rows,
                    method=challenger,
                    budget=None,
                    long_plus=False,
                ),
                _trajectory_values(
                    state_rows,
                    method=reference,
                    budget=None,
                    long_plus=False,
                ),
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + 100 * challenger_index + 10 * reference_index,
                interval=bootstrap_interval,
            )
            slices["long_plus_macro_B1_B4"] = _paired_bootstrap(
                _trajectory_values(
                    state_rows,
                    method=challenger,
                    budget=None,
                    long_plus=True,
                ),
                _trajectory_values(
                    state_rows,
                    method=reference,
                    budget=None,
                    long_plus=True,
                ),
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + 100 * challenger_index + 10 * reference_index + 9,
                interval=bootstrap_interval,
            )
            comparisons[f"{challenger}_minus_{reference}"] = slices

    result = _compact_signed(
        {
            "bindings": {
                "heldout_manifest_content_sha256": selector["bindings"][
                    "heldout_manifest_content_sha256"
                ],
                "input_content_sha256": selector["bindings"][
                    "input_content_sha256"
                ],
                "selection_output_content_sha256": selector["content_sha256"],
                "selection_output_file_sha256": sha256_file(selector_output_path),
                "truth": truth_bindings,
            },
            "bootstrap": {
                "interval": bootstrap_interval,
                "method": "paired_trajectory_percentile",
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
            },
            "budgets": list(BUDGETS),
            "methods": method_metrics,
            "normalization_floor": normalization_floor,
            "paired_deltas": comparisons,
            "records": state_rows,
            "schema_version": RESULT_SCHEMA,
            "state_count": len(state_rows),
            "status": RESULT_STATUS,
            "trajectory_count": len({row["trajectory_id"] for row in state_rows}),
        }
    )
    _write_or_verify(output_path, result, pretty=False)
    return result


__all__ = [
    "BUDGETS",
    "EXPECTED_STATE_COUNT",
    "METHODS",
    "MODEL_FAMILY",
    "REDUCED_METHODS",
    "RESULT_SCHEMA",
    "RESULT_STATUS",
    "SCHEDULE_SCHEMA",
    "SELECTOR_SCHEMA",
    "SELECTOR_STATUS",
    "load_sealed_truth_union",
    "plan_post_selection_truth",
    "reduce_post_selection_truth",
]
