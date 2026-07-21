"""Selection-only truth scheduling and reduction for budget deferral v1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
from typing import Any

from causalcache.set_utility_budget_deferral_evaluation import (
    verify_selection_and_seal,
)
from causalcache.set_utility_heldout_evaluation import (
    COMPLETED_LABEL,
    SKIPPED_LABEL,
    canonical_json_bytes,
    paired_trajectory_bootstrap,
    sha256_file,
)
from causalcache.set_utility_train_heldout_contract import sha256_json
from causalcache.set_utility_variable_history import history_bin


SELECTION_SCHEMA = "causalcache.budget_deferral_eval_selections.v1"
SELECTION_STATUS = "SEALED_BUDGET_DEFERRAL_EVALUATION_SELECTIONS"
SEAL_SCHEMA = "causalcache.budget_deferral_eval_selection_seal.v1"
SEAL_STATUS = "SEALED_BUDGET_DEFERRAL_EVALUATION_SELECTION_RECEIPT"
SCHEDULE_SCHEMA = "causalcache.budget_deferral_eval_truth_schedule.v1"
SCHEDULE_STATUS = "FROZEN_BUDGET_DEFERRAL_EVALUATION_TRUTH_SCHEDULE"
SHARD_STATUS = "FROZEN_BUDGET_DEFERRAL_EVALUATION_TRUTH_SCHEDULE_SHARD"
RESULT_SCHEMA = "causalcache.budget_deferral_eval_truth_result.v1"
COMPLETED_RESULT_STATUS = "COMPLETED_BUDGET_DEFERRAL_EVALUATION_TRUTH"
INCOMPLETE_RESULT_STATUS = "INCOMPLETE_BUDGET_DEFERRAL_EVALUATION_TRUTH"
LOGICAL_SHARD_COUNT = 256
BUDGETS = (1, 2, 3, 4)
METHODS = ("recent", "structured_deepsets_budget_deferral")
SLICES = ("all", "state_new", "trajectory_new")
_BINDING_KEYS = (
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
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at {path}:{line_number}")
            rows.append(value)
    return tuple(rows)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing frozen output drifted: {path}")
        return
    _write_atomic(path, payload)


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    return {**unsigned, "content_sha256": sha256_json(unsigned)}


def _read_signed(path: Path, *, schema: str, status: str) -> dict[str, Any]:
    value = _read_json(path)
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256", None)
    if claimed != sha256_json(unsigned):
        raise ValueError(f"signed JSON content hash drifted: {path}")
    if value.get("schema_version") != schema or value.get("status") != status:
        raise ValueError(f"signed JSON schema or status drifted: {path}")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{label} is outside its finite domain")
    return result


def _canonical_subset(
    value: Any,
    *,
    candidates: tuple[int, ...],
    budget: int | None,
    label: str,
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
        or not set(subset).issubset(candidates)
        or (budget is not None and len(subset) > budget)
    ):
        raise ValueError(f"{label} is not a canonical at-most-B subset")
    return subset


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read_json(path)
    deployment = config.get("deployment")
    evaluation = config.get("frozen_candidate_evaluation")
    boundary = evaluation.get("historical_access_boundary") if isinstance(
        evaluation, Mapping
    ) else None
    firewall = config.get("firewall")
    predictor = config.get("predictor")
    if (
        config.get("schema_version") != "1.0.0"
        or config.get("name") != "causalcache_set_utility_budget_deferral_v1"
        or config.get("status") != "FROZEN_BEFORE_CANDIDATE_EVALUATION"
        or not isinstance(deployment, Mapping)
        or tuple(deployment.get("budgets", ())) != BUDGETS
        or deployment.get("selection_by_budget")
        != {
            "1": "recent",
            "2": "recent",
            "3": "deepsets_direct_conditional_marginal",
            "4": "deepsets_direct_conditional_marginal",
        }
        or deployment.get("selection_cardinality") != "at_most_B"
        or deployment.get("post_evaluation_route_or_threshold_changes_allowed")
        is not False
        or not isinstance(evaluation, Mapping)
        or evaluation.get("role") != "evaluation"
        or evaluation.get("label_blind_selection_must_be_sealed_before_truth_access")
        is not True
        or not isinstance(boundary, Mapping)
        or not isinstance(firewall, Mapping)
        or firewall.get("evaluation_features_allowed_after_this_freeze") is not True
        or firewall.get("evaluation_truth_allowed_only_after_signed_selection_seal")
        is not True
        or firewall.get("evaluation_truth_for_training_or_calibration") is not False
        or not isinstance(predictor, Mapping)
        or predictor.get("model_family") != "deepsets_structured_marginal"
    ):
        raise ValueError("budget-deferral config is not the frozen candidate")
    state_count = _positive_int(evaluation.get("state_count"), label="state count")
    history_counts = evaluation.get("history_bin_counts")
    if (
        not isinstance(history_counts, Mapping)
        or set(history_counts) != {"short", "medium", "long", "very_long"}
        or any(type(value) is not int or value < 0 for value in history_counts.values())
        or sum(history_counts.values()) != state_count
    ):
        raise ValueError("frozen evaluation history-bin counts drifted")
    state_new = _positive_int(boundary.get("state_new_count"), label="state-new count")
    trajectory_new_states = _positive_int(
        boundary.get("trajectory_new_state_count"),
        label="trajectory-new state count",
    )
    trajectory_new_trajectories = _positive_int(
        boundary.get("trajectory_new_trajectory_count"),
        label="trajectory-new trajectory count",
    )
    previous_trajectories = _positive_int(
        boundary.get("previously_consumed_trajectory_count"),
        label="previous trajectory count",
    )
    if trajectory_new_states > state_new or state_new > state_count:
        raise ValueError("frozen evaluation slice counts are inconsistent")
    _sha(evaluation.get("state_inventory_sha256"), label="state inventory SHA256")
    checkpoint_sha = _sha(predictor.get("checkpoint_sha256"), label="checkpoint SHA256")
    identity = {
        "checkpoint_sha256": checkpoint_sha,
        "config_content_sha256": sha256_json(config),
        "config_file_sha256": sha256_file(path),
        "expected_history_bin_counts": dict(history_counts),
        "expected_slice_state_counts": {
            "all": state_count,
            "state_new": state_new,
            "trajectory_new": trajectory_new_states,
        },
        "expected_all_trajectory_count": (
            previous_trajectories + trajectory_new_trajectories
        ),
        "expected_trajectory_new_trajectory_count": trajectory_new_trajectories,
        "state_inventory_content_sha256": evaluation[
            "state_inventory_sha256"
        ],
    }
    return config, identity


def _validate_slice_counts(
    value: Any,
    *,
    records: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or set(value) != set(SLICES):
        raise ValueError("selection slice-count inventory drifted")
    observed = {}
    for name in SLICES:
        rows = tuple(row for row in records if name in row["slices"])
        counts = {
            "state_count": len(rows),
            "trajectory_count": len({row["trajectory_id"] for row in rows}),
        }
        if value.get(name) != counts:
            raise ValueError(f"selection {name} slice counts drifted")
        if counts["state_count"] != identity["expected_slice_state_counts"][name]:
            raise ValueError(f"selection {name} state denominator drifted")
        observed[name] = counts
    if observed["all"]["trajectory_count"] != identity[
        "expected_all_trajectory_count"
    ]:
        raise ValueError("selection all-trajectory denominator drifted")
    if observed["trajectory_new"]["trajectory_count"] != identity[
        "expected_trajectory_new_trajectory_count"
    ]:
        raise ValueError("selection trajectory-new denominator drifted")
    return observed


def _load_selection(
    path: Path,
    *,
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    selection = _read_signed(path, schema=SELECTION_SCHEMA, status=SELECTION_STATUS)
    bindings = selection.get("bindings")
    if (
        selection.get("truth_accessed") is not False
        or tuple(selection.get("methods", ())) != METHODS
        or selection.get("method_semantics")
        != {
            "recent": "deterministic_most_recent_at_most_B",
            "structured_deepsets_budget_deferral": (
                "B1_B2_recent_B3_B4_direct_conditional_marginal"
            ),
        }
        or not isinstance(bindings, Mapping)
        or set(bindings) != set(_BINDING_KEYS)
    ):
        raise ValueError("selection firewall or binding inventory drifted")
    for key in _BINDING_KEYS:
        _sha(bindings.get(key), label=f"selection {key}")
    for key in (
        "checkpoint_sha256",
        "config_content_sha256",
        "config_file_sha256",
        "state_inventory_content_sha256",
    ):
        if bindings[key] != identity[key]:
            raise ValueError(f"selection {key} does not bind the frozen config")
    raw_records = selection.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("selection records are missing")
    records = []
    seen = set()
    for ordinal, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"selection record {ordinal} is invalid")
        required = {
            "candidate_event_ids",
            "history_bin",
            "latency_ms",
            "logical_shard",
            "methods",
            "slices",
            "state_id",
            "trajectory_id",
        }
        if set(raw) != required:
            raise ValueError(f"selection record {ordinal} keys drifted")
        state_id = raw["state_id"]
        trajectory_id = raw["trajectory_id"]
        logical_shard = raw["logical_shard"]
        candidates = tuple(raw["candidate_event_ids"])
        if (
            not isinstance(state_id, str)
            or not isinstance(trajectory_id, str)
            or state_id in seen
            or type(logical_shard) is not int
            or not 0 <= logical_shard < LOGICAL_SHARD_COUNT
            or candidates != tuple(range(1, len(candidates) + 1))
            or len(candidates) < 5
            or state_id != f"{trajectory_id}:decision:{len(candidates) + 1:03d}"
            or raw["history_bin"] != history_bin(len(candidates))
        ):
            raise ValueError(f"selection record {ordinal} identity drifted")
        seen.add(state_id)
        slices = tuple(raw["slices"])
        expected_order = tuple(name for name in SLICES if name in slices)
        if (
            not slices
            or slices != expected_order
            or slices[0] != "all"
            or len(slices) != len(set(slices))
            or ("trajectory_new" in slices and "state_new" not in slices)
        ):
            raise ValueError(f"selection record {ordinal} slices drifted")
        methods = raw["methods"]
        if not isinstance(methods, Mapping) or set(methods) != set(METHODS):
            raise ValueError(f"selection record {ordinal} methods drifted")
        normalized_methods = {}
        for method in METHODS:
            by_budget = methods[method]
            if not isinstance(by_budget, Mapping) or set(by_budget) != {
                str(budget) for budget in BUDGETS
            }:
                raise ValueError(f"selection {method} budget inventory drifted")
            normalized_methods[method] = {
                str(budget): list(
                    _canonical_subset(
                        by_budget[str(budget)],
                        candidates=candidates,
                        budget=budget,
                        label=f"{state_id} {method} B{budget}",
                    )
                )
                for budget in BUDGETS
            }
        for budget in BUDGETS:
            if tuple(normalized_methods["recent"][str(budget)]) != candidates[-budget:]:
                raise ValueError(f"selection recent B{budget} is not recent-B")
        for budget in (1, 2):
            if normalized_methods["structured_deepsets_budget_deferral"][
                str(budget)
            ] != normalized_methods["recent"][str(budget)]:
                raise ValueError("budget-deferral B1/B2 must exactly equal recent")
        latency = raw["latency_ms"]
        if not isinstance(latency, Mapping):
            raise ValueError("selection latency metadata is invalid")
        for name, value in latency.items():
            if not isinstance(name, str):
                raise ValueError("selection latency key is invalid")
            _finite(value, label=f"selection latency {name}", nonnegative=True)
        records.append(
            {
                **raw,
                "candidate_event_ids": list(candidates),
                "methods": normalized_methods,
                "slices": list(slices),
            }
        )
    records.sort(key=lambda row: row["state_id"])
    denominator = selection.get("denominator")
    if not isinstance(denominator, Mapping):
        raise ValueError("selection denominator is missing")
    if denominator.get("state_count") != len(records) or denominator.get(
        "trajectory_count"
    ) != len({row["trajectory_id"] for row in records}):
        raise ValueError("selection denominator counts drifted")
    if denominator.get("history_bin_counts") != identity[
        "expected_history_bin_counts"
    ]:
        raise ValueError("selection history-bin denominator drifted")
    observed_history = Counter(row["history_bin"] for row in records)
    if dict(observed_history) != identity["expected_history_bin_counts"]:
        raise ValueError("selection record history-bin counts drifted")
    _validate_slice_counts(
        denominator.get("slice_counts"), records=records, identity=identity
    )
    trajectory_new_ids = {
        row["trajectory_id"] for row in records if "trajectory_new" in row["slices"]
    }
    if any(
        (row["trajectory_id"] in trajectory_new_ids)
        != ("trajectory_new" in row["slices"])
        for row in records
    ):
        raise ValueError("trajectory-new tags are not trajectory-consistent")
    return selection, tuple(records)


def _load_seal(
    path: Path,
    *,
    selection_path: Path,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    seal = _read_signed(path, schema=SEAL_SCHEMA, status=SEAL_STATUS)
    if (
        seal.get("truth_accessed") is not False
        or seal.get("selection_file_sha256") != sha256_file(selection_path)
        or seal.get("selection_content_sha256") != selection["content_sha256"]
        or seal.get("bindings") != selection["bindings"]
        or seal.get("denominator") != selection["denominator"]
    ):
        raise ValueError("selection seal does not bind the frozen selection")
    return seal


def _load_contract(
    *,
    config_path: Path,
    selection_path: Path,
    selection_seal_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    verified_selection, verified_seal = verify_selection_and_seal(
        selection_path, selection_seal_path
    )
    config, identity = _load_config(config_path)
    selection, records = _load_selection(selection_path, identity=identity)
    seal = _load_seal(
        selection_seal_path,
        selection_path=selection_path,
        selection=selection,
    )
    if selection != verified_selection or seal != verified_seal:
        raise ValueError("selection validator disagreement")
    return config, selection, seal, records, identity


def _truth_sources(record: Mapping[str, Any]) -> dict[tuple[int, ...], set[str]]:
    candidates = tuple(record["candidate_event_ids"])
    sources: dict[tuple[int, ...], set[str]] = defaultdict(set)
    sources[()].add("anchor:empty")
    for budget in BUDGETS:
        subset = tuple(record["methods"]["recent"][str(budget)])
        sources[subset].add(f"recent:B{budget}")
    for budget in (3, 4):
        subset = tuple(
            record["methods"]["structured_deepsets_budget_deferral"][str(budget)]
        )
        sources[subset].add(f"structured_deepsets_budget_deferral:B{budget}")
    sources[candidates].add("anchor:full")
    return sources


def _schedule_record(record: Mapping[str, Any]) -> dict[str, Any]:
    sources = _truth_sources(record)
    coalitions = [
        {
            "event_ids": list(subset),
            "source": "+".join(sorted(tags)),
            "sources": sorted(tags),
        }
        for subset, tags in sorted(
            sources.items(), key=lambda item: (len(item[0]), item[0])
        )
    ]
    return {
        "candidate_event_ids": record["candidate_event_ids"],
        "coalitions": coalitions,
        "logical_shard": record["logical_shard"],
        "role": "evaluation",
        "state_id": record["state_id"],
        "tracks": ["budget_deferral_selected_only", *record["slices"]],
        "trajectory_id": record["trajectory_id"],
    }


def _schedule_bindings(
    *,
    config_path: Path,
    selection_path: Path,
    selection_seal_path: Path,
    selection: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "config_content_sha256": selection["bindings"]["config_content_sha256"],
        "config_file_sha256": sha256_file(config_path),
        "selection_content_sha256": selection["content_sha256"],
        "selection_file_sha256": sha256_file(selection_path),
        "selection_seal_content_sha256": seal["content_sha256"],
        "selection_seal_file_sha256": sha256_file(selection_seal_path),
    }


def _schedule_payloads(
    *,
    config_path: Path,
    selection_path: Path,
    selection_seal_path: Path,
) -> tuple[
    dict[str, Any],
    dict[int, bytes],
    dict[int, dict[str, Any]],
    tuple[dict[str, Any], ...],
]:
    _, selection, seal, records, _ = _load_contract(
        config_path=config_path,
        selection_path=selection_path,
        selection_seal_path=selection_seal_path,
    )
    bindings = _schedule_bindings(
        config_path=config_path,
        selection_path=selection_path,
        selection_seal_path=selection_seal_path,
        selection=selection,
        seal=seal,
    )
    by_shard: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(LOGICAL_SHARD_COUNT)
    }
    schedule_records = tuple(_schedule_record(record) for record in records)
    for record in schedule_records:
        by_shard[record["logical_shard"]].append(record)
    payloads: dict[int, bytes] = {}
    receipts = {}
    for logical_shard in range(LOGICAL_SHARD_COUNT):
        rows = sorted(by_shard[logical_shard], key=lambda row: row["state_id"])
        payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        coalition_count = sum(len(row["coalitions"]) for row in rows)
        forward_count = sum(
            tuple(coalition["event_ids"]) != tuple(row["candidate_event_ids"])
            for row in rows
            for coalition in row["coalitions"]
        )
        payloads[logical_shard] = payload
        receipts[logical_shard] = _signed(
            {
                "bindings": bindings,
                "coalition_count": coalition_count,
                "forward_coalition_count": forward_count,
                "logical_shard": logical_shard,
                "role_state_counts": {"evaluation": len(rows)} if rows else {},
                "schedule_byte_count": len(payload),
                "schedule_sha256": hashlib.sha256(payload).hexdigest(),
                "schema_version": SCHEDULE_SCHEMA,
                "state_count": len(rows),
                "status": SHARD_STATUS,
            }
        )
    summary = _signed(
        {
            "bindings": bindings,
            "coalition_count": sum(
                receipt["coalition_count"] for receipt in receipts.values()
            ),
            "denominator": selection["denominator"],
            "forward_coalition_count": sum(
                receipt["forward_coalition_count"] for receipt in receipts.values()
            ),
            "logical_shard_count": LOGICAL_SHARD_COUNT,
            "schema_version": SCHEDULE_SCHEMA,
            "selected_truth_sources": [
                "anchor:empty",
                "recent:B1-B4",
                "structured_deepsets_budget_deferral:B3-B4",
                "anchor:full",
            ],
            "state_count": len(records),
            "status": SCHEDULE_STATUS,
            "trajectory_count": len({row["trajectory_id"] for row in records}),
        }
    )
    return summary, payloads, receipts, schedule_records


def materialize_budget_deferral_truth_schedule(
    *,
    config_path: Path,
    selection_path: Path,
    selection_seal_path: Path,
    output_root: Path,
    workers: int = 32,
) -> dict[str, Any]:
    """Materialize the selected-only evaluation schedule after the selection seal."""
    if workers <= 0:
        raise ValueError("schedule worker count must be positive")
    summary, payloads, receipts, _ = _schedule_payloads(
        config_path=config_path,
        selection_path=selection_path,
        selection_seal_path=selection_seal_path,
    )

    def write(logical_shard: int) -> None:
        schedule_path = (
            output_root
            / "schedule-shards"
            / f"shard-{logical_shard:03d}-of-{LOGICAL_SHARD_COUNT}.jsonl"
        )
        receipt_path = (
            output_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-{LOGICAL_SHARD_COUNT}.json"
        )
        _write_or_verify(schedule_path, payloads[logical_shard])
        _write_or_verify(
            receipt_path,
            canonical_json_bytes(receipts[logical_shard], pretty=True) + b"\n",
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        tuple(executor.map(write, range(LOGICAL_SHARD_COUNT)))
    _write_or_verify(
        output_root / "summary.json",
        canonical_json_bytes(summary, pretty=True) + b"\n",
    )
    return summary


def _validate_schedule_root(
    *,
    config_path: Path,
    selection_path: Path,
    selection_seal_path: Path,
    schedule_root: Path,
) -> tuple[
    dict[str, Any],
    tuple[dict[str, Any], ...],
    dict[int, str],
]:
    expected_summary, payloads, receipts, records = _schedule_payloads(
        config_path=config_path,
        selection_path=selection_path,
        selection_seal_path=selection_seal_path,
    )
    summary_path = schedule_root / "summary.json"
    summary = _read_signed(
        summary_path, schema=SCHEDULE_SCHEMA, status=SCHEDULE_STATUS
    )
    if summary != expected_summary:
        raise ValueError("truth schedule summary drifted from sealed selection")
    receipt_file_shas = {}
    for logical_shard in range(LOGICAL_SHARD_COUNT):
        schedule_path = (
            schedule_root
            / "schedule-shards"
            / f"shard-{logical_shard:03d}-of-{LOGICAL_SHARD_COUNT}.jsonl"
        )
        receipt_path = (
            schedule_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-{LOGICAL_SHARD_COUNT}.json"
        )
        if not schedule_path.is_file() or schedule_path.read_bytes() != payloads[
            logical_shard
        ]:
            raise ValueError("truth schedule shard drifted from sealed selection")
        receipt = _read_signed(
            receipt_path, schema=SCHEDULE_SCHEMA, status=SHARD_STATUS
        )
        if receipt != receipts[logical_shard]:
            raise ValueError("truth schedule receipt drifted from sealed selection")
        receipt_file_shas[logical_shard] = sha256_file(receipt_path)
    return summary, records, receipt_file_shas


def _source_shards(
    path: Path,
) -> tuple[dict[str, int], dict[int, str], dict[str, Any]]:
    manifest = _read_json(path)
    shards = manifest.get("shards")
    if (
        manifest.get("status") != "COMPLETED_VARIABLE_HISTORY_SOURCE"
        or not isinstance(shards, list)
        or len(shards) != LOGICAL_SHARD_COUNT
    ):
        raise ValueError("source manifest is not the complete 256-shard source")
    trajectories = {}
    shard_hashes = {}
    for row in shards:
        if not isinstance(row, Mapping):
            raise ValueError("source shard row is invalid")
        logical = row.get("logical_shard")
        if type(logical) is not int or not 0 <= logical < LOGICAL_SHARD_COUNT:
            raise ValueError("source logical shard is invalid")
        if logical in shard_hashes:
            raise ValueError("source logical shard is duplicated")
        shard_hashes[logical] = _sha(row.get("sha256"), label="source shard SHA256")
        ids = row.get("trajectory_ids")
        if not isinstance(ids, list):
            raise ValueError("source trajectory inventory is invalid")
        for trajectory_id in ids:
            if not isinstance(trajectory_id, str) or trajectory_id in trajectories:
                raise ValueError("source trajectory identity is invalid")
            trajectories[trajectory_id] = logical
    if set(shard_hashes) != set(range(LOGICAL_SHARD_COUNT)):
        raise ValueError("source shard inventory is incomplete")
    return trajectories, shard_hashes, manifest


def _runner_state_identity(
    *,
    schedule: Mapping[str, Any],
    execution_config_sha256: str,
    scientific_config_sha256: str,
    schedule_receipt_sha256: str,
    source_revision: str,
    source_shard_sha256: str,
) -> str:
    payload = {
        "candidate_event_ids": schedule["candidate_event_ids"],
        "coalitions": schedule["coalitions"],
        "execution_config_sha256": execution_config_sha256,
        "scientific_config_sha256": scientific_config_sha256,
        "schedule_receipt_sha256": schedule_receipt_sha256,
        "source_revision": source_revision,
        "source_shard_sha256": source_shard_sha256,
        "state_id": schedule["state_id"],
    }
    return hashlib.sha256(canonical_json_bytes(payload, pretty=True) + b"\n").hexdigest()


def _terminal_files(
    label_roots: Sequence[Path], *, expected_state_ids: set[str]
) -> tuple[dict[str, tuple[Path, dict[str, Any]]], int]:
    if not label_roots:
        raise ValueError("at least one label root is required")
    terminals = {}
    ignored = 0
    for root in label_roots:
        states_root = root / "states"
        if not states_root.is_dir():
            raise FileNotFoundError(states_root)
        for path in sorted(states_root.glob("*.json")):
            terminal = _read_json(path)
            state_id = terminal.get("state_id")
            if not isinstance(state_id, str):
                raise ValueError("truth terminal has no state identity")
            if state_id not in expected_state_ids:
                ignored += 1
                continue
            if state_id in terminals:
                raise ValueError("truth terminal state is duplicated")
            terminals[state_id] = (path, terminal)
    return terminals, ignored


def _trajectory_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    budgets: Sequence[int],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for budget in budgets:
            grouped[row["trajectory_id"]].append(
                row["methods"][method][str(budget)]["normalized_recovery"]
            )
    return {
        trajectory_id: fmean(values)
        for trajectory_id, values in sorted(grouped.items())
    }


def _coverage(
    *,
    records: Sequence[Mapping[str, Any]],
    completed_ids: set[str],
    skipped: Mapping[str, str],
    missing: set[str],
) -> dict[str, Any]:
    expected_ids = {row["state_id"] for row in records}
    expected_by_trajectory: dict[str, set[str]] = defaultdict(set)
    for row in records:
        expected_by_trajectory[row["trajectory_id"]].add(row["state_id"])
    complete_trajectories = sum(
        ids.issubset(completed_ids) for ids in expected_by_trajectory.values()
    )
    partial_trajectories = sum(
        bool(ids & completed_ids) and not ids.issubset(completed_ids)
        for ids in expected_by_trajectory.values()
    )
    return {
        "complete": expected_ids.issubset(completed_ids),
        "completed_history_bin_counts": dict(
            sorted(
                Counter(
                    row["history_bin"]
                    for row in records
                    if row["state_id"] in completed_ids
                ).items()
            )
        ),
        "completed_state_count": len(expected_ids & completed_ids),
        "completed_trajectory_count": len(
            {
                row["trajectory_id"]
                for row in records
                if row["state_id"] in completed_ids
            }
        ),
        "expected_history_bin_counts": dict(
            sorted(Counter(row["history_bin"] for row in records).items())
        ),
        "expected_state_count": len(expected_ids),
        "expected_trajectory_count": len(expected_by_trajectory),
        "fully_completed_trajectory_count": complete_trajectories,
        "missing_state_count": len(expected_ids & missing),
        "missing_state_ids": sorted(expected_ids & missing),
        "partially_completed_trajectory_count": partial_trajectories,
        "skip_failure_class_counts": dict(
            sorted(Counter(skipped[state_id] for state_id in expected_ids & skipped.keys()).items())
        ),
        "skipped_state_count": len(expected_ids & skipped.keys()),
        "skipped_states": {
            state_id: skipped[state_id]
            for state_id in sorted(expected_ids & skipped.keys())
        },
    }


def _slice_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    bootstrap_interval: float,
) -> dict[str, Any]:
    if not rows:
        return {"available": False, "reason": "NO_COMPLETED_STATES"}
    methods = {}
    for method in METHODS:
        budget_metrics = {}
        for budget in BUDGETS:
            values = _trajectory_values(rows, method=method, budgets=(budget,))
            budget_metrics[str(budget)] = {
                "trajectory_count": len(values),
                "trajectory_equal_normalized_recovery": fmean(values.values()),
            }
        macro = _trajectory_values(rows, method=method, budgets=BUDGETS)
        methods[method] = {
            "budgets": budget_metrics,
            "trajectory_count": len(macro),
            "trajectory_equal_B1_B4_macro": fmean(macro.values()),
        }
    paired = {}
    for index, budgets in enumerate(((1,), (2,), (3,), (4,), BUDGETS), start=1):
        challenger = _trajectory_values(
            rows,
            method="structured_deepsets_budget_deferral",
            budgets=budgets,
        )
        reference = _trajectory_values(rows, method="recent", budgets=budgets)
        if set(challenger) != set(reference):
            raise ValueError("paired method trajectory inventory drifted")
        name = f"B{budgets[0]}" if len(budgets) == 1 else "macro_B1_B4"
        paired[name] = paired_trajectory_bootstrap(
            {
                trajectory_id: challenger[trajectory_id] - reference[trajectory_id]
                for trajectory_id in challenger
            },
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + index,
            interval=bootstrap_interval,
        )
    return {
        "available": True,
        "methods": methods,
        "paired_delta": {
            "structured_deepsets_budget_deferral_minus_recent": paired
        },
    }


def reduce_budget_deferral_truth(
    *,
    config_path: Path,
    selection_path: Path,
    selection_seal_path: Path,
    schedule_root: Path,
    source_manifest_path: Path,
    label_roots: Sequence[Path],
    output_path: Path,
    normalization_floor: float = 0.01,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260721,
    bootstrap_interval: float = 0.95,
) -> dict[str, Any]:
    """Reduce selected truth with transparent coverage on three frozen slices."""
    floor = _finite(
        normalization_floor, label="normalization floor", nonnegative=True
    )
    if floor <= 0.0:
        raise ValueError("normalization floor must be positive")
    if bootstrap_resamples <= 0 or not 0.0 < bootstrap_interval < 1.0:
        raise ValueError("bootstrap settings are invalid")
    _, selection, seal, selection_records, _ = _load_contract(
        config_path=config_path,
        selection_path=selection_path,
        selection_seal_path=selection_seal_path,
    )
    summary, schedule_records, receipt_shas = _validate_schedule_root(
        config_path=config_path,
        selection_path=selection_path,
        selection_seal_path=selection_seal_path,
        schedule_root=schedule_root,
    )
    source_trajectories, shard_hashes, source_manifest = _source_shards(
        source_manifest_path
    )
    selection_by_id = {row["state_id"]: row for row in selection_records}
    schedule_by_id = {row["state_id"]: row for row in schedule_records}
    expected_ids = set(selection_by_id)
    terminals, ignored_terminal_count = _terminal_files(
        label_roots, expected_state_ids=expected_ids
    )
    completed_rows = []
    completed_ids = set()
    skipped = {}
    missing = set()
    terminal_hashes = {}
    scientific_shas = set()
    execution_shas = set()
    source_revisions = set()
    for state_id in sorted(expected_ids):
        selection_record = selection_by_id[state_id]
        schedule = schedule_by_id[state_id]
        terminal_item = terminals.get(state_id)
        if terminal_item is None:
            missing.add(state_id)
            continue
        path, terminal = terminal_item
        terminal_hashes[state_id] = sha256_file(path)
        trajectory_id = selection_record["trajectory_id"]
        logical_shard = selection_record["logical_shard"]
        if source_trajectories.get(trajectory_id) != logical_shard:
            raise ValueError(f"source shard drifted for {state_id}")
        scientific_sha = _sha(
            terminal.get("scientific_config_sha256"),
            label="truth scientific config SHA256",
        )
        execution_sha = _sha(
            terminal.get("execution_config_sha256"),
            label="truth execution config SHA256",
        )
        source_revision = terminal.get("source_revision")
        if not isinstance(source_revision, str) or _REVISION.fullmatch(
            source_revision
        ) is None:
            raise ValueError("truth source revision drifted")
        expected_identity = _runner_state_identity(
            schedule=schedule,
            execution_config_sha256=execution_sha,
            scientific_config_sha256=scientific_sha,
            schedule_receipt_sha256=receipt_shas[logical_shard],
            source_revision=source_revision,
            source_shard_sha256=shard_hashes[logical_shard],
        )
        if (
            terminal.get("state_id") != state_id
            or terminal.get("trajectory_id") != trajectory_id
            or terminal.get("role") != "evaluation"
            or terminal.get("state_identity_sha256") != expected_identity
        ):
            raise ValueError(f"truth terminal identity drifted for {state_id}")
        scientific_shas.add(scientific_sha)
        execution_shas.add(execution_sha)
        source_revisions.add(source_revision)
        status = terminal.get("status")
        if status == SKIPPED_LABEL:
            skipped[state_id] = str(terminal.get("failure_class", "UnknownFailure"))
            continue
        if status != COMPLETED_LABEL:
            raise ValueError(f"truth terminal status is invalid for {state_id}")
        candidates = tuple(selection_record["candidate_event_ids"])
        if tuple(terminal.get("candidate_event_step_ids", ())) != candidates:
            raise ValueError(f"truth candidate universe drifted for {state_id}")
        rows = terminal.get("distance_rows")
        if not isinstance(rows, list):
            raise ValueError(f"truth distance rows are missing for {state_id}")
        expected_subsets = tuple(
            tuple(row["event_ids"]) for row in schedule["coalitions"]
        )
        distances = {}
        observed = []
        for ordinal, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"truth distance row is invalid for {state_id}")
            subset = _canonical_subset(
                row.get("coalition_event_step_ids"),
                candidates=candidates,
                budget=None,
                label=f"{state_id} truth row {ordinal}",
            )
            if subset in distances:
                raise ValueError(f"truth subset is duplicated for {state_id}")
            distances[subset] = _finite(
                row.get("distance"),
                label=f"{state_id} distance {subset}",
                nonnegative=True,
            )
            observed.append(subset)
        if tuple(observed) != expected_subsets:
            raise ValueError(f"truth selected-only table drifted for {state_id}")
        if () not in distances or candidates not in distances:
            raise ValueError(f"truth anchors are missing for {state_id}")
        if distances[candidates] != 0.0:
            raise ValueError(f"truth full-history anchor is nonzero for {state_id}")
        empty_distance = distances[()]
        denominator = max(empty_distance, floor)
        method_metrics = {}
        for method in METHODS:
            method_metrics[method] = {}
            for budget in BUDGETS:
                subset = tuple(selection_record["methods"][method][str(budget)])
                distance = distances[subset]
                utility = empty_distance - distance
                method_metrics[method][str(budget)] = {
                    "distance": distance,
                    "normalized_recovery": utility / denominator,
                    "selected_cardinality": len(subset),
                    "selected_event_ids": list(subset),
                    "utility": utility,
                }
        completed_ids.add(state_id)
        completed_rows.append(
            {
                "empty_distance": empty_distance,
                "history_bin": selection_record["history_bin"],
                "methods": method_metrics,
                "normalization_denominator": denominator,
                "slices": selection_record["slices"],
                "state_id": state_id,
                "trajectory_id": trajectory_id,
            }
        )
    if len(scientific_shas) > 1 or len(execution_shas) > 1 or len(source_revisions) > 1:
        raise ValueError("truth terminals disagree on reference provenance")
    slices = {}
    for index, name in enumerate(SLICES):
        expected = tuple(row for row in selection_records if name in row["slices"])
        rows = tuple(row for row in completed_rows if name in row["slices"])
        slices[name] = {
            "coverage": _coverage(
                records=expected,
                completed_ids=completed_ids,
                skipped=skipped,
                missing=missing,
            ),
            "metrics": _slice_metrics(
                rows,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed + 100 * index,
                bootstrap_interval=bootstrap_interval,
            ),
        }
    complete = slices["all"]["coverage"]["complete"]
    result = _signed(
        {
            "bindings": {
                **summary["bindings"],
                "label_roots": [str(path.resolve()) for path in label_roots],
                "schedule_summary_content_sha256": summary["content_sha256"],
                "schedule_summary_file_sha256": sha256_file(
                    schedule_root / "summary.json"
                ),
                "source_manifest_content_sha256": source_manifest.get(
                    "content_sha256"
                ),
                "source_manifest_file_sha256": sha256_file(source_manifest_path),
                "terminal_inventory_sha256": sha256_json(terminal_hashes),
            },
            "bootstrap": {
                "interval": bootstrap_interval,
                "method": "paired_trajectory_percentile",
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
            },
            "budgets": list(BUDGETS),
            "coverage_complete": complete,
            "ignored_out_of_denominator_terminal_count": ignored_terminal_count,
            "methods": list(METHODS),
            "normalization_floor": floor,
            "provenance": {
                "execution_config_sha256": next(iter(execution_shas), None),
                "scientific_config_sha256": next(iter(scientific_shas), None),
                "source_revision": next(iter(source_revisions), None),
            },
            "records": completed_rows,
            "schema_version": RESULT_SCHEMA,
            "selection_content_sha256": selection["content_sha256"],
            "selection_seal_content_sha256": seal["content_sha256"],
            "slices": slices,
            "status": (
                COMPLETED_RESULT_STATUS if complete else INCOMPLETE_RESULT_STATUS
            ),
        }
    )
    _write_or_verify(output_path, canonical_json_bytes(result) + b"\n")
    return result


__all__ = [
    "BUDGETS",
    "COMPLETED_RESULT_STATUS",
    "INCOMPLETE_RESULT_STATUS",
    "METHODS",
    "RESULT_SCHEMA",
    "SCHEDULE_SCHEMA",
    "SCHEDULE_STATUS",
    "SEAL_SCHEMA",
    "SEAL_STATUS",
    "SELECTION_SCHEMA",
    "SELECTION_STATUS",
    "SHARD_STATUS",
    "SLICES",
    "materialize_budget_deferral_truth_schedule",
    "reduce_budget_deferral_truth",
]
