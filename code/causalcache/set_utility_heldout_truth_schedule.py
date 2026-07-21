"""Materialize shared heldout truth schedules from model rollout plans."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from causalcache.set_utility_direct_on_policy import runner_schedule_coalitions
from causalcache.set_utility_heldout_evaluation import canonical_json_bytes, sha256_file
from causalcache.set_utility_train_heldout_contract import sha256_json
from causalcache.set_utility_variable_history import history_bin


SCHEDULE_STATUS = "COMPLETED_SET_UTILITY_HELDOUT_TRUTH_SCHEDULE"
SHARD_STATUS = "COMPLETED_SET_UTILITY_HELDOUT_TRUTH_SCHEDULE_SHARD"
FORMAL_MANIFEST_STATUS = "COMPLETED_SET_UTILITY_FORMAL_TRUTH_MANIFEST"
FORMAL_RECEIPT_STATUS = "COMPLETED_SET_UTILITY_FORMAL_TRUTH_RECEIPT"
LABEL_STATUS = "COMPLETED_VARIABLE_HISTORY_LABEL_STATE"
EXPECTED_STATE_COUNT = 256
EXPECTED_SHARD_COUNT = 256
MODEL_FAMILIES = (
    "deepsets_structured_marginal",
    "set_transformer_direct_marginal",
)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _lower_sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


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
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return tuple(rows)


def _signed_compact(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    return {**unsigned, "content_sha256": sha256_json(unsigned)}


def _validate_compact_signature(
    path: Path, *, schema_version: str | None = None
) -> tuple[dict[str, Any], str]:
    value = _read_json(path)
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256", None)
    if claimed != sha256_json(unsigned):
        raise ValueError(f"signed JSON content hash drifted: {path}")
    if schema_version is not None and value.get("schema_version") != schema_version:
        raise ValueError(f"signed JSON schema drifted: {path}")
    return value, str(claimed)


def _epoch_checkpoint_bindings(schedule: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    family = schedule.get("model_family")
    if family not in MODEL_FAMILIES:
        raise ValueError("heldout truth schedule model family drifted")
    epochs = schedule.get("epochs")
    bindings = schedule.get("epoch_checkpoints")
    if (
        not isinstance(epochs, list)
        or not epochs
        or any(type(epoch) is not int for epoch in epochs)
        or epochs != list(range(1, len(epochs) + 1))
        or schedule.get("epoch_count") != len(epochs)
        or not isinstance(bindings, list)
        or len(bindings) != len(epochs)
    ):
        raise ValueError("heldout truth schedule epoch inventory drifted")
    normalized = []
    for epoch, binding in zip(epochs, bindings, strict=True):
        if not isinstance(binding, Mapping) or binding.get("epoch") != epoch:
            raise ValueError("heldout truth schedule checkpoint epoch drifted")
        normalized.append(
            {
                "checkpoint_sha256": _lower_sha(
                    binding.get("checkpoint_sha256"),
                    label="heldout truth checkpoint SHA256",
                ),
                "epoch": epoch,
            }
        )
    return tuple(normalized)


def _validate_signed_schedule(path: Path) -> tuple[dict[str, Any], str]:
    value = _read_json(path)
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256", None)
    digest = hashlib.sha256(
        canonical_json_bytes(unsigned, pretty=True)
    ).hexdigest()
    if claimed != digest:
        raise ValueError(f"heldout truth schedule signature drifted: {path}")
    if value.get("schema_version") != "causalcache.structured_truth_schedule.v1":
        raise ValueError(f"unsupported heldout truth schedule schema: {path}")
    _epoch_checkpoint_bindings(value)
    records = value.get("records")
    if (
        not isinstance(records, list)
        or len(records) != EXPECTED_STATE_COUNT
        or value.get("state_count") != len(records)
        or value.get("missing_coalition_count")
        != sum(len(row.get("missing_coalitions", ())) for row in records)
    ):
        raise ValueError(f"heldout truth schedule inventory drifted: {path}")
    complete_status, pending_status = {
        "deepsets_structured_marginal": (
            "COMPLETE_STRUCTURED_HELDOUT_TRUTH",
            "PENDING_STRUCTURED_HELDOUT_TRUTH",
        ),
        "set_transformer_direct_marginal": (
            "COMPLETE_SET_TRANSFORMER_CONTROL_TRUTH",
            "PENDING_SET_TRANSFORMER_CONTROL_TRUTH",
        ),
    }[value["model_family"]]
    expected_status = (
        complete_status if value["missing_coalition_count"] == 0 else pending_status
    )
    if value.get("status") != expected_status:
        raise ValueError(f"heldout truth schedule completion status drifted: {path}")
    return value, digest


def _validate_heldout_manifest(
    path: Path, *, expected_content_sha256: str
) -> dict[str, Any]:
    expected = _lower_sha(
        expected_content_sha256, label="heldout manifest content SHA256"
    )
    value = _read_json(path)
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256", None)
    if claimed != expected or sha256_json(unsigned) != claimed:
        raise ValueError("heldout manifest content drifted")
    state_ids = value.get("checkpoint_state_ids")
    state_rows = value.get("checkpoint_states")
    if (
        not isinstance(state_ids, list)
        or not isinstance(state_rows, list)
        or len(state_ids) != EXPECTED_STATE_COUNT
        or len(state_rows) != EXPECTED_STATE_COUNT
        or len(set(state_ids)) != EXPECTED_STATE_COUNT
        or {row.get("state_id") for row in state_rows} != set(state_ids)
        or value.get("firewall", {}).get("allowed_role", "train") != "train"
        or value.get("firewall", {}).get("tune_access") is not False
        or value.get("firewall", {}).get("evaluation_access") is not False
    ):
        raise ValueError("heldout manifest checkpoint denominator or firewall drifted")
    return value


def _load_input_states(
    input_root: Path, *, expected_content_sha256: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    expected = _lower_sha(expected_content_sha256, label="input content SHA256")
    manifest_path = input_root / "manifest.json"
    manifest = _read_json(manifest_path)
    states_name = manifest.get("states_jsonl")
    if (
        manifest.get("content_sha256") != expected
        or manifest.get("evaluation_labels_included") is not False
        or not isinstance(states_name, str)
    ):
        raise ValueError("heldout truth input identity or firewall drifted")
    states_path = input_root / states_name
    if (
        not states_path.is_file()
        or sha256_file(states_path) != manifest.get("states_sha256")
    ):
        raise ValueError("heldout truth input state payload drifted")
    states = _read_jsonl(states_path)
    if manifest.get("state_count") is not None and manifest["state_count"] != len(states):
        raise ValueError("heldout truth input state count drifted")
    by_id: dict[str, dict[str, Any]] = {}
    for state in states:
        state_id = state.get("state_id")
        if state.get("role") not in {"train", "tune"}:
            raise ValueError("heldout truth input crossed the evaluation firewall")
        if not isinstance(state_id, str) or state_id in by_id:
            raise ValueError("heldout truth input state identity is invalid")
        by_id[state_id] = state
    return manifest, by_id


def _source_trajectory_shards(
    path: Path,
    *,
    expected_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, int], dict[int, str]]:
    expected = _lower_sha(
        expected_file_sha256, label="source manifest file SHA256"
    )
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError("heldout truth source manifest file drifted")
    manifest = _read_json(path)
    shards = manifest.get("shards")
    if (
        manifest.get("status") != "COMPLETED_VARIABLE_HISTORY_SOURCE"
        or manifest.get("logical_shard_count") not in (None, EXPECTED_SHARD_COUNT)
        or not isinstance(shards, list)
        or len(shards) != EXPECTED_SHARD_COUNT
    ):
        raise ValueError("heldout truth source must expose 256 logical shards")
    trajectory_shards: dict[str, int] = {}
    shard_hashes: dict[int, str] = {}
    for shard in shards:
        logical_shard = shard.get("logical_shard")
        if (
            type(logical_shard) is not int
            or not 0 <= logical_shard < EXPECTED_SHARD_COUNT
            or logical_shard in shard_hashes
        ):
            raise ValueError("heldout truth source logical shard drifted")
        shard_hashes[logical_shard] = _lower_sha(
            shard.get("sha256"), label="source shard SHA256"
        )
        trajectory_ids = shard.get("trajectory_ids")
        if not isinstance(trajectory_ids, list):
            raise ValueError("heldout truth source trajectory inventory drifted")
        for trajectory_id in trajectory_ids:
            if not isinstance(trajectory_id, str) or trajectory_id in trajectory_shards:
                raise ValueError("heldout truth source trajectory is duplicated")
            trajectory_shards[trajectory_id] = logical_shard
    if set(shard_hashes) != set(range(EXPECTED_SHARD_COUNT)):
        raise ValueError("heldout truth source shard inventory is incomplete")
    return manifest, trajectory_shards, shard_hashes


def _canonical_candidates(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
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


def _union_truth_records(
    *,
    schedules: Mapping[str, Mapping[str, Any]],
    input_states: Mapping[str, Mapping[str, Any]],
    heldout_manifest: Mapping[str, Any],
    trajectory_shards: Mapping[str, int],
) -> tuple[dict[str, Any], ...]:
    checkpoint_rows = {
        row["state_id"]: row for row in heldout_manifest["checkpoint_states"]
    }
    checkpoint_ids = set(heldout_manifest["checkpoint_state_ids"])
    union: dict[str, dict[str, Any]] = {}
    for source_name, schedule in sorted(schedules.items()):
        records = schedule["records"]
        if {row.get("state_id") for row in records} != checkpoint_ids:
            raise ValueError(f"{source_name} truth schedule escaped checkpoint states")
        for record in records:
            state_id = record["state_id"]
            state = input_states.get(state_id)
            contract = checkpoint_rows[state_id]
            if state is None:
                raise ValueError("heldout truth state is absent from input")
            candidates = _canonical_candidates(
                record.get("candidate_event_ids"), label="truth schedule candidates"
            )
            input_candidates = tuple(state.get("candidate_event_step_ids", ()))
            trajectory_id = record.get("trajectory_id")
            logical_shard = record.get("logical_shard")
            if (
                state.get("role") != "train"
                or candidates != input_candidates
                or trajectory_id != state.get("trajectory_id")
                or trajectory_id != contract.get("trajectory_id")
                or len(candidates) != contract.get("candidate_count")
                or history_bin(len(candidates)) != contract.get("history_bin")
                or logical_shard != state.get("logical_shard")
                or trajectory_shards.get(str(trajectory_id)) != logical_shard
            ):
                raise ValueError("heldout truth schedule/input state identity drifted")
            current = union.setdefault(
                state_id,
                {
                    "candidate_event_ids": candidates,
                    "logical_shard": logical_shard,
                    "missing": defaultdict(set),
                    "state_id": state_id,
                    "trajectory_id": trajectory_id,
                },
            )
            if (
                current["candidate_event_ids"] != candidates
                or current["logical_shard"] != logical_shard
                or current["trajectory_id"] != trajectory_id
            ):
                raise ValueError("heldout truth schedules disagree on state identity")
            for index, missing in enumerate(record.get("missing_coalitions", ())):
                if not isinstance(missing, Mapping) or not isinstance(
                    missing.get("source"), str
                ):
                    raise ValueError("heldout truth missing-coalition row is invalid")
                subset = tuple(missing.get("event_ids", ()))
                if (
                    any(type(event) is not int for event in subset)
                    or subset != tuple(sorted(subset))
                    or len(subset) != len(set(subset))
                    or not set(subset).issubset(candidates)
                ):
                    raise ValueError(
                        f"heldout truth coalition {index} escaped candidate universe"
                    )
                current["missing"][subset].add(
                    f"{source_name}:{missing['source']}"
                )
    if set(union) != checkpoint_ids:
        raise ValueError("heldout truth union denominator drifted")
    result = []
    for state_id in sorted(union):
        row = union[state_id]
        missing = [
            {"event_ids": list(subset), "source": "+".join(sorted(sources))}
            for subset, sources in sorted(
                row.pop("missing").items(), key=lambda item: (len(item[0]), item[0])
            )
        ]
        result.append({**row, "missing_coalitions": missing})
    return tuple(result)


def materialize_heldout_truth_schedules(
    *,
    input_root: Path,
    expected_input_content_sha256: str,
    heldout_manifest_path: Path,
    expected_heldout_content_sha256: str,
    source_manifest_path: Path,
    expected_source_manifest_sha256: str,
    truth_schedule_paths: Mapping[str, Path],
    output_root: Path,
    workers: int,
) -> dict[str, Any]:
    """Union model-specific rollout requests into one 256-shard runner schedule."""
    if workers <= 0 or len(truth_schedule_paths) not in {1, 2}:
        raise ValueError("heldout truth materialization inputs are incomplete")
    if any(not name or ":" in name or "+" in name for name in truth_schedule_paths):
        raise ValueError("truth schedule source names are invalid")
    input_manifest, input_states = _load_input_states(
        input_root, expected_content_sha256=expected_input_content_sha256
    )
    heldout_manifest = _validate_heldout_manifest(
        heldout_manifest_path,
        expected_content_sha256=expected_heldout_content_sha256,
    )
    source_manifest, trajectory_shards, shard_hashes = _source_trajectory_shards(
        source_manifest_path,
        expected_file_sha256=expected_source_manifest_sha256,
    )
    schedules = {}
    schedule_bindings = {}
    families: set[str] = set()
    shared_epochs: tuple[int, ...] | None = None
    for name, path in sorted(truth_schedule_paths.items()):
        schedule, content_sha = _validate_signed_schedule(path)
        family = str(schedule["model_family"])
        if family in families:
            raise ValueError("heldout truth schedules duplicate one model family")
        families.add(family)
        epochs = tuple(schedule["epochs"])
        if shared_epochs is None:
            shared_epochs = epochs
        elif epochs != shared_epochs:
            raise ValueError("heldout truth schedules must bind the same epochs")
        schedules[name] = schedule
        schedule_bindings[name] = {
            "content_sha256": content_sha,
            "epoch_checkpoints": list(_epoch_checkpoint_bindings(schedule)),
            "epochs": list(epochs),
            "file_sha256": sha256_file(path),
            "model_family": family,
        }
    if len(schedules) == 2 and families != set(MODEL_FAMILIES):
        raise ValueError("two-model truth union must bind DeepSets and Set Transformer")
    records = _union_truth_records(
        schedules=schedules,
        input_states=input_states,
        heldout_manifest=heldout_manifest,
        trajectory_shards=trajectory_shards,
    )
    states_by_shard: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(EXPECTED_SHARD_COUNT)
    }
    requested_missing_count = 0
    for record in records:
        missing = record["missing_coalitions"]
        requested_missing_count += len(missing)
        if not missing:
            continue
        coalitions = runner_schedule_coalitions(
            record["candidate_event_ids"], missing
        )
        states_by_shard[record["logical_shard"]].append(
            {
                "candidate_event_ids": list(record["candidate_event_ids"]),
                "coalitions": list(coalitions),
                "logical_shard": record["logical_shard"],
                "role": "train",
                "state_id": record["state_id"],
                "tracks": ["heldout_truth_checkpoint_selection"],
                "trajectory_id": record["trajectory_id"],
            }
        )
    bindings = {
        "heldout_manifest_content_sha256": heldout_manifest["content_sha256"],
        "heldout_manifest_file_sha256": sha256_file(heldout_manifest_path),
        "input_content_sha256": input_manifest["content_sha256"],
        "input_manifest_file_sha256": sha256_file(input_root / "manifest.json"),
        "source_manifest_file_sha256": sha256_file(source_manifest_path),
        "truth_schedules": schedule_bindings,
    }
    binding_sha = sha256_json(bindings)

    def build(logical_shard: int) -> dict[str, Any]:
        rows = sorted(states_by_shard[logical_shard], key=lambda row: row["state_id"])
        payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        schedule_path = (
            output_root
            / "schedule-shards"
            / f"shard-{logical_shard:03d}-of-{EXPECTED_SHARD_COUNT}.jsonl"
        )
        receipt_path = (
            output_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-{EXPECTED_SHARD_COUNT}.json"
        )
        coalition_count = sum(len(row["coalitions"]) for row in rows)
        forward_count = sum(
            tuple(coalition["event_ids"]) != tuple(row["candidate_event_ids"])
            for row in rows
            for coalition in row["coalitions"]
        )
        receipt = {
            "binding_sha256": binding_sha,
            "coalition_count": coalition_count,
            "forward_coalition_count": forward_count,
            "logical_shard": logical_shard,
            "role_state_counts": {"train": len(rows)} if rows else {},
            "schedule_byte_count": len(payload),
            "schedule_sha256": hashlib.sha256(payload).hexdigest(),
            "source_shard_sha256": shard_hashes[logical_shard],
            "state_count": len(rows),
            "status": SHARD_STATUS,
        }
        receipt_payload = canonical_json_bytes(receipt, pretty=True) + b"\n"
        if schedule_path.exists() or receipt_path.exists():
            if (
                not schedule_path.is_file()
                or not receipt_path.is_file()
                or schedule_path.read_bytes() != payload
                or receipt_path.read_bytes() != receipt_payload
            ):
                raise ValueError("existing heldout truth schedule shard drifted")
            return receipt
        _write_atomic(schedule_path, payload)
        _write_atomic(receipt_path, receipt_payload)
        return receipt

    with ThreadPoolExecutor(max_workers=workers) as executor:
        receipts = list(executor.map(build, range(EXPECTED_SHARD_COUNT)))
    summary = {
        **bindings,
        "binding_sha256": binding_sha,
        "coalition_count": sum(row["coalition_count"] for row in receipts),
        "forward_coalition_count": sum(
            row["forward_coalition_count"] for row in receipts
        ),
        "logical_shard_count": EXPECTED_SHARD_COUNT,
        "requested_missing_coalition_count": requested_missing_count,
        "scheduled_state_count": sum(row["state_count"] for row in receipts),
        "schema_version": "causalcache.heldout_truth_schedule_summary.v1",
        "source_content_sha256": source_manifest.get("content_sha256"),
        "status": SCHEDULE_STATUS,
    }
    summary["content_sha256"] = sha256_json(summary)
    summary_payload = canonical_json_bytes(summary, pretty=True) + b"\n"
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        if summary_path.read_bytes() != summary_payload:
            raise ValueError("existing heldout truth schedule summary drifted")
    else:
        _write_atomic(summary_path, summary_payload)
    return summary


def _write_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value, pretty=True) + b"\n"
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing formal truth metadata drifted: {path}")
        return
    _write_atomic(path, payload)


def _canonical_coalition(
    value: Any, *, candidates: tuple[int, ...], label: str
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an event-id array")
    subset = tuple(value)
    if (
        any(type(event) is not int for event in subset)
        or subset != tuple(sorted(subset))
        or len(subset) != len(set(subset))
        or not set(subset).issubset(candidates)
    ):
        raise ValueError(f"{label} is not canonical")
    return subset


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


def _load_materialized_schedule(
    schedule_root: Path,
) -> tuple[dict[str, Any], str, tuple[dict[str, Any], ...]]:
    summary_path = schedule_root / "summary.json"
    summary, summary_sha = _validate_compact_signature(
        summary_path,
        schema_version="causalcache.heldout_truth_schedule_summary.v1",
    )
    if (
        summary.get("status") != SCHEDULE_STATUS
        or summary.get("logical_shard_count") != EXPECTED_SHARD_COUNT
        or not isinstance(summary.get("truth_schedules"), Mapping)
        or len(summary["truth_schedules"]) not in {1, 2}
    ):
        raise ValueError("formal truth schedule summary is incomplete")
    model_families = {
        binding.get("model_family")
        for binding in summary["truth_schedules"].values()
        if isinstance(binding, Mapping)
    }
    if len(summary["truth_schedules"]) == 2 and model_families != set(MODEL_FAMILIES):
        raise ValueError("formal truth schedule model union drifted")

    records: list[dict[str, Any]] = []
    seen_states: set[str] = set()
    receipt_file_shas: dict[int, str] = {}
    for logical_shard in range(EXPECTED_SHARD_COUNT):
        schedule_path = (
            schedule_root
            / "schedule-shards"
            / f"shard-{logical_shard:03d}-of-{EXPECTED_SHARD_COUNT}.jsonl"
        )
        receipt_path = (
            schedule_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-{EXPECTED_SHARD_COUNT}.json"
        )
        if not schedule_path.is_file() or not receipt_path.is_file():
            raise FileNotFoundError("formal truth schedule shard or receipt is missing")
        payload = schedule_path.read_bytes()
        receipt = _read_json(receipt_path)
        rows = _read_jsonl(schedule_path)
        if (
            receipt.get("status") != SHARD_STATUS
            or receipt.get("logical_shard") != logical_shard
            or receipt.get("binding_sha256") != summary.get("binding_sha256")
            or receipt.get("schedule_sha256") != hashlib.sha256(payload).hexdigest()
            or receipt.get("schedule_byte_count") != len(payload)
            or receipt.get("state_count") != len(rows)
            or receipt.get("coalition_count")
            != sum(len(row.get("coalitions", ())) for row in rows)
            or receipt.get("forward_coalition_count")
            != sum(
                tuple(coalition.get("event_ids", ()))
                != tuple(row.get("candidate_event_ids", ()))
                for row in rows
                for coalition in row.get("coalitions", ())
                if isinstance(coalition, Mapping)
            )
        ):
            raise ValueError("formal truth schedule shard receipt drifted")
        receipt_file_shas[logical_shard] = sha256_file(receipt_path)
        for row in rows:
            state_id = row.get("state_id")
            trajectory_id = row.get("trajectory_id")
            candidates = _canonical_candidates(
                row.get("candidate_event_ids"), label="formal truth candidates"
            )
            raw_coalitions = row.get("coalitions")
            if (
                not isinstance(state_id, str)
                or state_id in seen_states
                or not isinstance(trajectory_id, str)
                or row.get("role") != "train"
                or row.get("logical_shard") != logical_shard
                or not isinstance(raw_coalitions, list)
                or not raw_coalitions
            ):
                raise ValueError("formal truth scheduled state identity drifted")
            coalitions = []
            for ordinal, raw in enumerate(raw_coalitions):
                if not isinstance(raw, Mapping) or not isinstance(raw.get("source"), str):
                    raise ValueError("formal truth scheduled coalition is invalid")
                coalitions.append(
                    _canonical_coalition(
                        raw.get("event_ids"),
                        candidates=candidates,
                        label=f"formal truth coalition {ordinal}",
                    )
                )
            if (
                len(coalitions) != len(set(coalitions))
                or tuple(coalitions)
                != tuple(sorted(coalitions, key=lambda subset: (len(subset), subset)))
                or candidates not in coalitions
            ):
                raise ValueError("formal truth coalition inventory drifted")
            seen_states.add(state_id)
            records.append(
                {
                    **row,
                    "schedule_receipt_sha256": receipt_file_shas[logical_shard],
                    "source_shard_sha256": _lower_sha(
                        receipt.get("source_shard_sha256"),
                        label="formal truth source shard SHA256",
                    ),
                }
            )
    if (
        len(records) != summary.get("scheduled_state_count")
        or sum(len(row["coalitions"]) for row in records)
        != summary.get("coalition_count")
    ):
        raise ValueError("formal truth scheduled denominator drifted")
    return summary, summary_sha, tuple(sorted(records, key=lambda row: row["state_id"]))


def seal_formal_truth_root(
    *, schedule_root: Path, truth_root: Path
) -> dict[str, Any]:
    """Seal one complete runner output behind immutable schedule provenance."""
    schedule_root = schedule_root.resolve()
    truth_root = truth_root.resolve()
    summary, summary_sha, schedules = _load_materialized_schedule(schedule_root)
    if not schedules:
        raise ValueError("cannot seal an empty formal truth schedule")
    states_root = truth_root / "states"
    if not states_root.is_dir():
        raise FileNotFoundError(f"formal truth root has no states directory: {truth_root}")
    terminals: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(states_root.glob("*.json")):
        terminal = _read_json(path)
        state_id = terminal.get("state_id")
        if not isinstance(state_id, str) or state_id in terminals:
            raise ValueError("formal truth terminal state inventory is invalid")
        terminals[state_id] = (path, terminal)
    scheduled_ids = {row["state_id"] for row in schedules}
    if set(terminals) != scheduled_ids:
        missing = len(scheduled_ids - set(terminals))
        extra = len(set(terminals) - scheduled_ids)
        raise ValueError(
            f"formal truth terminal denominator drifted: missing={missing}, extra={extra}"
        )

    revisions: set[str] = set()
    scientific_shas: set[str] = set()
    execution_shas: set[str] = set()
    state_bindings = []
    total_rows = 0
    for schedule in schedules:
        state_id = schedule["state_id"]
        path, terminal = terminals[state_id]
        candidates = tuple(schedule["candidate_event_ids"])
        expected_coalitions = tuple(
            tuple(row["event_ids"]) for row in schedule["coalitions"]
        )
        source_revision = terminal.get("source_revision")
        scientific_sha = _lower_sha(
            terminal.get("scientific_config_sha256"),
            label="formal truth scientific config SHA256",
        )
        execution_sha = _lower_sha(
            terminal.get("execution_config_sha256"),
            label="formal truth execution config SHA256",
        )
        if not isinstance(source_revision, str) or re.fullmatch(
            r"[0-9a-f]{40}", source_revision
        ) is None:
            raise ValueError("formal truth source revision must be a full Git SHA")
        if (
            terminal.get("status") != LABEL_STATUS
            or terminal.get("role") != "train"
            or terminal.get("state_id") != state_id
            or terminal.get("trajectory_id") != schedule["trajectory_id"]
            or tuple(terminal.get("candidate_event_step_ids", ())) != candidates
        ):
            raise ValueError("formal truth terminal identity or status drifted")
        rows = terminal.get("distance_rows")
        if not isinstance(rows, list):
            raise ValueError("formal truth terminal distance table is missing")
        observed = []
        for ordinal, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError("formal truth distance row is invalid")
            subset = _canonical_coalition(
                row.get("coalition_event_step_ids"),
                candidates=candidates,
                label=f"formal truth distance row {ordinal}",
            )
            distance = row.get("distance")
            if (
                isinstance(distance, bool)
                or not isinstance(distance, (int, float))
                or not math.isfinite(float(distance))
                or float(distance) < 0.0
            ):
                raise ValueError("formal truth distance is not finite and nonnegative")
            if subset == candidates and float(distance) != 0.0:
                raise ValueError("formal truth full-history anchor is nonzero")
            observed.append(subset)
        if tuple(observed) != expected_coalitions:
            raise ValueError("formal truth terminal coalition identity drifted")
        expected_identity = _runner_state_identity(
            schedule=schedule,
            execution_config_sha256=execution_sha,
            scientific_config_sha256=scientific_sha,
            schedule_receipt_sha256=schedule["schedule_receipt_sha256"],
            source_revision=source_revision,
            source_shard_sha256=schedule["source_shard_sha256"],
        )
        if terminal.get("state_identity_sha256") != expected_identity:
            raise ValueError("formal truth runner state identity drifted")
        revisions.add(source_revision)
        scientific_shas.add(scientific_sha)
        execution_shas.add(execution_sha)
        total_rows += len(rows)
        state_bindings.append(
            {
                "candidate_event_ids": list(candidates),
                "coalition_count": len(rows),
                "logical_shard": schedule["logical_shard"],
                "state_id": state_id,
                "state_identity_sha256": expected_identity,
                "terminal_file_sha256": sha256_file(path),
                "terminal_relative_path": path.relative_to(truth_root).as_posix(),
                "trajectory_id": schedule["trajectory_id"],
            }
        )
    if len(revisions) != 1 or len(scientific_shas) != 1 or len(execution_shas) != 1:
        raise ValueError("formal truth label source/config revisions are inconsistent")

    summary_path = schedule_root / "summary.json"
    manifest = _signed_compact(
        {
            "coalition_count": total_rows,
            "execution_config_sha256": next(iter(execution_shas)),
            "heldout_manifest_content_sha256": summary[
                "heldout_manifest_content_sha256"
            ],
            "heldout_manifest_file_sha256": summary[
                "heldout_manifest_file_sha256"
            ],
            "input_content_sha256": summary["input_content_sha256"],
            "input_manifest_file_sha256": summary[
                "input_manifest_file_sha256"
            ],
            "model_schedules": summary["truth_schedules"],
            "schema_version": "causalcache.formal_heldout_truth_manifest.v1",
            "scientific_config_sha256": next(iter(scientific_shas)),
            "source_content_sha256": summary.get("source_content_sha256"),
            "source_manifest_file_sha256": summary[
                "source_manifest_file_sha256"
            ],
            "source_revision": next(iter(revisions)),
            "state_count": len(state_bindings),
            "states": state_bindings,
            "status": FORMAL_MANIFEST_STATUS,
            "truth_schedule_summary_content_sha256": summary_sha,
            "truth_schedule_summary_file_sha256": sha256_file(summary_path),
        }
    )
    manifest_path = truth_root / "formal-truth-manifest.json"
    _write_or_verify(manifest_path, manifest)
    receipt = _signed_compact(
        {
            "coalition_count": total_rows,
            "formal_manifest_content_sha256": manifest["content_sha256"],
            "formal_manifest_file_sha256": sha256_file(manifest_path),
            "heldout_manifest_content_sha256": manifest[
                "heldout_manifest_content_sha256"
            ],
            "input_content_sha256": manifest["input_content_sha256"],
            "schema_version": "causalcache.formal_heldout_truth_receipt.v1",
            "source_manifest_file_sha256": manifest[
                "source_manifest_file_sha256"
            ],
            "state_count": len(state_bindings),
            "status": FORMAL_RECEIPT_STATUS,
            "truth_schedule_summary_content_sha256": summary_sha,
            "truth_schedule_summary_file_sha256": sha256_file(summary_path),
        }
    )
    _write_or_verify(truth_root / "formal-truth-receipt.json", receipt)
    return manifest


__all__ = [
    "EXPECTED_SHARD_COUNT",
    "EXPECTED_STATE_COUNT",
    "SCHEDULE_STATUS",
    "SHARD_STATUS",
    "FORMAL_MANIFEST_STATUS",
    "FORMAL_RECEIPT_STATUS",
    "LABEL_STATUS",
    "MODEL_FAMILIES",
    "materialize_heldout_truth_schedules",
    "seal_formal_truth_root",
]
