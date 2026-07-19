"""Processor-only candidate-freeze orchestration primitives.

This module intentionally contains no Transformers model import and no policy
execution interface.  It prepares deterministic shard work, renders the full
reference prompt through an injected processor-only length function, and
materializes the exact variable-n label schedule after candidate freezing.
"""

from __future__ import annotations

import hashlib
import ast
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.low_fidelity_v2 import LowFidelityEventV2
from causalcache.set_utility_label_producer import (
    CandidateLengthAttempt,
    FrozenCandidateContext,
    MixedFidelityPromptPlan,
    PromptHistoryEvent,
)
from causalcache.set_utility_label_schedule import (
    LabelStateRequest,
    VariableNLabelSchedule,
    build_variable_n_label_schedule,
    validate_variable_n_label_schedule,
)
from causalcache.set_utility_processor_substrate import SelectedRowReadPlan, SelectedShardRead
from causalcache.set_utility_processor_substrate import SelectedTrajectoryAssignment


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_processor_freeze_execution_cf"
CONTEXT_LIMIT_TOKENS = 32_768
RESERVED_ACTION_TOKENS = 256
MINIMUM_FINAL_CANDIDATE_COUNT = 4
MAXIMUM_FINAL_CANDIDATE_COUNT = 16
WORKER_COUNT = 4
TARGET_PIXELS_PER_IMAGE = 2_621_440
_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
        return (json.dumps(value, **options) + "\n").encode("utf-8")
    options["separators"] = (",", ":")
    return json.dumps(value, **options).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def verify_processor_only_snapshot(
    *,
    model_dir: str | Path,
    snapshot_manifest: str | Path,
) -> dict[str, Any]:
    """Hash the frozen snapshot without importing or instantiating a model."""
    root = Path(model_dir).resolve()
    manifest_path = Path(snapshot_manifest).resolve()
    if not root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError("processor model directory or snapshot manifest is missing")
    manifest_bytes = manifest_path.read_bytes()
    expected = _strict_json_bytes(manifest_bytes, label="snapshot manifest")
    if set(expected) != {"repo", "revision", "files"}:
        raise ValueError("snapshot manifest fields drifted")
    if (
        expected["repo"] != "mPLUG/GUI-Owl-1.5-8B-Instruct"
        or expected["revision"]
        != "06d5faecff74840bab2be2425e9c42667a5d04fc"
    ):
        raise ValueError("processor snapshot model identity drifted")
    local_manifest = root / ".snapshot.json"
    if not local_manifest.is_file() or _strict_json_bytes(
        local_manifest.read_bytes(), label="local snapshot manifest"
    ) != expected:
        raise ValueError("local model snapshot differs from the repository manifest")
    raw_files = expected["files"]
    if not isinstance(raw_files, list) or len(raw_files) != 14:
        raise ValueError("processor snapshot file inventory drifted")
    records: list[dict[str, Any]] = []
    observed_paths: set[str] = set()
    total_bytes = 0
    for raw in raw_files:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "size", "sha256"}:
            raise ValueError("processor snapshot file fields drifted")
        relative = _safe_member_path(raw["path"], label="snapshot file path")
        if relative in observed_paths:
            raise ValueError("processor snapshot file path is duplicated")
        size = raw["size"]
        digest = raw["sha256"]
        if (
            type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError("processor snapshot file binding is invalid")
        path = root.joinpath(*PurePosixPath(relative).parts)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            raise ValueError(f"processor snapshot file drifted: {relative}")
        observed_paths.add(relative)
        total_bytes += size
        records.append({"path": relative, "size": size, "sha256": digest})
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != observed_paths | {".snapshot.json"}:
        raise ValueError("processor model directory contains an unexpected file")
    config = _strict_json_bytes((root / "config.json").read_bytes(), label="model config")
    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping) or (
        text_config.get("max_position_embeddings") != CONTEXT_LIMIT_TOKENS
    ):
        raise ValueError("processor snapshot context limit drifted")
    return {
        "model_dir": str(root),
        "model_repo": expected["repo"],
        "model_revision": expected["revision"],
        "snapshot_manifest_sha256": sha256_bytes(manifest_bytes),
        "verified_file_count": len(records),
        "verified_total_bytes": total_bytes,
        "file_inventory_sha256": sha256_bytes(canonical_json_bytes(records)),
        "maximum_context_tokens": CONTEXT_LIMIT_TOKENS,
        "model_or_policy_loaded": False,
    }


def _identity(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be non-empty canonical text")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _ordered_ids(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise TypeError(f"{label} must be an ordered sequence")
    result = tuple(value)
    if (
        not result
        or any(type(item) is not int or item <= 0 for item in result)
        or result != tuple(sorted(result))
        or len(result) != len(set(result))
    ):
        raise ValueError(f"{label} must contain sorted unique positive integers")
    return result


def _safe_member_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe relative POSIX path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"{label} must be a safe relative POSIX path")
    return value


@dataclass(frozen=True)
class ShardWorker:
    worker_index: int
    shards: tuple[SelectedShardRead, ...]
    observation_load: int

    def __post_init__(self) -> None:
        if type(self.worker_index) is not int or not 0 <= self.worker_index < WORKER_COUNT:
            raise ValueError("shard worker index is outside the four-worker schedule")
        if any(not isinstance(shard, SelectedShardRead) for shard in self.shards):
            raise TypeError("shard worker contains an invalid selected shard")
        paths = tuple(shard.source_file.transport_file for shard in self.shards)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("worker shard paths must be sorted and unique")
        expected = sum(
            assignment.decision_count + 1
            for shard in self.shards
            for assignment in shard.assignments
        )
        if self.observation_load != expected:
            raise ValueError("worker observation load drifted")


@dataclass(frozen=True)
class ShardWorkerSchedule:
    workers: tuple[ShardWorker, ...]
    selected_shard_count: int
    selected_trajectory_count: int
    selected_observation_count: int
    inventory_sha256: str

    def __post_init__(self) -> None:
        if len(self.workers) != WORKER_COUNT:
            raise ValueError("selected-row execution requires exactly four workers")
        if tuple(worker.worker_index for worker in self.workers) != tuple(
            range(WORKER_COUNT)
        ):
            raise ValueError("worker indices must be contiguous from zero")
        all_shards = tuple(shard for worker in self.workers for shard in worker.shards)
        paths = [shard.source_file.transport_file for shard in all_shards]
        if len(paths) != len(set(paths)) or len(paths) != self.selected_shard_count:
            raise ValueError("a selected shard was duplicated or omitted")
        assignments = [
            assignment
            for shard in all_shards
            for assignment in shard.assignments
        ]
        if len(assignments) != self.selected_trajectory_count:
            raise ValueError("selected trajectory count drifted")
        if sum(worker.observation_load for worker in self.workers) != (
            self.selected_observation_count
        ):
            raise ValueError("selected observation count drifted")
        if not isinstance(self.inventory_sha256, str) or _SHA256.fullmatch(
            self.inventory_sha256
        ) is None:
            raise ValueError("worker schedule inventory SHA256 is invalid")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "worker_count": WORKER_COUNT,
            "selected_shard_count": self.selected_shard_count,
            "selected_trajectory_count": self.selected_trajectory_count,
            "selected_observation_count": self.selected_observation_count,
            "inventory_sha256": self.inventory_sha256,
            "workers": [
                {
                    "worker_index": worker.worker_index,
                    "observation_load": worker.observation_load,
                    "transport_files": [
                        shard.source_file.transport_file for shard in worker.shards
                    ],
                    "trajectory_count": sum(
                        len(shard.assignments) for shard in worker.shards
                    ),
                }
                for worker in self.workers
            ],
        }


def build_shard_worker_schedule(plan: SelectedRowReadPlan) -> ShardWorkerSchedule:
    """Assign whole selected shards by deterministic LPT; never split a shard."""
    if not isinstance(plan, SelectedRowReadPlan):
        raise TypeError("selected-row plan must be SelectedRowReadPlan")
    worker_shards: list[list[SelectedShardRead]] = [[] for _ in range(WORKER_COUNT)]
    loads = [0] * WORKER_COUNT
    weighted = [
        (
            sum(item.decision_count + 1 for item in shard.assignments),
            shard.source_file.transport_file,
            shard,
        )
        for shard in plan.shards
    ]
    for weight, _, shard in sorted(weighted, key=lambda item: (-item[0], item[1])):
        worker_index = min(range(WORKER_COUNT), key=lambda index: (loads[index], index))
        worker_shards[worker_index].append(shard)
        loads[worker_index] += weight
    workers = tuple(
        ShardWorker(
            worker_index=index,
            shards=tuple(
                sorted(
                    worker_shards[index],
                    key=lambda shard: shard.source_file.transport_file,
                )
            ),
            observation_load=loads[index],
        )
        for index in range(WORKER_COUNT)
    )
    identity_payload = {
        "workers": [
            {
                "worker_index": worker.worker_index,
                "transport_files": [
                    shard.source_file.transport_file for shard in worker.shards
                ],
                "source_ids": [
                    assignment.source_id
                    for shard in worker.shards
                    for assignment in shard.assignments
                ],
            }
            for worker in workers
        ]
    }
    return ShardWorkerSchedule(
        workers=workers,
        selected_shard_count=len(plan.shards),
        selected_trajectory_count=plan.assignment_count,
        selected_observation_count=sum(loads),
        inventory_sha256=sha256_bytes(canonical_json_bytes(identity_payload)),
    )


def validate_freeze_query_topology(
    assignments: Sequence[Mapping[str, Any]],
    query_states: Sequence[Mapping[str, Any]],
    *,
    anchor_step_by_stratum: Mapping[str, int],
) -> tuple[tuple[SelectedTrajectoryAssignment, ...], dict[str, tuple[dict[str, Any], ...]]]:
    """Validate exactly one stratum anchor and terminal query per trajectory."""
    if (
        isinstance(assignments, (str, bytes, bytearray, Mapping))
        or isinstance(query_states, (str, bytes, bytearray, Mapping))
        or not assignments
        or not query_states
    ):
        raise ValueError("freeze assignments and query states must be non-empty sequences")
    normalized = tuple(
        SelectedTrajectoryAssignment.from_mapping(value) for value in assignments
    )
    by_source = {assignment.source_id: assignment for assignment in normalized}
    if len(by_source) != len(normalized):
        raise ValueError("freeze assignments contain duplicate source IDs")
    expected_anchor = {
        "decisions_6_9": 6,
        "decisions_10_17": 10,
        "decisions_18_plus": 18,
    }
    if dict(anchor_step_by_stratum) != expected_anchor:
        raise ValueError("stratum anchor decision mapping drifted")
    grouped: dict[str, list[dict[str, Any]]] = {
        source_id: [] for source_id in by_source
    }
    state_ids: set[str] = set()
    for raw in query_states:
        if not isinstance(raw, Mapping):
            raise TypeError("freeze query state must be a mapping")
        query = dict(raw)
        source_id = query.get("source_id")
        assignment = by_source.get(source_id)
        if assignment is None:
            raise ValueError("freeze query references an unknown selected source")
        state_id = _identity(query.get("state_id"), label="query state id")
        if state_id in state_ids:
            raise ValueError("freeze query state ID is duplicated")
        state_ids.add(state_id)
        if (
            query.get("trajectory_id") != assignment.trajectory_id
            or query.get("role") != assignment.role
            or query.get("candidate_capacity_stratum")
            != assignment.candidate_capacity_stratum
            or query.get("maximum_labeled_cardinality") != 2
            or query.get("processor_candidate_freeze_status")
            != "PENDING_SEPARATE_EXECUTION"
        ):
            raise ValueError("freeze query/assignment identity drifted")
        step = _positive_int(query.get("decision_step_id"), label="query decision step")
        current = _positive_int(
            query.get("current_equivalent_event_step_id"),
            label="query current-equivalent event step",
        )
        if current != step - 1:
            raise ValueError("freeze query current-equivalent event is not step - 1")
        candidates = _ordered_ids(
            query.get("initial_candidate_event_step_ids"),
            label="query initial candidate event ids",
        )
        expected_candidates = tuple(range(1, current))[-MAXIMUM_FINAL_CANDIDATE_COUNT:]
        if (
            candidates != expected_candidates
            or query.get("initial_candidate_count") != len(candidates)
        ):
            raise ValueError("freeze query candidate suffix drifted")
        grouped[source_id].append(query)
    if len(query_states) != 2 * len(normalized):
        raise ValueError("freeze query denominator must equal two per trajectory")
    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for source_id, assignment in by_source.items():
        queries = grouped[source_id]
        if len(queries) != 2 or {query.get("query_kind") for query in queries} != {
            "stratum_anchor",
            "terminal",
        }:
            raise ValueError("trajectory requires exactly one anchor and one terminal query")
        by_kind = {query["query_kind"]: query for query in queries}
        if by_kind["stratum_anchor"]["decision_step_id"] != expected_anchor[
            assignment.candidate_capacity_stratum
        ]:
            raise ValueError("trajectory anchor decision step drifted")
        if by_kind["terminal"]["decision_step_id"] != assignment.decision_count + 1:
            raise ValueError("trajectory terminal decision step drifted")
        result[source_id] = tuple(sorted(queries, key=lambda query: query["state_id"]))
    return tuple(sorted(normalized, key=lambda item: item.source_id)), result


def worker_read_plan(
    schedule: ShardWorkerSchedule,
    worker_index: int,
) -> SelectedRowReadPlan:
    if not isinstance(schedule, ShardWorkerSchedule):
        raise TypeError("schedule must be ShardWorkerSchedule")
    if type(worker_index) is not int or not 0 <= worker_index < WORKER_COUNT:
        raise ValueError("worker index must be in [0, 3]")
    shards = schedule.workers[worker_index].shards
    return SelectedRowReadPlan(
        shards=shards,
        assignment_count=sum(len(shard.assignments) for shard in shards),
    )


class CandidateMinimumViolation(ValueError):
    """No suffix at or above the frozen n>=4 floor fits the context."""

    def __init__(self, attempts: tuple[CandidateLengthAttempt, ...]) -> None:
        super().__init__("no processor suffix with at least four candidates fits")
        self.attempts = attempts


@dataclass(frozen=True)
class FrozenQueryCandidateRecord:
    state_id: str
    trajectory_id: str
    source_id: str
    role: str
    query_kind: str
    decision_step_id: int
    maximum_labeled_cardinality: int
    candidate_context: FrozenCandidateContext

    def __post_init__(self) -> None:
        for value, label in (
            (self.state_id, "state id"),
            (self.trajectory_id, "trajectory id"),
            (self.source_id, "source id"),
            (self.role, "role"),
            (self.query_kind, "query kind"),
        ):
            _identity(value, label=label)
        _positive_int(self.decision_step_id, label="decision step")
        if type(self.maximum_labeled_cardinality) is not int or (
            self.maximum_labeled_cardinality not in (2, 3, 4)
        ):
            raise ValueError("maximum labeled cardinality must be 2, 3, or 4")
        if not isinstance(self.candidate_context, FrozenCandidateContext):
            raise TypeError("candidate context must be FrozenCandidateContext")
        if not MINIMUM_FINAL_CANDIDATE_COUNT <= len(
            self.candidate_context.candidate_event_step_ids
        ) <= MAXIMUM_FINAL_CANDIDATE_COUNT:
            raise ValueError("final candidate count must be in [4, 16]")

    def to_payload(self) -> dict[str, Any]:
        context = self.candidate_context
        return {
            "schema_version": SCHEMA_VERSION,
            "state_id": self.state_id,
            "trajectory_id": self.trajectory_id,
            "source_id": self.source_id,
            "role": self.role,
            "query_kind": self.query_kind,
            "decision_step_id": self.decision_step_id,
            "maximum_labeled_cardinality": self.maximum_labeled_cardinality,
            "initial_candidate_event_step_ids": list(
                context.initial_candidate_event_step_ids
            ),
            "candidate_event_step_ids": list(context.candidate_event_step_ids),
            "dropped_event_step_ids": list(context.dropped_event_step_ids),
            "processor_input_token_count": context.processor_input_token_count,
            "reserved_action_tokens": context.reserved_action_tokens,
            "context_limit": context.context_limit,
            "attempts": [
                {
                    "candidate_event_step_ids": list(
                        attempt.candidate_event_step_ids
                    ),
                    "processor_input_token_count": (
                        attempt.processor_input_token_count
                    ),
                }
                for attempt in context.attempts
            ],
            "status": "PROCESSOR_ONLY_CANDIDATES_FROZEN",
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> FrozenQueryCandidateRecord:
        if not isinstance(value, Mapping):
            raise TypeError("frozen query payload must be a mapping")
        expected = {
            "schema_version",
            "state_id",
            "trajectory_id",
            "source_id",
            "role",
            "query_kind",
            "decision_step_id",
            "maximum_labeled_cardinality",
            "initial_candidate_event_step_ids",
            "candidate_event_step_ids",
            "dropped_event_step_ids",
            "processor_input_token_count",
            "reserved_action_tokens",
            "context_limit",
            "attempts",
            "status",
        }
        if set(value) != expected:
            raise ValueError("frozen query payload fields drifted")
        if (
            value["schema_version"] != SCHEMA_VERSION
            or value["status"] != "PROCESSOR_ONLY_CANDIDATES_FROZEN"
        ):
            raise ValueError("frozen query payload identity drifted")
        raw_attempts = value["attempts"]
        if not isinstance(raw_attempts, list) or not raw_attempts:
            raise ValueError("frozen query attempts must be a non-empty array")
        attempts = tuple(
            CandidateLengthAttempt(
                candidate_event_step_ids=tuple(item["candidate_event_step_ids"]),
                processor_input_token_count=item["processor_input_token_count"],
            )
            for item in raw_attempts
            if isinstance(item, Mapping)
            and set(item)
            == {"candidate_event_step_ids", "processor_input_token_count"}
        )
        if len(attempts) != len(raw_attempts):
            raise ValueError("frozen query attempt fields drifted")
        context = FrozenCandidateContext(
            initial_candidate_event_step_ids=tuple(
                value["initial_candidate_event_step_ids"]
            ),
            candidate_event_step_ids=tuple(value["candidate_event_step_ids"]),
            dropped_event_step_ids=tuple(value["dropped_event_step_ids"]),
            processor_input_token_count=value["processor_input_token_count"],
            reserved_action_tokens=value["reserved_action_tokens"],
            context_limit=value["context_limit"],
            attempts=attempts,
        )
        return cls(
            state_id=value["state_id"],
            trajectory_id=value["trajectory_id"],
            source_id=value["source_id"],
            role=value["role"],
            query_kind=value["query_kind"],
            decision_step_id=value["decision_step_id"],
            maximum_labeled_cardinality=value["maximum_labeled_cardinality"],
            candidate_context=context,
        )


def _low_fidelity(value: Mapping[str, Any]) -> LowFidelityEventV2:
    return LowFidelityEventV2(
        step_id=value.get("step_id"),
        action_type=value.get("action_type"),
        action_argument=value.get("action_argument"),
        foreground_app=value.get("foreground_app"),
        screen_text_added=tuple(value.get("screen_text_added", ())),
        screen_text_removed=tuple(value.get("screen_text_removed", ())),
        screen_change=value.get("screen_change"),
        executor_result=value.get("executor_result"),
    )


def build_full_reference_prompt_plan(
    trajectory_payload: Mapping[str, Any],
    query_payload: Mapping[str, Any],
    candidate_event_step_ids: Sequence[int],
) -> MixedFidelityPromptPlan:
    """Keep all summaries and restore every event in the attempted candidate suffix."""
    if not isinstance(trajectory_payload, Mapping) or not isinstance(
        query_payload, Mapping
    ):
        raise TypeError("trajectory and query payloads must be mappings")
    candidates = _ordered_ids(
        candidate_event_step_ids,
        label="attempted candidate event ids",
    )
    initial = _ordered_ids(
        query_payload.get("initial_candidate_event_step_ids"),
        label="query initial candidate event ids",
    )
    if candidates != initial[len(initial) - len(candidates) :]:
        raise ValueError("attempted candidates must be a non-empty suffix of the initial set")
    current_equivalent = _positive_int(
        query_payload.get("current_equivalent_event_step_id"),
        label="current-equivalent event step",
    )
    decision_step = _positive_int(
        query_payload.get("decision_step_id"),
        label="query decision step",
    )
    if decision_step != current_equivalent + 1:
        raise ValueError("query decision must immediately follow the current-equivalent event")
    raw_history = trajectory_payload.get("history_events")
    if not isinstance(raw_history, list) or len(raw_history) != current_equivalent:
        raise ValueError("query substrate must contain exactly its visible history prefix")
    history = raw_history
    if tuple(value.get("event_step_id") for value in history) != tuple(
        range(1, current_equivalent + 1)
    ):
        raise ValueError("trajectory history event IDs drifted")
    expected_initial = tuple(range(1, current_equivalent))[-MAXIMUM_FINAL_CANDIDATE_COUNT:]
    if initial != expected_initial:
        raise ValueError("query candidates are not the newest non-current history suffix")
    current_reference = _identity(
        query_payload.get("current_observation_ref"),
        label="current observation reference",
    )
    if history[-1].get("high_fidelity_observation_ref") != current_reference:
        raise ValueError("current observation differs from the final visible post-state")
    candidate_set = frozenset(candidates)
    events = tuple(
        PromptHistoryEvent(
            event_step_id=raw["event_step_id"],
            low_fidelity_summary=_low_fidelity(raw["low_fidelity_summary"]),
            is_frozen_candidate=raw["event_step_id"] in candidate_set,
            high_fidelity_observation_ref=(
                raw["high_fidelity_observation_ref"]
                if raw["event_step_id"] in candidate_set
                else None
            ),
        )
        for raw in history
    )
    return MixedFidelityPromptPlan(
        state_id=_identity(query_payload.get("state_id"), label="query state id"),
        task_instruction=_identity(
            trajectory_payload.get("task_instruction"),
            label="task instruction",
        ),
        restored_event_step_ids=candidates,
        history_events=events,
        current_observation_ref=current_reference,
    )


def freeze_query_candidates(
    trajectory_payload: Mapping[str, Any],
    query_payload: Mapping[str, Any],
    *,
    processor_only_length: Callable[[MixedFidelityPromptPlan], int],
) -> FrozenQueryCandidateRecord:
    """Freeze one query using only repeated AutoProcessor prompt lengths."""
    if not callable(processor_only_length):
        raise TypeError("processor-only length function must be callable")
    initial = _ordered_ids(
        query_payload.get("initial_candidate_event_step_ids"),
        label="query initial candidate event ids",
    )

    def attempted_length(candidates: tuple[int, ...]) -> int:
        plan = build_full_reference_prompt_plan(
            trajectory_payload,
            query_payload,
            candidates,
        )
        value = processor_only_length(plan)
        return _positive_int(value, label="processor input token count")

    if len(initial) > MAXIMUM_FINAL_CANDIDATE_COUNT:
        initial = initial[-MAXIMUM_FINAL_CANDIDATE_COUNT:]
    if len(initial) < MINIMUM_FINAL_CANDIDATE_COUNT:
        raise ValueError("query starts below the frozen four-candidate minimum")
    candidates = initial
    attempts: list[CandidateLengthAttempt] = []
    while len(candidates) >= MINIMUM_FINAL_CANDIDATE_COUNT:
        length = attempted_length(candidates)
        attempts.append(
            CandidateLengthAttempt(
                candidate_event_step_ids=candidates,
                processor_input_token_count=length,
            )
        )
        if length + RESERVED_ACTION_TOKENS <= CONTEXT_LIMIT_TOKENS:
            dropped_count = len(initial) - len(candidates)
            context = FrozenCandidateContext(
                initial_candidate_event_step_ids=initial,
                candidate_event_step_ids=candidates,
                dropped_event_step_ids=initial[:dropped_count],
                processor_input_token_count=length,
                reserved_action_tokens=RESERVED_ACTION_TOKENS,
                context_limit=CONTEXT_LIMIT_TOKENS,
                attempts=tuple(attempts),
            )
            break
        if len(candidates) == MINIMUM_FINAL_CANDIDATE_COUNT:
            raise CandidateMinimumViolation(tuple(attempts))
        candidates = candidates[1:]
    else:
        raise RuntimeError("candidate-freeze loop ended without a terminal outcome")
    return FrozenQueryCandidateRecord(
        state_id=query_payload["state_id"],
        trajectory_id=trajectory_payload["trajectory_id"],
        source_id=trajectory_payload["source_id"],
        role=trajectory_payload["role"],
        query_kind=query_payload["query_kind"],
        decision_step_id=query_payload["decision_step_id"],
        maximum_labeled_cardinality=query_payload[
            "maximum_labeled_cardinality"
        ],
        candidate_context=context,
    )


def build_final_label_schedule(
    records: Sequence[FrozenQueryCandidateRecord],
) -> VariableNLabelSchedule:
    if (
        isinstance(records, (str, bytes, bytearray, Mapping))
        or not isinstance(records, Sequence)
        or not records
    ):
        raise ValueError("frozen query records must be a non-empty sequence")
    if any(not isinstance(record, FrozenQueryCandidateRecord) for record in records):
        raise TypeError("frozen query inventory contains an invalid record")
    schedule = build_variable_n_label_schedule(
        tuple(
            LabelStateRequest(
                state_id=record.state_id,
                candidate_event_ids=record.candidate_context.candidate_event_step_ids,
                maximum_cardinality=record.maximum_labeled_cardinality,
            )
            for record in records
        ),
        worker_count=WORKER_COUNT,
    )
    return validate_variable_n_label_schedule(schedule)


def operation_budget_payload(schedule: VariableNLabelSchedule) -> dict[str, Any]:
    validate_variable_n_label_schedule(schedule)
    operations = schedule.operations
    return {
        "canonical_action_generations": operations.canonical_action_generations,
        "reference_teacher_forwards": operations.reference_teacher_forwards,
        "identical_reference_repeat_teacher_forwards": (
            operations.identical_reference_repeat_teacher_forwards
        ),
        "capped_candidate_teacher_forwards": (
            operations.capped_candidate_teacher_forwards
        ),
        "total_teacher_forwards": operations.total_teacher_forwards,
        "kl_measurements": operations.kl_measurements,
        "raw_label_rows": operations.raw_label_rows,
        "total_model_operations": operations.total_model_operations,
    }


def label_schedule_payload(schedule: VariableNLabelSchedule) -> dict[str, Any]:
    """Materialize exact subsets and deterministic post-freeze label workers."""
    validate_variable_n_label_schedule(schedule)
    return {
        "schema_version": schedule.schema_version,
        "worker_count": schedule.worker_count,
        "subset_identity_sha256": schedule.subset_identity_sha256,
        "inventory_sha256": schedule.inventory_sha256,
        "execution_sha256": schedule.execution_sha256,
        "subset_count_by_cardinality": dict(schedule.subset_count_by_cardinality),
        "state_count_by_candidate_count": dict(
            schedule.state_count_by_candidate_count
        ),
        "state_count_by_maximum_cardinality": dict(
            schedule.state_count_by_maximum_cardinality
        ),
        "operations": operation_budget_payload(schedule),
        "states": [
            {
                "state_id": state.request.state_id,
                "candidate_event_ids": list(state.request.candidate_event_ids),
                "maximum_cardinality": state.request.maximum_cardinality,
                "subsets": [
                    {
                        "subset_index": subset.subset_index,
                        "coalition_event_ids": list(subset.coalition_event_ids),
                    }
                    for subset in state.subsets
                ],
            }
            for state in schedule.states
        ],
        "workers": [
            {
                "worker_index": worker.worker_index,
                "state_ids": list(worker.state_ids),
                "operations": {
                    "canonical_action_generations": (
                        worker.operations.canonical_action_generations
                    ),
                    "reference_teacher_forwards": (
                        worker.operations.reference_teacher_forwards
                    ),
                    "identical_reference_repeat_teacher_forwards": (
                        worker.operations.identical_reference_repeat_teacher_forwards
                    ),
                    "capped_candidate_teacher_forwards": (
                        worker.operations.capped_candidate_teacher_forwards
                    ),
                    "total_teacher_forwards": worker.operations.total_teacher_forwards,
                    "kl_measurements": worker.operations.kl_measurements,
                    "raw_label_rows": worker.operations.raw_label_rows,
                    "total_model_operations": worker.operations.total_model_operations,
                },
            }
            for worker in schedule.workers
        ],
    }


def write_once_or_verify(path: str | Path, payload: bytes) -> str:
    """Create one immutable file, or verify an identical resumable result."""
    target = Path(path)
    if not isinstance(payload, bytes):
        raise TypeError("write-once payload must be bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_file() or target.is_symlink() or target.read_bytes() != payload:
            raise FileExistsError(f"existing immutable output differs: {target}")
        return sha256_bytes(payload)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_bytes(payload)


def assert_processor_only_import_state() -> None:
    """Fail if a Transformers architecture modeling module entered this process."""
    forbidden = sorted(
        name
        for name in sys.modules
        if name.startswith("transformers.models.") and ".modeling_" in name
    )
    if forbidden:
        raise RuntimeError(
            "processor-only execution imported forbidden modeling modules: "
            + ", ".join(forbidden)
        )


def validate_processor_only_source(path: str | Path) -> dict[str, Any]:
    """AST-audit a runner so AutoProcessor is its only pretrained loader."""
    source_path = Path(path)
    if not source_path.is_file() or source_path.is_symlink():
        raise ValueError("processor-only source must be a regular file")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    forbidden_symbols: set[str] = set()
    pretrained_loaders: list[str] = []
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if "AutoModel" in name:
                forbidden_symbols.add(name)
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute):
            if function.attr in {"forward", "generate"}:
                forbidden_calls.append(function.attr)
            if function.attr == "from_pretrained":
                base = function.value
                base_name = base.id if isinstance(base, ast.Name) else ast.dump(base)
                pretrained_loaders.append(base_name)
    if forbidden_symbols or forbidden_calls:
        raise ValueError(
            "processor-only source contains a forbidden model symbol or call: "
            f"symbols={sorted(forbidden_symbols)}, calls={sorted(forbidden_calls)}"
        )
    if pretrained_loaders != ["AutoProcessor"]:
        raise ValueError(
            "processor-only source must contain exactly one AutoProcessor.from_pretrained"
        )
    return {
        "source_sha256": sha256_bytes(source.encode("utf-8")),
        "from_pretrained_loaders": pretrained_loaders,
        "forbidden_model_symbol_count": 0,
        "forbidden_forward_or_generate_call_count": 0,
        "status": "VALID_PROCESSOR_ONLY_SOURCE",
    }


def aggregate_freeze_summary(
    records: Sequence[FrozenQueryCandidateRecord],
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty candidate-freeze inventory")
    state_ids = [record.state_id for record in records]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("candidate-freeze inventory contains duplicate state IDs")
    return {
        "state_count": len(records),
        "trajectory_count": len({record.trajectory_id for record in records}),
        "role_state_counts": dict(sorted(Counter(record.role for record in records).items())),
        "query_kind_counts": dict(
            sorted(Counter(record.query_kind for record in records).items())
        ),
        "initial_candidate_count_histogram": dict(
            sorted(
                Counter(
                    len(record.candidate_context.initial_candidate_event_step_ids)
                    for record in records
                ).items()
            )
        ),
        "final_candidate_count_histogram": dict(
            sorted(
                Counter(
                    len(record.candidate_context.candidate_event_step_ids)
                    for record in records
                ).items()
            )
        ),
        "dropped_candidate_count_histogram": dict(
            sorted(
                Counter(
                    len(record.candidate_context.dropped_event_step_ids)
                    for record in records
                ).items()
            )
        ),
        "state_inventory_sha256": sha256_bytes(
            canonical_json_bytes(
                [record.to_payload() for record in sorted(records, key=lambda row: row.state_id)]
            )
        ),
    }


__all__ = [
    "CONTEXT_LIMIT_TOKENS",
    "CandidateMinimumViolation",
    "FrozenQueryCandidateRecord",
    "MINIMUM_FINAL_CANDIDATE_COUNT",
    "PROTOCOL_ID",
    "RESERVED_ACTION_TOKENS",
    "SCHEMA_VERSION",
    "ShardWorker",
    "ShardWorkerSchedule",
    "TARGET_PIXELS_PER_IMAGE",
    "WORKER_COUNT",
    "aggregate_freeze_summary",
    "assert_processor_only_import_state",
    "build_final_label_schedule",
    "build_full_reference_prompt_plan",
    "build_shard_worker_schedule",
    "canonical_json_bytes",
    "freeze_query_candidates",
    "label_schedule_payload",
    "operation_budget_payload",
    "sha256_bytes",
    "sha256_file",
    "worker_read_plan",
    "verify_processor_only_snapshot",
    "validate_freeze_query_topology",
    "validate_processor_only_source",
    "write_once_or_verify",
]
