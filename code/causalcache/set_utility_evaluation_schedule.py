"""Label-blind held-out coalition schedules for frozen set selectors."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVALUATION_ROLE = "evaluation"
EXACT_TRACK = "exact_oracle"
LARGE_HISTORY_TRACK = "large_history"
TRACKS = (EXACT_TRACK, LARGE_HISTORY_TRACK)
SHARD_STATUS = "COMPLETED_VARIABLE_HISTORY_EVALUATION_SCHEDULE_SHARD"
PARTITION_STATUS = "COMPLETED_VARIABLE_HISTORY_EVALUATION_SCHEDULE_PARTITION"
SOURCE_STATUS = "COMPLETED_VARIABLE_HISTORY_SOURCE"
INVENTORY_STATUS = "FROZEN_VARIABLE_HISTORY_STATE_INVENTORY"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_METHOD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    keyword: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        keyword["indent"] = 2
    else:
        keyword["separators"] = (",", ":")
    return (json.dumps(value, **keyword) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _positive_count(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class EvaluationScheduleConfig:
    budgets: tuple[int, ...]
    exact_state_count: int
    large_history_state_count: int
    union_state_count: int
    logical_shard_count: int
    random_control_seed: int
    random_control_rule: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationScheduleConfig":
        root = _mapping(value, label="evaluation schedule config")
        section = _mapping(root.get("evaluation_schedule"), label="evaluation_schedule")
        budgets = tuple(_sequence(section.get("budgets"), label="evaluation budgets"))
        if (
            budgets != tuple(sorted(budgets))
            or len(set(budgets)) != len(budgets)
            or any(type(item) is not int or item <= 0 for item in budgets)
        ):
            raise ValueError(
                "evaluation budgets must be sorted unique positive integers"
            )
        if budgets != (1, 2, 3, 4):
            raise ValueError("held-out evaluation budgets must be exactly 1,2,3,4")
        random_control = _mapping(
            section.get("random_control"), label="random_control"
        )
        seed = random_control.get("seed")
        if type(seed) is not int:
            raise ValueError("random-control seed must be an integer")
        rule = random_control.get("rule")
        expected_rule = "sha256_seed_state_event_nested_prefix"
        if rule != expected_rule:
            raise ValueError(f"random-control rule must be {expected_rule}")
        return cls(
            budgets=budgets,
            exact_state_count=_positive_count(
                section.get("exact_state_count"), label="exact state count"
            ),
            large_history_state_count=_positive_count(
                section.get("large_history_state_count"),
                label="large-history state count",
            ),
            union_state_count=_positive_count(
                section.get("union_state_count"), label="union state count"
            ),
            logical_shard_count=_positive_count(
                section.get("logical_shard_count"), label="logical shard count"
            ),
            random_control_seed=seed,
            random_control_rule=rule,
        )


@dataclass(frozen=True)
class EvaluationTracks:
    exact_state_ids: frozenset[str]
    large_history_state_ids: frozenset[str]

    @property
    def union_state_ids(self) -> frozenset[str]:
        return self.exact_state_ids | self.large_history_state_ids

    def tags(self, state_id: str) -> tuple[str, ...]:
        result = []
        if state_id in self.exact_state_ids:
            result.append(EXACT_TRACK)
        if state_id in self.large_history_state_ids:
            result.append(LARGE_HISTORY_TRACK)
        return tuple(result)


def evaluation_tracks_from_inventory(
    inventory: Mapping[str, Any],
    *,
    config: EvaluationScheduleConfig,
) -> EvaluationTracks:
    value = _mapping(inventory, label="state inventory")
    if value.get("status") != INVENTORY_STATUS:
        raise ValueError("state inventory is not frozen")
    raw = _mapping(value.get("evaluation_tracks"), label="inventory evaluation tracks")

    def state_set(key: str) -> frozenset[str]:
        sequence = tuple(_sequence(raw.get(key), label=key))
        if any(not isinstance(item, str) or not item for item in sequence):
            raise ValueError(f"{key} must contain non-empty state IDs")
        if len(sequence) != len(set(sequence)):
            raise ValueError(f"{key} contains duplicate state IDs")
        return frozenset(sequence)

    tracks = EvaluationTracks(
        exact_state_ids=state_set("exact_state_ids"),
        large_history_state_ids=state_set("large_history_state_ids"),
    )
    observed = (
        len(tracks.exact_state_ids),
        len(tracks.large_history_state_ids),
        len(tracks.union_state_ids),
    )
    expected = (
        config.exact_state_count,
        config.large_history_state_count,
        config.union_state_count,
    )
    if observed != expected:
        raise ValueError(
            f"evaluation track counts drifted: expected {expected}, got {observed}"
        )
    return tracks


@dataclass(frozen=True)
class SelectorState:
    state_id: str
    trajectory_id: str
    logical_shard: int
    candidate_event_ids: tuple[int, ...]
    tracks: tuple[str, ...]
    methods: Mapping[str, Mapping[int, tuple[int, ...]]]


def _canonical_subset(
    value: Any,
    *,
    candidates: tuple[int, ...],
    budget: int,
    label: str,
) -> tuple[int, ...]:
    subset = tuple(_sequence(value, label=label))
    if (
        any(type(item) is not int for item in subset)
        or subset != tuple(sorted(subset))
        or len(subset) != len(set(subset))
        or len(subset) > budget
        or not set(subset).issubset(candidates)
    ):
        raise ValueError(f"{label} is not a canonical at-most-{budget} subset")
    return subset


def selector_states_from_payload(
    payload: Mapping[str, Any],
    *,
    config: EvaluationScheduleConfig,
    tracks: EvaluationTracks,
) -> tuple[SelectorState, ...]:
    root = _mapping(payload, label="sealed selector selections")
    records = tuple(_sequence(root.get("records"), label="selector records"))
    if len(records) != config.union_state_count:
        raise ValueError("selector record count differs from the evaluation union")
    results = []
    method_inventory: tuple[str, ...] | None = None
    seen = set()
    for ordinal, raw_record in enumerate(records):
        record = _mapping(raw_record, label=f"selector record {ordinal}")
        required = {
            "state_id",
            "trajectory_id",
            "logical_shard",
            "candidate_event_ids",
            "tracks",
            "methods",
        }
        allowed = required | {"model_metrics"}
        if not required.issubset(record) or not set(record).issubset(allowed):
            raise ValueError(f"selector record {ordinal} keys drifted")
        state_id = record["state_id"]
        trajectory_id = record["trajectory_id"]
        logical_shard = record["logical_shard"]
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("selector state_id must be non-empty text")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("selector trajectory_id must be non-empty text")
        if state_id in seen:
            raise ValueError("selector state IDs must be unique")
        seen.add(state_id)
        if state_id not in tracks.union_state_ids:
            raise ValueError("selector state is outside the frozen evaluation union")
        if type(logical_shard) is not int or not (
            0 <= logical_shard < config.logical_shard_count
        ):
            raise ValueError("selector logical shard is out of range")
        candidates = tuple(
            _sequence(record["candidate_event_ids"], label="candidate_event_ids")
        )
        if candidates != tuple(range(1, len(candidates) + 1)) or len(candidates) < 5:
            raise ValueError("selector candidates must be the complete history from 1")
        expected_state_id = f"{trajectory_id}:decision:{len(candidates) + 1:03d}"
        if state_id != expected_state_id:
            raise ValueError(
                "selector state identity differs from its candidate universe"
            )
        record_tracks = tuple(
            _sequence(record["tracks"], label="selector tracks")
        )
        if (
            any(item not in TRACKS for item in record_tracks)
            or len(record_tracks) != len(set(record_tracks))
            or set(record_tracks) != set(tracks.tags(state_id))
        ):
            raise ValueError("selector track tags differ from the frozen inventory")
        methods_raw = _mapping(record["methods"], label="selector methods")
        method_names = tuple(sorted(methods_raw))
        if not method_names or any(
            not isinstance(name, str) or _METHOD_NAME.fullmatch(name) is None
            for name in method_names
        ):
            raise ValueError("selector method names are invalid")
        if method_inventory is None:
            method_inventory = method_names
        elif method_names != method_inventory:
            raise ValueError("selector method inventory differs across states")
        methods: dict[str, dict[int, tuple[int, ...]]] = {}
        expected_budget_keys = {str(budget) for budget in config.budgets}
        for method in method_names:
            by_budget = _mapping(
                methods_raw[method], label=f"selector method {method}"
            )
            if set(by_budget) != expected_budget_keys:
                raise ValueError("selector method budget inventory drifted")
            methods[method] = {
                budget: _canonical_subset(
                    by_budget[str(budget)],
                    candidates=candidates,
                    budget=budget,
                    label=f"{method} budget {budget}",
                )
                for budget in config.budgets
            }
        results.append(
            SelectorState(
                state_id=state_id,
                trajectory_id=trajectory_id,
                logical_shard=logical_shard,
                candidate_event_ids=candidates,
                tracks=tracks.tags(state_id),
                methods=methods,
            )
        )
    if seen != set(tracks.union_state_ids):
        raise ValueError("selector records do not exactly cover the evaluation union")
    return tuple(sorted(results, key=lambda item: item.state_id))


def source_trajectory_shards(
    manifest: Mapping[str, Any],
    *,
    logical_shard_count: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    value = _mapping(manifest, label="source manifest")
    if value.get("status") != SOURCE_STATUS:
        raise ValueError("variable-history source manifest is incomplete")
    shards_raw = tuple(_sequence(value.get("shards"), label="source shards"))
    shards: dict[int, dict[str, Any]] = {}
    trajectories: dict[str, int] = {}
    for raw in shards_raw:
        shard = dict(_mapping(raw, label="source shard"))
        logical = shard.get("logical_shard")
        if type(logical) is not int or not 0 <= logical < logical_shard_count:
            raise ValueError("source logical shard is out of range")
        if logical in shards:
            raise ValueError("source logical shards must be unique")
        _sha256(shard.get("sha256"), label="source shard SHA256")
        trajectory_ids = tuple(
            _sequence(shard.get("trajectory_ids"), label="source trajectory IDs")
        )
        if any(not isinstance(item, str) or not item for item in trajectory_ids):
            raise ValueError("source trajectory IDs must be non-empty text")
        for trajectory_id in trajectory_ids:
            if trajectory_id in trajectories:
                raise ValueError("source trajectory appears in multiple logical shards")
            trajectories[trajectory_id] = logical
        shards[logical] = shard
    if set(shards) != set(range(logical_shard_count)):
        raise ValueError("source manifest does not cover every logical shard")
    return tuple(shards[index] for index in range(logical_shard_count)), trajectories


def deterministic_random_subset(
    state: SelectorState,
    *,
    budget: int,
    seed: int,
) -> tuple[int, ...]:
    if budget <= 0:
        raise ValueError("random-control budget must be positive")
    ranked = sorted(
        state.candidate_event_ids,
        key=lambda event_id: (
            hashlib.sha256(
                f"{seed}\0{state.state_id}\0{event_id}".encode("utf-8")
            ).digest(),
            event_id,
        ),
    )
    return tuple(sorted(ranked[: min(budget, len(ranked))]))


def build_evaluation_schedule(
    state: SelectorState,
    *,
    config: EvaluationScheduleConfig,
) -> dict[str, Any]:
    candidates = state.candidate_event_ids
    tags: dict[tuple[int, ...], set[str]] = {}

    def add(subset: tuple[int, ...], source: str) -> None:
        tags.setdefault(subset, set()).add(source)

    add((), "anchor:empty")
    add(candidates, "anchor:full")
    if EXACT_TRACK in state.tracks:
        add((), "exact:cardinality:0")
        for cardinality in (1, 2):
            for subset in itertools.combinations(candidates, cardinality):
                add(subset, f"exact:cardinality:{cardinality}")
        add(candidates, "exact:full")
    for method in sorted(state.methods):
        for budget in config.budgets:
            add(state.methods[method][budget], f"selector:{method}:B{budget}")
    for budget in config.budgets:
        add(
            deterministic_random_subset(
                state, budget=budget, seed=config.random_control_seed
            ),
            f"random:B{budget}",
        )
    coalitions = []
    for subset in sorted(tags, key=lambda item: (len(item), item)):
        sources = sorted(tags[subset])
        coalitions.append(
            {
                "event_ids": list(subset),
                "source": sources[0],
                "sources": sources,
            }
        )
    return {
        "candidate_event_ids": list(candidates),
        "coalitions": coalitions,
        "logical_shard": state.logical_shard,
        "role": EVALUATION_ROLE,
        "state_id": state.state_id,
        "tracks": list(state.tracks),
        "trajectory_id": state.trajectory_id,
    }


def _schedule_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        canonical_json_bytes(row)
        for row in sorted(rows, key=lambda item: str(item["state_id"]))
    )


def _payload_statistics(payload: bytes, *, logical_shard: int) -> dict[str, Any]:
    rows = tuple(
        json.loads(line)
        for line in payload.decode("utf-8").splitlines()
        if line
    )
    source_counts = Counter()
    method_names = set()
    coalition_count = 0
    forward_count = 0
    for row in rows:
        if (
            row.get("role") != EVALUATION_ROLE
            or row.get("logical_shard") != logical_shard
        ):
            raise ValueError("schedule payload role or logical shard drifted")
        candidates = tuple(row["candidate_event_ids"])
        coalitions = tuple(row["coalitions"])
        if not any(tuple(item["event_ids"]) == candidates for item in coalitions):
            raise ValueError("evaluation schedule is missing its full reference anchor")
        if not any(not item["event_ids"] for item in coalitions):
            raise ValueError("evaluation schedule is missing its empty utility anchor")
        coalition_count += len(coalitions)
        forward_count += sum(
            tuple(item["event_ids"]) != candidates for item in coalitions
        )
        for item in coalitions:
            sources = tuple(item["sources"])
            if item.get("source") != min(sources) or sources != tuple(
                sorted(set(sources))
            ):
                raise ValueError("evaluation coalition source tags drifted")
            source_counts.update(sources)
            for source in sources:
                if source.startswith("selector:"):
                    method_names.add(source.split(":", 2)[1])
    return {
        "coalition_count": coalition_count,
        "forward_coalition_count": forward_count,
        "method_names": sorted(method_names),
        "source_component_counts": dict(sorted(source_counts.items())),
        "state_count": len(rows),
    }


def _validated_existing(
    schedule_path: Path,
    receipt_path: Path,
    *,
    identity: str,
    bindings: Mapping[str, Any],
    logical_shard: int,
) -> dict[str, Any] | None:
    if not schedule_path.exists() and not receipt_path.exists():
        return None
    if not schedule_path.exists() or not receipt_path.exists():
        raise ValueError("incomplete evaluation schedule/receipt pair cannot resume")
    payload = schedule_path.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    statistics = _payload_statistics(payload, logical_shard=logical_shard)
    expected = {
        **statistics,
        **dict(bindings),
        "identity_sha256": identity,
        "logical_shard": logical_shard,
        "schedule_byte_count": len(payload),
        "schedule_sha256": hashlib.sha256(payload).hexdigest(),
        "status": SHARD_STATUS,
    }
    if receipt != expected:
        raise ValueError("existing evaluation schedule receipt drifted")
    return receipt


def materialize_evaluation_schedules(
    *,
    config_path: Path,
    inventory_path: Path,
    selections_path: Path,
    source_manifest_path: Path,
    output_root: Path,
    partition_index: int,
    partition_count: int,
    workers: int,
) -> dict[str, Any]:
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = EvaluationScheduleConfig.from_mapping(config_payload)
    if not 0 <= partition_index < partition_count:
        raise ValueError("partition index is outside partition count")
    if workers <= 0:
        raise ValueError("worker count must be positive")
    inventory_payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    tracks = evaluation_tracks_from_inventory(inventory_payload, config=config)
    selections_payload = json.loads(selections_path.read_text(encoding="utf-8"))
    states = selector_states_from_payload(
        selections_payload, config=config, tracks=tracks
    )
    source_payload = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_shards, trajectory_shards = source_trajectory_shards(
        source_payload, logical_shard_count=config.logical_shard_count
    )
    for state in states:
        if trajectory_shards.get(state.trajectory_id) != state.logical_shard:
            raise ValueError("selector logical shard differs from the source manifest")
    states_by_shard: dict[int, list[SelectorState]] = {
        index: [] for index in range(config.logical_shard_count)
    }
    for state in states:
        states_by_shard[state.logical_shard].append(state)

    hashes = {
        "config_sha256": sha256_file(config_path),
        "inventory_sha256": sha256_file(inventory_path),
        "selections_sha256": sha256_file(selections_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
    }
    selected_shards = tuple(
        shard
        for shard in source_shards
        if int(shard["logical_shard"]) % partition_count == partition_index
    )

    def build(source_shard: Mapping[str, Any]) -> dict[str, Any]:
        logical_shard = int(source_shard["logical_shard"])
        source_shard_sha = _sha256(
            source_shard["sha256"], label="source shard SHA256"
        )
        bindings = {**hashes, "source_shard_sha256": source_shard_sha}
        identity = hashlib.sha256(
            canonical_json_bytes(
                {**bindings, "logical_shard": logical_shard}, pretty=False
            )
        ).hexdigest()
        schedule_path = (
            output_root
            / "schedule-shards"
            / f"shard-{logical_shard:03d}-of-{config.logical_shard_count}.jsonl"
        )
        receipt_path = (
            output_root
            / "receipts"
            / f"shard-{logical_shard:03d}-of-{config.logical_shard_count}.json"
        )
        existing = _validated_existing(
            schedule_path,
            receipt_path,
            identity=identity,
            bindings=bindings,
            logical_shard=logical_shard,
        )
        if existing is not None:
            return existing
        rows = tuple(
            build_evaluation_schedule(state, config=config)
            for state in states_by_shard[logical_shard]
        )
        payload = _schedule_payload(rows)
        statistics = _payload_statistics(payload, logical_shard=logical_shard)
        receipt = {
            **statistics,
            **bindings,
            "identity_sha256": identity,
            "logical_shard": logical_shard,
            "schedule_byte_count": len(payload),
            "schedule_sha256": hashlib.sha256(payload).hexdigest(),
            "status": SHARD_STATUS,
        }
        _write_atomic(schedule_path, payload)
        _write_atomic(receipt_path, canonical_json_bytes(receipt, pretty=True))
        return receipt

    with ThreadPoolExecutor(max_workers=workers) as executor:
        receipts = list(executor.map(build, selected_shards))
    summary = {
        **hashes,
        "coalition_count": sum(item["coalition_count"] for item in receipts),
        "completed_during_or_before_invocation": len(receipts),
        "forward_coalition_count": sum(
            item["forward_coalition_count"] for item in receipts
        ),
        "method_names": sorted(
            {name for item in receipts for name in item["method_names"]}
        ),
        "partition_count": partition_count,
        "partition_index": partition_index,
        "state_count": sum(item["state_count"] for item in receipts),
        "status": PARTITION_STATUS,
    }
    worker_path = (
        output_root
        / "workers"
        / f"worker-{partition_index:03d}-of-{partition_count:03d}.json"
    )
    _write_atomic(worker_path, canonical_json_bytes(summary, pretty=True))
    return summary


__all__ = [
    "EVALUATION_ROLE",
    "EvaluationScheduleConfig",
    "EvaluationTracks",
    "SelectorState",
    "build_evaluation_schedule",
    "deterministic_random_subset",
    "evaluation_tracks_from_inventory",
    "materialize_evaluation_schedules",
    "selector_states_from_payload",
    "source_trajectory_shards",
]
