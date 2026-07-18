"""Policy-blind full-pool census for scalable set-utility data construction."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from causalcache.data.guiodyssey_independent import (
    Candidate,
    ExclusionReason,
    SourceFileSpec,
    inspect_candidate,
    normalize_app_label,
    validate_protocol_config,
)
from causalcache.set_utility_consumed_ledger import (
    EXPECTED_PARTITION_COUNTS,
    EXPECTED_UNION_COUNT,
    ROLE_SPECS,
    STATUS as CONSUMED_LEDGER_STATUS,
    source_ids_sha256,
    source_set_sha256,
)
from causalcache.set_utility_consumed_ledger_contract import (
    PROTOCOL_ID as CONSUMED_LEDGER_PROTOCOL_ID,
    SCHEMA_VERSION as CONSUMED_LEDGER_SCHEMA_VERSION,
    assignment_inventory_sha256,
    canonical_assignments,
)
from causalcache.set_utility_full_pool_inventory_v1 import (
    COMPLETE_STATUS as SOURCE_INVENTORY_COMPLETE_STATUS,
    EXPECTED_DIRECTORY,
    EXPECTED_FILE_COUNT,
    EXPECTED_REPO_ID,
    EXPECTED_REVISION,
    FROZEN_CONFIG_SHA256 as SOURCE_INVENTORY_CONFIG_SHA256,
    PROTOCOL_ID as SOURCE_INVENTORY_PROTOCOL_ID,
    SCHEMA_VERSION as SOURCE_INVENTORY_SCHEMA_VERSION,
    canonical_json_bytes,
    sha256_bytes,
)
from causalcache.set_utility_long_pool import instruction_app_group_sha256


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_full_pool_census_v2"
MINIMUM_DECISION_COUNT = 6
TOTAL_GUIODYSSEY_TRAIN_SHARDS = EXPECTED_FILE_COUNT
COMPATIBILITY_MAXIMUM_DECISION_COUNT = (1 << 63) - 1
SELECTION_SALT = "causalcache-set-utility-full-pool-census-v2"
FORMAT_SAFETY_EXCLUSION = "decision_count_above_format_safety_ceiling"
UNCONSUMED_MARKER = "UNCONSUMED"

_SHARD_PATH = re.compile(
    r"mobile/use/train/shard-(?P<index>[0-9]{5})-of-00610[.]parquet"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys drifted")


def _strict_positive_integer(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _normalized_text(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be non-empty normalized text")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def validate_complete_source_manifest(
    manifest: Mapping[str, Any],
    *,
    base_config: Mapping[str, Any],
) -> tuple[SourceFileSpec, ...]:
    """Validate the exact P-1 byte inventory for all 610 train shards."""
    manifest = _mapping(manifest, label="full-pool source manifest")
    validate_protocol_config(base_config)
    _exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "status",
            "source",
            "inventory",
            "authorization_scope",
        },
        label="full-pool source manifest",
    )
    if (
        manifest.get("schema_version") != SOURCE_INVENTORY_SCHEMA_VERSION
        or manifest.get("protocol_id") != SOURCE_INVENTORY_PROTOCOL_ID
        or manifest.get("status") != SOURCE_INVENTORY_COMPLETE_STATUS
    ):
        raise ValueError("full-pool source manifest identity drifted")

    source = _mapping(manifest.get("source"), label="source manifest source")
    _exact_keys(
        source,
        {
            "repo_id",
            "repo_type",
            "revision",
            "target_directory",
            "config_sha256",
        },
        label="source manifest source",
    )
    source_pool = base_config["source_pool"]
    expected_source = {
        "repo_id": EXPECTED_REPO_ID,
        "repo_type": "dataset",
        "revision": EXPECTED_REVISION,
        "target_directory": EXPECTED_DIRECTORY,
        "config_sha256": SOURCE_INVENTORY_CONFIG_SHA256,
    }
    if dict(source) != expected_source:
        raise ValueError("full-pool source manifest source binding drifted")
    if (
        source["repo_id"] != source_pool["transport_repo"]
        or source["revision"] != source_pool["transport_revision"]
    ):
        raise ValueError("full-pool source manifest disagrees with parser base config")

    inventory = _mapping(
        manifest.get("inventory"), label="source manifest inventory"
    )
    _exact_keys(
        inventory,
        {"file_count", "total_size_bytes", "files_sha256", "files"},
        label="source manifest inventory",
    )
    records = _sequence(inventory.get("files"), label="source manifest files")
    if (
        inventory.get("file_count") != TOTAL_GUIODYSSEY_TRAIN_SHARDS
        or len(records) != TOTAL_GUIODYSSEY_TRAIN_SHARDS
    ):
        raise ValueError("full-pool source manifest must contain exactly 610 shards")
    declared_inventory_hash = _sha256(
        inventory.get("files_sha256"), label="source manifest files_sha256"
    )
    if sha256_bytes(canonical_json_bytes(records)) != declared_inventory_hash:
        raise ValueError("full-pool source manifest files_sha256 mismatch")

    specs: list[SourceFileSpec] = []
    for expected_index, raw_record in enumerate(records):
        record = _mapping(raw_record, label="full-pool source file record")
        _exact_keys(
            record,
            {"path", "size_bytes", "lfs_sha256"},
            label="full-pool source file record",
        )
        path = record.get("path")
        match = _SHARD_PATH.fullmatch(path) if isinstance(path, str) else None
        if match is None or int(match.group("index")) != expected_index:
            raise ValueError(
                "full-pool source paths must be the complete canonical 00000..00609 order"
            )
        specs.append(
            SourceFileSpec(
                path,
                _strict_positive_integer(
                    record.get("size_bytes"), label="source file size_bytes"
                ),
                _sha256(record.get("lfs_sha256"), label="source file lfs_sha256"),
            )
        )
    total_bytes = _strict_positive_integer(
        inventory.get("total_size_bytes"), label="source total_size_bytes"
    )
    if sum(spec.size_bytes for spec in specs) != total_bytes:
        raise ValueError("full-pool source manifest total_size_bytes mismatch")

    authorization_scope = _mapping(
        manifest.get("authorization_scope"), label="source authorization_scope"
    )
    expected_scope = {
        "remote_metadata_inventory_only": True,
        "file_download_or_row_decode": False,
        "semantic_census_or_role_assignment": False,
        "restoration_labels_or_training": False,
        "model_or_gpu_operations": False,
        "hugging_face_mutation": False,
    }
    if dict(authorization_scope) != expected_scope:
        raise ValueError("full-pool source manifest authorization scope drifted")
    return tuple(specs)


def build_full_pool_inspection_config(
    base_config: Mapping[str, Any],
    *,
    transport_files: Sequence[str],
    format_safety_maximum_decisions: int | None = None,
) -> dict[str, Any]:
    """Adapt the frozen parser without preserving its old scientific max of 12."""
    validate_protocol_config(base_config)
    if not transport_files or len(set(transport_files)) != len(transport_files):
        raise ValueError("full-pool transport files must be non-empty and unique")
    if format_safety_maximum_decisions is None:
        ceiling = COMPATIBILITY_MAXIMUM_DECISION_COUNT
    else:
        ceiling = _strict_positive_integer(
            format_safety_maximum_decisions,
            label="format safety maximum decisions",
        )
        if ceiling < MINIMUM_DECISION_COUNT:
            raise ValueError("format safety ceiling cannot be below the census minimum")

    result = copy.deepcopy(dict(base_config))
    result["source_pool"]["transport_files"] = list(transport_files)
    result["eligibility"]["minimum_decisions_per_trajectory"] = (
        MINIMUM_DECISION_COUNT
    )
    result["eligibility"]["maximum_decisions_per_trajectory"] = ceiling
    result["selection"]["trajectory_salt"] = SELECTION_SALT
    validate_protocol_config(result)
    return result


@dataclass(frozen=True)
class ConsumedLedgerBinding:
    legacy_train_only_source_ids: frozenset[str]
    forbidden_consumed_source_ids: frozenset[str]
    assignment_inventory_sha256: str
    consumed_union_source_set_sha256: str
    manifest_sha256: str

    @property
    def role_by_source(self) -> dict[str, str]:
        result = {
            source_id: "legacy_train_only"
            for source_id in self.legacy_train_only_source_ids
        }
        result.update(
            {
                source_id: "forbidden_consumed"
                for source_id in self.forbidden_consumed_source_ids
            }
        )
        return result

    @property
    def source_to_role(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self.role_by_source.items()))


def _validate_role_inventories(
    manifest: Mapping[str, Any],
    *,
    partition_sets: Mapping[str, frozenset[str]],
) -> None:
    records = _sequence(
        manifest.get("role_inventories"), label="consumed ledger role_inventories"
    )
    if len(records) != len(ROLE_SPECS):
        raise ValueError("consumed ledger role inventory count drifted")
    partition_ordered_ids: dict[str, list[str]] = {
        partition: [] for partition in EXPECTED_PARTITION_COUNTS
    }
    seen_ids: set[str] = set()
    for raw_record, spec in zip(records, ROLE_SPECS, strict=True):
        record = _mapping(raw_record, label="consumed ledger role inventory")
        _exact_keys(
            record,
            {
                "ledger_role",
                "partition",
                "source_manifest_role",
                "source_id_count",
                "source_ids_sha256",
                "source_set_sha256",
                "source_ids",
            },
            label="consumed ledger role inventory",
        )
        (
            ledger_role,
            partition,
            _origin,
            source_manifest_role,
            _roster_name,
            expected_count,
        ) = spec
        if (
            record.get("ledger_role") != ledger_role
            or record.get("partition") != partition
            or record.get("source_manifest_role") != source_manifest_role
            or record.get("source_id_count") != expected_count
        ):
            raise ValueError("consumed ledger role inventory identity drifted")
        source_ids = tuple(
            _normalized_text(value, label="consumed ledger role source_id")
            for value in _sequence(
                record.get("source_ids"), label="consumed ledger role source_ids"
            )
        )
        if len(source_ids) != expected_count or len(set(source_ids)) != len(source_ids):
            raise ValueError("consumed ledger role source identity count drifted")
        if seen_ids & set(source_ids):
            raise ValueError("consumed ledger role source identities overlap")
        seen_ids.update(source_ids)
        if source_ids_sha256(source_ids) != record.get("source_ids_sha256"):
            raise ValueError("consumed ledger role source_ids_sha256 mismatch")
        if source_set_sha256(source_ids) != record.get("source_set_sha256"):
            raise ValueError("consumed ledger role source_set_sha256 mismatch")
        if not set(source_ids) <= partition_sets[partition]:
            raise ValueError("consumed ledger role identity escaped its partition")
        partition_ordered_ids[partition].extend(source_ids)
    if seen_ids != set().union(*partition_sets.values()):
        raise ValueError("consumed ledger role inventories do not cover the union")
    partitions = _mapping(manifest.get("partitions"), label="consumed partitions")
    for partition, source_ids in partition_ordered_ids.items():
        record = _mapping(partitions.get(partition), label=f"partition {partition}")
        if source_ids_sha256(tuple(source_ids)) != record.get("source_ids_sha256"):
            raise ValueError(f"consumed ledger partition {partition} order hash mismatch")


def parse_consumed_ledger(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> ConsumedLedgerBinding:
    """Validate the shared canonical consumed-identity firewall payload."""
    manifest = _mapping(manifest, label="consumed ledger")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "status",
            "assignment_count",
            "assignment_inventory_sha256",
            "assignments",
            "inputs",
            "role_inventories",
            "partitions",
            "consumed_union",
            "overlap_audit",
        },
        label="consumed ledger",
    )
    if (
        manifest.get("schema_version") != CONSUMED_LEDGER_SCHEMA_VERSION
        or manifest.get("protocol_id") != CONSUMED_LEDGER_PROTOCOL_ID
        or manifest.get("status") != CONSUMED_LEDGER_STATUS
    ):
        raise ValueError("consumed ledger identity drifted")
    assignments = canonical_assignments(manifest.get("assignments"))
    if list(assignments) != list(manifest["assignments"]):
        raise ValueError("consumed ledger assignments are not canonical")
    if (
        manifest.get("assignment_count") != EXPECTED_UNION_COUNT
        or len(assignments) != EXPECTED_UNION_COUNT
    ):
        raise ValueError("consumed ledger assignment count drifted")
    assignment_hash = assignment_inventory_sha256(assignments)
    if manifest.get("assignment_inventory_sha256") != assignment_hash:
        raise ValueError("consumed ledger assignment inventory hash mismatch")

    role_by_source = {record["source_id"]: record["role"] for record in assignments}
    partition_sets = {
        partition: frozenset(
            source_id
            for source_id, role in role_by_source.items()
            if role == partition
        )
        for partition in EXPECTED_PARTITION_COUNTS
    }
    if {
        partition: len(source_ids)
        for partition, source_ids in partition_sets.items()
    } != EXPECTED_PARTITION_COUNTS:
        raise ValueError("consumed ledger partition assignment counts drifted")

    partitions = _mapping(manifest.get("partitions"), label="consumed partitions")
    if set(partitions) != set(EXPECTED_PARTITION_COUNTS):
        raise ValueError("consumed ledger partition names drifted")
    for partition, expected_count in EXPECTED_PARTITION_COUNTS.items():
        record = _mapping(partitions.get(partition), label=f"partition {partition}")
        _exact_keys(
            record,
            {"source_id_count", "source_ids_sha256", "source_set_sha256"},
            label=f"partition {partition}",
        )
        _sha256(record.get("source_ids_sha256"), label=f"{partition} source_ids_sha256")
        if (
            record.get("source_id_count") != expected_count
            or record.get("source_set_sha256")
            != source_set_sha256(tuple(partition_sets[partition]))
        ):
            raise ValueError(f"consumed ledger partition {partition} drifted")

    union_record = _mapping(
        manifest.get("consumed_union"), label="consumed ledger union"
    )
    _exact_keys(
        union_record,
        {
            "source_id_count",
            "source_set_sha256",
            "assignment_inventory_sha256",
        },
        label="consumed ledger union",
    )
    union_ids = frozenset(role_by_source)
    if (
        union_record.get("source_id_count") != EXPECTED_UNION_COUNT
        or union_record.get("source_set_sha256")
        != source_set_sha256(tuple(union_ids))
        or union_record.get("assignment_inventory_sha256") != assignment_hash
    ):
        raise ValueError("consumed ledger union drifted")
    _validate_role_inventories(manifest, partition_sets=partition_sets)

    inputs = _sequence(manifest.get("inputs"), label="consumed ledger inputs")
    for raw_input in inputs:
        input_record = _mapping(raw_input, label="consumed ledger input")
        _exact_keys(input_record, {"path", "sha256"}, label="consumed ledger input")
        _normalized_text(input_record.get("path"), label="consumed ledger input path")
        _sha256(input_record.get("sha256"), label="consumed ledger input sha256")
    overlap = _mapping(
        manifest.get("overlap_audit"), label="consumed ledger overlap audit"
    )
    if (
        overlap.get("nonempty_role_pair_count") != 0
        or overlap.get("legacy_forbidden_intersection_count") != 0
    ):
        raise ValueError("consumed ledger overlap audit is not empty")
    return ConsumedLedgerBinding(
        legacy_train_only_source_ids=partition_sets["legacy_train_only"],
        forbidden_consumed_source_ids=partition_sets["forbidden_consumed"],
        assignment_inventory_sha256=assignment_hash,
        consumed_union_source_set_sha256=str(union_record["source_set_sha256"]),
        manifest_sha256=_sha256(
            manifest_sha256, label="consumed ledger manifest sha256"
        ),
    )


def candidate_capacity_stratum(decision_count: int) -> str:
    if type(decision_count) is not int:
        raise TypeError("decision_count must be an integer")
    if 6 <= decision_count <= 9:
        return "decisions_6_9"
    if 10 <= decision_count <= 17:
        return "decisions_10_17"
    if decision_count >= 18:
        return "decisions_18_plus"
    raise ValueError("candidate is below the full-pool census minimum")


def supported_candidate_counts(decision_count: int) -> tuple[int, ...]:
    stratum = candidate_capacity_stratum(decision_count)
    if stratum == "decisions_6_9":
        return (4,)
    if stratum == "decisions_10_17":
        return (4, 8)
    return (4, 8, 16)


def _first_instruction_text(row: Mapping[str, Any]) -> str:
    serialized = row.get("messages")
    if not isinstance(serialized, str):
        raise ValueError("included source row messages must be serialized JSON")
    try:
        messages = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ValueError("included source row messages are not valid JSON") from error
    if not isinstance(messages, list):
        raise ValueError("included source row messages must decode to a list")
    for turn in messages:
        if not isinstance(turn, Mapping) or turn.get("role") != "user":
            continue
        content = turn.get("content")
        if not isinstance(content, list):
            break
        for item in content:
            if (
                isinstance(item, Mapping)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
                and item["text"]
            ):
                return item["text"]
        break
    raise ValueError("included source row is missing its first user instruction")


def _candidate_record(candidate: Candidate, *, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": candidate.source_id,
        "transport_file": candidate.transport_file,
        "transport_row_index": candidate.transport_row_index,
        "selection_sha256": candidate.selection_sha256,
        "decision_count": candidate.decision_count,
        "candidate_capacity_stratum": candidate_capacity_stratum(
            candidate.decision_count
        ),
        "supported_candidate_counts": list(
            supported_candidate_counts(candidate.decision_count)
        ),
        "normalized_app_labels": list(candidate.normalized_app_labels),
        "action_type_counts": candidate.action_counts(),
        "instruction_app_group_sha256": instruction_app_group_sha256(
            _first_instruction_text(row),
            normalized_app_labels=candidate.normalized_app_labels,
        ),
        "consumed_role_marker": UNCONSUMED_MARKER,
    }


def _consumed_group_record(
    *,
    row: Mapping[str, Any],
    source_id: str,
    partition: str,
) -> dict[str, str]:
    serialized_metadata = row.get("metadata")
    if not isinstance(serialized_metadata, str):
        raise ValueError("consumed row metadata must be serialized JSON")
    try:
        metadata = json.loads(serialized_metadata)
    except json.JSONDecodeError as error:
        raise ValueError("consumed row metadata is not valid JSON") from error
    others = metadata.get("others") if isinstance(metadata, Mapping) else None
    apps = others.get("apps") if isinstance(others, Mapping) else None
    if (
        not isinstance(apps, list)
        or not apps
        or any(not isinstance(value, str) for value in apps)
    ):
        raise ValueError("consumed row is missing app labels for group firewall")
    normalized_apps = tuple(
        sorted(
            {
                normalized
                for value in apps
                if (normalized := normalize_app_label(value))
            }
        )
    )
    if not normalized_apps:
        raise ValueError("consumed row app labels normalize to an empty set")
    return {
        "source_id": source_id,
        "partition": partition,
        "instruction_app_group_sha256": instruction_app_group_sha256(
            _first_instruction_text(row),
            normalized_app_labels=normalized_apps,
        ),
    }


def _pool_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    strata: Counter[str] = Counter()
    decisions: Counter[int] = Counter()
    actions: Counter[str] = Counter()
    apps: Counter[str] = Counter()
    for record in records:
        strata[str(record["candidate_capacity_stratum"])] += 1
        decisions[int(record["decision_count"])] += 1
        actions.update(record["action_type_counts"])
        apps.update(set(record["normalized_app_labels"]))
    return {
        "trajectory_count": len(records),
        "decision_count": sum(int(record["decision_count"]) for record in records),
        "candidate_capacity_stratum_counts": dict(sorted(strata.items())),
        "decision_count_histogram": {
            str(key): value for key, value in sorted(decisions.items())
        },
        "distinct_app_label_count": len(apps),
        "app_trajectory_counts": dict(sorted(apps.items())),
        "action_type_counts": dict(sorted(actions.items())),
        "distinct_instruction_app_group_count": len(
            {record["instruction_app_group_sha256"] for record in records}
        ),
        "consumed_role_marker_counts": (
            {UNCONSUMED_MARKER: len(records)} if records else {}
        ),
    }


RowIteratorFactory = Callable[
    [SourceFileSpec], Iterable[tuple[int, Mapping[str, Any]]]
]


def build_full_pool_census(
    *,
    base_config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    row_iterator_factory: RowIteratorFactory,
    base_config_sha256: str,
    source_manifest_sha256: str,
    consumed_ledger: ConsumedLedgerBinding,
    format_safety_maximum_decisions: int | None = None,
) -> dict[str, Any]:
    """Stream every pinned shard and emit only never-consumed eligible rows."""
    if not callable(row_iterator_factory):
        raise TypeError("row_iterator_factory must be callable")
    if not isinstance(consumed_ledger, ConsumedLedgerBinding):
        raise TypeError("the canonical consumed ledger binding is required")
    base_hash = _sha256(base_config_sha256, label="base config sha256")
    source_hash = _sha256(
        source_manifest_sha256, label="source manifest sha256"
    )
    specs = validate_complete_source_manifest(source_manifest, base_config=base_config)
    inspection_config = build_full_pool_inspection_config(
        base_config,
        transport_files=tuple(spec.transport_file for spec in specs),
        format_safety_maximum_decisions=format_safety_maximum_decisions,
    )
    candidates: list[dict[str, Any]] = []
    parser_exclusions: Counter[str] = Counter()
    consumed_candidate_exclusions: Counter[str] = Counter()
    consumed_parser_exclusions: Counter[str] = Counter()
    consumed_parser_reason_exclusions: Counter[str] = Counter()
    consumed_seen: Counter[str] = Counter()
    consumed_group_records: list[dict[str, str]] = []
    row_counts: dict[str, int] = {}
    seen_source_ids: dict[str, tuple[str, int]] = {}
    role_by_source = consumed_ledger.role_by_source

    for spec in specs:
        expected_row_index = 0
        for row_index, row in row_iterator_factory(spec):
            if type(row_index) is not int or row_index != expected_row_index:
                raise ValueError(
                    f"row indices for {spec.transport_file} must be contiguous from zero"
                )
            if not isinstance(row, Mapping):
                raise TypeError("decoded parquet row must be a mapping")
            result = inspect_candidate(
                row,
                transport_file=spec.transport_file,
                transport_row_index=row_index,
                config=inspection_config,
            )
            consumed_partition: str | None = None
            if result.source_id is not None:
                previous = seen_source_ids.get(result.source_id)
                if previous is not None:
                    raise ValueError(
                        "duplicate source_id across complete source inventory: "
                        f"{result.source_id} at {previous} and "
                        f"{(spec.transport_file, row_index)}"
                    )
                seen_source_ids[result.source_id] = (
                    spec.transport_file,
                    row_index,
                )
                consumed_partition = role_by_source.get(result.source_id)
                if consumed_partition is not None:
                    consumed_seen[consumed_partition] += 1
                    consumed_group_records.append(
                        _consumed_group_record(
                            row=row,
                            source_id=result.source_id,
                            partition=consumed_partition,
                        )
                    )

            if result.candidate is None:
                if result.exclusion_reason is None:
                    raise RuntimeError("excluded row is missing its reason")
                reason = result.exclusion_reason.value
                if result.exclusion_reason == ExclusionReason.DECISION_COUNT_ABOVE_MAXIMUM:
                    if format_safety_maximum_decisions is None:
                        raise RuntimeError(
                            "unbounded compatibility adapter emitted an upper-length exclusion"
                        )
                    reason = FORMAT_SAFETY_EXCLUSION
                parser_exclusions[reason] += 1
                if consumed_partition is not None:
                    consumed_parser_exclusions[consumed_partition] += 1
                    consumed_parser_reason_exclusions[
                        f"{consumed_partition}:{reason}"
                    ] += 1
            elif consumed_partition is not None:
                consumed_candidate_exclusions[consumed_partition] += 1
            else:
                candidates.append(_candidate_record(result.candidate, row=row))
            expected_row_index += 1
        row_counts[spec.transport_file] = expected_row_index

    missing_consumed = sorted(set(role_by_source) - set(seen_source_ids))
    if missing_consumed:
        digest = sha256_bytes(canonical_json_bytes(missing_consumed))
        raise ValueError(
            "consumed ledger identities are absent from the complete source scan; "
            f"count={len(missing_consumed)}, sha256={digest}"
        )
    if dict(sorted(consumed_seen.items())) != EXPECTED_PARTITION_COUNTS:
        raise RuntimeError("consumed ledger observed partition counts drifted")
    consumed_group_records.sort(key=lambda record: record["source_id"])
    if len(consumed_group_records) != EXPECTED_UNION_COUNT:
        raise RuntimeError("consumed group firewall inventory count drifted")

    candidates.sort(
        key=lambda record: (
            record["selection_sha256"],
            record["source_id"],
            record["transport_file"],
            record["transport_row_index"],
        )
    )
    total_rows = sum(row_counts.values())
    if (
        len(candidates)
        + sum(parser_exclusions.values())
        + sum(consumed_candidate_exclusions.values())
        != total_rows
    ):
        raise RuntimeError("full-pool source row accounting is incomplete")
    if set(role_by_source) & {record["source_id"] for record in candidates}:
        raise RuntimeError("a consumed identity escaped into the candidate pool")
    inventory_hash = sha256_bytes(canonical_json_bytes(candidates))
    source = _mapping(source_manifest["source"], label="source manifest source")
    source_inventory = _mapping(
        source_manifest["inventory"], label="source manifest inventory"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "POLICY_BLIND_UNCONSUMED_FULL_POOL_CENSUS_COMPLETED",
        "source": {
            "repo_id": source["repo_id"],
            "repo_type": source["repo_type"],
            "revision": source["revision"],
            "target_directory": source["target_directory"],
            "base_config_sha256": base_hash,
            "source_manifest_sha256": source_hash,
            "source_inventory_files_sha256": source_inventory["files_sha256"],
            "file_count": len(specs),
            "total_size_bytes": sum(spec.size_bytes for spec in specs),
            "total_rows": total_rows,
            "row_counts_by_file": row_counts,
            "global_unique_source_id_count": len(seen_source_ids),
            "parser_exclusion_counts": dict(sorted(parser_exclusions.items())),
            "consumed_candidate_exclusion_counts_by_partition": dict(
                sorted(consumed_candidate_exclusions.items())
            ),
            "consumed_parser_exclusion_counts_by_partition": dict(
                sorted(consumed_parser_exclusions.items())
            ),
            "consumed_parser_exclusion_counts_by_partition_and_reason": dict(
                sorted(consumed_parser_reason_exclusions.items())
            ),
            "excluded_row_count": (
                sum(parser_exclusions.values())
                + sum(consumed_candidate_exclusions.values())
            ),
        },
        "eligibility": {
            "minimum_decision_count": MINIMUM_DECISION_COUNT,
            "scientific_maximum_decision_count": None,
            "format_safety_maximum_decision_count": (
                format_safety_maximum_decisions
            ),
            "old_parser_maximum_is_compatibility_only": (
                COMPATIBILITY_MAXIMUM_DECISION_COUNT
                if format_safety_maximum_decisions is None
                else format_safety_maximum_decisions
            ),
            "candidate_capacity_strata": [
                {"name": "decisions_6_9", "minimum": 6, "maximum": 9},
                {"name": "decisions_10_17", "minimum": 10, "maximum": 17},
                {"name": "decisions_18_plus", "minimum": 18, "maximum": None},
            ],
        },
        "consumed_ledger": {
            "status": "BOUND_VALIDATED_AND_APPLIED",
            "protocol_id": CONSUMED_LEDGER_PROTOCOL_ID,
            "manifest_sha256": consumed_ledger.manifest_sha256,
            "assignment_inventory_sha256": (
                consumed_ledger.assignment_inventory_sha256
            ),
            "consumed_union_source_set_sha256": (
                consumed_ledger.consumed_union_source_set_sha256
            ),
            "expected_partition_counts": EXPECTED_PARTITION_COUNTS,
            "observed_partition_counts": dict(sorted(consumed_seen.items())),
            "eligible_candidate_exclusion_counts_by_partition": dict(
                sorted(consumed_candidate_exclusions.items())
            ),
            "parser_exclusion_counts_by_partition": dict(
                sorted(consumed_parser_exclusions.items())
            ),
            "consumed_identity_count": len(role_by_source),
        },
        "consumed_group_audit": {
            "scope": "HISTORICALLY_CONSUMED_IDENTITIES_ONLY",
            "contains_raw_instruction": False,
            "record_count": len(consumed_group_records),
            "record_inventory_sha256": sha256_bytes(
                canonical_json_bytes(consumed_group_records)
            ),
            "records": consumed_group_records,
        },
        "selection": {
            "salt": SELECTION_SALT,
            "order": [
                "selection_sha256",
                "source_id",
                "transport_file",
                "transport_row_index",
            ],
            "candidate_count": len(candidates),
            "candidate_inventory_sha256": inventory_hash,
            "role_assignment_count": 0,
            "query_state_selection_count": 0,
        },
        "pool": {
            "scope": "UNCONSUMED_ELIGIBLE_TRAJECTORIES_ONLY",
            "summary": _pool_summary(candidates),
            "trajectories": candidates,
        },
        "operation_counts": {
            "ocr_access_count": 0,
            "policy_model_load_count": 0,
            "policy_forward_count": 0,
            "restoration_output_access_count": 0,
            "restoration_forward_count": 0,
            "gate_output_access_count": 0,
            "gate_forward_count": 0,
            "train_tune_evaluation_assignment_count": 0,
            "query_state_selection_count": 0,
        },
    }


__all__ = [
    "COMPATIBILITY_MAXIMUM_DECISION_COUNT",
    "CONSUMED_LEDGER_PROTOCOL_ID",
    "ConsumedLedgerBinding",
    "FORMAT_SAFETY_EXCLUSION",
    "MINIMUM_DECISION_COUNT",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "SELECTION_SALT",
    "TOTAL_GUIODYSSEY_TRAIN_SHARDS",
    "UNCONSUMED_MARKER",
    "build_full_pool_census",
    "build_full_pool_inspection_config",
    "candidate_capacity_stratum",
    "parse_consumed_ledger",
    "supported_candidate_counts",
    "validate_complete_source_manifest",
]
