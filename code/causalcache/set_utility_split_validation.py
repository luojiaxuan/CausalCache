"""Fail-closed group-aware split validation for set-utility development."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any, Literal


SCHEMA_VERSION = "0.1.0"
PROTOCOL_ID = "causalcache_set_utility_group_aware_split_audit"
AUDIT_STATUS = "VALID_GROUP_AWARE_SPLIT"
LEGACY_TRAIN_ROLE = "legacy_train_only"
NEW_DEVELOPMENT_ROLES = (
    "train",
    "tune",
    "evaluation",
    "phase2_calibration",
    "phase2_evaluation",
)
SPLIT_ROLES = (*NEW_DEVELOPMENT_ROLES, LEGACY_TRAIN_ROLE)
EFFECTIVE_GROUP_PARTITIONS = NEW_DEVELOPMENT_ROLES
PASSED_CHECKS = (
    "one_assignment_per_trajectory",
    "one_trajectory_per_source",
    "trajectory_role_disjoint",
    "group_partition_disjoint",
    "legacy_sources_bound_to_legacy_train_only",
    "forbidden_sources_absent",
)

SplitRole = Literal[
    "train",
    "tune",
    "evaluation",
    "phase2_calibration",
    "phase2_evaluation",
    "legacy_train_only",
]

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _identity(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be non-empty normalized text")
    return value


def _group_sha256(value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(
            "instruction_app_group_sha256 must be one lowercase SHA256 digest"
        )
    return value


def _role(value: Any) -> SplitRole:
    if value not in SPLIT_ROLES or not isinstance(value, str):
        raise ValueError(f"role must be one of {SPLIT_ROLES!r}")
    return value  # type: ignore[return-value]


def _explicit_source_set(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Set
    ):
        raise TypeError(f"{label} must be an explicit set of source IDs")
    source_ids = tuple(sorted(_identity(item, f"{label} item") for item in value))
    if len(source_ids) != len(value):
        raise ValueError(f"{label} contains duplicate normalized source IDs")
    return source_ids


def _effective_group_partition(role: SplitRole) -> str:
    if role == LEGACY_TRAIN_ROLE:
        return "train"
    return role


@dataclass(frozen=True)
class SetUtilitySplitAssignment:
    """One trajectory-level assignment supplied by an external split procedure."""

    trajectory_id: str
    source_id: str
    instruction_app_group_sha256: str
    role: SplitRole

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trajectory_id",
            _identity(self.trajectory_id, "trajectory_id"),
        )
        object.__setattr__(self, "source_id", _identity(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "instruction_app_group_sha256",
            _group_sha256(self.instruction_app_group_sha256),
        )
        object.__setattr__(self, "role", _role(self.role))

    def to_payload(self) -> dict[str, str]:
        return {
            "trajectory_id": self.trajectory_id,
            "source_id": self.source_id,
            "instruction_app_group_sha256": self.instruction_app_group_sha256,
            "role": self.role,
        }


@dataclass(frozen=True)
class GroupAwareSplitAudit:
    """Deterministic summary of a successfully validated external split."""

    assignment_count: int
    trajectory_count: int
    source_count: int
    instruction_app_group_count: int
    role_trajectory_counts: tuple[tuple[str, int], ...]
    effective_partition_group_counts: tuple[tuple[str, int], ...]
    assignment_inventory_sha256: str
    legacy_source_inventory_sha256: str
    forbidden_source_inventory_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "audit_status": AUDIT_STATUS,
            "assignment_count": self.assignment_count,
            "trajectory_count": self.trajectory_count,
            "source_count": self.source_count,
            "instruction_app_group_count": self.instruction_app_group_count,
            "role_trajectory_counts": dict(self.role_trajectory_counts),
            "effective_partition_group_counts": dict(
                self.effective_partition_group_counts
            ),
            "overlap_counts": {
                "trajectory_role": 0,
                "source_identity": 0,
                "instruction_app_group_partition": 0,
                "legacy_role": 0,
                "forbidden_source": 0,
            },
            "passed_checks": list(PASSED_CHECKS),
            "assignment_inventory_sha256": self.assignment_inventory_sha256,
            "legacy_source_inventory_sha256": self.legacy_source_inventory_sha256,
            "forbidden_source_inventory_sha256": self.forbidden_source_inventory_sha256,
        }

    @property
    def summary(self) -> dict[str, Any]:
        return self.to_payload()

    @property
    def summary_sha256(self) -> str:
        return _sha256(self.to_payload())


@dataclass(frozen=True)
class ValidatedGroupAwareSplit:
    assignments: tuple[SetUtilitySplitAssignment, ...]
    audit: GroupAwareSplitAudit


def validate_group_aware_split_assignments(
    assignments: Sequence[SetUtilitySplitAssignment],
    *,
    legacy_train_only_source_ids: Set[str],
    forbidden_source_ids: Set[str],
) -> ValidatedGroupAwareSplit:
    """Validate identities and overlaps without selecting or resizing any role."""
    if isinstance(assignments, (str, bytes, bytearray, Mapping)) or not isinstance(
        assignments, Sequence
    ):
        raise TypeError("assignments must be a non-empty sequence")
    if not assignments:
        raise ValueError("assignments cannot be empty")
    if any(not isinstance(item, SetUtilitySplitAssignment) for item in assignments):
        raise TypeError("every assignment must be SetUtilitySplitAssignment")

    legacy_sources = _explicit_source_set(
        legacy_train_only_source_ids,
        "legacy_train_only_source_ids",
    )
    forbidden_sources = _explicit_source_set(
        forbidden_source_ids,
        "forbidden_source_ids",
    )
    legacy_set = frozenset(legacy_sources)
    forbidden_set = frozenset(forbidden_sources)
    if legacy_set & forbidden_set:
        raise ValueError("legacy and forbidden source sets must be disjoint")

    canonical = tuple(
        sorted(
            assignments,
            key=lambda item: (
                item.trajectory_id,
                item.source_id,
                item.instruction_app_group_sha256,
                item.role,
            ),
        )
    )
    payloads = tuple(item.to_payload() for item in canonical)
    if len(set(_canonical_json_bytes(item) for item in payloads)) != len(payloads):
        raise ValueError("split roster contains a duplicate trajectory assignment")

    forbidden_hits = tuple(
        sorted({item.source_id for item in canonical if item.source_id in forbidden_set})
    )
    if forbidden_hits:
        raise ValueError(
            "split roster contains an explicitly forbidden source identity; "
            f"hit_count={len(forbidden_hits)}, hit_sha256={_sha256(forbidden_hits)}"
        )

    mislabeled_legacy = tuple(
        item
        for item in canonical
        if (item.source_id in legacy_set) != (item.role == LEGACY_TRAIN_ROLE)
    )
    if mislabeled_legacy:
        raise ValueError(
            "legacy formal-train sources must appear only as legacy_train_only, "
            "and that role may contain only declared legacy sources"
        )

    by_trajectory: dict[str, list[SetUtilitySplitAssignment]] = defaultdict(list)
    by_source: dict[str, list[SetUtilitySplitAssignment]] = defaultdict(list)
    by_group: dict[str, list[SetUtilitySplitAssignment]] = defaultdict(list)
    for item in canonical:
        by_trajectory[item.trajectory_id].append(item)
        by_source[item.source_id].append(item)
        by_group[item.instruction_app_group_sha256].append(item)

    for trajectory_id, records in by_trajectory.items():
        roles = {item.role for item in records}
        if len(roles) > 1:
            raise ValueError(
                "one trajectory_id cannot cross roles; "
                f"trajectory_sha256={_sha256(trajectory_id)}, roles={sorted(roles)!r}"
            )
        if len(records) != 1:
            raise ValueError("each trajectory_id must have exactly one assignment")

    for source_id, records in by_source.items():
        trajectories = {item.trajectory_id for item in records}
        if len(trajectories) > 1 or len(records) != 1:
            raise ValueError(
                "one source_id must identify exactly one trajectory assignment; "
                f"source_sha256={_sha256(source_id)}"
            )

    partition_by_group: dict[str, str] = {}
    for group_sha256, records in by_group.items():
        partitions = {
            _effective_group_partition(item.role) for item in records
        }
        if len(partitions) != 1:
            raise ValueError(
                "one instruction-app group cannot cross development partitions; "
                f"group_sha256={group_sha256}, partitions={sorted(partitions)!r}"
            )
        partition_by_group[group_sha256] = next(iter(partitions))

    role_counts = tuple(
        (role, sum(item.role == role for item in canonical)) for role in SPLIT_ROLES
    )
    partition_group_counts = tuple(
        (
            partition,
            sum(value == partition for value in partition_by_group.values()),
        )
        for partition in EFFECTIVE_GROUP_PARTITIONS
    )
    audit = GroupAwareSplitAudit(
        assignment_count=len(canonical),
        trajectory_count=len(by_trajectory),
        source_count=len(by_source),
        instruction_app_group_count=len(by_group),
        role_trajectory_counts=role_counts,
        effective_partition_group_counts=partition_group_counts,
        assignment_inventory_sha256=_sha256(payloads),
        legacy_source_inventory_sha256=_sha256(legacy_sources),
        forbidden_source_inventory_sha256=_sha256(forbidden_sources),
    )
    return ValidatedGroupAwareSplit(assignments=canonical, audit=audit)


__all__ = [
    "AUDIT_STATUS",
    "EFFECTIVE_GROUP_PARTITIONS",
    "GroupAwareSplitAudit",
    "LEGACY_TRAIN_ROLE",
    "NEW_DEVELOPMENT_ROLES",
    "PASSED_CHECKS",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "SPLIT_ROLES",
    "SetUtilitySplitAssignment",
    "SplitRole",
    "ValidatedGroupAwareSplit",
    "validate_group_aware_split_assignments",
]
