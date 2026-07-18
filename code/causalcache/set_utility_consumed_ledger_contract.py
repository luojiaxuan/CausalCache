"""Shared narrow contract between the consumed ledger and full-pool census."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "0.1.0"
PROTOCOL_ID = "causalcache_set_utility_consumed_ledger_v1"
ASSIGNMENT_KEYS = frozenset({"source_id", "role"})
ASSIGNMENT_ROLES = frozenset({"legacy_train_only", "forbidden_consumed"})

_SOURCE_ID = re.compile(r"[A-Za-z0-9._-]+")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_assignments(value: Any) -> tuple[dict[str, str], ...]:
    """Validate and sort the only assignment schema accepted by both sides."""
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError("consumed ledger assignments must be an explicit sequence")
    parsed: list[dict[str, str]] = []
    for record in value:
        if not isinstance(record, Mapping) or set(record) != ASSIGNMENT_KEYS:
            raise ValueError(
                "consumed ledger assignment must contain exactly source_id and role"
            )
        source_id = record.get("source_id")
        role = record.get("role")
        if not isinstance(source_id, str) or _SOURCE_ID.fullmatch(source_id) is None:
            raise ValueError("consumed ledger assignment source_id is malformed")
        if role not in ASSIGNMENT_ROLES or not isinstance(role, str):
            raise ValueError("consumed ledger assignment role is unknown")
        parsed.append({"source_id": source_id, "role": role})
    canonical = tuple(
        sorted(parsed, key=lambda record: (record["source_id"], record["role"]))
    )
    if len({record["source_id"] for record in canonical}) != len(canonical):
        raise ValueError(
            "one consumed source identity cannot have multiple assignments"
        )
    return canonical


def assignment_inventory_sha256(value: Any) -> str:
    payload = canonical_json_bytes(canonical_assignments(value))
    return hashlib.sha256(payload).hexdigest()
