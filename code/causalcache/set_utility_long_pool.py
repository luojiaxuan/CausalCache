"""Policy-blind discovery for previously unused long GUIOdyssey trajectories."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.data.guiodyssey_independent import Candidate


SCHEMA_VERSION = "0.1.0"
PROTOCOL_ID = "causalcache_set_utility_long_pool_discovery_v1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_long_pool_discovery_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "5282e505b31086141b10b8a3fdccd619fa748245e0a7c5835304d581af0c39c3"
)
LENGTH_STRATA = (
    ("long_13_16", 13, 16),
    ("long_17_24", 17, 24),
    ("long_25_64", 25, 64),
)


def validate_discovery_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION or config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("set-utility long-pool config identity drifted")
    inputs = config.get("immutable_inputs")
    eligibility = config.get("eligibility_override")
    selection = config.get("selection")
    limits = config.get("operation_limits")
    if not all(isinstance(value, Mapping) for value in (inputs, eligibility, selection, limits)):
        raise ValueError("set-utility long-pool config sections are malformed")
    for record_name in (
        "base_config",
        "source_file_manifest",
        "prior_source_pool_artifact",
        "prior_eligible_pool_manifest",
    ):
        record = inputs.get(record_name)
        if not isinstance(record, Mapping) or re.fullmatch(
            r"[0-9a-f]{64}", str(record.get("sha256", ""))
        ) is None:
            raise ValueError(f"immutable input {record_name} is malformed")
    if (
        eligibility.get("minimum_decisions_per_trajectory") != 13
        or eligibility.get("maximum_decisions_per_trajectory") != 64
        or eligibility.get("previously_used_decision_count_range") != [4, 12]
        or eligibility.get("range_disjoint_from_all_previous_output_roles") is not True
    ):
        raise ValueError("long-pool eligibility range drifted")
    expected_strata = [
        {"name": name, "minimum": minimum, "maximum": maximum}
        for name, minimum, maximum in LENGTH_STRATA
    ]
    if (
        selection.get("length_strata") != expected_strata
        or selection.get("order")
        != ["sha256_hex", "source_id", "transport_file", "transport_row_index"]
        or not isinstance(selection.get("salt"), str)
        or not selection["salt"]
        or selection.get("discovery_result_does_not_authorize_labels") is not True
        or selection.get("role_assignment_allowed") is not False
        or selection.get("query_state_selection_allowed") is not False
    ):
        raise ValueError("long-pool selection contract drifted")
    if any(value != 0 for value in limits.values()):
        raise ValueError("long-pool discovery must prohibit every model/GPU operation")


def validate_discovery_source_only(
    *, repository_root: str | Path, contract_path: str | Path = CANONICAL_CONFIG_PATH
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    supplied = Path(contract_path)
    supplied = supplied if supplied.is_absolute() else root / supplied
    canonical = root / CANONICAL_CONFIG_PATH
    if supplied.resolve() != canonical.resolve() or supplied.is_symlink() or canonical.is_symlink():
        raise ValueError("long-pool discovery contract path is not canonical")
    payload = canonical.read_bytes()
    observed = sha256_bytes(payload)
    if observed != FROZEN_CONFIG_SHA256:
        raise ValueError("long-pool discovery config SHA256 drifted")
    try:
        config = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("long-pool discovery config is not strict UTF-8 JSON") from error
    if not isinstance(config, Mapping):
        raise ValueError("long-pool discovery config must be an object")
    validate_discovery_config(config)
    return {
        "status": "VALID_SOURCE_ONLY_SET_UTILITY_LONG_POOL_DISCOVERY_V1",
        "protocol_id": PROTOCOL_ID,
        "config_sha256": observed,
        "config_read_count": 1,
        "data_read_count": 0,
        **dict(config["operation_limits"]),
    }


@dataclass(frozen=True)
class LongPoolCandidate:
    source_id: str
    transport_file: str
    transport_row_index: int
    decision_count: int
    normalized_app_labels: tuple[str, ...]
    action_type_counts: tuple[tuple[str, int], ...]
    selection_sha256: str
    length_stratum: str

    def manifest_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "transport_file": self.transport_file,
            "transport_row_index": self.transport_row_index,
            "decision_count": self.decision_count,
            "normalized_app_labels": list(self.normalized_app_labels),
            "action_type_counts": dict(self.action_type_counts),
            "selection_sha256": self.selection_sha256,
            "length_stratum": self.length_stratum,
        }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def length_stratum(decision_count: int) -> str:
    if type(decision_count) is not int:
        raise TypeError("decision_count must be an integer")
    for name, minimum, maximum in LENGTH_STRATA:
        if minimum <= decision_count <= maximum:
            return name
    raise ValueError("decision_count is outside the frozen long-pool range")


def long_selection_sha256(source_id: str, *, salt: str) -> str:
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be non-empty text")
    if not isinstance(salt, str) or not salt:
        raise ValueError("selection salt must be non-empty text")
    return sha256_bytes(f"{salt}\0{source_id}".encode("utf-8"))


def project_candidate(candidate: Candidate, *, salt: str) -> LongPoolCandidate:
    return LongPoolCandidate(
        source_id=candidate.source_id,
        transport_file=candidate.transport_file,
        transport_row_index=candidate.transport_row_index,
        decision_count=candidate.decision_count,
        normalized_app_labels=candidate.normalized_app_labels,
        action_type_counts=candidate.action_type_counts,
        selection_sha256=long_selection_sha256(candidate.source_id, salt=salt),
        length_stratum=length_stratum(candidate.decision_count),
    )


def _pool_summary(candidates: Sequence[LongPoolCandidate]) -> dict[str, Any]:
    apps: set[str] = set()
    action_counts: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    decision_counts: Counter[int] = Counter()
    for candidate in candidates:
        apps.update(candidate.normalized_app_labels)
        action_counts.update(dict(candidate.action_type_counts))
        strata[candidate.length_stratum] += 1
        decision_counts[candidate.decision_count] += 1
    return {
        "trajectory_count": len(candidates),
        "decision_count": sum(candidate.decision_count for candidate in candidates),
        "distinct_app_label_count": len(apps),
        "length_stratum_counts": dict(sorted(strata.items())),
        "decision_count_histogram": {
            str(key): value for key, value in sorted(decision_counts.items())
        },
        "action_type_counts": dict(sorted(action_counts.items())),
        "source_ids_sha256": sha256_bytes(
            canonical_json_bytes([candidate.source_id for candidate in candidates])
        ),
    }


def build_discovery_manifest(
    *,
    candidates: Sequence[LongPoolCandidate],
    exclusion_counts: Mapping[str, int],
    source_row_count: int,
    config_sha256: str,
    base_config_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    source_ids = [candidate.source_id for candidate in candidates]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("long-pool source ids must be unique")
    for candidate in candidates:
        if candidate.length_stratum != length_stratum(candidate.decision_count):
            raise ValueError("long-pool candidate stratum drifted")
    if len(candidates) + sum(int(value) for value in exclusion_counts.values()) != source_row_count:
        raise ValueError("long-pool source row accounting is incomplete")
    candidate_records = sorted(
        (candidate.manifest_record() for candidate in candidates),
        key=lambda record: (
            record["selection_sha256"],
            record["source_id"],
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "POLICY_BLIND_LONG_POOL_DISCOVERY_COMPLETED",
        "source": {
            "row_count": source_row_count,
            "config_sha256": config_sha256,
            "base_config_sha256": base_config_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
        },
        "selection": {
            "candidate_count": len(candidates),
            "candidate_inventory_sha256": sha256_bytes(
                canonical_json_bytes(candidate_records)
            ),
            "length_strata": [
                {"name": name, "minimum": minimum, "maximum": maximum}
                for name, minimum, maximum in LENGTH_STRATA
            ],
            "policy_output_access_count": 0,
            "restoration_output_access_count": 0,
            "gate_output_access_count": 0,
            "role_assignment_count": 0,
            "query_state_selection_count": 0,
        },
        "pool": {
            "summary": _pool_summary(candidates),
            "trajectories": candidate_records,
        },
    }


__all__ = [
    "LENGTH_STRATA",
    "LongPoolCandidate",
    "PROTOCOL_ID",
    "build_discovery_manifest",
    "CANONICAL_CONFIG_PATH",
    "canonical_json_bytes",
    "length_stratum",
    "long_selection_sha256",
    "project_candidate",
    "sha256_bytes",
    "validate_discovery_source_only",
    "validate_discovery_config",
]
