"""Canonical consumed-identity ledger for the set-utility data firewall."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.set_utility_consumed_ledger_contract import (
    PROTOCOL_ID,
    SCHEMA_VERSION,
    assignment_inventory_sha256,
    canonical_assignments,
    canonical_json_bytes,
)

STATUS = "VALID_SET_UTILITY_CONSUMED_IDENTITY_LEDGER_V1"
CANONICAL_LEDGER_MANIFEST_PATH = (
    "data/manifests/set_utility_consumed_identity_ledger_v1.json"
)

LONG_POOL_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_long_pool_discovery_v1.json"
)
LONG_POOL_CONFIG_SHA256 = (
    "d10484f53f579bf26012ba6fdd3e7e701c90d7e760db6667ebf7e74c75a98957"
)
PREREGISTRATION_CONFIG_PATH = "code/configs/causalcache_gate_v1_preregistration.json"
PREREGISTRATION_CONFIG_SHA256 = (
    "37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b"
)
FRESH16_CONFIG_PATH = "code/configs/causalcache_gate_v1_fresh16_evaluation_v1.json"
FRESH16_CONFIG_SHA256 = (
    "c98647aecf6b07e0ccccf1601b5e21289b595b7a4a45cc7ab9a542b1bd7dff2e"
)

CANONICAL_INPUT_PATHS = frozenset(
    {
        LONG_POOL_CONFIG_PATH,
        PREREGISTRATION_CONFIG_PATH,
        FRESH16_CONFIG_PATH,
        "data/manifests/independent_reference_gate_v1_artifact.json",
        "data/manifests/restoration_v2_2_label_expansion_selection.json",
        "data/manifests/restoration_v2_selection.json",
    }
)

LEGACY_SELECTION_ROLES = frozenset(
    {
        "v1_reference_contract_audit_only",
        "v2_confirm_primary",
        "v2_development",
        "v2_label_train",
    }
)
EXPANSION_SELECTION_ROLES = frozenset(
    {"gate_development_expansion", "gate_train_expansion"}
)
REFERENCE_ARTIFACT_SPLITS = frozenset({"reference_gate", "oracle_pilot"})

ROLE_SPECS = (
    (
        "legacy_label_train",
        "legacy_train_only",
        "legacy",
        "v2_label_train",
        "legacy_train",
        10,
    ),
    (
        "expansion_train",
        "legacy_train_only",
        "expansion",
        "gate_train_expansion",
        "fresh_train_expansion",
        48,
    ),
    (
        "reference8",
        "forbidden_consumed",
        "legacy",
        "v1_reference_contract_audit_only",
        None,
        8,
    ),
    (
        "old_development5",
        "forbidden_consumed",
        "legacy",
        "v2_development",
        "legacy_development",
        5,
    ),
    (
        "fresh16",
        "forbidden_consumed",
        "expansion",
        "gate_development_expansion",
        "fresh_development",
        16,
    ),
    (
        "confirm20",
        "forbidden_consumed",
        "legacy",
        "v2_confirm_primary",
        "sealed_confirm",
        20,
    ),
)

EXPECTED_PARTITION_COUNTS = {
    "legacy_train_only": 58,
    "forbidden_consumed": 49,
}
EXPECTED_UNION_COUNT = 107

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCE_ID = re.compile(r"[A-Za-z0-9._-]+")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_ids_sha256(source_ids: Sequence[str]) -> str:
    return sha256_bytes(canonical_json_bytes(list(source_ids)))


def source_set_sha256(source_ids: Sequence[str]) -> str:
    return source_ids_sha256(tuple(sorted(source_ids)))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be an array")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA256 digest")
    return value


def _source_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SOURCE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} is not one normalized source ID")
    return value


def _strict_json(payload: bytes, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains invalid JSON constant {value!r}")

    try:
        decoded = payload.decode("utf-8")
        value = json.loads(decoded, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    return _mapping(value, label)


@dataclass(frozen=True)
class InputBinding:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        path = Path(self.path)
        if (
            not isinstance(self.path, str)
            or not self.path
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError("input path must be a normalized repository-relative path")
        _sha256(self.sha256, f"input {self.path} SHA256")

    def to_payload(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ConsumedIdentityLedger:
    """Validated identities plus a deterministic Git-lightweight payload."""

    legacy_train_only_source_ids: frozenset[str]
    forbidden_consumed_source_ids: frozenset[str]
    payload: Mapping[str, Any]

    @property
    def summary_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload))


def ledger_manifest_bytes(ledger: ConsumedIdentityLedger) -> bytes:
    if not isinstance(ledger, ConsumedIdentityLedger):
        raise TypeError("ledger must be one validated ConsumedIdentityLedger")
    return canonical_json_bytes(ledger.payload) + b"\n"


def write_ledger_manifest_exclusive(
    ledger: ConsumedIdentityLedger,
    output_path: str | Path,
) -> str:
    """Write one byte-canonical ledger without replacing an existing artifact."""
    path = Path(output_path)
    payload = ledger_manifest_bytes(ledger)
    try:
        with path.open("xb") as handle:
            try:
                handle.write(payload)
                handle.flush()
            except BaseException:
                handle.close()
                path.unlink(missing_ok=True)
                raise
    except FileExistsError:
        raise
    return sha256_bytes(payload)


def _binding_from_record(value: Any, label: str) -> InputBinding:
    record = _mapping(value, label)
    if not {"path", "sha256"} <= set(record):
        raise ValueError(f"{label} must bind path and SHA256")
    path = record.get("path")
    if not isinstance(path, str):
        raise ValueError(f"{label} path must be text")
    return InputBinding(path=path, sha256=_sha256(record.get("sha256"), label))


def _load_canonical_input(
    repository_root: Path,
    binding: InputBinding,
) -> Mapping[str, Any]:
    root = repository_root.resolve()
    path = root / binding.path
    current = root
    for component in Path(binding.path).parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"canonical input path is symlinked: {binding.path}")
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"canonical input path escaped the repository: {binding.path}"
        ) from error
    if not path.is_file():
        raise ValueError(f"canonical input is not a regular file: {binding.path}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"canonical input is unreadable: {binding.path}") from error
    observed = sha256_bytes(payload)
    if observed != binding.sha256:
        raise ValueError(
            f"canonical input SHA256 drifted for {binding.path}: "
            f"expected {binding.sha256}, got {observed}"
        )
    return _strict_json(payload, binding.path)


def _top_level_config(
    repository_root: Path,
    path: str,
    expected_sha256: str,
) -> tuple[InputBinding, Mapping[str, Any]]:
    binding = InputBinding(path=path, sha256=expected_sha256)
    return binding, _load_canonical_input(repository_root, binding)


def _source_ids_from_trajectories(
    manifest: Mapping[str, Any],
    *,
    section_name: str,
    role: str,
) -> tuple[str, ...]:
    sections = _mapping(manifest.get(section_name), section_name)
    section = _mapping(sections.get(role), f"{section_name}.{role}")
    trajectories = _sequence(
        section.get("trajectories"), f"{section_name}.{role}.trajectories"
    )
    source_ids = tuple(
        _source_id(
            _mapping(record, f"{section_name}.{role} trajectory").get("source_id"),
            f"{section_name}.{role} source_id",
        )
        for record in trajectories
    )
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{section_name}.{role} contains duplicate source IDs")
    return source_ids


def _validate_exact_roles(
    manifest: Mapping[str, Any],
    *,
    section_name: str,
    expected: frozenset[str],
) -> None:
    observed = frozenset(_mapping(manifest.get(section_name), section_name))
    if observed != expected:
        raise ValueError(
            f"unknown or missing {section_name} role: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _roster_record(
    preregistration: Mapping[str, Any], roster_name: str
) -> Mapping[str, Any]:
    rosters = _mapping(preregistration.get("rosters"), "preregistration rosters")
    return _mapping(rosters.get(roster_name), f"preregistration roster {roster_name}")


def _validate_roster(
    preregistration: Mapping[str, Any],
    *,
    roster_name: str,
    source_ids: Sequence[str],
) -> None:
    roster = _roster_record(preregistration, roster_name)
    if roster.get("trajectory_count") != len(source_ids):
        raise ValueError(f"preregistration roster {roster_name} count drifted")
    expected_digest = _sha256(
        roster.get("source_ids_sha256"),
        f"preregistration roster {roster_name} source IDs",
    )
    if source_ids_sha256(source_ids) != expected_digest:
        raise ValueError(f"preregistration roster {roster_name} source IDs drifted")


def _role_payload(
    *,
    ledger_role: str,
    partition: str,
    source_manifest_role: str,
    source_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "ledger_role": ledger_role,
        "partition": partition,
        "source_manifest_role": source_manifest_role,
        "source_id_count": len(source_ids),
        "source_ids_sha256": source_ids_sha256(source_ids),
        "source_set_sha256": source_set_sha256(source_ids),
        "source_ids": list(source_ids),
    }


def derive_consumed_identity_ledger(
    *,
    preregistration: Mapping[str, Any],
    legacy_selection: Mapping[str, Any],
    expansion_selection: Mapping[str, Any],
    reference_artifact: Mapping[str, Any],
    inputs: Sequence[InputBinding],
) -> ConsumedIdentityLedger:
    """Derive and validate all consumed identities from frozen manifests."""
    _validate_exact_roles(
        legacy_selection,
        section_name="roles",
        expected=LEGACY_SELECTION_ROLES,
    )
    _validate_exact_roles(
        expansion_selection,
        section_name="splits",
        expected=EXPANSION_SELECTION_ROLES,
    )
    _validate_exact_roles(
        reference_artifact,
        section_name="splits",
        expected=REFERENCE_ARTIFACT_SPLITS,
    )

    source_by_origin_role = {
        ("legacy", role): _source_ids_from_trajectories(
            legacy_selection, section_name="roles", role=role
        )
        for role in LEGACY_SELECTION_ROLES
    }
    source_by_origin_role.update(
        {
            ("expansion", role): _source_ids_from_trajectories(
                expansion_selection, section_name="splits", role=role
            )
            for role in EXPANSION_SELECTION_ROLES
        }
    )

    reference_record = _mapping(
        _mapping(reference_artifact.get("splits"), "reference artifact splits").get(
            "reference_gate"
        ),
        "reference artifact reference_gate",
    )
    reference_ids = tuple(
        _source_id(value, "reference artifact source_id")
        for value in _sequence(
            reference_record.get("source_ids"), "reference artifact source_ids"
        )
    )
    selected_reference_ids = source_by_origin_role[
        ("legacy", "v1_reference_contract_audit_only")
    ]
    if (
        reference_record.get("trajectory_count") != 8
        or reference_ids != selected_reference_ids
    ):
        raise ValueError(
            "reference8 identity evidence disagrees across pinned manifests"
        )

    role_payloads: list[dict[str, Any]] = []
    role_sources: dict[str, tuple[str, ...]] = {}
    all_role_sets: dict[str, frozenset[str]] = {}
    for (
        ledger_role,
        partition,
        origin,
        source_manifest_role,
        roster_name,
        expected_count,
    ) in ROLE_SPECS:
        source_ids = source_by_origin_role[(origin, source_manifest_role)]
        if len(source_ids) != expected_count:
            raise ValueError(
                f"consumed role {ledger_role} count drifted: "
                f"expected {expected_count}, got {len(source_ids)}"
            )
        if roster_name is not None:
            _validate_roster(
                preregistration,
                roster_name=roster_name,
                source_ids=source_ids,
            )
        role_sources[ledger_role] = source_ids
        all_role_sets[ledger_role] = frozenset(source_ids)
        role_payloads.append(
            _role_payload(
                ledger_role=ledger_role,
                partition=partition,
                source_manifest_role=source_manifest_role,
                source_ids=source_ids,
            )
        )

    role_names = tuple(all_role_sets)
    intersections = {
        f"{left}__{right}": sorted(all_role_sets[left] & all_role_sets[right])
        for left_index, left in enumerate(role_names)
        for right in role_names[left_index + 1 :]
    }
    nonempty = {key: values for key, values in intersections.items() if values}
    if nonempty:
        raise ValueError(
            "consumed identity roles overlap: "
            "intersection_inventory_sha256="
            f"{sha256_bytes(canonical_json_bytes(nonempty))}"
        )

    legacy_train_ids = (
        role_sources["legacy_label_train"] + role_sources["expansion_train"]
    )
    forbidden_ids = (
        role_sources["reference8"]
        + role_sources["old_development5"]
        + role_sources["fresh16"]
        + role_sources["confirm20"]
    )
    _validate_roster(
        preregistration,
        roster_name="formal_train",
        source_ids=legacy_train_ids,
    )
    _validate_roster(
        preregistration,
        roster_name="combined_development",
        source_ids=(
            role_sources["old_development5"] + role_sources["fresh16"]
        ),
    )

    partition_sources = {
        "legacy_train_only": legacy_train_ids,
        "forbidden_consumed": forbidden_ids,
    }
    for partition, expected_count in EXPECTED_PARTITION_COUNTS.items():
        if len(partition_sources[partition]) != expected_count:
            raise ValueError(f"consumed partition {partition} count drifted")
    legacy_set = frozenset(legacy_train_ids)
    forbidden_set = frozenset(forbidden_ids)
    if legacy_set & forbidden_set:
        raise ValueError("legacy_train_only and forbidden_consumed must be disjoint")
    union = legacy_set | forbidden_set
    if len(union) != EXPECTED_UNION_COUNT:
        raise ValueError("consumed identity union count drifted")

    if isinstance(inputs, (str, bytes, bytearray, Mapping)) or not isinstance(
        inputs, Sequence
    ):
        raise ValueError("consumed ledger inputs must be an explicit sequence")
    if any(not isinstance(binding, InputBinding) for binding in inputs):
        raise ValueError("every consumed ledger input must be an InputBinding")
    input_payloads = [binding.to_payload() for binding in inputs]
    if len(input_payloads) != len({record["path"] for record in input_payloads}):
        raise ValueError("consumed ledger inputs contain duplicate paths")
    observed_input_paths = frozenset(record["path"] for record in input_payloads)
    if observed_input_paths != CANONICAL_INPUT_PATHS:
        raise ValueError(
            "consumed ledger input inventory drifted: "
            f"missing={sorted(CANONICAL_INPUT_PATHS - observed_input_paths)}, "
            f"extra={sorted(observed_input_paths - CANONICAL_INPUT_PATHS)}"
        )
    assignments = canonical_assignments(
        [
            {
                "source_id": source_id,
                "role": role["partition"],
            }
            for role in role_payloads
            for source_id in role["source_ids"]
        ]
    )
    assignment_digest = assignment_inventory_sha256(assignments)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": STATUS,
        "assignment_count": len(assignments),
        "assignment_inventory_sha256": assignment_digest,
        "assignments": list(assignments),
        "inputs": sorted(input_payloads, key=lambda record: record["path"]),
        "role_inventories": role_payloads,
        "partitions": {
            name: {
                "source_id_count": len(source_ids),
                "source_ids_sha256": source_ids_sha256(source_ids),
                "source_set_sha256": source_set_sha256(source_ids),
            }
            for name, source_ids in partition_sources.items()
        },
        "consumed_union": {
            "source_id_count": len(union),
            "source_set_sha256": source_set_sha256(tuple(union)),
            "assignment_inventory_sha256": assignment_digest,
        },
        "overlap_audit": {
            "role_pair_count": len(intersections),
            "nonempty_role_pair_count": 0,
            "legacy_forbidden_intersection_count": 0,
            "empty_intersection_inventory_sha256": sha256_bytes(
                canonical_json_bytes(intersections)
            ),
        },
    }
    return ConsumedIdentityLedger(
        legacy_train_only_source_ids=legacy_set,
        forbidden_consumed_source_ids=forbidden_set,
        payload=payload,
    )


def build_canonical_consumed_identity_ledger(
    *, repository_root: str | Path
) -> ConsumedIdentityLedger:
    """Read the canonical byte-pinned evidence and build the firewall ledger."""
    root = Path(repository_root).resolve()
    long_pool_binding, long_pool = _top_level_config(
        root, LONG_POOL_CONFIG_PATH, LONG_POOL_CONFIG_SHA256
    )
    prereg_binding, preregistration = _top_level_config(
        root, PREREGISTRATION_CONFIG_PATH, PREREGISTRATION_CONFIG_SHA256
    )
    fresh16_binding, fresh16 = _top_level_config(
        root, FRESH16_CONFIG_PATH, FRESH16_CONFIG_SHA256
    )

    immutable_inputs = _mapping(
        long_pool.get("immutable_inputs"), "long-pool immutable_inputs"
    )
    legacy_binding = _binding_from_record(
        immutable_inputs.get("prior_eligible_pool_manifest"),
        "long-pool prior eligible pool manifest",
    )
    reference_binding = _binding_from_record(
        immutable_inputs.get("prior_source_pool_artifact"),
        "long-pool prior source pool artifact",
    )
    prereg_legacy_binding = _binding_from_record(
        _mapping(preregistration.get("lineage"), "preregistration lineage").get(
            "legacy_selection"
        ),
        "preregistration legacy selection",
    )
    if prereg_legacy_binding != legacy_binding:
        raise ValueError("top-level configs disagree on the legacy selection binding")

    prerequisites = _sequence(
        _mapping(fresh16.get("source_freeze"), "fresh16 source_freeze").get(
            "git_prerequisites"
        ),
        "fresh16 git_prerequisites",
    )
    expansion_matches = tuple(
        _binding_from_record(record, "fresh16 expansion selection prerequisite")
        for record in prerequisites
        if isinstance(record, Mapping)
        and record.get("path")
        == "data/manifests/restoration_v2_2_label_expansion_selection.json"
    )
    if len(expansion_matches) != 1:
        raise ValueError("fresh16 config must bind exactly one expansion selection")
    expansion_binding = expansion_matches[0]

    legacy_selection = _load_canonical_input(root, legacy_binding)
    reference_artifact = _load_canonical_input(root, reference_binding)
    expansion_selection = _load_canonical_input(root, expansion_binding)
    expansion_parent_binding = _binding_from_record(
        _mapping(expansion_selection.get("inputs"), "expansion inputs").get(
            "parent_selection"
        ),
        "expansion parent selection",
    )
    if expansion_parent_binding != legacy_binding:
        raise ValueError("expansion manifest disagrees on the legacy selection binding")

    return derive_consumed_identity_ledger(
        preregistration=preregistration,
        legacy_selection=legacy_selection,
        expansion_selection=expansion_selection,
        reference_artifact=reference_artifact,
        inputs=(
            long_pool_binding,
            prereg_binding,
            fresh16_binding,
            legacy_binding,
            expansion_binding,
            reference_binding,
        ),
    )
