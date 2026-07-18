"""Deterministic structural selection for long-horizon development states."""

from __future__ import annotations

import copy
import hashlib
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.data.guiodyssey_independent import (
    Candidate,
    candidate_sort_key,
    canonical_json_bytes,
    inspect_candidate,
    source_file_specs,
    verify_local_source_files,
)
from causalcache.long_horizon_contract import (
    LongHorizonContract,
    PROTOCOL_ID,
    SELECTION_SALT,
    historical_role_inventory,
    sha256_bytes,
    source_ids_sha256,
)


MANIFEST_SCHEMA_VERSION = "1.0.0"
MANIFEST_STATUS = "SEALED_LONG_HORIZON_STRUCTURAL_SELECTION_V1"
STATE_ID_TEMPLATE = "{source_id}:decision_step:{decision_step_id:03d}"


@dataclass(frozen=True)
class LongHorizonPool:
    candidates: tuple[Candidate, ...]
    total_source_rows: int
    source_row_counts: Mapping[str, int]
    exclusion_counts: Mapping[str, int]


def pretty_json_bytes(value: Any) -> bytes:
    import json

    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def state_id(source_id: str, decision_step_id: int) -> str:
    if not source_id or decision_step_id not in (10, 18):
        raise ValueError("long-horizon state identity requires step 10 or 18")
    return STATE_ID_TEMPLATE.format(
        source_id=source_id,
        decision_step_id=decision_step_id,
    )


def derive_structural_inspection_config(
    *, parent_config: Mapping[str, Any], contract: LongHorizonContract
) -> dict[str, Any]:
    if parent_config.get("protocol_id") != "independent_reference_gate_v1":
        raise ValueError("unexpected parent structural protocol")
    expected_parent = contract.source_pool["parent_protocol_config"]
    if expected_parent["protocol_id"] != parent_config["protocol_id"]:
        raise ValueError("parent structural protocol ID drifted")
    derived = copy.deepcopy(dict(parent_config))
    eligibility = derived["eligibility"]
    frozen = contract.source_pool["structural_eligibility"]
    eligibility["minimum_decisions_per_trajectory"] = frozen[
        "minimum_decisions_per_trajectory"
    ]
    eligibility["maximum_decisions_per_trajectory"] = frozen[
        "maximum_decisions_per_trajectory"
    ]
    derived["selection"]["trajectory_salt"] = contract.selection["trajectory_salt"]
    return derived


def _iter_rows(parquet_file: Any) -> Iterator[tuple[int, Mapping[str, Any]]]:
    row_index = 0
    for row_group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_index)
        for row in table.to_pylist():
            if not isinstance(row, Mapping):
                raise ValueError("Parquet row must decode to a mapping")
            yield row_index, row
            row_index += 1
    if row_index != int(parquet_file.metadata.num_rows):
        raise RuntimeError("Parquet row iteration did not match metadata.num_rows")


def _source_path(source_root: Path, transport_file: str) -> Path:
    return source_root.joinpath(*PurePosixPath(transport_file).parts)


def reconstruct_long_horizon_pool(
    *,
    source_root: Path,
    parent_config: Mapping[str, Any],
    source_file_manifest: Mapping[str, Any],
    contract: LongHorizonContract,
) -> LongHorizonPool:
    """Scan every pinned Parquet row using the inherited full structural parser."""
    inspection_config = derive_structural_inspection_config(
        parent_config=parent_config,
        contract=contract,
    )
    specs = source_file_specs(inspection_config, source_file_manifest)
    verify_local_source_files(source_root, specs)

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for pinned raw Parquet scanning") from error

    candidates: list[Candidate] = []
    exclusions: Counter[str] = Counter()
    source_row_counts: dict[str, int] = {}
    seen_source_ids: dict[str, tuple[str, int]] = {}
    total_source_rows = 0
    for spec in specs:
        parquet = pq.ParquetFile(_source_path(source_root, spec.transport_file))
        row_count = int(parquet.metadata.num_rows)
        source_row_counts[spec.transport_file] = row_count
        total_source_rows += row_count
        for row_index, row in _iter_rows(parquet):
            inspection = inspect_candidate(
                row,
                transport_file=spec.transport_file,
                transport_row_index=row_index,
                config=inspection_config,
            )
            if inspection.source_id is not None:
                previous = seen_source_ids.get(inspection.source_id)
                if previous is not None:
                    raise ValueError(
                        "duplicate source_id across pinned raw rows: "
                        f"{inspection.source_id} at {previous} and "
                        f"{(spec.transport_file, row_index)}"
                    )
                seen_source_ids[inspection.source_id] = (
                    spec.transport_file,
                    row_index,
                )
            if inspection.candidate is None:
                if inspection.exclusion_reason is None:
                    raise RuntimeError("excluded structural row has no exclusion reason")
                exclusions[inspection.exclusion_reason.value] += 1
            else:
                candidates.append(inspection.candidate)
    if total_source_rows != sum(source_row_counts.values()):
        raise RuntimeError("source-row accounting mismatch")
    return LongHorizonPool(
        candidates=tuple(candidates),
        total_source_rows=total_source_rows,
        source_row_counts=dict(sorted(source_row_counts.items())),
        exclusion_counts=dict(sorted(exclusions.items())),
    )


def _candidate_record(
    candidate: Candidate,
    *,
    selection_rank: int,
    role: str,
    include_development_summary: bool,
) -> dict[str, Any]:
    action_counts = candidate.action_counts()
    if sum(action_counts.values()) != candidate.decision_count:
        raise ValueError(
            f"candidate action accounting drifted for {candidate.source_id}"
        )
    record: dict[str, Any] = {
        "trajectory_id": candidate.source_id,
        "shard_path": candidate.transport_file,
        "row_index": candidate.transport_row_index,
        "decision_count": candidate.decision_count,
        "selection_sha256": candidate.selection_sha256,
        "selection_rank": selection_rank,
        "role": role,
    }
    # note (luojiaxuan): Reserve records deliberately stop at frozen structural
    # locators and hashes so downstream builders cannot silently treat reserve
    # rows as development semantic inputs.
    if include_development_summary:
        record["action_type_counts"] = dict(sorted(action_counts.items()))
        record["normalized_app_labels"] = list(candidate.normalized_app_labels)
    return record


def _state_records(
    development: Sequence[Candidate], *, contract: LongHorizonContract
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in development:
        for geometry in contract.geometry["states"]:
            step = int(geometry["decision_step_id"])
            if candidate.decision_count < step:
                raise ValueError(
                    f"trajectory {candidate.source_id} cannot supply decision step {step}"
                )
            candidates = list(geometry["candidate_event_step_ids"])
            history = list(range(1, step))
            if candidates != history[:-1]:
                raise ValueError("candidate archive is not the strict history prefix")
            records.append(
                {
                    "state_id": state_id(candidate.source_id, step),
                    "trajectory_id": candidate.source_id,
                    "decision_step_id": step,
                    "history_event_step_ids": history,
                    "current_equivalent_event_step_id": geometry[
                        "current_equivalent_event_step_id"
                    ],
                    "candidate_event_count": geometry["candidate_event_count"],
                    "candidate_event_step_ids": candidates,
                }
            )
    return records


def _split_summary(
    records: Sequence[Mapping[str, Any]], *, include_development_summary: bool
) -> dict[str, Any]:
    apps: set[str] = set()
    action_counts: Counter[str] = Counter()
    for record in records:
        if include_development_summary:
            apps.update(str(value) for value in record["normalized_app_labels"])
            action_counts.update(record["action_type_counts"])
    source_ids = [str(record["trajectory_id"]) for record in records]
    summary = {
        "trajectory_count": len(records),
        "decision_count": sum(int(record["decision_count"]) for record in records),
        "trajectory_ids": source_ids,
        "trajectory_ids_sha256": source_ids_sha256(source_ids),
    }
    if include_development_summary:
        summary["distinct_app_labels"] = sorted(apps)
        summary["action_type_counts"] = dict(sorted(action_counts.items()))
    return summary


def build_long_horizon_selection_manifest(
    *,
    pool: LongHorizonPool,
    contract: LongHorizonContract,
    parent_config_sha256: str,
    source_file_manifest_sha256: str,
    generator: Mapping[str, Any],
) -> dict[str, Any]:
    expected_count = int(
        contract.source_pool["structural_eligibility"][
            "expected_eligible_trajectory_count"
        ]
    )
    ordered = tuple(sorted(pool.candidates, key=candidate_sort_key))
    if len(ordered) != expected_count:
        raise ValueError(
            f"eligible trajectory count must be exactly {expected_count}, got {len(ordered)}"
        )
    source_ids = [candidate.source_id for candidate in ordered]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("eligible source IDs must be unique")
    if any(candidate.decision_count < 18 or candidate.decision_count > 60 for candidate in ordered):
        raise ValueError("eligible trajectory escaped the frozen decision-count bounds")
    expected_hashes = [
        hashlib.sha256(f"{SELECTION_SALT}\0{source_id}".encode("utf-8")).hexdigest()
        for source_id in source_ids
    ]
    if [candidate.selection_sha256 for candidate in ordered] != expected_hashes:
        raise ValueError("eligible selection hashes do not use the frozen long-horizon salt")

    development_count = int(contract.selection["development_trajectory_count"])
    reserve_count = int(contract.selection["unopened_reserve_trajectory_count"])
    if development_count + reserve_count != expected_count:
        raise ValueError("development and reserve counts do not partition the eligible pool")
    development = ordered[:development_count]
    reserve = ordered[development_count:]
    development_records = [
        _candidate_record(
            candidate,
            selection_rank=index,
            role="development",
            include_development_summary=True,
        )
        for index, candidate in enumerate(development)
    ]
    reserve_records = [
        _candidate_record(
            candidate,
            selection_rank=development_count + index,
            role="unopened_reserve",
            include_development_summary=False,
        )
        for index, candidate in enumerate(reserve)
    ]
    eligible_locators = [
        {
            key: record[key]
            for key in (
                "trajectory_id",
                "shard_path",
                "row_index",
                "decision_count",
                "selection_sha256",
                "selection_rank",
                "role",
            )
        }
        for record in development_records + reserve_records
    ]

    historical = historical_role_inventory(
        contract.data,
        repository_root=contract.repository_root,
    )
    historical_ids = set(historical["union_source_ids"])
    overlap = sorted(historical_ids.intersection(source_ids))
    if overlap:
        raise ValueError(f"long-horizon IDs overlap historical roles: {overlap}")

    eligible_pool_sha256 = sha256_bytes(canonical_json_bytes(eligible_locators))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": MANIFEST_STATUS,
        "inputs": {
            "contract_path": "code/configs/causalcache_long_horizon_development_v1.json",
            "contract_sha256": contract.sha256,
            "parent_protocol_config_path": contract.source_pool[
                "parent_protocol_config"
            ]["path"],
            "parent_protocol_config_sha256": parent_config_sha256,
            "source_file_manifest_path": contract.source_pool[
                "source_file_manifest"
            ]["path"],
            "source_file_manifest_sha256": source_file_manifest_sha256,
            "transport_repo": contract.source_pool["source_file_manifest"]["repo"],
            "transport_revision": contract.source_pool["source_file_manifest"][
                "revision"
            ],
            "transport_file_count": contract.source_pool["source_file_manifest"][
                "file_count"
            ],
        },
        "generator": dict(generator),
        "semantics": {
            "structural_scan_only": True,
            "full_parent_structural_validation_except_decision_bounds": True,
            "minimum_decisions_per_trajectory": 18,
            "maximum_decisions_per_trajectory": 60,
            "trajectory_salt": SELECTION_SALT,
            "padding_count": 0,
            "trajectory_splicing_count": 0,
            "top_up_count": 0,
            "post_selection_filter_count": 0,
        },
        "reconstruction": {
            "total_source_rows": pool.total_source_rows,
            "source_row_counts": dict(pool.source_row_counts),
            "exclusion_counts": dict(pool.exclusion_counts),
            "eligible_trajectory_count": len(ordered),
            "eligible_pool_sha256": eligible_pool_sha256,
            "eligible_trajectory_locators": eligible_locators,
        },
        "historical_role_firewall": {
            **historical,
            "selected_source_id_count": len(source_ids),
            "selected_source_ids_sha256": source_ids_sha256(source_ids),
            "intersection_count": 0,
            "intersection_source_ids": [],
        },
        "splits": {
            "development": {
                "status": "CONSUMED_DEVELOPMENT_AFTER_EXECUTION_B_ONLY",
                **_split_summary(
                    development_records, include_development_summary=True
                ),
                "trajectories": development_records,
                "state_count": development_count * 2,
                "states": _state_records(development, contract=contract),
            },
            "unopened_reserve": {
                "status": "STRUCTURALLY_SELECTED_BUT_UNOPENED_FOR_POLICY_OCR_RESTORATION_AND_SELECTORS",
                **_split_summary(
                    reserve_records, include_development_summary=False
                ),
                "trajectories": reserve_records,
                "state_count": 0,
                "states": [],
            },
        },
        "reference_plan": copy.deepcopy(contract.data["reference_contract"]),
        "selector_plan": copy.deepcopy(contract.data["selector_matrix"]),
        "validity_and_go_plan": copy.deepcopy(
            contract.data["validity_and_go_contract"]
        ),
        "policy_context_profile": copy.deepcopy(
            contract.data["policy_context_profile"]
        ),
        "operation_plan": copy.deepcopy(contract.data["operation_accounting"]),
        "execution_b_plan": copy.deepcopy(contract.data["execution_b_plan"]),
        "artifact_plan": copy.deepcopy(contract.data["artifact_plan"]),
        "learned_model_artifacts": copy.deepcopy(
            contract.data["learned_model_artifacts"]
        ),
        "access_accounting": {
            "policy_load_count": 0,
            "policy_forward_count": 0,
            "ocr_access_count": 0,
            "restoration_distance_count": 0,
            "restoration_label_access_count": 0,
            "selector_scoring_count": 0,
            "selector_training_count": 0,
            "exact_subset_evaluation_count": 0,
            "pair_union_distance_count": 0,
            "reserve_semantic_access_count": 0,
            "old_confirm_access_count": 0,
            "closed_loop_access_count": 0,
            "sealed_test_access_count": 0,
        },
    }
    validate_long_horizon_selection_manifest(manifest, contract=contract)
    return manifest


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _validate_trajectory_record(
    record: Mapping[str, Any],
    index: int,
    *,
    expected_role: str,
    development: bool,
) -> None:
    expected_keys = {
        "trajectory_id",
        "shard_path",
        "row_index",
        "decision_count",
        "selection_sha256",
        "selection_rank",
        "role",
    }
    if development:
        expected_keys.update({"action_type_counts", "normalized_app_labels"})
    if set(record) != expected_keys:
        raise ValueError("eligible trajectory record schema drifted")
    if record["selection_rank"] != index or type(record["selection_rank"]) is not int:
        raise ValueError("selection rank drifted")
    if record["role"] != expected_role:
        raise ValueError("trajectory role drifted")
    source_id = record["trajectory_id"]
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("eligible source ID must be non-empty text")
    if type(record["row_index"]) is not int or record["row_index"] < 0:
        raise ValueError("transport row index must be a non-negative integer")
    if type(record["decision_count"]) is not int or not 18 <= record["decision_count"] <= 60:
        raise ValueError("decision count escaped [18, 60]")
    if development:
        actions = _require_mapping(record["action_type_counts"], "action counts")
        if (
            not actions
            or any(type(value) is not int or value <= 0 for value in actions.values())
            or sum(actions.values()) != record["decision_count"]
        ):
            raise ValueError("action counts do not cover the full decision count")
        apps = _require_sequence(record["normalized_app_labels"], "app labels")
        if not apps or list(apps) != sorted(set(apps)):
            raise ValueError("normalized app labels must be sorted and unique")
    expected_hash = hashlib.sha256(
        f"{SELECTION_SALT}\0{source_id}".encode("utf-8")
    ).hexdigest()
    if record["selection_sha256"] != expected_hash:
        raise ValueError("trajectory selection hash drifted")
    if re.fullmatch(r"mobile/use/train/shard-\d{5}-of-00610\.parquet", str(record["shard_path"])) is None:
        raise ValueError("transport file path is outside the frozen mobile train shards")


def validate_long_horizon_selection_manifest(
    manifest: Mapping[str, Any], *, contract: LongHorizonContract
) -> None:
    expected_top = {
        "schema_version",
        "protocol_id",
        "status",
        "inputs",
        "generator",
        "semantics",
        "reconstruction",
        "historical_role_firewall",
        "splits",
        "reference_plan",
        "selector_plan",
        "validity_and_go_plan",
        "policy_context_profile",
        "operation_plan",
        "execution_b_plan",
        "artifact_plan",
        "learned_model_artifacts",
        "access_accounting",
    }
    if set(manifest) != expected_top:
        raise ValueError("long-horizon selection manifest top-level schema drifted")
    if (
        manifest["schema_version"] != MANIFEST_SCHEMA_VERSION
        or manifest["protocol_id"] != PROTOCOL_ID
        or manifest["status"] != MANIFEST_STATUS
    ):
        raise ValueError("long-horizon selection identity drifted")
    generator = _require_mapping(manifest["generator"], "selection generator")
    expected_generator_keys = {
        "git_revision",
        "module_path",
        "module_sha256",
        "cli_path",
        "cli_sha256",
        "contract_validator_path",
        "contract_validator_sha256",
    }
    if set(generator) != expected_generator_keys:
        raise ValueError("selection generator schema drifted")
    if re.fullmatch(r"[0-9a-f]{40}", str(generator["git_revision"])) is None:
        raise ValueError("selection generator Git revision is invalid")
    for key in ("module_sha256", "cli_sha256", "contract_validator_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(generator[key])) is None:
            raise ValueError(f"selection generator {key} is invalid")
    if (
        generator["module_path"] != "code/causalcache/long_horizon_selection.py"
        or generator["cli_path"] != "code/scripts/build_long_horizon_selection.py"
        or generator["contract_validator_path"]
        != "code/scripts/validate_long_horizon_contract.py"
    ):
        raise ValueError("selection generator paths drifted")
    inputs = _require_mapping(manifest["inputs"], "selection inputs")
    if (
        inputs.get("contract_sha256") != contract.sha256
        or inputs.get("parent_protocol_config_sha256")
        != contract.source_pool["parent_protocol_config"]["sha256"]
        or inputs.get("source_file_manifest_sha256")
        != contract.source_pool["source_file_manifest"]["sha256"]
        or inputs.get("transport_file_count") != 16
    ):
        raise ValueError("selection input identity drifted")
    semantics = _require_mapping(manifest["semantics"], "selection semantics")
    for key in (
        "structural_scan_only",
        "full_parent_structural_validation_except_decision_bounds",
    ):
        if semantics.get(key) is not True:
            raise ValueError("selection is not a full structural-only scan")
    for key in (
        "padding_count",
        "trajectory_splicing_count",
        "top_up_count",
        "post_selection_filter_count",
    ):
        if type(semantics.get(key)) is not int or semantics[key] != 0:
            raise ValueError(f"forbidden selection operation recorded: {key}")

    reconstruction = _require_mapping(manifest["reconstruction"], "reconstruction")
    locators = list(
        _require_sequence(
            reconstruction.get("eligible_trajectory_locators"),
            "eligible trajectory locators",
        )
    )
    if len(locators) != 42 or reconstruction.get("eligible_trajectory_count") != 42:
        raise ValueError("eligible trajectory count must be exactly 42")
    sort_keys = [
        (
            record["selection_sha256"],
            record["trajectory_id"],
            record["shard_path"],
            record["row_index"],
        )
        for record in locators
    ]
    if sort_keys != sorted(sort_keys):
        raise ValueError("eligible trajectories are not in frozen hash order")
    if reconstruction.get("eligible_pool_sha256") != sha256_bytes(
        canonical_json_bytes(locators)
    ):
        raise ValueError("eligible-pool digest drifted")

    ids = [record["trajectory_id"] for record in locators]
    if len(ids) != len(set(ids)):
        raise ValueError("eligible source IDs are not unique")
    locators_as_tuples = [
        (record["shard_path"], record["row_index"]) for record in locators
    ]
    if len(locators_as_tuples) != len(set(locators_as_tuples)):
        raise ValueError("eligible source locators are not unique")
    firewall = _require_mapping(
        manifest["historical_role_firewall"], "historical firewall"
    )
    expected_historical = historical_role_inventory(
        contract.data,
        repository_root=contract.repository_root,
    )
    if (
        firewall.get("union_source_id_count")
        != expected_historical["union_source_id_count"]
        or firewall.get("union_source_ids_sha256")
        != expected_historical["union_source_ids_sha256"]
        or firewall.get("union_source_ids")
        != expected_historical["union_source_ids"]
        or firewall.get("intersection_count") != 0
        or firewall.get("intersection_source_ids") != []
        or set(ids).intersection(expected_historical["union_source_ids"])
    ):
        raise ValueError("historical role firewall failed")

    splits = _require_mapping(manifest["splits"], "splits")
    development = _require_mapping(splits.get("development"), "development split")
    reserve = _require_mapping(splits.get("unopened_reserve"), "reserve split")
    development_records = list(
        _require_sequence(development.get("trajectories"), "development trajectories")
    )
    reserve_records = list(
        _require_sequence(reserve.get("trajectories"), "reserve trajectories")
    )
    for index, raw_record in enumerate(development_records):
        _validate_trajectory_record(
            _require_mapping(raw_record, "development trajectory"),
            index,
            expected_role="development",
            development=True,
        )
    for offset, raw_record in enumerate(reserve_records):
        _validate_trajectory_record(
            _require_mapping(raw_record, "reserve trajectory"),
            24 + offset,
            expected_role="unopened_reserve",
            development=False,
        )
    projected_records = [
        {
            key: record[key]
            for key in (
                "trajectory_id",
                "shard_path",
                "row_index",
                "decision_count",
                "selection_sha256",
                "selection_rank",
                "role",
            )
        }
        for record in development_records + reserve_records
    ]
    if projected_records != locators:
        raise ValueError("split trajectory records do not reproduce eligible locators")
    if development.get("trajectory_count") != 24 or reserve.get("trajectory_count") != 18:
        raise ValueError("development/reserve partition counts drifted")
    development_ids = {record["trajectory_id"] for record in development_records}
    reserve_ids = {record["trajectory_id"] for record in reserve_records}
    development_locators = {
        (record["shard_path"], record["row_index"]) for record in development_records
    }
    reserve_locators = {
        (record["shard_path"], record["row_index"]) for record in reserve_records
    }
    if development_ids.intersection(reserve_ids) or development_locators.intersection(
        reserve_locators
    ):
        raise ValueError("development and reserve overlap by ID or locator")
    if reserve.get("state_count") != 0 or reserve.get("states") != []:
        raise ValueError("unopened reserve states must not be materialized")
    states = list(_require_sequence(development.get("states"), "development states"))
    if development.get("state_count") != 48 or len(states) != 48:
        raise ValueError("development must contain exactly 48 state skeletons")
    expected_states: list[dict[str, Any]] = []
    for record in development_records:
        for step, count in ((10, 8), (18, 16)):
            expected_states.append(
                {
                    "state_id": state_id(record["trajectory_id"], step),
                    "trajectory_id": record["trajectory_id"],
                    "decision_step_id": step,
                    "history_event_step_ids": list(range(1, step)),
                    "current_equivalent_event_step_id": step - 1,
                    "candidate_event_count": count,
                    "candidate_event_step_ids": list(range(1, count + 1)),
                }
            )
    if states != expected_states:
        raise ValueError("development state skeletons drifted")

    if manifest["reference_plan"] != contract.data["reference_contract"]:
        raise ValueError("reference plan drifted")
    if manifest["selector_plan"] != contract.data["selector_matrix"]:
        raise ValueError("selector plan drifted")
    if manifest["validity_and_go_plan"] != contract.data["validity_and_go_contract"]:
        raise ValueError("validity and GO plan drifted")
    if manifest["policy_context_profile"] != contract.data["policy_context_profile"]:
        raise ValueError("policy context profile drifted")
    if manifest["operation_plan"] != contract.data["operation_accounting"]:
        raise ValueError("operation accounting plan drifted")
    if manifest["execution_b_plan"] != contract.data["execution_b_plan"]:
        raise ValueError("Execution-B plan drifted")
    if manifest["artifact_plan"] != contract.data["artifact_plan"]:
        raise ValueError("artifact plan drifted")
    if manifest["learned_model_artifacts"] != contract.data["learned_model_artifacts"]:
        raise ValueError("learned model artifact roster drifted")
    accounting = _require_mapping(manifest["access_accounting"], "access accounting")
    if not accounting or any(type(value) is not int or value != 0 for value in accounting.values()):
        raise ValueError("structural selection accessed a forbidden semantic surface")


def development_input_records(
    manifest: Mapping[str, Any], *, contract: LongHorizonContract
) -> tuple[Mapping[str, Any], ...]:
    """Return only the consumed-development locators for downstream builders."""
    validate_long_horizon_selection_manifest(manifest, contract=contract)
    development = _require_mapping(manifest["splits"]["development"], "development")
    records = _require_sequence(development["trajectories"], "development trajectories")
    # note (luojiaxuan): This accessor intentionally has no role argument. A
    # downstream substrate builder cannot switch it to the unopened reserve by
    # configuration or by reusing a generic split loader.
    return tuple(_require_mapping(record, "development trajectory") for record in records)
