"""Policy-blind repair for GUIOdyssey terminal decision indexing in Freeze-B."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.set_utility_freeze_b_v1 import (
    MAXIMUM_CANDIDATE_COUNT,
    MAXIMUM_LABELED_CARDINALITY,
    ROLES,
    STRATA,
    STRATUM_ANCHOR,
    _historical_firewall,
    _mapping,
    _safe_relative_path,
    _selection_digest,
    _sequence,
    canonical_pretty_json_bytes,
    load_freeze_b_config,
    sha256_bytes,
    sha256_canonical_json,
    validate_freeze_b_config,
)
from causalcache.set_utility_label_schedule import (
    LabelStateRequest,
    TeacherOperationBudget,
    build_variable_n_label_schedule,
)
from causalcache.set_utility_split_validation import (
    SetUtilitySplitAssignment,
    validate_group_aware_split_assignments,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_freeze_b_v2_terminal_index_repair"
STATUS = "POLICY_BLIND_FREEZE_B_V2_TERMINAL_INDEX_REPAIR_COMPLETED"
EXPECTED_P0_STATUS = "POLICY_BLIND_UNCONSUMED_FULL_POOL_CENSUS_COMPLETED"
PARENT_CONFIG_PATH = "code/configs/causalcache_set_utility_freeze_b_v1.json"
PARENT_MANIFEST_PATH = "data/manifests/set_utility_freeze_b_v1.json"
DEFAULT_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_freeze_b_v2_terminal_index_repair.json"
)
DEFAULT_OUTPUT_PATH = (
    "data/manifests/set_utility_freeze_b_v2_terminal_index_repair.json"
)


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _regular_file(root: Path, relative: str, *, label: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be one regular file")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _lower_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields differ from the frozen schema")


def load_freeze_b_v2_config(
    repository_root: str | Path,
    relative_path: str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = _regular_file(
        root,
        _safe_relative_path(relative_path, label="Freeze-B v2 config path"),
        label="Freeze-B v2 config",
    )
    return _strict_json_object(path.read_bytes(), label="Freeze-B v2 config")


def _load_bound_json(
    root: Path,
    binding: Mapping[str, Any],
    *,
    expected_path: str,
    label: str,
) -> dict[str, Any]:
    _exact_keys(binding, {"path", "sha256"}, label=f"{label} binding")
    path = _safe_relative_path(binding["path"], label=f"{label} path")
    if path != expected_path:
        raise ValueError(f"{label} path drifted")
    expected_sha = _lower_sha256(binding["sha256"], label=f"{label} SHA256")
    source = _regular_file(root, path, label=label)
    if _sha256_file(source) != expected_sha:
        raise ValueError(f"{label} SHA256 drifted")
    return _strict_json_object(source.read_bytes(), label=label)


def validate_freeze_b_v2_config(
    config: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    _exact_keys(
        config,
        {
            "schema_version",
            "protocol_id",
            "status",
            "bindings",
            "repair",
            "authorization",
            "output",
        },
        label="Freeze-B v2 config",
    )
    if (
        config["schema_version"] != SCHEMA_VERSION
        or config["protocol_id"] != PROTOCOL_ID
        or config["status"] != "SOURCE_REPAIR_REQUIRES_MATERIALIZATION"
    ):
        raise ValueError("Freeze-B v2 config identity drifted")

    bindings = _mapping(config["bindings"], label="Freeze-B v2 bindings")
    _exact_keys(
        bindings,
        {"parent_config", "parent_manifest", "p0_census", "repair_sources"},
        label="Freeze-B v2 bindings",
    )
    parent_config = _load_bound_json(
        root,
        _mapping(bindings["parent_config"], label="parent config binding"),
        expected_path=PARENT_CONFIG_PATH,
        label="parent Freeze-B v1 config",
    )
    parent_validation = validate_freeze_b_config(
        parent_config,
        repository_root=root,
    )
    parent_manifest = _load_bound_json(
        root,
        _mapping(bindings["parent_manifest"], label="parent manifest binding"),
        expected_path=PARENT_MANIFEST_PATH,
        label="parent Freeze-B v1 manifest",
    )
    if parent_manifest.get("status") != (
        "POLICY_BLIND_FREEZE_B_ROSTER_AND_QUERY_PLAN_COMPLETED"
    ):
        raise ValueError("parent Freeze-B v1 manifest status drifted")
    p0_path = str(parent_config["source"]["p0_census_manifest_path"])
    p0 = _load_bound_json(
        root,
        _mapping(bindings["p0_census"], label="P0 binding"),
        expected_path=p0_path,
        label="P0 census manifest",
    )
    if p0.get("status") != EXPECTED_P0_STATUS:
        raise ValueError("P0 census status drifted")

    sources = _sequence(bindings["repair_sources"], label="repair sources")
    if not sources:
        raise ValueError("Freeze-B v2 repair source inventory is empty")
    seen_paths: set[str] = set()
    for index, source_value in enumerate(sources):
        source = _mapping(source_value, label=f"repair source {index}")
        _exact_keys(source, {"path", "sha256"}, label="repair source")
        path = _safe_relative_path(source["path"], label="repair source path")
        if path in seen_paths:
            raise ValueError("Freeze-B v2 repair source paths are not unique")
        expected = _lower_sha256(source["sha256"], label="repair source SHA256")
        if _sha256_file(_regular_file(root, path, label="repair source")) != expected:
            raise ValueError(f"Freeze-B v2 repair source SHA256 drifted: {path}")
        seen_paths.add(path)

    repair = _mapping(config["repair"], label="Freeze-B v2 repair")
    expected_repair = {
        "fault": "terminal_decision_step_used_decision_count_without_one_based_offset",
        "decision_count_semantics": "number_of_predictable_actions_after_the_initial_recorded_action",
        "valid_decision_step_id_range": "2..decision_count_plus_1_inclusive",
        "terminal_decision_step_formula": "decision_count_plus_1",
        "stratum_anchor_decision_step": STRATUM_ANCHOR,
        "minimum_decision_count_for_distinct_anchor_and_terminal": STRATUM_ANCHOR,
        "reuse_parent_selection_salt": True,
        "rerun_group_representative_and_role_assignment_on_corrected_eligible_universe": True,
        "states_per_trajectory": 2,
        "maximum_labeled_cardinality": 2,
        "worker_count": 4,
    }
    if dict(repair) != expected_repair:
        raise ValueError("Freeze-B v2 repair semantics drifted")

    authorization = _mapping(
        config["authorization"], label="Freeze-B v2 authorization"
    )
    allowed_true = {
        "read_parent_freeze_b_v1",
        "read_committed_p0_census",
        "materialize_corrected_policy_blind_roster",
        "materialize_corrected_policy_blind_query_plan",
        "write_one_git_manifest",
    }
    expected_auth_keys = allowed_true | {
        "read_raw_source_shards",
        "run_ocr",
        "select_candidates_with_processor",
        "load_policy_or_vision_model",
        "generate_restoration_labels",
        "train_predictor",
        "access_one_shot_evaluation_labels",
        "run_closed_loop",
        "access_sealed_test",
        "mutate_hugging_face",
    }
    if set(authorization) != expected_auth_keys or any(
        not isinstance(value, bool) or value is not (key in allowed_true)
        for key, value in authorization.items()
    ):
        raise ValueError("Freeze-B v2 authorization drifted")

    output = _mapping(config["output"], label="Freeze-B v2 output")
    if dict(output) != {
        "manifest_path": DEFAULT_OUTPUT_PATH,
        "canonical_pretty_json": True,
        "overwrite_allowed": False,
        "contains_raw_instruction": False,
    }:
        raise ValueError("Freeze-B v2 output contract drifted")

    return {
        "status": "VALID_SET_UTILITY_FREEZE_B_V2_TERMINAL_INDEX_REPAIR_SOURCE",
        "config_sha256": sha256_canonical_json(config),
        "parent_config_sha256": parent_validation["config_sha256"],
        "trajectory_count": 1200,
        "query_state_count": 2400,
        "raw_source_access_authorized": False,
        "processor_authorized": False,
        "label_generation_authorized": False,
        "training_authorized": False,
    }


def _query_record(
    assignment: Mapping[str, Any],
    *,
    kind: str,
    step: int,
) -> dict[str, Any]:
    decision_count = int(assignment["decision_count"])
    if not 2 <= step <= decision_count + 1:
        raise ValueError("query decision step is outside the recorded decision range")
    candidate_ids = tuple(range(1, step - 1))[-MAXIMUM_CANDIDATE_COUNT:]
    if len(candidate_ids) < 4:
        raise ValueError("Freeze-B v2 query must have at least four initial candidates")
    source_id = str(assignment["source_id"])
    return {
        "state_id": f"{source_id}:decision:{step:03d}",
        "trajectory_id": source_id,
        "source_id": source_id,
        "role": assignment["role"],
        "candidate_capacity_stratum": assignment["candidate_capacity_stratum"],
        "query_kind": kind,
        "decision_step_id": step,
        "current_equivalent_event_step_id": step - 1,
        "initial_candidate_event_step_ids": list(candidate_ids),
        "initial_candidate_count": len(candidate_ids),
        "maximum_labeled_cardinality": MAXIMUM_LABELED_CARDINALITY,
        "processor_candidate_freeze_status": "PENDING_SEPARATE_EXECUTION",
    }


def _operations_payload(value: TeacherOperationBudget) -> dict[str, int]:
    return {
        field: int(getattr(value, field))
        for field in TeacherOperationBudget.__dataclass_fields__
    }


def derive_freeze_b_v2_manifest(
    config: Mapping[str, Any],
    census: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    validation = validate_freeze_b_v2_config(
        config,
        repository_root=repository_root,
    )
    root = Path(repository_root).resolve()
    bindings = _mapping(config["bindings"], label="Freeze-B v2 bindings")
    parent_config = _load_bound_json(
        root,
        _mapping(bindings["parent_config"], label="parent config binding"),
        expected_path=PARENT_CONFIG_PATH,
        label="parent Freeze-B v1 config",
    )
    if census.get("status") != EXPECTED_P0_STATUS:
        raise ValueError("P0 census status drifted")
    source = _mapping(parent_config["source"], label="parent Freeze-B source")
    pool = _mapping(census.get("pool"), label="P0 pool")
    summary = _mapping(pool.get("summary"), label="P0 pool summary")
    trajectories = _sequence(pool.get("trajectories"), label="P0 trajectories")
    selection_p0 = _mapping(census.get("selection"), label="P0 selection")
    if (
        len(trajectories) != source["candidate_count"]
        or summary.get("trajectory_count") != source["candidate_count"]
        or selection_p0.get("candidate_inventory_sha256")
        != source["candidate_inventory_sha256"]
    ):
        raise ValueError("P0 census candidate inventory drifted")

    firewall = _historical_firewall(census)
    historical_groups = firewall["legacy_groups"] | firewall["forbidden_groups"]
    selection = _mapping(parent_config["selection"], label="parent selection")
    salt = str(selection["salt"])
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    historical_group_exclusion_count = 0
    for value in trajectories:
        record = _mapping(value, label="P0 trajectory")
        group = str(record["instruction_app_group_sha256"])
        if group in historical_groups:
            historical_group_exclusion_count += 1
            continue
        by_group[group].append(record)

    representatives = [
        min(
            records,
            key=lambda item: (
                _selection_digest(
                    salt,
                    "group_representative",
                    group,
                    item["source_id"],
                    item["selection_sha256"],
                ),
                item["source_id"],
            ),
        )
        for group, records in by_group.items()
    ]
    eligible_by_stratum: dict[str, list[Mapping[str, Any]]] = {
        stratum: [] for stratum in STRATA
    }
    too_short_for_two_states = Counter()
    for record in representatives:
        stratum = str(record["candidate_capacity_stratum"])
        if stratum not in STRATA:
            raise ValueError("P0 candidate capacity stratum drifted")
        minimum = STRATUM_ANCHOR[stratum]
        if int(record["decision_count"]) < minimum:
            too_short_for_two_states[stratum] += 1
            continue
        eligible_by_stratum[stratum].append(record)
    if too_short_for_two_states:
        raise RuntimeError("P0 stratum contains a trajectory below its own lower bound")

    quotas = _mapping(
        selection["role_stratum_trajectory_counts"], label="role quotas"
    )
    assignments: list[dict[str, Any]] = []
    for stratum in STRATA:
        ordered = sorted(
            eligible_by_stratum[stratum],
            key=lambda item: (
                _selection_digest(
                    salt,
                    "role_assignment",
                    stratum,
                    item["instruction_app_group_sha256"],
                    item["source_id"],
                    item["selection_sha256"],
                ),
                item["source_id"],
            ),
        )
        cursor = 0
        for role in selection["role_assignment_order"]:
            count = int(_mapping(quotas[role], label="role quota")[stratum])
            selected = ordered[cursor : cursor + count]
            if len(selected) != count:
                raise ValueError(f"insufficient Freeze-B v2 capacity for {role}/{stratum}")
            cursor += count
            assignments.extend(
                {
                    "trajectory_id": str(record["source_id"]),
                    "source_id": str(record["source_id"]),
                    "instruction_app_group_sha256": str(
                        record["instruction_app_group_sha256"]
                    ),
                    "role": str(role),
                    "candidate_capacity_stratum": stratum,
                    "decision_count": int(record["decision_count"]),
                    "terminal_decision_step_id": int(record["decision_count"]) + 1,
                    "transport_file": str(record["transport_file"]),
                    "transport_row_index": int(record["transport_row_index"]),
                    "p0_selection_sha256": str(record["selection_sha256"]),
                }
                for record in selected
            )

    validated = validate_group_aware_split_assignments(
        tuple(
            SetUtilitySplitAssignment(
                trajectory_id=record["trajectory_id"],
                source_id=record["source_id"],
                instruction_app_group_sha256=record[
                    "instruction_app_group_sha256"
                ],
                role=record["role"],
            )
            for record in assignments
        ),
        legacy_train_only_source_ids=firewall["legacy_sources"],
        forbidden_source_ids=firewall["forbidden_sources"],
        legacy_train_only_group_sha256s=firewall["legacy_groups"],
        forbidden_consumed_group_sha256s=firewall["forbidden_groups"],
    )
    assignments.sort(key=lambda item: (item["role"], item["source_id"]))

    query_states: list[dict[str, Any]] = []
    for assignment in assignments:
        stratum = assignment["candidate_capacity_stratum"]
        anchor = STRATUM_ANCHOR[stratum]
        terminal = int(assignment["decision_count"]) + 1
        if anchor >= terminal:
            raise RuntimeError("Freeze-B v2 query steps are not distinct")
        query_states.append(
            _query_record(assignment, kind="stratum_anchor", step=anchor)
        )
        query_states.append(
            _query_record(assignment, kind="terminal", step=terminal)
        )
    query_states.sort(key=lambda item: item["state_id"])
    if len({item["state_id"] for item in query_states}) != len(query_states):
        raise RuntimeError("Freeze-B v2 query state ids are not unique")

    schedule = build_variable_n_label_schedule(
        tuple(
            LabelStateRequest(
                state_id=record["state_id"],
                candidate_event_ids=tuple(record["initial_candidate_event_step_ids"]),
                maximum_cardinality=record["maximum_labeled_cardinality"],
            )
            for record in query_states
        ),
        worker_count=4,
    )
    role_counts = Counter(item["role"] for item in assignments)
    stratum_counts = Counter(item["candidate_capacity_stratum"] for item in assignments)
    query_role_counts = Counter(item["role"] for item in query_states)
    query_n_counts = Counter(item["initial_candidate_count"] for item in query_states)
    pilot_states: list[str] = []
    for stratum in STRATA:
        candidates = [
            item
            for item in query_states
            if item["role"] == "train"
            and item["candidate_capacity_stratum"] == stratum
            and item["query_kind"] == "stratum_anchor"
        ]
        pilot_states.extend(item["state_id"] for item in candidates[:4])
    if len(pilot_states) != 12:
        raise RuntimeError("Freeze-B v2 throughput pilot roster drifted")

    parent_manifest_binding = _mapping(
        bindings["parent_manifest"], label="parent manifest binding"
    )
    p0_binding = _mapping(bindings["p0_census"], label="P0 binding")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": STATUS,
        "repair": {
            **dict(config["repair"]),
            "parent_v1_status": "INVALID_QUERY_PLAN_TERMINAL_OFF_BY_ONE",
            "parent_v1_manifest_path": parent_manifest_binding["path"],
            "parent_v1_manifest_sha256": parent_manifest_binding["sha256"],
            "parent_v1_query_states_must_not_be_used_for_processor_or_labels": True,
        },
        "source": {
            "p0_census_manifest_path": p0_binding["path"],
            "p0_census_manifest_sha256": p0_binding["sha256"],
            "p0_candidate_inventory_sha256": source[
                "candidate_inventory_sha256"
            ],
            "p0_candidate_count": source["candidate_count"],
        },
        "selection": {
            "salt": salt,
            "one_representative_per_instruction_app_group": True,
            "historical_consumed_group_exclusion_count": (
                historical_group_exclusion_count
            ),
            "duplicate_group_nonrepresentative_count": sum(
                len(records) - 1 for records in by_group.values()
            ),
            "too_short_for_two_states_by_stratum": {},
            "eligible_representative_count_by_stratum": {
                stratum: len(eligible_by_stratum[stratum]) for stratum in STRATA
            },
            "assignment_inventory_sha256": sha256_canonical_json(assignments),
            "query_plan_inventory_sha256": sha256_canonical_json(query_states),
        },
        "summary": {
            "trajectory_count": len(assignments),
            "role_trajectory_counts": dict(sorted(role_counts.items())),
            "stratum_trajectory_counts": dict(sorted(stratum_counts.items())),
            "query_state_count": len(query_states),
            "role_query_state_counts": dict(sorted(query_role_counts.items())),
            "initial_candidate_count_histogram": {
                str(key): value for key, value in sorted(query_n_counts.items())
            },
            "initial_exact_schedule_raw_label_rows": (
                schedule.operations.raw_label_rows
            ),
            "contains_raw_instruction": False,
            "processor_candidate_freeze_status": "PENDING_SEPARATE_EXECUTION",
            "exact_operation_budget_status": "PENDING_PROCESSOR_CANDIDATE_FREEZE",
        },
        "split_audit": validated.audit.to_payload(),
        "throughput_pilot": {
            "role": "train_only",
            "state_count": 12,
            "state_ids": pilot_states,
            "processor_only_candidate_freeze_may_run_here": False,
            "policy_throughput_requires_separate_post_freeze_contract": True,
            "may_write_utility_or_kl": False,
            "may_change_roster_or_training_grid": False,
        },
        "candidate_freeze_contract": parent_config["candidate_freeze"],
        "teacher": parent_config["teacher"],
        "operation_ceiling": parent_config["operation_ceiling"],
        "initial_exact_schedule": {
            "status": "PRE_PROCESSOR_UPPER_BOUND_NOT_EXECUTION_AUTHORIZATION",
            "worker_count": schedule.worker_count,
            "state_count_by_candidate_count": {
                str(key): value for key, value in schedule.state_count_by_candidate_count
            },
            "subset_count_by_cardinality": {
                str(key): value for key, value in schedule.subset_count_by_cardinality
            },
            "operations": _operations_payload(schedule.operations),
            "subset_identity_sha256": schedule.subset_identity_sha256,
            "inventory_sha256": schedule.inventory_sha256,
            "execution_sha256": schedule.execution_sha256,
        },
        "features": parent_config["features"],
        "training": parent_config["training"],
        "evaluation": parent_config["evaluation"],
        "artifacts": parent_config["artifacts"],
        "assignments": assignments,
        "query_states": query_states,
        "operation_counts": {
            "parent_v1_manifest_read_count": 1,
            "p0_manifest_read_count": 1,
            "trajectory_role_assignment_count": len(assignments),
            "query_state_selection_count": len(query_states),
            "raw_source_shard_read_count": 0,
            "processor_call_count": 0,
            "ocr_call_count": 0,
            "policy_or_vision_model_load_count": 0,
            "policy_or_vision_forward_count": 0,
            "restoration_label_count": 0,
            "optimizer_step_count": 0,
            "one_shot_evaluation_label_access_count": 0,
            "closed_loop_episode_count": 0,
            "sealed_test_access_count": 0,
        },
        "authorization": {
            "processor_candidate_freeze_authorized": False,
            "throughput_pilot_authorized": False,
            "restoration_label_generation_authorized": False,
            "predictor_training_authorized": False,
            "one_shot_evaluation_authorized": False,
            "closed_loop_authorized": False,
            "sealed_test_authorized": False,
        },
        "source_config_sha256": validation["config_sha256"],
    }
    if manifest["summary"]["trajectory_count"] != 1200 or manifest["summary"][
        "query_state_count"
    ] != 2400:
        raise RuntimeError("Freeze-B v2 output denominator drifted")
    if schedule.operations.raw_label_rows > parent_config["operation_ceiling"][
        "maximum_raw_label_rows"
    ]:
        raise RuntimeError("Freeze-B v2 initial schedule exceeds the generic ceiling")
    return manifest


def materialize_freeze_b_v2_manifest(
    *,
    repository_root: str | Path,
    config_relative_path: str = DEFAULT_CONFIG_PATH,
    output_relative_path: str = DEFAULT_OUTPUT_PATH,
) -> tuple[dict[str, Any], str]:
    root = Path(repository_root).resolve()
    config = load_freeze_b_v2_config(root, config_relative_path)
    validate_freeze_b_v2_config(config, repository_root=root)
    p0_binding = _mapping(config["bindings"]["p0_census"], label="P0 binding")
    census_path = _regular_file(
        root,
        _safe_relative_path(p0_binding["path"], label="P0 census path"),
        label="P0 census manifest",
    )
    census = _strict_json_object(census_path.read_bytes(), label="P0 census manifest")
    manifest = derive_freeze_b_v2_manifest(
        config,
        census,
        repository_root=root,
    )
    output_relative = _safe_relative_path(
        output_relative_path,
        label="Freeze-B v2 output path",
    )
    if output_relative != config["output"]["manifest_path"]:
        raise ValueError("Freeze-B v2 output path differs from the source contract")
    output = root.joinpath(*PurePosixPath(output_relative).parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_pretty_json_bytes(manifest)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    if output.read_bytes() != payload:
        raise RuntimeError("Freeze-B v2 output readback differs from written bytes")
    return manifest, sha256_bytes(payload)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OUTPUT_PATH",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "STATUS",
    "derive_freeze_b_v2_manifest",
    "load_freeze_b_v2_config",
    "materialize_freeze_b_v2_manifest",
    "validate_freeze_b_v2_config",
]
