"""Deterministic policy-blind Freeze-B roster and query-plan materialization."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.set_utility_split_validation import (
    SetUtilitySplitAssignment,
    validate_group_aware_split_assignments,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_freeze_b_v1"
STATUS = "POLICY_BLIND_FREEZE_B_ROSTER_AND_QUERY_PLAN_COMPLETED"
EXPECTED_P0_STATUS = "POLICY_BLIND_UNCONSUMED_FULL_POOL_CENSUS_COMPLETED"
ROLES = ("train", "tune", "evaluation")
STRATA = ("decisions_6_9", "decisions_10_17", "decisions_18_plus")
STRATUM_ANCHOR = {
    "decisions_6_9": 6,
    "decisions_10_17": 10,
    "decisions_18_plus": 18,
}
STRATUM_MINIMUM_FOR_TWO_STATES = {
    "decisions_6_9": 7,
    "decisions_10_17": 11,
    "decisions_18_plus": 19,
}
MAXIMUM_CANDIDATE_COUNT = 16
MAXIMUM_LABELED_CARDINALITY = 2


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_canonical_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


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


def _safe_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe relative path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or str(parsed) != value or any(
        part in {".", ".."} for part in parsed.parts
    ):
        raise ValueError(f"{label} must be a safe relative path")
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


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be an ordered sequence")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields differ from the frozen schema")


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _lower_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA256")
    return value


def load_freeze_b_config(
    repository_root: str | Path,
    relative_path: str = "code/configs/causalcache_set_utility_freeze_b_v1.json",
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = _regular_file(
        root,
        _safe_relative_path(relative_path, label="Freeze-B config path"),
        label="Freeze-B config",
    )
    return _strict_json_object(path.read_bytes(), label="Freeze-B config")


def validate_freeze_b_config(
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
            "source",
            "selection",
            "query_plan",
            "candidate_freeze",
            "teacher",
            "features",
            "training",
            "evaluation",
            "artifacts",
            "operation_ceiling",
            "authorization",
        },
        label="Freeze-B config",
    )
    if (
        config["schema_version"] != SCHEMA_VERSION
        or config["protocol_id"] != PROTOCOL_ID
        or config["status"] != "SOURCE_FREEZE_REQUIRES_ROSTER_MATERIALIZATION"
    ):
        raise ValueError("Freeze-B config identity drifted")

    source = _mapping(config["source"], label="Freeze-B source")
    census_path = _safe_relative_path(
        source.get("p0_census_manifest_path"),
        label="P0 census manifest path",
    )
    census_sha = _lower_sha256(
        source.get("p0_census_manifest_sha256"),
        label="P0 census manifest SHA256",
    )
    if _sha256_file(_regular_file(root, census_path, label="P0 census manifest")) != census_sha:
        raise ValueError("P0 census manifest SHA256 drifted")
    if source.get("candidate_count") != 6933 or source.get(
        "candidate_inventory_sha256"
    ) != "b1eccb42df2e8cd7502c56c007ba01e5ddbebda72086f4b0d487c8865d68c11d":
        raise ValueError("Freeze-B P0 candidate binding drifted")

    selection = _mapping(config["selection"], label="Freeze-B selection")
    counts = _mapping(selection.get("role_stratum_trajectory_counts"), label="role counts")
    role_totals: Counter[str] = Counter()
    stratum_totals: Counter[str] = Counter()
    for role in ROLES:
        by_stratum = _mapping(counts.get(role), label=f"{role} stratum counts")
        if set(by_stratum) != set(STRATA):
            raise ValueError("Freeze-B role strata drifted")
        for stratum in STRATA:
            count = _positive_int(by_stratum[stratum], label=f"{role}/{stratum} count")
            role_totals[role] += count
            stratum_totals[stratum] += count
    if dict(role_totals) != {"train": 1000, "tune": 100, "evaluation": 100}:
        raise ValueError("Freeze-B role totals drifted")
    if any(stratum_totals[stratum] != 400 for stratum in STRATA):
        raise ValueError("Freeze-B strata are not balanced at 400 trajectories each")
    if selection.get("role_assignment_order") != ["evaluation", "tune", "train"]:
        raise ValueError("Freeze-B role assignment order drifted")
    if selection.get("one_representative_per_instruction_app_group") is not True:
        raise ValueError("Freeze-B group representative rule drifted")
    if selection.get("exclude_all_historical_consumed_groups") is not True:
        raise ValueError("Freeze-B historical group firewall drifted")

    query = _mapping(config["query_plan"], label="Freeze-B query plan")
    if (
        query.get("states_per_trajectory") != 2
        or query.get("maximum_labeled_cardinality") != 2
        or query.get("stratum_anchor_decision_step") != STRATUM_ANCHOR
        or query.get("second_state") != "terminal_decision_step"
    ):
        raise ValueError("Freeze-B query plan drifted")

    def validate_source_inventory(value: Any, *, label: str) -> None:
        source_files = _sequence(value, label=f"{label} source files")
        if not source_files:
            raise ValueError(f"Freeze-B {label} source inventory is empty")
        for index, record_value in enumerate(source_files):
            record = _mapping(record_value, label=f"{label} source file {index}")
            _exact_keys(record, {"path", "sha256"}, label=f"{label} source file")
            relative = _safe_relative_path(
                record["path"],
                label=f"{label} source path",
            )
            expected = _lower_sha256(
                record["sha256"],
                label=f"{label} source SHA256",
            )
            if _sha256_file(
                _regular_file(root, relative, label=f"{label} source")
            ) != expected:
                raise ValueError(f"Freeze-B {label} source SHA256 drifted: {relative}")

    features = _mapping(config["features"], label="Freeze-B features")
    if features.get("schema_id") != (
        "lightweight_q64_h75_pair8_plus_gui_owl_final_main_"
        "normalized_visual4096_v1"
    ):
        raise ValueError("Freeze-B feature schema drifted")
    validate_source_inventory(features.get("source_files"), label="feature")

    teacher = _mapping(config["teacher"], label="Freeze-B teacher")
    validate_source_inventory(teacher.get("source_files"), label="teacher")

    training = _mapping(config["training"], label="Freeze-B training")
    if training.get("model_families") != [
        "set_transformer",
        "deepsets",
        "pairwise_additive",
    ]:
        raise ValueError("Freeze-B model-family order drifted")
    if training.get("seeds") != [17, 29, 43]:
        raise ValueError("Freeze-B training seeds drifted")
    if training.get("budget_is_model_feature") is not False:
        raise ValueError("Freeze-B predictor may not receive budget")
    validate_source_inventory(training.get("source_files"), label="training")

    ceiling = _mapping(config["operation_ceiling"], label="operation ceiling")
    state_count = sum(role_totals.values()) * int(query["states_per_trajectory"])
    maximum_rows = state_count * (1 + 16 + math.comb(16, 2))
    expected_ceiling = {
        "trajectory_count": 1200,
        "query_state_count": 2400,
        "maximum_candidate_count": 16,
        "maximum_raw_label_rows": maximum_rows,
        "canonical_action_generations": 2 * state_count,
        "reference_teacher_forwards": state_count,
        "identical_reference_repeat_teacher_forwards": state_count,
        "maximum_capped_candidate_teacher_forwards": maximum_rows,
        "maximum_total_teacher_forwards": 2 * state_count + maximum_rows,
        "maximum_kl_measurements": state_count + maximum_rows,
        "maximum_total_model_operations": 4 * state_count + maximum_rows,
        "worker_count": 4,
    }
    if dict(ceiling) != expected_ceiling:
        raise ValueError("Freeze-B operation ceiling drifted")

    authorization = _mapping(config["authorization"], label="Freeze-B authorization")
    allowed_true = {
        "read_committed_p0_census",
        "materialize_policy_blind_roster",
        "materialize_policy_blind_query_plan",
        "write_one_git_manifest",
    }
    for key, value in authorization.items():
        if not isinstance(value, bool) or value is not (key in allowed_true):
            raise ValueError("Freeze-B authorization drifted")
    if set(authorization) != allowed_true | {
        "read_raw_source_shards",
        "select_candidates_with_processor",
        "load_policy_or_vision_model",
        "generate_restoration_labels",
        "train_predictor",
        "access_one_shot_evaluation_labels",
        "run_closed_loop",
        "access_sealed_test",
        "mutate_hugging_face",
    }:
        raise ValueError("Freeze-B authorization key set drifted")

    return {
        "status": "VALID_SET_UTILITY_FREEZE_B_V1_SOURCE",
        "config_sha256": sha256_canonical_json(config),
        "trajectory_count": sum(role_totals.values()),
        "query_state_count": state_count,
        "maximum_raw_label_rows": maximum_rows,
        "training_authorized": False,
        "label_generation_authorized": False,
    }


def _selection_digest(salt: str, purpose: str, *parts: Any) -> str:
    return sha256_canonical_json(
        {"salt": salt, "purpose": purpose, "parts": list(parts)}
    )


def _historical_firewall(census: Mapping[str, Any]) -> dict[str, set[str]]:
    audit = _mapping(census.get("consumed_group_audit"), label="consumed group audit")
    records = _sequence(audit.get("records"), label="consumed group records")
    result = {
        "legacy_sources": set(),
        "forbidden_sources": set(),
        "legacy_groups": set(),
        "forbidden_groups": set(),
    }
    for record_value in records:
        record = _mapping(record_value, label="consumed group record")
        source_id = str(record["source_id"])
        group = str(record["instruction_app_group_sha256"])
        partition = record["partition"]
        if partition == "legacy_train_only":
            result["legacy_sources"].add(source_id)
            result["legacy_groups"].add(group)
        elif partition == "forbidden_consumed":
            result["forbidden_sources"].add(source_id)
            result["forbidden_groups"].add(group)
        else:
            raise ValueError("consumed group partition drifted")
    if len(result["legacy_sources"]) != 58 or len(result["forbidden_sources"]) != 49:
        raise ValueError("historical consumed source denominator drifted")
    return result


def _query_record(assignment: Mapping[str, Any], *, kind: str, step: int) -> dict[str, Any]:
    candidate_ids = tuple(range(1, step - 1))[-MAXIMUM_CANDIDATE_COUNT:]
    if len(candidate_ids) < 4:
        raise ValueError("Freeze-B query must have at least four initial candidates")
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


def derive_freeze_b_manifest(
    config: Mapping[str, Any],
    census: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    validation = validate_freeze_b_config(config, repository_root=repository_root)
    if census.get("status") != EXPECTED_P0_STATUS:
        raise ValueError("P0 census status drifted")
    source = _mapping(config["source"], label="Freeze-B source")
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
    selection = _mapping(config["selection"], label="Freeze-B selection")
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

    representatives: list[Mapping[str, Any]] = []
    for group, records in by_group.items():
        representatives.append(
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
        )

    eligible_by_stratum: dict[str, list[Mapping[str, Any]]] = {
        stratum: [] for stratum in STRATA
    }
    too_short_for_two_states = Counter()
    for record in representatives:
        stratum = str(record["candidate_capacity_stratum"])
        if stratum not in STRATA:
            raise ValueError("P0 candidate capacity stratum drifted")
        if int(record["decision_count"]) < STRATUM_MINIMUM_FOR_TWO_STATES[stratum]:
            too_short_for_two_states[stratum] += 1
            continue
        eligible_by_stratum[stratum].append(record)

    quotas = _mapping(selection["role_stratum_trajectory_counts"], label="role quotas")
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
                raise ValueError(f"insufficient Freeze-B capacity for {role}/{stratum}")
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
        terminal = int(assignment["decision_count"])
        if anchor >= terminal:
            raise RuntimeError("Freeze-B query steps are not distinct")
        query_states.append(_query_record(assignment, kind="stratum_anchor", step=anchor))
        query_states.append(_query_record(assignment, kind="terminal", step=terminal))
    query_states.sort(key=lambda item: item["state_id"])
    if len({item["state_id"] for item in query_states}) != len(query_states):
        raise RuntimeError("Freeze-B query state ids are not unique")

    role_counts = Counter(item["role"] for item in assignments)
    stratum_counts = Counter(item["candidate_capacity_stratum"] for item in assignments)
    query_role_counts = Counter(item["role"] for item in query_states)
    query_n_counts = Counter(item["initial_candidate_count"] for item in query_states)
    pilot_states = []
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
        raise RuntimeError("Freeze-B throughput pilot roster drifted")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": STATUS,
        "source": {
            "p0_census_manifest_path": source["p0_census_manifest_path"],
            "p0_census_manifest_sha256": source["p0_census_manifest_sha256"],
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
            "too_short_for_two_states_by_stratum": dict(
                sorted(too_short_for_two_states.items())
            ),
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
            "contains_raw_instruction": False,
            "processor_candidate_freeze_status": "PENDING_SEPARATE_EXECUTION",
            "exact_operation_budget_status": "PENDING_PROCESSOR_CANDIDATE_FREEZE",
        },
        "split_audit": validated.audit.to_payload(),
        "throughput_pilot": {
            "role": "train_only",
            "state_count": 12,
            "state_ids": pilot_states,
            "may_write_utility_or_kl": False,
            "may_change_roster_or_training_grid": False,
        },
        "candidate_freeze_contract": config["candidate_freeze"],
        "teacher": config["teacher"],
        "operation_ceiling": config["operation_ceiling"],
        "features": config["features"],
        "training": config["training"],
        "evaluation": config["evaluation"],
        "artifacts": config["artifacts"],
        "assignments": assignments,
        "query_states": query_states,
        "operation_counts": {
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
        raise RuntimeError("Freeze-B output denominator drifted")
    return manifest


def materialize_freeze_b_manifest(
    *,
    repository_root: str | Path,
    config_relative_path: str,
    output_relative_path: str,
) -> tuple[dict[str, Any], str]:
    root = Path(repository_root).resolve()
    config = load_freeze_b_config(root, config_relative_path)
    validate_freeze_b_config(config, repository_root=root)
    source = _mapping(config["source"], label="Freeze-B source")
    census_path = _regular_file(
        root,
        _safe_relative_path(
            source["p0_census_manifest_path"],
            label="P0 census manifest path",
        ),
        label="P0 census manifest",
    )
    census = _strict_json_object(census_path.read_bytes(), label="P0 census manifest")
    manifest = derive_freeze_b_manifest(config, census, repository_root=root)
    output_relative = _safe_relative_path(
        output_relative_path,
        label="Freeze-B output path",
    )
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
        raise RuntimeError("Freeze-B output readback differs from written bytes")
    return manifest, sha256_bytes(payload)


__all__ = [
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "STATUS",
    "canonical_json_bytes",
    "canonical_pretty_json_bytes",
    "derive_freeze_b_manifest",
    "load_freeze_b_config",
    "materialize_freeze_b_manifest",
    "sha256_bytes",
    "sha256_canonical_json",
    "validate_freeze_b_config",
]
