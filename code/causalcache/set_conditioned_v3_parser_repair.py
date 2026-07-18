"""Parser-only continuation for the failed v3 consumed-development join."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.gate_v1_data import DEPLOYMENT_BUDGET, FeatureState
from causalcache.gate_v1_fresh16 import FRESH_STATE_COUNT
from causalcache.gate_v1_provenance import canonical_selection_sha256
from causalcache.set_conditioned_v3_exploration import (
    FRESH_FEATURE_SHA256,
    FRESH_LABEL_SHA256,
    HISTORICAL_INDEPENDENT_SHA256,
    SEEDS,
    build_evaluation_report,
    feasible_coalitions_from_ids,
    load_fresh_features,
    load_fresh_states,
    read_prediction_artifact,
    verify_fresh_label_access_claim,
    verify_label_blind_seal,
)
from causalcache.set_conditioned_v3_parser_repair_contract import (
    EXACT_HISTORICAL_PROTOCOL_ID,
    PROTOCOL_ID,
    REPAIR_REPORT_NAME,
    ParserRepairContract,
    canonical_json_bytes,
    sha256_bytes,
)
from causalcache.set_conditioned_v3_parser_repair_v2_contract import (
    PROTOCOL_ID as PROTOCOL_ID_V2,
    REPAIR_REPORT_NAME as REPAIR_REPORT_NAME_V2,
    ParserRepairV2Contract,
)


REPAIR_STATUS = "COMPLETED_SET_CONDITIONED_V3_FRESH16_PARSER_REPAIR_V1"
REPAIR_STATUS_V2 = "COMPLETED_SET_CONDITIONED_V3_FRESH16_PARSER_REPAIR_V2"
HISTORICAL_STATUS = "FROZEN_GATE_V1_FRESH16_LEARNED_SELECTIONS_V1"
HISTORICAL_DECISION_CONTRACT = "FEATURE_ONLY_ALL_ENSEMBLE_AND_SEED_DECISIONS_V1"


def _unique_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} has duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_historical_v1_independent_parser_repair_v1(
    payload: bytes,
    *,
    feature_states: Sequence[FeatureState],
) -> Mapping[str, tuple[int, ...]]:
    """Accept only the exact protocol emitted by the historical producer."""
    if sha256_bytes(payload) != HISTORICAL_INDEPENDENT_SHA256:
        raise ValueError("historical independent SHA-256 drifted")
    value = _unique_json(payload, label="historical independent decisions")
    records = value.get("records")
    if (
        payload != canonical_json_bytes(value) + b"\n"
        or set(value)
        != {
            "schema_version",
            "protocol_id",
            "status",
            "name",
            "selection_sha256",
            "decision_contract",
            "records",
        }
        or value.get("schema_version") != "1.0.0"
        or value.get("protocol_id") != EXACT_HISTORICAL_PROTOCOL_ID
        or value.get("status") != HISTORICAL_STATUS
        or value.get("name") != "independent"
        or value.get("decision_contract") != HISTORICAL_DECISION_CONTRACT
        or not isinstance(records, list)
        or len(records) != FRESH_STATE_COUNT
        or len(feature_states) != FRESH_STATE_COUNT
    ):
        raise ValueError("historical independent parser-repair schema drifted")
    features = {state.state_id: state for state in feature_states}
    if len(features) != FRESH_STATE_COUNT:
        raise ValueError("historical independent feature inventory drifted")
    selections: dict[str, tuple[int, ...]] = {}
    for feature, record in zip(feature_states, records, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "state_id",
            "ensemble_selected_event_step_ids",
            "seed_selected_event_step_ids",
        }:
            raise ValueError("historical independent state schema drifted")
        if record.get("state_id") != feature.state_id:
            raise ValueError("historical independent state order drifted")
        selected_raw = record.get("ensemble_selected_event_step_ids")
        if not isinstance(selected_raw, list) or any(
            type(event) is not int for event in selected_raw
        ):
            raise ValueError("historical independent selection is malformed")
        selected = tuple(selected_raw)
        if (
            selected != tuple(sorted(selected))
            or len(set(selected)) != len(selected)
            or len(selected) > DEPLOYMENT_BUDGET
            or not set(selected).issubset(feature.candidate_event_step_ids)
        ):
            raise ValueError("historical independent selection is infeasible")
        seed_rows = record.get("seed_selected_event_step_ids")
        if not isinstance(seed_rows, list) or len(seed_rows) != len(SEEDS):
            raise ValueError("historical independent seed inventory drifted")
        feasible = set(feasible_coalitions_from_ids(feature.candidate_event_step_ids))
        for seed, seed_record in zip(SEEDS, seed_rows, strict=True):
            if (
                not isinstance(seed_record, Mapping)
                or set(seed_record) != {"seed", "selected_event_step_ids"}
                or seed_record.get("seed") != seed
                or not isinstance(seed_record.get("selected_event_step_ids"), list)
                or tuple(seed_record["selected_event_step_ids"]) not in feasible
            ):
                raise ValueError("historical independent seed record drifted")
        selections[feature.state_id] = selected
    if value.get("selection_sha256") != canonical_selection_sha256(selections):
        raise ValueError("historical independent selection digest drifted")
    return selections


def load_historical_v1_independent_parser_repair_v2(
    payload: bytes,
    *,
    feature_states: Sequence[FeatureState],
) -> Mapping[str, tuple[int, ...]]:
    """Join the exact historical roster by unique state id, not file order."""
    if sha256_bytes(payload) != HISTORICAL_INDEPENDENT_SHA256:
        raise ValueError("historical independent SHA-256 drifted")
    value = _unique_json(payload, label="historical independent decisions")
    records = value.get("records")
    if (
        payload != canonical_json_bytes(value) + b"\n"
        or set(value)
        != {
            "schema_version",
            "protocol_id",
            "status",
            "name",
            "selection_sha256",
            "decision_contract",
            "records",
        }
        or value.get("schema_version") != "1.0.0"
        or value.get("protocol_id") != EXACT_HISTORICAL_PROTOCOL_ID
        or value.get("status") != HISTORICAL_STATUS
        or value.get("name") != "independent"
        or value.get("decision_contract") != HISTORICAL_DECISION_CONTRACT
        or not isinstance(records, list)
        or len(records) != FRESH_STATE_COUNT
        or len(feature_states) != FRESH_STATE_COUNT
    ):
        raise ValueError("historical independent parser-repair v2 schema drifted")
    features = {state.state_id: state for state in feature_states}
    if len(features) != FRESH_STATE_COUNT:
        raise ValueError("historical independent feature inventory drifted")
    records_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("historical independent state row is malformed")
        state_id = raw.get("state_id")
        if not isinstance(state_id, str) or state_id in records_by_id:
            raise ValueError("historical independent state id is malformed or duplicated")
        records_by_id[state_id] = raw
    if set(records_by_id) != set(features):
        raise ValueError("historical and feature state-id sets differ")
    selections: dict[str, tuple[int, ...]] = {}
    for feature in feature_states:
        record = records_by_id[feature.state_id]
        if set(record) != {
            "state_id",
            "ensemble_selected_event_step_ids",
            "seed_selected_event_step_ids",
        }:
            raise ValueError("historical independent state schema drifted")
        selected_raw = record.get("ensemble_selected_event_step_ids")
        if not isinstance(selected_raw, list) or any(
            type(event) is not int for event in selected_raw
        ):
            raise ValueError("historical independent selection is malformed")
        selected = tuple(selected_raw)
        if (
            selected != tuple(sorted(selected))
            or len(set(selected)) != len(selected)
            or len(selected) > DEPLOYMENT_BUDGET
            or not set(selected).issubset(feature.candidate_event_step_ids)
        ):
            raise ValueError("historical independent selection is infeasible")
        seed_rows = record.get("seed_selected_event_step_ids")
        if not isinstance(seed_rows, list) or len(seed_rows) != len(SEEDS):
            raise ValueError("historical independent seed inventory drifted")
        feasible = set(feasible_coalitions_from_ids(feature.candidate_event_step_ids))
        for seed, seed_record in zip(SEEDS, seed_rows, strict=True):
            seed_selection = (
                seed_record.get("selected_event_step_ids")
                if isinstance(seed_record, Mapping)
                else None
            )
            if (
                not isinstance(seed_record, Mapping)
                or set(seed_record) != {"seed", "selected_event_step_ids"}
                or type(seed_record.get("seed")) is not int
                or seed_record.get("seed") != seed
                or not isinstance(seed_selection, list)
                or any(type(event) is not int for event in seed_selection)
                or tuple(seed_selection) not in feasible
            ):
                raise ValueError("historical independent seed record drifted")
        selections[feature.state_id] = selected
    if value.get("selection_sha256") != canonical_selection_sha256(selections):
        raise ValueError("historical independent selection digest drifted")
    return selections


def _write_exclusive(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("parser-repair report must not preexist")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("parser-repair report write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def evaluate_parser_repair(
    *,
    contract: ParserRepairContract,
    output_dir: Path,
    feature_payload: bytes,
    label_payload: bytes,
    historical_independent_payload: bytes,
    repair_source_a_git_commit: str,
    repair_execution_b_git_commit: str,
    repair_runner_freeze_sha256: str,
) -> Mapping[str, Any]:
    seal = verify_label_blind_seal(output_dir)
    claim = verify_fresh_label_access_claim(output_dir)
    features = load_fresh_features(feature_payload)
    predictions = read_prediction_artifact(
        (output_dir / "fresh16-predictions.json").read_bytes(),
        features=features,
    )
    historical = load_historical_v1_independent_parser_repair_v1(
        historical_independent_payload,
        feature_states=features,
    )
    # note (luojiaxuan): Validate the repaired historical parser completely
    # before consuming the one authorized repair label replay.
    states = load_fresh_states(feature_payload, label_payload)
    base = dict(
        build_evaluation_report(
            states,
            prediction_records=predictions,
            historical_v1_independent=historical,
            label_blind_seal_sha256=sha256_bytes(
                (output_dir / "label-blind-seal.json").read_bytes()
            ),
            label_access_claim_sha256=sha256_bytes(
                (output_dir / "fresh-label-access-claim.json").read_bytes()
            ),
        )
    )
    base.pop("report_sha256")
    base.pop("fresh_label_decode_count")
    base["protocol_id"] = PROTOCOL_ID
    base["status"] = REPAIR_STATUS
    base["scope"] = (
        "consumed_development_parser_replay_not_pristine_not_confirmatory"
    )
    base["repair_provenance"] = {
        "parent_source_a_git_commit": seal["source_a_git_commit"],
        "parent_execution_b_git_commit": seal["execution_b_git_commit"],
        "parent_runner_freeze_sha256": seal["runner_freeze_sha256"],
        "repair_source_a_git_commit": repair_source_a_git_commit,
        "repair_execution_b_git_commit": repair_execution_b_git_commit,
        "repair_runner_freeze_sha256": repair_runner_freeze_sha256,
        "repair_contract_sha256": contract.sha256,
        "failed_expected_historical_protocol_id": (
            "causalcache_gate_v1_fresh16_primary_v1"
        ),
        "exact_historical_producer_protocol_id": EXACT_HISTORICAL_PROTOCOL_ID,
        "only_parser_binding_changed": True,
        "checkpoints_predictions_selector_thresholds_and_bootstrap_unchanged": True,
        "parent_development_report_created": False,
    }
    base["access_accounting"] = dict(contract.data["access_accounting"])
    base["fresh_label_access_claim_count"] = int(
        contract.data["access_accounting"]["fresh_label_access_claim_count_total"]
    )
    base["development_report_completion_count"] = 1
    base["may_authorize_confirm"] = False
    base["may_change_v1_verdict"] = False
    if claim.get("fresh_label_access_claim_count") != 1:
        raise ValueError("parent label-access claim count drifted")
    report_sha256 = sha256_bytes(canonical_json_bytes(base))
    report = {**base, "report_sha256": report_sha256}
    payload = canonical_json_bytes(report) + b"\n"
    _write_exclusive(output_dir / REPAIR_REPORT_NAME, payload)
    return report


def read_parser_repair_report(output_dir: Path) -> Mapping[str, Any]:
    path = output_dir / REPAIR_REPORT_NAME
    if path.is_symlink() or not path.is_file():
        raise ValueError("parser-repair report is missing or unsafe")
    payload = path.read_bytes()
    report = _unique_json(payload, label="parser-repair report")
    if (
        payload != canonical_json_bytes(report) + b"\n"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("status") != REPAIR_STATUS
        or report.get("fresh_label_access_claim_count") != 1
        or report.get("access_accounting", {}).get(
            "fresh_label_semantic_decode_attempt_count_total"
        )
        != 2
        or report.get("confirm_access_count") != 0
    ):
        raise ValueError("parser-repair report schema drifted")
    without_hash = dict(report)
    observed_hash = without_hash.pop("report_sha256", None)
    if observed_hash != sha256_bytes(canonical_json_bytes(without_hash)):
        raise ValueError("parser-repair report digest drifted")
    return report


def evaluate_parser_repair_v2(
    *,
    contract: ParserRepairV2Contract,
    output_dir: Path,
    feature_payload: bytes,
    label_payload: bytes,
    historical_independent_payload: bytes,
    repair_source_a_git_commit: str,
    repair_execution_b_git_commit: str,
    repair_runner_freeze_sha256: str,
) -> Mapping[str, Any]:
    seal = verify_label_blind_seal(output_dir)
    claim = verify_fresh_label_access_claim(output_dir)
    features = load_fresh_features(feature_payload)
    predictions = read_prediction_artifact(
        (output_dir / "fresh16-predictions.json").read_bytes(),
        features=features,
    )
    historical = load_historical_v1_independent_parser_repair_v2(
        historical_independent_payload,
        feature_states=features,
    )
    # note (luojiaxuan): The exact state-id join is fully validated before the
    # only v2 repair label replay begins.
    states = load_fresh_states(feature_payload, label_payload)
    base = dict(
        build_evaluation_report(
            states,
            prediction_records=predictions,
            historical_v1_independent=historical,
            label_blind_seal_sha256=sha256_bytes(
                (output_dir / "label-blind-seal.json").read_bytes()
            ),
            label_access_claim_sha256=sha256_bytes(
                (output_dir / "fresh-label-access-claim.json").read_bytes()
            ),
        )
    )
    base.pop("report_sha256")
    base.pop("fresh_label_decode_count")
    base["protocol_id"] = PROTOCOL_ID_V2
    base["status"] = REPAIR_STATUS_V2
    base["scope"] = (
        "consumed_development_state_id_join_replay_not_pristine_not_confirmatory"
    )
    base["repair_provenance"] = {
        "parent_source_a_git_commit": seal["source_a_git_commit"],
        "parent_execution_b_git_commit": seal["execution_b_git_commit"],
        "parent_runner_freeze_sha256": seal["runner_freeze_sha256"],
        "parser_repair_v1_execution_b_git_commit": (
            "f53954f9e2c142738c77ffa150f8ddc8c7088ff2"
        ),
        "parser_repair_v1_label_decode_attempt_count": 0,
        "repair_source_a_git_commit": repair_source_a_git_commit,
        "repair_execution_b_git_commit": repair_execution_b_git_commit,
        "repair_runner_freeze_sha256": repair_runner_freeze_sha256,
        "repair_contract_sha256": contract.sha256,
        "exact_historical_producer_protocol_id": EXACT_HISTORICAL_PROTOCOL_ID,
        "exact_state_id_set_join": True,
        "positional_order_required": False,
        "checkpoints_predictions_selector_thresholds_and_bootstrap_unchanged": True,
        "parent_and_v1_development_reports_created": False,
    }
    base["access_accounting"] = dict(contract.data["access_accounting"])
    base["fresh_label_access_claim_count"] = 1
    base["development_report_completion_count"] = 1
    base["may_authorize_confirm"] = False
    base["may_change_v1_verdict"] = False
    if claim.get("fresh_label_access_claim_count") != 1:
        raise ValueError("parent label-access claim count drifted")
    report_sha256 = sha256_bytes(canonical_json_bytes(base))
    report = {**base, "report_sha256": report_sha256}
    _write_exclusive(
        output_dir / REPAIR_REPORT_NAME_V2,
        canonical_json_bytes(report) + b"\n",
    )
    return report


def read_parser_repair_v2_report(
    output_dir: Path,
    *,
    contract: ParserRepairV2Contract,
    expected_report_sha256: str,
    expected_source_a_git_commit: str,
    expected_execution_b_git_commit: str,
    expected_runner_freeze_sha256: str,
) -> Mapping[str, Any]:
    path = output_dir / REPAIR_REPORT_NAME_V2
    if path.is_symlink() or not path.is_file():
        raise ValueError("parser-repair v2 report is missing or unsafe")
    payload = path.read_bytes()
    report = _unique_json(payload, label="parser-repair v2 report")
    accounting = report.get("access_accounting")
    provenance = report.get("repair_provenance")
    interpretation = report.get("development_interpretation")
    seal = verify_label_blind_seal(output_dir)
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "scope",
        "label_blind_seal_sha256",
        "label_access_claim_sha256",
        "fresh_feature_sha256",
        "fresh_label_sha256",
        "historical_v1_independent_sha256",
        "trajectory_count",
        "state_count",
        "metrics",
        "bootstrap",
        "switch_diagnostics",
        "development_interpretation",
        "state_records",
        "fresh_label_access_claim_count",
        "confirm_access_count",
        "legacy_development_access_count",
        "matched_nll_evaluation_count",
        "closed_loop_episode_count",
        "raw_gui_access_count",
        "policy_forward_count",
        "gpu_operation_count",
        "repair_provenance",
        "access_accounting",
        "development_report_completion_count",
        "may_authorize_confirm",
        "may_change_v1_verdict",
        "report_sha256",
    }
    expected_provenance_keys = {
        "parent_source_a_git_commit",
        "parent_execution_b_git_commit",
        "parent_runner_freeze_sha256",
        "parser_repair_v1_execution_b_git_commit",
        "parser_repair_v1_label_decode_attempt_count",
        "repair_source_a_git_commit",
        "repair_execution_b_git_commit",
        "repair_runner_freeze_sha256",
        "repair_contract_sha256",
        "exact_historical_producer_protocol_id",
        "exact_state_id_set_join",
        "positional_order_required",
        "checkpoints_predictions_selector_thresholds_and_bootstrap_unchanged",
        "parent_and_v1_development_reports_created",
    }
    if (
        payload != canonical_json_bytes(report) + b"\n"
        or set(report) != expected_keys
        or report.get("schema_version") != "1.0.0"
        or report.get("protocol_id") != PROTOCOL_ID_V2
        or report.get("status") != REPAIR_STATUS_V2
        or report.get("scope")
        != "consumed_development_state_id_join_replay_not_pristine_not_confirmatory"
        or not isinstance(accounting, Mapping)
        or dict(accounting) != dict(contract.data["access_accounting"])
        or not isinstance(provenance, Mapping)
        or set(provenance) != expected_provenance_keys
        or provenance.get("parent_source_a_git_commit")
        != seal["source_a_git_commit"]
        or provenance.get("parent_execution_b_git_commit")
        != seal["execution_b_git_commit"]
        or provenance.get("parent_runner_freeze_sha256")
        != seal["runner_freeze_sha256"]
        or provenance.get("parser_repair_v1_execution_b_git_commit")
        != "f53954f9e2c142738c77ffa150f8ddc8c7088ff2"
        or provenance.get("parser_repair_v1_label_decode_attempt_count") != 0
        or provenance.get("repair_contract_sha256") != contract.sha256
        or provenance.get("repair_source_a_git_commit")
        != expected_source_a_git_commit
        or provenance.get("repair_execution_b_git_commit")
        != expected_execution_b_git_commit
        or provenance.get("repair_runner_freeze_sha256")
        != expected_runner_freeze_sha256
        or provenance.get("exact_historical_producer_protocol_id")
        != EXACT_HISTORICAL_PROTOCOL_ID
        or provenance.get("exact_state_id_set_join") is not True
        or provenance.get("positional_order_required") is not False
        or provenance.get(
            "checkpoints_predictions_selector_thresholds_and_bootstrap_unchanged"
        )
        is not True
        or provenance.get("parent_and_v1_development_reports_created") is not False
        or not isinstance(interpretation, Mapping)
        or set(interpretation)
        != {
            "value",
            "primary_contrast",
            "mean_normalized_delta",
            "mean_raw_delta",
            "n4_mean_normalized_delta",
            "checks",
            "may_authorize_confirm",
            "may_change_v1_verdict",
        }
        or interpretation.get("may_authorize_confirm") is not False
        or interpretation.get("may_change_v1_verdict") is not False
        or interpretation.get("primary_contrast") != "v3_safe_minus_v3_additive"
        or report.get("label_blind_seal_sha256")
        != sha256_bytes((output_dir / "label-blind-seal.json").read_bytes())
        or report.get("label_access_claim_sha256")
        != sha256_bytes((output_dir / "fresh-label-access-claim.json").read_bytes())
        or report.get("fresh_feature_sha256") != FRESH_FEATURE_SHA256
        or report.get("fresh_label_sha256") != FRESH_LABEL_SHA256
        or report.get("historical_v1_independent_sha256")
        != HISTORICAL_INDEPENDENT_SHA256
        or report.get("trajectory_count") != 16
        or report.get("state_count") != FRESH_STATE_COUNT
        or not isinstance(report.get("metrics"), Mapping)
        or not isinstance(report.get("bootstrap"), Mapping)
        or not isinstance(report.get("switch_diagnostics"), Mapping)
        or not isinstance(report.get("state_records"), list)
        or len(report["state_records"]) != FRESH_STATE_COUNT
        or report.get("fresh_label_access_claim_count") != 1
        or report.get("development_report_completion_count") != 1
        or report.get("confirm_access_count") != 0
        or report.get("legacy_development_access_count") != 0
        or report.get("matched_nll_evaluation_count") != 0
        or report.get("closed_loop_episode_count") != 0
        or report.get("raw_gui_access_count") != 0
        or report.get("policy_forward_count") != 0
        or report.get("gpu_operation_count") != 0
        or report.get("may_authorize_confirm") is not False
        or report.get("may_change_v1_verdict") is not False
        or "fresh_label_decode_count" in report
    ):
        raise ValueError("parser-repair v2 report schema drifted")
    without_hash = dict(report)
    observed = without_hash.pop("report_sha256", None)
    if (
        observed != expected_report_sha256
        or observed != sha256_bytes(canonical_json_bytes(without_hash))
    ):
        raise ValueError("parser-repair v2 report digest drifted")
    return report


__all__ = [
    "HISTORICAL_DECISION_CONTRACT",
    "HISTORICAL_STATUS",
    "REPAIR_STATUS",
    "REPAIR_STATUS_V2",
    "evaluate_parser_repair",
    "evaluate_parser_repair_v2",
    "load_historical_v1_independent_parser_repair_v1",
    "load_historical_v1_independent_parser_repair_v2",
    "read_parser_repair_report",
    "read_parser_repair_v2_report",
]
