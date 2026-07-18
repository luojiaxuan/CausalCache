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


REPAIR_STATUS = "COMPLETED_SET_CONDITIONED_V3_FRESH16_PARSER_REPAIR_V1"
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


__all__ = [
    "HISTORICAL_DECISION_CONTRACT",
    "HISTORICAL_STATUS",
    "REPAIR_STATUS",
    "evaluate_parser_repair",
    "load_historical_v1_independent_parser_repair_v1",
    "read_parser_repair_report",
]
