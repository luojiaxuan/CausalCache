#!/usr/bin/env python3
"""Materialize the label-blind variable-history state and evaluation inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from causalcache.set_utility_variable_history import (
    select_evaluation_tracks,
    state_count_summary,
    states_from_assignments,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = _read_json(args.config)
    assignment_payload = _read_json(args.assignments)
    states = states_from_assignments(assignment_payload["assignments"])
    summary = state_count_summary(states)
    expected_by_role = dict(config["source"]["state_counts"])
    expected_by_role.pop("total")
    if summary["state_count_by_role"] != expected_by_role:
        raise ValueError("state role counts differ from the frozen contract")
    if summary["state_count"] != config["source"]["state_counts"]["total"]:
        raise ValueError("total state count differs from the frozen contract")
    if (
        summary["state_count_by_role_and_history_bin"]
        != config["state"]["state_counts_by_role_and_history_bin"]
    ):
        raise ValueError("state history-bin counts differ from the frozen contract")
    tracks = select_evaluation_tracks(
        states,
        exact_state_count=config["evaluation"]["exact_oracle_track"]["state_count"],
        large_history_state_count=config["evaluation"]["large_history_track"]["state_count"],
        seed=config["broad_training_labels"]["seed"],
    )
    rows = [state.to_payload() for state in states]
    identity_payload = [
        [row["state_id"], row["candidate_event_ids"]] for row in rows
    ]
    result = {
        "assignment_manifest_sha256": _sha256_file(args.assignments),
        "config_sha256": _sha256_file(args.config),
        "evaluation_tracks": {
            "exact_state_ids": list(tracks.exact_state_ids),
            "large_history_state_ids": list(tracks.large_history_state_ids),
        },
        "schema_version": "1.0.0",
        "state_materialization": {
            "candidate_event_ids": "range(1, decision_step_id)",
            "decision_steps": "range(6, decision_count + 2)",
            "implementation": "causalcache.set_utility_variable_history.states_from_assignments",
            "rows_embedded": False,
        },
        "state_identity_sha256": hashlib.sha256(
            json.dumps(identity_payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "status": "FROZEN_VARIABLE_HISTORY_STATE_INVENTORY",
        "summary": summary,
    }
    _write_atomic(args.output, _canonical_json_bytes(result))


if __name__ == "__main__":
    main()
