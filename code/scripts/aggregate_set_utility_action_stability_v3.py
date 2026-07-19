#!/usr/bin/env python3
"""Aggregate the exact three strict-determinism D2 state terminals."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causalcache.set_utility_action_stability_contract_v1 import (
    canonical_pretty_json_bytes,
    sha256_bytes,
)
from causalcache.set_utility_action_stability_contract_v3 import (
    D1B_AGGREGATE_SHA256,
    D1B_STATUS,
    D1B_VERDICT,
)
from causalcache.set_utility_action_stability_diagnostic_v1 import (
    _validate_metric_safe_tree,
)
from causalcache.set_utility_action_stability_diagnostic_v3 import (
    EXPECTED_ENCODE_CALL_CEILING,
    EXPECTED_GENERATION_CALL_CEILING,
    STATE_IDS,
    aggregate_action_stability_diagnostic_v3,
    build_strict_execution_failure_partial_v3,
    merge_action_stability_state_v3,
    validate_strict_partial_v3,
)
from causalcache.set_utility_action_stability_execution_v3 import (
    CLAIMED_STATUS,
    COMPLETED_STATUS,
    FAILED_STATUS,
    TERMINAL_PROTOCOL_ID,
    _strict_canonical,
    _write_exclusive,
    canonical_run_layout,
    load_action_stability_envelope_v3,
)


PROTOCOL_ID = "causalcache_set_utility_action_stability_aggregate_v3"
STATUS = "COMPLETED_EXACT_THREE_PROCESS_STRICT_DETERMINISM_D2_AGGREGATE"


def _parent_states(path: Path) -> tuple[dict[str, Mapping[str, Any]], str]:
    payload, digest = _strict_canonical(path, label="D1b aggregate")
    diagnostic = payload.get("diagnostic")
    states = diagnostic.get("states") if isinstance(diagnostic, Mapping) else None
    if (
        digest != D1B_AGGREGATE_SHA256
        or payload.get("status") != D1B_STATUS
        or payload.get("verdict") != D1B_VERDICT
        or not isinstance(states, list)
    ):
        raise ValueError("D2 parent D1b aggregate identity drifted")
    by_state = {state.get("state_id"): state for state in states}
    if any(state_id not in by_state for state_id in STATE_IDS):
        raise ValueError("D2 parent D1b state is missing")
    return by_state, digest


def aggregate_terminals_v3(projection: Mapping[str, Any]) -> dict[str, Any]:
    layout = canonical_run_layout(projection["run_root"])
    validation = projection["validation"]
    artifacts = projection["artifacts"]
    parents, parent_sha = _parent_states(Path(artifacts["d1b_aggregate"]["path"]))
    merged = []
    terminal_hashes = []
    process_ids = set()
    for index, state_id in enumerate(STATE_IDS):
        state_layout = layout["states"][index]
        attempt, attempt_sha = _strict_canonical(
            Path(state_layout["attempt_path"]), label=f"{state_id} attempt"
        )
        terminal, terminal_sha = _strict_canonical(
            Path(state_layout["terminal_path"]), label=f"{state_id} terminal"
        )
        if (
            attempt.get("status") != CLAIMED_STATUS
            or attempt.get("state_id") != state_id
            or attempt.get("state_index") != index
            or attempt.get("execution_envelope_sha256") != validation["envelope_sha256"]
            or attempt.get("source_config_sha256") != validation["source_config_sha256"]
            or attempt.get("source_inventory_sha256") != validation["source_inventory_sha256"]
            or attempt.get("retry_count") != 0
            or terminal.get("protocol_id") != TERMINAL_PROTOCOL_ID
            or terminal.get("status") not in {COMPLETED_STATUS, FAILED_STATUS}
            or terminal.get("attempt_sha256") != attempt_sha
            or terminal.get("state_id") != state_id
            or terminal.get("state_index") != index
            or terminal.get("process_identity_sha256") != attempt.get("process_identity_sha256")
            or terminal.get("execution_envelope_sha256") != validation["envelope_sha256"]
            or terminal.get("source_config_sha256") != validation["source_config_sha256"]
            or terminal.get("source_inventory_sha256") != validation["source_inventory_sha256"]
        ):
            raise ValueError("D2 attempt/terminal binding drifted")
        process_id = terminal["process_identity_sha256"]
        if process_id in process_ids:
            raise ValueError("D2 states did not use distinct OS processes")
        process_ids.add(process_id)
        if terminal["status"] == COMPLETED_STATUS:
            partial = validate_strict_partial_v3(terminal["partial_result"])
        else:
            failure = terminal.get("failure_class")
            if not isinstance(failure, str):
                raise ValueError("D2 failure terminal lost class-only failure")
            partial = build_strict_execution_failure_partial_v3(
                state_id, failure_class=failure
            )
        merged.append(merge_action_stability_state_v3(parents[state_id], partial))
        terminal_hashes.append({"sha256": terminal_sha, "state_id": state_id})
    diagnostic = aggregate_action_stability_diagnostic_v3(merged)
    counts = diagnostic["counts"]
    if (
        counts["encode_call_count"] > EXPECTED_ENCODE_CALL_CEILING
        or counts["generation_call_count"] > EXPECTED_GENERATION_CALL_CEILING
        or counts["retry_count"] != 0
        or (
            diagnostic["verdict"] != "INVALID_RUNTIME_FAILURE"
            and (
                counts["encode_call_count"] != EXPECTED_ENCODE_CALL_CEILING
                or counts["generation_call_count"] != EXPECTED_GENERATION_CALL_CEILING
            )
        )
    ):
        raise ValueError("D2 aggregate operation counts drifted")
    result = {
        "counts": {
            **counts,
            "encode_call_ceiling": EXPECTED_ENCODE_CALL_CEILING,
            "generation_call_ceiling": EXPECTED_GENERATION_CALL_CEILING,
        },
        "d1b_aggregate_sha256": parent_sha,
        "execution_envelope_sha256": validation["envelope_sha256"],
        "metric_safe": True,
        "protocol_id": PROTOCOL_ID,
        "schema_version": "1.0.0",
        "source_config_sha256": validation["source_config_sha256"],
        "source_inventory_sha256": validation["source_inventory_sha256"],
        "state_terminal_hashes": terminal_hashes,
        "status": STATUS,
        "diagnostic": diagnostic,
        "verdict": diagnostic["verdict"],
    }
    _validate_metric_safe_tree(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-envelope", type=Path, required=True)
    args = parser.parse_args()
    projection = load_action_stability_envelope_v3(
        args.execution_envelope,
        require_fresh_preflight=False,
        verify_current_container=False,
    )
    result = aggregate_terminals_v3(projection)
    output = Path(canonical_run_layout(projection["run_root"])["aggregate_path"])
    _write_exclusive(output, canonical_pretty_json_bytes(result))
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
