#!/usr/bin/env python3
"""Aggregate the exact six D1b state-process terminals once."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.set_utility_action_stability_contract_v1 import (
    canonical_pretty_json_bytes,
)
from causalcache.set_utility_action_stability_diagnostic_v1 import (
    _validate_metric_safe_tree,
)
from causalcache.set_utility_action_stability_diagnostic_v2 import (
    EXPECTED_ENCODE_CALL_CEILING,
    EXPECTED_GENERATION_CALL_CEILING,
    PROTOCOL_ID as DIAGNOSTIC_PROTOCOL_ID,
    STATE_IDS,
    STATE_WAVES,
    aggregate_action_stability_diagnostic_v2,
    build_sdpa_execution_failure_partial_v2,
    merge_action_stability_state_v2,
)
from causalcache.set_utility_action_stability_envelope_v2 import (
    HISTORICAL_D1_AGGREGATE_SHA256,
    HISTORICAL_D1_RESULT_COMMIT,
    HISTORICAL_D1_STATUS,
    HISTORICAL_D1_VERDICT,
    VALIDATION_STATUS,
    canonical_run_layout,
    load_set_utility_action_stability_envelope_v2,
)
from causalcache.set_utility_action_stability_execution_v2 import (
    SCHEMA_VERSION,
    _strict_canonical,
    validate_attempt_payload_v2,
    validate_terminal_payload_v2,
)


PROTOCOL_ID = "causalcache_set_utility_action_stability_aggregate_v2"
STATUS = "COMPLETED_EXACT_SIX_PROCESS_SDPA_CONTROL_AGGREGATE_V2"
_SAFE_FAILURE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_VALID_VERDICTS = {
    "INVALID_RUNTIME_FAILURE",
    "INVALID_STABLE_CONTROL_INSTABILITY",
    "NO_GO_SDPA_CONTROL_REPEAT_INSTABILITY",
    "PASS_MEMORY_SAFE_SDPA_NUMERICAL_CONTROL_REPEAT_STABILITY",
}


def _historical_states(path: Path) -> tuple[list[Mapping[str, Any]], str]:
    payload, digest = _strict_canonical(path, label="D1b historical D1 aggregate")
    diagnostic = payload.get("diagnostic")
    if (
        digest != HISTORICAL_D1_AGGREGATE_SHA256
        or payload.get("status") != HISTORICAL_D1_STATUS
        or not isinstance(diagnostic, Mapping)
        or diagnostic.get("verdict") != HISTORICAL_D1_VERDICT
        or not isinstance(diagnostic.get("states"), list)
        or [state.get("state_id") for state in diagnostic["states"]]
        != list(STATE_IDS)
    ):
        raise ValueError("D1b historical D1 aggregate identity drifted")
    return list(diagnostic["states"]), digest


def _validate_terminal(
    terminal: Mapping[str, Any],
    *,
    state_id: str,
    envelope_sha256: str,
    source_config_sha256: str,
    source_inventory_sha256: str,
) -> dict[str, Any] | None:
    _, partial = validate_terminal_payload_v2(
        terminal,
        state_id=state_id,
        envelope_sha256=envelope_sha256,
        source_config_sha256=source_config_sha256,
        source_inventory_sha256=source_inventory_sha256,
    )
    return partial


def aggregate_action_stability_terminals_v2(
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    validation = projection.get("validation")
    if not isinstance(validation, Mapping) or validation.get("status") != VALIDATION_STATUS:
        raise PermissionError("D1b aggregate requires one validated direct-child envelope")
    if (
        validation.get("state_process_count") != len(STATE_IDS)
        or validation.get("wave_state_ids") != [list(wave) for wave in STATE_WAVES]
        or validation.get("historical_d1_aggregate_sha256")
        != HISTORICAL_D1_AGGREGATE_SHA256
        or validation.get("this_validated_envelope_authorizes_gpu_execution") is not True
    ):
        raise PermissionError("D1b aggregate validation receipt drifted")
    layout = canonical_run_layout(projection["run_root"])
    envelope_sha = str(validation["envelope_sha256"])
    config_sha = str(validation["source_config_sha256"])
    inventory_sha = str(validation["source_inventory_sha256"])
    historical, historical_sha = _historical_states(
        Path(projection["historical_d1_aggregate_path"])
    )
    historical_by_state = {state["state_id"]: state for state in historical}
    partials: dict[str, dict[str, Any] | None] = {}
    process_failures: dict[str, str] = {}
    terminal_hashes = []
    process_hashes = set()
    wave_zero_terminal_mtimes = []
    wave_one_attempt_mtimes = []
    for state_id in STATE_IDS:
        state_index = STATE_IDS.index(state_id)
        state_layout = layout["states"][state_index]
        attempt_path = Path(state_layout["attempt_path"])
        terminal_path = Path(state_layout["terminal_path"])
        attempt, attempt_sha = _strict_canonical(
            attempt_path, label=f"D1b {state_id} attempt"
        )
        terminal, terminal_sha = _strict_canonical(
            terminal_path, label=f"D1b {state_id} terminal"
        )
        validate_attempt_payload_v2(
            attempt,
            state_id=state_id,
            envelope_sha256=envelope_sha,
            source_config_sha256=config_sha,
            source_inventory_sha256=inventory_sha,
        )
        if (
            terminal.get("attempt_sha256") != attempt_sha
            or terminal.get("process_identity_sha256")
            != attempt.get("process_identity_sha256")
        ):
            raise ValueError("D1b attempt/terminal binding drifted")
        partials[state_id] = _validate_terminal(
            terminal,
            state_id=state_id,
            envelope_sha256=envelope_sha,
            source_config_sha256=config_sha,
            source_inventory_sha256=inventory_sha,
        )
        if partials[state_id] is None:
            failure = terminal.get("failure_class")
            if not isinstance(failure, str):
                raise ValueError("D1b failed terminal lost its class-only failure")
            process_failures[state_id] = failure
        process_hash = str(terminal["process_identity_sha256"])
        if process_hash in process_hashes:
            raise ValueError("D1b states did not use distinct fresh process identities")
        process_hashes.add(process_hash)
        terminal_hashes.append(
            {
                "process_identity_sha256": process_hash,
                "sha256": terminal_sha,
                "state_id": state_id,
            }
        )
        if state_id in STATE_WAVES[0]:
            wave_zero_terminal_mtimes.append(terminal_path.stat().st_mtime_ns)
        else:
            wave_one_attempt_mtimes.append(attempt_path.stat().st_mtime_ns)
    if max(wave_zero_terminal_mtimes) > min(wave_one_attempt_mtimes):
        raise ValueError("D1b wave two began before all wave-one terminals completed")
    merged = [
        merge_action_stability_state_v2(
            historical_by_state[state_id],
            (
                partials[state_id]
                if partials[state_id] is not None
                else build_sdpa_execution_failure_partial_v2(
                    state_id, failure_class=process_failures[state_id]
                )
            ),
        )
        for state_id in STATE_IDS
    ]
    diagnostic = aggregate_action_stability_diagnostic_v2(merged)
    verdict = diagnostic.get("verdict")
    counts = diagnostic.get("counts")
    if (
        diagnostic.get("protocol_id") != DIAGNOSTIC_PROTOCOL_ID
        or verdict not in _VALID_VERDICTS
        or not isinstance(counts, Mapping)
        or counts.get("generation_call_count") > EXPECTED_GENERATION_CALL_CEILING
        or counts.get("encode_call_count") > EXPECTED_ENCODE_CALL_CEILING
        or counts.get("retry_count") != 0
        or (
            verdict != "INVALID_RUNTIME_FAILURE"
            and (
                counts.get("generation_call_count")
                != EXPECTED_GENERATION_CALL_CEILING
                or counts.get("encode_call_count") != EXPECTED_ENCODE_CALL_CEILING
            )
        )
    ):
        raise ValueError("D1b diagnostic verdict or operation counts drifted")
    payload = {
        "counts": {
            "encode_call_ceiling": EXPECTED_ENCODE_CALL_CEILING,
            "encode_call_count": counts["encode_call_count"],
            "generation_call_ceiling": EXPECTED_GENERATION_CALL_CEILING,
            "generation_call_count": counts["generation_call_count"],
            "historical_context_calls_counted": False,
            "retry_count": 0,
            "state_process_count": len(STATE_IDS),
        },
        "diagnostic": diagnostic,
        "execution_envelope_sha256": envelope_sha,
        "historical_d1": {
            "aggregate_sha256": historical_sha,
            "context_condition_ids": ["auto_fresh_encode", "auto_frozen_encoded"],
            "eager_rows_consumed": False,
            "result_commit": HISTORICAL_D1_RESULT_COMMIT,
            "status": HISTORICAL_D1_STATUS,
            "verdict": HISTORICAL_D1_VERDICT,
        },
        "metric_safe": True,
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "source_config_sha256": config_sha,
        "source_inventory_sha256": inventory_sha,
        "state_terminal_hashes": terminal_hashes,
        "status": STATUS,
        "verdict": verdict,
    }
    _validate_metric_safe_tree(payload)
    return payload


def _write_exclusive(path: Path, payload: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"D1b aggregate already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError("D1b aggregate write made no progress")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-envelope", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    envelope_loader: Callable[..., Mapping[str, Any]] = (
        load_set_utility_action_stability_envelope_v2
    ),
) -> Mapping[str, Any]:
    args = _parser().parse_args(argv)
    projection = envelope_loader(
        args.execution_envelope,
        require_fresh_preflight=False,
        verify_current_environment=False,
    )
    if projection.get("envelope_path") != str(args.execution_envelope):
        raise ValueError("D1b aggregate envelope projection drifted")
    aggregate = aggregate_action_stability_terminals_v2(projection)
    output = Path(canonical_run_layout(projection["run_root"])["aggregate_result_path"])
    _write_exclusive(output, canonical_pretty_json_bytes(aggregate))
    print(json.dumps(aggregate, allow_nan=False, sort_keys=True))
    return aggregate


def cli() -> None:
    try:
        main()
    except Exception as error:
        name = error.__class__.__name__
        failure = name if _SAFE_FAILURE.fullmatch(name) is not None else "UnexpectedException"
        print(
            json.dumps(
                {"failure_class": failure, "status": "FAILED_D1B_AGGREGATE_CLI_V2"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
