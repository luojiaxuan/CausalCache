#!/usr/bin/env python3
"""Aggregate the exact eight D1 profile-worker terminals once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.set_utility_action_stability_diagnostic_v1 import (
    PROTOCOL_ID as DIAGNOSTIC_PROTOCOL_ID,
    STATE_IDS,
    _validate_metric_safe_tree,
    aggregate_action_stability_diagnostic_v1,
    merge_action_stability_state_v1,
    state_ids_for_worker_v1,
)
from causalcache.set_utility_action_stability_envelope_v1 import (
    PARENT_EXECUTION_ENVELOPE_SHA256,
    PROFILE_ORDER,
    VALIDATION_STATUS,
    canonical_run_layout,
    load_set_utility_action_stability_envelope_v1,
)
from causalcache.set_utility_action_stability_execution_v1 import (
    TERMINAL_PROTOCOL_ID,
    _validate_partial_result,
)
from causalcache.set_utility_action_stability_contract_v1 import (
    ENCODE_CALL_CEILING,
    GENERATION_CALL_CEILING,
    canonical_pretty_json_bytes,
    sha256_bytes,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_action_stability_aggregate_v1"
STATUS = "COMPLETED_EXACT_SIX_STATE_ACTION_STABILITY_AGGREGATE_V1"
_SAFE_FAILURE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_TERMINAL_KEYS = {
    "attempt_sha256",
    "counts",
    "failure_class",
    "fresh_execution_envelope_sha256",
    "metric_safe",
    "parent_execution_envelope_sha256",
    "partial_results",
    "profile",
    "protocol_id",
    "runtime_identity_sha256",
    "schema_version",
    "source_config_sha256",
    "source_inventory_sha256",
    "state_ids",
    "status",
    "worker_index",
}


def _strict_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an absolute regular file")
    payload = path.read_bytes()

    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, nested in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate key")
            value[key] = nested
        return value

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError(f"{label} contains non-finite value {raw}")
        ),
    )
    if not isinstance(value, dict) or payload != canonical_pretty_json_bytes(value):
        raise ValueError(f"{label} must be canonical pretty JSON")
    return value, hashlib.sha256(payload).hexdigest()


def _validate_terminal(
    terminal: Mapping[str, Any],
    *,
    profile: str,
    worker_index: int,
    envelope_sha256: str,
    source_config_sha256: str,
    source_inventory_sha256: str,
) -> list[dict[str, Any]]:
    if set(terminal) != _TERMINAL_KEYS:
        raise ValueError("D1 terminal schema drifted")
    expected_states = list(state_ids_for_worker_v1(worker_index))
    if (
        terminal.get("protocol_id") != TERMINAL_PROTOCOL_ID
        or terminal.get("schema_version") != SCHEMA_VERSION
        or terminal.get("status") != "COMPLETED_ACTION_STABILITY_PROFILE_WORKER_V1"
        or terminal.get("failure_class") is not None
        or terminal.get("metric_safe") is not True
        or terminal.get("profile") != profile
        or terminal.get("worker_index") != worker_index
        or terminal.get("state_ids") != expected_states
        or terminal.get("fresh_execution_envelope_sha256") != envelope_sha256
        or terminal.get("source_config_sha256") != source_config_sha256
        or terminal.get("source_inventory_sha256") != source_inventory_sha256
        or terminal.get("parent_execution_envelope_sha256")
        != PARENT_EXECUTION_ENVELOPE_SHA256
        or not isinstance(terminal.get("runtime_identity_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", terminal["runtime_identity_sha256"])
        is None
    ):
        raise ValueError("D1 terminal identity or completion drifted")
    partials = terminal.get("partial_results")
    if not isinstance(partials, list) or [item.get("state_id") for item in partials] != expected_states:
        raise ValueError("D1 terminal partial roster drifted")
    validated_partials = [
        _validate_partial_result(raw, profile=profile, state_id=state_id)
        for raw, state_id in zip(partials, expected_states, strict=True)
    ]
    expected_conditions_per_state = 2 if profile == "auto" else 1
    generation_calls = sum(
        int(condition["generation_call_count"])
        for partial in validated_partials
        for condition in partial["conditions"]
    )
    encode_calls = sum(
        int(condition["encode_call_count"])
        for partial in validated_partials
        for condition in partial["conditions"]
    )
    counts = terminal.get("counts")
    expected_counts = {
        "condition_completed_count": expected_conditions_per_state * len(expected_states),
        "encode_call_count": encode_calls,
        "generation_call_count": generation_calls,
        "label_count": 0,
        "partial_state_completed_count": len(expected_states),
        "partial_state_expected_count": len(expected_states),
        "restoration_distance_count": 0,
        "retry_count": 0,
        "teacher_forward_call_count": 0,
        "training_example_count": 0,
    }
    if counts != expected_counts:
        raise ValueError("D1 terminal counts drifted")
    return validated_partials


def _write_exclusive(path: Path, payload: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"D1 aggregate already exists: {path}")
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
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("D1 aggregate write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def aggregate_action_stability_terminals_v1(
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    validation = projection.get("validation")
    if not isinstance(validation, Mapping) or validation.get("status") != VALIDATION_STATUS:
        raise PermissionError("D1 aggregate requires a validated exact envelope")
    layout = canonical_run_layout(projection["run_root"])
    envelope_sha = str(validation["envelope_sha256"])
    config_sha = str(validation["source_config_sha256"])
    inventory_sha = str(validation["source_inventory_sha256"])
    partials: dict[str, dict[int, list[dict[str, Any]]]] = {
        profile: {} for profile in PROFILE_ORDER
    }
    terminal_hashes = []
    auto_terminal_mtimes = []
    eager_attempt_mtimes = []
    for profile in PROFILE_ORDER:
        for worker_index in range(4):
            paths = layout["workers"][worker_index]["profiles"][profile]
            attempt_path = Path(paths["attempt_path"])
            terminal_path = Path(paths["terminal_path"])
            attempt, attempt_sha = _strict_canonical(
                attempt_path, label=f"D1 {profile} worker attempt"
            )
            terminal, terminal_sha = _strict_canonical(
                terminal_path, label=f"D1 {profile} worker terminal"
            )
            if terminal.get("attempt_sha256") != attempt_sha:
                raise ValueError("D1 terminal attempt binding drifted")
            partials[profile][worker_index] = _validate_terminal(
                terminal,
                profile=profile,
                worker_index=worker_index,
                envelope_sha256=envelope_sha,
                source_config_sha256=config_sha,
                source_inventory_sha256=inventory_sha,
            )
            terminal_hashes.append(
                {"profile": profile, "sha256": terminal_sha, "worker_index": worker_index}
            )
            if profile == "auto":
                auto_terminal_mtimes.append(terminal_path.stat().st_mtime_ns)
            else:
                eager_attempt_mtimes.append(attempt_path.stat().st_mtime_ns)
    if max(auto_terminal_mtimes) > min(eager_attempt_mtimes):
        raise ValueError("D1 eager stage began before all auto terminals completed")
    by_state: dict[str, dict[str, dict[str, Any]]] = {
        state_id: {} for state_id in STATE_IDS
    }
    for profile in PROFILE_ORDER:
        for worker_partials in partials[profile].values():
            for partial in worker_partials:
                state_id = partial["state_id"]
                if state_id not in by_state or profile in by_state[state_id]:
                    raise ValueError("D1 partial state/profile inventory drifted")
                by_state[state_id][profile] = partial
    merged = [
        merge_action_stability_state_v1(
            by_state[state_id]["auto"], by_state[state_id]["eager"]
        )
        for state_id in STATE_IDS
    ]
    diagnostic = aggregate_action_stability_diagnostic_v1(merged)
    actual_generation_calls = diagnostic["counts"]["generation_call_count"]
    actual_encode_calls = diagnostic["counts"]["encode_call_count"]
    if (
        diagnostic.get("protocol_id") != DIAGNOSTIC_PROTOCOL_ID
        or actual_generation_calls > GENERATION_CALL_CEILING
        or actual_encode_calls > ENCODE_CALL_CEILING
        or diagnostic["counts"]["retry_count"] != 0
    ):
        raise ValueError("D1 core aggregate operation counts drifted")
    payload = {
        "counts": {
            "encode_call_ceiling": ENCODE_CALL_CEILING,
            "encode_call_count": actual_encode_calls,
            "generation_call_ceiling": GENERATION_CALL_CEILING,
            "generation_call_count": actual_generation_calls,
            "profile_worker_terminal_count": 8,
            "retry_count": 0,
            "state_count": 6,
        },
        "diagnostic": diagnostic,
        "execution_envelope_sha256": envelope_sha,
        "metric_safe": True,
        "parent_execution_envelope_sha256": PARENT_EXECUTION_ENVELOPE_SHA256,
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "source_config_sha256": config_sha,
        "source_inventory_sha256": inventory_sha,
        "status": STATUS,
        "worker_terminal_hashes": terminal_hashes,
    }
    _validate_metric_safe_tree(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-envelope", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    envelope_loader: Callable[..., Mapping[str, Any]] = (
        load_set_utility_action_stability_envelope_v1
    ),
) -> Mapping[str, Any]:
    args = _parser().parse_args(argv)
    projection = envelope_loader(
        args.execution_envelope,
        require_fresh_preflight=False,
    )
    if projection.get("envelope_path") != str(args.execution_envelope):
        raise ValueError("D1 aggregate envelope projection drifted")
    aggregate = aggregate_action_stability_terminals_v1(projection)
    output = Path(canonical_run_layout(projection["run_root"])["aggregate_result_path"])
    _write_exclusive(output, canonical_pretty_json_bytes(aggregate))
    print(json.dumps(aggregate, allow_nan=False, sort_keys=True))
    return aggregate


def cli() -> None:
    try:
        main()
    except Exception as error:
        name = error.__class__.__name__
        failure = name if _SAFE_FAILURE.fullmatch(name) else "UnexpectedException"
        print(
            json.dumps(
                {"failure_class": failure, "status": "FAILED_D1_AGGREGATE_CLI_V1"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
