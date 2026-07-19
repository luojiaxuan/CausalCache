"""Future-envelope primitives for the D1b controlled-SDPA diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from causalcache.set_utility_action_stability_contract_v1 import (
    canonical_pretty_json_bytes,
    sha256_bytes,
)
from causalcache.set_utility_action_stability_diagnostic_v1 import STATE_IDS


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_set_utility_action_stability_execution_envelope_v2"
AUTHORIZED_STATUS = "AUTHORIZED_EXACT_SIX_PROCESS_SDPA_CONTROL_EXECUTION_V2"
VALIDATION_STATUS = "VALID_SET_UTILITY_ACTION_STABILITY_EXECUTION_ENVELOPE_V2"
PREFLIGHT_PROTOCOL_ID = "causalcache_set_utility_action_stability_preflight_v2"
PREFLIGHT_STATUS = "PASSED_FRESH_TEN_SECOND_FOUR_H200_PREFLIGHT_V2"
PREFLIGHT_MINIMUM_SECONDS = 10
PREFLIGHT_MAX_AGE_SECONDS = 15 * 60
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_action_stability_diagnostic_v2.json"
)
CANONICAL_GIT_ENVELOPE_PATH = (
    "code/configs/causalcache_set_utility_action_stability_"
    "diagnostic_v2_execution.json"
)
SOURCE_BRANCH = "main"
SOURCE_REMOTE = "origin"
CONDITION_ID = "sdpa_numerical_control_frozen_encoded"
REPEAT_COUNT = 2
GENERATION_CALL_CEILING = 12
ENCODE_CALL_CEILING = 6
STATE_PROCESS_COUNT = 6
GPU_COUNT = 4
MAX_CONCURRENT_STATE_PROCESSES = 4
PERSISTENT_DATA_ROOT = Path("/data")
STATE_WAVES = (
    (STATE_IDS[0], STATE_IDS[3], STATE_IDS[4], STATE_IDS[5]),
    (STATE_IDS[1], STATE_IDS[2]),
)
STATE_ENTRYPOINT = "code/scripts/run_set_utility_action_stability_state_v2.py"
AGGREGATE_ENTRYPOINT = "code/scripts/aggregate_set_utility_action_stability_v2.py"
HISTORICAL_D1_AGGREGATE_PATH = (
    "data/results/set_utility_action_stability_diagnostic_v1/aggregate.json"
)
HISTORICAL_D1_AGGREGATE_SHA256 = (
    "2debde6de3f552e9551d0ee37d25b82fa2ca85dfb42746d389dde39eff577ba2"
)
HISTORICAL_D1_RESULT_COMMIT = "a713555671d0c828b15d26f9652c7a889332cb57"
HISTORICAL_D1_STATUS = "COMPLETED_EXACT_SIX_STATE_ACTION_STABILITY_AGGREGATE_V1"
HISTORICAL_D1_VERDICT = "INVALID_RUNTIME_FAILURE"
HISTORICAL_D1_ENVELOPE_PATH = (
    "code/configs/causalcache_set_utility_action_stability_"
    "diagnostic_v1_execution.json"
)
HISTORICAL_D1_ENVELOPE_SHA256 = (
    "093eadeea6c05e1266c681aeea31fa48a86b6dff74d83404e115ce63c3b0a6ea"
)

_COMMIT = re.compile(r"[0-9a-f]{40}")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{16,}")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GPU_NAME = "NVIDIA H200"
_HOST = {"alias": "hyper00", "hostname": "node-radixark-16-0000"}


class SourceContractV2(Protocol):
    repository_root: Path
    config_sha256: str
    data: Mapping[str, Any]


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one mapping")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be one sequence")
    return value


def _absolute(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an absolute normalized path")
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        raise ValueError(f"{label} must be an absolute normalized path")
    return path


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, nested in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            value[key] = nested
        return value

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be whole-second UTC RFC3339")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be whole-second UTC RFC3339") from error
    if parsed.tzinfo != timezone.utc or parsed.microsecond:
        raise ValueError(f"{label} must be whole-second UTC RFC3339")
    return parsed


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _regular_binding(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an absolute regular non-symlink file")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        digest = hashlib.sha256()
        size = 0
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        size != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError(f"{label} changed while being hashed")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": size}


def _normalized_preflight_gpus(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(values) != GPU_COUNT:
        raise ValueError("D1b preflight requires exactly four H200 GPUs")
    normalized = []
    expected = {
        "host_index",
        "name",
        "sample_window_seconds",
        "sampled_utilization_percent",
        "total_memory_bytes",
        "uuid",
        "visible_index",
    }
    for visible_index, raw in enumerate(values):
        value = _mapping(raw, label=f"D1b preflight GPU {visible_index}")
        samples = _sequence(
            value.get("sampled_utilization_percent"),
            label="D1b GPU utilization samples",
        )
        if (
            set(value) != expected
            or value.get("visible_index") != visible_index
            or type(value.get("host_index")) is not int
            or int(value["host_index"]) < 0
            or value.get("name") != _GPU_NAME
            or type(value.get("total_memory_bytes")) is not int
            or int(value["total_memory_bytes"]) <= 0
            or not isinstance(value.get("uuid"), str)
            or _GPU_UUID.fullmatch(str(value["uuid"])) is None
            or not isinstance(value.get("sample_window_seconds"), (int, float))
            or isinstance(value.get("sample_window_seconds"), bool)
            or float(value["sample_window_seconds"]) < PREFLIGHT_MINIMUM_SECONDS
            or len(samples) < 2
            or any(type(sample) is not int or sample != 0 for sample in samples)
        ):
            raise ValueError("D1b selected GPU failed the ten-second idle sample")
        normalized.append(dict(value))
    if len({gpu["uuid"] for gpu in normalized}) != GPU_COUNT or len(
        {gpu["host_index"] for gpu in normalized}
    ) != GPU_COUNT:
        raise ValueError("D1b preflight GPU identities must be unique")
    return normalized


def _validate_canonical_cleanup_log(
    path: Path,
    *,
    started_at_utc: str,
    completed_at_utc: str,
    selected_gpus: Sequence[Mapping[str, Any]],
    killed_containers: Sequence[str],
) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("D1b cleanup log must be readable UTF-8") from error
    if (
        "Sampling GPU Utilization (10s continuous-zero window" not in text
        or "Reusing the 10s preflight sample" not in text
    ):
        raise ValueError("D1b cleanup log lacks the canonical ten-second sample markers")
    values: dict[str, str] = {}
    tracked = {
        "PREFLIGHT_STARTED_AT_UTC",
        "PREFLIGHT_COMPLETED_AT_UTC",
        "selected_gpu_csv",
        "killed_container_csv",
    }
    for raw in text.splitlines():
        key, separator, value = raw.strip().partition("=")
        if separator and key in tracked:
            if key in values:
                raise ValueError(f"D1b cleanup log repeats {key}")
            values[key] = value
    expected_exact = {
        "PREFLIGHT_STARTED_AT_UTC": started_at_utc,
        "PREFLIGHT_COMPLETED_AT_UTC": completed_at_utc,
        "killed_container_csv": ",".join(killed_containers),
    }
    if set(values) != tracked or any(
        values.get(key) != expected for key, expected in expected_exact.items()
    ):
        raise ValueError("D1b cleanup log machine-readable identity drifted")
    raw_selected = values["selected_gpu_csv"]
    tokens = raw_selected.split(",") if raw_selected else []
    if (
        not tokens
        or any(re.fullmatch(r"[0-9]+", token) is None for token in tokens)
        or len(set(tokens)) != len(tokens)
    ):
        raise ValueError("D1b cleanup log selected GPU CSV is malformed")
    selected_host_indices = {int(token) for token in tokens}
    required_host_indices = {int(gpu["host_index"]) for gpu in selected_gpus}
    if not required_host_indices.issubset(selected_host_indices):
        raise ValueError("D1b cleanup log omitted a bound H200 host index")


def build_action_stability_preflight_evidence_v2(
    *,
    raw_cleanup_log_path: str | Path,
    started_at_utc: str,
    completed_at_utc: str,
    host_alias: str,
    hostname: str,
    container_id: str,
    driver_version: str,
    selected_gpus: Sequence[Mapping[str, Any]],
    killed_containers: Sequence[str] = (),
) -> dict[str, Any]:
    started = _timestamp(started_at_utc, label="D1b preflight start")
    completed = _timestamp(completed_at_utc, label="D1b preflight completion")
    if (completed - started).total_seconds() < PREFLIGHT_MINIMUM_SECONDS:
        raise ValueError("D1b preflight must cover at least ten seconds")
    if {"alias": host_alias, "hostname": hostname} != _HOST:
        raise ValueError("D1b preflight host identity drifted")
    if _CONTAINER_ID.fullmatch(container_id) is None:
        raise ValueError("D1b preflight container identity is invalid")
    if not isinstance(driver_version, str) or not driver_version:
        raise ValueError("D1b driver version is invalid")
    if any(
        not isinstance(name, str)
        or not name
        or len(name) > 255
        or any(character.isspace() for character in name)
        for name in killed_containers
    ) or len(set(killed_containers)) != len(killed_containers):
        raise ValueError("D1b killed-container record is invalid")
    cleanup = _regular_binding(Path(raw_cleanup_log_path), label="D1b cleanup log")
    gpus = _normalized_preflight_gpus(selected_gpus)
    _validate_canonical_cleanup_log(
        Path(raw_cleanup_log_path),
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        selected_gpus=gpus,
        killed_containers=killed_containers,
    )
    return {
        "cleanup": {
            **cleanup,
            "all_containers_allowed": True,
            "killed_containers": list(killed_containers),
        },
        "completed_at_utc": completed_at_utc,
        "container_id": container_id,
        "driver_version": driver_version,
        "gpus": gpus,
        "host_alias": host_alias,
        "hostname": hostname,
        "minimum_idle_window_seconds": PREFLIGHT_MINIMUM_SECONDS,
        "protocol_id": PREFLIGHT_PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": started_at_utc,
        "status": PREFLIGHT_STATUS,
    }


def _default_source_contract_loader(**kwargs: Any) -> SourceContractV2:
    from causalcache.set_utility_action_stability_contract_v2 import (
        load_action_stability_source_v2_contract,
    )

    return load_action_stability_source_v2_contract(**kwargs)


def _source_fields(contract: SourceContractV2) -> dict[str, Any]:
    source = _mapping(contract.data.get("source"), label="D1b source identity")
    diagnostic = _mapping(
        contract.data.get("diagnostic"), label="D1b diagnostic contract"
    )
    boundary = _mapping(
        contract.data.get("execution_boundary"), label="D1b execution boundary"
    )
    inputs = _mapping(
        contract.data.get("immutable_inputs"), label="D1b immutable inputs"
    )
    parent = _mapping(
        inputs.get("parent_d1_aggregate"), label="D1b historical D1 aggregate"
    )
    authorization = _mapping(
        contract.data.get("authorization"), label="D1b source authorization"
    )
    expected_parent = {
        "path": HISTORICAL_D1_AGGREGATE_PATH,
        "sha256": HISTORICAL_D1_AGGREGATE_SHA256,
        "status": HISTORICAL_D1_STATUS,
        "verdict": HISTORICAL_D1_VERDICT,
    }
    expected_diagnostic = {
        "condition_id": CONDITION_ID,
        "state_ids": list(STATE_IDS),
        "generation_call_ceiling": GENERATION_CALL_CEILING,
        "encode_call_ceiling": ENCODE_CALL_CEILING,
        "repeat_count": REPEAT_COUNT,
    }
    for key, expected in expected_diagnostic.items():
        if diagnostic.get(key) != expected:
            raise ValueError(f"D1b source diagnostic {key} drifted")
    if dict(parent) != expected_parent:
        raise ValueError("D1b historical D1 aggregate source binding drifted")
    if source.get("inventory_sha256") is None or _SHA256.fullmatch(
        str(source.get("inventory_sha256"))
    ) is None:
        raise ValueError("D1b source inventory SHA256 is invalid")
    if authorization.get("execution_authorized") is not False:
        raise PermissionError("D1b Source-A must not authorize GPU execution")
    if (
        boundary.get("process_per_state") is not True
        or boundary.get("state_process_count") != STATE_PROCESS_COUNT
        or boundary.get("waves") != [list(wave) for wave in STATE_WAVES]
        or boundary.get("max_concurrent_state_processes")
        != MAX_CONCURRENT_STATE_PROCESSES
        or boundary.get("no_retry") is not True
        or boundary.get("no_top_up") is not True
    ):
        raise ValueError("D1b source execution boundary drifted")
    return {
        "config_sha256": contract.config_sha256,
        "inventory_sha256": source["inventory_sha256"],
    }


def canonical_run_layout(run_root: str | Path) -> dict[str, Any]:
    root = _absolute(str(run_root), label="D1b run root")
    try:
        relative = root.relative_to(PERSISTENT_DATA_ROOT)
    except ValueError as error:
        raise ValueError("D1b run root must be under persistent /data") from error
    if not relative.parts:
        raise ValueError("D1b run root must be task-specific")
    states = []
    wave_by_state = {
        state_id: wave_index
        for wave_index, wave in enumerate(STATE_WAVES)
        for state_id in wave
    }
    slot_by_state = {
        state_id: slot_index
        for wave in STATE_WAVES
        for slot_index, state_id in enumerate(wave)
    }
    for state_index, state_id in enumerate(STATE_IDS):
        state_root = root / "states" / f"state-{state_index:02d}"
        states.append(
            {
                "attempt_path": str(state_root / "attempt.json"),
                "state_id": state_id,
                "state_index": state_index,
                "stderr_log_path": str(root / "logs" / f"state-{state_index:02d}.stderr.log"),
                "stdout_log_path": str(root / "logs" / f"state-{state_index:02d}.stdout.log"),
                "terminal_path": str(state_root / "terminal.json"),
                "wave_index": wave_by_state[state_id],
                "wave_slot": slot_by_state[state_id],
            }
        )
    return {
        "aggregate_result_path": str(root / "aggregate.json"),
        "aggregate_stderr_log_path": str(root / "logs" / "aggregate.stderr.log"),
        "aggregate_stdout_log_path": str(root / "logs" / "aggregate.stdout.log"),
        "envelope_path": str(root / "execution-envelope.json"),
        "preflight_evidence_path": str(root / "preflight-evidence.json"),
        "preflight_raw_log_path": str(root / "preflight-raw.log"),
        "run_root": str(root),
        "states": states,
    }


def _normalized_gpus(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(values) != GPU_COUNT:
        raise ValueError("D1b envelope requires exactly four H200 GPUs")
    normalized = []
    expected_keys = {"host_index", "name", "total_memory_bytes", "uuid", "visible_index"}
    for visible_index, raw in enumerate(values):
        value = _mapping(raw, label=f"D1b GPU {visible_index}")
        if (
            set(value) != expected_keys
            or value.get("visible_index") != visible_index
            or type(value.get("host_index")) is not int
            or int(value["host_index"]) < 0
            or value.get("name") != _GPU_NAME
            or type(value.get("total_memory_bytes")) is not int
            or int(value["total_memory_bytes"]) <= 0
            or not isinstance(value.get("uuid"), str)
            or _GPU_UUID.fullmatch(str(value["uuid"])) is None
        ):
            raise ValueError("D1b GPU identity drifted")
        normalized.append(dict(value))
    if len({gpu["uuid"] for gpu in normalized}) != GPU_COUNT or len(
        {gpu["host_index"] for gpu in normalized}
    ) != GPU_COUNT:
        raise ValueError("D1b selected GPU identities must be unique")
    return normalized


def _canonical_commands(
    *,
    repository_root: Path,
    python_executable: str,
    layout: Mapping[str, Any],
    gpus: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    state_script = repository_root / STATE_ENTRYPOINT
    aggregate_script = repository_root / AGGREGATE_ENTRYPOINT
    if not state_script.is_file() or not aggregate_script.is_file():
        raise ValueError("D1b state runner or aggregate entrypoint is missing")
    layout_states = {item["state_id"]: item for item in layout["states"]}
    processes = []
    for wave_index, wave in enumerate(STATE_WAVES):
        for wave_slot, state_id in enumerate(wave):
            state_layout = layout_states[state_id]
            gpu = gpus[wave_slot]
            processes.append(
                {
                    "argv": [
                        python_executable,
                        str(state_script),
                        "--execution-envelope",
                        layout["envelope_path"],
                        "--state-id",
                        state_id,
                    ],
                    "cuda_visible_devices": gpu["uuid"],
                    "device": "cuda:0",
                    "gpu_uuid": gpu["uuid"],
                    "output": {
                        key: state_layout[key]
                        for key in (
                            "attempt_path",
                            "stderr_log_path",
                            "stdout_log_path",
                            "terminal_path",
                        )
                    },
                    "process_scope": "exactly_one_state_fresh_os_process",
                    "python_path": str(repository_root / "code"),
                    "state_id": state_id,
                    "state_index": STATE_IDS.index(state_id),
                    "wave_index": wave_index,
                    "wave_slot": wave_slot,
                }
            )
    aggregate = [
        python_executable,
        str(aggregate_script),
        "--execution-envelope",
        layout["envelope_path"],
    ]
    return processes, aggregate


def _historical_inputs(root: Path) -> dict[str, Any]:
    aggregate_path = root / HISTORICAL_D1_AGGREGATE_PATH
    aggregate = _regular_binding(aggregate_path, label="historical D1 aggregate")
    if aggregate["sha256"] != HISTORICAL_D1_AGGREGATE_SHA256:
        raise ValueError("historical D1 aggregate SHA256 drifted")
    aggregate_payload = _strict_json_object(
        aggregate_path.read_bytes(), label="historical D1 aggregate"
    )
    if (
        aggregate_path.read_bytes() != canonical_pretty_json_bytes(aggregate_payload)
        or aggregate_payload.get("status") != HISTORICAL_D1_STATUS
        or _mapping(aggregate_payload.get("diagnostic"), label="historical D1 diagnostic").get(
            "verdict"
        )
        != HISTORICAL_D1_VERDICT
    ):
        raise ValueError("historical D1 aggregate identity drifted")
    d1_envelope_path = root / HISTORICAL_D1_ENVELOPE_PATH
    d1_envelope = _regular_binding(d1_envelope_path, label="historical D1 envelope")
    if d1_envelope["sha256"] != HISTORICAL_D1_ENVELOPE_SHA256:
        raise ValueError("historical D1 envelope SHA256 drifted")
    envelope_payload = _strict_json_object(
        d1_envelope_path.read_bytes(), label="historical D1 envelope"
    )
    artifacts = _mapping(envelope_payload.get("artifacts"), label="historical D1 artifacts")
    return {
        "aggregate": aggregate,
        "container": envelope_payload["container"],
        "execution_envelope": d1_envelope,
        "host": envelope_payload["host"],
        "model_dir": artifacts["model_dir"],
        "parent_execution_envelope": artifacts["parent_execution_envelope"],
        "processor_root": artifacts["processor_root"],
        "runtime": envelope_payload["runtime"],
    }


def build_action_stability_envelope_v2(
    *,
    repository_root: str | Path,
    source_git_revision: str,
    run_root: str | Path,
    python_executable: str,
    preflight_evidence_path: str | Path,
    host_alias: str,
    hostname: str,
    container_id: str,
    container_image_reference: str,
    container_image_digest: str,
    driver_version: str,
    software_versions: Mapping[str, str],
    materialized_at_utc: str | None = None,
    source_contract_loader: Callable[..., SourceContractV2] = _default_source_contract_loader,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if not root.is_dir() or root.is_symlink() or _COMMIT.fullmatch(source_git_revision) is None:
        raise ValueError("D1b repository/source revision is invalid")
    contract = source_contract_loader(repository_root=root)
    source_fields = _source_fields(contract)
    if contract.repository_root.resolve() != root:
        raise ValueError("D1b source repository root drifted")
    if {"alias": host_alias, "hostname": hostname} != _HOST:
        raise ValueError("D1b must remain in the frozen Hyper00 host class")
    if _CONTAINER_ID.fullmatch(container_id) is None:
        raise ValueError("D1b container id is invalid")
    if not container_image_reference or _IMAGE_DIGEST.fullmatch(container_image_digest) is None:
        raise ValueError("D1b container image identity is invalid")
    required_versions = {
        "cuda_runtime_version",
        "python_version",
        "torch_version",
        "transformers_version",
    }
    if set(software_versions) != required_versions or any(
        not isinstance(value, str) or not value for value in software_versions.values()
    ):
        raise ValueError("D1b software version inventory drifted")
    python = _absolute(python_executable, label="D1b Python executable")
    if not python.is_file():
        raise ValueError("D1b Python executable does not exist")
    layout = canonical_run_layout(run_root)
    evidence_path = _absolute(
        str(preflight_evidence_path), label="D1b preflight evidence"
    )
    if str(evidence_path) != layout["preflight_evidence_path"]:
        raise ValueError("D1b preflight evidence path is not canonical")
    evidence_binding = _regular_binding(evidence_path, label="D1b preflight evidence")
    evidence_bytes = evidence_path.read_bytes()
    evidence = _strict_json_object(evidence_bytes, label="D1b preflight evidence")
    if evidence_bytes != canonical_pretty_json_bytes(evidence):
        raise ValueError("D1b preflight evidence is not canonical pretty JSON")
    if evidence.get("cleanup", {}).get("path") != layout["preflight_raw_log_path"]:
        raise ValueError("D1b raw preflight log path is not canonical for this run")
    rebuilt_evidence = build_action_stability_preflight_evidence_v2(
        raw_cleanup_log_path=evidence["cleanup"]["path"],
        started_at_utc=evidence["started_at_utc"],
        completed_at_utc=evidence["completed_at_utc"],
        host_alias=host_alias,
        hostname=hostname,
        container_id=container_id,
        driver_version=driver_version,
        selected_gpus=evidence["gpus"],
        killed_containers=evidence["cleanup"]["killed_containers"],
    )
    if evidence != rebuilt_evidence:
        raise ValueError("D1b preflight evidence reconstruction drifted")
    materialized_text = materialized_at_utc or utc_now()
    materialized = _timestamp(materialized_text, label="D1b materialization")
    completed = _timestamp(
        evidence["completed_at_utc"], label="D1b preflight completion"
    )
    if materialized < completed or (
        materialized - completed
    ).total_seconds() > PREFLIGHT_MAX_AGE_SECONDS:
        raise ValueError("D1b preflight is stale at materialization")
    gpus = _normalized_gpus(
        [
            {
                key: gpu[key]
                for key in (
                    "host_index",
                    "name",
                    "total_memory_bytes",
                    "uuid",
                    "visible_index",
                )
            }
            for gpu in evidence["gpus"]
        ]
    )
    historical = _historical_inputs(root)
    d1_runtime = _mapping(historical["runtime"], label="historical D1 runtime")
    d1_container = _mapping(
        historical["container"], label="historical D1 container"
    )
    if (
        d1_runtime.get("software_versions") != dict(software_versions)
        or d1_runtime.get("python_executable") != python_executable
        or historical.get("host") != _HOST
        or d1_container
        != {
            "id": container_id,
            "image_digest": container_image_digest,
            "image_reference": container_image_reference,
        }
    ):
        raise ValueError("D1b must use the historical D1 host/container/runtime stack")
    processes, aggregate_argv = _canonical_commands(
        repository_root=root,
        python_executable=python_executable,
        layout=layout,
        gpus=gpus,
    )
    config_path = root / CANONICAL_CONFIG_PATH
    raw_config_binding = _regular_binding(config_path, label="D1b source config")
    if raw_config_binding["sha256"] != source_fields["config_sha256"]:
        raise ValueError("D1b source config SHA256 drifted")
    config_binding = {
        "path": CANONICAL_CONFIG_PATH,
        "sha256": raw_config_binding["sha256"],
        "size_bytes": raw_config_binding["size_bytes"],
    }
    return {
        "artifacts": historical,
        "authorization": {
            "generate_restoration_labels": False,
            "run_closed_loop": False,
            "run_exact_six_process_sdpa_control": True,
            "source_a_alone_authorizes_execution": False,
            "train_predictor": False,
        },
        "container": {
            "id": container_id,
            "image_digest": container_image_digest,
            "image_reference": container_image_reference,
        },
        "execution": {
            "aggregate": {
                "argv": aggregate_argv,
                "output": {
                    key: layout[key]
                    for key in (
                        "aggregate_result_path",
                        "aggregate_stderr_log_path",
                        "aggregate_stdout_log_path",
                    )
                },
            },
            "condition_id": CONDITION_ID,
            "encode_call_ceiling": ENCODE_CALL_CEILING,
            "generation_call_ceiling": GENERATION_CALL_CEILING,
            "max_concurrent_state_processes": MAX_CONCURRENT_STATE_PROCESSES,
            "no_retry": True,
            "no_top_up": True,
            "output_layout": layout,
            "process_per_state": True,
            "state_process_count": STATE_PROCESS_COUNT,
            "state_processes": processes,
            "wave_count": len(STATE_WAVES),
            "wave_state_ids": [list(wave) for wave in STATE_WAVES],
            "wave_two_requires_all_wave_one_terminals": True,
        },
        "gpus": gpus,
        "host": dict(_HOST),
        "materialized_at_utc": materialized_text,
        "preflight": {
            "completed_at_utc": evidence["completed_at_utc"],
            "evidence": evidence_binding,
            "freshness_maximum_seconds": PREFLIGHT_MAX_AGE_SECONDS,
            "minimum_window_seconds": PREFLIGHT_MINIMUM_SECONDS,
            "started_at_utc": evidence["started_at_utc"],
        },
        "protocol_id": PROTOCOL_ID,
        "runtime": {
            "attention_backend": {"text": "sdpa", "top": "sdpa", "vision": "sdpa"},
            "fresh_os_process_per_state": True,
            "numerical_control_profile": {
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "float32_matmul_precision": "highest",
                "seed": 0,
                "strict_cuda_determinism_claimed": False,
                "tf32_allowed": False,
            },
            "python_executable": python_executable,
            "python_path": str(root / "code"),
            "software_versions": dict(software_versions),
        },
        "schema_version": SCHEMA_VERSION,
        "source": {
            "branch": SOURCE_BRANCH,
            "config": config_binding,
            "execution_config_path": CANONICAL_GIT_ENVELOPE_PATH,
            "git_revision": source_git_revision,
            "historical_d1_result_commit": HISTORICAL_D1_RESULT_COMMIT,
            "inventory_sha256": source_fields["inventory_sha256"],
            "remote": SOURCE_REMOTE,
        },
        "status": AUTHORIZED_STATUS,
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_clean_pushed_source_v2(
    *, repository_root: str | Path, expected_git_revision: str
) -> dict[str, str]:
    root = Path(repository_root).resolve()
    head = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    tracking = _git(root, "rev-parse", f"refs/remotes/{SOURCE_REMOTE}/{SOURCE_BRANCH}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if (
        head != expected_git_revision
        or _COMMIT.fullmatch(head) is None
        or branch != SOURCE_BRANCH
        or tracking != head
        or status
    ):
        raise ValueError("D1b envelope materialization requires clean pushed Source-A")
    committed = subprocess.run(
        ["git", "show", f"{head}:{CANONICAL_CONFIG_PATH}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if committed != (root / CANONICAL_CONFIG_PATH).read_bytes():
        raise ValueError("D1b source config differs from Source-A Git bytes")
    return {"branch": branch, "git_revision": head, "remote_revision": tracking}


def validate_committed_envelope_lifecycle_v2(
    *,
    repository_root: str | Path,
    source_git_revision: str,
    data_envelope_path: str | Path,
    expected_envelope: Mapping[str, Any],
) -> dict[str, str]:
    root = Path(repository_root).resolve()
    head = _git(root, "rev-parse", "HEAD")
    parents = _git(root, "rev-list", "--parents", "-n", "1", head).split()
    changed = _git(
        root, "diff", "--name-only", "--no-renames", source_git_revision, head
    ).splitlines()
    if (
        _git(root, "branch", "--show-current") != SOURCE_BRANCH
        or _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        or _git(root, "rev-parse", f"refs/remotes/{SOURCE_REMOTE}/{SOURCE_BRANCH}") != head
        or parents != [head, source_git_revision]
        or changed != [CANONICAL_GIT_ENVELOPE_PATH]
    ):
        raise ValueError("D1b runtime requires pushed direct-child envelope B")
    payload = canonical_pretty_json_bytes(expected_envelope)
    git_path = root.joinpath(*PurePosixPath(CANONICAL_GIT_ENVELOPE_PATH).parts)
    data_path = _absolute(str(data_envelope_path), label="D1b data envelope")
    committed = subprocess.run(
        ["git", "show", f"{head}:{CANONICAL_GIT_ENVELOPE_PATH}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if (
        not git_path.is_file()
        or git_path.is_symlink()
        or not data_path.is_file()
        or data_path.is_symlink()
        or git_path.read_bytes() != payload
        or data_path.read_bytes() != payload
        or committed != payload
    ):
        raise ValueError("D1b Git-B and /data envelope bytes differ")
    return {"envelope_git_revision": head, "source_git_revision": source_git_revision}


def validate_action_stability_envelope_v2(
    envelope: Mapping[str, Any],
    *,
    repository_root: str | Path,
    envelope_path: str | Path,
    verify_repository: bool = True,
    require_fresh_preflight: bool = True,
    verify_current_environment: bool = False,
    now_utc: str | None = None,
    source_contract_loader: Callable[..., SourceContractV2] = _default_source_contract_loader,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    required_top = {
        "artifacts",
        "authorization",
        "container",
        "execution",
        "gpus",
        "host",
        "materialized_at_utc",
        "preflight",
        "protocol_id",
        "runtime",
        "schema_version",
        "source",
        "status",
    }
    if set(envelope) != required_top:
        raise ValueError("D1b envelope top-level schema drifted")
    if (
        envelope.get("protocol_id") != PROTOCOL_ID
        or envelope.get("schema_version") != SCHEMA_VERSION
        or envelope.get("status") != AUTHORIZED_STATUS
    ):
        raise ValueError("D1b envelope identity drifted")
    contract = source_contract_loader(repository_root=root)
    source_fields = _source_fields(contract)
    source = _mapping(envelope["source"], label="D1b envelope source")
    raw_config_binding = _regular_binding(
        root / CANONICAL_CONFIG_PATH, label="D1b config"
    )
    config_binding = {
        "path": CANONICAL_CONFIG_PATH,
        "sha256": raw_config_binding["sha256"],
        "size_bytes": raw_config_binding["size_bytes"],
    }
    if (
        source.get("branch") != SOURCE_BRANCH
        or source.get("remote") != SOURCE_REMOTE
        or source.get("config") != config_binding
        or config_binding["sha256"] != source_fields["config_sha256"]
        or source.get("inventory_sha256") != source_fields["inventory_sha256"]
        or source.get("execution_config_path") != CANONICAL_GIT_ENVELOPE_PATH
        or source.get("historical_d1_result_commit") != HISTORICAL_D1_RESULT_COMMIT
        or not isinstance(source.get("git_revision"), str)
        or _COMMIT.fullmatch(str(source["git_revision"])) is None
    ):
        raise ValueError("D1b envelope source binding drifted")
    if envelope.get("authorization") != {
        "generate_restoration_labels": False,
        "run_closed_loop": False,
        "run_exact_six_process_sdpa_control": True,
        "source_a_alone_authorizes_execution": False,
        "train_predictor": False,
    }:
        raise ValueError("D1b envelope authorization drifted")
    gpus = _normalized_gpus(
        [dict(_mapping(item, label="D1b GPU")) for item in _sequence(envelope["gpus"], label="D1b GPUs")]
    )
    if envelope.get("host") != _HOST:
        raise ValueError("D1b host binding drifted")
    container = _mapping(envelope["container"], label="D1b container")
    if (
        set(container) != {"id", "image_digest", "image_reference"}
        or not isinstance(container.get("id"), str)
        or _CONTAINER_ID.fullmatch(str(container["id"])) is None
        or not isinstance(container.get("image_digest"), str)
        or _IMAGE_DIGEST.fullmatch(str(container["image_digest"])) is None
        or not isinstance(container.get("image_reference"), str)
        or not container["image_reference"]
    ):
        raise ValueError("D1b container identity drifted")
    historical = _historical_inputs(root)
    if envelope.get("artifacts") != historical:
        raise ValueError("D1b historical artifact binding drifted")
    execution = _mapping(envelope["execution"], label="D1b execution")
    layout = _mapping(execution.get("output_layout"), label="D1b output layout")
    expected_layout = canonical_run_layout(layout.get("run_root"))
    if dict(layout) != expected_layout or str(envelope_path) != layout["envelope_path"]:
        raise ValueError("D1b output layout drifted")
    runtime = _mapping(envelope["runtime"], label="D1b runtime")
    expected_controls = {
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "float32_matmul_precision": "highest",
        "seed": 0,
        "strict_cuda_determinism_claimed": False,
        "tf32_allowed": False,
    }
    processes, aggregate_argv = _canonical_commands(
        repository_root=root,
        python_executable=str(runtime.get("python_executable")),
        layout=layout,
        gpus=gpus,
    )
    if (
        execution.get("condition_id") != CONDITION_ID
        or execution.get("encode_call_ceiling") != ENCODE_CALL_CEILING
        or execution.get("generation_call_ceiling") != GENERATION_CALL_CEILING
        or execution.get("max_concurrent_state_processes")
        != MAX_CONCURRENT_STATE_PROCESSES
        or execution.get("no_retry") is not True
        or execution.get("no_top_up") is not True
        or execution.get("process_per_state") is not True
        or execution.get("state_process_count") != STATE_PROCESS_COUNT
        or execution.get("state_processes") != processes
        or execution.get("wave_count") != len(STATE_WAVES)
        or execution.get("wave_state_ids") != [list(wave) for wave in STATE_WAVES]
        or execution.get("wave_two_requires_all_wave_one_terminals") is not True
        or execution.get("aggregate")
        != {
            "argv": aggregate_argv,
            "output": {
                key: layout[key]
                for key in (
                    "aggregate_result_path",
                    "aggregate_stderr_log_path",
                    "aggregate_stdout_log_path",
                )
            },
        }
    ):
        raise ValueError("D1b exact process schedule drifted")
    if (
        set(runtime)
        != {
            "attention_backend",
            "fresh_os_process_per_state",
            "numerical_control_profile",
            "python_executable",
            "python_path",
            "software_versions",
        }
        or runtime.get("attention_backend")
        != {"text": "sdpa", "top": "sdpa", "vision": "sdpa"}
        or runtime.get("fresh_os_process_per_state") is not True
        or runtime.get("numerical_control_profile") != expected_controls
        or runtime.get("software_versions")
        != _mapping(historical["runtime"], label="historical D1 runtime").get(
            "software_versions"
        )
    ):
        raise ValueError("D1b controlled-SDPA runtime drifted")
    python_executable = _absolute(
        runtime.get("python_executable"), label="D1b Python executable"
    )
    if (
        not python_executable.is_file()
        or runtime.get("python_path") != str(root / "code")
    ):
        raise ValueError("D1b Python/PYTHONPATH binding drifted")
    preflight = _mapping(envelope["preflight"], label="D1b preflight")
    if set(preflight) != {
        "completed_at_utc",
        "evidence",
        "freshness_maximum_seconds",
        "minimum_window_seconds",
        "started_at_utc",
    }:
        raise ValueError("D1b preflight schema drifted")
    evidence_path = _absolute(preflight["evidence"]["path"], label="D1b evidence")
    evidence_binding = _regular_binding(evidence_path, label="D1b preflight evidence")
    evidence_bytes = evidence_path.read_bytes()
    evidence = _strict_json_object(evidence_bytes, label="D1b preflight evidence")
    if (
        preflight.get("evidence") != evidence_binding
        or evidence_bytes != canonical_pretty_json_bytes(evidence)
        or evidence.get("protocol_id") != PREFLIGHT_PROTOCOL_ID
        or evidence.get("status") != PREFLIGHT_STATUS
    ):
        raise ValueError("D1b preflight evidence bytes or identity drifted")
    rebuilt_evidence = build_action_stability_preflight_evidence_v2(
        raw_cleanup_log_path=evidence["cleanup"]["path"],
        started_at_utc=evidence["started_at_utc"],
        completed_at_utc=evidence["completed_at_utc"],
        host_alias=envelope["host"]["alias"],
        hostname=envelope["host"]["hostname"],
        container_id=container["id"],
        driver_version=evidence["driver_version"],
        selected_gpus=evidence["gpus"],
        killed_containers=evidence["cleanup"]["killed_containers"],
    )
    evidence_gpu_identity = [
        {
            key: gpu[key]
            for key in (
                "host_index",
                "name",
                "total_memory_bytes",
                "uuid",
                "visible_index",
            )
        }
        for gpu in evidence["gpus"]
    ]
    started = _timestamp(preflight["started_at_utc"], label="D1b preflight start")
    completed = _timestamp(
        preflight["completed_at_utc"], label="D1b preflight completion"
    )
    materialized = _timestamp(
        envelope["materialized_at_utc"], label="D1b materialization"
    )
    if (
        evidence != rebuilt_evidence
        or evidence_gpu_identity != gpus
        or preflight.get("started_at_utc") != evidence.get("started_at_utc")
        or preflight.get("completed_at_utc") != evidence.get("completed_at_utc")
        or preflight.get("minimum_window_seconds") != PREFLIGHT_MINIMUM_SECONDS
        or preflight.get("freshness_maximum_seconds") != PREFLIGHT_MAX_AGE_SECONDS
        or not started <= completed <= materialized
    ):
        raise ValueError("D1b preflight reconstruction or temporal binding drifted")
    if require_fresh_preflight:
        now = _timestamp(now_utc or utc_now(), label="D1b validation time")
        if now < materialized or (
            now - completed
        ).total_seconds() > PREFLIGHT_MAX_AGE_SECONDS:
            raise ValueError("D1b preflight is no longer fresh")
    if verify_current_environment:
        if os.environ.get("PYTHONPATH") != str(root / "code"):
            raise ValueError("D1b current PYTHONPATH differs from Source-A")
        from causalcache.set_utility_throughput_pilot_envelope_v1 import (
            _validate_current_execution_identity,
        )

        _validate_current_execution_identity(
            hostname=envelope["host"]["hostname"],
            container_id=container["id"],
            driver_version=evidence["driver_version"],
            gpus=gpus,
            python_executable=str(python_executable),
            software_versions=runtime["software_versions"],
            allow_single_gpu_isolation=True,
        )
    if verify_repository:
        validate_committed_envelope_lifecycle_v2(
            repository_root=root,
            source_git_revision=source["git_revision"],
            data_envelope_path=envelope_path,
            expected_envelope=envelope,
        )
    payload = canonical_pretty_json_bytes(envelope)
    return {
        "envelope_sha256": sha256_bytes(payload),
        "current_environment_validated": verify_current_environment,
        "freshness_validated": require_fresh_preflight,
        "historical_d1_aggregate_sha256": HISTORICAL_D1_AGGREGATE_SHA256,
        "source_config_sha256": source_fields["config_sha256"],
        "source_inventory_sha256": source_fields["inventory_sha256"],
        "state_process_count": STATE_PROCESS_COUNT,
        "status": VALIDATION_STATUS,
        "this_validated_envelope_authorizes_gpu_execution": True,
        "wave_state_ids": [list(wave) for wave in STATE_WAVES],
    }


def load_set_utility_action_stability_envelope_v2(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
    verify_repository: bool = True,
    require_fresh_preflight: bool = True,
    verify_current_environment: bool = True,
) -> dict[str, Any]:
    supplied = Path(path)
    if not supplied.is_absolute() or supplied.is_symlink() or not supplied.is_file():
        raise ValueError("D1b execution envelope must be an absolute regular file")
    payload = supplied.read_bytes()
    envelope = _strict_json_object(payload, label="D1b execution envelope")
    if payload != canonical_pretty_json_bytes(envelope):
        raise ValueError("D1b execution envelope must be canonical pretty JSON")
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    validation = validate_action_stability_envelope_v2(
        envelope,
        repository_root=root,
        envelope_path=supplied,
        verify_repository=verify_repository,
        require_fresh_preflight=require_fresh_preflight,
        verify_current_environment=verify_current_environment,
    )
    execution = envelope["execution"]
    return {
        "envelope_path": execution["output_layout"]["envelope_path"],
        "historical_d1_aggregate_path": envelope["artifacts"]["aggregate"]["path"],
        "model_dir": envelope["artifacts"]["model_dir"],
        "parent_envelope_path": envelope["artifacts"]["parent_execution_envelope"]["path"],
        "processor_root": envelope["artifacts"]["processor_root"],
        "repository_root": str(root),
        "run_root": execution["output_layout"]["run_root"],
        "source_config_path": str(root / CANONICAL_CONFIG_PATH),
        "state_processes": execution["state_processes"],
        "validation": validation,
    }


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"D1b output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("D1b exclusive write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_canonical_envelope_pair_exclusive_v2(
    envelope: Mapping[str, Any],
    *,
    repository_root: str | Path,
    data_envelope_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    data_path = _absolute(str(data_envelope_path), label="D1b data envelope")
    git_path = root.joinpath(*PurePosixPath(CANONICAL_GIT_ENVELOPE_PATH).parts)
    payload = canonical_pretty_json_bytes(envelope)
    _write_exclusive(data_path, payload)
    try:
        _write_exclusive(git_path, payload)
    except Exception:
        data_path.unlink(missing_ok=True)
        raise
    return {
        "data_envelope_path": str(data_path),
        "envelope_sha256": sha256_bytes(payload),
        "git_envelope_path": str(git_path),
    }


def write_preflight_evidence_exclusive_v2(
    *, run_root: str | Path, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(canonical_run_layout(run_root)["preflight_evidence_path"])
    payload = canonical_pretty_json_bytes(evidence)
    _write_exclusive(path, payload)
    return {"path": str(path), "sha256": sha256_bytes(payload)}


__all__ = [
    "AUTHORIZED_STATUS",
    "CANONICAL_CONFIG_PATH",
    "CANONICAL_GIT_ENVELOPE_PATH",
    "CONDITION_ID",
    "ENCODE_CALL_CEILING",
    "GENERATION_CALL_CEILING",
    "HISTORICAL_D1_AGGREGATE_SHA256",
    "MAX_CONCURRENT_STATE_PROCESSES",
    "PREFLIGHT_MAX_AGE_SECONDS",
    "PREFLIGHT_MINIMUM_SECONDS",
    "STATE_PROCESS_COUNT",
    "STATE_WAVES",
    "VALIDATION_STATUS",
    "build_action_stability_envelope_v2",
    "build_action_stability_preflight_evidence_v2",
    "canonical_run_layout",
    "load_set_utility_action_stability_envelope_v2",
    "validate_action_stability_envelope_v2",
    "validate_clean_pushed_source_v2",
    "validate_committed_envelope_lifecycle_v2",
    "write_canonical_envelope_pair_exclusive_v2",
    "write_preflight_evidence_exclusive_v2",
]
