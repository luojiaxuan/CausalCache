"""One-state-per-process execution primitives for D1b controlled SDPA."""

from __future__ import annotations

import io
import json
import os
import re
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from causalcache.set_utility_action_stability_contract_v1 import (
    canonical_json_bytes,
    canonical_pretty_json_bytes,
    sha256_bytes,
)
from causalcache.set_utility_action_stability_diagnostic_v1 import (
    WORKER_INDEX_BY_STATE,
    _validate_metric_safe_tree,
)
from causalcache.set_utility_action_stability_diagnostic_v2 import (
    SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION,
    SDPA_NUMERICAL_CONTROL_PROFILE,
    STATE_IDS,
    STATE_WAVES,
    _validate_sdpa_partial_v2,
    run_sdpa_numerical_control_condition_v2,
)
from causalcache.set_utility_action_stability_envelope_v2 import (
    CANONICAL_CONFIG_PATH,
    HISTORICAL_D1_AGGREGATE_SHA256,
    VALIDATION_STATUS,
    canonical_run_layout,
)
from causalcache.set_utility_throughput_pilot_contract_v1 import (
    CANONICAL_CONFIG_PATH as PARENT_SOURCE_CONFIG_PATH,
    load_train_only_throughput_pilot_v1_contract,
)
from causalcache.set_utility_throughput_pilot_execution_v1 import (
    ExecutionLaunchEnvelopeV1 as ParentExecutionLaunchV1,
    ValidatedExecutionEnvelopeV1 as ValidatedParentArtifactsV1,
    load_worker_semantic_inputs_v1,
    validated_execution_from_authorized_projection_v1,
)


SCHEMA_VERSION = "1.0.0"
EXECUTION_PROTOCOL_ID = "causalcache_set_utility_action_stability_execution_v2"
TERMINAL_PROTOCOL_ID = "causalcache_set_utility_action_stability_state_terminal_v2"
CLAIMED_STATUS = "CLAIMED_NO_RETRY_SDPA_CONTROL_STATE_PROCESS_V2"
COMPLETED_STATUS = "COMPLETED_SDPA_CONTROL_STATE_PROCESS_V2"
FAILED_STATUS = "FAILED_SDPA_CONTROL_STATE_PROCESS_V2"
PARENT_EXECUTION_ENVELOPE_SHA256 = (
    "dd0e64fb40bd39f839a1df91240a1e9a1a528a0bbfe9131d49ba726e8ecf4e3a"
)
PARENT_CANDIDATE_SCHEDULE_SHA256 = (
    "186f2952108273672c6cdbf963094754298d23693fd6268a72b6223e99c2299d"
)
PARENT_MODEL_INVENTORY_SHA256 = (
    "6faac059bb7feae4cd350537931ff68142d575a20a9e85ec1c05dedeacc088ee"
)
PARENT_PROCESSOR_INVENTORY_SHA256 = (
    "7c2a971658ad9a6bbc639446e9326e9dd4718b8d2d77186f52a17f10f3b32fc1"
)

_DEVICE = re.compile(r"cuda:[0-9]+")
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{16,}")
_SAFE_FAILURE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PARTIAL_KEYS = {"conditions", "metric_safe", "profile", "state_id"}
_ATTEMPT_KEYS = {
    "candidate_schedule_sha256",
    "condition_id",
    "device_id",
    "fresh_execution_envelope_sha256",
    "fresh_execution_validation_status",
    "historical_d1_aggregate_sha256",
    "metric_safe",
    "model_inventory_sha256",
    "process_identity_sha256",
    "process_scope",
    "processor_inventory_sha256",
    "profile",
    "protocol_id",
    "retry_count",
    "schema_version",
    "source_config_sha256",
    "source_inventory_sha256",
    "state_id",
    "state_index",
    "status",
    "wave_index",
    "wave_slot",
}
_TERMINAL_KEYS = {
    "attempt_sha256",
    "counts",
    "failure_class",
    "fresh_execution_envelope_sha256",
    "historical_d1_aggregate_sha256",
    "metric_safe",
    "partial_result",
    "process_identity_sha256",
    "profile",
    "protocol_id",
    "runtime_identity_sha256",
    "schema_version",
    "source_config_sha256",
    "source_inventory_sha256",
    "state_id",
    "state_index",
    "status",
    "wave_index",
    "wave_slot",
}


class SourceContractV2(Protocol):
    repository_root: Path
    config_sha256: str
    data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ActionStabilityStateLaunchV2:
    """Validated authorization for exactly one state in one fresh process."""

    repository_root: Path
    source_config_path: str
    execution_envelope_path: Path
    parent_envelope_path: Path
    historical_d1_aggregate_path: Path
    processor_root: Path
    model_dir: Path
    output_root: Path
    state_id: str
    state_index: int
    wave_index: int
    wave_slot: int
    device: str
    gpu_uuid: str
    execution_envelope_sha256: str
    source_config_sha256: str
    source_inventory_sha256: str
    historical_d1_aggregate_sha256: str
    validation_status: str
    gpu_execution_authorized: bool

    def __post_init__(self) -> None:
        for value, label in (
            (self.repository_root, "repository root"),
            (self.execution_envelope_path, "execution envelope"),
            (self.parent_envelope_path, "parent envelope"),
            (self.historical_d1_aggregate_path, "historical D1 aggregate"),
            (self.processor_root, "processor root"),
            (self.model_dir, "model directory"),
            (self.output_root, "output root"),
        ):
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"D1b {label} must be one absolute Path")
        if self.source_config_path != CANONICAL_CONFIG_PATH:
            raise ValueError("D1b source config path drifted")
        if (
            self.state_id not in STATE_IDS
            or self.state_index != STATE_IDS.index(self.state_id)
            or self.wave_index not in (0, 1)
            or self.state_id not in STATE_WAVES[self.wave_index]
            or self.wave_slot != STATE_WAVES[self.wave_index].index(self.state_id)
        ):
            raise ValueError("D1b state/wave identity drifted")
        if not isinstance(self.device, str) or _DEVICE.fullmatch(self.device) is None:
            raise ValueError("D1b launch requires one explicit CUDA device")
        if not isinstance(self.gpu_uuid, str) or _GPU_UUID.fullmatch(self.gpu_uuid) is None:
            raise ValueError("D1b launch GPU UUID is invalid")
        for value, label in (
            (self.execution_envelope_sha256, "execution envelope SHA256"),
            (self.source_config_sha256, "source config SHA256"),
            (self.source_inventory_sha256, "source inventory SHA256"),
            (self.historical_d1_aggregate_sha256, "historical aggregate SHA256"),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"D1b {label} is invalid")
        if self.historical_d1_aggregate_sha256 != HISTORICAL_D1_AGGREGATE_SHA256:
            raise ValueError("D1b historical aggregate binding drifted")
        if (
            self.validation_status != VALIDATION_STATUS
            or self.gpu_execution_authorized is not True
        ):
            raise PermissionError("D1b launch lacks direct-child envelope authorization")
        for source in (
            self.repository_root,
            self.parent_envelope_path,
            self.historical_d1_aggregate_path,
            self.processor_root,
            self.model_dir,
        ):
            if self.output_root == source or source in self.output_root.parents:
                raise ValueError("D1b output root must not mutate a bound input")


@dataclass(frozen=True, slots=True)
class ActionStabilityRuntimeBundleV2:
    runtime: object
    reference_input_builder_for: Callable[[object], Callable[[object], object]]
    runtime_identity_sha256: str

    def __post_init__(self) -> None:
        if not callable(self.reference_input_builder_for):
            raise TypeError("D1b reference-input builder factory is not callable")
        if _SHA256.fullmatch(self.runtime_identity_sha256) is None:
            raise ValueError("D1b runtime identity SHA256 is invalid")


SourceContractLoader = Callable[..., SourceContractV2]
ParentBindingLoader = Callable[
    [ActionStabilityStateLaunchV2], tuple[Any, ValidatedParentArtifactsV1]
]
SemanticLoader = Callable[[ValidatedParentArtifactsV1, Any], Sequence[Any]]
RuntimeFactory = Callable[
    [ActionStabilityStateLaunchV2, ValidatedParentArtifactsV1, Any],
    ActionStabilityRuntimeBundleV2,
]
ConditionRunner = Callable[..., Mapping[str, Any]]
StageGate = Callable[[ActionStabilityStateLaunchV2], None]


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
        raise ValueError(f"{label} must be one absolute normalized path")
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        raise ValueError(f"{label} must be one absolute normalized path")
    return path


def _safe_failure(error: BaseException) -> str:
    name = error.__class__.__name__
    return name if _SAFE_FAILURE.fullmatch(name) is not None else "UnexpectedException"


def _strict_regular_payload(path: Path, *, label: str) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be one regular file")
        chunks = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} could not be read safely") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    payload = b"".join(chunks)
    if (
        len(payload) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError(f"{label} changed while being read")
    return payload


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, nested in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate key")
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


def _strict_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    payload = _strict_regular_payload(path, label=label)
    value = _strict_json(payload, label=label)
    if payload != canonical_pretty_json_bytes(value):
        raise ValueError(f"{label} must be canonical pretty JSON")
    return value, sha256_bytes(payload)


def _default_source_contract_loader(**kwargs: Any) -> SourceContractV2:
    from causalcache.set_utility_action_stability_contract_v2 import (
        load_action_stability_source_v2_contract,
    )

    return load_action_stability_source_v2_contract(**kwargs)


def _source_identity(contract: SourceContractV2) -> tuple[str, str]:
    source = _mapping(contract.data.get("source"), label="D1b source identity")
    inventory = source.get("inventory_sha256")
    if (
        not isinstance(contract.config_sha256, str)
        or _SHA256.fullmatch(contract.config_sha256) is None
        or not isinstance(inventory, str)
        or _SHA256.fullmatch(inventory) is None
    ):
        raise ValueError("D1b source contract identity is invalid")
    return contract.config_sha256, inventory


def launch_from_fresh_projection_v2(
    projection: Mapping[str, Any],
    *,
    execution_envelope_path: str | Path,
    state_id: str,
    visible_gpu_uuid: str | None,
) -> ActionStabilityStateLaunchV2:
    """Construct one-state launch only from a validated future envelope."""
    required = {
        "envelope_path",
        "historical_d1_aggregate_path",
        "model_dir",
        "parent_envelope_path",
        "processor_root",
        "repository_root",
        "run_root",
        "source_config_path",
        "state_processes",
        "validation",
    }
    if not isinstance(projection, Mapping) or not required.issubset(projection):
        raise ValueError("D1b execution projection is incomplete")
    supplied = _absolute(str(execution_envelope_path), label="D1b supplied envelope")
    projected = _absolute(projection["envelope_path"], label="D1b projected envelope")
    root = _absolute(projection["repository_root"], label="D1b repository root")
    source_config = _absolute(projection["source_config_path"], label="D1b source config")
    processes = _sequence(projection["state_processes"], label="D1b state processes")
    matches = [item for item in processes if item.get("state_id") == state_id]
    validation = _mapping(projection["validation"], label="D1b validation receipt")
    if (
        supplied != projected
        or source_config != root / CANONICAL_CONFIG_PATH
        or state_id not in STATE_IDS
        or len(matches) != 1
    ):
        raise PermissionError("D1b state projection identity drifted")
    process = _mapping(matches[0], label="D1b selected state process")
    if (
        process.get("process_scope") != "exactly_one_state_fresh_os_process"
        or process.get("device") != "cuda:0"
        or process.get("gpu_uuid") != visible_gpu_uuid
        or process.get("cuda_visible_devices") != visible_gpu_uuid
        or process.get("state_index") != STATE_IDS.index(state_id)
        or process.get("wave_index") not in (0, 1)
        or state_id not in STATE_WAVES[int(process["wave_index"])]
        or process.get("wave_slot")
        != STATE_WAVES[int(process["wave_index"])].index(state_id)
    ):
        raise PermissionError("D1b isolated state/GPU process binding drifted")
    if (
        validation.get("status") != VALIDATION_STATUS
        or validation.get("state_process_count") != len(STATE_IDS)
        or validation.get("wave_state_ids") != [list(wave) for wave in STATE_WAVES]
        or validation.get("historical_d1_aggregate_sha256")
        != HISTORICAL_D1_AGGREGATE_SHA256
        or validation.get("freshness_validated") is not True
        or validation.get("current_environment_validated") is not True
        or validation.get("this_validated_envelope_authorizes_gpu_execution") is not True
    ):
        raise PermissionError("D1b validation receipt is not authorizing")
    return ActionStabilityStateLaunchV2(
        repository_root=root,
        source_config_path=CANONICAL_CONFIG_PATH,
        execution_envelope_path=projected,
        parent_envelope_path=_absolute(
            projection["parent_envelope_path"], label="D1b parent envelope"
        ),
        historical_d1_aggregate_path=_absolute(
            projection["historical_d1_aggregate_path"],
            label="D1b historical aggregate",
        ),
        processor_root=_absolute(projection["processor_root"], label="D1b processor"),
        model_dir=_absolute(projection["model_dir"], label="D1b model"),
        output_root=_absolute(projection["run_root"], label="D1b output root"),
        state_id=state_id,
        state_index=int(process["state_index"]),
        wave_index=int(process["wave_index"]),
        wave_slot=int(process["wave_slot"]),
        device="cuda:0",
        gpu_uuid=str(visible_gpu_uuid),
        execution_envelope_sha256=str(validation.get("envelope_sha256")),
        source_config_sha256=str(validation.get("source_config_sha256")),
        source_inventory_sha256=str(validation.get("source_inventory_sha256")),
        historical_d1_aggregate_sha256=str(
            validation.get("historical_d1_aggregate_sha256")
        ),
        validation_status=str(validation.get("status")),
        gpu_execution_authorized=(
            validation.get("this_validated_envelope_authorizes_gpu_execution") is True
        ),
    )


def _default_parent_envelope_loader(path: str | Path, **kwargs: Any) -> Mapping[str, Any]:
    from causalcache.set_utility_throughput_pilot_envelope_v1 import (
        load_set_utility_throughput_pilot_execution_envelope_v1,
    )

    return load_set_utility_throughput_pilot_execution_envelope_v1(path, **kwargs)


def _parent_repository_root(parent_envelope_path: Path) -> Path:
    parent, _ = _strict_canonical(parent_envelope_path, label="D1b parent envelope")
    execution = _mapping(parent.get("execution"), label="D1b parent execution")
    workers = _sequence(execution.get("worker_mapping"), label="D1b parent workers")
    first = _mapping(workers[0], label="D1b parent first worker")
    argv = _sequence(first.get("argv"), label="D1b parent worker argv")
    if len(argv) < 2:
        raise ValueError("D1b parent worker argv is incomplete")
    return Path(str(argv[1])).resolve().parents[2]


def load_parent_artifact_binding_v2(
    launch: ActionStabilityStateLaunchV2,
    *,
    parent_contract_loader: Callable[..., Any] = load_train_only_throughput_pilot_v1_contract,
    parent_envelope_loader: Callable[..., Mapping[str, Any]] = _default_parent_envelope_loader,
    projection_adapter: Callable[..., ValidatedParentArtifactsV1] = (
        validated_execution_from_authorized_projection_v1
    ),
) -> tuple[Any, ValidatedParentArtifactsV1]:
    """Reuse immutable parent artifacts without reusing its execution authority."""
    _, observed_sha = _strict_canonical(
        launch.parent_envelope_path, label="D1b parent execution envelope"
    )
    if observed_sha != PARENT_EXECUTION_ENVELOPE_SHA256:
        raise ValueError("D1b parent execution envelope bytes drifted")
    parent_contract = parent_contract_loader(
        repository_root=launch.repository_root,
        config_path=PARENT_SOURCE_CONFIG_PATH,
    )
    parent_root = _parent_repository_root(launch.parent_envelope_path)
    projection = parent_envelope_loader(
        launch.parent_envelope_path,
        repository_root=parent_root,
        verify_repository=False,
        verify_local_artifacts="stat",
        require_fresh_preflight=False,
        verify_current_environment=False,
    )
    if (
        projection.get("processor_root") != str(launch.processor_root)
        or projection.get("model_dir") != str(launch.model_dir)
    ):
        raise ValueError("D1b parent artifact paths drifted")
    parent_output = _absolute(projection["run_root"], label="D1b parent run root")
    if parent_output == launch.output_root:
        raise PermissionError("D1b output root must differ from parent output")
    parent_launch = ParentExecutionLaunchV1(
        repository_root=launch.repository_root,
        config_path=PARENT_SOURCE_CONFIG_PATH,
        processor_root=launch.processor_root,
        model_dir=launch.model_dir,
        output_root=parent_output,
        worker_index=WORKER_INDEX_BY_STATE[launch.state_id],
        device=launch.device,
    )
    return parent_contract, projection_adapter(
        parent_launch,
        parent_contract,
        {**projection, "repository_root": str(launch.repository_root)},
    )


def _select_state(parent_states: Sequence[Any], *, state_id: str) -> Any:
    matches = [state for state in parent_states if getattr(state, "state_id", None) == state_id]
    if len(matches) != 1:
        raise ValueError("D1b exact state is missing from parent semantic inputs")
    return matches[0]


def _validate_partial(value: Mapping[str, Any], *, state_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PARTIAL_KEYS:
        raise ValueError("D1b state partial schema drifted")
    projected = _validate_sdpa_partial_v2(value)
    if projected["state_id"] != state_id:
        raise ValueError("D1b state partial identity drifted")
    _validate_metric_safe_tree(projected)
    canonical_json_bytes(projected)
    return projected


def _expected_state_coordinates(state_id: str) -> tuple[int, int, int]:
    if state_id not in STATE_IDS:
        raise ValueError("D1b state identity is outside the exact roster")
    state_index = STATE_IDS.index(state_id)
    wave_index = 0 if state_id in STATE_WAVES[0] else 1
    return state_index, wave_index, STATE_WAVES[wave_index].index(state_id)


def validate_attempt_payload_v2(
    attempt: Mapping[str, Any],
    *,
    state_id: str,
    envelope_sha256: str,
    source_config_sha256: str,
    source_inventory_sha256: str,
) -> dict[str, Any]:
    """Validate every field of one immutable no-retry state claim."""
    state_index, wave_index, wave_slot = _expected_state_coordinates(state_id)
    if not isinstance(attempt, Mapping) or set(attempt) != _ATTEMPT_KEYS:
        raise ValueError("D1b attempt schema drifted")
    sha_fields = (
        "candidate_schedule_sha256",
        "fresh_execution_envelope_sha256",
        "historical_d1_aggregate_sha256",
        "model_inventory_sha256",
        "process_identity_sha256",
        "processor_inventory_sha256",
        "source_config_sha256",
        "source_inventory_sha256",
    )
    if any(
        not isinstance(attempt.get(key), str)
        or _SHA256.fullmatch(str(attempt[key])) is None
        for key in sha_fields
    ):
        raise ValueError("D1b attempt SHA256 identity is invalid")
    if (
        attempt.get("condition_id")
        != SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION
        or attempt.get("candidate_schedule_sha256")
        != PARENT_CANDIDATE_SCHEDULE_SHA256
        or attempt.get("device_id") != "cuda:0"
        or attempt.get("fresh_execution_envelope_sha256") != envelope_sha256
        or attempt.get("fresh_execution_validation_status") != VALIDATION_STATUS
        or attempt.get("historical_d1_aggregate_sha256")
        != HISTORICAL_D1_AGGREGATE_SHA256
        or attempt.get("metric_safe") is not True
        or attempt.get("model_inventory_sha256") != PARENT_MODEL_INVENTORY_SHA256
        or attempt.get("process_scope") != "exactly_one_state_fresh_os_process"
        or attempt.get("processor_inventory_sha256")
        != PARENT_PROCESSOR_INVENTORY_SHA256
        or attempt.get("profile") != SDPA_NUMERICAL_CONTROL_PROFILE
        or attempt.get("protocol_id") != EXECUTION_PROTOCOL_ID
        or attempt.get("retry_count") != 0
        or attempt.get("schema_version") != SCHEMA_VERSION
        or attempt.get("source_config_sha256") != source_config_sha256
        or attempt.get("source_inventory_sha256") != source_inventory_sha256
        or attempt.get("state_id") != state_id
        or attempt.get("state_index") != state_index
        or attempt.get("status") != CLAIMED_STATUS
        or attempt.get("wave_index") != wave_index
        or attempt.get("wave_slot") != wave_slot
    ):
        raise ValueError("D1b attempt identity or execution boundary drifted")
    projected = dict(attempt)
    _validate_metric_safe_tree(projected)
    return projected


def validate_terminal_payload_v2(
    terminal: Mapping[str, Any],
    *,
    state_id: str,
    envelope_sha256: str,
    source_config_sha256: str,
    source_inventory_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate a completed condition row or a class-only process failure."""
    state_index, wave_index, wave_slot = _expected_state_coordinates(state_id)
    if not isinstance(terminal, Mapping) or set(terminal) != _TERMINAL_KEYS:
        raise ValueError("D1b terminal schema drifted")
    if (
        terminal.get("protocol_id") != TERMINAL_PROTOCOL_ID
        or terminal.get("schema_version") != SCHEMA_VERSION
        or terminal.get("status") not in {COMPLETED_STATUS, FAILED_STATUS}
        or terminal.get("metric_safe") is not True
        or terminal.get("profile") != SDPA_NUMERICAL_CONTROL_PROFILE
        or terminal.get("state_id") != state_id
        or terminal.get("state_index") != state_index
        or terminal.get("wave_index") != wave_index
        or terminal.get("wave_slot") != wave_slot
        or terminal.get("fresh_execution_envelope_sha256") != envelope_sha256
        or terminal.get("historical_d1_aggregate_sha256")
        != HISTORICAL_D1_AGGREGATE_SHA256
        or terminal.get("source_config_sha256") != source_config_sha256
        or terminal.get("source_inventory_sha256") != source_inventory_sha256
        or not isinstance(terminal.get("attempt_sha256"), str)
        or _SHA256.fullmatch(str(terminal["attempt_sha256"])) is None
        or not isinstance(terminal.get("process_identity_sha256"), str)
        or _SHA256.fullmatch(str(terminal["process_identity_sha256"])) is None
    ):
        raise ValueError("D1b terminal identity drifted")
    partial: dict[str, Any] | None = None
    if terminal["status"] == COMPLETED_STATUS:
        if (
            terminal.get("failure_class") is not None
            or not isinstance(terminal.get("runtime_identity_sha256"), str)
            or _SHA256.fullmatch(str(terminal["runtime_identity_sha256"])) is None
        ):
            raise ValueError("completed D1b terminal lost its runtime identity")
        partial = _validate_partial(terminal.get("partial_result"), state_id=state_id)
        conditions = partial["conditions"]
        expected_counts = {
            "condition_completed_count": 1,
            "encode_call_count": sum(int(item["encode_call_count"]) for item in conditions),
            "generation_call_count": sum(
                int(item["generation_call_count"]) for item in conditions
            ),
            "label_count": 0,
            "restoration_distance_count": 0,
            "retry_count": 0,
            "teacher_forward_call_count": 0,
            "training_example_count": 0,
        }
    else:
        failure = terminal.get("failure_class")
        runtime_sha = terminal.get("runtime_identity_sha256")
        if (
            not isinstance(failure, str)
            or _SAFE_FAILURE.fullmatch(failure) is None
            or terminal.get("partial_result") is not None
            or (
                runtime_sha is not None
                and (
                    not isinstance(runtime_sha, str)
                    or _SHA256.fullmatch(runtime_sha) is None
                )
            )
        ):
            raise ValueError("failed D1b terminal must be class-only and metric-safe")
        expected_counts = {
            "condition_completed_count": 0,
            "encode_call_count": 0,
            "generation_call_count": 0,
            "label_count": 0,
            "restoration_distance_count": 0,
            "retry_count": 0,
            "teacher_forward_call_count": 0,
            "training_example_count": 0,
        }
    if terminal.get("counts") != expected_counts:
        raise ValueError("D1b terminal operation counts drifted")
    projected = dict(terminal)
    _validate_metric_safe_tree(projected)
    return projected, partial


def require_prior_wave_complete_v2(launch: ActionStabilityStateLaunchV2) -> None:
    """Gate wave two on all four immutable wave-one terminals."""
    if not isinstance(launch, ActionStabilityStateLaunchV2):
        raise TypeError("D1b stage gate requires one state launch")
    if launch.wave_index == 0:
        return
    layout = canonical_run_layout(launch.output_root)
    by_state = {item["state_id"]: item for item in layout["states"]}
    for state_id in STATE_WAVES[0]:
        state_layout = by_state[state_id]
        attempt, attempt_sha = _strict_canonical(
            Path(state_layout["attempt_path"]), label=f"D1b wave-one {state_id} attempt"
        )
        terminal, _ = _strict_canonical(
            Path(state_layout["terminal_path"]), label=f"D1b wave-one {state_id} terminal"
        )
        validate_attempt_payload_v2(
            attempt,
            state_id=state_id,
            envelope_sha256=launch.execution_envelope_sha256,
            source_config_sha256=launch.source_config_sha256,
            source_inventory_sha256=launch.source_inventory_sha256,
        )
        validate_terminal_payload_v2(
            terminal,
            state_id=state_id,
            envelope_sha256=launch.execution_envelope_sha256,
            source_config_sha256=launch.source_config_sha256,
            source_inventory_sha256=launch.source_inventory_sha256,
        )
        if (
            terminal.get("attempt_sha256") != attempt_sha
            or terminal.get("process_identity_sha256")
            != attempt.get("process_identity_sha256")
        ):
            raise PermissionError("D1b wave-two gate found a drifted wave-one terminal")


def build_production_runtime_bundle_v2(
    launch: ActionStabilityStateLaunchV2,
    validated: ValidatedParentArtifactsV1,
    parent_contract: Any,
) -> ActionStabilityRuntimeBundleV2:
    """Construct the sole controlled-SDPA runtime in this fresh process."""
    from PIL import Image

    from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v2 import (
        GUIOwlV21SDPANumericalControlActionStabilityRuntimeV2,
    )
    from causalcache.policy.gui_owl_v2_runtime import (
        FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    from causalcache.set_utility_gui_owl_v2_1_throughput_adapter import (
        build_gui_owl_v2_1_throughput_reference_input,
    )

    inputs = _mapping(parent_contract.data.get("inputs"), label="D1b parent inputs")
    model_binding = _mapping(
        inputs.get("model_snapshot_manifest"), label="D1b parent model binding"
    )
    runtime = GUIOwlV21SDPANumericalControlActionStabilityRuntimeV2(
        model_dir=launch.model_dir,
        expected_snapshot_manifest=launch.repository_root / str(model_binding["path"]),
        device=launch.device,
        target_effective_visual_tokens_per_image=(
            FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        ),
    )

    def decode_rgb(payload: bytes) -> Any:
        source = Image.open(io.BytesIO(payload))
        try:
            return source.convert("RGB")
        finally:
            source.close()

    def builder_for(joined: object) -> Callable[[object], object]:
        return lambda plan: build_gui_owl_v2_1_throughput_reference_input(
            joined, plan, image_decoder=decode_rgb
        )

    identity = {
        "device": launch.device,
        "model_inventory_sha256": validated.model_inventory_sha256,
        "profile": SDPA_NUMERICAL_CONTROL_PROFILE,
        "runtime_metadata_sha256": sha256_bytes(
            canonical_json_bytes(dict(runtime.metadata))
        ),
        "state_id": launch.state_id,
    }
    return ActionStabilityRuntimeBundleV2(
        runtime=runtime,
        reference_input_builder_for=builder_for,
        runtime_identity_sha256=sha256_bytes(canonical_json_bytes(identity)),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("D1b output parent must be one real directory")
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
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError("D1b exclusive write made no progress")
            view = view[count:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_atomic_no_clobber(path: Path, payload: bytes) -> None:
    partial = path.with_name(f".{path.name}.partial")
    _write_exclusive(partial, payload)
    try:
        os.link(partial, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        partial.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def _process_identity(launch: ActionStabilityStateLaunchV2) -> str:
    identity = {
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "state_id": launch.state_id,
        "wall_time_ns": time.time_ns(),
    }
    return sha256_bytes(canonical_json_bytes(identity))


def _attempt_payload(
    launch: ActionStabilityStateLaunchV2,
    *,
    process_identity_sha256: str,
    validated: ValidatedParentArtifactsV1,
) -> dict[str, Any]:
    return {
        "candidate_schedule_sha256": sha256_bytes(validated.candidate_schedule_bytes),
        "condition_id": SDPA_NUMERICAL_CONTROL_FROZEN_ENCODED_CONDITION,
        "device_id": launch.device,
        "fresh_execution_envelope_sha256": launch.execution_envelope_sha256,
        "fresh_execution_validation_status": launch.validation_status,
        "historical_d1_aggregate_sha256": launch.historical_d1_aggregate_sha256,
        "metric_safe": True,
        "model_inventory_sha256": validated.model_inventory_sha256,
        "process_identity_sha256": process_identity_sha256,
        "process_scope": "exactly_one_state_fresh_os_process",
        "processor_inventory_sha256": validated.processor_inventory_sha256,
        "profile": SDPA_NUMERICAL_CONTROL_PROFILE,
        "protocol_id": EXECUTION_PROTOCOL_ID,
        "retry_count": 0,
        "schema_version": SCHEMA_VERSION,
        "source_config_sha256": launch.source_config_sha256,
        "source_inventory_sha256": launch.source_inventory_sha256,
        "state_id": launch.state_id,
        "state_index": launch.state_index,
        "status": CLAIMED_STATUS,
        "wave_index": launch.wave_index,
        "wave_slot": launch.wave_slot,
    }


def run_action_stability_state_v2(
    launch: ActionStabilityStateLaunchV2,
    *,
    source_contract_loader: SourceContractLoader = _default_source_contract_loader,
    parent_binding_loader: ParentBindingLoader = load_parent_artifact_binding_v2,
    semantic_loader: SemanticLoader = load_worker_semantic_inputs_v1,
    runtime_factory: RuntimeFactory = build_production_runtime_bundle_v2,
    condition_runner: ConditionRunner = run_sdpa_numerical_control_condition_v2,
    stage_gate: StageGate = require_prior_wave_complete_v2,
) -> dict[str, Any]:
    """Claim once, execute one state, write one terminal, and then return."""
    if not isinstance(launch, ActionStabilityStateLaunchV2):
        raise TypeError("D1b launch has the wrong type")
    contract = source_contract_loader(
        repository_root=launch.repository_root,
        config_path=launch.source_config_path,
    )
    config_sha, inventory_sha = _source_identity(contract)
    if (
        contract.repository_root.resolve() != launch.repository_root.resolve()
        or config_sha != launch.source_config_sha256
        or inventory_sha != launch.source_inventory_sha256
    ):
        raise PermissionError("D1b envelope differs from its Source-A contract")
    stage_gate(launch)
    parent_contract, validated = parent_binding_loader(launch)
    layout = canonical_run_layout(launch.output_root)
    state_layout = layout["states"][launch.state_index]
    attempt_path = Path(state_layout["attempt_path"])
    terminal_path = Path(state_layout["terminal_path"])
    process_sha = _process_identity(launch)
    attempt = _attempt_payload(
        launch,
        process_identity_sha256=process_sha,
        validated=validated,
    )
    attempt_bytes = canonical_pretty_json_bytes(attempt)
    _write_exclusive(attempt_path, attempt_bytes)
    attempt_sha = sha256_bytes(attempt_bytes)

    partial: dict[str, Any] | None = None
    runtime_sha: str | None = None
    failure_class: str | None = None
    try:
        parent_states = semantic_loader(validated, parent_contract)
        state = _select_state(parent_states, state_id=launch.state_id)
        bundle = runtime_factory(launch, validated, parent_contract)
        runtime_sha = bundle.runtime_identity_sha256
        raw = condition_runner(
            state.query,
            reference_input_builder=bundle.reference_input_builder_for(state.joined_input),
            runtime=bundle.runtime,
        )
        partial = _validate_partial(raw, state_id=launch.state_id)
    except Exception as error:
        failure_class = _safe_failure(error)

    conditions = [] if partial is None else partial["conditions"]
    generation_calls = sum(int(item["generation_call_count"]) for item in conditions)
    encode_calls = sum(int(item["encode_call_count"]) for item in conditions)
    completed = failure_class is None and partial is not None
    terminal = {
        "attempt_sha256": attempt_sha,
        "counts": {
            "condition_completed_count": len(conditions),
            "encode_call_count": encode_calls,
            "generation_call_count": generation_calls,
            "label_count": 0,
            "restoration_distance_count": 0,
            "retry_count": 0,
            "teacher_forward_call_count": 0,
            "training_example_count": 0,
        },
        "failure_class": failure_class,
        "fresh_execution_envelope_sha256": launch.execution_envelope_sha256,
        "historical_d1_aggregate_sha256": launch.historical_d1_aggregate_sha256,
        "metric_safe": True,
        "partial_result": partial,
        "process_identity_sha256": process_sha,
        "profile": SDPA_NUMERICAL_CONTROL_PROFILE,
        "protocol_id": TERMINAL_PROTOCOL_ID,
        "runtime_identity_sha256": runtime_sha,
        "schema_version": SCHEMA_VERSION,
        "source_config_sha256": config_sha,
        "source_inventory_sha256": inventory_sha,
        "state_id": launch.state_id,
        "state_index": launch.state_index,
        "status": COMPLETED_STATUS if completed else FAILED_STATUS,
        "wave_index": launch.wave_index,
        "wave_slot": launch.wave_slot,
    }
    _validate_metric_safe_tree(terminal)
    _write_atomic_no_clobber(terminal_path, canonical_pretty_json_bytes(terminal))
    return terminal


__all__ = [
    "ActionStabilityRuntimeBundleV2",
    "ActionStabilityStateLaunchV2",
    "COMPLETED_STATUS",
    "EXECUTION_PROTOCOL_ID",
    "FAILED_STATUS",
    "SCHEMA_VERSION",
    "TERMINAL_PROTOCOL_ID",
    "_TERMINAL_KEYS",
    "_strict_canonical",
    "_validate_partial",
    "build_production_runtime_bundle_v2",
    "launch_from_fresh_projection_v2",
    "load_parent_artifact_binding_v2",
    "require_prior_wave_complete_v2",
    "run_action_stability_state_v2",
    "validate_attempt_payload_v2",
    "validate_terminal_payload_v2",
]
