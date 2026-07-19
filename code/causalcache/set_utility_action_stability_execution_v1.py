"""Fail-closed profile workers for the train-only D1 action-stability diagnostic.

The completed throughput-v2 envelope is reused only to bind the immutable
processor/model artifacts and to reconstruct its train-only semantic inputs.
Its expired preflight never authorizes this execution.  A separate fresh D1
envelope must be validated before constructing :class:`ActionStabilityLaunchV1`.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from causalcache.set_utility_action_stability_diagnostic_v1 import (
    AUTO_FRESH_ENCODE_CONDITION,
    AUTO_FROZEN_ENCODED_CONDITION,
    EAGER_FROZEN_ENCODED_CONTROL,
    STATE_IDS,
    _validate_condition_payload,
    _validate_metric_safe_tree,
    run_default_conditions_v1,
    run_eager_condition_v1,
    state_ids_for_worker_v1,
)
from causalcache.set_utility_throughput_pilot_contract_v1 import (
    CANONICAL_CONFIG_PATH as PARENT_SOURCE_CONFIG_PATH,
    canonical_json_bytes,
    canonical_pretty_json_bytes,
    load_train_only_throughput_pilot_v1_contract,
    sha256_bytes,
)
from causalcache.set_utility_throughput_pilot_execution_v1 import (
    ExecutionLaunchEnvelopeV1 as ParentExecutionLaunchV1,
    ValidatedExecutionEnvelopeV1 as ValidatedParentArtifactsV1,
    load_worker_semantic_inputs_v1,
    validated_execution_from_authorized_projection_v1,
)


SCHEMA_VERSION = "1.0.0"
EXECUTION_PROTOCOL_ID = "causalcache_set_utility_action_stability_execution_v1"
TERMINAL_PROTOCOL_ID = (
    "causalcache_set_utility_action_stability_profile_worker_terminal_v1"
)
FRESH_ENVELOPE_VALIDATION_STATUS = (
    "VALID_SET_UTILITY_ACTION_STABILITY_EXECUTION_ENVELOPE_V1"
)
CANONICAL_D1_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_action_stability_diagnostic_v1.json"
)
PARENT_EXECUTION_ENVELOPE_SHA256 = (
    "dd0e64fb40bd39f839a1df91240a1e9a1a528a0bbfe9131d49ba726e8ecf4e3a"
)
PROFILE_AUTO = "auto"
PROFILE_EAGER = "eager"
PROFILE_ORDER = (PROFILE_AUTO, PROFILE_EAGER)
WORKER_COUNT = 4

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{16,}")
_DEVICE = re.compile(r"cuda:[0-9]+")
_SAFE_FAILURE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_PARTIAL_KEYS = {"conditions", "metric_safe", "profile", "state_id"}
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
_PROFILE_RESULT_ID = {
    PROFILE_AUTO: "auto_default",
    PROFILE_EAGER: "eager_numerical_control_not_strict_cuda_determinism",
}
_PROFILE_CONDITIONS = {
    PROFILE_AUTO: (AUTO_FRESH_ENCODE_CONDITION, AUTO_FROZEN_ENCODED_CONDITION),
    PROFILE_EAGER: (EAGER_FROZEN_ENCODED_CONTROL,),
}


class D1SourceContract(Protocol):
    repository_root: Path
    config_sha256: str
    data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ActionStabilityLaunchV1:
    """Projection of one separately authorized profile/worker process."""

    repository_root: Path
    source_config_path: str
    execution_envelope_path: Path
    parent_envelope_path: Path
    processor_root: Path
    model_dir: Path
    output_root: Path
    worker_index: int
    profile: str
    device: str
    worker_gpu_uuid: str
    fresh_envelope_sha256: str
    source_config_sha256: str
    source_inventory_sha256: str
    parent_envelope_sha256: str
    fresh_validation_status: str
    freshness_validated: bool
    gpu_execution_authorized: bool

    def __post_init__(self) -> None:
        for value, label in (
            (self.repository_root, "repository root"),
            (self.execution_envelope_path, "execution envelope"),
            (self.parent_envelope_path, "parent envelope"),
            (self.processor_root, "processor root"),
            (self.model_dir, "model directory"),
            (self.output_root, "output root"),
        ):
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"D1 {label} must be one absolute Path")
        if self.source_config_path != CANONICAL_D1_CONFIG_PATH:
            raise ValueError("D1 source config path drifted")
        if type(self.worker_index) is not int or not 0 <= self.worker_index < WORKER_COUNT:
            raise ValueError("D1 worker index must be in [0, 3]")
        if self.profile not in PROFILE_ORDER:
            raise ValueError("D1 profile must be auto or eager")
        if not isinstance(self.device, str) or _DEVICE.fullmatch(self.device) is None:
            raise ValueError("D1 launch requires one explicit CUDA device")
        if (
            not isinstance(self.worker_gpu_uuid, str)
            or _GPU_UUID.fullmatch(self.worker_gpu_uuid) is None
        ):
            raise ValueError("D1 worker GPU UUID is invalid")
        for value, label in (
            (self.fresh_envelope_sha256, "fresh envelope SHA256"),
            (self.source_config_sha256, "source config SHA256"),
            (self.source_inventory_sha256, "source inventory SHA256"),
            (self.parent_envelope_sha256, "parent envelope SHA256"),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"D1 {label} is invalid")
        if self.parent_envelope_sha256 != PARENT_EXECUTION_ENVELOPE_SHA256:
            raise ValueError("D1 parent envelope binding drifted")
        if (
            self.fresh_validation_status != FRESH_ENVELOPE_VALIDATION_STATUS
            or self.freshness_validated is not True
            or self.gpu_execution_authorized is not True
        ):
            raise PermissionError("D1 launch lacks a fresh GPU execution authorization")
        for source in (self.repository_root, self.processor_root, self.model_dir):
            if self.output_root == source or source in self.output_root.parents:
                raise ValueError("D1 output root must not mutate a bound input tree")


@dataclass(frozen=True, slots=True)
class ActionStabilityRuntimeBundleV1:
    runtime: object
    reference_input_builder_for: Callable[[object], Callable[[object], object]]
    runtime_identity_sha256: str

    def __post_init__(self) -> None:
        if not callable(self.reference_input_builder_for):
            raise TypeError("D1 reference-input builder factory is not callable")
        if (
            not isinstance(self.runtime_identity_sha256, str)
            or _SHA256.fullmatch(self.runtime_identity_sha256) is None
        ):
            raise ValueError("D1 runtime identity SHA256 is invalid")


SourceContractLoader = Callable[..., D1SourceContract]
ParentContractLoader = Callable[..., Any]
ParentEnvelopeLoader = Callable[..., Mapping[str, Any]]
ParentProjectionAdapter = Callable[..., ValidatedParentArtifactsV1]
ParentBindingLoader = Callable[
    [ActionStabilityLaunchV1], tuple[Any, ValidatedParentArtifactsV1]
]
SemanticLoader = Callable[[ValidatedParentArtifactsV1, Any], Sequence[Any]]
RuntimeFactory = Callable[
    [ActionStabilityLaunchV1, ValidatedParentArtifactsV1, Any],
    ActionStabilityRuntimeBundleV1,
]
ProfileRunner = Callable[..., Mapping[str, Any]]
StageGate = Callable[[ActionStabilityLaunchV1], None]


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


def _absolute_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an absolute path string")
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        raise ValueError(f"{label} must be an absolute normalized path")
    return path


def _safe_failure(error: Exception) -> str:
    name = error.__class__.__name__
    return name if _SAFE_FAILURE.fullmatch(name) is not None else "UnexpectedException"


def _strict_file_sha256(path: Path, *, label: str) -> str:
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
        digest = hashlib.sha256()
        byte_count = 0
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} could not be read safely") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        byte_count != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError(f"{label} changed while being hashed")
    return digest.hexdigest()


def _strict_regular_file_payload(path: Path, *, label: str) -> bytes:
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
        chunks: list[bytes] = []
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


def _strict_json_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

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


def _default_source_contract_loader(**kwargs: Any) -> D1SourceContract:
    from causalcache.set_utility_action_stability_contract_v1 import (
        load_action_stability_source_v1_contract,
    )

    return load_action_stability_source_v1_contract(**kwargs)


def _default_parent_envelope_loader(
    path: str | Path,
    **kwargs: Any,
) -> Mapping[str, Any]:
    from causalcache.set_utility_throughput_pilot_envelope_v1 import (
        load_set_utility_throughput_pilot_execution_envelope_v1,
    )

    return load_set_utility_throughput_pilot_execution_envelope_v1(path, **kwargs)


def launch_from_fresh_projection_v1(
    projection: Mapping[str, Any],
    *,
    execution_envelope_path: str | Path,
    profile: str,
    worker_index: int,
    visible_gpu_uuid: str | None,
) -> ActionStabilityLaunchV1:
    """Construct a launch only from a separately validated fresh D1 envelope."""
    if not isinstance(projection, Mapping):
        raise TypeError("D1 fresh execution projection must be a mapping")
    required = {
        "repository_root",
        "source_config_path",
        "envelope_path",
        "run_root",
        "parent_envelope_path",
        "processor_root",
        "model_dir",
        "device_by_worker",
        "worker_gpu_uuid",
        "profiles",
        "validation",
    }
    if not required.issubset(projection):
        raise ValueError("D1 fresh projection is missing required fields")
    supplied_envelope = _absolute_path(
        str(execution_envelope_path), label="D1 supplied execution envelope"
    )
    projected_envelope = _absolute_path(
        projection["envelope_path"], label="D1 projected execution envelope"
    )
    repository_root = _absolute_path(
        projection["repository_root"], label="D1 repository root"
    )
    expected_source_config = repository_root / CANONICAL_D1_CONFIG_PATH
    source_config = _absolute_path(
        projection["source_config_path"], label="D1 source config"
    )
    profiles = tuple(_sequence(projection["profiles"], label="D1 profiles"))
    devices = _mapping(projection["device_by_worker"], label="D1 worker devices")
    uuids = _mapping(projection["worker_gpu_uuid"], label="D1 worker UUIDs")
    validation = _mapping(projection["validation"], label="D1 validation receipt")
    worker_key = str(worker_index)
    if (
        supplied_envelope != projected_envelope
        or source_config != expected_source_config
        or profiles != PROFILE_ORDER
        or devices.get(worker_key) != "cuda:0"
        or uuids.get(worker_key) != visible_gpu_uuid
    ):
        raise PermissionError("D1 fresh projection or isolated GPU binding drifted")
    if (
        validation.get("status") != FRESH_ENVELOPE_VALIDATION_STATUS
        or validation.get("worker_count") != WORKER_COUNT
        or validation.get("profiles") != list(PROFILE_ORDER)
        or validation.get("freshness_validated") is not True
        or validation.get("this_validated_envelope_authorizes_gpu_execution") is not True
        or validation.get("parent_envelope_sha256")
        != PARENT_EXECUTION_ENVELOPE_SHA256
    ):
        raise PermissionError("D1 fresh validation receipt is not authorizing")
    return ActionStabilityLaunchV1(
        repository_root=repository_root,
        source_config_path=CANONICAL_D1_CONFIG_PATH,
        execution_envelope_path=projected_envelope,
        parent_envelope_path=_absolute_path(
            projection["parent_envelope_path"], label="D1 parent envelope"
        ),
        processor_root=_absolute_path(
            projection["processor_root"], label="D1 processor root"
        ),
        model_dir=_absolute_path(projection["model_dir"], label="D1 model directory"),
        output_root=_absolute_path(projection["run_root"], label="D1 run root"),
        worker_index=worker_index,
        profile=profile,
        device="cuda:0",
        worker_gpu_uuid=str(visible_gpu_uuid),
        fresh_envelope_sha256=str(validation.get("envelope_sha256")),
        source_config_sha256=str(validation.get("source_config_sha256")),
        source_inventory_sha256=str(validation.get("source_inventory_sha256")),
        parent_envelope_sha256=str(validation.get("parent_envelope_sha256")),
        fresh_validation_status=str(validation.get("status")),
        freshness_validated=validation.get("freshness_validated") is True,
        gpu_execution_authorized=(
            validation.get("this_validated_envelope_authorizes_gpu_execution") is True
        ),
    )


def load_parent_artifact_binding_v1(
    launch: ActionStabilityLaunchV1,
    *,
    parent_contract_loader: ParentContractLoader = (
        load_train_only_throughput_pilot_v1_contract
    ),
    parent_envelope_loader: ParentEnvelopeLoader = _default_parent_envelope_loader,
    projection_adapter: ParentProjectionAdapter = (
        validated_execution_from_authorized_projection_v1
    ),
) -> tuple[Any, ValidatedParentArtifactsV1]:
    """Bind old artifacts with parent freshness disabled and no execution authority."""
    observed_parent_sha = _strict_file_sha256(
        launch.parent_envelope_path, label="D1 parent execution envelope"
    )
    if observed_parent_sha != PARENT_EXECUTION_ENVELOPE_SHA256:
        raise ValueError("D1 parent execution envelope bytes drifted")
    parent_contract = parent_contract_loader(
        repository_root=launch.repository_root,
        config_path=PARENT_SOURCE_CONFIG_PATH,
    )
    parent_projection = parent_envelope_loader(
        launch.parent_envelope_path,
        repository_root=launch.repository_root,
        verify_repository=False,
        verify_local_artifacts="stat",
        require_fresh_preflight=False,
        verify_current_environment=False,
    )
    if (
        parent_projection.get("processor_root") != str(launch.processor_root)
        or parent_projection.get("model_dir") != str(launch.model_dir)
    ):
        raise ValueError("D1 parent artifact paths differ from the fresh envelope")
    parent_run_root = _absolute_path(
        parent_projection.get("run_root"), label="D1 parent run root"
    )
    if parent_run_root == launch.output_root:
        raise PermissionError("D1 fresh output root must differ from the parent run")
    parent_launch = ParentExecutionLaunchV1(
        repository_root=launch.repository_root,
        config_path=PARENT_SOURCE_CONFIG_PATH,
        processor_root=launch.processor_root,
        model_dir=launch.model_dir,
        output_root=parent_run_root,
        worker_index=launch.worker_index,
        device=launch.device,
    )
    # note (luojiaxuan): The completed parent envelope records the old source-A
    # worktree path. D1 does not reuse that expired execution authorization; it
    # rebinds the byte-identical parent source contract from the fresh D1
    # repository while retaining the envelope's immutable artifact inventory.
    artifact_projection = {
        **parent_projection,
        "repository_root": str(launch.repository_root),
    }
    validated = projection_adapter(
        parent_launch,
        parent_contract,
        artifact_projection,
    )
    return parent_contract, validated


def _source_identity(contract: D1SourceContract) -> tuple[str, str]:
    if not isinstance(contract.repository_root, Path):
        raise TypeError("D1 source contract repository root is invalid")
    source = _mapping(contract.data.get("source"), label="D1 contract source")
    inventory_sha = source.get("inventory_sha256")
    if (
        not isinstance(contract.config_sha256, str)
        or _SHA256.fullmatch(contract.config_sha256) is None
        or not isinstance(inventory_sha, str)
        or _SHA256.fullmatch(inventory_sha) is None
    ):
        raise ValueError("D1 source contract identity is invalid")
    return contract.config_sha256, inventory_sha


def _filter_d1_states(
    parent_states: Sequence[Any],
    *,
    worker_index: int,
) -> tuple[Any, ...]:
    expected = state_ids_for_worker_v1(worker_index)
    by_state: dict[str, Any] = {}
    for state in parent_states:
        state_id = getattr(state, "state_id", None)
        if not isinstance(state_id, str) or state_id in by_state:
            raise ValueError("D1 parent semantic state inventory is malformed")
        by_state[state_id] = state
    selected = tuple(by_state[state_id] for state_id in expected if state_id in by_state)
    if tuple(getattr(state, "state_id", None) for state in selected) != expected:
        raise ValueError("D1 six-state roster is missing from parent semantic inputs")
    return selected


def _validate_partial_result(
    value: Mapping[str, Any],
    *,
    profile: str,
    state_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PARTIAL_KEYS:
        raise ValueError("D1 profile partial fields drifted")
    if (
        value.get("metric_safe") is not True
        or value.get("profile") != _PROFILE_RESULT_ID[profile]
        or value.get("state_id") != state_id
    ):
        raise ValueError("D1 profile partial identity drifted")
    raw_conditions = _sequence(value.get("conditions"), label="D1 partial conditions")
    expected_conditions = _PROFILE_CONDITIONS[profile]
    if len(raw_conditions) != len(expected_conditions):
        raise ValueError("D1 profile partial condition count drifted")
    conditions = [
        _validate_condition_payload(raw, expected_condition_id=condition_id)
        for raw, condition_id in zip(raw_conditions, expected_conditions, strict=True)
    ]
    projected = {
        "conditions": conditions,
        "metric_safe": True,
        "profile": _PROFILE_RESULT_ID[profile],
        "state_id": state_id,
    }
    _validate_metric_safe_tree(projected)
    canonical_json_bytes(projected)
    return projected


def require_auto_stage_complete_v1(launch: ActionStabilityLaunchV1) -> None:
    """Fail closed unless all four auto-profile workers completed successfully."""
    if not isinstance(launch, ActionStabilityLaunchV1):
        raise TypeError("D1 auto-stage gate requires one launch")
    if launch.profile != PROFILE_EAGER:
        return
    for worker_index in range(WORKER_COUNT):
        state_ids = state_ids_for_worker_v1(worker_index)
        worker_root = launch.output_root / "workers" / f"worker-{worker_index:02d}"
        terminal_path = worker_root / f"{PROFILE_AUTO}-terminal.json"
        payload = _strict_regular_file_payload(
            terminal_path,
            label=f"D1 auto terminal {worker_index}",
        )
        terminal = _strict_json_object_bytes(
            payload,
            label=f"D1 auto terminal {worker_index}",
        )
        if payload != canonical_pretty_json_bytes(terminal) or set(terminal) != _TERMINAL_KEYS:
            raise PermissionError("D1 eager stage requires canonical auto terminals")
        if (
            terminal.get("status")
            != "COMPLETED_ACTION_STABILITY_PROFILE_WORKER_V1"
            or terminal.get("failure_class") is not None
            or terminal.get("metric_safe") is not True
            or terminal.get("profile") != PROFILE_AUTO
            or terminal.get("protocol_id") != TERMINAL_PROTOCOL_ID
            or terminal.get("schema_version") != SCHEMA_VERSION
            or terminal.get("worker_index") != worker_index
            or terminal.get("state_ids") != list(state_ids)
            or terminal.get("fresh_execution_envelope_sha256")
            != launch.fresh_envelope_sha256
            or terminal.get("parent_execution_envelope_sha256")
            != launch.parent_envelope_sha256
            or terminal.get("source_config_sha256") != launch.source_config_sha256
            or terminal.get("source_inventory_sha256")
            != launch.source_inventory_sha256
            or not isinstance(terminal.get("runtime_identity_sha256"), str)
            or _SHA256.fullmatch(str(terminal.get("runtime_identity_sha256"))) is None
            or not isinstance(terminal.get("attempt_sha256"), str)
            or _SHA256.fullmatch(str(terminal.get("attempt_sha256"))) is None
        ):
            raise PermissionError("D1 eager stage auto-terminal identity drifted")
        partials = _sequence(
            terminal.get("partial_results"),
            label="D1 auto terminal partial results",
        )
        if len(partials) != len(state_ids):
            raise PermissionError("D1 eager stage auto-terminal roster is incomplete")
        validated = [
            _validate_partial_result(raw, profile=PROFILE_AUTO, state_id=state_id)
            for raw, state_id in zip(partials, state_ids, strict=True)
        ]
        generation_calls = sum(
            int(condition["generation_call_count"])
            for partial in validated
            for condition in partial["conditions"]
        )
        encode_calls = sum(
            int(condition["encode_call_count"])
            for partial in validated
            for condition in partial["conditions"]
        )
        expected_counts = {
            "condition_completed_count": len(state_ids) * 2,
            "encode_call_count": encode_calls,
            "generation_call_count": generation_calls,
            "label_count": 0,
            "partial_state_completed_count": len(state_ids),
            "partial_state_expected_count": len(state_ids),
            "restoration_distance_count": 0,
            "retry_count": 0,
            "teacher_forward_call_count": 0,
            "training_example_count": 0,
        }
        if terminal.get("counts") != expected_counts:
            raise PermissionError("D1 eager stage auto-terminal counts drifted")
        attempt_path = worker_root / f"{PROFILE_AUTO}-attempt.json"
        if _strict_file_sha256(
            attempt_path,
            label=f"D1 auto attempt {worker_index}",
        ) != terminal["attempt_sha256"]:
            raise PermissionError("D1 eager stage auto attempt binding drifted")
        _validate_metric_safe_tree(terminal)


def build_production_runtime_bundle_v1(
    launch: ActionStabilityLaunchV1,
    validated: ValidatedParentArtifactsV1,
    parent_contract: Any,
) -> ActionStabilityRuntimeBundleV1:
    """Construct exactly one auto or eager runtime in this profile process."""
    from PIL import Image

    from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v1 import (
        GUIOwlV21AutoActionStabilityRuntimeV1,
        GUIOwlV21EagerActionStabilityRuntimeV1,
    )
    from causalcache.policy.gui_owl_v2_runtime import (
        FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    from causalcache.set_utility_gui_owl_v2_1_throughput_adapter import (
        build_gui_owl_v2_1_throughput_reference_input,
    )

    inputs = _mapping(parent_contract.data.get("inputs"), label="parent inputs")
    model_binding = _mapping(
        inputs.get("model_snapshot_manifest"), label="parent model binding"
    )
    runtime_class = (
        GUIOwlV21AutoActionStabilityRuntimeV1
        if launch.profile == PROFILE_AUTO
        else GUIOwlV21EagerActionStabilityRuntimeV1
    )
    runtime = runtime_class(
        model_dir=launch.model_dir,
        expected_snapshot_manifest=(
            launch.repository_root / str(model_binding["path"])
        ),
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
            joined,
            plan,
            image_decoder=decode_rgb,
        )

    identity = {
        "device": launch.device,
        "model_inventory_sha256": validated.model_inventory_sha256,
        "profile": launch.profile,
        "runtime_metadata_sha256": sha256_bytes(
            canonical_json_bytes(dict(runtime.metadata))
        ),
    }
    return ActionStabilityRuntimeBundleV1(
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
        raise ValueError("D1 output parent must be one real directory")
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
                raise OSError("D1 exclusive write made no progress")
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


def _attempt_payload(
    launch: ActionStabilityLaunchV1,
    *,
    source_config_sha256: str,
    source_inventory_sha256: str,
    validated: ValidatedParentArtifactsV1,
) -> dict[str, Any]:
    return {
        "candidate_schedule_sha256": sha256_bytes(
            validated.candidate_schedule_bytes
        ),
        "device_id": launch.device,
        "fresh_execution_envelope_sha256": launch.fresh_envelope_sha256,
        "fresh_execution_validation_status": launch.fresh_validation_status,
        "freeze_manifest_sha256": sha256_bytes(validated.freeze_manifest_bytes),
        "metric_safe": True,
        "model_inventory_sha256": validated.model_inventory_sha256,
        "parent_envelope_role": "artifact_binding_only_not_execution_authorization",
        "parent_execution_envelope_sha256": launch.parent_envelope_sha256,
        "parent_preflight_freshness_required": False,
        "processor_inventory_sha256": validated.processor_inventory_sha256,
        "profile": launch.profile,
        "protocol_id": EXECUTION_PROTOCOL_ID,
        "retry_count": 0,
        "schema_version": SCHEMA_VERSION,
        "source_config_sha256": source_config_sha256,
        "source_inventory_sha256": source_inventory_sha256,
        "state_ids": list(state_ids_for_worker_v1(launch.worker_index)),
        "status": "CLAIMED_NO_RETRY_ACTION_STABILITY_PROFILE_WORKER_V1",
        "worker_index": launch.worker_index,
    }


def run_action_stability_worker_v1(
    launch: ActionStabilityLaunchV1,
    *,
    source_contract_loader: SourceContractLoader = _default_source_contract_loader,
    parent_binding_loader: ParentBindingLoader | None = None,
    semantic_loader: SemanticLoader = load_worker_semantic_inputs_v1,
    runtime_factory: RuntimeFactory = build_production_runtime_bundle_v1,
    profile_runner: ProfileRunner | None = None,
    stage_gate: StageGate = require_auto_stage_complete_v1,
) -> dict[str, Any]:
    """Claim once, run one profile over this worker's D1 subset, and terminate."""
    if not isinstance(launch, ActionStabilityLaunchV1):
        raise TypeError("D1 launch has the wrong type")
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
        raise PermissionError("D1 fresh envelope differs from the source contract")
    stage_gate(launch)
    selected_parent_loader = parent_binding_loader
    if selected_parent_loader is None:
        selected_parent_loader = load_parent_artifact_binding_v1
    parent_contract, validated = selected_parent_loader(launch)

    worker_root = launch.output_root / "workers" / f"worker-{launch.worker_index:02d}"
    attempt_path = worker_root / f"{launch.profile}-attempt.json"
    terminal_path = worker_root / f"{launch.profile}-terminal.json"
    attempt = _attempt_payload(
        launch,
        source_config_sha256=config_sha,
        source_inventory_sha256=inventory_sha,
        validated=validated,
    )
    attempt_bytes = canonical_pretty_json_bytes(attempt)
    _write_exclusive(attempt_path, attempt_bytes)
    attempt_sha = sha256_bytes(attempt_bytes)

    partial_results: list[dict[str, Any]] = []
    runtime_identity_sha: str | None = None
    failure_class: str | None = None
    expected_states = state_ids_for_worker_v1(launch.worker_index)
    try:
        parent_states = semantic_loader(validated, parent_contract)
        states = _filter_d1_states(parent_states, worker_index=launch.worker_index)
        bundle = runtime_factory(launch, validated, parent_contract)
        runtime_identity_sha = bundle.runtime_identity_sha256
        runner = profile_runner
        if runner is None:
            runner = (
                run_default_conditions_v1
                if launch.profile == PROFILE_AUTO
                else run_eager_condition_v1
            )
        for state in states:
            raw = runner(
                state.query,
                reference_input_builder=(
                    bundle.reference_input_builder_for(state.joined_input)
                ),
                runtime=bundle.runtime,
            )
            partial_results.append(
                _validate_partial_result(
                    raw,
                    profile=launch.profile,
                    state_id=state.state_id,
                )
            )
    except Exception as error:
        failure_class = _safe_failure(error)

    condition_count = sum(len(result["conditions"]) for result in partial_results)
    generation_calls = sum(
        int(condition["generation_call_count"])
        for result in partial_results
        for condition in result["conditions"]
    )
    encode_calls = sum(
        int(condition["encode_call_count"])
        for result in partial_results
        for condition in result["conditions"]
    )
    completed = failure_class is None and len(partial_results) == len(expected_states)
    terminal = {
        "attempt_sha256": attempt_sha,
        "counts": {
            "condition_completed_count": condition_count,
            "encode_call_count": encode_calls,
            "generation_call_count": generation_calls,
            "label_count": 0,
            "partial_state_completed_count": len(partial_results),
            "partial_state_expected_count": len(expected_states),
            "restoration_distance_count": 0,
            "retry_count": 0,
            "teacher_forward_call_count": 0,
            "training_example_count": 0,
        },
        "failure_class": failure_class,
        "fresh_execution_envelope_sha256": launch.fresh_envelope_sha256,
        "metric_safe": True,
        "parent_execution_envelope_sha256": launch.parent_envelope_sha256,
        "partial_results": partial_results,
        "profile": launch.profile,
        "protocol_id": TERMINAL_PROTOCOL_ID,
        "runtime_identity_sha256": runtime_identity_sha,
        "schema_version": SCHEMA_VERSION,
        "source_config_sha256": config_sha,
        "source_inventory_sha256": inventory_sha,
        "state_ids": list(expected_states),
        "status": (
            "COMPLETED_ACTION_STABILITY_PROFILE_WORKER_V1"
            if completed
            else "FAILED_ACTION_STABILITY_PROFILE_WORKER_V1"
        ),
        "worker_index": launch.worker_index,
    }
    _validate_metric_safe_tree(terminal)
    terminal_bytes = canonical_pretty_json_bytes(terminal)
    _write_atomic_no_clobber(terminal_path, terminal_bytes)
    return terminal


__all__ = [
    "ActionStabilityLaunchV1",
    "ActionStabilityRuntimeBundleV1",
    "CANONICAL_D1_CONFIG_PATH",
    "EXECUTION_PROTOCOL_ID",
    "FRESH_ENVELOPE_VALIDATION_STATUS",
    "PARENT_EXECUTION_ENVELOPE_SHA256",
    "PROFILE_AUTO",
    "PROFILE_EAGER",
    "PROFILE_ORDER",
    "SCHEMA_VERSION",
    "TERMINAL_PROTOCOL_ID",
    "build_production_runtime_bundle_v1",
    "launch_from_fresh_projection_v1",
    "load_parent_artifact_binding_v1",
    "require_auto_stage_complete_v1",
    "run_action_stability_worker_v1",
]
