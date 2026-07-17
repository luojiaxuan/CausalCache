"""Run the frozen 192-state label-expansion substrate on two H200 workers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.data.guiodyssey_restoration_v2 import (
    canonical_json_bytes,
    parse_canonical_jsonl,
    sha256_bytes,
    sha256_file,
)
from causalcache.data.guiodyssey_restoration_v2_expansion import (
    ARTIFACT_RELATIVE_PATHS,
    IMAGE_TAR_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    TRAJECTORY_JSONL_RELATIVE_PATH,
    artifact_tree_identity,
    validate_artifact,
)
from causalcache.policy.gui_owl_v2 import gui_owl_v2_action_to_androidworld
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_MOBILE_USE_TOOL,
    build_gui_owl_v2_1_mixed_fidelity_messages,
    parse_gui_owl_v2_1_output,
    validate_gui_owl_v2_1_native_messages,
)
from causalcache.restoration_v2_2_expansion_substrate_artifact import (
    AGGREGATE_FILENAME,
    ATTEMPT_ID,
    ATTEMPT_DIRECTORY,
    CANARY_ABORT_FILENAME as PROCESSOR_CANARY_ABORT_FILENAME,
    CANARY_RELEASE_FILENAME as PROCESSOR_CANARY_RELEASE_FILENAME,
    CANONICAL_LEDGER_PATH,
    CANONICAL_OUTPUT_DIR,
    CANONICAL_RUNNER_FREEZE_PATH,
    EXECUTION_EVIDENCE_FILENAME,
    EXPECTED_STATE_COUNT,
    FORBIDDEN_COUNT_KEYS,
    INVALID_OUTCOME,
    LOG_PATHS,
    MONITOR_SUMMARY_PATH,
    NO_GO_OUTCOME,
    PASS_OUTCOME,
    PER_STATE_PLANNED_COUNTS,
    PLANNED_COUNTS,
    PROCESSOR_IDENTITY_KEYS,
    RUN_MANIFEST_FILENAME,
    RUNTIME_BARRIER_ABORT_FILENAME,
    RUNTIME_BARRIER_RELEASE_FILENAME,
    SCHEMA_VERSION,
    STATE_DIRECTORY,
    STATE_OUTCOME_FAILED,
    STATE_OUTCOME_VALID,
    WORKER_CANARY_FILENAME as PROCESSOR_CANARY_FILENAME,
    WORKER_DIRECTORY,
    WORKER_LEDGER_FILENAME,
    WORKER_OUTCOME_COMPLETE,
    WORKER_OUTCOME_INVALID,
    WORKER_RUNTIME_FILENAME,
    WORKER_TERMINAL_FILENAME,
    WorkerSpec,
    _validate_monitor_gpu_uuid_binding,
    expected_worker_specs,
    load_and_validate_runner_freeze,
    reduce_expansion_substrate_gate,
    validate_execution_run_contract,
)
from causalcache.restoration_v2_2_expansion_substrate_contract import (
    CANONICAL_COMPLETION_PATH,
    CANONICAL_CONFIG_PATH,
    PROTOCOL_ID,
    load_and_validate_contract,
    load_json_object,
    pretty_json_bytes,
)
from causalcache.restoration_v2_2_expansion_substrate_inputs import (
    build_decision_view_input,
    validate_processor_byte_canary,
    validate_request_manifest,
)
from causalcache.restoration_v2_2_eager_artifact import (
    FORBIDDEN_OPERATION_COUNTS as EAGER_FORBIDDEN_OPERATION_COUNTS,
    MEASUREMENT_KERNEL_PROTOCOL_ID,
    MEASUREMENT_KERNEL_SCHEMA_VERSION,
)
from causalcache.restoration_v2_text_backend import load_backend_config
from scripts.run_restoration_v2_1_full_45_substrate import (
    GPUFullVocabularyKLBackend,
    _action_arguments,
    _decode_rgb_image,
    _failure,
    _screen_dimensions,
    _validate_generation_metadata,
)
from scripts.run_restoration_v2_1_interface_pilot import (
    validate_clean_pushed_main,
    validate_committed_source_blobs,
)


RUNNER_PROTOCOL_ID = PROTOCOL_ID
RUNNER_SOURCE_PATH = "code/scripts/run_restoration_v2_2_expansion_substrate.py"
CANONICAL_OCR_BACKEND_CONFIG_PATH = "code/configs/restoration_v2_ocr_backend.json"
CANONICAL_OCR_BACKEND_MANIFEST_PATH = (
    "data/manifests/restoration_v2_ocr_backend.json"
)
CANONICAL_SNAPSHOT_MANIFEST_PATH = "code/configs/gui_owl_1_5_8b_snapshot.json"
CANONICAL_MODEL_DIR = Path("/data/artifacts/models/GUI-Owl-1.5-8B-Instruct")
CANONICAL_IMAGE_DIGEST = (
    "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
)
PLANNED_OPERATION_COUNTS = dict(PLANNED_COUNTS)
PER_STATE_PLANNED_OPERATION_COUNTS = dict(PER_STATE_PLANNED_COUNTS)
FORBIDDEN_OPERATION_COUNTS = {key: 0 for key in sorted(FORBIDDEN_COUNT_KEYS)}
MEASUREMENT_FORBIDDEN_OPERATION_COUNTS = {
    key: value
    for key, value in EAGER_FORBIDDEN_OPERATION_COUNTS.items()
    if key not in {"gate_model_forward_count", "gate_selection_count"}
}
MEASUREMENT_STATE_OUTCOME_VALID = "VALID_V2_1_FULL_45_SUBSTRATE_STATE"
MEASUREMENT_STATE_OUTCOME_FAILED = "FAILED_V2_1_FULL_45_SUBSTRATE_STATE"
GLOBAL_CLAIM_STATUS = "GLOBAL_ATTEMPT_CLAIMED_BEFORE_WORKER_SPAWN"
MONITOR_SIDECAR_SUBCOMMAND = "monitor-sidecar"
MONITOR_SIDECAR_REQUIRED_OPTIONS = (
    "--ready-file",
    "--log-file",
    "--summary-file",
)
MONITOR_STOP_REQUEST_STATUS = "EXPANSION_SUBSTRATE_MONITOR_STOP_REQUEST"
MONITOR_READY_STATUS = "GPU_UTILIZATION_MONITOR_READY"
MONITOR_SAMPLE_STATUS = "GPU_UTILIZATION_SAMPLE"
MONITOR_SUMMARY_STATUS = "GPU_UTILIZATION_MONITOR_SUMMARY"
MONITOR_SAMPLE_INTERVAL_SECONDS = 1.0
MONITOR_STOP_POLL_SECONDS = 0.25
MONITOR_STOP_REQUEST_KEYS = frozenset(
    {"schema_version", "status", "run_contract_sha256", "requested_at_utc"}
)
MONITOR_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "run_contract_sha256",
        "container_name_prefix",
        "minimum_gpu_utilization_percent",
        "started_before_first_generation",
        "started_at_utc",
        "stopped_at_utc",
        "sample_count",
        "low_utilization_incident_count",
        "visible_gpu_count",
        "visible_gpu_uuids",
        "stop_request_sha256",
    }
)


@dataclass(frozen=True)
class ExpansionState:
    index: int
    role: str
    source_id: str
    decision: Mapping[str, Any]

    @property
    def state_id(self) -> str:
        return str(self.decision["state_id"])

    @property
    def decision_step_id(self) -> int:
        return int(self.decision["decision_step_id"])

    @property
    def candidate_event_step_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.decision["candidate_event_step_ids"])


@dataclass(frozen=True)
class ExpansionArtifact:
    root: Path
    tree: Mapping[str, Any]
    validation: Mapping[str, Any]
    trajectories: tuple[Mapping[str, Any], ...]
    states: tuple[ExpansionState, ...]
    image_payloads: Mapping[str, bytes]

    def trajectory(self, state: ExpansionState) -> Mapping[str, Any]:
        if state not in self.states:
            raise PermissionError("state is outside the frozen expansion denominator")
        matches = [
            record for record in self.trajectories if record.get("source_id") == state.source_id
        ]
        if len(matches) != 1:
            raise ValueError("expansion state does not bind exactly one trajectory")
        return matches[0]

    def image_bytes(self, member_path: str) -> bytes:
        try:
            return self.image_payloads[member_path]
        except KeyError as error:
            raise ValueError("image is outside the immutable expansion projection") from error


@dataclass(frozen=True)
class MessageBundle:
    messages: list[dict[str, Any]]
    request_manifest: Mapping[str, Any]
    slice_witness_sha256: str
    included_events: tuple[Mapping[str, Any], ...]
    restored_event_step_ids: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeBindings:
    runtime: Any
    parse_error_class: type[BaseException]
    distance_backend: Any
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class AuthorizedExpansion:
    repository_root: Path
    frozen_config: Mapping[str, Any]
    contract_validation: Mapping[str, Any]
    git_identity: Mapping[str, Any]
    source_inventory: tuple[Mapping[str, Any], ...]
    completion_manifest: Mapping[str, Any]
    artifact: ExpansionArtifact
    snapshot_manifest_path: Path
    model_identity: Mapping[str, Any]
    canonical_inputs: Mapping[str, Any]


@dataclass(frozen=True)
class ExpansionLayout:
    root: Path
    global_ledger: Path
    manifest: Path
    aggregate: Path
    execution_evidence: Path
    run_contract: Mapping[str, Any]
    run_contract_sha256: str
    states: tuple[ExpansionState, ...]
    projections: tuple[Mapping[str, Any], ...]
    started_at_utc: str
    worker_sibling_ledgers: Mapping[str, Path]
    publish_staging: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required JSON evidence is missing: {path}")
    pairs: list[tuple[str, Any]]

    def unique_object(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(
        path.read_bytes(),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write_json_exclusive(
    path: Path,
    value: Mapping[str, Any],
    *,
    staging_directory: Path,
) -> None:
    _publish_json_exclusive_atomic(
        path,
        value,
        staging_directory=staging_directory,
    )


def _publish_json_exclusive_atomic(
    path: Path,
    value: Mapping[str, Any],
    *,
    staging_directory: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not staging_directory.is_dir():
        raise FileNotFoundError("external atomic-publication staging directory vanished")
    temporary = staging_directory / f"{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as destination:
            destination.write(pretty_json_bytes(dict(value)))
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_json_durable(
    path: Path,
    value: Mapping[str, Any],
    *,
    staging_directory: Path,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"durable evidence disappeared: {path}")
    if not staging_directory.is_dir():
        raise FileNotFoundError("external durable-replacement staging directory vanished")
    temporary = staging_directory / f"{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as destination:
            destination.write(pretty_json_bytes(dict(value)))
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def validate_worker_topology(value: Mapping[str, Any]) -> None:
    expected = [spec.to_dict() for spec in expected_worker_specs()]
    if (
        value.get("worker_count") != 2
        or value.get("gpu_model") != "NVIDIA H200"
        or value.get("one_process_per_device") is not True
        or value.get("workers") != expected
        or value.get("cross_worker_state_stealing_allowed") is not False
        or value.get("worker_failure_invalidates_entire_attempt") is not True
        or value.get("worker_outputs_must_be_disjoint") is not True
        or value.get("worker_union_must_equal_fixed_denominator") is not True
    ):
        raise ValueError("two-worker exact parity topology drifted")


def _safe_tar_images(path: Path) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("expansion image tar is not sorted and unique")
        for member in members:
            parsed = PurePosixPath(member.name)
            if (
                parsed.is_absolute()
                or parsed.as_posix() != member.name
                or any(part in {"", ".", ".."} for part in parsed.parts)
                or not member.name.startswith("images/")
                or not member.isfile()
                or member.type != tarfile.REGTYPE
            ):
                raise ValueError("expansion image tar contains an unsafe member")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("expansion image tar member is unreadable")
            payload = handle.read()
            if not payload or len(payload) != member.size:
                raise ValueError("expansion image tar member bytes drifted")
            payloads[member.name] = payload
    return payloads


def _state_projection(state: ExpansionState) -> dict[str, Any]:
    decision = state.decision
    return {
        "state_index": state.index,
        "role": state.role,
        "source_id": state.source_id,
        "state_id": state.state_id,
        "decision_step_id": state.decision_step_id,
        "history_event_step_ids": list(decision["history_event_step_ids"]),
        "candidate_event_step_ids": list(decision["candidate_event_step_ids"]),
        "current_equivalent_event_step_id": int(
            decision["current_equivalent_event_step_id"]
        ),
    }


def load_immutable_expansion_artifact(
    *,
    artifact_root: str | Path,
    frozen_config: Mapping[str, Any],
    backend_config_path: str | Path,
    derived_repo: str,
    derived_revision: str,
) -> ExpansionArtifact:
    supplied_root = Path(artifact_root)
    if supplied_root.is_symlink():
        raise ValueError("derived artifact root cannot be a symlink")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise ValueError("derived artifact root must be one real directory")
    expected = frozen_config["immutable_inputs"]["derived_artifact"]
    if derived_repo != expected["repo"] or derived_revision != expected["immutable_revision"]:
        raise ValueError("explicit derived Hugging Face identity drifted")
    entries = list(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("derived artifact clean projection cannot contain symlinks")
    if any(not path.is_file() and not path.is_dir() for path in entries):
        raise ValueError("derived artifact clean projection has a special entry")
    observed_paths = sorted(
        path.relative_to(root).as_posix() for path in entries if path.is_file()
    )
    if observed_paths != sorted(ARTIFACT_RELATIVE_PATHS):
        raise ValueError("derived artifact must be the clean exact-six projection")
    allowed_directories = {
        PurePosixPath(path).parent.as_posix()
        for path in ARTIFACT_RELATIVE_PATHS
        if PurePosixPath(path).parent.as_posix() != "."
    }
    allowed_directories.update(
        parent.as_posix()
        for directory in tuple(allowed_directories)
        for parent in PurePosixPath(directory).parents
        if parent.as_posix() != "."
    )
    observed_directories = {
        path.relative_to(root).as_posix() for path in entries if path.is_dir()
    }
    if observed_directories != allowed_directories:
        raise ValueError("derived artifact clean projection has unexpected directories")
    tree = artifact_tree_identity(root)
    file_count = len(tree["files"])
    total_bytes = sum(int(record["size_bytes"]) for record in tree["files"])
    if (
        tree.get("artifact_tree_sha256") != expected["artifact_tree_sha256"]
        or file_count != expected["artifact_file_count"]
        or total_bytes != expected["artifact_total_bytes"]
        or tree.get("files") != expected["files"]
    ):
        raise ValueError("derived artifact bytes differ from the immutable HF revision")
    backend_path = Path(backend_config_path).resolve()
    backend = load_backend_config(backend_path)
    validation = validate_artifact(
        output_dir=root,
        backend_config=backend,
        backend_config_sha256=sha256_file(backend_path),
        expected_dataset_repo=derived_repo,
        require_formal=False,
        require_ocr_replay=False,
    )
    trajectories = tuple(
        parse_canonical_jsonl(
            (root / TRAJECTORY_JSONL_RELATIVE_PATH).read_bytes(),
            label="expansion substrate trajectories",
        )
    )
    roles = [record.get("role") for record in trajectories]
    if roles != ["gate_train_expansion"] * 48 + ["gate_development_expansion"] * 16:
        raise PermissionError("derived trajectory role order drifted")
    states: list[ExpansionState] = []
    for trajectory in trajectories:
        decisions = trajectory.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != 3:
            raise ValueError("each expansion trajectory must expose exactly three states")
        for decision in decisions:
            if not isinstance(decision, Mapping):
                raise ValueError("expansion decision must be an object")
            states.append(
                ExpansionState(
                    index=len(states),
                    role=str(trajectory["role"]),
                    source_id=str(trajectory["source_id"]),
                    decision=decision,
                )
            )
    projections = [_state_projection(state) for state in states]
    if (
        len(trajectories) != 64
        or len(states) != EXPECTED_STATE_COUNT
        or sha256_bytes(canonical_json_bytes(projections))
        != frozen_config["data_projection"]["state_projection_sha256"]
    ):
        raise ValueError("derived state projection differs from the frozen denominator")
    image_payloads = _safe_tar_images(root / IMAGE_TAR_RELATIVE_PATH)
    if len(image_payloads) != 384:
        raise ValueError("derived expansion image denominator drifted")
    return ExpansionArtifact(
        root=root,
        tree=dict(tree),
        validation=dict(validation),
        trajectories=trajectories,
        states=tuple(states),
        image_payloads=image_payloads,
    )


def build_expansion_messages(
    artifact: ExpansionArtifact,
    state: ExpansionState,
    fidelity: str,
    image_decoder: Callable[[bytes], Any],
    *,
    trajectory_override: Mapping[str, Any] | None = None,
) -> MessageBundle:
    trajectory = (
        trajectory_override if trajectory_override is not None else artifact.trajectory(state)
    )
    decision = state.decision
    view = build_decision_view_input(trajectory, decision)
    included_events = tuple(view["events"])
    validate_request_manifest(
        view["request_manifest"],
        decision,
        included_events=included_events,
    )
    if fidelity == "reference":
        restored = state.candidate_event_step_ids
    elif fidelity == "summary_only":
        restored = ()
    else:
        raise ValueError("expansion fidelity must be reference or summary_only")
    messages = _messages_from_decision_view(
        artifact=artifact,
        state=state,
        trajectory=trajectory,
        included_events=included_events,
        restored_event_step_ids=restored,
        image_decoder=image_decoder,
    )
    validate_gui_owl_v2_1_native_messages(messages)
    return MessageBundle(
        messages=messages,
        request_manifest=dict(view["request_manifest"]),
        slice_witness_sha256=str(view["slice_witness_sha256"]),
        included_events=included_events,
        restored_event_step_ids=tuple(restored),
    )


def _messages_from_decision_view(
    *,
    artifact: ExpansionArtifact,
    state: ExpansionState,
    trajectory: Mapping[str, Any],
    included_events: Sequence[Mapping[str, Any]],
    restored_event_step_ids: Sequence[int],
    image_decoder: Callable[[bytes], Any],
) -> list[dict[str, Any]]:
    sliced_trajectory = dict(trajectory)
    sliced_trajectory["events"] = list(included_events)
    sliced_trajectory["decisions"] = [dict(state.decision)]
    messages = build_gui_owl_v2_1_mixed_fidelity_messages(
        {"trajectories": [sliced_trajectory]},
        trajectory_id=state.source_id,
        decision_step_id=state.decision_step_id,
        restored_event_step_ids=tuple(restored_event_step_ids),
        image_bytes_loader=artifact.image_bytes,
        image_decoder=image_decoder,
    )
    validate_gui_owl_v2_1_native_messages(messages)
    return messages


def build_expansion_message_pair(
    artifact: ExpansionArtifact,
    state: ExpansionState,
    image_decoder: Callable[[bytes], Any],
) -> tuple[MessageBundle, MessageBundle]:
    """Build reference and summary requests from one validated decision slice."""
    trajectory = artifact.trajectory(state)
    decision = state.decision
    view = build_decision_view_input(trajectory, decision)
    included_events = tuple(view["events"])
    validate_request_manifest(
        view["request_manifest"],
        decision,
        included_events=included_events,
    )
    common = {
        "request_manifest": dict(view["request_manifest"]),
        "slice_witness_sha256": str(view["slice_witness_sha256"]),
        "included_events": included_events,
    }
    reference = MessageBundle(
        messages=_messages_from_decision_view(
            artifact=artifact,
            state=state,
            trajectory=trajectory,
            included_events=included_events,
            restored_event_step_ids=state.candidate_event_step_ids,
            image_decoder=image_decoder,
        ),
        restored_event_step_ids=state.candidate_event_step_ids,
        **common,
    )
    summary = MessageBundle(
        messages=_messages_from_decision_view(
            artifact=artifact,
            state=state,
            trajectory=trajectory,
            included_events=included_events,
            restored_event_step_ids=(),
            image_decoder=image_decoder,
        ),
        restored_event_step_ids=(),
        **common,
    )
    return reference, summary


def _tensor_list(value: Any, *, name: str) -> list[Any]:
    try:
        detached = value.detach().to(device="cpu")
        result = detached.tolist()
    except (AttributeError, TypeError, RuntimeError) as error:
        raise TypeError(f"processor {name} must be a tensor") from error
    if not isinstance(result, list):
        raise TypeError(f"processor {name} tensor must serialize to a list")
    return result


def _tensor_byte_sha256(value: Any, *, name: str, torch_module: Any) -> str:
    try:
        cpu = value.detach().contiguous().to(device="cpu")
        try:
            payload = cpu.numpy().tobytes(order="C")
        except TypeError:
            payload = cpu.view(torch_module.uint8).numpy().tobytes(order="C")
    except (AttributeError, TypeError, RuntimeError, ValueError) as error:
        raise TypeError(f"processor {name} must expose contiguous tensor bytes") from error
    return sha256_bytes(payload)


def processor_request_identity(runtime: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    validate_gui_owl_v2_1_native_messages(messages)
    rendered_value = runtime.processor.apply_chat_template(
        [messages],
        tools=[copy.deepcopy(GUI_OWL_V2_1_MOBILE_USE_TOOL)],
        tokenize=False,
        add_generation_prompt=True,
    )
    if (
        not isinstance(rendered_value, list)
        or len(rendered_value) != 1
        or not isinstance(rendered_value[0], str)
    ):
        raise ValueError("actual processor did not return one rendered prompt")
    model_inputs, image_counts = runtime._encode_exact_batch((messages,))
    if tuple(image_counts) != (
        validate_gui_owl_v2_1_native_messages(messages),
    ):
        raise ValueError("actual processor image count drifted")
    tensor_records: dict[str, Any] = {}
    for name in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        if name not in model_inputs:
            raise ValueError(f"actual processor request lacks {name}")
        if name == "pixel_values":
            digest = _tensor_byte_sha256(
                model_inputs[name],
                name=name,
                torch_module=runtime.torch,
            )
        else:
            values = _tensor_list(model_inputs[name], name=name)
            digest = sha256_bytes(canonical_json_bytes(values))
        tensor_records[name] = {
            "shape": list(model_inputs[name].shape),
            "dtype": str(model_inputs[name].dtype),
            "sha256": digest,
        }
    identity = {
        "processor_path": "GUIOwlV22EagerRuntime._encode_exact_batch",
        "rendered_prompt_sha256": sha256_bytes(rendered_value[0].encode("utf-8")),
        "input_ids_sha256": tensor_records["input_ids"]["sha256"],
        "attention_mask_sha256": tensor_records["attention_mask"]["sha256"],
        "pixel_values_sha256": tensor_records["pixel_values"]["sha256"],
        "image_grid_thw_sha256": tensor_records["image_grid_thw"]["sha256"],
        "input_ids_shape": tensor_records["input_ids"]["shape"],
        "input_ids_dtype": tensor_records["input_ids"]["dtype"],
        "attention_mask_shape": tensor_records["attention_mask"]["shape"],
        "attention_mask_dtype": tensor_records["attention_mask"]["dtype"],
        "pixel_values_shape": tensor_records["pixel_values"]["shape"],
        "pixel_values_dtype": tensor_records["pixel_values"]["dtype"],
        "image_grid_thw_shape": tensor_records["image_grid_thw"]["shape"],
        "image_grid_thw_dtype": tensor_records["image_grid_thw"]["dtype"],
        "image_count": image_counts[0],
        "policy_forward_executed": False,
    }
    identity["processor_request_sha256"] = sha256_bytes(canonical_json_bytes(identity))
    return identity


def run_state_processor_canary(
    *,
    artifact: ExpansionArtifact,
    state: ExpansionState,
    runtime: Any,
    image_decoder: Callable[[bytes], Any] = _decode_rgb_image,
) -> dict[str, Any]:
    trajectory = artifact.trajectory(state)
    identities: list[dict[str, Any]] = []

    def serialize_request(
        candidate_trajectory: Mapping[str, Any],
        candidate_decision: Mapping[str, Any],
    ) -> bytes:
        if dict(candidate_decision) != dict(state.decision):
            raise ValueError("processor canary decision drifted")
        bundle = build_expansion_messages(
            artifact,
            state,
            "reference",
            image_decoder,
            trajectory_override=candidate_trajectory,
        )
        identity = processor_request_identity(runtime, bundle.messages)
        identities.append(identity)
        return canonical_json_bytes(identity)

    canary = validate_processor_byte_canary(
        trajectory,
        state.decision,
        serialize_request=serialize_request,
    )
    excluded = list(canary["excluded_current_or_future_action_canary_results"])
    if len(identities) != 2 + len(excluded):
        raise RuntimeError("processor canary identity inventory drifted")
    base, included, *excluded_identities = identities
    return {
        "state_index": state.index,
        "state_id": state.state_id,
        "worker_parity": state.index % 2,
        "request_manifest": build_expansion_messages(
            artifact,
            state,
            "reference",
            image_decoder,
        ).request_manifest,
        "slice_witness_sha256": build_expansion_messages(
            artifact,
            state,
            "reference",
            image_decoder,
        ).slice_witness_sha256,
        "base": base,
        "included_history_action": {
            "event_step_id": int(state.decision["history_event_step_ids"][-1]),
            "identity": included,
            "changed": included != base,
        },
        "excluded_current_or_future_actions": [
            {
                "event_step_id": result["event_step_id"],
                "identity": identity,
                "unchanged": identity == base,
            }
            for result, identity in zip(excluded, excluded_identities, strict=True)
        ],
        "included_mutation_count": 1,
        "excluded_mutation_count": len(excluded),
        "excluded_state": bool(excluded),
        "policy_forward_count": 0,
        "canary": dict(canary),
    }


def _base_state_record(
    *,
    state: ExpansionState,
    run_contract_sha256: str,
    request_manifest: Mapping[str, Any],
    slice_witness_sha256: str,
    processor_canary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MEASUREMENT_KERNEL_SCHEMA_VERSION,
        "protocol_id": MEASUREMENT_KERNEL_PROTOCOL_ID,
        "run_contract_sha256": run_contract_sha256,
        "state": _state_projection(state),
        "outcome": MEASUREMENT_STATE_OUTCOME_FAILED,
        "failure": None,
        "slice_witness": {
            **dict(request_manifest),
            "slice_witness_sha256": slice_witness_sha256,
        },
        "request_manifest": dict(request_manifest),
        "processor_canary": {
            "passed": True,
            "real_request": {
                key: processor_canary[key] for key in PROCESSOR_IDENTITY_KEYS
            },
        },
        "parse_success": False,
        "parse_success_count": 0,
        "model_emitted_closer_count": 0,
        "androidworld_bridge_count": 0,
        "repeat_canonical_action_agreement": False,
        "finite_logit_distances": False,
        "canonical_action": None,
        "androidworld_bridge": None,
        "screen_dimensions": None,
        "native_generations": [],
        "teacher_forwards": {},
        "distances": {
            "repeat_reference_kl": None,
            "summary_reference_kl": None,
        },
        "distance_audits": {},
        "operation_counts": {
            "generation_call_count": 0,
            "teacher_forward_count": 0,
            "kl_measurement_count": 0,
            **MEASUREMENT_FORBIDDEN_OPERATION_COUNTS,
        },
        "started_at_utc": _utc_now(),
        "ended_at_utc": None,
        "duration_seconds": None,
    }


def _finalize_state_record(
    record: dict[str, Any],
    *,
    started: float,
    outcome: str,
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    record["outcome"] = outcome
    record["failure"] = dict(failure) if failure is not None else None
    record["ended_at_utc"] = _utc_now()
    record["duration_seconds"] = time.perf_counter() - started
    return record


def run_expansion_state_once(
    *,
    artifact: ExpansionArtifact,
    state: ExpansionState,
    runtime: Any,
    parse_error_class: type[BaseException],
    distance_backend: Any,
    image_decoder: Callable[[bytes], Any],
    run_contract_sha256: str,
    processor_canary: Mapping[str, Any],
    message_pair_builder: Callable[
        [ExpansionArtifact, ExpansionState, Callable[[bytes], Any]],
        tuple[MessageBundle, MessageBundle],
    ] = build_expansion_message_pair,
) -> dict[str, Any]:
    """Execute one fixed-denominator state once with no retry or target backfill."""
    started = time.perf_counter()
    reference, summary = message_pair_builder(artifact, state, image_decoder)
    if (
        reference.request_manifest != summary.request_manifest
        or reference.slice_witness_sha256 != summary.slice_witness_sha256
        or reference.restored_event_step_ids != state.candidate_event_step_ids
        or summary.restored_event_step_ids
    ):
        raise RuntimeError("reference and summary decision-view identities drifted")
    real_canary = processor_canary.get("real_request")
    if not isinstance(real_canary, Mapping):
        raise ValueError("state lacks its completed actual-processor canary")
    record = _base_state_record(
        state=state,
        run_contract_sha256=run_contract_sha256,
        request_manifest=reference.request_manifest,
        slice_witness_sha256=reference.slice_witness_sha256,
        processor_canary=real_canary,
    )
    reference_dimensions = _screen_dimensions(reference.messages)
    summary_dimensions = _screen_dimensions(summary.messages)
    if reference_dimensions != summary_dimensions:
        raise RuntimeError("reference and summary current-observation dimensions differ")
    width, height = reference_dimensions
    record["screen_dimensions"] = {"width": width, "height": height}
    parsed_generations: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    parse_failures: list[dict[str, Any]] = []
    for repeat_index in (1, 2):
        record["operation_counts"]["generation_call_count"] += 1
        try:
            generation = runtime.generate_native_action(reference.messages)
        except parse_error_class as error:
            output_text = getattr(error, "output_text", None)
            metadata = getattr(error, "metadata", None)
            if not isinstance(output_text, str) or not isinstance(metadata, Mapping):
                raise RuntimeError("generation parse error lost native evidence") from error
            _validate_generation_metadata(metadata, raw_output=output_text)
            generation_record = {
                "repeat_index": repeat_index,
                "output_text": output_text,
                "metadata": dict(metadata),
                "canonical_action": None,
                "androidworld_bridge": None,
                "parse_error_type": str(
                    getattr(error, "parse_error_type", error.__class__.__name__)
                ),
                "parse_error_message": str(
                    getattr(error, "parse_error_message", str(error))
                ),
            }
            record["native_generations"].append(generation_record)
            record["model_emitted_closer_count"] += int(
                metadata.get("model_emitted_tool_call_close") is True
            )
            parse_failures.append(generation_record)
            continue
        output_text = getattr(generation, "output_text", None)
        metadata = getattr(generation, "metadata", None)
        if not isinstance(output_text, str) or not isinstance(metadata, Mapping):
            raise RuntimeError("runtime generation result lost native evidence")
        _validate_generation_metadata(metadata, raw_output=output_text)
        parsed = parse_gui_owl_v2_1_output(output_text)
        if (
            metadata.get("model_emitted_tool_call_close") is not True
            or metadata.get("generated_tool_call_close_token_count") != 1
            or metadata.get("final_generated_token_id") != 151658
        ):
            raise RuntimeError("parseable generation lacks the model-emitted closer")
        runtime_arguments = _action_arguments(generation.parsed_output.canonical_action)
        parsed_arguments = _action_arguments(parsed.canonical_action)
        if runtime_arguments != parsed_arguments:
            raise RuntimeError("runtime and independent parser actions differ")
        bridge = gui_owl_v2_action_to_androidworld(
            parsed.canonical_action,
            screen_width=width,
            screen_height=height,
        )
        record["parse_success_count"] += 1
        record["model_emitted_closer_count"] += 1
        record["androidworld_bridge_count"] += 1
        record["native_generations"].append(
            {
                "repeat_index": repeat_index,
                "output_text": output_text,
                "metadata": dict(metadata),
                "canonical_action": parsed_arguments,
                "androidworld_bridge": bridge,
                "parse_error_type": None,
                "parse_error_message": None,
            }
        )
        parsed_generations.append((parsed.canonical_action, parsed_arguments, bridge))
    if parse_failures:
        return _finalize_state_record(
            record,
            started=started,
            outcome=MEASUREMENT_STATE_OUTCOME_FAILED,
            failure=_failure(
                stage="reference_generations",
                category="PARSE_FAILURE",
                exception_type=str(parse_failures[0]["parse_error_type"]),
                message=(
                    f"{len(parse_failures)} of 2 strict reference generations failed: "
                    f"{parse_failures[0]['parse_error_message']}"
                ),
            ),
        )
    record["parse_success"] = True
    first_action, first_arguments, first_bridge = parsed_generations[0]
    if first_arguments != parsed_generations[1][1]:
        return _finalize_state_record(
            record,
            started=started,
            outcome=MEASUREMENT_STATE_OUTCOME_FAILED,
            failure=_failure(
                stage="reference_generation_repeat_comparison",
                category="CANONICAL_ACTION_MISMATCH",
                exception_type=None,
                message="two deterministic full-history generations produced different actions",
            ),
        )
    record["repeat_canonical_action_agreement"] = True
    record["canonical_action"] = first_arguments
    record["androidworld_bridge"] = first_bridge

    record["operation_counts"]["teacher_forward_count"] += 1
    reference_logits, reference_metadata = runtime.teacher_forced_distance_logits(
        (reference.messages,),
        (first_action,),
    )
    reference_log_probs = distance_backend.prepare_reference(reference_logits)
    del reference_logits
    record["teacher_forwards"]["reference_1"] = dict(reference_metadata)

    record["operation_counts"]["teacher_forward_count"] += 1
    repeat_logits, repeat_metadata = runtime.teacher_forced_distance_logits(
        (reference.messages,),
        (first_action,),
    )
    record["operation_counts"]["kl_measurement_count"] += 1
    repeat_measurement = distance_backend.measure(reference_log_probs, repeat_logits)
    del repeat_logits
    record["teacher_forwards"]["reference_2"] = dict(repeat_metadata)
    repeat_value = float(repeat_measurement.value)
    repeat_finite = math.isfinite(repeat_value)
    record["distances"]["repeat_reference_kl"] = (
        repeat_value if repeat_finite else None
    )
    record["distance_audits"]["repeat_reference_kl"] = dict(
        repeat_measurement.audit
    )

    record["operation_counts"]["teacher_forward_count"] += 1
    summary_logits, summary_metadata = runtime.teacher_forced_distance_logits(
        (summary.messages,),
        (first_action,),
    )
    record["operation_counts"]["kl_measurement_count"] += 1
    summary_measurement = distance_backend.measure(reference_log_probs, summary_logits)
    del summary_logits
    del reference_log_probs
    record["teacher_forwards"]["summary_only"] = dict(summary_metadata)
    summary_value = float(summary_measurement.value)
    summary_finite = math.isfinite(summary_value)
    record["distances"]["summary_reference_kl"] = (
        summary_value if summary_finite else None
    )
    record["distance_audits"]["summary_reference_kl"] = dict(
        summary_measurement.audit
    )
    if not repeat_finite or not summary_finite:
        return _finalize_state_record(
            record,
            started=started,
            outcome=MEASUREMENT_STATE_OUTCOME_FAILED,
            failure=_failure(
                stage="gpu_distance_validation",
                category="NONFINITE_DISTANCE",
                exception_type=None,
                message="repeat or summary-reference KL is non-finite",
            ),
        )
    record["finite_logit_distances"] = True
    return _finalize_state_record(
        record,
        started=started,
        outcome=MEASUREMENT_STATE_OUTCOME_VALID,
        failure=None,
    )


def wrap_measurement_record(
    inner: Mapping[str, Any], *, spec: WorkerSpec
) -> dict[str, Any]:
    inner_outcome = inner.get("outcome")
    if inner_outcome not in {
        MEASUREMENT_STATE_OUTCOME_VALID,
        MEASUREMENT_STATE_OUTCOME_FAILED,
    }:
        raise ValueError("measurement kernel returned an unknown state outcome")
    outcome = (
        STATE_OUTCOME_VALID
        if inner_outcome == MEASUREMENT_STATE_OUTCOME_VALID
        else STATE_OUTCOME_FAILED
    )
    operations = inner.get("operation_counts")
    if not isinstance(operations, Mapping):
        raise ValueError("measurement kernel omitted operation counts")
    actual = {
        key: int(operations[key]) for key in PER_STATE_PLANNED_OPERATION_COUNTS
    }
    if any(
        actual[key] < 0 or actual[key] > PER_STATE_PLANNED_OPERATION_COUNTS[key]
        for key in actual
    ):
        raise ValueError("measurement kernel exceeded the per-state schedule")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": RUNNER_PROTOCOL_ID,
        "run_contract_sha256": inner["run_contract_sha256"],
        "worker": {
            "worker_id": spec.worker_id,
            "device": spec.device,
            "index_parity": spec.index_parity,
        },
        "state": dict(inner["state"]),
        "outcome": outcome,
        "planned_operation_counts": dict(PER_STATE_PLANNED_OPERATION_COUNTS),
        "actual_operation_counts": actual,
        "measurement_kernel": {
            "protocol_id": MEASUREMENT_KERNEL_PROTOCOL_ID,
            "schema_version": MEASUREMENT_KERNEL_SCHEMA_VERSION,
            "record": dict(inner),
        },
    }


def _device_identity(device: str, *, torch_module: Any) -> dict[str, Any]:
    logical_index = int(device.split(":", maxsplit=1)[1])
    properties = torch_module.cuda.get_device_properties(torch_module.device(device))
    property_uuid = getattr(properties, "uuid", None)
    if isinstance(property_uuid, bytes):
        expected_uuid = property_uuid.decode("ascii")
    else:
        expected_uuid = str(property_uuid) if property_uuid is not None else ""
    if expected_uuid and not expected_uuid.startswith("GPU-"):
        expected_uuid = f"GPU-{expected_uuid}"
    if not expected_uuid:
        raise RuntimeError("PyTorch did not expose the selected GPU UUID")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records = [
        [field.strip() for field in line.split(",")]
        for line in result.stdout.splitlines()
    ]
    fields = next(
        (record for record in records if len(record) == 5 and record[2] == expected_uuid),
        None,
    )
    if fields is None:
        raise RuntimeError("PyTorch and nvidia-smi GPU identities differ")
    nvidia_index, gpu_name, gpu_uuid, pci_bus_id, driver = fields
    return {
        "logical_device_index": logical_index,
        "nvidia_smi_index": int(nvidia_index),
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "gpu_pci_bus_id": pci_bus_id,
        "nvidia_driver_version": driver,
    }


def validate_worker_runtime_metadata(
    metadata: Mapping[str, Any], *, spec: WorkerSpec
) -> None:
    required = {
        "device": spec.device,
        "gpu_name": "NVIDIA H200",
        "logical_device_index": spec.index_parity,
        "container_image_digest": CANONICAL_IMAGE_DIGEST,
        "python_version": "3.12.3",
        "torch_version": "2.11.0+cu130",
        "torch_cuda_version": "13.0",
        "cudnn_version": 91900,
        "transformers_version": "5.6.0",
        "nvidia_driver_version": "570.172.08",
        "dtype": "torch.bfloat16",
        "requested_attention_implementation": "eager",
        "observed_attention_implementation": {
            "top": "eager",
            "text": "eager",
            "vision": "eager",
        },
        "seed": 0,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "strict_cuda_determinism_claimed": False,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError(f"{spec.worker_id} eager runtime metadata drifted")
    if (
        not isinstance(metadata.get("gpu_uuid"), str)
        or not metadata["gpu_uuid"]
        or not isinstance(metadata.get("gpu_pci_bus_id"), str)
        or not metadata["gpu_pci_bus_id"]
        or type(metadata.get("nvidia_smi_index")) is not int
    ):
        raise ValueError("worker physical GPU identity is incomplete")


def validate_worker_runtime_pair(
    runtimes: Mapping[str, Mapping[str, Any]],
) -> None:
    specs = expected_worker_specs()
    if set(runtimes) != {spec.worker_id for spec in specs}:
        raise ValueError("runtime pair worker inventory drifted")
    for spec in specs:
        validate_worker_runtime_metadata(runtimes[spec.worker_id], spec=spec)
    first, second = (runtimes[spec.worker_id] for spec in specs)
    allowed = {
        "device",
        "gpu_uuid",
        "gpu_pci_bus_id",
        "logical_device_index",
        "nvidia_smi_index",
    }
    if {key: value for key, value in first.items() if key not in allowed} != {
        key: value for key, value in second.items() if key not in allowed
    }:
        raise ValueError("worker scientific runtime metadata differs")
    if any(first.get(key) == second.get(key) for key in allowed):
        raise ValueError("workers did not bind two distinct physical devices")


def validate_worker_runtime_envelopes(
    envelopes: Mapping[str, Mapping[str, Any]],
    *,
    run_contract_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    specs = expected_worker_specs()
    if set(envelopes) != {spec.worker_id for spec in specs}:
        raise ValueError("runtime-envelope worker inventory drifted")
    runtimes: dict[str, Mapping[str, Any]] = {}
    expected_keys = {
        "schema_version",
        "protocol_id",
        "run_contract_sha256",
        "worker",
        "runtime_metadata",
        "created_at_utc",
    }
    for spec in specs:
        envelope = envelopes[spec.worker_id]
        created = envelope.get("created_at_utc")
        if (
            set(envelope) != expected_keys
            or envelope.get("schema_version") != SCHEMA_VERSION
            or envelope.get("protocol_id") != RUNNER_PROTOCOL_ID
            or envelope.get("run_contract_sha256") != run_contract_sha256
            or envelope.get("worker") != spec.to_dict()
            or not isinstance(created, str)
            or not created.endswith("Z")
        ):
            raise ValueError("worker runtime envelope drifted")
        try:
            parsed = datetime.fromisoformat(f"{created[:-1]}+00:00")
        except ValueError as error:
            raise ValueError("worker runtime timestamp is invalid") from error
        if parsed.tzinfo != timezone.utc:
            raise ValueError("worker runtime timestamp is not UTC")
        metadata = envelope.get("runtime_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("worker runtime metadata must be an object")
        runtimes[spec.worker_id] = metadata
    validate_worker_runtime_pair(runtimes)
    return runtimes


def load_worker_runtime(
    *,
    spec: WorkerSpec,
    model_dir: str | Path,
    snapshot_manifest: str | Path,
    container_image_digest: str,
) -> RuntimeBindings:
    from causalcache.policy.gui_owl_v2_1_runtime import (
        GUIOwlV21GenerationParseError,
    )
    from causalcache.policy.gui_owl_v2_2_eager_runtime import GUIOwlV22EagerRuntime
    from causalcache.restoration_v2_gpu_kl import gpu_resident_full_vocab_mean_kl

    runtime = GUIOwlV22EagerRuntime(
        model_dir=model_dir,
        expected_snapshot_manifest=snapshot_manifest,
        device=spec.device,
        target_effective_visual_tokens_per_image=2560,
    )
    metadata = {
        **dict(runtime.metadata),
        **_device_identity(spec.device, torch_module=runtime.torch),
        "container_image_digest": container_image_digest,
    }
    validate_worker_runtime_metadata(metadata, spec=spec)
    return RuntimeBindings(
        runtime=runtime,
        parse_error_class=GUIOwlV21GenerationParseError,
        distance_backend=GPUFullVocabularyKLBackend(
            torch_module=runtime.torch,
            kl_kernel=gpu_resident_full_vocab_mean_kl,
        ),
        metadata=metadata,
    )


def _processor_identity_projection(
    identity: Mapping[str, Any], *, decision_step_id: int
) -> dict[str, Any]:
    projected = {key: identity.get(key) for key in PROCESSOR_IDENTITY_KEYS}
    hash_keys = {
        "rendered_prompt_sha256",
        "input_ids_sha256",
        "attention_mask_sha256",
        "pixel_values_sha256",
        "image_grid_thw_sha256",
        "processor_request_sha256",
    }
    if any(
        not isinstance(projected.get(key), str)
        or re.fullmatch(r"[0-9a-f]{64}", projected[key]) is None
        for key in hash_keys
    ):
        raise ValueError("processor request hash identity is incomplete")
    for tensor in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        shape = projected.get(f"{tensor}_shape")
        dtype = projected.get(f"{tensor}_dtype")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(value) is not int or value <= 0 for value in shape)
            or not isinstance(dtype, str)
            or not dtype
        ):
            raise ValueError("processor tensor shape/dtype identity is incomplete")
    if (
        projected.get("processor_path")
        != "GUIOwlV22EagerRuntime._encode_exact_batch"
        or projected.get("image_count") != decision_step_id - 1
        or projected.get("policy_forward_executed") is not False
    ):
        raise ValueError("processor path/image-count/no-forward identity drifted")
    unhashed = dict(projected)
    request_sha = unhashed.pop("processor_request_sha256")
    if request_sha != sha256_bytes(canonical_json_bytes(unhashed)):
        raise ValueError("processor request aggregate SHA256 drifted")
    return projected


def build_worker_processor_canary_report(
    *,
    spec: WorkerSpec,
    artifact: ExpansionArtifact,
    runtime: Any,
    run_contract_sha256: str,
    image_decoder: Callable[[bytes], Any] = _decode_rgb_image,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    included_count = 0
    excluded_state_count = 0
    excluded_mutation_count = 0
    for index in spec.state_indices:
        raw = run_state_processor_canary(
            artifact=artifact,
            state=artifact.states[index],
            runtime=runtime,
            image_decoder=image_decoder,
        )
        decision_step = artifact.states[index].decision_step_id
        real = _processor_identity_projection(
            raw["base"], decision_step_id=decision_step
        )
        included = {
            "event_step_id": raw["included_history_action"]["event_step_id"],
            **_processor_identity_projection(
                raw["included_history_action"]["identity"],
                decision_step_id=decision_step,
            ),
        }
        excluded = [
            {
                "event_step_id": value["event_step_id"],
                **_processor_identity_projection(
                    value["identity"], decision_step_id=decision_step
                ),
            }
            for value in raw["excluded_current_or_future_actions"]
        ]
        if any(
            included[key] == real[key]
            for key in ("rendered_prompt_sha256", "input_ids_sha256")
        ):
            raise RuntimeError("included-history processor canary did not change")
        if any(
            included[key] != real[key]
            for key in (
                "pixel_values_sha256",
                "pixel_values_shape",
                "pixel_values_dtype",
                "image_grid_thw_sha256",
                "image_grid_thw_shape",
                "image_grid_thw_dtype",
                "image_count",
            )
        ):
            raise RuntimeError("action-only included canary changed visual routing")
        if any(
            any(value[key] != real[key] for key in PROCESSOR_IDENTITY_KEYS)
            for value in excluded
        ):
            raise RuntimeError("excluded-event action leaked into processor request")
        records.append(
            {
                "state_index": index,
                "real_request": real,
                "included_history_mutation": included,
                "excluded_event_mutations": excluded,
            }
        )
        included_count += 1
        excluded_state_count += int(bool(excluded))
        excluded_mutation_count += len(excluded)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": RUNNER_PROTOCOL_ID,
        "status": "WORKER_PROCESSOR_CANARY_COMPLETE",
        "run_contract_sha256": run_contract_sha256,
        "worker": spec.envelope_identity(),
        "state_indices": list(spec.state_indices),
        "records": records,
        "included_history_mutation_count": included_count,
        "excluded_state_count": excluded_state_count,
        "excluded_event_mutation_count": excluded_mutation_count,
        "completed_at_utc": _utc_now(),
    }


def validate_global_processor_canary_reports(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    run_contract_sha256: str,
    states: Sequence[ExpansionState],
) -> dict[str, Any]:
    specs = expected_worker_specs()
    if set(reports) != {spec.worker_id for spec in specs}:
        raise ValueError("processor canary worker inventory drifted")
    all_indices: list[int] = []
    totals = {
        "included_history_mutation_count": 0,
        "excluded_state_count": 0,
        "excluded_event_mutation_count": 0,
    }
    report_hashes: dict[str, str] = {}
    for spec in specs:
        report = reports[spec.worker_id]
        if (
            report.get("schema_version") != SCHEMA_VERSION
            or report.get("protocol_id") != RUNNER_PROTOCOL_ID
            or report.get("status") != "WORKER_PROCESSOR_CANARY_COMPLETE"
            or report.get("run_contract_sha256") != run_contract_sha256
            or report.get("worker") != spec.envelope_identity()
            or report.get("state_indices") != list(spec.state_indices)
        ):
            raise ValueError("processor canary worker identity drifted")
        records = report.get("records")
        if not isinstance(records, list) or len(records) != 96:
            raise ValueError("processor canary worker state denominator drifted")
        worker_included = 0
        worker_excluded_states = 0
        worker_excluded_mutations = 0
        for index, record in zip(spec.state_indices, records, strict=True):
            if not isinstance(record, Mapping) or record.get("state_index") != index:
                raise ValueError("processor canary record order drifted")
            real = record.get("real_request")
            included = record.get("included_history_mutation")
            excluded = record.get("excluded_event_mutations")
            if (
                not isinstance(real, Mapping)
                or not isinstance(included, Mapping)
                or not isinstance(excluded, list)
                or set(real) != PROCESSOR_IDENTITY_KEYS
                or set(included) != {"event_step_id", *PROCESSOR_IDENTITY_KEYS}
            ):
                raise ValueError("processor canary record schema drifted")
            step = states[index].decision_step_id
            history = list(states[index].decision["history_event_step_ids"])
            real = _processor_identity_projection(real, decision_step_id=step)
            included_identity = _processor_identity_projection(
                {key: included[key] for key in PROCESSOR_IDENTITY_KEYS},
                decision_step_id=step,
            )
            if included.get("event_step_id") != history[-1]:
                raise ValueError("included canary did not mutate the latest history action")
            if any(
                included_identity[key] == real[key]
                for key in ("rendered_prompt_sha256", "input_ids_sha256")
            ):
                raise ValueError("included action did not affect processor request")
            if any(
                included_identity[key] != real[key]
                for key in (
                    "pixel_values_sha256",
                    "pixel_values_shape",
                    "pixel_values_dtype",
                    "image_grid_thw_sha256",
                    "image_grid_thw_shape",
                    "image_grid_thw_dtype",
                    "image_count",
                )
            ):
                raise ValueError("included action canary changed visual routing")
            expected_excluded_steps = list(range(step, 6))
            if [value.get("event_step_id") for value in excluded] != (
                expected_excluded_steps
            ):
                raise ValueError("excluded canary event-step geometry drifted")
            for mutation in excluded:
                if (
                    not isinstance(mutation, Mapping)
                    or set(mutation) != {"event_step_id", *PROCESSOR_IDENTITY_KEYS}
                ):
                    raise ValueError("excluded action affected processor request")
                mutation_identity = _processor_identity_projection(
                    {key: mutation[key] for key in PROCESSOR_IDENTITY_KEYS},
                    decision_step_id=step,
                )
                if any(
                    mutation_identity[key] != real[key]
                    for key in PROCESSOR_IDENTITY_KEYS
                ):
                    raise ValueError("excluded action affected processor request")
            worker_included += 1
            worker_excluded_states += int(bool(excluded))
            worker_excluded_mutations += len(excluded)
        all_indices.extend(report["state_indices"])
        observed_worker_totals = {
            "included_history_mutation_count": worker_included,
            "excluded_state_count": worker_excluded_states,
            "excluded_event_mutation_count": worker_excluded_mutations,
        }
        if any(
            report.get(key) != value for key, value in observed_worker_totals.items()
        ):
            raise ValueError("processor canary worker aggregate counts drifted")
        for key, value in observed_worker_totals.items():
            totals[key] += value
        report_hashes[spec.worker_id] = sha256_bytes(pretty_json_bytes(report))
    if sorted(all_indices) != list(range(EXPECTED_STATE_COUNT)) or totals != {
        "included_history_mutation_count": 192,
        "excluded_state_count": 128,
        "excluded_event_mutation_count": 192,
    }:
        raise ValueError("global processor canary denominator drifted")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": RUNNER_PROTOCOL_ID,
        "status": "COORDINATOR_RELEASED_PROCESSOR_CANARY_BARRIER",
        "run_contract_sha256": run_contract_sha256,
        "worker_report_sha256": report_hashes,
        "state_count": EXPECTED_STATE_COUNT,
        **totals,
        "released_at_utc": _utc_now(),
    }


def _worker_root(layout: ExpansionLayout, spec: WorkerSpec) -> Path:
    return layout.root / WORKER_DIRECTORY / spec.worker_id


def _worker_sibling_paths(global_ledger: Path) -> dict[str, Path]:
    return {
        spec.worker_id: global_ledger.with_name(
            f"{global_ledger.name[:-5]}.{spec.worker_id}.json"
        )
        for spec in expected_worker_specs()
    }


def _empty_worker_high_water() -> dict[str, Any]:
    return {
        spec.worker_id: {
            "attempted_state_indices": [],
            "completed_state_indices": [],
            "worker_ledger_sha256": None,
        }
        for spec in expected_worker_specs()
    }


def _worker_ledger_record(
    *,
    layout: ExpansionLayout,
    spec: WorkerSpec,
    status: str,
    claimed_at_utc: str,
    attempted: Sequence[int],
    completed: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": RUNNER_PROTOCOL_ID,
        "status": status,
        "run_contract_sha256": layout.run_contract_sha256,
        "worker": spec.to_dict(),
        "claimed_at_utc": claimed_at_utc,
        "attempted_state_indices": list(attempted),
        "completed_state_indices": list(completed),
        "retry_count": 0,
        "top_up_count": 0,
    }


def _replace_worker_ledgers(
    *,
    layout: ExpansionLayout,
    spec: WorkerSpec,
    root_ledger: Path,
    value: Mapping[str, Any],
) -> None:
    _replace_json_durable(
        layout.worker_sibling_ledgers[spec.worker_id],
        value,
        staging_directory=layout.publish_staging,
    )
    _replace_json_durable(
        root_ledger,
        value,
        staging_directory=layout.publish_staging,
    )
    if (
        layout.worker_sibling_ledgers[spec.worker_id].read_bytes()
        != root_ledger.read_bytes()
    ):
        raise RuntimeError("worker sibling and root ledger bytes differ")


def claim_global_attempt(
    *,
    run_contract: Mapping[str, Any],
    states: Sequence[ExpansionState],
    output_dir: str | Path,
    global_ledger: str | Path,
) -> ExpansionLayout:
    validate_execution_run_contract(
        run_contract,
        require_canonical_attempt_identity=False,
    )
    root = Path(output_dir).resolve()
    ledger_path = Path(global_ledger).resolve()
    publish_staging = root.with_name(f".{root.name}.publish-staging")
    sibling_paths = _worker_sibling_paths(ledger_path)
    if root.exists() or ledger_path.exists() or publish_staging.exists() or any(
        path.exists() for path in sibling_paths.values()
    ):
        raise FileExistsError(
            "expansion substrate is one fresh attempt; output/ledger forbids retry"
        )
    if len(states) != EXPECTED_STATE_COUNT:
        raise ValueError("global attempt must bind exactly 192 states")
    run_contract_sha256 = sha256_bytes(canonical_json_bytes(run_contract))
    started = _utc_now()
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": RUNNER_PROTOCOL_ID,
        "status": GLOBAL_CLAIM_STATUS,
        "attempt_id": run_contract["attempt_identity"]["attempt_id"],
        "run_contract_sha256": run_contract_sha256,
        "worker_topology": run_contract["worker_topology"],
        "claimed_at_utc": started,
        "spawn_started_at_utc": None,
        "completed_state_count": 0,
        "attempted_state_count": 0,
        "planned_counts": dict(PLANNED_OPERATION_COUNTS),
        "actual_counts": {key: 0 for key in PLANNED_OPERATION_COUNTS},
        "worker_high_water": _empty_worker_high_water(),
        "worker_sibling_ledgers": {
            worker_id: str(path) for worker_id, path in sibling_paths.items()
        },
        "retry_count": 0,
        "top_up_count": 0,
    }
    publish_staging.mkdir(parents=False, exist_ok=False)
    _write_json_exclusive(
        ledger_path,
        ledger,
        staging_directory=publish_staging,
    )
    root.mkdir(parents=True, exist_ok=False)
    for spec in expected_worker_specs():
        worker_root = root / WORKER_DIRECTORY / spec.worker_id
        (worker_root / STATE_DIRECTORY).mkdir(parents=True)
        (worker_root / ATTEMPT_DIRECTORY).mkdir()
    (root / "logs").mkdir()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": RUNNER_PROTOCOL_ID,
        "status": "EXPANSION_SUBSTRATE_GLOBAL_ATTEMPT_CLAIMED",
        "run_contract_sha256": run_contract_sha256,
        "run_contract": dict(run_contract),
        "created_at_utc": started,
    }
    _write_json_exclusive(
        root / RUN_MANIFEST_FILENAME,
        manifest,
        staging_directory=publish_staging,
    )
    layout = ExpansionLayout(
        root=root,
        global_ledger=ledger_path,
        manifest=root / RUN_MANIFEST_FILENAME,
        aggregate=root / AGGREGATE_FILENAME,
        execution_evidence=root / EXECUTION_EVIDENCE_FILENAME,
        run_contract=dict(run_contract),
        run_contract_sha256=run_contract_sha256,
        states=tuple(states),
        projections=tuple(_state_projection(state) for state in states),
        started_at_utc=started,
        worker_sibling_ledgers=sibling_paths,
        publish_staging=publish_staging,
    )
    for spec in expected_worker_specs():
        record = _worker_ledger_record(
            layout=layout,
            spec=spec,
            status="WORKER_SIBLING_PREBOUND_BEFORE_SPAWN",
            claimed_at_utc=started,
            attempted=(),
            completed=(),
        )
        _write_json_exclusive(
            sibling_paths[spec.worker_id],
            record,
            staging_directory=publish_staging,
        )
    return layout


def _state_marker(
    layout: ExpansionLayout, spec: WorkerSpec, state_index: int
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": RUNNER_PROTOCOL_ID,
        "status": "STATE_ATTEMPT_CLAIMED_NO_RETRY_OR_TOP_UP",
        "run_contract_sha256": layout.run_contract_sha256,
        "worker": spec.envelope_identity(),
        "state_index": state_index,
        "claimed_at_utc": _utc_now(),
        "retry_count": 0,
        "top_up_count": 0,
    }


def _state_actual_counts(envelopes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        key: sum(int(record["actual_operation_counts"][key]) for record in envelopes)
        for key in PER_STATE_PLANNED_OPERATION_COUNTS
    }


def wait_for_release_or_abort(
    layout: ExpansionLayout,
    *,
    release_filename: str,
    abort_filename: str,
    timeout_seconds: float = 1800.0,
) -> None:
    release = layout.root / release_filename
    abort = layout.root / abort_filename
    deadline = time.monotonic() + timeout_seconds
    while True:
        if abort.is_file():
            record = _strict_json(abort)
            raise RuntimeError(f"coordinator aborted barrier: {record.get('failure')}")
        if release.is_file():
            record = _strict_json(release)
            _validate_worker_barrier_release(
                layout,
                release_filename=release_filename,
                record=record,
            )
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"coordinator barrier timed out: {release_filename}")
        time.sleep(0.25)


def _worker_evidence_hashes(
    layout: ExpansionLayout, *, filename: str
) -> dict[str, str]:
    result: dict[str, str] = {}
    for spec in expected_worker_specs():
        path = _worker_root(layout, spec) / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError("coordinator release lacks both worker evidence files")
        result[spec.worker_id] = sha256_file(path)
    return result


def _validate_worker_barrier_release(
    layout: ExpansionLayout,
    *,
    release_filename: str,
    record: Mapping[str, Any],
) -> None:
    released = record.get("released_at_utc")
    if not isinstance(released, str) or not released.endswith("Z"):
        raise ValueError("coordinator barrier release timestamp drifted")
    try:
        parsed = datetime.fromisoformat(f"{released[:-1]}+00:00")
    except ValueError as error:
        raise ValueError("coordinator barrier release timestamp is invalid") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError("coordinator barrier release timestamp is not UTC")
    common = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": RUNNER_PROTOCOL_ID,
        "run_contract_sha256": layout.run_contract_sha256,
    }
    if release_filename == RUNTIME_BARRIER_RELEASE_FILENAME:
        expected_keys = {
            *common,
            "status",
            "worker_runtime_sha256",
            "released_at_utc",
        }
        if (
            set(record) != expected_keys
            or any(record.get(key) != value for key, value in common.items())
            or record.get("status")
            != "COORDINATOR_RELEASED_BOTH_WORKER_RUNTIMES"
            or record.get("worker_runtime_sha256")
            != _worker_evidence_hashes(layout, filename=WORKER_RUNTIME_FILENAME)
        ):
            raise ValueError("runtime barrier release identity drifted")
        return
    if release_filename == PROCESSOR_CANARY_RELEASE_FILENAME:
        expected_keys = {
            *common,
            "status",
            "worker_report_sha256",
            "state_count",
            "included_history_mutation_count",
            "excluded_state_count",
            "excluded_event_mutation_count",
            "released_at_utc",
        }
        if (
            set(record) != expected_keys
            or any(record.get(key) != value for key, value in common.items())
            or record.get("status")
            != "COORDINATOR_RELEASED_PROCESSOR_CANARY_BARRIER"
            or record.get("worker_report_sha256")
            != _worker_evidence_hashes(layout, filename=PROCESSOR_CANARY_FILENAME)
            or record.get("state_count") != EXPECTED_STATE_COUNT
            or record.get("included_history_mutation_count") != 192
            or record.get("excluded_state_count") != 128
            or record.get("excluded_event_mutation_count") != 192
        ):
            raise ValueError("processor-canary barrier release identity drifted")
        return
    raise ValueError("worker requested an unfrozen coordinator release phase")


def run_worker_shard(
    *,
    layout: ExpansionLayout,
    spec: WorkerSpec,
    artifact: ExpansionArtifact,
    runtime_loader: Callable[[WorkerSpec], RuntimeBindings],
    state_kernel: Callable[..., Mapping[str, Any]] = run_expansion_state_once,
) -> Mapping[str, Any]:
    worker_root = _worker_root(layout, spec)
    ledger_path = worker_root / WORKER_LEDGER_FILENAME
    terminal_path = worker_root / WORKER_TERMINAL_FILENAME
    claimed = _utc_now()
    attempted: list[int] = []
    completed: list[int] = []
    envelopes: list[Mapping[str, Any]] = []
    ledger = _worker_ledger_record(
        layout=layout,
        spec=spec,
        status="WORKER_ATTEMPT_CLAIMED_BEFORE_RUNTIME_IMPORT",
        claimed_at_utc=claimed,
        attempted=attempted,
        completed=completed,
    )
    _write_json_exclusive(
        ledger_path,
        ledger,
        staging_directory=layout.publish_staging,
    )
    _replace_worker_ledgers(
        layout=layout,
        spec=spec,
        root_ledger=ledger_path,
        value=ledger,
    )
    try:
        bindings = runtime_loader(spec)
        runtime_record = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": RUNNER_PROTOCOL_ID,
            "run_contract_sha256": layout.run_contract_sha256,
            "worker": spec.to_dict(),
            "runtime_metadata": dict(bindings.metadata),
            "created_at_utc": _utc_now(),
        }
        _publish_json_exclusive_atomic(
            worker_root / WORKER_RUNTIME_FILENAME,
            runtime_record,
            staging_directory=layout.publish_staging,
        )
        wait_for_release_or_abort(
            layout,
            release_filename=RUNTIME_BARRIER_RELEASE_FILENAME,
            abort_filename=RUNTIME_BARRIER_ABORT_FILENAME,
        )
        canary_report = build_worker_processor_canary_report(
            spec=spec,
            artifact=artifact,
            runtime=bindings.runtime,
            run_contract_sha256=layout.run_contract_sha256,
        )
        _publish_json_exclusive_atomic(
            worker_root / PROCESSOR_CANARY_FILENAME,
            canary_report,
            staging_directory=layout.publish_staging,
        )
        wait_for_release_or_abort(
            layout,
            release_filename=PROCESSOR_CANARY_RELEASE_FILENAME,
            abort_filename=PROCESSOR_CANARY_ABORT_FILENAME,
        )
        canary_by_index = {
            int(record["state_index"]): record for record in canary_report["records"]
        }
        for index in spec.state_indices:
            attempted.append(index)
            ledger = _worker_ledger_record(
                layout=layout,
                spec=spec,
                status="WORKER_SHARD_RUNNING_NO_RETRY",
                claimed_at_utc=claimed,
                attempted=attempted,
                completed=completed,
            )
            _replace_json_durable(
                layout.worker_sibling_ledgers[spec.worker_id],
                ledger,
                staging_directory=layout.publish_staging,
            )
            marker_path = worker_root / ATTEMPT_DIRECTORY / f"{index:03d}.json"
            _write_json_exclusive(
                marker_path,
                _state_marker(layout, spec, index),
                staging_directory=layout.publish_staging,
            )
            _replace_json_durable(
                ledger_path,
                ledger,
                staging_directory=layout.publish_staging,
            )
            if (
                layout.worker_sibling_ledgers[spec.worker_id].read_bytes()
                != ledger_path.read_bytes()
            ):
                raise RuntimeError("worker ledgers differ before state generation")
            inner = state_kernel(
                artifact=artifact,
                state=artifact.states[index],
                runtime=bindings.runtime,
                parse_error_class=bindings.parse_error_class,
                distance_backend=bindings.distance_backend,
                image_decoder=_decode_rgb_image,
                run_contract_sha256=layout.run_contract_sha256,
                processor_canary=canary_by_index[index],
            )
            envelope = wrap_measurement_record(inner, spec=spec)
            _write_json_exclusive(
                worker_root / STATE_DIRECTORY / f"{index:03d}.json",
                envelope,
                staging_directory=layout.publish_staging,
            )
            envelopes.append(envelope)
            completed.append(index)
            ledger = _worker_ledger_record(
                layout=layout,
                spec=spec,
                status="WORKER_SHARD_RUNNING_NO_RETRY",
                claimed_at_utc=claimed,
                attempted=attempted,
                completed=completed,
            )
            _replace_worker_ledgers(
                layout=layout,
                spec=spec,
                root_ledger=ledger_path,
                value=ledger,
            )
        worker_outcome = WORKER_OUTCOME_COMPLETE
        failure = None
        ledger_status = "WORKER_SHARD_COMPLETED"
    except Exception as error:
        worker_outcome = WORKER_OUTCOME_INVALID
        failure = {
            "stage": "worker_runtime_canary_or_state_execution",
            "category": "WORKER_EXECUTION_EXCEPTION",
            "exception_type": error.__class__.__name__,
            "message": str(error),
        }
        ledger_status = "WORKER_SHARD_INVALID"
    ledger = _worker_ledger_record(
        layout=layout,
        spec=spec,
        status=ledger_status,
        claimed_at_utc=claimed,
        attempted=attempted,
        completed=completed,
    )
    _replace_worker_ledgers(
        layout=layout,
        spec=spec,
        root_ledger=ledger_path,
        value=ledger,
    )
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": RUNNER_PROTOCOL_ID,
        "run_contract_sha256": layout.run_contract_sha256,
        "worker": spec.to_dict(),
        "outcome": worker_outcome,
        "planned_counts": {
            key: value // 2 for key, value in PLANNED_OPERATION_COUNTS.items()
        },
        "actual_counts": _state_actual_counts(envelopes),
        "completed_state_indices": completed,
        "attempted_state_indices": attempted,
        "failure": failure,
        "ended_at_utc": _utc_now(),
        "retry_count": 0,
        "top_up_count": 0,
    }
    _write_json_exclusive(
        terminal_path,
        terminal,
        staging_directory=layout.publish_staging,
    )
    return terminal


def monitor_stop_request_path(summary_file: str | Path) -> Path:
    summary = Path(summary_file)
    return summary.with_name(f"{summary.name}.stop-request.json")


def _publish_external_json_exclusive(
    path: str | Path, value: Mapping[str, Any]
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(pretty_json_bytes(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def sample_visible_monitor_gpus() -> tuple[dict[str, Any], ...]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError("nvidia-smi monitor sample schema drifted")
        try:
            nvidia_index = int(fields[0])
            utilization = int(fields[2])
        except ValueError as error:
            raise ValueError("nvidia-smi monitor sample is not integral") from error
        if (
            nvidia_index < 0
            or not fields[1].startswith("GPU-")
            or not 0 <= utilization <= 100
        ):
            raise ValueError("nvidia-smi monitor sample value drifted")
        records.append(
            {
                "nvidia_smi_index": nvidia_index,
                "gpu_uuid": fields[1],
                "utilization_percent": utilization,
            }
        )
    if (
        len(records) != 2
        or len({record["nvidia_smi_index"] for record in records}) != 2
        or len({record["gpu_uuid"] for record in records}) != 2
    ):
        raise ValueError("monitor sidecar must observe exactly two visible GPUs")
    return tuple(records)


def _monitor_sample_record(
    *, sample_index: int, gpus: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": MONITOR_SAMPLE_STATUS,
        "sample_index": sample_index,
        "sampled_at_utc": _utc_now(),
        "visible_gpu_count": 2,
        "gpus": [dict(record) for record in gpus],
    }


def publish_monitor_stop_request(
    *, summary_file: str | Path, run_contract_sha256: str
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{64}", run_contract_sha256) is None:
        raise ValueError("monitor stop request requires the run-contract SHA256")
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": MONITOR_STOP_REQUEST_STATUS,
        "run_contract_sha256": run_contract_sha256,
        "requested_at_utc": _utc_now(),
    }
    _publish_external_json_exclusive(
        monitor_stop_request_path(summary_file),
        record,
    )
    return record


def run_monitor_sidecar(
    *,
    ready_file: str | Path,
    log_file: str | Path,
    summary_file: str | Path,
    sampler: Callable[[], Sequence[Mapping[str, Any]]] = sample_visible_monitor_gpus,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    ready_path = Path(ready_file)
    log_path = Path(log_file)
    summary_path = Path(summary_file)
    stop_path = monitor_stop_request_path(summary_path)
    paths = (ready_path, log_path, summary_path, stop_path)
    if any(not path.is_absolute() for path in paths):
        raise ValueError("monitor-sidecar paths must be absolute")
    if len({str(path) for path in paths}) != 4:
        raise ValueError("monitor-sidecar paths must be distinct")
    if any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError("monitor-sidecar requires four fresh O_EXCL paths")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    initial_gpus = tuple(dict(record) for record in sampler())
    initial_inventory = tuple(
        (record.get("nvidia_smi_index"), record.get("gpu_uuid"))
        for record in initial_gpus
    )
    if (
        len(initial_gpus) != 2
        or len(set(initial_inventory)) != 2
        or any(
            type(record.get("nvidia_smi_index")) is not int
            or not isinstance(record.get("gpu_uuid"), str)
            or not record["gpu_uuid"].startswith("GPU-")
            or type(record.get("utilization_percent")) is not int
            or not 0 <= record["utilization_percent"] <= 100
            for record in initial_gpus
        )
    ):
        raise ValueError("monitor-sidecar sampler did not return exactly two GPUs")
    sample_count = 1
    low_incidents = int(
        any(record["utilization_percent"] < 90 for record in initial_gpus)
    )
    with log_path.open("xb") as log_handle:
        log_handle.write(
            canonical_json_bytes(
                _monitor_sample_record(sample_index=0, gpus=initial_gpus)
            )
            + b"\n"
        )
        log_handle.flush()
        os.fsync(log_handle.fileno())
        pid = os.getpid()
        ready = {
            "schema_version": SCHEMA_VERSION,
            "status": MONITOR_READY_STATUS,
            "container_name_prefix": "sglang-omni-jaxan",
            "minimum_gpu_utilization_percent": 90,
            "monitor_process_alive": True,
            "monitor_pid": pid,
            "started_at_utc": started_at,
        }
        _publish_external_json_exclusive(ready_path, ready)
        os.kill(pid, 0)
        while not stop_path.is_file():
            if stop_path.is_symlink():
                raise ValueError("monitor stop request cannot be a symlink")
            sleep_fn(MONITOR_SAMPLE_INTERVAL_SECONDS)
            if stop_path.is_file():
                break
            current = tuple(dict(record) for record in sampler())
            inventory = tuple(
                (record.get("nvidia_smi_index"), record.get("gpu_uuid"))
                for record in current
            )
            if inventory != initial_inventory or any(
                type(record.get("utilization_percent")) is not int
                or not 0 <= record["utilization_percent"] <= 100
                for record in current
            ):
                raise ValueError("visible GPU inventory changed during monitoring")
            log_handle.write(
                canonical_json_bytes(
                    _monitor_sample_record(sample_index=sample_count, gpus=current)
                )
                + b"\n"
            )
            log_handle.flush()
            os.fsync(log_handle.fileno())
            sample_count += 1
            low_incidents += int(
                any(record["utilization_percent"] < 90 for record in current)
            )

    stop = _strict_json(stop_path)
    if (
        set(stop) != MONITOR_STOP_REQUEST_KEYS
        or stop.get("schema_version") != SCHEMA_VERSION
        or stop.get("status") != MONITOR_STOP_REQUEST_STATUS
        or re.fullmatch(r"[0-9a-f]{64}", str(stop.get("run_contract_sha256")))
        is None
        or not isinstance(stop.get("requested_at_utc"), str)
        or not stop["requested_at_utc"].endswith("Z")
    ):
        raise ValueError("monitor stop request identity drifted")
    try:
        datetime.fromisoformat(f"{stop['requested_at_utc'][:-1]}+00:00")
    except ValueError as error:
        raise ValueError("monitor stop request timestamp drifted") from error
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": MONITOR_SUMMARY_STATUS,
        "run_contract_sha256": stop["run_contract_sha256"],
        "container_name_prefix": "sglang-omni-jaxan",
        "minimum_gpu_utilization_percent": 90,
        "started_before_first_generation": True,
        "started_at_utc": started_at,
        "stopped_at_utc": _utc_now(),
        "sample_count": sample_count,
        "low_utilization_incident_count": low_incidents,
        "visible_gpu_count": 2,
        "visible_gpu_uuids": [record["gpu_uuid"] for record in initial_gpus],
        "stop_request_sha256": sha256_file(stop_path),
    }
    _publish_external_json_exclusive(summary_path, summary)
    return summary


def _monitor_ready(path: str | Path) -> Mapping[str, Any]:
    record = _strict_json(Path(path).resolve())
    expected_keys = {
        "schema_version",
        "status",
        "container_name_prefix",
        "minimum_gpu_utilization_percent",
        "monitor_process_alive",
        "monitor_pid",
        "started_at_utc",
    }
    started = record.get("started_at_utc")
    if (
        set(record) != expected_keys
        or record.get("schema_version") != "1.0.0"
        or record.get("status") != MONITOR_READY_STATUS
        or record.get("container_name_prefix") != "sglang-omni-jaxan"
        or record.get("minimum_gpu_utilization_percent") != 90
        or record.get("monitor_process_alive") is not True
        or type(record.get("monitor_pid")) is not int
        or record["monitor_pid"] <= 0
        or not isinstance(started, str)
        or not started.endswith("Z")
    ):
        raise ValueError("GPU utilization monitor is not ready at the frozen threshold")
    try:
        parsed = datetime.fromisoformat(f"{started[:-1]}+00:00")
    except ValueError as error:
        raise ValueError("GPU utilization monitor start time is invalid") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError("GPU utilization monitor start time is not UTC")
    try:
        os.kill(int(record["monitor_pid"]), 0)
    except OSError as error:
        raise ValueError("GPU utilization monitor PID is not alive") from error
    return record


def _wait_for_worker_files(
    *,
    layout: ExpansionLayout,
    filename: str,
    processes: Sequence[multiprocessing.Process],
    timeout_seconds: float,
) -> dict[str, dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    specs = expected_worker_specs()
    while True:
        paths = {
            spec.worker_id: _worker_root(layout, spec) / filename for spec in specs
        }
        if all(path.is_file() for path in paths.values()):
            return {worker_id: _strict_json(path) for worker_id, path in paths.items()}
        if any(process.exitcode is not None for process in processes):
            raise RuntimeError(f"worker exited before coordinator observed {filename}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"coordinator timed out waiting for {filename}")
        time.sleep(0.25)


def coordinate_global_pre_generation_barriers(
    layout: ExpansionLayout,
    *,
    processes: Sequence[multiprocessing.Process],
    monitor_ready_file: str | Path,
    timeout_seconds: float = 1800.0,
) -> None:
    try:
        runtime_envelopes = _wait_for_worker_files(
            layout=layout,
            filename=WORKER_RUNTIME_FILENAME,
            processes=processes,
            timeout_seconds=timeout_seconds,
        )
        runtimes = validate_worker_runtime_envelopes(
            runtime_envelopes,
            run_contract_sha256=layout.run_contract_sha256,
        )
        monitor_ready = _monitor_ready(monitor_ready_file)
        monitor_started = datetime.fromisoformat(
            f"{str(monitor_ready['started_at_utc'])[:-1]}+00:00"
        )
        runtime_created = [
            datetime.fromisoformat(
                f"{str(envelope['created_at_utc'])[:-1]}+00:00"
            )
            for envelope in runtime_envelopes.values()
        ]
        if monitor_started > min(runtime_created):
            raise ValueError("GPU monitor did not start before worker runtime creation")
        if not all(process.is_alive() for process in processes):
            raise RuntimeError("worker lost liveness before runtime barrier release")
        runtime_released_at = _utc_now()
        _publish_json_exclusive_atomic(
            layout.root / RUNTIME_BARRIER_RELEASE_FILENAME,
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": RUNNER_PROTOCOL_ID,
                "status": "COORDINATOR_RELEASED_BOTH_WORKER_RUNTIMES",
                "run_contract_sha256": layout.run_contract_sha256,
                "worker_runtime_sha256": {
                    worker_id: sha256_bytes(
                        pretty_json_bytes(runtime_envelopes[worker_id])
                    )
                    for worker_id in sorted(runtime_envelopes)
                },
                "released_at_utc": runtime_released_at,
            },
            staging_directory=layout.publish_staging,
        )
    except Exception as error:
        _publish_json_exclusive_atomic(
            layout.root / RUNTIME_BARRIER_ABORT_FILENAME,
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": RUNNER_PROTOCOL_ID,
                "status": "COORDINATOR_ABORTED_RUNTIME_BARRIER",
                "run_contract_sha256": layout.run_contract_sha256,
                "failure": {
                    "stage": "runtime_identity_barrier",
                    "category": "RUNTIME_BARRIER_FAILURE",
                    "exception_type": error.__class__.__name__,
                    "message": str(error),
                },
                "aborted_at_utc": _utc_now(),
            },
            staging_directory=layout.publish_staging,
        )
        raise
    try:
        canary_reports = _wait_for_worker_files(
            layout=layout,
            filename=PROCESSOR_CANARY_FILENAME,
            processes=processes,
            timeout_seconds=timeout_seconds,
        )
        release = validate_global_processor_canary_reports(
            canary_reports,
            run_contract_sha256=layout.run_contract_sha256,
            states=layout.states,
        )
        monitor_ready = _monitor_ready(monitor_ready_file)
        monitor_started = datetime.fromisoformat(
            f"{str(monitor_ready['started_at_utc'])[:-1]}+00:00"
        )
        runtime_released = datetime.fromisoformat(
            f"{runtime_released_at[:-1]}+00:00"
        )
        if monitor_started > runtime_released:
            raise ValueError("GPU monitor did not start before runtime release")
        if not all(process.is_alive() for process in processes):
            raise RuntimeError("worker lost liveness before processor-canary release")
        release["released_at_utc"] = _utc_now()
        _publish_json_exclusive_atomic(
            layout.root / PROCESSOR_CANARY_RELEASE_FILENAME,
            release,
            staging_directory=layout.publish_staging,
        )
    except Exception as error:
        _publish_json_exclusive_atomic(
            layout.root / PROCESSOR_CANARY_ABORT_FILENAME,
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": RUNNER_PROTOCOL_ID,
                "status": "COORDINATOR_ABORTED_PROCESSOR_CANARY_BARRIER",
                "run_contract_sha256": layout.run_contract_sha256,
                "failure": {
                    "stage": "global_192_state_processor_canary_barrier",
                    "category": "PROCESSOR_CANARY_BARRIER_FAILURE",
                    "exception_type": error.__class__.__name__,
                    "message": str(error),
                },
                "aborted_at_utc": _utc_now(),
            },
            staging_directory=layout.publish_staging,
        )
        raise


def _worker_observation(
    layout: ExpansionLayout, spec: WorkerSpec
) -> dict[str, Any]:
    root = _worker_root(layout, spec)
    sibling = layout.worker_sibling_ledgers[spec.worker_id]
    ledger = _strict_json(sibling)
    root_ledger = root / WORKER_LEDGER_FILENAME
    root_record = _strict_json(root_ledger) if root_ledger.is_file() else None
    return {
        "attempted": list(ledger["attempted_state_indices"]),
        "completed": list(ledger["completed_state_indices"]),
        "ledger_sha256": sha256_file(sibling),
        "root_ledger_matches": (
            root_record is not None and root_ledger.read_bytes() == sibling.read_bytes()
        ),
        "runtime": (
            _strict_json(root / WORKER_RUNTIME_FILENAME)
            if (root / WORKER_RUNTIME_FILENAME).is_file()
            else None
        ),
        "terminal": (
            _strict_json(root / WORKER_TERMINAL_FILENAME)
            if (root / WORKER_TERMINAL_FILENAME).is_file()
            else None
        ),
        "marker_indices": sorted(
            int(path.stem)
            for path in (root / ATTEMPT_DIRECTORY).glob("*.json")
            if path.stem.isdigit()
        ),
        "state_indices": sorted(
            int(path.stem)
            for path in (root / STATE_DIRECTORY).glob("*.json")
            if path.stem.isdigit()
        ),
    }


def _copy_regular_file(
    source: str | Path,
    destination: Path,
    *,
    staging_directory: Path,
) -> None:
    source_path = Path(source).resolve()
    if source_path.is_symlink() or not source_path.is_file() or source_path.stat().st_size <= 0:
        raise ValueError(f"required execution evidence file is missing or empty: {source}")
    if destination.exists():
        if destination.resolve() == source_path:
            return
        raise FileExistsError(f"execution evidence destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not staging_directory.is_dir():
        raise FileNotFoundError("external copy-publication staging directory vanished")
    temporary = staging_directory / f"{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with source_path.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _append_execution_log(layout: ExpansionLayout, message: str) -> None:
    path = layout.root / "logs/execution.log"
    with path.open("ab") as destination:
        destination.write(f"{_utc_now()} {message}\n".encode("utf-8"))
        destination.flush()
        os.fsync(destination.fileno())


def _external_witness(path: Path, *, relative: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"terminal execution evidence is missing: {relative}")
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _materialize_terminal_execution_evidence(
    *,
    layout: ExpansionLayout,
    runtime_values: Mapping[str, Mapping[str, Any]],
    utilization_monitor_log: str | Path,
    monitor_summary: str | Path,
) -> dict[str, Any]:
    runtime_projection = {
        worker_id: dict(runtime_values[worker_id]) for worker_id in sorted(runtime_values)
    }
    worker_projection = {
        worker_id: {
            key: runtime_values[worker_id][key]
            for key in (
                "device",
                "gpu_name",
                "gpu_uuid",
                "gpu_pci_bus_id",
                "logical_device_index",
                "nvidia_smi_index",
            )
        }
        for worker_id in sorted(runtime_values)
    }
    terminal_failure: dict[str, Any] | None = None
    monitor_witness: dict[str, Any] | None = None
    try:
        deadline = time.monotonic() + 60.0
        summary_path = Path(monitor_summary).resolve()
        while not summary_path.is_file():
            if summary_path.is_symlink():
                raise ValueError("terminal monitor summary cannot be a symlink")
            if time.monotonic() >= deadline:
                raise TimeoutError("terminal monitor summary did not become available")
            time.sleep(0.25)
        monitor = _strict_json(summary_path)
        stop_path = monitor_stop_request_path(summary_path)
        stop = _strict_json(stop_path)
        ready = _strict_json(
            layout.root / LOG_PATHS["utilization_monitor_ready"]
        )
        if (
            set(monitor) != MONITOR_SUMMARY_KEYS
            or monitor.get("schema_version") != SCHEMA_VERSION
            or monitor.get("status") != MONITOR_SUMMARY_STATUS
            or monitor.get("run_contract_sha256") != layout.run_contract_sha256
            or monitor.get("container_name_prefix") != "sglang-omni-jaxan"
            or monitor.get("minimum_gpu_utilization_percent") != 90
            or monitor.get("started_before_first_generation") is not True
            or type(monitor.get("sample_count")) is not int
            or monitor["sample_count"] <= 0
            or type(monitor.get("low_utilization_incident_count")) is not int
            or monitor["low_utilization_incident_count"] < 0
            or monitor["low_utilization_incident_count"] > monitor["sample_count"]
            or monitor.get("started_at_utc") != ready.get("started_at_utc")
            or not isinstance(monitor.get("stopped_at_utc"), str)
            or not monitor["stopped_at_utc"].endswith("Z")
            or monitor.get("visible_gpu_count") != 2
            or not isinstance(monitor.get("visible_gpu_uuids"), list)
            or len(monitor["visible_gpu_uuids"]) != 2
            or len(set(monitor["visible_gpu_uuids"])) != 2
            or any(
                not isinstance(value, str) or not value.startswith("GPU-")
                for value in monitor["visible_gpu_uuids"]
            )
            or set(stop) != MONITOR_STOP_REQUEST_KEYS
            or stop.get("schema_version") != SCHEMA_VERSION
            or stop.get("status") != MONITOR_STOP_REQUEST_STATUS
            or stop.get("run_contract_sha256") != layout.run_contract_sha256
            or not isinstance(stop.get("requested_at_utc"), str)
            or not stop["requested_at_utc"].endswith("Z")
            or monitor.get("stop_request_sha256") != sha256_file(stop_path)
        ):
            raise ValueError("terminal utilization-monitor summary drifted")
        try:
            stop_requested = datetime.fromisoformat(
                f"{stop['requested_at_utc'][:-1]}+00:00"
            )
            monitor_stopped = datetime.fromisoformat(
                f"{monitor['stopped_at_utc'][:-1]}+00:00"
            )
        except ValueError as error:
            raise ValueError("terminal monitor timestamp drifted") from error
        if monitor_stopped < stop_requested:
            raise ValueError("terminal monitor summary predates the stop request")
        _validate_monitor_gpu_uuid_binding(monitor, runtimes=runtime_values)
        # note (luojiaxuan): A valid terminal summary is the frozen signal that
        # the monitor stopped writing; only then is its log copied for hashing.
        _copy_regular_file(
            utilization_monitor_log,
            layout.root / LOG_PATHS["utilization_monitor"],
            staging_directory=layout.publish_staging,
        )
        _copy_regular_file(
            stop_path,
            layout.root / LOG_PATHS["utilization_monitor_stop_request"],
            staging_directory=layout.publish_staging,
        )
        _copy_regular_file(
            summary_path,
            layout.root / MONITOR_SUMMARY_PATH,
            staging_directory=layout.publish_staging,
        )
        monitor_witness = _external_witness(
            layout.root / MONITOR_SUMMARY_PATH,
            relative=MONITOR_SUMMARY_PATH,
        )
        status = "TERMINAL_EXECUTION_EVIDENCE"
    except Exception as error:
        status = "FAILED_TERMINAL_EXECUTION_EVIDENCE"
        terminal_failure = {
            "stage": "terminal_execution_evidence_finalization",
            "category": "MONITOR_SUMMARY_FINALIZATION_FAILURE",
            "exception_type": error.__class__.__name__,
            "message": str(error),
        }
        monitor_log_destination = layout.root / LOG_PATHS["utilization_monitor"]
        if not monitor_log_destination.is_file():
            _copy_regular_file(
                utilization_monitor_log,
                monitor_log_destination,
                staging_directory=layout.publish_staging,
            )
        stop_destination = (
            layout.root / LOG_PATHS["utilization_monitor_stop_request"]
        )
        if not stop_destination.is_file():
            _copy_regular_file(
                monitor_stop_request_path(monitor_summary),
                stop_destination,
                staging_directory=layout.publish_staging,
            )
    logs = {
        name: _external_witness(
            layout.root / relative,
            relative=relative,
        )
        for name, relative in LOG_PATHS.items()
    }
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": RUNNER_PROTOCOL_ID,
        "status": status,
        "run_contract_sha256": layout.run_contract_sha256,
        "runner_source_git_commit": layout.run_contract["source"][
            "runner_source_git_commit"
        ],
        "execution_git_commit": layout.run_contract["source"][
            "execution_git_commit"
        ],
        "config_sha256": layout.run_contract["frozen_config"]["sha256"],
        "completion_manifest_sha256": layout.run_contract["derived_completion"][
            "sha256"
        ],
        "derived_immutable_revision": layout.run_contract["derived_artifact"][
            "immutable_revision"
        ],
        "runtime_identity_sha256": sha256_bytes(
            canonical_json_bytes(runtime_projection)
        ),
        "worker_identity_sha256": sha256_bytes(
            canonical_json_bytes(worker_projection)
        ),
        "logs": logs,
        "monitor_summary": monitor_witness,
        "terminal_failure": terminal_failure,
        "forbidden_operation_counts": {
            key: 0 for key in sorted(FORBIDDEN_COUNT_KEYS)
        },
        "ended_at_utc": _utc_now(),
    }
    _write_json_exclusive(
        layout.execution_evidence,
        evidence,
        staging_directory=layout.publish_staging,
    )
    return evidence


def _update_global_terminal_ledger(
    *,
    layout: ExpansionLayout,
    observations: Mapping[str, Mapping[str, Any]],
    actual_counts: Mapping[str, int],
    valid: bool,
) -> None:
    ledger = _strict_json(layout.global_ledger)
    attempted = sum(len(value["attempted"]) for value in observations.values())
    completed = sum(len(value["completed"]) for value in observations.values())
    ledger.update(
        {
            "status": "GLOBAL_ATTEMPT_COMPLETED" if valid else "GLOBAL_ATTEMPT_INVALID",
            "attempted_state_count": attempted,
            "completed_state_count": completed,
            "actual_counts": dict(actual_counts),
            "worker_high_water": {
                spec.worker_id: {
                    "attempted_state_indices": list(
                        observations[spec.worker_id]["attempted"]
                    ),
                    "completed_state_indices": list(
                        observations[spec.worker_id]["completed"]
                    ),
                    "worker_ledger_sha256": observations[spec.worker_id][
                        "ledger_sha256"
                    ],
                }
                for spec in expected_worker_specs()
            },
        }
    )
    _replace_json_durable(
        layout.global_ledger,
        ledger,
        staging_directory=layout.publish_staging,
    )


def finalize_two_worker_attempt(
    layout: ExpansionLayout,
    *,
    utilization_monitor_log: str | Path,
    monitor_summary: str | Path,
) -> dict[str, Any]:
    specs = expected_worker_specs()
    observations = {
        spec.worker_id: _worker_observation(layout, spec) for spec in specs
    }
    envelopes: list[Mapping[str, Any]] = []
    invalid_error: BaseException | None = None
    invalid_failure_record: dict[str, Any] | None = None
    try:
        for spec in specs:
            observation = observations[spec.worker_id]
            terminal = observation["terminal"]
            if (
                not isinstance(terminal, Mapping)
                or terminal.get("outcome") != WORKER_OUTCOME_COMPLETE
                or observation["attempted"] != list(spec.state_indices)
                or observation["completed"] != list(spec.state_indices)
                or observation["marker_indices"] != list(spec.state_indices)
                or observation["state_indices"] != list(spec.state_indices)
                or observation["root_ledger_matches"] is not True
            ):
                raise RuntimeError("both exact parity shards did not terminate completely")
        runtime_values = {
            spec.worker_id: observations[spec.worker_id]["runtime"]["runtime_metadata"]
            for spec in specs
        }
        validate_worker_runtime_pair(runtime_values)
        for index in range(EXPECTED_STATE_COUNT):
            spec = specs[index % 2]
            envelopes.append(
                _strict_json(
                    _worker_root(layout, spec)
                    / STATE_DIRECTORY
                    / f"{index:03d}.json"
                )
            )
        actual_counts = _state_actual_counts(envelopes)
        reductions = []
        for envelope in envelopes:
            inner = envelope["measurement_kernel"]["record"]
            reductions.append(
                {
                    "state_id": envelope["state"]["state_id"],
                    "parse_success": inner["parse_success"] is True,
                    "finite_logit_distances": inner["finite_logit_distances"] is True,
                    "repeat_canonical_action_agreement": (
                        inner["repeat_canonical_action_agreement"] is True
                    ),
                    "repeat_reference_kl": inner["distances"][
                        "repeat_reference_kl"
                    ],
                    "summary_reference_kl": inner["distances"][
                        "summary_reference_kl"
                    ],
                    "processor_identity": dict(
                        inner["processor_canary"]["real_request"]
                    ),
                }
            )
        gate = reduce_expansion_substrate_gate(
            reductions,
            actual_counts=actual_counts,
            frozen_config=layout.run_contract["frozen_config"]["data"],
        )
        outcome = str(gate["derived_outcome"])
        valid_attempt = True
    except Exception as error:
        invalid_error = error
        invalid_failure_record = {
            "stage": "two_worker_terminal_merge_and_cpu_gate",
            "category": "TERMINAL_REDUCTION_EXCEPTION",
            "exception_type": error.__class__.__name__,
            "message": str(error),
        }
        runtime_values = {
            spec.worker_id: observation["runtime"]["runtime_metadata"]
            for spec in specs
            if isinstance((observation := observations[spec.worker_id])["runtime"], Mapping)
        }
        envelopes = []
        for spec in specs:
            for index in observations[spec.worker_id]["state_indices"]:
                try:
                    envelopes.append(
                        _strict_json(
                            _worker_root(layout, spec)
                            / STATE_DIRECTORY
                            / f"{index:03d}.json"
                        )
                    )
                except Exception:
                    continue
        actual_counts = _state_actual_counts(envelopes)
        gate = None
        outcome = INVALID_OUTCOME
        valid_attempt = False
    _append_execution_log(layout, f"terminal reduction outcome={outcome}")
    execution_evidence = _materialize_terminal_execution_evidence(
        layout=layout,
        runtime_values=runtime_values,
        utilization_monitor_log=utilization_monitor_log,
        monitor_summary=monitor_summary,
    )
    if execution_evidence["status"] == "FAILED_TERMINAL_EXECUTION_EVIDENCE":
        valid_attempt = False
        outcome = INVALID_OUTCOME
        gate = None
        invalid_failure_record = dict(execution_evidence["terminal_failure"])
    _update_global_terminal_ledger(
        layout=layout,
        observations=observations,
        actual_counts=actual_counts,
        valid=valid_attempt,
    )
    valid_count = sum(
        envelope.get("outcome") == STATE_OUTCOME_VALID for envelope in envelopes
    )
    failed_count = sum(
        envelope.get("outcome") == STATE_OUTCOME_FAILED for envelope in envelopes
    )
    failure_categories = Counter(
        envelope["measurement_kernel"]["record"].get("failure", {}).get("category")
        for envelope in envelopes
        if isinstance(
            envelope["measurement_kernel"]["record"].get("failure"), Mapping
        )
    )
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": RUNNER_PROTOCOL_ID,
        "status": "TERMINAL_EXPANSION_SUBSTRATE_ATTEMPT",
        "outcome": outcome,
        "run_contract_sha256": layout.run_contract_sha256,
        "fixed_state_denominator": EXPECTED_STATE_COUNT,
        "planned_counts": dict(PLANNED_OPERATION_COUNTS),
        "actual_counts": actual_counts,
        "attempted_state_count": sum(
            len(value["attempted"]) for value in observations.values()
        ),
        "completed_state_count": sum(
            len(value["completed"]) for value in observations.values()
        ),
        "valid_state_count": valid_count,
        "failed_state_count": failed_count,
        "failure_category_counts": dict(sorted(failure_categories.items())),
        "retry_performed": False,
        "top_up_performed": False,
        "started_at_utc": layout.started_at_utc,
        "ended_at_utc": _utc_now(),
        "gate": gate,
        "invalid_failure": invalid_failure_record,
        "execution_evidence_sha256": sha256_file(layout.execution_evidence),
    }
    _write_json_exclusive(
        layout.aggregate,
        aggregate,
        staging_directory=layout.publish_staging,
    )
    return aggregate


def _canonical_repo_input(
    supplied: str | Path,
    *,
    repository_root: Path,
    relative: str,
) -> Path:
    expected = (repository_root / relative).resolve()
    candidate = Path(supplied)
    if candidate.is_symlink():
        raise ValueError(f"canonical repository input cannot be a symlink: {relative}")
    actual = candidate.resolve() if candidate.is_absolute() else (
        repository_root / candidate
    ).resolve()
    if actual != expected or not actual.is_file():
        raise ValueError(f"canonical repository input must be {relative}")
    return actual


def _source_inventory_matches_checkout(
    *,
    repository_root: Path,
    git_commit: str,
    frozen_inventory: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    paths = [str(record["path"]) for record in frozen_inventory]
    committed = validate_committed_source_blobs(
        repository_root=repository_root,
        git_commit=git_commit,
        paths=paths,
    )
    by_path = {str(record["path"]): record for record in frozen_inventory}
    normalized: list[Mapping[str, Any]] = []
    for record in committed:
        path = str(record["path"])
        source = repository_root / path
        expected = by_path[path]
        observed = {
            "path": path,
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        }
        if observed != dict(expected):
            raise ValueError(f"runner-freeze source inventory drifted: {path}")
        normalized.append(observed)
    normalized.sort(key=lambda value: str(value["path"]))
    return tuple(normalized)


def authorize_production_expansion(
    args: argparse.Namespace,
) -> AuthorizedExpansion:
    root = Path(args.repository_root).resolve()
    frozen_config, contract_validation = load_and_validate_contract(
        args.contract,
        repository_root=root,
    )
    supplied_freeze = Path(args.runner_freeze)
    if supplied_freeze.is_symlink():
        raise ValueError("runner freeze supplied path cannot be a symlink")
    runner_freeze, runner_validation = load_and_validate_runner_freeze(
        args.runner_freeze,
        repository_root=root,
        require_committed_pushed_main=True,
    )
    git_execution = validate_clean_pushed_main(root)
    if (
        runner_validation.get("execution_head") != git_execution.get("commit")
        or runner_validation.get("origin_main_equals_execution_head") is not True
        or runner_validation.get("base_config_sha256")
        != contract_validation.get("config_sha256")
    ):
        raise ValueError("runner freeze, base contract, and execution HEAD differ")
    runner_commit = str(runner_validation["runner_git_commit"])
    source_inventory = _source_inventory_matches_checkout(
        repository_root=root,
        git_commit=runner_commit,
        frozen_inventory=runner_validation["source_inventory"],
    )
    execution = frozen_config["execution"]
    expected_hosts = {
        "hyper00": "node-radixark-16-0000",
        "hyper01": "node-radixark-16-0001",
    }
    if (
        Path(args.output_dir).resolve()
        != Path(execution["canonical_persistent_output_dir"])
        or Path(args.global_ledger).resolve()
        != Path(execution["canonical_global_attempt_ledger"])
        or Path(args.output_dir).resolve() != CANONICAL_OUTPUT_DIR
        or Path(args.global_ledger).resolve() != CANONICAL_LEDGER_PATH
        or args.container_image_digest != CANONICAL_IMAGE_DIGEST
        or args.host_alias not in expected_hosts
        or args.host_hostname != expected_hosts.get(args.host_alias)
        or re.fullmatch(r"[0-9a-f]{64}", args.container_id) is None
        or Path(args.model_dir).resolve() != CANONICAL_MODEL_DIR
        or not CANONICAL_MODEL_DIR.is_dir()
        or execution.get("required_host_class") != "Hyper_H200"
    ):
        raise ValueError("canonical expansion output, Hyper host, or container drifted")
    validate_worker_topology(execution["worker_topology"])
    ocr_path = _canonical_repo_input(
        args.ocr_backend_config,
        repository_root=root,
        relative=CANONICAL_OCR_BACKEND_CONFIG_PATH,
    )
    ocr_manifest_path = _canonical_repo_input(
        root / CANONICAL_OCR_BACKEND_MANIFEST_PATH,
        repository_root=root,
        relative=CANONICAL_OCR_BACKEND_MANIFEST_PATH,
    )
    snapshot_path = _canonical_repo_input(
        args.snapshot_manifest,
        repository_root=root,
        relative=CANONICAL_SNAPSHOT_MANIFEST_PATH,
    )
    completion_path = root / CANONICAL_COMPLETION_PATH
    completion_payload, completion = load_json_object(completion_path)
    frozen_completion = frozen_config["immutable_inputs"]["derived_completion"]
    if (
        sha256_bytes(completion_payload) != frozen_completion["sha256"]
        or completion_path.stat().st_size
        != frozen_completion.get("size_bytes", completion_path.stat().st_size)
    ):
        raise ValueError("derived completion manifest differs from frozen identity")
    derived = frozen_config["immutable_inputs"]["derived_artifact"]
    if args.derived_hf_tag != derived["tag"]:
        raise ValueError("explicit derived Hugging Face tag drifted")
    artifact = load_immutable_expansion_artifact(
        artifact_root=args.derived_artifact_root,
        frozen_config=frozen_config,
        backend_config_path=ocr_path,
        derived_repo=args.derived_hf_repo,
        derived_revision=args.derived_hf_revision,
    )
    # note (luojiaxuan): This file-only verification hashes the complete frozen
    # snapshot and pinned processor implementation after the exact-six immutable
    # artifact check and before the O_EXCL claim. It does not construct the model
    # or execute any policy forward.
    from causalcache.policy.gui_owl_v2_vision import verify_frozen_vision_runtime

    verified_model = verify_frozen_vision_runtime(
        model_dir=CANONICAL_MODEL_DIR,
        expected_snapshot_manifest=snapshot_path,
    )
    snapshot_payload, snapshot_data = load_json_object(snapshot_path)
    model_identity = {
        "local_path": str(CANONICAL_MODEL_DIR),
        "repo": verified_model.model_repo,
        "revision": verified_model.model_revision,
        "files": list(snapshot_data["files"]),
        "file_inventory_sha256": sha256_bytes(
            canonical_json_bytes(snapshot_data["files"])
        ),
        "all_files_verified": True,
    }
    for path in (
        args.preflight_log,
        args.utilization_monitor_log,
        args.monitor_ready_file,
    ):
        candidate = Path(path).resolve()
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"required pre-generation evidence is missing: {path}")
    for path in (
        Path(args.monitor_summary),
        monitor_stop_request_path(args.monitor_summary),
    ):
        if path.exists() or path.is_symlink():
            raise ValueError("monitor summary and stop request must be fresh preclaim")
    _monitor_ready(args.monitor_ready_file)
    inventory_by_path = {
        str(record["path"]): dict(record) for record in source_inventory
    }
    derived_identity = {
        key: derived[key]
        for key in (
            "repo",
            "tag",
            "immutable_revision",
            "payload_prefix",
            "artifact_tree_sha256",
            "artifact_file_count",
            "artifact_total_bytes",
        )
    }
    completion_identity = {
        "path": CANONICAL_COMPLETION_PATH,
        "sha256": sha256_bytes(completion_payload),
        "size_bytes": len(completion_payload),
    }
    canonical_inputs = {
        "derived_artifact_root": str(artifact.root),
        "derived_artifact": derived_identity,
        "completion_manifest": completion_identity,
        "ocr_backend_config": inventory_by_path[CANONICAL_OCR_BACKEND_CONFIG_PATH],
        "ocr_backend_manifest": inventory_by_path[
            CANONICAL_OCR_BACKEND_MANIFEST_PATH
        ],
        "model_snapshot_manifest": {
            **inventory_by_path[CANONICAL_SNAPSHOT_MANIFEST_PATH],
            "data": snapshot_data,
        },
        "model_snapshot": model_identity,
        "fresh_immutable_derived_validation": {
            "repo": derived_identity["repo"],
            "tag": derived_identity["tag"],
            "immutable_revision": derived_identity["immutable_revision"],
            "artifact_tree_sha256": derived_identity["artifact_tree_sha256"],
            "validation_passed": True,
            "validated_before_runtime_import": True,
        },
    }
    return AuthorizedExpansion(
        repository_root=root,
        frozen_config=frozen_config,
        contract_validation=contract_validation,
        git_identity={
            **dict(git_execution),
            "runner_source_commit": runner_commit,
            "runner_freeze_validation": dict(runner_validation),
            "runner_freeze": dict(runner_freeze),
        },
        source_inventory=source_inventory,
        completion_manifest=completion,
        artifact=artifact,
        snapshot_manifest_path=snapshot_path,
        model_identity=model_identity,
        canonical_inputs=canonical_inputs,
    )


def build_execution_run_contract(
    args: argparse.Namespace,
    authorized: AuthorizedExpansion,
) -> dict[str, Any]:
    config_path = authorized.repository_root / CANONICAL_CONFIG_PATH
    completion_path = authorized.repository_root / CANONICAL_COMPLETION_PATH
    freeze_path = authorized.repository_root / CANONICAL_RUNNER_FREEZE_PATH
    derived = authorized.frozen_config["immutable_inputs"]["derived_artifact"]
    topology = authorized.frozen_config["execution"]["worker_topology"]
    validate_worker_topology(topology)
    runner_validation = authorized.git_identity["runner_freeze_validation"]
    runner_freeze_data = dict(authorized.git_identity["runner_freeze"])
    runner_freeze_payload = pretty_json_bytes(runner_freeze_data)
    if freeze_path.read_bytes() != runner_freeze_payload:
        raise ValueError("runner-freeze file differs from embedded canonical bytes")
    contract = {
        "source": {
            "runner_source_git_commit": authorized.git_identity[
                "runner_source_commit"
            ],
            "execution_git_commit": authorized.git_identity["commit"],
            "branch": authorized.git_identity["branch"],
            "origin_url": authorized.git_identity["remote_url"],
            "clean_checkout": True,
            "head_equals_origin_main": True,
            "inventory": [dict(record) for record in authorized.source_inventory],
        },
        "frozen_config": {
            "path": CANONICAL_CONFIG_PATH,
            "sha256": sha256_file(config_path),
            "size_bytes": config_path.stat().st_size,
            "data": dict(authorized.frozen_config),
        },
        "runner_freeze": {
            "path": CANONICAL_RUNNER_FREEZE_PATH,
            "sha256": sha256_bytes(runner_freeze_payload),
            "size_bytes": len(runner_freeze_payload),
            "runner_git_commit": runner_validation["runner_git_commit"],
            "data": runner_freeze_data,
        },
        "derived_completion": {
            "path": CANONICAL_COMPLETION_PATH,
            "sha256": sha256_file(completion_path),
            "size_bytes": completion_path.stat().st_size,
        },
        "derived_artifact": {
            key: derived[key]
            for key in (
                "repo",
                "tag",
                "immutable_revision",
                "payload_prefix",
                "artifact_tree_sha256",
                "artifact_file_count",
                "artifact_total_bytes",
            )
        },
        "canonical_inputs": dict(authorized.canonical_inputs),
        "runtime_requirements": {
            **dict(authorized.frozen_config["runtime"]["expected_execution_stack"]),
            "dtype": "bfloat16",
            "attention_implementation": "eager",
        },
        "worker_topology": dict(topology),
        "states": [_state_projection(state) for state in authorized.artifact.states],
        "schedule": {
            "fixed_state_denominator": EXPECTED_STATE_COUNT,
            "planned_counts": dict(PLANNED_OPERATION_COUNTS),
            "per_state_planned_counts": dict(PER_STATE_PLANNED_OPERATION_COUNTS),
            "automatic_retry_allowed": False,
            "resume_allowed": False,
            "top_up_allowed": False,
            "replacement_allowed": False,
        },
        "attempt_identity": {
            "attempt_id": ATTEMPT_ID,
            "output_dir": str(Path(args.output_dir).resolve()),
            "global_ledger": str(Path(args.global_ledger).resolve()),
            "host_alias": args.host_alias,
            "host_hostname": args.host_hostname,
            "container_id": args.container_id,
            "container_image_digest": args.container_image_digest,
        },
        "execution_argv": list(args.execution_argv),
        "logs": dict(LOG_PATHS),
        "monitor": {
            "enabled": True,
            "minimum_gpu_utilization_percent": 90,
            "container_name_prefix": "sglang-omni-jaxan",
            "log_path": LOG_PATHS["utilization_monitor"],
            "summary_path": MONITOR_SUMMARY_PATH,
            "started_before_first_generation_required": True,
            "low_utilization_requires_immediate_inspection": True,
        },
    }
    validate_execution_run_contract(
        contract,
        require_canonical_attempt_identity=True,
    )
    return contract


def _worker_process_entry(
    layout: ExpansionLayout,
    spec: WorkerSpec,
    args: argparse.Namespace,
) -> None:
    authorized = authorize_production_expansion(args)
    rebuilt = build_execution_run_contract(args, authorized)
    if sha256_bytes(canonical_json_bytes(rebuilt)) != layout.run_contract_sha256:
        raise ValueError("worker rebuilt authorization differs from global claim")
    run_worker_shard(
        layout=layout,
        spec=spec,
        artifact=authorized.artifact,
        runtime_loader=lambda selected: load_worker_runtime(
            spec=selected,
            model_dir=args.model_dir,
            snapshot_manifest=authorized.snapshot_manifest_path,
            container_image_digest=args.container_image_digest,
        ),
    )


def launch_spawned_workers(
    *,
    layout: ExpansionLayout,
    args: argparse.Namespace,
) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_worker_process_entry,
            args=(layout, spec, args),
            name=f"causalcache-expansion-{spec.worker_id}",
        )
        for spec in expected_worker_specs()
    ]
    for process in processes:
        process.start()
    barrier_error: BaseException | None = None
    try:
        coordinate_global_pre_generation_barriers(
            layout,
            processes=processes,
            monitor_ready_file=(
                layout.root / "logs/gpu_utilization_monitor.ready.json"
            ),
        )
    except BaseException as error:
        barrier_error = error
    for process in processes:
        process.join()
    if barrier_error is not None:
        raise RuntimeError("global pre-generation barrier failed") from barrier_error
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError("one or more expansion workers exited abnormally")


def execute_production_expansion(
    args: argparse.Namespace,
    *,
    authorization_loader: Callable[
        [argparse.Namespace], AuthorizedExpansion
    ] = authorize_production_expansion,
    worker_launcher: Callable[[ExpansionLayout, argparse.Namespace], None] | None = None,
) -> dict[str, Any]:
    authorized = authorization_loader(args)
    run_contract = build_execution_run_contract(args, authorized)
    layout = claim_global_attempt(
        run_contract=run_contract,
        states=authorized.artifact.states,
        output_dir=args.output_dir,
        global_ledger=args.global_ledger,
    )
    _copy_regular_file(
        args.preflight_log,
        layout.root / LOG_PATHS["preflight"],
        staging_directory=layout.publish_staging,
    )
    _copy_regular_file(
        args.monitor_ready_file,
        layout.root / LOG_PATHS["utilization_monitor_ready"],
        staging_directory=layout.publish_staging,
    )
    _append_execution_log(layout, "global attempt claimed; spawning exact parity workers")
    ledger = _strict_json(layout.global_ledger)
    ledger["spawn_started_at_utc"] = _utc_now()
    _replace_json_durable(
        layout.global_ledger,
        ledger,
        staging_directory=layout.publish_staging,
    )
    launcher = worker_launcher or (
        lambda selected_layout, selected_args: launch_spawned_workers(
            layout=selected_layout,
            args=selected_args,
        )
    )
    try:
        launcher(layout, args)
    except Exception as error:
        _append_execution_log(
            layout,
            f"worker launch/barrier ended with {error.__class__.__name__}: {error}",
        )
    finally:
        try:
            publish_monitor_stop_request(
                summary_file=args.monitor_summary,
                run_contract_sha256=layout.run_contract_sha256,
            )
        except Exception as error:
            _append_execution_log(
                layout,
                "monitor stop request publication ended with "
                f"{error.__class__.__name__}: {error}",
            )
    return finalize_two_worker_attempt(
        layout,
        utilization_monitor_log=args.utilization_monitor_log,
        monitor_summary=args.monitor_summary,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--runner-freeze", required=True)
    parser.add_argument("--derived-artifact-root", required=True)
    parser.add_argument("--derived-hf-repo", required=True)
    parser.add_argument("--derived-hf-tag", required=True)
    parser.add_argument("--derived-hf-revision", required=True)
    parser.add_argument("--ocr-backend-config", required=True)
    parser.add_argument("--snapshot-manifest", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--host-hostname", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--global-ledger", required=True)
    parser.add_argument("--preflight-log", required=True)
    parser.add_argument("--utilization-monitor-log", required=True)
    parser.add_argument("--monitor-ready-file", required=True)
    parser.add_argument("--monitor-summary", required=True)
    return parser


def _build_monitor_sidecar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"{Path(__file__).name} monitor-sidecar")
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--summary-file", required=True)
    return parser


def monitor_sidecar_main(argv: Sequence[str]) -> int:
    args = _build_monitor_sidecar_parser().parse_args(list(argv))
    try:
        summary = run_monitor_sidecar(
            ready_file=args.ready_file,
            log_file=args.log_file,
            summary_file=args.summary_file,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "GPU_UTILIZATION_MONITOR_SIDECAR_FAILED",
                    "exception_type": error.__class__.__name__,
                    "message": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == MONITOR_SIDECAR_SUBCOMMAND:
        return monitor_sidecar_main(raw_argv[1:])
    args = _build_parser().parse_args(raw_argv)
    args.execution_argv = [sys.executable, str(Path(__file__).resolve()), *raw_argv]
    try:
        result = execute_production_expansion(args)
    except Exception as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": RUNNER_PROTOCOL_ID,
            "status": "INVALID_BEFORE_EXPANSION_GLOBAL_CLAIM",
            "outcome": INVALID_OUTCOME,
            "invalid_failure": {
                "stage": "preclaim_authorization",
                "category": "PRECLAIM_AUTHORIZATION_EXCEPTION",
                "exception_type": error.__class__.__name__,
                "message": str(error),
            },
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if result["outcome"] in {PASS_OUTCOME, NO_GO_OUTCOME} else 2


if __name__ == "__main__":
    raise SystemExit(main())
