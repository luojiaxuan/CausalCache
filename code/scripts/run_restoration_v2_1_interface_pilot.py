"""Run the frozen 15-state restoration-v2.1 official-tool pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causalcache.data.guiodyssey_restoration_v2 import sha256_file
from causalcache.data.restoration_v2_screening import (
    ScreeningState,
    ValidatedScreeningArtifact,
    load_validated_screening_artifact,
)
from causalcache.data.restoration_v2_1_processor_inputs import (
    V21ProcessorPromptSpec,
    build_v2_1_processor_messages,
)
from causalcache.policy.gui_owl_v2 import gui_owl_v2_action_to_androidworld
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_PROTOCOL_ID,
    parse_gui_owl_v2_1_output,
)
from causalcache.restoration_v2_1_contract import (
    CANONICAL_CONFIG_PATH,
    RestorationV21PilotContract,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = GUI_OWL_V2_1_PROTOCOL_ID
RUN_STATUS = "RESTORATION_V2_1_FIXED_15_STATE_INTERFACE_PILOT"
ATTEMPT_STATUS = "ATTEMPT_STARTED_NO_RETRY_OR_TOP_UP"
GLOBAL_ATTEMPT_STATUS = "GLOBAL_ATTEMPT_STARTED_DELETION_FORBIDDEN"
STATE_OUTCOME_VALID = "VALID_V2_1_NATIVE_ACTION"
STATE_OUTCOME_FAILED = "FAILED_V2_1_NATIVE_ACTION"
PASS_OUTCOME = "PASS_V2_1_INTERFACE_PILOT"
NO_GO_OUTCOME = "NO_GO_V2_1_INTERFACE_PILOT"
INVALID_OUTCOME = "INVALID_V2_1_PILOT"
RUN_MANIFEST_FILENAME = "run_manifest.json"
AGGREGATE_FILENAME = "aggregate.json"
RUNTIME_IDENTITY_FILENAME = "runtime_identity.json"
STATE_DIRECTORY = "states"
ATTEMPT_DIRECTORY = "attempts"
EXPECTED_STATE_COUNT = 15
SOURCE_STATE_START = 30
SOURCE_STATE_STOP = 45
CANONICAL_SCIENTIFIC_CONFIG_PATH = "code/configs/causalcache_restoration_v2.json"
CANONICAL_SELECTION_MANIFEST_PATH = "data/manifests/restoration_v2_selection.json"
CANONICAL_OCR_BACKEND_CONFIG_PATH = "code/configs/restoration_v2_ocr_backend.json"
PROCESSOR_AUDIT_SOURCE_PATH = (
    "code/causalcache/restoration_v2_1_processor_audit.py"
)
CANONICAL_PROCESSOR_PREFLIGHT_ARTIFACT_PATH = (
    "data/results/restoration_v2_1_processor_preflight/artifact.json"
)
RUNNER_SOURCE_PATH = "code/scripts/run_restoration_v2_1_interface_pilot.py"
CUDA_DEVICE_PATTERN = re.compile(r"cuda:[0-9]+")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
CONTAINER_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
CANONICAL_REMOTE_URL = "https://github.com/luojiaxuan/CausalCache.git"
CANONICAL_PILOT_HOST_ALIAS = "hyper00"
CANONICAL_PILOT_HOST_HOSTNAME = "node-radixark-16-0000"
CANONICAL_PILOT_CONTAINER_ID = (
    "69f2b1742e8fd9baac5080b2b97ee1f3c7c1df520f908a4541cadde9d28194df"
)
CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST = (
    "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
)
CANONICAL_PILOT_DEVICE = "cuda:0"
RUNTIME_METADATA_KEYS = {
    "assistant_prefix_token_ids",
    "chat_template_file_sha256",
    "chat_template_text_sha256",
    "device",
    "dtype",
    "frozen",
    "generation_eos_token_id",
    "generation_interface",
    "generation_num_beams",
    "generation_num_return_sequences",
    "generation_pad_token_id",
    "generation_standard_eos_suppression",
    "host_injected_tool_call_closer",
    "max_pixels",
    "min_pixels",
    "model_class",
    "model_dir",
    "model_repo",
    "model_revision",
    "official_tool_schema_sha256",
    "official_tools_argument_count",
    "output_recovery_or_normalization",
    "processor_class",
    "protocol_id",
    "single_device",
    "snapshot_manifest_sha256",
    "suppressed_standard_eos_token_ids",
    "target_effective_visual_tokens_per_image",
    "tool_call_close_token_id",
    "tool_call_open_token_id",
    "torch_version",
    "transformers_source_sha256",
    "transformers_version",
    "verified_model_file_count",
    "verified_model_total_bytes",
}
RUNTIME_GENERATION_BINDING_KEYS = {
    "assistant_prefix_token_ids",
    "chat_template_file_sha256",
    "chat_template_text_sha256",
    "generation_eos_token_id",
    "generation_interface",
    "generation_num_beams",
    "generation_num_return_sequences",
    "generation_pad_token_id",
    "generation_standard_eos_suppression",
    "host_injected_tool_call_closer",
    "official_tool_schema_sha256",
    "official_tools_argument_count",
    "output_recovery_or_normalization",
    "protocol_id",
    "suppressed_standard_eos_token_ids",
    "tool_call_close_token_id",
    "tool_call_open_token_id",
}

NEGATIVE_OPERATION_COUNTS = {
    "teacher_forward_count": 0,
    "kl_measurement_count": 0,
    "restoration_label_count": 0,
    "expert_action_read_count": 0,
    "confirm_state_generation_count": 0,
    "label_train_state_generation_count": 0,
}


@dataclass(frozen=True)
class PilotGate:
    required_exact_whole_output_parse_count: int
    required_model_emitted_closer_count: int
    required_androidworld_bridge_count: int
    maximum_max_token_truncation_count: int
    maximum_surrounding_prose_count: int
    maximum_action_line_count: int
    maximum_observation_count: int
    maximum_second_json_count: int
    maximum_second_tool_call_count: int
    maximum_retry_count: int
    maximum_top_up_count: int
    pass_outcome: str
    fail_outcome: str
    invalid_outcome: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PilotGate":
        expected = {
            "fixed_state_denominator": EXPECTED_STATE_COUNT,
            "required_exact_whole_output_parse_count": EXPECTED_STATE_COUNT,
            "required_model_emitted_closer_count": EXPECTED_STATE_COUNT,
            "required_androidworld_bridge_count": EXPECTED_STATE_COUNT,
            "maximum_max_token_truncation_count": 0,
            "maximum_surrounding_prose_count": 0,
            "maximum_action_line_count": 0,
            "maximum_observation_count": 0,
            "maximum_second_json_count": 0,
            "maximum_second_tool_call_count": 0,
            "maximum_retry_count": 0,
            "maximum_top_up_count": 0,
            "pass_outcome": PASS_OUTCOME,
            "fail_outcome": NO_GO_OUTCOME,
            "invalid_outcome": INVALID_OUTCOME,
        }
        if dict(value) != expected:
            raise ValueError("v2.1 pilot gate differs from the frozen contract")
        return cls(
            **{
                key: item
                for key, item in expected.items()
                if key != "fixed_state_denominator"
            }
        )


@dataclass(frozen=True)
class AuthorizedPilot:
    repository_root: Path
    contract: RestorationV21PilotContract
    processor_audit: Mapping[str, Any]
    git_identity: Mapping[str, Any]
    source_inventory: Sequence[Mapping[str, str]]
    artifact: ValidatedScreeningArtifact
    states: tuple[ScreeningState, ...]
    snapshot_manifest_path: Path
    target_effective_visual_tokens_per_image: int
    runtime_identity: Mapping[str, Any]
    canonical_inputs: Mapping[str, Any]
    attempt_identity: Mapping[str, Any]


@dataclass(frozen=True)
class RuntimeBindings:
    runtime_class: type[Any]
    parse_error_class: type[BaseException]


@dataclass(frozen=True)
class PilotRunLayout:
    selected: tuple[ScreeningState, ...]
    projections: tuple[dict[str, Any], ...]
    contract: dict[str, Any]
    contract_sha256: str
    root: Path
    manifest_path: Path
    state_root: Path
    attempt_root: Path
    aggregate_path: Path
    attempt_identity: Mapping[str, Any] | None
    ledger_path: Path | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _duration_seconds(started_at_utc: str, ended_at_utc: str) -> float:
    try:
        started = datetime.fromisoformat(started_at_utc.replace("Z", "+00:00"))
        ended = datetime.fromisoformat(ended_at_utc.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("pilot timestamps must be ISO-8601 UTC strings") from error
    if (
        not started_at_utc.endswith("Z")
        or not ended_at_utc.endswith("Z")
        or started.utcoffset() != timezone.utc.utcoffset(started)
        or ended.utcoffset() != timezone.utc.utcoffset(ended)
        or ended < started
    ):
        raise ValueError("pilot timestamps must be ordered UTC timestamps")
    return (ended - started).total_seconds()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {resolved}: {key}")
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {resolved}: {value}")

    value = json.loads(
        resolved.read_bytes(),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain one JSON object")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = _pretty_json_bytes(dict(value))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_repository_path(
    value: str | Path,
    *,
    repository_root: Path,
    relative_path: str,
    name: str,
) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute():
        raise ValueError(f"{name} must be supplied as an absolute path")
    resolved = supplied.resolve()
    expected = (repository_root / relative_path).resolve()
    if resolved != expected or not resolved.is_file():
        raise ValueError(f"{name} must be canonical Git path {relative_path}")
    return resolved


def _validate_absolute_production_paths(args: argparse.Namespace) -> None:
    path_fields = (
        "repository_root",
        "contract",
        "processor_preflight",
        "derived_artifact_root",
        "scientific_config",
        "selection_manifest",
        "ocr_backend_config",
        "model_dir",
        "output_dir",
    )
    for field in path_fields:
        value = getattr(args, field, None)
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError(
                f"production --{field.replace('_', '-')} must be an explicit absolute path"
            )


def validate_canonical_attempt_identity(
    contract: RestorationV21PilotContract,
    output_dir: str | Path,
) -> dict[str, Any]:
    execution = contract.data.get("pilot_execution")
    if not isinstance(execution, Mapping):
        raise ValueError("v2.1 contract lacks pilot_execution")
    expected_attempt_id = "restoration-v2-1-interface-pilot-v1"
    expected_output = Path(
        "/data/experiments/causalcache/restoration-v2-1-interface-pilot-v1"
    )
    if (
        execution.get("attempt_id") != expected_attempt_id
        or execution.get("canonical_persistent_output_dir") != str(expected_output)
        or execution.get("canonical_host_alias") != CANONICAL_PILOT_HOST_ALIAS
        or execution.get("canonical_host_hostname")
        != CANONICAL_PILOT_HOST_HOSTNAME
        or execution.get("canonical_container_id")
        != CANONICAL_PILOT_CONTAINER_ID
        or execution.get("canonical_container_image_digest")
        != CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST
        or execution.get("canonical_device") != CANONICAL_PILOT_DEVICE
        or execution.get("cross_host_attempt_allowed") is not False
        or execution.get("alternate_output_dir_allowed") is not False
        or execution.get("output_directory_deletion_after_first_attempt_allowed")
        is not False
    ):
        raise ValueError("v2.1 canonical no-retry attempt identity drifted")
    supplied = Path(output_dir).resolve()
    if supplied != expected_output:
        raise ValueError("alternate v2.1 pilot output directory is forbidden")
    ledger = expected_output.parent / f".{expected_attempt_id}.attempt.json"
    return {
        "attempt_id": expected_attempt_id,
        "canonical_persistent_output_dir": str(expected_output),
        "global_attempt_ledger": str(ledger),
        "canonical_host_alias": CANONICAL_PILOT_HOST_ALIAS,
        "canonical_host_hostname": CANONICAL_PILOT_HOST_HOSTNAME,
        "canonical_container_id": CANONICAL_PILOT_CONTAINER_ID,
        "canonical_container_image_digest": (
            CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST
        ),
        "canonical_device": CANONICAL_PILOT_DEVICE,
        "cross_host_attempt_allowed": False,
        "alternate_output_dir_allowed": False,
        "output_directory_deletion_after_first_attempt_allowed": False,
    }


def _run_git(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    text: bool = True,
) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout


def validate_clean_pushed_main(repository_root: str | Path) -> dict[str, str]:
    """Require the local, tracking, and currently advertised remote main SHA."""
    root = Path(repository_root).resolve()
    top_level = str(_run_git(root, ("rev-parse", "--show-toplevel"))).strip()
    if Path(top_level).resolve() != root:
        raise ValueError("repository root differs from the Git top level")
    branch = str(_run_git(root, ("branch", "--show-current"))).strip()
    if branch != "main":
        raise ValueError("v2.1 pilot must run from canonical main")
    status = str(
        _run_git(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    )
    if status:
        raise ValueError("v2.1 pilot requires a clean worktree including untracked files")
    head = str(_run_git(root, ("rev-parse", "HEAD"))).strip()
    tracking = str(_run_git(root, ("rev-parse", "refs/remotes/origin/main"))).strip()
    advertised = str(
        _run_git(root, ("ls-remote", "--heads", "origin", "refs/heads/main"))
    ).strip()
    expected_advertised = f"{head}\trefs/heads/main"
    if head != tracking or advertised != expected_advertised:
        raise ValueError("HEAD, origin/main, and remote main must be identical")
    remote_url = str(_run_git(root, ("remote", "get-url", "origin"))).strip()
    if remote_url != CANONICAL_REMOTE_URL:
        raise ValueError("origin URL differs from the canonical repository")
    if GIT_SHA_PATTERN.fullmatch(head) is None:
        raise ValueError("Git HEAD is not a full lowercase commit SHA")
    return {
        "branch": "main",
        "commit": head,
        "origin_main": tracking,
        "remote_main": head,
        "remote_url": remote_url,
        "worktree": "clean_including_untracked",
    }


def validate_committed_source_blobs(
    *,
    repository_root: Path,
    git_commit: str,
    paths: Sequence[str],
) -> list[dict[str, str]]:
    root = repository_root.resolve()
    records: list[dict[str, str]] = []
    if len(paths) != len(set(paths)):
        raise ValueError("source blob path inventory contains duplicates")
    for relative in paths:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"source blob path is missing or escapes Git: {relative}")
        committed = _run_git(
            root,
            ("show", f"{git_commit}:{relative}"),
            text=False,
        )
        if not isinstance(committed, bytes) or committed != path.read_bytes():
            raise ValueError(f"working source differs from committed Git blob: {relative}")
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(committed).hexdigest(),
                "git_commit": git_commit,
            }
        )
    return records


def revalidate_authorized_integrity(authorized: AuthorizedPilot) -> None:
    """Recheck current remote-main identity and every authorized live Git blob."""
    current_git = validate_clean_pushed_main(authorized.repository_root)
    if current_git != dict(authorized.git_identity):
        raise ValueError("authorized Git identity changed before pilot completion")
    paths = tuple(str(record["path"]) for record in authorized.source_inventory)
    current_sources = validate_committed_source_blobs(
        repository_root=authorized.repository_root,
        git_commit=str(current_git["commit"]),
        paths=paths,
    )
    if current_sources != list(authorized.source_inventory):
        raise ValueError("authorized source blob inventory changed before completion")


def _pilot_projection(state: ScreeningState, pilot_index: int) -> dict[str, Any]:
    return {
        "pilot_index": pilot_index,
        "source_state_index": state.index,
        "role": state.role,
        "trajectory_id": state.trajectory_id,
        "decision_step_id": state.decision_step_id,
        "state_id": state.state_id,
        "candidate_event_step_ids": list(state.candidate_event_step_ids),
    }


def select_exact_development_pilot_states(
    artifact: ValidatedScreeningArtifact,
    contract: RestorationV21PilotContract,
) -> tuple[ScreeningState, ...]:
    if len(artifact.states) != SOURCE_STATE_STOP:
        raise ValueError("confirm-safe artifact must expose exactly 45 screening states")
    selected = tuple(artifact.states[SOURCE_STATE_START:SOURCE_STATE_STOP])
    if len(selected) != EXPECTED_STATE_COUNT:
        raise ValueError("v2.1 pilot denominator must contain exactly 15 states")
    projections = [
        _pilot_projection(state, pilot_index)
        for pilot_index, state in enumerate(selected)
    ]
    data = contract.data["data"]
    if any(state.role != "v2_development" for state in selected):
        raise PermissionError("v2.1 pilot may use only v2_development states")
    if [state.index for state in selected] != list(range(30, 45)):
        raise ValueError("v2.1 source state index order drifted")
    if [state.state_id for state in selected] != data["pilot_state_ids"]:
        raise ValueError("v2.1 pilot state ID order drifted")
    projection_sha256 = hashlib.sha256(
        _canonical_json_bytes(projections)
    ).hexdigest()
    if projection_sha256 != data["pilot_state_projection_sha256"]:
        raise ValueError("v2.1 pilot projection SHA256 drifted")
    return selected


def build_confirm_safe_full_history_messages(
    artifact: ValidatedScreeningArtifact,
    state: ScreeningState,
    image_decoder: Callable[[bytes], Any],
) -> list[dict[str, Any]]:
    """Use the exact audited v2.1 reference-prompt builder for generation."""
    if state not in artifact.states or state.role != "v2_development":
        raise PermissionError("pilot message construction is development-only")
    spec = V21ProcessorPromptSpec(
        prompt_index=2 * state.index,
        state=state,
        fidelity="reference",
        restored_event_step_ids=tuple(state.candidate_event_step_ids),
    )
    return build_v2_1_processor_messages(
        artifact,
        spec,
        image_decoder=image_decoder,
    )


def _screen_dimensions(messages: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    images = [
        block["image"]
        for message in messages
        for block in message.get("content", ())
        if isinstance(block, Mapping) and block.get("type") == "image"
    ]
    if not images:
        raise ValueError("pilot messages lack a current observation image")
    size = getattr(images[-1], "size", None)
    if (
        not isinstance(size, Sequence)
        or isinstance(size, (str, bytes, bytearray))
        or len(size) != 2
        or any(type(value) is not int or value <= 0 for value in size)
    ):
        raise ValueError("current observation image lacks valid pixel dimensions")
    return int(size[0]), int(size[1])


def _action_arguments(action: Any) -> dict[str, Any]:
    arguments = action.arguments()
    if not isinstance(arguments, Mapping):
        raise TypeError("canonical action arguments must be a mapping")
    return dict(arguments)


def _attempt_marker(
    *,
    projection: Mapping[str, Any],
    run_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": ATTEMPT_STATUS,
        "run_contract_sha256": run_contract_sha256,
        "state": dict(projection),
        "attempt_ordinal": 1,
        "retry_count": 0,
        "top_up_count": 0,
        "created_at_utc": _utc_now(),
    }


def _global_attempt_ledger(
    *,
    attempt_identity: Mapping[str, Any],
    run_contract_sha256: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": GLOBAL_ATTEMPT_STATUS,
        "attempt_identity": dict(attempt_identity),
        "run_contract_sha256": run_contract_sha256,
        "created_at_utc": created_at_utc or _utc_now(),
    }


def _validate_global_attempt_ledger(
    path: Path,
    *,
    attempt_identity: Mapping[str, Any],
    run_contract_sha256: str,
) -> str:
    ledger = _strict_json_object(path)
    if (
        set(ledger)
        != {
            "schema_version",
            "protocol_id",
            "status",
            "attempt_identity",
            "run_contract_sha256",
            "created_at_utc",
        }
        or ledger["schema_version"] != SCHEMA_VERSION
        or ledger["protocol_id"] != PROTOCOL_ID
        or ledger["status"] != GLOBAL_ATTEMPT_STATUS
        or ledger["attempt_identity"] != dict(attempt_identity)
        or ledger["run_contract_sha256"] != run_contract_sha256
        or not isinstance(ledger["created_at_utc"], str)
        or not ledger["created_at_utc"]
    ):
        raise ValueError("global no-retry attempt ledger identity drifted")
    _duration_seconds(ledger["created_at_utc"], ledger["created_at_utc"])
    return ledger["created_at_utc"]


def _validate_run_manifest(
    manifest: Mapping[str, Any],
    *,
    run_contract: Mapping[str, Any],
    run_contract_sha256: str,
) -> str:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "run_contract_sha256",
        "run_contract",
        "created_at_utc",
    }
    created_at_utc = manifest.get("created_at_utc")
    if (
        set(manifest) != expected_keys
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != RUN_STATUS
        or manifest.get("run_contract_sha256") != run_contract_sha256
        or manifest.get("run_contract") != dict(run_contract)
        or not isinstance(created_at_utc, str)
    ):
        raise ValueError("run manifest identity drifted")
    _duration_seconds(created_at_utc, created_at_utc)
    return created_at_utc


def _runtime_identity_record(
    *,
    run_contract: Mapping[str, Any],
    run_contract_sha256: str,
    runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = dict(runtime_metadata)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "VALIDATED_POLICY_RUNTIME_AFTER_DURABLE_CLAIM",
        "run_contract_sha256": run_contract_sha256,
        "runtime_metadata_sha256": hashlib.sha256(
            _canonical_json_bytes(metadata)
        ).hexdigest(),
        "runtime_metadata": metadata,
        "created_at_utc": _utc_now(),
    }


def _validate_runtime_identity_record(
    record: Mapping[str, Any],
    *,
    run_contract: Mapping[str, Any],
    run_contract_sha256: str,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "run_contract_sha256",
        "runtime_metadata_sha256",
        "runtime_metadata",
        "created_at_utc",
    }
    metadata = record.get("runtime_metadata")
    policy = run_contract.get("policy")
    requirement = (
        policy.get("runtime_metadata_requirement")
        if isinstance(policy, Mapping)
        else None
    )
    if not isinstance(metadata, Mapping) or not isinstance(requirement, Mapping):
        raise ValueError("runtime identity metadata contract is missing")
    metadata_value = dict(metadata)
    expected_metadata_sha256 = hashlib.sha256(
        _canonical_json_bytes(metadata_value)
    ).hexdigest()
    if (
        set(record) != expected_keys
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("protocol_id") != PROTOCOL_ID
        or record.get("status")
        != "VALIDATED_POLICY_RUNTIME_AFTER_DURABLE_CLAIM"
        or record.get("run_contract_sha256") != run_contract_sha256
        or record.get("runtime_metadata_sha256") != expected_metadata_sha256
        or set(metadata_value) != RUNTIME_METADATA_KEYS
        or requirement.get("required_metadata_keys")
        != sorted(RUNTIME_METADATA_KEYS)
        or metadata_value.get("protocol_id") != PROTOCOL_ID
        or metadata_value.get("model_dir") != requirement.get("model_dir")
        or metadata_value.get("model_repo") != requirement.get("model_repo")
        or metadata_value.get("model_revision") != requirement.get("model_revision")
        or metadata_value.get("snapshot_manifest_sha256")
        != requirement.get("snapshot_manifest_sha256")
        or metadata_value.get("device") != requirement.get("device")
        or metadata_value.get("target_effective_visual_tokens_per_image")
        != requirement.get("target_effective_visual_tokens_per_image")
        or metadata_value.get("frozen") is not True
        or metadata_value.get("single_device") is not True
        or not isinstance(record.get("created_at_utc"), str)
    ):
        raise ValueError("validated policy runtime identity drifted")
    _duration_seconds(record["created_at_utc"], record["created_at_utc"])
    return metadata_value


def _validate_attempt_marker(
    path: Path,
    *,
    projection: Mapping[str, Any],
    run_contract_sha256: str,
) -> dict[str, Any]:
    marker = _strict_json_object(path)
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "run_contract_sha256",
        "state",
        "attempt_ordinal",
        "retry_count",
        "top_up_count",
        "created_at_utc",
    }
    if set(marker) != expected_keys:
        raise ValueError("attempt marker keys drifted")
    if (
        marker["schema_version"] != SCHEMA_VERSION
        or marker["protocol_id"] != PROTOCOL_ID
        or marker["status"] != ATTEMPT_STATUS
        or marker["run_contract_sha256"] != run_contract_sha256
        or marker["state"] != dict(projection)
        or marker["attempt_ordinal"] != 1
        or marker["retry_count"] != 0
        or marker["top_up_count"] != 0
        or not isinstance(marker["created_at_utc"], str)
        or not marker["created_at_utc"]
    ):
        raise ValueError("attempt marker identity or no-retry fields drifted")
    _duration_seconds(marker["created_at_utc"], marker["created_at_utc"])
    return marker


def _base_state_record(
    *,
    projection: Mapping[str, Any],
    run_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_contract_sha256": run_contract_sha256,
        "state": dict(projection),
        "fidelity": "full_history_reference",
        "restored_event_step_ids": list(projection["candidate_event_step_ids"]),
        "outcome": STATE_OUTCOME_FAILED,
        "failure": None,
        "generation_call_count": 1,
        "retry_count": 0,
        "top_up_count": 0,
        "raw_output": None,
        "generation_metadata": None,
        "canonical_action": None,
        "screen_dimensions": None,
        "androidworld_bridge": None,
        "negative_operation_counts": dict(NEGATIVE_OPERATION_COUNTS),
        "started_at_utc": _utc_now(),
        "ended_at_utc": None,
        "duration_seconds": None,
    }


def run_pilot_state_once(
    *,
    artifact: ValidatedScreeningArtifact,
    state: ScreeningState,
    pilot_index: int,
    runtime: Any,
    parse_error_class: type[BaseException],
    image_decoder: Callable[[bytes], Any],
    message_builder: Callable[
        [ValidatedScreeningArtifact, ScreeningState, Callable[[bytes], Any]],
        list[dict[str, Any]],
    ],
    run_contract_sha256: str,
) -> dict[str, Any]:
    """Make the one authorized generation call and preserve its native evidence."""
    started = time.perf_counter()
    projection = _pilot_projection(state, pilot_index)
    record = _base_state_record(
        projection=projection,
        run_contract_sha256=run_contract_sha256,
    )
    messages = message_builder(artifact, state, image_decoder)
    screen_width, screen_height = _screen_dimensions(messages)
    try:
        generation = runtime.generate_native_action(messages)
    except parse_error_class as error:
        output_text = getattr(error, "output_text", None)
        metadata = getattr(error, "metadata", None)
        if not isinstance(output_text, str) or not isinstance(metadata, Mapping):
            raise RuntimeError("generation parse error did not preserve raw evidence") from error
        record["raw_output"] = output_text
        record["generation_metadata"] = dict(metadata)
        record["failure"] = {
            "stage": "full_history_generation",
            "category": "PARSE_FAILURE",
            "exception_type": str(
                getattr(error, "parse_error_type", error.__class__.__name__)
            ),
            "message": str(getattr(error, "parse_error_message", str(error))),
        }
        record["ended_at_utc"] = _utc_now()
        record["duration_seconds"] = time.perf_counter() - started
        return record
    output_text = generation.output_text
    metadata = generation.metadata
    if not isinstance(output_text, str) or not isinstance(metadata, Mapping):
        raise RuntimeError("runtime generation result did not preserve native evidence")
    parsed = parse_gui_owl_v2_1_output(output_text)
    runtime_arguments = _action_arguments(generation.parsed_output.canonical_action)
    parsed_arguments = _action_arguments(parsed.canonical_action)
    if runtime_arguments != parsed_arguments:
        raise RuntimeError("runtime and independent parser canonical actions differ")
    bridge = gui_owl_v2_action_to_androidworld(
        parsed.canonical_action,
        screen_width=screen_width,
        screen_height=screen_height,
    )
    record.update(
        {
            "outcome": STATE_OUTCOME_VALID,
            "raw_output": output_text,
            "generation_metadata": dict(metadata),
            "canonical_action": parsed_arguments,
            "screen_dimensions": {
                "width": screen_width,
                "height": screen_height,
            },
            "androidworld_bridge": bridge,
            "ended_at_utc": _utc_now(),
            "duration_seconds": time.perf_counter() - started,
        }
    )
    return record


def _state_record_keys() -> set[str]:
    return {
        "schema_version",
        "protocol_id",
        "run_contract_sha256",
        "state",
        "fidelity",
        "restored_event_step_ids",
        "outcome",
        "failure",
        "generation_call_count",
        "retry_count",
        "top_up_count",
        "raw_output",
        "generation_metadata",
        "canonical_action",
        "screen_dimensions",
        "androidworld_bridge",
        "negative_operation_counts",
        "started_at_utc",
        "ended_at_utc",
        "duration_seconds",
    }


def _validate_terminal_state_record(
    record: Mapping[str, Any],
    *,
    projection: Mapping[str, Any],
    run_contract_sha256: str,
) -> None:
    if set(record) != _state_record_keys():
        raise ValueError("pilot terminal state record keys drifted")
    if (
        record["schema_version"] != SCHEMA_VERSION
        or record["protocol_id"] != PROTOCOL_ID
        or record["run_contract_sha256"] != run_contract_sha256
        or record["state"] != dict(projection)
        or record["fidelity"] != "full_history_reference"
        or record["restored_event_step_ids"]
        != list(projection["candidate_event_step_ids"])
    ):
        raise ValueError("pilot terminal state identity or full-history fidelity drifted")
    if (
        record["generation_call_count"] != 1
        or record["retry_count"] != 0
        or record["top_up_count"] != 0
        or record["negative_operation_counts"] != NEGATIVE_OPERATION_COUNTS
    ):
        raise ValueError("pilot call count, retry, top-up, or forbidden operation drifted")
    if record["outcome"] not in {STATE_OUTCOME_VALID, STATE_OUTCOME_FAILED}:
        raise ValueError("pilot state outcome is not terminal")
    if (
        not isinstance(record["raw_output"], str)
        or not isinstance(record["generation_metadata"], Mapping)
        or not isinstance(record["started_at_utc"], str)
        or not isinstance(record["ended_at_utc"], str)
        or not isinstance(record["duration_seconds"], (int, float))
        or isinstance(record["duration_seconds"], bool)
        or record["duration_seconds"] < 0
    ):
        raise ValueError("pilot terminal state lacks native evidence or timing")
    utc_duration = _duration_seconds(
        record["started_at_utc"],
        record["ended_at_utc"],
    )
    if abs(float(record["duration_seconds"]) - utc_duration) > 1.0:
        raise ValueError("pilot state monotonic duration differs from UTC interval")


def _validate_generation_metadata(
    metadata: Mapping[str, Any],
    *,
    raw_output: str,
) -> None:
    expected = {
        "protocol_id": PROTOCOL_ID,
        "generation_interface": "processor_apply_chat_template_official_tools_kwarg",
        "official_tools_argument_count": 1,
        "tool_call_close_token_id": 151658,
        "generation_eos_token_id": 151658,
        "suppressed_standard_eos_token_ids": [151645, 151643],
        "generation_pad_token_id": 151643,
        "generation_num_beams": 1,
        "generation_num_return_sequences": 1,
        "host_injected_tool_call_closer": False,
        "output_recovery_or_normalization": False,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "return_dict_in_generate": False,
        "max_new_tokens": 256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"generation metadata {key} drifted")
    digest = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
    if metadata.get("decoded_output_utf8_sha256") != digest:
        raise ValueError("raw output differs from runtime output SHA256")
    closer_count = metadata.get("generated_tool_call_close_token_count")
    if type(closer_count) is not int or closer_count < 0:
        raise ValueError("generation closer count metadata is invalid")
    reason = metadata.get("termination_reason")
    if reason not in {
        "model_emitted_tool_call_close",
        "max_new_tokens_without_tool_call_close",
    }:
        raise ValueError("generation termination reason is invalid")
    if metadata.get("model_emitted_tool_call_close") is not (closer_count == 1):
        raise ValueError("generation closer boolean differs from token count")
    final_id = metadata.get("final_generated_token_id")
    if final_id is not None and type(final_id) is not int:
        raise ValueError("final generated token ID metadata is invalid")


def _json_object_count(text: str) -> int:
    decoder = json.JSONDecoder()
    count = 0
    cursor = 0
    while cursor < len(text):
        opening = text.find("{", cursor)
        if opening < 0:
            break
        try:
            _, end = decoder.raw_decode(text, opening)
        except json.JSONDecodeError:
            cursor = opening + 1
            continue
        count += 1
        cursor = end
    return count


def _diagnose_native_output(raw_output: str, metadata: Mapping[str, Any]) -> dict[str, bool]:
    open_count = raw_output.count("<tool_call>")
    close_count = raw_output.count("</tool_call>")
    has_single_pair = open_count == 1 and close_count == 1
    if has_single_pair:
        opening = raw_output.find("<tool_call>")
        closing_end = raw_output.find("</tool_call>") + len("</tool_call>")
        surrounding_prose = opening != 0 or closing_end != len(raw_output)
    else:
        surrounding_prose = False
    return {
        "model_emitted_closer": (
            metadata.get("model_emitted_tool_call_close") is True
            and metadata.get("generated_tool_call_close_token_count") == 1
            and metadata.get("final_generated_token_id") == 151658
            and metadata.get("termination_reason") == "model_emitted_tool_call_close"
            and close_count == 1
            and raw_output.endswith("</tool_call>")
        ),
        "max_token_truncation": (
            metadata.get("termination_reason")
            == "max_new_tokens_without_tool_call_close"
        ),
        "surrounding_prose": surrounding_prose,
        "action_line": re.search(r"(?m)^Action:", raw_output) is not None,
        "observation": (
            "<observation>" in raw_output or "</observation>" in raw_output
        ),
        "second_json": _json_object_count(raw_output) > 1,
        "second_tool_call": open_count > 1 or close_count > 1,
    }


def aggregate_pilot_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_states: Sequence[ScreeningState],
    gate: PilotGate,
    run_contract_sha256: str,
    started_at_utc: str,
    ended_at_utc: str,
) -> dict[str, Any]:
    """Recompute every gate metric from the fixed native state records."""
    if len(expected_states) != EXPECTED_STATE_COUNT or len(records) != EXPECTED_STATE_COUNT:
        raise ValueError("v2.1 gate requires the exact 15-state denominator")
    duration_seconds = _duration_seconds(started_at_utc, ended_at_utc)
    parse_count = 0
    closer_count = 0
    bridge_count = 0
    truncation_count = 0
    surrounding_count = 0
    action_line_count = 0
    observation_count = 0
    second_json_count = 0
    second_tool_call_count = 0
    retry_count = 0
    top_up_count = 0
    failure_counts: dict[str, int] = {}
    for pilot_index, (record, state) in enumerate(zip(records, expected_states, strict=True)):
        projection = _pilot_projection(state, pilot_index)
        _validate_terminal_state_record(
            record,
            projection=projection,
            run_contract_sha256=run_contract_sha256,
        )
        raw_output = record["raw_output"]
        metadata = record["generation_metadata"]
        _validate_generation_metadata(metadata, raw_output=raw_output)
        diagnostics = _diagnose_native_output(raw_output, metadata)
        closer_count += int(diagnostics["model_emitted_closer"])
        truncation_count += int(diagnostics["max_token_truncation"])
        surrounding_count += int(diagnostics["surrounding_prose"])
        action_line_count += int(diagnostics["action_line"])
        observation_count += int(diagnostics["observation"])
        second_json_count += int(diagnostics["second_json"])
        second_tool_call_count += int(diagnostics["second_tool_call"])
        retry_count += int(record["retry_count"])
        top_up_count += int(record["top_up_count"])
        try:
            parsed = parse_gui_owl_v2_1_output(raw_output)
        except (TypeError, ValueError):
            if record["outcome"] != STATE_OUTCOME_FAILED:
                raise ValueError("parse failure is recorded as a valid state")
            failure = record["failure"]
            if not isinstance(failure, Mapping) or failure.get("category") != "PARSE_FAILURE":
                raise ValueError("parse failure terminal record lacks PARSE_FAILURE evidence")
            failure_counts["PARSE_FAILURE"] = failure_counts.get("PARSE_FAILURE", 0) + 1
            if any(
                record[field] is not None
                for field in (
                    "canonical_action",
                    "screen_dimensions",
                    "androidworld_bridge",
                )
            ):
                raise ValueError("parse failure record contains post-parse evidence")
            continue
        if record["outcome"] != STATE_OUTCOME_VALID or record["failure"] is not None:
            raise ValueError("parseable output is not recorded as a valid terminal state")
        parse_count += 1
        arguments = _action_arguments(parsed.canonical_action)
        if record["canonical_action"] != arguments:
            raise ValueError("stored canonical action differs from raw output")
        dimensions = record["screen_dimensions"]
        if not isinstance(dimensions, Mapping):
            raise ValueError("valid state lacks screen dimensions")
        width = dimensions.get("width")
        height = dimensions.get("height")
        if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
            raise ValueError("stored screen dimensions are invalid")
        bridge = gui_owl_v2_action_to_androidworld(
            parsed.canonical_action,
            screen_width=width,
            screen_height=height,
        )
        if record["androidworld_bridge"] != bridge:
            raise ValueError("stored AndroidWorld bridge differs from raw output")
        bridge_count += 1

    metrics = {
        "fixed_state_denominator": EXPECTED_STATE_COUNT,
        "exact_whole_output_parse_count": parse_count,
        "model_emitted_closer_count": closer_count,
        "androidworld_bridge_count": bridge_count,
        "max_token_truncation_count": truncation_count,
        "surrounding_prose_count": surrounding_count,
        "action_line_count": action_line_count,
        "observation_count": observation_count,
        "second_json_count": second_json_count,
        "second_tool_call_count": second_tool_call_count,
        "generation_call_count": EXPECTED_STATE_COUNT,
        "retry_count": retry_count,
        "top_up_count": top_up_count,
        **NEGATIVE_OPERATION_COUNTS,
    }
    checks = {
        "exact_whole_output_parse": (
            parse_count == gate.required_exact_whole_output_parse_count
        ),
        "model_emitted_closer": (
            closer_count == gate.required_model_emitted_closer_count
        ),
        "androidworld_bridge": (
            bridge_count == gate.required_androidworld_bridge_count
        ),
        "max_token_truncation": (
            truncation_count <= gate.maximum_max_token_truncation_count
        ),
        "surrounding_prose": surrounding_count <= gate.maximum_surrounding_prose_count,
        "action_line": action_line_count <= gate.maximum_action_line_count,
        "observation": observation_count <= gate.maximum_observation_count,
        "second_json": second_json_count <= gate.maximum_second_json_count,
        "second_tool_call": (
            second_tool_call_count <= gate.maximum_second_tool_call_count
        ),
        "retry": retry_count <= gate.maximum_retry_count,
        "top_up": top_up_count <= gate.maximum_top_up_count,
    }
    passed = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_contract_sha256": run_contract_sha256,
        "status": "COMPLETED_FIXED_15_STATE_INTERFACE_PILOT",
        "outcome": gate.pass_outcome if passed else gate.fail_outcome,
        "gate_passed": passed,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_seconds": duration_seconds,
        "metrics": metrics,
        "checks": checks,
        "failure_category_counts": dict(sorted(failure_counts.items())),
        "sample_mutation_performed": False,
        "top_up_performed": False,
        "confirm_role_used": False,
        "label_train_role_used": False,
    }


def _invalid_aggregate(
    *,
    run_contract_sha256: str,
    stage: str,
    error: BaseException,
    completed_state_count: int,
    attempted_state_count: int,
    started_at_utc: str,
) -> dict[str, Any]:
    ended_at_utc = _utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_contract_sha256": run_contract_sha256,
        "status": "TERMINATED_INVALID_FIXED_15_STATE_INTERFACE_PILOT",
        "outcome": INVALID_OUTCOME,
        "gate_passed": False,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_seconds": _duration_seconds(started_at_utc, ended_at_utc),
        "invalid_failure": {
            "stage": stage,
            "category": "OUT_OF_MEMORY" if _is_out_of_memory(error) else "CONTRACT_OR_RUNTIME",
            "exception_type": error.__class__.__name__,
            "message": str(error),
        },
        "completed_state_count": completed_state_count,
        "attempted_state_count": attempted_state_count,
        "retry_count": 0,
        "top_up_count": 0,
        **NEGATIVE_OPERATION_COUNTS,
    }


def _is_out_of_memory(error: BaseException) -> bool:
    return (
        "outofmemory" in error.__class__.__name__.casefold()
        or "out of memory" in str(error).casefold()
    )


def _validate_invalid_aggregate(
    aggregate: Mapping[str, Any],
    *,
    run_contract_sha256: str,
    started_at_utc: str,
) -> None:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "run_contract_sha256",
        "status",
        "outcome",
        "gate_passed",
        "started_at_utc",
        "ended_at_utc",
        "duration_seconds",
        "invalid_failure",
        "completed_state_count",
        "attempted_state_count",
        "retry_count",
        "top_up_count",
        *NEGATIVE_OPERATION_COUNTS,
    }
    failure = aggregate.get("invalid_failure")
    try:
        expected_duration_seconds = _duration_seconds(
            str(aggregate.get("started_at_utc")),
            str(aggregate.get("ended_at_utc")),
        )
    except ValueError as error:
        raise ValueError("existing INVALID aggregate identity drifted") from error
    if (
        set(aggregate) != expected_keys
        or aggregate.get("schema_version") != SCHEMA_VERSION
        or aggregate.get("protocol_id") != PROTOCOL_ID
        or aggregate.get("run_contract_sha256") != run_contract_sha256
        or aggregate.get("outcome") != INVALID_OUTCOME
        or aggregate.get("status")
        != "TERMINATED_INVALID_FIXED_15_STATE_INTERFACE_PILOT"
        or aggregate.get("gate_passed") is not False
        or aggregate.get("started_at_utc") != started_at_utc
        or aggregate.get("duration_seconds") != expected_duration_seconds
        or aggregate.get("retry_count") != 0
        or aggregate.get("top_up_count") != 0
        or any(aggregate.get(key) != 0 for key in NEGATIVE_OPERATION_COUNTS)
        or not isinstance(failure, Mapping)
        or set(failure) != {"stage", "category", "exception_type", "message"}
        or failure.get("category") not in {"OUT_OF_MEMORY", "CONTRACT_OR_RUNTIME"}
        or any(
            not isinstance(failure.get(key), str) or not failure[key]
            for key in ("stage", "exception_type", "message")
        )
        or type(aggregate.get("completed_state_count")) is not int
        or not 0 <= aggregate["completed_state_count"] <= EXPECTED_STATE_COUNT
        or type(aggregate.get("attempted_state_count")) is not int
        or not 0 <= aggregate["attempted_state_count"] <= EXPECTED_STATE_COUNT
        or aggregate["completed_state_count"] > aggregate["attempted_state_count"]
        or aggregate["attempted_state_count"] - aggregate["completed_state_count"] > 1
    ):
        raise ValueError("existing INVALID aggregate identity drifted")


def _validate_invalid_resume_inventory(
    *,
    state_root: Path,
    attempt_root: Path,
    aggregate: Mapping[str, Any],
    states: Sequence[ScreeningState],
    run_contract_sha256: str,
) -> None:
    completed = int(aggregate["completed_state_count"])
    attempted = int(aggregate["attempted_state_count"])
    expected_state_names = {f"{index:03d}.json" for index in range(completed)}
    expected_attempt_names = {f"{index:03d}.json" for index in range(attempted)}
    state_entries = tuple(state_root.iterdir())
    attempt_entries = tuple(attempt_root.iterdir())
    if (
        any(not path.is_file() for path in (*state_entries, *attempt_entries))
        or {path.name for path in state_entries} != expected_state_names
        or {path.name for path in attempt_entries} != expected_attempt_names
    ):
        raise ValueError("INVALID resume state/attempt directory inventory drifted")
    for pilot_index in range(attempted):
        projection = _pilot_projection(states[pilot_index], pilot_index)
        _validate_attempt_marker(
            attempt_root / f"{pilot_index:03d}.json",
            projection=projection,
            run_contract_sha256=run_contract_sha256,
        )
        if pilot_index < completed:
            record = _strict_json_object(
                state_root / f"{pilot_index:03d}.json"
            )
            _validate_terminal_state_record(
                record,
                projection=projection,
                run_contract_sha256=run_contract_sha256,
            )


def _prepare_pilot_layout(
    *,
    states: Sequence[ScreeningState],
    run_contract: Mapping[str, Any],
    output_dir: str | Path,
) -> PilotRunLayout:
    selected = tuple(states)
    if len(selected) != EXPECTED_STATE_COUNT:
        raise ValueError("pilot runner requires exactly 15 states")
    projections = tuple(
        _pilot_projection(state, pilot_index)
        for pilot_index, state in enumerate(selected)
    )
    contract = dict(run_contract)
    if contract.get("states") != list(projections):
        raise ValueError("run contract state denominator drifted")
    if contract.get("negative_operation_counts") != NEGATIVE_OPERATION_COUNTS:
        raise ValueError("run contract forbidden-operation declaration drifted")
    contract_sha256 = hashlib.sha256(_canonical_json_bytes(contract)).hexdigest()
    root = Path(output_dir)
    manifest_path = root / RUN_MANIFEST_FILENAME
    state_root = root / STATE_DIRECTORY
    attempt_root = root / ATTEMPT_DIRECTORY
    aggregate_path = root / AGGREGATE_FILENAME
    attempt_identity = contract.get("attempt_identity")
    ledger_path: Path | None = None
    if attempt_identity is not None:
        if not isinstance(attempt_identity, Mapping):
            raise ValueError("run contract attempt_identity must be a mapping")
        expected_attempt_keys = {
            "attempt_id",
            "canonical_persistent_output_dir",
            "global_attempt_ledger",
            "canonical_host_alias",
            "canonical_host_hostname",
            "canonical_container_id",
            "canonical_container_image_digest",
            "canonical_device",
            "cross_host_attempt_allowed",
            "alternate_output_dir_allowed",
            "output_directory_deletion_after_first_attempt_allowed",
        }
        ledger_path = Path(str(attempt_identity.get("global_attempt_ledger"))).resolve()
        expected_ledger_path = root.resolve().parent / (
            f".{attempt_identity.get('attempt_id')}.attempt.json"
        )
        if (
            set(attempt_identity) != expected_attempt_keys
            or attempt_identity.get("attempt_id")
            != "restoration-v2-1-interface-pilot-v1"
            or Path(
                str(attempt_identity.get("canonical_persistent_output_dir"))
            ).resolve()
            != root.resolve()
            or attempt_identity.get("canonical_host_alias")
            != CANONICAL_PILOT_HOST_ALIAS
            or attempt_identity.get("canonical_host_hostname")
            != CANONICAL_PILOT_HOST_HOSTNAME
            or attempt_identity.get("canonical_container_id")
            != CANONICAL_PILOT_CONTAINER_ID
            or attempt_identity.get("canonical_container_image_digest")
            != CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST
            or attempt_identity.get("canonical_device") != CANONICAL_PILOT_DEVICE
            or attempt_identity.get("cross_host_attempt_allowed") is not False
            or attempt_identity.get("alternate_output_dir_allowed") is not False
            or attempt_identity.get(
                "output_directory_deletion_after_first_attempt_allowed"
            )
            is not False
            or ledger_path != expected_ledger_path
        ):
            raise ValueError("run contract canonical attempt identity drifted")
    return PilotRunLayout(
        selected=selected,
        projections=projections,
        contract=contract,
        contract_sha256=contract_sha256,
        root=root,
        manifest_path=manifest_path,
        state_root=state_root,
        attempt_root=attempt_root,
        aggregate_path=aggregate_path,
        attempt_identity=attempt_identity,
        ledger_path=ledger_path,
    )


def _validate_resume_directory_inventory(
    *,
    state_root: Path,
    attempt_root: Path,
    require_completed: bool,
) -> None:
    allowed_names = tuple(f"{index:03d}.json" for index in range(EXPECTED_STATE_COUNT))
    allowed = set(allowed_names)
    state_entries = tuple(state_root.iterdir())
    attempt_entries = tuple(attempt_root.iterdir())
    state_names = {path.name for path in state_entries}
    attempt_names = {path.name for path in attempt_entries}
    if (
        any(not path.is_file() for path in (*state_entries, *attempt_entries))
        or not state_names <= allowed
        or not attempt_names <= allowed
        or state_names != set(allowed_names[: len(state_names)])
        or attempt_names != set(allowed_names[: len(attempt_names)])
        or not state_names <= attempt_names
        or len(attempt_names) - len(state_names) > 1
        or (require_completed and (state_names != allowed or attempt_names != allowed))
    ):
        raise ValueError("resume state/attempt directory inventory drifted")


def _run_manifest_record(
    layout: PilotRunLayout,
    *,
    created_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": RUN_STATUS,
        "run_contract_sha256": layout.contract_sha256,
        "run_contract": layout.contract,
        "created_at_utc": created_at_utc,
    }


def _ensure_recoverable_claim_root(
    layout: PilotRunLayout,
    *,
    ledger_created_at_utc: str,
) -> str:
    if layout.root.exists() and not layout.root.is_dir():
        raise ValueError("canonical pilot root exists but is not a directory")
    layout.root.mkdir(exist_ok=True)
    allowed_root_names = {
        RUN_MANIFEST_FILENAME,
        RUNTIME_IDENTITY_FILENAME,
        STATE_DIRECTORY,
        ATTEMPT_DIRECTORY,
        AGGREGATE_FILENAME,
    }
    if any(path.name not in allowed_root_names for path in layout.root.iterdir()):
        raise ValueError("canonical pilot root contains an unknown bootstrap entry")
    for directory in (layout.state_root, layout.attempt_root):
        if directory.exists() and not directory.is_dir():
            raise ValueError("canonical pilot state/attempt path is not a directory")
        directory.mkdir(exist_ok=True)
    if layout.manifest_path.exists():
        manifest = _strict_json_object(layout.manifest_path)
    else:
        manifest = _run_manifest_record(
            layout,
            created_at_utc=ledger_created_at_utc,
        )
        _write_json_exclusive(layout.manifest_path, manifest)
    return _validate_run_manifest(
        manifest,
        run_contract=layout.contract,
        run_contract_sha256=layout.contract_sha256,
    )


def _validate_completed_terminal_aggregate(
    layout: PilotRunLayout,
    *,
    gate: PilotGate,
    aggregate: Mapping[str, Any],
    started_at_utc: str,
) -> dict[str, Any]:
    runtime_identity_path = layout.root / RUNTIME_IDENTITY_FILENAME
    policy = layout.contract.get("policy")
    requires_runtime_identity = isinstance(policy, Mapping) and isinstance(
        policy.get("runtime_metadata_requirement"), Mapping
    )
    runtime_metadata: dict[str, Any] | None = None
    if requires_runtime_identity:
        if not runtime_identity_path.is_file():
            raise FileNotFoundError("completed pilot lacks validated runtime identity")
        runtime_metadata = _validate_runtime_identity_record(
            _strict_json_object(runtime_identity_path),
            run_contract=layout.contract,
            run_contract_sha256=layout.contract_sha256,
        )
    _validate_resume_directory_inventory(
        state_root=layout.state_root,
        attempt_root=layout.attempt_root,
        require_completed=True,
    )
    records = [
        _strict_json_object(layout.state_root / f"{index:03d}.json")
        for index in range(EXPECTED_STATE_COUNT)
    ]
    if runtime_metadata is not None:
        for record in records:
            generation_metadata = record.get("generation_metadata")
            if not isinstance(generation_metadata, Mapping) or any(
                generation_metadata.get(key) != runtime_metadata[key]
                for key in RUNTIME_GENERATION_BINDING_KEYS
            ):
                raise ValueError("state generation metadata differs from runtime identity")
    ended_at_utc = aggregate.get("ended_at_utc")
    if not isinstance(ended_at_utc, str):
        raise ValueError("completed aggregate end timestamp drifted")
    recomputed = aggregate_pilot_gate(
        records,
        expected_states=layout.selected,
        gate=gate,
        run_contract_sha256=layout.contract_sha256,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
    )
    if dict(aggregate) != recomputed:
        raise ValueError("completed aggregate differs from the exact raw denominator")
    return recomputed


def _persist_claimed_invalid(
    layout: PilotRunLayout,
    *,
    stage: str,
    error: BaseException,
) -> dict[str, Any]:
    if layout.ledger_path is not None:
        ledger_started_at_utc = _validate_global_attempt_ledger(
            layout.ledger_path,
            attempt_identity=layout.attempt_identity,
            run_contract_sha256=layout.contract_sha256,
        )
    else:
        if not layout.manifest_path.is_file():
            raise FileNotFoundError("claimed pilot lacks a canonical run manifest")
        ledger_started_at_utc = _strict_json_object(layout.manifest_path)[
            "created_at_utc"
        ]
    started_at_utc = _ensure_recoverable_claim_root(
        layout,
        ledger_created_at_utc=ledger_started_at_utc,
    )
    runtime_identity_path = layout.root / RUNTIME_IDENTITY_FILENAME
    if runtime_identity_path.exists():
        _validate_runtime_identity_record(
            _strict_json_object(runtime_identity_path),
            run_contract=layout.contract,
            run_contract_sha256=layout.contract_sha256,
        )
    _validate_resume_directory_inventory(
        state_root=layout.state_root,
        attempt_root=layout.attempt_root,
        require_completed=False,
    )
    completed_state_count = len(tuple(layout.state_root.iterdir()))
    attempted_state_count = len(tuple(layout.attempt_root.iterdir()))
    if layout.aggregate_path.exists():
        raise FileExistsError("claimed pilot already has a terminal aggregate")
    invalid = _invalid_aggregate(
        run_contract_sha256=layout.contract_sha256,
        stage=stage,
        error=error,
        completed_state_count=completed_state_count,
        attempted_state_count=attempted_state_count,
        started_at_utc=started_at_utc,
    )
    _write_json_exclusive(layout.aggregate_path, invalid)
    _validate_invalid_aggregate(
        invalid,
        run_contract_sha256=layout.contract_sha256,
        started_at_utc=started_at_utc,
    )
    _validate_invalid_resume_inventory(
        state_root=layout.state_root,
        attempt_root=layout.attempt_root,
        aggregate=invalid,
        states=layout.selected,
        run_contract_sha256=layout.contract_sha256,
    )
    return invalid


def claim_interface_pilot(
    *,
    states: Sequence[ScreeningState],
    gate: PilotGate,
    run_contract: Mapping[str, Any],
    output_dir: str | Path,
    resume: bool,
) -> dict[str, Any] | None:
    """Claim before policy import, or make resume terminal without generation."""
    layout = _prepare_pilot_layout(
        states=states,
        run_contract=run_contract,
        output_dir=output_dir,
    )
    if not resume:
        if layout.root.exists() or (
            layout.ledger_path is not None and layout.ledger_path.exists()
        ):
            raise FileExistsError("canonical no-retry pilot attempt already exists")
        started_at_utc = _utc_now()
        if layout.ledger_path is not None:
            if not layout.root.parent.is_dir():
                raise FileNotFoundError("canonical pilot output parent must already exist")
            _write_json_exclusive(
                layout.ledger_path,
                _global_attempt_ledger(
                    attempt_identity=layout.attempt_identity,
                    run_contract_sha256=layout.contract_sha256,
                    created_at_utc=started_at_utc,
                ),
            )
            layout.root.mkdir(exist_ok=False)
        else:
            layout.root.mkdir(parents=True, exist_ok=False)
        layout.state_root.mkdir()
        layout.attempt_root.mkdir()
        manifest = _run_manifest_record(layout, created_at_utc=started_at_utc)
        _write_json_exclusive(layout.manifest_path, manifest)
        _validate_run_manifest(
            manifest,
            run_contract=layout.contract,
            run_contract_sha256=layout.contract_sha256,
        )
        return None

    if layout.ledger_path is not None:
        if not layout.ledger_path.is_file():
            raise FileNotFoundError("resume requires the global no-retry ledger")
        ledger_started_at_utc = _validate_global_attempt_ledger(
            layout.ledger_path,
            attempt_identity=layout.attempt_identity,
            run_contract_sha256=layout.contract_sha256,
        )
    else:
        if not layout.manifest_path.is_file():
            raise FileNotFoundError("resume lacks a durable attempt claim")
        ledger_started_at_utc = _strict_json_object(layout.manifest_path)[
            "created_at_utc"
        ]
    started_at_utc = _ensure_recoverable_claim_root(
        layout,
        ledger_created_at_utc=ledger_started_at_utc,
    )
    runtime_identity_path = layout.root / RUNTIME_IDENTITY_FILENAME
    if runtime_identity_path.exists():
        _validate_runtime_identity_record(
            _strict_json_object(runtime_identity_path),
            run_contract=layout.contract,
            run_contract_sha256=layout.contract_sha256,
        )
    if layout.aggregate_path.is_file():
        aggregate = _strict_json_object(layout.aggregate_path)
        if aggregate.get("outcome") == INVALID_OUTCOME:
            _validate_invalid_aggregate(
                aggregate,
                run_contract_sha256=layout.contract_sha256,
                started_at_utc=started_at_utc,
            )
            _validate_invalid_resume_inventory(
                state_root=layout.state_root,
                attempt_root=layout.attempt_root,
                aggregate=aggregate,
                states=layout.selected,
                run_contract_sha256=layout.contract_sha256,
            )
            return aggregate
        if aggregate.get("outcome") in {PASS_OUTCOME, NO_GO_OUTCOME}:
            return _validate_completed_terminal_aggregate(
                layout,
                gate=gate,
                aggregate=aggregate,
                started_at_utc=started_at_utc,
            )
        raise ValueError("existing aggregate has an unknown terminal outcome")
    return _persist_claimed_invalid(
        layout,
        stage="resume_recovered_nonterminal_no_retry_claim",
        error=RuntimeError(
            "prior pilot process ended without a terminal aggregate; generation is forbidden"
        ),
    )


def run_interface_pilot(
    *,
    artifact: ValidatedScreeningArtifact,
    states: Sequence[ScreeningState],
    runtime: Any,
    parse_error_class: type[BaseException],
    image_decoder: Callable[[bytes], Any],
    gate: PilotGate,
    run_contract: Mapping[str, Any],
    output_dir: str | Path,
    resume: bool,
    preclaimed: bool = False,
    integrity_guard: Callable[[], None],
    message_builder: Callable[
        [ValidatedScreeningArtifact, ScreeningState, Callable[[bytes], Any]],
        list[dict[str, Any]],
    ] = build_confirm_safe_full_history_messages,
) -> dict[str, Any]:
    """Run or read-only resume the exact denominator without retry or top-up."""
    if type(resume) is not bool:
        raise TypeError("resume must be bool")
    if type(preclaimed) is not bool:
        raise TypeError("preclaimed must be bool")
    if resume and preclaimed:
        raise ValueError("resume cannot enter the generation runner as preclaimed")
    if not callable(integrity_guard):
        raise TypeError("integrity_guard must be callable")
    layout = _prepare_pilot_layout(
        states=states,
        run_contract=run_contract,
        output_dir=output_dir,
    )
    selected = layout.selected
    projections = list(layout.projections)
    contract = layout.contract
    contract_sha256 = layout.contract_sha256
    root = layout.root
    manifest_path = layout.manifest_path
    state_root = layout.state_root
    attempt_root = layout.attempt_root
    aggregate_path = layout.aggregate_path
    attempt_identity = layout.attempt_identity
    ledger_path = layout.ledger_path
    if resume:
        terminal = claim_interface_pilot(
            states=selected,
            gate=gate,
            run_contract=contract,
            output_dir=root,
            resume=True,
        )
        if terminal is None:
            raise AssertionError("resume must always return terminal evidence")
        return terminal
    if not preclaimed:
        claim_interface_pilot(
            states=selected,
            gate=gate,
            run_contract=contract,
            output_dir=root,
            resume=False,
        )
    if ledger_path is not None:
        _validate_global_attempt_ledger(
            ledger_path,
            attempt_identity=attempt_identity,
            run_contract_sha256=contract_sha256,
        )
    if not root.is_dir() or not state_root.is_dir() or not attempt_root.is_dir():
        raise FileNotFoundError("preclaimed pilot directory is incomplete")
    manifest = _strict_json_object(manifest_path)
    run_started_at_utc = _validate_run_manifest(
        manifest,
        run_contract=contract,
        run_contract_sha256=contract_sha256,
    )
    policy = contract.get("policy")
    runtime_identity_path_for_generation: Path | None = None
    if isinstance(policy, Mapping) and isinstance(
        policy.get("runtime_metadata_requirement"), Mapping
    ):
        runtime_identity_path = root / RUNTIME_IDENTITY_FILENAME
        if not runtime_identity_path.is_file():
            raise FileNotFoundError("preclaimed pilot lacks runtime identity")
        _validate_runtime_identity_record(
            _strict_json_object(runtime_identity_path),
            run_contract=contract,
            run_contract_sha256=contract_sha256,
        )
        runtime_identity_path_for_generation = runtime_identity_path
    _validate_resume_directory_inventory(
        state_root=state_root,
        attempt_root=attempt_root,
        require_completed=False,
    )
    if (
        any(state_root.iterdir())
        or any(attempt_root.iterdir())
        or aggregate_path.exists()
    ):
        raise ValueError("preclaimed generation must begin from an empty denominator")

    records: list[dict[str, Any]] = []
    for pilot_index, state in enumerate(selected):
        projection = projections[pilot_index]
        state_path = state_root / f"{pilot_index:03d}.json"
        attempt_path = attempt_root / f"{pilot_index:03d}.json"
        if state_path.exists():
            if not resume or not attempt_path.is_file():
                raise ValueError("terminal state cannot exist without a resume marker")
            _validate_attempt_marker(
                attempt_path,
                projection=projection,
                run_contract_sha256=contract_sha256,
            )
            record = _strict_json_object(state_path)
            _validate_terminal_state_record(
                record,
                projection=projection,
                run_contract_sha256=contract_sha256,
            )
            records.append(record)
            continue
        if attempt_path.exists():
            _validate_attempt_marker(
                attempt_path,
                projection=projection,
                run_contract_sha256=contract_sha256,
            )
            raise RuntimeError("incomplete prior attempt cannot be retried or topped up")
        try:
            integrity_guard()
            if runtime_identity_path_for_generation is not None:
                _validate_runtime_identity_record(
                    _strict_json_object(runtime_identity_path_for_generation),
                    run_contract=contract,
                    run_contract_sha256=contract_sha256,
                )
        except Exception as error:
            invalid = _invalid_aggregate(
                run_contract_sha256=contract_sha256,
                stage=f"pre_attempt_integrity_{pilot_index:03d}",
                error=error,
                completed_state_count=len(records),
                attempted_state_count=pilot_index,
                started_at_utc=run_started_at_utc,
            )
            _write_json_exclusive(aggregate_path, invalid)
            return invalid
        _write_json_exclusive(
            attempt_path,
            _attempt_marker(
                projection=projection,
                run_contract_sha256=contract_sha256,
            ),
        )
        try:
            integrity_guard()
            if runtime_identity_path_for_generation is not None:
                _validate_runtime_identity_record(
                    _strict_json_object(runtime_identity_path_for_generation),
                    run_contract=contract,
                    run_contract_sha256=contract_sha256,
                )
        except Exception as error:
            invalid = _invalid_aggregate(
                run_contract_sha256=contract_sha256,
                stage=f"pre_generation_integrity_{pilot_index:03d}",
                error=error,
                completed_state_count=len(records),
                attempted_state_count=pilot_index + 1,
                started_at_utc=run_started_at_utc,
            )
            _write_json_exclusive(aggregate_path, invalid)
            return invalid
        try:
            record = run_pilot_state_once(
                artifact=artifact,
                state=state,
                pilot_index=pilot_index,
                runtime=runtime,
                parse_error_class=parse_error_class,
                image_decoder=image_decoder,
                message_builder=message_builder,
                run_contract_sha256=contract_sha256,
            )
            _validate_terminal_state_record(
                record,
                projection=projection,
                run_contract_sha256=contract_sha256,
            )
            _write_json_exclusive(state_path, record)
        except parse_error_class:
            raise AssertionError("parse errors must become terminal scientific records")
        except Exception as error:
            invalid = _invalid_aggregate(
                run_contract_sha256=contract_sha256,
                stage=f"state_{pilot_index:03d}",
                error=error,
                completed_state_count=len(records),
                attempted_state_count=pilot_index + 1,
                started_at_utc=run_started_at_utc,
            )
            _write_json_exclusive(aggregate_path, invalid)
            return invalid
        records.append(record)

    try:
        integrity_guard()
    except Exception as error:
        invalid = _invalid_aggregate(
            run_contract_sha256=contract_sha256,
            stage="post_generation_integrity",
            error=error,
            completed_state_count=len(records),
            attempted_state_count=len(records),
            started_at_utc=run_started_at_utc,
        )
        if aggregate_path.exists():
            raise ValueError("existing aggregate blocks integrity INVALID") from error
        _write_json_exclusive(aggregate_path, invalid)
        return invalid
    ended_at_utc = _utc_now()
    try:
        aggregate = aggregate_pilot_gate(
            records,
            expected_states=selected,
            gate=gate,
            run_contract_sha256=contract_sha256,
            started_at_utc=run_started_at_utc,
            ended_at_utc=ended_at_utc,
        )
    except Exception as error:
        invalid = _invalid_aggregate(
            run_contract_sha256=contract_sha256,
            stage="aggregate_recomputation",
            error=error,
            completed_state_count=len(records),
            attempted_state_count=len(records),
            started_at_utc=run_started_at_utc,
        )
        if aggregate_path.exists():
            raise ValueError(
                "existing aggregate cannot be replaced after invalid recomputation"
            ) from error
        _write_json_exclusive(aggregate_path, invalid)
        return invalid
    if aggregate_path.exists():
        raise ValueError("terminal aggregate was created during active generation")
    _write_json_exclusive(aggregate_path, aggregate)
    return aggregate


def _load_processor_audit_validator() -> Callable[..., Mapping[str, Any]]:
    from causalcache.restoration_v2_1_processor_audit import (
        validate_restoration_v2_1_processor_audit,
    )

    return validate_restoration_v2_1_processor_audit


def _load_runtime_bindings() -> RuntimeBindings:
    from causalcache.policy.gui_owl_v2_1_runtime import (
        GUIOwlV21GenerationParseError,
        GUIOwlV21OfficialToolsRuntime,
    )

    return RuntimeBindings(
        runtime_class=GUIOwlV21OfficialToolsRuntime,
        parse_error_class=GUIOwlV21GenerationParseError,
    )


def _decode_rgb_image(payload: bytes) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def _load_live_runtime_identity(*, device: str) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("v2.1 pilot requires the pinned PyTorch runtime") from error
    if CUDA_DEVICE_PATTERN.fullmatch(device) is None or not torch.cuda.is_available():
        raise RuntimeError("v2.1 pilot requires one explicit available CUDA device")
    device_index = int(device.split(":", maxsplit=1)[1])
    visible_count = int(torch.cuda.device_count())
    if device_index >= visible_count:
        raise ValueError("selected CUDA device is not visible")
    selected = torch.device(device)
    torch.cuda.set_device(selected)
    properties = torch.cuda.get_device_properties(selected)
    property_uuid = getattr(properties, "uuid", None)
    if isinstance(property_uuid, bytes):
        gpu_uuid = property_uuid.decode("ascii")
    else:
        gpu_uuid = str(property_uuid) if property_uuid is not None else ""
    if gpu_uuid.startswith("GPU-"):
        gpu_uuid = gpu_uuid[4:]
    if not gpu_uuid:
        raise RuntimeError("PyTorch did not expose the selected GPU UUID")
    try:
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("nvidia-smi runtime identity query failed") from error
    nvidia_records = [
        [field.strip() for field in line.split(",", maxsplit=1)]
        for line in nvidia_smi.stdout.splitlines()
    ]
    matched = next(
        (
            record
            for record in nvidia_records
            if len(record) == 2 and record[0] == f"GPU-{gpu_uuid}"
        ),
        None,
    )
    if matched is None:
        raise RuntimeError("PyTorch and nvidia-smi GPU UUIDs differ")
    cudnn_version = (
        torch.backends.cudnn.version()
        if hasattr(torch.backends, "cudnn")
        else None
    )
    return {
        "platform_machine": platform.machine(),
        "gpu_name": str(properties.name),
        "gpu_uuid": gpu_uuid,
        "nvidia_smi_gpu_uuid": matched[0],
        "nvidia_driver_version": matched[1],
        "gpu_compute_capability": [int(properties.major), int(properties.minor)],
        "gpu_multiprocessor_count": int(properties.multi_processor_count),
        "visible_cuda_device_count": visible_count,
        "selected_device": str(selected),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_build_version": str(torch.version.cuda),
        "cudnn_version": cudnn_version,
        "transformers_version": importlib.metadata.version("transformers"),
    }


def _validate_runtime_cli_identity(
    args: argparse.Namespace,
    *,
    observed: Mapping[str, Any],
    attempt_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = (
        "host_alias",
        "host_hostname",
        "container_id",
        "container_image_digest",
        "device",
    )
    if any(
        not isinstance(getattr(args, field), str) or not getattr(args, field)
        for field in fields
    ):
        raise ValueError("runtime identity CLI values must be explicit non-empty strings")
    if CUDA_DEVICE_PATTERN.fullmatch(args.device) is None:
        raise ValueError("--device must be explicit cuda:N")
    canonical_values = {
        "host_alias": CANONICAL_PILOT_HOST_ALIAS,
        "host_hostname": CANONICAL_PILOT_HOST_HOSTNAME,
        "container_id": CANONICAL_PILOT_CONTAINER_ID,
        "container_image_digest": CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
        "device": CANONICAL_PILOT_DEVICE,
    }
    if any(getattr(args, field) != value for field, value in canonical_values.items()):
        raise ValueError(
            "v2.1 canonical attempt is frozen to the exact Hyper00 "
            "host/container/image/device identity"
        )
    if attempt_identity is not None:
        contract_values = {
            "host_alias": attempt_identity.get("canonical_host_alias"),
            "host_hostname": attempt_identity.get("canonical_host_hostname"),
            "container_id": attempt_identity.get("canonical_container_id"),
            "container_image_digest": attempt_identity.get(
                "canonical_container_image_digest"
            ),
            "device": attempt_identity.get("canonical_device"),
        }
        if (
            contract_values != canonical_values
            or attempt_identity.get("cross_host_attempt_allowed") is not False
        ):
            raise ValueError("authorized canonical runtime identity drifted")
    if CONTAINER_ID_PATTERN.fullmatch(args.container_id) is None:
        raise ValueError("--container-id must be one full lowercase Docker ID")
    if CONTAINER_DIGEST_PATTERN.fullmatch(args.container_image_digest) is None:
        raise ValueError("--container-image-digest must be sha256:<64 lowercase hex>")
    container_hostname = socket.gethostname()
    if not args.container_id.startswith(container_hostname):
        raise ValueError("live container hostname must prefix the full Docker ID")
    if observed.get("selected_device") != args.device:
        raise ValueError("live runtime selected device differs from CLI")
    if observed.get("visible_cuda_device_count") != 1:
        raise ValueError("v2.1 single-device pilot requires exactly one visible GPU")
    return {
        "declared_host": {
            "alias": args.host_alias,
            "hostname": args.host_hostname,
            "verification": "requires_independent_host_preflight",
        },
        "verified_container_runtime": {
            "container_id": args.container_id,
            "container_hostname": container_hostname,
            "verification": "live_hostname_prefix_of_full_container_id",
        },
        "declared_container_image": {
            "digest": args.container_image_digest,
            "verification": "requires_independent_host_preflight",
        },
        "selected_device": args.device,
        "live": dict(observed),
    }


def authorize_production_pilot(
    args: argparse.Namespace,
    *,
    processor_validator_loader: Callable[[], Callable[..., Mapping[str, Any]]] = (
        _load_processor_audit_validator
    ),
    git_validator: Callable[[str | Path], Mapping[str, Any]] = validate_clean_pushed_main,
    source_validator: Callable[..., Sequence[Mapping[str, str]]] = (
        validate_committed_source_blobs
    ),
    artifact_loader: Callable[..., ValidatedScreeningArtifact] = (
        load_validated_screening_artifact
    ),
    runtime_identity_loader: Callable[..., Mapping[str, Any]] = (
        _load_live_runtime_identity
    ),
) -> AuthorizedPilot:
    """Complete every CPU/data authorization before policy runtime import."""
    _validate_absolute_production_paths(args)
    repository_root = Path(args.repository_root).resolve()
    contract = RestorationV21PilotContract.load(
        args.contract,
        repository_root=repository_root,
    )
    attempt_identity = validate_canonical_attempt_identity(contract, args.output_dir)
    output_dir = Path(attempt_identity["canonical_persistent_output_dir"])
    if output_dir == repository_root or repository_root in output_dir.parents:
        raise ValueError("raw pilot output directory must be outside the Git worktree")
    git_identity = dict(git_validator(repository_root))
    git_commit = git_identity.get("commit")
    if not isinstance(git_commit, str) or GIT_SHA_PATTERN.fullmatch(git_commit) is None:
        raise ValueError("Git authorization did not return a full commit SHA")
    processor_preflight_path = Path(args.processor_preflight).resolve()
    if (
        not processor_preflight_path.is_file()
        or processor_preflight_path == repository_root
        or repository_root in processor_preflight_path.parents
    ):
        raise ValueError("processor preflight raw evidence must be an external file")
    processor_artifact_path = _canonical_repository_path(
        repository_root / CANONICAL_PROCESSOR_PREFLIGHT_ARTIFACT_PATH,
        repository_root=repository_root,
        relative_path=CANONICAL_PROCESSOR_PREFLIGHT_ARTIFACT_PATH,
        name="processor preflight artifact manifest",
    )
    processor_artifact = _strict_json_object(processor_artifact_path)
    processor_value = _strict_json_object(processor_preflight_path)
    processor_validator = processor_validator_loader()
    processor_audit = processor_validator(
        processor_value,
        repository_root=repository_root,
        current_git_commit=git_commit,
        mode="reuse",
        evidence_path=processor_preflight_path,
    )
    if not isinstance(processor_audit, Mapping):
        raise TypeError("independent processor audit validator must return a mapping")
    expected_processor_result_keys = {
        "status",
        "prompt_count",
        "prompt_records_sha256",
        "shape_records_sha256",
        "teacher_golden_records_sha256",
        "processor_classes_sha256",
        "evidence_git_commit",
        "current_git_commit",
        "validation_mode",
    }
    if (
        set(processor_audit) != expected_processor_result_keys
        or processor_audit.get("status")
        != contract.data["processor_preflight"]["required_success_status"]
        or processor_audit.get("prompt_count") != 90
        or processor_audit.get("validation_mode") != "reuse"
        or processor_audit.get("current_git_commit") != git_commit
        or not isinstance(processor_audit.get("evidence_git_commit"), str)
        or GIT_SHA_PATTERN.fullmatch(processor_audit["evidence_git_commit"]) is None
        or any(
            not isinstance(processor_audit.get(field), str)
            or SHA256_PATTERN.fullmatch(processor_audit[field]) is None
            for field in (
                "prompt_records_sha256",
                "shape_records_sha256",
                "teacher_golden_records_sha256",
                "processor_classes_sha256",
            )
        )
    ):
        raise PermissionError("independent processor audit did not authorize generation")

    scientific_path = _canonical_repository_path(
        args.scientific_config,
        repository_root=repository_root,
        relative_path=CANONICAL_SCIENTIFIC_CONFIG_PATH,
        name="parent scientific config",
    )
    selection_path = _canonical_repository_path(
        args.selection_manifest,
        repository_root=repository_root,
        relative_path=CANONICAL_SELECTION_MANIFEST_PATH,
        name="selection manifest",
    )
    ocr_path = _canonical_repository_path(
        args.ocr_backend_config,
        repository_root=repository_root,
        relative_path=CANONICAL_OCR_BACKEND_CONFIG_PATH,
        name="OCR backend config",
    )
    snapshot_record = contract.data["primary_policy"]["snapshot_manifest"]
    snapshot_path = _canonical_repository_path(
        repository_root / snapshot_record["path"],
        repository_root=repository_root,
        relative_path=snapshot_record["path"],
        name="policy snapshot manifest",
    )
    if sha256_file(snapshot_path) != snapshot_record["sha256"]:
        raise ValueError("policy snapshot manifest SHA256 drifted")
    selection_record = contract.data["data"]["selection_manifest"]
    if sha256_file(selection_path) != selection_record["sha256"]:
        raise ValueError("selection manifest SHA256 drifted")
    source_paths = (
        RUNNER_SOURCE_PATH,
        PROCESSOR_AUDIT_SOURCE_PATH,
        "code/scripts/audit_gui_owl_v2_1_processor.py",
        CANONICAL_PROCESSOR_PREFLIGHT_ARTIFACT_PATH,
        "code/causalcache/restoration_v2_1_contract.py",
        "code/causalcache/policy/gui_owl_v2_1.py",
        "code/causalcache/policy/gui_owl_v2_1_runtime.py",
        "code/causalcache/data/restoration_v2_screening.py",
        "code/causalcache/data/restoration_v2_1_processor_inputs.py",
        "code/causalcache/data/guiodyssey_restoration_v2.py",
        "code/causalcache/data/restoration_v2_selection.py",
        "code/causalcache/data/guiodyssey.py",
        "code/causalcache/data/guiodyssey_independent.py",
        "code/causalcache/low_fidelity_v2.py",
        "code/causalcache/restoration_v2_contract.py",
        "code/causalcache/restoration_v2_text_backend.py",
        "code/causalcache/schema.py",
        "code/causalcache/policy/gui_owl_v2.py",
        "code/causalcache/policy/gui_owl_v2_runtime.py",
        "code/causalcache/policy/gui_owl_v2_vision.py",
        "code/causalcache/restoration_v2_baselines.py",
        "code/causalcache/restoration_v2_1_pilot_artifact.py",
        "code/scripts/manage_restoration_v2_1_pilot_artifact.py",
        CANONICAL_CONFIG_PATH,
        CANONICAL_SCIENTIFIC_CONFIG_PATH,
        CANONICAL_SELECTION_MANIFEST_PATH,
        CANONICAL_OCR_BACKEND_CONFIG_PATH,
        snapshot_record["path"],
    )
    source_inventory = tuple(
        source_validator(
            repository_root=repository_root,
            git_commit=git_commit,
            paths=source_paths,
        )
    )
    artifact = artifact_loader(
        artifact_root=args.derived_artifact_root,
        backend_config_path=ocr_path,
        scientific_config_path=scientific_path,
        selection_manifest_path=selection_path,
        expected_artifact_tree_sha256=contract.data["data"]["derived_artifact"][
            "artifact_tree_sha256"
        ],
    )
    if (
        artifact.artifact_tree_sha256
        != contract.data["data"]["derived_artifact"]["artifact_tree_sha256"]
    ):
        raise ValueError("validated artifact tree differs from the v2.1 contract")
    states = select_exact_development_pilot_states(artifact, contract)
    if (
        contract.data["data"]["confirm_policy_output_allowed"] is not False
        or contract.data["data"]["label_train_policy_output_allowed"] is not False
        or contract.data["data"]["top_up_allowed"] is not False
        or contract.data["promotion"]["confirm_remains_locked_until_full_45_substrate_gate_passes"]
        is not True
    ):
        raise PermissionError("v2.1 confirm, label-train, or top-up lock drifted")
    _, scientific_config = _load_json_object_with_sha(scientific_path)
    primary_policy = scientific_config.get("primary_policy")
    visual = (
        primary_policy.get("visual_preprocessing")
        if isinstance(primary_policy, Mapping)
        else None
    )
    target_tokens = (
        visual.get("target_effective_tokens_per_image")
        if isinstance(visual, Mapping)
        else None
    )
    if type(target_tokens) is not int or target_tokens <= 0:
        raise ValueError("parent scientific visual token target is invalid")
    observed_runtime = runtime_identity_loader(device=args.device)
    runtime_identity = _validate_runtime_cli_identity(
        args,
        observed=observed_runtime,
        attempt_identity=attempt_identity,
    )
    canonical_inputs = {
        "contract": {"path": CANONICAL_CONFIG_PATH, "sha256": contract.source_sha256},
        "processor_preflight": {
            "external_path": str(processor_preflight_path),
            "sha256": sha256_file(processor_preflight_path),
            "size_bytes": processor_preflight_path.stat().st_size,
            "artifact_manifest_path": CANONICAL_PROCESSOR_PREFLIGHT_ARTIFACT_PATH,
            "artifact_manifest_sha256": sha256_file(processor_artifact_path),
            "artifact_manifest": processor_artifact,
        },
        "scientific_config": {
            "path": CANONICAL_SCIENTIFIC_CONFIG_PATH,
            "sha256": sha256_file(scientific_path),
        },
        "selection_manifest": dict(selection_record),
        "ocr_backend_config": {
            "path": CANONICAL_OCR_BACKEND_CONFIG_PATH,
            "sha256": sha256_file(ocr_path),
        },
        "snapshot_manifest": dict(snapshot_record),
    }
    return AuthorizedPilot(
        repository_root=repository_root,
        contract=contract,
        processor_audit=dict(processor_audit),
        git_identity=git_identity,
        source_inventory=source_inventory,
        artifact=artifact,
        states=states,
        snapshot_manifest_path=snapshot_path,
        target_effective_visual_tokens_per_image=target_tokens,
        runtime_identity=runtime_identity,
        canonical_inputs=canonical_inputs,
        attempt_identity=attempt_identity,
    )


def _load_json_object_with_sha(path: Path) -> tuple[str, dict[str, Any]]:
    return sha256_file(path), _strict_json_object(path)


def _scientific_execution_argv(execution_argv: Sequence[str]) -> list[str]:
    values = list(execution_argv)
    if values.count("--resume") > 1:
        raise ValueError("execution argv contains duplicate --resume flags")
    return [argument for argument in values if argument != "--resume"]


def build_production_run_contract(
    args: argparse.Namespace,
    authorized: AuthorizedPilot,
) -> dict[str, Any]:
    projections = [
        _pilot_projection(state, pilot_index)
        for pilot_index, state in enumerate(authorized.states)
    ]
    execution_argv = getattr(args, "execution_argv", None)
    if (
        isinstance(execution_argv, (str, bytes, bytearray, Mapping))
        or not isinstance(execution_argv, Sequence)
        or not execution_argv
        or any(not isinstance(argument, str) or not argument for argument in execution_argv)
    ):
        raise ValueError("production pilot must preserve its complete argv")
    scientific_argv = _scientific_execution_argv(execution_argv)
    primary_policy = authorized.contract.data["primary_policy"]
    snapshot_manifest = primary_policy.get("snapshot_manifest", {})
    snapshot_manifest_sha256 = (
        snapshot_manifest.get("sha256")
        if isinstance(snapshot_manifest, Mapping)
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_sha256": authorized.contract.source_sha256,
        "git_identity": dict(authorized.git_identity),
        "source_inventory": list(authorized.source_inventory),
        "processor_audit": dict(authorized.processor_audit),
        "canonical_inputs": dict(authorized.canonical_inputs),
        "artifact": {
            "artifact_tree_sha256": authorized.artifact.artifact_tree_sha256,
            "artifact_manifest_sha256": authorized.artifact.artifact_manifest_sha256,
            "screening_manifest_sha256": authorized.artifact.screening_manifest_sha256,
            "derived_repo": dict(
                authorized.contract.data["data"]["derived_artifact"]
            ),
        },
        "policy": {
            "repo": primary_policy["repo"],
            "revision": primary_policy["revision"],
            "model_dir": str(Path(args.model_dir).resolve()),
            "runtime_metadata_requirement": {
                "required_protocol_id": PROTOCOL_ID,
                "validated_after_claim_before_generation": True,
                "native_generation_metadata_persisted_per_state": True,
                "required_metadata_keys": sorted(RUNTIME_METADATA_KEYS),
                "model_dir": str(Path(args.model_dir).resolve()),
                "model_repo": primary_policy["repo"],
                "model_revision": primary_policy["revision"],
                "snapshot_manifest_sha256": snapshot_manifest_sha256,
                "device": args.device,
                "target_effective_visual_tokens_per_image": (
                    authorized.target_effective_visual_tokens_per_image
                ),
            },
        },
        "runtime_identity": dict(authorized.runtime_identity),
        "execution_argv": scientific_argv,
        "operational_argv_policy": {
            "resume_flag_excluded_from_scientific_run_identity": True,
            "initial_invocation_is_recorded_without_resume": True,
        },
        "seed_policy": {
            "decoding": "greedy_do_sample_false",
            "random_seed": None,
            "sampling_seed_not_applicable": True,
        },
        "output_dir": str(Path(args.output_dir).resolve()),
        "attempt_identity": dict(authorized.attempt_identity),
        "states": projections,
        "generation_plan": {
            "fidelity": "full_history_reference",
            "generation_calls_per_state": 1,
            "fixed_state_denominator": EXPECTED_STATE_COUNT,
            "automatic_retry": False,
            "top_up": False,
        },
        "negative_operation_counts": dict(NEGATIVE_OPERATION_COUNTS),
        "confirm_state": "LOCKED",
    }


def execute_production_pilot(
    args: argparse.Namespace,
    *,
    authorization_loader: Callable[[argparse.Namespace], AuthorizedPilot] = (
        authorize_production_pilot
    ),
    integrity_revalidator: Callable[[AuthorizedPilot], None] = (
        revalidate_authorized_integrity
    ),
    runtime_bindings_loader: Callable[[], RuntimeBindings] = _load_runtime_bindings,
    run_executor: Callable[..., dict[str, Any]] = run_interface_pilot,
) -> dict[str, Any]:
    """Durably claim the attempt before importing or constructing policy runtime."""
    authorized = authorization_loader(args)
    integrity_revalidator(authorized)
    run_contract = build_production_run_contract(args, authorized)
    gate = PilotGate.from_mapping(authorized.contract.data["pilot_gate"])
    try:
        terminal = claim_interface_pilot(
            states=authorized.states,
            gate=gate,
            run_contract=run_contract,
            output_dir=args.output_dir,
            resume=args.resume,
        )
    except Exception as error:
        partial_layout = _prepare_pilot_layout(
            states=authorized.states,
            run_contract=run_contract,
            output_dir=args.output_dir,
        )
        if (
            not args.resume
            and partial_layout.ledger_path is not None
            and partial_layout.ledger_path.is_file()
            and not partial_layout.aggregate_path.exists()
        ):
            return _persist_claimed_invalid(
                partial_layout,
                stage="durable_claim_bootstrap_failure",
                error=error,
            )
        raise
    if terminal is not None:
        return terminal
    layout = _prepare_pilot_layout(
        states=authorized.states,
        run_contract=run_contract,
        output_dir=args.output_dir,
    )
    try:
        integrity_revalidator(authorized)
        bindings = runtime_bindings_loader()
        runtime = bindings.runtime_class(
            model_dir=args.model_dir,
            expected_snapshot_manifest=authorized.snapshot_manifest_path,
            device=args.device,
            target_effective_visual_tokens_per_image=(
                authorized.target_effective_visual_tokens_per_image
            ),
        )
        runtime_metadata = getattr(runtime, "metadata", None)
        if (
            not isinstance(runtime_metadata, Mapping)
            or runtime_metadata.get("protocol_id") != PROTOCOL_ID
        ):
            raise RuntimeError("loaded runtime identity differs from the v2.1 protocol")
        runtime_record = _runtime_identity_record(
            run_contract=run_contract,
            run_contract_sha256=layout.contract_sha256,
            runtime_metadata=runtime_metadata,
        )
        _validate_runtime_identity_record(
            runtime_record,
            run_contract=run_contract,
            run_contract_sha256=layout.contract_sha256,
        )
        _write_json_exclusive(
            layout.root / RUNTIME_IDENTITY_FILENAME,
            runtime_record,
        )
    except Exception as error:
        return _persist_claimed_invalid(
            layout,
            stage="policy_runtime_initialization_after_durable_claim",
            error=error,
        )
    try:
        result = run_executor(
            artifact=authorized.artifact,
            states=authorized.states,
            runtime=runtime,
            parse_error_class=bindings.parse_error_class,
            image_decoder=_decode_rgb_image,
            gate=gate,
            run_contract=run_contract,
            output_dir=args.output_dir,
            resume=False,
            preclaimed=True,
            integrity_guard=lambda: integrity_revalidator(authorized),
        )
        if not isinstance(result, Mapping) or not layout.aggregate_path.is_file():
            raise RuntimeError("generation runner returned without terminal evidence")
        persisted = _strict_json_object(layout.aggregate_path)
        if dict(result) != persisted:
            raise ValueError("generation runner result differs from terminal evidence")
        return persisted
    except Exception as error:
        if layout.aggregate_path.exists():
            raise
        return _persist_claimed_invalid(
            layout,
            stage="generation_runner_failure_after_durable_claim",
            error=error,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--processor-preflight", required=True)
    parser.add_argument("--derived-artifact-root", required=True)
    parser.add_argument("--scientific-config", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--ocr-backend-config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--host-hostname", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(raw_argv)
    args.execution_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        *raw_argv,
    ]
    try:
        aggregate = execute_production_pilot(args)
    except Exception as error:
        aggregate = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "INVALID_BEFORE_V2_1_PILOT_RUN_DIRECTORY",
            "outcome": INVALID_OUTCOME,
            "invalid_failure": {
                "category": "OUT_OF_MEMORY" if _is_out_of_memory(error) else "CONTRACT_OR_RUNTIME",
                "exception_type": error.__class__.__name__,
                "message": str(error),
            },
        }
        print(json.dumps(aggregate, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps(aggregate, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if aggregate["outcome"] in {PASS_OUTCOME, NO_GO_OUTCOME} else 2


if __name__ == "__main__":
    raise SystemExit(main())
