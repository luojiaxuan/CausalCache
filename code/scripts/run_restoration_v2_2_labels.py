"""Run the frozen two-GPU v2.2 eager restoration-label attempt."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import re
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.data.restoration_v2_2_label_inputs import (
    build_v2_2_label_messages,
    label_prompt_image_inventory,
    make_v2_2_label_prompt_spec,
)
from causalcache.data.restoration_v2_screening import (
    ScreeningState,
    ValidatedScreeningArtifact,
    load_validated_screening_artifact,
)
from causalcache.restoration_v2_1_full_45_contract import (
    RestorationV21Full45Contract,
)
from causalcache.restoration_v2_2_eager_artifact import (
    WorkerSpec,
    canonical_json_bytes,
    expected_worker_specs,
    pretty_json_bytes,
    read_v22_evidence_archive,
    sha256_bytes,
    validate_worker_runtime_pair,
)
from causalcache.restoration_v2_2_label_contract import (
    CANONICAL_ATTEMPT_ID,
    EXPECTED_DEPLOYMENT_EDGES,
    EXPECTED_DISTANCE_ROWS,
    EXPECTED_FULL_EDGES,
    EXPECTED_KL_MEASUREMENTS,
    EXPECTED_PRIMARY_ORACLES,
    EXPECTED_STATE_COUNT,
    EXPECTED_TEACHER_FORWARDS,
    PROTOCOL_ID,
    RestorationV22LabelContract,
    RestorationLabelAttemptProfile,
    V1_ATTEMPT_PROFILE,
    label_attempt_profile_for_id,
    sha256_file,
)
from causalcache.restoration_v2_2_label_parent import (
    V22LabelParentState,
    extract_v22_label_parent_states,
)
from causalcache.restoration_v2_2_label_table import (
    deployment_conditional_edges,
    exact_permutation_average_attribution,
    full_conditional_edges,
    pair_interactions,
    primary_exact_subset_oracle,
    validate_complete_distance_table,
)
from causalcache.policy.gui_owl_v2_vision import verify_frozen_vision_runtime
from scripts.run_restoration_v2_1_full_45_substrate import (
    GPUFullVocabularyKLBackend,
    _decode_rgb_image,
    _full45_projection,
    select_exact_full45_states,
    validate_clean_pushed_main,
    validate_committed_source_blobs,
)
from scripts.run_restoration_v2_2_eager_substrate import (
    CANONICAL_IMAGE_DIGEST,
    CANONICAL_OCR_BACKEND_CONFIG_PATH,
    CANONICAL_SCIENTIFIC_CONFIG_PATH,
    CANONICAL_SELECTION_MANIFEST_PATH,
    CANONICAL_SNAPSHOT_MANIFEST_PATH,
    load_worker_runtime,
)


SCHEMA_VERSION = "1.0.0"
RUN_STATUS = "RESTORATION_V2_2_EAGER_LABEL_ATTEMPT"
GLOBAL_CLAIM_STATUS = "LABEL_ATTEMPT_CLAIMED_BEFORE_RUNTIME_IMPORT"
WORKER_CLAIM_STATUS = "LABEL_WORKER_CLAIMED_BEFORE_RUNTIME_IMPORT"
STATE_OUTCOME = "VALID_RESTORATION_V2_2_EAGER_LABEL_STATE"
WORKER_PASS_OUTCOME = "PASS_RESTORATION_V2_2_EAGER_LABEL_WORKER"
WORKER_INVALID_OUTCOME = "INVALID_RESTORATION_V2_2_EAGER_LABEL_WORKER"
GLOBAL_INVALID_STATUS = "LABEL_ATTEMPT_INVALID"
RUN_MANIFEST_FILENAME = "run_manifest.json"
AGGREGATE_FILENAME = "aggregate.json"
WORKER_DIRECTORY = "workers"
STATE_DIRECTORY = "states"
ATTEMPT_DIRECTORY = "attempts"
WORKER_RUNTIME_FILENAME = "runtime_identity.json"
WORKER_LEDGER_FILENAME = "worker_attempt_ledger.json"
WORKER_TERMINAL_FILENAME = "terminal.json"
PARENT_SOURCE_GIT_COMMIT = "8ae07519f14ac3635f292ee93a7b6d624507427e"
MAXIMUM_REPEAT_KL = 1e-4
NORMALIZATION_EPSILON = 1e-12


@dataclass(frozen=True)
class AuthorizedLabels:
    repository_root: Path
    contract: RestorationV22LabelContract
    git_identity: Mapping[str, Any]
    source_inventory: tuple[Mapping[str, str], ...]
    artifact: ValidatedScreeningArtifact
    states: tuple[ScreeningState, ...]
    parent_states: tuple[V22LabelParentState, ...]
    parent_evidence: Mapping[str, Any]
    canonical_inputs: Mapping[str, Any]
    snapshot_manifest_path: Path
    model_snapshot_preflight: Mapping[str, Any]


@dataclass(frozen=True)
class LabelLayout:
    root: Path
    global_ledger: Path
    run_contract: Mapping[str, Any]
    run_contract_sha256: str
    worker_sibling_ledgers: Mapping[str, Path]
    started_at_utc: str
    profile: RestorationLabelAttemptProfile = V1_ATTEMPT_PROFILE


@dataclass(frozen=True)
class LabelRuntime:
    runtime: Any
    distance_backend: GPUFullVocabularyKLBackend
    metadata: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(pretty_json_bytes(dict(value)))
        destination.flush()
        os.fsync(destination.fileno())


def _replace_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"durable evidence disappeared before update: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
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


def _strict_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required JSON evidence is missing: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _canonical_repo_path(
    supplied: str | Path,
    *,
    repository_root: Path,
    relative: str,
) -> Path:
    expected = (repository_root / relative).resolve()
    actual = Path(supplied).resolve()
    if actual != expected or not actual.is_file():
        raise ValueError(f"canonical repository input must be {relative}")
    return actual


def _external_evidence(path: Path, *, status: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"fresh external evidence is missing: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "validation_status": status,
    }


def _safe_snapshot_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("model snapshot path must be canonical relative POSIX")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed.as_posix() != value or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        raise ValueError("model snapshot path must be canonical relative POSIX")
    return value


def validate_model_snapshot_preclaim(
    *,
    model_dir: str | Path,
    snapshot_manifest: str | Path,
    profile: RestorationLabelAttemptProfile,
    snapshot_validator: Callable[..., Any] = verify_frozen_vision_runtime,
) -> dict[str, Any]:
    supplied = Path(model_dir)
    if not supplied.is_absolute():
        raise ValueError("model projection directory must be absolute")
    if supplied.is_symlink():
        raise ValueError("model projection directory must not be a symlink")
    try:
        resolved = supplied.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"model projection directory does not exist: {supplied}"
        ) from error
    if resolved != supplied or not resolved.is_dir():
        raise ValueError("model projection directory must be a real canonical directory")
    if profile.canonical_model_dir is not None and resolved != profile.canonical_model_dir:
        raise ValueError("model projection directory differs from the repair contract")

    manifest_path = Path(snapshot_manifest).resolve()
    manifest = _strict_json(manifest_path)
    if set(manifest) != {"repo", "revision", "files"}:
        raise ValueError("model snapshot manifest schema drifted")
    repo = manifest.get("repo")
    revision = manifest.get("revision")
    files = manifest.get("files")
    if (
        not isinstance(repo, str)
        or not repo
        or resolved.name != repo.rsplit("/", 1)[-1]
        or not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or not isinstance(files, list)
        or not files
    ):
        raise ValueError("model snapshot repo, revision, or projection basename drifted")

    local_snapshot = resolved / ".snapshot.json"
    if local_snapshot.is_symlink() or _strict_json(local_snapshot) != manifest:
        raise ValueError("model projection .snapshot.json differs from the manifest")
    seen: set[str] = set()
    total_bytes = 0
    for index, raw in enumerate(files):
        if not isinstance(raw, Mapping) or set(raw) != {"path", "size", "sha256"}:
            raise ValueError(f"model snapshot file record {index} drifted")
        relative = _safe_snapshot_relative_path(raw.get("path"))
        size = raw.get("size")
        digest = raw.get("sha256")
        if (
            relative in seen
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(f"model snapshot file record {index} is invalid")
        path = resolved.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"model snapshot file is missing or partial: {relative}")
        seen.add(relative)
        total_bytes += size

    identity = snapshot_validator(
        model_dir=resolved,
        expected_snapshot_manifest=manifest_path,
    )
    observed = {
        "model_dir": getattr(identity, "model_dir", None),
        "model_repo": getattr(identity, "model_repo", None),
        "model_revision": getattr(identity, "model_revision", None),
        "snapshot_manifest_sha256": getattr(
            identity,
            "snapshot_manifest_sha256",
            None,
        ),
        "verified_model_file_count": getattr(
            identity,
            "verified_model_file_count",
            None,
        ),
        "verified_model_total_bytes": getattr(
            identity,
            "verified_model_total_bytes",
            None,
        ),
    }
    expected = {
        "model_dir": str(resolved),
        "model_repo": repo,
        "model_revision": revision,
        "snapshot_manifest_sha256": sha256_file(manifest_path),
        "verified_model_file_count": len(files),
        "verified_model_total_bytes": total_bytes,
    }
    if observed != expected:
        raise ValueError("full model snapshot validator identity drifted")
    return {
        **expected,
        "validation_status": "VALIDATED_FULL_MODEL_SNAPSHOT_BEFORE_GLOBAL_CLAIM",
    }


def authorize_label_run(
    args: argparse.Namespace,
    *,
    git_validator: Callable[[str | Path], Mapping[str, Any]] = validate_clean_pushed_main,
    source_validator: Callable[..., Sequence[Mapping[str, str]]] = (
        validate_committed_source_blobs
    ),
    artifact_loader: Callable[..., ValidatedScreeningArtifact] = (
        load_validated_screening_artifact
    ),
) -> AuthorizedLabels:
    root = Path(args.repository_root).resolve()
    contract = RestorationV22LabelContract.load(args.contract, repository_root=root)
    if contract.profile is None:
        raise ValueError("restoration-label contract lacks an attempt profile")
    profile = contract.profile
    if args.output_dir is None:
        args.output_dir = profile.output_dir
    if args.global_ledger is None:
        args.global_ledger = profile.ledger_path
    snapshot = _canonical_repo_path(
        args.snapshot_manifest,
        repository_root=root,
        relative=CANONICAL_SNAPSHOT_MANIFEST_PATH,
    )
    model_snapshot_preflight = validate_model_snapshot_preclaim(
        model_dir=args.model_dir,
        snapshot_manifest=snapshot,
        profile=profile,
    )
    expected_hosts = {
        "hyper00": "node-radixark-16-0000",
        "hyper01": "node-radixark-16-0001",
    }
    if (
        Path(args.output_dir).resolve() != profile.output_dir
        or Path(args.global_ledger).resolve() != profile.ledger_path
        or args.container_image_digest != CANONICAL_IMAGE_DIGEST
        or args.host_alias not in expected_hosts
        or args.host_hostname != expected_hosts.get(args.host_alias)
        or re.fullmatch(r"[0-9a-f]{64}", args.container_id) is None
    ):
        raise ValueError("canonical label output, Hyper host, or container identity drifted")

    git_identity = dict(git_validator(root))
    commit = git_identity.get("commit")
    if not isinstance(commit, str):
        raise ValueError("label authorization lacks a committed source identity")
    source_paths = tuple(contract.data["formal_run_source_inventory_paths"])
    source_inventory = tuple(
        source_validator(repository_root=root, git_commit=commit, paths=source_paths)
    )

    scientific = _canonical_repo_path(
        args.scientific_config,
        repository_root=root,
        relative=CANONICAL_SCIENTIFIC_CONFIG_PATH,
    )
    selection = _canonical_repo_path(
        args.selection_manifest,
        repository_root=root,
        relative=CANONICAL_SELECTION_MANIFEST_PATH,
    )
    ocr = _canonical_repo_path(
        args.ocr_backend_config,
        repository_root=root,
        relative=CANONICAL_OCR_BACKEND_CONFIG_PATH,
    )
    artifact = artifact_loader(
        artifact_root=args.derived_artifact_root,
        backend_config_path=ocr,
        scientific_config_path=scientific,
        selection_manifest_path=selection,
        expected_artifact_tree_sha256=contract.data["data"]["derived_artifact"][
            "artifact_tree_sha256"
        ],
    )
    parent_contract = RestorationV21Full45Contract.load(
        root / "code/configs/causalcache_restoration_v2_1_full_45.json",
        repository_root=root,
    )
    states = select_exact_full45_states(artifact, parent_contract)

    parent_path = Path(args.parent_v22_evidence).resolve()
    parent_artifact = contract.data["authorization"]["v2_2_parent_artifact"]
    if (
        sha256_file(parent_path) != parent_artifact["required_raw_sha256"]
        or parent_path.stat().st_size != parent_artifact["required_raw_size_bytes"]
    ):
        raise PermissionError("fresh v2.2 parent archive bytes drifted")
    parent_evidence = read_v22_evidence_archive(
        parent_path,
        expected_source_git_commit=PARENT_SOURCE_GIT_COMMIT,
    )
    parent_states = extract_v22_label_parent_states(parent_evidence)
    for index, (state, parent) in enumerate(zip(states, parent_states, strict=True)):
        projection = _full45_projection(state, index)
        observed = {
            "index": parent.index,
            "role": parent.role,
            "trajectory_id": parent.trajectory_id,
            "decision_step_id": parent.decision_step_id,
            "state_id": parent.state_id,
            "candidate_event_step_ids": list(parent.candidate_event_step_ids),
        }
        if projection != observed:
            raise PermissionError(f"parent/derived state crosswalk drifted at index {index}")

    canonical_inputs = {
        "scientific_config": {
            "path": CANONICAL_SCIENTIFIC_CONFIG_PATH,
            "sha256": sha256_file(scientific),
        },
        "selection_manifest": {
            "path": CANONICAL_SELECTION_MANIFEST_PATH,
            "sha256": sha256_file(selection),
        },
        "ocr_backend_config": {
            "path": CANONICAL_OCR_BACKEND_CONFIG_PATH,
            "sha256": sha256_file(ocr),
        },
        "snapshot_manifest": {
            "path": CANONICAL_SNAPSHOT_MANIFEST_PATH,
            "sha256": sha256_file(snapshot),
        },
        "derived_artifact": dict(contract.data["data"]["derived_artifact"]),
        "model_snapshot_preclaim": dict(model_snapshot_preflight),
    }
    return AuthorizedLabels(
        repository_root=root,
        contract=contract,
        git_identity=git_identity,
        source_inventory=source_inventory,
        artifact=artifact,
        states=states,
        parent_states=parent_states,
        parent_evidence=_external_evidence(
            parent_path,
            status="VALIDATED_IMMUTABLE_V2_2_EAGER_PARENT_ARCHIVE",
        ),
        canonical_inputs=canonical_inputs,
        snapshot_manifest_path=snapshot,
        model_snapshot_preflight=model_snapshot_preflight,
    )


def build_run_contract(
    args: argparse.Namespace,
    authorized: AuthorizedLabels,
) -> dict[str, Any]:
    if authorized.contract.profile is None:
        raise ValueError("restoration-label contract lacks an attempt profile")
    profile = authorized.contract.profile
    parent_states = [
        {
            "index": parent.index,
            "state_id": parent.state_id,
            "member_name": parent.member_name,
            "member_sha256": parent.member_sha256,
            "canonical_action_sha256": parent.canonical_action_sha256,
        }
        for parent in authorized.parent_states
    ]
    projections = [
        _full45_projection(state, index)
        for index, state in enumerate(authorized.states)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_source": {
            "path": profile.config_path,
            "sha256": profile.frozen_config_sha256,
        },
        "git_identity": dict(authorized.git_identity),
        "source_inventory": [dict(record) for record in authorized.source_inventory],
        "parent_evidence": dict(authorized.parent_evidence),
        "canonical_inputs": dict(authorized.canonical_inputs),
        "states": projections,
        "parent_states": parent_states,
        "runtime_requirements": dict(authorized.contract.data["runtime"]),
        "worker_topology": dict(authorized.contract.data["worker_topology"]),
        "enumeration": dict(authorized.contract.data["enumeration"]),
        "reduction": dict(authorized.contract.data["reduction"]),
        "operation_schedule": dict(authorized.contract.data["operation_schedule"]),
        "prohibited_work": dict(authorized.contract.data["prohibited_work"]),
        "attempt_identity": {
            "attempt_id": profile.attempt_id,
            "attempt_revision": profile.attempt_revision,
            "supersedes_attempt_id": profile.supersedes_attempt_id,
            "pass_outcome": profile.pass_outcome,
            "invalid_outcome": profile.invalid_outcome,
            "aggregate_status": profile.aggregate_status,
            "output_dir": str(Path(args.output_dir).resolve()),
            "global_ledger": str(Path(args.global_ledger).resolve()),
            "raw_archive": str(profile.archive_path),
            "hf_repo": profile.hf_repo,
            "hf_tag": profile.hf_tag,
            "hf_path": profile.hf_path,
            "host_alias": args.host_alias,
            "host_hostname": args.host_hostname,
            "container_id": args.container_id,
            "container_image_digest": args.container_image_digest,
        },
        "execution_argv": list(args.execution_argv),
    }


def _worker_sibling_paths(global_ledger: Path) -> dict[str, Path]:
    return {
        spec.worker_id: global_ledger.with_name(
            f"{global_ledger.name[:-5]}.{spec.worker_id}.json"
        )
        for spec in expected_worker_specs()
    }


def _profile_from_run_contract(
    run_contract: Mapping[str, Any],
) -> RestorationLabelAttemptProfile:
    attempt = run_contract.get("attempt_identity")
    if not isinstance(attempt, Mapping) or not isinstance(
        attempt.get("attempt_id"),
        str,
    ):
        raise ValueError("label run contract lacks an attempt identity")
    profile = label_attempt_profile_for_id(attempt["attempt_id"])
    strict_keys = {
        "attempt_revision": profile.attempt_revision,
        "supersedes_attempt_id": profile.supersedes_attempt_id,
        "pass_outcome": profile.pass_outcome,
        "invalid_outcome": profile.invalid_outcome,
        "aggregate_status": profile.aggregate_status,
    }
    present = set(attempt).intersection(strict_keys)
    if present and any(attempt.get(key) != value for key, value in strict_keys.items()):
        raise ValueError("label run contract attempt profile drifted")
    return profile


def _worker_ledger(
    *,
    layout: LabelLayout,
    spec: WorkerSpec,
    status: str,
    attempted: Sequence[int],
    completed: Sequence[int],
    claimed_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "attempt_id": layout.profile.attempt_id,
        "attempt_revision": layout.profile.attempt_revision,
        "status": status,
        "run_contract_sha256": layout.run_contract_sha256,
        "worker": spec.to_dict(),
        "claimed_at_utc": claimed_at_utc,
        "attempted_state_indices": list(attempted),
        "completed_state_indices": list(completed),
        "retry_count": 0,
        "top_up_count": 0,
    }


def _worker_high_water(layout: LabelLayout, spec: WorkerSpec) -> dict[str, Any]:
    worker_root = layout.root / WORKER_DIRECTORY / spec.worker_id
    expected = set(spec.state_indices)
    attempted: set[int] = set()
    completed: set[int] = set()
    evidence_errors: list[str] = []
    evidence: dict[str, Any] = {}

    record_paths = {
        "sibling_ledger": layout.worker_sibling_ledgers[spec.worker_id],
        "root_ledger": worker_root / WORKER_LEDGER_FILENAME,
        "terminal": worker_root / WORKER_TERMINAL_FILENAME,
    }
    for label, path in record_paths.items():
        if not path.exists():
            evidence[label] = {"present": False, "sha256": None}
            continue
        try:
            record = _strict_json(path)
            evidence[label] = {
                "present": True,
                "sha256": sha256_file(path),
                "status": record.get("status"),
                "outcome": record.get("outcome"),
                "failure": record.get("failure") if label == "terminal" else None,
            }
            for key, destination in (
                ("attempted_state_indices", attempted),
                ("completed_state_indices", completed),
            ):
                values = record.get(key, [])
                if (
                    not isinstance(values, list)
                    or any(type(value) is not int for value in values)
                    or any(value not in expected for value in values)
                ):
                    evidence_errors.append(f"{label}.{key} is invalid")
                    continue
                destination.update(values)
        except Exception as error:
            evidence[label] = {"present": True, "sha256": None}
            evidence_errors.append(
                f"{label} could not be read: {error.__class__.__name__}: {error}"
            )

    observed_files: dict[str, list[int]] = {}
    for label, directory in (
        ("attempt_markers", worker_root / ATTEMPT_DIRECTORY),
        ("state_records", worker_root / STATE_DIRECTORY),
    ):
        indices: list[int] = []
        try:
            for path in directory.glob("*.json"):
                if not path.is_file() or path.is_symlink() or not path.stem.isdigit():
                    continue
                index = int(path.stem)
                if index not in expected:
                    evidence_errors.append(
                        f"{label} contains out-of-shard index {index}"
                    )
                    continue
                indices.append(index)
        except Exception as error:
            evidence_errors.append(
                f"{label} could not be inventoried: {error.__class__.__name__}: {error}"
            )
        observed_files[label] = sorted(set(indices))
    attempted.update(observed_files["attempt_markers"])
    completed.update(observed_files["state_records"])
    attempted.update(completed)

    return {
        "expected_state_indices": list(spec.state_indices),
        "attempted_state_indices": sorted(attempted),
        "completed_state_indices": sorted(completed),
        "observed_attempt_marker_indices": observed_files["attempt_markers"],
        "observed_state_record_indices": observed_files["state_records"],
        "evidence": evidence,
        "evidence_errors": evidence_errors,
    }


def seal_invalid_attempt(
    layout: LabelLayout,
    *,
    stage: str,
    phase: str,
    error: BaseException,
    process_failures: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    high_water = {
        spec.worker_id: _worker_high_water(layout, spec)
        for spec in expected_worker_specs()
    }
    attempted = sorted(
        {
            index
            for worker in high_water.values()
            for index in worker["attempted_state_indices"]
        }
    )
    completed = sorted(
        {
            index
            for worker in high_water.values()
            for index in worker["completed_state_indices"]
        }
    )
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": GLOBAL_INVALID_STATUS,
        "attempt_id": layout.profile.attempt_id,
        "attempt_revision": layout.profile.attempt_revision,
        "run_contract_sha256": layout.run_contract_sha256,
        "worker_sibling_ledgers": {
            key: str(value) for key, value in layout.worker_sibling_ledgers.items()
        },
        "invalid_failure": {
            "stage": stage,
            "phase": phase,
            "exception_type": error.__class__.__name__,
            "message": str(error),
            "worker_process_failures": [dict(value) for value in process_failures],
        },
        "attempted_state_indices": attempted,
        "completed_state_indices": completed,
        "attempted_state_count": len(attempted),
        "completed_state_count": len(completed),
        "worker_high_water": high_water,
        "outcome": layout.profile.invalid_outcome,
        "retry_count": 0,
        "top_up_count": 0,
        "retry_allowed": False,
        "resume_allowed": False,
        "claimed_at_utc": layout.started_at_utc,
        "ended_at_utc": _utc_now(),
    }
    _replace_json_durable(layout.global_ledger, terminal)
    return terminal


def claim_attempt(
    *,
    run_contract: Mapping[str, Any],
    output_dir: str | Path,
    global_ledger: str | Path,
) -> LabelLayout:
    root = Path(output_dir).resolve()
    ledger_path = Path(global_ledger).resolve()
    profile = _profile_from_run_contract(run_contract)
    siblings = _worker_sibling_paths(ledger_path)
    forbidden = (root, ledger_path, *siblings.values())
    if any(path.exists() for path in forbidden):
        raise FileExistsError("restoration-label attempt already exists; retry is forbidden")
    run_contract_sha256 = sha256_bytes(canonical_json_bytes(run_contract))
    started = _utc_now()
    global_record = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": GLOBAL_CLAIM_STATUS,
        "attempt_id": profile.attempt_id,
        "attempt_revision": profile.attempt_revision,
        "run_contract_sha256": run_contract_sha256,
        "worker_sibling_ledgers": {
            worker_id: str(path) for worker_id, path in siblings.items()
        },
        "claimed_at_utc": started,
        "retry_count": 0,
        "top_up_count": 0,
    }
    _write_json_exclusive(ledger_path, global_record)
    layout = LabelLayout(
        root=root,
        global_ledger=ledger_path,
        run_contract=dict(run_contract),
        run_contract_sha256=run_contract_sha256,
        worker_sibling_ledgers=siblings,
        started_at_utc=started,
        profile=profile,
    )
    try:
        root.mkdir(parents=True, exist_ok=False)
        for spec in expected_worker_specs():
            worker_root = root / WORKER_DIRECTORY / spec.worker_id
            (worker_root / STATE_DIRECTORY).mkdir(parents=True)
            (worker_root / ATTEMPT_DIRECTORY).mkdir()
        _write_json_exclusive(
            root / RUN_MANIFEST_FILENAME,
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "attempt_id": profile.attempt_id,
                "attempt_revision": profile.attempt_revision,
                "status": RUN_STATUS,
                "run_contract_sha256": run_contract_sha256,
                "run_contract": dict(run_contract),
                "created_at_utc": started,
            },
        )
        for spec in expected_worker_specs():
            value = _worker_ledger(
                layout=layout,
                spec=spec,
                status="LABEL_WORKER_SIBLING_PREBOUND_BEFORE_SPAWN",
                attempted=(),
                completed=(),
                claimed_at_utc=started,
            )
            _write_json_exclusive(siblings[spec.worker_id], value)
    except BaseException as error:
        try:
            seal_invalid_attempt(
                layout,
                stage="claim_initialization",
                phase="root_manifest_and_sibling_prebind",
                error=error,
            )
        except Exception as seal_error:
            error.add_note(
                "failed to durably seal invalid label claim: "
                f"{seal_error.__class__.__name__}: {seal_error}"
            )
        raise
    return layout


def _coalition_mask(event_ids: Sequence[int], coalition: Sequence[int]) -> int:
    restored = set(coalition)
    return sum(
        1 << index
        for index, event_id in enumerate(event_ids)
        if event_id in restored
    )


def _normalized(value: float, baseline_distance: float) -> float | None:
    if baseline_distance <= NORMALIZATION_EPSILON:
        return None
    return value / baseline_distance


def _distance_row_payload(
    *,
    row: Any,
    event_ids: Sequence[int],
    prompt_inventory: Mapping[str, Any],
    teacher_metadata: Mapping[str, Any],
    distance_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "coalition_mask": _coalition_mask(event_ids, row.coalition),
        "coalition_event_step_ids": list(row.coalition),
        "coalition_slot_cost": len(row.coalition),
        "distance_kl": row.distance,
        "restoration_utility": row.utility,
        "prompt_inventory": dict(prompt_inventory),
        "teacher_metadata": dict(teacher_metadata),
        "distance_audit": dict(distance_audit),
    }


def run_label_state_once(
    *,
    artifact: ValidatedScreeningArtifact,
    state: ScreeningState,
    parent: V22LabelParentState,
    runtime: Any,
    distance_backend: GPUFullVocabularyKLBackend,
    run_contract_sha256: str,
    image_decoder: Callable[[bytes], Any] = _decode_rgb_image,
    attempt_id: str = CANONICAL_ATTEMPT_ID,
    attempt_revision: str = "v1_initial",
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = _utc_now()
    if (
        state.index != parent.index
        or state.state_id != parent.state_id
        or state.role != parent.role
        or state.candidate_event_step_ids != parent.candidate_event_step_ids
    ):
        raise ValueError("label state differs from its frozen parent action")
    event_ids = state.candidate_event_step_ids
    full = tuple(event_ids)
    full_spec = make_v2_2_label_prompt_spec(
        artifact,
        state,
        restored_event_step_ids=full,
        budget_event_capacity=2,
    )
    full_messages = build_v2_2_label_messages(
        artifact,
        full_spec,
        image_decoder=image_decoder,
    )
    full_inventory = label_prompt_image_inventory(full_spec, full_messages)

    reference_logits, reference_metadata = runtime.teacher_forced_distance_logits(
        (full_messages,),
        (parent.canonical_action,),
    )
    reference_log_probs = distance_backend.prepare_reference(reference_logits)
    del reference_logits

    repeat_logits, repeat_metadata = runtime.teacher_forced_distance_logits(
        (full_messages,),
        (parent.canonical_action,),
    )
    repeat_measurement = distance_backend.measure(reference_log_probs, repeat_logits)
    del repeat_logits
    repeat_kl = float(repeat_measurement.value)
    if not math.isfinite(repeat_kl) or repeat_kl < 0 or repeat_kl > MAXIMUM_REPEAT_KL:
        raise RuntimeError("reference repeat KL exceeds the frozen numerical-noise bound")

    raw_distances: dict[frozenset[int], float] = {frozenset(full): 0.0}
    measurements: dict[int, dict[str, Any]] = {
        (1 << len(event_ids)) - 1: {
            "prompt_inventory": full_inventory,
            "teacher_metadata": dict(reference_metadata),
            "distance_audit": {
                "reference_identity_distance": True,
                "full_tensor_host_transfers": 0,
            },
        }
    }
    for mask in range(1 << len(event_ids)):
        coalition = tuple(
            event_id
            for bit, event_id in enumerate(event_ids)
            if mask & (1 << bit)
        )
        if coalition == full:
            continue
        spec = make_v2_2_label_prompt_spec(
            artifact,
            state,
            restored_event_step_ids=coalition,
            budget_event_capacity=2,
        )
        messages = build_v2_2_label_messages(
            artifact,
            spec,
            image_decoder=image_decoder,
        )
        inventory = label_prompt_image_inventory(spec, messages)
        candidate_logits, metadata = runtime.teacher_forced_distance_logits(
            (messages,),
            (parent.canonical_action,),
        )
        measurement = distance_backend.measure(reference_log_probs, candidate_logits)
        del candidate_logits
        value = float(measurement.value)
        if not math.isfinite(value) or value < 0:
            raise RuntimeError("coalition distance is non-finite or negative")
        raw_distances[frozenset(coalition)] = value
        measurements[mask] = {
            "prompt_inventory": inventory,
            "teacher_metadata": dict(metadata),
            "distance_audit": dict(measurement.audit),
        }
    del reference_log_probs

    table = validate_complete_distance_table(event_ids, raw_distances)
    baseline_distance = table.distance(())
    distance_rows = []
    for row in table.rows:
        mask = _coalition_mask(event_ids, row.coalition)
        distance_rows.append(
            _distance_row_payload(
                row=row,
                event_ids=event_ids,
                **measurements[mask],
            )
        )

    deployment_edges = []
    for edge in deployment_conditional_edges(table):
        value = asdict(edge)
        value["base_coalition"] = list(edge.base_coalition)
        value["restored_coalition"] = list(edge.restored_coalition)
        value["normalized_marginal_gain"] = _normalized(
            edge.marginal_gain,
            baseline_distance,
        )
        deployment_edges.append(value)
    full_edges = []
    for edge in full_conditional_edges(table):
        value = asdict(edge)
        value["base_coalition"] = list(edge.base_coalition)
        value["restored_coalition"] = list(edge.restored_coalition)
        full_edges.append(value)
    interactions = []
    for interaction in pair_interactions(table):
        value = asdict(interaction)
        value["conditioning_coalition"] = list(
            interaction.conditioning_coalition
        )
        interactions.append(value)
    attribution = []
    for item in exact_permutation_average_attribution(table):
        value = asdict(item)
        value["marginal_samples"] = list(item.marginal_samples)
        value["normalized_mean_marginal_gain"] = _normalized(
            item.mean_marginal_gain,
            baseline_distance,
        )
        attribution.append(value)
    oracle = primary_exact_subset_oracle(table)
    oracle_payload = asdict(oracle)
    oracle_payload["coalition"] = list(oracle.coalition)
    oracle_payload["normalized_recovery"] = _normalized(
        oracle.utility,
        baseline_distance,
    )

    expected_rows = 1 << len(event_ids)
    expected_deployment_edges = {2: 4, 3: 9, 4: 16}[len(event_ids)]
    expected_full_edges = len(event_ids) * (1 << (len(event_ids) - 1))
    operation_counts = {
        "generation_call_count": 0,
        "reference_teacher_forward_count": 1,
        "reference_repeat_teacher_forward_count": 1,
        "non_reference_coalition_teacher_forward_count": expected_rows - 1,
        "teacher_forward_call_count": expected_rows + 1,
        "teacher_forward_example_count": expected_rows + 1,
        "coalition_kl_measurement_count": expected_rows - 1,
        "reference_repeat_kl_measurement_count": 1,
        "kl_measurement_count": expected_rows,
        "raw_distance_row_count": len(distance_rows),
        "deployment_conditional_label_count": len(deployment_edges),
        "full_hypercube_edge_count": len(full_edges),
        "pair_interaction_count": len(interactions),
        "exact_permutation_attribution_count": len(attribution),
        "primary_exact_subset_oracle_count": 1,
        "retry_count": 0,
        "top_up_count": 0,
        "confirm_state_access_count": 0,
        "confirm_processor_prompt_count": 0,
        "confirm_decoder_input_count": 0,
        "confirm_generation_count": 0,
        "confirm_teacher_forward_count": 0,
        "expert_action_read_count": 0,
        "gate_training_example_count": 0,
        "gate_model_forward_count": 0,
        "gate_selection_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
    }
    if (
        len(distance_rows) != expected_rows
        or len(deployment_edges) != expected_deployment_edges
        or len(full_edges) != expected_full_edges
    ):
        raise RuntimeError("state label inventory differs from the frozen denominator")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "attempt_id": attempt_id,
        "attempt_revision": attempt_revision,
        "run_contract_sha256": run_contract_sha256,
        "status": STATE_OUTCOME,
        "state": _full45_projection(state, state.index),
        "parent_reference": {
            "member_name": parent.member_name,
            "member_sha256": parent.member_sha256,
            "canonical_action": parent.canonical_action.arguments(),
            "canonical_action_sha256": parent.canonical_action_sha256,
            "reference_mode": "parent_canonical_action_frozen_fresh_teacher_recompute",
            "new_reference_generation_count": 0,
        },
        "reference_teacher_metadata": dict(reference_metadata),
        "reference_repeat_teacher_metadata": dict(repeat_metadata),
        "reference_repeat_kl": repeat_kl,
        "reference_repeat_distance_audit": dict(repeat_measurement.audit),
        "baseline_summary_only_distance": baseline_distance,
        "distance_rows": distance_rows,
        "deployment_conditional_edges": deployment_edges,
        "full_hypercube_edges": full_edges,
        "pair_interactions": interactions,
        "exact_permutation_attribution": attribution,
        "primary_exact_subset_oracle": oracle_payload,
        "operation_counts": operation_counts,
        "started_at_utc": started_at,
        "ended_at_utc": _utc_now(),
        "duration_seconds": time.perf_counter() - started,
    }


def _runtime_barrier(layout: LabelLayout, *, timeout_seconds: float = 900.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    paths = {
        spec.worker_id: (
            layout.root
            / WORKER_DIRECTORY
            / spec.worker_id
            / WORKER_RUNTIME_FILENAME
        )
        for spec in expected_worker_specs()
    }
    while True:
        if all(path.is_file() for path in paths.values()):
            runtimes = {
                worker_id: _strict_json(path)["runtime_metadata"]
                for worker_id, path in paths.items()
            }
            validate_worker_runtime_pair(runtimes)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("two-worker restoration-label runtime barrier timed out")
        time.sleep(0.25)


def _runtime_for_worker(
    *,
    spec: WorkerSpec,
    authorized: AuthorizedLabels,
    args: argparse.Namespace,
) -> LabelRuntime:
    bindings = load_worker_runtime(
        spec=spec,
        model_dir=args.model_dir,
        snapshot_manifest=authorized.snapshot_manifest_path,
        container_image_digest=args.container_image_digest,
    )
    return LabelRuntime(
        runtime=bindings.runtime,
        distance_backend=bindings.distance_backend,
        metadata=dict(bindings.metadata),
    )


def run_worker(
    *,
    layout: LabelLayout,
    spec: WorkerSpec,
    authorized: AuthorizedLabels,
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    worker_root = layout.root / WORKER_DIRECTORY / spec.worker_id
    root_ledger = worker_root / WORKER_LEDGER_FILENAME
    sibling_ledger = layout.worker_sibling_ledgers[spec.worker_id]
    claimed = _utc_now()
    attempted: list[int] = []
    completed: list[int] = []
    initial = _worker_ledger(
        layout=layout,
        spec=spec,
        status=WORKER_CLAIM_STATUS,
        attempted=attempted,
        completed=completed,
        claimed_at_utc=claimed,
    )
    _write_json_exclusive(root_ledger, initial)
    _replace_json_durable(sibling_ledger, initial)
    try:
        runtime = _runtime_for_worker(spec=spec, authorized=authorized, args=args)
        _write_json_exclusive(
            worker_root / WORKER_RUNTIME_FILENAME,
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "attempt_id": layout.profile.attempt_id,
                "attempt_revision": layout.profile.attempt_revision,
                "run_contract_sha256": layout.run_contract_sha256,
                "worker": spec.to_dict(),
                "runtime_metadata": dict(runtime.metadata),
                "created_at_utc": _utc_now(),
            },
        )
        _runtime_barrier(layout)
        for index in spec.state_indices:
            attempted.append(index)
            running = _worker_ledger(
                layout=layout,
                spec=spec,
                status="LABEL_WORKER_RUNNING_NO_RETRY",
                attempted=attempted,
                completed=completed,
                claimed_at_utc=claimed,
            )
            _replace_json_durable(sibling_ledger, running)
            _write_json_exclusive(
                worker_root / ATTEMPT_DIRECTORY / f"{index:03d}.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol_id": PROTOCOL_ID,
                    "attempt_id": layout.profile.attempt_id,
                    "attempt_revision": layout.profile.attempt_revision,
                    "status": "LABEL_STATE_CLAIMED_NO_RETRY_OR_TOP_UP",
                    "run_contract_sha256": layout.run_contract_sha256,
                    "worker": spec.to_dict(),
                    "state_index": index,
                    "retry_count": 0,
                    "top_up_count": 0,
                    "claimed_at_utc": _utc_now(),
                },
            )
            _replace_json_durable(root_ledger, running)
            record = run_label_state_once(
                artifact=authorized.artifact,
                state=authorized.states[index],
                parent=authorized.parent_states[index],
                runtime=runtime.runtime,
                distance_backend=runtime.distance_backend,
                run_contract_sha256=layout.run_contract_sha256,
                attempt_id=layout.profile.attempt_id,
                attempt_revision=layout.profile.attempt_revision,
            )
            _write_json_exclusive(
                worker_root / STATE_DIRECTORY / f"{index:03d}.json",
                record,
            )
            completed.append(index)
            advanced = _worker_ledger(
                layout=layout,
                spec=spec,
                status="LABEL_WORKER_RUNNING_NO_RETRY",
                attempted=attempted,
                completed=completed,
                claimed_at_utc=claimed,
            )
            _replace_json_durable(sibling_ledger, advanced)
            _replace_json_durable(root_ledger, advanced)
        outcome = WORKER_PASS_OUTCOME
        failure = None
        final_status = "LABEL_WORKER_COMPLETED"
    except Exception as error:
        outcome = WORKER_INVALID_OUTCOME
        failure = {
            "exception_type": error.__class__.__name__,
            "message": str(error),
        }
        final_status = "LABEL_WORKER_INVALID"
    final_ledger = _worker_ledger(
        layout=layout,
        spec=spec,
        status=final_status,
        attempted=attempted,
        completed=completed,
        claimed_at_utc=claimed,
    )
    _replace_json_durable(sibling_ledger, final_ledger)
    _replace_json_durable(root_ledger, final_ledger)
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "attempt_id": layout.profile.attempt_id,
        "attempt_revision": layout.profile.attempt_revision,
        "run_contract_sha256": layout.run_contract_sha256,
        "worker": spec.to_dict(),
        "outcome": outcome,
        "attempted_state_indices": attempted,
        "completed_state_indices": completed,
        "failure": failure,
        "ended_at_utc": _utc_now(),
    }
    _write_json_exclusive(worker_root / WORKER_TERMINAL_FILENAME, terminal)
    return terminal


def _layout_from_existing(
    *,
    root: Path,
    global_ledger: Path,
) -> LabelLayout:
    manifest = _strict_json(root / RUN_MANIFEST_FILENAME)
    ledger = _strict_json(global_ledger)
    run_contract = manifest.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise ValueError("label run manifest lacks a run contract")
    profile = _profile_from_run_contract(run_contract)
    run_contract_sha256 = sha256_bytes(canonical_json_bytes(run_contract))
    if (
        manifest.get("run_contract_sha256") != run_contract_sha256
        or manifest.get("attempt_id") != profile.attempt_id
        or manifest.get("attempt_revision") != profile.attempt_revision
        or ledger.get("run_contract_sha256") != run_contract_sha256
        or ledger.get("protocol_id") != PROTOCOL_ID
        or ledger.get("attempt_id") != profile.attempt_id
        or ledger.get("attempt_revision") != profile.attempt_revision
        or ledger.get("status") != GLOBAL_CLAIM_STATUS
        or ledger.get("retry_count") != 0
        or ledger.get("top_up_count") != 0
    ):
        raise ValueError("label run manifest and global claim differ")
    sibling_values = ledger.get("worker_sibling_ledgers")
    if not isinstance(sibling_values, Mapping):
        raise ValueError("label global claim lacks worker sibling ledgers")
    siblings = {key: Path(value) for key, value in sibling_values.items()}
    return LabelLayout(
        root=root,
        global_ledger=global_ledger,
        run_contract=dict(run_contract),
        run_contract_sha256=run_contract_sha256,
        worker_sibling_ledgers=siblings,
        started_at_utc=str(ledger["claimed_at_utc"]),
        profile=profile,
    )


def _worker_process_entry(
    args_dict: Mapping[str, Any],
    worker_id: str,
) -> None:
    args = argparse.Namespace(**dict(args_dict))
    layout = _layout_from_existing(
        root=Path(args.output_dir).resolve(),
        global_ledger=Path(args.global_ledger).resolve(),
    )
    authorized = authorize_label_run(args)
    rebuilt = build_run_contract(args, authorized)
    if sha256_bytes(canonical_json_bytes(rebuilt)) != layout.run_contract_sha256:
        raise RuntimeError("worker rebuilt a different label run contract")
    specs = {spec.worker_id: spec for spec in expected_worker_specs()}
    terminal = run_worker(
        layout=layout,
        spec=specs[worker_id],
        authorized=authorized,
        args=args,
    )
    if terminal["outcome"] != WORKER_PASS_OUTCOME:
        raise RuntimeError(str(terminal["failure"]))


def _sum_operation_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for record in records:
        counts = record.get("operation_counts")
        if not isinstance(counts, Mapping):
            raise ValueError("label state lacks operation counts")
        for key, value in counts.items():
            if type(value) is not int or value < 0:
                raise ValueError("label operation counts must be non-negative integers")
            totals[key] += value
    return dict(sorted(totals.items()))


def aggregate_attempt(layout: LabelLayout) -> dict[str, Any]:
    terminals = []
    records = []
    for spec in expected_worker_specs():
        worker_root = layout.root / WORKER_DIRECTORY / spec.worker_id
        terminal = _strict_json(worker_root / WORKER_TERMINAL_FILENAME)
        terminals.append(terminal)
        for index in spec.state_indices:
            records.append(
                _strict_json(worker_root / STATE_DIRECTORY / f"{index:03d}.json")
            )
    records.sort(key=lambda record: int(record["state"]["index"]))
    if (
        any(terminal.get("outcome") != WORKER_PASS_OUTCOME for terminal in terminals)
        or len(records) != EXPECTED_STATE_COUNT
        or [record["state"]["index"] for record in records]
        != list(range(EXPECTED_STATE_COUNT))
        or any(record.get("status") != STATE_OUTCOME for record in records)
    ):
        raise RuntimeError("label worker terminals or fixed state inventory are invalid")
    counts = _sum_operation_counts(records)
    expected = {
        "generation_call_count": 0,
        "reference_teacher_forward_count": 45,
        "reference_repeat_teacher_forward_count": 45,
        "non_reference_coalition_teacher_forward_count": 375,
        "teacher_forward_call_count": EXPECTED_TEACHER_FORWARDS,
        "teacher_forward_example_count": EXPECTED_TEACHER_FORWARDS,
        "coalition_kl_measurement_count": 375,
        "reference_repeat_kl_measurement_count": 45,
        "kl_measurement_count": EXPECTED_KL_MEASUREMENTS,
        "raw_distance_row_count": EXPECTED_DISTANCE_ROWS,
        "deployment_conditional_label_count": EXPECTED_DEPLOYMENT_EDGES,
        "full_hypercube_edge_count": EXPECTED_FULL_EDGES,
        "pair_interaction_count": 465,
        "exact_permutation_attribution_count": 135,
        "primary_exact_subset_oracle_count": EXPECTED_PRIMARY_ORACLES,
        "retry_count": 0,
        "top_up_count": 0,
        "confirm_state_access_count": 0,
        "confirm_processor_prompt_count": 0,
        "confirm_decoder_input_count": 0,
        "confirm_generation_count": 0,
        "confirm_teacher_forward_count": 0,
        "expert_action_read_count": 0,
        "gate_training_example_count": 0,
        "gate_model_forward_count": 0,
        "gate_selection_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
    }
    if counts != expected:
        raise RuntimeError(f"aggregate label operation counts drifted: {counts}")
    repeat_values = [float(record["reference_repeat_kl"]) for record in records]
    if any(
        not math.isfinite(value) or value < 0 or value > MAXIMUM_REPEAT_KL
        for value in repeat_values
    ):
        raise RuntimeError("aggregate reference repeat KL is invalid")
    baseline_distances = [
        float(record["baseline_summary_only_distance"]) for record in records
    ]
    oracle_utilities = [
        float(record["primary_exact_subset_oracle"]["utility"])
        for record in records
    ]
    normalized = [
        value / baseline
        for value, baseline in zip(oracle_utilities, baseline_distances, strict=True)
        if baseline > NORMALIZATION_EPSILON
    ]
    interaction_values = [
        float(item["interaction"])
        for record in records
        for item in record["pair_interactions"]
    ]
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "attempt_id": layout.profile.attempt_id,
        "attempt_revision": layout.profile.attempt_revision,
        "status": layout.profile.aggregate_status,
        "outcome": layout.profile.pass_outcome,
        "run_contract_sha256": layout.run_contract_sha256,
        "fixed_state_denominator": EXPECTED_STATE_COUNT,
        "role_state_counts": dict(
            sorted(Counter(record["state"]["role"] for record in records).items())
        ),
        "operation_counts": counts,
        "reference_repeat_kl": {
            "maximum": max(repeat_values),
            "mean": math.fsum(repeat_values) / len(repeat_values),
            "threshold": MAXIMUM_REPEAT_KL,
        },
        "scientific_summary": {
            "positive_baseline_distance_state_count": sum(
                value > NORMALIZATION_EPSILON for value in baseline_distances
            ),
            "mean_summary_only_distance": math.fsum(baseline_distances)
            / len(baseline_distances),
            "mean_primary_exact_oracle_utility": math.fsum(oracle_utilities)
            / len(oracle_utilities),
            "mean_primary_exact_oracle_normalized_recovery": (
                math.fsum(normalized) / len(normalized) if normalized else None
            ),
            "positive_pair_interaction_count": sum(
                value > 0 for value in interaction_values
            ),
            "negative_pair_interaction_count": sum(
                value < 0 for value in interaction_values
            ),
            "zero_pair_interaction_count": sum(
                value == 0 for value in interaction_values
            ),
        },
        "confirm_role_used": False,
        "gate_training_performed": False,
        "matched_nll_evaluation_performed": False,
        "closed_loop_evaluation_performed": False,
        "started_at_utc": layout.started_at_utc,
        "ended_at_utc": _utc_now(),
    }
    _write_json_exclusive(layout.root / AGGREGATE_FILENAME, aggregate)
    global_terminal = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "LABEL_ATTEMPT_COMPLETED",
        "attempt_id": layout.profile.attempt_id,
        "attempt_revision": layout.profile.attempt_revision,
        "run_contract_sha256": layout.run_contract_sha256,
        "worker_sibling_ledgers": {
            key: str(value) for key, value in layout.worker_sibling_ledgers.items()
        },
        "attempted_state_count": EXPECTED_STATE_COUNT,
        "completed_state_count": EXPECTED_STATE_COUNT,
        "outcome": layout.profile.pass_outcome,
        "retry_count": 0,
        "top_up_count": 0,
        "claimed_at_utc": layout.started_at_utc,
        "ended_at_utc": aggregate["ended_at_utc"],
    }
    _replace_json_durable(layout.global_ledger, global_terminal)
    return aggregate


def execute_claimed_attempt(
    layout: LabelLayout,
    args: argparse.Namespace,
    *,
    context: Any | None = None,
    aggregate_fn: Callable[[LabelLayout], dict[str, Any]] = aggregate_attempt,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn") if context is None else context
    args_dict = vars(args).copy()
    processes: list[Any] = []
    failures: list[dict[str, Any]] = []
    stage = "worker_process"
    phase = "construct"
    try:
        processes = [
            context.Process(
                target=_worker_process_entry,
                args=(args_dict, spec.worker_id),
                name=f"causalcache-label-{spec.worker_id}",
            )
            for spec in expected_worker_specs()
        ]
        phase = "start"
        for process in processes:
            process.start()
        phase = "join"
        for process in processes:
            process.join()
        failures = [
            {"name": process.name, "exitcode": process.exitcode}
            for process in processes
            if process.exitcode != 0
        ]
        if failures:
            phase = "exit"
            raise RuntimeError(
                f"restoration-label worker process failed: {failures}"
            )
        stage = "aggregate"
        phase = "validation_and_write"
        return aggregate_fn(layout)
    except BaseException as error:
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
                process.join()
            except Exception:
                continue
        if stage == "worker_process" and not failures:
            failures = [
                {
                    "name": getattr(process, "name", "unknown"),
                    "exitcode": getattr(process, "exitcode", None),
                }
                for process in processes
                if getattr(process, "exitcode", None) not in {None, 0}
            ]
        try:
            seal_invalid_attempt(
                layout,
                stage=stage,
                phase=phase,
                error=error,
                process_failures=failures,
            )
        except Exception as seal_error:
            error.add_note(
                "failed to durably seal invalid label attempt: "
                f"{seal_error.__class__.__name__}: {seal_error}"
            )
        raise


def execute_attempt(args: argparse.Namespace) -> dict[str, Any]:
    authorized = authorize_label_run(args)
    run_contract = build_run_contract(args, authorized)
    layout = claim_attempt(
        run_contract=run_contract,
        output_dir=args.output_dir,
        global_ledger=args.global_ledger,
    )
    return execute_claimed_attempt(layout, args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--scientific-config", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--ocr-backend-config", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--derived-artifact-root", type=Path, required=True)
    parser.add_argument("--parent-v22-evidence", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--global-ledger", type=Path)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--host-hostname", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-image-digest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(effective_argv)
    args.execution_argv = effective_argv
    result = execute_attempt(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
