"""Run the frozen two-worker restoration-v2.2 eager substrate attempt."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
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
from causalcache.restoration_v2_1_full_45_contract import (
    RestorationV21Full45Contract,
)
from causalcache.restoration_v2_2_eager_artifact import (
    AGGREGATE_FILENAME,
    ATTEMPT_DIRECTORY,
    FORBIDDEN_OPERATION_COUNTS,
    INVALID_OUTCOME,
    MEASUREMENT_KERNEL_PROTOCOL_ID,
    MEASUREMENT_KERNEL_SCHEMA_VERSION,
    NO_GO_OUTCOME,
    PASS_OUTCOME,
    PROTOCOL_ID,
    RUN_MANIFEST_FILENAME,
    SCHEMA_VERSION,
    STATE_DIRECTORY,
    STATE_OUTCOME_FAILED,
    STATE_OUTCOME_VALID,
    WORKER_DIRECTORY,
    WORKER_LEDGER_FILENAME,
    WORKER_OUTCOME_COMPLETE,
    WORKER_OUTCOME_INVALID,
    WORKER_RUNTIME_FILENAME,
    WORKER_TERMINAL_FILENAME,
    WorkerSpec,
    aggregate_v22_gate,
    canonical_json_bytes,
    expected_worker_specs,
    pretty_json_bytes,
    sha256_bytes,
    strict_json_object_bytes,
    validate_v22_run_contract_identity,
    validate_worker_runtime_metadata,
    validate_worker_runtime_pair,
    validate_worker_topology,
)
from causalcache.restoration_v2_2_eager_contract import (
    CANONICAL_ATTEMPT_ID,
    CANONICAL_CONFIG_PATH,
    FROZEN_RESTORATION_V2_2_EAGER_SHA256,
    RestorationV22EagerContract,
)
from scripts.run_restoration_v2_1_full_45_substrate import (
    GPUFullVocabularyKLBackend,
    _decode_rgb_image,
    _full45_projection,
    build_full45_messages,
    run_full45_state_once,
    select_exact_full45_states,
    validate_clean_pushed_main,
    validate_committed_source_blobs,
)


RUN_STATUS = "V2_2_EAGER_GLOBAL_ATTEMPT_CLAIMED"
GLOBAL_CLAIM_STATUS = "GLOBAL_ATTEMPT_CLAIMED_BEFORE_WORKER_SPAWN"
WORKER_CLAIM_STATUS = "WORKER_ATTEMPT_CLAIMED_BEFORE_RUNTIME_IMPORT"
RUNTIME_BARRIER_RELEASE_FILENAME = "runtime_barrier_release.json"
RUNTIME_BARRIER_ABORT_FILENAME = "runtime_barrier_abort.json"
CANONICAL_IMAGE_DIGEST = (
    "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
)
CANONICAL_SCIENTIFIC_CONFIG_PATH = "code/configs/causalcache_restoration_v2.json"
CANONICAL_SELECTION_MANIFEST_PATH = "data/manifests/restoration_v2_selection.json"
CANONICAL_OCR_BACKEND_CONFIG_PATH = "code/configs/restoration_v2_ocr_backend.json"
CANONICAL_SNAPSHOT_MANIFEST_PATH = "code/configs/gui_owl_1_5_8b_snapshot.json"


@dataclass(frozen=True)
class AuthorizedV22:
    repository_root: Path
    contract: RestorationV22EagerContract
    git_identity: Mapping[str, Any]
    source_inventory: tuple[Mapping[str, str], ...]
    artifact: ValidatedScreeningArtifact
    states: tuple[ScreeningState, ...]
    parent_authorization: Mapping[str, Mapping[str, Any]]
    canonical_inputs: Mapping[str, Any]
    snapshot_manifest_path: Path


@dataclass(frozen=True)
class V22Layout:
    root: Path
    global_ledger: Path
    manifest: Path
    aggregate: Path
    run_contract: Mapping[str, Any]
    run_contract_sha256: str
    projections: tuple[Mapping[str, Any], ...]
    started_at_utc: str
    worker_sibling_ledgers: Mapping[str, Path]


@dataclass(frozen=True)
class RuntimeBindings:
    runtime: Any
    parse_error_class: type[BaseException]
    distance_backend: Any
    metadata: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pretty_json_bytes(dict(value))
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def _publish_json_exclusive_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
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
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required evidence file is missing: {path}")
    return strict_json_object_bytes(path.read_bytes(), label=str(path))


def _strict_processor_parent_json(path: Path) -> dict[str, Any]:
    from causalcache.restoration_v2_1_processor_audit import (
        strict_json_object_bytes as strict_processor_json_object_bytes,
    )

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required processor evidence file is missing: {path}")
    return strict_processor_json_object_bytes(
        path.read_bytes(), label="external processor evidence"
    )


def _external_record(
    path: str | Path, *, validation_status: str, validation_outcome: str
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"fresh parent evidence is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "validation_status": validation_status,
        "validation_outcome": validation_outcome,
    }


def _canonical_repo_input(
    supplied: str | Path, *, repository_root: Path, relative: str
) -> Path:
    expected = (repository_root / relative).resolve()
    actual = Path(supplied).resolve()
    if actual != expected or not actual.is_file():
        raise ValueError(f"canonical repository input must be {relative}")
    return actual


def _repaired_spatial_eager_stable_count(
    summary: Mapping[str, Any],
) -> int | None:
    profiles = summary.get("profile_summaries")
    if not isinstance(profiles, list):
        return None
    for profile in profiles:
        if not isinstance(profile, Mapping):
            return None
        if profile.get("profile_id") != "bf16_eager_control":
            continue
        metrics = profile.get("metrics")
        if not isinstance(metrics, Mapping):
            return None
        count = metrics.get("exact_canonical_action_stable_count")
        if isinstance(count, bool) or not isinstance(count, int):
            return None
        return count
    return None


def _validate_parent_evidence(
    args: argparse.Namespace,
    *,
    repository_root: Path,
    current_git_commit: str,
) -> dict[str, Mapping[str, Any]]:
    from causalcache.restoration_v2_1_full_45_artifact import (
        validate_committed_full_45_artifact,
    )
    from causalcache.restoration_v2_1_pilot_artifact import (
        validate_committed_pilot_artifact,
    )
    from causalcache.restoration_v2_1_processor_audit import (
        validate_restoration_v2_1_processor_audit,
    )
    from causalcache.spatial_reference_audit_v1_artifact import (
        read_spatial_reference_audit_archive,
    )

    spatial_path = Path(args.spatial_reference_evidence).resolve()
    spatial_files = read_spatial_reference_audit_archive(spatial_path)
    if "summary.json" not in spatial_files:
        raise ValueError("fresh spatial-reference archive lacks repaired summary")
    spatial_manifest = json.loads(
        (repository_root / "data/results/spatial_reference_audit_v1/artifact.json").read_text()
    )
    spatial_summary = json.loads(spatial_files["summary.json"])
    expected_spatial_raw = spatial_manifest["raw_archive"]
    if (
        spatial_manifest.get("status") != "VERIFIED_SPATIAL_REFERENCE_AUDIT_V1_ARTIFACT"
        or spatial_manifest.get("protocol_id") != "spatial_reference_audit_v1"
        or sha256_file(spatial_path) != expected_spatial_raw.get("sha256")
        or spatial_path.stat().st_size != expected_spatial_raw.get("size_bytes")
        or spatial_summary.get("status")
        != "VALID_SPATIAL_REFERENCE_AUDIT_V1_VALIDATION_REPAIR_V1"
        or spatial_summary.get("decision")
        != "EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY"
        or spatial_summary.get("claim_boundary", {}).get(
            "parent_v2_1_outcome_unchanged"
        )
        != "NO_GO_V2_1_FULL_45_SUBSTRATE"
        or spatial_summary.get("claim_boundary", {}).get(
            "strict_cuda_determinism_claimed"
        )
        is not False
        or _repaired_spatial_eager_stable_count(spatial_summary) != 13
    ):
        raise PermissionError("fresh spatial-reference evidence conclusion drifted")
    spatial = _external_record(
        spatial_path,
        validation_status="VALIDATED_CANONICAL_SPATIAL_REFERENCE_RAW_ARCHIVE",
        validation_outcome="EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY",
    )
    full45_validation = validate_committed_full_45_artifact(
        repository_root=repository_root,
        current_git_commit=current_git_commit,
        evidence_path=args.v2_1_full_45_evidence,
    )
    if (
        full45_validation.get("status")
        != "VALID_RESTORATION_V2_1_FULL_45_ARTIFACT"
        or full45_validation.get("outcome") != "NO_GO_V2_1_FULL_45_SUBSTRATE"
        or full45_validation.get("archive_hash_verified") is not True
        or full45_validation.get("file_count") != 94
    ):
        raise PermissionError("fresh v2.1 full-45 evidence conclusion drifted")
    full45 = _external_record(
        args.v2_1_full_45_evidence,
        validation_status=str(full45_validation["status"]),
        validation_outcome=str(full45_validation["outcome"]),
    )
    pilot_validation = validate_committed_pilot_artifact(
        repository_root=repository_root,
        current_git_commit=current_git_commit,
        evidence_path=args.pilot_evidence,
    )
    if (
        pilot_validation.get("status")
        != "VALID_RESTORATION_V2_1_INTERFACE_PILOT_ARTIFACT"
        or pilot_validation.get("outcome") != "PASS_V2_1_INTERFACE_PILOT"
        or pilot_validation.get("archive_hash_verified") is not True
        or pilot_validation.get("file_count") != 34
    ):
        raise PermissionError("fresh v2.1 pilot evidence conclusion drifted")
    pilot = _external_record(
        args.pilot_evidence,
        validation_status=str(pilot_validation["status"]),
        validation_outcome=str(pilot_validation["outcome"]),
    )
    processor_path = Path(args.processor_evidence).resolve()
    processor_validation = validate_restoration_v2_1_processor_audit(
        _strict_processor_parent_json(processor_path),
        repository_root=repository_root,
        current_git_commit=current_git_commit,
        mode="reuse",
        evidence_path=processor_path,
    )
    committed_processor = json.loads(
        (
            repository_root
            / "data/results/restoration_v2_1_processor_preflight/artifact.json"
        ).read_text()
    )
    expected_processor_raw = committed_processor["raw_evidence"]
    if (
        processor_validation.get("status")
        != "PASSED_RESTORATION_V2_1_90_PROMPT_PROCESSOR_PREFLIGHT"
        or processor_validation.get("prompt_count") != 90
        or processor_validation.get("validation_mode") != "reuse"
        or sha256_file(processor_path) != expected_processor_raw.get("sha256")
        or processor_path.stat().st_size != expected_processor_raw.get("size_bytes")
    ):
        raise PermissionError("fresh v2.1 processor evidence conclusion drifted")
    processor = _external_record(
        processor_path,
        validation_status=str(processor_validation["status"]),
        validation_outcome="PASSED_90_PROMPT_PROCESSOR_PREFLIGHT",
    )
    return {
        "spatial_reference_evidence": spatial,
        "v2_1_full_45_evidence": full45,
        "pilot_evidence": pilot,
        "processor_evidence": processor,
    }


def authorize_production_v22(
    args: argparse.Namespace,
    *,
    git_validator: Callable[[str | Path], Mapping[str, Any]] = validate_clean_pushed_main,
    source_validator: Callable[..., Sequence[Mapping[str, str]]] = (
        validate_committed_source_blobs
    ),
    artifact_loader: Callable[..., ValidatedScreeningArtifact] = (
        load_validated_screening_artifact
    ),
    parent_validator: Callable[..., Mapping[str, Mapping[str, Any]]] = (
        _validate_parent_evidence
    ),
) -> AuthorizedV22:
    root = Path(args.repository_root).resolve()
    contract = RestorationV22EagerContract.load(args.contract, repository_root=root)
    execution = contract.data["execution"]
    expected_hosts = {
        "hyper00": "node-radixark-16-0000",
        "hyper01": "node-radixark-16-0001",
    }
    if (
        Path(args.output_dir).resolve()
        != Path(execution["canonical_persistent_output_dir"])
        or Path(args.global_ledger).resolve()
        != Path(execution["canonical_global_attempt_ledger"])
        or args.container_image_digest != CANONICAL_IMAGE_DIGEST
        or args.host_alias not in expected_hosts
        or args.host_hostname != expected_hosts.get(args.host_alias)
        or re.fullmatch(r"[0-9a-f]{64}", args.container_id) is None
        or not Path(args.model_dir).is_absolute()
        or execution.get("required_host_class") != "Hyper_H200"
    ):
        raise ValueError("v2.2 canonical output, Hyper host, or container identity drifted")
    git = dict(git_validator(root))
    commit = git.get("commit")
    if not isinstance(commit, str):
        raise ValueError("v2.2 authorization lacks a committed source identity")
    source_paths = tuple(
        str(path)
        for path in contract.data["scientific_inheritance"][
            "formal_run_source_inventory_paths"
        ]
    )
    source_inventory = tuple(
        source_validator(repository_root=root, git_commit=commit, paths=source_paths)
    )
    scientific = _canonical_repo_input(
        args.scientific_config,
        repository_root=root,
        relative=CANONICAL_SCIENTIFIC_CONFIG_PATH,
    )
    selection = _canonical_repo_input(
        args.selection_manifest,
        repository_root=root,
        relative=CANONICAL_SELECTION_MANIFEST_PATH,
    )
    ocr = _canonical_repo_input(
        args.ocr_backend_config,
        repository_root=root,
        relative=CANONICAL_OCR_BACKEND_CONFIG_PATH,
    )
    snapshot = _canonical_repo_input(
        args.snapshot_manifest,
        repository_root=root,
        relative=CANONICAL_SNAPSHOT_MANIFEST_PATH,
    )
    parent_contract = RestorationV21Full45Contract.load(
        root / "code/configs/causalcache_restoration_v2_1_full_45.json",
        repository_root=root,
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
    states = select_exact_full45_states(artifact, parent_contract)
    parents = dict(
        parent_validator(
            args,
            repository_root=root,
            current_git_commit=commit,
        )
    )
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
    }
    return AuthorizedV22(
        repository_root=root,
        contract=contract,
        git_identity=git,
        source_inventory=source_inventory,
        artifact=artifact,
        states=states,
        parent_authorization=parents,
        canonical_inputs=canonical_inputs,
        snapshot_manifest_path=snapshot,
    )


def build_production_run_contract(
    args: argparse.Namespace, authorized: AuthorizedV22
) -> dict[str, Any]:
    topology = authorized.contract.data["execution"]["worker_topology"]
    validate_worker_topology(topology)
    projections = [
        _full45_projection(state, index)
        for index, state in enumerate(authorized.states)
    ]
    gate_source = authorized.contract.data["substrate_gate"]
    gate = {
        "minimum_screening_states": gate_source["minimum_screening_states"],
        "minimum_parse_coverage": gate_source["minimum_parse_coverage"],
        "minimum_finite_logit_coverage": gate_source[
            "minimum_finite_logit_coverage"
        ],
        "minimum_repeat_canonical_action_agreement": gate_source[
            "minimum_repeat_canonical_action_agreement"
        ],
        "minimum_memory_sensitive_states": gate_source[
            "minimum_memory_sensitive_states"
        ],
    }
    execution = authorized.contract.data["execution"]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_source": {
            "path": CANONICAL_CONFIG_PATH,
            "sha256": FROZEN_RESTORATION_V2_2_EAGER_SHA256,
        },
        "git_identity": dict(authorized.git_identity),
        "source_inventory": [dict(record) for record in authorized.source_inventory],
        "parent_authorization": {
            key: dict(value) for key, value in authorized.parent_authorization.items()
        },
        "canonical_inputs": dict(authorized.canonical_inputs),
        "runtime_requirements": {
            "gpu_name": "NVIDIA H200",
            "container_image_digest": CANONICAL_IMAGE_DIGEST,
            "python_version": "3.12.3",
            "torch_version": "2.11.0+cu130",
            "torch_cuda_version": "13.0",
            "cudnn_version": 91900,
            "transformers_version": "5.6.0",
            "nvidia_driver_version": "570.172.08",
            "runtime_profile_id": "causalcache_restoration_v2_2_eager_runtime",
            "attention_implementation": "eager",
        },
        "worker_topology": dict(topology),
        "states": projections,
        "gate": gate,
        "operation_policy": {
            "retry_count": 0,
            "top_up_count": 0,
            "v2_1_raw_state_reuse_count": 0,
            "same_state_repeats_must_remain_on_one_worker_device": True,
            "merge_only_after_both_worker_terminals": True,
            "cpu_recompute_gate_after_index_order_merge": True,
            "resume_allowed": False,
            "interrupted_attempt_is_terminal_invalid": True,
            "resume_or_top_up_count": 0,
        },
        "attempt_identity": {
            "attempt_id": execution["attempt_id"],
            "output_dir": str(Path(args.output_dir).resolve()),
            "global_ledger": str(Path(args.global_ledger).resolve()),
            "host_alias": args.host_alias,
            "host_hostname": args.host_hostname,
            "container_id": args.container_id,
            "container_image_digest": args.container_image_digest,
        },
        "execution_argv": list(args.execution_argv),
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


def _worker_sibling_paths(global_ledger: Path) -> dict[str, Path]:
    return {
        spec.worker_id: global_ledger.with_name(
            f"{global_ledger.name[:-5]}.{spec.worker_id}.json"
        )
        for spec in expected_worker_specs()
    }


def claim_global_attempt(
    *,
    run_contract: Mapping[str, Any],
    output_dir: str | Path,
    global_ledger: str | Path,
) -> V22Layout:
    root = Path(output_dir).resolve()
    ledger_path = Path(global_ledger).resolve()
    if root.exists() or ledger_path.exists():
        raise FileExistsError("v2.2 is one fresh attempt; existing root/ledger forbids retry")
    sibling_paths = _worker_sibling_paths(ledger_path)
    if any(path.exists() for path in sibling_paths.values()):
        raise FileExistsError("existing worker sibling ledger forbids v2.2 retry")
    run_contract_sha256 = sha256_bytes(canonical_json_bytes(run_contract))
    started = _utc_now()
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": GLOBAL_CLAIM_STATUS,
        "attempt_id": run_contract["attempt_identity"]["attempt_id"],
        "run_contract_sha256": run_contract_sha256,
        "worker_topology": run_contract["worker_topology"],
        "claimed_at_utc": started,
        "spawn_started_at_utc": None,
        "completed_state_count": 0,
        "attempted_state_count": 0,
        "worker_high_water": _empty_worker_high_water(),
        "worker_sibling_ledgers": {
            worker_id: str(path) for worker_id, path in sibling_paths.items()
        },
        "retry_count": 0,
        "top_up_count": 0,
    }
    _write_json_exclusive(ledger_path, ledger)
    try:
        root.mkdir(parents=True, exist_ok=False)
        for spec in expected_worker_specs():
            worker_root = root / WORKER_DIRECTORY / spec.worker_id
            (worker_root / STATE_DIRECTORY).mkdir(parents=True)
            (worker_root / ATTEMPT_DIRECTORY).mkdir()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": RUN_STATUS,
            "run_contract_sha256": run_contract_sha256,
            "run_contract": dict(run_contract),
            "created_at_utc": started,
        }
        _write_json_exclusive(root / RUN_MANIFEST_FILENAME, manifest)
    except Exception:
        raise
    projections = tuple(dict(value) for value in run_contract["states"])
    layout = V22Layout(
        root=root,
        global_ledger=ledger_path,
        manifest=root / RUN_MANIFEST_FILENAME,
        aggregate=root / AGGREGATE_FILENAME,
        run_contract=dict(run_contract),
        run_contract_sha256=run_contract_sha256,
        projections=projections,
        started_at_utc=started,
        worker_sibling_ledgers=sibling_paths,
    )
    for spec in expected_worker_specs():
        sibling = _worker_ledger_record(
            layout=layout,
            spec=spec,
            status="WORKER_SIBLING_PREBOUND_BEFORE_SPAWN",
            claimed_at_utc=started,
            attempted=(),
            completed=(),
        )
        _write_json_exclusive(sibling_paths[spec.worker_id], sibling)
    return layout


def _worker_root(layout: V22Layout, spec: WorkerSpec) -> Path:
    return layout.root / WORKER_DIRECTORY / spec.worker_id


def _worker_ledger_record(
    *,
    layout: V22Layout,
    spec: WorkerSpec,
    status: str,
    claimed_at_utc: str,
    attempted: Sequence[int],
    completed: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
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
    *, layout: V22Layout, spec: WorkerSpec, root_ledger: Path, value: Mapping[str, Any]
) -> None:
    _replace_json_durable(layout.worker_sibling_ledgers[spec.worker_id], value)
    _replace_json_durable(root_ledger, value)
    if (
        layout.worker_sibling_ledgers[spec.worker_id].read_bytes()
        != root_ledger.read_bytes()
    ):
        raise RuntimeError("worker sibling and root ledger bytes differ after update")


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
    if metadata.get("device") != spec.device:
        raise RuntimeError("loaded eager runtime differs from worker device")
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


def _state_marker(layout: V22Layout, spec: WorkerSpec, index: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "STATE_ATTEMPT_CLAIMED_NO_RETRY_OR_TOP_UP",
        "run_contract_sha256": layout.run_contract_sha256,
        "worker": {
            "worker_id": spec.worker_id,
            "device": spec.device,
            "index_parity": spec.index_parity,
        },
        "state_index": index,
        "retry_count": 0,
        "top_up_count": 0,
        "claimed_at_utc": _utc_now(),
    }


def wrap_measurement_record(
    inner: Mapping[str, Any], *, spec: WorkerSpec
) -> dict[str, Any]:
    if inner.get("outcome") == "VALID_V2_1_FULL_45_SUBSTRATE_STATE":
        outcome = STATE_OUTCOME_VALID
    elif inner.get("outcome") == "FAILED_V2_1_FULL_45_SUBSTRATE_STATE":
        outcome = STATE_OUTCOME_FAILED
    else:
        raise ValueError("measurement kernel returned an unknown state outcome")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_contract_sha256": inner["run_contract_sha256"],
        "worker": {
            "worker_id": spec.worker_id,
            "device": spec.device,
            "index_parity": spec.index_parity,
        },
        "state": dict(inner["state"]),
        "outcome": outcome,
        "measurement_kernel": {
            "protocol_id": MEASUREMENT_KERNEL_PROTOCOL_ID,
            "schema_version": MEASUREMENT_KERNEL_SCHEMA_VERSION,
            "record": dict(inner),
        },
    }


def run_worker_shard(
    *,
    layout: V22Layout,
    spec: WorkerSpec,
    artifact: ValidatedScreeningArtifact,
    states: Sequence[ScreeningState],
    runtime_loader: Callable[[WorkerSpec], RuntimeBindings],
    state_kernel: Callable[..., Mapping[str, Any]] = run_full45_state_once,
    runtime_barrier: Callable[[V22Layout], None] | None = None,
) -> Mapping[str, Any]:
    worker_root = _worker_root(layout, spec)
    ledger_path = worker_root / WORKER_LEDGER_FILENAME
    terminal_path = worker_root / WORKER_TERMINAL_FILENAME
    claimed = _utc_now()
    attempted: list[int] = []
    completed: list[int] = []
    ledger = _worker_ledger_record(
        layout=layout,
        spec=spec,
        status=WORKER_CLAIM_STATUS,
        claimed_at_utc=claimed,
        attempted=attempted,
        completed=completed,
    )
    _write_json_exclusive(ledger_path, ledger)
    _replace_worker_ledgers(
        layout=layout, spec=spec, root_ledger=ledger_path, value=ledger
    )
    try:
        bindings = runtime_loader(spec)
        runtime_record = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "run_contract_sha256": layout.run_contract_sha256,
            "worker": spec.to_dict(),
            "runtime_metadata": dict(bindings.metadata),
            "created_at_utc": _utc_now(),
        }
        _publish_json_exclusive_atomic(
            worker_root / WORKER_RUNTIME_FILENAME, runtime_record
        )
        if runtime_barrier is not None:
            runtime_barrier(layout)
        for index in spec.state_indices:
            marker_path = worker_root / ATTEMPT_DIRECTORY / f"{index:03d}.json"
            state_path = worker_root / STATE_DIRECTORY / f"{index:03d}.json"
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
                layout.worker_sibling_ledgers[spec.worker_id], ledger
            )
            _write_json_exclusive(marker_path, _state_marker(layout, spec, index))
            _replace_json_durable(ledger_path, ledger)
            if (
                layout.worker_sibling_ledgers[spec.worker_id].read_bytes()
                != ledger_path.read_bytes()
            ):
                raise RuntimeError("worker ledgers differ before state generation")
            inner = state_kernel(
                artifact=artifact,
                state=states[index],
                full45_index=index,
                runtime=bindings.runtime,
                parse_error_class=bindings.parse_error_class,
                distance_backend=bindings.distance_backend,
                image_decoder=_decode_rgb_image,
                run_contract_sha256=layout.run_contract_sha256,
                message_builder=build_full45_messages,
            )
            envelope = wrap_measurement_record(inner, spec=spec)
            _write_json_exclusive(state_path, envelope)
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
                layout=layout, spec=spec, root_ledger=ledger_path, value=ledger
            )
        outcome = WORKER_OUTCOME_COMPLETE
        failure = None
        ledger_status = "WORKER_SHARD_COMPLETED"
    except Exception as error:
        outcome = WORKER_OUTCOME_INVALID
        failure = {
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
        layout=layout, spec=spec, root_ledger=ledger_path, value=ledger
    )
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_contract_sha256": layout.run_contract_sha256,
        "worker": spec.to_dict(),
        "outcome": outcome,
        "completed_state_indices": completed,
        "attempted_state_indices": attempted,
        "failure": failure,
        "ended_at_utc": _utc_now(),
    }
    _write_json_exclusive(terminal_path, terminal)
    return terminal


def wait_for_two_worker_runtime_barrier(
    layout: V22Layout, *, timeout_seconds: float = 900.0
) -> None:
    deadline = time.monotonic() + timeout_seconds
    specs = expected_worker_specs()
    while True:
        paths = {
            spec.worker_id: _worker_root(layout, spec) / WORKER_RUNTIME_FILENAME
            for spec in specs
        }
        if all(path.is_file() for path in paths.values()):
            runtimes = {
                spec.worker_id: _strict_json(paths[spec.worker_id])["runtime_metadata"]
                for spec in specs
            }
            validate_worker_runtime_pair(runtimes)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("two-worker runtime identity barrier timed out")
        time.sleep(0.25)


def wait_for_coordinator_runtime_release(
    layout: V22Layout, *, timeout_seconds: float = 900.0
) -> None:
    release = layout.root / RUNTIME_BARRIER_RELEASE_FILENAME
    abort = layout.root / RUNTIME_BARRIER_ABORT_FILENAME
    deadline = time.monotonic() + timeout_seconds
    while True:
        for spec in expected_worker_specs():
            terminal = _worker_root(layout, spec) / WORKER_TERMINAL_FILENAME
            if terminal.is_file() and _strict_json(terminal).get("outcome") == WORKER_OUTCOME_INVALID:
                raise RuntimeError("peer worker became INVALID before runtime release")
        if abort.is_file():
            record = _strict_json(abort)
            raise RuntimeError(
                "coordinator rejected two-worker runtime barrier: "
                + str(record.get("message"))
            )
        if release.is_file():
            record = _strict_json(release)
            if record != {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": "COORDINATOR_RELEASED_BOTH_WORKERS",
                "run_contract_sha256": layout.run_contract_sha256,
            }:
                raise ValueError("coordinator runtime release token drifted")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("coordinator runtime release timed out")
        time.sleep(0.25)


def coordinate_two_worker_runtime_barrier(
    layout: V22Layout,
    *,
    processes: Sequence[multiprocessing.Process],
    timeout_seconds: float = 900.0,
) -> None:
    specs = expected_worker_specs()
    release = layout.root / RUNTIME_BARRIER_RELEASE_FILENAME
    abort = layout.root / RUNTIME_BARRIER_ABORT_FILENAME
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            paths = {
                spec.worker_id: _worker_root(layout, spec) / WORKER_RUNTIME_FILENAME
                for spec in specs
            }
            if any(process.exitcode is not None for process in processes):
                raise RuntimeError("worker exited before coordinator runtime release")
            if all(path.is_file() for path in paths.values()):
                runtimes = {
                    spec.worker_id: _strict_json(paths[spec.worker_id])[
                        "runtime_metadata"
                    ]
                    for spec in specs
                }
                validate_worker_runtime_pair(runtimes)
                if not all(process.is_alive() for process in processes):
                    raise RuntimeError("worker lost liveness at runtime release")
                _publish_json_exclusive_atomic(
                    release,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "protocol_id": PROTOCOL_ID,
                        "status": "COORDINATOR_RELEASED_BOTH_WORKERS",
                        "run_contract_sha256": layout.run_contract_sha256,
                    },
                )
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("coordinator runtime identity barrier timed out")
            time.sleep(0.25)
    except Exception as error:
        if not abort.exists():
            _publish_json_exclusive_atomic(
                abort,
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol_id": PROTOCOL_ID,
                    "status": "COORDINATOR_ABORTED_RUNTIME_BARRIER",
                    "run_contract_sha256": layout.run_contract_sha256,
                    "message": str(error),
                },
            )
        raise


def _worker_process_entry(
    layout: V22Layout,
    spec: WorkerSpec,
    args: argparse.Namespace,
) -> None:
    authorized = authorize_production_v22(args)
    rebuilt_contract = build_production_run_contract(args, authorized)
    if sha256_bytes(canonical_json_bytes(rebuilt_contract)) != layout.run_contract_sha256:
        raise ValueError("worker rebuilt authorization differs from global claim")
    run_worker_shard(
        layout=layout,
        spec=spec,
        artifact=authorized.artifact,
        states=authorized.states,
        runtime_loader=lambda selected: load_worker_runtime(
            spec=selected,
            model_dir=args.model_dir,
            snapshot_manifest=str(authorized.snapshot_manifest_path),
            container_image_digest=args.container_image_digest,
        ),
        runtime_barrier=wait_for_coordinator_runtime_release,
    )


def launch_spawned_workers(
    *,
    layout: V22Layout,
    specs: Sequence[WorkerSpec],
    args: argparse.Namespace,
) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_worker_process_entry,
            args=(
                layout,
                spec,
                args,
            ),
            name=f"causalcache-v22-{spec.worker_id}",
        )
        for spec in specs
    ]
    for process in processes:
        process.start()
    barrier_error: BaseException | None = None
    try:
        coordinate_two_worker_runtime_barrier(layout, processes=processes)
    except BaseException as error:
        barrier_error = error
    for process in processes:
        process.join()
    if barrier_error is not None:
        raise RuntimeError("two-worker coordinator barrier failed") from barrier_error
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError("one or more v2.2 worker processes exited abnormally")


def _worker_observation(layout: V22Layout, spec: WorkerSpec) -> dict[str, Any]:
    root = _worker_root(layout, spec)
    ledger_path = root / WORKER_LEDGER_FILENAME
    sibling_path = layout.worker_sibling_ledgers[spec.worker_id]
    if not sibling_path.is_file():
        return {
            "attempted": [],
            "completed": [],
            "ledger_sha256": None,
            "terminal": None,
            "root_ledger_matches": False,
            "marker_indices": [],
            "state_indices": [],
        }
    ledger = _strict_json(sibling_path)
    root_record = _strict_json(ledger_path) if ledger_path.is_file() else None
    root_matches = (
        root_record is not None and ledger_path.read_bytes() == sibling_path.read_bytes()
    )
    marker_indices = sorted(
        int(path.stem)
        for path in (root / ATTEMPT_DIRECTORY).glob("*.json")
        if path.stem.isdigit()
    )
    state_indices = sorted(
        int(path.stem)
        for path in (root / STATE_DIRECTORY).glob("*.json")
        if path.stem.isdigit()
    )
    return {
        "attempted": list(ledger["attempted_state_indices"]),
        "completed": list(ledger["completed_state_indices"]),
        "ledger_sha256": sha256_file(sibling_path),
        "terminal": (
            _strict_json(root / WORKER_TERMINAL_FILENAME)
            if (root / WORKER_TERMINAL_FILENAME).is_file()
            else None
        ),
        "root_ledger_matches": root_matches,
        "root_attempted": (
            list(root_record["attempted_state_indices"])
            if root_record is not None
            else None
        ),
        "root_completed": (
            list(root_record["completed_state_indices"])
            if root_record is not None
            else None
        ),
        "root_ledger_sha256": (
            sha256_file(ledger_path) if root_record is not None else None
        ),
        "marker_indices": marker_indices,
        "state_indices": state_indices,
    }


def _update_global_terminal_ledger(
    layout: V22Layout, *, observations: Mapping[str, Mapping[str, Any]], valid: bool
) -> None:
    ledger = _strict_json(layout.global_ledger)
    attempted = sum(len(value["attempted"]) for value in observations.values())
    completed = sum(len(value["completed"]) for value in observations.values())
    ledger.update(
        {
            "status": "GLOBAL_ATTEMPT_COMPLETED" if valid else "GLOBAL_ATTEMPT_INVALID",
            "completed_state_count": completed,
            "attempted_state_count": attempted,
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
    _replace_json_durable(layout.global_ledger, ledger)


def _invalid_aggregate(
    layout: V22Layout,
    *,
    observations: Mapping[str, Mapping[str, Any]],
    stage: str,
    error: BaseException,
) -> dict[str, Any]:
    forensic = {}
    for spec in expected_worker_specs():
        observation = observations[spec.worker_id]
        attempted = set(observation.get("attempted", []))
        completed = set(observation.get("completed", []))
        markers = set(observation.get("marker_indices", []))
        states = set(observation.get("state_indices", []))
        forensic[spec.worker_id] = {
            "sibling_attempted_state_indices": sorted(attempted),
            "sibling_completed_state_indices": sorted(completed),
            "observed_attempt_marker_indices": sorted(markers),
            "observed_state_record_indices": sorted(states),
            "missing_attempt_marker_indices": sorted(attempted - markers),
            "orphan_attempt_marker_indices": sorted(markers - attempted),
            "missing_state_record_indices": sorted(completed - states),
            "orphan_state_record_indices": sorted(states - completed),
            "root_ledger_status": (
                "exact_sibling_copy"
                if observation.get("root_ledger_matches") is True
                else "missing_or_stale"
            ),
            "root_ledger_attempted_state_indices": observation.get(
                "root_attempted"
            ),
            "root_ledger_completed_state_indices": observation.get(
                "root_completed"
            ),
            "root_ledger_sha256": observation.get("root_ledger_sha256"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_contract_sha256": layout.run_contract_sha256,
        "status": "INVALID_TWO_WORKER_V2_2_EAGER_SUBSTRATE",
        "outcome": INVALID_OUTCOME,
        "invalid_failure": {
            "stage": stage,
            "exception_type": error.__class__.__name__,
            "message": str(error),
        },
        "completed_state_count": sum(
            len(value["completed"]) for value in observations.values()
        ),
        "attempted_state_count": sum(
            len(value["attempted"]) for value in observations.values()
        ),
        "started_at_utc": layout.started_at_utc,
        "ended_at_utc": _utc_now(),
        "retry_performed": False,
        "top_up_performed": False,
        "forensic_inventory": forensic,
    }


def finalize_two_worker_attempt(layout: V22Layout) -> dict[str, Any]:
    specs = expected_worker_specs()
    observations = {
        spec.worker_id: _worker_observation(layout, spec) for spec in specs
    }
    try:
        terminals = {
            spec.worker_id: observations[spec.worker_id]["terminal"] for spec in specs
        }
        if any(
            not isinstance(terminals[spec.worker_id], Mapping)
            or terminals[spec.worker_id].get("outcome") != WORKER_OUTCOME_COMPLETE
            or observations[spec.worker_id]["attempted"] != list(spec.state_indices)
            or observations[spec.worker_id]["completed"] != list(spec.state_indices)
            or observations[spec.worker_id]["root_ledger_matches"] is not True
            for spec in specs
        ):
            raise RuntimeError("both exact parity shards did not terminate completely")
        runtime_pair = {
            spec.worker_id: _strict_json(
                _worker_root(layout, spec) / WORKER_RUNTIME_FILENAME
            )["runtime_metadata"]
            for spec in specs
        }
        validate_worker_runtime_pair(runtime_pair)
        envelopes: list[Mapping[str, Any]] = []
        for index in range(45):
            spec = specs[index % 2]
            path = (
                _worker_root(layout, spec)
                / STATE_DIRECTORY
                / f"{index:03d}.json"
            )
            envelopes.append(_strict_json(path))
        aggregate = aggregate_v22_gate(
            envelopes,
            projections=layout.projections,
            run_contract=layout.run_contract,
            run_contract_sha256=layout.run_contract_sha256,
            started_at_utc=layout.started_at_utc,
            ended_at_utc=_utc_now(),
        )
        _update_global_terminal_ledger(layout, observations=observations, valid=True)
    except Exception as error:
        aggregate = _invalid_aggregate(
            layout,
            observations=observations,
            stage="two_worker_terminal_merge_and_cpu_gate",
            error=error,
        )
        _update_global_terminal_ledger(layout, observations=observations, valid=False)
    _write_json_exclusive(layout.aggregate, aggregate)
    return aggregate


def execute_two_worker_attempt(
    *,
    run_contract: Mapping[str, Any],
    output_dir: str | Path,
    global_ledger: str | Path,
    worker_launcher: Callable[[V22Layout, Sequence[WorkerSpec]], None],
) -> dict[str, Any]:
    layout = claim_global_attempt(
        run_contract=run_contract,
        output_dir=output_dir,
        global_ledger=global_ledger,
    )
    ledger = _strict_json(layout.global_ledger)
    ledger["spawn_started_at_utc"] = _utc_now()
    _replace_json_durable(layout.global_ledger, ledger)
    try:
        worker_launcher(layout, expected_worker_specs())
    except Exception:
        pass
    return finalize_two_worker_attempt(layout)


def seal_interrupted_attempt(
    *, output_dir: str | Path, global_ledger: str | Path
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    ledger_path = Path(global_ledger).resolve()
    manifest = _strict_json(root / RUN_MANIFEST_FILENAME)
    run_contract = manifest.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise ValueError("interrupted v2.2 manifest lacks run contract")
    validate_v22_run_contract_identity(
        run_contract,
        require_canonical_attempt_identity=False,
    )
    run_contract_sha256 = sha256_bytes(canonical_json_bytes(run_contract))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != RUN_STATUS
        or manifest.get("run_contract_sha256") != run_contract_sha256
        or (root / AGGREGATE_FILENAME).exists()
    ):
        raise ValueError("interrupted v2.2 attempt is not sealable")
    attempt_identity = run_contract.get("attempt_identity")
    if (
        not isinstance(attempt_identity, Mapping)
        or root != Path(str(attempt_identity.get("output_dir"))).resolve()
        or ledger_path != Path(str(attempt_identity.get("global_ledger"))).resolve()
    ):
        raise ValueError("interrupted v2.2 paths differ from the claimed run contract")
    ledger = _strict_json(ledger_path)
    if (
        ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("protocol_id") != PROTOCOL_ID
        or ledger.get("attempt_id") != CANONICAL_ATTEMPT_ID
        or ledger.get("run_contract_sha256") != run_contract_sha256
        or ledger.get("status")
        not in {
            GLOBAL_CLAIM_STATUS,
            "GLOBAL_ATTEMPT_COMPLETED",
            "GLOBAL_ATTEMPT_INVALID",
        }
        or ledger.get("retry_count") != 0
        or ledger.get("top_up_count") != 0
    ):
        raise ValueError("interrupted global ledger differs from manifest")
    sibling_record = ledger.get("worker_sibling_ledgers")
    if not isinstance(sibling_record, Mapping):
        raise ValueError("interrupted global ledger lacks worker sibling paths")
    expected_sibling_paths = _worker_sibling_paths(ledger_path)
    sibling_paths = {
        worker_id: Path(str(path)).resolve()
        for worker_id, path in sibling_record.items()
    }
    if sibling_paths != expected_sibling_paths:
        raise ValueError("interrupted worker sibling paths differ from the global claim")
    layout = V22Layout(
        root=root,
        global_ledger=ledger_path,
        manifest=root / RUN_MANIFEST_FILENAME,
        aggregate=root / AGGREGATE_FILENAME,
        run_contract=dict(run_contract),
        run_contract_sha256=run_contract_sha256,
        projections=tuple(dict(value) for value in run_contract["states"]),
        started_at_utc=str(manifest["created_at_utc"]),
        worker_sibling_ledgers=sibling_paths,
    )
    observations = {
        spec.worker_id: _worker_observation(layout, spec)
        for spec in expected_worker_specs()
    }
    if any(
        observations[spec.worker_id]["ledger_sha256"] is None
        for spec in expected_worker_specs()
    ):
        raise ValueError("interrupted v2.2 attempt lacks a prebound sibling ledger")
    aggregate = _invalid_aggregate(
        layout,
        observations=observations,
        stage="sealed_non_resumable_interrupted_attempt",
        error=RuntimeError("coordinator or worker interrupted before terminal aggregate"),
    )
    _update_global_terminal_ledger(layout, observations=observations, valid=False)
    _write_json_exclusive(layout.aggregate, aggregate)
    return aggregate


def execute_production_v22(
    args: argparse.Namespace,
    *,
    authorization_loader: Callable[[argparse.Namespace], AuthorizedV22] = (
        authorize_production_v22
    ),
) -> dict[str, Any]:
    authorized = authorization_loader(args)
    run_contract = build_production_run_contract(args, authorized)
    return execute_two_worker_attempt(
        run_contract=run_contract,
        output_dir=args.output_dir,
        global_ledger=args.global_ledger,
        worker_launcher=lambda layout, specs: launch_spawned_workers(
            layout=layout,
            specs=specs,
            args=args,
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--spatial-reference-evidence", required=True)
    parser.add_argument("--v2-1-full-45-evidence", required=True)
    parser.add_argument("--pilot-evidence", required=True)
    parser.add_argument("--processor-evidence", required=True)
    parser.add_argument("--derived-artifact-root", required=True)
    parser.add_argument("--scientific-config", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--ocr-backend-config", required=True)
    parser.add_argument("--snapshot-manifest", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--host-hostname", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--global-ledger", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(raw_argv)
    args.execution_argv = [sys.executable, str(Path(__file__).resolve()), *raw_argv]
    try:
        result = execute_production_v22(args)
    except Exception as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "INVALID_BEFORE_V2_2_GLOBAL_CLAIM",
            "outcome": INVALID_OUTCOME,
            "invalid_failure": {
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
