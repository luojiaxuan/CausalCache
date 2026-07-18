#!/usr/bin/env python3
"""Run the one-shot independent confirm-20 protocol on four H200 GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.data.guiodyssey_restoration_v2 import (
    MANIFEST_RELATIVE_PATH,
    build_ocr_backend_provenance,
    load_json_object,
    sha256_file,
    source_dataset_identity_from_v1_config,
    validate_frozen_inputs,
    validate_ocr_runtime_identity,
)
from causalcache.independent_confirm_contract import (
    RUNNER_FREEZE_PATH,
    IndependentConfirmContract,
    canonical_json_bytes,
)
from causalcache.independent_confirm_data import load_frozen_confirm20_artifact
from causalcache.independent_confirm_execution import (
    HuggingFaceConfirmPublisher,
    bundle_from_validated_payloads,
    download_and_load_independent_ensemble,
    execute_independent_confirm,
    make_real_policy_vision_phase,
    make_real_restoration_phase,
    read_hf_token_file,
    validate_gpu_topology_smoke_receipt,
)
from causalcache.policy.gui_owl_v2_vision import verify_frozen_vision_runtime
from causalcache.restoration_v2_contract import RestorationV2Contract
from causalcache.restoration_v2_text_backend import load_backend_config
from scripts.manage_independent_confirm import validate_existing
from scripts.validate_restoration_v2_ocr_backend import validate_artifact_source
from scripts.validate_restoration_v2_real_screen import runtime_identity_and_engine


_COMMIT = re.compile(r"[0-9a-f]{40}")
_CONTAINER = re.compile(r"[0-9a-f]{64}")
_GPU_UUID = re.compile(
    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
)
_HOSTS = {
    "hyper00": "node-radixark-16-0000",
    "hyper01": "node-radixark-16-0001",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_GPU_TOPOLOGY_SMOKE_RECEIPT_BYTES = 2 * 1024 * 1024
_CANONICAL_INPUTS = {
    "scientific_contract": "code/configs/causalcache_restoration_v2.json",
    "v1_config": "code/configs/independent_reference_gate_v1.json",
    "source_file_manifest": (
        "data/manifests/independent_reference_gate_v1_source_files.json"
    ),
    "selection_manifest": "data/manifests/restoration_v2_selection.json",
    "exposure_manifest": "data/manifests/restoration_v2_exposure.json",
    "ocr_backend_config": "code/configs/restoration_v2_ocr_backend.json",
    "ocr_backend_manifest": "data/manifests/restoration_v2_ocr_backend.json",
    "snapshot_manifest": "code/configs/gui_owl_1_5_8b_snapshot.json",
}


def _outside_repository(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return resolved
    raise ValueError(f"{label} must remain outside the Git repository")


def _canonical_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"canonical input is missing or unsafe: {relative}")
    return path


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in GPU topology smoke receipt: {key}")
        result[key] = value
    return result


def _reject_non_finite_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON value in GPU topology smoke receipt: {value}")


def _load_gpu_topology_smoke_receipt(
    path: str | Path,
    *,
    repository_root: Path,
    expected_sha256: str,
) -> tuple[Path, Mapping[str, Any], str]:
    supplied = Path(path)
    if (
        not supplied.is_absolute()
        or not isinstance(expected_sha256, str)
        or _SHA256.fullmatch(expected_sha256) is None
    ):
        raise ValueError(
            "GPU topology smoke receipt requires an absolute path and explicit SHA256"
        )
    try:
        resolved = supplied.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise ValueError("GPU topology smoke receipt is missing or unsafe") from error
    if supplied != resolved or resolved.is_symlink():
        raise ValueError("GPU topology smoke receipt path must be canonical and non-symlink")
    _outside_repository(resolved, repository_root, label="GPU topology smoke receipt")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise ValueError("GPU topology smoke receipt is missing or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_GPU_TOPOLOGY_SMOKE_RECEIPT_BYTES
        ):
            raise ValueError("GPU topology smoke receipt must be a bounded regular file")
        chunks = []
        remaining = _MAX_GPU_TOPOLOGY_SMOKE_RECEIPT_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(payload) != before.st_size
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
        ):
            raise ValueError("GPU topology smoke receipt changed while being read")
    finally:
        os.close(descriptor)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("GPU topology smoke receipt SHA256 differs from explicit argv")
    try:
        text = payload.decode("utf-8", errors="strict")
        body = json.loads(
            text,
            object_pairs_hook=_json_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("GPU topology smoke receipt is not strict JSON") from error
    if not isinstance(body, dict):
        raise ValueError("GPU topology smoke receipt must be a JSON object")
    if payload != canonical_json_bytes(body) + b"\n":
        raise ValueError("GPU topology smoke receipt bytes are not canonical")
    return resolved, body, observed_sha256


def _load_runner_authorization(
    *,
    root: Path,
    contract_path: Path,
    expected_execution_b_commit: str,
) -> Mapping[str, Any]:
    if _COMMIT.fullmatch(expected_execution_b_commit) is None:
        raise ValueError("expected Execution-B commit must be a full Git SHA")
    result = validate_existing(
        repository_root=root,
        contract_path=contract_path,
        expected_execution_b_commit=expected_execution_b_commit,
    )
    freeze = result.get("runner_freeze")
    if (
        result.get("status")
        != "VALIDATED_CAUSALCACHE_INDEPENDENT_CONFIRM_EXECUTION_B_V1"
        or not isinstance(freeze, Mapping)
        or freeze.get("confirm_access_authorized") is not True
        or freeze.get("restoration_access_requires_payload_commit") is not True
        or freeze.get("sealed_test_requires_development_go") is not True
    ):
        raise PermissionError("Execution-B runner freeze does not authorize confirm")
    return result


def _formal_parent_view(
    *,
    repository_root: Path,
    generator_source_root: Path,
    parent_root: Path,
    ocr_model_dir: Path,
    ocr_wheel_dir: Path,
) -> Any:
    paths = {
        name: _canonical_path(repository_root, relative)
        for name, relative in _CANONICAL_INPUTS.items()
        if name != "snapshot_manifest"
    }
    v2_contract = RestorationV2Contract.load(paths["scientific_contract"])
    _, v1_config = load_json_object(paths["v1_config"])
    load_json_object(paths["source_file_manifest"])
    _, selection = load_json_object(paths["selection_manifest"])
    _, exposure = load_json_object(paths["exposure_manifest"])
    backend = load_backend_config(paths["ocr_backend_config"])
    _, backend_manifest = load_json_object(paths["ocr_backend_manifest"])
    validate_artifact_source(
        backend_config_path=paths["ocr_backend_config"],
        artifact_manifest_path=paths["ocr_backend_manifest"],
        repository_root=repository_root,
    )
    validate_frozen_inputs(
        v2_contract=v2_contract.data,
        selection_manifest=selection,
        selection_manifest_sha256=sha256_file(paths["selection_manifest"]),
        v1_config_sha256=sha256_file(paths["v1_config"]),
        source_file_manifest_sha256=sha256_file(paths["source_file_manifest"]),
        exposure_manifest=exposure,
        backend_config=backend,
    )
    runtime_identity, engine = runtime_identity_and_engine(
        backend_config=backend,
        model_dir=ocr_model_dir,
        wheel_dir=ocr_wheel_dir,
    )
    validate_ocr_runtime_identity(runtime_identity, backend_config=backend)
    _, parent_manifest = load_json_object(parent_root / MANIFEST_RELATIVE_PATH)
    generator = parent_manifest.get("generator")
    if not isinstance(generator, Mapping):
        raise ValueError("confirm parent manifest lacks generator identity")
    generator_revision = generator.get("git_revision")
    if not isinstance(generator_revision, str) or _COMMIT.fullmatch(
        generator_revision
    ) is None:
        raise ValueError("confirm parent generator revision is invalid")
    bindings = {
        "selection_manifest": selection,
        "expected_input_sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "expected_ocr_backend_provenance": build_ocr_backend_provenance(
            backend_config_sha256=sha256_file(paths["ocr_backend_config"]),
            backend_manifest_sha256=sha256_file(paths["ocr_backend_manifest"]),
            backend_manifest=backend_manifest,
        ),
        "expected_ocr_runtime_identity": runtime_identity,
        "expected_generator_git_revision": generator_revision,
        "expected_dataset_repo": str(
            v2_contract.data["data"]["derived_artifact"]["repo"]
        ),
        "expected_source_dataset": source_dataset_identity_from_v1_config(
            v1_config
        ),
        "repository_root": generator_source_root,
        "ocr_engine": engine,
    }
    return load_frozen_confirm20_artifact(
        output_dir=parent_root,
        backend_config=backend,
        backend_config_sha256=sha256_file(paths["ocr_backend_config"]),
        validation_bindings=bindings,
    )


def _model_snapshot_identity(
    *,
    contract: IndependentConfirmContract,
    repository_root: Path,
    model_dir: Path,
) -> tuple[Path, Mapping[str, Any]]:
    snapshot = _canonical_path(
        repository_root, _CANONICAL_INPUTS["snapshot_manifest"]
    )
    identity = verify_frozen_vision_runtime(
        model_dir=model_dir,
        expected_snapshot_manifest=snapshot,
    )
    observed = {
        "model_dir": getattr(identity, "model_dir", None),
        "model_repo": getattr(identity, "model_repo", None),
        "model_revision": getattr(identity, "model_revision", None),
        "snapshot_manifest_sha256": getattr(
            identity, "snapshot_manifest_sha256", None
        ),
        "verified_model_file_count": getattr(
            identity, "verified_model_file_count", None
        ),
        "verified_model_total_bytes": getattr(
            identity, "verified_model_total_bytes", None
        ),
    }
    restoration = contract.data["restoration_contract"]
    if (
        observed["model_repo"] != restoration["policy_repo"]
        or observed["model_revision"] != restoration["policy_revision"]
        or observed["snapshot_manifest_sha256"]
        != restoration["snapshot_manifest_sha256"]
    ):
        raise ValueError("local policy snapshot differs from the confirm contract")
    return snapshot, observed


def _validate_runtime_args(
    args: argparse.Namespace,
    *,
    contract: IndependentConfirmContract,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    devices = tuple(part.strip() for part in args.devices.split(","))
    gpu_uuids = tuple(part.strip() for part in args.gpu_uuids.split(","))
    runtime = contract.data["runtime_contract"]
    required_environment = runtime["thread_environment"]
    observed_environment = {
        name: os.environ.get(name) for name in required_environment
    }
    if observed_environment != required_environment:
        raise ValueError(
            "frozen thread/runtime environment must be set before launching the CLI"
        )
    if (
        args.host_alias not in _HOSTS
        or args.host_hostname != _HOSTS.get(args.host_alias)
        or args.container_image_digest != runtime["container_image_id"]
        or _CONTAINER.fullmatch(args.container_id) is None
        or len(devices) != 4
        or len(set(devices)) != 4
        or any(re.fullmatch(r"cuda:[0-9]+", item) is None for item in devices)
        or len(gpu_uuids) != 4
        or len(set(gpu_uuids)) != 4
        or any(_GPU_UUID.fullmatch(item) is None for item in gpu_uuids)
    ):
        raise ValueError("formal host, container, device, or GPU UUID identity drifted")
    return devices, gpu_uuids


def _run_contract(
    *,
    contract: IndependentConfirmContract,
    authorization: Mapping[str, Any],
    parent: Any,
    model_identity: Mapping[str, Any],
    args: argparse.Namespace,
    devices: Sequence[str],
    gpu_uuids: Sequence[str],
    gpu_topology_smoke_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    scientific_runtime = contract.data["runtime_contract"]
    return {
        "schema_version": "1.0.0",
        "protocol_id": "causalcache_independent_confirm20_execution_contract_v1",
        "contract_sha256": contract.sha256,
        "source_a_commit": authorization["source_a_commit"],
        "execution_b_commit": authorization["execution_b_commit"],
        "runner_freeze": dict(authorization["runner_freeze"]),
        "parent_artifact": {
            "artifact_tree_sha256": parent.artifact_tree_sha256,
            "source_ids": list(parent.source_ids),
            "state_count": len(parent.trajectories),
            "policy_output_present": False,
            "restoration_output_present": False,
        },
        "independent_model": {
            "repo": contract.model["repo"],
            "payload_commit": contract.model["payload_commit"],
            "manifest_commit": contract.model["manifest_commit"],
            "checkpoint_count": len(contract.model["checkpoints"]),
            "retraining_performed": False,
        },
        "policy_snapshot": dict(model_identity),
        "gpu_topology_smoke_receipt": dict(gpu_topology_smoke_receipt),
        "runtime": {
            "host_alias": args.host_alias,
            "host_hostname": args.host_hostname,
            "container_id": args.container_id,
            "container_image_digest": args.container_image_digest,
            "devices": list(devices),
            "gpu_uuids": list(gpu_uuids),
            "worker_count": 4,
            "states_per_worker": 5,
            "policy_vision_feature_repeats": 2,
            "restoration_teacher_forwards_per_valid_state": 17,
            "policy_vision_phase_timeout_seconds": scientific_runtime[
                "policy_vision_phase_timeout_seconds"
            ],
            "restoration_phase_timeout_seconds": scientific_runtime[
                "restoration_phase_timeout_seconds"
            ],
            "worker_termination_grace_seconds": scientific_runtime[
                "worker_termination_grace_seconds"
            ],
            "execution_argv": list(args.execution_argv),
        },
        "publication": dict(contract.data["output_contract"]["destination"]),
        "prohibited_counts": {
            "retry_count": 0,
            "top_up_count": 0,
            "filter_count": 0,
            "gate_training_count": 0,
            "threshold_update_count": 0,
            "sealed_androidworld_test_access_count": 0,
        },
        "execution_argv_sha256": hashlib.sha256(
            canonical_json_bytes(list(args.execution_argv))
        ).hexdigest(),
    }


def execute(args: argparse.Namespace) -> Mapping[str, Any]:
    root = Path(args.repository_root).resolve()
    contract_path = Path(args.contract).resolve()
    expected_contract = (
        root / "code/configs/causalcache_independent_confirm_closed_loop_v1.json"
    ).resolve()
    expected_freeze = (root / RUNNER_FREEZE_PATH).resolve()
    if contract_path != expected_contract or Path(args.runner_freeze).resolve() != expected_freeze:
        raise ValueError("confirm contract or runner freeze path is not canonical")
    authorization = _load_runner_authorization(
        root=root,
        contract_path=contract_path,
        expected_execution_b_commit=args.execution_b_git_commit,
    )
    contract = IndependentConfirmContract.load(
        contract_path,
        repository_root=root,
        require_runner_absent=False,
    )
    parent_root = _outside_repository(
        Path(args.parent_root), root, label="parent artifact root"
    )
    generator_root = Path(args.generator_source_root).resolve()
    model_dir = _outside_repository(Path(args.model_dir), root, label="model directory")
    ocr_model_dir = _outside_repository(
        Path(args.ocr_model_dir), root, label="OCR model directory"
    )
    ocr_wheel_dir = _outside_repository(
        Path(args.ocr_wheel_dir), root, label="OCR wheel directory"
    )
    cache_dir = _outside_repository(
        Path(args.hf_cache_dir), root, label="HF cache directory"
    )
    output_dir = _outside_repository(
        Path(args.output_dir), root, label="confirm output directory"
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("confirm output already exists; retry is forbidden")
    devices, gpu_uuids = _validate_runtime_args(args, contract=contract)
    snapshot = _canonical_path(root, _CANONICAL_INPUTS["snapshot_manifest"])
    receipt_path, receipt_body, receipt_sha256 = _load_gpu_topology_smoke_receipt(
        args.gpu_topology_smoke_receipt,
        repository_root=root,
        expected_sha256=args.gpu_topology_smoke_receipt_sha256,
    )
    validated_receipt = validate_gpu_topology_smoke_receipt(
        receipt_body,
        contract=contract,
        expected_devices=devices,
        expected_gpu_uuids=gpu_uuids,
        expected_model_dir=model_dir,
        expected_snapshot_manifest=snapshot,
    )
    receipt_binding = {
        "path": str(receipt_path),
        "sha256": receipt_sha256,
        "body": dict(validated_receipt),
    }
    payloads = _formal_parent_view(
        repository_root=root,
        generator_source_root=generator_root,
        parent_root=parent_root,
        ocr_model_dir=ocr_model_dir,
        ocr_wheel_dir=ocr_wheel_dir,
    )
    verified_snapshot, model_identity = _model_snapshot_identity(
        contract=contract,
        repository_root=root,
        model_dir=model_dir,
    )
    if verified_snapshot != snapshot:
        raise RuntimeError("canonical policy snapshot manifest path changed during preflight")
    run_contract = _run_contract(
        contract=contract,
        authorization=authorization,
        parent=payloads,
        model_identity=model_identity,
        args=args,
        devices=devices,
        gpu_uuids=gpu_uuids,
        gpu_topology_smoke_receipt=receipt_binding,
    )
    token = read_hf_token_file(args.hf_token_file)
    publisher = HuggingFaceConfirmPublisher(
        cache_dir=cache_dir,
        token=token,
    )
    publisher.preflight_destination()
    loaded = download_and_load_independent_ensemble(
        contract, publisher=publisher
    )
    if loaded.checkpoint_load_count != 5:
        raise RuntimeError("formal independent ensemble did not load five checkpoints")
    bundle = bundle_from_validated_payloads(payloads)
    result = execute_independent_confirm(
        contract=contract,
        bundle=bundle,
        ensemble=loaded.ensemble,
        run_contract=run_contract,
        output_dir=output_dir,
        publisher=publisher,
        policy_vision_phase=make_real_policy_vision_phase(
            model_dir=model_dir,
            snapshot_manifest=verified_snapshot,
            devices=devices,
            gpu_uuids=gpu_uuids,
            phase_timeout_seconds=run_contract["runtime"][
                "policy_vision_phase_timeout_seconds"
            ],
            worker_termination_grace_seconds=run_contract["runtime"][
                "worker_termination_grace_seconds"
            ],
        ),
        restoration_phase=make_real_restoration_phase(
            payloads=payloads,
            bundle=bundle,
            model_dir=model_dir,
            snapshot_manifest=verified_snapshot,
            devices=devices,
            gpu_uuids=gpu_uuids,
            phase_timeout_seconds=run_contract["runtime"][
                "restoration_phase_timeout_seconds"
            ],
            worker_termination_grace_seconds=run_contract["runtime"][
                "worker_termination_grace_seconds"
            ],
        ),
    )
    return result.completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--runner-freeze", type=Path, required=True)
    parser.add_argument("--execution-b-git-commit", required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--generator-source-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ocr-model-dir", type=Path, required=True)
    parser.add_argument("--ocr-wheel-dir", type=Path, required=True)
    parser.add_argument("--hf-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-token-file", type=Path, required=True)
    parser.add_argument(
        "--gpu-topology-smoke-receipt",
        type=Path,
        required=True,
        help="Canonical absolute data-blind topology smoke receipt outside Git",
    )
    parser.add_argument(
        "--gpu-topology-smoke-receipt-sha256",
        required=True,
        help="Explicit SHA256 of the canonical topology smoke receipt bytes",
    )
    parser.add_argument("--host-alias", choices=tuple(_HOSTS), required=True)
    parser.add_argument("--host-hostname", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument(
        "--devices",
        required=True,
        help="Exactly four comma-separated explicit CUDA devices",
    )
    parser.add_argument(
        "--gpu-uuids",
        required=True,
        help="Exactly four comma-separated physical GPU UUIDs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    effective = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(effective)
    args.execution_argv = effective
    completion = execute(args)
    print(json.dumps(completion, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
