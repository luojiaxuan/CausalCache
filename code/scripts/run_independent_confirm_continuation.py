#!/usr/bin/env python3
"""Run the restoration-only continuation from the sealed confirm-20 payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import queue as queue_module
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.independent_confirm_continuation import (
    execute_independent_confirm_continuation,
    validate_expected_payload_files,
)
from causalcache.independent_confirm_continuation_contract import (
    CONFIG_PATH,
    EXPECTED_BASE_COMMIT,
    EXPECTED_PAYLOAD_COMMIT,
    PARENT_CONFIG_PATH,
    RUNNER_FREEZE_PATH,
    IndependentConfirmContinuationContract,
    canonical_json_bytes,
)
from causalcache.independent_confirm_contract import IndependentConfirmContract
from causalcache.independent_confirm_artifact import PayloadPublicationReceipt
from causalcache.independent_confirm_execution import (
    DurableConfirmOutput,
    HuggingFaceConfirmPublisher,
    install_fork_parent_cuda_tripwire,
    make_real_restoration_phase,
    read_hf_token_file,
    release_fork_parent_cuda_tripwire,
    validate_gpu_topology_smoke_receipt,
)
from scripts.smoke_independent_confirm_gpu_topology import (
    CONTINUATION_ENVELOPE_PROTOCOL_ID,
    continuation_topology_challenge,
    continuation_topology_challenge_response,
)
from scripts.manage_independent_confirm_continuation import validate_existing
from scripts.run_independent_confirm import (
    _formal_parent_view,
    _load_gpu_topology_smoke_receipt,
    _validate_runtime_args,
)


_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _unwrap_fresh_topology_envelope(
    envelope: Mapping[str, Any],
    *,
    execution_b_commit: str,
    topology_nonce: str,
    expected_devices: Sequence[str],
    expected_gpu_uuids: Sequence[str],
    forbidden_parent_receipt_sha256: str,
) -> tuple[Mapping[str, Any], str, str, str]:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "execution_b_commit",
        "topology_nonce",
        "parent_receipt_file_sha256",
        "challenge_sha256",
        "challenge_responses",
        "parent_receipt",
    }
    parent = envelope.get("parent_receipt")
    digest = envelope.get("parent_receipt_file_sha256")
    challenge = envelope.get("challenge_sha256")
    responses = envelope.get("challenge_responses")
    if (
        set(envelope) != expected_keys
        or envelope.get("schema_version") != "1.0.0"
        or envelope.get("protocol_id") != CONTINUATION_ENVELOPE_PROTOCOL_ID
        or envelope.get("execution_b_commit") != execution_b_commit
        or envelope.get("topology_nonce") != topology_nonce
        or not isinstance(parent, Mapping)
        or _SHA256.fullmatch(str(digest)) is None
        or _SHA256.fullmatch(forbidden_parent_receipt_sha256) is None
        or hashlib.sha256(canonical_json_bytes(dict(parent)) + b"\n").hexdigest()
        != digest
        or digest == forbidden_parent_receipt_sha256
        or _SHA256.fullmatch(str(challenge)) is None
        or not isinstance(responses, Mapping)
        or set(responses) != {"policy_vision_spawn", "teacher_forced_fork"}
    ):
        raise ValueError("fresh continuation topology envelope binding drifted")
    inputs = parent.get("inputs")
    phases = parent.get("phases")
    if not isinstance(inputs, Mapping) or not isinstance(phases, Mapping):
        raise ValueError("fresh continuation topology parent receipt is malformed")
    expected_challenge = continuation_topology_challenge(
        execution_b_commit=execution_b_commit,
        topology_nonce=topology_nonce,
        model_dir=str(inputs.get("model_dir")),
        snapshot_manifest=str(inputs.get("snapshot_manifest")),
        devices=expected_devices,
        gpu_uuids=expected_gpu_uuids,
    )
    if challenge != expected_challenge:
        raise ValueError("fresh continuation topology challenge drifted")
    for phase_name in ("policy_vision_spawn", "teacher_forced_fork"):
        phase = phases.get(phase_name)
        workers = phase.get("workers") if isinstance(phase, Mapping) else None
        phase_responses = responses.get(phase_name)
        if (
            not isinstance(workers, list)
            or len(workers) != 4
            or not isinstance(phase_responses, list)
            or len(phase_responses) != 4
        ):
            raise ValueError("fresh continuation topology challenge coverage drifted")
        for worker, response in zip(workers, phase_responses, strict=True):
            if (
                not isinstance(worker, Mapping)
                or not isinstance(response, Mapping)
                or set(response) != {"worker_id", "response_sha256"}
                or response.get("worker_id") != worker.get("worker_id")
                or response.get("response_sha256")
                != continuation_topology_challenge_response(
                    challenge_sha256=expected_challenge,
                    phase=phase_name,
                    worker_record=worker,
                )
            ):
                raise ValueError("fresh continuation topology challenge response drifted")
    responses_sha256 = hashlib.sha256(
        canonical_json_bytes(dict(responses))
    ).hexdigest()
    return parent, str(digest), str(challenge), responses_sha256


def _outside_repository(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return resolved
    raise ValueError(f"{label} must remain outside the Git repository")


def _isolated_model_snapshot_worker(
    result_queue: Any,
    *,
    repository_root: str,
    parent_contract_path: str,
    model_dir: str,
) -> None:
    try:
        from scripts.run_independent_confirm import _model_snapshot_identity

        contract = IndependentConfirmContract.load(
            parent_contract_path,
            repository_root=repository_root,
            require_runner_absent=False,
        )
        snapshot, identity = _model_snapshot_identity(
            contract=contract,
            repository_root=Path(repository_root),
            model_dir=Path(model_dir),
        )
        result_queue.put(
            {
                "ok": True,
                "snapshot_manifest": str(snapshot),
                "identity": dict(identity),
            }
        )
    except BaseException as error:
        result_queue.put(
            {
                "ok": False,
                "exception_type": error.__class__.__name__,
                "message": str(error),
            }
        )
        raise


def isolated_model_snapshot_identity(
    *,
    repository_root: Path,
    parent_contract_path: Path,
    model_dir: Path,
    timeout_seconds: int,
    context: Any | None = None,
) -> tuple[Path, Mapping[str, Any]]:
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("isolated model verification requires a positive timeout")
    spawn = multiprocessing.get_context("spawn") if context is None else context
    result_queue = spawn.Queue()
    process = spawn.Process(
        target=_isolated_model_snapshot_worker,
        kwargs={
            "result_queue": result_queue,
            "repository_root": str(repository_root),
            "parent_contract_path": str(parent_contract_path),
            "model_dir": str(model_dir),
        },
        name="causalcache-confirm-continuation-model-verification",
    )
    process.start()
    deadline = time.monotonic() + timeout_seconds
    try:
        response = result_queue.get(timeout=timeout_seconds)
        process.join(timeout=max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            process.terminate()
            process.join(timeout=30)
            raise TimeoutError("isolated model verification did not join")
        if process.exitcode != 0 or not isinstance(response, Mapping):
            raise RuntimeError("isolated model verification process failed")
        if response.get("ok") is not True:
            raise RuntimeError(f"isolated model verification failed: {dict(response)}")
        snapshot = Path(str(response.get("snapshot_manifest"))).resolve()
        identity = response.get("identity")
        if not snapshot.is_file() or not isinstance(identity, Mapping):
            raise RuntimeError("isolated model verification returned malformed identity")
        return snapshot, dict(identity)
    except queue_module.Empty as error:
        if process.is_alive():
            process.terminate()
            process.join(timeout=30)
        raise TimeoutError("isolated model verification exceeded timeout") from error
    finally:
        result_queue.close()
        result_queue.join_thread()


def _load_runner_authorization(
    *,
    root: Path,
    contract_path: Path,
    expected_execution_b_commit: str,
) -> Mapping[str, Any]:
    if _COMMIT.fullmatch(expected_execution_b_commit) is None:
        raise ValueError("continuation Execution-B commit must be full SHA")
    result = validate_existing(
        repository_root=root,
        contract_path=contract_path,
        expected_execution_b_commit=expected_execution_b_commit,
    )
    freeze = result.get("runner_freeze")
    if (
        result.get("status")
        != "VALIDATED_CAUSALCACHE_INDEPENDENT_CONFIRM_CONTINUATION_EXECUTION_B_V1"
        or not isinstance(freeze, Mapping)
        or freeze.get("adopt_existing_payload_authorized") is not True
        or freeze.get("restoration_access_requires_adopted_payload_receipt") is not True
        or freeze.get("selector_reexecution_authorized") is not False
        or freeze.get("fresh_topology_receipt_required") is not True
        or _SHA256.fullmatch(str(freeze.get("topology_receipt_nonce"))) is None
    ):
        raise PermissionError("continuation runner freeze does not authorize restoration")
    return result


def _download_expected_payload(
    *,
    contract: IndependentConfirmContinuationContract,
    publisher: HuggingFaceConfirmPublisher,
) -> Mapping[str, bytes]:
    files = {
        row["path"]: publisher.download_bytes(
            repo=contract.payload["repo"],
            repo_type=contract.payload["repo_type"],
            revision=EXPECTED_PAYLOAD_COMMIT,
            path=row["path"],
        )
        for row in contract.payload["targets"]
    }
    validate_expected_payload_files(contract, files)
    return files


def _run_contract(
    *,
    contract: IndependentConfirmContinuationContract,
    parent_contract: IndependentConfirmContract,
    authorization: Mapping[str, Any],
    parent: Any,
    model_identity: Mapping[str, Any],
    topology_binding: Mapping[str, Any],
    args: argparse.Namespace,
    devices: Sequence[str],
    gpu_uuids: Sequence[str],
) -> Mapping[str, Any]:
    return {
        "schema_version": "1.0.0",
        "protocol_id": contract.continuation["run_contract_protocol_id"],
        "contract_sha256": contract.sha256,
        "source_a_commit": authorization["source_a_commit"],
        "execution_b_commit": authorization["execution_b_commit"],
        "runner_freeze": dict(authorization["runner_freeze"]),
        "parent_scientific_contract_sha256": parent_contract.sha256,
        "old_failure": {
            "status": contract.data["failed_attempt"]["status"],
            "evidence_commit": contract.data["failed_attempt"]["evidence_commit"],
            "failure_sha256": contract.data["failed_attempt"]["failure_sha256"],
            "reference_policy_output_count": 0,
            "restoration_output_count": 0,
        },
        "adopted_payload": {
            "base_commit": EXPECTED_BASE_COMMIT,
            "payload_commit": EXPECTED_PAYLOAD_COMMIT,
            "payload_inventory_sha256": contract.payload[
                "payload_inventory_sha256"
            ],
            "fresh_replay": True,
            "payload_mutation_count": 0,
            "report_target_count_before_continuation": 0,
            "tag_count_before_continuation": 0,
        },
        "parent_artifact": {
            "artifact_tree_sha256": parent.artifact_tree_sha256,
            "source_ids": list(parent.source_ids),
            "state_count": len(parent.trajectories),
            "policy_output_present": False,
            "restoration_output_present": False,
        },
        "policy_snapshot": dict(model_identity),
        "gpu_topology_smoke_receipt": dict(topology_binding),
        "runtime": {
            "host_alias": args.host_alias,
            "host_hostname": args.host_hostname,
            "container_id": args.container_id,
            "container_image_digest": args.container_image_digest,
            "devices": list(devices),
            "gpu_uuids": list(gpu_uuids),
            "worker_count": 4,
            "states_per_worker": 5,
            "model_verification_start_method": "spawn",
            "restoration_start_method": "fork",
            "parent_cuda_clean_guard_required": True,
            "parent_cuda_tripwire_required": True,
            "fresh_exec_parent_required": True,
            "restoration_phase_timeout_seconds": contract.continuation[
                "restoration_phase_timeout_seconds"
            ],
            "worker_termination_grace_seconds": contract.continuation[
                "worker_termination_grace_seconds"
            ],
            "execution_argv": list(args.execution_argv),
        },
        "operation_boundary": {
            "runtime_continuation_count": 1,
            "scientific_retry_count": 0,
            "top_up_count": 0,
            "filter_count": 0,
            "selector_checkpoint_load_count": 0,
            "independent_scorer_call_count": 0,
            "policy_vision_call_count": 0,
            "payload_publish_call_count": 0,
            "sealed_androidworld_test_access_count": 0,
        },
        "execution_argv_sha256": hashlib.sha256(
            canonical_json_bytes(list(args.execution_argv))
        ).hexdigest(),
    }


def execute(args: argparse.Namespace) -> Mapping[str, Any]:
    root = Path(args.repository_root).resolve()
    contract_path = Path(args.contract).resolve()
    parent_contract_path = (root / PARENT_CONFIG_PATH).resolve()
    if (
        contract_path != (root / CONFIG_PATH).resolve()
        or Path(args.runner_freeze).resolve() != (root / RUNNER_FREEZE_PATH).resolve()
    ):
        raise ValueError("continuation contract or runner path is not canonical")
    authorization = _load_runner_authorization(
        root=root,
        contract_path=contract_path,
        expected_execution_b_commit=args.execution_b_git_commit,
    )
    contract = IndependentConfirmContinuationContract.load(
        contract_path, repository_root=root, require_runner_absent=False
    )
    parent_contract = IndependentConfirmContract.load(
        parent_contract_path, repository_root=root, require_runner_absent=False
    )
    parent_root = _outside_repository(Path(args.parent_root), root, label="parent root")
    generator_root = Path(args.generator_source_root).resolve()
    model_dir = _outside_repository(Path(args.model_dir), root, label="model directory")
    ocr_model_dir = _outside_repository(
        Path(args.ocr_model_dir), root, label="OCR model directory"
    )
    ocr_wheel_dir = _outside_repository(
        Path(args.ocr_wheel_dir), root, label="OCR wheel directory"
    )
    cache_dir = _outside_repository(Path(args.hf_cache_dir), root, label="HF cache")
    output_dir = _outside_repository(
        Path(args.output_dir), root, label="continuation output"
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("continuation output already exists")
    devices, gpu_uuids = _validate_runtime_args(args, contract=parent_contract)
    output = DurableConfirmOutput(output_dir)
    output.start_attempt(
        {
            "schema_version": "1.0.0",
            "protocol_id": contract.data["protocol_id"],
            "status": "STARTED_INDEPENDENT_CONFIRM20_CONTINUATION_V1",
            "contract_sha256": contract.sha256,
            "execution_b_commit": args.execution_b_git_commit,
            "execution_argv_sha256": hashlib.sha256(
                canonical_json_bytes(list(args.execution_argv))
            ).hexdigest(),
            "runtime_continuation_count": 1,
            "scientific_retry_count": 0,
        }
    )
    stage = "install_cuda_parent_tripwire"
    publisher: HuggingFaceConfirmPublisher | None = None
    tripwire = None
    try:
        tripwire = install_fork_parent_cuda_tripwire(require_fresh_process=True)
        snapshot = (root / "code/configs/gui_owl_1_5_8b_snapshot.json").resolve()
        stage = "validate_fresh_topology_envelope"
        receipt_path, envelope, receipt_sha256 = _load_gpu_topology_smoke_receipt(
            args.gpu_topology_smoke_receipt,
            repository_root=root,
            expected_sha256=args.gpu_topology_smoke_receipt_sha256,
        )
        runner_freeze = authorization["runner_freeze"]
        (
            receipt_body,
            parent_receipt_file_sha256,
            challenge_sha256,
            challenge_responses_sha256,
        ) = _unwrap_fresh_topology_envelope(
            envelope,
            execution_b_commit=args.execution_b_git_commit,
            topology_nonce=runner_freeze["topology_receipt_nonce"],
            expected_devices=devices,
            expected_gpu_uuids=gpu_uuids,
            forbidden_parent_receipt_sha256=contract.data["failed_attempt"][
                "topology_receipt_sha256"
            ],
        )
        validated_topology = validate_gpu_topology_smoke_receipt(
            receipt_body,
            contract=parent_contract,
            expected_devices=devices,
            expected_gpu_uuids=gpu_uuids,
            expected_model_dir=model_dir,
            expected_snapshot_manifest=snapshot,
        )
        topology_binding = {
            "path": str(receipt_path),
            "sha256": receipt_sha256,
            "execution_b_commit": args.execution_b_git_commit,
            "topology_nonce": runner_freeze["topology_receipt_nonce"],
            "parent_receipt_file_sha256": parent_receipt_file_sha256,
            "challenge_sha256": challenge_sha256,
            "challenge_responses_sha256": challenge_responses_sha256,
            "body": dict(validated_topology),
        }

        stage = "read_hf_token_and_adopt_payload"
        token = read_hf_token_file(args.hf_token_file)
        publisher = HuggingFaceConfirmPublisher(cache_dir=cache_dir, token=token)
        payload_files = _download_expected_payload(
            contract=contract, publisher=publisher
        )
        adoption = publisher.adopt_existing_payload(
            expected_files=payload_files,
            expected_base_commit=EXPECTED_BASE_COMMIT,
            expected_base_title=contract.payload["base_commit_title"],
            expected_payload_commit=EXPECTED_PAYLOAD_COMMIT,
        )
        adopted_receipt = PayloadPublicationReceipt(
            base_commit=str(adoption["base_commit"]),
            payload_commit=str(adoption["payload_commit"]),
            payload_inventory_sha256=str(adoption["payload_inventory_sha256"]),
        )

        stage = "replay_formal_parent"
        payloads = _formal_parent_view(
            repository_root=root,
            generator_source_root=generator_root,
            parent_root=parent_root,
            ocr_model_dir=ocr_model_dir,
            ocr_wheel_dir=ocr_wheel_dir,
        )
        stage = "isolated_model_snapshot_verification"
        verified_snapshot, model_identity = isolated_model_snapshot_identity(
            repository_root=root,
            parent_contract_path=parent_contract_path,
            model_dir=model_dir,
            timeout_seconds=contract.continuation[
                "model_verification_timeout_seconds"
            ],
        )
        if verified_snapshot != snapshot:
            raise RuntimeError("isolated model snapshot path differs from canonical path")
        stage = "restoration_and_report"
        run_contract = _run_contract(
            contract=contract,
            parent_contract=parent_contract,
            authorization=authorization,
            parent=payloads,
            model_identity=model_identity,
            topology_binding=topology_binding,
            args=args,
            devices=devices,
            gpu_uuids=gpu_uuids,
        )
        def release_tripwire_after_restoration() -> None:
            nonlocal tripwire
            if tripwire is None:
                raise RuntimeError("CUDA parent tripwire was already released")
            release_fork_parent_cuda_tripwire(tripwire)
            tripwire = None

        result = execute_independent_confirm_continuation(
            contract=contract,
            parent_contract=parent_contract,
            payloads=payloads,
            adopted_payload=adopted_receipt,
            payload_files=payload_files,
            run_contract=run_contract,
            output_dir=output_dir,
            output_store=output,
            persist_failure_on_error=False,
            after_restoration=release_tripwire_after_restoration,
            publisher=publisher,
            restoration_phase_factory=lambda bundle: make_real_restoration_phase(
                payloads=payloads,
                bundle=bundle,
                model_dir=model_dir,
                snapshot_manifest=verified_snapshot,
                devices=devices,
                gpu_uuids=gpu_uuids,
                phase_timeout_seconds=contract.continuation[
                    "restoration_phase_timeout_seconds"
                ],
                worker_termination_grace_seconds=contract.continuation[
                    "worker_termination_grace_seconds"
                ],
                require_parent_cuda_tripwire=True,
            ),
        )
        if tripwire is not None:
            raise RuntimeError("CUDA parent tripwire survived restoration completion")
        return result.completion
    except BaseException as error:
        publication_audit = (
            dict(publisher.report_publication_audit())
            if publisher is not None
            else {
                "status": "REPORT_PUBLICATION_NOT_REACHED",
                "remote_mutation_count": 0,
                "minimum_remote_mutation_count": 0,
            }
        )
        failure = {
            "schema_version": "1.0.0",
            "protocol_id": contract.data["protocol_id"],
            "status": contract.data["output_contract"]["failure_status"],
            "failed_stage": stage,
            "exception_type": error.__class__.__name__,
            "message": str(error),
            "old_attempt_status": contract.data["failed_attempt"]["status"],
            "existing_payload_commit": EXPECTED_PAYLOAD_COMMIT,
            "report_publication_audit": publication_audit,
            "remote_mutation_count": publication_audit.get(
                "remote_mutation_count"
            ),
            "minimum_remote_mutation_count": publication_audit.get(
                "minimum_remote_mutation_count"
            ),
            "runtime_continuation_count": 1,
            "scientific_retry_count": 0,
            "selector_checkpoint_load_count": 0,
            "independent_scorer_call_count": 0,
            "policy_vision_call_count": 0,
            "payload_publish_call_count": 0,
            "sealed_androidworld_test_access_count": 0,
            "top_up_count": 0,
            "filter_count": 0,
        }
        try:
            output.write_failure(failure)
        except Exception as evidence_error:
            error.add_note(
                "failed to persist continuation terminal: "
                f"{evidence_error.__class__.__name__}: {evidence_error}"
            )
        raise
    finally:
        if tripwire is not None:
            try:
                release_fork_parent_cuda_tripwire(tripwire)
            except Exception as release_error:
                if not output.has_failure() and not output.has_completion():
                    try:
                        output.write_failure(
                            {
                                "schema_version": "1.0.0",
                                "protocol_id": contract.data["protocol_id"],
                                "status": contract.data["output_contract"][
                                    "failure_status"
                                ],
                                "failed_stage": "release_cuda_parent_tripwire",
                                "exception_type": release_error.__class__.__name__,
                                "message": str(release_error),
                                "runtime_continuation_count": 1,
                                "scientific_retry_count": 0,
                            }
                        )
                    except Exception:
                        pass


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
    parser.add_argument("--gpu-topology-smoke-receipt", type=Path, required=True)
    parser.add_argument("--gpu-topology-smoke-receipt-sha256", required=True)
    parser.add_argument("--host-alias", choices=("hyper00", "hyper01"), required=True)
    parser.add_argument("--host-hostname", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--gpu-uuids", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    effective = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(effective)
    args.execution_argv = effective
    result = execute(args)
    print(json.dumps(dict(result), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
