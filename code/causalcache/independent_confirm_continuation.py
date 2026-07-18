"""Restoration-only continuation for the sealed independent confirm-20 payload."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from causalcache.independent_confirm_artifact import (
    FEATURE_STATE_PATH,
    INDEPENDENT_DECISION_PATH,
    PAYLOAD_TARGETS,
    REPORT_TARGETS,
    SCORE_PATHS,
    SELECTION_PATHS,
    LabelBlindPayloadSeal,
    PayloadPublicationReceipt,
    build_report_files,
    canonical_json_bytes,
    feature_states_jsonl_bytes,
    read_feature_states_jsonl,
    read_score_records_jsonl,
    read_selection_artifact,
    score_records_jsonl_bytes,
    selection_artifact_bytes,
    seal_label_blind_payload,
)
from causalcache.independent_confirm_continuation_contract import (
    EXPECTED_BASE_COMMIT,
    EXPECTED_PAYLOAD_COMMIT,
    EXPECTED_PAYLOAD_INVENTORY_SHA256,
    PROTOCOL_ID,
    RUN_CONTRACT_PROTOCOL_ID,
    IndependentConfirmContinuationContract,
)
from causalcache.independent_confirm_contract import IndependentConfirmContract
from causalcache.independent_confirm_data import (
    CONFIRM_SOURCE_IDS,
    CONFIRM_STATE_COUNT,
    Confirm20LabelBlindBundle,
    ValidatedConfirm20Payloads,
    build_confirm20_label_blind_bundle,
)
from causalcache.independent_confirm_execution import (
    DurableConfirmOutput,
    RestorationPhase,
    validate_restoration_phase,
)
from causalcache.independent_confirm_runner import (
    ConfirmWorkerResult,
    aggregate_independent_confirm_workers,
    authorize_adopted_payload_commit,
    confirm_worker_assignments,
    persist_and_seal_label_blind_payload,
    replay_independent_selection_bundle,
)


SCHEMA_VERSION = "1.0.0"
COMPLETION_STATUS = "COMPLETED_AND_PUBLISHED_INDEPENDENT_CONFIRM20_CONTINUATION_V1"
FAILURE_STATUS = "INVALID_INDEPENDENT_CONFIRM20_CONTINUATION_V1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ContinuationPayloadView:
    seal: LabelBlindPayloadSeal
    bundle: Confirm20LabelBlindBundle
    feature_states: tuple[Any, ...]
    independent: Any
    heuristics: Mapping[str, Mapping[str, tuple[int, ...]]]


@dataclass(frozen=True)
class ContinuationExecutionResult:
    completion: Mapping[str, Any]
    report: Mapping[str, Any]
    report_files: Mapping[str, bytes]


def validate_expected_payload_files(
    contract: IndependentConfirmContinuationContract,
    files: Mapping[str, bytes],
) -> LabelBlindPayloadSeal:
    expected = {
        row["path"]: (row["size_bytes"], row["sha256"])
        for row in contract.payload["targets"]
    }
    if set(files) != set(PAYLOAD_TARGETS) or set(files) != set(expected):
        raise ValueError("continuation payload file inventory drifted")
    copied = {path: bytes(files[path]) for path in files}
    for path, (size_bytes, digest) in expected.items():
        if (
            len(copied[path]) != size_bytes
            or hashlib.sha256(copied[path]).hexdigest() != digest
        ):
            raise ValueError(f"continuation payload bytes drifted: {path}")
    seal = seal_label_blind_payload(copied)
    if (
        seal.inventory_sha256 != EXPECTED_PAYLOAD_INVENTORY_SHA256
        or seal.inventory_sha256 != contract.payload["payload_inventory_sha256"]
    ):
        raise ValueError("continuation full eight-file payload inventory drifted")
    return seal


def replay_payload_against_parent(
    *,
    contract: IndependentConfirmContinuationContract,
    payloads: ValidatedConfirm20Payloads,
    payload_files: Mapping[str, bytes],
) -> ContinuationPayloadView:
    """Bind the sealed files to the same deterministic parent without reselection."""
    seal = validate_expected_payload_files(contract, payload_files)
    regenerated = build_confirm20_label_blind_bundle(payloads)
    feature_states = read_feature_states_jsonl(seal.files[FEATURE_STATE_PATH])
    if feature_states_jsonl_bytes(regenerated.feature_states) != seal.files[FEATURE_STATE_PATH]:
        raise ValueError("sealed feature states differ from deterministic parent replay")

    dynamic = read_selection_artifact(
        seal.files[SELECTION_PATHS["dynamic_recent"]], name="dynamic_recent"
    )
    ocr = read_selection_artifact(
        seal.files[SELECTION_PATHS["ocr_rgb_v2"]], name="ocr_rgb_v2"
    )
    policy_vision = read_selection_artifact(
        seal.files[SELECTION_PATHS["policy_vision_v3"]], name="policy_vision_v3"
    )
    ocr_scores = read_score_records_jsonl(
        seal.files[SCORE_PATHS["ocr_rgb_v2"]], name="ocr_rgb_v2"
    )
    if (
        selection_artifact_bytes("dynamic_recent", regenerated.dynamic_recent)
        != seal.files[SELECTION_PATHS["dynamic_recent"]]
        or selection_artifact_bytes("ocr_rgb_v2", regenerated.ocr_rgb_v2)
        != seal.files[SELECTION_PATHS["ocr_rgb_v2"]]
        or score_records_jsonl_bytes("ocr_rgb_v2", regenerated.ocr_rgb_score_records)
        != seal.files[SCORE_PATHS["ocr_rgb_v2"]]
        or tuple(ocr_scores) != tuple(
            read_score_records_jsonl(
                score_records_jsonl_bytes(
                    "ocr_rgb_v2", regenerated.ocr_rgb_score_records
                ),
                name="ocr_rgb_v2",
            )
        )
        or dict(dynamic) != dict(regenerated.dynamic_recent)
        or dict(ocr) != dict(regenerated.ocr_rgb_v2)
    ):
        raise ValueError("sealed deterministic heuristic bytes differ from parent replay")

    independent = replay_independent_selection_bundle(
        seal.files[INDEPENDENT_DECISION_PATH]
    )
    return ContinuationPayloadView(
        seal=seal,
        bundle=regenerated,
        feature_states=tuple(feature_states),
        independent=independent,
        heuristics=MappingProxyType(
            {
                "dynamic_recent": dynamic,
                "ocr_rgb_v2": ocr,
                "policy_vision_v3": policy_vision,
            }
        ),
    )


def validate_continuation_run_contract(
    run_contract: Mapping[str, Any],
    *,
    contract: IndependentConfirmContinuationContract,
    parent_contract: IndependentConfirmContract,
) -> Mapping[str, Any]:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "contract_sha256",
        "source_a_commit",
        "execution_b_commit",
        "runner_freeze",
        "parent_scientific_contract_sha256",
        "old_failure",
        "adopted_payload",
        "parent_artifact",
        "policy_snapshot",
        "gpu_topology_smoke_receipt",
        "runtime",
        "operation_boundary",
        "execution_argv_sha256",
    }
    if not isinstance(run_contract, Mapping) or set(run_contract) != expected_keys:
        raise ValueError("continuation run contract schema drifted")
    source_a = run_contract.get("source_a_commit")
    execution_b = run_contract.get("execution_b_commit")
    runner_freeze = run_contract.get("runner_freeze")
    old_failure = run_contract.get("old_failure")
    adopted = run_contract.get("adopted_payload")
    parent_artifact = run_contract.get("parent_artifact")
    policy_snapshot = run_contract.get("policy_snapshot")
    topology = run_contract.get("gpu_topology_smoke_receipt")
    boundary = run_contract.get("operation_boundary")
    runtime = run_contract.get("runtime")
    execution_argv = runtime.get("execution_argv") if isinstance(runtime, Mapping) else None
    devices = runtime.get("devices") if isinstance(runtime, Mapping) else None
    gpu_uuids = runtime.get("gpu_uuids") if isinstance(runtime, Mapping) else None
    parent_restoration = parent_contract.data["restoration_contract"]
    if (
        run_contract.get("schema_version") != SCHEMA_VERSION
        or run_contract.get("protocol_id") != RUN_CONTRACT_PROTOCOL_ID
        or run_contract.get("contract_sha256") != contract.sha256
        or _COMMIT.fullmatch(str(source_a)) is None
        or _COMMIT.fullmatch(str(execution_b)) is None
        or not isinstance(runner_freeze, Mapping)
        or runner_freeze.get("source_a_commit") != source_a
        or runner_freeze.get("source_a_remote_main") != source_a
        or runner_freeze.get("adopt_existing_payload_authorized") is not True
        or runner_freeze.get("selector_reexecution_authorized") is not False
        or run_contract.get("parent_scientific_contract_sha256")
        != parent_contract.sha256
        or not isinstance(old_failure, Mapping)
        or dict(old_failure)
        != {
            "status": "INVALID_INDEPENDENT_CONFIRM20_V1_CUDA_FORK_BOUNDARY",
            "evidence_commit": contract.data["failed_attempt"]["evidence_commit"],
            "failure_sha256": contract.data["failed_attempt"]["failure_sha256"],
            "reference_policy_output_count": 0,
            "restoration_output_count": 0,
        }
        or not isinstance(adopted, Mapping)
        or dict(adopted)
        != {
            "base_commit": EXPECTED_BASE_COMMIT,
            "payload_commit": EXPECTED_PAYLOAD_COMMIT,
            "payload_inventory_sha256": EXPECTED_PAYLOAD_INVENTORY_SHA256,
            "fresh_replay": True,
            "payload_mutation_count": 0,
            "report_target_count_before_continuation": 0,
            "tag_count_before_continuation": 0,
        }
        or not isinstance(parent_artifact, Mapping)
        or dict(parent_artifact)
        != {
            "artifact_tree_sha256": parent_contract.data["confirm_input"][
                "artifact_tree_sha256"
            ],
            "source_ids": list(parent_contract.data["confirm_geometry"]["source_ids"]),
            "state_count": CONFIRM_STATE_COUNT,
            "policy_output_present": False,
            "restoration_output_present": False,
        }
        or not isinstance(policy_snapshot, Mapping)
        or policy_snapshot.get("model_repo") != parent_restoration["policy_repo"]
        or policy_snapshot.get("model_revision")
        != parent_restoration["policy_revision"]
        or policy_snapshot.get("snapshot_manifest_sha256")
        != parent_restoration["snapshot_manifest_sha256"]
        or policy_snapshot.get("verified_model_file_count") != 14
        or policy_snapshot.get("verified_model_total_bytes") != 17_545_907_171
        or not isinstance(topology, Mapping)
        or set(topology)
        != {
            "path",
            "sha256",
            "execution_b_commit",
            "topology_nonce",
            "parent_receipt_file_sha256",
            "challenge_sha256",
            "challenge_responses_sha256",
            "body",
        }
        or _SHA256.fullmatch(str(topology.get("sha256"))) is None
        or topology.get("execution_b_commit") != execution_b
        or topology.get("topology_nonce")
        != runner_freeze.get("topology_receipt_nonce")
        or _SHA256.fullmatch(str(topology.get("parent_receipt_file_sha256")))
        is None
        or _SHA256.fullmatch(str(topology.get("challenge_sha256"))) is None
        or _SHA256.fullmatch(str(topology.get("challenge_responses_sha256")))
        is None
        or not isinstance(topology.get("body"), Mapping)
        or hashlib.sha256(
            canonical_json_bytes(dict(topology["body"])) + b"\n"
        ).hexdigest()
        != topology.get("parent_receipt_file_sha256")
        or topology["body"].get("status") != "PASS"
        or not isinstance(boundary, Mapping)
        or dict(boundary)
        != {
            "runtime_continuation_count": 1,
            "scientific_retry_count": 0,
            "top_up_count": 0,
            "filter_count": 0,
            "selector_checkpoint_load_count": 0,
            "independent_scorer_call_count": 0,
            "policy_vision_call_count": 0,
            "payload_publish_call_count": 0,
            "sealed_androidworld_test_access_count": 0,
        }
        or not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "host_alias",
            "host_hostname",
            "container_id",
            "container_image_digest",
            "devices",
            "gpu_uuids",
            "worker_count",
            "states_per_worker",
            "model_verification_start_method",
            "restoration_start_method",
            "parent_cuda_clean_guard_required",
            "parent_cuda_tripwire_required",
            "fresh_exec_parent_required",
            "restoration_phase_timeout_seconds",
            "worker_termination_grace_seconds",
            "execution_argv",
        }
        or not isinstance(devices, list)
        or len(devices) != 4
        or len(set(devices)) != 4
        or not isinstance(gpu_uuids, list)
        or len(gpu_uuids) != 4
        or len(set(gpu_uuids)) != 4
        or runtime.get("worker_count") != 4
        or runtime.get("states_per_worker") != 5
        or runtime.get("model_verification_start_method") != "spawn"
        or runtime.get("restoration_start_method") != "fork"
        or runtime.get("parent_cuda_clean_guard_required") is not True
        or runtime.get("parent_cuda_tripwire_required") is not True
        or runtime.get("fresh_exec_parent_required") is not True
        or runtime.get("container_image_digest")
        != contract.runtime["container_image_id"]
        or not isinstance(execution_argv, list)
        or any(not isinstance(item, str) for item in execution_argv)
        or _SHA256.fullmatch(str(run_contract.get("execution_argv_sha256"))) is None
        or hashlib.sha256(canonical_json_bytes(execution_argv)).hexdigest()
        != run_contract["execution_argv_sha256"]
    ):
        raise ValueError("continuation run contract identity or zero-operation boundary drifted")
    frozen = json.loads(canonical_json_bytes(dict(run_contract)))
    if not isinstance(frozen, dict):
        raise AssertionError("canonical continuation run contract is not an object")
    return MappingProxyType(frozen)


def _ordered_state_records(
    worker_results: Sequence[ConfirmWorkerResult],
    *,
    selection_sha256: str,
    run_contract_sha256: str,
    payload_commit: str,
) -> tuple[Mapping[str, Any], ...]:
    records = [record for result in worker_results for record in result.state_records]
    records.sort(key=lambda item: int(item["state"]["ordinal"]))
    expected = [
        (ordinal, source_id, f"{source_id}:decision_step:006")
        for ordinal, source_id in enumerate(CONFIRM_SOURCE_IDS)
    ]
    observed = [
        (
            record["state"]["ordinal"],
            record["state"]["source_id"],
            record["state"]["state_id"],
        )
        for record in records
    ]
    if (
        observed != expected
        or any(record.get("selection_sha256") != selection_sha256 for record in records)
        or any(
            record.get("run_contract_sha256") != run_contract_sha256
            for record in records
        )
        or any(record.get("payload_commit") != payload_commit for record in records)
    ):
        raise ValueError("continuation raw state record binding drifted")
    return tuple(records)


def execute_independent_confirm_continuation(
    *,
    contract: IndependentConfirmContinuationContract,
    parent_contract: IndependentConfirmContract,
    payloads: ValidatedConfirm20Payloads,
    adopted_payload: PayloadPublicationReceipt,
    payload_files: Mapping[str, bytes],
    run_contract: Mapping[str, Any],
    output_dir: str | Path,
    publisher: Any,
    restoration_phase_factory: Callable[[Confirm20LabelBlindBundle], RestorationPhase],
    output_store: DurableConfirmOutput | None = None,
    persist_failure_on_error: bool = True,
    after_restoration: Callable[[], None] | None = None,
) -> ContinuationExecutionResult:
    """Adopt exact selector bytes, run restoration, and publish only the report."""
    frozen_run = validate_continuation_run_contract(
        run_contract, contract=contract, parent_contract=parent_contract
    )
    if (
        not isinstance(adopted_payload, PayloadPublicationReceipt)
        or adopted_payload.base_commit != EXPECTED_BASE_COMMIT
        or adopted_payload.payload_commit != EXPECTED_PAYLOAD_COMMIT
        or adopted_payload.payload_inventory_sha256
        != EXPECTED_PAYLOAD_INVENTORY_SHA256
    ):
        raise ValueError("continuation lacks the exact adopted payload receipt")
    output_path = Path(output_dir)
    output = DurableConfirmOutput(output_path) if output_store is None else output_store
    if type(persist_failure_on_error) is not bool or (
        output_store is not None and output.root != output_path
    ):
        raise ValueError("continuation durable output binding drifted")
    report_files: Mapping[str, bytes] = {}
    run_contract_sha256 = hashlib.sha256(
        canonical_json_bytes(dict(frozen_run))
    ).hexdigest()
    try:
        view = replay_payload_against_parent(
            contract=contract,
            payloads=payloads,
            payload_files=payload_files,
        )
        if view.seal.inventory_sha256 != adopted_payload.payload_inventory_sha256:
            raise ValueError("local payload replay differs from remote adoption receipt")
        restoration_phase = restoration_phase_factory(view.bundle)
        if not callable(restoration_phase):
            raise TypeError(
                "continuation restoration phase factory returned a non-callable"
            )
        local_seal = persist_and_seal_label_blind_payload(
            view.seal.files,
            independent_selections=view.independent,
            independent_selection_path=INDEPENDENT_DECISION_PATH,
            persist_fn=output.persist_label_blind,
        )
        receipt = authorize_adopted_payload_commit(
            local_seal,
            adopted_payload.payload_commit,
            publisher.verify_payload,
        )
        assignments = confirm_worker_assignments(view.bundle.work_items)
        restoration = validate_restoration_phase(
            restoration_phase(assignments, receipt, run_contract_sha256),
            assignments=assignments,
            expected_devices=frozen_run["runtime"]["devices"],
            expected_gpu_uuids=frozen_run["runtime"]["gpu_uuids"],
        )
        if after_restoration is not None:
            after_restoration()
        aggregate = aggregate_independent_confirm_workers(
            view.feature_states,
            assignments=assignments,
            worker_results=restoration.worker_results,
            independent_selections=view.independent,
            heuristic_selections=view.heuristics,
        )
        execution = aggregate.get("execution")
        if not isinstance(execution, Mapping):
            raise ValueError("continuation aggregate lacks execution evidence")
        fixed_report = {
            key: value for key, value in aggregate.items() if key != "execution"
        }
        raw_records = _ordered_state_records(
            restoration.worker_results,
            selection_sha256=view.independent.selection_sha256,
            run_contract_sha256=run_contract_sha256,
            payload_commit=receipt.payload_commit,
        )
        report_files = build_report_files(
            state_records=raw_records,
            fixed_report=fixed_report,
            run_manifest={
                "source_git_commit": frozen_run["execution_b_commit"],
                "contract_sha256": contract.sha256,
                "runtime_metadata": {
                    "run_contract_sha256": run_contract_sha256,
                    "run_contract": dict(frozen_run),
                    "runtime_continuation_count": 1,
                    "restoration_workers": [
                        dict(item) for item in restoration.worker_metadata
                    ],
                },
                "operation_counts": dict(execution["operation_counts"]),
            },
            payload_seal=view.seal,
            payload_commit=receipt.payload_commit,
        )
        if set(report_files) != set(REPORT_TARGETS):
            raise RuntimeError("continuation report is not the frozen four files")
        output.persist_report(report_files)
        final = publisher.publish_report(
            report_files, payload_commit=receipt.payload_commit
        )
        if (
            final.get("base_commit") != EXPECTED_BASE_COMMIT
            or final.get("payload_commit") != EXPECTED_PAYLOAD_COMMIT
            or not isinstance(final.get("report_commit"), str)
            or _COMMIT.fullmatch(final["report_commit"]) is None
            or not isinstance(final.get("annotated_tag_object"), str)
            or _COMMIT.fullmatch(final["annotated_tag_object"]) is None
            or final.get("created_tag_count") != 1
            or final.get("byte_identical_fresh_replay") is not True
        ):
            raise ValueError("continuation final publication receipt is malformed")
        completion = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": COMPLETION_STATUS,
            "old_attempt_status": "INVALID_INDEPENDENT_CONFIRM20_V1_CUDA_FORK_BOUNDARY",
            "confirm_outcome": fixed_report.get("status"),
            "confirm_go": fixed_report.get("go"),
            "base_commit": EXPECTED_BASE_COMMIT,
            "payload_commit": EXPECTED_PAYLOAD_COMMIT,
            "report_commit": final["report_commit"],
            "report_parent_commit": EXPECTED_PAYLOAD_COMMIT,
            "annotated_tag_object": final["annotated_tag_object"],
            "created_tag_count": 1,
            "byte_identical_fresh_replay": True,
            "fixed_state_denominator": CONFIRM_STATE_COUNT,
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
        output.write_completion(completion)
        return ContinuationExecutionResult(
            completion=MappingProxyType(completion),
            report=MappingProxyType(fixed_report),
            report_files=MappingProxyType(dict(report_files)),
        )
    except BaseException as error:
        audit_fn = getattr(publisher, "report_publication_audit", None)
        publication_audit = (
            dict(audit_fn())
            if callable(audit_fn)
            else {
                "status": "REPORT_PUBLICATION_AUDIT_UNAVAILABLE",
                "remote_mutation_count": None,
                "minimum_remote_mutation_count": 0,
            }
        )
        failure = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": FAILURE_STATUS,
            "old_attempt_status": "INVALID_INDEPENDENT_CONFIRM20_V1_CUDA_FORK_BOUNDARY",
            "exception_type": error.__class__.__name__,
            "message": str(error),
            "existing_payload_commit": EXPECTED_PAYLOAD_COMMIT,
            "report_publication_audit": publication_audit,
            "remote_mutation_count": publication_audit.get(
                "remote_mutation_count"
            ),
            "minimum_remote_mutation_count": publication_audit.get(
                "minimum_remote_mutation_count"
            ),
            "report_locally_persisted": bool(report_files),
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
        if persist_failure_on_error:
            try:
                output.write_failure(failure)
            except Exception as evidence_error:
                error.add_note(
                    "failed to persist continuation terminal: "
                    f"{evidence_error.__class__.__name__}: {evidence_error}"
                )
        raise


__all__ = [
    "COMPLETION_STATUS",
    "ContinuationExecutionResult",
    "ContinuationPayloadView",
    "FAILURE_STATUS",
    "execute_independent_confirm_continuation",
    "replay_payload_against_parent",
    "validate_continuation_run_contract",
    "validate_expected_payload_files",
]
