"""CPU-only exact-byte replay for the policy-vision v3 validation repair."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from causalcache.restoration_v2_2_policy_vision import load_identity_witness
from causalcache.restoration_v2_2_policy_vision_v3_contract import (
    RestorationV22PolicyVisionV3Contract,
)
from causalcache.restoration_v2_2_policy_vision_v3_validation_repair_contract import (
    ARTIFACT_GIT_COMMIT,
    EXPECTED_OUTPUT_FILES,
    FORMAL_ATTEMPT_LEDGER_PATH,
    FORMAL_COMPLETION_SEAL_PATH,
    PRODUCER_CONFIG_SHA256,
    PRODUCER_PROTOCOL_ID,
    PRODUCER_RUN_STATUS,
    PRODUCER_SOURCE_GIT_COMMIT,
    PROTOCOL_ID,
    RUN_STATUS,
    PolicyVisionV3ValidationRepairContract,
    canonical_json_bytes,
    pretty_json_bytes,
    sha256_bytes,
    validate_exact_artifact_directory,
)
from scripts.run_restoration_v2_2_policy_vision_baseline import (
    _expected_files_from_features,
    _feature_record_from_evaluated_row,
    _read_recorded_rows,
    _validate_labels_archive,
)
from scripts.run_restoration_v2_2_policy_vision_baseline_v3 import (
    V3_RUNNER_PROTOCOL,
)


EVALUATED_STATE_KEYS = frozenset(
    {
        "budget_event_capacity",
        "candidate_event_step_ids",
        "decision_step_id",
        "index",
        "role",
        "state_id",
        "trajectory_id",
    }
)
FEATURE_STATE_KEYS = ("index", "role", "trajectory_id", "state_id")
EXPECTED_ROLES = frozenset({"v2_label_train", "v2_development"})
FORBIDDEN_CPU_REPAIR_RUNTIME_MODULES = ("torch", "transformers", "PIL")
COMPLETION_SEAL_STATUS = (
    "SEALED_POLICY_VISION_V3_VALIDATION_REPAIR_V1_COMPLETION"
)
ZERO_CPU_REPAIR_OPERATIONS = {
    "nvidia_device_query_count": 0,
    "cuda_operation_count": 0,
    "model_load_count": 0,
    "image_decode_count": 0,
    "image_processor_batch_count": 0,
    "policy_vision_feature_forward_count": 0,
    "policy_forward_count": 0,
    "language_model_forward_count": 0,
    "lm_head_forward_count": 0,
    "generation_count": 0,
    "teacher_forward_count": 0,
    "kl_measurement_count": 0,
    "restoration_attribution_count": 0,
    "restoration_label_generation_count": 0,
    "gate_model_forward_count": 0,
    "gate_training_example_count": 0,
    "matched_nll_evaluation_count": 0,
    "closed_loop_episode_count": 0,
    "confirm_state_access_count": 0,
    "sealed_test_state_access_count": 0,
    "producer_artifact_write_count": 0,
    "producer_attempt_ledger_write_count": 0,
}


@dataclass(frozen=True)
class ReplayResult:
    rebuilt_files: Mapping[str, bytes]
    producer_files_before: Mapping[str, bytes]
    producer_files_after: Mapping[str, bytes]
    feature_record_count: int
    candidate_score_count: int


@dataclass(frozen=True)
class RepairLedgerSnapshot:
    payload: bytes
    record: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True)
class CompletionSealSnapshot:
    payload: bytes
    record: Mapping[str, Any]
    sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_no_nvidia_runtime() -> dict[str, Any]:
    return {
        "nvidia_device_nodes": [],
        "nvidia_device_node_count": 0,
        "nvidia_smi_invocation_count": 0,
        "cuda_runtime_import_count": 0,
        "forbidden_runtime_modules_checked": list(
            FORBIDDEN_CPU_REPAIR_RUNTIME_MODULES
        ),
        "forbidden_runtime_modules_imported": [],
    }


def project_evaluated_state_to_feature_state(
    state: Any,
    *,
    projection_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the sole repair: exact seven-key evaluated state to four keys."""
    if not isinstance(state, Mapping) or set(state) != EVALUATED_STATE_KEYS:
        raise ValueError("recorded evaluated-state schema drifted")
    expected_budget = 2
    expected_candidates = [1, 2, 3, 4]
    expected_step = 6
    expected_output_keys = FEATURE_STATE_KEYS
    if projection_contract is not None:
        if (
            projection_contract.get("evaluated_state_keys_exact")
            != sorted(EVALUATED_STATE_KEYS)
            or projection_contract.get("feature_state_keys_exact_in_output_order")
            != list(FEATURE_STATE_KEYS)
        ):
            raise ValueError("frozen projection contract drifted")
        expected_budget = projection_contract["budget_event_capacity"]
        expected_candidates = projection_contract["candidate_event_step_ids"]
        expected_step = projection_contract["decision_step_id"]
        expected_output_keys = tuple(
            projection_contract["feature_state_keys_exact_in_output_order"]
        )
    index = state.get("index")
    role = state.get("role")
    trajectory_id = state.get("trajectory_id")
    state_id = state.get("state_id")
    if (
        state.get("budget_event_capacity") != expected_budget
        or state.get("candidate_event_step_ids") != expected_candidates
        or state.get("decision_step_id") != expected_step
        or type(index) is not int
        or index < 0
        or role not in EXPECTED_ROLES
        or not isinstance(trajectory_id, str)
        or not trajectory_id
        or state_id != f"{trajectory_id}:decision_step:006"
        or expected_output_keys != FEATURE_STATE_KEYS
    ):
        raise ValueError("recorded evaluated-state values drifted")
    return {key: state[key] for key in FEATURE_STATE_KEYS}


def feature_records_from_evaluated_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    projection_contract: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    if len(rows) != 15:
        raise ValueError("validation repair requires exactly 15 evaluated rows")
    records = []
    for row in rows:
        state = row.get("state")
        projected = project_evaluated_state_to_feature_state(
            state,
            projection_contract=projection_contract,
        )
        record = _feature_record_from_evaluated_row(row)
        if record.get("state") != dict(state):
            raise ValueError("frozen v3 reconstructor changed the evaluated state")
        record["state"] = projected
        records.append(record)
    if sum(len(record["candidate_scores"]) for record in records) != 60:
        raise ValueError("validation repair candidate-score denominator drifted")
    return tuple(records)


def _artifact_snapshot(
    contract: PolicyVisionV3ValidationRepairContract,
) -> dict[str, bytes]:
    return validate_exact_artifact_directory(
        contract.artifact_directory,
        contract.data["producer_artifact"]["exact_files"],
    )


def _regular_bound_external_file(
    path: Path,
    binding: Mapping[str, Any],
    *,
    name: str,
    required_mode: int | None = None,
) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    if required_mode is not None and stat.S_IMODE(metadata.st_mode) != required_mode:
        raise ValueError(f"{name} mode drifted")
    payload = path.read_bytes()
    if (
        len(payload) != binding.get("size_bytes")
        or sha256_bytes(payload) != binding.get("sha256")
    ):
        raise ValueError(f"{name} bytes drifted")
    return payload


def validate_producer_attempt_ledger(
    path: Path,
    contract: PolicyVisionV3ValidationRepairContract,
) -> bytes:
    binding = contract.data["failure_evidence"]["producer_gpu_attempt_ledger"]
    if path.resolve() != Path(binding["canonical_container_path"]).resolve():
        raise ValueError("producer GPU attempt ledger path drifted")
    return _regular_bound_external_file(
        path,
        binding,
        name="producer GPU attempt ledger",
        required_mode=0o600,
    )


def validate_no_nvidia_devices() -> dict[str, Any]:
    imported_forbidden_modules = sorted(
        name for name in FORBIDDEN_CPU_REPAIR_RUNTIME_MODULES if name in sys.modules
    )
    if imported_forbidden_modules:
        raise ValueError(
            "formal CPU validation repair imported a forbidden runtime module"
        )
    device_nodes = sorted(
        {
            str(path)
            for pattern in (
                "/dev/nvidia[0-9]*",
                "/dev/nvidiactl",
                "/dev/nvidia-uvm",
                "/dev/nvidia-uvm-tools",
                "/dev/nvidia-modeset",
                "/dev/nvidia-caps/*",
            )
            for path in Path("/").glob(pattern.lstrip("/"))
            if path.exists() or path.is_symlink()
        }
    )
    if device_nodes:
        raise ValueError("formal CPU validation repair exposes NVIDIA device nodes")
    result = expected_no_nvidia_runtime()
    result["nvidia_device_nodes"] = device_nodes
    result["forbidden_runtime_modules_imported"] = imported_forbidden_modules
    return result


def replay_exact_producer_artifact(
    *,
    contract: PolicyVisionV3ValidationRepairContract,
    labels_archive: Path,
) -> ReplayResult:
    before = _artifact_snapshot(contract)
    producer_contract_path = (
        contract.repository_root / contract.data["producer"]["v3_contract"]["path"]
    )
    producer_contract = RestorationV22PolicyVisionV3Contract.load(
        producer_contract_path,
        repository_root=contract.repository_root,
    )
    if producer_contract.sha256 != PRODUCER_CONFIG_SHA256:
        raise ValueError("loaded producer contract identity drifted")
    _validate_labels_archive(producer_contract, labels_archive)
    rows = _read_recorded_rows(contract.artifact_directory)
    feature_records = feature_records_from_evaluated_rows(
        rows,
        projection_contract=contract.data["projection_contract"],
    )
    work_items, witness_by_state = load_identity_witness(producer_contract)
    summary = json.loads(before["summary.json"])
    if (
        summary.get("status") != PRODUCER_RUN_STATUS
        or summary.get("protocol_id") != PRODUCER_PROTOCOL_ID
        or summary.get("source_execution", {}).get("source_git_commit")
        != PRODUCER_SOURCE_GIT_COMMIT
    ):
        raise ValueError("producer summary identity drifted before replay")
    rebuilt = _expected_files_from_features(
        contract=producer_contract,
        labels_archive=labels_archive,
        source_commit=PRODUCER_SOURCE_GIT_COMMIT,
        work_items=work_items,
        witness_by_state=witness_by_state,
        feature_records=feature_records,
        execution=summary["execution"],
        protocol=V3_RUNNER_PROTOCOL,
    )
    if set(rebuilt) != set(before) or any(
        rebuilt[name] != before[name] for name in before
    ):
        raise ValueError("versioned CPU replay differs from producer exact-three bytes")
    after = _artifact_snapshot(contract)
    if before != after:
        raise ValueError("producer artifact bytes changed during CPU replay")
    return ReplayResult(
        rebuilt_files=rebuilt,
        producer_files_before=before,
        producer_files_after=after,
        feature_record_count=len(feature_records),
        candidate_score_count=sum(
            len(record["candidate_scores"]) for record in feature_records
        ),
    )


def _claim_exclusive_json_record(
    path: Path,
    record: Mapping[str, Any],
    *,
    name: str,
) -> tuple[bytes, str]:
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"{name} parent must preexist")
    payload = pretty_json_bytes(record)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"{name} write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if stat.S_IMODE(path.stat().st_mode) != 0o600 or path.read_bytes() != payload:
        raise ValueError(f"{name} write drifted")
    parent_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return payload, sha256_bytes(payload)


def claim_attempt_ledger(path: Path, record: Mapping[str, Any]) -> tuple[bytes, str]:
    return _claim_exclusive_json_record(
        path,
        record,
        name="validation-repair attempt ledger",
    )


def claim_completion_seal(path: Path, record: Mapping[str, Any]) -> tuple[bytes, str]:
    return _claim_exclusive_json_record(
        path,
        record,
        name="validation-repair completion seal",
    )


def read_repair_attempt_ledger(path: Path) -> RepairLedgerSnapshot:
    if path.resolve() != Path(FORMAL_ATTEMPT_LEDGER_PATH).resolve():
        raise ValueError("validation-repair attempt ledger path drifted")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError("validation-repair attempt ledger is missing") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ValueError("validation-repair attempt ledger type or mode drifted")
    payload = path.read_bytes()
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("validation-repair attempt ledger is not JSON") from error
    if pretty_json_bytes(record) != payload:
        raise ValueError("validation-repair attempt ledger bytes drifted")
    return RepairLedgerSnapshot(
        payload=payload,
        record=record,
        sha256=sha256_bytes(payload),
    )


def read_completion_seal(path: Path) -> CompletionSealSnapshot:
    if path.resolve() != Path(FORMAL_COMPLETION_SEAL_PATH).resolve():
        raise ValueError("validation-repair completion seal path drifted")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError("validation-repair completion seal is missing") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ValueError("validation-repair completion seal type or mode drifted")
    payload = path.read_bytes()
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("validation-repair completion seal is not JSON") from error
    if pretty_json_bytes(record) != payload:
        raise ValueError("validation-repair completion seal bytes drifted")
    return CompletionSealSnapshot(
        payload=payload,
        record=record,
        sha256=sha256_bytes(payload),
    )


def validate_repair_attempt_ledger(
    path: Path,
    *,
    summary: Mapping[str, Any],
) -> bytes:
    snapshot = read_repair_attempt_ledger(path)
    claim = summary.get("formal_attempt_claim")
    if (
        not isinstance(claim, Mapping)
        or set(claim) != {"record", "sha256"}
        or snapshot.record != claim["record"]
        or snapshot.sha256 != claim["sha256"]
    ):
        raise ValueError("validation-repair formal attempt claim drifted")
    return snapshot.payload


def build_runtime_identity(
    no_nvidia_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "device": "cpu",
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "working_directory": str(Path.cwd()),
        "no_nvidia_runtime": dict(no_nvidia_runtime),
    }


def build_runtime_record(
    *,
    runtime_identity: Mapping[str, Any],
    argv: Sequence[str],
    started_at_utc: str,
    finished_at_utc: str,
) -> dict[str, Any]:
    return {
        **dict(runtime_identity),
        "argv": list(argv),
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
    }


def validate_runtime_identity(
    identity: Any,
    *,
    actual_no_nvidia_runtime: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected_keys = {
        "device",
        "hostname",
        "python_version",
        "platform",
        "working_directory",
        "no_nvidia_runtime",
    }
    if not isinstance(identity, Mapping) or set(identity) != expected_keys:
        raise ValueError("validation-repair runtime identity schema drifted")
    for key in ("hostname", "python_version", "platform", "working_directory"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise ValueError(f"validation-repair runtime identity {key} drifted")
    if (
        identity.get("device") != "cpu"
        or identity.get("no_nvidia_runtime")
        != dict(actual_no_nvidia_runtime)
        or identity.get("no_nvidia_runtime") != expected_no_nvidia_runtime()
    ):
        raise ValueError("validation-repair runtime identity CPU binding drifted")
    return identity


def validate_runtime_record(
    runtime: Any,
    *,
    actual_no_nvidia_runtime: Mapping[str, Any],
    attempt_claim: Mapping[str, Any],
    expected_finished_at_utc: str,
) -> Mapping[str, Any]:
    expected_keys = {
        "device",
        "hostname",
        "python_version",
        "platform",
        "argv",
        "working_directory",
        "started_at_utc",
        "finished_at_utc",
        "no_nvidia_runtime",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != expected_keys:
        raise ValueError("validation-repair runtime schema drifted")
    identity = validate_runtime_identity(
        attempt_claim.get("runtime_identity"),
        actual_no_nvidia_runtime=actual_no_nvidia_runtime,
    )
    argv = runtime.get("argv")
    runtime_identity = {
        key: runtime.get(key)
        for key in (
            "device",
            "hostname",
            "python_version",
            "platform",
            "working_directory",
            "no_nvidia_runtime",
        )
    }
    if (
        runtime_identity != dict(identity)
        or not isinstance(argv, list)
        or not argv
        or any(not isinstance(value, str) or not value for value in argv)
        or argv != attempt_claim.get("argv")
        or runtime.get("started_at_utc") != attempt_claim.get("started_at_utc")
        or runtime.get("finished_at_utc") != expected_finished_at_utc
    ):
        raise ValueError("validation-repair runtime CPU or claim binding drifted")
    try:
        started = datetime.fromisoformat(str(runtime["started_at_utc"]))
        finished = datetime.fromisoformat(str(runtime["finished_at_utc"]))
    except ValueError as error:
        raise ValueError("validation-repair runtime timestamp drifted") from error
    if (
        started.tzinfo is None
        or finished.tzinfo is None
        or finished < started
    ):
        raise ValueError("validation-repair runtime UTC bracket drifted")
    return runtime


def _exact_output_file_bindings(
    files: Mapping[str, bytes],
) -> list[dict[str, Any]]:
    if set(files) != set(EXPECTED_OUTPUT_FILES):
        raise ValueError("validation-repair exact output inventory drifted")
    return [
        {
            "path": name,
            "size_bytes": len(files[name]),
            "sha256": sha256_bytes(files[name]),
        }
        for name in sorted(EXPECTED_OUTPUT_FILES)
    ]


def build_completion_seal_record(
    *,
    files: Mapping[str, bytes],
    repair_ledger: RepairLedgerSnapshot,
    finished_at_utc: str,
) -> dict[str, Any]:
    try:
        summary = json.loads(files["summary.json"])
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("completion seal requires the exact audit summary") from error
    validation_payload_sha256 = summary.get("validation_payload_sha256")
    if (
        not isinstance(validation_payload_sha256, str)
        or len(validation_payload_sha256) != 64
    ):
        raise ValueError("completion seal validation payload identity drifted")
    return {
        "schema_version": "1.0.0",
        "status": COMPLETION_SEAL_STATUS,
        "protocol_id": PROTOCOL_ID,
        "attempt_claim_sha256": repair_ledger.sha256,
        "finished_at_utc": finished_at_utc,
        "exact_output_files": _exact_output_file_bindings(files),
        "validation_payload_sha256": validation_payload_sha256,
    }


def validate_completion_seal_record(
    completion_seal: CompletionSealSnapshot,
    *,
    repair_ledger: RepairLedgerSnapshot,
) -> Mapping[str, Any]:
    record = completion_seal.record
    expected_keys = {
        "schema_version",
        "status",
        "protocol_id",
        "attempt_claim_sha256",
        "finished_at_utc",
        "exact_output_files",
        "validation_payload_sha256",
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_keys
        or pretty_json_bytes(record) != completion_seal.payload
        or sha256_bytes(completion_seal.payload) != completion_seal.sha256
    ):
        raise ValueError("validation-repair completion seal schema drifted")
    rows = record.get("exact_output_files")
    if (
        record.get("schema_version") != "1.0.0"
        or record.get("status") != COMPLETION_SEAL_STATUS
        or record.get("protocol_id") != PROTOCOL_ID
        or record.get("attempt_claim_sha256") != repair_ledger.sha256
        or not isinstance(record.get("finished_at_utc"), str)
        or not record["finished_at_utc"]
        or not isinstance(record.get("validation_payload_sha256"), str)
        or len(record["validation_payload_sha256"]) != 64
        or not isinstance(rows, list)
        or len(rows) != len(EXPECTED_OUTPUT_FILES)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"path", "size_bytes", "sha256"}
            for row in rows
        )
        or [row["path"] for row in rows] != sorted(EXPECTED_OUTPUT_FILES)
    ):
        raise ValueError("validation-repair completion seal values drifted")
    return record


def validate_completion_seal_output_bindings(
    record: Mapping[str, Any],
    *,
    files: Mapping[str, bytes],
    summary: Mapping[str, Any],
) -> None:
    if (
        record.get("exact_output_files") != _exact_output_file_bindings(files)
        or record.get("validation_payload_sha256")
        != summary.get("validation_payload_sha256")
    ):
        raise ValueError(
            "validation-repair completion seal exact output binding drifted"
        )


def _audit_readme(summary: Mapping[str, Any]) -> bytes:
    files = summary["exact_byte_replay"]["producer_files"]
    text = f"""# Policy-vision v3 validation repair v1

本记录是纯 CPU、只读的 validator repair audit。它只把 evaluated row 的七键 state 精确投影回
四键 feature state，并使用 producer `a935a3cf5efb1fa7a952ca6f45a5994609b367e9` 的冻结重建逻辑与
immutable labels/witness 逐 byte 重建原 exact-three artifact。没有重新运行 GPU、model、image processor、
vision feature、policy、generation、restoration attribution 或 gate；原 artifact 与原 GPU attempt ledger 未修改。

- status: `{summary['status']}`
- validation source: `{summary['source_execution']['validation_source_git_commit']}`
- producer source: `{summary['source_execution']['producer_source_git_commit']}`
- exact-byte matches: `{all(row['byte_equal'] for row in files)}`
- producer scientific payload: `{summary['producer_identity']['scientific_payload_sha256']}`
"""
    return text.encode("utf-8")


def build_audit_files(
    *,
    contract: PolicyVisionV3ValidationRepairContract,
    replay: ReplayResult,
    validation_source: Mapping[str, Any],
    attempt_claim: Mapping[str, Any],
    attempt_claim_sha256: str,
    producer_ledger_sha256: str,
    runtime_record: Mapping[str, Any],
) -> dict[str, bytes]:
    exact_files = []
    bindings = {
        binding["path"]: binding
        for binding in contract.data["producer_artifact"]["exact_files"]
    }
    for name in sorted(replay.rebuilt_files):
        payload = replay.rebuilt_files[name]
        exact_files.append(
            {
                "path": name,
                "size_bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "expected_size_bytes": bindings[name]["size_bytes"],
                "expected_sha256": bindings[name]["sha256"],
                "byte_equal": payload == replay.producer_files_before[name],
            }
        )
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": RUN_STATUS,
        "protocol_id": PROTOCOL_ID,
        "source_execution": {
            **dict(validation_source),
            "contract_sha256": contract.sha256,
        },
        "producer_identity": {
            "protocol_id": PRODUCER_PROTOCOL_ID,
            "producer_source_git_commit": PRODUCER_SOURCE_GIT_COMMIT,
            "artifact_git_commit": ARTIFACT_GIT_COMMIT,
            "producer_contract_sha256": PRODUCER_CONFIG_SHA256,
            "scientific_payload_sha256": contract.data["producer_artifact"][
                "scientific_payload_sha256"
            ],
            "producer_gpu_attempt_ledger_sha256": producer_ledger_sha256,
        },
        "repair": {
            "only_semantic_change": (
                "project_exact_seven_key_evaluated_state_to_exact_four_key_"
                "feature_state"
            ),
            "evaluated_state_keys_exact": sorted(EVALUATED_STATE_KEYS),
            "feature_state_keys_exact_in_output_order": list(FEATURE_STATE_KEYS),
            "all_non_state_reconstruction_fields_unchanged": True,
        },
        "exact_byte_replay": {
            "producer_files": exact_files,
            "all_files_byte_equal": all(row["byte_equal"] for row in exact_files),
            "feature_record_count": replay.feature_record_count,
            "candidate_score_count": replay.candidate_score_count,
            "producer_artifact_bytes_unchanged_before_after": (
                replay.producer_files_before == replay.producer_files_after
            ),
        },
        "formal_attempt_claim": {
            "record": dict(attempt_claim),
            "sha256": attempt_claim_sha256,
        },
        "runtime": dict(runtime_record),
        "operation_counts": dict(ZERO_CPU_REPAIR_OPERATIONS),
    }
    payload = dict(summary)
    summary["validation_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(payload)
    )
    return {
        "README.md": _audit_readme(summary),
        "summary.json": pretty_json_bytes(summary),
    }


def write_atomic_result(
    output_dir: Path,
    files: Mapping[str, bytes],
    *,
    pre_publish_check: Callable[[], None],
) -> None:
    if set(files) != set(EXPECTED_OUTPUT_FILES):
        raise ValueError("validation-repair output file inventory drifted")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("canonical validation-repair output already exists")
    staging = output_dir.with_name(f".{output_dir.name}.staging")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError("validation-repair staging output already exists")
    if not output_dir.parent.is_dir() or output_dir.parent.is_symlink():
        raise ValueError("validation-repair output parent must preexist")
    os.mkdir(staging, 0o700)
    try:
        for name in EXPECTED_OUTPUT_FILES:
            path = staging / name
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o644,
            )
            try:
                view = memoryview(files[name])
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("result write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        staging_descriptor = os.open(
            staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(staging_descriptor)
        finally:
            os.close(staging_descriptor)
        pre_publish_check()
        os.replace(staging, output_dir)
        parent_descriptor = os.open(
            output_dir.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_formal_validation_repair(
    *,
    contract: PolicyVisionV3ValidationRepairContract,
    labels_archive: Path,
    producer_attempt_ledger: Path,
    attempt_ledger: Path,
    completion_seal: Path,
    output_dir: Path,
    validation_source_git_commit: str,
    argv: Sequence[str],
    now: Callable[[], str] = utc_now,
) -> dict[str, bytes]:
    if output_dir.resolve() != contract.output_directory:
        raise ValueError("validation-repair output must use the canonical sibling path")
    if attempt_ledger.resolve() != Path(FORMAL_ATTEMPT_LEDGER_PATH).resolve():
        raise ValueError("validation-repair attempt ledger path drifted")
    if completion_seal.resolve() != Path(FORMAL_COMPLETION_SEAL_PATH).resolve():
        raise ValueError("validation-repair completion seal path drifted")
    no_nvidia = validate_no_nvidia_devices()
    contract.validate_bound_sources()
    source = contract.validate_formal_source(
        validation_source_git_commit,
        require_clean_pushed_main=True,
    )
    contract.validate_outputs_absent(
        ledger_path=attempt_ledger,
        completion_seal_path=completion_seal,
    )
    artifact_before = _artifact_snapshot(contract)
    producer_ledger_before = validate_producer_attempt_ledger(
        producer_attempt_ledger, contract
    )
    runtime_identity = build_runtime_identity(no_nvidia)
    started = now()
    claim = {
        "schema_version": "1.0.0",
        "status": "CLAIMED_POLICY_VISION_V3_VALIDATION_REPAIR_V1_ATTEMPT",
        "protocol_id": PROTOCOL_ID,
        "producer_source_git_commit": PRODUCER_SOURCE_GIT_COMMIT,
        "artifact_git_commit": ARTIFACT_GIT_COMMIT,
        "validation_source_git_commit": validation_source_git_commit,
        "contract_sha256": contract.sha256,
        "canonical_result_directory": contract.data["output_contract"][
            "canonical_result_directory"
        ],
        "formal_attempt_ledger_path": FORMAL_ATTEMPT_LEDGER_PATH,
        "formal_completion_seal_path": FORMAL_COMPLETION_SEAL_PATH,
        "producer_gpu_attempt_ledger_sha256": sha256_bytes(
            producer_ledger_before
        ),
        "argv": list(argv),
        "started_at_utc": started,
        "runtime_identity": runtime_identity,
    }
    claim_payload, claim_sha256 = claim_attempt_ledger(attempt_ledger, claim)
    repair_ledger_before = read_repair_attempt_ledger(attempt_ledger)
    if (
        repair_ledger_before.payload != claim_payload
        or repair_ledger_before.record != claim
        or repair_ledger_before.sha256 != claim_sha256
    ):
        raise ValueError("validation-repair attempt ledger changed after claim")
    producer_contract = RestorationV22PolicyVisionV3Contract.load(
        contract.repository_root / contract.data["producer"]["v3_contract"]["path"],
        repository_root=contract.repository_root,
    )
    _validate_labels_archive(producer_contract, labels_archive)
    replay = replay_exact_producer_artifact(
        contract=contract,
        labels_archive=labels_archive,
    )
    finished = now()
    runtime_record = build_runtime_record(
        runtime_identity=runtime_identity,
        argv=argv,
        started_at_utc=started,
        finished_at_utc=finished,
    )
    files = build_audit_files(
        contract=contract,
        replay=replay,
        validation_source=source,
        attempt_claim=claim,
        attempt_claim_sha256=claim_sha256,
        producer_ledger_sha256=sha256_bytes(producer_ledger_before),
        runtime_record=runtime_record,
    )
    seal_record = build_completion_seal_record(
        files=files,
        repair_ledger=repair_ledger_before,
        finished_at_utc=finished,
    )
    seal_payload, seal_sha256 = claim_completion_seal(
        completion_seal,
        seal_record,
    )
    completion_seal_before = read_completion_seal(completion_seal)
    if (
        completion_seal_before.payload != seal_payload
        or completion_seal_before.record != seal_record
        or completion_seal_before.sha256 != seal_sha256
    ):
        raise ValueError("validation-repair completion seal changed after creation")

    # note (luojiaxuan): Recheck the fully clean source immediately before the
    # canonical staging directory is created; staging has a separate exact
    # untracked-inventory guard below.
    contract.validate_formal_source(
        validation_source_git_commit,
        require_clean_pushed_main=True,
    )
    staging = output_dir.with_name(f".{output_dir.name}.staging")

    def pre_publish_check() -> None:
        contract.validate_source_during_staging(
            validation_source_git_commit,
            staging_directory=staging,
        )
        if _artifact_snapshot(contract) != artifact_before:
            raise ValueError("producer artifact changed before audit publication")
        if (
            validate_producer_attempt_ledger(producer_attempt_ledger, contract)
            != producer_ledger_before
        ):
            raise ValueError("producer GPU attempt ledger changed before publication")
        if read_repair_attempt_ledger(attempt_ledger) != repair_ledger_before:
            raise ValueError(
                "validation-repair attempt ledger changed before publication"
            )
        if read_completion_seal(completion_seal) != completion_seal_before:
            raise ValueError(
                "validation-repair completion seal changed before publication"
            )
        validate_no_nvidia_devices()

    write_atomic_result(output_dir, files, pre_publish_check=pre_publish_check)
    readback = validate_existing_audit_result(
        output_dir,
        contract=contract,
        replay=replay,
        validation_source=source,
        producer_ledger_payload=producer_ledger_before,
        repair_ledger=repair_ledger_before,
        completion_seal=completion_seal_before,
        actual_no_nvidia_runtime=no_nvidia,
    )
    contract.validate_source_during_staging(
        validation_source_git_commit,
        staging_directory=output_dir,
    )
    if _artifact_snapshot(contract) != artifact_before:
        raise ValueError("producer artifact changed after audit publication")
    if (
        validate_producer_attempt_ledger(producer_attempt_ledger, contract)
        != producer_ledger_before
    ):
        raise ValueError("producer GPU attempt ledger changed after publication")
    if read_repair_attempt_ledger(attempt_ledger) != repair_ledger_before:
        raise ValueError("validation-repair attempt ledger changed after publication")
    if read_completion_seal(completion_seal) != completion_seal_before:
        raise ValueError("validation-repair completion seal changed after publication")
    validate_no_nvidia_devices()
    return readback


def _read_existing_audit_files(
    output_dir: Path,
    *,
    contract: PolicyVisionV3ValidationRepairContract,
) -> dict[str, bytes]:
    if output_dir.resolve() != contract.output_directory:
        raise ValueError("validation-repair output must use the canonical sibling path")
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError("canonical validation-repair output is missing or symlinked")
    children = tuple(output_dir.iterdir())
    if (
        sorted(path.name for path in children) != sorted(EXPECTED_OUTPUT_FILES)
        or any(path.is_symlink() or not path.is_file() for path in children)
    ):
        raise ValueError("validation-repair output exact-file inventory drifted")
    return {name: (output_dir / name).read_bytes() for name in EXPECTED_OUTPUT_FILES}


def validate_audit_against_replay(
    files: Mapping[str, bytes],
    *,
    contract: PolicyVisionV3ValidationRepairContract,
    replay: ReplayResult,
    validation_source: Mapping[str, Any],
    producer_ledger_payload: bytes,
    repair_ledger: RepairLedgerSnapshot,
    completion_seal: CompletionSealSnapshot,
    actual_no_nvidia_runtime: Mapping[str, Any],
) -> dict[str, bytes]:
    if set(files) != set(EXPECTED_OUTPUT_FILES):
        raise ValueError("validation-repair audit input inventory drifted")
    if (
        pretty_json_bytes(repair_ledger.record) != repair_ledger.payload
        or sha256_bytes(repair_ledger.payload) != repair_ledger.sha256
    ):
        raise ValueError("validation-repair attempt ledger snapshot drifted")
    try:
        summary = json.loads(files["summary.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("validation-repair audit summary is not JSON") from error
    expected_summary_keys = {
        "schema_version",
        "status",
        "protocol_id",
        "source_execution",
        "producer_identity",
        "repair",
        "exact_byte_replay",
        "formal_attempt_claim",
        "runtime",
        "operation_counts",
        "validation_payload_sha256",
    }
    if not isinstance(summary, Mapping) or set(summary) != expected_summary_keys:
        raise ValueError("validation-repair audit summary schema drifted")
    claim = summary.get("formal_attempt_claim")
    if (
        not isinstance(claim, Mapping)
        or set(claim) != {"record", "sha256"}
        or claim.get("record") != repair_ledger.record
        or claim.get("sha256") != repair_ledger.sha256
    ):
        raise ValueError("validation-repair audit ledger claim drifted")
    claim_record = repair_ledger.record
    expected_claim_keys = {
        "schema_version",
        "status",
        "protocol_id",
        "producer_source_git_commit",
        "artifact_git_commit",
        "validation_source_git_commit",
        "contract_sha256",
        "canonical_result_directory",
        "formal_attempt_ledger_path",
        "formal_completion_seal_path",
        "producer_gpu_attempt_ledger_sha256",
        "argv",
        "started_at_utc",
        "runtime_identity",
    }
    if not isinstance(claim_record, Mapping) or set(claim_record) != expected_claim_keys:
        raise ValueError("validation-repair attempt claim schema drifted")
    producer_ledger_sha256 = sha256_bytes(producer_ledger_payload)
    expected_claim = {
        "schema_version": "1.0.0",
        "status": "CLAIMED_POLICY_VISION_V3_VALIDATION_REPAIR_V1_ATTEMPT",
        "protocol_id": PROTOCOL_ID,
        "producer_source_git_commit": PRODUCER_SOURCE_GIT_COMMIT,
        "artifact_git_commit": ARTIFACT_GIT_COMMIT,
        "validation_source_git_commit": validation_source[
            "validation_source_git_commit"
        ],
        "contract_sha256": contract.sha256,
        "canonical_result_directory": contract.data["output_contract"][
            "canonical_result_directory"
        ],
        "formal_attempt_ledger_path": FORMAL_ATTEMPT_LEDGER_PATH,
        "formal_completion_seal_path": FORMAL_COMPLETION_SEAL_PATH,
        "producer_gpu_attempt_ledger_sha256": producer_ledger_sha256,
        "argv": claim_record.get("argv"),
        "started_at_utc": claim_record.get("started_at_utc"),
        "runtime_identity": claim_record.get("runtime_identity"),
    }
    if dict(claim_record) != expected_claim:
        raise ValueError("validation-repair attempt claim values drifted")
    completion_record = validate_completion_seal_record(
        completion_seal,
        repair_ledger=repair_ledger,
    )
    runtime = validate_runtime_record(
        summary.get("runtime"),
        actual_no_nvidia_runtime=actual_no_nvidia_runtime,
        attempt_claim=claim_record,
        expected_finished_at_utc=completion_record["finished_at_utc"],
    )
    validate_completion_seal_output_bindings(
        completion_record,
        files=files,
        summary=summary,
    )
    source_keys = (
        "producer_source_git_commit",
        "validation_source_git_commit",
        "source_diff_baseline_git_commit",
        "formal_source_changed_paths_exact",
        "producer_python_source_closure",
        "validation_python_source_closure",
    )
    if any(key not in validation_source for key in source_keys):
        raise ValueError("fresh validation source identity is incomplete")
    normalized_source = {key: validation_source[key] for key in source_keys}
    rebuilt_audit = build_audit_files(
        contract=contract,
        replay=replay,
        validation_source=normalized_source,
        attempt_claim=claim_record,
        attempt_claim_sha256=repair_ledger.sha256,
        producer_ledger_sha256=producer_ledger_sha256,
        runtime_record=runtime,
    )
    if any(rebuilt_audit[name] != files[name] for name in EXPECTED_OUTPUT_FILES):
        raise ValueError(
            "validation-repair audit differs from fresh replay, source, or ledgers"
        )
    return dict(files)


def validate_existing_audit_result(
    output_dir: Path,
    *,
    contract: PolicyVisionV3ValidationRepairContract,
    replay: ReplayResult,
    validation_source: Mapping[str, Any],
    producer_ledger_payload: bytes,
    repair_ledger: RepairLedgerSnapshot,
    completion_seal: CompletionSealSnapshot,
    actual_no_nvidia_runtime: Mapping[str, Any],
) -> dict[str, bytes]:
    files = _read_existing_audit_files(output_dir, contract=contract)
    return validate_audit_against_replay(
        files,
        contract=contract,
        replay=replay,
        validation_source=validation_source,
        producer_ledger_payload=producer_ledger_payload,
        repair_ledger=repair_ledger,
        completion_seal=completion_seal,
        actual_no_nvidia_runtime=actual_no_nvidia_runtime,
    )
