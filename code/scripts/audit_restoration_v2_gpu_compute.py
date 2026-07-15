"""Run the policy-blind CUDA audit for restoration-v2 compute primitives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import causalcache.restoration_v2_batching as batching_module
import causalcache.restoration_v2_gpu_kl as gpu_kl_module
from causalcache.restoration_v2_batching import (
    FROZEN_COALITION_MICROBATCH_SIZE,
    CoalitionMicrobatchRecord,
    plan_coalition_microbatches,
)
from causalcache.restoration_v2_gpu_kl import (
    CPU_ORACLE_EQUIVALENCE_ATOL,
    CPU_ORACLE_EQUIVALENCE_RTOL,
    cpu_full_vocab_mean_kl_oracle_for_tests,
    gpu_resident_full_vocab_mean_kl,
)


SCHEMA_VERSION = "0.1.0"
PROTOCOL_ID = "causalcache_restoration_v2"
OUTCOME = "PASSED_RESTORATION_V2_GPU_COMPUTE_AUDIT"
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
CONTAINER_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
CUDA_DEVICE_PATTERN = re.compile(r"cuda:([0-9]+)")
HOST_IDENTITY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
AUDIT_SOURCE_RELATIVE_PATH = "code/scripts/audit_restoration_v2_gpu_compute.py"
SOURCE_MODULES = {
    "code/causalcache/restoration_v2_gpu_kl.py": gpu_kl_module,
    "code/causalcache/restoration_v2_batching.py": batching_module,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cli_identity(args: argparse.Namespace) -> int:
    if GIT_SHA_PATTERN.fullmatch(args.run_git_commit) is None:
        raise ValueError("--run-git-commit must be a full lowercase 40-hex SHA")
    if CONTAINER_DIGEST_PATTERN.fullmatch(args.container_image_digest) is None:
        raise ValueError(
            "--container-image-digest must use sha256:<64 lowercase hex>"
        )
    match = CUDA_DEVICE_PATTERN.fullmatch(args.device)
    if match is None:
        raise ValueError("--device must be explicit and use cuda:<non-negative index>")
    for field in ("host_alias", "host_hostname"):
        value = getattr(args, field)
        if not isinstance(value, str) or HOST_IDENTITY_PATTERN.fullmatch(value) is None:
            raise ValueError(
                f"--{field.replace('_', '-')} must be a non-empty host identity"
            )
    if CONTAINER_ID_PATTERN.fullmatch(args.container_id) is None:
        raise ValueError("--container-id must be a full lowercase 64-hex Docker ID")
    return int(match.group(1))


def _git_output(repository_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _discover_repository_root(explicit_root: Path | None) -> Path:
    if explicit_root is None:
        root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    else:
        root = explicit_root
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("repository root must be an existing directory")
    observed = Path(_git_output(root, "rev-parse", "--show-toplevel")).resolve()
    if observed != root:
        raise ValueError("--repository-root must be the exact Git worktree root")
    return root


def _validate_repository_state(
    repository_root: Path,
    *,
    run_git_commit: str,
) -> dict[str, object]:
    actual_head = _git_output(repository_root, "rev-parse", "HEAD")
    if GIT_SHA_PATTERN.fullmatch(actual_head) is None:
        raise ValueError("Git HEAD did not resolve to a full lowercase commit SHA")
    if actual_head != run_git_commit:
        raise ValueError("--run-git-commit must exactly equal the checked-out Git HEAD")
    verified_head = _git_output(repository_root, "rev-parse", "--verify", "HEAD^{commit}")
    if verified_head != actual_head:
        raise ValueError("Git HEAD commit verification drifted")
    status = _git_output(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise ValueError("formal CUDA audit requires a clean Git worktree")
    return {
        "repository_root": str(repository_root),
        "run_git_commit": actual_head,
        "head_verified_as_commit": True,
        "worktree_clean": True,
        "untracked_files_checked": True,
        "submodules_checked": True,
    }


def _source_file_inventory(repository_root: Path) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for relative_path, module in SOURCE_MODULES.items():
        source_path = (repository_root / relative_path).resolve()
        module_path_value = getattr(module, "__file__", None)
        if not isinstance(module_path_value, str):
            raise ValueError(f"imported module has no source file: {relative_path}")
        module_path = Path(module_path_value).resolve()
        if module_path != source_path:
            raise ValueError(
                "imported restoration-v2 compute module is not the checked-out source: "
                f"{relative_path}"
            )
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        inventory[relative_path] = {
            "sha256": _sha256_file(source_path),
            "size_bytes": source_path.stat().st_size,
        }
    audit_source_path = (repository_root / AUDIT_SOURCE_RELATIVE_PATH).resolve()
    if Path(__file__).resolve() != audit_source_path:
        raise ValueError("imported CUDA audit CLI is not the checked-out source")
    if not audit_source_path.is_file():
        raise FileNotFoundError(audit_source_path)
    inventory[AUDIT_SOURCE_RELATIVE_PATH] = {
        "sha256": _sha256_file(audit_source_path),
        "size_bytes": audit_source_path.stat().st_size,
    }
    return inventory


def _load_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("formal restoration-v2 GPU audit requires PyTorch") from error
    return torch


def _require_cuda_runtime(torch: Any, *, device: str, device_index: int) -> Any:
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("formal restoration-v2 GPU audit requires available CUDA")
    device_count = int(torch.cuda.device_count())
    if device_index >= device_count:
        raise ValueError(
            f"requested {device}, but only {device_count} CUDA device(s) are visible"
        )
    canonical_device = torch.device(device)
    if str(canonical_device) != device:
        raise ValueError("CUDA device canonicalization drifted")
    torch.cuda.set_device(canonical_device)
    return canonical_device


def _synthetic_inputs() -> tuple[
    list[list[float]],
    list[list[float]],
    list[list[float]],
]:
    reference_probabilities = [
        [0.40, 0.25, 0.15, 0.12, 0.08],
        [0.10, 0.20, 0.30, 0.25, 0.15],
        [0.05, 0.10, 0.15, 0.20, 0.50],
        [0.33, 0.22, 0.18, 0.17, 0.10],
    ]
    reference_log_probs = [
        [math.log(probability) for probability in row]
        for row in reference_probabilities
    ]
    candidate_a = [
        [1.00, 0.50, 0.00, -0.50, -1.00],
        [-0.50, 0.00, 0.50, 1.00, 1.50],
        [1.25, 0.75, 0.25, -0.25, -0.75],
        [0.00, 0.25, -0.25, 0.50, -0.50],
    ]
    candidate_b = [
        [-1.00, -0.50, 0.00, 0.50, 1.00],
        [1.50, 1.00, 0.50, 0.00, -0.50],
        [-0.75, -0.25, 0.25, 0.75, 1.25],
        [0.50, -0.50, 0.25, -0.25, 0.00],
    ]
    return reference_log_probs, candidate_a, candidate_b


def _equivalence_record(
    *,
    actual: float,
    expected: float,
    comparison: str,
) -> dict[str, object]:
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError(f"non-finite value in {comparison}")
    absolute_error = abs(actual - expected)
    allowed_error = CPU_ORACLE_EQUIVALENCE_ATOL + (
        CPU_ORACLE_EQUIVALENCE_RTOL * abs(expected)
    )
    passed = math.isclose(
        actual,
        expected,
        abs_tol=CPU_ORACLE_EQUIVALENCE_ATOL,
        rel_tol=CPU_ORACLE_EQUIVALENCE_RTOL,
    )
    record = {
        "comparison": comparison,
        "actual": actual,
        "expected": expected,
        "absolute_error": absolute_error,
        "allowed_error": allowed_error,
        "atol": CPU_ORACLE_EQUIVALENCE_ATOL,
        "rtol": CPU_ORACLE_EQUIVALENCE_RTOL,
        "passed": passed,
    }
    if not passed:
        raise ValueError(
            f"restoration-v2 GPU compute equivalence failed: {comparison}"
        )
    return record


def _timed_gpu_kl(
    torch: Any,
    *,
    device: Any,
    reference: Any,
    candidate: Any,
    candidate_representation: str,
) -> tuple[Any, dict[str, object]]:
    torch.cuda.synchronize(device)
    started_ns = time.perf_counter_ns()
    result = gpu_resident_full_vocab_mean_kl(
        reference,
        candidate,
        candidate_representation=candidate_representation,
    )
    torch.cuda.synchronize(device)
    ended_ns = time.perf_counter_ns()
    return result, {
        "elapsed_nanoseconds": ended_ns - started_ns,
        "elapsed_seconds": (ended_ns - started_ns) / 1_000_000_000,
    }


def _validate_primitive_audits(
    result_a: Any,
    result_b: Any,
    batch_result: Any,
    *additional_results: Any,
) -> dict[str, object]:
    batch_audit = batch_result.audit.to_dict()
    if batch_audit.get("reference_batch_stride") != 0:
        raise ValueError("batch-2 KL audit did not record zero reference stride")
    if batch_audit.get("reference_zero_copy_batch_expansion") is not True:
        raise ValueError("batch-2 KL audit did not record zero-copy expansion")
    if batch_audit.get("reference_compute_batch_size") != 1:
        raise ValueError("batch-2 KL audit did not reuse one reference compute batch")
    audited_results = (result_a, result_b, batch_result, *additional_results)
    if any(
        record.audit.validation_scalar_host_reads != 0
        for record in audited_results
    ):
        raise ValueError("GPU KL primitive reported a validation scalar host read")
    if any(
        record.audit.full_tensor_host_transfers != 0 for record in audited_results
    ):
        raise ValueError("GPU KL primitive reported a full-tensor host transfer")
    if any(
        record.audit.numeric_validation != "gpu_resident_per_example_predicates"
        for record in audited_results
    ):
        raise ValueError("GPU KL primitive numeric validation contract drifted")
    if any(
        record.audit.invalid_numeric_output != "nan_final_distance"
        for record in audited_results
    ):
        raise ValueError("GPU KL primitive invalid-numeric output contract drifted")
    if any(
        record.audit.reference_compute_batch_size != 1
        for record in (result_a, result_b)
    ):
        raise ValueError("batch-1 reference compute size drifted")
    return batch_audit


def _invalid_numeric_distance_record(final_distance: float) -> dict[str, object]:
    if not math.isnan(final_distance):
        raise ValueError("invalid synthetic numeric input did not become NaN distance")
    return {
        "status": "passed",
        "injected_invalidity": "candidate_logits_nan",
        "final_distance_is_nan": True,
        "serialized_final_distance": "nan",
        "host_read_scope": "final_distance_scalar_only",
    }


def _audit_planner() -> dict[str, object]:
    records = (
        CoalitionMicrobatchRecord(
            coalition_id="synthetic-state:candidate-a",
            image_count=2,
            sequence_length=128,
            input_index=0,
        ),
        CoalitionMicrobatchRecord(
            coalition_id="synthetic-state:candidate-b",
            image_count=2,
            sequence_length=128,
            input_index=1,
        ),
    )
    plan = plan_coalition_microbatches(
        records,
        microbatch_size=FROZEN_COALITION_MICROBATCH_SIZE,
    )
    audit = plan.audit.as_dict()
    observed_batches = [
        {
            "microbatch_index": batch.microbatch_index,
            "coalition_ids": list(batch.coalition_ids),
            "size": len(batch.records),
            "image_count": batch.image_count,
            "sequence_length": batch.sequence_length,
        }
        for batch in plan.microbatches
    ]
    if audit.get("microbatch_size") != 2:
        raise ValueError("restoration-v2 planner microbatch size drifted")
    if audit.get("automatic_oom_fallback") is not False:
        raise ValueError("restoration-v2 planner enabled automatic OOM fallback")
    if len(observed_batches) != 1 or observed_batches[0]["size"] != 2:
        raise ValueError("restoration-v2 planner did not create the exact batch-2 audit")
    return {
        "status": "passed",
        "planning_scope": "single_decision_state",
        "planner_audit": audit,
        "observed_microbatches": observed_batches,
        "automatic_oom_fallback": False,
    }


def _nvidia_smi_identity(
    device_index: int,
    *,
    preferred_gpu_uuid: str | None = None,
) -> dict[str, str | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"gpu_uuid": None, "driver_version": None}
    records: list[dict[str, str]] = []
    for raw_line in result.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",", maxsplit=2)]
        if len(fields) == 3 and all(fields):
            records.append(
                {
                    "index": fields[0],
                    "gpu_uuid": fields[1],
                    "driver_version": fields[2],
                }
            )
    uuid_match = next(
        (
            record
            for record in records
            if preferred_gpu_uuid is not None
            and record["gpu_uuid"] == preferred_gpu_uuid
        ),
        None,
    )
    exact_index = next(
        (record for record in records if record["index"] == str(device_index)),
        None,
    )
    selected = uuid_match if uuid_match is not None else exact_index
    if selected is None and len(records) == 1:
        selected = records[0]
    return {
        "gpu_uuid": selected["gpu_uuid"] if selected is not None else None,
        "driver_version": (
            selected["driver_version"]
            if selected is not None
            else (records[0]["driver_version"] if records else None)
        ),
    }


def _gpu_runtime_identity(torch: Any, *, device: Any, device_index: int) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    property_uuid = getattr(properties, "uuid", None)
    if isinstance(property_uuid, bytes):
        gpu_uuid = property_uuid.decode("ascii")
    else:
        gpu_uuid = str(property_uuid) if property_uuid is not None else None
    nvidia_smi = _nvidia_smi_identity(
        device_index,
        preferred_gpu_uuid=gpu_uuid,
    )
    if nvidia_smi["driver_version"] is None:
        raise RuntimeError("nvidia-smi did not report an NVIDIA driver version")
    uuid_source = "torch.cuda.get_device_properties"
    if not gpu_uuid:
        gpu_uuid = nvidia_smi["gpu_uuid"]
        uuid_source = "nvidia-smi" if gpu_uuid is not None else None
    cudnn_version = None
    if hasattr(torch.backends, "cudnn"):
        cudnn_version = torch.backends.cudnn.version()
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_build_version": str(torch.version.cuda),
        "cudnn_version": cudnn_version,
        "python_version": platform.python_version(),
        "platform_machine": platform.machine(),
        "nvidia_driver_version": nvidia_smi["driver_version"],
        "visible_cuda_device_count": int(torch.cuda.device_count()),
        "selected_device": str(device),
        "gpu_name": str(properties.name),
        "gpu_uuid": gpu_uuid,
        "gpu_uuid_source": uuid_source,
        "nvidia_smi_gpu_uuid": nvidia_smi["gpu_uuid"],
        "gpu_total_memory_bytes": int(properties.total_memory),
        "gpu_compute_capability": [int(properties.major), int(properties.minor)],
        "gpu_multiprocessor_count": int(properties.multi_processor_count),
    }


def _audit_cuda_compute(torch: Any, *, device: Any) -> dict[str, object]:
    reference_values, candidate_a_values, candidate_b_values = _synthetic_inputs()
    reference = torch.tensor(
        [reference_values],
        dtype=torch.float32,
        device=device,
    )
    candidate_a = torch.tensor(
        [candidate_a_values],
        dtype=torch.bfloat16,
        device=device,
    )
    candidate_b = torch.tensor(
        [candidate_b_values],
        dtype=torch.bfloat16,
        device=device,
    )
    candidate_batch = torch.tensor(
        [candidate_a_values, candidate_b_values],
        dtype=torch.bfloat16,
        device=device,
    )
    with torch.inference_mode():
        candidate_a_log_probs = torch.log_softmax(
            candidate_a.to(dtype=torch.float32),
            dim=-1,
        )
    invalid_candidate_values = [[list(row) for row in candidate_a_values]]
    invalid_candidate_values[0][0][0] = math.nan
    invalid_candidate = torch.tensor(
        invalid_candidate_values,
        dtype=torch.bfloat16,
        device=device,
    )
    if str(reference.dtype) != "torch.float32":
        raise ValueError("synthetic reference dtype drifted")
    if str(candidate_batch.dtype) != "torch.bfloat16":
        raise ValueError("synthetic candidate dtype drifted")
    if str(candidate_a_log_probs.dtype) != "torch.float32":
        raise ValueError("synthetic pre-normalized log-probability dtype drifted")

    torch.cuda.reset_peak_memory_stats(device)
    memory_before = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }
    result_a, latency_a = _timed_gpu_kl(
        torch,
        device=device,
        reference=reference,
        candidate=candidate_a,
        candidate_representation="logits",
    )
    result_b, latency_b = _timed_gpu_kl(
        torch,
        device=device,
        reference=reference,
        candidate=candidate_b,
        candidate_representation="logits",
    )

    expanded_reference = reference.expand(2, -1, -1)
    if int(expanded_reference.stride(0)) != 0:
        raise ValueError("synthetic reference expansion did not preserve zero stride")
    storage_shared = (
        expanded_reference.untyped_storage().data_ptr()
        == reference.untyped_storage().data_ptr()
    )
    if not storage_shared:
        raise ValueError("synthetic expanded reference did not share storage")
    batch_result, latency_batch = _timed_gpu_kl(
        torch,
        device=device,
        reference=expanded_reference,
        candidate=candidate_batch,
        candidate_representation="logits",
    )
    log_prob_result, latency_log_probs = _timed_gpu_kl(
        torch,
        device=device,
        reference=reference,
        candidate=candidate_a_log_probs,
        candidate_representation="log_probs",
    )
    invalid_result, latency_invalid = _timed_gpu_kl(
        torch,
        device=device,
        reference=reference,
        candidate=invalid_candidate,
        candidate_representation="logits",
    )

    scalar_a = float(result_a.per_example_mean_kl[0].item())
    scalar_b = float(result_b.per_example_mean_kl[0].item())
    batch_scalars = tuple(
        float(value) for value in batch_result.per_example_mean_kl.detach().tolist()
    )
    if len(batch_scalars) != 2:
        raise ValueError("synthetic batch-2 GPU result shape drifted")
    log_prob_scalar = float(log_prob_result.per_example_mean_kl[0].item())
    invalid_scalar = float(invalid_result.per_example_mean_kl[0].item())
    oracle_a = cpu_full_vocab_mean_kl_oracle_for_tests(
        [reference_values],
        [candidate_a_values],
        candidate_representation="logits",
    )[0]
    oracle_b = cpu_full_vocab_mean_kl_oracle_for_tests(
        [reference_values],
        [candidate_b_values],
        candidate_representation="logits",
    )[0]
    oracle_equivalence = [
        _equivalence_record(
            actual=scalar_a,
            expected=oracle_a,
            comparison="batch1_candidate_a_gpu_vs_independent_cpu_float64_oracle",
        ),
        _equivalence_record(
            actual=scalar_b,
            expected=oracle_b,
            comparison="batch1_candidate_b_gpu_vs_independent_cpu_float64_oracle",
        ),
    ]
    microbatch_equivalence = [
        _equivalence_record(
            actual=batch_scalars[0],
            expected=scalar_a,
            comparison="batch2_candidate_a_vs_independent_batch1_gpu_call",
        ),
        _equivalence_record(
            actual=batch_scalars[1],
            expected=scalar_b,
            comparison="batch2_candidate_b_vs_independent_batch1_gpu_call",
        ),
    ]
    representation_equivalence = _equivalence_record(
        actual=log_prob_scalar,
        expected=scalar_a,
        comparison="candidate_a_logits_vs_pre_normalized_log_probs",
    )
    invalid_numeric_audit = _invalid_numeric_distance_record(invalid_scalar)
    if log_prob_result.audit.candidate_representation != "log_probs":
        raise ValueError("pre-normalized candidate representation audit drifted")
    if invalid_result.audit.candidate_representation != "logits":
        raise ValueError("invalid candidate representation audit drifted")

    batch_audit = _validate_primitive_audits(
        result_a,
        result_b,
        batch_result,
        log_prob_result,
        invalid_result,
    )

    memory_after = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    return {
        "status": "passed",
        "synthetic_fixture": {
            "generator": "literal_fixed_values_no_rng",
            "batch_size": 2,
            "distance_token_count": len(reference_values),
            "vocabulary_size": len(reference_values[0]),
            "reference_dtype": str(reference.dtype),
            "candidate_logits_dtype": str(candidate_batch.dtype),
        },
        "equivalence_tolerance": {
            "atol": CPU_ORACLE_EQUIVALENCE_ATOL,
            "rtol": CPU_ORACLE_EQUIVALENCE_RTOL,
        },
        "batch1_gpu_vs_independent_cpu_oracle": oracle_equivalence,
        "batch2_vs_two_independent_batch1_gpu_calls": microbatch_equivalence,
        "logits_vs_pre_normalized_log_probs": representation_equivalence,
        "invalid_numeric_to_nan_final_distance": invalid_numeric_audit,
        "zero_stride_reference": {
            "input_batch_stride": int(expanded_reference.stride(0)),
            "storage_shared_with_batch1_reference": storage_shared,
            "reference_compute_batch_size": batch_audit[
                "reference_compute_batch_size"
            ],
            "primitive_audit": batch_audit,
        },
        "latency": {
            "batch1_candidate_a": latency_a,
            "batch1_candidate_b": latency_b,
            "batch2": latency_batch,
            "batch1_pre_normalized_log_probs": latency_log_probs,
            "batch1_invalid_numeric": latency_invalid,
            "total_measured_elapsed_nanoseconds": sum(
                int(record["elapsed_nanoseconds"])
                for record in (
                    latency_a,
                    latency_b,
                    latency_batch,
                    latency_log_probs,
                    latency_invalid,
                )
            ),
        },
        "cuda_memory_before_compute": memory_before,
        "cuda_memory_after_compute": memory_after,
        "returned_host_values": "final_distance_scalars_and_audit_metadata_only",
        "host_read_final_distance_scalar_count": 6,
        "host_read_intermediate_tensor_value_count": 0,
        "automatic_oom_fallback": False,
    }


def _execute_audit(
    args: argparse.Namespace,
    *,
    exact_argv: Sequence[str],
) -> dict[str, object]:
    started_at_utc = _utc_now()
    device_index = _validate_cli_identity(args)
    repository_root = _discover_repository_root(args.repository_root)
    repository = _validate_repository_state(
        repository_root,
        run_git_commit=args.run_git_commit,
    )
    source_files = _source_file_inventory(repository_root)
    policy_modules_before = sorted(
        name for name in sys.modules if name.startswith("causalcache.policy")
    )
    if policy_modules_before:
        raise ValueError("policy modules were loaded before the policy-blind CUDA audit")

    torch = _load_torch()
    device = _require_cuda_runtime(
        torch,
        device=args.device,
        device_index=device_index,
    )
    runtime = _gpu_runtime_identity(
        torch,
        device=device,
        device_index=device_index,
    )
    planner_audit = _audit_planner()
    compute_audit = _audit_cuda_compute(torch, device=device)
    policy_modules_after = sorted(
        name for name in sys.modules if name.startswith("causalcache.policy")
    )
    if policy_modules_after:
        raise ValueError("policy modules were loaded during the policy-blind CUDA audit")

    ended_at_utc = _utc_now()
    if started_at_utc >= ended_at_utc:
        raise ValueError("CUDA audit UTC bracket is not strictly increasing")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "evidence_type": "restoration_v2_policy_blind_gpu_compute_audit",
        "outcome": OUTCOME,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "argv": list(exact_argv),
        "argv_shell_quoted": shlex.join(exact_argv),
        "repository": repository,
        "source_files": source_files,
        "runtime_identity": {
            "host_alias": args.host_alias,
            "host_hostname": args.host_hostname,
            "container_id": args.container_id,
            "container_hostname": socket.gethostname(),
            "container_image_digest": args.container_image_digest,
            **runtime,
        },
        "planner_audit": planner_audit,
        "compute_audit": compute_audit,
        "policy_loaded": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "automatic_oom_fallback": False,
        "dependency_8_compute_primitives_audited": True,
    }


def _write_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as target:
        target.write(encoded)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--run-git-commit", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--host-hostname", required=True)
    parser.add_argument("--container-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    exact_argv = [
        sys.argv[0] if argv is None else str(Path(__file__).resolve()),
        *tokens,
    ]
    args = _build_parser().parse_args(tokens)
    if args.output_summary.exists():
        raise FileExistsError(
            f"--output-summary already exists: {args.output_summary}"
        )
    summary = _execute_audit(args, exact_argv=exact_argv)
    _write_exclusive(args.output_summary, summary)
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
