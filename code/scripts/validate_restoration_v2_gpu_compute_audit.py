"""Independently validate the formal restoration-v2 CUDA compute audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1.0"
PROTOCOL_ID = "causalcache_restoration_v2"
AUDIT_OUTCOME = "PASSED_RESTORATION_V2_GPU_COMPUTE_AUDIT"
VALIDATION_OUTCOME = (
    "PASSED_INDEPENDENT_RESTORATION_V2_GPU_COMPUTE_AUDIT_VALIDATION"
)
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CONTAINER_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
GPU_UUID_PATTERN = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
SOURCE_PATHS = (
    "code/causalcache/restoration_v2_batching.py",
    "code/causalcache/restoration_v2_gpu_kl.py",
    "code/scripts/audit_restoration_v2_gpu_compute.py",
)
TOP_LEVEL_KEYS = {
    "argv",
    "argv_shell_quoted",
    "automatic_oom_fallback",
    "compute_audit",
    "dependency_8_compute_primitives_audited",
    "ended_at_utc",
    "evidence_type",
    "outcome",
    "planner_audit",
    "policy_loaded",
    "policy_output_generated",
    "protocol_id",
    "repository",
    "restoration_output_generated",
    "runtime_identity",
    "schema_version",
    "source_files",
    "started_at_utc",
}
EQUIVALENCE_KEYS = {
    "comparison",
    "actual",
    "expected",
    "absolute_error",
    "allowed_error",
    "atol",
    "rtol",
    "passed",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys drifted")


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _parse_utc(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp") from error
    return parsed


def _git_bytes(repository_root: Path, commit: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository_root), "show", f"{commit}:{relative_path}"],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _validate_git_commit(repository_root: Path, commit: str) -> None:
    verified = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "--verify", f"{commit}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if verified != commit:
        raise ValueError("expected run commit is not an exact local Git commit")


def _validate_argv(summary: Mapping[str, Any], expected: argparse.Namespace) -> None:
    argv = _list(summary["argv"], name="argv")
    if len(argv) != 17 or not all(isinstance(value, str) for value in argv):
        raise ValueError("audit argv shape drifted")
    if not str(argv[0]).endswith("/code/scripts/audit_restoration_v2_gpu_compute.py"):
        raise ValueError("audit argv program path drifted")
    expected_options = [
        ("--device", "cuda:0"),
        ("--run-git-commit", expected.expected_run_git_commit),
        ("--container-image-digest", expected.expected_container_image_digest),
        ("--output-summary", str(argv[8])),
        ("--repository-root", str(summary["repository"]["repository_root"])),
        ("--host-alias", expected.expected_host_alias),
        ("--host-hostname", expected.expected_host_hostname),
        ("--container-id", expected.expected_container_id),
    ]
    flattened = [value for pair in expected_options for value in pair]
    if argv[1:] != flattened:
        raise ValueError("audit argv values or order drifted")
    if not isinstance(summary["argv_shell_quoted"], str):
        raise ValueError("audit argv_shell_quoted must be a string")
    if summary["argv_shell_quoted"] != shlex.join(argv):
        raise ValueError("audit argv_shell_quoted differs from argv")


def _validate_equivalence_record(
    value: Any,
    *,
    expected_comparison: str,
) -> Mapping[str, Any]:
    record = _object(value, name=expected_comparison)
    _exact_keys(record, EQUIVALENCE_KEYS, name=expected_comparison)
    if record["comparison"] != expected_comparison or record["passed"] is not True:
        raise ValueError(f"{expected_comparison} did not pass exactly")
    actual = _finite_number(record["actual"], name=f"{expected_comparison}.actual")
    expected = _finite_number(
        record["expected"], name=f"{expected_comparison}.expected"
    )
    absolute_error = _finite_number(
        record["absolute_error"], name=f"{expected_comparison}.absolute_error"
    )
    allowed_error = _finite_number(
        record["allowed_error"], name=f"{expected_comparison}.allowed_error"
    )
    if record["atol"] != 1e-6 or record["rtol"] != 1e-5:
        raise ValueError(f"{expected_comparison} tolerance drifted")
    recomputed_absolute_error = abs(actual - expected)
    recomputed_allowed_error = 1e-6 + 1e-5 * abs(expected)
    if not math.isclose(absolute_error, recomputed_absolute_error, abs_tol=1e-15):
        raise ValueError(f"{expected_comparison} absolute error was not recomputed")
    if not math.isclose(allowed_error, recomputed_allowed_error, abs_tol=1e-15):
        raise ValueError(f"{expected_comparison} allowed error was not recomputed")
    if absolute_error > allowed_error:
        raise ValueError(f"{expected_comparison} exceeded its tolerance")
    return record


def _validate_compute(summary: Mapping[str, Any]) -> dict[str, Any]:
    compute = _object(summary["compute_audit"], name="compute_audit")
    if compute.get("status") != "passed" or compute.get("automatic_oom_fallback") is not False:
        raise ValueError("compute audit did not pass without OOM fallback")
    tolerance = _object(compute.get("equivalence_tolerance"), name="equivalence_tolerance")
    if tolerance != {"atol": 1e-6, "rtol": 1e-5}:
        raise ValueError("compute equivalence tolerance drifted")
    fixture = _object(compute.get("synthetic_fixture"), name="synthetic_fixture")
    expected_fixture = {
        "generator": "literal_fixed_values_no_rng",
        "batch_size": 2,
        "distance_token_count": 4,
        "vocabulary_size": 5,
        "reference_dtype": "torch.float32",
        "candidate_logits_dtype": "torch.bfloat16",
    }
    if fixture != expected_fixture:
        raise ValueError("synthetic CUDA fixture drifted")

    batch1 = _list(
        compute.get("batch1_gpu_vs_independent_cpu_oracle"),
        name="batch1_gpu_vs_independent_cpu_oracle",
    )
    batch2 = _list(
        compute.get("batch2_vs_two_independent_batch1_gpu_calls"),
        name="batch2_vs_two_independent_batch1_gpu_calls",
    )
    if len(batch1) != 2 or len(batch2) != 2:
        raise ValueError("batch equivalence record count drifted")
    batch1_records = [
        _validate_equivalence_record(
            batch1[0],
            expected_comparison=(
                "batch1_candidate_a_gpu_vs_independent_cpu_float64_oracle"
            ),
        ),
        _validate_equivalence_record(
            batch1[1],
            expected_comparison=(
                "batch1_candidate_b_gpu_vs_independent_cpu_float64_oracle"
            ),
        ),
    ]
    batch2_records = [
        _validate_equivalence_record(
            batch2[0],
            expected_comparison=(
                "batch2_candidate_a_vs_independent_batch1_gpu_call"
            ),
        ),
        _validate_equivalence_record(
            batch2[1],
            expected_comparison=(
                "batch2_candidate_b_vs_independent_batch1_gpu_call"
            ),
        ),
    ]
    if [record["actual"] for record in batch2_records] != [
        record["actual"] for record in batch1_records
    ]:
        raise ValueError("batch-2 distances differ from the two batch-1 distances")
    representation = _validate_equivalence_record(
        compute.get("logits_vs_pre_normalized_log_probs"),
        expected_comparison="candidate_a_logits_vs_pre_normalized_log_probs",
    )
    if representation["actual"] != batch1_records[0]["actual"]:
        raise ValueError("logits and pre-normalized log-probability distances differ")

    invalid = _object(
        compute.get("invalid_numeric_to_nan_final_distance"),
        name="invalid_numeric_to_nan_final_distance",
    )
    if invalid != {
        "status": "passed",
        "injected_invalidity": "candidate_logits_nan",
        "final_distance_is_nan": True,
        "serialized_final_distance": "nan",
        "host_read_scope": "final_distance_scalar_only",
    }:
        raise ValueError("invalid-numeric final-distance behavior drifted")
    if compute.get("returned_host_values") != (
        "final_distance_scalars_and_audit_metadata_only"
    ):
        raise ValueError("returned host-value scope drifted")
    if compute.get("host_read_final_distance_scalar_count") != 6:
        raise ValueError("final-distance scalar host-read count drifted")
    if compute.get("host_read_intermediate_tensor_value_count") != 0:
        raise ValueError("an intermediate tensor value moved to the host")

    zero_stride = _object(compute.get("zero_stride_reference"), name="zero_stride_reference")
    if (
        zero_stride.get("input_batch_stride") != 0
        or zero_stride.get("storage_shared_with_batch1_reference") is not True
        or zero_stride.get("reference_compute_batch_size") != 1
    ):
        raise ValueError("zero-stride reference reuse drifted")
    primitive = _object(zero_stride.get("primitive_audit"), name="primitive_audit")
    expected_primitive = {
        "batch_size": 2,
        "candidate_input_dtype": "torch.bfloat16",
        "candidate_representation": "logits",
        "compute_dtype": "torch.float32",
        "device": "cuda:0",
        "device_validation_category_count": 4,
        "distance_tokens": 4,
        "full_tensor_host_transfers": 0,
        "invalid_numeric_output": "nan_final_distance",
        "log_normalization_atol": 0.0005,
        "negative_kl_atol": 1e-5,
        "numeric_validation": "gpu_resident_per_example_predicates",
        "operation": (
            "teacher_forced_full_vocabulary_mean_kl_on_distance_token_span"
        ),
        "output_dtype": "torch.float32",
        "reduction": (
            "full_vocabulary_sum_then_distance_token_mean_per_example"
        ),
        "reference_batch_stride": 0,
        "reference_compute_batch_size": 1,
        "reference_input_dtype": "torch.float32",
        "reference_zero_copy_batch_expansion": True,
        "validation_scalar_host_reads": 0,
        "vocabulary_size": 5,
    }
    if primitive != expected_primitive:
        raise ValueError("GPU KL primitive audit drifted")
    return {
        "batch1_cpu_oracle_comparisons": 2,
        "batch2_batch1_comparisons": 2,
        "representation_comparisons": 1,
        "maximum_batch1_cpu_oracle_absolute_error": max(
            float(record["absolute_error"]) for record in batch1_records
        ),
        "batch2_maximum_absolute_error": max(
            float(record["absolute_error"]) for record in batch2_records
        ),
        "validation_scalar_host_reads": primitive["validation_scalar_host_reads"],
        "full_tensor_host_transfers": primitive["full_tensor_host_transfers"],
        "invalid_numeric_output": primitive["invalid_numeric_output"],
    }


def _validate_planner(summary: Mapping[str, Any]) -> dict[str, Any]:
    wrapper = _object(summary["planner_audit"], name="planner_audit")
    if (
        wrapper.get("status") != "passed"
        or wrapper.get("planning_scope") != "single_decision_state"
        or wrapper.get("automatic_oom_fallback") is not False
    ):
        raise ValueError("single-decision-state planner audit drifted")
    observed = _list(wrapper.get("observed_microbatches"), name="observed_microbatches")
    if len(observed) != 1:
        raise ValueError("formal planner audit must contain one microbatch")
    batch = _object(observed[0], name="observed_microbatch")
    if batch != {
        "coalition_ids": [
            "synthetic-state:candidate-a",
            "synthetic-state:candidate-b",
        ],
        "image_count": 2,
        "microbatch_index": 0,
        "sequence_length": 128,
        "size": 2,
    }:
        raise ValueError("observed formal microbatch drifted")
    planner = _object(wrapper.get("planner_audit"), name="planner_audit.planner_audit")
    if (
        planner.get("schema_version") != "0.1.0"
        or planner.get("microbatch_size") != 2
        or planner.get("microbatch_size_source") != "explicit_argument"
        or planner.get("automatic_oom_fallback") is not False
        or planner.get("grouping_fields") != ["image_count", "sequence_length"]
        or planner.get("group_order") != (
            "ascending_image_count_then_sequence_length"
        )
        or planner.get("within_group_order") != "ascending_input_index"
        or planner.get("input_record_count") != 2
        or planner.get("group_count") != 1
        or planner.get("microbatch_count") != 1
    ):
        raise ValueError("deterministic planner metadata drifted")
    return {
        "planning_scope": "single_decision_state",
        "microbatch_size": 2,
        "automatic_oom_fallback": False,
    }


def _validate_runtime(summary: Mapping[str, Any], expected: argparse.Namespace) -> dict[str, Any]:
    runtime = _object(summary["runtime_identity"], name="runtime_identity")
    expected_values = {
        "host_alias": expected.expected_host_alias,
        "host_hostname": expected.expected_host_hostname,
        "container_id": expected.expected_container_id,
        "container_hostname": expected.expected_container_id[:12],
        "container_image_digest": expected.expected_container_image_digest,
        "selected_device": "cuda:0",
        "visible_cuda_device_count": 1,
        "gpu_name": expected.expected_gpu_name,
        "gpu_uuid": expected.expected_gpu_uuid,
        "nvidia_smi_gpu_uuid": expected.expected_nvidia_smi_gpu_uuid,
        "nvidia_driver_version": expected.expected_nvidia_driver_version,
        "torch_version": expected.expected_torch_version,
        "torch_cuda_build_version": expected.expected_torch_cuda_build_version,
        "platform_machine": "x86_64",
        "gpu_compute_capability": [9, 0],
        "gpu_multiprocessor_count": 132,
    }
    for key, value in expected_values.items():
        if runtime.get(key) != value:
            raise ValueError(f"runtime identity drifted: {key}")
    if runtime.get("gpu_uuid_source") != "torch.cuda.get_device_properties":
        raise ValueError("GPU UUID source drifted")
    if runtime.get("gpu_total_memory_bytes", 0) < 140_000_000_000:
        raise ValueError("H200 total-memory identity drifted")
    if not isinstance(runtime.get("cudnn_version"), int) or runtime["cudnn_version"] <= 0:
        raise ValueError("cuDNN identity is missing")
    if not isinstance(runtime.get("python_version"), str) or not runtime["python_version"]:
        raise ValueError("Python runtime identity is missing")
    return dict(expected_values)


def _validate_sources(
    summary: Mapping[str, Any],
    *,
    repository_root: Path,
    commit: str,
) -> dict[str, dict[str, Any]]:
    source_files = _object(summary["source_files"], name="source_files")
    if set(source_files) != set(SOURCE_PATHS):
        raise ValueError("formal audit source-file inventory drifted")
    validated: dict[str, dict[str, Any]] = {}
    for relative_path in SOURCE_PATHS:
        record = _object(source_files[relative_path], name=relative_path)
        if set(record) != {"sha256", "size_bytes"}:
            raise ValueError(f"source-file record keys drifted: {relative_path}")
        blob = _git_bytes(repository_root, commit, relative_path)
        expected_record = {
            "sha256": _sha256_bytes(blob),
            "size_bytes": len(blob),
        }
        if record != expected_record:
            raise ValueError(f"source file differs from run Git commit: {relative_path}")
        validated[relative_path] = expected_record
    return validated


def validate_summary(
    summary: Mapping[str, Any],
    *,
    summary_sha256: str,
    repository_root: Path,
    expected: argparse.Namespace,
) -> dict[str, Any]:
    _exact_keys(summary, TOP_LEVEL_KEYS, name="formal CUDA summary")
    if summary["schema_version"] != SCHEMA_VERSION:
        raise ValueError("formal CUDA summary schema version drifted")
    if summary["protocol_id"] != PROTOCOL_ID:
        raise ValueError("formal CUDA protocol id drifted")
    if summary["evidence_type"] != "restoration_v2_policy_blind_gpu_compute_audit":
        raise ValueError("formal CUDA evidence type drifted")
    if summary["outcome"] != AUDIT_OUTCOME:
        raise ValueError("formal CUDA audit outcome did not pass")
    if (
        summary["policy_loaded"] is not False
        or summary["policy_output_generated"] is not False
        or summary["restoration_output_generated"] is not False
    ):
        raise ValueError("formal CUDA audit touched policy or restoration output")
    if summary["automatic_oom_fallback"] is not False:
        raise ValueError("formal CUDA audit used automatic OOM fallback")
    if summary["dependency_8_compute_primitives_audited"] is not True:
        raise ValueError("formal CUDA compute primitive marker did not pass")
    started = _parse_utc(summary["started_at_utc"], name="started_at_utc")
    ended = _parse_utc(summary["ended_at_utc"], name="ended_at_utc")
    if ended <= started:
        raise ValueError("formal CUDA audit UTC bracket is not increasing")

    repository = _object(summary["repository"], name="repository")
    expected_repository_keys = {
        "repository_root",
        "run_git_commit",
        "head_verified_as_commit",
        "worktree_clean",
        "untracked_files_checked",
        "submodules_checked",
    }
    _exact_keys(repository, expected_repository_keys, name="repository")
    if repository["run_git_commit"] != expected.expected_run_git_commit:
        raise ValueError("formal CUDA audit run commit drifted")
    for field in (
        "head_verified_as_commit",
        "worktree_clean",
        "untracked_files_checked",
        "submodules_checked",
    ):
        if repository[field] is not True:
            raise ValueError(f"formal CUDA repository evidence failed: {field}")

    _validate_argv(summary, expected)
    source_files = _validate_sources(
        summary,
        repository_root=repository_root,
        commit=expected.expected_run_git_commit,
    )
    runtime = _validate_runtime(summary, expected)
    compute = _validate_compute(summary)
    planner = _validate_planner(summary)
    validator_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "evidence_type": "independent_restoration_v2_gpu_compute_audit_validation",
        "outcome": VALIDATION_OUTCOME,
        "summary_sha256": summary_sha256,
        "run_git_commit": expected.expected_run_git_commit,
        "source_files_verified_from_run_commit": source_files,
        "runtime_identity": runtime,
        "compute_validation": compute,
        "planner_validation": planner,
        "validator_source": {
            "path": "code/scripts/validate_restoration_v2_gpu_compute_audit.py",
            "sha256": _sha256_file(validator_path),
            "size_bytes": validator_path.stat().st_size,
        },
        "policy_loaded": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "dependency_8_closed": False,
        "screening_unlocked": False,
    }


def _validate_expected_args(args: argparse.Namespace) -> Path:
    if SHA256_PATTERN.fullmatch(args.expected_summary_sha256) is None:
        raise ValueError("--expected-summary-sha256 must be 64 lowercase hex")
    if GIT_SHA_PATTERN.fullmatch(args.expected_run_git_commit) is None:
        raise ValueError("--expected-run-git-commit must be 40 lowercase hex")
    if CONTAINER_ID_PATTERN.fullmatch(args.expected_container_id) is None:
        raise ValueError("--expected-container-id must be 64 lowercase hex")
    if CONTAINER_DIGEST_PATTERN.fullmatch(args.expected_container_image_digest) is None:
        raise ValueError("--expected-container-image-digest must use sha256:<64 hex>")
    if GPU_UUID_PATTERN.fullmatch(args.expected_gpu_uuid) is None:
        raise ValueError("--expected-gpu-uuid must be a lowercase UUID without GPU-")
    if args.expected_nvidia_smi_gpu_uuid != f"GPU-{args.expected_gpu_uuid}":
        raise ValueError("--expected-nvidia-smi-gpu-uuid must match --expected-gpu-uuid")
    repository_root = args.repository_root.expanduser().resolve()
    if not repository_root.is_dir():
        raise ValueError("--repository-root must be an existing directory")
    _validate_git_commit(repository_root, args.expected_run_git_commit)
    return repository_root


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as target:
        target.write(encoded)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--expected-run-git-commit", required=True)
    parser.add_argument("--expected-host-alias", required=True)
    parser.add_argument("--expected-host-hostname", required=True)
    parser.add_argument("--expected-container-id", required=True)
    parser.add_argument("--expected-container-image-digest", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-nvidia-smi-gpu-uuid", required=True)
    parser.add_argument("--expected-nvidia-driver-version", required=True)
    parser.add_argument("--expected-torch-version", required=True)
    parser.add_argument("--expected-torch-cuda-build-version", required=True)
    parser.add_argument("--output-validation", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.output_validation.exists():
        raise FileExistsError(
            f"--output-validation already exists: {args.output_validation}"
        )
    repository_root = _validate_expected_args(args)
    observed_summary_sha256 = _sha256_file(args.summary)
    if observed_summary_sha256 != args.expected_summary_sha256:
        raise ValueError("formal CUDA summary SHA256 drifted")
    summary = _load_json_object(args.summary)
    validation = validate_summary(
        summary,
        summary_sha256=observed_summary_sha256,
        repository_root=repository_root,
        expected=args,
    )
    _write_exclusive(args.output_validation, validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
