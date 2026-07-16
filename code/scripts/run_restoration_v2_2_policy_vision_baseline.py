"""Run or CPU-validate the frozen restoration-v2.2 policy-vision comparator."""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import os
import re
import shutil
import socket
import stat
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from causalcache.policy.gui_owl_v2_2_vision_runtime import (
    GUI_OWL_V2_2_VISION_RUNTIME_ID,
)
from causalcache.restoration_v2_2_policy_vision import (
    DEFAULT_IMAGE_PROCESSOR_SIZE_PROFILE,
    DEFAULT_GPU_UUID_TYPE_PROFILE,
    STATUS,
    PolicyVisionWorkItem,
    evaluate_feature_records,
    feature_worker_shards,
    load_identity_witness,
    load_primary_label_states,
    merge_policy_vision_workers,
    run_policy_vision_worker,
    summarize_policy_vision_records,
    verify_derived_projection,
)
from causalcache.restoration_v2_2_policy_vision_contract import (
    CANONICAL_CONFIG_PATH,
    EXPECTED_OPERATION_CEILING,
    EXPECTED_OUTPUT_FILES,
    PROTOCOL_ID,
    RestorationV22PolicyVisionContract,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    strict_json_object_bytes,
)
from causalcache.restoration_v2_baselines import select_top_two


RUN_STATUS = STATUS
VALID_STATUS = "VALID_RESTORATION_V2_2_POLICY_VISION_BASELINE_V1"
V1_RUN_TOMBSTONE_STATUS = "INVALID_POLICY_VISION_V1_ZERO_FEATURE_GPU_UUID_TYPE"
FORMAL_SOURCE_PATHS = (
    CANONICAL_CONFIG_PATH,
    "code/configs/gui_owl_1_5_8b_snapshot.json",
    "data/manifests/restoration_v2_baselines.json",
    "code/causalcache/restoration_v2_2_policy_vision_contract.py",
    "code/causalcache/restoration_v2_2_policy_vision.py",
    "code/causalcache/policy/gui_owl_v2_2_vision_runtime.py",
    "code/causalcache/policy/gui_owl_v2_2_eager_runtime.py",
    "code/causalcache/policy/gui_owl_v2_vision.py",
    "code/scripts/validate_restoration_v2_2_policy_vision_contract.py",
    "code/scripts/run_restoration_v2_2_policy_vision_baseline.py",
)
FORMAL_PYTHON_SOURCE_ROOTS = ("code/causalcache", "code/scripts")
GPU_UUID_PATTERN = re.compile(
    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]+)+"
)
CONTAINER_NAME_PATTERN = re.compile(r"sglang-omni-jaxan-[0-9]{8}")
HOST_IDENTITY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
EXPECTED_HYPER_HOSTNAMES = {
    "hyper00": "node-radixark-16-0000",
    "hyper01": "node-radixark-16-0001",
}
HOST_EVIDENCE_COLLECTOR = "docker_inspect_and_nvidia_smi_on_host_v1"
AUDITED_SCIENTIFIC_ENVIRONMENT_VARIABLES = (
    "CUBLAS_WORKSPACE_CONFIG",
    "NVIDIA_TF32_OVERRIDE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "FLASH_ATTENTION_DETERMINISTIC",
    "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT",
    "TORCH_CUDNN_V8_API_DISABLED",
    "TORCH_CUDNN_V8_API_ENABLED",
)


@dataclass(frozen=True)
class PolicyVisionRunnerProtocol:
    """Protocol-specific identity threaded through the shared frozen runner."""

    protocol_id: str
    run_status: str
    valid_status: str
    contract_class: type[Any]
    expected_output_files: tuple[str, ...]
    expected_operation_ceiling: Mapping[str, int]
    formal_source_paths: tuple[str, ...]
    runtime_profile_id: str
    gpu_uuid_type_profile: str | None
    image_processor_size_profile: str | None
    readme_title: str
    formal_run_allowed: bool
    run_tombstone_status: str | None
    formal_attempt_ledger_path: str | None
    include_repair_identity: bool = False


V1_RUNNER_PROTOCOL = PolicyVisionRunnerProtocol(
    protocol_id=PROTOCOL_ID,
    run_status=RUN_STATUS,
    valid_status=VALID_STATUS,
    contract_class=RestorationV22PolicyVisionContract,
    expected_output_files=tuple(EXPECTED_OUTPUT_FILES),
    expected_operation_ceiling=EXPECTED_OPERATION_CEILING,
    formal_source_paths=FORMAL_SOURCE_PATHS,
    runtime_profile_id=GUI_OWL_V2_2_VISION_RUNTIME_ID,
    gpu_uuid_type_profile=None,
    image_processor_size_profile=None,
    readme_title="Restoration v2.2 policy-vision baseline v1",
    formal_run_allowed=False,
    run_tombstone_status=V1_RUN_TOMBSTONE_STATUS,
    formal_attempt_ledger_path=None,
)


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        canonical_json_bytes(dict(record)) + b"\n" for record in records
    )


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _validate_run_git(root: Path, source_commit: str) -> None:
    if _git(root, "rev-parse", "HEAD") != source_commit:
        raise ValueError("policy-vision source commit must equal HEAD")
    if _git(root, "symbolic-ref", "--short", "HEAD") != "main":
        raise ValueError("formal policy-vision execution must use main")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("formal policy-vision execution requires a clean worktree")
    if _git(root, "rev-parse", "origin/main") != source_commit:
        raise ValueError("formal policy-vision source must already be pushed")


def _validate_run_source_snapshot(
    root: Path,
    source_commit: str,
    *,
    formal_source_paths: Sequence[str] = FORMAL_SOURCE_PATHS,
) -> None:
    if (
        _git(root, "rev-parse", "HEAD") != source_commit
        or _git(root, "symbolic-ref", "--short", "HEAD") != "main"
        or _git(root, "rev-parse", "origin/main") != source_commit
    ):
        raise ValueError("formal policy-vision source snapshot drifted during run")
    _validate_source_unchanged(
        root,
        source_commit,
        formal_source_paths=formal_source_paths,
    )


def _formal_python_source_closure(
    root: Path,
    source_commit: str,
) -> dict[str, Any]:
    paths = tuple(
        path
        for path in _git(
            root,
            "ls-tree",
            "-r",
            "--name-only",
            source_commit,
            "--",
            *FORMAL_PYTHON_SOURCE_ROOTS,
        ).splitlines()
        if path.endswith(".py")
    )
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("formal policy-vision Python source closure is invalid")
    records = [
        {
            "path": path,
            "git_blob_sha": _git(root, "rev-parse", f"{source_commit}:{path}"),
        }
        for path in paths
    ]
    return {
        "rule": (
            "all_tracked_python_under_code_causalcache_and_code_scripts_"
            "at_source_commit"
        ),
        "path_count": len(paths),
        "inventory_sha256": sha256_bytes(canonical_json_bytes(records)),
        "paths": paths,
    }


def _validate_source_unchanged(
    root: Path,
    source_commit: str,
    *,
    formal_source_paths: Sequence[str] = FORMAL_SOURCE_PATHS,
) -> None:
    if (
        GIT_COMMIT_PATTERN.fullmatch(source_commit) is None
        or _git(root, "cat-file", "-t", source_commit) != "commit"
    ):
        raise ValueError("policy-vision source_git_commit is not a full local commit")
    for descendant in (
        _git(root, "rev-parse", "HEAD"),
        _git(root, "rev-parse", "origin/main"),
    ):
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, descendant],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise ValueError(
                "policy-vision source commit is not an ancestor of current main"
            )
    closure = _formal_python_source_closure(root, source_commit)
    current_python_paths = tuple(
        sorted(
            path
            for path in _git(
                root,
                "ls-files",
                "--",
                *FORMAL_PYTHON_SOURCE_ROOTS,
            ).splitlines()
            if path.endswith(".py")
        )
    )
    if current_python_paths != tuple(sorted(closure["paths"])):
        raise ValueError("formal policy-vision Python source closure drifted")
    protected = tuple(sorted(set(formal_source_paths) | set(closure["paths"])))
    if _git(root, "diff", "--name-only", source_commit, "--", *protected):
        raise ValueError("formal policy-vision source changed after execution")
    untracked = _git(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        *FORMAL_PYTHON_SOURCE_ROOTS,
    )
    if untracked:
        raise ValueError("formal policy-vision import roots contain untracked files")


def _normalize_gpu_uuid(value: str) -> str:
    if GPU_UUID_PATTERN.fullmatch(value) is None:
        raise ValueError("expected GPU UUID must use the canonical GPU-UUID form")
    return "GPU-" + value[4:].lower()


def _validate_host_evidence_record(
    evidence: Any,
    *,
    host_alias: str,
    host_hostname: str,
    container_hostname: str,
    container_name: str,
    container_image_digest: str,
    nvidia_driver_version: str,
    expected_gpu_uuids: Sequence[str],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "collector",
        "host_alias",
        "host_hostname",
        "container_id",
        "container_name",
        "container_image_digest",
        "nvidia_driver_version",
        "gpu_uuids",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_keys:
        raise ValueError("host-side evidence schema drifted")
    expected_hostname = EXPECTED_HYPER_HOSTNAMES.get(host_alias)
    container_id = evidence.get("container_id")
    if (
        expected_hostname is None
        or host_hostname != expected_hostname
        or evidence.get("schema_version") != "1.0.0"
        or evidence.get("collector") != HOST_EVIDENCE_COLLECTOR
        or evidence.get("host_alias") != host_alias
        or evidence.get("host_hostname") != host_hostname
        or not isinstance(container_id, str)
        or CONTAINER_ID_PATTERN.fullmatch(container_id) is None
        or not isinstance(container_hostname, str)
        or re.fullmatch(r"[0-9a-f]{12,64}", container_hostname) is None
        or not container_id.startswith(container_hostname)
        or evidence.get("container_name") != container_name
        or evidence.get("container_image_digest") != container_image_digest
        or evidence.get("nvidia_driver_version") != nvidia_driver_version
        or evidence.get("gpu_uuids") != list(expected_gpu_uuids)
    ):
        raise ValueError("host-side Docker/GPU evidence differs from formal identity")
    return _json_copy(evidence)


def _load_host_evidence(
    path: Path,
    *,
    host_alias: str,
    host_hostname: str,
    container_hostname: str,
    container_name: str,
    container_image_digest: str,
    nvidia_driver_version: str,
    expected_gpu_uuids: Sequence[str],
) -> tuple[dict[str, Any], str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("host-side evidence cannot be opened safely") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o022:
            raise ValueError(
                "host-side evidence must be regular and not group/other writable"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
    finally:
        os.close(descriptor)
    evidence = strict_json_object_bytes(payload, label="host-side evidence")
    validated = _validate_host_evidence_record(
        evidence,
        host_alias=host_alias,
        host_hostname=host_hostname,
        container_hostname=container_hostname,
        container_name=container_name,
        container_image_digest=container_image_digest,
        nvidia_driver_version=nvidia_driver_version,
        expected_gpu_uuids=expected_gpu_uuids,
    )
    return validated, sha256_bytes(payload)


def _claim_formal_attempt(path: Path, claim: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if not path.is_absolute() or not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("formal attempt ledger parent must be an existing real directory")
    detached = _json_copy(claim)
    payload = _pretty_json_bytes(detached)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            "formal policy-vision attempt was already claimed"
        ) from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("formal attempt ledger write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os,
        "O_DIRECTORY",
        0,
    )
    directory_descriptor = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return detached, sha256_bytes(payload)


def _read_verified_labels_bytes(
    contract: RestorationV22PolicyVisionContract,
    labels_archive: Path,
) -> bytes:
    labels = contract.data["immutable_inputs"]["restoration_labels"]
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        descriptor = os.open(labels_archive, flags)
    except OSError as error:
        raise ValueError("policy-vision labels archive cannot be opened safely") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("policy-vision labels archive must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
    finally:
        os.close(descriptor)
    if (
        len(payload) != labels["raw_archive_size_bytes"]
        or sha256_bytes(payload) != labels["raw_archive_sha256"]
    ):
        raise ValueError("policy-vision labels archive identity drifted")
    return payload


def _validate_labels_archive(
    contract: RestorationV22PolicyVisionContract,
    labels_archive: Path,
) -> None:
    _read_verified_labels_bytes(contract, labels_archive)


def _load_primary_labels_from_verified_bytes(
    *,
    contract: RestorationV22PolicyVisionContract,
    labels_archive: Path,
    work_items: Sequence[PolicyVisionWorkItem],
) -> Mapping[str, Any]:
    payload = _read_verified_labels_bytes(contract, labels_archive)
    with tempfile.TemporaryDirectory(
        prefix="causalcache-policy-vision-label-snapshot-"
    ) as temporary:
        snapshot = Path(temporary) / "verified-labels.tar"
        snapshot.write_bytes(payload)
        if sha256_bytes(snapshot.read_bytes()) != sha256_bytes(payload):
            raise RuntimeError("verified label snapshot write drifted")
        return load_primary_label_states(snapshot, work_items)


def _validate_snapshot_path(
    contract: RestorationV22PolicyVisionContract,
    snapshot_manifest: Path,
) -> None:
    relative = contract.data["immutable_inputs"]["model_snapshot"][
        "manifest_path"
    ]
    canonical = (contract.repository_root / relative).resolve()
    if (
        snapshot_manifest.resolve() != canonical
        or not canonical.is_file()
        or canonical.is_symlink()
    ):
        raise ValueError("policy-vision snapshot manifest path drifted")


def _run_feature_workers(
    *,
    work_items: Sequence[PolicyVisionWorkItem],
    derived_root: Path,
    derived_payload_prefix: str,
    expected_tar_member_count: int,
    model_dir: Path,
    snapshot_manifest: Path,
    expected_gpu_uuids: Sequence[str],
    gpu_uuid_type_profile: str | None = None,
    image_processor_size_profile: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Launch exactly two spawn workers without giving either worker D(S)."""
    if len(expected_gpu_uuids) != 2 or len(set(expected_gpu_uuids)) != 2:
        raise ValueError("policy-vision requires two distinct expected GPU UUIDs")
    even, odd = feature_worker_shards(work_items)
    assignments = (
        {
            "worker_id": "even",
            "device": "cuda:0",
            "expected_gpu_uuid": expected_gpu_uuids[0],
            "canonical_items": even,
            "sentinel_item": None,
        },
        {
            "worker_id": "odd",
            "device": "cuda:1",
            "expected_gpu_uuid": expected_gpu_uuids[1],
            "canonical_items": odd,
            "sentinel_item": work_items[0],
        },
    )
    common = {
        "derived_root": derived_root,
        "derived_payload_prefix": derived_payload_prefix,
        "expected_tar_member_count": expected_tar_member_count,
        "model_dir": model_dir,
        "snapshot_manifest": snapshot_manifest,
    }
    if gpu_uuid_type_profile is not None:
        common["gpu_uuid_type_profile"] = gpu_uuid_type_profile
    if image_processor_size_profile is not None:
        common["image_processor_size_profile"] = image_processor_size_profile
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
        futures = [
            executor.submit(run_policy_vision_worker, **assignment, **common)
            for assignment in assignments
        ]
        outputs = tuple(future.result() for future in futures)
    return outputs  # type: ignore[return-value]


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _expected_runtime_metadata(
    *,
    contract: RestorationV22PolicyVisionContract,
    device: str,
    gpu_uuid: str,
    runtime_profile_id: str = GUI_OWL_V2_2_VISION_RUNTIME_ID,
) -> dict[str, Any]:
    runtime = contract.data["execution_contract"]["runtime"]
    stack = runtime["expected_execution_stack"]
    snapshot = contract.data["immutable_inputs"]["model_snapshot"]
    baseline = contract.data["immutable_inputs"]["baseline_implementation"]
    manifest_path = contract.repository_root / baseline["manifest_path"]
    manifest = strict_json_object_bytes(
        manifest_path.read_bytes(),
        label="policy-vision baseline manifest",
    )
    policy_vision = manifest["implementation_contract"]["policy_vision"]
    controls = runtime["eager_control_flags"]
    pixels_per_image = 2_621_440
    return {
        "runtime_profile_id": runtime_profile_id,
        "device": device,
        "gpu_uuid": gpu_uuid,
        "gpu_name": stack["gpu_name"],
        "python_version": stack["python_version"],
        "torch_version": stack["torch_version"],
        "torch_cuda_version": stack["torch_cuda_version"],
        "cudnn_version": stack["cudnn_version"],
        "transformers_version": stack["transformers_version"],
        "pillow_version": stack["pillow_version"],
        "model_repo": snapshot["repo"],
        "model_revision": snapshot["revision"],
        "snapshot_manifest_sha256": snapshot["manifest_sha256"],
        "verified_model_file_count": snapshot["file_count"],
        "verified_model_total_bytes": snapshot["total_bytes"],
        "transformers_source_sha256": policy_vision[
            "transformers_source_sha256"
        ],
        "dtype": "torch.bfloat16",
        "frozen": True,
        "image_processor_class": "Qwen2VLImageProcessor",
        "model_class": contract.data["feature_contract"]["model_class"],
        "requested_attention_implementation": "eager",
        "observed_attention_implementation": {
            "top": "eager",
            "text": "eager",
            "vision": "eager",
        },
        "feature_only": True,
        "single_device": True,
        "target_effective_visual_tokens_per_image": 2560,
        "min_pixels": pixels_per_image,
        "max_pixels": pixels_per_image,
        "seed": controls["seed"],
        "deterministic_algorithms_enabled": False,
        "deterministic_warn_only_enabled": False,
        "cudnn_deterministic": controls["cudnn_deterministic"],
        "cudnn_benchmark": controls["cudnn_benchmark"],
        "cuda_matmul_allow_tf32": controls["cuda_matmul_allow_tf32"],
        "cudnn_allow_tf32": controls["cudnn_allow_tf32"],
        "float32_matmul_precision": controls["float32_matmul_precision"],
        "tokenizer_loaded": False,
        "chat_template_called": False,
        "strict_cuda_determinism_claimed": False,
        "numerical_control_claim": (
            "eager_fixed_seed_tf32_disabled_numerical_control_"
            "not_strict_cuda_determinism"
        ),
        "scientific_environment_variables_set_by_runtime": False,
        "scientific_environment_audit": {
            "audited_names": list(AUDITED_SCIENTIFIC_ENVIRONMENT_VARIABLES),
            "present_names": [],
            "all_absent": True,
        },
        "processor_interface": "AutoImageProcessor.__call__",
        "processor_output_keys": ["image_grid_thw", "pixel_values"],
        "processor_size": {
            "shortest_edge": pixels_per_image,
            "longest_edge": pixels_per_image,
        },
        "forbidden_operation_guards_installed": [
            "top_model_forward_count",
            "language_model_forward_count",
            "lm_head_forward_count",
            "generation_count",
        ],
    }


def _validate_runtime_metadata(
    *,
    metadata: Any,
    contract: RestorationV22PolicyVisionContract,
    worker_id: str,
    device: str,
    gpu_uuid: str,
    runtime_profile_id: str = GUI_OWL_V2_2_VISION_RUNTIME_ID,
) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("policy-vision runtime metadata is absent")
    runtime = contract.data["execution_contract"]["runtime"]
    expected_values = _expected_runtime_metadata(
        contract=contract,
        device=device,
        gpu_uuid=gpu_uuid,
        runtime_profile_id=runtime_profile_id,
    )
    dynamic_keys = {
        "gpu_pci_bus_id",
        "logical_device_index",
        "nvidia_smi_index",
    }
    if set(metadata) != set(expected_values) | dynamic_keys or any(
        metadata.get(key) != value for key, value in expected_values.items()
    ):
        raise ValueError(f"{worker_id} runtime metadata differs from contract")
    expected_index = 0 if worker_id == "even" else 1
    if metadata.get("logical_device_index") != expected_index:
        raise ValueError("policy-vision logical CUDA index drifted")
    for key in runtime["worker_identity_fields_required"]:
        if key not in metadata:
            raise ValueError(f"policy-vision runtime identity field is absent: {key}")
    if (
        not isinstance(metadata.get("gpu_pci_bus_id"), str)
        or re.fullmatch(
            r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",
            metadata["gpu_pci_bus_id"],
        )
        is None
    ):
        raise ValueError("policy-vision GPU PCI identity is invalid")
    if type(metadata.get("nvidia_smi_index")) is not int:
        raise ValueError("policy-vision nvidia-smi index is invalid")


def _validate_worker_outputs(
    *,
    outputs: Sequence[Mapping[str, Any]],
    contract: RestorationV22PolicyVisionContract,
    expected_gpu_uuids: Sequence[str],
    runtime_profile_id: str = GUI_OWL_V2_2_VISION_RUNTIME_ID,
) -> None:
    if len(outputs) != 2:
        raise ValueError("policy-vision must return exactly two worker outputs")
    by_id = {output.get("worker_id"): output for output in outputs}
    if set(by_id) != {"even", "odd"}:
        raise ValueError("policy-vision worker output identities drifted")
    for ordinal, worker_id in enumerate(("even", "odd")):
        output = by_id[worker_id]
        device = f"cuda:{ordinal}"
        uuid = expected_gpu_uuids[ordinal]
        if output.get("device") != device or output.get("gpu_uuid") != uuid:
            raise ValueError("policy-vision worker device binding drifted")
        _validate_runtime_metadata(
            metadata=output.get("runtime_metadata"),
            contract=contract,
            worker_id=worker_id,
            device=device,
            gpu_uuid=uuid,
            runtime_profile_id=runtime_profile_id,
        )
        counts = output.get("runtime_operation_counts")
        expected_counts = {
            "image_processor_batch_count": 8,
            "policy_vision_feature_forward_count": 16 if worker_id == "even" else 15,
            "top_model_forward_count": 0,
            "language_model_forward_count": 0,
            "lm_head_forward_count": 0,
            "generation_count": 0,
        }
        if counts != expected_counts:
            raise ValueError("policy-vision per-worker operation ledger drifted")
        expected_inputs = {
            "canonical_state_count": 8 if worker_id == "even" else 7,
            "sentinel_state_count": 0 if worker_id == "even" else 1,
            "image_payload_count": 40,
        }
        if output.get("worker_input_counts") != expected_inputs:
            raise ValueError("policy-vision worker input denominator drifted")
        memory = output.get("peak_cuda_memory")
        if not isinstance(memory, Mapping) or any(
            type(memory.get(key)) is not int or memory[key] <= 0
            for key in (
                "max_memory_allocated_bytes",
                "max_memory_reserved_bytes",
            )
        ):
            raise ValueError("policy-vision peak CUDA memory record is invalid")
        _finite_nonnegative(output.get("elapsed_seconds"), "worker elapsed time")


def _execution_record(
    *,
    contract: RestorationV22PolicyVisionContract,
    worker_outputs: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    host_alias: str,
    host_hostname: str,
    container_hostname: str,
    container_name: str,
    container_image_digest: str,
    nvidia_driver_version: str,
    expected_gpu_uuids: Sequence[str],
    host_evidence: Mapping[str, Any],
    host_evidence_sha256: str,
    formal_attempt_claim: Mapping[str, Any],
    formal_attempt_claim_sha256: str,
    runtime_profile_id: str = GUI_OWL_V2_2_VISION_RUNTIME_ID,
    expected_operation_ceiling: Mapping[str, int] = EXPECTED_OPERATION_CEILING,
) -> dict[str, Any]:
    runtime = contract.data["execution_contract"]["runtime"]
    expected_stack = runtime["expected_execution_stack"]
    if host_alias not in EXPECTED_HYPER_HOSTNAMES:
        raise ValueError("formal host alias must identify a Hyper H200 host")
    if host_hostname != EXPECTED_HYPER_HOSTNAMES[host_alias]:
        raise ValueError("formal host alias/hostname binding drifted")
    if (
        HOST_IDENTITY_PATTERN.fullmatch(container_hostname) is None
        or container_hostname != socket.gethostname()
    ):
        raise ValueError(
            "formal container hostname must equal socket.gethostname()"
        )
    if CONTAINER_NAME_PATTERN.fullmatch(container_name) is None:
        raise ValueError("formal container name violates the ownership convention")
    if container_image_digest != expected_stack["container_image_digest"]:
        raise ValueError("formal container image digest drifted")
    if nvidia_driver_version != expected_stack["nvidia_driver_version"]:
        raise ValueError("formal NVIDIA driver version drifted")
    validated_evidence = _validate_host_evidence_record(
        host_evidence,
        host_alias=host_alias,
        host_hostname=host_hostname,
        container_hostname=container_hostname,
        container_name=container_name,
        container_image_digest=container_image_digest,
        nvidia_driver_version=nvidia_driver_version,
        expected_gpu_uuids=expected_gpu_uuids,
    )
    if (
        not isinstance(host_evidence_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", host_evidence_sha256) is None
    ):
        raise ValueError("host-side evidence SHA256 is invalid")
    detached_attempt_claim = _json_copy(formal_attempt_claim)
    if (
        not isinstance(formal_attempt_claim_sha256, str)
        or sha256_bytes(_pretty_json_bytes(detached_attempt_claim))
        != formal_attempt_claim_sha256
    ):
        raise ValueError("formal attempt claim SHA256 is invalid")
    observed_drivers = {
        line.strip()
        for line in subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
        if line.strip()
    }
    if observed_drivers != {nvidia_driver_version}:
        raise ValueError("nvidia-smi driver identity differs from formal provenance")
    _validate_worker_outputs(
        outputs=worker_outputs,
        contract=contract,
        expected_gpu_uuids=expected_gpu_uuids,
        runtime_profile_id=runtime_profile_id,
    )
    replay_record = _json_copy(replay)
    by_id = {output["worker_id"]: output for output in worker_outputs}
    sentinel = by_id["odd"]["cross_device_sentinel_record"]
    if not isinstance(sentinel, Mapping):
        raise ValueError("odd worker omitted the cross-device sentinel")
    cross = replay_record["cross_device_sentinel"]
    cross.update(
        {
            "verification_scores_by_event_step": _json_copy(
                sentinel["scores_by_event_step"]
            ),
            "verification_ranked_event_step_ids": list(
                sentinel["ranked_event_step_ids"]
            ),
            "verification_selected_coalition": list(
                sentinel["selected_coalition"]
            ),
            "verification_image_grid_thw": _json_copy(
                sentinel["image_grid_thw"]
            ),
            "verification_merged_token_counts": list(
                sentinel["merged_token_counts"]
            ),
        }
    )
    return {
        "host_runtime": {
            "host_class": "Hyper_H200",
            "host_alias": host_alias,
            "host_hostname": host_hostname,
            "container_hostname": container_hostname,
            "container_name": container_name,
            "container_image_digest": container_image_digest,
            "nvidia_driver_version": nvidia_driver_version,
            "host_evidence_collector": validated_evidence["collector"],
            "host_evidence_sha256": host_evidence_sha256,
            "host_container_id": validated_evidence["container_id"],
        },
        "worker_bindings": [
            {
                "worker_id": worker_id,
                "logical_device": f"cuda:{ordinal}",
                "expected_gpu_uuid": expected_gpu_uuids[ordinal],
            }
            for ordinal, worker_id in enumerate(("even", "odd"))
        ],
        "verification": replay_record,
        "operation_ledger": dict(expected_operation_ceiling),
        "formal_attempt_claim": {
            "record": detached_attempt_claim,
            "sha256": formal_attempt_claim_sha256,
        },
        "selection_completed_before_restoration_labels_loaded": True,
        "feature_worker_received_restoration_labels": False,
        "feature_worker_received_goal_text_or_ocr": False,
    }


def _feature_record_from_evaluated_row(row: Mapping[str, Any]) -> dict[str, Any]:
    state = row.get("state")
    worker = row.get("worker")
    current = row.get("current_image")
    candidates = row.get("candidate_scores")
    replay = row.get("score_replay")
    if (
        not isinstance(state, Mapping)
        or not isinstance(worker, Mapping)
        or not isinstance(current, Mapping)
        or not isinstance(candidates, list)
        or len(candidates) != 4
        or not isinstance(replay, Mapping)
    ):
        raise ValueError("recorded policy-vision state row schema drifted")
    if (
        replay.get("absolute_tolerance") != 1e-6
        or replay.get("ranking_equal") is not True
        or replay.get("selection_equal") is not True
        or _finite_nonnegative(
            replay.get("max_abs_score_difference"),
            "same-device replay difference",
        )
        > 1e-6
    ):
        raise ValueError("recorded same-device replay failed")
    repeat_results = replay.get("repeat_results")
    if not isinstance(repeat_results, list) or len(repeat_results) != 2:
        raise ValueError("recorded same-device replay must contain two repeats")
    canonical_scores = {
        str(candidate.get("event_step_id")): candidate.get(
            "policy_vision_cosine"
        )
        for candidate in candidates
    }
    repeat_scores = []
    for index, repeat in enumerate(repeat_results):
        if (
            not isinstance(repeat, Mapping)
            or repeat.get("repeat_index") != index
            or not isinstance(repeat.get("scores_by_event_step"), Mapping)
        ):
            raise ValueError("recorded feature repeat schema drifted")
        scores = repeat["scores_by_event_step"]
        if set(scores) != {"1", "2", "3", "4"}:
            raise ValueError("recorded feature repeat score keys drifted")
        numeric_scores = {key: float(value) for key, value in scores.items()}
        selection = select_top_two(
            {int(key): value for key, value in numeric_scores.items()}
        )
        norm_range = repeat.get("normalized_embedding_norm_range")
        if (
            repeat.get("ranked_event_step_ids")
            != list(selection.ranked_event_step_ids)
            or repeat.get("selected_event_step_ids")
            != list(selection.selected_event_step_ids)
            or not isinstance(norm_range, Mapping)
            or set(norm_range)
            != {
                "minimum",
                "maximum",
                "maximum_absolute_deviation_from_one",
            }
            or any(
                not math.isfinite(float(norm_range[key])) for key in norm_range
            )
        ):
            raise ValueError("recorded feature repeat selection or norm drifted")
        repeat_scores.append(numeric_scores)
    if repeat_scores[0] != {
        key: float(value) for key, value in canonical_scores.items()
    }:
        raise ValueError("recorded canonical scores differ from repeat zero")
    observed_replay = max(
        abs(repeat_scores[0][key] - repeat_scores[1][key])
        for key in ("1", "2", "3", "4")
    )
    if observed_replay != float(replay["max_abs_score_difference"]):
        raise ValueError("recorded same-device replay difference is inconsistent")
    if any(
        repeat["ranked_event_step_ids"] != row.get("ranked_event_step_ids")
        or repeat["selected_event_step_ids"] != row.get("selected_coalition")
        for repeat in repeat_results
    ):
        raise ValueError("recorded same-device replay changed ranking or selection")
    return {
        "state": dict(state),
        "worker_id": worker.get("worker_id"),
        "device": worker.get("device"),
        "gpu_uuid": worker.get("gpu_uuid"),
        "candidate_scores": [
            {
                key: candidate[key]
                for key in (
                    "event_step_id",
                    "post_image_member_path",
                    "post_image_sha256",
                    "policy_vision_cosine",
                    "image_grid_thw",
                    "merged_token_count",
                )
            }
            for candidate in candidates
        ],
        "current_image_member_path": current.get("image_member_path"),
        "current_image_sha256": current.get("image_sha256"),
        "current_image_grid_thw": current.get("image_grid_thw"),
        "current_merged_token_count": current.get("merged_token_count"),
        "ranked_event_step_ids": row.get("ranked_event_step_ids"),
        "selected_coalition": row.get("selected_coalition"),
        "score_replay": dict(replay),
    }


def _validate_preprocessing_geometry(
    feature_records: Sequence[Mapping[str, Any]],
    contract: RestorationV22PolicyVisionContract,
) -> None:
    if len(feature_records) != 15:
        raise ValueError("policy-vision preprocessing requires 15 canonical states")
    grid_histogram: Counter[str] = Counter()
    state_raw_rows: Counter[str] = Counter()
    raw_total = 0
    merged_total = 0
    for record in feature_records:
        candidates = record.get("candidate_scores")
        if not isinstance(candidates, list) or len(candidates) != 4:
            raise ValueError("policy-vision preprocessing candidate rows drifted")
        image_rows = [
            *[
                (candidate.get("image_grid_thw"), candidate.get("merged_token_count"))
                for candidate in candidates
            ],
            (
                record.get("current_image_grid_thw"),
                record.get("current_merged_token_count"),
            ),
        ]
        state_raw = 0
        for grid, merged in image_rows:
            if (
                not isinstance(grid, list)
                or len(grid) != 3
                or any(type(value) is not int or value <= 0 for value in grid)
                or type(merged) is not int
                or merged <= 0
            ):
                raise ValueError("policy-vision preprocessing geometry is invalid")
            raw = math.prod(grid)
            if raw % 4 != 0 or merged != raw // 4:
                raise ValueError("policy-vision merged-token geometry drifted")
            grid_histogram[",".join(str(value) for value in grid)] += 1
            state_raw += raw
            raw_total += raw
            merged_total += merged
        state_raw_rows[str(state_raw)] += 1
    expected = contract.data["execution_contract"]["preprocessing"]
    if dict(sorted(grid_histogram.items())) != dict(
        sorted(expected["grid_thw_histogram_over_75_canonical_images"].items())
    ):
        raise ValueError("policy-vision canonical grid histogram drifted")
    if dict(sorted(state_raw_rows.items())) != dict(
        sorted(
            expected[
                "state_pixel_value_row_histogram_over_15_canonical_batches"
            ].items()
        )
    ):
        raise ValueError("policy-vision per-state raw-patch histogram drifted")
    if (
        raw_total != expected["canonical_raw_patch_rows"]
        or merged_total != expected["canonical_merged_visual_tokens"]
    ):
        raise ValueError("policy-vision canonical visual-token total drifted")


def _validate_cross_device_replay(
    cross: Any,
    feature_records: Sequence[Mapping[str, Any]],
) -> None:
    rows_by_state = {
        record["state"]["state_id"]: record for record in feature_records
    }
    if not isinstance(cross, Mapping):
        raise ValueError("recorded cross-device sentinel is absent")
    sentinel_id = "0131649930078879:decision_step:006"
    if (
        cross.get("primary_ordinal") != 0
        or cross.get("state_index") != 2
        or cross.get("state_id") != sentinel_id
        or cross.get("source_worker") != "even"
        or cross.get("verification_worker") != "odd"
        or cross.get("ranking_equal") is not True
        or cross.get("selection_equal") is not True
        or cross.get("absolute_tolerance") != 1e-6
        or sentinel_id not in rows_by_state
    ):
        raise ValueError("recorded cross-device sentinel identity drifted")
    source = rows_by_state[sentinel_id]
    source_scores = {
        str(row["event_step_id"]): float(row["policy_vision_cosine"])
        for row in source["candidate_scores"]
    }
    verification_scores = cross.get("verification_scores_by_event_step")
    differences = cross.get("per_event_absolute_score_difference")
    if (
        not isinstance(verification_scores, Mapping)
        or set(verification_scores) != set(source_scores)
        or not isinstance(differences, Mapping)
        or set(differences) != set(source_scores)
    ):
        raise ValueError("recorded cross-device score schema drifted")
    observed = {
        key: abs(source_scores[key] - float(verification_scores[key]))
        for key in sorted(source_scores)
    }
    if observed != {key: float(differences[key]) for key in sorted(differences)}:
        raise ValueError("recorded cross-device score differences are inconsistent")
    maximum = max(observed.values())
    if maximum != float(cross.get("maximum_absolute_score_difference", -1)):
        raise ValueError("recorded cross-device maximum difference is inconsistent")
    if maximum > 1e-6:
        raise ValueError("recorded cross-device sentinel exceeds tolerance")
    selection = select_top_two(
        {int(key): float(value) for key, value in verification_scores.items()}
    )
    if (
        cross.get("verification_ranked_event_step_ids")
        != list(selection.ranked_event_step_ids)
        or cross.get("verification_selected_coalition")
        != list(selection.selected_event_step_ids)
        or cross["verification_ranked_event_step_ids"]
        != source["ranked_event_step_ids"]
        or cross["verification_selected_coalition"]
        != source["selected_coalition"]
    ):
        raise ValueError("recorded cross-device ranking or selection drifted")
    if (
        cross.get("verification_image_grid_thw")
        != [
            *[candidate["image_grid_thw"] for candidate in source["candidate_scores"]],
            source["current_image_grid_thw"],
        ]
        or cross.get("verification_merged_token_counts")
        != [
            *[
                candidate["merged_token_count"]
                for candidate in source["candidate_scores"]
            ],
            source["current_merged_token_count"],
        ]
    ):
        raise ValueError("recorded cross-device preprocessing geometry drifted")


def _validate_formal_attempt_claim(
    raw_claim: Any,
    *,
    contract: RestorationV22PolicyVisionContract,
    source_git_commit: str,
    host: Mapping[str, Any],
    expected_gpu_uuids: Sequence[str],
) -> None:
    if not isinstance(raw_claim, Mapping) or set(raw_claim) != {"record", "sha256"}:
        raise ValueError("recorded formal attempt claim schema drifted")
    record = raw_claim["record"]
    digest = raw_claim["sha256"]
    expected = {
        "schema_version": "1.0.0",
        "status": "CLAIMED_POLICY_VISION_FORMAL_ATTEMPT",
        "protocol_id": contract.data["protocol_id"],
        "source_git_commit": source_git_commit,
        "contract_sha256": contract.sha256,
        "canonical_result_directory": contract.data["output_contract"][
            "canonical_result_directory"
        ],
        "formal_attempt_ledger_path": contract.data["output_contract"][
            "formal_attempt_ledger_path"
        ],
        "host_alias": host["host_alias"],
        "host_hostname": host["host_hostname"],
        "container_hostname": host["container_hostname"],
        "container_name": host["container_name"],
        "container_image_digest": host["container_image_digest"],
        "nvidia_driver_version": host["nvidia_driver_version"],
        "expected_gpu_uuids": list(expected_gpu_uuids),
    }
    if record != expected or digest != sha256_bytes(_pretty_json_bytes(expected)):
        raise ValueError("recorded formal attempt claim identity drifted")


def _validate_recorded_execution(
    *,
    execution: Any,
    contract: RestorationV22PolicyVisionContract,
    feature_records: Sequence[Mapping[str, Any]],
    source_git_commit: str,
    expected_operation_ceiling: Mapping[str, int] = EXPECTED_OPERATION_CEILING,
    runtime_profile_id: str = GUI_OWL_V2_2_VISION_RUNTIME_ID,
) -> None:
    expected_keys = {
        "host_runtime",
        "worker_bindings",
        "verification",
        "operation_ledger",
        "formal_attempt_claim",
        "selection_completed_before_restoration_labels_loaded",
        "feature_worker_received_restoration_labels",
        "feature_worker_received_goal_text_or_ocr",
    }
    if not isinstance(execution, Mapping) or set(execution) != expected_keys:
        raise ValueError("recorded policy-vision execution schema drifted")
    if (
        execution["selection_completed_before_restoration_labels_loaded"]
        is not True
        or execution["feature_worker_received_restoration_labels"] is not False
        or execution["feature_worker_received_goal_text_or_ocr"] is not False
        or execution["operation_ledger"] != expected_operation_ceiling
    ):
        raise ValueError("recorded policy-vision execution boundary drifted")
    host = execution["host_runtime"]
    expected_stack = contract.data["execution_contract"]["runtime"][
        "expected_execution_stack"
    ]
    if not isinstance(host, Mapping) or set(host) != {
        "host_class",
        "host_alias",
        "host_hostname",
        "container_hostname",
        "container_name",
        "container_image_digest",
        "nvidia_driver_version",
        "host_evidence_collector",
        "host_evidence_sha256",
        "host_container_id",
    }:
        raise ValueError("recorded host provenance schema drifted")
    if (
        host["host_class"] != "Hyper_H200"
        or host["host_alias"] not in EXPECTED_HYPER_HOSTNAMES
        or host["host_hostname"]
        != EXPECTED_HYPER_HOSTNAMES.get(host["host_alias"])
        or not isinstance(host["container_hostname"], str)
        or re.fullmatch(r"[0-9a-f]{12,64}", host["container_hostname"])
        is None
        or CONTAINER_NAME_PATTERN.fullmatch(host["container_name"]) is None
        or host["container_image_digest"]
        != expected_stack["container_image_digest"]
        or host["nvidia_driver_version"]
        != expected_stack["nvidia_driver_version"]
        or host["host_evidence_collector"] != HOST_EVIDENCE_COLLECTOR
        or not isinstance(host["host_evidence_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", host["host_evidence_sha256"])
        is None
        or not isinstance(host["host_container_id"], str)
        or CONTAINER_ID_PATTERN.fullmatch(host["host_container_id"]) is None
        or not host["host_container_id"].startswith(host["container_hostname"])
    ):
        raise ValueError("recorded host provenance drifted")
    bindings = execution["worker_bindings"]
    if not isinstance(bindings, list) or len(bindings) != 2:
        raise ValueError("recorded worker bindings drifted")
    uuids = []
    for ordinal, (binding, worker_id) in enumerate(
        zip(bindings, ("even", "odd"), strict=True)
    ):
        if not isinstance(binding, Mapping) or binding.get("worker_id") != worker_id:
            raise ValueError("recorded worker binding identity drifted")
        if binding.get("logical_device") != f"cuda:{ordinal}":
            raise ValueError("recorded worker logical device drifted")
        uuid = binding.get("expected_gpu_uuid")
        if not isinstance(uuid, str) or _normalize_gpu_uuid(uuid) != uuid:
            raise ValueError("recorded worker GPU UUID drifted")
        uuids.append(uuid)
    if len(set(uuids)) != 2:
        raise ValueError("recorded workers must bind two distinct GPUs")
    _validate_formal_attempt_claim(
        execution["formal_attempt_claim"],
        contract=contract,
        source_git_commit=source_git_commit,
        host=host,
        expected_gpu_uuids=uuids,
    )
    verification = execution["verification"]
    if not isinstance(verification, Mapping):
        raise ValueError("recorded replay verification is absent")
    same = verification.get("same_device")
    if (
        not isinstance(same, Mapping)
        or same.get("state_count") != 15
        or same.get("all_rankings_equal") is not True
        or same.get("all_selections_equal") is not True
        or same.get("absolute_tolerance") != 1e-6
        or _finite_nonnegative(
            same.get("maximum_absolute_score_difference"),
            "recorded same-device maximum difference",
        )
        > 1e-6
    ):
        raise ValueError("recorded same-device verification drifted")
    _validate_cross_device_replay(
        verification.get("cross_device_sentinel"),
        feature_records,
    )
    expected_runtime_counts = {
        "image_processor_batch_count": 16,
        "policy_vision_feature_forward_count": 31,
        "top_model_forward_count": 0,
        "language_model_forward_count": 0,
        "lm_head_forward_count": 0,
        "generation_count": 0,
    }
    if verification.get("runtime_operation_counts") != expected_runtime_counts:
        raise ValueError("recorded runtime operation ledger drifted")
    workers = verification.get("workers")
    if not isinstance(workers, list) or len(workers) != 2:
        raise ValueError("recorded runtime worker provenance drifted")
    for ordinal, (worker, worker_id) in enumerate(
        zip(workers, ("even", "odd"), strict=True)
    ):
        if not isinstance(worker, Mapping):
            raise ValueError("recorded runtime worker entry is invalid")
        if (
            worker.get("worker_id") != worker_id
            or worker.get("device") != f"cuda:{ordinal}"
            or worker.get("gpu_uuid") != uuids[ordinal]
        ):
            raise ValueError("recorded runtime worker binding drifted")
        _validate_runtime_metadata(
            metadata=worker.get("runtime_metadata"),
            contract=contract,
            worker_id=worker_id,
            device=f"cuda:{ordinal}",
            gpu_uuid=uuids[ordinal],
            runtime_profile_id=runtime_profile_id,
        )
        expected_inputs = {
            "canonical_state_count": 8 if worker_id == "even" else 7,
            "sentinel_state_count": 0 if worker_id == "even" else 1,
            "image_payload_count": 40,
        }
        if worker.get("worker_input_counts") != expected_inputs:
            raise ValueError("recorded worker input denominator drifted")
        memory = worker.get("peak_cuda_memory")
        if not isinstance(memory, Mapping) or any(
            type(memory.get(key)) is not int or memory[key] <= 0
            for key in (
                "max_memory_allocated_bytes",
                "max_memory_reserved_bytes",
            )
        ):
            raise ValueError("recorded peak CUDA memory is invalid")
        _finite_nonnegative(worker.get("elapsed_seconds"), "recorded elapsed time")


def _without_elapsed(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_elapsed(item)
            for key, item in value.items()
            if key != "elapsed_seconds"
        }
    if isinstance(value, list):
        return [_without_elapsed(item) for item in value]
    if isinstance(value, tuple):
        return [_without_elapsed(item) for item in value]
    return value


def _input_identity(contract: RestorationV22PolicyVisionContract) -> dict[str, Any]:
    return _json_copy(contract.data["immutable_inputs"])


def _readme(
    summary: Mapping[str, Any],
    *,
    title: str = "Restoration v2.2 policy-vision baseline v1",
) -> bytes:
    aggregate = summary["aggregate"]["by_role"]
    rows = []
    for role, label in (
        ("v2_label_train", "Train"),
        ("v2_development", "Development"),
        ("overall_stratified", "Overall"),
    ):
        row = aggregate[role]
        rows.append(
            f"| {label} | {row['state_count']} | "
            f"{row['mean_normalized_recovery']:.6f} | "
            f"{row['exact_coalition_match_rate']:.6f} | "
            f"{row['exact_cardinality_coalition_match_rate']:.6f} |"
        )
    outlier = summary["aggregate"]["pre_registered_outlier"]
    text = f"""# {title}

本结果完成 frozen GUI-Owl policy-vision feature-only comparator。15 条 primary `n=4,B=2`
train/development states 只把 event 1--4 的 post-action screenshot 与 decision-step-6 current screenshot
送入 image processor 和 final main vision merger；feature workers 从未接收 goal、text、OCR 或 restoration
labels。两个 H200 worker 分别固定到 `cuda:0`/`cuda:1`，并通过全部 same-device replay 与一条
cross-device sentinel replay。language-model/policy forward、LM head、generation、teacher/KL、gate、
matched-NLL、closed-loop 与 confirm/test access 全部为 0。

| Split | States | Mean recovery | At-most-B exact match | Exact-B match |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

预注册 outlier `{outlier['trajectory_id']}` 的 recovery 为
`{outlier['normalized_recovery']:.6f}`；负值若出现会原样保留，不 clamp、不删除。

- source commit: `{summary['source_execution']['source_git_commit']}`
- contract SHA256: `{summary['source_execution']['contract_sha256']}`
- scientific payload SHA256: `{summary['scientific_payload_sha256']}`
- state score rows: `{summary['state_scores']['record_count']}` (`{summary['state_scores']['sha256']}`)
- worker GPUs: `{summary['execution']['worker_bindings'][0]['expected_gpu_uuid']}`, `{summary['execution']['worker_bindings'][1]['expected_gpu_uuid']}`
"""
    return text.encode("utf-8")


def _source_execution(
    *,
    contract: RestorationV22PolicyVisionContract,
    source_commit: str,
    protocol: PolicyVisionRunnerProtocol = V1_RUNNER_PROTOCOL,
) -> dict[str, Any]:
    source_closure = _formal_python_source_closure(
        contract.repository_root,
        source_commit,
    )
    result = {
        "source_git_commit": source_commit,
        "contract_sha256": contract.sha256,
        "formal_source_paths": list(protocol.formal_source_paths),
        "formal_python_source_closure": {
            key: source_closure[key]
            for key in ("rule", "path_count", "inventory_sha256")
        },
    }
    if protocol.include_repair_identity:
        repair_identity = getattr(contract, "repair_identity", None)
        if (
            not isinstance(repair_identity, Mapping)
            or contract.data.get("repair_identity") != repair_identity
        ):
            raise ValueError("policy-vision repair identity is absent or inconsistent")
        result["repair_identity"] = _json_copy(repair_identity)
    return result


def _expected_files_from_features(
    *,
    contract: RestorationV22PolicyVisionContract,
    labels_archive: Path,
    source_commit: str,
    work_items: Sequence[PolicyVisionWorkItem],
    witness_by_state: Mapping[str, Mapping[str, Any]],
    feature_records: Sequence[Mapping[str, Any]],
    execution: Mapping[str, Any],
    protocol: PolicyVisionRunnerProtocol = V1_RUNNER_PROTOCOL,
) -> dict[str, bytes]:
    _validate_preprocessing_geometry(feature_records, contract)
    _validate_recorded_execution(
        execution=execution,
        contract=contract,
        feature_records=feature_records,
        source_git_commit=source_commit,
        expected_operation_ceiling=protocol.expected_operation_ceiling,
        runtime_profile_id=protocol.runtime_profile_id,
    )
    primary_labels = _load_primary_labels_from_verified_bytes(
        contract=contract,
        labels_archive=labels_archive,
        work_items=work_items,
    )
    records = evaluate_feature_records(
        feature_records=feature_records,
        work_items=work_items,
        witness_by_state=witness_by_state,
        primary_label_states=primary_labels,
        protocol_id=protocol.protocol_id,
    )
    aggregate = summarize_policy_vision_records(
        records,
        operation_counts=protocol.expected_operation_ceiling,
    )
    record_bytes = _jsonl_bytes(records)
    input_identity = _input_identity(contract)
    source_execution = _source_execution(
        contract=contract,
        source_commit=source_commit,
        protocol=protocol,
    )
    scientific_payload = {
        "protocol_id": protocol.protocol_id,
        "source_execution": source_execution,
        "input_identity": input_identity,
        "feature_contract": _json_copy(contract.data["feature_contract"]),
        "statistics_contract": _json_copy(contract.data["statistics_contract"]),
        "execution_without_elapsed_seconds": _without_elapsed(execution),
        "aggregate": aggregate,
        "state_scores": list(records),
    }
    summary = {
        "schema_version": "1.0.0",
        "status": protocol.run_status,
        "protocol_id": protocol.protocol_id,
        "source_execution": source_execution,
        "input_identity": input_identity,
        "execution": _json_copy(execution),
        "aggregate": aggregate,
        "scientific_payload_sha256": sha256_bytes(
            canonical_json_bytes(scientific_payload)
        ),
        "state_scores": {
            "path": "state_scores.jsonl",
            "record_count": len(records),
            "candidate_score_count": sum(
                len(record["candidate_scores"]) for record in records
            ),
            "size_bytes": len(record_bytes),
            "sha256": sha256_bytes(record_bytes),
        },
    }
    return {
        "README.md": _readme(summary, title=protocol.readme_title),
        "state_scores.jsonl": record_bytes,
        "summary.json": _pretty_json_bytes(summary),
    }


def _read_recorded_rows(output_dir: Path) -> tuple[dict[str, Any], ...]:
    path = output_dir / "state_scores.jsonl"
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("recorded policy-vision state_scores must be JSONL")
    rows = []
    for index, line in enumerate(payload.splitlines(keepends=True)):
        if not line.endswith(b"\n"):
            raise ValueError("recorded policy-vision JSONL termination drifted")
        row = strict_json_object_bytes(
            line[:-1],
            label=f"recorded policy-vision row {index}",
        )
        if canonical_json_bytes(row) + b"\n" != line:
            raise ValueError("recorded policy-vision JSONL is not canonical")
        rows.append(row)
    if len(rows) != 15:
        raise ValueError("recorded policy-vision state denominator drifted")
    return tuple(rows)


def _write_new_result(
    output_dir: Path,
    files: Mapping[str, bytes],
    *,
    pre_publish_check: Callable[[], None] | None = None,
    expected_output_files: Sequence[str] = EXPECTED_OUTPUT_FILES,
) -> None:
    if output_dir.exists():
        raise FileExistsError("canonical policy-vision output already exists")
    staging = output_dir.with_name(f".{output_dir.name}.staging")
    if staging.exists():
        raise FileExistsError("policy-vision staging output already exists")
    staging.mkdir(parents=False)
    try:
        for name in expected_output_files:
            (staging / name).write_bytes(files[name])
        if pre_publish_check is not None:
            pre_publish_check()
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_existing_result(
    output_dir: Path,
    files: Mapping[str, bytes],
    *,
    expected_output_files: Sequence[str] = EXPECTED_OUTPUT_FILES,
) -> None:
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError("canonical policy-vision output is missing or symlinked")
    children = tuple(output_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise ValueError("policy-vision output contains a non-regular file")
    if sorted(path.name for path in children) != sorted(expected_output_files):
        raise ValueError("policy-vision output exact-three inventory drifted")
    for name in expected_output_files:
        if (output_dir / name).read_bytes() != files[name]:
            raise ValueError(f"policy-vision output bytes drifted: {name}")


def _validate_mode(
    *,
    contract: RestorationV22PolicyVisionContract,
    output_dir: Path,
    labels_archive: Path,
    source_commit: str,
    protocol: PolicyVisionRunnerProtocol = V1_RUNNER_PROTOCOL,
) -> dict[str, bytes]:
    """Rebuild every scientific field from recorded scores, labels, and witness."""
    _validate_source_unchanged(
        contract.repository_root,
        source_commit,
        formal_source_paths=protocol.formal_source_paths,
    )
    _validate_labels_archive(contract, labels_archive)
    rows = _read_recorded_rows(output_dir)
    feature_records = tuple(_feature_record_from_evaluated_row(row) for row in rows)
    summary = strict_json_object_bytes(
        (output_dir / "summary.json").read_bytes(),
        label="recorded policy-vision summary",
    )
    if (
        summary.get("status") != protocol.run_status
        or summary.get("protocol_id") != protocol.protocol_id
        or summary.get("source_execution", {}).get("source_git_commit")
        != source_commit
    ):
        raise ValueError("recorded policy-vision summary identity drifted")
    work_items, witness_by_state = load_identity_witness(contract)
    files = _expected_files_from_features(
        contract=contract,
        labels_archive=labels_archive,
        source_commit=source_commit,
        work_items=work_items,
        witness_by_state=witness_by_state,
        feature_records=feature_records,
        execution=summary["execution"],
        protocol=protocol,
    )
    _validate_existing_result(
        output_dir,
        files,
        expected_output_files=protocol.expected_output_files,
    )
    return files


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "validate"))
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--labels-archive", type=Path, required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--derived-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--expected-gpu-uuid", action="append", default=[])
    parser.add_argument("--host-alias")
    parser.add_argument("--host-hostname")
    parser.add_argument("--container-hostname")
    parser.add_argument("--container-name")
    parser.add_argument("--container-image-digest")
    parser.add_argument("--nvidia-driver-version")
    parser.add_argument("--host-evidence", type=Path)
    parser.add_argument("--attempt-ledger", type=Path)
    return parser


def _validate_runner_protocol(
    protocol: PolicyVisionRunnerProtocol,
    contract: RestorationV22PolicyVisionContract,
) -> None:
    output_contract = contract.data["output_contract"]
    tombstone = protocol.run_tombstone_status
    try:
        contract_relative_path = contract.path.relative_to(
            contract.repository_root
        ).as_posix()
    except ValueError as error:
        raise ValueError("policy-vision contract is outside the repository") from error
    if (
        contract.data.get("protocol_id") != protocol.protocol_id
        or tuple(output_contract["exact_files"]) != protocol.expected_output_files
        or contract.data.get("operation_ceiling")
        != protocol.expected_operation_ceiling
        or not protocol.formal_source_paths
        or len(set(protocol.formal_source_paths))
        != len(protocol.formal_source_paths)
        or contract_relative_path not in protocol.formal_source_paths
        or protocol.formal_run_allowed == (tombstone is not None)
        or (
            tombstone is not None
            and (
                not tombstone
                or re.fullmatch(r"[A-Z0-9_]+", tombstone) is None
            )
        )
    ):
        raise ValueError("policy-vision runner protocol identity drifted")
    if protocol.gpu_uuid_type_profile is None:
        if (
            protocol.runtime_profile_id != GUI_OWL_V2_2_VISION_RUNTIME_ID
            or protocol.image_processor_size_profile is not None
            or protocol.include_repair_identity
            or protocol.formal_run_allowed
            or protocol.run_tombstone_status != V1_RUN_TOMBSTONE_STATUS
            or protocol.formal_attempt_ledger_path is not None
        ):
            raise ValueError("default policy-vision runtime profile drifted")
        return
    repair_identity = getattr(contract, "repair_identity", None)
    image_processor_profile_is_repaired = (
        protocol.image_processor_size_profile is not None
    )
    if (
        protocol.gpu_uuid_type_profile == DEFAULT_GPU_UUID_TYPE_PROFILE
        or protocol.image_processor_size_profile
        == DEFAULT_IMAGE_PROCESSOR_SIZE_PROFILE
        or not protocol.include_repair_identity
        or not isinstance(repair_identity, Mapping)
        or repair_identity.get("gpu_uuid_runtime_type_profile")
        != protocol.gpu_uuid_type_profile
        or repair_identity.get("run_status") != protocol.run_status
        or repair_identity.get("valid_status") != protocol.valid_status
        or repair_identity.get("canonical_output_directory")
        != output_contract["canonical_result_directory"]
        or output_contract.get("run_status") != protocol.run_status
        or output_contract.get("valid_status") != protocol.valid_status
        or not isinstance(protocol.formal_attempt_ledger_path, str)
        or not Path(protocol.formal_attempt_ledger_path).is_absolute()
        or output_contract.get("formal_attempt_ledger_path")
        != protocol.formal_attempt_ledger_path
        or (
            image_processor_profile_is_repaired
            and (
                repair_identity.get(
                    "image_processor_size_runtime_type_profile"
                )
                != protocol.image_processor_size_profile
                or repair_identity.get("combined_runtime_profile_id")
                != protocol.runtime_profile_id
            )
        )
    ):
        raise ValueError("repaired policy-vision runner identity drifted")


def run_protocol_main(
    protocol: PolicyVisionRunnerProtocol,
    argv: Sequence[str] | None = None,
) -> None:
    args = _parser().parse_args(argv)
    if args.mode == "run" and not protocol.formal_run_allowed:
        if (
            not isinstance(protocol.run_tombstone_status, str)
            or not protocol.run_tombstone_status
            or re.fullmatch(
                r"[A-Z0-9_]+",
                protocol.run_tombstone_status,
            )
            is None
        ):
            raise ValueError("policy-vision run tombstone identity drifted")
        raise ValueError(
            f"{protocol.run_tombstone_status}: invalid policy-vision protocol "
            f"{protocol.protocol_id} cannot be rerun"
        )
    root = args.repository_root.resolve()
    contract = protocol.contract_class.load(
        args.contract.resolve(),
        repository_root=root,
        validate_bound_sources=True,
    )
    _validate_runner_protocol(protocol, contract)
    if protocol.include_repair_identity:
        validate_source_diff = getattr(
            contract,
            "validate_formal_source_diff",
            None,
        )
        if not callable(validate_source_diff):
            raise ValueError("repaired policy-vision source-diff validator is absent")
        validate_source_diff(args.source_git_commit)
    canonical_output = (
        root / contract.data["output_contract"]["canonical_result_directory"]
    ).resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else canonical_output
    )
    if output_dir != canonical_output or root not in output_dir.parents:
        raise ValueError("policy-vision output must use the canonical Git path")
    labels_archive = args.labels_archive.resolve()
    if args.mode == "run":
        required = {
            "derived_root": args.derived_root,
            "model_dir": args.model_dir,
            "snapshot_manifest": args.snapshot_manifest,
            "host_alias": args.host_alias,
            "host_hostname": args.host_hostname,
            "container_hostname": args.container_hostname,
            "container_name": args.container_name,
            "container_image_digest": args.container_image_digest,
            "nvidia_driver_version": args.nvidia_driver_version,
            "host_evidence": args.host_evidence,
            "attempt_ledger": args.attempt_ledger,
        }
        missing = sorted(key for key, value in required.items() if value is None)
        if missing:
            raise ValueError("formal policy-vision run lacks: " + ", ".join(missing))
        if len(args.expected_gpu_uuid) != 2:
            raise ValueError("formal policy-vision run requires two GPU UUIDs")
        gpu_uuids = tuple(_normalize_gpu_uuid(value) for value in args.expected_gpu_uuid)
        _validate_run_git(root, args.source_git_commit)
        host_evidence, host_evidence_sha256 = _load_host_evidence(
            args.host_evidence.resolve(),
            host_alias=args.host_alias,
            host_hostname=args.host_hostname,
            container_hostname=args.container_hostname,
            container_name=args.container_name,
            container_image_digest=args.container_image_digest,
            nvidia_driver_version=args.nvidia_driver_version,
            expected_gpu_uuids=gpu_uuids,
        )
        staging = output_dir.with_name(f".{output_dir.name}.staging")
        if output_dir.exists() or staging.exists():
            raise FileExistsError(
                "canonical policy-vision output or staging already exists"
            )
        _validate_labels_archive(contract, labels_archive)
        derived_root = args.derived_root.resolve()
        verify_derived_projection(contract, derived_root)
        snapshot_manifest = args.snapshot_manifest.resolve()
        _validate_snapshot_path(contract, snapshot_manifest)
        model_dir = args.model_dir.resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError("policy-vision model directory is missing")
        work_items, witness_by_state = load_identity_witness(contract)
        derived = contract.data["immutable_inputs"]["derived_dataset"]
        image_tar_record = next(
            record
            for record in derived["exact_files"]
            if record["path"].endswith("images-00000-of-00001.tar")
        )
        _validate_run_git(root, args.source_git_commit)
        attempt_ledger = args.attempt_ledger.resolve()
        expected_attempt_ledger = Path(protocol.formal_attempt_ledger_path).resolve()
        if attempt_ledger != expected_attempt_ledger:
            raise ValueError("formal policy-vision attempt ledger path drifted")
        formal_attempt_claim, formal_attempt_claim_sha256 = _claim_formal_attempt(
            attempt_ledger,
            {
                "schema_version": "1.0.0",
                "status": "CLAIMED_POLICY_VISION_FORMAL_ATTEMPT",
                "protocol_id": protocol.protocol_id,
                "source_git_commit": args.source_git_commit,
                "contract_sha256": contract.sha256,
                "canonical_result_directory": contract.data["output_contract"][
                    "canonical_result_directory"
                ],
                "formal_attempt_ledger_path": protocol.formal_attempt_ledger_path,
                "host_alias": args.host_alias,
                "host_hostname": args.host_hostname,
                "container_hostname": args.container_hostname,
                "container_name": args.container_name,
                "container_image_digest": args.container_image_digest,
                "nvidia_driver_version": args.nvidia_driver_version,
                "expected_gpu_uuids": list(gpu_uuids),
            },
        )
        worker_outputs = _run_feature_workers(
            work_items=work_items,
            derived_root=derived_root,
            derived_payload_prefix=derived["payload_prefix"],
            expected_tar_member_count=210,
            model_dir=model_dir,
            snapshot_manifest=snapshot_manifest,
            expected_gpu_uuids=gpu_uuids,
            gpu_uuid_type_profile=protocol.gpu_uuid_type_profile,
            image_processor_size_profile=protocol.image_processor_size_profile,
        )
        _validate_run_git(root, args.source_git_commit)
        if image_tar_record["size_bytes"] <= 0:
            raise RuntimeError("derived image tar immutable identity is invalid")
        _validate_worker_outputs(
            outputs=worker_outputs,
            contract=contract,
            expected_gpu_uuids=gpu_uuids,
            runtime_profile_id=protocol.runtime_profile_id,
        )
        feature_records, replay = merge_policy_vision_workers(
            worker_outputs=worker_outputs,
            work_items=work_items,
            operation_ceiling=protocol.expected_operation_ceiling,
        )
        execution = _execution_record(
            contract=contract,
            worker_outputs=worker_outputs,
            replay=replay,
            host_alias=args.host_alias,
            host_hostname=args.host_hostname,
            container_hostname=args.container_hostname,
            container_name=args.container_name,
            container_image_digest=args.container_image_digest,
            nvidia_driver_version=args.nvidia_driver_version,
            expected_gpu_uuids=gpu_uuids,
            host_evidence=host_evidence,
            host_evidence_sha256=host_evidence_sha256,
            formal_attempt_claim=formal_attempt_claim,
            formal_attempt_claim_sha256=formal_attempt_claim_sha256,
            runtime_profile_id=protocol.runtime_profile_id,
            expected_operation_ceiling=protocol.expected_operation_ceiling,
        )
        files = _expected_files_from_features(
            contract=contract,
            labels_archive=labels_archive,
            source_commit=args.source_git_commit,
            work_items=work_items,
            witness_by_state=witness_by_state,
            feature_records=feature_records,
            execution=execution,
            protocol=protocol,
        )

        def pre_publish_check() -> None:
            _validate_run_source_snapshot(
                root,
                args.source_git_commit,
                formal_source_paths=protocol.formal_source_paths,
            )
            _validate_labels_archive(contract, labels_archive)

        _write_new_result(
            output_dir,
            files,
            pre_publish_check=pre_publish_check,
            expected_output_files=protocol.expected_output_files,
        )
        status = protocol.run_status
    else:
        files = _validate_mode(
            contract=contract,
            output_dir=output_dir,
            labels_archive=labels_archive,
            source_commit=args.source_git_commit,
            protocol=protocol,
        )
        status = protocol.valid_status
    print(
        json.dumps(
            {
                "status": status,
                "source_git_commit": args.source_git_commit,
                "contract_sha256": contract.sha256,
                "output_dir": str(output_dir),
                "formal_state_count": 15,
                "candidate_comparison_count": 60,
                "unique_image_count": 75,
                "gpu_count": 2,
                "policy_vision_feature_forward_count": 31,
                "policy_forward_count": 0,
                "language_model_forward_count": 0,
                "generation_count": 0,
                "confirm_state_access_count": 0,
                "sealed_test_state_access_count": 0,
                "files": {
                    name: {
                        "size_bytes": len(files[name]),
                        "sha256": sha256_bytes(files[name]),
                    }
                    for name in protocol.expected_output_files
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    run_protocol_main(V1_RUNNER_PROTOCOL, argv)


if __name__ == "__main__":
    main()
