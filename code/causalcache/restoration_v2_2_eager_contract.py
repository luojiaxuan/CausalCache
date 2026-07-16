"""Fail-closed source contract for the restoration-v2.2 eager full-45 run."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from causalcache.restoration_v2_1_full_45_contract import (
    FULL_45_PROJECTION_SHA256,
    RestorationV21Full45Contract,
)


PROTOCOL_ID = "causalcache_restoration_v2_2_eager_full_45_substrate"
CANONICAL_CONFIG_PATH = "code/configs/causalcache_restoration_v2_2_eager.json"
FROZEN_RESTORATION_V2_2_EAGER_SHA256 = (
    "f473bb8a1657072235dd73bf78a93aff27b438d7baca65b7ef6096cf985effa7"
)
SOURCE_PARENT_GIT_COMMIT = "8a88b530bd72bb65f35d569aac1ea6e9263cc1a6"
CANONICAL_ATTEMPT_ID = "restoration-v2-2-eager-full-45-substrate-v1"
CANONICAL_OUTPUT_DIR = Path(
    "/data/experiments/causalcache/restoration-v2-2-eager-full-45-substrate-v1"
)
CANONICAL_LEDGER_PATH = Path(
    "/data/experiments/causalcache/"
    ".restoration-v2-2-eager-full-45-substrate-v1.attempt.json"
)
CANONICAL_ARCHIVE_PATH = Path(
    "/data/experiments/causalcache/"
    "restoration-v2-2-eager-full-45-substrate-v1.raw.tar"
)
CANONICAL_HF_REPO = (
    "gavinlaw/causalcache-restoration-v2-2-eager-full-45-substrate-mobile"
)
CANONICAL_HF_TAG = "v2.2-eager-full-45-substrate-v1"
CANONICAL_HF_PATH = "raw/restoration-v2-2-eager-full-45-substrate-v1.tar"
PASS_OUTCOME = "PASS_V2_2_EAGER_FULL_45_SUBSTRATE"
NO_GO_OUTCOME = "NO_GO_V2_2_EAGER_FULL_45_SUBSTRATE"
INVALID_OUTCOME = "INVALID_V2_2_EAGER_FULL_45_SUBSTRATE"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")

UNCHANGED_COMPONENTS = (
    "official_tool_schema",
    "system_prompt_bytes",
    "chat_template_mode",
    "prompt_builder",
    "canonical_action_parser",
    "teacher_target_serialization",
    "teacher_forced_distance_span",
    "masked_token_kl",
    "substrate_gate_thresholds",
    "per_state_operation_schedule",
)
ALLOWED_CHANGE_SCOPE = (
    "attention_backend_eager",
    "fixed_numerical_controls",
    "two_worker_transport",
    "non_resumable_attempt_lifecycle",
    "attempt_and_artifact_identity",
)
NON_RESUMABLE_SCHEDULE_OVERRIDES = {
    "resume_allowed": False,
    "resume_only_skips_existing_terminal_state_records": False,
    "resume_may_attempt_only_states_without_any_attempt_marker": False,
}
EXPECTED_EXECUTION_STACK = {
    "container_image_digest": (
        "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
    ),
    "python_version": "3.12.3",
    "torch_version": "2.11.0+cu130",
    "torch_cuda_version": "13.0",
    "cudnn_version": 91900,
    "transformers_version": "5.6.0",
    "nvidia_driver_version": "570.172.08",
    "gpu_name": "NVIDIA H200",
}
WORKER_IDENTITY_FIELDS_REQUIRED = (
    "device",
    "gpu_name",
    "gpu_uuid",
    "gpu_pci_bus_id",
    "logical_device_index",
    "nvidia_smi_index",
)
WORKER_IDENTITY_FIELDS_ALLOWED_TO_DIFFER = (
    "device",
    "gpu_uuid",
    "gpu_pci_bus_id",
    "logical_device_index",
    "nvidia_smi_index",
)
NEW_FORMAL_SOURCE_PATHS = {
    CANONICAL_CONFIG_PATH,
    "code/causalcache/restoration_v2_2_eager_contract.py",
    "code/causalcache/policy/gui_owl_v2_2_eager_runtime.py",
    "code/causalcache/restoration_v2_2_eager_artifact.py",
    "code/scripts/run_restoration_v2_2_eager_substrate.py",
    "code/scripts/manage_restoration_v2_2_eager_artifact.py",
    "code/scripts/validate_restoration_v2_2_eager_contract.py",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_strict_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_bytes(),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys drifted")


def _equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} must equal {expected!r}; got {actual!r}")


def _true(value: Any, name: str) -> None:
    if value is not True:
        raise ValueError(f"{name} must be true")


def _false(value: Any, name: str) -> None:
    if value is not False:
        raise ValueError(f"{name} must be false")


def _safe_relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"{name} must be a canonical repository-relative path")
    if path.as_posix() != value:
        raise ValueError(f"{name} must use canonical POSIX syntax")
    return value


def _resolve_record(
    repository_root: Path,
    record: Any,
    *,
    name: str,
    exact_extra_keys: set[str] | None = None,
) -> tuple[str, Path, Mapping[str, Any]]:
    value = _mapping(record, name)
    expected = {"path", "sha256"}
    if exact_extra_keys is not None:
        expected.update(exact_extra_keys)
        _exact_keys(value, expected, name)
    relative = _safe_relative_path(value.get("path"), f"{name}.path")
    digest = value.get("sha256")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name}.sha256 is invalid")
    root = repository_root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"{name}.path is missing or escapes the repository")
    _equal(_sha256_file(path), digest, f"{name} SHA256")
    return relative, path, value


def _git_blob(repository_root: Path, revision: str, relative: str) -> bytes:
    if revision != "HEAD" and GIT_SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError("source parent Git commit is invalid")
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=repository_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(f"{relative} is not committed at {revision}")
    return result.stdout


def _require_head_blob(repository_root: Path, relative: str, path: Path) -> None:
    if _git_blob(repository_root, "HEAD", relative) != path.read_bytes():
        raise ValueError(f"{relative} differs from its committed HEAD blob")


def _require_parent_blob(repository_root: Path, relative: str, path: Path) -> None:
    if _git_blob(repository_root, SOURCE_PARENT_GIT_COMMIT, relative) != path.read_bytes():
        raise ValueError(f"{relative} differs from the source-parent Git blob")


def _load_bound_record(
    repository_root: Path,
    record: Any,
    *,
    name: str,
    extra_keys: set[str],
    parent_blob: bool = False,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    relative, path, binding = _resolve_record(
        repository_root,
        record,
        name=name,
        exact_extra_keys=extra_keys,
    )
    _require_head_blob(repository_root, relative, path)
    if parent_blob:
        _require_parent_blob(repository_root, relative, path)
    return binding, load_strict_json_object(path)


def _validate_authorization(value: Mapping[str, Any], repository_root: Path) -> None:
    _exact_keys(
        value,
        {
            "source_parent_git_commit",
            "spatial_reference_artifact",
            "spatial_reference_summary",
            "v2_1_full_45_contract",
            "v2_1_full_45_no_go_artifact",
            "v2_1_full_45_no_go_summary",
            "v2_1_pilot_artifact",
            "v2_1_processor_artifact",
            "fresh_immutable_parent_evidence_validation_required_before_runtime_import",
            "authorization_scope",
        },
        "authorization",
    )
    _equal(
        value["source_parent_git_commit"],
        SOURCE_PARENT_GIT_COMMIT,
        "source parent Git commit",
    )
    _true(
        value["fresh_immutable_parent_evidence_validation_required_before_runtime_import"],
        "fresh immutable parent evidence requirement",
    )
    _equal(
        value["authorization_scope"],
        "fresh_v2_2_eager_full_45_substrate_only",
        "authorization scope",
    )

    spatial_artifact_binding, spatial_artifact = _load_bound_record(
        repository_root,
        value["spatial_reference_artifact"],
        name="spatial reference artifact",
        parent_blob=True,
        extra_keys={
            "required_status",
            "required_protocol_id",
            "required_hf_immutable_revision",
            "required_raw_archive_sha256",
        },
    )
    _equal(
        spatial_artifact.get("status"),
        spatial_artifact_binding["required_status"],
        "spatial artifact status",
    )
    _equal(
        spatial_artifact.get("protocol_id"),
        spatial_artifact_binding["required_protocol_id"],
        "spatial artifact protocol",
    )
    spatial_hf = _mapping(spatial_artifact.get("hf_artifact"), "spatial HF artifact")
    spatial_raw = _mapping(spatial_artifact.get("raw_archive"), "spatial raw archive")
    _equal(
        spatial_hf.get("immutable_revision"),
        spatial_artifact_binding["required_hf_immutable_revision"],
        "spatial immutable revision",
    )
    _equal(
        spatial_raw.get("sha256"),
        spatial_artifact_binding["required_raw_archive_sha256"],
        "spatial raw archive SHA256",
    )
    _true(
        spatial_raw.get("fresh_immutable_download_hash_verified"),
        "spatial fresh immutable hash verification",
    )
    _true(
        spatial_raw.get("fresh_immutable_download_rebuilt_canonical_bytes"),
        "spatial canonical archive rebuild",
    )

    spatial_summary_binding, spatial_summary = _load_bound_record(
        repository_root,
        value["spatial_reference_summary"],
        name="spatial reference summary",
        parent_blob=True,
        extra_keys={
            "required_status",
            "required_decision",
            "required_parent_v2_1_outcome",
            "required_bf16_eager_stable_state_count",
            "strict_cuda_determinism_claimed",
        },
    )
    _equal(
        spatial_summary.get("status"),
        spatial_summary_binding["required_status"],
        "spatial summary status",
    )
    _equal(
        spatial_summary.get("decision"),
        spatial_summary_binding["required_decision"],
        "spatial summary decision",
    )
    boundary = _mapping(spatial_summary.get("claim_boundary"), "spatial claim boundary")
    _equal(
        boundary.get("parent_v2_1_outcome_unchanged"),
        spatial_summary_binding["required_parent_v2_1_outcome"],
        "spatial parent outcome",
    )
    _false(
        spatial_summary_binding["strict_cuda_determinism_claimed"],
        "spatial binding strict determinism claim",
    )
    _false(
        boundary.get("strict_cuda_determinism_claimed"),
        "spatial strict determinism claim",
    )
    profiles = _sequence(spatial_summary.get("profiles"), "spatial profiles")
    eager = [
        _mapping(item, "spatial profile")
        for item in profiles
        if _mapping(item, "spatial profile").get("profile_id")
        == "bf16_eager_control"
    ]
    _equal(len(eager), 1, "BF16 eager profile count")
    expected_stable = spatial_summary_binding["required_bf16_eager_stable_state_count"]
    _equal(eager[0].get("state_count"), expected_stable, "BF16 eager denominator")
    _equal(
        eager[0].get("exact_generated_token_sequence_stable_count"),
        expected_stable,
        "BF16 eager token stability",
    )
    _equal(
        eager[0].get("exact_canonical_action_stable_count"),
        expected_stable,
        "BF16 eager action stability",
    )

    contract_binding, _ = _load_bound_record(
        repository_root,
        value["v2_1_full_45_contract"],
        name="v2.1 full-45 contract",
        extra_keys={"protocol_id"},
    )
    parent = RestorationV21Full45Contract.load(
        contract_binding["path"], repository_root=repository_root
    )
    _equal(parent.protocol_id, contract_binding["protocol_id"], "parent protocol")

    full_binding, full_artifact = _load_bound_record(
        repository_root,
        value["v2_1_full_45_no_go_artifact"],
        name="v2.1 full-45 no-go artifact",
        parent_blob=True,
        extra_keys={
            "required_status",
            "required_outcome",
            "required_fixed_state_denominator",
            "required_completed_state_count",
            "hf_immutable_revision",
            "raw_archive_sha256",
        },
    )
    full_result = _mapping(full_artifact.get("result"), "v2.1 full-45 result")
    for field in (
        "required_status",
        "required_outcome",
        "required_fixed_state_denominator",
        "required_completed_state_count",
    ):
        result_field = field.removeprefix("required_")
        _equal(full_result.get(result_field), full_binding[field], f"v2.1 {result_field}")
    full_hf = _mapping(full_artifact.get("hf_artifact"), "v2.1 full-45 HF")
    full_raw = _mapping(full_artifact.get("raw_archive"), "v2.1 full-45 raw")
    _equal(
        full_hf.get("immutable_revision"),
        full_binding["hf_immutable_revision"],
        "v2.1 immutable revision",
    )
    _equal(full_raw.get("sha256"), full_binding["raw_archive_sha256"], "v2.1 raw SHA")
    _true(full_raw.get("fresh_immutable_download_hash_verified"), "v2.1 fresh HF hash")

    summary_binding, full_summary = _load_bound_record(
        repository_root,
        value["v2_1_full_45_no_go_summary"],
        name="v2.1 full-45 no-go summary",
        parent_blob=True,
        extra_keys={
            "required_status",
            "required_outcome",
            "required_fixed_state_denominator",
        },
    )
    for field in (
        "required_status",
        "required_outcome",
        "required_fixed_state_denominator",
    ):
        result_field = field.removeprefix("required_")
        actual = (
            full_summary.get("metrics", {}).get(result_field)
            if result_field == "fixed_state_denominator"
            else full_summary.get(result_field)
        )
        _equal(actual, summary_binding[field], f"v2.1 summary {result_field}")

    pilot_binding, pilot_artifact = _load_bound_record(
        repository_root,
        value["v2_1_pilot_artifact"],
        name="v2.1 pilot artifact",
        parent_blob=True,
        extra_keys={
            "required_status",
            "required_outcome",
            "required_completed_state_count",
            "hf_immutable_revision",
        },
    )
    pilot_result = _mapping(pilot_artifact.get("result"), "v2.1 pilot result")
    pilot_hf = _mapping(pilot_artifact.get("hf_artifact"), "v2.1 pilot HF")
    for field in ("status", "outcome", "completed_state_count"):
        _equal(pilot_result.get(field), pilot_binding[f"required_{field}"], f"pilot {field}")
    _equal(
        pilot_hf.get("immutable_revision"),
        pilot_binding["hf_immutable_revision"],
        "pilot immutable revision",
    )

    processor_binding, processor_artifact = _load_bound_record(
        repository_root,
        value["v2_1_processor_artifact"],
        name="v2.1 processor artifact",
        parent_blob=True,
        extra_keys={
            "required_status",
            "required_state_count",
            "required_prompt_count",
            "hf_immutable_revision",
        },
    )
    reduction = _mapping(processor_artifact.get("compact_reduction"), "processor reduction")
    processor_hf = _mapping(processor_artifact.get("hf_artifact"), "processor HF")
    for field in ("status", "state_count", "prompt_count"):
        _equal(
            reduction.get(field),
            processor_binding[f"required_{field}"],
            f"processor {field}",
        )
    _equal(
        processor_hf.get("immutable_revision"),
        processor_binding["hf_immutable_revision"],
        "processor immutable revision",
    )


def _validate_inheritance(
    value: Mapping[str, Any], repository_root: Path, parent: Mapping[str, Any]
) -> None:
    _exact_keys(
        value,
        {
            "unchanged_components",
            "allowed_change_scope",
            "source_files",
            "formal_run_source_inventory_paths",
            "formal_run_must_bind_clean_pushed_main_git_commit",
            "scientific_source_change_after_first_attempt_allowed",
        },
        "scientific inheritance",
    )
    _equal(tuple(value["unchanged_components"]), UNCHANGED_COMPONENTS, "unchanged components")
    _equal(tuple(value["allowed_change_scope"]), ALLOWED_CHANGE_SCOPE, "allowed change scope")
    _true(value["formal_run_must_bind_clean_pushed_main_git_commit"], "formal Git binding")
    _false(
        value["scientific_source_change_after_first_attempt_allowed"],
        "post-attempt scientific source changes",
    )
    records = _sequence(value["source_files"], "scientific source files")
    parent_records = _sequence(
        _mapping(parent["source_lock"], "parent source lock")["files"],
        "parent scientific source files",
    )
    _equal(records, parent_records, "inherited scientific source records")
    for index, record in enumerate(records):
        relative, path, _ = _resolve_record(
            repository_root, record, name=f"scientific source file {index}", exact_extra_keys=set()
        )
        _require_head_blob(repository_root, relative, path)

    inventory = [
        _safe_relative_path(item, "formal source inventory path")
        for item in _sequence(
            value["formal_run_source_inventory_paths"], "formal source inventory"
        )
    ]
    if len(inventory) != len(set(inventory)):
        raise ValueError("formal source inventory contains duplicate paths")
    parent_inventory = set(
        _mapping(parent["source_lock"], "parent source lock")[
            "formal_run_source_inventory_paths"
        ]
    )
    required = parent_inventory | NEW_FORMAL_SOURCE_PATHS | {
        "data/results/restoration_v2_1_full_45_substrate/artifact.json",
        "data/results/restoration_v2_1_full_45_substrate/summary.json",
        "data/results/spatial_reference_audit_v1/artifact.json",
        "data/results/spatial_reference_audit_v1/summary.json",
    }
    missing = sorted(required.difference(inventory))
    if missing:
        raise ValueError(f"formal source inventory is incomplete: {missing}")


def _validate_data(value: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
    _exact_keys(
        value,
        {
            "derived_artifact",
            "selection_manifest",
            "roles_in_exact_order",
            "role_state_counts",
            "fixed_state_denominator",
            "state_projection_sha256",
            "fresh_state_output_required",
            "v2_1_raw_state_or_aggregate_output_may_be_imported",
            "v2_1_raw_output_may_count_toward_denominator",
            "v2_1_state_output_reuse_count",
            "pilot_output_may_count_toward_denominator",
            "state_filtering_or_top_up_allowed",
            "confirm_prompt_image_or_state_exposure_allowed",
        },
        "data",
    )
    parent_data = _mapping(parent["data"], "parent data")
    _equal(value.get("derived_artifact"), parent_data.get("derived_artifact"), "derived artifact")
    _equal(value.get("selection_manifest"), parent_data.get("selection_manifest"), "selection manifest")
    _equal(value.get("roles_in_exact_order"), parent_data.get("roles_in_exact_order"), "role order")
    _equal(value.get("role_state_counts"), parent_data.get("role_state_counts"), "role counts")
    _equal(value.get("fixed_state_denominator"), 45, "fixed state denominator")
    _equal(value.get("state_projection_sha256"), FULL_45_PROJECTION_SHA256, "state projection")
    _true(value.get("fresh_state_output_required"), "fresh state output")
    for field in (
        "v2_1_raw_state_or_aggregate_output_may_be_imported",
        "v2_1_raw_output_may_count_toward_denominator",
        "pilot_output_may_count_toward_denominator",
        "state_filtering_or_top_up_allowed",
        "confirm_prompt_image_or_state_exposure_allowed",
    ):
        _false(value.get(field), f"data.{field}")
    _equal(value.get("v2_1_state_output_reuse_count"), 0, "v2.1 output reuse count")


def _validate_runtime(value: Mapping[str, Any], repository_root: Path) -> None:
    _exact_keys(
        value,
        {
            "runtime_source",
            "dtype",
            "attention_implementation_requested",
            "attention_implementation_observed_must_equal",
            "eager_control_flags",
            "pytorch_strict_deterministic_algorithms_enabled",
            "scientific_environment_variables_allowed",
            "claim",
            "expected_execution_stack",
            "scientific_runtime_metadata_must_be_equal_across_workers",
            "worker_identity_fields_required",
            "worker_identity_fields_allowed_to_differ",
            "container_image_digest_must_be_equal_across_workers",
        },
        "runtime",
    )
    _resolve_record(
        repository_root,
        value["runtime_source"],
        name="v2.2 eager runtime source",
        exact_extra_keys=set(),
    )
    _equal(value["dtype"], "bfloat16", "runtime dtype")
    _equal(value["attention_implementation_requested"], "eager", "requested attention")
    _equal(
        value["attention_implementation_observed_must_equal"],
        "eager",
        "observed attention",
    )
    _equal(
        value["eager_control_flags"],
        {
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "seed": 0,
        },
        "eager numerical controls",
    )
    _false(
        value["pytorch_strict_deterministic_algorithms_enabled"],
        "strict deterministic algorithms",
    )
    _false(value["scientific_environment_variables_allowed"], "scientific environment variables")
    _equal(
        value["claim"],
        "eager_fixed_seed_tf32_disabled_numerical_control_not_strict_cuda_determinism",
        "runtime claim",
    )
    _equal(
        value["expected_execution_stack"],
        EXPECTED_EXECUTION_STACK,
        "recovered execution stack",
    )
    _true(
        value["scientific_runtime_metadata_must_be_equal_across_workers"],
        "cross-worker scientific runtime identity",
    )
    _equal(
        tuple(value["worker_identity_fields_required"]),
        WORKER_IDENTITY_FIELDS_REQUIRED,
        "required worker identity fields",
    )
    _equal(
        tuple(value["worker_identity_fields_allowed_to_differ"]),
        WORKER_IDENTITY_FIELDS_ALLOWED_TO_DIFFER,
        "cross-worker identity exceptions",
    )
    _true(
        value["container_image_digest_must_be_equal_across_workers"],
        "cross-worker image identity",
    )


def _validate_schedule_and_gate(
    schedule: Mapping[str, Any], gate: Mapping[str, Any], parent: Mapping[str, Any]
) -> None:
    parent_schedule = dict(
        _mapping(parent["computation_schedule"], "parent computation schedule")
    )
    parent_schedule.update(NON_RESUMABLE_SCHEDULE_OVERRIDES)
    _equal(
        schedule,
        parent_schedule,
        "unchanged scientific schedule with non-resumable v2.2 lifecycle",
    )
    parent_gate = _mapping(parent["substrate_gate"], "parent substrate gate")
    _exact_keys(gate, set(parent_gate), "substrate gate")
    inherited_gate_fields = set(parent_gate) - {"pass_outcome", "fail_outcome", "invalid_outcome"}
    for field in inherited_gate_fields:
        _equal(gate.get(field), parent_gate[field], f"unchanged gate {field}")
    _equal(gate.get("pass_outcome"), PASS_OUTCOME, "v2.2 pass outcome")
    _equal(gate.get("fail_outcome"), NO_GO_OUTCOME, "v2.2 no-go outcome")
    _equal(gate.get("invalid_outcome"), INVALID_OUTCOME, "v2.2 invalid outcome")
    _equal(gate.get("derived_required_parse_success_count"), math.ceil(45 * 0.99), "parse count")


def _validate_execution(value: Mapping[str, Any]) -> None:
    _exact_keys(
        value,
        {
            "attempt_id",
            "canonical_persistent_output_dir",
            "canonical_global_attempt_ledger",
            "canonical_raw_archive",
            "required_host_class",
            "worker_topology",
            "same_host_and_container_required_for_both_workers",
            "cross_host_attempt_allowed",
            "alternate_output_or_ledger_allowed",
            "output_or_ledger_deletion_after_first_attempt_allowed",
            "state_attempt_marker_written_before_first_generation",
            "runtime_import_requires_fresh_parent_evidence_validation",
        },
        "execution",
    )
    _equal(value["attempt_id"], CANONICAL_ATTEMPT_ID, "attempt ID")
    _equal(value["canonical_persistent_output_dir"], str(CANONICAL_OUTPUT_DIR), "output dir")
    _equal(value["canonical_global_attempt_ledger"], str(CANONICAL_LEDGER_PATH), "ledger")
    _equal(value["canonical_raw_archive"], str(CANONICAL_ARCHIVE_PATH), "archive")
    _equal(value["required_host_class"], "Hyper_H200", "host class")
    _true(
        value["same_host_and_container_required_for_both_workers"],
        "same worker host/container",
    )
    for field in (
        "cross_host_attempt_allowed",
        "alternate_output_or_ledger_allowed",
        "output_or_ledger_deletion_after_first_attempt_allowed",
    ):
        _false(value[field], f"execution.{field}")
    for field in (
        "state_attempt_marker_written_before_first_generation",
        "runtime_import_requires_fresh_parent_evidence_validation",
    ):
        _true(value[field], f"execution.{field}")

    topology = _mapping(value["worker_topology"], "worker topology")
    _exact_keys(
        topology,
        {
            "worker_count",
            "gpu_model",
            "one_process_per_device",
            "workers",
            "cross_worker_state_stealing_allowed",
            "worker_failure_invalidates_entire_attempt",
            "worker_outputs_must_be_disjoint",
            "worker_union_must_equal_fixed_denominator",
        },
        "worker topology",
    )
    _equal(topology["worker_count"], 2, "worker count")
    _equal(topology["gpu_model"], "NVIDIA H200", "worker GPU model")
    _true(topology["one_process_per_device"], "one process per device")
    _false(topology["cross_worker_state_stealing_allowed"], "worker state stealing")
    for field in (
        "worker_failure_invalidates_entire_attempt",
        "worker_outputs_must_be_disjoint",
        "worker_union_must_equal_fixed_denominator",
    ):
        _true(topology[field], f"worker_topology.{field}")
    workers = _sequence(topology["workers"], "workers")
    _equal(len(workers), 2, "worker record count")
    expected = (
        ("even", "cuda:0", 0, list(range(0, 45, 2))),
        ("odd", "cuda:1", 1, list(range(1, 45, 2))),
    )
    union: set[int] = set()
    for index, (worker_value, expected_record) in enumerate(zip(workers, expected)):
        worker = _mapping(worker_value, f"worker {index}")
        _exact_keys(
            worker,
            {"worker_id", "device", "index_parity", "state_indices"},
            f"worker {index}",
        )
        actual = (
            worker["worker_id"],
            worker["device"],
            worker["index_parity"],
            worker["state_indices"],
        )
        _equal(actual, expected_record, f"worker {index} parity shard")
        indices = set(worker["state_indices"])
        if union.intersection(indices):
            raise ValueError("worker state shards overlap")
        union.update(indices)
    _equal(union, set(range(45)), "worker state shard union")


def validate_restoration_v2_2_eager_contract(
    data: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    _exact_keys(
        data,
        {
            "schema_version",
            "protocol_id",
            "preregistration_status",
            "authorization",
            "scientific_inheritance",
            "data",
            "runtime",
            "computation_schedule",
            "substrate_gate",
            "prohibited_work",
            "execution",
            "artifact_destination",
            "promotion",
        },
        "v2.2 eager contract",
    )
    _equal(data["schema_version"], "0.1.0", "schema version")
    _equal(data["protocol_id"], PROTOCOL_ID, "protocol ID")
    _equal(
        data["preregistration_status"],
        "source_only_frozen_before_any_v2_2_eager_policy_output",
        "preregistration status",
    )
    root = repository_root.resolve()
    _validate_authorization(_mapping(data["authorization"], "authorization"), root)
    parent_contract = RestorationV21Full45Contract.load(
        root / "code/configs/causalcache_restoration_v2_1_full_45.json",
        repository_root=root,
    )
    parent = parent_contract.data
    _validate_inheritance(
        _mapping(data["scientific_inheritance"], "scientific inheritance"),
        root,
        parent,
    )
    _validate_data(_mapping(data["data"], "data"), parent)
    _validate_runtime(_mapping(data["runtime"], "runtime"), root)
    _validate_schedule_and_gate(
        _mapping(data["computation_schedule"], "computation schedule"),
        _mapping(data["substrate_gate"], "substrate gate"),
        parent,
    )
    prohibited = _mapping(data["prohibited_work"], "prohibited work")
    _exact_keys(
        prohibited,
        {
            "maximum_confirm_state_access_count",
            "maximum_confirm_processor_prompt_count",
            "maximum_confirm_decoder_input_count",
            "maximum_confirm_generation_count",
            "maximum_confirm_teacher_forward_count",
            "maximum_expert_action_read_count",
            "maximum_restoration_coalition_construction_count",
            "maximum_restoration_candidate_prompt_count",
            "maximum_restoration_label_count",
            "maximum_baseline_selection_count",
            "maximum_gate_training_example_count",
            "maximum_gate_model_forward_count",
            "maximum_gate_selection_count",
            "androidworld_test_split_access_allowed",
        },
        "prohibited work",
    )
    for field, value in prohibited.items():
        if field == "androidworld_test_split_access_allowed":
            _false(value, f"prohibited_work.{field}")
        else:
            _equal(value, 0, f"prohibited_work.{field}")
    _validate_execution(_mapping(data["execution"], "execution"))
    _equal(
        data["artifact_destination"],
        {
            "repo": CANONICAL_HF_REPO,
            "repo_type": "dataset",
            "visibility": "private",
            "tag": CANONICAL_HF_TAG,
            "path": CANONICAL_HF_PATH,
            "git_result_dir": "data/results/restoration_v2_2_eager_full_45_substrate",
            "fresh_immutable_download_and_byte_hash_required_before_git_manifest": True,
        },
        "artifact destination",
    )
    _equal(
        data["promotion"],
        {
            "pass_authorizes_only": (
                "freeze_and_audit_independent_v2_2_restoration_confirm_source"
            ),
            "pass_does_not_authorize_automatic_confirm_execution": True,
            "confirm_remains_locked_until_new_source_is_committed_and_pushed": True,
            "no_go_stops_v2_2_restoration_confirm": True,
            "source_only_validator_authorizes_policy_or_gpu_execution": False,
        },
        "promotion",
    )
    return {
        "fixed_state_denominator": 45,
        "state_projection_sha256": FULL_45_PROJECTION_SHA256,
        "worker_count": 2,
        "worker_state_counts": {"even": 23, "odd": 22},
        "maximum_generation_call_count": 90,
        "maximum_teacher_forward_count": 135,
        "maximum_kl_measurement_count": 90,
        "policy_or_gpu_execution_authorized": False,
    }


@dataclass(frozen=True)
class RestorationV22EagerContract:
    data: Mapping[str, Any]
    source_sha256: str
    validation: Mapping[str, Any]

    @property
    def protocol_id(self) -> str:
        return str(self.data["protocol_id"])

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
    ) -> "RestorationV22EagerContract":
        root = Path(repository_root).resolve()
        supplied = Path(path)
        resolved = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
        canonical = (root / CANONICAL_CONFIG_PATH).resolve()
        if resolved != canonical or not resolved.is_file():
            raise ValueError(f"v2.2 eager contract must be {CANONICAL_CONFIG_PATH}")
        source_sha256 = _sha256_file(resolved)
        if source_sha256 != FROZEN_RESTORATION_V2_2_EAGER_SHA256:
            raise ValueError("restoration-v2.2 eager contract SHA256 mismatch")
        data = load_strict_json_object(resolved)
        validation = validate_restoration_v2_2_eager_contract(
            data, repository_root=root
        )
        return cls(data=data, source_sha256=source_sha256, validation=validation)
