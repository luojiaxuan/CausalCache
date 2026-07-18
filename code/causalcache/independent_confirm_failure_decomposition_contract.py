"""Frozen Source-A/Execution-B contract for confirm-20 failure decomposition."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_independent_confirm20_failure_decomposition_v1"
SOURCE_STATUS = (
    "source_only_frozen_before_independent_confirm20_failure_decomposition"
)
RUNNER_STATUS = "frozen_independent_confirm20_failure_decomposition_execution_b"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_independent_confirm20_failure_decomposition_v1.json"
)
CANONICAL_RUNNER_FREEZE_PATH = (
    "code/configs/causalcache_independent_confirm20_failure_decomposition_runner_v1.json"
)
CONFIG_SHA256 = "e522f5cab8bfd359d8a869a5a21eac771ae3e81a204a872d619e51bdf31cf020"

PARENT_RESULT_COMMIT = "0a107b6fffa13c0f01dda04d775d7ba4adf48486"
PARENT_REPO = "gavinlaw/causalcache-independent-confirm20-mobile"
PARENT_TAG = "independent-confirm20-v1"
PARENT_TAG_OBJECT = "48921cafa00d7102d8b4c709d58061951f90e42c"
PARENT_REPORT_COMMIT = "a0b408e58d629299be334a74ecbd0ec2fa2ed1fc"
PARENT_PAYLOAD_COMMIT = "6d0cd95997186293e01c65276f3c082c11a9f52d"
PARENT_BASE_COMMIT = "c66a67c3451cee7e20ed737401dc8b074151574f"
CHILD_REPO = (
    "gavinlaw/causalcache-independent-confirm20-failure-decomposition-mobile"
)
CHILD_TAG = "independent-confirm20-failure-decomposition-v1"

EXPECTED_PARENT_TARGETS = (
    {
        "path": "independent-confirm20/v1/report/bundle-manifest-v1.json",
        "sha256": "ccc996283b89f41db77c06f320e3b1ac88f3933bc19cfeea4c8b5b508ae4b351",
        "size_bytes": 2261,
    },
    {
        "path": "independent-confirm20/v1/report/raw-state-records-v1.jsonl",
        "sha256": "208fb36abb36b5e82eadcc8dac4ed20c5a030919902afe3e5b69cf6cad76a52a",
        "size_bytes": 1388707,
        "expected_record_count": 20,
    },
    {
        "path": "independent-confirm20/v1/report/fixed-report-v1.json",
        "sha256": "969729deb66e610846694b0fa1139c6b47f3927db7a58b50367f8ed93365198b",
        "size_bytes": 36666,
    },
)
EXPECTED_OUTPUT_TARGETS = (
    "independent-confirm20-failure-decomposition/v1/state-decomposition-v1.jsonl",
    "independent-confirm20-failure-decomposition/v1/failure-decomposition-report-v1.json",
    "independent-confirm20-failure-decomposition/v1/bundle-manifest-v1.json",
)
ALLOWED_PARENT_HF_API_METHODS = (
    "dataset_info",
    "list_repo_commits",
    "list_repo_files",
    "list_repo_refs",
)

_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class IndependentConfirmFailureDecompositionContract:
    repository_root: Path
    source_path: Path
    data: Mapping[str, Any]
    sha256: str

    @property
    def source(self) -> Mapping[str, Any]:
        return _mapping(self.data["source_freeze"], "source freeze")

    @property
    def runner_path(self) -> Path:
        relative = _mapping(
            self.source["execution_b_runner_freeze"], "runner freeze"
        )["path"]
        return self.repository_root / _safe_relative(relative, "runner path")


@dataclass(frozen=True)
class SourceIdentity:
    git_commit: str
    origin_main_git_commit: str
    branch: str
    origin_url: str
    source_inventory: tuple[Mapping[str, Any], ...]
    loaded_module_inventory: tuple[Mapping[str, Any], ...]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return value


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != value:
        raise ValueError(f"{label} is unsafe or noncanonical")
    return value


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be LF-terminated bytes")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict JSON") from error
    return _mapping(value, label)


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing") from error
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path.read_bytes()


def _file_record(value: Any, *, allow_commit: bool) -> Mapping[str, Any]:
    record = _mapping(value, "file record")
    required = {"path", "sha256", "size_bytes"}
    allowed = required | ({"git_commit"} if allow_commit else set())
    if not required.issubset(record) or not set(record).issubset(allowed):
        raise ValueError("file record keys drifted")
    _safe_relative(record["path"], "file record path")
    if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
        raise ValueError("file record size is malformed")
    if not isinstance(record["sha256"], str) or _SHA256.fullmatch(record["sha256"]) is None:
        raise ValueError("file record SHA256 is malformed")
    commit = record.get("git_commit")
    if commit is not None and (
        not allow_commit or not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None
    ):
        raise ValueError("file record commit is malformed")
    return record


def _validate_parent(data: Mapping[str, Any]) -> None:
    parent = _mapping(data["parent_confirm"], "parent confirm")
    git = _mapping(parent.get("git"), "parent Git")
    if git.get("result_commit") != PARENT_RESULT_COMMIT:
        raise ValueError("parent result commit drifted")
    summary = _file_record(git.get("summary"), allow_commit=False)
    completion = _file_record(git.get("completion"), allow_commit=False)
    if (
        summary.get("sha256")
        != "ff1270f7580005c3c6e9aa4bdcf124e804efa9826023626ebcf3115d73254a74"
        or completion.get("sha256")
        != "38c659065ef069e3cd0be2cfcfe550f1e3dee81bc2ec8405c3526c80e38ccd77"
    ):
        raise ValueError("parent Git artifact identity drifted")
    hf = _mapping(parent.get("hf"), "parent HF")
    if dict(hf) != {
        "repo": PARENT_REPO,
        "repo_type": "dataset",
        "private": True,
        "tag": PARENT_TAG,
        "annotated_tag_object": PARENT_TAG_OBJECT,
        "base_commit": PARENT_BASE_COMMIT,
        "payload_commit": PARENT_PAYLOAD_COMMIT,
        "report_commit": PARENT_REPORT_COMMIT,
        "tag_resolved_commit": PARENT_REPORT_COMMIT,
    }:
        raise ValueError("parent immutable HF identity drifted")
    locked = _mapping(parent.get("locked_result"), "parent locked result")
    if dict(locked) != {
        "status": "NO_GO_INDEPENDENT_CONFIRM",
        "fixed_state_denominator": 20,
        "independent_raw_utility_sum": 0.7033851761698315,
        "ocr_rgb_v2_raw_utility_sum": 0.7832678721484285,
        "closed_loop_executed": False,
        "matched_nll_executed": False,
        "sealed_androidworld_test_executed": False,
    }:
        raise ValueError("parent frozen NO-GO drifted")


def _validate_analysis_and_routing(data: Mapping[str, Any]) -> None:
    analysis = _mapping(data["analysis_contract"], "analysis contract")
    if (
        analysis.get("diagnostic_only") is not True
        or analysis.get("fixed_state_denominator") != 20
        or analysis.get("budget") != 2
        or analysis.get("candidate_event_step_ids") != [1, 2, 3, 4]
        or analysis.get("comparators")
        != {
            "E": "exact_subset_oracle",
            "I": "sealed_learned_independent_ensemble",
            "J": "oracle_independent_projected_target",
            "O": "sealed_ocr_rgb_v2",
        }
    ):
        raise ValueError("analysis identity or denominator drifted")
    j = _mapping(analysis.get("j_selector"), "J selector")
    if j != {
        "definition": "0.5_times_empty_marginal_plus_mean_of_other_singleton_base_conditional_marginals",
        "event_score": "0.5_times_D_empty_minus_D_j_plus_mean_i_not_j_of_D_i_minus_D_ij",
        "forced_fill": False,
        "strictly_positive_score_required": True,
        "tie_break": "lower_event_step_id",
        "top_b": True,
    }:
        raise ValueError("oracle-independent J contract drifted")
    aggregation = _mapping(analysis.get("aggregation"), "aggregation")
    if (
        aggregation.get("primary_comparison") != "raw_utility"
        or aggregation.get("normalized_reporting_can_route") is not False
        or aggregation.get("unit")
        != "trajectory_equal_equivalent_to_state_equal_for_one_state_per_trajectory"
    ):
        raise ValueError("primary raw aggregation boundary drifted")
    replay = _mapping(analysis.get("sealed_parent_replay"), "parent replay")
    if not replay or any(value is not True for value in replay.values()):
        raise ValueError("sealed parent replay requirement drifted")
    routing = _mapping(data["routing_contract"], "routing contract")
    if routing.get("bootstrap") != {
        "confidence": 0.9,
        "interval": "percentile",
        "quantile": "Hyndman_Fan_type_7",
        "resamples": 10000,
        "seed": 271828,
        "unit": "trajectory",
    }:
        raise ValueError("bootstrap contract drifted")
    case_a = _mapping(routing.get("case_a"), "Case A")
    if (
        case_a.get("status") != "CASE_A_ORACLE_INDEPENDENT_LOSES_TO_OCR_RGB"
        or case_a.get("all_required") != ["J_minus_O_raw_mean_at_most_zero"]
        or case_a.get("maximum_raw_mean_delta_inclusive") != 0.0
    ):
        raise ValueError("Case A threshold drifted")
    case_b = _mapping(routing.get("case_b"), "Case B")
    if (
        case_b.get("status") != "CASE_B_TEACHER_VALID_STUDENT_DISTILLATION_GAP"
        or case_b.get("minimum_positive_trajectory_count") != 12
        or case_b.get("all_required")
        != [
            "J_minus_O_raw_mean_strictly_positive",
            "J_minus_O_paired_bootstrap_90_lower_strictly_positive",
            "J_minus_O_positive_trajectory_count_at_least_12_of_20",
            "sealed_I_minus_O_raw_mean_strictly_negative",
        ]
    ):
        raise ValueError("Case B threshold drifted")
    inconclusive = _mapping(routing.get("inconclusive"), "inconclusive route")
    if (
        inconclusive.get("status")
        != "INCONCLUSIVE_ORACLE_INDEPENDENT_CONFIRM_DECOMPOSITION"
        or inconclusive.get("action")
        != "NO_INDEPENDENT_V2_RESCUE_FROM_THIS_DIAGNOSTIC"
    ):
        raise ValueError("inconclusive route drifted")
    if routing.get("never_authorizes") != [
        "confirm_reclassification",
        "closed_loop",
        "matched_nll",
        "sealed_androidworld_test",
        "gate_training_on_confirm20",
    ]:
        raise ValueError("post-confirm authorization boundary drifted")


def _validate_boundaries(data: Mapping[str, Any]) -> None:
    if data.get("audit_scope") != {
        "confirm_consumed": True,
        "formal_run_not_yet_executed": True,
        "not_blind_holdout": True,
        "pre_source_a_exploratory_numeric_read": True,
        "user_case_split_predated_exploratory_read": True,
    }:
        raise ValueError("consumed-confirm audit scope drifted")
    inputs = _mapping(data["input_contract"], "input contract")
    if (
        tuple(inputs.get("allowed_parent_hf_api_methods", ()))
        != ALLOWED_PARENT_HF_API_METHODS
        or tuple(inputs.get("exact_force_download_targets", ()))
        != EXPECTED_PARENT_TARGETS
        or tuple(inputs.get("parent_commit_chain", ()))
        != (PARENT_REPORT_COMMIT, PARENT_PAYLOAD_COMMIT, PARENT_BASE_COMMIT)
        or inputs.get("force_download") is not True
        or inputs.get("fresh_download_directory_must_start_empty") is not True
        or inputs.get("selective_reader_only") is not True
        or inputs.get("pre_post_parent_identity_stable") is not True
        or inputs.get("parent_remote_mutation_call_count") != 0
    ):
        raise ValueError("selective immutable parent input drifted")
    destination = _mapping(data["destination"], "destination")
    if (
        destination.get("repo") != CHILD_REPO
        or destination.get("repo_type") != "dataset"
        or destination.get("private") is not True
        or destination.get("tag") != CHILD_TAG
        or destination.get("single_exact_report_commit") is not True
        or destination.get("annotated_tag_required") is not True
        or destination.get("tag_must_resolve_to_report_commit") is not True
        or destination.get("completed_replay_remote_mutation_count") != 0
    ):
        raise ValueError("child HF identity drifted")
    output = _mapping(data["output_contract"], "output contract")
    if (
        tuple(output.get("exact_targets", ())) != EXPECTED_OUTPUT_TARGETS
        or output.get("state_record_count") != 20
        or output.get("trajectory_record_count_embedded_in_report") != 20
        or output.get("parent_confirm_verdict_must_not_change") is not True
    ):
        raise ValueError("child output denominator drifted")
    if data.get("runtime_contract") != {
        "cpu_only": True,
        "gpu_count": 0,
        "gpu_preflight_required": False,
        "model_runtime_allowed": False,
        "network_scope": "parent_read_only_then_new_child_publication_only",
        "torch_import_required": False,
    }:
        raise ValueError("CPU-only runtime boundary drifted")
    authorization = _mapping(data["authorization"], "authorization")
    if not authorization or any(value is not False for value in authorization.values()):
        raise ValueError("Source-A authorization must remain entirely false")
    source_only = _mapping(
        data["source_only_operation_contract"], "source-only operations"
    )
    if not source_only or any(type(value) is not int or value != 0 for value in source_only.values()):
        raise ValueError("Source-A operation counts must remain zero")
    execution = _mapping(
        data["execution_fixed_operation_contract"], "execution operations"
    )
    nonzero = {
        "child_exact_target_count": 3,
        "parent_force_download_count": 3,
        "sealed_fixed_report_decode_count": 1,
        "sealed_raw_state_decode_count": 20,
        "state_decomposition_record_count": 20,
        "trajectory_decomposition_record_count": 20,
    }
    if any(execution.get(key) != value for key, value in nonzero.items()) or any(
        type(value) is not int or (key not in nonzero and value != 0)
        for key, value in execution.items()
    ):
        raise ValueError("execution operation contract drifted")
    local = _mapping(data["local_first_state_machine"], "local state machine")
    if (
        local.get("state_file_mode") != 0o600
        or local.get("artifact_file_mode") != 0o444
        or local.get("final_completion_is_hard_link_to_staging") is not True
        or local.get("no_overwrite") is not True
        or len(local.get("ordered_states", ())) != 9
    ):
        raise ValueError("local durable-state contract drifted")


def _validate_source(data: Mapping[str, Any]) -> None:
    source = _mapping(data["source_freeze"], "source freeze")
    if (
        source.get("branch") != "main"
        or source.get("origin_name") != "origin"
        or source.get("origin_url")
        != "https://github.com/luojiaxuan/CausalCache.git"
        or source.get("head_must_equal_origin_main") is not True
        or source.get("worktree_must_be_clean") is not True
        or source.get("source_a_local_tracking_validation_network_call_count") != 0
    ):
        raise ValueError("Source-A Git boundary drifted")
    runner = _mapping(source.get("execution_b_runner_freeze"), "runner freeze")
    if dict(runner) != {
        "path": CANONICAL_RUNNER_FREEZE_PATH,
        "must_be_absent_during_source_only_validation": True,
        "only_allowed_execution_b_source_tree_diff": True,
        "direct_single_parent_child_of_source_a": True,
        "bind_source_a_inventory_sha256": True,
        "bind_loaded_module_inventory_sha256": True,
        "bind_required_source_a_paths": True,
        "bind_git_prerequisites": True,
    }:
        raise ValueError("Execution-B freeze boundary drifted")
    paths = tuple(_sequence(source.get("required_source_a_paths"), "source paths"))
    if not paths or len(paths) != len(set(paths)) or CANONICAL_RUNNER_FREEZE_PATH in paths:
        raise ValueError("required Source-A paths drifted")
    for path in paths:
        _safe_relative(path, "required Source-A path")
    modules = tuple(_sequence(source.get("required_execution_modules"), "modules"))
    if modules != (
        "causalcache.independent_confirm_failure_decomposition_contract",
        "causalcache.independent_confirm_failure_decomposition",
        "causalcache.independent_confirm_failure_decomposition_runner",
        "scripts.manage_independent_confirm_failure_decomposition",
    ):
        raise ValueError("execution module inventory drifted")
    prerequisites = tuple(_sequence(source.get("git_prerequisites"), "prerequisites"))
    if len(prerequisites) != 6:
        raise ValueError("Git prerequisite denominator drifted")
    for record in prerequisites:
        _file_record(record, allow_commit=True)


def validate_contract_data(value: Any) -> Mapping[str, Any]:
    data = dict(_mapping(value, "failure decomposition contract"))
    if set(data) != {
        "schema_version",
        "protocol_id",
        "status",
        "parent_confirm",
        "analysis_contract",
        "audit_scope",
        "routing_contract",
        "input_contract",
        "destination",
        "output_contract",
        "runtime_contract",
        "authorization",
        "local_first_state_machine",
        "execution_fixed_operation_contract",
        "source_freeze",
        "source_only_operation_contract",
    }:
        raise ValueError("failure decomposition contract keys drifted")
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("protocol_id") != PROTOCOL_ID
        or data.get("status") != SOURCE_STATUS
    ):
        raise ValueError("failure decomposition contract identity drifted")
    _validate_parent(data)
    _validate_analysis_and_routing(data)
    _validate_boundaries(data)
    _validate_source(data)
    return data


def load_frozen_failure_decomposition_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> IndependentConfirmFailureDecompositionContract:
    root = Path(repository_root).resolve(strict=True)
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = root / source_path
    payload = _regular_file_bytes(source_path, label="failure decomposition config")
    digest = sha256_bytes(payload)
    if digest != CONFIG_SHA256:
        raise ValueError("failure decomposition config differs from frozen SHA256")
    data = validate_contract_data(
        _strict_json(payload, label="failure decomposition config")
    )
    return IndependentConfirmFailureDecompositionContract(
        repository_root=root,
        source_path=source_path,
        data=data,
        sha256=digest,
    )


def _git(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git validation failed: {message}") from error


def _git_blob(root: Path, revision_path: str) -> bytes:
    try:
        return subprocess.run(
            ("git", "show", revision_path),
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git blob validation failed: {message}") from error


def _validate_prerequisites(
    contract: IndependentConfirmFailureDecompositionContract,
    *,
    head: str,
) -> None:
    for raw in contract.source["git_prerequisites"]:
        record = _file_record(raw, allow_commit=True)
        relative = record["path"]
        payload = _regular_file_bytes(
            contract.repository_root / relative, label=f"prerequisite {relative}"
        )
        if len(payload) != record["size_bytes"] or sha256_bytes(payload) != record["sha256"]:
            raise ValueError(f"Git prerequisite bytes drifted: {relative}")
        commit = record.get("git_commit")
        if commit is not None:
            _git(contract.repository_root, "merge-base", "--is-ancestor", commit, head)
            if payload != _git_blob(contract.repository_root, f"{commit}:{relative}"):
                raise ValueError(f"historical prerequisite drifted: {relative}")


def _source_inventory(
    contract: IndependentConfirmFailureDecompositionContract,
    *,
    commit: str,
) -> tuple[Mapping[str, Any], ...]:
    result = []
    for relative in contract.source["required_source_a_paths"]:
        payload = _regular_file_bytes(
            contract.repository_root / relative, label=f"Source-A path {relative}"
        )
        if payload != _git_blob(contract.repository_root, f"{commit}:{relative}"):
            raise ValueError(f"Source-A blob differs from commit: {relative}")
        result.append(
            {"path": relative, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        )
    return tuple(result)


def _module_path(module: str) -> str:
    if not (module.startswith("causalcache.") or module.startswith("scripts.")):
        raise ValueError("execution module is outside the code package")
    return "code/" + module.replace(".", "/") + ".py"


def _loaded_module_inventory(
    contract: IndependentConfirmFailureDecompositionContract,
    *,
    commit: str,
) -> tuple[Mapping[str, Any], ...]:
    source_paths = set(contract.source["required_source_a_paths"])
    result = []
    for module in contract.source["required_execution_modules"]:
        relative = _module_path(module)
        if relative not in source_paths:
            raise ValueError("execution module is absent from Source-A paths")
        payload = _regular_file_bytes(
            contract.repository_root / relative, label=f"execution module {module}"
        )
        if payload != _git_blob(contract.repository_root, f"{commit}:{relative}"):
            raise ValueError(f"execution module differs from commit: {module}")
        result.append(
            {
                "module": module,
                "path": relative,
                "size_bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    return tuple(result)


def _source_record(identity: SourceIdentity) -> Mapping[str, Any]:
    source_inventory = list(identity.source_inventory)
    loaded = list(identity.loaded_module_inventory)
    return {
        "git_commit": identity.git_commit,
        "origin_main_git_commit": identity.origin_main_git_commit,
        "branch": identity.branch,
        "origin_url": identity.origin_url,
        "source_inventory": source_inventory,
        "source_inventory_sha256": sha256_bytes(canonical_json_bytes(source_inventory)),
        "loaded_module_inventory": loaded,
        "loaded_module_inventory_sha256": sha256_bytes(canonical_json_bytes(loaded)),
    }


def validate_source_a(
    contract: IndependentConfirmFailureDecompositionContract,
    *,
    expected_source_a_git_commit: str | None = None,
) -> Mapping[str, Any]:
    root = contract.repository_root
    source = contract.source
    head = _git(root, "rev-parse", "HEAD").decode("ascii")
    tracking = _git(root, "rev-parse", "origin/main").decode("ascii")
    branch = _git(root, "branch", "--show-current").decode("utf-8")
    origin_url = _git(root, "remote", "get-url", "origin").decode("utf-8")
    status_bytes = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    runner_tracked = _git(root, "ls-files", "--", CANONICAL_RUNNER_FREEZE_PATH)
    if (
        head != tracking
        or (expected_source_a_git_commit is not None and head != expected_source_a_git_commit)
        or branch != source["branch"]
        or origin_url != source["origin_url"]
        or status_bytes
        or contract.runner_path.exists()
        or runner_tracked
    ):
        raise ValueError("Source-A requires clean pushed main and absent Execution-B")
    _validate_prerequisites(contract, head=head)
    identity = SourceIdentity(
        git_commit=head,
        origin_main_git_commit=tracking,
        branch=branch,
        origin_url=origin_url,
        source_inventory=_source_inventory(contract, commit=head),
        loaded_module_inventory=_loaded_module_inventory(contract, commit=head),
    )
    return {
        "status": "VALIDATED_INDEPENDENT_CONFIRM20_FAILURE_DECOMPOSITION_SOURCE_A",
        "source_a": _source_record(identity),
        **dict(contract.data["source_only_operation_contract"]),
    }


def _runner_payload(
    contract: IndependentConfirmFailureDecompositionContract,
    validation: Mapping[str, Any],
) -> Mapping[str, Any]:
    source = _mapping(validation.get("source_a"), "validated Source-A")
    paths = list(contract.source["required_source_a_paths"])
    prerequisites = list(contract.source["git_prerequisites"])
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": RUNNER_STATUS,
        "contract_path": CANONICAL_CONFIG_PATH,
        "contract_sha256": contract.sha256,
        "contract_size_bytes": contract.source_path.stat().st_size,
        "source_a_git_commit": source["git_commit"],
        "source_a_origin_main_git_commit": source["origin_main_git_commit"],
        "execution_b_direct_parent_required": source["git_commit"],
        "execution_b_required_unique_diff": [CANONICAL_RUNNER_FREEZE_PATH],
        "required_source_a_paths": paths,
        "required_source_a_paths_sha256": sha256_bytes(canonical_json_bytes(paths)),
        "git_prerequisites": prerequisites,
        "git_prerequisites_sha256": sha256_bytes(canonical_json_bytes(prerequisites)),
        "source_blob_inventory": list(source["source_inventory"]),
        "source_blob_inventory_sha256": source["source_inventory_sha256"],
        "source_a_inventory_sha256": source["source_inventory_sha256"],
        "loaded_module_inventory": list(source["loaded_module_inventory"]),
        "loaded_module_inventory_sha256": source["loaded_module_inventory_sha256"],
        "parent_result_git_commit": PARENT_RESULT_COMMIT,
        "parent_repo": PARENT_REPO,
        "parent_report_commit": PARENT_REPORT_COMMIT,
        "parent_tag": PARENT_TAG,
        "parent_tag_object": PARENT_TAG_OBJECT,
        "parent_exact_force_download_targets": list(EXPECTED_PARENT_TARGETS),
        "child_repo": CHILD_REPO,
        "child_tag": CHILD_TAG,
        "execution_authorized_after_clean_pushed_b_only": True,
        "diagnostic_only": True,
        "closed_loop_authorized": False,
        "matched_nll_authorized": False,
        "sealed_androidworld_test_authorized": False,
        "gate_training_authorized": False,
    }


def runner_freeze_bytes(
    contract: IndependentConfirmFailureDecompositionContract,
    validation: Mapping[str, Any],
) -> bytes:
    return pretty_json_bytes(_runner_payload(contract, validation))


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short runner-freeze write")
            view = view[written:]
        os.fsync(descriptor)
    except FileExistsError as error:
        raise ValueError("Execution-B runner freeze already exists") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def materialize_runner_freeze(
    contract: IndependentConfirmFailureDecompositionContract,
    *,
    expected_source_a_git_commit: str,
) -> Mapping[str, Any]:
    validation = validate_source_a(
        contract, expected_source_a_git_commit=expected_source_a_git_commit
    )
    payload = runner_freeze_bytes(contract, validation)
    _exclusive_write(contract.runner_path, payload)
    return {
        "status": "MATERIALIZED_INDEPENDENT_CONFIRM20_FAILURE_DECOMPOSITION_EXECUTION_B",
        "path": CANONICAL_RUNNER_FREEZE_PATH,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "source_a_git_commit": expected_source_a_git_commit,
        "file_write_count": 1,
        "network_call_count": 0,
    }


def load_runner_freeze(
    contract: IndependentConfirmFailureDecompositionContract,
) -> Mapping[str, Any]:
    payload = _regular_file_bytes(contract.runner_path, label="Execution-B freeze")
    value = _strict_json(payload, label="Execution-B freeze")
    expected = _runner_payload(
        contract,
        {
            "source_a": {
                "git_commit": value.get("source_a_git_commit"),
                "origin_main_git_commit": value.get("source_a_origin_main_git_commit"),
                "source_inventory": value.get("source_blob_inventory"),
                "source_inventory_sha256": value.get("source_a_inventory_sha256"),
                "loaded_module_inventory": value.get("loaded_module_inventory"),
                "loaded_module_inventory_sha256": value.get("loaded_module_inventory_sha256"),
            }
        },
    )
    if dict(value) != expected or pretty_json_bytes(value) != payload:
        raise ValueError("Execution-B freeze identity or bytes drifted")
    source_commit = value.get("source_a_git_commit")
    if not isinstance(source_commit, str) or _COMMIT.fullmatch(source_commit) is None:
        raise ValueError("Execution-B Source-A commit is malformed")
    return value


def _live_remote_main(root: Path) -> str:
    output = _git(root, "ls-remote", "--heads", "origin", "refs/heads/main")
    lines = output.decode("ascii").splitlines()
    suffix = "\trefs/heads/main"
    if len(lines) != 1 or not lines[0].endswith(suffix):
        raise ValueError("live origin main identity is unavailable")
    commit = lines[0][: -len(suffix)]
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("live origin main commit is malformed")
    return commit


def validate_execution_b_source(
    contract: IndependentConfirmFailureDecompositionContract,
    *,
    expected_execution_b_git_commit: str,
) -> Mapping[str, Any]:
    if _COMMIT.fullmatch(expected_execution_b_git_commit) is None:
        raise ValueError("expected Execution-B commit is malformed")
    freeze = load_runner_freeze(contract)
    root = contract.repository_root
    head = _git(root, "rev-parse", "HEAD").decode("ascii")
    tracking = _git(root, "rev-parse", "origin/main").decode("ascii")
    branch = _git(root, "branch", "--show-current").decode("utf-8")
    origin_url = _git(root, "remote", "get-url", "origin").decode("utf-8")
    status_bytes = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    live = _live_remote_main(root)
    if (
        head != expected_execution_b_git_commit
        or head != tracking
        or head != live
        or branch != "main"
        or origin_url != "https://github.com/luojiaxuan/CausalCache.git"
        or status_bytes
    ):
        raise ValueError("Execution-B requires clean pushed live main")
    source_a = freeze["source_a_git_commit"]
    parents = _git(root, "rev-list", "--parents", "-n", "1", head).decode("ascii").split()
    if parents != [head, source_a]:
        raise ValueError("Execution-B is not the direct child of Source-A")
    diff = _git(root, "diff", "--name-only", source_a, head).decode("utf-8").splitlines()
    if diff != [CANONICAL_RUNNER_FREEZE_PATH]:
        raise ValueError("Execution-B unique tree diff drifted")
    if _regular_file_bytes(
        contract.runner_path, label="Execution-B freeze"
    ) != _git_blob(root, f"{head}:{CANONICAL_RUNNER_FREEZE_PATH}"):
        raise ValueError("Execution-B freeze differs from committed bytes")
    _validate_prerequisites(contract, head=head)
    inventory = _source_inventory(contract, commit=head)
    loaded = _loaded_module_inventory(contract, commit=head)
    if list(inventory) != freeze["source_blob_inventory"] or list(loaded) != freeze["loaded_module_inventory"]:
        raise ValueError("Execution-B Source-A inventory drifted")
    return {
        "source_a_git_commit": source_a,
        "execution_b_git_commit": head,
        "origin_main_git_commit": tracking,
        "live_origin_main_git_commit": live,
        "execution_b_direct_single_parent": True,
        "execution_b_unique_diff": CANONICAL_RUNNER_FREEZE_PATH,
        "contract_sha256": contract.sha256,
        "runner_freeze_sha256": sha256_bytes(
            _regular_file_bytes(contract.runner_path, label="Execution-B freeze")
        ),
        "source_inventory_sha256": freeze["source_a_inventory_sha256"],
        "loaded_module_inventory_sha256": freeze["loaded_module_inventory_sha256"],
        "parent_report_commit": PARENT_REPORT_COMMIT,
        "parent_tag_object": PARENT_TAG_OBJECT,
        "diagnostic_only": True,
    }


__all__ = [
    "ALLOWED_PARENT_HF_API_METHODS",
    "CANONICAL_CONFIG_PATH",
    "CANONICAL_RUNNER_FREEZE_PATH",
    "CHILD_REPO",
    "CHILD_TAG",
    "CONFIG_SHA256",
    "EXPECTED_OUTPUT_TARGETS",
    "EXPECTED_PARENT_TARGETS",
    "IndependentConfirmFailureDecompositionContract",
    "PARENT_REPORT_COMMIT",
    "PARENT_REPO",
    "PARENT_TAG",
    "PARENT_TAG_OBJECT",
    "canonical_json_bytes",
    "load_frozen_failure_decomposition_contract",
    "load_runner_freeze",
    "materialize_runner_freeze",
    "pretty_json_bytes",
    "runner_freeze_bytes",
    "sha256_bytes",
    "validate_contract_data",
    "validate_execution_b_source",
    "validate_source_a",
]
