"""Fail-closed source contract for v2.2 eager restoration labels."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "causalcache_restoration_v2_2_eager_labels_v1"
CANONICAL_CONFIG_PATH = "code/configs/causalcache_restoration_v2_2_labels.json"
FROZEN_CONFIG_SHA256 = (
    "56b29f6879ef14167b20a39d0d61ebd0e698e3bbf0457062b63453180e71cf87"
)
SOURCE_PARENT_GIT_COMMIT = "89a5863b12e03e6b2ab7d9dc659d81b121b34712"
CANONICAL_ATTEMPT_ID = "restoration-v2-2-eager-labels-v1"
CANONICAL_OUTPUT_DIR = Path(
    "/data/experiments/causalcache/restoration-v2-2-eager-labels-v1"
)
CANONICAL_LEDGER_PATH = Path(
    "/data/experiments/causalcache/.restoration-v2-2-eager-labels-v1.attempt.json"
)
CANONICAL_ARCHIVE_PATH = Path(
    "/data/experiments/causalcache/restoration-v2-2-eager-labels-v1.raw.tar"
)
CANONICAL_HF_REPO = "gavinlaw/causalcache-restoration-labels-mobile"
CANONICAL_HF_TAG = "v2.2-eager-train-dev-exact-v1"
CANONICAL_HF_PATH = "raw/v2.2-eager-train-dev-exact-v1.tar"
PASS_OUTCOME = "PASS_RESTORATION_V2_2_EAGER_LABELS_V1"
INVALID_OUTCOME = "INVALID_RESTORATION_V2_2_EAGER_LABELS_V1"
EXPECTED_STATE_COUNT = 45
EXPECTED_DISTANCE_ROWS = 420
EXPECTED_DEPLOYMENT_EDGES = 435
EXPECTED_FULL_EDGES = 720
EXPECTED_PRIMARY_ORACLES = 45
EXPECTED_TEACHER_FORWARDS = 465
EXPECTED_KL_MEASUREMENTS = 420
EXPECTED_SOURCE_PATHS = (
    CANONICAL_CONFIG_PATH,
    "code/causalcache/restoration_v2_2_label_contract.py",
    "code/causalcache/restoration_v2_2_label_table.py",
    "code/causalcache/restoration_v2_2_label_parent.py",
    "code/causalcache/restoration_v2_2_label_artifact.py",
    "code/causalcache/data/restoration_v2_2_label_inputs.py",
    "code/scripts/run_restoration_v2_2_labels.py",
    "code/scripts/manage_restoration_v2_2_label_artifact.py",
    "code/scripts/validate_restoration_v2_2_label_contract.py",
    "code/causalcache/policy/gui_owl_v2_2_eager_runtime.py",
    "code/causalcache/policy/gui_owl_v2_1_runtime.py",
    "code/causalcache/policy/gui_owl_v2_1.py",
    "code/causalcache/policy/gui_owl_v2.py",
    "code/causalcache/restoration_v2_gpu_kl.py",
    "code/causalcache/data/restoration_v2_screening.py",
    "code/causalcache/data/guiodyssey_restoration_v2.py",
    "code/causalcache/low_fidelity_v2.py",
    "code/causalcache/restoration_v2_2_eager_artifact.py",
    "code/causalcache/restoration_v2_2_eager_contract.py",
    "code/configs/causalcache_restoration_v2_2_eager.json",
    "code/configs/causalcache_restoration_v2_1_full_45.json",
    "code/configs/causalcache_restoration_v2_1_pilot.json",
    "code/configs/causalcache_restoration_v2.json",
    "code/configs/restoration_v2_ocr_backend.json",
    "code/configs/gui_owl_1_5_8b_snapshot.json",
    "data/manifests/restoration_v2_selection.json",
    "data/manifests/restoration_v2_derived_artifact.json",
    "data/results/restoration_v2_2_eager_full_45_substrate/artifact.json",
    "data/results/restoration_v2_2_eager_full_45_substrate/summary.json",
)
V2_REPAIR_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_labels_v2_repair.json"
)
V2_REPAIR_FROZEN_CONFIG_SHA256 = (
    "7fc96883b905a5f026c18e65c3c7e377972559c775ac7ec95030ea1f04c2f94c"
)
V2_REPAIR_SOURCE_PARENT_GIT_COMMIT = "bf9e377ceb9c656af8ae84c223ad23d730136cb1"
V2_REPAIR_ATTEMPT_ID = "restoration-v2-2-eager-labels-v2"
V2_REPAIR_OUTPUT_DIR = Path(
    "/data/experiments/causalcache/restoration-v2-2-eager-labels-v2"
)
V2_REPAIR_LEDGER_PATH = Path(
    "/data/experiments/causalcache/.restoration-v2-2-eager-labels-v2.attempt.json"
)
V2_REPAIR_ARCHIVE_PATH = Path(
    "/data/experiments/causalcache/restoration-v2-2-eager-labels-v2.raw.tar"
)
V2_REPAIR_HF_TAG = "v2.2-eager-train-dev-exact-v2"
V2_REPAIR_HF_PATH = "raw/v2.2-eager-train-dev-exact-v2.tar"
V2_REPAIR_CANONICAL_MODEL_DIR = Path(
    "/data/artifacts/models/GUI-Owl-1.5-8B-Instruct"
)
V2_REPAIR_PASS_OUTCOME = "PASS_RESTORATION_V2_2_EAGER_LABELS_V2"
V2_REPAIR_INVALID_OUTCOME = "INVALID_RESTORATION_V2_2_EAGER_LABELS_V2"
V2_REPAIR_AGGREGATE_STATUS = "COMPLETED_RESTORATION_V2_2_EAGER_LABELS_V2"
V2_REPAIR_EXPECTED_SOURCE_PATHS = (
    V2_REPAIR_CONFIG_PATH,
    *EXPECTED_SOURCE_PATHS,
    "code/causalcache/policy/gui_owl_v2_vision.py",
    "data/results/restoration_v2_2_eager_labels_v1_attempt/summary.json",
    "data/results/restoration_v2_2_eager_labels_v1_attempt/README.md",
)


@dataclass(frozen=True)
class RestorationLabelAttemptProfile:
    config_path: str
    frozen_config_sha256: str
    source_parent_git_commit: str
    attempt_id: str
    output_dir: Path
    ledger_path: Path
    archive_path: Path
    hf_repo: str
    hf_tag: str
    hf_path: str
    pass_outcome: str
    invalid_outcome: str
    aggregate_status: str
    attempt_revision: str
    preregistration_status: str
    expected_source_paths: tuple[str, ...]
    supersedes_attempt_id: str | None = None
    canonical_model_dir: Path | None = None


V1_ATTEMPT_PROFILE = RestorationLabelAttemptProfile(
    config_path=CANONICAL_CONFIG_PATH,
    frozen_config_sha256=FROZEN_CONFIG_SHA256,
    source_parent_git_commit=SOURCE_PARENT_GIT_COMMIT,
    attempt_id=CANONICAL_ATTEMPT_ID,
    output_dir=CANONICAL_OUTPUT_DIR,
    ledger_path=CANONICAL_LEDGER_PATH,
    archive_path=CANONICAL_ARCHIVE_PATH,
    hf_repo=CANONICAL_HF_REPO,
    hf_tag=CANONICAL_HF_TAG,
    hf_path=CANONICAL_HF_PATH,
    pass_outcome=PASS_OUTCOME,
    invalid_outcome=INVALID_OUTCOME,
    aggregate_status="COMPLETED_RESTORATION_V2_2_EAGER_LABELS_V1",
    attempt_revision="v1_initial",
    preregistration_status=(
        "source_only_frozen_before_any_restoration_label_forward"
    ),
    expected_source_paths=EXPECTED_SOURCE_PATHS,
)


def _v2_repair_profile() -> RestorationLabelAttemptProfile:
    return RestorationLabelAttemptProfile(
        config_path=V2_REPAIR_CONFIG_PATH,
        frozen_config_sha256=V2_REPAIR_FROZEN_CONFIG_SHA256,
        source_parent_git_commit=V2_REPAIR_SOURCE_PARENT_GIT_COMMIT,
        attempt_id=V2_REPAIR_ATTEMPT_ID,
        output_dir=V2_REPAIR_OUTPUT_DIR,
        ledger_path=V2_REPAIR_LEDGER_PATH,
        archive_path=V2_REPAIR_ARCHIVE_PATH,
        hf_repo=CANONICAL_HF_REPO,
        hf_tag=V2_REPAIR_HF_TAG,
        hf_path=V2_REPAIR_HF_PATH,
        pass_outcome=V2_REPAIR_PASS_OUTCOME,
        invalid_outcome=V2_REPAIR_INVALID_OUTCOME,
        aggregate_status=V2_REPAIR_AGGREGATE_STATUS,
        attempt_revision="v2_preclaim_repair",
        preregistration_status=(
            "repair_source_frozen_after_zero_forward_v1_before_any_v2_forward"
        ),
        expected_source_paths=V2_REPAIR_EXPECTED_SOURCE_PATHS,
        supersedes_attempt_id=CANONICAL_ATTEMPT_ID,
        canonical_model_dir=V2_REPAIR_CANONICAL_MODEL_DIR,
    )


def label_attempt_profile_for_config_path(
    value: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> RestorationLabelAttemptProfile:
    supplied = Path(value)
    if repository_root is not None:
        root = Path(repository_root).resolve()
        resolved = supplied.resolve()
        for profile in (V1_ATTEMPT_PROFILE, _v2_repair_profile()):
            if resolved == (root / profile.config_path).resolve():
                return profile
    else:
        normalized = supplied.as_posix()
        for profile in (V1_ATTEMPT_PROFILE, _v2_repair_profile()):
            if normalized == profile.config_path:
                return profile
    raise ValueError("restoration-label contract path is not an authorized profile")


def label_attempt_profile_for_id(value: str) -> RestorationLabelAttemptProfile:
    for profile in (V1_ATTEMPT_PROFILE, _v2_repair_profile()):
        if value == profile.attempt_id:
            return profile
    raise ValueError("restoration-label attempt id is not authorized")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
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


def load_strict_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_bytes(),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} drifted: expected {expected!r}, got {actual!r}")


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"{label} is not a canonical repository-relative path")
    if path.as_posix() != value:
        raise ValueError(f"{label} must use canonical POSIX syntax")
    return value


def _git_blob(repository_root: Path, revision: str, relative: str) -> bytes:
    if revision != "HEAD" and _GIT_SHA.fullmatch(revision) is None:
        raise ValueError("Git revision must be a full commit SHA")
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


def _bound_repository_json(
    repository_root: Path,
    binding: Any,
    *,
    label: str,
    source_parent_git_commit: str,
) -> Mapping[str, Any]:
    record = _mapping(binding, label)
    relative = _safe_relative_path(record.get("path"), f"{label}.path")
    digest = record.get("sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label}.sha256 is invalid")
    path = (repository_root / relative).resolve()
    if repository_root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"{label}.path is missing or escapes the repository")
    payload = path.read_bytes()
    _equal(sha256_bytes(payload), digest, f"{label} SHA256")
    _equal(_git_blob(repository_root, "HEAD", relative), payload, f"{label} HEAD blob")
    _equal(
        _git_blob(repository_root, source_parent_git_commit, relative),
        payload,
        f"{label} source-parent blob",
    )
    value = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    return _mapping(value, f"{label} value")


@dataclass(frozen=True)
class RestorationV22LabelContract:
    path: Path
    data: Mapping[str, Any]
    sha256: str
    profile: RestorationLabelAttemptProfile | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
    ) -> "RestorationV22LabelContract":
        root = Path(repository_root).resolve()
        resolved = Path(path).resolve()
        profile = label_attempt_profile_for_config_path(
            resolved,
            repository_root=root,
        )
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError("restoration-label contract must use a canonical config")
        digest = sha256_file(resolved)
        _equal(
            digest,
            profile.frozen_config_sha256,
            "restoration-label config SHA256",
        )
        _equal(
            _git_blob(root, "HEAD", profile.config_path),
            resolved.read_bytes(),
            "restoration-label config HEAD blob",
        )
        data = load_strict_json_object(resolved)
        contract = cls(path=resolved, data=data, sha256=digest, profile=profile)
        contract.validate(repository_root=root)
        return contract

    def validate(self, *, repository_root: Path) -> None:
        data = self.data
        profile = self.profile or label_attempt_profile_for_config_path(
            self.path,
            repository_root=repository_root,
        )
        _equal(
            self.sha256,
            profile.frozen_config_sha256,
            "restoration-label config SHA256",
        )
        _equal(data.get("schema_version"), "0.1.0", "schema version")
        _equal(data.get("protocol_id"), PROTOCOL_ID, "protocol id")
        _equal(
            data.get("preregistration_status"),
            profile.preregistration_status,
            "preregistration status",
        )
        authorization = _mapping(data.get("authorization"), "authorization")
        _equal(
            authorization.get("source_parent_git_commit"),
            profile.source_parent_git_commit,
            "source parent commit",
        )
        for key in (
            "fresh_immutable_parent_archive_validation_required_before_runtime_import",
            "parent_canonical_action_is_the_only_teacher_target",
        ):
            _equal(authorization.get(key), True, f"authorization.{key}")
        _equal(
            authorization.get("new_reference_generation_allowed"),
            False,
            "new reference generation permission",
        )
        _equal(
            authorization.get("authorization_scope"),
            "train_development_restoration_labels_only",
            "authorization scope",
        )

        artifact_binding = _mapping(
            authorization.get("v2_2_parent_artifact"), "parent artifact binding"
        )
        artifact = _bound_repository_json(
            repository_root,
            artifact_binding,
            label="parent artifact",
            source_parent_git_commit=profile.source_parent_git_commit,
        )
        artifact_result = _mapping(artifact.get("result"), "parent artifact result")
        source = _mapping(artifact.get("source_execution"), "parent source")
        hf = _mapping(artifact.get("hf_artifact"), "parent HF artifact")
        raw = _mapping(artifact.get("raw_archive"), "parent raw archive")
        expected_artifact = {
            "status": artifact.get("status"),
            "outcome": artifact_result.get("outcome"),
            "source_git_commit": source.get("source_git_commit"),
            "run_contract_sha256": source.get("run_contract_sha256"),
            "hf_repo": hf.get("repo"),
            "hf_revision": hf.get("immutable_revision"),
            "hf_path": hf.get("path"),
            "raw_sha256": raw.get("sha256"),
            "raw_size_bytes": raw.get("size_bytes"),
            "raw_file_count": raw.get("file_count"),
            "raw_tree_sha256": raw.get("tree_inventory_sha256"),
        }
        required_artifact = {
            "status": artifact_binding.get("required_status"),
            "outcome": artifact_binding.get("required_outcome"),
            "source_git_commit": artifact_binding.get("required_source_git_commit"),
            "run_contract_sha256": artifact_binding.get("required_run_contract_sha256"),
            "hf_repo": artifact_binding.get("required_hf_repo"),
            "hf_revision": artifact_binding.get("required_hf_revision"),
            "hf_path": artifact_binding.get("required_hf_path"),
            "raw_sha256": artifact_binding.get("required_raw_sha256"),
            "raw_size_bytes": artifact_binding.get("required_raw_size_bytes"),
            "raw_file_count": artifact_binding.get("required_raw_file_count"),
            "raw_tree_sha256": artifact_binding.get("required_raw_tree_sha256"),
        }
        _equal(expected_artifact, required_artifact, "parent artifact identity")

        summary_binding = _mapping(
            authorization.get("v2_2_parent_summary"), "parent summary binding"
        )
        summary = _bound_repository_json(
            repository_root,
            summary_binding,
            label="parent summary",
            source_parent_git_commit=profile.source_parent_git_commit,
        )
        metrics = _mapping(summary.get("metrics"), "parent summary metrics")
        _equal(summary.get("status"), summary_binding.get("required_status"), "parent status")
        _equal(summary.get("outcome"), summary_binding.get("required_outcome"), "parent outcome")
        required_metrics = {
            "fixed_state_denominator": summary_binding.get("required_state_count"),
            "repeat_canonical_action_agreement_count": summary_binding.get(
                "required_repeat_agreement_count"
            ),
            "finite_logit_state_count": summary_binding.get(
                "required_finite_logit_state_count"
            ),
            "restoration_label_count": summary_binding.get(
                "required_restoration_label_count"
            ),
            "confirm_state_access_count": summary_binding.get(
                "required_confirm_state_access_count"
            ),
        }
        _equal(
            {key: metrics.get(key) for key in required_metrics},
            required_metrics,
            "parent summary metric boundary",
        )

        data_contract = _mapping(data.get("data"), "data")
        _equal(data_contract.get("fixed_state_denominator"), EXPECTED_STATE_COUNT, "state count")
        _equal(
            data_contract.get("role_state_counts"),
            {"v2_label_train": 30, "v2_development": 15},
            "role state counts",
        )
        _equal(
            data_contract.get("confirm_state_prompt_image_or_action_access_allowed"),
            False,
            "confirm access permission",
        )

        enumeration = _mapping(data.get("enumeration"), "enumeration")
        _equal(enumeration.get("raw_distance_domain"), "complete_power_set", "distance domain")
        _equal(enumeration.get("primary_deployment_budget_slots"), 2, "primary budget")
        _equal(
            enumeration.get("raw_distance_row_counts"),
            {"v2_label_train": 280, "v2_development": 140, "total": EXPECTED_DISTANCE_ROWS},
            "distance row counts",
        )
        _equal(
            enumeration.get("deployment_conditional_edge_counts"),
            {"v2_label_train": 290, "v2_development": 145, "total": EXPECTED_DEPLOYMENT_EDGES},
            "deployment edge counts",
        )
        _equal(
            enumeration.get("full_hypercube_edge_counts"),
            {"v2_label_train": 480, "v2_development": 240, "total": EXPECTED_FULL_EDGES},
            "full edge counts",
        )
        _equal(enumeration.get("full_pair_interaction_count"), 465, "interaction count")

        reduction = _mapping(data.get("reduction"), "reduction")
        _equal(reduction.get("exact_oracle_tie_epsilon"), 0.0, "tie epsilon")
        _equal(reduction.get("greedy_stopping_threshold"), 0.0, "greedy threshold")
        _equal(
            reduction.get("conditional_edge_weighting_for_gate_training"),
            "deferred_to_separate_gate_training_contract",
            "gate-label weighting boundary",
        )

        schedule = _mapping(data.get("operation_schedule"), "operation schedule")
        expected_schedule = {
            "generation_call_count": 0,
            "reference_teacher_forward_count": 45,
            "reference_repeat_teacher_forward_count": 45,
            "non_reference_coalition_teacher_forward_count": 375,
            "total_teacher_forward_call_count": EXPECTED_TEACHER_FORWARDS,
            "total_teacher_forward_example_count": EXPECTED_TEACHER_FORWARDS,
            "coalition_kl_measurement_count": 375,
            "reference_repeat_kl_measurement_count": 45,
            "total_kl_measurement_count": EXPECTED_KL_MEASUREMENTS,
            "raw_distance_row_count": EXPECTED_DISTANCE_ROWS,
            "deployment_conditional_label_count": EXPECTED_DEPLOYMENT_EDGES,
            "primary_exact_subset_oracle_count": EXPECTED_PRIMARY_ORACLES,
            "retry_count": 0,
            "top_up_count": 0,
            "resume_allowed": False,
            "completed_state_regeneration_allowed": False,
            "incomplete_attempt_retry_allowed": False,
        }
        for key, expected in expected_schedule.items():
            _equal(schedule.get(key), expected, f"operation_schedule.{key}")
        maximum_repeat = schedule.get("maximum_reference_repeat_kl")
        if not isinstance(maximum_repeat, (int, float)) or isinstance(maximum_repeat, bool):
            raise ValueError("maximum reference repeat KL must be numeric")
        _equal(float(maximum_repeat), 1e-4, "maximum reference repeat KL")

        topology = _mapping(data.get("worker_topology"), "worker topology")
        workers = _sequence(topology.get("workers"), "workers")
        _equal(topology.get("worker_count"), 2, "worker count")
        worker_state_counts = [
            len(_mapping(worker, "worker").get("state_indices", []))
            for worker in workers
        ]
        _equal(worker_state_counts, [23, 22], "worker state counts")
        union = sorted(
            index
            for worker in workers
            for index in _mapping(worker, "worker").get("state_indices", [])
        )
        _equal(union, list(range(EXPECTED_STATE_COUNT)), "worker state union")

        prohibited = _mapping(data.get("prohibited_work"), "prohibited work")
        if any(value not in {0, False} for value in prohibited.values()):
            raise ValueError("all prohibited-work counts and permissions must be zero/false")

        execution = _mapping(data.get("execution"), "execution")
        expected_execution = {
            "attempt_id": profile.attempt_id,
            "canonical_persistent_output_dir": str(profile.output_dir),
            "canonical_global_attempt_ledger": str(profile.ledger_path),
            "canonical_raw_archive": str(profile.archive_path),
        }
        if profile.supersedes_attempt_id is not None:
            expected_execution["attempt_revision"] = profile.attempt_revision
        for key, expected in expected_execution.items():
            _equal(execution.get(key), expected, f"execution.{key}")

        destination = _mapping(data.get("artifact_destination"), "artifact destination")
        _equal(destination.get("repo"), profile.hf_repo, "HF repo")
        _equal(destination.get("tag"), profile.hf_tag, "HF tag")
        _equal(destination.get("raw_path"), profile.hf_path, "HF raw path")
        _equal(destination.get("immutable_revision"), None, "source-only HF revision")

        source_paths = tuple(data.get("formal_run_source_inventory_paths", ()))
        _equal(
            source_paths,
            profile.expected_source_paths,
            "formal source inventory paths",
        )
        for relative in source_paths:
            _safe_relative_path(relative, "formal source path")
            path = repository_root / relative
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"formal source path is missing: {relative}")

        if profile.supersedes_attempt_id is None:
            if "repair_lineage" in data:
                raise ValueError("v1 contract must not contain repair lineage")
        else:
            _equal(
                authorization.get("canonical_model_projection_dir"),
                str(profile.canonical_model_dir),
                "repair canonical model projection",
            )
            repair = _mapping(data.get("repair_lineage"), "repair lineage")
            expected_repair = {
                "supersedes_attempt_id": CANONICAL_ATTEMPT_ID,
                "superseded_status": "LABEL_ATTEMPT_INVALID",
                "superseded_outcome": INVALID_OUTCOME,
                "superseded_attempted_state_count": 0,
                "superseded_completed_state_count": 0,
                "superseded_teacher_forward_count": 0,
                "superseded_kl_measurement_count": 0,
                "superseded_attempt_reuse_or_resume_allowed": False,
                "repair_reason": (
                    "wrong_model_projection_path_and_postclaim_snapshot_validation"
                ),
            }
            for key, expected in expected_repair.items():
                _equal(repair.get(key), expected, f"repair_lineage.{key}")
            summary_binding = repair.get("superseded_attempt_summary")
            summary = _bound_repository_json(
                repository_root,
                summary_binding,
                label="superseded attempt summary",
                source_parent_git_commit=profile.source_parent_git_commit,
            )
            _equal(
                {
                    "status": summary.get("status"),
                    "outcome": summary.get("outcome"),
                    "attempt_id": summary.get("attempt_id"),
                    "attempted_state_count": summary.get("attempted_state_count"),
                    "completed_state_count": summary.get("completed_state_count"),
                    "teacher_forward_count": _mapping(
                        summary.get("formal_outputs"),
                        "superseded formal outputs",
                    ).get("teacher_forward_count"),
                    "kl_measurement_count": _mapping(
                        summary.get("formal_outputs"),
                        "superseded formal outputs",
                    ).get("kl_measurement_count"),
                },
                {
                    "status": "LABEL_ATTEMPT_INVALID",
                    "outcome": INVALID_OUTCOME,
                    "attempt_id": CANONICAL_ATTEMPT_ID,
                    "attempted_state_count": 0,
                    "completed_state_count": 0,
                    "teacher_forward_count": 0,
                    "kl_measurement_count": 0,
                },
                "superseded attempt summary boundary",
            )


def validate_contract(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    contract = RestorationV22LabelContract.load(
        path,
        repository_root=repository_root,
    )
    return {
        "status": "VALID_RESTORATION_V2_2_EAGER_LABEL_SOURCE_CONTRACT",
        "protocol_id": PROTOCOL_ID,
        "config_sha256": contract.sha256,
        "source_parent_git_commit": contract.profile.source_parent_git_commit,
        "attempt_id": contract.profile.attempt_id,
        "supersedes_attempt_id": contract.profile.supersedes_attempt_id,
        "fixed_state_denominator": EXPECTED_STATE_COUNT,
        "raw_distance_row_count": EXPECTED_DISTANCE_ROWS,
        "deployment_conditional_label_count": EXPECTED_DEPLOYMENT_EDGES,
        "full_hypercube_edge_count": EXPECTED_FULL_EDGES,
        "primary_exact_subset_oracle_count": EXPECTED_PRIMARY_ORACLES,
        "teacher_forward_count": EXPECTED_TEACHER_FORWARDS,
        "kl_measurement_count": EXPECTED_KL_MEASUREMENTS,
        "confirm_locked": True,
    }
