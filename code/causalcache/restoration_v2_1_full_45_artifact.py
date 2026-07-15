"""Raw evidence packaging for the v2.1 fixed-45 substrate run."""

from __future__ import annotations

import io
import os
import re
import subprocess
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causalcache.data.restoration_v2_screening import ScreeningState
from causalcache.policy.gui_owl_v2_1_runtime import (
    GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS,
)
from causalcache.restoration_v2_1_pilot_artifact import (
    canonical_json_bytes,
    pretty_json_bytes,
    require_source_commit_ancestor,
    sha256_bytes,
    sha256_file,
    strict_json_object_bytes,
    validate_clean_pushed_main,
    _validate_runtime_identity,
)
from scripts.run_restoration_v2_1_full_45_substrate import (
    ATTEMPT_STATUS,
    FORBIDDEN_OPERATION_COUNTS,
    GLOBAL_ATTEMPT_STATUS,
    RUNTIME_METADATA_KEYS,
    RUN_STATUS,
    SCHEMA_VERSION as RUN_SCHEMA_VERSION,
    Full45Gate,
    _duration_seconds,
    _full45_projection,
    _prepare_layout,
    _validate_runtime_identity_record,
    _validate_terminal_state_record,
    aggregate_full45_gate,
)


ARTIFACT_SCHEMA_VERSION = "0.1.0"
PROTOCOL_ID = "causalcache_restoration_v2_1_full_45_substrate"
ARTIFACT_TYPE = "private_hf_restoration_v2_1_full_45_substrate_raw_evidence"
CANONICAL_ATTEMPT_ID = "restoration-v2-1-full-45-substrate-v1"
CANONICAL_RAW_OUTPUT_DIR = Path(
    "/data/experiments/causalcache/restoration-v2-1-full-45-substrate-v1"
)
CANONICAL_GLOBAL_LEDGER_PATH = Path(
    "/data/experiments/causalcache/"
    ".restoration-v2-1-full-45-substrate-v1.attempt.json"
)
CANONICAL_RAW_ARCHIVE_PATH = Path(
    "/data/experiments/causalcache/"
    "restoration-v2-1-full-45-substrate-v1.raw.tar"
)
CANONICAL_GIT_ARTIFACT_PATH = (
    "data/results/restoration_v2_1_full_45_substrate/artifact.json"
)
CANONICAL_HF_REPO = (
    "gavinlaw/causalcache-restoration-v2-1-full-45-substrate-mobile"
)
CANONICAL_HF_TAG = "v2.1-full-45-substrate-v1"
CANONICAL_HF_PATH = "raw/restoration-v2-1-full-45-substrate-v1.tar"
ARCHIVE_FORMAT = "ustar"
ARCHIVE_MEMBER_PREFIX = CANONICAL_ATTEMPT_ID
ARCHIVE_LEDGER_NAME = "global_attempt_ledger.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"
RUNTIME_IDENTITY_FILENAME = "runtime_identity.json"
AGGREGATE_FILENAME = "aggregate.json"
STATE_DIRECTORY = "states"
ATTEMPT_DIRECTORY = "attempts"
FULL_45_CONFIG_PATH = "code/configs/causalcache_restoration_v2_1_full_45.json"
PILOT_CONTRACT_PATH = "code/configs/causalcache_restoration_v2_1_pilot.json"
V2_CONFIG_PATH = "code/configs/causalcache_restoration_v2.json"
SELECTION_MANIFEST_PATH = "data/manifests/restoration_v2_selection.json"
PILOT_ARTIFACT_PATH = (
    "data/results/restoration_v2_1_interface_pilot/artifact.json"
)
PROCESSOR_ARTIFACT_PATH = (
    "data/results/restoration_v2_1_processor_preflight/artifact.json"
)
CANONICAL_REMOTE_URL = "https://github.com/luojiaxuan/CausalCache.git"
EXPECTED_RUN_SOURCE_PATHS = (
    FULL_45_CONFIG_PATH,
    "code/causalcache/restoration_v2_1_full_45_contract.py",
    "code/causalcache/restoration_v2_1_full_45_artifact.py",
    "code/scripts/run_restoration_v2_1_full_45_substrate.py",
    "code/scripts/run_restoration_v2_1_interface_pilot.py",
    "code/scripts/manage_restoration_v2_1_full_45_artifact.py",
    "code/scripts/validate_restoration_v2_1_full_45_contract.py",
    "code/causalcache/restoration_v2_1_contract.py",
    "code/causalcache/restoration_v2_1_pilot_artifact.py",
    "code/scripts/manage_restoration_v2_1_pilot_artifact.py",
    "code/causalcache/restoration_v2_1_processor_audit.py",
    "code/scripts/audit_gui_owl_v2_1_processor.py",
    "code/causalcache/data/restoration_v2_1_processor_inputs.py",
    "code/causalcache/data/restoration_v2_screening.py",
    "code/causalcache/policy/gui_owl_v2_1.py",
    "code/causalcache/policy/gui_owl_v2_1_runtime.py",
    "code/causalcache/restoration_v2_gpu_kl.py",
    "code/causalcache/data/guiodyssey_restoration_v2.py",
    "code/causalcache/data/restoration_v2_selection.py",
    "code/causalcache/data/guiodyssey.py",
    "code/causalcache/data/guiodyssey_independent.py",
    "code/causalcache/low_fidelity_v2.py",
    "code/causalcache/restoration_v2_contract.py",
    "code/causalcache/restoration_v2_text_backend.py",
    "code/causalcache/schema.py",
    "code/causalcache/policy/gui_owl_v2.py",
    "code/causalcache/policy/gui_owl_v2_runtime.py",
    "code/causalcache/policy/gui_owl_v2_vision.py",
    "code/causalcache/restoration_v2_baselines.py",
    PILOT_CONTRACT_PATH,
    V2_CONFIG_PATH,
    "code/configs/restoration_v2_ocr_backend.json",
    "code/configs/gui_owl_1_5_8b_snapshot.json",
    SELECTION_MANIFEST_PATH,
    "data/manifests/restoration_v2_ocr_backend.json",
    "data/manifests/restoration_v2_derived_artifact.json",
    "data/manifests/restoration_v2_real_screen_source.json",
    PILOT_ARTIFACT_PATH,
    PROCESSOR_ARTIFACT_PATH,
)
EXPECTED_STATE_COUNT = 45
EXPECTED_GENERATION_COUNT = 90
EXPECTED_TEACHER_FORWARD_COUNT = 135
EXPECTED_KL_MEASUREMENT_COUNT = 90
PASS_OUTCOME = "PASS_V2_1_FULL_45_SUBSTRATE"
NO_GO_OUTCOME = "NO_GO_V2_1_FULL_45_SUBSTRATE"
INVALID_OUTCOME = "INVALID_V2_1_FULL_45_SUBSTRATE"
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
BFLOAT16_MIN_FINITE = -3.3895313892515355e38
EXPECTED_COMPUTATION_SCHEDULE = {
    "per_state_maximum": {
        "full_history_generation_calls": 2,
        "teacher_forward_calls_on_success": 3,
        "kl_measurements_on_success": 2,
    },
    "planned_counts_if_all_45_states_reach_success_stage": {
        "generation_call_count": 90,
        "teacher_forward_count": 135,
        "kl_measurement_count": 90,
    },
    "maximum_counts_for_entire_attempt": {
        "generation_call_count": 90,
        "teacher_forward_count": 135,
        "kl_measurement_count": 90,
    },
    "generation_order_per_state": [
        "full_history_reference_repeat_1",
        "full_history_reference_repeat_2",
    ],
    "teacher_order_on_success_per_state": [
        "full_history_reference_1",
        "full_history_reference_2",
        "summary_only",
    ],
    "kl_order_on_success_per_state": [
        "repeat_reference_kl",
        "summary_reference_kl",
    ],
    "automatic_retry_allowed": False,
    "state_retry_count": 0,
    "top_up_count": 0,
    "completed_state_regeneration_allowed": False,
    "incomplete_attempt_retry_allowed": False,
    "sample_mutation_allowed": False,
    "resume_allowed": True,
    "resume_only_skips_existing_terminal_state_records": True,
    "resume_may_attempt_only_states_without_any_attempt_marker": True,
    "attempt_marker_without_terminal_state_invalidates_entire_attempt": True,
}
EXPECTED_PROMOTION = {
    "pass_authorizes_only": (
        "freeze_and_audit_independent_v2_1_restoration_confirm_source"
    ),
    "pass_does_not_authorize_automatic_confirm_execution": True,
    "confirm_remains_locked_until_new_source_is_committed_and_pushed": True,
    "no_go_stops_v2_1_restoration_confirm": True,
    "source_only_validator_authorizes_policy_or_gpu_execution": False,
}


@dataclass(frozen=True)
class Full45Evidence:
    files: Mapping[str, bytes]
    run_contract: Mapping[str, Any]
    source_git_commit: str
    run_contract_sha256: str
    outcome: str
    status: str
    aggregate_sha256: str
    run_manifest_sha256: str
    global_attempt_ledger_sha256: str
    completed_state_count: int
    attempted_state_count: int
    generation_call_count: int
    teacher_forward_count: int
    kl_measurement_count: int
    inventory: tuple[Mapping[str, Any], ...]
    tree_inventory_sha256: str


def _safe_relative_path(value: Any, *, label: str) -> str:
    from pathlib import PurePosixPath

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} is unsafe")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"{label} is not canonical POSIX syntax")
    return normalized


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys drifted")


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value,
        Sequence,
    ):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA256")
    return value


def _fixed_gate() -> Full45Gate:
    return Full45Gate(
        minimum_screening_states=20,
        minimum_parse_coverage=0.99,
        minimum_finite_logit_coverage=1.0,
        minimum_repeat_canonical_action_agreement=1.0,
        minimum_memory_sensitive_states=8,
        pass_outcome=PASS_OUTCOME,
        fail_outcome=NO_GO_OUTCOME,
        invalid_outcome=INVALID_OUTCOME,
    )


def _states_from_run_contract(
    run_contract: Mapping[str, Any],
) -> tuple[ScreeningState, ...]:
    projections = _sequence(run_contract.get("states"), "full-45 states")
    if len(projections) != EXPECTED_STATE_COUNT:
        raise ValueError("run contract denominator is not exactly 45")
    states: list[ScreeningState] = []
    canonical: list[dict[str, Any]] = []
    for index, value in enumerate(projections):
        projection = _mapping(value, f"full-45 state {index}")
        _exact_keys(
            projection,
            {
                "index",
                "role",
                "trajectory_id",
                "decision_step_id",
                "state_id",
                "candidate_event_step_ids",
            },
            f"full-45 state {index}",
        )
        expected_role = "v2_label_train" if index < 30 else "v2_development"
        trajectory_id = projection.get("trajectory_id")
        decision_step_id = projection.get("decision_step_id")
        candidates = projection.get("candidate_event_step_ids")
        if (
            projection.get("index") != index
            or projection.get("role") != expected_role
            or not isinstance(trajectory_id, str)
            or not trajectory_id
            or decision_step_id not in {4, 5, 6}
            or candidates != list(range(1, int(decision_step_id) - 1))
            or projection.get("state_id")
            != f"{trajectory_id}:decision_step:{int(decision_step_id):03d}"
        ):
            raise ValueError("full-45 state order, role, or dependency set drifted")
        state = ScreeningState(
            index=index,
            role=expected_role,
            trajectory_id=trajectory_id,
            decision_step_id=int(decision_step_id),
            candidate_event_step_ids=tuple(candidates),
        )
        expected_projection = _full45_projection(state, index)
        if dict(projection) != expected_projection:
            raise ValueError("full-45 state projection is not canonical")
        canonical.append(expected_projection)
        states.append(state)
    from causalcache.restoration_v2_1_full_45_contract import (
        FULL_45_PROJECTION_SHA256,
    )

    if sha256_bytes(canonical_json_bytes(canonical)) != FULL_45_PROJECTION_SHA256:
        raise ValueError("full-45 state projection hash drifted")
    return tuple(states)


def _validate_source_identity(
    run_contract: Mapping[str, Any],
    *,
    expected_source_git_commit: str | None,
) -> str:
    git = _mapping(run_contract.get("git_identity"), "run contract git_identity")
    _exact_keys(
        git,
        {
            "branch",
            "commit",
            "origin_main",
            "remote_main",
            "remote_url",
            "worktree",
        },
        "run contract git_identity",
    )
    commit = git.get("commit")
    if (
        not isinstance(commit, str)
        or GIT_SHA_PATTERN.fullmatch(commit) is None
        or git.get("branch") != "main"
        or git.get("origin_main") != commit
        or git.get("remote_main") != commit
        or git.get("remote_url") != CANONICAL_REMOTE_URL
        or git.get("worktree") != "clean_including_untracked"
    ):
        raise ValueError("full-45 clean pushed Git identity drifted")
    if expected_source_git_commit is not None and commit != expected_source_git_commit:
        raise ValueError("full-45 source Git commit differs from expected source X")
    inventory = _sequence(
        run_contract.get("source_inventory"),
        "run contract source_inventory",
    )
    paths: list[str] = []
    for index, item in enumerate(inventory):
        record = _mapping(item, f"source_inventory[{index}]")
        _exact_keys(record, {"path", "sha256", "git_commit"}, "source record")
        path = _safe_relative_path(record.get("path"), label="source record path")
        _require_sha256(record.get("sha256"), name="source record SHA256")
        if record.get("git_commit") != commit:
            raise ValueError("source inventory commit binding drifted")
        paths.append(path)
    if paths != list(EXPECTED_RUN_SOURCE_PATHS):
        raise ValueError("full-45 source inventory path/order drifted")
    return commit


def _validate_execution_argv(
    value: Any,
    *,
    run_contract: Mapping[str, Any],
) -> None:
    argv = list(_sequence(value, "full-45 execution_argv"))
    if (
        len(argv) < 2
        or any(not isinstance(item, str) or not item for item in argv)
        or not Path(argv[0]).is_absolute()
        or argv[1]
        != "/data/CausalCache/code/scripts/run_restoration_v2_1_full_45_substrate.py"
        or "--resume" in argv
    ):
        raise ValueError("full-45 execution argv prefix or resume policy drifted")
    tokens = argv[2:]
    if len(tokens) % 2:
        raise ValueError("full-45 execution argv must contain flag-value pairs")
    pairs = dict(zip(tokens[::2], tokens[1::2], strict=True))
    flags = {
        "--repository-root",
        "--contract",
        "--pilot-evidence",
        "--processor-preflight",
        "--derived-artifact-root",
        "--scientific-config",
        "--selection-manifest",
        "--ocr-backend-config",
        "--model-dir",
        "--device",
        "--host-alias",
        "--host-hostname",
        "--container-id",
        "--container-image-digest",
        "--output-dir",
    }
    if len(pairs) != len(tokens) // 2 or set(pairs) != flags:
        raise ValueError("full-45 execution argv flags drifted or duplicated")
    attempt = _mapping(run_contract.get("attempt_identity"), "attempt identity")
    canonical = _mapping(run_contract.get("canonical_inputs"), "canonical inputs")
    pilot_external = _mapping(
        _mapping(canonical.get("pilot_evidence"), "pilot evidence input").get(
            "external"
        ),
        "pilot external input",
    )
    processor_external = _mapping(
        _mapping(
            canonical.get("processor_preflight"),
            "processor preflight input",
        ).get("external"),
        "processor external input",
    )
    policy = _mapping(run_contract.get("policy"), "policy")
    expected = {
        "--repository-root": "/data/CausalCache",
        "--contract": f"/data/CausalCache/{FULL_45_CONFIG_PATH}",
        "--pilot-evidence": str(pilot_external.get("path")),
        "--processor-preflight": str(processor_external.get("path")),
        "--scientific-config": f"/data/CausalCache/{V2_CONFIG_PATH}",
        "--selection-manifest": f"/data/CausalCache/{SELECTION_MANIFEST_PATH}",
        "--ocr-backend-config": (
            "/data/CausalCache/code/configs/restoration_v2_ocr_backend.json"
        ),
        "--model-dir": str(policy.get("model_dir")),
        "--device": str(attempt.get("canonical_device")),
        "--host-alias": str(attempt.get("canonical_host_alias")),
        "--host-hostname": str(attempt.get("canonical_host_hostname")),
        "--container-id": str(attempt.get("canonical_container_id")),
        "--container-image-digest": str(
            attempt.get("canonical_container_image_digest")
        ),
        "--output-dir": str(CANONICAL_RAW_OUTPUT_DIR),
    }
    if any(pairs.get(flag) != expected_value for flag, expected_value in expected.items()):
        raise ValueError("full-45 execution argv canonical binding drifted")
    if not Path(pairs["--derived-artifact-root"]).is_absolute():
        raise ValueError("full-45 derived artifact root must be absolute")


def _validate_run_contract(
    run_contract: Mapping[str, Any],
    *,
    expected_source_git_commit: str | None,
) -> tuple[str, tuple[ScreeningState, ...]]:
    _exact_keys(
        run_contract,
        {
            "schema_version",
            "protocol_id",
            "contract_sha256",
            "git_identity",
            "source_inventory",
            "parent_authorization",
            "canonical_inputs",
            "artifact",
            "policy",
            "runtime_identity",
            "execution_argv",
            "operational_argv_policy",
            "seed_policy",
            "output_dir",
            "attempt_identity",
            "states",
            "computation_schedule",
            "prohibited_operation_counts",
            "confirm_state",
            "promotion",
        },
        "full-45 run contract",
    )
    from causalcache.restoration_v2_1_full_45_contract import (
        FROZEN_RESTORATION_V2_1_FULL_45_SHA256,
    )

    if (
        run_contract.get("schema_version") != RUN_SCHEMA_VERSION
        or run_contract.get("protocol_id") != PROTOCOL_ID
        or run_contract.get("contract_sha256")
        != FROZEN_RESTORATION_V2_1_FULL_45_SHA256
        or run_contract.get("output_dir") != str(CANONICAL_RAW_OUTPUT_DIR)
        or run_contract.get("computation_schedule")
        != EXPECTED_COMPUTATION_SCHEDULE
        or run_contract.get("prohibited_operation_counts")
        != FORBIDDEN_OPERATION_COUNTS
        or run_contract.get("confirm_state")
        != "LOCKED_NO_PROMPT_IMAGE_DECODER_OR_POLICY_ACCESS"
        or run_contract.get("promotion") != EXPECTED_PROMOTION
    ):
        raise ValueError("full-45 run contract identity or operation lock drifted")
    commit = _validate_source_identity(
        run_contract,
        expected_source_git_commit=expected_source_git_commit,
    )
    parent = _mapping(
        run_contract.get("parent_authorization"),
        "parent_authorization",
    )
    _exact_keys(
        parent,
        {"pilot_artifact_validation", "processor_audit"},
        "parent_authorization",
    )
    pilot = _mapping(
        parent.get("pilot_artifact_validation"),
        "pilot artifact validation",
    )
    _exact_keys(
        pilot,
        {
            "status",
            "source_git_commit",
            "outcome",
            "file_count",
            "tree_inventory_sha256",
            "archive_hash_verified",
            "current_git_commit",
            "evidence_source_kind",
        },
        "pilot artifact validation",
    )
    if (
        pilot.get("status")
        != "VALID_RESTORATION_V2_1_INTERFACE_PILOT_ARTIFACT"
        or pilot.get("outcome") != "PASS_V2_1_INTERFACE_PILOT"
        or pilot.get("file_count") != 34
        or pilot.get("archive_hash_verified") is not True
        or pilot.get("current_git_commit") != commit
        or pilot.get("evidence_source_kind") != "raw_archive"
        or not isinstance(pilot.get("source_git_commit"), str)
        or GIT_SHA_PATTERN.fullmatch(pilot["source_git_commit"]) is None
    ):
        raise ValueError("fixed-15 pilot promotion authorization drifted")
    _require_sha256(
        pilot.get("tree_inventory_sha256"),
        name="pilot tree inventory SHA256",
    )
    processor = _mapping(parent.get("processor_audit"), "processor audit")
    _exact_keys(
        processor,
        {
            "status",
            "prompt_count",
            "prompt_records_sha256",
            "shape_records_sha256",
            "teacher_golden_records_sha256",
            "processor_classes_sha256",
            "evidence_git_commit",
            "current_git_commit",
            "validation_mode",
        },
        "processor audit",
    )
    if (
        processor.get("status")
        != "PASSED_RESTORATION_V2_1_90_PROMPT_PROCESSOR_PREFLIGHT"
        or processor.get("prompt_count") != 90
        or processor.get("current_git_commit") != commit
        or processor.get("validation_mode") != "reuse"
        or not isinstance(processor.get("evidence_git_commit"), str)
        or GIT_SHA_PATTERN.fullmatch(processor["evidence_git_commit"]) is None
    ):
        raise ValueError("processor promotion authorization drifted")
    for key in (
        "prompt_records_sha256",
        "shape_records_sha256",
        "teacher_golden_records_sha256",
        "processor_classes_sha256",
    ):
        _require_sha256(processor.get(key), name=f"processor audit {key}")

    canonical = _mapping(run_contract.get("canonical_inputs"), "canonical_inputs")
    _exact_keys(
        canonical,
        {
            "contract",
            "pilot_contract",
            "pilot_evidence",
            "processor_preflight",
            "scientific_config",
            "selection_manifest",
            "ocr_backend_config",
            "snapshot_manifest",
        },
        "canonical_inputs",
    )
    for key, expected_path in {
        "contract": FULL_45_CONFIG_PATH,
        "pilot_contract": PILOT_CONTRACT_PATH,
        "scientific_config": V2_CONFIG_PATH,
        "selection_manifest": SELECTION_MANIFEST_PATH,
        "ocr_backend_config": "code/configs/restoration_v2_ocr_backend.json",
        "snapshot_manifest": "code/configs/gui_owl_1_5_8b_snapshot.json",
    }.items():
        record = _mapping(canonical.get(key), f"canonical_inputs.{key}")
        _exact_keys(record, {"path", "sha256"}, f"canonical_inputs.{key}")
        if record.get("path") != expected_path:
            raise ValueError(f"canonical input {key} path drifted")
        _require_sha256(record.get("sha256"), name=f"canonical input {key} SHA256")
    if canonical["contract"]["sha256"] != FROZEN_RESTORATION_V2_1_FULL_45_SHA256:
        raise ValueError("run canonical contract SHA differs from frozen child config")
    for key, manifest_path in {
        "pilot_evidence": PILOT_ARTIFACT_PATH,
        "processor_preflight": PROCESSOR_ARTIFACT_PATH,
    }.items():
        record = _mapping(canonical.get(key), f"canonical_inputs.{key}")
        _exact_keys(
            record,
            {"external", "artifact_manifest_path", "artifact_manifest_sha256"},
            f"canonical_inputs.{key}",
        )
        external = _mapping(record.get("external"), f"{key} external")
        _exact_keys(external, {"path", "sha256", "size_bytes"}, f"{key} external")
        if (
            not isinstance(external.get("path"), str)
            or not Path(external["path"]).is_absolute()
            or type(external.get("size_bytes")) is not int
            or external["size_bytes"] <= 0
            or record.get("artifact_manifest_path") != manifest_path
        ):
            raise ValueError(f"canonical {key} external binding drifted")
        _require_sha256(external.get("sha256"), name=f"{key} external SHA256")
        _require_sha256(
            record.get("artifact_manifest_sha256"),
            name=f"{key} manifest SHA256",
        )

    artifact = _mapping(run_contract.get("artifact"), "artifact")
    _exact_keys(
        artifact,
        {
            "artifact_tree_sha256",
            "artifact_manifest_sha256",
            "screening_manifest_sha256",
            "derived_repo",
        },
        "artifact",
    )
    for key in (
        "artifact_tree_sha256",
        "artifact_manifest_sha256",
        "screening_manifest_sha256",
    ):
        _require_sha256(artifact.get(key), name=f"artifact {key}")
    derived = {
        "repo": "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
        "immutable_revision": "89f136abaff797e14fe758a198996e51032a10a6",
        "artifact_tree_sha256": (
            "475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e"
        ),
    }
    if artifact.get("derived_repo") != derived or artifact.get(
        "artifact_tree_sha256"
    ) != derived["artifact_tree_sha256"]:
        raise ValueError("derived screening artifact identity drifted")

    policy = _mapping(run_contract.get("policy"), "policy")
    _exact_keys(
        policy,
        {"repo", "revision", "model_dir", "runtime_metadata_requirement"},
        "policy",
    )
    model_dir = policy.get("model_dir")
    requirement = _mapping(
        policy.get("runtime_metadata_requirement"),
        "runtime metadata requirement",
    )
    _exact_keys(
        requirement,
        {
            "required_policy_protocol_id",
            "validated_after_claim_before_generation",
            "native_generation_metadata_persisted_twice_per_state",
            "teacher_metadata_persisted_three_times_per_eligible_state",
            "required_metadata_keys",
            "model_dir",
            "model_repo",
            "model_revision",
            "snapshot_manifest_sha256",
            "device",
            "target_effective_visual_tokens_per_image",
        },
        "runtime metadata requirement",
    )
    if (
        policy.get("repo") != "mPLUG/GUI-Owl-1.5-8B-Instruct"
        or policy.get("revision") != "06d5faecff74840bab2be2425e9c42667a5d04fc"
        or not isinstance(model_dir, str)
        or not Path(model_dir).is_absolute()
        or requirement.get("required_policy_protocol_id")
        != "causalcache_restoration_v2_1_official_tool_interface"
        or requirement.get("validated_after_claim_before_generation") is not True
        or requirement.get("native_generation_metadata_persisted_twice_per_state")
        is not True
        or requirement.get(
            "teacher_metadata_persisted_three_times_per_eligible_state"
        )
        is not True
        or requirement.get("required_metadata_keys") != sorted(RUNTIME_METADATA_KEYS)
        or requirement.get("model_dir") != model_dir
        or requirement.get("model_repo") != policy.get("repo")
        or requirement.get("model_revision") != policy.get("revision")
        or requirement.get("snapshot_manifest_sha256")
        != canonical["snapshot_manifest"]["sha256"]
        or requirement.get("device") != "cuda:0"
        or requirement.get("target_effective_visual_tokens_per_image") != 2560
    ):
        raise ValueError("full-45 policy/runtime requirement drifted")
    _validate_runtime_identity(run_contract.get("runtime_identity"))
    _validate_execution_argv(run_contract.get("execution_argv"), run_contract=run_contract)
    if run_contract.get("operational_argv_policy") != {
        "resume_flag_excluded_from_scientific_run_identity": True,
        "initial_invocation_is_recorded_without_resume": True,
        "resume_only_skips_terminal_prefix": True,
    } or run_contract.get("seed_policy") != {
        "decoding": "greedy_do_sample_false",
        "random_seed": None,
        "sampling_seed_not_applicable": True,
    }:
        raise ValueError("full-45 argv or seed policy drifted")
    return commit, _states_from_run_contract(run_contract)


def _inventory(files: Mapping[str, bytes]) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    for relative in sorted(files):
        _safe_relative_path(relative, label="full-45 evidence path")
        payload = files[relative]
        records.append(
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    return tuple(records)


def _expected_complete_file_names() -> set[str]:
    return {
        ARCHIVE_LEDGER_NAME,
        RUN_MANIFEST_FILENAME,
        RUNTIME_IDENTITY_FILENAME,
        AGGREGATE_FILENAME,
        *{
            f"{STATE_DIRECTORY}/{index:03d}.json"
            for index in range(EXPECTED_STATE_COUNT)
        },
        *{
            f"{ATTEMPT_DIRECTORY}/{index:03d}.json"
            for index in range(EXPECTED_STATE_COUNT)
        },
    }


def _validate_teacher_suppression_metadata(
    metadata: Any,
    *,
    runtime_metadata: Mapping[str, Any] | None,
) -> None:
    teacher = _mapping(metadata, "teacher metadata")
    required = {
        "batch_size",
        "device",
        "dtype",
        "vocabulary_size",
        "distance_span",
        "teacher_context",
        "teacher_carrier",
        "teacher_target_json_separators",
        "teacher_target_ends_with_model_generation_eos",
        "teacher_target_disjoint_from_suppressed_standard_eos",
        "teacher_standard_eos_suppressed_token_ids",
        "teacher_standard_eos_suppression_value",
        "teacher_standard_eos_suppression_semantics",
        "teacher_standard_eos_mask_application",
        "teacher_raw_logits_mutated",
        "finite_logits_validation",
        "logits_to_keep",
        "full_logit_tensor_host_transfers",
        "samples",
    }
    if not required.issubset(teacher):
        raise ValueError("teacher suppression metadata is incomplete")
    suppressed_ids = list(GUI_OWL_V2_1_EXPECTED_STANDARD_EOS_TOKEN_IDS)
    suppression_value = teacher.get("teacher_standard_eos_suppression_value")
    samples = teacher.get("samples")
    if (
        teacher.get("batch_size") != 1
        or teacher.get("device")
        != (runtime_metadata.get("device") if runtime_metadata is not None else "cuda:0")
        or teacher.get("dtype") != "torch.bfloat16"
        or type(teacher.get("vocabulary_size")) is not int
        or teacher["vocabulary_size"] <= 0
        or teacher.get("distance_span")
        != "official_tool_call_open_through_close_inclusive"
        or teacher.get("teacher_context")
        != "official_tools_prompt_plus_assistant_prefix_direct"
        or teacher.get("teacher_carrier") is not None
        or teacher.get("teacher_target_json_separators") != [", ", ": "]
        or teacher.get("teacher_target_ends_with_model_generation_eos") is not True
        or teacher.get("teacher_target_disjoint_from_suppressed_standard_eos")
        is not True
        or teacher.get("teacher_standard_eos_suppressed_token_ids")
        != suppressed_ids
        or suppression_value != BFLOAT16_MIN_FINITE
        or teacher.get("teacher_standard_eos_suppression_semantics")
        != "torch_finfo_bfloat16_min_finite_generation_alignment"
        or teacher.get("teacher_standard_eos_mask_application")
        != (
            "same_mask_on_every_reference_and_candidate_action_path_position_"
            "before_float32_log_softmax"
        )
        or teacher.get("teacher_raw_logits_mutated") is not False
        or teacher.get("finite_logits_validation")
        != "deferred_to_gpu_kl_invalid_to_nan_final_distance"
        or type(teacher.get("logits_to_keep")) is not int
        or teacher["logits_to_keep"] <= 0
        or teacher.get("full_logit_tensor_host_transfers") != 0
        or not isinstance(samples, list)
        or len(samples) != 1
    ):
        raise ValueError("teacher EOS suppression metadata drifted")
    if runtime_metadata is not None and runtime_metadata.get(
        "suppressed_standard_eos_token_ids"
    ) != suppressed_ids:
        raise ValueError("teacher suppression IDs differ from runtime identity")


def _validate_distance_audit(value: Any, *, teacher: Mapping[str, Any]) -> None:
    audit = _mapping(value, "distance audit")
    expected_keys = {
        "operation",
        "candidate_representation",
        "batch_size",
        "distance_tokens",
        "vocabulary_size",
        "device",
        "reference_input_dtype",
        "candidate_input_dtype",
        "compute_dtype",
        "output_dtype",
        "reference_batch_stride",
        "reference_zero_copy_batch_expansion",
        "reference_compute_batch_size",
        "reduction",
        "log_normalization_atol",
        "negative_kl_atol",
        "numeric_validation",
        "invalid_numeric_output",
        "device_validation_category_count",
        "validation_scalar_host_reads",
        "full_tensor_host_transfers",
    }
    _exact_keys(audit, expected_keys, "distance audit")
    if (
        audit.get("operation")
        != "teacher_forced_full_vocabulary_mean_kl_on_distance_token_span"
        or audit.get("candidate_representation") != "logits"
        or audit.get("batch_size") != 1
        or audit.get("distance_tokens") != teacher.get("logits_to_keep")
        or audit.get("vocabulary_size") != teacher.get("vocabulary_size")
        or audit.get("device") != teacher.get("device")
        or audit.get("reference_input_dtype") != "torch.float32"
        or audit.get("candidate_input_dtype") != "torch.bfloat16"
        or audit.get("compute_dtype") != "torch.float32"
        or audit.get("output_dtype") != "torch.float32"
        or audit.get("reference_batch_stride") <= 0
        or audit.get("reference_zero_copy_batch_expansion") is not False
        or audit.get("reference_compute_batch_size") != 1
        or audit.get("reduction")
        != "full_vocabulary_sum_then_distance_token_mean_per_example"
        or audit.get("log_normalization_atol") != 5e-4
        or audit.get("negative_kl_atol") != 1e-5
        or audit.get("numeric_validation") != "gpu_resident_per_example_predicates"
        or audit.get("invalid_numeric_output") != "nan_final_distance"
        or audit.get("device_validation_category_count") != 4
        or audit.get("validation_scalar_host_reads") != 0
        or audit.get("full_tensor_host_transfers") != 0
    ):
        raise ValueError("distance audit differs from the frozen GPU KL path")


def _validate_attempt_record(
    value: Mapping[str, Any],
    *,
    projection: Mapping[str, Any],
    run_contract_sha256: str,
) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "state",
            "attempt_ordinal",
            "retry_count",
            "top_up_count",
            "created_at_utc",
        },
        "full-45 attempt marker",
    )
    created = value.get("created_at_utc")
    if (
        value.get("schema_version") != RUN_SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != ATTEMPT_STATUS
        or value.get("run_contract_sha256") != run_contract_sha256
        or value.get("state") != dict(projection)
        or value.get("attempt_ordinal") != 1
        or value.get("retry_count") != 0
        or value.get("top_up_count") != 0
        or not isinstance(created, str)
    ):
        raise ValueError("full-45 attempt marker identity drifted")
    _duration_seconds(created, created)


def _validate_attempt_and_state_records(
    files: Mapping[str, bytes],
    *,
    states: Sequence[ScreeningState],
    run_contract_sha256: str,
    completed_state_count: int,
    attempted_state_count: int,
    runtime_metadata: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(attempted_state_count):
        projection = _full45_projection(states[index], index)
        attempt_path = f"{ATTEMPT_DIRECTORY}/{index:03d}.json"
        attempt = strict_json_object_bytes(files[attempt_path], label=attempt_path)
        _validate_attempt_record(
            attempt,
            projection=projection,
            run_contract_sha256=run_contract_sha256,
        )
        if index >= completed_state_count:
            continue
        state_path = f"{STATE_DIRECTORY}/{index:03d}.json"
        record = strict_json_object_bytes(files[state_path], label=state_path)
        _validate_terminal_state_record(
            record,
            projection=projection,
            run_contract_sha256=run_contract_sha256,
            runtime_metadata=runtime_metadata,
        )
        teacher_forwards = _mapping(
            record.get("teacher_forwards"),
            "state teacher_forwards",
        )
        for metadata in teacher_forwards.values():
            _validate_teacher_suppression_metadata(
                metadata,
                runtime_metadata=runtime_metadata,
            )
        audits = _mapping(record.get("distance_audits"), "state distance audits")
        if teacher_forwards:
            _validate_distance_audit(
                audits.get("repeat_reference_kl"),
                teacher=_mapping(teacher_forwards["reference_2"], "reference_2"),
            )
            _validate_distance_audit(
                audits.get("summary_reference_kl"),
                teacher=_mapping(teacher_forwards["summary_only"], "summary_only"),
            )
        records.append(record)
    return records


def _validate_observed_invalid_records(
    files: Mapping[str, bytes],
    *,
    states: Sequence[ScreeningState],
    run_contract_sha256: str,
    ledger_completed: int,
    ledger_attempted: int,
    runtime_metadata: Mapping[str, Any] | None,
    observed_inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    attempt_names = _sequence(
        observed_inventory.get("attempt_entries"),
        "observed attempt entries",
    )
    state_names = _sequence(
        observed_inventory.get("state_entries"),
        "observed state entries",
    )
    records: list[dict[str, Any]] = []
    for name in attempt_names:
        if not isinstance(name, str):
            raise ValueError("observed attempt filename is invalid")
        index = int(name.removesuffix(".json"))
        if index >= ledger_attempted:
            raise ValueError("attempt marker exceeds durable attempted high-water")
        projection = _full45_projection(states[index], index)
        path = f"{ATTEMPT_DIRECTORY}/{name}"
        _validate_attempt_record(
            strict_json_object_bytes(files[path], label=path),
            projection=projection,
            run_contract_sha256=run_contract_sha256,
        )
    for name in state_names:
        if not isinstance(name, str):
            raise ValueError("observed state filename is invalid")
        index = int(name.removesuffix(".json"))
        if index >= ledger_completed:
            raise ValueError("state record exceeds durable completed high-water")
        projection = _full45_projection(states[index], index)
        path = f"{STATE_DIRECTORY}/{name}"
        record = strict_json_object_bytes(files[path], label=path)
        _validate_terminal_state_record(
            record,
            projection=projection,
            run_contract_sha256=run_contract_sha256,
            runtime_metadata=runtime_metadata,
        )
        teacher_forwards = _mapping(
            record.get("teacher_forwards"),
            "INVALID state teacher_forwards",
        )
        for metadata in teacher_forwards.values():
            _validate_teacher_suppression_metadata(
                metadata,
                runtime_metadata=runtime_metadata,
            )
        audits = _mapping(
            record.get("distance_audits"),
            "INVALID state distance audits",
        )
        if teacher_forwards:
            _validate_distance_audit(
                audits.get("repeat_reference_kl"),
                teacher=_mapping(teacher_forwards["reference_2"], "reference_2"),
            )
            _validate_distance_audit(
                audits.get("summary_reference_kl"),
                teacher=_mapping(teacher_forwards["summary_only"], "summary_only"),
            )
        records.append(record)
    return records


def _deterministic_tar_bytes(files: Mapping[str, bytes]) -> bytes:
    destination = io.BytesIO()
    with tarfile.open(
        fileobj=destination,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for relative in sorted(files):
            payload = files[relative]
            info = tarfile.TarInfo(f"{ARCHIVE_MEMBER_PREFIX}/{relative}")
            info.size = len(payload)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return destination.getvalue()


def _write_deterministic_tar(path: Path, files: Mapping[str, bytes]) -> None:
    payload = _deterministic_tar_bytes(files)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def _read_archive_files(path: str | Path) -> dict[str, bytes]:
    archive_path = Path(path)
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("full-45 raw archive is missing or symlinked")
    files: dict[str, bytes] = {}
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("full-45 archive ordering or uniqueness drifted")
        prefix = f"{ARCHIVE_MEMBER_PREFIX}/"
        for member in members:
            if (
                not member.isfile()
                or not member.name.startswith(prefix)
                or member.mode != 0o644
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mtime != 0
                or member.pax_headers
            ):
                raise ValueError("full-45 archive contains a non-canonical member")
            relative = member.name[len(prefix) :]
            _safe_relative_path(relative, label="full-45 archive member")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("full-45 archive member cannot be read")
            payload = source.read()
            if len(payload) != member.size:
                raise ValueError("full-45 archive member size drifted")
            files[relative] = payload
    if archive_path.read_bytes() != _deterministic_tar_bytes(files):
        raise ValueError("full-45 archive bytes are not canonical deterministic USTAR")
    return files


def _read_extracted_files(path: str | Path) -> dict[str, bytes]:
    supplied = Path(path)
    prefixed = supplied / ARCHIVE_MEMBER_PREFIX
    root = prefixed if prefixed.is_dir() else supplied
    if root.is_symlink() or not root.is_dir() or root.name != ARCHIVE_MEMBER_PREFIX:
        raise ValueError("extracted full-45 evidence root is not canonical")
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ValueError("extracted full-45 evidence contains a symlink")
        if item.is_dir():
            directories.add(item.relative_to(root).as_posix())
            continue
        if not item.is_file():
            raise ValueError("extracted full-45 evidence contains a non-file")
        relative = item.relative_to(root).as_posix()
        _safe_relative_path(relative, label="extracted full-45 path")
        files[relative] = item.read_bytes()
    if directories != {STATE_DIRECTORY, ATTEMPT_DIRECTORY}:
        raise ValueError("extracted full-45 directory inventory drifted")
    return files


def _collect_raw_files(
    raw_output_dir: str | Path,
    global_attempt_ledger: str | Path,
    *,
    require_canonical_location: bool,
) -> dict[str, bytes]:
    root = Path(raw_output_dir)
    ledger = Path(global_attempt_ledger)
    if require_canonical_location and (
        root.resolve() != CANONICAL_RAW_OUTPUT_DIR
        or ledger.resolve() != CANONICAL_GLOBAL_LEDGER_PATH
    ):
        raise ValueError("full-45 raw directory or ledger is not canonical")
    if (
        root.is_symlink()
        or ledger.is_symlink()
        or not root.is_dir()
        or not ledger.is_file()
    ):
        raise ValueError("full-45 raw directory or ledger is missing or symlinked")
    required = {
        RUN_MANIFEST_FILENAME,
        AGGREGATE_FILENAME,
        STATE_DIRECTORY,
        ATTEMPT_DIRECTORY,
    }
    allowed = {*required, RUNTIME_IDENTITY_FILENAME}
    entries = tuple(root.iterdir())
    names = {item.name for item in entries}
    if (
        not required.issubset(names)
        or not names.issubset(allowed)
        or any(item.is_symlink() for item in entries)
        or not (root / STATE_DIRECTORY).is_dir()
        or not (root / ATTEMPT_DIRECTORY).is_dir()
    ):
        raise ValueError("full-45 raw root inventory drifted")
    files = {
        ARCHIVE_LEDGER_NAME: ledger.read_bytes(),
        RUN_MANIFEST_FILENAME: (root / RUN_MANIFEST_FILENAME).read_bytes(),
        AGGREGATE_FILENAME: (root / AGGREGATE_FILENAME).read_bytes(),
    }
    runtime = root / RUNTIME_IDENTITY_FILENAME
    if runtime.exists() and (runtime.is_symlink() or not runtime.is_file()):
        raise ValueError("full-45 runtime identity is not a regular file")
    if runtime.is_file():
        files[RUNTIME_IDENTITY_FILENAME] = runtime.read_bytes()
    for directory in (STATE_DIRECTORY, ATTEMPT_DIRECTORY):
        for item in sorted((root / directory).iterdir(), key=lambda value: value.name):
            if item.is_symlink() or not item.is_file():
                raise ValueError(f"full-45 {directory} contains a non-file")
            relative = f"{directory}/{item.name}"
            _safe_relative_path(relative, label=f"full-45 {directory} path")
            files[relative] = item.read_bytes()
    return files


def _validate_global_ledger_record(
    ledger: Mapping[str, Any],
    *,
    attempt_identity: Mapping[str, Any],
    run_contract_sha256: str,
) -> tuple[int, int, str]:
    _exact_keys(
        ledger,
        {
            "schema_version",
            "protocol_id",
            "status",
            "attempt_identity",
            "run_contract_sha256",
            "created_at_utc",
            "attempted_state_count",
            "completed_state_count",
            "journal",
        },
        "full-45 durable ledger",
    )
    created = ledger.get("created_at_utc")
    attempted = ledger.get("attempted_state_count")
    completed = ledger.get("completed_state_count")
    journal = ledger.get("journal")
    if (
        ledger.get("schema_version") != RUN_SCHEMA_VERSION
        or ledger.get("protocol_id") != PROTOCOL_ID
        or ledger.get("status") != GLOBAL_ATTEMPT_STATUS
        or ledger.get("attempt_identity") != dict(attempt_identity)
        or ledger.get("run_contract_sha256") != run_contract_sha256
        or not isinstance(created, str)
        or type(attempted) is not int
        or type(completed) is not int
        or not 0 <= completed <= attempted <= EXPECTED_STATE_COUNT
        or attempted - completed > 1
        or not isinstance(journal, list)
        or len(journal) != 1 + attempted + completed
    ):
        raise ValueError("full-45 durable ledger identity or high-water drifted")
    _duration_seconds(created, created)
    previous_attempted = 0
    previous_completed = 0
    previous_timestamp = created
    for sequence, value in enumerate(journal):
        entry = _mapping(value, f"durable journal entry {sequence}")
        _exact_keys(
            entry,
            {
                "sequence",
                "event",
                "state_index",
                "attempted_state_count",
                "completed_state_count",
                "created_at_utc",
            },
            "durable journal entry",
        )
        timestamp = entry.get("created_at_utc")
        if entry.get("sequence") != sequence or not isinstance(timestamp, str):
            raise ValueError("durable journal sequence or timestamp drifted")
        _duration_seconds(previous_timestamp, timestamp)
        event = entry.get("event")
        entry_attempted = entry.get("attempted_state_count")
        entry_completed = entry.get("completed_state_count")
        state_index = entry.get("state_index")
        if sequence == 0:
            if dict(entry) != {
                "sequence": 0,
                "event": "GLOBAL_ATTEMPT_CLAIMED",
                "state_index": None,
                "attempted_state_count": 0,
                "completed_state_count": 0,
                "created_at_utc": created,
            }:
                raise ValueError("initial durable journal entry drifted")
        elif event == "STATE_ATTEMPT_CLAIMED":
            if (
                previous_attempted != previous_completed
                or state_index != previous_attempted
                or entry_attempted != previous_attempted + 1
                or entry_completed != previous_completed
            ):
                raise ValueError("attempted durable high-water transition drifted")
        elif event == "STATE_TERMINAL_PERSISTED":
            if (
                previous_attempted != previous_completed + 1
                or state_index != previous_completed
                or entry_attempted != previous_attempted
                or entry_completed != previous_completed + 1
            ):
                raise ValueError("completed durable high-water transition drifted")
        else:
            raise ValueError("durable journal event drifted")
        previous_attempted = int(entry_attempted)
        previous_completed = int(entry_completed)
        previous_timestamp = timestamp
    if previous_attempted != attempted or previous_completed != completed:
        raise ValueError("durable journal tail differs from ledger high-water")
    return completed, attempted, created


def _evidence_inventory_snapshot(files: Mapping[str, bytes]) -> dict[str, Any]:
    state_names = sorted(
        path.removeprefix(f"{STATE_DIRECTORY}/")
        for path in files
        if path.startswith(f"{STATE_DIRECTORY}/")
    )
    attempt_names = sorted(
        path.removeprefix(f"{ATTEMPT_DIRECTORY}/")
        for path in files
        if path.startswith(f"{ATTEMPT_DIRECTORY}/")
    )
    allowed = tuple(f"{index:03d}.json" for index in range(EXPECTED_STATE_COUNT))
    if any(name not in allowed for name in (*state_names, *attempt_names)):
        raise ValueError("full-45 state/attempt filename drifted")
    state_set = set(state_names)
    attempt_set = set(attempt_names)
    contiguous = (
        state_set == set(allowed[: len(state_set)])
        and attempt_set == set(allowed[: len(attempt_set)])
        and state_set <= attempt_set
        and len(attempt_set) - len(state_set) <= 1
    )
    return {
        "state_entries": state_names,
        "attempt_entries": attempt_names,
        "state_file_count": len(state_names),
        "attempt_marker_file_count": len(attempt_names),
        "all_entries_are_regular_files": True,
        "contiguous_prefix": contiguous,
    }


def _validate_invalid_aggregate_record(
    aggregate: Mapping[str, Any],
    *,
    run_contract_sha256: str,
    started_at_utc: str,
    ledger_completed: int,
    ledger_attempted: int,
    observed_inventory: Mapping[str, Any],
) -> None:
    _exact_keys(
        aggregate,
        {
            "schema_version",
            "protocol_id",
            "run_contract_sha256",
            "status",
            "outcome",
            "gate_passed",
            "started_at_utc",
            "ended_at_utc",
            "duration_seconds",
            "invalid_failure",
            "completed_state_count",
            "attempted_state_count",
            "observed_inventory",
            "retry_count",
            "top_up_count",
            "prohibited_operation_counts",
        },
        "full-45 INVALID aggregate",
    )
    ended = aggregate.get("ended_at_utc")
    if not isinstance(ended, str):
        raise ValueError("full-45 INVALID aggregate lacks end timestamp")
    expected_duration = _duration_seconds(started_at_utc, ended)
    failure = _mapping(aggregate.get("invalid_failure"), "INVALID failure")
    _exact_keys(
        failure,
        {"stage", "category", "exception_type", "message"},
        "INVALID failure",
    )
    prohibited = {
        key: 0
        for key in FORBIDDEN_OPERATION_COUNTS
        if key not in {"retry_count", "top_up_count"}
    }
    if (
        aggregate.get("schema_version") != RUN_SCHEMA_VERSION
        or aggregate.get("protocol_id") != PROTOCOL_ID
        or aggregate.get("run_contract_sha256") != run_contract_sha256
        or aggregate.get("status")
        != "TERMINATED_INVALID_V2_1_FULL_45_SUBSTRATE"
        or aggregate.get("outcome") != INVALID_OUTCOME
        or aggregate.get("gate_passed") is not False
        or aggregate.get("started_at_utc") != started_at_utc
        or aggregate.get("duration_seconds") != expected_duration
        or aggregate.get("completed_state_count") != ledger_completed
        or aggregate.get("attempted_state_count") != ledger_attempted
        or aggregate.get("observed_inventory") != dict(observed_inventory)
        or aggregate.get("retry_count") != 0
        or aggregate.get("top_up_count") != 0
        or aggregate.get("prohibited_operation_counts") != prohibited
        or failure.get("category") not in {"OUT_OF_MEMORY", "CONTRACT_OR_RUNTIME"}
        or any(
            not isinstance(failure.get(key), str) or not failure[key]
            for key in ("stage", "exception_type", "message")
        )
    ):
        raise ValueError("full-45 INVALID aggregate identity drifted")


def _default_committed_blob_reader(
    repository_root: Path,
    source_git_commit: str,
    relative_path: str,
) -> bytes:
    return _git_bytes(
        repository_root,
        "show",
        f"{source_git_commit}:{relative_path}",
    )


def _default_ancestor_validator(
    repository_root: Path,
    ancestor: str,
    descendant: str,
) -> None:
    require_source_commit_ancestor(
        repository_root=repository_root,
        source_git_commit=ancestor,
        current_git_commit=descendant,
    )


def validate_source_x_run_contract(
    run_contract: Mapping[str, Any],
    *,
    repository_root: str | Path,
    source_git_commit: str,
    committed_blob_reader: Callable[[Path, str, str], bytes] = (
        _default_committed_blob_reader
    ),
    ancestor_validator: Callable[[Path, str, str], None] = (
        _default_ancestor_validator
    ),
) -> None:
    """Replay child config, parent promotion, and all source blobs at X."""
    root = Path(repository_root).resolve()
    validated_commit, _ = _validate_run_contract(
        run_contract,
        expected_source_git_commit=source_git_commit,
    )
    if validated_commit != source_git_commit:
        raise ValueError("source-X run contract Git identity drifted")
    inventory = _sequence(run_contract.get("source_inventory"), "source_inventory")
    for relative, item in zip(EXPECTED_RUN_SOURCE_PATHS, inventory, strict=True):
        record = _mapping(item, f"source inventory {relative}")
        payload = committed_blob_reader(root, source_git_commit, relative)
        if (
            record.get("path") != relative
            or record.get("git_commit") != source_git_commit
            or record.get("sha256") != sha256_bytes(payload)
        ):
            raise ValueError(f"source-X committed blob binding drifted: {relative}")

    committed: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for relative in (
        FULL_45_CONFIG_PATH,
        PILOT_CONTRACT_PATH,
        V2_CONFIG_PATH,
        SELECTION_MANIFEST_PATH,
        "code/configs/restoration_v2_ocr_backend.json",
        "code/configs/gui_owl_1_5_8b_snapshot.json",
        PILOT_ARTIFACT_PATH,
        PROCESSOR_ARTIFACT_PATH,
    ):
        payload = committed_blob_reader(root, source_git_commit, relative)
        committed[relative] = (
            payload,
            strict_json_object_bytes(payload, label=f"source-X {relative}"),
        )
    child_bytes, child = committed[FULL_45_CONFIG_PATH]
    from causalcache.restoration_v2_1_full_45_contract import (
        FROZEN_RESTORATION_V2_1_FULL_45_SHA256,
        _full_45_projection,
    )

    if (
        sha256_bytes(child_bytes) != FROZEN_RESTORATION_V2_1_FULL_45_SHA256
        or run_contract.get("contract_sha256")
        != FROZEN_RESTORATION_V2_1_FULL_45_SHA256
        or child.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("source-X full-45 child config bytes drifted")
    source_lock = _mapping(child.get("source_lock"), "source-X source_lock")
    if source_lock.get("formal_run_source_inventory_paths") != list(
        EXPECTED_RUN_SOURCE_PATHS
    ):
        raise ValueError("source-X formal source inventory declaration drifted")
    for index, item in enumerate(
        _sequence(source_lock.get("files"), "source-X locked source files")
    ):
        record = _mapping(item, f"source-X locked source file {index}")
        relative = _safe_relative_path(
            record.get("path"),
            label="source-X locked source path",
        )
        payload = committed_blob_reader(root, source_git_commit, relative)
        if record.get("sha256") != sha256_bytes(payload):
            raise ValueError(f"source-X locked parser/runtime blob drifted: {relative}")

    canonical = _mapping(run_contract.get("canonical_inputs"), "canonical_inputs")
    input_paths = {
        "contract": FULL_45_CONFIG_PATH,
        "pilot_contract": PILOT_CONTRACT_PATH,
        "scientific_config": V2_CONFIG_PATH,
        "selection_manifest": SELECTION_MANIFEST_PATH,
        "ocr_backend_config": "code/configs/restoration_v2_ocr_backend.json",
        "snapshot_manifest": "code/configs/gui_owl_1_5_8b_snapshot.json",
    }
    for label, relative in input_paths.items():
        expected = {
            "path": relative,
            "sha256": sha256_bytes(committed[relative][0]),
        }
        if canonical.get(label) != expected:
            raise ValueError(f"run canonical input differs from source-X {label}")

    parent = _mapping(child.get("parent_authorization"), "parent authorization")
    child_pilot = _mapping(
        parent.get("fixed_15_pass_manifest"),
        "child fixed-15 binding",
    )
    pilot_bytes, pilot_manifest = committed[PILOT_ARTIFACT_PATH]
    pilot_result = _mapping(pilot_manifest.get("result"), "pilot result")
    pilot_hf = _mapping(pilot_manifest.get("hf_artifact"), "pilot HF artifact")
    pilot_raw = _mapping(pilot_manifest.get("raw_archive"), "pilot raw archive")
    pilot_source = _mapping(
        pilot_manifest.get("source_execution"),
        "pilot source execution",
    )
    if (
        child_pilot.get("path") != PILOT_ARTIFACT_PATH
        or child_pilot.get("sha256") != sha256_bytes(pilot_bytes)
        or child_pilot.get("required_status") != pilot_result.get("status")
        or child_pilot.get("required_outcome") != pilot_result.get("outcome")
        or child_pilot.get("required_fixed_state_denominator")
        != pilot_result.get("fixed_state_denominator")
        or child_pilot.get("required_completed_state_count")
        != pilot_result.get("completed_state_count")
        or pilot_result.get("attempted_state_count") != 15
        or child_pilot.get("source_git_commit")
        != pilot_source.get("source_git_commit")
        or child_pilot.get("hf_repo") != pilot_hf.get("repo")
        or child_pilot.get("hf_immutable_revision")
        != pilot_hf.get("immutable_revision")
        or child_pilot.get("hf_tag") != pilot_hf.get("tag")
        or child_pilot.get("hf_path") != pilot_hf.get("path")
        or child_pilot.get("raw_archive_sha256") != pilot_raw.get("sha256")
        or child_pilot.get("raw_archive_size_bytes")
        != pilot_raw.get("size_bytes")
        or pilot_hf.get("repo_type") != "dataset"
        or pilot_hf.get("visibility") != "private"
    ):
        raise ValueError("source-X fixed-15 promotion manifest binding drifted")
    run_pilot = _mapping(
        _mapping(
            run_contract.get("parent_authorization"),
            "run parent authorization",
        ).get("pilot_artifact_validation"),
        "run pilot validation",
    )
    pilot_external = _mapping(
        _mapping(canonical.get("pilot_evidence"), "pilot evidence input").get(
            "external"
        ),
        "pilot external input",
    )
    if (
        run_pilot.get("source_git_commit") != pilot_source.get("source_git_commit")
        or run_pilot.get("outcome") != pilot_result.get("outcome")
        or run_pilot.get("file_count") != pilot_raw.get("file_count")
        or run_pilot.get("tree_inventory_sha256")
        != pilot_raw.get("tree_inventory_sha256")
        or pilot_external.get("sha256") != pilot_raw.get("sha256")
        or pilot_external.get("size_bytes") != pilot_raw.get("size_bytes")
        or _mapping(canonical.get("pilot_evidence"), "pilot evidence input").get(
            "artifact_manifest_sha256"
        )
        != sha256_bytes(pilot_bytes)
    ):
        raise ValueError("run fixed-15 fresh archive authorization drifted")
    ancestor_validator(
        root,
        str(pilot_source.get("source_git_commit")),
        source_git_commit,
    )

    child_processor = _mapping(
        parent.get("processor_pass_manifest"),
        "child processor binding",
    )
    processor_bytes, processor_manifest = committed[PROCESSOR_ARTIFACT_PATH]
    compact = _mapping(
        processor_manifest.get("compact_reduction"),
        "processor compact reduction",
    )
    processor_hf = _mapping(
        processor_manifest.get("hf_artifact"),
        "processor HF artifact",
    )
    processor_raw = _mapping(
        processor_manifest.get("raw_evidence"),
        "processor raw evidence",
    )
    if (
        child_processor.get("path") != PROCESSOR_ARTIFACT_PATH
        or child_processor.get("sha256") != sha256_bytes(processor_bytes)
        or child_processor.get("required_status") != compact.get("status")
        or child_processor.get("required_state_count") != compact.get("state_count")
        or child_processor.get("required_prompt_count")
        != compact.get("prompt_count")
        or child_processor.get("source_git_commit")
        != processor_raw.get("source_git_commit")
        or child_processor.get("hf_repo") != processor_hf.get("repo")
        or child_processor.get("hf_immutable_revision")
        != processor_hf.get("immutable_revision")
        or child_processor.get("hf_tag") != processor_hf.get("tag")
        or child_processor.get("hf_path") != processor_hf.get("path")
        or child_processor.get("raw_evidence_sha256")
        != processor_raw.get("sha256")
        or child_processor.get("raw_evidence_size_bytes")
        != processor_raw.get("size_bytes")
        or processor_hf.get("repo_type") != "dataset"
        or processor_hf.get("visibility") != "private"
    ):
        raise ValueError("source-X processor promotion manifest binding drifted")
    run_processor = _mapping(
        _mapping(
            run_contract.get("parent_authorization"),
            "run parent authorization",
        ).get("processor_audit"),
        "run processor audit",
    )
    for key in (
        "status",
        "prompt_count",
        "prompt_records_sha256",
        "shape_records_sha256",
        "teacher_golden_records_sha256",
        "processor_classes_sha256",
    ):
        if run_processor.get(key) != compact.get(key):
            raise ValueError("run processor audit differs from compact reduction")
    processor_external = _mapping(
        _mapping(
            canonical.get("processor_preflight"),
            "processor preflight input",
        ).get("external"),
        "processor external input",
    )
    if (
        run_processor.get("evidence_git_commit")
        != processor_raw.get("source_git_commit")
        or processor_external.get("sha256") != processor_raw.get("sha256")
        or processor_external.get("size_bytes") != processor_raw.get("size_bytes")
        or _mapping(
            canonical.get("processor_preflight"),
            "processor preflight input",
        ).get("artifact_manifest_sha256")
        != sha256_bytes(processor_bytes)
    ):
        raise ValueError("run processor raw evidence authorization drifted")
    ancestor_validator(
        root,
        str(processor_raw.get("source_git_commit")),
        source_git_commit,
    )

    selection = committed[SELECTION_MANIFEST_PATH][1]
    source_projection = _full_45_projection(selection)
    child_data = _mapping(child.get("data"), "child data")
    if (
        run_contract.get("states") != source_projection
        or child_data.get("fixed_state_denominator") != EXPECTED_STATE_COUNT
        or child_data.get("state_projection_sha256")
        != sha256_bytes(canonical_json_bytes(source_projection))
        or child_data.get("state_ids_in_exact_order")
        != [record["state_id"] for record in source_projection]
    ):
        raise ValueError("run denominator differs from source-X selection projection")
    pilot_contract = committed[PILOT_CONTRACT_PATH][1]
    primary = _mapping(pilot_contract.get("primary_policy"), "pilot primary policy")
    policy = _mapping(run_contract.get("policy"), "run policy")
    snapshot = _mapping(primary.get("snapshot_manifest"), "pilot snapshot manifest")
    requirement = _mapping(
        policy.get("runtime_metadata_requirement"),
        "runtime metadata requirement",
    )
    if (
        policy.get("repo") != primary.get("repo")
        or policy.get("revision") != primary.get("revision")
        or canonical.get("snapshot_manifest") != snapshot
        or requirement.get("snapshot_manifest_sha256") != snapshot.get("sha256")
        or run_contract.get("computation_schedule")
        != child.get("computation_schedule")
        or run_contract.get("promotion") != child.get("promotion")
        or _mapping(run_contract.get("artifact"), "run artifact").get(
            "derived_repo"
        )
        != child_data.get("derived_artifact")
    ):
        raise ValueError("run policy, schedule, or artifact differs from source X")


def validate_full_45_evidence_files(
    files: Mapping[str, bytes],
    *,
    expected_source_git_commit: str | None = None,
) -> Full45Evidence:
    """Recompute one PASS, NO_GO, or INVALID result from its raw evidence."""
    copied = dict(files)
    if any(
        not isinstance(name, str) or not isinstance(payload, bytes)
        for name, payload in copied.items()
    ):
        raise TypeError("full-45 evidence files must map POSIX paths to bytes")
    required = {ARCHIVE_LEDGER_NAME, RUN_MANIFEST_FILENAME, AGGREGATE_FILENAME}
    if not required.issubset(copied):
        raise ValueError("full-45 evidence lacks manifest, ledger, or aggregate")
    manifest = strict_json_object_bytes(
        copied[RUN_MANIFEST_FILENAME],
        label="full-45 run manifest",
    )
    _exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "status",
            "run_contract_sha256",
            "run_contract",
            "created_at_utc",
        },
        "full-45 run manifest",
    )
    run_contract = _mapping(manifest.get("run_contract"), "full-45 run contract")
    run_contract_sha256 = sha256_bytes(canonical_json_bytes(run_contract))
    created_at_utc = manifest.get("created_at_utc")
    if (
        manifest.get("schema_version") != RUN_SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != RUN_STATUS
        or manifest.get("run_contract_sha256") != run_contract_sha256
        or not isinstance(created_at_utc, str)
    ):
        raise ValueError("full-45 run manifest identity drifted")
    _duration_seconds(created_at_utc, created_at_utc)
    source_commit, states = _validate_run_contract(
        run_contract,
        expected_source_git_commit=expected_source_git_commit,
    )
    layout = _prepare_layout(
        states=states,
        run_contract=run_contract,
        output_dir=CANONICAL_RAW_OUTPUT_DIR,
    )
    if layout.contract_sha256 != run_contract_sha256:
        raise ValueError("runner and artifact contract hashes differ")

    ledger = strict_json_object_bytes(
        copied[ARCHIVE_LEDGER_NAME],
        label="full-45 global attempt ledger",
    )
    ledger_completed, ledger_attempted, ledger_created = (
        _validate_global_ledger_record(
            ledger,
            attempt_identity=layout.attempt_identity,
            run_contract_sha256=run_contract_sha256,
        )
    )
    if ledger_created != created_at_utc:
        raise ValueError("full-45 ledger and run manifest start times differ")

    aggregate = strict_json_object_bytes(
        copied[AGGREGATE_FILENAME],
        label="full-45 aggregate",
    )
    runtime_metadata: Mapping[str, Any] | None = None
    if RUNTIME_IDENTITY_FILENAME in copied:
        runtime = strict_json_object_bytes(
            copied[RUNTIME_IDENTITY_FILENAME],
            label="full-45 runtime identity",
        )
        runtime_metadata = _validate_runtime_identity_record(runtime, layout=layout)
    outcome = aggregate.get("outcome")
    if outcome in {PASS_OUTCOME, NO_GO_OUTCOME}:
        if set(copied) != _expected_complete_file_names():
            raise ValueError("completed evidence inventory is not exact fixed-45")
        if (ledger_completed, ledger_attempted) != (
            EXPECTED_STATE_COUNT,
            EXPECTED_STATE_COUNT,
        ):
            raise ValueError("completed evidence ledger high-water is not exact 45/45")
        if runtime_metadata is None:
            raise ValueError("completed full-45 evidence lacks runtime identity")
        records = _validate_attempt_and_state_records(
            copied,
            states=states,
            run_contract_sha256=run_contract_sha256,
            completed_state_count=EXPECTED_STATE_COUNT,
            attempted_state_count=EXPECTED_STATE_COUNT,
            runtime_metadata=runtime_metadata,
        )
        ended_at_utc = aggregate.get("ended_at_utc")
        if not isinstance(ended_at_utc, str):
            raise ValueError("completed full-45 aggregate lacks end timestamp")
        recomputed = aggregate_full45_gate(
            records,
            expected_states=states,
            gate=_fixed_gate(),
            run_contract_sha256=run_contract_sha256,
            started_at_utc=created_at_utc,
            ended_at_utc=ended_at_utc,
            runtime_metadata=runtime_metadata,
        )
        if aggregate != recomputed:
            raise ValueError("stored full-45 aggregate differs from raw reduction")
        completed_state_count = EXPECTED_STATE_COUNT
        attempted_state_count = EXPECTED_STATE_COUNT
    elif outcome == INVALID_OUTCOME:
        observed_inventory = _evidence_inventory_snapshot(copied)
        _validate_invalid_aggregate_record(
            aggregate,
            run_contract_sha256=run_contract_sha256,
            started_at_utc=created_at_utc,
            ledger_completed=ledger_completed,
            ledger_attempted=ledger_attempted,
            observed_inventory=observed_inventory,
        )
        completed_state_count = ledger_completed
        attempted_state_count = ledger_attempted
        failure = _mapping(aggregate.get("invalid_failure"), "INVALID failure")
        stage = failure.get("stage")
        missing_runtime_allowed = (
            stage
            in {
                "durable_claim_bootstrap_failure",
                "policy_runtime_initialization_after_durable_claim",
                "resume_detected_missing_runtime_identity",
                "resume_detected_deleted_durable_claim_component",
                "resume_detected_ledger_inventory_high_water_mismatch",
                "runner_detected_ledger_inventory_high_water_mismatch",
            }
            or isinstance(stage, str)
            and stage.startswith("resume_detected_incomplete_state_")
        )
        if runtime_metadata is None and not missing_runtime_allowed:
            raise ValueError("INVALID stage requires persisted runtime identity")
        expected_files = {
            ARCHIVE_LEDGER_NAME,
            RUN_MANIFEST_FILENAME,
            AGGREGATE_FILENAME,
            *{
                f"{STATE_DIRECTORY}/{name}"
                for name in observed_inventory["state_entries"]
            },
            *{
                f"{ATTEMPT_DIRECTORY}/{name}"
                for name in observed_inventory["attempt_entries"]
            },
        }
        if runtime_metadata is not None:
            expected_files.add(RUNTIME_IDENTITY_FILENAME)
        if set(copied) != expected_files:
            raise ValueError("INVALID full-45 evidence inventory drifted")
        records = _validate_observed_invalid_records(
            copied,
            states=states,
            run_contract_sha256=run_contract_sha256,
            ledger_completed=ledger_completed,
            ledger_attempted=ledger_attempted,
            runtime_metadata=runtime_metadata,
            observed_inventory=observed_inventory,
        )
    else:
        raise ValueError("full-45 aggregate outcome is not PASS, NO_GO, or INVALID")

    generation_count = sum(
        int(_mapping(record.get("operation_counts"), "operation counts")[
            "generation_call_count"
        ])
        for record in records
    )
    teacher_count = sum(
        int(_mapping(record.get("operation_counts"), "operation counts")[
            "teacher_forward_count"
        ])
        for record in records
    )
    kl_count = sum(
        int(_mapping(record.get("operation_counts"), "operation counts")[
            "kl_measurement_count"
        ])
        for record in records
    )
    if (
        generation_count > EXPECTED_GENERATION_COUNT
        or teacher_count > EXPECTED_TEACHER_FORWARD_COUNT
        or kl_count > EXPECTED_KL_MEASUREMENT_COUNT
    ):
        raise ValueError("full-45 raw operation counts exceed the frozen schedule")
    if outcome == PASS_OUTCOME and (
        generation_count,
        teacher_count,
        kl_count,
    ) != (
        EXPECTED_GENERATION_COUNT,
        EXPECTED_TEACHER_FORWARD_COUNT,
        EXPECTED_KL_MEASUREMENT_COUNT,
    ):
        raise ValueError("PASS full-45 evidence did not execute the exact schedule")
    inventory = _inventory(copied)
    return Full45Evidence(
        files=copied,
        run_contract=dict(run_contract),
        source_git_commit=source_commit,
        run_contract_sha256=run_contract_sha256,
        outcome=str(outcome),
        status=str(aggregate.get("status")),
        aggregate_sha256=sha256_bytes(copied[AGGREGATE_FILENAME]),
        run_manifest_sha256=sha256_bytes(copied[RUN_MANIFEST_FILENAME]),
        global_attempt_ledger_sha256=sha256_bytes(copied[ARCHIVE_LEDGER_NAME]),
        completed_state_count=completed_state_count,
        attempted_state_count=attempted_state_count,
        generation_call_count=generation_count,
        teacher_forward_count=teacher_count,
        kl_measurement_count=kl_count,
        inventory=inventory,
        tree_inventory_sha256=sha256_bytes(canonical_json_bytes(inventory)),
    )


def collect_raw_full_45_evidence(
    raw_output_dir: str | Path,
    global_attempt_ledger: str | Path,
    *,
    expected_source_git_commit: str | None = None,
    require_canonical_location: bool = True,
) -> Full45Evidence:
    """Collect and reduce one exact logical full-45 evidence tree."""
    files = _collect_raw_files(
        raw_output_dir,
        global_attempt_ledger,
        require_canonical_location=require_canonical_location,
    )
    return validate_full_45_evidence_files(
        files,
        expected_source_git_commit=expected_source_git_commit,
    )


def read_full_45_evidence_archive(
    archive_path: str | Path,
    *,
    expected_source_git_commit: str | None = None,
) -> Full45Evidence:
    """Read and independently reduce a canonical deterministic USTAR archive."""
    return validate_full_45_evidence_files(
        _read_archive_files(archive_path),
        expected_source_git_commit=expected_source_git_commit,
    )


def read_extracted_full_45_evidence(
    extracted_path: str | Path,
    *,
    expected_source_git_commit: str | None = None,
) -> Full45Evidence:
    """Read an extracted canonical evidence tree without trusting tar metadata."""
    return validate_full_45_evidence_files(
        _read_extracted_files(extracted_path),
        expected_source_git_commit=expected_source_git_commit,
    )


def package_raw_full_45_evidence(
    *,
    repository_root: str | Path,
    raw_output_dir: str | Path,
    global_attempt_ledger: str | Path,
    output_archive: str | Path,
    source_git_commit: str,
    require_canonical_location: bool = True,
    git_identity_validator: Callable[[str | Path], Mapping[str, Any]] = (
        validate_clean_pushed_main
    ),
    source_binding_validator: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Validate raw source-X evidence and write exactly one deterministic tar."""
    if GIT_SHA_PATTERN.fullmatch(source_git_commit) is None:
        raise ValueError("source Git commit must be a full lowercase commit SHA")
    archive = Path(output_archive)
    if (
        require_canonical_location
        and archive.resolve() != CANONICAL_RAW_ARCHIVE_PATH
    ):
        raise ValueError("full-45 archive output path is not canonical")
    git = dict(git_identity_validator(Path(repository_root).resolve()))
    if git.get("commit") != source_git_commit:
        raise ValueError("packaging Git identity differs from full-45 source X")
    evidence = collect_raw_full_45_evidence(
        raw_output_dir,
        global_attempt_ledger,
        expected_source_git_commit=source_git_commit,
        require_canonical_location=require_canonical_location,
    )
    validator = source_binding_validator or validate_source_x_run_contract
    validator(
        evidence.run_contract,
        repository_root=repository_root,
        source_git_commit=source_git_commit,
    )
    _write_deterministic_tar(archive, evidence.files)
    reread = read_full_45_evidence_archive(
        archive,
        expected_source_git_commit=source_git_commit,
    )
    if (
        reread.inventory != evidence.inventory
        or reread.outcome != evidence.outcome
        or reread.generation_call_count != evidence.generation_call_count
        or reread.teacher_forward_count != evidence.teacher_forward_count
        or reread.kl_measurement_count != evidence.kl_measurement_count
    ):
        raise ValueError("written full-45 archive differs from validated raw tree")
    return {
        "status": "PACKAGED_RESTORATION_V2_1_FULL_45_RAW_EVIDENCE",
        "source_git_commit": source_git_commit,
        "outcome": evidence.outcome,
        "archive_path": str(archive.resolve()),
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
        "file_count": len(evidence.inventory),
    }


def _validate_hf_identity(
    *,
    repo: str,
    immutable_revision: str,
    path: str,
) -> dict[str, str]:
    if repo != CANONICAL_HF_REPO:
        raise ValueError("full-45 HF repo differs from the canonical private repo")
    if GIT_SHA_PATTERN.fullmatch(immutable_revision) is None:
        raise ValueError("full-45 HF revision must be an immutable 40-hex commit")
    safe_path = _safe_relative_path(path, label="full-45 HF artifact path")
    if safe_path != CANONICAL_HF_PATH or not safe_path.endswith(".tar"):
        raise ValueError("full-45 HF path differs from the canonical tar path")
    return {
        "repo": CANONICAL_HF_REPO,
        "repo_type": "dataset",
        "visibility": "private",
        "tag": CANONICAL_HF_TAG,
        "immutable_revision": immutable_revision,
        "path": safe_path,
    }


def build_full_45_artifact_manifest(
    *,
    repository_root: str | Path,
    fresh_immutable_archive: str | Path,
    source_git_commit: str,
    hf_repo: str,
    hf_immutable_revision: str,
    hf_path: str,
    source_binding_validator: Callable[..., None] | None = None,
) -> dict[str, Any]:
    archive = Path(fresh_immutable_archive)
    if archive.resolve() == CANONICAL_RAW_ARCHIVE_PATH:
        raise ValueError("manifest requires a fresh immutable-HF archive download")
    evidence = read_full_45_evidence_archive(
        archive,
        expected_source_git_commit=source_git_commit,
    )
    validator = source_binding_validator or validate_source_x_run_contract
    validator(
        evidence.run_contract,
        repository_root=repository_root,
        source_git_commit=source_git_commit,
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": ARTIFACT_TYPE,
        "source_execution": {
            "attempt_id": CANONICAL_ATTEMPT_ID,
            "canonical_raw_output_dir": str(CANONICAL_RAW_OUTPUT_DIR),
            "canonical_global_attempt_ledger": str(
                CANONICAL_GLOBAL_LEDGER_PATH
            ),
            "source_git_commit": evidence.source_git_commit,
            "run_contract_sha256": evidence.run_contract_sha256,
        },
        "hf_artifact": _validate_hf_identity(
            repo=hf_repo,
            immutable_revision=hf_immutable_revision,
            path=hf_path,
        ),
        "raw_archive": {
            "format": ARCHIVE_FORMAT,
            "member_prefix": ARCHIVE_MEMBER_PREFIX,
            "fresh_immutable_download_hash_verified": True,
            "sha256": sha256_file(archive),
            "size_bytes": archive.stat().st_size,
            "file_count": len(evidence.inventory),
            "tree_inventory_sha256": evidence.tree_inventory_sha256,
            "inventory": list(evidence.inventory),
        },
        "result": {
            "outcome": evidence.outcome,
            "status": evidence.status,
            "fixed_state_denominator": EXPECTED_STATE_COUNT,
            "completed_state_count": evidence.completed_state_count,
            "attempted_state_count": evidence.attempted_state_count,
            "generation_call_count": evidence.generation_call_count,
            "teacher_forward_count": evidence.teacher_forward_count,
            "kl_measurement_count": evidence.kl_measurement_count,
            "aggregate_sha256": evidence.aggregate_sha256,
            "run_manifest_sha256": evidence.run_manifest_sha256,
            "global_attempt_ledger_sha256": (
                evidence.global_attempt_ledger_sha256
            ),
        },
    }


def validate_full_45_artifact_manifest(
    manifest: Mapping[str, Any],
    *,
    evidence: Full45Evidence,
    archive_path: str | Path | None,
) -> dict[str, Any]:
    _exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "artifact_type",
            "source_execution",
            "hf_artifact",
            "raw_archive",
            "result",
        },
        "full-45 artifact manifest",
    )
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("artifact_type") != ARTIFACT_TYPE
    ):
        raise ValueError("full-45 artifact manifest identity drifted")
    source = _mapping(manifest.get("source_execution"), "source_execution")
    if dict(source) != {
        "attempt_id": CANONICAL_ATTEMPT_ID,
        "canonical_raw_output_dir": str(CANONICAL_RAW_OUTPUT_DIR),
        "canonical_global_attempt_ledger": str(CANONICAL_GLOBAL_LEDGER_PATH),
        "source_git_commit": evidence.source_git_commit,
        "run_contract_sha256": evidence.run_contract_sha256,
    }:
        raise ValueError("full-45 source execution binding drifted")
    hf = _mapping(manifest.get("hf_artifact"), "hf_artifact")
    expected_hf = _validate_hf_identity(
        repo=str(hf.get("repo")),
        immutable_revision=str(hf.get("immutable_revision")),
        path=str(hf.get("path")),
    )
    if dict(hf) != expected_hf:
        raise ValueError("full-45 HF artifact metadata drifted")
    raw = _mapping(manifest.get("raw_archive"), "raw_archive")
    expected_raw = {
        "format": ARCHIVE_FORMAT,
        "member_prefix": ARCHIVE_MEMBER_PREFIX,
        "fresh_immutable_download_hash_verified": True,
        "sha256": raw.get("sha256"),
        "size_bytes": raw.get("size_bytes"),
        "file_count": len(evidence.inventory),
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
        "inventory": list(evidence.inventory),
    }
    if (
        not isinstance(raw.get("sha256"), str)
        or SHA256_PATTERN.fullmatch(raw["sha256"]) is None
        or type(raw.get("size_bytes")) is not int
        or raw["size_bytes"] <= 0
        or dict(raw) != expected_raw
    ):
        raise ValueError("full-45 raw archive inventory binding drifted")
    archive_hash_verified = False
    if archive_path is not None:
        archive = Path(archive_path)
        if (
            sha256_file(archive) != raw["sha256"]
            or archive.stat().st_size != raw["size_bytes"]
        ):
            raise ValueError("full-45 raw archive hash or size drifted")
        archive_hash_verified = True
    result = _mapping(manifest.get("result"), "result")
    expected_result = {
        "outcome": evidence.outcome,
        "status": evidence.status,
        "fixed_state_denominator": EXPECTED_STATE_COUNT,
        "completed_state_count": evidence.completed_state_count,
        "attempted_state_count": evidence.attempted_state_count,
        "generation_call_count": evidence.generation_call_count,
        "teacher_forward_count": evidence.teacher_forward_count,
        "kl_measurement_count": evidence.kl_measurement_count,
        "aggregate_sha256": evidence.aggregate_sha256,
        "run_manifest_sha256": evidence.run_manifest_sha256,
        "global_attempt_ledger_sha256": evidence.global_attempt_ledger_sha256,
    }
    if dict(result) != expected_result:
        raise ValueError("full-45 compact result differs from raw evidence")
    return {
        "status": "VALID_RESTORATION_V2_1_FULL_45_ARTIFACT",
        "source_git_commit": evidence.source_git_commit,
        "outcome": evidence.outcome,
        "file_count": len(evidence.inventory),
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
        "archive_hash_verified": archive_hash_verified,
    }


def write_full_45_artifact_manifest(
    *,
    repository_root: str | Path,
    fresh_immutable_archive: str | Path,
    source_git_commit: str,
    hf_repo: str,
    hf_immutable_revision: str,
    hf_path: str,
    output: str | Path,
    git_identity_validator: Callable[[str | Path], Mapping[str, Any]] = (
        validate_clean_pushed_main
    ),
    source_binding_validator: Callable[..., None] | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(output).resolve()
    canonical = (root / CANONICAL_GIT_ARTIFACT_PATH).resolve()
    if destination != canonical:
        raise ValueError("full-45 Git artifact output path is not canonical")
    if destination.exists():
        raise FileExistsError("full-45 Git artifact manifest already exists")
    git = dict(git_identity_validator(root))
    if git.get("commit") != source_git_commit:
        raise ValueError("manifest must be created from clean source commit X")
    manifest = build_full_45_artifact_manifest(
        repository_root=root,
        fresh_immutable_archive=fresh_immutable_archive,
        source_git_commit=source_git_commit,
        hf_repo=hf_repo,
        hf_immutable_revision=hf_immutable_revision,
        hf_path=hf_path,
        source_binding_validator=source_binding_validator,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(pretty_json_bytes(manifest))
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git command failed: {message or arguments}") from error


def validate_committed_full_45_artifact(
    *,
    repository_root: str | Path,
    current_git_commit: str,
    evidence_path: str | Path,
    artifact_path: str | Path = CANONICAL_GIT_ARTIFACT_PATH,
    source_binding_validator: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Validate source-X evidence from a clean descendant main commit Y."""
    root = Path(repository_root).resolve()
    canonical = (root / CANONICAL_GIT_ARTIFACT_PATH).resolve()
    supplied = Path(artifact_path)
    supplied = (
        supplied.resolve()
        if supplied.is_absolute()
        else (root / supplied).resolve()
    )
    if supplied != canonical or not canonical.is_file():
        raise ValueError("canonical full-45 Git artifact manifest is missing")
    live = canonical.read_bytes()
    committed = _git_bytes(
        root,
        "show",
        f"{current_git_commit}:{CANONICAL_GIT_ARTIFACT_PATH}",
    )
    if live != committed:
        raise ValueError("full-45 artifact differs from the current Git blob")
    manifest = strict_json_object_bytes(
        committed,
        label="full-45 artifact manifest",
    )
    source = _mapping(manifest.get("source_execution"), "source_execution")
    source_commit = source.get("source_git_commit")
    if not isinstance(source_commit, str):
        raise ValueError("full-45 source Git commit is missing")
    require_source_commit_ancestor(
        repository_root=root,
        source_git_commit=source_commit,
        current_git_commit=current_git_commit,
    )
    evidence_source = Path(evidence_path)
    if evidence_source.is_dir():
        evidence = read_extracted_full_45_evidence(
            evidence_source,
            expected_source_git_commit=source_commit,
        )
        archive: Path | None = None
        source_kind = "extracted_tree"
    else:
        evidence = read_full_45_evidence_archive(
            evidence_source,
            expected_source_git_commit=source_commit,
        )
        archive = evidence_source
        source_kind = "raw_archive"
    validator = source_binding_validator or validate_source_x_run_contract
    validator(
        evidence.run_contract,
        repository_root=root,
        source_git_commit=source_commit,
    )
    result = validate_full_45_artifact_manifest(
        manifest,
        evidence=evidence,
        archive_path=archive,
    )
    return {
        **result,
        "current_git_commit": current_git_commit,
        "evidence_source_kind": source_kind,
    }
