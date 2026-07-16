"""Contracts and reducers for the bounded spatial-reference numerical audit."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalcache.data.guiodyssey_restoration_v2 import (
    canonical_json_bytes,
    sha256_file,
)
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.restoration_v2_1_full_45_artifact import (
    read_full_45_evidence_archive,
)


PROTOCOL_ID = "spatial_reference_audit_v1"
PARENT_OUTCOME = "NO_GO_V2_1_FULL_45_SUBSTRATE"
PARENT_SOURCE_COMMIT = "7a5b6d5710fe4d054936b5aa474648149f725edb"
MISMATCH_CATEGORY = "CANONICAL_ACTION_MISMATCH"
EXPECTED_MISMATCH_INDICES = (1, 10, 16, 17, 20, 23, 24, 25, 26, 28, 30, 35, 39)
EXPECTED_PROFILE_IDS = (
    "bf16_auto",
    "bf16_eager_control",
    "fp32_eager_control",
)
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
EXPECTED_RUNTIME_CONSTRAINTS = {
    "container_image_digest": (
        "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
    ),
    "python_version": "3.12.3",
    "torch_version": "2.11.0+cu130",
    "torch_cuda_version": "13.0",
    "cudnn_version": 91900,
    "transformers_version": "5.6.0",
    "nvidia_driver_version": "570.172.08",
    "audited_scientific_environment_variables": list(
        AUDITED_SCIENTIFIC_ENVIRONMENT_VARIABLES
    ),
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ProfileSpec:
    profile_id: str
    dtype: str
    attention_implementation: str
    deterministic_algorithms: bool
    eager_numerical_controls: bool
    seed: int
    state_indices: tuple[int, ...]
    generation_repeat_count: int
    shared_prefix_forward_repeat_count: int
    full_parent_action_forward_count: int


@dataclass(frozen=True)
class ParentMismatch:
    index: int
    role: str
    state_id: str
    screen_dimensions: Mapping[str, int]
    actions: tuple[Mapping[str, Any], Mapping[str, Any]]
    output_texts: tuple[str, str]
    generated_token_ids_sha256: tuple[str, str]
    androidworld_bridges: tuple[Mapping[str, Any], Mapping[str, Any]]
    state_member_sha256: str


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def load_json_object(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def validate_exposure_ledger(
    value: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> None:
    if (
        value.get("schema_version") != "1.0.0"
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status")
        != "FROZEN_SOURCE_ONLY_BEFORE_AUDIT_POLICY_FORWARD"
    ):
        raise ValueError("spatial-reference exposure identity drifted")
    parent = _require_mapping(value.get("parent_exposure"), name="parent_exposure")
    if dict(parent) != {
        "path": "data/manifests/restoration_v2_exposure.json",
        "sha256": "bc1224826e0642c2374f6cfbebd676389d8c17ce6bf8a789f08903740cf97a95",
        "mutation_allowed": False,
    }:
        raise ValueError("spatial-reference parent exposure binding drifted")
    selection = _require_mapping(
        value.get("selection_manifest"),
        name="selection_manifest",
    )
    if dict(selection) != {
        "path": "data/manifests/restoration_v2_selection.json",
        "sha256": "292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b",
    }:
        raise ValueError("spatial-reference exposure selection binding drifted")
    known = value.get("known_policy_output_events")
    if not isinstance(known, list) or [item.get("event_id") for item in known] != [
        "restoration-v2-fixed-45-substrate-screening-v1",
        "restoration-v2-1-interface-pilot-v1",
        "restoration-v2-1-full-45-substrate-v1",
    ]:
        raise ValueError("known policy-output exposure events drifted")
    expected_known = (
        (45, 45, 0, "NO_GO_V2_SUBSTRATE"),
        (15, 15, 0, "PASS_V2_1_INTERFACE_PILOT"),
        (45, 90, 96, PARENT_OUTCOME),
    )
    for event, expected in zip(known, expected_known, strict=True):
        if (
            event.get("fixed_state_count"),
            event.get("generation_call_count"),
            event.get("teacher_forward_count"),
            event.get("outcome"),
        ) != expected or event.get("restoration_output_count") != 0:
            raise ValueError("known policy-output event counts or outcome drifted")
    v2_evidence = _require_mapping(known[0].get("evidence"), name="v2 evidence")
    if (
        v2_evidence.get("summary_sha256")
        != "57a00ac2363890bc2b45b0681dee6f785d05abeb90b4d98499d52d5f16e5c7e6"
        or v2_evidence.get("artifact_manifest_sha256")
        != "ce817c5047484d1f9cecb6b60423e318a6509f4480477c5a1504cff58bc456f2"
    ):
        raise ValueError("v2 exposure evidence drifted")
    pilot_evidence = _require_mapping(
        known[1].get("evidence"),
        name="v2.1 pilot evidence",
    )
    if (
        pilot_evidence.get("artifact_sha256")
        != "e7a92a0469e3b66113158d5c2ce713c363ecaff97f95498f3ef84f02a4a6bed8"
        or pilot_evidence.get("hf_revision")
        != "bdff8ca71f150afd80d6291b4ecec76cbf9e7432"
    ):
        raise ValueError("v2.1 pilot exposure evidence drifted")
    full_45_evidence = _require_mapping(known[2].get("evidence"), name="full-45 evidence")
    if (
        full_45_evidence.get("artifact_sha256")
        != "9aa7fd6bdc20173532649a4133faf565e99975af51e38d57cf9612f29af996be"
        or full_45_evidence.get("summary_sha256")
        != "f385baf261510dd82e510c8b40a05e5c2a5b91ef7e2329b4a4771b938644c6b8"
        or full_45_evidence.get("hf_revision")
        != "814506ef1450838d4bc6ed3d89fe53e0773d92fb"
    ):
        raise ValueError("full-45 exposure evidence drifted")
    planned = _require_mapping(
        value.get("planned_audit_event"),
        name="planned_audit_event",
    )
    denominator = _require_mapping(
        config.get("mismatch_denominator"),
        name="mismatch_denominator",
    )
    if (
        planned.get("fixed_parent_mismatch_state_indices")
        != denominator.get("state_indices")
        or planned.get("fixed_parent_mismatch_projection_sha256")
        != denominator.get("canonical_parent_projection_sha256")
        or planned.get("state_role_counts")
        != denominator.get("role_counts")
        or planned.get("planned_generation_call_count") != 60
        or planned.get("planned_teacher_forward_count") != 120
        or planned.get("planned_restoration_output_count") != 0
        or planned.get("planned_gate_training_example_count") != 0
        or planned.get("output_generated_before_freeze") is not False
    ):
        raise ValueError("planned spatial-reference exposure event drifted")
    artifact = _require_mapping(value.get("planned_artifact"), name="planned_artifact")
    raw = _require_mapping(artifact.get("raw"), name="planned_artifact.raw")
    if dict(raw) != {
        "hf_repo": "gavinlaw/causalcache-spatial-reference-audit-mobile",
        "hf_repo_type": "dataset",
        "visibility": "private",
        "tag": "spatial-reference-audit-v1",
        "path": "raw/spatial-reference-audit-v1.tar",
        "format": "ustar",
        "immutable_revision": None,
        "upload_status": "pending",
    }:
        raise ValueError("planned spatial-reference raw artifact drifted")
    compact = _require_mapping(
        artifact.get("git_compact_result"),
        name="planned_artifact.git_compact_result",
    )
    if dict(compact) != {
        "directory": "data/results/spatial_reference_audit_v1",
        "status": "pending_until_raw_hf_immutable_verification",
    }:
        raise ValueError("planned spatial-reference Git result drifted")
    roles = _require_mapping(value.get("role_state"), name="role_state")
    if _require_mapping(roles.get("v2_label_train"), name="label-train role") != {
        "source_id_count": 10,
        "known_policy_output": True,
        "known_restoration_output": False,
    } or _require_mapping(roles.get("v2_development"), name="development role") != {
        "source_id_count": 5,
        "known_policy_output": True,
        "known_restoration_output": False,
    }:
        raise ValueError("label-train/development exposure boundary drifted")
    confirm = _require_mapping(roles.get("v2_confirm_primary"), name="confirm role")
    if dict(confirm) != {
        "source_id_count": 20,
        "known_policy_output": False,
        "known_restoration_output": False,
        "policy_output_access_allowed_by_this_protocol": False,
    }:
        raise ValueError("confirm exposure boundary drifted")
    freeze = _require_mapping(value.get("freeze_declaration"), name="freeze_declaration")
    if dict(freeze) != {
        "parent_exposure_manifest_modified": False,
        "spatial_audit_policy_output_generated": False,
        "confirm_policy_output_generated": False,
        "restoration_output_generated": False,
        "semantic_equivalence_result_generated": False,
    }:
        raise ValueError("spatial-reference exposure freeze declaration drifted")
    semantics = _require_mapping(value.get("semantics"), name="semantics")
    if (
        semantics.get("append_only_child_ledger") is not True
        or semantics.get("v2_1_no_go_is_immutable") is not True
    ):
        raise ValueError("spatial-reference exposure semantics drifted")


def _profile_spec(value: Mapping[str, Any]) -> ProfileSpec:
    profile_id = value.get("profile_id")
    dtype = value.get("dtype")
    attention = value.get("attention_implementation")
    deterministic = value.get("deterministic_algorithms")
    eager_controls = value.get("eager_numerical_controls")
    seed = value.get("seed")
    indices = value.get("state_indices")
    generation_repeats = value.get("generation_repeat_count")
    shared_repeats = value.get("shared_prefix_forward_repeat_count")
    full_action_count = value.get("full_parent_action_forward_count")
    if profile_id not in EXPECTED_PROFILE_IDS:
        raise ValueError("unknown spatial-reference audit profile")
    if dtype not in {"bfloat16", "float32"}:
        raise ValueError("audit profile dtype must be bfloat16 or float32")
    if attention not in {"default", "eager"}:
        raise ValueError("audit attention implementation must be default or eager")
    if (
        type(deterministic) is not bool
        or type(eager_controls) is not bool
        or type(seed) is not int
        or seed != 0
    ):
        raise ValueError("audit deterministic flag or seed drifted")
    if not isinstance(indices, list) or any(type(item) is not int for item in indices):
        raise TypeError("audit profile state indices must be integers")
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise ValueError("audit profile state indices must be sorted and unique")
    if (
        type(generation_repeats) is not int
        or generation_repeats != 2
        or shared_repeats != 2
        or full_action_count != 2
    ):
        raise ValueError("audit profile repeat schedule drifted")
    return ProfileSpec(
        profile_id=profile_id,
        dtype=dtype,
        attention_implementation=attention,
        deterministic_algorithms=deterministic,
        eager_numerical_controls=eager_controls,
        seed=seed,
        state_indices=tuple(indices),
        generation_repeat_count=generation_repeats,
        shared_prefix_forward_repeat_count=shared_repeats,
        full_parent_action_forward_count=full_action_count,
    )


def validate_audit_config(value: Mapping[str, Any]) -> tuple[ProfileSpec, ...]:
    if value.get("schema_version") != "1.0.0" or value.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("spatial-reference audit config identity drifted")
    if value.get("status") != "source_only_frozen_before_any_audit_policy_forward":
        raise ValueError("spatial-reference audit status drifted")
    scope = _require_mapping(value.get("scope"), name="scope")
    required_false = (
        "confirm_policy_output_access_allowed",
        "restoration_allowed",
        "gate_training_allowed",
        "semantic_equivalence_promotion_allowed",
        "coordinate_radius_tuning_allowed",
    )
    if any(scope.get(key) is not False for key in required_false):
        raise ValueError("spatial-reference audit scope was relaxed")
    if scope.get("parent_v2_1_outcome_remains_final") != PARENT_OUTCOME:
        raise ValueError("parent v2.1 outcome must remain final")
    denominator = _require_mapping(
        value.get("mismatch_denominator"),
        name="mismatch_denominator",
    )
    if (
        denominator.get("fixed_state_count") != len(EXPECTED_MISMATCH_INDICES)
        or denominator.get("state_indices") != list(EXPECTED_MISMATCH_INDICES)
        or denominator.get("role_counts")
        != {"v2_label_train": 10, "v2_development": 3}
        or denominator.get("action_type_counts") != {"click": 12, "swipe": 1}
        or denominator.get("state_filtering_allowed") is not False
        or denominator.get("top_up_allowed") is not False
    ):
        raise ValueError("spatial-reference mismatch denominator drifted")
    _require_sha256(
        denominator.get("canonical_parent_projection_sha256"),
        name="canonical_parent_projection_sha256",
    )
    profiles_raw = value.get("profiles")
    if not isinstance(profiles_raw, list) or len(profiles_raw) != 3:
        raise ValueError("spatial-reference audit requires exactly three profiles")
    profiles = tuple(_profile_spec(_require_mapping(item, name="profile")) for item in profiles_raw)
    if tuple(profile.profile_id for profile in profiles) != EXPECTED_PROFILE_IDS:
        raise ValueError("spatial-reference audit profile order drifted")
    if profiles[0].state_indices != EXPECTED_MISMATCH_INDICES:
        raise ValueError("bf16 auto profile denominator drifted")
    if profiles[1].state_indices != EXPECTED_MISMATCH_INDICES:
        raise ValueError("bf16 eager profile denominator drifted")
    if profiles[2].state_indices != (10, 17, 26, 39):
        raise ValueError("fp32 sentinel denominator drifted")
    if (
        profiles[0].dtype != "bfloat16"
        or profiles[0].attention_implementation != "default"
        or profiles[0].deterministic_algorithms is not False
        or profiles[0].eager_numerical_controls is not False
        or profiles[1].dtype != "bfloat16"
        or profiles[1].attention_implementation != "eager"
        or profiles[1].deterministic_algorithms is not False
        or profiles[1].eager_numerical_controls is not True
        or profiles[2].dtype != "float32"
        or profiles[2].attention_implementation != "eager"
        or profiles[2].deterministic_algorithms is not False
        or profiles[2].eager_numerical_controls is not True
    ):
        raise ValueError("spatial-reference audit profile semantics drifted")
    backend = _require_mapping(
        value.get("backend_constraints"),
        name="backend_constraints",
    )
    if (
        backend.get("scientific_environment_variables_allowed") is not False
        or backend.get("cublas_workspace_config_allowed") is not False
        or backend.get("pytorch_strict_deterministic_algorithms_enabled") is not False
        or backend.get("claim")
        != "eager_fixed_seed_tf32_disabled_numerical_control_not_strict_cuda_determinism"
    ):
        raise ValueError("spatial-reference backend claim drifted")
    controls = _require_mapping(
        backend.get("eager_control_flags"),
        name="backend_constraints.eager_control_flags",
    )
    if dict(controls) != {
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "seed": 0,
    }:
        raise ValueError("spatial-reference eager numerical controls drifted")
    runtime = _require_mapping(
        value.get("runtime_constraints"),
        name="runtime_constraints",
    )
    if dict(runtime) != EXPECTED_RUNTIME_CONSTRAINTS:
        raise ValueError("spatial-reference pinned runtime constraints drifted")
    attempt = _require_mapping(value.get("attempt_identity"), name="attempt_identity")
    if dict(attempt) != {
        "attempt_id": "spatial-reference-audit-v1",
        "canonical_persistent_output_dir": "/data/experiments/causalcache/spatial-reference-audit-v1",
        "canonical_global_attempt_ledger": "/data/experiments/causalcache/.spatial-reference-audit-v1.attempt.json",
        "canonical_host_alias": "hyper01",
        "canonical_host_hostname": "node-radixark-16-0001",
        "canonical_device": "cuda:0",
        "profile_order": list(EXPECTED_PROFILE_IDS),
        "alternate_output_or_ledger_allowed": False,
        "incomplete_attempt_retry_allowed": False,
        "profile_retry_allowed": False,
        "state_retry_allowed": False,
        "output_or_ledger_deletion_after_claim_allowed": False,
    }:
        raise ValueError("spatial-reference one-attempt identity drifted")
    limits = _require_mapping(value.get("operation_limits"), name="operation_limits")
    if dict(limits) != {
        "generation_calls": 60,
        "teacher_forwards": 120,
        "confirm_state_accesses": 0,
        "restoration_coalition_constructions": 0,
        "gate_training_examples": 0,
    }:
        raise ValueError("spatial-reference audit operation limits drifted")
    diagnostics = _require_mapping(value.get("diagnostics"), name="diagnostics")
    if diagnostics.get("instrumented_generation_with_output_scores_allowed") is not False:
        raise ValueError("instrumented generation is not part of the primary audit")
    if diagnostics.get("full_vocabulary_logits_host_transfer_allowed") is not False:
        raise ValueError("full-vocabulary host transfer is forbidden")
    rule = _require_mapping(value.get("decision_rule"), name="decision_rule")
    if dict(rule) != {
        "exact_stability_criterion": (
            "all_profile_states_have_one_exact_generated_token_sequence_"
            "across_two_repeats"
        ),
        "if_bf16_eager_unstable": (
            "PERSISTENT_INSTABILITY_REQUIRES_SEMANTIC_REFERENCE"
        ),
        "if_bf16_eager_stable_and_bf16_auto_unstable": (
            "EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY"
        ),
        "if_bf16_eager_stable_and_bf16_auto_stable": (
            "BOTH_BF16_PROFILES_STABLE_THIS_ATTEMPT_NUMERICAL_AUDIT_INCONCLUSIVE"
        ),
        "both_stable_allows_eager_recovery_attribution": False,
        "strict_cuda_determinism_claimed": False,
        "margin_threshold_used_for_pass_fail": False,
        "fp32_probe_used_for_pass_fail": False,
    }:
        raise ValueError("spatial-reference audit decision rule drifted")
    return profiles


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    value = load_json_object(path)
    validate_audit_config(value)
    return value


def profile_by_id(config: Mapping[str, Any], profile_id: str) -> ProfileSpec:
    profiles = validate_audit_config(config)
    matches = [profile for profile in profiles if profile.profile_id == profile_id]
    if len(matches) != 1:
        raise ValueError("profile id is not uniquely defined")
    return matches[0]


def validate_repository_inputs(
    repository_root: str | Path,
    config: Mapping[str, Any],
) -> None:
    root = Path(repository_root).resolve()
    inputs = _require_mapping(config.get("inputs"), name="inputs")
    for name in (
        "parent_result_artifact",
        "parent_mismatch_fixture",
        "exposure_ledger",
        "derived_artifact",
        "selection_manifest",
        "scientific_config",
        "ocr_backend_config",
        "model_snapshot_manifest",
    ):
        record = _require_mapping(inputs.get(name), name=f"inputs.{name}")
        path_value = record.get("path") if name != "derived_artifact" else record.get("manifest_path")
        digest_value = record.get("sha256") if name != "derived_artifact" else record.get("manifest_sha256")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"inputs.{name} lacks a repository path")
        expected = _require_sha256(digest_value, name=f"inputs.{name}.sha256")
        path = root / path_value
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"repository input hash drifted: {name}")
    source_inventory = _require_mapping(config.get("source_inventory"), name="source_inventory")
    if not source_inventory:
        raise ValueError("spatial-reference source inventory is empty")
    for relative, digest in source_inventory.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("source inventory path is invalid")
        expected = _require_sha256(digest, name=f"source_inventory.{relative}")
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"spatial-reference source hash drifted: {relative}")
    exposure_record = _require_mapping(
        inputs.get("exposure_ledger"),
        name="inputs.exposure_ledger",
    )
    validate_exposure_ledger(
        load_json_object(root / str(exposure_record["path"])),
        config=config,
    )


def _canonical_action(value: Any) -> dict[str, Any]:
    action = _require_mapping(value, name="canonical_action")
    action_name = action.get("action")
    if action_name == "click":
        expected = {"action", "coordinate"}
    elif action_name == "swipe":
        expected = {"action", "coordinate", "coordinate2"}
    else:
        raise ValueError("parent mismatch action must be click or swipe")
    if set(action) != expected:
        raise ValueError("parent mismatch action schema drifted")
    for key in expected - {"action"}:
        coordinate = action.get(key)
        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 2
            or any(type(item) is not int or item < 0 or item > 999 for item in coordinate)
        ):
            raise ValueError("parent mismatch coordinate drifted")
    return dict(action)


def action_from_mapping(value: Mapping[str, Any]) -> GUIOwlV2Action:
    action = _canonical_action(value)
    coordinate = tuple(action["coordinate"])
    coordinate2 = tuple(action["coordinate2"]) if "coordinate2" in action else None
    return GUIOwlV2Action(
        action=str(action["action"]),
        coordinate=(int(coordinate[0]), int(coordinate[1])),
        coordinate2=(
            (int(coordinate2[0]), int(coordinate2[1]))
            if coordinate2 is not None
            else None
        ),
    )


def _fixture_by_index(fixture: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    if fixture.get("schema_version") != "1.0.0" or fixture.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("parent mismatch fixture identity drifted")
    records = fixture.get("mismatches")
    if not isinstance(records, list) or len(records) != len(EXPECTED_MISMATCH_INDICES):
        raise ValueError("parent mismatch fixture denominator drifted")
    result: dict[int, Mapping[str, Any]] = {}
    for raw in records:
        record = _require_mapping(raw, name="fixture mismatch")
        index = record.get("index")
        if type(index) is not int or index in result:
            raise ValueError("fixture mismatch index is invalid")
        result[index] = record
    if tuple(sorted(result)) != EXPECTED_MISMATCH_INDICES:
        raise ValueError("fixture mismatch indices drifted")
    return result


def load_parent_mismatches(
    *,
    raw_archive: str | Path,
    fixture_path: str | Path,
    config: Mapping[str, Any],
) -> tuple[ParentMismatch, ...]:
    validate_audit_config(config)
    parent_input = _require_mapping(
        _require_mapping(config.get("inputs"), name="inputs").get("parent_raw_archive"),
        name="inputs.parent_raw_archive",
    )
    expected_archive_sha = _require_sha256(parent_input.get("sha256"), name="parent archive sha256")
    archive = Path(raw_archive)
    if not archive.is_file() or sha256_file(archive) != expected_archive_sha:
        raise ValueError("parent raw archive hash drifted")
    evidence = read_full_45_evidence_archive(
        archive,
        expected_source_git_commit=PARENT_SOURCE_COMMIT,
    )
    if evidence.outcome != PARENT_OUTCOME or evidence.completed_state_count != 45:
        raise ValueError("parent full-45 evidence identity drifted")
    fixture = load_json_object(fixture_path)
    fixture_records = _fixture_by_index(fixture)
    mismatches: list[ParentMismatch] = []
    observed_failed: list[int] = []
    parent_projection: list[dict[str, Any]] = []
    for index in range(45):
        member = f"states/{index:03d}.json"
        payload = evidence.files.get(member)
        if payload is None:
            raise ValueError(f"parent evidence lacks state member {member}")
        record = json.loads(payload)
        if not isinstance(record, Mapping):
            raise TypeError("parent state record must be a mapping")
        failure = record.get("failure")
        if isinstance(failure, Mapping) and failure.get("category") == MISMATCH_CATEGORY:
            observed_failed.append(index)
            native = record.get("native_generations")
            if not isinstance(native, list) or len(native) != 2:
                raise ValueError("parent mismatch native-generation inventory drifted")
            parent_projection.append(
                {
                    "state": record.get("state"),
                    "actions": [item.get("canonical_action") for item in native],
                    "token_sha256": [
                        _require_mapping(item.get("metadata"), name="generation metadata").get(
                            "generated_token_ids_sha256"
                        )
                        for item in native
                    ],
                }
            )
        if index not in fixture_records:
            continue
        fixture_record = fixture_records[index]
        if not isinstance(failure, Mapping) or failure.get("category") != MISMATCH_CATEGORY:
            raise ValueError("fixture state is not a parent canonical-action mismatch")
        if record.get("teacher_forwards") != {}:
            raise ValueError("parent mismatch unexpectedly contains teacher forwards")
        operation_counts = _require_mapping(record.get("operation_counts"), name="operation_counts")
        if operation_counts.get("teacher_forward_count") != 0 or operation_counts.get("kl_measurement_count") != 0:
            raise ValueError("parent mismatch operation counts drifted")
        state = _require_mapping(record.get("state"), name="state")
        if (
            state.get("index") != index
            or state.get("role") != fixture_record.get("role")
            or state.get("state_id") != fixture_record.get("state_id")
            or record.get("screen_dimensions") != fixture_record.get("screen_dimensions")
        ):
            raise ValueError("parent mismatch state projection drifted")
        generations = record.get("native_generations")
        if not isinstance(generations, list) or len(generations) != 2:
            raise ValueError("parent mismatch must contain two native generations")
        expected_actions = fixture_record.get("actions")
        if not isinstance(expected_actions, list) or len(expected_actions) != 2:
            raise ValueError("fixture actions must contain two values")
        actions: list[Mapping[str, Any]] = []
        texts: list[str] = []
        token_hashes: list[str] = []
        bridges: list[Mapping[str, Any]] = []
        for generation, expected_action in zip(generations, expected_actions, strict=True):
            generation_map = _require_mapping(generation, name="native generation")
            action = _canonical_action(generation_map.get("canonical_action"))
            if action != _canonical_action(expected_action):
                raise ValueError("parent mismatch action differs from frozen fixture")
            output_text = generation_map.get("output_text")
            metadata = _require_mapping(generation_map.get("metadata"), name="generation metadata")
            bridge = _require_mapping(generation_map.get("androidworld_bridge"), name="bridge")
            if not isinstance(output_text, str) or not output_text:
                raise ValueError("parent mismatch output text is invalid")
            if metadata.get("decoded_output_utf8_sha256") != hashlib.sha256(
                output_text.encode("utf-8")
            ).hexdigest():
                raise ValueError("parent mismatch output-text SHA256 drifted")
            token_hash = _require_sha256(
                metadata.get("generated_token_ids_sha256"),
                name="generated_token_ids_sha256",
            )
            actions.append(action)
            texts.append(output_text)
            token_hashes.append(token_hash)
            bridges.append(dict(bridge))
        mismatches.append(
            ParentMismatch(
                index=index,
                role=str(state["role"]),
                state_id=str(state["state_id"]),
                screen_dimensions=dict(record["screen_dimensions"]),
                actions=(actions[0], actions[1]),
                output_texts=(texts[0], texts[1]),
                generated_token_ids_sha256=(token_hashes[0], token_hashes[1]),
                androidworld_bridges=(bridges[0], bridges[1]),
                state_member_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    if tuple(observed_failed) != EXPECTED_MISMATCH_INDICES:
        raise ValueError("parent mismatch set differs from the fixed 13-state denominator")
    expected_projection_sha = _require_sha256(
        _require_mapping(
            config.get("mismatch_denominator"),
            name="mismatch_denominator",
        ).get("canonical_parent_projection_sha256"),
        name="canonical_parent_projection_sha256",
    )
    projection_bytes = canonical_json_bytes(parent_projection) + b"\n"
    if hashlib.sha256(projection_bytes).hexdigest() != expected_projection_sha:
        raise ValueError("parent mismatch canonical projection SHA256 drifted")
    return tuple(mismatches)


def first_divergent_index(left: Sequence[int], right: Sequence[int]) -> int | None:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit if len(left) != len(right) else None


def retokenize_parent_mismatch(
    mismatch: ParentMismatch,
    tokenizer: Any,
) -> dict[str, Any]:
    token_rows = []
    for output_text, expected_hash in zip(
        mismatch.output_texts,
        mismatch.generated_token_ids_sha256,
        strict=True,
    ):
        values = tokenizer.encode(output_text, add_special_tokens=False)
        if not isinstance(values, list) or any(type(item) is not int or item < 0 for item in values):
            raise TypeError("retokenized parent output must be non-negative integer ids")
        digest = hashlib.sha256(canonical_json_bytes(values)).hexdigest()
        if digest != expected_hash:
            raise ValueError("retokenized parent output token hash differs from raw evidence")
        token_rows.append(values)
    divergence = first_divergent_index(token_rows[0], token_rows[1])
    if divergence is None:
        raise ValueError("parent mismatch output token paths unexpectedly agree")
    return {
        "token_ids": token_rows,
        "first_divergent_token_index": divergence,
        "first_divergent_token_ids": [
            token_rows[0][divergence] if divergence < len(token_rows[0]) else None,
            token_rows[1][divergence] if divergence < len(token_rows[1]) else None,
        ],
        "token_hash_validation": True,
    }


def coordinate_delta(mismatch: ParentMismatch) -> dict[str, Any]:
    first, second = mismatch.actions
    if first.get("action") != second.get("action"):
        raise ValueError("parent mismatch action types differ")
    normalized: dict[str, list[int]] = {}
    for key in ("coordinate", "coordinate2"):
        if key not in first and key not in second:
            continue
        left = first.get(key)
        right = second.get(key)
        if not isinstance(left, list) or not isinstance(right, list):
            raise ValueError("coordinate pair is incomplete")
        normalized[key] = [int(right[0]) - int(left[0]), int(right[1]) - int(left[1])]
    left_bridge, right_bridge = mismatch.androidworld_bridges
    pixel: dict[str, list[int]] = {}
    if first["action"] == "click":
        pixel["coordinate"] = [
            int(right_bridge["x"]) - int(left_bridge["x"]),
            int(right_bridge["y"]) - int(left_bridge["y"]),
        ]
    else:
        left_direction = left_bridge.get("direction")
        right_direction = right_bridge.get("direction")
        if (
            not isinstance(left_direction, list)
            or not isinstance(right_direction, list)
            or len(left_direction) != 4
            or len(right_direction) != 4
        ):
            raise ValueError("parent swipe bridge direction drifted")
        pixel["coordinate"] = [
            int(right_direction[0]) - int(left_direction[0]),
            int(right_direction[1]) - int(left_direction[1]),
        ]
        pixel["coordinate2"] = [
            int(right_direction[2]) - int(left_direction[2]),
            int(right_direction[3]) - int(left_direction[3]),
        ]
    return {
        "normalized_delta": normalized,
        "pixel_delta": pixel,
        "normalized_l2": {
            key: math.hypot(value[0], value[1]) for key, value in normalized.items()
        },
        "pixel_l2": {
            key: math.hypot(value[0], value[1]) for key, value in pixel.items()
        },
    }


def profile_operation_counts(profile: ProfileSpec) -> dict[str, int]:
    state_count = len(profile.state_indices)
    return {
        "generation_calls": state_count * profile.generation_repeat_count,
        "teacher_forwards": (
            state_count
            * (
                profile.shared_prefix_forward_repeat_count
                + profile.full_parent_action_forward_count
            )
        ),
        "confirm_state_accesses": 0,
        "restoration_coalition_constructions": 0,
        "gate_training_examples": 0,
    }


def canonical_profile_payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
