"""Fail-closed source contract for the restoration-v2.1 full-45 screen."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from causalcache.restoration_v2_1_contract import RestorationV21PilotContract


PROTOCOL_ID = "causalcache_restoration_v2_1_full_45_substrate"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_1_full_45.json"
)
FROZEN_RESTORATION_V2_1_FULL_45_SHA256 = (
    "0924dd66fab9440bed66585765e9b1f5ab6fb80fbdf65a1492efba7ae81e116b"
)
FULL_45_PROJECTION_SHA256 = (
    "65e7085f01bc26e425bab7f5148c6729bc8c6d0a62802ebc5e8dfeaed6439249"
)
PILOT_CONTRACT_SHA256 = (
    "9d51a2ed5d6cc382f297c1b8af3100d784090f72800d637136b88982763fdbf7"
)
PILOT_PASS_MANIFEST_SHA256 = (
    "e7a92a0469e3b66113158d5c2ce713c363ecaff97f95498f3ef84f02a4a6bed8"
)
PROCESSOR_PASS_MANIFEST_SHA256 = (
    "2689fe984215f308e5f5230de9656cfe0439bfd2df5e5f061ebac1a1cbc8299b"
)
PARENT_V2_CONTRACT_SHA256 = (
    "9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc"
)
SELECTION_MANIFEST_SHA256 = (
    "292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b"
)
CANONICAL_ATTEMPT_ID = "restoration-v2-1-full-45-substrate-v1"
CANONICAL_OUTPUT_DIR = Path(
    "/data/experiments/causalcache/restoration-v2-1-full-45-substrate-v1"
)
CANONICAL_LEDGER_PATH = Path(
    "/data/experiments/causalcache/"
    ".restoration-v2-1-full-45-substrate-v1.attempt.json"
)
CANONICAL_ARCHIVE_PATH = Path(
    "/data/experiments/causalcache/"
    "restoration-v2-1-full-45-substrate-v1.raw.tar"
)
CANONICAL_HF_REPO = (
    "gavinlaw/causalcache-restoration-v2-1-full-45-substrate-mobile"
)
CANONICAL_HF_TAG = "v2.1-full-45-substrate-v1"
CANONICAL_HF_PATH = "raw/restoration-v2-1-full-45-substrate-v1.tar"
PASS_OUTCOME = "PASS_V2_1_FULL_45_SUBSTRATE"
NO_GO_OUTCOME = "NO_GO_V2_1_FULL_45_SUBSTRATE"
INVALID_OUTCOME = "INVALID_V2_1_FULL_45_SUBSTRATE"
SCREENING_ROLES = ("v2_label_train", "v2_development")
ROLE_STATE_COUNTS = {"v2_label_train": 30, "v2_development": 15}
PROJECTION_SCHEMA = (
    "index",
    "role",
    "trajectory_id",
    "decision_step_id",
    "state_id",
    "candidate_event_step_ids",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


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
        value,
        Sequence,
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


def _resolve_repository_file(
    repository_root: Path,
    record: Any,
    *,
    name: str,
) -> tuple[str, Path]:
    value = _mapping(record, name)
    _exact_keys(value, {"path", "sha256"}, name)
    relative = _safe_relative_path(value["path"], f"{name}.path")
    digest = value["sha256"]
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name}.sha256 is invalid")
    root = repository_root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"{name}.path is missing or escapes the repository")
    if _sha256_file(path) != digest:
        raise ValueError(f"{name} SHA256 drifted")
    return relative, path


def _require_committed_file(
    repository_root: Path,
    relative: str,
    worktree_path: Path,
    *,
    name: str,
) -> None:
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=repository_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(f"{name} is not committed at HEAD")
    if result.stdout != worktree_path.read_bytes():
        raise ValueError(f"{name} differs from its committed HEAD blob")


def _source_record(
    value: Mapping[str, Any],
    *,
    repository_root: Path,
    name: str,
    committed: bool,
) -> Path:
    relative, path = _resolve_repository_file(repository_root, value, name=name)
    if committed:
        _require_committed_file(
            repository_root,
            relative,
            path,
            name=name,
        )
    return path


def _full_45_projection(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    roles = _mapping(selection.get("roles"), "selection.roles")
    projection: list[dict[str, Any]] = []
    for role in SCREENING_ROLES:
        role_record = _mapping(roles.get(role), f"selection.roles.{role}")
        states = _sequence(role_record.get("states"), f"selection {role} states")
        _equal(len(states), ROLE_STATE_COUNTS[role], f"selection {role} state count")
        previous_trajectory: str | None = None
        expected_step = 4
        for state_value in states:
            state = _mapping(state_value, f"selection {role} state")
            source_id = state.get("source_id")
            decision_step_id = state.get("decision_step_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"selection {role} source ID is invalid")
            if source_id != previous_trajectory:
                expected_step = 4
                previous_trajectory = source_id
            _equal(decision_step_id, expected_step, f"selection {role} step order")
            _equal(
                state.get("state_id"),
                f"{source_id}:decision_step:{expected_step:03d}",
                f"selection {role} state ID",
            )
            _equal(
                state.get("candidate_event_step_ids"),
                list(range(1, expected_step - 1)),
                f"selection {role} candidate event IDs",
            )
            projection.append(
                {
                    "index": len(projection),
                    "role": role,
                    "trajectory_id": source_id,
                    "decision_step_id": decision_step_id,
                    "state_id": state.get("state_id"),
                    "candidate_event_step_ids": state.get(
                        "candidate_event_step_ids"
                    ),
                }
            )
            expected_step += 1
            if expected_step == 7:
                previous_trajectory = None
    _equal(len(projection), 45, "full-45 projection denominator")
    return projection


def _validate_parent_authorization(
    value: Mapping[str, Any],
    *,
    repository_root: Path,
) -> None:
    _exact_keys(
        value,
        {
            "pilot_contract",
            "fixed_15_pass_manifest",
            "processor_pass_manifest",
            "parent_v2_scientific_contract",
            "pilot_outputs_may_be_counted_in_full_45_denominator",
            "parent_v2_outputs_may_be_relabelled_as_v2_1",
        },
        "parent_authorization",
    )
    _false(
        value["pilot_outputs_may_be_counted_in_full_45_denominator"],
        "pilot denominator reuse",
    )
    _false(
        value["parent_v2_outputs_may_be_relabelled_as_v2_1"],
        "parent v2 relabelling",
    )

    pilot_record = _mapping(value["pilot_contract"], "pilot contract")
    pilot_source = {key: pilot_record[key] for key in ("path", "sha256")}
    _equal(pilot_source["sha256"], PILOT_CONTRACT_SHA256, "pilot contract SHA256")
    _source_record(
        pilot_source,
        repository_root=repository_root,
        name="pilot contract",
        committed=True,
    )
    pilot = RestorationV21PilotContract.load(
        pilot_source["path"],
        repository_root=repository_root,
    )
    _equal(pilot.protocol_id, pilot_record.get("protocol_id"), "pilot protocol")
    _equal(
        pilot.data["promotion"]["pilot_pass_authorizes_only"],
        pilot_record.get("required_promotion"),
        "pilot promotion",
    )

    pilot_manifest_record = _mapping(
        value["fixed_15_pass_manifest"],
        "fixed-15 pass manifest",
    )
    pilot_manifest_source = {
        key: pilot_manifest_record[key] for key in ("path", "sha256")
    }
    _equal(
        pilot_manifest_source["sha256"],
        PILOT_PASS_MANIFEST_SHA256,
        "fixed-15 pass manifest SHA256",
    )
    pilot_manifest_path = _source_record(
        pilot_manifest_source,
        repository_root=repository_root,
        name="fixed-15 pass manifest",
        committed=True,
    )
    pilot_manifest = load_strict_json_object(pilot_manifest_path)
    pilot_result = _mapping(pilot_manifest.get("result"), "pilot result")
    pilot_hf = _mapping(pilot_manifest.get("hf_artifact"), "pilot HF artifact")
    pilot_raw = _mapping(pilot_manifest.get("raw_archive"), "pilot raw archive")
    pilot_execution = _mapping(
        pilot_manifest.get("source_execution"),
        "pilot source execution",
    )
    pilot_expected = {
        "required_status": pilot_result.get("status"),
        "required_outcome": pilot_result.get("outcome"),
        "required_fixed_state_denominator": pilot_result.get(
            "fixed_state_denominator"
        ),
        "required_completed_state_count": pilot_result.get(
            "completed_state_count"
        ),
        "source_git_commit": pilot_execution.get("source_git_commit"),
        "hf_repo": pilot_hf.get("repo"),
        "hf_immutable_revision": pilot_hf.get("immutable_revision"),
        "hf_tag": pilot_hf.get("tag"),
        "hf_path": pilot_hf.get("path"),
        "raw_archive_sha256": pilot_raw.get("sha256"),
        "raw_archive_size_bytes": pilot_raw.get("size_bytes"),
        "fresh_immutable_archive_validation_required_before_runtime_import": True,
    }
    for field, expected in pilot_expected.items():
        _equal(pilot_manifest_record.get(field), expected, f"pilot binding {field}")
    _equal(pilot_result.get("attempted_state_count"), 15, "pilot attempted states")
    _equal(pilot_hf.get("repo_type"), "dataset", "pilot HF repo type")
    _equal(pilot_hf.get("visibility"), "private", "pilot HF visibility")

    processor_record = _mapping(
        value["processor_pass_manifest"],
        "processor pass manifest",
    )
    processor_source = {
        key: processor_record[key] for key in ("path", "sha256")
    }
    _equal(
        processor_source["sha256"],
        PROCESSOR_PASS_MANIFEST_SHA256,
        "processor pass manifest SHA256",
    )
    processor_path = _source_record(
        processor_source,
        repository_root=repository_root,
        name="processor pass manifest",
        committed=True,
    )
    processor_manifest = load_strict_json_object(processor_path)
    processor_reduction = _mapping(
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
    processor_expected = {
        "required_status": processor_reduction.get("status"),
        "required_state_count": processor_reduction.get("state_count"),
        "required_prompt_count": processor_reduction.get("prompt_count"),
        "source_git_commit": processor_raw.get("source_git_commit"),
        "hf_repo": processor_hf.get("repo"),
        "hf_immutable_revision": processor_hf.get("immutable_revision"),
        "hf_tag": processor_hf.get("tag"),
        "hf_path": processor_hf.get("path"),
        "raw_evidence_sha256": processor_raw.get("sha256"),
        "raw_evidence_size_bytes": processor_raw.get("size_bytes"),
        "fresh_immutable_evidence_validation_required_before_runtime_import": True,
    }
    for field, expected in processor_expected.items():
        _equal(
            processor_record.get(field),
            expected,
            f"processor binding {field}",
        )
    _equal(
        processor_reduction.get("contract_sha256"),
        PILOT_CONTRACT_SHA256,
        "processor pilot contract binding",
    )
    _equal(
        processor_reduction.get("selection_manifest_sha256"),
        SELECTION_MANIFEST_SHA256,
        "processor selection binding",
    )
    _equal(processor_hf.get("repo_type"), "dataset", "processor HF repo type")
    _equal(processor_hf.get("visibility"), "private", "processor visibility")

    parent_record = _mapping(
        value["parent_v2_scientific_contract"],
        "parent v2 contract",
    )
    parent_source = {key: parent_record[key] for key in ("path", "sha256")}
    _equal(
        parent_source["sha256"],
        PARENT_V2_CONTRACT_SHA256,
        "parent v2 contract SHA256",
    )
    parent_path = _source_record(
        parent_source,
        repository_root=repository_root,
        name="parent v2 contract",
        committed=True,
    )
    parent = load_strict_json_object(parent_path)
    _equal(parent.get("protocol_id"), parent_record.get("protocol_id"), "parent protocol")


def _validate_source_lock(
    value: Mapping[str, Any],
    *,
    repository_root: Path,
) -> None:
    _exact_keys(
        value,
        {
            "unchanged_official_tool_interface_required",
            "unchanged_generation_parser_teacher_and_distance_required",
            "files",
            "formal_run_source_inventory_paths",
            "formal_run_must_bind_clean_pushed_main_git_commit",
            "scientific_source_change_after_first_attempt_allowed",
        },
        "source_lock",
    )
    _true(value["unchanged_official_tool_interface_required"], "interface lock")
    _true(
        value["unchanged_generation_parser_teacher_and_distance_required"],
        "generation/teacher lock",
    )
    _true(
        value["formal_run_must_bind_clean_pushed_main_git_commit"],
        "formal Git binding",
    )
    _false(
        value["scientific_source_change_after_first_attempt_allowed"],
        "post-attempt source changes",
    )
    records = _sequence(value["files"], "source_lock.files")
    paths: list[str] = []
    for index, record_value in enumerate(records):
        record = _mapping(record_value, f"source_lock.files[{index}]")
        relative, _ = _resolve_repository_file(
            repository_root,
            record,
            name=f"source_lock.files[{index}]",
        )
        paths.append(relative)
    if len(paths) != len(set(paths)):
        raise ValueError("source_lock.files contains duplicate paths")
    _equal(
        paths[:2],
        [
            "code/causalcache/policy/gui_owl_v2_1.py",
            "code/causalcache/policy/gui_owl_v2_1_runtime.py",
        ],
        "official-tool source order",
    )
    inventory = _sequence(
        value["formal_run_source_inventory_paths"],
        "formal run source inventory",
    )
    normalized = [
        _safe_relative_path(item, "formal run source inventory path")
        for item in inventory
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("formal run source inventory contains duplicate paths")
    required = {
        CANONICAL_CONFIG_PATH,
        "code/causalcache/restoration_v2_1_full_45_contract.py",
        "code/causalcache/restoration_v2_1_full_45_artifact.py",
        "code/scripts/run_restoration_v2_1_full_45_substrate.py",
        "code/scripts/run_restoration_v2_1_interface_pilot.py",
        "code/causalcache/restoration_v2_1_contract.py",
        "code/causalcache/restoration_v2_1_pilot_artifact.py",
        "code/scripts/manage_restoration_v2_1_pilot_artifact.py",
        "code/causalcache/restoration_v2_1_processor_audit.py",
        "code/causalcache/data/restoration_v2_screening.py",
        "code/causalcache/policy/gui_owl_v2_1.py",
        "code/causalcache/policy/gui_owl_v2_1_runtime.py",
        "code/causalcache/policy/gui_owl_v2.py",
        "code/causalcache/policy/gui_owl_v2_runtime.py",
        "code/causalcache/restoration_v2_gpu_kl.py",
        "code/configs/gui_owl_1_5_8b_snapshot.json",
        "code/configs/restoration_v2_ocr_backend.json",
        "data/manifests/restoration_v2_selection.json",
        "data/manifests/restoration_v2_ocr_backend.json",
        "data/manifests/restoration_v2_derived_artifact.json",
        "data/results/restoration_v2_1_interface_pilot/artifact.json",
        "data/results/restoration_v2_1_processor_preflight/artifact.json",
    }
    missing = sorted(required.difference(normalized))
    if missing:
        raise ValueError(f"formal run source inventory is incomplete: {missing}")


def _validate_data(
    value: Mapping[str, Any],
    *,
    repository_root: Path,
) -> list[dict[str, Any]]:
    selection_record = _mapping(value.get("selection_manifest"), "selection manifest")
    _equal(
        selection_record.get("sha256"),
        SELECTION_MANIFEST_SHA256,
        "selection manifest SHA256",
    )
    _, selection_path = _resolve_repository_file(
        repository_root,
        selection_record,
        name="selection manifest",
    )
    selection = load_strict_json_object(selection_path)
    projection = _full_45_projection(selection)
    _equal(value.get("roles_in_exact_order"), list(SCREENING_ROLES), "role order")
    _equal(value.get("role_state_counts"), ROLE_STATE_COUNTS, "role state counts")
    _equal(
        value.get("role_index_ranges"),
        {"v2_label_train": [0, 29], "v2_development": [30, 44]},
        "role index ranges",
    )
    _equal(value.get("fixed_state_denominator"), 45, "fixed denominator")
    _equal(
        value.get("decision_step_order_within_trajectory"),
        [4, 5, 6],
        "decision-step order",
    )
    _equal(
        tuple(value.get("state_projection_schema", ())),
        PROJECTION_SCHEMA,
        "projection schema",
    )
    _equal(
        _canonical_json_sha256(projection),
        FULL_45_PROJECTION_SHA256,
        "derived projection SHA256",
    )
    _equal(
        value.get("state_projection_sha256"),
        FULL_45_PROJECTION_SHA256,
        "declared projection SHA256",
    )
    _equal(
        value.get("state_ids_in_exact_order"),
        [record["state_id"] for record in projection],
        "exact state order",
    )
    _true(
        value.get("full_history_reference_and_summary_only_only"),
        "screening fidelity restriction",
    )
    _false(value.get("state_filtering_or_top_up_allowed"), "state filtering")
    _equal(value.get("confirm_role"), "v2_confirm_primary", "confirm role")
    _true(
        value.get("confirm_bytes_may_only_be_validated_by_full_artifact_loader"),
        "confirm loader-only access",
    )
    _false(
        value.get("confirm_prompt_image_or_state_exposure_allowed"),
        "confirm exposure",
    )
    derived = _mapping(value.get("derived_artifact"), "derived artifact")
    _equal(
        dict(derived),
        {
            "repo": "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
            "immutable_revision": "89f136abaff797e14fe758a198996e51032a10a6",
            "artifact_tree_sha256": "475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e",
        },
        "derived artifact identity",
    )
    return projection


def validate_restoration_v2_1_full_45_contract(
    data: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    _exact_keys(
        data,
        {
            "schema_version",
            "protocol_id",
            "preregistration_status",
            "parent_authorization",
            "source_lock",
            "data",
            "computation_schedule",
            "substrate_gate",
            "prohibited_work",
            "execution",
            "artifact_destination",
            "promotion",
        },
        "full-45 contract",
    )
    _equal(data["schema_version"], "0.1.0", "schema version")
    _equal(data["protocol_id"], PROTOCOL_ID, "protocol ID")
    _equal(
        data["preregistration_status"],
        "source_only_frozen_before_any_full_45_policy_output",
        "preregistration status",
    )
    root = repository_root.resolve()
    _validate_parent_authorization(
        _mapping(data["parent_authorization"], "parent_authorization"),
        repository_root=root,
    )
    _validate_source_lock(
        _mapping(data["source_lock"], "source_lock"),
        repository_root=root,
    )
    projection = _validate_data(
        _mapping(data["data"], "data"),
        repository_root=root,
    )

    schedule = _mapping(data["computation_schedule"], "computation schedule")
    _equal(
        schedule.get("per_state_maximum"),
        {
            "full_history_generation_calls": 2,
            "teacher_forward_calls_on_success": 3,
            "kl_measurements_on_success": 2,
        },
        "per-state maximum schedule",
    )
    expected_totals = {
        "generation_call_count": 90,
        "teacher_forward_count": 135,
        "kl_measurement_count": 90,
    }
    _equal(
        schedule.get("planned_counts_if_all_45_states_reach_success_stage"),
        expected_totals,
        "planned successful schedule",
    )
    _equal(
        schedule.get("maximum_counts_for_entire_attempt"),
        expected_totals,
        "maximum attempt schedule",
    )
    _equal(
        schedule.get("generation_order_per_state"),
        ["full_history_reference_repeat_1", "full_history_reference_repeat_2"],
        "generation order",
    )
    _equal(
        schedule.get("teacher_order_on_success_per_state"),
        [
            "full_history_reference_1",
            "full_history_reference_2",
            "summary_only",
        ],
        "teacher order",
    )
    _equal(
        schedule.get("kl_order_on_success_per_state"),
        ["repeat_reference_kl", "summary_reference_kl"],
        "KL order",
    )
    for field in (
        "automatic_retry_allowed",
        "completed_state_regeneration_allowed",
        "incomplete_attempt_retry_allowed",
        "sample_mutation_allowed",
    ):
        _false(schedule.get(field), f"schedule.{field}")
    for field in (
        "resume_allowed",
        "resume_only_skips_existing_terminal_state_records",
        "resume_may_attempt_only_states_without_any_attempt_marker",
        "attempt_marker_without_terminal_state_invalidates_entire_attempt",
    ):
        _true(schedule.get(field), f"schedule.{field}")
    _equal(schedule.get("state_retry_count"), 0, "state retry count")
    _equal(schedule.get("top_up_count"), 0, "top-up count")

    gate = _mapping(data["substrate_gate"], "substrate gate")
    parent_v2 = load_strict_json_object(
        root / "code/configs/causalcache_restoration_v2.json"
    )
    parent_gate = _mapping(parent_v2.get("substrate_gate"), "parent substrate gate")
    parent_fields = {
        key: parent_gate[key]
        for key in (
            "minimum_screening_states",
            "minimum_parse_coverage",
            "minimum_finite_logit_coverage",
            "minimum_repeat_canonical_action_agreement",
            "minimum_memory_sensitive_states",
            "memory_sensitive_definition",
        )
    }
    for field, expected in parent_fields.items():
        _equal(gate.get(field), expected, f"substrate gate {field}")
    _equal(gate.get("fixed_state_denominator"), 45, "gate denominator")
    _equal(
        gate.get("repeat_noise_epsilon"),
        "max(1e-4,10*mean_repeat_kl)",
        "repeat noise epsilon",
    )
    _equal(
        gate.get("derived_required_parse_success_count"),
        math.ceil(45 * 0.99),
        "required parse count",
    )
    _equal(
        gate.get("derived_required_finite_logit_state_count"),
        45,
        "required finite-logit count",
    )
    _equal(
        gate.get("derived_required_repeat_agreement_count"),
        45,
        "required repeat-agreement count",
    )
    _true(
        gate.get("failed_states_remain_in_fixed_denominator"),
        "fixed failure denominator",
    )
    _equal(gate.get("pass_outcome"), PASS_OUTCOME, "pass outcome")
    _equal(gate.get("fail_outcome"), NO_GO_OUTCOME, "no-go outcome")
    _equal(gate.get("invalid_outcome"), INVALID_OUTCOME, "invalid outcome")

    prohibited = _mapping(data["prohibited_work"], "prohibited work")
    for field, value in prohibited.items():
        if field == "androidworld_test_split_access_allowed":
            _false(value, f"prohibited_work.{field}")
        else:
            _equal(value, 0, f"prohibited_work.{field}")

    execution = _mapping(data["execution"], "execution")
    expected_execution = {
        "attempt_id": CANONICAL_ATTEMPT_ID,
        "canonical_persistent_output_dir": str(CANONICAL_OUTPUT_DIR),
        "canonical_global_attempt_ledger": str(CANONICAL_LEDGER_PATH),
        "canonical_raw_archive": str(CANONICAL_ARCHIVE_PATH),
        "canonical_host_alias": "hyper00",
        "canonical_host_hostname": "node-radixark-16-0000",
        "canonical_container_id": "69f2b1742e8fd9baac5080b2b97ee1f3c7c1df520f908a4541cadde9d28194df",
        "canonical_container_image_digest": "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa",
        "canonical_device": "cuda:0",
        "cross_host_attempt_allowed": False,
        "alternate_output_or_ledger_allowed": False,
        "output_or_ledger_deletion_after_first_attempt_allowed": False,
        "state_attempt_marker_written_before_first_generation": True,
        "runtime_import_requires_fresh_parent_evidence_validation": True,
    }
    _equal(dict(execution), expected_execution, "execution identity")

    destination = _mapping(data["artifact_destination"], "artifact destination")
    _equal(
        dict(destination),
        {
            "repo": CANONICAL_HF_REPO,
            "repo_type": "dataset",
            "visibility": "private",
            "tag": CANONICAL_HF_TAG,
            "path": CANONICAL_HF_PATH,
            "git_result_dir": "data/results/restoration_v2_1_full_45_substrate",
            "fresh_immutable_download_and_byte_hash_required_before_git_manifest": True,
        },
        "artifact destination",
    )
    promotion = _mapping(data["promotion"], "promotion")
    _equal(
        dict(promotion),
        {
            "pass_authorizes_only": (
                "freeze_and_audit_independent_v2_1_restoration_confirm_source"
            ),
            "pass_does_not_authorize_automatic_confirm_execution": True,
            "confirm_remains_locked_until_new_source_is_committed_and_pushed": True,
            "no_go_stops_v2_1_restoration_confirm": True,
            "source_only_validator_authorizes_policy_or_gpu_execution": False,
        },
        "promotion",
    )
    return {
        "fixed_state_denominator": len(projection),
        "role_state_counts": dict(ROLE_STATE_COUNTS),
        "state_projection_sha256": FULL_45_PROJECTION_SHA256,
        "maximum_generation_call_count": 90,
        "maximum_teacher_forward_count": 135,
        "maximum_kl_measurement_count": 90,
        "policy_or_gpu_execution_authorized": False,
    }


@dataclass(frozen=True)
class RestorationV21Full45Contract:
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
    ) -> "RestorationV21Full45Contract":
        root = Path(repository_root).resolve()
        supplied = Path(path)
        resolved = (
            supplied.resolve()
            if supplied.is_absolute()
            else (root / supplied).resolve()
        )
        canonical = (root / CANONICAL_CONFIG_PATH).resolve()
        if resolved != canonical or not resolved.is_file():
            raise ValueError(
                f"v2.1 full-45 contract must be {CANONICAL_CONFIG_PATH}"
            )
        source_sha256 = _sha256_file(resolved)
        if source_sha256 != FROZEN_RESTORATION_V2_1_FULL_45_SHA256:
            raise ValueError("restoration-v2.1 full-45 contract SHA256 mismatch")
        data = load_strict_json_object(resolved)
        validation = validate_restoration_v2_1_full_45_contract(
            data,
            repository_root=root,
        )
        return cls(
            data=data,
            source_sha256=source_sha256,
            validation=validation,
        )
