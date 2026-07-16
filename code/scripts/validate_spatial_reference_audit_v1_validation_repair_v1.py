"""Offline-only validation repair for the completed spatial-reference audit v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from causalcache.data.guiodyssey_restoration_v2 import (
    canonical_json_bytes,
    sha256_file,
)
from causalcache.restoration_v2_1_full_45_artifact import (
    read_full_45_evidence_archive,
)
from causalcache.spatial_reference_audit_v1 import (
    EXPECTED_PROFILE_IDS,
    load_and_validate_config,
    load_parent_mismatches,
    validate_repository_inputs,
)
from causalcache.spatial_reference_audit_v1_artifact import (
    _collect_tree_files,
    _tree_inventory_sha256,
    collect_spatial_reference_audit_files,
)
from scripts import validate_spatial_reference_audit_v1 as original_validator
from scripts.run_restoration_v2_1_interface_pilot import (
    validate_clean_pushed_main,
    validate_committed_source_blobs,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "spatial_reference_audit_v1_validation_repair_v1"
STATUS = "VALID_SPATIAL_REFERENCE_AUDIT_V1_VALIDATION_REPAIR_V1"
SOURCE_STATUS = "source_only_offline_validation_repair_frozen_before_execution"
PARENT_SHAPE_KEYS = (
    "image_count",
    "image_grid_thw",
    "effective_visual_tokens",
    "policy_visible_text_tokens",
    "prompt_input_tokens",
)
PARENT_GRID_BINDING_KEYS = (
    "image_count",
    "image_grid_thw",
    "effective_visual_tokens",
)
EXPECTED_ALIGNED_INPUTS = ("attention_mask", "mm_token_type_ids")
GRID_FORMULA = (
    "temporal_times_height_times_width_integer_divided_by_four_after_even_spatial_"
    "merge_validation"
)
ORIGINAL_FAILURE = {
    "stage": "independent_validator_before_summary_and_packaging",
    "exception_type": "ValueError",
    "message": "generation per-image effective visual token count drifted",
}


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} schema drifted")


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _full_git_sha(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase Git SHA")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_absolute_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{name} must be an absolute path")
    path = Path(value)
    if path.as_posix() != value or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{name} is not canonical")
    return path


def _validate_bound_file_config(
    value: Any,
    *,
    name: str,
    extra_keys: set[str] | None = None,
) -> Mapping[str, Any]:
    binding = _mapping(value, name=name)
    keys = {"path", "sha256", "size_bytes"} | (extra_keys or set())
    _exact_keys(binding, keys, name=name)
    _canonical_absolute_path(binding.get("path"), name=f"{name} path")
    _sha256(binding.get("sha256"), name=f"{name} SHA256")
    _positive_int(binding.get("size_bytes"), name=f"{name} size")
    return binding


def load_repair_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid validation-repair config") from error
    config = dict(_mapping(value, name="validation-repair config"))
    _exact_keys(
        config,
        {
            "schema_version",
            "protocol_id",
            "status",
            "repair_scope",
            "frozen_source",
            "parent_v2_1_archive",
            "pre_repair_evidence",
            "repair_contract",
            "outputs",
        },
        name="validation-repair config",
    )
    if (
        config["schema_version"] != SCHEMA_VERSION
        or config["protocol_id"] != PROTOCOL_ID
        or config["status"] != SOURCE_STATUS
    ):
        raise ValueError("validation-repair config identity drifted")

    scope = _mapping(config["repair_scope"], name="repair scope")
    _exact_keys(
        scope,
        {
            "purpose",
            "original_audit_source_git_commit",
            "parent_v2_1_outcome_remains_final",
            "policy_or_model_forward_allowed",
            "profile_reexecution_allowed",
            "profile_or_ledger_mutation_allowed",
            "confirm_policy_output_access_allowed",
            "restoration_allowed",
            "gate_training_allowed",
        },
        name="repair scope",
    )
    if scope.get("purpose") != (
        "repair_two_validator_only_shape_contract_defects_without_reexecuting_"
        "the_completed_policy_attempt"
    ):
        raise ValueError("validation-repair purpose drifted")
    _full_git_sha(
        scope.get("original_audit_source_git_commit"),
        name="original audit source commit",
    )
    for key in (
        "policy_or_model_forward_allowed",
        "profile_reexecution_allowed",
        "profile_or_ledger_mutation_allowed",
        "confirm_policy_output_access_allowed",
        "restoration_allowed",
        "gate_training_allowed",
    ):
        if scope.get(key) is not False:
            raise ValueError(f"validation repair must forbid {key}")
    if scope.get("parent_v2_1_outcome_remains_final") != (
        "NO_GO_V2_1_FULL_45_SUBSTRATE"
    ):
        raise ValueError("parent v2.1 outcome boundary drifted")

    source = _mapping(config["frozen_source"], name="frozen source")
    _exact_keys(
        source,
        {
            "original_config",
            "original_validator",
            "required_repository_state",
            "repair_git_commit_capture_required",
        },
        name="frozen source",
    )
    if (
        source.get("required_repository_state") != "clean_pushed_main"
        or source.get("repair_git_commit_capture_required") is not True
    ):
        raise ValueError("repair Git identity contract drifted")
    for key, expected_path in (
        ("original_config", "code/configs/spatial_reference_audit_v1.json"),
        ("original_validator", "code/scripts/validate_spatial_reference_audit_v1.py"),
    ):
        binding = _mapping(source.get(key), name=key)
        _exact_keys(binding, {"path", "sha256"}, name=key)
        if binding.get("path") != expected_path:
            raise ValueError(f"{key} path drifted")
        _sha256(binding.get("sha256"), name=f"{key} SHA256")

    parent = _mapping(config["parent_v2_1_archive"], name="parent archive")
    _exact_keys(
        parent,
        {
            "path",
            "sha256",
            "size_bytes",
            "source_git_commit",
            "required_outcome",
            "native_generation_repeat_count",
        },
        name="parent archive",
    )
    _canonical_absolute_path(parent.get("path"), name="parent archive path")
    _sha256(parent.get("sha256"), name="parent archive SHA256")
    _positive_int(parent.get("size_bytes"), name="parent archive size")
    _full_git_sha(parent.get("source_git_commit"), name="parent source commit")
    if (
        parent.get("required_outcome") != "NO_GO_V2_1_FULL_45_SUBSTRATE"
        or parent.get("native_generation_repeat_count") != 2
    ):
        raise ValueError("parent evidence contract drifted")

    evidence = _mapping(config["pre_repair_evidence"], name="pre-repair evidence")
    _exact_keys(
        evidence,
        {
            "audit_root",
            "root_tree_inventory",
            "packaging_tree_inventory",
            "global_attempt_ledger",
            "original_formal_log",
            "original_formal_exit",
            "original_failure",
            "profile_terminals",
        },
        name="pre-repair evidence",
    )
    audit_root = _canonical_absolute_path(
        evidence.get("audit_root"), name="audit root"
    )
    inventory_expectations = (
        (
            "root_tree_inventory",
            False,
            70,
            "spatial_reference_audit_v1_artifact._tree_inventory_sha256_over_"
            "root_regular_files",
        ),
        (
            "packaging_tree_inventory",
            True,
            71,
            "spatial_reference_audit_v1_artifact._tree_inventory_sha256_over_"
            "root_regular_files_plus_reserved_global_attempt_ledger",
        ),
    )
    for key, includes_ledger, count, algorithm in inventory_expectations:
        binding = _mapping(evidence.get(key), name=key)
        _exact_keys(
            binding,
            {
                "algorithm",
                "includes_reserved_global_attempt_ledger",
                "file_count",
                "sha256",
            },
            name=key,
        )
        if (
            binding.get("algorithm") != algorithm
            or binding.get("includes_reserved_global_attempt_ledger")
            is not includes_ledger
            or binding.get("file_count") != count
        ):
            raise ValueError(f"{key} contract drifted")
        _sha256(binding.get("sha256"), name=f"{key} SHA256")
    _validate_bound_file_config(
        evidence.get("global_attempt_ledger"), name="global attempt ledger"
    )
    _validate_bound_file_config(
        evidence.get("original_formal_log"), name="original formal log"
    )
    exit_binding = _validate_bound_file_config(
        evidence.get("original_formal_exit"),
        name="original formal exit",
        extra_keys={"utf8_content"},
    )
    if exit_binding.get("utf8_content") != "1\n":
        raise ValueError("original failed validator exit binding drifted")
    failure = _mapping(evidence.get("original_failure"), name="original failure")
    _exact_keys(
        failure,
        {"stage", "exception_type", "message"},
        name="original failure",
    )
    if dict(failure) != ORIGINAL_FAILURE:
        raise ValueError("original validator failure binding drifted")
    terminals = evidence.get("profile_terminals")
    if not isinstance(terminals, list) or len(terminals) != len(EXPECTED_PROFILE_IDS):
        raise ValueError("profile terminal binding inventory drifted")
    for position, binding_value in enumerate(terminals):
        binding = _validate_bound_file_config(
            binding_value,
            name="profile terminal",
            extra_keys={"profile_id"},
        )
        expected_profile_id = EXPECTED_PROFILE_IDS[position]
        expected_path = (
            audit_root
            / "profiles"
            / f"{position:03d}-{expected_profile_id}"
            / "terminal.json"
        )
        if (
            binding.get("profile_id") != expected_profile_id
            or Path(str(binding["path"])) != expected_path
        ):
            raise ValueError("profile terminal binding order or path drifted")

    contract = _mapping(config["repair_contract"], name="repair contract")
    _exact_keys(
        contract,
        {
            "validator_only_defects",
            "observed_pre_repair_effective_visual_tokens_per_image",
            "observed_effective_visual_token_set_used_as_acceptance_whitelist",
            "dynamic_grid_formula",
            "teacher_extended_prompt_aligned_inputs",
            "parent_native_generation_shape_keys",
            "all_shape_node_parent_binding_keys",
            "generation_shape_node_count",
            "teacher_shape_node_count",
            "total_shape_node_count",
            "generation_requires_full_parent_native_shape_match",
            "teacher_requires_parent_grid_and_effective_match",
            "original_validate_profile_required",
            "original_validate_attempt_ledger_required",
            "original_aggregate_profiles_required",
        },
        name="repair contract",
    )
    if contract.get("validator_only_defects") != [
        (
            "observed_original_failure_validator_hard_coded_2560_effective_visual_"
            "tokens_per_image_instead_of_validating_realized_grid_accounting"
        ),
        (
            "latent_read_only_dry_run_finding_validator_expected_input_ids_in_"
            "teacher_prompt_alignment_instead_of_the_runtime_emitted_mm_token_type_ids"
        ),
    ]:
        raise ValueError("validator-only defect inventory drifted")
    observed = contract.get(
        "observed_pre_repair_effective_visual_tokens_per_image"
    )
    if (
        observed != [2516, 2560, 2584]
        or contract.get(
            "observed_effective_visual_token_set_used_as_acceptance_whitelist"
        )
        is not False
        or contract.get("dynamic_grid_formula") != GRID_FORMULA
        or contract.get("teacher_extended_prompt_aligned_inputs")
        != list(EXPECTED_ALIGNED_INPUTS)
        or contract.get("parent_native_generation_shape_keys")
        != list(PARENT_SHAPE_KEYS)
        or contract.get("all_shape_node_parent_binding_keys")
        != list(PARENT_GRID_BINDING_KEYS)
        or contract.get("generation_shape_node_count") != 60
        or contract.get("teacher_shape_node_count") != 120
        or contract.get("total_shape_node_count") != 180
    ):
        raise ValueError("validation-repair shape contract drifted")
    for key in (
        "generation_requires_full_parent_native_shape_match",
        "teacher_requires_parent_grid_and_effective_match",
        "original_validate_profile_required",
        "original_validate_attempt_ledger_required",
        "original_aggregate_profiles_required",
    ):
        if contract.get(key) is not True:
            raise ValueError(f"repair contract must require {key}")

    outputs = _mapping(config["outputs"], name="repair outputs")
    _exact_keys(
        outputs,
        {"summary", "new_outputs_only", "write_only_after_all_validation_succeeds"},
        name="repair outputs",
    )
    summary_path = _canonical_absolute_path(outputs.get("summary"), name="summary")
    if (
        summary_path != audit_root / "summary.json"
        or outputs.get("new_outputs_only") is not True
        or outputs.get("write_only_after_all_validation_succeeds") is not True
    ):
        raise ValueError("validation-repair output contract drifted")

    return config


def validate_dynamic_shape_metadata(
    value: Mapping[str, Any],
    *,
    name: str,
    require_aligned_inputs: bool,
    expected_aligned_inputs: Sequence[str] = EXPECTED_ALIGNED_INPUTS,
) -> dict[str, Any]:
    """Validate dynamic-grid accounting without treating observed grids as a whitelist."""

    image_count = value.get("image_count")
    grids = value.get("image_grid_thw")
    effective = value.get("effective_visual_tokens")
    text_tokens = value.get("policy_visible_text_tokens")
    prompt_tokens = value.get("prompt_input_tokens")
    if type(image_count) is not int or image_count <= 0:
        raise ValueError(f"{name} image_count drifted")
    if not isinstance(grids, list) or len(grids) != image_count:
        raise ValueError(f"{name} image-grid inventory drifted")
    derived_effective = 0
    for grid in grids:
        if (
            not isinstance(grid, list)
            or len(grid) != 3
            or any(type(item) is not int or item <= 0 for item in grid)
        ):
            raise ValueError(f"{name} image-grid shape drifted")
        temporal, height, width = grid
        if height % 2 or width % 2:
            raise ValueError(f"{name} image-grid merge divisibility drifted")
        derived_effective += temporal * height * width // 4
    if (
        type(effective) is not int
        or effective <= 0
        or effective != derived_effective
        or type(text_tokens) is not int
        or text_tokens <= 0
        or type(prompt_tokens) is not int
        or prompt_tokens != effective + text_tokens
    ):
        raise ValueError(f"{name} prompt/visual token accounting drifted")
    result = {
        "image_count": image_count,
        "image_grid_thw": grids,
        "effective_visual_tokens": effective,
        "policy_visible_text_tokens": text_tokens,
        "prompt_input_tokens": prompt_tokens,
    }
    if require_aligned_inputs:
        aligned = value.get("extended_prompt_aligned_inputs")
        if aligned != list(expected_aligned_inputs):
            raise ValueError(f"{name} prompt-aligned input inventory drifted")
        result["extended_prompt_aligned_inputs"] = aligned
    return result


def _regular_file_bytes(path: Path, *, name: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    return path.read_bytes()


def _validate_file_binding(path: Path, binding: Mapping[str, Any], *, name: str) -> bytes:
    if path != Path(str(binding.get("path"))):
        raise ValueError(f"{name} canonical path drifted")
    payload = _regular_file_bytes(path, name=name)
    if (
        len(payload) != binding.get("size_bytes")
        or hashlib.sha256(payload).hexdigest() != binding.get("sha256")
    ):
        raise ValueError(f"{name} bytes drifted")
    return payload


def _validate_original_failure(
    *,
    log_payload: bytes,
    exit_payload: bytes,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    failure = dict(_mapping(evidence["original_failure"], name="original failure"))
    exit_binding = _mapping(
        evidence["original_formal_exit"], name="original formal exit"
    )
    try:
        observed_exit = exit_payload.decode("utf-8")
        log_lines = log_payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("original formal failure evidence is not UTF-8") from error
    marker = f'{failure["exception_type"]}: {failure["message"]}'
    if observed_exit != exit_binding["utf8_content"] or not log_lines:
        raise ValueError("original formal failure exit drifted")
    if log_lines[-1] != marker:
        raise ValueError("original formal failure traceback drifted")
    return {
        **failure,
        "formal_log_sha256": evidence["original_formal_log"]["sha256"],
        "formal_exit_sha256": exit_binding["sha256"],
        "formal_exit_utf8_content": observed_exit,
    }


def _validate_pre_repair_evidence(
    config: Mapping[str, Any],
) -> tuple[dict[str, bytes], bytes, dict[str, Any]]:
    evidence = _mapping(config["pre_repair_evidence"], name="pre-repair evidence")
    root = Path(str(evidence["audit_root"]))
    summary_path = Path(str(config["outputs"]["summary"]))
    if summary_path.exists():
        raise FileExistsError("canonical validation-repair summary already exists")
    root_files = _collect_tree_files(root)
    root_binding = _mapping(
        evidence["root_tree_inventory"], name="root tree inventory"
    )
    if (
        len(root_files) != root_binding["file_count"]
        or _tree_inventory_sha256(root_files) != root_binding["sha256"]
    ):
        raise ValueError("pre-repair root tree inventory drifted")
    ledger_binding = _mapping(
        evidence["global_attempt_ledger"], name="global attempt ledger"
    )
    ledger_path = Path(str(ledger_binding["path"]))
    ledger_payload = _validate_file_binding(
        ledger_path, ledger_binding, name="global attempt ledger"
    )
    packaging_files = collect_spatial_reference_audit_files(root, ledger_path)
    packaging_binding = _mapping(
        evidence["packaging_tree_inventory"], name="packaging tree inventory"
    )
    if (
        len(packaging_files) != packaging_binding["file_count"]
        or _tree_inventory_sha256(packaging_files) != packaging_binding["sha256"]
    ):
        raise ValueError("pre-repair packaging tree inventory drifted")
    log_binding = _mapping(
        evidence["original_formal_log"], name="original formal log"
    )
    exit_binding = _mapping(
        evidence["original_formal_exit"], name="original formal exit"
    )
    log_payload = _validate_file_binding(
        Path(str(log_binding["path"])), log_binding, name="original formal log"
    )
    exit_payload = _validate_file_binding(
        Path(str(exit_binding["path"])), exit_binding, name="original formal exit"
    )
    failure = _validate_original_failure(
        log_payload=log_payload,
        exit_payload=exit_payload,
        evidence=evidence,
    )
    return root_files, ledger_payload, failure


def validate_frozen_raw_archive_absent(original_config: Mapping[str, Any]) -> Path:
    archive_path = Path(
        str(original_config["artifact_packaging"]["canonical_local_archive_path"])
    )
    if archive_path.exists() or archive_path.is_symlink():
        raise FileExistsError("canonical spatial-reference raw archive already exists")
    return archive_path


def validate_original_audit_committed_source_blobs(
    *,
    repository_root: Path,
    original_config_path: Path,
    original_config: Mapping[str, Any],
    original_audit_git_commit: str,
) -> list[dict[str, str]]:
    expected_validator_path = (
        repository_root / "code/scripts/validate_spatial_reference_audit_v1.py"
    ).resolve()
    module_file = getattr(original_validator, "__file__", None)
    if not isinstance(module_file, str) or Path(module_file).resolve() != (
        expected_validator_path
    ):
        raise ValueError("imported original validator module path drifted")
    source_inventory = _mapping(
        original_config.get("source_inventory"), name="original source inventory"
    )
    relative_config_path = original_config_path.relative_to(repository_root).as_posix()
    paths = (relative_config_path, *source_inventory.keys())
    records = validate_committed_source_blobs(
        repository_root=repository_root,
        git_commit=original_audit_git_commit,
        paths=paths,
    )
    expected_sha256 = {
        relative_config_path: sha256_file(original_config_path),
        **{str(path): str(sha256) for path, sha256 in source_inventory.items()},
    }
    if len(records) != 34 or {
        record["path"]: record["sha256"] for record in records
    } != expected_sha256:
        raise ValueError("original audit committed source blob inventory drifted")
    return records


def write_canonical_summary(
    *,
    audit_root: Path,
    summary_path: Path,
    summary: Mapping[str, Any],
) -> None:
    if summary_path != audit_root / "summary.json":
        raise ValueError("validation repair may write only canonical summary.json")
    original_validator._write_json_exclusive(summary_path, summary)


def validate_post_write_state(
    *,
    audit_root: Path,
    summary_path: Path,
    expected_summary: Mapping[str, Any],
    pre_repair_files: Mapping[str, bytes],
    ledger_path: Path,
    ledger_payload: bytes,
    original_config: Mapping[str, Any],
) -> dict[str, int]:
    expected_summary_bytes = (
        json.dumps(
            dict(expected_summary),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if _regular_file_bytes(
        summary_path, name="canonical validation-repair summary"
    ) != expected_summary_bytes:
        raise ValueError("written canonical summary bytes drifted")
    written_summary = original_validator._load(summary_path)
    if written_summary != dict(expected_summary):
        raise ValueError("written canonical summary differs from validated payload")
    scientific_payload = dict(written_summary)
    observed_scientific_sha256 = scientific_payload.pop(
        "scientific_payload_sha256", None
    )
    expected_scientific_sha256 = hashlib.sha256(
        canonical_json_bytes(scientific_payload)
    ).hexdigest()
    if observed_scientific_sha256 != expected_scientific_sha256:
        raise ValueError("written canonical summary scientific hash drifted")

    post_root_files = _collect_tree_files(audit_root)
    expected_root_paths = set(pre_repair_files) | {"summary.json"}
    if (
        set(post_root_files) != expected_root_paths
        or any(
            post_root_files[path] != payload
            for path, payload in pre_repair_files.items()
        )
    ):
        raise ValueError("post-write audit root differs from old evidence plus summary")
    if _regular_file_bytes(
        ledger_path, name="global attempt ledger"
    ) != ledger_payload:
        raise ValueError("post-write global attempt ledger drifted")
    post_packaging_files = collect_spatial_reference_audit_files(
        audit_root, ledger_path
    )
    counts = {
        "root_file_count": len(post_root_files),
        "packaging_file_count": len(post_packaging_files),
    }
    recorded = expected_summary["validation_repair"]["post_write_file_counts"]
    if counts != {
        "root_file_count": recorded["observed_root_file_count"],
        "packaging_file_count": recorded["observed_packaging_file_count"],
    } or recorded != {
        "expected_root_file_count": len(pre_repair_files) + 1,
        "observed_root_file_count": len(pre_repair_files) + 1,
        "expected_packaging_file_count": len(pre_repair_files) + 2,
        "observed_packaging_file_count": len(pre_repair_files) + 2,
    }:
        raise ValueError("post-write evidence file counts drifted")
    validate_frozen_raw_archive_absent(original_config)
    return counts


def _parent_shapes(
    *,
    parent_archive: Path,
    config: Mapping[str, Any],
    state_indices: Sequence[int],
) -> tuple[dict[int, dict[str, Any]], str, list[int]]:
    parent_binding = _mapping(config["parent_v2_1_archive"], name="parent archive")
    _validate_file_binding(parent_archive, parent_binding, name="parent v2.1 archive")
    evidence = read_full_45_evidence_archive(
        parent_archive,
        expected_source_git_commit=str(parent_binding["source_git_commit"]),
    )
    if evidence.outcome != parent_binding["required_outcome"]:
        raise ValueError("validated parent v2.1 outcome drifted")
    result: dict[int, dict[str, Any]] = {}
    observed_per_image: set[int] = set()
    for index in state_indices:
        member = f"states/{index:03d}.json"
        try:
            state = json.loads(evidence.files[member])
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("validated parent native state is unavailable") from error
        if not isinstance(state, Mapping) or state.get("state", {}).get("index") != index:
            raise ValueError("validated parent native state identity drifted")
        generations = state.get("native_generations")
        if (
            not isinstance(generations, list)
            or len(generations) != parent_binding["native_generation_repeat_count"]
        ):
            raise ValueError("validated parent native generation inventory drifted")
        shapes = []
        for generation in generations:
            generation_value = _mapping(generation, name="parent native generation")
            metadata = _mapping(
                generation_value.get("metadata"), name="parent native metadata"
            )
            shape = validate_dynamic_shape_metadata(
                metadata,
                name="parent native generation",
                require_aligned_inputs=False,
            )
            shapes.append(shape)
            observed_per_image.update(
                temporal * height * width // 4
                for temporal, height, width in shape["image_grid_thw"]
            )
        if any(shape != shapes[0] for shape in shapes[1:]):
            raise ValueError("parent native generation repeat shapes drifted")
        result[index] = shapes[0]
    observed = sorted(observed_per_image)
    expected_observed = config["repair_contract"][
        "observed_pre_repair_effective_visual_tokens_per_image"
    ]
    if observed != expected_observed:
        raise ValueError("parent native observed visual-token evidence drifted")
    projection = [
        {"index": index, "shape": result[index]} for index in sorted(result)
    ]
    projection_sha256 = hashlib.sha256(canonical_json_bytes(projection)).hexdigest()
    return result, projection_sha256, observed


def validate_parent_bound_shape_nodes(
    profiles: Sequence[Mapping[str, Any]],
    *,
    parent_shapes: Mapping[int, Mapping[str, Any]],
    expected_aligned_inputs: Sequence[str],
    expected_generation_node_count: int,
    expected_teacher_node_count: int,
) -> dict[str, Any]:
    generation_count = 0
    teacher_count = 0
    observed_per_image: set[int] = set()
    for profile in profiles:
        records = profile.get("records")
        if not isinstance(records, list):
            raise ValueError("profile record inventory is invalid")
        for record_value in records:
            record = _mapping(record_value, name="profile record")
            index = record.get("index")
            if type(index) is not int or index not in parent_shapes:
                raise ValueError("profile record lacks a validated parent shape")
            expected = dict(parent_shapes[index])
            expected_grid = {key: expected[key] for key in PARENT_GRID_BINDING_KEYS}
            generations = record.get("generations")
            if not isinstance(generations, list):
                raise ValueError("profile generation inventory is invalid")
            for generation_value in generations:
                generation = _mapping(generation_value, name="generation")
                metadata = _mapping(
                    generation.get("metadata"), name="generation metadata"
                )
                shape = validate_dynamic_shape_metadata(
                    metadata,
                    name="generation",
                    require_aligned_inputs=False,
                )
                if shape != expected:
                    raise ValueError(
                        "generation full shape differs from validated parent native generation"
                    )
                generation_count += 1
                observed_per_image.update(
                    temporal * height * width // 4
                    for temporal, height, width in shape["image_grid_thw"]
                )
            teacher_groups = (
                record.get("shared_prefix_parent_pair_repeats"),
                record.get("full_parent_action_diagnostics"),
            )
            for group in teacher_groups:
                if not isinstance(group, list):
                    raise ValueError("profile teacher shape inventory is invalid")
                for teacher_value in group:
                    teacher = _mapping(teacher_value, name="teacher shape node")
                    shape = validate_dynamic_shape_metadata(
                        teacher,
                        name="teacher shape node",
                        require_aligned_inputs=True,
                        expected_aligned_inputs=expected_aligned_inputs,
                    )
                    observed_grid = {
                        key: shape[key] for key in PARENT_GRID_BINDING_KEYS
                    }
                    if observed_grid != expected_grid:
                        raise ValueError(
                            "teacher grid/effective shape differs from validated parent native generation"
                        )
                    teacher_count += 1
                    observed_per_image.update(
                        temporal * height * width // 4
                        for temporal, height, width in shape["image_grid_thw"]
                    )
    if (
        generation_count != expected_generation_node_count
        or teacher_count != expected_teacher_node_count
    ):
        raise ValueError("validation-repair shape-node denominator drifted")
    return {
        "generation_shape_node_count": generation_count,
        "teacher_shape_node_count": teacher_count,
        "total_shape_node_count": generation_count + teacher_count,
        "observed_effective_visual_tokens_per_image": sorted(observed_per_image),
    }


def _dynamic_validator_override(
    *,
    aligned: Sequence[str],
) -> Callable[..., dict[str, Any]]:
    def validate(
        value: Mapping[str, Any],
        *,
        name: str,
        require_aligned_inputs: bool,
    ) -> dict[str, Any]:
        return validate_dynamic_shape_metadata(
            value,
            name=name,
            require_aligned_inputs=require_aligned_inputs,
            expected_aligned_inputs=aligned,
        )

    return validate


@contextmanager
def _scoped_dynamic_validator_override(
    *,
    aligned: Sequence[str],
) -> Iterator[None]:
    # note (luojiaxuan): The completed terminals are immutable. The narrow
    # validator-only override exists only while the original validator traverses
    # those bytes, and is restored even when any downstream check fails.
    previous = original_validator._validate_shape_metadata
    original_validator._validate_shape_metadata = _dynamic_validator_override(
        aligned=aligned
    )
    try:
        yield
    finally:
        original_validator._validate_shape_metadata = previous


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repository_root = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    expected_config_path = (
        repository_root
        / "code/configs/spatial_reference_audit_v1_validation_repair_v1.json"
    )
    if config_path != expected_config_path:
        raise ValueError("validation-repair config path is not canonical")
    repair_config = load_repair_config(config_path)
    git_identity = dict(validate_clean_pushed_main(repository_root))
    repair_script_path = (
        repository_root
        / "code/scripts/validate_spatial_reference_audit_v1_validation_repair_v1.py"
    )
    source_records = validate_committed_source_blobs(
        repository_root=repository_root,
        git_commit=git_identity["commit"],
        paths=(
            config_path.relative_to(repository_root).as_posix(),
            repair_script_path.relative_to(repository_root).as_posix(),
        ),
    )

    source = _mapping(repair_config["frozen_source"], name="frozen source")
    original_config_binding = _mapping(
        source["original_config"], name="original config binding"
    )
    original_validator_binding = _mapping(
        source["original_validator"], name="original validator binding"
    )
    original_config_path = repository_root / str(original_config_binding["path"])
    original_validator_path = repository_root / str(
        original_validator_binding["path"]
    )
    if (
        sha256_file(original_config_path) != original_config_binding["sha256"]
        or sha256_file(original_validator_path) != original_validator_binding["sha256"]
    ):
        raise ValueError("frozen original validator source drifted")
    original_config = load_and_validate_config(original_config_path)
    validate_repository_inputs(repository_root, original_config)
    validate_frozen_raw_archive_absent(original_config)
    original_audit_git_commit = repair_config["repair_scope"][
        "original_audit_source_git_commit"
    ]
    if original_audit_git_commit != (
        "c093bd8f92ab97427acb427bd2d66fb6b20b556a"
    ):
        raise ValueError("original audit source commit drifted")
    original_audit_source_records = validate_original_audit_committed_source_blobs(
        repository_root=repository_root,
        original_config_path=original_config_path,
        original_config=original_config,
        original_audit_git_commit=original_audit_git_commit,
    )

    pre_repair_files, ledger_payload, original_failure = (
        _validate_pre_repair_evidence(repair_config)
    )
    evidence_binding = _mapping(
        repair_config["pre_repair_evidence"], name="pre-repair evidence"
    )
    terminal_bindings = list(evidence_binding["profile_terminals"])
    profile_paths = [Path(str(item["path"])) for item in terminal_bindings]
    for path, binding in zip(profile_paths, terminal_bindings, strict=True):
        _validate_file_binding(
            path, _mapping(binding, name="profile terminal"), name="profile terminal"
        )
    raw_profiles = [original_validator._load(path) for path in profile_paths]
    if {
        profile.get("source_git_commit") for profile in raw_profiles
    } != {repair_config["repair_scope"]["original_audit_source_git_commit"]}:
        raise ValueError("profile source commit differs from the frozen audit source")

    parent_archive = Path(str(repair_config["parent_v2_1_archive"]["path"]))
    state_indices = original_config["mismatch_denominator"]["state_indices"]
    parent_shapes, parent_shape_sha256, parent_observed = _parent_shapes(
        parent_archive=parent_archive,
        config=repair_config,
        state_indices=state_indices,
    )
    fixture_path = (
        repository_root
        / original_config["inputs"]["parent_mismatch_fixture"]["path"]
    )
    fixture = original_validator._fixture_by_index(fixture_path)
    parent_mismatches = load_parent_mismatches(
        raw_archive=parent_archive,
        fixture_path=fixture_path,
        config=original_config,
    )
    parent_by_index = {item.index: item for item in parent_mismatches}
    repair_contract = _mapping(
        repair_config["repair_contract"], name="repair contract"
    )
    aligned = list(repair_contract["teacher_extended_prompt_aligned_inputs"])

    with _scoped_dynamic_validator_override(aligned=aligned):
        validated_profiles = [
            original_validator.validate_profile(
                profile,
                config=original_config,
                repository_root=repository_root,
                fixture_by_index=fixture,
                parent_by_index=parent_by_index,
            )
            for profile in raw_profiles
        ]
        ledger = original_validator._validate_attempt_ledger(
            config=original_config,
            profiles=validated_profiles,
            repository_root=repository_root,
            parent_by_index=parent_by_index,
        )
        original_summary = original_validator.aggregate_profiles(
            repository_root=repository_root,
            config=original_config,
            profiles=validated_profiles,
            attempt_ledger=ledger,
        )

    shape_validation = validate_parent_bound_shape_nodes(
        validated_profiles,
        parent_shapes=parent_shapes,
        expected_aligned_inputs=aligned,
        expected_generation_node_count=repair_contract["generation_shape_node_count"],
        expected_teacher_node_count=repair_contract["teacher_shape_node_count"],
    )
    if (
        shape_validation["total_shape_node_count"]
        != repair_contract["total_shape_node_count"]
        or shape_validation["observed_effective_visual_tokens_per_image"]
        != parent_observed
    ):
        raise ValueError("validated observed shape evidence drifted")
    current_git_identity = dict(validate_clean_pushed_main(repository_root))
    if current_git_identity != git_identity:
        raise ValueError("repair Git identity changed during offline validation")
    current_root_files = _collect_tree_files(
        Path(str(evidence_binding["audit_root"]))
    )
    if current_root_files != pre_repair_files:
        raise ValueError("offline validation mutated the frozen profile tree")
    if _regular_file_bytes(
        Path(str(evidence_binding["global_attempt_ledger"]["path"])),
        name="global attempt ledger",
    ) != ledger_payload:
        raise ValueError("offline validation mutated the frozen attempt ledger")

    original_scientific_sha256 = original_summary.pop("scientific_payload_sha256")
    repair_config_sha256 = sha256_file(config_path)
    repair_script_sha256 = sha256_file(repair_script_path)
    repaired_summary = dict(original_summary)
    repaired_summary["status"] = STATUS
    repaired_summary["validation_repair"] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "repair_git_identity": git_identity,
        "repair_source_blobs": source_records,
        "repair_config_path": str(config_path),
        "repair_config_sha256": repair_config_sha256,
        "repair_validator_path": str(repair_script_path),
        "repair_validator_sha256": repair_script_sha256,
        "original_audit_source_git_commit": repair_config["repair_scope"][
            "original_audit_source_git_commit"
        ],
        "original_audit_committed_source_blobs": original_audit_source_records,
        "original_config_sha256": original_config_binding["sha256"],
        "original_validator_sha256": original_validator_binding["sha256"],
        "post_hoc_validation_repair": True,
        "original_failure": original_failure,
        "original_formal_log": dict(evidence_binding["original_formal_log"]),
        "original_formal_exit": dict(evidence_binding["original_formal_exit"]),
        "pre_repair_root_tree_inventory": dict(
            evidence_binding["root_tree_inventory"]
        ),
        "pre_repair_packaging_tree_inventory": dict(
            evidence_binding["packaging_tree_inventory"]
        ),
        "pre_repair_global_attempt_ledger": dict(
            evidence_binding["global_attempt_ledger"]
        ),
        "pre_repair_profile_terminals": [
            dict(item) for item in evidence_binding["profile_terminals"]
        ],
        "parent_v2_1_archive": dict(repair_config["parent_v2_1_archive"]),
        "parent_native_generation_shape_projection_sha256": parent_shape_sha256,
        "shape_validation": shape_validation,
        "original_validator_functions_reused": [
            "validate_profile",
            "_validate_attempt_ledger",
            "aggregate_profiles",
        ],
        "validator_only_defects": list(
            repair_contract["validator_only_defects"]
        ),
        "validator_only_overrides": {
            "dynamic_grid_formula": GRID_FORMULA,
            "observed_effective_visual_token_set_used_as_acceptance_whitelist": False,
            "observed_effective_visual_tokens_per_image": parent_observed,
            "teacher_extended_prompt_aligned_inputs": aligned,
        },
        "recomputed_original_validator_aggregate_scientific_payload_sha256": (
            original_scientific_sha256
        ),
        "added_operation_counts": {
            "generation_calls": 0,
            "teacher_forwards": 0,
            "confirm_state_accesses": 0,
            "restoration_coalition_constructions": 0,
            "gate_training_examples": 0,
        },
        "post_write_file_counts": {
            "expected_root_file_count": len(pre_repair_files) + 1,
            "observed_root_file_count": len(pre_repair_files) + 1,
            "expected_packaging_file_count": len(pre_repair_files) + 2,
            "observed_packaging_file_count": len(pre_repair_files) + 2,
        },
        "claim_boundary": {
            "processor_loads_performed": 0,
            "model_loads_performed": 0,
            "policy_or_model_forward_performed": False,
            "profile_reexecution_performed": False,
            "profile_or_ledger_mutation_performed": False,
            "confirm_policy_output_accessed": False,
            "restoration_work_performed": False,
            "gate_training_performed": False,
            "parent_v2_1_outcome_unchanged": "NO_GO_V2_1_FULL_45_SUBSTRATE",
        },
    }
    repaired_summary["scientific_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(repaired_summary)
    ).hexdigest()

    # note (luojiaxuan): This is the final fail-closed gate before the only
    # durable write. Recheck every Git, raw, and external binding together so a
    # concurrent mutation cannot create an irrecoverable partial repair.
    if dict(validate_clean_pushed_main(repository_root)) != git_identity:
        raise ValueError("repair Git identity drifted before summary write")
    if validate_committed_source_blobs(
        repository_root=repository_root,
        git_commit=git_identity["commit"],
        paths=(
            config_path.relative_to(repository_root).as_posix(),
            repair_script_path.relative_to(repository_root).as_posix(),
        ),
    ) != source_records:
        raise ValueError("repair committed source blobs drifted before summary write")
    if validate_original_audit_committed_source_blobs(
        repository_root=repository_root,
        original_config_path=original_config_path,
        original_config=original_config,
        original_audit_git_commit=original_audit_git_commit,
    ) != original_audit_source_records:
        raise ValueError("original audit source blobs drifted before summary write")
    audit_root = Path(str(evidence_binding["audit_root"]))
    if _collect_tree_files(audit_root) != pre_repair_files:
        raise ValueError("frozen audit root drifted before summary write")
    ledger_path = Path(str(evidence_binding["global_attempt_ledger"]["path"]))
    if _regular_file_bytes(
        ledger_path, name="global attempt ledger"
    ) != ledger_payload:
        raise ValueError("frozen attempt ledger drifted before summary write")
    _validate_file_binding(
        parent_archive,
        _mapping(
            repair_config["parent_v2_1_archive"], name="parent archive"
        ),
        name="parent v2.1 archive",
    )
    for key, label in (
        ("original_formal_log", "original formal log"),
        ("original_formal_exit", "original formal exit"),
    ):
        binding = _mapping(evidence_binding[key], name=label)
        _validate_file_binding(Path(str(binding["path"])), binding, name=label)
    validate_frozen_raw_archive_absent(original_config)
    summary_path = Path(str(repair_config["outputs"]["summary"]))
    write_canonical_summary(
        audit_root=audit_root,
        summary_path=summary_path,
        summary=repaired_summary,
    )
    validate_post_write_state(
        audit_root=audit_root,
        summary_path=summary_path,
        expected_summary=repaired_summary,
        pre_repair_files=pre_repair_files,
        ledger_path=ledger_path,
        ledger_payload=ledger_payload,
        original_config=original_config,
    )
    print(json.dumps(repaired_summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
