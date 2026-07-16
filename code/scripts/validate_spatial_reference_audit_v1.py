"""Independently validate and aggregate spatial_reference_audit_v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.data.guiodyssey_restoration_v2 import (
    canonical_json_bytes,
    sha256_file,
)
from causalcache.spatial_reference_audit_v1 import (
    EXPECTED_PROFILE_IDS,
    PROTOCOL_ID,
    ParentMismatch,
    canonical_profile_payload_sha256,
    coordinate_delta,
    first_divergent_index,
    load_and_validate_config,
    load_json_object,
    load_parent_mismatches,
    profile_by_id,
    profile_operation_counts,
    validate_repository_inputs,
)
from scripts.run_restoration_v2_1_interface_pilot import validate_clean_pushed_main


SCHEMA_VERSION = "1.0.0"
ATTEMPT_ID = "spatial-reference-audit-v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
CONTAINER_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
UTC_TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?Z"
)
OPERATION_COUNT_KEYS = (
    "generation_calls",
    "teacher_forwards",
    "confirm_state_accesses",
    "restoration_coalition_constructions",
    "gate_training_examples",
)
SHAPE_INVARIANT_KEYS = (
    "image_count",
    "image_grid_thw",
    "effective_visual_tokens",
    "policy_visible_text_tokens",
    "prompt_input_tokens",
    "extended_prompt_aligned_inputs",
)


def _load(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid spatial-reference JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError("spatial-reference JSON root must be an object")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{name} schema drifted: missing={missing}, extra={extra}")


def _timestamp(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical UTC timestamp")
    return value


def _expected_state_counts(profile: Any) -> dict[str, int]:
    return {
        "generation_calls": profile.generation_repeat_count,
        "teacher_forwards": (
            profile.shared_prefix_forward_repeat_count
            + profile.full_parent_action_forward_count
        ),
        "confirm_state_accesses": 0,
        "restoration_coalition_constructions": 0,
        "gate_training_examples": 0,
    }


def _validate_operation_counts(value: Any, *, name: str) -> dict[str, int]:
    mapped = _mapping(value, name=name)
    if set(mapped) != set(OPERATION_COUNT_KEYS) or any(
        type(mapped.get(key)) is not int or mapped[key] < 0
        for key in OPERATION_COUNT_KEYS
    ):
        raise ValueError(f"{name} schema or values drifted")
    return {key: int(mapped[key]) for key in OPERATION_COUNT_KEYS}


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or not path.parent.is_dir():
        raise FileExistsError("summary output must be new in an existing directory")
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_source_commit(repository_root: Path, source_commit: str) -> None:
    if GIT_SHA_PATTERN.fullmatch(source_commit) is None:
        raise ValueError("profile source commit is not a full Git SHA")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError("profile source commit is not an ancestor of current main")


def _scientific_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key
        not in {
            "started_at_utc",
            "ended_at_utc",
            "duration_seconds",
            "scientific_payload_sha256",
        }
    }


def _validate_score_summary(
    value: Mapping[str, Any],
    *,
    target_token_id: int,
) -> None:
    if value.get("target_token_id") != target_token_id:
        raise ValueError("teacher score summary target token drifted")
    for key in ("target_rank_strict", "top1_token_id", "top2_token_id"):
        observed = value.get(key)
        if type(observed) is not int or observed < 0:
            raise ValueError("teacher score summary integer field drifted")
    if value.get("target_rank_strict") < 1:
        raise ValueError("teacher target rank must be positive")
    target_logit = _finite(value.get("target_logit"), name="target_logit")
    _finite(value.get("target_log_probability"), name="target_log_probability")
    top1 = _finite(value.get("top1_logit"), name="top1_logit")
    top2 = _finite(value.get("top2_logit"), name="top2_logit")
    margin = _finite(
        value.get("top1_minus_top2_margin"),
        name="top1_minus_top2_margin",
    )
    if not math.isclose(margin, top1 - top2, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("teacher top-1/top-2 margin is internally inconsistent")
    if not math.isfinite(target_logit):
        raise ValueError("teacher target logit is non-finite")


def _validate_shape_metadata(
    value: Mapping[str, Any],
    *,
    name: str,
    require_aligned_inputs: bool,
) -> dict[str, Any]:
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
        merged = temporal * height * width // 4
        if merged != 2560:
            raise ValueError(f"{name} per-image effective visual token count drifted")
        derived_effective += merged
    if (
        effective != derived_effective
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
        if aligned != ["attention_mask", "input_ids"]:
            raise ValueError(f"{name} prompt-aligned input inventory drifted")
        result["extended_prompt_aligned_inputs"] = aligned
    return result


def _validate_generation(value: Mapping[str, Any], *, repeat_index: int) -> dict[str, Any]:
    if value.get("repeat_index") != repeat_index:
        raise ValueError("generation repeat index drifted")
    output_text = value.get("output_text")
    token_ids = value.get("generated_token_ids")
    metadata = _mapping(value.get("metadata"), name="generation metadata")
    if not isinstance(output_text, str) or not output_text:
        raise ValueError("generation output text is invalid")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
    ):
        raise ValueError("generation actual token ids are invalid")
    if metadata.get("generated_token_ids") != token_ids:
        raise ValueError("generation metadata differs from actual token ids")
    digest = hashlib.sha256(canonical_json_bytes(token_ids)).hexdigest()
    if metadata.get("generated_token_ids_sha256") != digest:
        raise ValueError("generation token SHA256 drifted")
    if metadata.get("generated_tokens") != len(token_ids):
        raise ValueError("generation token count drifted")
    reconstructed = metadata.get("decoded_output_retokenized_ids")
    if not isinstance(reconstructed, list) or any(
        type(token_id) is not int or token_id < 0 for token_id in reconstructed
    ):
        raise ValueError("decoded-output token reconstruction is invalid")
    if metadata.get("decoded_output_retokenization_matches_actual_ids") is not (
        reconstructed == token_ids
    ):
        raise ValueError("decoded-output reconstruction flag drifted")
    if (
        metadata.get("do_sample") is not False
        or metadata.get("num_beams") != 1
        or metadata.get("num_return_sequences") != 1
        or metadata.get("return_dict_in_generate") is not False
        or metadata.get("full_logit_tensor_host_transfers") != 0
    ):
        raise ValueError("generation execution semantics drifted")
    canonical_action = value.get("canonical_action")
    parse_type = value.get("parse_error_type")
    parse_message = value.get("parse_error_message")
    if canonical_action is None:
        if not isinstance(parse_type, str) or not isinstance(parse_message, str):
            raise ValueError("parse failure lacks an explicit error")
    elif not isinstance(canonical_action, Mapping) or parse_type is not None or parse_message is not None:
        raise ValueError("parsed generation status is inconsistent")
    return _validate_shape_metadata(
        metadata,
        name="generation",
        require_aligned_inputs=False,
    )


def _validate_parent_retokenization(record: Mapping[str, Any]) -> tuple[list[list[int]], int]:
    parent = _mapping(record.get("parent_retokenization"), name="parent retokenization")
    rows = parent.get("token_ids")
    hashes = record.get("parent_generated_token_ids_sha256")
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or not isinstance(hashes, list)
        or len(hashes) != 2
    ):
        raise ValueError("parent token rows or hashes drifted")
    for row, expected in zip(rows, hashes, strict=True):
        if not isinstance(row, list) or any(
            type(token_id) is not int or token_id < 0 for token_id in row
        ):
            raise ValueError("parent retokenized IDs are invalid")
        if hashlib.sha256(canonical_json_bytes(row)).hexdigest() != expected:
            raise ValueError("parent retokenized IDs differ from the frozen hash")
    divergence = first_divergent_index(rows[0], rows[1])
    if divergence is None:
        raise ValueError("parent token rows unexpectedly agree")
    expected_ids = [
        rows[0][divergence] if divergence < len(rows[0]) else None,
        rows[1][divergence] if divergence < len(rows[1]) else None,
    ]
    if (
        parent.get("first_divergent_token_index") != divergence
        or parent.get("first_divergent_token_ids") != expected_ids
        or parent.get("token_hash_validation") is not True
    ):
        raise ValueError("parent first-divergence reduction drifted")
    return rows, divergence


def _validate_shared_repeat(
    value: Mapping[str, Any],
    *,
    repeat_index: int,
    token_rows: Sequence[Sequence[int]],
    divergence: int,
) -> dict[str, Any]:
    candidate_ids = [token_rows[0][divergence], token_rows[1][divergence]]
    if (
        value.get("repeat_index") != repeat_index
        or value.get("teacher_token_ids") != list(token_rows)
        or value.get("first_divergent_token_index") != divergence
        or value.get("first_divergent_token_ids") != candidate_ids
        or value.get("shared_prefix_token_ids") != list(token_rows[0][:divergence])
        or value.get("full_vocabulary_logits_host_transfers") != 0
        or value.get("interpretation")
        != "single_shared_prefix_post_hoc_teacher_forced_competing_token_margin_not_original_generation_time_margin"
    ):
        raise ValueError("shared-prefix teacher diagnostic identity drifted")
    raw = value.get("raw_candidate_summaries")
    aligned = value.get("generation_aligned_candidate_summaries")
    if not isinstance(raw, list) or len(raw) != 2 or not isinstance(aligned, list) or len(aligned) != 2:
        raise ValueError("shared-prefix teacher candidate inventory drifted")
    for summary, token_id in zip(raw, candidate_ids, strict=True):
        _validate_score_summary(_mapping(summary, name="raw candidate"), target_token_id=token_id)
    for summary, token_id in zip(aligned, candidate_ids, strict=True):
        _validate_score_summary(
            _mapping(summary, name="aligned candidate"),
            target_token_id=token_id,
        )
    raw_pair = _finite(
        value.get("raw_first_minus_second_candidate_logit"),
        name="raw candidate pair margin",
    )
    aligned_pair = _finite(
        value.get("generation_aligned_first_minus_second_candidate_logit"),
        name="aligned candidate pair margin",
    )
    expected_raw = float(raw[0]["target_logit"]) - float(raw[1]["target_logit"])
    expected_aligned = float(aligned[0]["target_logit"]) - float(
        aligned[1]["target_logit"]
    )
    if not math.isclose(raw_pair, expected_raw, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("raw competing-token pair margin drifted")
    if not math.isclose(aligned_pair, expected_aligned, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("aligned competing-token pair margin drifted")
    return _validate_shape_metadata(
        value,
        name="shared-prefix repeat",
        require_aligned_inputs=True,
    )


def _validate_full_branch(
    value: Mapping[str, Any],
    *,
    action_index: int,
    parent_action: Mapping[str, Any],
    target_token_id: int,
    divergence: int,
    distance_action_tokens: int,
) -> dict[str, Any]:
    if (
        value.get("parent_action_index") != action_index
        or value.get("parent_action") != dict(parent_action)
        or value.get("divergence_index") != divergence
        or value.get("target_token_id") != target_token_id
        or value.get("distance_action_tokens") != distance_action_tokens
        or value.get("full_vocabulary_logits_host_transfers") != 0
        or value.get("interpretation")
        != "full_parent_action_branch_post_hoc_teacher_forced_diagnostic_with_shape_specific_suffix"
    ):
        raise ValueError("full-parent branch diagnostic identity drifted")
    _validate_score_summary(
        _mapping(value.get("raw"), name="full branch raw summary"),
        target_token_id=target_token_id,
    )
    _validate_score_summary(
        _mapping(
            value.get("generation_aligned_suppressed"),
            name="full branch aligned summary",
        ),
        target_token_id=target_token_id,
    )
    return _validate_shape_metadata(
        value,
        name="full-parent branch",
        require_aligned_inputs=True,
    )


def _validate_record(
    record: Mapping[str, Any],
    *,
    profile: Any,
    fixture: Mapping[str, Any],
    parent: ParentMismatch,
) -> None:
    index = record.get("index")
    if (
        index not in profile.state_indices
        or fixture.get("index") != index
        or parent.index != index
        or record.get("role") != fixture.get("role")
        or record.get("role") != parent.role
        or record.get("state_id") != fixture.get("state_id")
        or record.get("state_id") != parent.state_id
        or record.get("parent_actions") != fixture.get("actions")
        or record.get("parent_actions") != [dict(item) for item in parent.actions]
        or record.get("parent_output_texts") != list(parent.output_texts)
        or record.get("parent_generated_token_ids_sha256")
        != list(parent.generated_token_ids_sha256)
        or record.get("parent_state_member_sha256")
        != parent.state_member_sha256
        or record.get("coordinate_delta") != coordinate_delta(parent)
    ):
        raise ValueError("profile record differs from independently rederived parent evidence")
    state_member_sha = record.get("parent_state_member_sha256")
    if not isinstance(state_member_sha, str) or SHA256_PATTERN.fullmatch(state_member_sha) is None:
        raise ValueError("parent state-member SHA256 is invalid")
    token_rows, divergence = _validate_parent_retokenization(record)
    generations = record.get("generations")
    if not isinstance(generations, list) or len(generations) != profile.generation_repeat_count:
        raise ValueError("profile generation repeat count drifted")
    generation_shapes = []
    for repeat_index, generation in enumerate(generations, start=1):
        generation_shapes.append(
            _validate_generation(
                _mapping(generation, name="generation"),
                repeat_index=repeat_index,
            )
        )
    if any(shape != generation_shapes[0] for shape in generation_shapes[1:]):
        raise ValueError("generation repeat base input shapes drifted")
    parsed = [item for item in generations if item.get("canonical_action") is not None]
    expected_token_stable = (
        len(parsed) == profile.generation_repeat_count
        and len({tuple(item["generated_token_ids"]) for item in parsed}) == 1
    )
    expected_action_stable = (
        len(parsed) == profile.generation_repeat_count
        and len({json.dumps(item["canonical_action"], sort_keys=True) for item in parsed}) == 1
    )
    if (
        record.get("exact_generated_token_sequence_stable") is not expected_token_stable
        or record.get("exact_canonical_action_stable") is not expected_action_stable
    ):
        raise ValueError("profile stability metric differs from actual generated IDs")
    shared = record.get("shared_prefix_parent_pair_repeats")
    if not isinstance(shared, list) or len(shared) != profile.shared_prefix_forward_repeat_count:
        raise ValueError("shared-prefix repeat count drifted")
    shared_shapes = []
    for repeat_index, item in enumerate(shared, start=1):
        shared_shapes.append(
            _validate_shared_repeat(
                _mapping(item, name="shared-prefix repeat"),
                repeat_index=repeat_index,
                token_rows=token_rows,
                divergence=divergence,
            )
        )
    if any(shape != shared_shapes[0] for shape in shared_shapes[1:]):
        raise ValueError("shared-prefix repeat shapes are not exactly equal")
    branches = record.get("full_parent_action_diagnostics")
    if not isinstance(branches, list) or len(branches) != profile.full_parent_action_forward_count:
        raise ValueError("full-parent branch count drifted")
    parent_actions = record["parent_actions"]
    branch_shapes = []
    for action_index, branch in enumerate(branches):
        branch_shapes.append(
            _validate_full_branch(
                _mapping(branch, name="full-parent branch"),
                action_index=action_index,
                parent_action=parent_actions[action_index],
                target_token_id=token_rows[action_index][divergence],
                divergence=divergence,
                distance_action_tokens=len(token_rows[action_index]),
            )
        )
    base_shape = {
        key: shared_shapes[0][key]
        for key in SHAPE_INVARIANT_KEYS
        if key != "extended_prompt_aligned_inputs"
    }
    if (
        generation_shapes[0] != base_shape
        or any(
            {
                key: shape[key]
                for key in SHAPE_INVARIANT_KEYS
                if key != "extended_prompt_aligned_inputs"
            }
            != base_shape
            for shape in branch_shapes
        )
    ):
        raise ValueError("generation/shared/full-branch base image or prompt shape drifted")
    expected_counts = _expected_state_counts(profile)
    if record.get("operation_counts") != expected_counts:
        raise ValueError("profile state operation counts drifted")


def _validate_runtime(
    value: Mapping[str, Any],
    *,
    profile: Any,
    config: Mapping[str, Any],
    repository_root: Path,
) -> None:
    snapshot = load_json_object(
        repository_root / config["inputs"]["model_snapshot_manifest"]["path"]
    )
    runtime_constraints = _mapping(
        config.get("runtime_constraints"),
        name="runtime constraints",
    )
    expected_environment_audit = {
        "audited_names": runtime_constraints[
            "audited_scientific_environment_variables"
        ],
        "present_names": [],
        "all_absent": True,
    }
    expected_dtype = "torch.bfloat16" if profile.dtype == "bfloat16" else "torch.float32"
    if (
        value.get("audit_profile_id") != profile.profile_id
        or value.get("dtype") != expected_dtype
        or value.get("device") != config["attempt_identity"]["canonical_device"]
        or value.get("model_repo") != snapshot.get("repo")
        or value.get("model_revision") != snapshot.get("revision")
        or value.get("snapshot_manifest_sha256")
        != config["inputs"]["model_snapshot_manifest"]["sha256"]
        or value.get("verified_model_file_count") != len(snapshot.get("files", []))
        or value.get("verified_model_total_bytes")
        != sum(int(item["size"]) for item in snapshot.get("files", []))
        or value.get("python_version") != runtime_constraints["python_version"]
        or value.get("torch_version") != runtime_constraints["torch_version"]
        or value.get("torch_cuda_version")
        != runtime_constraints["torch_cuda_version"]
        or value.get("cudnn_version") != runtime_constraints["cudnn_version"]
        or value.get("transformers_version")
        != runtime_constraints["transformers_version"]
        or value.get("frozen") is not True
        or value.get("single_device") is not True
        or value.get("seed") != profile.seed
        or value.get("deterministic_algorithms_requested") is not False
        or value.get("deterministic_algorithms_enabled") is not False
        or value.get("deterministic_warn_only_enabled") is not False
        or value.get("strict_cuda_determinism_claimed") is not False
        or value.get("scientific_environment_variables_set_by_runtime") is not False
        or value.get("scientific_environment_audit") != expected_environment_audit
        or value.get("instrumented_generation_output_scores") is not False
        or value.get("eager_numerical_controls_requested")
        is not profile.eager_numerical_controls
        or value.get("requested_attention_implementation")
        != profile.attention_implementation
    ):
        raise ValueError("runtime profile/model/numerical identity drifted")
    observed_attention = _mapping(
        value.get("observed_attention_implementation"),
        name="observed attention",
    )
    observed = [item for item in observed_attention.values() if item is not None]
    if not observed or any(not isinstance(item, str) for item in observed):
        raise ValueError("attention implementation was not observably resolved")
    if profile.attention_implementation == "eager":
        if any(item != "eager" for item in observed):
            raise ValueError("eager attention was not applied everywhere")
    elif any(item == "eager" for item in observed):
        raise ValueError("auto attention unexpectedly resolved to eager")
    if profile.eager_numerical_controls:
        if (
            value.get("cudnn_deterministic") is not True
            or value.get("cudnn_benchmark") is not False
            or value.get("cuda_matmul_allow_tf32") is not False
            or value.get("cudnn_allow_tf32") is not False
            or value.get("float32_matmul_precision") != "highest"
            or value.get("numerical_control_claim")
            != "eager_fixed_seed_tf32_disabled_numerical_control_not_strict_cuda_determinism"
        ):
            raise ValueError("eager numerical controls were not applied exactly")
    elif value.get("numerical_control_claim") != "legacy_auto_backend_control":
        raise ValueError("auto backend control claim drifted")


def _validate_invocation_argv(
    invocation: Any,
    *,
    repository_root: Path,
    config: Mapping[str, Any],
    profile_id: str,
    source_commit: str,
    host: Mapping[str, Any],
    execution_inputs: Mapping[str, Any],
) -> list[str]:
    flags = (
        "--repository-root",
        "--config",
        "--source-git-commit",
        "--parent-raw-archive",
        "--derived-artifact-root",
        "--model-dir",
        "--device",
        "--profile-id",
        "--host-alias",
        "--host-hostname",
        "--container-id",
        "--container-image-digest",
    )
    if (
        not isinstance(invocation, list)
        or any(not isinstance(item, str) for item in invocation)
        or len(invocation) != 1 + 2 * len(flags)
        or invocation[0]
        != str(repository_root / "code/scripts/run_spatial_reference_audit_v1.py")
    ):
        raise ValueError("profile invocation argv schema drifted")
    values: dict[str, str] = {}
    position = 1
    while position < len(invocation):
        flag = invocation[position]
        if flag not in flags or flag in values:
            raise ValueError("profile invocation argv flags drifted")
        values[flag] = invocation[position + 1]
        position += 2
    expected = {
        "--repository-root": str(repository_root),
        "--config": str(
            repository_root / "code/configs/spatial_reference_audit_v1.json"
        ),
        "--source-git-commit": source_commit,
        "--parent-raw-archive": str(execution_inputs["parent_raw_archive"]),
        "--derived-artifact-root": str(execution_inputs["derived_artifact_root"]),
        "--model-dir": str(execution_inputs["model_dir"]),
        "--device": config["attempt_identity"]["canonical_device"],
        "--profile-id": profile_id,
        "--host-alias": config["attempt_identity"]["canonical_host_alias"],
        "--host-hostname": config["attempt_identity"]["canonical_host_hostname"],
        "--container-id": str(host["container_id"]),
        "--container-image-digest": str(host["container_image_digest"]),
    }
    if values != expected:
        raise ValueError("profile invocation argv values drifted")
    return list(invocation)


def _validate_host(value: Mapping[str, Any], *, config: Mapping[str, Any]) -> None:
    attempt = config["attempt_identity"]
    runtime = config["runtime_constraints"]
    container_id = value.get("container_id")
    container_hostname = value.get("container_hostname")
    digest = value.get("container_image_digest")
    if (
        value.get("alias") != attempt["canonical_host_alias"]
        or value.get("hostname") != attempt["canonical_host_hostname"]
        or value.get("device") != attempt["canonical_device"]
        or not isinstance(container_id, str)
        or CONTAINER_ID_PATTERN.fullmatch(container_id) is None
        or not isinstance(container_hostname, str)
        or not container_id.startswith(container_hostname)
        or not isinstance(digest, str)
        or CONTAINER_DIGEST_PATTERN.fullmatch(digest) is None
        or digest != runtime["container_image_digest"]
        or value.get("visible_gpu_count") != 1
        or value.get("cuda_visible_ordinal") != 0
        or value.get("gpu_name") != "NVIDIA H200"
        or not isinstance(value.get("gpu_uuid"), str)
        or not str(value.get("gpu_uuid")).startswith("GPU-")
        or value.get("nvidia_driver_version") != runtime["nvidia_driver_version"]
        or value.get("software")
        != {
            "python_version": runtime["python_version"],
            "torch_version": runtime["torch_version"],
            "torch_cuda_version": runtime["torch_cuda_version"],
            "cudnn_version": runtime["cudnn_version"],
            "transformers_version": runtime["transformers_version"],
        }
        or value.get("scientific_environment_audit")
        != {
            "audited_names": runtime["audited_scientific_environment_variables"],
            "present_names": [],
            "all_absent": True,
        }
        or value.get("nvidia_smi_query")
        != (
            "nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version "
            "--format=csv,noheader,nounits"
        )
    ):
        raise ValueError("live host/container/GPU identity drifted")


def validate_profile(
    value: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    repository_root: Path,
    fixture_by_index: Mapping[int, Mapping[str, Any]],
    parent_by_index: Mapping[int, ParentMismatch],
) -> dict[str, Any]:
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("attempt_id") != ATTEMPT_ID
        or value.get("status") != "COMPLETED_SPATIAL_REFERENCE_AUDIT_PROFILE"
    ):
        raise ValueError("spatial-reference profile identity drifted")
    profile_value = _mapping(value.get("profile"), name="profile")
    profile = profile_by_id(config, str(profile_value.get("profile_id")))
    position = EXPECTED_PROFILE_IDS.index(profile.profile_id)
    expected_profile = {
        "profile_id": profile.profile_id,
        "dtype": profile.dtype,
        "attention_implementation": profile.attention_implementation,
        "deterministic_algorithms": profile.deterministic_algorithms,
        "eager_numerical_controls": profile.eager_numerical_controls,
        "seed": profile.seed,
        "state_indices": list(profile.state_indices),
        "generation_repeat_count": profile.generation_repeat_count,
        "shared_prefix_forward_repeat_count": profile.shared_prefix_forward_repeat_count,
        "full_parent_action_forward_count": profile.full_parent_action_forward_count,
    }
    if dict(profile_value) != expected_profile or value.get("profile_position") != position:
        raise ValueError("spatial-reference profile contract drifted")
    source_commit = value.get("source_git_commit")
    if not isinstance(source_commit, str):
        raise ValueError("profile source commit is invalid")
    _validate_source_commit(repository_root, source_commit)
    config_path = repository_root / "code/configs/spatial_reference_audit_v1.json"
    if (
        value.get("config_path") != str(config_path)
        or value.get("config_sha256") != sha256_file(config_path)
        or value.get("parent_raw_archive_sha256")
        != config["inputs"]["parent_raw_archive"]["sha256"]
        or value.get("derived_artifact_tree_sha256")
        != config["inputs"]["derived_artifact"]["artifact_tree_sha256"]
    ):
        raise ValueError("profile input provenance drifted")
    parent_path = Path(str(value.get("parent_raw_archive_path")))
    if not parent_path.is_file() or sha256_file(parent_path) != value["parent_raw_archive_sha256"]:
        raise ValueError("profile parent raw archive is unavailable or changed")
    attempt = config["attempt_identity"]
    profile_root = Path(attempt["canonical_persistent_output_dir"]) / "profiles" / (
        f"{position:03d}-{profile.profile_id}"
    )
    expected_layout = {
        "root": attempt["canonical_persistent_output_dir"],
        "global_ledger": attempt["canonical_global_attempt_ledger"],
        "profile_start": str(profile_root / "start.json"),
        "profile_terminal": str(profile_root / "terminal.json"),
    }
    if value.get("attempt_layout") != expected_layout:
        raise ValueError("profile canonical attempt layout drifted")
    invocation = value.get("invocation_argv")
    if (
        not isinstance(invocation, list)
        or any(not isinstance(item, str) for item in invocation)
        or "--profile-id" not in invocation
        or invocation[invocation.index("--profile-id") + 1] != profile.profile_id
        or "--output" in invocation
        or "--output-dir" in invocation
    ):
        raise ValueError("profile invocation argv drifted")
    if (
        value.get("confirm_policy_output_accessed") is not False
        or value.get("restoration_work_performed") is not False
        or value.get("gate_training_performed") is not False
    ):
        raise ValueError("profile crossed a prohibited scientific boundary")
    _validate_host(_mapping(value.get("host"), name="profile host"), config=config)
    _validate_runtime(
        _mapping(value.get("runtime_metadata"), name="runtime metadata"),
        profile=profile,
        config=config,
        repository_root=repository_root,
    )
    records = value.get("records")
    if not isinstance(records, list) or [item.get("index") for item in records] != list(
        profile.state_indices
    ):
        raise ValueError("profile record denominator or ordering drifted")
    for record in records:
        mapped = _mapping(record, name="profile record")
        _validate_record(
            mapped,
            profile=profile,
            fixture=fixture_by_index[int(mapped["index"])],
            parent=parent_by_index[int(mapped["index"])],
        )
    expected_counts = profile_operation_counts(profile)
    if value.get("operation_counts") != expected_counts:
        raise ValueError("profile aggregate operation counts drifted")
    metrics = _mapping(value.get("metrics"), name="metrics")
    token_count = sum(item["exact_generated_token_sequence_stable"] for item in records)
    action_count = sum(item["exact_canonical_action_stable"] for item in records)
    expected_metrics = {
        "state_count": len(records),
        "exact_generated_token_sequence_stable_count": token_count,
        "exact_canonical_action_stable_count": action_count,
        "exact_generated_token_sequence_stable_rate": token_count / len(records),
        "exact_canonical_action_stable_rate": action_count / len(records),
    }
    if dict(metrics) != expected_metrics:
        raise ValueError("profile aggregate metrics drifted")
    observed_hash = value.get("scientific_payload_sha256")
    expected_hash = canonical_profile_payload_sha256(_scientific_payload(value))
    if observed_hash != expected_hash:
        raise ValueError("profile scientific payload SHA256 drifted")
    return dict(value)


def _validate_attempt_ledger(
    *,
    config: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    repository_root: Path,
    parent_by_index: Mapping[int, ParentMismatch],
) -> dict[str, Any]:
    attempt = config["attempt_identity"]
    root = Path(attempt["canonical_persistent_output_dir"])
    ledger_path = Path(attempt["canonical_global_attempt_ledger"])
    if not root.is_dir() or not ledger_path.is_file() or ledger_path.is_symlink():
        raise ValueError("canonical attempt root or sibling ledger is missing")
    ledger = _load(ledger_path)
    _exact_keys(
        ledger,
        {
            "schema_version",
            "protocol_id",
            "attempt_id",
            "status",
            "canonical_attempt_root",
            "canonical_global_ledger",
            "source_git_commit",
            "config_sha256",
            "host",
            "execution_inputs",
            "profile_order",
            "next_profile_position",
            "operation_counts",
            "profiles",
            "created_at_utc",
            "updated_at_utc",
            "completed_at_utc",
            "retry_allowed",
            "deletion_allowed",
        },
        name="canonical terminal attempt ledger",
    )
    if (
        ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("protocol_id") != PROTOCOL_ID
        or ledger.get("attempt_id") != ATTEMPT_ID
        or ledger.get("status") != "COMPLETED_SPATIAL_REFERENCE_AUDIT_ATTEMPT"
        or ledger.get("canonical_attempt_root") != str(root)
        or ledger.get("canonical_global_ledger") != str(ledger_path)
        or ledger.get("profile_order") != list(EXPECTED_PROFILE_IDS)
        or ledger.get("next_profile_position") != len(EXPECTED_PROFILE_IDS)
        or ledger.get("operation_counts") != config["operation_limits"]
        or ledger.get("retry_allowed") is not False
        or ledger.get("deletion_allowed") is not False
    ):
        raise ValueError("canonical global attempt ledger terminal identity drifted")
    _timestamp(ledger.get("created_at_utc"), name="ledger created_at_utc")
    _timestamp(ledger.get("updated_at_utc"), name="ledger updated_at_utc")
    _timestamp(ledger.get("completed_at_utc"), name="ledger completed_at_utc")
    source_commits = {profile["source_git_commit"] for profile in profiles}
    config_hashes = {profile["config_sha256"] for profile in profiles}
    hosts = {json.dumps(profile["host"], sort_keys=True) for profile in profiles}
    if (
        source_commits != {ledger.get("source_git_commit")}
        or config_hashes != {ledger.get("config_sha256")}
        or hosts != {json.dumps(ledger.get("host"), sort_keys=True)}
    ):
        raise ValueError("global ledger differs from profile provenance")
    execution_inputs = _mapping(
        ledger.get("execution_inputs"),
        name="ledger execution_inputs",
    )
    if set(execution_inputs) != {
        "repository_root",
        "config_path",
        "parent_raw_archive",
        "derived_artifact_root",
        "model_dir",
    }:
        raise ValueError("ledger execution-input schema drifted")
    if (
        execution_inputs.get("repository_root") != str(repository_root)
        or execution_inputs.get("config_path")
        != str(repository_root / "code/configs/spatial_reference_audit_v1.json")
        or {profile.get("parent_raw_archive_path") for profile in profiles}
        != {execution_inputs.get("parent_raw_archive")}
        or {profile.get("derived_artifact_root") for profile in profiles}
        != {execution_inputs.get("derived_artifact_root")}
        or any(
            not Path(str(execution_inputs[key])).is_absolute()
            for key in execution_inputs
        )
    ):
        raise ValueError("ledger execution inputs differ from profile provenance")
    attempt_start_path = root / "start.json"
    if not attempt_start_path.is_file() or attempt_start_path.is_symlink():
        raise ValueError("canonical durable attempt start is missing")
    attempt_start = _load(attempt_start_path)
    _exact_keys(
        attempt_start,
        {
            "schema_version",
            "protocol_id",
            "attempt_id",
            "status",
            "source_git_commit",
            "config_sha256",
            "host",
            "execution_inputs",
            "profile_order",
            "first_invocation_argv",
            "started_at_utc",
        },
        name="attempt start",
    )
    if (
        attempt_start.get("schema_version") != SCHEMA_VERSION
        or attempt_start.get("protocol_id") != PROTOCOL_ID
        or attempt_start.get("attempt_id") != ATTEMPT_ID
        or attempt_start.get("status") != "STARTED_NO_RETRY_OR_DELETION"
        or attempt_start.get("source_git_commit") != ledger.get("source_git_commit")
        or attempt_start.get("config_sha256") != ledger.get("config_sha256")
        or attempt_start.get("host") != ledger.get("host")
        or attempt_start.get("execution_inputs") != dict(execution_inputs)
        or attempt_start.get("profile_order") != list(EXPECTED_PROFILE_IDS)
    ):
        raise ValueError("attempt start differs from the terminal attempt ledger")
    _timestamp(attempt_start.get("started_at_utc"), name="attempt started_at_utc")
    ledger_profiles = ledger.get("profiles")
    if not isinstance(ledger_profiles, list) or len(ledger_profiles) != len(profiles):
        raise ValueError("global ledger profile inventory drifted")
    for position, (ledger_profile, profile) in enumerate(
        zip(ledger_profiles, profiles, strict=True)
    ):
        spec = profile_by_id(config, profile["profile"]["profile_id"])
        profile_root = root / "profiles" / f"{position:03d}-{spec.profile_id}"
        expected_start = profile_root / "start.json"
        expected_terminal = profile_root / "terminal.json"
        claim_path = root / "profiles" / f".{position:03d}-{spec.profile_id}.claim.json"
        states = ledger_profile.get("states")
        _exact_keys(
            _mapping(ledger_profile, name="ledger profile"),
            {
                "position",
                "profile_id",
                "status",
                "operation_counts",
                "states",
                "start_path",
                "terminal_path",
                "invocation_argv",
                "state_indices",
                "started_at_utc",
                "ended_at_utc",
            },
            name="ledger completed profile",
        )
        if (
            ledger_profile.get("position") != position
            or ledger_profile.get("profile_id") != spec.profile_id
            or ledger_profile.get("status") != "COMPLETED"
            or ledger_profile.get("start_path") != str(expected_start)
            or ledger_profile.get("terminal_path") != str(expected_terminal)
            or ledger_profile.get("state_indices") != list(spec.state_indices)
            or ledger_profile.get("operation_counts") != profile_operation_counts(spec)
            or not isinstance(states, list)
            or len(states) != len(spec.state_indices)
            or _load(expected_terminal) != dict(profile)
        ):
            raise ValueError("global ledger completed profile record drifted")
        _timestamp(
            ledger_profile.get("started_at_utc"),
            name="ledger profile started_at_utc",
        )
        _timestamp(
            ledger_profile.get("ended_at_utc"),
            name="ledger profile ended_at_utc",
        )
        invocation = _validate_invocation_argv(
            profile.get("invocation_argv"),
            repository_root=repository_root,
            config=config,
            profile_id=spec.profile_id,
            source_commit=str(ledger["source_git_commit"]),
            host=_mapping(ledger["host"], name="ledger host"),
            execution_inputs=execution_inputs,
        )
        if ledger_profile.get("invocation_argv") != invocation:
            raise ValueError("ledger profile invocation differs from terminal profile")
        if not claim_path.is_file() or claim_path.is_symlink():
            raise ValueError("durable hidden profile invocation claim is missing")
        claim = _load(claim_path)
        _exact_keys(
            claim,
            {
                "schema_version",
                "protocol_id",
                "attempt_id",
                "status",
                "position",
                "profile_id",
                "invocation_argv",
                "claimed_at_utc",
            },
            name="profile invocation claim",
        )
        if (
            claim.get("schema_version") != SCHEMA_VERSION
            or claim.get("protocol_id") != PROTOCOL_ID
            or claim.get("attempt_id") != ATTEMPT_ID
            or claim.get("status") != "PROFILE_INVOCATION_CLAIMED_NO_RETRY"
            or claim.get("position") != position
            or claim.get("profile_id") != spec.profile_id
            or claim.get("invocation_argv") != invocation
        ):
            raise ValueError("profile invocation claim drifted")
        _timestamp(claim.get("claimed_at_utc"), name="profile claimed_at_utc")
        if not expected_start.is_file() or expected_start.is_symlink():
            raise ValueError("durable profile start is missing")
        profile_start = _load(expected_start)
        _exact_keys(
            profile_start,
            {
                "schema_version",
                "protocol_id",
                "attempt_id",
                "status",
                "position",
                "profile_id",
                "state_indices",
                "invocation_argv",
                "started_at_utc",
            },
            name="profile start",
        )
        if (
            profile_start.get("schema_version") != SCHEMA_VERSION
            or profile_start.get("protocol_id") != PROTOCOL_ID
            or profile_start.get("attempt_id") != ATTEMPT_ID
            or profile_start.get("status") != "PROFILE_STARTED_NO_RETRY"
            or profile_start.get("position") != position
            or profile_start.get("profile_id") != spec.profile_id
            or profile_start.get("state_indices") != list(spec.state_indices)
            or profile_start.get("invocation_argv") != invocation
        ):
            raise ValueError("profile start drifted")
        _timestamp(
            profile_start.get("started_at_utc"),
            name="profile start started_at_utc",
        )
        if position == 0 and attempt_start.get("first_invocation_argv") != invocation:
            raise ValueError("attempt start first invocation differs from profile zero")
        for ordinal, (state, profile_record, state_index) in enumerate(
            zip(states, profile["records"], spec.state_indices, strict=True)
        ):
            state_root = profile_root / "states" / f"{ordinal:03d}-{state_index:03d}"
            start = state_root / "start.json"
            terminal = state_root / "terminal.json"
            _exact_keys(
                _mapping(state, name="ledger state"),
                {
                    "ordinal",
                    "index",
                    "state_id",
                    "status",
                    "start_path",
                    "terminal_path",
                    "operation_counts",
                    "ended_at_utc",
                },
                name="ledger completed state",
            )
            if not start.is_file() or start.is_symlink():
                raise ValueError("durable state start is missing")
            start_payload = _load(start)
            _exact_keys(
                start_payload,
                {
                    "schema_version",
                    "protocol_id",
                    "attempt_id",
                    "status",
                    "profile_id",
                    "ordinal",
                    "index",
                    "role",
                    "state_id",
                    "expected_operation_counts",
                    "started_at_utc",
                },
                name="state start",
            )
            parent = parent_by_index[state_index]
            expected_state_counts = _expected_state_counts(spec)
            if (
                start_payload.get("schema_version") != SCHEMA_VERSION
                or start_payload.get("protocol_id") != PROTOCOL_ID
                or start_payload.get("attempt_id") != ATTEMPT_ID
                or start_payload.get("status") != "STATE_STARTED_NO_RETRY"
                or start_payload.get("profile_id") != spec.profile_id
                or start_payload.get("ordinal") != ordinal
                or start_payload.get("index") != state_index
                or start_payload.get("role") != parent.role
                or start_payload.get("state_id") != parent.state_id
                or start_payload.get("expected_operation_counts")
                != expected_state_counts
            ):
                raise ValueError("state start drifted")
            _timestamp(
                start_payload.get("started_at_utc"),
                name="state start started_at_utc",
            )
            payload = _load(terminal)
            _exact_keys(
                payload,
                {
                    "schema_version",
                    "protocol_id",
                    "attempt_id",
                    "status",
                    "profile_id",
                    "ordinal",
                    "operation_counts",
                    "record",
                },
                name="state terminal",
            )
            if (
                state.get("ordinal") != ordinal
                or state.get("index") != state_index
                or state.get("state_id") != parent.state_id
                or state.get("status") != "COMPLETED"
                or state.get("start_path") != str(start)
                or state.get("terminal_path") != str(terminal)
                or state.get("operation_counts") != expected_state_counts
                or payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("protocol_id") != PROTOCOL_ID
                or payload.get("attempt_id") != ATTEMPT_ID
                or payload.get("status") != "COMPLETED_SPATIAL_REFERENCE_AUDIT_STATE"
                or payload.get("profile_id") != spec.profile_id
                or payload.get("ordinal") != ordinal
                or payload.get("record") != profile_record
                or payload.get("operation_counts") != state.get("operation_counts")
            ):
                raise ValueError("durable state terminal differs from profile aggregate")
            _timestamp(state.get("ended_at_utc"), name="ledger state ended_at_utc")
    return ledger


def _audit_decision(
    profiles: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> tuple[str, dict[str, bool]]:
    if tuple(item["profile"]["profile_id"] for item in profiles) != EXPECTED_PROFILE_IDS:
        raise ValueError("profile decision order drifted")
    auto_metrics = _mapping(profiles[0].get("metrics"), name="bf16 auto metrics")
    eager_metrics = _mapping(profiles[1].get("metrics"), name="bf16 eager metrics")
    auto_stable = (
        auto_metrics.get("state_count") == 13
        and auto_metrics.get("exact_generated_token_sequence_stable_count") == 13
    )
    eager_stable = (
        eager_metrics.get("state_count") == 13
        and eager_metrics.get("exact_generated_token_sequence_stable_count") == 13
    )
    rule = config["decision_rule"]
    if not eager_stable:
        decision = rule["if_bf16_eager_unstable"]
    elif not auto_stable:
        decision = rule["if_bf16_eager_stable_and_bf16_auto_unstable"]
    else:
        decision = rule["if_bf16_eager_stable_and_bf16_auto_stable"]
    return decision, {
        "bf16_auto_exact_token_stable": auto_stable,
        "bf16_eager_control_exact_token_stable": eager_stable,
    }


def aggregate_profiles(
    *,
    repository_root: Path,
    config: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    attempt_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    if tuple(item["profile"]["profile_id"] for item in profiles) != EXPECTED_PROFILE_IDS:
        raise ValueError("profile aggregation order drifted")
    source_commits = {item["source_git_commit"] for item in profiles}
    if len(source_commits) != 1:
        raise ValueError("profiles were not run from one source commit")
    hosts = [item["host"] for item in profiles]
    if len({json.dumps(item, sort_keys=True) for item in hosts}) != 1:
        raise ValueError("profiles differ in controlled live host identity")
    operation_counts = {
        key: sum(item["operation_counts"][key] for item in profiles)
        for key in config["operation_limits"]
    }
    if operation_counts != config["operation_limits"]:
        raise ValueError("three-profile operation schedule differs from the frozen total")
    decision, decision_basis = _audit_decision(profiles, config=config)
    profile_summaries = []
    for item in profiles:
        signed_margins = [
            repeat["generation_aligned_first_minus_second_candidate_logit"]
            for record in item["records"]
            for repeat in record["shared_prefix_parent_pair_repeats"]
        ]
        absolute_margins = [abs(value) for value in signed_margins]
        repeat_deltas = [
            abs(
                record["shared_prefix_parent_pair_repeats"][0][
                    "generation_aligned_first_minus_second_candidate_logit"
                ]
                - record["shared_prefix_parent_pair_repeats"][1][
                    "generation_aligned_first_minus_second_candidate_logit"
                ]
            )
            for record in item["records"]
        ]
        profile_summaries.append(
            {
                "profile_id": item["profile"]["profile_id"],
                "strict_cuda_determinism_claimed": False,
                "scientific_payload_sha256": item["scientific_payload_sha256"],
                "metrics": dict(item["metrics"]),
                "operation_counts": dict(item["operation_counts"]),
                "shared_prefix_competing_token_abs_margin_min": min(absolute_margins),
                "shared_prefix_competing_token_abs_margin_median": statistics.median(
                    absolute_margins
                ),
                "shared_prefix_competing_token_abs_margin_max": max(absolute_margins),
                "shared_prefix_repeat_pair_margin_delta_max": max(repeat_deltas),
                "duration_seconds": item["duration_seconds"],
                "peak_gpu_memory_allocated_bytes": max(
                    generation["metadata"]["peak_gpu_memory_allocated_bytes"]
                    for record in item["records"]
                    for generation in record["generations"]
                ),
            }
        )
    parent_deltas = [
        {
            "index": record["index"],
            "state_id": record["state_id"],
            **record["coordinate_delta"],
        }
        for record in profiles[0]["records"]
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "attempt_id": ATTEMPT_ID,
        "status": "VALID_SPATIAL_REFERENCE_AUDIT_V1",
        "decision": decision,
        "decision_basis": decision_basis,
        "source_git_commit": next(iter(source_commits)),
        "validated_git_commit": dict(validate_clean_pushed_main(repository_root))[
            "commit"
        ],
        "config_sha256": sha256_file(
            repository_root / "code/configs/spatial_reference_audit_v1.json"
        ),
        "global_attempt_ledger_sha256": sha256_file(
            Path(config["attempt_identity"]["canonical_global_attempt_ledger"])
        ),
        "profile_summaries": profile_summaries,
        "parent_coordinate_deltas": parent_deltas,
        "operation_counts": operation_counts,
        "claim_boundary": {
            "parent_v2_1_outcome_unchanged": "NO_GO_V2_1_FULL_45_SUBSTRATE",
            "confirm_policy_output_accessed": False,
            "restoration_work_performed": False,
            "gate_training_performed": False,
            "teacher_margin_is_original_generation_time_margin": False,
            "fp32_probe_controls_decision": False,
            "strict_cuda_determinism_claimed": False,
            "current_45_states_are_independent_validation": False,
        },
    }
    result["scientific_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def _fixture_by_index(path: Path) -> dict[int, Mapping[str, Any]]:
    fixture = load_json_object(path)
    records = fixture.get("mismatches")
    if not isinstance(records, list):
        raise ValueError("parent mismatch fixture inventory is invalid")
    result = {int(record["index"]): record for record in records}
    if tuple(sorted(result)) != tuple(
        load_and_validate_config(
            path.parents[2] / "code/configs/spatial_reference_audit_v1.json"
        )["mismatch_denominator"]["state_indices"]
    ):
        raise ValueError("parent mismatch fixture denominator drifted")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--bf16-auto", required=True)
    parser.add_argument("--bf16-eager", required=True)
    parser.add_argument("--fp32-eager", required=True)
    parser.add_argument("--summary-output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    if not root.is_absolute() or config_path != root / "code/configs/spatial_reference_audit_v1.json":
        raise ValueError("validator paths are not canonical absolute paths")
    validate_clean_pushed_main(root)
    config = load_and_validate_config(config_path)
    validate_repository_inputs(root, config)
    attempt = config["attempt_identity"]
    canonical_paths = [
        Path(attempt["canonical_persistent_output_dir"])
        / "profiles"
        / f"{position:03d}-{profile_id}"
        / "terminal.json"
        for position, profile_id in enumerate(EXPECTED_PROFILE_IDS)
    ]
    supplied_paths = [Path(path).resolve() for path in (args.bf16_auto, args.bf16_eager, args.fp32_eager)]
    if supplied_paths != canonical_paths:
        raise ValueError("validator profile paths differ from the canonical attempt")
    fixture = _fixture_by_index(
        root / config["inputs"]["parent_mismatch_fixture"]["path"]
    )
    raw = [_load(path) for path in supplied_paths]
    parent_archive_values = [item.get("parent_raw_archive_path") for item in raw]
    if not all(isinstance(item, str) for item in parent_archive_values) or len(
        set(parent_archive_values)
    ) != 1:
        raise ValueError("profiles do not bind one parent raw archive path")
    parent_mismatches = load_parent_mismatches(
        raw_archive=Path(str(parent_archive_values[0])),
        fixture_path=root / config["inputs"]["parent_mismatch_fixture"]["path"],
        config=config,
    )
    parent_by_index = {item.index: item for item in parent_mismatches}
    validated = [
        validate_profile(
            value,
            config=config,
            repository_root=root,
            fixture_by_index=fixture,
            parent_by_index=parent_by_index,
        )
        for value in raw
    ]
    ledger = _validate_attempt_ledger(
        config=config,
        profiles=validated,
        repository_root=root,
        parent_by_index=parent_by_index,
    )
    summary = aggregate_profiles(
        repository_root=root,
        config=config,
        profiles=validated,
        attempt_ledger=ledger,
    )
    if args.summary_output is not None:
        output = Path(args.summary_output).resolve()
        expected = Path(attempt["canonical_persistent_output_dir"]) / "summary.json"
        if output != expected:
            raise ValueError("summary output differs from the canonical attempt root")
        _write_json_exclusive(output, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
