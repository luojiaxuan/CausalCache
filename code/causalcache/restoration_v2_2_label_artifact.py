"""Validation and deterministic packaging for v2.2 eager restoration labels."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tarfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.restoration_v2_2_eager_artifact import (
    canonical_json_bytes,
    expected_worker_specs,
    pretty_json_bytes,
    strict_json_object_bytes,
    validate_v22_distance_audit,
    validate_v22_teacher_metadata,
    validate_worker_runtime_metadata,
    validate_worker_runtime_pair,
)
from causalcache.restoration_v2_2_label_contract import (
    EXPECTED_DEPLOYMENT_EDGES,
    EXPECTED_DISTANCE_ROWS,
    EXPECTED_FULL_EDGES,
    EXPECTED_KL_MEASUREMENTS,
    EXPECTED_PRIMARY_ORACLES,
    EXPECTED_STATE_COUNT,
    EXPECTED_TEACHER_FORWARDS,
    PROTOCOL_ID,
    RestorationLabelAttemptProfile,
    V1_ATTEMPT_PROFILE,
    label_attempt_profile_for_config_path,
    label_attempt_profile_for_id,
)
from causalcache.restoration_v2_2_label_table import (
    deployment_conditional_edges,
    exact_permutation_average_attribution,
    full_conditional_edges,
    pair_interactions,
    primary_exact_subset_oracle,
    validate_complete_distance_table,
)


SCHEMA_VERSION = "1.0.0"
ARCHIVE_FORMAT = "ustar"
GLOBAL_LEDGER_MEMBER = "global_attempt_ledger.json"
SIBLING_PREFIX = "worker_sibling_ledgers"
RUN_MANIFEST_FILENAME = "run_manifest.json"
AGGREGATE_FILENAME = "aggregate.json"
NORMALIZATION_EPSILON = 1e-12
MAXIMUM_REPEAT_KL = 1e-4
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
_FULL_45_PROJECTION_SHA256 = (
    "65e7085f01bc26e425bab7f5148c6729bc8c6d0a62802ebc5e8dfeaed6439249"
)
_CANONICAL_REMOTE_URL = "https://github.com/luojiaxuan/CausalCache.git"
_CANONICAL_IMAGE_DIGEST = (
    "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
)
_RUN_SECTION_SHA256 = {
    "enumeration": "33fc7fb27264b4a0962f3577be89e9e7dddc55744ee0056454f65765a51cd0a3",
    "reduction": "c4c972fea4aec4c3abb4b20e2ec1e080fca0aae4b3036984bb3b0fcb26385845",
    "runtime_requirements": "a30d60b7ed760aecaa984c7803f197f38e2cff4c0a5d465614d2288fb3753967",
    "operation_schedule": "8e3de4f3722dd48176456b7f1d1f1815b86e5ff41916c9951ea54e7667b466f2",
    "worker_topology": "bdd9840a53af1c5bfb2b6bd654e366b10a4b2c5a9d4d215c9d9d6cae33390069",
    "prohibited_work": "70eef75bc3ce10fe4358293ee5d991fd15c9f85d0a25dcba766544313360af6c",
}
_EXPECTED_CANONICAL_INPUTS = {
    "scientific_config": {
        "path": "code/configs/causalcache_restoration_v2.json",
        "sha256": "9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc",
    },
    "selection_manifest": {
        "path": "data/manifests/restoration_v2_selection.json",
        "sha256": "292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b",
    },
    "ocr_backend_config": {
        "path": "code/configs/restoration_v2_ocr_backend.json",
        "sha256": "51e08a9565f804ecb1ee7f12887cc12742b84c04996fffbbde7c0f304e4bd036",
    },
    "snapshot_manifest": {
        "path": "code/configs/gui_owl_1_5_8b_snapshot.json",
        "sha256": "50b675ec31c5c46dbb0d44c137a808fffb9d054916d39b596648d4eb9df7cbc3",
    },
    "derived_artifact": {
        "repo": "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
        "immutable_revision": "89f136abaff797e14fe758a198996e51032a10a6",
        "artifact_tree_sha256": "475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e",
    },
}
_EXPECTED_OPERATION_TOTALS = {
    "generation_call_count": 0,
    "reference_teacher_forward_count": 45,
    "reference_repeat_teacher_forward_count": 45,
    "non_reference_coalition_teacher_forward_count": 375,
    "teacher_forward_call_count": EXPECTED_TEACHER_FORWARDS,
    "teacher_forward_example_count": EXPECTED_TEACHER_FORWARDS,
    "coalition_kl_measurement_count": 375,
    "reference_repeat_kl_measurement_count": 45,
    "kl_measurement_count": EXPECTED_KL_MEASUREMENTS,
    "raw_distance_row_count": EXPECTED_DISTANCE_ROWS,
    "deployment_conditional_label_count": EXPECTED_DEPLOYMENT_EDGES,
    "full_hypercube_edge_count": EXPECTED_FULL_EDGES,
    "pair_interaction_count": 465,
    "exact_permutation_attribution_count": 135,
    "primary_exact_subset_oracle_count": EXPECTED_PRIMARY_ORACLES,
    "retry_count": 0,
    "top_up_count": 0,
    "confirm_state_access_count": 0,
    "confirm_processor_prompt_count": 0,
    "confirm_decoder_input_count": 0,
    "confirm_generation_count": 0,
    "confirm_teacher_forward_count": 0,
    "expert_action_read_count": 0,
    "gate_training_example_count": 0,
    "gate_model_forward_count": 0,
    "gate_selection_count": 0,
    "matched_nll_evaluation_count": 0,
    "closed_loop_episode_count": 0,
}


@dataclass(frozen=True)
class LabelEvidence:
    files: Mapping[str, bytes]
    source_git_commit: str
    run_contract_sha256: str
    outcome: str
    aggregate: Mapping[str, Any]
    inventory: tuple[Mapping[str, Any], ...]
    tree_inventory_sha256: str
    profile: RestorationLabelAttemptProfile = V1_ATTEMPT_PROFILE


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expected_file_names() -> frozenset[str]:
    names = {
        GLOBAL_LEDGER_MEMBER,
        f"{SIBLING_PREFIX}/even.json",
        f"{SIBLING_PREFIX}/odd.json",
        RUN_MANIFEST_FILENAME,
        AGGREGATE_FILENAME,
    }
    for worker, parity in (("even", 0), ("odd", 1)):
        base = f"workers/{worker}"
        names.update(
            {
                f"{base}/worker_attempt_ledger.json",
                f"{base}/runtime_identity.json",
                f"{base}/terminal.json",
            }
        )
        for index in range(parity, EXPECTED_STATE_COUNT, 2):
            names.add(f"{base}/attempts/{index:03d}.json")
            names.add(f"{base}/states/{index:03d}.json")
    return frozenset(names)


def _safe_member_name(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError("label evidence member path is not canonical relative POSIX")
    return value


def _json(files: Mapping[str, bytes], name: str) -> dict[str, Any]:
    return strict_json_object_bytes(files[name], label=name)


def _normalized(value: float, baseline: float) -> float | None:
    return None if baseline <= NORMALIZATION_EPSILON else value / baseline


def _validate_run_contract(
    run_contract: Mapping[str, Any],
) -> tuple[
    RestorationLabelAttemptProfile,
    str,
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "contract_source",
        "git_identity",
        "source_inventory",
        "parent_evidence",
        "canonical_inputs",
        "states",
        "parent_states",
        "runtime_requirements",
        "worker_topology",
        "enumeration",
        "reduction",
        "operation_schedule",
        "prohibited_work",
        "attempt_identity",
        "execution_argv",
    }
    if set(run_contract) != expected_keys:
        raise ValueError("label run contract field inventory drifted")
    contract_source = run_contract.get("contract_source")
    if not isinstance(contract_source, Mapping):
        raise ValueError("label run contract source identity drifted")
    profile = label_attempt_profile_for_config_path(str(contract_source.get("path")))
    if (
        run_contract.get("schema_version") != SCHEMA_VERSION
        or run_contract.get("protocol_id") != PROTOCOL_ID
        or contract_source
        != {
            "path": profile.config_path,
            "sha256": profile.frozen_config_sha256,
        }
    ):
        raise ValueError("label run contract source identity drifted")

    git_identity = run_contract.get("git_identity")
    if not isinstance(git_identity, Mapping):
        raise ValueError("label run contract lacks Git identity")
    source_git_commit = git_identity.get("commit")
    if (
        set(git_identity)
        != {
            "branch",
            "commit",
            "origin_main",
            "remote_main",
            "remote_url",
            "worktree",
        }
        or not isinstance(source_git_commit, str)
        or _GIT_SHA.fullmatch(source_git_commit) is None
        or git_identity.get("branch") != "main"
        or git_identity.get("origin_main") != source_git_commit
        or git_identity.get("remote_main") != source_git_commit
        or git_identity.get("remote_url") != _CANONICAL_REMOTE_URL
        or git_identity.get("worktree") != "clean_including_untracked"
    ):
        raise ValueError("label run contract Git identity drifted")

    inventory = run_contract.get("source_inventory")
    if not isinstance(inventory, list) or len(inventory) != len(
        profile.expected_source_paths
    ):
        raise ValueError("label run contract source inventory length drifted")
    observed_paths = []
    for record in inventory:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256", "git_commit"}
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("sha256"), str)
            or _SHA256.fullmatch(record["sha256"]) is None
            or record.get("git_commit") != source_git_commit
        ):
            raise ValueError("label run contract source inventory record drifted")
        observed_paths.append(record["path"])
    if tuple(observed_paths) != profile.expected_source_paths:
        raise ValueError("label run contract source paths drifted")
    if inventory[0]["sha256"] != profile.frozen_config_sha256:
        raise ValueError("label run contract config source blob drifted")

    parent_evidence = run_contract.get("parent_evidence")
    if (
        not isinstance(parent_evidence, Mapping)
        or set(parent_evidence) != {
            "path",
            "sha256",
            "size_bytes",
            "validation_status",
        }
        or not isinstance(parent_evidence.get("path"), str)
        or not Path(parent_evidence["path"]).is_absolute()
        or parent_evidence.get("sha256")
        != "b22827e6e2d8d33b03686fc177dc8f9c55c5133470fbe40f9fb3e33cce809fb5"
        or parent_evidence.get("size_bytes") != 1_269_760
        or parent_evidence.get("validation_status")
        != "VALIDATED_IMMUTABLE_V2_2_EAGER_PARENT_ARCHIVE"
    ):
        raise ValueError("label run contract parent evidence drifted")
    canonical_inputs = run_contract.get("canonical_inputs")
    if not isinstance(canonical_inputs, Mapping):
        raise ValueError("label run contract canonical inputs drifted")
    base_inputs = {
        key: canonical_inputs.get(key) for key in _EXPECTED_CANONICAL_INPUTS
    }
    if base_inputs != _EXPECTED_CANONICAL_INPUTS:
        raise ValueError("label run contract canonical inputs drifted")
    preclaim = canonical_inputs.get("model_snapshot_preclaim")
    if profile.supersedes_attempt_id is not None:
        expected_preclaim = {
            "model_dir": str(profile.canonical_model_dir),
            "model_repo": "mPLUG/GUI-Owl-1.5-8B-Instruct",
            "model_revision": "06d5faecff74840bab2be2425e9c42667a5d04fc",
            "snapshot_manifest_sha256": (
                "50b675ec31c5c46dbb0d44c137a808fffb9d054916d39b596648d4eb9df7cbc3"
            ),
            "verified_model_file_count": 14,
            "verified_model_total_bytes": 17_545_907_171,
            "validation_status": (
                "VALIDATED_FULL_MODEL_SNAPSHOT_BEFORE_GLOBAL_CLAIM"
            ),
        }
        if set(canonical_inputs) != set(_EXPECTED_CANONICAL_INPUTS).union(
            {"model_snapshot_preclaim"}
        ) or preclaim != expected_preclaim:
            raise ValueError("repair model snapshot preclaim identity drifted")
    elif set(canonical_inputs) not in (
        set(_EXPECTED_CANONICAL_INPUTS),
        set(_EXPECTED_CANONICAL_INPUTS).union({"model_snapshot_preclaim"}),
    ):
        raise ValueError("v1 canonical input inventory drifted")

    projections = run_contract.get("states")
    parents = run_contract.get("parent_states")
    if (
        not isinstance(projections, list)
        or not isinstance(parents, list)
        or len(projections) != EXPECTED_STATE_COUNT
        or len(parents) != EXPECTED_STATE_COUNT
        or sha256_bytes(canonical_json_bytes(projections))
        != _FULL_45_PROJECTION_SHA256
    ):
        raise ValueError("label run contract state projection drifted")
    observed_state_ids: set[str] = set()
    for index, (projection, parent) in enumerate(zip(projections, parents, strict=True)):
        expected_role = "v2_label_train" if index < 30 else "v2_development"
        expected_step = 4 + index % 3
        if (
            not isinstance(projection, Mapping)
            or set(projection)
            != {
                "index",
                "role",
                "trajectory_id",
                "decision_step_id",
                "state_id",
                "candidate_event_step_ids",
            }
            or projection.get("index") != index
            or projection.get("role") != expected_role
            or projection.get("decision_step_id") != expected_step
            or not isinstance(projection.get("trajectory_id"), str)
            or projection.get("state_id")
            != (
                f"{projection.get('trajectory_id')}:"
                f"decision_step:{expected_step:03d}"
            )
            or projection.get("candidate_event_step_ids")
            != list(range(1, expected_step - 1))
            or projection.get("state_id") in observed_state_ids
        ):
            raise ValueError(f"label run contract state {index} drifted")
        observed_state_ids.add(str(projection["state_id"]))
        expected_worker = "even" if index % 2 == 0 else "odd"
        if (
            not isinstance(parent, Mapping)
            or set(parent)
            != {
                "index",
                "state_id",
                "member_name",
                "member_sha256",
                "canonical_action_sha256",
            }
            or parent.get("index") != index
            or parent.get("state_id") != projection.get("state_id")
            or parent.get("member_name")
            != f"workers/{expected_worker}/states/{index:03d}.json"
            or not isinstance(parent.get("member_sha256"), str)
            or _SHA256.fullmatch(parent["member_sha256"]) is None
            or not isinstance(parent.get("canonical_action_sha256"), str)
            or _SHA256.fullmatch(parent["canonical_action_sha256"]) is None
        ):
            raise ValueError(f"label run contract parent state {index} drifted")

    for key, expected_sha256 in _RUN_SECTION_SHA256.items():
        if sha256_bytes(canonical_json_bytes(run_contract.get(key))) != expected_sha256:
            raise ValueError(f"label run contract {key} drifted")

    attempt = run_contract.get("attempt_identity")
    expected_hosts = {
        "hyper00": "node-radixark-16-0000",
        "hyper01": "node-radixark-16-0001",
    }
    rich_attempt = {
        "attempt_id": profile.attempt_id,
        "attempt_revision": profile.attempt_revision,
        "supersedes_attempt_id": profile.supersedes_attempt_id,
        "pass_outcome": profile.pass_outcome,
        "invalid_outcome": profile.invalid_outcome,
        "aggregate_status": profile.aggregate_status,
        "output_dir": str(profile.output_dir),
        "global_ledger": str(profile.ledger_path),
        "raw_archive": str(profile.archive_path),
        "hf_repo": profile.hf_repo,
        "hf_tag": profile.hf_tag,
        "hf_path": profile.hf_path,
    }
    common_attempt_keys = {
        "host_alias",
        "host_hostname",
        "container_id",
        "container_image_digest",
    }
    if (
        not isinstance(attempt, Mapping)
        or (
            profile.supersedes_attempt_id is not None
            and set(attempt) != set(rich_attempt).union(common_attempt_keys)
        )
        or any(attempt.get(key) != value for key, value in rich_attempt.items() if key in attempt or profile.supersedes_attempt_id is not None)
        or attempt.get("host_alias") not in expected_hosts
        or attempt.get("host_hostname") != expected_hosts.get(attempt.get("host_alias"))
        or not isinstance(attempt.get("container_id"), str)
        or _CONTAINER_ID.fullmatch(attempt["container_id"]) is None
        or attempt.get("container_image_digest") != _CANONICAL_IMAGE_DIGEST
    ):
        raise ValueError("label run contract attempt identity drifted")
    argv = run_contract.get("execution_argv")
    if not isinstance(argv, list) or not argv or any(
        not isinstance(item, str) or not item for item in argv
    ):
        raise ValueError("label run contract execution argv drifted")
    return profile, source_git_commit, projections, parents


def _expected_edge_payloads(table: Any, *, deployment: bool) -> list[dict[str, Any]]:
    edges = deployment_conditional_edges(table) if deployment else full_conditional_edges(table)
    result = []
    baseline = table.distance(())
    for edge in edges:
        value = asdict(edge)
        value["base_coalition"] = list(edge.base_coalition)
        value["restored_coalition"] = list(edge.restored_coalition)
        if deployment:
            value["normalized_marginal_gain"] = _normalized(
                edge.marginal_gain,
                baseline,
            )
        result.append(value)
    return result


def _expected_interactions(table: Any) -> list[dict[str, Any]]:
    result = []
    for item in pair_interactions(table):
        value = asdict(item)
        value["conditioning_coalition"] = list(item.conditioning_coalition)
        result.append(value)
    return result


def _expected_attribution(table: Any) -> list[dict[str, Any]]:
    result = []
    baseline = table.distance(())
    for item in exact_permutation_average_attribution(table):
        value = asdict(item)
        value["marginal_samples"] = list(item.marginal_samples)
        value["normalized_mean_marginal_gain"] = _normalized(
            item.mean_marginal_gain,
            baseline,
        )
        result.append(value)
    return result


def _expected_oracle(table: Any) -> dict[str, Any]:
    oracle = primary_exact_subset_oracle(table)
    value = asdict(oracle)
    value["coalition"] = list(oracle.coalition)
    value["normalized_recovery"] = _normalized(
        oracle.utility,
        table.distance(()),
    )
    return value


def _expected_state_operation_counts(event_count: int) -> dict[str, int]:
    rows = 1 << event_count
    return {
        "generation_call_count": 0,
        "reference_teacher_forward_count": 1,
        "reference_repeat_teacher_forward_count": 1,
        "non_reference_coalition_teacher_forward_count": rows - 1,
        "teacher_forward_call_count": rows + 1,
        "teacher_forward_example_count": rows + 1,
        "coalition_kl_measurement_count": rows - 1,
        "reference_repeat_kl_measurement_count": 1,
        "kl_measurement_count": rows,
        "raw_distance_row_count": rows,
        "deployment_conditional_label_count": {2: 4, 3: 9, 4: 16}[event_count],
        "full_hypercube_edge_count": event_count * (1 << (event_count - 1)),
        "pair_interaction_count": {2: 1, 3: 6, 4: 24}[event_count],
        "exact_permutation_attribution_count": event_count,
        "primary_exact_subset_oracle_count": 1,
        "retry_count": 0,
        "top_up_count": 0,
        "confirm_state_access_count": 0,
        "confirm_processor_prompt_count": 0,
        "confirm_decoder_input_count": 0,
        "confirm_generation_count": 0,
        "confirm_teacher_forward_count": 0,
        "expert_action_read_count": 0,
        "gate_training_example_count": 0,
        "gate_model_forward_count": 0,
        "gate_selection_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
    }


def _validate_profile_envelope(
    value: Mapping[str, Any],
    *,
    profile: RestorationLabelAttemptProfile,
    label: str,
) -> None:
    if profile.supersedes_attempt_id is not None:
        if (
            value.get("attempt_id") != profile.attempt_id
            or value.get("attempt_revision") != profile.attempt_revision
        ):
            raise ValueError(f"{label} attempt profile drifted")
    elif (
        "attempt_id" in value
        and value.get("attempt_id") != profile.attempt_id
    ) or (
        "attempt_revision" in value
        and value.get("attempt_revision") != profile.attempt_revision
    ):
        raise ValueError(f"{label} attempt profile drifted")


def _validate_prompt_inventory(
    value: Any,
    *,
    projection: Mapping[str, Any],
    coalition: tuple[int, ...],
    coalition_mask: int,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("label prompt inventory is not an object")
    expected_keys = {
        "schema_version",
        "state_id",
        "role",
        "trajectory_id",
        "decision_step_id",
        "candidate_event_step_ids",
        "coalition_mask",
        "fidelity",
        "high_fidelity_restored_event_step_ids",
        "high_fidelity_image_count",
        "current_observation_image_count",
        "image_count",
        "text_block_count",
        "policy_visible_text_sha256",
        "user_block_types",
    }
    candidates = tuple(projection["candidate_event_step_ids"])
    expected_fidelity = (
        "summary_only"
        if not coalition
        else "full_reference"
        if coalition == candidates
        else "mixed_fidelity"
    )
    block_types = value.get("user_block_types")
    if (
        set(value) != expected_keys
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("state_id") != projection.get("state_id")
        or value.get("role") != projection.get("role")
        or value.get("trajectory_id") != projection.get("trajectory_id")
        or value.get("decision_step_id") != projection.get("decision_step_id")
        or value.get("candidate_event_step_ids") != list(candidates)
        or value.get("coalition_mask") != coalition_mask
        or value.get("fidelity") != expected_fidelity
        or value.get("high_fidelity_restored_event_step_ids") != list(coalition)
        or value.get("high_fidelity_image_count") != len(coalition)
        or value.get("current_observation_image_count") != 1
        or value.get("image_count") != 1 + len(coalition)
        or type(value.get("text_block_count")) is not int
        or value["text_block_count"] <= 0
        or not isinstance(value.get("policy_visible_text_sha256"), str)
        or _SHA256.fullmatch(value["policy_visible_text_sha256"]) is None
        or not isinstance(block_types, list)
        or any(block_type not in {"text", "image"} for block_type in block_types)
        or block_types.count("image") != 1 + len(coalition)
        or block_types.count("text") > value.get("text_block_count")
    ):
        raise ValueError("label prompt inventory drifted")


def _validate_teacher_sample_binding(
    teacher: Mapping[str, Any],
    prompt_inventory: Mapping[str, Any],
) -> None:
    samples = teacher.get("samples")
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError("label teacher sample inventory drifted")
    sample = samples[0]
    expected_keys = {
        "image_count",
        "image_grid_thw",
        "effective_visual_tokens",
        "policy_visible_text_tokens",
        "prompt_input_tokens",
        "teacher_carrier_tokens",
        "distance_action_tokens",
        "model_input_tokens",
        "distance_logit_positions",
    }
    if not isinstance(sample, Mapping) or set(sample) != expected_keys:
        raise ValueError("label teacher sample schema drifted")
    image_count = sample.get("image_count")
    grids = sample.get("image_grid_thw")
    if (
        type(image_count) is not int
        or image_count != prompt_inventory.get("image_count")
        or not isinstance(grids, list)
        or len(grids) != image_count
    ):
        raise ValueError("label teacher sample image inventory drifted")
    merged_tokens = 0
    for grid in grids:
        if (
            not isinstance(grid, list)
            or len(grid) != 3
            or any(type(value) is not int or value <= 0 for value in grid)
            or grid[1] % 2
            or grid[2] % 2
        ):
            raise ValueError("label teacher sample image grid drifted")
        merged_tokens += grid[0] * grid[1] * grid[2] // 4
    effective_visual = sample.get("effective_visual_tokens")
    visible_text = sample.get("policy_visible_text_tokens")
    prompt_tokens = sample.get("prompt_input_tokens")
    distance_tokens = sample.get("distance_action_tokens")
    model_tokens = sample.get("model_input_tokens")
    positions = sample.get("distance_logit_positions")
    if (
        type(effective_visual) is not int
        or effective_visual != merged_tokens
        or type(visible_text) is not int
        or visible_text <= 0
        or type(prompt_tokens) is not int
        or prompt_tokens != effective_visual + visible_text
        or sample.get("teacher_carrier_tokens") != 0
        or type(distance_tokens) is not int
        or distance_tokens <= 0
        or distance_tokens != teacher.get("logits_to_keep")
        or type(model_tokens) is not int
        or model_tokens != prompt_tokens + distance_tokens - 1
        or positions
        != list(range(prompt_tokens - 1, prompt_tokens - 1 + distance_tokens))
    ):
        raise ValueError("label teacher sample token geometry drifted")


def _validate_state(
    record: Mapping[str, Any],
    *,
    index: int,
    projection: Mapping[str, Any],
    parent: Mapping[str, Any],
    run_contract_sha256: str,
    spec: Any,
    runtime_metadata: Mapping[str, Any],
    profile: RestorationLabelAttemptProfile,
) -> Mapping[str, int]:
    _validate_profile_envelope(record, profile=profile, label=f"label state {index}")
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("protocol_id") != PROTOCOL_ID
        or record.get("run_contract_sha256") != run_contract_sha256
        or record.get("status") != "VALID_RESTORATION_V2_2_EAGER_LABEL_STATE"
        or record.get("state") != projection
    ):
        raise ValueError(f"label state {index} identity drifted")
    reference = record.get("parent_reference")
    if not isinstance(reference, Mapping) or (
        reference.get("member_name") != parent.get("member_name")
        or reference.get("member_sha256") != parent.get("member_sha256")
        or reference.get("canonical_action_sha256")
        != parent.get("canonical_action_sha256")
        or reference.get("new_reference_generation_count") != 0
    ):
        raise ValueError(f"label state {index} parent reference drifted")
    canonical_action = reference.get("canonical_action")
    canonical_action_sha256 = reference.get("canonical_action_sha256")
    if (
        not isinstance(canonical_action, Mapping)
        or not isinstance(canonical_action_sha256, str)
        or _SHA256.fullmatch(canonical_action_sha256) is None
        or sha256_bytes(canonical_json_bytes(canonical_action))
        != canonical_action_sha256
    ):
        raise ValueError(f"label state {index} canonical action hash drifted")
    repeat_kl = record.get("reference_repeat_kl")
    if (
        isinstance(repeat_kl, bool)
        or not isinstance(repeat_kl, (int, float))
        or not math.isfinite(float(repeat_kl))
        or float(repeat_kl) < 0
        or float(repeat_kl) > MAXIMUM_REPEAT_KL
    ):
        raise ValueError(f"label state {index} repeat KL is invalid")
    reference_teacher = record.get("reference_teacher_metadata")
    repeat_teacher = record.get("reference_repeat_teacher_metadata")
    if not isinstance(reference_teacher, Mapping) or not isinstance(
        repeat_teacher, Mapping
    ):
        raise ValueError(f"label state {index} reference teacher metadata is missing")
    validate_v22_teacher_metadata(
        reference_teacher,
        spec=spec,
        runtime_metadata=runtime_metadata,
    )
    validate_v22_teacher_metadata(
        repeat_teacher,
        spec=spec,
        runtime_metadata=runtime_metadata,
    )
    validate_v22_distance_audit(
        record.get("reference_repeat_distance_audit"),
        teacher=repeat_teacher,
        spec=spec,
    )
    event_ids = tuple(projection["candidate_event_step_ids"])
    rows = record.get("distance_rows")
    if not isinstance(rows, list) or len(rows) != 1 << len(event_ids):
        raise ValueError(f"label state {index} distance-row count drifted")
    distances: dict[tuple[int, ...], float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"label state {index} distance row is not an object")
        coalition = tuple(row.get("coalition_event_step_ids", ()))
        distance = row.get("distance_kl")
        if coalition in distances:
            raise ValueError(f"label state {index} has duplicate coalition rows")
        distances[coalition] = distance
    table = validate_complete_distance_table(event_ids, distances)
    full_rows = [row for row in table.rows if row.coalition == event_ids]
    if len(full_rows) != 1 or full_rows[0].distance != 0.0:
        raise ValueError(f"label state {index} full-history distance is not zero")
    full_prompt_inventory: Mapping[str, Any] | None = None
    for row, expected in zip(rows, table.rows, strict=True):
        expected_mask = sum(
            1 << bit
            for bit, event_id in enumerate(event_ids)
            if event_id in expected.coalition
        )
        _validate_prompt_inventory(
            row.get("prompt_inventory"),
            projection=projection,
            coalition=expected.coalition,
            coalition_mask=expected_mask,
        )
        teacher = row.get("teacher_metadata")
        if not isinstance(teacher, Mapping):
            raise ValueError(f"label state {index} teacher metadata is missing")
        validate_v22_teacher_metadata(
            teacher,
            spec=spec,
            runtime_metadata=runtime_metadata,
        )
        prompt_inventory = row["prompt_inventory"]
        _validate_teacher_sample_binding(teacher, prompt_inventory)
        if expected.coalition == event_ids:
            full_prompt_inventory = prompt_inventory
            if (
                teacher != reference_teacher
                or row.get("distance_audit")
                != {
                    "reference_identity_distance": True,
                    "full_tensor_host_transfers": 0,
                }
            ):
                raise ValueError(f"label state {index} reference row drifted")
        else:
            validate_v22_distance_audit(
                row.get("distance_audit"),
                teacher=teacher,
                spec=spec,
            )
        if (
            row.get("coalition_mask") != expected_mask
            or row.get("coalition_event_step_ids") != list(expected.coalition)
            or row.get("coalition_slot_cost") != len(expected.coalition)
            or row.get("distance_kl") != expected.distance
            or row.get("restoration_utility") != expected.utility
        ):
            raise ValueError(f"label state {index} canonical distance row drifted")
    if full_prompt_inventory is None:
        raise ValueError(f"label state {index} lacks a full-reference prompt")
    _validate_teacher_sample_binding(repeat_teacher, full_prompt_inventory)
    if record.get("baseline_summary_only_distance") != table.distance(()) or (
        record.get("deployment_conditional_edges")
        != _expected_edge_payloads(table, deployment=True)
        or record.get("full_hypercube_edges")
        != _expected_edge_payloads(table, deployment=False)
        or record.get("pair_interactions") != _expected_interactions(table)
        or record.get("exact_permutation_attribution")
        != _expected_attribution(table)
        or record.get("primary_exact_subset_oracle") != _expected_oracle(table)
    ):
        raise ValueError(f"label state {index} derived scientific rows drifted")
    counts = record.get("operation_counts")
    if counts != _expected_state_operation_counts(len(event_ids)):
        raise ValueError(f"label state {index} operation counts drifted")
    return counts


def _validate_worker_evidence(
    files: Mapping[str, bytes],
    *,
    run_contract_sha256: str,
    profile: RestorationLabelAttemptProfile,
) -> Mapping[str, Mapping[str, Any]]:
    runtimes: dict[str, Mapping[str, Any]] = {}
    for spec in expected_worker_specs():
        expected_indices = list(spec.state_indices)
        base = f"workers/{spec.worker_id}"
        sibling = _json(files, f"{SIBLING_PREFIX}/{spec.worker_id}.json")
        root_ledger = _json(files, f"{base}/worker_attempt_ledger.json")
        expected_ledger_fields = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "LABEL_WORKER_COMPLETED",
            "run_contract_sha256": run_contract_sha256,
            "worker": spec.to_dict(),
            "attempted_state_indices": expected_indices,
            "completed_state_indices": expected_indices,
            "retry_count": 0,
            "top_up_count": 0,
        }
        for label, ledger in (("sibling", sibling), ("root", root_ledger)):
            _validate_profile_envelope(
                ledger,
                profile=profile,
                label=f"label {spec.worker_id} {label} ledger",
            )
            if any(
                ledger.get(key) != value
                for key, value in expected_ledger_fields.items()
            ):
                raise ValueError(
                    f"label {spec.worker_id} {label} worker ledger drifted"
                )

        terminal = _json(files, f"{base}/terminal.json")
        _validate_profile_envelope(
            terminal,
            profile=profile,
            label=f"label {spec.worker_id} worker terminal",
        )
        if (
            terminal.get("schema_version") != SCHEMA_VERSION
            or terminal.get("protocol_id") != PROTOCOL_ID
            or terminal.get("run_contract_sha256") != run_contract_sha256
            or terminal.get("worker") != spec.to_dict()
            or terminal.get("outcome")
            != "PASS_RESTORATION_V2_2_EAGER_LABEL_WORKER"
            or terminal.get("attempted_state_indices") != expected_indices
            or terminal.get("completed_state_indices") != expected_indices
            or terminal.get("failure") is not None
        ):
            raise ValueError(f"label {spec.worker_id} worker terminal drifted")

        runtime = _json(files, f"{base}/runtime_identity.json")
        _validate_profile_envelope(
            runtime,
            profile=profile,
            label=f"label {spec.worker_id} runtime",
        )
        runtime_metadata = runtime.get("runtime_metadata")
        if (
            runtime.get("schema_version") != SCHEMA_VERSION
            or runtime.get("protocol_id") != PROTOCOL_ID
            or runtime.get("run_contract_sha256") != run_contract_sha256
            or runtime.get("worker") != spec.to_dict()
            or not isinstance(runtime_metadata, Mapping)
        ):
            raise ValueError(f"label {spec.worker_id} runtime envelope drifted")
        runtimes[spec.worker_id] = validate_worker_runtime_metadata(
            runtime_metadata,
            spec=spec,
        )

        for index in spec.state_indices:
            marker = _json(files, f"{base}/attempts/{index:03d}.json")
            _validate_profile_envelope(
                marker,
                profile=profile,
                label=f"label attempt marker {index}",
            )
            if (
                marker.get("schema_version") != SCHEMA_VERSION
                or marker.get("protocol_id") != PROTOCOL_ID
                or marker.get("status")
                != "LABEL_STATE_CLAIMED_NO_RETRY_OR_TOP_UP"
                or marker.get("run_contract_sha256") != run_contract_sha256
                or marker.get("worker") != spec.to_dict()
                or marker.get("state_index") != index
                or marker.get("retry_count") != 0
                or marker.get("top_up_count") != 0
            ):
                raise ValueError(f"label attempt marker {index} drifted")
    validate_worker_runtime_pair(runtimes)
    return runtimes


def _inventory(files: Mapping[str, bytes]) -> tuple[tuple[Mapping[str, Any], ...], str]:
    records = tuple(
        {
            "path": name,
            "sha256": sha256_bytes(files[name]),
            "size_bytes": len(files[name]),
        }
        for name in sorted(files)
    )
    return records, sha256_bytes(canonical_json_bytes(records))


def _expected_aggregate_fields(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    repeat_values = [float(record["reference_repeat_kl"]) for record in records]
    baseline_distances = [
        float(record["baseline_summary_only_distance"]) for record in records
    ]
    oracle_utilities = [
        float(record["primary_exact_subset_oracle"]["utility"])
        for record in records
    ]
    normalized = [
        utility / baseline
        for utility, baseline in zip(
            oracle_utilities,
            baseline_distances,
            strict=True,
        )
        if baseline > NORMALIZATION_EPSILON
    ]
    interactions = [
        float(item["interaction"])
        for record in records
        for item in record["pair_interactions"]
    ]
    return {
        "role_state_counts": dict(
            sorted(Counter(record["state"]["role"] for record in records).items())
        ),
        "reference_repeat_kl": {
            "maximum": max(repeat_values),
            "mean": math.fsum(repeat_values) / len(repeat_values),
            "threshold": MAXIMUM_REPEAT_KL,
        },
        "scientific_summary": {
            "positive_baseline_distance_state_count": sum(
                value > NORMALIZATION_EPSILON for value in baseline_distances
            ),
            "mean_summary_only_distance": math.fsum(baseline_distances)
            / len(baseline_distances),
            "mean_primary_exact_oracle_utility": math.fsum(oracle_utilities)
            / len(oracle_utilities),
            "mean_primary_exact_oracle_normalized_recovery": (
                math.fsum(normalized) / len(normalized) if normalized else None
            ),
            "positive_pair_interaction_count": sum(value > 0 for value in interactions),
            "negative_pair_interaction_count": sum(value < 0 for value in interactions),
            "zero_pair_interaction_count": sum(value == 0 for value in interactions),
        },
    }


def validate_label_evidence_files(files: Mapping[str, bytes]) -> LabelEvidence:
    if not isinstance(files, Mapping):
        raise TypeError("label evidence files must be a mapping")
    normalized = {_safe_member_name(name): payload for name, payload in files.items()}
    if any(not isinstance(payload, bytes) for payload in normalized.values()):
        raise TypeError("label evidence payloads must be bytes")
    expected = expected_file_names()
    if set(normalized) != expected:
        raise ValueError(
            "label evidence file inventory drifted; "
            f"missing={sorted(expected - set(normalized))}, "
            f"extra={sorted(set(normalized) - expected)}"
        )
    manifest = _json(normalized, RUN_MANIFEST_FILENAME)
    run_contract = manifest.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise ValueError("label run manifest lacks a run contract")
    run_contract_sha256 = sha256_bytes(canonical_json_bytes(run_contract))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "RESTORATION_V2_2_EAGER_LABEL_ATTEMPT"
        or manifest.get("run_contract_sha256") != run_contract_sha256
    ):
        raise ValueError("label run contract identity drifted")
    profile, source_git_commit, projections, parents = _validate_run_contract(
        run_contract
    )
    global_ledger = _json(normalized, GLOBAL_LEDGER_MEMBER)
    aggregate = _json(normalized, AGGREGATE_FILENAME)
    _validate_profile_envelope(
        manifest,
        profile=profile,
        label="label run manifest",
    )
    _validate_profile_envelope(
        global_ledger,
        profile=profile,
        label="label global ledger",
    )
    _validate_profile_envelope(
        aggregate,
        profile=profile,
        label="label aggregate",
    )
    sibling_paths = global_ledger.get("worker_sibling_ledgers")
    if (
        global_ledger.get("schema_version") != SCHEMA_VERSION
        or global_ledger.get("protocol_id") != PROTOCOL_ID
        or global_ledger.get("status") != "LABEL_ATTEMPT_COMPLETED"
        or global_ledger.get("attempt_id") != profile.attempt_id
        or global_ledger.get("outcome") != profile.pass_outcome
        or global_ledger.get("attempted_state_count") != EXPECTED_STATE_COUNT
        or global_ledger.get("completed_state_count") != EXPECTED_STATE_COUNT
        or global_ledger.get("retry_count") != 0
        or global_ledger.get("top_up_count") != 0
        or not isinstance(sibling_paths, Mapping)
        or set(sibling_paths) != {"even", "odd"}
        or any(
            not isinstance(path, str) or not Path(path).is_absolute()
            for path in sibling_paths.values()
        )
        or aggregate.get("schema_version") != SCHEMA_VERSION
        or aggregate.get("protocol_id") != PROTOCOL_ID
        or aggregate.get("status")
        != profile.aggregate_status
        or aggregate.get("outcome") != profile.pass_outcome
        or aggregate.get("fixed_state_denominator") != EXPECTED_STATE_COUNT
        or aggregate.get("confirm_role_used") is not False
        or aggregate.get("gate_training_performed") is not False
        or aggregate.get("matched_nll_evaluation_performed") is not False
        or aggregate.get("closed_loop_evaluation_performed") is not False
        or aggregate.get("run_contract_sha256") != run_contract_sha256
        or global_ledger.get("run_contract_sha256") != run_contract_sha256
    ):
        raise ValueError("label global or aggregate outcome drifted")
    runtimes = _validate_worker_evidence(
        normalized,
        run_contract_sha256=run_contract_sha256,
        profile=profile,
    )
    specs = {spec.worker_id: spec for spec in expected_worker_specs()}
    records = []
    operation_totals: Counter[str] = Counter()
    for index in range(EXPECTED_STATE_COUNT):
        worker = "even" if index % 2 == 0 else "odd"
        record = _json(normalized, f"workers/{worker}/states/{index:03d}.json")
        counts = _validate_state(
            record,
            index=index,
            projection=projections[index],
            parent=parents[index],
            run_contract_sha256=run_contract_sha256,
            spec=specs[worker],
            runtime_metadata=runtimes[worker],
            profile=profile,
        )
        records.append(record)
        for key, value in counts.items():
            if type(value) is not int or value < 0:
                raise ValueError("label state operation count is invalid")
            operation_totals[key] += value
    observed_operation_totals = dict(sorted(operation_totals.items()))
    if aggregate.get("operation_counts") != observed_operation_totals:
        raise ValueError("label aggregate operation counts differ from raw states")
    if observed_operation_totals != _EXPECTED_OPERATION_TOTALS:
        raise ValueError("label aggregate exact operation schedule drifted")
    expected_aggregate = _expected_aggregate_fields(records)
    if any(
        aggregate.get(key) != value for key, value in expected_aggregate.items()
    ):
        raise ValueError("label aggregate scientific reduction drifted")
    inventory, tree_sha256 = _inventory(normalized)
    return LabelEvidence(
        files=normalized,
        source_git_commit=source_git_commit,
        run_contract_sha256=run_contract_sha256,
        outcome=profile.pass_outcome,
        aggregate=aggregate,
        inventory=inventory,
        tree_inventory_sha256=tree_sha256,
        profile=profile,
    )


def collect_label_evidence(
    raw_output_dir: str | Path,
    global_ledger: str | Path,
) -> LabelEvidence:
    root = Path(raw_output_dir).resolve()
    ledger = Path(global_ledger).resolve()
    files: dict[str, bytes] = {GLOBAL_LEDGER_MEMBER: ledger.read_bytes()}
    global_value = strict_json_object_bytes(files[GLOBAL_LEDGER_MEMBER], label="global ledger")
    siblings = global_value.get("worker_sibling_ledgers")
    if not isinstance(siblings, Mapping):
        raise ValueError("global label ledger lacks sibling paths")
    for worker in ("even", "odd"):
        files[f"{SIBLING_PREFIX}/{worker}.json"] = Path(siblings[worker]).read_bytes()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("label evidence cannot contain symlinks")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return validate_label_evidence_files(files)


def deterministic_tar_bytes(
    files: Mapping[str, bytes],
    *,
    profile: RestorationLabelAttemptProfile = V1_ATTEMPT_PROFILE,
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for relative in sorted(files):
            name = f"{profile.attempt_id}/{relative}"
            payload = files[relative]
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def package_label_evidence(
    *,
    raw_output_dir: str | Path,
    global_ledger: str | Path,
    output_archive: str | Path,
) -> dict[str, Any]:
    evidence = collect_label_evidence(raw_output_dir, global_ledger)
    output = Path(output_archive)
    if output.resolve() != evidence.profile.archive_path:
        raise ValueError("label archive output differs from the attempt profile")
    payload = deterministic_tar_bytes(evidence.files, profile=evidence.profile)
    with output.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    return {
        "status": "PACKAGED_RESTORATION_V2_2_EAGER_LABEL_EVIDENCE",
        "outcome": evidence.outcome,
        "output": str(output.resolve()),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "file_count": len(evidence.files),
        "tree_inventory_sha256": evidence.tree_inventory_sha256,
    }


def read_label_evidence_archive(path: str | Path) -> LabelEvidence:
    files: dict[str, bytes] = {}
    profile: RestorationLabelAttemptProfile | None = None
    with tarfile.open(path, mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                raise ValueError("label archive contains a non-regular member")
            member_prefix = PurePosixPath(member.name).parts[0]
            observed_profile = label_attempt_profile_for_id(member_prefix)
            if profile is None:
                profile = observed_profile
            elif profile != observed_profile:
                raise ValueError("label archive mixes attempt prefixes")
            prefix = f"{profile.attempt_id}/"
            if not member.name.startswith(prefix):
                raise ValueError("label archive member prefix drifted")
            relative = member.name[len(prefix) :]
            if relative in files:
                raise ValueError("label archive contains a duplicate member")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("label archive member cannot be read")
            files[relative] = source.read()
    if profile is None:
        raise ValueError("label archive is empty")
    evidence = validate_label_evidence_files(files)
    if evidence.profile != profile:
        raise ValueError("label archive prefix differs from embedded attempt profile")
    if deterministic_tar_bytes(
        evidence.files,
        profile=evidence.profile,
    ) != Path(path).read_bytes():
        raise ValueError("label archive is not canonical deterministic USTAR")
    return evidence


def build_artifact_manifest(
    *,
    source_archive: str | Path,
    fresh_immutable_archive: str | Path,
    hf_revision: str,
) -> dict[str, Any]:
    if _GIT_SHA.fullmatch(hf_revision) is None:
        raise ValueError("HF immutable revision must be a full commit SHA")
    source = Path(source_archive).resolve()
    fresh = Path(fresh_immutable_archive).resolve()
    if source == fresh:
        raise ValueError("fresh immutable archive must use an independent path")
    evidence = read_label_evidence_archive(source)
    read_label_evidence_archive(fresh)
    source_payload = source.read_bytes()
    fresh_payload = fresh.read_bytes()
    if source_payload != fresh_payload:
        raise ValueError("fresh immutable label archive differs from source bytes")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "attempt_id": evidence.profile.attempt_id,
        "attempt_revision": evidence.profile.attempt_revision,
        "status": "VERIFIED_RESTORATION_V2_2_EAGER_LABEL_ARTIFACT",
        "result": {
            "outcome": evidence.outcome,
            "fixed_state_denominator": EXPECTED_STATE_COUNT,
            "raw_distance_row_count": EXPECTED_DISTANCE_ROWS,
            "deployment_conditional_label_count": EXPECTED_DEPLOYMENT_EDGES,
            "primary_exact_subset_oracle_count": EXPECTED_PRIMARY_ORACLES,
        },
        "source_execution": {
            "source_git_commit": evidence.source_git_commit,
            "run_contract_sha256": evidence.run_contract_sha256,
        },
        "raw_archive": {
            "format": ARCHIVE_FORMAT,
            "sha256": sha256_bytes(source_payload),
            "size_bytes": len(source_payload),
            "file_count": len(evidence.files),
            "tree_inventory_sha256": evidence.tree_inventory_sha256,
        },
        "hf_artifact": {
            "repo": evidence.profile.hf_repo,
            "immutable_revision": hf_revision,
            "path": evidence.profile.hf_path,
            "fresh_immutable_download_verified": True,
            "fresh_archive_sha256": sha256_bytes(fresh_payload),
            "fresh_archive_size_bytes": len(fresh_payload),
        },
    }
