"""Deterministic selection and exposure records for restoration-v2."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

from causalcache.data.guiodyssey import build_pilot_manifest
from causalcache.data.guiodyssey_independent import (
    Candidate,
    canonical_json_bytes,
    inspect_candidate,
    select_splits,
    source_file_specs,
    verify_local_source_files,
)
from causalcache.restoration_v2_contract import FROZEN_RESTORATION_V2_SHA256


SELECTION_SCHEMA_VERSION = "0.1.0"
EXPOSURE_SCHEMA_VERSION = "0.1.0"
STATE_ID_TEMPLATE = "{source_id}:decision_step:{decision_step_id:03d}"
EXPOSURE_ROLE_NAMES = (
    "v1_reference_contract_audit_only",
    "v2_label_train",
    "v2_development",
    "v2_confirm_primary",
)
V1_POLICY_REPO = "ByteDance-Seed/UI-TARS-1.5-7B"
V1_POLICY_REVISION = "683d002dd99d8f95104d31e70391a39348857f4e"
V1_SUMMARY_PATH = "data/results/independent_reference_gate_v1/summary.json"
V1_SUMMARY_SHA256 = "c49d79dff67aabb0867bda56836f7337201a5015ba618160c45e8dd011b64c8e"
V1_OUTPUT_HF_REPO = "gavinlaw/causalcache-guiodyssey-independent-mobile"
V1_OUTPUT_HF_REVISION = "b3e1245c6c6a1723fe2ca3a861148008df39df46"
V1_OUTPUT_HF_PATH = "runs/independent-reference-gate-v1"
V1_CONFIG_CURRENT_SHA256 = "7b62aa31c80536f28bc4e8a3d684ff535bc2315d44a31cb5f002ed6dcb493fd8"
V1_CONFIG_BUILD_SHA256 = "40b00e0a8da9e4f573ffcea2447c5e569c795c1e980c5889cb1ec486822b2a95"
SOURCE_FILE_MANIFEST_SHA256 = "46f2240a07f46b6e283cb90e1f478f0b14a2e1499c52e2cc8e144898065adffc"
SOURCE_REPO = "cua-lite/GUIOdyssey"
SOURCE_REVISION = "ea08072b30e523fb4492e4f4597505879ffcd63b"
SOURCE_FILE_COUNT = 16
SOURCE_TOTAL_BYTES = 2_252_923_738
SOURCE_TOTAL_ROWS = 212
SOURCE_ELIGIBLE_COUNT = 111
SOURCE_EXCLUSION_COUNTS = {
    "decision_count_above_maximum": 81,
    "decision_count_below_minimum": 1,
    "excluded_source_id": 1,
    "invalid_executable_action": 6,
    "terminal_status_mismatch": 12,
}


@dataclass(frozen=True)
class PoolReconstruction:
    candidates: tuple[Candidate, ...]
    total_source_rows: int
    source_row_counts: Mapping[str, int]
    exclusion_counts: Mapping[str, int]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _observation_path_matches(path: Any, *, source_id: str, index: int) -> bool:
    if not isinstance(path, str):
        return False
    expected = rf"images/{re.escape(source_id)}/observation-{index:03d}\.(png|jpg|webp)"
    return re.fullmatch(expected, path) is not None


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def state_id(source_id: str, decision_step_id: int) -> str:
    if not source_id or decision_step_id < 2:
        raise ValueError("state identity requires a source_id and decision_step_id >= 2")
    return STATE_ID_TEMPLATE.format(
        source_id=source_id,
        decision_step_id=decision_step_id,
    )


def available_decision_step_ids(candidate: Candidate) -> tuple[int, ...]:
    return tuple(range(2, candidate.decision_count + 2))


def _iter_rows(parquet_file: Any) -> Iterator[tuple[int, Mapping[str, Any]]]:
    row_index = 0
    for row_group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_index)
        for row in table.to_pylist():
            yield row_index, row
            row_index += 1
    if row_index != parquet_file.metadata.num_rows:
        raise RuntimeError("Parquet row iteration did not match metadata.num_rows")


def _source_path(source_root: Path, transport_file: str) -> Path:
    return source_root.joinpath(*PurePosixPath(transport_file).parts)


def reconstruct_eligible_pool(
    *,
    source_root: Path,
    v1_config: Mapping[str, Any],
    source_file_manifest: Mapping[str, Any],
) -> PoolReconstruction:
    specs = source_file_specs(v1_config, source_file_manifest)
    verify_local_source_files(source_root, specs)

    import pyarrow.parquet as pq

    candidates: list[Candidate] = []
    exclusion_counts: Counter[str] = Counter()
    source_row_counts: dict[str, int] = {}
    seen_source_ids: dict[str, tuple[str, int]] = {}
    total_source_rows = 0
    for spec in specs:
        parquet = pq.ParquetFile(_source_path(source_root, spec.transport_file))
        row_count = int(parquet.metadata.num_rows)
        source_row_counts[spec.transport_file] = row_count
        total_source_rows += row_count
        for row_index, row in _iter_rows(parquet):
            inspection = inspect_candidate(
                row,
                transport_file=spec.transport_file,
                transport_row_index=row_index,
                config=v1_config,
            )
            if inspection.source_id is not None and bool(
                v1_config["eligibility"]["require_unique_source_id"]
            ):
                previous = seen_source_ids.get(inspection.source_id)
                if previous is not None:
                    raise ValueError(
                        "duplicate source_id across frozen source pool: "
                        f"{inspection.source_id} at {previous} and "
                        f"{(spec.transport_file, row_index)}"
                    )
                seen_source_ids[inspection.source_id] = (
                    spec.transport_file,
                    row_index,
                )
            if inspection.candidate is None:
                if inspection.exclusion_reason is None:
                    raise RuntimeError("excluded row is missing its reason")
                exclusion_counts[inspection.exclusion_reason.value] += 1
            else:
                candidates.append(inspection.candidate)

    if total_source_rows != sum(source_row_counts.values()):
        raise RuntimeError("source row accounting mismatch")
    return PoolReconstruction(
        candidates=tuple(candidates),
        total_source_rows=total_source_rows,
        source_row_counts=dict(sorted(source_row_counts.items())),
        exclusion_counts=dict(sorted(exclusion_counts.items())),
    )


def _candidate_record(
    candidate: Candidate,
    *,
    eligible_order_index: int,
) -> dict[str, Any]:
    return {
        **candidate.pool_record(),
        "eligible_order_index": eligible_order_index,
    }


def _parent_candidate_check(
    candidate: Candidate,
    parent_trajectory: Mapping[str, Any],
) -> None:
    expected = {
        "source_id": candidate.source_id,
        "transport_file": candidate.transport_file,
        "transport_row_index": candidate.transport_row_index,
        "selection_sha256": candidate.selection_sha256,
        "decision_count": candidate.decision_count,
        "normalized_app_labels": list(candidate.normalized_app_labels),
    }
    actual = {
        "source_id": parent_trajectory.get("source_id"),
        "transport_file": parent_trajectory.get("transport_file"),
        "transport_row_index": parent_trajectory.get("transport_row_index"),
        "selection_sha256": parent_trajectory.get("selection_sha256"),
        "decision_count": len(parent_trajectory.get("decisions", [])),
        "normalized_app_labels": parent_trajectory.get("normalized_app_labels"),
    }
    if actual != expected:
        raise ValueError(f"parent trajectory changed for {candidate.source_id}")


def _role_trajectory_records(
    candidates: Sequence[Candidate],
    *,
    eligible_indices: Mapping[str, int],
) -> list[dict[str, Any]]:
    return [
        _candidate_record(
            candidate,
            eligible_order_index=eligible_indices[candidate.source_id],
        )
        for candidate in candidates
    ]


def _requested_state_steps(
    candidates: Sequence[Candidate],
    requested: Sequence[int],
) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for candidate in candidates:
        available = set(available_decision_step_ids(candidate))
        result[candidate.source_id] = tuple(
            int(step) for step in requested if int(step) in available
        )
    return result


def _role_state_skeletons(
    candidates: Sequence[Candidate],
    *,
    state_steps: Mapping[str, Sequence[int]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        for step in state_steps[candidate.source_id]:
            history = list(range(1, int(step)))
            current_equivalent = int(step) - 1
            records.append(
                {
                    "state_id": state_id(candidate.source_id, int(step)),
                    "source_id": candidate.source_id,
                    "decision_step_id": int(step),
                    "history_event_step_ids": history,
                    "current_equivalent_event_step_id": current_equivalent,
                    "candidate_event_step_ids": history[:-1],
                }
            )
    return records


def _canonical_id_digest(source_ids: Sequence[str]) -> str:
    return sha256_bytes(canonical_json_bytes(list(source_ids)))


def _validate_parent_manifest_identity(
    *,
    parent_manifest: Mapping[str, Any],
    parent_manifest_sha256: str,
    v2_contract: Mapping[str, Any],
) -> None:
    expected = v2_contract["data"]["parent_artifact"]
    if parent_manifest_sha256 != expected["manifest_sha256"]:
        raise ValueError("parent manifest SHA256 mismatch")
    if parent_manifest.get("dataset_repo") != expected["repo"]:
        raise ValueError("parent dataset repo mismatch")
    selection = parent_manifest.get("selection", {})
    if selection.get("eligible_pool_sha256") != expected["eligible_pool_sha256"]:
        raise ValueError("parent eligible-pool SHA256 mismatch")


def build_selection_manifest(
    *,
    reconstruction: PoolReconstruction,
    v1_config: Mapping[str, Any],
    v1_config_sha256: str,
    v2_contract: Mapping[str, Any],
    v2_contract_sha256: str,
    parent_manifest: Mapping[str, Any],
    parent_manifest_sha256: str,
    source_file_manifest: Mapping[str, Any],
    source_file_manifest_sha256: str,
    generator: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_parent_manifest_identity(
        parent_manifest=parent_manifest,
        parent_manifest_sha256=parent_manifest_sha256,
        v2_contract=v2_contract,
    )
    if source_file_manifest_sha256 != parent_manifest["source"][
        "source_file_manifest_sha256"
    ]:
        raise ValueError("source-file manifest SHA256 differs from parent build")
    selected = select_splits(reconstruction.candidates, config=v1_config)
    expected_parent = v2_contract["data"]["parent_artifact"]
    if selected.eligible_pool_sha256 != expected_parent["eligible_pool_sha256"]:
        raise ValueError("reconstructed eligible-pool SHA256 mismatch")
    if len(selected.eligible_candidates) != int(
        parent_manifest["selection"]["eligible_pool_count"]
    ):
        raise ValueError("reconstructed eligible-pool count mismatch")
    if reconstruction.total_source_rows != int(parent_manifest["selection"]["total_source_rows"]):
        raise ValueError("reconstructed total source-row count mismatch")
    if dict(reconstruction.exclusion_counts) != dict(
        parent_manifest["selection"]["exclusion_counts"]
    ):
        raise ValueError("reconstructed exclusion counts mismatch")

    roles = v2_contract["data"]["roles"]
    expected_reference_ids = tuple(
        roles["v1_reference_contract_audit_only"]["source_ids"]
    )
    expected_train_ids = tuple(roles["v2_label_train"]["source_ids"])
    expected_development_ids = tuple(roles["v2_development"]["source_ids"])
    expected_oracle_ids = expected_train_ids + expected_development_ids
    reference_ids = tuple(candidate.source_id for candidate in selected.reference_gate)
    oracle_ids = tuple(candidate.source_id for candidate in selected.oracle_pilot)
    if reference_ids != expected_reference_ids:
        raise ValueError("reconstructed v1 reference IDs or order mismatch")
    if oracle_ids != expected_oracle_ids:
        raise ValueError("reconstructed v1 oracle IDs or order mismatch")
    if tuple(parent_manifest["splits"]["reference_gate"]["source_ids"]) != reference_ids:
        raise ValueError("parent reference IDs or order mismatch")
    if tuple(parent_manifest["splits"]["oracle_pilot"]["source_ids"]) != oracle_ids:
        raise ValueError("parent oracle IDs or order mismatch")

    candidate_by_id = {
        candidate.source_id: candidate for candidate in selected.eligible_candidates
    }
    if len(candidate_by_id) != len(selected.eligible_candidates):
        raise ValueError("eligible source IDs must be unique")
    parent_by_id = {
        str(trajectory["source_id"]): trajectory
        for trajectory in parent_manifest["trajectories"]
    }
    for source_id in reference_ids + oracle_ids:
        if source_id not in parent_by_id:
            raise ValueError(f"parent manifest is missing selected trajectory {source_id}")
        _parent_candidate_check(candidate_by_id[source_id], parent_by_id[source_id])

    excluded_ids = set(reference_ids + oracle_ids)
    remaining = tuple(
        candidate
        for candidate in selected.eligible_candidates
        if candidate.source_id not in excluded_ids
    )
    structural = tuple(
        candidate for candidate in remaining if candidate.decision_count >= 5
    )
    confirm_count = int(roles["v2_confirm_primary"]["trajectory_count"])
    if len(structural) < confirm_count:
        raise ValueError("INVALID_BEFORE_POLICY_OUTPUT: fewer than 20 structural candidates")
    confirm = structural[:confirm_count]
    distinct_apps = sorted(
        {
            app
            for candidate in confirm
            for app in candidate.normalized_app_labels
        }
    )
    minimum_apps = int(roles["v2_confirm_primary"]["minimum_distinct_app_labels"])
    if len(distinct_apps) < minimum_apps:
        raise ValueError(
            "INVALID_BEFORE_POLICY_OUTPUT: frozen first-20 prefix lacks app diversity"
        )
    confirm_step = int(roles["v2_confirm_primary"]["decision_step_id"])
    if any(confirm_step not in available_decision_step_ids(candidate) for candidate in confirm):
        raise ValueError("INVALID_BEFORE_POLICY_OUTPUT: confirm step 6 is unavailable")

    train = tuple(candidate_by_id[source_id] for source_id in expected_train_ids)
    development = tuple(
        candidate_by_id[source_id] for source_id in expected_development_ids
    )
    eligible_indices = {
        candidate.source_id: index
        for index, candidate in enumerate(selected.eligible_candidates)
    }
    structural_indices = {
        candidate.source_id: index for index, candidate in enumerate(structural)
    }
    train_steps = _requested_state_steps(
        train,
        roles["v2_label_train"]["state_decision_step_ids"],
    )
    development_steps = _requested_state_steps(
        development,
        roles["v2_development"]["state_decision_step_ids"],
    )
    confirm_steps = {
        candidate.source_id: (confirm_step,) for candidate in confirm
    }

    role_id_sets = {
        "v1_reference_contract_audit_only": reference_ids,
        "v1_oracle_pilot": oracle_ids,
        "v2_label_train": expected_train_ids,
        "v2_development": expected_development_ids,
        "v2_confirm_primary": tuple(candidate.source_id for candidate in confirm),
    }
    disjoint_groups = (
        "v1_reference_contract_audit_only",
        "v1_oracle_pilot",
        "v2_confirm_primary",
    )
    intersections: dict[str, list[str]] = {}
    for left_index, left in enumerate(disjoint_groups):
        for right in disjoint_groups[left_index + 1 :]:
            intersections[f"{left}__{right}"] = sorted(
                set(role_id_sets[left]).intersection(role_id_sets[right])
            )
    if any(intersections.values()):
        raise ValueError("reference, oracle, and confirm roles must be disjoint")
    union_ids = sorted(
        set(reference_ids).union(oracle_ids).union(role_id_sets["v2_confirm_primary"])
    )

    eligible_pool_records = [
        candidate.pool_record() for candidate in selected.eligible_candidates
    ]
    if sha256_bytes(canonical_json_bytes(eligible_pool_records)) != selected.eligible_pool_sha256:
        raise RuntimeError("eligible-pool record serialization drifted")
    source_files = source_file_manifest["files"]
    manifest = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "protocol_id": "causalcache_restoration_v2",
        "status": "PASSED_PREOUTPUT_SELECTION",
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "inputs": {
            "scientific_contract": {
                "path": "code/configs/causalcache_restoration_v2.json",
                "sha256": v2_contract_sha256,
            },
            "v1_selection_config": {
                "path": "code/configs/independent_reference_gate_v1.json",
                "current_sha256": v1_config_sha256,
                "build_sha256": parent_manifest["source"]["protocol_config_sha256"],
            },
            "parent_artifact": {
                "repo": expected_parent["repo"],
                "revision": expected_parent["revision"],
                "manifest_sha256": parent_manifest_sha256,
                "eligible_pool_count": len(selected.eligible_candidates),
                "eligible_pool_sha256": selected.eligible_pool_sha256,
            },
            "source_file_manifest": {
                "path": "data/manifests/independent_reference_gate_v1_source_files.json",
                "sha256": source_file_manifest_sha256,
                "repo": source_file_manifest["repo"],
                "revision": source_file_manifest["revision"],
                "file_count": len(source_files),
                "total_bytes": source_file_manifest["total_bytes"],
                "files": source_files,
            },
        },
        "generator": dict(generator),
        "reconstruction": {
            "total_source_rows": reconstruction.total_source_rows,
            "source_row_counts": dict(reconstruction.source_row_counts),
            "eligible_pool_count": len(selected.eligible_candidates),
            "eligible_pool_sha256": selected.eligible_pool_sha256,
            "exclusion_counts": dict(reconstruction.exclusion_counts),
            "trajectory_salt": v1_config["selection"]["trajectory_salt"],
            "trajectory_hash_input": v1_config["selection"]["trajectory_hash_input"],
            "trajectory_order": v1_config["selection"]["trajectory_order"],
            "eligible_pool": eligible_pool_records,
        },
        "confirm_selection_proof": {
            "excluded_prior_source_ids": list(reference_ids + oracle_ids),
            "excluded_prior_source_ids_sha256": _canonical_id_digest(
                reference_ids + oracle_ids
            ),
            "remaining_eligible_count": len(remaining),
            "structural_filter": "decision_count>=5",
            "structurally_eligible_count": len(structural),
            "selection": "first_20_then_check_app_diversity_without_top_up",
            "selected_source_ids": list(role_id_sets["v2_confirm_primary"]),
            "selected_global_order_indices": [
                eligible_indices[candidate.source_id] for candidate in confirm
            ],
            "selected_structural_order_indices": [
                structural_indices[candidate.source_id] for candidate in confirm
            ],
            "decision_step_id": confirm_step,
            "distinct_app_labels": distinct_apps,
            "distinct_app_label_count": len(distinct_apps),
            "minimum_distinct_app_labels": minimum_apps,
            "top_up_allowed": False,
        },
        "roles": {
            "v1_reference_contract_audit_only": {
                "trajectories": _role_trajectory_records(
                    selected.reference_gate,
                    eligible_indices=eligible_indices,
                ),
                "states": [],
            },
            "v2_label_train": {
                "trajectories": _role_trajectory_records(
                    train,
                    eligible_indices=eligible_indices,
                ),
                "states": _role_state_skeletons(train, state_steps=train_steps),
            },
            "v2_development": {
                "trajectories": _role_trajectory_records(
                    development,
                    eligible_indices=eligible_indices,
                ),
                "states": _role_state_skeletons(
                    development,
                    state_steps=development_steps,
                ),
            },
            "v2_confirm_primary": {
                "trajectories": _role_trajectory_records(
                    confirm,
                    eligible_indices=eligible_indices,
                ),
                "states": _role_state_skeletons(
                    confirm,
                    state_steps=confirm_steps,
                ),
            },
        },
        "overlap_proof": {
            "ordered_role_source_ids": {
                key: list(value) for key, value in role_id_sets.items()
            },
            "pairwise_intersections": intersections,
            "reference_oracle_confirm_union_count": len(union_ids),
            "reference_oracle_confirm_union_sha256": _canonical_id_digest(union_ids),
        },
    }
    validate_selection_manifest(manifest, v2_contract=v2_contract)
    return manifest


def _selected_candidates(manifest: Mapping[str, Any]) -> tuple[Candidate, ...]:
    records = manifest["reconstruction"]["eligible_pool"]
    return tuple(
        Candidate(
            source_id=str(record["source_id"]),
            transport_file=str(record["transport_file"]),
            transport_row_index=int(record["transport_row_index"]),
            selection_sha256=str(record["selection_sha256"]),
            decision_count=int(record["decision_count"]),
            normalized_app_labels=tuple(record["normalized_app_labels"]),
            action_type_counts=tuple(
                sorted(
                    (str(key), int(value))
                    for key, value in record["action_type_counts"].items()
                )
            ),
        )
        for record in records
    )


def validate_selection_manifest(
    manifest: Mapping[str, Any],
    *,
    v2_contract: Mapping[str, Any],
) -> None:
    if manifest.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("selection schema_version mismatch")
    if manifest.get("protocol_id") != "causalcache_restoration_v2":
        raise ValueError("selection protocol_id mismatch")
    if manifest.get("status") != "PASSED_PREOUTPUT_SELECTION":
        raise ValueError("selection status must be PASSED_PREOUTPUT_SELECTION")
    if manifest.get("policy_output_generated") is not False:
        raise ValueError("selection must precede policy output")
    if manifest.get("restoration_output_generated") is not False:
        raise ValueError("selection must precede restoration output")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("selection inputs must be an object")
    if inputs.get("scientific_contract") != {
        "path": "code/configs/causalcache_restoration_v2.json",
        "sha256": FROZEN_RESTORATION_V2_SHA256,
    }:
        raise ValueError("selection scientific-contract identity mismatch")
    if inputs.get("v1_selection_config") != {
        "path": "code/configs/independent_reference_gate_v1.json",
        "current_sha256": V1_CONFIG_CURRENT_SHA256,
        "build_sha256": V1_CONFIG_BUILD_SHA256,
    }:
        raise ValueError("selection v1-config identity mismatch")
    expected_parent = v2_contract["data"]["parent_artifact"]
    if inputs.get("parent_artifact") != {
        "repo": expected_parent["repo"],
        "revision": expected_parent["revision"],
        "manifest_sha256": expected_parent["manifest_sha256"],
        "eligible_pool_count": SOURCE_ELIGIBLE_COUNT,
        "eligible_pool_sha256": expected_parent["eligible_pool_sha256"],
    }:
        raise ValueError("selection parent-artifact identity mismatch")
    source_input = inputs.get("source_file_manifest")
    if not isinstance(source_input, Mapping):
        raise ValueError("selection source-file identity must be an object")
    expected_source_identity = {
        "path": "data/manifests/independent_reference_gate_v1_source_files.json",
        "sha256": SOURCE_FILE_MANIFEST_SHA256,
        "repo": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "file_count": SOURCE_FILE_COUNT,
        "total_bytes": SOURCE_TOTAL_BYTES,
    }
    if any(source_input.get(key) != value for key, value in expected_source_identity.items()):
        raise ValueError("selection source-file identity mismatch")
    files = source_input.get("files")
    if not isinstance(files, list) or len(files) != SOURCE_FILE_COUNT:
        raise ValueError("selection source-file inventory mismatch")
    if sum(int(record["size"]) for record in files) != SOURCE_TOTAL_BYTES:
        raise ValueError("selection source-file byte total mismatch")
    generator = manifest.get("generator")
    expected_generator_paths = {
        "module_path": "code/causalcache/data/restoration_v2_selection.py",
        "cli_path": "code/scripts/materialize_restoration_v2_selection.py",
        "validator_path": "code/scripts/validate_restoration_v2_selection.py",
    }
    expected_generator_keys = {
        "git_revision",
        *expected_generator_paths,
        "module_sha256",
        "cli_sha256",
        "validator_sha256",
    }
    if not isinstance(generator, Mapping) or set(generator) != expected_generator_keys:
        raise ValueError("selection generator schema mismatch")
    if re.fullmatch(r"[0-9a-f]{40}", str(generator["git_revision"])) is None:
        raise ValueError("selection generator Git revision is invalid")
    if any(generator[key] != value for key, value in expected_generator_paths.items()):
        raise ValueError("selection generator source path mismatch")
    if any(
        not _is_sha256(generator[f"{prefix}_sha256"])
        for prefix in ("module", "cli", "validator")
    ):
        raise ValueError("selection generator source SHA256 is invalid")

    reconstruction = manifest["reconstruction"]
    candidates = _selected_candidates(manifest)
    pool_records = [candidate.pool_record() for candidate in candidates]
    pool_sha = sha256_bytes(canonical_json_bytes(pool_records))
    if len(candidates) != reconstruction["eligible_pool_count"]:
        raise ValueError("eligible-pool count is not self-consistent")
    if pool_sha != reconstruction["eligible_pool_sha256"]:
        raise ValueError("eligible-pool records do not reproduce their SHA256")
    if pool_sha != expected_parent["eligible_pool_sha256"]:
        raise ValueError("eligible-pool SHA256 differs from the scientific contract")
    if reconstruction.get("total_source_rows") != SOURCE_TOTAL_ROWS:
        raise ValueError("reconstructed source-row count mismatch")
    if reconstruction.get("eligible_pool_count") != SOURCE_ELIGIBLE_COUNT:
        raise ValueError("reconstructed eligible-pool count mismatch")
    if reconstruction.get("exclusion_counts") != SOURCE_EXCLUSION_COUNTS:
        raise ValueError("reconstructed exclusion counts mismatch")
    if tuple(candidate.selection_sha256 for candidate in candidates) != tuple(
        sorted(candidate.selection_sha256 for candidate in candidates)
    ):
        raise ValueError("eligible-pool records are not in frozen order")

    role_names = (
        "v1_reference_contract_audit_only",
        "v2_label_train",
        "v2_development",
        "v2_confirm_primary",
    )
    role_ids = {
        name: tuple(
            str(record["source_id"])
            for record in manifest["roles"][name]["trajectories"]
        )
        for name in role_names
    }
    contract_roles = v2_contract["data"]["roles"]
    if role_ids["v1_reference_contract_audit_only"] != tuple(
        contract_roles["v1_reference_contract_audit_only"]["source_ids"]
    ):
        raise ValueError("reference role IDs differ from the scientific contract")
    if role_ids["v2_label_train"] != tuple(
        contract_roles["v2_label_train"]["source_ids"]
    ):
        raise ValueError("label-train role IDs differ from the scientific contract")
    if role_ids["v2_development"] != tuple(
        contract_roles["v2_development"]["source_ids"]
    ):
        raise ValueError("development role IDs differ from the scientific contract")
    if len(role_ids["v2_confirm_primary"]) != 20:
        raise ValueError("confirm role must contain exactly 20 trajectories")

    candidate_by_id = {candidate.source_id: candidate for candidate in candidates}
    eligible_indices = {
        candidate.source_id: index for index, candidate in enumerate(candidates)
    }
    for role_name in role_names:
        for record in manifest["roles"][role_name]["trajectories"]:
            source_id = str(record["source_id"])
            if source_id not in candidate_by_id:
                raise ValueError("a selected role contains a non-eligible source ID")
            expected_record = _candidate_record(
                candidate_by_id[source_id],
                eligible_order_index=eligible_indices[source_id],
            )
            if record != expected_record and set(record) != set(expected_record).union(
                {"instruction_sha256"}
            ):
                raise ValueError("role trajectory provenance differs from eligible pool")
            if any(record.get(key) != value for key, value in expected_record.items()):
                raise ValueError("role trajectory provenance differs from eligible pool")
            instruction_sha = record.get("instruction_sha256")
            if instruction_sha is not None and not _is_sha256(instruction_sha):
                raise ValueError("trajectory instruction SHA256 is invalid")

    reference = set(role_ids["v1_reference_contract_audit_only"])
    oracle = set(role_ids["v2_label_train"] + role_ids["v2_development"])
    confirm = set(role_ids["v2_confirm_primary"])
    if reference & oracle or reference & confirm or oracle & confirm:
        raise ValueError("reference, oracle, and confirm roles overlap")
    if len(reference | oracle | confirm) != 43:
        raise ValueError("reference/oracle/confirm union must contain 43 trajectories")

    expected_state_counts = {
        "v1_reference_contract_audit_only": 0,
        "v2_label_train": 30,
        "v2_development": 15,
        "v2_confirm_primary": 20,
    }
    for role_name, expected_count in expected_state_counts.items():
        states = manifest["roles"][role_name]["states"]
        if len(states) != expected_count:
            raise ValueError(f"{role_name} state count must equal {expected_count}")
        seen_state_ids: set[str] = set()
        for state in states:
            if str(state["source_id"]) not in set(role_ids[role_name]):
                raise ValueError("state source ID does not belong to its role")
            expected_state_id = state_id(
                str(state["source_id"]),
                int(state["decision_step_id"]),
            )
            if state["state_id"] != expected_state_id:
                raise ValueError("state_id does not match source and decision step")
            if expected_state_id in seen_state_ids:
                raise ValueError("state IDs must be unique within each role")
            seen_state_ids.add(expected_state_id)
            history = list(range(1, int(state["decision_step_id"])))
            if state["history_event_step_ids"] != history:
                raise ValueError("state history event IDs are not canonical")
            if state["current_equivalent_event_step_id"] != history[-1]:
                raise ValueError("current-equivalent event must be the latest history event")
            if state["candidate_event_step_ids"] != history[:-1]:
                raise ValueError("candidate events must exclude the current-equivalent event")
        if role_name == "v2_confirm_primary" and any(
            state["decision_step_id"] != 6 for state in states
        ):
            raise ValueError("every confirm state must use decision step 6")
        if role_name in ("v2_label_train", "v2_development"):
            steps_by_source: dict[str, list[int]] = defaultdict(list)
            for state in states:
                steps_by_source[str(state["source_id"])].append(
                    int(state["decision_step_id"])
                )
            if any(steps != [4, 5, 6] for steps in steps_by_source.values()):
                raise ValueError("every screening trajectory must use steps 4, 5, and 6")

    proof = manifest["confirm_selection_proof"]
    expected_excluded = (
        role_ids["v1_reference_contract_audit_only"]
        + role_ids["v2_label_train"]
        + role_ids["v2_development"]
    )
    if tuple(proof["excluded_prior_source_ids"]) != expected_excluded:
        raise ValueError("confirm exclusion list differs from exact prior roles")
    if proof["excluded_prior_source_ids_sha256"] != _canonical_id_digest(
        expected_excluded
    ):
        raise ValueError("confirm exclusion-list digest mismatch")
    excluded = set(proof["excluded_prior_source_ids"])
    remaining = [candidate for candidate in candidates if candidate.source_id not in excluded]
    structural = [candidate for candidate in remaining if candidate.decision_count >= 5]
    expected_confirm = tuple(candidate.source_id for candidate in structural[:20])
    if role_ids["v2_confirm_primary"] != expected_confirm:
        raise ValueError("confirm role is not the frozen first-20 structural prefix")
    if proof["selected_source_ids"] != list(expected_confirm):
        raise ValueError("confirm proof selected IDs differ from the role")
    if proof["remaining_eligible_count"] != len(remaining):
        raise ValueError("confirm remaining-pool count mismatch")
    if proof["structurally_eligible_count"] != len(structural):
        raise ValueError("confirm structural-pool count mismatch")
    if proof["structural_filter"] != "decision_count>=5":
        raise ValueError("confirm structural filter drifted")
    if proof["selection"] != "first_20_then_check_app_diversity_without_top_up":
        raise ValueError("confirm prefix-selection rule drifted")
    if proof["selected_structural_order_indices"] != list(range(20)):
        raise ValueError("confirm selection is not the first structural prefix")
    expected_global_indices = [eligible_indices[source_id] for source_id in expected_confirm]
    if proof["selected_global_order_indices"] != expected_global_indices:
        raise ValueError("confirm global-order indices mismatch")
    apps = sorted(
        {
            app
            for candidate in structural[:20]
            for app in candidate.normalized_app_labels
        }
    )
    if apps != proof["distinct_app_labels"] or len(apps) < int(
        proof["minimum_distinct_app_labels"]
    ):
        raise ValueError("confirm prefix app-diversity proof mismatch")
    frozen_minimum_apps = int(
        contract_roles["v2_confirm_primary"]["minimum_distinct_app_labels"]
    )
    if proof["minimum_distinct_app_labels"] != frozen_minimum_apps:
        raise ValueError("confirm app-diversity minimum differs from the scientific contract")
    if proof["distinct_app_label_count"] != len(apps):
        raise ValueError("confirm app-diversity count mismatch")
    if proof.get("top_up_allowed") is not False:
        raise ValueError("confirm selection must forbid top-up")
    overlap = manifest["overlap_proof"]
    ordered_role_ids = overlap["ordered_role_source_ids"]
    expected_ordered_role_ids = {
        "v1_reference_contract_audit_only": list(
            role_ids["v1_reference_contract_audit_only"]
        ),
        "v1_oracle_pilot": list(
            role_ids["v2_label_train"] + role_ids["v2_development"]
        ),
        "v2_label_train": list(role_ids["v2_label_train"]),
        "v2_development": list(role_ids["v2_development"]),
        "v2_confirm_primary": list(role_ids["v2_confirm_primary"]),
    }
    if ordered_role_ids != expected_ordered_role_ids:
        raise ValueError("overlap proof role IDs mismatch")
    if any(overlap["pairwise_intersections"].values()):
        raise ValueError("overlap proof contains a non-empty intersection")
    union_ids = sorted(reference | oracle | confirm)
    if overlap["reference_oracle_confirm_union_count"] != len(union_ids):
        raise ValueError("overlap proof union count mismatch")
    if overlap["reference_oracle_confirm_union_sha256"] != _canonical_id_digest(
        union_ids
    ):
        raise ValueError("overlap proof union digest mismatch")


def _trajectory_rows(
    *,
    source_root: Path,
    source_file_manifest: Mapping[str, Any],
    v1_config: Mapping[str, Any],
    selected_candidates: Sequence[Candidate],
) -> dict[str, Mapping[str, Any]]:
    specs = source_file_specs(v1_config, source_file_manifest)
    candidates_by_key = {
        (candidate.transport_file, candidate.transport_row_index): candidate
        for candidate in selected_candidates
    }
    keys_by_file: dict[str, set[int]] = defaultdict(set)
    for transport_file, row_index in candidates_by_key:
        keys_by_file[transport_file].add(row_index)

    import pyarrow.parquet as pq

    rows: dict[str, Mapping[str, Any]] = {}
    for spec in specs:
        wanted = keys_by_file.get(spec.transport_file, set())
        if not wanted:
            continue
        parquet = pq.ParquetFile(_source_path(source_root, spec.transport_file))
        for row_index, row in _iter_rows(parquet):
            if row_index not in wanted:
                continue
            candidate = candidates_by_key[(spec.transport_file, row_index)]
            rows[candidate.source_id] = row
    if set(rows) != {candidate.source_id for candidate in selected_candidates}:
        raise RuntimeError("failed to reload every selected source row")
    return rows


def attach_state_content_witnesses(
    manifest: Mapping[str, Any],
    *,
    source_root: Path,
    source_file_manifest: Mapping[str, Any],
    v1_config: Mapping[str, Any],
    v2_contract: Mapping[str, Any],
) -> dict[str, Any]:
    updated = json.loads(json.dumps(manifest))
    eligible = {
        candidate.source_id: candidate for candidate in _selected_candidates(updated)
    }
    selected_ids = []
    for role_name in ("v2_label_train", "v2_development", "v2_confirm_primary"):
        selected_ids.extend(
            record["source_id"]
            for record in updated["roles"][role_name]["trajectories"]
        )
    selected_candidates = [eligible[source_id] for source_id in selected_ids]
    rows = _trajectory_rows(
        source_root=source_root,
        source_file_manifest=source_file_manifest,
        v1_config=v1_config,
        selected_candidates=selected_candidates,
    )
    file_sha = {
        str(record["path"]): str(record["sha256"])
        for record in source_file_manifest["files"]
    }
    for role_name in ("v2_label_train", "v2_development", "v2_confirm_primary"):
        states_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for state in updated["roles"][role_name]["states"]:
            states_by_source[str(state["source_id"])].append(state)
        for trajectory_record in updated["roles"][role_name]["trajectories"]:
            source_id = str(trajectory_record["source_id"])
            candidate = eligible[source_id]
            pilot, images = build_pilot_manifest(
                rows[source_id],
                row_index=candidate.transport_row_index,
                upstream_repo=str(v1_config["source_pool"]["upstream_repo"]),
                upstream_revision=str(v1_config["source_pool"]["upstream_revision"]),
                transport_repo=str(v1_config["source_pool"]["transport_repo"]),
                transport_revision=str(v1_config["source_pool"]["transport_revision"]),
                transport_file=candidate.transport_file,
                transport_file_sha256=file_sha[candidate.transport_file],
                hf_destination=str(v2_contract["data"]["derived_artifact"]["repo"]),
                grid_size=int(v1_config["policy"]["coordinate_grid_size"]),
            )
            trajectory = pilot["trajectory"]
            if trajectory["source_id"] != source_id:
                raise RuntimeError("reloaded trajectory source ID changed")
            trajectory_record["instruction_sha256"] = sha256_bytes(
                str(trajectory["instruction"]).encode("utf-8")
            )
            decisions = {
                int(decision["decision_step_id"]): decision
                for decision in trajectory["decisions"]
            }
            events = {
                int(event["step_id"]): event for event in trajectory["events"]
            }
            for state in states_by_source[source_id]:
                step = int(state["decision_step_id"])
                decision = decisions[step]
                current_path = str(decision["current_observation_path"])
                current = {
                    "member_path": current_path,
                    "sha256": sha256_bytes(images[current_path]),
                }
                candidate_events = []
                for event_step in state["candidate_event_step_ids"]:
                    path = str(events[int(event_step)]["observation_after_path"])
                    candidate_events.append(
                        {
                            "event_step_id": int(event_step),
                            "post_state_member_path": path,
                            "post_state_sha256": sha256_bytes(images[path]),
                        }
                    )
                current_equivalent_step = int(
                    state["current_equivalent_event_step_id"]
                )
                current_equivalent_path = str(
                    events[current_equivalent_step]["observation_after_path"]
                )
                current_equivalent = {
                    "event_step_id": current_equivalent_step,
                    "post_state_member_path": current_equivalent_path,
                    "post_state_sha256": sha256_bytes(images[current_equivalent_path]),
                }
                if (
                    current_equivalent["post_state_member_path"] != current["member_path"]
                    or current_equivalent["post_state_sha256"] != current["sha256"]
                ):
                    raise ValueError("current-equivalent event does not match current observation")
                state["current_observation"] = current
                state["candidate_event_post_states"] = candidate_events
                state["current_equivalence_witness"] = current_equivalent
                state["validated_action_sha256"] = sha256_bytes(
                    canonical_json_bytes(decision["validated_action"])
                )
    updated["content_witness_status"] = "COMPLETE_FOR_45_SCREENING_AND_20_CONFIRM_STATES"
    validate_selection_manifest(updated, v2_contract=v2_contract)
    validate_state_content_witnesses(updated)
    return updated


def validate_state_content_witnesses(manifest: Mapping[str, Any]) -> None:
    if manifest.get("content_witness_status") != (
        "COMPLETE_FOR_45_SCREENING_AND_20_CONFIRM_STATES"
    ):
        raise ValueError("state content witnesses are incomplete")
    for role_name in ("v2_label_train", "v2_development", "v2_confirm_primary"):
        for state in manifest["roles"][role_name]["states"]:
            current = state.get("current_observation")
            if not isinstance(current, Mapping):
                raise ValueError("state is missing its current-observation witness")
            source_id = str(state["source_id"])
            decision_step = int(state["decision_step_id"])
            if not _observation_path_matches(
                current.get("member_path"),
                source_id=source_id,
                index=decision_step - 1,
            ):
                raise ValueError("current-observation path does not match the source state")
            expected_candidates = state["candidate_event_step_ids"]
            candidates = state.get("candidate_event_post_states")
            if not isinstance(candidates, list) or [
                record["event_step_id"] for record in candidates
            ] != expected_candidates:
                raise ValueError("candidate post-state witnesses are incomplete")
            equivalent = state.get("current_equivalence_witness")
            if not isinstance(equivalent, Mapping):
                raise ValueError("state is missing its current-equivalence witness")
            if equivalent["event_step_id"] != state["current_equivalent_event_step_id"]:
                raise ValueError("current-equivalence witness step mismatch")
            if (
                equivalent["post_state_member_path"] != current["member_path"]
                or equivalent["post_state_sha256"] != current["sha256"]
            ):
                raise ValueError("current-equivalence witness content mismatch")
            candidate_paths: set[str] = set()
            for record in candidates:
                event_step = int(record["event_step_id"])
                path = record.get("post_state_member_path")
                if not _observation_path_matches(
                    path,
                    source_id=source_id,
                    index=event_step,
                ):
                    raise ValueError("candidate post-state path does not match its event")
                if path == current["member_path"]:
                    raise ValueError("candidate image path must differ from current observation")
                if path in candidate_paths:
                    raise ValueError("candidate post-state paths must be unique")
                candidate_paths.add(str(path))
            for record in [current, equivalent, *candidates]:
                digest = record.get("sha256", record.get("post_state_sha256"))
                if not _is_sha256(digest):
                    raise ValueError("image witness SHA256 is invalid")
            action_sha = state.get("validated_action_sha256")
            if not _is_sha256(action_sha):
                raise ValueError("validated-action witness SHA256 is invalid")


def build_exposure_ledger(
    selection_manifest: Mapping[str, Any],
    *,
    selection_manifest_sha256: str,
    v1_summary_sha256: str,
    v1_summary: Mapping[str, Any],
) -> dict[str, Any]:
    validate_state_content_witnesses(selection_manifest)
    role_ids = {
        name: [
            record["source_id"]
            for record in selection_manifest["roles"][name]["trajectories"]
        ]
        for name in EXPOSURE_ROLE_NAMES
    }
    reference_ids = set(role_ids["v1_reference_contract_audit_only"])
    if v1_summary.get("outcome") != "NO_GO_CURRENT_REFERENCE_STACK":
        raise ValueError("v1 exposure evidence must preserve the frozen negative outcome")
    if set(v1_summary.get("by_trajectory", {})) != reference_ids:
        raise ValueError("v1 policy-output evidence does not cover the exact reference role")
    if v1_summary.get("oracle_split_status") != (
        "not_evaluated_due_to_reference_gate_failure"
    ):
        raise ValueError("v1 exposure evidence unexpectedly evaluated the oracle split")

    def event(
        event_id: str,
        *,
        roles: Sequence[str],
        access_kind: str,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        ids = [source_id for role in roles for source_id in role_ids[role]]
        return {
            "event_id": event_id,
            "roles": list(roles),
            "source_ids_sha256": _canonical_id_digest(ids),
            "source_id_count": len(ids),
            "access_kind": access_kind,
            "evidence": dict(evidence),
        }

    parent_roles = EXPOSURE_ROLE_NAMES[:3]
    raw_roles = (
        *parent_roles,
        "v2_confirm_primary",
    )
    expert_roles = raw_roles
    events = [
        event(
            "parent-artifact-v1-raw-machine-access",
            roles=parent_roles,
            access_kind="raw_source_machine_access",
            evidence={
                "parent_repo": selection_manifest["inputs"]["parent_artifact"]["repo"],
                "parent_revision": selection_manifest["inputs"]["parent_artifact"][
                    "revision"
                ],
                "parent_manifest_sha256": selection_manifest["inputs"][
                    "parent_artifact"
                ]["manifest_sha256"],
            },
        ),
        event(
            "confirm-selection-v2-raw-machine-access",
            roles=("v2_confirm_primary",),
            access_kind="raw_source_machine_access",
            evidence={
                "selection_manifest_sha256": selection_manifest_sha256,
                "source_file_manifest_sha256": selection_manifest["inputs"][
                    "source_file_manifest"
                ]["sha256"],
                "manual_screenshot_review_recorded": False,
                "selection_basis_uses_policy_output": False,
            },
        ),
        event(
            "recorded-expert-actions-machine-read",
            roles=expert_roles,
            access_kind="recorded_expert_action_read",
            evidence={
                "selection_manifest_sha256": selection_manifest_sha256,
                "uses_recorded_terminal_success": True,
                "uses_action_validity": True,
                "uses_v2_policy_output": False,
                "uses_postselection_expert_alignment": False,
            },
        ),
        event(
            "ui-tars-reference-gate-v1-policy-output",
            roles=("v1_reference_contract_audit_only",),
            access_kind="policy_output",
            evidence={
                "policy_repo": v1_summary["policy"]["repo"],
                "policy_revision": v1_summary["policy"]["revision"],
                "summary_path": V1_SUMMARY_PATH,
                "summary_sha256": v1_summary_sha256,
                "hf_repo": v1_summary["artifacts"]["hf_repo"],
                "hf_revision": v1_summary["artifacts"]["hf_revision"],
                "hf_path": v1_summary["artifacts"]["hf_path"],
            },
        ),
    ]
    ledger = {
        "schema_version": EXPOSURE_SCHEMA_VERSION,
        "protocol_id": "causalcache_restoration_v2",
        "status": "FROZEN_PREOUTPUT_EXPOSURE",
        "selection_manifest_sha256": selection_manifest_sha256,
        "semantics": {
            "raw_data_seen_is_distinct_from_policy_output_seen": True,
            "confirm_terminology": "policy-output untouched; not raw-unseen",
            "negative_output_claim": "process declaration at freeze, not cryptographic proof",
            "events_are_append_only": True,
        },
        "role_source_ids": role_ids,
        "events": events,
        "reduced_roles": {},
        "freeze_declaration": {
            "gui_owl_v2_policy_output_generated": False,
            "restoration_output_generated": False,
            "confirm_selection_changed_after_output": False,
        },
    }
    ledger["reduced_roles"] = reduce_exposure_events(ledger)
    validate_exposure_ledger(ledger, selection_manifest=selection_manifest)
    return ledger


def reduce_exposure_events(ledger: Mapping[str, Any]) -> dict[str, Any]:
    role_ids = ledger["role_source_ids"]
    reduced = {
        role: {
            "raw_machine_seen": False,
            "expert_action_machine_seen": False,
            "known_policy_outputs": [],
            "known_restoration_outputs": [],
        }
        for role in role_ids
    }
    seen_event_ids: set[str] = set()
    for record in ledger["events"]:
        event_id = str(record["event_id"])
        if event_id in seen_event_ids:
            raise ValueError("exposure event IDs must be unique")
        seen_event_ids.add(event_id)
        roles = tuple(record["roles"])
        if not roles or any(role not in role_ids for role in roles):
            raise ValueError("exposure event references an unknown role")
        ids = [source_id for role in roles for source_id in role_ids[role]]
        if int(record["source_id_count"]) != len(ids):
            raise ValueError("exposure event source count mismatch")
        if record["source_ids_sha256"] != _canonical_id_digest(ids):
            raise ValueError("exposure event source digest mismatch")
        access_kind = record["access_kind"]
        for role in roles:
            if access_kind == "raw_source_machine_access":
                reduced[role]["raw_machine_seen"] = True
            elif access_kind == "recorded_expert_action_read":
                reduced[role]["expert_action_machine_seen"] = True
            elif access_kind == "policy_output":
                reduced[role]["known_policy_outputs"].append(event_id)
            elif access_kind == "restoration_output":
                reduced[role]["known_restoration_outputs"].append(event_id)
            else:
                raise ValueError(f"unsupported exposure access kind: {access_kind}")
    return reduced


def validate_exposure_ledger(
    ledger: Mapping[str, Any],
    *,
    selection_manifest: Mapping[str, Any],
) -> None:
    if ledger.get("schema_version") != EXPOSURE_SCHEMA_VERSION:
        raise ValueError("exposure schema_version mismatch")
    if ledger.get("protocol_id") != "causalcache_restoration_v2":
        raise ValueError("exposure protocol_id mismatch")
    if ledger.get("status") != "FROZEN_PREOUTPUT_EXPOSURE":
        raise ValueError("exposure status mismatch")
    if ledger.get("semantics") != {
        "raw_data_seen_is_distinct_from_policy_output_seen": True,
        "confirm_terminology": "policy-output untouched; not raw-unseen",
        "negative_output_claim": "process declaration at freeze, not cryptographic proof",
        "events_are_append_only": True,
    }:
        raise ValueError("exposure semantics drifted")
    selection_sha = ledger.get("selection_manifest_sha256")
    if not _is_sha256(selection_sha):
        raise ValueError("exposure selection-manifest SHA256 is invalid")
    if set(ledger.get("role_source_ids", {})) != set(EXPOSURE_ROLE_NAMES):
        raise ValueError("exposure role inventory drifted")
    expected_role_ids = {
        name: [
            record["source_id"]
            for record in selection_manifest["roles"][name]["trajectories"]
        ]
        for name in EXPOSURE_ROLE_NAMES
    }
    if ledger["role_source_ids"] != expected_role_ids:
        raise ValueError("exposure role IDs differ from selection manifest")
    expected_event_ids = [
        "parent-artifact-v1-raw-machine-access",
        "confirm-selection-v2-raw-machine-access",
        "recorded-expert-actions-machine-read",
        "ui-tars-reference-gate-v1-policy-output",
    ]
    if [record.get("event_id") for record in ledger.get("events", [])] != expected_event_ids:
        raise ValueError("preoutput exposure event inventory or order drifted")
    events = {record["event_id"]: record for record in ledger["events"]}
    parent_event = events["parent-artifact-v1-raw-machine-access"]
    if parent_event["roles"] != [
        "v1_reference_contract_audit_only",
        "v2_label_train",
        "v2_development",
    ] or parent_event["access_kind"] != "raw_source_machine_access":
        raise ValueError("parent raw-access exposure event drifted")
    if parent_event["evidence"] != {
        "parent_repo": selection_manifest["inputs"]["parent_artifact"]["repo"],
        "parent_revision": selection_manifest["inputs"]["parent_artifact"]["revision"],
        "parent_manifest_sha256": selection_manifest["inputs"]["parent_artifact"][
            "manifest_sha256"
        ],
    }:
        raise ValueError("parent raw-access evidence mismatch")
    confirm_event = events["confirm-selection-v2-raw-machine-access"]
    if confirm_event["roles"] != ["v2_confirm_primary"] or confirm_event[
        "access_kind"
    ] != "raw_source_machine_access":
        raise ValueError("confirm raw-access exposure event drifted")
    if confirm_event["evidence"] != {
        "selection_manifest_sha256": selection_sha,
        "source_file_manifest_sha256": selection_manifest["inputs"][
            "source_file_manifest"
        ]["sha256"],
        "manual_screenshot_review_recorded": False,
        "selection_basis_uses_policy_output": False,
    }:
        raise ValueError("confirm raw-access evidence mismatch")
    expert_event = events["recorded-expert-actions-machine-read"]
    if expert_event["roles"] != list(EXPOSURE_ROLE_NAMES) or expert_event[
        "access_kind"
    ] != "recorded_expert_action_read":
        raise ValueError("expert-action exposure event drifted")
    if expert_event["evidence"] != {
        "selection_manifest_sha256": selection_sha,
        "uses_recorded_terminal_success": True,
        "uses_action_validity": True,
        "uses_v2_policy_output": False,
        "uses_postselection_expert_alignment": False,
    }:
        raise ValueError("expert-action exposure evidence mismatch")
    policy_event = events["ui-tars-reference-gate-v1-policy-output"]
    if policy_event["roles"] != ["v1_reference_contract_audit_only"] or policy_event[
        "access_kind"
    ] != "policy_output":
        raise ValueError("v1 policy-output exposure event drifted")
    if policy_event["evidence"] != {
        "policy_repo": V1_POLICY_REPO,
        "policy_revision": V1_POLICY_REVISION,
        "summary_path": V1_SUMMARY_PATH,
        "summary_sha256": V1_SUMMARY_SHA256,
        "hf_repo": V1_OUTPUT_HF_REPO,
        "hf_revision": V1_OUTPUT_HF_REVISION,
        "hf_path": V1_OUTPUT_HF_PATH,
    }:
        raise ValueError("v1 policy-output evidence mismatch")
    reduced = reduce_exposure_events(ledger)
    if ledger["reduced_roles"] != reduced:
        raise ValueError("exposure reducer projection mismatch")
    reference = reduced["v1_reference_contract_audit_only"]
    if len(reference["known_policy_outputs"]) != 1:
        raise ValueError("v1 reference role must record exactly one known policy output")
    if reference["known_restoration_outputs"]:
        raise ValueError("v1 reference role must not record restoration output")
    for role in ("v2_label_train", "v2_development", "v2_confirm_primary"):
        projection = reduced[role]
        if not projection["raw_machine_seen"] or not projection["expert_action_machine_seen"]:
            raise ValueError(f"{role} must record raw and expert machine access")
        if projection["known_policy_outputs"] or projection["known_restoration_outputs"]:
            raise ValueError(f"{role} must be policy/restoration-output untouched")
    freeze = ledger["freeze_declaration"]
    if freeze != {
        "gui_owl_v2_policy_output_generated": False,
        "restoration_output_generated": False,
        "confirm_selection_changed_after_output": False,
    }:
        raise ValueError("preoutput exposure freeze declaration drifted")
