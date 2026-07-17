"""Pre-policy-output exposure ledger for restoration-v2.2 label expansion."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.restoration_v2_2_label_expansion import (
    DEVELOPMENT_IDS,
    EXPANSION_IDS_SHA256,
    TRAIN_IDS,
    canonical_json_bytes,
    validate_config,
    validate_expansion_manifest,
    validate_parent_selection,
)


SCHEMA_VERSION = "0.1.0"
PROTOCOL_ID = "causalcache_restoration_v2_2_label_expansion_exposure_v1"
STATUS = "FROZEN_PRE_EXPANSION_POLICY_OUTPUT_EXPOSURE"
CONFIG_PATH = "code/configs/causalcache_restoration_v2_2_label_expansion_v1.json"
CONFIG_SHA256 = "e74904fb75cd72429e6adb51ab7500adcaa1915fb6cebc8567a2945bc9373947"
PARENT_PATH = "data/manifests/restoration_v2_selection.json"
PARENT_SHA256 = "292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b"
SOURCE_MANIFEST_PATH = (
    "data/manifests/independent_reference_gate_v1_source_files.json"
)
SOURCE_MANIFEST_SHA256 = (
    "46f2240a07f46b6e283cb90e1f478f0b14a2e1499c52e2cc8e144898065adffc"
)
PARENT_EXPOSURE_PATH = "data/manifests/restoration_v2_exposure.json"
PARENT_EXPOSURE_SHA256 = (
    "bc1224826e0642c2374f6cfbebd676389d8c17ce6bf8a789f08903740cf97a95"
)
PRIOR_OUTPUT_EXPOSURE_PATH = (
    "data/manifests/spatial_reference_audit_v1_exposure.json"
)
PRIOR_OUTPUT_EXPOSURE_SHA256 = (
    "277613bd0b0a771c0ff18d234f151459cb0dd03672112a521e67889c0165966a"
)
EXPANSION_SELECTION_PATH = (
    "data/manifests/restoration_v2_2_label_expansion_selection.json"
)

PRIOR_OUTPUT_ROLES = (
    "v1_reference_contract_audit_only",
    "v2_label_train",
    "v2_development",
)
CONFIRM_ROLE = "v2_confirm_primary"
GENERATOR_PATHS = {
    "module": (
        "code/causalcache/restoration_v2_2_label_expansion_exposure.py"
    ),
    "cli": (
        "code/scripts/materialize_restoration_v2_2_label_expansion_exposure.py"
    ),
    "validator": (
        "code/scripts/validate_restoration_v2_2_label_expansion_exposure.py"
    ),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_json_object(path: str | Path) -> tuple[bytes, dict[str, Any]]:
    resolved = Path(path)
    payload = resolved.read_bytes()
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object: {resolved}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a JSON object")
    return payload, value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _id_digest(source_ids: Sequence[str]) -> str:
    return sha256_bytes(canonical_json_bytes(list(source_ids)))


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _role_ids(parent: Mapping[str, Any], role: str) -> tuple[str, ...]:
    try:
        records = parent["roles"][role]["trajectories"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"parent role is missing: {role}") from exc
    if not isinstance(records, list):
        raise ValueError(f"parent role trajectories must be a list: {role}")
    values = tuple(str(record["source_id"]) for record in records)
    if len(values) != len(set(values)):
        raise ValueError(f"parent role has duplicate source IDs: {role}")
    return values


def _cohort(source_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "trajectory_count": len(source_ids),
        "source_ids_sha256": _id_digest(source_ids),
        "source_ids_emitted": False,
    }


def _validate_generator(generator: Mapping[str, Any]) -> None:
    expected_keys = {"git_revision"}
    for prefix in GENERATOR_PATHS:
        expected_keys.update({f"{prefix}_path", f"{prefix}_sha256"})
    if set(generator) != expected_keys:
        raise ValueError("exposure-ledger generator schema mismatch")
    if re.fullmatch(r"[0-9a-f]{40}", str(generator["git_revision"])) is None:
        raise ValueError("exposure-ledger generator Git revision is invalid")
    for prefix, path in GENERATOR_PATHS.items():
        if generator[f"{prefix}_path"] != path:
            raise ValueError(f"exposure-ledger generator {prefix} path mismatch")
        if not _is_sha256(generator[f"{prefix}_sha256"]):
            raise ValueError(f"exposure-ledger generator {prefix} SHA256 is invalid")


def _validate_source_manifest(
    source_manifest: Mapping[str, Any],
    *,
    source_manifest_sha256: str,
    parent: Mapping[str, Any],
) -> None:
    if source_manifest_sha256 != SOURCE_MANIFEST_SHA256:
        raise ValueError("source-file manifest SHA256 mismatch")
    parent_source = _require_mapping(
        _require_mapping(parent.get("inputs"), name="parent.inputs").get(
            "source_file_manifest"
        ),
        name="parent source-file manifest",
    )
    files = source_manifest.get("files")
    if not isinstance(files, list) or len(files) != 16:
        raise ValueError("source-file manifest must contain exactly 16 files")
    if (
        source_manifest.get("repo") != parent_source.get("repo")
        or source_manifest.get("revision") != parent_source.get("revision")
        or source_manifest.get("total_bytes") != parent_source.get("total_bytes")
        or len(files) != parent_source.get("file_count")
        or parent_source.get("path") != SOURCE_MANIFEST_PATH
        or parent_source.get("sha256") != source_manifest_sha256
    ):
        raise ValueError("source-file manifest and parent binding disagree")
    seen_paths: set[str] = set()
    total_bytes = 0
    for record in files:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "size",
        }:
            raise ValueError("source-file record schema mismatch")
        path = str(record["path"])
        if path in seen_paths or not _is_sha256(record["sha256"]):
            raise ValueError("source-file record identity mismatch")
        seen_paths.add(path)
        size = record["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("source-file record size is invalid")
        total_bytes += size
    if total_bytes != source_manifest.get("total_bytes"):
        raise ValueError("source-file total byte count mismatch")


def _validate_parent_exposure(
    exposure: Mapping[str, Any],
    *,
    exposure_sha256: str,
    parent: Mapping[str, Any],
) -> None:
    if exposure_sha256 != PARENT_EXPOSURE_SHA256:
        raise ValueError("parent exposure SHA256 mismatch")
    if (
        exposure.get("schema_version") != "0.1.0"
        or exposure.get("protocol_id") != "causalcache_restoration_v2"
        or exposure.get("status") != "FROZEN_PREOUTPUT_EXPOSURE"
        or exposure.get("selection_manifest_sha256") != PARENT_SHA256
    ):
        raise ValueError("parent exposure identity mismatch")
    role_source_ids = _require_mapping(
        exposure.get("role_source_ids"), name="parent exposure role_source_ids"
    )
    expected = {
        role: list(_role_ids(parent, role))
        for role in (*PRIOR_OUTPUT_ROLES, CONFIRM_ROLE)
    }
    if dict(role_source_ids) != expected:
        raise ValueError("parent exposure role identities disagree with parent selection")
    reduced = _require_mapping(
        exposure.get("reduced_roles"), name="parent exposure reduced_roles"
    )
    reference = _require_mapping(
        reduced.get("v1_reference_contract_audit_only"),
        name="reference exposure projection",
    )
    if reference.get("known_policy_outputs") != [
        "ui-tars-reference-gate-v1-policy-output"
    ]:
        raise ValueError("reference role policy-output evidence is missing")
    for role in ("v2_label_train", "v2_development", CONFIRM_ROLE):
        record = _require_mapping(
            reduced.get(role), name=f"parent exposure projection {role}"
        )
        if record.get("known_policy_outputs") or record.get(
            "known_restoration_outputs"
        ):
            raise ValueError("parent preoutput role projection drifted")


def _validate_prior_output_exposure(
    exposure: Mapping[str, Any],
    *,
    exposure_sha256: str,
) -> None:
    if exposure_sha256 != PRIOR_OUTPUT_EXPOSURE_SHA256:
        raise ValueError("prior-output exposure SHA256 mismatch")
    if (
        exposure.get("schema_version") != "1.0.0"
        or exposure.get("protocol_id") != "spatial_reference_audit_v1"
        or exposure.get("status")
        != "FROZEN_SOURCE_ONLY_BEFORE_AUDIT_POLICY_FORWARD"
    ):
        raise ValueError("prior-output exposure identity mismatch")
    if exposure.get("selection_manifest") != {
        "path": PARENT_PATH,
        "sha256": PARENT_SHA256,
    } or exposure.get("parent_exposure") != {
        "path": PARENT_EXPOSURE_PATH,
        "sha256": PARENT_EXPOSURE_SHA256,
        "mutation_allowed": False,
    }:
        raise ValueError("prior-output exposure parent binding mismatch")
    role_state = _require_mapping(
        exposure.get("role_state"), name="prior-output role_state"
    )
    expected = {
        "v2_label_train": {
            "source_id_count": 10,
            "known_policy_output": True,
            "known_restoration_output": False,
        },
        "v2_development": {
            "source_id_count": 5,
            "known_policy_output": True,
            "known_restoration_output": False,
        },
        CONFIRM_ROLE: {
            "source_id_count": 20,
            "known_policy_output": False,
            "known_restoration_output": False,
            "policy_output_access_allowed_by_this_protocol": False,
        },
    }
    if dict(role_state) != expected:
        raise ValueError("prior-output exposure role state drifted")


def _validate_inputs(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    parent: Mapping[str, Any],
    parent_sha256: str,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    parent_exposure: Mapping[str, Any],
    parent_exposure_sha256: str,
    prior_output_exposure: Mapping[str, Any],
    prior_output_exposure_sha256: str,
    expansion_selection: Mapping[str, Any] | None,
    expansion_selection_sha256: str | None,
) -> dict[str, Any]:
    validate_config(config, source_sha256=config_sha256)
    derived = validate_parent_selection(
        parent,
        source_sha256=parent_sha256,
        config=config,
    )
    _validate_source_manifest(
        source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        parent=parent,
    )
    _validate_parent_exposure(
        parent_exposure,
        exposure_sha256=parent_exposure_sha256,
        parent=parent,
    )
    _validate_prior_output_exposure(
        prior_output_exposure,
        exposure_sha256=prior_output_exposure_sha256,
    )
    if (expansion_selection is None) != (expansion_selection_sha256 is None):
        raise ValueError("expansion selection value and SHA256 must be provided together")
    if expansion_selection is not None:
        if not _is_sha256(expansion_selection_sha256):
            raise ValueError("expansion selection SHA256 is invalid")
        validate_expansion_manifest(
            expansion_selection,
            config=config,
            config_sha256=config_sha256,
            parent=parent,
            parent_sha256=parent_sha256,
        )
    return derived


def _input_bindings(
    *,
    parent: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    expansion_selection: Mapping[str, Any] | None,
    expansion_selection_sha256: str | None,
) -> dict[str, Any]:
    source = parent["inputs"]["source_file_manifest"]
    artifact = parent["inputs"]["parent_artifact"]
    expansion_binding = {
        "path": EXPANSION_SELECTION_PATH,
        "available_at_freeze": expansion_selection is not None,
        "sha256": expansion_selection_sha256,
        "status": (
            expansion_selection["status"] if expansion_selection is not None else None
        ),
        "generator_git_revision": (
            expansion_selection["generator"]["git_revision"]
            if expansion_selection is not None
            else None
        ),
    }
    return {
        "split_config": {"path": CONFIG_PATH, "sha256": CONFIG_SHA256},
        "parent_selection": {"path": PARENT_PATH, "sha256": PARENT_SHA256},
        "source_file_manifest": {
            "path": SOURCE_MANIFEST_PATH,
            "sha256": SOURCE_MANIFEST_SHA256,
            "repo": source["repo"],
            "revision": source["revision"],
            "file_count": len(source_manifest["files"]),
            "total_bytes": source_manifest["total_bytes"],
        },
        "eligible_pool_artifact": {
            "repo": artifact["repo"],
            "revision": artifact["revision"],
            "manifest_sha256": artifact["manifest_sha256"],
            "eligible_pool_sha256": artifact["eligible_pool_sha256"],
        },
        "parent_preoutput_exposure": {
            "path": PARENT_EXPOSURE_PATH,
            "sha256": PARENT_EXPOSURE_SHA256,
        },
        "prior_output_exposure": {
            "path": PRIOR_OUTPUT_EXPOSURE_PATH,
            "sha256": PRIOR_OUTPUT_EXPOSURE_SHA256,
        },
        "expansion_selection_manifest": expansion_binding,
    }


def _intersection_record(
    *,
    reserved_cohort: str,
    expansion_ids: Sequence[str],
    reserved_ids: Sequence[str],
) -> dict[str, Any]:
    intersection = tuple(sorted(set(expansion_ids).intersection(reserved_ids)))
    return {
        "expansion_cohort": "label_expansion_64",
        "reserved_cohort": reserved_cohort,
        "expansion_trajectory_count": len(expansion_ids),
        "reserved_trajectory_count": len(reserved_ids),
        "intersection_count": len(intersection),
        "intersection_source_ids_sha256": _id_digest(intersection),
        "intersection_source_ids_emitted": False,
    }


def _ledger_payload(
    *,
    parent: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    expansion_selection: Mapping[str, Any] | None,
    expansion_selection_sha256: str | None,
    generator: Mapping[str, Any],
) -> dict[str, Any]:
    role_ids = {role: _role_ids(parent, role) for role in PRIOR_OUTPUT_ROLES}
    confirm_ids = _role_ids(parent, CONFIRM_ROLE)
    expansion_ids = TRAIN_IDS + DEVELOPMENT_IDS
    prior_union = tuple(
        source_id for role in PRIOR_OUTPUT_ROLES for source_id in role_ids[role]
    )
    reserved_union = prior_union + confirm_ids
    comparisons = [
        _intersection_record(
            reserved_cohort=role,
            expansion_ids=expansion_ids,
            reserved_ids=role_ids[role],
        )
        for role in PRIOR_OUTPUT_ROLES
    ]
    comparisons.extend(
        [
            _intersection_record(
                reserved_cohort="prior_output_exposed_union",
                expansion_ids=expansion_ids,
                reserved_ids=prior_union,
            ),
            _intersection_record(
                reserved_cohort="sealed_confirm",
                expansion_ids=expansion_ids,
                reserved_ids=confirm_ids,
            ),
            _intersection_record(
                reserved_cohort="all_reserved_union",
                expansion_ids=expansion_ids,
                reserved_ids=reserved_union,
            ),
        ]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": STATUS,
        "semantics": {
            "identity_disjointness_is_mechanical_from_bound_inputs": True,
            "negative_access_assertions_are_process_declarations_not_cryptographic_proof": True,
            "raw_source_access_is_distinct_from_policy_or_restoration_output_access": True,
            "sealed_confirm_is_distinct_from_prior_output_exposure": True,
            "source_ids_are_withheld_in_favor_of_counts_and_digests": True,
        },
        "inputs": _input_bindings(
            parent=parent,
            source_manifest=source_manifest,
            expansion_selection=expansion_selection,
            expansion_selection_sha256=expansion_selection_sha256,
        ),
        "generator": dict(generator),
        "cohort_inventory": {
            "prior_output_exposed": {
                "roles": {
                    "v1_reference_contract_audit_only": {
                        **_cohort(role_ids["v1_reference_contract_audit_only"]),
                        "exposure_evidence": "parent_preoutput_exposure.policy_output_event",
                    },
                    "v2_label_train": {
                        **_cohort(role_ids["v2_label_train"]),
                        "exposure_evidence": "prior_output_exposure.role_state",
                    },
                    "v2_development": {
                        **_cohort(role_ids["v2_development"]),
                        "exposure_evidence": "prior_output_exposure.role_state",
                    },
                },
                "union": _cohort(prior_union),
            },
            "sealed_confirm": {
                **_cohort(confirm_ids),
                "known_policy_output": False,
                "known_restoration_output": False,
                "output_access_allowed_by_this_ledger": False,
                "exposure_evidence": "prior_output_exposure.role_state",
            },
            "label_expansion": {
                **_cohort(expansion_ids),
                "gate_train_expansion": _cohort(TRAIN_IDS),
                "gate_development_expansion": _cohort(DEVELOPMENT_IDS),
            },
            "all_reserved_union": _cohort(reserved_union),
        },
        "overlap_proof": {
            "empty_intersection_source_ids_sha256": _id_digest(()),
            "comparisons": comparisons,
            "all_required_intersections_are_empty": all(
                comparison["intersection_count"] == 0
                for comparison in comparisons
            ),
        },
        "selection_freeze": {
            "eligible_order": "parent_reconstruction_eligible_pool_order",
            "eligible_order_is_salted_selection_sha256_order": True,
            "prior_output_exposed_trajectories_excluded": 23,
            "remaining_after_prior_exclusion": 88,
            "structural_filter": "decision_count>=5",
            "structurally_eligible_trajectories": 84,
            "sealed_confirm_is_exact_first_structural_prefix": 20,
            "expansion_is_exact_remaining_structural_tail": 64,
            "train_is_first_expansion_prefix": 48,
            "development_is_remaining_expansion_suffix": 16,
            "top_up_allowed": False,
            "top_up_performed": False,
            "semantic_content_used_for_membership": False,
            "policy_output_accessed_for_membership": False,
            "restoration_output_accessed_for_membership": False,
        },
        "pre_policy_output_assertions": {
            "expansion_policy_output_generated_before_freeze": False,
            "expansion_restoration_output_generated_before_freeze": False,
            "expansion_gate_training_started_before_freeze": False,
            "selection_changed_after_expansion_output": False,
            "confirm_policy_output_accessed": False,
            "confirm_restoration_output_accessed": False,
        },
        "leakage_firewall": {
            "source_ids_emitted": False,
            "instructions_or_actions_emitted": False,
            "image_or_ocr_identity_emitted": False,
            "policy_or_restoration_values_emitted": False,
        },
    }


def build_exposure_ledger(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    parent: Mapping[str, Any],
    parent_sha256: str,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    parent_exposure: Mapping[str, Any],
    parent_exposure_sha256: str,
    prior_output_exposure: Mapping[str, Any],
    prior_output_exposure_sha256: str,
    expansion_selection: Mapping[str, Any] | None,
    expansion_selection_sha256: str | None,
    generator: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_inputs(
        config=config,
        config_sha256=config_sha256,
        parent=parent,
        parent_sha256=parent_sha256,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        parent_exposure=parent_exposure,
        parent_exposure_sha256=parent_exposure_sha256,
        prior_output_exposure=prior_output_exposure,
        prior_output_exposure_sha256=prior_output_exposure_sha256,
        expansion_selection=expansion_selection,
        expansion_selection_sha256=expansion_selection_sha256,
    )
    _validate_generator(generator)
    ledger = _ledger_payload(
        parent=parent,
        source_manifest=source_manifest,
        expansion_selection=expansion_selection,
        expansion_selection_sha256=expansion_selection_sha256,
        generator=generator,
    )
    validate_exposure_ledger(
        ledger,
        config=config,
        config_sha256=config_sha256,
        parent=parent,
        parent_sha256=parent_sha256,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        parent_exposure=parent_exposure,
        parent_exposure_sha256=parent_exposure_sha256,
        prior_output_exposure=prior_output_exposure,
        prior_output_exposure_sha256=prior_output_exposure_sha256,
        expansion_selection=expansion_selection,
        expansion_selection_sha256=expansion_selection_sha256,
    )
    return ledger


def validate_exposure_ledger(
    ledger: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    parent: Mapping[str, Any],
    parent_sha256: str,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    parent_exposure: Mapping[str, Any],
    parent_exposure_sha256: str,
    prior_output_exposure: Mapping[str, Any],
    prior_output_exposure_sha256: str,
    expansion_selection: Mapping[str, Any] | None,
    expansion_selection_sha256: str | None,
) -> None:
    _validate_inputs(
        config=config,
        config_sha256=config_sha256,
        parent=parent,
        parent_sha256=parent_sha256,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        parent_exposure=parent_exposure,
        parent_exposure_sha256=parent_exposure_sha256,
        prior_output_exposure=prior_output_exposure,
        prior_output_exposure_sha256=prior_output_exposure_sha256,
        expansion_selection=expansion_selection,
        expansion_selection_sha256=expansion_selection_sha256,
    )
    generator = _require_mapping(ledger.get("generator"), name="ledger.generator")
    _validate_generator(generator)
    expected = _ledger_payload(
        parent=parent,
        source_manifest=source_manifest,
        expansion_selection=expansion_selection,
        expansion_selection_sha256=expansion_selection_sha256,
        generator=generator,
    )
    if dict(ledger) != expected:
        raise ValueError("expansion exposure ledger differs from the frozen contract")
    if ledger["cohort_inventory"]["label_expansion"]["source_ids_sha256"] != (
        EXPANSION_IDS_SHA256
    ):
        raise ValueError("expansion exposure ledger expansion digest drifted")
    comparisons = ledger["overlap_proof"]["comparisons"]
    if len(comparisons) != 6 or any(
        comparison["intersection_count"] != 0
        or comparison["intersection_source_ids_sha256"] != _id_digest(())
        for comparison in comparisons
    ):
        raise ValueError("expansion exposure ledger overlap proof is not empty")


def summary(ledger: Mapping[str, Any], *, ledger_sha256: str) -> dict[str, Any]:
    return {
        "outcome": "FROZEN_LABEL_EXPANSION_PREOUTPUT_EXPOSURE",
        "ledger_sha256": ledger_sha256,
        "prior_output_exposed_trajectories": ledger["cohort_inventory"][
            "prior_output_exposed"
        ]["union"]["trajectory_count"],
        "sealed_confirm_trajectories": ledger["cohort_inventory"][
            "sealed_confirm"
        ]["trajectory_count"],
        "expansion_trajectories": ledger["cohort_inventory"]["label_expansion"][
            "trajectory_count"
        ],
        "required_empty_intersections": len(
            ledger["overlap_proof"]["comparisons"]
        ),
        "expansion_selection_bound": ledger["inputs"][
            "expansion_selection_manifest"
        ]["available_at_freeze"],
    }
