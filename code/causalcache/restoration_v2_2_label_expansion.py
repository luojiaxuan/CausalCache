"""Policy-blind split contract for restoration-v2.2 label expansion."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1.0"
PROTOCOL_ID = "causalcache_restoration_v2_2_label_expansion_v1"
STATUS = "FROZEN_POLICY_BLIND_STRUCTURAL_SPLIT"
CONFIG_PATH = "code/configs/causalcache_restoration_v2_2_label_expansion_v1.json"
CONFIG_SHA256 = "e74904fb75cd72429e6adb51ab7500adcaa1915fb6cebc8567a2945bc9373947"
PARENT_PATH = "data/manifests/restoration_v2_selection.json"
PARENT_SHA256 = "292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b"
ELIGIBLE_POOL_SHA256 = "84d6855b20227d7f884d4ce26cfae7188c1075bb7751cab78a3cf325eef192cb"
EXPANSION_IDS_SHA256 = "238a5608d08f6e4594195690b7595b1953e0865da2192e1e4b46d8b5795b3526"
TRAIN_IDS_SHA256 = "11904e1d102b69561b402bea4e42044be9572884f6168597c5f07c3ca346cb34"
DEVELOPMENT_IDS_SHA256 = "1c37cbf6b67b0ddee61b3efe27d33b30c8fbfb471ced12f12e49624a10b82454"

TRAIN_IDS = (
    "0081171303194449",
    "0016567388192150",
    "0081349207433064",
    "0106001886731694",
    "0013756686421890",
    "0224710820406687",
    "0096794886053450",
    "0119022933981939",
    "0149901478066310",
    "0031013035186872",
    "0156438551317768",
    "0185503498056981",
    "0213791829762595",
    "0154474824404795",
    "0026634903318160",
    "0177467386816906",
    "0131888848291691",
    "0076518844281084",
    "0072821463326485",
    "0135431381761380",
    "0107109187202943",
    "0135789980944621",
    "0114203784922462",
    "0234089730527063",
    "0218815826133241",
    "0134720948350318",
    "0204580742481366",
    "0084888665394360",
    "0037690483064801",
    "0235343430658003",
    "0206048389903107",
    "0220943321467127",
    "0167183648468991",
    "0122430166514446",
    "0119569788020140",
    "0092302829764693",
    "0227020388419136",
    "0041513657979544",
    "0192684638288967",
    "0108772154206513",
    "0000922671359408",
    "0066997985759988",
    "0137575778052679",
    "0114247701606163",
    "0191751033400150",
    "0123550855207688",
    "0157485992279511",
    "0039770017320665",
)

DEVELOPMENT_IDS = (
    "0113395203853614",
    "0085973327901251",
    "0217993280505982",
    "0204226739513713",
    "0027349945578994",
    "0063414321407600",
    "0124211143250759",
    "0158200624694354",
    "0104781033187528",
    "0042582068233164",
    "0095139503173521",
    "0028705883346522",
    "0069283472940099",
    "0090192011883108",
    "0017986853193264",
    "0002455403270123",
)

ROLE_NAMES = (
    "v1_reference_contract_audit_only",
    "v2_label_train",
    "v2_development",
    "v2_confirm_primary",
)
PRIOR_ROLE_NAMES = ROLE_NAMES[:3]
SPLIT_NAMES = ("gate_train_expansion", "gate_development_expansion")
PARENT_TRAJECTORY_KEYS = {
    "source_id",
    "transport_file",
    "transport_row_index",
    "selection_sha256",
    "decision_count",
    "normalized_app_labels",
    "action_type_counts",
}
TRAJECTORY_KEYS = {
    "source_id",
    "eligible_order_index",
    "transport_file",
    "transport_row_index",
    "selection_sha256",
    "decision_count",
}
STATE_KEYS = {
    "state_id",
    "source_id",
    "decision_step_id",
    "history_event_step_ids",
    "current_equivalent_event_step_id",
    "candidate_event_step_ids",
    "candidate_event_count",
    "full_subset_distance_rows",
    "deployment_conditional_edges",
    "teacher_forwards",
}
FORBIDDEN_SEMANTIC_KEYS = {
    "instruction",
    "instruction_sha256",
    "current_observation",
    "candidate_event_post_states",
    "current_equivalence_witness",
    "validated_action",
    "validated_action_sha256",
    "canonical_action",
    "ocr_text",
    "image_path",
    "image_sha256",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_json_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    value = json.loads(
        payload,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload, value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _id_digest(source_ids: Sequence[str]) -> str:
    return sha256_bytes(canonical_json_bytes(list(source_ids)))


def _require_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys mismatch")


def validate_config(config: Mapping[str, Any], *, source_sha256: str) -> None:
    if source_sha256 != CONFIG_SHA256:
        raise ValueError("label-expansion config SHA256 mismatch")
    _require_keys(
        config,
        {
            "schema_version",
            "protocol_id",
            "preregistration_status",
            "parent_selection",
            "selection",
            "label_geometry",
            "expected_workload",
            "output_firewall",
        },
        "config",
    )
    if config["schema_version"] != SCHEMA_VERSION or config["protocol_id"] != PROTOCOL_ID:
        raise ValueError("label-expansion config identity mismatch")
    if (
        config["preregistration_status"]
        != "policy_blind_split_frozen_before_expansion_policy_output"
    ):
        raise ValueError("label-expansion preregistration status mismatch")
    parent = config["parent_selection"]
    if parent != {
        "path": PARENT_PATH,
        "sha256": PARENT_SHA256,
        "protocol_id": "causalcache_restoration_v2",
        "status": "PASSED_PREOUTPUT_SELECTION",
        "eligible_pool_count": 111,
        "eligible_pool_sha256": ELIGIBLE_POOL_SHA256,
    }:
        raise ValueError("label-expansion parent identity mismatch")

    selection = config["selection"]
    _require_keys(
        selection,
        {
            "eligible_order",
            "prior_roles_to_exclude",
            "prior_excluded_trajectory_count",
            "remaining_after_prior_exclusion",
            "structural_filter",
            "structurally_eligible_count",
            "frozen_confirm_role_to_exclude",
            "frozen_confirm_trajectory_count",
            "expansion_rule",
            "expansion_trajectory_count",
            "expansion_source_ids_sha256",
            "gate_train_expansion",
            "gate_development_expansion",
        },
        "config.selection",
    )
    expected_scalars = {
        "eligible_order": "parent_reconstruction_eligible_pool_order",
        "prior_roles_to_exclude": list(PRIOR_ROLE_NAMES),
        "prior_excluded_trajectory_count": 23,
        "remaining_after_prior_exclusion": 88,
        "structural_filter": "decision_count>=5",
        "structurally_eligible_count": 84,
        "frozen_confirm_role_to_exclude": "v2_confirm_primary",
        "frozen_confirm_trajectory_count": 20,
        "expansion_rule": "structural_tail_after_exact_frozen_confirm_prefix",
        "expansion_trajectory_count": 64,
        "expansion_source_ids_sha256": EXPANSION_IDS_SHA256,
    }
    if any(selection.get(key) != value for key, value in expected_scalars.items()):
        raise ValueError("label-expansion selection rule mismatch")
    split_expectations = {
        "gate_train_expansion": (
            TRAIN_IDS,
            TRAIN_IDS_SHA256,
            "first_48_expansion_trajectories_in_parent_order",
        ),
        "gate_development_expansion": (
            DEVELOPMENT_IDS,
            DEVELOPMENT_IDS_SHA256,
            "last_16_expansion_trajectories_in_parent_order",
        ),
    }
    for split_name, (ids, digest, rule) in split_expectations.items():
        split = selection[split_name]
        if split != {
            "trajectory_count": len(ids),
            "selection": rule,
            "source_ids_sha256": digest,
            "source_ids": list(ids),
        }:
            raise ValueError(f"{split_name} frozen IDs or order mismatch")
        if _id_digest(ids) != digest:
            raise RuntimeError(f"{split_name} compiled digest is inconsistent")
    if _id_digest(TRAIN_IDS + DEVELOPMENT_IDS) != EXPANSION_IDS_SHA256:
        raise RuntimeError("compiled expansion ID digest is inconsistent")

    geometry = config["label_geometry"]
    if geometry != {
        "decision_step_ids": [4, 5, 6],
        "candidate_event_counts": [2, 3, 4],
        "full_subset_rows_by_candidate_count": {"2": 4, "3": 8, "4": 16},
        "deployment_budget": 2,
        "deployment_conditional_edges_by_candidate_count": {
            "2": 4,
            "3": 9,
            "4": 16,
        },
        "reference_forwards_per_state": 1,
        "teacher_forwards_per_trajectory": 31,
    }:
        raise ValueError("label-expansion geometry mismatch")
    expected_workload = {
        "gate_train_expansion": {
            "trajectories": 48,
            "states": 144,
            "full_subset_distance_rows": 1344,
            "deployment_conditional_edges": 1392,
            "teacher_forwards": 1488,
        },
        "gate_development_expansion": {
            "trajectories": 16,
            "states": 48,
            "full_subset_distance_rows": 448,
            "deployment_conditional_edges": 464,
            "teacher_forwards": 496,
        },
        "total": {
            "trajectories": 64,
            "states": 192,
            "full_subset_distance_rows": 1792,
            "deployment_conditional_edges": 1856,
            "teacher_forwards": 1984,
        },
    }
    if config["expected_workload"] != expected_workload:
        raise ValueError("label-expansion workload mismatch")
    if config["output_firewall"] != {
        "structural_manifest_only": True,
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "include_prior_role_source_ids": False,
        "include_parent_state_records": False,
        "include_image_or_ocr_identity": False,
        "include_instruction_or_action_identity": False,
    }:
        raise ValueError("label-expansion output firewall mismatch")


def _role_ids(parent: Mapping[str, Any], role_name: str) -> tuple[str, ...]:
    try:
        trajectories = parent["roles"][role_name]["trajectories"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"parent role {role_name} is missing") from exc
    if not isinstance(trajectories, list):
        raise ValueError(f"parent role {role_name} trajectories must be a list")
    source_ids = tuple(str(record["source_id"]) for record in trajectories)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"parent role {role_name} has duplicate source IDs")
    return source_ids


def _eligible_pool(parent: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    try:
        records = parent["reconstruction"]["eligible_pool"]
    except (KeyError, TypeError) as exc:
        raise ValueError("parent eligible pool is missing") from exc
    if not isinstance(records, list) or len(records) != 111:
        raise ValueError("parent eligible pool must contain exactly 111 records")
    for record in records:
        if not isinstance(record, Mapping) or set(record) != PARENT_TRAJECTORY_KEYS:
            raise ValueError("parent eligible-pool record schema mismatch")
        if not _is_sha256(record["selection_sha256"]):
            raise ValueError("parent trajectory selection SHA256 is invalid")
        if int(record["decision_count"]) < 1:
            raise ValueError("parent trajectory decision_count is invalid")
    source_ids = tuple(str(record["source_id"]) for record in records)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("parent eligible source IDs must be unique")
    if tuple(record["selection_sha256"] for record in records) != tuple(
        sorted(record["selection_sha256"] for record in records)
    ):
        raise ValueError("parent eligible pool is not in frozen salted order")
    if sha256_bytes(canonical_json_bytes(records)) != ELIGIBLE_POOL_SHA256:
        raise ValueError("parent eligible-pool records do not reproduce their SHA256")
    return tuple(records)


def _derive(parent: Mapping[str, Any]) -> dict[str, Any]:
    pool = _eligible_pool(parent)
    role_ids = {name: _role_ids(parent, name) for name in ROLE_NAMES}
    expected_role_counts = {
        "v1_reference_contract_audit_only": 8,
        "v2_label_train": 10,
        "v2_development": 5,
        "v2_confirm_primary": 20,
    }
    if any(len(role_ids[name]) != count for name, count in expected_role_counts.items()):
        raise ValueError("parent role trajectory counts mismatch")
    all_role_ids = tuple(source_id for name in ROLE_NAMES for source_id in role_ids[name])
    if len(all_role_ids) != len(set(all_role_ids)):
        raise ValueError("parent reference, legacy, and confirm roles must be disjoint")

    prior_ids = tuple(
        source_id for name in PRIOR_ROLE_NAMES for source_id in role_ids[name]
    )
    prior_set = set(prior_ids)
    remaining = tuple(record for record in pool if str(record["source_id"]) not in prior_set)
    structural = tuple(record for record in remaining if int(record["decision_count"]) >= 5)
    confirm_ids = role_ids["v2_confirm_primary"]
    structural_prefix_ids = tuple(str(record["source_id"]) for record in structural[:20])
    if structural_prefix_ids != confirm_ids:
        raise ValueError("parent confirm role is not the exact first-20 structural prefix")
    expansion = structural[20:]
    expansion_ids = tuple(str(record["source_id"]) for record in expansion)
    if len(remaining) != 88 or len(structural) != 84 or len(expansion) != 64:
        raise ValueError("parent expansion pool accounting mismatch")
    if expansion_ids != TRAIN_IDS + DEVELOPMENT_IDS:
        raise ValueError("derived expansion IDs or order differ from the frozen split")
    if _id_digest(expansion_ids) != EXPANSION_IDS_SHA256:
        raise ValueError("derived expansion ID digest mismatch")
    pool_indices = {str(record["source_id"]): index for index, record in enumerate(pool)}
    return {
        "pool": pool,
        "pool_indices": pool_indices,
        "role_ids": role_ids,
        "prior_ids": prior_ids,
        "remaining": remaining,
        "structural": structural,
        "confirm_ids": confirm_ids,
        "expansion": expansion,
    }


def validate_parent_selection(
    parent: Mapping[str, Any],
    *,
    source_sha256: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if source_sha256 != PARENT_SHA256:
        raise ValueError("parent selection SHA256 mismatch")
    if parent.get("protocol_id") != "causalcache_restoration_v2":
        raise ValueError("parent selection protocol mismatch")
    if parent.get("status") != "PASSED_PREOUTPUT_SELECTION":
        raise ValueError("parent selection status mismatch")
    if parent.get("policy_output_generated") is not False:
        raise ValueError("parent selection must remain pre-policy-output")
    if parent.get("restoration_output_generated") is not False:
        raise ValueError("parent selection must remain pre-restoration-output")
    reconstruction = parent.get("reconstruction")
    if not isinstance(reconstruction, Mapping):
        raise ValueError("parent reconstruction is missing")
    if reconstruction.get("eligible_pool_count") != 111:
        raise ValueError("parent eligible-pool count mismatch")
    if reconstruction.get("eligible_pool_sha256") != ELIGIBLE_POOL_SHA256:
        raise ValueError("parent eligible-pool identity mismatch")
    if config["parent_selection"]["sha256"] != source_sha256:
        raise ValueError("config and parent selection identity disagree")
    return _derive(parent)


def _distance_rows(candidate_count: int) -> int:
    return 2**candidate_count


def _deployment_edges(candidate_count: int, budget: int) -> int:
    return sum(
        math.comb(candidate_count, coalition_size)
        * (candidate_count - coalition_size)
        for coalition_size in range(min(candidate_count, budget))
    )


def _trajectory_record(
    source: Mapping[str, Any], *, eligible_order_index: int
) -> dict[str, Any]:
    return {
        "source_id": str(source["source_id"]),
        "eligible_order_index": eligible_order_index,
        "transport_file": str(source["transport_file"]),
        "transport_row_index": int(source["transport_row_index"]),
        "selection_sha256": str(source["selection_sha256"]),
        "decision_count": int(source["decision_count"]),
    }


def _state_record(source_id: str, decision_step_id: int, *, budget: int) -> dict[str, Any]:
    history = list(range(1, decision_step_id))
    candidates = history[:-1]
    candidate_count = len(candidates)
    distance_rows = _distance_rows(candidate_count)
    return {
        "state_id": f"{source_id}:decision_step:{decision_step_id:03d}",
        "source_id": source_id,
        "decision_step_id": decision_step_id,
        "history_event_step_ids": history,
        "current_equivalent_event_step_id": history[-1],
        "candidate_event_step_ids": candidates,
        "candidate_event_count": candidate_count,
        "full_subset_distance_rows": distance_rows,
        "deployment_conditional_edges": _deployment_edges(candidate_count, budget),
        "teacher_forwards": distance_rows + 1,
    }


def _split_records(
    records: Sequence[Mapping[str, Any]],
    *,
    pool_indices: Mapping[str, int],
    budget: int,
) -> dict[str, Any]:
    trajectories = [
        _trajectory_record(
            record,
            eligible_order_index=pool_indices[str(record["source_id"])],
        )
        for record in records
    ]
    states = [
        _state_record(str(record["source_id"]), step, budget=budget)
        for record in records
        for step in (4, 5, 6)
    ]
    return {"trajectories": trajectories, "states": states}


def _source_identities(parent: Mapping[str, Any]) -> dict[str, Any]:
    inputs = parent["inputs"]
    source = inputs["source_file_manifest"]
    artifact = inputs["parent_artifact"]
    scientific = inputs["scientific_contract"]
    return {
        "scientific_contract": {
            "path": scientific["path"],
            "sha256": scientific["sha256"],
        },
        "eligible_pool_artifact": {
            "repo": artifact["repo"],
            "revision": artifact["revision"],
            "manifest_sha256": artifact["manifest_sha256"],
            "eligible_pool_sha256": artifact["eligible_pool_sha256"],
        },
        "source_file_manifest": {
            "path": source["path"],
            "sha256": source["sha256"],
            "repo": source["repo"],
            "revision": source["revision"],
            "file_count": source["file_count"],
            "total_bytes": source["total_bytes"],
        },
    }


def _expected_derivation(derived: Mapping[str, Any]) -> dict[str, Any]:
    role_ids = derived["role_ids"]
    expansion_ids = tuple(str(record["source_id"]) for record in derived["expansion"])
    return {
        "eligible_order": "parent_reconstruction_eligible_pool_order",
        "eligible_pool_count": 111,
        "eligible_pool_sha256": ELIGIBLE_POOL_SHA256,
        "prior_roles_excluded": [
            {
                "role": role_name,
                "trajectory_count": len(role_ids[role_name]),
                "source_ids_sha256": _id_digest(role_ids[role_name]),
            }
            for role_name in PRIOR_ROLE_NAMES
        ],
        "prior_excluded_union_count": len(derived["prior_ids"]),
        "prior_excluded_union_sha256": _id_digest(derived["prior_ids"]),
        "remaining_after_prior_exclusion": len(derived["remaining"]),
        "structural_filter": "decision_count>=5",
        "structurally_eligible_count": len(derived["structural"]),
        "frozen_confirm_prefix": {
            "trajectory_count": len(derived["confirm_ids"]),
            "source_ids_sha256": _id_digest(derived["confirm_ids"]),
            "source_ids_emitted": False,
        },
        "expansion_rule": "structural_tail_after_exact_frozen_confirm_prefix",
        "expansion_trajectory_count": len(expansion_ids),
        "expansion_source_ids_sha256": _id_digest(expansion_ids),
        "expansion_global_order_indices": [
            derived["pool_indices"][source_id] for source_id in expansion_ids
        ],
    }


def _expected_inputs(
    parent: Mapping[str, Any], *, config_sha256: str, parent_sha256: str
) -> dict[str, Any]:
    return {
        "split_config": {"path": CONFIG_PATH, "sha256": config_sha256},
        "parent_selection": {
            "path": PARENT_PATH,
            "sha256": parent_sha256,
            "generator_git_revision": parent["generator"]["git_revision"],
            "eligible_pool_count": 111,
            "eligible_pool_sha256": ELIGIBLE_POOL_SHA256,
        },
        "source_identities": _source_identities(parent),
    }


def _validate_generator(generator: Mapping[str, Any]) -> None:
    expected_paths = {
        "module_path": "code/causalcache/restoration_v2_2_label_expansion.py",
        "cli_path": "code/scripts/materialize_restoration_v2_2_label_expansion.py",
        "validator_path": "code/scripts/validate_restoration_v2_2_label_expansion.py",
    }
    expected_keys = {"git_revision", *expected_paths}
    expected_keys.update({f"{name[:-5]}_sha256" for name in expected_paths})
    if set(generator) != expected_keys:
        raise ValueError("label-expansion generator schema mismatch")
    if re.fullmatch(r"[0-9a-f]{40}", str(generator["git_revision"])) is None:
        raise ValueError("label-expansion generator Git revision is invalid")
    if any(generator.get(key) != value for key, value in expected_paths.items()):
        raise ValueError("label-expansion generator paths mismatch")
    for prefix in ("module", "cli", "validator"):
        if not _is_sha256(generator.get(f"{prefix}_sha256")):
            raise ValueError("label-expansion generator SHA256 is invalid")


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _walk_string_values(value: Any):
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_string_values(child)
    elif isinstance(value, str):
        yield value


def build_expansion_manifest(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    parent: Mapping[str, Any],
    parent_sha256: str,
    generator: Mapping[str, Any],
) -> dict[str, Any]:
    validate_config(config, source_sha256=config_sha256)
    derived = validate_parent_selection(
        parent,
        source_sha256=parent_sha256,
        config=config,
    )
    _validate_generator(generator)
    budget = int(config["label_geometry"]["deployment_budget"])
    train_records = derived["expansion"][:48]
    development_records = derived["expansion"][48:]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": STATUS,
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "structural_manifest_only": True,
        "inputs": _expected_inputs(
            parent,
            config_sha256=config_sha256,
            parent_sha256=parent_sha256,
        ),
        "generator": dict(generator),
        "derivation": _expected_derivation(derived),
        "splits": {
            "gate_train_expansion": _split_records(
                train_records,
                pool_indices=derived["pool_indices"],
                budget=budget,
            ),
            "gate_development_expansion": _split_records(
                development_records,
                pool_indices=derived["pool_indices"],
                budget=budget,
            ),
        },
        "expected_workload": config["expected_workload"],
        "leakage_firewall": {
            "prior_role_source_ids_emitted": False,
            "parent_state_records_emitted": False,
            "image_or_ocr_identity_emitted": False,
            "instruction_or_action_identity_emitted": False,
        },
    }
    validate_expansion_manifest(
        manifest,
        config=config,
        config_sha256=config_sha256,
        parent=parent,
        parent_sha256=parent_sha256,
    )
    return manifest


def validate_expansion_manifest(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    parent: Mapping[str, Any],
    parent_sha256: str,
) -> None:
    validate_config(config, source_sha256=config_sha256)
    derived = validate_parent_selection(
        parent,
        source_sha256=parent_sha256,
        config=config,
    )
    _require_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "status",
            "policy_output_generated",
            "restoration_output_generated",
            "structural_manifest_only",
            "inputs",
            "generator",
            "derivation",
            "splits",
            "expected_workload",
            "leakage_firewall",
        },
        "expansion manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["protocol_id"] != PROTOCOL_ID:
        raise ValueError("label-expansion manifest identity mismatch")
    if manifest["status"] != STATUS:
        raise ValueError("label-expansion manifest status mismatch")
    if manifest["policy_output_generated"] is not False:
        raise ValueError("label-expansion manifest must precede policy output")
    if manifest["restoration_output_generated"] is not False:
        raise ValueError("label-expansion manifest must precede restoration output")
    if manifest["structural_manifest_only"] is not True:
        raise ValueError("label-expansion output must be structural only")
    if manifest["inputs"] != _expected_inputs(
        parent,
        config_sha256=config_sha256,
        parent_sha256=parent_sha256,
    ):
        raise ValueError("label-expansion input identities mismatch")
    generator = manifest["generator"]
    if not isinstance(generator, Mapping):
        raise ValueError("label-expansion generator must be an object")
    _validate_generator(generator)
    if manifest["derivation"] != _expected_derivation(derived):
        raise ValueError("label-expansion derivation proof mismatch")
    if manifest["expected_workload"] != config["expected_workload"]:
        raise ValueError("label-expansion workload mismatch")
    if manifest["leakage_firewall"] != {
        "prior_role_source_ids_emitted": False,
        "parent_state_records_emitted": False,
        "image_or_ocr_identity_emitted": False,
        "instruction_or_action_identity_emitted": False,
    }:
        raise ValueError("label-expansion leakage firewall mismatch")

    splits = manifest["splits"]
    if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_NAMES):
        raise ValueError("label-expansion split schema mismatch")
    budget = int(config["label_geometry"]["deployment_budget"])
    expected_records = {
        "gate_train_expansion": derived["expansion"][:48],
        "gate_development_expansion": derived["expansion"][48:],
    }
    emitted_ids: list[str] = []
    for split_name in SPLIT_NAMES:
        split = splits[split_name]
        if not isinstance(split, Mapping) or set(split) != {"trajectories", "states"}:
            raise ValueError(f"{split_name} schema mismatch")
        expected = _split_records(
            expected_records[split_name],
            pool_indices=derived["pool_indices"],
            budget=budget,
        )
        if split != expected:
            raise ValueError(f"{split_name} records differ from frozen derivation")
        for trajectory in split["trajectories"]:
            if set(trajectory) != TRAJECTORY_KEYS:
                raise ValueError(f"{split_name} trajectory schema mismatch")
            emitted_ids.append(str(trajectory["source_id"]))
        for state in split["states"]:
            if set(state) != STATE_KEYS:
                raise ValueError(f"{split_name} state schema mismatch")
    if tuple(emitted_ids) != TRAIN_IDS + DEVELOPMENT_IDS:
        raise ValueError("emitted expansion IDs or order mismatch")

    forbidden_keys = FORBIDDEN_SEMANTIC_KEYS.intersection(_walk_keys(manifest))
    if forbidden_keys:
        raise ValueError("semantic content is forbidden in the structural manifest")
    forbidden_ids = set(derived["prior_ids"]) | set(derived["confirm_ids"])
    leaked_ids = forbidden_ids.intersection(_walk_string_values(manifest))
    if leaked_ids:
        raise ValueError("prior or confirm source IDs leaked into the expansion manifest")


def summary(manifest: Mapping[str, Any], *, manifest_sha256: str) -> dict[str, Any]:
    workload = manifest["expected_workload"]["total"]
    return {
        "outcome": "FROZEN_LABEL_EXPANSION_SPLIT",
        "manifest_sha256": manifest_sha256,
        "train_trajectories": len(
            manifest["splits"]["gate_train_expansion"]["trajectories"]
        ),
        "development_trajectories": len(
            manifest["splits"]["gate_development_expansion"]["trajectories"]
        ),
        "states": workload["states"],
        "full_subset_distance_rows": workload["full_subset_distance_rows"],
        "deployment_conditional_edges": workload["deployment_conditional_edges"],
        "teacher_forwards": workload["teacher_forwards"],
    }
