"""Label-blind selector preparation for the frozen long-horizon study.

The deterministic selection seal and operational latency manifest are kept
separate deliberately.  Selector decisions depend only on the frozen
substrate, frozen model payloads, and the explicit random seed.  The module has
no restoration-table or policy-runtime input.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from causalcache.data.guiodyssey_restoration_v2 import (
    _validate_tar,
    parse_canonical_jsonl,
)
from causalcache.gate_v1_training import ensemble_score
from causalcache.long_horizon_contract import (
    SELECTION_MANIFEST_FREEZE_PATH,
    LongHorizonContract,
    load_strict_json,
    load_strict_json_bytes,
)
from causalcache.long_horizon_data import (
    FEATURE_JSONL_RELATIVE_PATH,
    IMAGE_TAR_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH as SUBSTRATE_MANIFEST_RELATIVE_PATH,
    OCR_JSONL_RELATIVE_PATH,
    TRAJECTORY_JSONL_RELATIVE_PATH,
    artifact_tree_identity as substrate_tree_identity,
    load_json_object,
    read_feature_states_jsonl,
    validate_artifact as validate_substrate_artifact,
)
from causalcache.long_horizon_selectors import (
    LabelBlindSelectionSeal,
    SelectionDecision,
    conditional_b2_selection,
    deterministic_random_selection,
    independent_selection,
    load_frozen_formal58_ensembles,
    ocr_rgb_score_selection,
    ocr_rgb_similarity_scores,
    read_label_blind_selection_seal,
    recent_selection,
    residual_b2_selections,
    seal_label_blind_selections,
    summary_only_selection,
)
from causalcache.long_horizon_v4 import load_frozen_v4_ensemble
from causalcache.restoration_v2_text_backend import prepare_image_bytes


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_long_horizon_label_blind_preparation_v1"
STATUS = "SEALED_LONG_HORIZON_LABEL_BLIND_SELECTOR_PREPARATION_V1"
PAYLOAD_PREFIX = "derived/long-horizon-development-v1/selector-preparation"
SELECTION_RELATIVE_PATH = f"{PAYLOAD_PREFIX}/selector-seal.json"
MANIFEST_RELATIVE_PATH = f"{PAYLOAD_PREFIX}/manifest.json"
ARTIFACT_RELATIVE_PATHS = (SELECTION_RELATIVE_PATH, MANIFEST_RELATIVE_PATH)
OCR_BACKEND_CONFIG_PATH = "code/configs/restoration_v2_ocr_backend.json"
V4_MANIFEST_PATH = "manifest.json"
V4_LABEL_BLIND_SEAL_PATH = "label-blind-seal.json"
V4_METADATA_PATH = "residual-model-metadata.json"
V4_PROTOCOL_ID = "causalcache_set_conditioned_v4_frozen_base_residual_development_v1"
V4_LABEL_BLIND_STATUS = (
    "SEALED_LABEL_BLIND_SET_CONDITIONED_V4_FROZEN_BASE_RESIDUAL_V1"
)
V4_ZERO_ACCESS_FIELDS = (
    "fresh_label_access_count",
    "fresh_label_semantic_decode_attempt_count",
    "confirm20_access_count",
    "legacy_dev5_access_count",
    "raw_gui_access_count",
    "policy_forward_count",
    "gpu_operation_count",
)
V4_OUTER_MANIFEST_KEYS = {
    "schema_version",
    "protocol_id",
    "artifact_role",
    "repository",
    "tag",
    "source_a_git_commit",
    "execution_b_git_commit",
    "contract_sha256",
    "base_model",
    "files",
}
V4_LABEL_BLIND_SEAL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "training_report",
    "base_invariant_report",
    "frozen_base_manifest_sha256",
    "frozen_base_input_inventory",
    "source_a_git_commit",
    "execution_b_git_commit",
    "runner_freeze_sha256",
    "contract_sha256",
    "payload_inventory",
    *V4_ZERO_ACCESS_FIELDS,
}
V4_SHARED_AUDIT_PATHS = {
    "base-invariant-report.json",
    "formal58-residual-training-report.json",
}
V4_FRESH_PREDICTION_PATH = "fresh16-feature-only-predictions.json"
BUILD_CLI_PATH = "code/scripts/build_long_horizon_selector_seal.py"
VALIDATOR_CLI_PATH = "code/scripts/validate_long_horizon_selector_seal.py"
MODULE_PATH = "code/causalcache/long_horizon_prepare.py"
EXPECTED_STATE_COUNT = 48
EXPECTED_N8_STATE_COUNT = 24
EXPECTED_N16_STATE_COUNT = 24
EXPECTED_SELECTION_RECORD_COUNT = 576
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class OCRRGBStateInput:
    """Exact frozen OCR tokens and resized RGB buffers for one feature state."""

    event_ocr_tokens: Mapping[int, tuple[str, ...]]
    current_ocr_tokens: tuple[str, ...]
    event_resized_rgb_bytes: Mapping[int, bytes]
    current_resized_rgb_bytes: bytes


@dataclass(frozen=True)
class LoadedPreparationInputs:
    """All verified label-blind inputs consumed by selector preparation."""

    states: tuple[Any, ...]
    ocr_rgb_by_state: Mapping[str, OCRRGBStateInput]
    formal_ensembles: Any
    v4_ensemble: Any
    contract_provenance: Mapping[str, Any]
    substrate_provenance: Mapping[str, Any]
    model_provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ScoredSelectionSeal:
    seal: LabelBlindSelectionSeal
    latency_by_arm: tuple[Mapping[str, Any], ...]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    parsed = PurePosixPath(value)
    if (
        not value
        or parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _regular_payload(root: Path, relative: str, *, label: str) -> bytes:
    safe = _safe_relative(relative, label=label)
    path = root.joinpath(*PurePosixPath(safe).parts)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path.read_bytes()


def _bound_payload(
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, bytes, dict[str, Any]]:
    path = _safe_relative(record.get("path"), label=f"{label} path")
    digest = _require_sha256(record.get("sha256"), label=f"{label} SHA256")
    payload = _regular_payload(root, path, label=label)
    if _sha256_bytes(payload) != digest:
        raise ValueError(f"{label} payload SHA256 drifted: {path}")
    if "size_bytes" in record and record["size_bytes"] != len(payload):
        raise ValueError(f"{label} payload size drifted: {path}")
    return (
        path,
        payload,
        {
            "path": path,
            "sha256": digest,
            "size_bytes": len(payload),
        },
    )


def _require_contract(contract: Any) -> LongHorizonContract:
    if not isinstance(contract, LongHorizonContract):
        raise TypeError("selector preparation requires a validated LongHorizonContract")
    return contract


def expected_selector_names_by_budget(
    contract: LongHorizonContract,
) -> Mapping[int, tuple[str, ...]]:
    """Read the exact mixed-budget selector matrix frozen in Source-A."""
    checked = _require_contract(contract)
    matrix = checked.data.get("selector_matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("Source-A selector matrix is missing")
    result: dict[int, tuple[str, ...]] = {}
    for budget in (2, 4):
        value = matrix.get(f"budget_{budget}")
        if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
            value, Sequence
        ):
            raise ValueError(f"Source-A B={budget} selector roster is malformed")
        names = tuple(value)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Source-A selector names must be non-empty strings")
        result[budget] = names
    expected = {
        2: (
            "restoration_independent_gate",
            "recent",
            "ocr_rgb_v2",
            "v1_conditional",
            "v4_safe_frozen_base_residual",
            "random",
            "summary_only",
        ),
        4: (
            "restoration_independent_gate",
            "recent",
            "ocr_rgb_v2",
            "random",
            "summary_only",
        ),
    }
    if result != expected:
        raise ValueError("Source-A mixed-budget selector matrix drifted")
    if (
        matrix.get(
            "selector_sets_must_be_label_blind_sealed_before_any_restoration_distance"
        )
        is not True
        or matrix.get("random_seed_roster_frozen_in_source_a") is not True
        or matrix.get("random_seed_roster") != [271828]
        or matrix.get("summary_only_high_fidelity_event_count") != 0
    ):
        raise ValueError("Source-A selector firewall drifted")
    return MappingProxyType(result)


def _validate_formal_state_roster(states: Sequence[Any]) -> None:
    rows = tuple(states)
    state_ids = tuple(getattr(state, "state_id", None) for state in rows)
    if len(rows) != EXPECTED_STATE_COUNT or len(set(state_ids)) != len(rows):
        raise ValueError(
            "long-horizon selector state denominator must be 48 unique states"
        )
    n8 = [state for state in rows if len(state.candidate_event_step_ids) == 8]
    n16 = [state for state in rows if len(state.candidate_event_step_ids) == 16]
    if len(n8) != EXPECTED_N8_STATE_COUNT or len(n16) != EXPECTED_N16_STATE_COUNT:
        raise ValueError("long-horizon selector states must contain 24 n=8 and 24 n=16")
    by_source: dict[str, list[Any]] = defaultdict(list)
    for state in rows:
        by_source[state.source_id].append(state)
    if len(by_source) != 24 or any(
        sorted(len(state.candidate_event_step_ids) for state in source_states)
        != [8, 16]
        for source_states in by_source.values()
    ):
        raise ValueError("each development trajectory must contribute n=8 and n=16")


def _load_substrate_inputs(
    *,
    contract: LongHorizonContract,
    substrate_dir: str | Path,
    substrate_revision: str,
) -> tuple[tuple[Any, ...], Mapping[str, OCRRGBStateInput], Mapping[str, Any]]:
    if (
        not isinstance(substrate_revision, str)
        or GIT_SHA_RE.fullmatch(substrate_revision) is None
    ):
        raise ValueError("substrate revision must be a full immutable Git SHA")
    root = Path(substrate_dir)
    selection_path = contract.repository_root / SELECTION_MANIFEST_FREEZE_PATH
    selection_manifest = load_strict_json(
        selection_path,
        label="long-horizon selection manifest",
    )
    backend_binding = contract.data["substrate_profile"]["ocr_backend_config"]
    if backend_binding.get("path") != OCR_BACKEND_CONFIG_PATH:
        raise ValueError("Source-A OCR backend config path drifted")
    backend_path = contract.repository_root / backend_binding["path"]
    backend_payload, backend_config = load_json_object(backend_path)
    backend_sha256 = _sha256_bytes(backend_payload)
    if backend_sha256 != backend_binding.get("sha256"):
        raise ValueError("Source-A OCR backend config SHA256 drifted")
    validation = validate_substrate_artifact(
        output_dir=root,
        backend_config=backend_config,
        backend_config_sha256=backend_sha256,
        selection_manifest=selection_manifest,
        expected_dataset_repo=str(contract.data["artifact_plan"]["repo"]),
        require_formal=True,
    )
    tree = substrate_tree_identity(root)
    manifest_payload, manifest = load_json_object(
        root / SUBSTRATE_MANIFEST_RELATIVE_PATH
    )
    if manifest["inputs"]["selection_manifest"]["sha256"] != _sha256_file(
        selection_path
    ):
        raise ValueError("substrate is not bound to the current frozen selection")
    if manifest["inputs"]["ocr_backend_config"]["sha256"] != backend_sha256:
        raise ValueError("substrate is not bound to the frozen OCR backend config")

    image_payloads, _image_members = _validate_tar(root / IMAGE_TAR_RELATIVE_PATH)
    ocr_records = parse_canonical_jsonl(
        (root / OCR_JSONL_RELATIVE_PATH).read_bytes(),
        label="long-horizon selector OCR",
    )
    ocr_by_path = {record["image_member_path"]: record for record in ocr_records}
    if len(ocr_by_path) != len(ocr_records) or set(ocr_by_path) != set(image_payloads):
        raise ValueError("selector preparation OCR/image inventory drifted")
    prepared_by_path: dict[str, Any] = {}
    for path in sorted(image_payloads):
        image = image_payloads[path]
        record = ocr_by_path[path]
        if record.get("image_sha256") != _sha256_bytes(image):
            raise ValueError("selector preparation image SHA256 drifted")
        prepared = prepare_image_bytes(image)
        if prepared.resized_rgb_bytes_sha256 != record.get(
            "resized_rgb_256x256_sha256"
        ):
            raise ValueError("selector preparation frozen resize replay drifted")
        prepared_by_path[path] = prepared

    trajectories = parse_canonical_jsonl(
        (root / TRAJECTORY_JSONL_RELATIVE_PATH).read_bytes(),
        label="long-horizon selector trajectories",
    )
    states = read_feature_states_jsonl(
        (root / FEATURE_JSONL_RELATIVE_PATH).read_bytes(),
        require_formal=True,
    )
    _validate_formal_state_roster(states)
    decisions: dict[str, Mapping[str, Any]] = {}
    for trajectory in trajectories:
        for decision in trajectory["decisions"]:
            state_id = decision["state_id"]
            if state_id in decisions:
                raise ValueError("substrate contains duplicate selector state IDs")
            decisions[state_id] = decision
    if set(decisions) != {state.state_id for state in states}:
        raise ValueError("selector feature/trajectory state join drifted")

    visual_inputs: dict[str, OCRRGBStateInput] = {}
    for state in states:
        decision = decisions[state.state_id]
        post_records = decision["candidate_event_post_states"]
        event_paths = {
            int(record["event_step_id"]): str(record["post_state_member_path"])
            for record in post_records
        }
        if set(event_paths) != set(state.candidate_event_step_ids):
            raise ValueError("selector candidate post-state inventory drifted")
        current_path = str(decision["current_observation_path"])
        visual_inputs[state.state_id] = OCRRGBStateInput(
            event_ocr_tokens=MappingProxyType(
                {
                    event: tuple(ocr_by_path[path]["full_spatial_tokens"])
                    for event, path in event_paths.items()
                }
            ),
            current_ocr_tokens=tuple(ocr_by_path[current_path]["full_spatial_tokens"]),
            event_resized_rgb_bytes=MappingProxyType(
                {
                    event: prepared_by_path[path].resized_rgb_bytes
                    for event, path in event_paths.items()
                }
            ),
            current_resized_rgb_bytes=prepared_by_path[current_path].resized_rgb_bytes,
        )
    provenance = {
        "repo": manifest["dataset_repo"],
        "repo_type": "dataset",
        "revision": substrate_revision,
        "artifact_id": manifest["artifact_id"],
        "status": manifest["status"],
        "artifact_tree_sha256": tree["artifact_tree_sha256"],
        "manifest": {
            "repo": manifest["dataset_repo"],
            "repo_type": "dataset",
            "revision": substrate_revision,
            "path": SUBSTRATE_MANIFEST_RELATIVE_PATH,
            "sha256": _sha256_bytes(manifest_payload),
            "size_bytes": len(manifest_payload),
        },
        "counts": dict(manifest["counts"]),
        "state_geometry": dict(manifest["state_geometry"]),
        "inputs": dict(manifest["inputs"]),
        "inventories": dict(manifest["inventories"]),
        "source_dataset": dict(manifest["source_dataset"]),
        "generator": dict(manifest["generator"]),
        "validation_outcome": validation["outcome"],
        "policy_output_generated": False,
        "restoration_output_generated": False,
    }
    return states, MappingProxyType(visual_inputs), MappingProxyType(provenance)


def _load_formal_payloads(
    contract: LongHorizonContract,
    root: Path,
) -> tuple[dict[str, bytes], tuple[Mapping[str, Any], ...]]:
    model = contract.data["learned_model_artifacts"]["formal58_base_and_conditional"]
    records = tuple(model["ensemble_manifests"]) + tuple(model["checkpoints"])
    payloads: dict[str, bytes] = {}
    inventory: list[Mapping[str, Any]] = []
    for record in records:
        path, payload, file_record = _bound_payload(
            root,
            record,
            label="formal58 model input",
        )
        if path in payloads:
            raise ValueError("formal58 model input path is duplicated")
        payloads[path] = payload
        inventory.append(file_record)
    if len(payloads) != 12:
        raise ValueError("formal58 selector preparation requires exactly 12 files")
    return payloads, tuple(inventory)


def _v4_file_inventory(
    records: Any,
    *,
    label: str,
) -> Mapping[str, Mapping[str, Any]]:
    if isinstance(records, (str, bytes, bytearray, Mapping)) or not isinstance(
        records, Sequence
    ):
        raise ValueError(f"{label} is malformed")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"{label} record is malformed")
        path = _safe_relative(record["path"], label=f"{label} path")
        _require_sha256(record["sha256"], label=f"{label} SHA256")
        if type(record["size_bytes"]) is not int or record["size_bytes"] <= 0:
            raise ValueError(f"{label} size must be positive")
        if path in result:
            raise ValueError(f"{label} contains a duplicate path")
        result[path] = MappingProxyType(dict(record))
    return MappingProxyType(result)


def _v4_inference_paths(model: Mapping[str, Any]) -> set[str]:
    checkpoints = model.get("residual_checkpoints")
    if isinstance(checkpoints, (str, bytes, bytearray, Mapping)) or not isinstance(
        checkpoints, Sequence
    ):
        raise ValueError("v4 Source-A residual checkpoint roster is malformed")
    paths: set[str] = {V4_METADATA_PATH}
    for record in checkpoints:
        if not isinstance(record, Mapping):
            raise ValueError("v4 Source-A residual checkpoint record is malformed")
        path = _safe_relative(
            record.get("path"), label="v4 Source-A residual checkpoint path"
        )
        _require_sha256(
            record.get("sha256"), label="v4 Source-A residual checkpoint SHA256"
        )
        if path in paths:
            raise ValueError("v4 Source-A residual checkpoint path is duplicated")
        paths.add(path)
    if len(paths) != 6:
        raise ValueError("v4 selector preparation requires five residuals and metadata")
    return paths


def _validate_v4_outer_manifest(
    manifest: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    formal_model: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    if (
        set(manifest) != V4_OUTER_MANIFEST_KEYS
        or manifest.get("schema_version") != "1.0.0"
        or manifest.get("protocol_id") != V4_PROTOCOL_ID
        or manifest.get("artifact_role") != "frozen_base_residual_model_outputs"
        or manifest.get("repository") != model.get("repo")
        or manifest.get("tag")
        != "set-conditioned-v4-frozen-base-residual-development-v1"
        or manifest.get("base_model")
        != f"{formal_model.get('repo')}@{formal_model.get('revision')}"
        or GIT_SHA_RE.fullmatch(str(manifest.get("source_a_git_commit"))) is None
        or GIT_SHA_RE.fullmatch(str(manifest.get("execution_b_git_commit"))) is None
        or SHA256_RE.fullmatch(str(manifest.get("contract_sha256"))) is None
    ):
        raise ValueError("v4 outer artifact manifest schema or provenance drifted")
    inventory = _v4_file_inventory(
        manifest.get("files"), label="v4 outer manifest file inventory"
    )
    inference_paths = _v4_inference_paths(model)
    expected_paths = {
        V4_LABEL_BLIND_SEAL_PATH,
        *V4_SHARED_AUDIT_PATHS,
        *inference_paths,
    }
    if set(inventory) != expected_paths or len(inventory) != 9:
        raise ValueError("v4 outer artifact manifest file roster drifted")
    return inventory


def _validate_v4_label_blind_seal(
    seal_payload: bytes,
    *,
    outer_manifest: Mapping[str, Any],
    outer_inventory: Mapping[str, Mapping[str, Any]],
    model: Mapping[str, Any],
    formal_model: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    seal = load_strict_json_bytes(seal_payload, label="v4 label-blind seal")
    if seal_payload != _canonical_json_bytes(seal) + b"\n":
        raise ValueError("v4 label-blind seal is not canonical JSONL")
    if (
        set(seal) != V4_LABEL_BLIND_SEAL_KEYS
        or seal.get("schema_version") != "1.0.0"
        or seal.get("protocol_id") != V4_PROTOCOL_ID
        or seal.get("status") != V4_LABEL_BLIND_STATUS
        or seal.get("source_a_git_commit")
        != outer_manifest.get("source_a_git_commit")
        or seal.get("execution_b_git_commit")
        != outer_manifest.get("execution_b_git_commit")
        or seal.get("contract_sha256") != outer_manifest.get("contract_sha256")
        or SHA256_RE.fullmatch(str(seal.get("runner_freeze_sha256"))) is None
    ):
        raise ValueError("v4 label-blind seal schema or provenance drifted")
    if any(
        type(seal.get(field)) is not int or seal[field] != 0
        for field in V4_ZERO_ACCESS_FIELDS
    ):
        raise ValueError("v4 label-blind seal records nonzero protected access")

    independent_manifests = [
        record
        for record in formal_model.get("ensemble_manifests", ())
        if isinstance(record, Mapping) and record.get("family") == "independent"
    ]
    if (
        len(independent_manifests) != 1
        or seal.get("frozen_base_manifest_sha256")
        != independent_manifests[0].get("sha256")
        or not isinstance(seal.get("frozen_base_input_inventory"), Sequence)
        or isinstance(
            seal.get("frozen_base_input_inventory"),
            (str, bytes, bytearray, Mapping),
        )
    ):
        raise ValueError("v4 label-blind seal frozen-base binding drifted")

    inventory = _v4_file_inventory(
        seal.get("payload_inventory"),
        label="v4 label-blind seal payload inventory",
    )
    inference_paths = _v4_inference_paths(model)
    expected_paths = {
        V4_FRESH_PREDICTION_PATH,
        *V4_SHARED_AUDIT_PATHS,
        *inference_paths,
    }
    if set(inventory) != expected_paths or len(inventory) != 9:
        raise ValueError("v4 label-blind seal payload roster drifted")

    shared_paths = expected_paths - {V4_FRESH_PREDICTION_PATH}
    if any(
        dict(inventory[path]) != dict(outer_inventory[path])
        for path in shared_paths
    ):
        raise ValueError("v4 outer/inner file binding drifted")
    for summary_key, expected_path in (
        ("training_report", "formal58-residual-training-report.json"),
        ("base_invariant_report", "base-invariant-report.json"),
    ):
        summary = seal.get(summary_key)
        if (
            not isinstance(summary, Mapping)
            or summary.get("path") != expected_path
            or summary.get("sha256") != inventory[expected_path]["sha256"]
        ):
            raise ValueError(f"v4 label-blind {summary_key} binding drifted")

    checkpoint_by_path = {
        record["path"]: record
        for record in model["residual_checkpoints"]
    }
    if any(
        inventory[path]["sha256"] != record["sha256"]
        for path, record in checkpoint_by_path.items()
    ):
        raise ValueError("v4 Source-A/manifest residual checkpoint binding drifted")
    return seal, inventory


def _load_v4_payloads(
    contract: LongHorizonContract,
    root: Path,
) -> tuple[dict[str, bytes], bytes, tuple[Mapping[str, Any], ...]]:
    model = contract.data["learned_model_artifacts"]["v4_safe_frozen_base_residual"]
    formal_model = contract.data["learned_model_artifacts"][
        "formal58_base_and_conditional"
    ]
    manifest_path = _safe_relative(
        model.get("manifest_path"), label="v4 Source-A manifest path"
    )
    seal_path = _safe_relative(
        model.get("label_blind_seal_path"),
        label="v4 Source-A label-blind seal path",
    )
    seal_sha256 = _require_sha256(
        model.get("label_blind_seal_sha256"),
        label="v4 Source-A label-blind seal SHA256",
    )
    if manifest_path != V4_MANIFEST_PATH or seal_path != V4_LABEL_BLIND_SEAL_PATH:
        raise ValueError("v4 Source-A manifest path binding drifted")
    manifest_payload = _regular_payload(
        root,
        manifest_path,
        label="v4 outer artifact manifest",
    )
    if _sha256_bytes(manifest_payload) != model["manifest_sha256"]:
        raise ValueError("v4 outer artifact manifest SHA256 drifted")
    manifest = load_strict_json_bytes(
        manifest_payload,
        label="v4 outer artifact manifest",
    )
    outer_inventory = _validate_v4_outer_manifest(
        manifest,
        model=model,
        formal_model=formal_model,
    )
    if outer_inventory[seal_path]["sha256"] != seal_sha256:
        raise ValueError("v4 outer manifest/Source-A seal SHA256 binding drifted")
    loaded_seal_path, seal_payload, seal_record = _bound_payload(
        root,
        outer_inventory[seal_path],
        label="v4 label-blind seal",
    )
    if loaded_seal_path != V4_LABEL_BLIND_SEAL_PATH:
        raise RuntimeError("v4 label-blind seal path changed after verification")
    _seal, inner_inventory = _validate_v4_label_blind_seal(
        seal_payload,
        outer_manifest=manifest,
        outer_inventory=outer_inventory,
        model=model,
        formal_model=formal_model,
    )
    expected_paths = _v4_inference_paths(model)
    payloads: dict[str, bytes] = {}
    provenance: list[Mapping[str, Any]] = [
        {
            "path": V4_MANIFEST_PATH,
            "sha256": _sha256_bytes(manifest_payload),
            "size_bytes": len(manifest_payload),
        },
        seal_record,
    ]
    for path in sorted(expected_paths):
        if dict(inner_inventory[path]) != dict(outer_inventory[path]):
            raise ValueError("v4 outer/inner inference payload binding drifted")
        loaded_path, payload, file_record = _bound_payload(
            root,
            inner_inventory[path],
            label="v4 frozen residual input",
        )
        payloads[loaded_path] = payload
        provenance.append(file_record)
    if len(payloads) != 6 or len(provenance) != 8:
        raise ValueError("v4 selector preparation requires five residuals and metadata")
    return payloads, manifest_payload, tuple(provenance)


def load_preparation_inputs(
    *,
    contract: LongHorizonContract,
    substrate_dir: str | Path,
    substrate_revision: str,
    formal_model_dir: str | Path,
    v4_model_dir: str | Path,
    selection_freeze_git_commit: str,
) -> LoadedPreparationInputs:
    """Fresh-load and verify every label-blind substrate/model input."""
    checked = _require_contract(contract)
    if (
        checked.data["learned_model_artifacts"].get(
            "execution_b_must_fresh_download_and_verify_exact_rosters_before_selector_scoring"
        )
        is not True
        or checked.data["learned_model_artifacts"].get(
            "training_optimizer_or_artifact_substitution_allowed"
        )
        is not False
    ):
        raise ValueError("Source-A learned-model firewall drifted")
    states, visual_inputs, substrate_provenance = _load_substrate_inputs(
        contract=checked,
        substrate_dir=substrate_dir,
        substrate_revision=substrate_revision,
    )
    formal_payloads, formal_files = _load_formal_payloads(
        checked,
        Path(formal_model_dir),
    )
    formal_ensembles = load_frozen_formal58_ensembles(checked, formal_payloads)
    residual_payloads, residual_manifest, residual_files = _load_v4_payloads(
        checked,
        Path(v4_model_dir),
    )
    formal_model = checked.data["learned_model_artifacts"][
        "formal58_base_and_conditional"
    ]
    independent_paths = {
        record["path"]
        for record in formal_model["ensemble_manifests"]
        if record["family"] == "independent"
    } | {
        record["path"]
        for record in formal_model["checkpoints"]
        if record["family"] == "independent"
    }
    v4_ensemble = load_frozen_v4_ensemble(
        checked,
        formal_payloads={path: formal_payloads[path] for path in independent_paths},
        residual_payloads=residual_payloads,
        residual_manifest_payload=residual_manifest,
    )
    residual_model = checked.data["learned_model_artifacts"][
        "v4_safe_frozen_base_residual"
    ]
    models = {
        "formal58": {
            "repo": formal_model["repo"],
            "repo_type": formal_model["repo_type"],
            "private": formal_model["private"],
            "revision": formal_model["revision"],
            "files": list(formal_files),
            "family_seed_roster": {
                "conditional": [0, 1, 2, 3, 4],
                "independent": [0, 1, 2, 3, 4],
            },
            "checkpoint_load_count": 10,
        },
        "v4_frozen_base_residual": {
            "repo": residual_model["repo"],
            "repo_type": residual_model["repo_type"],
            "private": residual_model["private"],
            "revision": residual_model["revision"],
            "files": list(residual_files),
            "residual_seed_roster": [0, 1, 2, 3, 4],
            "training_allowed": False,
        },
    }
    if (
        not isinstance(selection_freeze_git_commit, str)
        or GIT_SHA_RE.fullmatch(selection_freeze_git_commit) is None
    ):
        raise ValueError("selection-freeze Git commit must be a full lowercase SHA")
    selection_path = checked.repository_root / SELECTION_MANIFEST_FREEZE_PATH
    selection_payload = selection_path.read_bytes()
    selection_manifest = load_strict_json_bytes(
        selection_payload,
        label="long-horizon selection freeze",
    )
    source_a_git_commit = selection_manifest.get("generator", {}).get("git_revision")
    if (
        not isinstance(source_a_git_commit, str)
        or GIT_SHA_RE.fullmatch(source_a_git_commit) is None
    ):
        raise ValueError("selection freeze omits the Source-A Git commit")
    contract_provenance = {
        "protocol_id": checked.data["protocol_id"],
        "source_a": {
            "git_commit": source_a_git_commit,
            "config": {
                "path": checked.source_path.relative_to(
                    checked.repository_root
                ).as_posix(),
                "sha256": checked.sha256,
                "size_bytes": checked.source_path.stat().st_size,
            },
        },
        "selection_freeze": {
            "git_commit": selection_freeze_git_commit,
            "manifest": {
                "path": SELECTION_MANIFEST_FREEZE_PATH,
                "sha256": _sha256_bytes(selection_payload),
                "size_bytes": len(selection_payload),
            },
        },
    }
    return LoadedPreparationInputs(
        states=states,
        ocr_rgb_by_state=visual_inputs,
        formal_ensembles=formal_ensembles,
        v4_ensemble=v4_ensemble,
        contract_provenance=MappingProxyType(contract_provenance),
        substrate_provenance=substrate_provenance,
        model_provenance=MappingProxyType(models),
    )


def _latency_records(
    values: Mapping[tuple[int, str], Sequence[int]],
) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    for (budget, selector), samples in sorted(values.items()):
        durations = tuple(samples)
        if not durations or any(
            type(value) is not int or value < 0 for value in durations
        ):
            raise ValueError("selector latency samples must be non-negative integers")
        total = sum(durations)
        records.append(
            {
                "budget_event_capacity": budget,
                "selector_name": selector,
                "invocation_count": len(durations),
                "total_nanoseconds": total,
                "minimum_nanoseconds": min(durations),
                "maximum_nanoseconds": max(durations),
                "mean_nanoseconds": total / len(durations),
            }
        )
    return tuple(records)


def _ocr_rgb_decision(
    state: Any,
    visual: OCRRGBStateInput,
    budget: int,
) -> SelectionDecision:
    scores = ocr_rgb_similarity_scores(
        event_step_ids=state.candidate_event_step_ids,
        event_ocr_tokens=visual.event_ocr_tokens,
        current_ocr_tokens=visual.current_ocr_tokens,
        event_resized_rgb_bytes=visual.event_resized_rgb_bytes,
        current_resized_rgb_bytes=visual.current_resized_rgb_bytes,
    )
    return ocr_rgb_score_selection(
        state,
        scores,
        budget_event_capacity=budget,
    )


def _conditional_decision(state: Any, score: Callable[..., float]) -> SelectionDecision:
    return conditional_b2_selection(state, score).decision


def _v4_safe_decision(state: Any, ensemble: Any) -> SelectionDecision:
    return residual_b2_selections(state, ensemble.predict(state)).safe


def score_and_seal_label_blind_selectors(
    *,
    states: Sequence[Any],
    ocr_rgb_by_state: Mapping[str, OCRRGBStateInput],
    formal_ensembles: Any,
    v4_ensemble: Any,
    random_seed: int,
    selector_names_by_budget: Mapping[int, Sequence[str]],
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    require_formal: bool = True,
) -> ScoredSelectionSeal:
    """Run every frozen selector arm before any restoration output is available."""
    rows = tuple(states)
    if require_formal:
        _validate_formal_state_roster(rows)
    if not rows or len({state.state_id for state in rows}) != len(rows):
        raise ValueError("selector scoring requires non-empty unique feature states")
    if set(ocr_rgb_by_state) != {state.state_id for state in rows}:
        raise ValueError("selector OCR/RGB state inventory drifted")
    if type(random_seed) is not int or random_seed < 0:
        raise ValueError("selector random seed must be a non-negative integer")
    expected_matrix = {
        2: (
            "restoration_independent_gate",
            "recent",
            "ocr_rgb_v2",
            "v1_conditional",
            "v4_safe_frozen_base_residual",
            "random",
            "summary_only",
        ),
        4: (
            "restoration_independent_gate",
            "recent",
            "ocr_rgb_v2",
            "random",
            "summary_only",
        ),
    }
    observed_matrix = {
        budget: tuple(names) for budget, names in selector_names_by_budget.items()
    }
    if observed_matrix != expected_matrix:
        raise ValueError("selector scoring matrix differs from Source-A")
    independent_score = ensemble_score(formal_ensembles.independent)
    conditional_score = ensemble_score(formal_ensembles.conditional)
    decisions: list[SelectionDecision] = []
    latency: dict[tuple[int, str], list[int]] = defaultdict(list)

    def invoke(
        budget: int,
        selector: str,
        operation: Callable[[], SelectionDecision],
    ) -> None:
        start = clock_ns()
        decision = operation()
        end = clock_ns()
        if type(start) is not int or type(end) is not int or end < start:
            raise ValueError(
                "selector latency clock must be monotonic integer nanoseconds"
            )
        if (
            decision.selector_name != selector
            or decision.budget_event_capacity != budget
        ):
            raise RuntimeError("selector arm returned the wrong decision identity")
        decisions.append(decision)
        latency[(budget, selector)].append(end - start)

    for state in rows:
        visual = ocr_rgb_by_state[state.state_id]
        for budget, names in expected_matrix.items():
            for selector in names:
                if selector == "restoration_independent_gate":
                    operation = partial(
                        independent_selection,
                        state,
                        independent_score,
                        budget_event_capacity=budget,
                    )
                elif selector == "recent":
                    operation = partial(
                        recent_selection,
                        state,
                        budget_event_capacity=budget,
                    )
                elif selector == "ocr_rgb_v2":
                    operation = partial(_ocr_rgb_decision, state, visual, budget)
                elif selector == "v1_conditional":
                    operation = partial(_conditional_decision, state, conditional_score)
                elif selector == "v4_safe_frozen_base_residual":
                    operation = partial(_v4_safe_decision, state, v4_ensemble)
                elif selector == "random":
                    operation = partial(
                        deterministic_random_selection,
                        state,
                        budget_event_capacity=budget,
                        seed=random_seed,
                    )
                elif selector == "summary_only":
                    operation = partial(
                        summary_only_selection,
                        state,
                        budget_event_capacity=budget,
                    )
                else:
                    raise RuntimeError(f"unhandled frozen selector arm: {selector}")
                invoke(budget, selector, operation)

    seal = seal_label_blind_selections(
        rows,
        decisions,
        expected_selector_names_by_budget=expected_matrix,
    )
    expected_records = len(rows) * sum(len(names) for names in expected_matrix.values())
    if len(seal.records) != expected_records:
        raise RuntimeError("selector seal record denominator drifted")
    if require_formal and expected_records != EXPECTED_SELECTION_RECORD_COUNT:
        raise RuntimeError("formal selector seal must contain 576 records")
    return ScoredSelectionSeal(
        seal=seal,
        latency_by_arm=_latency_records(latency),
    )


def _generator_provenance(
    *,
    repository_root: Path,
    git_revision: str,
) -> Mapping[str, Any]:
    if not isinstance(git_revision, str) or GIT_SHA_RE.fullmatch(git_revision) is None:
        raise ValueError(
            "selector preparation Git revision must be a full lowercase SHA"
        )
    files = []
    for relative in (MODULE_PATH, BUILD_CLI_PATH, VALIDATOR_CLI_PATH):
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"selector preparation source is missing: {relative}"
            )
        files.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return MappingProxyType(
        {
            "git_revision": git_revision,
            "source_files": files,
        }
    )


def _manifest(
    *,
    loaded: LoadedPreparationInputs,
    scored: ScoredSelectionSeal,
    random_seed: int,
    selector_matrix: Mapping[int, Sequence[str]],
    generator: Mapping[str, Any],
) -> Mapping[str, Any]:
    states = loaded.states
    n8 = sum(len(state.candidate_event_step_ids) == 8 for state in states)
    n16 = sum(len(state.candidate_event_step_ids) == 16 for state in states)
    selection_record = {
        "path": SELECTION_RELATIVE_PATH,
        "sha256": scored.seal.sha256,
        "size_bytes": len(scored.seal.payload_bytes),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": STATUS,
        "contract": dict(loaded.contract_provenance),
        "substrate": dict(loaded.substrate_provenance),
        "models": dict(loaded.model_provenance),
        "random_seed": random_seed,
        "selector_names_by_budget": {
            str(budget): list(names) for budget, names in selector_matrix.items()
        },
        "counts": {
            "state_count": len(states),
            "n8_state_count": n8,
            "n16_state_count": n16,
            "selection_record_count": len(scored.seal.records),
            "budget_selector_arm_count": sum(
                len(names) for names in selector_matrix.values()
            ),
        },
        "selection_seal": selection_record,
        "latency": {
            "clock": "time.perf_counter_ns",
            "unit": "nanoseconds",
            "selection_sha256_excludes_operational_latency": True,
            "arms": [dict(record) for record in scored.latency_by_arm],
        },
        "generator": dict(generator),
        "restoration_label_access_count": 0,
        "restoration_distance_access_count": 0,
        "policy_forward_count": 0,
        "optimizer_step_count": 0,
        "unopened_reserve_row_access_count": 0,
        "selection_scoring_completed_before_restoration": True,
    }


def artifact_tree_identity(output_dir: str | Path) -> Mapping[str, Any]:
    root = Path(output_dir)
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != set(ARTIFACT_RELATIVE_PATHS):
        raise ValueError(
            "selector preparation exact-two inventory drifted: "
            f"missing={sorted(set(ARTIFACT_RELATIVE_PATHS) - observed)}, "
            f"extra={sorted(observed - set(ARTIFACT_RELATIVE_PATHS))}"
        )
    files = [
        {
            "path": relative,
            "sha256": _sha256_file(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in ARTIFACT_RELATIVE_PATHS
    ]
    return {
        "files": files,
        "artifact_tree_sha256": _sha256_bytes(_canonical_json_bytes(files)),
    }


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def write_selector_seal_artifact(
    *,
    output_dir: str | Path,
    loaded: LoadedPreparationInputs,
    scored: ScoredSelectionSeal,
    random_seed: int,
    selector_matrix: Mapping[int, Sequence[str]],
    generator: Mapping[str, Any],
) -> Mapping[str, Any]:
    root = Path(output_dir)
    if os.path.lexists(root):
        raise FileExistsError(f"selector preparation output already exists: {root}")
    manifest = _manifest(
        loaded=loaded,
        scored=scored,
        random_seed=random_seed,
        selector_matrix=selector_matrix,
        generator=generator,
    )
    root.mkdir(parents=True)
    _write_once(root / SELECTION_RELATIVE_PATH, scored.seal.payload_bytes)
    _write_once(root / MANIFEST_RELATIVE_PATH, _pretty_json_bytes(manifest))
    return artifact_tree_identity(root)


def build_selector_seal_artifact(
    *,
    contract: LongHorizonContract,
    substrate_dir: str | Path,
    substrate_revision: str,
    formal_model_dir: str | Path,
    v4_model_dir: str | Path,
    random_seed: int,
    git_revision: str,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    """Load, score, seal, and persist all 576 label-blind selector records."""
    checked = _require_contract(contract)
    matrix = expected_selector_names_by_budget(checked)
    if random_seed != checked.data["selector_matrix"]["random_seed_roster"][0]:
        raise ValueError("selector random seed differs from the Source-A roster")
    loaded = load_preparation_inputs(
        contract=checked,
        substrate_dir=substrate_dir,
        substrate_revision=substrate_revision,
        formal_model_dir=formal_model_dir,
        v4_model_dir=v4_model_dir,
        selection_freeze_git_commit=git_revision,
    )
    scored = score_and_seal_label_blind_selectors(
        states=loaded.states,
        ocr_rgb_by_state=loaded.ocr_rgb_by_state,
        formal_ensembles=loaded.formal_ensembles,
        v4_ensemble=loaded.v4_ensemble,
        random_seed=random_seed,
        selector_names_by_budget=matrix,
        require_formal=True,
    )
    generator = _generator_provenance(
        repository_root=checked.repository_root,
        git_revision=git_revision,
    )
    tree = write_selector_seal_artifact(
        output_dir=output_dir,
        loaded=loaded,
        scored=scored,
        random_seed=random_seed,
        selector_matrix=matrix,
        generator=generator,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "outcome": "BUILT_LONG_HORIZON_LABEL_BLIND_SELECTOR_SEAL",
        "selection_sha256": scored.seal.sha256,
        "record_count": len(scored.seal.records),
        **tree,
    }


def _validate_latency(
    value: Any,
    *,
    selector_matrix: Mapping[int, Sequence[str]],
    state_count: int,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Mapping) or set(value) != {
        "clock",
        "unit",
        "selection_sha256_excludes_operational_latency",
        "arms",
    }:
        raise ValueError("selector preparation latency schema drifted")
    if (
        value["clock"] != "time.perf_counter_ns"
        or value["unit"] != "nanoseconds"
        or value["selection_sha256_excludes_operational_latency"] is not True
        or not isinstance(value["arms"], list)
    ):
        raise ValueError("selector preparation latency identity drifted")
    expected_arms = {
        (budget, selector)
        for budget, names in selector_matrix.items()
        for selector in names
    }
    records: list[Mapping[str, Any]] = []
    observed: set[tuple[int, str]] = set()
    expected_keys = {
        "budget_event_capacity",
        "selector_name",
        "invocation_count",
        "total_nanoseconds",
        "minimum_nanoseconds",
        "maximum_nanoseconds",
        "mean_nanoseconds",
    }
    for record in value["arms"]:
        if not isinstance(record, Mapping) or set(record) != expected_keys:
            raise ValueError("selector latency arm schema drifted")
        key = (record["budget_event_capacity"], record["selector_name"])
        if key in observed or key not in expected_arms:
            raise ValueError("selector latency arm identity drifted")
        observed.add(key)
        integers = (
            record["total_nanoseconds"],
            record["minimum_nanoseconds"],
            record["maximum_nanoseconds"],
        )
        if (
            record["invocation_count"] != state_count
            or any(type(number) is not int or number < 0 for number in integers)
            or not isinstance(record["mean_nanoseconds"], (int, float))
            or isinstance(record["mean_nanoseconds"], bool)
            or not math.isfinite(float(record["mean_nanoseconds"]))
            or record["mean_nanoseconds"] < 0
            or record["minimum_nanoseconds"] > record["maximum_nanoseconds"]
            or not (
                record["minimum_nanoseconds"]
                <= float(record["mean_nanoseconds"])
                <= record["maximum_nanoseconds"]
            )
            or not math.isclose(
                float(record["mean_nanoseconds"]),
                record["total_nanoseconds"] / state_count,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("selector latency arm accounting drifted")
        records.append(record)
    if observed != expected_arms:
        raise ValueError("selector latency arm roster is incomplete")
    arm_order = [
        (record["budget_event_capacity"], record["selector_name"]) for record in records
    ]
    if arm_order != sorted(arm_order):
        raise ValueError("selector latency arms are not in canonical order")
    return tuple(records)


def validate_selector_seal_artifact(
    *,
    output_dir: str | Path,
    contract: LongHorizonContract,
    substrate_dir: str | Path,
    substrate_revision: str,
    formal_model_dir: str | Path,
    v4_model_dir: str | Path,
    random_seed: int,
    expected_git_revision: str | None = None,
) -> Mapping[str, Any]:
    """Fresh-load all inputs and byte-replay the complete 576-record seal."""
    checked = _require_contract(contract)
    root = Path(output_dir)
    tree = artifact_tree_identity(root)
    manifest_payload = (root / MANIFEST_RELATIVE_PATH).read_bytes()
    manifest = load_strict_json_bytes(
        manifest_payload,
        label="selector preparation manifest",
    )
    if manifest_payload != _pretty_json_bytes(manifest):
        raise ValueError("selector preparation manifest is not canonical pretty JSON")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "contract",
        "substrate",
        "models",
        "random_seed",
        "selector_names_by_budget",
        "counts",
        "selection_seal",
        "latency",
        "generator",
        "restoration_label_access_count",
        "restoration_distance_access_count",
        "policy_forward_count",
        "optimizer_step_count",
        "unopened_reserve_row_access_count",
        "selection_scoring_completed_before_restoration",
    }
    if set(manifest) != expected_keys or (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["protocol_id"] != PROTOCOL_ID
        or manifest["status"] != STATUS
        or manifest["random_seed"] != random_seed
        or manifest["restoration_label_access_count"] != 0
        or manifest["restoration_distance_access_count"] != 0
        or manifest["policy_forward_count"] != 0
        or manifest["optimizer_step_count"] != 0
        or manifest["unopened_reserve_row_access_count"] != 0
        or manifest["selection_scoring_completed_before_restoration"] is not True
    ):
        raise ValueError("selector preparation manifest identity or firewall drifted")
    matrix = expected_selector_names_by_budget(checked)
    if random_seed != checked.data["selector_matrix"]["random_seed_roster"][0]:
        raise ValueError("selector random seed differs from the Source-A roster")
    expected_matrix_payload = {
        str(budget): list(names) for budget, names in matrix.items()
    }
    if manifest["selector_names_by_budget"] != expected_matrix_payload:
        raise ValueError("selector preparation manifest matrix drifted")

    loaded = load_preparation_inputs(
        contract=checked,
        substrate_dir=substrate_dir,
        substrate_revision=substrate_revision,
        formal_model_dir=formal_model_dir,
        v4_model_dir=v4_model_dir,
        selection_freeze_git_commit=manifest["generator"].get("git_revision", ""),
    )
    selection_payload = (root / SELECTION_RELATIVE_PATH).read_bytes()
    selection_record = manifest["selection_seal"]
    if selection_record != {
        "path": SELECTION_RELATIVE_PATH,
        "sha256": _sha256_bytes(selection_payload),
        "size_bytes": len(selection_payload),
    }:
        raise ValueError("selector preparation seal file binding drifted")
    persisted = read_label_blind_selection_seal(
        selection_payload,
        loaded.states,
        expected_sha256=selection_record["sha256"],
    )
    replay = score_and_seal_label_blind_selectors(
        states=loaded.states,
        ocr_rgb_by_state=loaded.ocr_rgb_by_state,
        formal_ensembles=loaded.formal_ensembles,
        v4_ensemble=loaded.v4_ensemble,
        random_seed=random_seed,
        selector_names_by_budget=matrix,
        require_formal=True,
    )
    if replay.seal.payload_bytes != persisted.payload_bytes:
        raise ValueError("selector preparation decision byte replay drifted")

    counts = {
        "state_count": EXPECTED_STATE_COUNT,
        "n8_state_count": EXPECTED_N8_STATE_COUNT,
        "n16_state_count": EXPECTED_N16_STATE_COUNT,
        "selection_record_count": EXPECTED_SELECTION_RECORD_COUNT,
        "budget_selector_arm_count": 12,
    }
    if manifest["counts"] != counts or len(persisted.records) != 576:
        raise ValueError("selector preparation formal denominator drifted")
    latency_records = _validate_latency(
        manifest["latency"],
        selector_matrix=matrix,
        state_count=EXPECTED_STATE_COUNT,
    )
    generator = _generator_provenance(
        repository_root=checked.repository_root,
        git_revision=manifest["generator"].get("git_revision", ""),
    )
    if manifest["generator"] != dict(generator):
        raise ValueError("selector preparation generator provenance drifted")
    if (
        expected_git_revision is not None
        and manifest["generator"]["git_revision"] != expected_git_revision
    ):
        raise ValueError("selector preparation Git revision differs from expectation")
    if (
        manifest["contract"] != dict(loaded.contract_provenance)
        or manifest["substrate"] != dict(loaded.substrate_provenance)
        or manifest["models"] != dict(loaded.model_provenance)
    ):
        raise ValueError("selector preparation input provenance drifted")
    # note (luojiaxuan): This final reconstruction includes recorded latency
    # verbatim; selector decisions were independently recomputed above.
    reconstructed = _manifest(
        loaded=loaded,
        scored=ScoredSelectionSeal(
            seal=persisted,
            latency_by_arm=latency_records,
        ),
        random_seed=random_seed,
        selector_matrix=matrix,
        generator=generator,
    )
    if manifest_payload != _pretty_json_bytes(reconstructed):
        raise ValueError("selector preparation canonical manifest replay drifted")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "outcome": "PASSED_LONG_HORIZON_LABEL_BLIND_SELECTOR_SEAL_VALIDATION",
        "selection_sha256": persisted.sha256,
        "record_count": len(persisted.records),
        "state_count": len(loaded.states),
        "n8_state_count": EXPECTED_N8_STATE_COUNT,
        "n16_state_count": EXPECTED_N16_STATE_COUNT,
        "formal58_checkpoint_count": 10,
        "v4_residual_checkpoint_count": 5,
        "restoration_label_access_count": 0,
        "restoration_distance_access_count": 0,
        **tree,
    }


__all__ = [
    "ARTIFACT_RELATIVE_PATHS",
    "LoadedPreparationInputs",
    "MANIFEST_RELATIVE_PATH",
    "OCRRGBStateInput",
    "PROTOCOL_ID",
    "SELECTION_RELATIVE_PATH",
    "ScoredSelectionSeal",
    "artifact_tree_identity",
    "build_selector_seal_artifact",
    "expected_selector_names_by_budget",
    "load_preparation_inputs",
    "score_and_seal_label_blind_selectors",
    "validate_selector_seal_artifact",
    "write_selector_seal_artifact",
]
