"""Fail-closed source contract for the 12-state train-only throughput pilot."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.1.0"
PROTOCOL_ID = (
    "causalcache_set_utility_train_only_throughput_pilot_source_"
    "v2_candidate_schedule_key_repair"
)
STATUS = (
    "SOURCE_ONLY_TRAIN_THROUGHPUT_PILOT_V2_"
    "CANDIDATE_SCHEDULE_KEY_REPAIR_FROZEN"
)
VALIDATION_STATUS = (
    "VALID_SET_UTILITY_TRAIN_ONLY_THROUGHPUT_PILOT_SOURCE_V2_"
    "CANDIDATE_SCHEDULE_KEY_REPAIR"
)

CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_train_only_throughput_pilot_"
    "v2_candidate_schedule_key_repair.json"
)
CONTRACT_PATH = (
    "code/causalcache/set_utility_throughput_pilot_contract_v1.py"
)
CLI_PATH = (
    "code/scripts/validate_set_utility_throughput_pilot_contract_v1.py"
)
FREEZE_B_V2_MANIFEST_PATH = (
    "data/manifests/set_utility_freeze_b_v2_terminal_index_repair.json"
)
PROCESSOR_RESULT_SUMMARY_PATH = (
    "data/results/set_utility_processor_freeze_v2_image_contract_repair/summary.json"
)
PROCESSOR_PUBLICATION_SUMMARY_PATH = (
    "data/results/set_utility_processor_freeze_v2_image_contract_repair_"
    "publication_v1/summary.json"
)
MODEL_SNAPSHOT_MANIFEST_PATH = "code/configs/gui_owl_1_5_8b_snapshot.json"
PARENT_FAILURE_SUMMARY_PATH = (
    "data/results/set_utility_train_only_throughput_pilot_v1/summary.json"
)

FREEZE_B_V2_MANIFEST_SHA256 = (
    "915892ef2e0f1495da4b9e409b3e7a112dc86cda0b06384e8a1b7f8053581d30"
)
PROCESSOR_RESULT_SUMMARY_SHA256 = (
    "87a9a17eb07397f8395e63a2ed088f5eedb94d39d223f0be84836416fca930a3"
)
PROCESSOR_PUBLICATION_SUMMARY_SHA256 = (
    "ec415e6af2cd12ad1b32f30788b5aac669fcf68478a1e77c54295c400a721930"
)
MODEL_SNAPSHOT_MANIFEST_SHA256 = (
    "50b675ec31c5c46dbb0d44c137a808fffb9d054916d39b596648d4eb9df7cbc3"
)
PARENT_FAILURE_SUMMARY_SHA256 = (
    "d4fde65ff8c9fc1c9b67a38c980de1edd89c99b0d6958d238b6aa0417aa3f45c"
)

PARENT_SOURCE_GIT_REVISION = "5158f2ad04e456fd088765973da6a99064b4efcf"
PARENT_ENVELOPE_GIT_REVISION = "c7d5b31f8235124699ecbee5beb5240c72eb8977"
PARENT_EXECUTION_ENVELOPE_SHA256 = (
    "adf50363d9b3623b84b9229d709f796d7b1971faf7e4056c468cfe3b55ce228e"
)
EXPECTED_PARENT_FAILURE_STATUS = (
    "INVALID_SET_UTILITY_TRAIN_ONLY_THROUGHPUT_PILOT_V1_"
    "CANDIDATE_SCHEDULE_CANONICALIZATION_CONTRACT_DRIFT"
)
CANDIDATE_SCHEDULE_KEY_REPAIR_ID = (
    "candidate_schedule_state_count_integer_key_reconstruction_v2"
)
CANDIDATE_SCHEDULE_INTEGER_KEY_PATH = (
    "exact_label_schedule",
    "state_count_by_candidate_count",
)

EXPECTED_FREEZE_STATUS = (
    "POLICY_BLIND_FREEZE_B_V2_TERMINAL_INDEX_REPAIR_COMPLETED"
)
EXPECTED_PROCESSOR_RESULT_STATUS = (
    "VALID_RECORDED_SET_UTILITY_PROCESSOR_FREEZE_V2_IMAGE_CONTRACT_REPAIR"
)
EXPECTED_PROCESSOR_POSTFLIGHT_STATUS = (
    "VALID_COMPLETED_SET_UTILITY_PROCESSOR_FREEZE_V2_IMAGE_CONTRACT_REPAIR"
)
EXPECTED_PUBLICATION_FINALIZATION_STATUS = (
    "FINALIZED_PROCESSOR_V2_IMMUTABLE_HF_PUBLICATION"
)
EXPECTED_PUBLICATION_RECEIPT_STATUS = (
    "PUBLISHED_PROCESSOR_V2_IMMUTABLE_HF_VERIFIED"
)
EXPECTED_HF_REPO = "gavinlaw/causalcache-set-utility-new-development-mobile"
EXPECTED_HF_REVISION = "c20bab8df424dc9e45ece1084f3d1dc035dd1ed8"
EXPECTED_HF_TAG = "phase1-b2-processor-freeze-v2-image-contract-repair"
EXPECTED_HF_PREFIX = "artifacts/processor-freeze-v2-image-contract-repair"
EXPECTED_FORMAL_FILE_COUNT = 23
EXPECTED_FORMAL_TOTAL_BYTES = 18_730_620_511
EXPECTED_REMOTE_FILE_COUNT = 25
EXPECTED_FORMAL_INVENTORY_SHA256 = (
    "7c2a971658ad9a6bbc639446e9326e9dd4718b8d2d77186f52a17f10f3b32fc1"
)
EXPECTED_CANDIDATE_SCHEDULE_PATH = "processor-candidate-freeze-schedule.json"
EXPECTED_CANDIDATE_SCHEDULE_SHA256 = (
    "186f2952108273672c6cdbf963094754298d23693fd6268a72b6223e99c2299d"
)
EXPECTED_CANDIDATE_SCHEDULE_BYTES = 18_718_642
EXPECTED_CANDIDATE_INVENTORY_SHA256 = (
    "c0b0fa26c6d64705d5d533200d96c624eb58b75cd7f9b4b9564507d7442b04ad"
)
EXPECTED_MODEL_REPO = "mPLUG/GUI-Owl-1.5-8B-Instruct"
EXPECTED_MODEL_REVISION = "06d5faecff74840bab2be2425e9c42667a5d04fc"
EXPECTED_MODEL_FILE_COUNT = 14
EXPECTED_MODEL_TOTAL_BYTES = 17_545_907_171

EXPECTED_STATE_IDS = (
    "0296753837938323:decision:006",
    "0310939638496410:decision:006",
    "0336706763935531:decision:006",
    "0363106249041737:decision:006",
    "0214209291313446:decision:010",
    "0268406573756492:decision:010",
    "0271654003819383:decision:010",
    "0279447750102246:decision:010",
    "0018651612081817:decision:018",
    "0047881550315557:decision:018",
    "0053721915679562:decision:018",
    "0117550419342475:decision:018",
)
EXPECTED_STRATA = (
    {
        "candidate_capacity_stratum": "decisions_6_9",
        "decision_step_id": 6,
        "initial_candidate_count": 4,
        "state_count": 4,
    },
    {
        "candidate_capacity_stratum": "decisions_10_17",
        "decision_step_id": 10,
        "initial_candidate_count": 8,
        "state_count": 4,
    },
    {
        "candidate_capacity_stratum": "decisions_18_plus",
        "decision_step_id": 18,
        "initial_candidate_count": 16,
        "state_count": 4,
    },
)
ALLOWED_MICROBATCH_ORDER = (1, 2)
WORKER_COUNT = 4
NATIVE_CALL_CEILING = 84

RUNTIME_ENTRYPOINT_PATHS = (
    "code/causalcache/set_utility_throughput_pilot_envelope_v1.py",
    "code/causalcache/set_utility_throughput_pilot_execution_v1.py",
    "code/causalcache/set_utility_gui_owl_v2_1_throughput_adapter.py",
    "code/causalcache/set_utility_throughput_pilot.py",
    "code/causalcache/set_utility_throughput_pilot_pair_v1.py",
    "code/scripts/aggregate_set_utility_throughput_pilot_v1.py",
    "code/scripts/materialize_set_utility_throughput_pilot_envelope_v1.py",
    "code/scripts/run_set_utility_throughput_pilot_worker_v1.py",
    "code/scripts/validate_set_utility_throughput_pilot_envelope_v1.py",
)
EXPECTED_RUNTIME_TRANSITIVE_SOURCE_PATHS = (
    "code/causalcache/__init__.py",
    "code/causalcache/attribution.py",
    "code/causalcache/contracts.py",
    "code/causalcache/data/__init__.py",
    "code/causalcache/data/guiodyssey.py",
    "code/causalcache/data/guiodyssey_independent.py",
    "code/causalcache/data/guiodyssey_restoration_v2.py",
    "code/causalcache/data/restoration_v2_selection.py",
    "code/causalcache/low_fidelity_v2.py",
    "code/causalcache/policy/__init__.py",
    "code/causalcache/policy/gui_owl_v2.py",
    "code/causalcache/policy/gui_owl_v2_1.py",
    "code/causalcache/policy/gui_owl_v2_1_runtime.py",
    "code/causalcache/policy/gui_owl_v2_1_throughput_runtime.py",
    "code/causalcache/policy/gui_owl_v2_runtime.py",
    "code/causalcache/policy/gui_owl_v2_vision.py",
    "code/causalcache/restoration_v2_baselines.py",
    "code/causalcache/restoration_v2_contract.py",
    "code/causalcache/restoration_v2_text_backend.py",
    "code/causalcache/schema.py",
    "code/causalcache/set_utility_consumed_ledger.py",
    "code/causalcache/set_utility_consumed_ledger_contract.py",
    "code/causalcache/set_utility_full_pool.py",
    "code/causalcache/set_utility_full_pool_inventory_v1.py",
    "code/causalcache/set_utility_gui_owl_v2_1_throughput_adapter.py",
    "code/causalcache/set_utility_label_inputs.py",
    "code/causalcache/set_utility_label_producer.py",
    "code/causalcache/set_utility_label_schedule.py",
    "code/causalcache/set_utility_long_pool.py",
    "code/causalcache/set_utility_processor_artifacts.py",
    "code/causalcache/set_utility_processor_freeze.py",
    "code/causalcache/set_utility_processor_prompt.py",
    "code/causalcache/set_utility_processor_substrate.py",
    "code/causalcache/set_utility_split_validation.py",
    "code/causalcache/set_utility_throughput_pilot.py",
    "code/causalcache/set_utility_throughput_pilot_contract_v1.py",
    "code/causalcache/set_utility_throughput_pilot_envelope_v1.py",
    "code/causalcache/set_utility_throughput_pilot_execution_v1.py",
    "code/causalcache/set_utility_throughput_pilot_pair_v1.py",
    "code/scripts/aggregate_set_utility_throughput_pilot_v1.py",
    "code/scripts/materialize_set_utility_throughput_pilot_envelope_v1.py",
    "code/scripts/run_set_utility_throughput_pilot_worker_v1.py",
    "code/scripts/validate_set_utility_throughput_pilot_envelope_v1.py",
)


@dataclass(frozen=True, slots=True)
class TrainOnlyThroughputPilotSourceContractV1:
    data: Mapping[str, Any]
    repository_root: Path
    config_sha256: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _repository_file(root: Path, relative: str, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path must be a non-empty string")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or pure.as_posix() != relative
        or relative.startswith("./")
    ):
        raise ValueError(f"{label} path must be normalized and repository-relative")
    path = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} path must not traverse a symlink")
    if not path.is_file():
        raise ValueError(f"{label} must be one existing regular file")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escaped the repository") from error
    return path


def _file_binding(root: Path, relative: str) -> dict[str, Any]:
    payload = _repository_file(root, relative, label=relative).read_bytes()
    return {
        "byte_count": len(payload),
        "path": relative,
        "sha256": sha256_bytes(payload),
    }


def _load_frozen_json(
    root: Path,
    relative: str,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _file_binding(root, relative)
    if binding["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 drifted")
    payload = _repository_file(root, relative, label=label).read_bytes()
    return _strict_json_object(payload, label=label), binding


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one JSON object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be one JSON array")
    return value


def _freeze_roster(freeze: Mapping[str, Any]) -> list[dict[str, Any]]:
    if freeze.get("status") != EXPECTED_FREEZE_STATUS:
        raise ValueError("Freeze-B v2 status drifted")
    pilot = _mapping(freeze.get("throughput_pilot"), label="Freeze-B pilot")
    if (
        pilot.get("role") != "train_only"
        or pilot.get("state_count") != len(EXPECTED_STATE_IDS)
        or tuple(pilot.get("state_ids", ())) != EXPECTED_STATE_IDS
        or pilot.get("may_change_roster_or_training_grid") is not False
        or pilot.get("may_write_utility_or_kl") is not False
        or pilot.get("policy_throughput_requires_separate_post_freeze_contract")
        is not True
    ):
        raise ValueError("Freeze-B throughput pilot declaration drifted")
    queries = _sequence(freeze.get("query_states"), label="Freeze-B query states")
    by_state: dict[str, Mapping[str, Any]] = {}
    for raw in queries:
        query = _mapping(raw, label="Freeze-B query state")
        state_id = query.get("state_id")
        if not isinstance(state_id, str) or state_id in by_state:
            raise ValueError("Freeze-B query state identity is invalid or duplicated")
        by_state[state_id] = query

    roster: list[dict[str, Any]] = []
    observed_strata: list[dict[str, Any]] = []
    for stratum_index, stratum in enumerate(EXPECTED_STRATA):
        state_slice = EXPECTED_STATE_IDS[stratum_index * 4 : (stratum_index + 1) * 4]
        observed_strata.append({**stratum})
        for within_stratum_index, state_id in enumerate(state_slice):
            query = by_state.get(state_id)
            if query is None:
                raise ValueError("Freeze-B pilot state is missing from query states")
            decision_step = stratum["decision_step_id"]
            candidate_count = stratum["initial_candidate_count"]
            expected_candidate_ids = list(range(1, candidate_count + 1))
            source_id = state_id.split(":", 1)[0]
            if (
                query.get("state_id") != state_id
                or query.get("source_id") != source_id
                or query.get("trajectory_id") != source_id
                or query.get("role") != "train"
                or query.get("query_kind") != "stratum_anchor"
                or query.get("candidate_capacity_stratum")
                != stratum["candidate_capacity_stratum"]
                or query.get("decision_step_id") != decision_step
                or query.get("current_equivalent_event_step_id")
                != decision_step - 1
                or query.get("initial_candidate_count") != candidate_count
                or query.get("initial_candidate_event_step_ids")
                != expected_candidate_ids
                or query.get("maximum_labeled_cardinality") != 2
            ):
                raise ValueError(f"Freeze-B pilot query drifted: {state_id}")
            roster.append(
                {
                    "candidate_capacity_stratum": stratum[
                        "candidate_capacity_stratum"
                    ],
                    "decision_step_id": decision_step,
                    "initial_candidate_count": candidate_count,
                    "source_id": source_id,
                    "state_id": state_id,
                    "worker_index": within_stratum_index,
                }
            )
    if observed_strata != [dict(value) for value in EXPECTED_STRATA]:
        raise AssertionError("internal pilot stratum reconstruction drifted")
    return roster


def _processor_projection(
    result: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> dict[str, Any]:
    if result.get("status") != EXPECTED_PROCESSOR_RESULT_STATUS:
        raise ValueError("processor formal result status drifted")
    postflight = _mapping(result.get("postflight"), label="processor postflight")
    candidate = _mapping(
        postflight.get("candidate_artifact"), label="processor candidate schedule"
    )
    if (
        postflight.get("status") != EXPECTED_PROCESSOR_POSTFLIGHT_STATUS
        or candidate.get("filename") != EXPECTED_CANDIDATE_SCHEDULE_PATH
        or candidate.get("sha256") != EXPECTED_CANDIDATE_SCHEDULE_SHA256
        or candidate.get("byte_count") != EXPECTED_CANDIDATE_SCHEDULE_BYTES
        or candidate.get("candidate_inventory_sha256")
        != EXPECTED_CANDIDATE_INVENTORY_SHA256
        or candidate.get("state_count") != 2_400
        or candidate.get("subset_forward_count") != 93_914
        or candidate.get("total_model_operations") != 103_514
    ):
        raise ValueError("processor candidate schedule projection drifted")

    if publication.get("status") != EXPECTED_PUBLICATION_FINALIZATION_STATUS:
        raise ValueError("processor publication sibling status drifted")
    receipt = _mapping(
        publication.get("validated_publication_receipt"),
        label="validated processor publication receipt",
    )
    destination = _mapping(receipt.get("destination"), label="HF destination")
    source = _mapping(receipt.get("source"), label="publication source")
    fresh = _mapping(receipt.get("fresh_downloads"), label="fresh downloads")
    commit = _mapping(fresh.get("commit"), label="commit fresh download")
    tag = _mapping(fresh.get("tag"), label="tag fresh download")
    if (
        receipt.get("status") != EXPECTED_PUBLICATION_RECEIPT_STATUS
        or destination.get("repo") != EXPECTED_HF_REPO
        or destination.get("repo_type") != "dataset"
        or destination.get("immutable_revision") != EXPECTED_HF_REVISION
        or destination.get("tag") != EXPECTED_HF_TAG
        or destination.get("prefix") != EXPECTED_HF_PREFIX
        or destination.get("remote_file_count") != EXPECTED_REMOTE_FILE_COUNT
    ):
        raise ValueError("processor immutable publication destination drifted")

    inventory_raw = _sequence(
        source.get("formal_file_inventory"), label="formal file inventory"
    )
    inventory: list[dict[str, Any]] = []
    previous_path = ""
    for raw in inventory_raw:
        record = _mapping(raw, label="formal file inventory record")
        if set(record) != {"path", "sha256", "size_bytes"}:
            raise ValueError("formal file inventory record schema drifted")
        path = record.get("path")
        digest = record.get("sha256")
        size = record.get("size_bytes")
        if (
            not isinstance(path, str)
            or not path
            or path <= previous_path
            or not isinstance(digest, str)
            or len(digest) != 64
            or type(size) is not int
            or size <= 0
        ):
            raise ValueError("formal file inventory record is invalid")
        previous_path = path
        inventory.append(dict(record))
    if (
        len(inventory) != EXPECTED_FORMAL_FILE_COUNT
        or sum(record["size_bytes"] for record in inventory)
        != EXPECTED_FORMAL_TOTAL_BYTES
        or sha256_bytes(canonical_json_bytes(inventory))
        != EXPECTED_FORMAL_INVENTORY_SHA256
        or source.get("formal_file_count") != EXPECTED_FORMAL_FILE_COUNT
        or source.get("formal_total_byte_count") != EXPECTED_FORMAL_TOTAL_BYTES
        or source.get("formal_file_inventory_sha256")
        != EXPECTED_FORMAL_INVENTORY_SHA256
        or source.get("git_summary_sha256") != PROCESSOR_RESULT_SUMMARY_SHA256
    ):
        raise ValueError("processor formal publication inventory drifted")
    schedule_records = [
        record for record in inventory if record["path"] == EXPECTED_CANDIDATE_SCHEDULE_PATH
    ]
    if schedule_records != [
        {
            "path": EXPECTED_CANDIDATE_SCHEDULE_PATH,
            "sha256": EXPECTED_CANDIDATE_SCHEDULE_SHA256,
            "size_bytes": EXPECTED_CANDIDATE_SCHEDULE_BYTES,
        }
    ]:
        raise ValueError("published candidate schedule identity drifted")
    for replay, revision in ((commit, EXPECTED_HF_REVISION), (tag, EXPECTED_HF_TAG)):
        if (
            replay.get("revision") != revision
            or replay.get("remote_file_count") != EXPECTED_REMOTE_FILE_COUNT
            or replay.get("formal_file_count") != EXPECTED_FORMAL_FILE_COUNT
            or replay.get("formal_total_byte_count") != EXPECTED_FORMAL_TOTAL_BYTES
            or replay.get("formal_file_inventory_sha256")
            != EXPECTED_FORMAL_INVENTORY_SHA256
            or replay.get("byte_identical") is not True
            or replay.get("all_remote_files_byte_identical") is not True
        ):
            raise ValueError("processor publication fresh replay drifted")
    if (
        fresh.get("formal_files_verified_twice") != EXPECTED_FORMAL_FILE_COUNT
        or fresh.get("remote_files_verified_twice") != EXPECTED_REMOTE_FILE_COUNT
    ):
        raise ValueError("processor publication replay counts drifted")
    return {
        "candidate_schedule": {
            "byte_count": EXPECTED_CANDIDATE_SCHEDULE_BYTES,
            "candidate_inventory_sha256": EXPECTED_CANDIDATE_INVENTORY_SHA256,
            "path": EXPECTED_CANDIDATE_SCHEDULE_PATH,
            "sha256": EXPECTED_CANDIDATE_SCHEDULE_SHA256,
        },
        "formal_file_count": EXPECTED_FORMAL_FILE_COUNT,
        "formal_file_inventory": inventory,
        "formal_file_inventory_sha256": EXPECTED_FORMAL_INVENTORY_SHA256,
        "formal_total_byte_count": EXPECTED_FORMAL_TOTAL_BYTES,
        "hf_prefix": EXPECTED_HF_PREFIX,
        "hf_repo": EXPECTED_HF_REPO,
        "hf_revision": EXPECTED_HF_REVISION,
        "hf_tag": EXPECTED_HF_TAG,
        "publication_status": EXPECTED_PUBLICATION_FINALIZATION_STATUS,
        "remote_file_count": EXPECTED_REMOTE_FILE_COUNT,
    }


def _model_projection(model: Mapping[str, Any]) -> dict[str, Any]:
    files = _sequence(model.get("files"), label="model snapshot files")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in files:
        record = _mapping(raw, label="model snapshot file")
        if set(record) != {"path", "sha256", "size"}:
            raise ValueError("model snapshot file schema drifted")
        path = record.get("path")
        if not isinstance(path, str) or not path or path in seen:
            raise ValueError("model snapshot file path is invalid or duplicated")
        seen.add(path)
        normalized.append(dict(record))
    if (
        model.get("repo") != EXPECTED_MODEL_REPO
        or model.get("revision") != EXPECTED_MODEL_REVISION
        or len(normalized) != EXPECTED_MODEL_FILE_COUNT
        or sum(record["size"] for record in normalized) != EXPECTED_MODEL_TOTAL_BYTES
    ):
        raise ValueError("model snapshot manifest drifted")
    return {
        "file_count": EXPECTED_MODEL_FILE_COUNT,
        "repo": EXPECTED_MODEL_REPO,
        "revision": EXPECTED_MODEL_REVISION,
        "total_byte_count": EXPECTED_MODEL_TOTAL_BYTES,
    }


def _repair_lineage_projection(
    failure: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    counts = _mapping(failure.get("counts"), label="parent failure counts")
    failure_record = _mapping(
        failure.get("failure"), label="parent failure classification"
    )
    bindings = _mapping(
        failure.get("bindings"), label="parent failure source bindings"
    )
    repair = _mapping(failure.get("repair"), label="parent failure repair state")
    artifacts = _mapping(
        failure.get("artifacts"), label="parent failure artifacts"
    )
    if (
        failure.get("status") != EXPECTED_PARENT_FAILURE_STATUS
        or failure.get("result_id")
        != "set_utility_train_only_throughput_pilot_v1"
        or failure.get("scientific_eligibility") is not False
        or counts.get("actual_generation_calls") != 0
        or counts.get("actual_native_calls") != 0
        or counts.get("actual_teacher_calls") != 0
        or counts.get("actual_teacher_examples") != 0
        or counts.get("pair_completed_count") != 0
        or counts.get("retry_count") != 0
        or failure_record.get("stage")
        != "semantic_input_load_before_runtime_factory"
        or failure_record.get("reason_code")
        != "CANDIDATE_SCHEDULE_NOT_CANONICAL_PRETTY_JSON"
        or failure_record.get("model_runtime_factory_entered") is not False
        or repair.get("v1_retry_allowed") is not False
        or bindings.get("source_commit_a") != PARENT_SOURCE_GIT_REVISION
        or bindings.get("envelope_commit_b") != PARENT_ENVELOPE_GIT_REVISION
        or bindings.get("execution_envelope_sha256")
        != PARENT_EXECUTION_ENVELOPE_SHA256
        or artifacts.get("candidate_schedule_sha256")
        != EXPECTED_CANDIDATE_SCHEDULE_SHA256
    ):
        raise ValueError("parent throughput-pilot failure lineage drifted")
    return {
        "candidate_schedule_reconstruction": {
            "artifact_mutation_allowed": False,
            "integer_key_path": list(CANDIDATE_SCHEDULE_INTEGER_KEY_PATH),
            "raw_bytes_must_equal_producer_reconstruction": True,
            "raw_sha256": EXPECTED_CANDIDATE_SCHEDULE_SHA256,
            "repair_id": CANDIDATE_SCHEDULE_KEY_REPAIR_ID,
            "strict_json_unique_keys_and_finite_values_required": True,
            "string_key_contract": "canonical_positive_base10_without_leading_zero",
        },
        "new_execution_envelope_required": True,
        "parent_attempt": {
            "actual_native_calls": counts["actual_native_calls"],
            "envelope_git_revision": bindings["envelope_commit_b"],
            "execution_envelope_sha256": bindings["execution_envelope_sha256"],
            "failure_reason_code": failure_record["reason_code"],
            "failure_stage": failure_record["stage"],
            "source_git_revision": bindings["source_commit_a"],
            "summary": dict(binding),
            "summary_status": failure["status"],
        },
        "parent_run_retry_allowed": False,
        "scientific_contract_changes": [],
    }


def _module_path(root: Path, module: str) -> str | None:
    if not module.startswith("causalcache"):
        return None
    stem = "code/" + module.replace(".", "/")
    for relative in (stem + ".py", stem + "/__init__.py"):
        path = root.joinpath(*PurePosixPath(relative).parts)
        if path.is_file() and not path.is_symlink():
            return relative
    return None


def _discover_runtime_transitive_paths(root: Path) -> tuple[str, ...]:
    pending = list(RUNTIME_ENTRYPOINT_PATHS)
    paths: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in paths:
            continue
        _repository_file(root, relative, label="runtime source")
        paths.add(relative)
        tree = ast.parse(
            _repository_file(root, relative, label="runtime source").read_bytes(),
            filename=relative,
        )
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError("runtime source must use absolute project imports")
                if node.module is not None:
                    imported.append(node.module)
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            for dependency in imported:
                if dependency.startswith("causalcache"):
                    dependency_path = _module_path(root, dependency)
                    if dependency_path is None:
                        raise ValueError(
                            f"runtime project import does not resolve to one file: {dependency}"
                        )
                    pending.append(dependency_path)
    paths.update(
        {
            "code/causalcache/__init__.py",
            "code/causalcache/data/__init__.py",
            "code/causalcache/policy/__init__.py",
        }
    )
    observed = tuple(sorted(paths))
    if observed != EXPECTED_RUNTIME_TRANSITIVE_SOURCE_PATHS:
        raise ValueError("runtime transitive source path closure drifted")
    return observed


def _source_projection(root: Path) -> dict[str, Any]:
    paths = tuple(
        sorted(
            {
                CONTRACT_PATH,
                CLI_PATH,
                *_discover_runtime_transitive_paths(root),
            }
        )
    )
    inventory = [_file_binding(root, path) for path in paths]
    return {
        "entrypoint_paths": list(RUNTIME_ENTRYPOINT_PATHS),
        "file_count": len(inventory),
        "inventory": inventory,
        "inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
    }


def build_train_only_throughput_pilot_v1_config_skeleton(
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Rebuild the complete source-only contract from live repository bytes."""
    root = Path(repository_root).resolve()
    freeze, freeze_binding = _load_frozen_json(
        root,
        FREEZE_B_V2_MANIFEST_PATH,
        FREEZE_B_V2_MANIFEST_SHA256,
        label="Freeze-B v2 manifest",
    )
    result, result_binding = _load_frozen_json(
        root,
        PROCESSOR_RESULT_SUMMARY_PATH,
        PROCESSOR_RESULT_SUMMARY_SHA256,
        label="processor formal result summary",
    )
    publication, publication_binding = _load_frozen_json(
        root,
        PROCESSOR_PUBLICATION_SUMMARY_PATH,
        PROCESSOR_PUBLICATION_SUMMARY_SHA256,
        label="processor publication sibling summary",
    )
    model, model_binding = _load_frozen_json(
        root,
        MODEL_SNAPSHOT_MANIFEST_PATH,
        MODEL_SNAPSHOT_MANIFEST_SHA256,
        label="GUI-Owl snapshot manifest",
    )
    failure, failure_binding = _load_frozen_json(
        root,
        PARENT_FAILURE_SUMMARY_PATH,
        PARENT_FAILURE_SUMMARY_SHA256,
        label="parent throughput-pilot failure summary",
    )
    roster = _freeze_roster(freeze)
    processor = _processor_projection(result, publication)
    repair_lineage = _repair_lineage_projection(failure, failure_binding)
    worker_mapping = [
        {
            "state_ids": [
                record["state_id"]
                for record in roster
                if record["worker_index"] == worker_index
            ],
            "worker_index": worker_index,
        }
        for worker_index in range(WORKER_COUNT)
    ]
    if any(len(worker["state_ids"]) != 3 for worker in worker_mapping):
        raise AssertionError("pilot worker mapping is not balanced")
    operations_by_microbatch = [
        {
            "generation_calls_per_state": 2,
            "microbatch_size": microbatch,
            "native_calls_per_state": 2 + (2 // microbatch),
            "teacher_calls_per_state": 2 // microbatch,
            "teacher_examples_per_state": 2,
        }
        for microbatch in ALLOWED_MICROBATCH_ORDER
    ]
    native_ceiling = len(roster) * sum(
        record["native_calls_per_state"] for record in operations_by_microbatch
    )
    if native_ceiling != NATIVE_CALL_CEILING:
        raise AssertionError("pilot native-call ceiling reconstruction drifted")
    expected_success_counts = {
        "cross_variant_comparison_count": len(roster),
        "cross_variant_equal_count": len(roster),
        "native_call_count": native_ceiling,
        "native_completed_call_count": native_ceiling,
        "pair_attempt_count": len(roster),
        "pair_completed_count": len(roster),
        "reference_generation_call_count": 4 * len(roster),
        "reference_generation_completed_count": 4 * len(roster),
        "reference_input_build_call_count": 2 * len(roster),
        "reference_input_build_completed_count": 2 * len(roster),
        "reference_plan_build_count": 2 * len(roster),
        "reference_teacher_forward_call_count": 3 * len(roster),
        "reference_teacher_forward_completed_call_count": 3 * len(roster),
        "reference_teacher_forward_completed_example_count": 4 * len(roster),
        "reference_teacher_forward_example_count": 4 * len(roster),
        "variant_attempt_count": 2 * len(roster),
        "variant_completed_count": 2 * len(roster),
    }
    return {
        "authorization": {
            "access_sealed_test": False,
            "generate_restoration_labels": False,
            "load_policy_or_vision_model": False,
            "mutate_hugging_face": False,
            "read_remote_processor_artifact": False,
            "read_repository_bound_metadata": True,
            "run_closed_loop": False,
            "run_gpu_or_cuda": False,
            "train_predictor": False,
            "validate_source_contract": True,
            "write_pilot_result": False,
        },
        "deployment_decision_rule": {
            "cross_host_or_heterogeneous_metric_merge_allowed": False,
            "device_reserved_memory_fraction_maximum": 0.80,
            "execution_requires_separate_exact_envelope": True,
            "mb1_any_failure_or_memory_violation_outcome": "NO_GO",
            "mb2_failure_fallback_to_mb1": {
                "all_failures_must_be_mb2_teacher_stage": True,
                "cross_variant_action_equality_must_complete_for_every_state": True,
                "otherwise_outcome": "NO_GO",
            },
            "mb2_teacher_wall_ratio_maximum_for_selection": 0.95,
            "memory_headroom_fraction_minimum": 0.20,
            "memory_metric": "full_call_cuda_reserved_max",
            "memory_requirement_scope": (
                "every_state_variant_full_call_peak_must_satisfy_the_limit"
            ),
            "no_retry": True,
            "no_top_up": True,
            "selection": {
                "both_variants_all_12_success_and_cross_actions_equal": (
                    "select_mb2_only_if_teacher_wall_ratio_at_most_0.95_"
                    "otherwise_select_mb1"
                ),
                "comparison_metric": (
                    "sum_reference_teacher_forward_end_to_end_wall_total_"
                    "over_exact_12_states"
                ),
            },
        },
        "inputs": {
            "freeze_b_v2_manifest": freeze_binding,
            "model_snapshot_manifest": {
                **model_binding,
                **_model_projection(model),
            },
            "processor_formal_result_summary": result_binding,
            "processor_publication": {
                **publication_binding,
                **processor,
            },
        },
        "metric_only_result_contract": {
            "action_handle_lifetime": "in_process_only_not_serialized",
            "per_variant_core_allowed_top_level_fields": [
                "counts",
                "failure_class",
                "latency_seconds",
                "peak_memory_bytes",
            ],
            "failure_projection": "safe_error_class_identifier_and_call_metrics_only",
            "forbidden_serialized_fields": [
                "action",
                "action_text",
                "decoded_output",
                "kl",
                "logits",
                "messages",
                "native_output",
                "tokens",
                "utility",
            ],
            "measurement_boundary": (
                "encode_h2d_preparation_forward_decode_or_logit_disposal"
            ),
            "pair_metric_only_required": True,
            "retry_count": 0,
        },
        "negative_operations": {
            "closed_loop_episode_count": 0,
            "evaluation_state_access_count": 0,
            "gpu_or_cuda_call_count": 0,
            "hugging_face_mutation_count": 0,
            "matched_nll_count": 0,
            "policy_generation_count": 0,
            "predictor_training_count": 0,
            "processor_call_count": 0,
            "remote_artifact_read_count": 0,
            "restoration_label_count": 0,
            "sealed_test_access_count": 0,
            "teacher_forward_count": 0,
            "tune_state_access_count": 0,
            "utility_or_kl_write_count": 0,
        },
        "pilot": {
            "aggregation_order": list(range(WORKER_COUNT)),
            "allowed_microbatch_order": list(ALLOWED_MICROBATCH_ORDER),
            "expected_success_counts": expected_success_counts,
            "native_call_ceiling": native_ceiling,
            "operations_by_microbatch": operations_by_microbatch,
            "role": "train_only",
            "roster": roster,
            "state_count": len(roster),
            "state_ids_sha256": sha256_bytes(
                canonical_json_bytes(list(EXPECTED_STATE_IDS))
            ),
            "strata": [dict(value) for value in EXPECTED_STRATA],
            "worker_assignment_rule": (
                "within_stratum_state_order_index_modulo_four"
            ),
            "worker_count": WORKER_COUNT,
            "worker_mapping": worker_mapping,
        },
        "protocol_id": PROTOCOL_ID,
        "repair_lineage": repair_lineage,
        "schema_version": SCHEMA_VERSION,
        "source": _source_projection(root),
        "status": STATUS,
    }


def validate_train_only_throughput_pilot_v1_config(
    config: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("throughput pilot source config must be one mapping")
    expected = build_train_only_throughput_pilot_v1_config_skeleton(
        repository_root=repository_root
    )
    if dict(config) != expected:
        raise ValueError("throughput pilot source config differs from the live skeleton")
    source = expected["source"]
    processor = expected["inputs"]["processor_publication"]
    return {
        "candidate_schedule_sha256": processor["candidate_schedule"]["sha256"],
        "config_sha256": sha256_bytes(canonical_pretty_json_bytes(expected)),
        "hf_revision": processor["hf_revision"],
        "native_call_ceiling": expected["pilot"]["native_call_ceiling"],
        "source_file_count": source["file_count"],
        "source_inventory_sha256": source["inventory_sha256"],
        "state_count": expected["pilot"]["state_count"],
        "status": VALIDATION_STATUS,
        "worker_count": expected["pilot"]["worker_count"],
    }


def load_train_only_throughput_pilot_v1_contract(
    *,
    repository_root: str | Path,
    config_path: str | Path = CANONICAL_CONFIG_PATH,
) -> TrainOnlyThroughputPilotSourceContractV1:
    root = Path(repository_root).resolve()
    supplied = Path(config_path)
    path = supplied if supplied.is_absolute() else root / supplied
    expected_path = (root / CANONICAL_CONFIG_PATH).resolve()
    if path.resolve() != expected_path:
        raise ValueError(f"throughput pilot config must be canonical: {expected_path}")
    payload = _repository_file(
        root, CANONICAL_CONFIG_PATH, label="throughput pilot source config"
    ).read_bytes()
    config = _strict_json_object(payload, label="throughput pilot source config")
    if payload != canonical_pretty_json_bytes(config):
        raise ValueError("throughput pilot source config must be canonical pretty JSON")
    validate_train_only_throughput_pilot_v1_config(
        config,
        repository_root=root,
    )
    return TrainOnlyThroughputPilotSourceContractV1(
        data=config,
        repository_root=root,
        config_sha256=sha256_bytes(payload),
    )


__all__ = [
    "ALLOWED_MICROBATCH_ORDER",
    "CANDIDATE_SCHEDULE_INTEGER_KEY_PATH",
    "CANDIDATE_SCHEDULE_KEY_REPAIR_ID",
    "CANONICAL_CONFIG_PATH",
    "CLI_PATH",
    "CONTRACT_PATH",
    "EXPECTED_CANDIDATE_SCHEDULE_SHA256",
    "EXPECTED_HF_REVISION",
    "EXPECTED_RUNTIME_TRANSITIVE_SOURCE_PATHS",
    "EXPECTED_STATE_IDS",
    "MODEL_SNAPSHOT_MANIFEST_PATH",
    "MODEL_SNAPSHOT_MANIFEST_SHA256",
    "NATIVE_CALL_CEILING",
    "PARENT_ENVELOPE_GIT_REVISION",
    "PARENT_EXECUTION_ENVELOPE_SHA256",
    "PARENT_FAILURE_SUMMARY_PATH",
    "PARENT_FAILURE_SUMMARY_SHA256",
    "PARENT_SOURCE_GIT_REVISION",
    "PROTOCOL_ID",
    "RUNTIME_ENTRYPOINT_PATHS",
    "SCHEMA_VERSION",
    "STATUS",
    "TrainOnlyThroughputPilotSourceContractV1",
    "VALIDATION_STATUS",
    "WORKER_COUNT",
    "build_train_only_throughput_pilot_v1_config_skeleton",
    "canonical_json_bytes",
    "canonical_pretty_json_bytes",
    "load_train_only_throughput_pilot_v1_contract",
    "sha256_bytes",
    "validate_train_only_throughput_pilot_v1_config",
]
