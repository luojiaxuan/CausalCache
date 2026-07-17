"""Fail-closed source-only contract for formal-58 gate cache construction."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.gate_v1_contract import (
    EXPECTED_ASSIGNMENT_DIGEST,
    EXPECTED_FOLD_DIGESTS,
    EXPECTED_FOLD_SIZES,
    canonical_json_bytes,
    derive_oof_folds,
    derive_rosters,
    load_strict_json_object as load_gate_json_object,
)


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_gate_v1_formal_cache_v1"
SOURCE_STATUS = "source_only_frozen_before_formal_cache_execution"
VALIDATION_STATUS = "VALID_SOURCE_ONLY_GATE_V1_FORMAL_CACHE_CONTRACT_V1"
FEATURE_MANIFEST_STATUS = "VALID_GATE_V1_FORMAL58_FEATURE_CACHE_V1"
LABEL_MANIFEST_STATUS = "VALID_GATE_V1_FORMAL58_LABEL_CACHE_V1"
FINAL_COMPLETION_STATUS = "COMPLETED_GATE_V1_FORMAL58_CACHE_PUBLICATION_V1"
CANONICAL_CONFIG_PATH = "code/configs/causalcache_gate_v1_formal_cache_v1.json"
FROZEN_CONFIG_SHA256 = (
    "1d7527e8a7bede8aaab8a21f7757f786674530238ae99fe3ce5b196cbce67261"
)
FORMAL_SOURCE_IDS_SHA256 = (
    "ccec55afb0882e602f5d2c83ec420682488e2663fe26a7c574a851825c43f225"
)
RUNNER_FREEZE_B_PATH = (
    "code/configs/causalcache_gate_v1_formal_cache_runner_v1.json"
)
EXPECTED_SECTION_SHA256 = {
    "parent_gate_contract": "48132ed7035f59e2f7fa9784b13455c86d51fcaf684eeb05c1940c7a5fedafaf",
    "source_freeze": "af19e4813d5b01491b731ac11fd63a1841b2f1a0053d3c33736c45ba4ecb8240",
    "input_artifacts": "2c32b94e20b6b3c9fb80823a1ac6161f1e40358d85cccddb1ce271571ffa9b83",
    "publication_completion_binding": "0778b510b9131b81736d6b6fff9c2d8d5f363a551e3f956db26d5ec2c852c841",
    "formal_geometry": "5da623863743d3daf7db3a61ad707a685cf13d6435bb5ffbb937bebe9c2b1c66",
    "access_firewall": "c3e157a3079e55f66fa919daabf4cfeb41d5a866ae63728ced30ed6e2bab0841",
    "cache_formats": "b090524462ad6cf501f0dff2411f580b7aa53f4930c8fc001ad6b368b3ef422c",
    "local_first_state_machine": "c866842509c1642fd82b95caedf56ddb493c22d6a2240e3faaffe72c61d4e2f1",
    "destination": "e7209923c415630bb7328590d38757975972d9964b48545083f9a4b850cb5007",
    "runtime_contract": "af0362daab6ccbfd86fbd812f0eec53241708c0620b89cbe7ddc2b7cfe5d940b",
    "source_only_operation_contract": "29827efa954416710ae51a7671baaff5bccc4533944c4df94c5f706e0d0354b6",
    "execution_expected_operation_contract": "6f40aa1a2faf1dd41f064903d3f1c985a307113c17c81520fc46f227c0d9fadd",
    "authorization": "d36508bcddc47b6d86dace9de8f90be39b9b8fcdb8bd7bef82e1a44d6f6dbd36",
}
TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    *EXPECTED_SECTION_SHA256,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class FormalCacheContract:
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path

    @property
    def parent_gate(self) -> Mapping[str, Any]:
        return _mapping(self.data["parent_gate_contract"], "parent gate")

    @property
    def inputs(self) -> Mapping[str, Any]:
        return _mapping(self.data["input_artifacts"], "input artifacts")

    @property
    def firewall(self) -> Mapping[str, Any]:
        return _mapping(self.data["access_firewall"], "access firewall")

    @property
    def formats(self) -> Mapping[str, Any]:
        return _mapping(self.data["cache_formats"], "cache formats")

    @property
    def local_state(self) -> Mapping[str, Any]:
        return _mapping(self.data["local_first_state_machine"], "local state")

    @property
    def destination(self) -> Mapping[str, Any]:
        return _mapping(self.data["destination"], "destination")

    @property
    def runtime(self) -> Mapping[str, Any]:
        return _mapping(self.data["runtime_contract"], "runtime")

    @property
    def operations(self) -> Mapping[str, Any]:
        return _mapping(
            self.data["source_only_operation_contract"], "source-only operations"
        )

    @property
    def execution_operations(self) -> Mapping[str, Any]:
        return _mapping(
            self.data["execution_expected_operation_contract"],
            "execution operations",
        )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
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
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a sequence")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys drifted: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} drifted: expected {expected!r}, got {actual!r}")


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} is not a canonical relative POSIX path")
    return value


def _regular_file_bytes(path: Path, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path.read_bytes()


def _validate_frozen_sections(config: Mapping[str, Any]) -> None:
    for name, expected in EXPECTED_SECTION_SHA256.items():
        section = _mapping(config.get(name), name)
        _equal(
            sha256_bytes(canonical_json_bytes(section)),
            expected,
            f"{name} canonical SHA256",
        )


def _validate_source_remote_contract(config: Mapping[str, Any]) -> None:
    source = _mapping(config["source_freeze"], "source freeze")
    source_a = _mapping(
        source["source_a_remote_validation"], "Source-A remote validation"
    )
    _exact_keys(
        source_a,
        {"network_call_count", "remote_tracking_ref", "required_equalities"},
        "Source-A remote validation",
    )
    _equal(source_a["network_call_count"], 0, "Source-A remote network count")
    _equal(
        source_a["remote_tracking_ref"],
        "refs/remotes/origin/main",
        "Source-A tracking ref",
    )
    _equal(
        tuple(source_a["required_equalities"]),
        ("HEAD", "origin/main"),
        "Source-A source equality",
    )

    execution_b = _mapping(
        source["execution_b_live_remote_validation"],
        "Execution-B live remote validation",
    )
    _exact_keys(
        execution_b,
        {
            "required",
            "network_call_count",
            "argv",
            "expected_stdout_record_count",
            "expected_commit_pattern",
            "expected_ref",
            "required_equalities",
        },
        "Execution-B live remote validation",
    )
    _equal(execution_b["required"], True, "Execution-B live remote requirement")
    _equal(execution_b["network_call_count"], 1, "Execution-B remote network count")
    _equal(
        tuple(execution_b["argv"]),
        ("git", "ls-remote", "--exit-code", "origin", "refs/heads/main"),
        "Execution-B live remote argv",
    )
    _equal(
        execution_b["expected_stdout_record_count"],
        1,
        "Execution-B live remote record count",
    )
    _equal(
        execution_b["expected_commit_pattern"],
        "[0-9a-f]{40}",
        "Execution-B live commit pattern",
    )
    _equal(
        execution_b["expected_ref"],
        "refs/heads/main",
        "Execution-B live ref",
    )
    _equal(
        tuple(execution_b["required_equalities"]),
        ("HEAD", "origin/main", "live origin refs/heads/main"),
        "Execution-B source equality",
    )


def _validate_git_prerequisites(
    config: Mapping[str, Any], root: Path
) -> tuple[Mapping[str, Any], ...]:
    source = _mapping(config["source_freeze"], "source freeze")
    records = _sequence(source["git_prerequisites"], "Git prerequisites")
    normalized: list[Mapping[str, Any]] = []
    observed_paths: set[str] = set()
    for raw in records:
        record = _mapping(raw, "Git prerequisite")
        _exact_keys(record, {"path", "sha256", "size_bytes"}, "Git prerequisite")
        relative = _safe_relative_path(record["path"], "Git prerequisite path")
        if relative in observed_paths:
            raise ValueError("Git prerequisite paths must be unique")
        observed_paths.add(relative)
        payload = _regular_file_bytes(root / relative, f"Git prerequisite {relative}")
        _equal(len(payload), record["size_bytes"], f"{relative} size")
        _equal(sha256_bytes(payload), record["sha256"], f"{relative} SHA256")
        normalized.append(dict(record))
    return tuple(normalized)


def _validate_parent_and_publication(config: Mapping[str, Any], root: Path) -> None:
    parent = _mapping(config["parent_gate_contract"], "parent gate contract")
    parent_payload = _regular_file_bytes(root / parent["path"], "parent gate config")
    _equal(sha256_bytes(parent_payload), parent["sha256"], "parent gate SHA256")
    parent_json = load_strict_json_object(root / parent["path"])
    _equal(parent_json.get("protocol_id"), parent["protocol_id"], "parent protocol")
    _equal(
        parent_json.get("preregistration_status"),
        parent["status"],
        "parent preregistration status",
    )

    binding = _mapping(
        config["publication_completion_binding"], "publication completion binding"
    )
    result = _mapping(binding["result_summary"], "publication result summary")
    result_path = root / result["path"]
    payload = _regular_file_bytes(result_path, "publication result summary")
    _equal(len(payload), result["size_bytes"], "publication result size")
    _equal(sha256_bytes(payload), result["sha256"], "publication result SHA256")
    summary = load_strict_json_object(result_path)
    _equal(summary.get("protocol_id"), result["protocol_id"], "publication protocol")
    _equal(summary.get("status"), result["status"], "publication status")
    formal = _mapping(summary.get("formal_consumption"), "formal consumption")
    for key, expected in (
        ("formal_label_loader_eligible", True),
        ("gate_training_unlocked", True),
        ("gate_trained", False),
        ("matched_nll_unlocked", False),
        ("closed_loop_unlocked", False),
    ):
        _equal(formal.get(key), expected, f"publication {key}")
    hf = _mapping(summary.get("hf"), "publication HF identity")
    repaired = _mapping(
        _mapping(config["input_artifacts"], "inputs")[
            "repaired_expansion_restoration_labels"
        ],
        "repaired expansion labels",
    )
    _equal(hf.get("repo"), repaired["repo"], "repaired publication repo")
    _equal(
        hf.get("immutable_pair_commit"),
        repaired["immutable_revision"],
        "repaired publication revision",
    )
    _equal(hf.get("tag"), repaired["tag"], "repaired publication tag")
    _equal(
        hf.get("annotated_tag_object"),
        repaired["annotated_tag_object"],
        "repaired publication annotated tag",
    )


def _validate_inputs(config: Mapping[str, Any]) -> None:
    inputs = _mapping(config["input_artifacts"], "input artifacts")
    _exact_keys(
        inputs,
        {
            "legacy_derived_features",
            "expansion_derived_features",
            "legacy_restoration_labels",
            "repaired_expansion_restoration_labels",
        },
        "input artifacts",
    )
    expected_purpose = {
        "legacy_derived_features": "feature_only",
        "expansion_derived_features": "feature_only",
        "legacy_restoration_labels": "label_only",
        "repaired_expansion_restoration_labels": "label_only",
    }
    revisions: set[tuple[str, str]] = set()
    file_count = 0
    for name, purpose in expected_purpose.items():
        artifact = _mapping(inputs[name], f"input {name}")
        _equal(artifact["purpose"], purpose, f"{name} purpose")
        _equal(artifact["repo_type"], "dataset", f"{name} repo type")
        _equal(artifact["private"], True, f"{name} privacy")
        _equal(
            artifact["transport_contains_development"],
            True,
            f"{name} mixed transport declaration",
        )
        revision = artifact["immutable_revision"]
        if not isinstance(revision, str) or _GIT_COMMIT.fullmatch(revision) is None:
            raise ValueError(f"{name} immutable revision is malformed")
        revisions.add((artifact["repo"], revision))
        files = _sequence(artifact["files"], f"{name} files")
        if not files:
            raise ValueError(f"{name} must bind at least one exact file")
        paths: set[str] = set()
        for raw in files:
            file_record = _mapping(raw, f"{name} file")
            path = _safe_relative_path(file_record["path"], f"{name} file path")
            if path in paths:
                raise ValueError(f"{name} contains duplicate file paths")
            paths.add(path)
            digest = file_record["sha256"]
            size = file_record["size_bytes"]
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError(f"{name} file SHA256 is malformed")
            if type(size) is not int or size <= 0:
                raise ValueError(f"{name} file size must be positive")
            file_count += 1
    _equal(len(revisions), 4, "immutable logical input revision count")
    _equal(file_count, 9, "exact HF input file count")


def _validate_geometry(config: Mapping[str, Any], root: Path) -> Mapping[str, Any]:
    selection = load_gate_json_object(root / "data/manifests/restoration_v2_selection.json")
    expansion = load_gate_json_object(
        root / "code/configs/causalcache_restoration_v2_2_label_expansion_v1.json"
    )
    rosters = derive_rosters(selection, expansion)
    formal = rosters["formal_train"]
    geometry = _mapping(config["formal_geometry"], "formal geometry")
    trajectory_count = len(formal)
    candidate_counts = tuple(geometry["candidate_event_counts"])
    computed = {
        "trajectory_count": trajectory_count,
        "state_count": trajectory_count * len(candidate_counts),
        "candidate_feature_count": trajectory_count * sum(candidate_counts),
        "distance_value_count": trajectory_count
        * sum(2**count for count in candidate_counts),
        "conditional_edge_count": trajectory_count
        * sum(count + count * (count - 1) for count in candidate_counts),
    }
    for key, value in computed.items():
        _equal(geometry[key], value, f"formal {key}")
    roster_digest = sha256_bytes(canonical_json_bytes(list(formal)))
    _equal(roster_digest, FORMAL_SOURCE_IDS_SHA256, "derived formal roster digest")
    _equal(geometry["source_ids_sha256"], roster_digest, "config formal roster digest")
    if set(formal) & (
        set(rosters["combined_development"]) | set(rosters["sealed_confirm"])
    ):
        raise ValueError("formal roster overlaps development or confirm")

    assignments, folds = derive_oof_folds(formal)
    fold_digests = tuple(
        sha256_bytes(canonical_json_bytes(list(fold))) for fold in folds
    )
    assignment_payload = [
        {"source_id": source_id, "fold": fold} for source_id, fold in assignments
    ]
    assignment_digest = sha256_bytes(canonical_json_bytes(assignment_payload))
    _equal(tuple(len(fold) for fold in folds), EXPECTED_FOLD_SIZES, "fold sizes")
    _equal(fold_digests, EXPECTED_FOLD_DIGESTS, "fold digests")
    _equal(assignment_digest, EXPECTED_ASSIGNMENT_DIGEST, "assignment digest")
    _equal(tuple(geometry["oof_fold_sizes"]), EXPECTED_FOLD_SIZES, "config folds")
    _equal(
        tuple(geometry["oof_fold_source_ids_sha256"]),
        EXPECTED_FOLD_DIGESTS,
        "config fold digests",
    )
    _equal(
        geometry["oof_assignment_table_sha256"],
        EXPECTED_ASSIGNMENT_DIGEST,
        "config assignment digest",
    )
    return {**computed, "source_ids_sha256": roster_digest}


def _validate_firewall_formats_and_state(config: Mapping[str, Any]) -> None:
    firewall = _mapping(config["access_firewall"], "access firewall")
    for key in (
        "transport_byte_possession_is_not_semantic_decode",
        "mixed_train_development_transport_is_prevalidated_immutable",
        "label_access_requires_completed_feature_cache",
        "label_access_requires_durable_label_access_claim",
        "join_allowed_only_after_separate_cache_validation",
    ):
        _equal(firewall[key], True, key)
    for key in (
        "feature_phase_label_semantic_decode_count",
        "development_semantic_decode_count",
        "confirm_semantic_decode_count",
    ):
        _equal(firewall[key], 0, key)
    _equal(firewall["generic_full_artifact_readers_allowed"], False, "full readers")
    _equal(
        firewall["combined_feature_label_cache_persistence_allowed"],
        False,
        "combined cache persistence",
    )

    formats = _mapping(config["cache_formats"], "cache formats")
    common = _mapping(formats["common_ustar"], "common USTAR")
    _equal(common["format"], "ustar", "archive format")
    _equal(
        common["float_encoding"],
        "ieee754_binary64_big_endian_lowercase_hex",
        "float encoding",
    )
    _equal(common["float_hex_pattern"], "[0-9a-f]{16}", "float hex pattern")
    _equal(common["nonfinite_allowed"], False, "non-finite values")
    feature = _mapping(formats["feature"], "feature format")
    label = _mapping(formats["label"], "label format")
    _equal(feature["manifest_status"], FEATURE_MANIFEST_STATUS, "feature status")
    _equal(label["manifest_status"], LABEL_MANIFEST_STATUS, "label status")
    _equal(feature["row_count"], 174, "feature row count")
    _equal(label["row_count"], 174, "label row count")
    _equal(label["distance_value_count"], 1624, "label distance count")
    if feature["archive_path"] == label["archive_path"]:
        raise ValueError("feature and label cache paths must differ")
    if set(feature["state_exact_keys"]) & {"distance", "distance_rows"}:
        raise ValueError("feature schema contains a label field")
    if set(label["state_exact_keys"]) & {"q64_f64_hex", "candidates"}:
        raise ValueError("label schema contains a feature field")
    separation = _mapping(formats["physical_separation"], "physical separation")
    if any(value is not False for value in separation.values()):
        raise ValueError("all combined or shared cache forms must be forbidden")

    machine = _mapping(config["local_first_state_machine"], "state machine")
    states = _sequence(machine["ordered_states"], "ordered states")
    expected_names = (
        "global_claim",
        "feature_completion",
        "label_access_claim",
        "label_completion",
        "remote_base_receipt",
        "completion_staging",
        "final_completion",
    )
    _equal(tuple(state["name"] for state in states), expected_names, "state order")
    paths = tuple(state["path"] for state in states)
    if len(paths) != len(set(paths)):
        raise ValueError("local state paths must be unique")
    _equal(
        tuple(state["formal_label_semantic_access_allowed_after"] for state in states),
        (False, False, True, True, True, True, True),
        "label-access transition",
    )
    _equal(states[-1]["status"], FINAL_COMPLETION_STATUS, "final status")
    _equal(machine["local_first_before_any_hf_output_mutation"], True, "local first")
    _equal(machine["final_completion_is_hard_link_to_staging"], True, "hard link")
    _equal(
        machine["final_completion_is_last_first_execution_mutation"],
        True,
        "completion-last",
    )


def _validate_destination_runtime_and_authority(config: Mapping[str, Any]) -> None:
    destination = _mapping(config["destination"], "destination")
    _equal(
        destination["repo"],
        "gavinlaw/causalcache-gate-v1-formal58-cache-mobile",
        "destination repo",
    )
    _equal(destination["private"], True, "destination privacy")
    _equal(destination["tag"], "gate-v1-formal58-cache-v1", "destination tag")
    targets = tuple(destination["exact_three_targets"])
    _equal(len(targets), 3, "remote target count")
    if len(set(targets)) != 3:
        raise ValueError("remote targets must be distinct")
    for index, target in enumerate(targets):
        _safe_relative_path(target, f"remote target {index}")
    _equal(destination["completed_replay_remote_mutation_count"], 0, "replay mutation")

    runtime = _mapping(config["runtime_contract"], "runtime contract")
    _equal(runtime["device"], "cpu", "runtime device")
    _equal(runtime["gpu_required"], False, "GPU requirement")
    _equal(runtime["normalized_device_requests"], [], "device requests")
    _equal(runtime["nvidia_device_nodes"], [], "NVIDIA nodes")
    _equal(runtime["model_framework_import_allowed"], False, "model imports")
    _equal(set(runtime["thread_environment"].values()) >= {"0", "1", "UTC", "C.UTF-8"}, True, "thread freeze")

    operations = _mapping(
        config["source_only_operation_contract"], "source-only operations"
    )
    if not operations or any(type(value) is not int or value != 0 for value in operations.values()):
        raise ValueError("every source-only operation count must be exact zero")
    execution_operations = _mapping(
        config["execution_expected_operation_contract"], "execution operations"
    )
    expected_execution_operations = {
        "trajectory_semantic_decode_count": 58,
        "feature_state_count": 174,
        "ocr_semantic_decode_count": 290,
        "label_state_semantic_decode_count": 174,
        "distance_value_decode_count": 1624,
        "join_validation_count": 174,
        "candidate_feature_count": 522,
        "conditional_edge_count": 1682,
        "independent_target_count": 522,
        "development_semantic_decode_count": 0,
        "confirm_semantic_decode_count": 0,
        "training_example_count": 0,
        "optimizer_step_count": 0,
        "model_load_count": 0,
        "model_forward_count": 0,
        "oracle_metric_count": 0,
        "oof_metric_count": 0,
        "checkpoint_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
    }
    _equal(
        dict(execution_operations),
        expected_execution_operations,
        "execution operation schedule",
    )
    authorization = _mapping(config["authorization"], "authorization")
    for key, value in authorization.items():
        if value is not False:
            raise ValueError(f"source-only authorization must remain false: {key}")
    _equal(
        authorization["execution_allowed_without_machine_generated_runner_freeze_b"],
        False,
        "execution without generated runner freeze B",
    )


def validate_formal_cache_config(
    config: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    _exact_keys(config, TOP_LEVEL_KEYS, "formal-cache config")
    _equal(config["schema_version"], SCHEMA_VERSION, "schema version")
    _equal(config["protocol_id"], PROTOCOL_ID, "protocol ID")
    _equal(config["status"], SOURCE_STATUS, "source status")
    _validate_frozen_sections(config)
    _validate_source_remote_contract(config)
    prerequisites = _validate_git_prerequisites(config, root)
    _validate_parent_and_publication(config, root)
    _validate_inputs(config)
    geometry = _validate_geometry(config, root)
    _validate_firewall_formats_and_state(config)
    _validate_destination_runtime_and_authority(config)
    return {
        "status": VALIDATION_STATUS,
        "protocol_id": PROTOCOL_ID,
        "formal_train_trajectory_count": geometry["trajectory_count"],
        "formal_train_state_count": geometry["state_count"],
        "formal_train_candidate_feature_count": geometry["candidate_feature_count"],
        "formal_train_distance_value_count": geometry["distance_value_count"],
        "formal_train_conditional_edge_count": geometry["conditional_edge_count"],
        "formal_train_source_ids_sha256": geometry["source_ids_sha256"],
        "git_prerequisite_count": len(prerequisites),
        "input_artifact_count": 4,
        "exact_hf_input_file_count": 9,
        "execution_authorized": False,
        "formal_label_access_authorized": False,
        "development_semantic_decode_count": 0,
        "confirm_semantic_decode_count": 0,
        "matched_nll_authorized": False,
        "closed_loop_authorized": False,
    }


def _canonical_contract_path(root: Path, supplied: str | Path) -> Path:
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = root / candidate
    canonical = (root / CANONICAL_CONFIG_PATH).resolve()
    if candidate.resolve() != canonical:
        raise ValueError("formal-cache config must use its canonical path")
    return canonical


def load_frozen_formal_cache_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> FormalCacheContract:
    root = Path(repository_root).resolve()
    source = _canonical_contract_path(root, path)
    payload = _regular_file_bytes(source, "formal-cache config")
    _equal(sha256_bytes(payload), FROZEN_CONFIG_SHA256, "frozen config SHA256")
    config = load_strict_json_object(source)
    validate_formal_cache_config(config, repository_root=root)
    return FormalCacheContract(
        data=config,
        sha256=FROZEN_CONFIG_SHA256,
        repository_root=root,
        source_path=source,
    )


def validate_source_only_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    contract = load_frozen_formal_cache_contract(path, repository_root=repository_root)
    source = _mapping(contract.data["source_freeze"], "source freeze")
    inventory: list[Mapping[str, Any]] = []
    for raw in _sequence(source["required_source_a_paths"], "Source-A paths"):
        relative = _safe_relative_path(raw, "Source-A path")
        payload = _regular_file_bytes(
            contract.repository_root / relative, f"Source-A path {relative}"
        )
        inventory.append(
            {"path": relative, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        )
    pending = _mapping(source["execution_b_runner_freeze"], "runner freeze B")
    _equal(pending["path"], RUNNER_FREEZE_B_PATH, "runner freeze B path")
    _equal(
        pending["status"],
        "pending_machine_generated_after_source_a",
        "runner freeze B status",
    )
    for key in (
        "must_be_absent_during_source_only_validation",
        "only_allowed_execution_b_source_tree_diff",
        "bind_required_source_a_paths",
        "bind_git_prerequisites",
        "bind_source_a_inventory_sha256",
    ):
        _equal(pending[key], True, f"runner freeze B {key}")
    pending_path = contract.repository_root / RUNNER_FREEZE_B_PATH
    if pending_path.exists() or pending_path.is_symlink():
        raise ValueError("runner freeze B must remain absent in Source-A")
    result = validate_formal_cache_config(
        contract.data, repository_root=contract.repository_root
    )
    return {
        **result,
        "config_path": CANONICAL_CONFIG_PATH,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "source_a_path_count": len(inventory),
        "source_a_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "runner_freeze_b_path": RUNNER_FREEZE_B_PATH,
        "runner_freeze_b_status": pending["status"],
        "runner_freeze_b_present": False,
        "source_only_operation_counts": dict(contract.operations),
        "execution_expected_operation_counts": dict(
            contract.execution_operations
        ),
        "pending_runner_freeze": True,
        "execution_authorized": False,
    }


__all__ = [
    "CANONICAL_CONFIG_PATH",
    "FEATURE_MANIFEST_STATUS",
    "FINAL_COMPLETION_STATUS",
    "FROZEN_CONFIG_SHA256",
    "FormalCacheContract",
    "LABEL_MANIFEST_STATUS",
    "PROTOCOL_ID",
    "RUNNER_FREEZE_B_PATH",
    "SOURCE_STATUS",
    "VALIDATION_STATUS",
    "canonical_json_bytes",
    "load_frozen_formal_cache_contract",
    "load_strict_json_object",
    "sha256_bytes",
    "validate_formal_cache_config",
    "validate_source_only_contract",
]
