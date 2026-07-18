"""Fail-closed Source-A contract for the fresh-16 primary gate evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_gate_v1_fresh16_evaluation_v1"
SOURCE_STATUS = "source_only_frozen_before_fresh16_semantic_access"
VALIDATION_STATUS = "VALID_SOURCE_ONLY_GATE_V1_FRESH16_EVALUATION_V1"
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_gate_v1_fresh16_evaluation_v1.json"
)
RUNNER_FREEZE_B_PATH = (
    "code/configs/causalcache_gate_v1_fresh16_evaluation_runner_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "c98647aecf6b07e0ccccf1601b5e21289b595b7a4a45cc7ab9a542b1bd7dff2e"
)
FRESH_SOURCE_IDS_SHA256 = (
    "1c37cbf6b67b0ddee61b3efe27d33b30c8fbfb471ced12f12e49624a10b82454"
)
GATE_PREREGISTRATION_SHA256 = (
    "37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b"
)
FORMAL_MODEL_REVISION = "23f6786075c7bff91f93fd7e8a878e070efb72a9"
DERIVED_REVISION = "630363a6adb692d72774f16dd0653a50216313ff"
LABEL_REVISION = "7a6c254b8cec0dd3d8111dfc9c080de357e5cef3"

TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "lineage",
    "source_freeze",
    "formal_model_input",
    "derived_input",
    "label_input",
    "policy_vision_input",
    "fresh_geometry",
    "access_firewall",
    "evaluation_contract",
    "output_contract",
    "destination",
    "local_first_state_machine",
    "runtime_contract",
    "source_only_operation_contract",
    "execution_planned_operation_contract",
    "authorization",
}

PAYLOAD_TARGETS = (
    "fresh16-eval/v1/caches/feature-states-v1.jsonl",
    "fresh16-eval/v1/caches/label-states-v1.jsonl",
    "fresh16-eval/v1/heuristics/dynamic-recent-v1.json",
    "fresh16-eval/v1/heuristics/ocr-rgb-v2.json",
    "fresh16-eval/v1/heuristics/policy-vision-v3.json",
    "fresh16-eval/v1/scores/ocr-rgb-v2.jsonl",
    "fresh16-eval/v1/scores/policy-vision-v3.jsonl",
    "fresh16-eval/v1/learned/conditional-v1.json",
    "fresh16-eval/v1/learned/independent-v1.json",
)
REPORT_TARGETS = (
    "fresh16-eval/v1/reports/primary-state-records-v1.jsonl",
    "fresh16-eval/v1/reports/primary-report-v1.json",
    "fresh16-eval/v1/manifests/run-manifest-v1.json",
    "fresh16-eval/v1/manifests/bundle-manifest-v1.json",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Fresh16EvaluationContract:
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path
    source_path: Path

    @property
    def source(self) -> Mapping[str, Any]:
        return _mapping(self.data["source_freeze"], "source freeze")

    @property
    def model(self) -> Mapping[str, Any]:
        return _mapping(self.data["formal_model_input"], "formal model input")

    @property
    def derived(self) -> Mapping[str, Any]:
        return _mapping(self.data["derived_input"], "derived input")

    @property
    def labels(self) -> Mapping[str, Any]:
        return _mapping(self.data["label_input"], "label input")

    @property
    def geometry(self) -> Mapping[str, Any]:
        return _mapping(self.data["fresh_geometry"], "fresh geometry")

    @property
    def output(self) -> Mapping[str, Any]:
        return _mapping(self.data["output_contract"], "output contract")

    @property
    def destination(self) -> Mapping[str, Any]:
        return _mapping(self.data["destination"], "destination")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} drifted: expected {expected!r}, got {actual!r}")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys drifted: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    fingerprint = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if fingerprint(before) != fingerprint(after) or len(payload) != after.st_size:
        raise ValueError(f"{label} changed while being read")
    return payload


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {label}: {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    return _mapping(value, label)


def _validate_source(config: Mapping[str, Any]) -> None:
    source = _mapping(config["source_freeze"], "source freeze")
    _equal(source.get("branch"), "main", "source branch")
    _equal(source.get("origin_name"), "origin", "source origin name")
    _equal(
        source.get("origin_url"),
        "https://github.com/luojiaxuan/CausalCache.git",
        "source origin URL",
    )
    paths = tuple(
        _safe_relative(item, "Source-A path")
        for item in _sequence(source.get("required_source_a_paths"), "Source-A paths")
    )
    if not paths or len(paths) != len(set(paths)) or CANONICAL_CONFIG_PATH not in paths:
        raise ValueError("Source-A path inventory is empty, duplicated, or incomplete")
    runner = _mapping(source.get("execution_b_runner_freeze"), "runner freeze")
    _equal(runner.get("path"), RUNNER_FREEZE_B_PATH, "runner-freeze path")
    for key in (
        "must_be_absent_during_source_only_validation",
        "only_allowed_execution_b_source_tree_diff",
        "direct_single_parent_child_of_source_a",
        "bind_required_source_a_paths",
        "bind_git_prerequisites",
        "bind_source_a_inventory_sha256",
        "bind_loaded_module_inventory_sha256",
    ):
        _equal(runner.get(key), True, f"runner-freeze {key}")
    prerequisites = _sequence(source.get("git_prerequisites"), "Git prerequisites")
    if len(prerequisites) != 8:
        raise ValueError("Git prerequisite count drifted")
    seen: set[str] = set()
    for raw in prerequisites:
        record = _mapping(raw, "Git prerequisite")
        _exact_keys(record, {"path", "sha256", "size_bytes"}, "Git prerequisite")
        path = _safe_relative(record["path"], "Git prerequisite path")
        if (
            path in seen
            or _SHA256.fullmatch(str(record["sha256"])) is None
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] <= 0
        ):
            raise ValueError("Git prerequisite identity is malformed or duplicated")
        seen.add(path)


def _validate_inputs(config: Mapping[str, Any]) -> None:
    model = _mapping(config["formal_model_input"], "formal model input")
    required_model = {
        "repo": "gavinlaw/causalcache-gate-v1-formal58-selector-mobile",
        "repo_type": "model",
        "private": True,
        "payload_commit": "a6c9e7f6dab6bc27794438b5b66da07fd59b2889",
        "manifest_commit": FORMAL_MODEL_REVISION,
        "tag": "gate-v1-formal58-train-v1",
        "annotated_tag_object": "fa85e74685d5ab509e60b469c6c8cae61efab4d6",
        "model_load_device": "cpu",
        "training_or_optimizer_allowed": False,
    }
    for key, expected in required_model.items():
        _equal(model.get(key), expected, f"formal model {key}")
    manifests = _sequence(model.get("ensemble_manifests"), "ensemble manifests")
    auxiliary = _sequence(
        model.get("remote_tree_auxiliary_files"),
        "formal model remote-tree auxiliary files",
    )
    checkpoints = _sequence(model.get("checkpoints"), "checkpoints")
    if (
        len(manifests) != 2
        or [item.get("family") for item in manifests] != ["conditional", "independent"]
        or len(auxiliary) != 4
        or len(checkpoints) != 10
    ):
        raise ValueError("formal model manifest/checkpoint inventory drifted")
    _equal(
        [dict(_mapping(item, "formal model auxiliary file")) for item in auxiliary],
        [
            {
                "kind": "conditional_full_oof_report",
                "path": "formal58-train/v1/reports/conditional-full-oof-report.json",
                "sha256": "c959f16be25ab7dba9121e00e66237408df3c0f61eb51be959c3e5025436db60",
                "size_bytes": 29845,
            },
            {
                "kind": "independent_full_oof_report",
                "path": "formal58-train/v1/reports/independent-full-oof-report.json",
                "sha256": "0b1d2cca64be7b2c992bbee291aefc8df3e77e7aa4f1228be1b04330bea1a2e8",
                "size_bytes": 31172,
            },
            {
                "kind": "run_manifest",
                "path": "formal58-train/v1/manifests/run-manifest.json",
                "sha256": "873dba0bdf6e1905b36bb618107fc1bb5ccaa3f4bb878c0ab2774bfd3bd1850f",
                "size_bytes": 12208,
            },
            {
                "kind": "bundle_manifest",
                "path": "formal58-train/v1/manifests/bundle-manifest.json",
                "sha256": "5916e0a8bef8911cafe4e52db18ce595826577077857b8425ea5cb3783aa3d07",
                "size_bytes": 3181,
            },
        ],
        "formal model remote-tree auxiliary inventory",
    )
    expected_family_seed = [
        *(('conditional', seed) for seed in range(5)),
        *(('independent', seed) for seed in range(5)),
    ]
    observed_family_seed = []
    checkpoint_digests: set[str] = set()
    state_digests: set[str] = set()
    for record in checkpoints:
        item = _mapping(record, "formal checkpoint")
        observed_family_seed.append((item.get("family"), item.get("seed")))
        _safe_relative(item.get("path"), "formal checkpoint path")
        if (
            _SHA256.fullmatch(str(item.get("sha256"))) is None
            or _SHA256.fullmatch(str(item.get("model_state_sha256"))) is None
            or type(item.get("selected_epoch")) is not int
            or not 1 <= item["selected_epoch"] <= 500
            or type(item.get("size_bytes")) is not int
            or item["size_bytes"] <= 0
        ):
            raise ValueError("formal checkpoint binding is malformed")
        checkpoint_digests.add(item["sha256"])
        state_digests.add(item["model_state_sha256"])
    if (
        observed_family_seed != expected_family_seed
        or len(checkpoint_digests) != 10
        or len(state_digests) != 10
    ):
        raise ValueError("formal checkpoint order or distinctness drifted")

    derived = _mapping(config["derived_input"], "derived input")
    for key, expected in (
        ("repo", "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile"),
        ("repo_type", "dataset"),
        ("private", True),
        ("tag", "restoration-v2-label-expansion-v1.0.0"),
        (
            "artifact_tree_sha256",
            "9394b369e2b741e6aacf9ece4fc5dae3e6337b25a7e65402307e4e7862b94abc",
        ),
        ("generic_full_semantic_reader_allowed", False),
    ):
        _equal(derived.get(key), expected, f"derived {key}")
    _equal(derived.get("immutable_revision"), DERIVED_REVISION, "derived revision")
    _equal(derived.get("trajectory_rows_allowed_for_semantic_decode"), [48, 64], "trajectory rows")
    _equal(derived.get("selected_ocr_and_image_path_count"), 80, "selected image count")
    files = _sequence(derived.get("files"), "derived files")
    expected_derived = {
        "images": ("9e879e3ca59e7db058a68366a4289f578f0af67dabff2f3c517809fe03cb3ecf", 343674880, 384),
        "manifest": ("12d6ae78e8ef8750ee0dfb004653856316e0845bd3c498f96232b919bf8e7947", 8445, 1),
        "ocr": ("a1a0a979a74f026dd466df6c5e956c741110e34102cee0ff996695863c749dd8", 2972989, 384),
        "trajectories": ("fe93e9deeeb9a3c018227fb781b29f6efefefbb728db987584b650d9df353a6d", 1245673, 64),
    }
    if len(files) != 4:
        raise ValueError("derived exact-four file inventory drifted")
    for raw in files:
        item = _mapping(raw, "derived file")
        kind = item.get("kind")
        if kind not in expected_derived:
            raise ValueError("derived file kind drifted")
        _safe_relative(item.get("path"), "derived file path")
        expected = expected_derived[kind]
        _equal(
            (item.get("sha256"), item.get("size_bytes"), item.get("record_count")),
            expected,
            f"derived {kind} binding",
        )

    labels = _mapping(config["label_input"], "label input")
    for key, expected in (
        (
            "repo",
            "gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-repaired-mobile",
        ),
        ("repo_type", "dataset"),
        ("private", True),
        ("tag", "v2.2-expansion-exact-labels-scientific-repair-v1"),
        ("annotated_tag_object", "6a907ba2a3dce07c9f0810a84a8755c389da4a57"),
        ("generic_full_semantic_reader_allowed", False),
    ):
        _equal(labels.get(key), expected, f"label {key}")
    _equal(labels.get("immutable_revision"), LABEL_REVISION, "label revision")
    _equal(
        labels.get("archive_member_prefix"),
        "restoration-v2-2-expansion-exact-labels-scientific-repair-v1",
        "label archive member prefix",
    )
    _equal(labels.get("raw_state_rows_allowed_for_semantic_decode"), [144, 192], "label rows")
    _equal(labels.get("access_requires_completed_local_heuristic_seal"), True, "label access seal")
    _equal(
        dict(_mapping(labels.get("archive"), "label archive")),
        {
            "path": "repaired/v2.2-expansion-exact-labels-scientific-repair-v1/repaired-labels-v1.tar",
            "sha256": "1a9fdcc08aeb83f88bcd50957c3d890e3b103f2063a2aecbe08610d450950e01",
            "size_bytes": 3020800,
            "member_count": 4,
            "tree_inventory_sha256": "09bb681b2715997df326ab9648d8501955ec6ba127631a97141484631a98f44b",
        },
        "label archive binding",
    )
    _equal(
        dict(_mapping(labels.get("sidecar"), "label sidecar")),
        {
            "path": "repaired/v2.2-expansion-exact-labels-scientific-repair-v1/artifact-manifest-v1.json",
            "sha256": "9d86c75597a2a65eae952dd311373a7adbb0536d56d88e7fdb00d674f8dc4ee5",
            "size_bytes": 3641,
        },
        "label sidecar binding",
    )
    raw_states = _mapping(labels.get("raw_states_member"), "raw-state member")
    required_raw = {
        "path": "raw_states.jsonl",
        "sha256": "850aa0f1a6c0c6dd312573e4963384257a724ba01a3fe78ac4496cfd0fd14267",
        "size_bytes": 1126543,
        "record_count": 192,
        "run_contract_sha256": "63cddd712d9fa22bbd68110966161f36318d5fe1962b9b1477bae342da4180b8",
    }
    _equal(dict(raw_states), required_raw, "raw-state member binding")

    policy = _mapping(config["policy_vision_input"], "policy vision input")
    _equal(
        dict(policy),
        {
            "repo": "mPLUG/GUI-Owl-1.5-8B-Instruct",
            "immutable_revision": "06d5faecff74840bab2be2425e9c42667a5d04fc",
            "snapshot_manifest_sha256": "50b675ec31c5c46dbb0d44c137a808fffb9d054916d39b596648d4eb9df7cbc3",
            "snapshot_file_count": 14,
            "runtime_contract_sha256": "794474d8bc60463ba10fdd772691461f5910ca5e5501f7cccff4c53542b84b7f",
            "gpu_uuid_type_profile": "cuda_device_property_uuid_torch_c_cuuuid_v2",
            "image_processor_size_profile": "transformers_image_utils_size_dict_exact_edges_v3",
            "dtype": "bfloat16",
            "attention_implementation": "eager",
            "candidate_counts": [2, 3, 4],
            "image_counts": [3, 4, 5],
            "feature_repeats_per_state": 2,
            "cross_device_n4_sentinel_count": 1,
            "score_replay_max_abs_tolerance": 0.000001,
            "goal_text_or_ocr_or_labels_allowed": False,
        },
        "policy vision input binding",
    )


def _validate_geometry_evaluation_output(config: Mapping[str, Any]) -> None:
    geometry = _mapping(config["fresh_geometry"], "fresh geometry")
    required_geometry = {
        "roster": "fresh_development",
        "trajectory_count": 16,
        "state_count": 48,
        "candidate_feature_count": 144,
        "distance_value_count": 448,
        "conditional_edge_count": 464,
        "source_ids_sha256": FRESH_SOURCE_IDS_SHA256,
        "decision_steps": [4, 5, 6],
        "candidate_event_counts": [2, 3, 4],
        "budget_event_capacity": 2,
        "state_order": "source_order_then_decision_step_4_5_6",
    }
    for key, expected in required_geometry.items():
        _equal(geometry.get(key), expected, f"fresh geometry {key}")
    source_ids = _sequence(geometry.get("source_ids"), "fresh source IDs")
    if (
        len(source_ids) != 16
        or len(set(source_ids)) != 16
        or sha256_bytes(canonical_json_bytes(list(source_ids)))
        != FRESH_SOURCE_IDS_SHA256
    ):
        raise ValueError("fresh source roster order/digest drifted")

    evaluation = _mapping(config["evaluation_contract"], "evaluation contract")
    bootstrap = _mapping(evaluation.get("bootstrap"), "bootstrap")
    _equal(
        dict(bootstrap),
        {
            "unit": "trajectory",
            "resamples": 10000,
            "seed": 271828,
            "confidence": 0.9,
            "interval": "percentile",
            "quantile": "Hyndman_Fan_type_7",
        },
        "bootstrap contract",
    )
    _equal(
        evaluation.get("heuristic_comparators_in_fixed_order"),
        ["dynamic_recent", "ocr_rgb_v2", "policy_vision_v3"],
        "heuristic order",
    )
    _equal(
        dict(_mapping(evaluation.get("go_selector_all"), "selector GO thresholds")),
        {
            "ensemble_normalized_recovery_over_exact_minimum": 0.8,
            "ensemble_raw_utility_over_exact_raw_minimum": 0.8,
            "mean_normalized_delta_vs_each_heuristic_minimum": 0.05,
            "strongest_heuristic_positive_trajectory_count_minimum": 12,
            "strongest_heuristic_paired_bootstrap_lower_strictly_greater_than": 0.0,
            "individual_seed_exact_ratio_minimum": 0.75,
            "individual_seed_pass_count_minimum": 4,
            "seed_mean_recovery_population_std_maximum": 0.08,
            "hard_n3_n4_equal_trajectory_raw_utility_ratio_minimum": 0.7,
            "selected_true_nonpositive_addition_rate_maximum": 0.1,
        },
        "selector GO thresholds",
    )
    _equal(
        dict(
            _mapping(
                evaluation.get("go_set_conditioning_all"),
                "set-conditioning GO thresholds",
            )
        ),
        {
            "conditional_minus_independent_mean_normalized_delta_minimum": 0.02,
            "conditional_minus_independent_raw_utility_delta_strictly_greater_than": 0.0,
            "conditional_minus_independent_positive_trajectory_count_minimum": 12,
            "conditional_minus_independent_paired_bootstrap_lower_strictly_greater_than": 0.0,
            "paired_seed_positive_count_minimum": 4,
        },
        "set-conditioning GO thresholds",
    )
    for key in (
        "combined21_compatibility_evaluated",
        "confirm_access_authorized",
        "matched_nll_authorized",
        "closed_loop_authorized",
    ):
        _equal(evaluation.get(key), False, f"evaluation {key}")

    output = _mapping(config["output_contract"], "output contract")
    payload = _mapping(output.get("payload_commit"), "payload commit")
    report = _mapping(output.get("report_commit"), "report commit")
    _equal(tuple(payload.get("exact_targets", ())), PAYLOAD_TARGETS, "payload targets")
    _equal(tuple(report.get("exact_new_targets", ())), REPORT_TARGETS, "report targets")
    _equal(output.get("payload_target_count"), 9, "payload target count")
    _equal(output.get("report_target_count"), 4, "report target count")
    _equal(output.get("final_target_count"), 13, "final target count")
    if len(set((*PAYLOAD_TARGETS, *REPORT_TARGETS))) != 13:
        raise ValueError("fresh output target paths collide")
    for path in (*PAYLOAD_TARGETS, *REPORT_TARGETS):
        _safe_relative(path, "fresh output path")
    destination = _mapping(config["destination"], "destination")
    required_destination = {
        "repo": "gavinlaw/causalcache-gate-v1-fresh16-evaluation-mobile",
        "repo_type": "dataset",
        "private": True,
        "tag": "gate-v1-fresh16-evaluation-v1",
        "annotated_tag_required": True,
        "tag_message": "Freeze CausalCache gate v1 fresh16 primary evaluation",
        "tag_must_resolve_to_report_commit": True,
        "remote_base_receipt_before_first_mutation": True,
        "partial_or_conflicting_remote_state_action": "FAIL_CLOSED",
        "completed_replay_remote_mutation_count": 0,
    }
    _equal(dict(destination), required_destination, "destination binding")


def _validate_firewall_operations(config: Mapping[str, Any]) -> None:
    source_operations = _mapping(
        config["source_only_operation_contract"], "source-only operations"
    )
    expected_source_operation_keys = {
        "network_call_count",
        "hf_api_call_count",
        "file_write_count",
        "torch_import_count",
        "fresh16_trajectory_semantic_decode_count",
        "fresh16_ocr_semantic_decode_count",
        "fresh16_unique_image_identity_count",
        "fresh16_feature_image_decode_count",
        "policy_vision_image_decode_count",
        "fresh16_image_decode_count",
        "fresh16_label_semantic_decode_count",
        "model_load_count",
        "model_forward_count",
        "heuristic_selection_count",
        "primary_report_count",
        "legacy_dev5_semantic_decode_count",
        "confirm20_access_count",
        "matched_nll_evaluation_count",
        "closed_loop_episode_count",
    }
    _equal(
        dict(source_operations),
        {key: 0 for key in expected_source_operation_keys},
        "source-only operation contract",
    )
    authorization = _mapping(config["authorization"], "authorization")
    expected_authorization_keys = {
        "evaluation_executed",
        "execution_authorized",
        "fresh16_access_authorized",
        "label_access_authorized",
        "legacy_dev5_access_authorized",
        "confirm20_access_authorized",
        "matched_nll_authorized",
        "closed_loop_authorized",
        "execution_allowed_without_machine_generated_runner_freeze_b",
    }
    _equal(
        dict(authorization),
        {key: False for key in expected_authorization_keys},
        "Source-A authorization contract",
    )
    firewall = _mapping(config["access_firewall"], "access firewall")
    expected_firewall = {
        "source_a_fresh16_trajectory_semantic_decode_count": 0,
        "source_a_fresh16_ocr_semantic_decode_count": 0,
        "source_a_fresh16_image_decode_count": 0,
        "source_a_fresh16_label_semantic_decode_count": 0,
        "source_a_model_load_count": 0,
        "source_a_model_forward_count": 0,
        "source_a_legacy_dev5_semantic_decode_count": 0,
        "source_a_confirm20_access_count": 0,
        "source_a_matched_nll_evaluation_count": 0,
        "source_a_closed_loop_episode_count": 0,
        "execution_b_legacy_dev5_semantic_decode_count": 0,
        "execution_b_confirm20_access_count": 0,
        "execution_b_matched_nll_evaluation_count": 0,
        "execution_b_closed_loop_episode_count": 0,
        "all_heuristics_and_learned_selections_sealed_before_label_access": True,
        "primary_report_tag_and_readback_before_any_legacy_dev5_access": True,
    }
    _equal(dict(firewall), expected_firewall, "access firewall contract")

    runtime = _mapping(config["runtime_contract"], "runtime contract")
    _equal(runtime.get("host"), "hyper00_or_hyper01_h200", "runtime host")
    _equal(
        runtime.get("container_image_id"),
        "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa",
        "runtime container image",
    )
    for key, expected in (
        ("python_implementation", "CPython"),
        ("python_version", "3.12.3"),
        ("machine", "x86_64"),
        ("torch_version", "2.11.0+cu130"),
        ("safetensors_version", "0.7.0"),
        ("huggingface_hub_version", "1.16.1"),
        ("transformers_version", "5.6.0"),
        ("pillow_version", "12.2.0"),
    ):
        _equal(runtime.get(key), expected, f"runtime {key}")
    _equal(
        dict(
            _mapping(
                runtime.get("docker_inspect_receipt"),
                "Docker inspect receipt contract",
            )
        ),
        {
            "schema_version": "1.0.0",
            "status": "CAPTURED_GATE_V1_FRESH16_DOCKER_INSPECT_V1",
            "path": "/data/experiments/causalcache/.gate-v1-fresh16-evaluation-v1.docker-inspect.json",
            "mode": 0o600,
            "container_name_prefix": "sglang-omni-jaxan-",
            "runtime": "runc",
            "data_mount_destination": "/data",
            "data_mount_rw": True,
            "privileged": False,
            "gpu_device_request_count": 1,
            "gpu_count": 2,
        },
        "Docker inspect receipt contract",
    )
    _equal(
        dict(_mapping(runtime.get("policy_phase"), "policy runtime phase")),
        {
            "gpu_count": 2,
            "worker_count": 2,
            "states_per_worker": 24,
            "dtype": "bfloat16",
            "feature_repeats": 2,
            "processor_batch_count": 49,
            "vision_forward_count": 97,
            "cosine_scalar_transfer_count": 292,
        },
        "policy runtime phase",
    )
    _equal(
        dict(_mapping(runtime.get("finalize_phase"), "finalize runtime phase")),
        {
            "device": "cpu",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "gpu_required": False,
        },
        "finalize runtime phase",
    )
    _equal(
        dict(_mapping(runtime.get("thread_environment"), "runtime environment")),
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "TZ": "UTC",
            "LC_ALL": "C.UTF-8",
        },
        "runtime environment",
    )

    local = _mapping(config["local_first_state_machine"], "local state machine")
    _equal(
        dict(local),
        {
            "state_file_mode": 0o600,
            "artifact_file_mode": 0o444,
            "state_directory": "/data/experiments/causalcache",
            "artifact_directory": "/data/artifacts/causalcache/gate-v1-fresh16-evaluation-v1",
            "o_excl_or_byte_identical_reuse": True,
            "no_overwrite": True,
            "ordered_states": [
                "runtime_receipts",
                "global_claim",
                "transport_verification",
                "label_blind_cpu_completion",
                "checkpoint_replay_completion",
                "policy_worker_even_completion",
                "policy_worker_odd_completion",
                "heuristic_local_seal",
                "label_access_claim",
                "label_cache_completion",
                "remote_base_receipt",
                "payload_commit_receipt",
                "primary_report_completion",
                "report_commit_receipt",
                "completion_staging",
                "final_completion",
            ],
        },
        "local state-machine contract",
    )
    operations = _mapping(
        config["execution_planned_operation_contract"],
        "planned execution operations",
    )
    required_counts = {
        "fresh16_trajectory_semantic_decode_count": 16,
        "fresh16_ocr_semantic_decode_count": 80,
        "fresh16_unique_image_identity_count": 80,
        "fresh16_feature_image_decode_count": 80,
        "policy_vision_image_decode_count": 197,
        "fresh16_image_decode_count": 277,
        "feature_state_count": 48,
        "candidate_feature_count": 144,
        "dynamic_recent_selection_count": 48,
        "ocr_rgb_selection_count": 48,
        "policy_vision_selection_count": 48,
        "policy_vision_processor_batch_count": 49,
        "policy_vision_feature_forward_count": 97,
        "policy_vision_cosine_scalar_transfer_count": 292,
        "fresh16_label_semantic_decode_count": 48,
        "distance_value_count": 448,
        "conditional_edge_count": 464,
        "checkpoint_load_count": 10,
        "optimizer_step_count": 0,
        "exact_subset_oracle_state_count": 48,
        "exact_subset_oracle_invocation_count": 96,
        "bootstrap_interval_count": 2,
        "bootstrap_resamples_per_interval": 10000,
        "bootstrap_resample_draw_count": 20000,
        "primary_state_record_count": 48,
        "primary_report_count": 1,
        "payload_commit_count": 1,
        "report_commit_count": 1,
        "tag_count": 1,
        "legacy_dev5_semantic_decode_count": 0,
        "confirm20_access_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
    }
    _equal(dict(operations), required_counts, "execution operation contract")


def validate_fresh16_evaluation_contract_data(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    _exact_keys(config, TOP_LEVEL_KEYS, "fresh16 evaluation config")
    _equal(config.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(config.get("protocol_id"), PROTOCOL_ID, "protocol ID")
    _equal(config.get("status"), SOURCE_STATUS, "source status")
    _validate_source(config)
    _validate_inputs(config)
    _validate_geometry_evaluation_output(config)
    _validate_firewall_operations(config)
    return config


def _canonical_config(root: Path, supplied: str | Path) -> Path:
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = root / candidate
    canonical = (root / CANONICAL_CONFIG_PATH).resolve()
    if candidate.resolve() != canonical:
        raise ValueError("fresh16 evaluation config must use its canonical path")
    return canonical


def load_frozen_fresh16_evaluation_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> Fresh16EvaluationContract:
    root = Path(repository_root).resolve()
    source = _canonical_config(root, path)
    payload = _regular_file_bytes(source, label="fresh16 evaluation config")
    _equal(sha256_bytes(payload), FROZEN_CONFIG_SHA256, "frozen config SHA256")
    config = _strict_json(payload, label="fresh16 evaluation config")
    validate_fresh16_evaluation_contract_data(config)
    contract = Fresh16EvaluationContract(
        data=config,
        sha256=FROZEN_CONFIG_SHA256,
        repository_root=root,
        source_path=source,
    )
    for raw in _sequence(contract.source["git_prerequisites"], "Git prerequisites"):
        record = _mapping(raw, "Git prerequisite")
        relative = _safe_relative(record["path"], "Git prerequisite path")
        prerequisite = _regular_file_bytes(root / relative, label=relative)
        if (
            len(prerequisite) != record["size_bytes"]
            or sha256_bytes(prerequisite) != record["sha256"]
        ):
            raise ValueError(f"Git prerequisite identity drifted: {relative}")
    _validate_lightweight_lineage(contract)
    return contract


def _validate_lightweight_lineage(contract: Fresh16EvaluationContract) -> None:
    lineage = _mapping(contract.data["lineage"], "lineage")

    def load_binding(name: str) -> Mapping[str, Any]:
        binding = _mapping(lineage[name], f"lineage {name}")
        relative = _safe_relative(binding["path"], f"lineage {name} path")
        payload = _regular_file_bytes(
            contract.repository_root / relative,
            label=f"lineage {name}",
        )
        if (
            len(payload) != binding["size_bytes"]
            or sha256_bytes(payload) != binding["sha256"]
        ):
            raise ValueError(f"lineage {name} byte binding drifted")
        return _strict_json(payload, label=f"lineage {name}")

    prereg = load_binding("gate_preregistration")
    fresh = _mapping(
        _mapping(prereg.get("rosters"), "preregistered rosters").get(
            "fresh_development"
        ),
        "preregistered fresh development",
    )
    for key, expected in (
        ("trajectory_count", 16),
        ("state_count", 48),
        ("conditional_edge_count", 464),
        ("source_ids_sha256", FRESH_SOURCE_IDS_SHA256),
    ):
        _equal(fresh.get(key), expected, f"preregistered fresh {key}")

    split = load_binding("label_expansion_split")
    split_fresh = _mapping(
        _mapping(split.get("selection"), "split selection").get(
            "gate_development_expansion"
        ),
        "split fresh development",
    )
    _equal(
        split_fresh.get("source_ids_sha256"),
        FRESH_SOURCE_IDS_SHA256,
        "split fresh digest",
    )
    _equal(
        split_fresh.get("source_ids"),
        contract.geometry["source_ids"],
        "split fresh source order",
    )

    selection = load_binding("label_expansion_selection")
    for key, expected in (
        ("status", "FROZEN_POLICY_BLIND_STRUCTURAL_SPLIT"),
        ("structural_manifest_only", True),
        ("policy_output_generated", False),
        ("restoration_output_generated", False),
    ):
        _equal(selection.get(key), expected, f"selection {key}")
    selection_split = _mapping(
        _mapping(selection.get("splits"), "selection splits").get(
            "gate_development_expansion"
        ),
        "selection fresh development",
    )
    selected_trajectories = _sequence(
        selection_split.get("trajectories"), "selected fresh trajectories"
    )
    selected_states = _sequence(selection_split.get("states"), "selected states")
    _equal(
        [item.get("source_id") for item in selected_trajectories],
        contract.geometry["source_ids"],
        "selected fresh source order",
    )
    _equal(len(selected_states), 48, "selected fresh state count")
    _equal(
        [item.get("state_id") for item in selected_states],
        [
            f"{source_id}:decision_step:{decision_step:03d}"
            for source_id in contract.geometry["source_ids"]
            for decision_step in contract.geometry["decision_steps"]
        ],
        "selected fresh state order",
    )
    workload = _mapping(
        _mapping(selection.get("expected_workload"), "selection workload").get(
            "gate_development_expansion"
        ),
        "selection fresh workload",
    )
    for key, expected in (
        ("trajectories", 16),
        ("states", 48),
        ("full_subset_distance_rows", 448),
        ("deployment_conditional_edges", 464),
    ):
        _equal(workload.get(key), expected, f"selection fresh workload {key}")

    exposure = load_binding("label_expansion_exposure")
    _equal(
        exposure.get("status"),
        "FROZEN_PRE_EXPANSION_POLICY_OUTPUT_EXPOSURE",
        "exposure status",
    )
    fresh_exposure = _mapping(
        _mapping(
            _mapping(exposure.get("cohort_inventory"), "exposure cohorts").get(
                "label_expansion"
            ),
            "exposure expansion cohort",
        ).get("gate_development_expansion"),
        "exposure fresh development",
    )
    for key, expected in (
        ("trajectory_count", 16),
        ("source_ids_sha256", FRESH_SOURCE_IDS_SHA256),
        ("source_ids_emitted", False),
    ):
        _equal(fresh_exposure.get(key), expected, f"exposure fresh {key}")
    for key, expected in (
        ("all_required_intersections_are_empty", True),
    ):
        _equal(
            _mapping(exposure.get("overlap_proof"), "exposure overlap").get(key),
            expected,
            f"exposure overlap {key}",
        )
    for key, expected in (
        ("image_or_ocr_identity_emitted", False),
        ("instructions_or_actions_emitted", False),
        ("policy_or_restoration_values_emitted", False),
        ("source_ids_emitted", False),
    ):
        _equal(
            _mapping(exposure.get("leakage_firewall"), "exposure firewall").get(
                key
            ),
            expected,
            f"exposure firewall {key}",
        )
    for key, expected in (
        ("confirm_policy_output_accessed", False),
        ("confirm_restoration_output_accessed", False),
        ("expansion_gate_training_started_before_freeze", False),
        ("expansion_policy_output_generated_before_freeze", False),
        ("expansion_restoration_output_generated_before_freeze", False),
        ("selection_changed_after_expansion_output", False),
    ):
        _equal(
            _mapping(
                exposure.get("pre_policy_output_assertions"),
                "exposure pre-output assertions",
            ).get(key),
            expected,
            f"exposure pre-output assertion {key}",
        )

    model_summary = load_binding("formal_gate_completion")
    _equal(
        model_summary.get("status"),
        "COMPLETED_AND_REVALIDATED_GATE_V1_FORMAL58_TRAIN_V1",
        "formal gate completion status",
    )
    canonical = _mapping(
        _mapping(model_summary.get("artifact"), "model artifact").get("canonical_hf"),
        "canonical model artifact",
    )
    for key, expected in (
        ("repo", contract.model["repo"]),
        ("payload_commit", contract.model["payload_commit"]),
        ("manifest_commit", contract.model["manifest_commit"]),
        ("tag", contract.model["tag"]),
        ("annotated_tag_object", contract.model["annotated_tag_object"]),
    ):
        _equal(canonical.get(key), expected, f"formal model summary {key}")
    training = _mapping(model_summary.get("training"), "formal training summary")
    expected_checkpoints = []
    expected_manifests = []
    for family in ("conditional", "independent"):
        family_summary = _mapping(training.get(family), f"{family} summary")
        expected_manifests.append(
            (
                family,
                family_summary["ensemble_manifest_sha256"],
                family_summary["ensemble_provenance_sha256"],
            )
        )
        for item in _sequence(
            family_summary.get("checkpoints"), f"{family} checkpoints"
        ):
            expected_checkpoints.append(
                (
                    family,
                    item["seed"],
                    item["selected_epoch"],
                    item["artifact_sha256"],
                    item["size_bytes"],
                    item["model_state_sha256"],
                )
            )
    observed_checkpoints = [
        (
            item["family"],
            item["seed"],
            item["selected_epoch"],
            item["sha256"],
            item["size_bytes"],
            item["model_state_sha256"],
        )
        for item in contract.model["checkpoints"]
    ]
    observed_manifests = [
        (item["family"], item["sha256"], item["provenance_sha256"])
        for item in contract.model["ensemble_manifests"]
    ]
    _equal(observed_checkpoints, expected_checkpoints, "formal checkpoint lineage")
    _equal(observed_manifests, expected_manifests, "formal ensemble lineage")
    expected_auxiliary = []
    for family in ("conditional", "independent"):
        report = _mapping(training[family].get("oof_report"), f"{family} OOF report")
        expected_auxiliary.append(
            (
                f"{family}_full_oof_report",
                report["path"],
                report["sha256"],
                report["size_bytes"],
            )
        )
    for kind, key in (
        ("run_manifest", "run_manifest"),
        ("bundle_manifest", "bundle_manifest"),
    ):
        item = _mapping(canonical.get(key), f"formal {kind}")
        expected_auxiliary.append(
            (kind, item["path"], item["sha256"], item["size_bytes"])
        )
    observed_auxiliary = [
        (item["kind"], item["path"], item["sha256"], item["size_bytes"])
        for item in contract.model["remote_tree_auxiliary_files"]
    ]
    _equal(
        observed_auxiliary,
        expected_auxiliary,
        "formal remote-tree auxiliary lineage",
    )

    derived = load_binding("derived_completion")
    derived_artifact = _mapping(derived.get("artifact"), "derived artifact")
    _equal(
        derived_artifact.get("immutable_revision"),
        contract.derived["immutable_revision"],
        "derived completion revision",
    )
    _equal(
        _mapping(
            derived_artifact.get("fresh_download"),
            "derived fresh download",
        ).get("artifact_tree_sha256"),
        contract.derived["artifact_tree_sha256"],
        "derived completion tree",
    )
    label_summary = load_binding("repaired_label_completion")
    label_hf = _mapping(label_summary.get("hf"), "repaired label HF binding")
    for key, expected in (
        ("repo", contract.labels["repo"]),
        ("immutable_pair_commit", contract.labels["immutable_revision"]),
        ("tag", contract.labels["tag"]),
        ("annotated_tag_object", contract.labels["annotated_tag_object"]),
    ):
        _equal(label_hf.get(key), expected, f"repaired label {key}")
    snapshot = load_binding("gui_owl_snapshot")
    policy = _mapping(contract.data["policy_vision_input"], "policy vision input")
    _equal(snapshot.get("repo"), policy["repo"], "GUI-Owl repo")
    _equal(snapshot.get("revision"), policy["immutable_revision"], "GUI-Owl revision")
    _equal(
        len(_sequence(snapshot.get("files"), "GUI-Owl files")),
        14,
        "GUI-Owl file count",
    )


def _source_inventory(
    contract: Fresh16EvaluationContract,
) -> tuple[Mapping[str, Any], ...]:
    result = []
    for raw in _sequence(
        contract.source["required_source_a_paths"], "Source-A paths"
    ):
        relative = _safe_relative(raw, "Source-A path")
        payload = _regular_file_bytes(
            contract.repository_root / relative,
            label=f"Source-A path {relative}",
        )
        result.append(
            {"path": relative, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        )
    return tuple(result)


def validate_fresh16_evaluation_source_only_contract(
    path: str | Path = CANONICAL_CONFIG_PATH,
    *,
    repository_root: str | Path,
) -> Mapping[str, Any]:
    before_modules = frozenset(sys.modules)
    contract = load_frozen_fresh16_evaluation_contract(
        path,
        repository_root=repository_root,
    )
    if os.path.lexists(contract.repository_root / RUNNER_FREEZE_B_PATH):
        raise ValueError("fresh16 runner-freeze B must remain absent in Source-A")
    inventory = _source_inventory(contract)
    newly_loaded = frozenset(sys.modules) - before_modules
    prohibited = sorted(
        name
        for name in newly_loaded
        if name == "torch"
        or name.startswith("torch.")
        or name == "huggingface_hub"
        or name.startswith("huggingface_hub.")
    )
    if prohibited:
        raise ValueError("Source-A validation imported torch or Hugging Face")
    zero_counts = dict(
        _mapping(
            contract.data["source_only_operation_contract"],
            "source-only operation contract",
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": VALIDATION_STATUS,
        "config_path": CANONICAL_CONFIG_PATH,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "source_a_path_total": len(inventory),
        "source_a_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "runner_freeze_b_path": RUNNER_FREEZE_B_PATH,
        "runner_freeze_b_present": False,
        **zero_counts,
        "gate_trained": True,
        "evaluation_executed": False,
        "execution_authorized": False,
        "fresh16_access_authorized": False,
        "label_access_authorized": False,
        "legacy_dev5_access_authorized": False,
        "confirm20_access_authorized": False,
        "matched_nll_authorized": False,
        "closed_loop_authorized": False,
    }


__all__ = [
    "CANONICAL_CONFIG_PATH",
    "DERIVED_REVISION",
    "FORMAL_MODEL_REVISION",
    "FRESH_SOURCE_IDS_SHA256",
    "FROZEN_CONFIG_SHA256",
    "Fresh16EvaluationContract",
    "GATE_PREREGISTRATION_SHA256",
    "LABEL_REVISION",
    "PAYLOAD_TARGETS",
    "PROTOCOL_ID",
    "REPORT_TARGETS",
    "RUNNER_FREEZE_B_PATH",
    "SCHEMA_VERSION",
    "SOURCE_STATUS",
    "VALIDATION_STATUS",
    "canonical_json_bytes",
    "load_frozen_fresh16_evaluation_contract",
    "pretty_json_bytes",
    "sha256_bytes",
    "validate_fresh16_evaluation_contract_data",
    "validate_fresh16_evaluation_source_only_contract",
]
