"""Frozen source contract for the policy-vision v3 CPU validation repair."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = (
    "causalcache_restoration_v2_2_policy_vision_v3_validation_repair_v1"
)
SOURCE_STATUS = (
    "source_only_frozen_before_any_versioned_cpu_replay_or_audit_result"
)
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_restoration_v2_2_policy_vision_"
    "v3_validation_repair_v1.json"
)
FROZEN_CONFIG_SHA256 = (
    "64f63ab7563c227423c15ef82d5fd74d8be11579248908c7ec2880137e5d6ddf"
)
PRODUCER_SOURCE_GIT_COMMIT = "a935a3cf5efb1fa7a952ca6f45a5994609b367e9"
ARTIFACT_GIT_COMMIT = "597f050297342d9f29eb383985c2014f5782b2bb"
PRODUCER_PROTOCOL_ID = (
    "causalcache_restoration_v2_2_policy_vision_baseline_"
    "v3_size_dict_interface_repair"
)
PRODUCER_RUN_STATUS = (
    "COMPLETED_RESTORATION_V2_2_POLICY_VISION_BASELINE_"
    "V3_SIZE_DICT_INTERFACE_REPAIR"
)
PRODUCER_CONFIG_SHA256 = (
    "794474d8bc60463ba10fdd772691461f5910ca5e5501f7cccff4c53542b84b7f"
)
PRODUCER_CLOSURE_PATH_COUNT = 151
PRODUCER_CLOSURE_INVENTORY_SHA256 = (
    "91f380916ca54fd8baf8987d0c06c18e5728d6b5db99fb86b06cb1ff85114b6d"
)
RUN_STATUS = (
    "VALID_RESTORATION_V2_2_POLICY_VISION_V3_VALIDATION_REPAIR_V1"
)
VALIDATE_STATUS = (
    "REVALIDATED_RESTORATION_V2_2_POLICY_VISION_V3_VALIDATION_REPAIR_V1"
)
CANONICAL_RESULT_DIRECTORY = (
    "data/results/restoration_v2_2_policy_vision_baseline_"
    "v3_validation_repair_v1"
)
FORMAL_ATTEMPT_LEDGER_PATH = (
    "/data/experiments/causalcache/restoration-v2-2-policy-vision-"
    "v3-validation-repair-v1-attempt.json"
)
FORMAL_COMPLETION_SEAL_PATH = (
    "/data/experiments/causalcache/restoration-v2-2-policy-vision-"
    "v3-validation-repair-v1-completion-seal.json"
)
EXPECTED_OUTPUT_FILES = ("README.md", "summary.json")
FORMAL_PYTHON_SOURCE_ROOTS = ("code/causalcache", "code/scripts")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} schema drifted")


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _git_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or GIT_SHA_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a full lowercase Git SHA")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _git_text(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _git_bytes(root: Path, revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _require_commit(root: Path, revision: str) -> None:
    _git_sha(revision, "Git revision")
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError(f"required Git commit is absent: {revision}")


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError(f"{ancestor} is not an ancestor of {descendant}")


def _regular_file_bytes(path: Path, name: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    return path.read_bytes()


def _validate_file_binding(
    root: Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> bytes:
    _exact_keys(binding, {"path", "size_bytes", "sha256"}, name)
    path = binding.get("path")
    if (
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or Path(path).as_posix() != path
        or any(part in {".", ".."} for part in Path(path).parts)
    ):
        raise ValueError(f"{name} path must be a canonical repository-relative path")
    payload = _regular_file_bytes(root / path, name)
    if (
        len(payload) != _positive_int(binding.get("size_bytes"), f"{name} size")
        or sha256_bytes(payload) != _sha256(binding.get("sha256"), f"{name} SHA")
    ):
        raise ValueError(f"{name} bytes drifted")
    return payload


def validate_exact_artifact_directory(
    directory: Path,
    exact_files: Sequence[Mapping[str, Any]],
) -> dict[str, bytes]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("producer artifact directory is missing or symlinked")
    children = tuple(directory.iterdir())
    expected_names = [str(binding.get("path")) for binding in exact_files]
    if (
        len(expected_names) != len(set(expected_names))
        or sorted(path.name for path in children) != sorted(expected_names)
        or any(path.is_symlink() or not path.is_file() for path in children)
    ):
        raise ValueError("producer artifact exact-file inventory drifted")
    result: dict[str, bytes] = {}
    for binding in exact_files:
        _exact_keys(binding, {"path", "size_bytes", "sha256"}, "artifact file")
        name = binding.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("artifact filename must be a plain basename")
        payload = _regular_file_bytes(directory / name, f"artifact {name}")
        if (
            len(payload) != _positive_int(binding.get("size_bytes"), f"{name} size")
            or sha256_bytes(payload) != _sha256(binding.get("sha256"), f"{name} SHA")
        ):
            raise ValueError(f"producer artifact bytes drifted: {name}")
        result[name] = payload
    return result


def _python_source_closure(root: Path, revision: str) -> dict[str, Any]:
    paths = tuple(
        path
        for path in _git_text(
            root,
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "--",
            *FORMAL_PYTHON_SOURCE_ROOTS,
        ).splitlines()
        if path.endswith(".py")
    )
    records = [
        {
            "path": path,
            "git_blob_sha": _git_text(
                root,
                "rev-parse",
                f"{revision}:{path}",
            ),
        }
        for path in paths
    ]
    return {
        "rule": (
            "all_tracked_python_under_code_causalcache_and_code_scripts_"
            "at_source_commit"
        ),
        "path_count": len(paths),
        "inventory_sha256": sha256_bytes(canonical_json_bytes(records)),
        "paths": paths,
    }


def _producer_python_closure(root: Path) -> dict[str, Any]:
    return _python_source_closure(root, PRODUCER_SOURCE_GIT_COMMIT)


def _validate_config_semantics(config: Mapping[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema_version",
            "protocol_id",
            "preregistration_status",
            "repair_scope",
            "producer",
            "producer_artifact",
            "immutable_inputs",
            "failure_evidence",
            "projection_contract",
            "validation_source",
            "output_contract",
        },
        "validation-repair config",
    )
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("protocol_id") != PROTOCOL_ID
        or config.get("preregistration_status") != SOURCE_STATUS
    ):
        raise ValueError("validation-repair config identity drifted")

    scope = _mapping(config["repair_scope"], "repair scope")
    _exact_keys(
        scope,
        {
            "purpose",
            "producer_artifact_or_scientific_value_mutation_allowed",
            "producer_gpu_retry_allowed",
            "policy_or_vision_model_load_allowed",
            "image_decode_or_processor_batch_allowed",
            "gpu_query_or_cuda_operation_allowed",
            "policy_feature_or_language_forward_allowed",
            "restoration_label_regeneration_allowed",
            "reconstruction_must_match_all_producer_artifact_bytes",
        },
        "repair scope",
    )
    if scope.get("purpose") != (
        "repair_only_the_evaluated_state_to_feature_state_projection_in_the_"
        "v3_cpu_validator"
    ):
        raise ValueError("validation-repair purpose drifted")
    for key in (
        "producer_artifact_or_scientific_value_mutation_allowed",
        "producer_gpu_retry_allowed",
        "policy_or_vision_model_load_allowed",
        "image_decode_or_processor_batch_allowed",
        "gpu_query_or_cuda_operation_allowed",
        "policy_feature_or_language_forward_allowed",
        "restoration_label_regeneration_allowed",
    ):
        if scope.get(key) is not False:
            raise ValueError(f"validation repair must forbid {key}")
    if scope.get("reconstruction_must_match_all_producer_artifact_bytes") is not True:
        raise ValueError("validation repair must require exact-byte reconstruction")

    producer = _mapping(config["producer"], "producer")
    _exact_keys(
        producer,
        {
            "protocol_id",
            "run_status",
            "producer_source_git_commit",
            "artifact_git_commit",
            "v3_contract",
            "formal_python_source_closure",
        },
        "producer",
    )
    if (
        producer.get("protocol_id") != PRODUCER_PROTOCOL_ID
        or producer.get("run_status") != PRODUCER_RUN_STATUS
        or producer.get("producer_source_git_commit") != PRODUCER_SOURCE_GIT_COMMIT
        or producer.get("artifact_git_commit") != ARTIFACT_GIT_COMMIT
    ):
        raise ValueError("producer identity drifted")
    v3_contract = _mapping(producer["v3_contract"], "v3 contract")
    if (
        v3_contract.get("path")
        != "code/configs/causalcache_restoration_v2_2_policy_vision_baseline_"
        "v3_size_dict_interface_repair.json"
        or v3_contract.get("size_bytes") != 11037
        or v3_contract.get("sha256") != PRODUCER_CONFIG_SHA256
    ):
        raise ValueError("producer v3 contract binding drifted")
    closure = _mapping(
        producer["formal_python_source_closure"], "producer source closure"
    )
    if closure != {
        "rule": (
            "all_tracked_python_under_code_causalcache_and_code_scripts_"
            "at_source_commit"
        ),
        "path_count": PRODUCER_CLOSURE_PATH_COUNT,
        "inventory_sha256": PRODUCER_CLOSURE_INVENTORY_SHA256,
    }:
        raise ValueError("producer formal Python closure binding drifted")

    artifact = _mapping(config["producer_artifact"], "producer artifact")
    _exact_keys(
        artifact,
        {
            "directory",
            "exact_files",
            "scientific_payload_sha256",
            "formal_state_count",
            "candidate_score_count",
            "unique_image_count",
        },
        "producer artifact",
    )
    if (
        artifact.get("directory")
        != "data/results/restoration_v2_2_policy_vision_baseline_"
        "v3_size_dict_interface_repair"
        or artifact.get("scientific_payload_sha256")
        != "811e59c780851c19b7fe21314be1e52c1dbfcfa002e63a9857fd34d90ba94f48"
        or artifact.get("formal_state_count") != 15
        or artifact.get("candidate_score_count") != 60
        or artifact.get("unique_image_count") != 75
        or not isinstance(artifact.get("exact_files"), list)
        or len(artifact["exact_files"]) != 3
    ):
        raise ValueError("producer artifact contract drifted")
    expected_artifact_files = [
        {
            "path": "README.md",
            "size_bytes": 1504,
            "sha256": (
                "70ce9496983bfafeab04dd5184e5b7d5c8f02a70df9812893af0fe0aed47ed1f"
            ),
        },
        {
            "path": "state_scores.jsonl",
            "size_bytes": 68269,
            "sha256": (
                "8c5977260bb1a8618b43cd877d963bd637c197e589db532c7ccbcab9c55f58b9"
            ),
        },
        {
            "path": "summary.json",
            "size_bytes": 66273,
            "sha256": (
                "5ed21d6c397839bc60987bb3ee1db3a0560b6e923557cc1e8e524de921af1c0e"
            ),
        },
    ]
    if artifact["exact_files"] != expected_artifact_files:
        raise ValueError("producer exact-three file binding drifted")

    immutable = _mapping(config["immutable_inputs"], "immutable inputs")
    _exact_keys(
        immutable,
        {"restoration_labels", "ocr_rgb_identity_witness"},
        "immutable inputs",
    )
    labels = _mapping(immutable["restoration_labels"], "restoration labels")
    _exact_keys(
        labels,
        {
            "raw_archive_size_bytes",
            "raw_archive_sha256",
            "hf_repo",
            "hf_revision",
            "hf_path",
        },
        "restoration labels",
    )
    if labels != {
        "raw_archive_size_bytes": 3747840,
        "raw_archive_sha256": (
            "99120d5444d31962d9f4254c3e40bc5f06d3e4d3d90a74b1749e7ccd7aefb29e"
        ),
        "hf_repo": "gavinlaw/causalcache-restoration-labels-mobile",
        "hf_revision": "8f6baae5c0b23b08915fa1b0fb848dd519b4c8db",
        "hf_path": "raw/v2.2-eager-train-dev-exact-v2.tar",
    }:
        raise ValueError("restoration-label immutable identity drifted")
    witness = _mapping(
        immutable["ocr_rgb_identity_witness"], "OCR/RGB identity witness"
    )
    _exact_keys(
        witness,
        {"directory", "git_commit", "scientific_payload_sha256"},
        "OCR/RGB identity witness",
    )
    if witness != {
        "directory": "data/results/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair",
        "git_commit": "7e59591573cb31f178dfd07422cc2e3c8aeff573",
        "scientific_payload_sha256": (
            "5942519bdff8f3e8a64bbc8b32a5d42a64abff37daf0804fcce0f098a63765b2"
        ),
    }:
        raise ValueError("OCR/RGB immutable witness identity drifted")

    failure = _mapping(config["failure_evidence"], "failure evidence")
    _exact_keys(
        failure,
        {
            "status",
            "exception_type",
            "exception_message",
            "files",
            "bounded_diagnostic_source",
            "producer_gpu_attempt_ledger",
            "diagnostic_exact_file_matches",
        },
        "failure evidence",
    )
    if (
        failure.get("status")
        != "INVALID_POLICY_VISION_V3_CPU_VALIDATE_STATE_PROJECTION"
        or failure.get("exception_type") != "ValueError"
        or failure.get("exception_message")
        != "policy-vision row-to-worker provenance drifted"
        or failure.get("diagnostic_exact_file_matches")
        != {"README.md": True, "state_scores.jsonl": True, "summary.json": True}
    ):
        raise ValueError("CPU validation failure identity drifted")
    expected_failure_files = [
        {
            "path": (
                "data/results/restoration_v2_2_policy_vision_baseline_"
                "v3_cpu_validation_attempt/README.md"
            ),
            "size_bytes": 2208,
            "sha256": (
                "7caad61e23f51038b577906906c2e0004624bcff372db3bcfbe14512125ecc2a"
            ),
        },
        {
            "path": (
                "data/results/restoration_v2_2_policy_vision_baseline_"
                "v3_cpu_validation_attempt/failure.json"
            ),
            "size_bytes": 6344,
            "sha256": (
                "8b1c517838256665d378d100df1b0023b6360a71fe30f08296c54805f3443c29"
            ),
        },
    ]
    if failure.get("files") != expected_failure_files:
        raise ValueError("CPU validation failure file binding drifted")
    diagnostic = _mapping(
        failure["bounded_diagnostic_source"], "bounded diagnostic source"
    )
    _exact_keys(
        diagnostic, {"path", "size_bytes", "sha256"}, "bounded diagnostic source"
    )
    if diagnostic != {
        "path": "code/scripts/diagnose_restoration_v2_2_policy_vision_v3_cpu_replay.py",
        "size_bytes": 1866,
        "sha256": (
            "84e12ce1ccff68667e7008d9eb76f080230369bd75c0f4b3b04249cdf2f778e4"
        ),
    }:
        raise ValueError("bounded diagnostic source binding drifted")
    producer_ledger = _mapping(
        failure["producer_gpu_attempt_ledger"], "producer GPU attempt ledger"
    )
    _exact_keys(
        producer_ledger,
        {"canonical_container_path", "size_bytes", "sha256", "mode"},
        "producer GPU attempt ledger",
    )
    if producer_ledger != {
        "canonical_container_path": (
            "/data/experiments/causalcache/restoration-v2-2-policy-vision-"
            "v3-size-dict-interface-repair-attempt.json"
        ),
        "size_bytes": 1030,
        "sha256": (
            "6aabf8dc07dd45d422e8f21ee30538eded2e5b2a5eed2c36675338bdd15891ef"
        ),
        "mode": "0600",
    }:
        raise ValueError("producer GPU attempt-ledger binding drifted")

    projection = _mapping(config["projection_contract"], "projection contract")
    _exact_keys(
        projection,
        {
            "evaluated_state_keys_exact",
            "feature_state_keys_exact_in_output_order",
            "budget_event_capacity",
            "candidate_event_step_ids",
            "decision_step_id",
            "all_non_state_fields_must_use_frozen_v3_reconstruction_unchanged",
        },
        "projection contract",
    )
    if (
        projection.get("evaluated_state_keys_exact")
        != [
            "budget_event_capacity",
            "candidate_event_step_ids",
            "decision_step_id",
            "index",
            "role",
            "state_id",
            "trajectory_id",
        ]
        or projection.get("feature_state_keys_exact_in_output_order")
        != ["index", "role", "trajectory_id", "state_id"]
        or projection.get("budget_event_capacity") != 2
        or projection.get("candidate_event_step_ids") != [1, 2, 3, 4]
        or projection.get("decision_step_id") != 6
        or projection.get(
            "all_non_state_fields_must_use_frozen_v3_reconstruction_unchanged"
        )
        is not True
    ):
        raise ValueError("state projection contract drifted")

    source = _mapping(config["validation_source"], "validation source")
    _exact_keys(
        source,
        {
            "validation_source_git_commit_is_distinct_from_producer_source_git_commit",
            "required_repository_state",
            "source_diff_baseline_git_commit",
            "formal_source_changed_paths_exact",
            "producer_python_paths_must_remain_byte_identical_to_producer_commit",
            "validation_python_source_closure_must_be_recorded",
            "runtime_source_paths_protected_after_source_freeze",
        },
        "validation source",
    )
    changed = source.get("formal_source_changed_paths_exact")
    if (
        source.get(
            "validation_source_git_commit_is_distinct_from_producer_source_git_commit"
        )
        is not True
        or source.get("required_repository_state") != "clean_pushed_main"
        or source.get("source_diff_baseline_git_commit") != ARTIFACT_GIT_COMMIT
        or not isinstance(changed, list)
        or changed != sorted(changed)
        or len(changed) != len(set(changed))
        or source.get(
            "producer_python_paths_must_remain_byte_identical_to_producer_commit"
        )
        is not True
        or source.get("validation_python_source_closure_must_be_recorded")
        is not True
    ):
        raise ValueError("validation source contract drifted")
    protected = source.get("runtime_source_paths_protected_after_source_freeze")
    expected_protected = [
        "code/causalcache/restoration_v2_2_policy_vision_v3_validation_repair.py",
        (
            "code/causalcache/restoration_v2_2_policy_vision_"
            "v3_validation_repair_contract.py"
        ),
        (
            "code/configs/causalcache_restoration_v2_2_policy_vision_"
            "v3_validation_repair_v1.json"
        ),
        (
            "code/scripts/run_restoration_v2_2_policy_vision_"
            "v3_validation_repair_v1.py"
        ),
        (
            "code/scripts/validate_restoration_v2_2_policy_vision_"
            "v3_validation_repair_v1_contract.py"
        ),
    ]
    if protected != expected_protected:
        raise ValueError("validation runtime source protection drifted")

    output = _mapping(config["output_contract"], "output contract")
    _exact_keys(
        output,
        {
            "canonical_result_directory",
            "exact_files",
            "run_status",
            "validate_status",
            "formal_attempt_ledger_path",
            "formal_attempt_ledger_mode",
            "formal_attempt_ledger_creation",
            "formal_completion_seal_path",
            "formal_completion_seal_mode",
            "formal_completion_seal_creation",
            "source_freeze_requires_result_staging_ledgers_and_seal_absent",
            "incomplete_seal_without_published_exact_output_is_permanently_invalid",
            "producer_artifact_must_remain_read_only",
        },
        "output contract",
    )
    if (
        output.get("canonical_result_directory") != CANONICAL_RESULT_DIRECTORY
        or output.get("exact_files") != list(EXPECTED_OUTPUT_FILES)
        or output.get("run_status") != RUN_STATUS
        or output.get("validate_status") != VALIDATE_STATUS
        or output.get("formal_attempt_ledger_path") != FORMAL_ATTEMPT_LEDGER_PATH
        or output.get("formal_attempt_ledger_mode") != "0600"
        or output.get("formal_attempt_ledger_creation")
        != "O_CREAT_O_EXCL_before_label_parse_or_reconstruction"
        or output.get("formal_completion_seal_path")
        != FORMAL_COMPLETION_SEAL_PATH
        or output.get("formal_completion_seal_mode") != "0600"
        or output.get("formal_completion_seal_creation")
        != "O_CREAT_O_EXCL_after_exact_audit_bytes_before_atomic_publish"
        or output.get(
            "source_freeze_requires_result_staging_ledgers_and_seal_absent"
        )
        is not True
        or output.get(
            "incomplete_seal_without_published_exact_output_is_permanently_invalid"
        )
        is not True
        or output.get("producer_artifact_must_remain_read_only") is not True
    ):
        raise ValueError("validation-repair output contract drifted")


@dataclass(frozen=True)
class PolicyVisionV3ValidationRepairContract:
    path: Path
    data: Mapping[str, Any]
    sha256: str
    repository_root: Path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        validate_bound_sources: bool = True,
    ) -> "PolicyVisionV3ValidationRepairContract":
        root = Path(repository_root).resolve()
        resolved = Path(path).resolve()
        canonical = (root / CANONICAL_CONFIG_PATH).resolve()
        if resolved != canonical or not resolved.is_file() or resolved.is_symlink():
            raise ValueError("validation-repair contract must use the canonical config")
        payload = _regular_file_bytes(resolved, "validation-repair config")
        digest = sha256_bytes(payload)
        if digest != FROZEN_CONFIG_SHA256:
            raise ValueError("validation-repair frozen config SHA256 drifted")
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("validation-repair config is not valid JSON") from error
        config = _mapping(value, "validation-repair config")
        _validate_config_semantics(config)
        contract = cls(resolved, config, digest, root)
        if validate_bound_sources:
            contract.validate_bound_sources()
        return contract

    @property
    def artifact_directory(self) -> Path:
        return (
            self.repository_root / self.data["producer_artifact"]["directory"]
        ).resolve()

    @property
    def output_directory(self) -> Path:
        return (self.repository_root / CANONICAL_RESULT_DIRECTORY).resolve()

    def validate_bound_sources(self) -> None:
        root = self.repository_root
        for revision in (PRODUCER_SOURCE_GIT_COMMIT, ARTIFACT_GIT_COMMIT):
            _require_commit(root, revision)
        _require_ancestor(root, PRODUCER_SOURCE_GIT_COMMIT, ARTIFACT_GIT_COMMIT)
        for descendant in (
            _git_text(root, "rev-parse", "HEAD"),
            _git_text(root, "rev-parse", "origin/main"),
        ):
            _require_ancestor(root, ARTIFACT_GIT_COMMIT, descendant)

        producer = self.data["producer"]
        v3_binding = producer["v3_contract"]
        v3_payload = _regular_file_bytes(
            root / v3_binding["path"], "producer v3 contract"
        )
        if (
            len(v3_payload) != v3_binding["size_bytes"]
            or sha256_bytes(v3_payload) != v3_binding["sha256"]
            or v3_payload
            != _git_bytes(root, PRODUCER_SOURCE_GIT_COMMIT, v3_binding["path"])
        ):
            raise ValueError("producer v3 contract bytes drifted")

        artifact_files = validate_exact_artifact_directory(
            self.artifact_directory,
            self.data["producer_artifact"]["exact_files"],
        )
        artifact_relative = self.data["producer_artifact"]["directory"]
        for name, payload in artifact_files.items():
            if payload != _git_bytes(
                root,
                ARTIFACT_GIT_COMMIT,
                f"{artifact_relative}/{name}",
            ):
                raise ValueError(f"artifact Git binding drifted: {name}")

        failure = self.data["failure_evidence"]
        for binding in failure["files"]:
            payload = _validate_file_binding(root, binding, name="failure evidence")
            if payload != _git_bytes(root, ARTIFACT_GIT_COMMIT, binding["path"]):
                raise ValueError("failure-evidence Git binding drifted")
        diagnostic = failure["bounded_diagnostic_source"]
        diagnostic_payload = _validate_file_binding(
            root,
            diagnostic,
            name="bounded diagnostic source",
        )
        if diagnostic_payload != _git_bytes(
            root, ARTIFACT_GIT_COMMIT, diagnostic["path"]
        ):
            raise ValueError("bounded diagnostic Git binding drifted")

        closure = _producer_python_closure(root)
        if {
            key: closure[key]
            for key in ("rule", "path_count", "inventory_sha256")
        } != producer["formal_python_source_closure"]:
            raise ValueError("producer formal Python source closure drifted")

        summary = json.loads(artifact_files["summary.json"])
        if (
            summary.get("status") != PRODUCER_RUN_STATUS
            or summary.get("protocol_id") != PRODUCER_PROTOCOL_ID
            or summary.get("scientific_payload_sha256")
            != self.data["producer_artifact"]["scientific_payload_sha256"]
            or summary.get("source_execution", {}).get("source_git_commit")
            != PRODUCER_SOURCE_GIT_COMMIT
            or summary.get("source_execution", {}).get("contract_sha256")
            != PRODUCER_CONFIG_SHA256
            or summary.get("source_execution", {}).get(
                "formal_python_source_closure"
            )
            != producer["formal_python_source_closure"]
            or summary.get("state_scores", {}).get("record_count") != 15
            or summary.get("state_scores", {}).get("candidate_score_count") != 60
            or summary.get("state_scores", {}).get("sha256")
            != self.data["producer_artifact"]["exact_files"][1]["sha256"]
        ):
            raise ValueError("producer artifact summary identity drifted")
        if (
            summary.get("input_identity", {}).get("restoration_labels")
            != {
                **summary["input_identity"]["restoration_labels"],
                **self.data["immutable_inputs"]["restoration_labels"],
            }
        ):
            raise ValueError("restoration-label binding drifted")
        witness = summary.get("input_identity", {}).get("ocr_rgb_identity_witness")
        expected_witness = self.data["immutable_inputs"]["ocr_rgb_identity_witness"]
        if not isinstance(witness, Mapping) or any(
            witness.get(key) != value for key, value in expected_witness.items()
        ):
            raise ValueError("OCR/RGB witness binding drifted")

    def validate_outputs_absent(
        self,
        *,
        ledger_path: Path | None = None,
        completion_seal_path: Path | None = None,
    ) -> None:
        output = self.output_directory
        staging = output.with_name(f".{output.name}.staging")
        canonical_ledger = Path(FORMAL_ATTEMPT_LEDGER_PATH).resolve()
        canonical_completion_seal = Path(FORMAL_COMPLETION_SEAL_PATH).resolve()
        ledger = canonical_ledger if ledger_path is None else ledger_path.resolve()
        completion_seal = (
            canonical_completion_seal
            if completion_seal_path is None
            else completion_seal_path.resolve()
        )
        if ledger != canonical_ledger:
            raise ValueError("validation-repair attempt ledger path drifted")
        if completion_seal != canonical_completion_seal:
            raise ValueError("validation-repair completion seal path drifted")
        for path, name in (
            (output, "canonical validation-repair output"),
            (staging, "validation-repair staging output"),
            (ledger, "validation-repair attempt ledger"),
            (completion_seal, "validation-repair completion seal"),
        ):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"{name} already exists")

    def validate_formal_source(
        self,
        validation_source_git_commit: str,
        *,
        require_clean_pushed_main: bool,
    ) -> dict[str, Any]:
        root = self.repository_root
        _git_sha(validation_source_git_commit, "validation source Git commit")
        if validation_source_git_commit == PRODUCER_SOURCE_GIT_COMMIT:
            raise ValueError("validation source commit must differ from producer source")
        _require_commit(root, validation_source_git_commit)
        _require_ancestor(root, ARTIFACT_GIT_COMMIT, validation_source_git_commit)
        observed = tuple(
            sorted(
                line
                for line in _git_text(
                    root,
                    "diff",
                    "--name-only",
                    ARTIFACT_GIT_COMMIT,
                    validation_source_git_commit,
                    "--",
                ).splitlines()
                if line
            )
        )
        expected = tuple(
            self.data["validation_source"]["formal_source_changed_paths_exact"]
        )
        if observed != expected:
            raise ValueError("validation-repair formal source changed paths drifted")

        closure = _producer_python_closure(root)
        changed_producer_paths = _git_text(
            root,
            "diff",
            "--name-only",
            PRODUCER_SOURCE_GIT_COMMIT,
            validation_source_git_commit,
            "--",
            *closure["paths"],
        )
        if changed_producer_paths:
            raise ValueError("producer Python source bytes changed after execution")

        validation_closure = _python_source_closure(
            root, validation_source_git_commit
        )

        if require_clean_pushed_main:
            if (
                _git_text(root, "rev-parse", "HEAD")
                != validation_source_git_commit
                or _git_text(root, "rev-parse", "origin/main")
                != validation_source_git_commit
                or _git_text(root, "symbolic-ref", "--short", "HEAD") != "main"
                or _git_text(
                    root,
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                )
            ):
                raise ValueError(
                    "formal validation repair requires clean pushed main"
                )
        return {
            "producer_source_git_commit": PRODUCER_SOURCE_GIT_COMMIT,
            "validation_source_git_commit": validation_source_git_commit,
            "source_diff_baseline_git_commit": ARTIFACT_GIT_COMMIT,
            "formal_source_changed_paths_exact": list(expected),
            "producer_python_source_closure": {
                key: closure[key]
                for key in ("rule", "path_count", "inventory_sha256")
            },
            "validation_python_source_closure": {
                key: validation_closure[key]
                for key in ("rule", "path_count", "inventory_sha256")
            },
        }

    def validate_source_during_staging(
        self,
        validation_source_git_commit: str,
        *,
        staging_directory: Path,
    ) -> dict[str, Any]:
        source = self.validate_formal_source(
            validation_source_git_commit,
            require_clean_pushed_main=False,
        )
        root = self.repository_root
        try:
            staging_relative = staging_directory.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("validation-repair staging must remain in repository") from error
        expected_untracked = {
            f"?? {staging_relative}/{name}" for name in EXPECTED_OUTPUT_FILES
        }
        observed_status = {
            line
            for line in _git_text(
                root,
                "status",
                "--porcelain",
                "--untracked-files=all",
            ).splitlines()
            if line
        }
        if (
            _git_text(root, "rev-parse", "HEAD")
            != validation_source_git_commit
            or _git_text(root, "rev-parse", "origin/main")
            != validation_source_git_commit
            or _git_text(root, "symbolic-ref", "--short", "HEAD") != "main"
            or _git_text(root, "diff", "--name-only", "--")
            or _git_text(root, "diff", "--cached", "--name-only", "--")
            or observed_status != expected_untracked
        ):
            raise ValueError(
                "source changed or staging untracked inventory drifted during publish"
            )
        return source

    def validate_clean_pushed_descendant(
        self,
        validation_source_git_commit: str,
    ) -> dict[str, Any]:
        source = self.validate_formal_source(
            validation_source_git_commit,
            require_clean_pushed_main=False,
        )
        root = self.repository_root
        head = _git_text(root, "rev-parse", "HEAD")
        origin = _git_text(root, "rev-parse", "origin/main")
        if (
            head != origin
            or _git_text(root, "symbolic-ref", "--short", "HEAD") != "main"
            or _git_text(
                root,
                "status",
                "--porcelain",
                "--untracked-files=all",
            )
        ):
            raise ValueError(
                "read-only validation requires a clean pushed main descendant"
            )
        _require_ancestor(root, validation_source_git_commit, head)
        validation_closure = _python_source_closure(
            root, validation_source_git_commit
        )
        if _git_text(
            root,
            "diff",
            "--name-only",
            validation_source_git_commit,
            head,
            "--",
            *validation_closure["paths"],
        ):
            raise ValueError(
                "validation Python closure bytes changed after source freeze"
            )
        head_closure = _python_source_closure(root, head)
        if head_closure != validation_closure:
            raise ValueError(
                "validation Python closure inventory changed after source freeze"
            )
        producer_closure = _producer_python_closure(root)
        if _git_text(
            root,
            "diff",
            "--name-only",
            PRODUCER_SOURCE_GIT_COMMIT,
            head,
            "--",
            *producer_closure["paths"],
        ):
            raise ValueError(
                "producer Python closure bytes changed in validation descendant"
            )
        if source["validation_python_source_closure"] != {
            key: validation_closure[key]
            for key in ("rule", "path_count", "inventory_sha256")
        }:
            raise ValueError("recorded validation Python closure identity drifted")
        return {**source, "validation_descendant_git_commit": head}


def validate_contract(
    path: str | Path,
    *,
    repository_root: str | Path,
    validation_source_git_commit: str | None = None,
    require_clean_pushed_main: bool = False,
    require_outputs_absent: bool = True,
    ledger_path: Path | None = None,
    completion_seal_path: Path | None = None,
) -> dict[str, Any]:
    contract = PolicyVisionV3ValidationRepairContract.load(
        path,
        repository_root=repository_root,
    )
    source = None
    if validation_source_git_commit is not None:
        source = contract.validate_formal_source(
            validation_source_git_commit,
            require_clean_pushed_main=require_clean_pushed_main,
        )
    if require_outputs_absent:
        contract.validate_outputs_absent(
            ledger_path=ledger_path,
            completion_seal_path=completion_seal_path,
        )
    return {
        "status": "PASS_POLICY_VISION_V3_VALIDATION_REPAIR_V1_SOURCE_CONTRACT",
        "protocol_id": PROTOCOL_ID,
        "contract_sha256": contract.sha256,
        "producer_source_git_commit": PRODUCER_SOURCE_GIT_COMMIT,
        "artifact_git_commit": ARTIFACT_GIT_COMMIT,
        "validation_source": source,
        "canonical_result_directory": CANONICAL_RESULT_DIRECTORY,
        "formal_attempt_ledger_path": FORMAL_ATTEMPT_LEDGER_PATH,
        "formal_completion_seal_path": FORMAL_COMPLETION_SEAL_PATH,
        "source_freeze_outputs_absent": require_outputs_absent,
    }
