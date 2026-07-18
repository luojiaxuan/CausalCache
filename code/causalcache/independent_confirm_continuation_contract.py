"""Frozen Source-A contract for the independent confirm restoration continuation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "causalcache_independent_confirm20_restoration_continuation_v1"
RUN_CONTRACT_PROTOCOL_ID = (
    "causalcache_independent_confirm20_restoration_continuation_run_v1"
)
CONFIG_PATH = "code/configs/causalcache_independent_confirm_continuation_v1.json"
PARENT_CONFIG_PATH = "code/configs/causalcache_independent_confirm_closed_loop_v1.json"
RUNNER_FREEZE_PATH = (
    "code/configs/causalcache_independent_confirm_continuation_runner_v1.json"
)

PARENT_CONTRACT_SHA256 = (
    "33e5c0f856a9074cd5460f0d6f7cdda2108b120d50b208234253ff3136d27b63"
)
PARENT_SOURCE_A = "e1cc8b3a0b8d06433674fe32652078a47b256f10"
PARENT_EXECUTION_B = "f1e91964096dfa699c9fa930922bc2fe64c17bce"
FAILURE_EVIDENCE_COMMIT = "9b1828d37962ca65783c5a7c98b41b98f420dbd9"
FAILURE_SHA256 = "0aab8cad795ec6aad8fb8fc0e6602e54b85a07342a07fa139cf3ae6e0a044b8d"
FAILURE_SUMMARY_SHA256 = (
    "f3ea04e0976114e5aec0c744db8b4e4f65d97533b887200c68c7ff9e2304e9a0"
)
EXPECTED_BASE_COMMIT = "c66a67c3451cee7e20ed737401dc8b074151574f"
EXPECTED_PAYLOAD_COMMIT = "6d0cd95997186293e01c65276f3c082c11a9f52d"
EXPECTED_PAYLOAD_INVENTORY_SHA256 = (
    "861fa54d2b46509a62b3cfb7d376c9ab3298c18656ae2c4e80418f86593fa779"
)
REQUIRED_SOURCE_A_PATHS = (
    "code/configs/causalcache_independent_confirm_continuation_v1.json",
    "code/configs/causalcache_independent_confirm_closed_loop_v1.json",
    "code/causalcache/independent_confirm_continuation_contract.py",
    "code/causalcache/independent_confirm_continuation.py",
    "code/causalcache/independent_confirm_artifact.py",
    "code/causalcache/independent_confirm_runner.py",
    "code/causalcache/independent_confirm_execution.py",
    "code/causalcache/independent_confirm_data.py",
    "code/causalcache/independent_confirm_evaluation.py",
    "code/scripts/validate_independent_confirm_continuation.py",
    "code/scripts/manage_independent_confirm_continuation.py",
    "code/scripts/run_independent_confirm_continuation.py",
    "code/scripts/run_independent_confirm.py",
    "code/scripts/smoke_independent_confirm_gpu_topology.py",
    "code/tests/test_smoke_independent_confirm_gpu_topology.py",
    "code/tests/test_independent_confirm_continuation_contract.py",
    "code/tests/test_independent_confirm_continuation_adoption.py",
    "code/tests/test_independent_confirm_continuation_runner.py",
    "code/tests/test_independent_confirm_continuation_cuda.py",
    "code/tests/test_independent_confirm_continuation.py",
    "code/tests/test_run_independent_confirm_continuation.py",
    "code/tests/test_independent_confirm_artifact.py",
    "data/results/independent_confirm20_v1_attempt/failure.json",
    "data/results/independent_confirm20_v1_attempt/summary.json",
    "docs/independent_confirm_continuation_v1.md",
    "docs/independent_confirm_closed_loop_v1.md",
    "docs/execution.md",
    "README.md",
    "data/README.md",
    "docs/progress.md",
)
REQUIRED_EXECUTION_MODULES = (
    "causalcache.independent_confirm_continuation_contract",
    "causalcache.independent_confirm_continuation",
    "causalcache.independent_confirm_artifact",
    "causalcache.independent_confirm_runner",
    "causalcache.independent_confirm_execution",
    "causalcache.independent_confirm_data",
    "causalcache.independent_confirm_evaluation",
    "scripts.run_independent_confirm_continuation",
    "scripts.run_independent_confirm",
    "scripts.smoke_independent_confirm_gpu_topology",
    "scripts.manage_independent_confirm_continuation",
    "scripts.validate_independent_confirm_continuation",
)

_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def continuation_topology_nonce(*, source_a_commit: str, contract_sha256: str) -> str:
    if (
        _COMMIT.fullmatch(source_a_commit) is None
        or _SHA256.fullmatch(contract_sha256) is None
    ):
        raise ValueError("continuation topology nonce inputs are malformed")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "contract_sha256": contract_sha256,
                "purpose": "fresh-continuation-topology-smoke",
                "source_a_commit": source_a_commit,
            }
        )
    ).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite JSON: {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} drifted")


def _require_exact_json(value: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(value, Mapping) or set(value) != set(expected):
            raise ValueError(f"{label} drifted")
        for key, expected_item in expected.items():
            _require_exact_json(value[key], expected_item, label=label)
        return
    if isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise ValueError(f"{label} drifted")
        for item, expected_item in zip(value, expected, strict=True):
            _require_exact_json(item, expected_item, label=label)
        return
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{label} drifted")


def _safe_relative(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _validate_payload_targets(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError("continuation payload target denominator drifted")
    rows = []
    for raw in value:
        row = _require_mapping(raw, label="payload target")
        if (
            set(row) != {"path", "sha256", "size_bytes"}
            or _SHA256.fullmatch(str(row.get("sha256"))) is None
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] <= 0
        ):
            raise ValueError("continuation payload target schema drifted")
        rows.append(
            MappingProxyType(
                {
                    "path": _safe_relative(row.get("path"), label="payload target"),
                    "sha256": row["sha256"],
                    "size_bytes": row["size_bytes"],
                }
            )
        )
    if [row["path"] for row in rows] != sorted(row["path"] for row in rows):
        raise ValueError("continuation payload target order must be lexical")
    if len({row["path"] for row in rows}) != 8:
        raise ValueError("continuation payload target paths are duplicated")
    return tuple(rows)


@dataclass(frozen=True)
class IndependentConfirmContinuationContract:
    path: Path
    data: Mapping[str, Any]
    sha256: str

    @property
    def source(self) -> Mapping[str, Any]:
        return _require_mapping(self.data["source_freeze"], label="source freeze")

    @property
    def payload(self) -> Mapping[str, Any]:
        return _require_mapping(self.data["sealed_payload"], label="sealed payload")

    @property
    def runtime(self) -> Mapping[str, Any]:
        return _require_mapping(self.data["runtime_contract"], label="runtime contract")

    @property
    def continuation(self) -> Mapping[str, Any]:
        return _require_mapping(
            self.data["continuation_contract"], label="continuation contract"
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path,
        require_runner_absent: bool,
    ) -> "IndependentConfirmContinuationContract":
        root = Path(repository_root).resolve()
        supplied = Path(path)
        expected_path = root / CONFIG_PATH
        resolved = supplied.resolve()
        expected = expected_path.resolve()
        if (
            resolved != expected
            or not resolved.is_file()
            or supplied.is_symlink()
            or expected_path.is_symlink()
        ):
            raise ValueError("continuation contract path is not canonical")
        payload = resolved.read_bytes()
        data = _strict_json_object(payload, label="continuation contract")
        validate_contract(data, repository_root=root)
        runner = root / RUNNER_FREEZE_PATH
        if require_runner_absent and (runner.exists() or runner.is_symlink()):
            raise ValueError("continuation Execution-B freeze already exists in Source-A")
        return cls(
            path=resolved,
            data=MappingProxyType(json.loads(canonical_json_bytes(data))),
            sha256=hashlib.sha256(payload).hexdigest(),
        )


def validate_contract(data: Mapping[str, Any], *, repository_root: Path) -> None:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "scientific_contract",
        "failed_attempt",
        "sealed_payload",
        "continuation_contract",
        "runtime_contract",
        "output_contract",
        "source_freeze",
        "authorization",
    }
    if set(data) != expected_keys:
        raise ValueError("continuation contract top-level schema drifted")
    _require_exact(data.get("schema_version"), SCHEMA_VERSION, label="schema version")
    _require_exact(data.get("protocol_id"), PROTOCOL_ID, label="protocol id")
    _require_exact(
        data.get("status"),
        "source_only_frozen_before_continuation_restoration_access",
        label="source-only status",
    )

    science = _require_mapping(data["scientific_contract"], label="science")
    expected_science = {
        "parent_contract_path": PARENT_CONFIG_PATH,
        "parent_contract_sha256": PARENT_CONTRACT_SHA256,
        "source_a_commit": PARENT_SOURCE_A,
        "execution_b_commit": PARENT_EXECUTION_B,
        "runner_freeze_sha256": "e03d6ae0ee760841eb8ff405f2fef968fc17711467eefca087c4014a3b4f7263",
        "threshold_data_model_seed_comparator_or_go_change_allowed": False,
        "confirm_geometry_or_denominator_change_allowed": False,
        "old_attempt_reclassification_allowed": False,
    }
    _require_exact_json(
        science, expected_science, label="parent scientific contract"
    )
    parent_bytes = (repository_root / PARENT_CONFIG_PATH).read_bytes()
    if hashlib.sha256(parent_bytes).hexdigest() != PARENT_CONTRACT_SHA256:
        raise ValueError("parent scientific contract bytes drifted")

    failed = _require_mapping(data["failed_attempt"], label="failed attempt")
    required_failed = {
        "evidence_commit": FAILURE_EVIDENCE_COMMIT,
        "failure_path": "data/results/independent_confirm20_v1_attempt/failure.json",
        "failure_sha256": FAILURE_SHA256,
        "summary_path": "data/results/independent_confirm20_v1_attempt/summary.json",
        "summary_sha256": FAILURE_SUMMARY_SHA256,
        "topology_receipt_sha256": "8e06034e7b485a00fe288dc824eabd538cea02dee876027d54ac8b2432e22f13",
        "status": "INVALID_INDEPENDENT_CONFIRM20_V1_CUDA_FORK_BOUNDARY",
        "payload_locally_sealed": True,
        "reference_policy_output_count": 0,
        "restoration_output_count": 0,
        "exact_subset_oracle_count": 0,
        "report_count": 0,
        "retry_count": 0,
        "top_up_count": 0,
        "filter_count": 0,
        "same_identity_retry_or_resume_allowed": False,
    }
    _require_exact_json(failed, required_failed, label="failed attempt binding")
    for path_key, sha_key in (("failure_path", "failure_sha256"), ("summary_path", "summary_sha256")):
        observed = (repository_root / failed[path_key]).read_bytes()
        if hashlib.sha256(observed).hexdigest() != failed[sha_key]:
            raise ValueError("retained v1 failure evidence bytes drifted")

    sealed = _require_mapping(data["sealed_payload"], label="sealed payload")
    targets = _validate_payload_targets(sealed.get("targets"))
    expected_payload_scalars = {
        "repo": "gavinlaw/causalcache-independent-confirm20-mobile",
        "repo_type": "dataset",
        "private": True,
        "base_commit": EXPECTED_BASE_COMMIT,
        "payload_commit": EXPECTED_PAYLOAD_COMMIT,
        "main_must_equal_payload_commit_before_continuation": True,
        "base_commit_title": "initial commit",
        "payload_commit_title": "independent confirm-20 label-blind payload",
        "payload_inventory_sha256": EXPECTED_PAYLOAD_INVENTORY_SHA256,
        "manifest_seven_data_file_inventory_sha256": "c385359d44e546fb18feaa62c63c64038329d1d2c4cd0a17ae836c7844143c50",
        "remote_file_count_before_continuation": 9,
        "report_target_count_before_continuation": 0,
        "tag": "independent-confirm20-v1",
        "tag_count_before_continuation": 0,
        "fresh_immutable_byte_replay_required": True,
        "payload_republish_or_mutation_allowed": False,
        "selector_checkpoint_load_allowed": False,
        "independent_scorer_call_allowed": False,
        "policy_vision_call_allowed": False,
    }
    observed_payload_scalars = {key: sealed.get(key) for key in expected_payload_scalars}
    _require_exact_json(
        observed_payload_scalars, expected_payload_scalars, label="sealed payload identity"
    )
    if set(sealed) != set(expected_payload_scalars) | {"targets"}:
        raise ValueError("sealed payload contract keys drifted")
    if len(targets) != 8:
        raise AssertionError("payload target validator lost its fixed denominator")

    continuation = _require_mapping(
        data["continuation_contract"], label="continuation contract"
    )
    fixed_continuation = {
        "run_contract_protocol_id": RUN_CONTRACT_PROTOCOL_ID,
        "scope": [
            "adopt_existing_payload",
            "validate_parent_and_payload_replay",
            "isolated_model_snapshot_verification",
            "restoration",
            "fixed_evaluation",
            "report_direct_child_publication",
            "annotated_tag",
            "immutable_replay",
        ],
        "runtime_continuation_count": 1,
        "scientific_retry_count": 0,
        "top_up_count": 0,
        "filter_count": 0,
        "parent_model_verification_start_method": "spawn",
        "restoration_worker_start_method": "fork",
        "fresh_exec_parent_required": True,
        "pre_fork_cuda_tripwire_required": True,
        "cuda_tripwire_entry_points": [
            "torch.cuda.is_available",
            "torch.cuda.device_count",
            "torch.cuda._lazy_init",
        ],
        "parent_cuda_must_be_uninitialized_immediately_before_fork": True,
        "parent_bad_fork_flag_must_be_false": True,
        "cuda_guard_forbids_is_available_or_device_count": True,
        "restoration_worker_count": 4,
        "states_per_worker": 5,
        "restoration_phase_timeout_seconds": 43200,
        "worker_termination_grace_seconds": 30,
        "model_verification_timeout_seconds": 7200,
        "existing_topology_smoke_must_be_rerun_for_same_allocation": True,
        "fresh_topology_envelope_binds_execution_b_and_nonce": True,
        "fresh_topology_worker_challenge_required": True,
        "old_topology_receipt_sha256_denied": True,
        "durable_attempt_precedes_topology_hf_model_and_restoration": True,
        "partial_report_publication_reconciliation_required": True,
        "report_parent_must_equal_existing_payload_commit": True,
        "old_failure_and_payload_must_remain_unchanged": True,
        "closed_loop_requires_valid_confirm_go": True,
    }
    _require_exact_json(continuation, fixed_continuation, label="continuation scope")

    runtime = _require_mapping(data["runtime_contract"], label="runtime")
    expected_runtime = {
        "host_aliases": ["hyper00", "hyper01"],
        "gpu_count": 4,
        "container_image_id": (
            "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
        ),
        "python_version": "3.12.3",
        "torch_version": "2.11.0+cu130",
        "transformers_version": "5.6.0",
        "dtype": "bfloat16",
        "thread_environment": {
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    }
    _require_exact_json(runtime, expected_runtime, label="runtime contract")

    output = _require_mapping(data["output_contract"], label="output contract")
    expected_output = {
        "report_targets_and_tag_identical_to_parent": True,
        "git_lightweight_result_directory": (
            "data/results/independent_confirm20_continuation_v1"
        ),
        "local_execution_namespace": (
            "independent-confirm20-restoration-continuation/v1"
        ),
        "completion_status": (
            "COMPLETED_AND_PUBLISHED_INDEPENDENT_CONFIRM20_CONTINUATION_V1"
        ),
        "failure_status": "INVALID_INDEPENDENT_CONFIRM20_CONTINUATION_V1",
    }
    _require_exact_json(output, expected_output, label="output contract")

    source = _require_mapping(data["source_freeze"], label="source freeze")
    paths = source.get("required_source_a_paths")
    modules = source.get("required_execution_modules")
    runner = _require_mapping(
        source.get("execution_b_runner_freeze"), label="runner freeze"
    )
    if (
        set(source)
        != {
            "branch",
            "origin_url",
            "required_source_a_paths",
            "required_execution_modules",
            "execution_b_runner_freeze",
        }
        or paths != list(REQUIRED_SOURCE_A_PATHS)
        or modules != list(REQUIRED_EXECUTION_MODULES)
        or source.get("branch") != "main"
        or source.get("origin_url") != "https://github.com/luojiaxuan/CausalCache.git"
        or not isinstance(paths, list)
        or not paths
        or paths != list(dict.fromkeys(paths))
        or any(_safe_relative(path, label="Source-A path") != path for path in paths)
        or not isinstance(modules, list)
        or not modules
        or modules != list(dict.fromkeys(modules))
        or runner
        != {
            "path": RUNNER_FREEZE_PATH,
            "must_be_absent_during_source_only_validation": True,
            "only_allowed_execution_b_source_tree_diff": True,
            "direct_single_parent_child_of_source_a": True,
            "bind_source_inventory": True,
            "bind_loaded_module_inventory": True,
        }
    ):
        raise ValueError("continuation Source-A freeze contract drifted")
    if CONFIG_PATH not in paths or RUNNER_FREEZE_PATH in paths:
        raise ValueError("continuation Source-A path inventory is malformed")
    missing = [
        path
        for path in paths
        if not (repository_root / path).is_file()
        or (repository_root / path).is_symlink()
    ]
    if missing:
        raise ValueError(f"continuation Source-A paths are missing: {missing}")

    authorization = _require_mapping(data["authorization"], label="authorization")
    expected_authorization = {
        "source_a_validated": False,
        "execution_b_frozen": False,
        "payload_adoption_authorized": False,
        "restoration_access_authorized": False,
        "closed_loop_train_authorized": False,
        "sealed_test_authorized": False,
        "matched_nll_authorized": False,
    }
    _require_exact_json(
        authorization,
        expected_authorization,
        label="Source-A authorization contract",
    )


def source_inventory(
    contract: IndependentConfirmContinuationContract,
) -> tuple[Mapping[str, Any], ...]:
    root = contract.path.parents[2]
    records = []
    for relative in contract.source["required_source_a_paths"]:
        payload = (root / relative).read_bytes()
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return tuple(records)


def load_runner_freeze_file(path: str | Path) -> Mapping[str, Any]:
    return _strict_json_object(Path(path).read_bytes(), label="continuation runner freeze")


def validate_runner_freeze(
    contract: IndependentConfirmContinuationContract,
    value: Mapping[str, Any],
    *,
    source_a_commit: str,
) -> None:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "source_a_commit",
        "source_a_parent_count",
        "source_a_remote_main",
        "contract_sha256",
        "source_a_inventory",
        "source_a_inventory_sha256",
        "loaded_module_inventory",
        "loaded_module_inventory_sha256",
        "fresh_topology_receipt_required",
        "topology_receipt_nonce",
        "adopt_existing_payload_authorized",
        "restoration_access_requires_adopted_payload_receipt",
        "selector_reexecution_authorized",
        "closed_loop_requires_confirm_go",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status")
        != "FROZEN_CAUSALCACHE_INDEPENDENT_CONFIRM_CONTINUATION_EXECUTION_B_V1"
        or value.get("source_a_commit") != source_a_commit
        or _COMMIT.fullmatch(source_a_commit) is None
        or type(value.get("source_a_parent_count")) is not int
        or value.get("source_a_parent_count") != 1
        or value.get("source_a_remote_main") != source_a_commit
        or value.get("contract_sha256") != contract.sha256
        or value.get("fresh_topology_receipt_required") is not True
        or value.get("topology_receipt_nonce")
        != continuation_topology_nonce(
            source_a_commit=source_a_commit,
            contract_sha256=contract.sha256,
        )
        or value.get("adopt_existing_payload_authorized") is not True
        or value.get("restoration_access_requires_adopted_payload_receipt") is not True
        or value.get("selector_reexecution_authorized") is not False
        or value.get("closed_loop_requires_confirm_go") is not True
    ):
        raise ValueError("continuation runner freeze schema or authorization drifted")
    for inventory_key, digest_key in (
        ("source_a_inventory", "source_a_inventory_sha256"),
        ("loaded_module_inventory", "loaded_module_inventory_sha256"),
    ):
        inventory = value.get(inventory_key)
        digest = value.get(digest_key)
        if (
            not isinstance(inventory, list)
            or not inventory
            or type(digest) is not str
            or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError(f"continuation {inventory_key} binding drifted")
        normalized = []
        for item in inventory:
            if (
                not isinstance(item, Mapping)
                or set(item) != {"path", "sha256", "size_bytes"}
                or type(item.get("path")) is not str
                or type(item.get("sha256")) is not str
                or _SHA256.fullmatch(item["sha256"]) is None
                or type(item.get("size_bytes")) is not int
                or item["size_bytes"] <= 0
            ):
                raise ValueError(
                    f"continuation {inventory_key} item schema drifted"
                )
            normalized.append(
                {
                    "path": _safe_relative(
                        item["path"], label=f"continuation {inventory_key} item"
                    ),
                    "sha256": item["sha256"],
                    "size_bytes": item["size_bytes"],
                }
            )
        paths = [item["path"] for item in normalized]
        if inventory_key == "source_a_inventory":
            expected_paths = list(contract.source["required_source_a_paths"])
        else:
            expected_paths = sorted(
                f"code/{name.replace('.', '/')}.py"
                for name in contract.source["required_execution_modules"]
            )
        if paths != expected_paths or len(paths) != len(set(paths)):
            raise ValueError(f"continuation {inventory_key} order drifted")
        if (
            inventory != normalized
            or hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()
            != digest
        ):
            raise ValueError(f"continuation {inventory_key} binding drifted")


__all__ = [
    "CONFIG_PATH",
    "EXPECTED_BASE_COMMIT",
    "EXPECTED_PAYLOAD_COMMIT",
    "EXPECTED_PAYLOAD_INVENTORY_SHA256",
    "IndependentConfirmContinuationContract",
    "PARENT_CONFIG_PATH",
    "PROTOCOL_ID",
    "RUNNER_FREEZE_PATH",
    "RUN_CONTRACT_PROTOCOL_ID",
    "SCHEMA_VERSION",
    "canonical_json_bytes",
    "continuation_topology_nonce",
    "load_runner_freeze_file",
    "source_inventory",
    "validate_contract",
    "validate_runner_freeze",
]
