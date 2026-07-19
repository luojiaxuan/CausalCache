"""Byte-bound contract for the v2 selected-image column-projection repair."""

from __future__ import annotations

import ast
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.set_utility_selected_image_census_v2 import (
    EXPECTED_OBSERVATION_COUNT,
    EXPECTED_TRAJECTORY_COUNT,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    WORKER_COUNT,
    canonical_json_bytes,
    parquet_projection_payload,
    sha256_bytes,
)


STATUS = (
    "CPU_ONLY_READ_ONLY_SELECTED_IMAGE_CENSUS_V2_COLUMN_PROJECTION_REPAIR_"
    "AUTHORIZED"
)
VALIDATION_STATUS = (
    "VALID_SELECTED_IMAGE_FORMAT_CENSUS_V2_COLUMN_PROJECTION_REPAIR_CONTRACT"
)
SOURCE_VALIDATION_STATUS = (
    "VALID_CPU_ONLY_SELECTED_IMAGE_CENSUS_V2_COLUMN_PROJECTION_REPAIR_SOURCE"
)
CANONICAL_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_selected_image_format_census_v2_"
    "column_projection_repair.json"
)
RUNNER_PATH = "code/scripts/run_set_utility_selected_image_census_v2.py"
CORE_PATH = "code/causalcache/set_utility_selected_image_census_v2.py"
CONTRACT_PATH = (
    "code/causalcache/set_utility_selected_image_census_contract_v2.py"
)
FREEZE_B_V2_MANIFEST_PATH = (
    "data/manifests/set_utility_freeze_b_v2_terminal_index_repair.json"
)
FREEZE_B_V2_MANIFEST_SHA256 = (
    "915892ef2e0f1495da4b9e409b3e7a112dc86cda0b06384e8a1b7f8053581d30"
)
P0_CENSUS_MANIFEST_PATH = "data/manifests/set_utility_full_pool_census_v2.json"
FULL_POOL_INVENTORY_MANIFEST_PATH = (
    "data/manifests/set_utility_full_pool_inventory_v1.json"
)
BASE_CONFIG_PATH = "code/configs/independent_reference_gate_v1.json"
INVALID_V1_ATTEMPT_SUMMARY_PATH = (
    "data/results/set_utility_selected_image_format_census_v1_attempt/summary.json"
)
V1_EXECUTION_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_selected_image_format_census_v1.json"
)
V1_CORE_PATH = "code/causalcache/set_utility_selected_image_census.py"
V1_CONTRACT_PATH = "code/causalcache/set_utility_selected_image_census_contract.py"
V1_RUNNER_PATH = "code/scripts/run_set_utility_selected_image_census.py"
INVALID_V1_ATTEMPT_SUMMARY_SHA256 = (
    "a6419bf8d59b632d84b280b83e9f362cd3e86dbdd8c5708385b8a9facc617611"
)
V1_EXECUTION_CONFIG_SHA256 = (
    "c0ecbf59dc6bd77881503e92b0e3eb8c011c2768d0721fa5a74c82c8fe173d10"
)
V1_CORE_SHA256 = "1992727cfff3aa5f12683ea9e36179e9b08e5acca832f1ad759653c1b98c8219"
V1_CONTRACT_SHA256 = (
    "fbd699f00513e81cab7f2c5e42ec4a78164e8c2f217cebb5a26861cafaaa1f82"
)
V1_RUNNER_SHA256 = (
    "cfe11f16a29efc384f1c91d7c7ad8f35e9ed54b74f8103d58c4c62411082666e"
)
HF_REPO = "gavinlaw/causalcache-set-utility-new-development-mobile"
HF_TAG = "phase1-b2-image-format-census-v2-column-projection-repair"

REQUIRED_INPUT_PATHS = {
    "freeze_b_v2_manifest": FREEZE_B_V2_MANIFEST_PATH,
    "full_pool_inventory_manifest": FULL_POOL_INVENTORY_MANIFEST_PATH,
    "independent_reference_gate_base_config": BASE_CONFIG_PATH,
    "invalid_v1_attempt_summary": INVALID_V1_ATTEMPT_SUMMARY_PATH,
    "p0_census_manifest": P0_CENSUS_MANIFEST_PATH,
    "v1_core": V1_CORE_PATH,
    "v1_contract": V1_CONTRACT_PATH,
    "v1_execution_config": V1_EXECUTION_CONFIG_PATH,
    "v1_runner": V1_RUNNER_PATH,
}
REQUIRED_REPOSITORY_SOURCE_PATHS = (
    "code/causalcache/data/guiodyssey.py",
    "code/causalcache/data/guiodyssey_independent.py",
    "code/causalcache/data/guiodyssey_restoration_v2.py",
    "code/causalcache/data/restoration_v2_selection.py",
    "code/causalcache/low_fidelity_v2.py",
    "code/causalcache/policy/gui_owl_v2.py",
    "code/causalcache/policy/gui_owl_v2_1.py",
    "code/causalcache/restoration_v2_contract.py",
    "code/causalcache/restoration_v2_text_backend.py",
    "code/causalcache/schema.py",
    "code/causalcache/set_utility_consumed_ledger.py",
    "code/causalcache/set_utility_consumed_ledger_contract.py",
    "code/causalcache/set_utility_full_pool.py",
    "code/causalcache/set_utility_full_pool_inventory_v1.py",
    "code/causalcache/set_utility_label_producer.py",
    "code/causalcache/set_utility_label_schedule.py",
    "code/causalcache/set_utility_long_pool.py",
    "code/causalcache/set_utility_processor_freeze.py",
    "code/causalcache/set_utility_processor_substrate.py",
    V1_CORE_PATH,
    CORE_PATH,
    CONTRACT_PATH,
    "code/causalcache/set_utility_split_validation.py",
    RUNNER_PATH,
)
REQUIRED_PATH_ARGUMENTS = (
    "--repository-root",
    "--execution-config",
    "--source-root",
    "--output-root",
    "--python-executable",
)
REQUIRED_IDENTITY_ARGUMENTS = (
    "--git-revision",
    "--host-alias",
    "--host-hostname",
    "--container-id",
    "--container-image-digest",
    "--worker-count",
    "--python-version",
    "--pyarrow-version",
    "--pillow-version",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")


def canonical_pretty_json_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value, pretty=True)


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
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


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields drifted")


def _safe_repository_file(root: Path, relative_path: Any, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{label} path must be non-empty text")
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or relative_path != pure.as_posix()
        or relative_path.startswith("./")
    ):
        raise ValueError(f"{label} path must be normalized and repository-relative")
    path = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escaped the repository") from error
    if not path.is_file():
        raise ValueError(f"{label} must bind one regular file")
    return path


def _file_binding(root: Path, relative_path: str) -> dict[str, Any]:
    payload = _safe_repository_file(root, relative_path, label=relative_path).read_bytes()
    return {
        "byte_count": len(payload),
        "path": relative_path,
        "sha256": sha256_bytes(payload),
    }


def _validate_binding(
    root: Path,
    value: Any,
    *,
    label: str,
    expected_path: str,
) -> bytes:
    binding = _mapping(value, label=label)
    _exact_keys(binding, {"byte_count", "path", "sha256"}, label=label)
    if binding.get("path") != expected_path:
        raise ValueError(f"{label} path drifted")
    byte_count = binding.get("byte_count")
    digest = binding.get("sha256")
    if type(byte_count) is not int or byte_count <= 0:
        raise ValueError(f"{label} byte count is invalid")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} SHA256 is invalid")
    payload = _safe_repository_file(root, expected_path, label=label).read_bytes()
    if len(payload) != byte_count or sha256_bytes(payload) != digest:
        raise ValueError(f"{label} byte binding drifted")
    return payload


def _expected_selection() -> dict[str, Any]:
    return {
        "assignment_source": "frozen_freeze_b_v2_terminal_index_repair",
        "observation_scope": "all_unique_observations_of_selected_trajectories",
        "selected_observation_count": EXPECTED_OBSERVATION_COUNT,
        "selected_trajectory_count": EXPECTED_TRAJECTORY_COUNT,
        "unselected_row_semantic_use_allowed": False,
    }


def _expected_runtime() -> dict[str, Any]:
    return {
        "device": "cpu",
        "forbidden_interfaces": [
            "AutoModel",
            "AutoProcessor",
            "OCR",
            "policy_forward",
            "torch",
        ],
        "image_decoder": "PIL_via_prepare_image_bytes",
        "parquet_projection": parquet_projection_payload(),
        "parquet_reader": "pyarrow_projected_images_column_on_selected_rows",
        "required_identity_arguments": list(REQUIRED_IDENTITY_ARGUMENTS),
        "required_path_arguments": list(REQUIRED_PATH_ARGUMENTS),
        "runner_path": RUNNER_PATH,
        "selected_row_access": (
            "project_images_before_materialization_assert_exact_batch_schema_and_row_keys"
        ),
        "worker_count": WORKER_COUNT,
        "worker_schedule": "whole_shard_deterministic_lpt_from_processor_freeze",
    }


def _expected_output() -> dict[str, Any]:
    return {
        "atomic_publish_required": True,
        "existing_output_root_allowed": False,
        "final_manifest_contains_only_counts_histograms_and_digests": True,
        "output_root_must_be_absolute_external_persistent_path": True,
        "overwrite_allowed": False,
        "record_fields": sorted(
            {
                "alpha_extrema",
                "decoded_rgb_sha256",
                "exif_present",
                "format",
                "height",
                "image_sha256",
                "mode",
                "selector_identity_sha256",
                "width",
            }
        ),
        "staging_suffix": ".incomplete",
        "worker_artifact": "canonical_jsonl_plus_projection_bound_receipt",
    }


def _expected_artifact() -> dict[str, Any]:
    return {
        "hf_mutation_during_execution": False,
        "intended_private_hf_repo": HF_REPO,
        "intended_tag": HF_TAG,
        "post_execution_status": "AWAITING_COMMITTED_POSTFLIGHT",
    }


def _expected_authorization() -> dict[str, bool]:
    allowed = {
        "decode_selected_images_with_pil_allowed",
        "read_frozen_repository_inputs_allowed",
        "read_pinned_selected_source_rows_allowed",
        "write_external_pseudonymous_census_artifact_allowed",
    }
    all_keys = allowed | {
        "access_outcome_or_utility_allowed",
        "access_sealed_androidworld_test_allowed",
        "auto_processor_allowed",
        "closed_loop_allowed",
        "gpu_allowed",
        "hugging_face_mutation_allowed",
        "model_or_policy_load_allowed",
        "ocr_allowed",
        "restoration_label_generation_allowed",
        "training_allowed",
        "unselected_row_semantic_use_allowed",
    }
    return {key: key in allowed for key in sorted(all_keys)}


def _validate_original_inputs(
    documents: Mapping[str, Mapping[str, Any]], hashes: Mapping[str, str]
) -> None:
    freeze = documents["freeze_b_v2_manifest"]
    census = documents["p0_census_manifest"]
    inventory = documents["full_pool_inventory_manifest"]
    base = documents["independent_reference_gate_base_config"]
    if (
        hashes["freeze_b_v2_manifest"] != FREEZE_B_V2_MANIFEST_SHA256
        or freeze.get("protocol_id")
        != "causalcache_set_utility_freeze_b_v2_terminal_index_repair"
        or freeze.get("status")
        != "POLICY_BLIND_FREEZE_B_V2_TERMINAL_INDEX_REPAIR_COMPLETED"
    ):
        raise ValueError("Freeze-B v2 input identity drifted")
    summary = _mapping(freeze.get("summary"), label="Freeze-B v2 summary")
    assignments = _sequence(freeze.get("assignments"), label="Freeze-B v2 assignments")
    if (
        summary.get("trajectory_count") != EXPECTED_TRAJECTORY_COUNT
        or len(assignments) != EXPECTED_TRAJECTORY_COUNT
        or _mapping(freeze.get("repair"), label="Freeze-B v2 repair").get(
            "worker_count"
        )
        != WORKER_COUNT
    ):
        raise ValueError("Freeze-B v2 selected denominator drifted")
    observation_count = 0
    transport_identities: set[tuple[Any, Any]] = set()
    selector_ids: set[Any] = set()
    for raw in assignments:
        assignment = _mapping(raw, label="Freeze-B v2 assignment")
        decision_count = assignment.get("decision_count")
        if type(decision_count) is not int or decision_count < 6:
            raise ValueError("Freeze-B v2 assignment decision count drifted")
        observation_count += decision_count + 1
        transport_identities.add(
            (assignment.get("transport_file"), assignment.get("transport_row_index"))
        )
        selector_ids.add(assignment.get("p0_selection_sha256"))
    if (
        observation_count != EXPECTED_OBSERVATION_COUNT
        or len(transport_identities) != EXPECTED_TRAJECTORY_COUNT
        or len(selector_ids) != EXPECTED_TRAJECTORY_COUNT
    ):
        raise ValueError("Freeze-B v2 observation or selector inventory drifted")
    source = _mapping(freeze.get("source"), label="Freeze-B v2 source")
    if (
        source.get("p0_census_manifest_path") != P0_CENSUS_MANIFEST_PATH
        or source.get("p0_census_manifest_sha256") != hashes["p0_census_manifest"]
    ):
        raise ValueError("Freeze-B v2 to P0 census hash chain drifted")
    if (
        census.get("protocol_id") != "causalcache_set_utility_full_pool_census_v2"
        or census.get("status")
        != "POLICY_BLIND_UNCONSUMED_FULL_POOL_CENSUS_COMPLETED"
    ):
        raise ValueError("P0 census input identity drifted")
    census_source = _mapping(census.get("source"), label="P0 census source")
    if (
        census_source.get("source_manifest_sha256")
        != hashes["full_pool_inventory_manifest"]
        or census_source.get("base_config_sha256")
        != hashes["independent_reference_gate_base_config"]
    ):
        raise ValueError("P0 census input hash chain drifted")
    if (
        inventory.get("protocol_id")
        != "causalcache_set_utility_full_pool_inventory_v1"
        or inventory.get("status")
        != "COMPLETE_METADATA_ONLY_FULL_POOL_INVENTORY_V1"
        or base.get("protocol_id") != "independent_reference_gate_v1"
    ):
        raise ValueError("inventory or base config identity drifted")


def _validate_predecessor(
    documents: Mapping[str, Mapping[str, Any]], hashes: Mapping[str, str]
) -> None:
    expected_hashes = {
        "invalid_v1_attempt_summary": INVALID_V1_ATTEMPT_SUMMARY_SHA256,
        "v1_core": V1_CORE_SHA256,
        "v1_contract": V1_CONTRACT_SHA256,
        "v1_execution_config": V1_EXECUTION_CONFIG_SHA256,
        "v1_runner": V1_RUNNER_SHA256,
    }
    for name, expected in expected_hashes.items():
        if hashes[name] != expected:
            raise ValueError(f"frozen predecessor {name} identity drifted")
    predecessor = documents["invalid_v1_attempt_summary"]
    failure = _mapping(predecessor.get("failure"), label="invalid v1 failure")
    artifact = _mapping(predecessor.get("artifact"), label="invalid v1 artifact")
    negative = _mapping(
        predecessor.get("negative_operations"), label="invalid v1 negative operations"
    )
    if (
        predecessor.get("protocol_id")
        != "causalcache_set_utility_selected_image_format_census_v1"
        or predecessor.get("status")
        != "INVALID_SELECTED_IMAGE_FORMAT_CENSUS_V1_COLUMN_PROJECTION_CONTRACT_DRIFT"
        or artifact.get("status") != "INVALID_FAILED_PRESERVED_NO_HF_PUBLICATION"
        or failure.get("successor_allowed")
        != "versioned_column_projection_repair_only"
        or failure.get("formal_contract_satisfied") is not False
        or negative.get("full_row_parquet_materialization_occurred") is not True
        or negative.get("hugging_face_mutation_count") != 0
    ):
        raise ValueError("invalid v1 predecessor contract drifted")
    v1_config = documents["v1_execution_config"]
    if (
        v1_config.get("protocol_id")
        != "causalcache_set_utility_selected_image_format_census_v1"
        or predecessor.get("bindings", {}).get("config_sha256")
        != hashes["v1_execution_config"]
    ):
        raise ValueError("invalid v1 config hash chain drifted")


def build_execution_config_skeleton(*, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    inputs = {name: _file_binding(root, path) for name, path in REQUIRED_INPUT_PATHS.items()}
    documents: dict[str, Mapping[str, Any]] = {}
    for name, path in REQUIRED_INPUT_PATHS.items():
        if name in {"v1_core", "v1_contract", "v1_runner"}:
            continue
        documents[name] = _strict_json((root / path).read_bytes(), label=name)
    hashes = {name: binding["sha256"] for name, binding in inputs.items()}
    _validate_original_inputs(documents, hashes)
    _validate_predecessor(documents, hashes)
    return {
        "artifact": _expected_artifact(),
        "authorization": _expected_authorization(),
        "bindings": {
            "frozen_inputs": inputs,
            "runtime_import_closure": [
                _file_binding(root, path) for path in REQUIRED_REPOSITORY_SOURCE_PATHS
            ],
        },
        "output": _expected_output(),
        "protocol_id": PROTOCOL_ID,
        "runtime": _expected_runtime(),
        "schema_version": SCHEMA_VERSION,
        "selection": _expected_selection(),
        "status": STATUS,
    }


@dataclass(frozen=True)
class SelectedImageCensusContractV2:
    data: Mapping[str, Any]
    repository_root: Path
    config_sha256: str
    frozen_inputs: Mapping[str, Mapping[str, Any]]
    frozen_input_hashes: Mapping[str, str]


def validate_execution_config(
    config: Mapping[str, Any], *, repository_root: str | Path
) -> SelectedImageCensusContractV2:
    root = Path(repository_root).resolve()
    _exact_keys(
        config,
        {
            "artifact",
            "authorization",
            "bindings",
            "output",
            "protocol_id",
            "runtime",
            "schema_version",
            "selection",
            "status",
        },
        label="selected-image census v2 config",
    )
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("protocol_id") != PROTOCOL_ID
        or config.get("status") != STATUS
    ):
        raise ValueError("selected-image census v2 config identity drifted")
    if dict(_mapping(config.get("selection"), label="selection")) != _expected_selection():
        raise ValueError("selection contract drifted")
    if dict(_mapping(config.get("runtime"), label="runtime")) != _expected_runtime():
        raise ValueError("runtime contract drifted")
    if dict(_mapping(config.get("output"), label="output")) != _expected_output():
        raise ValueError("output contract drifted")
    if dict(_mapping(config.get("artifact"), label="artifact")) != _expected_artifact():
        raise ValueError("artifact contract drifted")
    if dict(
        _mapping(config.get("authorization"), label="authorization")
    ) != _expected_authorization():
        raise ValueError("authorization contract drifted")
    bindings = _mapping(config.get("bindings"), label="bindings")
    _exact_keys(bindings, {"frozen_inputs", "runtime_import_closure"}, label="bindings")
    frozen = _mapping(bindings.get("frozen_inputs"), label="frozen inputs")
    if set(frozen) != set(REQUIRED_INPUT_PATHS):
        raise ValueError("frozen input inventory drifted")
    documents: dict[str, Mapping[str, Any]] = {}
    hashes: dict[str, str] = {}
    for name, path in REQUIRED_INPUT_PATHS.items():
        payload = _validate_binding(root, frozen[name], label=name, expected_path=path)
        hashes[name] = sha256_bytes(payload)
        if name not in {"v1_core", "v1_contract", "v1_runner"}:
            documents[name] = _strict_json(payload, label=name)
    _validate_original_inputs(documents, hashes)
    _validate_predecessor(documents, hashes)
    sources = _sequence(
        bindings.get("runtime_import_closure"), label="runtime import closure"
    )
    if len(sources) != len(REQUIRED_REPOSITORY_SOURCE_PATHS):
        raise ValueError("runtime import closure length drifted")
    for index, path in enumerate(REQUIRED_REPOSITORY_SOURCE_PATHS):
        _validate_binding(
            root, sources[index], label=f"runtime source {index}", expected_path=path
        )
    return SelectedImageCensusContractV2(
        data=config,
        repository_root=root,
        config_sha256=sha256_bytes(canonical_pretty_json_bytes(config)),
        frozen_inputs=documents,
        frozen_input_hashes=hashes,
    )


def load_execution_contract(
    *, repository_root: str | Path, execution_config_path: str | Path
) -> SelectedImageCensusContractV2:
    root = Path(repository_root).resolve()
    canonical = (root / CANONICAL_CONFIG_PATH).resolve()
    supplied = Path(execution_config_path).resolve()
    if supplied != canonical:
        raise ValueError("execution config must use the canonical repository path")
    payload = _safe_repository_file(
        root, CANONICAL_CONFIG_PATH, label="execution config"
    ).read_bytes()
    config = _strict_json(payload, label="selected-image census v2 config")
    if canonical_pretty_json_bytes(config) != payload:
        raise ValueError("execution config must be canonical pretty JSON")
    contract = validate_execution_config(config, repository_root=root)
    if contract.config_sha256 != sha256_bytes(payload):
        raise RuntimeError("execution config hash drifted during validation")
    return contract


def _absolute_regular_directory(value: Any, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise ValueError(f"{label} must be one explicit absolute regular directory")
    return path.resolve()


def validate_runtime_cli_values(
    contract: SelectedImageCensusContractV2, values: Mapping[str, Any]
) -> None:
    expected_keys = {
        name.removeprefix("--").replace("-", "_")
        for name in (*REQUIRED_PATH_ARGUMENTS, *REQUIRED_IDENTITY_ARGUMENTS)
    }
    if set(values) != expected_keys:
        raise ValueError("runtime CLI value inventory drifted")
    root = _absolute_regular_directory(values["repository_root"], label="repository root")
    if root != contract.repository_root:
        raise ValueError("runtime repository root differs from contract root")
    config = Path(values["execution_config"])
    if not config.is_absolute() or config.resolve() != (root / CANONICAL_CONFIG_PATH).resolve():
        raise ValueError("runtime execution config path drifted")
    source = _absolute_regular_directory(values["source_root"], label="source root")
    executable = Path(values["python_executable"])
    if (
        not executable.is_absolute()
        or not executable.resolve().is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise ValueError("Python executable must have one absolute executable target")
    output = Path(values["output_root"])
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError("output root must be one nonexistent absolute path")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("output root parent must be one existing regular directory")
    for forbidden in (root, source):
        try:
            output.resolve().relative_to(forbidden)
        except ValueError:
            pass
        else:
            raise ValueError("output root must be outside repository and source roots")
    if values["worker_count"] != WORKER_COUNT:
        raise ValueError("runtime worker count must equal four")
    if (
        not isinstance(values["git_revision"], str)
        or _GIT_REVISION.fullmatch(values["git_revision"]) is None
    ):
        raise ValueError("Git revision must be one full lowercase commit SHA")
    for key in (
        "host_alias",
        "host_hostname",
        "container_id",
        "python_version",
        "pyarrow_version",
        "pillow_version",
    ):
        value = values[key]
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"runtime {key} must be explicit non-empty text")
    digest = values["container_image_digest"]
    if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ValueError("container image digest must be one sha256 digest")


def _iter_batches_projection_is_exact(call: ast.Call) -> bool:
    if call.args or len(call.keywords) != 2:
        return False
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    if set(keywords) != {"batch_size", "columns"}:
        return False
    batch_size = keywords["batch_size"]
    columns = keywords["columns"]
    return (
        isinstance(batch_size, ast.Constant)
        and batch_size.value == 8
        and isinstance(columns, ast.List)
        and len(columns.elts) == 1
        and isinstance(columns.elts[0], ast.Constant)
        and columns.elts[0].value == "images"
    )


def validate_cpu_only_source(*, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    forbidden_modules = {"torch", "transformers", "rapidocr", "onnxruntime"}
    forbidden_semantic_row_fields = {
        "actions",
        "messages",
        "metadata",
        "outcome",
        "reward",
        "terminal_status",
        "tool_calls",
    }
    forbidden_names = {
        "AutoModel",
        "AutoProcessor",
        "build_pilot_manifest",
        "build_selected_pilot",
        "forward",
        "generate",
        "inspect_candidate",
        "load_selected_rows_once",
        "read_table",
    }
    projection_calls: list[ast.Call] = []
    for relative in (RUNNER_PATH, CORE_PATH):
        source = _safe_repository_file(root, relative, label=relative).read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source, filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
                if roots & forbidden_modules:
                    raise ValueError(
                        f"CPU-only source imports forbidden module in {relative}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in forbidden_modules:
                    raise ValueError(
                        f"CPU-only source imports forbidden module in {relative}"
                    )
            elif isinstance(node, ast.Call):
                name = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else node.func.id
                    if isinstance(node.func, ast.Name)
                    else None
                )
                if name in forbidden_names:
                    raise ValueError(
                        f"CPU-only source calls forbidden interface in {relative}"
                    )
                if name == "iter_batches":
                    projection_calls.append(node)
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in forbidden_semantic_row_fields
            ):
                raise ValueError(
                    f"image-only source names forbidden row semantics in {relative}"
                )
    if len(projection_calls) != 1 or not _iter_batches_projection_is_exact(
        projection_calls[0]
    ):
        raise ValueError("v2 runner must use exactly one exact images-only projection")
    return {
        "forbidden_interface_count": 0,
        "model_or_policy_load_count": 0,
        "ocr_count": 0,
        "parquet_projection": parquet_projection_payload(),
        "projection_call_count": 1,
        "protocol_id": PROTOCOL_ID,
        "status": SOURCE_VALIDATION_STATUS,
    }


def validation_summary(contract: SelectedImageCensusContractV2) -> dict[str, Any]:
    return {
        "config_sha256": contract.config_sha256,
        "hf_mutation_authorized": False,
        "intended_hf_repo": HF_REPO,
        "intended_hf_tag": HF_TAG,
        "invalid_v1_attempt_summary_sha256": contract.frozen_input_hashes[
            "invalid_v1_attempt_summary"
        ],
        "model_or_policy_load_authorized": False,
        "ocr_authorized": False,
        "parquet_projection": parquet_projection_payload(),
        "protocol_id": PROTOCOL_ID,
        "selected_observation_count": EXPECTED_OBSERVATION_COUNT,
        "selected_trajectory_count": EXPECTED_TRAJECTORY_COUNT,
        "status": VALIDATION_STATUS,
        "worker_count": WORKER_COUNT,
    }


__all__ = [
    "BASE_CONFIG_PATH",
    "CANONICAL_CONFIG_PATH",
    "CONTRACT_PATH",
    "CORE_PATH",
    "FREEZE_B_V2_MANIFEST_PATH",
    "HF_REPO",
    "HF_TAG",
    "INVALID_V1_ATTEMPT_SUMMARY_PATH",
    "INVALID_V1_ATTEMPT_SUMMARY_SHA256",
    "REQUIRED_IDENTITY_ARGUMENTS",
    "REQUIRED_INPUT_PATHS",
    "REQUIRED_PATH_ARGUMENTS",
    "REQUIRED_REPOSITORY_SOURCE_PATHS",
    "RUNNER_PATH",
    "SelectedImageCensusContractV2",
    "build_execution_config_skeleton",
    "canonical_pretty_json_bytes",
    "load_execution_contract",
    "validate_cpu_only_source",
    "validate_execution_config",
    "validate_runtime_cli_values",
    "validation_summary",
]
