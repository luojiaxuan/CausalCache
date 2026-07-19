"""Fail-closed Execution-CF v2 contract for the processor image repair."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.set_utility_processor_freeze_contract import (
    CANONICAL_EXECUTION_CONFIG_PATH as V1_CANONICAL_EXECUTION_CONFIG_PATH,
)
from causalcache.set_utility_processor_freeze_contract import (
    FREEZE_B_V2_MANIFEST_SHA256,
    REQUIRED_IDENTITY_ARGUMENTS,
    REQUIRED_PATH_ARGUMENTS,
    REQUIRED_VERSION_ARGUMENTS_BY_PHASE,
    ProcessorFreezeExecutionContract,
)
from causalcache.set_utility_processor_freeze_contract import (
    load_execution_contract as load_v1_execution_contract,
)
from causalcache.set_utility_processor_freeze_contract import (
    validate_runtime_cli_values as validate_v1_runtime_cli_values,
)


SCHEMA_VERSION = "2.0.0"
PROTOCOL_ID = (
    "causalcache_set_utility_processor_freeze_execution_cf_v2_"
    "image_contract_repair"
)
STATUS = (
    "PROCESSOR_ONLY_CANDIDATE_FREEZE_EXECUTION_CF_V2_"
    "IMAGE_CONTRACT_REPAIR_AUTHORIZED"
)
VALIDATION_STATUS = (
    "VALID_SET_UTILITY_PROCESSOR_FREEZE_EXECUTION_CF_V2_"
    "IMAGE_CONTRACT_REPAIR"
)

CANONICAL_EXECUTION_CONFIG_PATH = (
    "code/configs/causalcache_set_utility_processor_freeze_execution_cf_v2_"
    "image_contract_repair.json"
)
CONTRACT_PATH = (
    "code/causalcache/set_utility_processor_freeze_contract_v2.py"
)
IMAGE_CONTRACT_PATH = (
    "code/causalcache/set_utility_processor_image_contract_v2.py"
)
RUNNER_PATH = "code/scripts/run_set_utility_processor_freeze_v2.py"
POSTFLIGHT_PATH = (
    "code/causalcache/set_utility_processor_postflight_v2.py"
)
EXECUTION_VALIDATOR_PATH = (
    "code/scripts/validate_set_utility_processor_freeze_v2_execution.py"
)
OUTPUT_VALIDATOR_PATH = (
    "code/scripts/validate_set_utility_processor_freeze_v2_output.py"
)
OUTPUT_NAMESPACE = (
    "causalcache-set-utility-processor-freeze-v2-image-contract-repair"
)

V1_EXECUTION_CONFIG_SHA256 = (
    "66fd93c64669be734f82830af7d623e9cb97263f90613a5809eb665078b2fba7"
)
V1_FAILURE_SUMMARY_PATH = (
    "data/results/set_utility_processor_freeze_execution_cf_v1_attempt/"
    "summary.json"
)
V1_FAILURE_SUMMARY_SHA256 = (
    "baed4f5fe5eb0d75fc19c9c94b2e9535c5b533e2f567f6641825568620d8addb"
)
CENSUS_V2_SUMMARY_PATH = (
    "data/results/set_utility_selected_image_format_census_v2_"
    "column_projection_repair/summary.json"
)
CENSUS_V2_SUMMARY_SHA256 = (
    "58230e7e9a723b5ee98fa1045d06173792dfc6a787b2914658bef93e5e97d7a8"
)
CENSUS_V2_CARD_PATH = (
    "data/cards/set_utility_selected_image_format_census_v2_"
    "column_projection_repair.md"
)
CENSUS_V2_CARD_SHA256 = (
    "b9e041c1bec6c1c4dbd4a6191555e0587e871ade3a3e1e5b44590584ef35d2c7"
)
CENSUS_V2_HF_REPO = (
    "gavinlaw/causalcache-set-utility-new-development-mobile"
)
CENSUS_V2_HF_TAG = (
    "phase1-b2-image-format-census-v2-column-projection-repair"
)
CENSUS_V2_HF_REVISION = (
    "c1d19eb96d7fa7926f1eb9db3328dbff4e88eae0"
)
CENSUS_V2_INVENTORY_SHA256 = (
    "6c687bec18fd9bcb8f06dfd31c9a1685a6de976eed67b610de18fd3d7933bd4f"
)

PROCESSOR_IMAGE_CONTRACT_V2_ID = (
    "causalcache_set_utility_processor_image_contract_v2_"
    "png_rgb_allowlist_repair"
)
EXPECTED_RGBA_COUNT = 18_768
EXPECTED_RGB_COUNT = 24
EXPECTED_IMAGE_COUNT = 18_792

REQUIRED_REPAIR_EVIDENCE_PATHS = {
    "selected_image_census_v2_card": CENSUS_V2_CARD_PATH,
    "selected_image_census_v2_summary": CENSUS_V2_SUMMARY_PATH,
    "v1_failure_summary": V1_FAILURE_SUMMARY_PATH,
}
REQUIRED_REPAIR_SOURCE_PATHS = (
    CONTRACT_PATH,
    IMAGE_CONTRACT_PATH,
    RUNNER_PATH,
    POSTFLIGHT_PATH,
    EXECUTION_VALIDATOR_PATH,
    OUTPUT_VALIDATOR_PATH,
)

_EXPECTED_EVIDENCE_SHA256 = {
    "selected_image_census_v2_card": CENSUS_V2_CARD_SHA256,
    "selected_image_census_v2_summary": CENSUS_V2_SUMMARY_SHA256,
    "v1_failure_summary": V1_FAILURE_SUMMARY_SHA256,
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
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
        raise ValueError(f"{label} path must be a non-empty string")
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
            raise ValueError(f"{label} path must not traverse a symlink")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} path escaped the repository") from error
    if not path.is_file():
        raise ValueError(f"{label} must bind one existing regular file")
    return path


def _file_binding(root: Path, relative_path: str) -> dict[str, Any]:
    payload = _safe_repository_file(root, relative_path, label=relative_path).read_bytes()
    return {
        "byte_count": len(payload),
        "path": relative_path,
        "sha256": sha256_bytes(payload),
    }


def _validate_file_binding(
    root: Path,
    value: Any,
    *,
    label: str,
    expected_path: str,
    expected_sha256: str | None = None,
) -> bytes:
    binding = _mapping(value, label=label)
    _exact_keys(binding, {"byte_count", "path", "sha256"}, label=label)
    if binding.get("path") != expected_path:
        raise ValueError(f"{label} path drifted")
    byte_count = binding.get("byte_count")
    digest = binding.get("sha256")
    if type(byte_count) is not int or byte_count <= 0:
        raise ValueError(f"{label} byte_count must be a positive integer")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} SHA256 is invalid")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{label} frozen SHA256 drifted")
    payload = _safe_repository_file(root, expected_path, label=label).read_bytes()
    if len(payload) != byte_count or sha256_bytes(payload) != digest:
        raise ValueError(f"{label} byte binding drifted")
    return payload


def _validate_predecessor(
    repository_root: Path,
) -> ProcessorFreezeExecutionContract:
    predecessor = load_v1_execution_contract(
        repository_root=repository_root,
        execution_config_path=(
            repository_root / V1_CANONICAL_EXECUTION_CONFIG_PATH
        ),
    )
    if predecessor.config_sha256 != V1_EXECUTION_CONFIG_SHA256:
        raise ValueError("canonical v1 predecessor execution config drifted")
    return predecessor


def _validate_v1_failure_summary(document: Mapping[str, Any]) -> None:
    identity = _mapping(
        document.get("attempt_identity"), label="v1 failure attempt identity"
    )
    failure = _mapping(document.get("failure"), label="v1 failure")
    observed = _mapping(failure.get("observed"), label="v1 failure observed image")
    expected = _mapping(failure.get("expected"), label="v1 failure expected image")
    boundary = _mapping(
        document.get("scientific_boundary"), label="v1 failure scientific boundary"
    )
    publication = _mapping(
        document.get("publication"), label="v1 failure publication"
    )
    if (
        document.get("schema_version") != "1.0.0"
        or document.get("status")
        != "INVALID_PROCESSOR_FREEZE_EXECUTION_CF_V1_IMAGE_CONTRACT_DRIFT"
        or identity.get("execution_config_sha256")
        != V1_EXECUTION_CONFIG_SHA256
        or identity.get("freeze_b_v2_manifest_sha256")
        != FREEZE_B_V2_MANIFEST_SHA256
    ):
        raise ValueError("v1 failure identity drifted")
    if dict(observed) != {
        "alpha_extrema": None,
        "exif_present": False,
        "source_format": "PNG",
        "source_mode": "RGB",
    } or dict(expected) != {
        "alpha_extrema": [255, 255],
        "exif_present": False,
        "source_format": "PNG",
        "source_mode": "RGBA",
    }:
        raise ValueError("v1 failure image-contract witness drifted")
    zero_keys = (
        "policy_or_vision_forward_count",
        "final_candidate_state_count",
        "restoration_label_count",
        "predictor_training_count",
        "matched_nll_count",
        "closed_loop_episode_count",
        "hugging_face_mutation_count",
        "threshold_or_denominator_change_count",
    )
    if any(boundary.get(key) != 0 for key in zero_keys):
        raise ValueError("v1 failure scientific boundary drifted")
    if (
        publication.get("status")
        != "INVALID_FAILED_PRESERVED_NO_HF_PUBLICATION"
        or publication.get("canonical_revision") is not None
    ):
        raise ValueError("v1 failure publication boundary drifted")


def _validate_census_v2_summary(document: Mapping[str, Any]) -> None:
    artifact = _mapping(document.get("artifact"), label="census v2 artifact")
    counts = _mapping(document.get("counts"), label="census v2 counts")
    histograms = _mapping(
        document.get("histograms"), label="census v2 histograms"
    )
    publication = _mapping(
        document.get("publication"), label="census v2 publication"
    )
    postflight = _mapping(
        document.get("postflight"), label="census v2 postflight"
    )
    if (
        document.get("schema_version") != "2.0.0"
        or document.get("protocol_id")
        != (
            "causalcache_set_utility_selected_image_format_census_v2_"
            "column_projection_repair"
        )
        or document.get("status")
        != (
            "VALID_COMPLETED_SELECTED_IMAGE_FORMAT_CENSUS_V2_"
            "COLUMN_PROJECTION_REPAIR"
        )
    ):
        raise ValueError("census v2 identity drifted")
    if (
        artifact.get("status") != "PUBLISHED_HF_IMMUTABLE_VERIFIED"
        or artifact.get("inventory_sha256") != CENSUS_V2_INVENTORY_SHA256
        or artifact.get("file_count") != 10
    ):
        raise ValueError("census v2 artifact binding drifted")
    if (
        counts.get("selected_observation_count") != EXPECTED_IMAGE_COUNT
        or counts.get("selected_trajectory_count") != 1_200
        or counts.get("model_or_policy_load_count") != 0
        or counts.get("ocr_count") != 0
    ):
        raise ValueError("census v2 denominator or negative count drifted")
    expected_histograms = {
        "alpha_extrema": {
            "255:255": EXPECTED_RGBA_COUNT,
            "null": EXPECTED_RGB_COUNT,
        },
        "exif_present": {"false": EXPECTED_IMAGE_COUNT},
        "format": {"PNG": EXPECTED_IMAGE_COUNT},
        "format_mode": {
            "PNG:RGB": EXPECTED_RGB_COUNT,
            "PNG:RGBA": EXPECTED_RGBA_COUNT,
        },
        "mode": {"RGB": EXPECTED_RGB_COUNT, "RGBA": EXPECTED_RGBA_COUNT},
    }
    for key, expected in expected_histograms.items():
        observed = _mapping(histograms.get(key), label=f"census v2 {key}")
        if dict(observed) != expected:
            raise ValueError(f"census v2 {key} histogram drifted")
    if (
        publication.get("status") != "PUBLISHED_HF_IMMUTABLE_VERIFIED"
        or publication.get("hf_repo") != CENSUS_V2_HF_REPO
        or publication.get("hf_tag") != CENSUS_V2_HF_TAG
        or publication.get("hf_tagged_revision") != CENSUS_V2_HF_REVISION
        or publication.get("fresh_download_verified") is not True
        or publication.get("fresh_download_formal_inventory_sha256")
        != CENSUS_V2_INVENTORY_SHA256
    ):
        raise ValueError("census v2 immutable HF publication drifted")
    if (
        postflight.get("formal_contract_satisfied") is not True
        or postflight.get("scientific_eligibility") is not True
        or postflight.get("status")
        != (
            "VALID_COMPLETED_SELECTED_IMAGE_FORMAT_CENSUS_V2_"
            "COLUMN_PROJECTION_REPAIR"
        )
    ):
        raise ValueError("census v2 postflight validity drifted")


def _validate_census_v2_card(payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("census v2 card must be UTF-8") from error
    required_fragments = (
        CENSUS_V2_HF_REPO,
        CENSUS_V2_HF_TAG,
        CENSUS_V2_HF_REVISION,
        CENSUS_V2_INVENTORY_SHA256,
        "RGBA=18,768",
        "RGB=24",
        "PNG=18,792",
    )
    if any(fragment not in text for fragment in required_fragments):
        raise ValueError("census v2 Git card identity drifted")


def _validate_image_contract_source(payload: bytes) -> None:
    try:
        tree = ast.parse(payload.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError("processor image-contract v2 source must be valid UTF-8 Python") from error
    names = {
        "PROCESSOR_IMAGE_CONTRACT_V2_ID",
        "PROCESSOR_IMAGE_CONTRACT_V2_ACCEPTED_INPUTS",
        "PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS",
        "PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_TOTAL",
    }
    assignments: dict[str, Any] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    assignments[target.id] = ast.literal_eval(statement.value)
                except (ValueError, TypeError) as error:
                    raise ValueError(
                        f"processor image-contract v2 {target.id} must be literal"
                    ) from error
    if assignments != {
        "PROCESSOR_IMAGE_CONTRACT_V2_ACCEPTED_INPUTS": (
            ("PNG", "RGBA", (255, 255), False),
            ("PNG", "RGB", None, False),
        ),
        "PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS": {
            "PNG:RGB": EXPECTED_RGB_COUNT,
            "PNG:RGBA": EXPECTED_RGBA_COUNT,
        },
        "PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_TOTAL": EXPECTED_IMAGE_COUNT,
        "PROCESSOR_IMAGE_CONTRACT_V2_ID": PROCESSOR_IMAGE_CONTRACT_V2_ID,
    }:
        raise ValueError("processor image-contract v2 exported constants drifted")


def _validate_repair_evidence(
    repository_root: Path,
    bindings: Mapping[str, Any],
) -> None:
    _exact_keys(
        bindings,
        set(REQUIRED_REPAIR_EVIDENCE_PATHS),
        label="repair evidence bindings",
    )
    payloads: dict[str, bytes] = {}
    for name, path in REQUIRED_REPAIR_EVIDENCE_PATHS.items():
        payloads[name] = _validate_file_binding(
            repository_root,
            bindings[name],
            label=f"repair evidence {name}",
            expected_path=path,
            expected_sha256=_EXPECTED_EVIDENCE_SHA256[name],
        )
    failure = _strict_json_object(
        payloads["v1_failure_summary"], label="v1 failure summary"
    )
    census = _strict_json_object(
        payloads["selected_image_census_v2_summary"],
        label="selected-image census v2 summary",
    )
    _validate_v1_failure_summary(failure)
    _validate_census_v2_summary(census)
    _validate_census_v2_card(payloads["selected_image_census_v2_card"])


def _expected_image_contract_repair() -> dict[str, Any]:
    return {
        "accepted_source_images": [
            {
                "alpha_extrema": [255, 255],
                "census_count": EXPECTED_RGBA_COUNT,
                "exif_present": False,
                "source_format": "PNG",
                "source_mode": "RGBA",
            },
            {
                "alpha_extrema": None,
                "census_count": EXPECTED_RGB_COUNT,
                "exif_present": False,
                "source_format": "PNG",
                "source_mode": "RGB",
            },
        ],
        "alpha_synthesis_allowed": False,
        "contract_id": PROCESSOR_IMAGE_CONTRACT_V2_ID,
        "decoded_rgb_conversion_required": True,
        "encoded_source_bytes_preserved": True,
        "exif_transpose_allowed": False,
        "expected_selected_image_count": EXPECTED_IMAGE_COUNT,
        "reencode_allowed": False,
        "repair_scope": "accept_census_observed_opaque_png_rgb_alongside_png_rgba",
    }


def _expected_runtime_cli(
    predecessor: ProcessorFreezeExecutionContract,
) -> dict[str, Any]:
    runtime = dict(_mapping(predecessor.data["runtime_cli"], label="v1 runtime_cli"))
    runtime["runner_path"] = RUNNER_PATH
    return runtime


def _expected_output(
    predecessor: ProcessorFreezeExecutionContract,
) -> dict[str, Any]:
    output = dict(_mapping(predecessor.data["output"], label="v1 output"))
    output["output_namespace"] = OUTPUT_NAMESPACE
    return output


def build_execution_config_skeleton(*, repository_root: str | Path) -> dict[str, Any]:
    """Build, but never write, the byte-bound processor Execution-CF v2."""
    root = Path(repository_root).resolve()
    predecessor = _validate_predecessor(root)
    evidence = {
        name: _file_binding(root, path)
        for name, path in REQUIRED_REPAIR_EVIDENCE_PATHS.items()
    }
    _validate_repair_evidence(root, evidence)
    predecessor_binding = _file_binding(root, V1_CANONICAL_EXECUTION_CONFIG_PATH)
    if predecessor_binding["sha256"] != V1_EXECUTION_CONFIG_SHA256:
        raise ValueError("canonical v1 predecessor byte binding drifted")
    repair_sources = [
        _file_binding(root, path) for path in REQUIRED_REPAIR_SOURCE_PATHS
    ]
    _validate_image_contract_source((root / IMAGE_CONTRACT_PATH).read_bytes())
    return {
        "authorization": dict(predecessor.data["authorization"]),
        "bindings": {
            "predecessor_execution_config": predecessor_binding,
            "repair_evidence": evidence,
            "repair_sources": repair_sources,
        },
        "image_contract_repair": _expected_image_contract_repair(),
        "output": _expected_output(predecessor),
        "phases": dict(predecessor.data["phases"]),
        "protocol_id": PROTOCOL_ID,
        "runtime_cli": _expected_runtime_cli(predecessor),
        "schema_version": SCHEMA_VERSION,
        "selection": dict(predecessor.data["selection"]),
        "status": STATUS,
    }


@dataclass(frozen=True)
class ProcessorFreezeExecutionContractV2:
    data: Mapping[str, Any]
    repository_root: Path
    config_sha256: str
    freeze_b_v2_manifest: Mapping[str, Any]
    predecessor_contract: ProcessorFreezeExecutionContract


def validate_execution_config(
    config: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> ProcessorFreezeExecutionContractV2:
    root = Path(repository_root).resolve()
    predecessor = _validate_predecessor(root)
    _exact_keys(
        config,
        {
            "authorization",
            "bindings",
            "image_contract_repair",
            "output",
            "phases",
            "protocol_id",
            "runtime_cli",
            "schema_version",
            "selection",
            "status",
        },
        label="processor-freeze v2 execution config",
    )
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("protocol_id") != PROTOCOL_ID
        or config.get("status") != STATUS
    ):
        raise ValueError("processor-freeze v2 execution identity drifted")

    bindings = _mapping(config.get("bindings"), label="v2 execution bindings")
    _exact_keys(
        bindings,
        {
            "predecessor_execution_config",
            "repair_evidence",
            "repair_sources",
        },
        label="v2 execution bindings",
    )
    _validate_file_binding(
        root,
        bindings["predecessor_execution_config"],
        label="v1 predecessor execution config",
        expected_path=V1_CANONICAL_EXECUTION_CONFIG_PATH,
        expected_sha256=V1_EXECUTION_CONFIG_SHA256,
    )
    evidence = _mapping(bindings["repair_evidence"], label="repair evidence")
    _validate_repair_evidence(root, evidence)

    source_values = _sequence(bindings["repair_sources"], label="repair sources")
    if len(source_values) != len(REQUIRED_REPAIR_SOURCE_PATHS):
        raise ValueError("repair source binding count drifted")
    observed_paths: list[str] = []
    for index, expected_path in enumerate(REQUIRED_REPAIR_SOURCE_PATHS):
        value = source_values[index]
        _validate_file_binding(
            root,
            value,
            label=f"repair source {index}",
            expected_path=expected_path,
        )
        observed_paths.append(str(_mapping(value, label="repair source")["path"]))
    if tuple(observed_paths) != REQUIRED_REPAIR_SOURCE_PATHS:
        raise ValueError("repair source path inventory drifted")
    _validate_image_contract_source((root / IMAGE_CONTRACT_PATH).read_bytes())

    expected_sections = {
        "authorization": dict(predecessor.data["authorization"]),
        "image_contract_repair": _expected_image_contract_repair(),
        "output": _expected_output(predecessor),
        "phases": dict(predecessor.data["phases"]),
        "runtime_cli": _expected_runtime_cli(predecessor),
        "selection": dict(predecessor.data["selection"]),
    }
    for name, expected in expected_sections.items():
        actual = _mapping(config.get(name), label=name)
        if dict(actual) != expected:
            raise ValueError(f"processor-freeze v2 {name} contract drifted")

    return ProcessorFreezeExecutionContractV2(
        data=config,
        repository_root=root,
        config_sha256=sha256_bytes(canonical_pretty_json_bytes(dict(config))),
        freeze_b_v2_manifest=predecessor.freeze_b_v2_manifest,
        predecessor_contract=predecessor,
    )


def load_execution_contract(
    *,
    repository_root: str | Path,
    execution_config_path: str | Path,
) -> ProcessorFreezeExecutionContractV2:
    root = Path(repository_root).resolve()
    supplied = Path(execution_config_path)
    path = supplied if supplied.is_absolute() else root / supplied
    expected = (root / CANONICAL_EXECUTION_CONFIG_PATH).resolve()
    if path.resolve() != expected:
        raise ValueError(
            f"execution config must be the canonical repository path {expected}"
        )
    if path.is_symlink() or not path.is_file():
        raise ValueError("processor-freeze v2 execution config must be a regular file")
    payload = path.read_bytes()
    config = _strict_json_object(payload, label="processor-freeze v2 execution config")
    if payload != canonical_pretty_json_bytes(config):
        raise ValueError(
            "processor-freeze v2 execution config must be canonical pretty JSON"
        )
    contract = validate_execution_config(config, repository_root=root)
    return ProcessorFreezeExecutionContractV2(
        data=contract.data,
        repository_root=contract.repository_root,
        config_sha256=sha256_bytes(payload),
        freeze_b_v2_manifest=contract.freeze_b_v2_manifest,
        predecessor_contract=contract.predecessor_contract,
    )


def validate_runtime_cli_values(
    contract: ProcessorFreezeExecutionContractV2,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate v2 paths, namespace, and the unchanged v1 runtime contract."""
    expected_keys = {
        argument.removeprefix("--").replace("-", "_")
        for argument in (*REQUIRED_PATH_ARGUMENTS, *REQUIRED_IDENTITY_ARGUMENTS)
    }
    expected_keys.update(
        argument.removeprefix("--").replace("-", "_")
        for arguments in REQUIRED_VERSION_ARGUMENTS_BY_PHASE.values()
        for argument in arguments
    )
    if set(values) != expected_keys:
        raise ValueError("processor-freeze v2 runtime CLI fields drifted")
    execution_config = values.get("execution_config")
    if not isinstance(execution_config, (str, Path)) or not Path(
        execution_config
    ).is_absolute():
        raise ValueError("--execution-config must be an explicit absolute path")
    expected_config = (
        contract.repository_root / CANONICAL_EXECUTION_CONFIG_PATH
    ).resolve()
    if Path(execution_config).resolve() != expected_config:
        raise ValueError("--execution-config differs from the canonical v2 config")
    output_root = values.get("output_root")
    if not isinstance(output_root, (str, Path)) or not Path(output_root).is_absolute():
        raise ValueError("--output-root must be an explicit absolute path")
    output_name = Path(output_root).name
    git_revision = values.get("git_revision")
    if not isinstance(git_revision, str) or len(git_revision) != 40:
        raise ValueError("--git-revision must be one full Git revision")
    expected_output_name = f"{OUTPUT_NAMESPACE}-{git_revision[:7]}"
    if output_name != expected_output_name:
        raise ValueError(
            "--output-root must use the frozen v2 output namespace and Git suffix"
        )

    predecessor_values = dict(values)
    predecessor_values["execution_config"] = (
        contract.repository_root / V1_CANONICAL_EXECUTION_CONFIG_PATH
    ).resolve()
    predecessor_result = validate_v1_runtime_cli_values(
        contract.predecessor_contract,
        predecessor_values,
    )
    return {
        "config_sha256": contract.config_sha256,
        "image_contract_id": PROCESSOR_IMAGE_CONTRACT_V2_ID,
        "ocr_and_processor_python_may_differ": predecessor_result[
            "ocr_and_processor_python_may_differ"
        ],
        "ocr_python_executable": predecessor_result["ocr_python_executable"],
        "output_namespace": OUTPUT_NAMESPACE,
        "output_overwrite_allowed": False,
        "output_root": predecessor_result["output_root"],
        "predecessor_config_sha256": V1_EXECUTION_CONFIG_SHA256,
        "processor_python_executable": predecessor_result[
            "processor_python_executable"
        ],
        "required_worker_count": predecessor_result["required_worker_count"],
        "status": "VALID_PROCESSOR_FREEZE_V2_IMAGE_CONTRACT_REPAIR_RUNTIME_CLI",
    }


def validation_summary(
    contract: ProcessorFreezeExecutionContractV2,
) -> dict[str, Any]:
    authorization = _mapping(
        contract.data["authorization"], label="processor-freeze v2 authorization"
    )
    return {
        "census_v2_hf_revision": CENSUS_V2_HF_REVISION,
        "census_v2_inventory_sha256": CENSUS_V2_INVENTORY_SHA256,
        "config_sha256": contract.config_sha256,
        "expected_image_counts": {
            "PNG:RGB": EXPECTED_RGB_COUNT,
            "PNG:RGBA": EXPECTED_RGBA_COUNT,
            "total": EXPECTED_IMAGE_COUNT,
        },
        "forbidden_authorization_count": sum(
            value is False for value in authorization.values()
        ),
        "image_contract_id": PROCESSOR_IMAGE_CONTRACT_V2_ID,
        "output_namespace": OUTPUT_NAMESPACE,
        "policy_or_vision_forward_authorized": False,
        "predecessor_config_sha256": V1_EXECUTION_CONFIG_SHA256,
        "repair_evidence_binding_count": len(REQUIRED_REPAIR_EVIDENCE_PATHS),
        "repair_source_binding_count": len(REQUIRED_REPAIR_SOURCE_PATHS),
        "required_runner_cli": {
            "identity": list(REQUIRED_IDENTITY_ARGUMENTS),
            "paths": list(REQUIRED_PATH_ARGUMENTS),
            "versions_by_phase": {
                key: list(value)
                for key, value in REQUIRED_VERSION_ARGUMENTS_BY_PHASE.items()
            },
        },
        "status": VALIDATION_STATUS,
        "throughput_pilot_authorized": False,
        "v1_failure_summary_sha256": V1_FAILURE_SUMMARY_SHA256,
    }


__all__ = [
    "CANONICAL_EXECUTION_CONFIG_PATH",
    "CENSUS_V2_CARD_PATH",
    "CENSUS_V2_HF_REVISION",
    "CENSUS_V2_INVENTORY_SHA256",
    "CENSUS_V2_SUMMARY_PATH",
    "CONTRACT_PATH",
    "EXECUTION_VALIDATOR_PATH",
    "EXPECTED_IMAGE_COUNT",
    "EXPECTED_RGB_COUNT",
    "EXPECTED_RGBA_COUNT",
    "FREEZE_B_V2_MANIFEST_SHA256",
    "IMAGE_CONTRACT_PATH",
    "OUTPUT_NAMESPACE",
    "OUTPUT_VALIDATOR_PATH",
    "POSTFLIGHT_PATH",
    "PROCESSOR_IMAGE_CONTRACT_V2_ID",
    "ProcessorFreezeExecutionContractV2",
    "PROTOCOL_ID",
    "REQUIRED_IDENTITY_ARGUMENTS",
    "REQUIRED_PATH_ARGUMENTS",
    "REQUIRED_REPAIR_EVIDENCE_PATHS",
    "REQUIRED_REPAIR_SOURCE_PATHS",
    "REQUIRED_VERSION_ARGUMENTS_BY_PHASE",
    "RUNNER_PATH",
    "SCHEMA_VERSION",
    "STATUS",
    "V1_CANONICAL_EXECUTION_CONFIG_PATH",
    "V1_EXECUTION_CONFIG_SHA256",
    "V1_FAILURE_SUMMARY_PATH",
    "build_execution_config_skeleton",
    "canonical_pretty_json_bytes",
    "load_execution_contract",
    "sha256_bytes",
    "validate_execution_config",
    "validate_runtime_cli_values",
    "validation_summary",
]
