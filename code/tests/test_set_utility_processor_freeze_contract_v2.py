from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from causalcache.set_utility_processor_freeze_contract import (
    FREEZE_B_V2_MANIFEST_PATH,
    REQUIRED_FROZEN_INPUT_PATHS as V1_REQUIRED_FROZEN_INPUT_PATHS,
    REQUIRED_REPOSITORY_SOURCE_PATHS as V1_REQUIRED_REPOSITORY_SOURCE_PATHS,
    SNAPSHOT_MANIFEST_PATH,
)
from causalcache.set_utility_processor_freeze_contract_v2 import (
    CANONICAL_EXECUTION_CONFIG_PATH,
    CENSUS_V2_HF_REVISION,
    CENSUS_V2_INVENTORY_SHA256,
    EXPECTED_IMAGE_COUNT,
    EXPECTED_RGB_COUNT,
    EXPECTED_RGBA_COUNT,
    OCR_CONCURRENCY_PER_LOGICAL_WORKER,
    OUTPUT_NAMESPACE,
    PROCESSOR_IMAGE_CONTRACT_V2_ID,
    REQUIRED_REPAIR_EVIDENCE_PATHS,
    REQUIRED_REPAIR_SOURCE_PATHS,
    V1_CANONICAL_EXECUTION_CONFIG_PATH,
    V1_EXECUTION_CONFIG_SHA256,
    build_execution_config_skeleton,
    canonical_pretty_json_bytes,
    load_execution_contract,
    validate_execution_config,
    validate_runtime_cli_values,
    validation_summary,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def contract_root(tmp_path: Path) -> Path:
    required_v1 = {
        FREEZE_B_V2_MANIFEST_PATH,
        V1_CANONICAL_EXECUTION_CONFIG_PATH,
        *V1_REQUIRED_FROZEN_INPUT_PATHS.values(),
        *V1_REQUIRED_REPOSITORY_SOURCE_PATHS,
    }
    for relative in sorted(required_v1):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    for relative in REQUIRED_REPAIR_EVIDENCE_PATHS.values():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    for index, relative in enumerate(REQUIRED_REPAIR_SOURCE_PATHS):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith("set_utility_processor_image_contract_v2.py"):
            shutil.copyfile(ROOT / relative, destination)
        else:
            destination.write_text(
                f"repair-source-{index}:{relative}\n",
                encoding="utf-8",
            )
    return tmp_path


def _materialize_config(root: Path) -> tuple[dict, Path]:
    config = build_execution_config_skeleton(repository_root=root)
    path = root / CANONICAL_EXECUTION_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_pretty_json_bytes(config))
    return config, path


def test_skeleton_wraps_canonical_v1_with_image_and_execution_repair(
    contract_root: Path,
) -> None:
    config = build_execution_config_skeleton(repository_root=contract_root)
    contract = validate_execution_config(config, repository_root=contract_root)
    summary = validation_summary(contract)
    predecessor = contract.predecessor_contract.data

    assert config["bindings"]["predecessor_execution_config"] == {
        "byte_count": (contract_root / V1_CANONICAL_EXECUTION_CONFIG_PATH).stat().st_size,
        "path": V1_CANONICAL_EXECUTION_CONFIG_PATH,
        "sha256": V1_EXECUTION_CONFIG_SHA256,
    }
    assert config["selection"] == predecessor["selection"]
    expected_phases = copy.deepcopy(predecessor["phases"])
    expected_phases["raw_decode_and_ocr"].update(
        {
            "execution_concurrency_scope": (
                "within_each_of_four_logical_artifact_workers"
            ),
            "ocr_concurrency_per_logical_worker": (
                OCR_CONCURRENCY_PER_LOGICAL_WORKER
            ),
            "ordered_bounded_submission_required": True,
            "separate_ocr_engine_per_execution_slot": True,
        }
    )
    assert config["phases"] == expected_phases
    assert config["authorization"] == predecessor["authorization"]
    assert config["runtime_cli"] == {
        **predecessor["runtime_cli"],
        "runner_path": "code/scripts/run_set_utility_processor_freeze_v2.py",
    }
    assert config["output"] == {
        **predecessor["output"],
        "output_namespace": OUTPUT_NAMESPACE,
    }
    assert config["image_contract_repair"] == {
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
    assert summary["census_v2_hf_revision"] == CENSUS_V2_HF_REVISION
    assert summary["census_v2_inventory_sha256"] == CENSUS_V2_INVENTORY_SHA256
    assert summary["expected_image_counts"] == {
        "PNG:RGB": 24,
        "PNG:RGBA": 18_768,
        "total": 18_792,
    }
    assert summary["ocr_concurrency_per_logical_worker"] == 8
    assert summary["policy_or_vision_forward_authorized"] is False


@pytest.mark.parametrize(
    "name",
    sorted(REQUIRED_REPAIR_EVIDENCE_PATHS),
)
def test_evidence_is_fixed_by_path_size_and_sha256(
    contract_root: Path,
    name: str,
) -> None:
    config = build_execution_config_skeleton(repository_root=contract_root)
    evidence = config["bindings"]["repair_evidence"][name]
    assert set(evidence) == {"byte_count", "path", "sha256"}
    (contract_root / evidence["path"]).write_bytes(b"drifted\n")
    with pytest.raises(ValueError, match="byte binding drifted"):
        validate_execution_config(config, repository_root=contract_root)


def test_predecessor_is_live_validated_not_only_hash_named(
    contract_root: Path,
) -> None:
    config = build_execution_config_skeleton(repository_root=contract_root)
    (contract_root / V1_CANONICAL_EXECUTION_CONFIG_PATH).write_bytes(b"{}\n")
    with pytest.raises(ValueError):
        validate_execution_config(config, repository_root=contract_root)


def test_each_new_source_is_byte_bound_in_exact_order(contract_root: Path) -> None:
    config = build_execution_config_skeleton(repository_root=contract_root)
    bindings = config["bindings"]["repair_sources"]
    assert [binding["path"] for binding in bindings] == list(
        REQUIRED_REPAIR_SOURCE_PATHS
    )
    path = contract_root / bindings[-1]["path"]
    path.write_bytes(b"drifted\n")
    with pytest.raises(ValueError, match="byte binding drifted"):
        validate_execution_config(config, repository_root=contract_root)


def test_image_contract_exported_constants_are_semantically_frozen(
    contract_root: Path,
) -> None:
    source = contract_root / (
        "code/causalcache/set_utility_processor_image_contract_v2.py"
    )
    text = source.read_text(encoding="utf-8")
    source.write_text(
        text.replace('"PNG:RGB": 24', '"PNG:RGB": 25'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exported constants drifted"):
        build_execution_config_skeleton(repository_root=contract_root)


@pytest.mark.parametrize(
    "section,mutate,match",
    [
        (
            "authorization",
            lambda value: value.__setitem__("policy_or_vision_forward_allowed", True),
            "authorization contract drifted",
        ),
        (
            "selection",
            lambda value: value.__setitem__("query_state_count", 2_399),
            "selection contract drifted",
        ),
        (
            "phases",
            lambda value: value["auto_processor"].__setitem__(
                "context_limit_tokens", 65_536
            ),
            "phases contract drifted",
        ),
        (
            "image_contract_repair",
            lambda value: value["accepted_source_images"][1].__setitem__(
                "census_count", 25
            ),
            "image_contract_repair contract drifted",
        ),
    ],
)
def test_frozen_science_scope_cannot_be_relaxed(
    contract_root: Path,
    section: str,
    mutate: object,
    match: str,
) -> None:
    config = build_execution_config_skeleton(repository_root=contract_root)
    changed = copy.deepcopy(config)
    mutate(changed[section])
    with pytest.raises(ValueError, match=match):
        validate_execution_config(changed, repository_root=contract_root)


def test_load_requires_new_canonical_path_and_canonical_json(
    contract_root: Path,
) -> None:
    _, config_path = _materialize_config(contract_root)
    contract = load_execution_contract(
        repository_root=contract_root,
        execution_config_path=config_path,
    )
    assert contract.config_sha256
    assert contract.predecessor_contract.config_sha256 == V1_EXECUTION_CONFIG_SHA256

    alternate = contract_root / "alternate.json"
    alternate.write_bytes(config_path.read_bytes())
    with pytest.raises(ValueError, match="canonical repository path"):
        load_execution_contract(
            repository_root=contract_root,
            execution_config_path=alternate,
        )

    config_path.write_text(
        json.dumps(json.loads(config_path.read_text(encoding="utf-8"))),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical pretty JSON"):
        load_execution_contract(
            repository_root=contract_root,
            execution_config_path=config_path,
        )


def _runtime_values(root: Path, config_path: Path, output: Path) -> dict:
    source = root / "external/source"
    model = root / "external/model"
    ocr_model = root / "external/ocr-model"
    ocr_wheels = root / "external/ocr-wheels"
    executable_dir = root / "external/bin"
    for path in (
        source,
        model,
        ocr_model,
        ocr_wheels,
        output.parent,
        executable_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    ocr_python = executable_dir / "ocr-python"
    processor_python = executable_dir / "processor-python"
    for executable in (ocr_python, processor_python):
        executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    return {
        "container_id": "container-123",
        "container_image_digest": "sha256:" + "a" * 64,
        "execution_config": config_path.resolve(),
        "git_revision": "b" * 40,
        "host_alias": "hyper00",
        "host_hostname": "node-radixark-16-0000",
        "model_dir": model.resolve(),
        "ocr_model_dir": ocr_model.resolve(),
        "ocr_python_executable": ocr_python.resolve(),
        "ocr_python_version": "3.10.14",
        "ocr_runtime_version": "rapidocr-1.4.4",
        "ocr_wheel_dir": ocr_wheels.resolve(),
        "output_root": output.resolve(),
        "pillow_version": "11.1.0",
        "processor_python_version": "3.12.3",
        "processor_python_executable": processor_python.resolve(),
        "pyarrow_version": "24.0.0",
        "repository_root": root.resolve(),
        "snapshot_manifest": (root / SNAPSHOT_MANIFEST_PATH).resolve(),
        "source_root": source.resolve(),
        "torch_version": "2.11.0+cu130",
        "transformers_version": "5.6.0",
        "worker_count": 4,
    }


def test_runtime_delegates_v1_checks_and_requires_new_namespace(
    contract_root: Path,
) -> None:
    _, config_path = _materialize_config(contract_root)
    contract = load_execution_contract(
        repository_root=contract_root,
        execution_config_path=config_path,
    )
    output = (
        contract_root.parent
        / "artifact-parent"
        / f"{OUTPUT_NAMESPACE}-{'b' * 7}"
    )
    values = _runtime_values(contract_root, config_path, output)
    result = validate_runtime_cli_values(contract, values)
    assert result["output_namespace"] == OUTPUT_NAMESPACE
    assert result["predecessor_config_sha256"] == V1_EXECUTION_CONFIG_SHA256
    assert result["status"] == (
        "VALID_PROCESSOR_FREEZE_V2_IMAGE_CONTRACT_REPAIR_RUNTIME_CLI"
    )

    wrong = dict(values)
    wrong["output_root"] = output.with_name("causalcache-set-utility-processor-freeze-v2")
    with pytest.raises(ValueError, match="frozen v2 output namespace"):
        validate_runtime_cli_values(contract, wrong)

    arbitrary_suffix = dict(values)
    arbitrary_suffix["output_root"] = output.with_name(f"{OUTPUT_NAMESPACE}-formal")
    with pytest.raises(ValueError, match="Git suffix"):
        validate_runtime_cli_values(contract, arbitrary_suffix)

    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite is forbidden"):
        validate_runtime_cli_values(contract, values)


def test_contract_and_validator_do_not_import_model_frameworks() -> None:
    sources = [
        ROOT / "code/causalcache/set_utility_processor_freeze_contract_v2.py",
        ROOT / "code/scripts/validate_set_utility_processor_freeze_v2_execution.py",
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert "from transformers import" not in text
        assert "import torch" not in text
        assert "AutoModel.from_pretrained" not in text
