from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from causalcache.set_utility_processor_freeze_contract import (
    CANONICAL_EXECUTION_CONFIG_PATH,
    FREEZE_B_V2_MANIFEST_PATH,
    FREEZE_B_V2_MANIFEST_SHA256,
    OCR_BACKEND_COMPLETION_MANIFEST_PATH,
    P0_CENSUS_MANIFEST_PATH,
    REQUIRED_FROZEN_INPUT_PATHS,
    REQUIRED_REPOSITORY_SOURCE_PATHS,
    SNAPSHOT_MANIFEST_PATH,
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
    destination = tmp_path / FREEZE_B_V2_MANIFEST_PATH
    destination.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / FREEZE_B_V2_MANIFEST_PATH, destination)
    for index, relative in enumerate(REQUIRED_REPOSITORY_SOURCE_PATHS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source-{index}:{relative}\n".encode("utf-8"))
    for relative in REQUIRED_FROZEN_INPUT_PATHS.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, path)
    return tmp_path


def _materialize_config(root: Path) -> tuple[dict, Path]:
    config = build_execution_config_skeleton(repository_root=root)
    path = root / CANONICAL_EXECUTION_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_pretty_json_bytes(config))
    return config, path


def test_skeleton_binds_v2_bytes_and_separates_policy_pilot(
    contract_root: Path,
) -> None:
    config = build_execution_config_skeleton(repository_root=contract_root)
    contract = validate_execution_config(config, repository_root=contract_root)
    summary = validation_summary(contract)

    assert config["bindings"]["freeze_b_v2_manifest"]["sha256"] == (
        FREEZE_B_V2_MANIFEST_SHA256
    )
    assert [
        binding["path"] for binding in config["bindings"]["repository_sources"]
    ] == list(REQUIRED_REPOSITORY_SOURCE_PATHS)
    assert {
        name: binding["path"]
        for name, binding in config["bindings"]["frozen_inputs"].items()
    } == REQUIRED_FROZEN_INPUT_PATHS
    assert config["phases"]["auto_processor"]["interface"] == (
        "transformers.AutoProcessor_only"
    )
    assert config["phases"]["throughput_pilot"]["policy_forward_allowed"] is False
    assert config["authorization"]["run_ocr_allowed"] is True
    assert config["authorization"]["apply_auto_processor_allowed"] is True
    for forbidden in (
        "access_outcome_or_utility_allowed",
        "auto_model_load_allowed",
        "closed_loop_allowed",
        "generate_restoration_labels_allowed",
        "hugging_face_mutation_allowed",
        "kl_measurement_allowed",
        "one_shot_evaluation_label_access_allowed",
        "policy_or_vision_forward_allowed",
        "policy_throughput_pilot_allowed",
        "predictor_training_allowed",
        "teacher_forced_action_forward_allowed",
    ):
        assert config["authorization"][forbidden] is False
    assert config["output"]["overwrite_allowed"] is False
    assert summary["status"] == "VALID_SET_UTILITY_PROCESSOR_FREEZE_EXECUTION_CF"
    assert summary["throughput_pilot_authorized"] is False


def test_every_repository_source_is_bound_by_path_size_and_raw_bytes(
    contract_root: Path,
) -> None:
    config = build_execution_config_skeleton(repository_root=contract_root)
    binding = config["bindings"]["repository_sources"][0]
    assert set(binding) == {"byte_count", "path", "sha256"}
    (contract_root / binding["path"]).write_bytes(b"drifted\n")
    with pytest.raises(ValueError, match="byte binding drifted"):
        validate_execution_config(config, repository_root=contract_root)


def test_v2_manifest_is_constant_bound_not_config_selected(
    contract_root: Path,
) -> None:
    config = build_execution_config_skeleton(repository_root=contract_root)
    changed = copy.deepcopy(config)
    changed["bindings"]["freeze_b_v2_manifest"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest binding drifted"):
        validate_execution_config(changed, repository_root=contract_root)


@pytest.mark.parametrize(
    "relative_path",
    [
        P0_CENSUS_MANIFEST_PATH,
        OCR_BACKEND_COMPLETION_MANIFEST_PATH,
        SNAPSHOT_MANIFEST_PATH,
    ],
)
def test_frozen_input_byte_drift_fails_closed(
    contract_root: Path,
    relative_path: str,
) -> None:
    config = build_execution_config_skeleton(repository_root=contract_root)
    (contract_root / relative_path).write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="byte binding drifted"):
        validate_execution_config(config, repository_root=contract_root)


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        (
            "authorization",
            "policy_or_vision_forward_allowed",
            True,
            "authorization contract drifted",
        ),
        (
            "authorization",
            "hugging_face_mutation_allowed",
            True,
            "authorization contract drifted",
        ),
        (
            "output",
            "overwrite_allowed",
            True,
            "output contract drifted",
        ),
        (
            "phases",
            "throughput_pilot",
            {"policy_forward_allowed": True, "status": "INCLUDED"},
            "phases contract drifted",
        ),
    ],
)
def test_forbidden_scope_and_no_overwrite_fail_closed(
    contract_root: Path,
    section: str,
    key: str,
    value: object,
    match: str,
) -> None:
    config = build_execution_config_skeleton(repository_root=contract_root)
    changed = copy.deepcopy(config)
    changed[section][key] = value
    with pytest.raises(ValueError, match=match):
        validate_execution_config(changed, repository_root=contract_root)


def test_load_requires_canonical_path_and_canonical_strict_json(
    contract_root: Path,
) -> None:
    _, config_path = _materialize_config(contract_root)
    contract = load_execution_contract(
        repository_root=contract_root,
        execution_config_path=config_path,
    )
    assert contract.config_sha256

    alternate = contract_root / "alternate.json"
    alternate.write_bytes(config_path.read_bytes())
    with pytest.raises(ValueError, match="canonical repository path"):
        load_execution_contract(
            repository_root=contract_root,
            execution_config_path=alternate,
        )

    parsed = json.loads(config_path.read_text(encoding="utf-8"))
    config_path.write_text(json.dumps(parsed), encoding="utf-8")
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


def test_runtime_paths_and_versions_are_explicit_and_two_phase(
    contract_root: Path,
) -> None:
    _, config_path = _materialize_config(contract_root)
    contract = load_execution_contract(
        repository_root=contract_root,
        execution_config_path=config_path,
    )
    output = contract_root.parent / "artifact-parent" / "candidate-freeze"
    values = _runtime_values(contract_root, config_path, output)
    result = validate_runtime_cli_values(contract, values)
    assert values["ocr_python_version"] != values["processor_python_version"]
    assert result["ocr_and_processor_python_may_differ"] is True
    assert result["output_overwrite_allowed"] is False

    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite is forbidden"):
        validate_runtime_cli_values(contract, values)


@pytest.mark.parametrize(
    "key",
    ["ocr_python_executable", "processor_python_executable"],
)
def test_python_executable_must_be_absolute_regular_and_executable(
    contract_root: Path,
    key: str,
) -> None:
    _, config_path = _materialize_config(contract_root)
    contract = load_execution_contract(
        repository_root=contract_root,
        execution_config_path=config_path,
    )
    values = _runtime_values(
        contract_root,
        config_path,
        contract_root.parent / "artifact-parent" / f"candidate-freeze-{key}",
    )
    executable = Path(values[key])
    executable.chmod(0o644)
    with pytest.raises(ValueError, match="regular executable file"):
        validate_runtime_cli_values(contract, values)

    executable.chmod(0o755)
    values[key] = Path(executable.name)
    with pytest.raises(ValueError, match="explicit absolute path"):
        validate_runtime_cli_values(contract, values)


def test_python_executable_symlink_preserves_venv_entrypoint(contract_root: Path) -> None:
    _, config_path = _materialize_config(contract_root)
    contract = load_execution_contract(
        repository_root=contract_root,
        execution_config_path=config_path,
    )
    values = _runtime_values(
        contract_root,
        config_path,
        contract_root.parent / "artifact-parent" / "candidate-freeze-symlink",
    )
    target = Path(values["ocr_python_executable"])
    symlink = target.with_name("ocr-python-link")
    symlink.symlink_to(target)
    values["ocr_python_executable"] = symlink.absolute()
    result = validate_runtime_cli_values(contract, values)
    assert result["ocr_python_executable"] == str(symlink)


def test_contract_and_validator_do_not_import_model_frameworks() -> None:
    sources = [
        ROOT / "code/causalcache/set_utility_processor_freeze_contract.py",
        ROOT / "code/scripts/validate_set_utility_processor_freeze_execution.py",
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert "from transformers import" not in text
        assert "import torch" not in text
        assert "AutoModel.from_pretrained" not in text
