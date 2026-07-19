from __future__ import annotations

import ast
import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from causalcache.set_utility_selected_image_census_contract import (
    CANONICAL_CONFIG_PATH,
    HF_REPO,
    HF_TAG,
    REQUIRED_IDENTITY_ARGUMENTS,
    REQUIRED_INPUT_PATHS,
    REQUIRED_PATH_ARGUMENTS,
    REQUIRED_REPOSITORY_SOURCE_PATHS,
    RUNNER_PATH,
    build_execution_config_skeleton,
    canonical_pretty_json_bytes,
    load_execution_contract,
    validate_cpu_only_source,
    validate_execution_config,
    validate_runtime_cli_values,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def contract_root(tmp_path: Path) -> Path:
    for relative in REQUIRED_INPUT_PATHS.values():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    for index, relative in enumerate(REQUIRED_REPOSITORY_SOURCE_PATHS):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"source-{index}:{relative}\n", encoding="utf-8")
    return tmp_path


def _materialize(root: Path) -> tuple[dict, Path]:
    config = build_execution_config_skeleton(repository_root=root)
    path = root / CANONICAL_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_pretty_json_bytes(config))
    return config, path


def test_config_binds_inputs_import_closure_and_cpu_only_scope(
    contract_root: Path,
) -> None:
    config = build_execution_config_skeleton(repository_root=contract_root)
    contract = validate_execution_config(config, repository_root=contract_root)
    assert contract.config_sha256
    assert config["selection"]["selected_trajectory_count"] == 1_200
    assert config["selection"]["selected_observation_count"] == 18_792
    assert config["runtime"]["worker_count"] == 4
    assert config["runtime"]["selected_row_access"] == (
        "images_column_only_no_message_metadata_action_or_outcome_decode"
    )
    assert [
        binding["path"] for binding in config["bindings"]["runtime_import_closure"]
    ] == list(REQUIRED_REPOSITORY_SOURCE_PATHS)
    assert config["artifact"] == {
        "hf_mutation_during_execution": False,
        "intended_private_hf_repo": HF_REPO,
        "intended_tag": HF_TAG,
        "post_execution_status": "PENDING_HF_UPLOAD",
    }
    for key in (
        "auto_processor_allowed",
        "gpu_allowed",
        "hugging_face_mutation_allowed",
        "model_or_policy_load_allowed",
        "ocr_allowed",
        "restoration_label_generation_allowed",
        "training_allowed",
    ):
        assert config["authorization"][key] is False


def test_bound_input_and_source_byte_drift_fail_closed(contract_root: Path) -> None:
    config = build_execution_config_skeleton(repository_root=contract_root)
    (contract_root / REQUIRED_INPUT_PATHS["p0_census_manifest"]).write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="byte binding drifted"):
        validate_execution_config(config, repository_root=contract_root)

    contract_root = contract_root.parent / "second"
    contract_root.mkdir()
    for relative in REQUIRED_INPUT_PATHS.values():
        destination = contract_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    for index, relative in enumerate(REQUIRED_REPOSITORY_SOURCE_PATHS):
        destination = contract_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"source-{index}:{relative}\n", encoding="utf-8")
    config = build_execution_config_skeleton(repository_root=contract_root)
    (contract_root / REQUIRED_REPOSITORY_SOURCE_PATHS[0]).write_bytes(b"drift\n")
    with pytest.raises(ValueError, match="byte binding drifted"):
        validate_execution_config(config, repository_root=contract_root)


def test_scope_and_hf_target_cannot_be_relaxed(contract_root: Path) -> None:
    config = build_execution_config_skeleton(repository_root=contract_root)
    changed = copy.deepcopy(config)
    changed["authorization"]["ocr_allowed"] = True
    with pytest.raises(ValueError, match="authorization contract drifted"):
        validate_execution_config(changed, repository_root=contract_root)
    changed = copy.deepcopy(config)
    changed["artifact"]["intended_tag"] = "post-hoc-tag"
    with pytest.raises(ValueError, match="artifact contract drifted"):
        validate_execution_config(changed, repository_root=contract_root)


def test_load_requires_canonical_path_and_canonical_bytes(contract_root: Path) -> None:
    _, path = _materialize(contract_root)
    assert load_execution_contract(
        repository_root=contract_root, execution_config_path=path
    ).config_sha256
    alternate = contract_root / "alternate.json"
    alternate.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="canonical repository path"):
        load_execution_contract(
            repository_root=contract_root, execution_config_path=alternate
        )
    path.write_text(json.dumps(json.loads(path.read_text())), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical pretty JSON"):
        load_execution_contract(
            repository_root=contract_root, execution_config_path=path
        )


def _runtime_values(root: Path, config: Path, output: Path) -> dict:
    source = root / "external-source"
    source.mkdir()
    executable = root / "python"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    values = {
        "repository_root": root.resolve(),
        "execution_config": config.resolve(),
        "source_root": source.resolve(),
        "output_root": output.resolve(),
        "python_executable": executable.resolve(),
        "git_revision": "a" * 40,
        "host_alias": "hyper00",
        "host_hostname": "node-radixark-16-0000",
        "container_id": "container",
        "container_image_digest": "sha256:" + "b" * 64,
        "worker_count": 4,
        "python_version": "3.12.3",
        "pyarrow_version": "24.0.0",
        "pillow_version": "12.2.0",
    }
    assert set(values) == {
        name.removeprefix("--").replace("-", "_")
        for name in (*REQUIRED_PATH_ARGUMENTS, *REQUIRED_IDENTITY_ARGUMENTS)
    }
    return values


def test_runtime_requires_external_no_overwrite_root_and_four_workers(
    contract_root: Path,
) -> None:
    _, config_path = _materialize(contract_root)
    contract = load_execution_contract(
        repository_root=contract_root, execution_config_path=config_path
    )
    values = _runtime_values(contract_root, config_path, contract_root.parent / "output")
    validate_runtime_cli_values(contract, values)
    changed = dict(values)
    changed["worker_count"] = 3
    with pytest.raises(ValueError, match="must equal four"):
        validate_runtime_cli_values(contract, changed)
    changed = dict(values)
    changed["output_root"] = contract_root / "inside"
    with pytest.raises(ValueError, match="outside repository"):
        validate_runtime_cli_values(contract, changed)


def _runtime_closure_from_ast() -> set[str]:
    seen: set[Path] = set()
    stack = [ROOT / RUNNER_PATH]
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("causalcache"):
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("causalcache")
                )
        for name in names:
            candidate = ROOT / "code" / Path(*name.split(".")).with_suffix(".py")
            if candidate.is_file() and candidate not in seen:
                stack.append(candidate)
    return {path.relative_to(ROOT).as_posix() for path in seen}


def test_bound_runtime_import_closure_is_complete_and_cpu_only() -> None:
    assert _runtime_closure_from_ast() == set(REQUIRED_REPOSITORY_SOURCE_PATHS)
    assert validate_cpu_only_source(repository_root=ROOT)["ocr_count"] == 0
    runner = (ROOT / RUNNER_PATH).read_text(encoding="utf-8")
    for forbidden in (
        "build_selected_pilot(",
        "inspect_candidate(",
        "load_selected_rows_once(",
        'row.get("messages")',
        'row.get("metadata")',
        'row["messages"]',
        'row["metadata"]',
    ):
        assert forbidden not in runner
    assert 'row.get("images")' in runner


def _load_runner_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "selected_image_census_runner", ROOT / RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _final_layout(root: Path) -> None:
    (root / "records").mkdir(parents=True)
    (root / "receipts").mkdir()
    (root / "manifest.json").write_bytes(b"{}\n")
    (root / "run-identity.json").write_bytes(b"{}\n")
    for index in range(4):
        (root / "records" / f"worker-{index:02d}.jsonl").write_bytes(b"{}\n")
        (root / "receipts" / f"worker-{index:02d}.json").write_bytes(b"{}\n")


def test_final_root_inventory_and_directory_publish_are_fail_closed(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    staging = tmp_path / "staging"
    _final_layout(staging)
    runner._validate_final_inventory(staging)
    (staging / "unexpected").write_bytes(b"x")
    with pytest.raises(ValueError, match="root inventory drifted"):
        runner._validate_final_inventory(staging)
    (staging / "unexpected").unlink()

    destination = tmp_path / "published"
    runner._rename_directory_no_replace(staging, destination)
    assert destination.is_dir() and not staging.exists()
    second = tmp_path / "second"
    _final_layout(second)
    with pytest.raises(FileExistsError):
        runner._rename_directory_no_replace(second, destination)
    assert second.is_dir() and (destination / "manifest.json").is_file()
