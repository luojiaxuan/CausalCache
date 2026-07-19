from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from causalcache.data.guiodyssey_independent import SourceFileSpec
from causalcache.set_utility_selected_image_census import (
    selector_identity_sha256 as selector_identity_sha256_v1,
)
from causalcache.set_utility_selected_image_census_contract_v2 import (
    CANONICAL_CONFIG_PATH,
    HF_TAG,
    INVALID_V1_ATTEMPT_SUMMARY_SHA256,
    REQUIRED_INPUT_PATHS,
    REQUIRED_REPOSITORY_SOURCE_PATHS,
    RUNNER_PATH,
    build_execution_config_skeleton,
    canonical_pretty_json_bytes,
    load_execution_contract,
    validate_cpu_only_source,
    validate_execution_config,
)
from causalcache.set_utility_selected_image_census_v2 import (
    PARQUET_PROJECTION,
    PROTOCOL_ID,
    WORKER_STATUS,
    build_selected_image_census_records_from_payloads,
    materialize_worker_records,
    selector_identity_sha256,
)


ROOT = Path(__file__).resolve().parents[2]


def _png() -> bytes:
    image = Image.new("RGBA", (7, 9), (1, 2, 3, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _load_runner_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "selected_image_census_v2_runner", ROOT / RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_pyarrow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    batches: list[object],
    num_rows: int,
    capture: list[dict[str, object]],
) -> None:
    class FakeParquetFile:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.metadata = types.SimpleNamespace(num_rows=num_rows)

        def iter_batches(self, *args: object, **kwargs: object) -> list[object]:
            capture.append({"args": args, "kwargs": kwargs})
            return batches

    parquet = types.ModuleType("pyarrow.parquet")
    parquet.ParquetFile = FakeParquetFile
    pyarrow = types.ModuleType("pyarrow")
    pyarrow.__path__ = []
    pyarrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)


class _FakeBatch:
    def __init__(self, names: list[str], rows: list[dict[str, object]]) -> None:
        self.schema = types.SimpleNamespace(names=names)
        self.rows = rows
        self.to_pylist_called = False

    def to_pylist(self) -> list[dict[str, object]]:
        self.to_pylist_called = True
        return self.rows


def test_parquet_factory_passes_exact_projection_to_pyarrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner_module()
    source = tmp_path / "source"
    shard = source / "mobile/use/train/part.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"pinned-shard")
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    spec = SourceFileSpec(
        "mobile/use/train/part.parquet", shard.stat().st_size, digest
    )
    batch = _FakeBatch(["images"], [{"images": []}, {"images": []}])
    capture: list[dict[str, object]] = []
    _install_fake_pyarrow(
        monkeypatch, batches=[batch], num_rows=2, capture=capture
    )

    assert list(runner.ParquetRowFactory(source)(spec)) == [
        (0, {"images": []}),
        (1, {"images": []}),
    ]
    assert capture == [
        {"args": (), "kwargs": {"batch_size": 8, "columns": ["images"]}}
    ]
    assert batch.to_pylist_called is True


def test_parquet_factory_asserts_schema_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner_module()
    source = tmp_path / "source"
    shard = source / "part.parquet"
    source.mkdir()
    shard.write_bytes(b"pinned-shard")
    spec = SourceFileSpec(
        "part.parquet",
        shard.stat().st_size,
        hashlib.sha256(shard.read_bytes()).hexdigest(),
    )
    batch = _FakeBatch(["images", "metadata"], [{"images": []}])
    capture: list[dict[str, object]] = []
    _install_fake_pyarrow(
        monkeypatch, batches=[batch], num_rows=1, capture=capture
    )

    with pytest.raises(ValueError, match="schema must be exactly images"):
        list(runner.ParquetRowFactory(source)(spec))
    assert capture[0]["kwargs"] == {"batch_size": 8, "columns": ["images"]}
    assert batch.to_pylist_called is False


def test_parquet_factory_asserts_exact_projected_row_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner_module()
    source = tmp_path / "source"
    shard = source / "part.parquet"
    source.mkdir()
    shard.write_bytes(b"pinned-shard")
    spec = SourceFileSpec(
        "part.parquet",
        shard.stat().st_size,
        hashlib.sha256(shard.read_bytes()).hexdigest(),
    )
    batch = _FakeBatch(["images"], [{"images": [], "metadata": {}}])
    capture: list[dict[str, object]] = []
    _install_fake_pyarrow(
        monkeypatch, batches=[batch], num_rows=1, capture=capture
    )

    with pytest.raises(ValueError, match="row keys must be exactly images"):
        list(runner.ParquetRowFactory(source)(spec))


def test_v2_selector_and_receipt_are_protocol_consistent(tmp_path: Path) -> None:
    selection = "a" * 64
    assert selector_identity_sha256(
        p0_selection_sha256=selection, observation_ordinal=0
    ) != selector_identity_sha256_v1(
        p0_selection_sha256=selection, observation_ordinal=0
    )
    records = build_selected_image_census_records_from_payloads(
        p0_selection_sha256=selection, image_payloads=[_png()]
    )
    materialize_worker_records(
        tmp_path,
        worker_index=0,
        expected_observation_count=1,
        records=records,
    )
    receipt = json.loads((tmp_path / "receipts/worker-00.json").read_text())
    assert receipt["protocol_id"] == PROTOCOL_ID
    assert receipt["status"] == WORKER_STATUS
    assert receipt["parquet_projection"] == PARQUET_PROJECTION


def test_contract_binds_invalid_predecessor_and_exact_projection() -> None:
    config = build_execution_config_skeleton(repository_root=ROOT)
    contract = validate_execution_config(config, repository_root=ROOT)
    assert contract.config_sha256
    assert config["runtime"]["parquet_projection"] == PARQUET_PROJECTION
    assert config["artifact"] == {
        "hf_mutation_during_execution": False,
        "intended_private_hf_repo": (
            "gavinlaw/causalcache-set-utility-new-development-mobile"
        ),
        "intended_tag": HF_TAG,
        "post_execution_status": "AWAITING_COMMITTED_POSTFLIGHT",
    }
    predecessor = config["bindings"]["frozen_inputs"][
        "invalid_v1_attempt_summary"
    ]
    assert predecessor["sha256"] == INVALID_V1_ATTEMPT_SUMMARY_SHA256
    assert config["authorization"]["access_outcome_or_utility_allowed"] is False
    assert config["authorization"]["hugging_face_mutation_allowed"] is False


def test_contract_fails_closed_on_projection_or_predecessor_drift(
    tmp_path: Path,
) -> None:
    config = build_execution_config_skeleton(repository_root=ROOT)
    changed = json.loads(json.dumps(config))
    changed["runtime"]["parquet_projection"]["columns"] = ["images", "metadata"]
    with pytest.raises(ValueError, match="runtime contract drifted"):
        validate_execution_config(changed, repository_root=ROOT)

    copied: set[str] = set()
    for relative in (*REQUIRED_INPUT_PATHS.values(), *REQUIRED_REPOSITORY_SOURCE_PATHS):
        if relative in copied:
            continue
        copied.add(relative)
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    isolated = build_execution_config_skeleton(repository_root=tmp_path)
    predecessor = tmp_path / REQUIRED_INPUT_PATHS["invalid_v1_attempt_summary"]
    predecessor.write_bytes(predecessor.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="byte binding drifted"):
        validate_execution_config(isolated, repository_root=tmp_path)


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
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("causalcache")
            ):
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


def test_runtime_import_closure_and_static_projection_are_exact() -> None:
    assert _runtime_closure_from_ast() == set(REQUIRED_REPOSITORY_SOURCE_PATHS)
    validation = validate_cpu_only_source(repository_root=ROOT)
    assert validation["projection_call_count"] == 1
    assert validation["parquet_projection"] == PARQUET_PROJECTION


def test_committed_config_is_canonical_and_loadable_after_materialization() -> None:
    path = ROOT / CANONICAL_CONFIG_PATH
    if not path.exists():
        pytest.skip("v2 execution config has not been materialized yet")
    payload = path.read_bytes()
    assert canonical_pretty_json_bytes(json.loads(payload)) == payload
    assert load_execution_contract(
        repository_root=ROOT, execution_config_path=path
    ).config_sha256
