from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import causalcache.set_utility_processor_postflight_parallel_contract_v1 as contract_module
from causalcache.set_utility_processor_freeze_contract_v2 import (
    CANONICAL_EXECUTION_CONFIG_PATH,
)
from causalcache.set_utility_processor_postflight_parallel_contract_v1 import (
    CANONICAL_CONFIG_PATH,
    CLI_PATH,
    HISTORICAL_EXECUTION_CONFIG_SHA256,
    HISTORICAL_POSTFLIGHT_PATH,
    HISTORICAL_POSTFLIGHT_SHA256,
    REQUIRED_SOURCE_PATHS,
    load_parallel_postflight_contract_v1,
)


ROOT = Path(__file__).resolve().parents[2]


def _copy_contract_fixture(destination: Path) -> None:
    paths = (
        CANONICAL_CONFIG_PATH,
        CANONICAL_EXECUTION_CONFIG_PATH,
        HISTORICAL_POSTFLIGHT_PATH,
        *REQUIRED_SOURCE_PATHS,
    )
    for relative in paths:
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def test_live_parallel_contract_binds_historical_and_versioned_sources() -> None:
    contract = load_parallel_postflight_contract_v1(repository_root=ROOT)

    assert contract.data["execution_contract"]["sha256"] == (
        HISTORICAL_EXECUTION_CONFIG_SHA256
    )
    assert contract.data["historical_postflight"]["sha256"] == (
        HISTORICAL_POSTFLIGHT_SHA256
    )
    assert tuple(item["path"] for item in contract.data["sources"]) == (
        REQUIRED_SOURCE_PATHS
    )
    assert contract.data["parallel_postflight"] == {
        "aggregation_order": [0, 1, 2, 3],
        "artifact_semantics": "historical_v2_unchanged",
        "executor": "ThreadPoolExecutor",
        "read_only": True,
        "worker_count": 4,
    }


def test_parallel_contract_fails_closed_on_versioned_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_contract_fixture(tmp_path)
    monkeypatch.setattr(
        contract_module,
        "load_execution_contract",
        lambda **kwargs: SimpleNamespace(config_sha256=HISTORICAL_EXECUTION_CONFIG_SHA256),
    )
    source = tmp_path / REQUIRED_SOURCE_PATHS[1]
    source.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="source binding drifted"):
        load_parallel_postflight_contract_v1(repository_root=tmp_path)


def test_parallel_contract_requires_canonical_path(tmp_path: Path) -> None:
    arbitrary = tmp_path / "parallel.json"
    arbitrary.write_bytes((ROOT / CANONICAL_CONFIG_PATH).read_bytes())

    with pytest.raises(ValueError, match="must be canonical"):
        load_parallel_postflight_contract_v1(
            repository_root=ROOT,
            contract_path=arbitrary,
        )


def test_parallel_postflight_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / CLI_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--parallel-contract" in result.stdout
    assert "--execution-config" in result.stdout
    assert "--producer-repository-root" in result.stdout
