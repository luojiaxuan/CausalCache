from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from causalcache import (
    set_utility_processor_postflight_parallel_contract_v1 as contract_module,
)
from causalcache.set_utility_processor_freeze_contract_v2 import (
    CANONICAL_EXECUTION_CONFIG_PATH,
)
from causalcache.set_utility_processor_postflight_parallel_contract_v1 import (
    CANONICAL_CONFIG_PATH,
    HISTORICAL_EXECUTION_CONFIG_SHA256,
    HISTORICAL_POSTFLIGHT_PATH,
    ParallelPostflightContractV1,
    REQUIRED_SOURCE_PATHS,
    load_parallel_postflight_contract_v1,
    load_producer_execution_contract_v1,
)


ROOT = Path(__file__).resolve().parents[2]


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit_fixture(root: Path, *, message: str) -> str:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.com")
    _git(root, "config", "user.name", "Fixture")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", message)
    revision = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    assert _git(root, "status", "--porcelain=v1", "--untracked-files=all") == ""
    return revision


def _copy_paths(destination: Path, paths: tuple[str, ...]) -> None:
    destination.mkdir(parents=True)
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


def _clean_validator_root(destination: Path) -> Path:
    _copy_paths(
        destination,
        (CANONICAL_CONFIG_PATH, *REQUIRED_SOURCE_PATHS),
    )
    _commit_fixture(destination, message="parallel validator fixture")
    return destination


def _clean_producer_root(destination: Path) -> tuple[Path, str]:
    _copy_paths(
        destination,
        (CANONICAL_EXECUTION_CONFIG_PATH, HISTORICAL_POSTFLIGHT_PATH),
    )
    revision = _commit_fixture(destination, message="historical producer fixture")
    return destination, revision


def _load_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    ParallelPostflightContractV1,
    Path,
    Path,
    str,
    list[dict[str, object]],
]:
    validator_root = _clean_validator_root(tmp_path / "validator-checkout")
    producer_root, producer_revision = _clean_producer_root(
        tmp_path / "producer-checkout"
    )
    calls: list[dict[str, object]] = []

    def fake_load_execution_contract(**kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        return SimpleNamespace(config_sha256=HISTORICAL_EXECUTION_CONFIG_SHA256)

    monkeypatch.setattr(
        contract_module,
        "load_execution_contract",
        fake_load_execution_contract,
    )
    parallel_contract = load_parallel_postflight_contract_v1(
        repository_root=validator_root,
    )
    return (
        parallel_contract,
        validator_root,
        producer_root,
        producer_revision,
        calls,
    )


def test_distinct_clean_validator_and_producer_roots_succeed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        parallel_contract,
        validator_root,
        producer_root,
        producer_revision,
        calls,
    ) = _load_roots(tmp_path, monkeypatch)
    producer_config = producer_root / CANONICAL_EXECUTION_CONFIG_PATH

    execution_contract = load_producer_execution_contract_v1(
        parallel_contract,
        producer_repository_root=producer_root,
        execution_config_path=producer_config,
        expected_git_revision=producer_revision,
    )

    assert validator_root.resolve() != producer_root.resolve()
    assert parallel_contract.repository_root == validator_root.resolve()
    assert execution_contract.config_sha256 == HISTORICAL_EXECUTION_CONFIG_SHA256
    assert calls == [
        {
            "repository_root": producer_root.resolve(),
            "execution_config_path": producer_config,
        }
    ]
    assert _git(validator_root, "status", "--porcelain=v1") == ""
    assert _git(producer_root, "status", "--porcelain=v1") == ""


def test_producer_root_rejects_relative_and_non_directory_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parallel_contract, _, producer_root, revision, _ = _load_roots(
        tmp_path, monkeypatch
    )
    config = producer_root / CANONICAL_EXECUTION_CONFIG_PATH

    with pytest.raises(ValueError, match="must be absolute"):
        load_producer_execution_contract_v1(
            parallel_contract,
            producer_repository_root=Path("producer-checkout"),
            execution_config_path=config,
            expected_git_revision=revision,
        )
    with pytest.raises(ValueError, match="must be a real directory"):
        load_producer_execution_contract_v1(
            parallel_contract,
            producer_repository_root=config,
            execution_config_path=config,
            expected_git_revision=revision,
        )


def test_producer_root_must_be_exact_checkout_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parallel_contract, _, producer_root, revision, _ = _load_roots(
        tmp_path, monkeypatch
    )

    with pytest.raises(ValueError, match="must be the Git checkout root"):
        load_producer_execution_contract_v1(
            parallel_contract,
            producer_repository_root=producer_root / "code",
            execution_config_path=producer_root / CANONICAL_EXECUTION_CONFIG_PATH,
            expected_git_revision=revision,
        )


def test_producer_root_rejects_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parallel_contract, _, producer_root, revision, _ = _load_roots(
        tmp_path, monkeypatch
    )
    linked_root = tmp_path / "producer-link"
    linked_root.symlink_to(producer_root, target_is_directory=True)

    with pytest.raises(ValueError, match="real directory|traverse symlinks"):
        load_producer_execution_contract_v1(
            parallel_contract,
            producer_repository_root=linked_root,
            execution_config_path=producer_root / CANONICAL_EXECUTION_CONFIG_PATH,
            expected_git_revision=revision,
        )


def test_producer_root_rejects_dirty_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parallel_contract, _, producer_root, revision, calls = _load_roots(
        tmp_path, monkeypatch
    )
    (producer_root / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkout must be clean"):
        load_producer_execution_contract_v1(
            parallel_contract,
            producer_repository_root=producer_root,
            execution_config_path=producer_root / CANONICAL_EXECUTION_CONFIG_PATH,
            expected_git_revision=revision,
        )
    assert calls == []


def test_producer_root_rejects_head_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parallel_contract, _, producer_root, _, calls = _load_roots(
        tmp_path, monkeypatch
    )

    with pytest.raises(ValueError, match="HEAD differs"):
        load_producer_execution_contract_v1(
            parallel_contract,
            producer_repository_root=producer_root,
            execution_config_path=producer_root / CANONICAL_EXECUTION_CONFIG_PATH,
            expected_git_revision="0" * 40,
        )
    assert calls == []


@pytest.mark.parametrize("config_kind", ["relative", "validator", "other"])
def test_execution_config_must_be_absolute_producer_canonical_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_kind: str,
) -> None:
    (
        parallel_contract,
        validator_root,
        producer_root,
        revision,
        calls,
    ) = _load_roots(tmp_path, monkeypatch)
    if config_kind == "relative":
        supplied_config = Path(CANONICAL_EXECUTION_CONFIG_PATH)
    elif config_kind == "validator":
        supplied_config = validator_root / CANONICAL_EXECUTION_CONFIG_PATH
    else:
        supplied_config = producer_root / "other-execution-config.json"

    with pytest.raises(ValueError, match="producer repository canonical config"):
        load_producer_execution_contract_v1(
            parallel_contract,
            producer_repository_root=producer_root,
            execution_config_path=supplied_config,
            expected_git_revision=revision,
        )
    assert calls == []
