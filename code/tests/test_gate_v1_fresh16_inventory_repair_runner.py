from __future__ import annotations

import hashlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from causalcache import gate_v1_fresh16_inventory_repair_runner as repair


PARENT_STATES = (
    "runtime_receipts",
    "global_claim",
    "transport_verification",
    "label_blind_cpu_completion",
    "checkpoint_replay_completion",
    "policy_worker_even_completion",
    "policy_worker_odd_completion",
    "heuristic_local_seal",
    "label_access_claim",
    "label_cache_completion",
    "remote_base_receipt",
    "payload_commit_receipt",
    "primary_report_completion",
    "report_commit_receipt",
    "completion_staging",
    "final_completion",
)


def _write(root: Path, canonical: str, payload: bytes, mode: int = 0o600) -> dict:
    relative = Path(*Path(canonical).parts[2:])
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return {
        "path": canonical,
        "mode": mode,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _fixture(tmp_path: Path) -> SimpleNamespace:
    namespace = "/data/experiments/causalcache/gate-v1-fresh16-evaluation-v1"
    receipts = []
    for ordinal, name in enumerate(PARENT_STATES[:2]):
        record = _write(
            tmp_path,
            f"{namespace}/ordered-state-receipts/{ordinal:02d}-{name}.json",
            f"receipt-{name}".encode(),
        )
        record["state_name"] = name
        receipts.append(record)
    artifact = tmp_path / "artifacts/causalcache/gate-v1-fresh16-evaluation-v1"
    artifact.mkdir(parents=True)
    failure = {
        "old_state_namespace": namespace,
        "ordered_receipts": receipts,
        "expected_absent_successor_state_names": list(PARENT_STATES[2:]),
        "old_artifact_directory": (
            "/data/artifacts/causalcache/gate-v1-fresh16-evaluation-v1"
        ),
        "old_artifact_directory_expected_empty": True,
        "old_runtime_receipt": _write(
            tmp_path,
            "/data/experiments/causalcache/.gate-v1-fresh16-evaluation-v1.docker-inspect.json",
            b"runtime",
        ),
        "run_evidence": {
            name: _write(
                tmp_path,
                f"/data/logs/gate-v1-fresh16-evaluation-v1.{name}",
                name.encode(),
            )
            for name in ("log", "started", "exit")
        },
        "old_destination": {
            "repo": "owner/old",
            "repo_type": "dataset",
            "tag": "old-v1",
            "expected_absent": True,
            "remote_mutation_count": 0,
        },
    }
    inventory_repair = {"parent_failure": failure}
    return SimpleNamespace(
        data={},
        repair=inventory_repair,
        overlay={"inventory_repair": inventory_repair},
        destination={"repo": "owner/repair", "repo_type": "dataset"},
    )


def test_retained_failure_exact_identity_and_absence_pass(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    before = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    result = repair.validate_retained_v1_failure(contract, tmp_path)
    after = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    assert result["ordered_receipt_state_names"] == list(PARENT_STATES[:2])
    assert result["absent_successor_receipt_count"] == 14
    assert result["old_artifact_entry_count"] == 0
    assert result["fresh_semantic_access_count"] == 0
    assert before == after


@pytest.mark.parametrize("field", ["mode", "size_bytes", "sha256"])
def test_retained_failure_rejects_receipt_identity_drift(
    tmp_path: Path, field: str
) -> None:
    contract = _fixture(tmp_path)
    record = contract.repair["parent_failure"]["ordered_receipts"][0]
    record[field] = {"mode": 0o400, "size_bytes": 999, "sha256": "0" * 64}[field]
    with pytest.raises(ValueError, match="drifted"):
        repair.validate_retained_v1_failure(contract, tmp_path)


def test_retained_failure_rejects_extra_receipt(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    _write(
        tmp_path,
        "/data/experiments/causalcache/gate-v1-fresh16-evaluation-v1/ordered-state-receipts/02-transport_verification.json",
        b"unexpected",
    )
    with pytest.raises(ValueError, match="exactly two"):
        repair.validate_retained_v1_failure(contract, tmp_path)


def test_retained_failure_rejects_unexpected_state_namespace_entry(
    tmp_path: Path,
) -> None:
    contract = _fixture(tmp_path)
    namespace = (
        tmp_path
        / "experiments/causalcache/gate-v1-fresh16-evaluation-v1"
    )
    (namespace / "label-access-claim.json").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="only ordered-state-receipts"):
        repair.validate_retained_v1_failure(contract, tmp_path)


def test_retained_failure_rejects_nonempty_artifact_directory(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    artifact = tmp_path / "artifacts/causalcache/gate-v1-fresh16-evaluation-v1"
    (artifact / "unexpected").write_bytes(b"x")
    with pytest.raises(ValueError, match="not empty"):
        repair.validate_retained_v1_failure(contract, tmp_path)


def test_retained_failure_rejects_symlinked_evidence(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    record = contract.repair["parent_failure"]["run_evidence"]["log"]
    path = tmp_path / Path(*Path(record["path"]).parts[2:])
    target = tmp_path / "target"
    target.write_bytes(path.read_bytes())
    target.chmod(0o600)
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="missing or unsafe"):
        repair.validate_retained_v1_failure(contract, tmp_path)


class RepositoryNotFoundError(Exception):
    pass


class _AbsentApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def repo_info(self, repo: str, *, repo_type: str) -> None:
        self.calls.append((repo, repo_type))
        raise RepositoryNotFoundError(repo)


def test_old_and_repair_destination_absence_are_read_only(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    api = _AbsentApi()
    old = repair.validate_abandoned_v1_destination_absent(contract, api)
    new = repair.validate_repair_destination_absent(contract, api)
    assert api.calls == [("owner/old", "dataset"), ("owner/repair", "dataset")]
    assert old["remote_mutation_count"] == new["remote_mutation_count"] == 0


def test_destination_absence_rejects_existing_repo(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    api = SimpleNamespace(repo_info=lambda *_args, **_kwargs: object())
    with pytest.raises(ValueError, match="must remain absent"):
        repair.validate_abandoned_v1_destination_absent(contract, api)


def _execute_kwargs(tmp_path: Path, contract: object, api: object) -> dict:
    return {
        "mode": "run",
        "contract": contract,
        "api": api,
        "download_fn": lambda **_kwargs: "",
        "operation_factory": lambda **kwargs: kwargs,
        "expected_execution_b_git_commit": "b" * 40,
        "data_root": tmp_path,
        "fresh_download_parent": tmp_path,
        "model_dir": tmp_path,
        "snapshot_manifest": tmp_path / "snapshot.json",
        "docker_inspect_receipt": tmp_path / "runtime.json",
        "devices": ("cuda:0", "cuda:1"),
        "gpu_uuids": ("GPU-0", "GPU-1"),
    }


def test_execute_checks_local_old_and_new_before_base_delegate(tmp_path: Path) -> None:
    events: list[str] = []
    delegated: dict = {}
    contract = object()

    def execute(**kwargs):
        events.append("base")
        delegated.update(kwargs)
        return {"status": "done"}

    result = repair.execute_fresh16_inventory_repair(
        **_execute_kwargs(tmp_path, contract, object()),
        evidence_validator=lambda *_args: events.append("local") or {"ok": True},
        old_destination_validator=lambda *_args: events.append("old") or {"ok": True},
        repair_destination_validator=lambda *_args: events.append("new")
        or {"ok": True},
        execute_fn=execute,
    )
    assert events == ["local", "old", "new", "base"]
    assert result["status"] == "done"
    assert result["inventory_repair_preflight"]["remote_mutation_before_delegate"] == 0
    assert delegated["execution_preflight"] == result["inventory_repair_preflight"]


def test_execute_validate_skips_new_destination_absence(tmp_path: Path) -> None:
    events: list[str] = []
    kwargs = _execute_kwargs(tmp_path, object(), object())
    kwargs["mode"] = "validate"
    repair.execute_fresh16_inventory_repair(
        **kwargs,
        evidence_validator=lambda *_args: events.append("local") or {},
        old_destination_validator=lambda *_args: events.append("old") or {},
        repair_destination_validator=lambda *_args: events.append("new") or {},
        execute_fn=lambda **_kwargs: events.append("base") or {"status": "validated"},
    )
    assert events == ["local", "old", "base"]


def test_execute_preflight_failure_stops_before_base_delegate(tmp_path: Path) -> None:
    delegate = mock.Mock()
    with pytest.raises(ValueError, match="old evidence drifted"):
        repair.execute_fresh16_inventory_repair(
            **_execute_kwargs(tmp_path, object(), object()),
            evidence_validator=lambda *_args: (_ for _ in ()).throw(
                ValueError("old evidence drifted")
            ),
            execute_fn=delegate,
        )
    delegate.assert_not_called()


def test_manager_parser_and_source_materialization_use_repair_contract() -> None:
    from scripts import manage_gate_v1_fresh16_inventory_repair as manager

    args = manager._parser().parse_args(
        ["validate-source", "--repository-root", "/repo"]
    )
    assert args.contract == manager.CANONICAL_CONFIG_PATH
    contract = object()
    with (
        mock.patch.object(
            manager,
            "load_frozen_fresh16_inventory_repair_contract",
            return_value=contract,
        ),
        mock.patch.object(
            manager,
            "validate_fresh16_inventory_repair_source_only_contract",
            return_value={"status": "overlay-ok"},
        ) as overlay,
        mock.patch.object(
            manager,
            "materialize_runner_freeze",
            return_value={"execution_b_required_unique_diff": [manager.CANONICAL_CONFIG_PATH]},
        ) as materialize,
    ):
        result = manager._run(
            SimpleNamespace(
                command="materialize-runner-freeze",
                repository_root=Path("/repo"),
                contract="repair.json",
                source_a_git_commit="a" * 40,
            )
        )
    assert result["execution_b_required_unique_diff"]
    overlay.assert_called_once_with("repair.json", repository_root=Path("/repo"))
    materialize.assert_called_once_with(
        contract, expected_source_a_git_commit="a" * 40
    )


def test_manager_forwards_secure_run_arguments_to_wrapper(tmp_path: Path) -> None:
    from scripts import manage_gate_v1_fresh16_inventory_repair as manager

    contract = object()
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.CommitOperationAdd = object
    fake_hub.HfApi = lambda *, token: SimpleNamespace(token=token)
    fake_hub.hf_hub_download = lambda **_kwargs: "/download"
    args = SimpleNamespace(
        command="run",
        repository_root=Path("/repo"),
        contract="repair.json",
        execution_b_git_commit="b" * 40,
        hf_token_file=tmp_path / "token",
        data_root=tmp_path,
        fresh_download_parent=tmp_path,
        model_dir=tmp_path / "model",
        snapshot_manifest=tmp_path / "snapshot.json",
        docker_inspect_receipt=tmp_path / "runtime.json",
        device=["cuda:0", "cuda:1"],
        gpu_uuid=["GPU-0", "GPU-1"],
    )
    with (
        mock.patch.object(
            manager,
            "load_frozen_fresh16_inventory_repair_contract",
            return_value=contract,
        ),
        mock.patch.object(manager.base_manager, "_secure_token", return_value="secret"),
        mock.patch.object(
            manager,
            "validate_retained_v1_failure",
            return_value={"status": "retained-ok"},
        ),
        mock.patch.object(
            manager,
            "validate_execution_b_source",
            return_value=object(),
        ),
        mock.patch.object(
            manager,
            "execute_fresh16_inventory_repair",
            return_value={"status": "done"},
        ) as execute,
        mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}, clear=False),
    ):
        result = manager._run(args)
    assert result == {"status": "done"}
    call = execute.call_args.kwargs
    assert call["contract"] is contract
    assert call["devices"] == ("cuda:0", "cuda:1")
    assert call["gpu_uuids"] == ("GPU-0", "GPU-1")
    assert call["evidence_validator"](contract, tmp_path) == {
        "status": "retained-ok"
    }


def test_manager_retained_evidence_failure_precedes_token_and_hf_client(
    tmp_path: Path,
) -> None:
    from scripts import manage_gate_v1_fresh16_inventory_repair as manager

    events: list[str] = []
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.CommitOperationAdd = object
    fake_hub.HfApi = lambda *, token: events.append("hf-client") or object()
    fake_hub.hf_hub_download = lambda **_kwargs: "/download"
    args = SimpleNamespace(
        command="run",
        repository_root=Path("/repo"),
        contract="repair.json",
        execution_b_git_commit="b" * 40,
        hf_token_file=tmp_path / "token",
        data_root=tmp_path,
        fresh_download_parent=tmp_path,
        model_dir=tmp_path / "model",
        snapshot_manifest=tmp_path / "snapshot.json",
        docker_inspect_receipt=tmp_path / "runtime.json",
        device=["cuda:0", "cuda:1"],
        gpu_uuid=["GPU-0", "GPU-1"],
    )
    with (
        mock.patch.object(
            manager,
            "load_frozen_fresh16_inventory_repair_contract",
            return_value=object(),
        ),
        mock.patch.object(
            manager,
            "validate_retained_v1_failure",
            side_effect=lambda *_args: events.append("evidence")
            or (_ for _ in ()).throw(ValueError("retained evidence drifted")),
        ),
        mock.patch.object(
            manager,
            "validate_execution_b_source",
            return_value=object(),
        ),
        mock.patch.object(
            manager.base_manager,
            "_secure_token",
            side_effect=lambda _path: events.append("token") or "secret",
        ) as token,
        mock.patch.object(manager, "execute_fresh16_inventory_repair") as execute,
        mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}, clear=False),
    ):
        with pytest.raises(ValueError, match="retained evidence drifted"):
            manager._run(args)
    assert events == ["evidence"]
    token.assert_not_called()
    execute.assert_not_called()


def test_manager_execution_b_failure_precedes_token_hf_and_evidence(
    tmp_path: Path,
) -> None:
    from scripts import manage_gate_v1_fresh16_inventory_repair as manager

    args = SimpleNamespace(
        command="run",
        repository_root=Path("/repo"),
        contract="repair.json",
        execution_b_git_commit="b" * 40,
        hf_token_file=tmp_path / "token",
        data_root=tmp_path,
        fresh_download_parent=tmp_path,
        model_dir=tmp_path / "model",
        snapshot_manifest=tmp_path / "snapshot.json",
        docker_inspect_receipt=tmp_path / "runtime.json",
        device=["cuda:0", "cuda:1"],
        gpu_uuid=["GPU-0", "GPU-1"],
    )
    with (
        mock.patch.object(
            manager,
            "load_frozen_fresh16_inventory_repair_contract",
            return_value=object(),
        ),
        mock.patch.object(
            manager,
            "validate_execution_b_source",
            side_effect=ValueError("Execution-B identity drifted"),
        ),
        mock.patch.object(manager, "validate_retained_v1_failure") as evidence,
        mock.patch.object(manager.base_manager, "_secure_token") as token,
        mock.patch.object(manager, "execute_fresh16_inventory_repair") as execute,
    ):
        with pytest.raises(ValueError, match="Execution-B identity drifted"):
            manager._run(args)
    evidence.assert_not_called()
    token.assert_not_called()
    execute.assert_not_called()


def test_manager_run_rejects_incomplete_two_gpu_arguments(tmp_path: Path) -> None:
    from scripts import manage_gate_v1_fresh16_inventory_repair as manager

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.CommitOperationAdd = object
    fake_hub.HfApi = lambda *, token: object()
    fake_hub.hf_hub_download = lambda **_kwargs: "/download"
    args = SimpleNamespace(
        command="run",
        repository_root=Path("/repo"),
        contract="repair.json",
        execution_b_git_commit="b" * 40,
        hf_token_file=tmp_path / "token",
        data_root=tmp_path,
        fresh_download_parent=tmp_path,
        model_dir=tmp_path / "model",
        snapshot_manifest=tmp_path / "snapshot.json",
        docker_inspect_receipt=tmp_path / "runtime.json",
        device=["cuda:0"],
        gpu_uuid=["GPU-0"],
    )
    with (
        mock.patch.object(
            manager,
            "load_frozen_fresh16_inventory_repair_contract",
            return_value=object(),
        ),
        mock.patch.object(
            manager.base_manager, "_secure_token", return_value="secret"
        ) as secure_token,
        mock.patch.object(
            manager,
            "validate_retained_v1_failure",
            return_value={"status": "retained-ok"},
        ),
        mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}, clear=False),
    ):
        with pytest.raises(ValueError, match="exactly two"):
            manager._run(args)
    secure_token.assert_not_called()
