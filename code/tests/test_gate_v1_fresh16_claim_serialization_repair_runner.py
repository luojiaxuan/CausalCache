from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest import mock

import pytest

from causalcache import gate_v1_fresh16_claim_serialization_repair_runner as repair
from causalcache import gate_v1_fresh16_evaluation_runner as base_runner
from causalcache.gate_v1_fresh16_evaluation_contract import canonical_json_bytes


RETAINED = (
    "runtime_receipts",
    "global_claim",
    "transport_verification",
    "label_blind_cpu_completion",
    "checkpoint_replay_completion",
    "policy_worker_even_completion",
    "policy_worker_odd_completion",
    "heuristic_local_seal",
)
SUCCESSORS = (
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
    path = root.joinpath(*PurePosixPath(canonical).parts[2:])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return {
        "path": canonical,
        "mode": mode,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _rebind(record: dict, path: Path) -> None:
    payload = path.read_bytes()
    record["mode"] = path.stat().st_mode & 0o777
    record["size_bytes"] = len(payload)
    record["sha256"] = hashlib.sha256(payload).hexdigest()


def _artifact_records(root: Path, artifact: Path) -> list[dict]:
    result = []
    for path in sorted(item for item in artifact.rglob("*") if item.is_file()):
        payload = path.read_bytes()
        result.append(
            {
                "path": path.relative_to(artifact).as_posix(),
                "mode": path.stat().st_mode & 0o777,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return result


def _fixture(tmp_path: Path) -> SimpleNamespace:
    namespace = (
        "/data/experiments/causalcache/"
        "gate-v1-fresh16-evaluation-inventory-repair-v1"
    )
    artifact_canonical = (
        "/data/artifacts/causalcache/"
        "gate-v1-fresh16-evaluation-inventory-repair-v1"
    )
    artifact = tmp_path.joinpath(*PurePosixPath(artifact_canonical).parts[2:])
    for index in range(8):
        path = artifact / "fresh16-eval" / f"blind-{index}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"blind-{index}\n".encode())
        path.chmod(0o444)
    for index in range(2):
        path = artifact / "selected-images" / f"image-{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"image-{index}".encode())
        path.chmod(0o444)
    artifact_records = _artifact_records(tmp_path, artifact)
    blind_inventory = [
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in artifact_records
        if item["path"].startswith("fresh16-eval/")
    ]
    blind_sha = hashlib.sha256(canonical_json_bytes(blind_inventory)).hexdigest()
    seal = {
        "schema_version": base_runner.SCHEMA_VERSION,
        "protocol_id": base_runner.PROTOCOL_ID,
        "status": base_runner.HEURISTIC_SEAL_STATUS,
        "inventory": blind_inventory,
        "inventory_sha256": blind_sha,
        "label_access_authorized": False,
    }
    seal_record = _write(
        tmp_path,
        f"{namespace}/heuristic-local-seal.json",
        base_runner.pretty_json_bytes(seal),
    )

    receipts = []
    previous = None
    for ordinal, name in enumerate(RETAINED):
        stage_payload = {"stage_marker": name}
        if name == "heuristic_local_seal":
            stage_payload = {
                "status": base_runner.HEURISTIC_SEAL_STATUS,
                "label_blind_file_count": 8,
                "label_blind_inventory_sha256": blind_sha,
                "label_access_authorized": False,
            }
        receipt = {
            "schema_version": base_runner.SCHEMA_VERSION,
            "protocol_id": base_runner.PROTOCOL_ID,
            "status": "COMPLETED_GATE_V1_FRESH16_ORDERED_STATE_V1",
            "ordinal": ordinal,
            "stage": name,
            "previous_receipt_sha256": previous,
            "payload_sha256": hashlib.sha256(
                canonical_json_bytes(stage_payload)
            ).hexdigest(),
            "payload": stage_payload,
        }
        payload = base_runner.pretty_json_bytes(receipt)
        bound = _write(
            tmp_path,
            f"{namespace}/ordered-state-receipts/{ordinal:02d}-{name}.json",
            payload,
        )
        bound["state_name"] = name
        receipts.append(bound)
        previous = hashlib.sha256(payload).hexdigest()

    artifact_payload = canonical_json_bytes(artifact_records)
    run_evidence = {
        name: _write(
            tmp_path,
            f"/data/logs/gate-v1-fresh16-evaluation-inventory-repair-v1.run.{name}",
            name.encode(),
        )
        for name in ("log", "started", "exit")
    }
    failure = {
        "old_state_namespace": namespace,
        "expected_state_root_entries": [
            "heuristic-local-seal.json",
            "ordered-state-receipts",
        ],
        "ordered_receipts": receipts,
        "expected_absent_successor_state_names": list(SUCCESSORS),
        "heuristic_local_seal": seal_record,
        "expected_absent_standalone_files": [
            f"{namespace}/label-access-claim.json",
            f"{namespace}/remote-base-pre-mutation.json",
            f"{namespace}/remote-base-receipt.json",
            f"{namespace}/final-completion.json",
        ],
        "artifact_tree": {
            "root": artifact_canonical,
            "top_level_entries": ["fresh16-eval", "selected-images"],
            "file_count": len(artifact_records),
            "total_bytes": sum(item["size_bytes"] for item in artifact_records),
            "canonical_inventory_json_size_bytes": len(artifact_payload),
            "canonical_inventory_sha256": hashlib.sha256(
                artifact_payload
            ).hexdigest(),
        },
        "run_evidence": run_evidence,
        "old_runtime_receipt": _write(
            tmp_path,
            "/data/experiments/causalcache/"
            ".gate-v1-fresh16-evaluation-inventory-repair-v1.docker-inspect.json",
            b"runtime",
        ),
        "old_destinations": [
            {"repo": "gavinlaw/old-v1", "repo_type": "dataset"},
            {"repo": "gavinlaw/old-repair", "repo_type": "dataset"},
        ],
    }
    repair_overlay = {"parent_failure": failure}
    return SimpleNamespace(
        repair=repair_overlay,
        overlay={"claim_serialization_repair": repair_overlay},
        data={
            "local_first_state_machine": {
                "artifact_directory": "/data/artifacts/causalcache/new-claim-repair",
                "state_directory": "/data/experiments/causalcache",
                "execution_namespace": "new-claim-repair",
            }
        },
        destination={"repo": "gavinlaw/new-repair", "repo_type": "dataset"},
        parent=object(),
    )


def _path(root: Path, canonical: str) -> Path:
    return root.joinpath(*PurePosixPath(canonical).parts[2:])


def test_retained_failure_validates_chain_seal_artifact_and_absence(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    before = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    result = repair.validate_retained_inventory_repair_failure(contract, tmp_path)
    after = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    assert result["ordered_receipt_count"] == 8
    assert result["absent_successor_receipt_count"] == 8
    assert result["sealed_label_blind_file_count"] == 8
    assert result["artifact_file_count"] == 10
    assert result["fresh_label_semantic_decode_count"] == 0
    assert before == after


@pytest.mark.parametrize("field", ["mode", "size_bytes", "sha256"])
def test_retained_failure_rejects_every_receipt_identity_drift(
    tmp_path: Path, field: str
) -> None:
    contract = _fixture(tmp_path)
    record = contract.repair["parent_failure"]["ordered_receipts"][4]
    record[field] = {"mode": 0o400, "size_bytes": 999, "sha256": "0" * 64}[field]
    with pytest.raises(ValueError, match="drifted"):
        repair.validate_retained_inventory_repair_failure(contract, tmp_path)


def test_retained_failure_rejects_strict_json_payload_hash_and_chain_drift(
    tmp_path: Path,
) -> None:
    contract = _fixture(tmp_path)
    record = contract.repair["parent_failure"]["ordered_receipts"][3]
    path = _path(tmp_path, record["path"])
    value = json.loads(path.read_text())
    value["payload_sha256"] = "0" * 64
    path.write_bytes(base_runner.pretty_json_bytes(value))
    path.chmod(0o600)
    _rebind(record, path)
    with pytest.raises(ValueError, match="JSON or predecessor chain"):
        repair.validate_retained_inventory_repair_failure(contract, tmp_path)


@pytest.mark.parametrize(
    ("record_path", "field"),
    [
        (("heuristic_local_seal",), "size_bytes"),
        (("run_evidence", "log"), "sha256"),
        (("old_runtime_receipt",), "mode"),
    ],
)
def test_retained_failure_rejects_seal_log_and_runtime_identity_drift(
    tmp_path: Path, record_path: tuple[str, ...], field: str
) -> None:
    contract = _fixture(tmp_path)
    record = contract.repair["parent_failure"]
    for key in record_path:
        record = record[key]
    record[field] = {"mode": 0o400, "size_bytes": 999, "sha256": "0" * 64}[field]
    with pytest.raises(ValueError, match="drifted"):
        repair.validate_retained_inventory_repair_failure(contract, tmp_path)


def test_retained_failure_rejects_symlinked_run_evidence(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    record = contract.repair["parent_failure"]["run_evidence"]["log"]
    path = _path(tmp_path, record["path"])
    target = tmp_path / "run-log-target"
    target.write_bytes(path.read_bytes())
    target.chmod(0o600)
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="missing or unsafe"):
        repair.validate_retained_inventory_repair_failure(contract, tmp_path)


def test_retained_failure_rejects_receipt_count_and_standalone_absence_drift(
    tmp_path: Path,
) -> None:
    contract = _fixture(tmp_path)
    failure = contract.repair["parent_failure"]
    _write(
        tmp_path,
        f"{failure['old_state_namespace']}/ordered-state-receipts/08-label_access_claim.json",
        b"unexpected",
    )
    with pytest.raises(ValueError, match="exactly states 0..7"):
        repair.validate_retained_inventory_repair_failure(contract, tmp_path)

    contract = _fixture(tmp_path / "second")
    failure = contract.repair["parent_failure"]
    _write(
        tmp_path / "second",
        f"{failure['old_state_namespace']}/label-access-claim.json",
        b"unexpected",
    )
    with pytest.raises(ValueError, match="entry inventory"):
        repair.validate_retained_inventory_repair_failure(contract, tmp_path / "second")


def test_retained_failure_rejects_heuristic_seal_artifact_contradiction(
    tmp_path: Path,
) -> None:
    contract = _fixture(tmp_path)
    record = contract.repair["parent_failure"]["heuristic_local_seal"]
    path = _path(tmp_path, record["path"])
    value = json.loads(path.read_text())
    value["inventory"][0]["sha256"] = "0" * 64
    value["inventory_sha256"] = hashlib.sha256(
        canonical_json_bytes(value["inventory"])
    ).hexdigest()
    path.write_bytes(base_runner.pretty_json_bytes(value))
    path.chmod(0o600)
    _rebind(record, path)
    with pytest.raises(ValueError, match="receipt and standalone seal contradict"):
        repair.validate_retained_inventory_repair_failure(contract, tmp_path)


@pytest.mark.parametrize("mutation", ["bytes", "mode", "symlink", "empty_dir"])
def test_retained_failure_rejects_artifact_byte_mode_symlink_or_empty_dir(
    tmp_path: Path, mutation: str
) -> None:
    contract = _fixture(tmp_path)
    spec = contract.repair["parent_failure"]["artifact_tree"]
    root = _path(tmp_path, spec["root"])
    target = root / "fresh16-eval/blind-0.jsonl"
    if mutation == "bytes":
        target.chmod(0o644)
        target.write_bytes(b"changed")
        target.chmod(0o444)
    elif mutation == "mode":
        target.chmod(0o644)
    elif mutation == "symlink":
        link_target = tmp_path / "artifact-target"
        link_target.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(link_target)
    else:
        (root / "fresh16-eval/extra-empty").mkdir()
    with pytest.raises(ValueError, match="artifact"):
        repair.validate_retained_inventory_repair_failure(contract, tmp_path)


def test_new_run_roots_must_both_be_absent(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    assert repair.validate_new_run_roots_absent(contract, tmp_path)["both_absent"]
    _path(tmp_path, contract.data["local_first_state_machine"]["artifact_directory"]).mkdir(
        parents=True
    )
    with pytest.raises(ValueError, match="new artifact root"):
        repair.validate_new_run_roots_absent(contract, tmp_path)


class RepositoryNotFoundError(Exception):
    pass


class _AbsentApi:
    def __init__(self, owner: str = "gavinlaw", role: str = "write") -> None:
        self.owner = owner
        self.role = role
        self.events: list[str] = []

    def whoami(self) -> dict:
        self.events.append("whoami")
        return {
            "name": self.owner,
            "auth": {
                "type": "access_token",
                "accessToken": {"role": self.role},
            },
        }

    def repo_info(self, repo: str, *, repo_type: str) -> None:
        self.events.append(f"repo:{repo}:{repo_type}")
        raise RepositoryNotFoundError(repo)


def test_hf_owner_precedes_three_unambiguous_absence_checks(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    api = _AbsentApi()
    result = repair.validate_destinations_before_delegate(
        contract, api, require_new_absent=True
    )
    assert api.events[0] == "whoami"
    assert len(api.events) == 4
    assert result["hf_identity"]["owner"] == "gavinlaw"
    assert result["hf_identity"]["access_token_role"] == "write"
    assert result["remote_mutation_count"] == 0


def test_hf_wrong_owner_or_existing_private_repo_fails_closed(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    wrong = _AbsentApi(owner="someone-else")
    with pytest.raises(ValueError, match="owner must be gavinlaw"):
        repair.validate_destinations_before_delegate(
            contract, wrong, require_new_absent=True
        )
    assert wrong.events == ["whoami"]

    fine_grained = _AbsentApi(role="fineGrained")
    with pytest.raises(ValueError, match="owner-wide write access"):
        repair.validate_destinations_before_delegate(
            contract, fine_grained, require_new_absent=True
        )
    assert fine_grained.events == ["whoami"]

    existing = _AbsentApi()
    existing.repo_info = lambda *_args, **_kwargs: object()
    with pytest.raises(ValueError, match="must remain absent"):
        repair.validate_destinations_before_delegate(
            contract, existing, require_new_absent=True
        )


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


def test_wrapper_checks_local_roots_remote_then_delegates(tmp_path: Path) -> None:
    events: list[str] = []
    delegated = {}

    def execute(**kwargs):
        events.append("base")
        delegated.update(kwargs)
        return {"status": "done"}

    result = repair.execute_fresh16_claim_serialization_repair(
        **_execute_kwargs(tmp_path, object(), object()),
        original_failure_validator=lambda *_args: events.append("original") or {},
        inventory_failure_validator=lambda *_args: events.append("inventory") or {},
        new_roots_validator=lambda *_args: events.append("roots") or {},
        destination_validator=lambda *_args, **_kwargs: events.append("remote") or {},
        execute_fn=execute,
    )
    assert events == ["original", "inventory", "roots", "remote", "base"]
    assert result["status"] == "done"
    assert delegated["execution_preflight"] == result[
        "claim_serialization_repair_preflight"
    ]


def test_wrapper_validate_allows_new_roots_and_destination(tmp_path: Path) -> None:
    events: list[str] = []
    kwargs = _execute_kwargs(tmp_path, object(), object())
    kwargs["mode"] = "validate"
    repair.execute_fresh16_claim_serialization_repair(
        **kwargs,
        original_failure_validator=lambda *_args: events.append("original") or {},
        inventory_failure_validator=lambda *_args: events.append("inventory") or {},
        new_roots_validator=lambda *_args: events.append("roots") or {},
        destination_validator=lambda *_args, **kwargs: events.append(
            f"remote:{kwargs['require_new_absent']}"
        )
        or {},
        execute_fn=lambda **_kwargs: events.append("base") or {"status": "valid"},
    )
    assert events == ["original", "inventory", "remote:False", "base"]


def test_manager_b_and_local_failures_precede_token_and_hf(tmp_path: Path) -> None:
    from scripts import manage_gate_v1_fresh16_claim_serialization_repair as manager

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
    events: list[str] = []
    contract = SimpleNamespace(parent=object())
    with (
        mock.patch.object(
            manager,
            "load_frozen_fresh16_claim_serialization_repair_contract",
            return_value=contract,
        ),
        mock.patch.object(
            manager,
            "validate_execution_b_source",
            side_effect=lambda *_args, **_kwargs: events.append("b") or object(),
        ),
        mock.patch.object(
            manager,
            "validate_retained_v1_failure",
            side_effect=lambda *_args: events.append("original") or {},
        ),
        mock.patch.object(
            manager,
            "validate_retained_inventory_repair_failure",
            side_effect=lambda *_args: events.append("inventory")
            or (_ for _ in ()).throw(ValueError("inventory evidence drifted")),
        ),
        mock.patch.object(
            manager,
            "validate_new_run_roots_absent",
            side_effect=lambda *_args: events.append("roots") or {},
        ),
        mock.patch.object(
            manager.base_manager,
            "_secure_token",
            side_effect=lambda *_args: events.append("token") or "secret",
        ) as token,
        mock.patch.object(manager, "execute_fresh16_claim_serialization_repair") as execute,
    ):
        with pytest.raises(ValueError, match="inventory evidence drifted"):
            manager._run(args)
    assert events == ["b", "original", "inventory"]
    token.assert_not_called()
    execute.assert_not_called()


def test_manager_execution_b_failure_precedes_all_evidence_token_and_hf(
    tmp_path: Path,
) -> None:
    from scripts import manage_gate_v1_fresh16_claim_serialization_repair as manager

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
    contract = SimpleNamespace(parent=object())
    with (
        mock.patch.object(
            manager,
            "load_frozen_fresh16_claim_serialization_repair_contract",
            return_value=contract,
        ),
        mock.patch.object(
            manager,
            "validate_execution_b_source",
            side_effect=ValueError("Execution-B identity drifted"),
        ),
        mock.patch.object(manager, "validate_retained_v1_failure") as original,
        mock.patch.object(
            manager, "validate_retained_inventory_repair_failure"
        ) as inventory,
        mock.patch.object(manager, "validate_new_run_roots_absent") as roots,
        mock.patch.object(manager.base_manager, "_secure_token") as token,
        mock.patch.object(manager, "execute_fresh16_claim_serialization_repair") as execute,
    ):
        with pytest.raises(ValueError, match="Execution-B identity drifted"):
            manager._run(args)
    original.assert_not_called()
    inventory.assert_not_called()
    roots.assert_not_called()
    token.assert_not_called()
    execute.assert_not_called()


def test_manager_success_caches_all_local_preflights_before_token(tmp_path: Path) -> None:
    from scripts import manage_gate_v1_fresh16_claim_serialization_repair as manager

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
    contract = SimpleNamespace(parent=object())
    events: list[str] = []
    captured = {}
    with (
        mock.patch.object(
            manager,
            "load_frozen_fresh16_claim_serialization_repair_contract",
            return_value=contract,
        ),
        mock.patch.object(
            manager,
            "validate_execution_b_source",
            side_effect=lambda *_args, **_kwargs: events.append("b") or object(),
        ),
        mock.patch.object(
            manager,
            "validate_retained_v1_failure",
            side_effect=lambda *_args: events.append("original") or {"old": 1},
        ),
        mock.patch.object(
            manager,
            "validate_retained_inventory_repair_failure",
            side_effect=lambda *_args: events.append("inventory") or {"old": 2},
        ),
        mock.patch.object(
            manager,
            "validate_new_run_roots_absent",
            side_effect=lambda *_args: events.append("roots") or {"absent": True},
        ),
        mock.patch.object(
            manager.base_manager,
            "_secure_token",
            side_effect=lambda *_args: events.append("token") or "secret",
        ),
        mock.patch.object(
            manager,
            "execute_fresh16_claim_serialization_repair",
            side_effect=lambda **kwargs: captured.update(kwargs) or {"status": "done"},
        ),
        mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}, clear=False),
    ):
        assert manager._run(args) == {"status": "done"}
    assert events == ["b", "original", "inventory", "roots", "token"]
    assert captured["original_failure_validator"](contract, tmp_path) == {"old": 1}
    assert captured["inventory_failure_validator"](contract, tmp_path) == {"old": 2}
    assert captured["new_roots_validator"](contract, tmp_path) == {"absent": True}
