from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_independent_confirm_continuation as run_module
from scripts.run_independent_confirm_continuation import (
    _unwrap_fresh_topology_envelope,
    isolated_model_snapshot_identity,
)
from scripts.smoke_independent_confirm_gpu_topology import (
    build_continuation_topology_envelope,
    continuation_topology_challenge,
    continuation_topology_challenge_response,
)


class _Queue:
    def __init__(self, response):
        self.response = response
        self.closed = False

    def get(self, timeout):
        assert timeout == 17
        return self.response

    def close(self):
        self.closed = True

    def join_thread(self):
        assert self.closed


class _Process:
    def __init__(self, *, target, kwargs, name):
        self.target = target
        self.kwargs = kwargs
        self.name = name
        self.exitcode = None
        self.started = False

    def start(self):
        self.started = True
        self.exitcode = 0

    def join(self, timeout):
        assert timeout >= 0

    def is_alive(self):
        return False


class _Context:
    def __init__(self, response):
        self.queue = _Queue(response)
        self.process = None

    def Queue(self):
        return self.queue

    def Process(self, **kwargs):
        self.process = _Process(**kwargs)
        return self.process


def test_isolated_model_verifier_accepts_only_plain_spawn_result(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}")
    context = _Context(
        {
            "ok": True,
            "snapshot_manifest": str(snapshot),
            "identity": {"model_repo": "mPLUG/GUI-Owl-1.5-8B-Instruct"},
        }
    )

    observed, identity = isolated_model_snapshot_identity(
        repository_root=tmp_path,
        parent_contract_path=tmp_path / "parent.json",
        model_dir=tmp_path / "model",
        timeout_seconds=17,
        context=context,
    )

    assert observed == snapshot
    assert identity["model_repo"] == "mPLUG/GUI-Owl-1.5-8B-Instruct"
    assert context.process is not None and context.process.started
    assert context.process.name == "causalcache-confirm-continuation-model-verification"


def test_isolated_model_verifier_rejects_child_failure(tmp_path: Path) -> None:
    context = _Context(
        {"ok": False, "exception_type": "RuntimeError", "message": "bad snapshot"}
    )

    with pytest.raises(RuntimeError, match="isolated model verification failed"):
        isolated_model_snapshot_identity(
            repository_root=tmp_path,
            parent_contract_path=tmp_path / "parent.json",
            model_dir=tmp_path / "model",
            timeout_seconds=17,
            context=context,
        )


def test_fresh_topology_envelope_binds_execution_b_and_nonce() -> None:
    devices = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    gpu_uuids = tuple(
        f"GPU-{index:08x}-0000-0000-0000-000000000000" for index in range(4)
    )
    workers = [
        {
            "worker_id": f"worker-{index}",
            "device": device,
            "gpu_uuid": gpu_uuid,
            "runtime_identity": {"model": "fixture"},
            "operation_counts": {"forward": 1},
        }
        for index, (device, gpu_uuid) in enumerate(
            zip(devices, gpu_uuids, strict=True)
        )
    ]
    parent = {
        "status": "PASS",
        "inputs": {
            "model_dir": "/model",
            "snapshot_manifest": "/snapshot.json",
        },
        "phases": {
            "policy_vision_spawn": {"workers": workers},
            "teacher_forced_fork": {"workers": workers},
        },
    }
    challenge = continuation_topology_challenge(
        execution_b_commit="b" * 40,
        topology_nonce="c" * 64,
        model_dir="/model",
        snapshot_manifest="/snapshot.json",
        devices=devices,
        gpu_uuids=gpu_uuids,
    )
    responses = {
        phase: [
            {
                "worker_id": worker["worker_id"],
                "response_sha256": continuation_topology_challenge_response(
                    challenge_sha256=challenge,
                    phase=phase,
                    worker_record=worker,
                ),
            }
            for worker in workers
        ]
        for phase in ("policy_vision_spawn", "teacher_forced_fork")
    }
    envelope = build_continuation_topology_envelope(
        parent,
        execution_b_commit="b" * 40,
        topology_nonce="c" * 64,
        challenge_sha256=challenge,
        challenge_responses=responses,
    )

    replay, digest, observed_challenge, responses_sha256 = (
        _unwrap_fresh_topology_envelope(
            envelope,
            execution_b_commit="b" * 40,
            topology_nonce="c" * 64,
            expected_devices=devices,
            expected_gpu_uuids=gpu_uuids,
            forbidden_parent_receipt_sha256="e" * 64,
        )
    )

    assert dict(replay) == parent
    assert digest == envelope["parent_receipt_file_sha256"]
    assert observed_challenge == challenge
    assert len(responses_sha256) == 64
    tampered = dict(envelope)
    tampered["topology_nonce"] = "d" * 64
    with pytest.raises(ValueError, match="topology envelope"):
        _unwrap_fresh_topology_envelope(
            tampered,
            execution_b_commit="b" * 40,
            topology_nonce="c" * 64,
            expected_devices=devices,
            expected_gpu_uuids=gpu_uuids,
            forbidden_parent_receipt_sha256="e" * 64,
        )

    old_receipt = dict(envelope)
    old_receipt_sha256 = old_receipt["parent_receipt_file_sha256"]
    with pytest.raises(ValueError, match="topology envelope"):
        _unwrap_fresh_topology_envelope(
            old_receipt,
            execution_b_commit="b" * 40,
            topology_nonce="c" * 64,
            expected_devices=devices,
            expected_gpu_uuids=gpu_uuids,
            forbidden_parent_receipt_sha256=old_receipt_sha256,
        )


def test_formal_execute_persists_tripwire_failure_before_external_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    output = external / "continuation-output"
    contract = SimpleNamespace(
        sha256="a" * 64,
        data={
            "protocol_id": "causalcache_independent_confirm20_restoration_continuation_v1",
            "output_contract": {
                "failure_status": "INVALID_INDEPENDENT_CONFIRM20_CONTINUATION_V1"
            },
            "failed_attempt": {
                "status": "INVALID_INDEPENDENT_CONFIRM20_V1_CUDA_FORK_BOUNDARY",
                "topology_receipt_sha256": "e" * 64,
            },
        },
    )
    parent_contract = SimpleNamespace()
    monkeypatch.setattr(
        run_module,
        "_load_runner_authorization",
        lambda **_kwargs: {"runner_freeze": {"topology_receipt_nonce": "c" * 64}},
    )
    monkeypatch.setattr(
        run_module.IndependentConfirmContinuationContract,
        "load",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        run_module.IndependentConfirmContract,
        "load",
        lambda *_args, **_kwargs: parent_contract,
    )
    monkeypatch.setattr(
        run_module,
        "_validate_runtime_args",
        lambda *_args, **_kwargs: (
            ("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
            ("GPU-a", "GPU-b", "GPU-c", "GPU-d"),
        ),
    )
    monkeypatch.setattr(
        run_module,
        "install_fork_parent_cuda_tripwire",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("torch preloaded")),
    )
    args = SimpleNamespace(
        repository_root=root,
        contract=root / run_module.CONFIG_PATH,
        runner_freeze=root / run_module.RUNNER_FREEZE_PATH,
        execution_b_git_commit="b" * 40,
        parent_root=external / "parent",
        generator_source_root=external / "generator",
        model_dir=external / "model",
        ocr_model_dir=external / "ocr-model",
        ocr_wheel_dir=external / "ocr-wheels",
        hf_cache_dir=external / "hf-cache",
        output_dir=output,
        hf_token_file=external / "token",
        execution_argv=["--formal-fixture"],
    )

    with pytest.raises(RuntimeError, match="torch preloaded"):
        run_module.execute(args)

    assert (output / "attempt.json").is_file()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["failed_stage"] == "install_cuda_parent_tripwire"
    assert failure["remote_mutation_count"] == 0
    with pytest.raises(FileExistsError, match="already exists"):
        run_module.execute(args)
