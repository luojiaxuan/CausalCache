from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import launch_set_utility_direct_selection_workers as launcher
from scripts import materialize_set_utility_selection_assignments as assignments


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _state(shard: int, *, state_id: str | None = None, role: str = "train") -> dict:
    return {
        "candidate_event_step_ids": [1, 2, 3, 4, 5],
        "logical_shard": shard,
        "role": role,
        "state_id": state_id or f"trajectory-{shard:03d}:decision:006",
    }


def _write_input(root: Path, states: list[dict]) -> None:
    root.mkdir()
    payload = "".join(
        json.dumps(state, sort_keys=True) + "\n" for state in states
    )
    (root / "states.jsonl").write_text(payload, encoding="utf-8")
    _write_json(
        root / "manifest.json",
        {
            "content_sha256": "input-content-sha",
            "states_jsonl": "states.jsonl",
        },
    )


def _worker(
    worker_id: int,
    host: str,
    gpu: int,
    shards: list[int],
) -> dict:
    return {
        "host": host,
        "logical_shard_ids": shards,
        "physical_gpu_id": gpu,
        "worker_id": worker_id,
    }


def _run_assignments(
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_root: Path,
    plan: Path,
    output_root: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_set_utility_selection_assignments.py",
            "--input-root",
            str(input_root),
            "--rollout-plan",
            str(plan),
            "--output-root",
            str(output_root),
            "--role",
            "train",
        ],
    )
    assignments.main()


def test_assignment_materialization_is_complete_and_preserves_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root = tmp_path / "input"
    _write_input(input_root, [_state(shard) for shard in range(256)])
    plan_path = tmp_path / "plan.json"
    plan = {
        "logical_shard_count": 256,
        "workers": [
            _worker(4, "hyper00", 5, list(range(128))),
            _worker(9, "hyper01", 5, list(range(128, 256))),
        ],
    }
    _write_json(plan_path, plan)
    output_root = tmp_path / "assignments"

    _run_assignments(
        monkeypatch,
        input_root=input_root,
        plan=plan_path,
        output_root=output_root,
    )

    printed = json.loads(capsys.readouterr().out)
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert printed == manifest
    assert manifest["state_count"] == 256
    assert manifest["input_content_sha256"] == "input-content-sha"
    receipts = {row["worker_id"]: row for row in manifest["workers"]}
    assert {
        worker_id: (row["host"], row["physical_gpu_id"], row["state_count"])
        for worker_id, row in receipts.items()
    } == {4: ("hyper00", 5, 128), 9: ("hyper01", 5, 128)}
    assigned_ids: list[str] = []
    for worker_id, receipt in receipts.items():
        payload = (output_root / receipt["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == receipt["file_sha256"]
        state_ids = payload.decode().splitlines()
        assert state_ids == sorted(state_ids)
        assert len(state_ids) == len(set(state_ids)) == 128
        assert receipt["candidate_event_count"] == 128 * 5
        assigned_ids.extend(state_ids)
        assert receipt["worker_id"] == worker_id
    assert sorted(assigned_ids) == sorted(
        state["state_id"] for state in [_state(shard) for shard in range(256)]
    )
    unsigned = dict(manifest)
    content_sha256 = unsigned.pop("content_sha256")
    unsigned_payload = json.dumps(
        unsigned, allow_nan=False, ensure_ascii=False, sort_keys=True
    ).encode()
    assert hashlib.sha256(unsigned_payload).hexdigest() == content_sha256


@pytest.mark.parametrize(
    "workers, error",
    [
        (
            [
                _worker(0, "hyper00", 0, list(range(128))),
                _worker(0, "hyper01", 0, list(range(128, 256))),
            ],
            "duplicates a worker",
        ),
        (
            [
                _worker(0, "hyper00", 0, list(range(128))),
                _worker(1, "hyper01", 0, [127, *range(128, 256)]),
            ],
            "duplicates a logical shard",
        ),
        (
            [
                _worker(0, "hyper00", 0, list(range(128))),
                _worker(1, "hyper00", 0, list(range(128, 256))),
            ],
            "duplicates a host/GPU mapping",
        ),
        (
            [
                _worker(0, "hyper00", 0, list(range(128))),
                {
                    **_worker(1, "hyper01", 0, list(range(128, 256))),
                    "logical_shard_ids": ["128", *range(129, 256)],
                },
            ],
            "invalid logical shard",
        ),
    ],
)
def test_assignment_plan_rejects_duplicate_workers_shards_and_host_gpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workers: list[dict],
    error: str,
) -> None:
    input_root = tmp_path / "input"
    _write_input(input_root, [_state(shard) for shard in range(256)])
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, {"logical_shard_count": 256, "workers": workers})
    output_root = tmp_path / "assignments"

    with pytest.raises(ValueError, match=error):
        _run_assignments(
            monkeypatch,
            input_root=input_root,
            plan=plan_path,
            output_root=output_root,
        )
    assert not output_root.exists()


def test_assignment_plan_rejects_empty_worker_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    _write_input(input_root, [_state(shard) for shard in range(255)])
    plan_path = tmp_path / "plan.json"
    _write_json(
        plan_path,
        {
            "logical_shard_count": 256,
            "workers": [
                _worker(0, "hyper00", 0, list(range(255))),
                _worker(1, "hyper00", 1, [255]),
            ],
        },
    )
    output_root = tmp_path / "assignments"

    with pytest.raises(ValueError, match="empty worker"):
        _run_assignments(
            monkeypatch,
            input_root=input_root,
            plan=plan_path,
            output_root=output_root,
        )
    assert not output_root.exists()


def test_assignment_rejects_duplicate_input_state_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    states = [_state(shard) for shard in range(256)]
    states.append(_state(1, state_id=states[0]["state_id"]))
    _write_input(input_root, states)
    plan_path = tmp_path / "plan.json"
    _write_json(
        plan_path,
        {
            "logical_shard_count": 256,
            "workers": [_worker(0, "hyper00", 0, list(range(256)))],
        },
    )
    output_root = tmp_path / "assignments"

    with pytest.raises(ValueError, match="duplicate state"):
        _run_assignments(
            monkeypatch,
            input_root=input_root,
            plan=plan_path,
            output_root=output_root,
        )
    assert not output_root.exists()


@pytest.mark.parametrize(
    "value",
    ["", "0:1", "gpu:1:/tmp/states", "0:worker:/tmp/states", "-1:0:/tmp/states"],
)
def test_launcher_job_parser_rejects_malformed_values(value: str) -> None:
    with pytest.raises(Exception, match="job must be"):
        launcher._job(value)


def test_launcher_job_parser_preserves_colons_in_state_path() -> None:
    assert launcher._job("3:7:/tmp/state:ids.txt") == (
        3,
        7,
        Path("/tmp/state:ids.txt"),
    )


class _FakeProcess:
    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code

    def wait(self) -> int:
        return self.exit_code


def _launcher_argv(tmp_path: Path, jobs: list[str]) -> list[str]:
    return [
        "launch_set_utility_direct_selection_workers.py",
        "--input-root",
        str(tmp_path / "input"),
        "--cache-root",
        str(tmp_path / "cache"),
        "--training-config",
        str(tmp_path / "training.json"),
        "--variant",
        "set_transformer",
        "--training-summary",
        str(tmp_path / "training-summary.json"),
        "--checkpoint",
        str(tmp_path / "best.safetensors"),
        "--collection-config",
        str(tmp_path / "collection.json"),
        "--output-root",
        str(tmp_path / "output"),
        *[item for job in jobs for item in ("--job", job)],
    ]


def test_launcher_builds_isolated_gpu_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "worker-02.txt"
    second = tmp_path / "worker-07.txt"
    first.write_text("state-a\n", encoding="utf-8")
    second.write_text("state-b\n", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        calls.append({"command": command, **kwargs})
        return _FakeProcess()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        sys,
        "argv",
        _launcher_argv(
            tmp_path,
            [f"4:2:{first}", f"1:7:{second}"],
        ),
    )

    launcher.main()

    assert json.loads(capsys.readouterr().out) == {
        "exit_codes": {"2": 0, "7": 0},
        "status": "COMPLETED",
    }
    assert len(calls) == 2
    for call, gpu, worker, state_file in zip(
        calls,
        (4, 1),
        (2, 7),
        (first, second),
        strict=True,
    ):
        command = call["command"]
        assert command[0] == sys.executable
        assert Path(command[1]).name == (
            "run_set_utility_direct_marginal_tune_selectors.py"
        )
        assert command[command.index("--role") + 1] == "train"
        assert command[command.index("--state-id-file") + 1] == str(state_file)
        assert command[command.index("--output") + 1] == str(
            tmp_path / "output" / f"worker-{worker:02d}.json"
        )
        assert command[command.index("--device") + 1] == "cuda:0"
        assert call["env"]["CUDA_VISIBLE_DEVICES"] == str(gpu)
        assert call["env"]["PYTHONPATH"].split(os.pathsep)[0].endswith("/code")
        assert call["stderr"] is subprocess.STDOUT
        assert call["stdout"].closed


@pytest.mark.parametrize(
    "jobs, error",
    [
        (["0:2:{path}", "0:7:{other}"], "unique GPUs and workers"),
        (["0:2:{path}", "1:2:{other}"], "unique GPUs and workers"),
    ],
)
def test_launcher_rejects_duplicate_jobs_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    jobs: list[str],
    error: str,
) -> None:
    path = tmp_path / "states-a.txt"
    other = tmp_path / "states-b.txt"
    path.write_text("state-a\n", encoding="utf-8")
    other.write_text("state-b\n", encoding="utf-8")
    formatted = [item.format(path=path, other=other) for item in jobs]
    monkeypatch.setattr(sys, "argv", _launcher_argv(tmp_path, formatted))

    with pytest.raises(ValueError, match=error):
        launcher.main()
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("payload", ["", "state-a\nstate-a\n"])
def test_launcher_rejects_empty_or_duplicate_worker_state_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
) -> None:
    state_file = tmp_path / "states.txt"
    state_file.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        _launcher_argv(tmp_path, [f"0:2:{state_file}"]),
    )

    with pytest.raises(ValueError, match="non-empty and unique"):
        launcher.main()
    assert not (tmp_path / "output").exists()
