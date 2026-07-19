from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "code/scripts/watch_set_utility_processor_freeze_v2_result.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_causalcache_processor_result_watcher_v2", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load processor result watcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _arguments(tmp_path: Path) -> argparse.Namespace:
    repository = tmp_path / "recorder"
    producer = tmp_path / "producer"
    evidence = tmp_path / "evidence"
    (repository / "data/results").mkdir(parents=True)
    producer.mkdir()
    evidence.mkdir()
    values: dict[str, object] = {
        "recorder_python_executable": Path("/usr/bin/python3"),
        "repository_root": repository,
        "producer_repository_root": producer,
        "execution_config": producer / "config.json",
        "output_root": tmp_path / "formal-output",
        "expected_git_revision": "a" * 40,
        "expected_recorder_git_revision": "b" * 40,
        "result_dir": repository / "data/results/formal-result",
        "intended_hf_repo": "owner/private-dataset",
        "intended_hf_tag": "processor-v2",
        "poll_seconds": 0.001,
        "timeout_seconds": 0.02,
    }
    filenames = {
        "run_argv_evidence": "argv.txt",
        "start_evidence": "start.json",
        "outer_log": "outer.log",
        "exit_code_evidence": "exit_code",
        "end_evidence": "end.json",
        "postflight_argv_evidence": "postflight-argv.txt",
        "postflight_evidence": "postflight.json",
        "postflight_exit_code_evidence": "postflight-exit-code",
        "supervisor_evidence": "supervisor.json",
    }
    values.update({key: evidence / filename for key, filename in filenames.items()})
    return argparse.Namespace(**values)


def _write_success_evidence(args: argparse.Namespace) -> None:
    Path(args.exit_code_evidence).write_bytes(b"0\n")
    Path(args.postflight_exit_code_evidence).write_bytes(b"0\n")
    Path(args.supervisor_evidence).write_bytes(
        _canonical_json(
            {
                "formal_exit_code": 0,
                "postflight_exit_code": 0,
                "supervisor_ended_at_utc": "2026-07-19T08:00:00Z",
                "supervisor_pid": 12345,
            }
        )
    )


def test_wait_accepts_only_terminal_double_zero(tmp_path: Path) -> None:
    module = _load_script()
    args = _arguments(tmp_path)
    _write_success_evidence(args)

    module._wait_for_success(args)


@pytest.mark.parametrize(
    ("formal_exit", "postflight_exit"),
    [(1, 0), (0, 1), (0, None)],
)
def test_terminal_failure_never_reaches_recorder(
    tmp_path: Path,
    formal_exit: int,
    postflight_exit: int | None,
) -> None:
    module = _load_script()
    args = _arguments(tmp_path)
    Path(args.supervisor_evidence).write_bytes(
        _canonical_json(
            {
                "formal_exit_code": formal_exit,
                "postflight_exit_code": postflight_exit,
                "supervisor_ended_at_utc": "2026-07-19T08:00:00Z",
                "supervisor_pid": 12345,
            }
        )
    )

    with pytest.raises((RuntimeError, ValueError)):
        module._wait_for_success(args)
    assert not Path(args.result_dir).exists()


def test_noncanonical_or_symlinked_supervisor_is_rejected(
    tmp_path: Path,
) -> None:
    module = _load_script()
    args = _arguments(tmp_path)
    Path(args.supervisor_evidence).write_text(
        '{"formal_exit_code": 0, "postflight_exit_code": 0, '
        '"supervisor_ended_at_utc": "2026-07-19T08:00:00Z", '
        '"supervisor_pid": 12345}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not canonical"):
        module._wait_for_success(args)

    Path(args.supervisor_evidence).unlink()
    target = tmp_path / "supervisor-target.json"
    target.write_bytes(
        _canonical_json(
            {
                "formal_exit_code": 0,
                "postflight_exit_code": 0,
                "supervisor_ended_at_utc": "2026-07-19T08:00:00Z",
                "supervisor_pid": 12345,
            }
        )
    )
    Path(args.supervisor_evidence).symlink_to(target)
    with pytest.raises(ValueError, match="regular non-symlink"):
        module._wait_for_success(args)


def test_timeout_and_frozen_evidence_layout_are_fail_closed(
    tmp_path: Path,
) -> None:
    module = _load_script()
    args = _arguments(tmp_path)
    with pytest.raises(TimeoutError, match="timed out"):
        module._wait_for_success(args)
    assert not Path(args.result_dir).exists()

    args.postflight_evidence = Path(args.postflight_evidence).with_name(
        "renamed-postflight.json"
    )
    with pytest.raises(ValueError, match="frozen filename"):
        module._validate_evidence_layout(args)


def test_main_invokes_exact_recorder_without_invalid_mode_or_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    args = _arguments(tmp_path)
    _write_success_evidence(args)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        calls.append((command, environment))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(module, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setenv("PYTHONPATH", "/must/not/reach/recorder")

    assert module.main([]) == 0
    assert len(calls) == 1
    command, environment = calls[0]
    assert command[0] == "/usr/bin/python3"
    assert command[1].endswith(
        "/code/scripts/record_set_utility_processor_freeze_v2_result.py"
    )
    assert "--record-invalid" not in command
    assert "--postflight-exit-code-evidence" not in command
    assert "--supervisor-evidence" not in command
    assert environment.get("PYTHONPATH") is None
