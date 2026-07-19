from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import causalcache.set_utility_processor_result_v2 as result_v2
from causalcache.set_utility_processor_artifacts import canonical_pretty_json_bytes
from causalcache.set_utility_processor_freeze_contract_v2 import (
    CANONICAL_EXECUTION_CONFIG_PATH,
    REQUIRED_IDENTITY_ARGUMENTS,
    REQUIRED_PATH_ARGUMENTS,
    REQUIRED_VERSION_ARGUMENTS_BY_PHASE,
    load_execution_contract,
)
from causalcache.set_utility_processor_postflight_v2 import (
    VALIDATION_STATUS,
    build_processor_freeze_postflight_context_v2,
)
from causalcache.set_utility_processor_result_v2 import (
    INVALID_PUBLICATION_STATUS,
    INVALID_RESULT_STATUS,
    PENDING_PUBLICATION_STATUS,
    VALID_RESULT_STATUS,
    record_processor_freeze_v2_result,
)


GIT_REVISION = "d" * 40
ROOT = Path(__file__).resolve().parents[2]
HF_REPO = "fixture-owner/causalcache-processor-freeze-v2"
HF_TAG = "phase1-b2-processor-freeze-v2"
ORIGINAL_VERIFY_MODULE_ORIGINS = result_v2._verify_module_origins


def _runtime(
    repo: Path,
    output: Path,
    *,
    git_revision: str,
) -> dict[str, str]:
    keys = {
        item.removeprefix("--").replace("-", "_")
        for item in (*REQUIRED_PATH_ARGUMENTS, *REQUIRED_IDENTITY_ARGUMENTS)
    }
    keys.update(
        item.removeprefix("--").replace("-", "_")
        for values in REQUIRED_VERSION_ARGUMENTS_BY_PHASE.values()
        for item in values
    )
    runtime = {key: f"fixture-{key}" for key in keys}
    for argument in REQUIRED_PATH_ARGUMENTS:
        key = argument.removeprefix("--").replace("-", "_")
        runtime[key] = str((output.parent / key).resolve())
    runtime.update(
        {
            "repository_root": str(repo.resolve()),
            "execution_config": str(
                (repo / CANONICAL_EXECUTION_CONFIG_PATH).resolve()
            ),
            "output_root": str(output.resolve()),
            "processor_python_executable": "/usr/bin/python3",
            "git_revision": git_revision,
            "host_alias": "hyper00",
            "host_hostname": "node-radixark-16-0000",
            "container_id": "a" * 64,
            "container_image_digest": "sha256:" + "b" * 64,
            "worker_count": "4",
            "ocr_python_version": "3.12.3",
            "ocr_runtime_version": "3.8.4",
            "pyarrow_version": "24.0.0",
            "processor_python_version": "3.12.3",
            "transformers_version": "5.6.0",
            "torch_version": "2.11.0+cu130",
            "pillow_version": "12.2.0",
        }
    )
    return runtime


def _postflight(
    config_sha: str,
    run_identity_sha: str,
    *,
    git_revision: str,
) -> dict[str, object]:
    return {
        "artifact_shards": [
            {
                "byte_count": worker + 1,
                "content_inventory_sha256": str(worker + 1) * 64,
                "filename": f"processor-substrate-worker-{worker:02d}.tar",
                "observation_count": 100,
                "sha256": str(worker + 5) * 64,
                "trajectory_count": 10,
                "worker_index": worker,
            }
            for worker in range(4)
        ],
        "candidate_artifact": {
            "byte_count": 42,
            "candidate_inventory_sha256": "9" * 64,
            "filename": "processor-candidate-freeze-schedule.json",
            "sha256": "8" * 64,
            "state_count": 2400,
            "subset_forward_count": 12345,
            "total_model_operations": 23456,
        },
        "execution_config_sha256": config_sha,
        "git_revision": git_revision,
        "global_format_mode_tally": {"PNG:RGB": 24, "PNG:RGBA": 18_768},
        "image_contract_sha256": "7" * 64,
        "metadata_only_ocr_validation_count": 17_592,
        "operation_budget": {
            "canonical_action_generations": 4800,
            "capped_candidate_teacher_forwards": 12345,
            "identical_reference_repeat_teacher_forwards": 2400,
            "kl_measurements": 14745,
            "raw_label_rows": 12345,
            "reference_teacher_forwards": 2400,
            "total_model_operations": 19545,
            "total_teacher_forwards": 17145,
        },
        "processor_freeze_summary_sha256": "6" * 64,
        "processor_image_contract_id": "fixture-image-contract-v2",
        "run_identity_sha256": run_identity_sha,
        "schema_version": "2.0.0",
        "stored_image_ocr_validation_count": 1200,
        "structural_validation_status": (
            "VALID_COMPLETED_SET_UTILITY_PROCESSOR_FREEZE"
        ),
        "status": VALIDATION_STATUS,
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    repo = tmp_path / "recorder-worktree"
    config = repo / CANONICAL_EXECUTION_CONFIG_PATH
    config.parent.mkdir(parents=True)
    config.write_bytes(b'{"fixture":true}\n')
    for relative in result_v2.RECORDER_SOURCE_PATHS:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    (repo / "data/results").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"], cwd=repo, check=True
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture recorder"], cwd=repo, check=True
    )
    recorder_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    config_sha = result_v2.sha256_bytes(config.read_bytes())
    producer_repo = tmp_path / "producer-worktree"
    producer_config = producer_repo / CANONICAL_EXECUTION_CONFIG_PATH
    producer_config.parent.mkdir(parents=True)
    producer_config.write_bytes(config.read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=producer_repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.com"],
        cwd=producer_repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"],
        cwd=producer_repo,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=producer_repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture producer"],
        cwd=producer_repo,
        check=True,
    )
    producer_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=producer_repo, text=True
    ).strip()
    output = tmp_path / (
        "causalcache-set-utility-processor-freeze-v2-image-contract-repair-"
        + producer_revision[:7]
    )
    for relative in sorted(result_v2._expected_formal_paths()):
        path = output.joinpath(*Path(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + "\n").encode("utf-8"))
    runtime = _runtime(
        producer_repo,
        output,
        git_revision=producer_revision,
    )
    run_identity_payload = canonical_pretty_json_bytes({"runtime_cli": runtime})
    (output / "run-identity.json").write_bytes(run_identity_payload)
    run_identity_sha = result_v2.sha256_bytes(run_identity_payload)

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    expected_argv = result_v2._expected_run_argv(runtime)
    argv = evidence / "formal.argv.txt"
    argv.write_text(
        shlex.join(["env", "-u", "PYTHONPATH", *expected_argv]) + "\n",
        encoding="utf-8",
    )
    start = evidence / "formal.start.json"
    start.write_bytes(
        canonical_pretty_json_bytes(
            {
                "container_id": runtime["container_id"],
                "git_revision": producer_revision,
                "host_alias": runtime["host_alias"],
                "host_hostname": runtime["host_hostname"],
                "output_root": str(output.resolve()),
                "shell_pid": 12345,
                "started_at_utc": "2026-07-19T05:00:00Z",
                "worker_count": 4,
            }
        )
    )
    outer = evidence / "formal.outer.log"
    outer.write_text("SECRET_RAW_LOG_MUST_NOT_BE_COPIED\n", encoding="utf-8")
    exit_code = evidence / "formal.exit_code"
    exit_code.write_text("0\n", encoding="ascii")
    end = evidence / "formal.end.json"
    end.write_bytes(
        canonical_pretty_json_bytes(
            {
                "ended_at_utc": "2026-07-19T05:10:00Z",
                "exit_code": 0,
                "output_root": str(output.resolve()),
                "watcher_pid": 12346,
            }
        )
    )
    postflight_argv = evidence / "formal.postflight.argv.txt"
    expected_postflight_argv = result_v2._expected_postflight_argv(
        producer_repo,
        producer_config,
        output,
        producer_revision,
        runtime,
    )
    postflight_argv.write_text(
        shlex.join(
            ["env", "-u", "PYTHONPATH", *expected_postflight_argv]
        )
        + "\n",
        encoding="utf-8",
    )
    postflight = evidence / "formal.postflight.json"
    postflight_summary = _postflight(
        config_sha,
        run_identity_sha,
        git_revision=producer_revision,
    )
    postflight.write_bytes(canonical_pretty_json_bytes(postflight_summary))

    monkeypatch.setattr(
        result_v2,
        "load_execution_contract",
        lambda **_: SimpleNamespace(config_sha256=config_sha),
    )
    monkeypatch.setattr(
        result_v2,
        "build_processor_freeze_postflight_context_v2",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        result_v2,
        "validate_completed_processor_freeze_root_v2",
        lambda *_args, **_kwargs: dict(postflight_summary),
    )
    monkeypatch.setattr(result_v2, "_verify_module_origins", lambda _root: None)
    return {
        "repo": repo,
        "producer_repo": producer_repo,
        "config": producer_config,
        "recorder_revision": recorder_revision,
        "producer_revision": producer_revision,
        "output": output,
        "result": repo / "data/results/processor-v2-formal",
        "argv": argv,
        "start": start,
        "outer": outer,
        "exit_code": exit_code,
        "end": end,
        "postflight_argv": postflight_argv,
        "postflight": postflight,
        "postflight_summary": postflight_summary,
    }


def _record(fixture: dict[str, object], **overrides: object):
    arguments: dict[str, object] = {
        "repository_root": fixture["repo"],
        "producer_repository_root": fixture["producer_repo"],
        "execution_config": fixture["config"],
        "output_root": fixture["output"],
        "expected_git_revision": fixture["producer_revision"],
        "expected_recorder_git_revision": fixture["recorder_revision"],
        "result_dir": fixture["result"],
        "run_argv_evidence": fixture["argv"],
        "start_evidence": fixture["start"],
        "outer_log": fixture["outer"],
        "exit_code_evidence": fixture["exit_code"],
        "end_evidence": fixture["end"],
        "postflight_argv_evidence": fixture["postflight_argv"],
        "postflight_evidence": fixture["postflight"],
        "intended_hf_repo": HF_REPO,
        "intended_hf_tag": HF_TAG,
    }
    arguments.update(overrides)
    return record_processor_freeze_v2_result(**arguments)


def test_records_valid_git_safe_result_with_exact_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    summary = _record(fixture)

    result = fixture["result"]
    assert isinstance(result, Path)
    assert {entry.name for entry in os.scandir(result)} == {
        "README.md",
        "summary.json",
    }
    assert summary["status"] == VALID_RESULT_STATUS
    assert summary["scientific_eligibility"] is True
    assert summary["postflight_accepted_for_formal_result"] is True
    assert summary["publication"]["status"] == PENDING_PUBLICATION_STATUS
    assert summary["artifact"]["file_count"] == 23
    assert summary["artifact"]["staging_root_present_after_success"] is False
    assert len(summary["artifact"]["file_inventory"]) == 23
    assert summary["postflight"]["operation_budget"]["raw_label_rows"] == 12345
    assert summary["postflight"]["global_format_mode_tally"] == {
        "PNG:RGB": 24,
        "PNG:RGBA": 18_768,
    }
    assert summary["bindings"]["image_contract_sha256"] == "7" * 64
    assert summary["bindings"]["recorder_git_revision"] == fixture[
        "recorder_revision"
    ]
    assert set(summary["bindings"]["recorder_sources"]) == set(
        result_v2.RECORDER_SOURCE_PATHS
    )
    assert summary["execution"]["evidence"]["run_argv"]["sha256"]
    assert summary["execution"]["evidence"]["postflight_argv"]["sha256"]
    assert summary["execution"]["ended_at_utc"] == "2026-07-19T05:10:00Z"
    assert summary["execution"]["elapsed_seconds"] == 600
    rendered = (result / "summary.json").read_text(encoding="utf-8")
    rendered += (result / "README.md").read_text(encoding="utf-8")
    assert "SECRET_RAW_LOG_MUST_NOT_BE_COPIED" not in rendered
    assert all(value == 0 for value in summary["negative_operations"].values())


def test_postflight_evidence_mismatch_fails_without_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    postflight = fixture["postflight"]
    assert isinstance(postflight, Path)
    payload = json.loads(postflight.read_text(encoding="utf-8"))
    payload["run_identity_sha256"] = "0" * 64
    postflight.write_bytes(canonical_pretty_json_bytes(payload))

    with pytest.raises(ValueError, match="differs from fresh"):
        _record(fixture)

    assert not Path(fixture["result"]).exists()


def test_invalid_record_never_claims_valid_or_pending_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    exit_code = fixture["exit_code"]
    assert isinstance(exit_code, Path)
    exit_code.write_text("17\n", encoding="ascii")
    end = fixture["end"]
    assert isinstance(end, Path)
    end_value = json.loads(end.read_text(encoding="utf-8"))
    end_value["exit_code"] = 17
    end.write_bytes(canonical_pretty_json_bytes(end_value))

    summary = _record(fixture, record_invalid=True)

    assert summary["status"] == INVALID_RESULT_STATUS
    assert summary["scientific_eligibility"] is False
    assert summary["postflight_accepted_for_formal_result"] is False
    assert summary["publication"]["status"] == INVALID_PUBLICATION_STATUS
    assert summary["failure"] == {
        "error_code": "FORMAL_EXIT_NONZERO",
        "exception_type": "ValueError",
        "stage": "formal_exit",
    }
    assert summary["execution"]["exit_code"] == 17


def test_end_evidence_exit_code_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    end = fixture["end"]
    assert isinstance(end, Path)
    value = json.loads(end.read_text(encoding="utf-8"))
    value["exit_code"] = 9
    end.write_bytes(canonical_pretty_json_bytes(value))

    with pytest.raises(ValueError, match="differs from exit-code evidence"):
        _record(fixture)


def test_end_evidence_cannot_precede_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    end = fixture["end"]
    assert isinstance(end, Path)
    value = json.loads(end.read_text(encoding="utf-8"))
    value["ended_at_utc"] = "2026-07-19T04:59:59Z"
    end.write_bytes(canonical_pretty_json_bytes(value))

    with pytest.raises(ValueError, match="precedes start evidence"):
        _record(fixture)


def test_argv_text_must_match_run_identity_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    argv = fixture["argv"]
    assert isinstance(argv, Path)
    argv.write_text(
        argv.read_text(encoding="utf-8").replace("--worker-count 4", "--worker-count 3"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="differs from the formal run identity"):
        _record(fixture)


def test_start_evidence_requires_exact_eight_key_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    start = fixture["start"]
    assert isinstance(start, Path)
    value = json.loads(start.read_text(encoding="utf-8"))
    value["extra"] = "not allowed"
    start.write_bytes(canonical_pretty_json_bytes(value))

    with pytest.raises(ValueError, match="fields drifted"):
        _record(fixture)


def test_formal_tree_requires_exact_23_file_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = fixture["output"]
    assert isinstance(output, Path)
    (output / "unexpected.bin").write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="23-file inventory drifted"):
        _record(fixture)


def test_formal_tree_mutation_during_postflight_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = fixture["output"]
    assert isinstance(output, Path)

    def mutate(*_args: object, **_kwargs: object) -> dict[str, object]:
        (output / "logs/ocr-worker-00.log").write_bytes(b"mutation")
        return dict(fixture["postflight_summary"])

    monkeypatch.setattr(
        result_v2,
        "validate_completed_processor_freeze_root_v2",
        mutate,
    )

    with pytest.raises(RuntimeError, match="changed during result recording"):
        _record(fixture)


def test_sibling_staging_root_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = fixture["output"]
    assert isinstance(output, Path)
    staging = output.parent / f".{output.name}.incomplete"
    staging.mkdir()

    with pytest.raises(ValueError, match="staging root remains"):
        _record(fixture)


def test_result_dir_must_be_new_and_under_git_data_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside-result"

    with pytest.raises(ValueError, match="under data/results"):
        _record(fixture, result_dir=outside)

    existing = fixture["result"]
    assert isinstance(existing, Path)
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        _record(fixture)


def test_evidence_symlink_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    linked = tmp_path / "outer-link.log"
    linked.symlink_to(fixture["outer"])

    with pytest.raises(ValueError, match="regular non-symlink"):
        _record(fixture, outer_log=linked)


def test_real_context_uses_distinct_producer_checkout(tmp_path: Path) -> None:
    producer = tmp_path / "producer"
    archive = subprocess.check_output(["git", "archive", "HEAD"], cwd=ROOT)
    producer.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(producer, filter="data")
    config = producer / CANONICAL_EXECUTION_CONFIG_PATH

    contract = load_execution_contract(
        repository_root=producer,
        execution_config_path=config,
    )
    context = build_processor_freeze_postflight_context_v2(
        contract,
        expected_git_revision=GIT_REVISION,
    )

    assert producer.resolve() != ROOT.resolve()
    assert context.structural_context.repository_root == producer.resolve()
    assert context.structural_context.execution_config_path == config.resolve()


def test_recorder_checkout_requires_exact_clean_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="HEAD differs"):
        _record(
            fixture,
            expected_recorder_git_revision="0" * 40,
        )

    repo = fixture["repo"]
    assert isinstance(repo, Path)
    (repo / "dirty-untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completely clean"):
        _record(fixture)


def test_producer_checkout_requires_exact_clean_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="producer checkout HEAD differs"):
        _record(fixture, expected_git_revision="0" * 40)

    producer = fixture["producer_repo"]
    assert isinstance(producer, Path)
    (producer / "dirty-untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="producer checkout must be completely clean"):
        _record(fixture)


def test_actual_module_origin_rejects_unrelated_clean_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        result_v2,
        "_verify_module_origins",
        ORIGINAL_VERIFY_MODULE_ORIGINS,
    )

    with pytest.raises(ValueError, match="module origin differs"):
        _record(fixture)


def test_cli_rejects_repository_root_mismatch(tmp_path: Path) -> None:
    fake = tmp_path / "fake-root"
    command = [
        sys.executable,
        str(ROOT / "code/scripts/record_set_utility_processor_freeze_v2_result.py"),
        "--repository-root",
        str(fake),
        "--producer-repository-root",
        str(fake),
        "--execution-config",
        str(fake / "config.json"),
        "--output-root",
        str(fake / "output"),
        "--expected-git-revision",
        "0" * 40,
        "--expected-recorder-git-revision",
        "1" * 40,
        "--result-dir",
        str(fake / "result"),
        "--run-argv-evidence",
        str(fake / "argv.txt"),
        "--start-evidence",
        str(fake / "start.json"),
        "--outer-log",
        str(fake / "outer.log"),
        "--exit-code-evidence",
        str(fake / "exit"),
        "--end-evidence",
        str(fake / "end.json"),
        "--postflight-argv-evidence",
        str(fake / "postflight.argv.txt"),
        "--postflight-evidence",
        str(fake / "postflight.json"),
        "--intended-hf-repo",
        HF_REPO,
        "--intended-hf-tag",
        HF_TAG,
    ]

    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "--repository-root differs" in completed.stderr


def test_postflight_argv_evidence_must_match_producer_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    argv = fixture["postflight_argv"]
    assert isinstance(argv, Path)
    producer_revision = fixture["producer_revision"]
    assert isinstance(producer_revision, str)
    argv.write_text(
        argv.read_text(encoding="utf-8").replace(
            "--expected-git-revision " + producer_revision,
            "--expected-git-revision " + "0" * 40,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="producer-bound validator"):
        _record(fixture)


def test_invalid_extra_filename_is_not_leaked_to_git_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = fixture["output"]
    assert isinstance(output, Path)
    secret_filename = "SECRET_FILENAME_MUST_NOT_LEAK.bin"
    (output / secret_filename).write_bytes(b"invalid")

    summary = _record(fixture, record_invalid=True)

    assert summary["failure"] == {
        "error_code": "FORMAL_TREE_PREFLIGHT_INVALID",
        "exception_type": "ValueError",
        "stage": "formal_tree_before",
    }
    result = fixture["result"]
    assert isinstance(result, Path)
    rendered = (result / "summary.json").read_text(encoding="utf-8")
    rendered += (result / "README.md").read_text(encoding="utf-8")
    assert secret_filename not in rendered


def test_run_identity_sha_must_match_formal_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = fixture["output"]
    assert isinstance(output, Path)
    original_snapshot = result_v2._formal_tree_snapshot
    mutated = False

    def mutate_before_inventory(root: Path):
        nonlocal mutated
        if not mutated:
            mutated = True
            identity = root / "run-identity.json"
            identity.write_bytes(identity.read_bytes() + b" ")
        return original_snapshot(root)

    monkeypatch.setattr(
        result_v2,
        "_formal_tree_snapshot",
        mutate_before_inventory,
    )

    with pytest.raises(RuntimeError, match="run identity changed"):
        _record(fixture)
