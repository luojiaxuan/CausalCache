from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import causalcache.set_utility_throughput_pilot_envelope_v1 as envelope_module
from causalcache.set_utility_throughput_pilot_contract_v1 import (
    CANONICAL_CONFIG_PATH,
    canonical_pretty_json_bytes,
    sha256_bytes,
)
from causalcache.set_utility_throughput_pilot_envelope_v1 import (
    ARTIFACT_VALIDATION_STATUS,
    AUTHORIZED_STATUS,
    CANONICAL_GIT_ENVELOPE_PATH,
    HOSTNAME,
    HOST_ALIAS,
    PREFLIGHT_EVIDENCE_PROTOCOL_ID,
    PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
    PREFLIGHT_EVIDENCE_STATUS,
    VALIDATION_STATUS,
    build_set_utility_throughput_pilot_envelope_v1,
    validate_committed_execution_envelope_lifecycle,
    validate_set_utility_throughput_pilot_envelope_v1,
    write_canonical_envelope_pair_exclusive,
)


ROOT = Path(__file__).resolve().parents[2]


def _source_contract() -> SimpleNamespace:
    payload = (ROOT / CANONICAL_CONFIG_PATH).read_bytes()
    return SimpleNamespace(data=json.loads(payload), config_sha256=sha256_bytes(payload))


@pytest.fixture()
def exact_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    contract = _source_contract()
    monkeypatch.setattr(
        envelope_module,
        "load_train_only_throughput_pilot_v1_contract",
        lambda **_: contract,
    )
    monkeypatch.setattr(envelope_module, "PERSISTENT_DATA_ROOT", tmp_path)
    gpus = [
        {
            "host_index": index + 1,
            "name": "NVIDIA H200",
            "total_memory_bytes": 150_000_000_000,
            "uuid": f"GPU-00000000-0000-0000-0000-00000000000{index}",
            "visible_index": index,
        }
        for index in range(4)
    ]
    software_versions = {
        "cuda_runtime_version": "13.0",
        "python_version": "3.11.13",
        "torch_version": "2.11.0+cu130",
        "transformers_version": "4.57.1",
    }
    raw_cleanup_log = tmp_path / "preflight.stdout.log"
    raw_cleanup_log.write_text("sampled 10 seconds; selected GPUs idle\n", encoding="utf-8")
    raw_cleanup_payload = raw_cleanup_log.read_bytes()
    evidence = tmp_path / "formal-run" / "preflight-evidence.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(
        canonical_pretty_json_bytes(
            {
                "completed_at_utc": "2026-07-19T10:00:10Z",
                "container": {
                    "id": "b" * 64,
                    "image_digest": "sha256:" + "c" * 64,
                    "image_reference": "hongccc/sglang-omni:dev",
                },
                "gpus": {
                    "devices": [
                        {
                            **gpu,
                            "post_cleanup_compute_app_count": 0,
                            "post_cleanup_memory_used_mib": 0,
                            "sampled_max_utilization_percent": 0,
                        }
                        for gpu in gpus
                    ],
                    "driver_version": "570.172.08",
                },
                "host": {"alias": HOST_ALIAS, "hostname": HOSTNAME},
                "protocol_id": PREFLIGHT_EVIDENCE_PROTOCOL_ID,
                "raw_cleanup_log": {
                    "path": str(raw_cleanup_log),
                    "sha256": sha256_bytes(raw_cleanup_payload),
                    "size_bytes": len(raw_cleanup_payload),
                },
                "runtime": {
                    "python_executable": "/usr/bin/python3",
                    "software_versions": software_versions,
                },
                "schema_version": PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
                "started_at_utc": "2026-07-19T10:00:00Z",
                "status": PREFLIGHT_EVIDENCE_STATUS,
            }
        )
    )

    def fake_inventory(_root, expected, **_kwargs):
        return [
            {
                "path": value["path"],
                "sha256": value["sha256"],
                "size_bytes": value.get("size_bytes", value.get("size")),
            }
            for value in expected
        ]

    def fake_stats(_root, expected, **_kwargs):
        return [
            {
                "device": 1,
                "inode": index + 1,
                "mode": 0o444,
                "mtime_ns": 1,
                "path": value["path"],
                "resolved_path": f"/fake/{value['path']}",
                "size_bytes": value["size_bytes"],
            }
            for index, value in enumerate(expected)
        ]

    monkeypatch.setattr(envelope_module, "_inventory_tree", fake_inventory)
    monkeypatch.setattr(envelope_module, "_stat_identity_inventory", fake_stats)
    return build_set_utility_throughput_pilot_envelope_v1(
        repository_root=ROOT,
        source_git_revision="a" * 40,
        run_root=tmp_path / "formal-run",
        python_executable="/usr/bin/python3",
        preflight_started_at_utc="2026-07-19T10:00:00Z",
        preflight_completed_at_utc="2026-07-19T10:00:10Z",
        preflight_evidence_path=evidence,
        host_alias=HOST_ALIAS,
        hostname=HOSTNAME,
        container_id="b" * 64,
        container_image_reference="hongccc/sglang-omni:dev",
        container_image_digest="sha256:" + "c" * 64,
        driver_version="570.172.08",
        gpus=gpus,
        model_local_path=tmp_path / "model",
        processor_local_root=tmp_path / "processor",
        software_versions=software_versions,
        materialized_at_utc="2026-07-19T10:00:11Z",
        inspect_local_artifacts=True,
        verify_current_environment=False,
    )


def test_exact_envelope_authorizes_only_the_preregistered_execution(
    exact_envelope: dict,
) -> None:
    validation = validate_set_utility_throughput_pilot_envelope_v1(
        exact_envelope,
        repository_root=ROOT,
        verify_repository=False,
        verify_local_artifacts=False,
        require_fresh_preflight=True,
        now_utc="2026-07-19T10:00:12Z",
    )
    assert exact_envelope["status"] == AUTHORIZED_STATUS
    assert validation["status"] == VALIDATION_STATUS
    assert exact_envelope["artifact_validation"]["status"] == (
        ARTIFACT_VALIDATION_STATUS
    )
    assert exact_envelope["source"]["source_authorization"] == {
        "load_policy_or_vision_model": False,
        "run_gpu_or_cuda": False,
        "write_pilot_result": False,
    }
    assert exact_envelope["authorization"] == {
        "authorized_native_call_ceiling": 84,
        "exact_four_gpu_execution_only": True,
        "may_generate_restoration_labels": False,
        "may_run_closed_loop": False,
        "may_train_predictor": False,
        "metric_only_output_required": True,
        "source_config_authorizes_gpu_execution": False,
        "this_validated_envelope_authorizes_gpu_execution": True,
    }


def test_gpu_worker_argv_and_output_layout_are_exact(exact_envelope: dict) -> None:
    execution = exact_envelope["execution"]
    assert execution["native_call_ceiling"] == 84
    assert execution["no_retry"] is True
    assert execution["no_top_up"] is True
    assert [gpu["visible_index"] for gpu in exact_envelope["gpus"]["devices"]] == [
        0,
        1,
        2,
        3,
    ]
    for index, worker in enumerate(execution["worker_mapping"]):
        assert worker["argv"][-4:] == [
            "--execution-envelope",
            execution["output_layout"]["envelope_path"],
            "--worker-index",
            str(index),
        ]
        assert worker["output"]["attempt_path"].endswith(
            f"workers/worker-{index:02d}/attempt.json"
        )
        assert worker["output"]["terminal_path"].endswith(
            f"workers/worker-{index:02d}/terminal.json"
        )
    assert execution["aggregate"]["argv"][-2:] == [
        "--execution-envelope",
        execution["output_layout"]["envelope_path"],
    ]
    assert execution["output_layout"]["aggregate_result_path"].endswith(
        "/aggregate.json"
    )


def test_local_model_inventory_includes_bound_snapshot_sidecar(
    exact_envelope: dict,
) -> None:
    model = exact_envelope["artifacts"]["model"]
    sidecars = [
        record for record in model["file_inventory"] if record["path"] == ".snapshot.json"
    ]
    source_binding = _source_contract().data["inputs"]["model_snapshot_manifest"]
    assert sidecars == [
        {
            "path": ".snapshot.json",
            "sha256": source_binding["sha256"],
            "size_bytes": source_binding["byte_count"],
        }
    ]
    assert model["file_count"] == 15


def test_envelope_mutations_fail_closed(exact_envelope: dict) -> None:
    mutations = (
        lambda value: value.__setitem__("status", "not-authorized"),
        lambda value: value["execution"].__setitem__("native_call_ceiling", 85),
        lambda value: value["execution"].__setitem__("no_retry", False),
        lambda value: value["gpus"]["devices"][0].__setitem__("name", "NVIDIA H100"),
        lambda value: value["source"]["source_authorization"].__setitem__(
            "run_gpu_or_cuda", True
        ),
        lambda value: value["artifacts"]["processor"].__setitem__(
            "hf_revision", "0" * 40
        ),
        lambda value: value["artifact_validation"].__setitem__(
            "completed_at_utc", "2026-07-19T10:00:10Z"
        ),
        lambda value: value.__setitem__("tokens", [1, 2, 3]),
    )
    for mutate in mutations:
        changed = copy.deepcopy(exact_envelope)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_set_utility_throughput_pilot_envelope_v1(
                changed,
                repository_root=ROOT,
                verify_repository=False,
                verify_local_artifacts=False,
                require_fresh_preflight=True,
                now_utc="2026-07-19T10:00:12Z",
            )


def test_stat_mode_uses_receipt_without_rehashing(
    exact_envelope: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = dict(exact_envelope["preflight"]["evidence"])
    monkeypatch.setattr(
        envelope_module,
        "_validated_preflight_evidence",
        lambda *args, **kwargs: ({}, binding),
    )
    monkeypatch.setattr(
        envelope_module,
        "_inventory_tree",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("stat verification must not hash artifact files")
        ),
    )
    validated: list[str] = []
    monkeypatch.setattr(
        envelope_module,
        "_validate_stat_identity_receipt",
        lambda *args, label, **kwargs: validated.append(label),
    )
    result = validate_set_utility_throughput_pilot_envelope_v1(
        exact_envelope,
        repository_root=ROOT,
        verify_repository=False,
        verify_local_artifacts="stat",
        require_fresh_preflight=True,
        now_utc="2026-07-19T10:00:12Z",
    )
    assert result["local_artifact_verification_mode"] == "stat"
    assert validated == ["model snapshot", "processor artifact"]


def test_current_gpu_check_does_not_confuse_visible_and_host_indices(
    exact_envelope: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    devices = copy.deepcopy(exact_envelope["gpus"]["devices"])
    for device, host_index in zip(devices, (0, 2, 5, 7), strict=True):
        device["host_index"] = host_index
    observed_devices = [
        {
            "name": device["name"],
            "reported_index": visible_index,
            "total_memory_bytes": device["total_memory_bytes"],
            "uuid": device["uuid"],
            "visible_index": visible_index,
        }
        for visible_index, device in enumerate(devices)
    ]
    monkeypatch.setattr(
        envelope_module,
        "_observe_current_execution_identity",
        lambda: {
            "container_id_fragments": ["b" * 12],
            "container_hostname": "b" * 12,
            "driver_version": "570.172.08",
            "gpus": observed_devices,
            "python_executable": str(Path("/usr/bin/python3").resolve()),
            "software_versions": exact_envelope["runtime"]["software_versions"],
        },
    )
    envelope_module._validate_current_execution_identity(
        hostname=HOSTNAME,
        container_id="b" * 64,
        driver_version="570.172.08",
        gpus=devices,
        python_executable="/usr/bin/python3",
        software_versions=exact_envelope["runtime"]["software_versions"],
    )


def test_current_gpu_check_accepts_one_uuid_isolated_worker(
    exact_envelope: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    devices = exact_envelope["gpus"]["devices"]
    selected = devices[2]
    monkeypatch.setattr(
        envelope_module,
        "_observe_current_execution_identity",
        lambda: {
            "container_id_fragments": ["b" * 12],
            "container_hostname": "b" * 12,
            "cuda_visible_devices": selected["uuid"],
            "driver_version": "570.172.08",
            "gpus": [
                {
                    "name": selected["name"],
                    "reported_index": 0,
                    "total_memory_bytes": selected["total_memory_bytes"],
                    "uuid": selected["uuid"],
                    "visible_index": 0,
                }
            ],
            "python_executable": str(Path("/usr/bin/python3").resolve()),
            "software_versions": exact_envelope["runtime"]["software_versions"],
        },
    )
    envelope_module._validate_current_execution_identity(
        hostname=HOSTNAME,
        container_id="b" * 64,
        driver_version="570.172.08",
        gpus=devices,
        python_executable="/usr/bin/python3",
        software_versions=exact_envelope["runtime"]["software_versions"],
        allow_single_gpu_isolation=True,
    )


def test_preflight_evidence_must_be_canonical_and_exactly_bound(
    exact_envelope: dict,
) -> None:
    evidence_path = Path(exact_envelope["preflight"]["evidence"]["path"])
    parsed = json.loads(evidence_path.read_bytes())
    evidence_path.write_text(json.dumps(parsed), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical pretty JSON"):
        validate_set_utility_throughput_pilot_envelope_v1(
            exact_envelope,
            repository_root=ROOT,
            verify_repository=False,
            verify_local_artifacts="stat",
            require_fresh_preflight=True,
            now_utc="2026-07-19T10:00:12Z",
        )


def test_preflight_post_cleanup_memory_cap_is_exact(
    exact_envelope: dict,
) -> None:
    evidence_path = Path(exact_envelope["preflight"]["evidence"]["path"])
    parsed = json.loads(evidence_path.read_bytes())
    parsed["gpus"]["devices"][0]["post_cleanup_memory_used_mib"] = 501
    evidence_path.write_bytes(canonical_pretty_json_bytes(parsed))
    with pytest.raises(ValueError, match="not idle after cleanup"):
        validate_set_utility_throughput_pilot_envelope_v1(
            exact_envelope,
            repository_root=ROOT,
            verify_repository=False,
            verify_local_artifacts="stat",
            require_fresh_preflight=True,
            now_utc="2026-07-19T10:00:12Z",
        )


def test_materializer_writes_identical_git_and_data_bytes_once(
    exact_envelope: dict, tmp_path: Path
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    data_path = Path(exact_envelope["execution"]["output_layout"]["envelope_path"])
    receipt = write_canonical_envelope_pair_exclusive(
        repository_root=repository,
        data_path=data_path,
        envelope=exact_envelope,
    )
    git_path = repository / CANONICAL_GIT_ENVELOPE_PATH
    assert data_path.read_bytes() == git_path.read_bytes()
    assert receipt["sha256"] == sha256_bytes(data_path.read_bytes())
    with pytest.raises(FileExistsError):
        write_canonical_envelope_pair_exclusive(
            repository_root=repository,
            data_path=data_path,
            envelope=exact_envelope,
        )


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def test_runtime_lifecycle_requires_single_envelope_commit_b(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    remote = tmp_path / "origin.git"
    repository.mkdir()
    subprocess.run(["git", "init", "--bare", remote], check=True, capture_output=True)
    subprocess.run(
        ["git", "init", "-b", "main", repository], check=True, capture_output=True
    )
    _git(repository, "remote", "add", "origin", str(remote))
    source_path = repository / CANONICAL_CONFIG_PATH
    source_path.parent.mkdir(parents=True)
    source_payload = b'{"source":true}\n'
    source_path.write_bytes(source_payload)
    _git(repository, "add", CANONICAL_CONFIG_PATH)
    _git(
        repository,
        "-c",
        "user.name=Codex Test",
        "-c",
        "user.email=codex@example.invalid",
        "commit",
        "-m",
        "source A",
    )
    source_revision = _git(repository, "rev-parse", "HEAD")
    _git(repository, "push", "-u", "origin", "main")

    data_path = tmp_path / "execution-envelope.json"
    envelope = {
        "source": {
            "config": {
                "path": CANONICAL_CONFIG_PATH,
                "sha256": sha256_bytes(source_payload),
                "size_bytes": len(source_payload),
            }
        },
        "status": AUTHORIZED_STATUS,
    }
    payload = canonical_pretty_json_bytes(envelope)
    data_path.write_bytes(payload)
    git_envelope = repository / CANONICAL_GIT_ENVELOPE_PATH
    git_envelope.parent.mkdir(parents=True, exist_ok=True)
    git_envelope.write_bytes(payload)
    _git(repository, "add", CANONICAL_GIT_ENVELOPE_PATH)
    _git(
        repository,
        "-c",
        "user.name=Codex Test",
        "-c",
        "user.email=codex@example.invalid",
        "commit",
        "-m",
        "envelope B",
    )
    envelope_revision = _git(repository, "rev-parse", "HEAD")
    _git(repository, "push", "origin", "main")

    result = validate_committed_execution_envelope_lifecycle(
        repository_root=repository,
        expected_source_git_revision=source_revision,
        data_envelope_path=data_path,
        expected_envelope=envelope,
    )
    assert result == {
        "envelope_git_revision": envelope_revision,
        "envelope_path": CANONICAL_GIT_ENVELOPE_PATH,
        "source_git_revision": source_revision,
        "status": "VALID_CLEAN_PUSHED_ENVELOPE_COMMIT_B",
    }


def test_validator_source_does_not_import_torch() -> None:
    sources = (
        ROOT / "code/causalcache/set_utility_throughput_pilot_envelope_v1.py",
        ROOT / "code/scripts/validate_set_utility_throughput_pilot_envelope_v1.py",
    )
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert "import torch" not in text
        assert "from torch" not in text
