from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import causalcache.set_utility_action_stability_envelope_v1 as envelope_module
from causalcache.set_utility_action_stability_contract_v1 import (
    CANONICAL_CONFIG_PATH,
    canonical_pretty_json_bytes,
    sha256_bytes,
)
from causalcache.set_utility_action_stability_envelope_v1 import (
    CANONICAL_GIT_ENVELOPE_PATH,
    build_action_stability_envelope_v1,
    build_action_stability_preflight_evidence_v1,
    canonical_run_layout,
    validate_action_stability_envelope_v1,
    write_canonical_envelope_pair_exclusive,
    write_preflight_evidence_exclusive,
)


@dataclass
class _Contract:
    data: dict[str, Any]
    config_sha256: str


def _gpu(index: int) -> dict[str, Any]:
    return {
        "host_index": index,
        "name": "NVIDIA H200",
        "sample_window_seconds": 10.0,
        "sampled_utilization_percent": [0, 0],
        "total_memory_bytes": 150_000_000_000,
        "uuid": f"GPU-00000000-0000-0000-0000-{index:012d}",
        "visible_index": index,
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "repo"
    run_root = tmp_path / "data" / "runs" / "d1"
    monkeypatch.setattr(envelope_module, "PERSISTENT_DATA_ROOT", tmp_path / "data")
    for relative in (
        CANONICAL_CONFIG_PATH,
        "code/scripts/run_set_utility_action_stability_worker_v1.py",
        "code/scripts/aggregate_set_utility_action_stability_v1.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if relative == CANONICAL_CONFIG_PATH else "# stub\n")
    config_payload = (root / CANONICAL_CONFIG_PATH).read_bytes()
    contract = _Contract(
        data={
            "diagnostic": {"generation_call_ceiling": 36},
            "source": {"inventory_sha256": "1" * 64},
        },
        config_sha256=sha256_bytes(config_payload),
    )
    model = tmp_path / "model"
    processor = tmp_path / "processor"
    model.mkdir()
    processor.mkdir()
    parent = {
        "artifacts": {
            "model": {"local_path": str(model)},
            "processor": {"local_root": str(processor)},
        },
        "execution": {
            "output_layout": {"envelope_path": str(tmp_path / "parent.json")}
        },
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_bytes(canonical_pretty_json_bytes(parent))
    monkeypatch.setattr(
        envelope_module,
        "PARENT_EXECUTION_ENVELOPE_SHA256",
        sha256_bytes(parent_path.read_bytes()),
    )
    cleanup = tmp_path / "cleanup.log"
    cleanup.write_text("sampled four idle H200 devices\n")
    evidence = build_action_stability_preflight_evidence_v1(
        raw_cleanup_log_path=cleanup,
        started_at_utc="2026-07-19T12:00:00Z",
        completed_at_utc="2026-07-19T12:00:10Z",
        host_alias="hyper01",
        hostname="node-radixark-16-0001",
        container_id="a" * 64,
        driver_version="570.172.08",
        selected_gpus=[_gpu(index) for index in range(4)],
    )
    evidence_receipt = write_preflight_evidence_exclusive(
        run_root=run_root, evidence=evidence
    )
    loader = lambda **_: contract
    built = build_action_stability_envelope_v1(
        repository_root=root,
        source_git_revision="b" * 40,
        run_root=run_root,
        python_executable=str(Path(sys.executable).resolve()),
        preflight_evidence_path=evidence_receipt["path"],
        parent_envelope_path=parent_path,
        processor_root=processor,
        model_dir=model,
        host_alias="hyper01",
        hostname="node-radixark-16-0001",
        container_id="a" * 64,
        container_image_reference="hongccc/sglang-omni:dev",
        container_image_digest="sha256:" + "c" * 64,
        driver_version="570.172.08",
        software_versions={
            "cuda_runtime_version": "13.0",
            "python_version": "3.12.3",
            "torch_version": "2.11.0+cu130",
            "transformers_version": "5.6.0",
        },
        materialized_at_utc="2026-07-19T12:00:11Z",
        source_contract_loader=loader,
    )
    return root, run_root, contract, loader, built


def test_preflight_layout_and_attention_backends_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run_root, _, loader, built = _fixture(tmp_path, monkeypatch)
    layout = canonical_run_layout(run_root)

    assert len(layout["workers"]) == 4
    assert built["execution"]["generation_call_ceiling"] == 36
    assert built["execution"]["encode_call_ceiling"] == 24
    assert built["execution"]["auto_all_four_terminals_required_before_any_eager_attempt"] is True
    assert [worker["profile"] for worker in built["execution"]["worker_mapping"]] == [
        "auto",
        "auto",
        "auto",
        "auto",
        "eager",
        "eager",
        "eager",
        "eager",
    ]
    assert built["runtime"]["attention_backend_by_profile"] == {
        "auto": {"text": "sdpa", "top": "sdpa", "vision": "sdpa"},
        "eager": {"text": "eager", "top": "eager", "vision": "eager"},
    }
    validation = validate_action_stability_envelope_v1(
        built,
        repository_root=root,
        envelope_path=layout["envelope_path"],
        verify_repository=False,
        require_fresh_preflight=True,
        verify_current_environment=False,
        now_utc="2026-07-19T12:00:12Z",
        source_contract_loader=loader,
    )
    assert validation["worker_count"] == 4
    assert validation["profiles"] == ["auto", "eager"]
    assert validation["this_validated_envelope_authorizes_gpu_execution"] is True


def test_envelope_mutation_and_second_write_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run_root, _, loader, built = _fixture(tmp_path, monkeypatch)
    layout = canonical_run_layout(run_root)
    changed = copy.deepcopy(built)
    changed["runtime"]["attention_backend_by_profile"]["auto"]["vision"] = "eager"
    with pytest.raises(ValueError, match="attention backend"):
        validate_action_stability_envelope_v1(
            changed,
            repository_root=root,
            envelope_path=layout["envelope_path"],
            verify_repository=False,
            require_fresh_preflight=False,
            source_contract_loader=loader,
        )

    receipt = write_canonical_envelope_pair_exclusive(
        repository_root=root,
        data_path=layout["envelope_path"],
        envelope=built,
    )
    assert receipt["git_path"] == CANONICAL_GIT_ENVELOPE_PATH
    with pytest.raises(FileExistsError):
        write_canonical_envelope_pair_exclusive(
            repository_root=root,
            data_path=layout["envelope_path"],
            envelope=built,
        )
