from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

import causalcache.set_utility_selected_image_census_postflight_v2 as postflight
from causalcache.data.guiodyssey_independent import SourceFileSpec
from causalcache.set_utility_processor_freeze import build_shard_worker_schedule
from causalcache.set_utility_processor_substrate import (
    SelectedRowReadPlan,
    SelectedShardRead,
    SelectedTrajectoryAssignment,
)
from causalcache.set_utility_selected_image_census import canonical_json_bytes
from causalcache.set_utility_selected_image_census_contract import (
    SelectedImageCensusContract,
)
from causalcache.set_utility_selected_image_census_contract_v2 import (
    CANONICAL_CONFIG_PATH,
    HF_REPO,
    HF_TAG,
    REQUIRED_IDENTITY_ARGUMENTS,
    REQUIRED_PATH_ARGUMENTS,
)
from causalcache.set_utility_selected_image_census_postflight_v2 import (
    INVALID_V1_FINAL_STATUS,
    INVALID_V1_PROTOCOL_ID,
    OUTPUT_NAMESPACE_PREFIX,
    SelectedImageCensusV2PostflightContext,
    VALIDATION_STATUS,
    build_selected_image_census_v2_postflight_context,
    validate_completed_selected_image_census_v2_root,
)
from causalcache.set_utility_selected_image_census_v2 import (
    FINAL_STATUS,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    WORKER_COUNT,
    aggregate_worker_records,
    build_selected_image_census_records_from_payloads,
    materialize_worker_records,
    parquet_projection_payload,
    sha256_bytes,
)


GIT_REVISION = "d" * 40


def _png(index: int) -> bytes:
    image = Image.new(
        "RGBA" if index % 2 else "RGB",
        (8 + index, 12 + index),
        (index + 1, index + 2, index + 3, 255)
        if index % 2
        else (index + 1, index + 2, index + 3),
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _schedule():
    shards = []
    for index, digest_character in enumerate("1234"):
        path = f"mobile/use/train/shard-{index:05d}.parquet"
        assignment = SelectedTrajectoryAssignment(
            trajectory_id=f"trajectory-{index}",
            source_id=f"trajectory-{index}",
            instruction_app_group_sha256="a" * 64,
            role="train",
            candidate_capacity_stratum="decisions_6_9",
            decision_count=6,
            terminal_decision_step_id=7,
            transport_file=path,
            transport_row_index=0,
            p0_selection_sha256=chr(ord("a") + index) * 64,
        )
        source = SourceFileSpec(path, 100, digest_character * 64)
        shards.append(
            SelectedShardRead(
                source_file=source,
                source_row_count=1,
                assignments=(assignment,),
            )
        )
    return build_shard_worker_schedule(
        SelectedRowReadPlan(shards=tuple(shards), assignment_count=4)
    )


def _hashes() -> dict[str, str]:
    names = (
        "freeze_b_v2_manifest",
        "full_pool_inventory_manifest",
        "independent_reference_gate_base_config",
        "invalid_v1_attempt_summary",
        "p0_census_manifest",
        "v1_contract",
        "v1_core",
        "v1_execution_config",
        "v1_runner",
    )
    return {
        name: format(index + 1, "x") * 64 for index, name in enumerate(names)
    }


def _context(tmp_path: Path) -> SelectedImageCensusV2PostflightContext:
    repository_root = (tmp_path / "repository").resolve()
    repository_root.mkdir()
    hashes = _hashes()
    return SelectedImageCensusV2PostflightContext(
        repository_root=repository_root,
        execution_config_path=(repository_root / CANONICAL_CONFIG_PATH).resolve(),
        execution_config_sha256="a" * 64,
        expected_git_revision=GIT_REVISION,
        worker_schedule=_schedule(),
        source_audit=hashes,
        frozen_input_sha256=hashes,
        predecessor_sha256={
            "invalid_v1_attempt_summary_sha256": hashes[
                "invalid_v1_attempt_summary"
            ],
            "v1_contract_sha256": hashes["v1_contract"],
            "v1_core_sha256": hashes["v1_core"],
            "v1_execution_config_sha256": hashes["v1_execution_config"],
            "v1_runner_sha256": hashes["v1_runner"],
        },
    )


def _runtime_cli(
    root: Path, context: SelectedImageCensusV2PostflightContext
) -> dict[str, object]:
    values: dict[str, object] = {
        name.removeprefix("--").replace("-", "_"): "fixture"
        for name in (*REQUIRED_PATH_ARGUMENTS, *REQUIRED_IDENTITY_ARGUMENTS)
    }
    for name in REQUIRED_PATH_ARGUMENTS:
        key = name.removeprefix("--").replace("-", "_")
        values[key] = str((root.parent / key).resolve())
    values.update(
        {
            "container_id": "fixture-container",
            "container_image_digest": f"sha256:{'e' * 64}",
            "execution_config": str(context.execution_config_path),
            "git_revision": GIT_REVISION,
            "host_alias": "fixture-host",
            "host_hostname": "fixture-hostname",
            "output_root": str(root),
            "pillow_version": "12.2.0",
            "pyarrow_version": "24.0.0",
            "python_version": "3.12.3",
            "repository_root": str(context.repository_root),
            "worker_count": WORKER_COUNT,
        }
    )
    return values


def _write_root(
    tmp_path: Path,
) -> tuple[Path, SelectedImageCensusV2PostflightContext]:
    context = _context(tmp_path)
    root = (tmp_path / f"{OUTPUT_NAMESPACE_PREFIX}{GIT_REVISION[:7]}").resolve()
    root.mkdir()
    for worker in context.worker_schedule.workers:
        assignment = worker.shards[0].assignments[0]
        records = build_selected_image_census_records_from_payloads(
            p0_selection_sha256=assignment.p0_selection_sha256,
            image_payloads=tuple(_png(index) for index in range(7)),
        )
        materialize_worker_records(
            root,
            worker_index=worker.worker_index,
            expected_observation_count=worker.observation_load,
            records=records,
        )
    runtime_cli = _runtime_cli(root, context)
    schedule = context.worker_schedule
    run_identity = {
        "config_sha256": context.execution_config_sha256,
        "container": {
            "id": runtime_cli["container_id"],
            "image_digest": runtime_cli["container_image_digest"],
        },
        "git_revision": GIT_REVISION,
        "host": {
            "alias": runtime_cli["host_alias"],
            "hostname": runtime_cli["host_hostname"],
        },
        "parquet_projection": parquet_projection_payload(),
        "protocol_id": PROTOCOL_ID,
        "runtime_cli": runtime_cli,
        "schedule": {
            "inventory_sha256": schedule.inventory_sha256,
            "selected_observation_count": schedule.selected_observation_count,
            "selected_shard_count": schedule.selected_shard_count,
            "selected_trajectory_count": schedule.selected_trajectory_count,
            "worker_observation_counts": [
                worker.observation_load for worker in schedule.workers
            ],
        },
        "schema_version": SCHEMA_VERSION,
        "source_audit": dict(context.source_audit),
        "versions": {
            "pillow": runtime_cli["pillow_version"],
            "pyarrow": runtime_cli["pyarrow_version"],
            "python": runtime_cli["python_version"],
        },
    }
    run_payload = canonical_json_bytes(run_identity, pretty=True)
    (root / "run-identity.json").write_bytes(run_payload)
    aggregate = aggregate_worker_records(
        root,
        expected_worker_observation_counts=tuple(
            worker.observation_load for worker in schedule.workers
        ),
    )
    manifest = {
        "artifact": {
            "hf_mutation_count": 0,
            "intended_private_hf_repo": HF_REPO,
            "intended_tag": HF_TAG,
            "status": "AWAITING_COMMITTED_POSTFLIGHT",
        },
        "bindings": {
            "config_sha256": context.execution_config_sha256,
            "container_image_digest": runtime_cli["container_image_digest"],
            "freeze_b_v2_manifest_sha256": context.frozen_input_sha256[
                "freeze_b_v2_manifest"
            ],
            "git_revision": GIT_REVISION,
            **dict(context.predecessor_sha256),
            "run_identity_sha256": sha256_bytes(run_payload),
            "schedule_inventory_sha256": schedule.inventory_sha256,
        },
        "counts": {
            "model_or_policy_load_count": 0,
            "ocr_count": 0,
            "selected_observation_count": aggregate["observation_count"],
            "selected_shard_count": schedule.selected_shard_count,
            "selected_trajectory_count": schedule.selected_trajectory_count,
            "worker_count": WORKER_COUNT,
        },
        "histograms": aggregate["histograms"],
        "protocol_id": PROTOCOL_ID,
        "record_inventory_sha256": aggregate["record_inventory_sha256"],
        "runtime": {
            "device": "cpu",
            "parquet_projection": parquet_projection_payload(),
            "pillow_version": runtime_cli["pillow_version"],
            "pyarrow_version": runtime_cli["pyarrow_version"],
            "python_version": runtime_cli["python_version"],
        },
        "schema_version": SCHEMA_VERSION,
        "selector_inventory_sha256": aggregate["selector_inventory_sha256"],
        "status": FINAL_STATUS,
        "unique_image_sha256_count": aggregate["unique_image_sha256_count"],
        "workers": aggregate["workers"],
    }
    (root / "manifest.json").write_bytes(
        canonical_json_bytes(manifest, pretty=True)
    )
    return root, context


def _snapshot(root: Path) -> tuple[tuple[str, int, bytes | None], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            path.lstat().st_mode,
            path.read_bytes() if path.is_file() and not path.is_symlink() else None,
        )
        for path in sorted(root.rglob("*"))
    )


def _rewrite_run_identity(root: Path, value: dict[str, object]) -> None:
    payload = canonical_json_bytes(value, pretty=True)
    (root / "run-identity.json").write_bytes(payload)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bindings"]["run_identity_sha256"] = sha256_bytes(payload)
    manifest_path.write_bytes(canonical_json_bytes(manifest, pretty=True))


def test_postflight_accepts_completed_root_and_is_read_only(tmp_path: Path) -> None:
    root, context = _write_root(tmp_path)
    before = _snapshot(root)

    result = validate_completed_selected_image_census_v2_root(
        root, context=context
    )

    assert result["status"] == VALIDATION_STATUS
    assert result["protocol_id"] == PROTOCOL_ID
    assert result["observation_count"] == 28
    assert result["parquet_projection"] == parquet_projection_payload()
    assert _snapshot(root) == before


@pytest.mark.parametrize("extra_path", ["extra.json", "records/x.partial"])
def test_postflight_rejects_extra_or_partial_artifact(
    tmp_path: Path, extra_path: str
) -> None:
    root, context = _write_root(tmp_path)
    path = root / extra_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="inventory drifted"):
        validate_completed_selected_image_census_v2_root(root, context=context)


def test_postflight_rejects_symlinked_expected_artifact(tmp_path: Path) -> None:
    root, context = _write_root(tmp_path)
    victim = root / "receipts/worker-00.json"
    victim.unlink()
    os.symlink(root / "receipts/worker-01.json", victim)

    with pytest.raises(ValueError, match="real regular file"):
        validate_completed_selected_image_census_v2_root(root, context=context)


def test_postflight_rejects_wrong_output_namespace(tmp_path: Path) -> None:
    root, context = _write_root(tmp_path)
    moved = root.parent / "unbound-output-root"
    root.rename(moved)

    with pytest.raises(ValueError, match="namespace differs"):
        validate_completed_selected_image_census_v2_root(moved, context=context)


def test_postflight_rejects_remaining_staging_root(tmp_path: Path) -> None:
    root, context = _write_root(tmp_path)
    staging = root.parent / f".{root.name}.incomplete"
    staging.mkdir()

    with pytest.raises(ValueError, match="staging root remains"):
        validate_completed_selected_image_census_v2_root(root, context=context)


def test_postflight_rejects_noncanonical_jsonl(tmp_path: Path) -> None:
    root, context = _write_root(tmp_path)
    path = root / "records/worker-00.jsonl"
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    path.write_bytes(b"".join(json.dumps(item).encode() + b"\n" for item in records))

    with pytest.raises(ValueError, match="not canonical compact JSON"):
        validate_completed_selected_image_census_v2_root(root, context=context)


def test_postflight_rejects_receipt_projection_tamper(tmp_path: Path) -> None:
    root, context = _write_root(tmp_path)
    path = root / "receipts/worker-02.json"
    receipt = json.loads(path.read_text())
    receipt["parquet_projection"]["columns"] = ["images", "messages"]
    path.write_bytes(canonical_json_bytes(receipt, pretty=True))

    with pytest.raises(ValueError, match="record/projection reconstruction"):
        validate_completed_selected_image_census_v2_root(root, context=context)


def test_postflight_rejects_run_projection_tamper_even_if_rehashed(
    tmp_path: Path,
) -> None:
    root, context = _write_root(tmp_path)
    value = json.loads((root / "run-identity.json").read_text())
    value["parquet_projection"]["expected_row_keys"] = ["images", "metadata"]
    _rewrite_run_identity(root, value)

    with pytest.raises(ValueError, match="parquet projection drifted"):
        validate_completed_selected_image_census_v2_root(root, context=context)


def test_postflight_rejects_v1_protocol_even_if_rehashed(tmp_path: Path) -> None:
    root, context = _write_root(tmp_path)
    value = json.loads((root / "run-identity.json").read_text())
    value["protocol_id"] = INVALID_V1_PROTOCOL_ID
    value["schema_version"] = "1.0.0"
    _rewrite_run_identity(root, value)

    with pytest.raises(ValueError, match="v1 or missing protocol"):
        validate_completed_selected_image_census_v2_root(root, context=context)


def test_postflight_rejects_v1_completion_status(tmp_path: Path) -> None:
    root, context = _write_root(tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = INVALID_V1_FINAL_STATUS
    path.write_bytes(canonical_json_bytes(manifest, pretty=True))

    with pytest.raises(ValueError, match="invalid v1 output root"):
        validate_completed_selected_image_census_v2_root(root, context=context)


def test_postflight_rejects_manifest_histogram_or_status_tamper(
    tmp_path: Path,
) -> None:
    root, context = _write_root(tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["histograms"]["format"]["PNG"] -= 1
    manifest["artifact"]["status"] = "PENDING_HF_UPLOAD"
    path.write_bytes(canonical_json_bytes(manifest, pretty=True))

    with pytest.raises(ValueError, match="whole-root reconstruction"):
        validate_completed_selected_image_census_v2_root(root, context=context)


def test_postflight_rejects_selector_order_tamper(tmp_path: Path) -> None:
    root, context = _write_root(tmp_path)
    path = root / "records/worker-01.jsonl"
    lines = path.read_bytes().splitlines()
    path.write_bytes(b"\n".join([lines[1], lines[0], *lines[2:]]) + b"\n")

    with pytest.raises(ValueError, match="selector inventory or canonical order"):
        validate_completed_selected_image_census_v2_root(root, context=context)


def test_postflight_detects_tree_mutation_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, context = _write_root(tmp_path)
    original = postflight._validate_completed_root

    def mutating_validator(path: Path, supplied_context: object):
        result = original(path, supplied_context)
        (path / "late-mutation").write_bytes(b"changed")
        return result

    monkeypatch.setattr(postflight, "_validate_completed_root", mutating_validator)
    with pytest.raises(RuntimeError, match="changed during read-only postflight"):
        validate_completed_selected_image_census_v2_root(root, context=context)


def test_v2_postflight_context_rejects_v1_contract_class(tmp_path: Path) -> None:
    v1_contract = SelectedImageCensusContract(
        data={},
        repository_root=tmp_path.resolve(),
        config_sha256="a" * 64,
        frozen_inputs={},
    )

    with pytest.raises(TypeError, match="validated v2 census contract"):
        build_selected_image_census_v2_postflight_context(
            v1_contract, expected_git_revision=GIT_REVISION  # type: ignore[arg-type]
        )
