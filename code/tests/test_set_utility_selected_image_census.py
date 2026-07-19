from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from causalcache.data.guiodyssey_independent import SourceFileSpec
from causalcache.set_utility_processor_substrate import (
    SelectedPilot,
    SelectedTrajectoryAssignment,
)
from causalcache.set_utility_selected_image_census import (
    RECORD_KEYS,
    aggregate_worker_records,
    build_selected_image_census_records,
    build_selected_image_census_records_from_payloads,
    canonical_json_bytes,
    materialize_worker_records,
)


def _png(mode: str, color: object, *, size: tuple[int, int]) -> bytes:
    image = Image.new(mode, size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _pilot(*, suffix: str = "a") -> SelectedPilot:
    digest_character = format(ord(suffix) - ord("a") + 1, "x")
    assignment = SelectedTrajectoryAssignment(
        trajectory_id=f"secret-source-{suffix}",
        source_id=f"secret-source-{suffix}",
        instruction_app_group_sha256="a" * 64,
        role="train",
        candidate_capacity_stratum="decisions_6_9",
        decision_count=6,
        terminal_decision_step_id=7,
        transport_file=f"mobile/use/train/secret-{suffix}.parquet",
        transport_row_index=3,
        p0_selection_sha256=digest_character * 64,
    )
    paths = [f"images/secret-source-{suffix}/observation-{index:03d}.png" for index in range(7)]
    payloads = {
        path: _png(
            "RGBA" if index == 0 else "RGB",
            (1, 2, 3, 128) if index == 0 else (index, index + 1, index + 2),
            size=(10 + index, 20 + index),
        )
        for index, path in enumerate(paths)
    }
    return SelectedPilot(
        assignment=assignment,
        source_file=SourceFileSpec(assignment.transport_file, 100, "d" * 64),
        manifest={
            "trajectory": {
                "instruction": "RAW INSTRUCTION MUST NOT LEAK",
                "steps": [
                    {"observation_index": index, "observation_path": path}
                    for index, path in enumerate(paths)
                ],
            }
        },
        image_payloads=payloads,
    )


def test_record_omits_raw_identity_and_decodes_original_rgb() -> None:
    records = build_selected_image_census_records(_pilot())
    assert len(records) == 7
    assert set(records[0].to_payload()) == RECORD_KEYS
    assert records[0].format == "PNG"
    assert records[0].mode == "RGBA"
    assert records[0].alpha_extrema == (128, 128)
    assert (records[0].width, records[0].height) == (10, 20)
    payload = b"\n".join(canonical_json_bytes(record.to_payload()) for record in records)
    assert b"secret-source" not in payload
    assert b"secret-a.parquet" not in payload
    assert b"RAW INSTRUCTION" not in payload
    assert len({record.selector_identity_sha256 for record in records}) == 7


def test_image_column_only_builder_needs_no_row_semantics() -> None:
    pilot = _pilot()
    ordered = tuple(
        pilot.image_payloads[step["observation_path"]]
        for step in pilot.manifest["trajectory"]["steps"]
    )
    records = build_selected_image_census_records_from_payloads(
        p0_selection_sha256=pilot.assignment.p0_selection_sha256,
        image_payloads=ordered,
    )
    assert records == build_selected_image_census_records(pilot)


def test_worker_jsonl_is_atomic_deterministic_and_resumable(tmp_path: Path) -> None:
    records = build_selected_image_census_records(_pilot())
    first = materialize_worker_records(
        tmp_path, worker_index=0, expected_observation_count=7, records=records
    )
    before = (tmp_path / "records/worker-00.jsonl").read_bytes()
    resumed = materialize_worker_records(
        tmp_path,
        worker_index=0,
        expected_observation_count=7,
        records=(),
    )
    assert resumed == first
    assert (tmp_path / "records/worker-00.jsonl").read_bytes() == before
    assert not list(tmp_path.rglob("*.tmp"))

    receipt = json.loads((tmp_path / "receipts/worker-00.json").read_text())
    assert receipt["observation_count"] == 7
    assert receipt["histograms"]["mode"] == {"RGB": 6, "RGBA": 1}


def test_resume_and_aggregate_fail_closed_on_tamper_or_duplicate(tmp_path: Path) -> None:
    workers = []
    for index in range(4):
        records = build_selected_image_census_records(_pilot(suffix=chr(ord("a") + index)))
        materialize_worker_records(
            tmp_path,
            worker_index=index,
            expected_observation_count=7,
            records=records,
        )
        workers.append(records)
    aggregate = aggregate_worker_records(
        tmp_path, expected_worker_observation_counts=(7, 7, 7, 7)
    )
    assert aggregate["observation_count"] == 28
    assert aggregate["histograms"]["format"] == {"PNG": 28}

    record_path = tmp_path / "records/worker-03.jsonl"
    record_path.write_bytes(record_path.read_bytes().replace(b'"PNG"', b'"JPEG"', 1))
    with pytest.raises(ValueError, match="receipt differs"):
        aggregate_worker_records(
            tmp_path, expected_worker_observation_counts=(7, 7, 7, 7)
        )


def test_partial_record_receipt_pair_cannot_resume(tmp_path: Path) -> None:
    path = tmp_path / "records/worker-00.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="incomplete worker"):
        materialize_worker_records(
            tmp_path,
            worker_index=0,
            expected_observation_count=1,
            records=(),
        )


def test_record_publish_is_no_clobber_under_destination_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = build_selected_image_census_records(_pilot())
    original_link = os.link

    def racing_link(source: object, destination: object, **kwargs: object) -> None:
        Path(destination).write_bytes(b"RACING WRITER\n")
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        materialize_worker_records(
            tmp_path,
            worker_index=0,
            expected_observation_count=7,
            records=records,
        )
    assert (tmp_path / "records/worker-00.jsonl").read_bytes() == b"RACING WRITER\n"
    assert not list(tmp_path.rglob("*.tmp"))


def test_symlinked_layout_and_noncanonical_receipt_fail_closed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "records").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="records must be one regular directory"):
        materialize_worker_records(
            staging,
            worker_index=0,
            expected_observation_count=7,
            records=build_selected_image_census_records(_pilot()),
        )

    clean = tmp_path / "clean"
    clean.mkdir()
    for index in range(4):
        materialize_worker_records(
            clean,
            worker_index=index,
            expected_observation_count=7,
            records=build_selected_image_census_records(
                _pilot(suffix=chr(ord("a") + index))
            ),
        )
    receipt = clean / "receipts/worker-02.json"
    receipt.write_text(json.dumps(json.loads(receipt.read_text())), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical pretty JSON"):
        aggregate_worker_records(
            clean, expected_worker_observation_counts=(7, 7, 7, 7)
        )
