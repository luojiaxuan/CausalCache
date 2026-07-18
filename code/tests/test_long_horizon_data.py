from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from causalcache.data.guiodyssey_independent import SourceFileSpec
from causalcache.long_horizon_data import (
    ARTIFACT_RELATIVE_PATHS,
    DATASET_REPO,
    FEATURE_JSONL_RELATIVE_PATH,
    build_long_horizon_records,
    build_ocr_backend_provenance,
    feature_states_jsonl_bytes,
    load_development_source_pilots,
    read_feature_states_jsonl,
    selection_records,
    sha256_bytes,
    validate_artifact,
    write_artifact,
)
from causalcache.restoration_v2_text_backend import (
    build_ocr_record,
    load_backend_config,
)


ROOT = Path(__file__).resolve().parents[2]
BACKEND_CONFIG_PATH = ROOT / "code/configs/restoration_v2_ocr_backend.json"
BACKEND_MANIFEST_PATH = ROOT / "data/manifests/restoration_v2_ocr_backend.json"


def _selection(development_count: int = 1, reserve_count: int = 1):
    development = [
        {
            "trajectory_id": f"development-{index:02d}",
            "shard_path": "mobile/use/train/shard-00000-of-00610.parquet",
            "row_index": index,
            "decision_count": 18,
            "selection_rank": index,
            "role": "development",
        }
        for index in range(development_count)
    ]
    reserve = [
        {
            "trajectory_id": f"reserve-{index:02d}",
            "shard_path": "mobile/use/train/shard-00000-of-00610.parquet",
            "row_index": development_count + index,
            "decision_count": 18,
            "selection_rank": development_count + index,
            "role": "unopened_reserve",
        }
        for index in range(reserve_count)
    ]
    return {
        "splits": {
            "development": {"trajectories": development},
            "unopened_reserve": {"trajectories": reserve},
        }
    }


def _png(index: int) -> bytes:
    image = Image.new("RGBA", (16, 20), (index, index * 2, index * 3, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _action() -> dict[str, object]:
    return {
        "action_type": "tap",
        "target": "coordinate_bin:x0_y0",
        "text_argument": None,
        "text_case_sensitive": False,
    }


def _tool_call() -> dict[str, object]:
    return {"function": {"name": "tap", "arguments": {"coordinate": [10, 10]}}}


def _pilot(source_id: str = "development-00"):
    paths = [f"images/{source_id}/observation-{index:03d}.png" for index in range(19)]
    images = {path: _png(index) for index, path in enumerate(paths)}
    events = [
        {
            "step_id": index + 1,
            "observation_before_path": paths[index],
            "observation_after_path": paths[index + 1],
            "executed_action": _action(),
            "source_tool_call": _tool_call(),
        }
        for index in range(18)
    ]
    decisions = [
        {
            "decision_step_id": step,
            "current_observation_path": paths[step - 1],
            "validated_action": _action(),
        }
        for step in range(2, 20)
    ]
    pilot = {
        "source": {
            "upstream_repo": "hflqf88888/GUIOdyssey",
            "upstream_revision": "61632d0f3f4d51d7e9561ce4f84347dd06b2019d",
            "transport_repo": "cua-lite/GUIOdyssey",
            "transport_revision": "ea08072b30e523fb4492e4f4597505879ffcd63b",
            "transport_file": "mobile/use/train/shard-00000-of-00610.parquet",
            "transport_file_sha256": "1" * 64,
            "transport_row_index": 0,
            "license": "cc-by-4.0",
        },
        "trajectory": {
            "source_id": source_id,
            "instruction": "Tap through the long task",
            "platform": "mobile",
            "apps": ["fixture"],
            "device_name": "fixture-device",
            "resolution": [16, 20],
            "terminal_status": "success",
            "events": events,
            "decisions": decisions,
        },
    }
    return pilot, images


def _backend_and_ocr(images):
    backend = load_backend_config(BACKEND_CONFIG_PATH)
    config_sha = sha256_bytes(BACKEND_CONFIG_PATH.read_bytes())
    records = {
        path: build_ocr_record(
            image_member_path=path,
            image_bytes=payload,
            backend_config_sha256=config_sha,
            boxes=None,
            texts=None,
            scores=None,
        )
        for path, payload in list(images.items())[:18]
    }
    return backend, config_sha, records


def _runtime(backend):
    return {
        "runtime_packages": backend["runtime_packages"],
        "rapidocr_package_file_sha256": {
            "config.yaml": backend["upstream_package_files"]["config_yaml_sha256"],
            "default_models.yaml": backend["upstream_package_files"][
                "default_models_yaml_sha256"
            ],
            "utils/load_image.py": backend["upstream_package_files"][
                "load_image_py_sha256"
            ],
            "inference_engine/onnxruntime/main.py": backend["upstream_package_files"][
                "onnxruntime_main_py_sha256"
            ],
            "ch_ppocr_rec/main.py": backend["upstream_package_files"][
                "recognizer_main_py_sha256"
            ],
        },
        "wheel_sha256": {
            name: record["sha256"] for name, record in backend["wheels"].items()
        },
        "model_sha256": {
            name: record["sha256"] for name, record in backend["models"].items()
        },
        "recognizer_character_inventory": {
            "metadata_key": backend["models"]["recognizer"][
                "embedded_character_metadata_key"
            ],
            "entry_count": backend["models"]["recognizer"][
                "embedded_character_entry_count"
            ],
            "utf8_sha256": backend["models"]["recognizer"][
                "embedded_character_utf8_sha256"
            ],
            "canonical_json_sha256": backend["models"]["recognizer"][
                "embedded_character_canonical_json_sha256"
            ],
        },
    }


def _built_fixture():
    selection = _selection()
    pilot, images = _pilot()
    backend, config_sha, records = _backend_and_ocr(images)
    trajectories, features, selected_images = build_long_horizon_records(
        pilots_by_source={"development-00": pilot},
        source_image_payloads=images,
        selection_manifest=selection,
        ocr_records_by_path=records,
        backend_config=backend,
        backend_config_sha256=config_sha,
        require_formal=False,
    )
    return (
        selection,
        pilot,
        images,
        backend,
        config_sha,
        records,
        trajectories,
        features,
        selected_images,
    )


def test_selection_uses_canonical_splits_and_rejects_overlap():
    development, reserve = selection_records(_selection(), require_formal=False)
    assert development[0]["role"] == "development"
    assert reserve[0]["role"] == "unopened_reserve"
    drifted = _selection()
    drifted["splits"]["unopened_reserve"]["trajectories"][0]["trajectory_id"] = (
        "development-00"
    )
    with pytest.raises(ValueError, match="disjoint"):
        selection_records(drifted, require_formal=False)


def test_loader_never_passes_reserve_rows_to_row_reader():
    selection = _selection()
    pilot, images = _pilot()
    observed = {}

    def reader(root, wanted):
        observed.update(wanted)
        return {"development-00": {"fixture": True}}

    def fake_builder(*args, **kwargs):
        return copy.deepcopy(pilot), dict(images)

    v1 = {
        "source_pool": {
            "upstream_repo": "upstream",
            "upstream_revision": "a" * 40,
            "transport_repo": "transport",
            "transport_revision": "b" * 40,
        },
        "policy": {"coordinate_grid_size": 10},
    }
    with (
        patch(
            "causalcache.long_horizon_data.source_file_specs",
            return_value=(
                SourceFileSpec(
                    "mobile/use/train/shard-00000-of-00610.parquet", 1, "1" * 64
                ),
            ),
        ),
        patch("causalcache.long_horizon_data.verify_local_source_files"),
        patch(
            "causalcache.long_horizon_data.build_pilot_manifest",
            side_effect=fake_builder,
        ),
    ):
        pilots, projected_images = load_development_source_pilots(
            source_root=Path("/unused"),
            source_file_manifest={},
            v1_config=v1,
            selection_manifest=selection,
            require_formal=False,
            row_reader=reader,
        )
    assert list(pilots) == ["development-00"]
    assert len(projected_images) == 18
    assert len(pilots["development-00"]["trajectory"]["events"]) == 17
    assert [
        value["decision_step_id"]
        for value in pilots["development-00"]["trajectory"]["decisions"]
    ] == [10, 18]
    assert observed == {
        "mobile/use/train/shard-00000-of-00610.parquet": {0: "development-00"}
    }


def test_record_geometry_feature_counts_and_current_action_firewall():
    fixture = _built_fixture()
    trajectories, features, selected_images = fixture[-3:]
    assert len(trajectories) == 1
    assert len(trajectories[0]["events"]) == 17
    assert len(selected_images) == 18
    assert [state.decision_step_id for state in features] == [10, 18]
    assert [len(state.candidates) for state in features] == [8, 16]
    for decision in trajectories[0]["decisions"]:
        assert decision["current_expert_action_payload_included"] is False
        assert "current_expert_action_sha256" not in decision
        assert "validated_action" not in decision
    assert trajectories[0]["decisions"][0]["current_equivalent_event_step_id"] == 9
    assert trajectories[0]["decisions"][1]["current_equivalent_event_step_id"] == 17


def test_feature_jsonl_is_canonical_and_tamper_evident():
    features = _built_fixture()[-2]
    payload = feature_states_jsonl_bytes(features)
    assert read_feature_states_jsonl(payload, require_formal=False) == features
    record = json.loads(payload.splitlines()[0])
    record["q64"][0] += 0.25
    lines = payload.splitlines()
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    tampered = b"\n".join(lines) + b"\n"
    replay = read_feature_states_jsonl(tampered, require_formal=False)
    assert replay != features


def test_exact_seven_file_artifact_and_internal_replay(tmp_path):
    (
        selection,
        _,
        _,
        backend,
        config_sha,
        records,
        trajectories,
        features,
        selected_images,
    ) = _built_fixture()
    _, backend_manifest = (
        BACKEND_MANIFEST_PATH.read_bytes(),
        json.loads(BACKEND_MANIFEST_PATH.read_text()),
    )
    provenance = build_ocr_backend_provenance(
        backend_config_sha256=config_sha,
        backend_manifest_sha256=sha256_bytes(BACKEND_MANIFEST_PATH.read_bytes()),
        backend_manifest=backend_manifest,
    )
    inputs = {
        key: {"path": f"fixture/{key}.json", "sha256": "2" * 64}
        for key in (
            "selection_manifest",
            "v1_config",
            "source_file_manifest",
            "ocr_backend_config",
            "ocr_backend_manifest",
        )
    }
    generator = {
        "git_revision": "3" * 40,
        "module_path": "code/causalcache/long_horizon_data.py",
        "module_sha256": "4" * 64,
        "build_cli_path": "code/scripts/build_long_horizon_substrate.py",
        "build_cli_sha256": "5" * 64,
        "validator_path": "code/scripts/validate_long_horizon_substrate.py",
        "validator_sha256": "6" * 64,
    }
    output = tmp_path / "artifact"
    write_artifact(
        output_dir=output,
        dataset_repo=DATASET_REPO,
        trajectories=trajectories,
        feature_states=features,
        image_payloads=selected_images,
        ocr_records_by_path=records,
        selection_manifest=selection,
        inputs=inputs,
        generator=generator,
        ocr_backend_provenance=provenance,
        ocr_runtime_identity=_runtime(backend),
        formal_counts_enforced=False,
    )
    observed = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert observed == set(ARTIFACT_RELATIVE_PATHS)
    result = validate_artifact(
        output_dir=output,
        backend_config=backend,
        backend_config_sha256=config_sha,
        selection_manifest=selection,
        require_formal=False,
    )
    assert result["outcome"] == "PASSED_LONG_HORIZON_SUBSTRATE_VALIDATION"
    assert result["counts"] == {
        "trajectory_count": 1,
        "event_count": 17,
        "state_count": 2,
        "candidate_count": 24,
        "image_member_count": 18,
        "ocr_record_count": 18,
    }
    feature_path = output / FEATURE_JSONL_RELATIVE_PATH
    feature_path.write_bytes(feature_path.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="identity"):
        validate_artifact(
            output_dir=output,
            backend_config=backend,
            backend_config_sha256=config_sha,
            selection_manifest=selection,
            require_formal=False,
        )
