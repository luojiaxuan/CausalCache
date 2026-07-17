from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from causalcache.data.guiodyssey_restoration_v2_expansion import (
    ARTIFACT_RELATIVE_PATHS,
    DERIVED_ROLE_ORDER,
    EXPECTED_FORMAL_COUNTS,
    MANIFEST_RELATIVE_PATH,
    _bind_supplied_manifest_sha256,
    _load_json_object,
    _validate_trajectories,
    build_trajectory_records,
    decision_view_events,
    pretty_json_bytes,
    validate_artifact,
    validate_formal_image_inventory,
    validate_frozen_inputs,
    write_artifact,
)
from causalcache.restoration_v2_2_label_expansion import (
    build_expansion_manifest,
)
from causalcache.restoration_v2_text_backend import load_backend_config


ROOT = Path(__file__).resolve().parents[2]
BACKEND_CONFIG_PATH = ROOT / "code/configs/restoration_v2_ocr_backend.json"
EXPANSION_CONFIG_PATH = (
    ROOT / "code/configs/causalcache_restoration_v2_2_label_expansion_v1.json"
)
PARENT_SELECTION_PATH = ROOT / "data/manifests/restoration_v2_selection.json"
V1_CONFIG_PATH = ROOT / "code/configs/independent_reference_gate_v1.json"
SOURCE_MANIFEST_PATH = (
    ROOT / "data/manifests/independent_reference_gate_v1_source_files.json"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _runtime_identity(backend: dict) -> dict:
    upstream = backend["upstream_package_files"]
    recognizer = backend["models"]["recognizer"]
    return {
        "runtime_packages": dict(backend["runtime_packages"]),
        "rapidocr_package_file_sha256": {
            "config.yaml": upstream["config_yaml_sha256"],
            "default_models.yaml": upstream["default_models_yaml_sha256"],
            "utils/load_image.py": upstream["load_image_py_sha256"],
            "inference_engine/onnxruntime/main.py": upstream[
                "onnxruntime_main_py_sha256"
            ],
            "ch_ppocr_rec/main.py": upstream["recognizer_main_py_sha256"],
        },
        "wheel_sha256": {
            name: record["sha256"] for name, record in backend["wheels"].items()
        },
        "model_sha256": {
            role: record["sha256"] for role, record in backend["models"].items()
        },
        "recognizer_character_inventory": {
            "metadata_key": recognizer["embedded_character_metadata_key"],
            "entry_count": recognizer["embedded_character_entry_count"],
            "utf8_sha256": recognizer["embedded_character_utf8_sha256"],
            "canonical_json_sha256": recognizer[
                "embedded_character_canonical_json_sha256"
            ],
        },
    }


def _ocr_provenance(backend_sha256: str) -> dict[str, str]:
    return {
        "backend_id": "rapidocr-3.8.4-ppocrv5-mobile-en-cpu-v1",
        "backend_config_sha256": backend_sha256,
        "hf_model_repo": "fixture/ocr-model",
        "hf_model_revision": "e" * 40,
        "hf_model_tag": "v1.0.0",
        "real_screen_golden_dataset_revision": "f" * 40,
    }


def _generator() -> dict[str, str]:
    return {
        "git_revision": "a" * 40,
        "module_path": (
            "code/causalcache/data/guiodyssey_restoration_v2_expansion.py"
        ),
        "module_sha256": "b" * 64,
        "build_cli_path": "code/scripts/build_guiodyssey_restoration_v2_expansion.py",
        "build_cli_sha256": "c" * 64,
        "validator_path": (
            "code/scripts/validate_guiodyssey_restoration_v2_expansion.py"
        ),
        "validator_sha256": "d" * 64,
    }


def _inputs(backend_sha256: str) -> dict[str, dict[str, str]]:
    names = (
        "label_expansion_config",
        "parent_selection_manifest",
        "expansion_selection_manifest",
        "v1_config",
        "source_file_manifest",
        "ocr_backend_config",
        "ocr_backend_manifest",
    )
    values = {
        name: {"path": f"fixture/{name}.json", "sha256": str(index) * 64}
        for index, name in enumerate(names, start=1)
    }
    values["ocr_backend_config"]["sha256"] = backend_sha256
    return values


def _state(source_id: str, step: int) -> dict:
    candidates = list(range(1, step - 1))
    return {
        "state_id": f"{source_id}:decision_step:{step:03d}",
        "source_id": source_id,
        "decision_step_id": step,
        "history_event_step_ids": list(range(1, step)),
        "current_equivalent_event_step_id": step - 1,
        "candidate_event_step_ids": candidates,
        "candidate_event_count": len(candidates),
        "full_subset_distance_rows": 2 ** len(candidates),
        "deployment_conditional_edges": {2: 4, 3: 9, 4: 16}[len(candidates)],
        "teacher_forwards": 2 ** len(candidates) + 1,
    }


def _selection(source_ids_by_role: dict[str, list[str]]) -> dict:
    return {
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "structural_manifest_only": True,
        "splits": {
            role: {
                "trajectories": [
                    {
                        "source_id": source_id,
                        "eligible_order_index": index,
                        "transport_file": "mobile/use/train/fixture.parquet",
                        "transport_row_index": index,
                        "selection_sha256": _sha256(
                            f"selection:{source_id}".encode("utf-8")
                        ),
                        "decision_count": 6,
                    }
                    for index, source_id in enumerate(source_ids_by_role[role])
                ],
                "states": [
                    _state(source_id, step)
                    for source_id in source_ids_by_role[role]
                    for step in (4, 5, 6)
                ],
            }
            for role in DERIVED_ROLE_ORDER
        },
    }


def _pilot(source_id: str, image_paths: list[str]) -> dict:
    source_tool_call = {
        "function": {
            "name": "tap",
            "arguments": {"coordinate": [100, 200], "clicks": 1},
        }
    }
    executed_action = {
        "action_type": "tap",
        "target": "coordinate_bin:x0_y1",
        "text_argument": None,
        "text_case_sensitive": False,
    }
    events = [
        {
            "step_id": step,
            "observation_before_path": image_paths[step - 1],
            "observation_after_path": image_paths[step],
            "executed_action": executed_action,
            "source_tool_call": source_tool_call,
        }
        for step in range(1, 6)
    ]
    decisions = [
        {
            "decision_step_id": step,
            "current_observation_path": image_paths[step - 1],
            "history_event_step_ids": list(range(1, step)),
            "validated_action": executed_action,
            "validation_source": "recorded_successful_guiodyssey_trajectory",
        }
        for step in (4, 5, 6)
    ]
    return {
        "source": {
            "upstream_repo": "hflqf88888/GUIOdyssey",
            "upstream_revision": "e" * 40,
            "transport_repo": "cua-lite/GUIOdyssey",
            "transport_revision": "f" * 40,
            "transport_file": "mobile/use/train/fixture.parquet",
            "transport_file_sha256": "1" * 64,
            "transport_row_index": 0,
            "license": "cc-by-4.0",
        },
        "trajectory": {
            "source_id": source_id,
            "instruction": "Fixture expansion instruction",
            "platform": "android",
            "apps": ["fixture"],
            "device_name": "fixture-device",
            "resolution": "8x8",
            "terminal_status": "success",
            "events": events,
            "decisions": decisions,
        },
    }


def _fake_ocr_validator(
    record: dict,
    *,
    image_bytes: bytes,
    backend_config: dict,
    backend_config_sha256: str,
):
    del backend_config
    if record["image_sha256"] != _sha256(image_bytes):
        raise ValueError("fake OCR image digest drifted")
    if record["backend_config_sha256"] != backend_config_sha256:
        raise ValueError("fake OCR backend digest drifted")
    return object()


def _fake_event_builder(
    event: dict,
    *,
    image_payloads: dict[str, bytes],
    ocr_records_by_path: dict,
    prepared_by_path: dict,
) -> dict:
    del prepared_by_path
    if "low_fidelity_v2" in event:
        return dict(event)
    step = int(event["step_id"])
    before = event["observation_before_path"]
    after = event["observation_after_path"]
    before_sha = _sha256(image_payloads[before])
    after_sha = _sha256(image_payloads[after])
    serialized = f"step_id: {step}\naction_type: click\n"
    return {
        "step_id": step,
        "observation_before_path": before,
        "observation_before_sha256": before_sha,
        "observation_after_path": after,
        "observation_after_sha256": after_sha,
        "executed_action": dict(event["executed_action"]),
        "source_tool_call": dict(event["source_tool_call"]),
        "low_fidelity_v2": {"step_id": step, "action_type": "click"},
        "low_fidelity_v2_serialized": serialized,
        "low_fidelity_v2_sha256": _sha256(serialized.encode("utf-8")),
        "low_fidelity_v2_metadata": {
            "before_full_spatial_tokens_sha256": ocr_records_by_path[before][
                "full_spatial_tokens_sha256"
            ],
            "after_full_spatial_tokens_sha256": ocr_records_by_path[after][
                "full_spatial_tokens_sha256"
            ],
        },
        "high_fidelity_v2": {
            "content_type": "image",
            "image_member_path": after,
            "image_sha256": after_sha,
        },
        "ocr_record_refs": {
            "before_canonical_ocr_record_sha256": ocr_records_by_path[before][
                "canonical_ocr_record_sha256"
            ],
            "after_canonical_ocr_record_sha256": ocr_records_by_path[after][
                "canonical_ocr_record_sha256"
            ],
        },
    }


class GUIOdysseyRestorationV2ExpansionContractTest(unittest.TestCase):
    def test_strict_json_loader_rejects_non_finite_constants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"value":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                _load_json_object(path)

    def test_supplied_manifest_digest_rejects_alternate_mapping(self) -> None:
        manifest = {"value": 1}
        digest = _sha256(pretty_json_bytes(manifest))
        _bind_supplied_manifest_sha256(
            manifest, expected_sha256=digest, field="selection"
        )
        with self.assertRaisesRegex(ValueError, "canonical pretty bytes"):
            _bind_supplied_manifest_sha256(
                {"value": 2}, expected_sha256=digest, field="selection"
            )

    def test_committed_split_derives_exact_formal_geometry(self) -> None:
        config_payload = EXPANSION_CONFIG_PATH.read_bytes()
        config = json.loads(config_payload)
        parent_payload = PARENT_SELECTION_PATH.read_bytes()
        parent = json.loads(parent_payload)
        manifest = build_expansion_manifest(
            config=config,
            config_sha256=_sha256(config_payload),
            parent=parent,
            parent_sha256=_sha256(parent_payload),
            generator={
                "git_revision": "a" * 40,
                "module_path": (
                    "code/causalcache/restoration_v2_2_label_expansion.py"
                ),
                "module_sha256": "b" * 64,
                "cli_path": (
                    "code/scripts/materialize_restoration_v2_2_label_expansion.py"
                ),
                "cli_sha256": "c" * 64,
                "validator_path": (
                    "code/scripts/validate_restoration_v2_2_label_expansion.py"
                ),
                "validator_sha256": "d" * 64,
            },
        )
        backend = load_backend_config(BACKEND_CONFIG_PATH)
        validate_frozen_inputs(
            expansion_manifest=manifest,
            expansion_manifest_sha256=_sha256(pretty_json_bytes(manifest)),
            expansion_config=config,
            expansion_config_sha256=_sha256(config_payload),
            parent_selection=parent,
            parent_selection_sha256=_sha256(parent_payload),
            v1_config=json.loads(V1_CONFIG_PATH.read_bytes()),
            v1_config_sha256=_sha256(V1_CONFIG_PATH.read_bytes()),
            source_file_manifest_sha256=_sha256(SOURCE_MANIFEST_PATH.read_bytes()),
            backend_config=backend,
        )
        with self.assertRaisesRegex(ValueError, "canonical pretty bytes"):
            validate_frozen_inputs(
                expansion_manifest=manifest,
                expansion_manifest_sha256="0" * 64,
                expansion_config=config,
                expansion_config_sha256=_sha256(config_payload),
                parent_selection=parent,
                parent_selection_sha256=_sha256(parent_payload),
                v1_config=json.loads(V1_CONFIG_PATH.read_bytes()),
                v1_config_sha256=_sha256(V1_CONFIG_PATH.read_bytes()),
                source_file_manifest_sha256=_sha256(
                    SOURCE_MANIFEST_PATH.read_bytes()
                ),
                backend_config=backend,
            )
        ids_by_role = {
            role: [
                record["source_id"]
                for record in manifest["splits"][role]["trajectories"]
            ]
            for role in DERIVED_ROLE_ORDER
        }
        self.assertEqual(
            {role: len(values) for role, values in ids_by_role.items()},
            {"gate_train_expansion": 48, "gate_development_expansion": 16},
        )
        paths = tuple(
            sorted(
                f"images/{source_id}/observation-{index:03d}.png"
                for values in ids_by_role.values()
                for source_id in values
                for index in range(6)
            )
        )
        self.assertEqual(len(paths), EXPECTED_FORMAL_COUNTS["image_member_count"])
        self.assertEqual(
            validate_formal_image_inventory(paths, selection_manifest=manifest),
            paths,
        )
        mixed_extensions = list(paths)
        mixed_extensions[0] = mixed_extensions[0].removesuffix(".png") + ".jpg"
        mixed_extensions = tuple(sorted(mixed_extensions))
        self.assertEqual(
            validate_formal_image_inventory(
                mixed_extensions, selection_manifest=manifest
            ),
            mixed_extensions,
        )
        with self.assertRaisesRegex(ValueError, "image inventory mismatch"):
            validate_formal_image_inventory(paths[:-1], selection_manifest=manifest)

    def test_self_consistent_role_reorder_cannot_override_frozen_selection(self) -> None:
        selection = _selection(
            {
                "gate_train_expansion": ["first", "second"],
                "gate_development_expansion": [],
            }
        )
        manifest = {
            "role_source_ids": {
                "gate_train_expansion": ["second", "first"],
                "gate_development_expansion": [],
            }
        }
        trajectories = [
            {"role": "gate_train_expansion", "source_id": source_id}
            for source_id in ("second", "first")
        ]
        with self.assertRaisesRegex(ValueError, "frozen selection"):
            _validate_trajectories(
                trajectories,
                manifest=manifest,
                image_payloads={},
                ocr_records_by_path={},
                prepared_by_path={},
                selection_manifest=selection,
                event_builder=_fake_event_builder,
            )


class GUIOdysseyRestorationV2ExpansionArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source_id = "fixture-expansion"
        self.image_paths = [
            f"images/{self.source_id}/observation-{index:03d}.png"
            for index in range(6)
        ]
        self.images = {
            path: f"deterministic-fixture-image-{index}".encode("utf-8")
            for index, path in enumerate(self.image_paths)
        }
        self.backend = load_backend_config(BACKEND_CONFIG_PATH)
        self.backend_sha = _sha256(BACKEND_CONFIG_PATH.read_bytes())
        self.ocr_records = {}
        for index, path in enumerate(self.image_paths):
            record = {
                "backend_config_sha256": self.backend_sha,
                "image_member_path": path,
                "image_sha256": _sha256(self.images[path]),
                "full_spatial_tokens_sha256": _sha256(
                    f"tokens-{index}".encode("utf-8")
                ),
            }
            record["canonical_ocr_record_sha256"] = _sha256(
                json.dumps(record, sort_keys=True).encode("utf-8")
            )
            self.ocr_records[path] = record
        self.selection = _selection(
            {
                "gate_train_expansion": [self.source_id],
                "gate_development_expansion": [],
            }
        )
        self.pilot = _pilot(self.source_id, self.image_paths)

    def _records(self):
        return build_trajectory_records(
            pilots_by_source={self.source_id: self.pilot},
            source_image_payloads=self.images,
            selection_manifest=self.selection,
            ocr_records_by_path=self.ocr_records,
            backend_config=self.backend,
            backend_config_sha256=self.backend_sha,
            ocr_record_validator=_fake_ocr_validator,
            event_builder=_fake_event_builder,
        )

    def test_policy_blind_fixture_round_trip_is_deterministic_and_replays_ocr(self) -> None:
        trajectories, images = self._records()
        self.assertEqual(len(trajectories[0]["events"]), 5)
        self.assertEqual(len(trajectories[0]["decisions"]), 3)
        for decision in trajectories[0]["decisions"]:
            self.assertFalse(decision["current_expert_action_payload_included"])
            self.assertNotIn("validated_action", decision)
            self.assertNotIn("validated_action_sha256", decision)
            view = decision_view_events(trajectories[0]["events"], decision)
            self.assertEqual(
                [event["step_id"] for event in view],
                decision["history_event_step_ids"],
            )
            self.assertTrue(
                all(
                    event["step_id"] < decision["decision_step_id"]
                    for event in view
                )
            )
        inputs = _inputs(self.backend_sha)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            tree_one = write_artifact(
                output_dir=first,
                dataset_repo="fixture/restoration-v2-expansion",
                trajectories=trajectories,
                image_payloads=images,
                ocr_records_by_path=self.ocr_records,
                inputs=inputs,
                generator=_generator(),
                ocr_backend_provenance=_ocr_provenance(self.backend_sha),
                ocr_runtime_identity=_runtime_identity(self.backend),
                formal_counts_enforced=False,
            )
            tree_two = write_artifact(
                output_dir=second,
                dataset_repo="fixture/restoration-v2-expansion",
                trajectories=trajectories,
                image_payloads=images,
                ocr_records_by_path=self.ocr_records,
                inputs=inputs,
                generator=_generator(),
                ocr_backend_provenance=_ocr_provenance(self.backend_sha),
                ocr_runtime_identity=_runtime_identity(self.backend),
                formal_counts_enforced=False,
            )
            self.assertEqual(tree_one, tree_two)
            for relative in ARTIFACT_RELATIVE_PATHS:
                self.assertEqual(
                    (first / relative).read_bytes(),
                    (second / relative).read_bytes(),
                )
            calls: list[str] = []

            def exact_runner(**kwargs):
                calls.append(kwargs["image_member_path"])
                return copy.deepcopy(self.ocr_records[kwargs["image_member_path"]])

            result = validate_artifact(
                output_dir=first,
                backend_config=self.backend,
                backend_config_sha256=self.backend_sha,
                selection_manifest=self.selection,
                ocr_engine=object(),
                require_ocr_replay=True,
                ocr_record_runner=exact_runner,
                ocr_record_validator=_fake_ocr_validator,
                event_builder=_fake_event_builder,
            )
            self.assertEqual(calls, sorted(self.ocr_records))
            self.assertEqual(result["ocr_replay_record_count"], 6)
            self.assertEqual(
                result["counts"],
                {
                    "trajectory_count": 1,
                    "event_count": 5,
                    "state_count": 3,
                    "image_member_count": 6,
                    "ocr_record_count": 6,
                },
            )

    def test_validator_rejects_current_expert_action_payload(self) -> None:
        trajectories, images = self._records()
        trajectories[0]["decisions"][0]["validated_action"] = {
            "action_type": "tap"
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            write_artifact(
                output_dir=root,
                dataset_repo="fixture/restoration-v2-expansion",
                trajectories=trajectories,
                image_payloads=images,
                ocr_records_by_path=self.ocr_records,
                inputs=_inputs(self.backend_sha),
                generator=_generator(),
                ocr_backend_provenance=_ocr_provenance(self.backend_sha),
                ocr_runtime_identity=_runtime_identity(self.backend),
                formal_counts_enforced=False,
            )
            with self.assertRaisesRegex(ValueError, "expert action leaked"):
                validate_artifact(
                    output_dir=root,
                    backend_config=self.backend,
                    backend_config_sha256=self.backend_sha,
                    selection_manifest=self.selection,
                    ocr_record_validator=_fake_ocr_validator,
                    event_builder=_fake_event_builder,
                )

    def test_validator_rejects_payload_count_metadata_drift(self) -> None:
        trajectories, images = self._records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            write_artifact(
                output_dir=root,
                dataset_repo="fixture/restoration-v2-expansion",
                trajectories=trajectories,
                image_payloads=images,
                ocr_records_by_path=self.ocr_records,
                inputs=_inputs(self.backend_sha),
                generator=_generator(),
                ocr_backend_provenance=_ocr_provenance(self.backend_sha),
                ocr_runtime_identity=_runtime_identity(self.backend),
                formal_counts_enforced=False,
            )
            manifest_path = root / MANIFEST_RELATIVE_PATH
            manifest = json.loads(manifest_path.read_bytes())
            manifest["payload_files"][1]["record_count"] = 999
            manifest_path.write_bytes(pretty_json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "member/record counts drifted"):
                validate_artifact(
                    output_dir=root,
                    backend_config=self.backend,
                    backend_config_sha256=self.backend_sha,
                    selection_manifest=self.selection,
                    ocr_record_validator=_fake_ocr_validator,
                    event_builder=_fake_event_builder,
                )


if __name__ == "__main__":
    unittest.main()
