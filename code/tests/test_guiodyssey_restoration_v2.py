from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from causalcache.data.guiodyssey_restoration_v2 import (
    ARTIFACT_RELATIVE_PATHS,
    EXPECTED_FORMAL_COUNTS,
    FROZEN_OCR_BACKEND_MANIFEST_SHA256,
    build_ocr_backend_provenance,
    build_trajectory_records,
    generate_ocr_records,
    load_ocr_records_jsonl,
    ocr_record_aggregate_sha256,
    parse_canonical_jsonl,
    source_tool_call_to_v2_arguments,
    validate_artifact,
    validate_formal_image_inventory,
    validate_frozen_inputs,
    validate_ocr_record,
    validate_ocr_runtime_identity,
    write_artifact,
)
from causalcache.data.guiodyssey import canonicalize_tool_call
from causalcache.low_fidelity_v2 import action_argument
from causalcache.restoration_v2_text_backend import (
    build_ocr_record,
    decode_fixture_image,
    load_backend_config,
)
from causalcache.restoration_v2_contract import RestorationV2Contract


ROOT = Path(__file__).resolve().parents[2]
SELECTION_PATH = ROOT / "data/manifests/restoration_v2_selection.json"
BACKEND_CONFIG_PATH = ROOT / "code/configs/restoration_v2_ocr_backend.json"
OCR_FIXTURE_PATH = ROOT / "data/fixtures/restoration_v2_ocr_golden.json"
PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_executed_action(tool_call: dict) -> dict:
    action, _ = canonicalize_tool_call(tool_call, grid_size=10)
    return {
        "action_type": action.action_type.value,
        "target": action.target,
        "text_argument": action.text_argument,
        "text_case_sensitive": action.text_case_sensitive,
    }


def _identity_inputs(backend_sha256: str) -> dict[str, dict[str, str]]:
    records = {
        key: {"path": f"fixture/{key}.json", "sha256": index * 64}
        for key, index in zip(
            (
                "scientific_contract",
                "v1_config",
                "source_file_manifest",
                "selection_manifest",
                "exposure_manifest",
                "ocr_backend_config",
                "ocr_backend_manifest",
            ),
            "1234567",
            strict=True,
        )
    }
    records["ocr_backend_config"]["sha256"] = backend_sha256
    return records


def _generator() -> dict[str, str]:
    return {
        "git_revision": "a" * 40,
        "module_path": "code/causalcache/data/guiodyssey_restoration_v2.py",
        "module_sha256": "b" * 64,
        "build_cli_path": "code/scripts/build_guiodyssey_restoration_v2.py",
        "build_cli_sha256": "c" * 64,
        "validator_path": "code/scripts/validate_guiodyssey_restoration_v2.py",
        "validator_sha256": "d" * 64,
    }


def _ocr_backend_provenance(backend_sha256: str) -> dict[str, str]:
    return {
        "backend_id": "rapidocr-3.8.4-ppocrv5-mobile-en-cpu-v1",
        "backend_config_sha256": backend_sha256,
        "hf_model_repo": "fixture/ocr-model",
        "hf_model_revision": "e" * 40,
        "hf_model_tag": "v1.0.0",
        "real_screen_golden_dataset_revision": "f" * 40,
    }


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


def _fixture_selection(
    *,
    image_payloads: dict[str, bytes],
    validated_action: dict,
) -> dict:
    source_id = "fixture"
    image_sha = {path: _sha256(payload) for path, payload in image_payloads.items()}
    selection_record = {
        "source_id": source_id,
        "transport_file": "mobile/use/train/fixture.parquet",
        "transport_row_index": 0,
        "decision_count": 3,
        "instruction_sha256": _sha256(b"Fixture instruction"),
    }
    state = {
        "state_id": "fixture:decision_step:004",
        "source_id": source_id,
        "decision_step_id": 4,
        "history_event_step_ids": [1, 2, 3],
        "candidate_event_step_ids": [1, 2],
        "current_equivalent_event_step_id": 3,
        "candidate_event_post_states": [
            {
                "event_step_id": step,
                "post_state_member_path": f"images/fixture/observation-{step:03d}.png",
                "post_state_sha256": image_sha[
                    f"images/fixture/observation-{step:03d}.png"
                ],
            }
            for step in (1, 2)
        ],
        "current_observation": {
            "member_path": "images/fixture/observation-003.png",
            "sha256": image_sha["images/fixture/observation-003.png"],
        },
        "current_equivalence_witness": {
            "event_step_id": 3,
            "post_state_member_path": "images/fixture/observation-003.png",
            "post_state_sha256": image_sha[
                "images/fixture/observation-003.png"
            ],
        },
        "validated_action_sha256": _sha256(_canonical(validated_action)),
    }
    return {
        "roles": {
            "v2_label_train": {
                "trajectories": [selection_record],
                "states": [state],
            },
            "v2_development": {"trajectories": [], "states": []},
            "v2_confirm_primary": {"trajectories": [], "states": []},
        }
    }


def _fixture_pilot(image_payloads: dict[str, bytes], validated_action: dict) -> dict:
    paths = sorted(image_payloads)
    source_actions = [
        {
            "function": {
                "name": "tap",
                "arguments": {"coordinate": [100, 200], "clicks": 1},
            }
        },
        {
            "function": {
                "name": "swipe",
                "arguments": {
                    "start_coordinate": [500, 800],
                    "coordinate": [500, 200],
                },
            }
        },
        {
            "function": {
                "name": "type",
                "arguments": {"text": "Caf\u00e9"},
            }
        },
    ]
    executed_actions = [
        {
            "action_type": "tap",
            "target": "coordinate_bin:x0_y1",
            "text_argument": None,
            "text_case_sensitive": False,
        },
        {
            "action_type": "swipe",
            "target": "scroll:down",
            "text_argument": None,
            "text_case_sensitive": False,
        },
        {
            "action_type": "type_text",
            "target": "text_field_unknown",
            "text_argument": "Caf\u00e9",
            "text_case_sensitive": False,
        },
    ]
    events = [
        {
            "step_id": index + 1,
            "observation_before_path": paths[index],
            "observation_after_path": paths[index + 1],
            "executed_action": executed_actions[index],
            "source_tool_call": source_actions[index],
        }
        for index in range(3)
    ]
    return {
        "source": {
            "upstream_repo": "cua-lite/GUIOdyssey",
            "upstream_revision": "e" * 40,
            "transport_repo": "cua-lite/GUIOdyssey",
            "transport_revision": "f" * 40,
            "transport_file": "mobile/use/train/fixture.parquet",
            "transport_file_sha256": "1" * 64,
            "transport_row_index": 0,
            "license": "cc-by-4.0",
        },
        "trajectory": {
            "source_id": "fixture",
            "instruction": "Fixture instruction",
            "platform": "android",
            "apps": ["fixture"],
            "device_name": "fixture-device",
            "resolution": "8x8",
            "terminal_status": "success",
            "events": events,
            "decisions": [
                {
                    "decision_step_id": 4,
                    "current_observation_path": paths[3],
                    "history_event_step_ids": [1, 2, 3],
                    "validated_action": validated_action,
                    "validation_source": (
                        "recorded_successful_guiodyssey_trajectory"
                    ),
                }
            ],
        },
    }


class GUIOdysseyRestorationV2SchemaTest(unittest.TestCase):
    def test_committed_selection_implies_exact_complete_derived_counts(self) -> None:
        selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        roles = selection["roles"]
        trajectories = [
            record
            for role in (
                "v2_label_train",
                "v2_development",
                "v2_confirm_primary",
            )
            for record in roles[role]["trajectories"]
        ]
        states = [
            state
            for role in (
                "v2_label_train",
                "v2_development",
                "v2_confirm_primary",
            )
            for state in roles[role]["states"]
        ]
        self.assertEqual(len(trajectories), EXPECTED_FORMAL_COUNTS["trajectory_count"])
        self.assertEqual(len(states), EXPECTED_FORMAL_COUNTS["state_count"])
        self.assertTrue(all(max(state["history_event_step_ids"]) <= 5 for state in states))
        source_ids = {record["source_id"] for record in trajectories}
        expected_paths = {
            f"images/{source_id}/observation-{index:03d}.png"
            for source_id in source_ids
            for index in range(6)
        }
        self.assertEqual(len(expected_paths), EXPECTED_FORMAL_COUNTS["image_member_count"])
        self.assertEqual(5 * len(source_ids), EXPECTED_FORMAL_COUNTS["event_count"])
        ordered_paths = validate_formal_image_inventory(
            tuple(sorted(expected_paths)),
            selection_manifest=selection,
        )
        self.assertEqual(len(ordered_paths), 210)
        with self.assertRaisesRegex(ValueError, "image inventory mismatch"):
            validate_formal_image_inventory(
                ordered_paths[:-1],
                selection_manifest=selection,
            )

    def test_frozen_selection_exposure_and_ocr_inputs_are_cross_bound(self) -> None:
        selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        exposure_path = ROOT / "data/manifests/restoration_v2_exposure.json"
        exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
        contract = RestorationV2Contract.load(
            ROOT / "code/configs/causalcache_restoration_v2.json"
        )
        backend = load_backend_config(BACKEND_CONFIG_PATH)
        selection_sha = _sha256(SELECTION_PATH.read_bytes())
        v1_config_path = ROOT / "code/configs/independent_reference_gate_v1.json"
        source_manifest_path = (
            ROOT
            / "data/manifests/independent_reference_gate_v1_source_files.json"
        )
        v1_config_sha = _sha256(v1_config_path.read_bytes())
        source_manifest_sha = _sha256(source_manifest_path.read_bytes())
        validate_frozen_inputs(
            v2_contract=contract.data,
            selection_manifest=selection,
            selection_manifest_sha256=selection_sha,
            v1_config_sha256=v1_config_sha,
            source_file_manifest_sha256=source_manifest_sha,
            exposure_manifest=exposure,
            backend_config=backend,
        )
        with self.assertRaisesRegex(ValueError, "selection input bytes"):
            validate_frozen_inputs(
                v2_contract=contract.data,
                selection_manifest=selection,
                selection_manifest_sha256="0" * 64,
                v1_config_sha256=v1_config_sha,
                source_file_manifest_sha256=source_manifest_sha,
                exposure_manifest=exposure,
                backend_config=backend,
            )
        with self.assertRaisesRegex(ValueError, "actual v1 config bytes"):
            validate_frozen_inputs(
                v2_contract=contract.data,
                selection_manifest=selection,
                selection_manifest_sha256=selection_sha,
                v1_config_sha256="0" * 64,
                source_file_manifest_sha256=source_manifest_sha,
                exposure_manifest=exposure,
                backend_config=backend,
            )
        with self.assertRaisesRegex(ValueError, "actual source-file manifest bytes"):
            validate_frozen_inputs(
                v2_contract=contract.data,
                selection_manifest=selection,
                selection_manifest_sha256=selection_sha,
                v1_config_sha256=v1_config_sha,
                source_file_manifest_sha256="0" * 64,
                exposure_manifest=exposure,
                backend_config=backend,
            )
        backend_manifest_path = (
            ROOT / "data/manifests/restoration_v2_ocr_backend.json"
        )
        backend_manifest = json.loads(
            backend_manifest_path.read_text(encoding="utf-8")
        )
        backend_manifest_sha = _sha256(backend_manifest_path.read_bytes())
        self.assertEqual(
            backend_manifest_sha,
            FROZEN_OCR_BACKEND_MANIFEST_SHA256,
        )
        provenance = build_ocr_backend_provenance(
            backend_config_sha256=_sha256(BACKEND_CONFIG_PATH.read_bytes()),
            backend_manifest_sha256=backend_manifest_sha,
            backend_manifest=backend_manifest,
        )
        self.assertEqual(
            provenance["hf_model_revision"],
            "0dbc766a73ee88d10d52285d434dbfec58617835",
        )
        with self.assertRaisesRegex(ValueError, "frozen baseline milestone"):
            build_ocr_backend_provenance(
                backend_config_sha256=_sha256(BACKEND_CONFIG_PATH.read_bytes()),
                backend_manifest_sha256="0" * 64,
                backend_manifest=backend_manifest,
            )

    def test_source_action_adapter_reuses_v2_argument_contract_without_clamp(self) -> None:
        tap_tool = {
            "function": {
                "name": "tap",
                "arguments": {"coordinate": [999, 0], "clicks": 1},
            }
        }
        tap = source_tool_call_to_v2_arguments(
            tap_tool,
            executed_action=_canonical_executed_action(tap_tool),
        )
        self.assertEqual(tap, {"action": "click", "coordinate": [999, 0]})
        self.assertEqual(action_argument(tap), "coordinate_bin:x9_y0")
        swipe_tool = {
            "function": {
                "name": "swipe",
                "arguments": {
                    "start_coordinate": [500, 800],
                    "coordinate": [500, 200],
                },
            }
        }
        swipe = source_tool_call_to_v2_arguments(
            swipe_tool,
            executed_action=_canonical_executed_action(swipe_tool),
        )
        self.assertEqual(action_argument(swipe), "viewport_down:medium")
        boundary_tool = {
            "function": {
                "name": "tap",
                "arguments": {"coordinate": [1000, 0], "clicks": 1},
            }
        }
        unclamped = source_tool_call_to_v2_arguments(
            boundary_tool,
            executed_action=_canonical_executed_action(boundary_tool),
        )
        with self.assertRaisesRegex(ValueError, r"\[0, 999\]"):
            action_argument(unclamped)

    def test_source_action_adapter_rejects_schema_and_action_drift(self) -> None:
        cases = [
            (
                {
                    "function": {
                        "name": "tap",
                        "arguments": {"coordinate": [1, 2], "clicks": 2},
                    }
                },
            ),
            (
                {
                    "function": {
                        "name": "system_button",
                        "arguments": {"button": "Menu"},
                    }
                },
            ),
            (
                {
                    "function": {
                        "name": "key",
                        "arguments": {"text": "A"},
                    }
                },
            ),
        ]
        for (tool_call,) in cases:
            with self.subTest(tool_call=tool_call):
                with self.assertRaises(ValueError):
                    source_tool_call_to_v2_arguments(
                        tool_call,
                        executed_action=_canonical_executed_action(tool_call),
                    )

    def test_source_action_adapter_rejects_full_canonical_action_drift(self) -> None:
        tool_call = {
            "function": {
                "name": "type",
                "arguments": {"text": "Case Sensitive"},
            }
        }
        expected = _canonical_executed_action(tool_call)
        for field, changed_value in (
            ("target", "different-target"),
            ("text_argument", "different text"),
            ("text_case_sensitive", True),
        ):
            changed = dict(expected)
            changed[field] = changed_value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "full canonical"):
                    source_tool_call_to_v2_arguments(
                        tool_call,
                        executed_action=changed,
                    )

    def test_ocr_generation_runner_is_injected_and_member_path_sorted(self) -> None:
        backend = load_backend_config(BACKEND_CONFIG_PATH)
        backend_sha = _sha256(BACKEND_CONFIG_PATH.read_bytes())
        calls = []

        def runner(**kwargs):
            calls.append(kwargs["image_member_path"])
            return {
                "image_member_path": kwargs["image_member_path"],
                "image_sha256": _sha256(kwargs["image_bytes"]),
                "canonical_ocr_record_sha256": _sha256(
                    kwargs["image_member_path"].encode("utf-8")
                ),
            }

        records, aggregate = generate_ocr_records(
            engine=object(),
            backend_config=backend,
            backend_config_sha256=backend_sha,
            image_payloads={
                "images/fixture/b.png": b"second",
                "images/fixture/a.png": b"first",
            },
            record_runner=runner,
        )
        self.assertEqual(
            calls,
            ["images/fixture/a.png", "images/fixture/b.png"],
        )
        self.assertEqual(list(records), calls)
        self.assertEqual(aggregate, ocr_record_aggregate_sha256(records))

    def test_runtime_identity_is_frozen_and_tamper_fails_closed(self) -> None:
        backend = load_backend_config(BACKEND_CONFIG_PATH)
        identity = _runtime_identity(backend)
        validate_ocr_runtime_identity(identity, backend_config=backend)
        changed = copy.deepcopy(identity)
        changed["model_sha256"]["recognizer"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "model hashes drifted"):
            validate_ocr_runtime_identity(changed, backend_config=backend)

    def test_ocr_jsonl_loader_rejects_unsorted_duplicate_and_noncanonical_input(self) -> None:
        records = [
            {"image_member_path": "images/b.png", "value": 1},
            {"image_member_path": "images/a.png", "value": 2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.jsonl"
            path.write_bytes(b"\n".join(_canonical(record) for record in records) + b"\n")
            with self.assertRaisesRegex(ValueError, "sorted"):
                load_ocr_records_jsonl(path)
            path.write_text(
                '{"image_member_path":"images/a.png","value":1,"value":2}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_ocr_records_jsonl(path)
            path.write_text(
                '{"value": 1, "image_member_path": "images/a.png"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "canonical compact"):
                load_ocr_records_jsonl(path)

    def test_missing_or_extra_ocr_image_records_fail_before_materialization(self) -> None:
        selection = {
            "roles": {
                "v2_label_train": {
                    "trajectories": [{"source_id": "fixture"}],
                    "states": [
                        {
                            "source_id": "fixture",
                            "decision_step_id": 2,
                            "history_event_step_ids": [1],
                            "current_observation": {
                                "member_path": "images/fixture/observation-001.png"
                            },
                        }
                    ],
                },
                "v2_development": {"trajectories": [], "states": []},
                "v2_confirm_primary": {"trajectories": [], "states": []},
            }
        }
        pilot = {
            "trajectory": {
                "events": [
                    {
                        "step_id": 1,
                        "observation_before_path": "images/fixture/observation-000.png",
                        "observation_after_path": "images/fixture/observation-001.png",
                    }
                ]
            }
        }
        with self.assertRaisesRegex(ValueError, "OCR image inventory mismatch"):
            build_trajectory_records(
                pilots_by_source={"fixture": pilot},
                source_image_payloads={},
                selection_manifest=selection,
                ocr_records_by_path={
                    "images/fixture/observation-000.png": {},
                    "images/fixture/observation-001.png": {},
                    "images/fixture/extra.png": {},
                },
                backend_config={},
                backend_config_sha256="0" * 64,
            )


@unittest.skipUnless(PIL_AVAILABLE, "Pillow is an optional OCR dependency")
class GUIOdysseyRestorationV2ArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture = json.loads(OCR_FIXTURE_PATH.read_text(encoding="utf-8"))
        first = decode_fixture_image(fixture["cases"][0])
        second = decode_fixture_image(fixture["cases"][1])
        self.image_payloads = {
            f"images/fixture/observation-{index:03d}.png": (
                first if index % 2 == 0 else second
            )
            for index in range(4)
        }
        self.backend = load_backend_config(BACKEND_CONFIG_PATH)
        self.backend_sha = _sha256(BACKEND_CONFIG_PATH.read_bytes())
        self.ocr_records = {}
        for index, (path, payload) in enumerate(self.image_payloads.items()):
            self.ocr_records[path] = build_ocr_record(
                image_member_path=path,
                image_bytes=payload,
                backend_config_sha256=self.backend_sha,
                boxes=[[[0, 0], [1, 0], [1, 1], [0, 1]]],
                texts=[f"Token {index} repeated repeated"],
                scores=[0.75],
            )
        self.validated_action = {
            "action_type": "tap",
            "target": "coordinate_bin:x1_y2",
            "text_argument": None,
            "text_case_sensitive": True,
        }
        self.selection = _fixture_selection(
            image_payloads=self.image_payloads,
            validated_action=self.validated_action,
        )
        self.pilot = _fixture_pilot(
            self.image_payloads,
            self.validated_action,
        )

    def _records(self) -> tuple[list[dict], dict[str, bytes]]:
        return build_trajectory_records(
            pilots_by_source={"fixture": self.pilot},
            source_image_payloads=self.image_payloads,
            selection_manifest=self.selection,
            ocr_records_by_path=self.ocr_records,
            backend_config=self.backend,
            backend_config_sha256=self.backend_sha,
        )

    def _source_dataset(self) -> dict[str, str]:
        return {
            field: self.pilot["source"][field]
            for field in (
                "upstream_repo",
                "upstream_revision",
                "transport_repo",
                "transport_revision",
                "license",
            )
        }

    def test_full_ocr_record_validation_preserves_uncapped_nodes_and_tokens(self) -> None:
        path = "images/fixture/observation-000.png"
        record = self.ocr_records[path]
        validate_ocr_record(
            record,
            image_bytes=self.image_payloads[path],
            backend_config=self.backend,
            backend_config_sha256=self.backend_sha,
        )
        self.assertEqual(record["full_spatial_tokens"], ["Token", "0", "repeated", "repeated"])
        changed = copy.deepcopy(record)
        changed["full_spatial_tokens"].pop()
        with self.assertRaisesRegex(ValueError, "uncapped"):
            validate_ocr_record(
                changed,
                image_bytes=self.image_payloads[path],
                backend_config=self.backend,
                backend_config_sha256=self.backend_sha,
            )

    def test_small_fixture_materializes_and_validates_deterministically(self) -> None:
        trajectories, images = self._records()
        self.assertEqual((len(trajectories), len(trajectories[0]["events"])), (1, 3))
        first_event = trajectories[0]["events"][0]
        self.assertEqual(first_event["source_tool_call"]["function"]["name"], "tap")
        self.assertTrue(first_event["low_fidelity_v2_serialized"].endswith("\n"))
        self.assertEqual(
            first_event["high_fidelity_v2"]["image_role"],
            "post_action_state",
        )
        self.assertFalse(first_event["high_fidelity_v2"]["include_before_image"])
        inputs = _identity_inputs(self.backend_sha)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            tree_one = write_artifact(
                output_dir=first,
                dataset_repo="fixture/restoration-v2",
                trajectories=trajectories,
                image_payloads=images,
                ocr_records_by_path=self.ocr_records,
                inputs=inputs,
                generator=_generator(),
                ocr_backend_provenance=_ocr_backend_provenance(self.backend_sha),
                ocr_runtime_identity=_runtime_identity(self.backend),
                formal_counts_enforced=False,
            )
            tree_two = write_artifact(
                output_dir=second,
                dataset_repo="fixture/restoration-v2",
                trajectories=trajectories,
                image_payloads=images,
                ocr_records_by_path=self.ocr_records,
                inputs=inputs,
                generator=_generator(),
                ocr_backend_provenance=_ocr_backend_provenance(self.backend_sha),
                ocr_runtime_identity=_runtime_identity(self.backend),
                formal_counts_enforced=False,
            )
            self.assertEqual(tree_one, tree_two)
            for relative in ARTIFACT_RELATIVE_PATHS:
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
            result = validate_artifact(
                output_dir=first,
                backend_config=self.backend,
                backend_config_sha256=self.backend_sha,
                selection_manifest=self.selection,
                expected_input_sha256={key: value["sha256"] for key, value in inputs.items()},
                expected_ocr_backend_provenance=_ocr_backend_provenance(
                    self.backend_sha
                ),
                expected_ocr_runtime_identity=_runtime_identity(self.backend),
                expected_generator_git_revision="a" * 40,
                expected_dataset_repo="fixture/restoration-v2",
                expected_source_dataset=self._source_dataset(),
            )
            self.assertEqual(
                result["outcome"],
                "PASSED_GUIODYSSEY_RESTORATION_V2_ARTIFACT_VALIDATION",
            )
            self.assertEqual(
                result["counts"],
                {
                    "trajectory_count": 1,
                    "event_count": 3,
                    "state_count": 1,
                    "image_member_count": 4,
                    "ocr_record_count": 4,
                },
            )
            parsed_ocr = parse_canonical_jsonl(
                (
                    first
                    / "derived/restoration-v2-v1/ocr-records-00000-of-00001.jsonl"
                ).read_bytes(),
                label="fixture OCR",
            )
            self.assertEqual(len(parsed_ocr), 4)
            source_record = self.ocr_records[parsed_ocr[0]["image_member_path"]]
            self.assertEqual(parsed_ocr[0]["nodes"], source_record["nodes"])

    def test_validator_replays_every_ocr_record_with_injected_runner(self) -> None:
        trajectories, images = self._records()
        inputs = _identity_inputs(self.backend_sha)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            write_artifact(
                output_dir=root,
                dataset_repo="fixture/restoration-v2",
                trajectories=trajectories,
                image_payloads=images,
                ocr_records_by_path=self.ocr_records,
                inputs=inputs,
                generator=_generator(),
                ocr_backend_provenance=_ocr_backend_provenance(self.backend_sha),
                ocr_runtime_identity=_runtime_identity(self.backend),
                formal_counts_enforced=False,
            )
            calls = []

            def exact_runner(**kwargs):
                calls.append(kwargs["image_member_path"])
                return copy.deepcopy(self.ocr_records[kwargs["image_member_path"]])

            result = validate_artifact(
                output_dir=root,
                backend_config=self.backend,
                backend_config_sha256=self.backend_sha,
                ocr_engine=object(),
                require_ocr_replay=True,
                ocr_record_runner=exact_runner,
            )
            self.assertEqual(calls, sorted(self.ocr_records))
            self.assertEqual(result["ocr_replay_record_count"], 4)
            self.assertEqual(
                result["ocr_replay_aggregate_sha256"],
                ocr_record_aggregate_sha256(self.ocr_records),
            )

            def drifted_runner(**kwargs):
                record = copy.deepcopy(
                    self.ocr_records[kwargs["image_member_path"]]
                )
                record["full_spatial_tokens"].append("drift")
                return record

            with self.assertRaisesRegex(ValueError, "replayed OCR record differs"):
                validate_artifact(
                    output_dir=root,
                    backend_config=self.backend,
                    backend_config_sha256=self.backend_sha,
                    ocr_engine=object(),
                    require_ocr_replay=True,
                    ocr_record_runner=drifted_runner,
                )

    def test_formal_validation_and_external_identity_bindings_fail_closed(self) -> None:
        trajectories, images = self._records()
        inputs = _identity_inputs(self.backend_sha)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            write_artifact(
                output_dir=root,
                dataset_repo="fixture/restoration-v2",
                trajectories=trajectories,
                image_payloads=images,
                ocr_records_by_path=self.ocr_records,
                inputs=inputs,
                generator=_generator(),
                ocr_backend_provenance=_ocr_backend_provenance(self.backend_sha),
                ocr_runtime_identity=_runtime_identity(self.backend),
                formal_counts_enforced=False,
            )
            with self.assertRaisesRegex(ValueError, "formal_counts_enforced=true"):
                validate_artifact(
                    output_dir=root,
                    backend_config=self.backend,
                    backend_config_sha256=self.backend_sha,
                    require_formal=True,
                )
            for kwargs, message in (
                ({"expected_generator_git_revision": "0" * 40}, "Git revision"),
                ({"expected_dataset_repo": "different/repo"}, "dataset repo"),
                (
                    {
                        "expected_source_dataset": {
                            **self._source_dataset(),
                            "upstream_repo": "different/source",
                        }
                    },
                    "source dataset",
                ),
            ):
                with self.subTest(kwargs=kwargs):
                    with self.assertRaisesRegex(ValueError, message):
                        validate_artifact(
                            output_dir=root,
                            backend_config=self.backend,
                            backend_config_sha256=self.backend_sha,
                            **kwargs,
                        )
            manifest_path = root / "derived/restoration-v2-v1/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["formal_counts_enforced"] = True
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "formal derived artifact counts"):
                validate_artifact(
                    output_dir=root,
                    backend_config=self.backend,
                    backend_config_sha256=self.backend_sha,
                    require_formal=True,
                )

    def test_validator_rejects_extra_files_and_payload_tampering(self) -> None:
        trajectories, images = self._records()
        inputs = _identity_inputs(self.backend_sha)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            write_artifact(
                output_dir=root,
                dataset_repo="fixture/restoration-v2",
                trajectories=trajectories,
                image_payloads=images,
                ocr_records_by_path=self.ocr_records,
                inputs=inputs,
                generator=_generator(),
                ocr_backend_provenance=_ocr_backend_provenance(self.backend_sha),
                ocr_runtime_identity=_runtime_identity(self.backend),
                formal_counts_enforced=False,
            )
            (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file inventory drifted"):
                validate_artifact(
                    output_dir=root,
                    backend_config=self.backend,
                    backend_config_sha256=self.backend_sha,
                )
            (root / "unexpected.txt").unlink()
            readme_path = root / "README.md"
            readme_bytes = readme_path.read_bytes()
            readme_path.write_bytes(readme_bytes + b"tamper\n")
            with self.assertRaisesRegex(ValueError, "README drifted"):
                validate_artifact(
                    output_dir=root,
                    backend_config=self.backend,
                    backend_config_sha256=self.backend_sha,
                )
            readme_path.write_bytes(readme_bytes)
            trajectory_path = (
                root
                / "derived/restoration-v2-v1/trajectories-00000-of-00001.jsonl"
            )
            trajectory_path.write_bytes(trajectory_path.read_bytes() + b"{}\n")
            with self.assertRaisesRegex(ValueError, "payload file identity drifted"):
                validate_artifact(
                    output_dir=root,
                    backend_config=self.backend,
                    backend_config_sha256=self.backend_sha,
                )

    def test_manifest_runtime_identity_tamper_is_rejected(self) -> None:
        trajectories, images = self._records()
        inputs = _identity_inputs(self.backend_sha)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            write_artifact(
                output_dir=root,
                dataset_repo="fixture/restoration-v2",
                trajectories=trajectories,
                image_payloads=images,
                ocr_records_by_path=self.ocr_records,
                inputs=inputs,
                generator=_generator(),
                ocr_backend_provenance=_ocr_backend_provenance(self.backend_sha),
                ocr_runtime_identity=_runtime_identity(self.backend),
                formal_counts_enforced=False,
            )
            manifest_path = root / "derived/restoration-v2-v1/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["ocr_runtime"]["model_sha256"]["recognizer"] = "0" * 64
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "model hashes drifted"):
                validate_artifact(
                    output_dir=root,
                    backend_config=self.backend,
                    backend_config_sha256=self.backend_sha,
                )


if __name__ == "__main__":
    unittest.main()
