from __future__ import annotations

import copy
import hashlib
import inspect
import io
import itertools
import json
import tarfile
import unittest
from pathlib import Path
from typing import Literal
from unittest import mock

from causalcache.gate_v1_contract import (
    canonical_json_bytes,
    derive_rosters,
    sha256_bytes,
)
from causalcache.gate_v1_formal_cache import (
    EXPANSION_FEATURE_MANIFEST,
    EXPANSION_FEATURE_OCR,
    EXPANSION_FEATURE_TRAJECTORIES,
    EXPANSION_LABEL_ARCHIVE,
    EXPANSION_LABEL_SIDECAR,
    EXPANSION_FEATURE_PREFIX,
    EXPANSION_LABEL_PREFIX,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_CONDITIONAL_TARGET_COUNT,
    EXPECTED_INDEPENDENT_TARGET_COUNT,
    EXPECTED_RAW_DISTANCE_ROW_COUNT,
    EXPECTED_STATE_COUNT,
    EXPECTED_TRAJECTORY_COUNT,
    FEATURE_CACHE_PREFIX,
    FEATURE_SOURCE_KEYS,
    LABEL_SOURCE_KEYS,
    LEGACY_FEATURE_MANIFEST,
    LEGACY_FEATURE_OCR,
    LEGACY_FEATURE_PREFIX,
    LEGACY_FEATURE_TRAJECTORIES,
    LEGACY_LABEL_ARCHIVE,
    LEGACY_LABEL_PREFIX,
    LABEL_CACHE_PREFIX,
    _cache_archive,
    _deterministic_ustar,
    _expected_phase_bindings,
    _float_to_hex,
    _ocr_path_without_json_decode,
    _read_cache_members,
    _strict_canonical_object,
    audit_formal_cache_join,
    audit_formal_cache_join_transport_repair_v1,
    build_formal_feature_cache,
    build_formal_feature_cache_transport_repair_v1,
    build_formal_label_cache,
    build_formal_label_cache_transport_repair_v1,
    read_feature_cache,
    read_label_cache,
    TRANSPORT_REPAIR_CORRECTED_SHA256,
    TRANSPORT_REPAIR_FILE_PATH,
    TRANSPORT_REPAIR_SIZE_BYTES,
)


SENTINEL = "EXCLUDED_SEMANTIC_SENTINEL"
SHA = "a" * 64


def _pretty(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _jsonl(values: list[object]) -> bytes:
    return b"".join(canonical_json_bytes(value) + b"\n" for value in values)


def _binding(payload: bytes) -> dict[str, object]:
    return {"sha256": sha256_bytes(payload), "size_bytes": len(payload)}


def _low_fidelity(step: int) -> dict[str, object]:
    return {
        "step_id": step,
        "action_type": "click",
        "action_argument": f"coordinate_bin:x{step}_y{step}",
        "foreground_app": "settings",
        "screen_text_added": [f"after-{step}"],
        "screen_text_removed": [],
        "screen_change": "low",
        "executor_result": "accepted",
    }


def _event(source_id: str, step: int) -> dict[str, object]:
    before = f"images/{source_id}/observation-{step - 1:03d}.png"
    after = f"images/{source_id}/observation-{step:03d}.png"
    low = _low_fidelity(step)
    return {
        "step_id": step,
        "observation_before_path": before,
        "observation_before_sha256": SHA,
        "observation_after_path": after,
        "observation_after_sha256": SHA,
        "executed_action": {"action": "click"},
        "source_tool_call": {"name": "click"},
        "low_fidelity_v2": low,
        "low_fidelity_v2_serialized": "low",
        "low_fidelity_v2_sha256": SHA,
        "low_fidelity_v2_metadata": {},
        "high_fidelity_v2": {},
        "ocr_record_refs": [],
    }


def _decision(source_id: str, step: int, *, expansion: bool) -> dict[str, object]:
    base: dict[str, object] = {
        "state_id": f"{source_id}:decision_step:{step:03d}",
        "decision_step_id": step,
        "history_event_step_ids": list(range(1, step)),
        "candidate_event_step_ids": list(range(1, step - 1)),
        "current_equivalent_event_step_id": step - 1,
        "current_observation_path": f"images/{source_id}/observation-{step - 1:03d}.png",
        "current_observation_sha256": SHA,
    }
    if expansion:
        base.update(
            {
                "candidate_event_post_states": [],
                "current_equivalence_witness": {},
                "content_witness_sha256": SHA,
                "current_expert_action_payload_included": False,
            }
        )
    else:
        base.update(
            {
                "validated_action": {"action": "click"},
                "validated_action_sha256": SHA,
                "validation_source": "fixture",
            }
        )
    return base


def _trajectory(source_id: str, *, expansion: bool) -> dict[str, object]:
    instruction = f"instruction-{source_id}"
    value: dict[str, object] = {
        "source_id": source_id,
        "role": "gate_train_expansion" if expansion else "v2_label_train",
        "instruction": instruction,
        "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        "platform": "android",
        "apps": ["settings"],
        "device_name": "fixture",
        "resolution": [1080, 1920],
        "terminal_status": "success",
        "source": {},
        "selection": {},
        "events": [_event(source_id, step) for step in range(1, 6)],
        "decisions": [
            _decision(source_id, step, expansion=expansion) for step in (4, 5, 6)
        ],
    }
    if expansion:
        value["content_witness_sha256"] = SHA
    return value


def _ocr_record(path: str) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "backend_id": "fixture",
        "backend_config_sha256": SHA,
        "image_member_path": path,
        "image_sha256": SHA,
        "source_format": "PNG",
        "source_mode": "RGB",
        "width": 1080,
        "height": 1920,
        "exif_present": False,
        "alpha_extrema": None,
        "rgb_bytes_sha256": SHA,
        "resized_rgb_256x256_sha256": SHA,
        "nodes": [],
        "full_spatial_tokens": [f"token:{path}"],
        "full_spatial_tokens_sha256": SHA,
        "canonical_ocr_record_sha256": SHA,
    }


def _feature_payloads(
    source_ids: tuple[str, ...], *, expansion: bool
) -> tuple[bytes, bytes, bytes]:
    full_trajectory_count = 64 if expansion else 35
    full_ocr_count = 384 if expansion else 210
    selected = [_trajectory(source_id, expansion=expansion) for source_id in source_ids]
    trajectory_lines = [canonical_json_bytes(value) for value in selected]
    trajectory_lines.extend(
        f'{{"source_id":"excluded-{index}",{SENTINEL}}}'.encode()
        for index in range(full_trajectory_count - len(selected))
    )
    trajectories = b"\n".join(trajectory_lines) + b"\n"

    selected_paths = [
        f"images/{source_id}/observation-{step:03d}.png"
        for source_id in source_ids
        for step in range(1, 6)
    ]
    path_lines = [(path, canonical_json_bytes(_ocr_record(path))) for path in selected_paths]
    path_lines.extend(
        (
            f"zz-{index:04d}/excluded.png",
            f'{{"image_member_path":"zz-{index:04d}/excluded.png",{SENTINEL}}}'.encode(),
        )
        for index in range(full_ocr_count - len(path_lines))
    )
    path_lines.sort(key=lambda item: item[0])
    ocr = b"\n".join(line for _, line in path_lines) + b"\n"

    prefix = EXPANSION_FEATURE_PREFIX if expansion else LEGACY_FEATURE_PREFIX
    roles = (
        {
            "gate_train_expansion": list(source_ids),
            "gate_development_expansion": [f"exp-dev-{index:02d}" for index in range(16)],
        }
        if expansion
        else {
            "v2_label_train": list(source_ids),
            "v2_development": [f"legacy-dev-{index:02d}" for index in range(5)],
            "v2_confirm_primary": [f"confirm-{index:02d}" for index in range(20)],
        }
    )
    counts = (
        {
            "trajectory_count": 64,
            "event_count": 320,
            "state_count": 192,
            "image_member_count": 384,
            "ocr_record_count": 384,
        }
        if expansion
        else {
            "trajectory_count": 35,
            "event_count": 175,
            "state_count": 65,
            "image_member_count": 210,
            "ocr_record_count": 210,
        }
    )
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "protocol_id": "fixture",
        "artifact_id": "fixture",
        "status": "fixture",
        "dataset_repo": SENTINEL,
        "license": "cc-by-4.0",
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "formal_counts_enforced": True,
        "inputs": {},
        "generator": {},
        "ocr_backend": {},
        "ocr_runtime": {},
        "source_dataset": {},
        "role_source_ids": roles,
        "counts": counts,
        "inventories": {},
        "payload_files": [
            {
                "path": f"{prefix}/images-00000-of-00001.tar",
                "sha256": SHA,
                "size_bytes": 1,
                "member_count": full_ocr_count,
            },
            {
                "path": f"{prefix}/ocr-records-00000-of-00001.jsonl",
                "sha256": sha256_bytes(ocr),
                "size_bytes": len(ocr),
                "record_count": full_ocr_count,
            },
            {
                "path": f"{prefix}/trajectories-00000-of-00001.jsonl",
                "sha256": sha256_bytes(trajectories),
                "size_bytes": len(trajectories),
                "record_count": full_trajectory_count,
            },
        ],
    }
    if expansion:
        manifest.update(
            {
                "per_decision_view_current_expert_action_payload_included": False,
                "consumer_must_slice_events_by_history_event_step_ids": True,
            }
        )
    return _pretty(manifest), trajectories, ocr


def _coalitions(event_ids: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [
        subset
        for size in range(len(event_ids) + 1)
        for subset in itertools.combinations(event_ids, size)
    ]


def _legacy_projection(index: int, source_ids: tuple[str, ...]) -> dict[str, object]:
    source_id = source_ids[index // 3] if index < 30 else f"legacy-dev-{index // 3:02d}"
    step = 4 + index % 3
    return {
        "index": index,
        "role": "v2_label_train" if index < 30 else "v2_development",
        "trajectory_id": source_id,
        "decision_step_id": step,
        "state_id": f"{source_id}:decision_step:{step:03d}",
        "candidate_event_step_ids": list(range(1, step - 1)),
    }


def _legacy_state(index: int, source_ids: tuple[str, ...]) -> dict[str, object]:
    state = _legacy_projection(index, source_ids)
    event_ids = tuple(state["candidate_event_step_ids"])
    rows = [
        {
            "coalition_event_step_ids": list(coalition),
            "distance_kl": float(len(event_ids) - len(coalition)),
        }
        for coalition in _coalitions(event_ids)
    ]
    return {"state": state, "distance_rows": rows}


def _legacy_label_archive(source_ids: tuple[str, ...]) -> bytes:
    relative_names = {
        "global_attempt_ledger.json",
        "worker_sibling_ledgers/even.json",
        "worker_sibling_ledgers/odd.json",
        "run_manifest.json",
        "aggregate.json",
    }
    for worker, parity in (("even", 0), ("odd", 1)):
        base = f"workers/{worker}"
        relative_names.update(
            {
                f"{base}/worker_attempt_ledger.json",
                f"{base}/runtime_identity.json",
                f"{base}/terminal.json",
            }
        )
        for index in range(parity, 45, 2):
            relative_names.add(f"{base}/attempts/{index:03d}.json")
            relative_names.add(f"{base}/states/{index:03d}.json")
    files = {name: SENTINEL.encode() for name in relative_names}
    files["run_manifest.json"] = SENTINEL.encode()
    for index in range(30):
        worker = "even" if index % 2 == 0 else "odd"
        files[f"workers/{worker}/states/{index:03d}.json"] = _pretty(
            _legacy_state(index, source_ids)
        )
    return _deterministic_ustar(
        {f"{LEGACY_LABEL_PREFIX}/{name}": value for name, value in files.items()}
    )


def _expansion_state(index: int, source_ids: tuple[str, ...]) -> dict[str, object]:
    source_id = source_ids[index // 3]
    step = 4 + index % 3
    event_ids = tuple(range(1, step - 1))
    rows = []
    for coalition in _coalitions(event_ids):
        is_full = coalition == event_ids
        rows.append(
            {
                "coalition": list(coalition),
                "distance": float(len(event_ids) - len(coalition)),
                "candidate_input_sha256": SHA,
                "teacher_forward_count": 0 if is_full else 1,
                "kl_measurement_count": 0 if is_full else 1,
                "scalar_host_transfer_count": 0 if is_full else 1,
                "is_full_history_reference": is_full,
                "full_logit_tensor_host_transfer_count": 0,
            }
        )
    return {
        "schema_version": "1.0.0",
        "protocol_id": "fixture",
        "status": "fixture",
        "run_contract_sha256": SHA,
        "worker": {},
        "state": {
            "state_index": index,
            "role": "gate_train_expansion",
            "source_id": source_id,
            "decision_step_id": step,
            "state_id": f"{source_id}:decision_step:{step:03d}",
            "candidate_event_step_ids": list(event_ids),
        },
        "reference_teacher": {},
        "distance_rows": rows,
        "operation_counts": {},
    }


def _expansion_label_archive(source_ids: tuple[str, ...]) -> bytes:
    lines = [canonical_json_bytes(_expansion_state(index, source_ids)) for index in range(144)]
    lines.extend(
        f'{{"state_index":{index},{SENTINEL}}}'.encode() for index in range(144, 192)
    )
    raw = b"\n".join(lines) + b"\n"
    files: dict[str, bytes] = {
        "audit.json": _pretty({"status": "fixture"}),
        "derived_labels.jsonl": f"{SENTINEL}\n".encode(),
        "raw_states.jsonl": raw,
    }
    inventory = [
        {"path": name, "sha256": sha256_bytes(files[name]), "size_bytes": len(files[name])}
        for name in ("audit.json", "derived_labels.jsonl", "raw_states.jsonl")
    ]
    files["manifest.json"] = _pretty(
        {
            "schema_version": "1.0.0",
            "protocol_id": "fixture",
            "status": "fixture",
            "runner_config_sha256": SHA,
            "source_git_commit": "a" * 40,
            "raw_state_count": 192,
            "derived_state_count": 192,
            "payload_inventory": inventory,
            "scientific_payload_sha256": SHA,
            "original_attempt_remains_invalid": True,
            "producer_reclassified": False,
            "formal_consumption": {},
        }
    )
    return _deterministic_ustar(
        {f"{EXPANSION_LABEL_PREFIX}/{name}": value for name, value in files.items()}
    )


def _fixture() -> tuple[
    dict[str, bytes],
    dict[str, dict[str, object]],
    dict[str, list[str]],
    dict[str, object],
]:
    root = Path(__file__).resolve().parents[2]
    legacy_selection = json.loads(
        (root / "data/manifests/restoration_v2_selection.json").read_text()
    )
    expansion_selection = json.loads(
        (
            root
            / "code/configs/causalcache_restoration_v2_2_label_expansion_v1.json"
        ).read_text()
    )
    derived = derive_rosters(legacy_selection, expansion_selection)
    legacy_ids = derived["legacy_train"]
    expansion_ids = derived["fresh_train_expansion"]
    legacy_manifest, legacy_trajectories, legacy_ocr = _feature_payloads(
        legacy_ids, expansion=False
    )
    expansion_manifest, expansion_trajectories, expansion_ocr = _feature_payloads(
        expansion_ids, expansion=True
    )
    downloaded = {
        LEGACY_FEATURE_MANIFEST: legacy_manifest,
        LEGACY_FEATURE_TRAJECTORIES: legacy_trajectories,
        LEGACY_FEATURE_OCR: legacy_ocr,
        EXPANSION_FEATURE_MANIFEST: expansion_manifest,
        EXPANSION_FEATURE_TRAJECTORIES: expansion_trajectories,
        EXPANSION_FEATURE_OCR: expansion_ocr,
        LEGACY_LABEL_ARCHIVE: _legacy_label_archive(legacy_ids),
        EXPANSION_LABEL_ARCHIVE: _expansion_label_archive(expansion_ids),
        EXPANSION_LABEL_SIDECAR: f"{SENTINEL}:sidecar".encode(),
    }
    bindings = {name: _binding(payload) for name, payload in downloaded.items()}
    rosters = {
        "legacy_train": list(legacy_ids),
        "fresh_train_expansion": list(expansion_ids),
        "formal_train": list(derived["formal_train"]),
    }
    config = json.loads(
        (root / "code/configs/causalcache_gate_v1_formal_cache_v1.json").read_text()
    )
    return downloaded, bindings, rosters, config


class GateV1FormalCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.downloaded, cls.bindings, cls.rosters, cls.config = _fixture()
        root = Path(__file__).resolve().parents[2]
        cls.transport_repair_config = json.loads(
            (
                root
                / "code/configs/causalcache_gate_v1_formal_cache_transport_repair_v1.json"
            ).read_text()
        )

    def _build_feature(self):
        expected = {name: self.bindings[name] for name in FEATURE_SOURCE_KEYS}
        with mock.patch(
            "causalcache.gate_v1_formal_cache._expected_phase_bindings",
            return_value=expected,
        ):
            return build_formal_feature_cache(
                {name: self.downloaded[name] for name in FEATURE_SOURCE_KEYS},
                transport_bindings=expected,
                frozen_rosters=self.rosters,
                frozen_config=self.config,
            )

    def _build_label(self):
        expected = {name: self.bindings[name] for name in LABEL_SOURCE_KEYS}
        with mock.patch(
            "causalcache.gate_v1_formal_cache._expected_phase_bindings",
            return_value=expected,
        ):
            return build_formal_label_cache(
                {name: self.downloaded[name] for name in LABEL_SOURCE_KEYS},
                transport_bindings=expected,
                frozen_rosters=self.rosters,
                frozen_config=self.config,
            )

    def _repair_bindings(
        self, *, kind: Literal["feature", "label"]
    ) -> dict[str, dict[str, object]]:
        return _expected_phase_bindings(self.transport_repair_config, kind=kind)

    def _build_repair_feature(self):
        expected = self._repair_bindings(kind="feature")
        with mock.patch(
            "causalcache.gate_v1_formal_cache._verify_downloads",
            return_value=expected,
        ):
            return build_formal_feature_cache_transport_repair_v1(
                {name: self.downloaded[name] for name in FEATURE_SOURCE_KEYS},
                transport_bindings=expected,
                frozen_rosters=self.rosters,
                frozen_config=self.transport_repair_config,
            )

    def _build_repair_label(self):
        expected = self._repair_bindings(kind="label")
        with mock.patch(
            "causalcache.gate_v1_formal_cache._verify_downloads",
            return_value=expected,
        ):
            return build_formal_label_cache_transport_repair_v1(
                {name: self.downloaded[name] for name in LABEL_SOURCE_KEYS},
                transport_bindings=expected,
                frozen_rosters=self.rosters,
                frozen_config=self.transport_repair_config,
            )

    def test_selective_materialization_is_deterministic_and_separated(self) -> None:
        original_loads = json.loads

        def guarded_loads(value: object, *args: object, **kwargs: object) -> object:
            text = value.decode() if isinstance(value, bytes) else str(value)
            if SENTINEL in text:
                raise AssertionError("excluded semantic payload reached json.loads")
            return original_loads(value, *args, **kwargs)

        with mock.patch(
            "causalcache.gate_v1_formal_cache.json.loads", side_effect=guarded_loads
        ):
            first_feature = self._build_feature()
            first_label = self._build_label()
            second_feature = self._build_feature()
            second_label = self._build_label()
            join = audit_formal_cache_join(
                first_feature.archive,
                first_label.archive,
                frozen_rosters=self.rosters,
                frozen_config=self.config,
            )
        self.assertEqual(first_feature.archive, second_feature.archive)
        self.assertEqual(first_label.archive, second_label.archive)
        self.assertNotEqual(first_feature.archive, first_label.archive)
        with tarfile.open(fileobj=io.BytesIO(first_feature.archive), mode="r:") as archive:
            self.assertEqual(
                [member.name for member in archive.getmembers()],
                [
                    f"{FEATURE_CACHE_PREFIX}/feature_states.jsonl",
                    f"{FEATURE_CACHE_PREFIX}/manifest.json",
                ],
            )
        with tarfile.open(fileobj=io.BytesIO(first_label.archive), mode="r:") as archive:
            self.assertEqual(
                [member.name for member in archive.getmembers()],
                [
                    f"{LABEL_CACHE_PREFIX}/label_states.jsonl",
                    f"{LABEL_CACHE_PREFIX}/manifest.json",
                ],
            )
        self.assertEqual(
            join.counts,
            {
                "trajectory_count": EXPECTED_TRAJECTORY_COUNT,
                "state_count": EXPECTED_STATE_COUNT,
                "candidate_feature_count": EXPECTED_CANDIDATE_COUNT,
                "distance_value_count": EXPECTED_RAW_DISTANCE_ROW_COUNT,
                "conditional_edge_count": EXPECTED_CONDITIONAL_TARGET_COUNT,
                "independent_target_count": EXPECTED_INDEPENDENT_TARGET_COUNT,
            },
        )
        self.assertEqual(len(read_feature_cache(first_feature.archive)), EXPECTED_STATE_COUNT)
        labels = read_label_cache(first_label.archive)
        self.assertEqual(len(labels), EXPECTED_STATE_COUNT)
        self.assertTrue(all(state.table.distance(state.table.event_ids) == 0.0 for state in labels))

        feature_files = _read_cache_members(first_feature.archive, kind="feature")
        label_files = _read_cache_members(first_label.archive, kind="label")
        self.assertNotIn(b"distance_kl_f64_hex", b"".join(feature_files.values()))
        self.assertNotIn(b"q64_f64_hex", b"".join(label_files.values()))
        self.assertNotIn(SENTINEL.encode(), b"".join(feature_files.values()))
        self.assertNotIn(SENTINEL.encode(), b"".join(label_files.values()))
        self.assertNotIn(b'"role"', b"".join(feature_files.values()))
        self.assertNotIn(b'"split"', b"".join(label_files.values()))
        feature_manifest = json.loads(feature_files["manifest.json"])
        label_manifest = json.loads(label_files["manifest.json"])
        self.assertEqual(set(feature_manifest["source_bindings"]), set(FEATURE_SOURCE_KEYS))
        self.assertEqual(set(label_manifest["source_bindings"]), set(LABEL_SOURCE_KEYS))
        self.assertEqual(set(feature_manifest["source_bindings"]) & set(label_manifest["source_bindings"]), set())

    def test_transport_and_roster_fail_closed(self) -> None:
        bad_bindings = {
            name: copy.deepcopy(self.bindings[name]) for name in FEATURE_SOURCE_KEYS
        }
        bad_bindings[LEGACY_FEATURE_MANIFEST]["size_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "transport identity"):
            build_formal_feature_cache(
                {name: self.downloaded[name] for name in FEATURE_SOURCE_KEYS},
                transport_bindings=bad_bindings,
                frozen_rosters=self.rosters,
                frozen_config=self.config,
            )

    def test_formal_phase_apis_and_frozen_bindings_are_separate(self) -> None:
        feature_parameters = set(inspect.signature(build_formal_feature_cache).parameters)
        label_parameters = set(inspect.signature(build_formal_label_cache).parameters)
        self.assertEqual(
            inspect.signature(build_formal_feature_cache),
            inspect.signature(build_formal_feature_cache_transport_repair_v1),
        )
        self.assertEqual(
            inspect.signature(build_formal_label_cache),
            inspect.signature(build_formal_label_cache_transport_repair_v1),
        )
        self.assertIn("downloaded_features", feature_parameters)
        self.assertNotIn("downloaded_labels", feature_parameters)
        self.assertIn("downloaded_labels", label_parameters)
        self.assertNotIn("downloaded_features", label_parameters)
        self.assertEqual(
            set(_expected_phase_bindings(self.config, kind="feature")),
            set(FEATURE_SOURCE_KEYS),
        )
        self.assertEqual(
            set(_expected_phase_bindings(self.config, kind="label")),
            set(LABEL_SOURCE_KEYS),
        )
        with self.assertRaisesRegex(ValueError, "blob inventory"):
            build_formal_feature_cache(
                self.downloaded,
                transport_bindings=self.bindings,
                frozen_rosters=self.rosters,
                frozen_config=self.config,
            )
        with self.assertRaisesRegex(ValueError, "frozen source contract"):
            build_formal_feature_cache(
                {name: self.downloaded[name] for name in FEATURE_SOURCE_KEYS},
                transport_bindings={
                    name: self.bindings[name] for name in FEATURE_SOURCE_KEYS
                },
                frozen_rosters=self.rosters,
                frozen_config=self.config,
            )
        bad_rosters = copy.deepcopy(self.rosters)
        bad_rosters["formal_train"][0], bad_rosters["formal_train"][1] = (
            bad_rosters["formal_train"][1],
            bad_rosters["formal_train"][0],
        )
        with self.assertRaisesRegex(ValueError, "composition"):
            build_formal_feature_cache(
                {name: self.downloaded[name] for name in FEATURE_SOURCE_KEYS},
                transport_bindings={
                    name: self.bindings[name] for name in FEATURE_SOURCE_KEYS
                },
                frozen_rosters=bad_rosters,
                frozen_config=self.config,
            )

    def test_transport_repair_uses_only_the_corrected_feature_leaf(self) -> None:
        repaired_inputs = self.transport_repair_config["input_artifacts"]
        files = repaired_inputs["expansion_derived_features"]["files"]
        matches = [
            record
            for record in files
            if record["path"] == TRANSPORT_REPAIR_FILE_PATH
        ]
        self.assertEqual(
            matches,
            [
                {
                    "path": TRANSPORT_REPAIR_FILE_PATH,
                    "sha256": TRANSPORT_REPAIR_CORRECTED_SHA256,
                    "size_bytes": TRANSPORT_REPAIR_SIZE_BYTES,
                }
            ],
        )
        expected = self._repair_bindings(kind="feature")
        self.assertEqual(
            expected[EXPANSION_FEATURE_TRAJECTORIES],
            {
                "sha256": TRANSPORT_REPAIR_CORRECTED_SHA256,
                "size_bytes": TRANSPORT_REPAIR_SIZE_BYTES,
            },
        )
        feature = self._build_repair_feature()
        label = self._build_repair_label()
        joined = audit_formal_cache_join_transport_repair_v1(
            feature.archive,
            label.archive,
            frozen_rosters=self.rosters,
            frozen_config=self.transport_repair_config,
        )
        self.assertEqual(joined.joined_state_count, EXPECTED_STATE_COUNT)
        feature_files = _read_cache_members(feature.archive, kind="feature")
        feature_manifest = json.loads(feature_files["manifest.json"])
        self.assertEqual(
            feature_manifest["source_bindings"][EXPANSION_FEATURE_TRAJECTORIES],
            expected[EXPANSION_FEATURE_TRAJECTORIES],
        )
        with tarfile.open(fileobj=io.BytesIO(feature.archive), mode="r:") as archive:
            self.assertEqual(
                [member.name for member in archive.getmembers()],
                [
                    f"{FEATURE_CACHE_PREFIX}/feature_states.jsonl",
                    f"{FEATURE_CACHE_PREFIX}/manifest.json",
                ],
            )

    def test_transport_repair_rejects_any_marker_or_leaf_drift(self) -> None:
        expected = self._repair_bindings(kind="feature")
        bad_marker = copy.deepcopy(self.transport_repair_config)
        bad_marker["source_freeze"]["transport_repair"]["protocol_id"] = "other"
        with self.assertRaisesRegex(ValueError, "marker protocol_id"):
            build_formal_feature_cache_transport_repair_v1(
                {name: self.downloaded[name] for name in FEATURE_SOURCE_KEYS},
                transport_bindings=expected,
                frozen_rosters=self.rosters,
                frozen_config=bad_marker,
            )

        bad_leaf = copy.deepcopy(self.transport_repair_config)
        for record in bad_leaf["input_artifacts"]["expansion_derived_features"][
            "files"
        ]:
            if record["path"] == TRANSPORT_REPAIR_FILE_PATH:
                record["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "corrected feature binding"):
            build_formal_feature_cache_transport_repair_v1(
                {name: self.downloaded[name] for name in FEATURE_SOURCE_KEYS},
                transport_bindings=expected,
                frozen_rosters=self.rosters,
                frozen_config=bad_leaf,
            )

        extra_input_drift = copy.deepcopy(self.transport_repair_config)
        extra_input_drift["input_artifacts"]["legacy_derived_features"]["repo"] = (
            "different-repo"
        )
        with self.assertRaisesRegex(ValueError, "more than the allowed input leaf"):
            build_formal_feature_cache_transport_repair_v1(
                {name: self.downloaded[name] for name in FEATURE_SOURCE_KEYS},
                transport_bindings=expected,
                frozen_rosters=self.rosters,
                frozen_config=extra_input_drift,
            )

        with self.assertRaisesRegex(ValueError, "marker is missing"):
            build_formal_feature_cache_transport_repair_v1(
                {name: self.downloaded[name] for name in FEATURE_SOURCE_KEYS},
                transport_bindings=expected,
                frozen_rosters=self.rosters,
                frozen_config=self.config,
            )

    def test_normal_v1_builders_do_not_accept_repair_overlay(self) -> None:
        normal_feature = self._build_feature()
        self.assertEqual(len(read_feature_cache(normal_feature.archive)), EXPECTED_STATE_COUNT)
        repair_expected = self._repair_bindings(kind="feature")
        with mock.patch(
            "causalcache.gate_v1_formal_cache._verify_downloads",
            return_value=repair_expected,
        ):
            with self.assertRaisesRegex(
                ValueError, "formal cache source contract section source_freeze drifted"
            ):
                build_formal_feature_cache(
                    {name: self.downloaded[name] for name in FEATURE_SOURCE_KEYS},
                    transport_bindings=repair_expected,
                    frozen_rosters=self.rosters,
                    frozen_config=self.transport_repair_config,
                )

        with self.assertRaisesRegex(ValueError, "corrected source contract"):
            build_formal_feature_cache_transport_repair_v1(
                {name: self.downloaded[name] for name in FEATURE_SOURCE_KEYS},
                transport_bindings={
                    name: self.bindings[name] for name in FEATURE_SOURCE_KEYS
                },
                frozen_rosters=self.rosters,
                frozen_config=self.transport_repair_config,
            )

    def test_json_float_and_path_canonicality(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            _strict_canonical_object(b'{"a":1,"a":2}', label="duplicate")
        for value in (float("nan"), float("inf"), float("-inf"), -0.0):
            with self.assertRaises(ValueError):
                _float_to_hex(value, label="bad")
        self.assertEqual(_float_to_hex(0.0, label="zero"), "0000000000000000")
        with self.assertRaisesRegex(ValueError, "relative POSIX"):
            _ocr_path_without_json_decode(
                b'{"image_member_path":"../escape.png"}', label="path"
            )
        with self.assertRaisesRegex(ValueError, "unescaped canonical"):
            _ocr_path_without_json_decode(
                b'{"image_member_path":"images\\u002fescape.png"}', label="path"
            )

    def test_cache_readback_rejects_extra_missing_and_bad_float(self) -> None:
        feature = self._build_feature()
        label = self._build_label()
        files = _read_cache_members(feature.archive, kind="feature")
        lines = files["feature_states.jsonl"].splitlines()
        first = json.loads(lines[0])
        first["extra"] = True
        lines[0] = canonical_json_bytes(first)
        feature_payload = b"\n".join(lines) + b"\n"
        manifest = json.loads(files["manifest.json"])
        manifest["payload"] = {
            "path": "feature_states.jsonl",
            "record_count": EXPECTED_STATE_COUNT,
            "sha256": sha256_bytes(feature_payload),
            "size_bytes": len(feature_payload),
        }
        tampered = _cache_archive(manifest, "feature_states.jsonl", feature_payload)
        with self.assertRaisesRegex(ValueError, "extra"):
            read_feature_cache(tampered)

        label_files = _read_cache_members(label.archive, kind="label")
        label_lines = label_files["label_states.jsonl"].splitlines()
        label_first = json.loads(label_lines[0])
        del label_first["decision_step_id"]
        label_lines[0] = canonical_json_bytes(label_first)
        label_payload = b"\n".join(label_lines) + b"\n"
        label_manifest = json.loads(label_files["manifest.json"])
        label_manifest["payload"] = {
            "path": "label_states.jsonl",
            "record_count": EXPECTED_STATE_COUNT,
            "sha256": sha256_bytes(label_payload),
            "size_bytes": len(label_payload),
        }
        missing = _cache_archive(label_manifest, "label_states.jsonl", label_payload)
        with self.assertRaisesRegex(ValueError, "missing"):
            read_label_cache(missing)

        label_first = json.loads(label_files["label_states.jsonl"].splitlines()[0])
        label_first["distance_rows"][0]["distance_kl_f64_hex"] = "7ff0000000000000"
        label_lines = label_files["label_states.jsonl"].splitlines()
        label_lines[0] = canonical_json_bytes(label_first)
        label_payload = b"\n".join(label_lines) + b"\n"
        label_manifest = json.loads(label_files["manifest.json"])
        label_manifest["payload"]["sha256"] = sha256_bytes(label_payload)
        label_manifest["payload"]["size_bytes"] = len(label_payload)
        nonfinite = _cache_archive(label_manifest, "label_states.jsonl", label_payload)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            read_label_cache(nonfinite)

    def test_cache_readback_rejects_ustar_metadata(self) -> None:
        feature = self._build_feature()
        files = _read_cache_members(feature.archive, kind="feature")
        destination = io.BytesIO()
        with tarfile.open(fileobj=destination, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name in sorted(files):
                payload = files[name]
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = 0o600
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(payload))
        with self.assertRaisesRegex(ValueError, "metadata"):
            read_feature_cache(destination.getvalue())
        with self.assertRaisesRegex(ValueError, "deterministic USTAR"):
            read_feature_cache(feature.archive + b"trailing")


if __name__ == "__main__":
    unittest.main()
