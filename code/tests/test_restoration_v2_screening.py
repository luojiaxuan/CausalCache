from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from causalcache.data.guiodyssey_restoration_v2 import (
    EXPECTED_FORMAL_COUNTS,
    IMAGE_TAR_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    TRAJECTORY_JSONL_RELATIVE_PATH,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)
from causalcache.data.restoration_v2_screening import (
    ScreeningState,
    load_validated_screening_artifact,
)
from causalcache.low_fidelity_v2 import LowFidelityEventV2, serialize_low_fidelity_v2


TREE_SHA256 = "a" * 64
LABEL_IDS = tuple(f"label-{index:02d}" for index in range(10))
DEVELOPMENT_IDS = tuple(f"development-{index:02d}" for index in range(5))
CONFIRM_IDS = tuple(f"confirm-{index:02d}" for index in range(20))


def _image_path(source_id: str, index: int) -> str:
    return f"images/{source_id}/observation-{index:03d}.png"


def _image_payload(source_id: str, index: int) -> bytes:
    return f"PNG:{source_id}:{index}".encode("utf-8")


def _trajectory(source_id: str, role: str) -> dict[str, object]:
    events = []
    for step_id in range(1, 6):
        low = LowFidelityEventV2(
            step_id=step_id,
            action_type="wait",
            action_argument="wait",
            foreground_app="fixture",
            screen_text_added=(),
            screen_text_removed=(),
            screen_change="low",
            executor_result="accepted",
        )
        serialized = serialize_low_fidelity_v2(low)
        before = _image_payload(source_id, step_id - 1)
        after = _image_payload(source_id, step_id)
        events.append(
            {
                "step_id": step_id,
                "observation_before_path": _image_path(source_id, step_id - 1),
                "observation_before_sha256": sha256_bytes(before),
                "observation_after_path": _image_path(source_id, step_id),
                "observation_after_sha256": sha256_bytes(after),
                "low_fidelity_v2": low.to_ordered_dict(),
                "low_fidelity_v2_serialized": serialized.decode("utf-8"),
                "low_fidelity_v2_sha256": sha256_bytes(serialized),
            }
        )
    decisions = []
    for decision_step in (4, 5, 6):
        history = list(range(1, decision_step))
        current = _image_payload(source_id, decision_step - 1)
        decisions.append(
            {
                "state_id": f"{source_id}:decision_step:{decision_step:03d}",
                "decision_step_id": decision_step,
                "history_event_step_ids": history,
                "candidate_event_step_ids": history[:-1],
                "current_equivalent_event_step_id": history[-1],
                "current_observation_path": _image_path(source_id, decision_step - 1),
                "current_observation_sha256": sha256_bytes(current),
            }
        )
    return {
        "source_id": source_id,
        "role": role,
        "instruction": f"Operate fixture {source_id}",
        "events": events,
        "decisions": decisions if role != "v2_confirm_primary" else [decisions[-1]],
    }


def _all_trajectories() -> list[dict[str, object]]:
    return [
        *(_trajectory(source_id, "v2_label_train") for source_id in LABEL_IDS),
        *(
            _trajectory(source_id, "v2_development")
            for source_id in DEVELOPMENT_IDS
        ),
        *(
            _trajectory(source_id, "v2_confirm_primary")
            for source_id in CONFIRM_IDS
        ),
    ]


def _scientific_config() -> dict[str, object]:
    return {
        "data": {
            "roles": {
                "v2_label_train": {
                    "source_ids": list(LABEL_IDS),
                    "state_decision_step_ids": [4, 5, 6],
                },
                "v2_development": {
                    "source_ids": list(DEVELOPMENT_IDS),
                    "state_decision_step_ids": [4, 5, 6],
                },
            }
        }
    }


def _selection_manifest() -> dict[str, object]:
    trajectories = _all_trajectories()
    return {
        "roles": {
            role: {
                "trajectories": [
                    {"source_id": trajectory["source_id"]}
                    for trajectory in trajectories
                    if trajectory["role"] == role
                ],
                "states": [
                    {
                        "state_id": decision["state_id"],
                        "source_id": trajectory["source_id"],
                        "decision_step_id": decision["decision_step_id"],
                        "history_event_step_ids": decision["history_event_step_ids"],
                        "candidate_event_step_ids": decision[
                            "candidate_event_step_ids"
                        ],
                        "current_equivalent_event_step_id": decision[
                            "current_equivalent_event_step_id"
                        ],
                    }
                    for trajectory in trajectories
                    if trajectory["role"] == role
                    for decision in trajectory["decisions"]
                ],
            }
            for role in (
                "v2_label_train",
                "v2_development",
                "v2_confirm_primary",
            )
        }
    }


def _tar_payloads() -> dict[str, bytes]:
    return {
        _image_path(source_id, index): _image_payload(source_id, index)
        for source_id in (*LABEL_IDS, *DEVELOPMENT_IDS, *CONFIRM_IDS)
        for index in range(6)
    }


def _write_tar(path: Path, payloads: dict[str, bytes]) -> list[dict[str, object]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with tarfile.open(path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for member_path in sorted(payloads):
            payload = payloads[member_path]
            info = tarfile.TarInfo(member_path)
            info.size = len(payload)
            info.type = tarfile.REGTYPE
            archive.addfile(info, io.BytesIO(payload))
            records.append(
                {
                    "path": member_path,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return records


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    trajectories = _all_trajectories()
    trajectory_path = root / TRAJECTORY_JSONL_RELATIVE_PATH
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.write_bytes(canonical_jsonl_bytes(trajectories))
    image_records = _write_tar(root / IMAGE_TAR_RELATIVE_PATH, _tar_payloads())
    manifest = {
        "formal_counts_enforced": True,
        "counts": dict(EXPECTED_FORMAL_COUNTS),
        "role_source_ids": {
            "v2_label_train": list(LABEL_IDS),
            "v2_development": list(DEVELOPMENT_IDS),
            "v2_confirm_primary": list(CONFIRM_IDS),
        },
        "inventories": {
            "image_members_sha256": sha256_bytes(canonical_json_bytes(image_records))
        },
    }
    manifest_path = root / MANIFEST_RELATIVE_PATH
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    backend_path = root / "backend.json"
    backend_path.write_text("{}", encoding="utf-8")
    scientific_path = root / "scientific.json"
    scientific_path.write_text(json.dumps(_scientific_config()), encoding="utf-8")
    selection_path = root / "selection.json"
    selection_path.write_text(json.dumps(_selection_manifest()), encoding="utf-8")
    return backend_path, scientific_path, selection_path


def _validation(*, tree_sha256: str = TREE_SHA256) -> dict[str, object]:
    return {
        "outcome": "PASSED_GUIODYSSEY_RESTORATION_V2_ARTIFACT_VALIDATION",
        "artifact_tree_sha256": tree_sha256,
        "counts": dict(EXPECTED_FORMAL_COUNTS),
        "formal_counts_enforced": True,
        "ocr_replay_performed": False,
        "policy_loaded": False,
    }


class RestorationV2ScreeningLoaderTest(unittest.TestCase):
    def _load(self, root: Path, **validation_kwargs: object):
        backend_path = root / "backend.json"
        scientific_path = root / "scientific.json"
        selection_path = root / "selection.json"
        with (
            mock.patch(
                "causalcache.data.restoration_v2_screening.load_backend_config",
                return_value={},
            ),
            mock.patch(
                "causalcache.data.restoration_v2_screening.validate_artifact",
                return_value=_validation(**validation_kwargs),
            ) as validator,
            mock.patch(
                "causalcache.data.restoration_v2_screening.validate_selection_manifest"
            ) as selection_validator,
            mock.patch(
                "causalcache.data.restoration_v2_screening.validate_state_content_witnesses"
            ) as witness_validator,
        ):
            artifact = load_validated_screening_artifact(
                artifact_root=root,
                backend_config_path=backend_path,
                scientific_config_path=scientific_path,
                selection_manifest_path=selection_path,
                expected_artifact_tree_sha256=TREE_SHA256,
            )
        selection_validator.assert_called_once()
        witness_validator.assert_called_once()
        return artifact, validator

    def test_returns_exact_read_only_screening_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            artifact, validator = self._load(root)

        self.assertEqual(len(artifact.states), 45)
        self.assertEqual(
            [(state.role, state.decision_step_id) for state in artifact.states[:3]],
            [("v2_label_train", 4), ("v2_label_train", 5), ("v2_label_train", 6)],
        )
        self.assertTrue(
            all(state.role != "v2_confirm_primary" for state in artifact.states)
        )
        validator.assert_called_once()
        self.assertFalse(validator.call_args.kwargs["require_formal"])
        self.assertFalse(validator.call_args.kwargs["require_ocr_replay"])
        with self.assertRaisesRegex(ValueError, "screening-role inventory"):
            artifact.image_bytes(_image_path(CONFIRM_IDS[0], 0))

    def test_builds_reference_and_summary_only_native_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            artifact, _ = self._load(root)
            state = artifact.states[2]
            reference = artifact.build_messages(
                state,
                restored_event_step_ids=state.candidate_event_step_ids,
                image_decoder=lambda payload: payload,
            )
            summary = artifact.build_messages(
                state,
                restored_event_step_ids=(),
                image_decoder=lambda payload: payload,
            )

        reference_images = sum(
            block["type"] == "image" for block in reference[1]["content"]
        )
        summary_images = sum(
            block["type"] == "image" for block in summary[1]["content"]
        )
        self.assertEqual(reference_images, 5)
        self.assertEqual(summary_images, 1)

    def test_rejects_confirm_state_even_if_manually_constructed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            artifact, _ = self._load(root)
            confirm_state = ScreeningState(
                index=45,
                role="v2_confirm_primary",
                trajectory_id=CONFIRM_IDS[0],
                decision_step_id=6,
                candidate_event_step_ids=(1, 2, 3, 4),
            )
            with self.assertRaisesRegex(ValueError, "frozen screening denominator"):
                artifact.build_messages(
                    confirm_state,
                    restored_event_step_ids=(),
                    image_decoder=lambda payload: payload,
                )

    def test_rejects_artifact_tree_not_bound_to_execution_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            with self.assertRaisesRegex(ValueError, "immutable execution binding"):
                self._load(root, tree_sha256="b" * 64)

    def test_rejects_tampered_image_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            payloads = _tar_payloads()
            payloads[_image_path(LABEL_IDS[0], 0)] = b"tampered"
            _write_tar(root / IMAGE_TAR_RELATIVE_PATH, payloads)
            with self.assertRaisesRegex(ValueError, "bytes or member inventory"):
                self._load(root)

    def test_rejects_screening_role_drift_without_top_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, scientific_path, _ = _write_fixture(root)
            config = _scientific_config()
            config["data"]["roles"]["v2_label_train"]["source_ids"][0] = "replacement"  # type: ignore[index]
            scientific_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from the frozen config"):
                self._load(root)

    def test_rejects_derived_state_id_drift_from_frozen_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            path = root / TRAJECTORY_JSONL_RELATIVE_PATH
            trajectories = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            trajectories[0]["decisions"][0]["state_id"] = "tampered-state"
            path.write_bytes(canonical_jsonl_bytes(trajectories))
            with self.assertRaisesRegex(ValueError, "decision state_id drifted"):
                self._load(root)

    def test_rejects_nonregular_tar_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            path = root / IMAGE_TAR_RELATIVE_PATH
            with tarfile.open(path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                info = tarfile.TarInfo("images/link.png")
                info.type = tarfile.SYMTYPE
                info.linkname = "../../escape"
                archive.addfile(info)
            with self.assertRaisesRegex(ValueError, "regular files"):
                self._load(root)


if __name__ == "__main__":
    unittest.main()
