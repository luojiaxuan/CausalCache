from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from causalcache.data.restoration_v2_real_screen import (
    ARTIFACT_RELATIVE_PATHS,
    EXPECTED_CONFIRM_UNIQUE_IMAGE_COUNT,
    EXPECTED_OCCURRENCE_COUNT,
    EXPECTED_ORIENTATION_COUNTS,
    EXPECTED_SELECTED_IMAGES,
    EXPECTED_UNIQUE_IMAGE_COUNT,
    artifact_gitattributes_bytes,
    artifact_tree_identity,
    build_ocr_jsonl,
    build_selected_image_tar_bytes,
    collect_real_screen_occurrences,
    select_real_screen_golden,
    validate_candidate_pool,
)
from scripts.validate_restoration_v2_real_screen import (
    _validate_image_tar,
    validate_source_contract,
)


ROOT = Path(__file__).resolve().parents[2]
SELECTION_PATH = ROOT / "data/manifests/restoration_v2_selection.json"
BACKEND_CONFIG_PATH = ROOT / "code/configs/restoration_v2_ocr_backend.json"
SOURCE_CONTRACT_PATH = (
    ROOT / "data/manifests/restoration_v2_real_screen_source.json"
)


def _occurrence(index: int, *, image_sha256: str, image_member_path: str) -> dict:
    return {
        "role": "v2_label_train",
        "state_id": f"fixture:decision_step:{index:03d}",
        "source_id": "fixture",
        "decision_step_id": 4,
        "image_role": "candidate_post_state",
        "event_step_id": 1,
        "image_member_path": image_member_path,
        "image_sha256": image_sha256,
    }


def _candidate_pool() -> list[dict]:
    records = []
    for orientation, selected in EXPECTED_SELECTED_IMAGES.items():
        for record in selected:
            records.append(
                {
                    **record,
                    "image_size_bytes": 1,
                    "source_format": "PNG",
                    "source_mode": "RGBA",
                    "exif_present": False,
                    "alpha_extrema": [255, 255],
                    "rgb_bytes_sha256": "a" * 64,
                    "resized_rgb_256x256_sha256": "b" * 64,
                    "orientation": orientation,
                    "eligible_occurrences": [],
                }
            )
    filler_counts = {
        "portrait": EXPECTED_ORIENTATION_COUNTS["portrait"] - 3,
        "landscape": EXPECTED_ORIENTATION_COUNTS["landscape"] - 3,
    }
    filler_index = 0
    for orientation, count in filler_counts.items():
        for _ in range(count):
            digest = f"f{filler_index:063x}"
            width, height = ((720, 1280) if orientation == "portrait" else (1280, 720))
            records.append(
                {
                    "image_sha256": digest,
                    "image_member_path": f"images/fixture/{digest}.png",
                    "image_size_bytes": 1,
                    "source_format": "PNG",
                    "source_mode": "RGBA",
                    "width": width,
                    "height": height,
                    "exif_present": False,
                    "alpha_extrema": [255, 255],
                    "rgb_bytes_sha256": "a" * 64,
                    "resized_rgb_256x256_sha256": "b" * 64,
                    "orientation": orientation,
                    "eligible_occurrences": [],
                }
            )
            filler_index += 1
    records.sort(key=lambda record: (record["image_sha256"], record["image_member_path"]))
    for index, record in enumerate(records):
        record["candidate_pool_index"] = index
        record["eligible_occurrences"].append(
            _occurrence(
                index,
                image_sha256=record["image_sha256"],
                image_member_path=record["image_member_path"],
            )
        )
    extra = EXPECTED_OCCURRENCE_COUNT - len(records)
    records[0]["eligible_occurrences"].extend(
        _occurrence(
            len(records) + index,
            image_sha256=records[0]["image_sha256"],
            image_member_path=records[0]["image_member_path"],
        )
        for index in range(extra)
    )
    return records


class RestorationV2RealScreenTest(unittest.TestCase):
    def test_preoutput_source_contract_is_complete_and_fail_closed(self) -> None:
        result = validate_source_contract(
            SOURCE_CONTRACT_PATH,
            repository_root=ROOT,
        )
        self.assertEqual(result["outcome"], "PASSED_REAL_SCREEN_SOURCE_VALIDATION")
        self.assertEqual(result["source_file_count"], 17)
        self.assertFalse(result["confirm_images_used"])

        manifest = json.loads(SOURCE_CONTRACT_PATH.read_text(encoding="utf-8"))
        manifest["confirm_images_may_be_materialized"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated-source.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exclude confirm images"):
                validate_source_contract(path, repository_root=ROOT)

    def test_committed_selection_has_frozen_policy_blind_pool(self) -> None:
        selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        occurrences, confirm_shas = collect_real_screen_occurrences(selection)
        self.assertEqual(len(occurrences), EXPECTED_OCCURRENCE_COUNT)
        self.assertEqual(
            len({record["image_sha256"] for record in occurrences}),
            EXPECTED_UNIQUE_IMAGE_COUNT,
        )
        self.assertEqual(len(confirm_shas), EXPECTED_CONFIRM_UNIQUE_IMAGE_COUNT)
        self.assertFalse(
            {record["image_sha256"] for record in occurrences}.intersection(
                confirm_shas
            )
        )
        self.assertEqual(
            sum(record["image_role"] == "candidate_post_state" for record in occurrences),
            135,
        )
        self.assertEqual(
            sum(record["image_role"] == "current_observation" for record in occurrences),
            45,
        )

    def test_frozen_orientation_selection_is_exact(self) -> None:
        pool = _candidate_pool()
        backend_config = json.loads(BACKEND_CONFIG_PATH.read_text(encoding="utf-8"))
        selected = select_real_screen_golden(pool, backend_config=backend_config)
        self.assertEqual(len(selected), 6)
        observed = {
            orientation: tuple(
                record["image_sha256"]
                for record in selected
                if record["orientation"] == orientation
            )
            for orientation in ("portrait", "landscape")
        }
        expected = {
            orientation: tuple(record["image_sha256"] for record in records)
            for orientation, records in EXPECTED_SELECTED_IMAGES.items()
        }
        self.assertEqual(observed, expected)
        self.assertEqual(
            [record["golden_case_id"] for record in selected],
            [
                "portrait-000",
                "portrait-001",
                "portrait-002",
                "landscape-000",
                "landscape-001",
                "landscape-002",
            ],
        )

    def test_candidate_pool_rejects_order_orientation_and_accounting_drift(self) -> None:
        pool = _candidate_pool()
        mutations = []
        changed = copy.deepcopy(pool)
        changed[0], changed[1] = changed[1], changed[0]
        mutations.append(changed)
        changed = copy.deepcopy(pool)
        changed[0]["width"] = changed[0]["height"]
        mutations.append(changed)
        changed = copy.deepcopy(pool)
        changed[0]["eligible_occurrences"].pop()
        mutations.append(changed)
        changed = copy.deepcopy(pool)
        changed[0]["eligible_occurrences"][0]["image_member_path"] = (
            "images/0000000000000000/observation-000.png"
        )
        mutations.append(changed)
        changed = copy.deepcopy(pool)
        changed[0]["eligible_occurrences"][0]["image_sha256"] = "0" * 64
        mutations.append(changed)
        for mutation in mutations:
            with self.assertRaises(ValueError):
                validate_candidate_pool(mutation)

    def test_ocr_jsonl_is_canonical_and_preserves_source_member_path(self) -> None:
        pool = _candidate_pool()
        backend_config = json.loads(BACKEND_CONFIG_PATH.read_text(encoding="utf-8"))
        selected = select_real_screen_golden(pool, backend_config=backend_config)
        ocr_records = {
            record["image_sha256"]: {
                "image_sha256": record["image_sha256"],
                "image_member_path": record["image_member_path"],
                "canonical_ocr_record_sha256": "c" * 64,
            }
            for record in selected
        }
        payload = build_ocr_jsonl(
            selected,
            ocr_records_by_sha256=ocr_records,
        )
        self.assertTrue(payload.endswith(b"\n"))
        lines = payload.splitlines()
        self.assertEqual(len(lines), 6)
        parsed = [json.loads(line) for line in lines]
        self.assertEqual(
            [record["ocr_record"]["image_member_path"] for record in parsed],
            [record["image_member_path"] for record in selected],
        )
        self.assertEqual(
            payload,
            b"\n".join(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                for record in parsed
            )
            + b"\n",
        )

    def test_image_tar_requires_exact_canonical_ustar_bytes(self) -> None:
        payloads = {
            "images/fixture/a.png": b"first-image",
            "images/fixture/b.png": b"second-image",
        }
        selected = []
        for index, (member_path, payload) in enumerate(payloads.items()):
            digest = hashlib.sha256(payload).hexdigest()
            selected.append(
                {
                    "image_sha256": digest,
                    "image_member_path": member_path,
                    "image_tar_member": f"images/{digest}.png",
                }
            )
        selected.sort(key=lambda record: record["image_tar_member"])
        tar_bytes, _ = build_selected_image_tar_bytes(
            selected,
            image_payloads=payloads,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.tar"
            path.write_bytes(tar_bytes)
            extracted = _validate_image_tar(
                path,
                selected,
                raw_payloads=payloads,
            )
            self.assertEqual(len(extracted), 2)
            path.write_bytes(tar_bytes + b"unexpected-trailer")
            with self.assertRaisesRegex(ValueError, "canonical USTAR"):
                _validate_image_tar(
                    path,
                    selected,
                    raw_payloads=payloads,
                )

    def test_artifact_tree_includes_pinned_hf_control_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ARTIFACT_RELATIVE_PATHS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = (
                    artifact_gitattributes_bytes()
                    if relative == ".gitattributes"
                    else relative.encode("utf-8")
                )
                path.write_bytes(payload)
            first = artifact_tree_identity(root)
            second = artifact_tree_identity(root)
            self.assertEqual(first, second)
            self.assertEqual(len(first["files"]), 5)
            self.assertEqual(first["files"][0]["path"], ".gitattributes")
            (root / "unexpected.txt").write_bytes(b"not allowed")
            with self.assertRaisesRegex(ValueError, "inventory"):
                artifact_tree_identity(root)


if __name__ == "__main__":
    unittest.main()
