from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from causalcache.restoration_v2_text_backend import (
    PreparedImage,
    build_ocr_record,
    canonical_json_bytes,
    canonicalize_rapidocr_nodes,
    decode_fixture_image,
    load_backend_config,
    mean_absolute_rgb_difference_from_prepared,
    prepare_image_bytes,
    sha256_bytes,
    validate_backend_config,
    validate_selected_guiodyssey_image,
)
from scripts.validate_restoration_v2_ocr_backend import config_only, validate_artifact_source


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "code/configs/restoration_v2_ocr_backend.json"
FIXTURE_PATH = ROOT / "data/fixtures/restoration_v2_ocr_golden.json"
ARTIFACT_MANIFEST_PATH = ROOT / "data/manifests/restoration_v2_ocr_backend.json"
PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None


class RestorationV2TextBackendTest(unittest.TestCase):
    def test_frozen_backend_config_and_fixture_source_validate(self) -> None:
        config = load_backend_config(CONFIG_PATH)
        self.assertEqual(
            config["backend_id"],
            "rapidocr-3.8.4-ppocrv5-mobile-en-cpu-v1",
        )
        result = config_only(
            backend_config_path=CONFIG_PATH,
            fixture_path=FIXTURE_PATH,
        )
        self.assertEqual(result["outcome"], "PASSED_OCR_CONFIG_SOURCE_VALIDATION")
        self.assertEqual(result["fixture_case_count"], 2)
        self.assertFalse(result["prepared_images_validated"])
        self.assertFalse(result["policy_output_generated"])

    def test_backend_identity_and_parameters_fail_closed(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        mutations = []
        changed = copy.deepcopy(config)
        changed["inference"]["intra_op_num_threads"] = 2
        mutations.append(changed)
        changed = copy.deepcopy(config)
        changed["models"]["recognizer"]["sha256"] = "0" * 64
        mutations.append(changed)
        changed = copy.deepcopy(config)
        changed["image_preprocessing"]["resample"] = "nearest"
        mutations.append(changed)
        changed = copy.deepcopy(config)
        changed["canonical_output"]["full_spatial_tokens_are_uncapped"] = False
        mutations.append(changed)
        changed = copy.deepcopy(config)
        changed["runtime_packages"]["requests"] = "0.0.0"
        mutations.append(changed)
        changed = copy.deepcopy(config)
        changed["golden_contract"]["confirm_images_may_be_used_for_golden_selection"] = True
        mutations.append(changed)
        changed = copy.deepcopy(config)
        changed["ocr_input"]["pre_resize"] = True
        mutations.append(changed)
        changed = copy.deepcopy(config)
        changed["unexpected_field"] = "must fail closed"
        mutations.append(changed)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    validate_backend_config(mutation)

    def test_frozen_expected_inspection_identity_is_complete(self) -> None:
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(fixture["status"], "synthetic_expected_inspection_frozen")
        expected = fixture["expected_inspection"]
        self.assertEqual(
            expected["backend_config_sha256"],
            "51e08a9565f804ecb1ee7f12887cc12742b84c04996fffbbde7c0f304e4bd036",
        )
        self.assertEqual(len(expected["rapidocr_package_file_sha256"]), 5)
        self.assertEqual(expected["recognizer_character_inventory"]["entry_count"], 436)
        self.assertEqual(
            expected["cases"][1]["ocr_record"]["full_spatial_tokens"],
            ["Causal", "Cache", "Step", "42"],
        )
        self.assertEqual(
            expected["cases"][1]["ocr_record"]["canonical_ocr_record_sha256"],
            "47941b52bf42df7c452119821e824953ab1a3ebd2aa8b1d31ef97b21d1093371",
        )

    def test_node_canonicalization_rounds_sorts_and_preserves_full_tokens(self) -> None:
        nodes = canonicalize_rapidocr_nodes(
            boxes=[
                [[9.6, 10.4], [19.7, 10.4], [19.7, 20.6], [9.6, 20.6]],
                [[1.2, 1.2], [7.7, 1.2], [7.7, 5.8], [1.2, 5.8]],
            ],
            texts=["  Beta\nTwo ", "Cafe\u0301 One"],
            scores=[0.8123456789, 0.999999999],
            width=20,
            height=30,
        )
        self.assertEqual(
            [node["normalized_text"] for node in nodes],
            ["Caf\u00e9 One", "Beta Two"],
        )
        self.assertEqual(nodes[0]["bbox_top_left_bottom_right"], [1, 1, 6, 8])
        self.assertEqual(nodes[1]["polygon_xy"][1], [19, 10])
        self.assertEqual(nodes[1]["confidence_decimal_string"], "0.81234568")

    @unittest.skipUnless(PIL_AVAILABLE, "Pillow is an optional OCR dependency")
    def test_record_hash_and_full_tokens_are_exact(self) -> None:
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        image_bytes = decode_fixture_image(fixture["cases"][0])
        record = build_ocr_record(
            image_member_path="fixtures/rgb-2x2.png",
            image_bytes=image_bytes,
            backend_config_sha256="1" * 64,
            boxes=[[[0, 0], [1, 0], [1, 1], [0, 1]]],
            texts=["Alpha Beta"],
            scores=[0.75],
        )
        self.assertEqual(record["full_spatial_tokens"], ["Alpha", "Beta"])
        digest = record.pop("canonical_ocr_record_sha256")
        self.assertEqual(digest, sha256_bytes(canonical_json_bytes(record)))

    @unittest.skipUnless(PIL_AVAILABLE, "Pillow is an optional OCR dependency")
    def test_prepared_image_and_mad_are_exact(self) -> None:
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        first = prepare_image_bytes(decode_fixture_image(fixture["cases"][0]))
        second = prepare_image_bytes(decode_fixture_image(fixture["cases"][1]))
        self.assertEqual(first.source_mode, "RGBA")
        self.assertEqual(first.alpha_extrema, (255, 255))
        self.assertEqual(
            first.resized_rgb_bytes_sha256,
            "abf1b6b546f81519a02053c5ba25b9a6d31fa9c0d44df6a1326780d468f80621",
        )
        value = mean_absolute_rgb_difference_from_prepared(first, second)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
        self.assertEqual(mean_absolute_rgb_difference_from_prepared(first, first), 0.0)

    def test_invalid_ocr_payloads_and_paths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            canonicalize_rapidocr_nodes(
                boxes=[[[0, 0], [1, 0], [1, 1], [0, 1]]],
                texts=[],
                scores=[],
                width=2,
                height=2,
            )
        for unsafe_path in ("", "/absolute.png", "../escape.png", "..\\escape.png", "./a.png"):
            with self.subTest(unsafe_path=unsafe_path):
                with self.assertRaisesRegex(ValueError, "safe relative"):
                    build_ocr_record(
                        image_member_path=unsafe_path,
                        image_bytes=b"not decoded because path fails first",
                        backend_config_sha256="1" * 64,
                        boxes=None,
                        texts=None,
                        scores=None,
                    )
        if PIL_AVAILABLE:
            fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "safe relative"):
                build_ocr_record(
                    image_member_path="../escape.png",
                    image_bytes=decode_fixture_image(fixture["cases"][0]),
                    backend_config_sha256="1" * 64,
                    boxes=None,
                    texts=None,
                    scores=None,
                )

    def test_selected_guiodyssey_image_contract_is_fail_closed(self) -> None:
        config = load_backend_config(CONFIG_PATH)
        prepared = PreparedImage(
            source_format="PNG",
            source_mode="RGBA",
            width=1080,
            height=2400,
            exif_present=False,
            alpha_extrema=(255, 255),
            rgb_bytes_sha256="1" * 64,
            resized_rgb_bytes=b"",
        )
        validate_selected_guiodyssey_image(prepared, config)
        invalid = PreparedImage(
            source_format="JPEG",
            source_mode="RGB",
            width=1080,
            height=2400,
            exif_present=False,
            alpha_extrema=None,
            rgb_bytes_sha256="1" * 64,
            resized_rgb_bytes=b"",
        )
        with self.assertRaisesRegex(
            ValueError,
            "INVALID_DERIVED_ARTIFACT_BEFORE_POLICY_OUTPUT",
        ):
            validate_selected_guiodyssey_image(invalid, config)

    def test_hf_artifact_and_source_manifest_validate_without_network(self) -> None:
        result = validate_artifact_source(
            backend_config_path=CONFIG_PATH,
            artifact_manifest_path=ARTIFACT_MANIFEST_PATH,
            repository_root=ROOT,
        )
        self.assertEqual(result["outcome"], "PASSED_OCR_ARTIFACT_SOURCE_VALIDATION")
        self.assertEqual(result["hf_file_count"], 6)
        self.assertEqual(result["hf_dataset_file_count"], 5)
        self.assertTrue(result["model_hashes_bound_to_backend_config"])
        self.assertTrue(result["real_screen_golden_passed"])
        self.assertTrue(result["dependency_5_closed"])

    def test_hf_artifact_model_hash_drift_fails_closed(self) -> None:
        manifest = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
        detector = next(
            record
            for record in manifest["hf_model_artifact"]["files"]
            if record["path"] == "ch_PP-OCRv5_det_mobile.onnx"
        )
        detector["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated-manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "detector SHA256 drifted"):
                validate_artifact_source(
                    backend_config_path=CONFIG_PATH,
                    artifact_manifest_path=path,
                    repository_root=ROOT,
                )

    def test_hf_artifact_source_omission_fails_closed(self) -> None:
        manifest = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["source_files"] = [
            record
            for record in manifest["source_files"]
            if record["path"] != "code/causalcache/restoration_v2_text_backend.py"
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated-manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source file inventory drifted"):
                validate_artifact_source(
                    backend_config_path=CONFIG_PATH,
                    artifact_manifest_path=path,
                    repository_root=ROOT,
                )

    def test_hf_artifact_preoutput_declaration_fails_closed(self) -> None:
        manifest = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["policy_output_generated_before_manifest"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated-manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must precede policy output"):
                validate_artifact_source(
                    backend_config_path=CONFIG_PATH,
                    artifact_manifest_path=path,
                    repository_root=ROOT,
                )

    def test_hf_real_screen_revision_and_file_identity_fail_closed(self) -> None:
        manifest = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
        mutations = []
        changed = copy.deepcopy(manifest)
        changed["real_screen_golden"]["immutable_revision"] = "0" * 40
        mutations.append((changed, "immutable revision drifted"))
        changed = copy.deepcopy(manifest)
        changed["real_screen_golden"]["files"][0]["sha256"] = "0" * 64
        mutations.append((changed, "file identity drifted"))
        changed = copy.deepcopy(manifest)
        changed["real_screen_golden"]["redownload"][
            "artifact_validator_replayed_on_hyper00"
        ] = False
        mutations.append((changed, "re-download is not verified"))
        for index, (changed, message) in enumerate(mutations):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"mutated-manifest-{index}.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    validate_artifact_source(
                        backend_config_path=CONFIG_PATH,
                        artifact_manifest_path=path,
                        repository_root=ROOT,
                    )

    def test_hf_real_screen_repeat_and_replay_evidence_fail_closed(self) -> None:
        mutations = [
            (
                lambda summary: summary["materialization_repeats"].pop(),
                "materialization evidence drifted",
            ),
            (
                lambda summary: summary["artifact_validations"].pop(),
                "validation evidence drifted",
            ),
            (
                lambda summary: summary["artifact_validations"][0].update(
                    {"ocr_replay_record_count": 5}
                ),
                "replay evidence drifted",
            ),
        ]
        for mutate, message in mutations:
            manifest = json.loads(
                ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8")
            )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for record in manifest["source_files"]:
                    source = ROOT / record["path"]
                    destination = root / record["path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                summary_relative = manifest["real_screen_golden"]["summary_path"]
                summary_path = root / summary_relative
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                mutate(summary)
                summary_path.write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                summary_sha256 = hashlib.sha256(
                    summary_path.read_bytes()
                ).hexdigest()
                source_record = next(
                    record
                    for record in manifest["source_files"]
                    if record["path"] == summary_relative
                )
                source_record["sha256"] = summary_sha256
                manifest["real_screen_golden"][
                    "summary_sha256"
                ] = summary_sha256
                manifest_path = (
                    root / "data/manifests/restoration_v2_ocr_backend.json"
                )
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    validate_artifact_source(
                        backend_config_path=(
                            root / "code/configs/restoration_v2_ocr_backend.json"
                        ),
                        artifact_manifest_path=manifest_path,
                        repository_root=root,
                    )


if __name__ == "__main__":
    unittest.main()
