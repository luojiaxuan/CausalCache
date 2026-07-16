from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from causalcache.spatial_reference_audit_v1 import (
    load_and_validate_config,
    validate_repository_inputs,
)
from causalcache.spatial_reference_audit_v1_artifact import (
    _collect_tree_files,
    _tree_inventory_sha256,
    collect_spatial_reference_audit_files,
)
from scripts.validate_spatial_reference_audit_v1_validation_repair_v1 import (
    ORIGINAL_FAILURE,
    _parent_shapes,
    _scoped_dynamic_validator_override,
    _validate_original_failure,
    _validate_pre_repair_evidence,
    load_repair_config,
    validate_dynamic_shape_metadata,
    validate_frozen_raw_archive_absent,
    validate_original_audit_committed_source_blobs,
    validate_parent_bound_shape_nodes,
    validate_post_write_state,
    write_canonical_summary,
)
from scripts import validate_spatial_reference_audit_v1 as original_validator
from causalcache.data.guiodyssey_restoration_v2 import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]
REPAIR_CONFIG_PATH = (
    ROOT / "code/configs/spatial_reference_audit_v1_validation_repair_v1.json"
)
ORIGINAL_CONFIG_PATH = ROOT / "code/configs/spatial_reference_audit_v1.json"


def _shape(
    *,
    grid: list[int] | None = None,
    text_tokens: int = 100,
    aligned: bool = False,
) -> dict[str, object]:
    realized_grid = grid or [1, 100, 108]
    effective = realized_grid[0] * realized_grid[1] * realized_grid[2] // 4
    result: dict[str, object] = {
        "image_count": 1,
        "image_grid_thw": [realized_grid],
        "effective_visual_tokens": effective,
        "policy_visible_text_tokens": text_tokens,
        "prompt_input_tokens": effective + text_tokens,
    }
    if aligned:
        result["extended_prompt_aligned_inputs"] = [
            "attention_mask",
            "mm_token_type_ids",
        ]
    return result


def _file_binding(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


class ValidationRepairConfigTest(unittest.TestCase):
    def test_frozen_config_has_one_output_and_explicit_failure(self) -> None:
        config = load_repair_config(REPAIR_CONFIG_PATH)
        self.assertEqual(
            config["outputs"],
            {
                "summary": (
                    "/data/experiments/causalcache/spatial-reference-audit-v1/"
                    "summary.json"
                ),
                "new_outputs_only": True,
                "write_only_after_all_validation_succeeds": True,
            },
        )
        self.assertEqual(
            config["pre_repair_evidence"]["original_failure"], ORIGINAL_FAILURE
        )
        contract = config["repair_contract"]
        self.assertFalse(
            contract[
                "observed_effective_visual_token_set_used_as_acceptance_whitelist"
            ]
        )
        self.assertEqual(
            contract["teacher_extended_prompt_aligned_inputs"],
            ["attention_mask", "mm_token_type_ids"],
        )

    def test_config_rejects_old_multi_output_and_contract_drift(self) -> None:
        original = json.loads(REPAIR_CONFIG_PATH.read_bytes())
        mutations = []
        old_outputs = copy.deepcopy(original)
        old_outputs["outputs"]["repair_directory"] = "/data/repair"
        mutations.append(old_outputs)
        whitelist = copy.deepcopy(original)
        whitelist["repair_contract"][
            "observed_effective_visual_token_set_used_as_acceptance_whitelist"
        ] = True
        mutations.append(whitelist)
        failure = copy.deepcopy(original)
        failure["pre_repair_evidence"]["original_failure"]["message"] = "other"
        mutations.append(failure)
        aligned = copy.deepcopy(original)
        aligned["repair_contract"]["teacher_extended_prompt_aligned_inputs"] = [
            "attention_mask",
            "input_ids",
        ]
        mutations.append(aligned)
        with tempfile.TemporaryDirectory() as directory:
            for index, mutation in enumerate(mutations):
                path = Path(directory) / f"mutation-{index}.json"
                path.write_text(json.dumps(mutation), encoding="utf-8")
                with self.subTest(index=index), self.assertRaises(ValueError):
                    load_repair_config(path)

    def test_original_33_file_inventory_remains_unchanged(self) -> None:
        original = load_and_validate_config(ORIGINAL_CONFIG_PATH)
        validate_repository_inputs(ROOT, original)
        self.assertEqual(len(original["source_inventory"]), 33)
        self.assertNotIn(
            "code/scripts/validate_spatial_reference_audit_v1_validation_repair_v1.py",
            original["source_inventory"],
        )
        repair = load_repair_config(REPAIR_CONFIG_PATH)
        self.assertEqual(
            repair["frozen_source"]["original_validator"]["sha256"],
            original["source_inventory"][
                "code/scripts/validate_spatial_reference_audit_v1.py"
            ],
        )
        records = validate_original_audit_committed_source_blobs(
            repository_root=ROOT,
            original_config_path=ORIGINAL_CONFIG_PATH,
            original_config=original,
            original_audit_git_commit=(
                "c093bd8f92ab97427acb427bd2d66fb6b20b556a"
            ),
        )
        self.assertEqual(len(records), 34)
        self.assertEqual(records[0]["path"], "code/configs/spatial_reference_audit_v1.json")


class DynamicShapeRepairTest(unittest.TestCase):
    def test_formula_accepts_valid_shape_outside_observed_set(self) -> None:
        shape = _shape()
        self.assertEqual(shape["effective_visual_tokens"], 2700)
        validated = validate_dynamic_shape_metadata(
            shape,
            name="novel valid grid",
            require_aligned_inputs=False,
        )
        self.assertEqual(validated["effective_visual_tokens"], 2700)

    def test_formula_accepts_realized_grids_and_multi_image_sum(self) -> None:
        grids = [
            [1, 148, 68],
            [1, 80, 128],
            [1, 136, 76],
            [1, 152, 68],
        ]
        per_image = [
            temporal * height * width // 4
            for temporal, height, width in grids
        ]
        self.assertEqual(per_image, [2516, 2560, 2584, 2584])
        value = {
            "image_count": len(grids),
            "image_grid_thw": grids,
            "effective_visual_tokens": sum(per_image),
            "policy_visible_text_tokens": 101,
            "prompt_input_tokens": sum(per_image) + 101,
        }
        validated = validate_dynamic_shape_metadata(
            value,
            name="realized multi-image grids",
            require_aligned_inputs=False,
        )
        self.assertEqual(validated["effective_visual_tokens"], 10244)

    def test_formula_rejects_invalid_merge_or_accounting(self) -> None:
        odd = _shape(grid=[1, 99, 108])
        with self.assertRaisesRegex(ValueError, "merge divisibility"):
            validate_dynamic_shape_metadata(
                odd,
                name="odd grid",
                require_aligned_inputs=False,
            )
        drifted = _shape()
        drifted["effective_visual_tokens"] = 2560
        with self.assertRaisesRegex(ValueError, "token accounting"):
            validate_dynamic_shape_metadata(
                drifted,
                name="drifted accounting",
                require_aligned_inputs=False,
            )

    def test_teacher_alignment_uses_runtime_emitted_inputs(self) -> None:
        validate_dynamic_shape_metadata(
            _shape(aligned=True),
            name="teacher",
            require_aligned_inputs=True,
        )
        old_contract = _shape(aligned=True)
        old_contract["extended_prompt_aligned_inputs"] = [
            "attention_mask",
            "input_ids",
        ]
        with self.assertRaisesRegex(ValueError, "prompt-aligned"):
            validate_dynamic_shape_metadata(
                old_contract,
                name="teacher",
                require_aligned_inputs=True,
            )

    def test_parent_witness_binds_generation_and_teacher_grid(self) -> None:
        parent = _shape()
        profile = {
            "records": [
                {
                    "index": 7,
                    "generations": [{"metadata": copy.deepcopy(parent)}],
                    "shared_prefix_parent_pair_repeats": [
                        _shape(text_tokens=110, aligned=True)
                    ],
                    "full_parent_action_diagnostics": [
                        _shape(text_tokens=110, aligned=True)
                    ],
                }
            ]
        }
        result = validate_parent_bound_shape_nodes(
            [profile],
            parent_shapes={7: parent},
            expected_aligned_inputs=["attention_mask", "mm_token_type_ids"],
            expected_generation_node_count=1,
            expected_teacher_node_count=2,
        )
        self.assertEqual(result["observed_effective_visual_tokens_per_image"], [2700])

        generation_drift = copy.deepcopy(profile)
        generation_drift["records"][0]["generations"][0]["metadata"] = _shape(
            text_tokens=101
        )
        with self.assertRaisesRegex(ValueError, "generation full shape"):
            validate_parent_bound_shape_nodes(
                [generation_drift],
                parent_shapes={7: parent},
                expected_aligned_inputs=["attention_mask", "mm_token_type_ids"],
                expected_generation_node_count=1,
                expected_teacher_node_count=2,
            )

        teacher_grid_drift = copy.deepcopy(profile)
        teacher_grid_drift["records"][0]["full_parent_action_diagnostics"][0] = (
            _shape(grid=[1, 90, 120], text_tokens=110, aligned=True)
        )
        with self.assertRaisesRegex(ValueError, "teacher grid/effective"):
            validate_parent_bound_shape_nodes(
                [teacher_grid_drift],
                parent_shapes={7: parent},
                expected_aligned_inputs=["attention_mask", "mm_token_type_ids"],
                expected_generation_node_count=1,
                expected_teacher_node_count=2,
            )

        with self.assertRaisesRegex(ValueError, "shape-node denominator"):
            validate_parent_bound_shape_nodes(
                [profile],
                parent_shapes={7: parent},
                expected_aligned_inputs=["attention_mask", "mm_token_type_ids"],
                expected_generation_node_count=2,
                expected_teacher_node_count=2,
            )

    def test_scoped_override_restores_original_on_success_and_error(self) -> None:
        sentinel = original_validator._validate_shape_metadata
        with _scoped_dynamic_validator_override(
            aligned=["attention_mask", "mm_token_type_ids"]
        ):
            self.assertIsNot(original_validator._validate_shape_metadata, sentinel)
        self.assertIs(original_validator._validate_shape_metadata, sentinel)

        with self.assertRaisesRegex(RuntimeError, "stop"):
            with _scoped_dynamic_validator_override(
                aligned=["attention_mask", "mm_token_type_ids"]
            ):
                raise RuntimeError("stop")
        self.assertIs(original_validator._validate_shape_metadata, sentinel)

    def test_parent_observed_set_is_bound_after_formula_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "parent.tar"
            archive.write_bytes(b"parent")
            parent_binding = {
                **_file_binding(archive),
                "source_git_commit": "a" * 40,
                "required_outcome": "NO_GO_V2_1_FULL_45_SUBSTRATE",
                "native_generation_repeat_count": 2,
            }
            state = {
                "state": {"index": 7},
                "native_generations": [
                    {"metadata": _shape()},
                    {"metadata": _shape()},
                ],
            }
            evidence = SimpleNamespace(
                outcome="NO_GO_V2_1_FULL_45_SUBSTRATE",
                files={"states/007.json": json.dumps(state).encode("utf-8")},
            )
            config = {
                "parent_v2_1_archive": parent_binding,
                "repair_contract": {
                    "observed_pre_repair_effective_visual_tokens_per_image": [2700]
                },
            }
            with mock.patch(
                "scripts.validate_spatial_reference_audit_v1_validation_repair_v1."
                "read_full_45_evidence_archive",
                return_value=evidence,
            ):
                shapes, _, observed = _parent_shapes(
                    parent_archive=archive,
                    config=config,
                    state_indices=[7],
                )
                self.assertEqual(shapes[7]["effective_visual_tokens"], 2700)
                self.assertEqual(observed, [2700])
                config["repair_contract"][
                    "observed_pre_repair_effective_visual_tokens_per_image"
                ] = [2560]
                with self.assertRaisesRegex(ValueError, "observed visual-token"):
                    _parent_shapes(
                        parent_archive=archive,
                        config=config,
                        state_indices=[7],
                    )


class RepairEvidenceAndOutputTest(unittest.TestCase):
    def test_pre_repair_evidence_binds_failure_and_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "audit"
            root.mkdir()
            (root / "terminal.json").write_text("{}\n", encoding="utf-8")
            ledger = base / "ledger.json"
            ledger.write_text("{}\n", encoding="utf-8")
            log = base / "formal.log"
            marker = (
                f'{ORIGINAL_FAILURE["exception_type"]}: '
                f'{ORIGINAL_FAILURE["message"]}'
            )
            log.write_text(f"Traceback line\n{marker}\n", encoding="utf-8")
            exit_file = base / "formal.exit"
            exit_file.write_text("1\n", encoding="utf-8")
            root_files = _collect_tree_files(root)
            packaging_files = collect_spatial_reference_audit_files(root, ledger)
            config = {
                "outputs": {"summary": str(root / "summary.json")},
                "pre_repair_evidence": {
                    "audit_root": str(root),
                    "root_tree_inventory": {
                        "file_count": len(root_files),
                        "sha256": _tree_inventory_sha256(root_files),
                    },
                    "packaging_tree_inventory": {
                        "file_count": len(packaging_files),
                        "sha256": _tree_inventory_sha256(packaging_files),
                    },
                    "global_attempt_ledger": _file_binding(ledger),
                    "original_formal_log": _file_binding(log),
                    "original_formal_exit": {
                        **_file_binding(exit_file),
                        "utf8_content": "1\n",
                    },
                    "original_failure": ORIGINAL_FAILURE,
                },
            }
            before = sorted(path.relative_to(base) for path in base.rglob("*"))
            observed_files, ledger_payload, failure = _validate_pre_repair_evidence(
                config
            )
            after = sorted(path.relative_to(base) for path in base.rglob("*"))
            self.assertEqual(observed_files, root_files)
            self.assertEqual(ledger_payload, ledger.read_bytes())
            self.assertEqual(failure["message"], ORIGINAL_FAILURE["message"])
            self.assertEqual(before, after)

            (root / "summary.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _validate_pre_repair_evidence(config)

    def test_failure_binding_rejects_wrong_traceback_tail(self) -> None:
        evidence = {
            "original_failure": ORIGINAL_FAILURE,
            "original_formal_log": {"sha256": "a" * 64},
            "original_formal_exit": {
                "sha256": "b" * 64,
                "utf8_content": "1\n",
            },
        }
        with self.assertRaisesRegex(ValueError, "traceback"):
            _validate_original_failure(
                log_payload=b"ValueError: different\n",
                exit_payload=b"1\n",
                evidence=evidence,
            )

    def test_only_canonical_summary_is_created_exclusively(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "audit"
            root.mkdir()
            (root / "old-terminal.json").write_text("{}\n", encoding="utf-8")
            pre_repair_files = _collect_tree_files(root)
            ledger = Path(directory) / "ledger.json"
            ledger.write_text("{}\n", encoding="utf-8")
            archive = Path(directory) / "audit.tar"
            summary = root / "summary.json"
            summary_payload = {
                "status": "valid",
                "validation_repair": {
                    "post_write_file_counts": {
                        "expected_root_file_count": 2,
                        "observed_root_file_count": 2,
                        "expected_packaging_file_count": 3,
                        "observed_packaging_file_count": 3,
                    }
                },
            }
            summary_payload["scientific_payload_sha256"] = hashlib.sha256(
                canonical_json_bytes(summary_payload)
            ).hexdigest()
            write_canonical_summary(
                audit_root=root,
                summary_path=summary,
                summary=summary_payload,
            )
            self.assertEqual(
                sorted(
                    path.relative_to(root).as_posix() for path in root.rglob("*")
                ),
                ["old-terminal.json", "summary.json"],
            )
            counts = validate_post_write_state(
                audit_root=root,
                summary_path=summary,
                expected_summary=summary_payload,
                pre_repair_files=pre_repair_files,
                ledger_path=ledger,
                ledger_payload=ledger.read_bytes(),
                original_config={
                    "artifact_packaging": {
                        "canonical_local_archive_path": str(archive)
                    }
                },
            )
            self.assertEqual(counts, {"root_file_count": 2, "packaging_file_count": 3})
            summary.write_text(
                json.dumps(summary_payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "summary bytes"):
                validate_post_write_state(
                    audit_root=root,
                    summary_path=summary,
                    expected_summary=summary_payload,
                    pre_repair_files=pre_repair_files,
                    ledger_path=ledger,
                    ledger_payload=ledger.read_bytes(),
                    original_config={
                        "artifact_packaging": {
                            "canonical_local_archive_path": str(archive)
                        }
                    },
                )
            with self.assertRaises(FileExistsError):
                write_canonical_summary(
                    audit_root=root,
                    summary_path=summary,
                    summary=summary_payload,
                )
            with self.assertRaisesRegex(ValueError, "only canonical"):
                write_canonical_summary(
                    audit_root=root,
                    summary_path=root / "repair" / "manifest.json",
                    summary={"status": "valid"},
                )
            self.assertFalse((root / "repair").exists())

    def test_preexisting_frozen_archive_is_rejected_including_broken_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "audit.tar"
            config = {
                "artifact_packaging": {"canonical_local_archive_path": str(archive)}
            }
            self.assertEqual(validate_frozen_raw_archive_absent(config), archive)
            archive.write_bytes(b"existing")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                validate_frozen_raw_archive_absent(config)
            archive.unlink()
            archive.symlink_to(Path(directory) / "missing")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                validate_frozen_raw_archive_absent(config)


if __name__ == "__main__":
    unittest.main()
