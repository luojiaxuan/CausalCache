from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.validate_restoration_v2_readiness import (
    ALLOWED_ROLES,
    CANONICAL_REMOTE,
    CONFIRM_STATE,
    DETERMINISTIC_PROCESSOR_IMAGES,
    EXPECTED_DEPENDENCY_EVIDENCE_PATHS,
    EXPECTED_DEPENDENCY_NAMES,
    PROCESSOR_AUDIT_SOURCE_PATH,
    PROCESSOR_AUDIT_SOURCE_PATHS,
    PROCESSOR_AUDIT_SUMMARY_PATH,
    READINESS_STATUS,
    TRANSFORMERS_SOURCE_SHA256,
    _dependency_map,
    _load_json_object,
    _validate_dependency_evidence_inventory,
    _validate_preclosure_gpu_compute_audit,
    _validate_processor_audit_summary,
    _validate_processor_audit_git_binding,
    _validate_readiness_manifest,
    _validate_repository_binding,
)


def _processor_sources() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for index, path in enumerate(PROCESSOR_AUDIT_SOURCE_PATHS):
        result[f"source-{index}"] = {
            "role": f"source-{index}",
            "path": path,
            "sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(),
        }
    return result


def _processor_runtime() -> dict[str, object]:
    return {
        "host_alias": "hyper00",
        "host_hostname": "node-radixark-16-0000",
        "container_id": "1" * 64,
        "container_hostname": "1" * 12,
        "container_image_digest": f"sha256:{'2' * 64}",
        "python_version": "3.12.3",
        "platform_machine": "x86_64",
        "transformers_version": "5.6.0",
    }


def _processor_case(name: str, image_counts: list[int]) -> dict[str, object]:
    effective_per_image = 2550
    text_tokens = 8
    sequence_length = effective_per_image * image_counts[0] + text_tokens
    samples = [
        {
            "image_count": image_count,
            "image_grid_thw": [[1, 150, 68] for _ in range(image_count)],
            "effective_visual_tokens_per_image": [
                effective_per_image for _ in range(image_count)
            ],
            "effective_visual_tokens": effective_per_image * image_count,
            "policy_visible_text_tokens": text_tokens,
            "sequence_length": effective_per_image * image_count + text_tokens,
        }
        for image_count in image_counts
    ]
    total_images = sum(image_counts)
    def tensor(shape: list[int], dtype: str) -> dict[str, object]:
        return {
            "shape": shape,
            "dtype": dtype,
            "device": "cpu",
            "requires_grad": False,
        }
    return {
        "status": "passed",
        "case": name,
        "batch_size": len(image_counts),
        "image_counts": image_counts,
        "padding": False,
        "no_padding": True,
        "equal_image_count": len(set(image_counts)) == 1,
        "equal_sequence_length": True,
        "attention_mask_all_one": True,
        "assistant_prefix_tail_exact": True,
        "tensor_inventory": {
            "input_ids": tensor([len(image_counts), sequence_length], "torch.int64"),
            "attention_mask": tensor(
                [len(image_counts), sequence_length], "torch.int64"
            ),
            "pixel_values": tensor([total_images * 10_200, 1536], "torch.float32"),
            "image_grid_thw": tensor([total_images, 3], "torch.int64"),
        },
        "samples": samples,
    }


def _processor_summary() -> dict[str, object]:
    sources = _processor_sources()
    negative = {
        "policy_model_loaded": False,
        "policy_loaded": False,
        "policy_forward_executed": False,
        "policy_generate_executed": False,
        "policy_output": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
    }
    return {
        "schema_version": "0.1.0",
        "protocol_id": "causalcache_restoration_v2",
        "evidence_type": "gui_owl_v2_policy_output_free_processor_audit",
        "outcome": "PASSED_GUI_OWL_V2_PROCESSOR_AUDIT",
        "auto_processor_only": True,
        "auto_processor_loaded": True,
        "model_weights_loaded": False,
        "pretrained_loader_calls": ["AutoProcessor.from_pretrained"],
        "model_weight_files_sha256_verified": True,
        "model_weights_materialized_as_tensors": False,
        "repository": {
            "repository_root": "/data/CausalCache",
            "run_git_commit": "c" * 40,
            "head_verified_as_commit": True,
            "worktree_clean": True,
            "untracked_files_checked": True,
            "submodules_checked": True,
        },
        "source_files": {
            record["path"]: {
                "sha256": record["sha256"],
                "size_bytes": 100 + index,
            }
            for index, record in enumerate(sources.values())
        },
        "runtime_identity": {
            **_processor_runtime(),
            "platform_system": "Linux",
            "torch_distribution_version": "2.11.0",
            "pillow_version": "11.2.1",
            "processor_device": "cpu",
        },
        "processor_audit": {
            "status": "passed",
            "auto_processor_only": True,
            "auto_processor_loaded": True,
            "model_weights_loaded": False,
            "model_identity": {
                "model_dir": "/data/models/gui-owl",
                "model_repo": "mPLUG/GUI-Owl-1.5-8B-Instruct",
                "model_revision": "06d5faecff74840bab2be2425e9c42667a5d04fc",
                "snapshot_manifest_sha256": (
                    "50b675ec31c5c46dbb0d44c137a808fffb9d054916d39b596648d4eb9df7cbc3"
                ),
                "verified_model_file_count": 14,
                "verified_model_total_bytes": 17_545_907_171,
                "transformers_version": "5.6.0",
                "transformers_source_sha256": TRANSFORMERS_SOURCE_SHA256,
            },
            "processor_identity": {
                "processor_class": "Qwen3VLProcessor",
                "tokenizer_class": "Qwen2TokenizerFast",
                "target_effective_visual_tokens_per_image": 2560,
                "target_pixels_per_image": 2_621_440,
                "actual_min_pixels": 2_621_440,
                "actual_max_pixels": 2_621_440,
                "actual_min_equals_max_equals_target": True,
                "spatial_merge_size": 2,
                "local_files_only": True,
            },
            "deterministic_images": DETERMINISTIC_PROCESSOR_IMAGES,
            "tokenizer_boundary": {
                "status": "passed",
                "action": "wait",
                "assistant_prefix_text": "<|im_start|>assistant\n",
                "carrier_text": "Action: Execute the selected mobile action.\n",
                "distance_text": (
                    '<tool_call>\n{"name":"mobile_use","arguments":{"action":"wait"}}\n'
                    "</tool_call>"
                ),
                "assistant_prefix_tokens": 3,
                "carrier_tokens": 8,
                "distance_tokens": 20,
                "joint_boundary_exact": True,
            },
            "single_conversation_one_image": _processor_case(
                "single_conversation_one_image", [1]
            ),
            "single_conversation_five_images": _processor_case(
                "single_conversation_five_images", [5]
            ),
            "nested_batch_two_equal_shape": _processor_case(
                "nested_batch_two_equal_shape", [1, 1]
            ),
            "pretrained_loader_calls": ["AutoProcessor.from_pretrained"],
            "model_weight_files_sha256_verified": True,
            "model_weights_materialized_as_tensors": False,
            **negative,
        },
        **negative,
    }


def _preclosure_gpu_evidence() -> tuple[dict[str, object], dict[str, object]]:
    summary = {
        "outcome": "PASSED_RESTORATION_V2_GPU_COMPUTE_AUDIT",
        "policy_loaded": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
    }
    validation = {
        "outcome": "PASSED_INDEPENDENT_RESTORATION_V2_GPU_COMPUTE_AUDIT_VALIDATION",
        "summary_sha256": "a" * 64,
        "policy_loaded": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "dependency_8_closed": False,
        "screening_unlocked": False,
        "compute_validation": {
            "validation_scalar_host_reads": 0,
            "full_tensor_host_transfers": 0,
            "invalid_numeric_output": "nan_final_distance",
        },
        "planner_validation": {
            "automatic_oom_fallback": False,
            "microbatch_size": 2,
            "planning_scope": "single_decision_state",
        },
    }
    return summary, validation


def _dependencies() -> list[dict[str, object]]:
    return [
        {
            "id": identifier,
            "name": name,
            "status": "passed",
            "evidence": [
                {
                    "path": f"data/evidence-{identifier}.json",
                    "sha256": str(identifier) * 64,
                }
            ],
        }
        for identifier, name in EXPECTED_DEPENDENCY_NAMES.items()
    ]


def _sources() -> dict[str, dict[str, str]]:
    return {
        "gpu_kl": {
            "path": "code/causalcache/restoration_v2_gpu_kl.py",
            "sha256": "a" * 64,
        },
        "readiness_validator": {
            "path": "code/scripts/validate_restoration_v2_readiness.py",
            "sha256": "b" * 64,
        },
    }


def _manifest(
    *,
    config_path: str,
    config_sha256: str,
    sources: dict[str, dict[str, str]],
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "protocol_id": "causalcache_restoration_v2",
        "status": READINESS_STATUS,
        "confirm_locked": True,
        "dependency_count": 8,
        "passed_dependency_count": 8,
        "execution_config": {"path": config_path, "sha256": config_sha256},
        "source_files": [
            {"role": role, **sources[role]} for role in sorted(sources)
        ],
        "implementation_git_commit": "c" * 40,
        "canonical_remote": CANONICAL_REMOTE,
        "branch": "main",
        "materialized_before_any_v2_policy_output": True,
        "policy_output_generated_before_manifest": False,
        "restoration_output_generated_before_manifest": False,
    }


class RestorationV2ReadinessSchemaTest(unittest.TestCase):
    def test_old_gpu_validation_remains_explicitly_preclosure(self) -> None:
        summary, validation = _preclosure_gpu_evidence()
        _validate_preclosure_gpu_compute_audit(
            summary,
            validation,
            summary_sha256="a" * 64,
        )
        for key in ("dependency_8_closed", "screening_unlocked"):
            with self.subTest(key=key):
                changed = copy.deepcopy(validation)
                changed[key] = True
                with self.assertRaisesRegex(ValueError, "GPU audit contract"):
                    _validate_preclosure_gpu_compute_audit(
                        summary,
                        changed,
                        summary_sha256="a" * 64,
                    )

    def test_real_processor_audit_is_fixed_and_fail_closed(self) -> None:
        summary = _processor_summary()
        _validate_processor_audit_summary(
            summary,
            configured_sources=_processor_sources(),
            execution_runtime=_processor_runtime(),
        )
        self.assertEqual(
            PROCESSOR_AUDIT_SOURCE_PATH,
            "code/scripts/audit_gui_owl_v2_processor.py",
        )
        self.assertEqual(
            PROCESSOR_AUDIT_SUMMARY_PATH,
            "data/results/restoration_v2_processor_audit/summary.json",
        )

        def remove_audited_source(value: dict[str, object]) -> None:
            source_files = value["source_files"]
            assert isinstance(source_files, dict)
            source_files.pop(PROCESSOR_AUDIT_SOURCE_PATH)

        def change_nested_flag(value: dict[str, object]) -> None:
            audit = value["processor_audit"]
            assert isinstance(audit, dict)
            audit["policy_forward_executed"] = True

        def change_snapshot(value: dict[str, object]) -> None:
            audit = value["processor_audit"]
            assert isinstance(audit, dict)
            model = audit["model_identity"]
            assert isinstance(model, dict)
            model["snapshot_manifest_sha256"] = "0" * 64

        def change_transformers_source(value: dict[str, object]) -> None:
            audit = value["processor_audit"]
            assert isinstance(audit, dict)
            model = audit["model_identity"]
            assert isinstance(model, dict)
            sources = model["transformers_source_sha256"]
            assert isinstance(sources, dict)
            sources["processing_qwen3_vl.py"] = "0" * 64

        def change_deterministic_image(value: dict[str, object]) -> None:
            audit = value["processor_audit"]
            assert isinstance(audit, dict)
            images = audit["deterministic_images"]
            assert isinstance(images, dict)
            portrait = images["portrait"]
            assert isinstance(portrait, dict)
            portrait["sha256"] = "0" * 64

        def change_pixel_target(value: dict[str, object]) -> None:
            audit = value["processor_audit"]
            assert isinstance(audit, dict)
            processor = audit["processor_identity"]
            assert isinstance(processor, dict)
            processor["target_pixels_per_image"] = 2_621_439

        def change_token_boundary(value: dict[str, object]) -> None:
            audit = value["processor_audit"]
            assert isinstance(audit, dict)
            tokenizer = audit["tokenizer_boundary"]
            assert isinstance(tokenizer, dict)
            tokenizer["joint_boundary_exact"] = False

        def change_five_image_case(value: dict[str, object]) -> None:
            audit = value["processor_audit"]
            assert isinstance(audit, dict)
            case = audit["single_conversation_five_images"]
            assert isinstance(case, dict)
            case["image_counts"] = [4]

        mutations = (
            (lambda value: value.__setitem__("auto_processor_only", False), "execution boundary"),
            (lambda value: value.__setitem__("auto_processor_loaded", False), "execution boundary"),
            (lambda value: value.__setitem__("model_weights_loaded", True), "execution boundary"),
            (
                lambda value: value.__setitem__(
                    "pretrained_loader_calls", ["AutoModel.from_pretrained"]
                ),
                "execution boundary",
            ),
            (
                lambda value: value.__setitem__(
                    "model_weights_materialized_as_tensors", True
                ),
                "execution boundary",
            ),
            (lambda value: value.__setitem__("policy_generate_executed", True), "execution boundary"),
            (change_nested_flag, "nested processor execution boundary"),
            (remove_audited_source, "source inventory"),
            (change_snapshot, "model snapshot"),
            (change_transformers_source, "model snapshot"),
            (change_deterministic_image, "deterministic image"),
            (change_pixel_target, "pixel target"),
            (change_token_boundary, "tokenizer boundary"),
            (change_five_image_case, "case single_conversation_five_images"),
        )
        for mutation, pattern in mutations:
            with self.subTest(pattern=pattern):
                changed = copy.deepcopy(summary)
                mutation(changed)
                with self.assertRaisesRegex(ValueError, pattern):
                    _validate_processor_audit_summary(
                        changed,
                        configured_sources=_processor_sources(),
                        execution_runtime=_processor_runtime(),
                    )

        changed_runtime = _processor_runtime()
        changed_runtime["host_alias"] = "hyper01"
        with self.assertRaisesRegex(ValueError, "runtime identity"):
            _validate_processor_audit_summary(
                summary,
                configured_sources=_processor_sources(),
                execution_runtime=changed_runtime,
            )

    def test_processor_audit_sources_are_rehashed_from_ancestor_commit(self) -> None:
        summary = _processor_summary()
        run_commit = "c" * 40
        head = "d" * 40
        inventory = summary["source_files"]
        assert isinstance(inventory, dict)

        def git_output(_root: Path, *arguments: str) -> str:
            if arguments == (
                "rev-parse",
                "--verify",
                f"{run_commit}^{{commit}}",
            ):
                return run_commit
            if arguments == ("rev-parse", "HEAD"):
                return head
            raise AssertionError(arguments)

        def git_bytes(_root: Path, commit: str, path: str) -> bytes:
            self.assertEqual(commit, run_commit)
            record = inventory[path]
            assert isinstance(record, dict)
            return str(record["sha256"]).encode("ascii")

        def fake_sha256(payload: bytes) -> str:
            return payload.decode("ascii")

        with (
            mock.patch(
                "scripts.validate_restoration_v2_readiness._git_output",
                side_effect=git_output,
            ),
            mock.patch(
                "scripts.validate_restoration_v2_readiness._git_bytes",
                side_effect=git_bytes,
            ),
            mock.patch(
                "scripts.validate_restoration_v2_readiness._sha256_bytes",
                side_effect=fake_sha256,
            ),
            mock.patch(
                "scripts.validate_restoration_v2_readiness.subprocess.run",
                return_value=mock.Mock(returncode=0),
            ),
        ):
            self.assertEqual(
                _validate_processor_audit_git_binding(
                    summary,
                    repository_root=Path("/repo"),
                ),
                run_commit,
            )

        def bad_git_bytes(_root: Path, _commit: str, _path: str) -> bytes:
            return b"0" * 64

        with (
            mock.patch(
                "scripts.validate_restoration_v2_readiness._git_output",
                side_effect=git_output,
            ),
            mock.patch(
                "scripts.validate_restoration_v2_readiness._git_bytes",
                side_effect=bad_git_bytes,
            ),
            mock.patch(
                "scripts.validate_restoration_v2_readiness._sha256_bytes",
                side_effect=fake_sha256,
            ),
            mock.patch(
                "scripts.validate_restoration_v2_readiness.subprocess.run",
                return_value=mock.Mock(returncode=0),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "committed source hash"):
                _validate_processor_audit_git_binding(
                    summary,
                    repository_root=Path("/repo"),
                )

    def test_dependency_eight_rejects_missing_real_processor_summary(self) -> None:
        dependencies = _dependencies()
        for record in dependencies:
            identifier = record["id"]
            assert isinstance(identifier, int)
            record["evidence"] = [
                {"path": path, "sha256": "a" * 64}
                for path in EXPECTED_DEPENDENCY_EVIDENCE_PATHS[identifier]
            ]
        dependency_map = _dependency_map(dependencies)
        _validate_dependency_evidence_inventory(dependency_map)
        dependency_eight = dependency_map[8]
        evidence = dependency_eight["evidence"]
        assert isinstance(evidence, list)
        dependency_eight["evidence"] = [
            record
            for record in evidence
            if record["path"] != PROCESSOR_AUDIT_SUMMARY_PATH
        ]
        with self.assertRaisesRegex(ValueError, "dependency 8 evidence inventory"):
            _validate_dependency_evidence_inventory(dependency_map)

    def test_dummy_real_processor_source_cannot_satisfy_summary_binding(self) -> None:
        sources = _processor_sources()
        audit_source = next(
            record
            for record in sources.values()
            if record["path"] == PROCESSOR_AUDIT_SOURCE_PATH
        )
        audit_source["path"] = "code/scripts/dummy_real_processor.py"
        with self.assertRaisesRegex(ValueError, "processor source hash"):
            _validate_processor_audit_summary(
                _processor_summary(),
                configured_sources=sources,
                execution_runtime=_processor_runtime(),
            )

    def test_dependency_inventory_requires_exact_eight_passed_records(self) -> None:
        observed = _dependency_map(_dependencies())
        self.assertEqual(tuple(sorted(observed)), tuple(range(1, 9)))

        for mutation, pattern in (
            (lambda values: values.pop(), "exactly eight"),
            (
                lambda values: values[0].__setitem__("status", "pending"),
                "did not pass",
            ),
            (
                lambda values: values[1].__setitem__("id", 1),
                "duplicate dependency",
            ),
        ):
            with self.subTest(pattern=pattern):
                changed = copy.deepcopy(_dependencies())
                mutation(changed)
                with self.assertRaisesRegex(ValueError, pattern):
                    _dependency_map(changed)

    def test_readiness_manifest_requires_screening_only_and_confirm_lock(self) -> None:
        config_path = "code/configs/restoration_v2_execution_hyper00_v1.json"
        config_sha = "d" * 64
        sources = _sources()
        manifest = _manifest(
            config_path=config_path,
            config_sha256=config_sha,
            sources=sources,
        )
        self.assertEqual(
            _validate_readiness_manifest(
                manifest,
                config_path=config_path,
                config_sha256=config_sha,
                source_files=sources,
            ),
            "c" * 40,
        )
        mutations = (
            ("status", "CONFIRM_ALLOWED"),
            ("confirm_locked", False),
            ("passed_dependency_count", 7),
            ("policy_output_generated_before_manifest", True),
            ("restoration_output_generated_before_manifest", True),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                changed = copy.deepcopy(manifest)
                changed[key] = value
                with self.assertRaisesRegex(ValueError, "state or pre-output"):
                    _validate_readiness_manifest(
                        changed,
                        config_path=config_path,
                        config_sha256=config_sha,
                        source_files=sources,
                    )

        changed = copy.deepcopy(manifest)
        changed["source_files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source hashes"):
            _validate_readiness_manifest(
                changed,
                config_path=config_path,
                config_sha256=config_sha,
                source_files=sources,
            )

    def test_strict_json_loader_rejects_duplicates_and_nonstandard_nan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"status":"a","status":"b"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                _load_json_object(duplicate)
            nonstandard = Path(directory) / "nan.json"
            nonstandard.write_text('{"value":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-standard JSON"):
                _load_json_object(nonstandard)


class RestorationV2ReadinessGitBindingTest(unittest.TestCase):
    def test_git_binding_requires_clean_pushed_main_and_ancestor_source_commit(self) -> None:
        root = Path("/repo")
        implementation = "c" * 40
        head = "d" * 40
        config_path = "code/configs/restoration_v2_execution_hyper00_v1.json"
        config_sha = "e" * 64
        sources = _sources()

        def git_output(_root: Path, *arguments: str) -> str:
            mapping = {
                ("status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"): "",
                ("rev-parse", "HEAD"): head,
                ("rev-parse", "origin/main"): head,
                ("remote", "get-url", "origin"): CANONICAL_REMOTE,
                ("rev-parse", "--verify", f"{implementation}^{{commit}}"): implementation,
            }
            return mapping[arguments]

        def git_bytes(_root: Path, _commit: str, path: str) -> bytes:
            import hashlib

            if path == config_path:
                payload = b"config"
                self.assertEqual(hashlib.sha256(payload).hexdigest(), config_sha)
                return payload
            raise AssertionError(path)

        # note (luojiaxuan): The fixture uses fixed digests, so return exact bytes
        # through a patched hash helper instead of coupling the test to SHA internals.
        source_by_path = {record["path"]: record["sha256"] for record in sources.values()}

        def fake_digest(payload: bytes) -> str:
            if payload == b"config":
                return config_sha
            return payload.decode("ascii")

        def git_source_bytes(_root: Path, _commit: str, path: str) -> bytes:
            if path == config_path:
                return b"config"
            return source_by_path[path].encode("ascii")

        completed = mock.Mock(returncode=0)
        with (
            mock.patch(
                "scripts.validate_restoration_v2_readiness._git_output",
                side_effect=git_output,
            ),
            mock.patch(
                "scripts.validate_restoration_v2_readiness._git_bytes",
                side_effect=git_source_bytes,
            ),
            mock.patch(
                "scripts.validate_restoration_v2_readiness._sha256_bytes",
                side_effect=fake_digest,
            ),
            mock.patch(
                "scripts.validate_restoration_v2_readiness.subprocess.run",
                return_value=completed,
            ),
        ):
            result = _validate_repository_binding(
                root,
                implementation_commit=implementation,
                execution_config_path=config_path,
                execution_config_sha256=config_sha,
                source_files=sources,
            )
        self.assertEqual(result["current_git_commit"], head)
        self.assertEqual(result["implementation_git_commit"], implementation)

    def test_public_state_contract_is_screening_not_confirm(self) -> None:
        self.assertEqual(READINESS_STATUS, "SCREENING_ALLOWED")
        self.assertEqual(CONFIRM_STATE, "CONFIRM_LOCKED")
        self.assertEqual(ALLOWED_ROLES, ("v2_label_train", "v2_development"))


if __name__ == "__main__":
    unittest.main()
