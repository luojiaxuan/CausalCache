from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from causalcache.restoration_v2_2_policy_vision import (
    PolicyVisionImageIdentity,
    PolicyVisionWorkItem,
)
from scripts.run_restoration_v2_2_policy_vision_baseline import (
    HOST_EVIDENCE_COLLECTOR,
    _expected_runtime_metadata,
    _load_primary_labels_from_verified_bytes,
    _run_feature_workers,
    _validate_cross_device_replay,
    _validate_host_evidence_record,
    _validate_preprocessing_geometry,
    _validate_mode,
    _validate_existing_result,
    _validate_runtime_metadata,
    _validate_source_unchanged,
    _write_new_result,
)


def _work_items() -> tuple[PolicyVisionWorkItem, ...]:
    items = []
    for ordinal, state_index in enumerate(range(2, 45, 3)):
        trajectory = "0131649930078879" if ordinal == 0 else f"trajectory-{ordinal:02d}"
        event_images = tuple(
            (
                step,
                PolicyVisionImageIdentity(
                    image_member_path=f"images/{ordinal:02d}-{step}.png",
                    image_sha256=f"{ordinal * 5 + step:064x}",
                ),
            )
            for step in (1, 2, 3, 4)
        )
        items.append(
            PolicyVisionWorkItem(
                primary_ordinal=ordinal,
                state_index=state_index,
                role="v2_label_train" if ordinal < 10 else "v2_development",
                trajectory_id=trajectory,
                state_id=f"{trajectory}:decision_step:006",
                event_images=event_images,
                current_image=PolicyVisionImageIdentity(
                    image_member_path=f"images/{ordinal:02d}-5.png",
                    image_sha256=f"{ordinal * 5 + 5:064x}",
                ),
            )
        )
    return tuple(items)


class _Future:
    def __init__(self, value: object) -> None:
        self.value = value

    def result(self) -> object:
        return self.value


class _Executor:
    submissions: list[tuple[object, dict[str, object]]] = []

    def __init__(self, *, max_workers: int, mp_context: object) -> None:
        if max_workers != 2 or mp_context != "spawn-context":
            raise AssertionError("runner did not freeze a two-worker spawn executor")

    def __enter__(self) -> "_Executor":
        type(self).submissions = []
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def submit(self, function: object, **kwargs: object) -> _Future:
        type(self).submissions.append((function, kwargs))
        return _Future({"worker_id": kwargs["worker_id"]})


class RunRestorationV22PolicyVisionBaselineTest(unittest.TestCase):
    def test_workers_receive_two_shards_and_never_receive_labels(self) -> None:
        items = _work_items()
        with patch(
            "scripts.run_restoration_v2_2_policy_vision_baseline."
            "ProcessPoolExecutor",
            _Executor,
        ), patch(
            "scripts.run_restoration_v2_2_policy_vision_baseline."
            "multiprocessing.get_context",
            return_value="spawn-context",
        ):
            outputs = _run_feature_workers(
                work_items=items,
                derived_root=Path("/derived"),
                derived_payload_prefix="derived/restoration-v2-v1",
                expected_tar_member_count=210,
                model_dir=Path("/model"),
                snapshot_manifest=Path("/snapshot.json"),
                expected_gpu_uuids=(
                    "GPU-00000000-0000-0000-0000-000000000001",
                    "GPU-00000000-0000-0000-0000-000000000002",
                ),
            )
        self.assertEqual([output["worker_id"] for output in outputs], ["even", "odd"])
        self.assertEqual(len(_Executor.submissions), 2)
        even = _Executor.submissions[0][1]
        odd = _Executor.submissions[1][1]
        self.assertEqual(len(even["canonical_items"]), 8)
        self.assertEqual(len(odd["canonical_items"]), 7)
        self.assertIsNone(even["sentinel_item"])
        self.assertEqual(odd["sentinel_item"], items[0])
        self.assertEqual(even["device"], "cuda:0")
        self.assertEqual(odd["device"], "cuda:1")
        exact_worker_keys = {
            "worker_id",
            "device",
            "expected_gpu_uuid",
            "canonical_items",
            "sentinel_item",
            "derived_root",
            "derived_payload_prefix",
            "expected_tar_member_count",
            "model_dir",
            "snapshot_manifest",
        }
        self.assertEqual(set(even), exact_worker_keys)
        self.assertEqual(set(odd), exact_worker_keys)
        self.assertFalse(
            any("label" in key or "ocr" in key or "goal" in key for key in even)
        )

    def test_atomic_output_is_exact_three_and_cleans_failed_staging(self) -> None:
        files = {
            "README.md": b"readme\n",
            "state_scores.jsonl": b"{}\n",
            "summary.json": b"{}\n",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            with patch(
                "scripts.run_restoration_v2_2_policy_vision_baseline.os.replace",
                side_effect=OSError("publish failed"),
            ):
                with self.assertRaisesRegex(OSError, "publish failed"):
                    _write_new_result(output, files)
            self.assertFalse(output.exists())
            self.assertFalse((Path(temporary) / ".result.staging").exists())
            callback_calls = []

            def pre_publish_check() -> None:
                callback_calls.append(True)
                self.assertFalse(output.exists())
                self.assertTrue((Path(temporary) / ".result.staging").is_dir())

            _write_new_result(
                output,
                files,
                pre_publish_check=pre_publish_check,
            )
            self.assertEqual(callback_calls, [True])
            _validate_existing_result(output, files)
            self.assertEqual(sorted(path.name for path in output.iterdir()), sorted(files))
            with self.assertRaises(FileExistsError):
                _write_new_result(output, files)

            rejected = Path(temporary) / "rejected"
            with self.assertRaisesRegex(ValueError, "snapshot drifted"):
                _write_new_result(
                    rejected,
                    files,
                    pre_publish_check=lambda: (_ for _ in ()).throw(
                        ValueError("snapshot drifted")
                    ),
                )
            self.assertFalse(rejected.exists())
            self.assertFalse((Path(temporary) / ".rejected.staging").exists())

    def test_verified_label_bytes_are_the_only_semantic_input(self) -> None:
        original_payload = b"immutable-label-payload"
        mutated_payload = b"mutated-after-snapshot"
        contract = SimpleNamespace(
            data={
                "immutable_inputs": {
                    "restoration_labels": {
                        "raw_archive_size_bytes": len(original_payload),
                        "raw_archive_sha256": hashlib.sha256(
                            original_payload
                        ).hexdigest(),
                    }
                }
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "labels.tar"
            archive.write_bytes(original_payload)

            def load_snapshot(
                snapshot: Path,
                work_items: object,
            ) -> dict[str, bytes]:
                self.assertNotEqual(snapshot, archive)
                archive.write_bytes(mutated_payload)
                self.assertEqual(snapshot.read_bytes(), original_payload)
                self.assertEqual(work_items, ())
                return {"loaded": snapshot.read_bytes()}

            with patch(
                "scripts.run_restoration_v2_2_policy_vision_baseline."
                "load_primary_label_states",
                side_effect=load_snapshot,
            ):
                loaded = _load_primary_labels_from_verified_bytes(
                    contract=contract,
                    labels_archive=archive,
                    work_items=(),
                )
            self.assertEqual(loaded, {"loaded": original_payload})
            self.assertEqual(archive.read_bytes(), mutated_payload)

    def test_source_closure_rejects_untracked_imports_and_unrelated_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "CausalCache Test"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=root,
                check=True,
            )
            package = root / "code" / "causalcache"
            scripts = root / "code" / "scripts"
            package.mkdir(parents=True)
            scripts.mkdir(parents=True)
            (package / "frozen.py").write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-m", "freeze source"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            source = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            subprocess.run(
                ["git", "update-ref", "refs/remotes/origin/main", source],
                cwd=root,
                check=True,
            )
            _validate_source_unchanged(root, source)

            injected = scripts / "injected.py"
            injected.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "untracked files"):
                _validate_source_unchanged(root, source)
            subprocess.run(["git", "add", str(injected)], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-m", "inject tracked source"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            descendant = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            subprocess.run(
                ["git", "update-ref", "refs/remotes/origin/main", descendant],
                cwd=root,
                check=True,
            )
            with self.assertRaisesRegex(ValueError, "source closure drifted"):
                _validate_source_unchanged(root, source)

            tree = subprocess.run(
                ["git", "rev-parse", f"{source}^{{tree}}"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            unrelated = subprocess.run(
                ["git", "commit-tree", tree],
                cwd=root,
                check=True,
                input="unrelated source\n",
                capture_output=True,
                text=True,
            ).stdout.strip()
            with self.assertRaisesRegex(ValueError, "not an ancestor"):
                _validate_source_unchanged(root, unrelated)

    def test_host_evidence_is_exact_and_alias_bound(self) -> None:
        gpu_uuids = (
            "GPU-00000000-0000-0000-0000-000000000001",
            "GPU-00000000-0000-0000-0000-000000000002",
        )
        container_id = "a" * 64
        evidence = {
            "schema_version": "1.0.0",
            "collector": HOST_EVIDENCE_COLLECTOR,
            "host_alias": "hyper00",
            "host_hostname": "node-radixark-16-0000",
            "container_id": container_id,
            "container_name": "sglang-omni-jaxan-07160000",
            "container_image_digest": "sha256:" + "b" * 64,
            "nvidia_driver_version": "570.172.08",
            "gpu_uuids": list(gpu_uuids),
        }
        arguments = {
            "host_alias": "hyper00",
            "host_hostname": "node-radixark-16-0000",
            "container_hostname": container_id[:12],
            "container_name": "sglang-omni-jaxan-07160000",
            "container_image_digest": "sha256:" + "b" * 64,
            "nvidia_driver_version": "570.172.08",
            "expected_gpu_uuids": gpu_uuids,
        }
        self.assertEqual(
            _validate_host_evidence_record(evidence, **arguments),
            evidence,
        )
        with self.assertRaisesRegex(ValueError, "schema drifted"):
            _validate_host_evidence_record(
                {**evidence, "host_evidence_sha256": "c" * 64},
                **arguments,
            )
        with self.assertRaisesRegex(ValueError, "differs from formal identity"):
            _validate_host_evidence_record(
                {**evidence, "host_hostname": "node-radixark-16-0001"},
                **arguments,
            )

    def test_runtime_metadata_requires_the_exact_frozen_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "baseline.json"
            manifest.write_text(
                json.dumps(
                    {
                        "implementation_contract": {
                            "policy_vision": {
                                "transformers_source_sha256": "c" * 64
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            contract = SimpleNamespace(
                repository_root=root,
                data={
                    "execution_contract": {
                        "runtime": {
                            "expected_execution_stack": {
                                "gpu_name": "NVIDIA H200",
                                "python_version": "3.12.0",
                                "torch_version": "2.11.0",
                                "torch_cuda_version": "13.0",
                                "cudnn_version": 91000,
                                "transformers_version": "4.57.6",
                                "pillow_version": "12.0.0",
                            },
                            "eager_control_flags": {
                                "seed": 20260716,
                                "cudnn_deterministic": False,
                                "cudnn_benchmark": False,
                                "cuda_matmul_allow_tf32": False,
                                "cudnn_allow_tf32": False,
                                "float32_matmul_precision": "highest",
                            },
                            "worker_identity_fields_required": [
                                "device",
                                "gpu_uuid",
                                "gpu_pci_bus_id",
                                "logical_device_index",
                                "nvidia_smi_index",
                            ],
                        }
                    },
                    "immutable_inputs": {
                        "model_snapshot": {
                            "repo": "example/model",
                            "revision": "d" * 40,
                            "manifest_sha256": "e" * 64,
                            "file_count": 5,
                            "total_bytes": 12345,
                        },
                        "baseline_implementation": {
                            "manifest_path": "baseline.json"
                        },
                    },
                    "feature_contract": {"model_class": "GuiOwlForConditionalGeneration"},
                },
            )
            gpu_uuid = "GPU-00000000-0000-0000-0000-000000000001"
            metadata = _expected_runtime_metadata(
                contract=contract,
                device="cuda:0",
                gpu_uuid=gpu_uuid,
            )
            metadata.update(
                {
                    "gpu_pci_bus_id": "00000000:01:00.0",
                    "logical_device_index": 0,
                    "nvidia_smi_index": 3,
                }
            )
            _validate_runtime_metadata(
                metadata=metadata,
                contract=contract,
                worker_id="even",
                device="cuda:0",
                gpu_uuid=gpu_uuid,
            )

            with self.assertRaisesRegex(ValueError, "differs from contract"):
                _validate_runtime_metadata(
                    metadata={**metadata, "unexpected": True},
                    contract=contract,
                    worker_id="even",
                    device="cuda:0",
                    gpu_uuid=gpu_uuid,
                )
            with self.assertRaisesRegex(ValueError, "differs from contract"):
                _validate_runtime_metadata(
                    metadata={**metadata, "processor_interface": "drifted"},
                    contract=contract,
                    worker_id="even",
                    device="cuda:0",
                    gpu_uuid=gpu_uuid,
                )

    def test_cross_device_replay_above_tolerance_fails_closed(self) -> None:
        source_scores = {"1": 0.9, "2": 0.8, "3": 0.7, "4": 0.6}
        verification_scores = dict(source_scores)
        verification_scores["1"] += 2e-6
        differences = {
            key: abs(source_scores[key] - verification_scores[key])
            for key in source_scores
        }
        feature = {
            "state": {"state_id": "0131649930078879:decision_step:006"},
            "candidate_scores": [
                {
                    "event_step_id": step,
                    "policy_vision_cosine": source_scores[str(step)],
                    "image_grid_thw": [1, 2, 2],
                    "merged_token_count": 1,
                }
                for step in (1, 2, 3, 4)
            ],
            "current_image_grid_thw": [1, 2, 2],
            "current_merged_token_count": 1,
            "ranked_event_step_ids": [1, 2, 3, 4],
            "selected_coalition": [1, 2],
        }
        cross = {
            "primary_ordinal": 0,
            "state_index": 2,
            "state_id": "0131649930078879:decision_step:006",
            "source_worker": "even",
            "verification_worker": "odd",
            "per_event_absolute_score_difference": differences,
            "maximum_absolute_score_difference": max(differences.values()),
            "ranking_equal": True,
            "selection_equal": True,
            "absolute_tolerance": 1e-6,
            "verification_scores_by_event_step": verification_scores,
            "verification_ranked_event_step_ids": [1, 2, 3, 4],
            "verification_selected_coalition": [1, 2],
            "verification_image_grid_thw": [[1, 2, 2]] * 5,
            "verification_merged_token_counts": [1] * 5,
        }
        with self.assertRaisesRegex(ValueError, "exceeds tolerance"):
            _validate_cross_device_replay(cross, (feature,))

    def test_preprocessing_geometry_requires_exact_fifteen_states(self) -> None:
        contract = SimpleNamespace(data={})
        with self.assertRaisesRegex(ValueError, "15 canonical states"):
            _validate_preprocessing_geometry((), contract)

    def test_validate_mode_never_enters_gpu_worker_path(self) -> None:
        files = {
            "README.md": b"readme\n",
            "state_scores.jsonl": b"{}\n",
            "summary.json": b"{}\n",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result"
            output.mkdir()
            (output / "summary.json").write_bytes(b"{}")
            contract = SimpleNamespace(repository_root=root)
            summary = {
                "status": "COMPLETED_RESTORATION_V2_2_POLICY_VISION_BASELINE_V1",
                "protocol_id": "causalcache_restoration_v2_2_policy_vision_baseline_v1",
                "source_execution": {"source_git_commit": "a" * 40},
                "execution": {},
            }
            with patch(
                "scripts.run_restoration_v2_2_policy_vision_baseline."
                "_validate_source_unchanged"
            ), patch(
                "scripts.run_restoration_v2_2_policy_vision_baseline."
                "_validate_labels_archive"
            ), patch(
                "scripts.run_restoration_v2_2_policy_vision_baseline."
                "_read_recorded_rows",
                return_value=({},),
            ), patch(
                "scripts.run_restoration_v2_2_policy_vision_baseline."
                "_feature_record_from_evaluated_row",
                return_value={"state": {"state_id": "state"}},
            ), patch(
                "scripts.run_restoration_v2_2_policy_vision_baseline."
                "strict_json_object_bytes",
                return_value=summary,
            ), patch(
                "scripts.run_restoration_v2_2_policy_vision_baseline."
                "load_identity_witness",
                return_value=((), {}),
            ), patch(
                "scripts.run_restoration_v2_2_policy_vision_baseline."
                "_expected_files_from_features",
                return_value=files,
            ), patch(
                "scripts.run_restoration_v2_2_policy_vision_baseline."
                "_validate_existing_result"
            ), patch(
                "scripts.run_restoration_v2_2_policy_vision_baseline."
                "_run_feature_workers",
                side_effect=AssertionError("validate entered GPU path"),
            ) as gpu_path:
                observed = _validate_mode(
                    contract=contract,
                    output_dir=output,
                    labels_archive=root / "labels.tar",
                    source_commit="a" * 40,
                )
            self.assertEqual(observed, files)
            gpu_path.assert_not_called()


if __name__ == "__main__":
    unittest.main()
