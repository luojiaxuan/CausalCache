from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from causalcache.data.guiodyssey_restoration_v2_expansion import (
    decision_view_events,
)
from causalcache.restoration_v2_2_expansion_substrate_contract import (
    DERIVED_ARTIFACT_PATHS,
    DERIVED_PAYLOAD_MANIFEST_PATH,
    EXPANSION_CONFIG_PATH,
    EXPANSION_SELECTION_PATH,
    EXPECTED_COMPLETION_PROTOCOL_ID,
    EXPECTED_COMPLETION_STATUS,
    EXPECTED_DERIVED_COUNTS,
    EXPECTED_DERIVED_PROTOCOL_ID,
    EXPECTED_DERIVED_STATUS,
    EXPECTED_VALIDATOR_OUTCOME,
    PROTOCOL_ID,
    build_contract,
    canonical_json_bytes,
    pretty_json_bytes,
    sha256_bytes,
    sha256_file,
    validate_completion_manifest,
    validate_contract_data,
)
from causalcache.restoration_v2_2_expansion_substrate_inputs import (
    build_decision_view_input,
    validate_processor_byte_canary,
    validate_request_manifest,
)
from causalcache.low_fidelity_v2 import (
    LowFidelityEventV2,
    serialize_low_fidelity_v2,
)
from scripts.materialize_restoration_v2_2_expansion_substrate_contract import (
    _exclusive_write,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "a" * 40


def completion_manifest() -> dict[str, object]:
    snapshot = {
        "schema_version": "1.0.0",
        "protocol_id": EXPECTED_DERIVED_PROTOCOL_ID,
        "status": EXPECTED_DERIVED_STATUS,
        "dataset_repo": "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
        "counts": EXPECTED_DERIVED_COUNTS,
        "formal_counts_enforced": True,
        "policy_output_generated": False,
        "restoration_output_generated": False,
        "per_decision_view_current_expert_action_payload_included": False,
        "consumer_must_slice_events_by_history_event_step_ids": True,
        "inputs": {
            "label_expansion_config": {
                "path": EXPANSION_CONFIG_PATH,
                "sha256": sha256_file(ROOT / EXPANSION_CONFIG_PATH),
            },
            "expansion_selection_manifest": {
                "path": EXPANSION_SELECTION_PATH,
                "sha256": sha256_file(ROOT / EXPANSION_SELECTION_PATH),
            },
        },
        "generator": {"git_revision": "b" * 40},
    }
    snapshot_payload = pretty_json_bytes(snapshot)
    payload_witness = {
        "path": DERIVED_PAYLOAD_MANIFEST_PATH,
        "sha256": sha256_bytes(snapshot_payload),
        "size_bytes": len(snapshot_payload),
    }
    files = [
        {"path": ".gitattributes", "sha256": "1" * 64, "size_bytes": 42},
        {"path": "README.md", "sha256": "2" * 64, "size_bytes": 2048},
        {
            "path": (
                "derived/restoration-v2-label-expansion-v1/"
                "images-00000-of-00001.tar"
            ),
            "sha256": "3" * 64,
            "size_bytes": 1000000,
        },
        payload_witness,
        {
            "path": (
                "derived/restoration-v2-label-expansion-v1/"
                "ocr-records-00000-of-00001.jsonl"
            ),
            "sha256": "4" * 64,
            "size_bytes": 200000,
        },
        {
            "path": (
                "derived/restoration-v2-label-expansion-v1/"
                "trajectories-00000-of-00001.jsonl"
            ),
            "sha256": "5" * 64,
            "size_bytes": 300000,
        },
    ]
    artifact_total_bytes = sum(record["size_bytes"] for record in files)
    tree_sha = sha256_bytes(canonical_json_bytes(files))
    commands = {
        "build": ["python", "-m", "scripts.build_expansion"],
        "standalone_preupload_validation": [
            "python",
            "-m",
            "scripts.validate_expansion",
        ],
        "hf_upload": ["hf", "upload", "private/repo"],
        "hf_tag_create": ["hf", "repos", "tag", "create", "private/repo"],
        "hf_download": ["hf", "download", "private/repo"],
        "fresh_download_validation": [
            "python",
            "-m",
            "scripts.validate_expansion",
        ],
    }
    return {
        "schema_version": "1.0.0",
        "protocol_id": EXPECTED_COMPLETION_PROTOCOL_ID,
        "status": EXPECTED_COMPLETION_STATUS,
        "source": {
            "builder_git_commit": "b" * 40,
            "clean_checkout": True,
            "head_equals_origin_main": True,
        },
        "artifact": {
            "repo": "gavinlaw/causalcache-guiodyssey-restoration-v2-mobile",
            "tag": "restoration-v2-label-expansion-v1.0.0",
            "immutable_revision": "c" * 40,
            "payload_prefix": "derived/restoration-v2-label-expansion-v1",
            "preupload": {
                "artifact_tree_sha256": tree_sha,
                "artifact_file_count": 6,
                "artifact_total_bytes": artifact_total_bytes,
                "files": files,
                "validator_outcome": EXPECTED_VALIDATOR_OUTCOME,
            },
            "fresh_download": {
                "artifact_tree_sha256": tree_sha,
                "artifact_file_count": 6,
                "artifact_total_bytes": artifact_total_bytes,
                "files": copy.deepcopy(files),
                "validator_outcome": EXPECTED_VALIDATOR_OUTCOME,
                "clean_projection_created": True,
                "direct_repo_snapshot_validation_used": False,
                "payload_prefix_exact_six_only": True,
            },
        },
        "execution": {
            "host": {"alias": "hyper00", "hostname": "node-radixark-16-0000"},
            "container": {
                "name": "sglang-omni-jaxan-07170914",
                "id": "6" * 64,
                "image": "hongccc/sglang-omni:dev",
                "image_digest": f"sha256:{'7' * 64}",
                "device_requests": [],
                "gpu_used": False,
            },
            "utc": {
                "started_at_utc": "2026-07-17T01:14:47Z",
                "ended_at_utc": "2026-07-17T02:00:00Z",
            },
            "commands": commands,
            "logs": {
                name: {
                    "path": f"/data/logs/{name}.log",
                    "sha256": ("8", "9", "a")[index] * 64,
                    "size_bytes": 100 + index,
                }
                for index, name in enumerate(
                    (
                        "build",
                        "standalone_preupload_validation",
                        "fresh_download_validation",
                    )
                )
            },
            "exit_codes": {name: 0 for name in commands},
            "runtime_versions": {
                "python": "3.12.3",
                "rapidocr": "3.8.4",
                "onnxruntime": "1.24.4",
                "Pillow": "12.2.0",
                "numpy": "2.3.5",
                "pyarrow": "24.0.0",
                "huggingface_hub": "1.23.0",
            },
            "source_preflight": {
                "source_file_rehash_count": 16,
                "selected_row_reload_count": 64,
            },
            "exposure_ledger_sha256": (
                "e5f7e7b5ffeb71ac117b5cb4a470eb31648978a734edb9a521125d3bda29a323"
            ),
            "hf_repo_private": True,
            "old_tags_preserved": [
                {
                    "tag": "restoration-v2-derived-v1.0.0",
                    "immutable_revision": (
                        "89f136abaff797e14fe758a198996e51032a10a6"
                    ),
                }
            ],
        },
        "payload_manifest_snapshot": snapshot,
        "negative_declarations": {
            "confirm_state_or_output_accessed": False,
            "policy_module_loaded": False,
            "policy_forward_called": False,
            "policy_output_generated": False,
            "restoration_output_generated": False,
            "gate_training_started": False,
            "matched_nll_started": False,
            "closed_loop_started": False,
        },
    }


def built_contract() -> tuple[dict[str, object], dict[str, object], str]:
    completion = completion_manifest()
    digest = sha256_bytes(pretty_json_bytes(completion))
    contract = build_contract(
        repository_root=ROOT,
        source_git_commit=SOURCE_COMMIT,
        completion_manifest=completion,
        completion_manifest_sha256=digest,
    )
    return contract, completion, digest


class ExpansionSubstrateContractTest(unittest.TestCase):
    def test_contract_freezes_exact_192_state_substrate_schedule(self) -> None:
        contract, completion, digest = built_contract()
        result = validate_contract_data(
            contract,
            repository_root=ROOT,
            completion_manifest=completion,
            completion_manifest_sha256=digest,
            require_git_blobs=False,
        )
        self.assertEqual(contract["protocol_id"], PROTOCOL_ID)
        self.assertEqual(
            contract["data_projection"]["fixed_trajectory_denominator"], 64
        )
        self.assertEqual(contract["data_projection"]["fixed_state_denominator"], 192)
        self.assertEqual(
            contract["substrate_schedule"]["planned_and_maximum_counts"],
            {
                "generation_call_count": 384,
                "teacher_forward_count": 576,
                "kl_measurement_count": 384,
            },
        )
        self.assertEqual(result["worker_state_counts"], {"even": 96, "odd": 96})
        self.assertEqual(result["restoration_label_count"], 0)
        self.assertFalse(result["policy_or_gpu_execution_authorized_by_this_validator"])
        source_paths = [
            record["path"]
            for record in contract["scientific_source_lock"]["source_files"]
        ]
        self.assertEqual(len(source_paths), len(set(source_paths)))
        for required in (
            "code/causalcache/policy/gui_owl_v2.py",
            "code/causalcache/policy/gui_owl_v2_1_runtime.py",
            "code/causalcache/policy/gui_owl_v2_runtime.py",
            "code/causalcache/policy/gui_owl_v2_vision.py",
            "code/causalcache/data/restoration_v2_1_processor_inputs.py",
            "code/configs/gui_owl_1_5_8b_snapshot.json",
            "code/causalcache/data/guiodyssey_restoration_v2_expansion.py",
            "code/causalcache/restoration_v2_2_label_expansion.py",
            "code/scripts/validate_guiodyssey_restoration_v2_expansion.py",
        ):
            self.assertIn(required, source_paths)

    def test_completion_requires_equal_clean_projection_and_preupload_tree(
        self,
    ) -> None:
        completion = completion_manifest()
        validate_completion_manifest(completion)
        mutations = (
            (
                "tree",
                lambda value: value["artifact"]["fresh_download"].__setitem__(
                    "artifact_tree_sha256", "4" * 64
                ),
                "aggregate identity",
            ),
            (
                "snapshot",
                lambda value: value["artifact"]["fresh_download"].__setitem__(
                    "direct_repo_snapshot_validation_used", True
                ),
                "clean exact-six",
            ),
            (
                "output",
                lambda value: value["negative_declarations"].__setitem__(
                    "policy_output_generated", True
                ),
                "negative declarations",
            ),
            (
                "gpu",
                lambda value: value["execution"]["container"].__setitem__(
                    "device_requests", [{"Driver": "nvidia"}]
                ),
                "no GPU device requests",
            ),
            (
                "nonexistent metadata",
                lambda value: value["artifact"]["preupload"]["files"][
                    0
                ].__setitem__("path", "metadata.json"),
                "path drifted",
            ),
            (
                "secret argv",
                lambda value: value["execution"]["commands"]["hf_upload"].append(
                    "--hf_token=secret"
                ),
                "secret-bearing",
            ),
            (
                "old tag",
                lambda value: value["execution"].__setitem__(
                    "old_tags_preserved", []
                ),
                "prior derived-artifact immutable tag",
            ),
        )
        for name, mutate, message in mutations:
            invalid = copy.deepcopy(completion)
            mutate(invalid)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                validate_completion_manifest(invalid)
        self.assertEqual(
            [record["path"] for record in completion["artifact"]["preupload"]["files"]],
            list(DERIVED_ARTIFACT_PATHS),
        )
        self.assertNotIn(
            "metadata.json",
            [record["path"] for record in completion["artifact"]["preupload"]["files"]],
        )

    def test_contract_refuses_schedule_prohibition_and_provenance_mutation(
        self,
    ) -> None:
        contract, completion, digest = built_contract()
        mutations = (
            lambda value: value["substrate_schedule"][
                "planned_and_maximum_counts"
            ].__setitem__("teacher_forward_count", 575),
            lambda value: value["prohibited_work"].__setitem__(
                "maximum_restoration_label_count", 1
            ),
            lambda value: value["immutable_inputs"]["derived_artifact"].__setitem__(
                "artifact_tree_sha256", "5" * 64
            ),
        )
        for mutate in mutations:
            invalid = copy.deepcopy(contract)
            mutate(invalid)
            with self.assertRaisesRegex(ValueError, "deterministic freeze"):
                validate_contract_data(
                    invalid,
                    repository_root=ROOT,
                    completion_manifest=completion,
                    completion_manifest_sha256=digest,
                    require_git_blobs=False,
                )

    def test_decision_view_loader_excludes_current_and_future_actions(self) -> None:
        events = tuple(
            {"step_id": step, "action": f"action-{step}"} for step in range(1, 6)
        )
        decision = {
            "decision_step_id": 4,
            "history_event_step_ids": [1, 2, 3],
        }
        view = decision_view_events(events, decision)
        self.assertEqual([event["step_id"] for event in view], [1, 2, 3])
        self.assertNotIn("action-4", {event["action"] for event in view})
        self.assertNotIn("action-5", {event["action"] for event in view})
        contract, _, _ = built_contract()
        loader = contract["data_projection"]["decision_view_loader"]
        self.assertEqual(loader["callable"], "build_decision_view_input")
        self.assertEqual(loader["strict_slice_primitive"], "decision_view_events")
        self.assertEqual(loader["required_call_count"], 192)
        self.assertFalse(loader["direct_full_trajectory_events_consumption_allowed"])
        self.assertEqual(loader["current_or_future_event_action_exposure_count"], 0)
        self.assertTrue(
            loader[
                "excluded_current_or_future_action_canary_must_leave_"
                "prompt_bytes_unchanged"
            ]
        )
        self.assertTrue(
            loader["included_history_action_canary_must_change_prompt_bytes"]
        )
        self.assertTrue(
            loader["raw_request_manifest_records_actual_included_event_step_ids"]
        )
        self.assertEqual(
            loader["canonical_action_source"],
            "frozen_full_history_policy_native_generation_only",
        )
        self.assertFalse(loader["expert_or_adjacent_event_action_backfill_allowed"])
        self.assertFalse(
            contract["data_projection"][
                "per_decision_view_current_expert_action_payload_included"
            ]
        )
        self.assertTrue(
            contract["data_projection"][
                "shared_trajectory_storage_contains_later_event_actions"
            ]
        )
        invalid = {"decision_step_id": 4, "history_event_step_ids": [1, 2, 3, 4]}
        with self.assertRaisesRegex(ValueError, "strict history prefix"):
            decision_view_events(events, invalid)

    def test_policy_blind_loader_records_actual_slice_and_canary_passes(self) -> None:
        trajectory = {
            "source_id": "trajectory-1",
            "events": [
                {
                    "step_id": step,
                    "executed_action": {"action": f"action-{step}"},
                    "source_tool_call": {"call": f"call-{step}"},
                }
                for step in range(1, 6)
            ],
        }
        decision = {
            "state_id": "trajectory-1:decision_step:004",
            "decision_step_id": 4,
            "history_event_step_ids": [1, 2, 3],
        }

        def serializer(trajectory_value, decision_value):
            decision_input = build_decision_view_input(
                trajectory_value, decision_value
            )
            validate_request_manifest(
                decision_input["request_manifest"],
                decision_value,
                included_events=decision_input["events"],
            )
            return pretty_json_bytes(
                {
                    "included_event_step_ids": decision_input[
                        "request_manifest"
                    ]["included_event_step_ids"],
                    "events": list(decision_input["events"]),
                }
            )

        decision_input = build_decision_view_input(trajectory, decision)
        self.assertEqual(
            decision_input["request_manifest"]["included_event_step_ids"],
            [1, 2, 3],
        )
        result = validate_processor_byte_canary(
            trajectory,
            decision,
            serialize_request=serializer,
        )
        self.assertTrue(result["included_history_action_canary_passed"])
        self.assertTrue(
            result["excluded_current_or_future_action_canary_applied"]
        )
        self.assertTrue(
            result["excluded_current_or_future_action_canary_passed"]
        )
        self.assertEqual(
            result[
                "excluded_current_or_future_action_canary_mutation_count"
            ],
            2,
        )
        self.assertEqual(
            [
                record["event_step_id"]
                for record in result[
                    "excluded_current_or_future_action_canary_results"
                ]
            ],
            [4, 5],
        )

    def test_decision_view_loader_rejects_cross_trajectory_state_identity(
        self,
    ) -> None:
        trajectory = {
            "source_id": "trajectory-1",
            "events": [{"step_id": step} for step in range(1, 6)],
        }
        wrong_decision = {
            "state_id": "trajectory-2:decision_step:004",
            "decision_step_id": 4,
            "history_event_step_ids": [1, 2, 3],
        }
        with self.assertRaisesRegex(ValueError, "does not bind"):
            build_decision_view_input(trajectory, wrong_decision)

    def test_request_manifest_recomputes_actual_slice_and_payload_hash(self) -> None:
        trajectory = {
            "source_id": "trajectory-1",
            "events": [
                {"step_id": step, "value": f"event-{step}"}
                for step in range(1, 6)
            ],
        }
        decision = {
            "state_id": "trajectory-1:decision_step:004",
            "decision_step_id": 4,
            "history_event_step_ids": [1, 2, 3],
        }
        decision_input = build_decision_view_input(trajectory, decision)
        validate_request_manifest(
            decision_input["request_manifest"],
            decision,
            included_events=decision_input["events"],
        )

        bad_hash = copy.deepcopy(decision_input["request_manifest"])
        bad_hash["included_event_payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "payload SHA256"):
            validate_request_manifest(
                bad_hash,
                decision,
                included_events=decision_input["events"],
            )

        bad_steps = copy.deepcopy(decision_input["request_manifest"])
        bad_steps["included_event_step_ids"] = [1, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, "exact history slice"):
            validate_request_manifest(
                bad_steps,
                decision,
                included_events=decision_input["events"],
            )

        with self.assertRaisesRegex(ValueError, "exact history slice"):
            validate_request_manifest(
                decision_input["request_manifest"],
                decision,
                included_events=[*decision_input["events"], trajectory["events"][3]],
            )

    def test_processor_canary_rejects_a_full_events_serializer(self) -> None:
        trajectory = {
            "source_id": "trajectory-1",
            "events": [
                {
                    "step_id": step,
                    "executed_action": {"action": f"action-{step}"},
                    "source_tool_call": {"call": f"call-{step}"},
                }
                for step in range(1, 6)
            ],
        }
        decision = {
            "state_id": "trajectory-1:decision_step:004",
            "decision_step_id": 4,
            "history_event_step_ids": [1, 2, 3],
        }

        def leaky_serializer(trajectory_value, _decision_value):
            return pretty_json_bytes(trajectory_value["events"])

        with self.assertRaisesRegex(ValueError, "excluded current or future"):
            validate_processor_byte_canary(
                trajectory,
                decision,
                serialize_request=leaky_serializer,
            )

    def test_processor_canary_rejects_only_last_future_event_leak(self) -> None:
        trajectory = {
            "source_id": "trajectory-1",
            "events": [
                {
                    "step_id": step,
                    "executed_action": {"action": f"action-{step}"},
                    "source_tool_call": {"call": f"call-{step}"},
                }
                for step in range(1, 6)
            ],
        }
        decision = {
            "state_id": "trajectory-1:decision_step:004",
            "decision_step_id": 4,
            "history_event_step_ids": [1, 2, 3],
        }

        def last_future_leak_serializer(trajectory_value, decision_value):
            decision_input = build_decision_view_input(
                trajectory_value, decision_value
            )
            events_by_step = {
                event["step_id"]: event for event in trajectory_value["events"]
            }
            return pretty_json_bytes(
                {
                    "history": list(decision_input["events"]),
                    "leaked_event_5_action": events_by_step[5]["executed_action"],
                }
            )

        with self.assertRaisesRegex(ValueError, "event step 5"):
            validate_processor_byte_canary(
                trajectory,
                decision,
                serialize_request=last_future_leak_serializer,
            )

    def test_processor_canary_tracks_low_fidelity_action_serialization(self) -> None:
        events = []
        for step in range(1, 6):
            summary = LowFidelityEventV2(
                step_id=step,
                action_type="wait",
                action_argument="wait",
                foreground_app="unknown",
                screen_text_added=(),
                screen_text_removed=(),
                screen_change="none",
                executor_result="unknown",
            )
            serialized = serialize_low_fidelity_v2(summary)
            events.append(
                {
                    "step_id": step,
                    "executed_action": {"action": "wait"},
                    "source_tool_call": {"action": "wait"},
                    "low_fidelity_v2": summary.to_ordered_dict(),
                    "low_fidelity_v2_serialized": serialized.decode("utf-8"),
                    "low_fidelity_v2_sha256": sha256_bytes(serialized),
                }
            )
        trajectory = {"source_id": "trajectory-1", "events": events}
        decision = {
            "state_id": "trajectory-1:decision_step:004",
            "decision_step_id": 4,
            "history_event_step_ids": [1, 2, 3],
        }

        def summary_serializer(trajectory_value, decision_value):
            decision_input = build_decision_view_input(
                trajectory_value, decision_value
            )
            return "".join(
                event["low_fidelity_v2_serialized"]
                for event in decision_input["events"]
            ).encode("utf-8")

        result = validate_processor_byte_canary(
            trajectory,
            decision,
            serialize_request=summary_serializer,
        )
        self.assertTrue(result["included_history_action_canary_passed"])
        self.assertTrue(
            result["excluded_current_or_future_action_canary_passed"]
        )
        self.assertEqual(
            result[
                "excluded_current_or_future_action_canary_mutation_count"
            ],
            2,
        )

    def test_canary_denominators_cover_every_excluded_event(self) -> None:
        contract, _, _ = built_contract()
        loader = contract["data_projection"]["decision_view_loader"]
        self.assertEqual(loader["excluded_action_canary_state_count"], 128)
        self.assertEqual(loader["excluded_action_canary_mutation_count"], 192)
        self.assertEqual(loader["included_history_action_canary_state_count"], 192)
        self.assertEqual(
            loader["included_history_action_canary_mutation_count"], 192
        )
        self.assertTrue(
            loader["each_excluded_event_action_must_be_mutated_separately"]
        )
        self.assertTrue(loader["state_id_must_bind_source_id_and_decision_step"])
        self.assertTrue(
            loader[
                "request_manifest_validation_requires_actual_included_events"
            ]
        )
        self.assertTrue(
            loader["request_manifest_recomputes_included_event_payload_sha256"]
        )

    def test_two_workers_have_exact_disjoint_parity_shards(self) -> None:
        contract, _, _ = built_contract()
        workers = contract["execution"]["worker_topology"]["workers"]
        self.assertEqual(workers[0]["state_indices"], list(range(0, 192, 2)))
        self.assertEqual(workers[1]["state_indices"], list(range(1, 192, 2)))
        self.assertEqual(
            set(workers[0]["state_indices"]) | set(workers[1]["state_indices"]),
            set(range(192)),
        )
        self.assertFalse(
            set(workers[0]["state_indices"]) & set(workers[1]["state_indices"])
        )

    def test_exclusive_materialization_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            _exclusive_write(path, b"first")
            self.assertEqual(path.read_bytes(), b"first")
            with self.assertRaises(FileExistsError):
                _exclusive_write(path, b"second")


if __name__ == "__main__":
    unittest.main()
