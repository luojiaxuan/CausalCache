from __future__ import annotations

import copy
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2 import gui_owl_v2_action_to_androidworld
from causalcache.policy.gui_owl_v2_1 import parse_gui_owl_v2_1_output
from causalcache.restoration_v2_1_pilot_artifact import (
    AGGREGATE_FILENAME,
    ARCHIVE_LEDGER_NAME,
    ARCHIVE_MEMBER_PREFIX,
    CANONICAL_GLOBAL_LEDGER_PATH,
    CANONICAL_HF_PATH,
    CANONICAL_HF_REPO,
    CANONICAL_RAW_OUTPUT_DIR,
    EXPECTED_RUN_SOURCE_PATHS,
    INVALID_OUTCOME,
    NO_GO_OUTCOME,
    PASS_OUTCOME,
    PROCESSOR_ARTIFACT_PATH,
    RUN_MANIFEST_FILENAME,
    RUNTIME_IDENTITY_FILENAME,
    RUNTIME_METADATA_KEYS,
    V2_1_CONFIG_PATH,
    V2_CONFIG_PATH,
    SELECTION_MANIFEST_PATH,
    OCR_CONFIG_PATH,
    SNAPSHOT_MANIFEST_PATH,
    _canonical_attempt_identity,
    _fixed_gate,
    build_pilot_artifact_manifest,
    canonical_json_bytes,
    package_raw_pilot_evidence,
    pretty_json_bytes,
    read_extracted_pilot_evidence,
    read_pilot_evidence_archive,
    sha256_bytes,
    validate_pilot_artifact_manifest,
    validate_pilot_evidence_files,
    validate_source_x_run_contract,
)
from scripts.manage_restoration_v2_1_pilot_artifact import _build_parser
from scripts.run_restoration_v2_1_interface_pilot import (
    ATTEMPT_STATUS,
    GLOBAL_ATTEMPT_STATUS,
    NEGATIVE_OPERATION_COUNTS,
    PROTOCOL_ID,
    RUN_STATUS,
    SCHEMA_VERSION,
    STATE_OUTCOME_VALID,
    _base_state_record,
    _pilot_projection,
    aggregate_pilot_gate,
)


SOURCE_COMMIT = "a" * 40
HF_REVISION = "b" * 40
RUN_STARTED = "2026-07-15T00:00:00Z"
RUN_ENDED = "2026-07-15T00:00:30Z"
VALID_OUTPUT = (
    '<tool_call>\n{"name":"mobile_use","arguments":{"action":"wait"}}\n'
    "</tool_call>"
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _source_blobs(repository_root: Path) -> dict[str, bytes]:
    processor_artifact = {
        "schema_version": "0.1.0",
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "private_hf_processor_preflight_evidence",
        "hf_artifact": {
            "repo": "gavinlaw/causalcache-restoration-v2-1-processor-preflight-mobile",
            "repo_type": "dataset",
            "visibility": "private",
            "tag": "v2.1-processor-preflight-v1",
            "immutable_revision": "f" * 40,
            "path": "processor-preflight-v1/formal-result.json",
        },
        "compact_reduction": {
            "status": "PASSED_RESTORATION_V2_1_90_PROMPT_PROCESSOR_PREFLIGHT",
            "contract_sha256": sha256_bytes(
                (repository_root / V2_1_CONFIG_PATH).read_bytes()
            ),
            "selection_manifest_sha256": sha256_bytes(
                (repository_root / SELECTION_MANIFEST_PATH).read_bytes()
            ),
            "policy_interface_source_sha256": sha256_bytes(
                (
                    repository_root
                    / "code/causalcache/policy/gui_owl_v2_1.py"
                ).read_bytes()
            ),
            "runtime_source_sha256": sha256_bytes(
                (
                    repository_root
                    / "code/causalcache/policy/gui_owl_v2_1_runtime.py"
                ).read_bytes()
            ),
            "state_count": 45,
            "prompt_count": 90,
            "official_tools_injected_prompt_count": 90,
            "image_count_distribution": {"1": 45, "3": 15, "4": 15, "5": 15},
            "context_overflow_count": 0,
            "prompt_records_sha256": "1" * 64,
            "shape_records_sha256": "2" * 64,
            "teacher_golden_records_sha256": "3" * 64,
            "processor_classes_sha256": "5" * 64,
        },
        "raw_evidence": {
            "source_git_commit": SOURCE_COMMIT,
            "sha256": "4" * 64,
            "size_bytes": 12345,
        },
    }
    blobs: dict[str, bytes] = {}
    for relative in EXPECTED_RUN_SOURCE_PATHS:
        path = repository_root / relative
        if relative == PROCESSOR_ARTIFACT_PATH:
            blobs[relative] = pretty_json_bytes(processor_artifact)
        else:
            blobs[relative] = path.read_bytes()
    return blobs


def _projection(repository_root: Path) -> list[dict[str, Any]]:
    selection = _json(repository_root / SELECTION_MANIFEST_PATH)
    records = selection["roles"]["v2_development"]["states"]
    return [
        {
            "pilot_index": index,
            "source_state_index": 30 + index,
            "role": "v2_development",
            "trajectory_id": record["source_id"],
            "decision_step_id": record["decision_step_id"],
            "state_id": record["state_id"],
            "candidate_event_step_ids": record["candidate_event_step_ids"],
        }
        for index, record in enumerate(records)
    ]


def _runtime_metadata(repository_root: Path) -> dict[str, Any]:
    contract = _json(repository_root / V2_1_CONFIG_PATH)
    preflight = contract["processor_preflight"]
    return {
        "protocol_id": PROTOCOL_ID,
        "generation_interface": "processor_apply_chat_template_official_tools_kwarg",
        "official_tools_argument_count": 1,
        "official_tool_schema_sha256": preflight["canonical_tool_schema_sha256"],
        "chat_template_file_sha256": preflight["chat_template_file_sha256"],
        "chat_template_text_sha256": preflight["chat_template_text_sha256"],
        "assistant_prefix_token_ids": [151644, 77091, 198],
        "tool_call_open_token_id": 151657,
        "tool_call_close_token_id": 151658,
        "generation_eos_token_id": 151658,
        "suppressed_standard_eos_token_ids": [151645, 151643],
        "generation_pad_token_id": 151643,
        "generation_num_beams": 1,
        "generation_num_return_sequences": 1,
        "generation_standard_eos_suppression": (
            "negative_infinity_via_transformers_suppress_tokens"
        ),
        "host_injected_tool_call_closer": False,
        "output_recovery_or_normalization": False,
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
        "frozen": True,
        "max_pixels": 2_621_440,
        "min_pixels": 2_621_440,
        "model_class": "GUIOwlForConditionalGeneration",
        "model_dir": "/data/artifacts/models/GUI-Owl-1.5-8B-Instruct",
        "model_repo": "mPLUG/GUI-Owl-1.5-8B-Instruct",
        "model_revision": "06d5faecff74840bab2be2425e9c42667a5d04fc",
        "processor_class": "GUIOwlProcessor",
        "single_device": True,
        "snapshot_manifest_sha256": contract["primary_policy"]["snapshot_manifest"][
            "sha256"
        ],
        "target_effective_visual_tokens_per_image": 2560,
        "torch_version": "2.11.0+cu130",
        "transformers_source_sha256": "8" * 64,
        "transformers_version": "5.6.0",
        "verified_model_file_count": 10,
        "verified_model_total_bytes": 1_000_000,
    }


def _runtime_identity() -> dict[str, Any]:
    from scripts.run_restoration_v2_1_interface_pilot import (
        CANONICAL_PILOT_CONTAINER_ID,
        CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
        CANONICAL_PILOT_HOST_ALIAS,
        CANONICAL_PILOT_HOST_HOSTNAME,
    )

    uuid = "GPU-TEST-UUID"
    return {
        "declared_host": {
            "alias": CANONICAL_PILOT_HOST_ALIAS,
            "hostname": CANONICAL_PILOT_HOST_HOSTNAME,
            "verification": "requires_independent_host_preflight",
        },
        "verified_container_runtime": {
            "container_id": CANONICAL_PILOT_CONTAINER_ID,
            "container_hostname": CANONICAL_PILOT_CONTAINER_ID[:12],
            "verification": "live_hostname_prefix_of_full_container_id",
        },
        "declared_container_image": {
            "digest": CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
            "verification": "requires_independent_host_preflight",
        },
        "selected_device": "cuda:0",
        "live": {
            "platform_machine": "x86_64",
            "gpu_name": "NVIDIA H200",
            "gpu_uuid": uuid,
            "nvidia_smi_gpu_uuid": f"GPU-{uuid}",
            "nvidia_driver_version": "570.172.08",
            "gpu_compute_capability": [9, 0],
            "gpu_multiprocessor_count": 132,
            "visible_cuda_device_count": 1,
            "selected_device": "cuda:0",
            "python_version": "3.12.11",
            "torch_version": "2.11.0+cu130",
            "torch_cuda_build_version": "13.0",
            "cudnn_version": 91000,
            "transformers_version": "5.6.0",
        },
    }


def _execution_argv() -> list[str]:
    from scripts.run_restoration_v2_1_interface_pilot import (
        CANONICAL_PILOT_CONTAINER_ID,
        CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
        CANONICAL_PILOT_HOST_ALIAS,
        CANONICAL_PILOT_HOST_HOSTNAME,
    )

    values = {
        "--repository-root": "/data/CausalCache",
        "--contract": f"/data/CausalCache/{V2_1_CONFIG_PATH}",
        "--processor-preflight": "/data/evidence/processor.json",
        "--derived-artifact-root": "/data/derived/restoration-v2-v1",
        "--scientific-config": f"/data/CausalCache/{V2_CONFIG_PATH}",
        "--selection-manifest": f"/data/CausalCache/{SELECTION_MANIFEST_PATH}",
        "--ocr-backend-config": f"/data/CausalCache/{OCR_CONFIG_PATH}",
        "--model-dir": "/data/artifacts/models/GUI-Owl-1.5-8B-Instruct",
        "--device": "cuda:0",
        "--host-alias": CANONICAL_PILOT_HOST_ALIAS,
        "--host-hostname": CANONICAL_PILOT_HOST_HOSTNAME,
        "--container-id": CANONICAL_PILOT_CONTAINER_ID,
        "--container-image-digest": CANONICAL_PILOT_CONTAINER_IMAGE_DIGEST,
        "--output-dir": str(CANONICAL_RAW_OUTPUT_DIR),
    }
    argv = [
        "/usr/bin/python3",
        "/data/CausalCache/code/scripts/run_restoration_v2_1_interface_pilot.py",
    ]
    for flag, value in values.items():
        argv.extend([flag, value])
    return argv


def _run_contract(repository_root: Path, blobs: dict[str, bytes]) -> dict[str, Any]:
    v21_bytes = blobs[V2_1_CONFIG_PATH]
    v21 = json.loads(v21_bytes)
    processor_manifest = json.loads(blobs[PROCESSOR_ARTIFACT_PATH])
    canonical_inputs = {
        "contract": {
            "path": V2_1_CONFIG_PATH,
            "sha256": sha256_bytes(v21_bytes),
        },
        "processor_preflight": {
            "external_path": "/data/evidence/processor.json",
            "sha256": "4" * 64,
            "size_bytes": 12345,
            "artifact_manifest_path": PROCESSOR_ARTIFACT_PATH,
            "artifact_manifest_sha256": sha256_bytes(
                blobs[PROCESSOR_ARTIFACT_PATH]
            ),
            "artifact_manifest": processor_manifest,
        },
        "scientific_config": {
            "path": V2_CONFIG_PATH,
            "sha256": sha256_bytes(blobs[V2_CONFIG_PATH]),
        },
        "selection_manifest": {
            "path": SELECTION_MANIFEST_PATH,
            "sha256": sha256_bytes(blobs[SELECTION_MANIFEST_PATH]),
        },
        "ocr_backend_config": {
            "path": OCR_CONFIG_PATH,
            "sha256": sha256_bytes(blobs[OCR_CONFIG_PATH]),
        },
        "snapshot_manifest": {
            "path": SNAPSHOT_MANIFEST_PATH,
            "sha256": sha256_bytes(blobs[SNAPSHOT_MANIFEST_PATH]),
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_sha256": sha256_bytes(v21_bytes),
        "git_identity": {
            "branch": "main",
            "commit": SOURCE_COMMIT,
            "origin_main": SOURCE_COMMIT,
            "remote_main": SOURCE_COMMIT,
            "remote_url": "https://github.com/luojiaxuan/CausalCache.git",
            "worktree": "clean_including_untracked",
        },
        "source_inventory": [
            {
                "path": path,
                "sha256": sha256_bytes(blobs[path]),
                "git_commit": SOURCE_COMMIT,
            }
            for path in EXPECTED_RUN_SOURCE_PATHS
        ],
        "processor_audit": {
            "status": "PASSED_RESTORATION_V2_1_90_PROMPT_PROCESSOR_PREFLIGHT",
            "prompt_count": 90,
            "prompt_records_sha256": "1" * 64,
            "shape_records_sha256": "2" * 64,
            "teacher_golden_records_sha256": "3" * 64,
            "processor_classes_sha256": "5" * 64,
            "evidence_git_commit": SOURCE_COMMIT,
            "current_git_commit": SOURCE_COMMIT,
            "validation_mode": "reuse",
        },
        "canonical_inputs": canonical_inputs,
        "artifact": {
            "artifact_tree_sha256": (
                "475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e"
            ),
            "artifact_manifest_sha256": "6" * 64,
            "screening_manifest_sha256": "7" * 64,
            "derived_repo": dict(v21["data"]["derived_artifact"]),
        },
        "policy": {
            "repo": "mPLUG/GUI-Owl-1.5-8B-Instruct",
            "revision": "06d5faecff74840bab2be2425e9c42667a5d04fc",
            "model_dir": "/data/artifacts/models/GUI-Owl-1.5-8B-Instruct",
            "runtime_metadata_requirement": {
                "required_protocol_id": PROTOCOL_ID,
                "validated_after_claim_before_generation": True,
                "native_generation_metadata_persisted_per_state": True,
                "required_metadata_keys": sorted(RUNTIME_METADATA_KEYS),
                "model_dir": "/data/artifacts/models/GUI-Owl-1.5-8B-Instruct",
                "model_repo": "mPLUG/GUI-Owl-1.5-8B-Instruct",
                "model_revision": "06d5faecff74840bab2be2425e9c42667a5d04fc",
                "snapshot_manifest_sha256": sha256_bytes(
                    blobs[SNAPSHOT_MANIFEST_PATH]
                ),
                "device": "cuda:0",
                "target_effective_visual_tokens_per_image": 2560,
            },
        },
        "runtime_identity": _runtime_identity(),
        "execution_argv": _execution_argv(),
        "operational_argv_policy": {
            "resume_flag_excluded_from_scientific_run_identity": True,
            "initial_invocation_is_recorded_without_resume": True,
        },
        "seed_policy": {
            "decoding": "greedy_do_sample_false",
            "random_seed": None,
            "sampling_seed_not_applicable": True,
        },
        "output_dir": str(CANONICAL_RAW_OUTPUT_DIR),
        "attempt_identity": _canonical_attempt_identity(),
        "states": _projection(repository_root),
        "generation_plan": {
            "fidelity": "full_history_reference",
            "generation_calls_per_state": 1,
            "fixed_state_denominator": 15,
            "automatic_retry": False,
            "top_up": False,
        },
        "negative_operation_counts": dict(NEGATIVE_OPERATION_COUNTS),
        "confirm_state": "LOCKED",
    }


def _metadata(repository_root: Path) -> dict[str, Any]:
    runtime = _runtime_metadata(repository_root)
    return {
        "protocol_id": PROTOCOL_ID,
        "generation_interface": "processor_apply_chat_template_official_tools_kwarg",
        "official_tools_argument_count": 1,
        "tool_call_close_token_id": 151658,
        "generation_eos_token_id": 151658,
        "suppressed_standard_eos_token_ids": [151645, 151643],
        "generation_pad_token_id": 151643,
        "generation_num_beams": 1,
        "generation_num_return_sequences": 1,
        "host_injected_tool_call_closer": False,
        "output_recovery_or_normalization": False,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "return_dict_in_generate": False,
        "max_new_tokens": 256,
        "decoded_output_utf8_sha256": sha256_bytes(VALID_OUTPUT.encode("utf-8")),
        "generated_tool_call_close_token_count": 1,
        "final_generated_token_id": 151658,
        "termination_reason": "model_emitted_tool_call_close",
        "model_emitted_tool_call_close": True,
        **{
            key: runtime[key]
            for key in (
                "official_tool_schema_sha256",
                "chat_template_file_sha256",
                "chat_template_text_sha256",
                "assistant_prefix_token_ids",
                "tool_call_open_token_id",
                "generation_standard_eos_suppression",
            )
        },
    }


def _complete_files(repository_root: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    blobs = _source_blobs(repository_root)
    run_contract = _run_contract(repository_root, blobs)
    contract_sha = sha256_bytes(canonical_json_bytes(run_contract))
    states = []
    records = []
    parsed = parse_gui_owl_v2_1_output(VALID_OUTPUT)
    action = parsed.canonical_action.arguments()
    bridge = gui_owl_v2_action_to_androidworld(
        parsed.canonical_action,
        screen_width=1080,
        screen_height=1920,
    )
    files: dict[str, bytes] = {}
    for index, projection in enumerate(run_contract["states"]):
        from causalcache.data.restoration_v2_screening import ScreeningState

        state = ScreeningState(
            index=projection["source_state_index"],
            role=projection["role"],
            trajectory_id=projection["trajectory_id"],
            decision_step_id=projection["decision_step_id"],
            candidate_event_step_ids=tuple(projection["candidate_event_step_ids"]),
        )
        states.append(state)
        record = _base_state_record(
            projection=_pilot_projection(state, index),
            run_contract_sha256=contract_sha,
        )
        record.update(
            {
                "outcome": STATE_OUTCOME_VALID,
                "failure": None,
                "raw_output": VALID_OUTPUT,
                "generation_metadata": _metadata(repository_root),
                "canonical_action": action,
                "screen_dimensions": {"width": 1080, "height": 1920},
                "androidworld_bridge": bridge,
                "started_at_utc": "2026-07-15T00:00:01Z",
                "ended_at_utc": "2026-07-15T00:00:02Z",
                "duration_seconds": 1.0,
            }
        )
        records.append(record)
        files[f"states/{index:03d}.json"] = pretty_json_bytes(record)
        files[f"attempts/{index:03d}.json"] = pretty_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": ATTEMPT_STATUS,
                "run_contract_sha256": contract_sha,
                "state": projection,
                "attempt_ordinal": 1,
                "retry_count": 0,
                "top_up_count": 0,
                "created_at_utc": "2026-07-15T00:00:01Z",
            }
        )
    aggregate = aggregate_pilot_gate(
        records,
        expected_states=states,
        gate=_fixed_gate(),
        run_contract_sha256=contract_sha,
        started_at_utc=RUN_STARTED,
        ended_at_utc=RUN_ENDED,
    )
    files[RUN_MANIFEST_FILENAME] = pretty_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": RUN_STATUS,
            "run_contract_sha256": contract_sha,
            "run_contract": run_contract,
            "created_at_utc": RUN_STARTED,
        }
    )
    files[AGGREGATE_FILENAME] = pretty_json_bytes(aggregate)
    runtime_metadata = _runtime_metadata(repository_root)
    files[RUNTIME_IDENTITY_FILENAME] = pretty_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "VALIDATED_POLICY_RUNTIME_AFTER_DURABLE_CLAIM",
            "run_contract_sha256": contract_sha,
            "runtime_metadata_sha256": sha256_bytes(
                canonical_json_bytes(runtime_metadata)
            ),
            "runtime_metadata": runtime_metadata,
            "created_at_utc": "2026-07-15T00:00:01Z",
        }
    )
    files[ARCHIVE_LEDGER_NAME] = pretty_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": GLOBAL_ATTEMPT_STATUS,
            "attempt_identity": _canonical_attempt_identity(),
            "run_contract_sha256": contract_sha,
            "created_at_utc": RUN_STARTED,
        }
    )
    return files, blobs


def _write_raw_tree(root: Path, files: dict[str, bytes]) -> tuple[Path, Path]:
    run = root / "restoration-v2-1-interface-pilot-v1"
    (run / "states").mkdir(parents=True)
    (run / "attempts").mkdir()
    ledger = root / ".restoration-v2-1-interface-pilot-v1.attempt.json"
    ledger.write_bytes(files[ARCHIVE_LEDGER_NAME])
    for relative, payload in files.items():
        if relative == ARCHIVE_LEDGER_NAME:
            continue
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return run, ledger


class RestorationV21PilotArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository_root = Path(__file__).resolve().parents[2]

    def test_source_x_replay_rejects_denominator_policy_config_and_blob_tamper(self) -> None:
        files, blobs = _complete_files(self.repository_root)
        evidence = validate_pilot_evidence_files(files)

        def reader(root: Path, commit: str, relative: str) -> bytes:
            del root, commit
            return blobs[relative]

        validate_source_x_run_contract(
            evidence.run_contract,
            repository_root=self.repository_root,
            source_git_commit=SOURCE_COMMIT,
            committed_blob_reader=reader,
            ancestor_validator=lambda *_: None,
        )
        raw_binding = copy.deepcopy(evidence.run_contract)
        raw_binding["canonical_inputs"]["processor_preflight"]["size_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "raw evidence hash/size"):
            validate_source_x_run_contract(
                raw_binding,
                repository_root=self.repository_root,
                source_git_commit=SOURCE_COMMIT,
                committed_blob_reader=reader,
                ancestor_validator=lambda *_: None,
            )
        class_binding = copy.deepcopy(evidence.run_contract)
        class_binding["processor_audit"]["processor_classes_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "compact reduction"):
            validate_source_x_run_contract(
                class_binding,
                repository_root=self.repository_root,
                source_git_commit=SOURCE_COMMIT,
                committed_blob_reader=reader,
                ancestor_validator=lambda *_: None,
            )
        denominator = copy.deepcopy(evidence.run_contract)
        denominator["states"][0]["trajectory_id"] = "forged-trajectory"
        with self.assertRaisesRegex(ValueError, "frozen source-X projection"):
            validate_source_x_run_contract(
                denominator,
                repository_root=self.repository_root,
                source_git_commit=SOURCE_COMMIT,
                committed_blob_reader=reader,
                ancestor_validator=lambda *_: None,
            )
        policy = copy.deepcopy(files)
        manifest = json.loads(policy[RUN_MANIFEST_FILENAME])
        manifest["run_contract"]["policy"]["repo"] = "forged/policy"
        manifest["run_contract_sha256"] = sha256_bytes(
            canonical_json_bytes(manifest["run_contract"])
        )
        policy[RUN_MANIFEST_FILENAME] = pretty_json_bytes(manifest)
        with self.assertRaisesRegex(ValueError, "policy snapshot"):
            validate_pilot_evidence_files(policy)

        config_blobs = dict(blobs)
        config_blobs[V2_1_CONFIG_PATH] += b"\n"
        with self.assertRaisesRegex(ValueError, "source-X.*(config|blob)"):
            validate_source_x_run_contract(
                evidence.run_contract,
                repository_root=self.repository_root,
                source_git_commit=SOURCE_COMMIT,
                committed_blob_reader=lambda root, commit, relative: config_blobs[
                    relative
                ],
                ancestor_validator=lambda *_: None,
            )
        source_blobs = dict(blobs)
        source_blobs[EXPECTED_RUN_SOURCE_PATHS[0]] += b"tamper"
        with self.assertRaisesRegex(ValueError, "committed blob"):
            validate_source_x_run_contract(
                evidence.run_contract,
                repository_root=self.repository_root,
                source_git_commit=SOURCE_COMMIT,
                committed_blob_reader=lambda root, commit, relative: source_blobs[
                    relative
                ],
                ancestor_validator=lambda *_: None,
            )

    def test_archive_is_deterministic_and_includes_output_plus_sibling_ledger(self) -> None:
        files, _ = _complete_files(self.repository_root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, ledger = _write_raw_tree(root, files)
            archives = [root / "one.tar", root / "two.tar"]
            for archive in archives:
                package_raw_pilot_evidence(
                    repository_root=self.repository_root,
                    raw_output_dir=raw,
                    global_attempt_ledger=ledger,
                    output_archive=archive,
                    source_git_commit=SOURCE_COMMIT,
                    require_canonical_location=False,
                    git_identity_validator=lambda _: {"commit": SOURCE_COMMIT},
                    source_binding_validator=lambda *_, **__: None,
                )
            self.assertEqual(archives[0].read_bytes(), archives[1].read_bytes())
            with tarfile.open(archives[0], "r:") as archive:
                names = archive.getnames()
            self.assertIn(
                f"{ARCHIVE_MEMBER_PREFIX}/{ARCHIVE_LEDGER_NAME}",
                names,
            )
            self.assertIn(
                f"{ARCHIVE_MEMBER_PREFIX}/{RUN_MANIFEST_FILENAME}",
                names,
            )
            evidence = read_pilot_evidence_archive(archives[0])
            self.assertEqual(evidence.outcome, PASS_OUTCOME)
            self.assertEqual(len(evidence.inventory), 34)

    def test_archive_and_extracted_redownload_bind_immutable_hf_manifest(self) -> None:
        files, _ = _complete_files(self.repository_root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, ledger = _write_raw_tree(root, files)
            archive = root / "raw.tar"
            package_raw_pilot_evidence(
                repository_root=self.repository_root,
                raw_output_dir=raw,
                global_attempt_ledger=ledger,
                output_archive=archive,
                source_git_commit=SOURCE_COMMIT,
                require_canonical_location=False,
                git_identity_validator=lambda _: {"commit": SOURCE_COMMIT},
                source_binding_validator=lambda *_, **__: None,
            )
            manifest = build_pilot_artifact_manifest(
                repository_root=self.repository_root,
                raw_archive=archive,
                source_git_commit=SOURCE_COMMIT,
                hf_repo=CANONICAL_HF_REPO,
                hf_immutable_revision=HF_REVISION,
                hf_path=CANONICAL_HF_PATH,
                source_binding_validator=lambda *_, **__: None,
            )
            redownload = root / "redownload.tar"
            shutil.copyfile(archive, redownload)
            evidence = read_pilot_evidence_archive(redownload)
            result = validate_pilot_artifact_manifest(
                manifest,
                evidence=evidence,
                archive_path=redownload,
            )
            self.assertTrue(result["archive_hash_verified"])
            extracted = root / "extracted"
            extracted.mkdir()
            with tarfile.open(redownload, "r:") as source:
                source.extractall(extracted, filter="data")
            extracted_evidence = read_extracted_pilot_evidence(extracted)
            extracted_result = validate_pilot_artifact_manifest(
                manifest,
                evidence=extracted_evidence,
                archive_path=None,
            )
            self.assertFalse(extracted_result["archive_hash_verified"])
            self.assertEqual(
                manifest["source_execution"]["canonical_raw_output_dir"],
                str(CANONICAL_RAW_OUTPUT_DIR),
            )
            self.assertEqual(
                manifest["source_execution"]["canonical_global_attempt_ledger"],
                str(CANONICAL_GLOBAL_LEDGER_PATH),
            )
            with self.assertRaisesRegex(ValueError, "immutable 40-hex"):
                build_pilot_artifact_manifest(
                    repository_root=self.repository_root,
                    raw_archive=archive,
                    source_git_commit=SOURCE_COMMIT,
                    hf_repo=CANONICAL_HF_REPO,
                    hf_immutable_revision="c" * 39,
                    hf_path=CANONICAL_HF_PATH,
                    source_binding_validator=lambda *_, **__: None,
                )
            with self.assertRaisesRegex(ValueError, "unsafe|canonical"):
                build_pilot_artifact_manifest(
                    repository_root=self.repository_root,
                    raw_archive=archive,
                    source_git_commit=SOURCE_COMMIT,
                    hf_repo=CANONICAL_HF_REPO,
                    hf_immutable_revision=HF_REVISION,
                    hf_path="../raw.tar",
                    source_binding_validator=lambda *_, **__: None,
                )

    def test_aggregate_tamper_and_invalid_partial_inventory_fail_closed(self) -> None:
        files, _ = _complete_files(self.repository_root)
        tampered = copy.deepcopy(files)
        aggregate = json.loads(tampered[AGGREGATE_FILENAME])
        aggregate["metrics"]["exact_whole_output_parse_count"] = 14
        tampered[AGGREGATE_FILENAME] = pretty_json_bytes(aggregate)
        with self.assertRaisesRegex(ValueError, "independent reduction"):
            validate_pilot_evidence_files(tampered)
        missing_runtime = copy.deepcopy(files)
        del missing_runtime[RUNTIME_IDENTITY_FILENAME]
        with self.assertRaisesRegex(ValueError, "runtime identity|inventory"):
            validate_pilot_evidence_files(missing_runtime)

        invalid = copy.deepcopy(files)
        for name in list(invalid):
            if name.startswith("states/") and name >= "states/003.json":
                del invalid[name]
            if name.startswith("attempts/") and name >= "attempts/004.json":
                del invalid[name]
        contract_sha = json.loads(invalid[RUN_MANIFEST_FILENAME])[
            "run_contract_sha256"
        ]
        invalid[AGGREGATE_FILENAME] = pretty_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "run_contract_sha256": contract_sha,
                "status": "TERMINATED_INVALID_FIXED_15_STATE_INTERFACE_PILOT",
                "outcome": INVALID_OUTCOME,
                "gate_passed": False,
                "started_at_utc": RUN_STARTED,
                "ended_at_utc": "2026-07-15T00:00:10Z",
                "duration_seconds": 10.0,
                "invalid_failure": {
                    "stage": "state_003",
                    "category": "CONTRACT_OR_RUNTIME",
                    "exception_type": "RuntimeError",
                    "message": "fixed failure",
                },
                "completed_state_count": 3,
                "attempted_state_count": 4,
                "retry_count": 0,
                "top_up_count": 0,
                **NEGATIVE_OPERATION_COUNTS,
            }
        )
        evidence = validate_pilot_evidence_files(invalid)
        self.assertEqual(evidence.outcome, INVALID_OUTCOME)
        extra = copy.deepcopy(invalid)
        extra["states/003.json"] = files["states/003.json"]
        with self.assertRaisesRegex(ValueError, "INVALID.*inventory"):
            validate_pilot_evidence_files(extra)
        missing = copy.deepcopy(invalid)
        del missing["attempts/003.json"]
        with self.assertRaisesRegex(ValueError, "INVALID.*inventory"):
            validate_pilot_evidence_files(missing)

        pre_runtime = copy.deepcopy(files)
        for name in list(pre_runtime):
            if name.startswith("states/") or name.startswith("attempts/"):
                del pre_runtime[name]
        del pre_runtime[RUNTIME_IDENTITY_FILENAME]
        pre_runtime[AGGREGATE_FILENAME] = pretty_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "run_contract_sha256": contract_sha,
                "status": "TERMINATED_INVALID_FIXED_15_STATE_INTERFACE_PILOT",
                "outcome": INVALID_OUTCOME,
                "gate_passed": False,
                "started_at_utc": RUN_STARTED,
                "ended_at_utc": "2026-07-15T00:00:01Z",
                "duration_seconds": 1.0,
                "invalid_failure": {
                    "stage": "policy_runtime_initialization_after_durable_claim",
                    "category": "CONTRACT_OR_RUNTIME",
                    "exception_type": "RuntimeError",
                    "message": "constructor failed",
                },
                "completed_state_count": 0,
                "attempted_state_count": 0,
                "retry_count": 0,
                "top_up_count": 0,
                **NEGATIVE_OPERATION_COUNTS,
            }
        )
        recovered = validate_pilot_evidence_files(pre_runtime)
        self.assertEqual(recovered.outcome, INVALID_OUTCOME)

    def test_cli_help_exposes_archive_manifest_and_reuse_validation(self) -> None:
        help_text = _build_parser().format_help()
        self.assertIn("archive", help_text)
        self.assertIn("create-manifest", help_text)
        self.assertIn("validate", help_text)
        self.assertNotEqual(PASS_OUTCOME, NO_GO_OUTCOME)


if __name__ == "__main__":
    unittest.main()
