from __future__ import annotations

import itertools
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_eager_artifact import (
    canonical_json_bytes,
    expected_worker_specs,
    pretty_json_bytes,
)
from causalcache.restoration_v2_2_label_artifact import (
    AGGREGATE_FILENAME,
    GLOBAL_LEDGER_MEMBER,
    RUN_MANIFEST_FILENAME,
    SIBLING_PREFIX,
    _expected_attribution,
    _expected_edge_payloads,
    _expected_interactions,
    _expected_oracle,
    _EXPECTED_CANONICAL_INPUTS,
    _expected_aggregate_fields,
    build_artifact_manifest,
    deterministic_tar_bytes,
    expected_file_names,
    read_label_evidence_archive,
    sha256_bytes,
    validate_label_evidence_files,
)
from causalcache.restoration_v2_2_label_contract import (
    EXPECTED_SOURCE_PATHS,
    FROZEN_CONFIG_SHA256,
    PASS_OUTCOME,
    PROTOCOL_ID,
    V2_REPAIR_CONFIG_PATH,
    label_attempt_profile_for_config_path,
)
from causalcache.restoration_v2_2_label_table import validate_complete_distance_table
from tests.test_run_restoration_v2_2_eager_substrate import (
    fake_distance_audit,
    fake_runtime_metadata,
    fake_teacher_metadata,
)


SOURCE_COMMIT = "1" * 40
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LABEL_CONFIG = json.loads(
    (REPOSITORY_ROOT / "code/configs/causalcache_restoration_v2_2_labels.json").read_bytes()
)
SELECTION = json.loads(
    (REPOSITORY_ROOT / "data/manifests/restoration_v2_selection.json").read_bytes()
)
def _frozen_projections() -> tuple[dict[str, object], ...]:
    records = []
    for role in ("v2_label_train", "v2_development"):
        for state in SELECTION["roles"][role]["states"]:
            records.append(
                {
                    "index": len(records),
                    "role": role,
                    "trajectory_id": state["source_id"],
                    "decision_step_id": state["decision_step_id"],
                    "state_id": state["state_id"],
                    "candidate_event_step_ids": state["candidate_event_step_ids"],
                }
            )
    return tuple(records)


PROJECTIONS = _frozen_projections()


def _projection(index: int) -> dict[str, object]:
    return dict(PROJECTIONS[index])


def _operation_counts(event_count: int) -> dict[str, int]:
    rows = 1 << event_count
    return {
        "generation_call_count": 0,
        "reference_teacher_forward_count": 1,
        "reference_repeat_teacher_forward_count": 1,
        "non_reference_coalition_teacher_forward_count": rows - 1,
        "teacher_forward_call_count": rows + 1,
        "teacher_forward_example_count": rows + 1,
        "coalition_kl_measurement_count": rows - 1,
        "reference_repeat_kl_measurement_count": 1,
        "kl_measurement_count": rows,
        "raw_distance_row_count": rows,
        "deployment_conditional_label_count": {2: 4, 3: 9, 4: 16}[event_count],
        "full_hypercube_edge_count": event_count * (1 << (event_count - 1)),
        "pair_interaction_count": {2: 1, 3: 6, 4: 24}[event_count],
        "exact_permutation_attribution_count": event_count,
        "primary_exact_subset_oracle_count": 1,
        "retry_count": 0,
        "top_up_count": 0,
        "confirm_state_access_count": 0,
        "confirm_processor_prompt_count": 0,
        "confirm_decoder_input_count": 0,
        "confirm_generation_count": 0,
        "confirm_teacher_forward_count": 0,
        "expert_action_read_count": 0,
        "gate_training_example_count": 0,
        "gate_model_forward_count": 0,
        "gate_selection_count": 0,
        "matched_nll_evaluation_count": 0,
        "closed_loop_episode_count": 0,
    }


def _prompt_inventory(
    projection: dict[str, object],
    coalition: tuple[int, ...],
) -> dict[str, object]:
    event_ids = tuple(projection["candidate_event_step_ids"])
    mask = sum(
        1 << bit
        for bit, event_id in enumerate(event_ids)
        if event_id in coalition
    )
    fidelity = (
        "summary_only"
        if not coalition
        else "full_reference"
        if coalition == event_ids
        else "mixed_fidelity"
    )
    return {
        "schema_version": "1.0.0",
        "state_id": projection["state_id"],
        "role": projection["role"],
        "trajectory_id": projection["trajectory_id"],
        "decision_step_id": projection["decision_step_id"],
        "candidate_event_step_ids": list(event_ids),
        "coalition_mask": mask,
        "fidelity": fidelity,
        "high_fidelity_restored_event_step_ids": list(coalition),
        "high_fidelity_image_count": len(coalition),
        "current_observation_image_count": 1,
        "image_count": 1 + len(coalition),
        "text_block_count": 2,
        "policy_visible_text_sha256": "c" * 64,
        "user_block_types": ["text", *("image" for _ in range(1 + len(coalition)))],
    }


def _teacher_metadata(device: str, image_count: int) -> dict[str, object]:
    teacher = fake_teacher_metadata(device)
    effective_visual_tokens = image_count
    policy_visible_text_tokens = 10
    prompt_input_tokens = effective_visual_tokens + policy_visible_text_tokens
    distance_action_tokens = int(teacher["logits_to_keep"])
    teacher["samples"] = [
        {
            "image_count": image_count,
            "image_grid_thw": [[1, 2, 2] for _ in range(image_count)],
            "effective_visual_tokens": effective_visual_tokens,
            "policy_visible_text_tokens": policy_visible_text_tokens,
            "prompt_input_tokens": prompt_input_tokens,
            "teacher_carrier_tokens": 0,
            "distance_action_tokens": distance_action_tokens,
            "model_input_tokens": prompt_input_tokens + distance_action_tokens - 1,
            "distance_logit_positions": list(
                range(
                    prompt_input_tokens - 1,
                    prompt_input_tokens - 1 + distance_action_tokens,
                )
            ),
        }
    ]
    return teacher


def _state_record(index: int, run_sha: str) -> dict[str, object]:
    projection = _projection(index)
    event_ids = tuple(projection["candidate_event_step_ids"])
    distances = {
        tuple(subset): float(len(event_ids) - len(subset))
        for size in range(len(event_ids) + 1)
        for subset in itertools.combinations(event_ids, size)
    }
    table = validate_complete_distance_table(event_ids, distances)
    spec = expected_worker_specs()[index % 2]
    reference_teacher = _teacher_metadata(spec.device, 1 + len(event_ids))
    rows = [
        {
            "coalition_mask": sum(
                1 << bit
                for bit, event_id in enumerate(event_ids)
                if event_id in row.coalition
            ),
            "coalition_event_step_ids": list(row.coalition),
            "coalition_slot_cost": len(row.coalition),
            "distance_kl": row.distance,
            "restoration_utility": row.utility,
            "prompt_inventory": _prompt_inventory(projection, row.coalition),
            "teacher_metadata": _teacher_metadata(
                spec.device,
                1 + len(row.coalition),
            ),
            "distance_audit": (
                {
                    "reference_identity_distance": True,
                    "full_tensor_host_transfers": 0,
                }
                if row.coalition == event_ids
                else fake_distance_audit(spec.device)
            ),
        }
        for row in table.rows
    ]
    canonical_action = {"action": "wait"}
    canonical_action_sha256 = sha256_bytes(canonical_json_bytes(canonical_action))
    return {
        "schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "run_contract_sha256": run_sha,
        "status": "VALID_RESTORATION_V2_2_EAGER_LABEL_STATE",
        "state": projection,
        "parent_reference": {
            "member_name": f"workers/{'even' if index % 2 == 0 else 'odd'}/states/{index:03d}.json",
            "member_sha256": "a" * 64,
            "canonical_action": canonical_action,
            "canonical_action_sha256": canonical_action_sha256,
            "reference_mode": "parent_canonical_action_frozen_fresh_teacher_recompute",
            "new_reference_generation_count": 0,
        },
        "reference_teacher_metadata": reference_teacher,
        "reference_repeat_teacher_metadata": reference_teacher,
        "reference_repeat_kl": 0.0,
        "reference_repeat_distance_audit": fake_distance_audit(spec.device),
        "baseline_summary_only_distance": table.distance(()),
        "distance_rows": rows,
        "deployment_conditional_edges": _expected_edge_payloads(table, deployment=True),
        "full_hypercube_edges": _expected_edge_payloads(table, deployment=False),
        "pair_interactions": _expected_interactions(table),
        "exact_permutation_attribution": _expected_attribution(table),
        "primary_exact_subset_oracle": _expected_oracle(table),
        "operation_counts": _operation_counts(len(event_ids)),
        "started_at_utc": "2026-07-16T00:00:00Z",
        "ended_at_utc": "2026-07-16T00:00:01Z",
        "duration_seconds": 1.0,
    }


def _files() -> dict[str, bytes]:
    projections = [_projection(index) for index in range(45)]
    canonical_action_sha256 = sha256_bytes(
        canonical_json_bytes({"action": "wait"})
    )
    parents = [
        {
            "index": index,
            "state_id": projection["state_id"],
            "member_name": f"workers/{'even' if index % 2 == 0 else 'odd'}/states/{index:03d}.json",
            "member_sha256": "a" * 64,
            "canonical_action_sha256": canonical_action_sha256,
        }
        for index, projection in enumerate(projections)
    ]
    run_contract = {
        "schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "contract_source": {
            "path": "code/configs/causalcache_restoration_v2_2_labels.json",
            "sha256": FROZEN_CONFIG_SHA256,
        },
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
                "sha256": FROZEN_CONFIG_SHA256 if index == 0 else "d" * 64,
                "git_commit": SOURCE_COMMIT,
            }
            for index, path in enumerate(EXPECTED_SOURCE_PATHS)
        ],
        "parent_evidence": {
            "path": "/data/tmp/restoration-v2-2-parent.raw.tar",
            "sha256": "b22827e6e2d8d33b03686fc177dc8f9c55c5133470fbe40f9fb3e33cce809fb5",
            "size_bytes": 1_269_760,
            "validation_status": "VALIDATED_IMMUTABLE_V2_2_EAGER_PARENT_ARCHIVE",
        },
        "canonical_inputs": _EXPECTED_CANONICAL_INPUTS,
        "states": projections,
        "parent_states": parents,
        "runtime_requirements": LABEL_CONFIG["runtime"],
        "worker_topology": LABEL_CONFIG["worker_topology"],
        "enumeration": LABEL_CONFIG["enumeration"],
        "reduction": LABEL_CONFIG["reduction"],
        "operation_schedule": LABEL_CONFIG["operation_schedule"],
        "prohibited_work": LABEL_CONFIG["prohibited_work"],
        "attempt_identity": {
            "attempt_id": "restoration-v2-2-eager-labels-v1",
            "output_dir": "/data/experiments/causalcache/restoration-v2-2-eager-labels-v1",
            "global_ledger": (
                "/data/experiments/causalcache/"
                ".restoration-v2-2-eager-labels-v1.attempt.json"
            ),
            "host_alias": "hyper00",
            "host_hostname": "node-radixark-16-0000",
            "container_id": "e" * 64,
            "container_image_digest": (
                "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
            ),
        },
        "execution_argv": ["--contract", "configs/causalcache_restoration_v2_2_labels.json"],
    }
    run_sha = sha256_bytes(canonical_json_bytes(run_contract))
    records = [_state_record(index, run_sha) for index in range(45)]
    totals: dict[str, int] = {}
    for record in records:
        for key, value in record["operation_counts"].items():
            totals[key] = totals.get(key, 0) + value
    aggregate_fields = _expected_aggregate_fields(records)
    files = {
        RUN_MANIFEST_FILENAME: pretty_json_bytes(
            {
                "schema_version": "1.0.0",
                "protocol_id": PROTOCOL_ID,
                "status": "RESTORATION_V2_2_EAGER_LABEL_ATTEMPT",
                "run_contract_sha256": run_sha,
                "run_contract": run_contract,
            }
        ),
        GLOBAL_LEDGER_MEMBER: pretty_json_bytes(
            {
                "schema_version": "1.0.0",
                "protocol_id": PROTOCOL_ID,
                "status": "LABEL_ATTEMPT_COMPLETED",
                "attempt_id": "restoration-v2-2-eager-labels-v1",
                "run_contract_sha256": run_sha,
                "outcome": PASS_OUTCOME,
                "attempted_state_count": 45,
                "completed_state_count": 45,
                "worker_sibling_ledgers": {
                    "even": "/data/experiments/causalcache/labels.even.json",
                    "odd": "/data/experiments/causalcache/labels.odd.json",
                },
                "retry_count": 0,
                "top_up_count": 0,
            }
        ),
        AGGREGATE_FILENAME: pretty_json_bytes(
            {
                "schema_version": "1.0.0",
                "protocol_id": PROTOCOL_ID,
                "status": "COMPLETED_RESTORATION_V2_2_EAGER_LABELS_V1",
                "run_contract_sha256": run_sha,
                "outcome": PASS_OUTCOME,
                "fixed_state_denominator": 45,
                "operation_counts": dict(sorted(totals.items())),
                **aggregate_fields,
                "confirm_role_used": False,
                "gate_training_performed": False,
                "matched_nll_evaluation_performed": False,
                "closed_loop_evaluation_performed": False,
            }
        ),
    }
    for spec in expected_worker_specs():
        worker = spec.worker_id
        parity = spec.index_parity
        base = f"workers/{worker}"
        indices = list(spec.state_indices)
        ledger = {
            "schema_version": "1.0.0",
            "protocol_id": PROTOCOL_ID,
            "status": "LABEL_WORKER_COMPLETED",
            "run_contract_sha256": run_sha,
            "worker": spec.to_dict(),
            "attempted_state_indices": indices,
            "completed_state_indices": indices,
            "retry_count": 0,
            "top_up_count": 0,
        }
        files[f"{SIBLING_PREFIX}/{worker}.json"] = pretty_json_bytes(ledger)
        files[f"{base}/worker_attempt_ledger.json"] = pretty_json_bytes(ledger)
        files[f"{base}/runtime_identity.json"] = pretty_json_bytes(
            {
                "schema_version": "1.0.0",
                "protocol_id": PROTOCOL_ID,
                "run_contract_sha256": run_sha,
                "worker": spec.to_dict(),
                "runtime_metadata": fake_runtime_metadata(spec),
            }
        )
        files[f"{base}/terminal.json"] = pretty_json_bytes(
            {
                "schema_version": "1.0.0",
                "protocol_id": PROTOCOL_ID,
                "run_contract_sha256": run_sha,
                "worker": spec.to_dict(),
                "outcome": "PASS_RESTORATION_V2_2_EAGER_LABEL_WORKER",
                "attempted_state_indices": indices,
                "completed_state_indices": indices,
                "failure": None,
            }
        )
        for index in range(parity, 45, 2):
            files[f"{base}/attempts/{index:03d}.json"] = pretty_json_bytes(
                {
                    "schema_version": "1.0.0",
                    "protocol_id": PROTOCOL_ID,
                    "status": "LABEL_STATE_CLAIMED_NO_RETRY_OR_TOP_UP",
                    "run_contract_sha256": run_sha,
                    "worker": spec.to_dict(),
                    "state_index": index,
                    "retry_count": 0,
                    "top_up_count": 0,
                }
            )
            files[f"{base}/states/{index:03d}.json"] = pretty_json_bytes(records[index])
    return files


def _v2_repair_files() -> dict[str, bytes]:
    profile = label_attempt_profile_for_config_path(V2_REPAIR_CONFIG_PATH)
    files = _files()
    manifest = json.loads(files[RUN_MANIFEST_FILENAME])
    run_contract = manifest["run_contract"]
    run_contract["contract_source"] = {
        "path": profile.config_path,
        "sha256": profile.frozen_config_sha256,
    }
    run_contract["source_inventory"] = [
        {
            "path": path,
            "sha256": profile.frozen_config_sha256 if index == 0 else "d" * 64,
            "git_commit": SOURCE_COMMIT,
        }
        for index, path in enumerate(profile.expected_source_paths)
    ]
    run_contract["canonical_inputs"]["model_snapshot_preclaim"] = {
        "model_dir": str(profile.canonical_model_dir),
        "model_repo": "mPLUG/GUI-Owl-1.5-8B-Instruct",
        "model_revision": "06d5faecff74840bab2be2425e9c42667a5d04fc",
        "snapshot_manifest_sha256": (
            "50b675ec31c5c46dbb0d44c137a808fffb9d054916d39b596648d4eb9df7cbc3"
        ),
        "verified_model_file_count": 14,
        "verified_model_total_bytes": 17_545_907_171,
        "validation_status": "VALIDATED_FULL_MODEL_SNAPSHOT_BEFORE_GLOBAL_CLAIM",
    }
    run_contract["attempt_identity"] = {
        "attempt_id": profile.attempt_id,
        "attempt_revision": profile.attempt_revision,
        "supersedes_attempt_id": profile.supersedes_attempt_id,
        "pass_outcome": profile.pass_outcome,
        "invalid_outcome": profile.invalid_outcome,
        "aggregate_status": profile.aggregate_status,
        "output_dir": str(profile.output_dir),
        "global_ledger": str(profile.ledger_path),
        "raw_archive": str(profile.archive_path),
        "hf_repo": profile.hf_repo,
        "hf_tag": profile.hf_tag,
        "hf_path": profile.hf_path,
        "host_alias": "hyper00",
        "host_hostname": "node-radixark-16-0000",
        "container_id": "e" * 64,
        "container_image_digest": (
            "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
        ),
    }
    run_sha = sha256_bytes(canonical_json_bytes(run_contract))
    manifest["run_contract_sha256"] = run_sha
    manifest["attempt_id"] = profile.attempt_id
    manifest["attempt_revision"] = profile.attempt_revision
    files[RUN_MANIFEST_FILENAME] = pretty_json_bytes(manifest)
    for name, payload in list(files.items()):
        if name == RUN_MANIFEST_FILENAME:
            continue
        value = json.loads(payload)
        value["attempt_id"] = profile.attempt_id
        value["attempt_revision"] = profile.attempt_revision
        if "run_contract_sha256" in value:
            value["run_contract_sha256"] = run_sha
        if name == GLOBAL_LEDGER_MEMBER:
            value["outcome"] = profile.pass_outcome
        elif name == AGGREGATE_FILENAME:
            value["status"] = profile.aggregate_status
            value["outcome"] = profile.pass_outcome
        files[name] = pretty_json_bytes(value)
    return files


class RestorationV22LabelArtifactTest(unittest.TestCase):
    def test_full_raw_inventory_recomputes_all_derived_labels(self) -> None:
        files = _files()
        evidence = validate_label_evidence_files(files)
        self.assertEqual(set(files), expected_file_names())
        self.assertEqual(len(files), 101)
        self.assertEqual(evidence.outcome, PASS_OUTCOME)
        self.assertEqual(evidence.aggregate["operation_counts"]["raw_distance_row_count"], 420)
        self.assertEqual(
            evidence.aggregate["operation_counts"][
                "deployment_conditional_label_count"
            ],
            435,
        )

    def test_tampered_derived_edge_and_missing_member_fail_closed(self) -> None:
        files = _files()
        state_name = "workers/even/states/000.json"
        state = json.loads(files[state_name])
        state["deployment_conditional_edges"][0]["marginal_gain"] += 1.0
        files[state_name] = pretty_json_bytes(state)
        with self.assertRaisesRegex(ValueError, "derived scientific"):
            validate_label_evidence_files(files)

        incomplete = _files()
        del incomplete[state_name]
        with self.assertRaisesRegex(ValueError, "inventory"):
            validate_label_evidence_files(incomplete)

    def test_tampered_action_and_worker_lifecycle_fail_closed(self) -> None:
        files = _files()
        state_name = "workers/even/states/000.json"
        state = json.loads(files[state_name])
        state["parent_reference"]["canonical_action"] = {"action": "terminate"}
        files[state_name] = pretty_json_bytes(state)
        with self.assertRaisesRegex(ValueError, "canonical action hash"):
            validate_label_evidence_files(files)

        files = _files()
        marker_name = "workers/odd/attempts/001.json"
        marker = json.loads(files[marker_name])
        marker["retry_count"] = 1
        files[marker_name] = pretty_json_bytes(marker)
        with self.assertRaisesRegex(ValueError, "attempt marker"):
            validate_label_evidence_files(files)

        files = _files()
        runtime_name = "workers/even/runtime_identity.json"
        runtime = json.loads(files[runtime_name])
        runtime["runtime_metadata"]["device"] = "cuda:1"
        files[runtime_name] = pretty_json_bytes(runtime)
        with self.assertRaisesRegex(ValueError, "runtime metadata"):
            validate_label_evidence_files(files)

    def test_run_projection_prompt_and_forbidden_work_are_bound(self) -> None:
        files = _files()
        manifest = json.loads(files[RUN_MANIFEST_FILENAME])
        manifest["run_contract"]["states"][0]["role"] = "v2_confirm_primary"
        manifest["run_contract_sha256"] = sha256_bytes(
            canonical_json_bytes(manifest["run_contract"])
        )
        files[RUN_MANIFEST_FILENAME] = pretty_json_bytes(manifest)
        with self.assertRaisesRegex(ValueError, "state projection"):
            validate_label_evidence_files(files)

        files = _files()
        state_name = "workers/even/states/000.json"
        state = json.loads(files[state_name])
        state["distance_rows"][0]["prompt_inventory"]["coalition_mask"] = 1
        files[state_name] = pretty_json_bytes(state)
        with self.assertRaisesRegex(ValueError, "prompt inventory"):
            validate_label_evidence_files(files)

        files = _files()
        state = json.loads(files[state_name])
        state["distance_rows"][0]["distance_audit"]["full_tensor_host_transfers"] = 1
        files[state_name] = pretty_json_bytes(state)
        with self.assertRaisesRegex(ValueError, "distance audit"):
            validate_label_evidence_files(files)

        files = _files()
        state = json.loads(files[state_name])
        state["distance_rows"][0]["teacher_metadata"]["samples"][0][
            "image_count"
        ] = 99
        files[state_name] = pretty_json_bytes(state)
        with self.assertRaisesRegex(ValueError, "teacher sample image"):
            validate_label_evidence_files(files)

        files = _files()
        state = json.loads(files[state_name])
        state["operation_counts"]["confirm_generation_count"] = 1
        files[state_name] = pretty_json_bytes(state)
        aggregate = json.loads(files[AGGREGATE_FILENAME])
        aggregate["operation_counts"]["confirm_generation_count"] = 1
        files[AGGREGATE_FILENAME] = pretty_json_bytes(aggregate)
        with self.assertRaisesRegex(ValueError, "operation counts drifted"):
            validate_label_evidence_files(files)

    def test_aggregate_scientific_reduction_is_recomputed(self) -> None:
        files = _files()
        aggregate = json.loads(files[AGGREGATE_FILENAME])
        aggregate["scientific_summary"]["positive_pair_interaction_count"] += 1
        files[AGGREGATE_FILENAME] = pretty_json_bytes(aggregate)
        with self.assertRaisesRegex(ValueError, "scientific reduction"):
            validate_label_evidence_files(files)

    def test_canonical_archive_and_fresh_manifest_are_byte_bound(self) -> None:
        files = _files()
        payload = deterministic_tar_bytes(files)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.tar"
            fresh = Path(directory) / "fresh.tar"
            source.write_bytes(payload)
            fresh.write_bytes(payload)
            evidence = read_label_evidence_archive(source)
            manifest = build_artifact_manifest(
                source_archive=source,
                fresh_immutable_archive=fresh,
                hf_revision="2" * 40,
            )
            self.assertEqual(evidence.files, files)
            self.assertEqual(manifest["raw_archive"]["file_count"], 101)
            fresh.write_bytes(payload + b"tampered")
            with self.assertRaises((ValueError, tarfile.ReadError, EOFError)):
                build_artifact_manifest(
                    source_archive=source,
                    fresh_immutable_archive=fresh,
                    hf_revision="2" * 40,
                )

    def test_v2_repair_identity_survives_validation_archive_and_manifest(self) -> None:
        profile = label_attempt_profile_for_config_path(V2_REPAIR_CONFIG_PATH)
        files = _v2_repair_files()
        evidence = validate_label_evidence_files(files)
        self.assertEqual(evidence.profile, profile)
        self.assertEqual(evidence.outcome, profile.pass_outcome)
        payload = deterministic_tar_bytes(files, profile=profile)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.tar"
            fresh = Path(directory) / "fresh.tar"
            source.write_bytes(payload)
            fresh.write_bytes(payload)
            reread = read_label_evidence_archive(source)
            manifest = build_artifact_manifest(
                source_archive=source,
                fresh_immutable_archive=fresh,
                hf_revision="3" * 40,
            )
        self.assertEqual(reread.profile.attempt_id, profile.attempt_id)
        self.assertEqual(manifest["attempt_id"], profile.attempt_id)
        self.assertEqual(manifest["attempt_revision"], profile.attempt_revision)
        self.assertEqual(manifest["result"]["outcome"], profile.pass_outcome)
        self.assertEqual(manifest["hf_artifact"]["path"], profile.hf_path)


if __name__ == "__main__":
    unittest.main()
