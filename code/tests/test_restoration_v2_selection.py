import copy
import hashlib
import json
import unittest
from pathlib import Path

from causalcache.data.guiodyssey_independent import (
    Candidate,
    canonical_json_bytes,
    trajectory_selection_sha256,
)
from causalcache.data.restoration_v2_selection import (
    PoolReconstruction,
    build_exposure_ledger,
    build_selection_manifest,
    pretty_json_bytes,
    reduce_exposure_events,
    sha256_bytes,
    state_id,
    validate_exposure_ledger,
    validate_selection_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "configs" / "independent_reference_gate_v1.json"
V2_CONFIG = ROOT / "configs" / "causalcache_restoration_v2.json"
SOURCE_MANIFEST = (
    ROOT.parent / "data" / "manifests" / "independent_reference_gate_v1_source_files.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _candidate(source_id: str, index: int, app: str) -> Candidate:
    salt = _load(V1_CONFIG)["selection"]["trajectory_salt"]
    return Candidate(
        source_id=source_id,
        transport_file="mobile/use/train/fake.parquet",
        transport_row_index=index,
        selection_sha256=trajectory_selection_sha256(source_id, salt=salt),
        decision_count=6,
        normalized_app_labels=(app,),
        action_type_counts=(("swipe", 1), ("tap", 4), ("type_text", 1)),
    )


def _fixture(*, confirm_apps: tuple[str, ...] | None = None) -> tuple:
    v1 = _load(V1_CONFIG)
    v2 = copy.deepcopy(_load(V2_CONFIG))
    roles = v2["data"]["roles"]
    historical_ids = (
        roles["v1_reference_contract_audit_only"]["source_ids"]
        + roles["v2_label_train"]["source_ids"]
        + roles["v2_development"]["source_ids"]
    )
    salt = v1["selection"]["trajectory_salt"]
    historical_hashes = [
        trajectory_selection_sha256(source_id, salt=salt)
        for source_id in historical_ids
    ]
    if historical_hashes != sorted(historical_hashes):
        raise AssertionError("frozen historical IDs are not in their expected order")
    threshold = historical_hashes[-1]
    generated = []
    counter = 0
    while len(generated) < 88:
        source_id = f"synthetic-confirm-{counter:05d}"
        digest = trajectory_selection_sha256(source_id, salt=salt)
        if digest > threshold:
            generated.append((digest, source_id))
        counter += 1
    generated.sort()
    if confirm_apps is None:
        confirm_apps = tuple(f"app-{index % 3}" for index in range(20))

    candidates = []
    for index, source_id in enumerate(historical_ids):
        candidates.append(_candidate(source_id, index, f"historical-app-{index % 3}"))
    for offset, (_, source_id) in enumerate(generated):
        app = confirm_apps[offset] if offset < len(confirm_apps) else f"tail-app-{offset}"
        candidates.append(_candidate(source_id, len(candidates), app))
    candidates = sorted(candidates, key=lambda candidate: candidate.selection_sha256)
    pool_sha = sha256_bytes(
        canonical_json_bytes([candidate.pool_record() for candidate in candidates])
    )
    parent_sha = "a" * 64
    source_manifest_sha = hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest()
    v2["data"]["parent_artifact"]["manifest_sha256"] = parent_sha
    v2["data"]["parent_artifact"]["eligible_pool_sha256"] = pool_sha

    by_id = {candidate.source_id: candidate for candidate in candidates}
    parent_trajectories = []
    for source_id in historical_ids:
        candidate = by_id[source_id]
        parent_trajectories.append(
            {
                "source_id": source_id,
                "transport_file": candidate.transport_file,
                "transport_row_index": candidate.transport_row_index,
                "selection_sha256": candidate.selection_sha256,
                "normalized_app_labels": list(candidate.normalized_app_labels),
                "decisions": [{} for _ in range(candidate.decision_count)],
            }
        )
    reference_ids = roles["v1_reference_contract_audit_only"]["source_ids"]
    oracle_ids = roles["v2_label_train"]["source_ids"] + roles["v2_development"][
        "source_ids"
    ]
    parent = {
        "dataset_repo": v2["data"]["parent_artifact"]["repo"],
        "source": {
            "protocol_config_sha256": (
                "40b00e0a8da9e4f573ffcea2447c5e569c795c1e980c5889cb1ec486822b2a95"
            ),
            "source_file_manifest_sha256": source_manifest_sha,
        },
        "selection": {
            "eligible_pool_sha256": pool_sha,
            "eligible_pool_count": len(candidates),
            "total_source_rows": 212,
            "exclusion_counts": {
                "decision_count_above_maximum": 81,
                "decision_count_below_minimum": 1,
                "excluded_source_id": 1,
                "invalid_executable_action": 6,
                "terminal_status_mismatch": 12,
            },
        },
        "splits": {
            "reference_gate": {"source_ids": reference_ids},
            "oracle_pilot": {"source_ids": oracle_ids},
        },
        "trajectories": parent_trajectories,
    }
    source_manifest = _load(SOURCE_MANIFEST)
    reconstruction = PoolReconstruction(
        candidates=tuple(reversed(candidates)),
        total_source_rows=212,
        source_row_counts={
            record["path"]: 0 for record in source_manifest["files"]
        },
        exclusion_counts={
            "decision_count_above_maximum": 81,
            "decision_count_below_minimum": 1,
            "excluded_source_id": 1,
            "invalid_executable_action": 6,
            "terminal_status_mismatch": 12,
        },
    )
    return (
        reconstruction,
        v1,
        v2,
        parent,
        parent_sha,
        source_manifest,
        source_manifest_sha,
    )


def _build(*, confirm_apps: tuple[str, ...] | None = None) -> dict:
    (
        reconstruction,
        v1,
        v2,
        parent,
        parent_sha,
        source_manifest,
        source_manifest_sha,
    ) = _fixture(confirm_apps=confirm_apps)
    return build_selection_manifest(
        reconstruction=reconstruction,
        v1_config=v1,
        v1_config_sha256=hashlib.sha256(V1_CONFIG.read_bytes()).hexdigest(),
        v2_contract=v2,
        v2_contract_sha256=hashlib.sha256(V2_CONFIG.read_bytes()).hexdigest(),
        parent_manifest=parent,
        parent_manifest_sha256=parent_sha,
        source_file_manifest=source_manifest,
        source_file_manifest_sha256=source_manifest_sha,
        generator={
            "git_revision": "d" * 40,
            "module_path": "code/causalcache/data/restoration_v2_selection.py",
            "module_sha256": "e" * 64,
            "cli_path": "code/scripts/materialize_restoration_v2_selection.py",
            "cli_sha256": "f" * 64,
            "validator_path": "code/scripts/validate_restoration_v2_selection.py",
            "validator_sha256": "1" * 64,
        },
    )


def _selection_with_content_witnesses() -> dict:
    selection = _build()
    for role_name in ("v2_label_train", "v2_development", "v2_confirm_primary"):
        for trajectory in selection["roles"][role_name]["trajectories"]:
            trajectory["instruction_sha256"] = "1" * 64
        for state in selection["roles"][role_name]["states"]:
            current = {
                "member_path": (
                    f"images/{state['source_id']}/observation-"
                    f"{state['decision_step_id'] - 1:03d}.png"
                ),
                "sha256": "2" * 64,
            }
            state["current_observation"] = current
            state["candidate_event_post_states"] = [
                {
                    "event_step_id": step,
                    "post_state_member_path": (
                        f"images/{state['source_id']}/observation-{step:03d}.png"
                    ),
                    "post_state_sha256": "3" * 64,
                }
                for step in state["candidate_event_step_ids"]
            ]
            state["current_equivalence_witness"] = {
                "event_step_id": state["current_equivalent_event_step_id"],
                "post_state_member_path": current["member_path"],
                "post_state_sha256": current["sha256"],
            }
            state["validated_action_sha256"] = "4" * 64
    selection["content_witness_status"] = (
        "COMPLETE_FOR_45_SCREENING_AND_20_CONFIRM_STATES"
    )
    return selection


class RestorationV2SelectionTest(unittest.TestCase):
    def test_step_six_identity_and_candidate_semantics(self) -> None:
        manifest = _build()
        states = manifest["roles"]["v2_confirm_primary"]["states"]
        self.assertEqual(len(states), 20)
        for state in states:
            self.assertEqual(state["decision_step_id"], 6)
            self.assertEqual(state["state_id"], state_id(state["source_id"], 6))
            self.assertEqual(state["history_event_step_ids"], [1, 2, 3, 4, 5])
            self.assertEqual(state["candidate_event_step_ids"], [1, 2, 3, 4])
            self.assertEqual(state["current_equivalent_event_step_id"], 5)

    def test_role_counts_and_disjointness_are_fail_closed(self) -> None:
        manifest = _build()
        self.assertEqual(len(manifest["roles"]["v2_label_train"]["states"]), 30)
        self.assertEqual(len(manifest["roles"]["v2_development"]["states"]), 15)
        self.assertEqual(
            manifest["overlap_proof"]["reference_oracle_confirm_union_count"],
            43,
        )
        invalid = copy.deepcopy(manifest)
        invalid["roles"]["v2_confirm_primary"]["trajectories"][0] = copy.deepcopy(
            invalid["roles"]["v2_label_train"]["trajectories"][0]
        )
        _, _, v2, _, _, _, _ = _fixture()
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_selection_manifest(invalid, v2_contract=v2)

    def test_app_diversity_cannot_be_repaired_by_top_up(self) -> None:
        apps = tuple("one-app" for _ in range(20))
        with self.assertRaisesRegex(ValueError, "lacks app diversity"):
            _build(confirm_apps=apps)

    def test_pool_record_mutation_is_rejected(self) -> None:
        manifest = _build()
        invalid = copy.deepcopy(manifest)
        invalid["reconstruction"]["eligible_pool"][30]["decision_count"] += 1
        _, _, v2, _, _, _, _ = _fixture()
        with self.assertRaisesRegex(ValueError, "reproduce their SHA256"):
            validate_selection_manifest(invalid, v2_contract=v2)

    def test_app_diversity_minimum_is_bound_to_scientific_contract(self) -> None:
        manifest = _build()
        invalid = copy.deepcopy(manifest)
        invalid["confirm_selection_proof"]["minimum_distinct_app_labels"] = 0
        _, _, v2, _, _, _, _ = _fixture()
        with self.assertRaisesRegex(ValueError, "scientific contract"):
            validate_selection_manifest(invalid, v2_contract=v2)

    def test_exposure_reducer_rejects_digest_drift(self) -> None:
        ledger = {
            "role_source_ids": {"role": ["one", "two"]},
            "events": [
                {
                    "event_id": "raw",
                    "roles": ["role"],
                    "source_ids_sha256": sha256_bytes(
                        canonical_json_bytes(["one", "two"])
                    ),
                    "source_id_count": 2,
                    "access_kind": "raw_source_machine_access",
                }
            ],
        }
        reduced = reduce_exposure_events(ledger)
        self.assertTrue(reduced["role"]["raw_machine_seen"])
        ledger["events"][0]["source_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            reduce_exposure_events(ledger)

    def test_exposure_validator_binds_required_evidence(self) -> None:
        selection = _selection_with_content_witnesses()
        selection_sha = sha256_bytes(pretty_json_bytes(selection))
        reference_ids = [
            record["source_id"]
            for record in selection["roles"][
                "v1_reference_contract_audit_only"
            ]["trajectories"]
        ]
        summary = {
            "outcome": "NO_GO_CURRENT_REFERENCE_STACK",
            "by_trajectory": {source_id: {} for source_id in reference_ids},
            "oracle_split_status": "not_evaluated_due_to_reference_gate_failure",
            "policy": {
                "repo": "ByteDance-Seed/UI-TARS-1.5-7B",
                "revision": "683d002dd99d8f95104d31e70391a39348857f4e",
            },
            "artifacts": {
                "hf_repo": "gavinlaw/causalcache-guiodyssey-independent-mobile",
                "hf_revision": "b3e1245c6c6a1723fe2ca3a861148008df39df46",
                "hf_path": "runs/independent-reference-gate-v1",
            },
        }
        ledger = build_exposure_ledger(
            selection,
            selection_manifest_sha256=selection_sha,
            v1_summary_sha256=(
                "c49d79dff67aabb0867bda56836f7337201a5015ba618160c45e8dd011b64c8e"
            ),
            v1_summary=summary,
        )
        validate_exposure_ledger(ledger, selection_manifest=selection)
        invalid = copy.deepcopy(ledger)
        del invalid["events"][3]["evidence"]["hf_revision"]
        with self.assertRaisesRegex(ValueError, "policy-output evidence mismatch"):
            validate_exposure_ledger(invalid, selection_manifest=selection)

    def test_current_image_cannot_be_reused_as_candidate(self) -> None:
        from causalcache.data.restoration_v2_selection import (
            validate_state_content_witnesses,
        )

        selection = _selection_with_content_witnesses()
        state = selection["roles"]["v2_confirm_primary"]["states"][0]
        state["candidate_event_post_states"][0]["post_state_member_path"] = state[
            "current_observation"
        ]["member_path"]
        state["candidate_event_post_states"][0]["post_state_sha256"] = state[
            "current_observation"
        ]["sha256"]
        with self.assertRaisesRegex(ValueError, "does not match its event|must differ"):
            validate_state_content_witnesses(selection)


if __name__ == "__main__":
    unittest.main()
