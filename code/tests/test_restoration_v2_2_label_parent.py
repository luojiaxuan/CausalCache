from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from causalcache.restoration_v2_2_eager_artifact import (
    NO_GO_OUTCOME,
    collect_raw_v22_evidence,
    deterministic_tar_bytes,
    pretty_json_bytes,
    read_v22_evidence_archive,
)
from causalcache.restoration_v2_2_label_parent import (
    extract_v22_label_parent_states,
)
from tests.test_run_restoration_v2_2_eager_substrate import (
    SOURCE_COMMIT,
    successful_attempt,
)


def _mutate_json(payload: bytes, mutation: object) -> bytes:
    value = json.loads(payload)
    mutation(value)
    return pretty_json_bytes(value)


class V22LabelParentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        base = Path(cls._temporary.name)
        _, root, ledger = successful_attempt(base)
        collected = collect_raw_v22_evidence(
            root,
            ledger,
            expected_source_git_commit=SOURCE_COMMIT,
            require_canonical_location=False,
        )
        archive = base / "v22-parent.tar"
        archive.write_bytes(deterministic_tar_bytes(collected.files))
        cls.evidence = read_v22_evidence_archive(
            archive,
            expected_source_git_commit=SOURCE_COMMIT,
            require_canonical_attempt_identity=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_extracts_sorted_immutable_parent_states_and_hashes(self) -> None:
        records = extract_v22_label_parent_states(self.evidence)
        self.assertEqual(tuple(record.index for record in records), tuple(range(45)))
        self.assertEqual(tuple(record.role for record in records[:30]), ("v2_label_train",) * 30)
        self.assertEqual(tuple(record.role for record in records[30:]), ("v2_development",) * 15)
        self.assertEqual(records[0].member_name, "workers/even/states/000.json")
        self.assertEqual(records[1].member_name, "workers/odd/states/001.json")
        self.assertEqual(records[2].candidate_event_step_ids, (1, 2, 3, 4))
        self.assertEqual(records[0].canonical_action.action, "wait")
        self.assertEqual(
            records[0].member_sha256,
            hashlib.sha256(self.evidence.files[records[0].member_name]).hexdigest(),
        )
        self.assertEqual(
            records[0].canonical_action_sha256,
            hashlib.sha256(b'{"action":"wait"}').hexdigest(),
        )
        with self.assertRaises(FrozenInstanceError):
            records[0].index = 7

    def test_rejects_nonpass_parent(self) -> None:
        evidence = replace(self.evidence, outcome=NO_GO_OUTCOME)
        with self.assertRaisesRegex(ValueError, "complete PASS"):
            extract_v22_label_parent_states(evidence)

    def test_rejects_missing_and_duplicate_state_members(self) -> None:
        files = dict(self.evidence.files)
        del files["workers/even/states/000.json"]
        with self.assertRaisesRegex(ValueError, "inventory"):
            extract_v22_label_parent_states(replace(self.evidence, files=files))

        files = dict(self.evidence.files)
        files["workers/odd/states/000.json"] = files["workers/even/states/000.json"]
        with self.assertRaisesRegex(ValueError, "duplicate state index"):
            extract_v22_label_parent_states(replace(self.evidence, files=files))

    def test_rejects_projection_candidate_drift_even_when_both_copies_match(self) -> None:
        files = dict(self.evidence.files)
        member = "workers/even/states/000.json"

        def mutate_state(value: dict[str, object]) -> None:
            value["state"]["candidate_event_step_ids"] = [1]
            value["measurement_kernel"]["record"]["state"][
                "candidate_event_step_ids"
            ] = [1]

        files[member] = _mutate_json(files[member], mutate_state)
        run_contract = copy.deepcopy(self.evidence.run_contract)
        run_contract["states"][0]["candidate_event_step_ids"] = [1]
        evidence = replace(self.evidence, files=files, run_contract=run_contract)
        with self.assertRaisesRegex(ValueError, "candidate event IDs drifted"):
            extract_v22_label_parent_states(evidence)

    def test_rejects_outer_state_outcome_drift(self) -> None:
        files = dict(self.evidence.files)
        member = "workers/even/states/000.json"

        def mutate(value: dict[str, object]) -> None:
            value["outcome"] = "FAILED_V2_2_EAGER_FULL_45_SUBSTRATE_STATE"

        files[member] = _mutate_json(files[member], mutate)
        with self.assertRaisesRegex(ValueError, "not a valid v2.2 state"):
            extract_v22_label_parent_states(replace(self.evidence, files=files))

    def test_rejects_generation_action_drift(self) -> None:
        files = dict(self.evidence.files)
        member = "workers/even/states/000.json"

        def mutate(value: dict[str, object]) -> None:
            value["measurement_kernel"]["record"]["native_generations"][1][
                "canonical_action"
            ] = {"action": "terminate", "status": "success"}

        files[member] = _mutate_json(files[member], mutate)
        with self.assertRaisesRegex(ValueError, "generation canonical actions drifted"):
            extract_v22_label_parent_states(replace(self.evidence, files=files))

    def test_rejects_record_action_drift(self) -> None:
        files = dict(self.evidence.files)
        member = "workers/even/states/000.json"

        def mutate(value: dict[str, object]) -> None:
            value["measurement_kernel"]["record"]["canonical_action"] = {
                "action": "terminate",
                "status": "success",
            }

        files[member] = _mutate_json(files[member], mutate)
        with self.assertRaisesRegex(ValueError, "record canonical action drifted"):
            extract_v22_label_parent_states(replace(self.evidence, files=files))

    def test_rejects_alias_instead_of_strict_canonical_action(self) -> None:
        files = dict(self.evidence.files)
        member = "workers/even/states/000.json"

        def mutate(value: dict[str, object]) -> None:
            record = value["measurement_kernel"]["record"]
            alias = {"action": "tap", "coordinate": [1, 2]}
            record["canonical_action"] = copy.deepcopy(alias)
            for generation in record["native_generations"]:
                generation["canonical_action"] = copy.deepcopy(alias)

        files[member] = _mutate_json(files[member], mutate)
        with self.assertRaisesRegex(ValueError, "canonical GUI-Owl v2 action"):
            extract_v22_label_parent_states(replace(self.evidence, files=files))


if __name__ == "__main__":
    unittest.main()
