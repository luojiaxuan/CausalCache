from __future__ import annotations

import copy
import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path

from causalcache.data.restoration_v2_2_expansion_label_inputs import (
    V22ExpansionLabelPromptSpec,
    build_v2_2_expansion_label_messages,
    expansion_label_prompt_inventory,
    iter_v2_2_expansion_label_prompt_specs,
    make_v2_2_expansion_label_prompt_spec,
)
from causalcache.data.restoration_v2_2_expansion_label_parent import (
    extract_expansion_label_parent_states,
)
from causalcache.restoration_v2_2_expansion_substrate_artifact import (
    PASS_OUTCOME,
    STATE_OUTCOME_VALID,
    ExpansionSubstrateEvidence,
    pretty_json_bytes,
)
from causalcache.restoration_v2_2_expansion_substrate_inputs import (
    build_decision_view_input,
)
from scripts.run_restoration_v2_2_expansion_substrate import (
    ExpansionArtifact,
    ExpansionState,
    build_expansion_messages,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROMPT_FIXTURE = (
    REPOSITORY_ROOT / "data/fixtures/restoration_v2_prompt_low_fidelity.json"
)
TREE_SHA = "1" * 64


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _projection(state: ExpansionState) -> dict[str, object]:
    return {
        "state_index": state.index,
        "role": state.role,
        "source_id": state.source_id,
        "state_id": state.state_id,
        "decision_step_id": state.decision_step_id,
        "history_event_step_ids": list(state.decision["history_event_step_ids"]),
        "candidate_event_step_ids": list(
            state.decision["candidate_event_step_ids"]
        ),
        "current_equivalent_event_step_id": state.decision[
            "current_equivalent_event_step_id"
        ],
    }


def _artifact() -> ExpansionArtifact:
    template = json.loads(PROMPT_FIXTURE.read_text(encoding="utf-8"))["trajectories"][0]
    trajectories: list[dict[str, object]] = []
    states: list[ExpansionState] = []
    image_payloads: dict[str, bytes] = {}
    for trajectory_index in range(64):
        trajectory = copy.deepcopy(template)
        source_id = f"expansion-fixture-{trajectory_index:03d}"
        role = (
            "gate_train_expansion"
            if trajectory_index < 48
            else "gate_development_expansion"
        )
        trajectory["source_id"] = source_id
        trajectory["role"] = role
        for decision in trajectory["decisions"]:
            step = decision["decision_step_id"]
            decision["state_id"] = f"{source_id}:decision_step:{step:03d}"
        trajectories.append(trajectory)
        for decision in trajectory["decisions"]:
            states.append(
                ExpansionState(
                    index=len(states),
                    role=role,
                    source_id=source_id,
                    decision=decision,
                )
            )
        for event in trajectory["events"]:
            path = event["observation_after_path"]
            image_payloads[path] = path.encode("utf-8")
        for decision in trajectory["decisions"]:
            path = decision["current_observation_path"]
            image_payloads[path] = path.encode("utf-8")
    return ExpansionArtifact(
        root=Path("/immutable-expansion"),
        tree={"artifact_tree_sha256": TREE_SHA},
        validation={
            "outcome": "PASSED_GUIODYSSEY_RESTORATION_V2_EXPANSION_VALIDATION",
            "counts": {"state_count": 192},
            "per_decision_view_current_expert_action_payload_included": False,
            "consumer_must_slice_events_by_history_event_step_ids": True,
        },
        trajectories=tuple(trajectories),
        states=tuple(states),
        image_payloads=image_payloads,
    )


def _evidence(artifact: ExpansionArtifact) -> ExpansionSubstrateEvidence:
    projections = [_projection(state) for state in artifact.states]
    files: dict[str, bytes] = {}
    for state, projection in zip(artifact.states, projections, strict=True):
        trajectory = artifact.trajectory(state)
        view = build_decision_view_input(trajectory, state.decision)
        request = dict(view["request_manifest"])
        action = {"action": "wait"}
        inner = {
            "state": projection,
            "outcome": "VALID_V2_1_FULL_45_SUBSTRATE_STATE",
            "failure": None,
            "slice_witness": {
                **request,
                "slice_witness_sha256": view["slice_witness_sha256"],
            },
            "request_manifest": request,
            "processor_canary": {"passed": True},
            "parse_success": True,
            "finite_logit_distances": True,
            "repeat_canonical_action_agreement": True,
            "canonical_action": action,
            "native_generations": [
                {"repeat_index": 1, "canonical_action": action},
                {"repeat_index": 2, "canonical_action": action},
            ],
        }
        worker = "even" if state.index % 2 == 0 else "odd"
        envelope = {
            "state": projection,
            "worker": {"worker_id": worker},
            "outcome": STATE_OUTCOME_VALID,
            "measurement_kernel": {"record": inner},
        }
        files[f"workers/{worker}/states/{state.index:03d}.json"] = pretty_json_bytes(
            envelope
        )
    return ExpansionSubstrateEvidence(
        files=files,
        run_contract={"states": projections},
        source_git_commit="a" * 40,
        execution_git_commit="b" * 40,
        run_contract_sha256="2" * 64,
        config_sha256="3" * 64,
        completion_manifest_sha256="4" * 64,
        derived_immutable_revision="c" * 40,
        outcome=PASS_OUTCOME,
        aggregate_sha256="5" * 64,
        fixed_state_denominator=192,
        attempted_state_count=192,
        completed_state_count=192,
        planned_counts={},
        actual_counts={},
        valid_state_count=192,
        failed_state_count=0,
        failure_category_counts={},
        gate_summary={"gate_passed": True},
        inventory=(),
        tree_inventory_sha256="6" * 64,
        runtime_identity_sha256="7" * 64,
        worker_identity_sha256="8" * 64,
        log_inventory_sha256="9" * 64,
        monitor_summary_sha256="0" * 64,
    )


def _decode(payload: bytes) -> str:
    return payload.decode("utf-8")


def _message_bytes(messages: object) -> bytes:
    return _canonical_json_bytes(messages)


class RestorationV22ExpansionLabelInputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = _artifact()
        cls.evidence = _evidence(cls.artifact)
        cls.parents = extract_expansion_label_parent_states(
            cls.evidence,
            cls.artifact,
        )

    def test_parent_extractor_requires_complete_pass_and_immutable_projection(self) -> None:
        self.assertEqual(len(self.parents), 192)
        self.assertEqual(tuple(parent.index for parent in self.parents), tuple(range(192)))
        self.assertEqual(self.parents[0].role, "gate_train_expansion")
        self.assertEqual(self.parents[144].role, "gate_development_expansion")
        self.assertEqual(self.parents[2].candidate_event_step_ids, (1, 2, 3, 4))
        self.assertEqual(self.parents[0].canonical_action.action, "wait")
        self.assertEqual(
            self.parents[0].canonical_action_sha256,
            hashlib.sha256(b'{"action":"wait"}').hexdigest(),
        )

        with self.assertRaisesRegex(ValueError, "complete PASS/192"):
            extract_expansion_label_parent_states(
                replace(self.evidence, completed_state_count=191),
                self.artifact,
            )
        drifted_states = list(self.artifact.states)
        drifted_states[0] = replace(drifted_states[0], role="v2_label_train")
        drifted_artifact = replace(self.artifact, states=tuple(drifted_states))
        with self.assertRaisesRegex(ValueError, "immutable manifest"):
            extract_expansion_label_parent_states(self.evidence, drifted_artifact)

    def test_all_n2_n3_n4_coalitions_preserve_request_identity_and_image_count(self) -> None:
        expected_counts = {2: 4, 3: 8, 4: 16}
        for parent in self.parents[:3]:
            n = len(parent.candidate_event_step_ids)
            specs = []
            for mask in range(1 << n):
                restored = tuple(
                    event_id
                    for bit, event_id in enumerate(parent.candidate_event_step_ids)
                    if mask & (1 << bit)
                )
                spec = make_v2_2_expansion_label_prompt_spec(
                    self.artifact,
                    parent,
                    restored_event_step_ids=restored,
                )
                bundle = build_v2_2_expansion_label_messages(
                    self.artifact,
                    spec,
                    image_decoder=_decode,
                )
                inventory = expansion_label_prompt_inventory(spec, bundle)
                specs.append(spec)
                self.assertEqual(bundle.restored_event_step_ids, restored)
                self.assertEqual(inventory["image_count"], 1 + len(restored))
                self.assertEqual(
                    tuple(bundle.request_manifest["included_event_step_ids"]),
                    parent.history_event_step_ids,
                )
                self.assertEqual(
                    bundle.request_manifest[
                        "current_or_future_event_action_exposure_count"
                    ],
                    0,
                )
                self.assertEqual(
                    hashlib.sha256(
                        _canonical_json_bytes(bundle.request_manifest)
                    ).hexdigest(),
                    parent.request_manifest_sha256,
                )
                self.assertEqual(
                    bundle.slice_witness_sha256,
                    parent.slice_witness_sha256,
                )
            self.assertEqual(len(specs), expected_counts[n])
            self.assertEqual(
                [spec.coalition_mask for spec in specs],
                list(range(expected_counts[n])),
            )

    def test_full_and_empty_are_byte_identical_to_substrate_endpoints(self) -> None:
        for parent in self.parents[:3]:
            state = self.artifact.states[parent.index]
            cases = (
                ((), "summary_only"),
                (parent.candidate_event_step_ids, "reference"),
            )
            for restored, endpoint in cases:
                with self.subTest(n=len(parent.candidate_event_step_ids), endpoint=endpoint):
                    spec = make_v2_2_expansion_label_prompt_spec(
                        self.artifact,
                        parent,
                        restored_event_step_ids=restored,
                    )
                    label_bundle = build_v2_2_expansion_label_messages(
                        self.artifact,
                        spec,
                        image_decoder=_decode,
                    )
                    substrate_bundle = build_expansion_messages(
                        self.artifact,
                        state,
                        endpoint,
                        _decode,
                    )
                    self.assertEqual(
                        _message_bytes(label_bundle.messages),
                        _message_bytes(substrate_bundle.messages),
                    )
                    self.assertEqual(label_bundle, substrate_bundle)

    def test_full_enumeration_is_1792_and_budget_does_not_filter(self) -> None:
        specs = list(
            iter_v2_2_expansion_label_prompt_specs(
                self.artifact,
                self.parents,
                budget_event_capacity=1,
            )
        )
        self.assertEqual(len(specs), 1792)
        self.assertTrue(all(spec.budget_event_capacity == 1 for spec in specs))
        self.assertEqual(specs[0].restored_event_step_ids, ())
        self.assertEqual(specs[-1].restored_event_step_ids, (1, 2, 3, 4))
        first_n3 = [
            spec.restored_event_step_ids
            for spec in specs
            if spec.parent.index == 1
        ]
        self.assertEqual(
            first_n3,
            [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)],
        )

    def test_current_and_future_action_mutation_is_not_exposed(self) -> None:
        parent = self.parents[0]
        spec = make_v2_2_expansion_label_prompt_spec(
            self.artifact,
            parent,
            restored_event_step_ids=(1,),
        )
        base = build_v2_2_expansion_label_messages(
            self.artifact,
            spec,
            image_decoder=_decode,
        )
        trajectories = copy.deepcopy(list(self.artifact.trajectories))
        for event in trajectories[0]["events"]:
            if event["step_id"] >= parent.decision_step_id:
                event["source_tool_call"] = {
                    "causalcache_canary": f"excluded-{event['step_id']}"
                }
                event["executed_action"] = {
                    "causalcache_canary": f"excluded-{event['step_id']}"
                }
        mutated_artifact = replace(self.artifact, trajectories=tuple(trajectories))
        mutated = build_v2_2_expansion_label_messages(
            mutated_artifact,
            spec,
            image_decoder=_decode,
        )
        self.assertEqual(base.messages, mutated.messages)
        self.assertEqual(base.request_manifest, mutated.request_manifest)
        self.assertEqual(
            mutated.request_manifest["current_or_future_event_action_exposure_count"],
            0,
        )

    def test_legacy_roles_and_noncanonical_coalitions_are_rejected(self) -> None:
        parent = self.parents[2]
        with self.assertRaisesRegex(ValueError, "legacy 45-state roles"):
            V22ExpansionLabelPromptSpec(
                parent=replace(parent, role="v2_label_train"),
                restored_event_step_ids=(),
            )
        for coalition, message in (
            ((1, 1), "duplicate"),
            ((2, 1), "canonical sorted"),
            ((1, 5), "unknown candidate"),
        ):
            with self.subTest(coalition=coalition), self.assertRaisesRegex(
                ValueError, message
            ):
                make_v2_2_expansion_label_prompt_spec(
                    self.artifact,
                    parent,
                    restored_event_step_ids=coalition,
                )


if __name__ == "__main__":
    unittest.main()
