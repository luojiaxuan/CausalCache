from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest

import causalcache.independent_confirm_data as confirm_data
from causalcache.independent_confirm_artifact import read_independent_decisions
from causalcache.gate_v1_data import CandidateFeatures, FeatureState
from causalcache.gate_v1_provenance import (
    GATE_V1_CONFIG_SHA256,
    FORMAL_TRAIN_SOURCE_IDS_SHA256,
    ArtifactBinding,
    FrozenEnsembleProvenance,
    FrozenTrainingProvenance,
    SeedCheckpointProvenance,
)
from causalcache.gate_v1_training import FittedEnsemble
from causalcache.independent_confirm_data import (
    CONFIRM_CANDIDATE_EVENT_STEP_IDS,
    CONFIRM_SOURCE_IDS,
    Confirm20ImageRequirement,
    Confirm20PolicyWorkItem,
    FROZEN_TRANSPORT_IDENTITY,
    ValidatedConfirm20Payloads,
)
from causalcache.independent_confirm_runner import (
    REFERENCE_FAILURE_OUTCOME,
    VALID_STATE_OUTCOME,
    ConfirmWorkerResult,
    PayloadCommitReceipt,
    aggregate_independent_confirm_workers,
    build_confirm_messages,
    confirm_worker_assignments,
    independent_selection_payload_bytes,
    join_confirm_worker_results,
    load_independent_ensemble,
    persist_and_seal_label_blind_payload,
    publish_payload_commit,
    run_confirm_state_once,
    score_and_select_independent,
)
from causalcache.low_fidelity_v2 import LowFidelityEventV2, serialize_low_fidelity_v2
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    GUI_OWL_V2_1_SYSTEM_PROMPT,
    parse_gui_owl_v2_1_output,
    validate_gui_owl_v2_1_native_messages,
)


RUN_SHA = "a" * 64
WAIT_OUTPUT = (
    '<tool_call>\n{"name":"mobile_use","arguments":{"action":"wait"}}\n'
    "</tool_call>"
)
CLICK_OUTPUT = (
    '<tool_call>\n{"name":"mobile_use","arguments":{"action":"click",'
    '"coordinate":[1,2]}}\n</tool_call>'
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _feature(index: int) -> FeatureState:
    source_id = CONFIRM_SOURCE_IDS[index]
    return FeatureState(
        source_id=source_id,
        state_id=f"{source_id}:decision_step:006",
        decision_step_id=6,
        candidate_event_step_ids=CONFIRM_CANDIDATE_EVENT_STEP_IDS,
        q64=(0.0,) * 64,
        candidates=tuple(
            CandidateFeatures(
                event_step_id=event,
                h64=(0.0,) * 64,
                g8=(0.0,) * 8,
            )
            for event in CONFIRM_CANDIDATE_EVENT_STEP_IDS
        ),
    )


def _requirement(index: int, event: int) -> Confirm20ImageRequirement:
    payload = f"image-{index}-{event}".encode()
    source_id = CONFIRM_SOURCE_IDS[index]
    return Confirm20ImageRequirement(
        image_member_path=f"images/{source_id}/{event}.png",
        image_sha256=_sha(payload),
        canonical_ocr_record_sha256=f"{index * 10 + event:064x}",
    )


def _work_item(index: int) -> Confirm20PolicyWorkItem:
    source_id = CONFIRM_SOURCE_IDS[index]
    return Confirm20PolicyWorkItem(
        ordinal=index,
        source_id=source_id,
        state_id=f"{source_id}:decision_step:006",
        decision_step_id=6,
        event_images=tuple(
            (event, _requirement(index, event))
            for event in CONFIRM_CANDIDATE_EVENT_STEP_IDS
        ),
        current_image=_requirement(index, 5),
    )


def _native_messages(
    _item: Confirm20PolicyWorkItem, coalition: tuple[int, ...]
) -> list[dict]:
    mask = sum(1 << (event - 1) for event in coalition)
    content = [{"type": "text", "text": f"coalition_mask={mask}"}]
    content.extend({"type": "image", "image": f"event-{event}"} for event in coalition)
    content.extend(
        [
            {"type": "image", "image": "current"},
            {"type": "text", "text": GUI_OWL_V2_1_FINAL_USER_INSTRUCTION},
        ]
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_V2_1_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]


def _mask(messages) -> int:
    text = messages[1]["content"][0]["text"]
    return int(text.split("=", maxsplit=1)[1])


class _FakeRuntime:
    def __init__(self, outputs=(WAIT_OUTPUT, WAIT_OUTPUT)) -> None:
        self.outputs = iter(outputs)
        self.generation_calls = 0
        self.teacher_calls = 0

    def generate_native_action(self, _messages):
        self.generation_calls += 1
        output = next(self.outputs)
        parsed = parse_gui_owl_v2_1_output(output)
        return SimpleNamespace(
            output_text=output,
            parsed_output=parsed,
            metadata={"generation_call": self.generation_calls},
        )

    def teacher_forced_distance_logits(self, messages_batch, actions):
        assert len(messages_batch) == len(actions) == 1
        self.teacher_calls += 1
        mask = _mask(messages_batch[0])
        return mask, {"teacher_call": self.teacher_calls, "coalition_mask": mask}


class _FakeDistanceBackend:
    def prepare_reference(self, logits):
        assert logits == 15
        return logits

    def measure(self, reference, candidate):
        return SimpleNamespace(
            value=abs(reference - candidate) / 100.0,
            audit={"full_tensor_host_transfers": 0},
        )


class _UnstableRepeatDistanceBackend(_FakeDistanceBackend):
    def __init__(self) -> None:
        self.measure_calls = 0

    def measure(self, reference, candidate):
        self.measure_calls += 1
        if self.measure_calls == 1:
            return SimpleNamespace(
                value=2e-4,
                audit={"full_tensor_host_transfers": 0},
            )
        return super().measure(reference, candidate)


def _receipt(events: list[str] | None = None) -> PayloadCommitReceipt:
    events = [] if events is None else events
    selections = _selections(tuple(_feature(index) for index in range(20)))
    selection_path = "confirm/independent.json"
    selection_payload = independent_selection_payload_bytes(selections)

    def persist(files, inventory):
        assert files and inventory
        events.append("persist")
        digest = _sha(
            __import__("json").dumps(
                list(inventory), sort_keys=True, separators=(",", ":")
            ).encode()
        )
        return {
            "durable": True,
            "file_count": len(inventory),
            "inventory_sha256": digest,
            "persistence_id": "local-fixed-payload",
        }

    seal = persist_and_seal_label_blind_payload(
        {selection_path: selection_payload},
        independent_selections=selections,
        independent_selection_path=selection_path,
        persist_fn=persist,
    )

    def publish(_files, _inventory):
        events.append("publish")
        return {
            "payload_commit": "b" * 40,
            "inventory_sha256": seal.inventory_sha256,
        }

    return publish_payload_commit(
        seal,
        publish_fn=publish,
        verify_fn=lambda commit, inventory: (
            events.append("verify") is None
            and commit == "b" * 40
            and len(inventory) == 1
        ),
    )


def _fake_ensemble() -> FittedEnsemble:
    return FittedEnsemble(
        family="independent",
        learning_rate=0.001,
        seeds=(0, 1, 2, 3, 4),
        selected_epochs=(1, 1, 1, 1, 1),
        selection_sha256="c" * 64,
        models=(0, 1, 2, 3, 4),
    )


def _selections(features):
    def predict(_model, vectors):
        return tuple((-1.0, 0.1, 0.2, 0.3)[index % 4] for index in range(len(vectors)))

    return score_and_select_independent(features, _fake_ensemble(), predict_fn=predict)


def test_runner_independent_payload_is_the_canonical_artifact_format() -> None:
    selections = _selections(tuple(_feature(index) for index in range(20)))
    payload = independent_selection_payload_bytes(selections)
    replay = read_independent_decisions(payload)
    assert set(replay["ensemble_scores"]) == {
        f"{source_id}:decision_step:006" for source_id in CONFIRM_SOURCE_IDS
    }
    assert _sha(payload) == selections.selection_sha256


def test_full_state_runs_two_generations_and_all_16_coalitions_after_commit() -> None:
    events: list[str] = []
    receipt = _receipt(events)
    runtime = _FakeRuntime()
    record = run_confirm_state_once(
        work_item=_work_item(0),
        message_builder=_native_messages,
        runtime=runtime,
        distance_backend=_FakeDistanceBackend(),
        payload_commit_receipt=receipt,
        run_contract_sha256=RUN_SHA,
    )

    assert events == ["persist", "publish", "verify"]
    assert record["outcome"] == VALID_STATE_OUTCOME
    assert record["canonical_action"] == {"action": "wait"}
    assert runtime.generation_calls == 2
    assert runtime.teacher_calls == 17
    assert len(record["distance_rows"]) == 16
    assert {row["coalition_mask"] for row in record["distance_rows"]} == set(range(16))
    full = next(row for row in record["distance_rows"] if row["coalition_mask"] == 15)
    assert full["distance"] == 0.0
    assert record["operation_counts"] == {
        "generation_call_count": 2,
        "reference_teacher_forward_count": 1,
        "reference_repeat_teacher_forward_count": 1,
        "non_reference_coalition_teacher_forward_count": 15,
        "teacher_forward_call_count": 17,
        "teacher_forward_example_count": 17,
        "reference_repeat_kl_measurement_count": 1,
        "coalition_kl_measurement_count": 15,
        "kl_measurement_count": 16,
        "raw_distance_row_count": 16,
        "retry_count": 0,
        "top_up_count": 0,
        "filter_count": 0,
    }


def test_reference_action_mismatch_is_fixed_denominator_failure_without_teacher_retry() -> None:
    runtime = _FakeRuntime((WAIT_OUTPUT, CLICK_OUTPUT))
    record = run_confirm_state_once(
        work_item=_work_item(0),
        message_builder=_native_messages,
        runtime=runtime,
        distance_backend=_FakeDistanceBackend(),
        payload_commit_receipt=_receipt(),
        run_contract_sha256=RUN_SHA,
    )
    assert record["outcome"] == REFERENCE_FAILURE_OUTCOME
    assert record["failure"]["category"] == "CANONICAL_ACTION_MISMATCH"
    assert record["distance_rows"] == []
    assert runtime.generation_calls == 2
    assert runtime.teacher_calls == 0
    assert record["operation_counts"]["retry_count"] == 0
    assert record["operation_counts"]["top_up_count"] == 0


def test_reference_repeat_kl_is_fixed_denominator_failure_not_execution_error() -> None:
    runtime = _FakeRuntime()
    record = run_confirm_state_once(
        work_item=_work_item(0),
        message_builder=_native_messages,
        runtime=runtime,
        distance_backend=_UnstableRepeatDistanceBackend(),
        payload_commit_receipt=_receipt(),
        run_contract_sha256=RUN_SHA,
    )
    assert record["outcome"] == REFERENCE_FAILURE_OUTCOME
    assert record["failure"]["category"] == "REFERENCE_REPEAT_KL_EXCEEDED"
    assert record["reference_repeat_kl"] == pytest.approx(2e-4)
    assert record["distance_rows"] == []
    assert runtime.generation_calls == 2
    assert runtime.teacher_calls == 2
    assert record["operation_counts"] == {
        "generation_call_count": 2,
        "reference_teacher_forward_count": 1,
        "reference_repeat_teacher_forward_count": 1,
        "non_reference_coalition_teacher_forward_count": 0,
        "teacher_forward_call_count": 2,
        "teacher_forward_example_count": 2,
        "reference_repeat_kl_measurement_count": 1,
        "coalition_kl_measurement_count": 0,
        "kl_measurement_count": 1,
        "raw_distance_row_count": 0,
        "retry_count": 0,
        "top_up_count": 0,
        "filter_count": 0,
    }


def test_reference_generation_requires_unforgeable_payload_commit_receipt() -> None:
    selections = _selections(tuple(_feature(index) for index in range(20)))
    with pytest.raises(ValueError, match="exact sealed independent decisions"):
        persist_and_seal_label_blind_payload(
            {"confirm/independent.json": b"placeholder"},
            independent_selections=selections,
            independent_selection_path="confirm/independent.json",
            persist_fn=lambda _files, _inventory: {},
        )
    with pytest.raises(TypeError, match="cannot be constructed"):
        PayloadCommitReceipt(
            payload_commit="b" * 40,
            inventory_sha256="c" * 64,
            selection_sha256="d" * 64,
            _token=object(),
        )
    with pytest.raises(PermissionError, match="payload commit receipt"):
        run_confirm_state_once(
            work_item=_work_item(0),
            message_builder=_native_messages,
            runtime=_FakeRuntime(),
            distance_backend=_FakeDistanceBackend(),
            payload_commit_receipt=object(),  # type: ignore[arg-type]
            run_contract_sha256=RUN_SHA,
        )


def test_worker_assignment_is_four_by_five_and_barrier_rejects_partial_results() -> None:
    items = tuple(_work_item(index) for index in range(20))
    assignments = confirm_worker_assignments(items)
    assert [assignment.ordinals for assignment in assignments] == [
        (0, 4, 8, 12, 16),
        (1, 5, 9, 13, 17),
        (2, 6, 10, 14, 18),
        (3, 7, 11, 15, 19),
    ]
    template = run_confirm_state_once(
        work_item=items[0],
        message_builder=_native_messages,
        runtime=_FakeRuntime(),
        distance_backend=_FakeDistanceBackend(),
        payload_commit_receipt=_receipt(),
        run_contract_sha256=RUN_SHA,
    )
    results = []
    for assignment in assignments:
        records = []
        for ordinal, state_id in zip(assignment.ordinals, assignment.state_ids, strict=True):
            record = copy.deepcopy(template)
            record["state"].update(
                {
                    "ordinal": ordinal,
                    "source_id": items[ordinal].source_id,
                    "state_id": state_id,
                }
            )
            records.append(record)
        results.append(ConfirmWorkerResult(assignment.worker_id, tuple(records)))
    with pytest.raises(ValueError, match="all four workers"):
        join_confirm_worker_results(assignments, results[:3])
    ordered = join_confirm_worker_results(assignments, results)
    assert tuple(record["state"]["ordinal"] for record in ordered) == tuple(range(20))


def test_worker_aggregation_joins_expansion_tables_and_calls_fixed_evaluator() -> None:
    features = tuple(_feature(index) for index in range(20))
    items = tuple(_work_item(index) for index in range(20))
    assignments = confirm_worker_assignments(items)
    template = run_confirm_state_once(
        work_item=items[0],
        message_builder=_native_messages,
        runtime=_FakeRuntime(),
        distance_backend=_FakeDistanceBackend(),
        payload_commit_receipt=_receipt(),
        run_contract_sha256=RUN_SHA,
    )
    worker_results = []
    for assignment in assignments:
        rows = []
        for ordinal, state_id in zip(assignment.ordinals, assignment.state_ids, strict=True):
            record = copy.deepcopy(template)
            record["state"].update(
                {
                    "ordinal": ordinal,
                    "source_id": features[ordinal].source_id,
                    "state_id": state_id,
                }
            )
            rows.append(record)
        worker_results.append(ConfirmWorkerResult(assignment.worker_id, tuple(rows)))
    selections = _selections(features)
    assert all(value == (3, 4) for value in selections.ensemble_selections.values())
    assert independent_selection_payload_bytes(selections)
    heuristics = {
        "dynamic_recent": {feature.state_id: (3, 4) for feature in features},
        "ocr_rgb_v2": {feature.state_id: (1, 2) for feature in features},
        "policy_vision_v3": {feature.state_id: (1, 2) for feature in features},
    }
    report = aggregate_independent_confirm_workers(
        features,
        assignments=assignments,
        worker_results=worker_results,
        independent_selections=selections,
        heuristic_selections=heuristics,
    )
    assert report["evaluation_performed"] is True
    assert report["fixed_state_denominator"] == 20
    assert len(report["records"]) == 20
    assert report["execution"]["operation_counts"]["teacher_forward_call_count"] == 340


def test_build_confirm_messages_uses_v21_n4_mixed_fidelity_contract() -> None:
    source_id = CONFIRM_SOURCE_IDS[0]
    payloads: dict[str, bytes] = {}
    requirements = []
    events = []
    for step in range(1, 6):
        path = f"images/{source_id}/{step}.png"
        payload = f"image-0-{step}".encode()
        payloads[path] = payload
        requirement = _requirement(0, step)
        requirements.append(requirement)
        low = LowFidelityEventV2(
            step_id=step,
            action_type="wait",
            action_argument="wait",
            foreground_app="fixture",
            screen_text_added=(),
            screen_text_removed=(),
            screen_change="none",
            executor_result="accepted",
        )
        serialized = serialize_low_fidelity_v2(low)
        events.append(
            {
                "step_id": step,
                "observation_after_path": path,
                "observation_after_sha256": _sha(payload),
                "low_fidelity_v2": {
                    "step_id": step,
                    "action_type": "wait",
                    "action_argument": "wait",
                    "foreground_app": "fixture",
                    "screen_text_added": [],
                    "screen_text_removed": [],
                    "screen_change": "none",
                    "executor_result": "accepted",
                },
                "low_fidelity_v2_serialized": serialized.decode(),
                "low_fidelity_v2_sha256": _sha(serialized),
            }
        )
    trajectory = {
        "source_id": source_id,
        "instruction": "wait",
        "events": events,
        "decisions": [
            {
                "state_id": f"{source_id}:decision_step:006",
                "decision_step_id": 6,
                "history_event_step_ids": [1, 2, 3, 4, 5],
                "candidate_event_step_ids": [1, 2, 3, 4],
                "current_equivalent_event_step_id": 5,
                "current_observation_path": f"images/{source_id}/5.png",
                "current_observation_sha256": _sha(payloads[f"images/{source_id}/5.png"]),
            }
        ],
    }
    validated = ValidatedConfirm20Payloads(
        artifact_tree_sha256="f" * 64,
        trajectories=(trajectory,),
        source_ids=(source_id,),
        ocr_records_by_path={},
        image_payloads_by_path=payloads,
        image_requirements=tuple(requirements),
        image_features_by_path={},
        transport_identity=FROZEN_TRANSPORT_IDENTITY,
        _seal=confirm_data._PAYLOAD_SEAL,
    )
    messages = build_confirm_messages(
        validated,
        _work_item(0),
        restored_event_step_ids=(1, 3),
        image_decoder=lambda payload: payload.decode(),
    )
    assert validate_gui_owl_v2_1_native_messages(messages) == 3


def test_strict_loader_binds_manifest_and_all_five_checkpoint_payloads() -> None:
    repo = "owner/model"
    payload_commit = "1" * 40
    manifest_commit = "2" * 40
    artifact = lambda path, revision, digest: ArtifactBinding(
        repository=repo,
        revision=revision,
        path=path,
        sha256=digest,
    )
    training = FrozenTrainingProvenance(
        family="independent",
        gate_config_sha256=GATE_V1_CONFIG_SHA256,
        formal_train_source_ids_sha256=FORMAL_TRAIN_SOURCE_IDS_SHA256,
        learning_rate=0.001,
        selected_epochs=(1, 2, 3, 4, 5),
        oof_selection_sha256="3" * 64,
        feature_artifact=artifact("train/features.jsonl", manifest_commit, "4" * 64),
        label_artifact=artifact("train/labels.jsonl", manifest_commit, "5" * 64),
        training_report_artifact=artifact("train/report.json", manifest_commit, "6" * 64),
    )
    checkpoint_payloads = [f"checkpoint-{seed}".encode() for seed in range(5)]
    checkpoint_provenance = tuple(
        SeedCheckpointProvenance(
            seed=seed,
            selected_epoch=seed + 1,
            model_state_sha256=f"{100 + seed:064x}",
            checkpoint_artifact=artifact(
                f"checkpoints/seed-{seed}.safetensors",
                payload_commit,
                _sha(checkpoint_payloads[seed]),
            ),
        )
        for seed in range(5)
    )
    provenance = FrozenEnsembleProvenance(training, checkpoint_provenance)
    manifest_payload = __import__("json").dumps(
        provenance.to_payload(), sort_keys=True, separators=(",", ":")
    ).encode()
    model_contract = {
        "repo": repo,
        "payload_commit": payload_commit,
        "ensemble_manifest": {
            "path": "manifests/independent.json",
            "sha256": _sha(manifest_payload),
            "provenance_sha256": provenance.sha256,
        },
        "checkpoints": [
            {
                "seed": seed,
                "selected_epoch": seed + 1,
                "path": f"checkpoints/seed-{seed}.safetensors",
                "sha256": _sha(payload),
                "size_bytes": len(payload),
                "model_state_sha256": f"{100 + seed:064x}",
            }
            for seed, payload in enumerate(checkpoint_payloads)
        ],
    }
    payload_map = {"manifests/independent.json": manifest_payload}
    payload_map.update(
        {
            f"checkpoints/seed-{seed}.safetensors": payload
            for seed, payload in enumerate(checkpoint_payloads)
        }
    )
    calls = []

    def loader(payload, **kwargs):
        calls.append((payload, kwargs))
        return kwargs["seed"]

    loaded = load_independent_ensemble(
        model_contract, payload_map, checkpoint_loader=loader
    )
    assert loaded.checkpoint_load_count == 5
    assert loaded.ensemble.models == (0, 1, 2, 3, 4)
    assert [call[1]["seed"] for call in calls] == list(range(5))
    with pytest.raises(ValueError, match="inventory drifted"):
        load_independent_ensemble(
            model_contract,
            {key: value for key, value in payload_map.items() if "seed-4" not in key},
            checkpoint_loader=loader,
        )
