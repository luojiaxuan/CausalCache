from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import causalcache.independent_confirm_execution as execution_module
import scripts.run_independent_confirm as confirm_cli

from causalcache.gate_v1_data import CandidateFeatures, FeatureState
from causalcache.gate_v1_training import FittedEnsemble
from causalcache.independent_confirm_artifact import (
    INDEPENDENT_DECISION_PATH,
    PAYLOAD_TARGETS,
    REPORT_TARGETS,
    RUN_MANIFEST_PATH,
    STATE_RECORDS_PATH,
    canonical_json_bytes,
    read_run_manifest,
    read_state_records_jsonl,
)
from causalcache.independent_confirm_contract import IndependentConfirmContract
from causalcache.independent_confirm_data import (
    CONFIRM_CANDIDATE_EVENT_STEP_IDS,
    CONFIRM_SOURCE_IDS,
    Confirm20ImageRequirement,
    Confirm20LabelBlindBundle,
    Confirm20PolicyWorkItem,
)
from causalcache.independent_confirm_execution import (
    PolicyVisionPhaseResult,
    RestorationPhaseResult,
    HuggingFaceConfirmPublisher,
    build_canonical_label_blind_payload,
    execute_independent_confirm,
    read_hf_token_file,
    run_four_worker_restoration,
    validate_policy_vision_phase,
    validate_restoration_phase,
)
from causalcache.independent_confirm_runner import (
    ConfirmWorkerResult,
    PayloadCommitReceipt,
    confirm_worker_assignments,
    independent_selection_payload_bytes,
    run_confirm_state_once,
    score_and_select_independent,
)
from causalcache.policy.gui_owl_v2_1 import (
    GUI_OWL_V2_1_FINAL_USER_INSTRUCTION,
    GUI_OWL_V2_1_SYSTEM_PROMPT,
    parse_gui_owl_v2_1_output,
)


WAIT_OUTPUT = (
    '<tool_call>\n{"name":"mobile_use","arguments":{"action":"wait"}}\n'
    "</tool_call>"
)
PAYLOAD_COMMIT = "2" * 40
REPORT_COMMIT = "3" * 40
TAG_OBJECT = "4" * 40
SOURCE_COMMIT = "5" * 40
SCIENTIFIC_CONTRACT_SHA256 = "6" * 64
GPU_UUIDS = tuple(
    f"GPU-00000000-0000-0000-0000-{index:012x}" for index in range(4)
)
TOPOLOGY_RECEIPT_PATH = "/evidence/independent-confirm-gpu-topology-smoke.json"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _feature(ordinal: int) -> FeatureState:
    source_id = CONFIRM_SOURCE_IDS[ordinal]
    return FeatureState(
        source_id=source_id,
        state_id=f"{source_id}:decision_step:006",
        decision_step_id=6,
        candidate_event_step_ids=CONFIRM_CANDIDATE_EVENT_STEP_IDS,
        q64=(0.0,) * 64,
        candidates=tuple(
            CandidateFeatures(
                event_step_id=event,
                h64=(float(event),) + (0.0,) * 63,
                g8=(float(event),) + (0.0,) * 7,
            )
            for event in CONFIRM_CANDIDATE_EVENT_STEP_IDS
        ),
    )


def _requirement(ordinal: int, event: int) -> Confirm20ImageRequirement:
    source_id = CONFIRM_SOURCE_IDS[ordinal]
    return Confirm20ImageRequirement(
        image_member_path=f"images/{source_id}/{event}.png",
        image_sha256=f"{ordinal * 10 + event:064x}",
        canonical_ocr_record_sha256=f"{ordinal * 10 + event + 1:064x}",
    )


def _work_item(ordinal: int) -> Confirm20PolicyWorkItem:
    source_id = CONFIRM_SOURCE_IDS[ordinal]
    return Confirm20PolicyWorkItem(
        ordinal=ordinal,
        source_id=source_id,
        state_id=f"{source_id}:decision_step:006",
        decision_step_id=6,
        event_images=tuple(
            (event, _requirement(ordinal, event))
            for event in CONFIRM_CANDIDATE_EVENT_STEP_IDS
        ),
        current_image=_requirement(ordinal, 5),
    )


def _score_record(ordinal: int) -> dict:
    source_id = CONFIRM_SOURCE_IDS[ordinal]
    return {
        "ordinal": ordinal,
        "source_id": source_id,
        "state_id": f"{source_id}:decision_step:006",
        "decision_step_id": 6,
        "candidate_event_step_ids": [1, 2, 3, 4],
        "scores_by_event_step": [
            {"event_step_id": 1, "score": 0.1},
            {"event_step_id": 2, "score": 0.4},
            {"event_step_id": 3, "score": 0.3},
            {"event_step_id": 4, "score": 0.2},
        ],
        "ranked_event_step_ids": [2, 3, 4, 1],
        "selected_event_step_ids": [2, 3],
    }


def _bundle() -> Confirm20LabelBlindBundle:
    features = tuple(_feature(ordinal) for ordinal in range(20))
    state_ids = tuple(feature.state_id for feature in features)
    return Confirm20LabelBlindBundle(
        feature_states=features,
        work_items=tuple(_work_item(ordinal) for ordinal in range(20)),
        image_requirements=(),
        image_payloads_by_path=MappingProxyType({}),
        dynamic_recent=MappingProxyType(
            {state_id: (3, 4) for state_id in state_ids}
        ),
        ocr_rgb_v2=MappingProxyType({state_id: (2, 3) for state_id in state_ids}),
        ocr_rgb_score_records=tuple(_score_record(ordinal) for ordinal in range(20)),
    )


def _ensemble() -> FittedEnsemble:
    return FittedEnsemble(
        family="independent",
        learning_rate=0.001,
        seeds=(0, 1, 2, 3, 4),
        selected_epochs=(1, 1, 1, 1, 1),
        selection_sha256="7" * 64,
        models=(0, 1, 2, 3, 4),
    )


def _scorer(features, ensemble):
    def predict(_model, vectors):
        scores = (0.9, -0.1, 0.4, 0.0)
        return tuple(scores[index % 4] for index in range(len(vectors)))

    return score_and_select_independent(features, ensemble, predict_fn=predict)


def _policy_vision(bundle: Confirm20LabelBlindBundle) -> PolicyVisionPhaseResult:
    records = []
    selections = {}
    for item in bundle.work_items:
        records.append(
            {
                "state": {
                    "ordinal": item.ordinal,
                    "source_id": item.source_id,
                    "state_id": item.state_id,
                },
                "worker_id": f"worker-{item.ordinal % 4}",
                "device": f"cuda:{item.ordinal % 4}",
                "gpu_uuid": GPU_UUIDS[item.ordinal % 4],
                "ordinal": item.ordinal,
                "source_id": item.source_id,
                "state_id": item.state_id,
                "decision_step_id": 6,
                "candidate_event_step_ids": [1, 2, 3, 4],
                "scores_by_event_step": [
                    {"event_step_id": 1, "score": 0.8},
                    {"event_step_id": 2, "score": 0.2},
                    {"event_step_id": 3, "score": 0.6},
                    {"event_step_id": 4, "score": 0.1},
                ],
                "ranked_event_step_ids": [1, 3, 2, 4],
                "selected_event_step_ids": [1, 3],
                "feature_repeats": 2,
                "same_device_replay": {
                    "performed": True,
                    "ranking_equal": True,
                    "selection_equal": True,
                    "max_abs_score_difference": 0.0,
                },
            }
        )
        selections[item.state_id] = (1, 3)
    return PolicyVisionPhaseResult(
        records=tuple(records),
        selections=MappingProxyType(selections),
        worker_metadata=tuple(
            {
                "worker_id": f"worker-{worker}",
                "state_count": 5,
                "device": f"cuda:{worker}",
                "gpu_uuid": GPU_UUIDS[worker],
            }
            for worker in range(4)
        ),
    )


def _native_messages(
    _item: Confirm20PolicyWorkItem, coalition: tuple[int, ...]
) -> list[dict]:
    mask = sum(1 << (event - 1) for event in coalition)
    content = [{"type": "text", "text": f"coalition_mask={mask}"}]
    content.extend(
        {"type": "image", "image": f"event-{event}"} for event in coalition
    )
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


class _Runtime:
    def generate_native_action(self, _messages):
        return SimpleNamespace(
            output_text=WAIT_OUTPUT,
            parsed_output=parse_gui_owl_v2_1_output(WAIT_OUTPUT),
            metadata={"fake": True},
        )

    def teacher_forced_distance_logits(self, messages_batch, actions):
        assert len(messages_batch) == len(actions) == 1
        raw = messages_batch[0][1]["content"][0]["text"]
        mask = int(raw.split("=", maxsplit=1)[1])
        return mask, {"coalition_mask": mask}


class _Distance:
    def prepare_reference(self, logits):
        assert logits == 15
        return logits

    def measure(self, reference, candidate):
        return SimpleNamespace(
            value=abs(reference - candidate) / 100.0,
            audit={"full_tensor_host_transfers": 0},
        )


class _Publisher:
    def __init__(self, events: list[str], *, verify: bool = True) -> None:
        self.events = events
        self.verify = verify
        self.payload_seal = None
        self.independent_bytes = None

    def preflight_destination(self):
        self.events.append("destination-preflight")
        return {"base_commit": "1" * 40, "tag_count": 0}

    def bind_payload_seal(self, seal):
        assert self.events == ["destination-preflight", "policy-vision"]
        assert set(seal.files) == set(PAYLOAD_TARGETS)
        self.payload_seal = seal
        self.independent_bytes = seal.files[INDEPENDENT_DECISION_PATH]
        self.events.append("bind-payload")

    def publish_payload(self, files, inventory):
        assert self.payload_seal is not None and dict(files) == dict(self.payload_seal.files)
        assert len(files) == len(inventory) == 8
        self.events.append("payload-commit")
        return {
            "payload_commit": PAYLOAD_COMMIT,
            "inventory_sha256": _sha(
                canonical_json_bytes([dict(item) for item in inventory])
            ),
        }

    def verify_payload(self, commit, inventory):
        assert commit == PAYLOAD_COMMIT and len(inventory) == 8
        self.events.append("payload-fresh-replay")
        return self.verify

    def publish_report(self, report_files, *, payload_commit):
        assert payload_commit == PAYLOAD_COMMIT
        assert set(report_files) == set(REPORT_TARGETS)
        self.events.extend(
            ["report-direct-child", "annotated-tag", "report-fresh-replay"]
        )
        return {
            "base_commit": "1" * 40,
            "payload_commit": PAYLOAD_COMMIT,
            "report_commit": REPORT_COMMIT,
            "annotated_tag_object": TAG_OBJECT,
            "created_tag_count": 1,
            "byte_identical_fresh_replay": True,
        }


def _restoration_phase(bundle: Confirm20LabelBlindBundle, events: list[str]):
    item_by_ordinal = {item.ordinal: item for item in bundle.work_items}

    def run(assignments, receipt, run_contract_sha256):
        assert isinstance(receipt, PayloadCommitReceipt)
        assert events[-1] == "payload-fresh-replay"
        events.append("restoration-4x5")
        results = []
        for assignment in assignments:
            records = tuple(
                run_confirm_state_once(
                    work_item=item_by_ordinal[ordinal],
                    message_builder=_native_messages,
                    runtime=_Runtime(),
                    distance_backend=_Distance(),
                    payload_commit_receipt=receipt,
                    run_contract_sha256=run_contract_sha256,
                )
                for ordinal in assignment.ordinals
            )
            results.append(
                ConfirmWorkerResult(
                    worker_id=assignment.worker_id, state_records=records
                )
            )
        return RestorationPhaseResult(
            worker_results=tuple(results),
            worker_metadata=tuple(
                {
                    "worker_id": assignment.worker_id,
                    "state_count": 5,
                    "device": f"cuda:{index}",
                    "gpu_uuid": GPU_UUIDS[index],
                }
                for index, assignment in enumerate(assignments)
            ),
        )

    return run


def _contract(root: Path) -> IndependentConfirmContract:
    data = json.loads(
        (root / "code/configs/causalcache_independent_confirm_closed_loop_v1.json")
        .read_text()
    )
    return IndependentConfirmContract(
        data=data,
        sha256=SCIENTIFIC_CONTRACT_SHA256,
        repository_root=root,
        source_path=root
        / "code/configs/causalcache_independent_confirm_closed_loop_v1.json",
    )


def _topology_receipt(
    contract: IndependentConfirmContract,
    *,
    model_dir: str = "/models/gui-owl",
    snapshot_manifest: str = "/source/code/configs/gui_owl_1_5_8b_snapshot.json",
) -> dict:
    identities = execution_module._gpu_topology_expected_runtime_identities(contract)
    vision_counts = dict(execution_module._GPU_TOPOLOGY_VISION_COUNTS_PER_WORKER)
    teacher_counts = dict(execution_module._GPU_TOPOLOGY_TEACHER_COUNTS_PER_WORKER)

    def workers(phase: str, counts: dict[str, int]) -> list[dict]:
        return [
            {
                "worker_id": f"worker-{index}",
                "device": f"cuda:{index}",
                "gpu_uuid": GPU_UUIDS[index],
                "fixed_rgb_image_set_sha256": (
                    execution_module.GPU_TOPOLOGY_SMOKE_FIXED_RGB_IMAGE_SET_SHA256
                ),
                "runtime_identity": dict(identities[phase]),
                "operation_counts": dict(counts),
            }
            for index in range(4)
        ]

    return {
        "schema_version": execution_module.GPU_TOPOLOGY_SMOKE_SCHEMA_VERSION,
        "protocol_id": execution_module.GPU_TOPOLOGY_SMOKE_PROTOCOL_ID,
        "status": "PASS",
        "scope": dict(execution_module._GPU_TOPOLOGY_SMOKE_SCOPE),
        "inputs": {
            "model_dir": model_dir,
            "snapshot_manifest": snapshot_manifest,
            "fixed_rgb_image_count": 5,
            "fixed_rgb_image_set_sha256": (
                execution_module.GPU_TOPOLOGY_SMOKE_FIXED_RGB_IMAGE_SET_SHA256
            ),
        },
        "controls": {
            "phase_timeout_seconds": contract.data["runtime_contract"][
                "gpu_topology_smoke_phase_timeout_seconds"
            ],
            "worker_termination_grace_seconds": contract.data["runtime_contract"][
                "worker_termination_grace_seconds"
            ],
        },
        "allocation": [
            {
                "worker_id": f"worker-{index}",
                "device": f"cuda:{index}",
                "gpu_uuid": GPU_UUIDS[index],
            }
            for index in range(4)
        ],
        "phase_order": ["policy_vision_spawn", "teacher_forced_fork"],
        "phases": {
            "policy_vision_spawn": {
                "start_method": "spawn",
                "worker_count": 4,
                "image_count_per_worker": 5,
                "feature_repeats": 1,
                "operation_counts": {
                    key: value * 4 for key, value in vision_counts.items()
                },
                "workers": workers("policy_vision_spawn", vision_counts),
            },
            "teacher_forced_fork": {
                "start_method": "fork",
                "worker_count": 4,
                "image_count_per_worker": 5,
                "canonical_teacher_action": "wait",
                "operation_counts": {
                    key: value * 4 for key, value in teacher_counts.items()
                },
                "workers": workers("teacher_forced_fork", teacher_counts),
            },
        },
    }


def _run_contract_payload(
    contract: IndependentConfirmContract,
    *,
    argv: tuple[str, ...] | None = None,
) -> dict:
    topology_body = _topology_receipt(contract)
    topology_sha256 = _sha(canonical_json_bytes(topology_body) + b"\n")
    if argv is None:
        argv = (
            "--hf-token-file",
            "/secrets/hf_key.txt",
            "--gpu-topology-smoke-receipt",
            TOPOLOGY_RECEIPT_PATH,
            "--gpu-topology-smoke-receipt-sha256",
            topology_sha256,
        )
    runtime = {
        "host_alias": "hyper00",
        "host_hostname": "node-radixark-16-0000",
        "container_id": "8" * 64,
        "container_image_digest": "9" * 64,
        "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        "gpu_uuids": list(GPU_UUIDS),
        "worker_count": 4,
        "states_per_worker": 5,
        "policy_vision_feature_repeats": 2,
        "restoration_teacher_forwards_per_valid_state": 17,
        "policy_vision_phase_timeout_seconds": contract.data["runtime_contract"][
            "policy_vision_phase_timeout_seconds"
        ],
        "restoration_phase_timeout_seconds": contract.data["runtime_contract"][
            "restoration_phase_timeout_seconds"
        ],
        "worker_termination_grace_seconds": contract.data["runtime_contract"][
            "worker_termination_grace_seconds"
        ],
        "execution_argv": list(argv),
    }
    return {
        "schema_version": "1.0.0",
        "protocol_id": "causalcache_independent_confirm20_execution_contract_v1",
        "contract_sha256": contract.sha256,
        "source_a_commit": "a" * 40,
        "execution_b_commit": SOURCE_COMMIT,
        "runner_freeze": {
            "source_a_commit": "a" * 40,
            "contract_sha256": contract.sha256,
            "confirm_access_authorized": True,
            "restoration_access_requires_payload_commit": True,
            "sealed_test_requires_development_go": True,
        },
        "parent_artifact": {
            "artifact_tree_sha256": "b" * 64,
            "source_ids": list(CONFIRM_SOURCE_IDS),
            "state_count": 20,
            "policy_output_present": False,
            "restoration_output_present": False,
        },
        "independent_model": {
            "repo": contract.model["repo"],
            "payload_commit": contract.model["payload_commit"],
            "manifest_commit": contract.model["manifest_commit"],
            "checkpoint_count": 5,
            "retraining_performed": False,
        },
        "policy_snapshot": {
            "model_dir": "/models/gui-owl",
            "model_repo": contract.data["restoration_contract"]["policy_repo"],
            "model_revision": contract.data["restoration_contract"][
                "policy_revision"
            ],
            "snapshot_manifest_sha256": contract.data["restoration_contract"][
                "snapshot_manifest_sha256"
            ],
            "verified_model_file_count": execution_module.MODEL_FILE_COUNT,
            "verified_model_total_bytes": execution_module.MODEL_TOTAL_BYTES,
        },
        "gpu_topology_smoke_receipt": {
            "path": TOPOLOGY_RECEIPT_PATH,
            "sha256": topology_sha256,
            "body": topology_body,
        },
        "runtime": runtime,
        "publication": dict(contract.destination),
        "prohibited_counts": {
            "retry_count": 0,
            "top_up_count": 0,
            "filter_count": 0,
            "gate_training_count": 0,
            "threshold_update_count": 0,
            "sealed_androidworld_test_access_count": 0,
        },
        "execution_argv_sha256": _sha(canonical_json_bytes(list(argv))),
    }


def test_gpu_topology_receipt_validation_binds_exact_runtime_and_allocation() -> None:
    root = Path(__file__).resolve().parents[2]
    contract = _contract(root)
    receipt = _topology_receipt(contract)
    validated = execution_module.validate_gpu_topology_smoke_receipt(
        receipt,
        contract=contract,
        expected_devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        expected_gpu_uuids=GPU_UUIDS,
        expected_model_dir="/models/gui-owl",
        expected_snapshot_manifest=(
            "/source/code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
    )
    receipt["status"] = "FAIL"
    assert validated["status"] == "PASS"
    assert validated["phase_order"] == [
        "policy_vision_spawn",
        "teacher_forced_fork",
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(status="FAIL"), "identity, status, or scope"),
        (
            lambda value: value["scope"].update(network_accessed=True),
            "identity, status, or scope",
        ),
        (
            lambda value: value["allocation"][0].update(device="cuda:3"),
            "allocation",
        ),
        (
            lambda value: value["controls"].update(phase_timeout_seconds=1),
            "runtime controls",
        ),
        (
            lambda value: value.update(
                phase_order=["teacher_forced_fork", "policy_vision_spawn"]
            ),
            "identity, status, or scope",
        ),
        (
            lambda value: value["phases"]["policy_vision_spawn"][
                "operation_counts"
            ].update(policy_vision_feature_forward_count=3),
            "order or counts",
        ),
        (
            lambda value: value["phases"]["teacher_forced_fork"]["workers"][0][
                "runtime_identity"
            ].update(snapshot_manifest_sha256="0" * 64),
            "identity or counts",
        ),
    ],
)
def test_gpu_topology_receipt_validation_rejects_tamper(mutate, message: str) -> None:
    root = Path(__file__).resolve().parents[2]
    contract = _contract(root)
    receipt = _topology_receipt(contract)
    mutate(receipt)
    with pytest.raises(ValueError, match=message):
        execution_module.validate_gpu_topology_smoke_receipt(
            receipt,
            contract=contract,
            expected_devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
            expected_gpu_uuids=GPU_UUIDS,
        )


def test_gpu_topology_receipt_file_requires_canonical_outside_bytes_and_sha(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    body = {"protocol_id": "test", "status": "PASS"}
    payload = canonical_json_bytes(body) + b"\n"
    receipt = evidence_root / "receipt.json"
    receipt.write_bytes(payload)
    digest = _sha(payload)
    resolved, observed, observed_digest = confirm_cli._load_gpu_topology_smoke_receipt(
        receipt,
        repository_root=repository_root,
        expected_sha256=digest,
    )
    assert resolved == receipt
    assert observed == body
    assert observed_digest == digest

    with pytest.raises(ValueError, match="absolute path"):
        confirm_cli._load_gpu_topology_smoke_receipt(
            Path("receipt.json"),
            repository_root=repository_root,
            expected_sha256=digest,
        )
    with pytest.raises(ValueError, match="differs from explicit argv"):
        confirm_cli._load_gpu_topology_smoke_receipt(
            receipt,
            repository_root=repository_root,
            expected_sha256="0" * 64,
        )
    pretty = evidence_root / "pretty.json"
    pretty.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        confirm_cli._load_gpu_topology_smoke_receipt(
            pretty,
            repository_root=repository_root,
            expected_sha256=_sha(pretty.read_bytes()),
        )
    duplicate = evidence_root / "duplicate.json"
    duplicate.write_bytes(b'{"status":"PASS","status":"PASS"}\n')
    with pytest.raises(ValueError, match="strict JSON"):
        confirm_cli._load_gpu_topology_smoke_receipt(
            duplicate,
            repository_root=repository_root,
            expected_sha256=_sha(duplicate.read_bytes()),
        )
    alias = evidence_root / "alias.json"
    alias.symlink_to(receipt)
    with pytest.raises(ValueError, match="canonical and non-symlink"):
        confirm_cli._load_gpu_topology_smoke_receipt(
            alias,
            repository_root=repository_root,
            expected_sha256=digest,
        )
    inside = repository_root / "receipt.json"
    inside.write_bytes(payload)
    with pytest.raises(ValueError, match="outside the Git repository"):
        confirm_cli._load_gpu_topology_smoke_receipt(
            inside,
            repository_root=repository_root,
            expected_sha256=digest,
        )


def test_cpu_e2e_enforces_preflight_eight_then_four_and_replay(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    bundle = _bundle()
    events: list[str] = []
    publisher = _Publisher(events)
    contract = _contract(root)
    run_contract = _run_contract_payload(contract)

    def vision(value):
        assert events == ["destination-preflight"]
        events.append("policy-vision")
        return _policy_vision(value)

    result = execute_independent_confirm(
        contract=contract,
        bundle=bundle,
        ensemble=_ensemble(),
        run_contract=run_contract,
        output_dir=tmp_path / "formal-output",
        publisher=publisher,
        policy_vision_phase=vision,
        restoration_phase=_restoration_phase(bundle, events),
        independent_scorer=_scorer,
    )

    expected_decisions = independent_selection_payload_bytes(
        _scorer(bundle.feature_states, _ensemble())
    )
    assert publisher.independent_bytes == expected_decisions
    assert events == [
        "destination-preflight",
        "policy-vision",
        "bind-payload",
        "payload-commit",
        "payload-fresh-replay",
        "restoration-4x5",
        "report-direct-child",
        "annotated-tag",
        "report-fresh-replay",
    ]
    assert result.completion["payload_file_count"] == 8
    assert result.completion["report_file_count"] == 4
    assert result.completion["report_parent_commit"] == PAYLOAD_COMMIT
    records = read_state_records_jsonl(result.report_files[STATE_RECORDS_PATH])
    assert len(records) == 20
    assert [record["state"]["ordinal"] for record in records] == list(range(20))
    selection_sha256 = _sha(result.payload_files[INDEPENDENT_DECISION_PATH])
    run_contract_sha256 = _sha(canonical_json_bytes(run_contract))
    assert {record["selection_sha256"] for record in records} == {selection_sha256}
    assert {record["run_contract_sha256"] for record in records} == {
        run_contract_sha256
    }
    manifest = read_run_manifest(
        result.report_files[RUN_MANIFEST_PATH], payload_commit=PAYLOAD_COMMIT
    )
    assert manifest["contract_sha256"] == SCIENTIFIC_CONTRACT_SHA256
    assert (
        manifest["runtime_metadata"]["run_contract_sha256"]
        == run_contract_sha256
    )
    assert manifest["runtime_metadata"]["run_contract"]["runtime"][
        "execution_argv"
    ] == list(run_contract["runtime"]["execution_argv"])
    assert b"hf_test_value" not in b"".join(result.report_files.values())
    assert set((tmp_path / "formal-output").rglob("*.json"))


def test_destination_preflight_precedes_policy_and_failed_replay_blocks_restoration(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    bundle = _bundle()
    events: list[str] = []
    publisher = _Publisher(events, verify=False)
    contract = _contract(root)

    def vision(value):
        assert events == ["destination-preflight"]
        events.append("policy-vision")
        return _policy_vision(value)

    with pytest.raises(ValueError, match="fresh replay verification"):
        execute_independent_confirm(
            contract=contract,
            bundle=bundle,
            ensemble=_ensemble(),
            run_contract=_run_contract_payload(contract),
            output_dir=tmp_path / "blocked-output",
            publisher=publisher,
            policy_vision_phase=vision,
            restoration_phase=lambda *_args: pytest.fail(
                "restoration ran before an immutable payload replay"
            ),
            independent_scorer=_scorer,
        )
    assert events[-1] == "payload-fresh-replay"
    assert (tmp_path / "blocked-output" / "failure.json").is_file()


def test_hf_token_reader_requires_private_regular_file(tmp_path: Path) -> None:
    token = tmp_path / "hf-token"
    token.write_text("hf_test_value\n")
    token.chmod(0o600)
    assert read_hf_token_file(token) == "hf_test_value"
    token.chmod(0o644)
    with pytest.raises(ValueError, match="0400 or 0600"):
        read_hf_token_file(token)
    token.chmod(0o600)
    alias = tmp_path / "hf-token-link"
    alias.symlink_to(token)
    with pytest.raises(ValueError, match="missing or unsafe"):
        read_hf_token_file(alias)


def test_payload_publication_rejects_preflight_base_toctou(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    independent = _scorer(bundle.feature_states, _ensemble())
    seal = build_canonical_label_blind_payload(
        bundle=bundle,
        independent=independent,
        policy_vision=validate_policy_vision_phase(
            _policy_vision(bundle), bundle=bundle
        ),
    )

    class Api:
        def __init__(self):
            self.main = "1" * 40
            self.create_commit_count = 0

        def create_repo(self, *_args, **_kwargs):
            return None

        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(private=True, sha=self.main)

        def list_repo_files(self, *_args, **_kwargs):
            return ["README.md"]

        def list_repo_refs(self, *_args, **_kwargs):
            return SimpleNamespace(tags=[])

        def create_commit(self, *_args, **_kwargs):
            self.create_commit_count += 1
            pytest.fail("payload commit ran after the preflight base moved")

    api = Api()
    publisher = object.__new__(HuggingFaceConfirmPublisher)
    publisher._api = api
    publisher._cache_dir = tmp_path
    publisher._download_fn = None
    publisher._operation_factory = lambda **_kwargs: object()
    publisher._payload_receipt = None
    publisher._payload_seal = None
    publisher._preflight_base_commit = None
    publisher._token = "redacted"
    assert publisher.preflight_destination()["base_commit"] == "1" * 40
    publisher.bind_payload_seal(seal)
    api.main = "9" * 40
    with pytest.raises(ValueError, match="non-conflicting base"):
        publisher.publish_payload(seal.files, seal.inventory)
    assert api.create_commit_count == 0


def test_core_rejects_payload_commit_tamper_in_raw_state(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    contract = _contract(root)
    bundle = _bundle()
    events: list[str] = []
    base_phase = _restoration_phase(bundle, events)

    def tampered(assignments, receipt, run_contract_sha256):
        result = base_phase(assignments, receipt, run_contract_sha256)
        first = result.worker_results[0]
        records = list(first.state_records)
        records[0] = {**dict(records[0]), "payload_commit": "f" * 40}
        workers = (
            ConfirmWorkerResult(
                worker_id=first.worker_id, state_records=tuple(records)
            ),
            *result.worker_results[1:],
        )
        return RestorationPhaseResult(
            worker_results=workers,
            worker_metadata=result.worker_metadata,
        )

    with pytest.raises(ValueError, match="raw state record payload commit"):
        execute_independent_confirm(
            contract=contract,
            bundle=bundle,
            ensemble=_ensemble(),
            run_contract=_run_contract_payload(contract),
            output_dir=tmp_path / "payload-tamper",
            publisher=_Publisher(events),
            policy_vision_phase=lambda value: (
                events.append("policy-vision") or _policy_vision(value)
            ),
            restoration_phase=tampered,
            independent_scorer=_scorer,
        )


def test_worker_metadata_and_device_format_tamper_are_rejected() -> None:
    bundle = _bundle()
    policy = _policy_vision(bundle)
    bad_policy_metadata = [dict(item) for item in policy.worker_metadata]
    bad_policy_metadata[0]["gpu_uuid"] = GPU_UUIDS[1]
    with pytest.raises(ValueError, match="metadata/device binding"):
        validate_policy_vision_phase(
            PolicyVisionPhaseResult(
                records=policy.records,
                selections=policy.selections,
                worker_metadata=tuple(bad_policy_metadata),
            ),
            bundle=bundle,
        )

    assignments = confirm_worker_assignments(bundle.work_items)
    worker_results = tuple(
        ConfirmWorkerResult(
            worker_id=assignment.worker_id,
            state_records=tuple({} for _ in range(5)),
        )
        for assignment in assignments
    )
    restoration_metadata = tuple(
        {
            "worker_id": assignment.worker_id,
            "state_count": 5,
            "device": f"cuda:{index}",
            "gpu_uuid": GPU_UUIDS[0] if index == 1 else GPU_UUIDS[index],
        }
        for index, assignment in enumerate(assignments)
    )
    with pytest.raises(ValueError, match="device identities must be unique"):
        validate_restoration_phase(
            RestorationPhaseResult(
                worker_results=worker_results,
                worker_metadata=restoration_metadata,
            ),
            assignments=assignments,
        )
    with pytest.raises(ValueError, match="explicit CUDA devices and GPU UUIDs"):
        run_four_worker_restoration(
            assignments=assignments,
            receipt=None,
            run_contract_sha256="a" * 64,
            payloads=None,
            work_items=bundle.work_items,
            model_dir="/model",
            snapshot_manifest="/snapshot.json",
            devices=("0", "1", "2", "3"),
            gpu_uuids=GPU_UUIDS,
            phase_timeout_seconds=43_200,
            worker_termination_grace_seconds=30,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(protocol_id="wrong"), "identity"),
        (lambda value: value.update(contract_sha256="0" * 64), "identity"),
        (
            lambda value: value["runtime"].update(worker_count=3),
            "four-by-five runtime",
        ),
        (
            lambda value: value["runtime"].update(
                policy_vision_phase_timeout_seconds=1
            ),
            "four-by-five runtime",
        ),
        (
            lambda value: value["prohibited_counts"].update(retry_count=1),
            "prohibited operation",
        ),
        (
            lambda value: value.update(execution_argv_sha256="0" * 64),
            "four-by-five runtime",
        ),
        (
            lambda value: value["gpu_topology_smoke_receipt"].update(
                sha256="0" * 64
            ),
            "topology receipt binding",
        ),
    ],
)
def test_core_rejects_run_contract_tamper_before_preflight(
    tmp_path: Path, mutate, message: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    contract = _contract(root)
    run_contract = copy.deepcopy(_run_contract_payload(contract))
    mutate(run_contract)
    events: list[str] = []
    with pytest.raises(ValueError, match=message):
        execute_independent_confirm(
            contract=contract,
            bundle=_bundle(),
            ensemble=_ensemble(),
            run_contract=run_contract,
            output_dir=tmp_path / f"run-contract-{message}",
            publisher=_Publisher(events),
            policy_vision_phase=pytest.fail,
            restoration_phase=pytest.fail,
            independent_scorer=_scorer,
        )
    assert events == []


def test_worker_barrier_timeout_terminates_and_joins_all_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        def __init__(self, name: str) -> None:
            self.name = name
            self.exitcode = None
            self.alive = True
            self.terminate_count = 0
            self.join_count = 0

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminate_count += 1
            self.alive = False
            self.exitcode = -15

        def join(self, timeout=None):
            assert timeout is not None and timeout >= 0
            self.join_count += 1

    processes = tuple(Process(f"worker-{index}") for index in range(4))
    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(
        execution_module.time,
        "monotonic",
        lambda: next(ticks, 2.0),
    )
    with pytest.raises(TimeoutError, match="policy-vision phase exceeded 1 seconds"):
        execution_module._collect_process_responses(
            processes=processes,
            queue=SimpleNamespace(),
            phase_label="policy-vision phase",
            phase_timeout_seconds=1,
            termination_grace_seconds=1,
        )
    assert [process.terminate_count for process in processes] == [1, 1, 1, 1]
    assert all(process.join_count >= 1 for process in processes)
    assert all(not process.is_alive() for process in processes)


def test_cli_preflights_destination_before_model_or_policy_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[2]
    events: list[str] = []
    contract = SimpleNamespace()
    parent = SimpleNamespace()
    bundle = object()
    ensemble = object()

    class Publisher:
        def __init__(self, **_kwargs):
            events.append("publisher-created")

        def preflight_destination(self):
            events.append("destination-preflight")

    monkeypatch.setattr(
        confirm_cli,
        "IndependentConfirmContract",
        SimpleNamespace(load=lambda *_args, **_kwargs: contract),
    )
    monkeypatch.setattr(
        confirm_cli,
        "_load_runner_authorization",
        lambda **_kwargs: {"execution_b_commit": SOURCE_COMMIT},
    )
    monkeypatch.setattr(
        confirm_cli,
        "_validate_runtime_args",
        lambda *_args, **_kwargs: (
            ("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
            GPU_UUIDS,
        ),
    )
    topology_body = {"status": "PASS"}
    topology_path = tmp_path / "outside" / "topology.json"
    topology_sha256 = "d" * 64

    def load_topology(*_args, **_kwargs):
        events.append("topology-receipt-read")
        return topology_path, topology_body, topology_sha256

    def validate_topology(*_args, **_kwargs):
        assert events == ["topology-receipt-read"]
        events.append("topology-receipt-validated")
        return topology_body

    def decode_parent(**_kwargs):
        assert events == [
            "topology-receipt-read",
            "topology-receipt-validated",
        ]
        events.append("confirm-semantic-decode")
        return parent

    monkeypatch.setattr(
        confirm_cli, "_load_gpu_topology_smoke_receipt", load_topology
    )
    monkeypatch.setattr(
        confirm_cli, "validate_gpu_topology_smoke_receipt", validate_topology
    )
    monkeypatch.setattr(confirm_cli, "_formal_parent_view", decode_parent)

    def verify_model(**_kwargs):
        assert events[-1] == "confirm-semantic-decode"
        events.append("model-snapshot-access")
        return (
            root / "code/configs/gui_owl_1_5_8b_snapshot.json",
            {"verified": True},
        )

    monkeypatch.setattr(
        confirm_cli,
        "_model_snapshot_identity",
        verify_model,
    )

    def build_run_contract(**kwargs):
        assert kwargs["gpu_topology_smoke_receipt"] == {
            "path": str(topology_path),
            "sha256": topology_sha256,
            "body": topology_body,
        }
        return {
            "execution_b_commit": SOURCE_COMMIT,
            "runtime": {
                "policy_vision_phase_timeout_seconds": 7_200,
                "restoration_phase_timeout_seconds": 43_200,
                "worker_termination_grace_seconds": 30,
            },
        }

    monkeypatch.setattr(confirm_cli, "_run_contract", build_run_contract)

    def read_token(_path):
        assert events[-1] == "model-snapshot-access"
        events.append("hf-token-read")
        return "secret"

    monkeypatch.setattr(confirm_cli, "read_hf_token_file", read_token)
    monkeypatch.setattr(confirm_cli, "HuggingFaceConfirmPublisher", Publisher)

    def download(_contract, *, publisher):
        assert events == [
            "topology-receipt-read",
            "topology-receipt-validated",
            "confirm-semantic-decode",
            "model-snapshot-access",
            "hf-token-read",
            "publisher-created",
            "destination-preflight",
        ]
        events.append("model-download")
        return SimpleNamespace(checkpoint_load_count=5, ensemble=ensemble)

    monkeypatch.setattr(confirm_cli, "download_and_load_independent_ensemble", download)
    monkeypatch.setattr(
        confirm_cli, "bundle_from_validated_payloads", lambda _payloads: bundle
    )
    monkeypatch.setattr(
        confirm_cli, "make_real_policy_vision_phase", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        confirm_cli, "make_real_restoration_phase", lambda **_kwargs: object()
    )

    def run_core(**kwargs):
        assert events == [
            "topology-receipt-read",
            "topology-receipt-validated",
            "confirm-semantic-decode",
            "model-snapshot-access",
            "hf-token-read",
            "publisher-created",
            "destination-preflight",
            "model-download",
        ]
        assert kwargs["publisher"].__class__ is Publisher
        events.append("core-policy-output-boundary")
        return SimpleNamespace(completion={"status": "complete"})

    monkeypatch.setattr(confirm_cli, "execute_independent_confirm", run_core)
    outside = tmp_path / "outside"
    outside.mkdir()
    args = SimpleNamespace(
        repository_root=root,
        contract=root
        / "code/configs/causalcache_independent_confirm_closed_loop_v1.json",
        runner_freeze=root
        / "code/configs/causalcache_independent_confirm_runner_v1.json",
        execution_b_git_commit=SOURCE_COMMIT,
        parent_root=outside / "parent",
        generator_source_root=root,
        model_dir=outside / "model",
        ocr_model_dir=outside / "ocr-model",
        ocr_wheel_dir=outside / "ocr-wheel",
        hf_cache_dir=outside / "cache",
        output_dir=outside / "formal-output",
        hf_token_file=outside / "token",
        gpu_topology_smoke_receipt=topology_path,
        gpu_topology_smoke_receipt_sha256=topology_sha256,
        host_alias="hyper00",
        host_hostname="node-radixark-16-0000",
        container_id="a" * 64,
        container_image_digest="ignored",
        devices="cuda:0,cuda:1,cuda:2,cuda:3",
        gpu_uuids=",".join(GPU_UUIDS),
        execution_argv=(
            "--gpu-topology-smoke-receipt",
            str(topology_path),
            "--gpu-topology-smoke-receipt-sha256",
            topology_sha256,
        ),
    )
    assert confirm_cli.execute(args) == {"status": "complete"}
    assert events[-1] == "core-policy-output-boundary"
