from __future__ import annotations

import hashlib
import io
import inspect
import json
import copy
import platform
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import causalcache.gate_v1_fresh16_evaluation_runner as runner
from causalcache.gate_v1_contract import canonical_json_bytes
from causalcache.gate_v1_data import CandidateFeatures, FeatureState
from causalcache.gate_v1_fresh16 import (
    HEURISTIC_STATUS,
    LEARNED_STATUS,
    Fresh16ImageRequirement,
    Fresh16PolicyWorkItem,
    feature_states_jsonl_bytes,
    jsonl_score_records_bytes,
    policy_selection_record,
    selection_artifact_bytes,
)
from causalcache.gate_v1_fresh16_evaluation_contract import (
    Fresh16EvaluationContract,
)
from causalcache.gate_v1_provenance import (
    FORMAL_TRAIN_SOURCE_IDS_SHA256,
    GATE_V1_CONFIG_SHA256,
    ArtifactBinding,
    FrozenEnsembleProvenance,
    FrozenTrainingProvenance,
    SeedCheckpointProvenance,
    canonical_selection_sha256,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _work_items() -> tuple[Fresh16PolicyWorkItem, ...]:
    result = []
    for ordinal in range(48):
        count = 2 + ordinal % 3
        identities = tuple(
            (
                step,
                Fresh16ImageRequirement(
                    image_member_path=f"images/{ordinal:02d}-{step}.png",
                    image_sha256=f"{ordinal * 10 + step:064x}",
                    canonical_ocr_record_sha256=f"{ordinal * 20 + step:064x}",
                ),
            )
            for step in range(1, count + 1)
        )
        current = Fresh16ImageRequirement(
            image_member_path=f"images/{ordinal:02d}-current.png",
            image_sha256=f"{ordinal * 10 + 9:064x}",
            canonical_ocr_record_sha256=f"{ordinal * 20 + 9:064x}",
        )
        result.append(
            Fresh16PolicyWorkItem(
                ordinal=ordinal,
                source_id=f"source-{ordinal // 3:02d}",
                state_id=f"state-{ordinal:02d}",
                decision_step_id=count + 2,
                event_images=identities,
                current_image=current,
            )
        )
    return tuple(result)


def _policy_record(
    item: Fresh16PolicyWorkItem,
    *,
    worker: str,
    device: str,
    gpu: str,
):
    return policy_selection_record(
        item,
        {step: float(10 - step) for step in item.candidate_event_step_ids},
        worker_id=worker,
        device=device,
        gpu_uuid=gpu,
    )


def _worker_results(*, same_sentinel_gpu: bool = False):
    items = _work_items()
    even, odd, sentinel = runner.policy_worker_partitions(items)
    even_records = tuple(
        _policy_record(item, worker="even", device="cuda:0", gpu="GPU-even")
        for item in even
    )
    odd_records = tuple(
        _policy_record(item, worker="odd", device="cuda:1", gpu="GPU-odd")
        for item in odd
    )
    replay = _policy_record(
        sentinel,
        worker="odd",
        device="cuda:1",
        gpu="GPU-even" if same_sentinel_gpu else "GPU-odd",
    )
    return items, sentinel, (
        runner.PolicyWorkerResult(
            worker_id="even",
            records=even_records,
            sentinel_record=None,
            operation_counts={
                "image_processor_batch_count": 24,
                "policy_vision_feature_forward_count": 48,
                "policy_vision_cosine_scalar_transfer_count": 144,
                "policy_vision_image_decode_count": 96,
                "generation_count": 0,
            },
            runtime_metadata={},
        ),
        runner.PolicyWorkerResult(
            worker_id="odd",
            records=odd_records,
            sentinel_record=replay,
            operation_counts={
                "image_processor_batch_count": 25,
                "policy_vision_feature_forward_count": 49,
                "policy_vision_cosine_scalar_transfer_count": 148,
                "policy_vision_image_decode_count": 101,
                "generation_count": 0,
            },
            runtime_metadata={},
        ),
    )


def _feature_states() -> tuple[FeatureState, ...]:
    states = []
    for ordinal in range(48):
        count = 2 + ordinal % 3
        candidates = tuple(range(1, count + 1))
        states.append(
            FeatureState(
                source_id=f"source-{ordinal // 3:02d}",
                state_id=f"state-{ordinal:02d}",
                decision_step_id=count + 2,
                candidate_event_step_ids=candidates,
                q64=(0.0,) * 64,
                candidates=tuple(
                    CandidateFeatures(
                        event_step_id=step,
                        h64=(float(step),) * 64,
                        g8=(0.0,) * 8,
                    )
                    for step in candidates
                ),
            )
        )
    return tuple(states)


def _label_blind_files() -> dict[str, bytes]:
    states = _feature_states()
    score_records = []
    policy_records = []
    score_selections = {}
    dynamic = {}
    conditional_decisions = {}
    independent_decisions = {}
    for ordinal, state in enumerate(states):
        scores = {step: float(10 - step) for step in state.candidate_event_step_ids}
        selected = (1, 2)
        score_selections[state.state_id] = selected
        dynamic[state.state_id] = state.candidate_event_step_ids[-2:]
        selected_before = []
        trace = []
        for round_index, selected_event in enumerate(selected):
            remaining = [
                event
                for event in state.candidate_event_step_ids
                if event not in selected_before
            ]
            selected_after = sorted((*selected_before, selected_event))
            trace.append(
                {
                    "round": round_index,
                    "selected_before": list(selected_before),
                    "candidate_predicted_marginal_gains": [
                        {
                            "event_step_id": event,
                            "predicted_marginal_gain": scores[event],
                        }
                        for event in remaining
                    ],
                    "decision": "add",
                    "selected_event_step_id": selected_event,
                    "selected_after": selected_after,
                }
            )
            selected_before = selected_after
        seeds = [
            {"seed": seed, "selected_event_step_ids": list(selected)}
            for seed in range(5)
        ]
        conditional_decisions[state.state_id] = {
            "ensemble_selected_event_step_ids": list(selected),
            "ensemble_trace": trace,
            "seed_selected_event_step_ids": seeds,
        }
        independent_decisions[state.state_id] = {
            "ensemble_selected_event_step_ids": list(selected),
            "seed_selected_event_step_ids": seeds,
        }
        rows = [
            {"event_step_id": step, "score": scores[step]}
            for step in state.candidate_event_step_ids
        ]
        common = {
            "schema_version": "1.0.0",
            "protocol_id": runner.PROTOCOL_ID,
            "ordinal": ordinal,
            "source_id": state.source_id,
            "state_id": state.state_id,
            "decision_step_id": state.decision_step_id,
            "candidate_event_step_ids": list(state.candidate_event_step_ids),
            "scores_by_event_step": rows,
            "ranked_event_step_ids": list(state.candidate_event_step_ids),
            "selected_event_step_ids": list(selected),
        }
        score_records.append(
            {"status": "FROZEN_GATE_V1_FRESH16_OCR_RGB_SCORES_V1", **common}
        )
        policy_records.append(
            {
                "status": "FROZEN_GATE_V1_FRESH16_POLICY_VISION_SCORES_V1",
                **common,
                "worker_id": "even" if ordinal % 2 == 0 else "odd",
                "device": "cuda:0" if ordinal % 2 == 0 else "cuda:1",
                "gpu_uuid": "GPU-even" if ordinal % 2 == 0 else "GPU-odd",
            }
        )
    return {
        runner.FEATURE_PAYLOAD_PATH: feature_states_jsonl_bytes(states),
        runner.HEURISTIC_PATHS["dynamic_recent"]: selection_artifact_bytes(
            "dynamic_recent", dynamic, status=HEURISTIC_STATUS
        ),
        runner.HEURISTIC_PATHS["ocr_rgb_v2"]: selection_artifact_bytes(
            "ocr_rgb_v2", score_selections, status=HEURISTIC_STATUS
        ),
        runner.HEURISTIC_PATHS["policy_vision_v3"]: selection_artifact_bytes(
            "policy_vision_v3", score_selections, status=HEURISTIC_STATUS
        ),
        runner.OCR_SCORE_PATH: jsonl_score_records_bytes(score_records),
        runner.POLICY_SCORE_PATH: jsonl_score_records_bytes(policy_records),
        runner.LEARNED_PATHS["conditional"]: runner.learned_decision_artifact_bytes(
            "conditional", conditional_decisions
        ),
        runner.LEARNED_PATHS["independent"]: runner.learned_decision_artifact_bytes(
            "independent", independent_decisions
        ),
    }


def _artifact(path: str, sha: str, *, revision: str) -> ArtifactBinding:
    return ArtifactBinding(
        repository="owner/model",
        revision=revision,
        path=path,
        sha256=sha,
    )


def _provenance(
    family: str,
    checkpoint_records,
    *,
    checkpoint_revision: str,
) -> FrozenEnsembleProvenance:
    epochs = tuple(item["selected_epoch"] for item in checkpoint_records)
    training = FrozenTrainingProvenance(
        family=family,
        gate_config_sha256=GATE_V1_CONFIG_SHA256,
        formal_train_source_ids_sha256=FORMAL_TRAIN_SOURCE_IDS_SHA256,
        learning_rate=0.0003,
        selected_epochs=epochs,
        oof_selection_sha256=("1" if family == "conditional" else "2") * 64,
        feature_artifact=_artifact("feature", "3" * 64, revision="a" * 40),
        label_artifact=_artifact("label", "4" * 64, revision="a" * 40),
        training_report_artifact=_artifact("report", "5" * 64, revision="a" * 40),
    )
    return FrozenEnsembleProvenance(
        training=training,
        checkpoints=tuple(
            SeedCheckpointProvenance(
                seed=item["seed"],
                selected_epoch=item["selected_epoch"],
                model_state_sha256=item["model_state_sha256"],
                checkpoint_artifact=_artifact(
                    item["path"],
                    item["sha256"],
                    revision=checkpoint_revision,
                ),
            )
            for item in checkpoint_records
        ),
    )


def _model_contract_and_payloads(*, bad_revision: bool = False):
    payload_commit = "a" * 40
    manifest_commit = "b" * 40
    payloads = {}
    checkpoints = []
    provenances = {}
    manifests = []
    for family_index, family in enumerate(("conditional", "independent")):
        family_records = []
        for seed in range(5):
            payload = f"{family}-{seed}".encode()
            path = f"checkpoints/{family}-{seed}.safetensors"
            record = {
                "family": family,
                "seed": seed,
                "selected_epoch": seed + 1,
                "path": path,
                "sha256": _sha(payload),
                "size_bytes": len(payload),
                "model_state_sha256": f"{family_index * 10 + seed + 10:064x}",
            }
            family_records.append(record)
            checkpoints.append(record)
            payloads[path] = payload
        provenance = _provenance(
            family,
            family_records,
            checkpoint_revision=manifest_commit if bad_revision else payload_commit,
        )
        provenances[family] = provenance
        manifest_payload = canonical_json_bytes({"family": family}) + b"\n"
        manifest_path = f"manifests/{family}.json"
        payloads[manifest_path] = manifest_payload
        manifests.append(
            {
                "family": family,
                "path": manifest_path,
                "sha256": _sha(manifest_payload),
                "provenance_sha256": provenance.sha256,
            }
        )
    contract = SimpleNamespace(
        model={
            "repo": "owner/model",
            "payload_commit": payload_commit,
            "manifest_commit": manifest_commit,
            "ensemble_manifests": manifests,
            "checkpoints": checkpoints,
        }
    )
    return contract, payloads, provenances


def test_policy_worker_partition_and_exact_operation_counts():
    items, sentinel, results = _worker_results()
    records, selections, counts = runner.validate_policy_worker_results(
        results, work_items=items, sentinel=sentinel
    )
    assert len(records) == len(selections) == 48
    assert counts["policy_vision_processor_batch_count"] == 49
    assert counts["policy_vision_feature_forward_count"] == 97
    assert counts["policy_vision_cosine_scalar_transfer_count"] == 292
    assert counts["policy_vision_image_decode_count"] == 197


def test_cross_device_sentinel_rejects_same_gpu_uuid():
    items, sentinel, results = _worker_results(same_sentinel_gpu=True)
    with pytest.raises(ValueError, match="cross-device"):
        runner.validate_policy_worker_results(
            results, work_items=items, sentinel=sentinel
        )


def test_heuristic_seal_is_immutable_and_claim_rebinds_payload():
    files = _label_blind_files()
    seal = runner.seal_label_blind_files(files)
    with pytest.raises(TypeError):
        seal.files[runner.OCR_SCORE_PATH] = b"changed"
    claim = runner.claim_label_access(
        seal,
        source_git_commit="a" * 40,
        contract_sha256="b" * 64,
    )
    payload = {**files, runner.LABEL_PAYLOAD_PATH: b"labels"}
    runner.validate_claim_against_payload(claim, seal, payload)
    payload[runner.OCR_SCORE_PATH] += b"\n"
    with pytest.raises(ValueError, match="changed after"):
        runner.validate_claim_against_payload(claim, seal, payload)


def test_heuristic_seal_rejects_score_selection_contradiction():
    files = _label_blind_files()
    lines = files[runner.OCR_SCORE_PATH].splitlines()
    first = json.loads(lines[0])
    first["scores_by_event_step"][0]["score"] = -100.0
    lines[0] = canonical_json_bytes(first)
    files[runner.OCR_SCORE_PATH] = b"\n".join(lines) + b"\n"
    with pytest.raises(ValueError, match="contradicts"):
        runner.seal_label_blind_files(files)


def test_heuristic_seal_recomputes_dynamic_recent_from_feature_order():
    files = _label_blind_files()
    selections = {
        state.state_id: state.candidate_event_step_ids[-2:]
        for state in _feature_states()
    }
    selections["state-00"] = (1,)
    files[runner.HEURISTIC_PATHS["dynamic_recent"]] = selection_artifact_bytes(
        "dynamic_recent", selections, status=HEURISTIC_STATUS
    )
    with pytest.raises(ValueError, match="dynamic-recent"):
        runner.seal_label_blind_files(files)


def test_heuristic_seal_rejects_learned_trace_score_order_mutation():
    files = _label_blind_files()
    path = runner.LEARNED_PATHS["conditional"]
    artifact = json.loads(files[path])
    first_round = artifact["records"][0]["ensemble_trace"][0]
    first_round["candidate_predicted_marginal_gains"][0][
        "predicted_marginal_gain"
    ] = -100.0
    files[path] = canonical_json_bytes(artifact) + b"\n"
    with pytest.raises(ValueError, match="contradicts"):
        runner.seal_label_blind_files(files)


def test_heuristic_seal_accepts_explicit_terminal_stop_round():
    files = _label_blind_files()
    path = runner.LEARNED_PATHS["conditional"]
    artifact = json.loads(files[path])
    first = artifact["records"][0]
    first["ensemble_selected_event_step_ids"] = []
    first["ensemble_trace"] = [
        {
            "round": 0,
            "selected_before": [],
            "candidate_predicted_marginal_gains": [
                {"event_step_id": 1, "predicted_marginal_gain": -1.0},
                {"event_step_id": 2, "predicted_marginal_gain": -2.0},
            ],
            "decision": "stop",
            "selected_event_step_id": None,
            "selected_after": [],
        }
    ]
    selections = {
        record["state_id"]: tuple(record["ensemble_selected_event_step_ids"])
        for record in artifact["records"]
    }
    artifact["selection_sha256"] = canonical_selection_sha256(selections)
    files[path] = canonical_json_bytes(artifact) + b"\n"
    assert runner.seal_label_blind_files(files).inventory_sha256


def test_post_claim_primary_builder_has_no_model_or_selector_input():
    parameters = inspect.signature(runner._build_primary_sealed_outputs).parameters
    assert "ensembles" not in parameters
    assert "conditional_scores" not in parameters
    assert "independent_scores" not in parameters


def test_prefixed_tar_member_is_required():
    member = b"raw-state-bytes\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path, payload in (
            ("prefix/raw_states.jsonl", member),
            ("prefix/other.json", b"{}"),
        ):
            info = tarfile.TarInfo(path)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    payload = buffer.getvalue()
    assert (
        runner.extract_exact_tar_member(
            payload,
            expected_archive_sha256=_sha(payload),
            expected_archive_size_bytes=len(payload),
            expected_member_count=2,
            member_path="prefix/raw_states.jsonl",
            expected_member_sha256=_sha(member),
            expected_member_size_bytes=len(member),
        )
        == member
    )
    with pytest.raises(ValueError, match="missing"):
        runner.extract_exact_tar_member(
            payload,
            expected_archive_sha256=_sha(payload),
            expected_archive_size_bytes=len(payload),
            expected_member_count=2,
            member_path="raw_states.jsonl",
            expected_member_sha256=_sha(member),
            expected_member_size_bytes=len(member),
        )


def test_checkpoint_provenance_binds_payload_commit(monkeypatch):
    contract, payloads, provenances = _model_contract_and_payloads()
    monkeypatch.setattr(
        runner,
        "frozen_ensemble_provenance_from_manifest",
        lambda value: provenances[value["family"]],
    )
    monkeypatch.setattr(
        runner,
        "load_safetensors_checkpoint",
        lambda payload, **kwargs: object(),
    )
    loaded = runner.load_formal_ensembles(contract, payloads)
    assert loaded.checkpoint_load_count == 10

    bad_contract, bad_payloads, bad_provenances = _model_contract_and_payloads(
        bad_revision=True
    )
    monkeypatch.setattr(
        runner,
        "frozen_ensemble_provenance_from_manifest",
        lambda value: bad_provenances[value["family"]],
    )
    with pytest.raises(ValueError, match="binding drifted"):
        runner.load_formal_ensembles(bad_contract, bad_payloads)


def test_model_backed_replay_rejects_self_consistent_report_provenance_mutation():
    loaded = runner.FormalDecisionProvenance(
        conditional="loaded-conditional",
        independent="loaded-independent",
    )
    assert runner._bind_loaded_formal_provenance(loaded, loaded) is loaded
    mutated_report = runner.FormalDecisionProvenance(
        conditional="self-rehashed-mutated-conditional",
        independent="loaded-independent",
    )
    with pytest.raises(ValueError, match="loaded models"):
        runner._bind_loaded_formal_provenance(mutated_report, loaded)


def test_public_input_repo_profile_and_exact_inventory():
    class Api:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(private=False, sha="a" * 40)

        def list_repo_files(self, *_args, **_kwargs):
            return ["weights", "README.md"]

    runner.validate_input_repo(
        Api(),
        repo="public/model",
        repo_type="model",
        revision="a" * 40,
        expected_paths=("weights",),
        expected_private=False,
    )
    with pytest.raises(ValueError, match="privacy"):
        runner.validate_input_repo(
            Api(),
            repo="public/model",
            repo_type="model",
            revision="a" * 40,
            expected_paths=("weights",),
            expected_private=True,
        )


def test_private_formal_input_accepts_exact_sixteen_tree_only():
    exact = tuple(f"artifact-{index}" for index in range(16))

    class Api:
        def __init__(self, extra=()):
            self.extra = tuple(extra)

        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(private=True, sha="a" * 40)

        def list_repo_files(self, *_args, **_kwargs):
            return [*exact, *self.extra, "README.md"]

    runner.validate_input_repo(
        Api(),
        repo="private/model",
        repo_type="model",
        revision="a" * 40,
        expected_paths=exact,
        expected_private=True,
    )
    with pytest.raises(ValueError, match="inventory"):
        runner.validate_input_repo(
            Api(("unexpected",)),
            repo="private/model",
            repo_type="model",
            revision="a" * 40,
            expected_paths=exact,
            expected_private=True,
        )


def test_real_frozen_completion_derives_exact_sixteen_model_tree():
    root = Path(__file__).resolve().parents[2]
    config_path = root / "code/configs/causalcache_gate_v1_fresh16_evaluation_v1.json"
    payload = config_path.read_bytes()
    contract = Fresh16EvaluationContract(
        data=json.loads(payload),
        sha256=_sha(payload),
        repository_root=root,
        source_path=config_path,
    )
    inventory = runner.formal_model_remote_inventory(contract)
    assert len(inventory) == len(set(inventory)) == 16
    assert sum("checkpoints/" in path for path in inventory) == 10
    assert sum("full-oof-report" in path for path in inventory) == 2


def test_commit_files_rejects_nonexact_stage_inventory():
    contract = SimpleNamespace(
        output={
            "payload_commit": {"commit_title": "payload"},
            "report_commit": {"commit_title": "report"},
        },
        destination={"repo": "owner/repo"},
    )
    with pytest.raises(ValueError, match="inventory"):
        runner.commit_files(
            api=object(),
            operation_factory=lambda **kwargs: kwargs,
            contract=contract,
            files={runner.FEATURE_PAYLOAD_PATH: b"x"},
            parent="a" * 40,
            title="payload",
        )


def test_operation_counts_require_exact_frozen_key_value_map():
    contract = SimpleNamespace(
        data={
            "execution_planned_operation_contract": {
                "policy_vision_processor_batch_count": 49,
                "policy_vision_feature_forward_count": 97,
                "policy_vision_cosine_scalar_transfer_count": 292,
            }
        }
    )
    counts = dict(contract.data["execution_planned_operation_contract"])
    runner.validate_operation_counts(contract, counts)
    with pytest.raises(ValueError, match="denominator"):
        runner.validate_operation_counts(
            contract, {**counts, "image_processor_batch_count": 49}
        )
    wrong = dict(counts)
    wrong["policy_vision_feature_forward_count"] = 96
    with pytest.raises(ValueError, match="denominator"):
        runner.validate_operation_counts(contract, wrong)


def _manifest_schema_fixtures():
    run = {
        key: None for key in runner._RUN_MANIFEST_KEYS
    }
    run.update(
        {
            "schema_version": runner.SCHEMA_VERSION,
            "protocol_id": runner.PROTOCOL_ID,
            "status": runner.FINAL_STATUS,
            "gate_trained": True,
            "fresh16_primary_evaluated": True,
            "legacy_dev5_semantic_decode_count": 0,
            "confirm20_access_count": 0,
            "matched_nll_evaluation_count": 0,
            "closed_loop_episode_count": 0,
        }
    )
    bundle = {key: None for key in runner._BUNDLE_MANIFEST_KEYS}
    bundle.update(
        {
            "schema_version": runner.SCHEMA_VERSION,
            "protocol_id": runner.PROTOCOL_ID,
            "status": "FROZEN_GATE_V1_FRESH16_BUNDLE_MANIFEST_V1",
            "final_target_count": 13,
            "report_commit_embedded": False,
        }
    )
    return run, bundle


@pytest.mark.parametrize(
    ("target", "mutation"),
    (
        ("run", lambda value: value.update(extra=True)),
        ("run", lambda value: value.update(gate_trained=False)),
        ("run", lambda value: value.update(fresh16_primary_evaluated=False)),
        ("bundle", lambda value: value.update(extra=True)),
        ("bundle", lambda value: value.update(report_commit_embedded=True)),
        ("bundle", lambda value: value.update(status="tampered")),
    ),
)
def test_run_and_bundle_manifest_exact_schema_rejects_mutations(target, mutation):
    run, bundle = _manifest_schema_fixtures()
    value = run if target == "run" else bundle
    mutation(value)
    with pytest.raises(ValueError, match="schema|status"):
        (
            runner._validate_run_manifest_schema(value)
            if target == "run"
            else runner._validate_bundle_manifest_schema(value)
        )


def test_run_manifest_source_and_runtime_are_exact_content_identities():
    files = _label_blind_files()
    source = runner.SourceIdentity(
        head="a" * 40,
        remote_main="a" * 40,
        branch="main",
        origin_url="origin",
        source_inventory=({"path": "code.py", "git_blob": "b" * 40},),
        loaded_module_inventory=({"module": "module", "sha256": "c" * 64},),
    )
    runtime = runner.RuntimeIdentity(
        payload={"container": {"image_id": "sha256:" + "d" * 64}},
        receipt={"status": "frozen", "container": {"id": "e" * 64}},
        receipt_sha256="f" * 64,
    )
    run = {
        "source": runner._source_record(source),
        "runtime_provenance": runner._run_runtime_provenance(runtime, files),
    }
    runner._validate_run_source_and_runtime_identity(
        run,
        source=source,
        runtime_identity=runtime,
        payload_files=files,
    )
    source_mutation = json.loads(json.dumps(run))
    source_mutation["source"]["source_inventory"][0]["git_blob"] = "0" * 40
    with pytest.raises(ValueError, match="source"):
        runner._validate_run_source_and_runtime_identity(
            source_mutation,
            source=source,
            runtime_identity=runtime,
            payload_files=files,
        )
    runtime_mutation = json.loads(json.dumps(run))
    runtime_mutation["runtime_provenance"]["validated_runtime_identity"][
        "container"
    ]["image_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="runtime"):
        runner._validate_run_source_and_runtime_identity(
            runtime_mutation,
            source=source,
            runtime_identity=runtime,
            payload_files=files,
        )


def _runtime_validation_fixture(tmp_path, monkeypatch):
    import importlib.metadata
    import sys
    import types

    thread_state = {"threads": 1, "interop": 1, "deterministic": False}
    torch = types.ModuleType("torch")
    torch.__version__ = "test-torch"
    torch.set_num_threads = lambda value: thread_state.update(threads=value)
    torch.get_num_threads = lambda: thread_state["threads"]
    torch.set_num_interop_threads = lambda value: thread_state.update(interop=value)
    torch.get_num_interop_threads = lambda: thread_state["interop"]
    torch.use_deterministic_algorithms = lambda value: thread_state.update(
        deterministic=value
    )
    torch.are_deterministic_algorithms_enabled = lambda: thread_state[
        "deterministic"
    ]
    torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
    module_versions = {
        "torch": torch,
        "safetensors": types.ModuleType("safetensors"),
        "huggingface_hub": types.ModuleType("huggingface_hub"),
        "transformers": types.ModuleType("transformers"),
    }
    module_versions["safetensors"].__version__ = "test-safetensors"
    module_versions["huggingface_hub"].__version__ = "test-hf"
    module_versions["transformers"].__version__ = "test-transformers"
    for name, module in module_versions.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "test-pillow")

    container_id = "1" * 64
    image_id = "sha256:" + "2" * 64
    source = runner.SourceIdentity(
        head="3" * 40,
        remote_main="3" * 40,
        branch="main",
        origin_url="origin",
        source_inventory=(),
        loaded_module_inventory=(),
    )
    specification = {
        "schema_version": "1.0.0",
        "status": "CAPTURED_GATE_V1_FRESH16_DOCKER_INSPECT_V1",
        "path": "/data/runtime.json",
        "mode": 0o600,
        "container_name_prefix": "sglang-omni-jaxan-",
        "runtime": "runc",
        "data_mount_destination": "/data",
        "data_mount_rw": True,
        "privileged": False,
        "gpu_device_request_count": 1,
        "gpu_count": 2,
    }
    runtime = {
        "host": "hyper00_or_hyper01_h200",
        "container_image_id": image_id,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "safetensors_version": "test-safetensors",
        "huggingface_hub_version": "test-hf",
        "transformers_version": "test-transformers",
        "pillow_version": "test-pillow",
        "docker_inspect_receipt": specification,
        "finalize_phase": {
            "device": "cpu",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "gpu_required": False,
        },
        "thread_environment": {},
    }
    contract = SimpleNamespace(data={"runtime_contract": runtime}, sha256="4" * 64)
    receipt = {
        "schema_version": "1.0.0",
        "protocol_id": runner.PROTOCOL_ID,
        "status": specification["status"],
        "contract_sha256": contract.sha256,
        "source": runner._source_record(source),
        "capture_host": "node-radixark-16-0000",
        "container": {
            "id": container_id,
            "name": "sglang-omni-jaxan-07171234",
            "hostname": container_id[:12],
            "image_id": image_id,
            "running": True,
        },
        "runtime": {
            "privileged": False,
            "runtime": "runc",
            "device_request_count": 1,
            "device_ids": ["0", "1"],
            "gpu_count": 2,
        },
        "data_mount": {
            "type": "bind",
            "source": "/data02/jaxan",
            "destination": "/data",
            "rw": True,
        },
        "evaluation_executed": False,
        "execution_authorized": True,
    }
    receipt_path = tmp_path / "runtime.json"
    monkeypatch.setattr(runner.socket, "gethostname", lambda: container_id[:12])
    return contract, source, receipt, receipt_path


def test_runtime_receipt_exact_schema_and_identity(tmp_path, monkeypatch):
    contract, source, receipt, receipt_path = _runtime_validation_fixture(
        tmp_path, monkeypatch
    )
    receipt_path.write_bytes(runner.pretty_json_bytes(receipt))
    receipt_path.chmod(0o600)
    identity = runner.validate_execution_runtime(
        contract,
        source=source,
        data_root=tmp_path,
        receipt_path=receipt_path,
    )
    assert identity.receipt == receipt
    assert identity.payload["data_mount"]["source"] == "/data02/jaxan"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update(extra=True),
        lambda value: value["container"].update(name="bad"),
        lambda value: value["container"].update(hostname="bad"),
        lambda value: value["runtime"].update(device_ids=["0", "0"]),
        lambda value: value["data_mount"].update(destination="/wrong"),
    ),
)
def test_runtime_receipt_mutations_fail_closed(tmp_path, monkeypatch, mutation):
    contract, source, receipt, receipt_path = _runtime_validation_fixture(
        tmp_path, monkeypatch
    )
    mutation(receipt)
    receipt_path.write_bytes(runner.pretty_json_bytes(receipt))
    receipt_path.chmod(0o600)
    with pytest.raises(ValueError, match="identity"):
        runner.validate_execution_runtime(
            contract,
            source=source,
            data_root=tmp_path,
            receipt_path=receipt_path,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (b'[{"HostConfig":{},"HostConfig":{}}]', "duplicate JSON key"),
        (b'[{"HostConfig":NaN}]', "non-finite JSON constant"),
    ),
)
def test_docker_capture_rejects_nonstrict_json(
    tmp_path, monkeypatch, payload, message
):
    specification = {
        "container_name_prefix": "sglang-omni-jaxan-",
    }
    contract = SimpleNamespace(
        data={
            "runtime_contract": {
                "host": "hyper00_or_hyper01_h200",
                "docker_inspect_receipt": specification,
            }
        }
    )
    monkeypatch.setattr(
        runner.socket, "gethostname", lambda: "node-radixark-16-0000"
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=payload,
            stderr=b"",
        ),
    )
    with pytest.raises(ValueError, match=message):
        runner.capture_docker_inspect_receipt(
            contract,
            source=object(),
            container_name="sglang-omni-jaxan-07171234",
            host_data_root=tmp_path,
        )


def test_evaluation_rejects_in_memory_states_not_replayed_payload(monkeypatch):
    files = {path: b"placeholder" for path in runner.PAYLOAD_TARGETS}
    files[runner.FEATURE_PAYLOAD_PATH] = feature_states_jsonl_bytes(_feature_states())
    contract = SimpleNamespace(
        geometry={"source_ids": [f"source-{index:02d}" for index in range(16)]},
        data={"execution_planned_operation_contract": {"count": 1}},
        destination={"repo": "owner/repo"},
    )
    monkeypatch.setattr(runner, "read_label_states_jsonl", lambda _payload: ("labels",))
    monkeypatch.setattr(
        runner,
        "join_feature_and_label_states",
        lambda *_args, **_kwargs: ("replayed-state",),
    )
    with pytest.raises(ValueError, match="in-memory states differ"):
        runner.evaluate_and_build_report_files(
            contract,
            states=("caller-state",),
            formal_provenance=object(),
            payload_files=files,
            payload_commit="a" * 40,
            source=object(),
            operation_counts={"count": 1},
            runtime_identity=object(),
        )


def _run(command, cwd):
    subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.PIPE)


def test_source_a_b_direct_child_unique_diff(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    _run(["git", "init", "--bare", str(remote)], tmp_path)
    _run(["git", "init", "-b", "main", str(work)], tmp_path)
    _run(["git", "config", "user.email", "test@example.com"], work)
    _run(["git", "config", "user.name", "Test"], work)
    (work / "source.txt").write_text("source-a\n")
    _run(["git", "add", "source.txt"], work)
    _run(["git", "commit", "-m", "source a"], work)
    _run(["git", "remote", "add", "origin", str(remote)], work)
    _run(["git", "push", "-u", "origin", "main"], work)
    source_a = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=work).decode().strip()
    contract = SimpleNamespace(
        repository_root=work,
        sha256="c" * 64,
        source={
            "branch": "main",
            "origin_name": "origin",
            "origin_url": str(remote),
            "required_source_a_paths": ["source.txt"],
            "git_prerequisites": [],
        },
    )
    monkeypatch.setattr(
        runner,
        "_loaded_module_inventory",
        lambda _root, _commit: (
            {
                "module": "causalcache.fake",
                "path": "code/causalcache/fake.py",
                "sha256": "d" * 64,
                "size_bytes": 1,
            },
        ),
    )
    assert runner.validate_source_a(
        contract, expected_source_a_git_commit=source_a
    )["remote_mutation_count"] == 0
    runner.materialize_runner_freeze(
        contract, expected_source_a_git_commit=source_a
    )
    _run(["git", "add", runner.RUNNER_FREEZE_B_PATH], work)
    _run(["git", "commit", "-m", "execution b"], work)
    _run(["git", "push", "origin", "main"], work)
    source_b = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=work).decode().strip()
    observed = runner.validate_execution_b_source(
        contract, expected_execution_b_git_commit=source_b
    )
    assert observed.head == source_b


def test_durable_state_chain_rejects_skip_and_detects_prefix(tmp_path):
    stages = tuple(f"stage_{index}" for index in range(16))
    chain = runner.DurableStateChain(tmp_path, stages)
    with pytest.raises(ValueError, match="skip"):
        chain.append(stages[1], {"value": 1})
    first = chain.append(stages[0], {"value": 0})
    assert len(chain.verify_prefix()) == 1
    assert chain.append(stages[0], {"value": 0}) == first
    with pytest.raises(ValueError, match="differs"):
        chain.append(stages[0], {"value": 9})


def test_absent_remote_claim_is_durable_before_create_mutation(tmp_path, monkeypatch):
    class RepositoryNotFoundError(Exception):
        pass

    monkeypatch.setattr(
        runner,
        "_repo_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RepositoryNotFoundError()),
    )

    class Api:
        def create_repo(self, *_args, **_kwargs):
            raise RuntimeError("simulated crash during first remote mutation")

    receipt = tmp_path / "pre-mutation.json"
    contract = SimpleNamespace(destination={"repo": "owner/repo"})
    with pytest.raises(RuntimeError, match="simulated crash"):
        runner._ensure_empty_destination_base(
            Api(), contract, pre_mutation_receipt_path=receipt
        )
    value = json.loads(receipt.read_bytes())
    assert value["repository_existed"] is False
    assert value["captured_before_first_mutation"] is True
    assert value["base_commit"] is None


def test_execute_validate_requires_finalize_runtime_but_no_gpu_arguments(
    tmp_path, monkeypatch
):
    source = runner.SourceIdentity(
        head="a" * 40,
        remote_main="a" * 40,
        branch="main",
        origin_url="origin",
        source_inventory=(),
        loaded_module_inventory=(),
    )
    monkeypatch.setattr(runner, "validate_execution_b_source", lambda *_args, **_kwargs: source)
    runtime_identity = runner.RuntimeIdentity(
        payload={"device": "cpu"},
        receipt={"status": "frozen"},
        receipt_sha256="b" * 64,
    )
    monkeypatch.setattr(
        runner,
        "validate_execution_runtime",
        lambda *_args, **_kwargs: runtime_identity,
    )
    monkeypatch.setattr(
        runner,
        "validate_completed_remote",
        lambda **kwargs: {
            "status": runner.VALIDATE_STATUS,
            "remote_mutation_count": 0,
            "runtime_forwarded": kwargs["runtime_identity"] is runtime_identity,
        },
    )
    result = runner.execute_fresh16_evaluation(
        mode="validate",
        contract=SimpleNamespace(),
        api=object(),
        download_fn=lambda **_kwargs: "",
        operation_factory=lambda **kwargs: kwargs,
        expected_execution_b_git_commit="a" * 40,
        data_root=tmp_path,
        fresh_download_parent=tmp_path,
        model_dir=Path("."),
        snapshot_manifest=Path("."),
        docker_inspect_receipt=tmp_path / "runtime.json",
        devices=(),
        gpu_uuids=(),
    )
    assert result["remote_mutation_count"] == 0
    assert result["runtime_forwarded"] is True


def test_execute_validate_rejects_missing_finalize_runtime_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "validate_execution_b_source",
        lambda *_args, **_kwargs: object(),
    )
    with pytest.raises(ValueError, match="run and validate"):
        runner.execute_fresh16_evaluation(
            mode="validate",
            contract=SimpleNamespace(),
            api=object(),
            download_fn=lambda **_kwargs: "",
            operation_factory=lambda **kwargs: kwargs,
            expected_execution_b_git_commit="a" * 40,
            data_root=tmp_path,
            fresh_download_parent=tmp_path,
            model_dir=Path("."),
            snapshot_manifest=Path("."),
            docker_inspect_receipt=None,
            devices=(),
            gpu_uuids=(),
        )


def test_validate_completed_remote_calls_two_commit_validator_with_exact_signature(
    tmp_path, monkeypatch
):
    report_commit, payload_commit, base_commit = ("a" * 40, "b" * 40, "c" * 40)
    files = tuple(sorted((*runner.PAYLOAD_TARGETS, *runner.REPORT_TARGETS)))
    monkeypatch.setattr(runner, "_repo_snapshot", lambda *_args, **_kwargs: (report_commit, files))
    monkeypatch.setattr(
        runner,
        "_commit_history",
        lambda *_args, **_kwargs: (
            (report_commit, "report"),
            (payload_commit, "payload"),
            (base_commit, "base"),
        ),
    )
    observed = {}

    def stop_after_signature(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        raise RuntimeError("signature boundary reached")

    monkeypatch.setattr(runner, "validate_two_commit_chain", stop_after_signature)
    source = runner.SourceIdentity(
        head="d" * 40,
        remote_main="d" * 40,
        branch="main",
        origin_url="origin",
        source_inventory=(),
        loaded_module_inventory=(),
    )
    runtime = runner.RuntimeIdentity(
        payload={"device": "cpu"},
        receipt={"status": "frozen"},
        receipt_sha256="e" * 64,
    )
    contract = SimpleNamespace(destination={"repo": "owner/repo", "tag": "v1"})
    with pytest.raises(RuntimeError, match="signature boundary"):
        runner.validate_completed_remote(
            api=object(),
            download_fn=lambda **_kwargs: "",
            contract=contract,
            fresh_parent=tmp_path,
            runtime_identity=runtime,
            source=source,
        )
    assert set(observed["kwargs"]) == {
        "base_commit",
        "payload_commit",
        "report_commit",
    }
