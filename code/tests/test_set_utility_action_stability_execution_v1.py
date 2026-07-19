from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

import causalcache.set_utility_action_stability_execution_v1 as execution
from causalcache.set_utility_action_stability_diagnostic_v1 import (
    AUTO_FRESH_ENCODE_CONDITION,
    AUTO_FROZEN_ENCODED_CONDITION,
    EAGER_FROZEN_ENCODED_CONTROL,
    state_ids_for_worker_v1,
)
from causalcache.set_utility_action_stability_execution_v1 import (
    ActionStabilityLaunchV1,
    ActionStabilityRuntimeBundleV1,
    PROFILE_AUTO,
    PROFILE_EAGER,
    load_parent_artifact_binding_v1,
    run_action_stability_worker_v1,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _launch(tmp_path: Path, *, profile: str, worker_index: int = 1) -> ActionStabilityLaunchV1:
    repository = tmp_path / "repository"
    processor = tmp_path / "processor"
    model = tmp_path / "model"
    for path in (repository, processor, model):
        path.mkdir(exist_ok=True)
    return ActionStabilityLaunchV1(
        repository_root=repository,
        source_config_path=execution.CANONICAL_D1_CONFIG_PATH,
        execution_envelope_path=tmp_path / "fresh-envelope.json",
        parent_envelope_path=tmp_path / "parent-envelope.json",
        processor_root=processor,
        model_dir=model,
        output_root=tmp_path / "fresh-run",
        worker_index=worker_index,
        profile=profile,
        device="cuda:0",
        worker_gpu_uuid=GPU_UUID,
        fresh_envelope_sha256=SHA_A,
        source_config_sha256=SHA_B,
        source_inventory_sha256=SHA_C,
        parent_envelope_sha256=execution.PARENT_EXECUTION_ENVELOPE_SHA256,
        fresh_validation_status=execution.FRESH_ENVELOPE_VALIDATION_STATUS,
        freshness_validated=True,
        gpu_execution_authorized=True,
    )


def _condition(condition_id: str) -> dict[str, Any]:
    prepared = condition_id != AUTO_FRESH_ENCODE_CONDITION
    return {
        "canonical_action_equal": True,
        "condition_id": condition_id,
        "decoded_output_equal": True,
        "encode_call_count": 1 if prepared else 2,
        "encoded_input_unchanged_after": True if prepared else None,
        "encoded_input_unchanged_before": True if prepared else None,
        "encoded_input_unchanged_between": True if prepared else None,
        "exact_generated_sequence_equal": True,
        "failure_class": None,
        "generation_call_count": 2,
        "generation_completed_count": 2,
        "metric_safe": True,
        "repeat_count": 2,
    }


def _partial(profile: str, state_id: str) -> dict[str, Any]:
    if profile == PROFILE_AUTO:
        profile_id = "auto_default"
        conditions = [
            _condition(AUTO_FRESH_ENCODE_CONDITION),
            _condition(AUTO_FROZEN_ENCODED_CONDITION),
        ]
    else:
        profile_id = "eager_numerical_control_not_strict_cuda_determinism"
        conditions = [_condition(EAGER_FROZEN_ENCODED_CONTROL)]
    return {
        "conditions": conditions,
        "metric_safe": True,
        "profile": profile_id,
        "state_id": state_id,
    }


def _dependencies(
    launch: ActionStabilityLaunchV1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calls: dict[str, Any] = {
        "runtime_profiles": [],
        "runner_states": [],
        "semantic_calls": 0,
    }
    source_contract = types.SimpleNamespace(
        repository_root=launch.repository_root,
        config_sha256=launch.source_config_sha256,
        data={"source": {"inventory_sha256": launch.source_inventory_sha256}},
    )
    validated = types.SimpleNamespace(
        candidate_schedule_bytes=b"candidate",
        freeze_manifest_bytes=b"freeze",
        model_inventory_sha256=SHA_A,
        processor_inventory_sha256=SHA_B,
    )
    parent_contract = object()
    expected = state_ids_for_worker_v1(launch.worker_index)
    parent_states = [
        *(
            types.SimpleNamespace(
                state_id=state_id,
                query=types.SimpleNamespace(state_id=state_id),
                joined_input=f"joined:{state_id}",
            )
            for state_id in expected
        ),
        types.SimpleNamespace(
            state_id="0047881550315557:decision:018",
            query=object(),
            joined_input=object(),
        ),
    ]

    def source_loader(**kwargs: Any) -> Any:
        assert kwargs == {
            "repository_root": launch.repository_root,
            "config_path": execution.CANONICAL_D1_CONFIG_PATH,
        }
        return source_contract

    def parent_loader(current: ActionStabilityLaunchV1) -> tuple[Any, Any]:
        assert current is launch
        return parent_contract, validated

    def semantic_loader(current: Any, contract: Any) -> list[Any]:
        assert current is validated
        assert contract is parent_contract
        calls["semantic_calls"] += 1
        return parent_states

    runtime = object()

    def builder_for(joined: object) -> Any:
        return lambda plan: (joined, plan)

    def runtime_factory(current: Any, artifacts: Any, contract: Any) -> Any:
        assert current is launch
        assert artifacts is validated
        assert contract is parent_contract
        calls["runtime_profiles"].append(current.profile)
        return ActionStabilityRuntimeBundleV1(
            runtime=runtime,
            reference_input_builder_for=builder_for,
            runtime_identity_sha256=SHA_C,
        )

    def runner(query: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["runtime"] is runtime
        assert callable(kwargs["reference_input_builder"])
        calls["runner_states"].append(query.state_id)
        return _partial(launch.profile, query.state_id)

    dependencies = {
        "source_contract_loader": source_loader,
        "parent_binding_loader": parent_loader,
        "semantic_loader": semantic_loader,
        "runtime_factory": runtime_factory,
        "profile_runner": runner,
        "stage_gate": lambda current: None,
    }
    return calls, dependencies


@pytest.mark.parametrize(
    (
        "profile",
        "expected_condition_count",
        "expected_encode_calls",
        "expected_generation_calls",
    ),
    ((PROFILE_AUTO, 4, 6, 8), (PROFILE_EAGER, 2, 2, 4)),
)
def test_profile_worker_filters_parent_states_and_writes_distinct_no_retry_terminals(
    tmp_path: Path,
    profile: str,
    expected_condition_count: int,
    expected_encode_calls: int,
    expected_generation_calls: int,
) -> None:
    launch = _launch(tmp_path, profile=profile)
    calls, dependencies = _dependencies(launch)

    terminal = run_action_stability_worker_v1(launch, **dependencies)

    expected_states = state_ids_for_worker_v1(1)
    assert calls["semantic_calls"] == 1
    assert calls["runtime_profiles"] == [profile]
    assert calls["runner_states"] == list(expected_states)
    assert terminal["status"] == "COMPLETED_ACTION_STABILITY_PROFILE_WORKER_V1"
    assert terminal["failure_class"] is None
    assert terminal["state_ids"] == list(expected_states)
    assert terminal["counts"] == {
        "condition_completed_count": expected_condition_count,
        "encode_call_count": expected_encode_calls,
        "generation_call_count": expected_generation_calls,
        "label_count": 0,
        "partial_state_completed_count": 2,
        "partial_state_expected_count": 2,
        "restoration_distance_count": 0,
        "retry_count": 0,
        "teacher_forward_call_count": 0,
        "training_example_count": 0,
    }
    worker_root = tmp_path / "fresh-run" / "workers" / "worker-01"
    assert {path.name for path in worker_root.iterdir()} == {
        f"{profile}-attempt.json",
        f"{profile}-terminal.json",
    }
    assert json.loads((worker_root / f"{profile}-terminal.json").read_text()) == terminal
    serialized = json.dumps(terminal, sort_keys=True).lower()
    for forbidden in ("output_text", "coordinate", "token_ids", "logits"):
        assert forbidden not in serialized

    with pytest.raises(FileExistsError):
        run_action_stability_worker_v1(launch, **dependencies)
    assert calls["runtime_profiles"] == [profile]


def test_auto_and_eager_are_independent_process_claims_and_filenames(
    tmp_path: Path,
) -> None:
    terminals = []
    for profile in (PROFILE_AUTO, PROFILE_EAGER):
        launch = _launch(tmp_path, profile=profile)
        _, dependencies = _dependencies(launch)
        terminals.append(run_action_stability_worker_v1(launch, **dependencies))

    worker_root = tmp_path / "fresh-run" / "workers" / "worker-01"
    assert {path.name for path in worker_root.iterdir()} == {
        "auto-attempt.json",
        "auto-terminal.json",
        "eager-attempt.json",
        "eager-terminal.json",
    }
    assert [terminal["profile"] for terminal in terminals] == ["auto", "eager"]


def test_eager_default_stage_gate_fails_before_attempt_without_all_auto_terminals(
    tmp_path: Path,
) -> None:
    launch = _launch(tmp_path, profile=PROFILE_EAGER, worker_index=0)
    calls, dependencies = _dependencies(launch)
    dependencies.pop("stage_gate")

    with pytest.raises(ValueError, match="could not be read safely"):
        run_action_stability_worker_v1(launch, **dependencies)

    assert calls["runtime_profiles"] == []
    assert not (tmp_path / "fresh-run" / "workers" / "worker-00").exists()


def test_eager_stage_gate_accepts_exact_completed_auto_roster(
    tmp_path: Path,
) -> None:
    for worker_index in range(4):
        auto_launch = _launch(
            tmp_path,
            profile=PROFILE_AUTO,
            worker_index=worker_index,
        )
        _, dependencies = _dependencies(auto_launch)
        run_action_stability_worker_v1(auto_launch, **dependencies)

    eager_launch = _launch(tmp_path, profile=PROFILE_EAGER, worker_index=0)
    execution.require_auto_stage_complete_v1(eager_launch)


def test_runtime_failure_still_writes_class_only_metric_safe_partial_terminal(
    tmp_path: Path,
) -> None:
    launch = _launch(tmp_path, profile=PROFILE_AUTO)
    _, dependencies = _dependencies(launch)

    class SensitiveRuntimeError(RuntimeError):
        pass

    def fail_runtime(*args: Any) -> Any:
        del args
        raise SensitiveRuntimeError(
            "output_text=SECRET coordinate=[1,2] tokens=[3] logits=SECRET"
        )

    dependencies["runtime_factory"] = fail_runtime
    terminal = run_action_stability_worker_v1(launch, **dependencies)

    assert terminal["status"] == "FAILED_ACTION_STABILITY_PROFILE_WORKER_V1"
    assert terminal["failure_class"] == "SensitiveRuntimeError"
    assert terminal["partial_results"] == []
    serialized = json.dumps(terminal, sort_keys=True).lower()
    for forbidden in ("secret", "output_text", "coordinate", "tokens", "logits"):
        assert forbidden not in serialized
    terminal_path = (
        tmp_path / "fresh-run" / "workers" / "worker-01" / "auto-terminal.json"
    )
    assert terminal_path.is_file()


def test_parent_envelope_is_loaded_with_freshness_disabled_only_for_artifact_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_repository = tmp_path / "parent-repository"
    parent_repository.mkdir()
    parent_worker = parent_repository / execution.PARENT_WORKER_ENTRYPOINT
    parent_aggregate = parent_repository / execution.PARENT_AGGREGATE_ENTRYPOINT
    parent_payload = execution.canonical_pretty_json_bytes(
        {
            "execution": {
                "aggregate": {"argv": ["/python", str(parent_aggregate)]},
                "worker_mapping": [
                    {
                        "argv": ["/python", str(parent_worker)],
                        "worker_index": worker_index,
                    }
                    for worker_index in range(4)
                ],
            }
        }
    )
    parent_sha = execution.sha256_bytes(parent_payload)
    monkeypatch.setattr(execution, "PARENT_EXECUTION_ENVELOPE_SHA256", parent_sha)
    launch = _launch(tmp_path, profile=PROFILE_AUTO)
    launch.parent_envelope_path.write_bytes(parent_payload)
    parent_run = tmp_path / "old-parent-run"
    parent_run.mkdir()
    calls: dict[str, Any] = {}
    parent_contract = object()
    validated = object()

    def contract_loader(**kwargs: Any) -> object:
        calls["contract"] = kwargs
        return parent_contract

    def envelope_loader(path: Path, **kwargs: Any) -> Mapping[str, Any]:
        calls["envelope_path"] = path
        calls["envelope_kwargs"] = kwargs
        return {
            "processor_root": str(launch.processor_root),
            "model_dir": str(launch.model_dir),
            "run_root": str(parent_run),
        }

    def adapter(parent_launch: Any, contract: Any, projection: Any) -> object:
        assert parent_launch.output_root == parent_run
        assert parent_launch.output_root != launch.output_root
        assert contract is parent_contract
        assert projection["run_root"] == str(parent_run)
        assert projection["repository_root"] == str(launch.repository_root)
        return validated

    observed_contract, observed_validated = load_parent_artifact_binding_v1(
        launch,
        parent_contract_loader=contract_loader,
        parent_envelope_loader=envelope_loader,
        projection_adapter=adapter,
    )

    assert observed_contract is parent_contract
    assert observed_validated is validated
    assert calls["envelope_path"] == launch.parent_envelope_path
    assert calls["envelope_kwargs"] == {
        "repository_root": parent_repository,
        "verify_repository": False,
        "verify_local_artifacts": "stat",
        "require_fresh_preflight": False,
        "verify_current_environment": False,
    }


def test_launch_rejects_parent_as_new_output_or_non_authorizing_fresh_receipt(
    tmp_path: Path,
) -> None:
    launch = _launch(tmp_path, profile=PROFILE_AUTO)
    with pytest.raises(PermissionError, match="fresh GPU"):
        ActionStabilityLaunchV1(
            **{
                field: getattr(launch, field)
                for field in launch.__dataclass_fields__
                if field != "freshness_validated"
            },
            freshness_validated=False,
        )
