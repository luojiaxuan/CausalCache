"""Focused tests for the four-worker throughput-pilot execution boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import aggregate_set_utility_throughput_pilot_v1 as aggregate_cli
from scripts import run_set_utility_throughput_pilot_worker_v1 as worker_cli

from causalcache.set_utility_throughput_pilot_contract_v1 import (
    CANONICAL_CONFIG_PATH,
    EXPECTED_STATE_IDS,
    TrainOnlyThroughputPilotSourceContractV1,
    canonical_pretty_json_bytes,
)
from causalcache.set_utility_throughput_pilot_execution_v1 import (
    AGGREGATE_FILENAME,
    ExecutionLaunchEnvelopeV1,
    LoadedPilotStateV1,
    ValidatedExecutionEnvelopeV1,
    WorkerRuntimeBundleV1,
    aggregate_throughput_pilot_workers_v1,
    canonical_candidate_schedule_producer_bytes_v2,
    run_throughput_pilot_worker_v1,
    select_reference_teacher_microbatch_v1,
)
from causalcache.set_utility_throughput_pilot_pair_v1 import _paired_payload


_DIGEST = "a" * 64


def test_candidate_schedule_reconstructs_only_the_producer_typed_key_path() -> None:
    producer = {
        "candidate_inventory": {"records": []},
        "exact_label_schedule": {
            "state_count_by_candidate_count": {4: 400, 10: 773},
        },
    }
    payload = canonical_pretty_json_bytes(producer)
    parsed = json.loads(payload)

    assert canonical_pretty_json_bytes(parsed) != payload
    assert canonical_candidate_schedule_producer_bytes_v2(parsed) == payload

    producer["other_numeric_mapping"] = {4: 1, 10: 2}
    payload_with_unregistered_typed_path = canonical_pretty_json_bytes(producer)
    parsed_with_unregistered_typed_path = json.loads(
        payload_with_unregistered_typed_path
    )
    assert (
        canonical_candidate_schedule_producer_bytes_v2(
            parsed_with_unregistered_typed_path
        )
        != payload_with_unregistered_typed_path
    )


@pytest.mark.parametrize("key", ["04", "0", "-1", "four", "٤"])
def test_candidate_schedule_rejects_noncanonical_integer_key_text(key: str) -> None:
    schedule = {
        "exact_label_schedule": {
            "state_count_by_candidate_count": {key: 1},
        }
    }
    with pytest.raises(ValueError, match="canonical positive base-10"):
        canonical_candidate_schedule_producer_bytes_v2(schedule)


def test_candidate_schedule_rejects_missing_or_empty_typed_key_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        canonical_candidate_schedule_producer_bytes_v2({})
    with pytest.raises(ValueError, match="must not be empty"):
        canonical_candidate_schedule_producer_bytes_v2(
            {
                "exact_label_schedule": {
                    "state_count_by_candidate_count": {},
                }
            }
        )


def _config() -> dict[str, object]:
    roster = [
        {"state_id": state_id, "worker_index": index % 4}
        for index, state_id in enumerate(EXPECTED_STATE_IDS)
    ]
    return {
        "inputs": {
            "freeze_b_v2_manifest": {"sha256": "b" * 64},
        },
        "pilot": {
            "roster": roster,
            "worker_count": 4,
            "worker_mapping": [
                {
                    "state_ids": list(EXPECTED_STATE_IDS[index::4]),
                    "worker_index": index,
                }
                for index in range(4)
            ],
        },
        "source": {"inventory_sha256": "c" * 64},
    }


def _contract(root: Path) -> TrainOnlyThroughputPilotSourceContractV1:
    return TrainOnlyThroughputPilotSourceContractV1(
        data=_config(),
        repository_root=root,
        config_sha256="d" * 64,
    )


def _launch(root: Path, output: Path, worker_index: int) -> ExecutionLaunchEnvelopeV1:
    return ExecutionLaunchEnvelopeV1(
        repository_root=root,
        config_path=CANONICAL_CONFIG_PATH,
        processor_root=root / "processor",
        model_dir=root / "model",
        output_root=output,
        worker_index=worker_index,
        device="cuda:0",
    )


def _validated(launch: ExecutionLaunchEnvelopeV1) -> ValidatedExecutionEnvelopeV1:
    return ValidatedExecutionEnvelopeV1(
        launch=launch,
        freeze_manifest_bytes=b"freeze",
        candidate_schedule_bytes=b"candidate",
        processor_inventory_sha256="e" * 64,
        model_inventory_sha256="f" * 64,
        processor_artifact_sha256_by_worker={index: _DIGEST for index in range(4)},
    )


def _core_variant(
    microbatch: int,
    *,
    teacher_wall: float,
    reserved_peak: int = 700,
    failure_class: str | None = None,
) -> dict[str, object]:
    teacher_calls = 2 if microbatch == 1 else 1
    completed_teacher_calls = 0 if failure_class is not None else teacher_calls
    completed_teacher_examples = 0 if failure_class is not None else 2
    return {
        "latency_seconds": {
            "reference_generation_end_to_end_wall_total": 1.0,
            "reference_teacher_forward_end_to_end_wall_total": teacher_wall,
            "native_call_end_to_end_wall_total": 1.0 + teacher_wall,
        },
        "peak_memory_bytes": {
            "full_call_cuda_allocated_max": min(600, reserved_peak),
            "full_call_cuda_reserved_max": reserved_peak,
        },
        "failure_class": failure_class,
        "counts": {
            "reference_teacher_microbatch_size": microbatch,
            "reference_plan_build_count": 1,
            "reference_input_build_call_count": 1,
            "reference_input_build_completed_count": 1,
            "reference_generation_call_count": 2,
            "reference_generation_completed_count": 2,
            "reference_teacher_forward_call_count": teacher_calls,
            "reference_teacher_forward_completed_call_count": (
                completed_teacher_calls
            ),
            "reference_teacher_forward_example_count": 2,
            "reference_teacher_forward_completed_example_count": (
                completed_teacher_examples
            ),
        },
    }


def _raw_pair(
    state_id: str,
    *,
    mb1_wall: float = 2.0,
    mb2_wall: float = 1.8,
    mb1_peak: int = 700,
    mb2_peak: int = 700,
    mb1_failure: str | None = None,
    mb2_failure: str | None = None,
    cross_equal: bool | None = True,
) -> dict[str, object]:
    variants = [
        _core_variant(
            1,
            teacher_wall=mb1_wall,
            reserved_peak=mb1_peak,
            failure_class=mb1_failure,
        )
    ]
    if mb1_failure is None:
        variants.append(
            _core_variant(
                2,
                teacher_wall=mb2_wall,
                reserved_peak=mb2_peak,
                failure_class=mb2_failure,
            )
        )
    if mb1_failure is not None:
        pair_failure = f"MICROBATCH_1__{mb1_failure.replace(':', '__')}"
        comparisons = 0
        cross = None
    elif mb2_failure is not None:
        pair_failure = f"MICROBATCH_2__{mb2_failure.replace(':', '__')}"
        comparisons = 1
        cross = cross_equal
    elif cross_equal is not True:
        variants[-1]["failure_class"] = "REFERENCE_ACTION_MISMATCH"
        pair_failure = "CROSS_VARIANT_REFERENCE_ACTION_MISMATCH"
        comparisons = 1
        cross = cross_equal
    else:
        pair_failure = None
        comparisons = 1
        cross = True
    return _paired_payload(
        state_id=state_id,
        variants=variants,
        cross_variant_equal=cross,
        cross_variant_comparison_count=comparisons,
        failure_class=pair_failure,
    )


def _projected_pair(state_id: str, **kwargs: object) -> dict[str, object]:
    raw = _raw_pair(state_id, **kwargs)
    return _project_pair_payload(raw)


def _project_pair_payload(raw: dict[str, object]) -> dict[str, object]:
    return {
        "counts": raw["counts"],
        "cross_variant_equal": raw["cross_variant_reference_action_equal"],
        "failure_class": raw["failure_class"],
        "latency_seconds": raw["latency_seconds"],
        "metric_only": raw["metric_only"],
        "peak_memory_bytes": raw["peak_memory_bytes"],
        "retry_count": raw["retry_count"],
        "state_id": raw["state_id"],
        "variant_order": raw["variant_order"],
        "variants": raw["variants"],
    }


def _selection(pairs: tuple[dict[str, object], ...]) -> dict[str, object]:
    worker_by_state = {
        state_id: index % 4 for index, state_id in enumerate(EXPECTED_STATE_IDS)
    }
    return select_reference_teacher_microbatch_v1(
        pairs,
        worker_index_by_state=worker_by_state,
        device_total_memory_bytes_by_worker={index: 1000 for index in range(4)},
    )


def test_selection_chooses_mb2_only_at_five_percent_speedup() -> None:
    pairs = tuple(_projected_pair(state_id) for state_id in EXPECTED_STATE_IDS)
    assert _selection(pairs)["outcome"] == "GO_MB2"

    slower = tuple(
        _projected_pair(state_id, mb2_wall=1.91)
        for state_id in EXPECTED_STATE_IDS
    )
    selected = _selection(slower)
    assert selected["outcome"] == "GO_MB1"
    assert selected["selected_reference_teacher_microbatch_size"] == 1


def test_selection_fails_closed_on_mb1_failure_or_either_peak() -> None:
    mb1_failure = list(
        _projected_pair(state_id) for state_id in EXPECTED_STATE_IDS
    )
    mb1_failure[0] = _projected_pair(
        EXPECTED_STATE_IDS[0],
        mb1_failure="REFERENCE_GENERATION:RuntimeError",
    )
    assert _selection(tuple(mb1_failure))["reason"] == "MICROBATCH_1_FAILURE"

    for microbatch in (1, 2):
        peaks = [
            _projected_pair(state_id) for state_id in EXPECTED_STATE_IDS
        ]
        kwargs = {f"mb{microbatch}_peak": 801}
        peaks[0] = _projected_pair(EXPECTED_STATE_IDS[0], **kwargs)
        assert _selection(tuple(peaks))["outcome"] == "NO_GO"


def test_selection_allows_only_mb2_teacher_failure_fallback() -> None:
    teacher_failure = tuple(
        _projected_pair(
            state_id,
            mb2_failure="REFERENCE_TEACHER_FORWARD:OutOfMemoryError",
        )
        for state_id in EXPECTED_STATE_IDS
    )
    selected = _selection(teacher_failure)
    assert selected["outcome"] == "GO_MB1"
    assert selected["reason"] == "MICROBATCH_2_TEACHER_FAILURE_FALLBACK"

    generation_failure = list(teacher_failure)
    generation_failure[0] = _projected_pair(
        EXPECTED_STATE_IDS[0],
        mb2_failure="REFERENCE_GENERATION:RuntimeError",
    )
    assert _selection(tuple(generation_failure))["outcome"] == "NO_GO"


def test_selection_maps_missing_invalid_core_variants_to_no_go() -> None:
    mb2_missing = tuple(
        _project_pair_payload(
            _paired_payload(
                state_id=state_id,
                variants=[_core_variant(1, teacher_wall=2.0)],
                cross_variant_equal=None,
                cross_variant_comparison_count=0,
                failure_class="MICROBATCH_2__INVALID_CORE_PROJECTION",
            )
        )
        for state_id in EXPECTED_STATE_IDS
    )
    selected = _selection(mb2_missing)
    assert selected["outcome"] == "NO_GO"
    assert selected["reason"] == "MICROBATCH_2_INVALID_CORE_PROJECTION"

    mb1_missing = tuple(
        _project_pair_payload(
            _paired_payload(
                state_id=state_id,
                variants=[],
                cross_variant_equal=None,
                cross_variant_comparison_count=0,
                failure_class="MICROBATCH_1__INVALID_CORE_PROJECTION",
            )
        )
        for state_id in EXPECTED_STATE_IDS
    )
    selected = _selection(mb1_missing)
    assert selected["outcome"] == "NO_GO"
    assert selected["reason"] == "MICROBATCH_1_INVALID_CORE_PROJECTION"


def test_worker_validates_contract_and_envelope_before_semantic_or_runtime(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    launch = _launch(tmp_path, output, 0)
    contract = _contract(tmp_path)
    calls: list[str] = []

    def contract_loader(**_: object) -> TrainOnlyThroughputPilotSourceContractV1:
        calls.append("contract")
        return contract

    def envelope_validator(
        _: ExecutionLaunchEnvelopeV1,
        __: TrainOnlyThroughputPilotSourceContractV1,
    ) -> ValidatedExecutionEnvelopeV1:
        calls.append("envelope")
        return _validated(launch)

    def semantic_loader(
        _: ValidatedExecutionEnvelopeV1,
        __: TrainOnlyThroughputPilotSourceContractV1,
    ) -> tuple[LoadedPilotStateV1, ...]:
        calls.append("semantic")
        return tuple(
            LoadedPilotStateV1(state_id, object(), object())
            for state_id in EXPECTED_STATE_IDS[0::4]
        )

    def runtime_factory(
        _: ValidatedExecutionEnvelopeV1,
        __: TrainOnlyThroughputPilotSourceContractV1,
    ) -> WorkerRuntimeBundleV1:
        calls.append("runtime")
        return WorkerRuntimeBundleV1(
            runtime=object(),
            reference_input_builder_for=lambda _: lambda __: object(),
            device_total_memory_bytes=1000,
            runtime_identity_sha256="1" * 64,
        )

    def pair_runner(query: object, **_: object) -> dict[str, object]:
        state_index = calls.count("pair")
        calls.append("pair")
        return _raw_pair(EXPECTED_STATE_IDS[0::4][state_index])

    terminal = run_throughput_pilot_worker_v1(
        launch,
        contract_loader=contract_loader,
        envelope_validator=envelope_validator,
        semantic_loader=semantic_loader,
        runtime_factory=runtime_factory,
        pair_runner=pair_runner,
    )
    assert calls[:4] == ["contract", "envelope", "semantic", "runtime"]
    assert terminal["status"] == "COMPLETED_THROUGHPUT_PILOT_WORKER_V1"
    assert terminal["device_total_memory_bytes"] == 1000
    assert len(terminal["pair_results"]) == 3
    serialized = json.dumps(terminal, sort_keys=True)
    assert "task instruction secret" not in serialized

    with pytest.raises(FileExistsError):
        run_throughput_pilot_worker_v1(
            launch,
            contract_loader=contract_loader,
            envelope_validator=envelope_validator,
            semantic_loader=semantic_loader,
            runtime_factory=runtime_factory,
            pair_runner=pair_runner,
        )


def _materialize_successful_workers(
    *,
    root: Path,
    output: Path,
    contract: TrainOnlyThroughputPilotSourceContractV1,
) -> None:
    def contract_loader(**_: object) -> TrainOnlyThroughputPilotSourceContractV1:
        return contract

    for worker_index in range(4):
        launch = _launch(root, output, worker_index)
        state_ids = EXPECTED_STATE_IDS[worker_index::4]

        def semantic_loader(
            _: ValidatedExecutionEnvelopeV1,
            __: TrainOnlyThroughputPilotSourceContractV1,
            *,
            ids: tuple[str, ...] = state_ids,
        ) -> tuple[LoadedPilotStateV1, ...]:
            return tuple(
                LoadedPilotStateV1(state_id, object(), object())
                for state_id in ids
            )

        pair_index = {"value": 0}

        def pair_runner(
            query: object,
            **_: object,
        ) -> dict[str, object]:
            index = pair_index["value"]
            pair_index["value"] += 1
            return _raw_pair(state_ids[index])

        run_throughput_pilot_worker_v1(
            launch,
            contract_loader=contract_loader,
            envelope_validator=lambda *_: _validated(launch),
            semantic_loader=semantic_loader,
            runtime_factory=lambda *_: WorkerRuntimeBundleV1(
                runtime=object(),
                reference_input_builder_for=lambda _: lambda __: object(),
                device_total_memory_bytes=1000,
                runtime_identity_sha256="1" * 64,
            ),
            pair_runner=pair_runner,
        )


def test_exact_four_worker_aggregate_is_atomic_and_selects_mb2(tmp_path: Path) -> None:
    output = tmp_path / "out"
    contract = _contract(tmp_path)

    def contract_loader(**_: object) -> TrainOnlyThroughputPilotSourceContractV1:
        return contract

    _materialize_successful_workers(
        root=tmp_path,
        output=output,
        contract=contract,
    )

    aggregate = aggregate_throughput_pilot_workers_v1(
        repository_root=tmp_path,
        output_root=output,
        contract_loader=contract_loader,
    )
    assert aggregate["selection"]["outcome"] == "GO_MB2"
    assert aggregate["metrics"]["counts"]["native_call_count"] == 84
    assert (output / AGGREGATE_FILENAME).is_file()
    with pytest.raises(FileExistsError):
        aggregate_throughput_pilot_workers_v1(
            repository_root=tmp_path,
            output_root=output,
            contract_loader=contract_loader,
        )


def test_aggregate_rejects_terminal_schema_runtime_and_device_drift(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    contract = _contract(tmp_path)

    def contract_loader(**_: object) -> TrainOnlyThroughputPilotSourceContractV1:
        return contract

    _materialize_successful_workers(
        root=tmp_path,
        output=output,
        contract=contract,
    )
    terminal_path = output / "workers/worker-01/terminal.json"
    original = json.loads(terminal_path.read_text(encoding="utf-8"))

    mutations = (
        lambda value: value.__setitem__("action", "forbidden"),
        lambda value: value.__setitem__("runtime_identity_sha256", "2" * 64),
        lambda value: value.__setitem__("device_id", "cuda:1"),
    )
    for mutate in mutations:
        changed = json.loads(json.dumps(original))
        mutate(changed)
        terminal_path.write_bytes(canonical_pretty_json_bytes(changed))
        with pytest.raises(ValueError):
            aggregate_throughput_pilot_workers_v1(
                repository_root=tmp_path,
                output_root=output,
                contract_loader=contract_loader,
            )
    terminal_path.write_bytes(canonical_pretty_json_bytes(original))


def test_aggregate_publishes_no_go_for_missing_mb2_projection(tmp_path: Path) -> None:
    output = tmp_path / "out"
    contract = _contract(tmp_path)

    def contract_loader(**_: object) -> TrainOnlyThroughputPilotSourceContractV1:
        return contract

    _materialize_successful_workers(
        root=tmp_path,
        output=output,
        contract=contract,
    )
    for worker_index in range(4):
        terminal_path = (
            output / f"workers/worker-{worker_index:02d}/terminal.json"
        )
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal["pair_results"] = [
            _project_pair_payload(
                _paired_payload(
                    state_id=state_id,
                    variants=[_core_variant(1, teacher_wall=2.0)],
                    cross_variant_equal=None,
                    cross_variant_comparison_count=0,
                    failure_class="MICROBATCH_2__INVALID_CORE_PROJECTION",
                )
            )
            for state_id in EXPECTED_STATE_IDS[worker_index::4]
        ]
        terminal_path.write_bytes(canonical_pretty_json_bytes(terminal))

    aggregate = aggregate_throughput_pilot_workers_v1(
        repository_root=tmp_path,
        output_root=output,
        contract_loader=contract_loader,
    )
    assert aggregate["selection"] == {
        "outcome": "NO_GO",
        "reason": "MICROBATCH_2_INVALID_CORE_PROJECTION",
        "selected_reference_teacher_microbatch_size": None,
    }
    assert aggregate["metrics"]["all_pairs_successful"] is False


def test_worker_cli_accepts_only_exact_envelope_and_worker_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope_path = tmp_path / "execution-envelope.json"
    gpu_uuid = "GPU-11111111-1111-1111-1111-111111111111"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_uuid)
    projection = {
        "config_path": str(tmp_path / CANONICAL_CONFIG_PATH),
        "device_by_worker": {str(index): "cuda:0" for index in range(4)},
        "envelope": {},
        "envelope_path": str(envelope_path),
        "model_dir": str(tmp_path / "model"),
        "processor_root": str(tmp_path / "processor"),
        "repository_root": str(tmp_path),
        "run_root": str(tmp_path / "run"),
        "validation": {},
        "worker_gpu_uuid": {str(index): gpu_uuid for index in range(4)},
    }
    observed: list[ExecutionLaunchEnvelopeV1] = []

    def runner(launch: ExecutionLaunchEnvelopeV1, **_: object) -> dict[str, object]:
        observed.append(launch)
        return {"metric_only": True, "status": "ok"}

    worker_cli.main(
        ["--execution-envelope", str(envelope_path), "--worker-index", "2"],
        envelope_loader=lambda _: projection,
        worker_runner=runner,
    )
    assert observed[0].worker_index == 2
    assert observed[0].device == "cuda:0"

    with pytest.raises(SystemExit):
        worker_cli.main(
            [
                "--execution-envelope",
                str(envelope_path),
                "--worker-index",
                "2",
                "--model-dir",
                "/tmp/bypass",
            ],
            envelope_loader=lambda _: projection,
            worker_runner=runner,
        )


def test_aggregate_cli_has_no_independent_artifact_or_output_flags(
    tmp_path: Path,
) -> None:
    envelope_path = tmp_path / "execution-envelope.json"
    projection = {
        "config_path": str(tmp_path / CANONICAL_CONFIG_PATH),
        "envelope_path": str(envelope_path),
        "repository_root": str(tmp_path),
        "run_root": str(tmp_path / "run"),
    }
    calls: list[dict[str, object]] = []

    def aggregator(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"metric_only": True, "status": "ok"}

    aggregate_cli.main(
        ["--execution-envelope", str(envelope_path)],
        envelope_loader=lambda _: projection,
        aggregator=aggregator,
    )
    assert calls[0]["output_root"] == tmp_path / "run"
    with pytest.raises(SystemExit):
        aggregate_cli.main(
            [
                "--execution-envelope",
                str(envelope_path),
                "--output-root",
                "/tmp/bypass",
            ],
            envelope_loader=lambda _: projection,
            aggregator=aggregator,
        )
