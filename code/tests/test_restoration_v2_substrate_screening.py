from __future__ import annotations

import argparse
import hashlib
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from causalcache.data.restoration_v2_screening import ScreeningState
from causalcache.policy.gui_owl_v2_runtime import GUIOwlV2GenerationParseError
from causalcache.data.guiodyssey_restoration_v2 import (
    EXPECTED_FORMAL_COUNTS,
    PAYLOAD_PREFIX,
)
from scripts.run_restoration_v2_substrate_screening import (
    CONFIRM_LOCKED,
    SCREENING_ALLOWED,
    STATE_OUTCOME_FAILED,
    STATE_OUTCOME_VALID,
    DistanceMeasurement,
    GPUFullVocabularyKLBackend,
    SubstrateGateContract,
    _canonical_input_bindings,
    _execution_bindings,
    aggregate_substrate_gate,
    execute_production_screening,
    preflight_screening_message_shapes,
    run_screening_state,
    run_substrate_screening,
    validate_screening_authorization_result,
)


RUN_SHA256 = "a" * 64
ROOT = Path(__file__).resolve().parents[2]


def _authorization() -> dict[str, object]:
    return {
        "state": SCREENING_ALLOWED,
        "confirm_state": CONFIRM_LOCKED,
        "confirm_locked": True,
        "allowed_roles": ["v2_label_train", "v2_development"],
        "passed_dependency_count": 8,
        "dependency_count": 8,
    }


def _states() -> tuple[ScreeningState, ...]:
    states = []
    for trajectory_index in range(15):
        role = "v2_label_train" if trajectory_index < 10 else "v2_development"
        trajectory_id = f"trajectory-{trajectory_index:02d}"
        for decision_step in (4, 5, 6):
            states.append(
                ScreeningState(
                    index=len(states),
                    role=role,
                    trajectory_id=trajectory_id,
                    decision_step_id=decision_step,
                    candidate_event_step_ids=tuple(range(1, decision_step - 1)),
                )
            )
    return tuple(states)


class _Action:
    def __init__(self, name: str = "wait") -> None:
        self.name = name

    def arguments(self) -> dict[str, object]:
        return {"action": self.name}


def _generation(name: str = "wait") -> object:
    action = _Action(name)
    return types.SimpleNamespace(
        output_text=f"Action: {name}",
        parsed_output=types.SimpleNamespace(canonical_action=action),
        metadata={"do_sample": False},
    )


class _Artifact:
    def __init__(self) -> None:
        self.states = _states()
        self.build_calls: list[tuple[int, tuple[int, ...]]] = []
        self.artifact_tree_sha256 = "b" * 64
        self.artifact_manifest_sha256 = "c" * 64
        self.screening_manifest_sha256 = "d" * 64

    def build_messages(
        self,
        state: ScreeningState,
        *,
        restored_event_step_ids: object,
        image_decoder: object,
    ) -> dict[str, object]:
        del image_decoder
        restored = tuple(restored_event_step_ids)
        self.build_calls.append((state.index, restored))
        return {
            "kind": "reference" if restored else "summary",
            "expected_image_count": len(restored) + 1 if restored else 1,
        }


class _Runtime:
    def __init__(self, generations: list[object] | None = None) -> None:
        self.generations = list(generations or [_generation(), _generation()])
        self.generation_calls = 0
        self.teacher_calls: list[str] = []
        self.shape_calls: list[str] = []

    def prepare_native_message_shape(self, messages: dict[str, object]) -> dict[str, object]:
        kind = str(messages["kind"])
        self.shape_calls.append(kind)
        return {
            "kind": kind,
            "policy_forward_executed": False,
            "image_count": messages["expected_image_count"],
            "sequence_length": 100 + int(messages["expected_image_count"]),
        }

    def maximum_context_tokens(self) -> int:
        return 4096

    def generate_native_action(self, messages: dict[str, object]) -> object:
        if messages["kind"] != "reference":
            raise AssertionError("generation must use full history")
        result = self.generations[self.generation_calls % len(self.generations)]
        self.generation_calls += 1
        if isinstance(result, Exception):
            raise result
        return result

    def teacher_forced_distance_logits(
        self,
        messages_batch: tuple[dict[str, object]],
        actions: tuple[object],
    ) -> tuple[str, dict[str, object]]:
        if len(messages_batch) != 1 or len(actions) != 1:
            raise AssertionError("teacher forwards must remain batch one")
        kind = str(messages_batch[0]["kind"])
        self.teacher_calls.append(kind)
        call_index = len(self.teacher_calls)
        return f"{kind}-{call_index}", {"batch_size": 1, "kind": kind}


class _DistanceBackend:
    def __init__(self, repeat: float = 1e-6, summary: float = 0.1) -> None:
        self.repeat = repeat
        self.summary = summary
        self.prepare_calls: list[object] = []
        self.measure_calls: list[tuple[object, object]] = []

    def prepare_reference(self, logits: object) -> str:
        self.prepare_calls.append(logits)
        return "reference-log-probs"

    def measure(self, reference: object, candidate: object) -> DistanceMeasurement:
        self.measure_calls.append((reference, candidate))
        value = self.summary if str(candidate).startswith("summary") else self.repeat
        return DistanceMeasurement(
            value=value,
            audit={"full_tensor_host_transfers": 0},
        )


def _gate() -> SubstrateGateContract:
    return SubstrateGateContract(
        minimum_screening_states=20,
        minimum_parse_coverage=0.99,
        minimum_finite_logit_coverage=1.0,
        minimum_repeat_canonical_action_agreement=1.0,
        minimum_memory_sensitive_states=8,
        fail_outcome="NO_GO_V2_SUBSTRATE",
        pass_outcome="RUN_UNTOUCHED_RESTORATION_CONFIRM",
    )


def _valid_record(
    state: ScreeningState,
    *,
    repeat: float = 1e-6,
    summary: float = 0.1,
) -> dict[str, object]:
    return {
        "run_contract_sha256": RUN_SHA256,
        "state": {
            "index": state.index,
            "role": state.role,
            "trajectory_id": state.trajectory_id,
            "decision_step_id": state.decision_step_id,
            "state_id": state.state_id,
            "candidate_event_step_ids": list(state.candidate_event_step_ids),
        },
        "outcome": STATE_OUTCOME_VALID,
        "failure": None,
        "parse_success": True,
        "finite_logit_distances": True,
        "repeat_canonical_action_agreement": True,
        "distances": {
            "repeat_reference_kl": repeat,
            "summary_reference_kl": summary,
        },
    }


class ScreeningAuthorizationTest(unittest.TestCase):
    def test_requires_exact_screening_only_authorization(self) -> None:
        self.assertEqual(
            validate_screening_authorization_result(_authorization())["state"],
            SCREENING_ALLOWED,
        )
        invalid = _authorization()
        invalid["allowed_roles"] = [
            "v2_label_train",
            "v2_development",
            "v2_confirm_primary",
        ]
        with self.assertRaises(PermissionError):
            validate_screening_authorization_result(invalid)

    def test_failed_readiness_prevents_artifact_and_runtime_loading(self) -> None:
        args = argparse.Namespace(
            repository_root=".",
            execution_config="missing",
            readiness_manifest="missing",
            device="cuda:0",
        )
        artifact_loader = mock.Mock()
        runtime_loader = mock.Mock()
        with self.assertRaises(PermissionError):
            execute_production_screening(
                args,
                authorization_validator=lambda **kwargs: {
                    **_authorization(),
                    "state": "NOT_ALLOWED",
                },
                artifact_loader=artifact_loader,
                runtime_class_loader=runtime_loader,
            )
        artifact_loader.assert_not_called()
        runtime_loader.assert_not_called()

    def test_flat_execution_runtime_identity_is_required_before_import(self) -> None:
        snapshot_path = ROOT / "code/configs/gui_owl_1_5_8b_snapshot.json"
        container_id = "c" * 64
        container_hostname = container_id[:12]
        digest = "sha256:" + "d" * 64
        config = {
            "canonical_data": {
                "repo": "fixture/repo",
                "immutable_revision": "e" * 40,
                "artifact_tree_sha256": "f" * 64,
                "payload_prefix": PAYLOAD_PREFIX,
                "counts": dict(EXPECTED_FORMAL_COUNTS),
            },
            "canonical_policy": {
                "snapshot_manifest": {
                    "path": "code/configs/gui_owl_1_5_8b_snapshot.json",
                    "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                }
            },
            "execution_runtime": {
                "host_alias": "hyper00",
                "host_hostname": "node-radixark-16-0000",
                "selected_device": "cuda:0",
                "container_id": container_id,
                "container_hostname": container_hostname,
                "container_image": "hongccc/sglang-omni:dev",
                "container_image_digest": digest,
            },
        }
        bindings = _execution_bindings(
            config,
            repository_root=ROOT,
            device="cuda:0",
            host_alias="hyper00",
            host_hostname="node-radixark-16-0000",
            container_id=container_id,
            container_image_digest=digest,
            actual_container_hostname=container_hostname,
        )
        self.assertEqual(
            bindings["actual_run_identity"]["selected_device"], "cuda:0"
        )
        nested = dict(config)
        nested["execution_runtime"] = {
            **config["execution_runtime"],
            "selected_device": None,
            "gpu": {"selected_device": "cuda:0"},
        }
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            _execution_bindings(
                nested,
                repository_root=ROOT,
                device="cuda:0",
                host_alias="hyper00",
                host_hostname="node-radixark-16-0000",
                container_id=container_id,
                container_image_digest=digest,
                actual_container_hostname=container_hostname,
            )

    def test_canonical_input_paths_and_dependency_hashes_are_required(self) -> None:
        scientific = ROOT / "code/configs/causalcache_restoration_v2.json"
        selection = ROOT / "data/manifests/restoration_v2_selection.json"
        ocr_manifest = ROOT / "data/manifests/restoration_v2_ocr_backend.json"
        config = {
            "scientific_contract": {
                "path": "code/configs/causalcache_restoration_v2.json",
                "sha256": hashlib.sha256(scientific.read_bytes()).hexdigest(),
            },
            "dependencies": [
                {
                    "id": 2,
                    "status": "passed",
                    "evidence": [
                        {
                            "path": "data/manifests/restoration_v2_selection.json",
                            "sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
                        }
                    ],
                },
                {
                    "id": 5,
                    "status": "passed",
                    "evidence": [
                        {
                            "path": "data/manifests/restoration_v2_ocr_backend.json",
                            "sha256": hashlib.sha256(ocr_manifest.read_bytes()).hexdigest(),
                        }
                    ],
                },
            ],
        }
        bindings = _canonical_input_bindings(
            config,
            repository_root=ROOT,
            scientific_config_path="code/configs/causalcache_restoration_v2.json",
            selection_manifest_path="data/manifests/restoration_v2_selection.json",
            ocr_backend_config_path="code/configs/restoration_v2_ocr_backend.json",
        )
        self.assertEqual(bindings["scientific_config_path"], scientific)
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "scientific.json"
            copied.write_bytes(scientific.read_bytes())
            with self.assertRaisesRegex(ValueError, "canonical Git path"):
                _canonical_input_bindings(
                    config,
                    repository_root=ROOT,
                    scientific_config_path=copied,
                    selection_manifest_path=selection,
                    ocr_backend_config_path=(
                        ROOT / "code/configs/restoration_v2_ocr_backend.json"
                    ),
                )

    def test_runtime_identity_mismatch_fails_before_runtime_import(self) -> None:
        args = argparse.Namespace(
            repository_root=".",
            execution_config="config",
            readiness_manifest="readiness",
            scientific_config="scientific",
            selection_manifest="selection",
            ocr_backend_config="backend",
            device="cuda:0",
            host_alias="wrong-host",
            host_hostname="wrong-hostname",
            container_id="a" * 64,
            container_image_digest="sha256:" + "b" * 64,
        )
        runtime_loader = mock.Mock()
        with (
            mock.patch(
                "scripts.run_restoration_v2_substrate_screening.load_json_object",
                return_value=(b"{}", {}),
            ),
            mock.patch(
                "scripts.run_restoration_v2_substrate_screening._canonical_input_bindings",
                return_value={},
            ),
            mock.patch(
                "scripts.run_restoration_v2_substrate_screening._execution_bindings",
                side_effect=ValueError("actual host_alias differs"),
            ),
            self.assertRaisesRegex(ValueError, "host_alias"),
        ):
            execute_production_screening(
                args,
                authorization_validator=lambda **kwargs: _authorization(),
                runtime_class_loader=runtime_loader,
            )
        runtime_loader.assert_not_called()

    def test_artifact_failure_prevents_runtime_import(self) -> None:
        args = argparse.Namespace(
            repository_root=".",
            execution_config="config",
            readiness_manifest="readiness",
            device="cuda:0",
            derived_artifact_root="artifact",
            ocr_backend_config="backend",
            scientific_config="scientific",
            selection_manifest="selection",
            host_alias="hyper00",
            host_hostname="host",
            container_id="a" * 64,
            container_image_digest="sha256:" + "b" * 64,
        )
        runtime_loader = mock.Mock()
        with (
            mock.patch(
                "scripts.run_restoration_v2_substrate_screening.load_json_object",
                return_value=(b"{}", {}),
            ),
            mock.patch(
                "scripts.run_restoration_v2_substrate_screening._execution_bindings",
                return_value={
                    "canonical_data": {"artifact_tree_sha256": "a" * 64},
                    "snapshot_manifest_path": Path("snapshot"),
                    "snapshot_manifest": {},
                    "execution_runtime": {},
                    "actual_run_identity": {},
                },
            ),
            mock.patch(
                "scripts.run_restoration_v2_substrate_screening._canonical_input_bindings",
                return_value={
                    "scientific_config_path": Path("scientific"),
                    "selection_manifest_path": Path("selection"),
                    "ocr_backend_config_path": Path("backend"),
                    "records": {},
                },
            ),
            self.assertRaisesRegex(ValueError, "artifact invalid"),
        ):
            execute_production_screening(
                args,
                authorization_validator=lambda **kwargs: _authorization(),
                artifact_loader=lambda **kwargs: (_ for _ in ()).throw(
                    ValueError("artifact invalid")
                ),
                runtime_class_loader=runtime_loader,
            )
        runtime_loader.assert_not_called()


class ScreeningStateTest(unittest.TestCase):
    def test_runs_two_generations_two_reference_forwards_and_summary(self) -> None:
        artifact = _Artifact()
        runtime = _Runtime()
        distance = _DistanceBackend()
        record = run_screening_state(
            artifact=artifact,
            state=artifact.states[0],
            runtime=runtime,
            distance_backend=distance,
            image_decoder=lambda payload: payload,
            run_contract_sha256=RUN_SHA256,
        )
        self.assertEqual(record["outcome"], STATE_OUTCOME_VALID)
        self.assertEqual(runtime.generation_calls, 2)
        self.assertEqual(runtime.teacher_calls, ["reference", "reference", "summary"])
        self.assertEqual(len(distance.prepare_calls), 1)
        self.assertEqual(len(distance.measure_calls), 2)
        self.assertEqual(record["distances"]["repeat_reference_kl"], 1e-6)
        self.assertEqual(record["distances"]["summary_reference_kl"], 0.1)

    def test_canonical_mismatch_is_fixed_denominator_failure(self) -> None:
        artifact = _Artifact()
        runtime = _Runtime([_generation("wait"), _generation("open")])
        record = run_screening_state(
            artifact=artifact,
            state=artifact.states[0],
            runtime=runtime,
            distance_backend=_DistanceBackend(),
            image_decoder=lambda payload: payload,
            run_contract_sha256=RUN_SHA256,
        )
        self.assertEqual(record["outcome"], STATE_OUTCOME_FAILED)
        self.assertEqual(record["failure"]["category"], "CANONICAL_ACTION_MISMATCH")
        self.assertEqual(runtime.teacher_calls, [])

    def test_parse_failure_is_classified_without_retry(self) -> None:
        artifact = _Artifact()
        runtime = _Runtime([ValueError("tool call parse failed")])
        record = run_screening_state(
            artifact=artifact,
            state=artifact.states[0],
            runtime=runtime,
            distance_backend=_DistanceBackend(),
            image_decoder=lambda payload: payload,
            run_contract_sha256=RUN_SHA256,
        )
        self.assertEqual(record["failure"]["category"], "PARSE_FAILURE")
        self.assertEqual(runtime.generation_calls, 1)

    def test_parse_failure_preserves_generated_text_when_runtime_provides_it(self) -> None:
        artifact = _Artifact()
        error = GUIOwlV2GenerationParseError(
            output_text="raw malformed generation",
            metadata={"do_sample": False},
            parse_error=ValueError("malformed tool call"),
        )
        record = run_screening_state(
            artifact=artifact,
            state=artifact.states[0],
            runtime=_Runtime([error]),
            distance_backend=_DistanceBackend(),
            image_decoder=lambda payload: payload,
            run_contract_sha256=RUN_SHA256,
        )
        self.assertEqual(record["failure"]["category"], "PARSE_FAILURE")
        self.assertEqual(
            record["native_generations"][0]["output_text"],
            "raw malformed generation",
        )

    def test_nonfinite_distance_is_null_and_classified(self) -> None:
        artifact = _Artifact()
        record = run_screening_state(
            artifact=artifact,
            state=artifact.states[0],
            runtime=_Runtime(),
            distance_backend=_DistanceBackend(repeat=math.nan),
            image_decoder=lambda payload: payload,
            run_contract_sha256=RUN_SHA256,
        )
        self.assertIsNone(record["distances"]["repeat_reference_kl"])
        self.assertEqual(record["failure"]["category"], "NONFINITE_DISTANCE")

    def test_out_of_memory_is_fatal_and_never_retried(self) -> None:
        artifact = _Artifact()
        runtime = _Runtime([RuntimeError("CUDA out of memory")])
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            run_screening_state(
                artifact=artifact,
                state=artifact.states[0],
                runtime=runtime,
                distance_backend=_DistanceBackend(),
                image_decoder=lambda payload: payload,
                run_contract_sha256=RUN_SHA256,
            )
        self.assertEqual(runtime.generation_calls, 1)

    def test_teacher_runtime_error_is_invalid_not_scientific_no_go(self) -> None:
        artifact = _Artifact()
        runtime = _Runtime()
        runtime.teacher_forced_distance_logits = mock.Mock(
            side_effect=RuntimeError("kernel launch failed")
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "INVALID_V2_SUBSTRATE_RUNTIME.*reference_teacher_forward_1",
        ):
            run_screening_state(
                artifact=artifact,
                state=artifact.states[0],
                runtime=runtime,
                distance_backend=_DistanceBackend(),
                image_decoder=lambda payload: payload,
                run_contract_sha256=RUN_SHA256,
            )


class ScreeningShapeSweepTest(unittest.TestCase):
    def test_all_90_shapes_complete_before_any_generation(self) -> None:
        artifact = _Artifact()
        runtime = _Runtime()
        summary = preflight_screening_message_shapes(
            artifact=artifact,
            runtime=runtime,
            image_decoder=lambda payload: payload,
            max_new_tokens=256,
        )
        self.assertEqual(summary["prompt_count"], 90)
        self.assertEqual(len(runtime.shape_calls), 90)
        self.assertEqual(runtime.generation_calls, 0)

    def test_context_overflow_is_invalid_before_policy_forward(self) -> None:
        artifact = _Artifact()
        runtime = _Runtime()
        runtime.maximum_context_tokens = lambda: 200  # type: ignore[method-assign]
        with self.assertRaisesRegex(ValueError, "INVALID_BEFORE_POLICY_FORWARD"):
            preflight_screening_message_shapes(
                artifact=artifact,
                runtime=runtime,
                image_decoder=lambda payload: payload,
                max_new_tokens=256,
            )
        self.assertEqual(runtime.generation_calls, 0)

    def test_production_calls_executor_only_after_complete_shape_sweep(self) -> None:
        artifact = _Artifact()
        runtime = _Runtime()
        runtime.torch = types.SimpleNamespace()
        runtime.metadata = {"fixture": True}
        scientific = {
            "primary_policy": {
                "visual_preprocessing": {
                    "target_effective_tokens_per_image": 2560
                },
                "decoding": {"max_new_tokens": 256},
            },
            "substrate_gate": {
                "screening_roles": ["v2_label_train", "v2_development"],
                "minimum_screening_states": 20,
                "minimum_parse_coverage": 0.99,
                "minimum_finite_logit_coverage": 1.0,
                "minimum_repeat_canonical_action_agreement": 1.0,
                "minimum_memory_sensitive_states": 8,
                "memory_sensitive_definition": (
                    "summary_reference_kl_greater_than_repeat_noise_epsilon"
                ),
                "fail_outcome": "NO_GO_V2_SUBSTRATE",
                "pass_outcome": "RUN_UNTOUCHED_RESTORATION_CONFIRM",
            },
        }
        args = argparse.Namespace(
            repository_root=str(ROOT),
            execution_config="execution",
            readiness_manifest="readiness",
            derived_artifact_root="artifact",
            scientific_config="scientific",
            selection_manifest="selection",
            ocr_backend_config="ocr",
            model_dir="model",
            device="cuda:0",
            host_alias="hyper00",
            host_hostname="host",
            container_id="a" * 64,
            container_image_digest="sha256:" + "b" * 64,
            output_dir="output",
            resume=False,
        )

        def json_loader(path: object) -> tuple[bytes, dict[str, object]]:
            return (b"{}", scientific if str(path) == "scientific-canonical" else {})

        def executor(**kwargs: object) -> dict[str, object]:
            self.assertEqual(len(runtime.shape_calls), 90)
            self.assertEqual(runtime.generation_calls, 0)
            self.assertEqual(
                kwargs["run_contract"]["processor_only_shape_sweep"]["prompt_count"],
                90,
            )
            return {"outcome": "fixture"}

        with (
            mock.patch(
                "scripts.run_restoration_v2_substrate_screening.load_json_object",
                side_effect=json_loader,
            ),
            mock.patch(
                "scripts.run_restoration_v2_substrate_screening._canonical_input_bindings",
                return_value={
                    "scientific_config_path": Path("scientific-canonical"),
                    "selection_manifest_path": Path("selection-canonical"),
                    "ocr_backend_config_path": Path("ocr-canonical"),
                    "records": {},
                },
            ),
            mock.patch(
                "scripts.run_restoration_v2_substrate_screening._execution_bindings",
                return_value={
                    "canonical_data": {"artifact_tree_sha256": "a" * 64},
                    "snapshot_manifest_path": Path("snapshot"),
                    "snapshot_manifest": {},
                    "execution_runtime": {},
                    "actual_run_identity": {},
                },
            ),
            mock.patch(
                "scripts.run_restoration_v2_substrate_screening._git_head",
                return_value="c" * 40,
            ),
            mock.patch(
                "scripts.run_restoration_v2_substrate_screening._source_identity",
                return_value={"path": "fixture.py", "sha256": "d" * 64},
            ),
            mock.patch(
                "scripts.run_restoration_v2_substrate_screening.sha256_file",
                return_value="e" * 64,
            ),
        ):
            result = execute_production_screening(
                args,
                authorization_validator=lambda **kwargs: _authorization(),
                artifact_loader=lambda **kwargs: artifact,
                runtime_class_loader=lambda: (
                    lambda **kwargs: runtime
                ),
                kl_kernel_loader=lambda: mock.Mock(),
                run_executor=executor,
            )
        self.assertEqual(result, {"outcome": "fixture"})


class SubstrateAggregateTest(unittest.TestCase):
    def test_uses_global_mean_repeat_noise_epsilon(self) -> None:
        states = _states()
        records = [_valid_record(state, repeat=0.001, summary=0.02) for state in states]
        aggregate = aggregate_substrate_gate(
            records,
            expected_states=states,
            gate=_gate(),
            run_contract_sha256=RUN_SHA256,
        )
        self.assertAlmostEqual(aggregate["metrics"]["mean_repeat_kl"], 0.001)
        self.assertAlmostEqual(aggregate["metrics"]["repeat_noise_epsilon"], 0.01)
        self.assertEqual(aggregate["metrics"]["memory_sensitive_state_count"], 45)
        self.assertTrue(aggregate["gate_passed"])

    def test_failure_remains_in_45_state_denominator(self) -> None:
        states = _states()
        records = [_valid_record(state) for state in states]
        records[0]["outcome"] = STATE_OUTCOME_FAILED
        records[0]["failure"] = {"category": "PARSE_FAILURE"}
        records[0]["parse_success"] = False
        records[0]["finite_logit_distances"] = False
        records[0]["repeat_canonical_action_agreement"] = False
        records[0]["distances"] = {
            "repeat_reference_kl": None,
            "summary_reference_kl": None,
        }
        aggregate = aggregate_substrate_gate(
            records,
            expected_states=states,
            gate=_gate(),
            run_contract_sha256=RUN_SHA256,
        )
        self.assertEqual(aggregate["metrics"]["fixed_state_denominator"], 45)
        self.assertFalse(aggregate["gate_passed"])
        self.assertEqual(aggregate["failure_category_counts"], {"PARSE_FAILURE": 1})
        with self.assertRaisesRegex(ValueError, "exactly 45 state records"):
            aggregate_substrate_gate(
                records[:-1],
                expected_states=states,
                gate=_gate(),
                run_contract_sha256=RUN_SHA256,
            )


class ResumeTest(unittest.TestCase):
    def test_exclusive_run_resumes_without_reexecuting_states(self) -> None:
        artifact = _Artifact()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            first_runtime = _Runtime()
            first = run_substrate_screening(
                artifact=artifact,
                runtime=first_runtime,
                distance_backend=_DistanceBackend(),
                image_decoder=lambda payload: payload,
                gate=_gate(),
                run_contract={"identity": "fixture"},
                output_dir=output,
                resume=False,
            )
            second_runtime = mock.Mock()
            resumed = run_substrate_screening(
                artifact=artifact,
                runtime=second_runtime,
                distance_backend=mock.Mock(),
                image_decoder=lambda payload: payload,
                gate=_gate(),
                run_contract={"identity": "fixture"},
                output_dir=output,
                resume=True,
            )
            (output / "states/000.json").unlink()
            (output / "aggregate.json").unlink()
            interrupted_runtime = mock.Mock()
            with self.assertRaisesRegex(RuntimeError, "cannot be retried"):
                run_substrate_screening(
                    artifact=artifact,
                    runtime=interrupted_runtime,
                    distance_backend=mock.Mock(),
                    image_decoder=lambda payload: payload,
                    gate=_gate(),
                    run_contract={"identity": "fixture"},
                    output_dir=output,
                    resume=True,
                )
            interrupted_runtime.prepare_native_message_shape.assert_not_called()
            with self.assertRaises(FileExistsError):
                run_substrate_screening(
                    artifact=artifact,
                    runtime=_Runtime(),
                    distance_backend=_DistanceBackend(),
                    image_decoder=lambda payload: payload,
                    gate=_gate(),
                    run_contract={"identity": "fixture"},
                    output_dir=output,
                    resume=False,
                )
        self.assertEqual(first, resumed)
        self.assertEqual(first_runtime.generation_calls, 90)
        second_runtime.prepare_native_message_shape.assert_not_called()


class _Device:
    type = "cuda"


class _Tensor:
    ndim = 3
    shape = (1, 2, 3)
    dtype = "torch.bfloat16"
    device = _Device()
    requires_grad = False

    def __init__(self) -> None:
        self.to_calls: list[dict[str, object]] = []

    def to(self, **kwargs: object) -> "_Reference":
        self.to_calls.append(kwargs)
        return _Reference()


class _Reference:
    dtype = "torch.float32"
    device = _Tensor.device


class _Scalar:
    ndim = 1
    shape = (1,)
    device = _Tensor.device

    def __init__(self) -> None:
        self.to_calls: list[dict[str, object]] = []

    def detach(self) -> "_Scalar":
        return self

    def to(self, **kwargs: object) -> "_Scalar":
        self.to_calls.append(kwargs)
        return self

    def item(self) -> float:
        return 0.25


class _InferenceMode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class GPUBackendTest(unittest.TestCase):
    def test_only_final_distance_scalar_moves_to_cpu(self) -> None:
        torch = types.SimpleNamespace(
            float32="torch.float32",
            inference_mode=lambda: _InferenceMode(),
            log_softmax=lambda tensor, dim: tensor,
        )
        scalar = _Scalar()
        audit = types.SimpleNamespace(
            to_dict=lambda: {"full_tensor_host_transfers": 0}
        )
        kernel = mock.Mock(
            return_value=types.SimpleNamespace(
                per_example_mean_kl=scalar,
                audit=audit,
            )
        )
        backend = GPUFullVocabularyKLBackend(torch_module=torch, kl_kernel=kernel)
        logits = _Tensor()
        reference = backend.prepare_reference(logits)
        measurement = backend.measure(reference, logits)
        self.assertEqual(measurement.value, 0.25)
        self.assertEqual(logits.to_calls, [{"dtype": "torch.float32"}])
        self.assertEqual(scalar.to_calls, [{"device": "cpu"}])


if __name__ == "__main__":
    unittest.main()
